import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class RoadThinningDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), transform=None, split='train', test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('image_') and f.endswith('.png')])
        self.target_files = sorted([f for f in os.listdir(data_dir) if f.startswith('target_') and f.endswith('.png')])
        
        # Make sure we have matching pairs
        assert len(self.image_files) == len(self.target_files), "Number of images and targets don't match!"
        
        # Split into train and validation sets
        image_train, image_val, target_train, target_val = train_test_split(
            self.image_files, self.target_files, test_size=test_size, random_state=random_state
        )
        
        if split == 'train':
            self.image_files = image_train
            self.target_files = target_train
        else:  # 'val'
            self.image_files = image_val
            self.target_files = target_val
        
        print(f"Loaded {len(self.image_files)} {split} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and target
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        target_path = os.path.join(self.data_dir, self.target_files[idx])
        
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        target = Image.open(target_path).convert('L')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
        else:
            # Default transformation
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
            target = transform(target)
        
        # Normalize target to binary (0 or 1)
        target = (target > 0.5).float()
        
        return image, target


# U-Net components
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # Use transpose conv if bilinear is False
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ensure x2 and x1 have same dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


# U-Net Model
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer
        logits = self.outc(x)
        return torch.sigmoid(logits)


# Loss function - Binary Cross Entropy with Dice Loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight, reduction='mean')
        
    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        smooth = 1e-5
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        
        # Combined loss
        return bce_loss + dice_loss


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward + optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(train_loader.dataset)


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
    
    return running_loss / len(val_loader.dataset)


# Visualization function
def visualize_predictions(model, val_dataset, device, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get a random sample
            idx = np.random.randint(0, len(val_dataset))
            image, target = val_dataset[idx]
            
            # Make prediction
            image_tensor = image.unsqueeze(0).to(device)
            prediction = model(image_tensor)
            
            # Convert to numpy for visualization
            image_np = image.squeeze().numpy()
            target_np = target.squeeze().numpy()
            prediction_np = prediction.squeeze().cpu().numpy()
            
            # Plot
            axs[i, 0].imshow(image_np, cmap='gray')
            axs[i, 0].set_title('Input Image')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(target_np, cmap='gray')
            axs[i, 1].set_title('Ground Truth')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(prediction_np, cmap='gray')
            axs[i, 2].set_title('Prediction')
            axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()


# Main function
def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    data_dir = 'data/thinning'  # Change this to your data directory
    
    # Create datasets and data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = RoadThinningDataset(data_dir, transform=transform, split='train')
    val_dataset = RoadThinningDataset(data_dir, transform=transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    # Loss function and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train and validate
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")
        
        # Visualize predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_predictions(model, val_dataset, device)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curves.png')
    plt.close()
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    final_val_loss = validate(model, val_loader, criterion, device)
    print(f'Final Validation Loss: {final_val_loss:.4f}')
    
    # Visualize final predictions
    visualize_predictions(model, val_dataset, device, num_samples=10)


if __name__ == '__main__':
    main()