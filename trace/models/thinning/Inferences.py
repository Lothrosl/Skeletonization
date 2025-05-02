import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Import the UNet model from our training script
# Make sure this is in the same directory or adjust the import path
from U_net import UNet

def process_single_image(model, image_path, output_path, device):
    """Process a single image and save the result"""
    # Load image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
    
    # Convert to binary
    output_np = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    
    # Save result
    output_img = Image.fromarray(output_np)
    output_img.save(output_path)
    
    return output_np

def process_directory(model, input_dir, output_dir, device):
    """Process all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all input images
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png') and f.startswith('image_')]
    
    for image_file in tqdm(image_files):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file.replace('image_', 'predicted_'))
        
        process_single_image(model, input_path, output_path, device)

def visualize_results(input_path, target_path, output_path, save_path=None):
    """Visualize input, target, and prediction side by side"""
    # Load images
    input_img = np.array(Image.open(input_path).convert('L'))
    target_img = np.array(Image.open(target_path).convert('L'))
    output_img = np.array(Image.open(output_path).convert('L'))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(target_img, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(output_img, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def evaluate_model(model, data_dir, output_dir, device, num_samples=5):
    """Evaluate model and visualize some examples"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    process_directory(model, data_dir, output_dir, device)
    
    # Get some random samples to visualize
    image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('image_') and f.endswith('.png')])
    target_files = sorted([f for f in os.listdir(data_dir) if f.startswith('target_') and f.endswith('.png')])
    output_files = sorted([f for f in os.listdir(output_dir) if f.startswith('predicted_') and f.endswith('.png')])
    
    # Ensure we have matching files
    assert len(image_files) == len(target_files) == len(output_files), "Number of files doesn't match"
    
    # Select random samples
    indices = np.random.choice(len(image_files), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        input_path = os.path.join(data_dir, image_files[idx])
        target_path = os.path.join(data_dir, target_files[idx])
        output_path = os.path.join(output_dir, output_files[idx])
        
        visualize_results(input_path, target_path, output_path, 
                         save_path=os.path.join(output_dir, f'comparison_{i}.png'))

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # Set directories
    data_dir = 'data/thinning'
    output_dir = 'results'
    
    # Evaluate
    evaluate_model(model, data_dir, output_dir, device, num_samples=10)
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()