import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk
import networkx as nx
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Import the UNet model from our training script
from Ablation import UNet, DiceBCELoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(model_path):
    """Load a trained model"""
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def process_image(model, image_path):
    """Process a single image and return the prediction"""
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
    output_np = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    
    return output_np


def calculate_mse_with_distance_transform(prediction, target):
    """Calculate MSE using distance transform from the ground truth"""
    # Ensure binary images
    target_binary = target.astype(bool)
    prediction_binary = prediction.astype(bool)
    
    # Compute distance transform from ground truth
    dist_transform = distance_transform_edt(~target_binary)
    
    # Calculate MSE based on distance transform values at predicted pixels
    mse = np.mean(dist_transform[prediction_binary]**2)
    
    return mse


def find_nodes(skeleton):
    """Find nodes (junctions) in a skeleton and classify by valence"""
    # Make sure skeleton is binary
    skeleton = skeleton.astype(bool)
    
    # Create a padded version to handle border pixels
    padded = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0)
    
    # Nodes dictionary to store coordinates by valence
    nodes = {1: [], 2: [], 3: [], 4: []}
    
    # Check each pixel in the skeleton
    for y in range(1, padded.shape[0]-1):
        for x in range(1, padded.shape[1]-1):
            if padded[y, x]:
                # Get 3x3 neighborhood
                neighbors = padded[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0  # Remove center pixel
                
                # Count connected neighbors
                count = np.sum(neighbors)
                
                # Classify node by valence (number of connections)
                if count in [1, 2, 3, 4]:
                    # Store coordinates (adjusting for padding)
                    nodes[count].append((y-1, x-1))
    
    return nodes


def match_nodes(pred_nodes, gt_nodes, distance_threshold=3):
    """Match nodes between prediction and ground truth"""
    results = {}
    
    for valence in [1, 2, 3, 4]:
        # Get nodes for current valence
        pred_nodes_val = pred_nodes.get(valence, [])
        gt_nodes_val = gt_nodes.get(valence, [])
        
        if not pred_nodes_val or not gt_nodes_val:
            # Handle empty sets
            precision = 1.0 if not pred_nodes_val else 0.0
            recall = 1.0 if not gt_nodes_val else 0.0
            results[valence] = {
                'precision': precision,
                'recall': recall,
                'f1': 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            }
            continue
        
        # Create distance matrix
        cost_matrix = np.zeros((len(pred_nodes_val), len(gt_nodes_val)))
        
        for i, pred_node in enumerate(pred_nodes_val):
            for j, gt_node in enumerate(gt_nodes_val):
                # Calculate Euclidean distance
                cost_matrix[i, j] = np.sqrt((pred_node[0] - gt_node[0])**2 + (pred_node[1] - gt_node[1])**2)
        
        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Count matches (distance < threshold)
        matches = sum(cost_matrix[row_ind[i], col_ind[i]] < distance_threshold for i in range(len(row_ind)))
        
        # Calculate precision and recall
        precision = matches / len(pred_nodes_val) if pred_nodes_val else 0
        recall = matches / len(gt_nodes_val) if gt_nodes_val else 0
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        
        results[valence] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return results


def calculate_segmentation_metrics(prediction, target):
    """Calculate common segmentation metrics: IoU and Dice coefficient"""
    # Flatten arrays
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    
    # Calculate metrics
    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (pred_flat.sum() + target_flat.sum()) if (pred_flat.sum() + target_flat.sum()) > 0 else 0
    
    return {
        'iou': iou,
        'dice': dice
    }


def evaluate_sample(model, image_path, target_path):
    """Evaluate a single sample"""
    # Load target
    target = np.array(Image.open(target_path).convert('L')) > 0
    
    # Process image
    prediction = process_image(model, image_path)
    
    # Calculate MSE with distance transform
    mse = calculate_mse_with_distance_transform(prediction, target)
    
    # Find nodes in both prediction and ground truth
    pred_nodes = find_nodes(prediction)
    gt_nodes = find_nodes(target)
    
    # Match nodes and get metrics
    node_metrics = match_nodes(pred_nodes, gt_nodes)
    
    # Calculate segmentation metrics
    seg_metrics = calculate_segmentation_metrics(prediction, target)
    
    return {
        'mse': mse,
        'node_metrics': node_metrics,
        'segmentation_metrics': seg_metrics
    }


def visualize_nodes(image, nodes, save_path=None):
    """Visualize nodes on an image, color-coded by valence"""
    # Convert to RGB for colored visualization
    rgb_image = np.stack([image, image, image], axis=-1) * 255
    
    # Define colors for different valences (B, G, R format)
    colors = {
        1: (0, 0, 255),  # Red for endpoints (valence 1)
        2: (0, 255, 0),  # Green for continuing segments (valence 2)
        3: (255, 0, 0),  # Blue for T-junctions (valence 3)
        4: (255, 255, 0)  # Cyan for crossings (valence 4)
    }
    
    # Place colored dots at node positions
    for valence, node_list in nodes.items():
        for node in node_list:
            y, x = node
            if 0 <= y < rgb_image.shape[0] and 0 <= x < rgb_image.shape[1]:
                # Draw a small circle around each node
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rgb_image.shape[0] and 0 <= nx < rgb_image.shape[1]:
                            # Only color pixels in a rough circle
                            if dx**2 + dy**2 <= 4:
                                rgb_image[ny, nx] = colors[valence]
    
    # Save or display
    if save_path:
        Image.fromarray(rgb_image.astype(np.uint8)).save(save_path)
    
    return rgb_image


def visualize_evaluation(image_path, target_path, prediction, metrics, output_dir, index):
    """Create visualization with input, ground truth, prediction, and node detection"""
    # Load images
    input_img = np.array(Image.open(image_path).convert('L'))
    target_img = np.array(Image.open(target_path).convert('L')) > 0
    
    # Find nodes
    pred_nodes = find_nodes(prediction)
    gt_nodes = find_nodes(target_img)
    
    # Create node visualizations
    target_with_nodes = visualize_nodes(target_img, gt_nodes)
    pred_with_nodes = visualize_nodes(prediction, pred_nodes)
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot images
    axs[0, 0].imshow(input_img, cmap='gray')
    axs[0, 0].set_title('Input Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(target_img, cmap='gray')
    axs[0, 1].set_title('Ground Truth')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(prediction, cmap='gray')
    axs[1, 0].set_title('Prediction')
    axs[1, 0].axis('off')
    
    # Plot metrics as text in the fourth subplot
    axs[1, 1].axis('off')
    axs[1, 1].text(0.1, 0.9, f"MSE: {metrics['mse']:.4f}", transform=axs[1, 1].transAxes, fontsize=9)
    axs[1, 1].text(0.1, 0.85, f"IoU: {metrics['segmentation_metrics']['iou']:.4f}", transform=axs[1, 1].transAxes, fontsize=9)
    axs[1, 1].text(0.1, 0.8, f"Dice: {metrics['segmentation_metrics']['dice']:.4f}", transform=axs[1, 1].transAxes, fontsize=9)
    
    # Add node metrics
    y_pos = 0.7
    axs[1, 1].text(0.1, 0.75, "Node Metrics:", transform=axs[1, 1].transAxes, fontsize=9, weight='bold')
    
    for valence in [1, 2, 3, 4]:
        metrics_val = metrics['node_metrics'][valence]
        axs[1, 1].text(0.1, y_pos, f"Valence {valence} - P: {metrics_val['precision']:.2f}, R: {metrics_val['recall']:.2f}, F1: {metrics_val['f1']:.2f}", 
                 transform=axs[1, 1].transAxes, fontsize=9)
        y_pos -= 0.05
    
    # Create a mini figure to show node colors
    axs[1, 1].text(0.1, 0.5, "Node Types:", transform=axs[1, 1].transAxes, fontsize=9, weight='bold')
    valences = [1, 2, 3, 4]
    colors = ['red', 'green', 'blue', 'cyan']
    descriptions = ['Endpoints', 'Continuations', 'T-junctions', 'Crossings']
    
    for i, (valence, color, desc) in enumerate(zip(valences, colors, descriptions)):
        y = 0.45 - i * 0.05
        axs[1, 1].plot(0.15, y, 'o', color=color, transform=axs[1, 1].transAxes, markersize=5)
        axs[1, 1].text(0.2, y - 0.01, f"Valence {valence}: {desc}", transform=axs[1, 1].transAxes, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'evaluation_{index}.png'), dpi=150)
    plt.close()
    
    # Also save node visualizations
    Image.fromarray(target_with_nodes.astype(np.uint8)).save(os.path.join(output_dir, f'gt_nodes_{index}.png'))
    Image.fromarray(pred_with_nodes.astype(np.uint8)).save(os.path.join(output_dir, f'pred_nodes_{index}.png'))


def calculate_test_loss(model, data_dir, criterion):
    """Calculate test loss on the dataset"""
    # Get all test image and target paths
    image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('image_') and f.endswith('.png')])
    target_files = sorted([f for f in os.listdir(data_dir) if f.startswith('target_') and f.endswith('.png')])
    
    # Ensure we have matching files
    assert len(image_files) == len(target_files), "Number of files doesn't match"
    
    total_loss = 0.0
    transform = transforms.Compose([transforms.ToTensor()])
    
    for img_file, tgt_file in tqdm(zip(image_files, target_files), total=len(image_files), desc="Calculating test loss"):
        img_path = os.path.join(data_dir, img_file)
        tgt_path = os.path.join(data_dir, tgt_file)
        
        # Load image and target
        image = Image.open(img_path).convert('L')
        target = Image.open(tgt_path).convert('L')
        
        # Transform
        image_tensor = transform(image).unsqueeze(0).to(device)
        target_tensor = (transform(target) > 0.5).float().unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            loss = criterion(output, target_tensor).item()
            
        total_loss += loss
    
    return total_loss / len(image_files)


def evaluate_dataset(model, data_dir, output_dir, num_samples=10):
    """Evaluate model on the dataset and visualize results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image and target paths
    image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('image_') and f.endswith('.png')])
    target_files = sorted([f for f in os.listdir(data_dir) if f.startswith('target_') and f.endswith('.png')])
    
    # Ensure we have matching files
    assert len(image_files) == len(target_files), "Number of files doesn't match"
    
    # Calculate test loss
    criterion = DiceBCELoss()
    test_loss = calculate_test_loss(model, data_dir, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Random sample selection
    if num_samples < len(image_files):
        indices = np.random.choice(len(image_files), num_samples, replace=False)
    else:
        indices = range(len(image_files))
    
    # Metrics storage
    all_metrics = []
    
    # Process selected samples
    for i, idx in enumerate(tqdm(indices, desc="Evaluating samples")):
        img_path = os.path.join(data_dir, image_files[idx])
        tgt_path = os.path.join(data_dir, target_files[idx])
        
        # Process and evaluate
        prediction = process_image(model, img_path)
        metrics = evaluate_sample(model, img_path, tgt_path)
        
        # Store metrics
        metrics_row = {
            'sample_id': idx,
            'mse': metrics['mse'],
            'iou': metrics['segmentation_metrics']['iou'],
            'dice': metrics['segmentation_metrics']['dice']
        }
        
        # Add node metrics
        for valence in [1, 2, 3, 4]:
            metrics_row[f'precision_v{valence}'] = metrics['node_metrics'][valence]['precision']
            metrics_row[f'recall_v{valence}'] = metrics['node_metrics'][valence]['recall']
            metrics_row[f'f1_v{valence}'] = metrics['node_metrics'][valence]['f1']
        
        all_metrics.append(metrics_row)
        
        # Visualize
        visualize_evaluation(img_path, tgt_path, prediction, metrics, output_dir, i)
    
    # Create a DataFrame with metrics
    df_metrics = pd.DataFrame(all_metrics)
    
    # Calculate average metrics
    avg_metrics = df_metrics.drop('sample_id', axis=1).mean().to_dict()
    
    # Save metrics to CSV
    df_metrics.to_csv(os.path.join(output_dir, 'sample_metrics.csv'), index=False)
    
    # Create summary report
    create_summary_report(avg_metrics, test_loss, output_dir)
    
    return avg_metrics


def create_summary_report(avg_metrics, test_loss, output_dir):
    """Create a summary report with all metrics"""
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Road Skeleton Extraction - Evaluation Summary\n")
        f.write("=========================================\n\n")
        
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        
        f.write("Distance-based Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {avg_metrics['mse']:.4f}\n\n")
        
        f.write("Segmentation Metrics:\n")
        f.write(f"Intersection over Union (IoU): {avg_metrics['iou']:.4f}\n")
        f.write(f"Dice Coefficient: {avg_metrics['dice']:.4f}\n\n")
        
        f.write("Node Detection Metrics:\n")
        for valence in [1, 2, 3, 4]:
            f.write(f"Valence {valence} (")
            if valence == 1:
                f.write("endpoints")
            elif valence == 2:
                f.write("continuations")
            elif valence == 3:
                f.write("T-junctions")
            else:
                f.write("crossings")
            f.write(f"):\n")
            f.write(f"  Precision: {avg_metrics[f'precision_v{valence}']:.4f}\n")
            f.write(f"  Recall: {avg_metrics[f'recall_v{valence}']:.4f}\n")
            f.write(f"  F1 Score: {avg_metrics[f'f1_v{valence}']:.4f}\n\n")
    
    # Also create a visual summary with matplotlib
    create_visual_summary(avg_metrics, test_loss, output_dir)


def create_visual_summary(avg_metrics, test_loss, output_dir):
    """Create visual summary of metrics"""
    # Node metrics visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Bar chart for node metrics
    valences = [1, 2, 3, 4]
    x = np.arange(len(valences))
    width = 0.25
    
    precision_vals = [avg_metrics[f'precision_v{v}'] for v in valences]
    recall_vals = [avg_metrics[f'recall_v{v}'] for v in valences]
    f1_vals = [avg_metrics[f'f1_v{v}'] for v in valences]
    
    ax1.bar(x - width, precision_vals, width, label='Precision')
    ax1.bar(x, recall_vals, width, label='Recall')
    ax1.bar(x + width, f1_vals, width, label='F1')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Node Detection Metrics by Valence')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Endpoints', 'Continuations', 'T-junctions', 'Crossings'])
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Pie chart for valence distribution
    labels = ['Endpoints', 'Continuations', 'T-junctions', 'Crossings']
    sizes = [avg_metrics[f'precision_v{v}'] * avg_metrics[f'recall_v{v}'] for v in valences]
    if sum(sizes) == 0:
        sizes = [1, 1, 1, 1]  # Avoid division by zero
    sizes = [s/sum(sizes) for s in sizes]
    
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Distribution of Node Types')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'node_metrics_summary.png'))
    plt.close()
    
    # Overall metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Test Loss', 'MSE', 'IoU', 'Dice']
    values = [test_loss, avg_metrics['mse'], avg_metrics['iou'], avg_metrics['dice']]
    
    # Normalize MSE for better visualization (it can be much larger than other metrics)
    if values[1] > 1:
        metrics[1] = f'MSE (÷{10**int(np.log10(values[1]))})'
        values[1] = values[1] / (10**int(np.log10(values[1])))
    
    bars = ax.bar(metrics, values, color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'])
    
    # Add labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title('Overall Model Performance Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics_summary.png'))
    plt.close()


def main():
    """Main function"""
    # Parameters
    model_path = 'ablation_model7.pth'
    data_dir = 'data/thinning'
    output_dir = 'evaluation_results_7'
    num_samples = 20  # Number of random samples to evaluate
    
    # Load model
    model = load_model(model_path)
    
    # Evaluate
    print("Evaluating model...")
    avg_metrics = evaluate_dataset(model, data_dir, output_dir, num_samples)
    
    print("\nEvaluation Complete. Summary of average metrics:")
    print(f"MSE: {avg_metrics['mse']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    print(f"Dice: {avg_metrics['dice']:.4f}")
    
    print("\nNode Detection Metrics:")
    for valence in [1, 2, 3, 4]:
        print(f"Valence {valence}:")
        print(f"  Precision: {avg_metrics[f'precision_v{valence}']:.4f}")
        print(f"  Recall: {avg_metrics[f'recall_v{valence}']:.4f}")
        print(f"  F1: {avg_metrics[f'f1_v{valence}']:.4f}")
    
    print(f"\nDetailed results saved to {output_dir}")


if __name__ == '__main__':
    main()