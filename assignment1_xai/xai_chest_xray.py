"""
Assignment 1: Explainable AI (XAI) for Chest X-Ray Pneumonia Classification
This script implements Grad-CAM and LIME visualization techniques on a CNN model
trained on the Kaggle Chest X-Ray Pneumonia dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ChestXRayDataset(Dataset):
    """Custom dataset for Chest X-Ray images"""
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Directory with all the images (should contain NORMAL and PNEUMONIA folders)
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from NORMAL folder (label 0)
        normal_dir = os.path.join(root_dir, 'NORMAL')
        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(normal_dir, img_name))
                    self.labels.append(0)
        
        # Load images from PNEUMONIA folder (label 1)
        pneumonia_dir = os.path.join(root_dir, 'PNEUMONIA')
        if os.path.exists(pneumonia_dir):
            for img_name in os.listdir(pneumonia_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(pneumonia_dir, img_name))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ChestXRayCNN(nn.Module):
    """CNN model for chest X-ray classification"""
    
    def __init__(self, num_classes: int = 2):
        super(ChestXRayCNN, self).__init__()
        
        # Use ResNet18 as backbone (pre-trained on ImageNet)
        self.backbone = models.resnet18(pretrained=True)
        
        # Modify final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Store feature maps for Grad-CAM
        self.gradients = None
        self.feature_maps = None
    
    def activations_hook(self, grad):
        """Hook to capture gradients"""
        self.gradients = grad
    
    def forward(self, x):
        # Get features from the last convolutional layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Register hook on the feature maps
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        self.feature_maps = x
        
        # Continue with rest of the network
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        
        return x
    
    def get_activations_gradient(self):
        """Return the gradients"""
        return self.gradients
    
    def get_activations(self):
        """Return the feature maps"""
        return self.feature_maps


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class for visualization (None for predicted class)
        
        Returns:
            cam: Grad-CAM heatmap
        """
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and feature maps
        gradients = self.model.get_activations_gradient()
        feature_maps = self.model.get_activations()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)  # Apply ReLU
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self, original_image: np.ndarray, cam: np.ndarray, 
                  alpha: float = 0.5) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image
        
        Args:
            original_image: Original image (H, W, C)
            cam: Grad-CAM heatmap
            alpha: Transparency factor
        
        Returns:
            Visualization with heatmap overlay
        """
        # Resize CAM to match original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize original image to [0, 255]
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        superimposed = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
        
        return superimposed


class LIMEExplainer:
    """LIME implementation for image classification"""
    
    def __init__(self, model: nn.Module, transform):
        self.model = model
        self.transform = transform
        self.model.eval()
    
    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME
        
        Args:
            images: Batch of images (N, H, W, C)
        
        Returns:
            Predictions (N, num_classes)
        """
        self.model.eval()
        
        # Convert to tensors
        batch = []
        for img in images:
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_tensor = self.transform(img_pil)
            batch.append(img_tensor)
        
        batch = torch.stack(batch)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def explain(self, image: np.ndarray, num_samples: int = 1000, 
                num_features: int = 10) -> Tuple:
        """
        Generate LIME explanation
        
        Args:
            image: Original image (H, W, C)
            num_samples: Number of perturbed samples
            num_features: Number of super-pixels to show
        
        Returns:
            explanation: LIME explanation object
        """
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries
            
            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Generate explanation
            explanation = explainer.explain_instance(
                image,
                self.predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples
            )
            
            return explanation
        
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return None
    
    def visualize(self, image: np.ndarray, explanation, label: int, 
                  num_features: int = 10) -> np.ndarray:
        """
        Visualize LIME explanation
        
        Args:
            image: Original image
            explanation: LIME explanation object
            label: Target label
            num_features: Number of features to highlight
        
        Returns:
            Visualization with highlighted regions
        """
        from skimage.segmentation import mark_boundaries
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        # Create visualization
        img_boundry = mark_boundaries(temp / 255.0, mask)
        
        return (img_boundry * 255).astype(np.uint8)


def train_model(model: nn.Module, train_loader: DataLoader, 
                num_epochs: int = 5, device: str = 'cuda') -> nn.Module:
    """
    Train the CNN model
    
    Args:
        model: CNN model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] completed: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    return model


def visualize_xai_comparison(original_image: np.ndarray, gradcam_viz: np.ndarray,
                              lime_viz: np.ndarray, prediction: str, 
                              save_path: str = None):
    """
    Create side-by-side visualization of Grad-CAM and LIME
    
    Args:
        original_image: Original image
        gradcam_viz: Grad-CAM visualization
        lime_viz: LIME visualization
        prediction: Model prediction
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gradcam_viz)
    axes[1].set_title(f'Grad-CAM\nPrediction: {prediction}')
    axes[1].axis('off')
    
    axes[2].imshow(lime_viz)
    axes[2].set_title(f'LIME\nPrediction: {prediction}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def main():
    """Main function to demonstrate XAI techniques"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create output directory
    os.makedirs('outputs/xai_visualizations', exist_ok=True)
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = ChestXRayCNN(num_classes=2)
    
    # Note: For demonstration, we'll use a pre-trained model
    # In practice, you would train on the Kaggle Chest X-Ray dataset
    # Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    
    # Example: Load dataset (uncomment when dataset is available)
    # train_dataset = ChestXRayDataset('path/to/train', transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # model = train_model(model, train_loader, num_epochs=5, device=device)
    # torch.save(model.state_dict(), 'chest_xray_cnn.pth')
    
    # For demo purposes, we'll use the model without training
    # In production, load trained weights: model.load_state_dict(torch.load('chest_xray_cnn.pth'))
    model = model.to(device)
    model.eval()
    
    # Create a sample image for demonstration
    print("\n=== Creating Sample Image ===")
    sample_image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    
    # Save sample image
    Image.fromarray(sample_image).save('outputs/xai_visualizations/sample_xray.png')
    
    # Prepare image for model
    img_pil = Image.fromarray(sample_image)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)
    
    class_names = ['NORMAL', 'PNEUMONIA']
    pred_class = class_names[prediction]
    pred_prob = probs[0, prediction].item()
    
    print(f"\nPrediction: {pred_class} (confidence: {pred_prob:.2%})")
    
    # Generate Grad-CAM
    print("\n=== Generating Grad-CAM ===")
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(img_tensor, target_class=prediction)
    gradcam_viz = gradcam.visualize(sample_image, cam)
    
    # Generate LIME explanation
    print("\n=== Generating LIME Explanation ===")
    lime_explainer = LIMEExplainer(model, transform)
    explanation = lime_explainer.explain(sample_image, num_samples=100)
    
    if explanation:
        lime_viz = lime_explainer.visualize(sample_image, explanation, prediction)
    else:
        lime_viz = sample_image
    
    # Create comparison visualization
    print("\n=== Creating Visualization ===")
    visualize_xai_comparison(
        sample_image,
        gradcam_viz,
        lime_viz,
        f"{pred_class} ({pred_prob:.2%})",
        save_path='outputs/xai_visualizations/xai_comparison.png'
    )
    
    print("\n=== XAI Analysis Complete ===")
    print("Outputs saved to: outputs/xai_visualizations/")
    print("\nHow to use with real dataset:")
    print("1. Download Kaggle Chest X-Ray Pneumonia dataset")
    print("2. Extract to a directory with NORMAL and PNEUMONIA subfolders")
    print("3. Uncomment the training code in main()")
    print("4. Run the script to train and visualize")


if __name__ == "__main__":
    main()
