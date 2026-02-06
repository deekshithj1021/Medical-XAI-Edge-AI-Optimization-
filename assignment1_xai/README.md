# Assignment 1: Explainable AI for Chest X-Ray Classification

This assignment implements Grad-CAM and LIME explainability techniques for a CNN trained on chest X-ray images to classify pneumonia cases.

## Overview

The code demonstrates two popular XAI (Explainable AI) techniques:
1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Visualizes which regions of an X-ray image are important for the model's prediction
2. **LIME (Local Interpretable Model-agnostic Explanations)**: Identifies super-pixels that contribute to the classification decision

## Dataset

**Kaggle Chest X-Ray Pneumonia Dataset**
- Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Classes: NORMAL, PNEUMONIA
- Contains chest X-ray images from pediatric patients

## Features

### 1. CNN Model (`ChestXRayCNN`)
- Based on ResNet18 architecture
- Pre-trained on ImageNet and fine-tuned for binary classification
- Includes hooks for Grad-CAM visualization

### 2. Grad-CAM Implementation
- Captures gradients from the final convolutional layer
- Generates class activation maps highlighting important regions
- Overlays heatmaps on original images for visualization

### 3. LIME Implementation
- Uses super-pixel segmentation
- Generates perturbations to identify important image regions
- Provides interpretable explanations for predictions

### 4. Visualization
- Side-by-side comparison of original image, Grad-CAM, and LIME
- High-quality heatmap overlays
- Saves visualizations to disk

## Installation

```bash
# Install required packages
pip install -r ../requirements.txt
```

## Usage

### Basic Usage

```bash
python xai_chest_xray.py
```

This will:
1. Initialize a MobileNet model
2. Generate sample visualizations (demo mode)
3. Save outputs to `outputs/xai_visualizations/`

### Using with Real Dataset

1. **Download the dataset:**
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip
   ```

2. **Modify the code:**
   - Uncomment the training section in `main()`
   - Update the dataset path to point to your extracted dataset
   - The dataset should have structure:
     ```
     chest_xray/
     ├── train/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     └── test/
         ├── NORMAL/
         └── PNEUMONIA/
     ```

3. **Train and visualize:**
   ```python
   # In xai_chest_xray.py, uncomment:
   train_dataset = ChestXRayDataset('chest_xray/train', transform=transform)
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   model = train_model(model, train_loader, num_epochs=5, device=device)
   torch.save(model.state_dict(), 'chest_xray_cnn.pth')
   ```

## Code Structure

```
xai_chest_xray.py
├── ChestXRayDataset       # Custom dataset loader
├── ChestXRayCNN           # CNN model with Grad-CAM hooks
├── GradCAM                # Grad-CAM implementation
├── LIMEExplainer          # LIME implementation
├── train_model()          # Model training function
├── visualize_xai_comparison()  # Visualization function
└── main()                 # Main execution
```

## Key Functions

### Training
```python
model = train_model(model, train_loader, num_epochs=5, device='cuda')
```

### Grad-CAM
```python
gradcam = GradCAM(model)
cam = gradcam.generate_cam(img_tensor, target_class=1)
visualization = gradcam.visualize(original_image, cam)
```

### LIME
```python
lime_explainer = LIMEExplainer(model, transform)
explanation = lime_explainer.explain(image, num_samples=1000)
visualization = lime_explainer.visualize(image, explanation, label=1)
```

## Output

The script generates:
- `sample_xray.png`: Sample chest X-ray image
- `xai_comparison.png`: Side-by-side comparison of Grad-CAM and LIME

## Understanding the Visualizations

### Grad-CAM Heatmaps
- **Red regions**: Most important for the prediction
- **Blue regions**: Least important
- The heatmap shows where the model "looks" to make its decision

### LIME Explanations
- **Highlighted regions**: Super-pixels that positively contribute to the prediction
- **Boundaries**: Show the segmentation used by LIME
- Helps understand which parts of the image support the classification

## Technical Details

### Model Architecture
- Backbone: ResNet18
- Input size: 224×224×3
- Output: 2 classes (NORMAL, PNEUMONIA)
- Pre-trained on ImageNet

### Grad-CAM Process
1. Forward pass through the model
2. Backward pass to compute gradients
3. Global average pooling of gradients
4. Weighted combination of feature maps
5. ReLU activation and normalization

### LIME Process
1. Segment image into super-pixels
2. Generate perturbed versions by hiding super-pixels
3. Get model predictions for perturbed images
4. Train interpretable linear model
5. Extract important super-pixels

## Performance Considerations

- **Grad-CAM**: Fast, can be computed in real-time
- **LIME**: Slower, requires multiple model evaluations (controlled by `num_samples`)
- Reduce `num_samples` in LIME for faster results with slightly less accuracy

## Customization

### Change number of LIME samples
```python
explanation = lime_explainer.explain(image, num_samples=500)  # Faster
```

### Adjust Grad-CAM overlay transparency
```python
visualization = gradcam.visualize(image, cam, alpha=0.3)  # More transparent
```

### Use different CNN backbone
```python
self.backbone = models.resnet50(pretrained=True)  # Use ResNet50 instead
```

## References

1. **Grad-CAM**: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
2. **LIME**: Ribeiro et al. (2016) "Why Should I Trust You?: Explaining the Predictions of Any Classifier"
3. **Dataset**: Kermany et al. (2018) "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification"

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or image resolution:
```python
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Smaller size
    ...
])
```

### LIME Not Installed
```bash
pip install lime
```

### Slow Training
Use GPU acceleration:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## License

This code is provided for educational purposes.
