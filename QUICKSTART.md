# Quick Start Guide

This guide will help you get started with both assignments quickly.

## Prerequisites

```bash
# Install Python 3.8+
python --version

# Clone the repository
git clone https://github.com/deekshithj1021/Medical-XAI-Edge-AI-Optimization-.git
cd Medical-XAI-Edge-AI-Optimization-

# Install dependencies
pip install -r requirements.txt
```

## Assignment 1: XAI for Chest X-Ray Classification

### Quick Demo (No dataset required)

```bash
cd assignment1_xai
python xai_chest_xray.py
```

This runs a demo with synthetic data. Outputs will be saved to `outputs/xai_visualizations/`.

### With Real Dataset

1. **Download the Kaggle Chest X-Ray Pneumonia dataset:**
   ```bash
   # Option 1: Using Kaggle CLI
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip
   
   # Option 2: Manual download
   # Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   # Download and extract to get this structure:
   # chest_xray/
   # ├── train/
   # │   ├── NORMAL/
   # │   └── PNEUMONIA/
   # └── test/
   #     ├── NORMAL/
   #     └── PNEUMONIA/
   ```

2. **Modify the code to use the dataset:**
   
   In `xai_chest_xray.py`, find and uncomment these lines in the `main()` function:
   ```python
   # Uncomment these lines:
   train_dataset = ChestXRayDataset('chest_xray/train', transform=transform)
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   model = train_model(model, train_loader, num_epochs=5, device=device)
   torch.save(model.state_dict(), 'chest_xray_cnn.pth')
   ```

3. **Run the script:**
   ```bash
   python xai_chest_xray.py
   ```

### Expected Outputs

- `sample_xray.png`: Input chest X-ray image
- `xai_comparison.png`: Side-by-side comparison of Grad-CAM and LIME visualizations

### Understanding the Results

- **Grad-CAM (middle)**: Red/yellow regions show where the model focuses its attention
- **LIME (right)**: Yellow highlighted areas are important for the prediction

## Assignment 2: Pruning Comparison on MobileNet

### Quick Demo (Uses synthetic data)

```bash
cd assignment2_pruning
python pruning_demo.py
```

This runs a fast demo with synthetic data. Perfect for testing the code quickly.

### Full Experiment (Uses CIFAR-10)

```bash
cd assignment2_pruning
python pruning_comparison.py
```

**Note**: This will download CIFAR-10 dataset (~170MB) on first run and requires internet access.

### Expected Outputs

- `pruning_comparison.png`: 4-panel comparison plot showing:
  1. Accuracy vs Sparsity
  2. Model Size vs Sparsity
  3. Accuracy Drop vs Sparsity
  4. Compression Ratio vs Sparsity

### Understanding the Results

- **One-Shot Pruning** (blue line): Fast but may lose more accuracy
- **Iterative Pruning** (orange line): Slower but better accuracy preservation
- Look for the "sweet spot" where compression is high but accuracy loss is acceptable

## Customization Tips

### Assignment 1: XAI

**Change the number of LIME samples** (faster but less accurate):
```python
explanation = lime_explainer.explain(sample_image, num_samples=500)  # Default: 1000
```

**Use a different CNN backbone:**
```python
self.backbone = models.resnet50(pretrained=True)  # Instead of ResNet18
```

**Adjust Grad-CAM heatmap transparency:**
```python
gradcam_viz = gradcam.visualize(sample_image, cam, alpha=0.3)  # More transparent
```

### Assignment 2: Pruning

**Try different pruning amounts:**
```python
pruning_amounts = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]  # More granular
```

**Adjust iterative pruning iterations:**
```python
iterative.prune(
    target_sparsity=0.9,
    num_iterations=10,  # More gradual (default: 5)
    fine_tune_epochs=2
)
```

**Use your own dataset:**
```python
# Replace CIFAR-10 with your dataset
from torchvision import datasets
train_dataset = datasets.ImageFolder('path/to/train', transform=transform)
test_dataset = datasets.ImageFolder('path/to/test', transform=transform_test)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```python
train_loader = DataLoader(train_dataset, batch_size=16, ...)  # Default: 32
```

### Slow Training

Use GPU if available:
```python
# The scripts automatically use GPU if available
# To force CPU:
device = torch.device('cpu')
```

### LIME Not Working

Install LIME:
```bash
pip install lime
```

### Network Issues (Can't download datasets)

For **Assignment 1**: Use the demo mode (synthetic data) or manually download the dataset

For **Assignment 2**: Use `pruning_demo.py` which doesn't require downloads

## Performance Tips

### For Faster Experiments

1. **Use smaller subsets:**
   ```python
   # In pruning_comparison.py
   train_subset_indices = np.random.choice(len(train_dataset), 1000, replace=False)
   ```

2. **Reduce fine-tuning epochs:**
   ```python
   one_shot.prune(amount=0.5, fine_tune_epochs=1)  # Default: 2-5
   ```

3. **Use CPU for small experiments:**
   ```python
   device = torch.device('cpu')  # Avoids GPU overhead for small models
   ```

### For Best Results

1. **Use full datasets**
2. **Train for more epochs**
3. **Use GPU acceleration**
4. **Fine-tune after pruning**

## Next Steps

### Assignment 1
1. Try different CNN architectures (VGG, DenseNet, EfficientNet)
2. Compare Grad-CAM with Grad-CAM++
3. Experiment with different LIME parameters
4. Apply to other medical imaging datasets

### Assignment 2
1. Compare L1 vs L2 pruning
2. Try structured pruning (prune entire channels)
3. Combine pruning with quantization
4. Measure actual inference latency on edge devices
5. Test on your own models

## Resources

- **Grad-CAM Paper**: https://arxiv.org/abs/1610.02391
- **LIME Paper**: https://arxiv.org/abs/1602.04938
- **Pruning Survey**: https://arxiv.org/abs/2101.09671
- **PyTorch Pruning Tutorial**: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

## Support

For issues or questions:
1. Check the README files in each assignment directory
2. Review the code comments
3. Open an issue on GitHub

## Citation

If you use this code for research, please cite the original papers:
- Selvaraju et al. (2017) - Grad-CAM
- Ribeiro et al. (2016) - LIME
- Frankle & Carbin (2019) - Lottery Ticket Hypothesis
