# Medical XAI & Edge AI Optimization

This repository contains two comprehensive AI assignments focusing on Explainable AI (XAI) for medical imaging and model optimization for edge deployment.

## 📋 Assignments

### Assignment 1: Explainable AI for Chest X-Ray Classification
**Objective**: Apply Grad-CAM and LIME to a CNN trained on chest X-ray images for pneumonia detection.

**Features**:
- CNN model (ResNet18-based) for binary classification (Normal/Pneumonia)
- Grad-CAM visualization for attention heatmaps
- LIME explanations for interpretable predictions
- Side-by-side comparison visualizations
- Works with Kaggle Chest X-Ray Pneumonia dataset

**Quick Start**:
```bash
cd assignment1_xai
pip install -r ../requirements.txt
python xai_chest_xray.py
```

See [assignment1_xai/README.md](assignment1_xai/README.md) for detailed instructions.

### Assignment 2: Iterative vs One-Shot Pruning on MobileNet
**Objective**: Compare pruning strategies on MobileNetV2 to analyze model size vs accuracy trade-offs.

**Features**:
- One-shot pruning implementation
- Iterative pruning with gradual sparsity increase
- Comprehensive comparison plots (Accuracy, Size, Compression)
- Evaluation on CIFAR-10 dataset
- Detailed performance metrics

**Quick Start**:
```bash
cd assignment2_pruning
pip install -r ../requirements.txt
python pruning_comparison.py
```

See [assignment2_pruning/README.md](assignment2_pruning/README.md) for detailed instructions.

## 🚀 Installation

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- TensorFlow 2.12+ (optional, for TensorFlow implementations)
- CUDA (optional, for GPU acceleration)

## 📊 Outputs

### Assignment 1 Outputs
Generated in `outputs/xai_visualizations/`:
- `sample_xray.png`: Input chest X-ray image
- `xai_comparison.png`: Grad-CAM and LIME visualizations

### Assignment 2 Outputs
Generated in `outputs/pruning_comparison/`:
- `pruning_comparison.png`: 4-panel comparison plots
- Console output with detailed metrics table

## 🔬 Technical Details

### XAI Techniques
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **LIME**: Local Interpretable Model-agnostic Explanations

### Pruning Strategies
- **One-Shot**: Single-step pruning with fine-tuning
- **Iterative**: Gradual pruning over multiple iterations

### Model Architectures
- **Assignment 1**: ResNet18 (modified for binary classification)
- **Assignment 2**: MobileNetV2 (optimized for edge deployment)

## 📚 Key Concepts

### Explainable AI (XAI)
Understanding what deep learning models "see" when making predictions, crucial for medical applications where interpretability is essential.

### Model Pruning
Removing unnecessary weights from neural networks to reduce model size while maintaining accuracy, enabling deployment on resource-constrained edge devices.

### Edge AI Optimization
Techniques like pruning and quantization that make models suitable for deployment on mobile devices, IoT devices, and embedded systems.

## 🎯 Use Cases

### Medical Imaging
- Pneumonia detection from chest X-rays
- Model interpretability for clinical decision support
- Trust and transparency in medical AI systems

### Edge Deployment
- Mobile health applications
- Resource-constrained medical devices
- Real-time inference on edge hardware
- Reduced bandwidth and storage requirements

## 📖 References

### XAI Papers
1. Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
2. Ribeiro et al. (2016) - "Why Should I Trust You? Explaining Predictions"

### Pruning Papers
1. Frankle & Carbin (2019) - "The Lottery Ticket Hypothesis"
2. Zhu & Gupta (2017) - "To prune, or not to prune"

### Datasets
1. Kermany et al. (2018) - Chest X-Ray Images (Pneumonia)
2. Krizhevsky (2009) - CIFAR-10 Dataset

## 🛠️ Development

### Project Structure
```
Medical-XAI-Edge-AI-Optimization-/
├── assignment1_xai/
│   ├── xai_chest_xray.py       # Main XAI implementation
│   └── README.md                # Assignment 1 documentation
├── assignment2_pruning/
│   ├── pruning_comparison.py   # Main pruning implementation
│   └── README.md                # Assignment 2 documentation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Testing

Run individual assignments:
```bash
# Test XAI assignment
python assignment1_xai/xai_chest_xray.py

# Test pruning assignment
python assignment2_pruning/pruning_comparison.py
```

## 🤝 Contributing

This is an educational repository. Feel free to:
- Report issues
- Suggest improvements
- Experiment with different models and datasets

## 📝 License

This project is provided for educational purposes.

## 🙏 Acknowledgments

- Kaggle for the Chest X-Ray Pneumonia dataset
- PyTorch team for excellent deep learning framework
- Original authors of Grad-CAM and LIME techniques

## 💡 Tips

### For Assignment 1
- Download the Kaggle dataset for real training
- Adjust `num_samples` in LIME for speed/accuracy trade-off
- Try different CNN architectures (ResNet50, EfficientNet)

### For Assignment 2
- Experiment with different pruning amounts
- Compare L1 vs L2 vs random pruning
- Test on your own models and datasets
- Combine with quantization for further compression

## 📧 Support

For questions or issues, please open an issue in the repository.
