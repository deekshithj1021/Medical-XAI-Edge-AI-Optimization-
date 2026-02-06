# Assignment 2: Iterative vs One-Shot Pruning on MobileNet

This assignment compares two neural network pruning strategies (iterative and one-shot) on a MobileNetV2 model, analyzing trade-offs between model size, sparsity, and accuracy.

## Overview

Model pruning is a compression technique that removes unnecessary weights from neural networks. This code implements and compares:

1. **One-Shot Pruning**: Prunes a specified percentage of weights all at once, then fine-tunes
2. **Iterative Pruning**: Gradually prunes weights over multiple iterations with fine-tuning after each step

## Key Concepts

### Sparsity
The percentage of weights set to zero. Higher sparsity = more compression.

### Pruning Methods
- **L1 Unstructured Pruning**: Removes weights with smallest absolute values
- **Fine-tuning**: Retraining after pruning to recover accuracy

## Features

### 1. One-Shot Pruning (`OneShotPruning`)
- Applies pruning in a single step
- Faster to execute
- May result in larger accuracy drops
- Good for aggressive compression

### 2. Iterative Pruning (`IterativePruning`)
- Gradually increases sparsity over multiple iterations
- Better accuracy preservation
- Takes longer to execute
- More stable training

### 3. Comprehensive Comparison
- Accuracy vs Sparsity plots
- Model Size reduction analysis
- Compression ratio visualization
- Accuracy drop comparison

## Installation

```bash
# Install required packages
pip install -r ../requirements.txt
```

## Usage

### Basic Usage

```bash
python pruning_comparison.py
```

This will:
1. Load pre-trained MobileNetV2
2. Fine-tune on CIFAR-10 dataset
3. Apply both pruning strategies
4. Generate comparison plots
5. Save results to `outputs/pruning_comparison/`

### Quick Test (Smaller Dataset)

The script uses a subset of CIFAR-10 for faster demonstration:
- Training: 5,000 images (reduced from 50,000)
- Testing: 1,000 images (reduced from 10,000)

For full dataset, modify:
```python
# Use full datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

## Code Structure

```
pruning_comparison.py
├── PruningStrategy        # Base class with common utilities
├── OneShotPruning         # One-shot pruning implementation
├── IterativePruning       # Iterative pruning implementation
├── compare_pruning_strategies()  # Visualization function
├── print_summary_table()  # Results table
└── main()                 # Main execution
```

## Key Functions

### One-Shot Pruning
```python
one_shot = OneShotPruning(model)
result = one_shot.prune(
    amount=0.7,              # Prune 70% of weights
    train_loader=train_loader,
    fine_tune_epochs=2
)
accuracy = one_shot.evaluate(test_loader)
```

### Iterative Pruning
```python
iterative = IterativePruning(model)
results = iterative.prune(
    target_sparsity=0.9,     # Target 90% sparsity
    num_iterations=5,         # Over 5 iterations
    train_loader=train_loader,
    test_loader=test_loader,
    fine_tune_epochs=1
)
```

## Output

### Console Output
```
=== PRUNING STRATEGY COMPARISON SUMMARY ===

One-Shot Pruning Results:
Sparsity (%)    Model Size (MB)       Accuracy (%)
    0.00               8.85               75.20
   30.00               6.20               74.50
   50.00               4.42               72.80
   70.00               2.65               68.90
   90.00               0.88               55.30

Iterative Pruning Results:
Iteration    Sparsity (%)    Model Size (MB)       Accuracy (%)
    0               0.00               8.85               75.20
    1              36.90               5.58               74.80
    2              59.05               3.62               73.90
    3              74.38               2.27               72.50
    4              83.95               1.42               70.80
    5              90.02               0.88               68.20
```

### Visualization Files
- `pruning_comparison.png`: 4-panel comparison plot with:
  1. Accuracy vs Sparsity
  2. Model Size vs Sparsity
  3. Accuracy Drop vs Sparsity
  4. Compression Ratio vs Sparsity

## Understanding the Results

### Accuracy vs Sparsity
Shows how model performance degrades as more weights are pruned. Iterative pruning typically maintains higher accuracy.

### Model Size vs Sparsity
Demonstrates storage savings. Both methods achieve similar compression ratios.

### Accuracy Drop
Measures performance loss from baseline. Lower is better. Iterative pruning shows smaller drops.

### Compression Ratio
Shows how much smaller the model becomes (e.g., 5x = model is 5 times smaller).

## Experimental Parameters

### One-Shot Pruning Amounts
```python
pruning_amounts = [0.0, 0.3, 0.5, 0.7, 0.9]
```
- 0%: Baseline (no pruning)
- 30%: Light pruning
- 50%: Medium pruning
- 70%: Heavy pruning
- 90%: Extreme pruning

### Iterative Pruning Settings
```python
target_sparsity = 0.9      # Final sparsity goal
num_iterations = 5          # Number of pruning steps
fine_tune_epochs = 1        # Epochs between iterations
```

## Customization

### Use Different Model
```python
# Use MobileNetV3 instead
base_model = models.mobilenet_v3_small(pretrained=True)
```

### Change Dataset
```python
# Use MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
```

### Adjust Pruning Schedule
```python
# More aggressive one-shot
pruning_amounts = [0.0, 0.5, 0.8, 0.95, 0.99]

# More gradual iterative
iterative.prune(target_sparsity=0.9, num_iterations=10)
```

### Different Pruning Method
```python
# Use structured pruning (prune entire channels)
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
```

## Performance Considerations

### Speed
- **One-Shot**: Faster (single pruning + fine-tuning)
- **Iterative**: Slower (multiple pruning + fine-tuning cycles)

### Accuracy
- **One-Shot**: Larger accuracy drops at high sparsity
- **Iterative**: Better accuracy preservation

### Recommended Use Cases
- **One-Shot**: When speed is critical, moderate compression needed
- **Iterative**: When accuracy is critical, aggressive compression needed

## Advanced Features

### Calculate Statistics
```python
pruner = PruningStrategy(model)
total_params, nonzero_params = pruner.count_parameters()
sparsity = pruner.get_sparsity()
model_size_mb = pruner.get_model_size()
```

### Custom Fine-tuning
```python
def custom_fine_tune(model, train_loader, epochs=5):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    # ... training loop
```

## Technical Details

### L1 Unstructured Pruning
Removes individual weights based on magnitude:
1. Calculate absolute value of each weight
2. Sort weights by magnitude
3. Remove smallest X% of weights
4. Set pruned weights to zero

### Iterative Sparsity Calculation
For target sparsity S and N iterations:
```
per_iteration_amount = 1 - (1 - S)^(1/N)
```

Example: 90% sparsity in 5 iterations = ~36.9% per iteration

### Making Pruning Permanent
```python
prune.remove(module, 'weight')  # Removes pruning reparameterization
```

## Comparison with Literature

### Expected Results
- One-shot at 90% sparsity: ~10-20% accuracy drop
- Iterative at 90% sparsity: ~5-10% accuracy drop
- Compression ratios: 5-10x for mobile models

### References
1. **Lottery Ticket Hypothesis**: Frankle & Carbin (2019)
2. **Gradual Pruning**: Zhu & Gupta (2017)
3. **MobileNet**: Howard et al. (2017)

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```python
train_loader = DataLoader(train_subset, batch_size=16, ...)
```

### Slow Execution
- Use smaller dataset subset
- Reduce fine-tuning epochs
- Use CPU if GPU unavailable

### Poor Accuracy After Pruning
- Increase fine-tuning epochs
- Use lower learning rate
- Try fewer pruning iterations

## Key Findings

Based on typical experiments:

1. **Iterative pruning preserves accuracy better** at high sparsity levels
2. **One-shot pruning is 3-5x faster** to execute
3. **Both achieve similar compression ratios** (model size reduction)
4. **Fine-tuning is crucial** - without it, accuracy drops significantly
5. **Sweet spot**: 50-70% sparsity balances size and accuracy

## Extensions

### Try These Experiments
1. Compare different pruning criteria (L1 vs L2 vs random)
2. Test structured pruning (prune entire filters)
3. Combine pruning with quantization
4. Measure inference latency on edge devices

### Production Deployment
```python
# Save pruned model
torch.save(model.state_dict(), 'pruned_model.pth')

# Convert to mobile format
from torch.utils.mobile_optimizer import optimize_for_mobile
scripted = torch.jit.script(model)
optimized = optimize_for_mobile(scripted)
optimized._save_for_lite_interpreter('model_mobile.ptl')
```

## License

This code is provided for educational purposes.
