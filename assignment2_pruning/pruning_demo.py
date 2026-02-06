"""
Demo script for Assignment 2: Iterative vs One-Shot Pruning on MobileNet
This version creates synthetic data for demonstration purposes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import numpy as np
import sys
import os

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pruning_comparison import (
    OneShotPruning, IterativePruning, PruningStrategy,
    compare_pruning_strategies, print_summary_table
)
import copy


def create_synthetic_dataset(num_samples=1000, num_classes=10):
    """Create synthetic dataset for demonstration"""
    # Create random images (num_samples, 3, 224, 224)
    images = torch.randn(num_samples, 3, 224, 224)
    # Create random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    return dataset


def main():
    """Main function with synthetic data"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('outputs/pruning_comparison', exist_ok=True)
    
    # Load pre-trained MobileNetV2
    print("\n=== Loading MobileNetV2 Model ===")
    base_model = models.mobilenet_v2(pretrained=True)
    
    # Modify for 10 classes
    num_classes = 10
    base_model.classifier[1] = nn.Linear(base_model.last_channel, num_classes)
    
    # Create synthetic datasets
    print("\n=== Creating Synthetic Dataset ===")
    print("(For real experiments, use CIFAR-10 or your own dataset)")
    
    train_dataset = create_synthetic_dataset(num_samples=500, num_classes=num_classes)
    test_dataset = create_synthetic_dataset(num_samples=200, num_classes=num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate baseline model
    print("\n=== Evaluating Baseline Model ===")
    baseline_strategy = PruningStrategy(copy.deepcopy(base_model))
    baseline_accuracy = baseline_strategy.evaluate(test_loader, device=device)
    baseline_size = baseline_strategy.get_model_size()
    
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"Baseline Model Size: {baseline_size:.2f} MB")
    
    # One-Shot Pruning
    print("\n" + "="*80)
    print("ONE-SHOT PRUNING EXPERIMENTS")
    print("="*80)
    
    one_shot_results = []
    pruning_amounts = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    for amount in pruning_amounts:
        # Create fresh copy of model
        model_copy = copy.deepcopy(base_model)
        one_shot = OneShotPruning(model_copy)
        
        # Apply pruning
        result = one_shot.prune(amount, train_loader=None, 
                               device=device, fine_tune_epochs=0)
        
        # Evaluate
        accuracy = one_shot.evaluate(test_loader, device=device)
        result['accuracy'] = accuracy
        
        one_shot_results.append(result)
        
        print(f"Amount: {amount*100:.1f}% -> Sparsity: {result['sparsity']:.2f}%, "
              f"Accuracy: {accuracy:.2f}%, Size: {result['model_size']:.2f} MB")
    
    # Iterative Pruning
    print("\n" + "="*80)
    print("ITERATIVE PRUNING EXPERIMENTS")
    print("="*80)
    
    # Create fresh copy of model
    model_copy = copy.deepcopy(base_model)
    iterative = IterativePruning(model_copy)
    
    # Apply iterative pruning (no fine-tuning for demo)
    print("Running iterative pruning with 3 iterations (no fine-tuning for demo)...")
    
    # Manual iterative pruning for demo
    iterative_results = []
    target_amounts = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    for i, amount in enumerate(target_amounts):
        if amount == 0.0:
            # Baseline
            model_copy_iter = copy.deepcopy(base_model)
            iter_strat = PruningStrategy(model_copy_iter)
            sparsity = 0.0
            model_size = iter_strat.get_model_size()
            accuracy = iter_strat.evaluate(test_loader, device=device)
        else:
            # Apply pruning
            model_copy_iter = copy.deepcopy(base_model)
            iter_strat = IterativePruning(model_copy_iter)
            
            # Apply pruning without fine-tuning
            import torch.nn.utils.prune as prune
            for name, module in iter_strat.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=amount)
            
            sparsity = iter_strat.get_sparsity()
            model_size = iter_strat.get_model_size()
            accuracy = iter_strat.evaluate(test_loader, device=device)
            
            # Make permanent
            for name, module in iter_strat.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.remove(module, 'weight')
        
        iterative_results.append({
            'iteration': i,
            'sparsity': sparsity,
            'model_size': model_size,
            'accuracy': accuracy
        })
        
        print(f"Iteration {i}: Sparsity: {sparsity:.2f}%, "
              f"Accuracy: {accuracy:.2f}%, Size: {model_size:.2f} MB")
    
    # Print summary
    print_summary_table(one_shot_results, iterative_results)
    
    # Visualize comparison
    print("\n=== Creating Comparison Visualizations ===")
    compare_pruning_strategies(
        one_shot_results,
        iterative_results,
        save_path='outputs/pruning_comparison/pruning_comparison.png'
    )
    
    print("\n=== Pruning Comparison Complete ===")
    print("Outputs saved to: outputs/pruning_comparison/")
    
    # Key findings
    print("\n" + "="*80)
    print("DEMO NOTES")
    print("="*80)
    print("This demo uses synthetic data for quick testing.")
    print("For real experiments:")
    print("1. Use the full pruning_comparison.py script")
    print("2. Ensure internet access to download CIFAR-10")
    print("3. Enable fine-tuning for better accuracy preservation")
    print("="*80)


if __name__ == "__main__":
    main()
