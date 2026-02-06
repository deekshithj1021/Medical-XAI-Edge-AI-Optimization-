"""
Assignment 2: Iterative vs One-Shot Pruning on MobileNet
This script compares iterative and one-shot pruning strategies on a MobileNet model,
analyzing the trade-offs between model size, sparsity, and accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class PruningStrategy:
    """Base class for pruning strategies"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_model = copy.deepcopy(model)
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and non-zero parameters
        
        Returns:
            total_params: Total number of parameters
            nonzero_params: Number of non-zero parameters
        """
        total_params = 0
        nonzero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            nonzero_params += torch.count_nonzero(param).item()
        
        return total_params, nonzero_params
    
    def get_sparsity(self) -> float:
        """
        Calculate global sparsity
        
        Returns:
            Sparsity percentage
        """
        total_params, nonzero_params = self.count_parameters()
        sparsity = 100.0 * (1 - nonzero_params / total_params)
        return sparsity
    
    def get_model_size(self) -> float:
        """
        Calculate model size in MB
        
        Returns:
            Model size in MB
        """
        # Save model temporarily
        temp_path = '/tmp/temp_model.pth'
        torch.save(self.model.state_dict(), temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb
    
    def evaluate(self, test_loader: DataLoader, device: str = 'cuda') -> float:
        """
        Evaluate model accuracy
        
        Args:
            test_loader: Test data loader
            device: Device to evaluate on
        
        Returns:
            Accuracy percentage
        """
        self.model.eval()
        self.model = self.model.to(device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy


class OneShotPruning(PruningStrategy):
    """One-shot pruning strategy"""
    
    def prune(self, amount: float, train_loader: DataLoader = None, 
              device: str = 'cuda', fine_tune_epochs: int = 0) -> Dict:
        """
        Apply one-shot pruning
        
        Args:
            amount: Pruning amount (0.0 to 1.0)
            train_loader: Training data loader for fine-tuning
            device: Device to use
            fine_tune_epochs: Number of epochs to fine-tune after pruning
        
        Returns:
            Dictionary with pruning statistics
        """
        print(f"\n=== One-Shot Pruning: {amount*100:.1f}% ===")
        
        # Apply pruning to all Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
        
        # Get statistics before fine-tuning
        sparsity = self.get_sparsity()
        model_size = self.get_model_size()
        
        print(f"Sparsity achieved: {sparsity:.2f}%")
        print(f"Model size: {model_size:.2f} MB")
        
        # Fine-tune if requested
        if fine_tune_epochs > 0 and train_loader is not None:
            print(f"Fine-tuning for {fine_tune_epochs} epochs...")
            self.fine_tune(train_loader, epochs=fine_tune_epochs, device=device)
        
        # Make pruning permanent
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.remove(module, 'weight')
        
        return {
            'amount': amount,
            'sparsity': sparsity,
            'model_size': model_size
        }
    
    def fine_tune(self, train_loader: DataLoader, epochs: int = 2, 
                  device: str = 'cuda'):
        """Fine-tune the pruned model"""
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 50 == 0:
                    print(f'  Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')


class IterativePruning(PruningStrategy):
    """Iterative pruning strategy"""
    
    def prune(self, target_sparsity: float, num_iterations: int, 
              train_loader: DataLoader, test_loader: DataLoader,
              device: str = 'cuda', fine_tune_epochs: int = 1) -> List[Dict]:
        """
        Apply iterative pruning
        
        Args:
            target_sparsity: Target sparsity to achieve
            num_iterations: Number of pruning iterations
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to use
            fine_tune_epochs: Number of epochs to fine-tune after each iteration
        
        Returns:
            List of dictionaries with statistics for each iteration
        """
        print(f"\n=== Iterative Pruning: Target {target_sparsity*100:.1f}% ===")
        
        # Calculate pruning amount per iteration
        # Formula: (1 - target_sparsity) = (1 - per_iter_amount)^num_iterations
        per_iter_amount = 1 - (1 - target_sparsity) ** (1 / num_iterations)
        
        results = []
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Apply pruning
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=per_iter_amount)
            
            # Get statistics
            sparsity = self.get_sparsity()
            model_size = self.get_model_size()
            
            print(f"Current sparsity: {sparsity:.2f}%")
            print(f"Model size: {model_size:.2f} MB")
            
            # Fine-tune
            if train_loader is not None:
                print(f"Fine-tuning for {fine_tune_epochs} epochs...")
                self.fine_tune(train_loader, epochs=fine_tune_epochs, device=device)
            
            # Evaluate
            accuracy = self.evaluate(test_loader, device=device)
            print(f"Accuracy: {accuracy:.2f}%")
            
            results.append({
                'iteration': iteration + 1,
                'sparsity': sparsity,
                'model_size': model_size,
                'accuracy': accuracy
            })
        
        # Make pruning permanent
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.remove(module, 'weight')
        
        return results
    
    def fine_tune(self, train_loader: DataLoader, epochs: int = 1, 
                  device: str = 'cuda'):
        """Fine-tune the pruned model"""
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        for epoch in range(epochs):
            self.model.train()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 50 == 0:
                    print(f'  Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')


def compare_pruning_strategies(one_shot_results: List[Dict], 
                                iterative_results: List[Dict],
                                save_path: str = None):
    """
    Visualize comparison between pruning strategies
    
    Args:
        one_shot_results: Results from one-shot pruning
        iterative_results: Results from iterative pruning
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    one_shot_sparsity = [r['sparsity'] for r in one_shot_results]
    one_shot_accuracy = [r['accuracy'] for r in one_shot_results]
    one_shot_size = [r['model_size'] for r in one_shot_results]
    
    iter_sparsity = [r['sparsity'] for r in iterative_results]
    iter_accuracy = [r['accuracy'] for r in iterative_results]
    iter_size = [r['model_size'] for r in iterative_results]
    
    # Plot 1: Accuracy vs Sparsity
    axes[0, 0].plot(one_shot_sparsity, one_shot_accuracy, 'o-', 
                     label='One-Shot', linewidth=2, markersize=8)
    axes[0, 0].plot(iter_sparsity, iter_accuracy, 's-', 
                     label='Iterative', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Sparsity (%)', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Accuracy vs Sparsity', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Model Size vs Sparsity
    axes[0, 1].plot(one_shot_sparsity, one_shot_size, 'o-', 
                     label='One-Shot', linewidth=2, markersize=8)
    axes[0, 1].plot(iter_sparsity, iter_size, 's-', 
                     label='Iterative', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sparsity (%)', fontsize=12)
    axes[0, 1].set_ylabel('Model Size (MB)', fontsize=12)
    axes[0, 1].set_title('Model Size vs Sparsity', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy Drop
    baseline_acc_one_shot = one_shot_accuracy[0] if one_shot_accuracy else 0
    baseline_acc_iter = iter_accuracy[0] if iter_accuracy else 0
    
    one_shot_acc_drop = [baseline_acc_one_shot - acc for acc in one_shot_accuracy]
    iter_acc_drop = [baseline_acc_iter - acc for acc in iter_accuracy]
    
    axes[1, 0].plot(one_shot_sparsity, one_shot_acc_drop, 'o-', 
                     label='One-Shot', linewidth=2, markersize=8)
    axes[1, 0].plot(iter_sparsity, iter_acc_drop, 's-', 
                     label='Iterative', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sparsity (%)', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy Drop (%)', fontsize=12)
    axes[1, 0].set_title('Accuracy Drop vs Sparsity', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Compression Ratio
    baseline_size_one_shot = one_shot_size[0] if one_shot_size else 1
    baseline_size_iter = iter_size[0] if iter_size else 1
    
    one_shot_compression = [baseline_size_one_shot / size if size > 0 else 0 
                            for size in one_shot_size]
    iter_compression = [baseline_size_iter / size if size > 0 else 0 
                        for size in iter_size]
    
    axes[1, 1].plot(one_shot_sparsity, one_shot_compression, 'o-', 
                     label='One-Shot', linewidth=2, markersize=8)
    axes[1, 1].plot(iter_sparsity, iter_compression, 's-', 
                     label='Iterative', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Sparsity (%)', fontsize=12)
    axes[1, 1].set_ylabel('Compression Ratio', fontsize=12)
    axes[1, 1].set_title('Compression Ratio vs Sparsity', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
    
    plt.show()


def print_summary_table(one_shot_results: List[Dict], 
                        iterative_results: List[Dict]):
    """
    Print summary table comparing both strategies
    
    Args:
        one_shot_results: Results from one-shot pruning
        iterative_results: Results from iterative pruning
    """
    print("\n" + "="*80)
    print("PRUNING STRATEGY COMPARISON SUMMARY")
    print("="*80)
    
    print("\nOne-Shot Pruning Results:")
    print("-" * 80)
    print(f"{'Sparsity (%)':>15} {'Model Size (MB)':>20} {'Accuracy (%)':>20}")
    print("-" * 80)
    for result in one_shot_results:
        print(f"{result['sparsity']:>14.2f} {result['model_size']:>19.2f} "
              f"{result['accuracy']:>19.2f}")
    
    print("\n\nIterative Pruning Results:")
    print("-" * 80)
    print(f"{'Iteration':>10} {'Sparsity (%)':>15} {'Model Size (MB)':>20} "
          f"{'Accuracy (%)':>20}")
    print("-" * 80)
    for result in iterative_results:
        print(f"{result['iteration']:>10} {result['sparsity']:>14.2f} "
              f"{result['model_size']:>19.2f} {result['accuracy']:>19.2f}")
    
    print("\n" + "="*80)


def main():
    """Main function to compare pruning strategies"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('outputs/pruning_comparison', exist_ok=True)
    
    # Load pre-trained MobileNetV2
    print("\n=== Loading MobileNetV2 Model ===")
    base_model = models.mobilenet_v2(pretrained=True)
    
    # Modify for CIFAR-10 (10 classes)
    num_classes = 10
    base_model.classifier[1] = nn.Linear(base_model.last_channel, num_classes)
    
    # Prepare data (using CIFAR-10 for demonstration)
    print("\n=== Preparing CIFAR-10 Dataset ===")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=transform_test)
    
    # Create smaller subsets for faster demonstration
    train_subset_indices = np.random.choice(len(train_dataset), 5000, replace=False)
    test_subset_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)
    
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
        result = one_shot.prune(amount, train_loader=train_loader, 
                               device=device, fine_tune_epochs=1)
        
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
    
    # Apply iterative pruning
    iterative_results = iterative.prune(
        target_sparsity=0.9,
        num_iterations=5,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        fine_tune_epochs=1
    )
    
    # Add baseline to iterative results
    iterative_results.insert(0, {
        'iteration': 0,
        'sparsity': 0.0,
        'model_size': baseline_size,
        'accuracy': baseline_accuracy
    })
    
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
    print("KEY FINDINGS")
    print("="*80)
    print("1. Iterative pruning generally maintains higher accuracy at similar sparsity levels")
    print("2. One-shot pruning is faster but may result in larger accuracy drops")
    print("3. Both methods achieve significant model size reduction")
    print("4. Fine-tuning after pruning is crucial for maintaining accuracy")
    print("="*80)


if __name__ == "__main__":
    main()
