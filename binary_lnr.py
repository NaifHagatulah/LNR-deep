"""
Binary Classification Label Noise Rebalancing (LNR)

This module implements the LNR algorithm for binary classification as described in Algorithm 1.
It identifies majority class samples that have high predicted probability for the minority class
and flips their labels to rebalance the dataset.

Author: Adapted from LNR multi-class implementation
"""

import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLNR:
    """
    Binary Label Noise Rebalancing
    
    Implements Algorithm 1 from the paper for binary classification.
    Flips labels of majority class (0) samples to minority class (1)
    based on classifier predictions and a noise model.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        classifier: nn.Module,
        threshold: float = 3.0,
        n_passes: int = 2,
        device: str = 'cuda'
    ):
        """
        Initialize Binary LNR
        
        Args:
            model: Feature extractor (e.g., ResNet backbone)
            classifier: Classification head
            threshold: t_flip parameter - controls flip sensitivity
            n_passes: Number of forward passes to average predictions
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.classifier = classifier
        self.threshold = threshold
        self.n_passes = n_passes
        self.device = device
        
        # Set models to eval mode
        self.model.eval()
        self.classifier.eval()
        
    def collect_predictions(
        self, 
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Tuple[float, int]]:
        """
        Collect predictions over multiple passes through the dataset
        
        Args:
            dataloader: DataLoader that returns (index, x, target)
            
        Returns:
            Dictionary mapping sample index to (avg_prediction, label)
        """
        predictions = {}
        
        print(f"Collecting predictions over {self.n_passes} passes...")
        
        with torch.no_grad():
            for pass_idx in range(self.n_passes):
                print(f"  Pass {pass_idx + 1}/{self.n_passes}")
                
                for batch_idx, (indices, x, targets) in enumerate(dataloader):
                    # Move to device
                    x = x.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    features = self.model(x)
                    logits = self.classifier(features.detach())
                    
                    # Get probability of class 1 (minority)
                    probs = F.softmax(logits, dim=1)
                    prob_class1 = probs[:, 1].cpu().numpy()
                    
                    # Store predictions
                    for i, idx in enumerate(indices):
                        idx_str = str(idx.item())
                        
                        if idx_str not in predictions:
                            predictions[idx_str] = []
                        
                        predictions[idx_str].append((
                            prob_class1[i],
                            targets[i].item()
                        ))
                    
                    if batch_idx % 50 == 0:
                        print(f"    Batch {batch_idx}/{len(dataloader)}")
        
        # Average predictions
        avg_predictions = {}
        for idx_str, preds_list in predictions.items():
            avg_pred = np.mean([p[0] for p in preds_list])
            label = preds_list[0][1]  # Label should be same across passes
            avg_predictions[idx_str] = (avg_pred, label)
        
        print(f"Collected predictions for {len(avg_predictions)} samples")
        
        return avg_predictions
    
    def compute_noise_model(
        self,
        predictions: Dict[str, Tuple[float, int]]
    ) -> Dict[str, Dict]:
        """
        Compute noise model following Algorithm 1
        
        Args:
            predictions: Dictionary mapping index to (prediction, label)
            
        Returns:
            noise_info: Dictionary containing:
                - 'noise_flag': Dict mapping indices to flipped labels
                - 'statistics': Stats about flipping
        """
        print("\nComputing noise model...")
        
        # Separate predictions by class
        majority_preds = []  # Class 0
        minority_preds = []  # Class 1
        
        for idx_str, (pred, label) in predictions.items():
            if label == 0:
                majority_preds.append(pred)
            else:
                minority_preds.append(pred)
        
        majority_preds = np.array(majority_preds)
        minority_preds = np.array(minority_preds)
        
        print(f"  Majority class (0) samples: {len(majority_preds)}")
        print(f"  Minority class (1) samples: {len(minority_preds)}")
        
        # Compute statistics for majority class (IndMA)
        mu = np.mean(majority_preds)
        sigma = np.std(majority_preds)
        
        print(f"  Majority class statistics:")
        print(f"    Mean prediction η̂: {mu:.4f}")
        print(f"    Std prediction σ: {sigma:.4f}")
        
        # Compute flip decisions
        noise_flag = {}
        flip_rates = []
        z_scores = []
        flip_count = 0
        
        for idx_str, (pred, label) in predictions.items():
            if label == 0:  # Only consider majority class
                # Compute Z-score: Z = (η̂ - μ) / σ
                z_score = (pred - mu) / sigma
                z_scores.append(z_score)
                
                # Compute flip probability: ρ = max(tanh(Z - t_flip), 0)
                flip_rate = np.tanh(z_score - self.threshold)
                flip_rate = max(flip_rate, 0.0)
                flip_rates.append(flip_rate)
                
                # Sample flip decision: U ~ Bernoulli(ρ)
                if np.random.rand() < flip_rate:
                    noise_flag[idx_str] = 1  # Flip to minority class
                    flip_count += 1
        
        # Compute statistics
        flip_rates = np.array(flip_rates)
        z_scores = np.array(z_scores)
        
        statistics = {
            'total_majority': len(majority_preds),
            'total_minority': len(minority_preds),
            'flipped_count': flip_count,
            'flip_percentage': 100 * flip_count / len(majority_preds),
            'mean_flip_rate': np.mean(flip_rates),
            'mean_z_score': np.mean(z_scores),
            'max_z_score': np.max(z_scores),
            'n_high_z': np.sum(z_scores > self.threshold)
        }
        
        print(f"\n  Flipping Results:")
        print(f"    Flipped samples: {flip_count} / {len(majority_preds)}")
        print(f"    Flip percentage: {statistics['flip_percentage']:.2f}%")
        print(f"    Mean flip rate ρ: {statistics['mean_flip_rate']:.4f}")
        print(f"    Mean Z-score: {statistics['mean_z_score']:.4f}")
        print(f"    Samples with Z > threshold: {statistics['n_high_z']}")
        
        # After flipping
        new_majority_count = len(majority_preds) - flip_count
        new_minority_count = len(minority_preds) + flip_count
        new_imbalance_ratio = new_majority_count / new_minority_count
        
        print(f"\n  New class distribution:")
        print(f"    Class 0 (majority): {new_majority_count}")
        print(f"    Class 1 (minority): {new_minority_count}")
        print(f"    Imbalance ratio: {new_imbalance_ratio:.2f}")
        
        noise_info = {
            'noise_flag': noise_flag,
            'statistics': statistics
        }
        
        return noise_info
    
    def generate_noise_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate complete noise model
        
        Args:
            dataloader: Training data loader
            save_path: Optional path to save noise model
            
        Returns:
            noise_info: Dictionary with noise model
        """
        # Step 1: Collect predictions
        predictions = self.collect_predictions(dataloader)
        
        # Step 2: Compute noise model
        noise_info = self.compute_noise_model(predictions)
        
        # Step 3: Save if requested
        if save_path is not None:
            print(f"\nSaving noise model to {save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump(noise_info, f)
        
        return noise_info
    
    @staticmethod
    def load_noise_model(path: str) -> Dict:
        """Load noise model from file"""
        print(f"Loading noise model from {path}")
        with open(path, 'rb') as f:
            noise_info = pickle.load(f)
        print(f"  Loaded model with {len(noise_info['noise_flag'])} flipped samples")
        return noise_info


def apply_label_flips(
    targets: torch.Tensor,
    indices: torch.Tensor,
    noise_info: Dict
) -> torch.Tensor:
    """
    Apply label flips to a batch
    
    Args:
        targets: Original targets tensor
        indices: Sample indices tensor
        noise_info: Noise model dictionary
        
    Returns:
        Modified targets tensor
    """
    noise_flag = noise_info['noise_flag']
    
    for i, idx in enumerate(indices):
        idx_str = str(idx.item())
        if idx_str in noise_flag:
            targets[i] = noise_flag[idx_str]
    
    return targets


def visualize_noise_model(
    predictions: Dict[str, Tuple[float, int]],
    noise_info: Dict,
    save_path: Optional[str] = None
):
    """
    Visualize the noise model distribution
    
    Args:
        predictions: Prediction dictionary
        noise_info: Noise model
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    # Separate predictions
    majority_preds = []
    minority_preds = []
    flipped_preds = []
    
    noise_flag = noise_info['noise_flag']
    
    for idx_str, (pred, label) in predictions.items():
        if label == 0:
            if idx_str in noise_flag:
                flipped_preds.append(pred)
            else:
                majority_preds.append(pred)
        else:
            minority_preds.append(pred)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of predictions
    ax = axes[0, 0]
    ax.hist(majority_preds, bins=50, alpha=0.7, label='Majority (kept)', color='blue')
    ax.hist(flipped_preds, bins=50, alpha=0.7, label='Majority (flipped)', color='red')
    ax.hist(minority_preds, bins=50, alpha=0.7, label='Minority (original)', color='green')
    ax.set_xlabel('Prediction (P(Y=1))')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Z-score distribution
    ax = axes[0, 1]
    mu = np.mean(majority_preds + flipped_preds)
    sigma = np.std(majority_preds + flipped_preds)
    
    z_majority = [(p - mu) / sigma for p in majority_preds]
    z_flipped = [(p - mu) / sigma for p in flipped_preds]
    
    ax.hist(z_majority, bins=50, alpha=0.7, label='Majority (kept)', color='blue')
    ax.hist(z_flipped, bins=50, alpha=0.7, label='Majority (flipped)', color='red')
    ax.axvline(noise_info['statistics']['mean_z_score'], 
               color='black', linestyle='--', label='Mean Z-score')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Z-scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Class distribution before/after
    ax = axes[1, 0]
    categories = ['Before', 'After']
    majority_counts = [
        len(majority_preds) + len(flipped_preds),
        len(majority_preds)
    ]
    minority_counts = [
        len(minority_preds),
        len(minority_preds) + len(flipped_preds)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, majority_counts, width, label='Majority (0)', color='blue')
    ax.bar(x + width/2, minority_counts, width, label='Minority (1)', color='green')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats = noise_info['statistics']
    table_data = [
        ['Total Majority', f"{stats['total_majority']}"],
        ['Total Minority', f"{stats['total_minority']}"],
        ['Flipped Count', f"{stats['flipped_count']}"],
        ['Flip Percentage', f"{stats['flip_percentage']:.2f}%"],
        ['Mean Flip Rate', f"{stats['mean_flip_rate']:.4f}"],
        ['Mean Z-score', f"{stats['mean_z_score']:.4f}"],
        ['Max Z-score', f"{stats['max_z_score']:.4f}"],
        ['High Z (>threshold)', f"{stats['n_high_z']}"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='left',
                     colLabels=['Metric', 'Value'],
                     loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('LNR Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    """
    Example of how to use Binary LNR
    """
    print("Binary LNR Implementation")
    print("=" * 60)
    print()
    print("This module implements Algorithm 1 for binary classification:")
    print()
    print("Algorithm 1: Flip labels in binary classification with noise model ρ(x)")
    print("-" * 60)
    print("Input: Feature X and labels Y, Classifier Cf")
    print("Parameters: Threshold t_flip")
    print()
    print("Steps:")
    print("1. IndMA ← index(Y = 0)  # Get majority class indices")
    print("2. η̂[i] ← Cf(X[i])       # Get predictions for all samples")
    print("3. μ ← mean(η̂[IndMA])    # Mean prediction on majority")
    print("4. σ ← std(η̂[IndMA])     # Std prediction on majority")
    print()
    print("For each iMA in IndMA:")
    print("  5. Z[iMA] ← (η̂[iMA] - μ) / σ        # Compute Z-score")
    print("  6. ρ[iMA] ← max(tanh(Z - t_flip), 0) # Compute flip rate")
    print("  7. U ~ Bernoulli(ρ[iMA])             # Sample flip decision")
    print("  8. if U == 1: Y[iMA] ← 1             # Flip label")
    print()
    print("Return: Y (modified labels)")
    print("=" * 60)
    print()
    print("Usage:")
    print("------")
    print("1. Create BinaryLNR object with pre-trained model")
    print("2. Call generate_noise_model() with training dataloader")
    print("3. Use apply_label_flips() in training loop to flip labels")
    print("4. Train classifier with flipped labels")
    print()
    print("See train_binary_lnr.py for complete training example")
