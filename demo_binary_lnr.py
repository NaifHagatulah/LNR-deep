"""
Simple Binary LNR Example

This script demonstrates the LNR algorithm on a toy dataset to show
how the label flipping works step-by-step.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def create_imbalanced_dataset(n_samples=1000, imbalance_ratio=0.1, random_state=42):
    """
    Create synthetic imbalanced binary classification dataset
    """
    # Generate balanced dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        flip_y=0.01,
        random_state=random_state
    )
    
    # Create imbalance by subsampling minority class
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    
    n_minority = int(len(majority_idx) * imbalance_ratio)
    np.random.seed(random_state)
    minority_idx = np.random.choice(minority_idx, size=n_minority, replace=False)
    
    # Combine indices
    indices = np.concatenate([majority_idx, minority_idx])
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    print(f"Created imbalanced dataset:")
    print(f"  Total samples: {len(y)}")
    print(f"  Class 0 (majority): {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)")
    print(f"  Class 1 (minority): {np.sum(y == 1)} ({100*np.mean(y == 1):.1f}%)")
    print(f"  Imbalance ratio: {np.sum(y == 1) / np.sum(y == 0):.3f}")
    
    return X, y


def binary_lnr_algorithm(X, y, classifier, threshold=3.0, verbose=True):
    """
    Implement Algorithm 1: Binary LNR
    
    Args:
        X: Features
        y: Labels (0 or 1)
        classifier: Trained classifier
        threshold: t_flip parameter
        verbose: Print step-by-step info
        
    Returns:
        y_flipped: Modified labels
        flip_info: Dictionary with flipping information
    """
    if verbose:
        print("\n" + "="*80)
        print("Algorithm 1: Binary Label Noise Rebalancing")
        print("="*80)
    
    # Step 1: Get indices of majority class (Y = 0)
    IndMA = np.where(y == 0)[0]
    
    if verbose:
        print(f"\nStep 1: Identify majority class")
        print(f"  IndMA = indices where Y = 0")
        print(f"  |IndMA| = {len(IndMA)}")
    
    # Step 2: Get predictions from classifier
    # η̂[i] ← Cf(X[i])
    # For binary classification, this is P(Y=1|X)
    eta_hat = classifier.predict_proba(X)[:, 1]
    
    if verbose:
        print(f"\nStep 2: Get predictions")
        print(f"  η̂[i] ← Cf(X[i]) for all i")
        print(f"  η̂ represents P(Y=1|X)")
    
    # Step 3: Calculate statistics on majority class
    # μ ← mean(η̂[IndMA])
    # σ ← std(η̂[IndMA])
    mu = np.mean(eta_hat[IndMA])
    sigma = np.std(eta_hat[IndMA])
    
    if verbose:
        print(f"\nStep 3: Calculate statistics on majority class")
        print(f"  μ = mean(η̂[IndMA]) = {mu:.4f}")
        print(f"  σ = std(η̂[IndMA]) = {sigma:.4f}")
    
    # Step 4-8: For each majority class sample, compute flip decision
    y_flipped = y.copy()
    Z = np.zeros(len(y))
    rho = np.zeros(len(y))
    flipped_indices = []
    
    if verbose:
        print(f"\nStep 4-8: Compute flip decisions for each majority sample")
        print(f"  For each iMA in IndMA:")
    
    for iMA in IndMA:
        # Step 4: Compute Z-score
        # Z[iMA] ← (η̂[iMA] - μ) / σ
        Z[iMA] = (eta_hat[iMA] - mu) / sigma
        
        # Step 5: Compute flip probability (noise rate)
        # ρ[iMA] ← max(tanh(Z[iMA] - t_flip), 0)
        rho[iMA] = max(np.tanh(Z[iMA] - threshold), 0.0)
        
        # Step 6: Sample from Bernoulli distribution
        # U ← Bernoulli(ρ[iMA])
        U = np.random.binomial(1, rho[iMA])
        
        # Step 7-8: Flip label if U = 1
        # if U == 1: Y[iMA] ← 1
        if U == 1:
            y_flipped[iMA] = 1
            flipped_indices.append(iMA)
    
    if verbose:
        print(f"    Z-score: (η̂ - μ) / σ")
        print(f"    ρ (flip rate): max(tanh(Z - {threshold}), 0)")
        print(f"    U ~ Bernoulli(ρ)")
        print(f"    if U = 1: flip label 0 → 1")
        
        print(f"\nResults:")
        print(f"  Samples with Z > {threshold}: {np.sum(Z[IndMA] > threshold)}")
        print(f"  Mean Z-score: {np.mean(Z[IndMA]):.4f}")
        print(f"  Max Z-score: {np.max(Z[IndMA]):.4f}")
        print(f"  Mean flip rate: {np.mean(rho[IndMA]):.4f}")
        print(f"  Flipped samples: {len(flipped_indices)} / {len(IndMA)}")
        print(f"  Flip percentage: {100*len(flipped_indices)/len(IndMA):.2f}%")
        
        print(f"\nFinal class distribution:")
        print(f"  Class 0: {np.sum(y_flipped == 0)} (was {np.sum(y == 0)})")
        print(f"  Class 1: {np.sum(y_flipped == 1)} (was {np.sum(y == 1)})")
        print(f"  New imbalance ratio: {np.sum(y_flipped == 1) / np.sum(y_flipped == 0):.3f}")
    
    flip_info = {
        'eta_hat': eta_hat,
        'Z': Z,
        'rho': rho,
        'mu': mu,
        'sigma': sigma,
        'IndMA': IndMA,
        'flipped_indices': flipped_indices,
        'threshold': threshold
    }
    
    return y_flipped, flip_info


def visualize_lnr(X, y, y_flipped, flip_info, classifier):
    """
    Visualize the LNR algorithm
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Extract info
    eta_hat = flip_info['eta_hat']
    Z = flip_info['Z']
    rho = flip_info['rho']
    IndMA = flip_info['IndMA']
    flipped_indices = flip_info['flipped_indices']
    threshold = flip_info['threshold']
    
    # 1. Prediction distribution
    ax = axes[0, 0]
    kept_majority = [i for i in IndMA if i not in flipped_indices]
    ax.hist(eta_hat[kept_majority], bins=30, alpha=0.7, label='Majority (kept)', color='blue')
    ax.hist(eta_hat[flipped_indices], bins=30, alpha=0.7, label='Majority (flipped)', color='red')
    ax.hist(eta_hat[y == 1], bins=30, alpha=0.7, label='Minority (original)', color='green')
    ax.axvline(flip_info['mu'], color='black', linestyle='--', linewidth=2, label='μ')
    ax.set_xlabel('Prediction η̂ (P(Y=1))')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Z-score distribution
    ax = axes[0, 1]
    ax.hist(Z[kept_majority], bins=30, alpha=0.7, label='Majority (kept)', color='blue')
    ax.hist(Z[flipped_indices], bins=30, alpha=0.7, label='Majority (flipped)', color='red')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f't_flip={threshold}')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Z-scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Flip rate distribution
    ax = axes[0, 2]
    ax.hist(rho[kept_majority], bins=30, alpha=0.7, label='Majority (kept)', color='blue')
    ax.hist(rho[flipped_indices], bins=30, alpha=0.7, label='Majority (flipped)', color='red')
    ax.set_xlabel('Flip rate ρ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Flip Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Z vs Prediction scatter
    ax = axes[1, 0]
    ax.scatter(eta_hat[kept_majority], Z[kept_majority], alpha=0.5, s=20, label='Kept', color='blue')
    ax.scatter(eta_hat[flipped_indices], Z[flipped_indices], alpha=0.7, s=30, label='Flipped', color='red')
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label=f'Z={threshold}')
    ax.axvline(flip_info['mu'], color='gray', linestyle='--', linewidth=1, label=f'μ={flip_info["mu"]:.3f}')
    ax.set_xlabel('Prediction η̂')
    ax.set_ylabel('Z-score')
    ax.set_title('Z-score vs Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Flip rate curve
    ax = axes[1, 1]
    z_range = np.linspace(-3, 10, 100)
    rho_curve = np.maximum(np.tanh(z_range - threshold), 0)
    ax.plot(z_range, rho_curve, 'b-', linewidth=2, label='ρ(Z) = max(tanh(Z - t_flip), 0)')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f't_flip={threshold}')
    ax.scatter(Z[flipped_indices], rho[flipped_indices], alpha=0.7, s=30, color='red', label='Flipped samples')
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Flip rate ρ')
    ax.set_title('Flip Rate Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Class distribution before/after
    ax = axes[1, 2]
    categories = ['Before', 'After']
    class0_counts = [np.sum(y == 0), np.sum(y_flipped == 0)]
    class1_counts = [np.sum(y == 1), np.sum(y_flipped == 1)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, class0_counts, width, label='Class 0 (Majority)', color='blue')
    ax.bar(x + width/2, class1_counts, width, label='Class 1 (Minority)', color='green')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add numbers on bars
    for i, v in enumerate(class0_counts):
        ax.text(i - width/2, v + 5, str(v), ha='center', va='bottom')
    for i, v in enumerate(class1_counts):
        ax.text(i + width/2, v + 5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('binary_lnr_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: binary_lnr_visualization.png")
    plt.show()


def main():
    print("="*80)
    print("Binary LNR Demo: Step-by-Step Walkthrough")
    print("="*80)
    
    # Set random seed
    np.random.seed(42)
    
    # Create imbalanced dataset
    print("\n--- Creating Dataset ---")
    X, y = create_imbalanced_dataset(n_samples=1000, imbalance_ratio=0.1)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    # Train initial classifier
    print("\n--- Training Initial Classifier ---")
    classifier = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        verbose=False
    )
    classifier.fit(X_train, y_train)
    
    print("Initial classifier trained")
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    print("\nInitial Performance (on test set):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    # Apply LNR algorithm
    print("\n--- Applying LNR Algorithm ---")
    y_train_flipped, flip_info = binary_lnr_algorithm(
        X_train, y_train, classifier, threshold=3.0, verbose=True
    )
    
    # Train new classifier with flipped labels
    print("\n--- Training Classifier with Flipped Labels ---")
    classifier_lnr = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        verbose=False
    )
    classifier_lnr.fit(X_train, y_train_flipped)
    
    print("LNR classifier trained")
    
    # Evaluate on test set
    y_pred_lnr = classifier_lnr.predict(X_test)
    print("\nLNR Performance (on test set):")
    print(confusion_matrix(y_test, y_pred_lnr))
    print(classification_report(y_test, y_pred_lnr, target_names=['Class 0', 'Class 1']))
    
    # Compare performance
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    
    acc_before = accuracy_score(y_test, y_pred)
    acc_after = accuracy_score(y_test, y_pred_lnr)
    
    bal_acc_before = balanced_accuracy_score(y_test, y_pred)
    bal_acc_after = balanced_accuracy_score(y_test, y_pred_lnr)
    
    f1_before = f1_score(y_test, y_pred)
    f1_after = f1_score(y_test, y_pred_lnr)
    
    print(f"\nAccuracy:")
    print(f"  Before LNR: {acc_before:.4f}")
    print(f"  After LNR:  {acc_after:.4f}")
    print(f"  Change:     {acc_after - acc_before:+.4f}")
    
    print(f"\nBalanced Accuracy:")
    print(f"  Before LNR: {bal_acc_before:.4f}")
    print(f"  After LNR:  {bal_acc_after:.4f}")
    print(f"  Change:     {bal_acc_after - bal_acc_before:+.4f}")
    
    print(f"\nF1 Score:")
    print(f"  Before LNR: {f1_before:.4f}")
    print(f"  After LNR:  {f1_after:.4f}")
    print(f"  Change:     {f1_after - f1_before:+.4f}")
    
    # Visualize
    print("\n--- Creating Visualizations ---")
    try:
        visualize_lnr(X_train, y_train, y_train_flipped, flip_info, classifier)
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
