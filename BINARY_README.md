# Binary Classification with Label Noise Rebalancing (LNR)

This directory contains a complete implementation of the **Binary LNR Algorithm** for imbalanced binary classification, adapted from the paper "Learning Imbalanced Data with Beneficial Label Noise" (ICML 2025).

## Algorithm Overview

### The Problem
In imbalanced binary classification, the majority class (label 0) has many more samples than the minority class (label 1). Traditional training methods struggle because:
- Models become biased toward the majority class
- Minority class samples are often misclassified
- Standard accuracy metrics are misleading

### The Solution: Label Noise Rebalancing

LNR identifies majority class samples that are **feature-wise similar to the minority class** and flips their labels. This "beneficial noise" helps rebalance the dataset without discarding data or creating synthetic samples.

### Algorithm 1: Binary LNR

```
Input: 
  - X: Features
  - Y: Labels (0 = majority, 1 = minority)
  - Cf: Pre-trained classifier
  - t_flip: Threshold parameter

Steps:
1. IndMA ← indices where Y = 0            # Get majority class indices
2. η̂[i] ← Cf(X[i]) for all i             # Get predictions P(Y=1|X)
3. μ ← mean(η̂[IndMA])                     # Mean prediction on majority
4. σ ← std(η̂[IndMA])                      # Std prediction on majority

For each iMA in IndMA:
  5. Z[iMA] ← (η̂[iMA] - μ) / σ            # Compute Z-score
  6. ρ[iMA] ← max(tanh(Z - t_flip), 0)    # Compute flip probability
  7. U ~ Bernoulli(ρ[iMA])                # Sample flip decision
  8. if U = 1: Y[iMA] ← 1                 # Flip label

Return: Y (with flipped labels)
```

### Key Intuitions

1. **Z-score Normalization**: High Z-score means the sample's prediction for minority class is much higher than the average majority class sample → likely an outlier that should be minority

2. **Tanh Transformation**: Creates smooth transition, only flips when Z > threshold, saturates at ρ=1 for very confused samples

3. **Stochastic Flipping**: Higher ρ → higher chance of flip, but not deterministic (prevents overfitting)

---

## File Structure

```
binary_lnr.py           # Core LNR implementation (BinaryLNR class)
train_binary_lnr.py     # Complete training pipeline for CIFAR
demo_binary_lnr.py      # Simple demo on synthetic data
BINARY_README.md        # This file
```

---

## Quick Start

### 1. Run Demo (Simplest)

See the algorithm in action on a toy dataset:

```bash
python demo_binary_lnr.py
```

This will:
- Create a synthetic imbalanced dataset (10% minority)
- Train a baseline classifier
- Apply LNR algorithm with step-by-step output
- Train a new classifier with flipped labels
- Compare performance
- Create visualizations

**Expected Output**: Visualization showing which samples were flipped and why.

### 2. Train on CIFAR-10 Binary

#### Stage 1: Pre-train Model

```bash
python train_binary_lnr.py \
  --stage 1 \
  --class0 0 --class1 1 \
  --imbalance 0.1 \
  --epochs 100 \
  --save-dir ./saved_binary/exp1
```

This creates an imbalanced binary dataset (e.g., airplane vs automobile) and trains a baseline model.

#### Stage 2: Apply LNR and Fine-tune

```bash
python train_binary_lnr.py \
  --stage 2 \
  --resume ./saved_binary/exp1/stage1_best.pth \
  --threshold 3.0 \
  --epochs 100 \
  --save-dir ./saved_binary/exp1
```

This:
1. Loads pre-trained model
2. Applies LNR to flip labels
3. Fine-tunes classifier with flipped labels
4. Saves noise model and visualizations

---

## Detailed Usage

### BinaryLNR Class

```python
from binary_lnr import BinaryLNR

# Create LNR object
lnr = BinaryLNR(
    model=feature_extractor,      # PyTorch feature extractor
    classifier=classification_head, # PyTorch classifier
    threshold=3.0,                 # t_flip parameter
    n_passes=2,                    # Number of forward passes to average
    device='cuda'
)

# Generate noise model
noise_info = lnr.generate_noise_model(
    dataloader=train_loader,
    save_path='noise_model.pkl'
)

# In training loop, apply flips
for indices, images, targets in train_loader:
    targets = apply_label_flips(targets, indices, noise_info)
    # ... train with flipped targets
```

### Key Parameters

#### threshold (t_flip)
- Controls sensitivity of label flipping
- **Higher** threshold → **fewer** flips (more conservative)
- **Lower** threshold → **more** flips (more aggressive)
- Typical range: 2.0 - 5.0
- Default: 3.0

**How to choose:**
- Start with 3.0
- If minority class still underperforming: decrease to 2.5
- If majority class accuracy drops: increase to 4.0
- Use validation set to tune

#### imbalance_ratio
- Ratio of minority to majority class
- 0.1 = 10% minority (severe imbalance)
- 0.5 = 50% minority (moderate imbalance)

---

## Understanding the Output

### Noise Model Statistics

```
Flipping Results:
  Flipped samples: 94 / 4500
  Flip percentage: 2.09%
  Mean flip rate ρ: 0.0234
  Mean Z-score: -0.8234
  Samples with Z > threshold: 112
  
New class distribution:
  Class 0 (majority): 4406 (was 4500)
  Class 1 (minority): 594 (was 500)
  New imbalance ratio: 0.135 (was 0.111)
```

**What this means:**
- Only 2% of majority samples were flipped (minimal intervention)
- 112 samples had Z > threshold, but only 94 actually flipped (stochastic)
- Imbalance improved from 11.1% to 13.5%
- Most majority samples kept (mean Z-score negative)

### Visualizations

The algorithm generates several plots:

1. **Prediction Distribution**: Shows predictions for kept/flipped/minority samples
2. **Z-score Distribution**: Shows which samples had high Z-scores
3. **Flip Rate Distribution**: Shows computed flip probabilities
4. **Z vs Prediction Scatter**: Shows relationship between prediction and Z-score
5. **Flip Rate Curve**: Shows the tanh(Z - threshold) function
6. **Class Distribution Before/After**: Shows rebalancing effect

---

## Expected Performance Improvements

### Metrics that Improve with LNR:

✅ **Balanced Accuracy**: Average of per-class accuracies (most important)
✅ **Minority Class Recall**: Detection rate for minority class
✅ **F1 Score**: Harmonic mean of precision and recall
✅ **Calibration (ECE)**: Confidence calibration

### Metrics that May Slightly Decrease:

⚠️ **Overall Accuracy**: Flipping introduces some "noise" in majority class
⚠️ **Majority Class Accuracy**: Some clean majority samples are sacrificed

**Trade-off**: We accept small decrease in majority accuracy for large gain in minority accuracy. Net result: better balanced performance.

---

## Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Random Undersampling** | Simple | Discards data, loses information |
| **SMOTE** | Creates synthetic samples | May create unrealistic samples |
| **Cost-Sensitive Learning** | No data modification | Requires careful weight tuning |
| **LNR (Ours)** | Minimal intervention, uses real samples | Requires pre-trained model |

---

## Troubleshooting

### Problem: No samples being flipped

**Cause**: Threshold too high or model too confident

**Solution**:
- Decrease threshold (try 2.0)
- Check if model predictions are well-separated
- Verify σ is not too small

### Problem: Too many samples flipped

**Cause**: Threshold too low

**Solution**:
- Increase threshold (try 4.0 or 5.0)
- Check mean Z-score (should be slightly negative)

### Problem: Performance worse after LNR

**Cause**: Inappropriate threshold or poor pre-trained model

**Solution**:
- Ensure stage 1 model is well-trained
- Try different thresholds
- Verify imbalance ratio is correct
- Check if flipped samples make sense (visualize)

### Problem: Visualizations not showing

**Cause**: Matplotlib not installed

**Solution**:
```bash
pip install matplotlib
```

---

## Advanced Usage

### Custom Dataset

```python
# Your dataset should return (index, features, labels)
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return idx, features[idx], labels[idx]
    
# Create dataloader
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Apply LNR
lnr = BinaryLNR(model, classifier, threshold=3.0)
noise_info = lnr.generate_noise_model(dataloader)
```

### Multiple Iterations

For extreme imbalance, you can apply LNR multiple times:

```python
for iteration in range(3):
    # Generate noise model
    noise_info = lnr.generate_noise_model(dataloader)
    
    # Train with flipped labels
    train_epoch(..., noise_info=noise_info)
    
    # Re-create LNR with updated model
    lnr = BinaryLNR(model, classifier, threshold=3.0)
```

### Threshold Tuning

Use validation set to find optimal threshold:

```python
thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
best_threshold = None
best_score = 0

for threshold in thresholds:
    lnr = BinaryLNR(model, classifier, threshold=threshold)
    noise_info = lnr.generate_noise_model(train_loader)
    
    # Train and evaluate
    score = validate(val_loader, model, classifier, noise_info)
    
    if score > best_score:
        best_score = score
        best_threshold = threshold

print(f"Best threshold: {best_threshold}")
```

---

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{hu2025lnr,
  title={Learning Imbalanced Data with Beneficial Label Noise},
  author={Hu, Guangzheng and Liu, Feng and Gong, Mingming and Wang, Guanghui and Peng, Liuhua},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

---

## Additional Resources

- **Full Repository**: Original multi-class implementation
- **Paper**: [ICML 2025](https://icml.cc/virtual/2025/poster/46163)
- **REPO_ANALYSIS.md**: Detailed analysis of the full codebase

---

## FAQ

**Q: Can I use LNR for multi-class problems?**  
A: Yes! The main repository has full multi-class implementation. Binary version is simplified for clarity.

**Q: Does LNR work with any classifier?**  
A: Yes, as long as it outputs probabilities. Works with neural networks, logistic regression, random forests, etc.

**Q: How is this different from label smoothing?**  
A: Label smoothing changes loss function. LNR actually flips hard labels in the dataset based on model predictions.

**Q: Why not just use class weights?**  
A: Class weights apply uniform reweighting. LNR is instance-specific: only flips samples that "look like" minority class.

**Q: Can I use LNR without pre-training?**  
A: Not recommended. The noise model relies on a reasonably trained classifier to identify which samples to flip.

**Q: What if my dataset is balanced?**  
A: LNR won't help. It's specifically designed for imbalanced scenarios. With threshold=3.0, very few samples would flip in balanced data.

---

## Contact

For questions or issues, please open an issue in the repository or refer to the original paper.

---

**Happy Training! 🚀**
