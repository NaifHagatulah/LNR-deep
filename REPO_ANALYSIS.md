# LNR-Deep Repository Analysis

## Overview
This repository implements **Label Noise Rebalancing (LNR)**, a novel approach to handle imbalanced datasets by strategically flipping labels of majority class samples to minority classes. It's built on top of MiSLAS (Mixed Supervised Learning with Auxiliary Signals).

## Core Concept
**Key Innovation**: Instead of traditional data augmentation or loss reweighting, LNR identifies majority class samples that are feature-wise similar to minority classes and flips their labels. This "beneficial label noise" helps rebalance the dataset.

---

## Architecture Overview

### Two-Stage Training Pipeline

#### **Stage 1: Feature Learning** (`train_stage1.py`)
- **Purpose**: Train a base model to learn robust feature representations
- **Method**: Standard training with mixup augmentation
- **Output**: Pre-trained feature extractor + classifier
- **Key Components**:
  - Model: ResNet backbone (ResNet32 for CIFAR, ResNet50/152 for ImageNet/Places)
  - Loss: Standard Cross-Entropy with optional Mixup
  - Training: Instance-based sampling (all samples)

#### **Stage 2: LNR Fine-tuning** (`train_stage2_lnr.py`)
- **Purpose**: Apply label noise rebalancing and fine-tune classifier
- **Method**: Flip labels based on noise model, then train with Label-Aware Smoothing
- **Input**: Pre-trained model from Stage 1
- **Key Innovation**: `label_noise_rebalance()` function

---

## Core Algorithm: Label Noise Rebalancing

### Location: `train_stage2_lnr.py` - Function `label_noise_rebalance()`

### Multi-Class Algorithm Flow:

```
For each class k in dataset:
  1. Compute predictions for all samples NOT in class k
  2. Calculate mean and std of prediction probability for class k
     across non-k samples
  
For each sample (x, c) where c is current label:
  1. Compute Z-scores for all classes:
     z_score[k] = (pred_prob[k] - mean[k]) / std[k]
  
  2. Compute prior weights (inverse of class frequencies):
     prior_weight[k] = max(prior[k] - prior[c], 0)
  
  3. Compute flip rate for each class:
     flip_rate[k] = tanh(z_score[k] - threshold) * prior_weight[k]
  
  4. Sample flip decision:
     uniform_rand ~ Uniform(0, 1) for each class k
     noise_classes = {k : flip_rate[k] > uniform_rand[k]}
  
  5. If noise_classes is not empty:
     - Select class with highest flip_rate
     - Flip label from c to selected class
```

### Key Parameters:
- **Threshold (`thre`)**: Controls sensitivity of flipping
  - CIFAR-10: 7.5
  - CIFAR-100: 14.5
- **Prior Weighting**: Prevents flipping to already over-represented classes
- **Stochastic Flipping**: Bernoulli sampling based on flip_rate

---

## Binary Classification Algorithm (Your Provided)

### Algorithm Breakdown:

```python
Input: 
  - X: Features
  - Y: Labels (0 or 1)
  - Cf: Classifier
  - t_flip: Threshold

# Step 1: Get predictions for all samples
IndMA ← indices where Y = 0  # Majority class indices
η̂[i] ← Cf(X[i]) for all i   # Predictions (probability of class 1)

# Step 2: Calculate statistics for majority class
μ ← mean(η̂[IndMA])          # Mean prediction on majority class
σ ← std(η̂[IndMA])           # Std prediction on majority class

# Step 3: For each majority class sample
For each iMA in IndMA:
  # Compute Z-score
  Z[iMA] ← (η̂[iMA] - μ) / σ
  
  # Compute flip probability (noise rate)
  ρ[iMA] ← max(tanh(Z[iMA] - t_flip), 0)
  
  # Sample flip decision
  U ~ Bernoulli(ρ[iMA])
  
  # Flip label if sampled
  if U == 1:
    Y[iMA] ← 1  # Flip from 0 to 1

Return: Y (modified labels)
```

### Mathematical Intuition:

1. **Z-score Normalization**: 
   - High Z-score → Sample prediction is much higher than average majority class
   - Indicates sample is "confused" and might belong to minority class

2. **Tanh Transformation**: 
   - `tanh(Z - t_flip)` creates smooth transition
   - Only positive values → flips only when Z > t_flip
   - Saturates at 1 for very high Z-scores

3. **Stochastic Flipping**:
   - Higher ρ → Higher chance of flip
   - Adds randomness to prevent overfitting

---

## Code Flow for Binary Classification

### Connecting to Repository Structure:

```
1. Stage 1: Pre-train model
   ├─ train_stage1.py
   ├─ Uses standard CE loss
   └─ Outputs: checkpoint with model + classifier

2. Stage 2: Apply LNR
   ├─ Load pre-trained model
   ├─ For epoch 0:
   │  └─ Call label_noise_rebalance(store=True)
   │     ├─ Run inference on all training data (2 passes)
   │     ├─ Average predictions
   │     ├─ Compute Z-scores and flip rates
   │     └─ Store noise_info to pickle file
   │
   ├─ For each epoch:
   │  ├─ Load noise_info (read=True)
   │  ├─ In training loop:
   │  │  └─ For each batch:
   │  │     └─ Flip labels based on noise_info
   │  └─ Train with flipped labels
   │
   └─ Use Label-Aware Smoothing loss
```

---

## Key Files and Their Roles

### Training Files:
- **`train_stage1.py`**: Pre-training with mixup
- **`train_stage2_lnr.py`**: LNR fine-tuning (main algorithm)
- **`train_stage2_selmix.py`**: Alternative method (SelMix)

### Method Files:
- **`methods.py`**: 
  - Mixup augmentation
  - Label-Aware Smoothing loss
  - Learnable Weight Scaling

### Dataset Files:
- **`datasets/cifar10.py`**: CIFAR-10 with imbalance
- **`datasets/cifar100.py`**: CIFAR-100 with imbalance
- Creates imbalanced datasets with exponential decay

### Model Files:
- **`models/resnet_cifar.py`**: ResNet for CIFAR
- **`models/resnet.py`**: ResNet for ImageNet
- Separates feature extractor and classifier

### Config Files:
- **`config/*/`**: YAML configs for experiments
- Define: dataset, imbalance ratio, hyperparameters

---

## How to Adapt for Binary Classification

### Simplifications from Multi-Class:

1. **No Prior Weighting**: In binary case, only flip 0→1, so no need for class selection
2. **Single Z-score**: Only compute Z-score for minority class (class 1)
3. **No Argmax**: Directly flip to class 1 if condition met

### Implementation Strategy:

```python
def label_noise_rebalance_binary(
    train_dataloader, 
    model, 
    classifier, 
    threshold=3.0
):
    """
    Binary classification version of LNR
    
    Args:
        train_dataloader: DataLoader with (index, x, y)
        model: Feature extractor
        classifier: Classification head
        threshold: t_flip parameter
    
    Returns:
        noise_info: Dict with samples to flip
    """
    
    # Step 1: Collect predictions over 2 epochs
    predictions = {}
    for epoch in range(2):
        for index, x, target in train_dataloader:
            x = x.cuda()
            feat = model(x)
            out = classifier(feat)
            prob = F.softmax(out, dim=1)[:, 1]  # P(Y=1)
            
            for i, idx in enumerate(index):
                if str(idx) not in predictions:
                    predictions[str(idx)] = []
                predictions[str(idx)].append((
                    prob[i].item(), 
                    target[i].item()
                ))
    
    # Step 2: Average predictions
    avg_predictions = {}
    for idx, preds in predictions.items():
        avg_pred = np.mean([p[0] for p in preds])
        label = preds[0][1]
        avg_predictions[idx] = (avg_pred, label)
    
    # Step 3: Get majority class (label=0) statistics
    majority_preds = [
        pred for pred, label in avg_predictions.values() 
        if label == 0
    ]
    mu = np.mean(majority_preds)
    sigma = np.std(majority_preds)
    
    # Step 4: Compute flip decisions
    noise_flag = {}
    flip_count = 0
    
    for idx, (pred, label) in avg_predictions.items():
        if label == 0:  # Only consider majority class
            # Compute Z-score
            z_score = (pred - mu) / sigma
            
            # Compute flip rate
            flip_rate = np.tanh(z_score - threshold)
            flip_rate = max(flip_rate, 0)
            
            # Sample flip decision
            if np.random.rand() < flip_rate:
                noise_flag[idx] = 1  # Flip to minority
                flip_count += 1
    
    print(f"Flipped {flip_count} samples from majority to minority")
    
    return {'noise_flag': noise_flag}
```

---

## Dataset Structure

### Imbalanced CIFAR Creation:
```python
# From datasets/cifar10.py
class IMBALANCECIFAR10:
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = total_images / cls_num
        for cls_idx in range(cls_num):
            # Exponential decay
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1)))
            img_num_per_cls.append(int(num))
```

**Example** (CIFAR-10, imb_factor=0.01):
- Class 0: 5000 samples (majority)
- Class 1: ~4217 samples
- Class 2: ~3548 samples
- ...
- Class 9: 50 samples (minority)

---

## Training Configuration

### Typical Stage 2 Config (CIFAR-10):
```yaml
dataset: 'cifar10'
imb_factor: 0.01
num_classes: 10
backbone: 'resnet32_fe'
batch_size: 128
lr: 0.01
num_epochs: 100
smooth_head: 0.3  # Label smoothing for head classes
smooth_tail: 0.0  # No smoothing for tail classes
```

### Threshold Selection:
- **CIFAR-10**: threshold = 7.5
- **CIFAR-100**: threshold = 14.5
- Generally: Higher imbalance → Higher threshold

---

## Loss Functions

### Label-Aware Smoothing:
- Different smoothing factors for different classes
- More smoothing for head (majority) classes
- Less/no smoothing for tail (minority) classes
- Prevents overconfidence on majority classes

### Formula:
```python
confidence = 1 - smoothing[target_class]
loss = confidence * CE_loss + smoothing * uniform_loss
```

---

## Evaluation Metrics

### Per-Class Accuracy:
- **Many-shot**: Classes with most samples
- **Medium-shot**: Classes with medium samples  
- **Few-shot**: Classes with fewest samples

### Calibration:
- **ECE (Expected Calibration Error)**: Measures confidence calibration
- Lower ECE = better calibrated predictions

---

## How LNR Differs from Traditional Methods

### Traditional Approaches:
1. **Re-sampling**: Oversample minority, undersample majority
   - Loss: Discards majority data
   
2. **Re-weighting**: Higher loss weight for minority
   - Issue: Can cause overfitting on minority
   
3. **Mixup/Augmentation**: Synthesize minority samples
   - Issue: May not preserve semantic meaning

### LNR Approach:
- **Identifies** majority samples that already "look like" minority
- **Flips** their labels (controlled by threshold)
- **Enriches** minority class with hard examples
- **Preserves** all original data (no discarding)
- **Minimal** intervention (only ~100 flips on CIFAR-10)

---

## Next Steps for Binary Classification

1. **Create simplified training script** for binary classification
2. **Implement binary LNR function** as shown above
3. **Test on binary imbalanced dataset** (e.g., CIFAR-10 reduced to 2 classes)
4. **Tune threshold** parameter for binary case
5. **Compare** with baseline methods (no flipping, random flipping, etc.)

---

## Questions to Explore

1. How does threshold affect flip rate in binary case?
2. What's optimal threshold for different imbalance ratios?
3. How does LNR compare to SMOTE/ADASYN in binary case?
4. Can we learn threshold automatically?
5. Does multiple iterations of flipping help?

