# Quick Reference: Binary LNR Algorithm

## Algorithm at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│  Binary Label Noise Rebalancing (LNR)                       │
│  For Imbalanced Binary Classification                        │
└─────────────────────────────────────────────────────────────┘

INPUT:
  X: Features
  Y: Labels (0=majority, 1=minority)
  Cf: Pre-trained classifier
  t_flip: Threshold (default=3.0)

ALGORITHM:
  1. Get majority indices: IndMA ← {i : Y[i] = 0}
  
  2. Get predictions: η̂[i] ← P(Y=1|X[i]) for all i
  
  3. Compute statistics on majority class:
     μ ← mean(η̂[IndMA])
     σ ← std(η̂[IndMA])
  
  4. For each iMA in IndMA:
     a) Z[iMA] ← (η̂[iMA] - μ) / σ         # Z-score
     b) ρ[iMA] ← max(tanh(Z - t_flip), 0)  # Flip rate
     c) U ~ Bernoulli(ρ[iMA])              # Sample
     d) if U = 1: Y[iMA] ← 1               # Flip

OUTPUT:
  Y: Modified labels (some 0→1 flips)
```

---

## Files Quick Reference

| File | Purpose | Run Command |
|------|---------|-------------|
| `demo_binary_lnr.py` | See algorithm in action | `python demo_binary_lnr.py` |
| `train_binary_lnr.py` | Train on CIFAR | `python train_binary_lnr.py --stage 1 ...` |
| `binary_lnr.py` | Core implementation | Import as module |
| `BINARY_README.md` | Complete documentation | Read for details |
| `REPO_ANALYSIS.md` | Understand repo | Read for architecture |

---

## Quick Start Commands

### Run Demo (No Setup Needed)
```bash
python demo_binary_lnr.py
```

### Train Stage 1 (Pre-train)
```bash
python train_binary_lnr.py \
  --stage 1 \
  --class0 0 --class1 1 \
  --imbalance 0.1 \
  --epochs 100 \
  --save-dir ./saved_binary/exp1
```

### Train Stage 2 (Apply LNR)
```bash
python train_binary_lnr.py \
  --stage 2 \
  --resume ./saved_binary/exp1/stage1_best.pth \
  --threshold 3.0 \
  --epochs 100 \
  --save-dir ./saved_binary/exp1
```

---

## Parameter Guide

### threshold (t_flip)

```
┌─────────────┬──────────────┬─────────────┐
│ Value       │ Flip Amount  │ Use Case    │
├─────────────┼──────────────┼─────────────┤
│ 1.5 - 2.5   │ Aggressive   │ Very severe │
│             │ (3-5% flips) │ imbalance   │
├─────────────┼──────────────┼─────────────┤
│ 2.5 - 3.5   │ Moderate     │ Typical     │
│             │ (1-3% flips) │ imbalance   │
├─────────────┼──────────────┼─────────────┤
│ 4.0 - 5.0   │ Conservative │ Mild        │
│             │ (<1% flips)  │ imbalance   │
└─────────────┴──────────────┴─────────────┘
```

**Default: 3.0** (good starting point)

---

## Understanding Z-scores

```
Z = (prediction - mean) / std

┌────────────────────────────────────────────────┐
│ Z-score Interpretation                          │
├────────────────────────────────────────────────┤
│ Z < 0     : Below average (typical majority)   │
│ Z = 0     : Average majority sample            │
│ 0 < Z < 3 : Above average (confused sample)    │
│ Z > 3     : High outlier (flip candidate!)     │
└────────────────────────────────────────────────┘

Flip Rate: ρ = max(tanh(Z - t_flip), 0)

┌────────────────────────────────────────────────┐
│ If t_flip = 3.0:                               │
│   Z = 2.0 → ρ = 0.00 (no flip)                 │
│   Z = 3.0 → ρ = 0.00 (threshold)               │
│   Z = 4.0 → ρ = 0.76 (likely flip)             │
│   Z = 5.0 → ρ = 0.96 (very likely flip)        │
│   Z = 10  → ρ ≈ 1.0  (almost certain flip)     │
└────────────────────────────────────────────────┘
```

---

## Expected Performance

```
Example: 10% Minority Class (Severe Imbalance)

┌─────────────────┬──────────┬──────────┬─────────┐
│ Metric          │ Before   │ After    │ Change  │
├─────────────────┼──────────┼──────────┼─────────┤
│ Overall Acc     │ 92%      │ 90%      │ -2%     │
│ Majority Acc    │ 99%      │ 95%      │ -4%     │
│ Minority Acc    │ 40%      │ 85%      │ +45%    │
│ Balanced Acc    │ 70%      │ 90%      │ +20%    │
│ F1 Score        │ 0.55     │ 0.88     │ +0.33   │
└─────────────────┴──────────┴──────────┴─────────┘

Trade-off: Small sacrifice in majority for large
           gain in minority → Better balance!
```

---

## Code Snippet - Basic Usage

```python
from binary_lnr import BinaryLNR, apply_label_flips

# 1. Create LNR object
lnr = BinaryLNR(
    model=feature_extractor,
    classifier=classification_head,
    threshold=3.0,
    device='cuda'
)

# 2. Generate noise model (once at start)
noise_info = lnr.generate_noise_model(
    dataloader=train_loader,
    save_path='noise_model.pkl'
)

# 3. Training loop
for epoch in range(epochs):
    for indices, images, targets in train_loader:
        # Apply label flips
        targets = apply_label_flips(
            targets, indices, noise_info
        )
        
        # Train with flipped labels
        loss = train_step(images, targets)
```

---

## Troubleshooting

```
┌──────────────────────┬────────────────────────┐
│ Problem              │ Solution               │
├──────────────────────┼────────────────────────┤
│ No flips happening   │ • Lower threshold      │
│                      │ • Check predictions    │
├──────────────────────┼────────────────────────┤
│ Too many flips       │ • Raise threshold      │
│                      │ • Verify imbalance     │
├──────────────────────┼────────────────────────┤
│ Worse performance    │ • Check stage 1 model  │
│                      │ • Tune threshold       │
│                      │ • Try val set tuning   │
├──────────────────────┼────────────────────────┤
│ Import errors        │ • Check requirements   │
│                      │ • Verify file paths    │
└──────────────────────┴────────────────────────┘
```

---

## Mathematical Formulas

### Z-score Normalization
```
Z = (η̂ - μ) / σ

where:
  η̂ = prediction for current sample
  μ = mean prediction on majority class
  σ = std prediction on majority class
```

### Flip Rate Function
```
ρ = max(tanh(Z - t_flip), 0)

Properties:
  • ρ ∈ [0, 1]
  • ρ = 0 when Z ≤ t_flip
  • ρ → 1 as Z → ∞
  • Smooth transition at threshold
```

### Bernoulli Sampling
```
U ~ Bernoulli(ρ)

P(U = 1) = ρ
P(U = 0) = 1 - ρ

Flip label if U = 1
```

---

## Visualization Legend

```
When you see plots:

Colors:
  🔵 Blue   = Majority samples (kept)
  🔴 Red    = Majority samples (flipped)
  🟢 Green  = Minority samples (original)
  ⚫ Black  = Thresholds/means

Lines:
  ━━━ Solid     = Function/curve
  ╌╌╌ Dashed    = Threshold/reference
```

---

## Key Insights

```
💡 High Z-score = Outlier in majority class
   → Sample's prediction much higher than average
   → Likely similar to minority class
   → Good candidate for flipping

💡 Tanh function = Smooth transition
   → Not a hard cutoff
   → Gradual increase in flip probability
   → Mathematically elegant

💡 Stochastic flipping = Regularization
   → Not all high-Z samples flipped
   → Adds randomness
   → Different flips each run

💡 Minimal intervention = Data efficiency
   → Only ~2% of samples flipped
   → Preserves most original labels
   → Uses real samples (not synthetic)
```

---

## Workflow Diagram

```
┌──────────────┐
│ Imbalanced   │
│   Dataset    │
│ (90% / 10%)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Stage 1:    │
│  Pre-train   │ ← Standard training
│   Classifier │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Apply LNR    │
│ Algorithm 1  │ ← Compute flips
│ (noise model)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Stage 2:    │
│  Fine-tune   │ ← Train with flips
│ with Flipped │
│    Labels    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Improved    │
│   Balanced   │ ← Better minority
│  Classifier  │   class accuracy
└──────────────┘
```

---

## Requirements

```bash
# Core
torch
torchvision
numpy

# Optional (for demo/visualization)
matplotlib
scikit-learn
```

Install:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## Citation

```bibtex
@inproceedings{hu2025lnr,
  title={Learning Imbalanced Data with Beneficial Label Noise},
  author={Hu, Guangzheng and Liu, Feng and Gong, Mingming and 
          Wang, Guanghui and Peng, Liuhua},
  booktitle={ICML},
  year={2025}
}
```

---

## Quick Help

```
For help, refer to:

📖 Full docs:      BINARY_README.md
🔍 Repo analysis:  REPO_ANALYSIS.md
🎯 This guide:     QUICK_REFERENCE.md
💻 Demo code:      demo_binary_lnr.py
🚀 Train code:     train_binary_lnr.py
⚙️  Core code:      binary_lnr.py
```

---

**Remember:** Start with `python demo_binary_lnr.py` to understand the algorithm!
