# mRNA Stability Prediction — Model Training

Codon-Aware Multi-Branch CNN (`mRNAStabilityNet`) for binary mRNA stability classification.

## Model Architecture

![Model Architecture](figures/keras_multibranch_cnn.png)

The model has **four parallel branches** that are concatenated and fed into a classification head:

| Branch | Input | Encoding | Conv Kernel | Output |
|--------|-------|----------|-------------|--------|
| **5'UTR** | Nucleotide sequence (max 512 nt) | Embedding(6, 16) | k=7, dilation 1/2/4 | 192d |
| **CDS** | Codon sequence (max 700 codons) | Embedding(66, 16) | k=5, dilation 1/2/4 | 192d |
| **3'UTR** | Nucleotide sequence (max 1024 nt) | Embedding(6, 16) | k=7, dilation 1/2/4 | 192d |
| **Features** | 75-dim engineered features | StandardScaler | — | 75d |

Each conv branch uses **3 dilated Conv1D blocks** (with ChannelLayerNorm + ReLU + Dropout)
followed by **triple pooling** (masked max + mean + attention) to produce a 192-d vector.

**Classification head:** Linear(651→128) → LayerNorm → ReLU → Dropout → Linear(128→64) → LayerNorm → ReLU → Dropout → Linear(64→1)

**Total trainable parameters:** 221,508

### Engineered Features (75-dim)

- **11 scalar features:** log-transformed region lengths, GC content (per region), AU-rich element count (AUUUA in 3'UTR), upstream AUG count (5'UTR), 5' CDS ramp GC, UTR-to-CDS length ratios
- **64-dim codon frequency:** CDS codon usage distribution over all 64 codons

## Training

5-fold cross-validation with one model per fold, ensembled via probability averaging.

```bash
python train_multibranch_cnn.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | BCEWithLogitsLoss (class-weighted) |
| Batch size | 32 |
| Max epochs | 40 |
| Early stopping | patience=10, monitored by val auROC |
| Gradient clipping | max_norm=5.0 |
| Dropout | 0.3 |
| Seed | 42 |

### Learning Curves

![Learning Curves](figures/learning_curves.png)

All 5 folds converge stably with early stopping triggered between epochs 15–21.

## Results

### 5-Fold Cross-Validation (mean ± std)

| Metric | Score |
|--------|-------|
| **auROC** | **0.8041 ± 0.0207** |
| **auPRC** | **0.7900 ± 0.0176** |
| Recall | 0.8052 ± 0.0387 |
| Precision | 0.6937 ± 0.0214 |
| Specificity | 0.6679 ± 0.0368 |
| F1 | 0.7445 ± 0.0193 |
| Accuracy | 0.7339 ± 0.0185 |

### Per-Fold auROC

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|--------|--------|--------|--------|--------|
| 0.8115 | 0.7763 | 0.7846 | 0.8320 | 0.8163 |

**Grading check:** CV auROC 0.8041 ≥ 0.75 minimum threshold ✓

### OOF Overall

| Metric | Score |
|--------|-------|
| auROC | 0.7970 |
| auPRC | 0.7815 |
| F1 | 0.7317 |
| Threshold | 0.329 |

## Key Design Decisions

1. **Codon-level CDS encoding:** The CDS branch tokenizes at the codon level (66 tokens) instead of single nucleotides, directly capturing codon usage bias — a primary determinant of mRNA stability.
2. **Dilated convolutions:** Dilation rates of 1→2→4 expand the receptive field without increasing parameter count, allowing the model to capture long-range sequence patterns.
3. **ChannelLayerNorm:** Normalizes over channels rather than batch statistics, avoiding distortion from padded positions — critical when sequences vary greatly in length.
4. **Triple pooling:** Combining max, mean, and attention pooling retains complementary information from the sequence representation.
5. **Lightweight design:** ~221K parameters to reduce overfitting on the ~3,000-sample dataset.

## File Structure

```
Training/
├── multibranch_cnn_model.py      # Model + dataset + feature engineering
├── train_multibranch_cnn.py      # 5-fold CV training script
├── keras_multibranch_cnn.py      # Keras equivalent (for architecture diagram)
├── figures/
│   ├── keras_multibranch_cnn.png # Architecture diagram (plot_model)
│   ├── keras_multibranch_cnn.svg
│   ├── keras_multibranch_cnn.pdf
│   ├── learning_curves.png       # 5-fold CV learning curves
│   ├── multibranch_cnn_summary.png
│   └── multibranch_cnn_summary.txt
└── outputs_multibranch_cnn/
    ├── fold{1-5}_best.pt         # Trained checkpoints
    ├── cv_metrics.json           # Full metric suite
    ├── oof_predictions.csv       # Out-of-fold predictions
    ├── test_predictions.csv      # Ensembled test predictions
    └── learning_curves.png       # Learning curve plot
```

## Inference

```python
from multibranch_cnn_model import mRNAStabilityNet, mRNADataset, load_folds, load_test, DEFAULT_MAXLEN
import torch, glob, numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load all fold checkpoints and average predictions
ckpts = sorted(glob.glob("outputs_multibranch_cnn/fold*_best.pt"))
# ... (see train_multibranch_cnn.py for full inference loop)
```
