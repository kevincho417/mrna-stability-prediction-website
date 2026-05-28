# Transformer Training Summary

Model: end-to-end Conv1D + Transformer Encoder over AUCG token sequences.
Tokenization: PAD=0, A=1, U=2, C=3, G=4, unknown=5.
Architecture: embedding dim 96, 3 Conv1D downsampling layers, 2 Transformer Encoder layers, 4 heads.
Optimizer: AdamW, lr=0.0005, weight_decay=0.0001, batch_size=16.

## 5-fold CV mean metrics

| metric | mean | std |
|---|---:|---:|
| recall | 0.5960 | 0.1093 |
| precision | 0.7583 | 0.0639 |
| specificity | 0.8058 | 0.1114 |
| f1 | 0.6560 | 0.0391 |
| auROC | 0.7927 | 0.0136 |
| auPRC | 0.7657 | 0.0142 |

See `cv_metrics.csv` for per-fold results and `learning_curves.csv` for epoch-level curves.
