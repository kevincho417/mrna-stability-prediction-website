# Training Summary

Model: MLP over per-region RNA composition, motif, and k-mer frequency features.
Architecture: input -> 384 -> 128 -> 32 -> 1, ReLU, BatchNorm1d, dropout [0.3, 0.2, 0.1].
Optimizer: AdamW, lr=0.0008, weight_decay=0.0003, batch_size=64.

## 5-fold CV mean metrics

| metric | mean | std |
|---|---:|---:|
| recall | 0.7120 | 0.0374 |
| precision | 0.7144 | 0.0287 |
| specificity | 0.7342 | 0.0336 |
| f1 | 0.7122 | 0.0206 |
| auROC | 0.7936 | 0.0185 |
| auPRC | 0.7804 | 0.0174 |

See `cv_metrics.csv` for per-fold results and `learning_curves.csv` for epoch-level curves.
