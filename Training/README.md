# mRNA Stability Model Training

This folder contains the training pipeline for the final project network training task.

## Run

Feature-based MLP baseline:

```powershell
python .\Training\train.py
```

End-to-end Conv1D + Transformer sequence model:

```powershell
python .\Training\train_transformer.py
```

End-to-end Conv1D + Transformer with overlapping 3-mer tokens:

```powershell
python .\Training\train_transformer.py --kmer-size 3 --output-dir .\Training\transformer_kmer3_outputs
```

Ensemble the feature MLP and a Transformer run:

```powershell
python .\Training\ensemble_predictions.py --transformer-dir .\Training\transformer_kmer3_outputs
```

The script uses the five files in `Dataset/training` as predefined CV folds, trains one PyTorch MLP per held-out fold, and writes the following files to `Training/outputs`:

- `cv_metrics.csv`: per-fold recall, precision, specificity, F1, auROC, and auPRC.
- `learning_curves.csv`: epoch-level train/validation loss, auROC, and auPRC.
- `learning_curves.png`: learning curve plot for the report.
- `oof_predictions.csv`: out-of-fold predictions on the training rows.
- `test_predictions.csv`: 5-fold ensemble predictions for `Dataset/test/test_without_label.csv`.
- `models/fold_*.pt`: trained fold checkpoints with scaler parameters.
- `summary.json` and `report_summary.md`: report-ready model and metric summary.

## Model

The network is an MLP over per-region RNA features. For each of 5'UTR, CDS, and 3'UTR it uses sequence length, base composition, GC content, homopolymer run ratios, selected motif frequencies, and normalized 1-mer through 4-mer frequencies. Missing UTR values are treated as empty sequences.

`train_transformer.py` is the end-to-end version. It maps RNA bases as `A=1, U=2, C=3, G=4` with `0` reserved for padding, embeds the token sequence, applies 1D convolutional downsampling, then sends the learned sequence representation into Transformer Encoder layers. This version does not use engineered k-mer or composition features.

With `--kmer-size 3`, the Transformer uses overlapping sequence tokens such as `AUG`, `UGG`, and `GGC` instead of single-base tokens. These are learned token embeddings, not precomputed k-mer frequency features.
