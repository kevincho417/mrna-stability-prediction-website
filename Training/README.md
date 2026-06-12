# mRNA Stability Model Training

This folder contains the training pipeline for the final project network training
task. The model is a **codon-aware multi-branch CNN** (PyTorch) trained with
5-fold cross-validation and served as a 5-model ensemble.

`Label 0 = highly degraded (unstable)`, `Label 1 = lowly degraded (stable)`.
The graded metric is **auROC**.

## Run

5-fold CV training + test predictions (ensemble of the 5 fold models):

```powershell
python .\Training\train.py --data_dir .\Dataset --out_dir .\Training\outputs --epochs 40 --batch_size 32
```

Stand-alone inference later from the saved checkpoints:

```powershell
python .\Training\predict.py --ckpt_dir .\Training\outputs --data_dir .\Dataset `
    --input .\Dataset\test\test_without_label.csv `
    --output .\Training\outputs\test_predictions.csv
```

The script uses the five files in `Dataset/training` as predefined CV folds: for
each held-out fold it trains on the other four and validates on that fold. The
five fold models are ensembled (mean probability) for the test prediction. It
writes the following to `Training/outputs`:

- `fold{k}_best.pt`: best checkpoint per fold (selected by validation auROC).
- `cv_metrics.json`: full metric suite per fold + mean/std + pooled OOF + the
  validation-tuned `global_threshold`.
- `learning_curves.csv` / `learning_curves.png`: the 5-fold CV learning curves.
- `oof_predictions.csv`: out-of-fold predictions for every training row.
- `test_predictions.csv`: `TranscriptID, prob, Label` for the test set (keep
  `prob` for auROC scoring).
- `feature_stats.json`: engineered-feature normalization (mean/std over the
  training rows) so inference can normalize without re-reading the dataset.

## Model

A multi-branch network over the three transcript regions, concatenated with
engineered features and passed to a 2-layer MLP head:

- **5'UTR / 3'UTR**: nucleotide-level tokens (`A C G U N` + pad), embedded.
- **CDS**: tokenized as **codons** (non-overlapping triplets, 64-codon vocab),
  which exposes *codon optimality*, a dominant determinant of mRNA stability
  that nucleotide-level models miss.
- Each branch is a stacked dilated `Conv1d` tower (`ChannelLayerNorm` + ReLU +
  Dropout) with masked global **max + mean + attention** pooling. Empty
  5'UTR / 3'UTR pool to exactly 0; the region is still seen through the
  engineered features. LayerNorm (not BatchNorm) keeps padded timesteps and
  batch composition from distorting normalization.
- Lightweight by design (embedding 16, channels 32/64/64, head 128, ~220K
  params) to limit overfitting on the ~3k-sample dataset.
- Pooled vectors are concatenated with **75 engineered features** (log-lengths,
  GC per region, AU-rich-element & uAUG counts, 5'-CDS GC ramp, length ratios,
  and a 64-dim codon-frequency vector) before the MLP head.
- Loss: `BCEWithLogitsLoss` with `pos_weight`; AdamW + cosine LR; early stopping
  and model selection on **validation auROC**.

Truncation caps (in `data.py`): 5'UTR 512 nt, 3'UTR 1024 nt, CDS 700 codons
(2100 nt). The architecture diagram is in `full_model_lite.png` / `.svg`.

## 5-fold CV results

| Metric | 5-fold CV mean ± std |
|---|---|
| auROC | **0.8032 ± 0.0230** |
| auPRC | 0.7893 ± 0.0170 |
| Recall | 0.8264 ± 0.0386 |
| Precision | 0.6858 ± 0.0446 |
| Specificity | 0.6377 ± 0.1017 |
| F1 | 0.7477 ± 0.0233 |
| Accuracy | 0.7297 ± 0.0396 |

Per-fold auROC: 0.814, 0.772, 0.782, 0.834, 0.815. Mean CV auROC **0.803**
clears the 0.75 minimum. The validation-tuned decision threshold (`global_threshold`
≈ 0.366) is used only for the threshold-dependent metrics; auROC and auPRC are
threshold-free. Full numbers are in `outputs/cv_metrics.json`.

## Files

| File | Purpose |
|---|---|
| `model.py` | network definition (`mRNAStabilityNet`) |
| `data.py` | tokenization, feature engineering, dataset, fold loading |
| `train.py` | 5-fold CV training + 5-model test ensemble |
| `predict.py` | stand-alone inference from saved checkpoints |
| `gen_v2.py` | renders the architecture diagram |
| `requirements.txt` | dependencies |
| `literature_review.md` | related-work summary |
| `outputs/` | checkpoints, metrics, predictions, learning curves, feature stats |
