# mRNA Stability Prediction

Deep Learning & Big Data 2026 final project for mRNA stability / degradation prediction.

The system predicts whether an mRNA is highly degraded or relatively stable from transcript sequence regions:

- `5'UTR sequence`
- `CDS sequence`
- `3'UTR sequence`

Labels:

- `0`: highly degraded mRNA
- `1`: lowly degraded / stable mRNA

## Final Model

The final model is an ensemble of two models:

| Component | Description |
|---|---|
| Feature MLP | Uses RNA length, base composition, GC ratio, motif frequency, and 1-mer to 4-mer frequency features |
| ConvTransformer | End-to-end AUCG token sequence model with 1D convolutional downsampling and Transformer Encoder layers |

Final ensemble:

```text
prediction = 0.67 * MLP_probability + 0.33 * ConvTransformer_probability
threshold = 0.5
```

## 5-fold CV Results

| Model | Recall | Precision | Specificity | F1 | auROC | auPRC |
|---|---:|---:|---:|---:|---:|---:|
| Feature MLP | 0.7120 | 0.7144 | 0.7342 | 0.7122 | 0.7936 | 0.7804 |
| ConvTransformer 8192 | 0.5960 | 0.7583 | 0.8058 | 0.6560 | 0.7927 | 0.7657 |
| Final ensemble | 0.6985 | 0.7389 | 0.7658 | 0.7162 | 0.8041 | 0.7877 |

Detailed output files:

- `Training/outputs/report_summary.md`
- `Training/transformer_outputs_8192/report_summary.md`
- `Training/ensemble_outputs_base8192/report_summary.md`

## Repository Structure

```text
Training/
  train.py                          feature MLP training
  train_transformer.py              ConvTransformer training
  ensemble_predictions.py           ensemble prediction builder
  literature_review.md              related work summary
  outputs/                          final MLP outputs and checkpoints
  transformer_outputs_8192/         final ConvTransformer outputs and checkpoints
  ensemble_outputs_base8192/        final ensemble predictions and metrics
Server/
  inference.py                      model loading and prediction wrapper
  prediction_socket_server.py       Python TCP socket inference server
  codeigniter_overlay/              CodeIgniter MVC files for VM deployment
  templates/ static/                local Flask demo UI
VM/
  setup_ubuntu.sh                   Ubuntu dependency setup
  setup_apache.sh                   Apache VirtualHost on port 17888
  setup_socket_service.sh           systemd service for model socket
  run_codeigniter.sh                restart Apache server
  run_socket.sh                     manual socket server mode
Web/
  0529_frontend_submission.html     standalone front-end milestone HTML
```

## Dataset

The dataset is not included in this public repository. Place the course dataset locally with this structure before training:

```text
Dataset/
  training/
    training_fold_1.csv
    training_fold_2.csv
    training_fold_3.csv
    training_fold_4.csv
    training_fold_5.csv
  test/
    test_without_label.csv
```

Expected columns:

```text
TranscriptID, 5UTRseq, CDSseq, 3UTRseq, Label
```

## Python Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Models

Feature MLP:

```bash
python Training/train.py
```

ConvTransformer:

```bash
python Training/train_transformer.py --max-len 8192 --d-model 96 --conv-layers 3 --transformer-layers 2 --n-heads 4 --ff-dim 192 --epochs 30 --patience 6 --batch-size 16 --output-dir Training/transformer_outputs_8192
```

Ensemble:

```bash
python Training/ensemble_predictions.py --transformer-dir Training/transformer_outputs_8192 --output-dir Training/ensemble_outputs_base8192
```

## Local Flask Demo

```bash
python Server/run_server.py
```

Open:

```text
http://127.0.0.1:17888/2026Project/
```

## Formal VM Deployment

The formal project server follows `Project.pdf`:

- Ubuntu 22.04 VM
- Apache2 on port `17888`
- CodeIgniter 4 / PHP backend
- Python prediction socket on `127.0.0.1:16888`
- URL: `http://localhost:17888/2026Project/`

Inside the Ubuntu VM:

```bash
chmod +x VM/*.sh
./VM/setup_ubuntu.sh
./VM/setup_apache.sh
./VM/setup_socket_service.sh
```

Open:

```text
http://localhost:17888/2026Project/
```

## Front-end Milestone

For the 2026-05-29 front-end submission:

```text
Web/G4_Project Ckt2 .html
```

This standalone page uses Bootstrap 5.3.3 and Bootstrap Icons through CDN links and partitions the page into header, main body, and footer.

## Notes

The course dataset is intentionally excluded from Git.
