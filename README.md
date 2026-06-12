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

The final model is a **codon-aware multi-branch CNN** (PyTorch), trained with
5-fold cross-validation and served as a 5-model ensemble (mean probability).

| Branch | Input | Description |
|---|---|---|
| 5'UTR / 3'UTR | nucleotide tokens (`A C G U N`) | dilated Conv1d tower + masked max/mean/attention pooling |
| CDS | codon tokens (64-codon vocab) | codon-level Conv1d tower capturing codon optimality |
| Engineered features | 75-dim vector | log-lengths, per-region GC, AU-rich-element & uAUG counts, 5'-CDS GC ramp, length ratios, 64-dim codon frequency |

The three pooled branch vectors and the engineered features are concatenated and
passed to a 2-layer MLP head. The model is lightweight (~220K params) to limit
overfitting on the ~3k-sample dataset. Loss is `BCEWithLogitsLoss` with
`pos_weight`; model selection and early stopping monitor validation auROC.

```text
prediction = mean(fold_1 .. fold_5 sigmoid probabilities)
threshold  = 0.366   # validation-tuned (cv_metrics.json global_threshold)
```

## 5-fold CV Results

| Metric | Recall | Precision | Specificity | F1 | auROC | auPRC |
|---|---:|---:|---:|---:|---:|---:|
| 5-fold CV mean | 0.8264 | 0.6858 | 0.6377 | 0.7477 | **0.8032** | 0.7893 |
| Pooled OOF | 0.8052 | 0.6846 | 0.6552 | 0.7400 | 0.7999 | 0.7821 |

Per-fold auROC: 0.814, 0.772, 0.782, 0.834, 0.815. Mean CV auROC **0.803**
clears the 0.75 minimum. auROC / auPRC are threshold-free; the threshold above is
used only for the threshold-dependent metrics. Detailed numbers:

- `Training/outputs/cv_metrics.json`
- `Training/outputs/learning_curves.png`

## Repository Structure

```text
Training/
  model.py                          codon-aware multi-branch CNN definition
  data.py                           tokenization, feature engineering, fold loading
  train.py                          5-fold CV training + 5-model test ensemble
  predict.py                        stand-alone inference from checkpoints
  gen_v2.py                         architecture diagram renderer
  literature_review.md              related work summary
  full_model_lite.png / .svg        architecture diagram
  outputs/                          fold checkpoints, metrics, predictions, curves, feature stats
Server/
  inference.py                      model loading and prediction wrapper
  prediction_socket_server.py       Python TCP socket inference server
  codeigniter_overlay/              CodeIgniter Controller, Model, Views for VM deployment
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

5-fold CV training + test predictions (ensemble of the 5 fold models):

```bash
python Training/train.py --data_dir Dataset --out_dir Training/outputs --epochs 40 --batch_size 32
```

Re-run inference later from the saved checkpoints:

```bash
python Training/predict.py --ckpt_dir Training/outputs --data_dir Dataset \
    --input Dataset/test/test_without_label.csv \
    --output Training/outputs/test_predictions.csv
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
- CodeIgniter Model with SQLite prediction history
- Python prediction socket on `127.0.0.1:16888`
- URL: `http://localhost:17888/2026Project/`

CodeIgniter pages:

- `/2026Project`: prediction form and prediction result
- `/2026Project/history`: SQL prediction history table
- `/2026Project/health`: server, socket, and model health table

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
