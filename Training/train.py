from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEGMENTS = (("5UTRseq", "utr5"), ("CDSseq", "cds"), ("3UTRseq", "utr3"))
ALPHABET = "ACGU"
MOTIFS = ("AUG", "UAA", "UAG", "UGA", "AUUUA", "AAAAA", "UUUUU", "GGGG", "CCCC")


@dataclass
class TrainConfig:
    max_k: int = 4
    hidden_dims: tuple[int, ...] = (384, 128, 32)
    dropout: tuple[float, ...] = (0.30, 0.20, 0.10)
    batch_size: int = 64
    epochs: int = 200
    patience: int = 25
    learning_rate: float = 8e-4
    weight_decay: float = 3e-4
    seed: int = 42
    threshold: float = 0.5


class MRNAStabilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: Iterable[float]):
        super().__init__()
        hidden_dims = list(hidden_dims)
        dropout = list(dropout)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if idx < len(dropout) and dropout[idx] > 0:
                layers.append(nn.Dropout(dropout[idx]))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train an mRNA stability deep network.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "Dataset")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Training" / "outputs")
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    for fold in range(1, 6):
        fold_path = data_dir / "training" / f"training_fold_{fold}.csv"
        df = pd.read_csv(fold_path)
        df["fold"] = fold
        train_parts.append(df)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.read_csv(data_dir / "test" / "test_without_label.csv")
    return train_df, test_df


def kmer_index_by_k(max_k: int) -> list[dict[str, int]]:
    return [
        {"".join(kmer): idx for idx, kmer in enumerate(itertools.product(ALPHABET, repeat=k))}
        for k in range(1, max_k + 1)
    ]


def build_feature_names(max_k: int) -> list[str]:
    names: list[str] = []
    kmer_groups = kmer_index_by_k(max_k)
    for _, segment_name in SEGMENTS:
        names.extend(
            [
                f"{segment_name}_log_len",
                f"{segment_name}_len_scaled",
                f"{segment_name}_A_ratio",
                f"{segment_name}_C_ratio",
                f"{segment_name}_G_ratio",
                f"{segment_name}_U_ratio",
                f"{segment_name}_GC_ratio",
            ]
        )
        names.extend([f"{segment_name}_max_run_{base}" for base in ALPHABET])
        names.extend([f"{segment_name}_motif_{motif}" for motif in MOTIFS])
        for k, kmer_index in enumerate(kmer_groups, start=1):
            ordered_kmers = sorted(kmer_index, key=kmer_index.get)
            names.extend([f"{segment_name}_{k}mer_{kmer}" for kmer in ordered_kmers])
    names.extend(["all_log_len", "all_len_scaled", "all_GC_ratio"])
    names.extend([f"all_motif_{motif}" for motif in MOTIFS])
    return names


def clean_sequence(value: object) -> str:
    if isinstance(value, str):
        return value.upper().replace("T", "U")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).upper().replace("T", "U")


def longest_run_ratio(sequence: str, base: str) -> float:
    if not sequence:
        return 0.0
    best = 0
    current = 0
    for char in sequence:
        if char == base:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best / len(sequence)


def motif_frequency(sequence: str, motif: str) -> float:
    if not sequence:
        return 0.0
    denominator = max(len(sequence) - len(motif) + 1, 1)
    return sequence.count(motif) / denominator


def segment_features(sequence: str, kmer_groups: list[dict[str, int]]) -> list[float]:
    values: list[float] = []
    length = len(sequence)
    values.extend([math.log1p(length), length / 10000.0])
    values.extend([sequence.count(base) / length if length else 0.0 for base in ALPHABET])
    values.append((sequence.count("G") + sequence.count("C")) / length if length else 0.0)
    values.extend([longest_run_ratio(sequence, base) for base in ALPHABET])
    values.extend([motif_frequency(sequence, motif) for motif in MOTIFS])

    valid_chars = set(ALPHABET)
    for k, kmer_index in enumerate(kmer_groups, start=1):
        counts = np.zeros(len(kmer_index), dtype=np.float32)
        window_count = max(length - k + 1, 0)
        for start in range(window_count):
            kmer = sequence[start : start + k]
            if set(kmer) <= valid_chars:
                counts[kmer_index[kmer]] += 1
        denominator = max(window_count, 1)
        values.extend((counts / denominator).tolist())
    return values


def extract_features(df: pd.DataFrame, max_k: int) -> np.ndarray:
    kmer_groups = kmer_index_by_k(max_k)
    features: list[list[float]] = []
    for _, row in df.iterrows():
        row_values: list[float] = []
        sequences = []
        for column_name, _ in SEGMENTS:
            sequence = clean_sequence(row.get(column_name, ""))
            sequences.append(sequence)
            row_values.extend(segment_features(sequence, kmer_groups))
        combined = "".join(sequences)
        combined_len = len(combined)
        row_values.extend(
            [
                math.log1p(combined_len),
                combined_len / 10000.0,
                (combined.count("G") + combined.count("C")) / combined_len if combined_len else 0.0,
            ]
        )
        row_values.extend([motif_frequency(combined, motif) for motif in MOTIFS])
        features.append(row_values)
    return np.asarray(features, dtype=np.float32)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_probabilities(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auROC": roc_auc_score(y_true, probability),
        "auPRC": average_precision_score(y_true, probability),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def evaluate_loss(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            total_count += len(xb)
    return total_loss / total_count


def train_one_fold(
    fold: int,
    x_all: np.ndarray,
    y_all: np.ndarray,
    folds: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    model_dir: Path,
) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    train_mask = folds != fold
    valid_mask = folds == fold

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_all[train_mask]).astype(np.float32)
    x_valid = scaler.transform(x_all[valid_mask]).astype(np.float32)
    y_train = y_all[train_mask].astype(np.float32)
    y_valid = y_all[valid_mask].astype(np.float32)

    model = MRNAStabilityMLP(x_train.shape[1], config.hidden_dims, config.dropout).to(device)
    negative_count = float((y_train == 0).sum())
    positive_count = float((y_train == 1).sum())
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_loader(x_train, y_train, config.batch_size, shuffle=True, seed=config.seed + fold)

    best_state = copy.deepcopy(model.state_dict())
    best_auc = -np.inf
    best_epoch = 0
    stale_epochs = 0
    curve_rows: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_count += len(xb)

        train_loss = train_loss_sum / train_count
        valid_loss = evaluate_loss(model, x_valid, y_valid, criterion, device, config.batch_size)
        train_prob = predict_probabilities(model, x_train, device, config.batch_size)
        valid_prob = predict_probabilities(model, x_valid, device, config.batch_size)
        train_auc = roc_auc_score(y_train, train_prob)
        valid_auc = roc_auc_score(y_valid, valid_prob)
        train_auprc = average_precision_score(y_train, train_prob)
        valid_auprc = average_precision_score(y_valid, valid_prob)
        curve_rows.append(
            {
                "fold": float(fold),
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "train_auROC": float(train_auc),
                "valid_auROC": float(valid_auc),
                "train_auPRC": float(train_auprc),
                "valid_auPRC": float(valid_auprc),
            }
        )

        if valid_auc > best_auc + 1e-5:
            best_auc = valid_auc
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    valid_prob = predict_probabilities(model, x_valid, device, config.batch_size)
    metrics = binary_metrics(y_valid.astype(int), valid_prob, config.threshold)
    metrics.update({"fold": float(fold), "best_epoch": float(best_epoch), "n_train": float(len(y_train)), "n_valid": float(len(y_valid))})

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "config": asdict(config),
        },
        model_dir / f"fold_{fold}.pt",
    )
    return metrics, curve_rows, valid_prob


def predict_test_ensemble(
    x_test: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    model_dir: Path,
) -> np.ndarray:
    fold_predictions = []
    for fold in range(1, 6):
        checkpoint = torch.load(model_dir / f"fold_{fold}.pt", map_location=device, weights_only=False)
        scaler_mean = checkpoint["scaler_mean"]
        scaler_scale = checkpoint["scaler_scale"]
        x_scaled = ((x_test - scaler_mean) / scaler_scale).astype(np.float32)
        model = MRNAStabilityMLP(checkpoint["input_dim"], config.hidden_dims, config.dropout).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        fold_predictions.append(predict_probabilities(model, x_scaled, device, config.batch_size))
    return np.mean(np.vstack(fold_predictions), axis=0)


def save_learning_curve_plot(curve_df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping learning curve plot because matplotlib is unavailable: {exc}")
        return

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for fold, fold_df in curve_df.groupby("fold"):
        axes[0].plot(fold_df["epoch"], fold_df["train_loss"], alpha=0.45, label=f"fold {int(fold)} train")
        axes[0].plot(fold_df["epoch"], fold_df["valid_loss"], alpha=0.85, linestyle="--", label=f"fold {int(fold)} valid")
        axes[1].plot(fold_df["epoch"], fold_df["valid_auROC"], alpha=0.85, label=f"fold {int(fold)}")
    axes[0].set_title("BCE loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation auROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("auROC")
    axes[1].set_ylim(0.45, 1.0)
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def write_summary(metrics_df: pd.DataFrame, config: TrainConfig, output_dir: Path) -> None:
    metric_columns = ["recall", "precision", "specificity", "f1", "auROC", "auPRC"]
    summary = {
        "model": "MLP over per-region RNA composition, motif, and k-mer frequency features",
        "architecture": {
            "hidden_dims": list(config.hidden_dims),
            "dropout": list(config.dropout),
            "activation": "ReLU",
            "normalization": "BatchNorm1d after each hidden Linear layer",
            "loss": "BCEWithLogitsLoss with fold-specific positive class weight",
            "optimizer": "AdamW",
        },
        "hyperparameters": asdict(config),
        "cv_mean": {column: float(metrics_df[column].mean()) for column in metric_columns},
        "cv_std": {column: float(metrics_df[column].std(ddof=0)) for column in metric_columns},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Training Summary",
        "",
        "Model: MLP over per-region RNA composition, motif, and k-mer frequency features.",
        f"Architecture: input -> {' -> '.join(map(str, config.hidden_dims))} -> 1, ReLU, BatchNorm1d, dropout {list(config.dropout)}.",
        f"Optimizer: AdamW, lr={config.learning_rate}, weight_decay={config.weight_decay}, batch_size={config.batch_size}.",
        "",
        "## 5-fold CV mean metrics",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for column in metric_columns:
        lines.append(f"| {column} | {metrics_df[column].mean():.4f} | {metrics_df[column].std(ddof=0):.4f} |")
    lines.extend(["", "See `cv_metrics.csv` for per-fold results and `learning_curves.csv` for epoch-level curves."])
    (output_dir / "report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        max_k=args.max_k,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    seed_everything(config.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"

    print(f"Using device: {device}")
    print("Reading data...")
    train_df, test_df = read_data(args.data_dir)
    y = train_df["Label"].astype(int).to_numpy()
    folds = train_df["fold"].to_numpy()

    print("Extracting features...")
    feature_names = build_feature_names(config.max_k)
    x_train = extract_features(train_df, config.max_k)
    x_test = extract_features(test_df, config.max_k)
    if x_train.shape[1] != len(feature_names):
        raise RuntimeError(f"Feature mismatch: {x_train.shape[1]} values vs {len(feature_names)} names.")
    (args.output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    print(f"Training feature matrix: {x_train.shape}")
    all_metrics = []
    all_curves = []
    out_of_fold = np.zeros(len(train_df), dtype=np.float32)
    for fold in range(1, 6):
        print(f"Training fold {fold}/5...")
        metrics, curves, valid_prob = train_one_fold(fold, x_train, y, folds, config, device, model_dir)
        valid_indices = np.where(folds == fold)[0]
        out_of_fold[valid_indices] = valid_prob
        all_metrics.append(metrics)
        all_curves.extend(curves)
        print(
            "Fold {fold}: auROC={auroc:.4f}, auPRC={auprc:.4f}, F1={f1:.4f}, best_epoch={epoch}".format(
                fold=fold,
                auroc=metrics["auROC"],
                auprc=metrics["auPRC"],
                f1=metrics["f1"],
                epoch=int(metrics["best_epoch"]),
            )
        )

    metrics_df = pd.DataFrame(all_metrics)
    curve_df = pd.DataFrame(all_curves)
    metrics_df.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    curve_df.to_csv(args.output_dir / "learning_curves.csv", index=False)
    save_learning_curve_plot(curve_df, args.output_dir / "learning_curves.png")

    train_pred_df = train_df[["TranscriptID", "fold", "Label"]].copy()
    train_pred_df["predicted_probability"] = out_of_fold
    train_pred_df["predicted_label"] = (out_of_fold >= config.threshold).astype(int)
    train_pred_df.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    print("Predicting test set with the 5-fold ensemble...")
    test_probability = predict_test_ensemble(x_test, config, device, model_dir)
    test_pred_df = test_df[["TranscriptID"]].copy()
    test_pred_df["predicted_probability"] = test_probability
    test_pred_df["predicted_label"] = (test_probability >= config.threshold).astype(int)
    test_pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False)

    write_summary(metrics_df, config, args.output_dir)
    mean_auc = metrics_df["auROC"].mean()
    mean_auprc = metrics_df["auPRC"].mean()
    print(f"Done. Mean CV auROC={mean_auc:.4f}, mean CV auPRC={mean_auprc:.4f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
