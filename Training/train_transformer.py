from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

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
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_TOKEN_TO_ID = {"A": 1, "U": 2, "C": 3, "G": 4}
PAD_ID = 0
BASE_UNK_ID = 5
KMER_UNK_ID = 1
SEGMENTS = (("5UTRseq", 1), ("CDSseq", 2), ("3UTRseq", 3))
ALPHABET = "AUCG"


@dataclass
class TransformerConfig:
    max_len: int = 8192
    kmer_size: int = 1
    vocab_size: int = 6
    d_model: int = 128
    conv_layers: int = 3
    conv_kernel_size: int = 7
    conv_stride: int = 2
    transformer_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.15
    classifier_hidden: int = 128
    batch_size: int = 16
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    threshold: float = 0.5


class SequenceDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, segment_ids: np.ndarray, lengths: np.ndarray, labels: np.ndarray | None = None):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.lengths = lengths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int):
        item = {
            "token_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            "segment_ids": torch.tensor(self.segment_ids[idx], dtype=torch.long),
            "length": torch.tensor(self.lengths[idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class ConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvTransformerClassifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        self.segment_embedding = nn.Embedding(4, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.conv = nn.Sequential(
            *[
                ConvBlock(
                    d_model=config.d_model,
                    kernel_size=config.conv_kernel_size,
                    stride=config.conv_stride,
                    dropout=config.dropout,
                )
                for _ in range(config.conv_layers)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, 1),
        )

    def downsampled_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for _ in range(self.config.conv_layers):
            lengths = torch.div(lengths + self.config.conv_stride - 1, self.config.conv_stride, rounding_mode="floor")
        return torch.clamp(lengths, min=1)

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(token_ids) + self.segment_embedding(segment_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)

        down_lengths = self.downsampled_lengths(lengths)
        time_steps = x.shape[1]
        mask = torch.arange(time_steps, device=x.device).unsqueeze(0) >= down_lengths.unsqueeze(1)
        x = self.transformer(x, src_key_padding_mask=mask)

        valid = (~mask).unsqueeze(-1)
        mean_pool = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        x_for_max = x.masked_fill(mask.unsqueeze(-1), torch.finfo(x.dtype).min)
        max_pool = x_for_max.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled).squeeze(1)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train an end-to-end Conv+Transformer mRNA classifier.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "Dataset")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Training" / "transformer_outputs")
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--kmer-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--conv-layers", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for fold in range(1, 6):
        path = data_dir / "training" / f"training_fold_{fold}.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        parts.append(df)
    train_df = pd.concat(parts, ignore_index=True)
    test_df = pd.read_csv(data_dir / "test" / "test_without_label.csv")
    return train_df, test_df


def clean_sequence(value: object) -> str:
    if isinstance(value, str):
        return value.upper().replace("T", "U")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).upper().replace("T", "U")


def build_token_to_id(kmer_size: int) -> tuple[dict[str, int], int]:
    if kmer_size < 1:
        raise ValueError("--kmer-size must be >= 1")
    if kmer_size == 1:
        return BASE_TOKEN_TO_ID.copy(), BASE_UNK_ID
    kmers = ("".join(kmer) for kmer in itertools.product(ALPHABET, repeat=kmer_size))
    return {kmer: idx for idx, kmer in enumerate(kmers, start=2)}, KMER_UNK_ID


def sequence_token_count(sequence: str, kmer_size: int) -> int:
    if kmer_size == 1:
        return len(sequence)
    return max(len(sequence) - kmer_size + 1, 0)


def tokenize_sequence(
    sequence: str,
    target_token_count: int,
    token_to_id: dict[str, int],
    unk_id: int,
    kmer_size: int,
) -> list[int]:
    token_count = sequence_token_count(sequence, kmer_size)
    if target_token_count <= 0 or token_count <= 0:
        return []

    target_token_count = min(target_token_count, token_count)
    if target_token_count == token_count:
        starts = range(token_count)
    else:
        left_count = target_token_count // 2
        right_count = target_token_count - left_count
        starts = list(range(left_count)) + list(range(token_count - right_count, token_count))

    if kmer_size == 1:
        return [token_to_id.get(sequence[start], unk_id) for start in starts]
    return [token_to_id.get(sequence[start : start + kmer_size], unk_id) for start in starts]


def allocate_segment_lengths(lengths: list[int], max_len: int) -> list[int]:
    total = sum(lengths)
    if total <= max_len:
        return lengths
    allocation = [0, 0, 0]
    nonzero = [idx for idx, length in enumerate(lengths) if length > 0]
    for idx in nonzero:
        allocation[idx] = max(1, int(max_len * lengths[idx] / total))
        allocation[idx] = min(allocation[idx], lengths[idx])
    while sum(allocation) > max_len:
        idx = max(nonzero, key=lambda i: allocation[i])
        allocation[idx] -= 1
    while sum(allocation) < max_len:
        candidates = [idx for idx in nonzero if allocation[idx] < lengths[idx]]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: lengths[i] - allocation[i])
        allocation[idx] += 1
    return allocation


def encode_dataframe(
    df: pd.DataFrame,
    max_len: int,
    token_to_id: dict[str, int],
    unk_id: int,
    kmer_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_ids = np.zeros((len(df), max_len), dtype=np.uint8)
    segment_ids = np.zeros((len(df), max_len), dtype=np.uint8)
    lengths = np.zeros(len(df), dtype=np.int64)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        raw_sequences = [clean_sequence(row.get(column_name, "")) for column_name, _ in SEGMENTS]
        target_lengths = allocate_segment_lengths([sequence_token_count(sequence, kmer_size) for sequence in raw_sequences], max_len)
        encoded_tokens: list[int] = []
        encoded_segments: list[int] = []
        for sequence, target_len, (_, segment_id) in zip(raw_sequences, target_lengths, SEGMENTS):
            tokens = tokenize_sequence(sequence, target_len, token_to_id, unk_id, kmer_size)
            encoded_tokens.extend(tokens)
            encoded_segments.extend([segment_id] * len(tokens))
        length = min(len(encoded_tokens), max_len)
        if length:
            token_ids[row_idx, :length] = np.asarray(encoded_tokens[:length], dtype=np.uint8)
            segment_ids[row_idx, :length] = np.asarray(encoded_segments[:length], dtype=np.uint8)
        lengths[row_idx] = max(length, 1)
    return token_ids, segment_ids, lengths


def make_loader(
    token_ids: np.ndarray,
    segment_ids: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SequenceDataset(token_ids, segment_ids, lengths, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["token_ids"], batch["segment_ids"], batch["length"])
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probabilities)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["token_ids"], batch["segment_ids"], batch["length"])
            loss = criterion(logits, batch["label"])
            loss_sum += loss.item() * len(batch["label"])
            count += len(batch["label"])
    return loss_sum / count


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


def train_one_fold(
    fold: int,
    train_tokens: np.ndarray,
    train_segments: np.ndarray,
    train_lengths: np.ndarray,
    y_all: np.ndarray,
    folds: np.ndarray,
    config: TransformerConfig,
    device: torch.device,
    model_dir: Path,
) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    train_mask = folds != fold
    valid_mask = folds == fold
    y_train = y_all[train_mask].astype(np.float32)
    y_valid = y_all[valid_mask].astype(np.float32)

    train_loader = make_loader(
        train_tokens[train_mask],
        train_segments[train_mask],
        train_lengths[train_mask],
        y_train,
        config.batch_size,
        True,
        config.seed + fold,
    )
    valid_loader = make_loader(
        train_tokens[valid_mask],
        train_segments[valid_mask],
        train_lengths[valid_mask],
        y_valid,
        config.batch_size,
        False,
        config.seed,
    )

    model = ConvTransformerClassifier(config).to(device)
    negative_count = float((y_train == 0).sum())
    positive_count = float((y_train == 1).sum())
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_auc = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    curve_rows: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["token_ids"], batch["segment_ids"], batch["length"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(batch["label"])
            train_count += len(batch["label"])

        train_loss = train_loss_sum / train_count
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
        train_eval_loader = make_loader(
            train_tokens[train_mask],
            train_segments[train_mask],
            train_lengths[train_mask],
            y_train,
            config.batch_size,
            False,
            config.seed,
        )
        train_prob = predict_probabilities(model, train_eval_loader, device)
        valid_prob = predict_probabilities(model, valid_loader, device)
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
    valid_prob = predict_probabilities(model, valid_loader, device)
    metrics = binary_metrics(y_valid.astype(int), valid_prob, config.threshold)
    metrics.update({"fold": float(fold), "best_epoch": float(best_epoch), "n_train": float(len(y_train)), "n_valid": float(len(y_valid))})

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "token_to_id": token_info_for_checkpoint(config),
            "pad_id": PAD_ID,
            "unk_id": BASE_UNK_ID if config.kmer_size == 1 else KMER_UNK_ID,
        },
        model_dir / f"fold_{fold}.pt",
    )
    return metrics, curve_rows, valid_prob


def predict_test_ensemble(
    test_tokens: np.ndarray,
    test_segments: np.ndarray,
    test_lengths: np.ndarray,
    config: TransformerConfig,
    device: torch.device,
    model_dir: Path,
) -> np.ndarray:
    loader = make_loader(test_tokens, test_segments, test_lengths, None, config.batch_size, False, config.seed)
    fold_predictions = []
    for fold in range(1, 6):
        checkpoint = torch.load(model_dir / f"fold_{fold}.pt", map_location=device, weights_only=False)
        model = ConvTransformerClassifier(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        fold_predictions.append(predict_probabilities(model, loader, device))
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


def token_info_for_checkpoint(config: TransformerConfig) -> dict[str, int]:
    token_to_id, _ = build_token_to_id(config.kmer_size)
    return token_to_id


def write_summary(metrics_df: pd.DataFrame, config: TransformerConfig, output_dir: Path) -> None:
    metric_columns = ["recall", "precision", "specificity", "f1", "auROC", "auPRC"]
    summary = {
        "model": "End-to-end Conv1D + Transformer Encoder over RNA token sequences",
        "tokenization": {
            "kmer_size": config.kmer_size,
            "PAD": PAD_ID,
            "UNK": BASE_UNK_ID if config.kmer_size == 1 else KMER_UNK_ID,
            "vocab_size": config.vocab_size,
        },
        "architecture": {
            "embedding_dim": config.d_model,
            "conv_layers": config.conv_layers,
            "conv_kernel_size": config.conv_kernel_size,
            "conv_stride": config.conv_stride,
            "transformer_layers": config.transformer_layers,
            "attention_heads": config.n_heads,
            "feedforward_dim": config.ff_dim,
            "pooling": "masked mean pooling concatenated with masked max pooling",
            "loss": "BCEWithLogitsLoss with fold-specific positive class weight",
            "optimizer": "AdamW",
        },
        "hyperparameters": asdict(config),
        "cv_mean": {column: float(metrics_df[column].mean()) for column in metric_columns},
        "cv_std": {column: float(metrics_df[column].std(ddof=0)) for column in metric_columns},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Transformer Training Summary",
        "",
        "Model: end-to-end Conv1D + Transformer Encoder over RNA token sequences.",
        f"Tokenization: overlapping {config.kmer_size}-mer tokens, PAD=0, vocab_size={config.vocab_size}.",
        f"Architecture: embedding dim {config.d_model}, {config.conv_layers} Conv1D downsampling layers, {config.transformer_layers} Transformer Encoder layers, {config.n_heads} heads.",
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
    config = TransformerConfig(
        max_len=args.max_len,
        kmer_size=args.kmer_size,
        d_model=args.d_model,
        conv_layers=args.conv_layers,
        transformer_layers=args.transformer_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    token_to_id, unk_id = build_token_to_id(config.kmer_size)
    config.vocab_size = max(max(token_to_id.values()), unk_id) + 1
    seed_everything(config.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"

    print(f"Using device: {device}")
    print("Reading data...")
    train_df, test_df = read_data(args.data_dir)
    y = train_df["Label"].astype(int).to_numpy()
    folds = train_df["fold"].to_numpy()

    print(f"Encoding RNA sequences as overlapping {config.kmer_size}-mer token ids...")
    train_tokens, train_segments, train_lengths = encode_dataframe(train_df, config.max_len, token_to_id, unk_id, config.kmer_size)
    test_tokens, test_segments, test_lengths = encode_dataframe(test_df, config.max_len, token_to_id, unk_id, config.kmer_size)
    token_info = {
        "token_to_id": token_to_id,
        "pad_id": PAD_ID,
        "unk_id": unk_id,
        "kmer_size": config.kmer_size,
        "vocab_size": config.vocab_size,
        "segment_id": {"5UTRseq": 1, "CDSseq": 2, "3UTRseq": 3},
        "max_len": config.max_len,
        "truncation": "If token count is longer than max_len, each region is allocated proportional token length and cropped by preserving both ends.",
    }
    (args.output_dir / "tokenization.json").write_text(json.dumps(token_info, indent=2), encoding="utf-8")
    print(f"Token matrix: {train_tokens.shape}")

    all_metrics = []
    all_curves = []
    out_of_fold = np.zeros(len(train_df), dtype=np.float32)
    for fold in range(1, 6):
        print(f"Training fold {fold}/5...")
        metrics, curves, valid_prob = train_one_fold(
            fold,
            train_tokens,
            train_segments,
            train_lengths,
            y,
            folds,
            config,
            device,
            model_dir,
        )
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
    test_probability = predict_test_ensemble(test_tokens, test_segments, test_lengths, config, device, model_dir)
    test_pred_df = test_df[["TranscriptID"]].copy()
    test_pred_df["predicted_probability"] = test_probability
    test_pred_df["predicted_label"] = (test_probability >= config.threshold).astype(int)
    test_pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False)

    write_summary(metrics_df, config, args.output_dir)
    print(f"Done. Mean CV auROC={metrics_df['auROC'].mean():.4f}, mean CV auPRC={metrics_df['auPRC'].mean():.4f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
