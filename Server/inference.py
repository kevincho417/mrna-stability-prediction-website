from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "Training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train import MRNAStabilityMLP, extract_features  # noqa: E402
from train_transformer import (  # noqa: E402
    ConvTransformerClassifier,
    TransformerConfig,
    build_token_to_id,
    encode_dataframe,
    predict_probabilities,
    make_loader,
)


DEFAULT_MLP_DIR = PROJECT_ROOT / "Training" / "outputs"
DEFAULT_TRANSFORMER_DIR = PROJECT_ROOT / "Training" / "transformer_outputs_8192"
DEFAULT_ENSEMBLE_SUMMARY = PROJECT_ROOT / "Training" / "ensemble_outputs_base8192" / "summary.json"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_sequence(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper().replace("T", "U")
    return "".join(char for char in text if char in {"A", "U", "C", "G", "N"})


def build_single_row(payload: dict[str, Any]) -> pd.DataFrame:
    transcript_id = str(payload.get("transcript_id") or payload.get("TranscriptID") or "query")
    utr5 = payload.get("utr5", payload.get("5UTRseq", ""))
    cds = payload.get("cds", payload.get("CDSseq", ""))
    utr3 = payload.get("utr3", payload.get("3UTRseq", ""))
    return pd.DataFrame(
        [
            {
                "TranscriptID": transcript_id,
                "5UTRseq": normalize_sequence(utr5),
                "CDSseq": normalize_sequence(cds),
                "3UTRseq": normalize_sequence(utr3),
            }
        ]
    )


def load_ensemble_settings(summary_path: Path | None) -> tuple[float, float]:
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(summary.get("mlp_weight", 0.67)), float(summary.get("threshold", 0.5))
    return 0.67, 0.5


def config_from_checkpoint(raw_config: dict[str, Any], token_to_id: dict[str, int], unk_id: int) -> TransformerConfig:
    config = TransformerConfig()
    for key, value in raw_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.vocab_size = max(max(token_to_id.values(), default=0), unk_id) + 1
    if not hasattr(config, "kmer_size") or config.kmer_size is None:
        config.kmer_size = 1
    return config


class PredictionService:
    def __init__(
        self,
        mlp_dir: Path = DEFAULT_MLP_DIR,
        transformer_dir: Path = DEFAULT_TRANSFORMER_DIR,
        ensemble_summary: Path | None = DEFAULT_ENSEMBLE_SUMMARY,
    ) -> None:
        self.device = get_device()
        self.mlp_dir = Path(mlp_dir)
        self.transformer_dir = Path(transformer_dir)
        self.mlp_weight, self.threshold = load_ensemble_settings(ensemble_summary)
        self.transformer_weight = 1.0 - self.mlp_weight
        self._lock = threading.Lock()

        self.mlp_models = self._load_mlp_models()
        self.transformer_models, self.transformer_config, self.token_to_id, self.unk_id = self._load_transformer_models()

    def _load_mlp_models(self) -> list[dict[str, Any]]:
        models = []
        for fold in range(1, 6):
            checkpoint_path = self.mlp_dir / "models" / f"fold_{fold}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model = MRNAStabilityMLP(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dims=checkpoint["config"]["hidden_dims"],
                dropout=checkpoint["config"]["dropout"],
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(
                {
                    "model": model,
                    "mean": np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
                    "scale": np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
                    "batch_size": int(checkpoint["config"].get("batch_size", 64)),
                }
            )
        return models

    def _load_transformer_models(self) -> tuple[list[ConvTransformerClassifier], TransformerConfig, dict[str, int], int]:
        first_checkpoint = torch.load(
            self.transformer_dir / "models" / "fold_1.pt",
            map_location=self.device,
            weights_only=False,
        )
        token_to_id = first_checkpoint.get("token_to_id")
        raw_config = first_checkpoint.get("config", {})
        kmer_size = int(raw_config.get("kmer_size", 1))
        if not token_to_id:
            token_to_id, _ = build_token_to_id(kmer_size)
        unk_id = int(first_checkpoint.get("unk_id", 5 if kmer_size == 1 else 1))
        config = config_from_checkpoint(raw_config, token_to_id, unk_id)

        models = []
        for fold in range(1, 6):
            checkpoint = torch.load(
                self.transformer_dir / "models" / f"fold_{fold}.pt",
                map_location=self.device,
                weights_only=False,
            )
            model = ConvTransformerClassifier(config).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)
        return models, config, token_to_id, unk_id

    def _predict_mlp(self, df: pd.DataFrame) -> float:
        features = extract_features(df, max_k=4)
        fold_probabilities = []
        for item in self.mlp_models:
            scaled = ((features - item["mean"]) / item["scale"]).astype(np.float32)
            with torch.no_grad():
                tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
                logits = item["model"](tensor)
                probability = torch.sigmoid(logits).detach().cpu().numpy()[0]
            fold_probabilities.append(float(probability))
        return float(np.mean(fold_probabilities))

    def _predict_transformer(self, df: pd.DataFrame) -> float:
        token_ids, segment_ids, lengths = encode_dataframe(
            df,
            self.transformer_config.max_len,
            self.token_to_id,
            self.unk_id,
            self.transformer_config.kmer_size,
        )
        loader = make_loader(
            token_ids,
            segment_ids,
            lengths,
            labels=None,
            batch_size=int(self.transformer_config.batch_size),
            shuffle=False,
            seed=int(self.transformer_config.seed),
        )
        probabilities = []
        for model in self.transformer_models:
            probabilities.append(float(predict_probabilities(model, loader, self.device)[0]))
        return float(np.mean(probabilities))

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        df = build_single_row(payload)
        if not df.loc[0, "CDSseq"]:
            raise ValueError("CDS sequence is required for prediction.")

        threshold = float(payload.get("threshold", self.threshold))
        with self._lock:
            mlp_probability = self._predict_mlp(df)
            transformer_probability = self._predict_transformer(df)

        probability = self.mlp_weight * mlp_probability + self.transformer_weight * transformer_probability
        label = int(probability >= threshold)
        return {
            "transcript_id": df.loc[0, "TranscriptID"],
            "predicted_probability": probability,
            "predicted_label": label,
            "class_name": "lowly degraded / stable mRNA" if label == 1 else "highly degraded mRNA",
            "threshold": threshold,
            "mlp_probability": mlp_probability,
            "transformer_probability": transformer_probability,
            "mlp_weight": self.mlp_weight,
            "transformer_weight": self.transformer_weight,
            "sequence_lengths": {
                "5UTRseq": int(len(df.loc[0, "5UTRseq"])),
                "CDSseq": int(len(df.loc[0, "CDSseq"])),
                "3UTRseq": int(len(df.loc[0, "3UTRseq"])),
                "total": int(len(df.loc[0, "5UTRseq"]) + len(df.loc[0, "CDSseq"]) + len(df.loc[0, "3UTRseq"])),
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "mlp_dir": str(self.mlp_dir),
            "transformer_dir": str(self.transformer_dir),
            "ensemble": {
                "mlp_weight": self.mlp_weight,
                "transformer_weight": self.transformer_weight,
                "threshold": self.threshold,
            },
            "transformer_config": asdict(self.transformer_config),
            "labels": {
                "0": "highly degraded mRNA",
                "1": "lowly degraded / stable mRNA",
            },
        }
