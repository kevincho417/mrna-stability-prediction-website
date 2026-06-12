from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "Training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data import DEFAULT_MAXLEN, mRNADataset  # noqa: E402
from model import mRNAStabilityNet  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "Training" / "outputs"


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


class PredictionService:
    """Serve the codon-aware multi-branch CNN as a 5-fold ensemble.

    Each fold checkpoint (`fold{k}_best.pt`) holds only model weights, so the
    feature normalization computed during training is loaded from
    `feature_stats.json` (written alongside the checkpoints). The decision
    threshold is the validation-tuned `global_threshold` from `cv_metrics.json`.
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.device = get_device()
        self.model_dir = Path(model_dir)

        stats = json.loads((self.model_dir / "feature_stats.json").read_text(encoding="utf-8"))
        self.feat_mean = np.asarray(stats["feat_mean"], dtype=np.float32)
        self.feat_std = np.asarray(stats["feat_std"], dtype=np.float32)
        self.maxlen = stats.get("maxlen", DEFAULT_MAXLEN)

        self.threshold = 0.5
        metrics_path = self.model_dir / "cv_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.threshold = float(metrics.get("global_threshold", 0.5))

        self._lock = threading.Lock()
        self.models = self._load_models()

    def _load_models(self) -> list[mRNAStabilityNet]:
        models = []
        for checkpoint_path in sorted(self.model_dir.glob("fold*_best.pt")):
            model = mRNAStabilityNet().to(self.device)
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        if not models:
            raise FileNotFoundError(f"No fold*_best.pt checkpoints found in {self.model_dir}")
        return models

    def _fold_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        dataset = mRNADataset(df, self.maxlen, self.feat_mean, self.feat_std, has_label=False)
        loader = DataLoader(dataset, batch_size=64)
        per_model = []
        for model in self.models:
            batch_probs = []
            with torch.no_grad():
                for batch in loader:
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    batch_probs.append(torch.sigmoid(model(batch)).cpu().numpy())
            per_model.append(np.concatenate(batch_probs))
        return np.stack(per_model)  # (n_models, n_rows)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        df = build_single_row(payload)
        if not df.loc[0, "CDSseq"]:
            raise ValueError("CDS sequence is required for prediction.")

        threshold = float(payload.get("threshold", self.threshold))
        with self._lock:
            fold_probabilities = self._fold_probabilities(df)[:, 0]

        probability = float(np.mean(fold_probabilities))
        label = int(probability >= threshold)
        return {
            "transcript_id": df.loc[0, "TranscriptID"],
            "predicted_probability": probability,
            "predicted_label": label,
            "class_name": "lowly degraded / stable mRNA" if label == 1 else "highly degraded mRNA",
            "threshold": threshold,
            "fold_probabilities": [float(value) for value in fold_probabilities],
            # Backward-compatible keys for the prediction-history DB and views.
            # The model is now a single CNN architecture trained as a 5-fold
            # ensemble, so these carry the lowest / highest fold probability.
            "mlp_probability": float(np.min(fold_probabilities)),
            "transformer_probability": float(np.max(fold_probabilities)),
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
            "model_dir": str(self.model_dir),
            "model_type": "codon-aware multi-branch CNN (5-fold ensemble)",
            "n_models": len(self.models),
            "threshold": self.threshold,
            "labels": {
                "0": "highly degraded mRNA",
                "1": "lowly degraded / stable mRNA",
            },
        }
