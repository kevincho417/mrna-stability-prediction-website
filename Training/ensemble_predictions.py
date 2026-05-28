from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Ensemble MLP feature-model and ConvTransformer predictions.")
    parser.add_argument("--mlp-dir", type=Path, default=project_root / "Training" / "outputs")
    parser.add_argument("--transformer-dir", type=Path, default=project_root / "Training" / "transformer_outputs_8192")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Training" / "ensemble_outputs")
    parser.add_argument("--mlp-weight", type=float, default=None, help="Optional fixed MLP weight. Transformer weight is 1 - MLP weight.")
    parser.add_argument("--optimize-metric", choices=("auROC", "auPRC", "f1"), default="auROC")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true", help="Tune classification threshold on OOF predictions for best F1.")
    return parser.parse_args()


def load_oof_predictions(model_dir: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(model_dir / "oof_predictions.csv")
    required = {"TranscriptID", "fold", "Label", "predicted_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_dir / 'oof_predictions.csv'} is missing columns: {sorted(missing)}")
    return df.rename(columns={"predicted_probability": f"{prefix}_prob"})[
        ["TranscriptID", "fold", "Label", f"{prefix}_prob"]
    ]


def load_test_predictions(model_dir: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(model_dir / "test_predictions.csv")
    required = {"TranscriptID", "predicted_probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_dir / 'test_predictions.csv'} is missing columns: {sorted(missing)}")
    return df.rename(columns={"predicted_probability": f"{prefix}_prob"})[["TranscriptID", f"{prefix}_prob"]]


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


def weighted_probability(df: pd.DataFrame, mlp_weight: float) -> np.ndarray:
    return mlp_weight * df["mlp_prob"].to_numpy() + (1.0 - mlp_weight) * df["transformer_prob"].to_numpy()


def score_probability(y_true: np.ndarray, probability: np.ndarray, metric: str, threshold: float) -> float:
    if metric == "auROC":
        return roc_auc_score(y_true, probability)
    if metric == "auPRC":
        return average_precision_score(y_true, probability)
    return f1_score(y_true, (probability >= threshold).astype(int), zero_division=0)


def choose_weight(df: pd.DataFrame, metric: str, threshold: float, fixed_weight: float | None) -> tuple[float, pd.DataFrame]:
    y_true = df["Label"].astype(int).to_numpy()
    if fixed_weight is not None:
        if not 0.0 <= fixed_weight <= 1.0:
            raise ValueError("--mlp-weight must be between 0 and 1")
        probability = weighted_probability(df, fixed_weight)
        score = score_probability(y_true, probability, metric, threshold)
        return fixed_weight, pd.DataFrame([{"mlp_weight": fixed_weight, "transformer_weight": 1.0 - fixed_weight, metric: score}])

    rows = []
    best_weight = 0.0
    best_score = -np.inf
    for weight in np.linspace(0.0, 1.0, 101):
        probability = weighted_probability(df, float(weight))
        score = score_probability(y_true, probability, metric, threshold)
        rows.append({"mlp_weight": float(weight), "transformer_weight": float(1.0 - weight), metric: float(score)})
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight, pd.DataFrame(rows)


def choose_threshold(y_true: np.ndarray, probability: np.ndarray, default_threshold: float, tune: bool) -> tuple[float, pd.DataFrame]:
    if not tune:
        return default_threshold, pd.DataFrame(
            [{"threshold": default_threshold, "f1": f1_score(y_true, probability >= default_threshold, zero_division=0)}]
        )

    rows = []
    best_threshold = default_threshold
    best_f1 = -np.inf
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        rows.append({"threshold": float(threshold), "f1": float(score)})
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, pd.DataFrame(rows)


def per_fold_metrics(df: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["ensemble_prob"] = probability
    for fold, fold_df in work.groupby("fold"):
        metrics = binary_metrics(
            fold_df["Label"].astype(int).to_numpy(),
            fold_df["ensemble_prob"].to_numpy(),
            threshold,
        )
        metrics["fold"] = float(fold)
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mlp_oof = load_oof_predictions(args.mlp_dir, "mlp")
    transformer_oof = load_oof_predictions(args.transformer_dir, "transformer")
    oof = mlp_oof.merge(transformer_oof, on=["TranscriptID", "fold", "Label"], how="inner")
    if len(oof) != len(mlp_oof) or len(oof) != len(transformer_oof):
        raise RuntimeError("OOF prediction files do not align cleanly by TranscriptID, fold, and Label.")

    mlp_test = load_test_predictions(args.mlp_dir, "mlp")
    transformer_test = load_test_predictions(args.transformer_dir, "transformer")
    test = mlp_test.merge(transformer_test, on="TranscriptID", how="inner")
    if len(test) != len(mlp_test) or len(test) != len(transformer_test):
        raise RuntimeError("Test prediction files do not align cleanly by TranscriptID.")

    weight, weight_search_df = choose_weight(oof, args.optimize_metric, args.threshold, args.mlp_weight)
    oof_probability = weighted_probability(oof, weight)
    y_true = oof["Label"].astype(int).to_numpy()
    threshold, threshold_search_df = choose_threshold(y_true, oof_probability, args.threshold, args.tune_threshold)

    metrics_df = per_fold_metrics(oof, oof_probability, threshold)
    metrics_df.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    weight_search_df.to_csv(args.output_dir / "weight_search.csv", index=False)
    threshold_search_df.to_csv(args.output_dir / "threshold_search.csv", index=False)

    oof_output = oof.copy()
    oof_output["predicted_probability"] = oof_probability
    oof_output["predicted_label"] = (oof_probability >= threshold).astype(int)
    oof_output.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    test_probability = weighted_probability(test, weight)
    test_output = test.copy()
    test_output["predicted_probability"] = test_probability
    test_output["predicted_label"] = (test_probability >= threshold).astype(int)
    test_output.to_csv(args.output_dir / "test_predictions.csv", index=False)

    metric_columns = ["recall", "precision", "specificity", "f1", "auROC", "auPRC"]
    summary = {
        "mlp_dir": str(args.mlp_dir),
        "transformer_dir": str(args.transformer_dir),
        "mlp_weight": weight,
        "transformer_weight": 1.0 - weight,
        "optimize_metric": args.optimize_metric,
        "threshold": threshold,
        "cv_mean": {column: float(metrics_df[column].mean()) for column in metric_columns},
        "cv_std": {column: float(metrics_df[column].std(ddof=0)) for column in metric_columns},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Ensemble Summary",
        "",
        f"MLP directory: `{args.mlp_dir}`",
        f"Transformer directory: `{args.transformer_dir}`",
        f"Best MLP weight: {weight:.2f}",
        f"Best Transformer weight: {1.0 - weight:.2f}",
        f"Threshold: {threshold:.3f}",
        "",
        "## 5-fold CV mean metrics",
        "",
        "| metric | mean | std |",
        "|---|---:|---:|",
    ]
    for column in metric_columns:
        lines.append(f"| {column} | {metrics_df[column].mean():.4f} | {metrics_df[column].std(ddof=0):.4f} |")
    (args.output_dir / "report_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Best MLP weight: {weight:.2f}; Transformer weight: {1.0 - weight:.2f}; threshold: {threshold:.3f}")
    print(f"Mean CV auROC={metrics_df['auROC'].mean():.4f}, auPRC={metrics_df['auPRC'].mean():.4f}, F1={metrics_df['f1'].mean():.4f}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
