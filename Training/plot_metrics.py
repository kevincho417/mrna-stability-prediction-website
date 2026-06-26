"""
Generate comprehensive training evaluation charts:
  1. Per-fold ROC curves + mean ROC
  2. Per-fold Precision-Recall curves + mean PRC
  3. Confusion matrix (OOF overall)
  4. Per-fold metric comparison bar chart
  5. Score distribution histogram
  6. Threshold vs F1/Precision/Recall curve
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix,
                             f1_score)

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────
oof = pd.read_csv(os.path.join(os.path.dirname(__file__),
                               "outputs_multibranch_cnn", "oof_predictions.csv"))
with open(os.path.join(os.path.dirname(__file__),
                       "outputs_multibranch_cnn", "cv_metrics.json")) as f:
    metrics = json.load(f)

folds = sorted(oof["fold"].unique())
global_thr = metrics["global_threshold"]

COLORS = ["#4A90D9", "#E8913A", "#2EAD6B", "#D94452", "#7B68EE"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. ROC Curves (per-fold + mean)
# ═══════════════════════════════════════════════════════════════════════════
def plot_roc():
    fig, ax = plt.subplots(figsize=(7, 7))
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 200)

    for i, k in enumerate(folds):
        d = oof[oof["fold"] == k]
        fpr, tpr, _ = roc_curve(d["Label"], d["prob"])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, color=COLORS[i], alpha=0.5, lw=1.2,
                label=f"Fold {k} (auROC = {roc_auc:.4f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="#1a1a2e", lw=2.5,
            label=f"Mean (auROC = {mean_auc:.4f} ± {std_auc:.4f})")

    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="#1a1a2e", alpha=0.1, label="± 1 std")

    ax.plot([0, 1], [0, 1], ls="--", color="#AAAAAA", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — 5-Fold Cross-Validation", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  roc_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Precision-Recall Curves (per-fold + mean)
# ═══════════════════════════════════════════════════════════════════════════
def plot_prc():
    fig, ax = plt.subplots(figsize=(7, 7))
    precs, aps = [], []
    mean_recall = np.linspace(0, 1, 200)

    for i, k in enumerate(folds):
        d = oof[oof["fold"] == k]
        prec, rec, _ = precision_recall_curve(d["Label"], d["prob"])
        ap = average_precision_score(d["Label"], d["prob"])
        aps.append(ap)
        precs.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
        ax.plot(rec, prec, color=COLORS[i], alpha=0.5, lw=1.2,
                label=f"Fold {k} (auPRC = {ap:.4f})")

    mean_prec = np.mean(precs, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    ax.plot(mean_recall, mean_prec, color="#1a1a2e", lw=2.5,
            label=f"Mean (auPRC = {mean_ap:.4f} ± {std_ap:.4f})")

    std_prec = np.std(precs, axis=0)
    ax.fill_between(mean_recall, mean_prec - std_prec, mean_prec + std_prec,
                    color="#1a1a2e", alpha=0.1, label="± 1 std")

    baseline = oof["Label"].mean()
    ax.axhline(baseline, ls="--", color="#AAAAAA", lw=1, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — 5-Fold Cross-Validation", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([0.3, 1.02])
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "prc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  prc_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Confusion Matrix (OOF overall)
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion():
    y = oof["Label"].values
    pred = (oof["prob"].values >= global_thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            total = cm[i].sum()
            pct = val / total * 100
            ax.text(j, i, f"{val}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    color="white" if val > cm.max() * 0.5 else "#1a1a2e")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Degraded (0)", "Stable (1)"], fontsize=11)
    ax.set_yticklabels(["Degraded (0)", "Stable (1)"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (OOF, threshold = {global_thr:.3f})",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-fold Metric Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_fold_bars():
    metric_keys = ["auroc", "auprc", "f1", "recall", "precision", "specificity"]
    labels = ["auROC", "auPRC", "F1", "Recall", "Precision", "Specificity"]
    n_metrics = len(metric_keys)
    n_folds = len(folds)
    x = np.arange(n_metrics)
    w = 0.14

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, k in enumerate(folds):
        vals = [metrics[f"fold{k}"][m] for m in metric_keys]
        ax.bar(x + (i - 2) * w, vals, w, label=f"Fold {k}", color=COLORS[i], alpha=0.85)

    # mean line
    means = [metrics["mean"][m] for m in metric_keys]
    for j, m in enumerate(means):
        ax.plot([x[j] - 2.5 * w, x[j] + 2.5 * w], [m, m],
                color="#1a1a2e", lw=2, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Fold Metrics Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim([0.55, 0.9])
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.75, ls=":", color="gray", lw=1, label="min 0.75")
    fig.savefig(os.path.join(OUT, "fold_metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  fold_metrics_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Score Distribution Histogram
# ═══════════════════════════════════════════════════════════════════════════
def plot_score_dist():
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 50)

    for label, name, color in [(0, "Degraded (0)", "#D94452"), (1, "Stable (1)", "#2EAD6B")]:
        vals = oof[oof["Label"] == label]["prob"]
        ax.hist(vals, bins=bins, alpha=0.6, color=color, label=name, edgecolor="white", lw=0.5)

    ax.axvline(global_thr, color="#1a1a2e", ls="--", lw=2,
               label=f"Threshold = {global_thr:.3f}")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution by Class (OOF)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "score_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  score_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Threshold vs F1/Precision/Recall
# ═══════════════════════════════════════════════════════════════════════════
def plot_threshold():
    y = oof["Label"].values
    p = oof["prob"].values
    thresholds = np.linspace(0.1, 0.8, 100)
    f1s, precs, recs = [], [], []

    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0
        re = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0
        f1s.append(f1); precs.append(pr); recs.append(re)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, color="#4A90D9", lw=2, label="F1")
    ax.plot(thresholds, precs, color="#E8913A", lw=2, label="Precision")
    ax.plot(thresholds, recs, color="#2EAD6B", lw=2, label="Recall")
    ax.axvline(global_thr, color="#1a1a2e", ls="--", lw=1.5,
               label=f"Selected threshold = {global_thr:.3f}")

    best_idx = np.argmax(f1s)
    ax.plot(thresholds[best_idx], f1s[best_idx], "o", color="#D94452", ms=8,
            label=f"Best F1 = {f1s[best_idx]:.4f} @ {thresholds[best_idx]:.3f}")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Tuning — F1 / Precision / Recall", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.1, 0.8])
    fig.savefig(os.path.join(OUT, "threshold_tuning.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  threshold_tuning.png")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating evaluation charts...")
    plot_roc()
    plot_prc()
    plot_confusion()
    plot_fold_bars()
    plot_score_dist()
    plot_threshold()
    print(f"\nDone! All charts saved to {OUT}/")
