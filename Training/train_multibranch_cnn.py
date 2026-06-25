"""
5-fold CV training for the Codon-Aware Multi-Branch CNN (mRNAStabilityNet).
Replicates the GitHub repo model and evaluates on the same dataset.
"""
from __future__ import annotations
import os, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

from multibranch_cnn_model import (mRNAStabilityNet, mRNADataset,
                                    load_folds, load_test, DEFAULT_MAXLEN)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optim=None, crit=None):
    train = optim is not None
    model.train(train)
    losses, ys, ps = [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        logits = model(b)
        if "label" in b:
            loss = crit(logits, b["label"])
            if train:
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
            losses.append(loss.item())
            ys.append(b["label"].detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    probs = np.concatenate(ps)
    if ys:
        return float(np.mean(losses)), np.concatenate(ys), probs
    return None, None, probs


def evaluate(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "specificity": float(spec),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "acc": float(accuracy_score(y, pred)),
        "thr": float(thr),
    }


def best_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 91)))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


METRIC_KEYS = ["auroc", "auprc", "recall", "precision", "specificity", "f1", "acc"]


def plot_curves(history, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, h in history.items():
        ep = range(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], label=f"fold {k}")
        axes[1].plot(ep, h["auroc"], label=f"fold {k}")
        axes[2].plot(ep, h["auprc"], label=f"fold {k}")
    axes[0].set_title("Train loss"); axes[0].set_xlabel("epoch")
    axes[1].set_title("Val auROC"); axes[1].set_xlabel("epoch")
    axes[1].axhline(0.75, ls="--", c="gray", lw=1, label="min 0.75")
    axes[2].set_title("Val auPRC"); axes[2].set_xlabel("epoch")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Multi-Branch CNN — 5-fold CV learning curves")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def main():
    set_seed(42)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "Dataset")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs_multibranch_cnn")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    epochs = 40
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4
    patience = 10
    dropout = 0.3

    folds = load_folds(data_dir)
    fold_ids = sorted(folds)
    test_df = load_test(data_dir)

    cv_metrics, oof_rows, test_prob_acc, fold_thresholds = {}, [], [], []
    history = {}

    for k in fold_ids:
        print(f"\n{'='*50}")
        print(f"  Fold {k}")
        print(f"{'='*50}")

        val_df = folds[k]
        train_df = pd.concat([folds[j] for j in fold_ids if j != k], ignore_index=True)

        train_ds = mRNADataset(train_df, DEFAULT_MAXLEN)
        val_ds = mRNADataset(val_df, DEFAULT_MAXLEN, train_ds.feat_mean, train_ds.feat_std)
        test_ds = mRNADataset(test_df, DEFAULT_MAXLEN, train_ds.feat_mean, train_ds.feat_std, has_label=False)

        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size, num_workers=0)
        test_dl = DataLoader(test_ds, batch_size, num_workers=0)

        model = mRNAStabilityNet(dropout=dropout).to(device)
        if k == fold_ids[0]:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total trainable params: {total_params:,}")

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
        pos_w = (train_df["Label"] == 0).sum() / max((train_df["Label"] == 1).sum(), 1)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))

        history[k] = {"loss": [], "auroc": [], "auprc": []}
        best_auc, best_state, wait = -1, None, 0

        for ep in range(1, epochs + 1):
            tr_loss, _, _ = run_epoch(model, train_dl, device, optim, crit)
            _, yv, pv = run_epoch(model, val_dl, device, crit=crit)
            auroc = roc_auc_score(yv, pv)
            auprc = average_precision_score(yv, pv)
            sched.step()

            history[k]["loss"].append(tr_loss)
            history[k]["auroc"].append(auroc)
            history[k]["auprc"].append(auprc)

            print(f"  [fold {k}] ep {ep:02d}  loss {tr_loss:.4f}  "
                  f"val auROC {auroc:.4f}  auPRC {auprc:.4f}")

            if auroc > best_auc:
                best_auc, wait = auroc, 0
                best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print(f"  [fold {k}] early stop at epoch {ep}")
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(out_dir, f"fold{k}_best.pt"))

        _, yv, pv = run_epoch(model, val_dl, device, crit=crit)
        thr = best_threshold(yv, pv)
        fold_thresholds.append(thr)
        cv_metrics[f"fold{k}"] = evaluate(yv, pv, thr)
        m = cv_metrics[f"fold{k}"]
        print(f"  [fold {k}] BEST auROC={m['auroc']:.4f} auPRC={m['auprc']:.4f} "
              f"F1={m['f1']:.4f} thr={thr:.3f}")

        tmp = val_df[["TranscriptID"]].copy()
        tmp["Label"] = yv.astype(int); tmp["prob"] = pv; tmp["fold"] = k
        oof_rows.append(tmp)

        _, _, pt = run_epoch(model, test_dl, device)
        test_prob_acc.append(pt)

    # ── aggregate ────────────────────────────────────────────────────────────
    global_thr = float(np.mean(fold_thresholds))
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    oof_metrics = evaluate(oof["Label"].values, oof["prob"].values, global_thr)

    mean = {m: float(np.mean([cv_metrics[f][m] for f in cv_metrics])) for m in METRIC_KEYS}
    std  = {m: float(np.std([cv_metrics[f][m] for f in cv_metrics]))  for m in METRIC_KEYS}
    cv_metrics["mean"] = mean
    cv_metrics["std"] = std
    cv_metrics["oof_overall"] = oof_metrics
    cv_metrics["global_threshold"] = global_thr

    with open(os.path.join(out_dir, "cv_metrics.json"), "w") as f:
        json.dump(cv_metrics, f, indent=2)

    plot_curves(history, os.path.join(out_dir, "learning_curves.png"))

    test_prob = np.mean(test_prob_acc, axis=0)
    sub = test_df[["TranscriptID"]].copy()
    sub["prob"] = test_prob
    sub["Label"] = (test_prob >= global_thr).astype(int)
    sub.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    print(f"\n{'='*60}")
    print("  Multi-Branch CNN — 5-fold CV Results (mean ± std)")
    print(f"{'='*60}")
    for m in METRIC_KEYS:
        print(f"  {m:14s}: {mean[m]:.4f} ± {std[m]:.4f}")
    print(f"\n  OOF overall auROC: {oof_metrics['auroc']:.4f}")
    print(f"  Grading check: CV auROC {mean['auroc']:.4f} "
          f"({'PASS' if mean['auroc'] >= 0.75 else 'BELOW'} 0.75 minimum)")
    print(f"\n  Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
