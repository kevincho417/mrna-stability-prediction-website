"""
Stand-alone inference: load the 5 fold checkpoints and predict on any CSV
with columns TranscriptID,5UTRseq,CDSseq,3UTRseq.

Usage:
    python predict.py --ckpt_dir ../outputs --data_dir ../Dataset \
        --input ../Dataset/test/test_without_label.csv \
        --output ../outputs/test_predictions.csv
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import mRNADataset, load_folds, DEFAULT_MAXLEN
from model import mRNAStabilityNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="../outputs")
    ap.add_argument("--data_dir", default="../Dataset")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="../outputs/predictions.csv")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.input, dtype=str).fillna("")

    # recompute training feature normalization (must match training)
    folds = load_folds(args.data_dir)
    train_all = pd.concat(list(folds.values()), ignore_index=True)
    ref = mRNADataset(train_all, DEFAULT_MAXLEN)
    ds = mRNADataset(df, DEFAULT_MAXLEN, ref.feat_mean, ref.feat_std, has_label=False)
    dl = DataLoader(ds, args.batch_size)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "fold*_best.pt")))
    assert ckpts, f"no checkpoints found in {args.ckpt_dir}"
    probs = []
    for c in ckpts:
        model = mRNAStabilityNet().to(device)
        model.load_state_dict(torch.load(c, map_location=device))
        model.eval()
        ps = []
        with torch.no_grad():
            for batch in dl:
                b = {k: v.to(device) for k, v in batch.items()}
                ps.append(torch.sigmoid(model(b)).cpu().numpy())
        probs.append(np.concatenate(ps))
    prob = np.mean(probs, axis=0)

    # reuse the validation-tuned threshold from training if available
    thr = 0.5
    mpath = os.path.join(args.ckpt_dir, "cv_metrics.json")
    if os.path.exists(mpath):
        thr = json.load(open(mpath)).get("global_threshold", 0.5)

    out = df[["TranscriptID"]].copy()
    out["prob"] = prob
    out["Label"] = (prob >= thr).astype(int)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions ({len(ckpts)}-model ensemble, "
          f"thr={thr:.3f}) -> {args.output}")


if __name__ == "__main__":
    main()
