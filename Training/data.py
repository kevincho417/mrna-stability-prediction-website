"""Dataset, tokenization, and feature engineering for mRNA stability.

Sequence inputs to the model:
  - 5'UTR : nucleotide-level token ids
  - 3'UTR : nucleotide-level token ids
  - CDS   : CODON-level token ids (non-overlapping triplets) -- captures codon
            optimality, a dominant determinant of mRNA stability.
Plus an engineered feature vector (lengths, GC, motif counts, codon usage).
"""
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from model import NT_VOCAB

REGIONS = ["5UTRseq", "CDSseq", "3UTRseq"]

# nucleotide caps (UTRs) and codon cap (CDS, in codons => 700*3 = 2100 nt)
DEFAULT_MAXLEN = {"5UTRseq": 512, "3UTRseq": 1024, "CDS_CODONS": 700}

# codon vocabulary: 0 = PAD, 1..64 = codons, 65 = unknown (N / incomplete)
CODONS = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"]
CODON_IDX = {c: i + 1 for i, c in enumerate(CODONS)}
CODON_UNK = 65
N_EXTRA_FEATS = 11 + 64  # 11 scalar features + 64-dim codon frequency


def encode_nt(seq: str, max_len: int) -> np.ndarray:
    seq = (seq or "").upper().replace("T", "U")[:max_len]
    ids = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq):
        ids[i] = NT_VOCAB.get(ch, NT_VOCAB["N"])
    return ids


def encode_codons(seq: str, max_codons: int) -> np.ndarray:
    seq = (seq or "").upper().replace("T", "U")
    ids = np.zeros(max_codons, dtype=np.int64)
    n = min(len(seq) // 3, max_codons)
    for i in range(n):
        ids[i] = CODON_IDX.get(seq[i * 3:i * 3 + 3], CODON_UNK)
    return ids


def _gc(s):
    s = (s or "").upper(); n = len(s) or 1
    return (s.count("G") + s.count("C")) / n


def _codon_freq(seq):
    seq = (seq or "").upper().replace("T", "U")
    v = np.zeros(64, dtype=np.float32)
    n = len(seq) // 3
    for i in range(n):
        c = seq[i * 3:i * 3 + 3]
        if c in CODON_IDX:
            v[CODON_IDX[c] - 1] += 1
    return v / n if n else v


def featurize(row) -> np.ndarray:
    u5, cds, u3 = [(row[r] or "").upper().replace("T", "U") for r in REGIONS]
    scal = np.array([
        np.log1p(len(u5)), np.log1p(len(cds)), np.log1p(len(u3)),
        _gc(u5), _gc(cds), _gc(u3),
        np.log1p(u3.count("AUUUA")),            # AU-rich element (destabilizer)
        np.log1p(u5.count("AUG")),              # upstream AUGs in 5'UTR
        _gc(cds[:90]) if len(cds) >= 90 else _gc(cds),   # 5' CDS "ramp" GC
        len(u3) / (len(cds) + 1), len(u5) / (len(cds) + 1),
    ], dtype=np.float32)
    return np.concatenate([scal, _codon_freq(cds)])


class mRNADataset(Dataset):
    def __init__(self, df, maxlen=None, feat_mean=None, feat_std=None, has_label=True):
        self.df = df.reset_index(drop=True)
        self.maxlen = maxlen or DEFAULT_MAXLEN
        self.has_label = has_label
        feats = np.stack([featurize(r) for _, r in self.df.iterrows()])
        if feat_mean is None:
            feat_mean, feat_std = feats.mean(0), feats.std(0) + 1e-6
        self.feat_mean, self.feat_std = feat_mean, feat_std
        self.feats = (feats - feat_mean) / feat_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        out = {}
        out["utr5"] = torch.from_numpy(encode_nt(row["5UTRseq"], self.maxlen["5UTRseq"]))
        out["utr3"] = torch.from_numpy(encode_nt(row["3UTRseq"], self.maxlen["3UTRseq"]))
        out["cds"]  = torch.from_numpy(encode_codons(row["CDSseq"], self.maxlen["CDS_CODONS"]))
        for key in ("utr5", "utr3", "cds"):
            out[key + "_mask"] = out[key] != 0
        out["extra"] = torch.from_numpy(self.feats[i].astype(np.float32))
        if self.has_label:
            out["label"] = torch.tensor(float(row["Label"]))
        return out


def load_folds(data_dir: str):
    folds = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "training", "*.csv"))):
        fid = int(os.path.basename(fp).replace(".csv", "").split("_")[-1])
        d = pd.read_csv(fp, dtype=str).fillna("")
        d["Label"] = d["Label"].astype(int)
        folds[fid] = d
    return folds


def load_test(data_dir: str):
    return pd.read_csv(os.path.join(data_dir, "test", "test_without_label.csv"),
                       dtype=str).fillna("")
