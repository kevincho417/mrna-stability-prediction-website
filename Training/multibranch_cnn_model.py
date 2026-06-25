"""
Codon-aware multi-branch CNN for mRNA stability prediction.
Replicated from GitHub repo: kevincho417/mrna-stability-prediction-website
"""
from __future__ import annotations
import torch
import torch.nn as nn

NT_VOCAB = {"PAD": 0, "A": 1, "C": 2, "G": 3, "U": 4, "T": 4, "N": 5}
NT_SIZE = 6
CODON_SIZE = 66
N_EXTRA_FEATS = 11 + 64

CODONS = [a + b + c for a in "ACGU" for b in "ACGU" for c in "ACGU"]
CODON_IDX = {c: i + 1 for i, c in enumerate(CODONS)}
CODON_UNK = 65

REGIONS = ["5UTRseq", "CDSseq", "3UTRseq"]
DEFAULT_MAXLEN = {"5UTRseq": 512, "3UTRseq": 1024, "CDS_CODONS": 700}


class ChannelLayerNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, mask):
        s = self.score(h.transpose(1, 2)).squeeze(-1)
        neg = torch.finfo(s.dtype).min
        s = s.masked_fill(~mask, neg)
        w = torch.softmax(s, dim=1).unsqueeze(1)
        out = (h * w).sum(dim=2)
        has_token = mask.any(dim=1, keepdim=True).to(out.dtype)
        return out * has_token


class ConvBranch(nn.Module):
    def __init__(self, vocab, emb_dim=16, channels=(32, 64, 64), kernel=7, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        layers, in_ch = [], emb_dim
        for i, ch in enumerate(channels):
            d = 2 ** i
            layers += [nn.Conv1d(in_ch, ch, kernel, padding=(kernel - 1) // 2 * d, dilation=d),
                       ChannelLayerNorm(ch), nn.ReLU(), nn.Dropout(dropout)]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.attn = AttnPool(in_ch)
        self.out_dim = in_ch * 3

    def forward(self, x, mask):
        h = self.emb(x).transpose(1, 2)
        h = self.conv(h)
        m = mask.unsqueeze(1)
        h = h.masked_fill(~m, 0.0)
        mean = h.sum(2) / m.sum(2).clamp(min=1)
        mx = torch.nan_to_num(h.masked_fill(~m, float("-inf")).max(2).values, neginf=0.0)
        at = self.attn(h, mask)
        return torch.cat([mx, mean, at], dim=1)


class mRNAStabilityNet(nn.Module):
    def __init__(self, emb_dim=16, channels=(32, 64, 64), kernel=7,
                 dropout=0.3, head_hidden=128, n_extra=N_EXTRA_FEATS):
        super().__init__()
        self.b5   = ConvBranch(NT_SIZE, emb_dim, channels, kernel, dropout)
        self.b3   = ConvBranch(NT_SIZE, emb_dim, channels, kernel, dropout)
        self.bcds = ConvBranch(CODON_SIZE, emb_dim, channels, 5, dropout)
        cat = self.b5.out_dim + self.b3.out_dim + self.bcds.out_dim + n_extra
        self.head = nn.Sequential(
            nn.Linear(cat, head_hidden), nn.LayerNorm(head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.LayerNorm(head_hidden // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, b):
        z = torch.cat([
            self.b5(b["utr5"], b["utr5_mask"]),
            self.b3(b["utr3"], b["utr3_mask"]),
            self.bcds(b["cds"], b["cds_mask"]),
            b["extra"],
        ], dim=1)
        return self.head(z).squeeze(1)


# ── Data utilities ───────────────────────────────────────────────────────────
import os, glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def encode_nt(seq, max_len):
    seq = (seq or "").upper().replace("T", "U")[:max_len]
    ids = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(seq):
        ids[i] = NT_VOCAB.get(ch, NT_VOCAB["N"])
    return ids


def encode_codons(seq, max_codons):
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


def featurize(row):
    u5, cds, u3 = [(row[r] or "").upper().replace("T", "U") for r in REGIONS]
    scal = np.array([
        np.log1p(len(u5)), np.log1p(len(cds)), np.log1p(len(u3)),
        _gc(u5), _gc(cds), _gc(u3),
        np.log1p(u3.count("AUUUA")),
        np.log1p(u5.count("AUG")),
        _gc(cds[:90]) if len(cds) >= 90 else _gc(cds),
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


def load_folds(data_dir):
    folds = {}
    for fp in sorted(glob.glob(os.path.join(data_dir, "training", "*.csv"))):
        fid = int(os.path.basename(fp).replace(".csv", "").split("_")[-1])
        d = pd.read_csv(fp, dtype=str).fillna("")
        d["Label"] = d["Label"].astype(int)
        folds[fid] = d
    return folds


def load_test(data_dir):
    # Try both filenames
    for name in ("test_without_label.csv", "test_with_label.csv"):
        p = os.path.join(data_dir, "test", name)
        if os.path.exists(p):
            return pd.read_csv(p, dtype=str).fillna("")
    raise FileNotFoundError("No test CSV found in " + os.path.join(data_dir, "test"))
