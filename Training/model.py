"""
Codon-aware multi-branch CNN for mRNA stability prediction.

  - 5'UTR / 3'UTR : nucleotide-level conv towers
  - CDS           : CODON-level conv tower (vocab of 64 codons) -> codon
                    optimality signal
  - engineered features (lengths, GC, motif counts, 64-dim codon frequency)

All branches use masked global (max + mean + attention) pooling, are
concatenated with the engineered features, and passed to an MLP head.

Design notes (v2):
  * Attention pooling zeroes fully-empty (all-padding) rows and uses a finite
    mask fill, so empty 5'UTR / 3'UTR contribute exactly 0 with no NaN grads.
  * LayerNorm (over channels / features) instead of BatchNorm, so padded
    timesteps and batch composition do not distort the normalization.
  * Lighter capacity (emb 16, channels 32/64/64) to reduce overfitting on the
    ~3k-sample dataset.
"""
from __future__ import annotations
import torch
import torch.nn as nn

# nucleotide vocab: 0 PAD, A C G U/T N
NT_VOCAB = {"PAD": 0, "A": 1, "C": 2, "G": 3, "U": 4, "T": 4, "N": 5}
NT_SIZE = 6
CODON_SIZE = 66          # 0 PAD, 1..64 codons, 65 unknown
N_EXTRA_FEATS = 11 + 64  # must match data.py


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim for (B, C, L) tensors.

    Normalizes each position over its channels, independently of the batch and
    of padded timesteps -- unlike BatchNorm1d, which pools statistics across
    the batch and across padded positions.
    """
    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):                       # (B, C, L)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class AttnPool(nn.Module):
    """Masked attention pooling -> single vector. Empty rows pool to 0."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, mask):                 # h: (B,C,L)  mask: (B,L) bool
        s = self.score(h.transpose(1, 2)).squeeze(-1)        # (B,L)
        neg = torch.finfo(s.dtype).min                       # finite, not -inf
        s = s.masked_fill(~mask, neg)
        w = torch.softmax(s, dim=1).unsqueeze(1)             # (B,1,L)
        out = (h * w).sum(dim=2)                             # (B,C)
        has_token = mask.any(dim=1, keepdim=True).to(out.dtype)
        return out * has_token                               # zero empty rows


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
        self.out_dim = in_ch * 3  # max + mean + attn

    def forward(self, x, mask):
        h = self.emb(x).transpose(1, 2)
        h = self.conv(h)
        m = mask.unsqueeze(1)                                # (B,1,L)
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
