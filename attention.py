"""Trainable cross-modal fusion layers (torch).

Unlike the previous implementation - which used a *fixed*, randomly initialised
projection as a stand-in for attention - the gating network here is a genuine
learnable module: the per-modality weights are produced by an MLP over the
projected modality representations and are optimised end-to-end with the
classification objective.

Two fusion strategies are provided:

* :class:`ConcatFusion` - naive concatenation (the base-paper baseline).
* :class:`GatedAttentionFusion` - per-sample, learned modality weighting.
"""

from config import ATTENTION_HIDDEN, MODEL_DIM  # set OpenMP env before torch

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def forward(self, xs, weights=None):
        return torch.cat(xs, dim=1)


class GatedAttentionFusion(nn.Module):
    """Learned gated fusion over N modalities.

    Each modality is linearly projected to a shared ``model_dim``; an MLP scores
    the concatenation of projected modalities; a softmax turns those scores into
    per-sample modality weights; the fused vector is the weighted sum of the
    projected modalities (Optionally weighted concatenation).
    """

    def __init__(self, dims, model_dim=MODEL_DIM, hidden=ATTENTION_HIDDEN, concat_out=False):
        super().__init__()
        self.dims = list(dims)
        self.n_modalities = len(dims)
        self.model_dim = model_dim
        self.concat_out = concat_out

        self.projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, model_dim), nn.ReLU(), nn.LayerNorm(model_dim))
             for d in dims]
        )
        self.gate = nn.Sequential(
            nn.Linear(model_dim * self.n_modalities, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_modalities),
        )

    def forward(self, xs, return_weights=False):
        projs = [proj(x) for proj, x in zip(self.projections, xs)]
        context = torch.cat(projs, dim=1)
        logits = self.gate(context)
        weights = torch.softmax(logits, dim=1)  # (B, N)
        stacked = torch.stack(projs, dim=1)      # (B, N, model_dim)
        weighted = stacked * weights.unsqueeze(-1)
        if self.concat_out:
            fused = weighted.reshape(xs[0].shape[0], -1)
        else:
            fused = weighted.sum(dim=1)
        if return_weights:
            return fused, weights
        return fused


class FusionNet(nn.Module):
    """Attention fusion + classification head, trained end-to-end."""

    def __init__(self, dims, model_dim=MODEL_DIM, hidden=ATTENTION_HIDDEN,
                 dropout=0.2, modality_dropout=0.1):
        super().__init__()
        self.fusion = GatedAttentionFusion(dims, model_dim=model_dim, hidden=hidden)
        self.modality_dropout = modality_dropout
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def _maybe_dropout(self, xs):
        if not self.training or self.modality_dropout <= 0:
            return xs
        out = []
        for x in xs:
            keep = (torch.rand(x.shape[0], 1, device=x.device) > self.modality_dropout).float()
            out.append(x * keep)
        return out

    def forward(self, xs, return_weights=False):
        xs = self._maybe_dropout(xs)
        fused, weights = self.fusion(xs, return_weights=True)
        logits = self.head(fused).squeeze(-1)
        if return_weights:
            return logits, weights
        return logits

    @torch.no_grad()
    def embed(self, xs):
        self.eval()
        fused, _ = self.fusion(xs, return_weights=True)
        return fused


if __name__ == "__main__":
    torch.manual_seed(0)
    dims = [24, 31, 6]
    xs = [torch.randn(8, d) for d in dims]
    fusion = GatedAttentionFusion(dims)
    fused, w = fusion(xs, return_weights=True)
    print("fused:", tuple(fused.shape), "weights:", tuple(w.shape))
    print("weight row sums:", w.sum(dim=1).detach().numpy().round(3))
    net = FusionNet(dims)
    logits, w = net(xs, return_weights=True)
    print("net logits:", tuple(logits.shape))
