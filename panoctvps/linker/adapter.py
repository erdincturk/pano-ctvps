"""Appearance adapter of the linker: psi(e) = normalize(e + U e + b).

The segmenter embeddings were trained for contrast over time within a single
camera. The adapter is a residual linear map that adapts them to the
comparison across cameras:

    psi(e) = normalize(e_hat + U e_hat + b),   e_hat = e / ||e||

with U in R^{256x256} and b in R^{256} (65,792 parameters), both initialised to
zero, so that psi is the identity on normalised embeddings before training.
It operates on cached embeddings only; the segmenter and tracker are unchanged.

It is trained with a symmetric InfoNCE loss on pairs of detections from
adjacent cameras of the training split (scripts/build_pairs.py,
scripts/train_adapter.py).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 256


class EmbeddingAdapter(nn.Module):
    """Residual linear adapter over frozen embeddings (U = weight, b = bias)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(EMB_DIM, EMB_DIM)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e_hat = F.normalize(e, dim=-1)
        return F.normalize(e_hat + self.linear(e_hat), dim=-1)

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingAdapter":
        """Load an adapter file written by :meth:`save`.

        Also reads files that store the linear map under the key prefix
        ``mlp.0.`` together with ``depth`` = 0.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("depth", 0) != 0:
            raise ValueError(f"{path}: not a linear adapter")
        state = {k.replace("mlp.0.", "linear.", 1): v for k, v in ckpt["state_dict"].items()}
        m = cls()
        m.load_state_dict(state)
        m.eval()
        return m


def info_nce_pair_loss(za: torch.Tensor, zb: torch.Tensor, labels: torch.Tensor,
                       gid_a: torch.Tensor, gid_b: torch.Tensor,
                       temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of detection pairs.

    za/zb: (B, D) adapted (unit-norm) embeddings of the two sides of each pair;
    labels: (B,) 1 = same instance, 0 = different. For every positive pair i,
    the other pairs of the batch are the negatives. Entries whose ground-truth
    instance equals the anchor's (the same instance in another pair of the
    batch) are not used as negatives. Averaged over both directions, A->B and
    B->A.
    """
    pos = labels == 1
    if not bool(pos.any()):
        return za.sum() * 0.0
    eye = torch.eye(len(za), dtype=torch.bool, device=za.device)

    def one_side(z_anchor, z_other, gid_anchor, gid_other):
        sim = z_anchor @ z_other.T / temperature                    # (B, B)
        false_neg = (gid_anchor[:, None] == gid_other[None, :]) & ~eye
        sim = sim.masked_fill(false_neg, float("-inf"))
        idx = torch.arange(len(z_anchor), device=z_anchor.device)
        return F.cross_entropy(sim[pos], idx[pos])

    return 0.5 * (one_side(za, zb, gid_a, gid_b) + one_side(zb, za, gid_b, gid_a))
