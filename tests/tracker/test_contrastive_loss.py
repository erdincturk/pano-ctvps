"""Tests for the contrastive loss (InfoNCE + auxiliary cosine term)."""

import torch
import pytest

from panoctvps.data.class_weights import WOD_THING_ANCHOR_WEIGHTS
from panoctvps.tracker.contrastive_loss import (
    build_multi_ref_contrastive_items,
    contrastive_loss,
)
from panoctvps.tracker.memory_bank_train import TrainMemoryBank


def _make_matching_indices(BT, Q, N_per_frame):
    """Each frame matches N_per_frame queries to N_per_frame targets."""
    indices = []
    for _ in range(BT):
        n = min(N_per_frame, Q)
        indices.append((torch.arange(n), torch.arange(n)))
    return indices


def _make_global_ids(BT, N_per_frame, num_things=5):
    """Global instance IDs 1..num_things in every frame; the rest are stuff (-1)."""
    ids = []
    for _ in range(BT):
        gids = torch.full((N_per_frame,), -1, dtype=torch.int64)
        for i in range(min(num_things, N_per_frame)):
            gids[i] = i + 1
        ids.append(gids)
    return ids


def _items(embeds, T, N_per_frame=10, num_things=5, num_negatives=199):
    B = embeds.shape[0] // T
    Q = embeds.shape[1]
    indices = _make_matching_indices(B * T, Q, N_per_frame)
    gids = _make_global_ids(B * T, N_per_frame, num_things)
    classes = [torch.full((N_per_frame,), 2, dtype=torch.int64) for _ in range(B * T)]
    return build_multi_ref_contrastive_items(
        embeds, indices, gids, classes, clip_length=T, batch_size=B,
        train_memory_bank=TrainMemoryBank(), num_negatives=num_negatives,
        rng_seed=0,
    )


class TestContrastiveLoss:
    def test_zero_loss_no_items(self):
        dummy = torch.randn(1)
        losses = contrastive_loss([], dummy, WOD_THING_ANCHOR_WEIGHTS)
        assert losses["loss_reid"].item() == 0.0
        assert losses["loss_aux_reid"].item() == 0.0

    def test_loss_is_positive(self):
        embeds = torch.randn(3, 200, 256)
        items = _items(embeds, T=3)
        assert len(items) > 0
        losses = contrastive_loss(items, embeds, WOD_THING_ANCHOR_WEIGHTS)
        assert losses["loss_reid"].item() > 0
        assert losses["loss_aux_reid"].item() >= 0

    def test_loss_gradient_flow(self):
        embeds = torch.randn(2, 50, 256, requires_grad=True)
        items = _items(embeds, T=2, num_things=3, num_negatives=49)
        losses = contrastive_loss(items, embeds, WOD_THING_ANCHOR_WEIGHTS)
        (losses["loss_reid"] + losses["loss_aux_reid"]).backward()
        assert embeds.grad is not None
        assert embeds.grad.abs().sum() > 0

    def test_loss_finite_with_extreme_dot_products(self):
        """The masked logsumexp stays finite for very large dot products."""
        torch.manual_seed(0)
        embeds = torch.randn(2, 100, 256) * 50.0
        items = _items(embeds, T=2, N_per_frame=5, num_things=3, num_negatives=50)
        losses = contrastive_loss(items, embeds, WOD_THING_ANCHOR_WEIGHTS)
        assert torch.isfinite(losses["loss_reid"])
        assert torch.isfinite(losses["loss_aux_reid"])

    def test_perfect_matching_low_aux_loss(self):
        """Identical embeddings of the same instance give a lower aux loss."""
        embeds = torch.randn(2, 50, 256)
        for q in range(5):
            embeds[1, q] = embeds[0, q].clone()
        embeds_random = torch.randn(2, 50, 256)
        loss_perfect = contrastive_loss(_items(embeds, T=2, num_negatives=49),
                                        embeds, WOD_THING_ANCHOR_WEIGHTS)
        loss_random = contrastive_loss(_items(embeds_random, T=2, num_negatives=49),
                                       embeds_random, WOD_THING_ANCHOR_WEIGHTS)
        assert loss_perfect["loss_aux_reid"].item() < loss_random["loss_aux_reid"].item()
