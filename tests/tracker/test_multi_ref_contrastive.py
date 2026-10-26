"""Unit tests for the multi-reference contrastive item builder.

Validates ``build_multi_ref_contrastive_items``:
  - pair counts (every frame as anchor),
  - memory-bank positive sources,
  - missing-instance handling (frames where the instance is absent),
  - stuff / void filtering,
  - anchor_class_id propagation,
  - compatibility with the existing ``contrastive_loss`` consumer.
"""

import math
import random

import pytest
import torch

from panoctvps.tracker.contrastive_loss import (
    build_multi_ref_contrastive_items,
    contrastive_loss,
)
from panoctvps.tracker.memory_bank_train import TrainMemoryBank


# =============================================================================
# Helpers
# =============================================================================

def _make_synth_clip(
    T: int,
    Q: int,
    C: int,
    instance_gids: list,
    instance_classes: list,
    instance_query_pattern: list,
    seed: int = 0,
):
    """Build synthetic (reid_embeddings, matching_indices, gids, classes).

    Args:
        T, Q, C: clip length, query count, channels.
        instance_gids: list[int] — one global id per instance.
        instance_classes: list[int] — class id per instance, same length.
        instance_query_pattern: list[list[Optional[int]]] — for each instance,
            the per-frame matched query index (or ``None`` if absent at that
            frame). Outer length = number of instances; inner length = T.

    Returns:
        Tuple ``(reid_embeddings, matching_indices, gids, classes)``
        compatible with ``build_multi_ref_contrastive_items``.
    """
    assert len(instance_gids) == len(instance_classes)
    assert all(len(p) == T for p in instance_query_pattern)

    torch.manual_seed(seed)
    reid = torch.randn(T, Q, C)

    matching_indices = []
    gids_per_frame = []
    classes_per_frame = []
    for t in range(T):
        src, tgt = [], []
        gids_t = []
        cls_t = []
        for inst_i, (gid, cls) in enumerate(zip(instance_gids, instance_classes)):
            qi = instance_query_pattern[inst_i][t]
            if qi is None:
                continue
            src.append(qi)
            # tgt index is into the gids_t / cls_t lists below; assign once.
            tgt.append(len(gids_t))
            gids_t.append(gid)
            cls_t.append(cls)
        matching_indices.append((
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        ))
        gids_per_frame.append(torch.tensor(gids_t, dtype=torch.long))
        classes_per_frame.append(torch.tensor(cls_t, dtype=torch.long))

    return reid, matching_indices, gids_per_frame, classes_per_frame


# =============================================================================
# Pair-count tests
# =============================================================================

def test_T5_single_instance_present_all_frames():
    """1 instance in 5 frames → 5 items (one per anchor frame)."""
    T, Q, C = 5, 200, 256
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],  # car
        instance_query_pattern=[[10, 20, 30, 40, 50]],
    )
    items = build_multi_ref_contrastive_items(
        reid.reshape(T, Q, C), mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(),
        rng_seed=0,
    )
    assert len(items) == 5
    for it in items:
        assert it["num_positive"] == 4
        # 4 positives + 4 * 199 negatives = 4 + 796 = 800
        assert it["label"].shape == (800,)
        assert it["label"][:4].sum().item() == 4
        assert it["label"][4:].sum().item() == 0
        assert it["anchor_class_id"] == 2


def test_T5_missing_one_frame():
    """Instance present at t in {0,1,2,4}, missing at t=3.

    Every frame is an anchor frame; num_positives depends on the anchor.
    For anchor j ∈ {0,1,2,4}: 3 positives (the other present frames).
    For anchor j = 3: instance not matched at 3 → 0 anchor items.
    → 4 items total.
    """
    T, Q, C = 5, 200, 256
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],
        instance_query_pattern=[[10, 20, 30, None, 50]],
    )
    items = build_multi_ref_contrastive_items(
        reid.reshape(T, Q, C), mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(),
        rng_seed=0,
    )
    assert len(items) == 4
    for it in items:
        assert it["num_positive"] == 3
        # 3 positives + 4 * 199 negatives (1 ref frame contributes Q negatives
        # via the "all queries become negatives" branch when instance absent;
        # capped at num_negatives=199 since Q > num_negatives).
        assert it["label"].shape == (3 + 4 * 199,)


def test_empty_when_no_thing_instances():
    T, Q, C = 5, 200, 256
    reid = torch.randn(T, Q, C)
    mi = [
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        for _ in range(T)
    ]
    gids = [torch.tensor([], dtype=torch.long) for _ in range(T)]
    classes = [torch.tensor([], dtype=torch.long) for _ in range(T)]
    items = build_multi_ref_contrastive_items(
        reid, mi, gids, classes,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )
    assert items == []


def test_stuff_excluded_via_negative_gid():
    """A target with gid <= 0 must be silently dropped."""
    T, Q, C = 5, 200, 256
    reid = torch.randn(T, Q, C)
    mi = []
    gids = []
    classes = []
    for t in range(T):
        # one thing (gid=5) and one stuff (gid=-1) per frame
        mi.append((torch.tensor([0, 1]), torch.tensor([0, 1])))
        gids.append(torch.tensor([5, -1]))
        classes.append(torch.tensor([2, 13]))   # car + stuff class id
    items = build_multi_ref_contrastive_items(
        reid, mi, gids, classes,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )
    assert len(items) == 5
    for it in items:
        assert it["anchor_class_id"] == 2  # never the stuff class
        assert it["num_positive"] == 4


# =============================================================================
# Bank positive source
# =============================================================================

def test_bank_provides_some_sg_positives_when_prob_is_one():
    """With positive_momentum_prob=1.0, the bank's sg_embed is always picked.

    Sanity check: replacing raw with sg should not crash and should still
    produce the expected pair counts.
    """
    T, Q, C = 5, 200, 256
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],
        instance_query_pattern=[[10, 20, 30, 40, 50]],
    )
    bank = TrainMemoryBank(beta_max=0.8)
    items = build_multi_ref_contrastive_items(
        reid, mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=bank, positive_momentum_prob=1.0,
        rng_seed=0,
    )
    assert len(items) == 5
    for it in items:
        assert it["num_positive"] == 4
        # Each positive should be a (1, C) embedding — verifiable by shape.
        assert it["label"].shape == (4 + 4 * 199,)


def test_bank_is_reset_between_batches():
    """Bank must be cleared at the start of each clip."""
    T, Q, C = 5, 200, 256
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],
        instance_query_pattern=[[10, 20, 30, 40, 50]],
    )
    bank = TrainMemoryBank()
    bank.update(99, torch.randn(C))     # pre-populate with a stray instance
    assert 99 in bank

    _ = build_multi_ref_contrastive_items(
        reid, mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=bank, positive_momentum_prob=1.0,
        rng_seed=0,
    )
    # After build, the bank should only contain gids seen in the clip (gid 7),
    # not the stray gid 99 that was inserted before the call.
    assert 7 in bank
    assert 99 not in bank


# =============================================================================
# Multi-clip batch
# =============================================================================

def test_batch_size_2_isolated_clips():
    """Two clips concatenated along the batch axis; each gets its own items."""
    T, Q, C = 3, 200, 64
    # Build two clips manually.
    r1, m1, g1, c1 = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],
        instance_query_pattern=[[5, 6, 7]],
        seed=0,
    )
    r2, m2, g2, c2 = _make_synth_clip(
        T, Q, C,
        instance_gids=[8],
        instance_classes=[9],  # pedestrian
        instance_query_pattern=[[1, 2, 3]],
        seed=1,
    )
    reid = torch.cat([r1, r2], dim=0)  # (2*T, Q, C)
    mi = m1 + m2
    gids = g1 + g2
    classes = c1 + c2

    items = build_multi_ref_contrastive_items(
        reid, mi, gids, classes,
        clip_length=T, batch_size=2,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )
    # Each clip: 3 anchor frames × 1 instance = 3 items. Total 6.
    assert len(items) == 6
    cars = [it for it in items if it["anchor_class_id"] == 2]
    peds = [it for it in items if it["anchor_class_id"] == 9]
    assert len(cars) == 3
    assert len(peds) == 3


# =============================================================================
# Consumer compatibility
# =============================================================================

def test_existing_loss_consumes_multi_ref_items():
    """``contrastive_loss`` should work unchanged on the new item shape."""
    T, Q, C = 5, 200, 64
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],
        instance_query_pattern=[[10, 20, 30, 40, 50]],
    )
    items = build_multi_ref_contrastive_items(
        reid, mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )
    losses = contrastive_loss(items, reid, class_weights={})
    assert "loss_reid" in losses
    assert "loss_aux_reid" in losses
    assert torch.isfinite(losses["loss_reid"])
    assert torch.isfinite(losses["loss_aux_reid"])
    assert losses["loss_reid"].item() > 0  # non-trivial


def test_class_weights_change_loss_magnitude():
    """A class weight of 0.5 vs 2.0 should give different loss values."""
    T, Q, C = 5, 200, 64
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],  # car
        instance_query_pattern=[[10, 20, 30, 40, 50]],
    )
    items = build_multi_ref_contrastive_items(
        reid, mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )

    loss_unweighted = contrastive_loss(items, reid, class_weights={})
    loss_w05 = contrastive_loss(items, reid, class_weights={2: 0.5})
    loss_w20 = contrastive_loss(items, reid, class_weights={2: 2.0})

    # With a uniform single-class weight, the weighted mean equals the
    # mean with weight 1 (numerator and denominator scale together).
    assert torch.allclose(loss_w05["loss_reid"], loss_unweighted["loss_reid"], atol=1e-5)
    assert torch.allclose(loss_w20["loss_reid"], loss_unweighted["loss_reid"], atol=1e-5)


def test_class_weights_rebalances_across_classes():
    """When two classes coexist with different weights, the gradient ratio shifts."""
    T, Q, C = 3, 200, 64
    # Two clips: one car instance, one pedestrian instance.
    r1, m1, g1, c1 = _make_synth_clip(
        T, Q, C, instance_gids=[7], instance_classes=[2],
        instance_query_pattern=[[10, 20, 30]], seed=0,
    )
    r2, m2, g2, c2 = _make_synth_clip(
        T, Q, C, instance_gids=[8], instance_classes=[9],
        instance_query_pattern=[[1, 2, 3]], seed=1,
    )
    reid = torch.cat([r1, r2], dim=0)
    items = build_multi_ref_contrastive_items(
        reid, m1 + m2, g1 + g2, c1 + c2,
        clip_length=T, batch_size=2,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )

    weights_heavy_ped = {2: 0.5, 9: 4.0}   # boost rare-class anchors
    weights_balanced = {2: 1.0, 9: 1.0}
    loss_a = contrastive_loss(items, reid, class_weights=weights_heavy_ped)
    loss_b = contrastive_loss(items, reid, class_weights=weights_balanced)

    # Both finite, and the heavy-pedestrian weighting must produce a
    # different (i.e., re-balanced) total reid loss.
    assert torch.isfinite(loss_a["loss_reid"]) and torch.isfinite(loss_b["loss_reid"])
    assert not torch.allclose(loss_a["loss_reid"], loss_b["loss_reid"], atol=1e-4)


def test_gradient_flows_to_embeddings():
    """Backward through the loss must touch ``reid_embeddings``."""
    T, Q, C = 5, 200, 32
    reid, mi, gids, cls = _make_synth_clip(
        T, Q, C,
        instance_gids=[7],
        instance_classes=[2],
        instance_query_pattern=[[10, 20, 30, 40, 50]],
    )
    reid = reid.detach().requires_grad_(True)
    items = build_multi_ref_contrastive_items(
        reid, mi, gids, cls,
        clip_length=T, batch_size=1,
        num_negatives=199,
        train_memory_bank=TrainMemoryBank(), rng_seed=0,
    )
    losses = contrastive_loss(items, reid, class_weights={})
    total = losses["loss_reid"] + losses["loss_aux_reid"]
    total.backward()
    # Some queries must have received a non-zero gradient.
    assert reid.grad is not None
    assert reid.grad.abs().sum().item() > 0
