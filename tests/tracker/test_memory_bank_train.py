"""Unit tests for the training-time memory bank.

Validates the CTVIS SimpleTrainMemory port (``panoctvps/tracker/memory_bank_train.py``):
seeding, β clamping at frame 2+, mean-cosine update, multi-instance isolation,
reset, fp32 numerical safety.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from panoctvps.tracker.memory_bank_train import TrainMemoryBank


# =============================================================================
# Construction and validation
# =============================================================================

def test_construction_defaults():
    bank = TrainMemoryBank()
    assert bank.beta_min == 0.0
    assert bank.beta_max == 0.8
    assert bank.maximum_cache == 10
    assert bank.store_dtype == torch.float32
    assert len(bank) == 0


def test_invalid_beta_range_raises():
    with pytest.raises(ValueError):
        TrainMemoryBank(beta_min=0.5, beta_max=0.3)
    with pytest.raises(ValueError):
        TrainMemoryBank(beta_min=-0.1, beta_max=0.8)
    with pytest.raises(ValueError):
        TrainMemoryBank(beta_min=0.0, beta_max=1.1)


def test_rejects_non_thing_gid():
    bank = TrainMemoryBank()
    with pytest.raises(ValueError):
        bank.update(0, torch.zeros(256))
    with pytest.raises(ValueError):
        bank.update(-1, torch.zeros(256))


# =============================================================================
# Empty state
# =============================================================================

def test_empty_bank_queries():
    bank = TrainMemoryBank()
    assert bank.get_sg_embed(1) is None
    assert not bank.contains(1)
    assert 1 not in bank
    assert bank.exist_frames(1) == 0


# =============================================================================
# Seeding (first valid update)
# =============================================================================

def test_seed_with_1d_embed():
    bank = TrainMemoryBank()
    e = torch.tensor([1.0, 2.0, 3.0, 4.0])
    bank.update(7, e)
    sg = bank.get_sg_embed(7)
    assert sg.shape == (1, 4)
    assert torch.allclose(sg.squeeze(0), e)
    assert bank.exist_frames(7) == 1
    assert bank.contains(7)


def test_seed_with_2d_embed():
    bank = TrainMemoryBank()
    e = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    bank.update(7, e)
    sg = bank.get_sg_embed(7)
    assert sg.shape == (1, 4)
    assert torch.allclose(sg, e)


def test_seed_stores_in_fp32_even_if_input_is_fp16():
    bank = TrainMemoryBank()
    e = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
    bank.update(7, e)
    sg = bank.get_sg_embed(7)
    assert sg.dtype == torch.float32


# =============================================================================
# Two-update β math
# =============================================================================

def test_two_updates_identical_embed_beta_clamps_to_max():
    """cos(e, e) = 1; without clamp β=1 → sg=new; with beta_max=0.8 → blend."""
    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    e = torch.tensor([1.0, 0.0, 0.0, 0.0])
    bank.update(1, e)
    bank.update(1, e)

    sg = bank.get_sg_embed(1)
    # With identical embeds and β clamped to 0.8:
    #   sg = (1 - 0.8) * seed + 0.8 * new = (0.2 + 0.8) * e = e
    # (still equals e since seed == new). But exist_frames bumped to 2.
    assert torch.allclose(sg.squeeze(0), e)
    assert bank.exist_frames(1) == 2


def test_two_updates_orthogonal_embed_beta_floor():
    """cos(e1, e2) = 0; β = max(0, 0) = 0 → sg unchanged from seed."""
    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    bank.update(1, e1)
    bank.update(1, e2)

    sg = bank.get_sg_embed(1)
    # With β=0: sg = 1 * seed + 0 * new = seed = e1.
    assert torch.allclose(sg.squeeze(0), e1)


def test_two_updates_negative_cos_beta_clamps_to_zero():
    """cos(e1, -e1) = -1; β = max(0, -1) = 0."""
    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    e2 = -e1
    bank.update(1, e1)
    bank.update(1, e2)

    sg = bank.get_sg_embed(1)
    assert torch.allclose(sg.squeeze(0), e1)


def test_two_updates_partial_overlap_beta_in_range():
    """cos = 0.5 → β = 0.5 → sg = 0.5 * seed + 0.5 * new."""
    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    e1 = F.normalize(torch.tensor([1.0, 0.0]), dim=0)
    # Construct e2 with exactly 0.5 cosine to e1.
    e2 = F.normalize(torch.tensor([0.5, math.sqrt(0.75)]), dim=0)

    bank.update(1, e1)
    bank.update(1, e2)
    sg = bank.get_sg_embed(1)

    expected_beta = 0.5
    expected = (1 - expected_beta) * e1 + expected_beta * e2
    assert torch.allclose(sg.squeeze(0), expected, atol=1e-6)


# =============================================================================
# Three-update mean cosine
# =============================================================================

def test_three_updates_uses_mean_cosine_over_past():
    """At frame 3, β = mean(cos(new, e1), cos(new, e2)), clamped."""
    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    e1 = F.normalize(torch.tensor([1.0, 0.0]), dim=0)
    e2 = F.normalize(torch.tensor([0.0, 1.0]), dim=0)
    e3 = F.normalize(torch.tensor([1.0, 1.0]), dim=0)  # cos with each = 1/sqrt(2)

    bank.update(1, e1)
    bank.update(1, e2)
    # After update 2: cos(e2, e1) = 0 → β = 0 → sg = e1.
    sg_after_2 = bank.get_sg_embed(1).squeeze(0).clone()
    assert torch.allclose(sg_after_2, e1, atol=1e-6)

    bank.update(1, e3)
    # mean cos to past [e1, e2] = (1/sqrt(2) + 1/sqrt(2)) / 2 = 1/sqrt(2) ≈ 0.707
    expected_beta = 1.0 / math.sqrt(2.0)
    expected_sg = (1 - expected_beta) * sg_after_2 + expected_beta * e3
    sg_after_3 = bank.get_sg_embed(1).squeeze(0)
    assert torch.allclose(sg_after_3, expected_sg, atol=1e-6)
    assert bank.exist_frames(1) == 3


# =============================================================================
# Multi-instance isolation
# =============================================================================

def test_multiple_instances_independent():
    bank = TrainMemoryBank()
    e_a = torch.tensor([1.0, 0.0])
    e_b = torch.tensor([0.0, 1.0])
    bank.update(1, e_a)
    bank.update(2, e_b)
    assert torch.allclose(bank.get_sg_embed(1).squeeze(0), e_a)
    assert torch.allclose(bank.get_sg_embed(2).squeeze(0), e_b)
    assert len(bank) == 2
    assert set(bank.gids) == {1, 2}


# =============================================================================
# Reset
# =============================================================================

def test_reset_clears_all_state():
    bank = TrainMemoryBank()
    bank.update(1, torch.tensor([1.0, 0.0]))
    bank.update(2, torch.tensor([0.0, 1.0]))
    assert len(bank) == 2
    bank.reset()
    assert len(bank) == 0
    assert not bank.contains(1)
    assert bank.get_sg_embed(1) is None


# =============================================================================
# AMP / fp16 numerical safety
# =============================================================================

def test_update_with_fp16_input_stays_fp32_internally():
    """Caller may pass fp16 under AMP; bank stores in fp32, sg_embed is fp32."""
    bank = TrainMemoryBank()
    e1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float16)
    e2 = torch.tensor([0.6, 0.8, 0.0, 0.0], dtype=torch.float16)
    bank.update(5, e1)
    bank.update(5, e2)
    sg = bank.get_sg_embed(5)
    assert sg.dtype == torch.float32
    # Result direction should match the fp32 reference computation.
    expected_beta = 0.6  # cos(e1, e2) = 0.6 normalised
    expected = (1 - expected_beta) * e1.float() + expected_beta * e2.float()
    assert torch.allclose(sg.squeeze(0), expected, atol=1e-3)


# =============================================================================
# T=5 trajectory
# =============================================================================

def test_T5_clip_typical_trajectory_keeps_seed_alive():
    """Same instance across T=5 frames with high but non-1 cosine.

    With β_max = 0.8 the memory embedding stays close to both the first and
    the last embedding of the clip.
    """
    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    rng = torch.Generator().manual_seed(0)
    base = F.normalize(torch.randn(256, generator=rng), dim=0)

    seeds_log = []
    for _ in range(5):
        # Per-frame small perturbation that yields high cosine to base.
        noise = 0.05 * F.normalize(torch.randn(256, generator=rng), dim=0)
        e = F.normalize(base + noise, dim=0)
        seeds_log.append(e)

    for e in seeds_log:
        bank.update(1, e)

    sg = bank.get_sg_embed(1).squeeze(0)
    # The seed (first frame embed) should still contribute meaningfully.
    seed_cos = F.cosine_similarity(sg.unsqueeze(0), seeds_log[0].unsqueeze(0)).item()
    last_cos = F.cosine_similarity(sg.unsqueeze(0), seeds_log[-1].unsqueeze(0)).item()
    assert seed_cos > 0.95, f"Seed contribution lost: cos(sg, seed)={seed_cos:.3f}"
    assert last_cos > 0.95, f"Latest contribution lost: cos(sg, last)={last_cos:.3f}"
    assert bank.exist_frames(1) == 5
