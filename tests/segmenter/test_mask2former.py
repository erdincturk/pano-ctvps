"""Tests for Mask2Former wrapper helpers.

These tests target the helpers that operate on `model.named_parameters()` and
checkpoints, so they need only a small dummy module that mimics the HF
parameter naming (`model.pixel_level_module.encoder.*` for the backbone,
everything else for the decoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from panoctvps.segmenter.mask2former import (
    freeze_backbone,
    load_checkpoint,
)


class _Encoder(nn.Module):
    """Stand-in for the Swin backbone — two linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(8, 8)
        self.layer2 = nn.Linear(8, 8)


class _PixelLevelModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()           # → backbone params
        self.decoder = nn.Linear(8, 8)      # pixel-decoder, NOT a backbone param


class _InnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pixel_level_module = _PixelLevelModule()
        self.transformer_module = nn.Linear(8, 8)


class _FakeMask2Former(nn.Module):
    """Mimics the HF Mask2Former parameter-name hierarchy without loading
    the real model. Parameter names produced:

      - model.pixel_level_module.encoder.layer1.{weight,bias}      ← backbone
      - model.pixel_level_module.encoder.layer2.{weight,bias}      ← backbone
      - model.pixel_level_module.decoder.{weight,bias}             ← decoder
      - model.transformer_module.{weight,bias}                     ← decoder
      - class_predictor.{weight,bias}                              ← decoder
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = _InnerModel()
        self.class_predictor = nn.Linear(8, 8)


@pytest.fixture
def fake_model() -> _FakeMask2Former:
    torch.manual_seed(0)
    return _FakeMask2Former()


# ── freeze_backbone ───────────────────────────────────────────────────────


class TestFreezeBackbone:
    def test_freezes_encoder_only(self, fake_model: _FakeMask2Former) -> None:
        freeze_backbone(fake_model)
        for name, p in fake_model.named_parameters():
            if "model.pixel_level_module.encoder" in name:
                assert not p.requires_grad, f"backbone param {name} should be frozen"
            else:
                assert p.requires_grad, f"non-backbone param {name} should stay trainable"

    def test_returns_correct_numel(self, fake_model: _FakeMask2Former) -> None:
        # Encoder: 2 × Linear(8, 8) = 2 × (8*8 + 8) = 144 params.
        expected = sum(
            p.numel()
            for name, p in fake_model.named_parameters()
            if "model.pixel_level_module.encoder" in name
        )
        assert freeze_backbone(fake_model) == expected
        assert expected == 144

    def test_idempotent(self, fake_model: _FakeMask2Former) -> None:
        # Calling twice should yield the same count and leave state unchanged.
        first = freeze_backbone(fake_model)
        second = freeze_backbone(fake_model)
        assert first == second


# ── load_checkpoint ───────────────────────────────────────────────────────


class TestLoadCheckpoint:
    def test_round_trip_restores_weights(
        self, fake_model: _FakeMask2Former, tmp_path,
    ) -> None:
        ckpt_path = tmp_path / "ckpt.pth"
        # Snapshot the state we want to recover, then save the checkpoint.
        saved_state = {k: v.clone() for k, v in fake_model.state_dict().items()}
        torch.save({"model": fake_model.state_dict(), "iteration": 12345}, ckpt_path)

        # Build a fresh model with different weights and load into it.
        loaded = _FakeMask2Former()
        with torch.no_grad():
            for p in loaded.parameters():
                p.add_(7.0)            # diverge from saved state

        iteration = load_checkpoint(loaded, str(ckpt_path))
        assert iteration == 12345
        for name, p in loaded.named_parameters():
            assert torch.allclose(p, saved_state[name]), (
                f"{name}: loaded weights should match saved state"
            )

    def test_missing_iteration_returns_zero(
        self, fake_model: _FakeMask2Former, tmp_path,
    ) -> None:
        ckpt_path = tmp_path / "ckpt_no_iter.pth"
        torch.save({"model": fake_model.state_dict()}, ckpt_path)
        assert load_checkpoint(_FakeMask2Former(), str(ckpt_path)) == 0
