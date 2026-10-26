"""Tests for EmbeddingHead (identity embedding MLP)."""

import torch
import pytest

from panoctvps.tracker.embedding_head import EmbeddingHead


class TestEmbeddingHead:
    """Unit tests for the 3-layer MLP embedding head."""

    def test_output_shape(self):
        """Output shape matches input shape for default config."""
        head = EmbeddingHead(input_dim=256, hidden_dim=256, output_dim=256)
        x = torch.randn(10, 200, 256)  # (B*T, Q=200, C=256)
        out = head(x)
        assert out.shape == (10, 200, 256)

    def test_output_shape_single_query(self):
        """Works with single query (edge case)."""
        head = EmbeddingHead()
        x = torch.randn(1, 1, 256)
        out = head(x)
        assert out.shape == (1, 1, 256)

    def test_gradient_flow(self):
        """Gradients flow through all layers."""
        head = EmbeddingHead()
        x = torch.randn(2, 200, 256, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        for layer in head.layers:
            assert layer.weight.grad is not None
            assert layer.weight.grad.abs().sum() > 0

    def test_num_layers(self):
        """Module has exactly 3 linear layers (CTVIS default)."""
        head = EmbeddingHead(num_layers=3)
        assert len(head.layers) == 3
        for layer in head.layers:
            assert isinstance(layer, torch.nn.Linear)

    def test_no_final_activation(self):
        """Last layer has no activation (output is unbounded)."""
        head = EmbeddingHead()
        # With large input, output should be able to go negative (no ReLU)
        x = torch.randn(2, 200, 256) * 10
        out = head(x)
        assert out.min() < 0, "Output should contain negative values (no final ReLU)"

    def test_param_count(self):
        """Parameter count matches expected for 3-layer 256-dim MLP."""
        head = EmbeddingHead(input_dim=256, hidden_dim=256, output_dim=256, num_layers=3)
        n_params = sum(p.numel() for p in head.parameters())
        # 3 layers × (256×256 weights + 256 biases) = 3 × 65,792 = 197,376
        expected = 3 * (256 * 256 + 256)
        assert n_params == expected, f"Expected {expected}, got {n_params}"

    def test_different_dims(self):
        """Works with non-default dimensions."""
        head = EmbeddingHead(input_dim=512, hidden_dim=128, output_dim=64, num_layers=2)
        x = torch.randn(4, 100, 512)
        out = head(x)
        assert out.shape == (4, 100, 64)

    def test_wod_realistic_shapes(self):
        """Test with WOD-realistic shapes: B=2, T=5, Q=200, C=256."""
        head = EmbeddingHead()
        # B=2 clips × T=5 frames = 10 forward passes
        x = torch.randn(10, 200, 256)
        out = head(x)
        assert out.shape == (10, 200, 256)
        assert out.dtype == torch.float32
