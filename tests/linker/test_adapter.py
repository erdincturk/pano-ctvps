"""Tests for panoctvps/linker/adapter.py — residual identity, loss, save/load."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from panoctvps.linker.adapter import EMB_DIM, EmbeddingAdapter, info_nce_pair_loss


def _emb(n, seed=0, scale=2.5):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(0, scale, (n, EMB_DIM)).astype(np.float32))


class TestAdapter:
    def test_identity_at_init(self):
        # U, b zero-initialised -> psi(e) == normalized(e) exactly
        m = EmbeddingAdapter().eval()
        e = _emb(16)
        with torch.no_grad():
            z = m(e)
        cos = (z * F.normalize(e, dim=-1)).sum(-1)
        assert torch.allclose(cos, torch.ones(16), atol=1e-6)

    def test_parameter_count(self):
        assert sum(p.numel() for p in EmbeddingAdapter().parameters()) == 65_792

    def test_output_unit_norm(self):
        m = EmbeddingAdapter()
        for p in m.linear.parameters():         # un-zero the residual
            torch.nn.init.normal_(p, 0, 0.1)
        with torch.no_grad():
            z = m(_emb(8, seed=1))
        assert torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-5)

    def test_save_load_round_trip(self, tmp_path):
        m = EmbeddingAdapter()
        for p in m.parameters():
            torch.nn.init.normal_(p, 0, 0.05)
        path = tmp_path / "a.pt"
        m.save(path)
        m2 = EmbeddingAdapter.load(path)
        e = _emb(4, seed=2)
        with torch.no_grad():
            assert torch.allclose(m.eval()(e), m2(e), atol=1e-7)

    def test_load_mlp_prefixed_file(self, tmp_path):
        # files storing the linear map as "mlp.0.*" with depth 0 load identically
        m = EmbeddingAdapter()
        for p in m.parameters():
            torch.nn.init.normal_(p, 0, 0.05)
        sd = {k.replace("linear.", "mlp.0."): v for k, v in m.state_dict().items()}
        path = tmp_path / "b.pt"
        torch.save({"depth": 0, "hidden": 256, "state_dict": sd}, path)
        m2 = EmbeddingAdapter.load(path)
        e = _emb(4, seed=3)
        with torch.no_grad():
            assert torch.equal(m.eval()(e), m2(e))


class TestLoss:
    def test_training_separates_synthetic_objects(self):
        # 8 synthetic objects; each pair = two noisy views of one object
        # (pos) or views of two objects (neg). Loss must drive same-object
        # cosine above different-object cosine.
        rng = np.random.default_rng(3)
        protos = torch.from_numpy(rng.normal(0, 1, (8, EMB_DIM)).astype(np.float32))

        def view(g):
            return protos[g] + torch.from_numpy(
                rng.normal(0, 1.2, EMB_DIM).astype(np.float32))

        gA = torch.from_numpy(rng.integers(0, 8, 256))
        same = torch.from_numpy(rng.random(256) < 0.5)
        gB = torch.where(same, gA, (gA + 1 + torch.from_numpy(rng.integers(0, 7, 256))) % 8)
        eA = torch.stack([view(int(g)) for g in gA])
        eB = torch.stack([view(int(g)) for g in gB])
        lab = (gA == gB).long()

        m = EmbeddingAdapter()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(60):
            loss = info_nce_pair_loss(m(eA), m(eB), lab, gA + 1, gB + 1)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cos = (m(eA) * m(eB)).sum(-1)
        assert cos[lab == 1].mean() > cos[lab == 0].mean() + 0.2

    def test_no_positives_gives_zero_loss(self):
        eA, eB = _emb(8, 4), _emb(8, 5)
        lab = torch.zeros(8, dtype=torch.long)
        g = torch.arange(8) + 1
        m = EmbeddingAdapter()
        loss = info_nce_pair_loss(m(eA), m(eB), lab, g, g + 100)
        assert float(loss) == 0.0

    def test_false_negative_masking_finite(self):
        # Two pairs of the SAME object: each other's columns must be masked,
        # loss stays finite and the diagonal target is preserved.
        eA, eB = _emb(4, 6), _emb(4, 7)
        lab = torch.tensor([1, 1, 0, 0])
        gA = torch.tensor([5, 5, 6, 7])
        gB = torch.tensor([5, 5, 8, 9])
        m = EmbeddingAdapter()
        loss = info_nce_pair_loss(m(eA), m(eB), lab, gA, gB)
        assert torch.isfinite(loss)
