"""Tests for panoctvps/linker/scorer.compute_pair_cosines — the cosine cue.

Uses 256-d float32 embeddings with non-unit norms (the cache stores raw
embeddings): the function must normalise before the dot product. Also covers
caches without embeddings and pairs without a shared frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from panoctvps.linker.scorer import COSINE_KEYS, compute_pair_cosines

REC_DTYPE = np.dtype([("camera_id", "i1"), ("timestamp", "i8"),
                      ("tracker_id_deployed", "i4"), ("class_id", "i1")])


def _records(rows):
    arr = np.zeros(len(rows), dtype=REC_DTYPE)
    for i, r in enumerate(rows):
        arr[i] = r
    return arr


def _unit(axis: int, scale: float = 1.0) -> np.ndarray:
    v = np.zeros(256, dtype=np.float32)
    v[axis] = scale
    return v


def _pair(u, v):
    return frozenset({u, v})


class TestComputePairCosines:
    def test_normalises_before_dot(self):
        # Parallel embeddings with very different raw norms must give cos 1.0;
        # orthogonal give 0.0; antiparallel give -1.0.
        records = _records([(1, 100, 1, 2), (2, 100, 1, 2),
                            (1, 200, 1, 2), (2, 200, 1, 2)])
        emb = np.stack([_unit(0, 3.7), _unit(0, 0.9),     # ts=100: parallel
                        _unit(1, 2.1), _unit(2, 1.4)])     # ts=200: orthogonal
        npz = {"records": records, "embeddings": emb}
        out = compute_pair_cosines(npz, {_pair((1, 1), (2, 1))})
        s = out[_pair((1, 1), (2, 1))]
        assert s["cos_med"] == pytest.approx(0.5, abs=1e-6)

    def test_antiparallel(self):
        records = _records([(1, 100, 1, 9), (2, 100, 1, 9)])
        emb = np.stack([_unit(0, 1.5), _unit(0, -2.5)])
        npz = {"records": records, "embeddings": emb}
        s = compute_pair_cosines(npz, {_pair((1, 1), (2, 1))})[_pair((1, 1), (2, 1))]
        assert s["cos_med"] == pytest.approx(-1.0, abs=1e-6)

    def test_median_uses_shared_frames_only(self):
        # Track (1,1) has detections at ts 100 (shared) and ts 300 (not shared).
        records = _records([(1, 100, 1, 2), (1, 300, 1, 2), (2, 100, 1, 2)])
        e0, e1 = _unit(0), _unit(1)
        emb = np.stack([e0, e1, e0])
        npz = {"records": records, "embeddings": emb}
        s = compute_pair_cosines(npz, {_pair((1, 1), (2, 1))})[_pair((1, 1), (2, 1))]
        assert s["cos_med"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_embeddings_returns_empty(self):
        records = _records([(1, 100, 1, 2), (2, 100, 1, 2)])
        npz = {"records": records, "embeddings": np.zeros((0, 256), np.float32)}
        assert compute_pair_cosines(npz, {_pair((1, 1), (2, 1))}) == {}

    def test_missing_embeddings_key_returns_empty(self):
        records = _records([(1, 100, 1, 2), (2, 100, 1, 2)])
        assert compute_pair_cosines({"records": records}, {_pair((1, 1), (2, 1))}) == {}

    def test_no_shared_timestamp_gives_none_keys(self):
        records = _records([(1, 100, 1, 2), (2, 200, 1, 2)])
        emb = np.stack([_unit(0), _unit(0)])
        npz = {"records": records, "embeddings": emb}
        s = compute_pair_cosines(npz, {_pair((1, 1), (2, 1))})[_pair((1, 1), (2, 1))]
        assert all(s[k] is None for k in COSINE_KEYS)

    def test_tid_zero_skipped(self):
        # Unassigned detections (tid<=0) must not enter the index.
        records = _records([(1, 100, 0, 2), (1, 100, 1, 2), (2, 100, 1, 2)])
        emb = np.stack([_unit(3, 9.0), _unit(0), _unit(0)])
        npz = {"records": records, "embeddings": emb}
        s = compute_pair_cosines(npz, {_pair((1, 1), (2, 1))})[_pair((1, 1), (2, 1))]
        assert s["cos_med"] == pytest.approx(1.0, abs=1e-6)
