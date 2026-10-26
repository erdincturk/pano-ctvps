"""Tests for MinVISTracker (MinVIS-style frame-to-frame tracker).

Mirrors tests/tracker/test_memory_bank.py so the two trackers are exercised
against the same scenarios. The tracker must: match a consistent object across
frames (cosine + Hungarian), split identities when embeddings diverge, respect
class consistency, honour the dead-frame budget, and handle empty frames.
"""

import numpy as np
import torch

from panoctvps.tracker.minvis_tracker import MinVISTracker


def _make_predictions(embeds, cls_ids=None, H=128, W=192, score=0.9):
    """Build a synthetic frame from a list of (256,) embeddings.

    Each instance occupies a disjoint horizontal band. Returns the
    (semantic, instance, embeddings, scores, classes) tuple the tracker expects.
    """
    n = len(embeds)
    cls_ids = cls_ids or [2] * n          # default: all 'car'
    semantic = np.zeros((H, W), dtype=np.int32)
    instance = np.zeros((H, W), dtype=np.int32)
    embeddings, scores, classes = {}, {}, {}
    for i, (emb, c) in enumerate(zip(embeds, cls_ids)):
        y0 = i * 40
        semantic[y0:y0 + 30, 10:50] = c
        instance[y0:y0 + 30, 10:50] = i + 1
        embeddings[i + 1] = np.asarray(emb, dtype=np.float32)
        scores[i + 1] = score
        classes[i + 1] = c
    return semantic, instance, embeddings, scores, classes


class TestMinVISTracker:
    def test_first_frame_creates_tracks(self):
        e = [np.random.randn(256) for _ in range(3)]
        tr = MinVISTracker(init_score_threshold=0.5)
        tracked = tr.update(*_make_predictions(e), frame_id=0)
        assert len(set(np.unique(tracked)) - {0}) == 3

    def test_consistent_object_keeps_id(self):
        """A near-identical embedding across frames keeps its track id."""
        base = np.ones(256, dtype=np.float32)
        tr = MinVISTracker(match_threshold=0.3)
        t0 = tr.update(*_make_predictions([base]), frame_id=0)
        t1 = tr.update(*_make_predictions([base + 0.01 * np.random.randn(256)]),
                       frame_id=1)
        assert (set(np.unique(t0)) - {0}) == (set(np.unique(t1)) - {0})

    def test_divergent_embeddings_split_identity(self):
        """Orthogonal embeddings below threshold start a new track."""
        a = np.zeros(256, dtype=np.float32); a[0] = 1.0
        b = np.zeros(256, dtype=np.float32); b[1] = 1.0   # cosine(a,b)=0
        tr = MinVISTracker(match_threshold=0.5)
        t0 = tr.update(*_make_predictions([a]), frame_id=0)
        t1 = tr.update(*_make_predictions([b]), frame_id=1)
        id0 = (set(np.unique(t0)) - {0}).pop()
        id1 = (set(np.unique(t1)) - {0}).pop()
        assert id0 != id1, "Dissimilar embedding must not inherit the track id"

    def test_class_consistency_enforced(self):
        """Same embedding but different class → not matched (new track)."""
        emb = np.ones(256, dtype=np.float32)
        tr = MinVISTracker(match_threshold=0.1)
        t0 = tr.update(*_make_predictions([emb], cls_ids=[2]), frame_id=0)   # car
        t1 = tr.update(*_make_predictions([emb], cls_ids=[9]), frame_id=1)   # pedestrian
        assert (set(np.unique(t0)) - {0}).pop() != (set(np.unique(t1)) - {0}).pop()

    def test_hungarian_resolves_swap(self):
        """Two objects that swap spatial bands keep their ids via embedding."""
        a = np.zeros(256, dtype=np.float32); a[0] = 1.0
        b = np.zeros(256, dtype=np.float32); b[5] = 1.0
        tr = MinVISTracker(match_threshold=0.5)
        # frame 0: a in band0, b in band1
        t0 = tr.update(*_make_predictions([a, b]), frame_id=0)
        id_a = t0[0, 10]      # band0 pixel
        id_b = t0[40, 10]     # band1 pixel
        # frame 1: positions swapped (b in band0, a in band1)
        t1 = tr.update(*_make_predictions([b, a]), frame_id=1)
        assert t1[0, 10] == id_b and t1[40, 10] == id_a

    def test_dead_frame_budget(self):
        """dead=0 drops a track after one missed frame; dead>=1 keeps it."""
        emb = np.ones(256, dtype=np.float32)
        empty = (np.zeros((128, 192), np.int32), np.zeros((128, 192), np.int32),
                 {}, {}, {})
        # dead=0: gap of one empty frame loses the track → new id on return
        tr0 = MinVISTracker(match_threshold=0.3, num_dead_frames=0)
        a = tr0.update(*_make_predictions([emb]), frame_id=0)
        tr0.update(*empty, frame_id=1)
        c = tr0.update(*_make_predictions([emb]), frame_id=2)
        assert (set(np.unique(a)) - {0}) != (set(np.unique(c)) - {0})
        # dead=2: same gap is survivable → id preserved
        tr2 = MinVISTracker(match_threshold=0.3, num_dead_frames=2)
        a2 = tr2.update(*_make_predictions([emb]), frame_id=0)
        tr2.update(*empty, frame_id=1)
        c2 = tr2.update(*_make_predictions([emb]), frame_id=2)
        assert (set(np.unique(a2)) - {0}) == (set(np.unique(c2)) - {0})

    def test_empty_frame(self):
        tr = MinVISTracker()
        tr.update(*_make_predictions([np.random.randn(256)]), frame_id=0)
        tracked = tr.update(np.zeros((128, 192), np.int32),
                            np.zeros((128, 192), np.int32), {}, {}, {}, frame_id=1)
        assert tracked.sum() == 0

    def test_reset(self):
        tr = MinVISTracker()
        tr.update(*_make_predictions([np.random.randn(256)]), frame_id=0)
        tr.reset()
        assert tr._next_id == 1 and len(tr._tracks) == 0

    def test_wod_realistic_sequence(self):
        """20 frames, varying instance counts, one persistent object."""
        tr = MinVISTracker(match_threshold=0.5, num_dead_frames=5)
        base = np.ones(256, dtype=np.float32)
        for fid in range(20):
            n = max(1, 3 + (fid % 3) - 1)
            e = [np.random.randn(256) for _ in range(n)]
            e[0] = base + 0.1 * np.random.randn(256)   # persistent first object
            sem, inst, embs, sc, cl = _make_predictions(e)
            tracked = tr.update(sem, inst, embs, sc, cl, frame_id=fid)
            assert tracked.shape == sem.shape

    def test_min_pixel_gate(self):
        """Instances under 10 px are ignored (matches the bank's gate)."""
        sem = np.zeros((128, 192), np.int32)
        inst = np.zeros((128, 192), np.int32)
        sem[0:2, 0:3] = 2          # 6 px < 10
        inst[0:2, 0:3] = 1
        tr = MinVISTracker(init_score_threshold=0.0)
        tracked = tr.update(sem, inst, {1: np.ones(256, np.float32)},
                            {1: 0.9}, {1: 2}, frame_id=0)
        assert tracked.sum() == 0
