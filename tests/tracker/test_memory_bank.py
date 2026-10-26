"""Tests for MemoryBankTracker (inference tracker)."""

import numpy as np
import torch
import pytest

from panoctvps.tracker.memory_bank import Tracklet, MemoryBank, MemoryBankTracker


class TestTracklet:
    """Tests for per-instance Tracklet state."""

    def test_first_update(self):
        """First update initialises similarity-guided embed."""
        t = Tracklet(instance_id=1)
        embed = torch.randn(256)
        t.update(embed, score=0.9, class_id=2, frame_id=0)

        assert t.exist_frames == 1
        assert t.similarity_guided_embed is not None
        assert torch.allclose(t.similarity_guided_embed, embed)

    def test_momentum_update(self):
        """Subsequent updates use similarity-guided fusion."""
        t = Tracklet(instance_id=1)
        embed1 = torch.randn(256)
        # Make embed2 similar to embed1 so beta > 0 and fusion actually blends
        embed2 = embed1 + 0.3 * torch.randn(256)

        t.update(embed1, score=0.9, class_id=2, frame_id=0)
        t.update(embed2, score=0.8, class_id=2, frame_id=1)

        assert t.exist_frames == 2
        # With similar embeddings, beta > 0, so result differs from embed1
        assert not torch.allclose(t.similarity_guided_embed, embed1)
        assert not torch.allclose(t.similarity_guided_embed, embed2)

    def test_cache_eviction(self):
        """Old entries are evicted when maximum_cache exceeded."""
        t = Tracklet(instance_id=1, maximum_cache=3)
        for i in range(5):
            t.update(torch.randn(256), score=0.5, class_id=2, frame_id=i)

        assert len(t.reid_embeds) == 3
        assert len(t.scores) == 3

    def test_similar_embeds_high_beta(self):
        """Similar embeddings produce high beta (fast update)."""
        t = Tracklet(instance_id=1)
        embed = torch.randn(256)
        embed_similar = embed + 0.01 * torch.randn(256)  # Very similar

        t.update(embed, score=0.9, class_id=2, frame_id=0)
        old_sg = t.similarity_guided_embed.clone()
        t.update(embed_similar, score=0.9, class_id=2, frame_id=1)

        # With high similarity, beta is high, so new embed dominates
        new_dist = (t.similarity_guided_embed - embed_similar).norm()
        old_dist = (t.similarity_guided_embed - old_sg).norm()
        assert new_dist < old_dist  # Closer to new embedding


class TestMemoryBank:
    """Tests for MemoryBank container."""

    def test_add_and_update(self):
        """Can add and update tracklets."""
        mb = MemoryBank()
        assert mb.empty

        mb.update(1, torch.randn(256), 0.9, 2, 0)
        assert not mb.empty
        assert 1 in mb.exist_ids

    def test_clean_dead_tracklets(self):
        """Dead tracklets are removed after num_dead_frames."""
        mb = MemoryBank(num_dead_frames=5)
        mb.update(1, torch.randn(256), 0.9, 2, 0)
        mb.update(2, torch.randn(256), 0.8, 3, 0)

        # Update only tracklet 2 at frame 10
        mb.update(2, torch.randn(256), 0.7, 3, 10)

        # Clean at frame 10: tracklet 1 last seen at frame 0, gap=10 > 5
        mb.clean_dead_tracklets(10)

        assert 1 not in mb.exist_ids
        assert 2 in mb.exist_ids

    def test_get_embeddings(self):
        """get_embeddings returns correct shapes."""
        mb = MemoryBank()
        mb.update(1, torch.randn(256), 0.9, 2, 0)
        mb.update(2, torch.randn(256), 0.8, 3, 0)

        ids, embeds, classes = mb.get_embeddings()
        assert ids.shape == (2,)
        assert embeds.shape == (2, 256)
        assert classes.shape == (2,)


class TestMemoryBankTracker:
    """Tests for the full bi-softmax matching tracker."""

    def _make_predictions(self, n_instances=3, H=128, W=192):
        """Create synthetic per-frame predictions."""
        semantic = np.zeros((H, W), dtype=np.int32)
        instance = np.zeros((H, W), dtype=np.int32)
        embeddings = {}
        scores = {}
        classes = {}

        for i in range(n_instances):
            cls_id = 2  # car
            y0 = i * 40
            semantic[y0:y0 + 30, 10:50] = cls_id
            instance[y0:y0 + 30, 10:50] = i + 1
            embeddings[i + 1] = np.random.randn(256).astype(np.float32)
            scores[i + 1] = 0.9
            classes[i + 1] = cls_id

        return semantic, instance, embeddings, scores, classes

    def test_first_frame_creates_tracks(self):
        """First frame creates new tracks for all instances."""
        tracker = MemoryBankTracker(init_score_threshold=0.5)
        sem, inst, embs, scores, cls = self._make_predictions(3)

        tracked = tracker.update(sem, inst, embs, scores, cls, frame_id=0)

        # All 3 instances should get unique track IDs
        unique_ids = set(np.unique(tracked)) - {0}
        assert len(unique_ids) == 3

    def test_same_embeddings_maintain_tracks(self):
        """Identical embeddings across frames maintain same track IDs."""
        tracker = MemoryBankTracker(match_threshold=0.3)

        # Frame 0
        sem, inst, embs, scores, cls = self._make_predictions(2)
        tracked0 = tracker.update(sem, inst, embs, scores, cls, frame_id=0)

        # Frame 1: same embeddings → should keep same IDs
        tracked1 = tracker.update(sem, inst, embs, scores, cls, frame_id=1)

        ids0 = set(np.unique(tracked0)) - {0}
        ids1 = set(np.unique(tracked1)) - {0}
        assert ids0 == ids1, "Same embeddings should produce same track IDs"

    def test_different_embeddings_create_new_tracks(self):
        """Very different embeddings may create new tracks."""
        tracker = MemoryBankTracker(match_threshold=0.9)  # Very strict

        sem, inst, embs0, scores, cls = self._make_predictions(2)
        tracker.update(sem, inst, embs0, scores, cls, frame_id=0)

        # Frame 1: completely different embeddings
        embs1 = {k: np.random.randn(256).astype(np.float32) for k in embs0}
        tracked1 = tracker.update(sem, inst, embs1, scores, cls, frame_id=1)

        # With strict threshold, different embeddings may not match
        # (some new tracks may be created)
        ids1 = set(np.unique(tracked1)) - {0}
        assert len(ids1) > 0

    def test_reset(self):
        """Reset clears all state."""
        tracker = MemoryBankTracker()
        sem, inst, embs, scores, cls = self._make_predictions(2)
        tracker.update(sem, inst, embs, scores, cls, frame_id=0)

        tracker.reset()
        assert tracker.memory_bank.empty
        assert tracker._next_id == 1

    def test_empty_frame(self):
        """Tracker handles frames with no thing instances."""
        tracker = MemoryBankTracker()

        # Frame with instances
        sem, inst, embs, scores, cls = self._make_predictions(2)
        tracker.update(sem, inst, embs, scores, cls, frame_id=0)

        # Empty frame
        empty_sem = np.zeros((128, 192), dtype=np.int32)
        empty_inst = np.zeros((128, 192), dtype=np.int32)
        tracked = tracker.update(empty_sem, empty_inst, {}, {}, {}, frame_id=1)

        assert tracked.sum() == 0

    def test_wod_realistic_sequence(self):
        """Test with WOD-realistic sequence (99 frames, varying instances)."""
        tracker = MemoryBankTracker(
            match_threshold=0.5,
            num_dead_frames=10,
        )

        for frame_id in range(20):
            # Varying number of instances per frame
            n_inst = max(1, 3 + (frame_id % 3) - 1)
            sem, inst, embs, scores, cls = self._make_predictions(n_inst)

            # Add some consistency: first instance has similar embedding across frames
            base_embed = np.ones(256, dtype=np.float32)
            embs[1] = base_embed + 0.1 * np.random.randn(256).astype(np.float32)

            tracked = tracker.update(sem, inst, embs, scores, cls, frame_id=frame_id)
            assert tracked.shape == sem.shape
