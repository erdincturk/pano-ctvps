"""Tests for video dataset utilities."""

import numpy as np
import torch
import pytest

from panoctvps.data.video_dataset import (
    _detect_windows,
    panoptic_to_segments_with_ids,
)
from panoctvps.data.constants import THING_CLASSES, STUFF_CLASSES


class TestDetectWindows:
    """Tests for annotation window detection from timestamps."""

    def test_single_window(self):
        """5 timestamps within 2s form one window."""
        # WOD-realistic: 5 frames at offsets {0, 400, 600, 800, 1200} ms
        base = 1550083469000000
        ts = [
            base,
            base + 400_000,
            base + 600_000,
            base + 800_000,
            base + 1_200_000,
        ]
        windows = _detect_windows(ts)
        assert len(windows) == 1
        assert len(windows[0]) == 5

    def test_two_windows(self):
        """Gap > 2s splits into separate windows."""
        base = 1550083469000000
        # Window 1
        ts = [base + i * 200_000 for i in range(5)]
        # Window 2 (4 seconds later)
        ts += [base + 4_000_000 + i * 200_000 for i in range(5)]

        windows = _detect_windows(ts)
        assert len(windows) == 2
        assert len(windows[0]) == 5
        assert len(windows[1]) == 5

    def test_four_windows(self):
        """WOD-realistic: 4 windows per scene at ~5s spacing."""
        base = 1550083469000000
        ts = []
        for w in range(4):
            window_start = base + w * 5_000_000
            for f in range(5):
                ts.append(window_start + f * 200_000)

        windows = _detect_windows(ts)
        assert len(windows) == 4
        assert all(len(w) == 5 for w in windows)

    def test_short_window_preserved(self):
        """Windows with < 5 frames are still detected (filtered later)."""
        base = 1550083469000000
        ts = [base + i * 200_000 for i in range(3)]
        ts += [base + 4_000_000 + i * 200_000 for i in range(5)]

        windows = _detect_windows(ts)
        assert len(windows) == 2
        assert len(windows[0]) == 3  # Short window
        assert len(windows[1]) == 5  # Full window

    def test_empty_timestamps(self):
        """Empty input returns empty list."""
        assert _detect_windows([]) == []


class TestPanopticToSegmentsWithIds:
    """Tests for panoptic → segments + global instance IDs."""

    def test_stuff_only(self):
        """Frame with only stuff classes gets global_id = -1."""
        H, W = 128, 192
        semantic = np.full((H, W), 20, dtype=np.int32)  # road
        instance = np.zeros((H, W), dtype=np.int32)

        masks, labels, gids = panoptic_to_segments_with_ids(semantic, instance)

        assert masks.shape[0] == 1
        assert labels[0] == 20
        assert gids[0] == -1

    def test_thing_with_global_ids(self):
        """Thing instances get correct global IDs from mapping."""
        H, W = 128, 192
        semantic = np.zeros((H, W), dtype=np.int32)
        instance = np.zeros((H, W), dtype=np.int32)

        # Car (class 2) with local instance ID 5
        semantic[10:50, 10:50] = 2
        instance[10:50, 10:50] = 5

        # Pedestrian (class 9) with local instance ID 7
        semantic[60:90, 60:90] = 9
        instance[60:90, 60:90] = 7

        local_to_global = {5: 1001, 7: 2003}
        masks, labels, gids = panoptic_to_segments_with_ids(
            semantic, instance, local_to_global
        )

        # Find the car and pedestrian segments
        thing_mask = labels > 0
        thing_labels = labels[thing_mask]
        thing_gids = gids[thing_mask]

        # Should have exactly 2 thing segments
        thing_count = sum(1 for l in labels if l in THING_CLASSES)
        assert thing_count == 2

        # Check global IDs are mapped correctly
        for i, (lbl, gid) in enumerate(zip(labels, gids)):
            if lbl == 2:
                assert gid == 1001
            elif lbl == 9:
                assert gid == 2003

    def test_thing_without_mapping(self):
        """Thing instances without mapping get global_id = -1."""
        H, W = 128, 192
        semantic = np.zeros((H, W), dtype=np.int32)
        instance = np.zeros((H, W), dtype=np.int32)

        semantic[10:50, 10:50] = 2
        instance[10:50, 10:50] = 5

        masks, labels, gids = panoptic_to_segments_with_ids(
            semantic, instance, local_to_global=None
        )

        thing_idx = [i for i, l in enumerate(labels) if l == 2]
        assert len(thing_idx) == 1
        assert gids[thing_idx[0]] == -1

    def test_mixed_stuff_and_things(self):
        """Frame with both stuff and thing segments."""
        H, W = 128, 192
        semantic = np.full((H, W), 20, dtype=np.int32)  # road (stuff)
        instance = np.zeros((H, W), dtype=np.int32)

        # Add car
        semantic[10:30, 10:30] = 2
        instance[10:30, 10:30] = 1

        # Add sky (stuff)
        semantic[0:10, :] = 25

        local_to_global = {1: 500}
        masks, labels, gids = panoptic_to_segments_with_ids(
            semantic, instance, local_to_global
        )

        # Check we have stuff segments with -1 and thing with global ID
        stuff_count = sum(1 for g in gids if g == -1)
        thing_count = sum(1 for g in gids if g > 0)
        assert stuff_count >= 2  # road + sky
        assert thing_count == 1  # car
        assert 500 in gids

    def test_empty_frame(self):
        """Frame with all IGNORE_LABEL gets dummy segment."""
        H, W = 128, 192
        semantic = np.zeros((H, W), dtype=np.int32)  # All undefined (class 0)
        instance = np.zeros((H, W), dtype=np.int32)

        masks, labels, gids = panoptic_to_segments_with_ids(semantic, instance)

        assert masks.shape[0] == 1
        assert labels[0] == 0  # IGNORE_LABEL
        assert gids[0] == -1

    def test_multiple_instances_same_class(self):
        """Multiple instances of same class get separate masks."""
        H, W = 128, 192
        semantic = np.zeros((H, W), dtype=np.int32)
        instance = np.zeros((H, W), dtype=np.int32)

        # Two cars
        semantic[10:30, 10:30] = 2
        instance[10:30, 10:30] = 1
        semantic[50:70, 50:70] = 2
        instance[50:70, 50:70] = 2

        local_to_global = {1: 100, 2: 200}
        masks, labels, gids = panoptic_to_segments_with_ids(
            semantic, instance, local_to_global
        )

        car_indices = [i for i, l in enumerate(labels) if l == 2]
        assert len(car_indices) == 2
        car_gids = sorted([gids[i] for i in car_indices])
        assert car_gids == [100, 200]
