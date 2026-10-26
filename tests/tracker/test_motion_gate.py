"""Tests for the motion gate of MotionGateMemoryBankTracker."""

import numpy as np

from panoctvps.tracker.motion_gate import MotionGateMemoryBankTracker

H, W = 64, 400
EMB = np.ones(16, dtype=np.float32)


def _frame(x0):
    """One car (class 2) as a 20x20 mask whose left edge is at column x0."""
    sem = np.zeros((H, W), dtype=np.int32)
    inst = np.zeros((H, W), dtype=np.int32)
    sem[10:30, x0:x0 + 20] = 2
    inst[10:30, x0:x0 + 20] = 1
    return sem, inst, {1: EMB.copy()}, {1: 0.9}, {1: 2}


def _run(shift_px, gate_px=50.0):
    tracker = MotionGateMemoryBankTracker(motion_gate_px=gate_px)
    t0 = tracker.update(*_frame(10), frame_id=0)
    t1 = tracker.update(*_frame(10 + shift_px), frame_id=1)
    return int(t0[t0 > 0][0]), int(t1[t1 > 0][0])


def test_within_bound_is_matched():
    tid0, tid1 = _run(shift_px=30)
    assert tid0 == tid1


def test_beyond_bound_is_not_matched():
    tid0, tid1 = _run(shift_px=80)
    assert tid0 != tid1


def test_bound_uses_supplied_centroids():
    tracker = MotionGateMemoryBankTracker(motion_gate_px=50.0)
    sem, inst, emb, sc, cl = _frame(10)
    t0 = tracker.update(sem, inst, emb, sc, cl, 0, instance_centroids={1: (20.0, 20.0)})
    # The mask moves by 200 px, but the supplied centroid is only 10 px away.
    sem, inst, emb, sc, cl = _frame(210)
    t1 = tracker.update(sem, inst, emb, sc, cl, 1, instance_centroids={1: (30.0, 20.0)})
    assert int(t0[t0 > 0][0]) == int(t1[t1 > 0][0])


def test_reset_clears_tracks():
    tracker = MotionGateMemoryBankTracker(motion_gate_px=50.0)
    tracker.update(*_frame(10), frame_id=0)
    tracker.reset()
    assert tracker.memory_bank.empty
    t = tracker.update(*_frame(10), frame_id=0)
    assert int(t[t > 0][0]) == 1
