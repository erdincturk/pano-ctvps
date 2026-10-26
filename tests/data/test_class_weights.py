"""Unit tests for the per-class anchor weights."""

import math

import pytest

from panoctvps.data.class_weights import (
    WOD_TRAINING_THING_INSTANCE_COUNTS,
    WOD_THING_ANCHOR_WEIGHTS,
    compute_class_weights,
)


def test_counts_cover_the_thing_classes():
    assert set(WOD_TRAINING_THING_INSTANCE_COUNTS.keys()) == {2, 3, 4, 5, 8, 9, 10, 11}
    assert all(v > 0 for v in WOD_TRAINING_THING_INSTANCE_COUNTS.values())


def test_default_weights_sane():
    """Car and pedestrian at the floor 0.5, motorcyclist boosted, all in [0.5, 4.0]."""
    w = WOD_THING_ANCHOR_WEIGHTS
    assert w[2] == 0.5   # car — floor clamp
    assert w[9] == 0.5   # pedestrian — floor clamp
    assert w[11] > 1.0   # motorcyclist — rare, boosted
    for val in w.values():
        assert 0.5 <= val <= 4.0


def test_compute_class_weights_sqrt_inv_freq():
    """sqrt(median / count). Median of [100, 200, 400] is 200."""
    weights = compute_class_weights({1: 100, 2: 200, 3: 400}, clamp=(0.0, 10.0))
    assert weights[1] == pytest.approx(math.sqrt(2.0))     # rare
    assert weights[2] == pytest.approx(1.0)                # median
    assert weights[3] == pytest.approx(math.sqrt(0.5))     # common


def test_clamp_applies():
    weights = compute_class_weights({1: 1, 2: 1_000_000, 3: 100}, clamp=(0.5, 4.0))
    assert weights[1] == 4.0           # huge raw → clamped
    assert weights[2] == 0.5           # tiny raw → clamped
    assert 0.5 <= weights[3] <= 4.0


def test_invalid_clamp_raises():
    with pytest.raises(ValueError):
        compute_class_weights({1: 100}, clamp=(4.0, 0.5))


def test_non_positive_count_raises():
    with pytest.raises(ValueError):
        compute_class_weights({1: 0})
    with pytest.raises(ValueError):
        compute_class_weights({1: -3})


def test_empty_input_returns_empty():
    assert compute_class_weights({}) == {}


def test_median_even_uses_midpoint_average():
    """Median of [10, 20, 30, 40] is 25 → w_1 = sqrt(25/10)."""
    weights = compute_class_weights({1: 10, 2: 20, 3: 30, 4: 40}, clamp=(0.0, 10.0))
    assert weights[1] == pytest.approx(math.sqrt(25 / 10))
    assert weights[4] == pytest.approx(math.sqrt(25 / 40))
