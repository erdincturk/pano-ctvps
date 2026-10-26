"""Tests for the camera-intrinsic helpers.

Validates that PARQUET_COLS exposes the full 9-element WOD intrinsic vector
(pinhole + OpenCV distortion) and that the split helper produces the
(K, dist) pair OpenCV's cv2.undistortPoints expects.
"""

import numpy as np
import pytest

from panoctvps.data.constants import (
    INTRINSIC_VECTOR_KEYS,
    PARQUET_COLS,
    intrinsic_columns,
    split_intrinsic_vector,
)


def test_intrinsic_vector_keys_are_canonical():
    expected = [
        "cal_fu", "cal_fv", "cal_cu", "cal_cv",
        "cal_k1", "cal_k2", "cal_p1", "cal_p2", "cal_k3",
    ]
    assert INTRINSIC_VECTOR_KEYS == expected


def test_all_intrinsic_keys_resolve_in_parquet_cols():
    for k in INTRINSIC_VECTOR_KEYS:
        assert k in PARQUET_COLS, f"missing PARQUET_COLS entry for {k}"


def test_intrinsic_columns_returns_wod_dataset_proto_paths():
    cols = intrinsic_columns()
    expected_paths = [
        "[CameraCalibrationComponent].intrinsic.f_u",
        "[CameraCalibrationComponent].intrinsic.f_v",
        "[CameraCalibrationComponent].intrinsic.c_u",
        "[CameraCalibrationComponent].intrinsic.c_v",
        "[CameraCalibrationComponent].intrinsic.k1",
        "[CameraCalibrationComponent].intrinsic.k2",
        "[CameraCalibrationComponent].intrinsic.p1",
        "[CameraCalibrationComponent].intrinsic.p2",
        "[CameraCalibrationComponent].intrinsic.k3",
    ]
    assert cols == expected_paths


def test_split_intrinsic_vector_basic():
    # Plausible WOD-FRONT-like values: ~2050 px focal length, principal at center
    fu, fv, cu, cv = 2059.5, 2058.0, 939.4, 641.7
    k1, k2, p1, p2, k3 = -0.328, 0.094, 1.0e-4, -2.0e-4, 0.0
    vec = np.array([fu, fv, cu, cv, k1, k2, p1, p2, k3])
    K, dist = split_intrinsic_vector(vec)

    assert K.shape == (3, 3)
    assert K.dtype == np.float64
    assert K[0, 0] == fu
    assert K[1, 1] == fv
    assert K[0, 2] == cu
    assert K[1, 2] == cv
    assert K[2, 2] == 1.0
    # Off-diagonal zeros
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0
    assert K[2, 0] == 0.0 and K[2, 1] == 0.0

    # OpenCV distortion vector ordering
    assert dist.shape == (5,)
    assert dist.dtype == np.float64
    np.testing.assert_array_equal(dist, [k1, k2, p1, p2, k3])


def test_split_intrinsic_vector_accepts_list_and_tuple():
    vec_list = [1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.01, 0.02, 0.3]
    K1, d1 = split_intrinsic_vector(vec_list)
    K2, d2 = split_intrinsic_vector(tuple(vec_list))
    np.testing.assert_array_equal(K1, K2)
    np.testing.assert_array_equal(d1, d2)


@pytest.mark.parametrize("bad_shape", [(), (4,), (8,), (10,), (3, 3)])
def test_split_intrinsic_vector_rejects_wrong_shape(bad_shape):
    bad = np.zeros(bad_shape) if bad_shape != () else np.array(1.0)
    with pytest.raises(ValueError, match="shape"):
        split_intrinsic_vector(bad)


def test_distortion_keys_use_opencv_convention():
    # Smoke-check: the order of distortion coefficients in PARQUET_COLS keys
    # matches OpenCV's [k1, k2, p1, p2, k3] convention, so a change in the WOD
    # field naming is caught before projection code uses wrong distortion
    # parameters.
    distortion_keys = INTRINSIC_VECTOR_KEYS[4:]
    assert distortion_keys == ["cal_k1", "cal_k2", "cal_p1", "cal_p2", "cal_k3"]
