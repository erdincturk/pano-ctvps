"""Panoramic projection of the linker.

The panorama is an equirectangular grid of 0.1 degree cells, aligned with the
vehicle frame, spanning 220 degrees of longitude and 60 degrees of latitude
around the forward axis (600 x 2200 cells). Only the rotation of each camera
is used; the camera offset is ignored because the cameras are closely spaced.

  ``build_panorama_lut``                panorama cell -> source pixel of one
                                        camera (used to compute R_AB)
  ``project_mask_to_panorama``          apply such a LUT to a mask
  ``build_overlap_pano_mask``           R_AB, the cells covered by both cameras
  ``project_mask_to_panorama_forward``  map each mask pixel to the nearest
                                        cell along its ray, after undistortion
                                        and rotation into the vehicle frame

Conventions
-----------

  Vehicle frame (WOD): ``+x = forward, +y = left, +z = up``.
  Panorama: row index increases downward, column index increases rightward.
  Longitude is signed: ``lon > 0`` is right (vehicle ``-y``).
  Latitude  is signed: ``lat > 0`` is up    (vehicle ``+z``).

A panorama cell ``(i, j)`` corresponds to the direction at its centre:

    lon = ((j + 0.5) / W_pano - 0.5) * lon_range_rad
    lat = -((i + 0.5) / H_pano - 0.5) * lat_range_rad    # row flip
    d_vehicle = (cos(lat)*cos(lon),
                 -cos(lat)*sin(lon),
                  sin(lat))
"""

from __future__ import annotations

import warnings
from typing import Tuple

import cv2
import numpy as np

from panoctvps.data.constants import split_intrinsic_vector

# =============================================================================
# WOD <-> OpenCV camera-frame permutation
# =============================================================================
# OpenCV camera frame:  x = right,  y = down,    z = forward (depth)
# WOD    camera frame:  x = forward, y = left,    z = up
#
# A point p_o in OpenCV-camera coords maps to p_w in WOD-camera coords via:
#   x_WOD = +z_OpenCV
#   y_WOD = -x_OpenCV
#   z_WOD = -y_OpenCV
# i.e. p_WOD = P_O2W @ p_OpenCV
P_O2W: np.ndarray = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
P_W2O: np.ndarray = P_O2W.T  # orthogonal -> transpose is inverse

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PANO_SHAPE: Tuple[int, int] = (600, 2200)   # (H_pano, W_pano), 0.1 deg cells
DEFAULT_LON_RANGE_DEG: float = 220.0
DEFAULT_LAT_RANGE_DEG: float = 60.0


# =============================================================================
# LUT construction
# =============================================================================


def build_panorama_lut(
    extrinsic_wod: np.ndarray,
    intrinsic_9: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    pano_shape: Tuple[int, int] = DEFAULT_PANO_SHAPE,
    lon_range_deg: float = DEFAULT_LON_RANGE_DEG,
    lat_range_deg: float = DEFAULT_LAT_RANGE_DEG,
) -> np.ndarray:
    """Build a (H_pano, W_pano, 2) int32 panorama-to-camera sampling LUT.

    ``lut[i, j, 0]`` = source-camera column ``u`` (or -1 if outside FOV).
    ``lut[i, j, 1]`` = source-camera row    ``v`` (or -1 if outside FOV).

    Args:
        extrinsic_wod: (4, 4) WOD cal_extrinsic, camera-to-vehicle (the same
            frame returned by ``npz["extrinsics"][cam_slot]``).
        intrinsic_9: (9,) flat intrinsics ``[fu, fv, cu, cv, k1, k2, p1, p2, k3]``.
        image_shape: ``(H, W)`` of the source camera image (original, not padded).
        pano_shape: ``(H_pano, W_pano)``. Defaults to ``(600, 2200)``.
        lon_range_deg: total horizontal coverage of the panorama in degrees (220).
        lat_range_deg: total vertical coverage in degrees (60).

    Returns:
        (H_pano, W_pano, 2) int32 LUT.
    """
    H_pano, W_pano = pano_shape
    H_img, W_img = int(image_shape[0]), int(image_shape[1])

    # ── Panorama direction grid in vehicle frame ────────────────────────────
    lon_range = np.radians(lon_range_deg)
    lat_range = np.radians(lat_range_deg)
    # Pixel-centre sampling: a +0.5 offset so cell W//2 lands exactly at
    # lon=0 for odd-sized panos.
    j_grid, i_grid = np.meshgrid(np.arange(W_pano), np.arange(H_pano))
    lon = ((j_grid + 0.5) / W_pano - 0.5) * lon_range  # rightward +
    lat = -((i_grid + 0.5) / H_pano - 0.5) * lat_range # upward +, row flip
    cos_lat = np.cos(lat)
    d_vehicle = np.stack(
        [cos_lat * np.cos(lon),
         -cos_lat * np.sin(lon),   # WOD +y = left, so lon>0 = -y
         np.sin(lat)],
        axis=-1,
    ).astype(np.float64)                                # (H_pano, W_pano, 3)

    # ── Rotate vehicle-frame directions into the WOD camera frame ───────────
    # extrinsic is camera-to-vehicle; vehicle-to-camera rotation is its transpose.
    R_cam_to_vehicle = np.asarray(extrinsic_wod[:3, :3], dtype=np.float64)
    R_vehicle_to_cam = R_cam_to_vehicle.T
    d_cam_wod = d_vehicle @ R_vehicle_to_cam.T          # (H_pano, W_pano, 3)

    # ── WOD camera frame -> OpenCV camera frame (so cv2.projectPoints applies) ─
    d_cam_opencv = d_cam_wod @ P_W2O.T                   # (H_pano, W_pano, 3)
    pts_3d = d_cam_opencv.reshape(-1, 3)                 # (H_pano*W_pano, 3)

    # ── Project to image, masking points behind the camera AND those
    # whose linear projection lies wildly outside the image bounds. The
    # latter filter is critical: the radial-distortion polynomial
    #   r_distorted = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)
    # is non-monotonic, so for far-off-axis 3D points cv2.projectPoints
    # can return pixel coordinates that numerically land inside the image
    # bounds (back-projection / wrap-around artefact). Pre-filtering by
    # linear normalized image coordinates with a generous margin keeps
    # only points whose UNDISTORTED projection is in or near the image,
    # which is the regime where the distortion model is well-behaved.
    lut = np.full((H_pano * W_pano, 2), -1, dtype=np.int32)
    in_front = pts_3d[:, 2] > 1e-6                        # OpenCV +z = forward
    if not in_front.any():
        return lut.reshape(H_pano, W_pano, 2)

    K, dist = split_intrinsic_vector(intrinsic_9)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Linear normalized image coordinates (no distortion applied)
    z = np.maximum(pts_3d[:, 2], 1e-9)
    x_norm = pts_3d[:, 0] / z
    y_norm = pts_3d[:, 1] / z
    # Allowed range: 1.5x the camera's linear half-FOV in each direction.
    # 1.5x absorbs reasonable barrel distortion bringing in nearby pixels;
    # excludes the far-off-axis directions where the polynomial folds back.
    x_max_lin = 1.5 * max(cx, W_img - cx) / fx
    y_max_lin = 1.5 * max(cy, H_img - cy) / fy
    in_linear_fov = (np.abs(x_norm) < x_max_lin) & (np.abs(y_norm) < y_max_lin)
    in_front = in_front & in_linear_fov
    if not in_front.any():
        return lut.reshape(H_pano, W_pano, 2)

    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    uv, _ = cv2.projectPoints(
        pts_3d[in_front].astype(np.float32),
        rvec, tvec, K, dist,
    )
    uv = uv.reshape(-1, 2)                                # (n_in_front, 2)

    # ── Filter NaNs (cv2.projectPoints can return NaN at extreme angles
    # despite the in_front filter), round to nearest pixel, bounds-check.
    finite = np.isfinite(uv).all(axis=1)
    # Suppress the float-NaN-to-int cast warning: the result is correct
    # because the ``finite`` mask zeroes out non-finite cells via in_bounds.
    with np.errstate(invalid="ignore"):
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
    in_bounds = (
        finite
        & (u_int >= 0) & (u_int < W_img)
        & (v_int >= 0) & (v_int < H_img)
    )

    in_front_idx = np.nonzero(in_front)[0]
    valid_flat = in_front_idx[in_bounds]
    lut[valid_flat, 0] = u_int[in_bounds]
    lut[valid_flat, 1] = v_int[in_bounds]

    return lut.reshape(H_pano, W_pano, 2)


# =============================================================================
# Mask projection and IoU
# =============================================================================


def project_mask_to_panorama(
    mask: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """Project a (H, W) binary mask to (H_pano, W_pano) bool panorama.

    Args:
        mask: (H, W) bool or 0/1 array — the camera-frame mask.
        lut:  (H_pano, W_pano, 2) int32 panorama-to-camera LUT from
            ``build_panorama_lut``.

    Returns:
        (H_pano, W_pano) bool array. ``True`` only where the LUT is valid
        AND the source-camera mask is True at the sampled pixel.

    Notes:
        Uses nearest-neighbour sampling (the LUT already stores rounded
        integer pixel coords). Panorama pixels where the LUT is invalid
        ((u, v) = (-1, -1)) are always False.

        The LUT's stored ``(u, v)`` may exceed ``mask.shape`` if the LUT
        was built for the padded-to-32 cache shape but the mask was
        cropped to the original ``(image_h, image_w)``. The bounds check
        here guards against that mismatch.
    """
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    u = lut[..., 0]
    v = lut[..., 1]
    valid = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    out = np.zeros(lut.shape[:2], dtype=bool)
    out[valid] = m[v[valid], u[valid]]
    return out


def build_overlap_pano_mask(
    lut_a: np.ndarray, lut_b: np.ndarray,
    coverage_a: np.ndarray, coverage_b: np.ndarray,
) -> np.ndarray:
    """Panorama region covered by both cameras of a pair (R_AB).

    A cell belongs to the region iff both cameras' LUTs sample a valid pixel
    at which the given per-camera mask is True. The linker passes all-True
    masks, so R_AB depends on the calibration alone.

    Args:
        lut_a, lut_b: (H_pano, W_pano, 2) int32 LUTs from
            :func:`build_panorama_lut`.
        coverage_a, coverage_b: (H_cam, W_cam) bool masks of the two cameras.

    Returns:
        (H_pano, W_pano) bool.
    """
    cov_pano_a = project_mask_to_panorama(coverage_a, lut_a)
    cov_pano_b = project_mask_to_panorama(coverage_b, lut_b)
    return cov_pano_a & cov_pano_b


def project_mask_to_panorama_forward(
    mask: np.ndarray,
    extrinsic_wod: np.ndarray,
    intrinsic_9: np.ndarray,
    *,
    pano_shape: Tuple[int, int] = DEFAULT_PANO_SHAPE,
    lon_range_deg: float = DEFAULT_LON_RANGE_DEG,
    lat_range_deg: float = DEFAULT_LAT_RANGE_DEG,
) -> np.ndarray:
    """Forward-project a (H, W) camera mask to a (H_pano, W_pano) bool panorama.

    Unlike :func:`project_mask_to_panorama` (which samples pano <- cam via a
    LUT), this function projects each mask pixel directly to its panorama
    cell, so the footprint has no sampling gaps.

    Process per mask pixel ``(u, v)``:
      1. Undistort via ``cv2.undistortPoints`` → normalized OpenCV image (x', y')
      2. Form 3D direction (x', y', 1) in OpenCV camera frame, normalize
      3. Transform OpenCV → WOD camera frame  (multiply by P_O2W)
      4. Transform WOD camera → vehicle frame (multiply by R_cam_to_vehicle)
      5. Map (vehicle-frame direction) → (lat, lon)
      6. Map (lat, lon) → panorama pixel (i, j); mark True

    Args:
        mask: (H, W) bool or 0/1 array — camera-frame mask.
        extrinsic_wod: (4, 4) camera-to-vehicle transform (same as LUT).
        intrinsic_9: (9,) flat intrinsics.
        pano_shape: ``(H_pano, W_pano)``. Defaults to ``DEFAULT_PANO_SHAPE``.
        lon_range_deg / lat_range_deg: panorama angular coverage.

    Returns:
        (H_pano, W_pano) bool array, True wherever the mask's camera pixels
        project. Dense by construction (one True pixel per mask pixel, modulo
        rounding to integer panorama coords).
    """
    m = np.asarray(mask).astype(bool)
    H_pano, W_pano = pano_shape
    out = np.zeros((H_pano, W_pano), dtype=bool)

    ys, xs = np.where(m)
    if xs.size == 0:
        return out

    # 1. Undistort cam pixels to normalized OpenCV image coords.
    K, dist = split_intrinsic_vector(intrinsic_9)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts = pts.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)  # (n, 2)

    # 2. 3D direction in OpenCV camera frame (z=1 plane), then normalize.
    d_opencv = np.column_stack(
        [undistorted, np.ones(undistorted.shape[0], dtype=np.float64)],
    )                                                              # (n, 3)
    d_opencv = d_opencv / np.linalg.norm(d_opencv, axis=1, keepdims=True)

    # 3. OpenCV cam frame -> WOD cam frame (p_WOD = P_O2W @ p_OpenCV).
    d_cam_wod = d_opencv @ P_O2W.T                                 # (n, 3)

    # 4. WOD cam frame -> vehicle frame (multiply by R_cam_to_vehicle).
    R_cam_to_vehicle = np.asarray(extrinsic_wod[:3, :3], dtype=np.float64)
    d_vehicle = d_cam_wod @ R_cam_to_vehicle.T                     # (n, 3)
    # Re-normalize (rotation preserves length but be safe with numerics)
    d_vehicle = d_vehicle / np.maximum(
        np.linalg.norm(d_vehicle, axis=1, keepdims=True), 1e-12,
    )

    # 5. (lat, lon) from vehicle-frame direction.
    # WOD vehicle frame: +x forward, +y left, +z up.
    # Panorama convention (mirrors build_panorama_lut):
    #   lon = atan2(-y, x)  ∈ (-pi, pi), positive = right
    #   lat = asin(z)        ∈ (-pi/2, pi/2), positive = up
    lon = np.arctan2(-d_vehicle[:, 1], d_vehicle[:, 0])             # (n,)
    lat = np.arcsin(np.clip(d_vehicle[:, 2], -1.0, 1.0))            # (n,)

    # 6. (lat, lon) -> panorama (i, j). Mirror the lat/lon → (i, j) mapping
    # in build_panorama_lut: cell (i, j) centre at
    #   lon = ((j + 0.5) / W_pano - 0.5) * lon_range
    #   lat = -((i + 0.5) / H_pano - 0.5) * lat_range
    # Invert:
    lon_range = np.radians(lon_range_deg)
    lat_range = np.radians(lat_range_deg)
    j = (lon / lon_range + 0.5) * W_pano - 0.5
    i = (-lat / lat_range + 0.5) * H_pano - 0.5
    j_int = np.round(j).astype(np.int32)
    i_int = np.round(i).astype(np.int32)

    in_bounds = (
        (j_int >= 0) & (j_int < W_pano)
        & (i_int >= 0) & (i_int < H_pano)
    )
    # Pixels projecting beyond the panorama span (|lon| > lon_range/2 or
    # |lat| > lat_range/2) fall out of bounds and are dropped; warn when this
    # concerns a noticeable share of the mask.
    n_drop = int((~in_bounds).sum())
    if n_drop and n_drop > 0.02 * in_bounds.size:
        # Static message (no per-call counts) so Python's default warning dedup
        # prints it once, not once per side-camera projection.
        warnings.warn(
            "project_mask_to_panorama_forward: some mask pixels project beyond "
            f"the panorama span (|lon|>{lon_range_deg / 2:.0f}deg or "
            f"|lat|>{lat_range_deg / 2:.0f}deg) and are dropped.",
            stacklevel=2,
        )
    out[i_int[in_bounds], j_int[in_bounds]] = True
    return out
