"""Overlap bands of the linker, computed from the camera calibration alone.

For an adjacent pair (A, B), the overlap band Omega_{A->B} is the set of pixels
of camera A whose viewing ray, back-projected under the full distortion model,
falls inside the image of camera B. Only the rotation of each camera is used;
the camera offset is ignored because the cameras are closely spaced. The
calibration is available on every split, so the same code path serves
validation and test.

Fold guard: the radial distortion polynomial r(1 + k1 r^2 + k2 r^4 + k3 r^6)
is not monotonic, so cv2.projectPoints can map far off-axis rays back into the
image. Rays are therefore kept only when their linear (undistorted)
projection lies within 1.5 times the half field of view, before distortion is
applied (the same guard as in panorama.build_panorama_lut).

WOD camera convention: x=forward/depth, y=left, z=up; `extrinsic.transform` is
T_vehicle<-camera. OpenCV (for cv2): x=right, y=down, z=forward (via P_W2O/P_O2W).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd

from panoctvps.data.constants import split_intrinsic_vector
from panoctvps.linker.panorama import P_W2O

CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT",
                4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

# Adjacent camera pairs of the rig (SL-FL-F-FR-SR chain).
ADJACENT_PAIRS = [(1, 2), (1, 3), (2, 4), (3, 5)]


def load_calibration(cal_parquet: Path | str) -> Dict[int, dict]:
    """Load intrinsics + extrinsics for all cameras from a camera_calibration parquet.

    Works identically on val and test (camera_calibration is present on both).
    Returns cam_id -> {fu,fv,cu,cv, intr9, w,h, R_cam2veh, R_veh2cam, t_veh}.
    """
    df = pd.read_parquet(cal_parquet)
    cams: Dict[int, dict] = {}
    for _, row in df.iterrows():
        cam = int(row["key.camera_name"])
        ext = np.array(row["[CameraCalibrationComponent].extrinsic.transform"],
                       dtype=np.float64).reshape(4, 4)
        fu = float(row["[CameraCalibrationComponent].intrinsic.f_u"])
        fv = float(row["[CameraCalibrationComponent].intrinsic.f_v"])
        cu = float(row["[CameraCalibrationComponent].intrinsic.c_u"])
        cv = float(row["[CameraCalibrationComponent].intrinsic.c_v"])
        k1 = float(row["[CameraCalibrationComponent].intrinsic.k1"])
        k2 = float(row["[CameraCalibrationComponent].intrinsic.k2"])
        k3 = float(row["[CameraCalibrationComponent].intrinsic.k3"])
        p1 = float(row["[CameraCalibrationComponent].intrinsic.p1"])
        p2 = float(row["[CameraCalibrationComponent].intrinsic.p2"])
        cams[cam] = {
            "fu": fu, "fv": fv, "cu": cu, "cv": cv,
            "w": int(row["[CameraCalibrationComponent].width"]),
            "h": int(row["[CameraCalibrationComponent].height"]),
            "R_cam2veh": ext[:3, :3],
            "R_veh2cam": ext[:3, :3].T,
            "t_veh": ext[:3, 3],
            # WOD 9-vector order for split_intrinsic_vector: [fu,fv,cu,cv,k1,k2,p1,p2,k3].
            "intr9": np.array([fu, fv, cu, cv, k1, k2, p1, p2, k3], dtype=np.float64),
        }
    return cams


def calibration_from_arrays(extrinsics: np.ndarray, intrinsics: np.ndarray,
                            wh: Dict[int, tuple]) -> Dict[int, dict]:
    """Build the cams dict from cache-bundled arrays instead of the parquet.

    extrinsics: (5,4,4) cam->vehicle; intrinsics: (5,9) [fu,fv,cu,cv,k1,k2,p1,p2,k3];
    both indexed by (cam_id - 1). wh: {cam_id: (width, height)} (from frame_records,
    since image size isn't in the intrinsic vector). Lets the linker run entirely
    off the cache (validation and test).
    """
    cams: Dict[int, dict] = {}
    for i in range(len(extrinsics)):
        cam = i + 1
        if cam not in wh:
            continue
        ext = np.asarray(extrinsics[i], dtype=np.float64)
        fu, fv, cu, cv, k1, k2, p1, p2, k3 = [float(x) for x in intrinsics[i]]
        w, h = wh[cam]
        cams[cam] = {
            "fu": fu, "fv": fv, "cu": cu, "cv": cv,
            "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2,
            "w": int(w), "h": int(h),
            "R_cam2veh": ext[:3, :3], "R_veh2cam": ext[:3, :3].T, "t_veh": ext[:3, 3],
            "intr9": np.array([fu, fv, cu, cv, k1, k2, p1, p2, k3], dtype=np.float64),
        }
    return cams


def backproject_rays(cal: dict) -> np.ndarray:
    """Per-pixel ray directions in the vehicle frame. Returns (H, W, 3).

    Each pixel is undistorted (cv2.undistortPoints) before back-projection.
    """
    H, W = cal["h"], cal["w"]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))  # (H, W), xy indexing
    K, dist = split_intrinsic_vector(cal["intr9"])
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(H, W, 2)  # OpenCV normalized
    x_n, y_n = und[..., 0], und[..., 1]
    # OpenCV normalized (x_n, y_n, 1) -> WOD camera ray via P_O2W = [1, -x_n, -y_n].
    d_cam = np.stack([np.ones_like(x_n, dtype=np.float64), -x_n, -y_n], axis=-1)
    return d_cam @ cal["R_cam2veh"].T  # (H, W, 3) in vehicle frame


def sees(cal_b: dict, d_veh: np.ndarray) -> np.ndarray:
    """Boolean (H, W): does camera B's image contain ray d_veh?

    Projects into B with distortion (cv2.projectPoints) behind the fold guard.
    FOV membership is [0, w) x [0, h).
    """
    d_cam_wod = d_veh @ cal_b["R_veh2cam"].T  # (H, W, 3) WOD camera B
    H, W = d_cam_wod.shape[:2]

    # WOD -> OpenCV, fold guard on linear coords, then distorted projection.
    d_ocv = (d_cam_wod @ P_W2O.T).reshape(-1, 3)
    z = d_ocv[:, 2]
    zc = np.maximum(z, 1e-9)
    in_front = z > 1e-6
    K, dist = split_intrinsic_vector(cal_b["intr9"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    in_lin = ((np.abs(d_ocv[:, 0] / zc) < 1.5 * max(cx, cal_b["w"] - cx) / fx)
              & (np.abs(d_ocv[:, 1] / zc) < 1.5 * max(cy, cal_b["h"] - cy) / fy))
    valid = in_front & in_lin
    out = np.zeros(d_ocv.shape[0], dtype=bool)
    if valid.any():
        proj, _ = cv2.projectPoints(d_ocv[valid].astype(np.float64),
                                    np.zeros(3), np.zeros(3), K, dist)
        proj = proj.reshape(-1, 2)
        with np.errstate(invalid="ignore"):
            ub, vb = proj[:, 0], proj[:, 1]
            inb = (np.isfinite(proj).all(axis=1)
                   & (ub >= 0) & (ub < cal_b["w"]) & (vb >= 0) & (vb < cal_b["h"]))
        out[np.nonzero(valid)[0][inb]] = True
    return out.reshape(H, W)


def overlap_by_partner(cams: Dict[int, dict]) -> Dict[tuple, np.ndarray]:
    """(cam_a, cam_b) -> (H_a, W_a) bool for each adjacent pair (both directions):
    the overlap band Omega_{A->B}, i.e. the pixels of A that camera B also sees.
    """
    out: Dict[tuple, np.ndarray] = {}
    rays = {a: backproject_rays(cal) for a, cal in cams.items()}
    for a, b in ADJACENT_PAIRS:
        if a in cams and b in cams:
            out[(a, b)] = sees(cams[b], rays[a])
            out[(b, a)] = sees(cams[a], rays[b])
    return out
