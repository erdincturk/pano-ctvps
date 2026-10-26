"""Read/write helpers for the per-scene prediction cache (.npz).

All scripts that write or read .npz cache files use these functions.

Mask encoding:
  mask_rle      per-record binary mask: COCO RLE dict {'size': [H,W], 'counts': str}
  pred_sem_rle  per-frame semantic map: {'h': H, 'w': W,
                                          'classes': [int, ...],
                                          'rles': [COCO-RLE-dict, ...]}

decode_sem_rle() also accepts 'rles' entries stored as raw counts
(bytes/str) instead of COCO-RLE dicts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq

import numpy as np
from pycocotools import mask as cocomask


def encode_rle(mask_hw: np.ndarray) -> dict:
    """Encode a (H, W) uint8 binary mask as a COCO RLE dict (counts as str)."""
    rle = cocomask.encode(np.asfortranarray(mask_hw.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def encode_sem_rle(pred_sem: np.ndarray) -> dict:
    """Encode a (H, W) int32 semantic label map as a per-class RLE dict.

    All non-zero classes are encoded. Zero (void/background) is skipped.
    """
    h, w = pred_sem.shape
    cls_ids = [int(c) for c in np.unique(pred_sem) if c != 0]
    rles = []
    for cls_id in cls_ids:
        m = (pred_sem == cls_id).astype(np.uint8)
        rle = cocomask.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return {"h": h, "w": w, "classes": cls_ids, "rles": rles}


def decode_rle(rle: dict, h: int | None = None, w: int | None = None) -> np.ndarray:
    """Decode a single COCO RLE dict to a (H, W) uint8 binary mask.

    h/w are only used when the rle dict lacks a 'size' key.
    """
    return cocomask.decode(rle).astype(np.uint8)


def decode_sem_rle(pred_sem_rle: dict) -> np.ndarray:
    """Decode a per-class RLE dict to a (H, W) int32 semantic label map.

    Each entry of 'rles' is either a COCO-RLE dict {'size':…, 'counts': str}
    or the raw counts (bytes/str).
    """
    h, w = pred_sem_rle["h"], pred_sem_rle["w"]
    pred_sem = np.zeros((h, w), dtype=np.int32)
    for cls_id, rle in zip(pred_sem_rle["classes"], pred_sem_rle["rles"]):
        counts = rle["counts"] if isinstance(rle, dict) else rle
        m = cocomask.decode({"size": [h, w], "counts": counts})
        pred_sem[m.astype(bool)] = cls_id
    return pred_sem


# =============================================================================
# Cache record layout + per-segment writer
#
# One .npz per scene. `records` holds one row per predicted thing instance,
# `frame_records` one row per (frame, camera) with the slice
# [records_start, records_end) of its instances; `embeddings` and `mask_rle`
# are aligned with `records`. `frame_queries` (F, 200, 256) holds the
# decoder query states of every (frame, camera) and `frame_class_logits`
# (F, 200, 30) the class logits: 29 class ids (0-28) plus no-object.
# =============================================================================
# Schema version written by write_segment_npz.
CACHE_VERSION = 4

RECORD_DTYPE = np.dtype([
    ('segment_idx', '<i2'), ('timestamp', '<i8'), ('camera_id', 'i1'),
    ('local_id', '<i4'), ('tracker_id', '<i4'), ('class_id', 'i1'),
    ('score', '<f4'), ('centroid_u', '<f4'), ('centroid_v', '<f4'),
    ('bbox_u0', '<f4'), ('bbox_v0', '<f4'), ('bbox_u1', '<f4'), ('bbox_v1', '<f4'),
    ('pixel_count', '<i4'), ('gt_global_id', '<i8'), ('gt_iou', '<f4'),
    ('frame_idx', '<i4'), ('query_idx', '<i2'), ('sequence_idx', '<i4'),
])

FRAME_RECORD_DTYPE = np.dtype([
    ('segment_idx', '<i2'), ('timestamp', '<i8'), ('camera_id', 'i1'),
    ('image_h', '<i4'), ('image_w', '<i4'), ('n_instances', '<i4'),
    ('records_start', '<i4'), ('records_end', '<i4'),
])


def load_geometry(dataset_root: str, split: str, seg_file: str, cameras: List[int]):
    """Load extrinsics (5,4,4), intrinsics (5,9), ego_pose_by_ts from parquet."""
    cal_path = os.path.join(dataset_root, split, 'camera_calibration', seg_file)
    cal_df = pq.read_table(cal_path, columns=[
        'key.camera_name',
        '[CameraCalibrationComponent].extrinsic.transform',
        '[CameraCalibrationComponent].intrinsic.f_u',
        '[CameraCalibrationComponent].intrinsic.f_v',
        '[CameraCalibrationComponent].intrinsic.c_u',
        '[CameraCalibrationComponent].intrinsic.c_v',
        '[CameraCalibrationComponent].intrinsic.k1',
        '[CameraCalibrationComponent].intrinsic.k2',
        '[CameraCalibrationComponent].intrinsic.p1',
        '[CameraCalibrationComponent].intrinsic.p2',
        '[CameraCalibrationComponent].intrinsic.k3',
    ]).to_pandas()

    extrinsics = np.zeros((5, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((5, 9), dtype=np.float32)
    for _, row in cal_df.iterrows():
        cam_id = int(row['key.camera_name']) - 1  # 1-indexed → 0-indexed
        ext = np.array(row['[CameraCalibrationComponent].extrinsic.transform'],
                       dtype=np.float32).reshape(4, 4)
        extrinsics[cam_id] = ext
        intrinsics[cam_id] = [
            row['[CameraCalibrationComponent].intrinsic.f_u'],
            row['[CameraCalibrationComponent].intrinsic.f_v'],
            row['[CameraCalibrationComponent].intrinsic.c_u'],
            row['[CameraCalibrationComponent].intrinsic.c_v'],
            row['[CameraCalibrationComponent].intrinsic.k1'],
            row['[CameraCalibrationComponent].intrinsic.k2'],
            row['[CameraCalibrationComponent].intrinsic.p1'],
            row['[CameraCalibrationComponent].intrinsic.p2'],
            row['[CameraCalibrationComponent].intrinsic.k3'],
        ]
    return extrinsics, intrinsics


def write_segment_npz(
    cache_dir: Path,
    seg_file: str,
    seg_name: str,
    seg_idx: int,
    unique_sequence_ids: np.ndarray,
    unique_timestamps: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    ego_poses: np.ndarray,
    coverage_by_cam: Dict,
    frame_records_list: List,
    records_list: List,
    embeddings_list: List,
    mask_rles: List,
    pred_sem_rles: List,
    frame_queries_list: List,
    frame_class_logits_list: List,
):
    """Write one per-scene .npz in the cache layout described above."""
    out_path = cache_dir / (Path(seg_file).stem + '.npz')

    records_arr = np.array(records_list, dtype=RECORD_DTYPE) if records_list \
        else np.zeros(0, dtype=RECORD_DTYPE)
    frame_records_arr = np.array(frame_records_list, dtype=FRAME_RECORD_DTYPE)
    embeddings_arr = np.array(embeddings_list, dtype=np.float32) if embeddings_list \
        else np.zeros((0, 256), dtype=np.float32)

    # Stack frame queries/logits: (F, 200, C)
    fq = np.stack(frame_queries_list, axis=0).astype(np.float32) \
        if frame_queries_list else np.zeros((0, 200, 256), dtype=np.float32)
    fl = np.stack(frame_class_logits_list, axis=0).astype(np.float32) \
        if frame_class_logits_list else np.zeros((0, 200, 30), dtype=np.float32)

    # coverage_id: 0 for all cameras (the coverage maps are not stored here)
    coverage_id = np.zeros(5, dtype=np.int32)

    np.savez(
        out_path,
        cache_version=np.int32(CACHE_VERSION),
        segment_file=seg_file,
        segment_name=seg_name,
        unique_timestamps=unique_timestamps,
        unique_sequence_ids=unique_sequence_ids,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        ego_pose=ego_poses,
        coverage_id=coverage_id,
        frame_records=frame_records_arr,
        records=records_arr,
        embeddings=embeddings_arr,
        mask_rle=np.array(mask_rles, dtype=object),
        pred_sem_rle=np.array(pred_sem_rles, dtype=object),
        frame_queries=fq,
        frame_class_logits=fl,
    )
    return out_path




def reconstruct_frame(npz, fr_idx: int):
    """Build per-frame tracker inputs.

    Returns:
        pred_sem, pred_inst, embeddings, scores, classes, centroids, pixel_counts
    centroids (x, y) and pixel_counts are per-instance dicts computed from
    the decoded masks.
    """
    fr = npz["frame_records"][fr_idx]
    h, w = int(fr["image_h"]), int(fr["image_w"])
    rs, re = int(fr["records_start"]), int(fr["records_end"])

    pred_sem = decode_sem_rle(npz["pred_sem_rle"][fr_idx])
    pred_inst = np.zeros((h, w), dtype=np.int32)

    embeddings: Dict[int, np.ndarray] = {}
    scores: Dict[int, float] = {}
    classes: Dict[int, int] = {}
    centroids: Dict[int, np.ndarray] = {}
    pixel_counts: Dict[int, int] = {}

    records = npz["records"][rs:re]
    embeds = npz["embeddings"][rs:re]
    masks = npz["mask_rle"][rs:re]

    for k, (rec, emb, m_rle) in enumerate(zip(records, embeds, masks)):
        inst_id = k + 1  # 1..N within the frame
        m = decode_rle(m_rle)
        m = m[:h, :w]
        cls_id = int(rec["class_id"])
        valid = m.astype(bool) & (pred_sem == cls_id)
        pred_inst[valid] = inst_id

        embeddings[inst_id] = np.asarray(emb, dtype=np.float32)
        scores[inst_id] = float(rec["score"])
        classes[inst_id] = cls_id

        ys, xs = np.where(valid)
        px = int(xs.size)
        pixel_counts[inst_id] = px
        if px > 0:
            centroids[inst_id] = np.array(
                [float(xs.mean()), float(ys.mean())], dtype=np.float64
            )

    return pred_sem, pred_inst, embeddings, scores, classes, centroids, pixel_counts

