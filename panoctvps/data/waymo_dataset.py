"""
PyTorch Dataset Classes for WOD PVPS
=====================================
- WODPVPSDataset:         Single-frame dataset for training Mask2Former.
- WODPVPSSequenceDataset: Full-sequence dataset for wSTQ evaluation.

All decoding is TF-free: parquet -> pandas -> PIL -> numpy -> torch.
"""

import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from panoctvps.data.constants import CAMERA_NAMES, PARQUET_COLS
from panoctvps.data.decode import decode_jpeg, decode_png, decode_panoptic
from panoctvps.data.index import WODPVPSIndex


# =============================================================================
# Parquet DataFrame cache
# =============================================================================

class _ParquetCache:
    """LRU cache for loaded parquet DataFrames.

    Since consecutive samples often come from the same segment,
    caching avoids redundant parquet reads (~300MB per file).
    """

    def __init__(self, max_size: int = 4):
        self._cache: Dict[tuple, object] = {}
        self._order: list = []
        self._max_size = max_size

    def get(self, path: str, columns: Optional[List[str]] = None):
        key = (path, tuple(columns) if columns else None)
        if key in self._cache:
            return self._cache[key]

        df = pq.read_table(path, columns=columns).to_pandas()
        self._cache[key] = df
        self._order.append(key)

        while len(self._order) > self._max_size:
            old = self._order.pop(0)
            self._cache.pop(old, None)

        return df

    def clear(self):
        self._cache.clear()
        self._order.clear()


# =============================================================================
# Single-frame dataset (for Mask2Former training)
# =============================================================================

class WODPVPSDataset(Dataset):
    """PyTorch Dataset for single-frame panoptic segmentation.

    Each __getitem__ returns a dict:
        image:            (H, W, 3) uint8 RGB
        semantic_label:   (H, W) int32, class IDs 0-28
        instance_label:   (H, W) int32, instance IDs (0 = no instance)
        meta:             dict with segment_name, timestamp, camera_id, etc.

    Optional fields (controlled by flags):
        num_cameras_covered: (H, W) uint8 - camera overlap count
        instance_mapping:    dict with local_ids, global_ids, is_tracked

    For Mask2Former training it is wrapped by
    ``panoctvps.data.training_dataset.WODMask2FormerDataset``, which applies
    the augmentations and the ImageNet normalisation.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "training",
        cameras: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        load_coverage: bool = False,
        load_mapping: bool = False,
        index: Optional[WODPVPSIndex] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_root: path to waymo_open_dataset_v_2_0_1/.
            split: "training" or "validation".
            cameras: camera IDs to include (default: [1] = FRONT only).
            transform: optional augmentation callable.
            load_coverage: load num_cameras_covered (needed for wSTQ).
            load_mapping: load global instance ID mapping (needed for tracking).
            index: pre-built index (avoids re-scanning).
            cache_dir: where to cache the index JSON.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.cameras = cameras or [1]
        self.transform = transform
        self.load_coverage = load_coverage
        self.load_mapping = load_mapping

        self.index = index or WODPVPSIndex(
            dataset_root, split, self.cameras, cache_dir=cache_dir
        )

        self._seg_cache = _ParquetCache(max_size=4)
        self._img_cache = _ParquetCache(max_size=4)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.index[idx]
        seg_file = entry["segment_file"]
        timestamp = entry["timestamp"]
        camera_id = entry["camera_id"]

        # ---- Determine which columns to load ----
        seg_columns = [
            PARQUET_COLS["seg_key_timestamp"],
            PARQUET_COLS["seg_key_camera"],
            PARQUET_COLS["seg_panoptic_label"],
            PARQUET_COLS["seg_divisor"],
        ]
        if self.load_coverage:
            seg_columns.append(PARQUET_COLS["seg_num_cameras"])
        if self.load_mapping:
            seg_columns.extend([
                PARQUET_COLS["seg_local_ids"],
                PARQUET_COLS["seg_global_ids"],
                PARQUET_COLS["seg_is_tracked"],
                PARQUET_COLS["seg_sequence_id"],
            ])

        # ---- Load and decode segmentation ----
        seg_path = os.path.join(self.index.seg_dir, seg_file)
        seg_df = self._seg_cache.get(seg_path, seg_columns)
        seg_row = seg_df[
            (seg_df[PARQUET_COLS["seg_key_timestamp"]] == timestamp)
            & (seg_df[PARQUET_COLS["seg_key_camera"]] == camera_id)
        ].iloc[0]

        panoptic_raw = decode_png(seg_row[PARQUET_COLS["seg_panoptic_label"]], np.uint16)
        divisor = int(seg_row[PARQUET_COLS["seg_divisor"]])
        semantic_label, instance_label = decode_panoptic(panoptic_raw, divisor)

        # ---- Load and decode RGB image ----
        img_path = os.path.join(self.index.img_dir, seg_file)
        img_df = self._img_cache.get(img_path, [
            PARQUET_COLS["img_key_timestamp"],
            PARQUET_COLS["img_key_camera"],
            PARQUET_COLS["img_image"],
        ])
        img_row = img_df[
            (img_df[PARQUET_COLS["img_key_timestamp"]] == timestamp)
            & (img_df[PARQUET_COLS["img_key_camera"]] == camera_id)
        ].iloc[0]

        image = decode_jpeg(img_row[PARQUET_COLS["img_image"]])

        # ---- Build sample ----
        sample = {
            "image": image,                       # (H, W, 3) uint8
            "semantic_label": semantic_label,      # (H, W) int32
            "instance_label": instance_label,      # (H, W) int32
            "meta": {
                "segment_name": entry["segment_name"],
                "segment_file": seg_file,
                "timestamp": timestamp,
                "camera_id": camera_id,
                "camera_name": CAMERA_NAMES[camera_id],
            },
        }

        if self.load_coverage:
            coverage = decode_png(seg_row[PARQUET_COLS["seg_num_cameras"]], np.uint8)
            sample["num_cameras_covered"] = coverage

        if self.load_mapping:
            sample["instance_mapping"] = {
                "local_ids": seg_row[PARQUET_COLS["seg_local_ids"]],
                "global_ids": seg_row[PARQUET_COLS["seg_global_ids"]],
                "is_tracked": seg_row[PARQUET_COLS["seg_is_tracked"]],
                "sequence_id": seg_row[PARQUET_COLS["seg_sequence_id"]],
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# =============================================================================
# Sequence dataset (for wSTQ evaluation)
# =============================================================================

class WODPVPSSequenceDataset(Dataset):
    """Returns full sequences for temporal evaluation (wSTQ computation).

    Each __getitem__ returns all frames for one validation sequence,
    organized by timestamp and camera. Used during evaluation only.
    """

    def __init__(self, dataset_root: str, split: str = "validation"):
        self.dataset_root = dataset_root
        self.split = split
        self.seg_dir = os.path.join(dataset_root, split, "camera_segmentation")
        self.img_dir = os.path.join(dataset_root, split, "camera_image")

        # Find non-empty segmentation files
        self.segment_files = []
        for f in sorted(os.listdir(self.seg_dir)):
            if f.endswith(".parquet") and not f.startswith("_"):
                path = os.path.join(self.seg_dir, f)
                if pq.read_metadata(path).num_rows > 0:
                    self.segment_files.append(f)

        print(f"[WODPVPSSequenceDataset] {split}: {len(self.segment_files)} sequences")

    def __len__(self) -> int:
        return len(self.segment_files)

    def __getitem__(self, idx: int) -> Dict:
        seg_file = self.segment_files[idx]
        seg_path = os.path.join(self.seg_dir, seg_file)
        seg_df = pq.read_table(seg_path).to_pandas()

        timestamps = sorted(seg_df[PARQUET_COLS["seg_key_timestamp"]].unique())

        # num_cameras_covered is the same for every annotated timestamp of a
        # (scene, camera): decode it once per camera and share it across frames.
        coverage_by_cam: Dict[int, np.ndarray] = {}
        pixel_weights_by_cam: Dict[int, np.ndarray] = {}

        frames = []
        for ts in timestamps:
            frame_data = {"timestamp": int(ts), "cameras": {}}
            ts_rows = seg_df[seg_df[PARQUET_COLS["seg_key_timestamp"]] == ts]

            for _, row in ts_rows.iterrows():
                cam_id = int(row[PARQUET_COLS["seg_key_camera"]])

                panoptic_raw = decode_png(
                    row[PARQUET_COLS["seg_panoptic_label"]], np.uint16
                )
                divisor = int(row[PARQUET_COLS["seg_divisor"]])
                semantic, instance = decode_panoptic(panoptic_raw, divisor)

                if cam_id not in coverage_by_cam:
                    coverage = decode_png(
                        row[PARQUET_COLS["seg_num_cameras"]], np.uint8
                    )
                    coverage_by_cam[cam_id] = coverage
                    pixel_weights_by_cam[cam_id] = np.where(
                        coverage > 0,
                        1.0 / coverage.astype(np.float32),
                        0.0,
                    )

                frame_data["cameras"][cam_id] = {
                    "semantic_label": semantic,
                    "instance_label": instance,
                    "num_cameras_covered": coverage_by_cam[cam_id],
                    "instance_mapping": {
                        "local_ids": row[PARQUET_COLS["seg_local_ids"]],
                        "global_ids": row[PARQUET_COLS["seg_global_ids"]],
                        "is_tracked": row[PARQUET_COLS["seg_is_tracked"]],
                    },
                }

            frames.append(frame_data)

        return {
            "segment_file": seg_file,
            "segment_name": seg_df[PARQUET_COLS["seg_key_context"]].iloc[0],
            "sequence_id": seg_df[PARQUET_COLS["seg_sequence_id"]].iloc[0],
            "num_timestamps": len(timestamps),
            "frames": frames,
            "coverage": coverage_by_cam,
            "pixel_weights": pixel_weights_by_cam,
        }
