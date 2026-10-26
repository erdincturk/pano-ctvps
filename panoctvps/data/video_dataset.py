"""Video clip dataset for the contrastive training of the tracker.

Each sample is a T=5 annotation window from one camera. Returns per-frame
M2F training data (pixel_values, mask_labels, class_labels) plus global
instance IDs for contrastive loss (positive pair identification across frames).

Training temporal structure (WOD v2.0.1):
  - Annotated frames come in windows; consecutive frames of a window are
    0.2 s to 0.4 s apart
  - Windows are separated by a timestamp gap of more than 2 s
  - Windows with a frame count other than T are skipped

Augmentation is clip-level: the same spatial transform (resize, crop) is
applied to all T frames. No horizontal flip. Color jitter is per-frame.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from panoctvps.data.constants import (
    NUM_CLASSES, THING_CLASSES, STUFF_CLASSES, IGNORE_LABEL, PARQUET_COLS,
)
from panoctvps.data.decode import decode_jpeg, decode_png, decode_panoptic
from panoctvps.data.index import WODPVPSIndex
from panoctvps.data.transforms.formatting import ToTensorNormalize


# Window detection threshold: gaps > 2s indicate separate annotation windows
_WINDOW_GAP_US = 2_000_000


def panoptic_to_segments_with_ids(
    semantic: np.ndarray,
    instance: np.ndarray,
    local_to_global: Optional[Dict[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert panoptic labels to per-segment masks with global instance IDs.

    Like training_dataset.panoptic_to_segments but also returns global instance
    IDs for contrastive learning (positive pair identification across frames).

    Args:
        semantic: (H, W) int32, class IDs 0-28.
        instance: (H, W) int32, local instance IDs (0 = no instance).
        local_to_global: mapping from local instance ID to global tracked ID.

    Returns:
        masks:  (N, H, W) bool — one binary mask per segment.
        labels: (N,) int32 — semantic class ID per segment.
        global_ids: (N,) int64 — global instance ID per segment (-1 for stuff).
    """
    masks: List[np.ndarray] = []
    labels: List[int] = []
    global_ids: List[int] = []

    # Stuff classes: one segment per class
    for c in STUFF_CLASSES:
        if c == IGNORE_LABEL:
            continue
        mask = semantic == c
        if mask.any():
            masks.append(mask)
            labels.append(c)
            global_ids.append(-1)

    # Thing classes: one segment per (class, instance_id) pair
    for c in THING_CLASSES:
        class_mask = semantic == c
        if not class_mask.any():
            continue
        unique_ids = np.unique(instance[class_mask])
        for iid in unique_ids:
            if iid == 0:
                continue
            mask = class_mask & (instance == iid)
            if mask.any():
                masks.append(mask)
                labels.append(c)
                gid = local_to_global.get(iid, -1) if local_to_global else -1
                global_ids.append(gid)

    if not masks:
        h, w = semantic.shape
        masks.append(np.zeros((h, w), dtype=bool))
        labels.append(IGNORE_LABEL)
        global_ids.append(-1)

    return (
        np.stack(masks, axis=0),
        np.array(labels, dtype=np.int32),
        np.array(global_ids, dtype=np.int64),
    )


def _detect_windows(timestamps: List[int]) -> List[List[int]]:
    """Group sorted timestamps into annotation windows.

    Windows are separated by gaps > 2 seconds. Within a window, frames
    are spaced at 200-400ms intervals.
    """
    if not timestamps:
        return []

    windows: List[List[int]] = []
    current: List[int] = [timestamps[0]]

    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i - 1] > _WINDOW_GAP_US:
            windows.append(current)
            current = [timestamps[i]]
        else:
            current.append(timestamps[i])
    windows.append(current)
    return windows


class WODVideoDataset(Dataset):
    """T=5 video clip dataset for contrastive training.

    Each __getitem__ returns a dict:
        pixel_values:      (T, 3, H, W) float32, ImageNet normalised
        mask_labels:       list of T tensors, each (N_t, H, W) float32
        class_labels:      list of T tensors, each (N_t,) int64
        global_instance_ids: list of T tensors, each (N_t,) int64
        meta:              dict with clip info
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "training",
        cameras: Optional[List[int]] = None,
        crop_size: Tuple[int, int] = (640, 640),
        min_scale: float = 0.75,
        max_scale: float = 1.25,
        color_jitter: bool = True,
        clip_length: int = 5,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.cameras = cameras or [1]
        self.crop_h, self.crop_w = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.clip_length = clip_length

        self._format = ToTensorNormalize()
        self._color_jitter = None
        if color_jitter:
            from panoctvps.data.transforms.augmentations import ColorJitter
            self._color_jitter = ColorJitter()

        # No parquet caching — only the rows of each clip are read (whole
        # parquet files with image data are several GB in pandas, per worker).

        # Build clip index
        self.clips = self._build_clip_index(cache_dir)
        print(
            f"[WODVideoDataset] {split}: {len(self.clips)} clips "
            f"(T={clip_length}, cameras={self.cameras})"
        )

    def _build_clip_index(self, cache_dir: Optional[str]) -> List[Dict]:
        """Build clip index by grouping frames into T=5 annotation windows."""
        index = WODPVPSIndex(
            self.dataset_root, self.split, self.cameras, cache_dir=cache_dir
        )

        # Group entries by (segment_file, camera_id)
        groups: Dict[tuple, List[Dict]] = defaultdict(list)
        for entry in index.entries:
            key = (entry["segment_file"], entry["camera_id"])
            groups[key].append(entry)

        # Detect windows and filter to clip_length
        clips = []
        skipped = 0
        for (seg_file, cam_id), entries in groups.items():
            timestamps = sorted(set(e["timestamp"] for e in entries))
            seg_name = entries[0]["segment_name"]

            for window in _detect_windows(timestamps):
                if len(window) == self.clip_length:
                    clips.append({
                        "segment_file": seg_file,
                        "segment_name": seg_name,
                        "timestamps": window,
                        "camera_id": cam_id,
                    })
                else:
                    skipped += 1

        if skipped > 0:
            print(
                f"[WODVideoDataset] Skipped {skipped} windows with "
                f"T != {self.clip_length}"
            )

        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict:
        clip = self.clips[idx]
        seg_file = clip["segment_file"]
        cam_id = clip["camera_id"]
        timestamps = clip["timestamps"]

        # Load only the rows we need — no caching of entire DataFrames.
        # Uses pyarrow row-group filtering to read only matching rows,
        # avoiding the multi-GB memory footprint of caching full parquet files.
        ts_col = PARQUET_COLS["seg_key_timestamp"]
        cam_col = PARQUET_COLS["seg_key_camera"]

        # Pyarrow DNF filter: list of ANDed conditions
        row_filter = [
            (cam_col, "=", cam_id),
            (ts_col, "in", timestamps),
        ]

        seg_columns = [
            ts_col, cam_col,
            PARQUET_COLS["seg_panoptic_label"],
            PARQUET_COLS["seg_divisor"],
            PARQUET_COLS["seg_local_ids"],
            PARQUET_COLS["seg_global_ids"],
            PARQUET_COLS["seg_is_tracked"],
        ]
        seg_path = os.path.join(
            self.dataset_root, self.split, "camera_segmentation", seg_file
        )
        seg_df = pq.read_table(
            seg_path, columns=seg_columns, filters=row_filter,
        ).to_pandas()

        img_columns = [
            PARQUET_COLS["img_key_timestamp"],
            PARQUET_COLS["img_key_camera"],
            PARQUET_COLS["img_image"],
        ]
        img_path = os.path.join(
            self.dataset_root, self.split, "camera_image", seg_file
        )
        img_df = pq.read_table(
            img_path, columns=img_columns, filters=row_filter,
        ).to_pandas()

        # Load T frames
        raw_frames = []
        for ts in timestamps:
            frame = self._load_frame(seg_df, img_df, ts, cam_id)
            raw_frames.append(frame)

        # Clip-level spatial augmentation
        raw_frames = self._augment_clip(raw_frames, cam_id=cam_id)

        # Convert to tensors and build segments
        pixel_values = []
        mask_labels = []
        class_labels = []
        global_instance_ids = []

        for frame in raw_frames:
            # Per-frame color jitter (image-only)
            if self._color_jitter is not None:
                frame = self._color_jitter(frame)

            # To tensor + normalise
            frame = self._format(frame)

            # Build per-segment masks with global instance IDs
            masks, labels, gids = panoptic_to_segments_with_ids(
                frame["semantic_label"],
                frame["instance_label"],
                frame.get("local_to_global"),
            )

            pixel_values.append(frame["image"])
            mask_labels.append(torch.from_numpy(masks.astype(np.float32)))
            class_labels.append(torch.from_numpy(labels.astype(np.int64)))
            global_instance_ids.append(torch.from_numpy(gids))

        return {
            "pixel_values": torch.stack(pixel_values),  # (T, 3, H, W)
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "global_instance_ids": global_instance_ids,
            "meta": {
                "segment_name": clip["segment_name"],
                "camera_id": cam_id,
                "timestamps": timestamps,
            },
        }

    def _load_frame(self, seg_df, img_df, timestamp: int, camera_id: int) -> Dict:
        """Load a single frame's image and labels from cached DataFrames."""
        # Segmentation
        seg_row = seg_df[
            (seg_df[PARQUET_COLS["seg_key_timestamp"]] == timestamp)
            & (seg_df[PARQUET_COLS["seg_key_camera"]] == camera_id)
        ].iloc[0]

        panoptic_raw = decode_png(seg_row[PARQUET_COLS["seg_panoptic_label"]], np.uint16)
        divisor = int(seg_row[PARQUET_COLS["seg_divisor"]])
        semantic_label, instance_label = decode_panoptic(panoptic_raw, divisor)

        # Build local → global instance ID mapping
        local_to_global = {}
        local_ids = seg_row[PARQUET_COLS["seg_local_ids"]]
        global_ids = seg_row[PARQUET_COLS["seg_global_ids"]]
        is_tracked = seg_row[PARQUET_COLS["seg_is_tracked"]]
        if local_ids is not None:
            for loc, glob, trk in zip(local_ids, global_ids, is_tracked):
                if trk and loc > 0 and glob > 0:
                    local_to_global[int(loc)] = int(glob)

        # Image
        img_row = img_df[
            (img_df[PARQUET_COLS["img_key_timestamp"]] == timestamp)
            & (img_df[PARQUET_COLS["img_key_camera"]] == camera_id)
        ].iloc[0]
        image = decode_jpeg(img_row[PARQUET_COLS["img_image"]])

        return {
            "image": image,
            "semantic_label": semantic_label,
            "instance_label": instance_label,
            "local_to_global": local_to_global,
        }

    def _augment_clip(self, frames: List[Dict], cam_id: int) -> List[Dict]:
        """Apply clip-level spatial augmentation (same transform to all frames).

        Samples augmentation parameters once, then applies deterministically
        to every frame in the clip. Color jitter is handled separately (per-frame).
        """
        from panoctvps.data.constants import SIDE_CAMERA_IDS
        pad_top = cam_id in SIDE_CAMERA_IDS

        if self.split != "training":
            # Validation: pad to crop size, no augmentation
            for frame in frames:
                frame = self._pad_and_center_crop(frame, pad_top=pad_top)
            return frames

        # 1. Random scale (same for all frames)
        scale = random.uniform(self.min_scale, self.max_scale)
        for frame in frames:
            self._resize_frame(frame, scale)

        # 2. Random crop (same position for all frames)
        h, w = frames[0]["image"].shape[:2]
        # Pad if needed
        pad_h = max(0, self.crop_h - h)
        pad_w = max(0, self.crop_w - w)
        if pad_h > 0 or pad_w > 0:
            for frame in frames:
                self._pad_frame(frame, pad_h, pad_w, pad_top=pad_top)
            h += pad_h
            w += pad_w

        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        for frame in frames:
            self._crop_frame(frame, top, left)

        return frames

    def _resize_frame(self, frame: Dict, scale: float) -> None:
        """Resize image and labels by scale factor (in-place)."""
        h, w = frame["image"].shape[:2]
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        frame["image"] = np.array(
            Image.fromarray(frame["image"]).resize((new_w, new_h), Image.BILINEAR)
        )
        frame["semantic_label"] = np.array(
            Image.fromarray(frame["semantic_label"].astype(np.int32)).resize(
                (new_w, new_h), Image.NEAREST
            ),
            dtype=np.int32,
        )
        frame["instance_label"] = np.array(
            Image.fromarray(frame["instance_label"].astype(np.int32)).resize(
                (new_w, new_h), Image.NEAREST
            ),
            dtype=np.int32,
        )

    def _pad_frame(self, frame: Dict, pad_h: int, pad_w: int, pad_top: bool = False) -> None:
        """Pad frame if smaller than crop size (in-place).

        Args:
            pad_top: If True, add height padding at the TOP (correct for
                     SIDE_LEFT/SIDE_RIGHT which align at the bottom of the scene).
        """
        h_pad = (pad_h, 0) if pad_top else (0, pad_h)
        frame["image"] = np.pad(
            frame["image"], (h_pad, (0, pad_w), (0, 0)), mode="constant"
        )
        frame["semantic_label"] = np.pad(
            frame["semantic_label"],
            (h_pad, (0, pad_w)),
            constant_values=IGNORE_LABEL,
        )
        frame["instance_label"] = np.pad(
            frame["instance_label"],
            (h_pad, (0, pad_w)),
            constant_values=0,
        )

    def _crop_frame(self, frame: Dict, top: int, left: int) -> None:
        """Crop frame to crop_h x crop_w at given position (in-place)."""
        frame["image"] = frame["image"][
            top:top + self.crop_h, left:left + self.crop_w
        ]
        frame["semantic_label"] = frame["semantic_label"][
            top:top + self.crop_h, left:left + self.crop_w
        ]
        frame["instance_label"] = frame["instance_label"][
            top:top + self.crop_h, left:left + self.crop_w
        ]

    def _pad_and_center_crop(self, frame: Dict, pad_top: bool = False) -> Dict:
        """Pad to crop size for validation (no random augmentation)."""
        h, w = frame["image"].shape[:2]
        pad_h = max(0, self.crop_h - h)
        pad_w = max(0, self.crop_w - w)
        if pad_h > 0 or pad_w > 0:
            self._pad_frame(frame, pad_h, pad_w, pad_top=pad_top)
            h += pad_h
            w += pad_w

        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        self._crop_frame(frame, top, left)
        return frame


def video_collate_fn(batch: List[Dict]) -> Dict:
    """Collate video clips into a batch.

    Flattens T frames across clips for M2F forward pass.
    pixel_values: (B*T, 3, H, W) — all frames from all clips.
    mask_labels, class_labels, global_instance_ids: flat lists of B*T tensors.
    clip_lengths: (B,) int tensor — T for each clip (for reshaping after forward).
    """
    B = len(batch)
    T = batch[0]["pixel_values"].shape[0]

    # Flatten pixel_values: (B, T, 3, H, W) → (B*T, 3, H, W)
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)

    # Flatten per-frame lists
    mask_labels = []
    class_labels = []
    global_instance_ids = []
    for b in batch:
        mask_labels.extend(b["mask_labels"])
        class_labels.extend(b["class_labels"])
        global_instance_ids.extend(b["global_instance_ids"])

    return {
        "pixel_values": pixel_values,           # (B*T, 3, H, W)
        "mask_labels": mask_labels,              # list of B*T tensors
        "class_labels": class_labels,            # list of B*T tensors
        "global_instance_ids": global_instance_ids,  # list of B*T tensors
        "clip_length": T,
        "batch_size": B,
        "meta": [b["meta"] for b in batch],
    }
