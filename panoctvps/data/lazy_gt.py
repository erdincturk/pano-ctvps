"""Memory-light drop-in for panoctvps.data.waymo_dataset.WODPVPSSequenceDataset.

WODPVPSSequenceDataset loads the ground truth of a whole scene at once (99 timestamps x 5 cameras, semantic and
instance labels as int32: about 8-9 GB per scene). This class returns the same structure with the same arrays, but
each camera-frame decodes its panoptic PNG only when its labels are read, and nothing is kept: a one-slot cache
serves the semantic and instance labels of the frame being read. Same files, same decode functions (decode_png,
decode_panoptic), same order of timestamps and cameras, same pixel weights.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Dict

import numpy as np
import pyarrow.parquet as pq

from panoctvps.data.constants import PARQUET_COLS
from panoctvps.data.decode import decode_panoptic, decode_png

_COLS = ["seg_key_context", "seg_key_timestamp", "seg_key_camera", "seg_panoptic_label", "seg_divisor",
         "seg_num_cameras", "seg_sequence_id", "seg_local_ids", "seg_global_ids", "seg_is_tracked"]
_slot = {"key": None, "val": None}


class _LazyCam(Mapping):
    __slots__ = ("_png", "_div", "_cov", "_map")

    def __init__(self, png, div, cov, mapping):
        self._png, self._div, self._cov, self._map = png, div, cov, mapping

    def _labels(self):
        if _slot["key"] is not self:
            _slot["key"] = None
            _slot["val"] = decode_panoptic(decode_png(self._png, np.uint16), self._div)
            _slot["key"] = self
        return _slot["val"]

    def __getitem__(self, k):
        if k == "semantic_label":
            return self._labels()[0]
        if k == "instance_label":
            return self._labels()[1]
        if k == "num_cameras_covered":
            return self._cov
        if k == "instance_mapping":
            return self._map
        raise KeyError(k)

    def __iter__(self):
        return iter(("semantic_label", "instance_label", "num_cameras_covered", "instance_mapping"))

    def __len__(self):
        return 4


class LazySequenceDataset:
    def __init__(self, dataset_root: str, split: str = "validation"):
        self.dataset_root = dataset_root
        self.split = split
        self.seg_dir = os.path.join(dataset_root, split, "camera_segmentation")
        self.segment_files = []
        for f in sorted(os.listdir(self.seg_dir)):
            if f.endswith(".parquet") and not f.startswith("_"):
                if pq.read_metadata(os.path.join(self.seg_dir, f)).num_rows > 0:
                    self.segment_files.append(f)
        print(f"[LazySequenceDataset] {split}: {len(self.segment_files)} sequences")

    def __len__(self):
        return len(self.segment_files)

    def __getitem__(self, idx: int) -> Dict:
        seg_file = self.segment_files[idx]
        cols = [PARQUET_COLS[c] for c in _COLS]
        df = pq.read_table(os.path.join(self.seg_dir, seg_file), columns=cols).to_pandas()
        C = {c: PARQUET_COLS[c] for c in _COLS}
        timestamps = sorted(df[C["seg_key_timestamp"]].unique())
        coverage_by_cam, weights_by_cam = {}, {}
        frames = []
        for ts in timestamps:
            fd = {"timestamp": int(ts), "cameras": {}}
            for _, row in df[df[C["seg_key_timestamp"]] == ts].iterrows():
                cam = int(row[C["seg_key_camera"]])
                if cam not in coverage_by_cam:
                    cov = decode_png(row[C["seg_num_cameras"]], np.uint8)
                    coverage_by_cam[cam] = cov
                    weights_by_cam[cam] = np.where(cov > 0, 1.0 / cov.astype(np.float32), 0.0)
                fd["cameras"][cam] = _LazyCam(
                    row[C["seg_panoptic_label"]], int(row[C["seg_divisor"]]), coverage_by_cam[cam],
                    {"local_ids": row[C["seg_local_ids"]], "global_ids": row[C["seg_global_ids"]],
                     "is_tracked": row[C["seg_is_tracked"]]})
            frames.append(fd)
        return {"segment_file": seg_file, "segment_name": df[C["seg_key_context"]].iloc[0],
                "sequence_id": df[C["seg_sequence_id"]].iloc[0], "num_timestamps": len(timestamps),
                "frames": frames, "coverage": coverage_by_cam, "pixel_weights": weights_by_cam}
