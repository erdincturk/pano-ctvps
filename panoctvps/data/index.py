"""
Dataset Index
==============
Scans parquet files and builds frame-level lookup tables for fast access.
Supports caching the index to disk to avoid re-scanning on every run.
"""

import json
import os
import time
from typing import Dict, List, Optional

import pyarrow.parquet as pq

from panoctvps.data.constants import PARQUET_COLS


class WODPVPSIndex:
    """Index of all labeled frames in the WOD PVPS dataset.

    Scans camera_segmentation parquet files and builds a flat list of
    (segment_file, segment_name, timestamp, camera_id) entries for
    random access during training.

    Index can be saved/loaded to avoid re-scanning.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "training",
        cameras: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_root: path to waymo_open_dataset_v_2_0_1/.
            split: "training" or "validation".
            cameras: list of camera IDs to include (default: all 5).
            cache_dir: directory to cache the index file (default: dataset_root).
        """
        self.dataset_root = dataset_root
        self.split = split
        self.cameras = cameras or [1, 2, 3, 4, 5]

        self.seg_dir = os.path.join(dataset_root, split, "camera_segmentation")
        self.img_dir = os.path.join(dataset_root, split, "camera_image")
        self.cal_dir = os.path.join(dataset_root, split, "camera_calibration")

        self.entries: List[Dict] = []
        self.segment_files: List[str] = []

        # Segment-level lookups (populated during indexing)
        self._segment_to_timestamps: Dict[str, List[int]] = {}
        self._segment_to_entries: Dict[str, List[int]] = {}

        # Try loading cached index, else build from scratch
        cache_dir = cache_dir or dataset_root
        cache_path = os.path.join(
            cache_dir,
            f"pvps_index_{split}_cam{''.join(map(str, sorted(self.cameras)))}.json",
        )

        if os.path.exists(cache_path):
            self._load_cache(cache_path)
        else:
            self._build_index()
            os.makedirs(cache_dir, exist_ok=True)
            self._save_cache(cache_path)

    def _build_index(self):
        """Scan all segmentation parquet files and build frame index."""
        t0 = time.time()

        seg_files = sorted([
            f for f in os.listdir(self.seg_dir)
            if f.endswith(".parquet") and not f.startswith("_")
        ])

        for seg_file in seg_files:
            seg_path = os.path.join(self.seg_dir, seg_file)

            try:
                table = pq.read_table(seg_path, columns=[
                    PARQUET_COLS["seg_key_context"],
                    PARQUET_COLS["seg_key_timestamp"],
                    PARQUET_COLS["seg_key_camera"],
                ])
            except Exception as e:
                print(f"  Warning: skipping {seg_file}: {e}")
                continue

            if table.num_rows == 0:
                continue

            self.segment_files.append(seg_file)
            df = table.to_pandas()
            segment_name = df[PARQUET_COLS["seg_key_context"]].iloc[0]

            timestamps = set()
            for _, row in df.iterrows():
                cam_id = int(row[PARQUET_COLS["seg_key_camera"]])
                if cam_id not in self.cameras:
                    continue

                ts = int(row[PARQUET_COLS["seg_key_timestamp"]])
                timestamps.add(ts)

                entry_idx = len(self.entries)
                self.entries.append({
                    "segment_file": seg_file,
                    "segment_name": segment_name,
                    "timestamp": ts,
                    "camera_id": cam_id,
                })

                if seg_file not in self._segment_to_entries:
                    self._segment_to_entries[seg_file] = []
                self._segment_to_entries[seg_file].append(entry_idx)

            self._segment_to_timestamps[seg_file] = sorted(timestamps)

        elapsed = time.time() - t0
        print(
            f"[WODPVPSIndex] {self.split}: {len(self.entries)} frames "
            f"from {len(self.segment_files)} segments, "
            f"cameras={self.cameras} ({elapsed:.1f}s)"
        )

    def _save_cache(self, path: str):
        """Save index to JSON for fast reloading."""
        data = {
            "split": self.split,
            "cameras": self.cameras,
            "segment_files": self.segment_files,
            "entries": self.entries,
            "segment_to_timestamps": self._segment_to_timestamps,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[WODPVPSIndex] Index cached to {path}")

    def _load_cache(self, path: str):
        """Load index from cached JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.segment_files = data["segment_files"]
        self.entries = data["entries"]
        self._segment_to_timestamps = data["segment_to_timestamps"]

        # Rebuild segment_to_entries
        for idx, entry in enumerate(self.entries):
            seg = entry["segment_file"]
            if seg not in self._segment_to_entries:
                self._segment_to_entries[seg] = []
            self._segment_to_entries[seg].append(idx)

        print(
            f"[WODPVPSIndex] Loaded cached index: {len(self.entries)} frames "
            f"from {len(self.segment_files)} segments ({path})"
        )

    # ---- Public API ----

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        return self.entries[idx]

    def get_segment_timestamps(self, segment_file: str) -> List[int]:
        """Return sorted list of timestamps for a segment."""
        return self._segment_to_timestamps.get(segment_file, [])

    def get_segment_entries(self, segment_file: str) -> List[int]:
        """Return list of entry indices belonging to a segment."""
        return self._segment_to_entries.get(segment_file, [])

    @property
    def num_segments(self) -> int:
        return len(self.segment_files)
