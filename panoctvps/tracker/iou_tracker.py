"""Mask-IoU tracker: the alternative tracker without a learned appearance cue.

Links each detection to the track of the same class whose most recent mask has
the highest IoU with it, greedily, if that IoU reaches ``iou_threshold``.
Unmatched tracks are kept for ``max_age`` frames. No embeddings are used.
"""
from typing import List

import numpy as np

from panoctvps.data.constants import THING_CLASSES


class IoUTracker:
    """Per-camera greedy mask-IoU tracker with a class gate.

    Args:
        iou_threshold: minimum mask IoU to accept a match.
        max_age: frames an unmatched track is kept.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict = {}

    def reset(self):
        self._next_id = 1
        self._tracks.clear()

    def _compute_score(self, inst: dict, track: dict) -> float:
        """Mask IoU between a detection and a track (-1 across classes)."""
        if track["class"] != inst["class_id"]:
            return -1.0

        intersection = (inst["mask"] & track["mask"]).sum()
        union = (inst["mask"] | track["mask"]).sum()
        iou = intersection / (union + 1e-6)
        return iou

    def update(
        self,
        pred_semantic: np.ndarray,
        pred_instance: np.ndarray,
        instance_embeddings=None,
        instance_scores=None,
        instance_classes=None,
        frame_id: int = 0,
    ) -> np.ndarray:
        """Assign consistent track IDs to predicted instances.

        The signature matches the other trackers; embeddings, scores, classes
        and the frame index are accepted and not used.

        Args:
            pred_semantic: (H, W) int32 predicted semantic labels.
            pred_instance: (H, W) int32 predicted instance labels (per-frame).

        Returns:
            (H, W) int32 tracked instance labels.
        """
        tracked = np.zeros_like(pred_instance)

        current_instances = []
        for cls_id in THING_CLASSES:
            cls_mask = pred_semantic == cls_id
            if not cls_mask.any():
                continue
            for iid in np.unique(pred_instance[cls_mask]):
                if iid == 0:
                    continue
                mask = cls_mask & (pred_instance == iid)
                if mask.sum() < 10:
                    continue
                current_instances.append({
                    "mask": mask,
                    "class_id": cls_id,
                    "local_id": iid,
                })

        if not current_instances:
            self._age_tracks()
            return tracked

        assignments = self._match_greedy(current_instances)

        new_tracks = {}
        for inst, tid in assignments:
            track_data = {
                "mask": inst["mask"],
                "age": 0,
                "class": inst["class_id"],
            }

            if tid is not None:
                new_tracks[tid] = track_data
                tracked[inst["mask"]] = tid
            else:
                new_id = self._next_id
                self._next_id += 1
                new_tracks[new_id] = track_data
                tracked[inst["mask"]] = new_id

        # Keep unmatched tracks alive for max_age frames
        used_tracks = {tid for _, tid in assignments if tid is not None}
        for tid, track in self._tracks.items():
            if tid not in used_tracks:
                track["age"] += 1
                if track["age"] <= self.max_age:
                    new_tracks[tid] = track

        self._tracks = new_tracks
        return tracked

    def _match_greedy(self, instances: List[dict]) -> List[tuple]:
        """Greedy matching: each instance picks its best available track."""
        used_tracks = set()
        assignments = []

        for inst in instances:
            best_score = -1.0
            best_tid = -1
            for tid, track in self._tracks.items():
                if tid in used_tracks:
                    continue
                score = self._compute_score(inst, track)
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_score >= self.iou_threshold and best_tid >= 0:
                assignments.append((inst, best_tid))
                used_tracks.add(best_tid)
            else:
                assignments.append((inst, None))

        return assignments

    def _age_tracks(self):
        """Age all tracks when no instances are detected."""
        new_tracks = {}
        for tid, track in self._tracks.items():
            track["age"] += 1
            if track["age"] <= self.max_age:
                new_tracks[tid] = track
        self._tracks = new_tracks
