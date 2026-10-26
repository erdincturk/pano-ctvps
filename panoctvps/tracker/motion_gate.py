"""Memory-bank tracker with a per-camera motion gate.

Extends :class:`MemoryBankTracker` with a distance gate: a detection may
continue a track only if its mask centroid lies within ``motion_gate_px``
pixels (Euclidean) of the centroid of that track's most recent match. The
bound is fixed per camera and does not grow with the number of frames since
that match. Each camera runs its own tracker instance, so a per-camera bound is
set by constructing each instance with its own ``motion_gate_px``.

A candidate track is considered for a detection only if the class gate, the
motion gate and the match threshold all hold; among those candidates the
detection is bound to the track with the highest bi-softmax score.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from panoctvps.data.constants import THING_CLASSES
from panoctvps.tracker.memory_bank import MemoryBank, MemoryBankTracker, Tracklet


class MotionGateTracklet(Tracklet):
    """Tracklet that also stores the mask centroid of its most recent match."""

    def __init__(self, instance_id: int, maximum_cache: int = 10):
        super().__init__(instance_id, maximum_cache)
        self.last_centroid: Optional[np.ndarray] = None    # (2,) (u, v) in pixels

    def update_centroid(self, centroid: np.ndarray) -> None:
        self.last_centroid = np.asarray(centroid, dtype=np.float64)


class MotionGateMemoryBank(MemoryBank):
    """MemoryBank that creates MotionGateTracklet instances."""

    def add(self, instance_id: int) -> None:
        self.tracklets[instance_id] = MotionGateTracklet(instance_id, self.maximum_cache)


class MotionGateMemoryBankTracker(MemoryBankTracker):
    """MemoryBankTracker with a motion gate on the centroid displacement.

    Args (in addition to MemoryBankTracker):
        motion_gate_px: maximum distance in pixels between a detection's
            centroid and the centroid of a track's most recent match for the
            two to be matched.
    """

    def __init__(
        self,
        motion_gate_px: float,
        match_threshold: float = 0.7,
        init_score_threshold: float = 0.01,
        num_dead_frames: int = 5,
        maximum_cache: int = 10,
    ):
        super().__init__(
            match_threshold=match_threshold,
            init_score_threshold=init_score_threshold,
            num_dead_frames=num_dead_frames,
            maximum_cache=maximum_cache,
        )
        self.memory_bank = MotionGateMemoryBank(num_dead_frames, maximum_cache)
        self.motion_gate_px = motion_gate_px

    def reset(self) -> None:
        self.memory_bank = MotionGateMemoryBank(
            self.memory_bank.num_dead_frames,
            self.memory_bank.maximum_cache,
        )
        self._next_id = 1

    def _outside_motion_gate(
        self, tracklet: MotionGateTracklet, centroid: np.ndarray,
    ) -> bool:
        """True if the detection's centroid is farther than the bound from the
        centroid of the track's most recent match."""
        if tracklet.last_centroid is None:
            return False
        dist = float(np.linalg.norm(
            np.asarray(centroid) - tracklet.last_centroid
        ))
        return dist > self.motion_gate_px

    def update(
        self,
        pred_semantic: np.ndarray,
        pred_instance: np.ndarray,
        instance_embeddings: Dict[int, np.ndarray],
        instance_scores: Optional[Dict[int, float]] = None,
        instance_classes: Optional[Dict[int, int]] = None,
        frame_id: int = 0,
        instance_centroids: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Assign tracked IDs to predicted thing instances.

        Args (in addition to MemoryBankTracker.update):
            instance_centroids: optional dict inst_id -> (u, v) mask centroid
                in pixels. Centroids that are not supplied are computed from
                the mask.
        """
        tracked = np.zeros_like(pred_instance)
        scores = instance_scores or {}
        centroids = instance_centroids or {}

        # Collect current frame's thing instances
        current_instances = []
        for cls_id in THING_CLASSES:
            cls_mask = pred_semantic == cls_id
            if not cls_mask.any():
                continue
            for iid in np.unique(pred_instance[cls_mask]):
                if iid == 0:
                    continue
                mask = cls_mask & (pred_instance == iid)
                px = int(mask.sum())
                if px < 10:
                    continue
                if iid not in instance_embeddings:
                    continue
                if iid in centroids:
                    centroid = np.asarray(centroids[iid], dtype=np.float64)
                else:
                    ys, xs = np.where(mask)
                    centroid = np.array(
                        [float(xs.mean()), float(ys.mean())], dtype=np.float64
                    )
                current_instances.append({
                    "local_id": iid,
                    "mask": mask,
                    "class_id": int(cls_id),
                    "embedding": torch.from_numpy(
                        instance_embeddings[iid]
                    ).float(),
                    "score": float(scores.get(iid, 0.5)),
                    "centroid": centroid,
                })

        if not current_instances:
            self.memory_bank.clean_dead_tracklets(frame_id)
            return tracked

        cur_embeds = torch.stack(
            [inst["embedding"] for inst in current_instances], dim=0
        )

        if self.memory_bank.empty:
            for inst in current_instances:
                if inst["score"] >= self.init_score_threshold:
                    tid = self._next_id
                    self._next_id += 1
                    tracked[inst["mask"]] = tid
                    self._store(tid, inst, frame_id)
        else:
            track_ids, track_embeds, track_classes = \
                self.memory_bank.get_embeddings()
            track_embeds = track_embeds.to(cur_embeds.device)

            # Bi-softmax match scores
            similarity = torch.mm(cur_embeds, track_embeds.t())
            d2t_scores = similarity.softmax(dim=1)
            t2d_scores = similarity.softmax(dim=0)
            match_scores = (d2t_scores + t2d_scores) / 2

            used_tracks = set()
            assignments = []

            # Higher-confidence detections are matched first
            inst_order = sorted(
                range(len(current_instances)),
                key=lambda i: current_instances[i]["score"],
                reverse=True,
            )

            for i in inst_order:
                inst = current_instances[i]
                best_score = -1.0
                best_tid = -1

                for j in range(len(track_ids)):
                    tid_j = int(track_ids[j].item())
                    if tid_j in used_tracks:
                        continue
                    # Class gate
                    if int(track_classes[j].item()) != inst["class_id"]:
                        continue
                    # Motion gate
                    tracklet = self.memory_bank.tracklets[tid_j]
                    if self._outside_motion_gate(tracklet, inst["centroid"]):
                        continue
                    # Match threshold
                    score = float(match_scores[i, j].item())
                    if score < self.match_threshold:
                        continue
                    if score > best_score:
                        best_score = score
                        best_tid = tid_j

                if best_tid >= 0:
                    assignments.append((i, best_tid))
                    used_tracks.add(best_tid)
                else:
                    assignments.append((i, None))

            for i, tid in assignments:
                inst = current_instances[i]
                if tid is not None:
                    tracked[inst["mask"]] = tid
                    self._store(tid, inst, frame_id)
                elif inst["score"] >= self.init_score_threshold:
                    new_tid = self._next_id
                    self._next_id += 1
                    tracked[inst["mask"]] = new_tid
                    self._store(new_tid, inst, frame_id)

        self.memory_bank.clean_dead_tracklets(frame_id)
        return tracked

    def _store(self, tid: int, inst: dict, frame_id: int) -> None:
        """Memory update of track ``tid`` with detection ``inst``."""
        self.memory_bank.update(
            tid, inst["embedding"], inst["score"], inst["class_id"], frame_id,
        )
        self.memory_bank.tracklets[tid].update_centroid(inst["centroid"])
