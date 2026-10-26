"""MinVIS-style frame-to-frame tracker (alternative tracker).

The association of MinVIS (Huang et al., NeurIPS 2022), applied to the thing
detections of each frame. MinVIS matches the segmenter's own query embeddings;
in this repository ``scripts/run_baseline_cell.py`` supplies those embeddings
to the tracker. The association is

    cur, tgt = L2norm(cur_embds), L2norm(tgt_embds)
    cost = 1 - cur @ tgt.T                       # cosine distance
    indices = linear_sum_assignment(cost)        # Hungarian (bipartite)

against the last observed embedding of each track, with no memory bank, no
moving average and no bi-softmax.

Differences from upstream MinVIS:
  * MinVIS aligns a fixed set of queries frame to frame. Here the number of
    thing detections varies per frame, so association is a rectangular
    assignment with an accept threshold and new-track creation.
  * Pairs of different classes are never matched (class gate).
  * A track survives ``num_dead_frames`` frames without a match.

The ``.update()`` / ``.reset()`` signature is the same as that of
``MemoryBankTracker``.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from panoctvps.data.constants import THING_CLASSES

# Cost assigned to forbidden (cross-class) pairs so Hungarian never prefers them.
_FORBIDDEN_COST = 1.0e6


class MinVISTracker:
    """Frame-to-frame Hungarian tracker on cosine similarity.

    Args:
        match_threshold: minimum cosine similarity to accept a Hungarian match.
            Operates on the cosine scale ([-1, 1]); not comparable to the
            memory-bank bi-softmax threshold.
        init_score_threshold: minimum detection score to create a new track.
        num_dead_frames: frames a track survives without a match.
    """

    def __init__(
        self,
        match_threshold: float = 0.9,
        init_score_threshold: float = 0.01,
        num_dead_frames: int = 5,
    ):
        self.match_threshold = match_threshold
        self.init_score_threshold = init_score_threshold
        self.num_dead_frames = num_dead_frames
        # tid -> {"embed": (C,) tensor (raw last-observed), "class_id", "last_frame"}
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Reset tracker state for a new sequence."""
        self._tracks = {}
        self._next_id = 1

    def _new_track(self, embed: torch.Tensor, class_id: int, frame_id: int) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "embed": embed.clone(),
            "class_id": int(class_id),
            "last_frame": frame_id,
        }
        return tid

    def _clean_dead(self, cur_frame_id: int) -> None:
        dead = [
            tid for tid, t in self._tracks.items()
            if cur_frame_id - t["last_frame"] > self.num_dead_frames
        ]
        for tid in dead:
            del self._tracks[tid]

    def update(
        self,
        pred_semantic: np.ndarray,
        pred_instance: np.ndarray,
        instance_embeddings: Dict[int, np.ndarray],
        instance_scores: Optional[Dict[int, float]] = None,
        instance_classes: Optional[Dict[int, int]] = None,
        frame_id: int = 0,
    ) -> np.ndarray:
        """Assign tracked IDs to predicted thing instances (see MemoryBankTracker)."""
        tracked = np.zeros_like(pred_instance)
        scores = instance_scores or {}

        # Collect current frame's thing instances (same detection filter as the memory bank).
        current = []
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
                if iid not in instance_embeddings:
                    continue
                current.append({
                    "mask": mask,
                    "class_id": int(cls_id),
                    "embedding": torch.from_numpy(
                        instance_embeddings[iid]).float(),
                    "score": float(scores.get(iid, 0.5)),
                })

        if not current:
            self._clean_dead(frame_id)
            return tracked

        # No active tracks → initialise everything above threshold.
        if not self._tracks:
            for inst in current:
                if inst["score"] >= self.init_score_threshold:
                    tid = self._new_track(inst["embedding"], inst["class_id"], frame_id)
                    tracked[inst["mask"]] = tid
            self._clean_dead(frame_id)
            return tracked

        # Build cosine-similarity cost against last-observed track embeddings.
        track_ids = list(self._tracks.keys())
        cur_embeds = F.normalize(
            torch.stack([c["embedding"] for c in current], dim=0), dim=1)   # (N, C)
        trk_embeds = F.normalize(
            torch.stack([self._tracks[t]["embed"] for t in track_ids], dim=0), dim=1)  # (M, C)
        cos = torch.mm(cur_embeds, trk_embeds.t())                          # (N, M)
        cost = (1.0 - cos).numpy()

        # Forbid cross-class assignments (class gate).
        cur_cls = np.array([c["class_id"] for c in current])
        trk_cls = np.array([self._tracks[t]["class_id"] for t in track_ids])
        cost[cur_cls[:, None] != trk_cls[None, :]] += _FORBIDDEN_COST

        # Hungarian assignment on the cost.
        rows, cols = linear_sum_assignment(cost)
        matched_cur = set()
        for r, c in zip(rows.tolist(), cols.tolist()):
            inst = current[r]
            tid = track_ids[c]
            if trk_cls[c] != cur_cls[r]:
                continue                                       # forbidden pair fell through
            if cos[r, c].item() < self.match_threshold:
                continue                                       # below accept gate
            tracked[inst["mask"]] = tid
            self._tracks[tid]["embed"] = inst["embedding"].clone()   # last-observed, no EMA
            self._tracks[tid]["last_frame"] = frame_id
            matched_cur.add(r)

        # Unmatched detections above init threshold → new tracks.
        for r, inst in enumerate(current):
            if r in matched_cur:
                continue
            if inst["score"] >= self.init_score_threshold:
                tid = self._new_track(inst["embedding"], inst["class_id"], frame_id)
                tracked[inst["mask"]] = tid

        self._clean_dead(frame_id)
        return tracked
