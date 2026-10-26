"""Memory-bank tracker (inference).

Adapted from CTVIS (simple_tracker.py + memory_bank.py). Each camera keeps its
own bank of active tracks; every new thing detection is matched against the
bank by bi-softmax similarity and greedy assignment.

  - Tracklet: per-track state (class, raw embedding history, memory embedding)
  - MemoryBank: container for the active tracklets of one camera
  - MemoryBankTracker: bi-softmax matching, class gate, track management

Bi-softmax matching:
  S = current_embeds @ memory_embeds^T
  match_score = (softmax(S, dim=tracks) + softmax(S, dim=detections)) / 2

Memory update on a match:
  beta = max(0, mean cosine(new_embed, past raw embeds of the track))
  memory_embed = (1 - beta) * memory_embed + beta * new_embed
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from panoctvps.data.constants import THING_CLASSES


class Tracklet:
    """Per-instance tracking state."""

    def __init__(self, instance_id: int, maximum_cache: int = 10):
        self.instance_id = instance_id
        self.reid_embeds: List[torch.Tensor] = []
        self.scores: List[float] = []
        self.class_id: int = -1
        self.frame_ids: List[int] = []
        self.exist_frames: int = 0
        self.maximum_cache = maximum_cache

        # Memory embedding (similarity-guided moving average)
        self.similarity_guided_embed: Optional[torch.Tensor] = None

    def update(
        self,
        reid_embed: torch.Tensor,
        score: float,
        class_id: int,
        frame_id: int,
    ) -> None:
        """Update the tracklet with a matched detection."""
        self.reid_embeds.append(reid_embed)
        self.scores.append(score)
        self.class_id = class_id
        self.frame_ids.append(frame_id)

        if self.exist_frames == 0:
            # First observation: the memory embedding is the raw embedding.
            self.similarity_guided_embed = reid_embed.clone()
        else:
            # Similarity-guided moving average.
            all_past = torch.stack(self.reid_embeds[:-1], dim=0)
            similarity = torch.sum(
                torch.einsum(
                    "nc,c->n",
                    F.normalize(all_past, dim=-1),
                    F.normalize(reid_embed, dim=-1),
                )
            ) / len(self.reid_embeds[:-1])
            beta = max(0.0, similarity.item())
            self.similarity_guided_embed = (
                (1 - beta) * self.similarity_guided_embed + beta * reid_embed
            )

        self.exist_frames += 1

        # Evict old entries to bound memory
        if len(self.scores) > self.maximum_cache:
            self.scores.pop(0)
            self.reid_embeds.pop(0)


class MemoryBank:
    """Container for active tracklets."""

    def __init__(
        self,
        num_dead_frames: int = 5,
        maximum_cache: int = 10,
    ):
        self.tracklets: Dict[int, Tracklet] = {}
        self.num_dead_frames = num_dead_frames
        self.maximum_cache = maximum_cache

    @property
    def empty(self) -> bool:
        return len(self.tracklets) == 0

    @property
    def exist_ids(self):
        return self.tracklets.keys()

    def add(self, instance_id: int) -> None:
        self.tracklets[instance_id] = Tracklet(instance_id, self.maximum_cache)

    def update(
        self,
        instance_id: int,
        reid_embed: torch.Tensor,
        score: float,
        class_id: int,
        frame_id: int,
    ) -> None:
        if instance_id not in self.exist_ids:
            self.add(instance_id)
        self.tracklets[instance_id].update(reid_embed, score, class_id, frame_id)

    def clean_dead_tracklets(self, cur_frame_id: int) -> None:
        """Remove tracklets not seen for num_dead_frames."""
        dead = [
            iid for iid, t in self.tracklets.items()
            if cur_frame_id - t.frame_ids[-1] > self.num_dead_frames
        ]
        for iid in dead:
            del self.tracklets[iid]

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get stacked embeddings, IDs, and class IDs for matching.

        Returns:
            ids: (M,) int64 tensor of tracklet IDs.
            embeds: (M, C) float tensor of similarity-guided embeddings.
            class_ids: (M,) int64 tensor of class IDs.
        """
        ids, embeds, class_ids = [], [], []
        for iid, tracklet in self.tracklets.items():
            ids.append(iid)
            embeds.append(tracklet.similarity_guided_embed)
            class_ids.append(tracklet.class_id)

        if not ids:
            raise ValueError("MemoryBank is empty")

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.stack(embeds, dim=0),
            torch.tensor(class_ids, dtype=torch.long),
        )


class MemoryBankTracker:
    """Bi-softmax memory-bank tracker for one camera.

    Uses the identity embeddings of the thing detections of each frame and
    assigns consistent track ids across frames.

    Args:
        match_threshold: minimum bi-softmax score to accept a match.
        init_score_threshold: minimum detection score to start a new track.
        num_dead_frames: frames an unmatched track is kept in the bank.
        maximum_cache: max stored raw embeddings per tracklet.
    """

    def __init__(
        self,
        match_threshold: float = 0.7,
        init_score_threshold: float = 0.01,
        num_dead_frames: int = 5,
        maximum_cache: int = 10,
    ):
        self.match_threshold = match_threshold
        self.init_score_threshold = init_score_threshold
        self.memory_bank = MemoryBank(num_dead_frames, maximum_cache)
        self._next_id = 1

    def reset(self) -> None:
        """Reset tracker state for a new sequence."""
        self.memory_bank = MemoryBank(
            self.memory_bank.num_dead_frames,
            self.memory_bank.maximum_cache,
        )
        self._next_id = 1

    def update(
        self,
        pred_semantic: np.ndarray,
        pred_instance: np.ndarray,
        instance_embeddings: Dict[int, np.ndarray],
        instance_scores: Optional[Dict[int, float]] = None,
        instance_classes: Optional[Dict[int, int]] = None,
        frame_id: int = 0,
    ) -> np.ndarray:
        """Assign tracked IDs to predicted thing instances.

        Args:
            pred_semantic: (H, W) int32 predicted semantic labels.
            pred_instance: (H, W) int32 predicted instance labels (per-frame).
            instance_embeddings: dict mapping inst_id → (C,) embedding.
            instance_scores: dict mapping inst_id → confidence score.
            instance_classes: dict mapping inst_id → class_id.
            frame_id: current frame index in sequence.

        Returns:
            (H, W) int32 tracked instance labels.
        """
        tracked = np.zeros_like(pred_instance)
        scores = instance_scores or {}
        classes = instance_classes or {}

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
                if mask.sum() < 10:
                    continue
                if iid not in instance_embeddings:
                    continue
                current_instances.append({
                    "local_id": iid,
                    "mask": mask,
                    "class_id": cls_id,
                    "embedding": torch.from_numpy(
                        instance_embeddings[iid]
                    ).float(),
                    "score": scores.get(iid, 0.5),
                })

        if not current_instances:
            self.memory_bank.clean_dead_tracklets(frame_id)
            return tracked

        # Stack current embeddings
        cur_embeds = torch.stack(
            [inst["embedding"] for inst in current_instances], dim=0
        )  # (N, C)

        if self.memory_bank.empty:
            # First frame: create new tracks for all instances above threshold
            for inst in current_instances:
                if inst["score"] >= self.init_score_threshold:
                    tid = self._next_id
                    self._next_id += 1
                    tracked[inst["mask"]] = tid
                    self.memory_bank.update(
                        tid, inst["embedding"], inst["score"],
                        inst["class_id"], frame_id,
                    )
        else:
            track_ids, track_embeds, track_classes = \
                self.memory_bank.get_embeddings()
            track_embeds = track_embeds.to(cur_embeds.device)

            # Bi-softmax match scores
            similarity = torch.mm(cur_embeds, track_embeds.t())
            d2t_scores = similarity.softmax(dim=1)
            t2d_scores = similarity.softmax(dim=0)
            match_scores = (d2t_scores + t2d_scores) / 2

            # Greedy matching
            used_tracks = set()
            assignments = []

            # Sort by score (higher-confidence detections matched first)
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
                    if track_ids[j].item() in used_tracks:
                        continue
                    # Class consistency check
                    if track_classes[j].item() != inst["class_id"]:
                        continue
                    score = match_scores[i, j].item()
                    if score > best_score:
                        best_score = score
                        best_tid = track_ids[j].item()

                if best_score >= self.match_threshold and best_tid >= 0:
                    assignments.append((i, best_tid))
                    used_tracks.add(best_tid)
                else:
                    assignments.append((i, None))

            # Apply assignments
            for i, tid in assignments:
                inst = current_instances[i]
                if tid is not None:
                    tracked[inst["mask"]] = tid
                    self.memory_bank.update(
                        tid, inst["embedding"], inst["score"],
                        inst["class_id"], frame_id,
                    )
                elif inst["score"] >= self.init_score_threshold:
                    new_tid = self._next_id
                    self._next_id += 1
                    tracked[inst["mask"]] = new_tid
                    self.memory_bank.update(
                        new_tid, inst["embedding"], inst["score"],
                        inst["class_id"], frame_id,
                    )

        self.memory_bank.clean_dead_tracklets(frame_id)
        return tracked
