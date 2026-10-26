"""Training-time memory bank for consistent contrastive training.

Port of CTVIS ``SimpleTrainMemory`` / ``TrainTracklet``
(``ctvis/modeling/cl_plugin/ct_cl_plugin.py``). The memory embedding of each
instance is updated frame by frame over a clip with the rule of the inference
tracker (``panoctvps/tracker/memory_bank.py``):

    beta = clamp(mean cosine(new, past raw embeds), beta_min, beta_max)
    memory_embed = (1 - beta) * memory_embed + beta * new

Differences from CTVIS:

  * ``beta`` is capped at ``beta_max`` (0.8 by default) during training. CTVIS
    clamps only at the floor ``beta = max(0, mean_cos)``; inference uses no cap.
  * The cosine similarity is computed in fp32 (embeddings are stored in fp32)
    so that mixed precision does not perturb ``beta``.

Typical usage in the training loop::

    bank = TrainMemoryBank(beta_min=0.0, beta_max=0.8)
    for clip in batch:
        bank.reset()                                # one bank per clip
        for t in range(T):                          # temporal order
            for gid, embed in matched_at_t.items():
                bank.update(gid, embed)             # raw query embed
                positive_for_loss = bank.get_sg_embed(gid)
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class _TrainTracklet:
    """Per-instance state inside a ``TrainMemoryBank``.

    Internal class — not exported. Stores the raw past embeddings and the
    memory embedding (similarity-guided moving average).
    """

    __slots__ = (
        "reid_embeds",
        "similarity_guided_embed",
        "exist_frames",
        "maximum_cache",
    )

    def __init__(self, maximum_cache: int = 10) -> None:
        self.reid_embeds: List[torch.Tensor] = []
        self.similarity_guided_embed: Optional[torch.Tensor] = None
        self.exist_frames: int = 0
        self.maximum_cache = maximum_cache

    def update(
        self,
        positive_embed: torch.Tensor,
        beta_min: float = 0.0,
        beta_max: float = 0.8,
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Apply one update step.

        Args:
            positive_embed: ``(C,)`` or ``(1, C)`` raw query embedding for
                this instance at the current frame.
            beta_min: lower clamp for β (typically 0.0).
            beta_max: upper clamp for β. Default 0.8.
            store_dtype: dtype to cast embeddings to inside the bank.
                fp32 by default for numerical safety under AMP.
        """
        # Normalise shape to (1, C) for consistent stacking.
        if positive_embed.dim() == 1:
            positive_embed = positive_embed.unsqueeze(0)

        positive_embed = positive_embed.to(store_dtype)
        self.reid_embeds.append(positive_embed)

        if self.exist_frames == 0:
            # First valid frame: seed the EMA with the raw embed itself.
            self.similarity_guided_embed = positive_embed.clone()
            self.exist_frames = 1
            self._evict()
            return

        # Past embeddings (exclude the just-appended current one).
        past = list(self.reid_embeds[:-1])
        past_stack = torch.cat(past, dim=0)  # (N, C), N == self.exist_frames

        # β = max(0, mean_cosine(new, past_i)) ; clamp to [beta_min, beta_max].
        # Computed in fp32 / store_dtype outside any AMP autocast scope.
        past_n = F.normalize(past_stack, dim=-1)
        new_n = F.normalize(positive_embed, dim=-1)  # (1, C)
        cos = (past_n @ new_n.t()).squeeze(-1)        # (N,)
        mean_cos = cos.sum() / self.exist_frames
        beta = mean_cos.clamp(min=beta_min, max=beta_max)

        # EMA update.
        self.similarity_guided_embed = (
            (1.0 - beta) * self.similarity_guided_embed + beta * positive_embed
        )
        self.exist_frames += 1
        self._evict()

    def _evict(self) -> None:
        """FIFO eviction of oldest raw embeds beyond ``maximum_cache``.

        For clips of T=5 frames, ``len(reid_embeds) <= 5 < 10``, so this never
        fires.
        """
        while len(self.reid_embeds) > self.maximum_cache:
            self.reid_embeds.pop(0)


class TrainMemoryBank:
    """Training-time memory bank for CTVIS-style consistent contrastive learning.

    Tracks one ``_TrainTracklet`` per global instance id (``gid``) seen during
    the current clip. Designed to be ``reset()``-d at the start of every clip;
    state never persists across clips.

    Args:
        beta_min: lower clamp for β. Default 0.0 (matches CTVIS floor).
        beta_max: upper clamp for β. Default 0.8 (see module docstring).
        maximum_cache: max raw past-embed entries per instance. For T=5 this
            is never reached; included as a safety bound.
        store_dtype: dtype for embeddings inside the bank. Default fp32 to
            keep AMP-induced precision drift out of β.
    """

    def __init__(
        self,
        beta_min: float = 0.0,
        beta_max: float = 0.8,
        maximum_cache: int = 10,
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not (0.0 <= beta_min <= beta_max <= 1.0):
            raise ValueError(
                f"Invalid β clamp: [{beta_min}, {beta_max}]; expected 0 ≤ min ≤ max ≤ 1"
            )
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.maximum_cache = int(maximum_cache)
        self.store_dtype = store_dtype
        self._tracklets: Dict[int, _TrainTracklet] = {}

    def reset(self) -> None:
        """Drop all tracklet state. Call at the start of each clip."""
        self._tracklets = {}

    def update(self, gid: int, embed: torch.Tensor) -> None:
        """Apply one frame's observation for instance ``gid``.

        Args:
            gid: global instance id (int). Stuff/void should be filtered out
                by the caller; passing ``gid <= 0`` is a programming error.
            embed: ``(C,)`` or ``(1, C)`` raw query embedding.
        """
        if gid <= 0:
            raise ValueError(
                f"TrainMemoryBank.update: gid must be > 0 (thing instance); got {gid}"
            )
        if gid not in self._tracklets:
            self._tracklets[gid] = _TrainTracklet(self.maximum_cache)
        self._tracklets[gid].update(
            embed,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            store_dtype=self.store_dtype,
        )

    def get_sg_embed(self, gid: int) -> Optional[torch.Tensor]:
        """Return the current sg_embed ``(1, C)`` for ``gid``, or ``None``.

        ``None`` is returned when the instance has not been seen in the clip.
        """
        if gid not in self._tracklets:
            return None
        return self._tracklets[gid].similarity_guided_embed

    def contains(self, gid: int) -> bool:
        """True iff ``gid`` has at least one valid past observation."""
        return (
            gid in self._tracklets
            and self._tracklets[gid].similarity_guided_embed is not None
        )

    def exist_frames(self, gid: int) -> int:
        """Number of valid past frames for ``gid`` (0 if never seen)."""
        if gid not in self._tracklets:
            return 0
        return self._tracklets[gid].exist_frames

    def __contains__(self, gid: int) -> bool:
        return self.contains(gid)

    def __len__(self) -> int:
        return len(self._tracklets)

    @property
    def gids(self):
        return self._tracklets.keys()
