"""Training dataset adapter for Mask2Former on WOD PVPS.

Wraps WODPVPSDataset and converts panoptic labels into the per-segment
binary mask format required by HuggingFace Mask2Former:

    pixel_values:  (3, H, W) float32 tensor, ImageNet normalised
    mask_labels:   (N, H, W) float32 tensor  — one binary mask per segment
    class_labels:  (N,)      int64 tensor    — semantic class ID per segment

where N = (number of stuff regions present) + (number of thing instances).
IGNORE_LABEL (class 0, undefined) segments are skipped.

Modes: training (scale jitter, random crop, optional flip and colour jitter,
optional copy-paste), full resolution (bottom/right padding to a multiple of
32), and a centre crop for the in-training validation PQ.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from panoctvps.data.constants import (
    NUM_CLASSES, THING_CLASSES, STUFF_CLASSES, IGNORE_LABEL,
)
from panoctvps.data.waymo_dataset import WODPVPSDataset
from panoctvps.data.transforms.augmentations import (
    build_train_transforms, build_val_transforms, build_fullres_val_transforms,
)
from panoctvps.data.transforms.formatting import ToTensorNormalize
from panoctvps.data.transforms.copy_paste import CopyPaste


def panoptic_to_segments(
    semantic: np.ndarray,
    instance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert semantic + instance maps into per-segment binary masks.

    Args:
        semantic:  (H, W) int32, class IDs 0-28
        instance:  (H, W) int32, instance IDs (0 = no instance)

    Returns:
        masks:  (N, H, W) bool numpy array
        labels: (N,)      int32 numpy array of class IDs
    """
    masks = []
    labels = []

    # --- Stuff classes: one segment per class (ignoring IGNORE_LABEL) ---
    for c in STUFF_CLASSES:
        if c == IGNORE_LABEL:
            continue
        mask = semantic == c
        if mask.any():
            masks.append(mask)
            labels.append(c)

    # --- Thing classes: one segment per (class, instance_id) pair ---
    for c in THING_CLASSES:
        class_mask = semantic == c
        if not class_mask.any():
            continue
        unique_ids = np.unique(instance[class_mask])
        for iid in unique_ids:
            if iid == 0:
                continue  # pixels without a valid instance id
            mask = class_mask & (instance == iid)
            if mask.any():
                masks.append(mask)
                labels.append(c)

    if not masks:
        # Degenerate frame — return a single all-ignore dummy segment
        h, w = semantic.shape
        masks.append(np.zeros((h, w), dtype=bool))
        labels.append(IGNORE_LABEL)

    return np.stack(masks, axis=0), np.array(labels, dtype=np.int32)


class WODMask2FormerDataset(Dataset):
    """Dataset that returns samples formatted for HF Mask2FormerForUniversalSegmentation.

    Each __getitem__ returns:
        pixel_values:  (3, H, W) float32, ImageNet normalised
        mask_labels:   (N, H, W) float32 binary masks
        class_labels:  (N,)      int64 class IDs
        meta:          dict with segment info
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "training",
        cameras: Optional[List[int]] = None,
        crop_size: Tuple[int, int] = (512, 512),
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        flip_prob: float = 0.5,
        color_jitter: bool = True,
        full_resolution: bool = False,
        cache_dir: Optional[str] = None,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.5,
        copy_paste_max: int = 3,
    ):
        self._base = WODPVPSDataset(
            dataset_root=dataset_root,
            split=split,
            cameras=cameras or [1],
            cache_dir=cache_dir,
        )

        is_train = split == "training"
        if full_resolution:
            self._spatial = build_fullres_val_transforms(
                ignore_label=IGNORE_LABEL,
            )
        elif is_train:
            self._spatial = build_train_transforms(
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                flip_prob=flip_prob,
                color_jitter=color_jitter,
                ignore_label=IGNORE_LABEL,
            )
        else:
            self._spatial = build_val_transforms(
                crop_size=crop_size,
                ignore_label=IGNORE_LABEL,
            )

        # Copy-paste augmentation (training only, applied after the target's
        # spatial augmentation and colour jitter)
        self._copy_paste = None
        if copy_paste and is_train:
            # Build spatial-only augmentation for source frames (no color jitter)
            spatial_for_source = build_train_transforms(
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                flip_prob=flip_prob,
                color_jitter=False,
                ignore_label=IGNORE_LABEL,
            )
            self._copy_paste = CopyPaste(
                dataset=self._base,
                spatial_aug=spatial_for_source,
                paste_prob=copy_paste_prob,
                max_paste=copy_paste_max,
            )

        self._format = ToTensorNormalize()

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Dict:
        raw = self._base[idx]

        sample = {
            "image":          raw["image"],
            "semantic_label": raw["semantic_label"],
            "instance_label": raw["instance_label"],
            "meta":           raw.get("meta", {}),
        }

        sample = self._spatial(sample)

        if self._copy_paste is not None:
            sample = self._copy_paste(sample)

        sample = self._format(sample)

        masks_np, labels_np = panoptic_to_segments(
            sample["semantic_label"], sample["instance_label"]
        )

        result = {
            "pixel_values":   sample["image"],                               # (3, H, W)
            "mask_labels":    torch.from_numpy(masks_np.astype(np.float32)), # (N, H, W)
            "class_labels":   torch.from_numpy(labels_np.astype(np.int64)),  # (N,)
            "semantic_label": sample["semantic_label"],                      # (H, W) int32
            "instance_label": sample["instance_label"],                      # (H, W) int32
            "meta":           raw.get("meta", {}),
        }
        if "original_size" in sample:
            result["original_size"] = sample["original_size"]
        return result


# =============================================================================
# Collate functions
# =============================================================================

def collate_fn(batch: List[Dict]) -> Dict:
    """DataLoader collate: stack pixel_values, keep mask/class lists."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels":  [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
        "meta":         [b["meta"] for b in batch],
    }
