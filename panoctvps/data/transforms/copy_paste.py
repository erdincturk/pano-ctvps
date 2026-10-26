"""Copy-paste augmentation for panoptic segmentation.

Pastes thing instances from a randomly sampled source image onto the target
image, updating semantic and instance labels. This increases instance diversity
and teaches the model to segment overlapping objects.

Reference: Ghiasi et al. "Simple Copy-Paste is a Strong Data Augmentation
Method for Instance Segmentation" (CVPR 2021)

Usage:
    copy_paste = CopyPaste(dataset=base_dataset, spatial_aug=spatial_transform)
    augmented_sample = copy_paste(sample)
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from panoctvps.data.constants import THING_CLASSES


class CopyPaste:
    """Copy-paste augmentation that pastes thing instances from source images.

    Applied to the target after its full training augmentation (resize +
    crop + flip, then colour jitter when enabled). Source frames receive the
    same spatial augmentation but no colour jitter, so source and target have
    the same spatial size and pasted pixels are not colour-jittered.
    """

    def __init__(
        self,
        dataset,
        spatial_aug,
        paste_prob: float = 0.5,
        max_paste: int = 3,
        min_area: int = 100,
    ):
        """
        Args:
            dataset: WODPVPSDataset instance for sampling source frames.
            spatial_aug: Compose of spatial augmentations (resize, crop, flip)
                to apply to source frames so they match target spatial size.
            paste_prob: probability of applying copy-paste per sample.
            max_paste: maximum number of instances to paste.
            min_area: minimum instance area (pixels) to be eligible for pasting.
        """
        self.dataset = dataset
        self.spatial_aug = spatial_aug
        self.paste_prob = paste_prob
        self.max_paste = max_paste
        self.min_area = min_area
        self._thing_set = set(THING_CLASSES)

    def _extract_thing_instances(
        self, semantic: np.ndarray, instance: np.ndarray, image: np.ndarray,
    ) -> List[Dict]:
        """Extract individual thing instances from a frame.

        Returns:
            List of dicts, each with: mask (H,W bool), class_id, image_crop (H,W,3),
            area (int).
        """
        instances = []
        for cls_id in THING_CLASSES:
            cls_mask = semantic == cls_id
            if not cls_mask.any():
                continue
            for iid in np.unique(instance[cls_mask]):
                if iid == 0:
                    continue
                mask = cls_mask & (instance == iid)
                area = int(mask.sum())
                if area < self.min_area:
                    continue
                instances.append({
                    "mask": mask,
                    "class_id": cls_id,
                    "image": image,
                    "area": area,
                })
        return instances

    def __call__(self, sample: Dict) -> Dict:
        """Apply copy-paste augmentation to a sample.

        Args:
            sample: dict with image (H,W,3), semantic_label (H,W), instance_label (H,W).
                Must already be spatially augmented (cropped to target size).

        Returns:
            Augmented sample with pasted instances.
        """
        if random.random() > self.paste_prob:
            return sample

        # Sample a random source frame and apply same spatial augmentation
        src_idx = random.randint(0, len(self.dataset) - 1)
        src_raw = self.dataset[src_idx]
        src_sample = {
            "image": src_raw["image"],
            "semantic_label": src_raw["semantic_label"],
            "instance_label": src_raw["instance_label"],
        }
        src_sample = self.spatial_aug(src_sample)

        # Extract thing instances from source
        src_instances = self._extract_thing_instances(
            src_sample["semantic_label"],
            src_sample["instance_label"],
            src_sample["image"],
        )

        if not src_instances:
            return sample

        # Select random subset to paste
        n_paste = random.randint(1, min(self.max_paste, len(src_instances)))
        selected = random.sample(src_instances, n_paste)

        # Sort by area (largest first) — smaller instances occlude larger ones
        selected.sort(key=lambda x: -x["area"])

        # Determine next available instance ID in target
        target_image = sample["image"].copy()
        target_semantic = sample["semantic_label"].copy()
        target_instance = sample["instance_label"].copy()
        next_id = int(target_instance.max()) + 1

        for inst in selected:
            mask = inst["mask"]

            # Paste image pixels
            target_image[mask] = inst["image"][mask]

            # Update semantic label
            target_semantic[mask] = inst["class_id"]

            # Update instance label with new unique ID
            target_instance[mask] = next_id
            next_id += 1

        sample["image"] = target_image
        sample["semantic_label"] = target_semantic
        sample["instance_label"] = target_instance

        return sample
