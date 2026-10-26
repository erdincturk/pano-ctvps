"""Training augmentations for panoptic segmentation.

All transforms operate on a sample dict with keys:
    image:          (H, W, 3) uint8 numpy array
    semantic_label: (H, W) int32 numpy array
    instance_label: (H, W) int32 numpy array

Transforms apply consistently to image AND labels (same crop/flip/resize).
Colour jitter is image-only.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomResize:
    """Randomly scale image and labels by a factor in [min_scale, max_scale]."""

    def __init__(self, min_scale: float = 0.5, max_scale: float = 2.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample: Dict) -> Dict:
        scale = random.uniform(self.min_scale, self.max_scale)
        h, w = sample["image"].shape[:2]
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        sample["image"] = np.array(
            Image.fromarray(sample["image"]).resize((new_w, new_h), Image.BILINEAR)
        )
        sample["semantic_label"] = np.array(
            Image.fromarray(sample["semantic_label"].astype(np.int32)).resize(
                (new_w, new_h), Image.NEAREST
            ),
            dtype=np.int32,
        )
        sample["instance_label"] = np.array(
            Image.fromarray(sample["instance_label"].astype(np.int32)).resize(
                (new_w, new_h), Image.NEAREST
            ),
            dtype=np.int32,
        )
        return sample


class RandomCrop:
    """Random crop to (crop_h, crop_w). Pads smaller images before cropping."""

    def __init__(self, crop_size: Tuple[int, int], ignore_label: int = 0):
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label

    def __call__(self, sample: Dict) -> Dict:
        h, w = sample["image"].shape[:2]

        # Pad if image is smaller than crop size
        pad_h = max(0, self.crop_h - h)
        pad_w = max(0, self.crop_w - w)
        if pad_h > 0 or pad_w > 0:
            sample["image"] = np.pad(
                sample["image"], ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )
            sample["semantic_label"] = np.pad(
                sample["semantic_label"],
                ((0, pad_h), (0, pad_w)),
                constant_values=self.ignore_label,
            )
            sample["instance_label"] = np.pad(
                sample["instance_label"],
                ((0, pad_h), (0, pad_w)),
                constant_values=0,
            )
            h, w = sample["image"].shape[:2]

        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)

        sample["image"] = sample["image"][top:top + self.crop_h, left:left + self.crop_w]
        sample["semantic_label"] = sample["semantic_label"][top:top + self.crop_h, left:left + self.crop_w]
        sample["instance_label"] = sample["instance_label"][top:top + self.crop_h, left:left + self.crop_w]
        return sample


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            sample["image"] = np.fliplr(sample["image"]).copy()
            sample["semantic_label"] = np.fliplr(sample["semantic_label"]).copy()
            sample["instance_label"] = np.fliplr(sample["instance_label"]).copy()
        return sample


class ColorJitter:
    """Random colour jitter (torchvision) applied to the image only."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        from torchvision import transforms as T
        self._jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = np.array(self._jitter(Image.fromarray(sample["image"])))
        return sample


def build_train_transforms(
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    crop_size: Tuple[int, int] = (512, 512),
    flip_prob: float = 0.5,
    color_jitter: bool = True,
    ignore_label: int = 0,
) -> Compose:
    transforms = [
        RandomResize(min_scale=min_scale, max_scale=max_scale),
        RandomCrop(crop_size=crop_size, ignore_label=ignore_label),
        RandomHorizontalFlip(prob=flip_prob),
    ]
    if color_jitter:
        transforms.append(ColorJitter())
    return Compose(transforms)


class CenterCrop:
    """Deterministic center crop. Pads smaller images before cropping."""

    def __init__(self, crop_size: Tuple[int, int], ignore_label: int = 0):
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label

    def __call__(self, sample: Dict) -> Dict:
        h, w = sample["image"].shape[:2]

        pad_h = max(0, self.crop_h - h)
        pad_w = max(0, self.crop_w - w)
        if pad_h > 0 or pad_w > 0:
            sample["image"] = np.pad(
                sample["image"], ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )
            sample["semantic_label"] = np.pad(
                sample["semantic_label"],
                ((0, pad_h), (0, pad_w)),
                constant_values=self.ignore_label,
            )
            sample["instance_label"] = np.pad(
                sample["instance_label"],
                ((0, pad_h), (0, pad_w)),
                constant_values=0,
            )
            h, w = sample["image"].shape[:2]

        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2

        sample["image"] = sample["image"][top:top + self.crop_h, left:left + self.crop_w]
        sample["semantic_label"] = sample["semantic_label"][
            top:top + self.crop_h, left:left + self.crop_w
        ]
        sample["instance_label"] = sample["instance_label"][
            top:top + self.crop_h, left:left + self.crop_w
        ]
        return sample


def build_val_transforms(
    crop_size: Tuple[int, int] = (512, 512),
    ignore_label: int = 0,
) -> Compose:
    """Deterministic centre crop (used for the in-training validation PQ)."""
    return Compose([
        CenterCrop(crop_size=crop_size, ignore_label=ignore_label),
    ])


class PadToMultiple:
    """Pad image and labels at the bottom and right so spatial dims are
    divisible by `multiple`.

    The image is padded with 0 (black, before normalisation), the semantic
    label with `ignore_label` and the instance label with 0. Records the
    original (H, W) in sample["original_size"] for unpadding after model
    inference.

    Args:
        multiple:     spatial dims are padded to a multiple of this (32 for M2F).
        ignore_label: fill value for semantic label padding.
    """

    def __init__(self, multiple: int = 32, ignore_label: int = 0):
        self.multiple = multiple
        self.ignore_label = ignore_label

    def __call__(self, sample: Dict) -> Dict:
        h, w = sample["image"].shape[:2]
        sample["original_size"] = (h, w)

        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple

        if pad_h > 0 or pad_w > 0:
            h_pad = (0, pad_h)
            sample["image"] = np.pad(
                sample["image"], (h_pad, (0, pad_w), (0, 0)), mode="constant"
            )
            sample["semantic_label"] = np.pad(
                sample["semantic_label"],
                (h_pad, (0, pad_w)),
                constant_values=self.ignore_label,
            )
            sample["instance_label"] = np.pad(
                sample["instance_label"],
                (h_pad, (0, pad_w)),
                constant_values=0,
            )
        return sample


def build_fullres_val_transforms(
    ignore_label: int = 0,
    multiple: int = 32,
) -> Compose:
    """Full-resolution evaluation: bottom/right padding to a multiple of 32
    (no resize, no crop)."""
    return Compose([
        PadToMultiple(multiple=multiple, ignore_label=ignore_label),
    ])