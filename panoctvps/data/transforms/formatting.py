"""Convert numpy samples to tensors, normalize with ImageNet stats.

These are applied AFTER spatial augmentations and operate on the dict produced
by augmentations.py. Returns tensors ready for model input.
"""

from typing import Dict

import numpy as np
import torch


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ToTensor:
    """Convert image (H,W,3) uint8 → (3,H,W) float32 in [0,1].

    Labels stay as numpy; they are converted selectively downstream.
    """

    def __call__(self, sample: Dict) -> Dict:
        img = sample["image"].astype(np.float32) / 255.0  # [0, 1]
        sample["image"] = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)
        return sample


class Normalize:
    """Normalize image tensor with ImageNet mean/std (in-place)."""

    def __init__(
        self,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray  = IMAGENET_STD,
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample


class ToTensorNormalize:
    """Convenience: ToTensor + Normalize in one step."""

    def __init__(self):
        self._to_tensor  = ToTensor()
        self._normalize  = Normalize()

    def __call__(self, sample: Dict) -> Dict:
        sample = self._to_tensor(sample)
        sample = self._normalize(sample)
        return sample