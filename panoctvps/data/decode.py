"""
Decoding Utilities
==================
PNG/JPEG decoding and panoptic label encoding/decoding.
No TensorFlow dependency — pure numpy + PIL.
"""

import io
from typing import Tuple

import numpy as np
from PIL import Image

from panoctvps.data.constants import PANOPTIC_LABEL_DIVISOR


# =============================================================================
# PNG / JPEG decoding
# =============================================================================

def decode_png(png_bytes: bytes, dtype=np.uint16) -> np.ndarray:
    """Decode PNG bytes to numpy array.

    Args:
        png_bytes: raw PNG file bytes.
        dtype: target dtype (uint16 for panoptic, uint8 for coverage).

    Returns:
        2D numpy array.
    """
    return np.array(Image.open(io.BytesIO(png_bytes))).astype(dtype)


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array.

    Returns:
        (H, W, 3) uint8 array.
    """
    return np.array(Image.open(io.BytesIO(jpeg_bytes)))


# =============================================================================
# Panoptic label encoding / decoding
# =============================================================================

def decode_panoptic(
    panoptic_label: np.ndarray,
    divisor: int = PANOPTIC_LABEL_DIVISOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split panoptic label into semantic and instance components.

    Args:
        panoptic_label: uint16 array, value = semantic_id * divisor + instance_id.
        divisor: panoptic label divisor (default 1000).

    Returns:
        semantic_ids: (H, W) int32, class IDs 0-28.
        instance_ids: (H, W) int32, instance IDs (0 = no instance).
    """
    return np.divmod(panoptic_label.astype(np.int32), divisor)


def encode_panoptic(
    semantic_ids: np.ndarray,
    instance_ids: np.ndarray,
    divisor: int = PANOPTIC_LABEL_DIVISOR,
) -> np.ndarray:
    """Encode semantic + instance into panoptic label.

    Args:
        semantic_ids: (H, W) array of class IDs.
        instance_ids: (H, W) array of instance IDs.
        divisor: panoptic label divisor.

    Returns:
        (H, W) uint32 panoptic label.
    """
    return semantic_ids.astype(np.uint32) * divisor + instance_ids.astype(np.uint32)
