"""Ground-truth helpers shared by the tracking evaluation scripts."""

import numpy as np


def build_gt_global_instance(gt_inst: np.ndarray, mapping: dict) -> np.ndarray:
    """Remap local instance IDs to global tracked IDs."""
    gt_global = np.zeros_like(gt_inst)
    for loc, glob, trk in zip(
        mapping["local_ids"], mapping["global_ids"], mapping["is_tracked"]
    ):
        if trk and loc > 0 and glob > 0:
            gt_global[gt_inst == loc] = glob
    return gt_global


def build_pixel_weights(coverage: np.ndarray) -> np.ndarray:
    """Build pixel weights for wSTQ from camera coverage map."""
    return np.where(coverage > 0, 1.0 / coverage.astype(np.float32), 0.0)
