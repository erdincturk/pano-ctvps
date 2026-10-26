"""Training utilities shared by the segmenter and tracker training scripts.

Provides config loading, LR scheduling, checkpointing, and CSV logging.
"""

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config with _base_ merging.

    Supports a list of base configs that are merged in order, with the
    main config taking precedence. Paths in _base_ are relative to the
    config file's directory.

    Args:
        config_path: Path to the experiment YAML config.

    Returns:
        Fully merged config dict.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base_list = cfg.pop("_base_", [])
    if isinstance(base_list, str):
        base_list = [base_list]
    merged: dict = {}
    config_dir = Path(config_path).parent
    for bp in base_list:
        with open(config_dir / bp) as f:
            base_cfg = yaml.safe_load(f)
        _deep_merge(merged, base_cfg)
    _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# =============================================================================
# LR scheduling
# =============================================================================

def get_lr_scale(
    iteration: int,
    warmup_iters: int,
    milestones: List[int],
    gamma: float,
) -> float:
    """Compute LR multiplier for step decay with linear warmup.

    Args:
        iteration: Current training iteration.
        warmup_iters: Number of warmup iterations.
        milestones: Iterations at which to decay by gamma.
        gamma: Decay factor at each milestone.

    Returns:
        LR scale factor to multiply base LR by.
    """
    if iteration < warmup_iters:
        return max(1e-6, iteration / warmup_iters)
    scale = 1.0
    for ms in milestones:
        if iteration >= ms:
            scale *= gamma
    return scale


def get_lr_scale_cosine(
    iteration: int,
    warmup_iters: int,
    max_iterations: int,
    base_lr: float,
    eta_min: float,
) -> float:
    """Compute LR multiplier for cosine decay with linear warmup.

    Linear warmup from 0 to base_lr over warmup_iters, then cosine annealing
    from base_lr down to eta_min over the remaining iterations.

    Returns the multiplier to apply to base_lr (so the effective LR at the
    final step is eta_min / base_lr × base_lr = eta_min, matching the
    PyTorch CosineAnnealingLR convention).
    """
    if iteration < warmup_iters:
        return max(1e-6, iteration / warmup_iters)
    progress = (iteration - warmup_iters) / max(1, max_iterations - warmup_iters)
    progress = min(1.0, max(0.0, progress))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = eta_min / base_lr
    return min_ratio + (1.0 - min_ratio) * cos_decay


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    output_dir: str,
    iteration: int,
    keep_last: int = 3,
) -> str:
    """Save a training checkpoint and prune old ones.

    Args:
        state: Arbitrary state dict (model weights, optimizer, scaler, etc).
        output_dir: Directory to save checkpoints in.
        iteration: Current iteration (used in filename).
        keep_last: Number of recent checkpoints to keep.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ckpt_{iteration:07d}.pth")
    torch.save(state, path)
    if keep_last > 0:
        ckpts = sorted(Path(output_dir).glob("ckpt_*.pth"))
        for old in ckpts[:-keep_last]:
            old.unlink()
    return path


def save_best(state: Dict[str, Any], output_dir: str) -> str:
    """Save the best checkpoint (overwrites previous best).

    Args:
        state: Arbitrary state dict.
        output_dir: Directory to save in.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "best.pth")
    torch.save(state, path)
    return path


def load_latest_checkpoint(
    output_dir: str,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Load the most recent ckpt_*.pth from output_dir.

    Args:
        output_dir: Directory containing checkpoints.
        device: Device to map tensors to.

    Returns:
        State dict, or None if no checkpoint found.
    """
    ckpts = sorted(Path(output_dir).glob("ckpt_*.pth"))
    if not ckpts:
        return None
    return torch.load(str(ckpts[-1]), map_location=device, weights_only=False)


# =============================================================================
# CSV logging
# =============================================================================

class TrainingLogger:
    """Simple CSV logger for training metrics.

    Opens in append mode so resuming training continues the same file.
    Writes a header row only when starting from scratch (iteration 0).
    """

    def __init__(
        self,
        output_dir: str,
        columns: List[str],
        resume_iteration: int = 0,
        filename: str = "train_log.csv",
    ):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, filename)
        self._file = open(self._path, "a", newline="")
        self._writer = csv.writer(self._file)
        self._columns = columns
        if resume_iteration == 0:
            self._writer.writerow(columns)

    def log(self, **values: Any) -> None:
        """Write a row. Keys must match column names."""
        row = [values.get(col, "") for col in self._columns]
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()
