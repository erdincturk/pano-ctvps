"""Per-class anchor weights for the contrastive loss.

The weight of a thing class c is

    w(c) = clip(sqrt(f_med / f(c)), 0.5, 4.0)

where f(c) is the number of training-split instances of class c and f_med the
median over the thing classes. Frequent classes (car, pedestrian) are damped
and rare classes are boosted.

The counts are stored as a literal so that the loss needs no I/O.

Example::

    from panoctvps.data.class_weights import WOD_THING_ANCHOR_WEIGHTS
    # {2: 0.5, 3: 0.7395, 4: 0.9356, 5: 1.0798, 8: 1.7139,
    #  9: 0.5, 10: 1.0896, 11: 3.1060}
"""

import math
from typing import Dict, Tuple


# Training-split instance counts per thing class (WOD v2.0.1), summed over the
# five cameras. Class IDs from panoctvps/data/constants.py
# (THING_CLASSES = {2, 3, 4, 5, 8, 9, 10, 11}).
WOD_TRAINING_THING_INSTANCE_COUNTS: Dict[int, int] = {
    2:  75886,   # car
    3:   1676,   # truck
    4:   1047,   # bus
    5:    786,   # other_large_vehicle
    8:    312,   # trailer
    9:  24733,   # pedestrian
    10:   772,   # cyclist
    11:    95,   # motorcyclist
}


def compute_class_weights(
    class_counts: Dict[int, int],
    clamp: Tuple[float, float] = (0.5, 4.0),
) -> Dict[int, float]:
    """Compute per-class anchor weights ``clip(sqrt(median / count_c), *clamp)``.

    Args:
        class_counts: ``{class_id: count}`` over the training split.
        clamp: ``(min_w, max_w)`` applied after the square root.

    Returns:
        ``{class_id: weight}`` with the same keys as ``class_counts``.

    Raises:
        ValueError: if a count is not positive or ``clamp`` is malformed.
    """
    if clamp[0] > clamp[1] or clamp[0] < 0.0:
        raise ValueError(f"Invalid clamp range: {clamp}")
    if not class_counts:
        return {}

    counts = sorted(class_counts.values())
    n = len(counts)
    median = (
        counts[n // 2]
        if n % 2 == 1
        else 0.5 * (counts[n // 2 - 1] + counts[n // 2])
    )

    weights: Dict[int, float] = {}
    for cid, count in class_counts.items():
        if count <= 0:
            raise ValueError(f"Non-positive count for class {cid}: {count}")
        w = math.sqrt(median / count)
        weights[cid] = max(clamp[0], min(clamp[1], w))

    return weights


# Weights used in training (clip to [0.5, 4.0]).
WOD_THING_ANCHOR_WEIGHTS: Dict[int, float] = compute_class_weights(
    WOD_TRAINING_THING_INSTANCE_COUNTS,
    clamp=(0.5, 4.0),
)
