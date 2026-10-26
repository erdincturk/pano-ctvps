"""
STQ / wSTQ: Segmentation and Tracking Quality
==============================================

Primary evaluation metric for WOD PVPS benchmark.

Reference:
    - Weber et al. 2021, "STEP: Segmenting and Tracking Every Pixel" (NeurIPS)
    - Mei et al. 2022, "WOD: Panoramic Video Panoptic Segmentation" (Eq. 3)

STQ = sqrt(AQ * SQ)

where:
    SQ = mean IoU over all semantic classes (thing + stuff)
    AQ = mean weighted IoU over all GT instances (thing classes only)

wSTQ extends STQ for multi-camera by weighting each pixel by 1/N,
where N = number of cameras covering that pixel.

Usage:
    metric = STQuality(num_classes=29, thing_list=[2,3,4,5,8,9,10,11])
    for frame in sequence:
        metric.update(pred_semantic, pred_instance, gt_semantic, gt_instance, weights)
    results = metric.result()
    # results = {"STQ": stq, "AQ": aq, "SQ": sq, ...}
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES


class STQuality:
    """Segmentation and Tracking Quality metric.

    Accumulates predictions across all frames in a sequence, then
    computes STQ = sqrt(AQ * SQ) at the end.

    Supports pixel-level weighting for wSTQ computation (multi-camera).

    Ported from deeplab2.evaluation.segmentation_and_tracking_quality
    to standalone numpy (no TensorFlow dependency).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        thing_list: Optional[List[int]] = None,
        ignore_label: int = 0,
        label_divisor: int = 1000,
        offset: int = 256**3,
    ):
        """
        Args:
            num_classes: number of semantic classes (29 for WOD PVPS).
            thing_list: list of semantic class IDs that have instances.
            ignore_label: semantic class ID to ignore (0 = UNDEFINED).
            label_divisor: divisor for panoptic label encoding.
            offset: large offset to separate GT and pred instance IDs
                    when building the joint ID space.
        """
        self.num_classes = num_classes
        self.thing_list = thing_list if thing_list is not None else THING_CLASSES
        self.ignore_label = ignore_label
        self.label_divisor = label_divisor
        self.offset = offset

        # Boolean mask: which classes are "things" (have instances)
        self.is_thing = np.zeros(num_classes, dtype=bool)
        for c in self.thing_list:
            self.is_thing[c] = True

        self.reset()

    def reset(self):
        """Clear all accumulated state."""
        # SQ accumulators: per-class intersection and union (weighted)
        self._iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        self._iou_union_per_class = np.zeros(self.num_classes, dtype=np.float64)

        # AQ accumulators: instance-level confusion data
        # Maps (gt_instance_id, pred_instance_id) -> weighted pixel count
        self._instance_confusion = defaultdict(float)
        # Maps gt_instance_id -> total weighted pixel count
        self._gt_instance_size = defaultdict(float)
        # Maps pred_instance_id -> total weighted pixel count
        self._pred_instance_size = defaultdict(float)

        # Track which GT instance IDs exist and their semantic class
        self._gt_instance_ids = set()
        self._gt_instance_class = {}  # gt_instance_id -> semantic class

        self._num_frames = 0

    def update(
        self,
        pred_semantic: np.ndarray,
        pred_instance: np.ndarray,
        gt_semantic: np.ndarray,
        gt_instance: np.ndarray,
        pixel_weights: Optional[np.ndarray] = None,
    ):
        """Update metric with one frame of predictions.

        All arrays must be (H, W) with matching shapes.

        Args:
            pred_semantic: predicted semantic class IDs.
            pred_instance: predicted instance IDs (0 = no instance / stuff).
            gt_semantic: ground truth semantic class IDs.
            gt_instance: ground truth instance IDs (0 = no instance / stuff).
            pixel_weights: optional (H, W) float array of per-pixel weights.
                          For wSTQ, use 1.0 / num_cameras_covered.
                          Default: all ones (standard STQ).
        """
        assert pred_semantic.shape == gt_semantic.shape
        assert pred_instance.shape == gt_instance.shape
        assert pred_semantic.shape == pred_instance.shape

        if pixel_weights is None:
            pixel_weights = np.ones_like(gt_semantic, dtype=np.float64)
        else:
            pixel_weights = pixel_weights.astype(np.float64)

        # Flatten for efficiency
        pred_sem = pred_semantic.ravel().astype(np.int32)
        pred_inst = pred_instance.ravel().astype(np.int64)
        gt_sem = gt_semantic.ravel().astype(np.int32)
        gt_inst = gt_instance.ravel().astype(np.int64)
        weights = pixel_weights.ravel()

        # Ignore pixels where GT is the ignore label
        valid = gt_sem != self.ignore_label

        # ---- SQ: Semantic segmentation quality (weighted mIoU) ----
        # Note: predictions on ignore-label GT pixels still count as FP
        # (matching DeepLab2 confusion matrix behavior)
        for c in range(self.num_classes):
            if c == self.ignore_label:
                continue

            pred_c = pred_sem == c           # all predicted pixels for class c
            gt_c = (gt_sem == c) & valid     # GT pixels for class c (excluding ignore)

            intersection_mask = pred_c & gt_c
            union_mask = pred_c | gt_c

            self._iou_per_class[c] += np.sum(weights[intersection_mask])
            self._iou_union_per_class[c] += np.sum(weights[union_mask])

        # ---- AQ: Association quality (instance tracking) ----
        # Only consider pixels that belong to "thing" classes in GT
        gt_is_thing = np.zeros(len(gt_sem), dtype=bool)
        for c in self.thing_list:
            gt_is_thing |= (gt_sem == c)
        thing_mask = gt_is_thing & valid & (gt_inst > 0)

        if thing_mask.sum() == 0:
            self._num_frames += 1
            return

        # Filter to thing pixels only
        gt_inst_thing = gt_inst[thing_mask]
        pred_inst_thing = pred_inst[thing_mask]
        pred_sem_thing = pred_sem[thing_mask]
        gt_sem_thing = gt_sem[thing_mask]
        weights_thing = weights[thing_mask]

        # For predicted instances: only count if pred semantic is also thing
        pred_is_thing = np.zeros(len(pred_sem_thing), dtype=bool)
        for c in self.thing_list:
            pred_is_thing |= (pred_sem_thing == c)

        # Build unique instance IDs incorporating semantic class:
        # id = semantic * offset + instance_id
        # This ensures instances of different classes don't merge.
        gt_ids = gt_sem_thing.astype(np.int64) * self.offset + gt_inst_thing

        pred_ids = np.where(
            pred_is_thing & (pred_inst_thing > 0),
            pred_sem_thing.astype(np.int64) * self.offset + pred_inst_thing,
            0,  # stuff predictions or instance_id=0 -> no association
        )

        # Accumulate GT instance sizes
        unique_gt_ids = np.unique(gt_ids)
        for gt_id in unique_gt_ids:
            gt_id = int(gt_id)
            if gt_id == 0:
                continue
            gt_mask = gt_ids == gt_id
            self._gt_instance_ids.add(gt_id)
            self._gt_instance_size[gt_id] += np.sum(weights_thing[gt_mask])
            # Record semantic class (derived from composite ID)
            if gt_id not in self._gt_instance_class:
                self._gt_instance_class[gt_id] = int(gt_id // self.offset)

            # Find overlapping pred instances for confusion accumulation
            overlap_pred = pred_ids[gt_mask]
            overlap_w = weights_thing[gt_mask]

            for pred_id in np.unique(overlap_pred):
                pred_id = int(pred_id)
                match = overlap_pred == pred_id
                self._instance_confusion[(gt_id, pred_id)] += np.sum(overlap_w[match])

        # Accumulate pred instance sizes using prediction_mask
        # (matching DeepLab2: pred is thing class AND GT is not crowd)
        # prediction_mask = pred semantic is thing AND NOT (GT is thing with inst=0)
        is_crowd = gt_is_thing & valid & (gt_inst == 0)
        not_crowd = ~is_crowd

        pred_is_thing_all = np.zeros(len(pred_sem), dtype=bool)
        for c in self.thing_list:
            pred_is_thing_all |= (pred_sem == c)
        prediction_mask = pred_is_thing_all & not_crowd & (pred_inst > 0)

        if prediction_mask.sum() > 0:
            pred_ids_pm = pred_sem[prediction_mask].astype(np.int64) * self.offset + pred_inst[prediction_mask]
            weights_pm = weights[prediction_mask]

            unique_pred_ids = np.unique(pred_ids_pm)
            for pred_id in unique_pred_ids:
                pred_id = int(pred_id)
                if pred_id == 0:
                    continue
                pred_mask = pred_ids_pm == pred_id
                self._pred_instance_size[pred_id] += np.sum(weights_pm[pred_mask])

        self._num_frames += 1

    def result(self) -> Dict[str, float]:
        """Compute final STQ/wSTQ metrics.

        Returns:
            Dict with keys: "STQ", "AQ", "SQ", and per-class IoU values.
        """
        # ---- SQ: mean IoU over valid classes ----
        valid_classes = (self._iou_union_per_class > 0) & (
            np.arange(self.num_classes) != self.ignore_label
        )
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)
        per_class_iou[valid_classes] = (
            self._iou_per_class[valid_classes] / self._iou_union_per_class[valid_classes]
        )

        if valid_classes.sum() > 0:
            sq = per_class_iou[valid_classes].mean()
        else:
            sq = 0.0

        # ---- AQ: mean weighted IoU over GT instances ----
        # Also track per-class AQ
        per_class_aq_sum = defaultdict(float)
        per_class_aq_count = defaultdict(int)

        if len(self._gt_instance_ids) == 0:
            aq = 0.0
        else:
            aq_sum = 0.0
            for gt_id in self._gt_instance_ids:
                gt_size = self._gt_instance_size[gt_id]
                if gt_size == 0:
                    continue

                gt_class = self._gt_instance_class.get(gt_id, -1)

                # Sum over all predicted instances overlapping this GT instance
                # AQ per instance = (1/|g_id|) * Σ TPA(z_f, z_g) * IoU_id(z_f, z_g)
                gt_weighted_score = 0.0

                for (g, p), tpa in self._instance_confusion.items():
                    if g != gt_id or p == 0:
                        continue

                    # IoU_id = TPA / (TPA + FNA + FPA)
                    fna = gt_size - tpa
                    fpa = self._pred_instance_size.get(p, 0) - tpa
                    iou_id = tpa / max(tpa + fna + fpa, 1e-10)

                    gt_weighted_score += tpa * iou_id

                instance_aq = gt_weighted_score / max(gt_size, 1e-10)
                aq_sum += instance_aq

                if gt_class >= 0:
                    per_class_aq_sum[gt_class] += instance_aq
                    per_class_aq_count[gt_class] += 1

            aq = aq_sum / len(self._gt_instance_ids)

        # Build per-class AQ array
        per_class_aq = np.zeros(self.num_classes, dtype=np.float64)
        per_class_aq_n = np.zeros(self.num_classes, dtype=np.int64)
        for c in self.thing_list:
            if per_class_aq_count[c] > 0:
                per_class_aq[c] = per_class_aq_sum[c] / per_class_aq_count[c]
                per_class_aq_n[c] = per_class_aq_count[c]

        # ---- STQ = sqrt(AQ * SQ) ----
        stq = float(np.sqrt(max(aq * sq, 0.0)))

        return {
            "STQ": stq,
            "AQ": float(aq),
            "SQ": float(sq),
            "num_frames": self._num_frames,
            "num_gt_instances": len(self._gt_instance_ids),
            "per_class_sq": per_class_iou.tolist(),
            "per_class_aq": per_class_aq.tolist(),
            "per_class_aq_n": per_class_aq_n.tolist(),
        }


def merge_sequence_results(sequence_metrics: List['STQuality']) -> Dict[str, float]:
    """Merge per-sequence STQuality results into aggregate metrics.

    Matches DeepLab2's aggregation:
    - AQ: sum(aq_numerator_per_seq) / sum(num_gt_instances_per_seq)
    - SQ: mean IoU from summed confusion matrix across all sequences
    - STQ: sqrt(AQ * SQ)

    Args:
        sequence_metrics: list of STQuality objects, one per sequence.

    Returns:
        Dict with "STQ", "AQ", "SQ", etc.
    """
    if not sequence_metrics:
        return {"STQ": 0.0, "AQ": 0.0, "SQ": 0.0, "num_frames": 0,
                "num_gt_instances": 0, "per_class_sq": []}

    ref = sequence_metrics[0]
    num_classes = ref.num_classes

    # Aggregate SQ: sum confusion matrices (intersection + union per class)
    total_iou = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    for m in sequence_metrics:
        total_iou += m._iou_per_class
        total_union += m._iou_union_per_class

    valid_classes = (total_union > 0) & (np.arange(num_classes) != ref.ignore_label)
    per_class_iou = np.zeros(num_classes, dtype=np.float64)
    per_class_iou[valid_classes] = total_iou[valid_classes] / total_union[valid_classes]
    sq = float(per_class_iou[valid_classes].mean()) if valid_classes.sum() > 0 else 0.0

    # Aggregate AQ: sum numerators / sum GT instance counts
    # Also aggregate per-class AQ
    total_aq_numerator = 0.0
    total_gt_instances = 0
    total_frames = 0
    per_class_aq_num = np.zeros(num_classes, dtype=np.float64)
    per_class_aq_den = np.zeros(num_classes, dtype=np.int64)

    for m in sequence_metrics:
        r = m.result()
        n_gt = len(m._gt_instance_ids)
        total_gt_instances += n_gt
        total_frames += m._num_frames
        total_aq_numerator += r["AQ"] * n_gt

        # Per-class AQ: accumulate numerator and count
        for c in ref.thing_list:
            pc_aq = r["per_class_aq"][c]
            pc_n = r["per_class_aq_n"][c]
            per_class_aq_num[c] += pc_aq * pc_n
            per_class_aq_den[c] += pc_n

    aq = total_aq_numerator / max(total_gt_instances, 1e-10)
    stq = float(np.sqrt(max(aq * sq, 0.0)))

    per_class_aq = np.zeros(num_classes, dtype=np.float64)
    for c in ref.thing_list:
        if per_class_aq_den[c] > 0:
            per_class_aq[c] = per_class_aq_num[c] / per_class_aq_den[c]

    return {
        "STQ": stq,
        "AQ": float(aq),
        "SQ": sq,
        "num_frames": total_frames,
        "num_gt_instances": total_gt_instances,
        "per_class_sq": per_class_iou.tolist(),
        "per_class_aq": per_class_aq.tolist(),
        "per_class_aq_n": per_class_aq_den.tolist(),
    }


def compute_stq(
    pred_semantics: List[np.ndarray],
    pred_instances: List[np.ndarray],
    gt_semantics: List[np.ndarray],
    gt_instances: List[np.ndarray],
    pixel_weights: Optional[List[np.ndarray]] = None,
    num_classes: int = NUM_CLASSES,
    thing_list: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Convenience function: compute STQ/wSTQ for a full sequence.

    Args:
        pred_semantics: list of (H, W) predicted semantic labels per frame.
        pred_instances: list of (H, W) predicted instance IDs per frame.
        gt_semantics: list of (H, W) GT semantic labels per frame.
        gt_instances: list of (H, W) GT instance IDs per frame.
        pixel_weights: optional list of (H, W) weight arrays (for wSTQ).
        num_classes: number of semantic classes.
        thing_list: list of thing class IDs.

    Returns:
        Dict with "STQ", "AQ", "SQ" and supporting info.
    """
    metric = STQuality(num_classes=num_classes, thing_list=thing_list)

    for i in range(len(pred_semantics)):
        w = pixel_weights[i] if pixel_weights is not None else None
        metric.update(
            pred_semantics[i], pred_instances[i],
            gt_semantics[i], gt_instances[i],
            pixel_weights=w,
        )

    return metric.result()
