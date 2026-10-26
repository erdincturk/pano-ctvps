"""
PQ: Panoptic Quality
=====================

Image-level evaluation metric for panoptic segmentation.

Reference:
    - Kirillov et al. 2019, "Panoptic Segmentation" (CVPR), Eq. 1
    - Mei et al. 2022, WOD PVPS (Eq. 4)

PQ = Σ (IoU(z_f, z_g)) / (|TP| + 0.5*|FP| + 0.5*|FN|)

Decomposed as: PQ = SQ * RQ
where:
    SQ = Σ IoU / |TP|          (average IoU of matched pairs)
    RQ = |TP| / (|TP| + 0.5*|FP| + 0.5*|FN|)  (recognition quality)

Usage:
    metric = PanopticQuality(num_classes=29, thing_list=[2,3,4,5,8,9,10,11])
    for image in dataset:
        metric.update(pred_semantic, pred_instance, gt_semantic, gt_instance)
    results = metric.result()
    # results = {"PQ": pq, "SQ": sq, "RQ": rq, ...}
"""

from typing import Dict, List, Optional

import numpy as np

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES, CLASS_NAMES


class PanopticQuality:
    """Panoptic Quality metric.

    Matches predicted segments to GT segments per class using IoU > 0.5.
    Accumulates TP/FP/FN counts across images, computes PQ at the end.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        thing_list: Optional[List[int]] = None,
        ignore_label: int = 0,
        iou_threshold: float = 0.5,
        offset: int = 256**3,
    ):
        """
        Args:
            num_classes: number of semantic classes.
            thing_list: class IDs with instances (thing classes).
            ignore_label: class ID to ignore.
            iou_threshold: IoU threshold for TP matching (default 0.5).
            offset: offset to create unique segment IDs.
        """
        self.num_classes = num_classes
        self.thing_list = thing_list if thing_list is not None else THING_CLASSES
        self.ignore_label = ignore_label
        self.iou_threshold = iou_threshold
        self.offset = offset

        self.is_thing = np.zeros(num_classes, dtype=bool)
        for c in self.thing_list:
            self.is_thing[c] = True

        self.reset()

    def reset(self):
        """Clear accumulated state."""
        # Per-class accumulators
        self._tp = np.zeros(self.num_classes, dtype=np.int64)
        self._fp = np.zeros(self.num_classes, dtype=np.int64)
        self._fn = np.zeros(self.num_classes, dtype=np.int64)
        self._iou_sum = np.zeros(self.num_classes, dtype=np.float64)

        self._num_images = 0

    def update(
        self,
        pred_semantic: np.ndarray,
        pred_instance: np.ndarray,
        gt_semantic: np.ndarray,
        gt_instance: np.ndarray,
    ):
        """Update metric with one image.

        Args:
            pred_semantic: (H, W) predicted class IDs.
            pred_instance: (H, W) predicted instance IDs.
            gt_semantic: (H, W) GT class IDs.
            gt_instance: (H, W) GT instance IDs.
        """
        assert pred_semantic.shape == gt_semantic.shape

        pred_sem = pred_semantic.ravel().astype(np.int32)
        pred_inst = pred_instance.ravel().astype(np.int32)
        gt_sem = gt_semantic.ravel().astype(np.int32)
        gt_inst = gt_instance.ravel().astype(np.int32)

        # Build panoptic IDs: class * offset + instance_id
        # For stuff classes, all pixels of same class share instance_id=0
        pred_panoptic = pred_sem.astype(np.int64) * self.offset + pred_inst
        gt_panoptic = gt_sem.astype(np.int64) * self.offset + gt_inst

        # Ignore pixels
        valid = gt_sem != self.ignore_label

        # Process each class independently
        for c in range(self.num_classes):
            if c == self.ignore_label:
                continue

            gt_c = (gt_sem == c) & valid
            pred_c = pred_sem == c

            if not self.is_thing[c]:
                # ---- Stuff class: one segment per class ----
                intersection = np.sum(gt_c & pred_c)
                union = np.sum(gt_c | pred_c)

                if union == 0:
                    continue

                iou = intersection / union

                if gt_c.sum() > 0 and pred_c.sum() > 0 and iou >= self.iou_threshold:
                    self._tp[c] += 1
                    self._iou_sum[c] += iou
                else:
                    if gt_c.sum() > 0:
                        self._fn[c] += 1
                    if pred_c.sum() > 0:
                        self._fp[c] += 1
            else:
                # ---- Thing class: match instance segments ----
                # Find unique GT segments for this class
                gt_segments = {}
                gt_ids = gt_panoptic[gt_c]
                for seg_id in np.unique(gt_ids):
                    if seg_id == c * self.offset:  # instance 0 = no instance
                        continue
                    gt_segments[int(seg_id)] = (gt_panoptic == seg_id)

                # Find unique pred segments for this class
                pred_segments = {}
                pred_ids = pred_panoptic[pred_c]
                for seg_id in np.unique(pred_ids):
                    if seg_id == c * self.offset:  # instance 0
                        continue
                    pred_segments[int(seg_id)] = (pred_panoptic == seg_id)

                # Match GT to pred using IoU > 0.5 (greedy, unique matching)
                matched_pred = set()
                matched_gt = set()

                # Compute all IoUs
                iou_pairs = []
                for gt_id, gt_mask in gt_segments.items():
                    for pred_id, pred_mask in pred_segments.items():
                        intersection = np.sum(gt_mask & pred_mask)
                        if intersection == 0:
                            continue
                        union = np.sum(gt_mask | pred_mask)
                        iou = intersection / union
                        if iou >= self.iou_threshold:
                            iou_pairs.append((iou, gt_id, pred_id))

                # Greedy matching: highest IoU first
                iou_pairs.sort(reverse=True)
                for iou, gt_id, pred_id in iou_pairs:
                    if gt_id in matched_gt or pred_id in matched_pred:
                        continue
                    matched_gt.add(gt_id)
                    matched_pred.add(pred_id)
                    self._tp[c] += 1
                    self._iou_sum[c] += iou

                # Unmatched GT = FN, unmatched pred = FP
                self._fn[c] += len(gt_segments) - len(matched_gt)
                self._fp[c] += len(pred_segments) - len(matched_pred)

        self._num_images += 1

    def result(self) -> Dict[str, float]:
        """Compute final PQ metrics.

        Returns:
            Dict with PQ, SQ, RQ (overall, thing, stuff) and per-class values.
        """
        per_class_pq = np.zeros(self.num_classes, dtype=np.float64)
        per_class_sq = np.zeros(self.num_classes, dtype=np.float64)
        per_class_rq = np.zeros(self.num_classes, dtype=np.float64)

        valid_mask = np.zeros(self.num_classes, dtype=bool)

        for c in range(self.num_classes):
            if c == self.ignore_label:
                continue

            tp, fp, fn = self._tp[c], self._fp[c], self._fn[c]

            if tp + fp + fn == 0:
                continue  # Class not present in any image

            valid_mask[c] = True
            denom = tp + 0.5 * fp + 0.5 * fn

            if tp > 0:
                per_class_sq[c] = self._iou_sum[c] / tp
                per_class_rq[c] = tp / denom
                per_class_pq[c] = per_class_sq[c] * per_class_rq[c]
            else:
                per_class_sq[c] = 0.0
                per_class_rq[c] = 0.0
                per_class_pq[c] = 0.0

        # Aggregate: all, thing, stuff
        thing_mask = valid_mask & self.is_thing
        stuff_mask = valid_mask & ~self.is_thing

        def safe_mean(arr, mask):
            return float(arr[mask].mean()) if mask.sum() > 0 else 0.0

        return {
            "PQ": safe_mean(per_class_pq, valid_mask),
            "SQ": safe_mean(per_class_sq, valid_mask),
            "RQ": safe_mean(per_class_rq, valid_mask),
            "PQ_thing": safe_mean(per_class_pq, thing_mask),
            "SQ_thing": safe_mean(per_class_sq, thing_mask),
            "RQ_thing": safe_mean(per_class_rq, thing_mask),
            "PQ_stuff": safe_mean(per_class_pq, stuff_mask),
            "SQ_stuff": safe_mean(per_class_sq, stuff_mask),
            "RQ_stuff": safe_mean(per_class_rq, stuff_mask),
            "num_images": self._num_images,
            "per_class_pq": per_class_pq.tolist(),
            "per_class_sq": per_class_sq.tolist(),
            "per_class_rq": per_class_rq.tolist(),
            "per_class_tp": self._tp.tolist(),
            "per_class_fp": self._fp.tolist(),
            "per_class_fn": self._fn.tolist(),
        }
