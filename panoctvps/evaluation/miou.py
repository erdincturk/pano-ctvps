"""
mIoU: Mean Intersection over Union
====================================

Standard semantic segmentation metric.

Usage:
    metric = MeanIoU(num_classes=29)
    for image in dataset:
        metric.update(pred_semantic, gt_semantic)
    results = metric.result()
    # results = {"mIoU": miou, "per_class_iou": [...], ...}
"""

from typing import Dict, Optional

import numpy as np

from panoctvps.data.constants import NUM_CLASSES, CLASS_NAMES


class MeanIoU:
    """Confusion-matrix based mean IoU metric.

    Efficient implementation using numpy bincount for fast accumulation.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_label: int = 0,
    ):
        """
        Args:
            num_classes: number of semantic classes.
            ignore_label: class ID to ignore in evaluation.
        """
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        """Clear accumulated state."""
        self._confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self._num_images = 0

    def update(
        self,
        pred_semantic: np.ndarray,
        gt_semantic: np.ndarray,
    ):
        """Update confusion matrix with one image.

        Args:
            pred_semantic: (H, W) predicted class IDs.
            gt_semantic: (H, W) GT class IDs.
        """
        assert pred_semantic.shape == gt_semantic.shape

        pred = pred_semantic.ravel().astype(np.int64)
        gt = gt_semantic.ravel().astype(np.int64)

        # Filter out ignore label
        valid = gt != self.ignore_label
        pred = pred[valid]
        gt = gt[valid]

        # Clamp to valid range
        pred = np.clip(pred, 0, self.num_classes - 1)
        gt = np.clip(gt, 0, self.num_classes - 1)

        # Fast confusion matrix via bincount
        indices = gt * self.num_classes + pred
        cm = np.bincount(indices, minlength=self.num_classes**2)
        cm = cm.reshape(self.num_classes, self.num_classes)

        self._confusion_matrix += cm
        self._num_images += 1

    def result(self) -> Dict[str, float]:
        """Compute mean IoU from accumulated confusion matrix.

        Returns:
            Dict with mIoU, per-class IoU, and accuracy metrics.
        """
        cm = self._confusion_matrix

        # Per-class IoU = TP / (TP + FP + FN)
        # TP_c = cm[c, c]
        # FP_c = sum(cm[:, c]) - cm[c, c]  (predicted as c, but GT is other)
        # FN_c = sum(cm[c, :]) - cm[c, c]  (GT is c, but predicted as other)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        denom = tp + fp + fn
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)

        valid = denom > 0
        per_class_iou[valid] = tp[valid].astype(np.float64) / denom[valid]

        # Exclude ignore label from mean
        eval_mask = valid.copy()
        eval_mask[self.ignore_label] = False

        if eval_mask.sum() > 0:
            miou = per_class_iou[eval_mask].mean()
        else:
            miou = 0.0

        # Overall pixel accuracy
        total_pixels = cm.sum()
        correct_pixels = tp.sum()
        pixel_accuracy = correct_pixels / max(total_pixels, 1)

        # Per-class accuracy
        per_class_acc = np.zeros(self.num_classes, dtype=np.float64)
        class_totals = cm.sum(axis=1)
        valid_acc = class_totals > 0
        per_class_acc[valid_acc] = tp[valid_acc].astype(np.float64) / class_totals[valid_acc]

        mean_acc_mask = valid_acc.copy()
        mean_acc_mask[self.ignore_label] = False
        mean_acc = per_class_acc[mean_acc_mask].mean() if mean_acc_mask.sum() > 0 else 0.0

        # Build per-class detail
        per_class_detail = {}
        for c in range(self.num_classes):
            if c == self.ignore_label or not valid[c]:
                continue
            per_class_detail[CLASS_NAMES.get(c, f"class_{c}")] = {
                "iou": float(per_class_iou[c]),
                "accuracy": float(per_class_acc[c]),
                "tp": int(tp[c]),
                "fp": int(fp[c]),
                "fn": int(fn[c]),
            }

        return {
            "mIoU": float(miou),
            "pixel_accuracy": float(pixel_accuracy),
            "mean_accuracy": float(mean_acc),
            "num_images": self._num_images,
            "num_valid_classes": int(eval_mask.sum()),
            "per_class_iou": per_class_iou.tolist(),
            "per_class_detail": per_class_detail,
        }
