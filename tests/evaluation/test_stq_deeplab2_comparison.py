"""Verify STQuality matches DeepLab2's reference implementation.

Test cases extracted from:
  deeplab2/evaluation/segmentation_and_tracking_quality_test.py

These tests ensure our numpy-only STQ implementation produces
identical results to the TensorFlow-based DeepLab2 reference.
"""

import numpy as np
import pytest

from panoctvps.evaluation.stq import STQuality, merge_sequence_results


# DeepLab2 test parameters
N_CLASSES = 2
THINGS_LIST = [0]  # class 0 is "thing"
IGNORE_LABEL = 255
MAX_INSTANCES = 1000
OFFSET = 256 * 256


def _make_panoptic(semantic: np.ndarray, instance: np.ndarray, divisor: int = MAX_INSTANCES) -> tuple:
    """Convert DeepLab2-style panoptic label to our (semantic, instance) format.

    DeepLab2 test cases use panoptic = semantic * divisor + instance.
    Our STQuality takes separate semantic and instance arrays.
    """
    return semantic, instance


def _run_stq(gt_frames, pred_frames, n_classes=N_CLASSES, things_list=THINGS_LIST,
             weights_list=None):
    """Run STQ on a sequence of frames, return (STQ, AQ, SQ) tuple."""
    metric = STQuality(
        num_classes=n_classes,
        thing_list=things_list,
        ignore_label=IGNORE_LABEL,
        offset=OFFSET,
    )

    for i, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        gt_sem, gt_inst = gt
        pred_sem, pred_inst = pred
        w = weights_list[i] if weights_list else None
        metric.update(pred_sem, pred_inst, gt_sem, gt_inst, pixel_weights=w)

    result = metric.result()
    return result["STQ"], result["AQ"], result["SQ"]


class TestSTQMatchesDeepLab2:
    """Test cases from deeplab2/evaluation/segmentation_and_tracking_quality_test.py"""

    def test_perfect_prediction(self):
        """Example 0: perfect prediction -> STQ=1.0"""
        gt = (np.array([[1, 1, 1, 1, 1]], dtype=np.int32),
              np.array([[1, 1, 1, 1, 1]], dtype=np.int32))  # class 0 doesn't exist, class 1 stuff
        pred = (np.array([[1, 1, 1, 1, 1]], dtype=np.int32),
                np.array([[1, 1, 1, 1, 1]], dtype=np.int32))

        # But wait - in DeepLab2 tests, ground_truth=[[1,1,1,1,1]] means
        # panoptic_label = 1 for each pixel. semantic = 1//1000 = 0, instance = 1%1000 = 1.
        # Class 0 is a "thing" class. So this is one thing instance (class 0, instance 1).
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(_make_panoptic(gt_sem, gt_inst))],
            [(_make_panoptic(pred_sem, pred_inst))],
        )

        assert stq == pytest.approx(1.0, abs=1e-6)
        assert aq == pytest.approx(1.0, abs=1e-6)
        assert sq == pytest.approx(1.0, abs=1e-6)

    def test_split_into_two(self):
        """Example 1: one GT instance split into two predicted instances.

        GT: [1, 1, 1, 1, 1] -> class 0, instance 1 (5 pixels)
        Pred: [1, 1, 2, 2, 2] -> class 0, inst 1 (2px) + inst 2 (3px)

        AQ for GT instance 1:
          - pred inst 1: tpa=2, fna=3, fpa=0, iou=2/5=0.4, contribution=2*0.4=0.8
          - pred inst 2: tpa=3, fna=2, fpa=0, iou=3/5=0.6, contribution=3*0.6=1.8
          - total = (0.8 + 1.8) / 5 = 0.52
        AQ = 0.52, IoU = 1.0, STQ = sqrt(0.52) ≈ 0.72111
        """
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 1, 2, 2, 2]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem, gt_inst)],
            [(pred_sem, pred_inst)],
        )

        assert aq == pytest.approx(0.52, abs=1e-6)
        assert sq == pytest.approx(1.0, abs=1e-6)
        assert stq == pytest.approx(0.72111026, abs=1e-5)

    def test_split_1_4(self):
        """Example 2: split 1+4 pixels.

        AQ: pred1 tpa=1, iou=1/5=0.2, c=0.2; pred2 tpa=4, iou=4/5=0.8, c=3.2
        AQ = (0.2+3.2)/5 = 0.68
        """
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 2, 2, 2, 2]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem, gt_inst)],
            [(pred_sem, pred_inst)],
        )

        assert aq == pytest.approx(0.68, abs=1e-6)
        assert stq == pytest.approx(0.82462113, abs=1e-5)

    def test_split_into_five(self):
        """Example 3: each pixel is its own predicted instance.

        5 predicted instances, each tpa=1, iou=1/5=0.2, contribution=0.2
        AQ = (5*0.2)/5 = 0.2
        """
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem, gt_inst)],
            [(pred_sem, pred_inst)],
        )

        assert aq == pytest.approx(0.2, abs=1e-6)
        assert stq == pytest.approx(0.447213596, abs=1e-5)

    def test_split_alternating(self):
        """Example 4: alternating split [1,2,1,2,2].

        pred1: 2px, tpa=2, fna=3, fpa=0, iou=2/5=0.4, c=0.8
        pred2: 3px, tpa=3, fna=2, fpa=0, iou=3/5=0.6, c=1.8
        AQ = (0.8+1.8)/5 = 0.52
        """
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 2, 1, 2, 2]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem, gt_inst)],
            [(pred_sem, pred_inst)],
        )

        assert aq == pytest.approx(0.52, abs=1e-6)
        assert stq == pytest.approx(0.72111026, abs=1e-5)

    def test_with_stuff_class_no_instance(self):
        """Example 6: stuff pixel (class 0 no instance) + thing instance.

        GT: [0, 1, 1, 1, 1, 1] -> pixel 0 is class 0 with inst 0 (stuff-like)
        Pred: [1, 1, 1, 1, 1, 1] -> all class 0 instance 1

        The first GT pixel has inst=0 so it's crowd (excluded from AQ).
        AQ considers only the 5 pixels with inst=1. Perfect match -> AQ=1.0
        """
        gt_sem = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[0, 1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem, gt_inst)],
            [(pred_sem, pred_inst)],
        )

        assert aq == pytest.approx(1.0, abs=1e-6)
        assert sq == pytest.approx(1.0, abs=1e-6)
        assert stq == pytest.approx(1.0, abs=1e-6)

    def test_stuff_with_split(self):
        """Example 7: stuff pixel + split prediction.

        GT: [0, 1, 1, 1, 1, 1]
        Pred: [2, 2, 2, 2, 1, 1]

        GT inst 1 (5 pixels): pred inst 2 has tpa=3, pred inst 1 has tpa=2
        AQ = same as split 2+3 = 0.52
        """
        gt_sem = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[0, 1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[2, 2, 2, 2, 1, 1]], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem, gt_inst)],
            [(pred_sem, pred_inst)],
        )

        assert aq == pytest.approx(0.52, abs=1e-6)
        assert stq == pytest.approx(0.72111026, abs=1e-5)

    def test_complex_example(self):
        """Complex 3-frame example with thing class 2 and stuff classes 0, 1.

        Expected: STQ=0.66841773352, AQ=0.55366581415, IoU=0.8069529580
        """
        # Frame 1
        gt_sem_1 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 1, 1, 1],
            [0, 2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 1],
            [2, 2, 2, 2, 2, 2, 2, 1],
            [2, 2, 2, 2, 2, 2, 1, 1],
        ], dtype=np.int32)
        gt_inst_1 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 0],
            [2, 2, 2, 2, 2, 2, 2, 0],
            [2, 2, 2, 2, 2, 2, 0, 0],
        ], dtype=np.int32)

        pred_sem_1 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 2, 2, 1, 1, 1],
            [0, 2, 2, 2, 2, 2, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 1],
            [2, 2, 2, 2, 2, 2, 2, 1],
            [2, 2, 2, 2, 2, 2, 2, 1],
        ], dtype=np.int32)
        pred_inst_1 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 0, 0, 0],
            [0, 2, 2, 2, 2, 1, 0, 0],
            [2, 2, 2, 2, 2, 1, 1, 0],
            [2, 2, 2, 2, 1, 1, 1, 0],
            [2, 2, 2, 2, 1, 1, 1, 0],
        ], dtype=np.int32)

        # Frame 2
        gt_sem_2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 1, 1, 0, 0],
            [2, 2, 2, 1, 1, 1, 1, 0],
            [2, 2, 2, 2, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1, 1],
        ], dtype=np.int32)
        gt_inst_2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 0, 0, 0],
            [2, 2, 2, 2, 2, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0],
        ], dtype=np.int32)

        pred_sem_2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 2, 2, 2, 1, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 1, 1],
            [2, 2, 2, 2, 2, 1, 1, 1],
        ], dtype=np.int32)
        pred_inst_2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 0, 0, 0],
            [2, 2, 2, 2, 1, 1, 0, 0],
            [2, 2, 2, 2, 1, 0, 0, 0],
        ], dtype=np.int32)

        # Frame 3
        gt_sem_3 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 1, 1, 1, 0, 0, 0],
            [2, 2, 1, 1, 1, 1, 0, 0],
            [2, 2, 2, 1, 1, 1, 1, 0],
            [2, 2, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 1],
        ], dtype=np.int32)
        gt_inst_3 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
        ], dtype=np.int32)

        pred_sem_3 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [2, 2, 2, 1, 1, 1, 0, 0],
            [2, 2, 2, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1, 1],
            [2, 2, 2, 2, 1, 1, 1, 1],
        ], dtype=np.int32)
        pred_inst_3 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0],
        ], dtype=np.int32)

        stq, aq, sq = _run_stq(
            [(gt_sem_1, gt_inst_1), (gt_sem_2, gt_inst_2), (gt_sem_3, gt_inst_3)],
            [(pred_sem_1, pred_inst_1), (pred_sem_2, pred_inst_2), (pred_sem_3, pred_inst_3)],
            n_classes=3,
            things_list=[2],
        )

        assert sq == pytest.approx(0.8069529580, abs=1e-4)
        assert aq == pytest.approx(0.55366581415, abs=1e-4)
        assert stq == pytest.approx(0.66841773352, abs=1e-4)


class TestPixelWeightedSTQ:
    """wSTQ correctness — verifies pixel_weights wiring against DeepLab2-derived
    ground-truth and a hand-computed non-uniform-weight case.

    The DeepLab2 reference cases above don't exercise the pixel_weights path,
    so a bug in the weighted accumulator could otherwise pass the suite but
    break our reported wSTQ.
    """

    def test_weights_ones_match_unweighted(self):
        """Passing pixel_weights=ones reproduces the unweighted DeepLab2 result.

        This catches any silent change in the weighted path that would shift
        the unweighted-equivalent case (e.g. accumulator double-counting).
        """
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 1, 2, 2, 2]], dtype=np.int32)
        ones = np.ones_like(gt_sem, dtype=np.float64)

        # With weights=ones
        stq_w, aq_w, sq_w = _run_stq(
            [(gt_sem, gt_inst)], [(pred_sem, pred_inst)],
            weights_list=[ones],
        )
        # Without weights
        stq_u, aq_u, sq_u = _run_stq(
            [(gt_sem, gt_inst)], [(pred_sem, pred_inst)],
        )

        assert aq_w == pytest.approx(aq_u, abs=1e-9)
        assert sq_w == pytest.approx(sq_u, abs=1e-9)
        assert stq_w == pytest.approx(stq_u, abs=1e-9)
        # Sanity: matches the unweighted DeepLab2 reference (test_split_into_two)
        assert aq_w == pytest.approx(0.52, abs=1e-6)

    def test_weights_nonuniform_handcomputed(self):
        """Non-uniform pixel_weights produce a hand-computed wSTQ.

        Setup (1×8 frame, classes: 0=ignore, 1=car/thing, 2=road/stuff):
            px:        0    1    2    3    4    5    6    7
            wt:        0.5  0.5  0.5  0.5  1.0  1.0  1.0  1.0   (overlap region px 0-3)
            gt_sem:    1    1    2    2    1    1    2    2
            gt_inst:   1    1    0    0    1    1    0    0
            pred_sem:  2    2    2    2    1    1    2    2     (misses car in overlap)
            pred_inst: 0    0    0    0    10   10   0    0

        Weighted hand-comp:
            class 1 (car):  inter=2.0  union=3.0   IoU=2/3
            class 2 (road): inter=3.0  union=4.0   IoU=3/4
            wSQ = (2/3 + 3/4) / 2 = 17/24
            GT inst 1 weighted size = 3.0; TPA(pred 10) = 2.0
            FNA = 1.0; FPA = 0; IoU_id = 2/(2+1) = 2/3
            score = (2 · 2/3) / 3 = 4/9
            wAQ = 4/9
            wSTQ = sqrt((17/24) · (4/9)) ≈ 0.5611
        """
        gt_sem = np.array([[1, 1, 2, 2, 1, 1, 2, 2]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.int32)
        pred_sem = np.array([[2, 2, 2, 2, 1, 1, 2, 2]], dtype=np.int32)
        pred_inst = np.array([[0, 0, 0, 0, 10, 10, 0, 0]], dtype=np.int32)
        weights = np.array([[0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)

        # Use 3 classes (0=ignore, 1=thing, 2=stuff) — different from the
        # default test config but consistent with the reviewer's setup.
        metric = STQuality(num_classes=3, thing_list=[1], ignore_label=0)
        metric.update(pred_sem, pred_inst, gt_sem, gt_inst, pixel_weights=weights)
        r = metric.result()

        assert r["SQ"] == pytest.approx(17 / 24, abs=1e-6)
        assert r["AQ"] == pytest.approx(4 / 9, abs=1e-6)
        assert r["STQ"] == pytest.approx(np.sqrt((17 / 24) * (4 / 9)), abs=1e-6)


class TestMergeSequenceResults:
    """merge_sequence_results aggregation — verifies multi-segment summing.

    Aggregate wSTQ in our pipeline is the result of
    `merge_sequence_results(per_segment_metrics)`, but the DeepLab2 comparison
    test only exercises single-stream STQ. A bug in the merge path (e.g. wrong
    AQ-numerator weighting, summing per-class SQ wrongly) would pass the
    single-stream suite but break headline wSTQ.
    """

    def test_merge_two_sequences_perfect(self):
        """Two perfect-prediction sequences merge to perfect aggregate."""
        # Sequence A: 5 pixels, perfect
        gt_sem_a = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst_a = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        # Sequence B: 4 pixels, perfect, but uses the SAME numeric instance ID
        # (different physical instance — testing that merge treats them
        # independently, not by ID)
        gt_sem_b = np.array([[0, 0, 0, 0]], dtype=np.int32)
        gt_inst_b = np.array([[1, 1, 1, 1]], dtype=np.int32)

        m_a = STQuality(num_classes=N_CLASSES, thing_list=THINGS_LIST,
                        ignore_label=IGNORE_LABEL, offset=OFFSET)
        m_a.update(gt_sem_a, gt_inst_a, gt_sem_a, gt_inst_a)

        m_b = STQuality(num_classes=N_CLASSES, thing_list=THINGS_LIST,
                        ignore_label=IGNORE_LABEL, offset=OFFSET)
        m_b.update(gt_sem_b, gt_inst_b, gt_sem_b, gt_inst_b)

        merged = merge_sequence_results([m_a, m_b])
        assert merged["AQ"] == pytest.approx(1.0, abs=1e-9)
        assert merged["SQ"] == pytest.approx(1.0, abs=1e-9)
        assert merged["STQ"] == pytest.approx(1.0, abs=1e-9)
        assert merged["num_gt_instances"] == 2  # NOT 1 — instance IDs not shared

    def test_merge_two_sequences_imperfect(self):
        """Per-sequence AQ averages by GT instance count.

        Seq A (5 px, split 2+3 → AQ=0.52, n_gt=1).
        Seq B (5 px, perfect → AQ=1.0, n_gt=1).
        Merged AQ = (0.52·1 + 1.0·1) / 2 = 0.76.
        SQ stays 1.0 (semantic perfect in both); STQ = sqrt(0.76·1.0) ≈ 0.8717.
        """
        # Sequence A: split prediction (the test_split_into_two case)
        gt_sem_a = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst_a = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem_a = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst_a = np.array([[1, 1, 2, 2, 2]], dtype=np.int32)

        # Sequence B: perfect single instance
        gt_sem_b = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst_b = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)

        m_a = STQuality(num_classes=N_CLASSES, thing_list=THINGS_LIST,
                        ignore_label=IGNORE_LABEL, offset=OFFSET)
        m_a.update(pred_sem_a, pred_inst_a, gt_sem_a, gt_inst_a)
        assert m_a.result()["AQ"] == pytest.approx(0.52, abs=1e-6)

        m_b = STQuality(num_classes=N_CLASSES, thing_list=THINGS_LIST,
                        ignore_label=IGNORE_LABEL, offset=OFFSET)
        m_b.update(gt_sem_b, gt_inst_b, gt_sem_b, gt_inst_b)
        assert m_b.result()["AQ"] == pytest.approx(1.0, abs=1e-6)

        merged = merge_sequence_results([m_a, m_b])
        # AQ = (0.52*1 + 1.0*1) / 2 = 0.76
        assert merged["AQ"] == pytest.approx(0.76, abs=1e-6)
        assert merged["SQ"] == pytest.approx(1.0, abs=1e-6)
        assert merged["STQ"] == pytest.approx(np.sqrt(0.76), abs=1e-6)
        assert merged["num_gt_instances"] == 2

    def test_merge_empty(self):
        """Empty list returns zero metrics without crashing."""
        merged = merge_sequence_results([])
        assert merged["AQ"] == 0.0
        assert merged["SQ"] == 0.0
        assert merged["STQ"] == 0.0
        assert merged["num_gt_instances"] == 0

    def test_merge_single_sequence_matches_direct(self):
        """Merging a single sequence returns its direct result."""
        gt_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 1, 1, 1]], dtype=np.int32)
        pred_sem = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        pred_inst = np.array([[1, 1, 2, 2, 2]], dtype=np.int32)

        m = STQuality(num_classes=N_CLASSES, thing_list=THINGS_LIST,
                      ignore_label=IGNORE_LABEL, offset=OFFSET)
        m.update(pred_sem, pred_inst, gt_sem, gt_inst)
        direct = m.result()

        merged = merge_sequence_results([m])
        assert merged["AQ"] == pytest.approx(direct["AQ"], abs=1e-9)
        assert merged["SQ"] == pytest.approx(direct["SQ"], abs=1e-9)
        assert merged["STQ"] == pytest.approx(direct["STQ"], abs=1e-9)
