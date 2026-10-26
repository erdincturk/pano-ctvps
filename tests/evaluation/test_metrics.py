#!/usr/bin/env python3
"""
Tests for Evaluation Metrics: STQ/wSTQ, PQ, mIoU
=================================================

Verifies metric correctness on synthetic data with known expected results.

Usage:
    python -m pytest tests/evaluation/test_metrics.py
"""

import sys
import numpy as np


def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name: str, value: float, expected: float, tol: float = 1e-6):
    passed = abs(value - expected) < tol
    status = "✓" if passed else "✗"
    print(f"  {status} {name}: {value:.6f} (expected {expected:.6f})")
    if not passed:
        raise AssertionError(f"FAILED: {name}: got {value}, expected {expected}")


# ==================================================================
# mIoU Tests
# ==================================================================

def test_miou():
    header("mIoU Tests")
    from panoctvps.evaluation import MeanIoU

    # ---- Test 1: Perfect prediction ----
    print("\n  [1] Perfect prediction")
    m = MeanIoU(num_classes=5, ignore_label=0)
    gt = np.array([[1, 2, 3], [4, 1, 2]], dtype=np.int32)
    m.update(gt, gt)
    r = m.result()
    check("mIoU (perfect)", r["mIoU"], 1.0)
    check("pixel_accuracy (perfect)", r["pixel_accuracy"], 1.0)

    # ---- Test 2: Completely wrong ----
    print("\n  [2] Completely wrong")
    m = MeanIoU(num_classes=5, ignore_label=0)
    gt = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32)
    pred = np.array([[2, 2, 2], [2, 2, 2]], dtype=np.int32)
    m.update(pred, gt)
    r = m.result()
    check("mIoU (wrong)", r["mIoU"], 0.0)

    # ---- Test 3: 50% overlap on one class ----
    print("\n  [3] Partial overlap")
    m = MeanIoU(num_classes=3, ignore_label=0)
    #    gt:   [1, 1, 2, 2]
    #    pred: [1, 2, 2, 2]
    # class 1: TP=1, FP=0, FN=1 → IoU = 1/2
    # class 2: TP=2, FP=1, FN=0 → IoU = 2/3
    # mIoU = (0.5 + 0.6667) / 2 = 0.5833
    gt = np.array([1, 1, 2, 2], dtype=np.int32).reshape(1, 4)
    pred = np.array([1, 2, 2, 2], dtype=np.int32).reshape(1, 4)
    m.update(pred, gt)
    r = m.result()
    expected_miou = (1/2 + 2/3) / 2
    check("mIoU (partial)", r["mIoU"], expected_miou)

    # ---- Test 4: Ignore label ----
    print("\n  [4] Ignore label handling")
    m = MeanIoU(num_classes=3, ignore_label=0)
    gt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32).reshape(1, 6)
    pred = np.array([1, 2, 1, 1, 2, 2], dtype=np.int32).reshape(1, 6)
    m.update(pred, gt)
    r = m.result()
    # GT=0 pixels are EXCLUDED entirely (not counted as FP for pred)
    # Only evaluating on gt={1,1,2,2}, pred={1,1,2,2} → perfect match
    # Class 1: TP=2, FP=0, FN=0 → IoU = 1.0
    # Class 2: TP=2, FP=0, FN=0 → IoU = 1.0
    # mIoU = 1.0
    check("mIoU (ignore)", r["mIoU"], 1.0)

    # ---- Test 5: Multi-image accumulation ----
    print("\n  [5] Multi-image accumulation")
    m = MeanIoU(num_classes=3, ignore_label=0)
    gt1 = np.array([1, 1, 1, 1], dtype=np.int32).reshape(2, 2)
    pred1 = np.array([1, 1, 1, 1], dtype=np.int32).reshape(2, 2)
    gt2 = np.array([2, 2, 2, 2], dtype=np.int32).reshape(2, 2)
    pred2 = np.array([1, 1, 1, 1], dtype=np.int32).reshape(2, 2)
    m.update(pred1, gt1)
    m.update(pred2, gt2)
    r = m.result()
    # class 1: TP=4, FP=4, FN=0 → IoU = 4/8 = 0.5
    # class 2: TP=0, FP=0, FN=4 → IoU = 0/4 = 0.0
    # mIoU = (0.5 + 0.0) / 2 = 0.25
    check("mIoU (multi-image)", r["mIoU"], 0.25)

    print("\n  ✅ mIoU: all tests passed")


# ==================================================================
# PQ Tests
# ==================================================================

def test_pq():
    header("PQ Tests")
    from panoctvps.evaluation import PanopticQuality

    # ---- Test 1: Perfect prediction ----
    print("\n  [1] Perfect prediction")
    pq = PanopticQuality(num_classes=5, thing_list=[1, 2], ignore_label=0)
    gt_sem = np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 2, 2], [0, 0, 0, 0]], dtype=np.int32)
    pq.update(gt_sem, gt_inst, gt_sem, gt_inst)
    r = pq.result()
    check("PQ (perfect)", r["PQ"], 1.0)
    check("SQ (perfect)", r["SQ"], 1.0)
    check("RQ (perfect)", r["RQ"], 1.0)

    # ---- Test 2: Thing class - two instances, one matched ----
    print("\n  [2] Partial thing matching")
    pq = PanopticQuality(num_classes=3, thing_list=[1], ignore_label=0)
    # GT: class 1 instance 1 (4 pixels), class 1 instance 2 (4 pixels)
    gt_sem = np.ones((2, 4), dtype=np.int32)
    gt_inst = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    # Pred: matches instance 1 perfectly, misses instance 2 entirely
    pred_sem = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    pred_inst = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.int32)
    pq.update(pred_sem, pred_inst, gt_sem, gt_inst)
    r = pq.result()
    # class 1: TP=1 (inst 1, IoU=1.0), FN=1 (inst 2 missed), FP=0
    # PQ = 1.0 / (1 + 0 + 0.5) = 0.6667
    # class 2: stuff, only pred pixels → FP=1, no GT → skip? Actually class 2 has
    # pred pixels but no GT class 2 pixels... Let me reconsider.
    # Actually gt_sem is all 1s, pred_sem right half is 2.
    # class 1: GT has 8 pixels (inst 1: 4, inst 2: 4)
    #          Pred has 4 pixels (inst 1: 4)
    #          inst 1: IoU = 4/(4+0+0)=1.0 ✓ TP
    #          inst 2: no pred match → FN
    #          TP=1, FN=1, FP=0
    #          PQ_1 = 1.0 / (1 + 0 + 0.5) = 0.6667
    # class 2: stuff, GT has 0 pixels, pred has 4 pixels → FP=1
    #          PQ_2 = 0 / (0 + 0.5 + 0) = 0
    # PQ = (0.6667 + 0) / 2 = 0.3333
    check("PQ (partial thing)", r["PQ_thing"], 1.0 / 1.5)

    # ---- Test 3: Stuff class IoU ----
    print("\n  [3] Stuff class")
    pq = PanopticQuality(num_classes=3, thing_list=[], ignore_label=0)
    # All stuff: class 1 and class 2
    gt_sem = np.array([[1, 1, 2, 2]], dtype=np.int32)
    gt_inst = np.zeros((1, 4), dtype=np.int32)
    pred_sem = np.array([[1, 2, 2, 2]], dtype=np.int32)
    pred_inst = np.zeros((1, 4), dtype=np.int32)
    pq.update(pred_sem, pred_inst, gt_sem, gt_inst)
    r = pq.result()
    # class 1: intersection=1, union=2, IoU=0.5 → TP (just barely at threshold)
    # class 2: intersection=2, union=3, IoU=0.667 → TP
    # PQ = (0.5 + 0.667) / 2 = 0.5833
    expected_pq = (1/2 + 2/3) / 2
    check("PQ (stuff)", r["PQ"], expected_pq)

    # ---- Test 4: Completely wrong ----
    print("\n  [4] Completely wrong")
    pq = PanopticQuality(num_classes=3, thing_list=[1], ignore_label=0)
    gt_sem = np.ones((2, 2), dtype=np.int32)
    gt_inst = np.ones((2, 2), dtype=np.int32)
    pred_sem = np.full((2, 2), 2, dtype=np.int32)
    pred_inst = np.zeros((2, 2), dtype=np.int32)
    pq.update(pred_sem, pred_inst, gt_sem, gt_inst)
    r = pq.result()
    check("PQ (wrong)", r["PQ"], 0.0)

    print("\n  ✅ PQ: all tests passed")


# ==================================================================
# STQ Tests
# ==================================================================

def test_stq():
    header("STQ Tests")
    from panoctvps.evaluation import STQuality

    # Use a small class set for clarity:
    # Classes: 0=ignore, 1=car (thing), 2=road (stuff)
    thing_list = [1]
    NUM_CLS = 3

    # ---- Test 1: Perfect prediction ----
    print("\n  [1] Perfect prediction (single frame)")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    gt_sem = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.int32)
    stq.update(gt_sem, gt_inst, gt_sem, gt_inst)
    r = stq.result()
    check("SQ (perfect)", r["SQ"], 1.0)
    check("AQ (perfect)", r["AQ"], 1.0)
    check("STQ (perfect)", r["STQ"], 1.0)

    # ---- Test 2: Perfect prediction (multi-frame, consistent tracking) ----
    print("\n  [2] Perfect tracking across frames")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    for _ in range(5):
        stq.update(gt_sem, gt_inst, gt_sem, gt_inst)
    r = stq.result()
    check("SQ (5 frames)", r["SQ"], 1.0)
    check("AQ (5 frames)", r["AQ"], 1.0)
    check("STQ (5 frames)", r["STQ"], 1.0)

    # ---- Test 3: Perfect semantic, wrong instance IDs ----
    print("\n  [3] Correct semantic, wrong instances (ID switch every frame)")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    gt_sem = np.array([[1, 1, 2, 2]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 0, 0]], dtype=np.int32)

    for i in range(5):
        pred_inst = np.array([[i + 1, i + 1, 0, 0]], dtype=np.int32)  # new ID each frame
        stq.update(gt_sem, pred_inst, gt_sem, gt_inst)

    r = stq.result()
    check("SQ (wrong IDs)", r["SQ"], 1.0)
    # AQ: GT instance has 10 pixels total (2 per frame * 5 frames)
    # Each pred instance has 2 pixels, overlap = 2
    # For each pred: TPA=2, FNA=10-2=8, FPA=2-2=0 → IoU = 2/10 = 0.2
    # AQ = (1/10) * Σ TPA * IoU = (1/10) * 5 * (2 * 0.2) = (1/10) * 2.0 = 0.2
    check("AQ (ID switch)", r["AQ"], 0.2)

    # ---- Test 4: Completely wrong semantic ----
    print("\n  [4] Completely wrong semantic")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    gt_sem = np.array([[1, 1, 2, 2]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 0, 0]], dtype=np.int32)
    pred_sem = np.array([[2, 2, 1, 1]], dtype=np.int32)
    pred_inst = np.array([[0, 0, 1, 1]], dtype=np.int32)
    stq.update(pred_sem, pred_inst, gt_sem, gt_inst)
    r = stq.result()
    check("SQ (wrong semantic)", r["SQ"], 0.0)
    check("AQ (wrong semantic)", r["AQ"], 0.0)
    check("STQ (wrong semantic)", r["STQ"], 0.0)

    # ---- Test 5: No things in GT (stuff only) ----
    print("\n  [5] Stuff only (no instances)")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    gt_sem = np.array([[2, 2, 2, 2]], dtype=np.int32)
    gt_inst = np.zeros((1, 4), dtype=np.int32)
    stq.update(gt_sem, gt_inst, gt_sem, gt_inst)
    r = stq.result()
    check("SQ (stuff only)", r["SQ"], 1.0)
    check("AQ (stuff only, no instances)", r["AQ"], 0.0)

    # ---- Test 6: wSTQ with pixel weights ----
    print("\n  [6] wSTQ with pixel weights")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    gt_sem = np.array([[1, 1, 2, 2]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 0, 0]], dtype=np.int32)
    # First pixel covered by 2 cameras (weight=0.5), rest by 1 (weight=1.0)
    weights = np.array([[0.5, 1.0, 1.0, 1.0]], dtype=np.float64)
    stq.update(gt_sem, gt_inst, gt_sem, gt_inst, pixel_weights=weights)
    r = stq.result()
    check("SQ (weighted, perfect)", r["SQ"], 1.0)
    check("AQ (weighted, perfect)", r["AQ"], 1.0)

    # ---- Test 7b: wSTQ differs from STQ on imperfect prediction ----
    # 1x8 frame, classes: 0=ignore, 1=car(thing), 2=road(stuff)
    # Pixels 0-3: overlap region (weight 0.5)
    # Pixels 4-7: non-overlap region (weight 1.0)
    #
    #            px:  0    1    2    3    4    5    6    7
    #            wt:  0.5  0.5  0.5  0.5  1.0  1.0  1.0  1.0
    #        gt_sem:  1    1    2    2    1    1    2    2
    #       gt_inst:  1    1    0    0    1    1    0    0
    #      pred_sem:  2    2    2    2    1    1    2    2   (misses car in overlap)
    #     pred_inst:  0    0    0    0   10   10    0    0
    #
    # Hand-computed weighted (wSTQ):
    #   class 1 (car):  inter={4,5}=2.0   union={0,1,4,5}=3.0   IoU=2/3
    #   class 2 (road): inter={2,3,6,7}=3.0  union={0,1,2,3,6,7}=4.0  IoU=3/4
    #   wSQ = (2/3 + 3/4) / 2 = 17/24 ≈ 0.70833
    #   GT inst (px 0,1,4,5) weighted size = 3.0;  TPA(pred 10) = 2.0
    #   FNA = 3-2 = 1.0;  FPA = 2-2 = 0;  IoU_id = 2/(2+1+0) = 2/3
    #   per-instance score = TPA·IoU_id / gt_size = (2 · 2/3) / 3 = 4/9
    #   wAQ = 4/9
    #   wSTQ = sqrt((17/24)·(4/9)) ≈ 0.56111
    #
    # Hand-computed unweighted (STQ, weights=1):
    #   class 1: inter=2 union=4 IoU=0.5;  class 2: inter=4 union=6 IoU=2/3
    #   SQ = (0.5 + 2/3) / 2 = 7/12 ≈ 0.58333
    #   GT size=4; TPA=2; FNA=2; FPA=0; IoU_id=0.5; score = (2·0.5)/4 = 0.25
    #   AQ = 0.25
    #   STQ = sqrt(0.25 · 7/12) ≈ 0.38188
    #
    # The two STQ values must differ — otherwise weights are silently ignored.
    print("\n  [7b] wSTQ differs from STQ when weights are non-uniform")
    gt_sem = np.array([[1, 1, 2, 2, 1, 1, 2, 2]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.int32)
    pred_sem = np.array([[2, 2, 2, 2, 1, 1, 2, 2]], dtype=np.int32)
    pred_inst = np.array([[0, 0, 0, 0, 10, 10, 0, 0]], dtype=np.int32)
    weights = np.array([[0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)

    stq_w = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    stq_w.update(pred_sem, pred_inst, gt_sem, gt_inst, pixel_weights=weights)
    rw = stq_w.result()
    check("wSQ", rw["SQ"], 17 / 24)
    check("wAQ", rw["AQ"], 4 / 9)
    check("wSTQ", rw["STQ"], np.sqrt((17 / 24) * (4 / 9)))

    stq_u = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    stq_u.update(pred_sem, pred_inst, gt_sem, gt_inst)  # no weights -> all 1.0
    ru = stq_u.result()
    check("SQ (unweighted)", ru["SQ"], 7 / 12)
    check("AQ (unweighted)", ru["AQ"], 0.25)
    check("STQ (unweighted)", ru["STQ"], np.sqrt(0.25 * 7 / 12))

    assert abs(rw["STQ"] - ru["STQ"]) > 0.1, (
        f"wSTQ ({rw['STQ']:.4f}) and STQ ({ru['STQ']:.4f}) are too close — "
        "weights may be silently ignored"
    )

    # ---- Test 7: Two GT instances, one tracked correctly, one not ----
    print("\n  [7] Two instances, partial tracking")
    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list, ignore_label=0)
    # Frame 1: two car instances
    gt_sem = np.array([[1, 1, 1, 1]], dtype=np.int32)
    gt_inst = np.array([[1, 1, 2, 2]], dtype=np.int32)
    # Pred: matches instance 1 perfectly, assigns instance 2 as stuff (class 2)
    pred_sem = np.array([[1, 1, 2, 2]], dtype=np.int32)
    pred_inst = np.array([[1, 1, 0, 0]], dtype=np.int32)
    stq.update(pred_sem, pred_inst, gt_sem, gt_inst)
    r = stq.result()
    # SQ: class 1 IoU = 2/(2+0+2)=0.5, class 2 IoU = 0/(0+2+2)=0.0
    # SQ = (0.5 + 0) / 2 = 0.25
    check("SQ (partial)", r["SQ"], 0.25)
    # AQ: GT inst 1 (2px): pred inst 1 matches, TPA=2, FNA=0, FPA=0 → IoU=1.0
    #     score = (1/2) * 2 * 1.0 = 1.0
    #     GT inst 2 (2px): no thing pred overlaps → score = 0
    #     AQ = (1.0 + 0) / 2 = 0.5
    check("AQ (partial)", r["AQ"], 0.5)

    print("\n  ✅ STQ: all tests passed")


# ==================================================================
# Integration test: All metrics together
# ==================================================================

def test_integration():
    header("Integration Test: All Metrics")
    from panoctvps.evaluation import STQuality, PanopticQuality, MeanIoU

    # Simulate a 3-frame sequence with 3 classes:
    # 0=ignore, 1=car(thing), 2=road(stuff)
    thing_list = [1]
    NUM_CLS = 3

    stq = STQuality(num_classes=NUM_CLS, thing_list=thing_list)
    pq = PanopticQuality(num_classes=NUM_CLS, thing_list=thing_list)
    miou = MeanIoU(num_classes=NUM_CLS)

    # Create 3 frames of perfect predictions
    for _ in range(3):
        gt_sem = np.array([[1, 1, 2, 2], [2, 2, 2, 2]], dtype=np.int32)
        gt_inst = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int32)

        stq.update(gt_sem, gt_inst, gt_sem, gt_inst)
        pq.update(gt_sem, gt_inst, gt_sem, gt_inst)
        miou.update(gt_sem, gt_sem)

    stq_r = stq.result()
    pq_r = pq.result()
    miou_r = miou.result()

    check("STQ", stq_r["STQ"], 1.0)
    check("PQ", pq_r["PQ"], 1.0)
    check("mIoU", miou_r["mIoU"], 1.0)

    print(f"\n  STQ={stq_r['STQ']:.3f}  AQ={stq_r['AQ']:.3f}  SQ={stq_r['SQ']:.3f}")
    print(f"  PQ={pq_r['PQ']:.3f}   SQ={pq_r['SQ']:.3f}   RQ={pq_r['RQ']:.3f}")
    print(f"  mIoU={miou_r['mIoU']:.3f}")

    print("\n  ✅ Integration: all tests passed")


# ==================================================================
# Main
# ==================================================================

def main():
    print("=" * 60)
    print("  Evaluation Metrics Test Suite")
    print("=" * 60)

    tests = [
        ("mIoU", test_miou),
        ("PQ", test_pq),
        ("STQ/wSTQ", test_stq),
        ("Integration", test_integration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ {name} FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"  {passed}/{passed+failed} test groups passed")
    if failed == 0:
        print("  ✅ ALL METRIC TESTS PASSED")
    else:
        print(f"  ❌ {failed} TEST GROUP(S) FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
