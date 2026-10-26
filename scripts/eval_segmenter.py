#!/usr/bin/env python3
"""Evaluate a trained Mask2Former checkpoint on the WOD PVPS validation set.

Each image is evaluated at its native resolution, padded at the bottom and
right to a multiple of 32; predictions are cropped back to the original size
before scoring. Computes PQ (PQ_thing, PQ_stuff, SQ, RQ) and mIoU with a
per-camera and per-class breakdown, and saves them to JSON.

Usage:
    python scripts/eval_segmenter.py \
        --config configs/training/segmenter_mapillary_refined.yaml \
        --checkpoint checkpoints/segmenter.pth
"""

import argparse
import json
import os
import time

import torch

from panoctvps.data.training_dataset import WODMask2FormerDataset
from panoctvps.segmenter.mask2former import build_model, load_checkpoint, predict_panoptic
from panoctvps.evaluation import PanopticQuality, MeanIoU
from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES, CLASS_NAMES, CAMERA_NAMES
from panoctvps.training.engine import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mask2Former on WOD PVPS")
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.pth)")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root")
    parser.add_argument("--split", default="validation", choices=["training", "validation"])
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Number of samples (0 = all)")
    parser.add_argument("--output", default=None, help="JSON output path (default: auto)")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Override camera IDs to evaluate (e.g., --cameras 1 2 3 4 5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("dataset", {})

    dataset_root = args.dataset_root or data_cfg.get(
        "root", "../data/waymo_open_dataset_v_2_0_1"
    )
    backbone = cfg.get("model", {}).get("backbone", "swin-large-mapillary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Model ----
    print(f"Loading model (backbone={backbone}) ...")
    model = build_model(backbone=backbone, num_classes=NUM_CLASSES, pretrained=False)
    iteration = load_checkpoint(model, args.checkpoint, device=str(device))
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {args.checkpoint} (iteration {iteration})")

    # ---- Dataset ----
    cameras = args.cameras or data_cfg.get("cameras", [1])
    print(f"Loading {args.split} dataset (full resolution, cameras={cameras}) ...")
    val_ds = WODMask2FormerDataset(
        dataset_root=dataset_root,
        split=args.split,
        cameras=cameras,
        full_resolution=True,
    )
    n_total = len(val_ds)
    n_eval = n_total if args.n_samples <= 0 else min(args.n_samples, n_total)
    print(f"  {n_eval}/{n_total} samples")

    # ---- Metrics: aggregate + per-camera ----
    pq_metric   = PanopticQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES)
    miou_metric = MeanIoU(num_classes=NUM_CLASSES)

    per_cam_pq   = {c: PanopticQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES) for c in cameras}
    per_cam_miou = {c: MeanIoU(num_classes=NUM_CLASSES) for c in cameras}
    per_cam_count = {c: 0 for c in cameras}

    # ---- Evaluate ----
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_eval):
            sample = val_ds[i]
            cam_id = sample["meta"]["camera_id"]
            pv = sample["pixel_values"].unsqueeze(0).to(device)

            preds = predict_panoptic(model, pv)
            pred_sem  = preds[0]["semantic"]
            pred_inst = preds[0]["instance"]

            gt_semantic = sample["semantic_label"]
            gt_instance = sample["instance_label"]

            if "original_size" in sample:
                oh, ow = sample["original_size"]
                pred_sem  = pred_sem[:oh, :ow]
                pred_inst = pred_inst[:oh, :ow]
                gt_semantic = gt_semantic[:oh, :ow]
                gt_instance = gt_instance[:oh, :ow]

            # Aggregate
            pq_metric.update(pred_sem, pred_inst, gt_semantic, gt_instance)
            miou_metric.update(pred_sem, gt_semantic)

            # Per-camera
            if cam_id in per_cam_pq:
                per_cam_pq[cam_id].update(pred_sem, pred_inst, gt_semantic, gt_instance)
                per_cam_miou[cam_id].update(pred_sem, gt_semantic)
                per_cam_count[cam_id] += 1

            if (i + 1) % 50 == 0 or i == n_eval - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"\r  [{i+1}/{n_eval}]  {rate:.1f} img/s", end="", flush=True)

    print()
    elapsed = time.time() - t0

    pq_r   = pq_metric.result()
    miou_r = miou_metric.result()

    # ---- Print aggregate results ----
    print(f"\n{'='*60}")
    print(f"  Aggregate Results  ({n_eval} images, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  PQ:         {pq_r['PQ']:.4f}")
    print(f"  PQ_thing:   {pq_r['PQ_thing']:.4f}")
    print(f"  PQ_stuff:   {pq_r['PQ_stuff']:.4f}")
    print(f"  SQ:         {pq_r['SQ']:.4f}")
    print(f"  RQ:         {pq_r['RQ']:.4f}")
    print(f"  mIoU:       {miou_r['mIoU']:.4f}")
    print(f"  pixel_acc:  {miou_r['pixel_accuracy']:.4f}")

    # ---- Print per-camera results ----
    print(f"\n{'='*60}")
    print(f"  Per-Camera Results")
    print(f"{'='*60}")
    print(f"  {'Camera':<15s} {'PQ':>6s} {'PQ_th':>6s} {'PQ_st':>6s} {'SQ':>6s} {'RQ':>6s} {'mIoU':>6s} {'N':>6s}")
    print(f"  {'-'*63}")

    per_cam_results = {}
    for cam_id in sorted(cameras):
        cam_name = CAMERA_NAMES.get(cam_id, str(cam_id))
        cam_pq = per_cam_pq[cam_id].result()
        cam_miou = per_cam_miou[cam_id].result()
        n_cam = per_cam_count[cam_id]
        print(f"  {cam_name:<15s} {cam_pq['PQ']:6.4f} {cam_pq['PQ_thing']:6.4f} "
              f"{cam_pq['PQ_stuff']:6.4f} {cam_pq['SQ']:6.4f} {cam_pq['RQ']:6.4f} "
              f"{cam_miou['mIoU']:6.4f} {n_cam:6d}")
        per_cam_results[cam_name] = {
            "camera_id": cam_id,
            "n_samples": n_cam,
            "PQ": cam_pq["PQ"], "PQ_thing": cam_pq["PQ_thing"],
            "PQ_stuff": cam_pq["PQ_stuff"],
            "SQ": cam_pq["SQ"], "RQ": cam_pq["RQ"],
            "mIoU": cam_miou["mIoU"],
            "pixel_accuracy": cam_miou["pixel_accuracy"],
            "per_class_pq": cam_pq["per_class_pq"],
            "per_class_sq": cam_pq["per_class_sq"],
            "per_class_rq": cam_pq["per_class_rq"],
            "per_class_tp": cam_pq["per_class_tp"],
            "per_class_fp": cam_pq["per_class_fp"],
            "per_class_fn": cam_pq["per_class_fn"],
            "per_class_iou": cam_miou["per_class_iou"],
        }

    # ---- Print per-class PQ ----
    print(f"\n  Per-class PQ:")
    print(f"  {'Class':<25s} {'PQ':>6s} {'SQ':>6s} {'RQ':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print(f"  {'-'*73}")
    for c in range(NUM_CLASSES):
        if c == 0:
            continue
        tp = pq_r["per_class_tp"][c]
        fp = pq_r["per_class_fp"][c]
        fn = pq_r["per_class_fn"][c]
        if tp + fp + fn == 0:
            continue
        marker = " *" if c in THING_CLASSES else ""
        print(f"  {CLASS_NAMES[c]:<25s} {pq_r['per_class_pq'][c]:6.4f} "
              f"{pq_r['per_class_sq'][c]:6.4f} {pq_r['per_class_rq'][c]:6.4f} "
              f"{tp:5d} {fp:5d} {fn:5d}{marker}")

    # ---- Save results ----
    output_path = args.output
    if output_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(ckpt_dir, "eval_results.json")

    results = {
        "checkpoint": args.checkpoint,
        "iteration": iteration,
        "split": args.split,
        "n_samples": n_eval,
        "cameras": cameras,
        "eval_mode": "full-resolution",
        "backbone": backbone,
        "aggregate": {
            "PQ": pq_r["PQ"], "SQ": pq_r["SQ"], "RQ": pq_r["RQ"],
            "PQ_thing": pq_r["PQ_thing"], "PQ_stuff": pq_r["PQ_stuff"],
            "mIoU": miou_r["mIoU"],
            "pixel_accuracy": miou_r["pixel_accuracy"],
        },
        "per_camera": per_cam_results,
        "per_class_pq": pq_r["per_class_pq"],
        "per_class_sq": pq_r["per_class_sq"],
        "per_class_rq": pq_r["per_class_rq"],
        "per_class_tp": pq_r["per_class_tp"],
        "per_class_fp": pq_r["per_class_fp"],
        "per_class_fn": pq_r["per_class_fn"],
        "per_class_iou": miou_r["per_class_iou"],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
