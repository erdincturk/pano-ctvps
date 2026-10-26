#!/usr/bin/env python3
"""Write the per-scene prediction cache with ground-truth ids, and report the
live wSTQ of the per-camera tracker.

For every frame and camera, runs the segmenter with the embedding head
(native resolution, bottom/right padding to a multiple of 32), tracks the
thing detections of each camera with the bi-softmax memory-bank tracker, and,
with --cache-dir, writes one .npz per scene (layout in panoctvps/data/cache_io.py).
Each cached detection carries the ground-truth global id it overlaps most
(majority vote over its mask) and the IoU with that ground-truth mask
(gt_global_id, gt_iou), and the sequence index of its annotation window.
This script writes the validation and training caches; the test split has no
ground truth and is cached by scripts/build_cache.py.

`records['tracker_id']` holds the ids of the tracker run here (no motion
gate). The tracker settings and the motion gate used downstream are applied
afterwards from the cache (scripts/eval_tracker_cached.py,
scripts/build_cache_trackids.py).

Usage:
    python scripts/build_cache_gt.py \
        --checkpoint checkpoints/tracker.pth \
        --split validation --cache-dir outputs/tracker_ctvis_val_cache

    # Subset for quick testing:
    python scripts/build_cache_gt.py \
        --checkpoint checkpoints/tracker.pth --n-sequences 2 --cameras 1
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
import torch

from panoctvps.data.cache_io import (
    encode_rle, encode_sem_rle, load_geometry, write_segment_npz,
)
from panoctvps.data.constants import (
    NUM_CLASSES, THING_CLASSES, CAMERA_NAMES, PARQUET_COLS,
)
from panoctvps.data.decode import decode_jpeg
from panoctvps.data.transforms.augmentations import PadToMultiple
from panoctvps.data.transforms.formatting import ToTensorNormalize
from panoctvps.data.waymo_dataset import WODPVPSSequenceDataset
from panoctvps.evaluation import STQuality, merge_sequence_results
from panoctvps.evaluation.tracking_utils import build_gt_global_instance
from panoctvps.segmenter.inference import predict_panoptic_with_learned_embeddings
from panoctvps.segmenter.mask2former import build_model
from panoctvps.tracker.embedding_head import EmbeddingHead
from panoctvps.tracker.memory_bank import MemoryBankTracker


def _match_gt(inst_mask: np.ndarray, gt_global: np.ndarray):
    """Return (gt_global_id, gt_iou) for best IoU GT match, or (0, 0.0)."""
    gt_in_mask = gt_global[inst_mask]
    candidates = gt_in_mask[gt_in_mask > 0]
    if len(candidates) == 0:
        return 0, 0.0
    counts = np.bincount(candidates)
    best_gt_id = int(np.argmax(counts))
    gt_mask = gt_global == best_gt_id
    inter = int((inst_mask & gt_mask).sum())
    union = int((inst_mask | gt_mask).sum())
    iou = inter / (union + 1e-6)
    return best_gt_id, float(iou)


def main():
    parser = argparse.ArgumentParser(
        description="Write the prediction cache with ground-truth ids and report "
                    "the live wSTQ of the per-camera tracker."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Identity-training checkpoint (segmenter + embedding head)")
    parser.add_argument("--dataset-root", default="../data/waymo_open_dataset_v_2_0_1")
    parser.add_argument("--backbone", default="swin-large-mapillary",
                        help="Backbone name (default: swin-large-mapillary)")
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Camera IDs (default: 1 2 3 4 5)")
    parser.add_argument("--match-threshold", type=float, default=0.7,
                        help="Bi-softmax match threshold of the tracker")
    parser.add_argument("--init-score-threshold", type=float, default=0.01,
                        help="Minimum detection score to start a new track")
    parser.add_argument("--num-dead-frames", type=int, default=5,
                        help="Frames an unmatched track is kept")
    parser.add_argument("--obj-threshold", type=float, default=0.5,
                        help="M2F panoptic class-probability cutoff (HF default 0.5)")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="M2F mask-sigmoid binarisation cutoff (HF default 0.5)")
    parser.add_argument("--overlap-threshold", type=float, default=0.8,
                        help="M2F overlap-mask-area threshold (HF default 0.8)")
    parser.add_argument("--n-sequences", type=int, default=0,
                        help="Number of sequences to process (0 = all). "
                             "Ignored if --sequence-indices is provided.")
    parser.add_argument("--sequence-indices", type=int, nargs="+", default=None,
                        help="Specific sequence indices to process (subset mode).")
    parser.add_argument("--output", default=None, help="JSON output path for the wSTQ")
    parser.add_argument("--cache-dir", default=None, type=Path,
                        help="If set, write one .npz cache file per scene here.")
    parser.add_argument("--split", default="validation",
                        choices=["training", "validation"],
                        help="Dataset split (default: validation).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output path
    output_path = args.output
    if output_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint) or "."
        output_path = os.path.join(ckpt_dir, f"wstq_live_{args.split}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load model + embedding head
    print(f"Loading model ({args.backbone}) from {args.checkpoint} ...")
    model = build_model(backbone=args.backbone, num_classes=NUM_CLASSES, pretrained=False)
    embed_head = EmbeddingHead()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    embed_head.load_state_dict(ckpt["embed_head"])

    model = model.to(device)
    embed_head = embed_head.to(device)
    model.eval()
    embed_head.eval()

    # Load sequences
    dataset = WODPVPSSequenceDataset(
        dataset_root=args.dataset_root, split=args.split
    )

    all_seq_metrics = []
    per_sequence_results = []
    per_cam_stq = {
        cam_id: STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)
        for cam_id in args.cameras
    }

    _pads = {c: PadToMultiple(multiple=32) for c in args.cameras}
    fmt = ToTensorNormalize()

    if args.sequence_indices is not None:
        seq_indices = sorted(set(args.sequence_indices))
    else:
        n_seq_lim = len(dataset) if args.n_sequences <= 0 else min(args.n_sequences, len(dataset))
        seq_indices = list(range(n_seq_lim))
    n_seq = len(seq_indices)
    print(f"\nProcessing {n_seq} sequences (indices={seq_indices}), "
          f"cameras={args.cameras} ...")
    print(f"M2F post-proc: obj={args.obj_threshold}, mask={args.mask_threshold}, "
          f"overlap={args.overlap_threshold}")
    print(f"Tracker: match={args.match_threshold}, init={args.init_score_threshold}, "
          f"dead_frames={args.num_dead_frames}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for loop_i, i in enumerate(seq_indices):
        sequence = dataset[i]
        seg_name = sequence["segment_name"][:40]
        n_ts = sequence["num_timestamps"]

        print(f"  [{loop_i + 1}/{n_seq}] (idx {i}) {seg_name}... ({n_ts} timestamps)", end="", flush=True)
        t_seq = time.time()

        seq_stq = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)

        # Per-camera trackers
        trackers = {
            cam_id: MemoryBankTracker(
                match_threshold=args.match_threshold,
                init_score_threshold=args.init_score_threshold,
                num_dead_frames=args.num_dead_frames,
            )
            for cam_id in args.cameras
        }

        # Load images (and pose for ego_pose)
        split_dir = os.path.join(args.dataset_root, args.split)
        img_path = os.path.join(split_dir, "camera_image", sequence["segment_file"])
        img_cols = [
            PARQUET_COLS["img_key_timestamp"],
            PARQUET_COLS["img_key_camera"],
            PARQUET_COLS["img_image"],
        ]
        if cache_dir is not None:
            img_cols.append(PARQUET_COLS["img_pose"])
        img_df = pq.read_table(img_path, columns=img_cols).to_pandas()

        # Load geometry once per segment for cache
        if cache_dir is not None:
            extrinsics, intrinsics = load_geometry(
                args.dataset_root, args.split, sequence["segment_file"],
                args.cameras,
            )
            unique_timestamps = np.array(
                [f["timestamp"] for f in sequence["frames"]], dtype=np.int64
            )
            # ego_pose: use FRONT camera (cam_id=1) pose per timestamp as vehicle pose proxy
            ego_by_ts = {}
            front_rows = img_df[img_df[PARQUET_COLS["img_key_camera"]] == 1]
            for _, row in front_rows.iterrows():
                ts = int(row[PARQUET_COLS["img_key_timestamp"]])
                ego_by_ts[ts] = np.array(row[PARQUET_COLS["img_pose"]],
                                         dtype=np.float32).reshape(4, 4)
            ego_poses = np.stack(
                [ego_by_ts.get(int(ts), np.eye(4, dtype=np.float32))
                 for ts in unique_timestamps], axis=0
            )  # (T, 4, 4)
            # Read sequence_id per timestamp from the seg parquet so the cache
            # carries one sequence_idx per annotation window (a training scene
            # has several windows, a validation scene one). The sequence
            # dataset exposes only the first window, so read it directly here.
            seg_path = os.path.join(
                args.dataset_root, args.split, "camera_segmentation",
                sequence["segment_file"],
            )
            seg_seq_df = pq.read_table(seg_path, columns=[
                PARQUET_COLS["seg_key_timestamp"],
                PARQUET_COLS["seg_sequence_id"],
            ]).to_pandas()
            ts_to_seq: Dict[int, str] = {}
            for _, row in seg_seq_df.iterrows():
                ts_to_seq.setdefault(
                    int(row[PARQUET_COLS["seg_key_timestamp"]]),
                    str(row[PARQUET_COLS["seg_sequence_id"]]),
                )
            unique_sequence_ids = np.array(
                sorted(set(ts_to_seq.values())), dtype=object,
            )
            seq_id_to_idx = {s: i for i, s in enumerate(unique_sequence_ids.tolist())}
            ts_to_seq_idx: Dict[int, int] = {
                t: seq_id_to_idx[s] for t, s in ts_to_seq.items()
            }
            # Cache accumulators
            cache_frame_records: List = []
            cache_records: List = []
            cache_embeddings: List = []
            cache_mask_rles: List = []
            cache_pred_sem_rles: List = []
            cache_frame_queries: List = []
            cache_frame_class_logits: List = []
            cache_frame_idx = 0  # global frame index within segment

        frame_counter = 0
        for frame in sequence["frames"]:
            timestamp = frame["timestamp"]

            for cam_id in args.cameras:
                if cam_id not in frame["cameras"]:
                    continue
                gt_data = frame["cameras"][cam_id]

                img_row = img_df[
                    (img_df[PARQUET_COLS["img_key_timestamp"]] == timestamp)
                    & (img_df[PARQUET_COLS["img_key_camera"]] == cam_id)
                ]
                if img_row.empty:
                    continue

                image = decode_jpeg(img_row.iloc[0][PARQUET_COLS["img_image"]])
                oh, ow = image.shape[:2]

                sample = {
                    "image": image,
                    "semantic_label": np.zeros(image.shape[:2], dtype=np.int32),
                    "instance_label": np.zeros(image.shape[:2], dtype=np.int32),
                }
                sample = _pads[cam_id](sample)
                sample = fmt(sample)
                ph, pw = sample["image"].shape[-2], sample["image"].shape[-1]

                # M2F + embedding head inference
                pv = sample["image"].unsqueeze(0).to(device)
                preds = predict_panoptic_with_learned_embeddings(
                    model, embed_head, pv,
                    threshold=args.obj_threshold,
                    mask_threshold=args.mask_threshold,
                    overlap_mask_area_threshold=args.overlap_threshold,
                    return_raw=(cache_dir is not None),
                )

                pred_sem = preds[0]["semantic"][:oh, :ow]
                pred_inst = preds[0]["instance"][:oh, :ow]
                pred_embs = preds[0]["instance_embeddings"]
                pred_scores = preds[0]["instance_scores"]
                pred_classes = preds[0]["instance_classes"]

                gt_sem = gt_data["semantic_label"]
                gt_global = build_gt_global_instance(
                    gt_data["instance_label"], gt_data["instance_mapping"]
                )
                pixel_weights = sequence["pixel_weights"][cam_id]

                # Per-camera tracker (bi-softmax + memory bank)
                tracked = trackers[cam_id].update(
                    pred_sem, pred_inst, pred_embs,
                    pred_scores, pred_classes, frame_counter,
                )

                # Metrics: per-camera STQ and per-sequence wSTQ
                tracked_agg = np.where(tracked > 0, tracked + cam_id * 100000, 0)
                seq_offset = i * 1_000_000
                per_cam_stq[cam_id].update(
                    pred_semantic=pred_sem,
                    pred_instance=np.where(tracked > 0, tracked + seq_offset, 0),
                    gt_semantic=gt_sem,
                    gt_instance=np.where(gt_global > 0, gt_global + seq_offset, 0),
                )
                seq_stq.update(
                    pred_semantic=pred_sem,
                    pred_instance=tracked_agg,
                    gt_semantic=gt_sem,
                    gt_instance=gt_global,
                    pixel_weights=pixel_weights,
                )

                # ── Cache writing ─────────────────────────────────────────
                if cache_dir is not None:
                    cache_frame_queries.append(preds[0]["frame_queries"])      # (200, 256)
                    cache_frame_class_logits.append(preds[0]["frame_class_logits"])  # (200, 30)

                    # Encode semantic prediction as per-class RLE
                    cache_pred_sem_rles.append(encode_sem_rle(pred_sem))

                    inst_ids = [k for k in pred_embs.keys()]
                    rec_start = len(cache_records)

                    query_idx_map = preds[0].get("instance_query_idx", {})

                    for local_idx, inst_id in enumerate(inst_ids):
                        mask_orig = (pred_inst == inst_id).astype(np.uint8)
                        # Masks are stored at the padded resolution
                        mask_padded = np.zeros((ph, pw), dtype=np.uint8)
                        mask_padded[:oh, :ow] = mask_orig
                        cache_mask_rles.append(encode_rle(mask_padded))

                        # Tracker id for this instance
                        t_ids = tracked[pred_inst == inst_id]
                        t_ids_pos = t_ids[t_ids > 0]
                        tracker_id = int(np.bincount(t_ids_pos).argmax()) \
                            if len(t_ids_pos) > 0 else 0

                        # GT match
                        gt_gid, gt_iou = _match_gt(
                            pred_inst == inst_id, gt_global
                        )

                        # Centroid + bbox from original-res mask
                        ys, xs = np.where(mask_orig)
                        centroid_u = float(xs.mean()) if len(xs) > 0 else 0.0
                        centroid_v = float(ys.mean()) if len(ys) > 0 else 0.0
                        bbox_u0 = float(xs.min()) if len(xs) > 0 else 0.0
                        bbox_v0 = float(ys.min()) if len(ys) > 0 else 0.0
                        bbox_u1 = float(xs.max()) if len(xs) > 0 else 0.0
                        bbox_v1 = float(ys.max()) if len(ys) > 0 else 0.0

                        cache_records.append((
                            i,                          # segment_idx
                            timestamp,                  # timestamp
                            cam_id,                     # camera_id
                            local_idx,                  # local_id
                            tracker_id,                 # tracker_id
                            pred_classes.get(inst_id, 0),  # class_id
                            pred_scores.get(inst_id, 0.0),  # score
                            centroid_u, centroid_v,
                            bbox_u0, bbox_v0, bbox_u1, bbox_v1,
                            int(mask_orig.sum()),       # pixel_count
                            gt_gid,                     # gt_global_id
                            gt_iou,                     # gt_iou
                            cache_frame_idx,            # frame_idx
                            query_idx_map.get(inst_id, -1),  # query_idx
                            ts_to_seq_idx[int(timestamp)],  # sequence_idx
                        ))
                        cache_embeddings.append(pred_embs[inst_id])

                    cache_frame_records.append((
                        i, timestamp, cam_id, oh, ow,
                        len(inst_ids), rec_start, len(cache_records),
                    ))
                    cache_frame_idx += 1

            frame_counter += 1

        # Write segment cache npz
        if cache_dir is not None:
            npz_path = write_segment_npz(
                cache_dir=cache_dir,
                seg_file=sequence["segment_file"],
                seg_name=sequence["segment_name"],
                seg_idx=i,
                unique_sequence_ids=unique_sequence_ids,
                unique_timestamps=unique_timestamps,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                ego_poses=ego_poses,
                coverage_by_cam=sequence["pixel_weights"],
                frame_records_list=cache_frame_records,
                records_list=cache_records,
                embeddings_list=cache_embeddings,
                mask_rles=cache_mask_rles,
                pred_sem_rles=cache_pred_sem_rles,
                frame_queries_list=cache_frame_queries,
                frame_class_logits_list=cache_frame_class_logits,
            )
            print(f" [cached→{npz_path.name}]", end="")

        # Per-sequence results
        seq_r = seq_stq.result()
        all_seq_metrics.append(seq_stq)
        per_sequence_results.append({
            "segment_name": sequence["segment_name"],
            "wSTQ": seq_r["STQ"],
            "wAQ": seq_r["AQ"],
            "wSQ": seq_r["SQ"],
        })
        elapsed_seq = time.time() - t_seq
        print(f"  wSTQ={seq_r['STQ']:.3f} ({elapsed_seq:.1f}s)")

    total_elapsed = time.time() - t0

    # ---- Results ----
    agg = merge_sequence_results(all_seq_metrics)
    print(f"\n{'=' * 60}")
    print(f"  {n_seq} sequences, {total_elapsed:.0f}s total")
    print(f"{'=' * 60}")
    print(f"  wSTQ {agg['STQ']:.4f}   wAQ {agg['AQ']:.4f}   wSQ {agg['SQ']:.4f}")

    per_cam = {}
    for cam_id in args.cameras:
        cam_r = per_cam_stq[cam_id].result()
        per_cam[CAMERA_NAMES[cam_id]] = {
            "STQ": cam_r["STQ"],
            "AQ": cam_r["AQ"],
            "SQ": cam_r["SQ"],
        }
    if len(args.cameras) > 1:
        print(f"\n  {'Camera':<15s} {'STQ':>7s} {'AQ':>7s} {'SQ':>7s}")
        print(f"  {'-' * 40}")
        for name, r in per_cam.items():
            print(f"  {name:<15s} {r['STQ']:7.4f} {r['AQ']:7.4f} {r['SQ']:7.4f}")

    all_results = {
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "split": args.split,
        "cameras": args.cameras,
        "match_threshold": args.match_threshold,
        "init_score_threshold": args.init_score_threshold,
        "num_dead_frames": args.num_dead_frames,
        "obj_threshold": args.obj_threshold,
        "mask_threshold": args.mask_threshold,
        "overlap_threshold": args.overlap_threshold,
        "sequence_indices": seq_indices,
        "n_sequences": n_seq,
        "timing_seconds": round(total_elapsed, 1),
        "aggregate": {
            "wSTQ": agg["STQ"],
            "wAQ": agg["AQ"],
            "wSQ": agg["SQ"],
            "per_class_aq": agg["per_class_aq"],
        },
        "per_camera": per_cam,
        "per_sequence": per_sequence_results,
    }

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
