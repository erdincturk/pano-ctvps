#!/usr/bin/env python3
"""Run a tracker on the cached segmenter outputs and evaluate it (CPU, no GPU needed).

Reads the per-scene prediction cache (segments, scores, classes and identity
embeddings of every frame), runs one tracker per camera and computes the
panoramic, pixel-weighted wSTQ / wAQ / wSQ and the STQ of each camera.

Trackers (--tracker):
  memorybank   bi-softmax memory-bank tracker; with --motion-gate, the motion
               gate on the centroid displacement is added (one bound per camera)
  minvis       MinVIS-style frame-to-frame cosine + Hungarian matching
  iou          greedy mask-IoU matching, no embeddings

Usage:
    # our tracker at the settings of the paper
    python scripts/eval_tracker_cached.py \
        --cache-dir outputs/tracker_ctvis_val_cache --tracker memorybank \
        --match-threshold 0.7 --init-score-threshold 0.01 --num-dead-frames 5 \
        --motion-gate --motion-gate-px-per-cam "1:600,2:800,3:800,4:1000,5:1000" \
        --output outputs/tracker_ctvis/eval.json

    # one bound for all cameras
    python scripts/eval_tracker_cached.py ... --motion-gate --motion-gate-px 800 ...

    # tracker only (no ground truth, e.g. the test split): save the track ids
    python scripts/eval_tracker_cached.py ... --no-eval --save-trajectories traj.npz --output unused.json
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
from panoctvps.data.cache_io import reconstruct_frame

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES, CAMERA_NAMES
from panoctvps.data.waymo_dataset import WODPVPSSequenceDataset
from panoctvps.evaluation import STQuality, merge_sequence_results
from panoctvps.evaluation.tracking_utils import build_gt_global_instance
from panoctvps.tracker.iou_tracker import IoUTracker
from panoctvps.tracker.memory_bank import MemoryBankTracker
from panoctvps.tracker.minvis_tracker import MinVISTracker
from panoctvps.tracker.motion_gate import MotionGateMemoryBankTracker


def load_cache_segments(cache_dir: str):
    """Return the sorted list of per-scene cache files."""
    files = [f for f in sorted(os.listdir(cache_dir)) if f.endswith(".npz")]
    return [os.path.join(cache_dir, f) for f in files]


def parse_per_cam(spec: str) -> Dict[int, float]:
    """Parse 'cam_id:value,cam_id:value,...' into {cam_id: value}."""
    out: Dict[int, float] = {}
    if spec:
        for kv in spec.split(","):
            k, v = kv.strip().split(":")
            out[int(k)] = float(v)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run a tracker on the cached segmenter outputs and evaluate it",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--cache-dir", required=True,
                        help="Directory of per-scene prediction caches (*.npz)")
    parser.add_argument("--dataset-root", default="../data/waymo_open_dataset_v_2_0_1")
    parser.add_argument("--cameras", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--tracker", choices=("memorybank", "minvis", "iou"),
                        default="memorybank",
                        help="'memorybank' = our tracker (bi-softmax memory bank). "
                             "'minvis' = frame-to-frame cosine + Hungarian. "
                             "'iou' = greedy mask-IoU, no embeddings.")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="(--tracker iou) minimum mask IoU to accept a match.")
    parser.add_argument("--match-threshold", type=float, default=0.7,
                        help="Minimum match score (bi-softmax for memorybank, "
                             "cosine for minvis).")
    parser.add_argument("--init-score-threshold", type=float, default=0.01,
                        help="Minimum detection score to start a new track.")
    parser.add_argument("--num-dead-frames", type=int, default=5,
                        help="Frames an unmatched track is kept.")
    parser.add_argument("--motion-gate", action="store_true",
                        help="(--tracker memorybank) enable the motion gate.")
    parser.add_argument("--motion-gate-px", type=float, default=None,
                        help="Motion-gate bound in pixels for every camera.")
    parser.add_argument("--motion-gate-px-per-cam", type=str, default="",
                        help="Motion-gate bound per camera, 'cam_id:px,...', e.g. "
                             "'1:600,2:800,3:800,4:1000,5:1000'. Cameras not listed "
                             "use --motion-gate-px.")
    parser.add_argument("--n-sequences", type=int, default=0,
                        help="0 = all scenes in the cache")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--no-eval", action="store_true",
                        help="Run the tracker and save trajectories only: no ground "
                             "truth, no STQ (for splits without ground truth).")
    parser.add_argument("--save-trajectories", type=str, default="",
                        help="Optional NPZ path for the track id of every cache record: "
                             "{segment_files: str[N_scenes], tracker_ids: "
                             "object[N_scenes] of int32[N_records_in_scene]}, 0 = no track.")
    args = parser.parse_args()

    # Motion-gate bound of each camera
    gate_px: Dict[int, float] = {}
    if args.motion_gate:
        if args.tracker != "memorybank":
            parser.error("--motion-gate applies to --tracker memorybank only")
        per_cam = parse_per_cam(args.motion_gate_px_per_cam)
        for cam in args.cameras:
            px = per_cam.get(cam, args.motion_gate_px)
            if px is None:
                parser.error(f"no motion-gate bound for camera {cam}: "
                             "set --motion-gate-px or --motion-gate-px-per-cam")
            gate_px[cam] = px
    elif args.motion_gate_px is not None or args.motion_gate_px_per_cam:
        parser.error("--motion-gate-px / --motion-gate-px-per-cam require --motion-gate")

    tracker_config = {
        "tracker": args.tracker,
        "match_threshold": args.match_threshold,
        "init_score_threshold": args.init_score_threshold,
        "num_dead_frames": args.num_dead_frames,
        "iou_threshold": args.iou_threshold if args.tracker == "iou" else None,
        "motion_gate": args.motion_gate,
        "motion_gate_px_per_cam": gate_px,
    }

    print(f"Cache dir: {args.cache_dir}")
    print(f"tracker: {args.tracker}")
    if args.tracker == "iou":
        print(f"iou_threshold: {args.iou_threshold}")
    else:
        print(f"match_threshold: {args.match_threshold}")
        print(f"init_score_threshold: {args.init_score_threshold}")
    print(f"num_dead_frames: {args.num_dead_frames}")
    if args.motion_gate:
        print(f"motion gate (px): {gate_px}")
    print(f"cameras: {args.cameras}")

    cache_paths = load_cache_segments(args.cache_dir)
    if args.n_sequences > 0:
        cache_paths = cache_paths[:args.n_sequences]
    print(f"Sequences to process: {len(cache_paths)}")

    # Lookup from segment_file -> sequence index (for the ground truth)
    if args.no_eval:
        gt_dataset = None
        seg_file_to_idx = {}
    else:
        gt_dataset = WODPVPSSequenceDataset(
            dataset_root=args.dataset_root, split="validation"
        )
        seg_file_to_idx = {f: i for i, f in enumerate(gt_dataset.segment_files)}

    # Per-camera STQ (unweighted) and aggregate wSTQ (weighted)
    per_cam_stq = {
        cam: STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)
        for cam in args.cameras
    }
    seq_metrics: List[STQuality] = []
    per_seq_results = []

    # Track id of every cache record, per scene (--save-trajectories)
    save_traj = bool(args.save_trajectories)
    traj_per_seg_tids: Dict[str, np.ndarray] = {}    # seg_file -> int32[n_records_in_seg]
    traj_per_seg_files: List[str] = []

    t0 = time.time()
    for s_idx, cache_path in enumerate(cache_paths):
        npz = np.load(cache_path, allow_pickle=True)
        seg_file = str(npz["segment_file"])
        seg_name = str(npz["segment_name"])
        if args.no_eval:
            gt_by_ts_cam = None
            pixel_weights = None
        else:
            if seg_file not in seg_file_to_idx:
                print(f"  [{s_idx + 1}/{len(cache_paths)}] {seg_name[:36]} — "
                      f"GT sequence not found, skipping")
                continue
            gt_seq = gt_dataset[seg_file_to_idx[seg_file]]

            # Index GT by (timestamp, camera)
            gt_by_ts_cam = {}
            for frame in gt_seq["frames"]:
                ts = frame["timestamp"]
                for cam_id, cam_data in frame["cameras"].items():
                    gt_by_ts_cam[(ts, cam_id)] = cam_data
            pixel_weights = gt_seq["pixel_weights"]

        # One tracker per camera (independent state per camera)
        if args.tracker == "iou":
            trackers = {
                cam: IoUTracker(
                    iou_threshold=args.iou_threshold,
                    max_age=args.num_dead_frames,
                )
                for cam in args.cameras
            }
        elif args.tracker == "minvis":
            trackers = {
                cam: MinVISTracker(
                    match_threshold=args.match_threshold,
                    init_score_threshold=args.init_score_threshold,
                    num_dead_frames=args.num_dead_frames,
                )
                for cam in args.cameras
            }
        elif args.motion_gate:
            trackers = {
                cam: MotionGateMemoryBankTracker(
                    motion_gate_px=gate_px[cam],
                    match_threshold=args.match_threshold,
                    init_score_threshold=args.init_score_threshold,
                    num_dead_frames=args.num_dead_frames,
                )
                for cam in args.cameras
            }
        else:
            trackers = {
                cam: MemoryBankTracker(
                    match_threshold=args.match_threshold,
                    init_score_threshold=args.init_score_threshold,
                    num_dead_frames=args.num_dead_frames,
                )
                for cam in args.cameras
            }

        # Per-camera frame counter (frame_id within the camera's stream)
        cam_frame_counter = defaultdict(int)

        # Track id per record of this scene, 0 = no track
        if save_traj:
            seg_traj_tids = np.zeros(len(npz["records"]), dtype=np.int32)

        # Per-sequence weighted STQ (for aggregate wSTQ)
        seq_stq = STQuality(
            num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0
        )

        # Frames in temporal order, per-camera streams
        order = sorted(
            range(len(npz["frame_records"])),
            key=lambda i: (
                int(npz["frame_records"][i]["timestamp"]),
                int(npz["frame_records"][i]["camera_id"]),
            ),
        )

        t_seq = time.time()
        n_processed = 0
        for fr_idx in order:
            fr = npz["frame_records"][fr_idx]
            ts = int(fr["timestamp"])
            cam_id = int(fr["camera_id"])
            if cam_id not in args.cameras:
                continue
            if not args.no_eval and (ts, cam_id) not in gt_by_ts_cam:
                continue

            pred_sem, pred_inst, embs, scores, classes, centroids, px_counts = \
                reconstruct_frame(npz, fr_idx)
            if not args.no_eval:
                gt_cam = gt_by_ts_cam[(ts, cam_id)]
                gt_sem = gt_cam["semantic_label"]
                gt_global = build_gt_global_instance(
                    gt_cam["instance_label"], gt_cam["instance_mapping"]
                )

                # Skip if shape mismatch (cache vs GT) — safety net
                if pred_sem.shape != gt_sem.shape:
                    continue

            if args.motion_gate:
                tracked = trackers[cam_id].update(
                    pred_sem, pred_inst, embs, scores, classes,
                    cam_frame_counter[cam_id],
                    instance_centroids=centroids,
                )
            else:
                tracked = trackers[cam_id].update(
                    pred_sem, pred_inst, embs, scores, classes,
                    cam_frame_counter[cam_id],
                )
            cam_frame_counter[cam_id] += 1

            # Track id per record: record k of this frame is local instance k+1
            if save_traj:
                rs = int(fr["records_start"])
                re_ = int(fr["records_end"])
                for k in range(re_ - rs):
                    local_id = k + 1
                    inst_mask = (pred_inst == local_id)
                    if not inst_mask.any():
                        continue
                    tids_here = np.unique(tracked[inst_mask])
                    nz = tids_here[tids_here > 0]
                    if nz.size > 0:
                        seg_traj_tids[rs + k] = int(nz[0])

            # Per-camera (unweighted) — namespace by sequence to avoid ID collisions
            if not args.no_eval:
                seq_offset = s_idx * 1_000_000
                per_cam_stq[cam_id].update(
                    pred_semantic=pred_sem,
                    pred_instance=np.where(tracked > 0, tracked + seq_offset, 0),
                    gt_semantic=gt_sem,
                    gt_instance=np.where(gt_global > 0, gt_global + seq_offset, 0),
                )

                # Aggregate (weighted, panoramic merge) — namespace by camera within seq
                tracked_agg = np.where(tracked > 0, tracked + cam_id * 100000, 0)
                seq_stq.update(
                    pred_semantic=pred_sem,
                    pred_instance=tracked_agg,
                    gt_semantic=gt_sem,
                    gt_instance=gt_global,
                    pixel_weights=pixel_weights[cam_id],
                )

            n_processed += 1

        if not args.no_eval:
            seq_metrics.append(seq_stq)
            seq_r = seq_stq.result()
            per_seq_results.append({
                "segment_name": seg_name,
                "wSTQ": seq_r["STQ"],
                "wAQ": seq_r["AQ"],
                "wSQ": seq_r["SQ"],
            })

        # Keyed by the cache file name ("<segment>.npz")
        if save_traj:
            cache_basename = os.path.basename(cache_path)
            traj_per_seg_files.append(cache_basename)
            traj_per_seg_tids[cache_basename] = seg_traj_tids
        elapsed = time.time() - t_seq
        wstq_str = "" if args.no_eval else f", wSTQ={seq_r['STQ']:.4f}"
        print(f"  [{s_idx + 1}/{len(cache_paths)}] {seg_name[:36]} — "
              f"{n_processed} frames{wstq_str} ({elapsed:.1f}s)")

    total = time.time() - t0
    print(f"\nTotal: {total:.0f}s")

    if not args.no_eval:
        agg = merge_sequence_results(seq_metrics)
        print(f"\nAggregate (panoramic, pixel-weighted):")
        print(f"  wSTQ={agg['STQ']:.4f}  wAQ={agg['AQ']:.4f}  wSQ={agg['SQ']:.4f}")

        print(f"\nPer-camera (unweighted STQ):")
        print(f"  {'Camera':<14s} {'STQ':>7s} {'AQ':>7s} {'SQ':>7s}")
        print(f"  {'-' * 38}")
        per_cam_results = {}
        for cam in args.cameras:
            r = per_cam_stq[cam].result()
            per_cam_results[CAMERA_NAMES[cam]] = {
                "STQ": r["STQ"], "AQ": r["AQ"], "SQ": r["SQ"],
            }
            print(f"  {CAMERA_NAMES[cam]:<14s} {r['STQ']:7.4f} {r['AQ']:7.4f} {r['SQ']:7.4f}")

        out = {
            "cache_dir": args.cache_dir,
            "cameras": args.cameras,
            "tracker_config": tracker_config,
            "n_sequences": len(seq_metrics),
            "timing_seconds": round(total, 1),
            "aggregate": {"wSTQ": agg["STQ"], "wAQ": agg["AQ"], "wSQ": agg["SQ"]},
            "per_camera": per_cam_results,
            "per_sequence": per_seq_results,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to: {args.output}")

    if save_traj and traj_per_seg_files:
        traj_path = args.save_trajectories
        os.makedirs(os.path.dirname(traj_path) or ".", exist_ok=True)
        # segment_files[i] -> tracker_ids[i] (object array: lengths differ per scene)
        np.savez_compressed(
            traj_path,
            segment_files=np.array(traj_per_seg_files, dtype=object),
            tracker_ids=np.array(
                [traj_per_seg_tids[s] for s in traj_per_seg_files], dtype=object,
            ),
            tracker_config=json.dumps(tracker_config),
        )
        print(f"Saved trajectories to: {traj_path}  "
              f"({sum(t.size for t in traj_per_seg_tids.values()):,} records, "
              f"{len(traj_per_seg_files)} segments)")


if __name__ == "__main__":
    main()
