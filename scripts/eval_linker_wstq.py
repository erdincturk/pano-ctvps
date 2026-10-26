#!/usr/bin/env python3
"""eval_linker_wstq.py — wSTQ of the linker on the validation split.

Takes the per-camera tracks stored in the prediction cache
(`records['tracker_id_deployed']`), runs the linker (proposal -> scoring ->
assignment) for each configuration of --configs, and evaluates STQ with the
benchmark conventions (STQuality per scene, merge_sequence_results).

Two kinds of identity maps are scored in one pass per scene:

  * tracker only: id = track id, namespaced by camera (cam * 100000 + tid), so
    no identity is shared across cameras;
  * linker: id = the global identity of the linker, shared across cameras by
    linked tracks. Unlinked tracks keep a unique id.

With --percam-check, the per-camera STQ of the first configuration is also
accumulated; it equals that of the tracker alone, because the linker never
merges two tracks of the same camera.

Usage:
  python scripts/eval_linker_wstq.py --cache-dir outputs/tracker_ctvis_val_cache \
      --cosines-dir outputs/linker/adapted_cosines --configs configs/linker_table7.json \
      --output outputs/linker/table7.json

--configs is a JSON file of named configurations:
  {"name": {"tau": 0.4 | {"F-FR|vehicle": 0.35, "human": 0.25, "vehicle": 0.45, ...},
            "rank_key": "cos_med" | null,
            "veto_key": "cos_med_adapted", "veto_min": 0.5, "veto_rescue": 0.6}, ...}
(the veto keys are optional).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.data.constants import CAMERA_NAMES, NUM_CLASSES, THING_CLASSES
from panoctvps.data.waymo_dataset import WODPVPSSequenceDataset
from panoctvps.evaluation import STQuality, merge_sequence_results
from panoctvps.evaluation.tracking_utils import build_gt_global_instance
from panoctvps.linker.overlap import calibration_from_arrays
from panoctvps.linker.pipeline import DEFAULT_TID_FIELD as TID_FIELD, linker_assignment

REPO = Path(__file__).resolve().parents[1]
UNMATCHED_OFFSET = 1 << 40           # keep unlinked tracks distinct, no collision


def new_stq():
    return STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "outputs/tracker_ctvis_val_cache")
    ap.add_argument("--dataset-root", default="../data/waymo_open_dataset_v_2_0_1")
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--cosines-dir", type=Path,
                    default=REPO / "outputs/linker/adapted_cosines",
                    help="Adapted cosines (scripts/build_cosines.py) for the appearance veto.")
    ap.add_argument("--configs", type=Path, default=REPO / "configs/linker_deployed.json",
                    help="JSON file of named linker configurations (see above).")
    ap.add_argument("--n-sequences", type=int, default=0,
                    help="Evaluate only the first N scenes (0 = all).")
    ap.add_argument("--percam-check", action="store_true",
                    help="Also accumulate the per-camera STQ of the first configuration "
                         "(equal to that of the tracker alone).")
    ap.add_argument("--output", type=Path, default=REPO / "outputs/linker/linker_wstq.json")
    args = ap.parse_args()
    configs = [(name, cfg) for name, cfg in json.loads(args.configs.read_text()).items()]
    names = [n for n, _ in configs]

    gt_dataset = WODPVPSSequenceDataset(dataset_root=args.dataset_root, split="validation")
    seg_file_to_idx = {f: i for i, f in enumerate(gt_dataset.segment_files)}
    cache_paths = sorted(args.cache_dir.glob("*.npz"))
    if args.n_sequences > 0:
        cache_paths = cache_paths[:args.n_sequences]
    print(f"wSTQ evaluation over {len(cache_paths)} scenes; configurations: {names}")

    # Per-scene STQ accumulators merged at the end.
    floor_seq: List[STQuality] = []
    linker_seq: Dict[str, List[STQuality]] = {n: [] for n in names}
    floor_per_cam = {c: new_stq() for c in args.cameras}
    check_per_cam = {c: new_stq() for c in args.cameras}
    check_name = names[0]
    per_scene_rows = []

    t0 = time.time()
    for s_idx, cpath in enumerate(cache_paths):
        npz = np.load(cpath, allow_pickle=True)
        seg_file = str(npz["segment_file"]); seg_name = str(npz["segment_name"])
        if seg_file not in seg_file_to_idx:
            print(f"  [{s_idx+1}] {seg_name[:34]} — GT not found, skip"); continue
        gt_seq = gt_dataset[seg_file_to_idx[seg_file]]
        gt_by_ts_cam = {}
        for frame in gt_seq["frames"]:
            for cam_id, cam_data in frame["cameras"].items():
                gt_by_ts_cam[(frame["timestamp"], cam_id)] = cam_data
        pixel_weights = gt_seq["pixel_weights"]

        wh = {}
        for fr in npz["frame_records"]:
            wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
        cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)
        assign = linker_assignment(npz, cams, configs, cosines_path=args.cosines_dir / cpath.name)

        floor_stq = new_stq()
        m_stq = {n: new_stq() for n in names}
        records = npz["records"]
        seq_off = s_idx * 1_000_000

        for fr_idx in range(len(npz["frame_records"])):
            fr = npz["frame_records"][fr_idx]
            ts = int(fr["timestamp"]); cam_id = int(fr["camera_id"])
            if cam_id not in args.cameras or (ts, cam_id) not in gt_by_ts_cam:
                continue
            pred_sem, pred_inst = reconstruct_frame(npz, fr_idx)[:2]
            gt_cam = gt_by_ts_cam[(ts, cam_id)]
            gt_sem = gt_cam["semantic_label"]
            gt_global = build_gt_global_instance(gt_cam["instance_label"], gt_cam["instance_mapping"])
            if pred_sem.shape != gt_sem.shape:
                continue
            pw = pixel_weights[cam_id]
            rs, re = int(fr["records_start"]), int(fr["records_end"])

            # local instance id -> track id
            floor_inst = np.zeros(pred_inst.shape, dtype=np.int64)      # camera-namespaced
            m_inst = {n: np.zeros(pred_inst.shape, dtype=np.int64) for n in names}  # global
            for k in range(re - rs):
                tid = int(records[rs + k][TID_FIELD])
                if tid <= 0:
                    continue
                sel = pred_inst == (k + 1)
                if not sel.any():
                    continue
                uid = cam_id * 100_000 + tid + UNMATCHED_OFFSET
                floor_inst[sel] = tid + cam_id * 100_000
                for n in names:
                    gid = assign[n].get((cam_id, tid))
                    m_inst[n][sel] = gid if gid is not None else uid

            # tracker only: aggregate + per-camera
            floor_stq.update(pred_semantic=pred_sem, pred_instance=floor_inst,
                             gt_semantic=gt_sem, gt_instance=gt_global, pixel_weights=pw)
            floor_per_cam[cam_id].update(
                pred_semantic=pred_sem,
                pred_instance=np.where(floor_inst > 0, floor_inst + seq_off, 0),
                gt_semantic=gt_sem,
                gt_instance=np.where(gt_global > 0, gt_global + seq_off, 0))

            # linker: aggregate (shared global id) for each configuration
            for n in names:
                m_stq[n].update(pred_semantic=pred_sem, pred_instance=m_inst[n],
                                gt_semantic=gt_sem, gt_instance=gt_global, pixel_weights=pw)
            if args.percam_check:
                mc = m_inst[check_name]
                check_per_cam[cam_id].update(
                    pred_semantic=pred_sem,
                    pred_instance=np.where(mc > 0, mc + seq_off, 0),
                    gt_semantic=gt_sem,
                    gt_instance=np.where(gt_global > 0, gt_global + seq_off, 0))

        floor_seq.append(floor_stq)
        for n in names:
            linker_seq[n].append(m_stq[n])
        floor_res = floor_stq.result()
        mres = {n: m_stq[n].result() for n in names}
        per_scene_rows.append({
            "scene": seg_name,
            "tracker_only": {"wSTQ": floor_res["STQ"], "wAQ": floor_res["AQ"], "wSQ": floor_res["SQ"]},
            "linker": {n: {"wSTQ": mres[n]["STQ"], "wAQ": mres[n]["AQ"], "wSQ": mres[n]["SQ"]}
                       for n in names}})
        print(f"  [{s_idx+1}/{len(cache_paths)}] {seg_name[:34]:34s} tracker_only={floor_res['STQ']:.4f} "
              + " ".join(f"{n}={mres[n]['STQ']:.4f}" for n in names)
              + f"  ({time.time()-t0:.0f}s)", flush=True)

    # Aggregate
    floor_agg = merge_sequence_results(floor_seq)
    m_agg = {n: merge_sequence_results(linker_seq[n]) for n in names}
    floor_pc = {CAMERA_NAMES[c]: floor_per_cam[c].result() for c in args.cameras}
    floor_mstq = float(np.mean([floor_pc[CAMERA_NAMES[c]]["STQ"] for c in args.cameras]))
    if args.percam_check:
        check_pc = {CAMERA_NAMES[c]: check_per_cam[c].result() for c in args.cameras}
        check_mstq = float(np.mean([check_pc[CAMERA_NAMES[c]]["STQ"] for c in args.cameras]))

    print("\n========= RESULTS =========")
    print("Tracker only (no cross-camera identities):")
    print(f"  wSTQ={floor_agg['STQ']:.5f}  wAQ={floor_agg['AQ']:.5f}  wSQ={floor_agg['SQ']:.5f}")
    print("  per-camera STQ: " + "  ".join(f"{CAMERA_NAMES[c]}={floor_pc[CAMERA_NAMES[c]]['STQ']:.4f}"
                                       for c in args.cameras)
          + f"  (mean {floor_mstq:.5f})")
    if args.percam_check:
        print(f"\nLinker '{check_name}' per-camera STQ mean: {check_mstq:.5f}")
    print("\nLinker:")
    print(f"  {'configuration':>34} {'wSTQ':>9} {'Δ wSTQ':>9} {'wAQ':>9} {'wSQ':>9}")
    for n in names:
        a = m_agg[n]
        print(f"  {n:>34} {a['STQ']:>9.5f} {a['STQ']-floor_agg['STQ']:>+9.5f} {a['AQ']:>9.5f} {a['SQ']:>9.5f}")

    out = {
        "n_scenes": len(floor_seq),
        "configs": {n: cfg for n, cfg in configs},
        "tracker_only": {"wSTQ": floor_agg["STQ"], "wAQ": floor_agg["AQ"], "wSQ": floor_agg["SQ"],
                         "mSTQ": floor_mstq, "per_camera": floor_pc},
        "linker_per_camera_check": ({"config": check_name, "mSTQ": check_mstq, "per_camera": check_pc}
                                    if args.percam_check else None),
        "linker": {n: {"wSTQ": m_agg[n]["STQ"], "wAQ": m_agg[n]["AQ"], "wSQ": m_agg[n]["SQ"],
                       "delta_wSTQ": m_agg[n]["STQ"] - floor_agg["STQ"]} for n in names},
        "per_scene": per_scene_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
