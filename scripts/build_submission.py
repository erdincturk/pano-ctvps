"""build_submission.py — WOD 2D VPS submission binprotos from a prediction cache.

For each scene of the prediction cache:
  1. run the linker (proposal -> scoring + adapted cosines -> assignment with
     configs/linker_deployed.json) to get one global identity per
     (camera, track);
  2. per frame, reconstruct (pred_sem, pred_inst_local) and map each local
     detection -> its track -> global identity, building a per-pixel global
     instance map (stuff pixels and unassigned detections carry 0);
  3. pack each (frame, camera) into a CameraSegmentationLabel and accumulate
     one CameraSegmentationSubmission per scene, with the metadata below;
  4. write `<context>_camera_segmentation_submission.binproto`.

The identity maps are those of scripts/eval_linker_wstq.py, so the packed
validation predictions reproduce its wSTQ (check with verify_submission.py).
Only the timestamps listed in configs/splits/{split}_frames.txt are written.

Usage:
  python scripts/build_submission.py --split test --account <registered e-mail> \
      --cache-dir outputs/tracker_ctvis_test_cache --cosines-dir outputs/linker/adapted_cosines_test
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.linker.overlap import calibration_from_arrays
from panoctvps.linker.pipeline import DEFAULT_TID_FIELD as TID_FIELD, linker_assignment
from panoctvps.submission import packer

REPO = Path(__file__).resolve().parents[1]
CACHE_DIRS = {
    "validation": REPO / "outputs/tracker_ctvis_val_cache",
    "test": REPO / "outputs/tracker_ctvis_test_cache",
}
COSINE_DIRS = {
    "validation": REPO / "outputs/linker/adapted_cosines",
    "test": REPO / "outputs/linker/adapted_cosines_test",
}
SPLIT_FILES = {
    "validation": REPO / "configs/splits/validation_frames.txt",
    "test": REPO / "configs/splits/test_frames.txt",
}
LINKER_CONFIG = REPO / "configs/linker_deployed.json"


def requested_timestamps(split: str):
    by_ctx = collections.defaultdict(set)
    for line in SPLIT_FILES[split].read_text().splitlines():
        if line.strip():
            ctx, ts = line.strip().split(",")
            by_ctx[ctx].add(int(ts))
    return by_ctx


def build_scene(npz_path: Path, cosines_path: Path, want_ts, args) -> tuple:
    npz = np.load(npz_path, allow_pickle=True)
    context = str(npz["segment_name"])
    records = npz["records"]
    wh = {}
    for fr in npz["frame_records"]:
        wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
    cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)

    assign = linker_assignment(
        npz, cams, [("linker", args.linker_cfg)],
        cosines_path=cosines_path if cosines_path.exists() else None)["linker"]

    sub = packer.make_submission(
        account_name=args.account, unique_method_name=args.method_name,
        authors=args.authors.split(";"), affiliation=args.affiliation,
        description=args.description, method_link=args.method_link,
        frame_dt=1, runtime_ms=args.runtime_ms)

    n_frames = 0
    for fr_idx in range(len(npz["frame_records"])):
        fr = npz["frame_records"][fr_idx]
        ts, cam_id = int(fr["timestamp"]), int(fr["camera_id"])
        if ts not in want_ts:
            continue
        pred_sem, pred_inst = reconstruct_frame(npz, fr_idx)[:2]
        global_inst = np.zeros(pred_sem.shape, dtype=np.int64)
        rs, re = int(fr["records_start"]), int(fr["records_end"])
        for k in range(re - rs):
            tid = int(records[rs + k][TID_FIELD])
            if tid <= 0:
                continue
            sel = pred_inst == (k + 1)
            if not sel.any():
                continue
            gid = assign.get((cam_id, tid))
            if gid is None:                 # not a thing track
                continue
            global_inst[sel] = gid
        label = packer.pack_camera_label(pred_sem, global_inst, sequence_id=context)
        packer.add_frame(sub, context_name=context, frame_timestamp_micros=ts,
                         camera_name=cam_id, label=label)
        n_frames += 1
    return context, sub, n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--cosines-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: submissions/<split>/binprotos")
    ap.add_argument("--config", type=Path, default=LINKER_CONFIG,
                    help="linker configuration file (key 'deployed')")
    # submission metadata
    ap.add_argument("--account", required=True,
                    help="e-mail address of the registered Waymo Open Dataset account")
    ap.add_argument("--method-name", default="Pano-CTVPS")
    ap.add_argument("--authors", default="Erdinç Türk;Taner Danışman",
                    help="semicolon-separated author list")
    ap.add_argument("--affiliation", default="Akdeniz University")
    ap.add_argument("--description",
                    default="Per-camera panoptic segmenter, contrastive temporal tracker, and "
                            "cross-camera linker based on panoramic mask overlap with an "
                            "appearance veto.")
    ap.add_argument("--method-link", default="https://github.com/erdincturk/pano-ctvps")
    ap.add_argument("--runtime-ms", type=float, default=4100.0,
                    help="approximate per-frame runtime in ms")
    args = ap.parse_args()
    args.linker_cfg = json.loads(args.config.read_text())["deployed"]

    cache_dir = args.cache_dir or CACHE_DIRS[args.split]
    cosines_dir = args.cosines_dir or COSINE_DIRS[args.split]
    out_dir = args.out_dir or (REPO / "submissions" / args.split / "binprotos")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ctx = requested_timestamps(args.split)

    paths = sorted(cache_dir.glob("*.npz"))
    print(f"split={args.split}: {len(paths)} scenes, cache={cache_dir}")
    print(f"cosines={cosines_dir} (exists={cosines_dir.exists()}), out={out_dir}")
    t0 = time.time()
    tot_frames = 0
    for i, p in enumerate(paths, 1):
        ctx = str(np.load(p, allow_pickle=True)["segment_name"])
        want = by_ctx.get(ctx, set())
        context, sub, n = build_scene(p, cosines_dir / p.name, want, args)
        out = out_dir / f"{context}_camera_segmentation_submission.binproto"
        out.write_bytes(sub.SerializeToString())
        tot_frames += n
        print(f"  [{i}/{len(paths)}] {context[:34]:34s} frames={n:4d} "
              f"({out.stat().st_size/1e6:.1f} MB) ({time.time()-t0:.0f}s)", flush=True)
    exp = sum(len(v) for v in by_ctx.values())
    print(f"done in {time.time()-t0:.0f}s: {tot_frames} frame-cameras "
          f"(expected {exp*5} = {exp} ts x 5 cams)")


if __name__ == "__main__":
    main()
