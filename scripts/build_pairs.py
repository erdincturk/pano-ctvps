"""build_pairs.py — detection pairs for training the appearance adapter.

Per frame and adjacent camera pair of the training split: all thing-class
detections whose masks reach the overlap band towards the other camera,
paired across the two cameras when they have the same super-class. Each
detection carries the ground-truth instance stored in the cache
(`gt_global_id`) and its IoU with that instance (`gt_iou`):

  label  1  same instance        (gidA == gidB > 0)
  label  0  different instances  (gidA, gidB > 0, different)
  label -1  no instance on one side (gid == 0)

scripts/train_adapter.py discards detections with an IoU below 0.3.

Usage:
  python scripts/build_pairs.py --cache-dir outputs/tracker_ctvis_train_cache \
      --out-dir outputs/linker/pairs_train --workers 16
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from panoctvps.data.cache_io import decode_rle
from panoctvps.linker.overlap import ADJACENT_PAIRS, calibration_from_arrays, overlap_by_partner
from panoctvps.linker.proposer import SUPERCLASS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO / "outputs/tracker_ctvis_train_cache"
DEFAULT_OUT = REPO / "outputs/linker/pairs_train"

PAIR_DTYPE = np.dtype([
    ("recA", "i4"), ("recB", "i4"), ("camA", "i1"), ("camB", "i1"),
    ("classA", "i2"), ("classB", "i2"), ("gidA", "i8"), ("gidB", "i8"),
    ("gt_iouA", "f4"), ("gt_iouB", "f4"), ("scoreA", "f4"), ("scoreB", "f4"),
    ("label", "i1"), ("timestamp", "i8"),
])


def extract_scene(cache_npz: Path, out_path: Path) -> dict:
    """One scene: in-band thing detections per (timestamp, camera, partner),
    paired across adjacent cameras, labelled by gt_global_id."""
    npz = np.load(cache_npz, allow_pickle=True)
    records, mask_rle = npz["records"], npz["mask_rle"]
    wh = {}
    for fr in npz["frame_records"]:
        wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
    cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)
    ovbp = overlap_by_partner(cams)

    # in-band membership per (record, partner) for thing-class detections
    by_ts_cam: dict = {}
    in_band: dict = {}
    for k, x in enumerate(records):
        cls = int(x["class_id"])
        if SUPERCLASS.get(cls) is None:
            continue
        cam, ts = int(x["camera_id"]), int(x["timestamp"])
        m = None
        for a, b in ADJACENT_PAIRS:
            if cam == a:
                partner = b
            elif cam == b:
                partner = a
            else:
                continue
            if m is None:
                m = decode_rle(mask_rle[k])
                h, w = cams[cam]["h"], cams[cam]["w"]
                if m.shape != (h, w):
                    m = m[:h, :w]
                m = m.astype(bool)
            if (m & ovbp[(cam, partner)]).any():
                in_band[(k, partner)] = True
                by_ts_cam.setdefault((ts, cam, partner), []).append(k)

    rows = []
    for a, b in ADJACENT_PAIRS:
        for (ts, cam, partner), ks_a in by_ts_cam.items():
            if cam != a or partner != b:
                continue
            ks_b = by_ts_cam.get((ts, b, a), [])
            for ka in ks_a:
                xa = records[ka]
                sca = SUPERCLASS[int(xa["class_id"])]
                for kb in ks_b:
                    xb = records[kb]
                    if SUPERCLASS[int(xb["class_id"])] != sca:
                        continue
                    gA, gB = int(xa["gt_global_id"]), int(xb["gt_global_id"])
                    if gA > 0 and gA == gB:
                        label = 1
                    elif gA > 0 and gB > 0:
                        label = 0
                    else:
                        label = -1
                    rows.append((ka, kb, a, b, int(xa["class_id"]), int(xb["class_id"]),
                                 gA, gB, float(xa["gt_iou"]), float(xb["gt_iou"]),
                                 float(xa["score"]), float(xb["score"]), label, ts))
    arr = np.array(rows, dtype=PAIR_DTYPE) if rows else np.zeros(0, dtype=PAIR_DTYPE)
    np.savez_compressed(out_path, pairs=arr)
    return {"pairs": len(arr), "pos": int((arr["label"] == 1).sum()),
            "neg": int((arr["label"] == 0).sum())}


def _worker(args):
    cache_npz, out_path = args
    try:
        return cache_npz.stem, extract_scene(cache_npz, out_path), None
    except Exception as e:                                    # noqa: BLE001
        return cache_npz.stem, None, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                    help="prediction cache of the training split")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke)")
    args = ap.parse_args()

    cache_dir, out_dir = args.cache_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(cache_dir.glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    jobs = [(p, out_dir / p.name) for p in paths]
    print(f"{len(jobs)} scenes -> {out_dir} (workers={args.workers})")

    t0 = time.time()
    tot = {"pairs": 0, "pos": 0, "neg": 0}
    done = 0
    if args.workers <= 1:
        results = (_worker(j) for j in jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        results = (f.result() for f in as_completed([ex.submit(_worker, j) for j in jobs]))
    for stem, stats, err in results:
        done += 1
        if err:
            print(f"  [{done}/{len(jobs)}] {stem[:34]} ERROR: {err}", flush=True)
            continue
        for k in tot:
            tot[k] += stats[k]
        if done % 50 == 0 or done == len(jobs):
            print(f"  [{done}/{len(jobs)}] pairs={tot['pairs']:8d} pos={tot['pos']:7d} "
                  f"neg={tot['neg']:8d}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s: {tot}")


if __name__ == "__main__":
    main()
