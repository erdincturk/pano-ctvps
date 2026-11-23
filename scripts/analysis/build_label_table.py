#!/usr/bin/env python3
"""build_label_table.py — per-class statistics of the ground-truth thing masks (training or validation split).
Its per-mask table is the input of scripts/analysis/analyze_dataset.py (Table 2 of the paper; supplement, Tables S1, S3 and
S5) and of the ground-truth columns of scripts/analysis/analyze_age_gap.py (Tables S4 and S15).

Scene list and frames from configs/splits/{split}_frames.txt. For every listed camera image,
decodes the panoptic label (decode_png + the panoptic divisor) and counts the pixels of every thing instance
(semantic class in {car, truck, bus, other_large_vehicle, trailer, pedestrian, cyclist, motorcyclist}, instance
id > 0). An instance is tracked when the label's instance-to-global mapping marks it is_tracked with a global id > 0.

Per thing class:
- tracked instances: unique (scene, global id) among tracked masks; class = majority class of its masks
- masks: all thing masks; tracked masks: masks of tracked instances
- median mask area (pixels) and share of masks under 32 x 32 = 1,024 pixels, over tracked masks and over all masks
- mean number of annotated frames per object: mean over tracked instances of their distinct timestamps
Writes a per-mask table (npz) and the summary (json).

Usage:
  python scripts/analysis/build_label_table.py --split validation --output-dir outputs/analyses/labels
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from panoctvps.data.constants import PARQUET_COLS
from panoctvps.data.decode import decode_png

REPO = Path(__file__).resolve().parents[2]
THING = {2: "car", 3: "truck", 4: "bus", 5: "other_large_vehicle", 8: "trailer", 9: "pedestrian",
         10: "cyclist", 11: "motorcyclist"}
REC = np.dtype([("scene", "i4"), ("ts", "i8"), ("cam", "i1"), ("cls", "i1"), ("inst", "i4"), ("gid", "i8"),
                ("tracked", "?"), ("px", "i4")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, default=REPO.parent / "data/waymo_open_dataset_v_2_0_1")
    ap.add_argument("--split", choices=("training", "validation"), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    frames = collections.defaultdict(set)
    for line in (REPO / "configs/splits" / f"{args.split}_frames.txt").read_text().split():
        if "," in line and not line.startswith("segment"):
            s, t = line.split(",")[:2]
            frames[s].add(int(t))
    scenes = sorted(frames)
    if args.limit:
        scenes = scenes[:args.limit]
    C = {k: PARQUET_COLS[k] for k in ("seg_key_context", "seg_key_timestamp", "seg_key_camera", "seg_panoptic_label",
                                       "seg_divisor", "seg_local_ids", "seg_global_ids", "seg_is_tracked")}
    rows, n_img = [], 0
    t0 = time.time()
    for si, s in enumerate(scenes):
        path = args.dataset_root / args.split / "camera_segmentation" / f"{s}.parquet"
        df = pq.read_table(path, columns=list(C.values())).to_pandas()
        df = df[df[C["seg_key_timestamp"]].isin(frames[s])]
        for _, r in df.iterrows():
            pan = decode_png(r[C["seg_panoptic_label"]], np.uint16)
            div = int(r[C["seg_divisor"]])
            vals, cnts = np.unique(pan, return_counts=True)
            loc = r[C["seg_local_ids"]]
            mp = {}
            if loc is not None:
                for l, g, k in zip(loc, r[C["seg_global_ids"]], r[C["seg_is_tracked"]]):
                    mp[int(l)] = (int(g), bool(k))
            for v, c in zip(vals.tolist(), cnts.tolist()):
                sem, inst = divmod(int(v), div)
                if sem in THING and inst > 0:
                    g, k = mp.get(inst, (0, False))
                    rows.append((si, int(r[C["seg_key_timestamp"]]), int(r[C["seg_key_camera"]]), sem, inst, g,
                                 bool(k and g > 0), c))
            n_img += 1
        if (si + 1) % 25 == 0 or si + 1 == len(scenes):
            print(f"[{si+1}/{len(scenes)}] images={n_img} masks={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    a = np.array(rows, dtype=REC)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.split}_thing_masks.npz", masks=a,
                        scenes=np.array(scenes))
    # instances
    inst_cls = collections.defaultdict(collections.Counter)
    inst_ts = collections.defaultdict(set)
    tr = a[a["tracked"]]
    for r in tr:
        key = (int(r["scene"]), int(r["gid"]))
        inst_cls[key][int(r["cls"])] += 1
        inst_ts[key].add(int(r["ts"]))
    per = {}
    for c, name in THING.items():
        allm = a[a["cls"] == c]
        trm = tr[tr["cls"] == c]
        keys = [k for k, cc in inst_cls.items() if cc.most_common(1)[0][0] == c]
        per[name] = {
            "tracked_instances": len(keys),
            "masks": int(len(allm)), "tracked_masks": int(len(trm)),
            "median_area_tracked": float(np.median(trm["px"])) if len(trm) else None,
            "median_area_all": float(np.median(allm["px"])) if len(allm) else None,
            "share_under_1024px_tracked": round(float((trm["px"] < 1024).mean()), 4) if len(trm) else None,
            "share_under_1024px_all": round(float((allm["px"] < 1024).mean()), 4) if len(allm) else None,
            "mean_annotated_frames_per_object": round(float(np.mean([len(inst_ts[k]) for k in keys])), 4) if keys else None,
        }
    out = {"definitions": __doc__, "split": args.split, "scenes": len(scenes), "images": n_img,
           "total_tracked_instances": len(inst_cls), "total_masks": int(len(a)), "total_tracked_masks": int(len(tr)),
           "per_class": per, "seconds": round(time.time() - t0, 1)}
    (args.output_dir / f"{args.split}_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("scenes", "images", "total_tracked_instances", "total_masks",
                                          "total_tracked_masks")}))
    for n, v in per.items():
        print(n, v)


if __name__ == "__main__":
    main()
