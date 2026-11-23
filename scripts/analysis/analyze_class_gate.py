#!/usr/bin/env python3
"""analyze_class_gate.py — matches that continue a track of another class without the class gate (supplement,
Table S14, note on the row "Without the class gate").

Inputs: the track ids of two cells of scripts/experiments/run_tracker_cells.py on the deployed cache: the deployed tracker
(configs/tracker_cells/full_tracker.json, cell ours_full_tracker_deployed) and the same tracker without the class
gate (configs/tracker_cells/tracker_components.json, cell comp_no_class_gate); the cache and its overlap tables
(scripts/experiments/build_overlap_tables.py) for the frames and the ground-truth instance of every detection.

A match is a detection that continues an existing track: within a camera, the detections of a track are taken in
frame order and every detection after the first is a match. The class of the track at a match is the class of its
previous detection (the tracker stores the class of the last detection). A match continues a track of another class
if the two classes differ. Each such match is split by instance: both detections belong to the same tracked
instance, to different instances, or one of them to none. A detection belongs to the instance that covers most of
its pixels (cache field gt_global_id) if its IoU with it (gt_iou) is at least 0.5, else to none.

Check: the deployed run (class gate on) has no such match.

Usage:
  python scripts/analysis/analyze_class_gate.py --cache-dir outputs/tracker_ctvis_val_cache \
      --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --cells-dir outputs/tracker_cells/tracker_ctvis_val --output outputs/analyses/class_gate.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from panoctvps.data.constants import CLASS_NAMES

NAMES = dict(enumerate(CLASS_NAMES)) if isinstance(CLASS_NAMES, (list, tuple)) else dict(CLASS_NAMES)


def obj_of(rec, rs, k):
    r = rec[rs + k - 1]
    g = int(r["gt_global_id"])
    return g if g > 0 and float(r["gt_iou"]) >= 0.5 else 0


def load_ids(path):
    p = np.load(path, allow_pickle=True)
    names = [str(n) for n in p["scenes"]]
    out = {n: {} for n in names}
    for r in p["ids"]:
        out[names[int(r["scene"])]][(int(r["f"]), int(r["k"]), int(r["s"]))] = int(r["tid"])
    return out


def cross_class(ids_by_scene, cache_dir, tables_dir):
    C = collections.Counter()
    pc = collections.Counter()
    for name, ids in sorted(ids_by_scene.items()):
        tab = np.load(Path(tables_dir) / f"{name}.npz", allow_pickle=True)
        npz = np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=True)
        rec, frs = npz["records"], npz["frame_records"]
        fr_list = tab["frames"].tolist()
        prev = {}
        for (f, k, s), tid in sorted(ids.items()):
            fi, cam, ts = fr_list[f]
            o = obj_of(rec, int(frs[fi]["records_start"]), k)
            key = (cam, tid)
            if key in prev:
                ps, po, pf = prev[key]
                assert pf < f
                C["matches"] += 1
                if ps != s:
                    C["cross_class"] += 1
                    pc[(NAMES[s], NAMES[ps])] += 1
                    if o > 0 and o == po:
                        C["cross_class_same_instance"] += 1
                    elif o > 0 and po > 0:
                        C["cross_class_different_instances"] += 1
                    else:
                        C["cross_class_one_side_no_instance"] += 1
            prev[key] = (s, o, f)
    return dict(C), {f"{a}|{b}": v for (a, b), v in pc.most_common()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/tracker_ctvis_val_cache"))
    ap.add_argument("--tables-dir", type=Path, required=True)
    ap.add_argument("--cells-dir", type=Path, required=True, help="Output folder of scripts/experiments/run_tracker_cells.py.")
    ap.add_argument("--deployed-cell", default="ours_full_tracker_deployed")
    ap.add_argument("--no-class-gate-cell", default="comp_no_class_gate")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    ncg, cls_pairs = cross_class(load_ids(a.cells_dir / f"{a.no_class_gate_cell}_ids.npz"), a.cache_dir, a.tables_dir)
    dep, _ = cross_class(load_ids(a.cells_dir / f"{a.deployed_cell}_ids.npz"), a.cache_dir, a.tables_dir)
    out = {"definitions": __doc__, "without_class_gate": ncg, "class_pairs_detection_track": cls_pairs,
           "check_deployed_cross_class_matches": dep.get("cross_class", 0), "deployed_matches": dep["matches"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=1))


if __name__ == "__main__":
    main()
