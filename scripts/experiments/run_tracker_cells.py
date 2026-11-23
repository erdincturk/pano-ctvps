#!/usr/bin/env python3
"""run_tracker_cells.py — tracker runs ("cells") on a cache, scored exactly from its overlap tables.

Each cell of --cells (JSON list, see configs/tracker_cells/) names a tracker and its settings: our tracker
("memorybank": thresholds, motion gate bounds, component switches), MinVIS or the IoU tracker. For every scene the
trackers run per camera over the evaluated frames; the scores are rebuilt from the overlap tables
(scripts/experiments/build_overlap_tables.py). Each cell writes <out-dir>/<name>.json (aggregate, per camera, per scene, per
scene and camera) and <out-dir>/<name>_ids.npz (the track id of every predicted thing segment). Serves Table 6
of the paper and the supplement, Tables S9 to S12 (grids and motion gate bounds) and S14 (tracker components), and
gives the inputs of the held-out check (Table S13). Scenes can be run as separate processes (--scenes) and merged with scripts/experiments/merge_tracker_cells.py.

Usage:
  python scripts/experiments/run_tracker_cells.py --cache-dir outputs/tracker_ctvis_val_cache \
      --tables-dir outputs/overlap_tables/tracker_ctvis_val --cells configs/tracker_cells/ctvis_grid.json \
      --out-dir outputs/tracker_cells/tracker_ctvis_val
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from panoctvps.evaluation import overlap_tables
from panoctvps.evaluation.tracker_cells import fill_defaults, run_scene, score_cell


def write_cell(out_dir: Path, c: dict, cache_dir, tabs, idmaps, seconds: float, extra: dict | None = None) -> dict:
    sc = score_cell(tabs, idmaps)
    agg, per_scene, per_cam_rows, per_camera = overlap_tables.merge(sc)
    per_gt = {name: {str(g): v for g, v in o["agg"]["per_gt"].items()} for name, o in sc}
    out = {"cache_dir": str(cache_dir), "tracker_config": c, "n_sequences": len(per_scene), **(extra or {}),
           "timing_seconds": round(seconds, 1), "aggregate": agg, "per_camera": per_camera,
           "per_sequence": [{"segment_name": r["scene"], "wSTQ": r["wSTQ"], "wAQ": r["wAQ"], "wSQ": r["wSQ"]}
                            for r in per_scene],
           "per_scene": per_scene, "per_scene_camera": per_cam_rows}
    (out_dir / f"{c['name']}.json").write_text(json.dumps(out, indent=1, default=float))
    rows = [(si, f, k, s, t) for si, idm in enumerate(idmaps) for (f, k, s), t in idm.items()]
    np.savez_compressed(out_dir / f"{c['name']}_ids.npz", scenes=np.array([n for n, _ in tabs]),
                        ids=np.array(rows, dtype=[("scene", "i2"), ("f", "i4"), ("k", "i4"), ("s", "i2"),
                                                  ("tid", "i8")]),
                        per_gt=json.dumps(per_gt))
    print(f"CELL DONE {c['name']}: wSTQ {agg['wSTQ']:.5f} wAQ {agg['wAQ']:.5f} wSQ {agg['wSQ']:.5f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--tables-dir", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True, help="JSON list of cells")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scenes", default="", help="Comma-separated scene names (default: all scenes of the tables).")
    args = ap.parse_args()
    cells = fill_defaults(json.loads(args.cells.read_text()))
    want = set(s for s in args.scenes.split(",") if s)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tabs, ids_all = [], {c["name"]: [] for c in cells}
    t0 = time.time()
    for tp in sorted(args.tables_dir.glob("*.npz")):
        tab = dict(np.load(tp, allow_pickle=True))
        name = str(tab["segment_name"])
        if want and name not in want:
            continue
        npz = np.load(args.cache_dir / tp.name, allow_pickle=True)
        assert str(npz["segment_name"]) == name
        ids = run_scene(npz, tab, cells)
        tabs.append((name, tab))
        for c in cells:
            ids_all[c["name"]].append(ids[c["name"]])
        print(f"[{len(tabs)}] {name[:34]} ({time.time()-t0:.0f}s)", flush=True)
    for c in cells:
        write_cell(args.out_dir, c, args.cache_dir, tabs, ids_all[c["name"]], time.time() - t0)


if __name__ == "__main__":
    main()
