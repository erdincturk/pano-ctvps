#!/usr/bin/env python3
"""merge_tracker_cells.py — merge per-scene runs of scripts/experiments/run_tracker_cells.py into one result per cell (exact; Tables 6, S9 to S14).

scripts/experiments/run_tracker_cells.py can run each scene as its own process (--scenes <name> --out-dir <parts>/<name>). This
script reads the track ids of all parts and the overlap tables, scores each cell over all scenes (the same functions
as a single run over all scenes), and writes <out-dir>/<cell>.json and <out-dir>/<cell>_ids.npz.

Usage:
  python scripts/experiments/merge_tracker_cells.py --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --parts-dir outputs/tracker_cells/parts --cells configs/tracker_cells/ctvis_grid.json \
      --out-dir outputs/tracker_cells/tracker_ctvis_val
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from panoctvps.evaluation.tracker_cells import fill_defaults

import importlib.util
_spec = importlib.util.spec_from_file_location("run_tracker_cells", Path(__file__).resolve().parent / "run_tracker_cells.py")
_rtc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rtc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", type=Path, required=True)
    ap.add_argument("--parts-dir", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()
    cells = fill_defaults(json.loads(args.cells.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tabs = []
    for tp in sorted(args.tables_dir.glob("*.npz")):
        tab = dict(np.load(tp, allow_pickle=True))
        tabs.append((str(tab["segment_name"]), tab))
    names = [n for n, _ in tabs]
    for c in cells:
        idmaps = []
        for name in names:
            p = np.load(args.parts_dir / name / f"{c['name']}_ids.npz", allow_pickle=True)
            assert list(p["scenes"]) == [name], (c["name"], name, p["scenes"])
            idmaps.append({(int(r["f"]), int(r["k"]), int(r["s"])): int(r["tid"]) for r in p["ids"]})
        cfg = json.loads((args.parts_dir / names[0] / f"{c['name']}.json").read_text())["tracker_config"]
        _rtc.write_cell(args.out_dir, cfg, args.cache_dir, tabs, idmaps, time.time() - t0,
                        extra={"merged_from": str(args.parts_dir)})


if __name__ == "__main__":
    main()
