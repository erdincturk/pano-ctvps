#!/usr/bin/env python3
"""eval_linker_tables.py — the tracker alone and the full system, scored exactly from the overlap tables.

Per scene: the linker (panoctvps.linker.pipeline.linker_assignment with the first configuration of --config and
the adapted cosines of scripts/pipeline/build_cosines.py) on the track ids of the cache field `tracker_id_deployed`. The
predicted identity of a segment is then, as in scripts/pipeline/eval_linker_wstq.py:
  tracker only : tid + camera * 100000
  full system  : the global id of the linker, or camera * 100000 + tid + 2**40 for a track that is not linked
and both are scored from the overlap tables (scripts/experiments/build_overlap_tables.py). With the deployed configuration this
gives the tracker alone and the full system of Table 7 (wSTQ 0.3149 and 0.3228). The per-scene scores give Table S28
and the gain of the linker in Table S27. With --keep-per-gt the AQ term of every ground-truth instance of the full
system is stored (association quality per class, Table S18(c)).

Usage:
  python scripts/experiments/eval_linker_tables.py --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --output outputs/linker/linker_tables_deployed.json --keep-per-gt
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from panoctvps.evaluation import overlap_tables as aq_tables
from panoctvps.linker.overlap import calibration_from_arrays
from panoctvps.linker.pipeline import linker_assignment

REPO = Path(__file__).resolve().parents[2]
UNMATCHED_OFFSET = 1 << 40


def scene(job):
    tp, cache_dir, cos_dir, config, tid_field = job
    tab = dict(np.load(tp, allow_pickle=True))
    npz = np.load(Path(cache_dir) / Path(tp).name, allow_pickle=True)
    configs = list(json.loads(Path(config).read_text()).items())[:1]
    wh = {}
    for fr in npz["frame_records"]:
        wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
    cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)
    t0 = time.time()
    assign = linker_assignment(npz, cams, configs, cosines_path=Path(cos_dir) / Path(tp).name,
                               tid_field=tid_field)[configs[0][0]]
    t_link = time.time() - t0
    rec = npz["records"]
    trk, full = {}, {}
    for f_row, (fi, cam, ts) in enumerate(tab["frames"].tolist()):
        fr = npz["frame_records"][fi]
        rs, re = int(fr["records_start"]), int(fr["records_end"])
        for k in range(re - rs):
            tid = int(rec[rs + k][tid_field])
            if tid <= 0:
                continue
            trk[(f_row, k + 1)] = tid + cam * 100_000
            gid = assign.get((cam, tid))
            full[(f_row, k + 1)] = gid if gid is not None else cam * 100_000 + tid + UNMATCHED_OFFSET

    def fn(m):
        def ident(f, k, s):
            return np.array([m.get(key, 0) for key in zip(f.tolist(), k.tolist())], dtype=np.int64)
        return ident

    cam_ids = {key: v % 100_000 if v < UNMATCHED_OFFSET else v for key, v in trk.items()}
    o_trk = aq_tables.score_scene(tab, fn(trk), fn(cam_ids))
    o_full = aq_tables.score_scene(tab, fn(full), fn(cam_ids))
    per_gt_full = {str(g): v for g, v in o_full["agg"]["per_gt"].items()}
    for o in (o_trk, o_full):
        o["agg"].pop("per_gt", None)
    n_links = len({v for v in assign.values()}) if assign else 0
    return str(tab["segment_name"]), o_trk, o_full, {"linker_seconds": round(t_link, 1), "n_tracks_assigned": len(assign),
                                                     "n_global_ids": n_links}, per_gt_full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/tracker_ctvis_val_cache"))
    ap.add_argument("--cosines-dir", type=Path, default=Path("outputs/linker/adapted_cosines"))
    ap.add_argument("--config", type=Path, default=REPO / "configs/linker_deployed.json")
    ap.add_argument("--scenes", default="", help="Comma-separated scene names (default: all scenes of the tables).")
    ap.add_argument("--tid-field", default="tracker_id_deployed")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-per-gt", action="store_true", help="store the AQ term of every object (full system)")
    args = ap.parse_args()
    t0 = time.time()
    want = set(x for x in args.scenes.split(",") if x)
    jobs = [(str(tp), str(args.cache_dir), str(args.cosines_dir), str(args.config), args.tid_field)
            for tp in sorted(args.tables_dir.glob("*.npz")) if not want or tp.stem in want]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(scene, jobs))
    out = {"inputs": {k: str(v) for k, v in vars(args).items()}, "n_scenes": len(res)}
    for key, idx in (("tracker_only", 1), ("full_system", 2)):
        agg, per_scene, _, per_camera = aq_tables.merge([(r[0], r[idx]) for r in res])
        out[key] = {"aggregate": agg, "per_scene": per_scene}
    out["per_scene_linker"] = {r[0]: r[3] for r in res}
    if args.keep_per_gt:
        out["full_system_per_gt"] = {r[0]: r[4] for r in res}
    out["seconds"] = round(time.time() - t0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float))
    for key in ("tracker_only", "full_system"):
        a = out[key]["aggregate"]
        print(f"{key}: wSTQ {a['wSTQ']:.5f} wAQ {a['wAQ']:.5f} wSQ {a['wSQ']:.5f}")


if __name__ == "__main__":
    main()
