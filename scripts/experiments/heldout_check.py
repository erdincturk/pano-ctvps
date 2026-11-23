#!/usr/bin/env python3
"""heldout_check.py — held-out check (leave one scene out) of the tuned rows of Table 6 (supplement, Table S13).

Rule: in each of the 20 rounds, the setting with the highest pooled wAQ on the other 19 scenes is selected and scored
on the held-out scene; the 20 held-out scores are pooled as the evaluator pools scenes (sum of the per-scene AQ
numerators / sum of the ground-truth instances). Within one cache wSQ is the same for every cell, so selecting by
wAQ is selecting by wSTQ, and the pooled held-out wSTQ is sqrt(held-out wAQ * wSQ). The numerator of a scene is its
wAQ times its number of ground-truth instances (--counts-from: any cell JSON of scripts/experiments/run_tracker_cells.py, which
stores the counts per scene). Check: every cell's pooled wAQ, rebuilt from its per-scene values, equals its stored
wAQ.

A row (--row "label|glob|table cell") is the grid of one tracker on one cache: the cell JSONs matched by the glob
(scripts/experiments/run_tracker_cells.py, or scripts/pipeline/eval_tracker_cached.py output with per_sequence), named by their file stem;
the table cell is the stem of the setting printed in Table 6 (empty: the best cell on all 20 scenes).

Full tracker (--full-tracker): in each round the thresholds are selected as in the row named by --full-tracker; at
the selected thresholds the motion gate bound of each camera is selected from a gate sweep (--sweep
"<threshold cell stem>=<path prefix of the sweep cells>", bounds 200 to 1,200 px) by the pooled AQ of that camera on
the other 19 scenes (each camera's bound is read from its own STQ; the SQ of a camera is fixed). The held-out scene
is then scored exactly: the track ids of each camera come from the sweep cell with that camera's bound (the tracker
handles each camera alone), scored with the overlap tables.

Usage:
  python scripts/experiments/heldout_check.py --counts-from outputs/tracker_cells/tracker_ctvis_val/ours_m0.70_i0.01_d5.json \
      --row "CTVIS (retuned thresholds)|outputs/tracker_cells/tracker_ctvis_val/ours_m*.json|ours_m0.70_i0.01_d5" \
      --row "MinVIS|outputs/tracker_cells/tracker_ctvis_val/minvis_m*.json|minvis_m0.90_i0.01_d5" \
      --row "IoU tracker|outputs/tracker_cells/tracker_ctvis_val/iou_t*.json|iou_t0.30_d5" \
      --full-tracker "CTVIS (retuned thresholds)" --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --sweep "ours_m0.70_i0.01_d5=outputs/tracker_cells/tracker_ctvis_val/ours_m0.70_i0.01_d5_gate" \
      --sweep "ours_m0.60_i0.01_d5=outputs/tracker_cells/tracker_ctvis_val/ours_m0.60_i0.01_d5_gate" \
      --deployed outputs/tracker_cells/tracker_ctvis_val/ours_full_tracker_deployed.json \
      --output outputs/heldout/heldout.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from panoctvps.evaluation import overlap_tables
from panoctvps.evaluation.tracker_cells import score_cell

TOL = 1e-12
BOUNDS = list(range(200, 1201, 100))
CAMS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]


def jl(p):
    return json.loads(Path(p).read_text())


def cell_entry(res, n_gt):
    agg = res["aggregate"]
    ps = {r["segment_name"]: r["wAQ"] for r in res["per_sequence"]}
    assert set(ps) == set(n_gt), "scene sets differ"
    num = {s: ps[s] * n_gt[s] for s in ps}
    rebuilt = sum(num.values()) / sum(n_gt.values())
    assert abs(rebuilt - agg["wAQ"]) < 1e-9, (rebuilt, agg["wAQ"])
    return {"num": num, "wAQ": agg["wAQ"], "wSTQ": agg["wSTQ"], "wSQ": agg["wSQ"], "rebuild_diff": abs(rebuilt - agg["wAQ"])}


def loso(entries, scenes):
    sel, ho = {}, {}
    for r in scenes:
        best, bval, ties = None, -1.0, []
        for k, e in entries.items():
            v = sum(e["num"][s] for s in scenes if s != r)
            if v > bval + TOL:
                best, bval, ties = k, v, []
            elif abs(v - bval) <= TOL:
                ties.append(k)
        sel[r] = (best, ties)
        ho[r] = entries[best]["num"][r]
    return sel, ho


def row_threshold(label, files, table_cell, n_gt):
    entries = {Path(p).stem: cell_entry(jl(p), n_gt) for p in files}
    scenes = sorted(n_gt)
    best_all = max(entries, key=lambda k: entries[k]["wAQ"])
    table_cell = table_cell or best_all
    assert table_cell in entries, (label, table_cell, sorted(entries))
    sel, ho = loso(entries, scenes)
    N = sum(n_gt.values())
    t = entries[table_cell]
    ho_waq = sum(ho.values()) / N
    return {"label": label, "n_cells": len(entries), "table_cell": table_cell, "best_on_all_20": best_all,
            "table_wSTQ": t["wSTQ"], "table_wAQ": t["wAQ"], "wSQ": t["wSQ"], "heldout_wAQ": ho_waq,
            "heldout_wSTQ": math.sqrt(ho_waq * t["wSQ"]),
            "rounds_selecting_table": sum(1 for r in scenes if sel[r][0] == table_cell),
            "selections": {r: sel[r][0] for r in scenes}, "ties": {r: sel[r][1] for r in scenes if sel[r][1]},
            "heldout_per_scene_wAQ": {r: ho[r] / n_gt[r] for r in scenes},
            "table_per_scene_wAQ": {r: t["num"][r] / n_gt[r] for r in scenes},
            "n_gt": n_gt, "max_rebuild_diff": max(e["rebuild_diff"] for e in entries.values())}


def load_ids(path, names):
    p = np.load(path, allow_pickle=True)
    assert list(p["scenes"]) == names
    out = [dict() for _ in names]
    for r in p["ids"]:
        out[int(r["scene"])][(int(r["f"]), int(r["k"]), int(r["s"]))] = int(r["tid"])
    return out


def full_tracker(thr_row, sweeps, tables_dir, deployed_bounds, deployed_json):
    needed = set(thr_row["selections"].values())
    missing = [c for c in needed if c not in sweeps]
    assert not missing, f"no gate sweep given for {missing}"
    tabs = []
    for tp in sorted(Path(tables_dir).glob("*.npz")):
        tab = dict(np.load(tp, allow_pickle=True))
        tabs.append((str(tab["segment_name"]), tab))
    names = [n for n, _ in tabs]
    scenes = sorted(names)
    pc, ids = {}, {}
    for c in needed:
        for b in BOUNDS:
            res = jl(f"{sweeps[c]}{b}.json")
            pc[(c, b)] = {(x["scene"], x["camera"]): (x["aq_numerator"], x["num_gt_instances"])
                          for x in res["per_scene_camera"]}
            ids[(c, b)] = load_ids(f"{sweeps[c]}{b}_ids.npz", names)

    def select_bounds(c, rest):
        out = {}
        for cam in CAMS:
            best, bv, ties = None, -1.0, []
            for b in BOUNDS:
                v = sum(pc[(c, b)][(s, cam)][0] for s in rest if (s, cam) in pc[(c, b)])
                if v > bv + TOL:
                    best, bv, ties = b, v, []
                elif abs(v - bv) <= TOL:
                    ties.append(b)
            out[cam] = best
            if ties:
                out[cam + "_ties"] = ties
        return out

    def combined_score(c, bounds, scene_list):
        sel = []
        for name, tab in tabs:
            if name not in scene_list:
                continue
            si = names.index(name)
            cam_of = tab["frames"]["cam"]
            idm = {}
            for cam_i, cam in enumerate(CAMS, start=1):
                for key, tid in ids[(c, bounds[cam])][si].items():
                    if int(cam_of[key[0]]) == cam_i:
                        idm[key] = tid
            sel.append(((name, tab), idm))
        sc = score_cell([t for t, _ in sel], [m for _, m in sel])
        return {n: (o["agg"]["aq_num"], o["agg"]["N"]) for n, o in sc}, sc

    tc = thr_row["table_cell"]
    out = {"label": "full tracker", "table_cell": f"{tc}, bounds {deployed_bounds}"}
    if tc in needed:
        dep, sc = combined_score(tc, deployed_bounds, set(names))
        agg = overlap_tables.merge(sc)[0]
        out.update(table_wSTQ=agg["wSTQ"], table_wAQ=agg["wAQ"], wSQ=agg["wSQ"])
        if deployed_json:
            rec = jl(deployed_json)["aggregate"]
            out["check_deployed_from_sweep_ids"] = max(abs(agg[k] - rec[k]) for k in ("wSTQ", "wAQ", "wSQ"))
        out["bounds_selected_on_all_20"] = select_bounds(tc, scenes)
    rounds, ho = {}, {}
    for r in scenes:
        c = thr_row["selections"][r]
        bounds = select_bounds(c, [s for s in scenes if s != r])
        rounds[r] = {"thresholds": c, "bounds": bounds}
        ho[r] = combined_score(c, bounds, {r})[0][r]
    N = sum(n for _, n in ho.values())
    ho_waq = sum(a for a, _ in ho.values()) / N
    out.update(heldout_wAQ=ho_waq, heldout_wSTQ=math.sqrt(ho_waq * out.get("wSQ", thr_row["wSQ"])),
               rounds_selecting_table_thresholds=sum(1 for r in scenes if rounds[r]["thresholds"] == tc),
               rounds_selecting_table=sum(1 for r in scenes if rounds[r]["thresholds"] == tc and all(
                   rounds[r]["bounds"][k] == v for k, v in deployed_bounds.items())),
               rounds_selecting_deployed_bound_per_camera={cam: sum(
                   1 for r in scenes if rounds[r]["bounds"][cam] == deployed_bounds[cam]) for cam in CAMS},
               rounds=rounds, heldout_per_scene_wAQ={r: a / n for r, (a, n) in ho.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-from", type=Path, required=True,
                    help="A cell JSON of scripts/experiments/run_tracker_cells.py (ground-truth instances per scene).")
    ap.add_argument("--row", action="append", default=[], help='"label|glob of cell JSONs|table cell stem"')
    ap.add_argument("--full-tracker", default="", help="Label of the row whose thresholds the full tracker uses.")
    ap.add_argument("--tables-dir", type=Path, default=None)
    ap.add_argument("--sweep", action="append", default=[], help='"<threshold cell stem>=<path prefix>"')
    ap.add_argument("--deployed-bounds", default="1:600,2:800,3:800,4:1000,5:1000")
    ap.add_argument("--deployed", type=Path, default=None, help="Cell JSON of the deployed full tracker (check).")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    n_gt = {r["scene"]: r["num_gt_instances"] for r in jl(args.counts_from)["per_scene"]}
    out = {}
    for spec in args.row:
        label, pattern, table = (spec.split("|") + ["", "", ""])[:3]
        files = sorted(glob.glob(pattern))
        assert files, f"no cell matches {pattern}"
        res = row_threshold(label, files, table, n_gt)
        out[label] = res
        print(f"{label}: table {res['table_cell']} {res['table_wSTQ']:.5f}/{res['table_wAQ']:.5f} | held-out "
              f"{res['heldout_wSTQ']:.5f}/{res['heldout_wAQ']:.5f} | rounds {res['rounds_selecting_table']}/20 | "
              f"cells {res['n_cells']} | ties {len(res['ties'])}")
    if args.full_tracker:
        sweeps = dict(s.split("=", 1) for s in args.sweep)
        bounds = {CAMS[int(k) - 1]: int(v) for k, v in (x.split(":") for x in args.deployed_bounds.split(","))}
        res = full_tracker(out[args.full_tracker], sweeps, args.tables_dir, bounds, args.deployed)
        out["full tracker"] = res
        print(f"full tracker: held-out {res['heldout_wSTQ']:.5f}/{res['heldout_wAQ']:.5f} | rounds thresholds "
              f"{res['rounds_selecting_table_thresholds']}/20, all {res['rounds_selecting_table']}/20 | check "
              f"{res.get('check_deployed_from_sweep_ids')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
