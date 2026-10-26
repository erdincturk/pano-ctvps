#!/usr/bin/env python3
"""run_online_linker.py — the causal linker on the validation split (Table S6).

Stages (CPU only, from the cached validation predictions):
  build   per scene (parallel): pair table (geometry once) and frame tables
          (STQ confusion rows keyed by the tracker-only id, STQuality of the
          tracker alone and of the per-scene linker), cached as pickles under
          <out-dir>/cache/.
  sweep   the causal linker for k in {1, 2, 3, 5, 8}: wSTQ / wAQ / wSQ, link
          counts, timing -> <out-dir>/results.json and <out-dir>/REPORT.md.

The causal rule is described in scripts/online_linker.py.

Usage (from the repository root):
  python scripts/run_online_linker.py --stage build --workers 8
  python scripts/run_online_linker.py --stage sweep
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import online_linker as ol  # noqa: E402
from panoctvps.linker.link_eval import link_counts  # noqa: E402

KS = [1, 2, 3, 5, 8]


def _scene_names(cache_dir: Path):
    return [p.stem for p in sorted(cache_dir.glob("*.npz"))]


def _build_one(scene: str, cache_dir: Path, adapter: Path, cos_dir: Path, dataset_root: str,
               out_cache: Path, cfg: dict) -> str:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out_pt = out_cache / f"{scene}.pairs.pkl"
    out_ft = out_cache / f"{scene}.frames.pkl"
    t0 = time.time()
    if not out_pt.exists():
        pt = ol.build_pair_table(cache_dir / f"{scene}.npz", adapter, cos_dir / f"{scene}.npz", cfg)
        with open(out_pt, "wb") as f:
            pickle.dump(pt, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(out_pt, "rb") as f:
            pt = pickle.load(f)
    if not out_ft.exists():
        ft = ol.build_frame_tables(cache_dir / f"{scene}.npz", dataset_root, pt["deployed_assign"])
        with open(out_ft, "wb") as f:
            pickle.dump(ft, f, protocol=pickle.HIGHEST_PROTOCOL)
    return f"{scene[:24]} pairs={len(pt['pairs'])} {time.time()-t0:.0f}s"


def stage_build(args, cfg: dict) -> None:
    import multiprocessing as mp
    out_cache = args.out_dir / "cache"
    out_cache.mkdir(parents=True, exist_ok=True)
    scenes = _scene_names(args.cache_dir)
    # largest scenes first for load balance
    sizes = {s: (args.cache_dir / f"{s}.npz").stat().st_size for s in scenes}
    scenes.sort(key=lambda s: -sizes[s])
    fn = partial(_build_one, cache_dir=args.cache_dir, adapter=args.adapter, cos_dir=args.cosines_dir,
                 dataset_root=args.dataset_root, out_cache=out_cache, cfg=cfg)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for msg in pool.imap_unordered(fn, scenes):
            print(f"  {msg}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"build done in {time.time()-t0:.0f}s")


def _links_row(lt: dict) -> dict:
    return {k: lt[k] for k in ("true", "correct", "wrong", "missed", "precision", "recall", "per_pair")}


def stage_sweep(args) -> None:
    out_cache = args.out_dir / "cache"
    scenes = _scene_names(args.cache_dir)
    pts, fts = [], []
    for s in scenes:
        with open(out_cache / f"{s}.pairs.pkl", "rb") as f:
            pts.append(pickle.load(f))
        with open(out_cache / f"{s}.frames.pkl", "rb") as f:
            fts.append(pickle.load(f))
    structs = [pt["struct"] for pt in pts]
    print(f"loaded {len(pts)} scenes")

    # ---- tracker only and per-scene linker: STQuality and the fast evaluator ----
    floor_real = ol.aggregate(fts, [(ft["floor"]["AQ"], ft["floor"]["n_gt"]) for ft in fts])
    dep_real = ol.aggregate(fts, [(ft["deployed"]["AQ"], ft["deployed"]["n_gt"]) for ft in fts])
    floor_fast = ol.aggregate(fts, [ol.scene_aq(ft, lambda ts, cam, tid: cam * ol.CAM_NS + tid) for ft in fts])
    dep_fast = ol.aggregate(fts, [ol.scene_aq(ft, lambda ts, cam, tid, d=pt["deployed_assign"]: d[(cam, tid)])
                                  for pt, ft in zip(pts, fts)])
    for name, a, b in (("tracker only", floor_real, floor_fast), ("per-scene linker", dep_real, dep_fast)):
        for m in ("wSTQ", "wAQ", "wSQ"):
            assert abs(a[m] - b[m]) < 1e-9, f"fast evaluator differs from STQuality on {name} {m}"
    print("tracker only     ", floor_real)
    print("per-scene linker ", dep_real)

    dep_links = link_counts(structs, [pt["deployed_assign"] for pt in pts])
    merged = [(pt, pair) for pt in pts for pair in pt["deployed_merged"]]
    n_single = sum(1 for pt, pair in merged if pt["scored"][pair]["n_scored"] == 1)
    results = {
        "n_scenes": len(pts),
        "tracker_only": floor_real,
        "per_scene_linker": {**dep_real, "links": _links_row(dep_links),
                             "merged_links": len(merged),
                             "merged_links_single_scored_frame": n_single},
        "causal": {}}
    print(f"per-scene linker: correct={dep_links['correct']} wrong={dep_links['wrong']} "
          f"missed={dep_links['missed']}; merges {len(merged)}, of which {n_single} on one scored frame")

    # ---- the causal linker ----
    step_k1 = None
    for k in KS:
        sims = [ol.simulate_scene(pt, k, args.cfg) for pt in pts]
        links = link_counts(structs, [sim["final"] for sim in sims])
        aqs = [ol.scene_aq(ft, lambda ts, cam, tid, em=sim["emitted"]: em[(ts, cam, tid)])
               for sim, ft in zip(sims, fts)]
        agg = ol.aggregate(fts, aqs)
        step = np.concatenate([sim["step_times"] for sim in sims])
        if k == 1:
            step_k1 = step
        results["causal"][f"k{k}"] = {"k": k, **agg, "links": _links_row(links),
                                      "decision_step_ms_per_frame": float(step.mean() * 1000)}
        print(f"  k={k} wSTQ={agg['wSTQ']:.5f} wAQ={agg['wAQ']:.5f} wSQ={agg['wSQ']:.5f} "
              f"correct={links['correct']} wrong={links['wrong']} missed={links['missed']} "
              f"step={step.mean()*1000:.3f} ms", flush=True)

    tot_geo = sum(pt["timing"]["pass1_proposal_s"] + pt["timing"]["pass2_scoring_s"] for pt in pts)
    n_ts = sum(pt["n_ts"] for pt in pts)
    results["timing"] = {"frames": n_ts,
                         "decision_step_ms_per_frame_k1": float(step_k1.mean() * 1000),
                         "projection_and_comparison_s_per_frame": tot_geo / n_ts}
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_report(results, args.out_dir)
    print(f"saved {args.out_dir/'results.json'} and {args.out_dir/'REPORT.md'}")


def _pair_rows(lt: dict) -> str:
    rows = ["| Camera pair | True links | Correct | Wrong | Missed |", "|---|---:|---:|---:|---:|"]
    for pn, p in lt["per_pair"].items():
        rows.append(f"| {pn} | {p['true']} | {p['correct']} | {p['wrong']} | {p['missed']} |")
    rows.append(f"| All | {lt['true']} | {lt['correct']} | {lt['wrong']} | {lt['missed']} |")
    return "\n".join(rows)


def write_report(res: dict, out_dir: Path) -> None:
    fl, dep, causal = res["tracker_only"], res["per_scene_linker"], res["causal"]
    L = ["# Causal linker on the validation split\n",
         f"{res['n_scenes']} validation scenes, from the cached predictions.\n",
         "## Table S6\n",
         "| Configuration | k | wSTQ | wAQ | Correct | Wrong | Missed |",
         "|---|---:|---:|---:|---:|---:|---:|",
         f"| Tracker only | - | {fl['wSTQ']:.4f} | {fl['wAQ']:.4f} | 0 | 0 | {dep['links']['true']} |",
         f"| Per-scene linker | - | {dep['wSTQ']:.4f} | {dep['wAQ']:.4f} | {dep['links']['correct']} | "
         f"{dep['links']['wrong']} | {dep['links']['missed']} |"]
    for k in KS:
        r = causal[f"k{k}"]
        L.append(f"| Causal linker | {k} | {r['wSTQ']:.4f} | {r['wAQ']:.4f} | {r['links']['correct']} | "
                 f"{r['links']['wrong']} | {r['links']['missed']} |")
    L.append(f"\nOf the {dep['merged_links']} merges of the per-scene linker, "
             f"{dep['merged_links_single_scored_frame']} "
             f"({100 * dep['merged_links_single_scored_frame'] / dep['merged_links']:.0f}%) "
             "are based on a single scored frame.\n")
    L.append("## Links per camera pair, per-scene linker\n")
    L.append(_pair_rows(dep["links"]))
    L.append("\n## Links per camera pair, causal linker, k = 1\n")
    L.append(_pair_rows(causal["k1"]["links"]))
    t = res["timing"]
    L.append("\n## Timing (single CPU thread)\n")
    L.append(f"- Causal decision step (k = 1): {t['decision_step_ms_per_frame_k1']:.2f} ms per frame "
             f"(mean over {t['frames']} frames).")
    L.append(f"- Projection and comparison of the candidate masks: "
             f"{t['projection_and_comparison_s_per_frame']:.2f} s per frame.")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["build", "sweep"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "outputs/tracker_ctvis_val_cache")
    ap.add_argument("--adapter", type=Path, default=REPO / "outputs/linker/adapter.pt")
    ap.add_argument("--cosines-dir", type=Path, default=REPO / "outputs/linker/adapted_cosines")
    ap.add_argument("--dataset-root", default=str((REPO.parent / "data/waymo_open_dataset_v_2_0_1").resolve()))
    ap.add_argument("--config", type=Path, default=ol.DEFAULT_CONFIG,
                    help="linker configuration file (key 'deployed')")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/online_linker")
    args = ap.parse_args()
    args.cfg = ol.load_config(args.config)
    if args.stage == "build":
        stage_build(args, args.cfg)
    else:
        stage_sweep(args)


if __name__ == "__main__":
    main()
