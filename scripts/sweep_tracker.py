#!/usr/bin/env python3
"""sweep_tracker.py — the searches of our tracker on the cached segmenter outputs (Table 6; Tables S3 and S4).

Launcher only: every cell is one call of scripts/eval_tracker_cached.py (CPU, deterministic given the cache).
  thresholds   the published setting (θ_match, θ_init, N_dead) = (0.3, 0.01, 20), then one threshold varied at a
               time with the others at (0.7, 0.01, 5): θ_match ∈ {0.5, ..., 0.9}, θ_init ∈ {0.01, 0.05},
               N_dead ∈ {5, 10}; motion gate disabled (Table S3)
  gate         one bound d_k for all cameras, d_k ∈ {200, 300, ..., 1200} px, at (0.7, 0.01, 5) (Table S4)
  deployed     the bound chosen for each camera: 600 / 800 / 800 / 1000 / 1000 px
               (Front, Front-Left, Front-Right, Side-Left, Side-Right)

  python scripts/sweep_tracker.py --grid thresholds --workers 5
  python scripts/sweep_tracker.py --grid gate --workers 5
  python scripts/sweep_tracker.py --grid deployed
  python scripts/sweep_tracker.py --summary          # Tables S3 and S4 from the finished cells

Each cell needs about 18 GB of memory (one scene of full-resolution ground truth); choose --workers accordingly.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAM = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
RETUNED = (0.7, 0.01, 5)


def name_of(m, i, d):
    return f"match{round(m * 100):03d}_init{round(i * 100):03d}_dead{d}"


def thresholds():
    """(θ_match, θ_init, N_dead) of Table S3, without duplicates."""
    m0, i0, d0 = RETUNED
    out = [(0.3, 0.01, 20)]
    out += [(m, i0, d0) for m in (0.5, 0.6, 0.7, 0.8, 0.9)]
    out += [(m0, i, d0) for i in (0.01, 0.05)]
    out += [(m0, i0, d) for d in (5, 10)]
    return list(dict.fromkeys(out))


def cells(grid):
    m0, i0, d0 = RETUNED
    base = ["--match-threshold", str(m0), "--init-score-threshold", str(i0), "--num-dead-frames", str(d0)]
    if grid == "thresholds":
        for m, i, d in thresholds():
            yield (name_of(m, i, d),
                   ["--match-threshold", str(m), "--init-score-threshold", str(i), "--num-dead-frames", str(d)])
    elif grid == "gate":
        for px in range(200, 1201, 100):
            yield (f"flatpx{px}", base + ["--motion-gate", "--motion-gate-px", str(px)])
    elif grid == "deployed":
        yield ("deployed", base + ["--motion-gate",
                                   "--motion-gate-px-per-cam", "1:600,2:800,3:800,4:1000,5:1000"])


def run(cell, args):
    name, extra = cell
    out = args.out_dir / f"{name}.json"
    if out.exists():
        return name, "exists"
    cmd = [sys.executable, str(REPO / "scripts" / "eval_tracker_cached.py"), "--cache-dir", str(args.cache_dir),
           "--dataset-root", str(args.dataset_root), "--tracker", "memorybank", *extra, "--output", str(out)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(args.out_dir / f"{name}.log", "w") as f:
        r = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, env=env)
    return name, f"rc={r.returncode}"


def load(out_dir, name):
    p = out_dir / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def summary(out_dir):
    print("Table S3: our tracker, motion gate disabled")
    print(f"  {'(θ_match, θ_init, N_dead)':28s} {'wSTQ':>7s} {'wAQ':>7s} {'wSQ':>7s}")
    for m, i, d in thresholds():
        r = load(out_dir, name_of(m, i, d))
        a = r["aggregate"] if r else None
        vals = f"{a['wSTQ']:7.4f} {a['wAQ']:7.4f} {a['wSQ']:7.4f}" if a else "   (not run)"
        print(f"  {str((m, i, d)):28s} {vals}")
    rows = [("no gate", load(out_dir, name_of(*RETUNED)))]
    rows += [(f"{px}", load(out_dir, f"flatpx{px}")) for px in range(200, 1201, 100)]
    print("\nTable S4: one bound d_k for all cameras, wSTQ and the STQ of each camera")
    print(f"  {'d_k (px)':9s} {'wSTQ':>7s} " + " ".join(f"{c:>11s}" for c in CAM.values()))
    for label, r in rows:
        if r is None:
            print(f"  {label:9s}    (not run)")
            continue
        print(f"  {label:9s} {r['aggregate']['wSTQ']:7.4f} "
              + " ".join(f"{r['per_camera'][c]['STQ']:11.4f}" for c in CAM.values()))
    gate = [(px, r) for px, r in rows[1:] if r is not None]
    if gate:
        print("\n  best bound per camera: "
              + ", ".join(f"{c} {max(gate, key=lambda g: g[1]['per_camera'][c]['STQ'])[0]} px" for c in CAM.values()))
    dep = load(out_dir, "deployed")
    if dep:
        print(f"  chosen bounds (600/800/800/1000/1000 px): wSTQ {dep['aggregate']['wSTQ']:.4f}, "
              f"wAQ {dep['aggregate']['wAQ']:.4f}, wSQ {dep['aggregate']['wSQ']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=("thresholds", "gate", "deployed"))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "outputs/tracker_ctvis_val_cache")
    ap.add_argument("--dataset-root", type=Path, default=REPO.parent / "data/waymo_open_dataset_v_2_0_1")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/tracker_sweep")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.grid:
        todo = list(cells(args.grid))
        with ThreadPoolExecutor(args.workers) as ex:
            for name, status in ex.map(lambda c: run(c, args), todo):
                print(f"{name:24s} {status}", flush=True)
    if args.summary or not args.grid:
        summary(args.out_dir)


if __name__ == "__main__":
    main()
