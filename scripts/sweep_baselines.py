"""The searches of the alternative trackers on the cached segmenter outputs (Table 6; Tables S1 and S2).

  IoU tracker   minimum IoU ∈ {0.1, 0.2, 0.3, 0.5} x frames kept ∈ {5, 10}                          (Table S1)
  MinVIS        match ∈ {0.5, 0.7, 0.8, 0.9} x initiation ∈ {0.01, 0.05}, 5 frames kept,
                and match 0.9 / initiation 0.01 / 10 frames kept                                     (Table S2)

  python scripts/sweep_baselines.py --workers 5
  python scripts/baseline_tables.py

Every cell is one call of scripts/run_baseline_cell.py. Each cell needs about 18 GB (the harness holds one scene
of ground truth); choose --workers accordingly. Cells whose JSON exists are skipped. Results go to
$BASELINE_OUT_DIR (default outputs/baselines).
"""
import argparse, itertools, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = Path(sys.executable)
OUT = Path(os.environ.get("BASELINE_OUT_DIR", HERE.parent / "outputs" / "baselines"))

IOU = list(itertools.product((0.1, 0.2, 0.3, 0.5), (5, 10)))
MINVIS = [(m, i, 5) for i in (0.01, 0.05) for m in (0.5, 0.7, 0.8, 0.9)] + [(0.9, 0.01, 10)]


def iou_name(t, d):
    return f"iou_t{t:.2f}_d{d}"


def minvis_name(m, i, d):
    return f"minvis_m{m:.2f}_i{i:.2f}_d{d}"


def cells():
    out = [(iou_name(t, d), ["--tracker", "iou", "--iou-threshold", str(t), "--num-dead-frames", str(d)])
           for t, d in IOU]
    out += [(minvis_name(m, i, d), ["--tracker", "minvis", "--match-threshold", str(m),
                                    "--init-score-threshold", str(i), "--num-dead-frames", str(d)])
            for m, i, d in MINVIS]
    return out


def run(cell):
    name, args = cell
    out = OUT / "cells" / f"{name}.json"
    if out.exists():
        return name, "exists"
    log = OUT / "logs" / f"{name}.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run([str(PY), str(HERE / "run_baseline_cell.py"), *args, "--output", str(out)],
                           cwd=str(HERE.parent), stdout=f, stderr=subprocess.STDOUT, env=env)
    return name, f"rc={r.returncode} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=5)
    workers = ap.parse_args().workers
    (OUT / "cells").mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(parents=True, exist_ok=True)
    todo = cells()
    print(f"{len(todo)} cells, {workers} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for name, status in ex.map(run, todo):
            print(f"{time.time() - t0:7.0f}s  {name:24s} {status}", flush=True)
    print("ALL DONE", flush=True)
