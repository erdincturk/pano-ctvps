"""Print Tables S1 and S2 (wSTQ of the alternative trackers) from the cells of scripts/sweep_baselines.py and write
them to $BASELINE_OUT_DIR/TABLES.md (default outputs/baselines). Cells that have not run are shown as "–"."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_baselines import OUT, iou_name, minvis_name  # noqa: E402


def wstq(name):
    p = OUT / "cells" / f"{name}.json"
    return f"{json.load(open(p))['aggregate']['wSTQ']:.4f}" if p.exists() else "–"


def table(head, cols, rows):
    out = ["| " + head + " | " + " | ".join(str(c) for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    out += ["| " + label + " | " + " | ".join(cells) + " |" for label, cells in rows]
    return "\n".join(out)


ious = (0.1, 0.2, 0.3, 0.5)
matches = (0.5, 0.7, 0.8, 0.9)
L = ["Table S1. The IoU tracker: minimum IoU and frames kept.", "",
     table("Minimum IoU", ious, [(f"{d} frames kept", [wstq(iou_name(t, d)) for t in ious]) for d in (5, 10)]), "",
     "Table S2. MinVIS: match and initiation thresholds, and frames kept.", "",
     table("Match", matches,
           [(f"Initiation {i}, 5 frames kept", [wstq(minvis_name(m, i, 5)) for m in matches]) for i in (0.01, 0.05)]
           + [("Initiation 0.01, 10 frames kept",
               [wstq(minvis_name(m, 0.01, 10)) if m == 0.9 else "–" for m in matches])]), ""]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "TABLES.md").write_text("\n".join(L))
print("\n".join(L))
