"""analyze_significance.py — one-sided one-sample Wilcoxon test of the per-scene wSTQ.

Tests whether the per-scene validation wSTQ of the full linker lies above the
validation wSTQ of the ViP-DeepLab baseline of Mei et al., 0.1778 (one-sided,
one-sample Wilcoxon signed-rank test). The benchmark publishes only aggregate
scores of other methods, so no paired test is possible.

The per-scene values (Table S7) are read from the JSON written by
scripts/eval_linker_wstq.py (`per_scene`), for one configuration key.

Usage:
  python scripts/analyze_significance.py --wstq-json outputs/linker/table7.json \
      --config appearance_veto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
REFERENCE = 0.1778          # Mei et al., ViP-DeepLab, validation wSTQ


def one_sample_wilcoxon(vals: list, ref: float) -> dict:
    """One-sided one-sample Wilcoxon signed-rank test of `vals` against `ref`."""
    a = np.array(vals, dtype=float)
    d = a - ref
    W, p = wilcoxon(d, alternative="greater")          # H1: per-scene > ref
    return {"n": int(a.size), "ref": ref, "n_above": int((d > 0).sum()),
            "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max()),
            "wilcoxon_W": float(W), "p_one_sided": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wstq-json", type=Path, default=REPO / "outputs/linker/linker_wstq.json",
                    help="output of scripts/eval_linker_wstq.py")
    ap.add_argument("--config", default="deployed", help="configuration key in that file")
    ap.add_argument("--ref", type=float, default=REFERENCE)
    ap.add_argument("--output", type=Path, default=REPO / "outputs/linker/significance.json")
    args = ap.parse_args()

    res = json.loads(args.wstq_json.read_text())
    rows = res["per_scene"]
    per_scene = {r["scene"]: r["linker"][args.config] for r in rows}
    vals = [v["wSTQ"] for v in per_scene.values()]

    out = {"config": args.config, "reference": args.ref,
           "per_scene": per_scene, "test": one_sample_wilcoxon(vals, args.ref)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print(f"{'scene':>40} {'wSTQ':>7} {'wAQ':>7} {'wSQ':>7}")
    for i, (scene, v) in enumerate(sorted(per_scene.items()), 1):
        print(f"{i:>3} {scene:>36} {v['wSTQ']:7.4f} {v['wAQ']:7.4f} {v['wSQ']:7.4f}")
    t = out["test"]
    print(f"\n{t['n_above']}/{t['n']} scenes above {args.ref}; one-sided Wilcoxon "
          f"W={t['wilcoxon_W']:.0f}, p={t['p_one_sided']:.3e}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
