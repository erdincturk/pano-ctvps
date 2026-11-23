#!/usr/bin/env python3
"""build_overlap_tables.py — overlap tables of a cache, for exact fast scoring of tracker and linker runs.

The trackers and the linker change only identities, never masks or classes. For every frame and camera, the
overlap of every predicted thing segment with every ground-truth thing object, the size of every predicted segment
and the size of every ground-truth object are stored once per cache (pixel-weighted as in wSTQ, and unweighted for
the per-camera STQ), with the per-class IoU sums of SQ (panoctvps/evaluation/overlap_tables.py). Scoring an
assignment of identities from these tables gives the scores of scripts/pipeline/eval_tracker_cached.py and
scripts/pipeline/eval_linker_wstq.py to rounding. Used by Tables 6 and 7 of the paper and the supplement,
Tables S9 to S14, S23, S27 and S28 (see README).

Usage:
  python scripts/experiments/build_overlap_tables.py --cache-dir outputs/tracker_ctvis_val_cache \
      --out-dir outputs/overlap_tables/tracker_ctvis_val
"""
from __future__ import annotations

import argparse
from pathlib import Path

from panoctvps.evaluation.overlap_tables import build

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True, help="Cache of segmenter outputs (one npz per scene).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output folder (one npz per scene).")
    ap.add_argument("--dataset-root", default=str(REPO.parent / "data/waymo_open_dataset_v_2_0_1"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--scenes", default="", help="Comma-separated scene names (default: all scenes of the cache).")
    args = ap.parse_args()
    build(args.cache_dir, args.out_dir, args.dataset_root, args.split,
          [s for s in args.scenes.split(",") if s] or None)


if __name__ == "__main__":
    main()
