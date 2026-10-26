"""build_scores.py — persist the scored candidates of the linker per scene.

Runs proposal -> scoring once per scene of the prediction cache and saves one
labelled candidate table per scene (schema in panoctvps/linker/link_eval.py).
scripts/build_cosines.py and scripts/eval_linker_links.py read these tables,
so the projection pass is not repeated.

Usage:
  python scripts/build_scores.py --cache-dir outputs/tracker_ctvis_val_cache \
      --out-dir outputs/linker/scored_candidates
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from panoctvps.linker.link_eval import build_scene_struct, save_scene_table
from panoctvps.linker.overlap import calibration_from_arrays, overlap_by_partner
from panoctvps.linker.pipeline import DEFAULT_TID_FIELD as TID_FIELD
from panoctvps.linker.proposer import propose_candidates
from panoctvps.linker.scorer import score_candidates

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO / "outputs/tracker_ctvis_val_cache"
DEFAULT_OUT = REPO / "outputs/linker/scored_candidates"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--segment", default=None, help="single segment stem (default: all)")
    args = ap.parse_args()

    paths = ([args.cache_dir / f"{args.segment}.npz"] if args.segment
             else sorted(args.cache_dir.glob("*.npz")))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scoring {len(paths)} scene(s) [{args.cache_dir}] -> {args.out_dir}")

    t0 = time.time()
    for i, p in enumerate(paths, 1):
        npz = np.load(p, allow_pickle=True)
        wh = {}
        for fr in npz["frame_records"]:
            wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
        cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)
        cands = propose_candidates(npz, overlap_by_partner(cams), tid_field=TID_FIELD)
        scored = score_candidates(npz, cams, cands, tid_field=TID_FIELD)
        struct = build_scene_struct(npz, TID_FIELD)
        save_scene_table(args.out_dir / p.name, scored, struct, segment=p.stem,
                         tid_field=TID_FIELD)
        print(f"  [{i}/{len(paths)}] {p.stem[:34]:34s} candidates={len(scored):6d} "
              f"tracks={len(struct['all_tr']):6d} ({time.time()-t0:6.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
