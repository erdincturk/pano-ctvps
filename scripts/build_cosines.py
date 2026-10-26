"""build_cosines.py — adapted cosines of the scored candidates.

Applies the trained adapter psi to the cached embeddings and computes, for
every scored candidate pair (scripts/build_scores.py), the median over all
shared frames of the cosine between the adapted embeddings
(``cos_med_adapted``), with the same function as the raw ``cos_med``.
Writes one npz per scene; the appearance veto reads it.

Usage:
  python scripts/build_cosines.py --adapter outputs/linker/adapter.pt \
      --cache-dir outputs/tracker_ctvis_val_cache \
      --scores-dir outputs/linker/scored_candidates --out-dir outputs/linker/adapted_cosines
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from panoctvps.linker.adapter import EmbeddingAdapter
from panoctvps.linker.link_eval import load_scene_table
from panoctvps.linker.pipeline import DEFAULT_TID_FIELD as TID_FIELD
from panoctvps.linker.scorer import compute_pair_cosines

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO / "outputs/tracker_ctvis_val_cache"
DEFAULT_SCORES = REPO / "outputs/linker/scored_candidates"
DEFAULT_OUT = REPO / "outputs/linker/adapted_cosines"

COS_DTYPE = np.dtype([
    ("camA", "i1"), ("tidA", "i4"), ("camB", "i1"), ("tidB", "i4"),
    ("cos_med_adapted", "f8"),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    model = EmbeddingAdapter.load(args.adapter)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = sorted(args.scores_dir.glob("*.npz"))
    t0 = time.time()
    for i, sp in enumerate(src, 1):
        npz = np.load(args.cache_dir / sp.name, allow_pickle=True)
        with torch.no_grad():
            emb = model(torch.from_numpy(npz["embeddings"].astype(np.float32))).numpy()
        candidates = set(load_scene_table(sp)["scored"].keys())
        cos = compute_pair_cosines({"records": npz["records"], "embeddings": emb},
                                   candidates, tid_field=TID_FIELD)

        rows = np.zeros(len(cos), dtype=COS_DTYPE)
        for j, (pair, d) in enumerate(cos.items()):
            (a, ta), (b, tb) = sorted(pair)
            rows[j] = (a, ta, b, tb, np.nan if d["cos_med"] is None else d["cos_med"])
        np.savez_compressed(args.out_dir / sp.name, cosines=rows)
        print(f"  [{i}/{len(src)}] {sp.stem[:34]:34s} pairs={len(rows):6d} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
