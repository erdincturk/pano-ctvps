#!/usr/bin/env python3
"""build_segmenter_checkpoint.py — a checkpoint for caching the outputs of the segmenter before identity training.

scripts/pipeline/build_cache_gt.py loads the segmenter and an embedding head from one checkpoint. For the segmenter before
identity training (rows 1 and 2 of Table 6; supplement, Tables S9 and S10), this script copies the
segmenter weights unchanged and adds an untrained embedding head (EmbeddingHead(), torch.manual_seed(0)). The head is
never used by these rows: the IoU tracker uses the masks only and MinVIS uses the frame queries of the cache.

Usage:
  python scripts/experiments/build_segmenter_checkpoint.py --segmenter checkpoints/segmenter.pth \
      --output checkpoints/segmenter_for_cache.pth
  python scripts/pipeline/build_cache_gt.py --checkpoint checkpoints/segmenter_for_cache.pth --split validation \
      --cache-dir outputs/segmenter_val_cache
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from panoctvps.tracker.embedding_head import EmbeddingHead


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segmenter", type=Path, required=True, help="Checkpoint of scripts/pipeline/train_segmenter.py.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    src = torch.load(args.segmenter, map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    head = EmbeddingHead()
    out = {"model": src["model"], "embed_head": head.state_dict(), "iteration": src.get("iteration"),
           "source": args.segmenter.name,
           "embed_head_note": "untrained EmbeddingHead(), torch.manual_seed(0); not used by the IoU tracker or MinVIS"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    (args.output.parent / (args.output.name + ".sha256")).write_text(f"{digest}  {args.output.name}\n")
    print(f"saved {args.output} (sha256 {digest[:16]}...)")


if __name__ == "__main__":
    main()
