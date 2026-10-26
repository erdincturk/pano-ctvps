"""Write the per-scene prediction cache without ground truth (test split).

Same cache layout and prediction path as scripts/build_cache_gt.py, for the
test split, whose camera_segmentation files hold no labels. Differences, all
due to the missing ground truth:

  * Scenes and frames come from configs/splits/{split}_frames.txt.
  * A test scene may be stored in either download folder (testing_location/
    or testing/); both are searched.
  * gt_global_id / gt_iou are 0; the sequence id is the context name (one
    sequence per scene, sequence_idx = 0), as in the submission format.
  * No evaluation. `records['tracker_id']` holds the ids of the per-camera
    tracker run here (no motion gate); the ids used downstream are added
    afterwards from the cache (scripts/eval_tracker_cached.py
    --save-trajectories, then scripts/build_cache_trackids.py), as for the
    validation cache.

Usage:
  python scripts/build_cache.py --split test --dry-run        # 1 scene
  python scripts/build_cache.py --split test
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from panoctvps.data.cache_io import encode_rle, encode_sem_rle              # noqa: E402
from panoctvps.data.constants import NUM_CLASSES, PARQUET_COLS              # noqa: E402
from panoctvps.data.decode import decode_jpeg                               # noqa: E402
from panoctvps.data.transforms.augmentations import PadToMultiple           # noqa: E402
from panoctvps.data.transforms.formatting import ToTensorNormalize          # noqa: E402
from panoctvps.tracker.embedding_head import EmbeddingHead                   # noqa: E402
from panoctvps.segmenter.inference import predict_panoptic_with_learned_embeddings  # noqa: E402
from panoctvps.segmenter.mask2former import build_model                       # noqa: E402
from panoctvps.tracker.memory_bank import MemoryBankTracker                  # noqa: E402
from panoctvps.data.cache_io import (                            # noqa: E402
    RECORD_DTYPE, load_geometry, write_segment_npz,
)

SPLIT_DIRS = {"test": ("testing_location", "testing"),
              "validation": ("validation",), "training": ("training",)}
CAMERAS = (1, 2, 3, 4, 5)


def load_split_frames(split: str):
    """{context_name: [timestamps]} from the canonical split file."""
    name = {"test": "test_frames.txt",
            "validation": "validation_frames.txt",
            "training": "training_frames.txt"}[split]
    by_ctx = collections.defaultdict(list)
    for line in (REPO / "configs/splits" / name).read_text().splitlines():
        if line.strip():
            ctx, ts = line.strip().split(",")
            by_ctx[ctx].append(int(ts))
    return {c: sorted(v) for c, v in by_ctx.items()}


def resolve_subdir(dataset_root: Path, split: str, seg_file: str) -> str:
    """Which split folder holds this scene's parquets (test spans two)."""
    for sub in SPLIT_DIRS[split]:
        if (dataset_root / sub / "camera_image" / seg_file).exists():
            return sub
    raise FileNotFoundError(f"{seg_file} not found in any of {SPLIT_DIRS[split]}")


def build_scene(model, embed_head, device, args,
                seg_idx: int, ctx: str, timestamps, pads, fmt) -> Path:
    seg_file = f"{ctx}.parquet"
    subdir = resolve_subdir(Path(args.dataset_root), args.split, seg_file)
    img_path = Path(args.dataset_root) / subdir / "camera_image" / seg_file
    img_df = pq.read_table(img_path, columns=[
        PARQUET_COLS["img_key_timestamp"], PARQUET_COLS["img_key_camera"],
        PARQUET_COLS["img_image"], PARQUET_COLS["img_pose"],
    ]).to_pandas()
    extrinsics, intrinsics = load_geometry(args.dataset_root, subdir,
                                            seg_file, list(CAMERAS))

    unique_timestamps = np.array(timestamps, dtype=np.int64)
    ego_by_ts = {}
    front = img_df[img_df[PARQUET_COLS["img_key_camera"]] == 1]
    for _, row in front.iterrows():
        ego_by_ts[int(row[PARQUET_COLS["img_key_timestamp"]])] = np.array(
            row[PARQUET_COLS["img_pose"]], dtype=np.float32).reshape(4, 4)
    ego_poses = np.stack([ego_by_ts.get(int(t), np.eye(4, dtype=np.float32))
                          for t in unique_timestamps], axis=0)

    trackers = {c: MemoryBankTracker(match_threshold=args.match_threshold,
                                     init_score_threshold=args.init_score_threshold,
                                     num_dead_frames=args.num_dead_frames)
                for c in CAMERAS}

    frame_records, records, embeddings = [], [], []
    mask_rles, pred_sem_rles, frame_queries, frame_class_logits = [], [], [], []
    cache_frame_idx = 0

    for frame_counter, timestamp in enumerate(timestamps):
        for cam_id in CAMERAS:
            img_row = img_df[
                (img_df[PARQUET_COLS["img_key_timestamp"]] == timestamp)
                & (img_df[PARQUET_COLS["img_key_camera"]] == cam_id)]
            if img_row.empty:
                raise RuntimeError(f"{ctx}: missing image ts={timestamp} cam={cam_id}")
            image = decode_jpeg(img_row.iloc[0][PARQUET_COLS["img_image"]])
            oh, ow = image.shape[:2]
            sample = {"image": image,
                      "semantic_label": np.zeros(image.shape[:2], np.int32),
                      "instance_label": np.zeros(image.shape[:2], np.int32)}
            sample = fmt(pads[cam_id](sample))
            ph, pw = sample["image"].shape[-2], sample["image"].shape[-1]

            pv = sample["image"].unsqueeze(0).to(device)
            preds = predict_panoptic_with_learned_embeddings(
                model, embed_head, pv,
                threshold=args.obj_threshold,
                mask_threshold=args.mask_threshold,
                overlap_mask_area_threshold=args.overlap_threshold,
                return_raw=True)

            pred_sem = preds[0]["semantic"][:oh, :ow]
            pred_inst = preds[0]["instance"][:oh, :ow]
            pred_embs = preds[0]["instance_embeddings"]
            pred_scores = preds[0]["instance_scores"]
            pred_classes = preds[0]["instance_classes"]

            tracked = trackers[cam_id].update(
                pred_sem, pred_inst, pred_embs, pred_scores, pred_classes,
                frame_counter)

            frame_queries.append(preds[0]["frame_queries"])
            frame_class_logits.append(preds[0]["frame_class_logits"])
            pred_sem_rles.append(encode_sem_rle(pred_sem))
            query_idx_map = preds[0].get("instance_query_idx", {})

            inst_ids = list(pred_embs.keys())
            rec_start = len(records)
            for local_idx, inst_id in enumerate(inst_ids):
                mask_orig = (pred_inst == inst_id).astype(np.uint8)
                mask_padded = np.zeros((ph, pw), np.uint8)
                mask_padded[:oh, :ow] = mask_orig
                mask_rles.append(encode_rle(mask_padded))

                t_ids = tracked[pred_inst == inst_id]
                t_ids_pos = t_ids[t_ids > 0]
                tracker_id = int(np.bincount(t_ids_pos).argmax()) \
                    if len(t_ids_pos) > 0 else 0

                ys, xs = np.where(mask_orig)
                records.append((
                    seg_idx, timestamp, cam_id, local_idx, tracker_id,
                    pred_classes.get(inst_id, 0), pred_scores.get(inst_id, 0.0),
                    float(xs.mean()) if len(xs) else 0.0,
                    float(ys.mean()) if len(ys) else 0.0,
                    float(xs.min()) if len(xs) else 0.0,
                    float(ys.min()) if len(ys) else 0.0,
                    float(xs.max()) if len(xs) else 0.0,
                    float(ys.max()) if len(ys) else 0.0,
                    int(mask_orig.sum()),
                    0, 0.0,                      # gt_global_id / gt_iou: no GT
                    cache_frame_idx,
                    query_idx_map.get(inst_id, -1),
                    0,                           # sequence_idx: one sequence/scene
                ))
                embeddings.append(pred_embs[inst_id])

            frame_records.append((seg_idx, timestamp, cam_id, oh, ow,
                                  len(inst_ids), rec_start, len(records)))
            cache_frame_idx += 1

    return write_segment_npz(
        cache_dir=Path(args.cache_dir), seg_file=seg_file, seg_name=ctx,
        seg_idx=seg_idx,
        unique_sequence_ids=np.array([ctx], dtype=object),
        unique_timestamps=unique_timestamps,
        extrinsics=extrinsics, intrinsics=intrinsics, ego_poses=ego_poses,
        coverage_by_cam={},
        frame_records_list=frame_records, records_list=records,
        embeddings_list=embeddings, mask_rles=mask_rles,
        pred_sem_rles=pred_sem_rles, frame_queries_list=frame_queries,
        frame_class_logits_list=frame_class_logits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("test", "validation", "training"),
                    required=True)
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints/tracker.pth"))
    ap.add_argument("--backbone", default="swin-large-mapillary")
    ap.add_argument("--dataset-root", default=str(REPO.parent / "data/waymo_open_dataset_v_2_0_1"))
    ap.add_argument("--cache-dir", default=str(REPO / "outputs/tracker_ctvis_test_cache"))
    # M2F post-processing (HF defaults, as in build_cache_gt.py)
    ap.add_argument("--obj-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--overlap-threshold", type=float, default=0.8)
    # Per-camera tracker whose ids fill records['tracker_id']
    ap.add_argument("--match-threshold", type=float, default=0.7)
    ap.add_argument("--init-score-threshold", type=float, default=0.01)
    ap.add_argument("--num-dead-frames", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="first scene only")
    ap.add_argument("--start", type=int, default=0, help="resume from scene index")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    by_ctx = load_split_frames(args.split)
    contexts = sorted(by_ctx)
    if args.dry_run:
        contexts = contexts[:1]
    print(f"split={args.split}: {len(contexts)} scenes, "
          f"{sum(len(v) for v in by_ctx.values())} timestamps | device={device}")

    print(f"Loading model ({args.backbone}) from {args.checkpoint} ...")
    model = build_model(backbone=args.backbone, num_classes=NUM_CLASSES,
                        pretrained=False)
    embed_head = EmbeddingHead()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    embed_head.load_state_dict(ckpt["embed_head"])
    model = model.to(device).eval()
    embed_head = embed_head.to(device).eval()

    pads = {c: PadToMultiple(multiple=32) for c in CAMERAS}
    fmt = ToTensorNormalize()
    t0 = time.time()
    with torch.no_grad():
        for i, ctx in enumerate(contexts):
            if i < args.start:
                continue
            out = build_scene(model, embed_head, device,
                              args, i, ctx, by_ctx[ctx], pads, fmt)
            print(f"  [{i+1}/{len(contexts)}] {ctx[:36]:36s} "
                  f"ts={len(by_ctx[ctx]):3d} -> {out.name} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
