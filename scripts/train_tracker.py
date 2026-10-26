#!/usr/bin/env python3
"""Identity training of the tracker (consistent contrastive training).

Jointly trains the Mask2Former decoder and the embedding head with:
  - the segmentation losses (CE + mask BCE + dice) on every frame
  - the contrastive losses (InfoNCE + auxiliary cosine term) across the frames
    of each clip, with memory-bank positives and per-class anchor weights

The backbone is frozen. Learning rates and the schedule are set in the config.

Usage:
    python scripts/train_tracker.py --config configs/training/tracker_ctvis.yaml

    # Dry-run:
    python scripts/train_tracker.py \\
        --config configs/training/tracker_ctvis.yaml \\
        --max-iterations 10 --no-eval

    # Resume:
    python scripts/train_tracker.py \\
        --config configs/training/tracker_ctvis.yaml --resume
"""

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from panoctvps.data.class_weights import (
    WOD_TRAINING_THING_INSTANCE_COUNTS,
    compute_class_weights,
)
from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES
from panoctvps.data.video_dataset import WODVideoDataset, video_collate_fn
from panoctvps.segmenter.mask2former import build_model, extract_query_embeddings, freeze_backbone
from panoctvps.tracker.embedding_head import EmbeddingHead
from panoctvps.tracker.contrastive_loss import (
    build_multi_ref_contrastive_items,
    contrastive_loss,
)
from panoctvps.tracker.memory_bank_train import TrainMemoryBank
from panoctvps.training.engine import (
    load_config, get_lr_scale_cosine, save_checkpoint, save_best,
    load_latest_checkpoint, TrainingLogger,
)


def build_optimizer(
    model: nn.Module,
    embed_head: nn.Module,
    cfg: dict,
) -> torch.optim.Optimizer:
    """Build AdamW with three param groups: backbone (0), decoder, embed_head."""
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    lr_decoder = opt_cfg.get("lr_decoder", 2e-5)
    lr_embed = opt_cfg.get("lr_embed", 2e-4)
    wd = opt_cfg.get("weight_decay", 0.05)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    decoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # All non-frozen params are decoder (backbone is frozen)
        decoder_params.append(p)

    param_groups = [
        {"params": decoder_params, "lr": lr_decoder},
        {"params": list(embed_head.parameters()), "lr": lr_embed},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=wd, betas=betas)


def hungarian_indices(model, outputs, mask_labels, class_labels, device):
    """Run the Hungarian matching of the segmentation loss (query→GT assignments).

    Extracts matching indices without computing loss (we already have
    the M2F loss from the forward pass).

    Args:
        model: HF Mask2FormerForUniversalSegmentation model.
        outputs: model forward pass outputs.
        mask_labels: list of (N, H, W) tensors per frame.
        class_labels: list of (N,) tensors per frame.
        device: torch device.

    Returns:
        List of (src_indices, tgt_indices) tuples, one per frame.
    """
    hungarian = model.criterion.matcher   # transformers' Hungarian matching module

    # Call signature:
    #   (masks_queries_logits, class_queries_logits, mask_labels, class_labels)
    with torch.no_grad():
        indices = hungarian(
            outputs.masks_queries_logits,
            outputs.class_queries_logits,
            mask_labels,
            class_labels,
        )

    return indices


@torch.no_grad()
def evaluate_pq(model, dataset_root: str, cfg: dict, device: torch.device,
                n_samples: int = 200):
    """In-training PQ on a validation subset.

    Runs single-frame inference on the first n_samples validation frames at the
    crop resolution of the config, to monitor the segmentation during training.
    """
    from panoctvps.data.training_dataset import WODMask2FormerDataset
    from panoctvps.segmenter.mask2former import predict_panoptic
    from panoctvps.evaluation import PanopticQuality, MeanIoU

    model.eval()

    data_cfg = cfg.get("dataset", {})
    crop = tuple(cfg.get("augmentation", {}).get("crop_size", [640, 640]))

    val_ds = WODMask2FormerDataset(
        dataset_root=dataset_root,
        split="validation",
        cameras=data_cfg.get("cameras", [1]),
        crop_size=crop,
        min_scale=1.0, max_scale=1.0,
        flip_prob=0.0, color_jitter=False,
    )

    pq_metric = PanopticQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES)
    miou_metric = MeanIoU(num_classes=NUM_CLASSES)

    n = min(n_samples, len(val_ds))
    for i in range(n):
        sample = val_ds[i]
        pv = sample["pixel_values"].unsqueeze(0).to(device)
        preds = predict_panoptic(model, pv)
        pq_metric.update(
            preds[0]["semantic"], preds[0]["instance"],
            sample["semantic_label"], sample["instance_label"],
        )
        miou_metric.update(preds[0]["semantic"], sample["semantic_label"])

    pq_r = pq_metric.result()
    miou_r = miou_metric.result()
    model.train()

    return {
        "PQ": pq_r["PQ"], "SQ": pq_r["SQ"], "RQ": pq_r["RQ"],
        "PQ_thing": pq_r["PQ_thing"], "PQ_stuff": pq_r["PQ_stuff"],
        "mIoU": miou_r["mIoU"],
        "num_images": n,
    }


@torch.no_grad()
def evaluate_tracking(model, embed_head, dataset_root: str, cfg: dict,
                      device: torch.device, n_sequences: int = 3,
                      sequence_indices: Optional[List[int]] = None,
                      obj_threshold: float = 0.5,
                      mask_threshold: float = 0.5,
                      overlap_threshold: float = 0.8,
                      match_threshold: float = 0.6,
                      init_score_threshold: float = 0.3,
                      num_dead_frames: int = 10):
    """In-training tracking evaluation on selected validation sequences.

    Runs full-resolution inference with the memory-bank tracker on the selected
    validation sequences and reports the panoramic wSTQ / wAQ / wSQ, computed
    as in scripts/eval_tracker_cached.py.

    Args:
        sequence_indices: 0-based indices into the validation set. If None,
            the first n_sequences are used.
        obj_threshold, mask_threshold, overlap_threshold: panoptic
            post-processing thresholds.
        match_threshold, init_score_threshold, num_dead_frames: tracker
            settings (from the ``evaluation`` block of the config).
    """
    import numpy as np
    import pyarrow.parquet as pq

    from panoctvps.data.constants import PARQUET_COLS
    from panoctvps.data.decode import decode_jpeg
    from panoctvps.data.transforms.augmentations import PadToMultiple
    from panoctvps.data.transforms.formatting import ToTensorNormalize
    from panoctvps.data.waymo_dataset import WODPVPSSequenceDataset
    from panoctvps.evaluation import STQuality, merge_sequence_results
    from panoctvps.evaluation.tracking_utils import build_gt_global_instance
    from panoctvps.tracker.memory_bank import MemoryBankTracker
    from panoctvps.segmenter.inference import predict_panoptic_with_learned_embeddings

    model.eval()
    embed_head.eval()

    data_cfg = cfg.get("dataset", {})
    cameras = data_cfg.get("cameras", [1])

    dataset = WODPVPSSequenceDataset(
        dataset_root=dataset_root, split="validation"
    )

    if sequence_indices is not None:
        seq_indices = sequence_indices
    else:
        seq_indices = list(range(min(n_sequences, len(dataset))))

    all_seq_metrics = []

    _pads = {c: PadToMultiple(multiple=32) for c in cameras}
    fmt = ToTensorNormalize()

    for i in seq_indices:
        sequence = dataset[i]

        seq_stq = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES,
                            ignore_label=0)

        # One tracker per camera
        trackers = {
            cam_id: MemoryBankTracker(
                match_threshold=match_threshold,
                init_score_threshold=init_score_threshold,
                num_dead_frames=num_dead_frames,
            )
            for cam_id in cameras
        }
        # Pixel weights for panoramic wSTQ (1 / num_cameras_covered)
        pixel_weights = sequence["pixel_weights"]

        split_dir = os.path.join(dataset_root, "validation")
        img_path = os.path.join(
            split_dir, "camera_image", sequence["segment_file"]
        )
        img_df = pq.read_table(img_path, columns=[
            PARQUET_COLS["img_key_timestamp"],
            PARQUET_COLS["img_key_camera"],
            PARQUET_COLS["img_image"],
        ]).to_pandas()

        frame_counter = 0
        for frame in sequence["frames"]:
            timestamp = frame["timestamp"]
            for cam_id in cameras:
                if cam_id not in frame["cameras"]:
                    continue
                gt_data = frame["cameras"][cam_id]

                img_row = img_df[
                    (img_df[PARQUET_COLS["img_key_timestamp"]] == timestamp)
                    & (img_df[PARQUET_COLS["img_key_camera"]] == cam_id)
                ]
                if img_row.empty:
                    continue

                image = decode_jpeg(
                    img_row.iloc[0][PARQUET_COLS["img_image"]]
                )
                sample = {
                    "image": image,
                    "semantic_label": np.zeros(
                        image.shape[:2], dtype=np.int32
                    ),
                    "instance_label": np.zeros(
                        image.shape[:2], dtype=np.int32
                    ),
                }
                sample = _pads[cam_id](sample)
                sample = fmt(sample)
                oh, ow = sample.get("original_size", image.shape[:2])

                pv = sample["image"].unsqueeze(0).to(device)
                preds = predict_panoptic_with_learned_embeddings(
                    model, embed_head, pv,
                    threshold=obj_threshold,
                    mask_threshold=mask_threshold,
                    overlap_mask_area_threshold=overlap_threshold,
                )

                pred_sem = preds[0]["semantic"][:oh, :ow]
                pred_inst = preds[0]["instance"][:oh, :ow]
                pred_embs = preds[0]["instance_embeddings"]
                pred_scores = preds[0]["instance_scores"]
                pred_classes = preds[0]["instance_classes"]

                gt_sem = gt_data["semantic_label"]
                gt_global = build_gt_global_instance(
                    gt_data["instance_label"], gt_data["instance_mapping"]
                )
                tracked = trackers[cam_id].update(
                    pred_sem, pred_inst, pred_embs,
                    pred_scores, pred_classes, frame_counter,
                )

                # Panoramic wSTQ: namespace IDs by camera within the sequence
                tracked_agg = np.where(
                    tracked > 0, tracked + cam_id * 100000, 0,
                )
                seq_stq.update(
                    pred_semantic=pred_sem,
                    pred_instance=tracked_agg,
                    gt_semantic=gt_sem,
                    gt_instance=gt_global,
                    pixel_weights=pixel_weights[cam_id],
                )

            frame_counter += 1

        all_seq_metrics.append(seq_stq)

    model.train()
    embed_head.train()

    agg = merge_sequence_results(all_seq_metrics)
    return {"STQ": agg["STQ"], "AQ": agg["AQ"], "SQ": agg["SQ"]}


def _build_state(model, embed_head, optimizer, scaler, iteration, best_metric):
    """Build the checkpoint state dict."""
    return {
        "model": model.state_dict(),
        "embed_head": embed_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_metric": best_metric,
    }


def main():
    parser = argparse.ArgumentParser(description="Identity training of the tracker")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override max training iterations")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation during training")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Config extraction ----
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})
    exp_cfg = cfg.get("experiment", {})
    aug_cfg = cfg.get("augmentation", {})
    log_cfg = cfg.get("logging", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    cl_cfg = cfg.get("contrastive", {})
    model_cfg = cfg.get("model", {})

    dataset_root = args.dataset_root or data_cfg.get("root", "../data/waymo_open_dataset_v_2_0_1")
    output_dir = exp_cfg.get("output_dir", "outputs/tracker_ctvis")
    max_iters = args.max_iterations or train_cfg.get("max_iterations", 28000)
    backbone = model_cfg.get("backbone", "swin-large-mapillary")
    batch_size = cfg.get("dataloader", {}).get("batch_size", 8)
    num_workers = cfg.get("dataloader", {}).get("num_workers", 4)
    crop_size = tuple(aug_cfg.get("crop_size", [640, 640]))
    save_interval = ckpt_cfg.get("save_interval", 4000)
    keep_last = ckpt_cfg.get("keep_last", 0)
    print_interval = log_cfg.get("print_interval", 50)
    use_amp = train_cfg.get("amp", True)
    grad_clip = train_cfg.get("gradient_clip", 0.1)
    warmup_iters = train_cfg.get("warmup", {}).get("iterations", 500)

    # Contrastive config
    reid_weight = cl_cfg.get("reid_weight", 2.0)
    aux_reid_weight = cl_cfg.get("aux_reid_weight", 1.0)
    num_negatives = cl_cfg.get("num_negatives", 199)
    clip_length = cl_cfg.get("clip_length", 5)

    # Per-class anchor weights.
    arw_cfg = cl_cfg.get("anchor_reweighting", {})
    class_weights = compute_class_weights(
        WOD_TRAINING_THING_INSTANCE_COUNTS,
        clamp=(
            float(arw_cfg.get("clamp_min", 0.5)),
            float(arw_cfg.get("clamp_max", 4.0)),
        ),
    )

    # Training-time memory bank (positives for consistent training).
    tmb_cfg = cfg.get("train_memory_bank", {})
    train_memory_bank = TrainMemoryBank(
        beta_min=float(tmb_cfg.get("beta_min", 0.0)),
        beta_max=float(tmb_cfg.get("beta_max", 0.8)),
        maximum_cache=int(tmb_cfg.get("maximum_cache", 10)),
    )
    positive_momentum_prob = float(tmb_cfg.get("positive_momentum_prob", 0.5))

    # Segmenter checkpoint to start from
    from_checkpoint = model_cfg.get("from_checkpoint",
                                     "checkpoints/segmenter.pth")

    # Cosine LR schedule with linear warm-up.
    sched_cfg = train_cfg.get("lr_scheduler", {})
    cosine_eta_min = sched_cfg.get("eta_min", 1.0e-6)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ---- Dataset ----
    cameras = data_cfg.get("cameras", [1])
    print(f"Loading video dataset from {dataset_root} ...")
    train_ds = WODVideoDataset(
        dataset_root=dataset_root,
        split="training",
        cameras=cameras,
        crop_size=crop_size,
        min_scale=aug_cfg.get("min_scale", 0.75),
        max_scale=aug_cfg.get("max_scale", 1.25),
        color_jitter=aug_cfg.get("color_jitter", True),
        clip_length=clip_length,
    )
    print(f"  Training clips: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=video_collate_fn,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    # ---- Model ----
    print(f"Building model (backbone={backbone}) ...")
    model = build_model(backbone=backbone, num_classes=NUM_CLASSES, pretrained=False)

    # Load the segmenter checkpoint
    if not args.resume:
        print(f"Loading segmenter checkpoint: {from_checkpoint}")
        ckpt = torch.load(from_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

    model = model.to(device)

    # Freeze the backbone
    n_frozen = freeze_backbone(model)
    print(f"  Backbone frozen: {n_frozen:,} params")

    # Count trainable decoder params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Decoder trainable: {n_trainable:,} params")

    # ---- Embedding head ----
    embed_dim = model_cfg.get("embedding_dim", 256)
    embed_head = EmbeddingHead(
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
    ).to(device)
    n_embed = sum(p.numel() for p in embed_head.parameters())
    print(f"  Embedding head: {n_embed:,} params")

    # ---- Optimizer ----
    optimizer = build_optimizer(model, embed_head, cfg)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")

    # ---- Resume ----
    start_iter = 0
    best_metric = 0.0
    if args.resume:
        ckpt = load_latest_checkpoint(output_dir, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            embed_head.load_state_dict(ckpt["embed_head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_iter = ckpt.get("iteration", 0)
            best_metric = ckpt.get("best_metric", 0.0)
            # Re-freeze backbone after loading
            freeze_backbone(model)
            print(f"Resumed at iteration {start_iter}")
        else:
            print("No checkpoint found, starting from scratch.")
            print(f"Loading segmenter checkpoint: {from_checkpoint}")
            ckpt = torch.load(from_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])

    # ---- Training loop ----
    print(f"\nStarting training: {start_iter} → {max_iters} iterations")
    print(f"  Batch size: {batch_size} clips × T={clip_length} frames = {batch_size * clip_length} frames/iter")
    print(f"  Crop size:  {crop_size}")
    print(f"  Output dir: {output_dir}")
    print(f"  Loss weights: CE=2 Mask=5 Dice=5 | InfoNCE={reid_weight} auxMSE={aux_reid_weight}")
    print(f"  Contrastive: every frame as anchor (num_negatives={num_negatives})")
    print(f"  Train memory bank: β=[{train_memory_bank.beta_min},"
          f"{train_memory_bank.beta_max}] positive_mix={positive_momentum_prob:.2f}")
    print(f"  Anchor weights: " + ", ".join(
        f"{cid}:{w:.2f}" for cid, w in sorted(class_weights.items())
    ))
    print(f"  LR schedule: cosine — warmup={warmup_iters}, eta_min={cosine_eta_min:.1e}")
    print(f"  LR: decoder={optimizer.param_groups[0]['lr']:.1e}  embed={optimizer.param_groups[1]['lr']:.1e}")
    print()

    # Set initial_lr to the base config values (not the current decayed LR):
    # on resume the schedule is recomputed from the base.
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    base_lrs = [opt_cfg.get("lr_decoder", 2e-5), opt_cfg.get("lr_embed", 2e-4)]
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["initial_lr"] = base_lr

    logger = TrainingLogger(
        output_dir=output_dir,
        columns=[
            "iteration", "loss_total", "loss_m2f", "loss_reid", "loss_aux_reid",
            "lr_decoder", "lr_embed", "it_per_sec",
        ],
        resume_iteration=start_iter,
    )

    model.train()
    embed_head.train()
    data_iter = iter(train_loader)
    iteration = start_iter
    t_start = time.time()
    accum = {"total": 0.0, "m2f": 0.0, "reid": 0.0, "aux_reid": 0.0, "n": 0}

    while iteration < max_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # LR scheduling
        for pg in optimizer.param_groups:
            scale = get_lr_scale_cosine(
                iteration, warmup_iters, max_iters,
                base_lr=pg["initial_lr"], eta_min=cosine_eta_min,
            )
            pg["lr"] = pg["initial_lr"] * scale

        optimizer.zero_grad(set_to_none=True)

        pixel_values = batch["pixel_values"].to(device)  # (B*T, 3, H, W)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]
        global_instance_ids = batch["global_instance_ids"]
        B = batch["batch_size"]
        T = batch["clip_length"]

        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            # Forward pass (computes the segmentation loss internally)
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            m2f_loss = outputs.loss

            # Query embeddings (after the decoder's final layer normalization)
            # through the embedding head.
            query_embeds = extract_query_embeddings(model, outputs)  # (B*T, Q, C)
            reid_embeds = embed_head(query_embeds)  # (B*T, Q, C)

            # Get matching indices for contrastive loss
            indices = hungarian_indices(model, outputs, mask_labels, class_labels, device)

            # Build the contrastive items and compute the loss.
            cl_items = build_multi_ref_contrastive_items(
                reid_embeds, indices, global_instance_ids, class_labels,
                clip_length=T, batch_size=B,
                num_negatives=num_negatives,
                train_memory_bank=train_memory_bank,
                positive_momentum_prob=positive_momentum_prob,
            )
            cl_losses = contrastive_loss(
                cl_items, reid_embeds, class_weights=class_weights,
            )

            # Total loss
            loss_reid = cl_losses["loss_reid"] * reid_weight
            loss_aux_reid = cl_losses["loss_aux_reid"] * aux_reid_weight
            total_loss = m2f_loss + loss_reid + loss_aux_reid

        scaler.scale(total_loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            all_params = list(model.parameters()) + list(embed_head.parameters())
            nn.utils.clip_grad_norm_(all_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()

        iteration += 1
        accum["total"] += total_loss.item()
        accum["m2f"] += m2f_loss.item()
        accum["reid"] += loss_reid.item()
        accum["aux_reid"] += loss_aux_reid.item()
        accum["n"] += 1

        # Logging
        if iteration % print_interval == 0:
            elapsed = time.time() - t_start
            n = accum["n"]
            avg_total = accum["total"] / n
            avg_m2f = accum["m2f"] / n
            avg_reid = accum["reid"] / n
            avg_aux = accum["aux_reid"] / n
            its = iteration / elapsed
            lr_dec = optimizer.param_groups[0]["lr"]
            lr_emb = optimizer.param_groups[1]["lr"]

            print(
                f"[{iteration:6d}/{max_iters}]  "
                f"loss={avg_total:.4f} (m2f={avg_m2f:.4f} reid={avg_reid:.4f} "
                f"aux={avg_aux:.4f})  "
                f"lr=({lr_dec:.1e},{lr_emb:.1e})  {its:.2f} it/s"
            )
            logger.log(
                iteration=iteration,
                loss_total=f"{avg_total:.6f}",
                loss_m2f=f"{avg_m2f:.6f}",
                loss_reid=f"{avg_reid:.6f}",
                loss_aux_reid=f"{avg_aux:.6f}",
                lr_decoder=f"{lr_dec:.2e}",
                lr_embed=f"{lr_emb:.2e}",
                it_per_sec=f"{its:.4f}",
            )
            accum = {"total": 0.0, "m2f": 0.0, "reid": 0.0, "aux_reid": 0.0, "n": 0}

        # Checkpoint + Evaluation
        if iteration % save_interval == 0 or iteration == max_iters:
            state = _build_state(
                model, embed_head, optimizer, scaler, iteration, best_metric,
            )
            ckpt_path = save_checkpoint(state, output_dir, iteration, keep_last)
            print(f"Saved checkpoint: {ckpt_path}")

            # Evaluation (PQ + tracking).
            # Wrapped so that an eval failure (OOM, file IO, ...) does not stop
            # training; the checkpoint is already saved above. Cached GPU
            # memory is released before and after the full-resolution eval.
            if not args.no_eval:
                eval_samples = cfg.get("evaluation", {}).get("num_samples", 200)
                eval_n_seq = cfg.get("evaluation", {}).get("tracking_sequences", 3)
                eval_seq_indices = cfg.get("evaluation", {}).get("sequence_indices", None)

                print(f"\n--- Evaluation at iteration {iteration} ---")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    # 1. PQ on a validation subset
                    metrics = evaluate_pq(model, dataset_root, cfg, device,
                                          n_samples=eval_samples)
                    print(f"  PQ:   {metrics['PQ']:.4f}  (thing={metrics['PQ_thing']:.4f}  stuff={metrics['PQ_stuff']:.4f})")
                    print(f"  mIoU: {metrics['mIoU']:.4f}")

                    # 2. Tracking on selected validation sequences
                    n_eval = len(eval_seq_indices) if eval_seq_indices else eval_n_seq
                    print(f"  Tracking eval ({n_eval} sequences) ...")
                    eval_cfg = cfg.get("evaluation", {})
                    tracking = evaluate_tracking(
                        model, embed_head, dataset_root, cfg, device,
                        n_sequences=eval_n_seq,
                        sequence_indices=eval_seq_indices,
                        obj_threshold=eval_cfg.get("obj_threshold", 0.5),
                        mask_threshold=eval_cfg.get("mask_threshold", 0.5),
                        overlap_threshold=eval_cfg.get("overlap_threshold", 0.8),
                        match_threshold=eval_cfg.get("match_threshold", 0.6),
                        init_score_threshold=eval_cfg.get("init_score_threshold", 0.3),
                        num_dead_frames=eval_cfg.get("num_dead_frames", 10),
                    )
                    print(f"  Tracker: wSTQ={tracking['STQ']:.4f}  "
                          f"wAQ={tracking['AQ']:.4f}  "
                          f"wSQ={tracking['SQ']:.4f}")

                    # Save all eval results
                    eval_results = {
                        "iteration": iteration,
                        **metrics,
                        "tracking": tracking,
                    }
                    eval_path = os.path.join(output_dir, f"eval_{iteration:07d}.json")
                    with open(eval_path, "w") as f:
                        json.dump(eval_results, f, indent=2)

                    # best.pth: highest in-training tracking wSTQ so far
                    tracking_stq = tracking["STQ"]
                    if tracking_stq > best_metric:
                        best_metric = tracking_stq
                        bp = save_best(
                            _build_state(model, embed_head, optimizer, scaler,
                                         iteration, best_metric),
                            output_dir,
                        )
                        print(f"  ** New best wSTQ={best_metric:.4f} — saved {bp}")
                except (RuntimeError, MemoryError) as e:
                    # OOM, CUDA assert, dataset IO, etc.: keep training.
                    print(f"\n  !! Inline eval failed at iter {iteration}: "
                          f"{type(e).__name__}: {e}")
                    print("  !! Training continues; milestone checkpoint is saved.")
                    print("  !! Stack trace:")
                    traceback.print_exc()
                finally:
                    # Release whatever the eval allocated before next train step.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print()

    logger.close()
    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed / 60:.1f} min  (best wSTQ={best_metric:.4f})")


if __name__ == "__main__":
    main()
