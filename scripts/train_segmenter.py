#!/usr/bin/env python3
"""Panoptic fine-tuning of the Mask2Former segmenter on WOD PVPS.

The paper's segmenter is trained in two runs: 20K iterations on the FRONT
camera, then 80K iterations on all five cameras, warm-started from the first:

    python scripts/train_segmenter.py \
        --config configs/training/segmenter_mapillary_refined_front.yaml
    python scripts/train_segmenter.py \
        --config configs/training/segmenter_mapillary_refined.yaml \
        --resume-from outputs/segmenter_mapillary_refined_front/best.pth

Add --resume to continue an interrupted run from its latest checkpoint.

During training, best.pth is selected by an in-training PQ computed on the
first --eval-samples validation samples, centre-cropped to the training crop
size. The reported numbers are computed at full resolution with
scripts/eval_segmenter.py.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from panoctvps.data.training_dataset import WODMask2FormerDataset, collate_fn
from panoctvps.segmenter.mask2former import build_model, predict_panoptic
from panoctvps.evaluation import PanopticQuality, MeanIoU
from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES
from panoctvps.training.engine import (
    load_config, get_lr_scale, get_lr_scale_cosine, save_checkpoint, save_best,
    load_latest_checkpoint, TrainingLogger,
)


# =============================================================================
# Optimizer
# =============================================================================

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW with separate backbone/head LR groups."""
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    lr = opt_cfg.get("lr", 1e-4)
    wd = opt_cfg.get("weight_decay", 0.05)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "model.pixel_level_module.encoder" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=wd, betas=betas)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, dataset_root: str, cfg: dict, device: torch.device,
             n_samples: int = 200):
    """In-training validation: PQ + mIoU on the first `n_samples` validation
    samples, centre-cropped to the training crop size."""
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

    pq_metric   = PanopticQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES)
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

    pq_r   = pq_metric.result()
    miou_r = miou_metric.result()
    model.train()

    return {
        "PQ": pq_r["PQ"], "SQ": pq_r["SQ"], "RQ": pq_r["RQ"],
        "PQ_thing": pq_r["PQ_thing"], "PQ_stuff": pq_r["PQ_stuff"],
        "mIoU": miou_r["mIoU"],
        "num_images": n,
    }


def _build_state(model, optimizer, scaler, iteration, best_pq):
    """Build the checkpoint state dict."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_pq": best_pq,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Mask2Former on WOD PVPS")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", default=None,
                        help="Warm start: load model weights from a checkpoint "
                             "(optimizer, schedule and iteration count start fresh)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Override max training iterations")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation during training")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from config")
    parser.add_argument("--eval-samples", type=int, default=200,
                        help="In-training PQ is computed on this many validation "
                             "samples at the training crop (default 200)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Config extraction ----
    data_cfg    = cfg.get("dataset", {})
    train_cfg   = cfg.get("training", {})
    exp_cfg     = cfg.get("experiment", {})
    log_cfg     = cfg.get("logging", {})
    ckpt_cfg    = cfg.get("checkpoint", {})
    aug_cfg     = cfg.get("augmentation", {})

    dataset_root   = args.dataset_root or data_cfg.get("root", "../data/waymo_open_dataset_v_2_0_1")
    output_dir     = exp_cfg.get("output_dir", "outputs/segmenter")
    max_iters      = args.max_iterations or train_cfg.get("max_iterations", 80000)
    backbone       = cfg.get("model", {}).get("backbone", "swin-large-mapillary")
    batch_size     = cfg.get("dataloader", {}).get("batch_size", 8)
    num_workers    = cfg.get("dataloader", {}).get("num_workers", 0)
    persistent_w   = cfg.get("dataloader", {}).get("persistent_workers", True)
    crop_size      = tuple(aug_cfg.get("crop_size", [640, 640]))
    eval_interval  = cfg.get("evaluation", {}).get("interval", 5000)
    save_interval  = ckpt_cfg.get("save_interval", 5000)
    keep_last      = ckpt_cfg.get("keep_last", 3)
    print_interval = log_cfg.get("print_interval", 50)
    use_amp        = train_cfg.get("amp", True)
    grad_clip      = train_cfg.get("gradient_clip", 0.1)
    warmup_iters   = train_cfg.get("warmup", {}).get("iterations", 2000)
    sched_cfg      = train_cfg.get("lr_scheduler", {})
    sched_type     = sched_cfg.get("type", "cosine")   # "cosine" or "multistep"
    milestones     = sched_cfg.get("milestones", [])
    gamma          = sched_cfg.get("gamma", 0.1)
    eta_min        = sched_cfg.get("eta_min", 1e-6)
    base_lr        = train_cfg.get("optimizer", {}).get("lr", 1e-4)

    # Copy-paste augmentation
    cp_cfg         = aug_cfg.get("copy_paste", {})
    use_copy_paste = cp_cfg.get("enabled", True)
    cp_prob        = cp_cfg.get("paste_prob", 0.4)
    cp_max         = cp_cfg.get("max_paste", 3)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ---- Dataset ----
    print(f"Loading training dataset from {dataset_root} ...")
    train_ds = WODMask2FormerDataset(
        dataset_root=dataset_root,
        split="training",
        cameras=data_cfg.get("cameras", [1]),
        crop_size=crop_size,
        min_scale=aug_cfg.get("min_scale", 0.75),
        max_scale=aug_cfg.get("max_scale", 1.25),
        flip_prob=aug_cfg.get("flip_prob", 0.0),
        color_jitter=aug_cfg.get("color_jitter", True),
        copy_paste=use_copy_paste,
        copy_paste_prob=cp_prob,
        copy_paste_max=cp_max,
    )
    print(f"  Training samples: {len(train_ds)}")
    if use_copy_paste:
        print(f"  Copy-paste: prob={cp_prob}, max={cp_max}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,       # GB10 unified memory: pinning is a no-op
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
        persistent_workers=(persistent_w and num_workers > 0),
    )

    # ---- Model ----
    pretrained_path = cfg.get("model", {}).get("pretrained_path", None)
    use_pretrained = not args.resume and not args.resume_from
    print(f"Building model (backbone={backbone}, pretrained={use_pretrained}) ...")
    model = build_model(
        backbone=backbone, num_classes=NUM_CLASSES, pretrained=use_pretrained,
        pretrained_path=pretrained_path,
    )
    model = model.to(device)

    # ---- Optimizer ----
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")

    # ---- Resume ----
    start_iter = 0
    best_pq = 0.0
    if args.resume:
        ckpt = load_latest_checkpoint(output_dir, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_iter = ckpt.get("iteration", 0)
            best_pq = ckpt.get("best_pq", 0.0)
            print(f"Resumed at iteration {start_iter} (best_pq={best_pq:.4f})")
        else:
            print("No checkpoint found, starting from scratch.")
    elif args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        src_iter = ckpt.get("iteration", "?")
        src_pq = ckpt.get("best_pq", 0.0)
        print(f"Loaded weights from {args.resume_from} (iter={src_iter}, pq={src_pq:.4f})")
        print(f"  Starting fresh training (optimizer reset, iteration=0)")

    # ---- Training loop ----
    print(f"\nStarting training: {start_iter} -> {max_iters} iterations")
    print(f"  Batch size:    {batch_size}")
    print(f"  Crop size:     {crop_size}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Eval interval: {eval_interval}")
    print()

    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    # CSV logging
    logger = TrainingLogger(
        output_dir=output_dir,
        columns=["iteration", "loss", "lr", "it_per_sec"],
        resume_iteration=start_iter,
    )

    model.train()
    data_iter  = iter(train_loader)
    iteration  = start_iter
    t_start    = time.time()
    loss_accum = 0.0
    loss_count = 0

    while iteration < max_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # LR scaling
        if sched_type == "cosine":
            scale = get_lr_scale_cosine(
                iteration, warmup_iters, max_iters, base_lr, eta_min,
            )
        else:
            scale = get_lr_scale(iteration, warmup_iters, milestones, gamma)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * scale

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)

        pixel_values = batch["pixel_values"].to(device)
        mask_labels  = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        iteration += 1
        loss_accum += loss.item()
        loss_count += 1

        # Logging
        if iteration % print_interval == 0:
            elapsed = time.time() - t_start
            avg_loss = loss_accum / loss_count
            its = iteration / elapsed
            lr_now = optimizer.param_groups[1]["lr"]
            print(f"[{iteration:6d}/{max_iters}]  loss={avg_loss:.4f}  lr={lr_now:.2e}  {its:.2f} it/s")
            logger.log(
                iteration=iteration,
                loss=f"{avg_loss:.6f}",
                lr=f"{lr_now:.2e}",
                it_per_sec=f"{its:.4f}",
            )
            loss_accum = 0.0
            loss_count = 0

        # Evaluation
        if not args.no_eval and iteration % eval_interval == 0:
            print(f"\n--- Evaluation at iteration {iteration} ---")
            metrics = evaluate(model, dataset_root, cfg, device, n_samples=args.eval_samples)
            print(f"  PQ:   {metrics['PQ']:.4f}  (thing={metrics['PQ_thing']:.4f}  stuff={metrics['PQ_stuff']:.4f})")
            print(f"  SQ:   {metrics['SQ']:.4f}   RQ: {metrics['RQ']:.4f}")
            print(f"  mIoU: {metrics['mIoU']:.4f}")

            # Save eval results to JSON
            eval_path = os.path.join(output_dir, f"eval_{iteration:07d}.json")
            with open(eval_path, "w") as f:
                json.dump({"iteration": iteration, **metrics}, f, indent=2)

            if metrics["PQ"] > best_pq:
                best_pq = metrics["PQ"]
                bp = save_best(_build_state(model, optimizer, scaler, iteration, best_pq), output_dir)
                print(f"  ** New best PQ={best_pq:.4f} -- saved {bp}")
            print()

        # Checkpoint
        if iteration % save_interval == 0 or iteration == max_iters:
            state = _build_state(model, optimizer, scaler, iteration, best_pq)
            ckpt_path = save_checkpoint(state, output_dir, iteration, keep_last)
            print(f"Saved checkpoint: {ckpt_path}")

    logger.close()
    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min  (best PQ={best_pq:.4f})")


if __name__ == "__main__":
    main()
