"""train_adapter.py — train the appearance adapter psi of the linker.

Trains panoctvps/linker/adapter.EmbeddingAdapter (residual linear map, U and b
zero-initialised) on the detection pairs of the training split
(scripts/build_pairs.py) with a symmetric InfoNCE loss (temperature 0.07),
AdamW (lr 1e-3, weight decay 1e-4), batch 1024, up to 30 epochs. Pairs in
which a detection has an IoU below 0.3 with its ground-truth instance, and
pairs without an instance on one side, are discarded. Every tenth training
scene (sorted) is held out for early stopping: the epoch with the highest
holdout AUC of the adapted cosine is kept (patience 5 epochs).

Writes <out-dir>/adapter.pt and <out-dir>/adapter_training.json.

Usage:
  python scripts/train_adapter.py --pairs-dir outputs/linker/pairs_train \
      --cache-dir outputs/tracker_ctvis_train_cache --out-dir outputs/linker
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from panoctvps.linker.adapter import EmbeddingAdapter, info_nce_pair_loss

REPO = Path(__file__).resolve().parents[1]

GT_IOU_MIN = 0.3
HOLDOUT_EVERY = 10          # every 10th scene (sorted) -> holdout


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based ROC-AUC with midrank tie handling."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    both = np.concatenate([pos, neg])
    order = both.argsort()
    ranks = np.empty(len(both))
    sv = both[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def load_split(pairs_dir: Path, cache_dir: Path):
    """Gather (embA, embB, label, gidA, gidB, scene_idx) over all scenes, after
    the label filter (label >= 0, gt_iou >= GT_IOU_MIN on both sides)."""
    eA, eB, lab, gA, gB, sc = [], [], [], [], [], []
    paths = sorted(pairs_dir.glob("*.npz"))
    for s_idx, p in enumerate(paths):
        pairs = np.load(p)["pairs"]
        if len(pairs) == 0:
            continue
        keep = pairs["label"] >= 0
        keep &= ((pairs["label"] != 1) | ((pairs["gt_iouA"] >= GT_IOU_MIN) &
                                          (pairs["gt_iouB"] >= GT_IOU_MIN)))
        keep &= ((pairs["label"] != 0) | ((pairs["gt_iouA"] >= GT_IOU_MIN) &
                                          (pairs["gt_iouB"] >= GT_IOU_MIN)))
        pairs = pairs[keep]
        if len(pairs) == 0:
            continue
        emb = np.load(cache_dir / p.name, allow_pickle=True)["embeddings"]
        eA.append(emb[pairs["recA"]]); eB.append(emb[pairs["recB"]])
        lab.append(pairs["label"].astype(np.int64))
        gA.append(pairs["gidA"]); gB.append(pairs["gidB"])
        # gids are scene-local: offset so they never collide across scenes
        off = (s_idx + 1) * 10_000_000
        gA[-1] = np.where(pairs["gidA"] > 0, pairs["gidA"] + off, 0)
        gB[-1] = np.where(pairs["gidB"] > 0, pairs["gidB"] + off, 0)
        sc.append(np.full(len(pairs), s_idx, dtype=np.int32))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return (cat(eA).astype(np.float32), cat(eB).astype(np.float32), cat(lab),
            cat(gA), cat(gB), cat(sc))


@torch.no_grad()
def cosine_auc(model, eA, eB, lab, device, batch=8192) -> float:
    """AUC of the adapted cosine between pair sides, label 1 vs 0."""
    cs = []
    for i in range(0, len(eA), batch):
        za = model(torch.from_numpy(eA[i:i + batch]).to(device))
        zb = torch.from_numpy(eB[i:i + batch]).to(device)
        zb = model(zb)
        cs.append((za * zb).sum(-1).cpu().numpy())
    c = np.concatenate(cs)
    return auc(c[lab == 1], c[lab == 0])


def train(tr, ho, device, epochs, batch, lr, seed=0):
    """AdamW training; keeps the epoch with the best holdout AUC (patience 5)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    eA, eB, lab, gA, gB = tr
    model = EmbeddingAdapter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc, best_state, patience = -1.0, None, 0
    log = []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(eA))
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            za = model(torch.from_numpy(eA[idx]).to(device))
            zb = model(torch.from_numpy(eB[idx]).to(device))
            loss = info_nce_pair_loss(za, zb,
                                      torch.from_numpy(lab[idx]).to(device),
                                      torch.from_numpy(gA[idx]).to(device),
                                      torch.from_numpy(gB[idx]).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        model.eval()
        a = cosine_auc(model, *ho, device)
        flag = ""
        if a > best_auc:
            best_auc, patience = a, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            patience += 1
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "holdout_auc": a})
        print(f"    ep{ep:02d} loss={np.mean(losses):.4f} holdout_auc={a:.4f}{flag}", flush=True)
        if patience >= 5:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc, log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", type=Path, default=REPO / "outputs/linker/pairs_train")
    ap.add_argument("--cache-dir", type=Path, default=REPO / "outputs/tracker_ctvis_train_cache")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/linker")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    t0 = time.time()
    eA, eB, lab, gA, gB, sc = load_split(args.pairs_dir, args.cache_dir)
    ho_mask = (sc % HOLDOUT_EVERY) == 0
    tr = (eA[~ho_mask], eB[~ho_mask], lab[~ho_mask], gA[~ho_mask], gB[~ho_mask])
    ho = (eA[ho_mask], eB[ho_mask], lab[ho_mask])
    print(f"train pairs: {len(tr[0])} (pos={int((tr[2]==1).sum())}) | "
          f"holdout: {len(ho[0])} (pos={int((ho[2]==1).sum())}) | load {time.time()-t0:.0f}s")

    model, ho_auc, log = train(tr, ho, device, args.epochs, args.batch, args.lr)
    n_par = sum(p.numel() for p in model.parameters())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "adapter.pt"
    model.cpu().save(ckpt)
    report = {"n_train_pairs": int(len(tr[0])), "n_holdout_pairs": int(len(ho[0])),
              "params": n_par, "best_holdout_auc": ho_auc, "epochs": log,
              "settings": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                           "weight_decay": 1e-4, "temperature": 0.07,
                           "gt_iou_min": GT_IOU_MIN, "holdout_every": HOLDOUT_EVERY}}
    out = args.out_dir / "adapter_training.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"params={n_par} -> {ckpt}\nsaved {out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
