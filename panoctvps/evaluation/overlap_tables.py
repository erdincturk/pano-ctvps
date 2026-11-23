#!/usr/bin/env python3
"""Overlap tables: exact scoring of many identity assignments on one cache of segmenter outputs.

Used by the tracker cells (scripts/experiments/run_tracker_cells.py), the linker evaluations from the tables
(scripts/experiments/eval_linker_tables.py, scripts/experiments/eval_linker_alternatives.py) and the held-out check.

The trackers and the linker change only identities, never masks or classes. STQ's association term AQ is a sum over
pairs (ground-truth id, predicted id) of pixel weights, so it can be rebuilt from per-frame tables that do not depend
on the identities:
  tpa   : for every frame, ground-truth thing id g and predicted segment (local instance k, predicted class s):
          the sum of pixel weights (and the pixel count) on the ground-truth thing pixels where they overlap
  psize : for every frame and predicted segment (k, s) of a thing class: the sum of pixel weights (and pixel count)
          over STQ's prediction mask (predicted thing, not on a ground-truth crowd pixel, instance id > 0)
  gsize : for every frame and ground-truth id g: the sum of pixel weights (and pixel count) on its thing pixels
The SQ term does not depend on identities; it is stored per scene (pixel-weighted) and per scene and camera
(unweighted) as per-class IoU sums. A predicted id of an assignment is (class s, identity of segment k); segments
without an identity (track id 0) are not predicted instances, exactly as in STQuality.update.

`build` writes one npz per scene (scripts/experiments/build_overlap_tables.py). `score_scene` turns an assignment of one scene
into its AQ numerators and SQ sums; `merge` pools scenes as scripts/pipeline/eval_tracker_cached.py does: aggregate
(pixel-weighted, panoramic merge per scene), per scene, per scene and camera (unweighted), per camera pooled over
scenes. The scores equal those of scripts/pipeline/eval_tracker_cached.py and scripts/pipeline/eval_linker_wstq.py to rounding.

Rule of the evaluator that the tables follow: when a camera frame has no ground-truth pixel of a thing class,
STQuality.update returns before it adds the predicted instances of that frame to their sizes, so predicted sizes of
such frames are not counted.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.data.constants import CAMERA_NAMES, NUM_CLASSES, THING_CLASSES
from panoctvps.data.lazy_gt import LazySequenceDataset
from panoctvps.evaluation import STQuality
from panoctvps.evaluation.tracking_utils import build_gt_global_instance

OFFSET = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0).offset
THING = np.array(sorted(THING_CLASSES))


def frame_order(npz):
    return sorted(range(len(npz["frame_records"])),
                  key=lambda i: (int(npz["frame_records"][i]["timestamp"]), int(npz["frame_records"][i]["camera_id"])))


def build(cache_dir: Path, out_dir: Path, dataset_root: str, split: str = "validation", scenes=None):
    """Writes <out_dir>/<scene>.npz for every cached scene (or the named scenes); existing files are kept."""
    gt = LazySequenceDataset(dataset_root, split)
    s2i = {f: i for i, f in enumerate(gt.segment_files)}
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    paths = sorted(Path(cache_dir).glob("*.npz"))
    if scenes:
        paths = [p for p in paths if p.stem in set(scenes)]
    for n, cp in enumerate(paths, 1):
        outp = out_dir / cp.name
        if outp.exists():
            continue
        npz = np.load(cp, allow_pickle=True)
        g = gt[s2i[str(npz["segment_file"])]]
        gbt = {(fr["timestamp"], c): d for fr in g["frames"] for c, d in fr["cameras"].items()}
        pw = g["pixel_weights"]
        T, P, G, F = [], [], [], []
        sq_w = {c: np.zeros((2, NUM_CLASSES)) for c in (0,)}
        sq_cam = defaultdict(lambda: np.zeros((2, NUM_CLASSES)))
        for fi in frame_order(npz):
            fr = npz["frame_records"][fi]
            ts, cam = int(fr["timestamp"]), int(fr["camera_id"])
            if (ts, cam) not in gbt:
                continue
            ps, pi = reconstruct_frame(npz, fi)[:2]
            gc = gbt[(ts, cam)]
            gs = gc["semantic_label"]
            gg = build_gt_global_instance(gc["instance_label"], gc["instance_mapping"])
            if ps.shape != gs.shape:
                continue
            F.append((fi, cam, ts))
            w = pw[cam].astype(np.float64).ravel()
            ps_, pi_, gs_, gg_ = ps.ravel().astype(np.int64), pi.ravel().astype(np.int64), gs.ravel().astype(np.int64), gg.ravel().astype(np.int64)
            valid = gs_ != 0
            # SQ sums (STQuality.update: inter_c = pred c & gt c & valid; union_c = pred c | (gt c & valid))
            NC = NUM_CLASSES
            pw_c = np.bincount(ps_, weights=w, minlength=NC)[:NC]; pu_c = np.bincount(ps_, minlength=NC)[:NC]
            gw_c = np.bincount(gs_[valid], weights=w[valid], minlength=NC)[:NC]
            gu_c = np.bincount(gs_[valid], minlength=NC)[:NC]
            eq = valid & (ps_ == gs_)
            iw_c = np.bincount(ps_[eq], weights=w[eq], minlength=NC)[:NC]; iu_c = np.bincount(ps_[eq], minlength=NC)[:NC]
            sq_w[0][0, 1:] += iw_c[1:]; sq_w[0][1, 1:] += (pw_c + gw_c - iw_c)[1:]
            sq_cam[cam][0, 1:] += iu_c[1:]; sq_cam[cam][1, 1:] += (pu_c + gu_c - iu_c)[1:]
            gthing = np.isin(gs_, THING)
            tm = gthing & valid & (gg_ > 0)
            if tm.any():
                gid = gs_[tm] * OFFSET + gg_[tm]
                pt = np.isin(ps_[tm], THING) & (pi_[tm] > 0)
                kk = np.where(pt, pi_[tm], 0); ss = np.where(pt, ps_[tm], 0)
                ww = w[tm]
                ug, ig = np.unique(gid, return_inverse=True)
                ig = ig.ravel()
                key = (ig.astype(np.int64) * 100000 + kk) * 64 + ss
                u, inv = np.unique(key, return_inverse=True)
                inv = inv.ravel()
                sw = np.bincount(inv, weights=ww); sc = np.bincount(inv)
                gi, rest = np.divmod(u, 100000 * 64)
                kq, sq_ = np.divmod(rest, 64)
                for a, b, c, x, y in zip(ug[gi].tolist(), kq.tolist(), sq_.tolist(), sw.tolist(), sc.tolist()):
                    if b > 0:
                        T.append((len(F) - 1, a, b, c, x, y))
                gw = np.bincount(ig, weights=ww); gcnt = np.bincount(ig)
                for a, x, y in zip(ug.tolist(), gw.tolist(), gcnt.tolist()):
                    G.append((len(F) - 1, a, x, y))
            crowd = gthing & valid & (gg_ == 0)
            pm = np.isin(ps_, THING) & ~crowd & (pi_ > 0)
            if pm.any():
                key = pi_[pm] * 64 + ps_[pm]
                u, inv = np.unique(key, return_inverse=True)
                inv = inv.ravel()
                pw_ = np.bincount(inv, weights=w[pm]); pc_ = np.bincount(inv)
                kq, sq_ = np.divmod(u, 64)
                for b, c, x, y in zip(kq.tolist(), sq_.tolist(), pw_.tolist(), pc_.tolist()):
                    P.append((len(F) - 1, b, c, x, y))
        cams = sorted(sq_cam)
        np.savez_compressed(
            outp, segment_name=str(npz["segment_name"]),
            frames=np.array(F, dtype=[("fi", "i4"), ("cam", "i1"), ("ts", "i8")]),
            tpa=np.array(T, dtype=[("f", "i4"), ("g", "i8"), ("k", "i4"), ("s", "i2"), ("w", "f8"), ("u", "i8")]),
            psize=np.array(P, dtype=[("f", "i4"), ("k", "i4"), ("s", "i2"), ("w", "f8"), ("u", "i8")]),
            gsize=np.array(G, dtype=[("f", "i4"), ("g", "i8"), ("w", "f8"), ("u", "i8")]),
            sq_weighted=sq_w[0], sq_cams=np.array(cams), sq_cam=np.stack([sq_cam[c] for c in cams]))
        print(f"[{n}] {str(npz['segment_name'])[:34]} frames={len(F)} tpa={len(T)} ({time.time()-t0:.0f}s)", flush=True)


def _sq(iou, uni):
    valid = (uni > 0) & (np.arange(NUM_CLASSES) != 0)
    return float((iou[valid] / uni[valid]).mean()) if valid.any() else 0.0


def _aq(tpa_g, tpa_p, tpa_v, ps_p, ps_v, gs_g, gs_v):
    """AQ numerator and N from arrays (pred ids > 0 only in tpa/psize)."""
    gsize = defaultdict(float)
    for a, v in zip(gs_g.tolist(), gs_v.tolist()):
        gsize[a] += v
    psize = defaultdict(float)
    for a, v in zip(ps_p.tolist(), ps_v.tolist()):
        psize[a] += v
    conf = defaultdict(float)
    for a, b, v in zip(tpa_g.tolist(), tpa_p.tolist(), tpa_v.tolist()):
        conf[(a, b)] += v
    num = defaultdict(float)
    for (a, b), t in conf.items():
        num[a] += t * (t / max(t + (gsize[a] - t) + (psize.get(b, 0.0) - t), 1e-10))
    s = 0.0
    per = {}
    for a, gsz in gsize.items():
        if gsz == 0:
            continue
        v = num.get(a, 0.0) / max(gsz, 1e-10)
        per[a] = v
        s += v
    return s, len(gsize), per


def score_scene(tab, ident_agg, ident_cam):
    """ident_agg / ident_cam: functions (frame_row_index, k, s) -> predicted instance id (0 = none), vectorised over
    arrays; ident_agg for the pixel-weighted aggregate (camera-namespaced or global ids), ident_cam for the
    per-camera STQ (ids only need to be unique within the camera)."""
    fr = tab["frames"]
    T, P, G = tab["tpa"], tab["psize"], tab["gsize"]
    # STQuality.update returns before it adds the predicted-instance sizes when a frame has no
    # ground-truth thing pixel (panoctvps/evaluation/stq.py, lines 160-162). Predicted sizes of such frames are
    # therefore not counted; the frames with a ground-truth thing pixel are exactly those with gsize rows.
    p_in = np.isin(P["f"], np.unique(G["f"]))
    out = {}
    pa = ident_agg(T["f"], T["k"], T["s"]); pp = ident_agg(P["f"], P["k"], P["s"])
    mt, mp = pa > 0, (pp > 0) & p_in
    tp_ids = T["s"][mt].astype(np.int64) * OFFSET + pa[mt]
    ps_ids = P["s"][mp].astype(np.int64) * OFFSET + pp[mp]
    s, n, per = _aq(T["g"][mt], tp_ids, T["w"][mt], ps_ids, P["w"][mp], G["g"], G["w"])
    out["agg"] = {"aq_num": s, "N": n, "per_gt": per, "iou": tab["sq_weighted"][0], "uni": tab["sq_weighted"][1],
                  "SQ": _sq(tab["sq_weighted"][0], tab["sq_weighted"][1])}
    out["cams"] = {}
    tc, pc, gcam = fr["cam"][T["f"]], fr["cam"][P["f"]], fr["cam"][G["f"]]
    ca = ident_cam(T["f"], T["k"], T["s"]); cpp = ident_cam(P["f"], P["k"], P["s"])
    for j, c in enumerate(tab["sq_cams"].tolist()):
        m1 = (tc == c) & (ca > 0); m2 = (pc == c) & (cpp > 0) & p_in; m3 = gcam == c
        s, n, _ = _aq(T["g"][m1], T["s"][m1].astype(np.int64) * OFFSET + ca[m1], T["u"][m1].astype(np.float64),
                      P["s"][m2].astype(np.int64) * OFFSET + cpp[m2], P["u"][m2].astype(np.float64),
                      G["g"][m3], G["u"][m3].astype(np.float64))
        iou, uni = tab["sq_cam"][j]
        out["cams"][int(c)] = {"aq_num": s, "N": n, "iou": iou, "uni": uni, "SQ": _sq(iou, uni)}
    return out


def merge(scenes):
    """scenes: list of (name, score_scene output). Returns aggregate, per scene, per scene and camera, per camera."""
    A_num = sum(o["agg"]["aq_num"] for _, o in scenes); A_n = sum(o["agg"]["N"] for _, o in scenes)
    A_sq = _sq(np.sum([o["agg"]["iou"] for _, o in scenes], 0), np.sum([o["agg"]["uni"] for _, o in scenes], 0))
    per_scene, per_cam_rows = [], []
    cam_acc = defaultdict(lambda: [0.0, 0, np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)])
    for name, o in scenes:
        aq = o["agg"]["aq_num"] / max(o["agg"]["N"], 1e-10)
        sq = o["agg"]["SQ"]
        per_scene.append({"scene": name, "wSTQ": float(np.sqrt(max(aq * sq, 0))), "wAQ": aq, "wSQ": sq,
                          "num_gt_instances": o["agg"]["N"], "aq_numerator": o["agg"]["aq_num"]})
        for c, d in o["cams"].items():
            caq = d["aq_num"] / max(d["N"], 1e-10)
            per_cam_rows.append({"scene": name, "camera": CAMERA_NAMES[c], "AQ": caq, "SQ": d["SQ"],
                                 "STQ": float(np.sqrt(max(caq * d["SQ"], 0))), "num_gt_instances": d["N"],
                                 "aq_numerator": d["aq_num"], "iou_sum": d["iou"].tolist(), "union_sum": d["uni"].tolist()})
            acc = cam_acc[c]
            acc[0] += d["aq_num"]; acc[1] += d["N"]; acc[2] += d["iou"]; acc[3] += d["uni"]
    aq = A_num / max(A_n, 1e-10)
    agg = {"wSTQ": float(np.sqrt(max(aq * A_sq, 0))), "wAQ": aq, "wSQ": A_sq, "num_gt_instances": A_n}
    per_camera = {}
    for c, (num, n, iou, uni) in sorted(cam_acc.items()):
        caq, csq = num / max(n, 1e-10), _sq(iou, uni)
        per_camera[CAMERA_NAMES[c]] = {"STQ": float(np.sqrt(max(caq * csq, 0))), "AQ": caq, "SQ": csq,
                                       "num_gt_instances": n}
    return agg, per_scene, per_cam_rows, per_camera


def gt_objects(tab):
    """Ground-truth thing objects of a scene, from its overlap tables (gsize: one row per ground-truth mask).

    Returns ({global id: majority class over its masks, ties broken by the class seen first in frame order},
    {(global id, camera): set of timestamps with a mask}).
    The id of a table row is class * OFFSET + global id (as in STQuality)."""
    fr = tab["frames"]
    cls_count = defaultdict(Counter)
    frames = defaultdict(set)
    for f, g in zip(tab["gsize"]["f"].tolist(), tab["gsize"]["g"].tolist()):   # frame order: (timestamp, camera)
        gid, cl = g % OFFSET, g // OFFSET
        cls_count[gid][cl] += 1
        frames[(gid, int(fr["cam"][f]))].add(int(fr["ts"][f]))
    major = {gid: c.most_common(1)[0][0] for gid, c in cls_count.items()}   # ties: the class seen first
    return major, frames
