"""Causal (online) form of the linker, evaluated on the cached validation predictions.

At frame t only frames <= t are used. A candidate pair (same proposal test as
panoctvps/linker/proposer.py) becomes available at the first frame at which
both masks reach their overlap bands. At each of its shared frames its two
cues are updated as running statistics over the frames observed so far: S is
the running mean of the clipped panoramic IoU over the frames with evidence,
s_bar and s_tilde the running medians of the raw and adapted cosines over the
shared frames; the dominant classes are the running dominant classes of the
two tracks. Once at least k frames with evidence have been observed, the link
rule (Eq. 16) is checked at every shared frame, and the link is accepted at
the first frame at which it holds. Links accepted at the same frame are
inserted in decreasing order of S * rho(s_bar); merges are never undone and a
pair is not checked again once it has been inserted (merged or skipped). From
the frame of its merge onward, a track adopts the identity of the oldest track
in its component; frames already output are not relabelled.

The geometry is computed with the functions of panoctvps.linker (overlap
bands, band test, panoramic projection and clipping, thresholds, rho).

Building blocks, cached once per scene:

1. build_pair_table      per candidate pair: per shared frame clipped IoU
                         (NaN = no evidence), raw cosine, adapted cosine;
                         first band-touch frame; per track: frames, classes,
                         first frame. Also the per-scene linker's scored
                         candidates, its assignment and its merged links.
2. build_frame_tables    per evaluated camera frame: the weighted
                         (gt_id, pred_id) confusion rows and pred-size rows
                         that STQuality.update accumulates, keyed by the
                         tracker-only id (cam*100000 + tid), plus the
                         STQuality results of the tracker alone and of the
                         per-scene linker.
3. simulate_scene        the causal linker; returns the per-frame emitted ids
                         and the final components.

scene_aq re-labels the cached confusion rows with any run's ids and evaluates
the AQ formula of STQuality.result exactly (SQ is not changed by linking).
"""
from __future__ import annotations

import bisect
import collections
import json
import sys
import time
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from panoctvps.data.cache_io import decode_rle  # noqa: E402
from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES  # noqa: E402
from panoctvps.linker.assigner import (_rank_factor, assign_global_ids,  # noqa: E402
                                       link_rule, resolve_threshold)
from panoctvps.linker.link_eval import build_scene_struct, load_adapted_cosines  # noqa: E402
from panoctvps.linker.overlap import ADJACENT_PAIRS, calibration_from_arrays, overlap_by_partner  # noqa: E402
from panoctvps.linker.panorama import (build_overlap_pano_mask, build_panorama_lut,  # noqa: E402
                                       project_mask_to_panorama_forward)
from panoctvps.linker.pipeline import DEFAULT_TID_FIELD as TID_FIELD  # noqa: E402
from panoctvps.linker.proposer import PARTNERS, SUPERCLASS, _in_overlap, propose_candidates  # noqa: E402
from panoctvps.linker.scorer import _ext4, score_candidates  # noqa: E402

OFFSET = 256 ** 3                     # STQuality default offset (class * OFFSET + inst)
CAM_NS = 100_000                      # tracker-only id namespace: cam * CAM_NS + tid
DEFAULT_CONFIG = REPO / "configs/linker_deployed.json"

Tracklet = Tuple[int, int]
Pair = FrozenSet[Tracklet]


def load_config(path: Path = DEFAULT_CONFIG, key: str = "deployed") -> dict:
    """One named linker configuration (format of configs/linker_deployed.json)."""
    return json.loads(Path(path).read_text())[key]


# --------------------------------------------------------------------------- #
# 1. per-scene pair table (geometry computed once)
# --------------------------------------------------------------------------- #
def _load_cams(npz):
    wh = {}
    for fr in npz["frame_records"]:
        wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
    return calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)


def _adapted_embeddings(npz, adapter_path: Path) -> np.ndarray:
    import torch
    torch.set_num_threads(1)
    from panoctvps.linker.adapter import EmbeddingAdapter
    model = EmbeddingAdapter.load(adapter_path)
    with torch.no_grad():
        return model(torch.from_numpy(npz["embeddings"].astype(np.float32))).numpy()


def build_pair_table(cache_path: Path, adapter_path: Path, cosines_path: Path, cfg: dict) -> dict:
    """Per-frame pair statistics for one scene (see module docstring)."""
    npz = np.load(cache_path, allow_pickle=True)
    records = npz["records"]
    frame_records = npz["frame_records"]
    mask_rle = npz["mask_rle"]
    cams = _load_cams(npz)
    ovbp = overlap_by_partner(cams)

    ts_list = sorted({int(fr["timestamp"]) for fr in frame_records})
    ts_index = {ts: i for i, ts in enumerate(ts_list)}

    # ---- track bookkeeping (thing classes, track id > 0) ----
    det_index: Dict[tuple, int] = {}
    trk: Dict[Tracklet, dict] = {}
    for k, x in enumerate(records):
        if SUPERCLASS.get(int(x["class_id"])) is None:
            continue
        tid = int(x[TID_FIELD])
        if tid <= 0:
            continue
        t = (int(x["camera_id"]), tid)
        ti = ts_index[int(x["timestamp"])]
        det_index[(t[0], t[1], ti)] = k
        d = trk.setdefault(t, {"ts": [], "cls": [], "k": []})
        d["ts"].append(ti); d["cls"].append(int(x["class_id"])); d["k"].append(k)
    for t, d in trk.items():
        order = np.argsort(d["ts"], kind="stable")
        d["ts"] = np.asarray(d["ts"], np.int32)[order]
        d["cls"] = np.asarray(d["cls"], np.int16)[order]
        d["k"] = np.asarray(d["k"], np.int32)[order]
        d["born"] = int(d["ts"][0])

    # ---- per-scene linker: proposal -> scoring -> assignment ----
    cands = propose_candidates(npz, ovbp, tid_field=TID_FIELD)
    scored = score_candidates(npz, cams, cands, tid_field=TID_FIELD)
    idx = load_adapted_cosines(cosines_path)
    for pair, s in scored.items():
        s["cos_med_adapted"] = idx.get(pair)
    all_tr = set(trk)
    merged: List[Pair] = []
    deployed_assign = assign_global_ids(all_tr, scored, threshold=cfg["tau"],
                                        rank_key=cfg.get("rank_key"),
                                        veto_key=cfg.get("veto_key"),
                                        veto_min=cfg.get("veto_min", 0.0),
                                        veto_rescue=cfg.get("veto_rescue", 1.0),
                                        merged_out=merged)

    # ---- pass 1: band-touch frames per pair (proposal loop, per timestamp) ----
    t0 = time.perf_counter()
    frames_by_ts: Dict[int, list] = collections.defaultdict(list)
    for fr in frame_records:
        frames_by_ts[ts_index[int(fr["timestamp"])]].append(fr)
    first_prop: Dict[Pair, int] = {}
    for ti in range(len(ts_list)):
        cam_dets: Dict[int, list] = {}
        for fr in frames_by_ts[ti]:
            cam = int(fr["camera_id"])
            if cam not in PARTNERS:
                continue
            dets = []
            for k in range(int(fr["records_start"]), int(fr["records_end"])):
                rec = records[k]
                sc = SUPERCLASS.get(int(rec["class_id"]))
                if sc is None:
                    continue
                tid = int(rec[TID_FIELD])
                if tid <= 0:
                    continue
                mask = decode_rle(mask_rle[k])
                mem = {b: _in_overlap(mask, ovbp.get((cam, b))) for b in PARTNERS[cam]}
                dets.append((tid, sc, mem))
            cam_dets[cam] = dets
        for a, b in ADJACENT_PAIRS:
            if a not in cam_dets or b not in cam_dets:
                continue
            a_in = [(t, sc) for (t, sc, mem) in cam_dets[a] if mem.get(b)]
            b_in = [(t, sc) for (t, sc, mem) in cam_dets[b] if mem.get(a)]
            for ta, sa in a_in:
                for tb, sb in b_in:
                    if sa == sb:
                        pr = frozenset({(a, ta), (b, tb)})
                        if pr not in first_prop:
                            first_prop[pr] = ti
    assert set(first_prop) == cands, "per-frame proposal disagrees with propose_candidates"
    t_pass1 = time.perf_counter() - t0

    # ---- pass 2: per shared frame clipped IoU + cosines ----
    t0 = time.perf_counter()
    ext4 = {c: _ext4(cams[c]) for c in cams}
    luts = {c: build_panorama_lut(ext4[c], cams[c]["intr9"], (cams[c]["h"], cams[c]["w"]))
            for c in cams}
    ov_flat: Dict[Tuple[int, int], np.ndarray] = {}
    for a, b in ADJACENT_PAIRS:
        if a in cams and b in cams:
            ones_a = np.ones((cams[a]["h"], cams[a]["w"]), bool)
            ones_b = np.ones((cams[b]["h"], cams[b]["w"]), bool)
            ov = build_overlap_pano_mask(luts[a], luts[b], ones_a, ones_b).ravel()
            ov_flat[(a, b)] = ov; ov_flat[(b, a)] = ov

    emb = np.asarray(npz["embeddings"], np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    emb_ad = _adapted_embeddings(npz, adapter_path)
    emb_ad = emb_ad / np.maximum(np.linalg.norm(emb_ad, axis=1, keepdims=True), 1e-8)

    proj_cache: Dict[int, np.ndarray] = {}

    def proj(cam: int, k: int) -> np.ndarray:
        if k not in proj_cache:
            m = decode_rle(mask_rle[k])
            h, w = cams[cam]["h"], cams[cam]["w"]
            if m.shape != (h, w):
                m = m[:h, :w]
            pano = project_mask_to_panorama_forward(m, ext4[cam], cams[cam]["intr9"])
            proj_cache[k] = np.flatnonzero(pano)
        return proj_cache[k]

    pairs: List[dict] = []
    for pr in sorted(cands, key=lambda p: sorted(p)):
        (a, ta), (b, tb) = sorted(pr)
        ovf = ov_flat[(a, b)]
        shared = np.intersect1d(trk[(a, ta)]["ts"], trk[(b, tb)]["ts"])
        iou = np.full(len(shared), np.nan)
        cr = np.zeros(len(shared)); ca = np.zeros(len(shared))
        for j, ti in enumerate(shared):
            ka = det_index[(a, ta, int(ti))]; kb = det_index[(b, tb, int(ti))]
            ia, ib = proj(a, ka), proj(b, kb)
            ia = ia[ovf[ia]]; ib = ib[ovf[ib]]
            if ia.size or ib.size:
                inter = np.intersect1d(ia, ib, assume_unique=True).size
                union = ia.size + ib.size - inter
                iou[j] = inter / union if union else 0.0
            cr[j] = float(emb[ka] @ emb[kb]); ca[j] = float(emb_ad[ka] @ emb_ad[kb])
        s = scored[pr]
        assert len(shared) == s["n_frames"] and int((~np.isnan(iou)).sum()) == s["n_scored"]
        pairs.append({"a": (a, ta), "b": (b, tb), "first_prop": first_prop[pr],
                      "ts": shared.astype(np.int32), "iou": iou, "cos": cr, "cos_ad": ca})
    t_pass2 = time.perf_counter() - t0

    struct = build_scene_struct(npz, TID_FIELD)
    return {"segment": str(npz["segment_name"]), "segment_file": str(npz["segment_file"]),
            "n_ts": len(ts_list), "tracklets": trk, "pairs": pairs,
            "scored": scored, "deployed_assign": deployed_assign, "deployed_merged": merged,
            "struct": struct,
            "timing": {"pass1_proposal_s": t_pass1, "pass2_scoring_s": t_pass2}}


# --------------------------------------------------------------------------- #
# 2. per-scene frame tables (STQ confusion rows keyed by the tracker-only id)
# --------------------------------------------------------------------------- #
def _group_sum(keys: np.ndarray, w: np.ndarray):
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=w, minlength=len(u))


def build_frame_tables(cache_path: Path, dataset_root: str, deployed_assign: Dict[Tracklet, int]) -> dict:
    """Confusion / size rows per frame + STQuality of the tracker alone and of
    the per-scene linker."""
    from panoctvps.data.waymo_dataset import WODPVPSSequenceDataset
    from panoctvps.evaluation import STQuality
    from panoctvps.evaluation.tracking_utils import build_gt_global_instance
    from panoctvps.data.cache_io import reconstruct_frame

    npz = np.load(cache_path, allow_pickle=True)
    ds = WODPVPSSequenceDataset(dataset_root=dataset_root, split="validation")
    seg_file = str(npz["segment_file"])
    gt_seq = ds[ds.segment_files.index(seg_file)]
    gt_by_ts_cam = {}
    for frame in gt_seq["frames"]:
        for cam_id, cam_data in frame["cameras"].items():
            gt_by_ts_cam[(frame["timestamp"], cam_id)] = cam_data
    pixel_weights = gt_seq["pixel_weights"]
    ts_list = sorted({int(fr["timestamp"]) for fr in npz["frame_records"]})
    ts_index = {ts: i for i, ts in enumerate(ts_list)}

    is_thing = np.zeros(NUM_CLASSES, bool); is_thing[THING_CLASSES] = True
    floor_stq = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)
    dep_stq = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)
    records = npz["records"]
    conf_rows = []   # (ts_idx, cam, gt_id, pred_id_with_class, w)
    size_rows = []   # (ts_idx, cam, pred_id_with_class, w)
    gt_rows = []     # (gt_id, w)
    n_frames = 0
    for fr_idx in range(len(npz["frame_records"])):
        fr = npz["frame_records"][fr_idx]
        ts = int(fr["timestamp"]); cam = int(fr["camera_id"])
        if (ts, cam) not in gt_by_ts_cam:
            continue
        pred_sem, pred_inst = reconstruct_frame(npz, fr_idx)[:2]
        gt_cam = gt_by_ts_cam[(ts, cam)]
        gt_sem = gt_cam["semantic_label"]
        if pred_sem.shape != gt_sem.shape:
            continue
        gt_global = build_gt_global_instance(gt_cam["instance_label"], gt_cam["instance_mapping"])
        pw = pixel_weights[cam]
        rs, re = int(fr["records_start"]), int(fr["records_end"])
        floor_inst = np.zeros(pred_inst.shape, np.int64)
        dep_inst = np.zeros(pred_inst.shape, np.int64)
        for k in range(re - rs):
            tid = int(records[rs + k][TID_FIELD])
            if tid <= 0:
                continue
            sel = pred_inst == (k + 1)
            if not sel.any():
                continue
            floor_inst[sel] = cam * CAM_NS + tid
            dep_inst[sel] = deployed_assign[(cam, tid)]
        floor_stq.update(pred_semantic=pred_sem, pred_instance=floor_inst, gt_semantic=gt_sem,
                         gt_instance=gt_global, pixel_weights=pw)
        dep_stq.update(pred_semantic=pred_sem, pred_instance=dep_inst, gt_semantic=gt_sem,
                       gt_instance=gt_global, pixel_weights=pw)
        n_frames += 1

        # ---- AQ rows, replicating STQuality.update ----
        ps = pred_sem.ravel().astype(np.int32); pi = floor_inst.ravel()
        gs = gt_sem.ravel().astype(np.int32); gi = gt_global.ravel().astype(np.int64)
        w = pw.ravel().astype(np.float64)
        valid = gs != 0
        gt_thing = is_thing[gs]
        thing_mask = gt_thing & valid & (gi > 0)
        pred_thing = is_thing[ps]
        ti = ts_index[ts]
        if thing_mask.any():
            gt_ids = gs[thing_mask].astype(np.int64) * OFFSET + gi[thing_mask]
            pred_ids = np.where(pred_thing[thing_mask] & (pi[thing_mask] > 0),
                                ps[thing_mask].astype(np.int64) * OFFSET + pi[thing_mask], 0)
            wt = w[thing_mask]
            key = np.stack([gt_ids, pred_ids], 1)
            u, inv = np.unique(key, axis=0, return_inverse=True)
            ws = np.bincount(inv.ravel(), weights=wt, minlength=len(u))
            for (g, p), x in zip(u, ws):
                conf_rows.append((ti, cam, int(g), int(p), float(x)))
            ug, wg = _group_sum(gt_ids, wt)
            for g, x in zip(ug, wg):
                gt_rows.append((int(g), float(x)))
        # STQuality.update returns early (no pred-size accumulation) when the
        # frame has no GT thing pixels; replicate that.
        is_crowd = gt_thing & valid & (gi == 0)
        pm = pred_thing & ~is_crowd & (pi > 0)
        if thing_mask.any() and pm.any():
            pids = ps[pm].astype(np.int64) * OFFSET + pi[pm]
            up, wp = _group_sum(pids, w[pm])
            for p, x in zip(up, wp):
                size_rows.append((ti, cam, int(p), float(x)))

    conf = np.array(conf_rows, dtype=[("ts", "i4"), ("cam", "i1"), ("gt", "i8"), ("pred", "i8"), ("w", "f8")])
    size = np.array(size_rows, dtype=[("ts", "i4"), ("cam", "i1"), ("pred", "i8"), ("w", "f8")])
    gtr = np.array(gt_rows, dtype=[("gt", "i8"), ("w", "f8")])
    fres = floor_stq.result(); dres = dep_stq.result()
    return {"segment": str(npz["segment_name"]), "n_frames": n_frames,
            "conf": conf, "size": size, "gt": gtr,
            "sq_inter": floor_stq._iou_per_class.copy(), "sq_union": floor_stq._iou_union_per_class.copy(),
            "floor": {"STQ": fres["STQ"], "AQ": fres["AQ"], "SQ": fres["SQ"], "n_gt": fres["num_gt_instances"]},
            "deployed": {"STQ": dres["STQ"], "AQ": dres["AQ"], "SQ": dres["SQ"], "n_gt": dres["num_gt_instances"]}}


# --------------------------------------------------------------------------- #
# 3. fast AQ from cached rows
# --------------------------------------------------------------------------- #
def _decode_pred(pred: np.ndarray):
    cls = pred // OFFSET
    fl = pred % OFFSET
    return cls, (fl // CAM_NS).astype(np.int64), (fl % CAM_NS).astype(np.int64)


def relabel(pred: np.ndarray, ts: np.ndarray, id_fn) -> np.ndarray:
    """Map tracker-only-id predictions to run ids. id_fn(ts_idx, cam, tid) -> int (>0)."""
    cls, cam, tid = _decode_pred(pred)
    out = np.zeros_like(pred)
    nz = pred > 0
    keys = np.stack([ts[nz], cam[nz], tid[nz]], 1)
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    gids = np.array([id_fn(int(t), int(c), int(d)) for t, c, d in u], dtype=np.int64)
    out[nz] = cls[nz] * OFFSET + gids[inv.ravel()]
    return out


def scene_aq(ft: dict, id_fn) -> Tuple[float, int]:
    """AQ numerator-weighted mean for one scene (== STQuality.result()['AQ'])."""
    conf, size, gtr = ft["conf"], ft["size"], ft["gt"]
    cpred = relabel(conf["pred"], conf["ts"], id_fn)
    spred = relabel(size["pred"], size["ts"], id_fn)
    # aggregate over frames
    key = np.stack([conf["gt"], cpred], 1)
    u, inv = np.unique(key, axis=0, return_inverse=True)
    tpa = np.bincount(inv.ravel(), weights=conf["w"], minlength=len(u))
    up, wp = _group_sum(spred, size["w"])
    psize = dict(zip(up.tolist(), wp.tolist()))
    ug, wg = _group_sum(gtr["gt"], gtr["w"])
    gsize = dict(zip(ug.tolist(), wg.tolist()))
    score = collections.defaultdict(float)
    for (g, p), t in zip(u.tolist(), tpa.tolist()):
        if p == 0:
            continue
        gsz = gsize[g]
        fna = gsz - t
        fpa = psize.get(p, 0.0) - t
        score[g] += t * (t / max(t + fna + fpa, 1e-10))
    n_gt = len(gsize)
    aq_sum = sum(score[g] / max(gsize[g], 1e-10) for g in gsize)
    return aq_sum / n_gt if n_gt else 0.0, n_gt


def aggregate(frame_tables: List[dict], aq_per_scene: List[Tuple[float, int]]) -> dict:
    total_i = sum(ft["sq_inter"] for ft in frame_tables)
    total_u = sum(ft["sq_union"] for ft in frame_tables)
    valid = (total_u > 0) & (np.arange(NUM_CLASSES) != 0)
    sq = float((total_i[valid] / total_u[valid]).mean())
    num = sum(a * n for a, n in aq_per_scene); den = sum(n for _, n in aq_per_scene)
    aq = num / den
    return {"wSTQ": float(np.sqrt(aq * sq)), "wAQ": float(aq), "wSQ": sq}


# --------------------------------------------------------------------------- #
# 4. causal simulation
# --------------------------------------------------------------------------- #
class _RunningMedian:
    __slots__ = ("v",)

    def __init__(self):
        self.v: List[float] = []

    def add(self, x: float) -> None:
        bisect.insort(self.v, x)

    def median(self) -> Optional[float]:
        n = len(self.v)
        if n == 0:
            return None
        m = n // 2
        return self.v[m] if n % 2 else 0.5 * (self.v[m - 1] + self.v[m])


def _rule(S: float, cos_ad: Optional[float], cls_a: int, cls_b: int, a: Tracklet, b: Tracklet,
          cfg: dict) -> bool:
    bar = resolve_threshold(cfg["tau"], a[0], b[0], SUPERCLASS.get(cls_a))
    veto = cfg.get("veto_key") is not None
    return link_rule(S, cls_a == cls_b, bar, veto_cos=cos_ad if veto else None,
                     veto_min=cfg.get("veto_min") if veto else None,
                     veto_rescue=cfg.get("veto_rescue"))


def simulate_scene(pt: dict, k: int, cfg: dict) -> dict:
    """Run the causal linker over one scene with minimum evidence k.

    Returns the emitted ids {(ts_idx, cam, tid): id}, the final components
    {(cam, tid): id} (for the link counts) and the time of the decision step
    per frame.
    """
    trk = pt["tracklets"]; pairs = pt["pairs"]; n_ts = pt["n_ts"]
    # tracks present per frame; pairs with a shared frame per frame
    trk_at: List[list] = [[] for _ in range(n_ts)]
    for t, d in trk.items():
        for j, ti in enumerate(d["ts"]):
            trk_at[int(ti)].append((t, int(d["cls"][j])))
    pair_at: List[list] = [[] for _ in range(n_ts)]
    for i, p in enumerate(pairs):
        for j, ti in enumerate(p["ts"]):
            pair_at[int(ti)].append((i, j))

    # union-find with camera sets and identity (earliest-born member)
    parent: Dict[Tracklet, Tracklet] = {t: t for t in trk}
    cams: Dict[Tracklet, Set[int]] = {t: {t[0]} for t in trk}
    ident: Dict[Tracklet, tuple] = {t: (trk[t]["born"], t[0], t[1]) for t in trk}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    cls_cnt: Dict[Tracklet, collections.Counter] = {t: collections.Counter() for t in trk}
    dom: Dict[Tracklet, int] = {}
    n_ok = np.zeros(len(pairs), np.int32); S_sum = np.zeros(len(pairs))
    med_raw = [_RunningMedian() for _ in pairs]; med_ad = [_RunningMedian() for _ in pairs]
    decided = np.zeros(len(pairs), bool)          # inserted (merged or skipped)
    emitted: Dict[tuple, int] = {}
    step_times: List[float] = []

    for ti in range(n_ts):
        t0 = time.perf_counter()
        # 1. track class state
        for t, c in trk_at[ti]:
            cnt = cls_cnt[t]; cnt[c] += 1
            dom[t] = cnt.most_common(1)[0][0]
        # 2. pair running statistics
        accepted = []
        for i, j in pair_at[ti]:
            p = pairs[i]
            v = p["iou"][j]
            if not np.isnan(v):
                n_ok[i] += 1; S_sum[i] += v
            med_raw[i].add(float(p["cos"][j])); med_ad[i].add(float(p["cos_ad"][j]))
            if decided[i] or p["first_prop"] > ti:
                continue
            a, b = p["a"], p["b"]
            if n_ok[i] < k:
                continue
            S = S_sum[i] / n_ok[i]; c_ad = med_ad[i].median(); c_raw = med_raw[i].median()
            if _rule(S, c_ad, dom[a], dom[b], a, b, cfg):
                accepted.append((S * _rank_factor(c_raw), i))
        # 3. merge in rank order (within this frame only)
        accepted.sort(key=lambda e: (-e[0], e[1]))
        for _, i in accepted:
            p = pairs[i]; a, b = p["a"], p["b"]
            decided[i] = True
            ra, rb = find(a), find(b)
            if ra == rb or cams[ra] & cams[rb]:
                continue
            parent[rb] = ra; cams[ra] |= cams[rb]
            ident[ra] = min(ident[ra], ident[rb])
        # 4. emission for this frame
        for t, _ in trk_at[ti]:
            idt = ident[find(t)]
            emitted[(ti, t[0], t[1])] = idt[1] * CAM_NS + idt[2]
        step_times.append(time.perf_counter() - t0)

    final = {t: (lambda idt: idt[1] * CAM_NS + idt[2])(ident[find(t)]) for t in trk}
    return {"emitted": emitted, "final": final, "step_times": step_times}
