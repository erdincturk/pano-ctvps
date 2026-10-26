"""Scoring module of the linker: geometric score S and the cosine cues.

Per candidate pair and shared frame, both masks are forward-projected onto the
panorama and clipped to the region R_AB covered by both cameras; u is the IoU
of the clipped projections. Frames in which both clipped projections are empty
provide no evidence and are excluded; S (``iou_mean``) is the mean of u over
the remaining frames (0 if none remain).

``cos_med`` is the median, over all shared frames of the pair, of the cosine
between the normalised embeddings of the two detections. The same median of
the adapted embeddings (``cos_med_adapted``) is computed by
scripts/build_cosines.py with the same function on psi(e).
"""
from __future__ import annotations

import collections
from typing import Dict, FrozenSet, Set, Tuple

import numpy as np

from panoctvps.data.cache_io import decode_rle
from panoctvps.linker.overlap import ADJACENT_PAIRS
from panoctvps.linker.panorama import (
    DEFAULT_PANO_SHAPE, build_overlap_pano_mask, build_panorama_lut,
    project_mask_to_panorama_forward,
)

Pair = FrozenSet[Tuple[int, int]]


def _ext4(cal: dict) -> np.ndarray:
    e = np.eye(4, dtype=np.float64)
    e[:3, :3] = cal["R_cam2veh"]      # build_panorama_lut/forward use the rotation block
    return e


COSINE_KEYS = ("cos_med",)


def compute_pair_cosines(npz, candidates: Set[Pair],
                         tid_field: str = "tracker_id_deployed") -> Dict[Pair, dict]:
    """Median cosine per candidate pair over its shared frames.

    The cached embeddings are not unit-norm; they are L2-normalised here before
    the dot product. Returns {} when the cache has no embeddings.
    """
    records = npz["records"]
    if "embeddings" not in npz:
        return {}
    emb = np.asarray(npz["embeddings"], dtype=np.float32)
    if emb.shape[0] != len(records) or len(records) == 0:
        return {}
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)

    det_index: Dict[tuple, int] = {}
    tid_ts: Dict[tuple, set] = collections.defaultdict(set)
    for k, x in enumerate(records):
        tid = int(x[tid_field])
        if tid <= 0:
            continue
        cam, ts = int(x["camera_id"]), int(x["timestamp"])
        det_index[(cam, tid, ts)] = k
        tid_ts[(cam, tid)].add(ts)

    out: Dict[Pair, dict] = {}
    for pair in candidates:
        (a, ta), (b, tb) = sorted(pair)
        shared = tid_ts[(a, ta)] & tid_ts[(b, tb)]
        if not shared:
            out[pair] = dict.fromkeys(COSINE_KEYS)
            continue
        cos = np.array([float(emb[det_index[(a, ta, ts)]] @ emb[det_index[(b, tb, ts)]])
                        for ts in sorted(shared)])
        out[pair] = {"cos_med": float(np.median(cos))}
    return out


def score_candidates(npz, cams: Dict[int, dict], candidates: Set[Pair],
                     tid_field: str = "tracker_id_deployed") -> Dict[Pair, dict]:
    """Return {pair: {iou_mean, n_frames, n_scored, class_match, classA, classB, cos_med}}.

    R_AB is computed from the calibration alone (cells sampled by both
    cameras' panorama LUTs). classA/classB are the dominant (most frequent)
    classes of the two tracks; class_match is their equality.
    tid_field: record field holding the track id (must match the proposer's).
    """
    records = npz["records"]
    mask_rle = npz["mask_rle"]

    ext4 = {c: _ext4(cams[c]) for c in cams}
    luts = {c: build_panorama_lut(ext4[c], cams[c]["intr9"], (cams[c]["h"], cams[c]["w"]))
            for c in cams}
    # R_AB: panorama cells sampled by both cameras.
    ov_flat: Dict[Tuple[int, int], np.ndarray] = {}
    for a, b in ADJACENT_PAIRS:
        if a in cams and b in cams:
            ones_a = np.ones((cams[a]["h"], cams[a]["w"]), bool)
            ones_b = np.ones((cams[b]["h"], cams[b]["w"]), bool)
            ov = build_overlap_pano_mask(luts[a], luts[b], ones_a, ones_b).ravel()
            ov_flat[(a, b)] = ov
            ov_flat[(b, a)] = ov

    # Indices: (cam,tid,ts) -> record k; tracklet -> timestamps; tracklet -> dominant class.
    det_index: Dict[tuple, int] = {}
    tid_ts: Dict[tuple, set] = collections.defaultdict(set)
    tid_cls: Dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for k, x in enumerate(records):
        tid = int(x[tid_field])
        if tid <= 0:
            continue
        cam, ts = int(x["camera_id"]), int(x["timestamp"])
        det_index[(cam, tid, ts)] = k
        tid_ts[(cam, tid)].add(ts)
        tid_cls[(cam, tid)][int(x["class_id"])] += 1
    dom = {key: c.most_common(1)[0][0] for key, c in tid_cls.items()}

    proj_cache: Dict[int, np.ndarray] = {}

    def proj(cam: int, tid: int, ts: int) -> np.ndarray:
        k = det_index[(cam, tid, ts)]
        if k not in proj_cache:
            m = decode_rle(mask_rle[k])
            h, w = cams[cam]["h"], cams[cam]["w"]
            if m.shape != (h, w):        # prediction masks are bottom-padded to ×32
                m = m[:h, :w]
            pano = project_mask_to_panorama_forward(m, ext4[cam], cams[cam]["intr9"])
            proj_cache[k] = np.flatnonzero(pano)
        return proj_cache[k]

    cosines = compute_pair_cosines(npz, candidates, tid_field=tid_field)

    results: Dict[Pair, dict] = {}
    for pair in candidates:
        (a, ta), (b, tb) = sorted(pair)
        ovf = ov_flat.get((a, b))
        shared = sorted(tid_ts[(a, ta)] & tid_ts[(b, tb)])
        ious = []
        for ts in shared:
            ia, ib = proj(a, ta, ts), proj(b, tb, ts)
            if ovf is not None:                  # clip both to the pair's overlap region
                ia = ia[ovf[ia]]
                ib = ib[ovf[ib]]
            if ia.size == 0 and ib.size == 0:
                continue                         # no evidence in this frame
            inter = np.intersect1d(ia, ib, assume_unique=True).size
            union = ia.size + ib.size - inter
            ious.append(inter / union if union else 0.0)
        arr = np.array(ious) if ious else np.array([0.0])
        results[pair] = {
            "iou_mean": float(arr.mean()), "n_frames": len(shared),
            "n_scored": len(ious),   # frames with evidence
            "class_match": bool(dom.get((a, ta)) == dom.get((b, tb))),
            "classA": dom.get((a, ta)), "classB": dom.get((b, tb)),
            **cosines.get(pair, dict.fromkeys(COSINE_KEYS)),
        }
    return results
