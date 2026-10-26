"""Proposal module of the linker.

A pair of tracks (camA, tidA) and (camB, tidB) on adjacent cameras is proposed
as a candidate when, in at least one shared frame, both detections reach their
overlap bands (mask of A intersects Omega_{A->B} and mask of B intersects
Omega_{B->A}) and the two detections of that frame have the same super-class.

Super-classes:
  vehicle = {car, truck, bus, other_large_vehicle, trailer}
  human   = {pedestrian, cyclist, motorcyclist}
"""
from __future__ import annotations

import collections
from typing import Dict, FrozenSet, Set, Tuple

import numpy as np

from panoctvps.data.cache_io import decode_rle
from panoctvps.linker.overlap import ADJACENT_PAIRS

SUPERCLASS = {2: "vehicle", 3: "vehicle", 4: "vehicle", 5: "vehicle", 8: "vehicle",
              9: "human", 10: "human", 11: "human"}

# cam_id -> set of adjacent cam_ids (both directions of ADJACENT_PAIRS)
PARTNERS: Dict[int, Set[int]] = collections.defaultdict(set)
for _a, _b in ADJACENT_PAIRS:
    PARTNERS[_a].add(_b)
    PARTNERS[_b].add(_a)

Pair = FrozenSet[Tuple[int, int]]   # frozenset{(camA, tidA), (camB, tidB)}


def _in_overlap(mask: np.ndarray, ov: np.ndarray) -> bool:
    """True when the mask has at least one pixel in the overlap band ov."""
    if ov is None:
        return False
    if mask.shape != ov.shape:        # prediction masks are bottom-padded to x32
        mask = mask[:ov.shape[0], :ov.shape[1]]
    return int(np.logical_and(mask, ov).sum()) > 0


def propose_candidates(npz, overlap_by_partner: Dict[Tuple[int, int], np.ndarray],
                       tid_field: str = "tracker_id_deployed") -> Set[Pair]:
    """Return the set of candidate cross-camera track pairs.

    npz: loaded prediction cache (.npz) with `records`, `frame_records`, `mask_rle`.
    overlap_by_partner: {(camA, camB): (H,W) bool} from overlap.overlap_by_partner.
    tid_field: record field holding the track id; detections with a
        non-positive id (not assigned to a track) are skipped.
    """
    records = npz["records"]
    frame_records = npz["frame_records"]
    mask_rle = npz["mask_rle"]

    # frames grouped by timestamp
    frames_by_ts: Dict[int, list] = collections.defaultdict(list)
    for fr in frame_records:
        frames_by_ts[int(fr["timestamp"])].append(fr)

    candidates: Set[Pair] = set()
    for ts, frames in frames_by_ts.items():
        # Decode each detection's mask once; precompute overlap membership per partner.
        # cam -> list of (tid, superclass, {partner: bool})
        cam_dets: Dict[int, list] = {}
        for fr in frames:
            cam = int(fr["camera_id"])
            if cam not in PARTNERS:
                continue
            dets = []
            for k in range(int(fr["records_start"]), int(fr["records_end"])):
                rec = records[k]
                sc = SUPERCLASS.get(int(rec["class_id"]))
                if sc is None:
                    continue
                tid = int(rec[tid_field])
                if tid <= 0:
                    continue
                mask = decode_rle(mask_rle[k])
                mem = {b: _in_overlap(mask, overlap_by_partner.get((cam, b)))
                       for b in PARTNERS[cam]}
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
                        candidates.add(frozenset({(a, ta), (b, tb)}))
    return candidates
