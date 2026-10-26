"""Assignment module of the linker: link rule, ordering and union-find.

A scored candidate pair is accepted as a link when (Eq. 16 of the paper)

    S >= eta(A, B, superclass)  and  same dominant class
    and (adapted cosine >= veto_min  or  S >= veto_rescue).

The appearance veto (last condition) is applied only when a veto key is
given; a pair without an adapted cosine (no shared frame with embeddings) is
not vetoed. Accepted links are inserted into a union-find structure over all
tracks of the scene in decreasing order of S * rho(raw cosine); a merge is
skipped when the two components already contain tracks of the same camera.
Each connected component receives one global identity.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

from panoctvps.linker.proposer import SUPERCLASS

Tracklet = Tuple[int, int]          # (camera_id, tracker_id)
Pair = FrozenSet[Tracklet]

CAM_NAME = {1: "F", 2: "FL", 3: "FR", 4: "SL", 5: "SR"}

# rho(s) = min{1, max{0.25, (s + 0.1) / 0.6}}
RANK_COS_LO = -0.1
RANK_COS_HI = 0.5
RANK_FLOOR = 0.25


def _rank_factor(cos: Optional[float]) -> float:
    """Ordering factor rho of a raw cosine; 1.0 when the cosine is missing."""
    if cos is None:
        return 1.0
    f = (cos - RANK_COS_LO) / (RANK_COS_HI - RANK_COS_LO)
    return min(1.0, max(RANK_FLOOR, f))


def resolve_threshold(threshold: Union[float, Dict[str, float]],
                      cam_a: int, cam_b: int, superclass: str) -> float:
    """Resolve eta for one pair.

    Scalar: one eta for every pair. Dict: the value for the combination of
    camera pair and super-class ("F-FR|vehicle", lower camera id first) if
    defined, otherwise the value for the super-class ("vehicle", "human").
    """
    if not isinstance(threshold, dict):
        return float(threshold)
    a, b = sorted((cam_a, cam_b))
    key = f"{CAM_NAME[a]}-{CAM_NAME[b]}|{superclass}"
    if key in threshold:
        return float(threshold[key])
    return float(threshold[superclass])


def link_rule(score: float, same_class: bool, bar: float,
              veto_cos: Optional[float] = None,
              veto_min: Optional[float] = None,
              veto_rescue: Optional[float] = None) -> bool:
    """Eq. (16): geometric bar, exact class check and (optional) appearance veto."""
    if not (score >= bar and same_class):
        return False
    if veto_min is not None and veto_cos is not None \
            and veto_cos < veto_min and score < veto_rescue:
        return False
    return True


def assign_global_ids(all_tracklets: Set[Tracklet], scored: Dict[Pair, dict],
                      threshold: Union[float, Dict[str, float]] = 0.0,
                      rank_key: Optional[str] = None,
                      veto_key: Optional[str] = None,
                      veto_min: float = 0.0,
                      veto_rescue: float = 1.0,
                      merged_out: Optional[List[Pair]] = None) -> Dict[Tracklet, int]:
    """Return {(camera, tracker_id): global_id}.

    all_tracklets: every track of the scene (unlinked tracks keep their own id).
    scored: scorer output {pair: {"iou_mean", "class_match", "classA", ...}}.
    threshold: eta, a scalar or a dict (see resolve_threshold).
    rank_key: scored-dict key of the raw cosine used by rho for the insertion
        order (None: order by S alone).
    veto_key/veto_min/veto_rescue: the appearance veto (veto_key None: no veto).
    merged_out: if a list is given, the merged links are appended to it in
        insertion order.
    """
    parent: Dict[Tracklet, Tracklet] = {t: t for t in all_tracklets}
    cams: Dict[Tracklet, Set[int]] = {t: {t[0]} for t in all_tracklets}  # cameras per root

    def find(x: Tracklet) -> Tracklet:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    # Accepted links, best rank first.
    edges = []
    for pair, s in scored.items():
        u, v = sorted(pair)
        bar = resolve_threshold(threshold, u[0], v[0], SUPERCLASS.get(s.get("classA")))
        ok = link_rule(s["iou_mean"], s["class_match"], bar,
                       veto_cos=s.get(veto_key) if veto_key is not None else None,
                       veto_min=veto_min if veto_key is not None else None,
                       veto_rescue=veto_rescue)
        if ok:
            if u in parent and v in parent:
                rank = s["iou_mean"] * _rank_factor(s.get(rank_key)) if rank_key else s["iou_mean"]
                edges.append((rank, u, v))
    edges.sort(key=lambda e: -e[0])       # stable: ties keep scored-dict order

    for _, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if cams[ru] & cams[rv]:           # one track per camera in a component
            continue
        parent[rv] = ru
        cams[ru] |= cams[rv]
        if merged_out is not None:
            merged_out.append(frozenset({u, v}))

    out: Dict[Tracklet, int] = {}
    roots: Dict[Tracklet, int] = {}
    g = 1
    for t in all_tracklets:
        r = find(t)
        if r not in roots:
            roots[r] = g
            g += 1
        out[t] = roots[r]
    return out
