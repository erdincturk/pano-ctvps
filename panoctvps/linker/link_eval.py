"""Link counting of the linker against the ground truth, and the per-scene
scored-candidate tables it runs on.

A true link is a ground-truth object g seen by tracks on two adjacent cameras
A and B that share a frame (each track is assigned the dominant ground-truth
identity of its detections). For one assignment of global identities:

  correct  true links whose A-side and B-side tracks share a global identity
  missed   true links that are not correct
  wrong    for each global identity and adjacent camera pair (A, B) it spans,
           one wrong link when its A-side and B-side tracks share no
           ground-truth object
  precision = correct / (correct + wrong),  recall = correct / (correct + missed)
  contam   multi-camera ground-truth objects (with at least one true link)
           whose global identity also contains a track of another object

Scored-candidate table (np.savez_compressed, one file per scene, written by
scripts/build_scores.py):
  candidates : one row per scored pair, in the scorer's order -
               camA/tidA/camB/tidB, iou_mean, n_frames/n_scored,
               class_match/classA/classB, cos_med (NaN = None),
               gidA/gidB (dominant gt_global_id per endpoint; 0 = background)
  tracklets  : cam/tid/dom_gid for every track
  camlinks   : gid/camA/camB of the true links
  multicam   : gids of multi-camera ground-truth objects
  meta       : segment name, tid_field, count of handoffs without a shared frame
Columns other than these are ignored when reading.
"""
from __future__ import annotations

import collections
import itertools
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

from panoctvps.linker.assigner import CAM_NAME, assign_global_ids
from panoctvps.linker.overlap import ADJACENT_PAIRS
from panoctvps.linker.proposer import SUPERCLASS

Tracklet = Tuple[int, int]
Pair = FrozenSet[Tracklet]

ADJ = {frozenset(p) for p in ADJACENT_PAIRS}

# Float fields are f8: the scorer emits Python floats (f64) and the assigner
# compares with `>= eta`; f4 storage would change decisions at boundaries.
CAND_DTYPE = np.dtype([
    ("camA", "i1"), ("tidA", "i4"), ("camB", "i1"), ("tidB", "i4"),
    ("iou_mean", "f8"),
    ("n_frames", "i4"), ("n_scored", "i4"),
    ("class_match", "?"), ("classA", "i2"), ("classB", "i2"),
    ("cos_med", "f8"),
    ("gidA", "i8"), ("gidB", "i8"),
])
TRACKLET_DTYPE = np.dtype([("cam", "i1"), ("tid", "i4"), ("dom_gid", "i8")])
CAMLINK_DTYPE = np.dtype([("gid", "i8"), ("camA", "i1"), ("camB", "i1")])


def build_scene_struct(npz, tid_field: str) -> dict:
    """Extract the ground-truth structure of the link counts from a cache npz:
    all tracks, per-track dominant gid, true links, count of handoffs without
    a shared frame, multi-camera ground-truth objects."""
    gidc = collections.defaultdict(collections.Counter)
    tid_ts = collections.defaultdict(set)
    for x in npz["records"]:
        if SUPERCLASS.get(int(x["class_id"])) is None:
            continue
        tid = int(x[tid_field])
        if tid <= 0:
            continue
        t = (int(x["camera_id"]), tid)
        gidc[t][int(x["gt_global_id"])] += 1
        tid_ts[t].add(int(x["timestamp"]))
    all_tr = set(gidc)
    dom_gid = {t: c.most_common(1)[0][0] for t, c in gidc.items()}

    by_gid = collections.defaultdict(lambda: collections.defaultdict(list))
    for t, g in dom_gid.items():
        if g > 0:
            by_gid[g][t[0]].append(t)

    true_camlinks: Set[tuple] = set()
    unreachable = 0
    multicam: List[int] = []
    for g, cm in by_gid.items():
        cams_g = sorted(cm)
        if len(cams_g) > 1:
            multicam.append(g)
        for A, B in itertools.combinations(cams_g, 2):
            if frozenset((A, B)) not in ADJ:
                continue
            tsA = set().union(*[tid_ts[t] for t in cm[A]])
            tsB = set().union(*[tid_ts[t] for t in cm[B]])
            if tsA & tsB:
                true_camlinks.add((g, frozenset((A, B))))
            else:
                unreachable += 1

    return {"all_tr": all_tr, "dom_gid": dom_gid,
            "by_gid": {g: dict(cm) for g, cm in by_gid.items()},
            "true_camlinks": true_camlinks, "unreachable": unreachable,
            "multicam": multicam}


def save_scene_table(path: Path, scored: Dict[Pair, dict], struct: dict,
                     segment: str, tid_field: str) -> None:
    """Persist one scene's scored candidates + ground-truth structure (scorer order)."""
    rows = np.zeros(len(scored), dtype=CAND_DTYPE)
    for i, (pair, s) in enumerate(scored.items()):
        (a, ta), (b, tb) = sorted(pair)
        rows[i]["camA"], rows[i]["tidA"], rows[i]["camB"], rows[i]["tidB"] = a, ta, b, tb
        for f in ("iou_mean", "n_frames", "n_scored", "class_match", "classA", "classB"):
            rows[i][f] = s[f]
        rows[i]["cos_med"] = np.nan if s.get("cos_med") is None else s["cos_med"]
        rows[i]["gidA"] = struct["dom_gid"].get((a, ta), 0)
        rows[i]["gidB"] = struct["dom_gid"].get((b, tb), 0)

    tr = np.zeros(len(struct["all_tr"]), dtype=TRACKLET_DTYPE)
    for i, t in enumerate(sorted(struct["all_tr"])):
        tr[i] = (t[0], t[1], struct["dom_gid"][t])

    cl = np.zeros(len(struct["true_camlinks"]), dtype=CAMLINK_DTYPE)
    for i, (g, ab) in enumerate(sorted(struct["true_camlinks"],
                                       key=lambda x: (x[0], sorted(x[1])))):
        A, B = sorted(ab)
        cl[i] = (g, A, B)

    np.savez_compressed(
        path, candidates=rows, tracklets=tr, camlinks=cl,
        multicam=np.array(sorted(struct["multicam"]), dtype=np.int64),
        segment=segment, tid_field=tid_field,
        unreachable=np.int64(struct["unreachable"]))


def load_scene_table(path: Path) -> dict:
    """Rebuild the scene dict (scored / all_tr / dom_gid / by_gid /
    true_camlinks / unreachable / multicam) from a persisted table."""
    z = np.load(path, allow_pickle=False)
    scored: Dict[Pair, dict] = {}
    for r in z["candidates"]:                  # row order == original scored order
        pair = frozenset({(int(r["camA"]), int(r["tidA"])),
                          (int(r["camB"]), int(r["tidB"]))})
        v = float(r["cos_med"])
        scored[pair] = {"iou_mean": float(r["iou_mean"]), "n_frames": int(r["n_frames"]),
                        "n_scored": int(r["n_scored"]), "class_match": bool(r["class_match"]),
                        "classA": int(r["classA"]), "classB": int(r["classB"]),
                        "cos_med": None if np.isnan(v) else v}

    dom_gid = {(int(r["cam"]), int(r["tid"])): int(r["dom_gid"]) for r in z["tracklets"]}
    all_tr = set(dom_gid)
    by_gid = collections.defaultdict(lambda: collections.defaultdict(list))
    for t, g in dom_gid.items():
        if g > 0:
            by_gid[g][t[0]].append(t)
    true_camlinks = {(int(r["gid"]), frozenset((int(r["camA"]), int(r["camB"]))))
                     for r in z["camlinks"]}
    return {"scored": scored, "all_tr": all_tr, "dom_gid": dom_gid,
            "by_gid": {g: dict(cm) for g, cm in by_gid.items()},
            "true_camlinks": true_camlinks,
            "unreachable": int(z["unreachable"]),
            "multicam": [int(g) for g in z["multicam"]],
            "segment": str(z["segment"])}


def merge_adapted_cosines(scenes: List[dict], cos_dir: Path) -> None:
    """Join the adapted cosines (scripts/build_cosines.py) into the scenes'
    scored dicts in place, as ``cos_med_adapted`` (missing / NaN -> None)."""
    for s in scenes:
        idx = load_adapted_cosines(cos_dir / f"{s['segment']}.npz")
        for pair, d in s["scored"].items():
            d["cos_med_adapted"] = idx.get(pair)


def load_adapted_cosines(path: Path) -> Dict[Pair, Optional[float]]:
    """{pair: cos_med_adapted or None} from one adapted-cosine file."""
    out: Dict[Pair, Optional[float]] = {}
    for r in np.load(path)["cosines"]:
        v = float(r["cos_med_adapted"])
        out[frozenset({(int(r["camA"]), int(r["tidA"])),
                       (int(r["camB"]), int(r["tidB"]))})] = None if np.isnan(v) else v
    return out


def link_counts(scenes: List[dict], assigns: List[Dict[Tracklet, int]]) -> dict:
    """Correct / wrong / missed links, precision, recall, per camera pair, and
    contam (see the module docstring) for one assignment per scene."""
    cl_tp = cl_fn = cl_fp = 0
    pair_tp = collections.Counter(); pair_gt = collections.Counter(); pair_fp = collections.Counter()
    contam_n = 0
    for s, assign in zip(scenes, assigns):
        comp = collections.defaultdict(set)
        for t, a in assign.items():
            comp[a].add(t)

        for members in comp.values():
            cams_mem = {t[0] for t in members}
            gids_by_cam = collections.defaultdict(set)
            for t in members:
                g = s["dom_gid"][t]
                if g > 0:
                    gids_by_cam[t[0]].add(g)
            for A, B in ADJACENT_PAIRS:
                if A in cams_mem and B in cams_mem and not (gids_by_cam[A] & gids_by_cam[B]):
                    cl_fp += 1
                    pair_fp[frozenset((A, B))] += 1

        for (g, ab) in s["true_camlinks"]:
            A, B = sorted(ab)
            cm = s["by_gid"][g]
            compsA = {assign[t] for t in cm[A]}
            compsB = {assign[t] for t in cm[B]}
            pair_gt[ab] += 1
            if compsA & compsB:
                cl_tp += 1; pair_tp[ab] += 1
            else:
                cl_fn += 1

        for g in s["multicam"]:
            cm = s["by_gid"][g]
            reach = [(g, ab) for ab in (frozenset(p) for p in itertools.combinations(sorted(cm), 2))
                     if (g, ab) in s["true_camlinks"]]
            if not reach:
                continue
            g_comps = {assign[t] for cam in cm for t in cm[cam]}
            for c in g_comps:
                gids = {s["dom_gid"][t] for t in comp[c]}
                if any(x > 0 and x != g for x in gids):
                    contam_n += 1
                    break

    per_pair = {}
    for A, B in ADJACENT_PAIRS:
        ab = frozenset((A, B))
        per_pair[f"{CAM_NAME[A]}-{CAM_NAME[B]}"] = {
            "true": pair_gt[ab], "correct": pair_tp[ab], "wrong": pair_fp[ab],
            "missed": pair_gt[ab] - pair_tp[ab]}
    return {"true": cl_tp + cl_fn, "correct": cl_tp, "wrong": cl_fp, "missed": cl_fn,
            "precision": cl_tp / (cl_tp + cl_fp) if cl_tp + cl_fp else 0.0,
            "recall": cl_tp / (cl_tp + cl_fn) if cl_tp + cl_fn else 0.0,
            "contam": contam_n, "per_pair": per_pair}


def eval_config(scenes: List[dict], tau: Union[float, Dict[str, float]],
                rank_key: Optional[str] = None, veto_key: Optional[str] = None,
                veto_min: float = 0.0, veto_rescue: float = 1.0) -> dict:
    """Run the assigner with one configuration on every scene and count links."""
    assigns = [assign_global_ids(s["all_tr"], s["scored"], threshold=tau, rank_key=rank_key,
                                 veto_key=veto_key, veto_min=veto_min, veto_rescue=veto_rescue)
               for s in scenes]
    return link_counts(scenes, assigns)
