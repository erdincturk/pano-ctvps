#!/usr/bin/env python3
"""analyze_failures.py — failure analysis of the deployed full system (wSTQ 0.3228): the tracker and linker rows
of Table 8 of the paper, and the supplement, Tables S18 and S26.

Inputs: the cache (tracks in records['tracker_id_deployed']), the overlap tables of the cache (ground-truth masks and
classes; scripts/experiments/build_overlap_tables.py), the scored candidates (scripts/pipeline/build_scores.py), the adapted cosines
(scripts/pipeline/build_cosines.py) and the deployed linker configuration. Selects nothing. Checks: 4,355 tracked instances,
1,030 multicamera instances, 949 seen by two cameras in the same frame; 1,022 true links with a shared frame, 726
correct, 874 wrong, 296 missed.

Definitions:
- Frames: the 99 annotated timestamps of a scene, indexed 0..98 (200 ms apart).
- Object: a tracked ground-truth thing instance. Its class: the majority class of its masks.
- A detection belongs to an object: the object with its majority ground-truth identifier (gt_global_id), kept only if
  their IoU (gt_iou) is at least 0.5; otherwise it belongs to no object. Panoptic masks do not overlap, so an object
  has at most one detection per frame and camera.
- (g, c) is detected in frame t if a detection of camera c in frame t belongs to g and has a track id > 0; its
  identity there is that track id (the linker never joins two tracks of one camera, so inside a camera the identity
  of the full system is the track).
- Gap between two consecutive detected frames t1 < t2 of (g, c): t2 - t1 - 1 frames without a detection. Bins 0,
  1-2, 3-5, 6-10, 11-20, >20. A track that matches nothing for 6 frames is removed
  (memory_bank.clean_dead_tracklets: removed when cur - last > num_dead_frames = 5).
- Switch: the identity at t2 differs from the identity at t1. At a gap of 0, split by a change of the predicted class
  of the object's detection between t1 and t2.
- Return: a pair with gap >= 1. Hidden: g has no ground-truth mask in camera c in any frame of the gap. Visible,
  missed: g has a ground-truth mask in c in every frame of the gap. Partly hidden: otherwise. Outcome: the same
  identity as at t1 / an identity that g had earlier in c / an identity new to g in c.
- Delayed re-identification: a return with another identity, after which the identity of t1 is given to g again in
  camera c at a later frame; delay = frames from the return to that frame.
- Merge, majority rule: a track whose detections belong to two or more objects; same class = all of them have one
  class. Coverage rule (overlap tables): a track covers object g in frame t if its mask there holds at least half of
  the pixels of g's ground-truth mask in that camera frame. A merge track covers two or more objects over the scene;
  "same frames": it covers two of them in at least one frame (one mask over two objects); "one after the other":
  otherwise.
- Wrong link (as in panoctvps/linker/link_eval.py): per (identity, adjacent pair) whose tracks on the two cameras
  share no object (track object = majority gt_global_id over the track's detections, as in Table S20). Class: the
  classes of the two track objects (same / different / one side on no object). Similarity: the adapted similarity of
  (15) of the candidate between the two tracks. Correct link: a true link whose tracks share an identity; its
  similarity: that of the candidate whose two tracks it joins (the highest if several).
- Handoff link (as in link_eval): (object g, adjacent pair) whose tracks on the two cameras share no frame; objects:
  those with at least one handoff link. Causes: (a) the ground truth has no frame with a mask of g in both cameras;
  (b) it has such frames; for (b), over these shared frames: in every shared frame at least one camera has no
  detection of g with a track id (split: no detection of g there at all, or only detections without a track id), or
  "other" = in some shared frame both cameras have a tracked detection of g, but at least one of them lies in a track
  whose majority object is another object. Ground-truth variant: (g, adjacent pair) with ground-truth masks on both
  cameras but never in the same frame. Time between the cameras: the smallest number of frames between a frame of g
  on one camera and a frame of g on the other; bins 1-2, 3-5, 6-10, 11-20, >20.
- Non-adjacent: objects with ground-truth masks on two cameras that are not adjacent (pairs other than F-FL, F-FR,
  FL-SL, FR-SR).
- per_camera: the counts inside a camera, per camera.

Usage:
  python scripts/analysis/analyze_failures.py --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --output outputs/analyses/failures.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import time
from pathlib import Path

import numpy as np

from panoctvps.evaluation.overlap_tables import gt_objects
from panoctvps.linker.assigner import CAM_NAME, assign_global_ids
from panoctvps.linker.link_eval import link_counts, load_adapted_cosines, load_scene_table
from panoctvps.linker.overlap import ADJACENT_PAIRS

REPO = Path(__file__).resolve().parents[2]
TID = "tracker_id_deployed"
THING = {2, 3, 4, 5, 8, 9, 10, 11}
ADJ = {frozenset(p) for p in ADJACENT_PAIRS}
PNAME = {frozenset((a, b)): f"{CAM_NAME[a]}-{CAM_NAME[b]}" for a, b in ADJACENT_PAIRS}


GBINS = ("0", "1-2", "3-5", "6-10", "11-20", ">20")


def gbin(g):
    return "0" if g == 0 else "1-2" if g <= 2 else "3-5" if g <= 5 else "6-10" if g <= 10 else "11-20" if g <= 20 else ">20"


def hbin(g):
    return "1-2" if g <= 2 else "3-5" if g <= 5 else "6-10" if g <= 10 else "11-20" if g <= 20 else ">20"


def quart(v):
    if not v:
        return None
    q = np.percentile(np.array(v, float), [25, 50, 75])
    return {"n": len(v), "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2])}


def dbin(d):
    return "1-2" if d <= 2 else "3-5" if d <= 5 else ">5"


def min_dist(a, b):
    a, b = np.sort(np.fromiter(a, int)), np.sort(np.fromiter(b, int))
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    d = np.abs(a - b[idx])
    if len(b) > 1:
        d = np.minimum(d, np.abs(a - b[idx - 1]))
    return int(d.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/tracker_ctvis_val_cache"))
    ap.add_argument("--tables-dir", type=Path, required=True, help="Overlap tables of the cache.")
    ap.add_argument("--scores-dir", type=Path, default=Path("outputs/linker/scored_candidates"))
    ap.add_argument("--cosines-dir", type=Path, default=Path("outputs/linker/adapted_cosines"))
    ap.add_argument("--config", type=Path, default=REPO / "configs/linker/deployed.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())["deployed"]

    cams_all = [1, 2, 3, 4, 5]
    K = lambda: collections.Counter()  # noqa: E731
    sw = {c: {b: [0, 0] for b in GBINS} for c in cams_all}       # [pairs, switches]
    sw0 = {c: {"with_class_change": [0, 0], "without_class_change": [0, 0]} for c in cams_all}  # gap 0: [pairs, switches]
    msplit = {c: K() for c in cams_all}
    ho_cause = {"a_no_shared_gt_frame": K(), "b_no_tracked_detection_segmenter_none": K(),
                "b_no_tracked_detection_untracked": K(), "b_other": K()}
    sims = {"wrong": [], "correct": []}
    ret = {c: collections.defaultdict(K) for c in cams_all}                            # (vis, bin) -> outcome
    delayed = {c: K() for c in cams_all}
    returns_other = {c: 0 for c in cams_all}
    merges = {c: {"tracks_with_object": 0, "merge_tracks": 0, "merge_same_class": 0} for c in cams_all}
    frag = {c: {"objects": 0, "identities": 0} for c in cams_all}
    untracked_dets = 0
    gt_tot = K()
    ho_track = {"links": K(), "objects": 0, "time": K()}
    ho_gt = {"links": K(), "objects": 0, "time": K()}
    nonadj = {"objects": 0, "objects_also_adjacent_pair": 0, "pairs": K()}
    wrong = {"by_class": K(), "by_similarity": K(), "by_class_and_similarity": K()}
    scenes, assigns = [], []
    t0 = time.time()
    paths = sorted(args.scores_dir.glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    for i, sp in enumerate(paths, 1):
        seg = sp.stem
        npz = np.load(args.cache_dir / sp.name, allow_pickle=True)
        tsi = {int(t): k for k, t in enumerate(sorted(int(x) for x in npz["unique_timestamps"]))}

        # ground truth from the overlap tables: (g, c) -> frames; class of g
        tab = np.load(args.tables_dir / sp.name, allow_pickle=True)
        gcls, gfr = gt_objects(tab)
        gt_fr = collections.defaultdict(set)
        for (g, c), tss in gfr.items():
            gt_fr[(g, c)] = {tsi[t] for t in tss}
        cams_of = collections.defaultdict(set)
        for (g, c) in gt_fr:
            cams_of[g].add(c)
        gt_tot["objects"] += len(cams_of)
        mc = [g for g in cams_of if len(cams_of[g]) > 1]
        gt_tot["multicam"] += len(mc)
        covis = 0
        for g in mc:
            byt = collections.Counter()
            for c in cams_of[g]:
                byt.update(gt_fr[(g, c)])
            covis += any(v > 1 for v in byt.values())
        gt_tot["seen_by_two_at_once"] += covis

        # detections: (g, c) -> {frame: tid}; track -> objects
        det = collections.defaultdict(dict)
        dcls = collections.defaultdict(dict)
        anydet = collections.defaultdict(set)   # (g, c) -> frames with a detection of g (tracked or not)
        trk_obj = collections.defaultdict(set)
        for x in npz["records"]:
            g, cl = int(x["gt_global_id"]), int(x["class_id"])
            if g <= 0 or float(x["gt_iou"]) < 0.5:
                continue
            c, t, tid = int(x["camera_id"]), tsi[int(x["timestamp"])], int(x[TID])
            anydet[(g, c)].add(t)
            if tid <= 0:
                untracked_dets += 1
                continue
            det[(g, c)][t] = tid
            dcls[(g, c)][t] = cl
            trk_obj[(c, tid)].add(g)

        # inside a camera: switches, returns, delayed re-identification, fragmentation
        for (g, c), d in det.items():
            fr = sorted(d)
            frag[c]["objects"] += 1
            frag[c]["identities"] += len(set(d.values()))
            seen = {d[fr[0]]}
            for k in range(1, len(fr)):
                t1, t2 = fr[k - 1], fr[k]
                gap = t2 - t1 - 1
                b = gbin(gap)
                sw[c][b][0] += 1
                sw[c][b][1] += int(d[t2] != d[t1])
                if gap == 0:
                    kk = "with_class_change" if dcls[(g, c)][t2] != dcls[(g, c)][t1] else "without_class_change"
                    sw0[c][kk][0] += 1
                    sw0[c][kk][1] += int(d[t2] != d[t1])
                if gap >= 1:
                    gapfr = set(range(t1 + 1, t2))
                    vis = gapfr & gt_fr[(g, c)]
                    kind = "hidden" if not vis else "visible_missed" if vis == gapfr else "partly_hidden"
                    out = "same" if d[t2] == d[t1] else "earlier_identity" if d[t2] in seen else "new_identity"
                    ret[c][(kind, b)][out] += 1
                    if out != "same":
                        returns_other[c] += 1
                        later = [t for t in fr[k + 1:] if d[t] == d[t1]]
                        if later:
                            delayed[c][dbin(later[0] - t2)] += 1
                seen.add(d[t2])
        # merges, split from the overlap tables (coverage = at least half of the GT mask of the frame)
        tfr, frs = tab["frames"], npz["frame_records"]
        gsz = {(int(r["f"]), int(r["g"])): int(r["u"]) for r in tab["gsize"]}
        cov = collections.defaultdict(lambda: collections.defaultdict(set))   # (c, tid) -> t -> objects covered
        for r in tab["tpa"]:
            f, gg, k, u = int(r["f"]), int(r["g"]), int(r["k"]), int(r["u"])
            if 2 * u < gsz[(f, gg)] or (gg >> 24) not in THING:
                continue
            fi = int(tfr["fi"][f])
            xr = npz["records"][int(frs[fi]["records_start"]) + k - 1]
            tid = int(xr[TID])
            if tid <= 0:
                continue
            cov[(int(tfr["cam"][f]), tid)][tsi[int(frs[fi]["timestamp"])]].add(gg & 0xFFFFFF)
        for (c, tid), byt in cov.items():
            objs = set().union(*byt.values())
            if len(objs) > 1:
                kind = "same_frames" if any(len(v) > 1 for v in byt.values()) else "one_after_the_other"
                msplit[c][kind] += 1
                msplit[c][kind + "_same_class"] += int(len({gcls.get(g) for g in objs}) == 1)
        for (c, tid), objs in trk_obj.items():
            merges[c]["tracks_with_object"] += 1
            if len(objs) > 1:
                merges[c]["merge_tracks"] += 1
                merges[c]["merge_same_class"] += int(len({gcls.get(g) for g in objs}) == 1)

        # across cameras
        tbl = load_scene_table(sp)
        cos = load_adapted_cosines(args.cosines_dir / sp.name)
        for pair, s in tbl["scored"].items():
            s["cos_med_adapted"] = cos.get(pair)
        lib = assign_global_ids(tbl["all_tr"], tbl["scored"], threshold=cfg["tau"], rank_key=cfg.get("rank_key"),
                                veto_key=cfg.get("veto_key"), veto_min=cfg.get("veto_min", 0.0),
                                veto_rescue=cfg.get("veto_rescue", 1.0))
        scenes.append(tbl); assigns.append(lib)
        dom, by_gid = tbl["dom_gid"], tbl["by_gid"]
        # handoffs from the tracks (same rule as link_eval)
        tid_ts = collections.defaultdict(set)
        for x in npz["records"]:
            tid = int(x[TID])
            if tid > 0 and int(x["class_id"]) in THING:
                tid_ts[(int(x["camera_id"]), tid)].add(tsi[int(x["timestamp"])])
        objs_ho = set()
        for g, cm in by_gid.items():
            for A, B in itertools.combinations(sorted(cm), 2):
                ab = frozenset((A, B))
                if ab not in ADJ:
                    continue
                fa = set().union(*[tid_ts[t] for t in cm[A]])
                fb = set().union(*[tid_ts[t] for t in cm[B]])
                if not fa & fb:
                    ho_track["links"][PNAME[ab]] += 1
                    hb = hbin(min_dist(fa, fb))
                    ho_track["time"][hb] += 1
                    objs_ho.add(g)
                    shared = gt_fr[(g, A)] & gt_fr[(g, B)]
                    if not shared:
                        ho_cause["a_no_shared_gt_frame"][hb] += 1
                    elif any(t in det[(g, A)] and t in det[(g, B)] for t in shared):
                        ho_cause["b_other"][hb] += 1
                    elif any(t in anydet[(g, A)] - set(det[(g, A)]) or t in anydet[(g, B)] - set(det[(g, B)])
                             for t in shared):
                        ho_cause["b_no_tracked_detection_untracked"][hb] += 1
                    else:
                        ho_cause["b_no_tracked_detection_segmenter_none"][hb] += 1
        ho_track["objects"] += len(objs_ho)
        assert sum(1 for _ in objs_ho) <= len(by_gid)
        # handoffs from the ground truth
        objs_ho = set()
        for g, cs in cams_of.items():
            for A, B in itertools.combinations(sorted(cs), 2):
                ab = frozenset((A, B))
                if ab not in ADJ:
                    continue
                if not gt_fr[(g, A)] & gt_fr[(g, B)]:
                    ho_gt["links"][PNAME[ab]] += 1
                    ho_gt["time"][hbin(min_dist(gt_fr[(g, A)], gt_fr[(g, B)]))] += 1
                    objs_ho.add(g)
        ho_gt["objects"] += len(objs_ho)
        # non-adjacent camera pairs
        for g, cs in cams_of.items():
            na = [frozenset(p) for p in itertools.combinations(sorted(cs), 2) if frozenset(p) not in ADJ]
            if na:
                nonadj["objects"] += 1
                nonadj["objects_also_adjacent_pair"] += int(
                    any(frozenset(p) in ADJ for p in itertools.combinations(sorted(cs), 2)))
                for p in na:
                    a, b = sorted(p)
                    nonadj["pairs"][f"{CAM_NAME[a]}-{CAM_NAME[b]}"] += 1
        # wrong links
        comp = collections.defaultdict(set)
        for t, a in lib.items():
            comp[a].add(t)
        for members in comp.values():
            bycam = {t[0]: t for t in members}          # one track per camera
            for A, B in ADJACENT_PAIRS:
                if A in bycam and B in bycam:
                    ta, tb = bycam[A], bycam[B]
                    ga, gb = dom.get(ta, 0), dom.get(tb, 0)
                    s0 = tbl["scored"].get(frozenset({ta, tb}))
                    sim0 = None if s0 is None else s0.get("cos_med_adapted")
                    if ga > 0 and ga == gb:
                        continue
                    if sim0 is not None:
                        sims["wrong"].append(float(sim0))
                    if ga <= 0 or gb <= 0:
                        kc = "background_side"
                    else:
                        kc = "same_gt_class" if gcls.get(ga) == gcls.get(gb) else "different_gt_class"
                    s = tbl["scored"].get(frozenset({ta, tb}))
                    sim = None if s is None else s.get("cos_med_adapted")
                    ks = "no_direct_candidate" if s is None else "no_similarity" if sim is None else \
                        "similarity>=0.5" if sim >= 0.5 else "similarity<0.5"
                    wrong["by_class"][kc] += 1
                    wrong["by_similarity"][ks] += 1
                    wrong["by_class_and_similarity"][f"{kc}|{ks}"] += 1
        # similarity of the correct links (link_eval unit: a true link (g, pair) whose tracks share an identity);
        # several linked same-object pairs: the highest similarity
        for (g, ab) in tbl["true_camlinks"]:
            A, B = sorted(ab)
            cm = by_gid[g]
            v = [tbl["scored"][frozenset({ta, tb})].get("cos_med_adapted") for ta in cm[A] for tb in cm[B]
                 if lib.get(ta) is not None and lib.get(ta) == lib.get(tb) and frozenset({ta, tb}) in tbl["scored"]]
            v = [x for x in v if x is not None]
            if any(lib.get(ta) is not None and lib.get(ta) == lib.get(tb) for ta in cm[A] for tb in cm[B]):
                if v:
                    sims["correct"].append(float(max(v)))
                else:
                    sims["correct_without_direct_candidate"] = sims.get("correct_without_direct_candidate", 0) + 1
        print(f"[{i}/{len(paths)}] {seg[:34]} ({time.time()-t0:.0f}s)", flush=True)

    lc = link_counts(scenes, assigns)
    checks = {"gt_totals": dict(gt_tot), "gt_expected": {"objects": 4355, "multicam": 1030, "seen_by_two_at_once": 949},
              "links": [lc["true"], lc["correct"], lc["wrong"], lc["missed"]], "links_expected": [1022, 726, 874, 296],
              "wrong_links_classified": sum(wrong["by_class"].values()),
              "detections_belonging_to_an_object_without_track": untracked_dets, "n_scenes": len(paths)}

    def tot_cam(d):
        return {k: sum(d[c][k] for c in cams_all) for k in d[1]}

    sw_tot = {b: [sum(sw[c][b][0] for c in cams_all), sum(sw[c][b][1] for c in cams_all)] for b in sw[1]}
    ret_tot = collections.defaultdict(K)
    for c in cams_all:
        for k, v in ret[c].items():
            ret_tot[k].update(v)
    del_tot = K()
    for c in cams_all:
        del_tot.update(delayed[c])
    fmt_sw = lambda d: {b: {"pairs": v[0], "switches": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None}  # noqa
                        for b, v in d.items()}
    fmt_ret = lambda d: {f"{k[0]}|{k[1]}": dict(v) for k, v in sorted(d.items())}  # noqa
    out = {
        "definitions": __doc__, "inputs": {k: str(v) for k, v in vars(args).items()}, "checks": checks,
        "switches_by_gap": fmt_sw(sw_tot),
        "gap0_split": {k: {"pairs": sum(sw0[c][k][0] for c in cams_all),
                                 "switches": sum(sw0[c][k][1] for c in cams_all)} for k in sw0[1]},
        "merge_split": dict(sum((msplit[c] for c in cams_all), K())),
        "handoff_causes": {k: {"links": sum(v.values()), "time_between_cameras": dict(v)} for k, v in ho_cause.items()},
        "similarity": {k: (quart(v) if isinstance(v, list) else v) for k, v in sims.items()} | {"wrong_below_0.5": sum(x < 0.5 for x in sims["wrong"]),
                                                                       "wrong_without_similarity": sum(wrong["by_similarity"].get(k, 0) for k in ("no_direct_candidate", "no_similarity"))},
        "returns": fmt_ret(ret_tot),
        "delayed_reidentification": {"returns_with_another_identity": sum(returns_other.values()),
                                           "delayed_by_frames": dict(del_tot)},
        "merges": tot_cam(merges),
        "fragmentation": {**tot_cam(frag), "mean_identities_per_object_and_camera":
                          round(sum(frag[c]["identities"] for c in cams_all) / max(1, sum(frag[c]["objects"] for c in cams_all)), 4)},
        "missed_links": {"true": lc["true"], "missed": lc["missed"], "note": "stage counts in scripts/analysis/analyze_linker_stages.py"},
        "handoffs_tracks": {"links": dict(ho_track["links"]), "links_total": sum(ho_track["links"].values()),
                                  "objects": ho_track["objects"], "time_between_cameras": dict(ho_track["time"])},
        "handoffs_gt": {"links": dict(ho_gt["links"]), "links_total": sum(ho_gt["links"].values()),
                              "objects": ho_gt["objects"], "time_between_cameras": dict(ho_gt["time"])},
        "non_adjacent": {"objects": nonadj["objects"], "objects_also_on_an_adjacent_pair":
                               nonadj["objects_also_adjacent_pair"], "pairs": dict(nonadj["pairs"])},
        "wrong_links": {k: dict(v) for k, v in wrong.items()},
        "per_camera": {CAM_NAME[c]: {"switches_by_gap": fmt_sw(sw[c]), "returns": fmt_ret(ret[c]),
                                            "returns_with_another_identity": returns_other[c],
                                            "gap0_split": sw0[c], "merge_split": dict(msplit[c]),
                                            "delayed": dict(delayed[c]), "merges": merges[c], "fragmentation": frag[c]}
                              for c in cams_all},
        "seconds": round(time.time() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("checks", "switches_by_gap", "merges", "handoffs_tracks",
                                          "handoffs_gt", "wrong_links")}, indent=1))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
