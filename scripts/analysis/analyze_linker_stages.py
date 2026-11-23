#!/usr/bin/env python3
"""analyze_linker_stages.py — logging run of the deployed linker: the stage table, the two-by-two tables of geometry
against appearance, the recall by the number of frames with evidence, links per class and per scene (supplement,
Tables S21, S22, S24 and S27; the run reproduces the links of the full linker in Table S20: 726 correct, 874
incorrect, 296 missed).

Reads the cache (tracks in records['tracker_id_deployed']), the scored candidates (scripts/pipeline/build_scores.py), the
adapted cosines (scripts/pipeline/build_cosines.py), the deployed linker configuration (configs/linker/deployed.json) and the
overlap tables (scripts/experiments/build_overlap_tables.py; ground-truth class of each object). Selects nothing.


Definitions (panoctvps/linker/link_eval.py):
- Track object: the majority gt_global_id over the track's detections (thing classes, track id > 0); 0 = background.
- True link: (object g, adjacent camera pair A-B) with tracks of g on A and on B that share a frame.
- Handoff: (object g, adjacent pair) with tracks of g on A and B that share no frame (same unit as true links).
- Candidate: a pair of tracks proposed by (13). A candidate is "true" when both tracks have the same object g > 0.
- A true link survives a stage if any candidate between a track of g on A and a track of g on B survives it.
- Stages, in the order of (16) and Algorithm 2: (1) proposal; (2) S >= eta (structured eta); (3) exact class check
  (most frequent predicted class of the two tracks equal); (4) appearance veto: rejected when the adapted similarity
  of (15) < 0.5 and S < 0.6; kept by S >= 0.6 despite similarity < 0.5 = "rescued"; (5) assignment (insertion by
  rank, one track per camera per identity). Stage 5 for true links = correct links.
- Correct / wrong / missed links as in link_eval.link_counts (wrong: per (identity, camera pair)).
- Lost in the assignment: a true link that survives stage 4 but is not correct. Each of its accepted candidates was
  skipped because the two identities already had a track on a common camera. "Lost to a wrong link": for at least one
  skipped candidate, a track on a common camera belongs to another object or to background. Otherwise "lost to
  another track of the same object" (both tracks on every common camera belong to g).
- Ambiguity per camera pair A-B: candidates; tracks that reach the overlap band (a track of A with at least one
  detection whose mask meets the band Omega_{A->B}, and the same for B); mean candidates per band track
  = sum of candidates of each band track / band tracks = 2 x candidates / band tracks.
- Two-by-two table over all proposed candidates: geometry accepts = S >= eta; appearance accepts = adapted similarity
  >= 0.5.
- Evidence frames of a true link: max over its candidates of n_scored (shared frames in which the clipped panoramic
  projections are not both empty); 0 when it has no candidate.
- Object class: majority class of the ground-truth masks of g (overlap tables). Wrong links by class: the most
  frequent predicted class over the tracks of the identity on A and B.

Usage:
  python scripts/analysis/analyze_linker_stages.py --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --linker-tables outputs/linker/linker_tables_deployed.json --output outputs/analyses/linker_stages.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import time
from pathlib import Path

import numpy as np

from panoctvps.data.cache_io import decode_rle
from panoctvps.evaluation.overlap_tables import gt_objects
from panoctvps.linker.assigner import CAM_NAME, _rank_factor, assign_global_ids, resolve_threshold
from panoctvps.linker.link_eval import link_counts, load_adapted_cosines, load_scene_table
from panoctvps.linker.overlap import ADJACENT_PAIRS, calibration_from_arrays, overlap_by_partner
from panoctvps.linker.proposer import PARTNERS, SUPERCLASS, _in_overlap

REPO = Path(__file__).resolve().parents[2]
TID = "tracker_id_deployed"
PAIRS = [frozenset(p) for p in ADJACENT_PAIRS]
PNAME = {frozenset((a, b)): f"{CAM_NAME[a]}-{CAM_NAME[b]}" for a, b in ADJACENT_PAIRS}
CLASS_NAME = {2: "car", 3: "truck", 4: "bus", 5: "other_large_vehicle", 8: "trailer", 9: "pedestrian",
              10: "cyclist", 11: "motorcyclist"}
STAGES = ["proposed", "geometry", "class", "veto", "assigned"]


def band_pass(npz, ovbp):
    """Replicates propose_candidates and records, per track, the partners whose band it reaches."""
    records, frame_records, mask_rle = npz["records"], npz["frame_records"], npz["mask_rle"]
    by_ts = collections.defaultdict(list)
    for fr in frame_records:
        by_ts[int(fr["timestamp"])].append(fr)
    reach = collections.defaultdict(set)           # (cam, tid) -> {partner cams}
    cands = set()
    for ts, frames in by_ts.items():
        cam_dets = {}
        for fr in frames:
            cam = int(fr["camera_id"])
            if cam not in PARTNERS:
                continue
            dets = []
            for k in range(int(fr["records_start"]), int(fr["records_end"])):
                rec = records[k]
                sc = SUPERCLASS.get(int(rec["class_id"]))
                if sc is None or int(rec[TID]) <= 0:
                    continue
                m = decode_rle(mask_rle[k])
                mem = {b: _in_overlap(m, ovbp.get((cam, b))) for b in PARTNERS[cam]}
                tid = int(rec[TID])
                for b, ok in mem.items():
                    if ok:
                        reach[(cam, tid)].add(b)
                dets.append((tid, sc, mem))
            cam_dets[cam] = dets
        for a, b in ADJACENT_PAIRS:
            if a not in cam_dets or b not in cam_dets:
                continue
            a_in = [(t, s) for t, s, mem in cam_dets[a] if mem.get(b)]
            b_in = [(t, s) for t, s, mem in cam_dets[b] if mem.get(a)]
            for ta, sa in a_in:
                for tb, sb in b_in:
                    if sa == sb:
                        cands.add(frozenset({(a, ta), (b, tb)}))
    return reach, cands


def gt_struct(npz):
    """True links and handoffs per camera pair, as link_eval.build_scene_struct, plus per-track predicted classes."""
    gidc = collections.defaultdict(collections.Counter)
    clsc = collections.defaultdict(collections.Counter)
    tid_ts = collections.defaultdict(set)
    for x in npz["records"]:
        if SUPERCLASS.get(int(x["class_id"])) is None or int(x[TID]) <= 0:
            continue
        t = (int(x["camera_id"]), int(x[TID]))
        gidc[t][int(x["gt_global_id"])] += 1
        clsc[t][int(x["class_id"])] += 1
        tid_ts[t].add(int(x["timestamp"]))
    dom = {t: c.most_common(1)[0][0] for t, c in gidc.items()}
    by_gid = collections.defaultdict(lambda: collections.defaultdict(list))
    for t, g in dom.items():
        if g > 0:
            by_gid[g][t[0]].append(t)
    true_links, handoffs = set(), set()
    for g, cm in by_gid.items():
        for A, B in itertools.combinations(sorted(cm), 2):
            if frozenset((A, B)) not in PAIRS:
                continue
            tsA = set().union(*[tid_ts[t] for t in cm[A]])
            tsB = set().union(*[tid_ts[t] for t in cm[B]])
            (true_links if tsA & tsB else handoffs).add((g, frozenset((A, B))))
    return true_links, handoffs, clsc


def gt_classes(tables_dir, seg_name):
    """Majority ground-truth class per object, from the overlap tables."""
    return gt_objects(np.load(Path(tables_dir) / f"{seg_name}.npz", allow_pickle=True))[0]


def instrumented_assign(all_tr, scored, cfg, dom):
    """Same insertion loop as assigner.assign_global_ids, recording why each accepted edge was skipped."""
    tau, rank_key = cfg["tau"], cfg.get("rank_key")
    veto_key, veto_min, veto_rescue = cfg.get("veto_key"), cfg.get("veto_min", 0.0), cfg.get("veto_rescue", 1.0)
    parent = {t: t for t in all_tr}
    members = {t: {t[0]: t} for t in all_tr}     # root -> {cam: track}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    edges = []
    for pair, s in scored.items():
        u, v = sorted(pair)
        bar = resolve_threshold(tau, u[0], v[0], SUPERCLASS.get(s.get("classA")))
        ok = s["iou_mean"] >= bar and s["class_match"]
        vc = s.get(veto_key) if veto_key else None
        if ok and veto_key and vc is not None and vc < veto_min and s["iou_mean"] < veto_rescue:
            ok = False
        if ok and u in parent and v in parent:
            rank = s["iou_mean"] * _rank_factor(s.get(rank_key)) if rank_key else s["iou_mean"]
            edges.append((rank, u, v))
    edges.sort(key=lambda e: -e[0])
    merged, skipped = [], {}
    for _, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        common = set(members[ru]) & set(members[rv])
        if common:
            kind = "same_object"
            g = dom.get(u, 0) if dom.get(u, 0) == dom.get(v, 0) else None
            for c in common:
                if g is None or g <= 0 or dom.get(members[ru][c], 0) != g or dom.get(members[rv][c], 0) != g:
                    kind = "other_object"
            skipped[frozenset({u, v})] = kind
            continue
        parent[rv] = ru
        members[ru].update(members[rv])
        merged.append(frozenset({u, v}))
    out, roots = {}, {}
    for t in all_tr:
        r = find(t)
        out[t] = roots.setdefault(r, len(roots) + 1)
    return out, merged, skipped


def partition(assign):
    comp = collections.defaultdict(set)
    for t, a in assign.items():
        comp[a].add(t)
    return {frozenset(m) for m in comp.values()}


def quart(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "q25": round(float(q[0]), 4), "median": round(float(q[1]), 4), "q75": round(float(q[2]), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/tracker_ctvis_val_cache"))
    ap.add_argument("--scores-dir", type=Path, default=Path("outputs/linker/scored_candidates"))
    ap.add_argument("--cosines-dir", type=Path, default=Path("outputs/linker/adapted_cosines"))
    ap.add_argument("--tables-dir", type=Path, required=True, help="Overlap tables of the cache.")
    ap.add_argument("--config", type=Path, default=REPO / "configs/linker/deployed.json")
    ap.add_argument("--linker-tables", type=Path, default=None,
                    help="Output of scripts/experiments/eval_linker_tables.py (per-scene scores of the tracker and the full system).")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (test)")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())["deployed"]
    tau = cfg["tau"]

    C = lambda: collections.Counter()  # noqa: E731
    tl_stage = {s: C() for s in STAGES}            # true links surviving each stage, per pair
    nt_stage = {s: C() for s in STAGES}            # non-true candidates surviving each stage, per pair
    tc_stage = {s: C() for s in STAGES}            # true candidates surviving each stage, per pair
    rescued = {"true_candidates": C(), "nontrue_candidates": C(), "true_links_only_by_rescue": C(),
               "nontrue_merged_by_rescue": C()}
    lost = {"to_wrong_link": C(), "to_same_object": C()}
    handoffs = C(); n_true = C()
    band_tracks = C(); cand_per_pair = C(); multi_cand_tracks = C()
    two_by_two = {f"geo{'+' if g else '-'}_app{'+' if a else '-'}": {"true": 0, "nontrue": 0}
                  for g in (1, 0) for a in (1, 0)}
    two_by_two["no_similarity"] = {"true": 0, "nontrue": 0}
    sim_true, sim_nontrue = [], []
    tbt_sc = {sc: {k: {"true": 0, "nontrue": 0} for k in two_by_two} for sc in ("vehicle", "human")}
    sim_sc = {sc: {"true": [], "nontrue": []} for sc in ("vehicle", "human")}
    band_cam = C(); deg_sum = C(); cand_track_outside_band = 0
    evid = collections.defaultdict(lambda: [0, 0])  # bin -> [true, correct]
    evid_nf = collections.defaultdict(lambda: [0, 0])
    cls_links = collections.defaultdict(lambda: {"true": 0, "correct": 0, "missed": 0, "handoffs": 0, "wrong": 0})
    per_scene, scenes_tbl, assigns = [], [], []
    checks = {}
    perscene_w = {}
    if args.linker_tables is not None:
        pw = json.loads(args.linker_tables.read_text())
        fl = {r["scene"]: r for r in pw["tracker_only"]["per_scene"]}
        perscene_w = {r["scene"]: (fl[r["scene"]], r) for r in pw["full_system"]["per_scene"]}
        checks["perscene_record"] = {"floor_wSTQ": pw["tracker_only"]["aggregate"]["wSTQ"],
                                     "linker_wSTQ": pw["full_system"]["aggregate"]["wSTQ"]}

    t0 = time.time()
    paths = sorted(args.scores_dir.glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    for i, sp in enumerate(paths, 1):
        seg = sp.stem
        tbl = load_scene_table(sp)
        cos = load_adapted_cosines(args.cosines_dir / sp.name)
        for pair, d in tbl["scored"].items():
            d["cos_med_adapted"] = cos.get(pair)
        npz = np.load(args.cache_dir / sp.name, allow_pickle=True)
        assert str(npz["segment_name"]) == seg
        wh = {}
        for fr in npz["frame_records"]:
            wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
        cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)
        reach, cands = band_pass(npz, overlap_by_partner(cams))
        assert cands == set(tbl["scored"]), f"{seg}: proposal differs from the scored table"
        true_links, ho, clsc = gt_struct(npz)
        assert true_links == tbl["true_camlinks"] and len(ho) == tbl["unreachable"], f"{seg}: GT structure differs"
        gcls = gt_classes(args.tables_dir, seg)
        dom = tbl["dom_gid"]
        by_gid = tbl["by_gid"]

        # library assignment and the instrumented copy must agree
        lib = assign_global_ids(tbl["all_tr"], tbl["scored"], threshold=tau, rank_key=cfg.get("rank_key"),
                                veto_key=cfg.get("veto_key"), veto_min=cfg.get("veto_min", 0.0),
                                veto_rescue=cfg.get("veto_rescue", 1.0))
        mine, merged, skipped = instrumented_assign(tbl["all_tr"], tbl["scored"], cfg, dom)
        assert partition(lib) == partition(mine), f"{seg}: instrumented assignment differs"
        merged_set = set(merged)

        # candidate stages
        st_of = {}
        for pair, s in tbl["scored"].items():
            u, v = sorted(pair)
            ab = frozenset((u[0], v[0]))
            bar = resolve_threshold(tau, u[0], v[0], SUPERCLASS.get(s.get("classA")))
            geo = s["iou_mean"] >= bar
            cl = geo and s["class_match"]
            sim = s.get("cos_med_adapted")
            low = sim is not None and sim < cfg["veto_min"]
            ve = cl and not (low and s["iou_mean"] < cfg["veto_rescue"])
            resc = ve and low
            asg = pair in merged_set
            st_of[pair] = (True, geo, cl, ve, asg, resc)
            is_true = dom.get(u, 0) > 0 and dom.get(u, 0) == dom.get(v, 0)
            tgt = tc_stage if is_true else nt_stage
            for sname, ok in zip(STAGES, (True, geo, cl, ve, asg)):
                if ok:
                    tgt[sname][ab] += 1
            if resc:
                rescued["true_candidates" if is_true else "nontrue_candidates"][ab] += 1
                if asg and not is_true:
                    rescued["nontrue_merged_by_rescue"][ab] += 1
            cand_per_pair[ab] += 1
            scn = SUPERCLASS.get(s.get("classA"))
            kk = "true" if is_true else "nontrue"
            if sim is None:
                two_by_two["no_similarity"][kk] += 1
                if scn in tbt_sc:
                    tbt_sc[scn]["no_similarity"][kk] += 1
            else:
                cell = f"geo{'+' if geo else '-'}_app{'+' if sim >= cfg['veto_min'] else '-'}"
                two_by_two[cell][kk] += 1
                (sim_true if is_true else sim_nontrue).append(sim)
                if scn in tbt_sc:
                    tbt_sc[scn][cell][kk] += 1
                    sim_sc[scn][kk].append(sim)

        # ambiguity: band tracks per pair
        deg = collections.Counter()
        for pair in tbl["scored"]:
            for t in pair:
                other = next(iter(pair - {t}))
                deg[(t, frozenset((t[0], other[0])))] += 1
        for (t, ab), n in deg.items():
            other_cam = next(iter(ab - {t[0]}))
            if other_cam not in reach.get(t, ()):
                cand_track_outside_band += 1
        for t, parts in reach.items():
            for b in parts:
                ab = frozenset((t[0], b))
                band_tracks[ab] += 1
                band_cam[(ab, t[0])] += 1
                deg_sum[ab] += deg[(t, ab)]
                if deg[(t, ab)] >= 2:
                    multi_cand_tracks[ab] += 1

        # true links through the stages
        comp = collections.defaultdict(set)
        for t, a in lib.items():
            comp[a].add(t)
        scene_counts = {"true": 0, "correct": 0}
        for (g, ab) in true_links:
            A, B = sorted(ab)
            n_true[ab] += 1
            scene_counts["true"] += 1
            pairs_g = [frozenset({ta, tb}) for ta in by_gid[g][A] for tb in by_gid[g][B]
                       if frozenset({ta, tb}) in st_of]
            surv = [any(st_of[p][k] for p in pairs_g) for k in range(4)]
            correct = bool({lib[t] for t in by_gid[g][A]} & {lib[t] for t in by_gid[g][B]})
            for sname, ok in zip(STAGES, surv + [correct]):
                if ok:
                    tl_stage[sname][ab] += 1
            if surv[3] and all(st_of[p][5] for p in pairs_g if st_of[p][3]):
                rescued["true_links_only_by_rescue"][ab] += 1
            if surv[3] and not correct:
                kinds = [skipped.get(p) for p in pairs_g if st_of[p][3]]
                lost["to_wrong_link" if "other_object" in kinds else "to_same_object"][ab] += 1
            e = max((tbl["scored"][p]["n_scored"] for p in pairs_g), default=0)
            nf = max((tbl["scored"][p]["n_frames"] for p in pairs_g), default=0)
            for store, val in ((evid, e), (evid_nf, nf)):
                b = "0" if val == 0 else "1" if val == 1 else "2-3" if val <= 3 else "4-10" if val <= 10 else ">10"
                store[b][0] += 1
                store[b][1] += int(correct)
            cname = CLASS_NAME.get(gcls.get(g), str(gcls.get(g)))
            cls_links[cname]["true"] += 1
            cls_links[cname]["correct" if correct else "missed"] += 1
            scene_counts["correct"] += int(correct)
        for (g, ab) in ho:
            handoffs[ab] += 1
            cls_links[CLASS_NAME.get(gcls.get(g), str(gcls.get(g)))]["handoffs"] += 1
        # wrong links by predicted class of the identity
        n_wrong_scene = 0
        for members in comp.values():
            camsm = {t[0] for t in members}
            gb = collections.defaultdict(set)
            for t in members:
                if dom.get(t, 0) > 0:
                    gb[t[0]].add(dom[t])
            for A, B in ADJACENT_PAIRS:
                if A in camsm and B in camsm and not (gb[A] & gb[B]):
                    n_wrong_scene += 1
                    cc = collections.Counter()
                    for t in members:
                        if t[0] in (A, B):
                            cc.update(clsc[t])
                    cls_links[CLASS_NAME.get(cc.most_common(1)[0][0], "?")]["wrong"] += 1
        scenes_tbl.append(tbl)
        assigns.append(lib)
        row = {"scene": seg, "true_links": scene_counts["true"], "correct": scene_counts["correct"],
               "missed": scene_counts["true"] - scene_counts["correct"], "wrong": n_wrong_scene,
               "handoffs": len(ho), "candidates": len(tbl["scored"]),
               "tracks": len(tbl["all_tr"])}
        # per adjacent camera pair (Table S27): candidates, true, correct and incorrect links of this scene
        lcs = link_counts([tbl], [lib])["per_pair"]
        candp = collections.Counter(PNAME[frozenset(t[0] for t in pair)] for pair in tbl["scored"])
        row["per_pair"] = {PNAME[ab]: {"candidates": candp[PNAME[ab]],
                                       "true_links": lcs.get(PNAME[ab], {}).get("true", 0),
                                       "correct": lcs.get(PNAME[ab], {}).get("correct", 0),
                                       "wrong": lcs.get(PNAME[ab], {}).get("wrong", 0)} for ab in PAIRS}
        for k_row, k_pair in (("candidates", "candidates"), ("true_links", "true_links"), ("correct", "correct"),
                              ("wrong", "wrong")):
            assert sum(v[k_pair] for v in row["per_pair"].values()) == row[k_row], (seg, k_row)
        if seg in perscene_w:
            f, m = perscene_w[seg]
            row.update({"wSTQ_tracker": f["wSTQ"], "wSTQ_linker": m["wSTQ"],
                        "gain_wSTQ": m["wSTQ"] - f["wSTQ"], "gain_wAQ": m["wAQ"] - f["wAQ"]})
        per_scene.append(row)
        print(f"[{i}/{len(paths)}] {seg[:34]} cands={len(tbl['scored'])} true={scene_counts['true']} "
              f"correct={scene_counts['correct']} wrong={n_wrong_scene} ({time.time()-t0:.0f}s)", flush=True)

    lc = link_counts(scenes_tbl, assigns)
    tot = lambda c: sum(c.values())  # noqa: E731
    checks["table_s20"] = {"n_scenes": len(paths), "correct": lc["correct"], "wrong": lc["wrong"], "missed": lc["missed"],
                          "expected": [726, 874, 296],
                          "ok": [lc["correct"], lc["wrong"], lc["missed"]] == [726, 874, 296]}
    assert tot(tl_stage["assigned"]) == lc["correct"]
    assert sum(r["wrong"] for r in per_scene) == lc["wrong"]
    assert sum(v["wrong"] for v in cls_links.values()) == lc["wrong"]

    def per_pair(c):
        d = {PNAME[ab]: c[ab] for ab in PAIRS}
        d["total"] = tot(c)
        return d

    out = {
        "definitions": __doc__,
        "inputs": {k: str(v) for k, v in vars(args).items()},
        "config": cfg,
        "checks": checks,
        "link_counts": lc,
        "stages_true_links": {s: per_pair(tl_stage[s]) for s in STAGES},
        "stages_true_candidates": {s: per_pair(tc_stage[s]) for s in STAGES},
        "stages_nontrue_candidates": {s: per_pair(nt_stage[s]) for s in STAGES},
        "rescued_by_overlap": {k: per_pair(v) for k, v in rescued.items()},
        "lost_in_assignment": {k: per_pair(v) for k, v in lost.items()},
        "true_links": per_pair(n_true),
        "handoff_links": per_pair(handoffs),
        "recall_all_true_links_incl_handoffs": {
            "correct": lc["correct"], "true_with_shared_frame": lc["true"], "handoffs": tot(handoffs),
            "recall": round(lc["correct"] / (lc["true"] + tot(handoffs)), 4),
            "per_pair": {PNAME[ab]: round(tl_stage["assigned"][ab] / max(1, n_true[ab] + handoffs[ab]), 4)
                         for ab in PAIRS}},
        "ambiguity": {PNAME[ab]: {"candidates": cand_per_pair[ab], "band_tracks": band_tracks[ab],
                                  "mean_candidates_per_band_track": round(2 * cand_per_pair[ab] / max(1, band_tracks[ab]), 3),
                                  "band_tracks_with_2plus_candidates": multi_cand_tracks[ab]} for ab in PAIRS},
        "two_by_two": two_by_two,
        "two_by_two_by_superclass": tbt_sc,
        "adapted_similarity_by_superclass": {sc: {k: quart(v) for k, v in d.items()} for sc, d in sim_sc.items()},
        "band_tracks_per_camera": {PNAME[ab]: {CAM_NAME[c]: band_cam[(ab, c)] for c in sorted(ab)} for ab in PAIRS},
        "mean_candidates_per_band_track_direct": {PNAME[ab]: round(deg_sum[ab] / max(1, band_tracks[ab]), 4) for ab in PAIRS},
        "candidate_tracks_outside_band": cand_track_outside_band,
        "adapted_similarity": {"true_candidates": quart(sim_true), "nontrue_candidates": quart(sim_nontrue)},
        "recall_by_evidence_frames": {b: {"true": v[0], "correct": v[1], "recall": round(v[1] / v[0], 4) if v[0] else None}
                                      for b, v in sorted(evid.items())},
        "recall_by_shared_frames": {b: {"true": v[0], "correct": v[1], "recall": round(v[1] / v[0], 4) if v[0] else None}
                                    for b, v in sorted(evid_nf.items())},
        "links_per_class": dict(cls_links),
        "per_scene": per_scene,
        "seconds": round(time.time() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("checks", "stages_true_links", "lost_in_assignment")}, indent=1))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
