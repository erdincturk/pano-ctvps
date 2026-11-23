#!/usr/bin/env python3
"""eval_linker_alternatives.py — the alternative linking strategies of Table 7 of the paper and the supplement,
Table S23: six alternatives on the deployed tracks.

Every run starts from the same per-camera tracks (cache field tracker_id_deployed) and differs from the deployed
linker in one part only. Each cell is scored exactly from the overlap tables (scripts/experiments/build_overlap_tables.py, as
scripts/experiments/eval_linker_tables.py: predicted identity = the global identity of the track's component; tracks that no run could link
keep an identity of their own) and its links are counted with panoctvps.linker.link_eval.link_counts (Table 7:
precision = correct / (correct + wrong), recall = correct / true).

Shared definitions
- Track: (camera, track id) of a thing detection with a track id > 0. Dominant class of a track: its most frequent
  predicted class (scorer.score_candidates). Super-classes: vehicle {car, truck, bus, other_large_vehicle, trailer},
  human {pedestrian, cyclist, motorcyclist} (proposer.SUPERCLASS).
- Exact class check: the dominant classes of the two tracks are equal (as deployed).
- One track per camera: a merge of two components is skipped when they already hold tracks of a common camera
  (assigner.assign_global_ids). Without this rule, one identity can hold two tracks of one camera at the same time;
  the evaluator then gives both masks of a frame the same instance id, so they form one predicted instance (the
  overlap tables add the overlaps and sizes of the two masks, which is the same, since panoptic masks do not overlap).
- Adapted similarity of (15): the median, over all shared frames of the two tracks (annotated frames in which both
  tracks have a detection), of the cosine between the adapted embeddings psi(e) of the two detections
  (scorer.compute_pair_cosines on the embeddings mapped by the deployed adapter, as scripts/pipeline/build_cosines.py).
- Clipped overlap u of two masks in one frame: both masks forward-projected onto the panorama and clipped to the
  region R_AB seen by both cameras; u = IoU of the clipped projections (scorer.score_candidates). S of a candidate
  pair = the mean of u over its shared frames in which the two clipped projections are not both empty.

Runs
1. deployed: configs/linker/deployed.json through this code path (check run).
2. appearance_only (threshold grid): no calibration at inference. Candidate: a pair of tracks in adjacent cameras
   whose detections have the same super-class in at least one shared frame (the proposal (13) without the band
   condition). Score: the adapted similarity. Accepted when the score is at or above the threshold and the exact
   class check passes. Links inserted by score, highest first (assign_global_ids with the score in place of S and
   no rho), one track per camera.
3. appearance_scoring (threshold grid): candidates as deployed (proposal (13)). Accepted when the adapted
   similarity is at or above the threshold and the exact class check passes; inserted in decreasing order of the
   adapted similarity; one track per camera. A candidate without a similarity is not accepted.
4. per_frame (threshold grid): no score over a track and no appearance. Frames in time order; in each frame, for
   each adjacent camera pair (in the order F-FL, F-FR, FL-SL, FR-SR), the tracked thing masks of the two cameras
   are matched one to one, greedily by decreasing u (ties: record order); a pair is matchable when u is at or
   above the threshold and the two masks have the same predicted class in that frame (the class of a mask in a
   single frame is its own predicted class; equal classes imply the same super-class, the condition of the
   proposal). Every match merges the two tracks' identities at once (one track per camera).
5. per_frame_no_rule: run 4 at its selected threshold, without the rule of one track per camera.
6. global_assignment: the links accepted by the deployed rule (16) (structured eta, exact class check, veto 0.5 /
   rescue 0.6). Each adjacent camera pair is solved as one assignment problem (scipy linear_sum_assignment) that
   maximizes the sum of S x rho(raw cosine) over the chosen links, the score of the greedy order; nothing is tuned.
   The camera pairs form a chain (Side-Left, Front-Left, Front, Front-Right, Side-Right), so one-to-one assignments
   per pair keep one track per camera (checked).
7. deployed_no_rule: the deployed accepted links in the deployed order, without the rule of one track per camera.

Grids: thresholds 0.25 to 0.60 in steps of 0.05 for runs 2-4, selected on all 20 scenes by wSTQ (wSQ is fixed, so
this is the selection by wAQ). If the best threshold is at an edge, that side is extended by two steps, repeated
until the best lies inside the grid, within 0.05 to 0.95.

Checks (written to the output): run 1 reproduces wSTQ 0.32280 / wAQ 0.17360 and 726 / 874 / 296; run 3 has
45,384 candidates (row 1a of the stage table); the recomputed adapted similarity equals the stored one on every
deployed candidate; the recomputed per-frame u reproduces S and its number of frames on every deployed candidate;
run 6 has no identity with two tracks of one camera.

The run has two stages: "prep" writes per-scene inputs (candidates, similarities, per-frame overlaps) to
<out-dir>/prep/, "cells" runs and scores all cells and writes <out-dir>/linker_alternatives.json.

Usage:
  python scripts/experiments/eval_linker_alternatives.py prep  --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --out-dir outputs/linker/alternatives
  python scripts/experiments/eval_linker_alternatives.py cells --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --out-dir outputs/linker/alternatives
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from panoctvps.evaluation import overlap_tables as aq_tables  # noqa: E402
from panoctvps.data.cache_io import decode_rle  # noqa: E402
from panoctvps.linker.adapter import EmbeddingAdapter  # noqa: E402
from panoctvps.linker.assigner import _rank_factor, assign_global_ids, link_rule, resolve_threshold  # noqa: E402
from panoctvps.linker.link_eval import link_counts, load_adapted_cosines, load_scene_table  # noqa: E402
from panoctvps.linker.overlap import ADJACENT_PAIRS, calibration_from_arrays  # noqa: E402
from panoctvps.linker.panorama import (build_overlap_pano_mask, build_panorama_lut,  # noqa: E402
                                       project_mask_to_panorama_forward)
from panoctvps.linker.proposer import SUPERCLASS  # noqa: E402
from panoctvps.linker.scorer import _ext4, compute_pair_cosines  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# paths, set from the command line (and in every worker process, see _set_paths)
CACHE = Path("outputs/tracker_ctvis_val_cache")
SCORES = Path("outputs/linker/scored_candidates")
COSINES = Path("outputs/linker/adapted_cosines")
ADAPTER = Path("outputs/linker/adapter.pt")
TABLES = Path("outputs/overlap_tables/tracker_ctvis_val")
OUT = Path("outputs/linker/alternatives")
TID = "tracker_id_deployed"
UNMATCHED_OFFSET = 1 << 40
CFG = json.loads((REPO / "configs/linker/deployed.json").read_text())["deployed"]


def _set_paths(paths):
    globals().update(paths)


def _paths():
    return {k: globals()[k] for k in ("CACHE", "SCORES", "COSINES", "ADAPTER", "TABLES", "OUT", "CFG")}
GRID = [round(0.25 + 0.05 * i, 2) for i in range(8)]
MIN_TH, MAX_TH = 0.05, 0.95       # limits of the edge extension


# ----------------------------------------------------------------------------------------------- preparation
def prep_scene(name):
    npz = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    rec, frs = npz["records"], npz["frame_records"]
    tbl = load_scene_table(SCORES / f"{name}.npz")
    cos = load_adapted_cosines(COSINES / f"{name}.npz")
    for pair, s in tbl["scored"].items():
        s["cos_med_adapted"] = cos.get(pair)
    # tracked thing detections
    dets = []          # (k, cam, tid, ts, cls)
    for k, x in enumerate(rec):
        tid, cl = int(x[TID]), int(x["class_id"])
        if tid > 0 and SUPERCLASS.get(cl) is not None:
            dets.append((k, int(x["camera_id"]), tid, int(x["timestamp"]), cl))
    tid_ts = collections.defaultdict(set)
    for k, c, t, ts, cl in dets:
        tid_ts[(c, t)].add(ts)
    # appearance-only candidates: same super-class in a shared frame, adjacent cameras (no band)
    by_ts = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, c, t, ts, cl in dets:
        by_ts[ts][c].append((t, SUPERCLASS[cl]))
    app_cands = set()
    for ts, cd in by_ts.items():
        for a, b in ADJACENT_PAIRS:
            for ta, sa in cd.get(a, []):
                for tb, sb in cd.get(b, []):
                    if sa == sb:
                        app_cands.add(frozenset({(a, ta), (b, tb)}))
    # adapted similarity of every appearance-only candidate and of every deployed candidate
    model = EmbeddingAdapter.load(ADAPTER)
    with torch.no_grad():
        emb = model(torch.from_numpy(npz["embeddings"].astype(np.float32))).numpy()
    allc = app_cands | set(tbl["scored"])
    sims = compute_pair_cosines({"records": rec, "embeddings": emb}, allc, tid_field=TID)
    sims = {p: v["cos_med_adapted" if "cos_med_adapted" in v else "cos_med"] for p, v in sims.items()}
    dom_cls = {}
    cnt = collections.defaultdict(collections.Counter)
    for k, c, t, ts, cl in dets:
        cnt[(c, t)][cl] += 1
    dom_cls = {key: v.most_common(1)[0][0] for key, v in cnt.items()}
    # per-frame clipped overlaps u between tracked thing masks of adjacent cameras
    wh = {}
    for fr in frs:
        wh.setdefault(int(fr["camera_id"]), (int(fr["image_w"]), int(fr["image_h"])))
    cams = calibration_from_arrays(npz["extrinsics"], npz["intrinsics"], wh)
    ext4 = {c: _ext4(cams[c]) for c in cams}
    luts = {c: build_panorama_lut(ext4[c], cams[c]["intr9"], (cams[c]["h"], cams[c]["w"])) for c in cams}
    ov = {}
    for a, b in ADJACENT_PAIRS:
        if a in cams and b in cams:
            o = build_overlap_pano_mask(luts[a], luts[b], np.ones((cams[a]["h"], cams[a]["w"]), bool),
                                        np.ones((cams[b]["h"], cams[b]["w"]), bool)).ravel()
            ov[(a, b)] = o
    proj = {}
    for k, c, t, ts, cl in dets:
        m = decode_rle(npz["mask_rle"][k])
        h, w = cams[c]["h"], cams[c]["w"]
        if m.shape != (h, w):
            m = m[:h, :w]
        proj[k] = np.flatnonzero(project_mask_to_panorama_forward(m, ext4[c], cams[c]["intr9"]))
    frame_u = []       # (ts, pair_index, kA, kB, tidA, tidB, clsA, clsB, u) for pairs with u > 0
    clipped_nonempty = {}
    det_by_ts_cam = collections.defaultdict(list)
    for k, c, t, ts, cl in dets:
        det_by_ts_cam[(ts, c)].append((k, t, cl))
    for ts in sorted(by_ts):
        for pi, (a, b) in enumerate(ADJACENT_PAIRS):
            o = ov.get((a, b))
            if o is None:
                continue
            A = [(k, t, cl, proj[k][o[proj[k]]]) for k, t, cl in det_by_ts_cam.get((ts, a), [])]
            B = [(k, t, cl, proj[k][o[proj[k]]]) for k, t, cl in det_by_ts_cam.get((ts, b), [])]
            for k, t, cl, pa in A:
                clipped_nonempty[(pi, k)] = pa.size > 0
            for k, t, cl, pb in B:
                clipped_nonempty[(pi, k)] = pb.size > 0
            A = [x for x in A if x[3].size]
            B = [x for x in B if x[3].size]
            for ka, ta, ca, pa in A:
                for kb, tb, cb, pb in B:
                    inter = np.intersect1d(pa, pb, assume_unique=True).size
                    if inter:
                        frame_u.append((ts, pi, ka, kb, ta, tb, ca, cb, inter / (pa.size + pb.size - inter)))
    # check: S and n_scored of every deployed candidate from the per-frame u
    det_k = {(c, t, ts): k for k, c, t, ts, cl in dets}
    uval = {(pi, ka, kb): u for ts, pi, ka, kb, ta, tb, ca, cb, u in frame_u}
    pidx = {frozenset(p): i for i, p in enumerate(ADJACENT_PAIRS)}
    s_diff, n_diff = 0.0, 0
    for pair, s in tbl["scored"].items():
        (a, ta), (b, tb) = sorted(pair)
        pi = pidx[frozenset((a, b))]
        A_, B_ = ADJACENT_PAIRS[pi]
        if (a, b) != (A_, B_):
            (a, ta), (b, tb) = (b, tb), (a, ta)
        us = []
        for ts in sorted(tid_ts[(a, ta)] & tid_ts[(b, tb)]):
            ka, kb = det_k[(a, ta, ts)], det_k[(b, tb, ts)]
            if not (clipped_nonempty.get((pi, ka)) or clipped_nonempty.get((pi, kb))):
                continue
            us.append(uval.get((pi, ka, kb), 0.0))
        S = float(np.array(us).mean()) if us else 0.0
        s_diff = max(s_diff, abs(S - s["iou_mean"]))
        n_diff += int(len(us) != s["n_scored"])
    sim_diff = max((abs(sims[p] - s["cos_med_adapted"]) for p, s in tbl["scored"].items()
                    if s["cos_med_adapted"] is not None and sims.get(p) is not None), default=0.0)
    sim_none_mismatch = sum((sims.get(p) is None) != (s["cos_med_adapted"] is None) for p, s in tbl["scored"].items())
    out = {"name": name, "tbl": tbl, "app_cands": app_cands, "sims": sims, "dom_cls": dom_cls,
           "frame_u": frame_u,
           "checks": {"S_max_abs_diff": s_diff, "n_scored_mismatch": n_diff, "sim_max_abs_diff": sim_diff,
                      "sim_none_mismatch": sim_none_mismatch, "n_deployed_candidates": len(tbl["scored"]),
                      "n_appearance_only_candidates": len(app_cands),
                      "deployed_candidates_not_in_appearance_only": len(set(tbl["scored"]) - app_cands)}}
    (OUT / "prep").mkdir(parents=True, exist_ok=True)
    with open(OUT / "prep" / f"{name}.pkl", "wb") as f:
        pickle.dump(out, f)
    return name, out["checks"]


# ----------------------------------------------------------------------------------------------- assignments
class UF:
    def __init__(self, tracks):
        self.p = {t: t for t in tracks}
        self.c = {t: {t[0]} for t in tracks}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, u, v, rule=True):
        ru, rv = self.find(u), self.find(v)
        if ru == rv or (rule and self.c[ru] & self.c[rv]):
            return False
        self.p[rv] = ru
        self.c[ru] |= self.c[rv]
        return True

    def ids(self):
        roots, out = {}, {}
        for t in self.p:
            out[t] = roots.setdefault(self.find(t), len(roots) + 1)
        return out


def deployed_edges(scored):
    """Accepted links of rule (16) in the deployed insertion order (as assign_global_ids)."""
    edges = []
    for pair, s in scored.items():
        u, v = sorted(pair)
        bar = resolve_threshold(CFG["tau"], u[0], v[0], SUPERCLASS.get(s.get("classA")))
        if link_rule(s["iou_mean"], s["class_match"], bar, veto_cos=s.get(CFG["veto_key"]),
                     veto_min=CFG["veto_min"], veto_rescue=CFG["veto_rescue"]):
            edges.append((s["iou_mean"] * _rank_factor(s.get(CFG["rank_key"])), u, v))
    edges.sort(key=lambda e: -e[0])
    return edges


def assign(cell, P):
    tbl, run, th = P["tbl"], cell["run"], cell.get("threshold")
    all_tr = tbl["all_tr"]
    if run == "deployed":
        return assign_global_ids(all_tr, tbl["scored"], threshold=CFG["tau"], rank_key=CFG["rank_key"],
                                 veto_key=CFG["veto_key"], veto_min=CFG["veto_min"], veto_rescue=CFG["veto_rescue"])
    if run in ("appearance_only", "appearance_scoring"):
        cands = P["app_cands"] if run == "appearance_only" else tbl["scored"].keys()
        sc = {}
        for pair in (sorted(cands, key=lambda p: tuple(sorted(p))) if run == "appearance_only" else cands):
            (a, ta), (b, tb) = sorted(pair)
            sim = P["sims"].get(pair)
            if sim is None:
                continue
            ca, cb = P["dom_cls"].get((a, ta)), P["dom_cls"].get((b, tb))
            sc[pair] = {"iou_mean": sim, "class_match": ca == cb, "classA": ca}
        return assign_global_ids(all_tr, sc, threshold=th)
    if run in ("per_frame", "per_frame_no_rule"):
        uf = UF(all_tr)
        by_frame = collections.defaultdict(list)
        for ts, pi, ka, kb, ta, tb, ca, cb, u in P["frame_u"]:
            if u >= th and ca == cb:
                by_frame[(ts, pi)].append((-u, ka, kb, ta, tb))
        for key in sorted(by_frame):
            a, b = ADJACENT_PAIRS[key[1]]
            usedA, usedB = set(), set()
            for nu, ka, kb, ta, tb in sorted(by_frame[key]):
                if ka in usedA or kb in usedB:
                    continue
                usedA.add(ka); usedB.add(kb)
                uf.union((a, ta), (b, tb), rule=(run == "per_frame"))
        return uf.ids()
    if run == "deployed_no_rule":
        uf = UF(all_tr)
        for _, u, v in deployed_edges(tbl["scored"]):
            uf.union(u, v, rule=False)
        return uf.ids()
    if run == "global_assignment":
        uf = UF(all_tr)
        by_pair = collections.defaultdict(list)
        for w, u, v in deployed_edges(tbl["scored"]):
            by_pair[(u[0], v[0])].append((w, u, v))
        for key, es in by_pair.items():
            us = sorted({u for _, u, _ in es}); vs = sorted({v for _, _, v in es})
            iu = {t: i for i, t in enumerate(us)}; iv = {t: i for i, t in enumerate(vs)}
            W = np.zeros((len(us), len(vs)))
            for w, u, v in es:
                W[iu[u], iv[v]] = w
            r, c = linear_sum_assignment(W, maximize=True)
            for i, j in zip(r, c):
                if W[i, j] > 0:
                    assert uf.union(us[i], vs[j], rule=False)
        ids = uf.ids()
        comp = collections.defaultdict(list)
        for t, g in ids.items():
            comp[g].append(t[0])
        assert all(len(v) == len(set(v)) for v in comp.values()), "two tracks of one camera in one identity"
        return ids
    raise ValueError(run)


def cells_scene(job):
    name, cells = job
    with open(OUT / "prep" / f"{name}.pkl", "rb") as f:
        P = pickle.load(f)
    tab = dict(np.load(TABLES / f"{name}.npz", allow_pickle=True))
    npz = np.load(CACHE / f"{name}.npz", allow_pickle=True)
    rec = npz["records"]
    keys = []
    for f_row, (fi, cam, ts) in enumerate(tab["frames"].tolist()):
        fr = npz["frame_records"][fi]
        rs, re = int(fr["records_start"]), int(fr["records_end"])
        for k in range(re - rs):
            tid = int(rec[rs + k][TID])
            if tid > 0:
                keys.append((f_row, k + 1, cam, tid))
    cam_ids = {(f, k): tid for f, k, cam, tid in keys}

    def fn(m):
        def ident(f, k, s):
            return np.array([m.get(key, 0) for key in zip(f.tolist(), k.tolist())], dtype=np.int64)
        return ident

    res = {}
    for cell in cells:
        a = assign(cell, P)
        full = {(f, k): (a[(cam, tid)] if (cam, tid) in a else cam * 100_000 + tid + UNMATCHED_OFFSET)
                for f, k, cam, tid in keys}
        o = aq_tables.score_scene(tab, fn(full), fn(cam_ids))
        o["agg"].pop("per_gt", None)
        lc = link_counts([P["tbl"]], [a])
        res[cell["name"]] = (o, lc)
    return name, res


def cell_list(thresholds=None):
    cells = [{"name": "deployed", "run": "deployed"}, {"name": "deployed_no_rule", "run": "deployed_no_rule"},
             {"name": "global_assignment", "run": "global_assignment"}]
    for run in ("appearance_only", "appearance_scoring", "per_frame"):
        for th in (thresholds or {}).get(run, GRID):
            cells.append({"name": f"{run}_t{th:.2f}", "run": run, "threshold": th})
    return cells


def run_cells(names, cells, workers):
    with ProcessPoolExecutor(workers, initializer=_set_paths, initargs=(_paths(),)) as ex:
        outs = list(ex.map(cells_scene, [(n, cells) for n in names]))
    results = {}
    for c in cells:
        sc = [(n, r[c["name"]][0]) for n, r in outs]
        agg, per_scene, _, _ = aq_tables.merge(sc)
        tot = collections.Counter()
        pp = collections.defaultdict(collections.Counter)
        for n, r in outs:
            lc = r[c["name"]][1]
            for k in ("true", "correct", "wrong", "missed", "contam"):
                tot[k] += lc[k]
            for p, v in lc["per_pair"].items():
                pp[p].update(v)
        results[c["name"]] = {**c, "aggregate": agg, "per_scene": per_scene,
                              "links": {**tot, "precision": tot["correct"] / max(1, tot["correct"] + tot["wrong"]),
                                        "recall": tot["correct"] / max(1, tot["true"]),
                                        "per_pair": {p: dict(v) for p, v in pp.items()}}}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "cells"))
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--scores-dir", type=Path, default=SCORES)
    ap.add_argument("--cosines-dir", type=Path, default=COSINES)
    ap.add_argument("--adapter", type=Path, default=ADAPTER, help="Adapter weights of scripts/pipeline/train_adapter.py.")
    ap.add_argument("--tables-dir", type=Path, default=TABLES)
    ap.add_argument("--config", type=Path, default=REPO / "configs/linker/deployed.json")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _set_paths({"CACHE": args.cache_dir, "SCORES": args.scores_dir, "COSINES": args.cosines_dir, "ADAPTER": args.adapter,
                "TABLES": args.tables_dir, "OUT": args.out_dir,
                "CFG": json.loads(args.config.read_text())["deployed"]})
    names = sorted(p.stem for p in TABLES.glob("*.npz"))
    if args.limit:
        names = names[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.cmd == "prep":
        with ProcessPoolExecutor(args.workers, initializer=_set_paths, initargs=(_paths(),)) as ex:
            ck = dict(ex.map(prep_scene, names))
        (OUT / "prep_checks.json").write_text(json.dumps(ck, indent=1))
        tot = {k: (max if "diff" in k else sum)(v[k] for v in ck.values()) for k in next(iter(ck.values()))}
        print("prep checks:", tot, f"{time.time() - t0:.0f}s")
        return
    res = run_cells(names, cell_list(), args.workers)
    # extensions at the edges (two steps), selection by wSTQ
    sel, ext = {}, {}
    for run in ("appearance_only", "appearance_scoring", "per_frame"):
        ext[run] = []
        while True:
            g = {c["threshold"]: c for c in res.values() if c["run"] == run}
            ths = sorted(g)
            best = max(g, key=lambda t: g[t]["aggregate"]["wSTQ"])
            if best == ths[-1] and best < MAX_TH - 1e-9:
                more = [t for t in (round(best + 0.05, 2), round(best + 0.10, 2)) if t <= MAX_TH + 1e-9]
            elif best == ths[0] and best > MIN_TH + 1e-9:
                more = [t for t in (round(best - 0.05, 2), round(best - 0.10, 2)) if t >= MIN_TH - 1e-9]
            else:
                break
            extra = [c for c in cell_list({run: more}) if c["run"] == run and c.get("threshold") in more]
            res.update(run_cells(names, extra, args.workers))
            ext[run] += more
    for run in ("appearance_only", "appearance_scoring", "per_frame"):
        g = {c["threshold"]: c for c in res.values() if c["run"] == run}
        sel[run] = max(g, key=lambda t: g[t]["aggregate"]["wSTQ"])
    nr = [{"name": f"per_frame_no_rule_t{sel['per_frame']:.2f}", "run": "per_frame_no_rule",
           "threshold": sel["per_frame"]}]
    res.update(run_cells(names, nr, args.workers))
    d = res["deployed"]
    checks = {"deployed_reproduces_0.32280_0.17360": abs(d["aggregate"]["wSTQ"] - 0.3228011192907819) < 1e-12
              and round(d["aggregate"]["wAQ"], 5) == 0.17360,
              "deployed_links_726_874_296": (d["links"]["correct"], d["links"]["wrong"], d["links"]["missed"]) == (726, 874, 296),
              "deployed_wSTQ": d["aggregate"]["wSTQ"], "deployed_wAQ": d["aggregate"]["wAQ"]}
    pc = json.loads((OUT / "prep_checks.json").read_text())
    checks["appearance_scoring_candidates"] = sum(v["n_deployed_candidates"] for v in pc.values())
    checks["appearance_scoring_candidates_equal_45384"] = checks["appearance_scoring_candidates"] == 45384
    checks["appearance_only_candidates"] = sum(v["n_appearance_only_candidates"] for v in pc.values())
    checks["deployed_candidates_not_in_appearance_only"] = sum(v["deployed_candidates_not_in_appearance_only"] for v in pc.values())
    checks["S_recomputed_max_abs_diff"] = max(v["S_max_abs_diff"] for v in pc.values())
    checks["n_scored_mismatches"] = sum(v["n_scored_mismatch"] for v in pc.values())
    checks["adapted_similarity_max_abs_diff"] = max(v["sim_max_abs_diff"] for v in pc.values())
    checks["adapted_similarity_none_mismatches"] = sum(v["sim_none_mismatch"] for v in pc.values())
    out = {"definitions": __doc__, "grid": GRID, "extensions": ext, "selected": sel, "checks": checks,
           "cells": res, "seconds": round(time.time() - t0, 1)}
    (OUT / "linker_alternatives.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(checks, indent=1))
    for n, c in res.items():
        a, l = c["aggregate"], c["links"]
        print(f"{n:32s} wSTQ {a['wSTQ']:.5f} wAQ {a['wAQ']:.5f} C {l['correct']} W {l['wrong']} M {l['missed']} "
              f"P {l['precision']:.3f} R {l['recall']:.3f}")


if __name__ == "__main__":
    main()
