#!/usr/bin/env python3
"""analyze_tracker_trace.py — the trace of the identity switch of Fig. 7(c4) and the breakdown of the tracker's
decisions by the number of active tracks (supplement, Tables S19 and S17), for our tracker at the deployed setting
(thresholds (0.7, 0.01, 5) and motion gate bounds 600 / 800 / 800 / 1,000 / 1,000 px).

The tracker is panoctvps.evaluation.tracker_cells.VariantTracker (the decisions of MotionGateMemoryBankTracker).
With --check-ids (the ids of the deployed full tracker from scripts/experiments/run_tracker_cells.py) the run must reproduce
every track id.

Order of the tracker (panoctvps/tracker/motion_gate.py, update): the bi-softmax (10) is computed over all detections
and all tracks of the memory bank; then detections are taken in decreasing score order, and for each one a track is
skipped if it was already bound in this frame (used_tracks), if its class differs (class gate), if the detection's
centroid is farther than the bound from the track's last centroid (motion gate), or if the score is below θ_match;
the best remaining track is taken. So the gates come after the normalization.

Gates before the normalization (counterfactual): at the same tracker state, the pairs that a gate rejects are masked
(-inf) before (10); the decisions of the frame are then taken in the same order with the same threshold (a detection
whose candidates are all masked starts a new track, under the usual initiation rule). The state is never changed by
the counterfactual, so only the order of the gates differs.

Definitions: a detection belongs to an object (majority identifier, IoU >= 0.5); the track of the object = the track
that held the object at its most recent matched detection in that camera. A decision of the breakdown: a detection
that belongs to an object whose track is still in the memory bank (active). Correct: the detection joins that track.
Active tracks: the tracks in the memory bank of that camera when the decision is taken. Switch at a gap of 0: the
object was detected (with a track id) in the previous frame of that camera, and the decision is not correct ("with
an identity" if the detection still gets a track id). Bins of active tracks: 1-5, 6-10, 11-20, >20.

The trace (--trace-scene, --trace-camera, --trace-timestamp = the frame of the switch): from three frames before to
three frames after; for every detection and every active track: the raw inner product, the cosine, the two
normalized parts of (10), the score, the class gate, the motion gate with the distance and the bound, the score with
the gates before the normalization, and the action.

Usage:
  python scripts/analysis/analyze_tracker_trace.py --cache-dir outputs/tracker_ctvis_val_cache \
      --tables-dir outputs/overlap_tables/tracker_ctvis_val \
      --check-ids outputs/tracker_cells/tracker_ctvis_val/ours_full_tracker_deployed_ids.npz \
      --output outputs/analyses/tracker_trace.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.evaluation.tracker_cells import VariantTracker, frame_dets

GATE = {1: 600.0, 2: 800.0, 3: 800.0, 4: 1000.0, 5: 1000.0}
TH = dict(match_threshold=0.7, init_score_threshold=0.01, num_dead_frames=5)
BINS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20), (">20", 21, 10**9)]
TRACE = {"scene": "7119831293178745002_1094_720_1114_720", "cam": 1, "tb": 1510868513104507}


def nbin(n):
    for name, lo, hi in BINS:
        if lo <= n <= hi:
            return name
    return "0"


def counterfactual(cur, tids, similarity, allowed, thr):
    sim = similarity.masked_fill(torch.from_numpy(~allowed), float("-inf"))
    ms = torch.nan_to_num((sim.softmax(dim=1) + sim.softmax(dim=0)) / 2, nan=0.0).numpy()
    used, out = set(), {}
    for i in sorted(range(len(cur)), key=lambda k: cur[k]["score"], reverse=True):
        best, bt = -1.0, None
        for j in np.nonzero(allowed[i])[0].tolist():
            tj = int(tids[j])
            if tj in used or ms[i, j] < thr:
                continue
            if ms[i, j] > best:
                best, bt = float(ms[i, j]), tj
        if bt is not None:
            used.add(bt)
        out[i] = bt
    return out, ms


def run(job):
    name, cache_dir, tables_dir, trace = job
    TRACE.update(trace)
    npz = np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=True)
    tab = np.load(Path(tables_dir) / f"{name}.npz", allow_pickle=True)
    rec, frs, uts = npz["records"], npz["frame_records"], npz["unique_timestamps"]
    trk = {c: VariantTracker(motion_gate_px=GATE[c], **TH) for c in GATE}
    counter = defaultdict(int)
    ids = {}
    last_tid = {}            # (g, cam) -> track id at the most recent matched detection
    last_t = {}              # (g, cam) -> frame counter of that detection
    rows = []                # decisions
    trace = []
    state = {}

    def obj_of(fi, iid):
        x = rec[int(frs[fi]["records_start"]) + iid - 1]
        g = int(x["gt_global_id"])
        return g if g > 0 and float(x["gt_iou"]) >= 0.5 else 0

    def log(frame_id, cur, tids, similarity, ms, allowed, assignments, tracker):
        cf, ms_cf = counterfactual(cur, tids, similarity, allowed, tracker.match_threshold)
        _, Tm, tcls = tracker.memory_bank.get_embeddings()
        lastc = [tracker.memory_bank.tracklets[int(tj)].last_centroid for tj in tids.tolist()]
        state["log"] = (cur, tids, similarity, ms, allowed, dict(assignments), cf, ms_cf, tracker,
                        Tm.clone(), tcls.numpy().copy(), [None if c is None else np.array(c, float) for c in lastc])

    for f_row, (fi, cam, ts) in enumerate(tab["frames"].tolist()):
        pred_sem, pred_inst, embs, scores, classes, centroids, px = reconstruct_frame(npz, fi)
        dets = frame_dets(pred_sem, pred_inst, centroids or {})
        t = trk[cam]
        t.log = log
        state.clear()
        bank_before = set(int(k) for k in t.memory_bank.tracklets) if not t.memory_bank.empty else set()
        t.update_dets(dets, embs, scores, counter[cam])
        final = {k: tid for k, s, tid in t.assigned}
        for k, s, tid in t.assigned:
            ids[(f_row, k, s)] = tid
        if "log" in state:
            cur, tids, sim, ms, allowed, asg, cf, ms_cf, tr, Tm, tcls, lastc = state["log"]
            n_active = len(tids)
            for i, x in enumerate(cur):
                g = obj_of(fi, x["local_id"])
                if g <= 0:
                    continue
                key = (g, cam)
                prev = last_tid.get(key)
                if prev is None or prev not in set(int(v) for v in tids):
                    continue
                chosen = asg.get(i)                      # matched track or None (new track / dropped)
                gap0 = last_t.get(key) == counter[cam] - 1
                noid = final.get(x["local_id"]) is None
                rows.append((nbin(n_active), int(chosen == prev), int(gap0 and chosen != prev),
                             int(cf[i] != chosen), int(cf[i] != chosen and cf[i] == prev), int(noid),
                             int(gap0 and chosen != prev and not noid), int(gap0)))
            # trace
            if name == TRACE["scene"] and cam == TRACE["cam"]:
                tsl = sorted(int(v) for v in uts)
                k_tb = tsl.index(TRACE["tb"])
                k = tsl.index(int(ts))
                if k_tb - 3 <= k <= k_tb + 3:
                    order = sorted(range(len(cur)), key=lambda q: cur[q]["score"], reverse=True)
                    E = torch.stack([x["embedding"] for x in cur])
                    cos = (E @ Tm.t()) / (E.norm(dim=1)[:, None] * Tm.norm(dim=1)[None, :])
                    d2t, t2d = sim.softmax(dim=1).numpy(), sim.softmax(dim=0).numpy()
                    dets_out = []
                    for rank, i in enumerate(order):
                        x = cur[i]
                        g = obj_of(fi, x["local_id"])
                        cands = []
                        for j, tj in enumerate(tids.tolist()):
                            lc = lastc[j]
                            dist = None if lc is None else float(np.linalg.norm(np.asarray(x["centroid"]) - np.asarray(lc)))
                            cands.append({"track": int(tj), "raw": float(sim[i, j]), "cos": float(cos[i, j]),
                                          "d2t": float(d2t[i, j]), "t2d": float(t2d[i, j]), "score": float(ms[i, j]),
                                          "class_gate": bool(int(tcls[j]) == x["class_id"]),
                                          "motion_dist": dist, "motion_bound": GATE[cam],
                                          "motion_gate": dist is None or dist <= GATE[cam],
                                          "score_gates_before_norm": float(ms_cf[i, j])})
                        dets_out.append({"order": rank, "local_id": x["local_id"], "object": g, "class": x["class_id"],
                                         "det_score": x["score"], "matched": asg.get(i), "final_id": final.get(x["local_id"]),
                                         "final_id_was_in_bank": final.get(x["local_id"]) in bank_before,
                                         "gates_before_norm_match": cf[i], "candidates": cands})
                    trace.append({"frame_offset": k - k_tb, "timestamp": int(ts), "active_tracks": n_active,
                                  "threshold": tr.match_threshold, "detections": dets_out})
        for k, s, tid in t.assigned:
            g = obj_of(fi, k)
            if g > 0:
                last_tid[(g, cam)] = tid
                last_t[(g, cam)] = counter[cam]
        counter[cam] += 1
    return name, ids, rows, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/tracker_ctvis_val_cache"))
    ap.add_argument("--tables-dir", type=Path, required=True)
    ap.add_argument("--check-ids", type=Path, default=None, help="Ids of the deployed full tracker (check).")
    ap.add_argument("--trace-scene", default=TRACE["scene"])
    ap.add_argument("--trace-camera", type=int, default=TRACE["cam"])
    ap.add_argument("--trace-timestamp", type=int, default=TRACE["tb"], help="Frame of the switch.")
    ap.add_argument("--scenes", default="", help="Comma-separated scene names (default: all scenes of the tables).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    trace = {"scene": a.trace_scene, "cam": a.trace_camera, "tb": a.trace_timestamp}
    want = set(x for x in a.scenes.split(",") if x)
    names = sorted(p.stem for p in a.tables_dir.glob("*.npz") if not want or p.stem in want)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, [(n, str(a.cache_dir), str(a.tables_dir), trace) for n in names]))
    diff = None
    if a.check_ids:
        ref = np.load(a.check_ids, allow_pickle=True)
        rn = ref["scenes"].tolist()
        diff = 0
        for name, ids, _, _ in res:
            si = rn.index(name)
            r = {(int(x["f"]), int(x["k"]), int(x["s"])): int(x["tid"]) for x in ref["ids"][ref["ids"]["scene"] == si]}
            diff += sum(1 for k in set(r) | set(ids) if r.get(k) != ids.get(k))
    agg = {b: Counter() for b, _, _ in BINS}
    for _, _, rows, _ in res:
        for b, ok, sw0, ch, chok, noid, sw0id, g0 in rows:
            c = agg[b]
            c["decisions"] += 1; c["correct"] += ok; c["switch_gap0"] += sw0; c["changed"] += ch; c["changed_correct"] += chok
            c["no_identity"] += noid; c["switch_gap0_with_identity"] += sw0id; c["decisions_gap0"] += g0
    trace = [x for r in res for x in r[3]]
    out = {"definitions": __doc__, "check_ids_differ_from_deployed": diff,
           "breakdown": {b: dict(c) for b, c in agg.items()}, "trace": sorted(trace, key=lambda x: x["frame_offset"])}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else str(o)))
    print("ids differing from the deployed run:", diff)
    for b, c in agg.items():
        print(b, dict(c))


if __name__ == "__main__":
    main()
