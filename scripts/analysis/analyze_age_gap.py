#!/usr/bin/env python3
"""analyze_age_gap.py — decisions of the instances and of the tracks by gap and age, for our tracker on the two
identity models (supplement, Tables S15 and S16; the ground-truth track spans of Table S4), and paired per-scene
differences of the two models (Section VI-B2: "higher in 13 of the 20 scenes").

Inputs: cells of scripts/experiments/run_tracker_cells.py (<stem>.json and <stem>_ids.npz) for each model and setting, with the
cache and the overlap tables of each model; for the ground-truth column of Table S15, the per-mask table of
scripts/analysis/label_statistics.py (--gt-masks). Selects nothing.

Definitions:
- Frames: the 99 annotated timestamps of a scene, indexed 0..98 (the tracker's own frame counter).
- A detection belongs to an object: its majority ground-truth identifier (cache field gt_global_id > 0), kept only if
  their IoU (gt_iou) is at least 0.5. An object has at most one detection per frame and camera.
- (g, c) is detected in frame t if a detection of camera c in frame t belongs to g and has a track id > 0 in the cell.
- A decision: a detected frame t2 of (g, c) that has an earlier detected frame of (g, c); t1 is the most recent one.
- A correct match: the track id at t2 equals the track id at t1 (the detection joins the track that held this object
  at its most recent matched detection).
- Gap: t2 - t1 - 1 frames. Object age: t2 - t0 frames, t0 the first detected frame of (g, c).
- Bins: gap 0, 1-2, 3-5, 6-10, 11-20, >20; age 1-6, 7-12, 13-25, 26-50, 51-99.
Track side (Table S16), on the same detections and track ids: a track is a track id in one camera; detections that
belong to no instance are left out (they neither break nor end a track). A decision of a track is every such
detection of the track after its first; it is correct if it belongs to the same instance as the one before. Gap of a
track: the frames between the two detections; age: the frames since its first used detection. Purity: the share of
used detections that belong to the most frequent instance of their track (pooled). Also counted: the tracks that hold
more than one instance, and the instances per camera held by more than one track.
Ground truth (Table S15, last column; --gt-masks): the same instance-side definitions on the ground-truth masks of
the tracked instances; an instance is present in a frame and camera if it has a mask there (two masks of one
instance in one image count once). The span of a ground-truth track (first to last mask in a camera) is also counted.
Paired differences: for each setting given for two models, the per-scene wAQ and wSTQ differences (second model minus
first), the number of scenes in which each is higher, and the median difference.

Usage:
  python scripts/analysis/analyze_age_gap.py \
      --model "p0:outputs/tracker_ctvis_p0_val_cache:outputs/overlap_tables/tracker_ctvis_p0_val" \
      --model "deployed:outputs/tracker_ctvis_val_cache:outputs/overlap_tables/tracker_ctvis_val" \
      --cell "p0:published:outputs/tracker_cells/tracker_ctvis_p0_val/ours_published_m0.30_i0.01_d20" \
      --cell "p0:retuned:outputs/tracker_cells/tracker_ctvis_p0_val/ours_m0.70_i0.01_d5" \
      --cell "deployed:published:outputs/tracker_cells/tracker_ctvis_val/ours_published_m0.30_i0.01_d20" \
      --cell "deployed:retuned:outputs/tracker_cells/tracker_ctvis_val/ours_m0.70_i0.01_d5" \
      --output outputs/analyses/age_gap.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GAP_BINS = [("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-20", 11, 20), (">20", 21, 10**9)]
AGE_BINS = [("1-6", 1, 6), ("7-12", 7, 12), ("13-25", 13, 25), ("26-50", 26, 50), ("51-99", 51, 99)]


def binof(v, bins):
    for name, lo, hi in bins:
        if lo <= v <= hi:
            return name
    raise ValueError(v)


def scene_decisions(job):
    cache_dir, tables_dir, name, si, ids_path = job
    tab = np.load(Path(tables_dir) / f"{name}.npz", allow_pickle=True)
    npz = np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=True)
    rec, frs, uts = npz["records"], npz["frame_records"], npz["unique_timestamps"]
    p = np.load(ids_path, allow_pickle=True)
    assert str(p["scenes"][si]) == name
    rows = p["ids"][p["ids"]["scene"] == si]
    frames = tab["frames"]
    det = defaultdict(dict)  # (g, cam) -> {t: tid}
    trk = defaultdict(dict)  # (tid, cam) -> {t: g}, detections that belong to an instance only
    all_tracks = set()       # (tid, cam) of every detection with a track id
    n_det = n_left = 0
    for r in rows:
        f, k, tid = int(r["f"]), int(r["k"]), int(r["tid"])
        if tid <= 0:
            continue
        fi = int(frames["fi"][f])
        fr = frs[fi]
        x = rec[int(fr["records_start"]) + k - 1]
        assert int(x["camera_id"]) == int(frames["cam"][f])
        g = int(x["gt_global_id"])
        n_det += 1
        all_tracks.add((tid, int(x["camera_id"])))
        if g <= 0 or float(x["gt_iou"]) < 0.5:
            n_left += 1
            continue
        t = int(np.searchsorted(uts, int(fr["timestamp"])))
        key = (g, int(x["camera_id"]))
        assert t not in det[key], "two detections of one object in one frame and camera"
        det[key][t] = tid
        assert t not in trk[(tid, key[1])], "two detections of one track in one frame and camera"
        trk[(tid, key[1])][t] = g
    out = defaultdict(lambda: [0, 0])
    for (g, cam), d in det.items():
        ts = sorted(d)
        for a, b in zip(ts, ts[1:]):
            gap, age = b - a - 1, b - ts[0]
            ok = int(d[b] == d[a])
            for key in (("gap", binof(gap, GAP_BINS)), ("age", binof(age, AGE_BINS)), ("all", "all")):
                out[key][0] += 1
                out[key][1] += ok
    # track side: decisions of a track, purity, tracks that hold more than one instance
    for d in trk.values():
        ts = sorted(d)
        for a, b in zip(ts, ts[1:]):
            ok = int(d[b] == d[a])
            for key in (("trk", "dec"), ("tgap", binof(b - a - 1, GAP_BINS)), ("tage", binof(b - ts[0], AGE_BINS))):
                out[key][0] += 1
                out[key][1] += ok
        cnt = defaultdict(int)
        for g in d.values():
            cnt[g] += 1
        out[("trk", "pur")][0] += len(d)
        out[("trk", "pur")][1] += max(cnt.values())
        out[("trk", "cnt")][0] += 1
        out[("trk", "cnt")][1] += int(len(cnt) > 1)
    used = sum(len(d) for d in det.values())
    assert used == sum(len(d) for d in trk.values()) == n_det - n_left
    out[("base", "det")][0] += n_det
    out[("base", "used")][0] += used
    out[("base", "left")][0] += n_left
    out[("base", "objects")][0] += len(det)
    out[("base", "objects_multi")][0] += sum(len(set(d.values())) > 1 for d in det.values())
    out[("base", "tracks_all")][0] += len(all_tracks)
    out[("base", "tracks_counted")][0] += len(trk)
    return name, {f"{k[0]}|{k[1]}": v for k, v in out.items()}


def gt_reference(masks_path, cache_dir):
    """The instance-side definitions on the ground-truth masks of the tracked instances (per-mask table of
    scripts/analysis/label_statistics.py); frame counter = unique_timestamps of the cache."""
    t = np.load(masks_path, allow_pickle=True)
    m, scenes = t["masks"], [str(x) for x in t["scenes"]]
    tr = m[m["tracked"] & (m["gid"] > 0)]
    out = defaultdict(int)
    out["masks"] = len(tr)
    for si, name in enumerate(scenes):
        uts = np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=True)["unique_timestamps"]
        ms = tr[tr["scene"] == si]
        ti = np.searchsorted(uts, ms["ts"])
        assert (uts[ti] == ms["ts"]).all(), "a mask outside the frame counter"
        pres = defaultdict(set)
        for g, c, x in zip(ms["gid"].tolist(), ms["cam"].tolist(), ti.tolist()):
            pres[(g, c)].add(x)
        for d in pres.values():
            ts = sorted(d)
            out["present"] += len(ts)
            out["objects"] += 1
            span = ts[-1] - ts[0]
            out["span|0" if span == 0 else f"span|{binof(span, AGE_BINS)}"] += 1
            for a, b in zip(ts, ts[1:]):
                out[f"gap|{binof(b - a - 1, GAP_BINS)}"] += 1
                out[f"age|{binof(b - ts[0], AGE_BINS)}"] += 1
                out["dec"] += 1
    assert out["dec"] == out["present"] - out["objects"]
    assert sum(out[f"gap|{n}"] for n, _, _ in GAP_BINS) == out["dec"] == sum(out[f"age|{n}"] for n, _, _ in AGE_BINS)
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help='"name:cache dir:overlap tables dir"')
    ap.add_argument("--cell", action="append", required=True, help='"model:setting:cell path without .json"')
    ap.add_argument("--gt-masks", type=Path, default=None,
                    help="Per-mask table of scripts/analysis/label_statistics.py (validation) for the ground-truth columns.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes", default="", help="Comma-separated scene names (default: all scenes of the cells).")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    models = {m.split(":")[0]: m.split(":")[1:3] for m in args.model}
    cells = [c.split(":", 2) for c in args.cell]
    want = set(s for s in args.scenes.split(",") if s)
    jobs, keys = [], []
    for model, setting, stem in cells:
        p = np.load(f"{stem}_ids.npz", allow_pickle=True)
        for si, name in enumerate(p["scenes"].tolist()):
            if want and name not in want:
                continue
            jobs.append((models[model][0], models[model][1], name, si, f"{stem}_ids.npz"))
            keys.append((model, setting))
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(scene_decisions, jobs))
    a1 = {}
    for key, (name, o) in zip(keys, outs):
        acc = a1.setdefault(f"{key[0]}|{key[1]}", defaultdict(lambda: [0, 0]))
        for k, (n, c) in o.items():
            acc[k][0] += n
            acc[k][1] += c
    a1 = {k: dict(v) for k, v in a1.items()}
    for k, v in a1.items():
        for kind in ("gap", "age"):
            assert sum(x[0] for kk, x in v.items() if kk.startswith(kind + "|")) == v["all|all"][0]
        for kind in ("tgap", "tage"):
            assert sum(x[0] for kk, x in v.items() if kk.startswith(kind + "|")) == v["trk|dec"][0]
        u, o, tr = v["base|used"][0], v["base|objects"][0], v["base|tracks_counted"][0]
        assert v["all|all"][0] == u - o and v["trk|dec"][0] == u - tr
    # paired differences: settings present for exactly two models (second given minus first given)
    a2 = {}
    by_setting = defaultdict(list)
    for model, setting, stem in cells:
        by_setting[setting].append((model, stem))
    for setting, lst in by_setting.items():
        if len(lst) != 2:
            continue
        (m1, s1), (m2, s2) = lst
        b = {x["segment_name"]: x for x in json.loads(Path(f"{s1}.json").read_text())["per_sequence"]}
        c = {x["segment_name"]: x for x in json.loads(Path(f"{s2}.json").read_text())["per_sequence"]}
        if want:
            b = {k: v for k, v in b.items() if k in want}; c = {k: v for k, v in c.items() if k in want}
        assert set(b) == set(c)
        for m in ("wAQ", "wSTQ"):
            d = [c[s][m] - b[s][m] for s in sorted(b)]
            a2[f"{setting}|{m}"] = {"difference": f"{m2} minus {m1}", f"{m2}_higher": sum(x > 1e-12 for x in d),
                                     f"{m1}_higher": sum(x < -1e-12 for x in d),
                                     "equal": sum(abs(x) <= 1e-12 for x in d),
                                     "median": statistics.median(d), "per_scene": dict(zip(sorted(b), d))}
    gt = None
    if args.gt_masks:
        gt = gt_reference(args.gt_masks, models[cells[0][0]][0])
    out = {"definitions": __doc__, "decisions": a1, "paired_differences": a2, "ground_truth": gt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    for k, v in a1.items():
        n, c = v["all|all"]
        print(f"{k}: {n} decisions, {c / n:.3f} correct")


if __name__ == "__main__":
    main()
