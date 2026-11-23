#!/usr/bin/env python3
"""analyze_dataset.py — statistics of the released labels of WOD: PVPS (Table 2 of the paper; supplement,
Tables S1, S2, S3, S5 and S6). No model output.

Scenes and frames come from configs/splits/{training,validation,test}_frames.txt. Four parts:

labels      Table 2 (annotated images, tracked instances, multicamera instances, instances seen by two cameras at
            once), Table S1 (per thing class: tracked instances, share, masks, mask size, masks on an instance of
            another class), Table S3 (tracked instances per camera, cameras per tracked instance) and the instance
            columns of Table S5 (tracked instances in both cameras of each adjacent camera pair, with and without a
            shared frame). Input: the per-mask tables of scripts/analysis/build_label_table.py (training and validation).
            A tracked instance is a (scene, global id) whose masks are marked as tracked. It is in a camera if it
            has a mask there, multicamera if it is in two or more cameras, and seen by two cameras at once if two
            cameras hold a mask of it in the same annotated frame (a shared frame). Annotated images = annotated
            frames x 5 cameras. Table S1: a mask is one tracked instance in one image (an instance that carries two
            thing classes in one image gives two masks; these cases are counted). An instance is counted under its
            most frequent class over its masks, a tie going to the lowest class id. The column "masks on an instance
            of another class" takes the most frequent class with a tie going to the class whose first mask comes
            first in the scan (frames in time order, cameras in file order, classes ascending), the rule of
            collections.Counter. Mask sizes are in pixels; the paper prints the median rounded half up and the 5th
            and 95th percentiles (linear) rounded down. Per scene (Table S27): tracked instances, and those whose
            class (as above) is a vehicle class (car, truck, bus, other_large_vehicle, trailer) or a human class.
motion      Table S2: mask IoU and centroid displacement between two masks of the same tracked instance in the same
            camera, 0.2 s apart (consecutive frames) and 0.4 s apart. Validation labels are decoded frame by frame.
            For each camera, the frames that hold a tracked thing instance are taken in time order, and each is
            compared with the one and the two such frames before it; a pair is kept when the two timestamps are
            200 ms or 400 ms apart (within 20 ms). The mask of an instance is its pixels of its most frequent
            semantic class (which must be a thing class); the pair takes the class of the later mask. Centroid: the
            mean pixel position. Percentiles are linear (numpy.percentile).
band        Table S5, overlap band: the share of the image width, at the middle row, that the other camera of an
            adjacent pair also sees (panoctvps.linker.overlap, from the calibration), in both cameras of each pair,
            over all scenes of the three splits; also over the validation scenes alone.
conditions  Table S6, and the conditions columns of Table S28: time of day, weather and location of every scene
            (stats component, read on every frame row; each field must have one value per scene). A test scene
            without a file in testing/ is read from testing_location/.

Usage:
  python scripts/analysis/build_label_table.py --split training --output-dir outputs/analyses/labels
  python scripts/analysis/build_label_table.py --split validation --output-dir outputs/analyses/labels
  python scripts/analysis/analyze_dataset.py labels --masks-dir outputs/analyses/labels \
      --output outputs/analyses/dataset_labels.json
  python scripts/analysis/analyze_dataset.py motion --output outputs/analyses/dataset_motion.json
  python scripts/analysis/analyze_dataset.py band --output outputs/analyses/dataset_band.json
  python scripts/analysis/analyze_dataset.py conditions --output outputs/analyses/dataset_conditions.json
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from panoctvps.data.constants import PARQUET_COLS
from panoctvps.data.decode import decode_png

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("training", "validation", "test")
FOLDERS = {"training": ["training"], "validation": ["validation"], "test": ["testing", "testing_location"]}
THING = {2: "car", 3: "truck", 4: "bus", 5: "other_large_vehicle", 8: "trailer", 9: "pedestrian",
         10: "cyclist", 11: "motorcyclist"}
VEHICLE = {2, 3, 4, 5, 8}                                # the other thing classes are "human"
CAMERAS = {1: "Front", 2: "Front-Left", 3: "Front-Right", 4: "Side-Left", 5: "Side-Right"}
ADJACENT = [(1, 2), (1, 3), (2, 4), (3, 5)]
GAPS_MS, GAP_TOL_MS = (200, 400), 20
CONDITIONS = {"time_of_day": ["Day", "Dawn/Dusk", "Night"], "weather": ["sunny", "rain"],
              "location": ["location_sf", "location_phx", "location_other"]}


def split_frames(split):
    """{scene: set of annotated timestamps} from the splits file."""
    frames = collections.defaultdict(set)
    for line in (REPO / "configs/splits" / f"{split}_frames.txt").read_text().split():
        if "," in line and not line.startswith("#"):
            s, t = line.split(",")[:2]
            frames[s].add(int(t))
    return frames


def data_file(root, split, component, scene):
    for d in FOLDERS[split]:
        f = Path(root) / d / component / f"{scene}.parquet"
        if f.exists():
            return f
    raise FileNotFoundError(f"{component}/{scene}.parquet ({split})")


# ---- labels: Table 2, Table S3, Table S5 (instances) ---------------------------------------------------------------

def label_part(masks_path):
    t = np.load(masks_path, allow_pickle=True)
    m = t["masks"]
    tr = m[m["tracked"] & (m["gid"] > 0)]
    cams = collections.defaultdict(set)                     # (scene, gid) -> cameras
    at = collections.defaultdict(lambda: collections.defaultdict(set))   # (scene, gid) -> ts -> cameras
    for s, g, c, ts in zip(tr["scene"].tolist(), tr["gid"].tolist(), tr["cam"].tolist(), tr["ts"].tolist()):
        cams[(s, g)].add(c)
        at[(s, g)][ts].add(c)
    n = len(cams)
    multi = [k for k, v in cams.items() if len(v) > 1]
    once = [k for k in multi if any(len(v) > 1 for v in at[k].values())]
    per_cam = {CAMERAS[c]: sum(c in v for v in cams.values()) for c in CAMERAS}
    ncam = collections.Counter(len(v) for v in cams.values())
    out = {"tracked_instances": n, "multicamera_instances": len(multi), "seen_by_two_cameras_at_once": len(once),
           "instances_per_camera": per_cam, "instances_per_camera_sum": sum(per_cam.values()),
           "cameras_per_instance": {str(k): ncam.get(k, 0) for k in range(1, 6)},
           "cameras_per_instance_percent": {str(k): round(100 * ncam.get(k, 0) / n, 2) for k in range(1, 6)},
           "two_or_more_cameras_percent": round(100 * len(multi) / n, 2)}
    pairs, any_pair, any_pair_shared = {}, set(), set()
    for a, b in ADJACENT:
        both = [k for k, v in cams.items() if a in v and b in v]
        shared = [k for k in both if any(a in v and b in v for v in at[k].values())]
        any_pair.update(both)
        any_pair_shared.update(shared)
        pairs[f"{CAMERAS[a]} / {CAMERAS[b]}"] = {"both_cameras": len(both), "with_shared_frame": len(shared),
                                                 "without_shared_frame": len(both) - len(shared)}
    out["adjacent_pairs"] = pairs
    out["adjacent_pairs_all"] = {k: sum(v[k] for v in pairs.values()) for k in next(iter(pairs.values()))}
    out["instances_on_an_adjacent_pair"] = len(any_pair)
    out["instances_on_an_adjacent_pair_with_shared_frame"] = len(any_pair_shared)
    out["multicamera_instances_on_no_adjacent_pair"] = len(set(multi) - any_pair)
    return out


def class_part(masks_path):
    """Table S1, one split; and the tracked instances of each scene by super-class (Table S27)."""
    t = np.load(masks_path, allow_pickle=True)
    m, scene_names = t["masks"], [str(x) for x in t["scenes"]]
    tr = m[m["tracked"] & (m["gid"] > 0)]
    for si in np.unique(tr["scene"]):
        assert (np.diff(tr[tr["scene"] == si]["ts"]) >= 0).all(), "the per-mask table is not in scan order"
    cnt = collections.defaultdict(collections.Counter)
    for s, g, c in zip(tr["scene"].tolist(), tr["gid"].tolist(), tr["cls"].tolist()):
        cnt[(s, g)][c] += 1
    low = {k: min(c for c, n in v.items() if n == max(v.values())) for k, v in cnt.items()}   # tie: lowest class id
    first = {k: v.most_common(1)[0][0] for k, v in cnt.items()}                               # tie: first in the scan
    n = len(cnt)
    per = {}
    for c, name in THING.items():
        mk = tr[tr["cls"] == c]
        px = mk["px"].astype(np.float64)
        k = sum(1 for v in low.values() if v == c)
        per[name] = {"tracked_instances": k, "share_percent": round(100 * k / n, 2), "masks": int(len(mk)),
                     "mask_size_median": float(np.median(px)) if len(px) else None,
                     "mask_size_p5": float(np.percentile(px, 5)) if len(px) else None,
                     "mask_size_p95": float(np.percentile(px, 95)) if len(px) else None,
                     "masks_on_an_instance_of_another_class": sum(
                         1 for s, g in zip(mk["scene"].tolist(), mk["gid"].tolist()) if first[(s, g)] != c)}
    two = collections.Counter(zip(tr["scene"].tolist(), tr["ts"].tolist(), tr["cam"].tolist(), tr["inst"].tolist()))
    two = [x for x, v in two.items() if v > 1]
    return {"per_class": per, "all_classes": {"tracked_instances": n, "masks": int(len(tr)),
                                              "masks_on_an_instance_of_another_class": sum(
                                                  v["masks_on_an_instance_of_another_class"] for v in per.values())},
            "instance_with_two_classes_in_one_image": {"cases": len(two), "images": len({x[:3] for x in two})},
            "instances_with_a_tie": sum(1 for v in cnt.values() if list(v.values()).count(max(v.values())) > 1),
            "per_scene": {scene_names[si]: per_scene_counts(low, si) for si in range(len(scene_names))}}


def per_scene_counts(cls_of, si):
    c = [v for (s, _), v in cls_of.items() if s == si]
    return {"tracked_instances": len(c), "vehicle": sum(x in VEHICLE for x in c),
            "human": sum(x not in VEHICLE for x in c)}


def cmd_labels(a):
    out = {"definitions": __doc__}
    for split in SPLITS:
        fr = split_frames(split)
        out.setdefault("splits", {})[split] = {"scenes": len(fr), "annotated_frames": sum(map(len, fr.values())),
                                               "annotated_images": 5 * sum(map(len, fr.values())),
                                               "frames_per_scene_min": min(map(len, fr.values())),
                                               "frames_per_scene_max": max(map(len, fr.values()))}
    for split in ("training", "validation"):
        out[split] = label_part(a.masks_dir / f"{split}_thing_masks.npz")
        out[split]["classes"] = class_part(a.masks_dir / f"{split}_thing_masks.npz")
    return out


# ---- motion: Table S2 ----------------------------------------------------------------------------------------------

def motion_scene(job):
    root, scene, stamps = job
    C = {k: PARQUET_COLS[k] for k in ("seg_key_timestamp", "seg_key_camera", "seg_panoptic_label", "seg_divisor",
                                       "seg_local_ids", "seg_global_ids", "seg_is_tracked")}
    path = data_file(root, "validation", "camera_segmentation", scene)
    df = pq.read_table(path, columns=list(C.values())).to_pandas()
    df = df[df[C["seg_key_timestamp"]].isin(stamps)]
    pairs = []
    for cam, cdf in df.groupby(C["seg_key_camera"]):
        prev = []                                           # up to two earlier frames with tracked thing instances
        for _, r in cdf.sort_values(C["seg_key_timestamp"]).iterrows():
            loc = r[C["seg_local_ids"]]
            tracked = {}
            if loc is not None:
                for lid, gid, trk in zip(loc, r[C["seg_global_ids"]], r[C["seg_is_tracked"]]):
                    if bool(trk):
                        tracked[int(lid)] = int(gid)
            if not tracked:
                continue
            pan = decode_png(r[C["seg_panoptic_label"]], np.uint16).astype(np.int64)
            div = int(r[C["seg_divisor"]])
            sem, inst = pan // div, pan % div
            w = sem.shape[1]
            cur = {}
            for lid, gid in tracked.items():
                m = inst == lid
                if not m.any():
                    continue
                v, n = np.unique(sem[m], return_counts=True)
                cls = int(v[int(np.argmax(n))])
                if cls not in THING:
                    continue
                idx = np.flatnonzero(m & (sem == cls))
                cur[gid] = (idx, cls, float((idx // w).mean()), float((idx % w).mean()))
            if not cur:
                continue
            ts = int(r[C["seg_key_timestamp"]])
            for pts, pinst in prev:
                dt = (ts - pts) / 1000.0
                gap = next((g for g in GAPS_MS if abs(dt - g) <= GAP_TOL_MS), None)
                if gap is None:
                    continue
                for gid, (idx, cls, cr, cc) in cur.items():
                    if gid not in pinst:
                        continue
                    pidx, _, pr, pc = pinst[gid]
                    inter = np.intersect1d(idx, pidx, assume_unique=True).size
                    iou = inter / (idx.size + pidx.size - inter)
                    pairs.append((gap, cls, iou, float(np.hypot(cr - pr, cc - pc))))
            prev.append((ts, cur))
            if len(prev) > 2:
                prev.pop(0)
    return scene, pairs


def cmd_motion(a):
    fr = split_frames("validation")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(motion_scene, [(a.dataset_root, s, fr[s]) for s in sorted(fr)]))
    p = np.array([x for _, ps in res for x in ps], dtype=[("gap", "i4"), ("cls", "i4"), ("iou", "f8"), ("disp", "f8")])

    def row(s):
        if not len(s):
            return {"pairs": 0}
        return {"pairs": int(len(s)), "iou_p25": float(np.percentile(s["iou"], 25)),
                "iou_median": float(np.percentile(s["iou"], 50)), "iou_p75": float(np.percentile(s["iou"], 75)),
                "displacement_px_median": float(np.percentile(s["disp"], 50)),
                "share_iou_below_0.5": float((s["iou"] < 0.5).mean())}
    s200 = p[p["gap"] == 200]
    out = {"definitions": __doc__, "per_class_0.2s": {THING[c]: row(s200[s200["cls"] == c]) for c in THING},
           "all_classes_0.2s": row(s200), "all_classes_0.4s": row(p[p["gap"] == 400])}
    return out


# ---- band: Table S5 (overlap band) ---------------------------------------------------------------------------------

def band_scene(job):
    from panoctvps.linker.overlap import backproject_rays, load_calibration, sees
    root, split, scene = job
    cal = load_calibration(data_file(root, split, "camera_calibration", scene))
    rays = {c: backproject_rays(v) for c, v in cal.items()}
    rows = []
    for a, b in ADJACENT:
        for x, y in ((a, b), (b, a)):
            band = sees(cal[y], rays[x])
            rows.append((split, scene, f"{CAMERAS[a]} / {CAMERAS[b]}", x, float(band[band.shape[0] // 2].mean())))
    return rows


def cmd_band(a):
    jobs = [(a.dataset_root, sp, s) for sp in SPLITS for s in sorted(split_frames(sp))]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(band_scene, jobs, chunksize=4) for r in rs]
    df = pd.DataFrame(rows, columns=["split", "scene", "pair", "camera", "midrow_width_share"])
    rng = lambda d: {"min_percent": round(100 * d.midrow_width_share.min(), 1),
                     "max_percent": round(100 * d.midrow_width_share.max(), 1)}
    out = {"definitions": __doc__, "scenes": int(df.scene.nunique()),
           "per_pair_all_splits": {p: rng(d) for p, d in df.groupby("pair", sort=False)},
           "all_pairs_all_splits": rng(df), "all_pairs_validation": rng(df[df.split == "validation"])}
    if a.csv:
        df.to_csv(a.csv, index=False)
    return out


# ---- conditions: Table S6 ------------------------------------------------------------------------------------------

def cmd_conditions(a):
    out = {"definitions": __doc__}
    for split in SPLITS:
        cnt = {k: collections.Counter() for k in CONDITIONS}
        per_scene = {}
        for s in sorted(split_frames(split)):
            df = pd.read_parquet(data_file(a.dataset_root, split, "stats", s),
                                 columns=["key.segment_context_name"] + [f"[StatsComponent].{k}" for k in CONDITIONS])
            assert (df["key.segment_context_name"] == s).all()
            for k in CONDITIONS:
                v = df[f"[StatsComponent].{k}"]
                assert not v.isna().any() and v.astype(str).nunique() == 1, (s, k)
                assert str(v.iloc[0]) in CONDITIONS[k], (s, k, v.iloc[0])
                cnt[k][str(v.iloc[0])] += 1
                per_scene.setdefault(s, {})[k] = str(v.iloc[0])
        n = len(split_frames(split))
        out[split] = {"scenes": n, **{k: {v: {"scenes": cnt[k][v], "percent": round(100 * cnt[k][v] / n, 2)}
                                          for v in CONDITIONS[k]} for k in CONDITIONS}}
        if split != "training":
            out[split]["per_scene"] = per_scene
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("labels", "motion", "band", "conditions"))
    ap.add_argument("--dataset-root", type=Path, default=REPO.parent / "data/waymo_open_dataset_v_2_0_1")
    ap.add_argument("--masks-dir", type=Path, default=Path("outputs/analyses/labels"),
                    help="labels: folder of the per-mask tables of scripts/analysis/build_label_table.py.")
    ap.add_argument("--csv", type=Path, default=None, help="band: also write the width of every camera and scene.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = {"labels": cmd_labels, "motion": cmd_motion, "band": cmd_band, "conditions": cmd_conditions}[a.part](a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=1))


if __name__ == "__main__":
    main()
