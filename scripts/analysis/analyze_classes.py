#!/usr/bin/env python3
"""analyze_classes.py — what the segmenter outputs on the ground-truth thing masks, per thing class (Table 8 of
the paper, segmenter rows; supplement, Tables S7 and S8, and the identities per instance and camera of Table S18(c)).

For every ground-truth mask of a tracked thing instance in the validation split (labels decoded per frame), per
ground-truth class:
- matched, same class: a predicted thing mask of the output with IoU > 0.5 with this ground-truth mask (the cache
  stores, per predicted mask, the ground-truth global id it overlaps most and the IoU with it: gt_global_id, gt_iou;
  IoU > 0.5 implies that this ground-truth mask is the majority under the predicted mask) and the same class;
- matched, other thing class: the same with another thing class (counted by that class);
- not matched: the class that the segmenter predicts on most pixels of the ground-truth mask (predicted semantic map
  of the cache), or "none" when that is class 0.
With --tid-field (the field of the track ids, e.g. tracker_id_deployed): only predicted masks with a track id > 0
are in the output (a mask without a track id is not part of the predicted panoptic output), so only they can match;
and the mean number of predicted identities per object and camera, per class, over (object, camera) with at least
one detection that belongs to the object (gt_iou >= 0.5) and has a track id > 0.
False thing masks (with --tid-field): every output mask of a thing class that matches no ground-truth mask. Split,
per predicted class:
  - on no object: the mask overlaps no pixel of a tracked ground-truth object (it lies on stuff, on unlabeled pixels,
    or on a thing that the labels do not track);
  - second mask on an object already matched: the object it overlaps most is matched by another output mask in the
    same frame and camera;
  - partial mask on an object that no mask matches: the object it overlaps most is matched by no output mask there.
The association quality per class comes from scripts/experiments/eval_linker_tables.py --keep-per-gt; PQ, SQ and RQ per class
from scripts/pipeline/eval_segmenter.py (Table 5).

Usage:
  python scripts/analysis/analyze_classes.py --cache-dir outputs/tracker_ctvis_val_cache --tid-field tracker_id_deployed \
      --output outputs/analyses/classes.json
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.data.constants import CLASS_NAMES
from panoctvps.data.lazy_gt import LazySequenceDataset

REPO = Path(__file__).resolve().parents[2]
THING = {2: "car", 3: "truck", 4: "bus", 5: "other_large_vehicle", 8: "trailer", 9: "pedestrian",
         10: "cyclist", 11: "motorcyclist"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--dataset-root", default=str(REPO.parent / "data/waymo_open_dataset_v_2_0_1"))
    ap.add_argument("--tid-field", default="")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    names = {i: n for i, n in enumerate(CLASS_NAMES)} if isinstance(CLASS_NAMES, (list, tuple)) else dict(CLASS_NAMES)
    gt = LazySequenceDataset(args.dataset_root, "validation")
    s2i = {f: i for i, f in enumerate(gt.segment_files)}
    res = {c: {"gt_masks": 0, "matched_same_class": 0, "matched_other_thing": collections.Counter(),
               "not_matched_predicted_class": collections.Counter()} for c in THING}
    ids = {c: [0, 0] for c in THING}          # [objects x cameras, identities]
    false = {c: collections.Counter() for c in THING}   # predicted class -> kind
    paths = sorted(args.cache_dir.glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    t0 = time.time()
    for si, cp in enumerate(paths, 1):
        npz = np.load(cp, allow_pickle=True)
        g = gt[s2i[str(npz["segment_file"])]]
        gbt = {(fr["timestamp"], c): d for fr in g["frames"] for c, d in fr["cameras"].items()}
        rec = npz["records"]
        gcls_scene = collections.defaultdict(collections.Counter)
        for fi in range(len(npz["frame_records"])):
            fr = npz["frame_records"][fi]
            ts, cam = int(fr["timestamp"]), int(fr["camera_id"])
            if (ts, cam) not in gbt:
                continue
            gc = gbt[(ts, cam)]
            sem, inst = gc["semantic_label"], gc["instance_label"]
            mp = gc["instance_mapping"]
            l2g = {}
            if mp["local_ids"] is not None:
                for l, gg, k in zip(mp["local_ids"], mp["global_ids"], mp["is_tracked"]):
                    if k and l > 0 and gg > 0:
                        l2g[int(l)] = int(gg)
            rs, re = int(fr["records_start"]), int(fr["records_end"])
            match = {}
            inout = [k for k in range(rs, re) if not args.tid_field or int(rec[k][args.tid_field]) > 0]
            for k in inout:
                if float(rec[k]["gt_iou"]) > 0.5 and int(rec[k]["gt_global_id"]) > 0:
                    match[int(rec[k]["gt_global_id"])] = int(rec[k]["class_id"])
            for k in inout:
                pc_ = int(rec[k]["class_id"])
                if pc_ not in THING:
                    continue
                false[pc_]["output_masks"] += 1
                gid_, iou_ = int(rec[k]["gt_global_id"]), float(rec[k]["gt_iou"])
                if gid_ > 0 and iou_ > 0.5:
                    continue
                kind = "on_no_object" if gid_ <= 0 else ("second_mask_on_matched_object" if gid_ in match
                                                          else "partial_mask_on_unmatched_object")
                false[pc_][kind] += 1
            pred_sem = None
            vals = np.unique(sem.astype(np.int64) * 100000 + inst)
            for v in vals.tolist():
                c, ins = divmod(v, 100000)
                if c not in THING or ins <= 0 or ins not in l2g:
                    continue
                gid = l2g[ins]
                gcls_scene[gid][c] += 1
                r = res[c]
                r["gt_masks"] += 1
                if gid in match:
                    if match[gid] == c:
                        r["matched_same_class"] += 1
                    else:
                        r["matched_other_thing"][names.get(match[gid], str(match[gid]))] += 1
                else:
                    if pred_sem is None:
                        pred_sem = reconstruct_frame(npz, fi)[0]
                        if pred_sem.shape != sem.shape:
                            pred_sem = pred_sem[:sem.shape[0], :sem.shape[1]]
                    m = (sem == c) & (inst == ins)
                    pc = np.bincount(pred_sem[m].astype(np.int64)).argmax()
                    r["not_matched_predicted_class"]["none" if pc == 0 else names.get(int(pc), str(pc))] += 1
        if args.tid_field:
            gcls = {gid: cc.most_common(1)[0][0] for gid, cc in gcls_scene.items()}
            tids = collections.defaultdict(set)
            for x in rec:
                gid, tid = int(x["gt_global_id"]), int(x[args.tid_field])
                if gid > 0 and float(x["gt_iou"]) >= 0.5 and tid > 0:
                    tids[(gid, int(x["camera_id"]))].add(tid)
            for (gid, cam), s in tids.items():
                c = gcls.get(gid)
                if c in ids:
                    ids[c][0] += 1
                    ids[c][1] += len(s)
        print(f"[{si}/{len(paths)}] {str(npz['segment_name'])[:34]} ({time.time()-t0:.0f}s)", flush=True)
    out = {"definitions": __doc__, "inputs": {k: str(v) for k, v in vars(args).items()}, "per_class": {}}
    for c, n in THING.items():
        r = res[c]
        N = max(1, r["gt_masks"])
        other = sum(r["matched_other_thing"].values())
        d = {"gt_masks": r["gt_masks"],
             "share_matched_same_class": round(r["matched_same_class"] / N, 4),
             "share_matched_any_thing_class": round((r["matched_same_class"] + other) / N, 4),
             "matched_other_thing_by_class": dict(r["matched_other_thing"].most_common()),
             "not_matched": r["gt_masks"] - r["matched_same_class"] - other,
             "not_matched_predicted_class": dict(r["not_matched_predicted_class"].most_common())}
        if args.tid_field:
            d["object_camera_pairs"] = ids[c][0]
            d["mean_identities_per_object_and_camera"] = round(ids[c][1] / ids[c][0], 4) if ids[c][0] else None
        d["false_thing_masks_by_predicted_class"] = dict(false[c])
        out["per_class"][n] = d
    out["seconds"] = round(time.time() - t0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for n, d in out["per_class"].items():
        print(n, {k: v for k, v in d.items() if not isinstance(v, dict)})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
