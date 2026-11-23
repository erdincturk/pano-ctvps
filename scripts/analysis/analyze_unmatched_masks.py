#!/usr/bin/env python3
"""analyze_unmatched_masks.py — the ground-truth masks without a match, split by cause (Table 8 of the paper, segmenter
rows; supplement, note of Table S7), and the matched ground-truth masks against the matched output masks.

Data and rules as scripts/analysis/analyze_classes.py: the cache with the track ids of the full system (--tid-field), the
ground truth decoded per frame (panoctvps.data.lazy_gt); a ground-truth mask is one (thing class, instance id) of a
tracked instance with a global id > 0; the output is the predicted thing masks with a track id > 0; a ground-truth
mask is matched if an output mask has it as its majority instance (cache field gt_global_id) with an IoU above 0.5
(gt_iou).

A ground-truth mask without a match is "not found" (row1a) if the class predicted on most of its pixels is not a
thing class. Otherwise (row1b) the first rule that applies:
  (i)   a predicted thing mask with an IoU above 0.5 with it exists, but has no track id (accurate mask without an
        identity);
  (iii) the predicted thing mask that holds most of its pixels matches another tracked instance (in the mask of
        another object);
  (ii)  the predicted thing mask with the highest IoU with it stays at or below 0.5 and belongs to this instance
        (inaccurate mask);
  rest: the mask holding most pixels belongs to another instance without matching it, lies on no tracked instance,
        or no predicted thing mask overlaps it.
For (i), the reason the mask has no track id: not received by the tracker (not a thing class in the semantic map,
fewer than 10 pixels, no embedding) or received, unmatched and below the initiation threshold of 0.01.
IoUs here are computed from the pixels of the frame (the predicted instance map cropped to the label size); the
check is that rule (i) finds every mask that the cache marks as IoU above 0.5 without a track id.
Matched counts: ground-truth masks whose instance has an output match in that frame and camera, against output thing
masks that match an instance; the sources of a difference are counted one by one (instances with two ground-truth
masks in one image, outputs of a stuff class, instances with two matching outputs).

Usage:
  python scripts/analysis/analyze_unmatched_masks.py --cache-dir outputs/tracker_ctvis_val_cache \
      --tid-field tracker_id_deployed --output outputs/analyses/unmatched_masks.json
"""
from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.data.lazy_gt import LazySequenceDataset
from panoctvps.evaluation.tracker_cells import frame_dets

REPO = Path(__file__).resolve().parents[2]
THING = {2, 3, 4, 5, 8, 9, 10, 11}


def scene(job):
    name, cache_dir, data, TID = job
    npz = np.load(Path(cache_dir) / f"{name}.npz", allow_pickle=True)
    gt = LazySequenceDataset(data, "validation")
    s2i = {f: i for i, f in enumerate(gt.segment_files)}
    g = gt[s2i[str(npz["segment_file"])]]
    gbt = {(fr["timestamp"], c): d for fr in g["frames"] for c, d in fr["cameras"].items()}
    rec = npz["records"]
    C = collections.Counter()
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
        inout = [k for k in range(rs, re) if int(rec[k][TID]) > 0]
        match, n_match = {}, collections.Counter()
        for k in inout:
            if float(rec[k]["gt_iou"]) > 0.5 and int(rec[k]["gt_global_id"]) > 0:
                match[int(rec[k]["gt_global_id"])] = int(rec[k]["class_id"])
                n_match[int(rec[k]["gt_global_id"])] += 1
        # ground-truth masks of the frame
        masks = []
        vals = np.unique(sem.astype(np.int64) * 100000 + inst)
        for v in vals.tolist():
            c, ins = divmod(v, 100000)
            if c in THING and ins > 0 and ins in l2g:
                masks.append((c, ins, l2g[ins]))
        gids_frame = collections.Counter(gid for _, _, gid in masks)
        # matched counts
        for c, ins, gid in masks:
            C["gt_masks"] += 1
            if gid in match:
                C["gt_matched"] += 1
                if gids_frame[gid] > 1:
                    C["gt_matched_object_with_two_masks"] += 1
                if match[gid] not in THING:
                    C["gt_matched_by_non_thing_output"] += 1
        for gid, n in n_match.items():
            if n > 1:
                C["objects_with_two_matching_outputs"] += 1
        for gid in match:
            if gids_frame[gid] > 1:
                C["extra_gt_masks_of_matched_objects"] += gids_frame[gid] - 1
            if gid not in gids_frame:
                C["matched_object_without_gt_mask"] += 1
                C["matched_object_without_gt_mask_thing_output"] += int(match[gid] in THING)
        for k in inout:
            if int(rec[k]["class_id"]) in THING:
                C["out_thing_masks"] += 1
                if int(rec[k]["gt_global_id"]) > 0 and float(rec[k]["gt_iou"]) > 0.5:
                    C["out_matched"] += 1
            elif int(rec[k]["gt_global_id"]) > 0 and float(rec[k]["gt_iou"]) > 0.5:
                C["out_non_thing_matched"] += 1
        # row 1b
        pred_sem = pred_inst = None
        for c, ins, gid in masks:
            if gid in match:
                continue
            if pred_sem is None:
                full = reconstruct_frame(npz, fi)
                pred_sem, pred_inst = full[:2]
                dets = {iid: px for iid, cls, px, c in frame_dets(full[0], full[1], full[5] or {})}
                pred_sem = pred_sem[:sem.shape[0], :sem.shape[1]]
                pred_inst = pred_inst[:sem.shape[0], :sem.shape[1]]
                psize = np.bincount(pred_inst.ravel())
            m = (sem == c) & (inst == ins)
            pc = int(np.bincount(pred_sem[m].astype(np.int64)).argmax())
            if pc not in THING:
                C["row1a"] += 1
                continue
            C["row1b"] += 1
            ov = np.bincount(pred_inst[m])
            area = int(m.sum())
            cand = []
            for lid in np.nonzero(ov)[0].tolist():
                if lid == 0:
                    continue
                r = rec[rs + lid - 1]
                if int(r["class_id"]) not in THING:
                    continue
                n = int(ov[lid])
                cand.append(dict(lid=lid, n=n, iou=n / (area + int(psize[lid]) - n), tid=int(r[TID]),
                                 g=int(r["gt_global_id"]), giou=float(r["gt_iou"])))
            if not cand:
                C["1b_rest_no_thing_mask_overlaps"] += 1
                continue
            hi = max(cand, key=lambda x: x["iou"])
            big = max(cand, key=lambda x: x["n"])
            cache_untracked = [x for x in cand if x["g"] == gid and x["giou"] > 0.5 and x["tid"] <= 0]
            if hi["iou"] > 0.5 and hi["tid"] <= 0:
                C["1b_i_iou_above_half_no_track_id"] += 1
                lid = hi["lid"]
                sc = float(full[3].get(lid, 0.5))
                if lid not in dets:
                    C["i_reason_not_received_not_thing_in_semantic_map"] += 1
                elif dets[lid] < 10:
                    C["i_reason_not_received_under_10_pixels"] += 1
                elif lid not in full[2]:
                    C["i_reason_not_received_no_embedding"] += 1
                elif sc < 0.01:
                    C["i_reason_received_unmatched_confidence_below_0.01"] += 1
                    C["i_conf_below_0.001"] += int(sc < 0.001)
                else:
                    C["i_reason_received_confidence_at_least_0.01_contradiction"] += 1
                C["check_i_cache_agrees"] += int(len(cache_untracked) == 1 and cache_untracked[0]["lid"] == hi["lid"])
            elif big["g"] > 0 and big["g"] != gid and big["giou"] > 0.5:
                C["1b_iii_in_mask_matching_another_object"] += 1
                C["1b_iii_that_mask_has_track_id"] += int(big["tid"] > 0)
            elif hi["iou"] <= 0.5 and hi["g"] == gid:
                C["1b_ii_best_mask_this_object_iou_at_most_half"] += 1
            elif big["g"] > 0 and big["g"] != gid:
                C["1b_rest_most_pixels_in_mask_of_another_object_not_matching"] += 1
            elif big["g"] <= 0:
                C["1b_rest_most_pixels_in_mask_on_no_tracked_object"] += 1
            else:
                C["1b_rest_other"] += 1
            if cache_untracked and not (hi["iou"] > 0.5 and hi["tid"] <= 0):
                C["check_i_missed"] += 1
    return name, dict(C)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/tracker_ctvis_val_cache"))
    ap.add_argument("--dataset-root", default=str(REPO.parent / "data/waymo_open_dataset_v_2_0_1"))
    ap.add_argument("--tid-field", default="tracker_id_deployed")
    ap.add_argument("--scenes", default="", help="Comma-separated scene names (default: the validation split).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    names = []
    for line in (REPO / "configs/splits/validation_frames.txt").read_text().split():
        s = line.split(",")[0]
        if s not in names:
            names.append(s)
    want = set(x for x in a.scenes.split(",") if x)
    names = [n for n in names if not want or n in want]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(scene, [(n, str(a.cache_dir), a.dataset_root, a.tid_field) for n in names]))
    tot = collections.Counter()
    for _, c in res:
        tot.update(c)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"definitions": __doc__, "total": dict(sorted(tot.items())),
                                    "per_scene": {n: c for n, c in res}}, indent=1))
    for k, v in sorted(tot.items()):
        print(k, v)


if __name__ == "__main__":
    main()
