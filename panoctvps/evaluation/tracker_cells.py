#!/usr/bin/env python3
"""Tracker cells scored from the overlap tables (panoctvps.evaluation.overlap_tables), exact and fast.

For each scene: runs the trackers per camera over the frames that the evaluation uses, in the order of
scripts/pipeline/eval_tracker_cached.py (timestamp, camera), with the panoptic reconstruction of the cache
(reconstruct_frame), and records the track id of every predicted thing segment (local instance k, class s). The
scores are then rebuilt from the tables (overlap_tables.score_scene / merge).

Trackers of a cell (a dict, see configs/tracker_cells/): "memorybank" (our tracker; optional motion gate
"gate_px" or "gate_px_per_cam"; component switches "no_class_gate", "beta_one" = the last embedding instead of the
memory embedding, "gates_before_norm" = gated pairs masked before the bi-softmax), "minvis" (final-layer queries
frame_queries of the cache), "iou". The memorybank variant gives the decisions of
MotionGateMemoryBankTracker.update, with the gate checks computed for all pairs at once (a distance within 1e-9 px
of the bound is recomputed with np.linalg.norm as in that code).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.data.constants import THING_CLASSES
from panoctvps.evaluation import overlap_tables
from panoctvps.tracker.iou_tracker import IoUTracker
from panoctvps.tracker.minvis_tracker import MinVISTracker
from panoctvps.tracker.motion_gate import MotionGateMemoryBank, MotionGateMemoryBankTracker, MotionGateTracklet


class LastEmbedTracklet(MotionGateTracklet):
    def update(self, reid_embed, score, class_id, frame_id):
        super().update(reid_embed, score, class_id, frame_id)
        self.similarity_guided_embed = reid_embed.clone()      # beta = 1


class LastEmbedBank(MotionGateMemoryBank):
    def add(self, instance_id):
        self.tracklets[instance_id] = LastEmbedTracklet(instance_id, self.maximum_cache)


class VariantTracker(MotionGateMemoryBankTracker):
    def __init__(self, motion_gate_px, class_gate=True, beta_one=False, gates_before_norm=False, log=None, **kw):
        super().__init__(motion_gate_px=motion_gate_px, **kw)
        self.class_gate, self.beta_one, self.gates_before_norm = class_gate, beta_one, gates_before_norm
        self.log = log                      # optional callable(frame_id, cur, track_ids, sim, ms, allowed, assignments)
        if beta_one:
            self.memory_bank = LastEmbedBank(self.memory_bank.num_dead_frames, self.memory_bank.maximum_cache)
        self.assigned = []

    def update_dets(self, dets, instance_embeddings, instance_scores, frame_id):
        """dets: frame_dets() output (shared by all cells of a frame); same order and filters as
        MotionGateMemoryBankTracker.update."""
        self.assigned = []
        scores = instance_scores or {}
        cur = [{"local_id": iid, "class_id": cls, "embedding": torch.from_numpy(instance_embeddings[iid]).float(),
                "score": float(scores.get(iid, 0.5)), "centroid": c}
               for iid, cls, px, c in dets if px >= 10 and iid in instance_embeddings]
        if not cur:
            self.memory_bank.clean_dead_tracklets(frame_id)
            return None
        cur_embeds = torch.stack([x["embedding"] for x in cur], dim=0)
        if self.memory_bank.empty:
            for x in cur:
                if x["score"] >= self.init_score_threshold:
                    tid = self._next_id
                    self._next_id += 1
                    self._store(tid, x, frame_id)
                    self.assigned.append((x["local_id"], x["class_id"], tid))
        else:
            track_ids, track_embeds, track_classes = self.memory_bank.get_embeddings()
            tids = track_ids.numpy()
            similarity = torch.mm(cur_embeds, track_embeds.t())
            dcls = np.array([x["class_id"] for x in cur])
            allowed = np.ones((len(cur), len(tids)), dtype=bool)
            if self.class_gate:
                allowed &= dcls[:, None] == track_classes.numpy()[None, :]
            if math.isfinite(self.motion_gate_px):
                last = np.array([self.memory_bank.tracklets[int(t)].last_centroid
                                 if self.memory_bank.tracklets[int(t)].last_centroid is not None else (np.nan, np.nan)
                                 for t in tids], dtype=np.float64)
                dc = np.stack([x["centroid"] for x in cur])
                diff = dc[:, None, :] - last[None, :, :]
                dist = np.sqrt(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1])
                near = np.abs(dist - self.motion_gate_px) < 1e-9
                for i, j in zip(*np.nonzero(near)):
                    dist[i, j] = float(np.linalg.norm(dc[i] - last[j]))
                outside = dist > self.motion_gate_px            # NaN (no centroid yet) -> False, as MotionGateMemoryBankTracker
                allowed &= ~outside
            if self.gates_before_norm:
                sim = similarity.masked_fill(torch.from_numpy(~allowed), float("-inf"))
                ms = torch.nan_to_num((sim.softmax(dim=1) + sim.softmax(dim=0)) / 2, nan=0.0).numpy()
            else:
                ms = ((similarity.softmax(dim=1) + similarity.softmax(dim=0)) / 2).numpy()
            used, assignments = set(), []
            for i in sorted(range(len(cur)), key=lambda k: cur[k]["score"], reverse=True):
                best_score, best_tid = -1.0, -1
                for j in np.nonzero(allowed[i])[0].tolist():
                    tj = int(tids[j])
                    if tj in used:
                        continue
                    s = float(ms[i, j])
                    if s < self.match_threshold:
                        continue
                    if s > best_score:
                        best_score, best_tid = s, tj
                if best_tid >= 0:
                    assignments.append((i, best_tid)); used.add(best_tid)
                else:
                    assignments.append((i, None))
            if self.log is not None:
                self.log(frame_id, cur, tids, similarity, ms, allowed, assignments, self)
            for i, tid in assignments:
                x = cur[i]
                if tid is not None:
                    self._store(tid, x, frame_id)
                    self.assigned.append((x["local_id"], x["class_id"], tid))
                elif x["score"] >= self.init_score_threshold:
                    new = self._next_id
                    self._next_id += 1
                    self._store(new, x, frame_id)
                    self.assigned.append((x["local_id"], x["class_id"], new))
        self.memory_bank.clean_dead_tracklets(frame_id)
        return None


def frame_dets(pred_semantic, pred_instance, centroids):
    """(local id, class, pixel count, centroid) of every thing segment, in the tracker's order (THING_CLASSES, then
    ascending instance id); centroid from the cache when given, else from the mask (as MotionGateMemoryBankTracker)."""
    out = []
    for cls_id in THING_CLASSES:
        cls_mask = pred_semantic == cls_id
        if not cls_mask.any():
            continue
        vals = pred_instance[cls_mask]
        cnt = np.bincount(vals.astype(np.int64))
        for iid in np.unique(vals).tolist():
            if iid == 0:
                continue
            px = int(cnt[iid])
            if iid in centroids:
                c = np.asarray(centroids[iid], dtype=np.float64)
            else:
                ys, xs = np.where(cls_mask & (pred_instance == iid))
                c = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
            out.append((int(iid), int(cls_id), px, c))
    return out


def make_trackers(c, cams):
    if c["tracker"] == "iou":
        return {k: IoUTracker(iou_threshold=c["iou_threshold"], max_age=c["num_dead_frames"]) for k in cams}
    if c["tracker"] == "minvis":
        return {k: MinVISTracker(match_threshold=c["match_threshold"], init_score_threshold=c["init_score_threshold"],
                                 num_dead_frames=c["num_dead_frames"]) for k in cams}
    gate = {int(k): float(v) for k, v in (c.get("gate_px_per_cam") or {}).items()}
    if c.get("gate_px") is not None:
        gate = {k: float(c["gate_px"]) for k in cams}
    return {k: VariantTracker(motion_gate_px=gate.get(k, math.inf), class_gate=not c.get("no_class_gate", False),
                              beta_one=c.get("beta_one", False), gates_before_norm=c.get("gates_before_norm", False),
                              match_threshold=c["match_threshold"], init_score_threshold=c["init_score_threshold"],
                              num_dead_frames=c["num_dead_frames"]) for k in cams}


def run_scene(npz, tab, cells):
    """Runs every cell on one scene; returns {cell name: {(f_row, k, s): tid}}."""
    cams = [1, 2, 3, 4, 5]
    trk = {c["name"]: make_trackers(c, cams) for c in cells}
    ids = {c["name"]: {} for c in cells}
    counter = {c["name"]: defaultdict(int) for c in cells}
    need_q = any(c["tracker"] == "minvis" for c in cells)
    fq = npz["frame_queries"] if need_q else None
    for f_row, (fi, cam, ts) in enumerate(tab["frames"].tolist()):
        fr = npz["frame_records"][fi]
        pred_sem, pred_inst, embs, scores, classes, centroids, px = reconstruct_frame(npz, fi)
        embs_q = None
        if need_q:
            embs_q = dict(embs)
            rs, re = int(fr["records_start"]), int(fr["records_end"])
            for k, rec in enumerate(npz["records"][rs:re]):
                embs_q[k + 1] = np.asarray(fq[int(rec["frame_idx"]), int(rec["query_idx"])], dtype=np.float32)
        segs = None
        dets = frame_dets(pred_sem, pred_inst, centroids or {})
        for c in cells:
            name, t = c["name"], trk[c["name"]][cam]
            if c["tracker"] == "memorybank":
                t.update_dets(dets, embs, scores, counter[name][cam])
                for k, s, tid in t.assigned:
                    ids[name][(f_row, k, s)] = tid
            else:
                e = embs_q if c["tracker"] == "minvis" else embs
                tracked = t.update(pred_sem, pred_inst, e, scores, classes, counter[name][cam])
                if segs is None:
                    segs = []
                    for s in THING_CLASSES:
                        cm = pred_sem == s
                        if cm.any():
                            for k in np.unique(pred_inst[cm]).tolist():
                                if k > 0:
                                    segs.append((k, s, cm & (pred_inst == k)))
                for k, s, m in segs:
                    v = np.unique(tracked[m]); v = v[v > 0]
                    if v.size:
                        ids[name][(f_row, k, s)] = int(v[0])
            counter[name][cam] += 1
    return ids


def lookup(idmap, f, k, s):
    out = np.zeros(len(f), dtype=np.int64)
    for n, key in enumerate(zip(f.tolist(), k.tolist(), s.tolist())):
        out[n] = idmap.get(key, 0)
    return out


def score_cell(tables, idmaps):
    scenes = []
    for (name, tab), idm in zip(tables, idmaps):
        cam_of = tab["frames"]["cam"].astype(np.int64)

        def ident_cam(f, k, s, idm=idm):
            return lookup(idm, f, k, s)

        def ident_agg(f, k, s, idm=idm, cam_of=cam_of):
            t = lookup(idm, f, k, s)
            return np.where(t > 0, t + cam_of[f] * 100000, 0)
        scenes.append((name, overlap_tables.score_scene(tab, ident_agg, ident_cam)))
    return scenes


def fill_defaults(cells):
    for c in cells:
        c.setdefault("match_threshold", 0.7); c.setdefault("init_score_threshold", 0.01)
        c.setdefault("num_dead_frames", 5); c.setdefault("iou_threshold", 0.3)
    return cells
