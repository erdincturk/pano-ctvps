"""time_pipeline.py — runtime of the pipeline per frame (Section VI-C of the paper): the three stages one after another, the causal linker per frame
and camera pair, and the tracker against the frame index.

Method as for the other timing scripts: time.perf_counter around each call, torch.cuda.synchronize() before and
after every GPU call, one CPU thread, fp32, batch size 1.

Subcommands
  pipeline  One validation scene (default: the scene with the median number of tracked detections per frame).
            Per timestamp: the five images are loaded (read and decoded from the dataset files and normalised as
            in the evaluation), segmented (forward and panoptic post-processing), then the tracker and the causal
            linker process the frame. The tracker and the linker read the cached outputs of the deployed model, so
            their times are those of the deployed system. Times are reported with and without image loading.
  linker    The causal linker (scripts/experiments/online_linker.py, k = 1) on the cached tracks of all validation scenes,
            computed live frame by frame (the adapter of (15) is applied to the embeddings of the scene once, at
            setup, as in online_linker): per timestamp, the band test of every tracked thing mask (proposal (13)),
            the panoramic projection of the masks that a candidate pair needs (once per detection), the clipped
            overlap u of every candidate pair present in the frame (for a pair proposed in this frame, also over
            its earlier shared frames, as the running statistics of the causal linker include them), the raw and
            adapted cosines, then the decision step. Logged per frame and camera pair: the time of the geometry
            (band test, projection, comparison), the number of candidate pairs present, the tracks of each camera
            and those in the overlap band; per frame: the time of the decision step. Check: the identities equal
            those of online_linker.simulate_scene (k = 1) and the overlaps equal its pair table (--check).
  tracker   The deployed tracker (θ_match 0.7, θ_init 0.01, N_dead 5, motion gate 600/800/800/1000/1000 px) on the
            cached outputs of all validation scenes, with the frame index, the number of detections and the number
            of active tracks logged per frame and camera (decoding of the cached masks not timed).

Each subcommand also reports the peak resident memory of its process; pipeline reports the peak GPU memory
allocated by tensors (torch.cuda.max_memory_allocated) and reserved by the allocator.

Usage:
  python scripts/experiments/time_pipeline.py tracker --output outputs/timing/timing_tracker_frames.json
  python scripts/experiments/time_pipeline.py linker --output outputs/timing/timing_linker_frames.json [--check 3]
  python scripts/experiments/time_pipeline.py pipeline --checkpoint <segmenter ckpt> --output outputs/timing/timing_pipeline.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # online_linker.py, in this folder

from panoctvps.data.cache_io import decode_rle, reconstruct_frame  # noqa: E402
from panoctvps.linker.adapter import EmbeddingAdapter  # noqa: E402
from panoctvps.linker.assigner import _rank_factor  # noqa: E402
from panoctvps.linker.overlap import ADJACENT_PAIRS, overlap_by_partner  # noqa: E402
from panoctvps.linker.panorama import (build_overlap_pano_mask, build_panorama_lut,  # noqa: E402
                                       project_mask_to_panorama_forward)
from panoctvps.linker.proposer import PARTNERS, SUPERCLASS, _in_overlap  # noqa: E402
from panoctvps.linker.scorer import _ext4  # noqa: E402
from panoctvps.tracker.motion_gate import MotionGateMemoryBankTracker  # noqa: E402

import online_linker as ol  # noqa: E402

torch.set_num_threads(1)
GATE = {1: 600.0, 2: 800.0, 3: 800.0, 4: 1000.0, 5: 1000.0}
TID = "tracker_id_deployed"
PNAME = {frozenset((a, b)): f"{a}-{b}" for a, b in ADJACENT_PAIRS}
CAM_NAME = {1: "F", 2: "FL", 3: "FR", 4: "SL", 5: "SR"}


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def stats(v):
    v = list(v)
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": len(v), "mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def remap_key(k):
    """Swin key names of checkpoints saved with transformers 5.3 -> the names of transformers 5.15 (no-op otherwise)."""
    k = k.replace(".encoder.embeddings.", ".encoder.swin.embeddings.").replace(".encoder.encoder.", ".encoder.swin.encoder.")
    k = k.replace(".attention.self.query.", ".attention.q_proj.").replace(".attention.self.key.", ".attention.k_proj.")
    k = k.replace(".attention.self.value.", ".attention.v_proj.").replace(".attention.output.dense.", ".attention.o_proj.")
    k = k.replace(".attention.self.relative_position_bias_table",
                  ".attention.relative_position_bias.relative_position_bias_table")
    k = k.replace(".intermediate.dense.", ".mlp.fc1.").replace(".output.dense.", ".mlp.fc2.")
    return k


# --------------------------------------------------------------------------------------------------------------
class LiveLinker:
    """Causal linker (k = 1), computed frame by frame from the cache of one scene."""

    def __init__(self, npz, adapter: EmbeddingAdapter, cfg: dict):
        t0 = time.perf_counter()
        self.npz, self.cfg = npz, cfg
        self.rec, self.frs, self.rle = npz["records"], npz["frame_records"], npz["mask_rle"]
        self.cams = ol._load_cams(npz)
        self.ovbp = overlap_by_partner(self.cams)
        self.ext4 = {c: _ext4(self.cams[c]) for c in self.cams}
        luts = {c: build_panorama_lut(self.ext4[c], self.cams[c]["intr9"], (self.cams[c]["h"], self.cams[c]["w"]))
                for c in self.cams}
        self.ovf = {}
        for a, b in ADJACENT_PAIRS:
            if a in self.cams and b in self.cams:
                o = build_overlap_pano_mask(luts[a], luts[b], np.ones((self.cams[a]["h"], self.cams[a]["w"]), bool),
                                            np.ones((self.cams[b]["h"], self.cams[b]["w"]), bool)).ravel()
                self.ovf[(a, b)] = o; self.ovf[(b, a)] = o
        emb = np.asarray(npz["embeddings"], np.float32)
        self.emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        with torch.no_grad():                         # adapter of (15), on the raw embeddings, as online_linker
            ad = adapter(torch.from_numpy(np.asarray(npz["embeddings"], np.float32))).numpy()
        self.emb_ad = ad / np.maximum(np.linalg.norm(ad, axis=1, keepdims=True), 1e-8)
        self.setup_s = time.perf_counter() - t0
        ts_list = sorted({int(fr["timestamp"]) for fr in self.frs})
        self.ts_index = {ts: i for i, ts in enumerate(ts_list)}
        self.frames_at = collections.defaultdict(list)
        for fr in self.frs:
            self.frames_at[self.ts_index[int(fr["timestamp"])]].append(fr)
        self.n_ts = len(ts_list)
        # state
        self.hist = collections.defaultdict(dict)       # track -> {ti: record k}
        self.born = {}
        self.proj_cache, self.ad_cache = {}, {}
        self.pairs = {}                                 # pair key (sorted tuple) -> state
        self.parent, self.camset, self.ident = {}, {}, {}
        self.cls_cnt, self.dom = collections.defaultdict(collections.Counter), {}
        self.emitted = {}
        self.u_log = {}                                 # (pair, ti) -> u (nan = no evidence)

    def _proj(self, cam, k):
        if k not in self.proj_cache:
            m = decode_rle(self.rle[k])
            h, w = self.cams[cam]["h"], self.cams[cam]["w"]
            if m.shape != (h, w):
                m = m[:h, :w]
            self.proj_cache[k] = np.flatnonzero(project_mask_to_panorama_forward(m, self.ext4[cam], self.cams[cam]["intr9"]))
        return self.proj_cache[k]

    def _ad(self, k):
        return self.emb_ad[k]

    def _find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def _u(self, a, ka, b, kb):
        ia, ib = self._proj(a, ka), self._proj(b, kb)
        o = self.ovf[(a, b)]
        ia = ia[o[ia]]; ib = ib[o[ib]]
        if ia.size or ib.size:
            inter = np.intersect1d(ia, ib, assume_unique=True).size
            union = ia.size + ib.size - inter
            return inter / union if union else 0.0
        return float("nan")

    def step(self, ti):
        """Process timestamp ti. Returns the per-pair log and the decision time."""
        pl = {PNAME[frozenset(p)]: {"band_s": 0.0, "proposal_s": 0.0, "proj_cmp_s": 0.0, "pairs_present": 0,
                                    "pairs_new": 0, "backfilled_frames": 0, "tracks_a": 0, "tracks_b": 0,
                                    "band_a": 0, "band_b": 0} for p in ADJACENT_PAIRS}
        # 1. detections, band test (proposal (13)), track bookkeeping
        cam_dets = {}
        for fr in self.frames_at[ti]:
            cam = int(fr["camera_id"])
            dets = []
            for k in range(int(fr["records_start"]), int(fr["records_end"])):
                x = self.rec[k]
                sc = SUPERCLASS.get(int(x["class_id"]))
                tid = int(x[TID])
                if sc is None or tid <= 0:
                    continue
                t = (cam, tid)
                if t not in self.parent:
                    self.parent[t] = t; self.camset[t] = {cam}; self.born[t] = ti; self.ident[t] = (ti, cam, tid)
                self.hist[t][ti] = k
                mask = decode_rle(self.rle[k])
                mem = {}
                for b in PARTNERS[cam]:
                    t0 = time.perf_counter()
                    mem[b] = _in_overlap(mask, self.ovbp.get((cam, b)))
                    pl[PNAME[frozenset((cam, b))]]["band_s"] += time.perf_counter() - t0
                dets.append((tid, sc, mem, k, int(x["class_id"])))
            cam_dets[cam] = dets
        for a, b in ADJACENT_PAIRS:
            r = pl[PNAME[frozenset((a, b))]]
            r["tracks_a"] = len(cam_dets.get(a, [])); r["tracks_b"] = len(cam_dets.get(b, []))
            if a not in cam_dets or b not in cam_dets:
                continue
            t0 = time.perf_counter()
            a_in = [(d[0], d[1]) for d in cam_dets[a] if d[2].get(b)]
            b_in = [(d[0], d[1]) for d in cam_dets[b] if d[2].get(a)]
            r["band_a"], r["band_b"] = len(a_in), len(b_in)
            for ta, sa in a_in:
                for tb, sb in b_in:
                    if sa == sb:
                        key = tuple(sorted([(a, ta), (b, tb)]))
                        if key not in self.pairs:
                            self.pairs[key] = {"first": ti, "n_ok": 0, "S": 0.0, "raw": ol._RunningMedian(),
                                               "ad": ol._RunningMedian(), "decided": False, "new": True}
                            r["pairs_new"] += 1
            r["proposal_s"] += time.perf_counter() - t0
        # 2. per pair present in this frame: u and cosines (and backfill for new pairs)
        present = []
        for key, p in self.pairs.items():
            (a, ta), (b, tb) = key
            ha, hb = self.hist[(a, ta)], self.hist[(b, tb)]
            if ti not in ha or ti not in hb:
                continue
            r = pl[PNAME[frozenset((a, b))]]
            t0 = time.perf_counter()
            frames = [ti]
            if p.pop("new", False):
                frames = sorted(set(ha) & set(hb))      # all shared frames so far (<= ti)
                r["backfilled_frames"] += len(frames) - 1
            for tj in frames:
                ka, kb = ha[tj], hb[tj]
                u = self._u(a, ka, b, kb)
                self.u_log[(key, tj)] = u
                if not np.isnan(u):
                    p["n_ok"] += 1; p["S"] += u
                p["raw"].add(float(self.emb[ka] @ self.emb[kb])); p["ad"].add(float(self._ad(ka) @ self._ad(kb)))
            r["proj_cmp_s"] += time.perf_counter() - t0
            r["pairs_present"] += 1
            present.append(key)
        # 3. decision step (as online_linker.simulate_scene, k = 1)
        t0 = time.perf_counter()
        for cam, dets in cam_dets.items():
            for d in dets:
                t = (cam, d[0]); cnt = self.cls_cnt[t]; cnt[d[4]] += 1
                self.dom[t] = cnt.most_common(1)[0][0]
        acc = []
        for key in present:
            p = self.pairs[key]
            if p["decided"] or p["n_ok"] < 1:
                continue
            a, b = key
            S = p["S"] / p["n_ok"]
            if ol._rule(S, p["ad"].median(), self.dom[a], self.dom[b], a, b, self.cfg):
                acc.append((S * _rank_factor(p["raw"].median()), key))
        acc.sort(key=lambda e: (-e[0], e[1]))
        for _, key in acc:
            self.pairs[key]["decided"] = True
            a, b = key
            ra, rb = self._find(a), self._find(b)
            if ra == rb or self.camset[ra] & self.camset[rb]:
                continue
            self.parent[rb] = ra; self.camset[ra] |= self.camset[rb]
            self.ident[ra] = min(self.ident[ra], self.ident[rb])
        for cam, dets in cam_dets.items():
            for d in dets:
                idt = self.ident[self._find((cam, d[0]))]
                self.emitted[(ti, cam, d[0])] = idt[1] * ol.CAM_NS + idt[2]
        dec = time.perf_counter() - t0
        return pl, dec

    def final(self):
        return {t: (lambda idt: idt[1] * ol.CAM_NS + idt[2])(self.ident[self._find(t)]) for t in self.parent}


def cmd_linker(args):
    cfg = ol.load_config(args.config)
    adapter = EmbeddingAdapter.load(args.adapter)
    paths = sorted(args.cache_dir.glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    frames, setups, checks = [], {}, {}
    for si, p in enumerate(paths):
        npz = np.load(p, allow_pickle=True)
        L = LiveLinker(npz, adapter, cfg)
        setups[p.stem] = L.setup_s
        for ti in range(L.n_ts):
            t0 = time.perf_counter()
            pl, dec = L.step(ti)
            tot = time.perf_counter() - t0
            frames.append({"scene": p.stem, "frame": ti, "total_s": tot, "decision_s": dec, "pairs": pl})
        if si < args.check:
            pt = ol.build_pair_table(p, args.adapter, args.cosines_dir / p.name, cfg)
            sim = ol.simulate_scene(pt, 1, cfg)
            fin = L.final()
            same_ids = fin == sim["final"]
            emitted_same = L.emitted == sim["emitted"]
            du, n = 0.0, 0
            for q in pt["pairs"]:
                key = tuple(sorted([q["a"], q["b"]]))
                for j, tj in enumerate(q["ts"].tolist()):
                    if (key, tj) in L.u_log:
                        a_, b_ = q["iou"][j], L.u_log[(key, tj)]
                        if not (np.isnan(a_) and np.isnan(b_)):
                            du = max(du, abs(a_ - b_)); n += 1
            n_pairs_same = len(pt["pairs"]) == len(L.pairs)
            checks[p.stem] = {"final_ids_equal": same_ids, "emitted_ids_equal": emitted_same, "pairs_equal": n_pairs_same,
                              "u_compared": n, "u_max_abs_diff": du}
            print("check", p.stem[:24], checks[p.stem], flush=True)
        print(f"[{si + 1}/{len(paths)}] {p.stem[:24]} frames={L.n_ts} pairs={len(L.pairs)} "
              f"mean={np.mean([f['total_s'] for f in frames if f['scene'] == p.stem]):.3f}s", flush=True)
    out = {"definitions": __doc__, "cache_dir": str(args.cache_dir), "n_frames": len(frames), "setup_s_per_scene": setups,
           "frame_total_s": stats(f["total_s"] for f in frames), "decision_s": stats(f["decision_s"] for f in frames),
           "geometry_s": stats(f["total_s"] - f["decision_s"] for f in frames),
           "peak_rss_mb": maxrss_mb(), "checks": checks, "frames": frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, default=float))
    print(json.dumps({k: out[k] for k in ("n_frames", "frame_total_s", "decision_s", "geometry_s", "peak_rss_mb")}, indent=1))


# --------------------------------------------------------------------------------------------------------------
def cmd_tracker(args):
    paths = sorted(args.cache_dir.glob("*.npz"))
    if args.limit:
        paths = paths[:args.limit]
    rows = []
    for p in paths:
        npz = np.load(p, allow_pickle=True)
        fr = npz["frame_records"]
        trk = {c: MotionGateMemoryBankTracker(motion_gate_px=GATE[c], match_threshold=0.7, init_score_threshold=0.01,
                                              num_dead_frames=5) for c in range(1, 6)}
        counter = {c: 0 for c in range(1, 6)}
        order = sorted(range(len(fr)), key=lambda i: (int(fr[i]["timestamp"]), int(fr[i]["camera_id"])))
        for i in order:
            cam = int(fr[i]["camera_id"])
            pred_sem, pred_inst, embs, scores, classes, centroids, px = reconstruct_frame(npz, i)   # not timed
            n_active = 0 if trk[cam].memory_bank.empty else len(trk[cam].memory_bank.tracklets)
            t0 = time.perf_counter()
            trk[cam].update(pred_sem, pred_inst, embs, scores, classes, counter[cam], instance_centroids=centroids)
            dt = time.perf_counter() - t0
            rows.append({"scene": p.stem, "camera": cam, "frame": counter[cam], "time_s": dt, "detections": len(embs),
                         "active_tracks": n_active})
            counter[cam] += 1
        print(p.stem[:24], len(rows), flush=True)
    by_idx = collections.defaultdict(list)
    for r in rows:
        by_idx[r["frame"]].append(r["time_s"])
    out = {"definitions": __doc__, "cache_dir": str(args.cache_dir), "n_frames": len(rows),
           "time_s": stats(r["time_s"] for r in rows),
           "mean_time_s_by_frame_index": {int(k): float(np.mean(v)) for k, v in sorted(by_idx.items())},
           "peak_rss_mb": maxrss_mb(), "frames": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, default=float))
    print(json.dumps({k: out[k] for k in ("n_frames", "time_s", "peak_rss_mb")}, indent=1))


# --------------------------------------------------------------------------------------------------------------
def cmd_pipeline(args):
    from panoctvps.data.training_dataset import WODMask2FormerDataset
    from panoctvps.segmenter.mask2former import build_model, predict_panoptic
    cfg = ol.load_config(args.config)
    adapter = EmbeddingAdapter.load(args.adapter)
    paths = sorted(args.cache_dir.glob("*.npz"))
    if args.scene:
        scene = args.scene
    else:
        dens = {}
        for p in paths:
            r = np.load(p, allow_pickle=True)["records"]
            ok = (r[TID] > 0) & np.isin(r["class_id"], list(SUPERCLASS))
            dens[p.stem] = ok.sum() / len(np.load(p, allow_pickle=True)["frame_records"])
        order = sorted(dens, key=dens.get)
        scene = order[len(order) // 2]
    npz = np.load(args.cache_dir / f"{scene}.npz", allow_pickle=True)
    dev = "cuda"
    model = build_model(backbone="swin-large-mapillary", pretrained=False).to(dev).eval()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)["model"]
    new = {remap_key(k): v for k, v in sd.items() if not k.endswith("relative_position_index")}
    ln = "model.pixel_level_module.encoder.swin.layernorm."
    if ln + "weight" not in new:
        new[ln + "weight"] = torch.ones(1536); new[ln + "bias"] = torch.zeros(1536)
    missing, unexpected = model.load_state_dict(new, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    ds = WODMask2FormerDataset(dataset_root=args.dataset_root, split="validation", cameras=[1, 2, 3, 4, 5],
                               full_resolution=True)
    seg_file = str(npz["segment_file"])
    idx = collections.defaultdict(dict)
    for i in range(len(ds)):
        e = ds._base.index[i]
        if Path(str(e["segment_file"])).name == Path(seg_file).name:
            idx[int(e["timestamp"])][int(e["camera_id"])] = i
    fr = npz["frame_records"]
    fr_at = {(int(f["timestamp"]), int(f["camera_id"])): j for j, f in enumerate(fr)}
    trk = {c: MotionGateMemoryBankTracker(motion_gate_px=GATE[c], match_threshold=0.7, init_score_threshold=0.01,
                                          num_dead_frames=5) for c in range(1, 6)}
    counter = {c: 0 for c in range(1, 6)}
    L = LiveLinker(npz, adapter, cfg)
    ts_sorted = sorted(L.ts_index, key=L.ts_index.get)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with torch.inference_mode():
        # warm-up: 5 images, not timed
        for c in range(1, 6):
            pv = ds[idx[ts_sorted[0]][c]]["pixel_values"].unsqueeze(0).to(dev)
            predict_panoptic(model, pv); torch.cuda.synchronize()
        for ts in (ts_sorted[:args.max_frames] if args.max_frames else ts_sorted):
            t_load = t_seg = t_trk = 0.0
            for c in range(1, 6):
                t0 = time.perf_counter(); pv = ds[idx[ts][c]]["pixel_values"].unsqueeze(0); t_load += time.perf_counter() - t0
                pv = pv.to(dev); torch.cuda.synchronize()
                t0 = time.perf_counter(); predict_panoptic(model, pv); torch.cuda.synchronize(); t_seg += time.perf_counter() - t0
            for c in range(1, 6):
                j = fr_at[(ts, c)]
                pred_sem, pred_inst, embs, scores, classes, centroids, px = reconstruct_frame(npz, j)   # cached outputs
                t0 = time.perf_counter()
                trk[c].update(pred_sem, pred_inst, embs, scores, classes, counter[c], instance_centroids=centroids)
                t_trk += time.perf_counter() - t0
                counter[c] += 1
            t0 = time.perf_counter(); L.step(L.ts_index[ts]); t_lnk = time.perf_counter() - t0
            rows.append({"timestamp": ts, "load_s": t_load, "segmenter_s": t_seg, "tracker_s": t_trk, "linker_s": t_lnk,
                         "total_without_loading_s": t_seg + t_trk + t_lnk, "total_with_loading_s": t_load + t_seg + t_trk + t_lnk})
            print(ts, {k: round(v, 3) for k, v in rows[-1].items() if k != "timestamp"}, flush=True)
    out = {"definitions": __doc__, "scene": scene, "checkpoint": str(args.checkpoint), "n_frames": len(rows),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           **{k: stats(r[k] for r in rows) for k in ("load_s", "segmenter_s", "tracker_s", "linker_s",
                                                    "total_without_loading_s", "total_with_loading_s")},
           "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
           "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "peak_rss_mb": maxrss_mb(),
           "linker_setup_s": L.setup_s, "frames": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("frames", "definitions")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pipeline", "linker", "tracker"))
    ap.add_argument("--cache-dir", type=Path, default=REPO / "outputs/tracker_ctvis_val_cache")
    ap.add_argument("--adapter", type=Path, default=REPO / "outputs/linker/adapter.pt")
    ap.add_argument("--cosines-dir", type=Path, default=REPO / "outputs/linker/adapted_cosines")
    ap.add_argument("--config", type=Path, default=REPO / "configs/linker_deployed.json")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "checkpoints/segmenter.pth")
    ap.add_argument("--dataset-root", default=str(REPO.parent / "data/waymo_open_dataset_v_2_0_1"))
    ap.add_argument("--scene", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (test)")
    ap.add_argument("--max-frames", type=int, default=0, help="pipeline: first N timestamps only (test)")
    ap.add_argument("--check", type=int, default=0, help="linker: compare the first N scenes with simulate_scene")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    {"pipeline": cmd_pipeline, "linker": cmd_linker, "tracker": cmd_tracker}[args.cmd](args)


if __name__ == "__main__":
    main()
