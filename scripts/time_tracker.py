"""Tracker update timing per frame per camera on the cached deployed validation embeddings."""
import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--cache-dir", default="outputs/tracker_ctvis_val_cache")
_ap.add_argument("--output", default="outputs/timing/timing_tracker.json")
_args = _ap.parse_args()
__import__("pathlib").Path(_args.output).parent.mkdir(parents=True, exist_ok=True)

import sys, time, json, glob, statistics, platform
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from panoctvps.data.cache_io import reconstruct_frame
from panoctvps.tracker.motion_gate import MotionGateMemoryBankTracker

CACHE = _args.cache_dir
GATE = {1: 600.0, 2: 800.0, 3: 800.0, 4: 1000.0, 5: 1000.0}
torch.set_num_threads(1)
per_cam = {c: [] for c in range(1, 6)}; n_inst = []
for p in sorted(glob.glob(CACHE + "/*.npz")):
    npz = np.load(p, allow_pickle=True)
    fr = npz["frame_records"]
    trackers = {c: MotionGateMemoryBankTracker(motion_gate_px=GATE[c], match_threshold=0.7, init_score_threshold=0.01,
                                               num_dead_frames=5) for c in range(1, 6)}
    counter = {c: 0 for c in range(1, 6)}
    order = sorted(range(len(fr)), key=lambda i: (int(fr[i]["timestamp"]), int(fr[i]["camera_id"])))
    for i in order:
        cam = int(fr[i]["camera_id"])
        pred_sem, pred_inst, embs, scores, classes, centroids, px = reconstruct_frame(npz, i)   # RLE decode: NOT timed
        t0 = time.perf_counter()
        trackers[cam].update(pred_sem, pred_inst, embs, scores, classes, counter[cam],
                             instance_centroids=centroids)
        per_cam[cam].append(time.perf_counter() - t0); n_inst.append(len(embs)); counter[cam] += 1
    print(p.split("/")[-1][:24], "done", flush=True)
allt = [t for v in per_cam.values() for t in v]
res = {"device": "CPU (tracker tensors are created on CPU; no GPU use)", "cpu": platform.processor() or platform.machine(),
       "torch_threads": 1, "precision": "fp32", "config": "deployed: match 0.7, init 0.01, dead 5, motion gate F600/FL800/FR800/SL1000/SR1000",
       "n_frames_timed": len(allt), "n_scenes": len(glob.glob(CACHE + "/*.npz")), "timed_scope": "MotionGateMemoryBankTracker.update() only: per-frame detection bookkeeping at native resolution + bi-softmax matching + memory update; RLE decoding excluded",
       "mean_detections_per_frame": statistics.mean(n_inst),
       "update_ms_mean": 1000 * statistics.mean(allt), "update_ms_median": 1000 * statistics.median(allt),
       "update_ms_p90": 1000 * sorted(allt)[int(0.9 * len(allt))], "update_ms_max": 1000 * max(allt),
       "per_camera_ms_mean": {c: 1000 * statistics.mean(v) for c, v in per_cam.items()}}
json.dump(res, open(_args.output, "w"), indent=2)
print(json.dumps(res, indent=1))