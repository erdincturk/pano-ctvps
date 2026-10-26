"""One cell of the alternative trackers (IoU tracker or MinVIS) on the cached segmenter outputs of the validation
split (Table 6; Tables S1 and S2).

Runs scripts/eval_tracker_cached.py unchanged (same cache, same 20 scenes, same STQ code and panoptic
post-processing as the rows of our tracker in Table 6). MinVIS matches the segmenter's own query embeddings: the
final-layer decoder queries after the output layer normalization (cache field frame_queries[frame_idx, query_idx]).
The harness hands the tracker the embeddings of the identity head; this wrapper swaps them for those queries.
The IoU tracker uses no embeddings.
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "scripts" / "eval_tracker_cached.py"

ap = argparse.ArgumentParser()
ap.add_argument("--tracker", choices=("iou", "minvis"), required=True)
ap.add_argument("--iou-threshold", type=float, default=0.3)
ap.add_argument("--num-dead-frames", type=int, default=5)
ap.add_argument("--match-threshold", type=float, default=0.9)
ap.add_argument("--init-score-threshold", type=float, default=0.01)
ap.add_argument("--output", required=True)
ap.add_argument("--cache-dir", default=str(REPO / "outputs" / "tracker_ctvis_val_cache"))
ap.add_argument("--dataset-root", default=str(REPO.parent / "data" / "waymo_open_dataset_v_2_0_1"))
a = ap.parse_args()

sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("harness", HARNESS)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

if a.tracker == "minvis":
    _orig = mod.reconstruct_frame
    def reconstruct_raw(npz, fr_idx):
        pred_sem, pred_inst, embeddings, scores, classes, centroids, px = _orig(npz, fr_idx)
        fr = npz["frame_records"][fr_idx]
        rs, re = int(fr["records_start"]), int(fr["records_end"])
        fq = npz["frame_queries"]
        for k, rec in enumerate(npz["records"][rs:re]):
            embeddings[k + 1] = np.asarray(fq[int(rec["frame_idx"]), int(rec["query_idx"])], dtype=np.float32)
        return pred_sem, pred_inst, embeddings, scores, classes, centroids, px
    mod.reconstruct_frame = reconstruct_raw

sys.argv = ["eval_tracker_cached.py", "--cache-dir", a.cache_dir,
            "--dataset-root", a.dataset_root,
            "--tracker", a.tracker, "--iou-threshold", str(a.iou_threshold),
            "--num-dead-frames", str(a.num_dead_frames), "--match-threshold", str(a.match_threshold),
            "--init-score-threshold", str(a.init_score_threshold), "--output", a.output]
t0 = time.time()
mod.main()
out = json.load(open(a.output))
out["iou_threshold"] = a.iou_threshold if a.tracker == "iou" else None
out["wall_seconds"] = round(time.time() - t0, 1)
json.dump(out, open(a.output, "w"), indent=2)
print(f"CELL DONE {Path(a.output).name}: wSTQ {out['aggregate']['wSTQ']:.4f} wAQ {out['aggregate']['wAQ']:.4f} "
      f"wSQ {out['aggregate']['wSQ']:.4f} ({out['wall_seconds']:.0f}s)")
