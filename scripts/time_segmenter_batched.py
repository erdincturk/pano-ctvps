"""Segmenter timing with the five images of one timestep processed as two batches
(FRONT/FRONT_LEFT/FRONT_RIGHT 1920x1280 as one batch of 3; SIDE_LEFT/SIDE_RIGHT 1920x886,
padded to 896 rows, as one batch of 2), plus a batch-1 FRONT reference in the same session.
Same protocol as scripts/time_segmenter.py: fp32, inputs on the GPU before timing,
torch.cuda.synchronize() around each call, HF panoptic post-processing included, 20 warm-up
timesteps then 100 timed timesteps taken in order from the validation split. Timing only; the
evaluation code of the package is used unchanged.
"""
import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--checkpoint", default="checkpoints/segmenter.pth")
_ap.add_argument("--dataset-root", default="../data/waymo_open_dataset_v_2_0_1")
_ap.add_argument("--output", default="outputs/timing/timing_segmenter_batched.json")
_args = _ap.parse_args()
__import__("pathlib").Path(_args.output).parent.mkdir(parents=True, exist_ok=True)

import sys, time, json, statistics
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import transformers
from panoctvps.segmenter.mask2former import build_model, predict_panoptic
from panoctvps.data.constants import NUM_CLASSES, PARQUET_COLS
from panoctvps.data.decode import decode_jpeg
from panoctvps.data.transforms.augmentations import PadToMultiple
from panoctvps.data.transforms.formatting import ToTensorNormalize

CKPT = _args.checkpoint
ROOT = Path(_args.dataset_root) / "validation"
OUT = Path(_args.output)
WARM, N = 20, 100
FRONT_CAMS, SIDE_CAMS = (1, 2, 3), (4, 5)

# Checkpoints saved with older transformers versions use older Swin parameter
# names; remap_key() maps them to the current names. The current Swin model
# also has a final backbone LayerNorm (swin.layernorm) that the Mask2Former
# forward does not use; it is filled with identity values below.
def remap_key(k):
    k = k.replace(".encoder.embeddings.", ".encoder.swin.embeddings.").replace(".encoder.encoder.", ".encoder.swin.encoder.")
    k = k.replace(".attention.self.query.", ".attention.q_proj.").replace(".attention.self.key.", ".attention.k_proj.").replace(".attention.self.value.", ".attention.v_proj.")
    k = k.replace(".attention.output.dense.", ".attention.o_proj.")
    k = k.replace(".attention.self.relative_position_bias_table", ".attention.relative_position_bias.relative_position_bias_table")
    k = k.replace(".intermediate.dense.", ".mlp.fc1.").replace(".output.dense.", ".mlp.fc2.")
    return k

# --- model (as in scripts/time_segmenter.py) ------------------------------------------------
dev = "cuda"
model = build_model(backbone="swin-large-mapillary", num_classes=NUM_CLASSES, pretrained=False).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
new = {remap_key(k): v for k, v in sd.items() if not k.endswith("relative_position_index")}
ln = "model.pixel_level_module.encoder.swin.layernorm."
new[ln + "weight"] = torch.ones(1536); new[ln + "bias"] = torch.zeros(1536)   # unused by the forward
missing, unexpected = model.load_state_dict(new, strict=False)
assert not missing and not unexpected, (missing, unexpected)

# --- first WARM+N timesteps of the validation split, in split-file order ----------------------
ts_by_ctx = {}
for line in (REPO / "configs/splits/validation_frames.txt").read_text().splitlines():
    if line.strip():
        c, t = line.strip().split(","); ts_by_ctx.setdefault(c, []).append(int(t))
needed = []
for ctx in sorted(ts_by_ctx):
    for t in sorted(ts_by_ctx[ctx]):
        needed.append((ctx, t))
        if len(needed) == WARM + N: break
    if len(needed) == WARM + N: break

pad, fmt = PadToMultiple(multiple=32), ToTensorNormalize()
def prep(img):
    s = {"image": img, "semantic_label": np.zeros(img.shape[:2], np.int32),
         "instance_label": np.zeros(img.shape[:2], np.int32)}
    return fmt(pad(s))["image"]

timesteps = []          # list of (front_batch (3,3,1280,1920), side_batch (2,3,896,1920)) on CPU
cur_ctx, df = None, None
for ctx, t in needed:
    if ctx != cur_ctx:
        df = pq.read_table(ROOT / "camera_image" / f"{ctx}.parquet",
                           columns=[PARQUET_COLS["img_key_timestamp"], PARQUET_COLS["img_key_camera"],
                                    PARQUET_COLS["img_image"]]).to_pandas()
        cur_ctx = ctx
    rows = df[df[PARQUET_COLS["img_key_timestamp"]] == t]
    imgs = {int(r[PARQUET_COLS["img_key_camera"]]): decode_jpeg(r[PARQUET_COLS["img_image"]]) for _, r in rows.iterrows()}
    assert set(imgs) >= {1, 2, 3, 4, 5}, (ctx, t, sorted(imgs))
    timesteps.append((torch.stack([prep(imgs[c]) for c in FRONT_CAMS]),
                      torch.stack([prep(imgs[c]) for c in SIDE_CAMS])))
print(f"loaded {len(timesteps)} timesteps; shapes {tuple(timesteps[0][0].shape)} {tuple(timesteps[0][1].shape)}", flush=True)

results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__, "precision": "fp32 (no autocast, as in evaluation)",
           "cuda_sync": True, "warmup_timesteps": WARM, "n_timed_timesteps": N, "checkpoint": CKPT,
           "split": "validation", "first_timestep": list(needed[0]), "last_timestep": list(needed[-1]),
           "postprocessing": "HF Mask2FormerImageProcessor.post_process_panoptic_segmentation (predict_panoptic), included"}

def stats(ts):
    return {"ms_mean": 1000 * statistics.mean(ts), "ms_std": 1000 * statistics.pstdev(ts),
            "ms_median": 1000 * statistics.median(ts), "n": len(ts)}

# --- 1. two batches per timestep, back to back ------------------------------------------------
torch.cuda.reset_peak_memory_stats()
t_front, t_side, t_total = [], [], []
with torch.inference_mode():
    for j, (fb, sb) in enumerate(timesteps):
        fb = fb.to(dev); sb = sb.to(dev); torch.cuda.synchronize()
        t0 = time.perf_counter(); _ = predict_panoptic(model, fb); torch.cuda.synchronize(); t1 = time.perf_counter()
        _ = predict_panoptic(model, sb); torch.cuda.synchronize(); t2 = time.perf_counter()
        if j >= WARM:
            t_front.append(t1 - t0); t_side.append(t2 - t1); t_total.append(t2 - t0)
        del fb, sb
results["batched_per_timestep"] = {
    "front_batch_3x1920x1280_fwd_plus_postproc": stats(t_front),
    "side_batch_2x1920x886_pad896_fwd_plus_postproc": stats(t_side),
    "both_batches_total": stats(t_total),
    "peak_gpu_mem_allocated_GB": torch.cuda.max_memory_allocated() / 1e9,
    "peak_gpu_mem_reserved_GB": torch.cuda.max_memory_reserved() / 1e9,
}
print("1:", json.dumps(results["batched_per_timestep"]), flush=True)

# --- 2. batch-1 FRONT reference in the same session ------------------------------------------
t_b1 = []
with torch.inference_mode():
    for j, (fb, _) in enumerate(timesteps):
        pv = fb[0:1].to(dev); torch.cuda.synchronize()
        t0 = time.perf_counter(); _ = predict_panoptic(model, pv); torch.cuda.synchronize(); t1 = time.perf_counter()
        if j >= WARM:
            t_b1.append(t1 - t0)
results["front_batch1_1920x1280_fwd_plus_postproc"] = stats(t_b1)
print("2:", json.dumps(results["front_batch1_1920x1280_fwd_plus_postproc"]), flush=True)
results["peak_gpu_mem_allocated_GB_session"] = torch.cuda.max_memory_allocated() / 1e9
json.dump(results, open(OUT, "w"), indent=2)
print("saved", OUT)
