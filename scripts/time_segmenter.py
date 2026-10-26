"""Segmenter timing: forward only vs forward + panoptic post-processing, batch 1, native resolution."""
import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--checkpoint", default="checkpoints/segmenter.pth")
_ap.add_argument("--dataset-root", default="../data/waymo_open_dataset_v_2_0_1")
_ap.add_argument("--output", default="outputs/timing/timing_segmenter.json")
_args = _ap.parse_args()
__import__("pathlib").Path(_args.output).parent.mkdir(parents=True, exist_ok=True)

import sys, time, json, statistics, platform
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from panoctvps.segmenter.mask2former import build_model, predict_panoptic
from panoctvps.data.training_dataset import WODMask2FormerDataset
import transformers

CKPT = _args.checkpoint
ROOT = _args.dataset_root
WARM, N = 20, 100

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

dev = "cuda"
model = build_model(backbone="swin-large-mapillary", pretrained=False).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["model"]
new = {remap_key(k): v for k, v in sd.items() if not k.endswith("relative_position_index")}
ln = "model.pixel_level_module.encoder.swin.layernorm."
new[ln + "weight"] = torch.ones(1536); new[ln + "bias"] = torch.zeros(1536)   # unused by the Mask2Former forward
missing, unexpected = model.load_state_dict(new, strict=False)
assert not unexpected and not missing, (missing, unexpected)
nparams = sum(p.numel() for p in model.parameters())

results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__, "precision": "fp32 (no autocast, as in evaluation)",
           "cuda_sync": True, "batch_size": 1, "warmup_iters": WARM, "n_images_per_camera": N,
           "checkpoint": CKPT, "params": nparams, "cameras": {}}

for cam, name in [(1, "FRONT 1920x1280"), (4, "SIDE_LEFT 1920x886")]:
    ds = WODMask2FormerDataset(dataset_root=ROOT, split="validation", cameras=[cam],
                               full_resolution=True)
    idx = list(range(min(len(ds), WARM + N)))
    samples = [ds[i]["pixel_values"].unsqueeze(0) for i in idx]       # preloaded, excluded from timing
    shape = tuple(samples[0].shape[-2:])
    t_fwd, t_full = [], []
    with torch.inference_mode():
        for j, pv in enumerate(samples):
            pv = pv.to(dev, non_blocking=True); torch.cuda.synchronize()
            t0 = time.perf_counter(); out = model(pixel_values=pv); torch.cuda.synchronize(); t1 = time.perf_counter()
            t0b = time.perf_counter(); _ = predict_panoptic(model, pv); torch.cuda.synchronize(); t1b = time.perf_counter()
            if j >= WARM:
                t_fwd.append(t1 - t0); t_full.append(t1b - t0b)
    results["cameras"][name] = {
        "input_tensor_hw": shape, "n_timed": len(t_fwd),
        "forward_ms_mean": 1000 * statistics.mean(t_fwd), "forward_ms_std": 1000 * statistics.pstdev(t_fwd),
        "forward_ms_median": 1000 * statistics.median(t_fwd),
        "forward_plus_postproc_ms_mean": 1000 * statistics.mean(t_full), "forward_plus_postproc_ms_std": 1000 * statistics.pstdev(t_full),
        "forward_plus_postproc_ms_median": 1000 * statistics.median(t_full),
    }
    print(name, json.dumps(results["cameras"][name]), flush=True)
results["peak_gpu_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
json.dump(results, open(_args.output, "w"), indent=2)
print("saved")