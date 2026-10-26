"""verify_submission.py — decode submission binprotos and recompute wSTQ.

Decodes each scene's CameraSegmentationLabel back to (pred_sem,
global_instance) and evaluates STQ against the ground truth with the same
metric as scripts/eval_linker_wstq.py, so that the packed validation
predictions can be checked against the validation wSTQ of the full linker
(0.3228). On the test split (no ground truth) it reports the frame counts only.

Usage:
  python scripts/verify_submission.py --split validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES                # noqa: E402
from panoctvps.data.waymo_dataset import WODPVPSSequenceDataset                # noqa: E402
from panoctvps.evaluation import STQuality, merge_sequence_results            # noqa: E402
from panoctvps.evaluation.tracking_utils import build_gt_global_instance      # noqa: E402
from panoctvps.submission import packer                                       # noqa: E402
from panoctvps.submission.proto import wod_camseg_submission_pb2 as pb        # noqa: E402

SUB_DIRS = {"validation": REPO / "submissions/validation/binprotos",
            "test": REPO / "submissions/test/binprotos"}


def decode_submission(path: Path):
    """-> {(timestamp, camera): (pred_sem, global_instance)}, context."""
    sub = pb.CameraSegmentationSubmission()
    sub.ParseFromString(path.read_bytes())
    out, ctx = {}, None
    for fr in sub.predicted_segmentation_labels.frames:
        ctx = fr.context_name
        sem, glob = packer.decode_camera_label(fr.camera_segmentation_label)
        out[(int(fr.frame_timestamp_micros), int(fr.camera_name))] = (sem, glob)
    return out, ctx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--sub-dir", type=Path, default=None)
    ap.add_argument("--dataset-root", default=str(REPO.parent / "data/waymo_open_dataset_v_2_0_1"))
    args = ap.parse_args()
    sub_dir = args.sub_dir or SUB_DIRS[args.split]
    binprotos = sorted(sub_dir.glob("*.binproto"))
    print(f"{len(binprotos)} binprotos in {sub_dir}")

    if args.split == "test":
        tot = 0
        for p in binprotos:
            frames, ctx = decode_submission(p)
            tot += len(frames)
            print(f"  {ctx[:34]:34s} frames={len(frames)}")
        print(f"total frame-cameras: {tot}")
        return

    gt = WODPVPSSequenceDataset(dataset_root=args.dataset_root, split="validation")
    # map context (segment_name) -> gt index via the dataset's segment_files (file stems)
    name_to_gt = {}
    for i in range(len(gt)):
        name_to_gt[gt[i]["segment_name"]] = i

    seq_metrics = []
    t = 0
    for p in binprotos:
        frames, ctx = decode_submission(p)
        if ctx not in name_to_gt:
            print(f"  {ctx[:34]} — GT not found, skip"); continue
        gt_seq = gt[name_to_gt[ctx]]
        gt_by = {}
        for frame in gt_seq["frames"]:
            for cam_id, cam in frame["cameras"].items():
                gt_by[(frame["timestamp"], cam_id)] = cam
        pw = gt_seq["pixel_weights"]
        seq_stq = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES,
                            ignore_label=0)
        for (ts, cam), (pred_sem, glob) in frames.items():
            if (ts, cam) not in gt_by:
                continue
            gt_cam = gt_by[(ts, cam)]
            gt_global = build_gt_global_instance(
                gt_cam["instance_label"], gt_cam["instance_mapping"])
            if pred_sem.shape != gt_cam["semantic_label"].shape:
                continue
            seq_stq.update(pred_semantic=pred_sem.astype(np.int32),
                           pred_instance=glob.astype(np.int64),
                           gt_semantic=gt_cam["semantic_label"],
                           gt_instance=gt_global, pixel_weights=pw[cam])
        seq_metrics.append(seq_stq)
        r = seq_stq.result()
        t += 1
        print(f"  [{t}/{len(binprotos)}] {ctx[:34]:34s} wSTQ={r['STQ']:.4f}", flush=True)

    agg = merge_sequence_results(seq_metrics)
    print("\n=== submission round trip ===")
    print(f"  wSTQ={agg['STQ']:.5f}  wAQ={agg['AQ']:.5f}  wSQ={agg['SQ']:.5f}")


if __name__ == "__main__":
    main()
