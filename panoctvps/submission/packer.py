"""TF-free packer for the WOD 2D Video Panoptic Segmentation submission.

Reimplements waymo_open_dataset.utils.camera_segmentation_utils
.save_panoptic_label_to_proto in pure numpy + cv2 (the upstream util requires
TensorFlow, which does not build on this hardware). Produces the same
CameraSegmentationLabel wire format via the vendored proto
(panoctvps/submission/proto/wod_camseg_submission_pb2.py).

Encoding contract (matches the upstream util and what the Waymo scorer expects):
  * The panoptic PNG is uint16, value = semantic_class * divisor + LOCAL
    instance id, where local ids are compact sequential values (1..N) WITHIN
    one camera frame; stuff pixels carry local id 0.
  * instance_id_to_global_id_mapping carries local -> GLOBAL id (our
    cross-camera/temporal pr_global_id); is_tracked = True for thing classes.
    The scorer reads global ids through this mapping, which is how cross-camera
    and temporal instance consistency is conveyed.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import cv2
import numpy as np

from panoctvps.data.constants import PANOPTIC_LABEL_DIVISOR, THING_CLASSES
from panoctvps.submission.proto import wod_camseg_submission_pb2 as pb

CAMERA_LEFT_TO_RIGHT = (4, 2, 1, 3, 5)        # SIDE_LEFT..SIDE_RIGHT
_THING = set(THING_CLASSES)
_U16_MAX = np.iinfo(np.uint16).max


def encode_panoptic_png(panoptic_u16: np.ndarray) -> bytes:
    """16-bit grayscale PNG bytes from a uint16 panoptic map."""
    if panoptic_u16.dtype != np.uint16:
        panoptic_u16 = panoptic_u16.astype(np.uint16)
    ok, buf = cv2.imencode(".png", panoptic_u16)
    if not ok:
        raise RuntimeError("cv2 failed to encode panoptic PNG")
    return buf.tobytes()


def pack_camera_label(pred_semantic: np.ndarray,
                      pred_global_instance: np.ndarray,
                      sequence_id: str,
                      divisor: int = PANOPTIC_LABEL_DIVISOR
                      ) -> pb.CameraSegmentationLabel:
    """Pack one camera frame's prediction into a CameraSegmentationLabel.

    pred_semantic: (H, W) int — semantic class id per pixel (0..28).
    pred_global_instance: (H, W) int — pr_global_id per thing pixel, 0 for
        stuff / no-instance. Global ids must be consistent across cameras and
        time within the sequence (the output of cross-camera linking).
    """
    h, w = pred_semantic.shape
    sem = pred_semantic.astype(np.int64)
    gi = pred_global_instance.astype(np.int64)
    # Thing pixels only carry an instance; force stuff/non-thing to 0.
    is_thing = np.isin(sem, list(_THING)) & (gi > 0)
    gi = np.where(is_thing, gi, 0)

    # Compact relabel global -> sequential local (1..N) within this frame.
    globals_present = np.unique(gi[gi > 0])
    local_of_global: Dict[int, int] = {int(g): i + 1
                                       for i, g in enumerate(globals_present.tolist())}
    if len(globals_present):
        max_local = len(globals_present)
        eff_div = max(divisor, max_local + 1)
    else:
        eff_div = divisor

    local = np.zeros((h, w), np.int64)
    if local_of_global:
        lut = np.zeros(int(globals_present.max()) + 1, np.int64)
        for g, l in local_of_global.items():
            lut[g] = l
        nz = gi > 0
        local[nz] = lut[gi[nz]]

    panoptic = sem * eff_div + local
    if panoptic.max() >= _U16_MAX:
        raise ValueError(f"panoptic max {panoptic.max()} overflows uint16; "
                         f"divisor {eff_div} too large for {len(globals_present)} instances")

    label = pb.CameraSegmentationLabel()
    label.panoptic_label_divisor = int(eff_div)
    label.panoptic_label = encode_panoptic_png(panoptic.astype(np.uint16))
    label.sequence_id = sequence_id
    for g, l in local_of_global.items():
        m = label.instance_id_to_global_id_mapping.add()
        m.local_instance_id = int(l)
        m.global_instance_id = int(g)
        m.is_tracked = True
    return label


def make_submission(account_name: str, unique_method_name: str,
                    authors: Sequence[str], affiliation: str, description: str,
                    method_link: str, frame_dt: int = 1,
                    runtime_ms: float = 0.0) -> pb.CameraSegmentationSubmission:
    """An empty submission proto with metadata filled in."""
    sub = pb.CameraSegmentationSubmission()
    sub.account_name = account_name
    sub.unique_method_name = unique_method_name
    sub.authors.extend(list(authors))
    sub.affiliation = affiliation
    sub.description = description
    sub.method_link = method_link
    sub.frame_dt = int(frame_dt)
    sub.runtime_ms = float(runtime_ms)
    return sub


def add_frame(sub: pb.CameraSegmentationSubmission, *,
              context_name: str, frame_timestamp_micros: int, camera_name: int,
              label: pb.CameraSegmentationLabel) -> None:
    """Append one (frame, camera) prediction to a submission proto."""
    fr = sub.predicted_segmentation_labels.frames.add()
    fr.context_name = context_name
    fr.frame_timestamp_micros = int(frame_timestamp_micros)
    fr.camera_name = int(camera_name)
    fr.camera_segmentation_label.CopyFrom(label)


def decode_camera_label(label: pb.CameraSegmentationLabel):
    """Inverse of pack_camera_label: -> (semantic (H,W) int32,
    global_instance (H,W) int64). Stuff/no-instance pixels get global 0.
    Reads the PNG with the repo's PIL-based decoder (a different codec than the
    cv2 encoder — interop is part of the correctness check)."""
    from panoctvps.data.decode import decode_png
    panoptic = decode_png(label.panoptic_label, np.uint16).astype(np.int64)
    div = int(label.panoptic_label_divisor)
    sem, local = np.divmod(panoptic, div)
    glob = np.zeros_like(local)
    if len(label.instance_id_to_global_id_mapping):
        max_local = max(m.local_instance_id
                        for m in label.instance_id_to_global_id_mapping)
        lut = np.zeros(max_local + 1, np.int64)
        for m in label.instance_id_to_global_id_mapping:
            lut[m.local_instance_id] = m.global_instance_id
        nz = local > 0
        glob[nz] = lut[np.clip(local[nz], 0, max_local)]
    return sem.astype(np.int32), glob
