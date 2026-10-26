"""Round-trip tests for the TF-free submission packer."""
from __future__ import annotations

import numpy as np
import pytest

from panoctvps.submission import packer
from panoctvps.submission.proto import wod_camseg_submission_pb2 as pb


def _synthetic_frame(h=40, w=60, seed=0):
    rng = np.random.default_rng(seed)
    sem = np.full((h, w), 21, np.int32)            # stuff class everywhere
    gi = np.zeros((h, w), np.int64)
    # three thing instances with arbitrary (non-sequential) global ids
    sem[2:10, 2:12] = 2;   gi[2:10, 2:12] = 5007    # car
    sem[15:30, 20:35] = 9; gi[15:30, 20:35] = 88    # pedestrian
    sem[5:12, 40:55] = 3;  gi[5:12, 40:55] = 311    # truck
    return sem, gi


class TestRoundTrip:
    def test_semantics_and_instances_preserved(self):
        sem, gi = _synthetic_frame()
        label = packer.pack_camera_label(sem, gi, sequence_id="ctx")
        dsem, dgi = packer.decode_camera_label(label)
        assert np.array_equal(dsem, sem)                       # semantics exact
        # instance partition + global ids preserved on thing pixels
        assert np.array_equal(dgi, gi)

    def test_mapping_uses_global_ids_and_is_tracked(self):
        sem, gi = _synthetic_frame()
        label = packer.pack_camera_label(sem, gi, sequence_id="ctx")
        globals_in_map = {m.global_instance_id for m in
                          label.instance_id_to_global_id_mapping}
        assert globals_in_map == {5007, 88, 311}
        assert all(m.is_tracked for m in label.instance_id_to_global_id_mapping)
        # local ids are compact 1..N
        locals_in_map = sorted(m.local_instance_id for m in
                               label.instance_id_to_global_id_mapping)
        assert locals_in_map == [1, 2, 3]

    def test_stuff_only_frame(self):
        sem = np.full((20, 20), 21, np.int32)
        gi = np.zeros((20, 20), np.int64)
        label = packer.pack_camera_label(sem, gi, sequence_id="c")
        assert len(label.instance_id_to_global_id_mapping) == 0
        dsem, dgi = packer.decode_camera_label(label)
        assert np.array_equal(dsem, sem) and dgi.max() == 0

    def test_non_thing_global_ids_dropped(self):
        # a stuff class carrying a stray instance id must not become an instance
        sem = np.full((10, 10), 21, np.int32)
        gi = np.zeros((10, 10), np.int64); gi[1:3, 1:3] = 42
        label = packer.pack_camera_label(sem, gi, sequence_id="c")
        assert len(label.instance_id_to_global_id_mapping) == 0

    def test_uint16_overflow_raises(self):
        # semantic 28 with divisor forced high -> overflow
        sem = np.full((4, 4), 28, np.int32)
        gi = np.zeros((4, 4), np.int64)
        with pytest.raises(ValueError, match="overflow"):
            packer.pack_camera_label(sem, gi, sequence_id="c", divisor=5000)


class TestPngCodecInterop:
    def test_cv2_encode_pil_decode_identical(self):
        # encoder is cv2, the repo/Waymo decode path is PIL/tf — they must agree
        from panoctvps.data.decode import decode_png
        rng = np.random.default_rng(1)
        pano = (rng.integers(0, 28000, (50, 70))).astype(np.uint16)
        png = packer.encode_panoptic_png(pano)
        back = decode_png(png, np.uint16)
        assert np.array_equal(back, pano) and back.dtype == np.uint16


class TestSubmissionProto:
    def test_full_submission_serialises_and_parses(self):
        sem, gi = _synthetic_frame()
        sub = packer.make_submission(
            account_name="me@x.com", unique_method_name="Pano-CTVPS",
            authors=["A"], affiliation="Akdeniz", description="d",
            method_link="http://x", frame_dt=1, runtime_ms=1000.0)
        label = packer.pack_camera_label(sem, gi, sequence_id="ctxA")
        packer.add_frame(sub, context_name="ctxA",
                         frame_timestamp_micros=123456, camera_name=1, label=label)
        blob = sub.SerializeToString()
        sub2 = pb.CameraSegmentationSubmission()
        sub2.ParseFromString(blob)
        assert sub2.frame_dt == 1
        assert sub2.predicted_segmentation_labels.frames[0].camera_name == 1
        assert sub2.predicted_segmentation_labels.frames[0].context_name == "ctxA"
        dsem, dgi = packer.decode_camera_label(
            sub2.predicted_segmentation_labels.frames[0].camera_segmentation_label)
        assert np.array_equal(dsem, sem) and np.array_equal(dgi, gi)
