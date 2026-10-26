"""
WOD PVPS Constants
==================
All constants derived from official Waymo Open Dataset source:
  - camera_segmentation.proto  (semantic class enum)
  - camera_segmentation_utils.py  (color map)
  - Sun et al. 2020  (camera specs)
  - Mei et al. 2022  (dataset structure, thing/stuff split)
"""

import numpy as np

# =============================================================================
# Panoptic label encoding
# =============================================================================

NUM_CLASSES = 29           # Class IDs 0-28: 28 classes plus 0 = undefined
PANOPTIC_LABEL_DIVISOR = 1000
IGNORE_LABEL = 0           # TYPE_UNDEFINED is ignored in evaluation

# =============================================================================
# Semantic class definitions (from camera_segmentation.proto)
# =============================================================================

CLASS_NAMES = {
    0:  "undefined",
    1:  "ego_vehicle",
    2:  "car",
    3:  "truck",
    4:  "bus",
    5:  "other_large_vehicle",
    6:  "bicycle",
    7:  "motorcycle",
    8:  "trailer",
    9:  "pedestrian",
    10: "cyclist",
    11: "motorcyclist",
    12: "bird",
    13: "ground_animal",
    14: "construction_cone_pole",
    15: "pole",
    16: "pedestrian_object",
    17: "sign",
    18: "traffic_light",
    19: "building",
    20: "road",
    21: "lane_marker",
    22: "road_marker",
    23: "sidewalk",
    24: "vegetation",
    25: "sky",
    26: "ground",
    27: "dynamic",
    28: "static",
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# =============================================================================
# Thing / Stuff split
# =============================================================================
# 28 classes (IDs 1-28): 8 thing classes and 20 stuff classes. ID 0 is the
# undefined class, ignored in evaluation.
# 8 thing classes — ONLY these have instance IDs and participate in tracking.
# bicycle(6) and motorcycle(7) are STUFF (no rider = no instance).
# bird(12) and ground_animal(13) are STUFF (no instance IDs).

THING_CLASSES = [2, 3, 4, 5, 8, 9, 10, 11]
THING_NAMES = [CLASS_NAMES[c] for c in THING_CLASSES]

# Non-thing IDs: the 20 stuff classes plus 0 (undefined). Semantic labels
# only, no instance IDs.
STUFF_CLASSES = [c for c in range(NUM_CLASSES) if c not in THING_CLASSES]
STUFF_NAMES = [CLASS_NAMES[c] for c in STUFF_CLASSES]

# Boolean list used by STQuality metric
CLASS_HAS_INSTANCES_LIST = [c in THING_CLASSES for c in range(NUM_CLASSES)]

# All 8 thing classes are also the wSTQ tracking classes
TRACKED_CLASSES = [2, 3, 4, 5, 8, 9, 10, 11]

# Super-class groupings (Mei et al. 2022 Fig 3)
# NOTE: ego_vehicle(1) is "vehicle", ground(26) is "void", ID 0 has no super-class
SUPER_CLASSES = {
    "vehicle": [1, 2, 3, 4, 5, 6, 7, 8],     # ego_vehicle + all vehicle types
    "human": [9, 10, 11],                      # pedestrian, cyclist, motorcyclist
    "animal": [12, 13],                        # bird, ground_animal
    "object": [14, 15, 16, 17, 18],           # cone_pole, pole, ped_object, sign, traffic_light
    "building": [19],                          # building
    "flat": [20, 21, 22, 23],                 # road, lane_marker, road_marker, sidewalk
    "vegetation": [24],                        # vegetation
    "sky": [25],                               # sky
    "void": [26, 27, 28],                     # ground, dynamic, static (ID 0 excluded)
}

# =============================================================================
# Camera definitions
# =============================================================================

CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}
CAMERA_IDS = {v: k for k, v in CAMERA_NAMES.items()}

# Camera image resolutions (H, W) in WOD v2.0.1
CAMERA_RESOLUTION = {
    1: (1280, 1920),   # FRONT
    2: (1280, 1920),   # FRONT_LEFT
    3: (1280, 1920),   # FRONT_RIGHT
    4: (886, 1920),    # SIDE_LEFT
    5: (886, 1920),    # SIDE_RIGHT
}

# Side cameras (SIDE_LEFT=4, SIDE_RIGHT=5) have a lower height (886 vs 1280).
SIDE_CAMERA_IDS: frozenset = frozenset({4, 5})

# =============================================================================
# RGB Color map for visualization (from camera_segmentation_utils.py)
# =============================================================================

SEGMENTATION_COLOR_MAP = {
    0: [0, 0, 0],           # undefined
    1: [102, 102, 102],     # ego_vehicle
    2: [0, 0, 142],         # car
    3: [0, 0, 70],          # truck
    4: [0, 60, 100],        # bus
    5: [61, 133, 198],      # other_large_vehicle
    6: [119, 11, 32],       # bicycle
    7: [0, 0, 230],         # motorcycle
    8: [111, 168, 220],     # trailer
    9: [220, 20, 60],       # pedestrian
    10: [255, 0, 0],        # cyclist
    11: [180, 0, 0],        # motorcyclist
    12: [127, 96, 0],       # bird
    13: [91, 15, 0],        # ground_animal
    14: [230, 145, 56],     # construction_cone_pole
    15: [153, 153, 153],    # pole
    16: [234, 153, 153],    # pedestrian_object
    17: [246, 178, 107],    # sign
    18: [250, 170, 30],     # traffic_light
    19: [70, 70, 70],       # building
    20: [128, 64, 128],     # road
    21: [234, 209, 220],    # lane_marker
    22: [217, 210, 233],    # road_marker
    23: [244, 35, 232],     # sidewalk
    24: [107, 142, 35],     # vegetation
    25: [70, 130, 180],     # sky
    26: [102, 102, 102],    # ground
    27: [102, 102, 102],    # dynamic
    28: [102, 102, 102],    # static
}


def get_color_map() -> np.ndarray:
    """Returns [NUM_CLASSES, 3] uint8 array: class_id -> RGB color."""
    cmap = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for cls_id, color in SEGMENTATION_COLOR_MAP.items():
        cmap[cls_id] = color
    return cmap


COLOR_MAP = get_color_map()

# =============================================================================
# Parquet column names (for explicit column selection during loading)
# =============================================================================

PARQUET_COLS = {
    # camera_segmentation
    "seg_key_context": "key.segment_context_name",
    "seg_key_timestamp": "key.frame_timestamp_micros",
    "seg_key_camera": "key.camera_name",
    "seg_panoptic_label": "[CameraSegmentationLabelComponent].panoptic_label",
    "seg_divisor": "[CameraSegmentationLabelComponent].panoptic_label_divisor",
    "seg_num_cameras": "[CameraSegmentationLabelComponent].num_cameras_covered",
    "seg_sequence_id": "[CameraSegmentationLabelComponent].sequence_id",
    "seg_local_ids": "[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.local_instance_ids",
    "seg_global_ids": "[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.global_instance_ids",
    "seg_is_tracked": "[CameraSegmentationLabelComponent].instance_id_to_global_id_mapping.is_tracked",

    # camera_image
    "img_key_timestamp": "key.frame_timestamp_micros",
    "img_key_camera": "key.camera_name",
    "img_image": "[CameraImageComponent].image",
    "img_pose": "[CameraImageComponent].pose.transform",

    # camera_calibration
    "cal_key_camera": "key.camera_name",
    # Intrinsic — WOD v2.0.1 splits the 9-element intrinsic vector
    # (per dataset.proto: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]) into 9
    # separate scalar float64 columns. Pinhole + OpenCV-convention distortion:
    # radial k1/k2/k3, tangential p1/p2; see `intrinsic_columns()` below.
    "cal_fu": "[CameraCalibrationComponent].intrinsic.f_u",
    "cal_fv": "[CameraCalibrationComponent].intrinsic.f_v",
    "cal_cu": "[CameraCalibrationComponent].intrinsic.c_u",
    "cal_cv": "[CameraCalibrationComponent].intrinsic.c_v",
    "cal_k1": "[CameraCalibrationComponent].intrinsic.k1",
    "cal_k2": "[CameraCalibrationComponent].intrinsic.k2",
    "cal_p1": "[CameraCalibrationComponent].intrinsic.p1",
    "cal_p2": "[CameraCalibrationComponent].intrinsic.p2",
    "cal_k3": "[CameraCalibrationComponent].intrinsic.k3",
    "cal_extrinsic": "[CameraCalibrationComponent].extrinsic.transform",
    "cal_width": "[CameraCalibrationComponent].width",
    "cal_height": "[CameraCalibrationComponent].height",
}

# =============================================================================
# Intrinsic vector helpers
# =============================================================================
# Projection between cameras uses the full 9-element intrinsic vector
# (pinhole + lens distortion), as cv2.undistortPoints requires.

INTRINSIC_VECTOR_KEYS = [
    "cal_fu", "cal_fv", "cal_cu", "cal_cv",
    "cal_k1", "cal_k2", "cal_p1", "cal_p2", "cal_k3",
]


def intrinsic_columns() -> list:
    """Return the canonical 9-element list of parquet column paths for
    camera_calibration intrinsics, in WOD dataset.proto order:
        [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
    Use this when loading the full intrinsic vector for back-projection
    (cv2.undistortPoints).
    """
    return [PARQUET_COLS[k] for k in INTRINSIC_VECTOR_KEYS]


def split_intrinsic_vector(intrinsic_9):
    """Split a 9-element WOD intrinsic vector into the (K, dist) pair that
    OpenCV functions like cv2.undistortPoints expect.

    Args:
        intrinsic_9: array-like of length 9, ordered as
            [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3] per WOD dataset.proto.

    Returns:
        K:    (3, 3) float64 — pinhole intrinsic matrix.
        dist: (5,)   float64 — OpenCV distortion vector [k1, k2, p1, p2, k3].
    """
    arr = np.asarray(intrinsic_9, dtype=np.float64)
    if arr.shape != (9,):
        raise ValueError(
            f"intrinsic_9 must have shape (9,), got {arr.shape}; "
            "expected ordering [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]"
        )
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = arr
    K = np.array([
        [fu, 0.0, cu],
        [0.0, fv, cv],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist
