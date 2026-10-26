"""
verify_dataset.py
-----------------
Check that the local Waymo Open Dataset v2.0.1 copy holds every scene the
pipeline reads.

Two checks:

1. **Scene coverage** — every scene listed in
   `configs/splits/{training,validation,test}_frames.txt` is present in each
   component the pipeline reads (camera_image, camera_segmentation,
   camera_calibration). Test scenes may be stored in either `testing/` or
   `testing_location/`; both are searched.
2. **Readability** (`--deep`) — opens the parquet files of a sample of those
   scenes and reads their metadata, catching truncated or corrupt files.

The exit code is 0 only when every requested check passes.

Usage:
  python scripts/verify_dataset.py --data-root <root>          # check 1
  python scripts/verify_dataset.py --data-root <root> --deep   # + readability
  python scripts/verify_dataset.py --deep --sample 10 --split validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data" / "waymo_open_dataset_v_2_0_1"
SPLITS_DIR = REPO_ROOT / "configs" / "splits"

# Components the pipeline reads.
PIPELINE_COMPONENTS: List[str] = [
    "camera_image",
    "camera_segmentation",
    "camera_calibration",
]

# Split file -> release folders the scenes may be stored in.
SPLIT_FOLDERS: Dict[str, List[str]] = {
    "training": ["training"],
    "validation": ["validation"],
    "test": ["testing", "testing_location"],
}

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


def scenes_on_disk(data_root: Path, folder: str, component: str) -> Set[str]:
    """Return the scene names present for one release folder/component.

    Args:
        data_root: path to `waymo_open_dataset_v_2_0_1/`.
        folder: release folder name (e.g. "validation").
        component: component directory name (e.g. "camera_image").

    Returns:
        Scene names with the `.parquet` suffix stripped. Empty if the directory
        does not exist.
    """
    d = data_root / folder / component
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.parquet")}


def required_scenes(split: str) -> Set[str]:
    """Read the scene list of one split from its frame index file."""
    f = SPLITS_DIR / f"{split}_frames.txt"
    if not f.is_file():
        raise FileNotFoundError(f"missing split file: {f}")
    with f.open() as fh:
        return {line.split(",")[0].strip() for line in fh if line.strip()}


def check_coverage(data_root: Path, splits: List[str]) -> List[str]:
    """Verify every required scene exists in every pipeline component."""
    problems: List[str] = []
    print("\n=== 1. Scene coverage ===")
    for split in splits:
        folders = SPLIT_FOLDERS[split]
        need = required_scenes(split)
        print(f"\n{split}  ({len(need)} scenes, folders: {', '.join(folders)})")
        for comp in PIPELINE_COMPONENTS:
            have: Set[str] = set()
            for folder in folders:
                have |= scenes_on_disk(data_root, folder, comp)
            missing = need - have
            status = OK if not missing else BAD
            if missing:
                problems.append(
                    f"{split}/{comp}: {len(missing)} of {len(need)} scenes missing")
            print(f"  [{status}] {comp:<22} {len(need) - len(missing):>4}/{len(need)}")
            for s in sorted(missing)[:3]:
                print(f"         missing e.g. {s}")
            if len(missing) > 3:
                print(f"         ... and {len(missing) - 3} more")
    return problems


def check_readable(data_root: Path, splits: List[str], sample: int) -> List[str]:
    """Open the parquets of a sample of required scenes and read their metadata."""
    import pyarrow.parquet as pq

    problems: List[str] = []
    print(f"\n=== 2. Readability (sampling {sample} scene(s) per component) ===")
    for split in splits:
        folders = SPLIT_FOLDERS[split]
        scenes = sorted(required_scenes(split))[:sample]
        print(f"\n{split}")
        for comp in PIPELINE_COMPONENTS:
            files = []
            for s in scenes:
                for folder in folders:
                    f = data_root / folder / comp / f"{s}.parquet"
                    if f.is_file():
                        files.append(f)
                        break
            if not files:
                print(f"  [{WARN}] {comp:<22} nothing to sample")
                continue
            bad: List[str] = []
            rows = 0
            for f in files:
                try:
                    rows += pq.ParquetFile(f).metadata.num_rows
                except Exception as exc:  # noqa: BLE001 - report, don't crash
                    bad.append(f"{f.name}: {type(exc).__name__}")
            if bad:
                problems.append(f"{split}/{comp}: {len(bad)} unreadable")
                print(f"  [{BAD}] {comp:<22} {len(bad)}/{len(files)} unreadable")
                for b in bad[:2]:
                    print(f"         {b}")
            else:
                print(f"  [{OK}] {comp:<22} {len(files)} file(s), {rows} rows")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--split",
        action="append",
        choices=list(SPLIT_FOLDERS),
        help="limit the checks to this split (repeatable; default: all)",
    )
    ap.add_argument("--deep", action="store_true", help="also open parquets")
    ap.add_argument("--sample", type=int, default=3, help="scenes per component for --deep")
    args = ap.parse_args()

    data_root: Path = args.data_root
    print(f"data root: {data_root}")
    if not data_root.is_dir():
        print(f"\n[{BAD}] data root does not exist.")
        return 1

    splits = args.split or list(SPLIT_FOLDERS)

    problems = check_coverage(data_root, splits)
    if args.deep:
        problems += check_readable(data_root, splits, args.sample)

    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
