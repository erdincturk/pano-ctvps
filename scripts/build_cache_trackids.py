"""
Write the track ids of a tracker run into the per-camera prediction cache as a
new record field, leaving the `tracker_id` field written at cache-build time
(and every other array) untouched.

The track ids of our tracker at the settings of the paper (match 0.7,
initiation 0.01, 5 frames kept, motion gate 600 / 800 / 800 / 1000 / 1000 px)
are stored as `tracker_id_deployed`; the linker reads this field.

The trajectory NPZ is produced by:
    scripts/eval_tracker_cached.py ... --save-trajectories <path>
and is aligned 1:1 to each segment's `records` array (0 = no track).

Safety:
  * default is DRY-RUN — pass --write to modify the cache. The dry run reports,
    per segment, how many records differ from the field if it already exists.
  * each .npz is rewritten to a temp file, every non-records array is asserted
    unchanged, the overlay length is asserted == len(records), then the temp
    is atomically os.replace()d over the original.
  * idempotent — re-running overwrites the field's values, never duplicates it.

Usage:
    python scripts/build_cache_trackids.py \
        --cache-dir outputs/tracker_ctvis_val_cache \
        --trajectories outputs/tracker_ctvis/deployed_trajectories.npz \
        --field-name tracker_id_deployed \
        [--write]
"""

import argparse
import os
from pathlib import Path

import numpy as np


def add_field(records: np.ndarray, name: str, values: np.ndarray) -> np.ndarray:
    """Return a copy of `records` with int32 field `name` set to `values`.

    If `name` already exists it is overwritten in place (idempotent); otherwise
    a new structured array with the appended field is built and all existing
    fields are copied verbatim.
    """
    if name in records.dtype.names:
        out = records.copy()
        out[name] = values.astype(np.int32)
        return out
    new_dtype = np.dtype(records.dtype.descr + [(name, "<i4")])
    out = np.empty(records.shape, dtype=new_dtype)
    for f in records.dtype.names:
        out[f] = records[f]
    out[name] = values.astype(np.int32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Directory of per-segment *.npz prediction caches.")
    ap.add_argument("--trajectories", type=Path, required=True,
                    help="Trajectory NPZ from eval_tracker_cached.py --save-trajectories.")
    ap.add_argument("--field-name", default="tracker_id_deployed",
                    help="New record field to write (default: tracker_id_deployed).")
    ap.add_argument("--write", action="store_true",
                    help="Actually modify the cache. Without it, DRY-RUN only.")
    args = ap.parse_args()

    traj = np.load(args.trajectories, allow_pickle=True)
    seg_files = [str(s) for s in traj["segment_files"]]
    tid_arrays = list(traj["tracker_ids"])
    traj_map = {sf: np.asarray(t) for sf, t in zip(seg_files, tid_arrays)}
    print(f"Trajectory overlay: {len(traj_map)} segments from {args.trajectories}")
    for key in ("tracker_config", "tier1_config"):   # key name differs in older files
        if key in traj.files:
            print(f"  tracker config: {str(traj[key])[:120]}...")
            break

    cache_files = sorted(args.cache_dir.glob("*.npz"))
    print(f"Cache: {len(cache_files)} segments in {args.cache_dir}")
    print(f"Mode: {'WRITE' if args.write else 'DRY-RUN (no changes)'}\n")

    n_ok = 0
    n_total_diff_existing = 0
    for cpath in cache_files:
        base = cpath.name
        if base not in traj_map:
            print(f"  [SKIP] {base}: no trajectory entry")
            continue
        npz = np.load(cpath, allow_pickle=True)
        records = npz["records"]
        overlay = traj_map[base]

        if len(overlay) != len(records):
            print(f"  [FAIL] {base}: overlay len {len(overlay)} != records len {len(records)}")
            continue

        # Coverage stats vs base for a sanity signal (not a gate).
        base_tid = records["tracker_id"]
        n_changed = int(np.count_nonzero(base_tid != overlay))
        n_assigned = int(np.count_nonzero(overlay > 0))
        already = args.field_name in records.dtype.names
        n_diff_existing = (
            int(np.count_nonzero(records[args.field_name] != overlay)) if already else 0
        )
        if already:
            n_total_diff_existing += n_diff_existing
        print(f"  [{'OK ' if not already else 'UPD'}] {base}: "
              f"{len(records):>6d} records | assigned={n_assigned:>6d} | "
              f"differ-from-base={n_changed:>6d}"
              + (f" | field '{args.field_name}' exists, differs in "
                 f"{n_diff_existing} records → overwrite" if already else ""))

        if not args.write:
            n_ok += 1
            continue

        new_records = add_field(records, args.field_name, overlay)
        # Re-assemble all arrays; only `records` changes.
        data = {k: npz[k] for k in npz.files}
        data["records"] = new_records

        # Temp must end in `.npz` — np.savez_compressed appends `.npz`
        # otherwise, which would desync the verify-load path below.
        tmp = cpath.parent / (cpath.name + ".tmp.npz")
        np.savez_compressed(tmp, **data)

        # Verify temp before replacing.
        chk = np.load(tmp, allow_pickle=True)
        assert set(chk.files) == set(npz.files), "key set changed"
        assert len(chk["records"]) == len(records), "record count changed"
        assert args.field_name in chk["records"].dtype.names, "field missing"
        assert np.array_equal(chk["records"][args.field_name], overlay.astype(np.int32)), "field mismatch"
        for f in records.dtype.names:
            assert np.array_equal(chk["records"][f], records[f]), f"base field '{f}' changed"
        # spot-check a couple of large non-records arrays are intact
        for arr_key in ("embeddings", "mask_rle", "frame_queries"):
            if arr_key in npz.files:
                a, b = npz[arr_key], chk[arr_key]
                if a.dtype == object:
                    assert a.shape == b.shape, f"{arr_key} shape changed"
                else:
                    assert np.array_equal(a, b), f"{arr_key} changed"
        os.replace(tmp, cpath)
        n_ok += 1

    print(f"\n{'Wrote' if args.write else 'Validated (dry-run)'} "
          f"{n_ok}/{len(cache_files)} segments.")
    print(f"Records that differ from the existing '{args.field_name}' field: "
          f"{n_total_diff_existing}")
    if not args.write:
        print("Re-run with --write to apply.")


if __name__ == "__main__":
    main()
