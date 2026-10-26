"""package_submission.py — check coverage and build the tar.gz for upload.

Operates on the binprotos written by scripts/build_submission.py
(default submissions/<split>/binprotos/):
  1. optionally (--metadata) replace the submission metadata of every
     binproto by the values of a JSON file (keys: account_name,
     unique_method_name, authors, affiliation, description, method_link,
     frame_dt, runtime_ms), without touching the predictions;
  2. check that exactly the scenes and frames of the split file are present,
     each with five cameras;
  3. write submissions/<split>/submit_<split>.tar.gz (the upload file).

Usage:
  python scripts/package_submission.py --split test
"""
from __future__ import annotations
import argparse, collections, json, subprocess
from pathlib import Path

from panoctvps.submission.proto import wod_camseg_submission_pb2 as pb

REPO = Path(__file__).resolve().parents[1]

SPLIT_FILES = {"validation": REPO / "configs/splits/validation_frames.txt",
               "test": REPO / "configs/splits/test_frames.txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--sub-dir", type=Path, default=None,
                    help="binproto directory (default: submissions/<split>/binprotos)")
    ap.add_argument("--metadata", type=Path, default=None,
                    help="optional JSON file with the metadata to write into every binproto")
    args = ap.parse_args()
    bp_dir = args.sub_dir or (REPO / "submissions" / args.split / "binprotos")
    base = bp_dir.parent
    binprotos = sorted(bp_dir.glob("*.binproto"))

    # requested scene -> timestamp count
    req = collections.defaultdict(set)
    for line in SPLIT_FILES[args.split].read_text().splitlines():
        if line.strip():
            ctx, ts = line.strip().split(","); req[ctx].add(int(ts))

    m = json.loads(args.metadata.read_text()) if args.metadata else None
    print(f"checking {len(binprotos)} binprotos in {bp_dir}")
    seen_ctx, tot_frames, problems = set(), 0, []
    for p in binprotos:
        sub = pb.CameraSegmentationSubmission()
        sub.ParseFromString(p.read_bytes())
        if m is not None:
            sub.account_name = m["account_name"]; sub.unique_method_name = m["unique_method_name"]
            del sub.authors[:]; sub.authors.extend(m["authors"])
            sub.affiliation = m["affiliation"]; sub.description = m["description"]
            sub.method_link = m["method_link"]; sub.frame_dt = int(m.get("frame_dt", 1))
            sub.runtime_ms = float(m["runtime_ms"])
            p.write_bytes(sub.SerializeToString())
        # coverage
        frames = sub.predicted_segmentation_labels.frames
        ctxs = {f.context_name for f in frames}
        if len(ctxs) != 1:
            problems.append(f"{p.name}: {len(ctxs)} contexts"); continue
        ctx = next(iter(ctxs)); seen_ctx.add(ctx)
        ts_present = {f.frame_timestamp_micros for f in frames}
        cams_per_ts = collections.Counter(f.frame_timestamp_micros for f in frames)
        if ctx not in req:
            problems.append(f"{ctx}: not in split file")
        elif ts_present != req[ctx]:
            problems.append(f"{ctx}: ts {len(ts_present)} vs requested {len(req[ctx])}")
        bad = [t for t, c in cams_per_ts.items() if c != 5]
        if bad:
            problems.append(f"{ctx}: {len(bad)} ts without 5 cameras")
        tot_frames += len(frames)

    missing = set(req) - seen_ctx
    exp = sum(len(v) for v in req.values()) * 5
    print(f"scenes: {len(seen_ctx)}/{len(req)} | frame-cameras: {tot_frames}/{exp}")
    if missing: print(f"MISSING scenes: {sorted(missing)}")
    if problems:
        print("PROBLEMS:"); [print("  -", x) for x in problems[:10]]
        raise SystemExit("coverage check failed")
    if missing or tot_frames != exp:
        raise SystemExit("coverage check failed")
    print("coverage check passed")

    tar = base / f"submit_{args.split}.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(bp_dir), "."], check=True)
    print(f"\nArchive: {tar}  ({tar.stat().st_size/1e6:.1f} MB)")
    print(f"Upload this to the {args.split} leaderboard at waymo.com/open.")


if __name__ == "__main__":
    main()
