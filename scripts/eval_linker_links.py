#!/usr/bin/env python3
"""eval_linker_links.py — link precision and recall of the linker.

For each named configuration of --configs (same format as
scripts/eval_linker_wstq.py), runs the assigner on the scored candidates of
every validation scene (scripts/build_scores.py, adapted cosines from
scripts/build_cosines.py) and counts the links against the true links between
tracks that share a frame (definitions in panoctvps/linker/link_eval.py):
correct, wrong, missed, precision, recall, the same per adjacent camera pair,
and contam, the number of multi-camera ground-truth objects whose global
identity also contains a track of another object.

Usage:
  python scripts/eval_linker_links.py --scores-dir outputs/linker/scored_candidates \
      --cosines-dir outputs/linker/adapted_cosines --configs configs/linker_table7.json \
      --output outputs/linker/table7_links.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from panoctvps.linker.link_eval import eval_config, load_scene_table, merge_adapted_cosines

REPO = Path(__file__).resolve().parents[1]


def print_counts(name: str, r: dict) -> None:
    print(f"\n{name}")
    print(f"  correct={r['correct']}  wrong={r['wrong']}  missed={r['missed']}  "
          f"(true links {r['true']})  precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
          f"contam={r['contam']}")
    print(f"  {'pair':>6} {'true':>5} {'correct':>8} {'wrong':>6} {'missed':>7}")
    for pn, p in r["per_pair"].items():
        print(f"  {pn:>6} {p['true']:>5} {p['correct']:>8} {p['wrong']:>6} {p['missed']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", type=Path, default=REPO / "outputs/linker/scored_candidates")
    ap.add_argument("--cosines-dir", type=Path, default=REPO / "outputs/linker/adapted_cosines")
    ap.add_argument("--configs", type=Path, default=REPO / "configs/linker_deployed.json")
    ap.add_argument("--output", type=Path, default=REPO / "outputs/linker/linker_links.json")
    args = ap.parse_args()

    configs = json.loads(args.configs.read_text())
    scenes = [load_scene_table(p) for p in sorted(args.scores_dir.glob("*.npz"))]
    if any(cfg.get("veto_key") for cfg in configs.values()):
        merge_adapted_cosines(scenes, args.cosines_dir)
    print(f"{len(scenes)} scenes, {sum(len(s['true_camlinks']) for s in scenes)} true links")

    out = {"n_scenes": len(scenes), "configs": configs, "links": {}}
    for name, cfg in configs.items():
        r = eval_config(scenes, cfg["tau"], rank_key=cfg.get("rank_key"),
                        veto_key=cfg.get("veto_key"), veto_min=cfg.get("veto_min", 0.0),
                        veto_rescue=cfg.get("veto_rescue", 1.0))
        out["links"][name] = r
        print_counts(name, r)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
