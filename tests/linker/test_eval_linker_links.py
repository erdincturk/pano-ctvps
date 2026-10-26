"""Test for scripts/eval_linker_links.py on a hand-built scene table."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from panoctvps.linker.link_eval import save_scene_table

REPO = Path(__file__).resolve().parents[2]


def _entry(iou, cos):
    return {"iou_mean": iou, "n_frames": 3, "n_scored": 3, "class_match": True,
            "classA": 2, "classB": 2, "cos_med": cos}


def test_counts_per_config(tmp_path):
    # cam1 t1 and cam2 t1 are object 5; cam2 t2 is object 7. The false pair has
    # the higher geometric score; its adapted cosine is low.
    a, b, c = (1, 1), (2, 1), (2, 2)
    scored = {frozenset({a, b}): _entry(0.5, 0.8), frozenset({a, c}): _entry(0.55, -0.2)}
    struct = {"all_tr": {a, b, c}, "dom_gid": {a: 5, b: 5, c: 7},
              "true_camlinks": {(5, frozenset((1, 2)))}, "multicam": [5], "unreachable": 0}
    scores, cos = tmp_path / "scores", tmp_path / "cos"
    scores.mkdir(); cos.mkdir()
    save_scene_table(scores / "s.npz", scored, struct, segment="s", tid_field="tracker_id_deployed")
    rows = np.array([(1, 1, 2, 1, 0.9), (1, 1, 2, 2, 0.1)],
                    dtype=[("camA", "i1"), ("tidA", "i4"), ("camB", "i1"), ("tidB", "i4"),
                           ("cos_med_adapted", "f8")])
    np.savez_compressed(cos / "s.npz", cosines=rows)
    cfg = {"geometry": {"tau": 0.4, "rank_key": None},
           "veto": {"tau": 0.4, "rank_key": None, "veto_key": "cos_med_adapted",
                    "veto_min": 0.5, "veto_rescue": 0.6}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "out.json"
    subprocess.run([sys.executable, str(REPO / "scripts/eval_linker_links.py"),
                    "--scores-dir", str(scores), "--cosines-dir", str(cos),
                    "--configs", str(tmp_path / "cfg.json"), "--output", str(out)],
                   check=True, capture_output=True)
    res = json.loads(out.read_text())["links"]
    g, v = res["geometry"], res["veto"]
    assert (g["correct"], g["wrong"], g["missed"], g["contam"]) == (0, 1, 1, 1)
    assert (v["correct"], v["wrong"], v["missed"], v["contam"]) == (1, 0, 0, 0)
    assert v["precision"] == 1.0 and v["recall"] == 1.0
    assert v["per_pair"]["F-FL"] == {"true": 1, "correct": 1, "wrong": 0, "missed": 0}
