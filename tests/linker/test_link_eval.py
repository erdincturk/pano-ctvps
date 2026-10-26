"""Tests for panoctvps/linker/link_eval.py — table round trip + link counts.

The save/load round trip must preserve the candidate row order (ties in the
assigner's ordering keep the scored-dict order) and None cosines (NaN
encoding). eval_config is checked on a hand-built scene with known
correct / wrong / missed / contam outcomes.
"""
from __future__ import annotations

import numpy as np
import pytest

from panoctvps.linker.link_eval import (
    eval_config, load_scene_table, save_scene_table,
)


def _pair(u, v):
    return frozenset({u, v})


def _entry(iou, cls=2, cos=None, class_match=True):
    return {"iou_mean": iou, "n_frames": 3, "n_scored": 3, "class_match": class_match,
            "classA": cls, "classB": cls, "cos_med": cos}


def _scene():
    """cam1 t1 and cam2 t1 are GT object 5 (co-visible); cam2 t2 is object 7.
    Candidates: the true pair (iou 0.6) and a false pair (iou 0.5)."""
    scored = {
        _pair((1, 1), (2, 1)): _entry(0.6, cos=0.8),
        _pair((1, 1), (2, 2)): _entry(0.5, cos=-0.2),
    }
    return {"scored": scored,
            "all_tr": {(1, 1), (2, 1), (2, 2)},
            "dom_gid": {(1, 1): 5, (2, 1): 5, (2, 2): 7},
            "by_gid": {5: {1: [(1, 1)], 2: [(2, 1)]}, 7: {2: [(2, 2)]}},
            "true_camlinks": {(5, frozenset((1, 2)))},
            "unreachable": 0, "multicam": [5]}


class TestEvalConfig:
    def test_true_link_recovered(self):
        r = eval_config([_scene()], tau=0.4)
        assert (r["correct"], r["wrong"], r["missed"]) == (1, 0, 0)
        assert (r["precision"], r["recall"], r["contam"]) == (1.0, 1.0, 0)
        assert r["per_pair"]["F-FL"] == {"true": 1, "correct": 1, "wrong": 0, "missed": 0}

    def test_high_tau_loses_link(self):
        r = eval_config([_scene()], tau=0.65)
        assert (r["correct"], r["wrong"], r["missed"]) == (0, 0, 1)
        assert r["contam"] == 0

    def test_wrong_merge_counts_fp_and_contam(self):
        # Force the false edge to outrank the true one: geometry 0.7 vs 0.6.
        s = _scene()
        s["scored"][_pair((1, 1), (2, 2))] = _entry(0.7, cos=-0.2)
        r = eval_config([s], tau=0.4)
        # (1,1)-(2,2) merges first; true pair rejected by one-per-camera.
        assert (r["correct"], r["wrong"], r["missed"]) == (0, 1, 1)
        # object 7 is single-camera (not counted); object 5's global identity
        # now holds a track of another object -> contam.
        assert r["contam"] == 1

    def test_rank_key_rescues_contested_slot(self):
        s = _scene()
        s["scored"][_pair((1, 1), (2, 2))] = _entry(0.7, cos=-0.2)
        r = eval_config([s], tau=0.4, rank_key="cos_med")
        # cosine demotes the false link (factor 0.25 -> rank 0.175 < 0.6).
        assert (r["correct"], r["wrong"], r["missed"]) == (1, 0, 0)
        assert r["contam"] == 0

    def test_structured_threshold(self):
        r = eval_config([_scene()], tau={"F-FL|vehicle": 0.55, "vehicle": 0.4, "human": 0.4})
        # F-FL bar 0.55: true link (0.6) passes, false (0.5) rejected.
        assert (r["correct"], r["wrong"], r["missed"]) == (1, 0, 0)

    def test_veto(self):
        s = _scene()
        s["scored"][_pair((1, 1), (2, 2))] = _entry(0.7, cos=-0.2)
        s["scored"][_pair((1, 1), (2, 1))]["cos_med_adapted"] = 0.9
        s["scored"][_pair((1, 1), (2, 2))]["cos_med_adapted"] = 0.1
        r = eval_config([s], tau=0.4)
        assert (r["correct"], r["wrong"]) == (0, 1)
        # the false link (adapted cosine 0.1 < 0.5, S 0.7 >= 0.6) is rescued by geometry
        r = eval_config([s], tau=0.4, veto_key="cos_med_adapted", veto_min=0.5, veto_rescue=0.6)
        assert (r["correct"], r["wrong"]) == (0, 1)
        # with a rescue bar above 0.7 it is vetoed, and the true link is merged
        r = eval_config([s], tau=0.4, veto_key="cos_med_adapted", veto_min=0.5, veto_rescue=0.8)
        assert (r["correct"], r["wrong"], r["missed"], r["contam"]) == (1, 0, 0, 0)


class TestTableRoundTrip:
    def test_round_trip_preserves_order_values_and_none(self, tmp_path):
        s = _scene()
        # Add a None-cosine candidate to exercise the NaN encoding.
        s["scored"][_pair((2, 2), (4, 9))] = _entry(0.2, cos=None)
        s["all_tr"].add((4, 9))
        s["dom_gid"][(4, 9)] = 0
        path = tmp_path / "scene.npz"
        save_scene_table(path, s["scored"],
                         {"all_tr": s["all_tr"], "dom_gid": s["dom_gid"],
                          "true_camlinks": s["true_camlinks"],
                          "multicam": s["multicam"], "unreachable": s["unreachable"]},
                         segment="test_scene", tid_field="tracker_id_deployed")
        loaded = load_scene_table(path)

        assert list(loaded["scored"].keys()) == list(s["scored"].keys())  # order
        for pair, orig in s["scored"].items():
            got = loaded["scored"][pair]
            assert got["iou_mean"] == pytest.approx(orig["iou_mean"], abs=1e-7)
            assert got["class_match"] == orig["class_match"]
            if orig["cos_med"] is None:
                assert got["cos_med"] is None
            else:
                assert got["cos_med"] == pytest.approx(orig["cos_med"], abs=1e-6)
        assert loaded["all_tr"] == s["all_tr"]
        assert loaded["dom_gid"] == s["dom_gid"]
        assert loaded["true_camlinks"] == s["true_camlinks"]
        assert loaded["multicam"] == s["multicam"]
        assert loaded["unreachable"] == 0

    def test_eval_identical_through_round_trip(self, tmp_path):
        s = _scene()
        path = tmp_path / "scene.npz"
        save_scene_table(path, s["scored"],
                         {"all_tr": s["all_tr"], "dom_gid": s["dom_gid"],
                          "true_camlinks": s["true_camlinks"],
                          "multicam": s["multicam"], "unreachable": s["unreachable"]},
                         segment="test_scene", tid_field="tracker_id_deployed")
        loaded = load_scene_table(path)
        for tau in (0.1, 0.4, 0.65):
            a = eval_config([s], tau=tau)
            b = eval_config([loaded], tau=tau)
            assert a == b, tau

    def test_load_ignores_extra_columns(self, tmp_path):
        # tables with additional candidate columns load to the same scene dict
        import numpy as np
        from panoctvps.linker.link_eval import CAND_DTYPE
        s = _scene()
        path = tmp_path / "scene.npz"
        save_scene_table(path, s["scored"],
                         {"all_tr": s["all_tr"], "dom_gid": s["dom_gid"],
                          "true_camlinks": s["true_camlinks"],
                          "multicam": s["multicam"], "unreachable": s["unreachable"]},
                         segment="test_scene", tid_field="tracker_id_deployed")
        z = dict(np.load(path))
        wide = np.dtype(CAND_DTYPE.descr + [("extra", "f8")])
        rows = np.zeros(len(z["candidates"]), dtype=wide)
        for f in CAND_DTYPE.names:
            rows[f] = z["candidates"][f]
        z["candidates"] = rows
        path2 = tmp_path / "wide.npz"
        np.savez_compressed(path2, **z)
        assert load_scene_table(path2)["scored"] == load_scene_table(path)["scored"]
