"""Tests for panoctvps/linker/assigner.py — link rule, eta resolution, ordering, veto.

Covers: geometry only with a scalar eta, the exact class check, the
one-track-per-camera rule, eta per super-class and camera pair, the ordering
by S * rho (changes which competing link is merged, never which links are
accepted) and the appearance veto.
"""
from __future__ import annotations

import pytest

from panoctvps.linker.assigner import (
    RANK_FLOOR, _rank_factor, assign_global_ids, link_rule, resolve_threshold,
)


def _pair(u, v):
    return frozenset({u, v})


def _scored(entries):
    """entries: {(u, v): (iou_mean, class_match, classA, cos_med)}"""
    out = {}
    for (u, v), (iou, cm, cls, cos) in entries.items():
        out[_pair(u, v)] = {"iou_mean": iou, "class_match": cm,
                            "classA": cls, "classB": cls, "cos_med": cos}
    return out


class TestDefaults:
    def test_singletons_get_distinct_ids(self):
        tr = {(1, 1), (2, 1), (3, 7)}
        out = assign_global_ids(tr, {})
        assert set(out) == tr
        assert len(set(out.values())) == 3

    def test_accepted_edge_merges(self):
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.6, True, 2, None)})
        out = assign_global_ids(tr, scored, threshold=0.4)
        assert out[(1, 1)] == out[(2, 1)]

    def test_threshold_and_class_check(self):
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.6, False, 2, None)})
        assert len(set(assign_global_ids(tr, scored, threshold=0.4).values())) == 2
        scored2 = _scored({((1, 1), (2, 1)): (0.3, True, 2, None)})
        assert len(set(assign_global_ids(tr, scored2, threshold=0.4).values())) == 2

    def test_one_per_camera_best_first(self):
        # Two cam2 tracklets compete for the cam1 slot: higher score wins,
        # the loser's merge is rejected (one-per-camera constraint).
        tr = {(1, 1), (2, 1), (2, 2)}
        scored = _scored({((1, 1), (2, 1)): (0.5, True, 2, None),
                          ((1, 1), (2, 2)): (0.7, True, 2, None)})
        out = assign_global_ids(tr, scored, threshold=0.4)
        assert out[(1, 1)] == out[(2, 2)]
        assert out[(1, 1)] != out[(2, 1)]


class TestResolveThreshold:
    def test_scalar_passthrough(self):
        assert resolve_threshold(0.4, 1, 2, "vehicle") == 0.4

    def test_pair_and_superclass_before_superclass(self):
        d = {"F-FR|vehicle": 0.35, "human": 0.25, "vehicle": 0.45}
        assert resolve_threshold(d, 1, 3, "vehicle") == 0.35    # pair|super-class
        assert resolve_threshold(d, 1, 3, "human") == 0.25      # super-class
        assert resolve_threshold(d, 1, 2, "vehicle") == 0.45    # super-class

    def test_pair_name_is_sorted_by_camera_id(self):
        d = {"FR-SR|vehicle": 0.35, "vehicle": 0.45}
        assert resolve_threshold(d, 5, 3, "vehicle") == 0.35    # (5,3) -> FR-SR

    def test_per_pair_end_to_end(self):
        # Same score on two pairs; the F-FR bar accepts, the FL-SL bar rejects.
        tr = {(1, 1), (3, 1), (2, 2), (4, 1)}
        scored = _scored({((1, 1), (3, 1)): (0.40, True, 2, None),
                          ((2, 2), (4, 1)): (0.40, True, 2, None)})
        out = assign_global_ids(tr, scored,
                                threshold={"F-FR|vehicle": 0.35, "vehicle": 0.45})
        assert out[(1, 1)] == out[(3, 1)]
        assert out[(2, 2)] != out[(4, 1)]

    def test_per_superclass_end_to_end(self):
        tr = {(1, 1), (2, 1), (1, 2), (2, 2)}
        scored = _scored({((1, 1), (2, 1)): (0.45, True, 9, None),    # human
                          ((1, 2), (2, 2)): (0.45, True, 2, None)})   # vehicle
        out = assign_global_ids(tr, scored,
                                threshold={"human": 0.3, "vehicle": 0.5})
        assert out[(1, 1)] == out[(2, 1)]
        assert out[(1, 2)] != out[(2, 2)]


class TestLinkRule:
    def test_rule(self):
        assert link_rule(0.5, True, 0.4)
        assert not link_rule(0.3, True, 0.4)
        assert not link_rule(0.5, False, 0.4)
        # veto: rejected below 0.5 unless S >= 0.6; missing cosine never vetoes
        assert not link_rule(0.5, True, 0.4, veto_cos=0.2, veto_min=0.5, veto_rescue=0.6)
        assert link_rule(0.6, True, 0.4, veto_cos=0.2, veto_min=0.5, veto_rescue=0.6)
        assert link_rule(0.5, True, 0.4, veto_cos=0.5, veto_min=0.5, veto_rescue=0.6)
        assert link_rule(0.5, True, 0.4, veto_cos=None, veto_min=0.5, veto_rescue=0.6)

    def test_merged_out_lists_merges(self):
        tr = {(1, 1), (2, 1), (2, 2)}
        scored = _scored({((1, 1), (2, 1)): (0.5, True, 2, None),
                          ((1, 1), (2, 2)): (0.7, True, 2, None)})
        merged = []
        assign_global_ids(tr, scored, threshold=0.4, merged_out=merged)
        assert merged == [_pair((1, 1), (2, 2))]


class TestRankKey:
    def test_rank_factor_mapping(self):
        assert _rank_factor(None) == 1.0
        assert _rank_factor(0.5) == 1.0
        assert _rank_factor(0.9) == 1.0
        assert _rank_factor(0.2) == pytest.approx(0.5)
        assert _rank_factor(-0.1) == RANK_FLOOR
        assert _rank_factor(-1.0) == RANK_FLOOR

    def test_rank_key_none_matches_default(self):
        tr = {(1, 1), (2, 1), (2, 2)}
        scored = _scored({((1, 1), (2, 1)): (0.5, True, 2, 0.9),
                          ((1, 1), (2, 2)): (0.7, True, 2, -0.5)})
        a = assign_global_ids(tr, scored, threshold=0.4)
        b = assign_global_ids(tr, scored, threshold=0.4, rank_key=None)
        assert a == b

    def test_rank_flips_contested_slot(self):
        # Geometry alone picks (2,2) (0.7 > 0.5); cosine ranking demotes it
        # (cos -0.5 -> floor 0.25 -> rank 0.175) below (2,1) (cos 0.9 -> 0.5),
        # so the appearance-consistent partner wins the cam2 slot.
        tr = {(1, 1), (2, 1), (2, 2)}
        scored = _scored({((1, 1), (2, 1)): (0.5, True, 2, 0.9),
                          ((1, 1), (2, 2)): (0.7, True, 2, -0.5)})
        geo = assign_global_ids(tr, scored, threshold=0.4)
        assert geo[(1, 1)] == geo[(2, 2)]
        ranked = assign_global_ids(tr, scored, threshold=0.4, rank_key="cos_med")
        assert ranked[(1, 1)] == ranked[(2, 1)]
        assert ranked[(1, 1)] != ranked[(2, 2)]

    def test_rank_cannot_resurrect_subthreshold_edge(self):
        # rank_key only reorders; an edge below the bar stays rejected even
        # with perfect cosine.
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.3, True, 2, 1.0)})
        out = assign_global_ids(tr, scored, threshold=0.4, rank_key="cos_med")
        assert out[(1, 1)] != out[(2, 1)]

    def test_missing_cosine_ranks_by_geometry(self):
        # cos None -> factor 1.0: pure-geometry rank, not penalised.
        tr = {(1, 1), (2, 1), (2, 2)}
        scored = _scored({((1, 1), (2, 1)): (0.5, True, 2, None),
                          ((1, 1), (2, 2)): (0.7, True, 2, None)})
        out = assign_global_ids(tr, scored, threshold=0.4, rank_key="cos_med")
        assert out[(1, 1)] == out[(2, 2)]



class TestVeto:
    def test_veto_rejects_low_cosine_accepted_edge(self):
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.45, True, 2, None)})
        scored[_pair((1, 1), (2, 1))]["cos_med_adapted"] = 0.1
        out = assign_global_ids(tr, scored, threshold=0.4)
        assert out[(1, 1)] == out[(2, 1)]            # no veto -> merged
        out = assign_global_ids(tr, scored, threshold=0.4,
                                veto_key="cos_med_adapted", veto_min=0.4,
                                veto_rescue=0.6)
        assert out[(1, 1)] != out[(2, 1)]            # vetoed

    def test_high_geometry_rescues_from_veto(self):
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.7, True, 2, None)})
        scored[_pair((1, 1), (2, 1))]["cos_med_adapted"] = -0.5
        out = assign_global_ids(tr, scored, threshold=0.4,
                                veto_key="cos_med_adapted", veto_min=0.4,
                                veto_rescue=0.6)
        assert out[(1, 1)] == out[(2, 1)]            # iou 0.7 >= rescue 0.6

    def test_missing_veto_value_never_vetoes(self):
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.45, True, 2, None)})
        out = assign_global_ids(tr, scored, threshold=0.4,
                                veto_key="cos_med_adapted", veto_min=0.4,
                                veto_rescue=0.6)
        assert out[(1, 1)] == out[(2, 1)]

    def test_veto_cannot_accept_subthreshold(self):
        tr = {(1, 1), (2, 1)}
        scored = _scored({((1, 1), (2, 1)): (0.3, True, 2, None)})
        scored[_pair((1, 1), (2, 1))]["cos_med_adapted"] = 0.99
        out = assign_global_ids(tr, scored, threshold=0.4,
                                veto_key="cos_med_adapted", veto_min=0.4,
                                veto_rescue=0.6)
        assert out[(1, 1)] != out[(2, 1)]
