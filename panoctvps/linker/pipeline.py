"""The linker on one cached scene: proposal -> scoring -> assignment.

One entry point shared by the wSTQ evaluation and the submission builder.
"""
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from panoctvps.linker.assigner import assign_global_ids
from panoctvps.linker.link_eval import load_adapted_cosines
from panoctvps.linker.overlap import overlap_by_partner
from panoctvps.linker.proposer import SUPERCLASS, propose_candidates
from panoctvps.linker.scorer import score_candidates

#: Cache field holding the per-camera track ids of the tracker.
DEFAULT_TID_FIELD = "tracker_id_deployed"


def linker_assignment(npz, cams, configs,
                      cosines_path: Optional[Path] = None,
                      tid_field: str = DEFAULT_TID_FIELD,
                      ) -> Dict[str, Dict[Tuple[int, int], int]]:
    """Run proposal + scoring once, return {config_name: {(cam, tid): global_id}}.

    configs: list of (name, cfg) with cfg = {"tau": float|dict, "rank_key":
    str|None, and optionally "veto_key", "veto_min", "veto_rescue"}, passed to
    assign_global_ids. cosines_path: the scene's adapted-cosine file
    (scripts/build_cosines.py), needed by the appearance veto.
    """
    ovbp = overlap_by_partner(cams)
    cands = propose_candidates(npz, ovbp, tid_field=tid_field)
    scored = score_candidates(npz, cams, cands, tid_field=tid_field)
    if cosines_path is not None and cosines_path.exists():
        idx = load_adapted_cosines(cosines_path)
        for pair, s in scored.items():
            s["cos_med_adapted"] = idx.get(pair)
    all_tr: Set[Tuple[int, int]] = set()
    for x in npz["records"]:
        if SUPERCLASS.get(int(x["class_id"])) is None:
            continue
        tid = int(x[tid_field])
        if tid > 0:
            all_tr.add((int(x["camera_id"]), tid))
    return {name: assign_global_ids(all_tr, scored, threshold=cfg["tau"],
                                    rank_key=cfg.get("rank_key"),
                                    veto_key=cfg.get("veto_key"),
                                    veto_min=cfg.get("veto_min", 0.0),
                                    veto_rescue=cfg.get("veto_rescue", 1.0))
            for name, cfg in configs}
