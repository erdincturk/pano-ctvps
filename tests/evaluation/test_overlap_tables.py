"""Overlap tables: AQ from the tables equals STQuality on a small synthetic sequence."""
import numpy as np

from panoctvps.data.constants import NUM_CLASSES, THING_CLASSES
from panoctvps.evaluation import STQuality
from panoctvps.evaluation.overlap_tables import OFFSET, gt_objects, score_scene

CAR = 2


def _frames():
    """Two frames of one camera, 4 x 6 pixels: two ground-truth cars; the prediction splits car 1 over two tracks."""
    gs = np.zeros((4, 6), np.int64); gi = np.zeros((4, 6), np.int64)
    gs[:, :3] = CAR; gi[:, :3] = 1
    gs[:, 3:] = CAR; gi[:, 3:] = 2
    ps = gs.copy()
    pi = [np.where(gi == 1, 1, 2), np.where(gi == 1, 1, 2)]   # local instances 1 and 2 in each frame
    tid = [{1: 10, 2: 20}, {1: 11, 2: 20}]                    # track ids: car 1 switches from 10 to 11
    return gs, gi, ps, pi, tid


def test_aq_matches_stquality():
    gs, gi, ps, pi, tid = _frames()
    stq = STQuality(num_classes=NUM_CLASSES, thing_list=THING_CLASSES, ignore_label=0)
    T, P, G, F = [], [], [], []
    for f in range(2):
        pred_ids = np.vectorize(lambda k: tid[f].get(int(k), 0))(pi[f])
        stq.update(ps, pred_ids, gs, gi)
        F.append((f, 1, f))
        for g in (1, 2):
            m = gi == g
            G.append((f, CAR * OFFSET + g, float(m.sum()), int(m.sum())))
            for k in (1, 2):
                ov = int((m & (pi[f] == k)).sum())
                if ov:
                    T.append((f, CAR * OFFSET + g, k, CAR, float(ov), ov))
        for k in (1, 2):
            n = int((pi[f] == k).sum())
            P.append((f, k, CAR, float(n), n))
    tab = {"frames": np.array(F, dtype=[("fi", "i4"), ("cam", "i1"), ("ts", "i8")]),
           "tpa": np.array(T, dtype=[("f", "i4"), ("g", "i8"), ("k", "i4"), ("s", "i2"), ("w", "f8"), ("u", "i8")]),
           "psize": np.array(P, dtype=[("f", "i4"), ("k", "i4"), ("s", "i2"), ("w", "f8"), ("u", "i8")]),
           "gsize": np.array(G, dtype=[("f", "i4"), ("g", "i8"), ("w", "f8"), ("u", "i8")]),
           "sq_weighted": np.zeros((2, NUM_CLASSES)), "sq_cams": np.array([1]), "sq_cam": np.zeros((1, 2, NUM_CLASSES))}

    def ident(f, k, s):
        return np.array([tid[int(a)][int(b)] for a, b in zip(f, k)], dtype=np.int64)

    out = score_scene(tab, ident, ident)
    aq_tables = out["agg"]["aq_num"] / out["agg"]["N"]
    aq_ref = stq.result()["AQ"]
    assert abs(aq_tables - aq_ref) < 1e-12
    gcls, gfr = gt_objects(tab)
    assert gcls == {1: CAR, 2: CAR} and gfr[(1, 1)] == {0, 1}
