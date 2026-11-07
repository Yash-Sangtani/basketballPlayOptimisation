import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.optimize import linear_sum_assignment
import defense as DF


def _pos(rng, n=50):
    off = rng.uniform([5, 2], [40, 48], (n, 5, 2))
    dfn = off + rng.normal(0, 3, (n, 5, 2))
    perm = np.array([rng.permutation(5) for _ in range(n)])
    return np.concatenate([off, np.take_along_axis(dfn, perm[:, :, None], 1)], 1), perm


def test_assign_matches_hungarian():
    rng = np.random.default_rng(0)
    pos, _ = _pos(rng)
    man = DF.assign(pos)
    for n in range(len(pos)):
        D = np.hypot(pos[n, 5:, None, 0] - pos[n, None, :5, 0], pos[n, 5:, None, 1] - pos[n, None, :5, 1])
        r, c = linear_sum_assignment(D)
        assert np.isclose(D[np.arange(5), man[n]].sum(), D[r, c].sum())


def test_knobs_man_on_top_of_man():
    rng = np.random.default_rng(1)
    off = rng.uniform([10, 5], [40, 45], (20, 5, 2))
    pos = np.concatenate([off, off + 0.01], 1)
    k = DF.knobs(pos, off[:, 0])
    assert k.shape == (20, 6) and np.isfinite(k).all()
    assert k[:, 0].max() < 0.1            # gap ~ 0
    assert k[:, 2].max() < 0.1            # ball pressure ~ 0
    mf = DF.man_features(pos, off[:, 0])
    assert mf.shape == (20, 5, 4) and mf[..., 2].max() < 0.01
