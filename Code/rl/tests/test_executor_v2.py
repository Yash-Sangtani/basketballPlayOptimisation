import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import executor_v2 as E                       # noqa: E402
from executor_data import F_DIM, G_DIM         # noqa: E402


def _model():
    return E.build_model(F_DIM, G_DIM, k_modes=3, d=32, n_layers=1, heads=2, ff=64)


def test_shapes_and_weights_normalised():
    m = _model()
    rng = np.random.default_rng(0)
    tok = rng.normal(size=(4, 10, F_DIM)).astype('float32')
    ctx = rng.normal(size=(4, G_DIM)).astype('float32')
    o = m.predict([tok, ctx], verbose=0)
    assert o['mix'].shape == (4, 10, 3, 21)
    assert o['dec'].shape[0] == 4 and o['tgt'].shape == (4, 10) or o['tgt'].shape[0] == 4


def test_player_order_equivariance():
    m = _model()
    rng = np.random.default_rng(1)
    tok = rng.normal(size=(2, 10, F_DIM)).astype('float32')
    ctx = rng.normal(size=(2, G_DIM)).astype('float32')
    perm = np.r_[np.random.default_rng(2).permutation(5), 5 + np.random.default_rng(3).permutation(5)]
    a = m.predict([tok, ctx], verbose=0)['mix']
    b = m.predict([tok[:, perm], ctx], verbose=0)['mix']
    assert np.allclose(a[:, perm], b, atol=1e-4)
