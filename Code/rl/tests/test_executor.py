import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from executor_data import F_DIM, G_DIM, make_tokens, make_ctx, game_samples, N_PROFILE   # noqa: E402
from profiles import build_profiles, profile_lookup, PROFILE_COLS    # noqa: E402


def fake_game(n=240, seed=0, fast_player=101, big=True):
    rng = np.random.default_rng(seed)
    ids = np.tile(np.arange(101, 111), (n, 1))
    pos = np.cumsum(rng.normal(0, 0.15, (n, 10, 2)), axis=0).astype(np.float32) + np.array([30.0, 25.0], np.float32)
    if fast_player and big:
        pos[:, 0] += np.cumsum(np.full((n, 2), 0.6), axis=0).astype(np.float32) % 20
    vel = np.diff(pos, axis=0, prepend=pos[:1]) / 0.2
    roster = pd.DataFrame(dict(player=list(range(101, 111)), team=[1] * 5 + [2] * 5, first='a', last='b', position=['G', 'G', 'F', 'F', 'C'] * 2, jersey='0'))
    return dict(game='g%d' % seed, frame=np.arange(n) * 5, quarter=np.ones(n, np.int8), quarter_time=np.linspace(700, 600, n).astype(np.float32),
                shot_clock=np.linspace(24, 5, n).astype(np.float32), live=np.ones(n, bool), ctrl_team=np.full(n, 1), poss_id=np.zeros(n, np.int64),
                t_in_poss=np.linspace(0, 20, n).astype(np.float32), frontcourt=np.ones(n, bool), t_front=np.linspace(2, 20, n).astype(np.float32),
                holder=np.full(n, 103), pos=pos, vel=vel.astype(np.float32), ids=ids, ball=np.c_[pos[:, 2], np.full(n, 3.0)].astype(np.float32),
                flipped=np.zeros(n, bool), passes=pd.DataFrame(columns=['frame', 'passer', 'actual_target', 'heading_target', 'label', 'outcome', 'live', 'team']),
                shots=pd.DataFrame(dict(frame=[300, 600], shooter=[103, 103], made=[1, 0], is3=[0, 1], team=[1, 1], assisted=[0, 0], assister=[0, 0], quarter=[1, 1])), roster=roster)


def test_token_and_ctx_dims():
    d = fake_game()
    s = game_samples(d, build_profiles([d]), stride=3)
    assert s['tokens'].shape[1:] == (10, F_DIM) and s['ctx'].shape[1] == G_DIM
    assert s['y'].shape[1:] == (10, 5, 2) and not np.isnan(s['tokens']).any()
    assert set(np.unique(s['action'])) <= {-1, 0, 1, 2}


def test_targets_are_future_displacements():
    d = fake_game()
    s = game_samples(d, build_profiles([d]), stride=1)
    k = s['state'][0]
    np.testing.assert_allclose(s['y'][0, 2, 4] * 3.0, d['pos'][k + 5, 2] - d['pos'][k, 2], atol=1e-4)


def test_profiles_shrink_and_unknown_is_zero():
    games = [fake_game(seed=i) for i in range(3)]
    t = build_profiles(games)
    assert list(t[PROFILE_COLS].columns) == PROFILE_COLS and len(t) == 10
    assert t.loc[101, 'off_speed'] > t.loc[102, 'off_speed']             # the fast player has a higher speed profile
    z = profile_lookup(t, np.array([[999, 101]]))
    assert np.allclose(z[0, 0], 0) and not np.allclose(z[0, 1], 0)


def test_model_is_permutation_equivariant():
    from executor_model import build_model
    m = build_model(F_DIM, G_DIM, d=32, n_layers=2, heads=2, ff=48)
    rng = np.random.default_rng(1)
    tok = rng.normal(size=(6, 10, F_DIM)).astype(np.float32)
    ctx = rng.normal(size=(6, G_DIM)).astype(np.float32)
    tok[..., 13] = 0.0                                                    # no holder: the decision head is not under test here
    perm = rng.permutation(10)
    a = m.predict([tok, ctx], verbose=0)['move']
    b = m.predict([tok[:, perm], ctx], verbose=0)['move']
    np.testing.assert_allclose(a[:, perm], b, atol=1e-4)
