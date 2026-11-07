import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import executor_v2 as E                       # noqa: E402
from executor_data import F_DIM, G_DIM         # noqa: E402
from profiles import build_profiles            # noqa: E402
from play_cond import PlayTargets, TOK_EXTRA   # noqa: E402
from game_engine import Engine, rot            # noqa: E402
from test_executor import fake_game            # noqa: E402


class StubDiff:
    ranges = None
    shot = None

    def _xpts(self, models, F):
        return pd.DataFrame({'p_make': [0.45], 'p_three': [0.3], 'xpts': [1.0]})

    def predict(self, P, T, defenders, basket, shot_clock=np.nan, p_vel=(0, 0), t_vel=(0, 0)):
        return dict(p_complete=0.9, in_distribution=True, extrapolated=False)


def _eng(seed=0, **kw):
    games = [fake_game(300, seed=s) for s in range(2)]
    for d in games:
        d['quarter'] = np.ones(len(d['frame']), int)
    table = build_profiles(games)
    model = E.build_model(F_DIM + TOK_EXTRA, G_DIM + 3, k_modes=3, d=32, n_layers=1, heads=2, ff=64)
    return Engine(model, StubDiff(), plays=PlayTargets(), seed=seed, **kw).reset(games, table, [(0, 60), (1, 70), (0, 90)])


def test_quarters_finish_and_box_score_is_consistent():
    eng = _eng(shoot_scale=6.0, pass_scale=4.0, tov_mult=2.0)
    eng.game_clock[:] = 60.0
    eng.run(max_blocks=400)
    assert not eng.alive.any()
    s = eng.stats
    assert np.allclose(s['pts'], 2 * (s['fgm'] - s['fg3m']) + 3 * s['fg3m'] + s['ftm'])
    assert np.allclose(eng.score, s['pts'])
    assert (s['fga'] >= s['fgm']).all() and (s['fta'] >= s['ftm']).all()
    assert s['poss'].sum() >= 3


def test_possession_change_keeps_players_and_rotates_court():
    eng = _eng(seed=2)
    ids_before = sorted(eng.ids(0).tolist())
    pos_before = eng.hist[2][0].copy()
    eng._flip(0)
    assert sorted(eng.ids(0).tolist()) == ids_before
    assert np.allclose(eng.hist[2][0][:5], rot(pos_before[5:]))
    assert eng._off(0) == 1 and eng.shot_clock[0] == 24.0


def test_positions_stay_on_court():
    eng = _eng(seed=3, shoot_scale=3.0)
    eng.game_clock[:] = 30.0
    eng.run(max_blocks=200)
    P = np.stack(eng.traj)
    assert P[..., 0].min() >= -1e-6 and P[..., 0].max() <= 94 + 1e-6 and P[..., 1].min() >= -1e-6 and P[..., 1].max() <= 50 + 1e-6
