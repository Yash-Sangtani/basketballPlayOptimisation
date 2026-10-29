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
from sim import Sim                            # noqa: E402
from test_executor import fake_game            # noqa: E402


class StubDiff:
    c_fail = 1.16

    def shot_xpts(self, S, D, basket, vel=(0, 0), shot_clock=np.nan):
        return 1.0

    def predict_many(self, states):
        n = len(states)
        return pd.DataFrame({'p_complete': np.full(n, 0.9), 'in_distribution': np.ones(n, bool), 'extrapolated': np.zeros(n, bool)})


def _setup():
    games = [fake_game(300, seed=s) for s in range(2)]
    table = build_profiles(games)
    model = E.build_model(F_DIM, G_DIM, k_modes=3, d=32, n_layers=1, heads=2, ff=64)
    return games, table, model


def test_all_episodes_terminate_with_valid_outcomes():
    games, table, model = _setup()
    starts = [(g, k) for g in range(2) for k in (50, 100, 150)]
    sim = Sim(model, StubDiff(), seed=1, shoot_scale=5.0, pass_scale=5.0, loss_rate=0.1).reset(games, table, starts).run(max_blocks=200)
    assert not sim.alive.any()
    assert set(np.unique(sim.end)) <= {1, 2, 3, 4}
    assert np.isfinite(sim.reward).all()
    assert (sim.reward[sim.end == 1] == 1.0).all() and (sim.reward[sim.end == 2] == -1.16).all()


def test_positions_stay_in_court_and_shot_clock_violation():
    games, table, model = _setup()
    sim = Sim(model, StubDiff(), seed=2, shoot_scale=0.0, pass_scale=0.0).reset(games, table, [(0, 60), (1, 60)])
    sim.shot_clock[:] = 1.0
    sim.run(max_blocks=20)
    assert (sim.end == 3).all()
    P = np.stack(sim.traj)
    assert P[..., 0].min() >= 0 and P[..., 0].max() <= 94 and P[..., 1].min() >= 0 and P[..., 1].max() <= 50
