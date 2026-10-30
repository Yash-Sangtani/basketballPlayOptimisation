import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from play_cond import PlayTargets, TOK_EXTRA, CTX_EXTRA, top5_cells   # noqa: E402


def test_features_zero_without_play_or_after_window():
    pt = PlayTargets()
    pos = np.random.default_rng(0).uniform([5, 5], [45, 45], (4, 10, 2))
    tok, ctx = pt.features(pos, np.array([2.0, 2.0, 7.0, 2.0]), np.array([3, -1, 3, 5]))
    assert tok.shape == (4, 10, TOK_EXTRA) and ctx.shape == (4, CTX_EXTRA)
    assert np.abs(tok[1]).sum() == 0 and np.abs(tok[2]).sum() == 0 and ctx[1].sum() == 0 and ctx[2].sum() == 0     # no play / after the window
    assert np.abs(tok[0, :5]).sum() > 0 and np.abs(tok[0, 5:]).sum() == 0 and ctx[0, 0] == 1                         # offence only


def test_targets_are_five_distinct_cells_with_mass():
    pt = PlayTargets()
    for k in range(pt.K):
        c = top5_cells(pt.grid[k, 3])
        assert len(c) == 5 and np.isfinite(c).all()


def test_lead_pass_ball_arrives_at_receiver():
    from test_sim import StubDiff, _setup
    from sim import Sim
    games, table, model = _setup()
    sim = Sim(model, StubDiff(), seed=3, shoot_scale=0.0, pass_scale=8.0, loss_rate=0.0).reset(games, table, [(0, 60), (1, 80), (0, 120), (1, 40)])
    sim.run(max_blocks=100)
    gaps = np.array(sim.arrival_gap)
    assert len(gaps) > 0 and gaps.max() < 0.5
