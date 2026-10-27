"""Cheap unit tests for the WS3 helpers.  Run:  cd Code && ../.venv/bin/python -m pytest analysis/test_ws3.py -q"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from ws3_common import shuffle_within_groups, auc_pr                      # noqa: E402
from ws3_baseline_tests import build_matches, concordance                 # noqa: E402
from ws3_sample_tests import tag_trk                                      # noqa: E402


def test_shuffle_keeps_game_rates_and_breaks_link():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(50), 200)
    x = rng.normal(size=len(g))
    y = (x + rng.normal(size=len(g)) > 1.5).astype(int)
    assert auc_pr(y, x)[0] > 0.7
    ys = shuffle_within_groups(y, g, seed=1)
    for k in range(50):
        assert ys[g == k].sum() == y[g == k].sum()
    assert abs(auc_pr(ys, x)[0] - 0.5) < 0.05


def test_build_matches_same_team_same_game_window():
    X = pd.DataFrame({'game': ['a', 'a', 'a', 'a', 'b'], 'fail': [1, 0, 0, 0, 0], 'backcourt': 0})
    M = pd.DataFrame({'team_passer': [1, 1, 2, 1, 1], 'game_time': [100.0, 95.0, 100.0, 120.0, 100.0]})
    m, nomatch = build_matches(X, M)
    assert m == {0: [1]} and nomatch == 0          # only the same-team, same-game, within 8 s completion
    X2 = X.copy()
    M2 = M.copy()
    M2.loc[1, 'game_time'] = 50.0
    m2, nomatch2 = build_matches(X2, M2)
    assert m2 == {} and nomatch2 == 1


def test_concordance_perfect_and_tied():
    matches = {0: [1, 2], 3: [4]}
    games = np.array(['a', 'a', 'a', 'b', 'b'])
    c, _, n = concordance(matches, np.array([0.9, 0.1, 0.2, 0.8, 0.3]), games, n_boot=20)
    assert c == 1.0 and n == 2
    c2, _, _ = concordance(matches, np.array([0.5, 0.5, 0.5, 0.5, 0.5]), games, n_boot=20)
    assert c2 == 0.5


def test_tag_priority():
    base = dict(pbp_tags='', clock_running=1.0, inbound=0.0, pbp_to_by_passer=0, trk_outcome='intercepted', oob_next_same=np.nan)
    row = lambda **k: pd.Series({**base, **k})
    assert tag_trk(row(clock_running=0.0, pbp_tags='shot_made')) == 'dead_ball_or_inbound'
    assert tag_trk(row(pbp_to_by_passer=2)) == 'pbp_lost_ball_by_releaser'
    assert tag_trk(row(pbp_tags='shot_miss,rebound')) == 'shot_or_rebound_context'
    assert tag_trk(row(trk_outcome='out_of_bounds', oob_next_same=1.0)) == 'oob_kept_by_offense'
    assert tag_trk(row()) == 'no_pbp_event_live'
