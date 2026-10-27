"""Tests for states.py helpers on small synthetic frames (no tracking archives needed)."""
import numpy as np
import pandas as pd

from states import state_features, PossessionIndex, clock_running, BASKET, FEATURES

OFF, DEF = 1, 2


def frame(basket2=False):
    """One frame: offense attacking the basket at x=5.35 (or its 180-degree rotation when basket2)."""
    off = np.array([[20., 25.], [15., 10.], [30., 40.], [12., 30.], [40., 20.]])
    dfn = np.array([[18., 24.], [14., 12.], [28., 38.], [10., 26.], [42., 25.]])
    ball = np.array([20., 25., 3.])
    xy = np.vstack([off, dfn])
    if basket2:
        xy = np.array([94., 50.]) - xy
        ball = np.array([94 - 20., 50 - 25., 3.])
    pid = np.arange(101, 111)
    team = np.array([OFF] * 5 + [DEF] * 5)
    return dict(n=1, has_ball=np.array([True]), ball=ball[None], pid=pid[None], team=team[None], xy=xy[None],
                quarter=np.array([1]), qt=np.array([300.0]), block=np.array([1]), ut=np.array([0.0]))


def test_rotation_makes_both_directions_identical():
    f1 = state_features(frame(False), 0, 101, OFF, 5.35, 14.0, 3)
    f2 = state_features(frame(True), 0, 101, OFF, 88.65, 14.0, 3)
    assert f1 is not None and f2 is not None
    for k in FEATURES:
        if k == 'holder_speed':
            continue                      #needs earlier frames
        assert np.allclose(f1[k], f2[k], equal_nan=True), k


def test_feature_values():
    f = state_features(frame(False), 0, 101, OFF, 5.35, 14.0, 3)
    assert np.isclose(f['ball_dist'], np.hypot(20 - BASKET[0], 0))
    assert np.isclose(f['ball_xn'], 20 - 5.35) and np.isclose(f['ball_yn'], 0) and np.isclose(f['ball_angle'], 0)
    assert np.isclose(f['closest_def'], np.hypot(2, 1))      #defender at (18, 24)
    assert f['n_def_6'] == 1 and f['n_def_10'] == 2          #(18,24) and (14,12)? (14,12) is 13.4 away -> check below
    assert f['margin'] == 3 and f['shot_clock'] == 14.0


def test_missing_player_gives_none():
    fr = frame(False)
    fr['xy'][0, 3] = np.nan
    assert state_features(fr, 0, 101, OFF, 5.35, 14.0, 0) is None


def test_clock_running_detects_stopped_clock():
    n = 100
    qt = np.concatenate([np.linspace(300, 296.04, 50), np.full(50, 296.0)])     #25 Hz then stopped
    fr = dict(n=n, qt=qt, block=np.ones(n, dtype=int))
    run = clock_running(fr)
    assert run[20] and not run[80]


def test_possession_index_matches_team_and_clock():
    table = pd.DataFrame(dict(poss_idx=[0, 1], period=[1, 1], offense_team=[OFF, DEF], start_clock=[720, 700],
                              end_clock=[700, 680], points=[2, 0], off_margin_start=[0, 0]))
    idx = PossessionIndex(table)
    row, status, bd = idx.locate(1, 710.0, OFF)
    assert status == 'ok' and row.poss_idx == 0 and np.isclose(bd, 10)
    assert idx.locate(1, 710.0, DEF)[1] == 'team_mismatch'
    row, status, _ = idx.locate(1, 700.4, DEF)               #inside the 1 s tolerance of the boundary
    assert status == 'ok' and row.poss_idx == 1
    assert idx.locate(1, 500.0, OFF)[1] == 'uncovered'
