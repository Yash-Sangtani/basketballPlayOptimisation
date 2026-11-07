import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from pass_features import pass_features, FEATURE_NAMES, SITUATION_FEATURES   # noqa: E402

BASKET = np.array([5.35, 25.0])
P, T = np.array([30.0, 25.0]), np.array([18.0, 35.0])
D = np.array([[28.0, 24.0], [10.0, 20.0], [22.0, 30.0], [40.0, 10.0], [15.0, 40.0]])


def mirror(a):
    return np.array([94.0, 50.0]) - np.asarray(a, float)


def test_all_features_present_and_finite():
    f = pass_features(P, T, D, BASKET, 14.0, (2.0, 1.0), (-1.0, 3.0))
    assert list(f) == FEATURE_NAMES
    assert all(np.isfinite(v) for v in f.values())
    assert set(SITUATION_FEATURES) <= set(FEATURE_NAMES)


def test_court_mirror_gives_identical_features():
    a = pass_features(P, T, D, BASKET, 14.0, (2.0, 1.0), (-1.0, 3.0))
    b = pass_features(mirror(P), mirror(T), mirror(D), mirror(BASKET), 14.0, (-2.0, -1.0), (1.0, -3.0))
    for k in FEATURE_NAMES:
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


def test_defender_on_the_line_is_interceptable_and_far_defenders_are_not():
    clear = np.array([[60.0, 5.0], [70.0, 45.0]])
    on_line = np.vstack([clear, [24.0, 30.0]])            # roughly midway along P -> T
    f0, f1 = pass_features(P, T, clear, BASKET), pass_features(P, T, on_line, BASKET)
    assert f0['intercept_margin'] > 0 and f0['n_def_can_intercept'] == 0
    assert f1['intercept_margin'] < 0 and f1['n_def_can_intercept'] >= 1
    assert f1['closest_def_trajectory'] < f0['closest_def_trajectory']


def test_longer_pass_has_smaller_margin_with_the_same_defender():
    d = np.array([[20.0, 28.0]])
    near = pass_features(P, np.array([22.0, 27.0]), d, BASKET)
    far = pass_features(P, np.array([8.0, 40.0]), d, BASKET)
    assert far['dist'] > near['dist']


def test_receiver_motion_signs():
    f = pass_features(P, T, D, BASKET, 10.0, (0, 0), (-3.0, 4.0))      # receiver running away from the passer along the pass
    assert f['recv_speed_along'] > 0
    g = pass_features(P, T, D, BASKET, 10.0, (0, 0), (3.0, -4.0))
    assert g['recv_speed_along'] < 0


def test_undefined_geometry_returns_none():
    assert pass_features(P, P + 0.5, D, BASKET) is None
    assert pass_features(P, T, np.empty((0, 2)), BASKET) is None
    assert pass_features(np.array([np.nan, 1.0]), T, D, BASKET) is None
