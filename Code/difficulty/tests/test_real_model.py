import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from api import PassValueModel, DEFAULT_MODEL, DEFAULT_SHOT_MODEL   # noqa: E402

pytestmark = pytest.mark.skipif(not (os.path.exists(DEFAULT_MODEL) and os.path.exists(DEFAULT_SHOT_MODEL)), reason='trained models not present')
B = [5.35, 25.0]
P = [30, 25]


def test_defender_on_the_lane_is_harder_than_an_open_pass():
    m = PassValueModel()
    open_ = m.evaluate(P, [12, 45], [[28, 24], [40, 30], [22, 18], [35, 12], [18, 26]], B, 12.0, (2, 0))
    blocked = m.evaluate(P, [22, 38], [[26, 31], [27, 30], [12, 18], [40, 10], [16, 24]], B, 12.0, (2, 0))
    assert blocked['difficulty'] > open_['difficulty']
    assert blocked['relative_difficulty'] > open_['relative_difficulty']
    for r in (open_, blocked):
        assert 0 <= r['p_complete'] <= 1 and np.isfinite(r['ev_pass'])


def test_mirror_symmetry_of_the_trained_model():
    m = PassValueModel()
    D = np.array([[28, 24], [12, 18], [22, 31], [40, 10], [16, 40]], float)
    mir = lambda a: np.array([94.0, 50.0]) - np.asarray(a, float)
    a = m.predict(P, [22, 38], D, B, 12.0, (2, 0), (0, 1))
    b = m.predict(mir(P), mir([22, 38]), mir(D), mir(B), 12.0, (-2, 0), (0, -1))
    assert a['p_complete'] == pytest.approx(b['p_complete'], abs=1e-9)


def test_long_shots_are_valued_as_heaves_not_extrapolated():
    m = PassValueModel()
    far = m.shot_xpts([55, 25], [[50, 20], [60, 30], [58, 25]], B, (0, 0), 10.0)       # about 50 ft from the basket
    assert far == pytest.approx(3.0 * m.ranges['heave_fg'], abs=1e-9)
    near = m.shot_xpts([9, 25], [[20, 20], [25, 30], [30, 25]], B, (0, 0), 10.0)        # open layup
    assert near > far


def test_out_of_range_inputs_are_flagged():
    m = PassValueModel()
    ok = m.predict(P, [22, 38], [[26, 31], [27, 30], [12, 18], [40, 10], [16, 24]], B, 12.0, (2, 0))
    odd = m.predict([2, 3], [90, 48], [[50, 20]], B, 12.0, (40, 0), (0, 40))
    assert odd['in_distribution'] is False and odd['n_ood_features'] > ok['n_ood_features']
