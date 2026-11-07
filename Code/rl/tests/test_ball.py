import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from ball import held_position, flight_position, flight_time, shot_position, shot_time, HELD_Z, PASS_T0, PASS_T1, RIM_Z   # noqa: E402


def test_held_ball_is_at_the_holder():
    b = held_position(np.array([[10.0, 20.0], [30.0, 5.0]]))
    assert b.shape == (2, 3) and np.allclose(b[:, 2], HELD_Z) and np.allclose(b[:, :2] - np.array([[10, 20], [30, 5]]), [-0.26, 0.0])


def test_flight_time_follows_the_fitted_line_and_the_ball_arrives_at_the_receiver():
    p0, p1 = np.array([10.0, 10.0]), np.array([38.0, 10.0])
    assert abs(flight_time(p0, p1) - (PASS_T0 + PASS_T1 * 28.0)) < 1e-9
    a, mid, b = flight_position(p0, p1, 0.0), flight_position(p0, p1, flight_time(p0, p1) / 2), flight_position(p0, p1, 5.0)
    assert np.allclose(a[:2], p0 + [-0.26, 0]) and np.allclose(b[:2], p1) and mid[2] > a[2]
    assert np.isclose(b[2], HELD_Z)


def test_pass_speed_is_in_the_measured_range():
    for d, lo, hi in ((8.0, 12.0, 20.0), (17.0, 20.0, 28.0), (40.0, 28.0, 40.0)):
        T = flight_time(np.zeros(2), np.array([d, 0.0]))
        assert lo <= d / T <= hi, (d, d / T)


def test_shot_arc_goes_up_and_ends_at_the_rim_height():
    rim = np.array([5.35, 25.0])
    p0 = np.array([25.0, 25.0])
    T = shot_time(p0, rim)
    zs = [shot_position(p0, rim, t)[2] for t in np.linspace(0, T, 9)]
    assert max(zs) > RIM_Z + 2 and abs(zs[-1] - RIM_Z) < 1e-6 and np.allclose(shot_position(p0, rim, T)[:2], rim)
