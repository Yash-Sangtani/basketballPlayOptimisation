import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from ball import held_position, flight_position, flight_time, PASS_SPEED, HELD_Z   # noqa: E402


def test_held_ball_is_at_the_holder():
    b = held_position(np.array([[10.0, 20.0], [30.0, 5.0]]))
    assert b.shape == (2, 3) and np.allclose(b[:, 2], HELD_Z) and np.allclose(b[:, :2] - np.array([[10, 20], [30, 5]]), [-0.26, 0.0])


def test_flight_starts_at_passer_ends_at_receiver_and_takes_distance_over_speed():
    p0, p1 = np.array([10.0, 10.0]), np.array([38.0, 10.0])
    assert abs(flight_time(p0, p1) - 28.0 / PASS_SPEED) < 1e-9
    a, mid, b = flight_position(p0, p1, 0.0), flight_position(p0, p1, 0.5), flight_position(p0, p1, 5.0)
    assert np.allclose(a[:2], p0 + [-0.26, 0]) and np.allclose(b[:2], p1) and mid[2] > a[2]
    assert np.isclose(b[2], HELD_Z)
