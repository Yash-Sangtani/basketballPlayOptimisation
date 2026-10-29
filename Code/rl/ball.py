"""A minimal ball model for roll-outs and the simulator.

  held    the ball sits at the holder's position with a small offset (measured: dx -0.26 ft, dy 0, height 3.4 ft above the floor)
  flight  a pass leaves the passer's position at the release time and moves in a straight line at PASS_SPEED (28 ft/s, the median first-0.3 s
          ball speed of live completions measured in the difficulty work), rising along a parabola whose height grows with the distance,
          and arrives after distance / speed seconds. Whether the pass is caught is NOT decided here: the simulator asks the difficulty model
          (difficulty/api.py) and a shot model for outcomes.
"""
import numpy as np

HELD_OFFSET = np.array([-0.26, 0.0])
HELD_Z = 3.4
PASS_SPEED = 28.0
ARC_PER_FT = 0.06          # peak extra height per foot of pass distance (a 30 ft pass peaks about 1.8 ft above the held height)
MIN_FLIGHT_S = 0.2


def held_position(holder_xy):
    """Ball (x, y, z) when held by a player at holder_xy (..., 2)."""
    holder_xy = np.asarray(holder_xy, float)
    xy = holder_xy + HELD_OFFSET
    return np.concatenate([xy, np.full(holder_xy.shape[:-1] + (1,), HELD_Z)], axis=-1)


def flight_time(p0, p1):
    d = float(np.hypot(*(np.asarray(p1, float) - np.asarray(p0, float))))
    return max(d / PASS_SPEED, MIN_FLIGHT_S)


def flight_position(p0, p1, t):
    """Ball (x, y, z) t seconds after release for a pass from p0 to p1 (positions of the passer and the intended receiver at release)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    T = flight_time(p0, p1)
    u = min(max(t / T, 0.0), 1.0)
    xy = p0 + (p1 - p0) * u + HELD_OFFSET * (1 - u)
    d = float(np.hypot(*(p1 - p0)))
    z = HELD_Z + 4.0 * ARC_PER_FT * d * u * (1 - u)
    return np.array([xy[0], xy[1], z])
