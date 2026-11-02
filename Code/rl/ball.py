"""A minimal ball model for roll-outs and the simulator, with flight times and heights fitted to real passes (tracking, 5 Hz, 56,000 live completed passes).

  held    the ball sits at the holder's position with a small offset (measured: dx -0.26 ft, dy 0, height 3.4 ft above the floor)
  pass    straight line from the passer to the receiver (lead point) in  T = 0.33 + 0.0212 * d  seconds  (measured 0.43 + 0.0212 d between the release state
          and the state where the receiver is detected as holding the ball; 0.1 s is taken off for the detection delay), i.e. an average speed of about 22 ft/s
          for a 17-ft pass (real median 22.4 ft/s, 16.5 ft/s on 5-12 ft passes, 31.6 ft/s beyond 30 ft). Height: a parabola peaking at 4.67 + 0.0874 d ft
          (measured peak height of real passes).
  shot    arc from the shooter's hand (about 8 ft) to the rim (10 ft) in T = 0.45 + 0.035 d seconds, peaking about 0.3 d + 1 ft above the straight line.
Whether a pass is caught or a shot goes in is decided elsewhere (difficulty and shot models).
"""
import numpy as np

HELD_OFFSET = np.array([-0.26, 0.0])
HELD_Z = 3.4
PASS_T0, PASS_T1 = 0.33, 0.0212
PEAK0, PEAK1 = 4.67, 0.0874
MIN_FLIGHT_S = 0.3
RIM_Z = 10.0
RELEASE_Z = 8.0


def held_position(holder_xy):
    """Ball (x, y, z) when held by a player at holder_xy (..., 2)."""
    holder_xy = np.asarray(holder_xy, float)
    xy = holder_xy + HELD_OFFSET
    return np.concatenate([xy, np.full(holder_xy.shape[:-1] + (1,), HELD_Z)], axis=-1)


def flight_time(p0, p1):
    d = float(np.hypot(*(np.asarray(p1, float) - np.asarray(p0, float))))
    return max(PASS_T0 + PASS_T1 * d, MIN_FLIGHT_S)


def flight_position(p0, p1, t):
    """Ball (x, y, z) t seconds after release for a pass from p0 to p1 (positions of the passer and the intended receiver at release)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    T = flight_time(p0, p1)
    u = min(max(t / T, 0.0), 1.0)
    xy = p0 + (p1 - p0) * u + HELD_OFFSET * (1 - u)
    d = float(np.hypot(*(p1 - p0)))
    peak = max(PEAK0 + PEAK1 * d - HELD_Z, 0.0)
    z = HELD_Z + 4.0 * peak * u * (1 - u)
    return np.array([xy[0], xy[1], z])


def shot_time(p0, rim):
    d = float(np.hypot(*(np.asarray(rim, float) - np.asarray(p0, float))))
    return 0.45 + 0.035 * d


def shot_position(p0, rim, t):
    """Ball (x, y, z) t seconds after a shot released at p0 towards the rim."""
    p0, rim = np.asarray(p0, float), np.asarray(rim, float)
    T = shot_time(p0, rim)
    u = min(max(t / T, 0.0), 1.0)
    d = float(np.hypot(*(rim - p0)))
    xy = p0 + (rim - p0) * u
    z = RELEASE_Z + (RIM_Z - RELEASE_Z) * u + 4.0 * (0.3 * d + 1.0) * u * (1 - u)
    return np.array([xy[0], xy[1], z])
