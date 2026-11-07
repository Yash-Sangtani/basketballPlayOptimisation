"""Pass-difficulty features as a PURE function of one court state.

Used identically by the training extractor (extract.py) and by the RL-facing API (api.py), so a pass scored
in a simulator gets exactly the features the model was trained on. Inputs are plain arrays in feet / ft/s:

    P, T        passer and (candidate) receiver position, shape (2,)
    D           defender positions, shape (k, 2)
    basket      the basket the offense attacks, shape (2,)  (the model is symmetric in left/right)
    shot_clock  seconds (NaN allowed)
    p_vel, t_vel  passer / receiver velocity (2,), ft/s (zeros if unknown)

Everything is measured at the release instant; nothing looks at what happens after it.
"""
import numpy as np

BALL_SPEED = 30.0       # ft/s, typical pass speed (median first-0.3 s ball speed of live completions: ~28 ft/s)
DEF_SPEED = 15.0        # ft/s, average defender closing speed over a short burst
DEF_REACH = 1.5         # ft, a defender can get a hand this far from his position
N_LANE_PTS = 9

FEATURE_NAMES = [
    # original 15 (same definitions as Code/passtracking.py `features`)
    'shot_clock', 'backcourt', 'dist', 'passer_basket_dist', 'ball_end_basket_dist', 'basket_angle',
    'closest_def_dist_passer', 'closest_def_dist_ball_end', 'closest_def_trajectory',
    'closest_def_angle_passer', 'closest_def_angle_ball_end', 'closest_def_trajectory_angle',
    'lane_n_def_5ft', 'passer_speed', 'target_speed',
    # new: interception geometry and kinematics
    'intercept_margin', 'n_def_can_intercept', 'second_def_dist_passer', 'second_def_dist_ball_end',
    'recv_speed_along', 'passer_speed_toward_recv',
]
# features that depend only on the passer's situation, not on who the receiver is
SITUATION_FEATURES = ['shot_clock', 'backcourt', 'passer_basket_dist', 'closest_def_dist_passer',
                      'second_def_dist_passer', 'passer_speed']


def _angle(a, v, b):
    """Interior angle (rad) at vertex v between rays v->a and v->b; NaN if degenerate."""
    r1, r2 = np.asarray(a, float) - v, np.asarray(b, float) - v
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(np.arccos(np.clip(r1 @ r2 / (n1 * n2), -1.0, 1.0)))


def pass_features(P, T, D, basket, shot_clock=np.nan, p_vel=(0.0, 0.0), t_vel=(0.0, 0.0)):
    """Feature dict (keys = FEATURE_NAMES) or None if the pass geometry is undefined."""
    P, T, D = np.asarray(P, float), np.asarray(T, float), np.asarray(D, float).reshape(-1, 2)
    basket = np.asarray(basket, float)
    p_vel, t_vel = np.asarray(p_vel, float), np.asarray(t_vel, float)
    if len(D) == 0 or np.isnan(P).any() or np.isnan(T).any() or np.isnan(D).any():
        return None
    vec = T - P
    L = float(np.hypot(*vec))
    if L < 1.0:
        return None
    # distance of each defender to the pass SEGMENT
    u = np.clip(((D - P) @ vec) / L ** 2, 0, 1)
    lane_d = np.hypot(*(D - (P + u[:, None] * vec)).T)
    d_p = np.hypot(*(D - P).T)
    d_t = np.hypot(*(D - T).T)
    ip, it, il = d_p.argmin(), d_t.argmin(), lane_d.argmin()
    sp, st = np.sort(d_p), np.sort(d_t)

    # interception margin: for points along the flight, how much earlier the ball gets there than the
    # best-placed defender could (seconds; negative = some defender can beat the ball to the line)
    s = np.linspace(1.0 / N_LANE_PTS, 1.0, N_LANE_PTS)
    pts = P[None, :] + s[:, None] * vec[None, :]
    t_ball = (s * L) / BALL_SPEED
    dd = np.hypot(D[:, None, 0] - pts[None, :, 0], D[:, None, 1] - pts[None, :, 1])
    t_def = np.maximum(dd - DEF_REACH, 0.0) / DEF_SPEED
    margin = t_def - t_ball[None, :]                       # (k, n_pts)
    unit = vec / L

    sc = shot_clock
    return dict(
        shot_clock=float(sc),
        backcourt=float(abs(P[0] - basket[0]) > 47),
        dist=L,
        passer_basket_dist=float(np.hypot(*(P - basket))),
        ball_end_basket_dist=float(np.hypot(*(T - basket))),
        basket_angle=_angle(P, basket, T),            # angle AT the basket between the passer and the receiver
        closest_def_dist_passer=float(d_p[ip]),
        closest_def_dist_ball_end=float(d_t[it]),
        closest_def_trajectory=float(lane_d[il]),
        closest_def_angle_passer=_angle(T, P, D[ip]),
        closest_def_angle_ball_end=_angle(P, T, D[it]),
        closest_def_trajectory_angle=_angle(T, P, D[il]),
        lane_n_def_5ft=float((lane_d <= 5).sum()),
        passer_speed=float(np.hypot(*p_vel)),
        target_speed=float(np.hypot(*t_vel)),
        intercept_margin=float(margin.min()),
        n_def_can_intercept=float((margin.min(axis=1) < 0).sum()),
        second_def_dist_passer=float(sp[1]) if len(sp) > 1 else float(sp[0]),
        second_def_dist_ball_end=float(st[1]) if len(st) > 1 else float(st[0]),
        recv_speed_along=float(t_vel @ unit),
        passer_speed_toward_recv=float(p_vel @ unit),
    )


SHOT_FEATURE_NAMES = ['shot_dist', 'shot_angle', 'def1', 'def2', 'def_cos', 'shooter_speed', 'shot_clock']


def shot_features(S, D, basket, vel=(0.0, 0.0), shot_clock=np.nan):
    """Features of a (hypothetical or real) shot from position S: distance and angle to the basket, the two nearest
    defenders, whether the nearest one is between the shooter and the basket (cosine), shooter speed, shot clock."""
    S, D, basket = np.asarray(S, float), np.asarray(D, float).reshape(-1, 2), np.asarray(basket, float)
    if len(D) == 0 or np.isnan(S).any() or np.isnan(D).any():
        return None
    to_b = basket - S
    dist = float(np.hypot(*to_b))
    dd = np.hypot(*(D - S).T)
    o = np.argsort(dd)
    dn = D[o[0]] - S
    cos = float((dn @ to_b) / (np.hypot(*dn) * dist)) if dist > 0 and np.hypot(*dn) > 0 else np.nan
    return dict(shot_dist=dist, shot_angle=float(np.degrees(np.arctan2(abs(to_b[1]), abs(to_b[0])))),
                def1=float(dd[o[0]]), def2=float(dd[o[1]]) if len(dd) > 1 else float(dd[o[0]]),
                def_cos=cos, shooter_speed=float(np.hypot(*np.asarray(vel, float))), shot_clock=float(shot_clock))
