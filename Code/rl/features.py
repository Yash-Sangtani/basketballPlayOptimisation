"""Per-state features for the behaviour models (movement and ball-handler decisions).

All inputs come from one `dataset.py` game dict; everything is in the attack-normalised frame (offense attacks (5.35, 25)).
"""
import numpy as np

BASKET = np.array([5.35, 25.0])
POS_CODE = {'G': 0, 'G-F': 1, 'F-G': 1, 'F': 2, 'F-C': 3, 'C-F': 3, 'C': 4}


def roster_positions(roster):
    return {int(r.player): POS_CODE.get(str(r.position), 2) for r in roster.itertuples()}


def state_context(d, k):
    """Scalars shared by all rows of state k."""
    return dict(shot_clock=d['shot_clock'][k], quarter_time=d['quarter_time'][k], t_in_poss=d['t_in_poss'][k], t_front=d['t_front'][k],
                quarter=float(d['quarter'][k]), ball_x=d['ball'][k, 0], ball_y=d['ball'][k, 1], ball_z=d['ball'][k, 2])


def player_row(d, k, j):
    """Features of offensive slot j (0-4) at state k."""
    pos, vel = d['pos'][k], d['vel'][k]
    p, v = pos[j], vel[j]
    dfn = pos[5:]
    dd = np.hypot(*(dfn - p).T)
    o = np.argsort(dd)
    mates = np.delete(np.arange(5), j)
    dm = np.hypot(*(pos[mates] - p).T)
    to_b = BASKET - p
    holder = d['holder'][k]
    is_h = float(d['ids'][k, j] == holder)
    hi = np.flatnonzero(d['ids'][k, :5] == holder)
    hrel = (pos[hi[0]] - p) if len(hi) and not is_h else np.zeros(2)
    n_def6 = float((dd <= 6).sum())
    paint = lambda P: float(((P[:, 0] <= 19) & (np.abs(P[:, 1] - 25) <= 8)).sum())
    f = dict(x=p[0], y=p[1], vx=v[0], vy=v[1], speed=float(np.hypot(*v)), basket_dist=float(np.hypot(*to_b)),
             basket_angle=float(np.degrees(np.arctan2(abs(to_b[1]), max(to_b[0], 1e-3)))),
             is_holder=is_h, hold_dx=hrel[0], hold_dy=hrel[1], ball_rel_x=d['ball'][k, 0] - p[0], ball_rel_y=d['ball'][k, 1] - p[1],
             def1=float(dd[o[0]]), def1_dx=float(dfn[o[0]][0] - p[0]), def1_dy=float(dfn[o[0]][1] - p[1]), def2=float(dd[o[1]]),
             n_def6=n_def6, mate_min=float(dm.min()), mate_mean=float(dm.mean()), paint_off=paint(pos[:5]), paint_def=paint(dfn))
    f.update(state_context(d, k))
    return f


def holder_row(d, k):
    """Features of the ball handler's situation at state k (the holder must be an offensive slot)."""
    hi = np.flatnonzero(d['ids'][k, :5] == d['holder'][k])
    if len(hi) == 0:
        return None
    j = int(hi[0])
    f = player_row(d, k, j)
    pos = d['pos'][k]
    dfn = pos[5:]
    open_m = []
    for m in range(5):
        if m != j:
            open_m.append(float(np.hypot(*(dfn - pos[m]).T).min()))
    f.update(mate_open_max=max(open_m), mate_open_mean=float(np.mean(open_m)), n_mates_open6=float(sum(o >= 6 for o in open_m)))
    return j, f
