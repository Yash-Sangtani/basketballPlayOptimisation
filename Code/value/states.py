"""
State features and per-game extraction for the value function V(s) (WS5b).

`state_features` is the single function used for the 1 Hz samples AND for the pass
release / reception frames, so a pass's delta V is computed on exactly the same
feature definition the model was trained on.

Frames are oriented so the attacked basket is always at (5.35, 25): when the offense
attacks the basket at x = 88.65 the court is rotated 180 degrees (x -> 94 - x, y -> 50 - y).
Possessions come from possessions.py (pbp); frames are matched to them by
(period, clock, team): a frame is kept only if the clock is running, somebody holds the
ball, and the holder's team is the offense of a possession whose clock interval
(+- CLOCK_TOL s, the pbp resolution) contains the frame. That drops dead balls, shots in
flight, and frames misassigned across a possession boundary by a pbp/tracking clock offset.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

from possessions import build_possessions  # noqa: E402

BASKET = np.array([5.35, 25.0])
CLOCK_TOL = 1.0         # s; pbp clock resolution
RUN_K = 12              # frames either side used to decide the game clock is running (0.48 s)
RUN_MIN_DROP = 0.6      # s of clock that must elapse over 2*RUN_K frames (0.96 s) to count as running
SAMPLE_STEP = 25        # frames; 25 Hz tracking -> about 1 Hz
BOUNDARY_NEAR = 3.0     # s; a team mismatch farther than this from a possession boundary cannot be a clock offset

FEATURES = ['shot_clock', 'ball_dist', 'ball_angle', 'ball_xn', 'ball_yn',                      # baseline
            'ball_z', 'holder_speed', 'closest_def', 'second_def', 'n_def_6', 'n_def_10',
            'period', 'period_time_left', 'margin',
            'n_def_paint', 'n_off_paint', 'off_hull_area', 'off_mean_basket_dist', 'def_mean_basket_dist',
            'mate_open_max', 'n_mates_open6']
BASELINE = ['shot_clock', 'ball_dist', 'ball_angle', 'ball_xn', 'ball_yn']


def filled_shot_clock(moments):
    """Shot clock, filled with the period clock when the shot clock is off and <= 24 s remain."""
    sc = pd.to_numeric(moments['shot_clock'], errors='coerce').values.astype(float)
    qt = moments['quarter_time'].values.astype(float)
    return np.where(np.isnan(sc) & (qt <= 24), qt, sc)


def clock_running(fr):
    """True where the period clock ticks over +-RUN_K frames (gap-free), i.e. live ball."""
    n, qt, block = fr['n'], fr['qt'].astype(float), fr['block']
    run = np.zeros(n, dtype=bool)
    if n > 2 * RUN_K:
        lo = np.arange(0, n - 2 * RUN_K)
        hi = lo + 2 * RUN_K
        run[lo + RUN_K] = (block[lo] == block[hi]) & ((qt[lo] - qt[hi]) > RUN_MIN_DROP)
    return run


def segment_holders(fr, segments):
    """Per-frame holder id / team from the hold segments (>= 4 frames, short dropouts filled)."""
    pid = np.zeros(fr['n'], dtype=np.int64)
    team = np.zeros(fr['n'], dtype=np.int64)
    for s in segments:
        pid[s['start']:s['end'] + 1] = s['holder']
        team[s['start']:s['end'] + 1] = s['team']
    return pid, team


def _speed(fr, i, player, k=5):
    j = i - k
    if j < 0 or fr['block'][j] != fr['block'][i]:
        return np.nan
    a, b = np.flatnonzero(fr['pid'][j] == player), np.flatnonzero(fr['pid'][i] == player)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    dt = fr['ut'][i] - fr['ut'][j]
    return float(np.hypot(*(fr['xy'][i][b[0]] - fr['xy'][j][a[0]])) / dt) if dt > 0 else np.nan


def state_features(fr, i, holder, team, basket_x, shot_clock, margin):
    """Feature dict for frame i with `holder` (player id) holding the ball for offense `team`
    attacking the basket at x = basket_x. None if the frame lacks what is needed."""
    if not fr['has_ball'][i] or not np.isfinite(fr['ball'][i]).all():
        return None
    pid, tm = fr['pid'][i], fr['team'][i]
    xy, ball = fr['xy'][i].copy(), fr['ball'][i].copy()
    if basket_x > 47:
        xy = np.array([94.0, 50.0]) - xy
        ball[:2] = np.array([94.0, 50.0]) - ball[:2]
    hi = np.flatnonzero(pid == holder)
    off = (tm == team) & (pid != 0)
    dfn = (tm != team) & (pid != 0)
    if len(hi) == 0 or off.sum() < 5 or dfn.sum() < 5 or np.isnan(xy[off | dfn]).any():
        return None
    H = xy[hi[0]]
    D, O = xy[dfn], xy[off]
    dh = np.sort(np.hypot(*(D - H).T))
    mates = xy[off & (pid != holder)]
    open_m = np.array([np.hypot(*(D - m).T).min() for m in mates])
    try:
        hull = ConvexHull(O).volume      # area in 2-D
    except Exception:
        hull = np.nan
    bx, by = ball[0] - BASKET[0], ball[1] - BASKET[1]
    paint = lambda P: int(((P[:, 0] <= 19.0) & (np.abs(P[:, 1] - 25.0) <= 8.0)).sum())
    return dict(
        shot_clock=shot_clock,
        ball_dist=float(np.hypot(bx, by)), ball_angle=float(np.degrees(np.arctan2(abs(by), bx))),
        ball_xn=float(bx), ball_yn=float(by),
        ball_z=float(ball[2]), holder_speed=_speed(fr, i, holder),
        closest_def=float(dh[0]), second_def=float(dh[1]),
        n_def_6=int((dh <= 6).sum()), n_def_10=int((dh <= 10).sum()),
        period=float(min(int(fr['quarter'][i]), 5)), period_time_left=float(fr['qt'][i]), margin=float(margin),
        n_def_paint=paint(D), n_off_paint=paint(O), off_hull_area=float(hull),
        off_mean_basket_dist=float(np.hypot(*(O - BASKET).T).mean()),
        def_mean_basket_dist=float(np.hypot(*(D - BASKET).T).mean()),
        mate_open_max=float(open_m.max()) if len(open_m) else np.nan,
        n_mates_open6=int((open_m >= 6).sum()) if len(open_m) else 0)


class PossessionIndex:
    """Match (period, clock, team) to a row of the possession table."""

    def __init__(self, table):
        self.t = table.reset_index(drop=True)
        self.by_period = {p: g for p, g in self.t.groupby('period')}

    def locate(self, period, clock, team):
        """(poss_row or None, status, boundary_distance). status: ok | uncovered | team_mismatch."""
        g = self.by_period.get(int(period))
        if g is None:
            return None, 'uncovered', np.nan
        s, e = g.start_clock.values, g.end_clock.values
        cover = (e - CLOCK_TOL <= clock) & (clock <= s + CLOCK_TOL)
        if not cover.any():
            return None, 'uncovered', np.nan
        rows = g[cover]
        out = np.maximum(np.maximum(clock - rows.start_clock.values, rows.end_clock.values - clock), 0)
        bd = np.minimum(np.abs(clock - rows.start_clock.values), np.abs(clock - rows.end_clock.values))
        okm = (rows.offense_team.values == team)
        if not okm.any():
            return None, 'team_mismatch', float(bd.min())
        k = np.flatnonzero(okm)[np.argmin(out[okm])]
        return rows.iloc[k], 'ok', float(bd[k])


def extract_game(game, seed=0):
    """Run detection on one game and return dict(samples, pass_states, pass_meta, diag)."""
    import passtracking   # imported here: getting_data reads TRACKING_DATA_DIR at import time
    parts = game.split('.')
    date, away, home = '.'.join(parts[:3]), parts[3], parts[5]
    pt = passtracking.PassTracking(date, home, away)
    table, info = build_possessions(pt.pbp.copy(), game_id=int(pt.game_id))
    pt.track_passes()
    fr, m = pt._fr, pt.moments
    pidx = PossessionIndex(table)
    shot_clock = filled_shot_clock(m)
    run = clock_running(fr)
    seg_pid, seg_team = segment_holders(fr, pt.segments)
    basket_of = {}
    for (team, q), side in pt.attack_basket.items():
        basket_of[(int(team), int(q))] = pt.basket1[0] if side == 1 else pt.basket2[0]
    qt, quarter = fr['qt'].astype(float), fr['quarter']

    # ---- 1 Hz samples
    ok = fr['has_ball'] & (seg_pid != 0) & run
    cand = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    cand = cand[(cand % SAMPLE_STEP) == rng.integers(SAMPLE_STEP)]
    diag = dict(game=game, candidates=len(cand), uncovered=0, team_mismatch=0, mismatch_interior=0,
                matched=0, no_basket=0, no_features=0, kept=0, matched_boundary=0)
    rows = []
    for i in cand:
        team, q = int(seg_team[i]), int(quarter[i])
        prow, status, bd = pidx.locate(q, qt[i], team)
        if status == 'uncovered':
            diag['uncovered'] += 1
            continue
        if status == 'team_mismatch':
            diag['team_mismatch'] += 1
            diag['mismatch_interior'] += int(bd > BOUNDARY_NEAR)
            continue
        diag['matched'] += 1
        diag['matched_boundary'] += int(bd <= BOUNDARY_NEAR)
        bx = basket_of.get((team, q))
        if bx is None:
            diag['no_basket'] += 1
            continue
        f = state_features(fr, i, int(seg_pid[i]), team, bx, shot_clock[i], prow.off_margin_start)
        if f is None:
            diag['no_features'] += 1
            continue
        f.update(game=game, frame=int(i), poss_idx=int(prow.poss_idx), holder=int(seg_pid[i]),
                 pts=int(prow.points), end_reason=prow.end_reason, to_end=float(qt[i] - prow.end_clock),
                 poss_fga=int(prow.fga))
        rows.append(f)
        diag['kept'] += 1
    samples = pd.DataFrame(rows)

    # ---- completed passes: state at release and at reception, plus assist / next-shot labels
    pa = pt.pass_attempt
    comp = pa[(pa['pass_successful'] == 1)] if pa is not None and len(pa) else pa
    pbp = pt.pbp
    clock = pbp['Qtime'].values
    fga = pbp[pbp['EVENTMSGTYPE'].isin([1, 2])]
    assists = pbp[(pbp['EVENTMSGTYPE'] == 1) & pbp['PLAYER2_ID'].notna() & (pbp['PLAYER2_ID'] != 0)]
    states, metas = [], []
    claimed = set()
    diag.update(passes_completed=0 if comp is None else len(comp), pass_ok=0, pass_not_live=0, pass_unlocated=0,
                pass_poss_changed=0, pass_no_features=0, assists_pbp=len(assists), assists_matched=0)
    ordered = [] if comp is None else list(comp.itertuples())
    # assist matching over ALL completed passes (before validity filters), same window as validate_passes.py:
    # the credited passer -> scorer pass whose RECEPTION is -1..+8 s (clock) before the pbp make; the latest one wins
    assist_pass = {}
    for _, a in assists.iterrows():
        cands = []
        for p in ordered:
            gap = qt[int(p.recv_frame)] - a['Qtime']
            if (int(p.passer) == int(a['PLAYER2_ID']) and p.target == int(a['PLAYER1_ID']) and int(p.quarter) == int(a['PERIOD'])
                    and -1.0 <= gap <= 8.0 and p.frame not in claimed):
                cands.append((gap if gap >= 0 else 100 - gap, p.frame))
        if cands:
            best = min(cands)[1]
            claimed.add(best)
            assist_pass[best] = True
    diag['assists_matched'] = len(assist_pass)
    for pass_id, p in enumerate(ordered):
        r, v, q, team = int(p.frame), int(p.recv_frame), int(p.quarter), int(p.team_passer)
        if not (run[r] and run[v]):
            diag['pass_not_live'] += 1
            continue
        pr, s1, _ = pidx.locate(q, qt[r], team)
        pv, s2, _ = pidx.locate(q, qt[v], team)
        if pr is None or pv is None:
            diag['pass_unlocated'] += 1
            continue
        if pr.poss_idx != pv.poss_idx:
            diag['pass_poss_changed'] += 1
            continue
        bx = basket_of.get((team, q))
        if bx is None:
            diag['no_basket'] += 1
            continue
        f1 = state_features(fr, r, int(p.passer), team, bx, shot_clock[r], pr.off_margin_start)
        f2 = state_features(fr, v, int(p.target), team, bx, shot_clock[v], pr.off_margin_start)
        if f1 is None or f2 is None:
            diag['pass_no_features'] += 1
            continue
        # next field-goal attempt in this possession at or after the release (pbp clock, +2 s tolerance)
        in_p = fga[(fga['PERIOD'] == q) & (fga['Qtime'] <= pr.start_clock + CLOCK_TOL) & (fga['Qtime'] >= pr.end_clock - CLOCK_TOL)
                   & (fga['Qtime'] <= qt[r] + 2.0)]
        if len(in_p):
            nxt = in_p.loc[in_p['Qtime'].idxmax()]
            nxt_shooter, nxt_made, nxt_gap = int(nxt['PLAYER1_ID']), int(nxt['EVENTMSGTYPE'] == 1), float(qt[r] - nxt['Qtime'])
        else:
            nxt_shooter, nxt_made, nxt_gap = 0, -1, np.nan
        diag['pass_ok'] += 1
        for kind, f in (('release', f1), ('recv', f2)):
            f = dict(f, game=game, pass_id=pass_id, kind=kind)
            states.append(f)
        metas.append(dict(game=game, pass_id=pass_id, passer=int(p.passer), target=int(p.target), frame=r,
                          recv_frame=v, period=q, qt_release=float(qt[r]), flight_s=float(p.flight_s),
                          poss_idx=int(pr.poss_idx), poss_pts=int(pr.points), poss_end=pr.end_reason,
                          poss_fga=int(pr.fga), assist=bool(assist_pass.get(r, False)),
                          next_shooter=nxt_shooter, next_made=nxt_made, next_gap=nxt_gap,
                          receiver_shoots_next=bool(nxt_shooter == int(p.target) and 0 <= nxt_gap <= 8)))
    return dict(samples=samples, pass_states=pd.DataFrame(states), pass_meta=pd.DataFrame(metas), diag=diag,
                poss_diag=info['diag'])
