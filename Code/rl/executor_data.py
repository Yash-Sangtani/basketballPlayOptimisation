"""Training samples for the behaviour (executor) models: one sample = one half-court live state with its recent history and its future.

Per sample (all in the attack-normalised frame of dataset.py, offence = slots 0-4, defence = slots 5-9):
  tokens  (10, F)   per-player features: position, velocities over 0.2 s and 0.4 s (BACKWARD differences, so a rollout can recompute
                    them from its own past), role, position group, holder flag, ball and basket geometry, nearest opponent, profile (12)
  ctx     (G,)      ball height, shot clock (+ missing flag), time in possession, time since midcourt, period clock
  y       (10,5,2)  displacement of every player at +0.2 .. +1.0 s from the current position, in units of 3 ft
  action  scalar    ball handler's action over the next 2 s: 0 shoot, 1 pass, 2 keep (-1 if the holder is not an offensive slot)
Samples need an unbroken 1.4 s window (2 past + 5 future states, same possession, same ten players).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
from features import BASKET, roster_positions    # noqa: E402
from profiles import half_court_mask, profile_lookup, PROFILE_COLS   # noqa: E402

N_PROFILE = len(PROFILE_COLS)
HIST = 2
FUT = 5
Y_SCALE = 3.0
DT = 0.2
G_DIM = 6


def nearest_opp(p):
    """p (N,10,2) -> (N,10) distance of each player to the nearest player of the other side."""
    off, dfn = p[:, :5], p[:, 5:]
    dd = np.hypot(off[:, :, None, 0] - dfn[:, None, :, 0], off[:, :, None, 1] - dfn[:, None, :, 1])
    return np.concatenate([dd.min(axis=2), dd.min(axis=1)], axis=1)


def make_tokens(p0, p1, p2, ids, holder, ball, pos_code, prof):
    """Token features from the current positions p0 and the two previous states p1 (k-1), p2 (k-2).
    p*: (N,10,2) float; ids (N,10) int; holder (N,) int; ball (N,3); pos_code (N,10) int 0-4; prof (N,10,12) float or zeros."""
    N = len(p0)
    vs = (p0 - p1) / DT / 10.0
    vl = (p0 - p2) / (2 * DT) / 10.0
    role = np.zeros((N, 10, 2), np.float32)
    role[:, :5, 0] = 1
    role[:, 5:, 1] = 1
    oh = np.eye(5, dtype=np.float32)[pos_code]
    is_h = (ids == holder[:, None]).astype(np.float32)[..., None]
    brel = (ball[:, None, :2] - p0) / 20.0
    to_b = BASKET[None, None, :] - p0
    bdist = np.hypot(to_b[..., 0], to_b[..., 1])[..., None] / 47.0
    no = (nearest_opp(p0) / 10.0)[..., None]
    f = np.concatenate([((p0[..., 0:1] - 47.0) / 47.0), ((p0[..., 1:2] - 25.0) / 25.0), vs, vl, role, oh, is_h, brel, bdist,
                        to_b / np.array([47.0, 25.0]), no, prof], axis=-1)
    return f.astype(np.float32)


def make_ctx(d_ball_z, shot_clock, t_in_poss, t_front, qt):
    sc = np.nan_to_num(shot_clock, nan=24.0)
    miss = np.isnan(shot_clock).astype(np.float32)
    return np.stack([d_ball_z / 10.0, sc / 24.0, miss, t_in_poss / 24.0, t_front / 24.0, qt / 720.0], axis=1).astype(np.float32)


F_DIM = 2 + 2 + 2 + 2 + 5 + 1 + 2 + 1 + 2 + 1 + N_PROFILE


def pos_codes(d, ids):
    m = roster_positions(d['roster'])
    return np.vectorize(lambda p: m.get(int(p), 2))(ids).astype(np.int64)


def _has_event(ev, lo, hi):
    """ev sorted frames; True where some event lies in (lo, hi]."""
    if len(ev) == 0:
        return np.zeros(len(lo), bool)
    i = np.searchsorted(ev, lo, side='right')
    return (i < len(ev)) & (ev[np.minimum(i, len(ev) - 1)] <= hi)


def game_samples(d, table, stride=3, rng=None, profile_mode='real', perm=None):
    """Samples of one game. profile_mode: 'real' (the player's profile), 'none' (zeros), 'shuffled' (another player's profile: `perm`
    maps player id -> player id)."""
    n = len(d['frame'])
    fr = d['frame']
    step_ok = np.r_[np.diff(fr) == 5, False]                     # step_ok[i]: state i -> i+1 is contiguous
    c = np.r_[0, np.cumsum(step_ok.astype(int))]                 # c[j] = number of ok steps among 0..j-1
    ks = np.arange(HIST, n - FUT)
    run_ok = (c[ks + FUT] - c[ks - HIST]) == (HIST + FUT)
    same_poss = d['poss_id'][ks - HIST] == d['poss_id'][ks + FUT]
    same_ids = (d['ids'][ks - HIST] == d['ids'][ks + FUT]).all(axis=1) & (d['ids'][ks] == d['ids'][ks + FUT]).all(axis=1)
    hc = half_court_mask(d)[ks]
    sel = ks[run_ok & same_poss & same_ids & hc]
    sel = sel[::stride]
    if len(sel) == 0:
        return None
    if rng is not None and stride > 1:
        sel = sel + rng.integers(0, 1, len(sel))                 # (offset kept deterministic; stride already thins)
    ids = d['ids'][sel]
    ids_prof = ids if profile_mode == 'real' else (np.vectorize(lambda p: perm.get(int(p), int(p)))(ids) if profile_mode == 'shuffled' else None)
    prof = np.zeros(ids.shape + (N_PROFILE,), np.float32) if ids_prof is None else profile_lookup(table, ids_prof)
    p0, p1, p2 = d['pos'][sel], d['pos'][sel - 1], d['pos'][sel - 2]
    tok = make_tokens(p0, p1, p2, ids, d['holder'][sel], d['ball'][sel], pos_codes(d, ids), prof)
    ctx = make_ctx(d['ball'][sel, 2], d['shot_clock'][sel], d['t_in_poss'][sel], d['t_front'][sel], d['quarter_time'][sel])
    y = np.stack([(d['pos'][sel + s] - p0) / Y_SCALE for s in range(1, FUT + 1)], axis=2).astype(np.float32)   # (N,10,5,2)
    # decision label for holder states
    action = np.full(len(sel), -1, np.int8)
    hold = d['holder'][sel]
    in_off = (ids[:, :5] == hold[:, None]).any(axis=1)
    f0 = fr[sel]
    shot_by = {int(p): np.sort(g['frame'].to_numpy()) for p, g in d['shots'].groupby('shooter')} if len(d['shots']) else {}
    live = d['passes'][d['passes']['live'] == 1] if len(d['passes']) else d['passes']
    pass_by = {int(p): np.sort(g['frame'].to_numpy()) for p, g in live.groupby('passer')} if len(live) else {}
    target_slot = np.full(len(sel), -1, np.int8)
    tgt_by = {}
    if len(live):
        for p_, g in live.groupby('passer'):
            g = g.sort_values('frame')
            tgt_by[int(p_)] = (g['frame'].to_numpy(), g['actual_target'].to_numpy())
    for p in np.unique(hold[in_off]):
        m = in_off & (hold == p)
        shoot = _has_event(shot_by.get(int(p), np.array([])), f0[m], f0[m] + 50)
        passed = _has_event(pass_by.get(int(p), np.array([])), f0[m] - 5, f0[m] + 50)
        action[m] = np.where(shoot, 0, np.where(passed, 1, 2))
        if int(p) in tgt_by:                                      # who receives the first pass in the next 2 s
            pf, pt = tgt_by[int(p)]
            idx = np.flatnonzero(m)
            i = np.searchsorted(pf, f0[idx] - 5, side='right')
            have = (i < len(pf)) & (pf[np.minimum(i, len(pf) - 1)] <= f0[idx] + 50)
            for q, ok_ in zip(idx[have], i[have]):
                hit = np.flatnonzero(ids[q, :5] == pt[ok_])
                if len(hit):
                    target_slot[q] = hit[0]
    return dict(tokens=tok, ctx=ctx, y=y, action=action, target_slot=target_slot, game_idx=np.full(len(sel), -1, np.int32), state=sel.astype(np.int32))


def build(games, table, stride=3, max_n=None, seed=0, profile_mode='real', all_ids=None):
    rng = np.random.default_rng(seed)
    perm = None
    if profile_mode == 'shuffled':
        ids = np.array(sorted(table.index))
        perm = dict(zip(ids.tolist(), rng.permutation(ids).tolist()))
    parts = []
    for gi, d in enumerate(games):
        s = game_samples(d, table, stride, rng, profile_mode, perm)
        if s is not None:
            s['game_idx'][:] = gi
            parts.append(s)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if max_n is not None and len(out['y']) > max_n:
        sel = np.sort(rng.choice(len(out['y']), max_n, replace=False))
        out = {k: v[sel] for k, v in out.items()}
    return out
