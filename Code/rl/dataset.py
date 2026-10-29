"""Stage 0 of the RL project: turn a tracked game into attack-normalised 5 Hz states, events and a roster.

One .npz per game (git-ignored, under data/rl/). Everything is derived from tracking; play-by-play is used only for the shot
and pass events that `difficulty/extract.py` already matches to tracking frames.

Conventions (the same as `value/states.py`):
  * court 94 x 50 ft; the controlling team (offense) is always attacking the basket at (5.35, 25): when it really attacks
    x = 88.65 the court is rotated 180 degrees (x -> 94 - x, y -> 50 - y), velocities rotate with it;
  * per frame, slots 0-4 are the offense and 5-9 the defense, each sorted by player id (so a lineup keeps its slots);
  * "controlling team" = the team of the most recent ball holder (a hold is >= 4 frames, short dropouts filled), carried
    forward through the flight; a possession id increments when control changes, the period changes or tracking has a gap.

    cd Code && python rl/dataset.py <game list> --out DIR      (resumable; raw data from $TRACKING_DATA_DIR)
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
sys.path.insert(0, os.path.join(HERE, '..', 'value'))
os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import passtracking as P                # noqa: E402
import extract as X                     # noqa: E402  (difficulty/extract.py: DiffPT with pass and shot events)
from states import clock_running, filled_shot_clock, segment_holders   # noqa: E402

STEP = 5                 # frames between states: 25 Hz -> 5 Hz
VEL_K = 5                # velocity = (x[i+K] - x[i-K]) / (2 K / 25 s), 0.4 s central difference
HZ = 25.0 / STEP
BASKET_X = 5.35
MIN_RUN_S = 2.0           # a control run shorter than this, between two runs of the same team, is a deflection / tip, not a possession
MAX_SPEED = 30.0          # ft/s; faster than any player can run = tracking glitch, the state is dropped
SCHEMA = 1


class RLPT(X.DiffPT):
    """DiffPT that also keeps the roster (names, positions) that get_data reads from the tracking JSON."""

    def __init__(self, date, home, away):
        orig = P.get_data
        holder = {}

        def spy(*a, **k):
            d = orig(*a, **k)
            holder['d'] = d
            return d
        P.get_data = spy
        try:
            super().__init__(date, home, away)
        finally:
            P.get_data = orig
        ev0 = holder['d'].tracking_data['events'][0]
        rows = []
        for side, tid in (('home', self.home_id), ('visitor', self.away_id)):
            for p in ev0[side]['players']:
                rows.append(dict(player=int(p['playerid']), team=int(tid), first=p['firstname'], last=p['lastname'],
                                 position=p['position'], jersey=p['jersey']))
        self.roster = pd.DataFrame(rows)


def merge_short_runs(ctrl, ut, quarter, min_s=MIN_RUN_S):
    """Absorb control runs shorter than `min_s` seconds that sit between two runs of the same team inside one period."""
    ctrl = ctrl.copy()
    while True:
        change = np.flatnonzero((ctrl[1:] != ctrl[:-1]) | (quarter[1:] != quarter[:-1])) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(ctrl)] - 1
        merged = False
        for j in range(1, len(starts) - 1):
            a, b = starts[j], ends[j]
            if (ut[b] - ut[a] < min_s and ctrl[starts[j - 1]] == ctrl[starts[j + 1]] != ctrl[a]
                    and quarter[starts[j - 1]] == quarter[a] == quarter[starts[j + 1]] and ctrl[a] != 0 and ctrl[starts[j - 1]] != 0):
                ctrl[a:b + 1] = ctrl[starts[j - 1]]
                merged = True
                break
        if not merged:
            return ctrl


def build_game(game):
    date, away, home = X.split(game)
    pt = RLPT(date, home, away)
    pt.build()
    pt.build_shots()
    fr, segs, m = pt._fr, pt.segments, pt.moments
    n = fr['n']
    seg_pid, seg_team = segment_holders(fr, segs)
    run = clock_running(fr)
    sc = filled_shot_clock(m)

    # controlling team: last holder's team, carried forward (reset at gaps / new period)
    ctrl = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(n):
        if i == 0 or fr['block'][i] != fr['block'][i - 1]:
            cur = 0
        if seg_team[i] != 0:
            cur = int(seg_team[i])
        ctrl[i] = cur
    ctrl = merge_short_runs(ctrl, fr['ut'], fr['quarter'])
    new_poss = np.zeros(n, dtype=bool)
    new_poss[1:] = (ctrl[1:] != ctrl[:-1]) | (fr['quarter'][1:] != fr['quarter'][:-1]) | (fr['block'][1:] != fr['block'][:-1])
    poss_id = np.cumsum(new_poss)

    idx = np.arange(VEL_K, n - VEL_K, STEP)
    idx = idx[fr['has_ball'][idx] & (ctrl[idx] != 0)]
    N = len(idx)
    pos = np.full((N, 10, 2), np.nan, np.float32)
    vel = np.zeros((N, 10, 2), np.float32)
    ids = np.zeros((N, 10), np.int64)
    ball = np.zeros((N, 3), np.float32)
    flip = np.zeros(N, bool)
    vel_ok = np.zeros(N, bool)
    for k, i in enumerate(idx):
        pid, tm, xy = fr['pid'][i], fr['team'][i], fr['xy'][i]
        off = np.flatnonzero((tm == ctrl[i]) & (pid != 0))
        dfn = np.flatnonzero((tm != ctrl[i]) & (pid != 0))
        if len(off) != 5 or len(dfn) != 5:
            continue
        off = off[np.argsort(pid[off])]
        dfn = dfn[np.argsort(pid[dfn])]
        order = np.r_[off, dfn]
        basket = pt._attack_basket(int(ctrl[i]), int(fr['quarter'][i]))
        f = basket is not None and basket[0] > 47
        flip[k] = f
        ids[k] = pid[order]
        p0 = xy[order]
        b = fr['ball'][i].copy()
        a, c = i - VEL_K, i + VEL_K
        if fr['block'][a] == fr['block'][i] == fr['block'][c]:
            ia = {int(q): j for j, q in enumerate(fr['pid'][a])}
            ic = {int(q): j for j, q in enumerate(fr['pid'][c])}
            if all(int(q) in ia and int(q) in ic for q in pid[order]):
                pa = np.array([fr['xy'][a][ia[int(q)]] for q in pid[order]])
                pc = np.array([fr['xy'][c][ic[int(q)]] for q in pid[order]])
                vel[k] = (pc - pa) / (2 * VEL_K / 25.0)
                vel_ok[k] = True
        if f:
            p0 = np.array([94.0, 50.0]) - p0
            vel[k] = -vel[k]
            b[:2] = np.array([94.0, 50.0]) - b[:2]
        pos[k], ball[k] = p0, b
    good = ~np.isnan(pos).any(axis=(1, 2)) & vel_ok & (np.hypot(vel[..., 0], vel[..., 1]).max(axis=1) <= MAX_SPEED)
    idx, pos, vel, ids, ball, flip = idx[good], pos[good], vel[good], ids[good], ball[good], flip[good]

    ut = fr['ut']
    pid_of = poss_id[idx]
    t_start = {}
    for k, i in enumerate(idx):
        t_start.setdefault(int(pid_of[k]), ut[i])
    t_in_poss = np.array([ut[i] - t_start[int(pid_of[k])] for k, i in enumerate(idx)], np.float32)
    frontcourt = ball[:, 0] < 47
    t_front = np.zeros(len(idx), np.float32)
    first_front = {}
    for k, i in enumerate(idx):
        if frontcourt[k]:
            first_front.setdefault(int(pid_of[k]), ut[i])
            t_front[k] = ut[i] - first_front[int(pid_of[k])]
    holder = seg_pid[idx]
    out = dict(
        schema=np.int64(SCHEMA), frame=idx.astype(np.int64), quarter=fr['quarter'][idx].astype(np.int8),
        quarter_time=fr['qt'][idx].astype(np.float32), shot_clock=sc[idx].astype(np.float32), live=run[idx],
        ctrl_team=ctrl[idx], poss_id=pid_of.astype(np.int64), t_in_poss=t_in_poss, frontcourt=frontcourt, t_front=t_front,
        holder=holder.astype(np.int64), pos=pos, vel=vel, ids=ids, ball=ball, flipped=flip)
    # events
    passes = pt.pass_rows[['frame', 'passer', 'actual_target', 'heading_target', 'label', 'outcome', 'live', 'team']].copy() if len(pt.pass_rows) else pd.DataFrame()
    shots = pt.shot_rows[['frame', 'shooter', 'made', 'is3', 'team', 'assisted', 'assister', 'quarter']].copy() if len(pt.shot_rows) else pd.DataFrame()
    rep = dict(game=game, game_id=int(pt.game_id), n_states=int(len(idx)), n_dropped=int((~good).sum()), n_poss=int(len(np.unique(pid_of))),
               n_passes=len(passes), n_shots=len(shots), live_share=float(out['live'].mean()), code=P.CODE_VERSION)
    return out, passes, shots, pt.roster, rep


def save_game(game, out_dir):
    t0 = time.perf_counter()
    arrays, passes, shots, roster, rep = build_game(game)
    rep['seconds'] = time.perf_counter() - t0
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{game}.npz')
    np.savez_compressed(path + '.tmp.npz', **arrays, passes_json=np.array(passes.to_json(orient='split')),
                        shots_json=np.array(shots.to_json(orient='split')), roster_json=np.array(roster.to_json(orient='split')),
                        report_json=np.array(json.dumps(rep)))
    os.replace(path + '.tmp.npz', path)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game_list')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    games = [l.strip().replace('.7z', '') for l in open(a.game_list) if l.strip()]
    todo = [g for g in games if not os.path.exists(os.path.join(a.out, f'{g}.npz'))]
    print(f'{len(games)} games, {len(todo)} to do', flush=True)
    for k, g in enumerate(todo):
        try:
            r = save_game(g, a.out)
            print(f"[{k + 1}/{len(todo)}] {g}: {r['n_states']} states, {r['n_poss']} possessions, {r['n_passes']} passes, {r['n_shots']} shots, "
                  f"{r['seconds']:.1f}s", flush=True)
        except Exception as e:
            print(f'[{k + 1}/{len(todo)}] {g}: FAILED {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
