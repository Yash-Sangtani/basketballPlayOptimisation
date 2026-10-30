"""Stage 3: set-play discovery. Plays are not labelled in tracking data, so they are discovered and then TESTED.

A half-court possession (live, frontcourt, at least 6 s after crossing midcourt) is described over its first 6 s at 1 s steps by
  * where the five offensive players are, as counts on a 6 x 5 grid over the attacking half-court (permutation-invariant: it does not
    matter which player is which),
  * where the ball is, and the offence's mean speed.
That is 7 x (30 + 2 + 1) = 231 numbers, standardised and reduced by PCA, then clustered with k-means. The descriptor sees only the
first 6 s of the possession (outcomes and the later part are not in it).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402

SLICES = 7                # 0..6 s after midcourt
MIN_DUR = 6.0
X_EDGES = np.array([0, 8, 16, 24, 32, 40, 47.0])
Y_EDGES = np.array([0, 10, 20, 30, 40, 50.0])
GRID = (len(X_EDGES) - 1, len(Y_EDGES) - 1)
N_FEATS = SLICES * (GRID[0] * GRID[1] + 3)


def grid_counts(pos):
    """pos (5,2) -> (30,) counts of players per cell (positions outside the grid are clipped to the edge cells)."""
    ix = np.clip(np.searchsorted(X_EDGES, pos[:, 0], side='right') - 1, 0, GRID[0] - 1)
    iy = np.clip(np.searchsorted(Y_EDGES, pos[:, 1], side='right') - 1, 0, GRID[1] - 1)
    out = np.zeros(GRID[0] * GRID[1], np.float32)
    np.add.at(out, ix * GRID[1] + iy, 1.0)
    return out


def possessions(d):
    """List of (poss_id, state indices of the possession's half-court part) with at least MIN_DUR seconds."""
    hc = d['live'] & d['frontcourt'] & (d['t_front'] >= 0.0)
    out = []
    pid = d['poss_id']
    for p in np.unique(pid[hc]):
        idx = np.flatnonzero((pid == p) & hc)
        if len(idx) >= 30 and d['t_front'][idx[-1]] - d['t_front'][idx[0]] >= MIN_DUR:
            out.append((int(p), idx))
    return out


def describe(d, idx):
    """231-dim descriptor of one possession (idx: its half-court states, in order)."""
    tf = d['t_front'][idx] - d['t_front'][idx[0]]
    f = []
    for s in range(SLICES):
        k = idx[min(np.searchsorted(tf, s), len(idx) - 1)]
        f.append(grid_counts(d['pos'][k, :5]))
        f.append(d['ball'][k, :2] / np.array([47.0, 25.0]))
        f.append(np.array([np.hypot(*d['vel'][k, :5].T).mean() / 10.0], np.float32))
    return np.concatenate([np.asarray(x, np.float32).ravel() for x in f])


def outcome(d, idx):
    """Shots in the possession (matched events): count, points, three-point attempts, assisted makes. FG points only (no free throws or
    turnovers), so 'points' is a partial outcome."""
    lo, hi = d['frame'][idx[0]], d['frame'][idx[-1]] + 25
    sh = d['shots']
    if len(sh) == 0:
        return dict(n_shots=0, fg_pts=0.0, n_three=0, n_assisted=0, duration=float(d['t_front'][idx[-1]] - d['t_front'][idx[0]]))
    m = (sh['frame'] >= lo) & (sh['frame'] <= hi) & (sh['team'] == d['ctrl_team'][idx[0]])
    s = sh[m]
    return dict(n_shots=int(len(s)), fg_pts=float((s['made'] * np.where(s['is3'] == 1, 3, 2)).sum()), n_three=int(s['is3'].sum()),
                n_assisted=int(((s['made'] == 1) & (s['assisted'] == 1)).sum()), duration=float(d['t_front'][idx[-1]] - d['t_front'][idx[0]]))


def build_table(games):
    """-> X (N, 231) float32, meta list of dicts (game, poss_id, team, outcome fields, start state)."""
    X, meta = [], []
    for d in games:
        for p, idx in possessions(d):
            X.append(describe(d, idx))
            meta.append(dict(game=d['game'], poss_id=p, team=int(d['ctrl_team'][idx[0]]), start=int(idx[0]), n=int(len(idx)), **outcome(d, idx)))
    return np.array(X, np.float32), meta
