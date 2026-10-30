"""Play conditioning for executors (v3): turn a chosen play (Stage-3 cluster) into per-player targets.

A play's prototype is the mean court-grid occupancy of its member possessions at 1-s slices 0..6 s after midcourt (model/play_protos.npz,
`grid` (K,7,30)). At time t_front < 6 s with play k active, for two horizons (about 1 s and 3 s ahead) the prototype's five most occupied cells
(greedy, subtracting one player per pick) are matched one-to-one to the five attackers by distance (Hungarian); each attacker gets the vector to his
cell. Token features (6): dx1, dy1, dx2, dy2 (ft / 20) and the two distances (ft / 20); zero for defenders and when no play is active.
Context features (2): active flag and time left in the window / 6.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
from scipy.optimize import linear_sum_assignment   # noqa: E402
from plays import X_EDGES, Y_EDGES, GRID, SLICES    # noqa: E402

PROTO_PATH = os.path.join(HERE, 'model', 'play_protos.npz')
TOK_EXTRA = 6
CTX_EXTRA = 2
WINDOW = 6.0
CELLS = np.array([[(X_EDGES[i] + X_EDGES[i + 1]) / 2, (Y_EDGES[j] + Y_EDGES[j + 1]) / 2] for i in range(GRID[0]) for j in range(GRID[1])])   # (30,2) in grid order


def top5_cells(g):
    g = g.astype(float).copy()
    out = []
    for _ in range(5):
        c = int(g.argmax())
        out.append(c)
        g[c] = max(g[c] - 1.0, 0.0) if g[c] > 1.0 else -1e9
    return CELLS[out]


class PlayTargets:
    def __init__(self, path=PROTO_PATH):
        z = np.load(path)
        self.grid = z['grid']                                  # (K,7,30)
        self.freq = z['freq']                                  # (K,) share of possessions
        self.thr = z['thr'] if 'thr' in z.files else None      # (K,7) partial-follow thresholds
        self.K = self.grid.shape[0]
        self.targets = np.stack([[top5_cells(self.grid[k, s]) for s in range(SLICES)] for k in range(self.K)])   # (K,7,5,2)

    def features(self, pos, t_front, play):
        """pos (N,10,2); t_front (N,); play (N,) int (-1 = none) -> tok_extra (N,10,6) float32, ctx_extra (N,2) float32."""
        N = len(pos)
        tok = np.zeros((N, 10, TOK_EXTRA), np.float32)
        ctx = np.zeros((N, CTX_EXTRA), np.float32)
        act = np.flatnonzero((play >= 0) & (t_front < WINDOW))
        for n in act:
            k = int(play[n])
            ctx[n] = [1.0, (WINDOW - t_front[n]) / WINDOW]
            for h, dt in enumerate((1.0, 3.0)):
                s = int(min(SLICES - 1, np.ceil(t_front[n] + dt)))
                tg = self.targets[k, s]
                cost = np.hypot(pos[n, :5, None, 0] - tg[None, :, 0], pos[n, :5, None, 1] - tg[None, :, 1])
                r, c = linear_sum_assignment(cost)
                d = tg[c] - pos[n, r]
                tok[n, r, 2 * h] = d[:, 0] / 20.0
                tok[n, r, 2 * h + 1] = d[:, 1] / 20.0
        return tok, ctx
