"""Play-run reward: did the simulated offence run one of the discovered plays (Stage 3 library) cleanly, and did it end in a good shot?

A possession "runs play k" when the 231-number formation descriptor of its first 6 s after midcourt (plays.py: grid counts of the five attackers,
ball, mean speed at 1-s slices) is assigned to cluster k and lies within the cluster's median in-cluster distance (PCA space) of its centre
(`thresholds` in model/play_library.npz, fitted on the 320 fitting games). The possession must last at least 6 s after midcourt for the
descriptor to exist. "Successfully" = the possession ends in a shot whose xPts >= SUCCESS_XPTS (no turnover, no violation).
Bonus = LAMBDA when both hold. Speeds in the descriptor are finite differences of consecutive 5 Hz states (the library was built with the dataset's
smoothed velocities; `build_play_thresholds.py` reports how often the two give the same cluster).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
from plays import grid_counts, SLICES    # noqa: E402

LIB_PATH = os.path.join(HERE, 'model', 'play_library.npz')
LAMBDA = 0.25
SUCCESS_XPTS = 1.0
WINDOW_S = 6.0


def describe_seq(pos, ball_xy, tf):
    """pos (T,10,2) attack-normalised (slots 0-4 offence), ball_xy (T,2), tf (T,) seconds since midcourt, increasing. -> (231,) descriptor."""
    t = tf - tf[0]
    sp = np.zeros(len(tf))
    d = np.hypot(*(pos[1:, :5] - pos[:-1, :5]).transpose(2, 0, 1)) / 0.2          # (T-1,5)
    sp[1:] = d.mean(1)
    f = []
    for s in range(SLICES):
        k = min(np.searchsorted(t, s), len(t) - 1)
        f.append(grid_counts(pos[k, :5]))
        f.append(ball_xy[k] / np.array([47.0, 25.0]))
        f.append(np.array([sp[k] / 10.0], np.float32))
    return np.concatenate([np.asarray(x, np.float32).ravel() for x in f])


class PlayLibrary:
    def __init__(self, path=LIB_PATH):
        z = np.load(path)
        self.mu, self.sd, self.comp, self.pmean, self.centers = z['mu'], z['sd'], z['pca_components'], z['pca_mean'], z['centers']
        self.thr = z['thresholds'] if 'thresholds' in z.files else None
        self.K = int(z['K'])

    def assign(self, X):
        """X (N,231) -> (cluster (N,), distance (N,))."""
        Z = ((X - self.mu) / self.sd - self.pmean) @ self.comp.T
        dist = np.sqrt(((Z[:, None, :] - self.centers[None]) ** 2).sum(-1))
        k = dist.argmin(1)
        return k, dist[np.arange(len(k)), k]

    def clean(self, X):
        k, d = self.assign(X)
        return k, d, d <= self.thr[k]
