"""Build model/play_protos.npz (prototype grids per play) and assign every fitting possession to a play.
    cd Code && python rl/build_play_protos.py --data ../data/rl
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
from load import load_games              # noqa: E402
from plays import possessions            # noqa: E402
from play_reward import PlayLibrary, describe_seq   # noqa: E402
from play_cond import PROTO_PATH         # noqa: E402
from train_executor import fit_val_games   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
a = ap.parse_args()
lib = PlayLibrary()
fit_g, _ = fit_val_games()
X = []
for d in load_games(a.data, fit_g):
    for p, idx in possessions(d):
        X.append(describe_seq(d['pos'][idx], d['ball'][idx, :2], d['t_front'][idx]))
X = np.array(X)
k, dist = lib.assign(X)
grid = np.zeros((lib.K, 7, 30))
for c in range(lib.K):
    m = k == c
    for s in range(7):
        grid[c, s] = X[m][:, s * 33:s * 33 + 30].mean(0)
freq = np.bincount(k, minlength=lib.K) / len(k)
# partial-follow thresholds: for n slices observed (1..7), the median over member possessions of the mean (over those slices) cell-count L1 distance /2 to the prototype
thr = np.zeros((lib.K, 7))
for c in range(lib.K):
    m = np.flatnonzero(k == c)
    dist = np.stack([0.5 * np.abs(X[m][:, s * 33:s * 33 + 30] - grid[c, s]).sum(1) for s in range(7)], axis=1)      # (members, 7)
    for ns in range(1, 8):
        thr[c, ns - 1] = np.median(dist[:, :ns].mean(1))
np.savez_compressed(PROTO_PATH, grid=grid, freq=freq, thr=thr)
print(len(X), 'fit possessions; play shares', np.round(freq, 3))
print('prototype mass per slice (should be 5):', grid[0].sum(1).round(2))
