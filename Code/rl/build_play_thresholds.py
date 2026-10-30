"""Fit the per-cluster median distance thresholds used by play_reward.py on the 320 fitting games, and check descriptor consistency.
    cd Code && python rl/build_play_thresholds.py --data ../data/rl
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
from load import load_games              # noqa: E402
from plays import possessions, describe  # noqa: E402
from play_reward import PlayLibrary, describe_seq, LIB_PATH   # noqa: E402
from train_executor import fit_val_games   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
a = ap.parse_args()
lib = PlayLibrary()
fit_g, _ = fit_val_games()
G = load_games(a.data, fit_g)
Xa, Xb = [], []
for d in G:
    for p, idx in possessions(d):
        Xa.append(describe(d, idx))
        Xb.append(describe_seq(d['pos'][idx], d['ball'][idx, :2], d['t_front'][idx]))
Xa, Xb = np.array(Xa), np.array(Xb)
ka, da = lib.assign(Xa)
kb, db = lib.assign(Xb)
print(f'{len(Xa)} fit possessions; same cluster with dataset velocities vs finite-difference speeds: {(ka == kb).mean():.3f}')
thr = np.array([np.median(db[kb == k]) if (kb == k).sum() else np.inf for k in range(lib.K)])
print('median in-cluster distance per cluster:', np.round(thr, 2))
z = dict(np.load(LIB_PATH))
z['thresholds'] = thr
np.savez_compressed(LIB_PATH, **z)
# base rate: share of real possessions (>= 6 s) that count as a clean play, by finite-difference descriptors
print('share of fit possessions within threshold (should be ~0.5):', float((db <= thr[kb]).mean()))
