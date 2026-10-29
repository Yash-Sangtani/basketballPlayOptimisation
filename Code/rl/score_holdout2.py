"""Score the saved Stage 2b executor (real profiles, seed 0) ONCE on the 100 RL holdout games (profiles from the fitting games only).
    cd Code && python rl/score_holdout2.py --data ../data/rl --model rl/model/executor_v2.keras --out ../reports/rl-stage2b/holdout.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
import executor_v2 as E                  # noqa: E402
from executor_data import build          # noqa: E402
from load import load_games             # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games                 # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from train_executor2 import per_sample, summarise   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
fit_g, _ = fit_val_games()
table = build_profiles(load_games(a.data, fit_g))
H = load_games(a.data, games('holdout'))
model = keras.models.load_model(a.model, custom_objects=E.CUSTOM_OBJECTS, safe_mode=False)
S = build(H, table, 3, 120000, 1, 'real')
ps = per_sample(model, S)
out = summarise(ps)
out.update(n_games=len(H), n=int(len(S['y'])))
# game-clustered 95% intervals
g = ps['game_idx']
ug = np.unique(g)
rng = np.random.default_rng(0)
ci = {}
for k, v in ps.items():
    if k == 'game_idx':
        continue
    m = np.array([np.nanmean(v[g == u]) for u in ug])
    bs = [np.nanmean(m[rng.integers(0, len(m), len(m))]) for _ in range(500)]
    ci[k] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out['ci95'] = ci
json.dump(out, open(a.out, 'w'), indent=1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != 'ci95'}))
