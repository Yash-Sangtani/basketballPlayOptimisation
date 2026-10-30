"""Stage 4c realism check: the simulator with play-conditioned executors (plays drawn from the real play shares) against real possessions.
    cd Code && python rl/sim_validate3.py --data ../data/rl --model rl/model/executor_v3g.keras --out ../reports/rl-stage4c/sim_val --split val
Same five statistics, start states and calibration constants (0.85 / 1.0 / 0.03) as Stage 4a; v2 without plays is run alongside for comparison."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
import executor_v2 as E                  # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from sim import Sim                      # noqa: E402
from sim_validate import find_starts, real_outcome, summarise   # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from train_caller import SIM             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--v2', default='rl/model/executor_v2.keras')
ap.add_argument('--out', required=True)
ap.add_argument('--split', default='val')
ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--grid', default='', help='comma list shoot:pass:loss for the v3 model (calibration on development games)')
a = ap.parse_args()
from api import PassValueModel
fit_g, val_g = fit_val_games()
table = build_profiles(load_games(a.data, fit_g))
G = load_games(a.data, val_g if a.split == 'val' else split_games('holdout'))
rng = np.random.default_rng(0)
starts = find_starts(G)
starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
real = [real_outcome(G[g], k) for g, k in starts]
R = summarise([r['shot'] for r in real], [r['t'] for r in real], [r['n_pass'] for r in real], [r['pts'] for r in real])
g = np.array([s[0] for s in starts])
ug = np.unique(g)
boots = []
for _ in range(300):
    ii = np.concatenate([np.flatnonzero(g == u) for u in rng.choice(ug, len(ug))])
    boots.append(summarise(*[[real[i][k] for i in ii] for k in ('shot', 't', 'n_pass', 'pts')]))
ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in R if k != 'n'}
diff, pt = PassValueModel(), PlayTargets()
out = {'real': R, 'real_ci95': ci}
configs = [('v2_no_plays', a.v2, None, (0.85, 1.0, 0.03)), ('v3_prior_plays', a.model, 'prior', (0.85, 1.0, 0.03))]
for c in [x for x in a.grid.split(',') if x]:
    ss, pp, ll = (float(t) for t in c.split(':'))
    configs.append((f'v3_prior_plays_{c}', a.model, 'prior', (ss, pp, ll)))
plays_draw = rng.choice(pt.K, size=len(starts), p=pt.freq / pt.freq.sum())
models = {}
for name, path, plays, (ss, pp, ll) in configs:
    if path not in models:
        models[path] = keras.saving.load_model(path, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    play = None if plays is None else plays_draw
    cfg = dict(SIM, shoot_scale=ss, pass_scale=pp, loss_rate=ll)
    sim = Sim(models[path], diff, seed=12345, play_targets=pt, **cfg).reset(G, table, starts, play=play)
    sim.run()
    S = summarise((sim.end == 1).astype(int), sim.shot_info[:, 3], sim.n_pass, np.nan_to_num(sim.shot_info[:, 1]))
    gap = np.array(sim.arrival_gap)
    S.update(p_turnover=float((sim.end == 2).mean()), dur_mean=float(sim.elapsed.mean()), arrival_gap=float(gap.mean()) if len(gap) else float('nan'))
    out[name] = S
    print(name, json.dumps({k: round(v, 3) for k, v in S.items()}), flush=True)
for k, v in ci.items():
    print(f'  {k}: real {R[k]:.3f} [{v[0]:.3f}, {v[1]:.3f}]  ' + '  '.join(f'{n[:22]}={out[n][k]:.3f}' for n in out if n not in ('real', 'real_ci95')))
os.makedirs(a.out, exist_ok=True)
json.dump(out, open(os.path.join(a.out, f'sim_validate3_{a.split}.json'), 'w'), indent=1)
