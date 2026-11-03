"""Diagnose the rim gap: where do holders and offensive players spend time (real vs engine), and how often does a state near the rim end in a shot."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import keras, executor_v2 as E
from load import load_games
from profiles import build_profiles
from splits import games as split_games
from train_executor import fit_val_games
from play_cond import PlayTargets
from game_engine import Engine
from engine_validate import quarter_starts
from api import PassValueModel
BASKET = np.array([5.35, 25.0])
cal = json.load(open(os.path.join(HERE, 'model', 'engine_calibration.json')))
fit_g, _ = fit_val_games()
table = build_profiles(load_games('../data/rl', fit_g))
G = load_games('../data/rl', split_games('holdout'))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
qs = quarter_starts(G); rng = np.random.default_rng(0); qs = [qs[i] for i in np.sort(rng.choice(len(qs), n, replace=False))]
# real
hd, od = [], []
for d in G:
    m = d['live'] & (d['holder'] != 0) & (d['pos'][:, :5, 0].mean(1) < 47)
    for k in np.flatnonzero(m):
        ids = d['ids'][k][:5]; j = np.flatnonzero(ids == d['holder'][k])
        if len(j):
            hd.append(np.hypot(*(d['pos'][k, j[0]] - BASKET)))
            od.append(np.hypot(*(d['pos'][k, :5] - BASKET).T))
hd, od = np.array(hd), np.concatenate(od)
model = keras.saving.load_model('rl/model/executor_v4g.keras', compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
eng = Engine(model, PassValueModel(), plays=PlayTargets(), seed=0, **{k: v for k, v in cal.items() if k != 'pass_scale'}, pass_scale=cal['pass_scale']).reset(G, table, [(g, k) for g, k, _ in qs])
eng.run()
T = np.stack(eng.traj); H = np.stack(eng.holder_traj); O = np.stack(eng.order_traj); IDS = eng.R_ids
shd, sod = [], []
for t in range(len(T)):
    for i in range(eng.N):
        h = H[t, i]
        if h == 0: continue
        ids = IDS[i, O[t, i]]
        j = np.flatnonzero(ids[:5] == h)
        if not len(j) or T[t, i, :5, 0].mean() >= 47: continue
        shd.append(np.hypot(*(T[t, i, j[0]] - BASKET))); sod.append(np.hypot(*(T[t, i, :5] - BASKET).T))
shd, sod = np.array(shd), np.concatenate(sod)
bins = [0, 4, 6, 10, 15, 20, 25, 30, 94]
for nm, a, b in (('holder', hd, shd), ('all offence', od, sod)):
    print(nm, 'real', np.round(np.histogram(a, bins)[0] / len(a), 3).tolist(), '\n   sim ', np.round(np.histogram(b, bins)[0] / len(b), 3).tolist())
print('real holder states', len(hd), 'sim', len(shd))
import pandas as pd
sl = pd.DataFrame(eng.shot_log); sl.to_csv('/tmp/claude-0/-home-user/bb835e0e-1ba5-530d-a467-365f61c7b413/scratchpad/shot_log_base.csv', index=False)
z = np.load('rl/model/shot_table.npz'); cnt = z['cnt']
from shot_audit import D_EDGES, C_EDGES, bin_index
bi, bj, _ = bin_index(sl.dist.to_numpy(), 24 - sl.clock.to_numpy(), sl.def1.to_numpy())
sim = np.zeros((8, 4)); np.add.at(sim, (bi, bj), 1)
real = cnt.sum(2)
print('dist bins', D_EDGES.tolist(), ' elapsed bins', C_EDGES.tolist())
print('real share (rows dist, cols elapsed)\n', np.round(real / real.sum(), 3)); print('sim\n', np.round(sim / sim.sum(), 3))
