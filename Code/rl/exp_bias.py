import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import keras, executor_v2 as E
from executor_data import build, Y_SCALE
from load import load_games
from profiles import build_profiles
from splits import games as split_games
from train_executor import fit_val_games
from train_executor2 import mixture_stats
from train_executor4 import attach
BASKET = np.array([5.35, 25.0])
fit_g, _ = fit_val_games()
table = build_profiles(load_games('../data/rl', fit_g))
G = load_games('../data/rl', split_games('holdout'))
S0 = build(G, table, 6, 150000, 0, 'real', mask_mode='all_live')
S = attach(S0, G, 0, goals=False)
model = keras.saving.load_model('rl/model/executor_v4g.keras', compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
o = model.predict([S['tokens'], S['ctx']], verbose=0, batch_size=4096)
w, mu, lv = mixture_stats(o['mix'])
mean = (w[..., None, None] * mu).sum(2) * Y_SCALE            # (N,10,5,2)
real = S['y'] * Y_SCALE
pos = S['tokens'][:, :, 0:2] * np.array([47.0, 25.0]) + np.array([47.0, 25.0])
hold = S['tokens'][:, :, 14 + 0 - 0] if False else None
# holder flag: token index of is_h: 2+2+2+2+2+5 = position(2) vs(2) vl(2) role(2) onehot(5) -> is_h at 13
ish = S['tokens'][:, :5, 13] > 0.5
fc = S['fc']
to_b = BASKET - pos[:, :5]; dist = np.hypot(*to_b.transpose(2, 0, 1)); rh = to_b / dist[..., None]
rad_real = (real[:, :5, 4] * rh).sum(-1); rad_pred = (mean[:, :5, 4] * rh).sum(-1)
sp_real = np.hypot(*real[:, :5, 4].transpose(2, 0, 1)); sp_pred = np.hypot(*mean[:, :5, 4].transpose(2, 0, 1))
bins = [0, 4, 8, 12, 16, 20, 25, 30, 40, 94]
for nm, m in (('holder', ish & fc[:, None]), ('off-ball', (~ish) & fc[:, None])):
    print(nm)
    for a, b in zip(bins[:-1], bins[1:]):
        k = m & (dist >= a) & (dist < b)
        if k.sum() < 50: continue
        print(f'  {a:2d}-{b:2d} ft n={k.sum():6d}  radial toward basket in 1s: real {rad_real[k].mean():+.2f} pred-mean {rad_pred[k].mean():+.2f}   |disp| real {sp_real[k].mean():.2f} pred-mean {sp_pred[k].mean():.2f}')
