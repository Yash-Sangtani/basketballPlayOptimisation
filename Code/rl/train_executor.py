"""Train and evaluate the executor on TRAIN games (a 320 / 81 fit / validation split of the 401 tuning games; the holdout and test
games are not read). Profiles are estimated from the 320 fitting games only.

    cd Code && python rl/train_executor.py --data ../data/rl --mode real --out ../reports/rl-stage2/real
modes: real (own profile) | none (position only: zero profile) | shuffled (another player's profile: a no-information control)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
from executor_data import build, F_DIM, G_DIM, Y_SCALE, FUT, DT   # noqa: E402
from executor_model import build_model, move_nll   # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402


def log(*a):
    print(*a, flush=True)


def fit_val_games(seed=0):
    g = split_games('train')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(g))
    n_val = 81
    val = [g[i] for i in perm[:n_val]]
    fit = [g[i] for i in perm[n_val:]]
    return sorted(fit), sorted(val)


def evaluate(model, S, batch=4096):
    mv, dc = [], []
    for i in range(0, len(S['y']), batch):
        o = model.predict([S['tokens'][i:i + batch], S['ctx'][i:i + batch]], verbose=0)
        mv.append(o['move'])
        dc.append(o['dec'])
    mv, dc = np.concatenate(mv), np.concatenate(dc)
    y = S['y']
    mu, lv = mv[..., :2], mv[..., 2:]
    err = np.hypot(*(np.moveaxis(mu - y, -1, 0))) * Y_SCALE                  # (N,10,5) ft
    out = {}
    cv = S['tokens'][..., 4:6] * 10.0                                         # 0.4 s backward velocity, ft/s
    steps = np.arange(1, FUT + 1) * DT
    cvd = np.hypot(*(np.moveaxis(cv[:, :, None, :] * steps[None, None, :, None] / Y_SCALE - y, -1, 0))) * Y_SCALE
    zero = np.hypot(*(np.moveaxis(y, -1, 0))) * Y_SCALE
    for name, sl in (('offence', slice(0, 5)), ('defence', slice(5, 10))):
        out[name] = dict(ade_model=err[:, sl].mean(axis=(0, 1)).round(3).tolist(), ade_const_vel=cvd[:, sl].mean(axis=(0, 1)).round(3).tolist(),
                         ade_zero=zero[:, sl].mean(axis=(0, 1)).round(3).tolist())
    nll = 0.5 * (lv + (y - mu) ** 2 * np.exp(-lv))
    out['nll'] = dict(offence=float(nll[:, :5].mean()), defence=float(nll[:, 5:].mean()))
    out['calibration_1s_offence'] = float(np.mean(((y - mu)[:, :5, -1] ** 2) / np.exp(lv[:, :5, -1])))     # ~1 if the variances are right
    a = S['action']
    m = a >= 0
    p = np.clip(dc[m][np.arange(m.sum()), a[m]], 1e-9, 1)
    prior = np.bincount(a[m], minlength=3) / m.sum()
    out['decision'] = dict(logloss=float(-np.log(p).mean()), prior_logloss=float(-np.log(prior[a[m]]).mean()),
                           accuracy=float((dc[m].argmax(1) == a[m]).mean()), n=int(m.sum()), prior=prior.round(3).tolist())
    return out, mv, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--mode', default='real', choices=['real', 'none', 'shuffled'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-n', type=int, default=600000)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--n-games', type=int, default=0, help='debug: use only the first N fit games')
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    keras.utils.set_random_seed(0)
    fit_g, val_g = fit_val_games()
    if a.n_games:
        fit_g, val_g = fit_g[:a.n_games], val_g[:max(5, a.n_games // 4)]
    t0 = time.time()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    log(f'{len(FIT)} fit / {len(VAL)} validation games; {len(table)} player profiles ({time.time() - t0:.0f}s)')
    S = build(FIT, table, a.stride, a.max_n, 0, a.mode)
    V = build(VAL, table, a.stride, 150000, 1, a.mode)
    log(f"samples: fit {len(S['y'])}, validation {len(V['y'])}; decision labels {(S['action'] >= 0).sum()} ({time.time() - t0:.0f}s)")
    model = build_model(F_DIM, G_DIM)
    log(f'model parameters: {model.count_params():,}')
    n_steps = (len(S['y']) // a.batch) * a.epochs
    lr = keras.optimizers.schedules.CosineDecay(2e-3, n_steps, alpha=0.05)
    model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0), loss={'move': move_nll, 'dec': 'sparse_categorical_crossentropy'},
                  loss_weights={'move': 1.0, 'dec': 0.3})
    w = (S['action'] >= 0).astype('float32')
    hist = model.fit([S['tokens'], S['ctx']], {'move': S['y'], 'dec': np.maximum(S['action'], 0)}, sample_weight={'move': np.ones(len(w), 'float32'), 'dec': w},
                     batch_size=a.batch, epochs=a.epochs, verbose=2,
                     validation_data=([V['tokens'], V['ctx']], {'move': V['y'], 'dec': np.maximum(V['action'], 0)},
                                      {'move': np.ones(len(V['y']), 'float32'), 'dec': (V['action'] >= 0).astype('float32')}))
    res, _, _ = evaluate(model, V)
    res.update(mode=a.mode, n_fit=int(len(S['y'])), n_val=int(len(V['y'])), epochs=a.epochs, params=int(model.count_params()),
               history={k: [float(x) for x in v] for k, v in hist.history.items()}, seconds=time.time() - t0)
    log(json.dumps({k: v for k, v in res.items() if k != 'history'}, indent=1))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    if a.save:
        os.makedirs(os.path.dirname(a.save), exist_ok=True)
        model.save(a.save)


if __name__ == '__main__':
    main()
