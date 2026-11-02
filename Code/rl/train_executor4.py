"""Executor v4g: the goal-conditioned executor (v3g) trained on ALL live states (back court and transition included) with a front-court flag in the
context, so the game engine can use one movement model through a whole possession and across possession changes.

    cd Code && python rl/train_executor4.py --data ../data/rl --out ../reports/rl-stage6a/exec4 --save rl/model/executor_v4g.keras

Goals (hindsight, as in v3g) are given only for front-court window samples (t_front < 6 s), 30% dropped. Evaluation: one-step error on validation games,
split into front court (compared with v2 / v3g) and back court / transition, with goals switched off.
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
import executor_v2 as E                  # noqa: E402
from executor_data import build, F_DIM, G_DIM   # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from play_cond import TOK_EXTRA          # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from train_executor2 import per_sample, summarise   # noqa: E402
from plays import X_EDGES, Y_EDGES, GRID   # noqa: E402

CTX4 = 3


def log(*a):
    print(*a, flush=True)


def goal_features4(S, games, seed, drop=0.3, goals=True):
    N = len(S['y'])
    pos = S['tokens'][:, :, 0:2] * np.array([47.0, 25.0], np.float32) + np.array([47.0, 25.0], np.float32)
    tf = S['ctx'][:, 4] * 24.0
    tok = np.zeros((N, 10, TOK_EXTRA), np.float32)
    ctx = np.zeros((N, CTX4), np.float32)
    rng = np.random.default_rng(seed + 11)
    for gi, d in enumerate(games):
        m0 = np.flatnonzero(S['game_idx'] == gi)
        if len(m0) == 0:
            continue
        fcf = d['frontcourt'][S['state'][m0]].astype(np.float32)
        ctx[m0, 2] = fcf
        if not goals:
            continue
        m = m0[(fcf > 0) & (tf[m0] < 6.0)]
        if len(m) == 0:
            continue
        st = S['state'][m]
        st_c = np.minimum(st, len(d['frame']) - 16)
        ok = (st + 15 < len(d['frame']))
        ok &= (d['frame'][st_c + 15] - d['frame'][st_c] == 75) & (d['poss_id'][st_c + 15] == d['poss_id'][st_c])
        ok &= rng.random(len(m)) >= drop
        for h, k in enumerate((5, 15)):
            fp = d['pos'][st_c + k][:, :5]
            ix = np.clip(np.searchsorted(X_EDGES, fp[..., 0], side='right') - 1, 0, GRID[0] - 1)
            iy = np.clip(np.searchsorted(Y_EDGES, fp[..., 1], side='right') - 1, 0, GRID[1] - 1)
            cx, cy = (X_EDGES[ix] + X_EDGES[ix + 1]) / 2, (Y_EDGES[iy] + Y_EDGES[iy + 1]) / 2
            tok[m[ok], :5, 2 * h] = (cx - pos[m, :5, 0])[ok] / 20.0
            tok[m[ok], :5, 2 * h + 1] = (cy - pos[m, :5, 1])[ok] / 20.0
        ctx[m[ok], 0] = 1.0
        ctx[m[ok], 1] = (6.0 - tf[m[ok]]) / 6.0
    return tok, ctx


def attach(S, games, seed, goals=True):
    tok, ctx = goal_features4(S, games, seed, goals=goals)
    fc = ctx[:, 2] > 0
    return dict(S, tokens=np.concatenate([S['tokens'], tok], axis=2), ctx=np.concatenate([S['ctx'], ctx], axis=1), fc=fc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--max-n', type=int, default=400000)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--n-games', type=int, default=0)
    a = ap.parse_args()
    fit_g, val_g = fit_val_games()
    if a.n_games:
        fit_g, val_g = fit_g[:a.n_games], val_g[:max(4, a.n_games // 4)]
    t0 = time.time()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    S0 = build(FIT, table, 3, a.max_n, 0, 'real', mask_mode='all_live')
    V0 = build(VAL, table, 3, 150000, 1, 'real', mask_mode='all_live')
    log(f'data ready ({time.time() - t0:.0f}s): {len(S0["y"])} fit, {len(V0["y"])} val samples')
    S = attach(S0, FIT, 0)
    V_nogoal = attach(V0, VAL, 1, goals=False)
    keras.utils.set_random_seed(0)
    model = E.build_model(F_DIM + TOK_EXTRA, G_DIM + CTX4, k_modes=3)
    n_steps = (len(S['y']) // 512) * a.epochs
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(2e-3, n_steps, alpha=0.05), clipnorm=1.0),
                  loss={'mix': E.mix_nll, 'dec': 'sparse_categorical_crossentropy', 'tgt': E.tgt_loss}, loss_weights={'mix': 1.0, 'dec': 0.3, 'tgt': 0.3})
    sw = {'mix': np.ones(len(S['y']), 'float32'), 'dec': (S['action'] >= 0).astype('float32'), 'tgt': (S['target_slot'] >= 0).astype('float32')}
    model.fit([S['tokens'], S['ctx']], {'mix': S['y'], 'dec': np.maximum(S['action'], 0), 'tgt': np.maximum(S['target_slot'], 0)}, sample_weight=sw, batch_size=512,
              epochs=a.epochs, verbose=2)
    ps = per_sample(model, V_nogoal)
    fc = V_nogoal['fc']
    res = dict(n_fit=int(len(S['y'])), n_val=int(len(fc)), share_frontcourt=float(fc.mean()))
    for name, m in (('frontcourt', fc), ('backcourt_transition', ~fc)):
        res[name] = {k: float(np.nanmean(v[m])) for k, v in ps.items() if k not in ('game_idx',) and k in ('nll_off', 'nll_def', 'ade1_off', 'ade1_def', 'dec_ll', 'tgt_ll')}
    log(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'exec4_eval.json'), 'w'), indent=1)
    model.save(a.save)


if __name__ == '__main__':
    main()
