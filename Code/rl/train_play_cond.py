"""Stage 3 test: does knowing the play (cluster id of the possession's first 6 s) lower the executor's movement error afterwards?

    cd Code && python rl/train_play_cond.py --data ../data/rl --labels ../reports/rl-stage3/labels.npz --out ../reports/rl-stage3/cond

Leakage guard: the play id is computed from t_front 0-6 s of a possession, so only samples with t_front >= 6 s (history and 1-s target
both after the descriptor window) are used, for every condition. Conditions (same samples, same seeds protocol as Stage 2b): 'base' (no play
id), 'play' (24-way one-hot appended to the context), 'shuffled' (play ids permuted across possessions). Executor v2 architecture,
profiles real, fit/validation games as in Stage 2b; holdout/test not read.
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
from executor_data import build, F_DIM, G_DIM   # noqa: E402
import executor_v2 as E                  # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from train_executor2 import per_sample, summarise   # noqa: E402

K = 24


def log(*a):
    print(*a, flush=True)


def attach(S, games, lab, shuffle_seed=None):
    """Keep samples with t_front >= 6 whose possession has a play label; return S with 'play' (N,) and ctx extended by one-hot."""
    keep = (S['ctx'][:, 4] * 24.0) >= 6.0
    play = np.full(len(keep), -1, np.int32)
    for gi, d in enumerate(games):
        m = np.flatnonzero(S['game_idx'] == gi)
        if len(m) == 0:
            continue
        pp = d['poss_id'][S['state'][m]]
        play[m] = [lab.get((d['game'], int(p)), -1) for p in pp]
    keep &= play >= 0
    S = {k: v[keep] for k, v in S.items()}
    S['play'] = play[keep]
    return S


def with_play(S, mode, seed):
    ctx = S['ctx']
    if mode == 'base':
        return dict(S, ctx=ctx)
    p = S['play'].copy()
    if mode == 'shuffled':
        # permute labels across possessions (not across samples of one possession)
        key = S['game_idx'].astype(np.int64) * 10_000_000 + S['poss_key']
        u, inv = np.unique(key, return_inverse=True)
        first = np.zeros(len(u), np.int32)
        first[inv] = p
        perm = np.random.default_rng(seed + 99).permutation(first)
        p = perm[inv]
    oh = np.eye(K, dtype=np.float32)[p]
    return dict(S, ctx=np.concatenate([ctx, oh], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', default='base:0,play:0,shuffled:0,base:1,play:1')
    ap.add_argument('--max-n', type=int, default=300000)
    ap.add_argument('--epochs', type=int, default=5)
    a = ap.parse_args()
    z = np.load(a.labels, allow_pickle=True)
    lab = {}
    for g, p, l in zip(z['fit_game'], z['fit_poss'], z['fit']):
        lab[(str(g), int(p))] = int(l)
    for g, p, l in zip(z['val_game'], z['val_poss'], z['val']):
        lab[(str(g), int(p))] = int(l)
    fit_g, val_g = fit_val_games()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    def prep(games, max_n, seed):
        S = build(games, table, 3, None, seed, 'real')
        S = attach(S, games, lab)
        S['poss_key'] = np.zeros(len(S['y']), np.int64)
        for gi, d in enumerate(games):
            m = S['game_idx'] == gi
            S['poss_key'][m] = d['poss_id'][S['state'][m]]
        if len(S['y']) > max_n:
            sel = np.sort(np.random.default_rng(seed).choice(len(S['y']), max_n, replace=False))
            S = {k: v[sel] for k, v in S.items()}
        return S
    S0, V0 = prep(FIT, a.max_n, 0), prep(VAL, 120000, 1)
    log(f'samples with t_front >= 6 s and a play label: fit {len(S0["y"])}, validation {len(V0["y"])}')
    os.makedirs(a.out, exist_ok=True)
    for ri, run in enumerate(a.runs.split(',')):
        mode, seed = run.split(':')
        seed = int(seed)
        keras.utils.set_random_seed(seed)
        S, V = with_play(S0, mode, seed), with_play(V0, mode, 1)
        g_dim = S['ctx'].shape[1]
        model = E.build_model(F_DIM, g_dim, k_modes=3)
        n_steps = (len(S['y']) // 512) * a.epochs
        lr = keras.optimizers.schedules.CosineDecay(2e-3, n_steps, alpha=0.05)
        model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
                      loss={'mix': E.mix_nll, 'dec': 'sparse_categorical_crossentropy', 'tgt': E.tgt_loss},
                      loss_weights={'mix': 1.0, 'dec': 0.3, 'tgt': 0.3})
        sw = {'mix': np.ones(len(S['y']), 'float32'), 'dec': (S['action'] >= 0).astype('float32'), 'tgt': (S['target_slot'] >= 0).astype('float32')}
        t1 = time.time()
        model.fit([S['tokens'], S['ctx']], {'mix': S['y'], 'dec': np.maximum(S['action'], 0), 'tgt': np.maximum(S['target_slot'], 0)},
                  sample_weight=sw, batch_size=512, epochs=a.epochs, verbose=2)
        ps = per_sample(model, V)
        summ = summarise(ps)
        summ.update(mode=mode, seed=seed, n_fit=int(len(S['y'])), n_val=int(len(V['y'])), train_seconds=time.time() - t1)
        log(f'[{ri + 1}] {mode} seed {seed}: ' + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()}))
        np.savez_compressed(os.path.join(a.out, f'{mode}_s{seed}.npz'), **ps)
        json.dump(summ, open(os.path.join(a.out, f'{mode}_s{seed}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
