"""Executor v3: v2 plus play conditioning (play_cond.py). Trains 'play' / 'shuffled' / 'none' variants and compares them where it matters:
samples inside the play window (t_front < 6 s of a possession that belongs to a play).

    cd Code && python rl/train_executor3.py --data ../data/rl --out ../reports/rl-stage4c/exec3 --runs play:0,none:0,shuffled:0 --save rl/model/executor_v3.keras

Same fit/validation games and training budget as Stage 2b (300k samples, 5 epochs, 3 mixture modes). Play of a possession = library assignment of its
first-6-s descriptor (play_reward.py); the executor is told that play for the part of the possession inside the window. 'shuffled' gives each possession
another possession's play. 'none' has the same inputs with the play features set to zero.
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
from plays import possessions            # noqa: E402
from profiles import build_profiles      # noqa: E402
from play_reward import PlayLibrary, describe_seq   # noqa: E402
from play_cond import PlayTargets, TOK_EXTRA, CTX_EXTRA   # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from train_executor2 import per_sample, summarise   # noqa: E402


def log(*a):
    print(*a, flush=True)


def assign_plays(games, lib):
    """{(game index, poss_id): play} for every possession of >= 6 s."""
    lab = {}
    for gi, d in enumerate(games):
        P, X = [], []
        for p, idx in possessions(d):
            P.append(p)
            X.append(describe_seq(d['pos'][idx], d['ball'][idx, :2], d['t_front'][idx]))
        if X:
            k, _ = lib.assign(np.array(X))
            for p, c in zip(P, k):
                lab[(gi, int(p))] = int(c)
    return lab


def goal_features(S, games, seed, drop=0.3):
    """Hindsight goals: for samples in the play window, the cell centre (grid of plays.py) where each attacker actually is 1 s and 3 s later. A random 30%
    of window samples get no goal (flag 0), so the executor still moves freely when no play is chosen. Same 6 + 2 feature layout as play_cond."""
    from plays import X_EDGES, Y_EDGES, GRID
    N = len(S['y'])
    pos = S['tokens'][:, :, 0:2] * np.array([47.0, 25.0], np.float32) + np.array([47.0, 25.0], np.float32)
    tf = S['ctx'][:, 4] * 24.0
    tok = np.zeros((N, 10, TOK_EXTRA), np.float32)
    ctx = np.zeros((N, CTX_EXTRA), np.float32)
    rng = np.random.default_rng(seed + 11)
    for gi, d in enumerate(games):
        m = np.flatnonzero((S['game_idx'] == gi) & (tf < 6.0))
        if len(m) == 0:
            continue
        st = S['state'][m]
        ok = (st + 15 < len(d['frame']))
        st_c = np.minimum(st, len(d['frame']) - 16)
        ok &= (d['frame'][st_c + 15] - d['frame'][st_c] == 75) & (d['poss_id'][st_c + 15] == d['poss_id'][st_c])
        ok &= rng.random(len(m)) >= drop
        for h, k in enumerate((5, 15)):
            fp = d['pos'][st_c + k][:, :5]
            ix = np.clip(np.searchsorted(X_EDGES, fp[..., 0], side='right') - 1, 0, GRID[0] - 1)
            iy = np.clip(np.searchsorted(Y_EDGES, fp[..., 1], side='right') - 1, 0, GRID[1] - 1)
            cx = (X_EDGES[ix] + X_EDGES[ix + 1]) / 2
            cy = (Y_EDGES[iy] + Y_EDGES[iy + 1]) / 2
            dx, dy = cx - pos[m, :5, 0], cy - pos[m, :5, 1]
            tok[m[ok], :5, 2 * h] = (dx[ok] / 20.0)
            tok[m[ok], :5, 2 * h + 1] = (dy[ok] / 20.0)
        ctx[m[ok]] = np.stack([np.ones(ok.sum()), (6.0 - tf[m[ok]]) / 6.0], axis=1)
    return tok, ctx


def with_play(S, games, lab, pt, mode, seed):
    """Return S with tokens/ctx extended by the play features."""
    pos = S['tokens'][:, :, 0:2] * np.array([47.0, 25.0], np.float32) + np.array([47.0, 25.0], np.float32)
    tf = S['ctx'][:, 4] * 24.0
    play = np.full(len(tf), -1, np.int32)
    key = np.zeros(len(tf), np.int64)
    for gi, d in enumerate(games):
        m = np.flatnonzero(S['game_idx'] == gi)
        if len(m):
            pp = d['poss_id'][S['state'][m]]
            key[m] = gi * 10_000_000 + pp
            play[m] = [lab.get((gi, int(p)), -1) for p in pp]
    S = dict(S, play_real=play.copy())
    if mode == 'shuffled':
        u, inv = np.unique(key, return_inverse=True)
        first = np.full(len(u), -1, np.int32)
        first[inv] = play
        lab_u = first.copy()
        have = np.flatnonzero(lab_u >= 0)
        lab_u[have] = np.random.default_rng(seed + 5).permutation(lab_u[have])
        play = lab_u[inv]
    if mode == 'none':
        tok_x = np.zeros((len(tf), 10, TOK_EXTRA), np.float32)
        ctx_x = np.zeros((len(tf), CTX_EXTRA), np.float32)
    elif mode == 'goal':
        tok_x, ctx_x = goal_features(S, games, seed)
    else:
        tok_x, ctx_x = pt.features(pos, tf, play)
    return dict(S, tokens=np.concatenate([S['tokens'], tok_x], axis=2), ctx=np.concatenate([S['ctx'], ctx_x], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', default='play:0,none:0,shuffled:0')
    ap.add_argument('--max-n', type=int, default=300000)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--save', default=None)
    ap.add_argument('--save-mode', default='play')
    ap.add_argument('--n-games', type=int, default=0)
    a = ap.parse_args()
    fit_g, val_g = fit_val_games()
    if a.n_games:
        fit_g, val_g = fit_g[:a.n_games], val_g[:max(4, a.n_games // 4)]
    t0 = time.time()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    lib, pt = PlayLibrary(), PlayTargets()
    lab_f, lab_v = assign_plays(FIT, lib), assign_plays(VAL, lib)
    S0 = build(FIT, table, 3, a.max_n, 0, 'real')
    V0 = build(VAL, table, 3, 120000, 1, 'real')
    log(f'data ready ({time.time() - t0:.0f}s): {len(S0["y"])} fit, {len(V0["y"])} val samples')
    os.makedirs(a.out, exist_ok=True)
    for ri, run in enumerate(a.runs.split(',')):
        mode, seed = run.split(':')
        seed = int(seed)
        keras.utils.set_random_seed(seed)
        S = with_play(S0, FIT, lab_f, pt, mode, seed)
        V = with_play(V0, VAL, lab_v, pt, mode, 1)
        model = E.build_model(F_DIM + TOK_EXTRA, G_DIM + CTX_EXTRA, k_modes=3)
        n_steps = (len(S['y']) // 512) * a.epochs
        model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(2e-3, n_steps, alpha=0.05), clipnorm=1.0),
                      loss={'mix': E.mix_nll, 'dec': 'sparse_categorical_crossentropy', 'tgt': E.tgt_loss}, loss_weights={'mix': 1.0, 'dec': 0.3, 'tgt': 0.3})
        sw = {'mix': np.ones(len(S['y']), 'float32'), 'dec': (S['action'] >= 0).astype('float32'), 'tgt': (S['target_slot'] >= 0).astype('float32')}
        t1 = time.time()
        model.fit([S['tokens'], S['ctx']], {'mix': S['y'], 'dec': np.maximum(S['action'], 0), 'tgt': np.maximum(S['target_slot'], 0)},
                  sample_weight=sw, batch_size=512, epochs=a.epochs, verbose=2)
        ps = per_sample(model, V)
        win = (V['play_real'] >= 0) & (V['ctx'][:, 4] * 24.0 < 6.0)
        summ = {f'all_{k}': v for k, v in summarise(ps).items()}
        summ.update({f'win_{k}': float(np.nanmean(v[win])) for k, v in ps.items() if k != 'game_idx'})
        summ.update(mode=mode, seed=seed, n_win=int(win.sum()), n_val=int(len(win)), train_seconds=time.time() - t1)
        log(f'[{ri + 1}] {mode} seed {seed}: ' + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()}))
        np.savez_compressed(os.path.join(a.out, f'{mode}_s{seed}.npz'), win=win, **ps)
        json.dump(summ, open(os.path.join(a.out, f'{mode}_s{seed}.json'), 'w'), indent=1)
        if a.save and mode == a.save_mode:
            model.save(a.save)


if __name__ == '__main__':
    main()
