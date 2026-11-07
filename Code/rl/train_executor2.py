"""Train and evaluate executor v2 for several (profile mode, seed) runs in one process (the games and the profile table are loaded once).

    cd Code && python rl/train_executor2.py --data ../data/rl --runs real:0,none:0,real:1,none:1,real:2,none:2,shuffled:0 --out ../reports/rl-stage2b

Split and data discipline as in train_executor.py: 320 fit / 81 validation games from the 401 `train` games, profiles from the fit games
only, holdout/test not read. The validation sample is the SAME for every run (fixed seed); `seed` changes the network initialisation, the
data order and the training subsample, so runs of one mode measure seed-to-seed variation. Per-sample validation losses are saved
(`<mode>_s<seed>.npz`) for paired game-clustered bootstrap comparisons (compare_runs.py).
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
from executor_data import build, F_DIM, G_DIM, Y_SCALE   # noqa: E402
import executor_v2 as E                  # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from train_executor import fit_val_games  # noqa: E402


def log(*a):
    print(*a, flush=True)


def predict_all(model, S, batch=4096):
    outs = {'mix': [], 'dec': [], 'tgt': []}
    for i in range(0, len(S['y']), batch):
        o = model.predict([S['tokens'][i:i + batch], S['ctx'][i:i + batch]], verbose=0)
        for k in outs:
            outs[k].append(o[k])
    return {k: np.concatenate(v) for k, v in outs.items()}


def mixture_stats(mix):
    """numpy version of split_mix -> weights (B,10,K), mu (B,10,K,5,2) in feet, logvar."""
    logit = mix[..., 0]
    K = mix.shape[2]
    mu = mix[..., 1:11].reshape(mix.shape[0], 10, K, 5, 2)
    lv = np.clip(mix[..., 11:21].reshape(mix.shape[0], 10, K, 5, 2), -6, 3)
    w = np.exp(logit - logit.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w, mu, lv


def per_sample(model, S):
    o = predict_all(model, S)
    w, mu, lv = mixture_stats(o['mix'])
    y = S['y'][:, :, None]
    comp = -0.5 * (lv + (y - mu) ** 2 * np.exp(-lv)).sum(axis=(-2, -1))                 # (B,10,K)
    a = np.log(np.clip(w, 1e-12, 1)) + comp
    mx = a.max(-1, keepdims=True)
    ll = (mx[..., 0] + np.log(np.exp(a - mx).sum(-1)))                                  # (B,10)
    nll = -ll / 10.0
    mean_traj = (w[..., None, None] * mu).sum(2)                                        # (B,10,5,2)
    err_mean = np.hypot(*np.moveaxis(mean_traj - S['y'], -1, 0)) * Y_SCALE              # (B,10,5)
    err_k = np.hypot(*np.moveaxis(mu - S['y'][:, :, None], -1, 0)) * Y_SCALE            # (B,10,K,5)
    min_err = err_k[..., -1].min(-1)                                                     # best-of-K error at 1 s (B,10)
    a_ = S['action']
    dec_ll = np.full(len(a_), np.nan)
    m = a_ >= 0
    dec_ll[m] = -np.log(np.clip(o['dec'][m][np.arange(m.sum()), a_[m]], 1e-9, 1))
    t_ = S['target_slot']
    tgt_ll = np.full(len(t_), np.nan)
    m2 = t_ >= 0
    tgt_ll[m2] = -np.log(np.clip(o['tgt'][m2][np.arange(m2.sum()), t_[m2]], 1e-9, 1))
    tgt_acc = np.full(len(t_), np.nan)
    tgt_acc[m2] = (o['tgt'][m2].argmax(1) == t_[m2]).astype(float)
    return dict(nll_off=nll[:, :5].mean(1), nll_def=nll[:, 5:].mean(1), ade1_off=err_mean[:, :5, -1].mean(1), ade1_def=err_mean[:, 5:, -1].mean(1),
                minade1_off=min_err[:, :5].mean(1), minade1_def=min_err[:, 5:].mean(1), dec_ll=dec_ll, tgt_ll=tgt_ll, tgt_acc=tgt_acc,
                top_weight=w.max(-1).mean(axis=1), game_idx=S['game_idx'])


def summarise(ps):
    return {k: float(np.nanmean(v)) for k, v in ps.items() if k != 'game_idx'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--runs', default='real:0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-n', type=int, default=300000)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--k-modes', type=int, default=3)
    ap.add_argument('--n-games', type=int, default=0)
    ap.add_argument('--save', default=None, help='save the model of the first run to this path')
    a = ap.parse_args()
    fit_g, val_g = fit_val_games()
    if a.n_games:
        fit_g, val_g = fit_g[:a.n_games], val_g[:max(5, a.n_games // 4)]
    t0 = time.time()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    log(f'{len(FIT)} fit / {len(VAL)} validation games, {len(table)} profiles ({time.time() - t0:.0f}s)')
    os.makedirs(a.out, exist_ok=True)
    Vcache = {}
    for ri, run in enumerate(a.runs.split(',')):
        mode, seed = run.split(':')
        seed = int(seed)
        keras.utils.set_random_seed(seed)
        S = build(FIT, table, a.stride, a.max_n, seed, mode)
        if mode not in Vcache:
            Vcache[mode] = build(VAL, table, a.stride, 120000, 1, mode)
        V = Vcache[mode]
        model = E.build_model(F_DIM, G_DIM, k_modes=a.k_modes)
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
        summ.update(mode=mode, seed=seed, n_fit=int(len(S['y'])), n_val=int(len(V['y'])), params=int(model.count_params()), train_seconds=time.time() - t1)
        log(f'[{ri + 1}] {mode} seed {seed}: ' + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items()}))
        np.savez_compressed(os.path.join(a.out, f'{mode}_s{seed}.npz'), **ps)
        json.dump(summ, open(os.path.join(a.out, f'{mode}_s{seed}.json'), 'w'), indent=1)
        if a.save and ri == 0:
            os.makedirs(os.path.dirname(a.save), exist_ok=True)
            model.save(a.save)


if __name__ == '__main__':
    main()
