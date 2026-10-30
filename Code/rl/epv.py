"""EPV: expected points the controlling team scores in the rest of the possession, from the full court state (positions, velocities, ball, clocks).

    cd Code && python rl/epv.py --data ../data/rl --out ../reports/rl-stage4c/epv [--save rl/model/epv_v1.keras]

Target: field-goal points (made shots by the team in control; no free throws) between the state and the end of its possession (poss_id change).
Training: 320 fitting games, evaluation: 81 validation games (R^2 and a game-clustered bootstrap of the difference against baselines).
Baselines: constant; gradient boosting on the Stage-5b-style features (shot clock, time in possession, ball distance to the basket, ball y).
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
from keras import layers                 # noqa: E402
from executor_data import build, F_DIM, G_DIM   # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from executor_v2 import HolderPool, FirstTokens   # noqa: E402,F401


def log(*a):
    print(*a, flush=True)


def points_after(d, ks):
    """FG points by the team in control from state ks to the end of its possession."""
    fr = d['frame']
    pid = d['poss_id']
    last = {}
    for p in np.unique(pid):
        ix = np.flatnonzero(pid == p)
        last[p] = fr[ix[-1]] + 25
    sh = d['shots']
    out = np.zeros(len(ks))
    if len(sh) == 0:
        return out
    sf = sh['frame'].to_numpy()
    sp = (sh['made'].to_numpy() * np.where(sh['is3'].to_numpy() == 1, 3, 2)).astype(float)
    sid = sh['shooter'].to_numpy()
    for n, k in enumerate(ks):
        f0, f1 = fr[k], last[pid[k]]
        team = set(d['ids'][k, :5].tolist())
        m = (sf >= f0) & (sf <= f1) & np.isin(sid, list(team))
        out[n] = sp[m].sum()
    return out


def make(games, table, max_n, seed):
    S = build(games, table, 3, max_n, seed, 'real')
    y = np.zeros(len(S['y']))
    for gi, d in enumerate(games):
        m = np.flatnonzero(S['game_idx'] == gi)
        if len(m):
            y[m] = points_after(d, S['state'][m])
    S['pts'] = y.astype(np.float32)
    # hand features for the baseline
    ball = np.zeros((len(y), 2))
    for gi, d in enumerate(games):
        m = np.flatnonzero(S['game_idx'] == gi)
        if len(m):
            ball[m] = d['ball'][S['state'][m], :2]
    S['hand'] = np.c_[S['ctx'][:, 1], S['ctx'][:, 3], np.hypot(ball[:, 0] - 5.35, ball[:, 1] - 25), ball[:, 1], ball[:, 0]]
    return S


def build_model(d=64, n_layers=2, heads=4):
    tok = keras.Input((10, F_DIM), name='tokens')
    ctx = keras.Input((G_DIM,), name='ctx')
    x = layers.Dense(d)(tok)
    c = layers.Reshape((1, d))(layers.Dense(d)(ctx))
    h = layers.Concatenate(axis=1)([x, c])
    for _ in range(n_layers):
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads)(h, h)
        h = layers.LayerNormalization()(h + layers.Dropout(0.2)(a))
        h = layers.LayerNormalization()(h + layers.Dropout(0.2)(layers.Dense(d)(layers.Dense(2 * d, activation='gelu')(h))))
    p = layers.Concatenate()([layers.GlobalAveragePooling1D()(h), layers.Flatten()(h[:, 10:11])])
    out = layers.Dense(1, activation='softplus', name='pts')(layers.Dropout(0.2)(layers.Dense(64, activation='gelu')(p)))
    return keras.Model([tok, ctx], out)


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save', default=None)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max-n', type=int, default=400000)
    ap.add_argument('--n-games', type=int, default=0)
    a = ap.parse_args()
    fit_g, val_g = fit_val_games()
    if a.n_games:
        fit_g, val_g = fit_g[:a.n_games], val_g[:max(4, a.n_games // 4)]
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    S, V = make(FIT, table, a.max_n, 0), make(VAL, table, 150000, 1)
    log(f'fit {len(S["pts"])} val {len(V["pts"])}; mean points {S["pts"].mean():.3f} (val {V["pts"].mean():.3f})')
    from sklearn.ensemble import HistGradientBoostingRegressor
    gb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(S['hand'], S['pts'])
    p_gb = gb.predict(V['hand'])
    p_const = np.full(len(V['pts']), S['pts'].mean())
    keras.utils.set_random_seed(0)
    m = build_model()
    steps = (len(S['pts']) // 512) * a.epochs
    m.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(2e-3, steps, alpha=0.05), clipnorm=1.0), loss='mse')
    m.fit([S['tokens'], S['ctx']], S['pts'], batch_size=512, epochs=a.epochs, verbose=2, validation_data=([V['tokens'], V['ctx']], V['pts']))
    p_nn = m.predict([V['tokens'], V['ctx']], batch_size=4096, verbose=0)[:, 0]
    y = V['pts']
    g = V['game_idx']
    ug = np.unique(g)
    rng = np.random.default_rng(0)

    def boot(pa, pb):
        ea, eb = (y - pa) ** 2, (y - pb) ** 2
        d = np.array([(eb[g == u] - ea[g == u]).sum() for u in ug])
        c = np.array([(g == u).sum() for u in ug])
        out = [d[w].sum() / c[w].sum() for w in (rng.integers(0, len(ug), len(ug)) for _ in range(400))]
        return float(d.sum() / c.sum()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    res = dict(r2_const=r2(y, p_const), r2_gb_hand=r2(y, p_gb), r2_nn=r2(y, p_nn), mse_gain_nn_vs_gb=boot(p_nn, p_gb), mse_gain_nn_vs_const=boot(p_nn, p_const),
               calib=[[float(p_nn[(p_nn >= lo) & (p_nn < hi)].mean()), float(y[(p_nn >= lo) & (p_nn < hi)].mean()), int(((p_nn >= lo) & (p_nn < hi)).sum())]
                      for lo, hi in ((0, .5), (.5, .8), (.8, 1.1), (1.1, 1.5), (1.5, 9))])
    res = json.loads(json.dumps(res, default=float))
    log(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'epv_eval.json'), 'w'), indent=1)
    if a.save:
        m.save(a.save)


if __name__ == '__main__':
    main()
