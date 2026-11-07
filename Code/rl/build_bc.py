"""Stage 6B.4: behaviour-cloned scheme prior. What scheme do real defences play, given the state?

    cd Code && python rl/build_bc.py --out ../reports/rl-stage6b

Label = library scheme of the knob vector the defence showed over the next 3 s (as in the executor's hindsight goals); features = the scheme caller's state features
(current knobs, ball position, shot clock, front-court time, offence near the basket, scheme of the previous 3 s). A small MLP; reported against the
state-independent library frequencies on validation games (log-loss, top-1). It is the human-like prior of the defence agents (the KL budget is measured against it).
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, keras
from load import load_games
from train_executor import fit_val_games
import defense as DF

BC_PATH = os.path.join(HERE, 'model', 'scheme_bc.keras')


def game_data(d, lib):
    n = len(d['frame'])
    ks = np.arange(15, n - 16, 5)
    ok = d['live'][ks] & d['frontcourt'][ks] & (d['holder'][ks] != 0)
    ok &= (d['frame'][ks + 15] - d['frame'][ks] == 75) & (d['poss_id'][ks + 15] == d['poss_id'][ks]) & (d['ids'][ks] == d['ids'][ks + 15]).all(1)
    ks = ks[ok]
    if len(ks) == 0:
        return None
    pos, ball = d['pos'][ks], d['ball'][ks, :2]
    man = DF.assign(pos)
    fut = lib.nearest(np.mean([DF.knobs(d['pos'][ks + s], d['ball'][ks + s, :2], man) for s in (5, 10, 15)], axis=0))
    # previous scheme = the scheme of the window that ended now (states -15, -10, -5), only when the possession has been in the front court for >= 3 s
    past_ok = (d['t_front'][ks] >= 3.0) & (d['poss_id'][ks - 15] == d['poss_id'][ks]) & (d['frame'][ks] - d['frame'][ks - 15] == 75)
    prev = np.full(len(ks), -1)
    if past_ok.any():
        kk = ks[past_ok]
        manp = DF.assign(d['pos'][kk - 15])
        prev[past_ok] = lib.nearest(np.mean([DF.knobs(d['pos'][kk + s], d['ball'][kk + s, :2], manp) for s in (-15, -10, -5)], axis=0))
    X = DF.agent_features(lib, pos, ball, d['shot_clock'][ks], d['t_front'][ks], prev)
    return X, fut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/rl'); ap.add_argument('--out', default='../reports/rl-stage6b'); ap.add_argument('--epochs', type=int, default=8)
    a = ap.parse_args()
    lib = DF.SchemeLibrary()
    fit_g, val_g = fit_val_games()

    def collect(names):
        Xs, ys = [], []
        for g in names:
            d = load_games(a.data, [g])
            if d:
                r = game_data(d[0], lib)
                if r is not None:
                    Xs.append(r[0]); ys.append(r[1])
        return np.concatenate(Xs), np.concatenate(ys)
    X, y = collect(fit_g)
    Xv, yv = collect(val_g)
    print(len(X), len(Xv), flush=True)
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input((DF.D_AGENT,)), keras.layers.Dense(64, activation='gelu'), keras.layers.Dense(64, activation='gelu'), keras.layers.Dense(lib.K, activation='softmax')])
    m.compile(optimizer=keras.optimizers.Adam(2e-3), loss='sparse_categorical_crossentropy')
    m.fit(X, y, batch_size=512, epochs=a.epochs, verbose=2, validation_data=(Xv, yv))
    p = m.predict(Xv, batch_size=4096, verbose=0)
    ll = -np.log(np.clip(p[np.arange(len(yv)), yv], 1e-9, 1)).mean()
    fr = np.bincount(y, minlength=lib.K) / len(y)
    ll0 = -np.log(np.clip(fr[yv], 1e-9, 1)).mean()
    res = dict(n_fit=int(len(X)), n_val=int(len(Xv)), logloss_state_conditional=float(ll), logloss_frequency_only=float(ll0), top1_state_conditional=float((p.argmax(1) == yv).mean()),
               top1_frequency_only=float((yv == fr.argmax()).mean()), mean_entropy=float(-(p * np.log(np.clip(p, 1e-9, 1))).sum(1).mean()))
    print(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'scheme_bc.json'), 'w'), indent=1)
    m.save(BC_PATH)


if __name__ == '__main__':
    main()
