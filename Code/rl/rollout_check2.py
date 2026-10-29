"""Roll-out realism check for executor v2 (mixture outputs), same protocol as rollout_check.py (3 s, replan every 0.4 s, real ball/holder
/clocks, validation games, profiles from the fitting games).

Roll-out policies: 'mean' (mixture mean), 'top' (most likely mode, no noise), 'sample' (draw a mode per player and call from the mixture
weights, then one correlated Gaussian draw per player inside the mode), 'sample_t0.7' (the same with the noise scaled by 0.7).

    cd Code && python rl/rollout_check2.py --data ../data/rl --model rl/model/executor_v2.keras --out ../reports/rl-stage2b/rollout
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
import executor_v2 as E                  # noqa: E402
from executor_data import make_tokens, make_ctx, pos_codes, Y_SCALE, DT   # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles, profile_lookup   # noqa: E402
from rollout_check import starts, realism, H, REPLAN   # noqa: E402
from train_executor import fit_val_games   # noqa: E402
from train_executor2 import mixture_stats  # noqa: E402


def roll(model, d, ks, table, policy, rng, tau=1.0):
    ids = d['ids'][ks]
    prof = profile_lookup(table, ids)
    pc = pos_codes(d, ids)
    hist = [d['pos'][ks - 2], d['pos'][ks - 1], d['pos'][ks]]
    traj = [hist[2]]
    t = 0
    while t < H:
        k_t = ks + t
        tok = make_tokens(hist[2], hist[1], hist[0], ids, d['holder'][k_t], d['ball'][k_t], pc, prof)
        ctx = make_ctx(d['ball'][k_t, 2], d['shot_clock'][k_t], d['t_in_poss'][k_t], d['t_front'][k_t], d['quarter_time'][k_t])
        o = model.predict([tok, ctx], verbose=0, batch_size=2048)['mix']
        w, mu, lv = mixture_stats(o)                                      # (N,10,K), (N,10,K,5,2)
        N = len(ks)
        if policy == 'mean':
            step = (w[..., None, None] * mu).sum(2)
        elif policy == 'top':
            k = w.argmax(-1)
            step = np.take_along_axis(mu, k[:, :, None, None, None], axis=2)[:, :, 0]
        else:
            u = rng.random((N, 10, 1))
            k = (np.cumsum(w, -1) < u).sum(-1).clip(0, w.shape[-1] - 1)
            m = np.take_along_axis(mu, k[:, :, None, None, None], axis=2)[:, :, 0]
            s = np.exp(0.5 * np.take_along_axis(lv, k[:, :, None, None, None], axis=2)[:, :, 0])
            z = rng.standard_normal((N, 10, 1, 2)) * tau
            step = m + z * s
        p0 = hist[2]
        for s_ in range(REPLAN):
            pos_s = p0 + step[:, :, s_, :] * Y_SCALE
            hist = [hist[1], hist[2], pos_s]
            traj.append(pos_s)
        t += REPLAN
    return np.stack(traj, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n-games', type=int, default=40)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    fit_g, val_g = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    VAL = load_games(a.data, val_g[:a.n_games])
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    rng = np.random.default_rng(0)
    pol = {'mean': ('mean', 1.0), 'top': ('top', 1.0), 'sample': ('sample', 1.0), 'sample_t0.7': ('sample', 0.7)}
    rows = {k: [] for k in ['real', 'constvel', *pol]}
    for d in VAL:
        ks = starts(d, 25)
        if len(ks) == 0:
            continue
        real = np.stack([d['pos'][ks + t] for t in range(H + 1)], axis=1)
        vel = (d['pos'][ks] - d['pos'][ks - 2]) / (2 * DT)
        rows['real'].append(real)
        rows['constvel'].append(np.stack([d['pos'][ks] + vel * (DT * t) for t in range(H + 1)], axis=1))
        for name, (p, tau) in pol.items():
            r = roll(model, d, ks, table, p, rng, tau)
            rows[name].append(r[:, :H + 1])
    P = {k: np.concatenate(v) for k, v in rows.items()}
    res = {}
    print(f"{len(P['real'])} roll-outs from {len(VAL)} validation games", flush=True)
    for name in ['constvel', *pol]:
        e = {}
        for sec in (1, 2, 3):
            t = int(sec / DT)
            for side, sl in (('off', slice(0, 5)), ('def', slice(5, 10))):
                e[f'ade_{side}_{sec}s'] = float(np.hypot(*np.moveaxis(P[name][:, t, sl] - P['real'][:, t, sl], -1, 0)).mean())
        res[name] = dict(ade=e, **realism(P[name], None, None))
    res['real'] = realism(P['real'], None, None)
    for k, v in res.items():
        print(k, json.dumps({kk: round(vv, 3) for kk, vv in v.items() if kk != 'ade'}))
        if 'ade' in v:
            print('   ADE (ft):', {kk: round(vv, 2) for kk, vv in v['ade'].items()})
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, 'rollout.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
