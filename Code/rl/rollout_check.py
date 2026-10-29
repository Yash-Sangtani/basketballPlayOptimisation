"""Does the executor produce realistic multi-second trajectories, not just good one-step predictions?

From real half-court states of the VALIDATION games (profiles from the fitting games only) the model rolls all ten players forward for
3 s: predict 1 s, keep the first 0.4 s, recompute velocities from its own positions, repeat. The ball, the holder and the clocks come
from the real data at every step (this tests movement given the ball, not decisions). Compared with the real future and with a
constant-velocity extrapolation:
  * ADE at 1, 2, 3 s for offence and defence
  * realism: speeds (mean, 99th percentile) at 3 s, offensive spacing (mean pairwise distance), share of players outside the court,
    share of player pairs closer than 1.5 ft ('collisions'), defenders' mean distance to the ball handler
Two roll-outs: the mean of the predicted Gaussian, and samples from it.

    cd Code && python rl/rollout_check.py --data ../data/rl --model ../models/executor_real.keras --out ../reports/rl-stage2/rollout
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
import executor_model                    # noqa: E402,F401  (registers the custom layers)
from executor_data import make_tokens, make_ctx, pos_codes, Y_SCALE, DT, N_PROFILE   # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles, half_court_mask, profile_lookup   # noqa: E402
from train_executor import fit_val_games  # noqa: E402

H = 15           # states in the roll-out (3 s)
REPLAN = 2       # states kept per model call (0.4 s)


def starts(d, stride):
    n = len(d['frame'])
    fr = d['frame']
    step_ok = np.r_[np.diff(fr) == 5, False]
    c = np.r_[0, np.cumsum(step_ok.astype(int))]
    ks = np.arange(2, n - H)
    ok = ((c[ks + H] - c[ks - 2]) == (2 + H)) & (d['poss_id'][ks - 2] == d['poss_id'][ks + H]) & \
         (d['ids'][ks - 2] == d['ids'][ks + H]).all(axis=1) & half_court_mask(d)[ks]
    return ks[ok][::stride]


def roll(model, d, ks, table, sample, rng):
    ids = d['ids'][ks]
    prof = profile_lookup(table, ids)
    pc = pos_codes(d, ids)
    hist = [d['pos'][ks - 2], d['pos'][ks - 1], d['pos'][ks]]            # p(k-2), p(k-1), p(k)
    traj = [hist[2]]
    t = 0
    while t < H:
        k_t = ks + t
        tok = make_tokens(hist[2], hist[1], hist[0], ids, d['holder'][k_t], d['ball'][k_t], pc, prof)
        ctx = make_ctx(d['ball'][k_t, 2], d['shot_clock'][k_t], d['t_in_poss'][k_t], d['t_front'][k_t], d['quarter_time'][k_t])
        o = model.predict([tok, ctx], verbose=0, batch_size=2048)['move']                 # (N,10,5,4)
        mu, lv = o[..., :2], o[..., 2:]
        if sample == 'independent':                               # one fresh draw per player AND step (unstable: see the report)
            step = mu + rng.standard_normal(mu.shape) * np.exp(0.5 * lv)
        elif sample:                                              # one draw per player, applied along the whole predicted second
            z = rng.standard_normal((mu.shape[0], mu.shape[1], 1, 2)) * float(sample)
            step = mu + z * np.exp(0.5 * lv)
        else:
            step = mu
        p0 = hist[2]
        for s_ in range(REPLAN):                                  # keep the first 0.4 s of the predicted second
            pos_s = p0 + step[:, :, s_, :] * Y_SCALE
            hist = [hist[1], hist[2], pos_s]
            traj.append(pos_s)
        t += REPLAN
    return np.stack(traj, axis=1)                                      # (N, H+1 or H+2, 10, 2)


def realism(P, d, ks):
    """P (N, T, 10, 2) trajectories; stats at the last time step."""
    last = P[:, -1]
    prev = P[:, -3]
    sp = np.hypot(*(np.moveaxis(last - prev, -1, 0))) / (2 * DT)
    off = last[:, :5]
    pd_ = np.hypot(off[:, :, None, 0] - off[:, None, :, 0], off[:, :, None, 1] - off[:, None, :, 1])
    iu = np.triu_indices(5, 1)
    allp = np.hypot(last[:, :, None, 0] - last[:, None, :, 0], last[:, :, None, 1] - last[:, None, :, 1])
    iu10 = np.triu_indices(10, 1)
    out_of = ((last[..., 0] < 0) | (last[..., 0] > 94) | (last[..., 1] < 0) | (last[..., 1] > 50)).mean()
    return dict(speed_mean_off=float(sp[:, :5].mean()), speed_p99_off=float(np.percentile(sp[:, :5], 99)),
                speed_mean_def=float(sp[:, 5:].mean()), speed_p99_def=float(np.percentile(sp[:, 5:], 99)),
                spacing_mean_pairwise=float(pd_[:, iu[0], iu[1]].mean()), outside_court=float(out_of),
                pairs_closer_1p5ft=float((allp[:, iu10[0], iu10[1]] < 1.5).mean()))


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
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=executor_model.CUSTOM_OBJECTS)   # our own artifact (older files contain a lambda layer)
    rng = np.random.default_rng(0)
    res = {}
    rows = {k: [] for k in ('real', 'constvel', 'mean', 'sample_corr', 'sample_corr_t0.5', 'sample_independent')}
    for d in VAL:
        ks = starts(d, 25)
        if len(ks) == 0:
            continue
        real = np.stack([d['pos'][ks + t] for t in range(H + 1)], axis=1)
        cv_vel = (d['pos'][ks] - d['pos'][ks - 2]) / (2 * DT)
        cv = np.stack([d['pos'][ks] + cv_vel * (DT * t) for t in range(H + 1)], axis=1)
        pm = roll(model, d, ks, table, False, rng)
        ps = roll(model, d, ks, table, 1.0, rng)
        pi = roll(model, d, ks, table, 'independent', rng)
        pt = roll(model, d, ks, table, 0.5, rng)
        T = min(pm.shape[1], H + 1)
        for name, arr in (('real', real), ('constvel', cv), ('mean', pm[:, :T]), ('sample_corr', ps[:, :T]), ('sample_corr_t0.5', pt[:, :T]), ('sample_independent', pi[:, :T])):
            rows[name].append(arr[:, :T])
    P = {k: np.concatenate(v) for k, v in rows.items()}
    log = lambda *x: print(*x, flush=True)
    n = len(P['real'])
    log(f'{n} roll-outs from {len(VAL)} validation games')
    for name in ('constvel', 'mean', 'sample_corr', 'sample_corr_t0.5', 'sample_independent'):
        e = {}
        for sec in (1, 2, 3):
            t = int(sec / DT)
            for side, sl in (('off', slice(0, 5)), ('def', slice(5, 10))):
                err = np.hypot(*(np.moveaxis(P[name][:, t, sl] - P['real'][:, t, sl], -1, 0)))
                e[f'ade_{side}_{sec}s'] = float(err.mean())
        res[name] = dict(ade=e, **realism(P[name], None, None))
    res['real'] = realism(P['real'], None, None)
    for k, v in res.items():
        log(k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in (v.items() if k == 'real' else v.items()) if kk != 'ade'}))
        if 'ade' in v:
            log('   ADE (ft):', {kk: round(vv, 2) for kk, vv in v['ade'].items()})
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, 'rollout.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
