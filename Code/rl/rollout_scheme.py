"""Stage 6B.3: do the executors follow a told defensive scheme? (a test that can fail)
    cd Code && python rl/rollout_scheme.py --n 1500 --out ../reports/rl-stage6b

Holdout front-court states with an unbroken 3.4 s real future (same possession) are rolled forward 3.2 s by the engine's movement model with nobody shooting or
passing (ball stays with the holder, shot clock held). The defence is told: nothing (free), the scheme it REALLY showed (own), or a random other scheme (other).
After the roll the knob vector averaged over +1, +2, +3 s is assigned to the nearest library scheme. Reported per condition: hit rate (realised scheme == the
scheme that was told, for free: told is the random 'other' one, i.e. chance), mean z-distance of the realised knobs to the told centroid, per-knob slope of the
realised z-knob on the told z-knob, defender speed and the share of positions off court.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
from load import load_games
from profiles import build_profiles
from splits import games as split_games
from train_executor import fit_val_games
import defense as DF
import league as L
from build_schemes import windows


def starts_for(G, n, rng):
    out = []
    for gi, d in enumerate(G):
        nn = len(d['frame'])
        ks = np.arange(2, nn - 17, 5)
        ok = d['live'][ks] & d['frontcourt'][ks] & (d['holder'][ks] != 0)
        ok &= (d['frame'][ks + 16] - d['frame'][ks] == 80) & (d['poss_id'][ks + 16] == d['poss_id'][ks]) & (d['frame'][ks] - d['frame'][ks - 2] == 10)
        ok &= (d['ids'][ks] == d['ids'][ks + 16]).all(1) & (d['t_front'][ks] > 0.5)
        out += [(gi, int(k)) for k in ks[ok]]
    sel = np.sort(rng.choice(len(out), min(n, len(out)), replace=False))
    return [out[i] for i in sel]


def roll(st, G, table, starts, told, seed):
    eng = L.make_engine(st, seed, audit=False, shoot_scale=0.0, pass_scale=0.0, tov_mult=0.0, force_p=0.0)
    eng.reset(G, table, starts)
    for r_, (g, k) in enumerate(starts):
        eng.t_front[r_] = G[g]['t_front'][k]; eng.t_in[r_] = G[g]['t_in_poss'][k]
    eng.crossed[:] = True
    eng.play[:] = -1
    eng.scheme[:, 1] = told
    for _ in range(8):
        eng.shot_clock[:] = 24.0
        eng.step_block()
    T = np.stack(eng.traj)                                   # (17,N,10,2)
    man = DF.assign(T[0])
    ball = np.stack(eng.ball_traj)[..., :2]
    raw = np.mean([DF.knobs(T[s], ball[s], man) for s in (5, 10, 15)], axis=0)
    sp = np.hypot(*(T[1:, :, 5:] - T[:-1, :, 5:]).transpose(3, 0, 1, 2)) / 0.2
    off = ((T[..., 0] < -0.5) | (T[..., 0] > 94.5) | (T[..., 1] < -0.5) | (T[..., 1] > 50.5)).mean()
    return raw, float(sp.mean()), float(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500); ap.add_argument('--out', default='../reports/rl-stage6b'); ap.add_argument('--model', default='rl/model/executor_v5g.keras')
    ap.add_argument('--split', default='holdout')
    a = ap.parse_args()
    st = L.load_stack(a.model); lib = st['lib']
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games('../data/rl', fit_g))
    G = load_games('../data/rl', split_games(a.split))
    rng = np.random.default_rng(0)
    starts = starts_for(G, a.n, rng)
    n = len(starts)
    # the scheme the defence really showed (hindsight, as in training) and a random other one
    real_raw = []
    for g, k in starts:
        d = G[g]
        man = DF.assign(d['pos'][k:k + 1])
        real_raw.append(np.mean([DF.knobs(d['pos'][k + s:k + s + 1], d['ball'][k + s:k + s + 1, :2], man) for s in (5, 10, 15)], axis=0)[0])
    real_raw = np.array(real_raw)
    own = lib.nearest(real_raw)
    ok = lib.freq[own] > 0.01
    other = np.array([(o + rng.integers(1, lib.K)) % lib.K for o in own])
    valid = lib.freq[other] >= 0.01
    res = dict(n=n, real_scheme_share=np.bincount(own, minlength=lib.K).tolist())
    for name, told in (('free', np.full(n, -1)), ('own', own), ('other', other)):
        t0 = time.time()
        raw, spd, off = roll(st, G, table, starts, told, 3)
        z = lib.z(raw)
        tgt = other if name != 'own' else own
        m = valid if name != 'own' else ok
        hit = float((lib.nearest(raw) == tgt)[m].mean())
        dist = float(np.sqrt(((z - lib.C[tgt]) ** 2).sum(1))[m].mean())
        zt = lib.C[tgt]
        slopes = [float(np.polyfit(zt[m, j], z[m, j], 1)[0]) for j in range(DF.N_KNOB)]
        res[name] = dict(target='own' if name == 'own' else 'other', hit=hit, z_dist=dist, slopes=slopes, speed_def=spd, off_court=off, seconds=time.time() - t0)
        if name == 'free':
            res[name]['hit_own'] = float((lib.nearest(raw) == own)[ok].mean())
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}), flush=True)
    res['chance_hit'] = float((lib.freq ** 2).sum())
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'scheme_following.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
