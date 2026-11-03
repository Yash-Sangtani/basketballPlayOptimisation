"""Short engine runs for calibration against TRACKING-based real statistics (shot distance from the shooter's position at release, as the sim measures it).
    python rl/calib_engine.py --n 48 --blocks 750 --variants 1.0:1.0,1.0:1.1 [--dist-mult file.json]
Variant = tau:gain[:shoot_scale]. Prints shot distance shares (tracking), elapsed shares, speeds, holder distance histogram vs real."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, pandas as pd, keras, executor_v2 as E
from load import load_games
from profiles import build_profiles
from splits import games as split_games
from train_executor import fit_val_games
from play_cond import PlayTargets
from game_engine import Engine
from engine_validate import quarter_starts
from shot_audit import D_EDGES, C_EDGES, bin_index
BASKET = np.array([5.35, 25.0])
BINS = [0, 4, 6, 10, 15, 20, 25, 30, 94]


def real_ref(G, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    hd, sp_o, sp_d = [], [], []
    for d in G:
        fr = d['frame']
        m = d['live'] & (d['holder'] != 0) & (d['pos'][:, :5, 0].mean(1) < 47)
        ks = np.flatnonzero(m)
        for k in ks:
            ids = d['ids'][k][:5]; j = np.flatnonzero(ids == d['holder'][k])
            if len(j): hd.append(np.hypot(*(d['pos'][k, j[0]] - BASKET)))
        ok = ks[(ks >= 1)]
        ok = ok[(fr[ok] - fr[ok - 1] == 5)]
        v = np.hypot(*(d['pos'][ok] - d['pos'][ok - 1]).transpose(2, 0, 1)) / 0.2
        sp_o.append(v[:, :5].mean()); sp_d.append(v[:, 5:].mean())
    z = np.load(os.path.join(HERE, 'model', 'shot_table.npz'))['cnt']
    r = dict(holder=(np.histogram(hd, BINS)[0] / len(hd)).tolist(), speed_off=float(np.mean(sp_o)), speed_def=float(np.mean(sp_d)),
             shot_dist=(z.sum((1, 2)) / z.sum()).tolist(), shot_clock=(z.sum((0, 2)) / z.sum()).tolist(), n_shots=float(z.sum()))
    json.dump(r, open(cache, 'w'))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48); ap.add_argument('--blocks', type=int, default=750)
    ap.add_argument('--variants', default='1.0:1.0'); ap.add_argument('--dist-mult', default=None); ap.add_argument('--model', default='rl/model/executor_v4g.keras')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--split', default='holdout'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from api import PassValueModel
    cal = json.load(open(os.path.join(HERE, 'model', 'engine_calibration.json')))
    if a.dist_mult: cal['dist_mult'] = json.load(open(a.dist_mult))
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games('../data/rl', fit_g))
    G = load_games('../data/rl', split_games(a.split))
    ref = real_ref(G, f'/tmp/claude-0/-home-user/bb835e0e-1ba5-530d-a467-365f61c7b413/scratchpad/real_ref_{a.split}.json')
    qs = quarter_starts(G); rng = np.random.default_rng(a.seed); qs = [qs[i] for i in np.sort(rng.choice(len(qs), a.n, replace=False))]
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff, pt = PassValueModel(), PlayTargets()
    print('real  shot dist', np.round(ref['shot_dist'], 3).tolist(), '\n      clock', np.round(ref['shot_clock'], 3).tolist(), '\n      holder', np.round(ref['holder'], 3).tolist(), 'speed off/def %.2f/%.2f' % (ref['speed_off'], ref['speed_def']))
    out = {}
    for v in a.variants.split(','):
        x = [float(t) for t in v.split(':')]
        kw = dict(cal); kw['tau'], kw['gain'] = x[0], x[1]
        if len(x) > 2: kw['shoot_scale'] = x[2]
        t0 = time.time()
        eng = Engine(model, diff, plays=pt, seed=a.seed, **kw).reset(G, table, [(g, k) for g, k, _ in qs]); eng.run(a.blocks)
        sl = pd.DataFrame(eng.shot_log)
        bi, bj, _ = bin_index(sl.dist.to_numpy(), 24 - sl.clock.to_numpy(), sl.def1.to_numpy())
        sd = np.bincount(bi, minlength=8) / len(sl); sc = np.bincount(bj, minlength=4) / len(sl)
        T = np.stack(eng.traj); Hh = np.stack(eng.holder_traj); Oo = np.stack(eng.order_traj)
        live = (Hh[1:] != 0) & (T[1:, :, :5, 0].mean(2) < 47)
        sp = np.hypot(*(T[1:] - T[:-1]).transpose(3, 0, 1, 2)) / 0.2
        hd = []
        for t in range(len(T)):
            for i in range(eng.N):
                if Hh[t, i] == 0 or T[t, i, :5, 0].mean() >= 47: continue
                j = np.flatnonzero(eng.R_ids[i, Oo[t, i]][:5] == Hh[t, i])
                if len(j): hd.append(np.hypot(*(T[t, i, j[0]] - BASKET)))
        hh = np.histogram(hd, BINS)[0] / len(hd)
        secs = args_secs = a.blocks * 0.4
        per_team_q = lambda k: float(eng.stats[k].sum() / (2 * eng.N) * 720.0 / secs)
        r = dict(shot_dist=sd.tolist(), shot_clock=sc.tolist(), holder=hh.tolist(), speed_off=float(sp[live[..., None].repeat(5, 2)[:, :, :5]].mean()) if False else float(np.mean([sp[:, :, :5][live].mean()])), speed_def=float(sp[:, :, 5:][live].mean()),
                 fga=per_team_q('fga'), fg3a=per_team_q('fg3a'), pts=per_team_q('pts'), tov=per_team_q('tov'), poss=per_team_q('poss'), seconds=time.time() - t0)
        out[v] = r
        print(f'[{v}] {r["seconds"]:.0f}s  FGA/q {r["fga"]:.1f} 3PA/q {r["fg3a"]:.1f} pts/q {r["pts"]:.1f} tov {r["tov"]:.1f} poss {r["poss"]:.1f}  speed off/def {r["speed_off"]:.2f}/{r["speed_def"]:.2f}')
        print('   shot dist ', np.round(sd, 3).tolist(), '\n   clock     ', np.round(sc, 3).tolist(), '\n   holder    ', np.round(hh, 3).tolist(), flush=True)
    if a.out: json.dump(dict(ref=ref, variants=out), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
