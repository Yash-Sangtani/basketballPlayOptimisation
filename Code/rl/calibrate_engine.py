"""Calibrate the engine's shot-selection constants against TRACKING-based real shot statistics of the RL-train games (never the holdout):
the shoot hazard multiplier by shot distance bin (dist_mult), so that the simulated distribution of field-goal attempts by release distance and the number of attempts per team-quarter match the
empirical table (shot_table.npz counts, measured at the shooter's position at release, exactly as the engine measures them).
    python rl/calibrate_engine.py --model rl/model/executor_v5g.keras --iters 4 --n 96 --blocks 450 --out ../reports/rl-stage6a/calib_v7
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, pandas as pd
from load import load_games
from profiles import build_profiles
from splits import games as split_games
from train_executor import fit_val_games
from engine_validate import quarter_starts
from shot_audit import bin_index
import league as L

FGA_TQ = 21.025       # real field-goal attempts per team-quarter (play-by-play, holdout) -- the same number engine_validate uses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--iters', type=int, default=4); ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--blocks', type=int, default=450); ap.add_argument('--out', required=True); ap.add_argument('--dec-h', type=float, default=0.4); ap.add_argument('--damp', type=float, default=0.8)
    ap.add_argument('--start-cal', default=None, help='calibration json to start from (default: shoot_scale 1, dist_mult 1)')
    ap.add_argument('--mode', default='shots', choices=['shots', 'events'], help='shots: update dist_mult; events: update foul_mult, tov_mult, steal_share to the real TRAIN team-quarter means')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games('../data/rl', fit_g))
    G = load_games('../data/rl', split_games('train'))
    qs = quarter_starts(G)
    rng = np.random.default_rng(1)
    cnt = np.load(os.path.join(HERE, 'model', 'shot_table.npz'))['cnt']
    real = cnt.sum((1, 2)) / cnt.sum()
    cal = dict(st['cal'])
    if a.start_cal:
        cal.update(json.load(open(a.start_cal)))
    else:
        cal.update(dict(shoot_scale=1.0, dec_h=a.dec_h, dist_mult=[1.0] * 8))
    cal.pop('audit', None)
    target = None
    if a.mode == 'events':
        from engine_validate import real_quarter_stats
        R = real_quarter_stats('../reports/rl-stage6a', 'train', None)
        target = R[['fta', 'tov', 'stl', 'poss', 'pts', 'fga']].mean().to_dict()
        print('real train targets', {k: round(v, 2) for k, v in target.items()}, flush=True)
    hist = []
    for it in range(a.iters):
        pick = rng.choice(len(qs), a.n, replace=False)
        starts = [(qs[j][0], qs[j][1]) for j in pick]
        t0 = time.time()
        eng = L.make_engine(st, 11 + it, audit=True, **cal)
        eng.reset(G, table, starts); eng.run(a.blocks)
        sl = pd.DataFrame(eng.shot_log)
        sl = sl[~sl.fouled]
        bi, _, _ = bin_index(sl.dist.to_numpy(), 24 - sl.clock.to_numpy(), sl.def1.to_numpy())
        secs = a.blocks * 0.4
        per_q = len(sl) / (2 * eng.N) * 720.0 / secs
        sim = np.bincount(bi, minlength=8) / len(sl)
        want = FGA_TQ * real
        have = per_q * sim
        r = np.where(have > 0, want / np.maximum(have, 1e-6), 3.0)
        newm = np.clip(np.array(cal['dist_mult']) * np.clip(r, 0.25, 4.0) ** a.damp, 0.02, 400.0)
        s = {k: float(v.sum() / (2 * eng.N) * 720.0 / secs) for k, v in eng.stats.items()}
        print(f'it {it} {time.time() - t0:.0f}s  FGA/q {per_q:.2f} (real {FGA_TQ})  3PA {s["fg3a"]:.2f} pts {s["pts"]:.2f} tov {s["tov"]:.2f} poss {s["poss"]:.2f}', flush=True)
        print('   sim share ', np.round(sim, 3).tolist(), '\n   real share', np.round(real, 3).tolist(), '\n   dist_mult ->', np.round(newm, 3).tolist(), flush=True)
        hist.append(dict(it=it, dist_mult=cal['dist_mult'], fga=per_q, share=sim.tolist(), stats=s))
        if a.mode == 'shots':
            cal['dist_mult'] = newm.tolist()
        else:
            cal['foul_mult'] = float(cal['foul_mult'] * (target['fta'] / max(s['fta'], 1e-6)) ** a.damp)
            cal['tov_mult'] = float(cal['tov_mult'] * (target['tov'] / max(s['tov'], 1e-6)) ** a.damp)
            cal['steal_share'] = float(np.clip(cal['steal_share'] * (target['stl'] / max(s['stl'], 1e-6)) ** a.damp, 0.1, 0.9))
            print(f'   fta {s["fta"]:.2f} stl {s["stl"]:.2f} -> foul_mult {cal["foul_mult"]:.3f} tov_mult {cal["tov_mult"]:.3f} steal_share {cal["steal_share"]:.3f}', flush=True)
    json.dump(dict(final=cal, hist=hist), open(os.path.join(a.out, 'calib.json'), 'w'), indent=1)
    print('final dist_mult', cal['dist_mult'])


if __name__ == '__main__':
    main()
