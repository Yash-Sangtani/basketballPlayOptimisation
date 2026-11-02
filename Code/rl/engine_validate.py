"""Stage 6A.5: simulate quarters from real quarter starts and compare with real quarters.

    cd Code && python rl/engine_validate.py --split holdout --n 200 --out ../reports/rl-stage6a [--configs shoot:pass:tov:foul,...]

Real quarter statistics per team come from the play-by-play (period totals); the simulated quarter starts from the first live state with a ball holder of that
real quarter (same ten players). Per-team-quarter means are compared with game-clustered 95% intervals of the real side.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import keras                             # noqa: E402
import executor_v2 as E                  # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from game_engine import Engine, STATS    # noqa: E402


def quarter_starts(games, quarters=(1, 2, 3, 4)):
    out = []
    for gi, d in enumerate(games):
        fr = d['frame']
        for q in quarters:
            m = np.flatnonzero((d['quarter'] == q) & d['live'] & (d['holder'] != 0))
            m = m[m >= 2]
            if len(m) == 0:
                continue
            k = int(m[0])
            if d['quarter_time'][k] < 690:               # the first live state of the period should be near its start
                continue
            out.append((gi, k, q))
    return out


def real_quarter_stats(stats_dir, split, game_ids):
    S = pd.read_csv(os.path.join(stats_dir, 'events_shots.csv'))
    F = pd.read_csv(os.path.join(stats_dir, 'events_ft.csv'))
    R = pd.read_csv(os.path.join(stats_dir, 'events_reb.csv'))
    T = pd.read_csv(os.path.join(stats_dir, 'events_tov.csv'))
    rows = []
    for (g, per, team), a in S[S.split == split].groupby(['game', 'period', 'team']):
        if per > 4:
            continue
        f = F[(F.game == g) & (F.period == per) & (F.team == team)]
        r = R[(R.game == g) & (R.period == per) & (R.team == team)]
        t = T[(T.game == g) & (T.period == per) & (T.team == team)]
        tt = T[(T.game == g) & (T.period == per) & (T.team != team)]
        orb = int((r.off == 1).sum())
        rows.append(dict(game=g, period=per, team=team, pts=int(a.pts.sum() + f.made.sum()), fga=len(a), fgm=int(a.made.sum()), fg3a=int(a.is3.sum()), fg3m=int(a[a.is3 == 1].made.sum()),
                         fta=len(f), ftm=int(f.made.sum()), orb=orb, drb=len(r) - orb, tov=len(t), stl=int((tt.stealer > 0).sum()), poss=len(a) + 0.44 * len(f) - orb + len(t)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/rl')
    ap.add_argument('--model', default='rl/model/executor_v4g.keras')
    ap.add_argument('--split', default='holdout')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--out', default='../reports/rl-stage6a')
    ap.add_argument('--configs', default='1.0:1.0:1.0:1.0', help='shoot:pass:tov:foul[:fail_keep[:dead_make]] per config, comma separated')
    ap.add_argument('--dist-mult', default=None, help='json file with the shoot multipliers by distance bin')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-blocks', type=int, default=2400)
    a = ap.parse_args()
    from api import PassValueModel
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games(a.split) if a.split in ('train', 'holdout') else split_games('holdout'))
    qs = quarter_starts(G)
    rng = np.random.default_rng(a.seed)
    qs = [qs[i] for i in np.sort(rng.choice(len(qs), min(a.n, len(qs)), replace=False))]
    starts = [(g, k) for g, k, _ in qs]
    gids = [G[g]['report']['game_id'] for g, k in starts]
    real = real_quarter_stats(a.out, a.split, set(gids))
    real = real[real.apply(lambda r: any((r.game == G[g]['report']['game_id']) and (r.period == q) for g, k, q in qs), axis=1)]
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff, pt = PassValueModel(), PlayTargets()
    RS = [k for k in STATS if k in real.columns]
    R = real[RS].mean()
    ug = real.game.unique()
    bs = []
    for _ in range(200):
        pick = rng.choice(ug, len(ug))
        bs.append(pd.concat([real[real.game == u] for u in pick])[RS].mean())
    ci = pd.DataFrame(bs).quantile([0.025, 0.975])
    print(f'{len(starts)} simulated quarters; {len(real)} real team-quarters from the same games')
    out = dict(real=R.to_dict(), real_ci=ci.to_dict(), configs={})
    for cfg in a.configs.split(','):
        x = [float(v_) for v_ in cfg.split(':')]
        ss, ps_, tm, fm = x[:4]
        fk = x[4] if len(x) > 4 else 0.0
        dm = x[5] if len(x) > 5 else 4.0
        om = x[6] if len(x) > 6 else 1.0
        sst = x[7] if len(x) > 7 else 0.55
        fcl = x[8] if len(x) > 8 else 1.2
        dmult = json.load(open(a.dist_mult)) if a.dist_mult else None
        t0 = time.time()
        eng = Engine(model, diff, plays=pt, seed=a.seed, shoot_scale=ss, pass_scale=ps_, tov_mult=tm, foul_mult=fm, fail_keep=fk, dead_make=dm, dist_mult=dmult, orb_mult=om, steal_share=sst, force_clock=fcl).reset(G, table, starts)
        eng.run(a.max_blocks)
        sm = {k: float(v.sum(1).mean() / 2) for k, v in eng.stats.items()}
        sm['game_clock_left'] = float(eng.game_clock.mean())
        sm['unfinished'] = float(eng.alive.mean())
        out['configs'][cfg] = sm
        sl = pd.DataFrame(eng.shot_log)
        edges = eng.dist_edges
        sim_sh = np.histogram(sl.dist, edges)[0] / len(sl)
        real_s = pd.read_csv(os.path.join(a.out, 'events_shots.csv'))
        real_s = real_s[real_s.split == 'train']
        real_sh = np.histogram(np.clip(real_s.dist, 0, 199), edges)[0] / len(real_s)
        sm['dist_shares_sim'] = sim_sh.tolist()
        sm['dist_shares_real'] = real_sh.tolist()
        print('   shot distance shares sim ', np.round(sim_sh, 3).tolist(), '\n   real                     ', np.round(real_sh, 3).tolist())
        print(f'[{cfg}] {time.time() - t0:.0f}s', json.dumps({k: round(v, 2) for k, v in sm.items() if not isinstance(v, list)}))
    print('real per team-quarter:', json.dumps({k: round(float(v), 2) for k, v in R.items()}))
    json.dump(out, open(os.path.join(a.out, f'engine_validate_{a.split}.json'), 'w'), indent=1, default=float)
    for k in STATS:
        if k in R:
            print(f'  {k:7s} real {R[k]:7.2f} [{ci.loc[0.025, k]:.2f}, {ci.loc[0.975, k]:.2f}]  ' + '  '.join(f'{c}={out["configs"][c][k]:.2f}' for c in out['configs']))


if __name__ == '__main__':
    main()
