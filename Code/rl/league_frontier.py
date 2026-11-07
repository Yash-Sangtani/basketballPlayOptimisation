"""Strength-versus-realism frontier of a league run: every listed checkpoint plays both teams (offence + defence agents) on the same holdout chains; compared with the imitation engine.
    python rl/league_frontier.py --run ../reports/rl-stage6d/run3 --its 0,10,20,30,40,60,80,100 --n 100 --blocks 450 --out ../reports/rl-stage6d/frontier.json
Reported per checkpoint: points per possession, the pre-registered realism ratios to the imitation engine (steals, turnovers, free throws per possession), threes per possession, rim share."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import league as L
import league_eval as LE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--its', default='0,10,20,30,40,60,80,100'); ap.add_argument('--n', type=int, default=100); ap.add_argument('--blocks', type=int, default=450)
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl'); ap.add_argument('--split', default='holdout')
    a = ap.parse_args()
    st, G, table, starts, gi = LE.setup(a)
    base, _ = LE.run(st, G, table, starts, 5, a.blocks, (None, None), (None, None))
    mb = L.match_metrics(base)
    res = dict(base=mb, ckpts={})
    for it in [int(x) for x in a.its.split(',')]:
        A = LE.agents(st, os.path.join(a.run, f'ckpt_{it:04d}.pkl'), 1)
        e, _ = LE.run(st, G, table, starts, 5, a.blocks, A, A)
        m = L.match_metrics(e)
        d, ci = L.paired_ci(e, base, gi)
        m.update(ppp_diff=d, ppp_diff_ci95=ci, ratios={k: m[k] / mb[k] for k in ('stl_per_poss', 'tov_per_poss', 'fta_per_poss', 'fg3a_per_poss', 'fga_per_poss')}, formation=L.formation_stats(e))
        res['ckpts'][it] = m
        r = m['ratios']
        print(f"it {it:3d} ppp {m['ppp']:.3f} ({d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}])  ratios stl {r['stl_per_poss']:.2f} tov {r['tov_per_poss']:.2f} fta {r['fta_per_poss']:.2f} 3pa {r['fg3a_per_poss']:.2f} fga {r['fga_per_poss']:.2f}  rim {m.get('rim_share', 0):.3f}", flush=True)
        json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
