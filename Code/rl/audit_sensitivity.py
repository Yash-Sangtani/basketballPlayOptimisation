"""Stage 6F-B: do the headline conclusions survive perturbations of the engine constants? (pre-registered)
    python rl/audit_sensitivity.py --out ../reports/rl-stage6f/sensitivity.json
Each perturbed engine has its own imitation baseline; policies are frozen; 100 holdout chains of 180 s."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import league as L
import league_eval as LE


def perturbations(cal):
    dm = list(cal['dist_mult'])
    dm3 = list(dm); [dm3.__setitem__(j, dm[j] * 1.25) for j in (4, 5)]          # 22-25 and 25-28 ft bins
    return {
        'v8': {},
        'tov x1.25': dict(tov_mult=cal['tov_mult'] * 1.25),
        'tov x0.8': dict(tov_mult=cal['tov_mult'] * 0.8),
        'steal share +0.15': dict(steal_share=min(cal['steal_share'] + 0.15, 0.95)),
        'foul x1.25': dict(foul_mult=cal['foul_mult'] * 1.25),
        'orb x1.15': dict(orb_mult=cal['orb_mult'] * 1.15),
        'threes shoot x1.25': dict(dist_mult=dm3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=100); ap.add_argument('--blocks', type=int, default=450); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl'); ap.add_argument('--split', default='holdout')
    ap.add_argument('--policies', default='def_seed0=../reports/rl-stage6c/run3/ckpt_0060.pkl,loose_league=../reports/rl-stage6d/run3/ckpt_0100.pkl,tight_league=../reports/rl-stage6d/run_tight/ckpt_0050.pkl')
    a = ap.parse_args()
    st, G, table, starts, gi = LE.setup(a)
    pols = {x.split('=')[0]: LE.agents(st, x.split('=')[1], 1) for x in a.policies.split(',')}
    res = {}
    t0 = time.time()
    for pname, over in perturbations(st['cal']).items():
        base = L.run_match(st, G, table, starts, 5, a.blocks, (None, None), (None, None), **over)
        mb = L.match_metrics(base)
        r = dict(base_ppp=mb['ppp'], base_tov=mb['tov_per_poss'], base_stl=mb['stl_per_poss'], policies={})
        for nm, A in pols.items():
            e = L.run_match(st, G, table, starts, 5, a.blocks, A, A, **over)
            m = L.match_metrics(e)
            d, ci = L.paired_ci(e, base, gi)
            r['policies'][nm] = dict(ppp=m['ppp'], diff=d, ci95=ci, ratios={k: m[k] / mb[k] for k in ('stl_per_poss', 'tov_per_poss', 'fta_per_poss', 'fg3a_per_poss')})
        res[pname] = r
        print(f"{pname:20s} base {mb['ppp']:.3f}  " + '  '.join(f"{nm} {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for nm, v in r['policies'].items()) + f'  ({time.time() - t0:.0f}s)', flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
