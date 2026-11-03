"""Evaluation of engine agents on the HOLDOUT games (never used for training or calibration).

    # one agent configuration against the imitation engine, paired starts and seeds
    python rl/league_eval.py gain --ckpt X.pkl --n 200 --blocks 450 --out ../reports/rl-stage6c/eval_X.json
    # matchup matrix of league checkpoints (+ the imitation engine), both side orderings
    python rl/league_eval.py matrix --ckpts name=path,name=path --n 64 --blocks 300 --out ../reports/rl-stage6d/matrix.json
    # exploitability probe: a fresh best response (trained against the frozen target) vs the league policy itself, both against the frozen target
    python rl/league_eval.py br --target X.pkl --br Y.pkl --ref Z.pkl --side def --n 200 --out ...

Metrics: points per completed possession (pooled over both teams' possessions), per-chain points difference, steals / turnovers / shot mix per possession, with
game-clustered bootstrap intervals over the source games of the chains.
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from engine_validate import quarter_starts   # noqa: E402
import league as L                       # noqa: E402


def setup(a):
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games(a.split))
    qs = quarter_starts(G)
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(len(qs), min(a.n, len(qs)), replace=False))
    starts = [(qs[j][0], qs[j][1]) for j in pick]
    return st, G, table, starts, np.array([s[0] for s in starts])


def agents(st, path, seed=0):
    if path in (None, 'imitation'):
        return (None, None)
    ck = pickle.load(open(path, 'rb'))
    o, d = L.make_agents(st, ck, learn=False, seed=seed)
    return (o if ck.get('off') is not None else None, d if ck.get('def') is not None else None)


def run(st, G, table, starts, seed, blocks, A, B):
    t0 = time.time()
    eng = L.run_match(st, G, table, starts, seed, blocks, A=A, B=B)
    return eng, time.time() - t0


def ppp_by_team(eng):
    log = np.array(eng.poss_log, float).reshape(-1, 4)
    return [float(log[log[:, 2] == t, 3].mean()) for t in (0, 1)]


def boot_diff(eng_a, eng_b, gi, n_boot=400, seed=0):
    return L.paired_ci(eng_a, eng_b, gi, n_boot, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gain', 'matrix', 'br', 'recheck'])
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--ckpts', default=None)
    ap.add_argument('--also', default=None, help='gain mode: second checkpoint (e.g. the zero-residual prior) evaluated on the same starts; the agent is also compared with it')
    ap.add_argument('--target', default=None); ap.add_argument('--br', default=None); ap.add_argument('--ref', default=None)
    ap.add_argument('--side', default='def')
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--blocks', type=int, default=450); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl'); ap.add_argument('--split', default='holdout')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    st, G, table, starts, gi = setup(a)
    res = dict(n=len(starts), blocks=a.blocks, split=a.split)
    if a.mode == 'gain':
        # the agent on both teams (defence and/or offence it was trained for) against the all-imitation engine
        A = agents(st, a.ckpt, 1)
        base, t0 = run(st, G, table, starts, 5, a.blocks, (None, None), (None, None))
        ag, t1 = run(st, G, table, starts, 5, a.blocks, A, A)
        d, ci = boot_diff(ag, base, gi)
        res.update(base=L.match_metrics(base), agent=L.match_metrics(ag), ppp_diff=d, ppp_diff_ci95=ci, base_formation=L.formation_stats(base), agent_formation=L.formation_stats(ag))
        print(f"agent {L.match_metrics(ag)['ppp']:.3f} vs imitation {L.match_metrics(base)['ppp']:.3f}  diff {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        if a.also:
            B2 = agents(st, a.also, 2)
            rf, _ = run(st, G, table, starts, 5, a.blocks, B2, B2)
            d2, ci2 = boot_diff(rf, base, gi)
            d3, ci3 = boot_diff(ag, rf, gi)
            res.update(ref=L.match_metrics(rf), ref_diff_vs_imitation=d2, ref_diff_vs_imitation_ci95=ci2, agent_minus_ref=d3, agent_minus_ref_ci95=ci3, ref_formation=L.formation_stats(rf))
            print(f"reference {L.match_metrics(rf)['ppp']:.3f} vs imitation: {d2:+.3f} [{ci2[0]:+.3f}, {ci2[1]:+.3f}];  agent minus reference {d3:+.3f} [{ci3[0]:+.3f}, {ci3[1]:+.3f}]")
    elif a.mode == 'recheck':
        # several agents (both teams play them) against the imitation engine and the zero-residual prior on the same starts: used to re-evaluate trained policies in a recalibrated engine
        base, _ = run(st, G, table, starts, 5, a.blocks, (None, None), (None, None))
        res['base'] = L.match_metrics(base)
        if a.also:
            B2 = agents(st, a.also, 2)
            rf, _ = run(st, G, table, starts, 5, a.blocks, B2, B2)
            res['ref'] = L.match_metrics(rf)
        res['agents'] = {}
        for item in a.ckpts.split(','):
            nm, path = item.split('=')
            A = agents(st, path, 1)
            e, _ = run(st, G, table, starts, 5, a.blocks, A, A)
            m = L.match_metrics(e)
            d, ci = boot_diff(e, base, gi)
            r = dict(metrics=m, ppp_diff=d, ppp_diff_ci95=ci, ratios={k: m[k] / res['base'][k] for k in ('stl_per_poss', 'tov_per_poss', 'fta_per_poss', 'fg3a_per_poss', 'fga_per_poss', 'n_poss')})
            if a.also:
                d2, ci2 = boot_diff(e, rf, gi)
                r.update(minus_ref=d2, minus_ref_ci95=ci2)
            res['agents'][nm] = r
            print(f"{nm:10s} ppp {m['ppp']:.3f}  vs imitation {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" + (f"  vs prior {r['minus_ref']:+.3f} [{ci2[0]:+.3f}, {ci2[1]:+.3f}]" if a.also else '') + '  ratios ' + ' '.join(f"{k[:5]} {v:.2f}" for k, v in r['ratios'].items()), flush=True)
            json.dump(res, open(a.out, 'w'), indent=1, default=float)
    elif a.mode == 'matrix':
        names = ['imitation'] + [x.split('=')[0] for x in a.ckpts.split(',')]
        paths = [None] + [x.split('=')[1] for x in a.ckpts.split(',')]
        m = len(names)
        M = np.full((m, m), np.nan)
        P = {}
        for i in range(m):
            for j in range(m):
                if i == j:
                    M[i, j] = 0.0
                    continue
                if i > j:
                    continue
                Ai, Bj = agents(st, paths[i], 1), agents(st, paths[j], 2)
                e1, _ = run(st, G, table, starts, 21, a.blocks, Ai, Bj)          # i as roster team A, j as B
                e2, _ = run(st, G, table, starts, 22, a.blocks, Bj, Ai)          # sides swapped
                c1 = e1.score[:, 0] - e1.score[:, 1]
                c2 = e2.score[:, 1] - e2.score[:, 0]
                d1, d2 = c1.mean(), c2.mean()
                n1 = max(float(e1.stats['poss'].sum() + e2.stats['poss'].sum()) / 2, 1.0)
                per100 = 100.0 * len(starts) / n1
                M[i, j] = per100 * (d1 + d2) / 2                                  # points per 100 possessions of i over j
                M[j, i] = -M[i, j]
                rng_ = np.random.default_rng(0)
                ug = np.unique(gi)
                bs = []
                for _ in range(400):
                    w = np.concatenate([np.flatnonzero(gi == u) for u in rng_.choice(ug, len(ug))])
                    bs.append(per100 * (c1[w].mean() + c2[w].mean()) / 2)
                ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                P[f'{names[i]}_vs_{names[j]}'] = dict(diff_ordered=float(d1), diff_swapped=float(d2), pts_per_100_poss=float(M[i, j]), ci95=ci, i_ppp_off=[ppp_by_team(e1)[0], ppp_by_team(e2)[1]], j_ppp_off=[ppp_by_team(e1)[1], ppp_by_team(e2)[0]],
                                                    i_metrics=L.match_metrics(e1), j_metrics=L.match_metrics(e2))
                print(f'{names[i]:>12s} vs {names[j]:<12s} {M[i, j]:+6.1f} pts/100 poss [{ci[0]:+.1f}, {ci[1]:+.1f}] (ordered {d1:+.2f}, swapped {d2:+.2f} per chain)', flush=True)
        res.update(names=names, matrix=M.tolist(), pairs=P)
    else:
        # exploitability: points difference of the best response vs the frozen target, minus that of the reference policy vs the same frozen target, on the same starts
        tgt = agents(st, a.target, 3)
        br, ref = agents(st, a.br, 4), agents(st, a.ref, 5)
        if a.side == 'def':
            Abr, Aref = (None, br[1]), (None, ref[1])
        elif a.side == 'off':
            Abr, Aref = (br[0], None), (ref[0], None)
        else:
            Abr, Aref = br, ref
        e_br, _ = run(st, G, table, starts, 31, a.blocks, Abr, tgt)
        e_ref, _ = run(st, G, table, starts, 31, a.blocks, Aref, tgt)
        e_br2, _ = run(st, G, table, starts, 32, a.blocks, tgt, Abr)
        e_ref2, _ = run(st, G, table, starts, 32, a.blocks, tgt, Aref)
        gain = ((e_br.score[:, 0] - e_br.score[:, 1]) + (e_br2.score[:, 1] - e_br2.score[:, 0]) - (e_ref.score[:, 0] - e_ref.score[:, 1]) - (e_ref2.score[:, 1] - e_ref2.score[:, 0])) / 2
        rng = np.random.default_rng(0)
        ug = np.unique(gi)
        bs = [gain[np.concatenate([np.flatnonzero(gi == u) for u in rng.choice(ug, len(ug))])].mean() for _ in range(400)]
        res.update(gain_points_per_chain=float(gain.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   br_diff=float(((e_br.score[:, 0] - e_br.score[:, 1]).mean() + (e_br2.score[:, 1] - e_br2.score[:, 0]).mean()) / 2),
                   ref_diff=float(((e_ref.score[:, 0] - e_ref.score[:, 1]).mean() + (e_ref2.score[:, 1] - e_ref2.score[:, 0]).mean()) / 2))
        print(f"best response gain over the reference vs the frozen target: {res['gain_points_per_chain']:+.2f} points per chain [{res['ci95'][0]:+.2f}, {res['ci95'][1]:+.2f}]  (BR diff {res['br_diff']:+.2f}, ref diff {res['ref_diff']:+.2f})")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
