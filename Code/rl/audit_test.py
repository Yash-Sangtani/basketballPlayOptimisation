"""Stage 6F-D: one-time confirmation on the TEST split (130 games). Refuses to run twice. Pre-registered claims C1-C3 in reports/rl-stage6f-prereg.md.
    python rl/audit_test.py --confirm-test --out ../reports/rl-stage6f/test_once.json
Frozen: engine v8, executor v5g, the five checkpoints. Nothing is tuned."""
import argparse, json, os, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, pandas as pd
from load import load_games
from profiles import build_profiles
from train_executor import fit_val_games
from engine_validate import quarter_starts, real_quarter_stats
import game_stats as GS
import league as L
import league_eval as LE


def test_events(G, out_dir):
    S, F, R, T = [], [], [], []
    for d in G:
        s, f, r, t, fo, df = GS.parse_game(d['report']['game_id'], 'test')
        S += s; F += f; R += r; T += t
    os.makedirs(out_dir, exist_ok=True)
    for nm, rows in (('shots', S), ('ft', F), ('reb', R), ('tov', T)):
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'events_{nm}.csv'), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--confirm-test', action='store_true')
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--blocks', type=int, default=450); ap.add_argument('--n-quarters', type=int, default=200)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl'); ap.add_argument('--test-data', default='../data/rl_test')
    a = ap.parse_args()
    if not a.confirm_test:
        raise SystemExit('the test split is scored once: pass --confirm-test')
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} exists: the test split has been scored for Stage 6; it is not scored twice')
    open(a.out + '.lock', 'w').write('started')
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    names = [l.strip().replace('.7z', '') for l in open(os.path.join(HERE, '..', 'splits', 'test_games.txt')) if l.strip()]
    G = load_games(a.test_data, names)
    res = dict(n_games=len(G))
    t0 = time.time()
    # D1: engine validation on test quarters (v8 calibration), descriptive
    ev_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), 'test_events')
    test_events(G, ev_dir)
    real = real_quarter_stats(ev_dir, 'test', None)
    qs = quarter_starts(G)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(qs), min(a.n_quarters, len(qs)), replace=False))
    qsel = [qs[j] for j in pick]
    gid = {G[g]['report']['game_id']: g for g in range(len(G))}
    eng = L.run_match(st, G, table, [(g, k) for g, k, q in qsel], 4, 1800)
    keys = ['pts', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'orb', 'drb', 'tov', 'stl', 'poss']
    sim = {k: float(eng.stats[k].sum(1).mean() / 2) for k in keys}
    sel_keys = {(G[g]['report']['game_id'], q) for g, k, q in qsel}
    rr = real[[(r.game, r.period) in sel_keys for r in real.itertuples()]]
    ug = rr.game.unique()
    inside = {}
    bs = pd.DataFrame([pd.concat([rr[rr.game == u] for u in rng.choice(ug, len(ug))])[keys].mean() for _ in range(200)])
    for k in keys:
        lo, hi = bs[k].quantile([0.025, 0.975])
        inside[k] = dict(real=float(rr[k].mean()), ci95=[float(lo), float(hi)], sim=sim[k], inside=bool(lo <= sim[k] <= hi))
    res['engine_validation'] = dict(n_quarters=len(qsel), stats=inside, n_inside=int(sum(v['inside'] for v in inside.values())))
    print(f"D1 engine on test quarters: {res['engine_validation']['n_inside']}/12 inside ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(a.out + '.partial', 'w'), indent=1, default=float)
    # D2-D4: policies against the imitation engine on test chains
    class NS: pass
    qs2 = quarter_starts(G)
    pick = np.sort(np.random.default_rng(0).choice(len(qs2), min(a.n, len(qs2)), replace=False))
    starts = [(qs2[j][0], qs2[j][1]) for j in pick]
    gi = np.array([s[0] for s in starts])
    base = L.run_match(st, G, table, starts, 5, a.blocks)
    mb = L.match_metrics(base)
    res['base'] = mb
    prior = LE.agents(st, '../reports/rl-stage6c/prior_def_ckpt.pkl', 2)
    rf = L.run_match(st, G, table, starts, 5, a.blocks, prior, prior)
    res['prior'] = dict(metrics=L.match_metrics(rf), diff=L.paired_ci(rf, base, gi))
    pols = {'def_seed0': '../reports/rl-stage6c/run3/ckpt_0060.pkl', 'def_seed3': '../reports/rl-stage6c/run4/ckpt_0060.pkl', 'def_seed5': '../reports/rl-stage6c/run5/ckpt_0060.pkl',
            'loose_league': '../reports/rl-stage6d/run3/ckpt_0100.pkl', 'tight_league': '../reports/rl-stage6d/run_tight/ckpt_0050.pkl'}
    res['policies'] = {}
    for nm, path in pols.items():
        A = LE.agents(st, path, 1)
        e = L.run_match(st, G, table, starts, 5, a.blocks, A, A)
        m = L.match_metrics(e)
        d, ci = L.paired_ci(e, base, gi)
        d2, ci2 = L.paired_ci(e, rf, gi)
        res['policies'][nm] = dict(metrics=m, diff=d, ci95=ci, minus_prior=d2, minus_prior_ci95=ci2, ratios={k: m[k] / mb[k] for k in ('stl_per_poss', 'tov_per_poss', 'fta_per_poss', 'fg3a_per_poss', 'fga_per_poss', 'n_poss')})
        print(f"{nm:13s} ppp {m['ppp']:.3f} vs imitation {mb['ppp']:.3f}: {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  ratios " + ' '.join(f'{k[:5]} {v:.2f}' for k, v in res['policies'][nm]['ratios'].items()), flush=True)
        json.dump(res, open(a.out + '.partial', 'w'), indent=1, default=float)
    ds = [res['policies'][k]['diff'] for k in ('def_seed0', 'def_seed3', 'def_seed5')]
    ci_clear = [res['policies'][k]['ci95'][1] < 0 for k in ('def_seed0', 'def_seed3', 'def_seed5')]
    lg, tg = res['policies']['loose_league'], res['policies']['tight_league']
    res['claims'] = dict(
        C1_mean_defence_diff_below_0=bool(np.mean(ds) < 0), C1_mean=float(np.mean(ds)), C1b_at_least_two_seeds_clear=bool(sum(ci_clear) >= 2), C1b_n_clear=int(sum(ci_clear)),
        C2_loose_ppp_above_imitation=bool(lg['ci95'][0] > 0), C2b_loose_steals_tov_outside_15pct=bool(all(abs(lg['ratios'][k] - 1) > 0.15 for k in ('stl_per_poss', 'tov_per_poss'))),
        C3_tight_within_15pct=bool(all(abs(tg['ratios'][k] - 1) <= 0.15 for k in ('stl_per_poss', 'tov_per_poss', 'fta_per_poss'))), C3b_tight_ppp_not_different=bool(tg['ci95'][0] <= 0 <= tg['ci95'][1]))
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps(res['claims'], indent=1))


if __name__ == '__main__':
    main()
