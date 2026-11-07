"""Stage 6E: compare simulated and real game scores (replay of the real quarters' ten players and start formations).
    python rl/analyze_replay.py ../reports/rl-stage6e/replay_holdout.json --out ../reports/rl-stage6e/replay_summary.json"""
import json, sys, argparse
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('inp'); ap.add_argument('--out', required=True); a = ap.parse_args()
d = json.load(open(a.inp)); S = pd.DataFrame(d['rows']); R = pd.DataFrame(d['real'])
R = R.assign(team=R.team.astype(np.int64))
# per (game, period): real points of teamA and teamB
rr = R.set_index(['game', 'period', 'team']).pts
S['realA'] = [rr.get((g, p, t), np.nan) for g, p, t in zip(S.game, S.period, S.teamA)]
S['realB'] = [rr.get((g, p, t), np.nan) for g, p, t in zip(S.game, S.period, S.teamB)]
S = S.dropna(subset=['realA', 'realB'])
print('quarters', len(S), 'games', S.game.nunique(), 'reps', S.rep.nunique(), 'unfinished', S.unfinished.mean().round(3))
# quarter level
q = S.groupby(['game', 'period', 'teamA', 'teamB']).agg(ptsA=('ptsA', 'mean'), ptsB=('ptsB', 'mean'), realA=('realA', 'first'), realB=('realB', 'first')).reset_index()
res = dict(n_quarters=int(len(q)), n_games=int(q.game.nunique()))
for nm in ('A', 'B'):
    res[f'quarter_points_corr_{nm}'] = float(np.corrcoef(q[f'pts{nm}'], q[f'real{nm}'])[0, 1])
res['quarter_total_mean_sim'] = float((q.ptsA + q.ptsB).mean()); res['quarter_total_mean_real'] = float((q.realA + q.realB).mean())
res['quarter_total_corr'] = float(np.corrcoef(q.ptsA + q.ptsB, q.realA + q.realB)[0, 1])
res['quarter_margin_corr'] = float(np.corrcoef(q.ptsA - q.ptsB, q.realA - q.realB)[0, 1])
# game level: orient by the lower team id
q['lo'] = np.minimum(q.teamA, q.teamB)
def orient(r, col):
    return r[col + 'A'] if r.teamA == r.lo else r[col + 'B']
q['sim_lo'] = q.apply(lambda r: r.ptsA if r.teamA == r.lo else r.ptsB, axis=1); q['sim_hi'] = q.ptsA + q.ptsB - q.sim_lo
q['real_lo'] = q.apply(lambda r: r.realA if r.teamA == r.lo else r.realB, axis=1); q['real_hi'] = q.realA + q.realB - q.real_lo
# only games with all four quarters present
cnt = q.groupby('game').period.nunique(); full = cnt[cnt == 4].index
G = q[q.game.isin(full)].groupby('game')[['sim_lo', 'sim_hi', 'real_lo', 'real_hi']].sum()
G['sim_m'] = G.sim_lo - G.sim_hi; G['real_m'] = G.real_lo - G.real_hi; G['sim_t'] = G.sim_lo + G.sim_hi; G['real_t'] = G.real_lo + G.real_hi
rng = np.random.default_rng(0)
def boot(f, n=400):
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(G), len(G)); out.append(f(G.iloc[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
corr = lambda g: np.corrcoef(g.sim_m, g.real_m)[0, 1]
slope = lambda g: np.polyfit(g.sim_m, g.real_m, 1)[0]
win = lambda g: float((np.sign(g.sim_m) == np.sign(g.real_m)).mean())
res['games_with_four_quarters'] = int(len(G))
res['game_margin'] = dict(corr=float(corr(G)), corr_ci95=boot(corr), slope_real_on_sim=float(slope(G)), slope_ci95=boot(slope), winner_agreement=win(G), winner_ci95=boot(win), sd_sim=float(G.sim_m.std()), sd_real=float(G.real_m.std()))
res['game_total'] = dict(mean_sim=float(G.sim_t.mean()), mean_real=float(G.real_t.mean()), corr=float(np.corrcoef(G.sim_t, G.real_t)[0, 1]), sd_sim=float(G.sim_t.std()), sd_real=float(G.real_t.std()))
# calibration of the sign by tercile of simulated margin
G['bin'] = pd.qcut(G.sim_m, 4, labels=False, duplicates='drop')
res['margin_calibration'] = [dict(bin=int(b), sim_mean=float(g.sim_m.mean()), real_mean=float(g.real_m.mean()), n=int(len(g))) for b, g in G.groupby('bin')]
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, 'w'), indent=1)
