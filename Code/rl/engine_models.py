"""Stage 6A.2: event models for the game engine, fitted on the 401 RL-train games (the holdout is only used to check them).

    cd Code && python rl/engine_models.py --data ../data/rl --stats ../reports/rl-stage6a

Writes model/engine_params.json:
  orb          P(offensive rebound | miss) by shot type and distance bin (play-by-play)
  ft           P(fouled attempt) for 2-pt and 3-pt attempts, trip-size distribution, and-one rate per make, FT% per player (shrunk), league FT%
  rebounder    conditional-logit weights choosing the rebounder among the ten players from tracking at the rebound time (distance to the basket,
               offence/defence, position code, distance squared)
  turnover     logistic hazard of losing the ball without a pass or shot (per 0.2-s state with the ball held), on nearest-defender distance, speed,
               time in possession, shot clock, backcourt flag
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy.optimize import minimize      # noqa: E402
from load import load_games              # noqa: E402
from splits import games as split_games  # noqa: E402
from executor_data import pos_codes      # noqa: E402

BASKET = np.array([5.35, 25.0])
D_BINS = [-1, 2, 6, 12, 20, 23, 99]


def fit_basic(stats):
    S = pd.read_csv(os.path.join(stats, 'events_shots.csv'))
    F = pd.read_csv(os.path.join(stats, 'events_ft.csv'))
    R = pd.read_csv(os.path.join(stats, 'events_reb.csv'))
    S, F, R = (x[x.split == 'train'] for x in (S, F, R))
    m = R.dropna(subset=['off'])
    orb = {}
    for is3 in (0, 1):
        sub = m[m.is3 == is3]
        b = pd.cut(sub.dist, D_BINS)
        g = sub.groupby(b, observed=False).off.agg(['mean', 'count'])
        orb[str(is3)] = [float(v) if c > 100 else float(sub.off.mean()) for v, c in zip(g['mean'], g['count'])]
    F = F.sort_values(['game', 't'])
    trips = F[F.k == 1]
    made = S[S.made == 1]
    mk = {(g, t_): np.sort(x.t.values) for (g, t_), x in made.groupby(['game', 'team'])}

    def after_make(r):
        a = mk.get((r.game, r.team))
        if a is None:
            return False
        i = np.searchsorted(a, r.t, side='right') - 1
        return i >= 0 and r.t - a[i] <= 3
    and1 = trips.apply(after_make, axis=1)
    nontrips = trips[~and1]
    fp = F.groupby('player').made.agg(['sum', 'count'])
    league = float(F.made.mean())
    ftp = ((fp['sum'] + 40 * league) / (fp['count'] + 40)).to_dict()
    n2 = (S.is3 == 0).sum()
    n3 = (S.is3 == 1).sum()
    # fouled attempts are not logged as field-goal attempts: attempts = logged FGA + fouled-attempt trips (size 2 or 3, excluding and-ones)
    f2 = ((nontrips.n == 2).sum() + (nontrips.n == 1).sum()) / (n2 + (nontrips.n != 3).sum())
    f3 = (nontrips.n == 3).sum() / (n3 + (nontrips.n == 3).sum())
    return dict(orb=orb, orb_bins=D_BINS, ft=dict(p_foul_2pt=float(f2), p_foul_3pt=float(f3), trip_sizes={int(k): float(v) for k, v in nontrips.n.value_counts(normalize=True).items()},
                                                  and_one_per_make=float(and1.sum() / len(made)), ft_pct=float(league), player_ft={str(int(k)): float(v) for k, v in ftp.items()}))


def rebound_data(games, S, R):
    """Rows: for every missed tracked shot matched to a play-by-play miss and its rebound, the ten players' features at the rebound time and the rebounder slot."""
    rows = []
    sh = S.set_index('game')
    for d in games:
        gid = d['report']['game_id']
        if gid not in sh.index:
            continue
        shg = sh.loc[[gid]] if not isinstance(sh.loc[gid], pd.Series) else sh.loc[[gid]]
        rg = R[(R.game == gid) & R.off.notna()]
        if len(rg) == 0:
            continue
        fr = d['frame']
        shots = d['shots'][d['shots'].made == 0]
        for r in shots.itertuples():
            k = int(np.searchsorted(fr, r.frame, side='right') - 1)
            if k < 0 or k + 12 >= len(fr) or r.frame - fr[k] > 10:
                continue
            el = (int(d['quarter'][k]) - 1) * 720 + (720 - float(d['quarter_time'][k])) if d['quarter'][k] <= 4 else None
            if el is None:
                continue
            cand = rg[(rg.t >= el - 1) & (rg.t <= el + 8)]
            cand = cand.sort_values('t')
            if len(cand) == 0:
                continue
            c = cand.iloc[0]
            kk = k + 8
            slots = np.flatnonzero(d['ids'][kk] == c.player)
            if len(slots) == 0:
                continue
            ids = d['ids'][kk]
            pos = d['pos'][kk]
            pc = pos_codes(d, ids[None])[0]
            rows.append(dict(pos=pos, pc=pc, slot=int(slots[0]), off=int(c.off), dist=c.dist))
    return rows


def feats(pos, pc):
    dist = np.hypot(*(pos - BASKET).T) / 10.0                  # (10,)
    off = np.r_[np.ones(5), np.zeros(5)]
    X = np.stack([dist, dist ** 2, off, off * (pc == 4), off * (pc == 3), (1 - off) * (pc == 4), (1 - off) * (pc == 3), (pc == 0) * 1.0], axis=1)
    return X


def fit_rebounder(rows):
    X = np.stack([feats(r['pos'], r['pc']) for r in rows])
    y = np.array([r['slot'] for r in rows])

    def nll(w):
        z = X @ w
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean() + 1e-3 * (w ** 2).sum()
    res = minimize(nll, np.zeros(X.shape[2]), method='L-BFGS-B')
    return res.x, X, y


def eval_rebounder(w, rows):
    X = np.stack([feats(r['pos'], r['pc']) for r in rows])
    y = np.array([r['slot'] for r in rows])
    z = X @ w
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return dict(n=int(len(y)), top1=float((p.argmax(1) == y).mean()), logloss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()), baseline_logloss=float(np.log(10)),
                orb_share_pred=float(p[:, :5].sum(1).mean()), orb_share_real=float((y < 5).mean()))


def fit_turnover(games):
    """Per-state logistic hazard: the offence loses the ball within the next 0.2 s without a pass or shot (steal, strip, travel, offensive foul...)."""
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for d in games:
        n = len(d['frame'])
        fr = d['frame']
        pf = np.sort(d['passes']['frame'].values) if len(d['passes']) else np.array([])
        sf = np.sort(d['shots']['frame'].values) if len(d['shots']) else np.array([])
        ev = np.sort(np.r_[pf, sf])
        pid = d['poss_id']
        ok = d['live'] & (d['holder'] != 0)
        for k in np.flatnonzero(ok[:-8]):
            if fr[k + 1] - fr[k] != 5:
                continue
            j = np.searchsorted(ev, fr[k] - 10)
            near_ev = j < len(ev) and ev[j] <= fr[k + 1] + 10
            if near_ev:
                continue                                  # a pass or shot is leaving: not a non-pass loss
            lost = (pid[k + 1] != pid[k]) and (d['ctrl_team'][k + 1] != d['ctrl_team'][k])
            h = np.flatnonzero(d['ids'][k, :5] == d['holder'][k])
            if len(h) == 0:
                continue
            hp = d['pos'][k, h[0]]
            dd = np.sort(np.hypot(*(d['pos'][k, 5:] - hp).T))
            sp = np.hypot(*d['vel'][k, h[0]])
            X.append([dd[0], dd[1], sp, d['t_in_poss'][k] / 24.0, float(d['frontcourt'][k]), 1.0 / (0.5 + dd[0])])
            y.append(int(lost))
    X, y = np.array(X), np.array(y)
    lr = LogisticRegression(C=10.0, max_iter=500).fit(X, y)
    return dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), base_rate=float(y.mean()), n=int(len(y)), n_pos=int(y.sum()), feature_names=['def1', 'def2', 'speed', 't_in_poss', 'frontcourt', 'inv_def1'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'model', 'engine_params.json'))
    a = ap.parse_args()
    P = fit_basic(a.stats)
    S = pd.read_csv(os.path.join(a.stats, 'events_shots.csv'))
    R = pd.read_csv(os.path.join(a.stats, 'events_reb.csv'))
    tr = load_games(a.data, split_games('train'))
    rows = rebound_data(tr, S, R)
    w, _, _ = fit_rebounder(rows)
    ho = rebound_data(load_games(a.data, split_games('holdout')), S, R)
    P['rebounder'] = dict(w=w.tolist(), train=eval_rebounder(w, rows), holdout=eval_rebounder(w, ho))
    print('rebounder', json.dumps(P['rebounder']['holdout']), flush=True)
    P['turnover'] = fit_turnover(tr)
    print('turnover hazard', json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in P['turnover'].items() if k != 'feature_names'}), flush=True)
    json.dump(P, open(a.out, 'w'), indent=1)
    print('p_foul 2pt/3pt', round(P['ft']['p_foul_2pt'], 4), round(P['ft']['p_foul_3pt'], 4), 'trip sizes', P['ft']['trip_sizes'], 'and-one', round(P['ft']['and_one_per_make'], 4), 'FT%', round(P['ft']['ft_pct'], 3))
    print('orb 2pt', np.round(P['orb']['0'], 3), '3pt', np.round(P['orb']['1'], 3))


if __name__ == '__main__':
    main()
