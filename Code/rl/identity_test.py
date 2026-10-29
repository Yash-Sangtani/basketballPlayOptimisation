"""Stage 1: does knowing WHICH player it is help predict what he does next? (movement and ball-handler decisions)

Two tasks on half-court live possession states (frontcourt, at least 1 s after crossing midcourt, someone holds the ball),
sampled at 1 Hz, TRAIN games only (the holdout and test games are not read):
  A  movement: each offensive player's displacement over the next 1 s (dx, dy), from his situation
  B  decision: the ball handler's action in the next 2 s: shoot / pass / keep the ball

Model variants (gradient boosting, GroupKFold by game):
  M0 situation only | M1 + position (G/F/C) | M2 + player identity | M3 + SHUFFLED identity (control: same capacity, no information)
Identity is the player id as a categorical feature for the 250 players with the most rows (all others share one 'other' level).

    cd Code && python rl/identity_test.py --data DIR --n-games 150 --out ../reports/rl-identity
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor   # noqa: E402
from features import holder_row, player_row, roster_positions    # noqa: E402
from load import load_games             # noqa: E402
from splits import games as split_games  # noqa: E402

ONLY = ''
STRIDE = 5            # states between samples: 5 Hz -> 1 Hz
H_MOVE = 5            # states ahead for the movement target (1 s)
H_DEC_FRAMES = 50     # 25 Hz frames ahead for the decision label (2 s)
N_CAT = 250
HGB_R = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, random_state=0)
HGB_C = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=1.0, random_state=0)


def log(*a):
    print(*a, flush=True)


def build_rows(games_data):
    """-> (move DataFrame, decision DataFrame); each row carries game, player id, position code and features."""
    mv, dc = [], []
    for d in games_data:
        posmap = roster_positions(d['roster'])
        n = len(d['frame'])
        passes, shots = d['passes'], d['shots']
        sh_by = {int(p): np.sort(g['frame'].values) for p, g in shots.groupby('shooter')} if len(shots) else {}
        ps_by = {int(p): np.sort(g[g['live'] == 1]['frame'].values) for p, g in passes.groupby('passer')} if len(passes) else {}
        ok = d['live'] & d['frontcourt'] & (d['t_front'] >= 1.0) & (d['holder'] != 0)
        for k in range(0, n - H_MOVE, STRIDE):
            if not ok[k]:
                continue
            k2 = k + H_MOVE
            same = (d['frame'][k2] - d['frame'][k] == H_MOVE * 5 and d['poss_id'][k2] == d['poss_id'][k] and (d['ids'][k2] == d['ids'][k]).all())
            if same:
                for j in range(5):
                    f = player_row(d, k, j)
                    pid = int(d['ids'][k, j])
                    disp = d['pos'][k2, j] - d['pos'][k, j]
                    mv.append(dict(game=d['game'], player=pid, pos=posmap.get(pid, 2), dx=disp[0], dy=disp[1], **f))
            h = holder_row(d, k)
            if h is not None:
                j, f = h
                pid = int(d['ids'][k, j])
                fr0 = int(d['frame'][k])
                s = sh_by.get(pid, np.array([]))
                p = ps_by.get(pid, np.array([]))
                shoot = bool(((s > fr0) & (s <= fr0 + H_DEC_FRAMES)).any())
                passed = bool(((p > fr0 - 5) & (p <= fr0 + H_DEC_FRAMES)).any())
                dc.append(dict(game=d['game'], player=pid, pos=posmap.get(pid, 2), action=0 if shoot else (1 if passed else 2), **f))
    return pd.DataFrame(mv), pd.DataFrame(dc)


def add_identity(df, rng):
    """player_cat: top-N players by rows (others 'other'); player_shuf: the same ids assigned to random rows (a no-information control)."""
    top = df['player'].value_counts().index[:N_CAT]
    code = {p: i for i, p in enumerate(top)}
    df['player_cat'] = df['player'].map(lambda p: code.get(p, N_CAT)).astype(int)
    # control: each row gets a RANDOM player's id (same marginal distribution, no link to the row). A relabelling of the ids
    # would give a tree model exactly the same information, so it would not be a control.
    df['player_shuf'] = rng.permutation(df['player_cat'].to_numpy())
    return df


STYLE_COLS = ['st_speed', 'st_basket_dist', 'st_abs_y', 'st_def1', 'st_usage', 'st_shoot', 'st_pass', 'st_hold_basket']
SHRINK_K = 500.0


def style_table(Mtr, Dtr):
    """Per-player style statistics from TRAINING rows only, each shrunk toward the average of the player's position group:
    off-ball movement speed, how far from the basket and from the sideline-centre line he stands, how closely he is guarded,
    usage (share of his states holding the ball), and, as a ball handler, his shoot / pass rates and distance from the basket."""
    a = Mtr.assign(abs_y=(Mtr['y'] - 25).abs(), holder=Mtr['is_holder']).groupby('player').agg(
        n=('x', 'size'), pos=('pos', 'first'), st_speed=('speed', 'mean'), st_basket_dist=('basket_dist', 'mean'), st_abs_y=('abs_y', 'mean'),
        st_def1=('def1', 'mean'), st_usage=('holder', 'mean'))
    b = Dtr.assign(shoot=(Dtr['action'] == 0).astype(float), passed=(Dtr['action'] == 1).astype(float)).groupby('player').agg(
        nd=('x', 'size'), st_shoot=('shoot', 'mean'), st_pass=('passed', 'mean'), st_hold_basket=('basket_dist', 'mean'))
    t = a.join(b, how='left')
    t['nd'] = t['nd'].fillna(0)
    out = pd.DataFrame(index=t.index)
    for c, nn in (('st_speed', 'n'), ('st_basket_dist', 'n'), ('st_abs_y', 'n'), ('st_def1', 'n'), ('st_usage', 'n'),
                  ('st_shoot', 'nd'), ('st_pass', 'nd'), ('st_hold_basket', 'nd')):
        g = t.groupby('pos')[c].transform(lambda v: np.nanmean(v))
        w = t[nn] / (t[nn] + SHRINK_K)
        out[c] = (w * t[c].fillna(g) + (1 - w) * g)
    return out


def attach_style(df, table, shuffle=False):
    if shuffle:      # control: every player gets another player's style statistics
        table = pd.DataFrame(table.to_numpy()[np.random.default_rng(0).permutation(len(table))], index=table.index, columns=table.columns)
    return df.join(table, on='player')


def folds_by_game(games, k=5, seed=0):
    ug = np.array(sorted(set(games)))
    perm = np.random.default_rng(seed).permutation(len(ug))
    fo = {ug[i]: j % k for j, i in enumerate(perm)}
    return np.array([fo[g] for g in games])


def variants(num_cols):
    allv = {'M0 situation': (num_cols, []), 'M1 +position': (num_cols + ['pos'], ['pos']),
            'M2 +identity': (num_cols + ['pos', 'player_cat'], ['pos', 'player_cat']),
            'M3 shuffled id (control)': (num_cols + ['pos', 'player_shuf'], ['pos', 'player_shuf']),
            'M4 +style stats': (num_cols + ['pos'] + STYLE_COLS, ['pos']),
            'M5 shuffled style (control)': (num_cols + ['pos'] + STYLE_COLS, ['pos'])}
    keep = [v for v in ONLY.split(',') if v] if ONLY else list(allv)
    return {k: v for k, v in allv.items() if k.split()[0] in keep or k in keep}


def fit_predict(kind, tr, te, X, y, cols, cats, **kw):
    mask = [c in cats for c in cols]
    params = HGB_R if kind == 'reg' else HGB_C
    m = (HistGradientBoostingRegressor if kind == 'reg' else HistGradientBoostingClassifier)(categorical_features=mask if any(mask) else None, **params)
    m.fit(X.loc[tr, cols], y[tr])
    return m.predict(X.loc[te, cols]) if kind == 'reg' else m.predict_proba(X.loc[te, cols])


def boot_game_ratio(game, a, b, n=500, seed=0):
    """Game-clustered bootstrap CI of 1 - sum(a)/sum(b) (relative reduction of loss a vs b)."""
    ug, gi = np.unique(game, return_inverse=True)
    sa, sb = np.bincount(gi, weights=a), np.bincount(gi, weights=b)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        out.append(1 - (w @ sa) / (w @ sb))
    return float(1 - sa.sum() / sb.sum()), np.percentile(out, [2.5, 97.5]).tolist()


STYLE_D = STYLE_M = None


def task_movement(M, out):
    num = [c for c in M.columns if c not in ('game', 'player', 'pos', 'dx', 'dy', 'player_cat', 'player_shuf')]
    folds = folds_by_game(M['game'].to_numpy())
    log(f'\n[A] movement: {len(M)} player-states, {M.game.nunique()} games, {M.player.nunique()} players')
    base = {'zero move': (M.dx ** 2 + M.dy ** 2).to_numpy(),
            'constant velocity': ((M.dx - M.vx) ** 2 + (M.dy - M.vy) ** 2).to_numpy()}
    sse = {}
    for name, (cols, cats) in variants(num).items():
        t0 = time.time()
        pred = np.zeros((len(M), 2))
        for k in range(5):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            Xk = M
            if 'st_speed' in cols:
                tab = style_table(M.iloc[tr], STYLE_D[STYLE_D['game'].isin(set(M.game.iloc[tr]))])
                Xk = attach_style(M, tab, 'shuffled' in name)
            for a, tgt in enumerate(('dx', 'dy')):
                pred[te, a] = fit_predict('reg', tr, te, Xk, M[tgt].to_numpy(), cols, cats)
        sse[name] = (M.dx - pred[:, 0]) ** 2 + (M.dy - pred[:, 1]) ** 2
        log(f'   {name:26s} RMSE of 1-s displacement {np.sqrt(sse[name].mean() / 2):.3f} ft/axis   [{time.time() - t0:.0f}s]')
    res = {k: float(np.sqrt(v.mean() / 2)) for k, v in sse.items()}
    for k, v in base.items():
        res[k] = float(np.sqrt(v.mean() / 2))
        log(f'   {k:26s} RMSE {res[k]:.3f}')
    g = M['game'].to_numpy()
    cmp = {}
    for a, b in (('M1 +position', 'M0 situation'), ('M2 +identity', 'M1 +position'), ('M2 +identity', 'M0 situation'), ('M3 shuffled id (control)', 'M1 +position'),
                 ('M4 +style stats', 'M1 +position'), ('M4 +style stats', 'M2 +identity'), ('M5 shuffled style (control)', 'M1 +position')):
        if a not in sse or b not in sse:
            continue
        r, ci = boot_game_ratio(g, sse[a].to_numpy(), sse[b].to_numpy())
        cmp[f'{a} vs {b}'] = dict(mse_reduction=r, ci=ci)
        log(f'   {a} vs {b}: squared-error reduction {r * 100:+.2f}% [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]')
    # who gains: per-player error ratio, M2 vs M1, for players with many rows
    pp = None if 'M2 +identity' not in sse else pd.DataFrame({'player': M.player, 'm1': sse['M1 +position'], 'm2': sse['M2 +identity']}).groupby('player').agg(n=('m1', 'size'), m1=('m1', 'sum'), m2=('m2', 'sum'))
    if pp is not None:
        pp = pp[pp.n >= 1500]
        pp['gain'] = 1 - pp.m2 / pp.m1
        log(f"   per-player gain from identity (players with >= 1500 rows, n={len(pp)}): median {pp.gain.median() * 100:+.2f}%, "
            f"share of players that improve {(pp.gain > 0).mean():.2f}, 10th/90th pct {pp.gain.quantile(.1) * 100:+.1f}% / {pp.gain.quantile(.9) * 100:+.1f}%")
    out['movement'] = dict(rmse=res, comparisons=cmp, per_player=(dict(n=int(len(pp)), median_gain=float(pp.gain.median()), share_improved=float((pp.gain > 0).mean())) if pp is not None else None),
                           n_rows=int(len(M)), n_games=int(M.game.nunique()))


def task_decision(Dc, out):
    num = [c for c in Dc.columns if c not in ('game', 'player', 'pos', 'action', 'player_cat', 'player_shuf')]
    folds = folds_by_game(Dc['game'].to_numpy(), seed=1)
    y = Dc['action'].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    log(f"\n[B] ball-handler decision: {len(Dc)} states, {Dc.game.nunique()} games; next 2 s: shoot {prior[0]:.3f}, pass {prior[1]:.3f}, keep {prior[2]:.3f}")
    ll = {}
    base = -np.log(np.clip(prior[y], 1e-9, 1))
    for name, (cols, cats) in variants(num).items():
        t0 = time.time()
        P = np.zeros((len(Dc), 3))
        for k in range(5):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            Xk = Dc
            if 'st_speed' in cols:
                tab = style_table(STYLE_M[STYLE_M['game'].isin(set(Dc.game.iloc[tr]))], Dc.iloc[tr])
                Xk = attach_style(Dc, tab, 'shuffled' in name)
            P[te] = fit_predict('clf', tr, te, Xk, y, cols, cats)
        ll[name] = -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))
        log(f'   {name:26s} log loss {ll[name].mean():.4f} (prior {base.mean():.4f})   [{time.time() - t0:.0f}s]')
    g = Dc['game'].to_numpy()
    cmp = {}
    for a, b in (('M1 +position', 'M0 situation'), ('M2 +identity', 'M1 +position'), ('M2 +identity', 'M0 situation'), ('M3 shuffled id (control)', 'M1 +position'),
                 ('M4 +style stats', 'M1 +position'), ('M4 +style stats', 'M2 +identity'), ('M5 shuffled style (control)', 'M1 +position')):
        if a not in ll or b not in ll:
            continue
        r, ci = boot_game_ratio(g, ll[a], ll[b])
        cmp[f'{a} vs {b}'] = dict(logloss_reduction=r, ci=ci)
        log(f'   {a} vs {b}: log-loss reduction {r * 100:+.2f}% [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]')
    out['decision'] = dict(logloss={k: float(v.mean()) for k, v in ll.items()}, prior_logloss=float(base.mean()), comparisons=cmp,
                           classes=dict(shoot=float(prior[0]), pas=float(prior[1]), keep=float(prior[2])), n_rows=int(len(Dc)), n_games=int(Dc.game.nunique()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n-games', type=int, default=150)
    ap.add_argument('--out', default=None)
    ap.add_argument('--tasks', default='movement,decision')
    ap.add_argument('--variants', default='', help='comma list of M0..M4 (default all)')
    a = ap.parse_args()
    global ONLY, STYLE_D, STYLE_M
    ONLY = a.variants
    games = split_games('train')
    games = games[::max(1, len(games) // a.n_games)][:a.n_games]
    t0 = time.time()
    G = load_games(a.data, games)
    log(f'loaded {len(G)} of {len(games)} train games ({time.time() - t0:.0f}s)')
    M, Dc = build_rows(G)
    rng = np.random.default_rng(0)
    M, Dc = add_identity(M, rng), add_identity(Dc, rng)
    log(f'rows built ({time.time() - t0:.0f}s): movement {len(M)}, decision {len(Dc)}')
    STYLE_M, STYLE_D = M, Dc
    out = {}
    if 'movement' in a.tasks:
        task_movement(M, out)
    if 'decision' in a.tasks:
        task_decision(Dc, out)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(out, open(os.path.join(a.out, 'identity_test.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
