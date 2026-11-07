"""Stage 1b: is a player's shot-location profile identifiable? (the most direct test of "plays like himself")

For every matched shot (TRAIN games) take the shooter's distance from the basket at release: rim (< 5 ft), short (5-14), mid (14-22),
three (>= 22). Predict the category of each held-out shot (GroupKFold by game) from: the league rates, the shooter's position group,
and the shooter's own rates from the training games, shrunk toward his position group with k pseudo-shots. Reports log-loss
reductions with game-clustered bootstrap intervals, and the same test with the player labels shuffled as a control.

    cd Code && python rl/shot_profile_test.py --data ../data/rl --out ../reports/rl-shot-profile
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from features import roster_positions, BASKET     # noqa: E402
from load import load_games              # noqa: E402
from splits import games as split_games  # noqa: E402
from identity_test import boot_game_ratio, folds_by_game   # noqa: E402

BINS = [0, 5, 14, 22, 100]


def shot_table(G):
    rows = []
    for d in G:
        pos = roster_positions(d['roster'])
        fr = d['frame']
        for s in d['shots'].itertuples():
            k = int(np.searchsorted(fr, s.frame))
            cand = [c for c in (k - 1, k) if 0 <= c < len(fr) and abs(int(fr[c]) - int(s.frame)) <= 2]
            if not cand:
                continue
            c = cand[0]
            if d['ctrl_team'][c] != s.team:
                continue
            j = np.flatnonzero(d['ids'][c, :5] == s.shooter)
            if len(j) == 0:
                continue
            dist = float(np.hypot(*(d['pos'][c, j[0]] - BASKET)))
            rows.append(dict(game=d['game'], player=int(s.shooter), pos=pos.get(int(s.shooter), 2), dist=dist, made=int(s.made)))
    T = pd.DataFrame(rows)
    T['cat'] = pd.cut(T['dist'], BINS, labels=False, right=False).astype(int)
    return T


def predict_probs(train, test, k, shuffle_players=False, seed=0):
    n_cat = len(BINS) - 1
    league = np.bincount(train['cat'], minlength=n_cat) + 1.0
    league /= league.sum()
    pos = {p: (np.bincount(g['cat'], minlength=n_cat) + league * 5) / (len(g) + 5) for p, g in train.groupby('pos')}
    tr = train
    if shuffle_players:
        tr = train.copy()
        tr['player'] = np.random.default_rng(seed).permutation(tr['player'].to_numpy())
    cnt = {p: np.bincount(g['cat'], minlength=n_cat) for p, g in tr.groupby('player')}
    out_pos, out_pl = [], []
    for r in test.itertuples():
        pp = pos.get(r.pos, league)
        c = cnt.get(r.player, np.zeros(n_cat))
        out_pos.append(pp)
        out_pl.append((c + k * pp) / (c.sum() + k))
    return np.array(out_pos), np.array(out_pl), league


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--k', type=float, default=50.0)
    a = ap.parse_args()
    G = load_games(a.data, split_games('train'))
    T = shot_table(G)
    print(f'{len(G)} train games, {len(T)} matched shots by {T.player.nunique()} players; category shares {np.bincount(T.cat) / len(T)}')
    folds = folds_by_game(T['game'].to_numpy())
    ll = {n: np.zeros(len(T)) for n in ('league', 'position', 'player', 'player_shuffled')}
    y = T['cat'].to_numpy()
    for f in range(5):
        tr, te = T[folds != f], T[folds == f]
        pp, pl, lg = predict_probs(tr, te, a.k)
        _, pls, _ = predict_probs(tr, te, a.k, shuffle_players=True)
        idx = np.flatnonzero(folds == f)
        yt = te['cat'].to_numpy()
        ll['league'][idx] = -np.log(lg[yt])
        ll['position'][idx] = -np.log(pp[np.arange(len(yt)), yt])
        ll['player'][idx] = -np.log(pl[np.arange(len(yt)), yt])
        ll['player_shuffled'][idx] = -np.log(pls[np.arange(len(yt)), yt])
    res = {k: float(v.mean()) for k, v in ll.items()}
    print('log loss:', {k: round(v, 4) for k, v in res.items()})
    g = T['game'].to_numpy()
    cmp = {}
    for A, B in (('position', 'league'), ('player', 'position'), ('player_shuffled', 'position')):
        r, ci = boot_game_ratio(g, ll[A], ll[B])
        cmp[f'{A} vs {B}'] = dict(logloss_reduction=r, ci=ci)
        print(f'  {A} vs {B}: log-loss reduction {r * 100:+.2f}% [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]')
    # how different are players? spread of three-point share among shooters with >= 200 shots
    big = T.groupby('player').agg(n=('cat', 'size'), three=('cat', lambda c: (c == 3).mean()), rim=('cat', lambda c: (c == 0).mean()))
    big = big[big.n >= 200]
    print(f'  players with >= 200 shots: {len(big)}; three-point share 10th/50th/90th pct {big.three.quantile(.1):.2f}/{big.three.median():.2f}/{big.three.quantile(.9):.2f}; '
          f'rim share {big.rim.quantile(.1):.2f}/{big.rim.median():.2f}/{big.rim.quantile(.9):.2f}')
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(dict(logloss=res, comparisons=cmp, n_shots=int(len(T)), n_players=int(T.player.nunique()), k=a.k), open(os.path.join(a.out, 'shot_profile.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
