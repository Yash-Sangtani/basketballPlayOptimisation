"""Stage 3: cluster half-court possessions into candidate plays and test whether the clusters are real.

    cd Code && python rl/cluster_plays.py --data ../data/rl --out ../reports/rl-stage3

Fit on the 320 fitting games, validate on the 81 validation games (the Stage 2 split); holdout/test not read. Tests, for K in --ks:
  stability   fit k-means on two random halves of the fitting GAMES, assign all fitting possessions with each, adjusted Rand index
  teams       do teams run different mixes of plays? chi-square-like statistic of the team x cluster table against a null that shuffles team labels
              between possessions of the same game side (preserves game structure)
  outcome     on the validation games, share of the variance of FG points per possession explained by the cluster (eta^2), against a null that
              shuffles cluster labels
  silhouette  on a sample
Selection rule fixed in advance: among K with stability ARI >= 0.4 take the K with the highest validation eta^2 (K=8 if none qualifies).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from sklearn.cluster import KMeans      # noqa: E402
from sklearn.decomposition import PCA   # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score   # noqa: E402
from load import load_games              # noqa: E402
from plays import build_table            # noqa: E402
from train_executor import fit_val_games   # noqa: E402


def log(*a):
    print(*a, flush=True)


def team_stat(labels, teams, K):
    """Sum over teams of the squared deviation of the team's cluster shares from the league shares, weighted by team size."""
    T = pd.crosstab(teams, labels).reindex(columns=range(K), fill_value=0)
    share = T.div(T.sum(1), axis=0).to_numpy()
    league = T.sum(0).to_numpy() / T.to_numpy().sum()
    return float(((share - league) ** 2 / np.maximum(league, 1e-9)).sum(1).dot(T.sum(1).to_numpy()) / T.to_numpy().sum())


def eta2(labels, y):
    m = y.mean()
    ss_tot = ((y - m) ** 2).sum()
    g = pd.Series(y).groupby(labels)
    ss_b = (g.count() * (g.mean() - m) ** 2).sum()
    return float(ss_b / ss_tot) if ss_tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ks', default='6,10,16,24')
    ap.add_argument('--pca', type=int, default=24)
    a = ap.parse_args()
    fit_g, val_g = fit_val_games()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    Xf, Mf = build_table(FIT)
    Xv, Mv = build_table(VAL)
    del FIT, VAL
    log(f'possessions: fit {len(Xf)} ({len(set(m["game"] for m in Mf))} games), validation {len(Xv)}')
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    pca = PCA(a.pca, random_state=0).fit((Xf - mu) / sd)
    Zf, Zv = pca.transform((Xf - mu) / sd), pca.transform((Xv - mu) / sd)
    log(f'PCA {a.pca} components explain {pca.explained_variance_ratio_.sum():.2f} of the variance')
    games = np.array([m['game'] for m in Mf])
    ug = np.unique(games)
    rng = np.random.default_rng(0)
    yv = np.array([m['fg_pts'] for m in Mv])
    teams = np.array([m['team'] for m in Mf])
    res = {}
    for K in [int(k) for k in a.ks.split(',')]:
        km = KMeans(K, n_init=3, random_state=0).fit(Zf)
        lf, lv = km.labels_, km.predict(Zv)
        # stability: two halves of the games
        aris = []
        for r in range(3):
            half = rng.permutation(ug)
            A, B = set(half[:len(ug) // 2]), set(half[len(ug) // 2:])
            ma, mb = np.isin(games, list(A)), np.isin(games, list(B))
            ka = KMeans(K, n_init=2, random_state=r).fit(Zf[ma])
            kb = KMeans(K, n_init=2, random_state=r + 10).fit(Zf[mb])
            aris.append(adjusted_rand_score(ka.predict(Zf), kb.predict(Zf)))
        sil = float(silhouette_score(Zf, lf, sample_size=min(8000, len(Zf)), random_state=0))
        # teams: shuffle team labels between the two sides of a game is not available per possession; shuffle across possessions within game
        obs = team_stat(lf, teams, K)
        null = []
        for _ in range(100):
            t2 = teams.copy()
            for g in ug:
                ix = np.flatnonzero(games == g)
                t2[ix] = rng.permutation(t2[ix])
            null.append(team_stat(lf, t2, K))
        # outcome
        e_obs = eta2(lv, yv)
        e_null = [eta2(rng.permutation(lv), yv) for _ in range(300)]
        res[K] = dict(stability_ari=float(np.mean(aris)), stability_ari_sd=float(np.std(aris)), silhouette=sil, team_stat=obs,
                      team_null_mean=float(np.mean(null)), team_null_p95=float(np.percentile(null, 95)),
                      eta2_val=e_obs, eta2_null_mean=float(np.mean(e_null)), eta2_null_p95=float(np.percentile(e_null, 95)),
                      cluster_sizes=np.bincount(lf, minlength=K).tolist())
        log(f"K={K:2d}: stability ARI {res[K]['stability_ari']:.2f}+-{res[K]['stability_ari_sd']:.2f}  silhouette {sil:.3f}  "
            f"team stat {obs:.3f} (null mean {res[K]['team_null_mean']:.3f}, 95th {res[K]['team_null_p95']:.3f})  "
            f"outcome eta2 (validation) {e_obs:.4f} (null mean {res[K]['eta2_null_mean']:.4f}, 95th {res[K]['eta2_null_p95']:.4f})")
    ok = [K for K, r in res.items() if r['stability_ari'] >= 0.4]
    Kbest = max(ok, key=lambda K: res[K]['eta2_val']) if ok else 8
    log(f'selected K = {Kbest} (rule: highest validation eta2 among K with stability ARI >= 0.4; 8 if none)')
    os.makedirs(a.out, exist_ok=True)
    km = KMeans(Kbest, n_init=5, random_state=0).fit(Zf)
    np.savez_compressed(os.path.join(a.out, 'play_library.npz'), mu=mu, sd=sd, pca_components=pca.components_, pca_mean=pca.mean_, centers=km.cluster_centers_, K=Kbest)
    # per-cluster summary on fit + validation
    lf, lv = km.labels_, km.predict(Zv)
    rows = []
    for k in range(Kbest):
        for nm, lab, M in (('fit', lf, Mf), ('val', lv, Mv)):
            sel = [m for m, l in zip(M, lab) if l == k]
            rows.append(dict(cluster=k, split=nm, n=len(sel), fg_pts=np.mean([m['fg_pts'] for m in sel]) if sel else np.nan,
                             shots=np.mean([m['n_shots'] for m in sel]) if sel else np.nan, three_share=(sum(m['n_three'] for m in sel) / max(sum(m['n_shots'] for m in sel), 1)),
                             duration=np.mean([m['duration'] for m in sel]) if sel else np.nan, n_teams=len(set(m['team'] for m in sel))))
    pd.DataFrame(rows).round(3).to_csv(os.path.join(a.out, 'cluster_summary.csv'), index=False)
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    json.dump(dict(per_k=res, selected=Kbest, n_fit=len(Xf), n_val=len(Xv)), open(os.path.join(a.out, 'cluster_tests.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(a.out, 'labels.npz'), fit=lf, val=lv, fit_game=np.array([m['game'] for m in Mf]), fit_poss=np.array([m['poss_id'] for m in Mf]),
                        val_game=np.array([m['game'] for m in Mv]), val_poss=np.array([m['poss_id'] for m in Mv]))


if __name__ == '__main__':
    main()
