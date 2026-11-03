"""Stage 6B.1: defensive scheme library.

    cd Code && python rl/build_schemes.py --data ../data/rl --out ../reports/rl-stage6b --k 8

Windows: every front-court live state k (stride 5 = 1 s) with an unbroken 3-s future in the same possession. Descriptor = the six knobs of defense.py, averaged
over the states +1 s, +2 s, +3 s (man assignment taken at the window start). K-means on the z-scored descriptors (train-fit games), K = --k.
Validation like the plays (Stage 3): (a) stability: cluster two halves of the games separately, match centroids, agreement of the assignments of all windows;
(b) team differences: the cluster mix of the defending teams against the same teams with shuffled windows; (c) outcome signal: points the offence scores in the
5 s after the window by cluster (game-clustered bootstrap) and the share of outcome variance explained; (d) is it a continuum: silhouette.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
from scipy.optimize import linear_sum_assignment   # noqa: E402
from sklearn.cluster import KMeans       # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_rand_score   # noqa: E402
from load import load_games             # noqa: E402
from train_executor import fit_val_games   # noqa: E402
import defense as DF                      # noqa: E402


def windows(d, stride=5):
    n = len(d['frame'])
    ks = np.arange(0, n - 16, stride)
    ok = d['live'][ks] & d['frontcourt'][ks] & (d['holder'][ks] != 0)
    ok &= (d['frame'][ks + 15] - d['frame'][ks] == 75) & (d['poss_id'][ks + 15] == d['poss_id'][ks])
    ok &= (d['ids'][ks] == d['ids'][ks + 15]).all(1)
    ks = ks[ok]
    if len(ks) == 0:
        return None
    man = DF.assign(d['pos'][ks])
    raw = np.mean([DF.knobs(d['pos'][ks + s], d['ball'][ks + s, :2], man) for s in (5, 10, 15)], axis=0)
    team = {int(r.player): int(r.team) for r in d['roster'].itertuples()}
    dteam = np.array([team.get(int(p), 0) for p in d['ids'][ks, 5]])
    f0 = d['frame'][ks]
    pts = np.zeros(len(ks))
    sh = d['shots']
    if len(sh):
        fr = sh['frame'].to_numpy()
        val = (sh['made'].to_numpy() * np.where(sh['is3'].to_numpy() == 1, 3, 2)).astype(float)
        for q, f in enumerate(f0):
            m = (fr > f) & (fr <= f + 125)
            pts[q] = val[m].sum()
    return dict(raw=raw, dteam=dteam, pts=pts, k=ks, t_front=d['t_front'][ks])


def match(Ca, Cb):
    cost = ((Ca[:, None] - Cb[None]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    perm = np.zeros(len(Ca), int)
    perm[r] = c
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/rl')
    ap.add_argument('--out', default='../reports/rl-stage6b')
    ap.add_argument('--k', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fit_g, val_g = fit_val_games()
    parts, gidx = [], []
    games = fit_g + val_g
    for gi, g in enumerate(games):
        d = load_games(a.data, [g])
        if not d:
            continue
        w = windows(d[0])
        if w is not None:
            parts.append(w)
            gidx.append(np.full(len(w['raw']), gi))
    raw = np.concatenate([p['raw'] for p in parts])
    dteam = np.concatenate([p['dteam'] for p in parts])
    pts = np.concatenate([p['pts'] for p in parts])
    gid = np.concatenate(gidx)
    isfit = gid < len(fit_g)
    print(f'{len(raw)} windows from {len(parts)} games ({isfit.sum()} fit, {(~isfit).sum()} val)', flush=True)
    mu, sd = raw[isfit].mean(0), raw[isfit].std(0) + 1e-6
    Z = ((raw - mu) / sd).astype(np.float32)
    km = KMeans(a.k, n_init=10, random_state=0).fit(Z[isfit])
    lab = km.predict(Z)
    freq = np.bincount(lab[isfit], minlength=a.k) / isfit.sum()
    res = dict(n_windows=int(len(raw)), k=a.k, knob_names=DF.KNOB_NAMES, raw_mean=mu.tolist(), raw_sd=sd.tolist())
    # (a) stability: two halves of the games
    half = (gid % 2 == 0)
    k1 = KMeans(a.k, n_init=10, random_state=1).fit(Z[half])
    k2 = KMeans(a.k, n_init=10, random_state=2).fit(Z[~half])
    perm = match(k1.cluster_centers_, k2.cluster_centers_)
    l1, l2 = k1.predict(Z), perm[np.arange(a.k)][k1.predict(Z)] if False else None
    inv = np.zeros(a.k, int)
    inv[perm] = np.arange(a.k)                               # cluster of half-2 -> matching cluster of half 1
    agree = float((inv[k2.predict(Z)] == k1.predict(Z)).mean())
    res['stability'] = dict(agreement_between_halves=agree, ari=float(adjusted_rand_score(k1.predict(Z), inv[k2.predict(Z)])))
    sub = np.random.default_rng(0).choice(len(Z), min(20000, len(Z)), replace=False)
    res['silhouette'] = float(silhouette_score(Z[sub], lab[sub]))
    # (b) team differences: chi-square-like spread of the cluster mix by defending team against shuffled labels
    teams = np.unique(dteam)
    mix = np.array([np.bincount(lab[dteam == t], minlength=a.k) / max((dteam == t).sum(), 1) for t in teams])
    spread = float(np.abs(mix - freq).sum(1).mean())
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(100):
        sh_ = rng.permutation(dteam)
        nulls.append(np.abs(np.array([np.bincount(lab[sh_ == t], minlength=a.k) / max((sh_ == t).sum(), 1) for t in teams]) - freq).sum(1).mean())
    res['team_mix'] = dict(mean_abs_deviation=spread, null_mean=float(np.mean(nulls)), ratio=spread / float(np.mean(nulls)))
    # (c) outcome signal
    ug = np.unique(gid)
    byc = {}
    for c in range(a.k):
        m = lab == c
        bs = []
        for _ in range(200):
            pick = rng.choice(ug, len(ug))
            sel = np.isin(gid, pick) & m
            bs.append(pts[sel].mean() if sel.any() else np.nan)
        byc[int(c)] = dict(share=float(m.mean()), pts_next5s=float(pts[m].mean()), ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                           centroid_raw=(km.cluster_centers_[c] * sd + mu).round(2).tolist())
    ss_tot = ((pts - pts.mean()) ** 2).sum()
    ss_in = sum(((pts[lab == c] - pts[lab == c].mean()) ** 2).sum() for c in range(a.k))
    res['outcome'] = dict(by_cluster=byc, variance_explained=float(1 - ss_in / ss_tot))
    json.dump(res, open(os.path.join(a.out, 'scheme_library.json'), 'w'), indent=1)
    np.savez(DF.LIB_PATH, mu=mu.astype(np.float32), sd=sd.astype(np.float32), C=km.cluster_centers_.astype(np.float32), freq=freq.astype(np.float32))
    print(json.dumps({k: v for k, v in res.items() if k != 'outcome'}, indent=1))
    for c, v in byc.items():
        print(c, f"share {v['share']:.3f} pts5s {v['pts_next5s']:.3f} {np.round(v['ci95'], 3).tolist()} centroid(gap,sag,press,paint,help,depth) {v['centroid_raw']}")
    print('variance explained by scheme', res['outcome']['variance_explained'])


if __name__ == '__main__':
    main()
