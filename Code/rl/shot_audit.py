"""Stage 5, part 1: is the shot model's expected-points value right for the shots the policy takes?

    cd Code && python rl/shot_audit.py --data ../data/rl --out ../reports/rl-stage5

(a) Real shots of the 100 RL-holdout games (never seen by the shot model): model xPts (same guarded rules as the simulator) against the points the
    shots actually scored, by shot distance, by seconds elapsed on the shot clock and by nearest-defender distance, with game-clustered 95% intervals.
(b) An empirical shot value table (points per real shot by distance x shot-clock-elapsed x nearest-defender bins, shrunk to the distance-bin mean) fitted on the
    401 RL-train games, saved to Code/rl/model/shot_table.npz. It is the 'audited' value used by the stricter reward.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
from load import load_games              # noqa: E402
from splits import games as split_games  # noqa: E402
from features import BASKET              # noqa: E402
from pass_features import shot_features  # noqa: E402

TABLE_PATH = os.path.join(HERE, 'model', 'shot_table.npz')
D_EDGES = np.array([0, 6, 12, 18, 22, 25, 28, 33, 200.0])
C_EDGES = np.array([0, 6, 12, 18, 25.0])          # seconds elapsed on the shot clock
F_EDGES = np.array([0, 3, 5, 8, 100.0])           # nearest defender (ft)
SHRINK = 30.0


def real_shots(games):
    """-> dict of arrays: dist, elapsed, def1, pts, game, plus the feature dicts for the model."""
    rows, feats = [], []
    for gi, d in enumerate(games):
        sh = d['shots']
        if len(sh) == 0:
            continue
        fr = d['frame']
        for r in sh.itertuples():
            k = int(np.searchsorted(fr, r.frame, side='right') - 1)
            if k < 0 or r.frame - fr[k] > 10 or not d['live'][k]:
                continue
            slot = np.flatnonzero(d['ids'][k, :5] == r.shooter)
            if len(slot) == 0:
                continue
            s = slot[0]
            sc = d['shot_clock'][k]
            if np.isnan(sc):
                sc = max(24.0 - d['t_in_poss'][k], 0.0)
            f = shot_features(d['pos'][k, s], d['pos'][k, 5:], BASKET, d['vel'][k, s], sc)
            if f is None or f['shot_dist'] > 60:
                continue
            rows.append((f['shot_dist'], 24.0 - sc, f['def1'], float(r.made) * (3 if r.is3 == 1 else 2), gi))
            feats.append(f)
    a = np.array(rows)
    return dict(dist=a[:, 0], elapsed=np.clip(a[:, 1], 0, 24.9), def1=a[:, 2], pts=a[:, 3], game=a[:, 4].astype(int)), feats


def bin_index(dist, elapsed, def1):
    return (np.clip(np.searchsorted(D_EDGES, dist, side='right') - 1, 0, len(D_EDGES) - 2), np.clip(np.searchsorted(C_EDGES, elapsed, side='right') - 1, 0, len(C_EDGES) - 2),
            np.clip(np.searchsorted(F_EDGES, def1, side='right') - 1, 0, len(F_EDGES) - 2))


def fit_table(S):
    nd, nc, nf = len(D_EDGES) - 1, len(C_EDGES) - 1, len(F_EDGES) - 1
    i, j, k = bin_index(S['dist'], S['elapsed'], S['def1'])
    tot = np.zeros((nd, nc, nf))
    cnt = np.zeros((nd, nc, nf))
    np.add.at(tot, (i, j, k), S['pts'])
    np.add.at(cnt, (i, j, k), 1)
    prior = np.array([S['pts'][i == a].mean() if (i == a).any() else S['pts'].mean() for a in range(nd)])
    val = (tot + SHRINK * prior[:, None, None]) / (cnt + SHRINK)
    return val, cnt


class AuditedShotValue:
    """Empirical points per real shot in (distance, clock elapsed, nearest defender) bins (shrunk). Call with arrays or scalars."""

    def __init__(self, path=TABLE_PATH):
        z = np.load(path)
        self.val, self.cnt = z['val'], z['cnt']

    def __call__(self, dist, elapsed, def1):
        i, j, k = bin_index(np.asarray(dist), np.asarray(elapsed), np.asarray(def1))
        return self.val[i, j, k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from api import PassValueModel
    from caller import batch_xpts
    diff = PassValueModel()
    os.makedirs(a.out, exist_ok=True)
    Gt = load_games(a.data, split_games('train'))
    St, _ = real_shots(Gt)
    val, cnt = fit_table(St)
    np.savez_compressed(TABLE_PATH, val=val, cnt=cnt)
    print(f'empirical table from {len(St["pts"])} real shots of the {len(Gt)} train games; mean {St["pts"].mean():.3f} points per shot')
    del Gt
    Gh = load_games(a.data, split_games('holdout'))
    Sh, feats = real_shots(Gh)
    items = [(np.array([0.0, 0.0]),) for _ in feats]
    # model xPts from the stored feature dicts (guarded: clamp + heave rule, as in the simulator)
    import pandas as pd
    lo_hi = diff.ranges['shot_features']
    rows, heave = [], np.zeros(len(feats), bool)
    for n, f in enumerate(feats):
        if f['shot_dist'] > lo_hi['shot_dist'][1]:
            heave[n] = True
        rows.append({k: (min(max(v, lo_hi[k][0]), lo_hi[k][1]) if k in lo_hi and k != 'shot_dist' and np.isfinite(v) else v) for k, v in f.items()})
    xp = np.array(diff._xpts(diff.shot, pd.DataFrame(rows))['xpts'].to_numpy(), dtype=float)
    xp[heave] = 3.0 * diff.ranges['heave_fg']
    emp = AuditedShotValue()(Sh['dist'], Sh['elapsed'], Sh['def1'])
    g = Sh['game']
    ug = np.unique(g)
    rng = np.random.default_rng(0)

    def boot(mask):
        if mask.sum() < 30:
            return None
        d = Sh['pts'][mask] - xp[mask]
        gg = g[mask]
        u = np.unique(gg)
        s = np.array([d[gg == x].sum() for x in u])
        c = np.array([(gg == x).sum() for x in u])
        b = [s[w].sum() / c[w].sum() for w in (rng.integers(0, len(u), len(u)) for _ in range(300))]
        return dict(n=int(mask.sum()), real_pts=float(Sh['pts'][mask].mean()), model_xpts=float(xp[mask].mean()), gap_real_minus_model=float(d.mean()),
                    ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    res = dict(all=boot(np.ones(len(xp), bool)))
    for name, edges, v in (('distance_ft', D_EDGES, Sh['dist']), ('clock_elapsed_s', C_EDGES, Sh['elapsed']), ('nearest_defender_ft', F_EDGES, Sh['def1'])):
        res[name] = {f'{edges[b]:g}-{edges[b + 1]:g}': boot((v >= edges[b]) & (v < edges[b + 1])) for b in range(len(edges) - 1)}
    res['early_long'] = boot((Sh['dist'] >= 22) & (Sh['elapsed'] < 8))
    res['early_long_share_of_real_shots'] = float(((Sh['dist'] >= 22) & (Sh['elapsed'] < 8)).mean())
    res['empirical_table_vs_real_holdout_mse'] = dict(model=float(((Sh['pts'] - xp) ** 2).mean()), table=float(((Sh['pts'] - emp) ** 2).mean()), const=float(((Sh['pts'] - Sh['pts'].mean()) ** 2).mean()))
    json.dump(res, open(os.path.join(a.out, 'shot_audit_real.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('all', 'early_long', 'early_long_share_of_real_shots', 'empirical_table_vs_real_holdout_mse')}, indent=1))
    for nm in ('distance_ft', 'clock_elapsed_s', 'nearest_defender_ft'):
        for b, v in res[nm].items():
            if v:
                print(f'{nm:20s} {b:8s} n={v["n"]:5d} real {v["real_pts"]:.3f} model {v["model_xpts"]:.3f} gap {v["gap_real_minus_model"]:+.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]')


if __name__ == '__main__':
    main()
