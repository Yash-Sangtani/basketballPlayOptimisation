"""Stage 6F-A: can a classifier tell real 3-s snippets of front-court play from simulated ones? (pre-registered in reports/rl-stage6f-prereg.md)

    python rl/audit_discriminator.py --out ../reports/rl-stage6f/discriminator.json

Per source (imitation engine, defence agent seed 0, loose league final, tight league final): HistGradientBoosting real vs source, fitted on train-game windows (real) and engine runs
from train quarter starts, scored on holdout-game windows and engine runs from holdout quarter starts. ROC AUC with a bootstrap over games (real) and chains (simulated).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
from scipy.spatial import ConvexHull     # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier   # noqa: E402
from sklearn.metrics import roc_auc_score   # noqa: E402
from load import load_games             # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from engine_validate import quarter_starts   # noqa: E402
import defense as DF                     # noqa: E402
import league as L                       # noqa: E402
import league_eval as LE                 # noqa: E402

BASKET = DF.BASKET
FEATURES = ['off_speed_mean', 'off_speed_sd', 'off_speed_max', 'def_speed_mean', 'def_speed_sd', 'def_speed_max', 'ball_speed', 'off_hull', 'def_hull', 'off_basket', 'def_basket',
            'gap', 'sag', 'press', 'paint', 'help', 'depth', 'holder_basket', 'holder_def', 'shot_clock', 't_front']
W = 15


def hull(p):
    try:
        return ConvexHull(p).volume
    except Exception:
        return 0.0


def featurize(pos, ball, sc, tf):
    """pos (n,W,10,2), ball (n,W,3), sc (n,), tf (n,) -> (n, len(FEATURES))"""
    n = len(pos)
    v = np.hypot(*(pos[:, 1:] - pos[:, :-1]).transpose(3, 0, 1, 2)) / 0.2          # (n,W-1,10)
    vo, vd = v[..., :5], v[..., 5:]
    bs = np.hypot(*(ball[:, 1:, :2] - ball[:, :-1, :2]).transpose(2, 0, 1)) / 0.2
    idx = (0, 7, 14)
    oh = np.array([[hull(pos[i, t, :5]) for t in idx] for i in range(n)]).mean(1)
    dh = np.array([[hull(pos[i, t, 5:]) for t in idx] for i in range(n)]).mean(1)
    ob = np.mean([np.hypot(*(pos[:, t, :5] - BASKET).transpose(2, 0, 1)).mean(1) for t in idx], axis=0)
    db = np.mean([np.hypot(*(pos[:, t, 5:] - BASKET).transpose(2, 0, 1)).mean(1) for t in idx], axis=0)
    man = DF.assign(pos[:, 0])
    kn = np.mean([DF.knobs(pos[:, t], ball[:, t, :2], man) for t in idx], axis=0)
    hb = np.mean([np.hypot(*(ball[:, t, :2] - BASKET).T) for t in idx], axis=0)
    hd = np.mean([np.hypot(*(pos[:, t, 5:] - ball[:, t, None, :2]).transpose(2, 0, 1)).min(1) for t in idx], axis=0)
    X = np.column_stack([vo.mean((1, 2)), vo.std((1, 2)), vo.max((1, 2)), vd.mean((1, 2)), vd.std((1, 2)), vd.max((1, 2)), bs.mean(1), oh, dh, ob, db, kn, hb, hd, sc, tf])
    return X.astype(np.float32)


def real_windows(G, per_game=40, rng=None):
    Xs, gid = [], []
    for gi, d in enumerate(G):
        n = len(d['frame'])
        ks = np.arange(0, n - W, 5)
        ok = d['live'][ks] & d['frontcourt'][ks] & (d['holder'][ks] != 0) & (d['t_front'][ks] >= 1.0)
        ok &= (d['frame'][ks + W - 1] - d['frame'][ks] == 5 * (W - 1)) & (d['poss_id'][ks + W - 1] == d['poss_id'][ks]) & (d['ids'][ks] == d['ids'][ks + W - 1]).all(1)
        ks = ks[ok]
        ks = ks[(d['holder'][ks[:, None] + np.arange(W)[None]] != 0).all(1)]          # same rule as the simulated windows: the ball is held in every state (no pass in flight)
        if len(ks) > per_game:
            ks = rng.choice(ks, per_game, replace=False)
        if len(ks) == 0:
            continue
        sel = ks[:, None] + np.arange(W)[None]
        sc = np.nan_to_num(d['shot_clock'][ks], nan=24.0)
        Xs.append(featurize(d['pos'][sel], d['ball'][sel], sc, d['t_front'][ks]))
        gid.append(np.full(len(ks), gi))
    return np.concatenate(Xs), np.concatenate(gid)


def sim_windows(eng, stride=5):
    T = np.stack(eng.traj)                         # (S,N,10,2)
    Bl = np.stack(eng.ball_traj)
    H = np.stack(eng.holder_traj)
    O = np.stack(eng.order_traj)
    SC = None
    S, N = T.shape[:2]
    pos, ball, ch, tf = [], [], [], []
    tfront = None
    for i in range(N):
        for s0 in range(0, S - W, stride):
            sl = slice(s0, s0 + W)
            if (H[sl, i] == 0).any() or (O[sl, i] != O[s0, i]).any():
                continue
            if (T[sl, i, :5, 0].mean(1) >= 47.0).any():
                continue
            pos.append(T[sl, i]); ball.append(Bl[sl, i]); ch.append(i)
    if not pos:
        return np.zeros((0, len(FEATURES)), np.float32), np.zeros(0, int)
    pos, ball = np.array(pos), np.array(ball)
    n = len(pos)
    # the engine's shot clock / front-court time are not stored per state; use neutral values from the positions: shot clock unknown -> 12, time since midcourt -> 3 s (excluded from comparison by fixing both)
    X = featurize(pos, ball, np.full(n, 12.0), np.full(n, 3.0))
    return X, np.array(ch)


def boot_auc(y, s, groups, rng, n=200):
    out = []
    ug_r = np.unique(groups[y == 0]); ug_s = np.unique(groups[y == 1])
    for _ in range(n):
        pr = rng.choice(ug_r, len(ug_r)); ps = rng.choice(ug_s, len(ug_s))
        m_r = np.concatenate([np.flatnonzero((groups == g) & (y == 0)) for g in pr]); m_s = np.concatenate([np.flatnonzero((groups == g) & (y == 1)) for g in ps])
        idx = np.concatenate([m_r, m_s])
        out.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=96); ap.add_argument('--blocks', type=int, default=450); ap.add_argument('--train-games', type=int, default=120)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl')
    ap.add_argument('--sources', default='imitation=,def_seed0=../reports/rl-stage6c/run3/ckpt_0060.pkl,loose_league=../reports/rl-stage6d/run3/ckpt_0100.pkl,tight_league=../reports/rl-stage6d/run_tight/ckpt_0050.pkl')
    a = ap.parse_args()
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    rng = np.random.default_rng(0)
    Gtr = load_games(a.data, split_games('train')[:a.train_games])
    Gho = load_games(a.data, split_games('holdout'))
    t0 = time.time()
    Xr_tr, gr_tr = real_windows(Gtr, 40, rng)
    Xr_ho, gr_ho = real_windows(Gho, 60, rng)
    print(f'real windows: train {len(Xr_tr)}, holdout {len(Xr_ho)}  ({time.time() - t0:.0f}s)', flush=True)
    res = dict(n_real_train=int(len(Xr_tr)), n_real_holdout=int(len(Xr_ho)), features=FEATURES, sources={})
    for item in a.sources.split(','):
        name, path = item.split('=')
        A = LE.agents(st, path or None, 1)
        out = {}
        sims = {}
        for split, G, key in (('train', Gtr, 'tr'), ('holdout', Gho, 'ho')):
            qs = quarter_starts(G)
            pick = rng.choice(len(qs), min(a.n, len(qs)), replace=False)
            starts = [(qs[j][0], qs[j][1]) for j in pick]
            eng = L.run_match(st, G, table, starts, seed=77, blocks=a.blocks, A=A, B=A)
            sims[key] = sim_windows(eng)
        Xs_tr, cs_tr = sims['tr']; Xs_ho, cs_ho = sims['ho']
        # shot clock and front-court time are not recorded per simulated state: drop both from every source (the real windows too) so the comparison is on movement and formation only
        keep = [j for j, f in enumerate(FEATURES) if f not in ('shot_clock', 't_front')]
        variants = {'all_features': keep, 'without_ball_speed': [j for j in keep if FEATURES[j] != 'ball_speed']}
        vres = {}
        for vname, cols in variants.items():
            Xtr = np.concatenate([Xr_tr, Xs_tr])[:, cols]; ytr = np.r_[np.zeros(len(Xr_tr)), np.ones(len(Xs_tr))]
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, ytr)
            Xte = np.concatenate([Xr_ho, Xs_ho])[:, cols]; yte = np.r_[np.zeros(len(Xr_ho)), np.ones(len(Xs_ho))]
            s_ = clf.predict_proba(Xte)[:, 1]
            grp = np.r_[gr_ho, 1000 + cs_ho]
            auc_v = float(roc_auc_score(yte, s_))
            ci_v = boot_auc(yte, s_, grp, rng)
            imp = {}
            for j, f in enumerate([FEATURES[k] for k in cols]):
                Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
                imp[f] = float(auc_v - roc_auc_score(yte, clf.predict_proba(Xp)[:, 1]))
            vres[vname] = dict(auc=auc_v, ci95=ci_v, top_features=sorted(imp.items(), key=lambda kv: -kv[1])[:4])
        auc, ci, top = vres['all_features']['auc'], vres['all_features']['ci95'], vres['all_features']['top_features']
        res['sources'][name] = dict(variants=vres, n_sim_train=int(len(Xs_tr)), n_sim_holdout=int(len(Xs_ho)),
                                    sim_means={f: float(Xs_ho[:, k].mean()) for f, k in zip([FEATURES[k] for k in keep], keep)}, real_means={f: float(Xr_ho[:, k].mean()) for f, k in zip([FEATURES[k] for k in keep], keep)})
        v2 = vres['without_ball_speed']
        print(f'{name:14s} AUC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  without ball speed {v2["auc"]:.3f} [{v2["ci95"][0]:.3f}, {v2["ci95"][1]:.3f}]  top {[(f, round(v, 3)) for f, v in v2["top_features"]]}  ({time.time() - t0:.0f}s)', flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
