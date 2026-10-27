"""
Fit and evaluate the value function V(s) (expected points in the rest of the possession) and score passes.

    python fit_value.py <states dir> <reports dir> [--folds 5] [--seed 0]

Everything is GroupKFold by game (shuffled with a fixed seed), so no game is ever scored by a model
that saw it, including the passes used for delta V.

  1. V(s): HistGradientBoostingRegressor on the rich state vector vs the same model on the baseline
     (shot clock + ball position) vs a constant (training mean). Pooled out-of-fold RMSE and R^2, per fold,
     calibration by decile, game-clustered bootstrap CI for the RMSE/R^2 gain of the rich model.
  2. delta V = V(reception state) - V(release state) for completed passes, out of fold, and a comparison of
     assisted passes with other groups of passes, with game-clustered bootstrap CIs.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from states import FEATURES, BASELINE  # noqa: E402

B_BOOT = 2000
HGB = dict(loss='squared_error', learning_rate=0.05, max_iter=150, max_leaf_nodes=31, min_samples_leaf=300,
           l2_regularization=1.0, early_stopping=False, random_state=0)
STAGE_EVERY = 10


def load(dirpath):
    res = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.path.join(dirpath, '*.pkl')))]
    samples = pd.concat([r['samples'] for r in res], ignore_index=True)
    ps = pd.concat([r['pass_states'] for r in res if len(r['pass_states'])], ignore_index=True)
    pm = pd.concat([r['pass_meta'] for r in res if len(r['pass_meta'])], ignore_index=True)
    diag = pd.DataFrame([r['diag'] for r in res])
    secs = [r['seconds'] for r in res]
    return samples, ps, pm, diag, secs


def boot_weights(n_games, B, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_games, size=(B, n_games))
    W = np.zeros((B, n_games))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=n_games)
    return W


def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def fit_cv(samples, ps, folds, seed):
    """Out-of-fold predictions for samples and (release/recv) pass states, for rich / baseline / constant."""
    games = samples.game.values
    gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    y = samples.pts.values.astype(float)
    n = len(samples)
    oof = {k: np.full(n, np.nan) for k in ('rich', 'base', 'const')}
    ps_pred = {k: np.full(len(ps), np.nan) for k in ('rich', 'base')}
    fold_of_game = {}
    stage_sse = {k: {} for k in ('rich', 'base')}      # iteration -> SSE summed over folds
    fold_id = np.zeros(n, dtype=int)
    importances = []
    for f, (tr, te) in enumerate(gkf.split(samples, y, games)):
        test_games = set(games[te])
        for g in test_games:
            fold_of_game[g] = f
        fold_id[te] = f
        for name, cols in (('rich', FEATURES), ('base', BASELINE)):
            m = HistGradientBoostingRegressor(**HGB).fit(samples.iloc[tr][cols], y[tr])
            oof[name][te] = m.predict(samples.iloc[te][cols])
            for k, p in enumerate(m.staged_predict(samples.iloc[te][cols]), 1):
                if k % STAGE_EVERY == 0:
                    stage_sse[name][k] = stage_sse[name].get(k, 0.0) + float(((y[te] - p) ** 2).sum())
            mask = ps.game.isin(test_games).values
            if mask.any():
                ps_pred[name][mask] = m.predict(ps.loc[mask, cols])
            if name == 'rich':
                # permutation importance on the held-out fold: increase in MSE when one feature is shuffled
                rng = np.random.default_rng(seed + f)
                Xte = samples.iloc[te][cols].copy()
                base_mse = float(((y[te] - oof['rich'][te]) ** 2).mean())
                imp = {}
                for c in cols:
                    keep = Xte[c].values.copy()
                    Xte[c] = rng.permutation(keep)
                    imp[c] = float(((y[te] - m.predict(Xte)) ** 2).mean()) - base_mse
                    Xte[c] = keep
                importances.append(imp)
        oof['const'][te] = y[tr].mean()
    return oof, ps_pred, fold_id, stage_sse, pd.DataFrame(importances)


def metrics(y, p):
    sse = float(((y - p) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return dict(rmse=(sse / len(y)) ** 0.5, r2=1 - sse / sst, mean_pred=float(p.mean()), mean_obs=float(y.mean()))


def calibration(y, p, games, W, deciles=10):
    """By decile of the out-of-fold prediction: mean prediction, mean outcome, n, and a game-bootstrap CI of
    the mean outcome."""
    q = pd.qcut(pd.Series(p).rank(method='first'), deciles, labels=False).values
    ug, gi = np.unique(games, return_inverse=True)
    rows = []
    for d in range(deciles):
        m = q == d
        n_g = np.bincount(gi[m], minlength=len(ug)).astype(float)
        s_g = np.bincount(gi[m], weights=y[m], minlength=len(ug))
        num, den = W @ s_g, W @ n_g
        boot = num / den
        lo, hi = ci(boot)
        rows.append(dict(decile=d + 1, n=int(m.sum()), mean_pred=float(p[m].mean()), mean_obs=float(y[m].mean()),
                         obs_ci_lo=lo, obs_ci_hi=hi))
    return pd.DataFrame(rows)


def calibration_slope(y, p, games, W):
    ug, gi = np.unique(games, return_inverse=True)
    stats = np.vstack([np.bincount(gi, minlength=len(ug)).astype(float), np.bincount(gi, weights=p, minlength=len(ug)),
                       np.bincount(gi, weights=y, minlength=len(ug)), np.bincount(gi, weights=p * p, minlength=len(ug)),
                       np.bincount(gi, weights=p * y, minlength=len(ug))]).T
    def slope(S):
        n, sp, sy, spp, spy = S
        return (n * spy - sp * sy) / (n * spp - sp * sp)
    full = slope(stats.sum(0))
    bs = np.array([slope(w @ stats) for w in W])
    return float(full), ci(bs)


def group_means(values, masks, games):
    """per-game sums for bootstrap of group means / contrasts / Cohen's d."""
    ug, gi = np.unique(games, return_inverse=True)
    out = {}
    for name, m in masks.items():
        out[name] = np.vstack([np.bincount(gi[m], minlength=len(ug)).astype(float),
                               np.bincount(gi[m], weights=values[m], minlength=len(ug)),
                               np.bincount(gi[m], weights=values[m] ** 2, minlength=len(ug))]).T
    return out


def contrast(stats, a, b, W):
    """mean difference a-b and Cohen's d, with game-bootstrap CIs. stats[g] = (n, sum, sumsq) per game."""
    def one(Sa, Sb):
        na, sa, qa = Sa
        nb, sb, qb = Sb
        ma, mb = sa / na, sb / nb
        va, vb = (qa - na * ma ** 2) / (na - 1), (qb - nb * mb ** 2) / (nb - 1)
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        return ma - mb, (ma - mb) / pooled
    full = one(stats[a].sum(0), stats[b].sum(0))
    bs = np.array([one(w @ stats[a], w @ stats[b]) for w in W])
    return dict(diff=float(full[0]), diff_lo=ci(bs[:, 0])[0], diff_hi=ci(bs[:, 0])[1],
                d=float(full[1]), d_lo=ci(bs[:, 1])[0], d_hi=ci(bs[:, 1])[1],
                n_a=int(stats[a][:, 0].sum()), n_b=int(stats[b][:, 0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('states')
    ap.add_argument('reports')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    samples, ps, pm, diag, secs = load(a.states)
    games = samples.game.values
    ug = np.unique(games)
    print(f'{len(ug)} games, {len(samples)} sampled states, {len(pm)} completed passes with both states '
          f'({len(ps)} state rows); extraction {np.mean(secs):.1f} s/game')
    print('target pts: mean %.3f sd %.3f; distribution %s' % (samples.pts.mean(), samples.pts.std(),
          samples.pts.value_counts(normalize=True).sort_index().round(3).to_dict()))
    W = boot_weights(len(ug), B_BOOT, a.seed)
    oof, ps_pred, fold_id, stage_sse, imps = fit_cv(samples, ps, a.folds, a.seed)
    y = samples.pts.values.astype(float)
    os.makedirs(a.reports, exist_ok=True)
    print(f'fit+CV {time.perf_counter() - t0:.0f} s')

    # ---------------- V(s) accuracy
    print('\n== out-of-fold accuracy (GroupKFold by game, %d folds) ==' % a.folds)
    res = {k: metrics(y, oof[k]) for k in ('const', 'base', 'rich')}
    for k, v in res.items():
        print(f'{k:6s} RMSE {v["rmse"]:.4f}  R2 {v["r2"]:.4f}  mean pred {v["mean_pred"]:.3f} (obs {v["mean_obs"]:.3f})')
    per_fold = pd.DataFrame([dict(fold=f, **{f'{k}_{m}': metrics(y[fold_id == f], oof[k][fold_id == f])[m]
                                             for k in ('const', 'base', 'rich') for m in ('rmse', 'r2')})
                             for f in range(a.folds)])
    print(per_fold.round(4).to_string(index=False))
    # game-clustered bootstrap of the rich - baseline gain
    gi = np.unique(games, return_inverse=True)[1]
    n_g = np.bincount(gi).astype(float)
    sse = {k: np.bincount(gi, weights=(y - oof[k]) ** 2) for k in oof}
    sy, syy = np.bincount(gi, weights=y), np.bincount(gi, weights=y * y)
    def rmse_r2(w, k):
        n = w @ n_g
        s = w @ sse[k]
        sst = w @ syy - (w @ sy) ** 2 / n
        return (s / n) ** 0.5, 1 - s / sst
    full_w = np.ones(len(ug))
    boot = {k: np.array([rmse_r2(w, k) for w in W]) for k in ('const', 'base', 'rich')}
    for a_, b_ in (('base', 'rich'), ('const', 'base'), ('const', 'rich')):
        d_rmse = boot[a_][:, 0] - boot[b_][:, 0]
        d_r2 = boot[b_][:, 1] - boot[a_][:, 1]
        f_rmse = rmse_r2(full_w, a_)[0] - rmse_r2(full_w, b_)[0]
        f_r2 = rmse_r2(full_w, b_)[1] - rmse_r2(full_w, a_)[1]
        print(f'{b_} vs {a_}: RMSE gain {f_rmse:+.4f} [{ci(d_rmse)[0]:+.4f}, {ci(d_rmse)[1]:+.4f}]  '
              f'R2 gain {f_r2:+.4f} [{ci(d_r2)[0]:+.4f}, {ci(d_r2)[1]:+.4f}]  '
              f'share of resamples where {b_} is better: {(d_rmse > 0).mean():.3f}')
    # sensitivity to the number of boosting iterations
    print('\niterations vs pooled out-of-fold RMSE (fixed a priori setting is max_iter=%d)' % HGB['max_iter'])
    stages = pd.DataFrame({k: {i: (s / len(y)) ** 0.5 for i, s in stage_sse[k].items()} for k in stage_sse})
    print(stages.round(4).T.to_string())
    stages.to_csv(os.path.join(a.reports, 'ws5b-value-iterations.csv'))
    # calibration
    for k in ('rich', 'base'):
        cal = calibration(y, oof[k], games, W)
        sl, (slo, shi) = calibration_slope(y, oof[k], games, W)
        print(f'\n== calibration by decile of out-of-fold prediction: {k} (slope of outcome on prediction {sl:.3f} [{slo:.3f}, {shi:.3f}]) ==')
        print(cal.round(3).to_string(index=False))
        cal.to_csv(os.path.join(a.reports, f'ws5b-value-calibration-{k}.csv'), index=False)
    # feature importance and a few interpretability slices
    imp = imps.mean().sort_values(ascending=False)
    print('\n== permutation importance (increase in held-out MSE, mean over folds) ==')
    print(imp.round(5).to_string())
    imp.to_csv(os.path.join(a.reports, 'ws5b-value-importance.csv'), header=['mse_increase'])
    sl = samples.assign(pred=oof['rich'])
    sl['dist_bin'] = pd.cut(sl.ball_dist, [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 95])
    sl['sc_bin'] = pd.cut(sl.shot_clock, [0, 4, 8, 12, 16, 20, 24])
    for col in ('dist_bin', 'sc_bin'):
        t = sl.groupby(col, observed=True).agg(n=('pts', 'size'), obs=('pts', 'mean'), pred=('pred', 'mean'))
        print(f'\nmean outcome and V by {col}'); print(t.round(3).to_string())
        t.to_csv(os.path.join(a.reports, f'ws5b-value-by-{col}.csv'))
    per_fold.to_csv(os.path.join(a.reports, 'ws5b-value-per-fold.csv'), index=False)
    pd.DataFrame(res).T.to_csv(os.path.join(a.reports, 'ws5b-value-oof-metrics.csv'))

    # ---------------- delta V
    print('\n== pass value: delta V = V(reception) - V(release), completed passes, out-of-fold ==')
    rel = ps[ps.kind == 'release'].set_index(['game', 'pass_id'])
    rec = ps[ps.kind == 'recv'].set_index(['game', 'pass_id'])
    pm2 = pm.set_index(['game', 'pass_id'])
    rel_i = pd.Series(np.arange(len(ps)), index=pd.MultiIndex.from_frame(ps[['game', 'pass_id', 'kind']]))
    dv = {}
    for name in ('rich', 'base'):
        pr = pd.Series(ps_pred[name], index=pd.MultiIndex.from_frame(ps[['game', 'pass_id', 'kind']]))
        v_rel = pr.xs('release', level='kind')
        v_rec = pr.xs('recv', level='kind')
        dv[name] = (v_rec - v_rel).rename(f'dv_{name}')
        dv[f'v_rel_{name}'] = v_rel.rename(f'v_rel_{name}')
        dv[f'v_rec_{name}'] = v_rec.rename(f'v_rec_{name}')
    P = pm2.join(pd.concat(dv.values(), axis=1), how='inner').reset_index()
    P['d_ball_dist'] = (rec.ball_dist.reindex(pd.MultiIndex.from_frame(P[['game', 'pass_id']])).values
                        - rel.ball_dist.reindex(pd.MultiIndex.from_frame(P[['game', 'pass_id']])).values)
    P['d_shot_clock'] = (rec.shot_clock.reindex(pd.MultiIndex.from_frame(P[['game', 'pass_id']])).values
                         - rel.shot_clock.reindex(pd.MultiIndex.from_frame(P[['game', 'pass_id']])).values)
    P = P.dropna(subset=['dv_rich', 'dv_base'])
    P['group'] = np.select([P.assist, (~P.assist) & P.receiver_shoots_next & (P.next_made == 0),
                            (~P.assist) & P.receiver_shoots_next & (P.next_made == 1)],
                           ['A_assist', 'B_next_shot_missed', 'C_next_shot_made_unassisted'], 'D_no_shot_by_receiver')
    P.to_csv('/private/tmp/ws5b/pass_dv.csv.gz', index=False)
    print(f'{len(P)} passes scored (of {len(pm)}); mean dV rich {P.dv_rich.mean():+.4f} (sd {P.dv_rich.std():.3f}), '
          f'baseline {P.dv_base.mean():+.4f}; mean change in shot clock {P.d_shot_clock.mean():+.2f} s; '
          f'mean change in ball distance to basket {P.d_ball_dist.mean():+.2f} ft')
    print('correlation of dV(rich) with the drop in ball distance to the basket: %.3f' %
          np.corrcoef(P.dv_rich, -P.d_ball_dist.fillna(0))[0, 1])
    pg = P.game.values
    ug2 = np.unique(pg)
    W2 = boot_weights(len(ug2), B_BOOT, a.seed + 1)
    summary = []
    for name in ('dv_rich', 'dv_base'):
        v = P[name].values
        masks = {g: (P.group == g).values for g in sorted(P.group.unique())}
        masks['E_all_not_assist'] = ~P.assist.values
        masks['ALL'] = np.ones(len(P), bool)
        st = group_means(v, masks, pg)
        print(f'\n-- {name}: mean by group (game-bootstrap 95% CI) --')
        for g, S in st.items():
            n, s = S[:, 0], S[:, 1]
            full = s.sum() / n.sum()
            bs = (W2 @ s) / (W2 @ n)
            lo, hi = ci(bs)
            print(f'  {g:30s} n={int(n.sum()):6d}  mean {full:+.4f} [{lo:+.4f}, {hi:+.4f}]')
            summary.append(dict(model=name, group=g, n=int(n.sum()), mean=full, lo=lo, hi=hi))
        print(f'-- {name}: contrasts (difference in mean dV, Cohen d, game-bootstrap 95% CI) --')
        for x, y_ in (('A_assist', 'E_all_not_assist'), ('A_assist', 'D_no_shot_by_receiver'),
                      ('A_assist', 'B_next_shot_missed'), ('A_assist', 'C_next_shot_made_unassisted'),
                      ('B_next_shot_missed', 'D_no_shot_by_receiver'), ('C_next_shot_made_unassisted', 'B_next_shot_missed')):
            c = contrast(st, x, y_, W2)
            print(f'  {x} - {y_}: diff {c["diff"]:+.4f} [{c["diff_lo"]:+.4f}, {c["diff_hi"]:+.4f}]  d {c["d"]:+.3f} '
                  f'[{c["d_lo"]:+.3f}, {c["d_hi"]:+.3f}]  (n {c["n_a"]} vs {c["n_b"]})')
            summary.append(dict(model=name, contrast=f'{x} - {y_}', **c))
    pd.DataFrame(summary).to_csv(os.path.join(a.reports, 'ws5b-passvalue-summary.csv'), index=False)
    json.dump(dict(games=len(ug), samples=len(samples), passes=len(P), hgb=HGB), open(os.path.join(a.reports, 'ws5b-value-run.json'), 'w'))
    print(f'\ntotal {time.perf_counter() - t0:.0f} s')


if __name__ == '__main__':
    main()
