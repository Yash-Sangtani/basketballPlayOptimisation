"""Stage 2: shot value (expected points of a shot from a court state) and the value-weighted pass score.

    cd Code && python difficulty/shot_value.py --data DIR --out ../reports/difficulty-dev        # dev CV + assist check
    cd Code && python difficulty/shot_value.py --data DIR --fit-final --model-out difficulty/model

Shot model: for every pbp field-goal attempt matched to the shooter's release (extract.py `shots` table) the features of
the moment of release (distance/angle to the basket, two nearest defenders, nearest defender between shooter and basket,
shooter speed, shot clock) -> P(make); and P(three | distance, angle). xPts = P(make) * (2 + P(three)).

Value-weighted pass score (everything in expected points, per candidate pass from one court state):
    threat_receiver  = xPts if the intended receiver shot from where he stands at release
    threat_passer    = xPts if the passer shot now
    ev_pass          = p_complete * threat_receiver  -  p_fail * C_FAIL
    pass_minus_shoot = ev_pass - threat_passer
C_FAIL is the points swing of losing the ball: the possession's mean points plus the transition bonus the other team
gets after a live-ball turnover. Measured on the 501 dev games' possession table (94,762 possessions): mean 1.050 points
per possession; the next possession scores 1.137 after a turnover vs 1.01-1.04 after other endings -> C_FAIL = 1.16.

Limits (they matter for RL): `threat` is the value of a shot taken right now, not of everything the receiver could do
next (drive, pass again), so ev_pass is a lower bound that is only meaningful in the frontcourt; the receiver is taken
at his position at the release instant, not where he will be when the ball arrives.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_eval as T                                               # noqa: E402
from pass_features import SHOT_FEATURE_NAMES                         # noqa: E402

C_FAIL = 1.16
HGB = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, random_state=0)
MAKE_FEATURES = SHOT_FEATURE_NAMES
LOC_FEATURES = ['shot_dist', 'shot_angle']


def log(*a):
    print(*a, flush=True)


def load_shots(data_dir, games):
    ps = [pd.read_pickle(os.path.join(data_dir, f'{g}.pkl'))['shots'] for g in games if os.path.exists(os.path.join(data_dir, f'{g}.pkl'))]
    S = pd.concat(ps, ignore_index=True)
    return S[(S['shot_dist'] < 60)].reset_index(drop=True)          # drop heaves / bad matches


def fit_models(S):
    """-> dict(make, make_iso, three). Isotonic on a held-out share of games (same recipe as the pass model)."""
    g = S['game'].to_numpy()
    cal_g, fit_g = T.split_inner(g, 0.25, 0)
    fit = np.isin(g, list(fit_g))
    X, y = S[MAKE_FEATURES].to_numpy(float), S['made'].to_numpy()
    m = HistGradientBoostingClassifier(**HGB).fit(X[fit], y[fit])
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.01, y_max=0.99).fit(m.predict_proba(X[~fit])[:, 1], y[~fit])
    three = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, random_state=0).fit(S[LOC_FEATURES].to_numpy(float), S['is3'].to_numpy())
    return dict(make=m, make_iso=iso, three=three)


def xpts(models, F):
    """F: DataFrame with SHOT_FEATURE_NAMES -> DataFrame(p_make, p_three, xpts)."""
    pm = models['make_iso'].predict(models['make'].predict_proba(F[MAKE_FEATURES].to_numpy(float))[:, 1])
    p3 = models['three'].predict_proba(F[LOC_FEATURES].to_numpy(float))[:, 1]
    return pd.DataFrame({'p_make': pm, 'p_three': p3, 'xpts': pm * (2 + p3)}, index=F.index)


def cv_shots(S, folds=5, seed=0):
    g = S['game'].to_numpy()
    ug = np.array(sorted(set(g)))
    perm = np.random.default_rng(seed).permutation(len(ug))
    fold_of = {ug[i]: k % folds for k, i in enumerate(perm)}
    f = np.array([fold_of[x] for x in g])
    return f, fold_of


def shot_report(S, out):
    f, _ = cv_shots(S)
    y = S['made'].to_numpy()
    res = {}
    preds = {k: np.full(len(S), np.nan) for k in ('league', 'distance_logit', 'hgb_raw', 'hgb_cal')}
    xp = np.full(len(S), np.nan)
    for k in range(5):
        tr, te = f != k, f == k
        preds['league'][te] = y[tr].mean()
        lg = LogisticRegression(max_iter=1000).fit(np.c_[S.loc[tr, 'shot_dist'], S.loc[tr, 'shot_dist'] ** 2], y[tr])
        preds['distance_logit'][te] = lg.predict_proba(np.c_[S.loc[te, 'shot_dist'], S.loc[te, 'shot_dist'] ** 2])[:, 1]
        mods = fit_models(S[tr].reset_index(drop=True))
        x = xpts(mods, S[te])
        preds['hgb_cal'][te] = x['p_make'].to_numpy()
        preds['hgb_raw'][te] = mods['make'].predict_proba(S.loc[te, MAKE_FEATURES].to_numpy(float))[:, 1]
        xp[te] = x['xpts'].to_numpy()
    base_ll = None
    for k, p in preds.items():
        p = np.clip(p, 1e-4, 1 - 1e-4)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        br = float(np.mean((p - y) ** 2))
        if k == 'league':
            base_ll, base_br = ll, br
        res[k] = dict(logloss=ll, brier_skill=1 - br / base_br, roc=float(roc_auc_score(y, p)) if k != 'league' else 0.5)
        log(f'  shot P(make) {k:15s} logloss {ll:.4f}  Brier skill vs league avg {1 - br / base_br:+.4f}  ROC {res[k]["roc"]:.3f}')
    q = pd.qcut(pd.Series(preds['hgb_cal']).rank(method='first'), 10, labels=False)
    rel = pd.DataFrame({'pred': pd.Series(preds['hgb_cal']).groupby(q).mean(), 'observed': pd.Series(y).groupby(q).mean()}).round(3)
    log('  calibration by decile (hgb, calibrated):\n' + rel.to_string())
    pts = np.where(S['made'] == 1, np.where(S['is3'] == 1, 3, 2), 0)
    log(f"  xPts vs realised points per shot: predicted mean {xp.mean():.3f}, actual {pts.mean():.3f}; "
        f"corr(xPts, points) {np.corrcoef(xp, pts)[0, 1]:.3f}")
    res['xpts'] = dict(pred_mean=float(xp.mean()), actual_mean=float(pts.mean()), corr=float(np.corrcoef(xp, pts)[0, 1]))
    by = S.assign(xp=xp).groupby(pd.cut(S['shot_dist'], [0, 4, 10, 16, 22, 25, 30, 60]), observed=True).agg(
        n=('made', 'size'), fg=('made', 'mean'), xpts=('xp', 'mean'))
    log('  by distance:\n' + by.round(3).to_string())
    res['by_distance'] = by.round(4).reset_index().astype(str).to_dict('records')
    return res, f


def assist_check(D, S, models_by_fold, game_fold, out):
    """Does the receiver's shot threat at the moment of release identify the passes that become assists?
    D: clean completed passes of the tuning games with rs_* features; S: matched shots. Cross-fitted: a game's rows are
    scored by a shot model that never saw that game."""
    C = D[(D['clean'] == 1) & (D['fail'] == 0)].reset_index(drop=True)
    C['fold'] = C['game'].map(game_fold)
    C = C[C['fold'].notna()].reset_index(drop=True)
    F = C[['rs_shot_dist', 'rs_shot_angle', 'rs_def1', 'rs_def2', 'rs_def_cos', 'rs_shooter_speed', 'shot_clock']].rename(
        columns=lambda c: c[3:] if c.startswith('rs_') else c)
    Fp = C[['ps_shot_dist', 'ps_shot_angle', 'ps_def1', 'ps_def2', 'ps_def_cos', 'ps_shooter_speed', 'shot_clock']].rename(
        columns=lambda c: c[3:] if c.startswith('ps_') else c)
    C['threat_recv'], C['threat_pass'] = np.nan, np.nan
    for k, mods in models_by_fold.items():
        m = (C['fold'] == k).to_numpy()
        C.loc[m, 'threat_recv'] = xpts(mods, F[m])['xpts'].to_numpy()
        C.loc[m, 'threat_pass'] = xpts(mods, Fp[m])['xpts'].to_numpy()
    # assist link: a matched assisted make by the intended-or-actual receiver, credited to the passer, 0-4 s after the release
    a = S[(S['assisted'] == 1) & (S['made'] == 1)][['game', 'quarter', 'shooter', 'assister', 'quarter_time']]
    C['assist'] = 0
    key = {}
    for r in a.itertuples():
        key.setdefault((r.game, r.quarter, r.assister, r.shooter), []).append(r.quarter_time)
    hit = np.zeros(len(C), dtype=int)
    for i, r in enumerate(C.itertuples()):
        qts = key.get((r.game, r.quarter, r.passer, r.actual_target))
        if qts and any(0 <= r.quarter_time - q <= 6 for q in qts):
            hit[i] = 1
    C['assist'] = hit
    y = C['assist'].to_numpy()
    res = dict(n_completions=int(len(C)), n_assist_linked=int(y.sum()))
    log(f"\nassist check: {len(C)} clean completed passes, {int(y.sum())} linked to an assisted make ({y.mean():.4f})")
    for name, col in (('receiver threat xPts (shot model)', 'threat_recv'), ('receiver distance to basket (negated)', 'neg_dist'),
                      ('receiver nearest defender distance', 'rs_def1'), ('passer threat xPts', 'threat_pass')):
        v = -C['rs_shot_dist'] if col == 'neg_dist' else C[col]
        ok = v.notna().to_numpy()
        auc = roc_auc_score(y[ok], v[ok])
        res[col] = float(auc)
        log(f'  AUC for becoming an assist: {name:40s} {auc:.3f}')
    d = C['threat_recv'] - C['threat_pass']
    ok = d.notna().to_numpy()
    res['gain_auc'] = float(roc_auc_score(y[ok], d[ok]))
    log(f"  AUC for becoming an assist: receiver threat minus passer threat (the pass's threat gain)  {res['gain_auc']:.3f}")
    res['mean_threat'] = dict(assist=float(C.loc[C.assist == 1, 'threat_recv'].mean()), other=float(C.loc[C.assist == 0, 'threat_recv'].mean()))
    log(f"  mean receiver threat: assist-linked {res['mean_threat']['assist']:.3f} vs other completions {res['mean_threat']['other']:.3f} xPts")
    if out:
        json.dump(res, open(os.path.join(out, 'assist_check.json'), 'w'), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model-out', default=os.path.join(HERE, 'model'))
    ap.add_argument('--fit-final', action='store_true')
    a = ap.parse_args()
    tune, _ = T.split_games()
    S = load_shots(a.data, tune)
    log(f'{S.game.nunique()} tuning games, {len(S)} matched shots, FG {S.made.mean():.3f}, 3PA share {S.is3.mean():.3f}, assisted share of makes {S[S.made == 1].assisted.mean():.3f}')
    if a.fit_final:
        mods = fit_models(S)
        os.makedirs(a.model_out, exist_ok=True)
        joblib.dump(dict(models=mods, features=MAKE_FEATURES, loc_features=LOC_FEATURES, c_fail=C_FAIL,
                         trained_on=dict(games=int(S.game.nunique()), shots=int(len(S)))), os.path.join(a.model_out, 'shot_v1.joblib'))
        log('saved shot model')
        return
    os.makedirs(a.out, exist_ok=True) if a.out else None
    log('\nshot model, GroupKFold(5) by game:')
    res, f = shot_report(S, a.out)
    if a.out:
        json.dump(res, open(os.path.join(a.out, 'shot_metrics.json'), 'w'), indent=1, default=float)
    # cross-fitted models for the assist check
    ug = np.array(sorted(set(S['game'])))
    perm = np.random.default_rng(0).permutation(len(ug))
    game_fold = {ug[i]: k % 5 for k, i in enumerate(perm)}
    folds = np.array([game_fold[g] for g in S['game']])
    mods = {k: fit_models(S[folds != k].reset_index(drop=True)) for k in range(5)}
    D = T.load(a.data, tune, T.REGIMES['narrow'])
    assist_check(D, S, mods, game_fold, a.out)


if __name__ == '__main__':
    main()
