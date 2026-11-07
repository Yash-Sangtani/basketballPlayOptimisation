"""Train and evaluate the pass-difficulty (completion-probability) model on DEV games only.

    cd Code && python difficulty/train_eval.py --data DIR --out ../reports/difficulty-dev            # dev CV
    cd Code && python difficulty/train_eval.py --data DIR --fit-final --model-out difficulty/model     # fit on the tuning games
    cd Code && python difficulty/train_eval.py --data DIR --score-holdout --model-out difficulty/model # ONE look at the 100 held-out dev games

Games: splits/dev_games.txt minus splits/difficulty_final_games.txt (the "tuning" games, 401) are used for all CV and model
choice. splits/difficulty_final_games.txt (100 dev games, drawn once with a fixed seed) is scored once at the end, and the
frozen splits/test_games.txt is never read here.

Population ("clean passes"): live (clock running, passer in bounds), and no pbp-flagged non-pass context for the releaser
(own shot, rebound/block scramble, own lost ball / travel / offensive foul). Positive class for scoring = FAILED pass
(label 0); the model's output is P(fail), difficulty = P(fail), p_complete = 1 - P(fail).
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pass_features import FEATURE_NAMES, SITUATION_FEATURES      # noqa: E402

SPLITS = os.path.join(HERE, '..', 'splits')
ORIGINAL15 = FEATURE_NAMES[:15]
RECEIVER_SIDE = ['dist', 'ball_end_basket_dist', 'basket_angle', 'closest_def_dist_ball_end', 'closest_def_trajectory',
                 'closest_def_angle_passer', 'closest_def_angle_ball_end', 'closest_def_trajectory_angle', 'lane_n_def_5ft',
                 'target_speed', 'intercept_margin', 'n_def_can_intercept', 'second_def_dist_ball_end', 'recv_speed_along']
FEATURE_SETS = {'situation': SITUATION_FEATURES, 'original15': ORIGINAL15, 'full': FEATURE_NAMES}
HGB = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def log(*a):
    print(*a, flush=True)


def read_list(path):
    return [l.strip().replace('.7z', '') for l in open(path) if l.strip()]


def split_games():
    dev = read_list(os.path.join(SPLITS, 'dev_games.txt'))
    final = read_list(os.path.join(SPLITS, 'difficulty_final_games.txt'))
    assert set(final) <= set(dev) and len(final) == 100
    return [g for g in dev if g not in set(final)], final


# pbp exclusion windows in seconds, lag = release clock - event clock (negative: the event is before the release).
# Masks from extract.py have one bit per second from -5 to +8; edges here are integers.
WINDOWS = dict(shot=(-2, 4), reb=(-3, 4), blk=(-3, 4), dead=(-4, 1), np_to=(-3, 8), badpass=(-3, 8))
REGIMES = {'wide': {}, 'narrow': dict(reb=(-3, 1), blk=(-3, 1), dead=(-2, 1))}
MASK_COLS = dict(shot='m_shot_by_releaser', reb='m_reb', blk='m_blk', dead='m_dead', np_to='m_to_np_by_releaser',
                 badpass='m_to_badpass_by_releaser')


def in_window(mask, lo, hi):
    """True where an event of this class fell in [lo, hi) seconds around the release."""
    bits = 0
    for k in range(lo + 5, hi + 5):
        bits |= 1 << k
    return (np.asarray(mask, dtype=np.int64) & bits) != 0


def load(data_dir, games, windows=None, drop_fumbled=False):
    w = dict(WINDOWS, **(windows or {}))
    ps = []
    for g in games:
        p = os.path.join(data_dir, f'{g}.pkl')
        if os.path.exists(p):
            ps.append(pd.read_pickle(p)['passes'])
    D = pd.concat(ps, ignore_index=True)
    for k, col in MASK_COLS.items():
        D['x_' + k] = in_window(D[col], *w[k]).astype(int)
    D['pbp_badpass'] = D['x_badpass']
    D['excl_nonpass'] = ((D['x_shot'] | D['x_reb'] | D['x_blk'] | D['x_dead'] | D['x_np_to']) == 1).astype(int)
    D['clean'] = ((D['live'] == 1) & (D['excl_nonpass'] == 0)).astype(int)
    if drop_fumbled:
        D.loc[D['outcome'] == 'fumbled', 'clean'] = 0
    D['fail'] = (D['label'] == 0).astype(int)
    return D


def make_model(kind):
    if kind == 'logit':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == 'hgb':
        return HistGradientBoostingClassifier(**HGB)
    raise ValueError(kind)


def split_inner(games, frac, seed):
    rng = np.random.default_rng(seed)
    g = np.array(sorted(set(games)))
    rng.shuffle(g)
    k = max(1, int(round(len(g) * frac)))
    return set(g[:k]), set(g[k:])


def fit_calibrated(X, y, games, kind, seed=0, cal_frac=0.25):
    """Fit on a share of the training games, isotonic-calibrate on the remaining games."""
    cal_g, fit_g = split_inner(games, cal_frac, seed)
    fit = np.isin(games, list(fit_g))
    m = make_model(kind).fit(X[fit], y[fit])
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(m.predict_proba(X[~fit])[:, 1], y[~fit])
    return m, iso


def cv_scores(D, cols, kind, folds=5, seed=0, train_ok=None):
    """Out-of-fold raw and calibrated P(fail) for every row of D (GroupKFold by game, shuffled by a fixed seed)."""
    X, y, g = D[cols].to_numpy(float), D['fail'].to_numpy(), D['game'].to_numpy()
    ug = np.array(sorted(set(g)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ug))
    fold_of = {ug[i]: k % folds for k, i in enumerate(perm)}
    f = np.array([fold_of[x] for x in g])
    raw, cal = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for k in range(folds):
        tr, te = f != k, f == k
        if train_ok is not None:
            tr = tr & train_ok
        m, iso = fit_calibrated(X[tr], y[tr], g[tr], kind, seed + k)
        raw[te] = m.predict_proba(X[te])[:, 1]
        cal[te] = iso.predict(raw[te])
    return raw, cal, f


def metrics(y, p):
    base = y.mean()
    b = float(np.mean((p - y) ** 2))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(roc=float(roc_auc_score(y, p)), pr=float(average_precision_score(y, p)), prevalence=float(base),
                brier=b, bss=float(1 - b / (base * (1 - base))),
                logloss=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                logloss_base=float(-(base * np.log(base) + (1 - base) * np.log(1 - base))), mean_pred=float(p.mean()))


def boot_ci(y, p, g, n=200, seed=0):
    ug, gi = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(gi == k) for k in range(len(ug))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(ug), len(ug))])
        out.append((roc_auc_score(y[pick], p[pick]), average_precision_score(y[pick], p[pick])))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0).T.tolist()


def reliability(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method='first'), bins, labels=False)
    return pd.DataFrame({'n': pd.Series(y).groupby(q).size(), 'mean_pred': pd.Series(p).groupby(q).mean(),
                         'observed': pd.Series(y).groupby(q).mean()}).round(4)


def stratified_concordance(D, score, nq=3):
    """Within strata of identical passer situation, the share of completions that score LOWER than a failure
    (0.5 = no information about the pass itself beyond the situation)."""
    keys = [D['backcourt'].fillna(-1).astype(int).astype(str)]
    for c in ('passer_basket_dist', 'closest_def_dist_passer', 'shot_clock'):
        keys.append(pd.qcut(D[c], nq, labels=False, duplicates='drop').fillna(-1).astype(int).astype(str))
    strata = keys[0].str.cat(keys[1:], sep='_').values
    df = pd.DataFrame({'s': strata, 'sc': score, 'y': D['fail'].values})
    vals = []
    for _, d in df.groupby('s'):
        f, c = d[d.y == 1], d[d.y == 0]
        if len(f) == 0 or len(c) == 0:
            continue
        cs = np.sort(c.sc.values)
        lo, hi = np.searchsorted(cs, f.sc.values, 'left'), np.searchsorted(cs, f.sc.values, 'right')
        vals.extend(((lo + hi) / 2 / len(cs)).tolist())
    return float(np.mean(vals)), len(vals)


def dev_report(D, out, seed=0):
    C = D[D['clean'] == 1].reset_index(drop=True)
    y, g = C['fail'].to_numpy(), C['game'].to_numpy()
    log(f"tuning games {C['game'].nunique()}: {len(D)} flights, {len(C)} clean passes, {y.sum()} failures ({y.mean():.4f}); "
        f"pbp-confirmed bad passes among clean fails: {int(C.loc[C.fail == 1, 'pbp_badpass'].sum())}")
    res, oof = {}, {}
    for fs, cols in FEATURE_SETS.items():
        for kind in ('logit', 'hgb'):
            t0 = time.time()
            raw, cal, _ = cv_scores(C, cols, kind, seed=seed)
            m = metrics(y, cal)
            ci = boot_ci(y, raw, g)
            m.update(roc_raw=float(roc_auc_score(y, raw)), pr_raw=float(average_precision_score(y, raw)), roc_ci=ci[0], pr_ci=ci[1])
            res[f'{fs}|{kind}'] = m
            oof[f'{fs}|{kind}'] = (raw, cal)
            log(f"  {fs:10s} {kind:5s}  ROC {m['roc_raw']:.3f} [{ci[0][0]:.3f},{ci[0][1]:.3f}]  PR {m['pr_raw']:.4f} [{ci[1][0]:.4f},{ci[1][1]:.4f}] "
                f"(prev {m['prevalence']:.4f})  calibrated: BSS {m['bss']:.4f} logloss {m['logloss']:.4f} vs {m['logloss_base']:.4f}  [{time.time() - t0:.0f}s]")
    best = 'full|hgb'
    raw, cal = oof[best]
    log('\nreliability (full|hgb, calibrated):\n' + reliability(y, cal).to_string())

    # --- does the pass itself add to the situation? receiver-side-only model scored inside identical situations
    rs_raw, _, _ = cv_scores(C, RECEIVER_SIDE, 'hgb', seed=seed)
    conc, n = stratified_concordance(C, rs_raw)
    sit_conc, _ = stratified_concordance(C, oof['situation|hgb'][0])
    log(f"\nwithin-situation concordance of failures over completions: receiver-side-only model {conc:.3f} (n={n}); "
        f"situation-only model {sit_conc:.3f} (should be near 0.5)")
    res['strata'] = dict(receiver_side_only=conc, situation_only=sit_conc, n_fail=n)

    # --- cross-label check: never train on pbp-confirmed bad passes, then rank them against completions
    is_pbp = ((C.fail == 1) & (C.pbp_badpass == 1)).to_numpy()
    raw_t, _, _ = cv_scores(C, FEATURE_SETS['full'], 'hgb', seed=seed, train_ok=~is_pbp)
    sel = is_pbp | (C.fail == 0).to_numpy()
    a = roc_auc_score(is_pbp[sel].astype(int), raw_t[sel])
    log(f"cross-label: a model never trained on pbp-confirmed bad passes ranks them above completions with AUC {a:.3f} (n={int(is_pbp.sum())})")
    res['cross_label_auc'] = dict(auc=float(a), n=int(is_pbp.sum()))

    # --- non-pass turnovers (excluded from training) under the final model
    Dn = D[(D['live'] == 1) & (D['x_np_to'] == 1) & (D['x_shot'] == 0)]
    if len(Dn):
        m_, iso_ = fit_calibrated(C[FEATURE_SETS['full']].to_numpy(float), y, g, 'hgb', seed)
        sn = m_.predict_proba(Dn[FEATURE_SETS['full']].to_numpy(float))[:, 1]
        sc_c = m_.predict_proba(C.loc[C.fail == 0, FEATURE_SETS['full']].to_numpy(float))[:, 1]
        # in-sample for completions, so compare the two excluded-group scores only qualitatively
        log(f"non-pass turnovers (lost ball/travel/off. foul, excluded from training): n={len(Dn)}, mean P(fail) {sn.mean():.4f} "
            f"vs clean completions {sc_c.mean():.4f} vs clean failures {m_.predict_proba(C.loc[C.fail == 1, FEATURE_SETS['full']].to_numpy(float))[:, 1].mean():.4f}")
        res['nonpass'] = dict(n=int(len(Dn)), mean_p_fail=float(sn.mean()), mean_p_comp=float(sc_c.mean()))

    # --- sanity: observed failure rate by interpretable bins (calibrated OOF prediction beside it)
    tab = {}
    for c, bins in (('dist', [0, 8, 14, 20, 26, 34, 100]), ('intercept_margin', [-5, -0.3, -0.1, 0, 0.1, 0.3, 5]),
                    ('closest_def_trajectory', [0, 2, 4, 6, 9, 100]), ('closest_def_dist_ball_end', [0, 3, 6, 9, 14, 100])):
        b = pd.cut(C[c], bins)
        t = pd.DataFrame({'n': y, 'obs': y, 'pred': cal}).groupby(b, observed=True).agg(n=('n', 'size'), obs=('obs', 'mean'), pred=('pred', 'mean')).round(4)
        tab[c] = t
        log(f'\nfailure rate by {c}:\n' + t.to_string())
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, 'dev_metrics.json'), 'w'), indent=1, default=float)
        reliability(y, cal).to_csv(os.path.join(out, 'reliability_full_hgb.csv'))
        for c, t in tab.items():
            t.to_csv(os.path.join(out, f'by_{c}.csv'))
    return res


def fit_final(D, model_dir):
    C = D[D['clean'] == 1].reset_index(drop=True)
    y, g = C['fail'].to_numpy(), C['game'].to_numpy()
    cols = FEATURE_NAMES
    m, iso = fit_calibrated(C[cols].to_numpy(float), y, g, 'hgb', 0)
    ms, isos = fit_calibrated(C[SITUATION_FEATURES].to_numpy(float), y, g, 'hgb', 0)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(dict(model=m, iso=iso, features=cols, situation_model=ms, situation_iso=isos, situation_features=SITUATION_FEATURES,
                     trained_on=dict(games=int(C['game'].nunique()), passes=int(len(C)), failures=int(y.sum())),
                     definition='P(fail) = probability the pass is not caught by the intended teammate (intercepted / out of bounds / fumbled); '
                                'difficulty = P(fail); p_complete = 1 - P(fail). Clean live passes only.'),
                os.path.join(model_dir, 'difficulty_v1.joblib'))
    log(f"saved model trained on {C['game'].nunique()} games, {len(C)} passes, {int(y.sum())} failures")


def score_holdout(D, model_dir, out):
    bundle = joblib.load(os.path.join(model_dir, 'difficulty_v1.joblib'))
    C = D[D['clean'] == 1].reset_index(drop=True)
    y, g = C['fail'].to_numpy(), C['game'].to_numpy()
    raw = bundle['model'].predict_proba(C[bundle['features']].to_numpy(float))[:, 1]
    cal = bundle['iso'].predict(raw)
    sraw = bundle['situation_model'].predict_proba(C[bundle['situation_features']].to_numpy(float))[:, 1]
    scal = bundle['situation_iso'].predict(sraw)
    res = {}
    for name, r, c in (('full', raw, cal), ('situation_only', sraw, scal)):
        m = metrics(y, c)
        ci = boot_ci(y, r, g, n=500)
        m.update(roc_raw=float(roc_auc_score(y, r)), pr_raw=float(average_precision_score(y, r)), roc_ci=ci[0], pr_ci=ci[1])
        res[name] = m
        log(f"HOLDOUT {name:15s} ROC {m['roc_raw']:.3f} [{ci[0][0]:.3f},{ci[0][1]:.3f}]  PR {m['pr_raw']:.4f} [{ci[1][0]:.4f},{ci[1][1]:.4f}] "
            f"(prev {m['prevalence']:.4f})  calibrated BSS {m['bss']:.4f}, mean pred {m['mean_pred']:.4f}")
    log('reliability (calibrated, holdout):\n' + reliability(y, cal).to_string())
    conc, n = stratified_concordance(C, raw)
    res['strata_full_model'] = conc
    log(f'within-situation concordance of the full model {conc:.3f} (n={n})')
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, 'holdout_metrics.json'), 'w'), indent=1, default=float)
        reliability(y, cal).to_csv(os.path.join(out, 'reliability_holdout.csv'))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--model-out', default=os.path.join(HERE, 'model'))
    ap.add_argument('--fit-final', action='store_true')
    ap.add_argument('--regime', choices=list(REGIMES), default='wide', help='which pbp exclusion windows define a clean pass')
    ap.add_argument('--drop-fumbled', action='store_true', help='drop flights that end as a brief teammate touch then a loss')
    ap.add_argument('--score-holdout', action='store_true', help='ONE look at the 100 held-out dev games')
    a = ap.parse_args()
    tune, final = split_games()
    if a.score_holdout:
        log(f'scoring the {len(final)} held-out dev games (once)')
        score_holdout(load(a.data, final, REGIMES[a.regime], a.drop_fumbled), a.model_out, a.out)
    elif a.fit_final:
        fit_final(load(a.data, tune, REGIMES[a.regime], a.drop_fumbled), a.model_out)
    else:
        dev_report(load(a.data, tune, REGIMES[a.regime], a.drop_fumbled), a.out)


if __name__ == '__main__':
    main()
