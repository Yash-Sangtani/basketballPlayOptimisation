"""Shared helpers for the WS3 selection-caveat tests (dev games only; the frozen test list is never read).

Run everything from Code/ with the repo venv.  Heavy runs go through ./heavy.sh.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..'))
BASELINE = '/Users/yash/dev/basketballPlayOptimisation/Tracking_Data/pass_features'   # READ-ONLY shared baseline
RES = os.environ.get('WS3_RES', '/private/tmp/ws3/results')
os.makedirs(RES, exist_ok=True)

FEATS = ['shot_clock', 'backcourt', 'dist', 'passer_basket_dist', 'ball_end_basket_dist', 'basket_angle',
         'closest_def_dist_passer', 'closest_def_dist_ball_end', 'closest_def_trajectory',
         'closest_def_angle_passer', 'closest_def_angle_ball_end', 'closest_def_trajectory_angle',
         'lane_n_def_5ft', 'passer_speed', 'target_speed']
PASSER = ['shot_clock', 'backcourt', 'passer_basket_dist', 'closest_def_dist_passer', 'passer_speed']
PHASE = ['shot_clock', 'backcourt', 'passer_basket_dist', 'ball_end_basket_dist', 'basket_angle']
DEFENDER = ['closest_def_dist_passer', 'closest_def_dist_ball_end', 'closest_def_trajectory',
            'closest_def_angle_passer', 'closest_def_angle_ball_end', 'closest_def_trajectory_angle', 'lane_n_def_5ft']
SUBSETS = {
    'shot_clock': ['shot_clock'],
    'dist': ['dist'],
    'shot_clock+dist': ['shot_clock', 'dist'],
    'shot_clock+dist+backcourt': ['shot_clock', 'dist', 'backcourt'],
    'passer-only (no target)': PASSER,
    'phase/location only': PHASE,
    'defenders only': DEFENDER,
    'all 15': FEATS,
    'all minus defenders': [f for f in FEATS if f not in DEFENDER],
    'all minus phase': [f for f in FEATS if f not in PHASE],
}


def log(*a):
    print(*a, flush=True)


def save_json(name, obj):
    with open(f'{RES}/{name}.json', 'w') as f:
        json.dump(obj, f, indent=1, default=float)


# ------------------------------------------------------------------ data
def fn(game):
    """'10.27.2015.CLE.at.CHI' -> pickle stem '10.27.2015.CLE.CHI'."""
    p = game.split('.')
    return f"{'.'.join(p[:3])}.{p[3]}.{p[5]}"


def read_list(path):
    return [l.strip().replace('.7z', '') for l in open(path) if l.strip()]


def dev_games():
    return read_list(os.path.join(CODE, 'splits', 'dev_games.txt'))   # NEVER read test_games.txt


def load_baseline(games=None):
    """Shared baseline pickles for the given (dev) games -> X (features + game + fail), M (meta)."""
    games = games or dev_games()
    xs, ms = [], []
    for g in games:
        f = pd.read_pickle(f'{BASELINE}/{fn(g)}.pkl')
        m = pd.read_pickle(f'{BASELINE}/meta/{fn(g)}.pkl')
        assert (f.index == m.index).all()
        f['game'] = g
        xs.append(f)
        ms.append(m)
    X = pd.concat(xs, ignore_index=True)
    M = pd.concat(ms, ignore_index=True)
    X['fail'] = (X['pass_successful'] == 0).astype(int)
    return X, M


def fold_map(games, n=5):
    """game -> fold id, from GroupKFold on one row per game so every experiment on the same games
    uses identical folds (paired comparisons)."""
    games = sorted(set(games))
    dummy = pd.DataFrame({'g': games})
    fm = {}
    for k, (_, te) in enumerate(GroupKFold(n_splits=n).split(dummy, groups=dummy['g'])):
        for i in te:
            fm[games[i]] = k
    return fm


# ---------------------------------------------------------------- models
def make_model(name):
    if name == 'logit':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                             LogisticRegression(class_weight='balanced', max_iter=1000))
    if name == 'gb':     # the model eval_by_game.py used
        return make_pipeline(SimpleImputer(strategy='median'), GradientBoostingClassifier(subsample=0.5, random_state=0))
    if name == 'hgb':    # fast stand-in for gb (same family), native NaN handling
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=0)
    raise ValueError(name)


def oof_predict(X, y, folds, model='hgb'):
    """Out-of-fold scores; `folds` is an integer fold id per row (grouped by game upstream)."""
    oof = np.full(len(X), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        m = make_model(model).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def auc_pr(y, p):
    ok = ~np.isnan(p)
    y, p = np.asarray(y)[ok], np.asarray(p)[ok]
    if y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, p), average_precision_score(y, p)


def _game_index(groups):
    g = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(lambda s: s.values)
    return list(g.values)


def shuffle_within_groups(y, groups, seed):
    """Permute labels inside each group (game): keeps every game's base rate, destroys any feature link."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    ys = y.copy()
    for idx in _game_index(groups):
        ys[idx] = y[idx][rng.permutation(len(idx))]
    return ys


def boot_ci(y, p, groups, n=200, seed=0):
    """95% CI of pooled ROC and PR-AUC, resampling whole games with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    idx = _game_index(groups)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        if y[pick].min() == y[pick].max():
            continue
        out.append((roc_auc_score(y[pick], p[pick]), average_precision_score(y[pick], p[pick])))
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5], axis=0).T.tolist()   # [[roc_lo, roc_hi], [pr_lo, pr_hi]]


def paired_boot(y, pa, pb, groups, n=200, seed=0):
    """Paired game-level bootstrap of metric(pa) - metric(pb) on the SAME rows.
    Returns {'roc': (diff, lo, hi), 'pr': (diff, lo, hi)}."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    idx = _game_index(groups)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        if y[pick].min() == y[pick].max():
            continue
        d.append((roc_auc_score(y[pick], pa[pick]) - roc_auc_score(y[pick], pb[pick]),
                  average_precision_score(y[pick], pa[pick]) - average_precision_score(y[pick], pb[pick])))
    d = np.array(d)
    ra, pra = auc_pr(y, pa)
    rb, prb = auc_pr(y, pb)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {'roc': (ra - rb, lo[0], hi[0]), 'pr': (pra - prb, lo[1], hi[1])}


def fmt_ci(v, ci, nd=3):
    return f'{v:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]'


# ------------------------------------------------------- sample extraction
def load_sample(out_dir=None, games=None):
    """Load the per-game WS3 extraction files into pooled frames.
    Returns dict(ext, solo, feat={'actual': df, 'heading': df}, meta=..., np_events, pbp={game: df}, report)
    with a global row id `rid` (game + index) shared by all tables."""
    out_dir = out_dir or os.environ.get('WS3_OUT', '/private/tmp/ws3/out')
    files = sorted(glob.glob(f'{out_dir}/*.pkl'))
    if games is not None:
        keep = set(games)
        files = [f for f in files if os.path.basename(f)[:-4] in keep]
    ext, solo, npe, rep, pbp = [], [], [], [], {}
    feat = {'actual': [], 'heading': []}
    for f in files:
        d = pickle.load(open(f, 'rb'))
        g = os.path.basename(f)[:-4]
        e = d['ext'].copy()
        e['idx'] = e.index
        e['rid'] = g + ':' + e.index.astype(str)
        ext.append(e)
        s = d['solo'].copy()
        s['rid'] = g + ':' + s.index.astype(str)
        solo.append(s)
        for mode in feat:
            ft, mt = d['feats'][mode]
            ft = ft.copy()
            ft['rid'] = g + ':' + ft.index.astype(str)
            ft['game'] = g
            feat[mode].append(ft)
        npe.append(d['np_events'])
        rep.append(d['report'])
        pbp[g] = d['pbp']
    ext = pd.concat(ext, ignore_index=True)
    solo = pd.concat(solo, ignore_index=True).set_index('rid')
    feat = {m: pd.concat(v, ignore_index=True).set_index('rid') for m, v in feat.items()}
    return dict(ext=ext.set_index('rid', drop=False), solo=solo, feat=feat, np_events=pd.concat(npe, ignore_index=True),
                pbp=pbp, report=pd.DataFrame(rep))
