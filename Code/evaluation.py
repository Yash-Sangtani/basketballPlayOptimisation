"""Honest, grouped-by-game evaluation of failed-pass prediction.

Everything here is a pure function or class: importing this module loads no data, imports no
TensorFlow and prints nothing. `pass_model.py` is the command-line front end.

Conventions
-----------
* Positive class = FAILED pass (`pass_successful == 0`); every score is P(fail).
* The unit of independence is the GAME (`meta.game`, e.g. '11.20.2015.SAS.at.NOP'). Passes in one
  game share players, lineups, pace and the detector's per-game quirks, so a row-level split
  would overstate skill. Every split below (outer folds, the calibration split, the MLP's early-
  stopping split) is by game, and `verify_split_log` re-checks that nothing straddles a split.
* Only the 15 columns in FEATURES are ever given to a model. The metadata columns `outcome`
  and `target_inferred` are label-equivalent and are kept out of X by an explicit whitelist.
* Per outer fold: games are split into outer-train / outer-test. Outer-train is split again, by
  game, into fit games and calibration games. The model is fit on the fit games only (the MLP
  also holds out validation games from the fit games for early stopping), an isotonic
  calibrator is fit on the calibration games only, and the outer-test games are scored with
  both the raw model and the calibrated model. Raw and calibrated predictions therefore come
  from the same base fit and differ only by the isotonic map. Outer-test labels are never read
  while predicting (a test sets them to NaN to prove it).
* Confidence intervals come from a game-clustered bootstrap: whole games are resampled with
  replacement. The same resamples are used for every model so differences are paired.
  The bootstrap captures test-set sampling noise only, not refit-to-refit variability.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_DIR = os.path.join(HERE, 'splits')
DEFAULT_PICKLES = os.path.join(HERE, '..', 'Tracking_Data', 'pass_features')

#Explicit whitelist: never "every column except the label" (see module docstring).
FEATURES = ['shot_clock', 'backcourt', 'dist', 'passer_basket_dist', 'ball_end_basket_dist',
            'basket_angle', 'closest_def_dist_passer', 'closest_def_dist_ball_end',
            'closest_def_trajectory', 'closest_def_angle_passer', 'closest_def_angle_ball_end',
            'closest_def_trajectory_angle', 'lane_n_def_5ft', 'passer_speed', 'target_speed']
#Features that depend only on the passer and the basket, not on who the receiver is. In the data, failures always
#have an INFERRED receiver (meta.target_inferred is True for every failure and False for every completion), so the
#other features are built from a different kind of target in the two classes. `--feature-set passer` keeps only these.
PASSER_FEATURES = ['shot_clock', 'backcourt', 'passer_basket_dist', 'closest_def_dist_passer', 'passer_speed']
LABEL = 'pass_successful'                      #1 = completed, 0 = failed
SIMPLE_FEATURES = ['shot_clock', 'dist']       #the "simple baseline" uses only these two
PROB_EPS = 1e-4                                #probability floor/ceiling for log-loss and isotonic output
T0 = time.perf_counter()


def stamp(msg: str) -> None:
    print(f"[time] {time.perf_counter() - T0:7.1f}s  {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Game lists and loading
# ----------------------------------------------------------------------------------------------
def read_game_list(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip().replace('.7z', '') for ln in f if ln.strip()]


def game_to_pickle_name(game: str) -> str:
    """'01.01.2016.CHA.at.TOR' -> '01.01.2016.CHA.TOR.pkl' (pickles drop the '.at')."""
    return game.replace('.at.', '.') + '.pkl'


def check_lists_disjoint(dev: list[str], test: list[str]) -> None:
    both = set(dev) & set(test)
    if both:
        raise AssertionError(f"{len(both)} games are in both dev and test lists, e.g. {sorted(both)[:3]}")


def load_status(pickle_dir: str) -> pd.DataFrame | None:
    """Latest `_status.csv` row per game (the file is appended to while a batch runs)."""
    p = os.path.join(pickle_dir, '_status.csv')
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, keep_default_na=False, on_bad_lines='skip', dtype=str)
    return s.drop_duplicates('game', keep='last').set_index('game')


@dataclass
class Dataset:
    X: np.ndarray              #(n, len(features)) float64, NaNs kept (imputed inside each fold)
    y: np.ndarray              #int8, 1 = failed pass
    g: np.ndarray              #game name per row (meta.game)
    features: list[str]
    outcome: np.ndarray        #meta.outcome, for sanity checks and reporting only, never a feature
    info: dict = field(default_factory=dict)
    live: np.ndarray | None = None   #meta.live (True = clock running and passer in bounds), None for older pickles

    @property
    def games(self) -> np.ndarray:
        return np.unique(self.g)

    def summary(self) -> str:
        n, f = len(self.y), int(self.y.sum())
        return f"{len(self.games)} games, {n} passes, {f} failures ({f / max(n, 1):.4%})"


def load_games(pickle_dir: str, games: list[str], check_status: bool = True) -> Dataset:
    """Load exactly the listed games (no other pickle in the folder is opened).

    A game is used only if its feature pickle AND its `meta/` pickle exist and (when
    `check_status`) the latest `_status.csv` row for it is `ok`. Everything skipped is listed in
    `info['skipped']`; the set of code versions of the loaded games is in `info['versions']`.
    """
    status = load_status(pickle_dir) if check_status else None
    if check_status and status is None:
        raise FileNotFoundError(f"{pickle_dir}/_status.csv not found (pass check_status=False to skip that check)")
    skipped = {'no_pickle': [], 'no_meta': [], 'status_not_ok': []}
    versions: dict[str, int] = {}
    Xs, ys, gs, outs, lives = [], [], [], [], []
    for game in games:
        fn = game_to_pickle_name(game)
        fp, mp = os.path.join(pickle_dir, fn), os.path.join(pickle_dir, 'meta', fn)
        if not os.path.exists(fp):
            skipped['no_pickle'].append(game); continue
        if not os.path.exists(mp):
            skipped['no_meta'].append(game); continue
        if status is not None:
            if game not in status.index or status.loc[game, 'status'] != 'ok':
                skipped['status_not_ok'].append(game); continue
            v = status.loc[game, 'version']
            versions[v] = versions.get(v, 0) + 1
        df, meta = pd.read_pickle(fp), pd.read_pickle(mp)
        missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
        if missing:
            raise ValueError(f"{fn}: missing columns {missing}")
        if len(df) != len(meta) or not df.index.equals(meta.index):
            raise ValueError(f"{fn}: features and meta do not line up ({len(df)} vs {len(meta)} rows or different index)")
        if len(df) == 0:
            continue
        if meta['game'].nunique() != 1 or meta['game'].iloc[0] != game:
            raise ValueError(f"{fn}: meta.game is {meta['game'].unique()[:3]}, expected {game}")
        ps = df[LABEL].to_numpy(float)
        if not np.isin(ps, (0.0, 1.0)).all():
            raise ValueError(f"{fn}: {LABEL} has values other than 0/1")
        Xs.append(df[FEATURES].to_numpy(float))
        ys.append((ps == 0).astype(np.int8))
        gs.append(np.full(len(df), game))
        outs.append(meta['outcome'].astype(str).to_numpy())
        lives.append(meta['live'].to_numpy(bool) if 'live' in meta else None)
    if not Xs:
        raise ValueError(f"no usable games in {pickle_dir} for the {len(games)} requested "
                         f"(skipped: { {k: len(v) for k, v in skipped.items()} })")
    ds = Dataset(np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), list(FEATURES), np.concatenate(outs),
                 live=np.concatenate(lives) if all(l is not None for l in lives) else None)
    #The label must agree with the detector's outcome column (completed <=> pass_successful == 1).
    ds.info = dict(requested=len(games), skipped=skipped, versions=versions,
                   label_outcome_mismatch=int(((ds.outcome != 'completed').astype(np.int8) != ds.y).sum()))
    return ds


def restrict_population(ds: Dataset, population: str) -> Dataset:
    """'all' (default); 'frontcourt' (backcourt == 0; the then-constant column is dropped); 'live' (meta.live: clock
    running and the passer in bounds, i.e. no dead-ball or inbound passes; needs pickles that carry it); or
    'live_frontcourt' (both)."""
    if population == 'all':
        return ds
    if population not in ('frontcourt', 'live', 'live_frontcourt'):
        raise ValueError(f"unknown population {population!r}")
    keep = np.ones(len(ds.y), bool)
    if population in ('live', 'live_frontcourt'):
        if ds.live is None:
            raise ValueError("these pickles have no meta.live column (made by code before the WS4 regeneration)")
        keep &= ds.live
    cols = list(range(len(ds.features)))
    if population in ('frontcourt', 'live_frontcourt'):
        j = ds.features.index('backcourt')
        keep &= ds.X[:, j] == 0
        cols = [i for i in cols if i != j]
    return Dataset(ds.X[keep][:, cols], ds.y[keep], ds.g[keep], [ds.features[i] for i in cols],
                   ds.outcome[keep], dict(ds.info, population=population), None if ds.live is None else ds.live[keep])


def restrict_features(ds: Dataset, names: list[str]) -> Dataset:
    """Keep only the named feature columns (all must exist)."""
    cols = [ds.features.index(n) for n in names]
    return Dataset(ds.X[:, cols], ds.y, ds.g, list(names), ds.outcome, dict(ds.info, feature_set=list(names)), ds.live)


def dataset_diagnostics(ds: Dataset) -> dict:
    """Descriptive, pooled (NOT cross-validated): NaN rates, near-duplicate columns, single-feature AUCs."""
    df = pd.DataFrame(ds.X, columns=ds.features)
    nan_rate = df.isna().mean()
    corr = df.corr()
    dup = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(ds.features) for b in ds.features[i + 1:]
           if abs(corr.loc[a, b]) > 0.95]
    single = {}
    for c in ds.features:
        ok = df[c].notna().to_numpy()
        if ds.y[ok].min() == ds.y[ok].max() or df[c][ok].nunique() < 2:
            continue
        a = roc_auc_score(ds.y[ok], df[c][ok])
        single[c] = float(max(a, 1 - a))   #orientation-free
    return dict(nan_rate=nan_rate, near_duplicates=dup, single_feature_auc=pd.Series(single).sort_values(ascending=False))


# ----------------------------------------------------------------------------------------------
# Splits (all by game) and the record of every split used
# ----------------------------------------------------------------------------------------------
class SplitRecord(NamedTuple):
    fold: int
    kind: str                  #'outer' | 'cal' | 'val'
    train: frozenset           #games the model/calibrator learns from
    held: frozenset            #games held out from that step


def outer_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by game; `seed` shuffles which games land in which fold."""
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in gkf.split(np.zeros(len(groups)), groups=groups)]


def game_date(game: str):
    from datetime import datetime
    return datetime.strptime('.'.join(game.split('.')[:3]), '%m.%d.%Y')


def date_block_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds are contiguous blocks of games in date order (test = one block of the season). Unlike
    shuffled GroupKFold this asks whether a model fit on the rest of the season holds up on a different
    stretch of it, which is closer to how the frozen test games (the last ~20% by date) are used."""
    games = np.unique(groups)
    games = games[np.argsort([game_date(x).toordinal() for x in games], kind='stable')]
    out = []
    for block in np.array_split(games, n_splits):
        held = np.isin(groups, block)
        out.append((np.flatnonzero(~held), np.flatnonzero(held)))
    return out


def split_by_game(idx: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split row positions `idx` into (rest, held) so that no game appears in both.
    About `frac` of the games (at least one, never all) go to `held`."""
    games = np.unique(groups[idx])
    if len(games) < 2:
        raise ValueError("need at least 2 games to split by game")
    perm = np.random.default_rng(seed).permutation(games)
    n_held = min(max(1, int(round(frac * len(games)))), len(games) - 1)
    mask = np.isin(groups[idx], perm[:n_held])
    return idx[~mask], idx[mask]


def inner_seed(seed: int, fold: int) -> int:
    return seed * 1000 + fold


def assert_game_disjoint(groups: np.ndarray, a: np.ndarray, b: np.ndarray, what: str = 'split') -> None:
    both = set(groups[a]) & set(groups[b])
    if both:
        raise AssertionError(f"{what}: {len(both)} game(s) on both sides, e.g. {sorted(both)[:3]}")


def verify_split_log(records: list[SplitRecord]) -> None:
    """Every inner split must lie inside its fold's outer-train games and share nothing with its
    outer-test games; each split's two sides must share no game; the MLP validation games must
    not overlap the calibration games."""
    by_fold: dict[int, list[SplitRecord]] = {}
    for r in records:
        by_fold.setdefault(r.fold, []).append(r)
    for k, recs in by_fold.items():
        outer = [r for r in recs if r.kind == 'outer']
        if len(outer) != 1:
            raise AssertionError(f"fold {k}: expected one outer record, found {len(outer)}")
        o = outer[0]
        if o.train & o.held:
            raise AssertionError(f"fold {k}: train/test share games {sorted(o.train & o.held)[:3]}")
        cal_held = set().union(*[r.held for r in recs if r.kind == 'cal'])
        for r in recs:
            if r.train & r.held:
                raise AssertionError(f"fold {k} {r.kind}: both sides share games {sorted(r.train & r.held)[:3]}")
            if r.kind == 'outer':
                continue
            used = r.train | r.held
            if used & o.held:
                raise AssertionError(f"fold {k} {r.kind}: uses outer-test games {sorted(used & o.held)[:3]}")
            if not used <= o.train:
                raise AssertionError(f"fold {k} {r.kind}: uses games outside the outer-train set")
            if r.kind == 'val' and (used & cal_held):
                raise AssertionError(f"fold {k}: MLP validation games overlap the calibration games")


# ----------------------------------------------------------------------------------------------
# Models. Each exposes fit(X, y, groups, logger=None) and score(X) -> P(fail)
# ----------------------------------------------------------------------------------------------
class SkModel:
    """An sklearn pipeline (median imputer fit on the training rows only is its first step)."""

    def __init__(self, pipeline, cols: list[int] | None = None):
        self.pipeline, self.cols, self.info = pipeline, cols, {}

    def _x(self, X):
        return X if self.cols is None else X[:, self.cols]

    def fit(self, X, y, groups=None, logger=None):
        self.pipeline.fit(self._x(X), y)
        return self

    def score(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(self._x(X))[:, 1]


class KerasMLP:
    """Dense(128, relu) + Dropout(0.2) + sigmoid, like the old `pass_model.py`, but with balanced
    class weights, a by-game validation split for early stopping (weighted validation loss, best
    weights restored), imputer+scaler fit on the fit rows only, and seeded numpy/python/TensorFlow.
    Batch size is 512 (the old script used 32) so 5 folds finish in minutes on this machine."""

    def __init__(self, seed: int, units: int = 128, dropout: float = 0.2, lr: float = 1e-3,
                 batch_size: int = 512, max_epochs: int = 50, patience: int = 5, val_frac: float = 0.15):
        self.seed, self.units, self.dropout, self.lr = seed, units, dropout, lr
        self.batch_size, self.max_epochs, self.patience, self.val_frac = batch_size, max_epochs, patience, val_frac
        self.info: dict = {}

    def fit(self, X, y, groups, logger=None):
        import keras                                  #lazy: TensorFlow import is 7-60 s
        keras.utils.set_random_seed(self.seed)        #python, numpy and TensorFlow
        try:
            keras.config.enable_op_determinism()
        except Exception:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        idx = np.arange(len(y))
        fit_idx, val_idx = split_by_game(idx, groups, self.val_frac, self.seed)
        if logger is not None:
            logger('val', frozenset(groups[fit_idx]), frozenset(groups[val_idx]))
        self.imp = SimpleImputer(strategy='median').fit(X[fit_idx])
        self.scaler = StandardScaler().fit(self.imp.transform(X[fit_idx]))
        Xf = self.scaler.transform(self.imp.transform(X[fit_idx])).astype('float32')
        Xv = self.scaler.transform(self.imp.transform(X[val_idx])).astype('float32')
        yf, yv = y[fit_idx].astype('float32'), y[val_idx].astype('float32')
        n, n1 = len(yf), float(yf.sum())
        cw = {0: n / (2 * (n - n1)), 1: n / (2 * n1)}   #sklearn's 'balanced' rule, from fit rows only
        sw_val = np.where(yv == 1, cw[1], cw[0]).astype('float32')
        self.model = keras.Sequential([
            keras.layers.Input(shape=(Xf.shape[1],)),
            keras.layers.Dense(self.units, activation='relu'),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(1, activation='sigmoid')])
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr), loss='binary_crossentropy')
        stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        hist = self.model.fit(Xf, yf, validation_data=(Xv, yv, sw_val), class_weight=cw, epochs=self.max_epochs,
                              batch_size=self.batch_size, callbacks=[stop], shuffle=True, verbose=0)
        val_loss = hist.history['val_loss']
        self.info = dict(epochs_run=len(val_loss), best_epoch=int(np.argmin(val_loss)) + 1,
                         best_val_loss=float(np.min(val_loss)), n_val_games=len(set(groups[val_idx])))
        return self

    def score(self, X) -> np.ndarray:
        Xs = self.scaler.transform(self.imp.transform(X)).astype('float32')
        #Direct calls in fixed-size batches: avoids the tf.data pipeline of predict() (retracing warnings).
        return np.concatenate([np.asarray(self.model(Xs[i:i + 8192], training=False)).ravel()
                               for i in range(0, len(Xs), 8192)]).astype(float)


MODEL_NAMES = ('simple', 'logistic', 'gboost', 'mlp')


def make_model(name: str, seed: int, features: list[str]):
    imp = lambda: SimpleImputer(strategy='median')
    if name == 'simple':       #shot_clock + dist only
        cols = [features.index(c) for c in SIMPLE_FEATURES]
        return SkModel(make_pipeline(imp(), StandardScaler(),
                                     LogisticRegression(class_weight='balanced', max_iter=1000)), cols)
    if name == 'logistic':
        return SkModel(make_pipeline(imp(), StandardScaler(),
                                     LogisticRegression(class_weight='balanced', max_iter=1000)))
    if name == 'gboost':       #same settings as eval_by_game.py
        return SkModel(make_pipeline(imp(), GradientBoostingClassifier(subsample=0.5, random_state=seed)))
    if name == 'mlp':
        return KerasMLP(seed)
    if name == 'prior':        #constant = training prevalence (used by tests)
        return SkModel(make_pipeline(imp(), DummyClassifier(strategy='prior')))
    raise ValueError(f"unknown model {name!r}")


# ----------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------
def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Monotone map from raw score to P(fail). Output is floored/capped at PROB_EPS so a calibrator
    that saw no positives in a score range cannot output an exact 0 (infinite log-loss)."""
    return IsotonicRegression(y_min=PROB_EPS, y_max=1 - PROB_EPS, out_of_bounds='clip').fit(scores, y)


@dataclass
class EvalResult:
    scored: np.ndarray                     #row positions that were scored (every outer-test row, once)
    fold: np.ndarray                       #fold id for each scored row
    raw: dict[str, np.ndarray]             #model -> raw P(fail) for the scored rows
    cal: dict[str, np.ndarray]             #model -> isotonic-calibrated P(fail)
    seconds: dict[str, float]
    notes: dict[str, list]
    split_log: list[SplitRecord]


def evaluate_splits(X: np.ndarray, y: np.ndarray, g: np.ndarray, splits, model_names, features: list[str],
                    seed: int = 0, cal_frac: float = 0.25, log: Callable[[str], None] | None = None) -> EvalResult:
    """Fit every model on each (train, test) split and score the test rows, raw and calibrated.
    Test labels are never read here: `y[test]` may be NaN. Returns predictions plus a log of every
    game-level split that was made (verified before returning)."""
    log = log or (lambda m: None)
    raw = {m: [] for m in model_names}; cal = {m: [] for m in model_names}
    secs = {m: 0.0 for m in model_names}; notes = {m: [] for m in model_names}
    scored, fold_ids, records = [], [], []
    for k, (tr, te) in enumerate(splits):
        assert_game_disjoint(g, tr, te, f"fold {k} outer")
        records.append(SplitRecord(k, 'outer', frozenset(g[tr]), frozenset(g[te])))
        fit_idx, cal_idx = split_by_game(tr, g, cal_frac, inner_seed(seed, k))
        assert_game_disjoint(g, fit_idx, cal_idx, f"fold {k} calibration")
        records.append(SplitRecord(k, 'cal', frozenset(g[fit_idx]), frozenset(g[cal_idx])))
        if y[cal_idx].min() == y[cal_idx].max():
            raise ValueError(f"fold {k}: calibration games contain a single class; need more games or failures")
        for m in model_names:
            t0 = time.perf_counter()
            model = make_model(m, seed + k, features)
            model.fit(X[fit_idx], y[fit_idx], g[fit_idx],
                      logger=lambda kind, a, b, k=k: records.append(SplitRecord(k, kind, a, b)))
            p_te = model.score(X[te])
            iso = fit_isotonic(model.score(X[cal_idx]), y[cal_idx])    #calibration games only
            raw[m].append(p_te); cal[m].append(iso.predict(p_te))
            secs[m] += time.perf_counter() - t0
            if model.info:
                notes[m].append(dict(fold=k, **model.info))
            log(f"fold {k + 1}/{len(splits)} {m}: {time.perf_counter() - t0:.1f}s")
        scored.append(te); fold_ids.append(np.full(len(te), k))
    verify_split_log(records)
    return EvalResult(np.concatenate(scored), np.concatenate(fold_ids),
                      {m: np.concatenate(v) for m, v in raw.items()}, {m: np.concatenate(v) for m, v in cal.items()},
                      secs, notes, records)


# ----------------------------------------------------------------------------------------------
# Metrics: weighted ROC-AUC / PR-AUC without re-sorting, so a bootstrap replicate costs O(n)
# ----------------------------------------------------------------------------------------------
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


class RankedScores:
    """Scores sorted once. `curves(w)` returns the weighted (ROC-AUC, average precision) in O(n);
    tied scores are treated as one threshold, exactly like sklearn's roc_auc_score and
    average_precision_score (the test suite checks the agreement, with ties)."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        s = np.asarray(s, float)
        self.order = np.argsort(-s, kind='stable')
        ss = s[self.order]
        self.y = np.asarray(y)[self.order].astype(float)
        self.ends = np.flatnonzero(np.append(ss[1:] != ss[:-1], True))      #last row of each tie group

    def curves(self, w: np.ndarray | None = None) -> tuple[float, float]:
        ws = np.ones(len(self.y)) if w is None else np.asarray(w, float)[self.order]
        tp = np.cumsum(ws * self.y)[self.ends]
        fp = np.cumsum(ws * (1.0 - self.y))[self.ends]
        P, N = tp[-1], fp[-1]
        if P <= 0 or N <= 0:
            return float('nan'), float('nan')
        auc = float(_trapezoid(np.concatenate(([0.0], tp / P)), np.concatenate(([0.0], fp / N))))
        tot = tp + fp
        prec = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
        rec = tp / P
        ap = float(np.sum(np.diff(np.concatenate(([0.0], rec))) * prec))
        return auc, ap


def weighted_brier(y, p, w=None) -> float:
    return float(np.average((np.asarray(p, float) - y) ** 2, weights=w))


def weighted_logloss(y, p, w=None) -> float:
    p = np.clip(np.asarray(p, float), PROB_EPS, 1 - PROB_EPS)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability (fewer if many predictions tie)."""
    bins = pd.qcut(pd.Series(np.asarray(p, float)), n_bins, duplicates='drop')
    df = pd.DataFrame({'y': np.asarray(y, float), 'p': np.asarray(p, float), 'bin': bins})
    t = df.groupby('bin', observed=True).agg(n=('y', 'size'), failures=('y', 'sum'),
                                             mean_pred=('p', 'mean'), observed=('y', 'mean'))
    return t.reset_index(drop=True).assign(failures=lambda d: d['failures'].astype(int))


def by_outcome_table(y: np.ndarray, outcome: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """For each kind of failure (detector `outcome`), ROC-AUC / PR-AUC of that kind against ALL completed
    passes: shows what the labelled failures are made of and which kind the model can rank."""
    rows = []
    for kind in sorted(set(outcome[y == 1])):
        keep = (y == 0) | (outcome == kind)
        yy = (outcome[keep] == kind).astype(int)
        row = dict(failure_type=kind, n=int(yy.sum()))
        for name, sc in scores.items():
            row[f'{name} ROC'] = roc_auc_score(yy, sc[keep]); row[f'{name} PR'] = average_precision_score(yy, sc[keep])
        row['PR no-skill'] = yy.mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Game-clustered bootstrap
# ----------------------------------------------------------------------------------------------
class GameBootstrap:
    """`draws[b]` lists the games (as indices into `games`) drawn, with replacement, in replicate b.
    A game's rows are always taken whole, any number of times (including zero)."""

    def __init__(self, groups: np.ndarray, n_boot: int, seed: int = 0):
        self.games, self.codes = np.unique(groups, return_inverse=True)
        self.n_boot = n_boot
        self.draws = np.random.default_rng(seed).integers(0, len(self.games), size=(n_boot, len(self.games)))
        order = np.argsort(self.codes, kind='stable')
        self._rows_of_game = np.split(order, np.cumsum(np.bincount(self.codes))[:-1])

    def game_counts(self, b: int) -> np.ndarray:
        return np.bincount(self.draws[b], minlength=len(self.games))

    def row_weights(self, b: int) -> np.ndarray:
        """Per-row multiplicity in replicate b (constant within a game)."""
        return self.game_counts(b)[self.codes].astype(float)

    def resample_rows(self, b: int) -> np.ndarray:
        """Explicit row indices of replicate b (whole games, concatenated; for inspection and tests)."""
        return np.concatenate([self._rows_of_game[j] for j in self.draws[b]])


def bootstrap_metrics(y: np.ndarray, preds: dict[str, np.ndarray], boot: GameBootstrap,
                      prob_keys=()) -> dict[str, dict[str, np.ndarray]]:
    """Per-replicate ROC-AUC and PR-AUC for every score vector in `preds`; for keys in `prob_keys`
    (vectors that are probabilities) also Brier, log-loss and Brier skill score (reference =
    that replicate's own prevalence, p(1-p)). Same game resamples for every key."""
    ranked = {k: RankedScores(y, v) for k, v in preds.items()}
    out = {k: {m: np.full(boot.n_boot, np.nan) for m in (('roc', 'pr', 'brier', 'logloss', 'bss') if k in prob_keys
                                                       else ('roc', 'pr'))} for k in preds}
    yf = np.asarray(y, float)
    for b in range(boot.n_boot):
        w = boot.row_weights(b)
        prev = np.average(yf, weights=w)
        for k, rs in ranked.items():
            out[k]['roc'][b], out[k]['pr'][b] = rs.curves(w)
            if k in prob_keys:
                br = weighted_brier(yf, preds[k], w)
                out[k]['brier'][b] = br
                out[k]['logloss'][b] = weighted_logloss(yf, preds[k], w)
                out[k]['bss'][b] = 1 - br / (prev * (1 - prev))
    return out


def ci(a: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------------------------
# Summary tables
# ----------------------------------------------------------------------------------------------
def summarize(ds_y: np.ndarray, ds_g: np.ndarray, res: EvalResult, model_names, n_boot: int = 1000,
              seed: int = 0, ref: str = 'logistic') -> dict:
    """Pooled out-of-fold metrics with game-clustered CIs, paired differences against `ref`,
    calibration results and reliability tables. All rows of `res.scored` are test rows."""
    y, g, fold = ds_y[res.scored], ds_g[res.scored], res.fold
    prev = float(y.mean())
    boot = GameBootstrap(g, n_boot, seed)
    preds = {'no-skill|raw': np.zeros(len(y)), 'no-skill|cal': np.full(len(y), prev)}
    for m in model_names:
        preds[f'{m}|raw'], preds[f'{m}|cal'] = res.raw[m], res.cal[m]
    prob_keys = [k for k in preds if k.endswith('|cal')] + [f'{m}|raw' for m in model_names]
    bm = bootstrap_metrics(y, preds, boot, prob_keys)
    point = {k: dict(zip(('roc', 'pr'), RankedScores(y, v).curves())) for k, v in preds.items()}
    for k in prob_keys:
        br = weighted_brier(y.astype(float), preds[k])
        point[k].update(brier=br, logloss=weighted_logloss(y.astype(float), preds[k]), bss=1 - br / (prev * (1 - prev)))

    disc = []
    for name in ['no-skill'] + list(model_names):
        k = f'{name}|raw'
        row = dict(model=name, roc=point[k]['roc'], pr=point[k]['pr'])
        row['roc_lo'], row['roc_hi'] = ci(bm[k]['roc'])
        row['pr_lo'], row['pr_hi'] = ci(bm[k]['pr'])
        if name != 'no-skill':
            usable = [f for f in np.unique(fold) if len(np.unique(y[fold == f])) == 2] if len(np.unique(fold)) > 1 else []
            r_f = [roc_auc_score(y[fold == f], res.raw[name][fold == f]) for f in usable] or [np.nan]
            p_f = [average_precision_score(y[fold == f], res.raw[name][fold == f]) for f in usable] or [np.nan]
            row.update(fold_roc_mean=np.mean(r_f), fold_roc_sd=np.std(r_f, ddof=1) if len(r_f) > 1 else np.nan,
                       fold_pr_mean=np.mean(p_f), fold_pr_sd=np.std(p_f, ddof=1) if len(p_f) > 1 else np.nan,
                       fit_seconds=res.seconds[name])
        disc.append(row)

    paired = []
    if ref in model_names:
        for m in model_names:
            if m == ref:
                continue
            row = dict(model=f'{m} - {ref}')
            for metric in ('roc', 'pr'):
                d = bm[f'{m}|raw'][metric] - bm[f'{ref}|raw'][metric]
                row[metric] = point[f'{m}|raw'][metric] - point[f'{ref}|raw'][metric]
                row[f'{metric}_lo'], row[f'{metric}_hi'] = ci(d)
            paired.append(row)

    calib = []
    for name in ['no-skill'] + list(model_names):
        row = dict(model=name)
        if name == 'no-skill':
            row.update(brier_ref=prev * (1 - prev))
            row.update(cal_brier=point['no-skill|cal']['brier'], cal_logloss=point['no-skill|cal']['logloss'], cal_bss=point['no-skill|cal']['bss'])
        else:
            r, c = point[f'{name}|raw'], point[f'{name}|cal']
            row.update(raw_brier=r['brier'], cal_brier=c['brier'], raw_logloss=r['logloss'], cal_logloss=c['logloss'],
                       raw_bss=r['bss'], cal_bss=c['bss'], cal_roc=c['roc'], cal_pr=c['pr'], raw_pr=r['pr'],
                       mean_pred_raw=float(res.raw[name].mean()), mean_pred_cal=float(res.cal[name].mean()))
            row['cal_brier_lo'], row['cal_brier_hi'] = ci(bm[f'{name}|cal']['brier'])
            row['cal_bss_lo'], row['cal_bss_hi'] = ci(bm[f'{name}|cal']['bss'])
        calib.append(row)

    reliability = {f'{m}|{kind}': reliability_table(y, (res.raw if kind == 'raw' else res.cal)[m])
                   for m in model_names for kind in ('raw', 'cal')}
    return dict(prevalence=prev, n_games=int(len(np.unique(g))), n_rows=int(len(y)), n_failures=int(y.sum()),
                discrimination=pd.DataFrame(disc), paired=pd.DataFrame(paired), calibration=pd.DataFrame(calib),
                reliability=reliability, boot=boot)
