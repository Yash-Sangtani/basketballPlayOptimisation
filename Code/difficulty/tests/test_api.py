import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from pass_features import FEATURE_NAMES, SITUATION_FEATURES, pass_features   # noqa: E402
from api import PassDifficultyModel                                           # noqa: E402


def toy_model(tmp_path):
    """A model trained on synthetic geometry where failure is likelier with a defender on the line."""
    rng = np.random.default_rng(0)
    rows, y = [], []
    for _ in range(600):
        P, T = rng.uniform([5, 5], [45, 45]), rng.uniform([5, 5], [45, 45])
        D = rng.uniform([0, 0], [47, 50], size=(5, 2))
        f = pass_features(P, T, D, np.array([5.35, 25.0]), rng.uniform(0, 24))
        if f is None:
            continue
        rows.append(f)
        y.append(int(rng.random() < 1 / (1 + np.exp(2 + 3 * f['intercept_margin']))))
    X = pd.DataFrame(rows)
    y = np.array(y)
    m = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X[FEATURE_NAMES].to_numpy(), y)
    ms = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X[SITUATION_FEATURES].to_numpy(), y)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4).fit(m.predict_proba(X[FEATURE_NAMES].to_numpy())[:, 1], y)
    isos = IsotonicRegression(out_of_bounds='clip', y_min=1e-4, y_max=1 - 1e-4).fit(ms.predict_proba(X[SITUATION_FEATURES].to_numpy())[:, 1], y)
    path = str(tmp_path / 'm.joblib')
    joblib.dump(dict(model=m, iso=iso, features=FEATURE_NAMES, situation_model=ms, situation_iso=isos,
                     situation_features=SITUATION_FEATURES), path)
    return path


def test_predict_ranges_and_consistency(tmp_path):
    m = PassDifficultyModel(toy_model(tmp_path))
    r = m.predict([30, 25], [18, 35], [[28, 24], [10, 20], [22, 30]], [5.35, 25], 12.0)
    assert 0 <= r['p_complete'] <= 1 and abs(r['p_complete'] + r['difficulty'] - 1) < 1e-9
    assert r['extrapolated'] is False and np.isfinite(r['relative_difficulty'])


def test_teammates_table_and_trivial_pass(tmp_path):
    m = PassDifficultyModel(toy_model(tmp_path))
    df = m.score_teammates([30, 25], [[18, 35], [30.2, 25.1], [40, 10]], [[28, 24], [10, 20]], [5.35, 25], 9.0)
    assert len(df) == 3
    assert df.loc[1, 'p_complete'] == 1.0 and bool(df.loc[1, 'extrapolated'])
