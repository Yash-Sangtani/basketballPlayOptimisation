"""RL-facing interface to the pass-difficulty model.

    from difficulty.api import PassDifficultyModel
    m = PassDifficultyModel()                       # loads difficulty/model/difficulty_v1.joblib
    r = m.predict(P, T, defenders, basket, shot_clock, p_vel, t_vel)
    r['p_complete']      # calibrated probability the pass reaches the intended teammate        (the number to use as a transition probability)
    r['difficulty']      # = 1 - p_complete = P(pass fails: intercepted / out of bounds / fumbled)
    r['relative_difficulty']   # log-odds of failure minus log-odds of what the passer's situation alone predicts
                               # (> 0: this pass is harder than the situation explains; < 0: easier)
    df = m.score_teammates(holder_xy, mates_xy, defenders_xy, basket, shot_clock, holder_vel, mates_vel)   # one row per teammate

Units: feet and ft/s, court 94 x 50. The model is mirror-symmetric, `basket` is the basket the offense attacks.
The model was trained on live-ball passes only (not inbounds, not dead balls) and describes the *completion*
of a pass thrown now from the given state; it says nothing about what the receiver does next.
Passes shorter than 1 ft (hand-offs) have no defined geometry: they are returned as p_complete = 1.0, extrapolated=True.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pass_features import pass_features   # noqa: E402

DEFAULT_MODEL = os.path.join(HERE, 'model', 'difficulty_v1.joblib')
DEFAULT_RANGES = os.path.join(HERE, 'model', 'ranges.json')


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class PassDifficultyModel:

    def __init__(self, path=DEFAULT_MODEL):
        self.b = joblib.load(path)
        self.features = self.b['features']
        self.situation_features = self.b['situation_features']
        self.ranges = json.load(open(DEFAULT_RANGES)) if os.path.exists(DEFAULT_RANGES) else None

    def ood_columns(self, F: pd.DataFrame) -> pd.Series:
        """Number of features outside the training 0.5%-99.5% range, per row (NaN if no ranges file)."""
        if self.ranges is None:
            return pd.Series(np.nan, index=F.index)
        n = pd.Series(0, index=F.index)
        for c, (lo, hi) in self.ranges['pass_features'].items():
            n += ((F[c] < lo) | (F[c] > hi)).astype(int)
        return n

    def _p_fail(self, F):
        """F: DataFrame with all feature columns -> (calibrated P(fail), situation-only calibrated P(fail))."""
        raw = self.b['model'].predict_proba(F[self.features].to_numpy(float))[:, 1]
        sraw = self.b['situation_model'].predict_proba(F[self.situation_features].to_numpy(float))[:, 1]
        return self.b['iso'].predict(raw), self.b['situation_iso'].predict(sraw)

    def predict_features(self, F: pd.DataFrame) -> pd.DataFrame:
        pf, ps = self._p_fail(F)
        n_ood = self.ood_columns(F)
        return pd.DataFrame({'p_complete': 1 - pf, 'difficulty': pf, 'situation_difficulty': ps,
                             'relative_difficulty': _logit(pf) - _logit(ps), 'n_ood_features': n_ood,
                             'in_distribution': n_ood == 0}, index=F.index)

    def predict(self, P, T, defenders, basket, shot_clock=np.nan, p_vel=(0.0, 0.0), t_vel=(0.0, 0.0)):
        f = pass_features(P, T, defenders, basket, shot_clock, p_vel, t_vel)
        if f is None:
            return dict(p_complete=1.0, difficulty=0.0, situation_difficulty=np.nan, relative_difficulty=np.nan, n_ood_features=np.nan,
                        in_distribution=False, extrapolated=True)
        r = self.predict_features(pd.DataFrame([f])).iloc[0].to_dict()
        r['extrapolated'] = False
        return r

    def predict_many(self, states):
        """states: iterable of dicts with keys P, T, defenders, basket and optional shot_clock, p_vel, t_vel.
        One model pass for all of them (use this from a vectorised RL environment)."""
        feats, idx = [], []
        for i, st in enumerate(states):
            f = pass_features(st['P'], st['T'], st['defenders'], st['basket'], st.get('shot_clock', np.nan),
                              st.get('p_vel', (0.0, 0.0)), st.get('t_vel', (0.0, 0.0)))
            if f is not None:
                feats.append(f)
                idx.append(i)
        n = len(states) if hasattr(states, '__len__') else i + 1
        out = pd.DataFrame({'p_complete': 1.0, 'difficulty': 0.0, 'situation_difficulty': np.nan,
                            'relative_difficulty': np.nan, 'n_ood_features': np.nan, 'in_distribution': False,
                            'extrapolated': True}, index=range(n))
        if feats:
            r = self.predict_features(pd.DataFrame(feats))
            r['extrapolated'] = False
            r.index = idx
            out.loc[idx, r.columns] = r.to_numpy()
            out['extrapolated'] = out['extrapolated'].astype(bool)
            out['in_distribution'] = out['in_distribution'].astype(bool)
        return out

    def score_teammates(self, holder_xy, mates_xy, defenders_xy, basket, shot_clock=np.nan,
                        holder_vel=(0.0, 0.0), mates_vel=None):
        mates_xy = np.asarray(mates_xy, float).reshape(-1, 2)
        mates_vel = np.zeros_like(mates_xy) if mates_vel is None else np.asarray(mates_vel, float).reshape(-1, 2)
        return self.predict_many([dict(P=holder_xy, T=mates_xy[k], defenders=defenders_xy, basket=basket, shot_clock=shot_clock,
                                       p_vel=holder_vel, t_vel=mates_vel[k]) for k in range(len(mates_xy))])


DEFAULT_SHOT_MODEL = os.path.join(HERE, 'model', 'shot_v1.joblib')


class PassValueModel(PassDifficultyModel):
    """PassDifficultyModel plus the stage-2 expected-points terms (see shot_value.py for definitions and limits).

    evaluate(...) returns the difficulty fields and, in expected points:
      threat_receiver / threat_passer   xPts of a shot taken right now by the receiver / the passer
      ev_pass          = p_complete * threat_receiver - p_fail * c_fail
      pass_minus_shoot = ev_pass - threat_passer
    These are LOWER BOUNDS on what a pass is worth (they ignore everything after the catch other than an immediate
    shot) and were not found to predict assists better than distance to the basket (reports/difficulty-results.md);
    use p_complete / difficulty as the dependable output and c_fail as a turnover penalty.
    """

    def __init__(self, path=DEFAULT_MODEL, shot_path=DEFAULT_SHOT_MODEL):
        super().__init__(path)
        from shot_value import xpts, MAKE_FEATURES
        self._xpts, self._shot_cols = xpts, MAKE_FEATURES
        sb = joblib.load(shot_path)
        self.shot = sb['models']
        self.c_fail = sb['c_fail']

    def shot_xpts(self, S, defenders, basket, vel=(0.0, 0.0), shot_clock=np.nan):
        """xPts of a shot from S now, with two guards against exploitation: every feature is clamped to the range seen in
        training, and shots beyond the longest distance seen in training (about 33 ft, 99.5th pct) are valued as heaves
        (observed 25% FG, three points) instead of extrapolating the make model."""
        from pass_features import shot_features
        f = shot_features(S, defenders, basket, vel, shot_clock)
        if f is None:
            return np.nan
        if self.ranges is not None:
            lo_hi = self.ranges['shot_features']
            heave = f['shot_dist'] > lo_hi['shot_dist'][1]
            f = {k: (min(max(v, lo_hi[k][0]), lo_hi[k][1]) if k in lo_hi and k != 'shot_dist' and np.isfinite(v) else v) for k, v in f.items()}
            if heave:
                return 3.0 * self.ranges['heave_fg']
        return float(self._xpts(self.shot, pd.DataFrame([f]))['xpts'].iloc[0])

    def evaluate(self, P, T, defenders, basket, shot_clock=np.nan, p_vel=(0.0, 0.0), t_vel=(0.0, 0.0)):
        r = self.predict(P, T, defenders, basket, shot_clock, p_vel, t_vel)
        tr = self.shot_xpts(T, defenders, basket, t_vel, shot_clock)
        tp = self.shot_xpts(P, defenders, basket, p_vel, shot_clock)
        r.update(threat_receiver=tr, threat_passer=tp, c_fail=self.c_fail,
                 ev_pass=r['p_complete'] * tr - r['difficulty'] * self.c_fail)
        r['pass_minus_shoot'] = r['ev_pass'] - tp
        return r
