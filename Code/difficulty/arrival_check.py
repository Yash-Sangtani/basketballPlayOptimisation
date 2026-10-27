"""Diagnostic: is the receiver's shot threat more informative at the moment the ball ARRIVES than at release?
Re-runs tracking on a few tuning games, computes the actual receiver's shot features at the reception frame, and compares
the AUC for "this completed pass becomes an assist" with the release-time threat. Cross-fitted shot model (games in the
check are excluded from its training). Tuning games only.

    cd Code && python difficulty/arrival_check.py --data DIR --n 30
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TQDM_DISABLE', '1')
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from sklearn.metrics import roc_auc_score   # noqa: E402
import extract as X                     # noqa: E402
import shot_value as SV                 # noqa: E402
import train_eval as T                  # noqa: E402
from pass_features import shot_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    tune, _ = T.split_games()
    pick = tune[::max(1, len(tune) // a.n)][:a.n]
    S = SV.load_shots(a.data, tune)
    mods = SV.fit_models(S[~S['game'].isin(pick)].reset_index(drop=True))
    rows = []
    for g in pick:
        date, away, home = X.split(g)
        pt = X.DiffPT(date, home, away)
        pt.build()
        fr = pt._fr
        P = pt.pass_rows
        P = P[(P.label == 1) & (P.live == 1) & P.actual_target.notna()]
        sh = S[(S.game == g) & (S.assisted == 1) & (S.made == 1)]
        link = {}
        for r in sh.itertuples():
            link.setdefault((r.quarter, r.assister, r.shooter), []).append(r.quarter_time)
        for r in P.itertuples():
            ut = fr['ut'][r.frame] + r.flight_s
            rf = int(np.argmin(np.abs(fr['ut'] - ut)))
            pid, xy = fr['pid'][rf], fr['xy'][rf]
            it = np.flatnonzero(pid == int(r.actual_target))
            opp = (fr['team'][rf] != r.team) & (pid != 0)
            basket = pt._attack_basket(r.team, r.quarter)
            if len(it) == 0 or not opp.any() or basket is None:
                continue
            tv, _ = pt._vel(rf, int(r.actual_target))
            sf = shot_features(xy[it[0]], xy[opp], np.array([basket[0], 25.0]), tv, r.shot_clock)
            if sf is None:
                continue
            qts = link.get((r.quarter, r.passer, r.actual_target), [])
            rows.append(dict(game=g, assist=int(any(0 <= r.quarter_time - q <= 6 for q in qts)), rel_xpts=np.nan, **{'a_' + k: v for k, v in sf.items()},
                             **{'r_' + k: getattr(r, 'rs_' + k) for k in ('shot_dist', 'shot_angle', 'def1', 'def2', 'def_cos', 'shooter_speed')}, r_shot_clock=r.shot_clock))
    D = pd.DataFrame(rows)
    fa = D[[c for c in D if c.startswith('a_')]].rename(columns=lambda c: c[2:])
    fr_ = D[[c for c in D if c.startswith('r_')]].rename(columns=lambda c: c[2:])
    D['arrival_xpts'] = SV.xpts(mods, fa[SV.MAKE_FEATURES])['xpts'].to_numpy()
    D['release_xpts'] = SV.xpts(mods, fr_[SV.MAKE_FEATURES])['xpts'].to_numpy()
    y = D['assist'].to_numpy()
    print(f'{len(pick)} tuning games, {len(D)} live completed passes to the actual receiver, {int(y.sum())} assist-linked ({y.mean():.3f})')
    for name, v in (('threat at ARRIVAL (xPts)', D['arrival_xpts']), ('threat at RELEASE (xPts)', D['release_xpts']),
                    ('arrival distance to basket (negated)', -D['a_shot_dist']), ('release distance to basket (negated)', -D['r_shot_dist']),
                    ('arrival nearest defender distance', D['a_def1'])):
        ok = v.notna().to_numpy()
        print(f'  AUC for becoming an assist: {name:40s} {roc_auc_score(y[ok], v[ok]):.3f}')


if __name__ == '__main__':
    main()
