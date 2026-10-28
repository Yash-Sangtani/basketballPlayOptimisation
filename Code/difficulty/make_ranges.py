"""Save the training ranges (0.5%-99.5% quantiles) that the API uses to flag out-of-distribution passes and to clamp shot features.

    cd Code && python difficulty/make_ranges.py --data DIR      (tuning games only, same population as the model)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_eval as T                                   # noqa: E402
import shot_value as SV                                  # noqa: E402
from pass_features import FEATURE_NAMES, SHOT_FEATURE_NAMES   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
a = ap.parse_args()
tune, _ = T.split_games()
C = T.load(a.data, tune, T.REGIMES['narrow'])
C = C[C.clean == 1]
S = SV.load_shots(a.data, tune)
q = lambda df, cols: {c: [float(df[c].quantile(0.005)), float(df[c].quantile(0.995))] for c in cols}
out = dict(pass_features=q(C, FEATURE_NAMES), shot_features=q(S, [c for c in SHOT_FEATURE_NAMES]),
           heave_fg=float(S.loc[S.shot_dist > 30, 'made'].mean()), n_passes=int(len(C)), n_shots=int(len(S)))
json.dump(out, open(os.path.join(HERE, 'model', 'ranges.json'), 'w'), indent=1)
print('saved', {k: (v if not isinstance(v, dict) else len(v)) for k, v in out.items()})
