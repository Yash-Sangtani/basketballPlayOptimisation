"""Stage 5: how much of the arm-C gain depends on the simulator's time cost (non-pass ball-loss hazard)? Holdout, frozen policy, audited value.
    cd Code && python rl/sens_stage5.py --policy ../reports/rl-stage5/seeds/C_s12 --out ../reports/rl-stage5"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
import executor_v2 as E                  # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from sim_validate import find_starts     # noqa: E402
from train_caller3 import SIM as SIM_V3  # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from shot_audit import AuditedShotValue  # noqa: E402
from eval_stage5 import run_policy, paired, load_caller   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=2000)
a = ap.parse_args()
from api import PassValueModel
fit_g, _ = fit_val_games()
table = build_profiles(load_games('../data/rl', fit_g))
G = load_games('../data/rl', split_games('holdout'))
m3 = keras.saving.load_model('rl/model/executor_v3g.keras', compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
diff, pt, aud = PassValueModel(), PlayTargets(), AuditedShotValue()
rng = np.random.default_rng(0)
starts = find_starts(G)
starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
game = np.array([s[0] for s in starts])
plays = rng.choice(pt.K, size=len(starts), p=pt.freq / pt.freq.sum())
res = {}
for name, over in (('default (0.03/s)', {}), ('no time cost (0/s)', dict(loss_rate=0.0)), ('double time cost (0.06/s)', dict(loss_rate=0.06)), ('shot shooting scale x0.5', dict(shoot_scale=0.4))):
    cfg = dict(SIM_V3, **over)
    base = run_policy('base ' + name, m3, cfg, None, plays, G, table, starts, diff, pt, aud)
    c = load_caller(a.policy, diff, pt, aud)
    pol = run_policy('policy ' + name, m3, cfg, c, c.choose_plays(len(starts)), G, table, starts, diff, pt, aud)
    d, ci = paired(pol['r_aud'], base['r_aud'], game)
    res[name] = dict(base=float(base['r_aud'].mean()), policy=float(pol['r_aud'].mean()), gain=d, ci95=ci, policy_duration=pol['beh']['duration'])
    print(f"{name:28s} baseline {res[name]['base']:.3f} policy {res[name]['policy']:.3f} gain {d:+.3f} {np.round(ci, 3)}", flush=True)
json.dump(res, open(os.path.join(a.out, 'sensitivity_C_s12.json'), 'w'), indent=1)
