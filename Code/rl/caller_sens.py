"""Stage 4b robustness: re-evaluate the frozen policy under altered simulator assumptions (development games, same starts and seeds).

    cd Code && python rl/caller_sens.py --data ../data/rl --model rl/model/executor_v2.keras --policy ../reports/rl-stage4b/policy.weights.h5 --out ../reports/rl-stage4b

Scenarios: default; loss_rate 0 (no time cost of holding the ball); loss_rate 0.06 (double); shot value x0.9 (xPts of every shot 10% lower,
a crude allowance for the shot model's optimism on shots the policy takes that real players rarely take).
Reported: reward of baseline (executor's own decisions) and trained policy, paired difference with game-clustered CI.
"""
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
from train_executor import fit_val_games  # noqa: E402
from sim import Sim                      # noqa: E402
from sim_validate import find_starts     # noqa: E402
from caller import Caller                # noqa: E402
from train_caller import SIM, stats, ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1500)
    a = ap.parse_args()
    from api import PassValueModel
    fit_g, val_g = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, val_g)
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff = PassValueModel()
    rng = np.random.default_rng(0)
    starts = find_starts(G)
    starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
    game = np.array([s[0] for s in starts])
    pol = Caller(diff, learn=False)
    pol.pi.load_weights(a.policy)
    out = {}
    for name, over, scale in (('default', {}, 1.0), ('no_time_cost', dict(loss_rate=0.0), 1.0), ('double_time_cost', dict(loss_rate=0.06), 1.0), ('shot_value_x0.9', {}, 0.9)):
        cfg = dict(SIM, **over)
        sims = {}
        for key, c in (('baseline', None), ('trained', pol)):
            sim = Sim(model, diff, seed=12345, **cfg).reset(G, table, starts)
            sim.caller = c
            sim.run()
            if scale != 1.0:
                sim.reward = np.where(sim.end == 1, sim.reward * scale, sim.reward)
            sims[key] = sim
        d, c95 = ci(sims['trained'], sims['baseline'], game)
        out[name] = dict(baseline=stats(sims['baseline']), trained=stats(sims['trained']), diff=d, ci95=c95)
        print(name, 'baseline', round(out[name]['baseline']['reward'], 3), 'trained', round(out[name]['trained']['reward'], 3), 'diff', round(d, 3), [round(x, 3) for x in c95], flush=True)
    json.dump(out, open(os.path.join(a.out, 'sensitivity_val.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
