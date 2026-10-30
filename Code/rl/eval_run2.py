"""Evaluate run-2 (play-bonus) policy against baseline, masked prior and the run-1 policy. Rewards reported without and with the play bonus.
    cd Code && python rl/eval_run2.py --data ../data/rl --model rl/model/executor_v2.keras --out ../reports/rl-stage4b/run2 [--holdout]
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
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from sim import Sim                      # noqa: E402
from sim_validate import find_starts     # noqa: E402
from caller import Caller                # noqa: E402
from train_caller import SIM, stats, ci  # noqa: E402
from train_caller2 import load_old_policy, play_bonus   # noqa: E402
from play_reward import PlayLibrary      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--holdout', action='store_true')
    a = ap.parse_args()
    from api import PassValueModel
    fit_g, val_g = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games('holdout') if a.holdout else val_g)
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff, lib = PassValueModel(), PlayLibrary()
    rng = np.random.default_rng(0)
    starts = find_starts(G)
    starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
    game = np.array([s[0] for s in starts])
    run1 = Caller(diff, learn=False)
    load_old_policy(run1, os.path.join(a.out, '..', 'policy.weights.h5'))
    run2 = Caller(diff, learn=False)
    run2.pi.load_weights(os.path.join(a.out, 'ckpt_pi.weights.h5'))
    zero = Caller(diff, learn=False)
    res, sims = {}, {}
    for name, c in (('baseline', None), ('prior_masked', zero), ('run1_policy', run1), ('run2_policy', run2)):
        sim = Sim(model, diff, seed=12345, **SIM).reset(G, table, starts)
        sim.caller = c
        sim.run()
        bonus, cl, reached = play_bonus(sim, lib)
        s = stats(sim)
        s.update(reward_with_bonus=float((sim.reward + bonus).mean()), play_bonus_rate=float((bonus > 0).mean()), clean_play_rate=float((cl >= 0).mean()), reached_6s=float(reached.mean()),
                 plays_used=int(len(np.unique(cl[cl >= 0]))))
        sims[name] = sim
        res[name] = s
        print(name, json.dumps({k: round(v, 3) for k, v in s.items()}), flush=True)
    for x, y in (('run2_policy', 'baseline'), ('run2_policy', 'run1_policy'), ('run1_policy', 'baseline')):
        d, c = ci(sims[x], sims[y], game)
        res[f'diff_{x}_minus_{y}'] = dict(mean=d, ci95=c)
        print(f'{x} - {y}: {d:+.3f} {np.round(c, 3)}', flush=True)
    json.dump(dict(eval=res, split='holdout' if a.holdout else 'val', n=len(starts)), open(os.path.join(a.out, f"eval_run2_{'holdout' if a.holdout else 'val'}.json"), 'w'), indent=1)


if __name__ == '__main__':
    main()
