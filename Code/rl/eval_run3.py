"""Evaluate the Stage-4c policy (play choice + shoot/pass policy, play-conditioned executors, lead passes) against baselines.
    cd Code && python rl/eval_run3.py --data ../data/rl --v2 rl/model/executor_v2.keras --v3 rl/model/executor_v3g.keras --run ../reports/rl-stage4c/run [--holdout]
Rows: v2 executors with their own decisions (the Stage-4b baseline, no plays); v3 executors with plays drawn from the real play shares and their own decisions;
the same with the OOD-masked zero-residual caller; the trained policy (its own play choices). Reward is reported without and with the play bonus."""
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
from train_caller import SIM as SIM_V2, stats, ci   # noqa: E402
from train_caller3 import SIM as SIM_V3, play_bonus  # noqa: E402
from play_reward import PlayLibrary      # noqa: E402
from play_cond import PlayTargets        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--v2', required=True)
ap.add_argument('--v3', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--holdout', action='store_true')
a = ap.parse_args()
from api import PassValueModel
fit_g, val_g = fit_val_games()
table = build_profiles(load_games(a.data, fit_g))
G = load_games(a.data, split_games('holdout') if a.holdout else val_g)
m2 = keras.saving.load_model(a.v2, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
m3 = keras.saving.load_model(a.v3, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
diff, lib, pt = PassValueModel(), PlayLibrary(), PlayTargets()
rng = np.random.default_rng(0)
starts = find_starts(G)
starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
game = np.array([s[0] for s in starts])
prior_plays = rng.choice(pt.K, size=len(starts), p=pt.freq / pt.freq.sum())
trained = Caller(diff, learn=False)
trained.set_play_prior(pt.freq)
trained.pi.load_weights(os.path.join(a.run, 'ckpt_pi.weights.h5'))
trained.play_b.assign(np.load(os.path.join(a.run, 'ckpt_play_b.npy')))
masked = Caller(diff, learn=False)
masked.set_play_prior(pt.freq)
res, sims = {}, {}
for name, model, cfg, caller, plays in (('v2_own_decisions', m2, SIM_V2, None, None), ('v3_plays_own_decisions', m3, SIM_V3, None, prior_plays),
                                        ('v3_plays_masked_prior', m3, SIM_V3, masked, prior_plays), ('trained_policy', m3, SIM_V3, trained, 'own')):
    pl = trained.choose_plays(len(starts)) if isinstance(plays, str) else plays
    sim = Sim(model, diff, seed=12345, play_targets=pt, **cfg).reset(G, table, starts, play=pl)
    sim.caller = caller
    sim.run()
    if pl is not None:
        bonus, cl, reached, follow = play_bonus(sim, lib, pl, pt)
    else:
        bonus, cl, reached, follow = np.zeros(sim.N), np.full(sim.N, -1), np.zeros(sim.N, bool), np.zeros(sim.N, bool)
    s = stats(sim)
    gap = np.array(sim.arrival_gap) if sim.arrival_gap else np.zeros(1)
    s.update(reward_with_bonus=float((sim.reward + bonus).mean()), bonus_rate=float((bonus > 0).mean()), follow_rate=float(follow.sum() / max(reached.sum(), 1)),
             reached_3s=float(reached.mean()), arrival_gap=float(gap.mean()), plays_used=int(len(np.unique(pl))) if pl is not None else 0)
    sims[name], res[name] = sim, s
    print(name, json.dumps({k: round(v, 3) for k, v in s.items()}), flush=True)
for x, y in (('trained_policy', 'v2_own_decisions'), ('trained_policy', 'v3_plays_own_decisions'), ('trained_policy', 'v3_plays_masked_prior')):
    d, c = ci(sims[x], sims[y], game)
    res[f'diff_{x}_minus_{y}'] = dict(mean=d, ci95=c)
    print(f'{x} - {y}: {d:+.3f} {np.round(c, 3)}', flush=True)
res['play_choice_distribution'] = [float(v) for v in np.exp(trained.play_logits())[:] / np.exp(trained.play_logits()).sum()]
tag = 'holdout' if a.holdout else 'val'
json.dump(dict(eval=res, split=tag, n=len(starts)), open(os.path.join(a.run, f'eval_{tag}.json'), 'w'), indent=1)
