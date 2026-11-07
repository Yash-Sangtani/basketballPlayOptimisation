"""Does a chosen play make the executors produce that play's formation? (test that can fail)

    cd Code && python rl/rollout_follow.py --data ../data/rl --v2 rl/model/executor_v2.keras --v3 rl/model/executor_v3.keras --out ../reports/rl-stage4c

Validation-game possessions (>= 6 s, start state t_front >= 1.2 s) are rolled forward by the executors alone (nobody shoots or passes; ball stays with the
holder; shot clock held) until 6 s after midcourt, then the formation descriptor is assigned to a play. Conditions:
  v2_free          unconditioned executors (v2), target play = the possession's real play and a random other play
  v3_real_play     executors told the possession's real play
  v3_other_play    executors told a random different play (the real test: does the formation move to the chosen play?)
  v3_none          executors with the play features zeroed (v3 weights, no play) as a control for the changed network
Reported: share of rollouts assigned to the target play (hit), share that are clean instances of it, mean PCA distance to the target play's centre,
mean offence speed, share of positions off court.
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
from train_executor3 import assign_plays   # noqa: E402
from sim import Sim                      # noqa: E402
from sim_validate import find_starts     # noqa: E402
from play_reward import PlayLibrary      # noqa: E402
from play_cond import PlayTargets        # noqa: E402


def run_cond(model, diff, G, table, starts, play, lib, pt, seed, target=None):
    sim = Sim(model, diff, seed=seed, shoot_scale=0.0, pass_scale=0.0, loss_rate=0.0, play_targets=pt).reset(G, table, starts, play=play)
    sim.shot_clock[:] = 30.0
    sim.run(max_blocks=16)
    idx, X = sim.play_descriptors()
    k, d, clean = lib.clean(X)
    Z = ((X - lib.mu) / lib.sd - lib.pmean) @ lib.comp.T
    tgt = (play if target is None else target)[idx]
    dist_t = np.sqrt(((Z - lib.centers[tgt]) ** 2).sum(1))
    T = np.stack(sim.traj)
    sp = np.hypot(*(T[1:, :, :5] - T[:-1, :, :5]).transpose(3, 0, 1, 2)) / 0.2
    off = ((T[..., 0] < 0) | (T[..., 0] > 94) | (T[..., 1] < 0) | (T[..., 1] > 50)).mean()
    return dict(n=int(len(idx)), n_total=int(sim.N), hit=float((k == tgt).mean()), clean_hit=float(((k == tgt) & clean).mean()), dist_to_target=float(dist_t.mean()),
                speed_off=float(sp.mean()), off_court=float(off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--v2', required=True)
    ap.add_argument('--v3', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1200)
    a = ap.parse_args()
    from api import PassValueModel
    fit_g, val_g = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, val_g)
    lib, pt = PlayLibrary(), PlayTargets()
    lab = assign_plays(G, lib)
    starts = [s for s in find_starts(G) if (s[0], int(G[s[0]]['poss_id'][s[1]])) in lab]
    rng = np.random.default_rng(0)
    starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
    real = np.array([lab[(g, int(G[g]['poss_id'][k]))] for g, k in starts])
    other = np.array([(r + rng.integers(1, lib.K)) % lib.K for r in real])
    diff = PassValueModel()
    m2 = keras.saving.load_model(a.v2, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    m3 = keras.saving.load_model(a.v3, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    res = {}
    for name, model, play in (('v2_free_target_real', m2, real), ('v2_free_target_other', m2, other), ('v3_real_play', m3, real), ('v3_other_play', m3, other),
                              ('v3_no_play_target_real', m3, np.full(len(real), -1))):
        tgt = real if 'real' in name else other
        res[name] = run_cond(model, diff, G, table, starts, play, lib, pt, 5, target=tgt)
        print(name, json.dumps({k: round(v, 3) for k, v in res[name].items()}), flush=True)
    res['chance_hit'] = float(np.mean(lib_share(lib, pt)))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'follow_test.json'), 'w'), indent=1)


def lib_share(lib, pt):
    return pt.freq ** 2


if __name__ == '__main__':
    main()
