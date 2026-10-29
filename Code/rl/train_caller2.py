"""Stage 4b, run 2: continue training the decision-level policy with a play-run bonus, a checkpoint and a GIF every 10 iterations.

    cd Code && python rl/train_caller2.py --data ../data/rl --model rl/model/executor_v2.keras --out ../reports/rl-stage4b/run2 \
        --start-weights ../reports/rl-stage4b/policy.weights.h5 --iters 500 --n 800

Reward = shot xPts or -0.10 (turnover / violation) [as in run 1] + LAMBDA (0.25) when the possession ran a library play cleanly and ended in a
shot with xPts >= 1.0 (play_reward.py). Resumable: rerun the same command and it continues from the last checkpoint (every 10 iterations).
"""
import argparse
import json
import os
import sys
import time

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
from caller import Caller, D_FEAT        # noqa: E402
from train_caller import SIM, stats      # noqa: E402
from play_reward import PlayLibrary, LAMBDA, SUCCESS_XPTS   # noqa: E402
from gif_episode import render           # noqa: E402


def log(*a):
    print(*a, flush=True)


def load_old_policy(caller, path):
    """Old policy file has 12 input features; the new one has 13 (time left in the play window): copy and zero-pad the first kernel."""
    tmp = keras.Sequential([keras.Input((12,)), keras.layers.Dense(32, activation='gelu'), keras.layers.Dense(32, activation='gelu'),
                            keras.layers.Dense(1, kernel_initializer='zeros')])
    tmp.load_weights(path)
    for new, old in zip(caller.pi.layers, tmp.layers):
        w = old.get_weights()
        if w[0].shape[0] == 12:
            w[0] = np.concatenate([w[0], np.zeros((1, w[0].shape[1]), np.float32)], axis=0)
        new.set_weights(w)


def play_bonus(sim, lib):
    idx, X = sim.play_descriptors()
    bonus = np.zeros(sim.N)
    cl = np.full(sim.N, -1)
    reached = np.zeros(sim.N, bool)
    reached[idx] = True
    if len(idx):
        k, d, clean = lib.clean(X)
        ok = clean & (sim.end[idx] == 1) & (np.nan_to_num(sim.shot_info[idx, 1]) >= SUCCESS_XPTS)
        bonus[idx[ok]] = LAMBDA
        cl[idx[clean]] = k[clean]
    return bonus, cl, reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start-weights', required=True)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--beta', type=float, default=0.02)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gif-every', type=int, default=10)
    ap.add_argument('--n-fit-games', type=int, default=0)
    a = ap.parse_args()
    from api import PassValueModel
    os.makedirs(os.path.join(a.out, 'gifs'), exist_ok=True)
    fit_g, _ = fit_val_games()
    if a.n_fit_games:
        fit_g = fit_g[:a.n_fit_games]
    FIT = load_games(a.data, fit_g)
    table = build_profiles(FIT)
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff = PassValueModel()
    lib = PlayLibrary()
    caller = Caller(diff, beta=a.beta, lr=a.lr, seed=a.seed)
    ck_pi, ck_v, ck_h = (os.path.join(a.out, f) for f in ('ckpt_pi.weights.h5', 'ckpt_v.weights.h5', 'train_history.json'))
    hist, start_it = [], 0
    if os.path.exists(ck_h) and os.path.exists(ck_pi):
        hist = json.load(open(ck_h))
        start_it = len(hist)
        caller.pi.load_weights(ck_pi)
        caller.v.load_weights(ck_v)
        log(f'resumed from iteration {start_it}')
    else:
        load_old_policy(caller, a.start_weights)
        log('started from the run-1 policy weights')
    starts_fit = find_starts(FIT)
    log(f'{len(starts_fit)} training starts from {len(FIT)} games; lambda {LAMBDA}, success xPts >= {SUCCESS_XPTS}')
    for it in range(start_it, a.iters):
        t0 = time.time()
        rng = np.random.default_rng(a.seed * 100003 + it)
        sel = rng.choice(len(starts_fit), min(a.n, len(starts_fit)), replace=False)
        sim = Sim(model, diff, seed=a.seed * 1000 + it, **SIM).reset(FIT, table, [starts_fit[i] for i in sel])
        sim.caller = caller
        sim.run()
        bonus, cl, reached = play_bonus(sim, lib)
        base_reward = sim.reward.copy()
        sim.reward = base_reward + bonus
        u = caller.update(sim.reward)
        s = stats(sim)
        s.update(u)
        s.update(iter=it, sec=time.time() - t0, reward_base=float(base_reward.mean()), bonus_rate=float((bonus > 0).mean()), clean_play_rate=float((cl >= 0).mean()),
                 reached_6s=float(reached.mean()))
        hist.append(s)
        log(f"it {it:3d} reward {s['reward']:.3f} (xPts part {s['reward_base']:.3f}) play-bonus {s['bonus_rate']:.3f} clean {s['clean_play_rate']:.3f} 6s {s['reached_6s']:.3f} "
            f"p_shot {s['p_shot']:.3f} to {s['p_turnover']:.3f} passes {s['passes']:.2f} dur {s['dur']:.2f} kl {u.get('kl', 0):.3f} ({s['sec']:.0f}s)")
        if (it + 1) % a.gif_every == 0:
            caller.pi.save_weights(ck_pi)
            caller.v.save_weights(ck_v)
            json.dump(hist, open(ck_h, 'w'), indent=1)
            win = np.flatnonzero(bonus > 0)
            pool = win if len(win) else np.flatnonzero(sim.elapsed >= 4.0)
            if len(pool) == 0:
                pool = np.arange(sim.N)
            i = int(pool[np.argmax(sim.elapsed[pool])]) if len(win) == 0 else int(pool[0])
            code = {1: f'SHOT xPts {sim.shot_info[i, 1]:.2f}', 2: 'TURNOVER', 3: 'SHOT CLOCK VIOLATION', 4: 'TIMEOUT'}[int(sim.end[i])]
            ptxt = f'play {cl[i]} run cleanly, +{bonus[i]:.2f} bonus' if bonus[i] > 0 else (f'play {cl[i]} (no bonus)' if cl[i] >= 0 else 'no play bonus')
            render(sim, i, os.path.join(a.out, 'gifs', f'iter_{it + 1:03d}.gif'), f'iter {it + 1} | policy sample | {ptxt}', f'{code} | {ptxt}')
            log(f'   checkpoint + gif iter_{it + 1:03d}.gif')


if __name__ == '__main__':
    main()
