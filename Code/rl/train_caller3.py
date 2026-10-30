"""Stage 4c: play-calling policy. The play-caller chooses a play (Stage-3 library) at the start of each possession; the play-conditioned executors (v3)
move the players towards it; the decision-level policy (shoot / pass / who) acts on top. Lead passes (the ball no longer snaps to the receiver).

    cd Code && python rl/train_caller3.py --data ../data/rl --model rl/model/executor_v3.keras --out ../reports/rl-stage4c/run \
        --start-weights ../reports/rl-stage4b/run2/ckpt_pi.weights.h5 --iters 500 --n 800

Reward = shot xPts or -0.10 (turnover / violation) + 0.25 when the possession ran the CHOSEN play cleanly (its first-6-s formation is assigned to the
chosen play and within that play's median distance) and ended in a shot with xPts >= 1.0. A play choice is one categorical decision per possession
(prior = real play shares, bounded residual, REINFORCE with the same baseline logic as the shot/pass decisions). Resumable; checkpoint + GIF every 10 iterations.
EPV is not used (reports/rl-stage4c-epv.md: it explains no more variance than shot clock and ball position).
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
from caller import Caller                # noqa: E402
from train_caller import SIM as SIM_V2, stats      # noqa: E402
SIM = dict(SIM_V2, shoot_scale=1.05, pass_scale=0.9)      # recalibrated on development games for the play-conditioned executors (reports/rl-stage4c.md)
from play_reward import PlayLibrary, LAMBDA, SUCCESS_XPTS   # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from gif_episode import render           # noqa: E402


def log(*a):
    print(*a, flush=True)


def load_old_policy(caller, path):
    """Copy a previous policy file (13 input features) into the 14-feature network, zero-padding the new input rows."""
    tmp = keras.Sequential([keras.Input((13,)), keras.layers.Dense(32, activation='gelu'), keras.layers.Dense(32, activation='gelu'),
                            keras.layers.Dense(1, kernel_initializer='zeros')])
    tmp.load_weights(path)
    for new, old in zip(caller.pi.layers, tmp.layers):
        w = old.get_weights()
        if w[0].shape[0] == 13:
            w[0] = np.concatenate([w[0], np.zeros((1, w[0].shape[1]), np.float32)], axis=0)
        new.set_weights(w)


def play_bonus(sim, lib, plays, pt):
    """Bonus when, up to the moment the possession ended (at least 3 s after midcourt), the formation followed the CHOSEN play (mean cell-count distance to
    the play's prototype over the 1-s slices observed <= the median of that play's real members over the same slices) and the possession ended in a shot
    with xPts >= SUCCESS_XPTS. Also returns the cluster the full 6-s formation falls in when the possession lasted 6 s (diagnostic)."""
    idx, X = sim.play_descriptors(window=3.0)
    bonus = np.zeros(sim.N)
    follow = np.zeros(sim.N, bool)
    reached = np.zeros(sim.N, bool)
    reached[idx] = True
    cl = np.full(sim.N, -1)
    if len(idx):
        tf_end = sim.t_front0[idx] + sim.elapsed[idx]
        ns = np.minimum(7, np.floor(tf_end).astype(int) + 1)
        kk = plays[idx]
        dist = np.stack([0.5 * np.abs(X[:, s * 33:s * 33 + 30] - pt.grid[kk, s]).sum(1) for s in range(7)], axis=1)
        mean_d = np.array([dist[n, :ns[n]].mean() for n in range(len(idx))])
        ok_form = mean_d <= pt.thr[kk, ns - 1]
        follow[idx] = ok_form
        ok = ok_form & (sim.end[idx] == 1) & (np.nan_to_num(sim.shot_info[idx, 1]) >= SUCCESS_XPTS)
        bonus[idx[ok]] = LAMBDA
        full = tf_end >= 6.0
        if full.any():
            k6, d6, clean6 = lib.clean(X[full])
            cl[idx[full][clean6]] = k6[clean6]
    return bonus, cl, reached, follow


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
    ap.add_argument('--seed', type=int, default=3)
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
    diff, lib, pt = PassValueModel(), PlayLibrary(), PlayTargets()
    caller = Caller(diff, beta=a.beta, lr=a.lr, seed=a.seed)
    caller.set_play_prior(pt.freq)
    ck = {k: os.path.join(a.out, f) for k, f in (('pi', 'ckpt_pi.weights.h5'), ('v', 'ckpt_v.weights.h5'), ('pb', 'ckpt_play_b.npy'), ('h', 'train_history.json'))}
    hist, start_it = [], 0
    if os.path.exists(ck['h']) and os.path.exists(ck['pi']):
        hist = json.load(open(ck['h']))
        start_it = len(hist)
        caller.pi.load_weights(ck['pi'])
        caller.v.load_weights(ck['v'])
        caller.play_b.assign(np.load(ck['pb']))
        log(f'resumed from iteration {start_it}')
    else:
        load_old_policy(caller, a.start_weights)
        log('started from the run-2 policy weights (shoot/pass part); play head at the prior')
    starts_fit = find_starts(FIT)
    log(f'{len(starts_fit)} training starts from {len(FIT)} games; lambda {LAMBDA}, success xPts >= {SUCCESS_XPTS}')
    for it in range(start_it, a.iters):
        t0 = time.time()
        rng = np.random.default_rng(a.seed * 100003 + it)
        sel = rng.choice(len(starts_fit), min(a.n, len(starts_fit)), replace=False)
        plays = caller.choose_plays(len(sel))
        sim = Sim(model, diff, seed=a.seed * 1000 + it, play_targets=pt, **SIM).reset(FIT, table, [starts_fit[i] for i in sel], play=plays)
        sim.caller = caller
        sim.run()
        bonus, cl, reached, hit = play_bonus(sim, lib, plays, pt)
        base_reward = sim.reward.copy()
        sim.reward = base_reward + bonus
        u = caller.update(sim.reward)
        s = stats(sim)
        gap = np.array(sim.arrival_gap) if sim.arrival_gap else np.zeros(1)
        s.update(u)
        s.update(iter=it, sec=time.time() - t0, reward_base=float(base_reward.mean()), bonus_rate=float((bonus > 0).mean()), clean_play_rate=float((cl >= 0).mean()),
                 reached_6s=float(reached.mean()), follow_rate=float(hit.sum() / max(reached.sum(), 1)), arrival_gap=float(gap.mean()),
                 plays_entropy=float(-(np.bincount(plays, minlength=lib.K) / len(plays) * np.log(np.clip(np.bincount(plays, minlength=lib.K) / len(plays), 1e-9, 1))).sum()))
        hist.append(s)
        log(f"it {it:3d} reward {s['reward']:.3f} (xPts part {s['reward_base']:.3f}) bonus {s['bonus_rate']:.3f} follow {s['follow_rate']:.3f} 6s {s['reached_6s']:.3f} "
            f"p_shot {s['p_shot']:.3f} to {s['p_turnover']:.3f} passes {s['passes']:.2f} dur {s['dur']:.2f} kl {u.get('kl', 0):.3f} playKL {u.get('play_kl', 0):.3f} ({s['sec']:.0f}s)")
        if (it + 1) % a.gif_every == 0:
            caller.pi.save_weights(ck['pi'])
            caller.v.save_weights(ck['v'])
            np.save(ck['pb'], np.asarray(caller.play_b))
            json.dump(hist, open(ck['h'], 'w'), indent=1)
            win = np.flatnonzero(bonus > 0)
            pool = win if len(win) else np.flatnonzero(sim.elapsed >= 4.0)
            if len(pool) == 0:
                pool = np.arange(sim.N)
            i = int(pool[0]) if len(win) else int(pool[np.argmax(sim.elapsed[pool])])
            code = {1: f'SHOT xPts {sim.shot_info[i, 1]:.2f}', 2: 'TURNOVER', 3: 'SHOT CLOCK VIOLATION', 4: 'TIMEOUT'}[int(sim.end[i])]
            ptxt = f'chosen play {plays[i]}' + (f', formation {cl[i]}' if cl[i] >= 0 else ', no clean formation') + (f', +{bonus[i]:.2f} bonus' if bonus[i] > 0 else '')
            render(sim, i, os.path.join(a.out, 'gifs', f'iter_{it + 1:03d}.gif'), f'iter {it + 1} | {ptxt}', f'{code} | {ptxt}')
            log(f'   checkpoint + gif iter_{it + 1:03d}.gif')


if __name__ == '__main__':
    main()
