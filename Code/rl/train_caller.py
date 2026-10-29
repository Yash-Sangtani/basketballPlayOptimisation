"""Stage 4b: train the decision-level play-caller in the simulator and evaluate it against the baselines.

    cd Code && python rl/train_caller.py --data ../data/rl --model rl/model/executor_v2.keras --out ../reports/rl-stage4b [--iters 40 --n 800]

Training starts: random half-court possession starts from the 320 fitting games. Development evaluation: fixed starts from the 81 validation
games (--eval-n). The holdout (100 games) is only read with --final (scored once with the frozen policy).
Simulator constants: shoot 0.85, pass 1.0, loss hazard 0.03/s (Stage 4a calibration).
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
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from sim import Sim                      # noqa: E402
from sim_validate import find_starts     # noqa: E402
from caller import Caller                # noqa: E402

SIM = dict(shoot_scale=0.85, pass_scale=1.0, loss_rate=0.03, mask_ood=False, c_fail=0.10)


def log(*a):
    print(*a, flush=True)


def run(model, diff, G, table, starts, caller, seed):
    sim = Sim(model, diff, seed=seed, **SIM).reset(G, table, starts)
    sim.caller = caller
    sim.run()
    return sim


def stats(sim):
    shot = sim.end == 1
    return dict(reward=float(sim.reward.mean()), p_shot=float(shot.mean()), p_turnover=float((sim.end == 2).mean()), p_violation=float((sim.end == 3).mean()),
                passes=float(sim.n_pass.mean()), dur=float(sim.elapsed.mean()), shot_dist=float(np.nanmean(sim.shot_info[shot, 0])) if shot.any() else float('nan'),
                xpts_shot=float(np.nanmean(sim.shot_info[shot, 1])) if shot.any() else float('nan'),
                t_shot=float(np.nanmean(sim.shot_info[shot, 3])) if shot.any() else float('nan'),
                pass_len=float(np.mean([x for l in sim.pass_len for x in l])) if any(sim.pass_len) else float('nan'))


def ci(sim_a, sim_b, game, n=500, seed=0):
    """paired game-clustered bootstrap of mean reward difference a - b (same starts)."""
    d = sim_a.reward - sim_b.reward
    ug = np.unique(game)
    rng = np.random.default_rng(seed)
    gm = np.array([d[game == u].sum() for u in ug])
    gc = np.array([(game == u).sum() for u in ug])
    out = []
    for _ in range(n):
        w = rng.integers(0, len(ug), len(ug))
        out.append(gm[w].sum() / gc[w].sum())
    return float(d.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--eval-n', type=int, default=1500)
    ap.add_argument('--beta', type=float, default=0.02)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--policy', default=None, help='saved policy weights (for --final)')
    ap.add_argument('--n-fit-games', type=int, default=0)
    a = ap.parse_args()
    from api import PassValueModel
    os.makedirs(a.out, exist_ok=True)
    fit_g, val_g = fit_val_games()
    if a.n_fit_games:
        fit_g, val_g = fit_g[:a.n_fit_games], val_g[:max(4, a.n_fit_games // 4)]
    FIT = load_games(a.data, fit_g)
    table = build_profiles(FIT)
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff = PassValueModel()
    rng = np.random.default_rng(a.seed)
    caller = Caller(diff, beta=a.beta, lr=a.lr, seed=a.seed)
    if not a.final:
        starts_fit = find_starts(FIT)
        log(f'{len(starts_fit)} training starts from {len(FIT)} games')
        hist = []
        for it in range(a.iters):
            t0 = time.time()
            sel = rng.choice(len(starts_fit), min(a.n, len(starts_fit)), replace=False)
            sim = run(model, diff, FIT, table, [starts_fit[i] for i in sel], caller, a.seed * 1000 + it)
            u = caller.update(sim.reward)
            s = stats(sim)
            s.update(u)
            s['iter'], s['sec'] = it, time.time() - t0
            hist.append(s)
            log(f"it {it:3d} reward {s['reward']:.3f} p_shot {s['p_shot']:.3f} to {s['p_turnover']:.3f} passes {s['passes']:.2f} dur {s['dur']:.2f} kl {u.get('kl', 0):.4f} ({s['sec']:.0f}s)")
        caller.pi.save_weights(os.path.join(a.out, 'policy.weights.h5'))
        json.dump(hist, open(os.path.join(a.out, 'train_history.json'), 'w'), indent=1)
        del FIT
        G = load_games(a.data, val_g)
        split = 'val'
    else:
        del FIT
        G = load_games(a.data, split_games('holdout'))
        caller.pi.load_weights(a.policy)
        split = 'holdout'
    starts = find_starts(G)
    sel = np.sort(rng.choice(len(starts), min(a.eval_n, len(starts)), replace=False))
    starts = [starts[i] for i in sel]
    game = np.array([s[0] for s in starts])
    log(f'evaluating on {len(starts)} {split} starts')
    trained = caller
    trained.learn = False
    zero = Caller(diff, seed=a.seed + 7, learn=False)            # zero residual = the executor's own decisions with OOD passes masked
    res, sims = {}, {}
    for name, c in (('baseline', None), ('prior_masked', zero), ('trained', trained)):
        sims[name] = run(model, diff, G, table, starts, c, 12345)       # same seed for every policy
        res[name] = stats(sims[name])
        log(name, json.dumps({k: round(v, 3) for k, v in res[name].items()}))
    res['diff_trained_minus_prior'] = dict(zip(('mean', 'ci95'), ci(sims['trained'], sims['prior_masked'], game)))
    res['diff_trained_minus_baseline'] = dict(zip(('mean', 'ci95'), ci(sims['trained'], sims['baseline'], game)))
    res['diff_prior_minus_baseline'] = dict(zip(('mean', 'ci95'), ci(sims['prior_masked'], sims['baseline'], game)))
    for k in ('diff_trained_minus_prior', 'diff_trained_minus_baseline', 'diff_prior_minus_baseline'):
        log(k, json.dumps(res[k]))
    json.dump(dict(args=vars(a), eval=res, split=split, n=len(starts), sim=SIM), open(os.path.join(a.out, f'eval_{split}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
