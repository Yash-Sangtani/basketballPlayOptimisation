"""Stage 5 evaluation of frozen policies, under the model's shot value and under the audited (empirical) shot value.

    holdout:  cd Code && python rl/eval_stage5.py --split holdout --seeds-dir ../reports/rl-stage5/seeds --out ../reports/rl-stage5
    test:     cd Code && python rl/eval_stage5.py --split test --policy A=<dir> --policy C=<dir> --out ../reports/rl-stage5 --confirm-test   (refuses to run twice)

Rewards exclude the play bonus (points only). The audited reward replaces each simulated shot's value by the empirical points of real shots in the same
distance x clock-elapsed x nearest-defender bin (fitted on the 401 train games). Paired game-clustered 95% intervals against the baseline (v3g executors,
plays from the real shares, their own decisions). Realism of the baseline against real possessions (the five Stage-4a statistics) is reported on the same starts.
"""
import argparse
import glob
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
from sim_validate import find_starts, real_outcome, summarise   # noqa: E402
from caller import Caller                # noqa: E402
from train_caller import SIM as SIM_V2   # noqa: E402
from train_caller3 import SIM as SIM_V3  # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from shot_audit import AuditedShotValue, bin_index   # noqa: E402


def paired(a, b, game, seed=0, n=500):
    d = a - b
    ug = np.unique(game)
    s = np.array([d[game == u].sum() for u in ug])
    c = np.array([(game == u).sum() for u in ug])
    rng = np.random.default_rng(seed)
    bs = [s[w].sum() / c[w].sum() for w in (rng.integers(0, len(ug), len(ug)) for _ in range(n))]
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def run_policy(name, model, cfg, caller, plays, G, table, starts, diff, pt, aud):
    log = []
    sim = Sim(model, diff, seed=12345, play_targets=pt, shot_log=log, **cfg).reset(G, table, starts, play=plays)
    sim.caller = caller
    sim.run()
    r_model = sim.reward.copy()
    r_aud = sim.reward.copy()
    for s in log:
        r_aud[s['i']] = float(aud(s['dist'], s['elapsed'], s['def1']))
    d = np.array([s['dist'] for s in log])
    e = np.array([s['elapsed'] for s in log])
    f = np.array([s['def1'] for s in log])
    i, j, k = bin_index(d, e, f)
    T = np.stack(sim.traj)
    nst = np.round(sim.elapsed / 0.2).astype(int)
    sp = []
    for ep in range(sim.N):
        n = max(nst[ep], 1)
        sp.append(np.hypot(*(T[1:n + 1, ep, :5] - T[:n, ep, :5]).transpose(2, 0, 1)).mean() / 0.2)
    shot = sim.end == 1
    beh = dict(shot_share=float(shot.mean()), turnover=float((sim.end == 2).mean()), passes=float(sim.n_pass.mean()), duration=float(sim.elapsed.mean()),
               shot_dist=float(d.mean()) if len(d) else np.nan, shot_elapsed=float(e.mean()) if len(e) else np.nan, shot_def1=float(f.mean()) if len(f) else np.nan,
               share_28ft_plus=float((d >= 28).mean()) if len(d) else np.nan, share_thin_bins=float((aud.cnt[i, j, k] < 100).mean()) if len(d) else np.nan,
               model_xpts_per_shot=float(np.mean([s['xpts'] for s in log])) if log else np.nan, audited_per_shot=float(aud(d, e, f).mean()) if len(d) else np.nan,
               pass_p_complete=float(np.mean(caller.pass_pcs)) if (caller is not None and caller.pass_pcs) else np.nan, offence_speed=float(np.mean(sp)))
    print(name, json.dumps({k_: round(v, 3) for k_, v in beh.items()}), flush=True)
    return dict(r_model=r_model, r_aud=r_aud, beh=beh, sim=sim)


def real_stats(G, starts):
    real = [real_outcome(G[g], k) for g, k in starts]
    R = summarise([r['shot'] for r in real], [r['t'] for r in real], [r['n_pass'] for r in real], [r['pts'] for r in real])
    g = np.array([s[0] for s in starts])
    ug = np.unique(g)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(300):
        ii = np.concatenate([np.flatnonzero(g == u) for u in rng.choice(ug, len(ug))])
        boots.append(summarise(*[[real[i][k] for i in ii] for k in ('shot', 't', 'n_pass', 'pts')]))
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in R if k != 'n'}
    return R, ci


def sim_stats(sim):
    return summarise((sim.end == 1).astype(int), sim.shot_info[:, 3], sim.n_pass, np.nan_to_num(sim.shot_info[:, 1]))


def load_caller(path, diff, pt, support=None):
    c = Caller(diff, learn=False)
    c.set_play_prior(pt.freq)
    c.pi.load_weights(os.path.join(path, 'ckpt_pi.weights.h5'))
    c.play_b.assign(np.load(os.path.join(path, 'ckpt_play_b.npy')))
    if support is not None:
        c.support, c.min_support = support, 100
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['holdout', 'test'])
    ap.add_argument('--data', default='../data/rl')
    ap.add_argument('--test-data', default='../data/rl_test')
    ap.add_argument('--v2', default='rl/model/executor_v2.keras')
    ap.add_argument('--v3', default='rl/model/executor_v3g.keras')
    ap.add_argument('--seeds-dir', default=None)
    ap.add_argument('--policy', action='append', default=[], help='NAME=checkpoint dir (arm C policies get the support guard)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--confirm-test', action='store_true')
    a = ap.parse_args()
    from api import PassValueModel
    out_path = os.path.join(a.out, f'stage5_{a.split}.json')
    if a.split == 'test':
        if not a.confirm_test:
            raise SystemExit('the test split is scored once: pass --confirm-test')
        if os.path.exists(out_path):
            raise SystemExit(f'{out_path} exists: the test split has been scored; it is not scored twice')
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    if a.split == 'holdout':
        G = load_games(a.data, split_games('holdout'))
    else:
        names = [l.strip().replace('.7z', '') for l in open(os.path.join(HERE, '..', 'splits', 'test_games.txt')) if l.strip()]
        G = load_games(a.test_data, names)
    m2 = keras.saving.load_model(a.v2, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    m3 = keras.saving.load_model(a.v3, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff, pt, aud = PassValueModel(), PlayTargets(), AuditedShotValue()
    rng = np.random.default_rng(0)
    starts = find_starts(G)
    starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
    game = np.array([s[0] for s in starts])
    prior_plays = rng.choice(pt.K, size=len(starts), p=pt.freq / pt.freq.sum())
    R, ci = real_stats(G, starts)
    res = dict(split=a.split, n_starts=len(starts), n_games=len(G), real=R, real_ci95=ci, policies={})
    runs = {}
    runs['v2_own_decisions'] = run_policy('v2_own_decisions', m2, SIM_V2, None, None, G, table, starts, diff, pt, aud)
    runs['v3g_own_decisions'] = run_policy('v3g_own_decisions', m3, SIM_V3, None, prior_plays, G, table, starts, diff, pt, aud)
    res['baseline_realism'] = {'v2': sim_stats(runs['v2_own_decisions']['sim']), 'v3g': sim_stats(runs['v3g_own_decisions']['sim'])}
    pol = {}
    if a.seeds_dir:
        for d in sorted(glob.glob(os.path.join(a.seeds_dir, '*_s*'))):
            pol[os.path.basename(d)] = d
    for spec in a.policy:
        n, p = spec.split('=')
        pol[n] = p
    for name, path in pol.items():
        arm = name[0]
        c = load_caller(path, diff, pt, aud if arm == 'C' else None)
        runs[name] = run_policy(name, m3, SIM_V3, c, c.choose_plays(len(starts)), G, table, starts, diff, pt, aud)
    base = runs['v3g_own_decisions']
    for name, r in runs.items():
        e = dict(behaviour=r['beh'], reward_model=float(r['r_model'].mean()), reward_audited=float(r['r_aud'].mean()))
        if name != 'v3g_own_decisions':
            e['gain_model'] = dict(zip(('mean', 'ci95'), paired(r['r_model'], base['r_model'], game)))
            e['gain_audited'] = dict(zip(('mean', 'ci95'), paired(r['r_aud'], base['r_aud'], game)))
        res['policies'][name] = e
        if 'gain_audited' in e:
            print(f"{name:22s} reward model {e['reward_model']:.3f} audited {e['reward_audited']:.3f} | gain vs v3g own: model {e['gain_model']['mean']:+.3f} {np.round(e['gain_model']['ci95'], 3)} audited {e['gain_audited']['mean']:+.3f} {np.round(e['gain_audited']['ci95'], 3)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
