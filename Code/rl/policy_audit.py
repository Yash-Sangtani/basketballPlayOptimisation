"""Stage 5, part 2: what do the policy's shots look like against real shots, and what are they worth in real-data terms?

    cd Code && python rl/policy_audit.py --data ../data/rl --run ../reports/rl-stage4c/run --out ../reports/rl-stage5 [--policy-ckpt DIR]

Holdout starts (2,000). For (a) the executors' own decisions (v3g, plays from the real shares) and (b) the trained policy: every shot's features are logged
(distance, seconds elapsed on the shot clock, nearest defender). Reported: the distribution over the audit bins next to real shots, the model xPts, the audited
value (empirical points of real shots in the same bin), the number of real shots behind each policy shot's bin, and the same-bin audited value for the real shots.
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
from train_caller3 import SIM            # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from shot_audit import AuditedShotValue, real_shots, bin_index, D_EDGES, C_EDGES, F_EDGES   # noqa: E402


def summarize(log, aud):
    d = np.array([s['dist'] for s in log])
    e = np.array([s['elapsed'] for s in log])
    f = np.array([s['def1'] for s in log])
    x = np.array([s['xpts'] for s in log])
    i, j, k = bin_index(d, e, f)
    return dict(n=int(len(d)), dist_mean=float(d.mean()), elapsed_mean=float(e.mean()), def1_mean=float(f.mean()), model_xpts=float(x.mean()), audited=float(aud(d, e, f).mean()),
                bin_support_median=float(np.median(aud.cnt[i, j, k])), share_in_thin_bins=float((aud.cnt[i, j, k] < 100).mean()),
                dist_shares={f'{D_EDGES[b]:g}-{D_EDGES[b + 1]:g}': float(((d >= D_EDGES[b]) & (d < D_EDGES[b + 1])).mean()) for b in range(len(D_EDGES) - 1)},
                clock_shares={f'{C_EDGES[b]:g}-{C_EDGES[b + 1]:g}': float(((e >= C_EDGES[b]) & (e < C_EDGES[b + 1])).mean()) for b in range(len(C_EDGES) - 1)},
                def_shares={f'{F_EDGES[b]:g}-{F_EDGES[b + 1]:g}': float(((f >= F_EDGES[b]) & (f < F_EDGES[b + 1])).mean()) for b in range(len(F_EDGES) - 1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', default='rl/model/executor_v3g.keras')
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=2000)
    a = ap.parse_args()
    from api import PassValueModel
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games('holdout'))
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff, pt, aud = PassValueModel(), PlayTargets(), AuditedShotValue()
    rng = np.random.default_rng(0)
    starts = find_starts(G)
    starts = [starts[i] for i in np.sort(rng.choice(len(starts), min(a.n, len(starts)), replace=False))]
    prior_plays = rng.choice(pt.K, size=len(starts), p=pt.freq / pt.freq.sum())
    trained = Caller(diff, learn=False)
    trained.set_play_prior(pt.freq)
    trained.pi.load_weights(os.path.join(a.run, 'ckpt_pi.weights.h5'))
    trained.play_b.assign(np.load(os.path.join(a.run, 'ckpt_play_b.npy')))
    out = {}
    Sh, _ = real_shots(G)
    out['real_holdout_shots'] = dict(n=int(len(Sh['pts'])), real_pts=float(Sh['pts'].mean()), audited=float(aud(Sh['dist'], Sh['elapsed'], Sh['def1']).mean()),
                                     dist_mean=float(Sh['dist'].mean()), elapsed_mean=float(Sh['elapsed'].mean()), def1_mean=float(Sh['def1'].mean()))
    for name, caller, plays in (('executors_own_decisions', None, prior_plays), ('trained_policy', trained, trained.choose_plays(len(starts)))):
        log = []
        sim = Sim(model, diff, seed=12345, play_targets=pt, shot_log=log, **SIM).reset(G, table, starts, play=plays)
        sim.caller = caller
        sim.run()
        out[name] = summarize(log, aud)
        out[name]['reward_model_xpts'] = float(sim.reward.mean())
        shot = sim.end == 1
        # audited reward of the same episodes: replace each shot's model xPts by the audited value of its bin
        sv = {s['i']: float(aud(s['dist'], s['elapsed'], s['def1'])) for s in log}
        r = sim.reward.copy()
        for i_, v in sv.items():
            r[i_] = v
        out[name]['reward_audited'] = float(r.mean())
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items() if not isinstance(v, dict)}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'policy_audit.json'), 'w'), indent=1)
    print('real holdout shots', json.dumps({k: round(v, 3) for k, v in out['real_holdout_shots'].items()}))
    for nm in ('executors_own_decisions', 'trained_policy'):
        print(nm, 'dist', {k: round(v, 3) for k, v in out[nm]['dist_shares'].items()}, '\n   clock', {k: round(v, 3) for k, v in out[nm]['clock_shares'].items()}, '\n   def', {k: round(v, 3) for k, v in out[nm]['def_shares'].items()})
    rs = {'dist': {f'{D_EDGES[b]:g}-{D_EDGES[b + 1]:g}': float(((Sh['dist'] >= D_EDGES[b]) & (Sh['dist'] < D_EDGES[b + 1])).mean()) for b in range(len(D_EDGES) - 1)},
          'clock': {f'{C_EDGES[b]:g}-{C_EDGES[b + 1]:g}': float(((Sh['elapsed'] >= C_EDGES[b]) & (Sh['elapsed'] < C_EDGES[b + 1])).mean()) for b in range(len(C_EDGES) - 1)},
          'def': {f'{F_EDGES[b]:g}-{F_EDGES[b + 1]:g}': float(((Sh['def1'] >= F_EDGES[b]) & (Sh['def1'] < F_EDGES[b + 1])).mean()) for b in range(len(F_EDGES) - 1)}}
    print('real shares', {k: {a_: round(b_, 3) for a_, b_ in v.items()} for k, v in rs.items()})
    out['real_shares'] = rs
    json.dump(out, open(os.path.join(a.out, 'policy_audit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
