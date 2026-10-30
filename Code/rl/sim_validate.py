"""Stage 4a validation: simulated possessions against real ones from the same start states.

    cd Code && python rl/sim_validate.py --data ../data/rl --model rl/model/executor_v2.keras --out ../reports/rl-stage4a [--n 3000] [--split val|holdout]

Start states: the first live half-court state (t_front >= 1.2 s, ball held by an offensive player, two history states available) of each
possession in the chosen games. `val` = the 81 Stage-2 validation games (development; default). `holdout` is scored once, at the end.
Real outcomes for the same starts: first shot by the controlling team in the possession (time, distance, points), passes before it,
possession end without a shot (turnover / violation / other). Sim and real are compared by game-clustered bootstrap of the real side.
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
from sim import Sim, END_NAMES           # noqa: E402
from features import BASKET              # noqa: E402


def log(*a):
    print(*a, flush=True)


def find_starts(games, per_game_cap=None, rng=None):
    out = []
    for gi, d in enumerate(games):
        fr = d['frame']
        ok_hist = np.r_[False, False, (np.diff(fr)[1:] == 5) & (np.diff(fr)[:-1] == 5)]
        cand = d['live'] & d['frontcourt'] & (d['t_front'] >= 1.2) & (d['holder'] != 0) & ok_hist
        in_off = np.array([d['holder'][k] in d['ids'][k, :5] for k in range(len(fr))]) if cand.any() else cand
        cand &= in_off
        seen = set()
        ks = []
        for k in np.flatnonzero(cand):
            p = d['poss_id'][k]
            if p in seen:
                continue
            seen.add(p)
            ks.append(k)
        if per_game_cap and len(ks) > per_game_cap:
            ks = list(rng.choice(ks, per_game_cap, replace=False))
        out += [(gi, int(k)) for k in ks]
    return out


def real_outcome(d, k):
    """Outcome of the real possession from state k: first shot (by the team in control) / end."""
    p = d['poss_id'][k]
    same = np.flatnonzero(d['poss_id'] == p)
    same = same[same >= k]
    fr = d['frame']
    f0, f1 = fr[k], fr[same[-1]] + 5
    team_ids = set(d['ids'][k, :5].tolist())
    sh = d['shots']
    s = sh[(sh['frame'] >= f0) & (sh['frame'] <= f1 + 25) & (sh['shooter'].isin(team_ids))] if len(sh) else sh
    ps = d['passes']
    pp = ps[(ps['frame'] >= f0) & (ps['frame'] <= f1) & (ps['live'] == 1) & (ps['passer'].isin(team_ids))] if len(ps) else ps
    dur = (f1 - f0) / 25.0
    if len(s):
        s = s.sort_values('frame')
        fs = s.iloc[0]
        t_shot = (fs['frame'] - f0) / 25.0
        n_before = int((pp['frame'] < fs['frame']).sum())
        pts = float(fs['made'] * (3 if fs['is3'] == 1 else 2))
        return dict(shot=1, t=t_shot, n_pass=n_before, pts=pts, dur=dur, fails=int((pp[pp['frame'] < fs['frame']]['label'] == 0).sum()) if 'label' in pp else 0)
    return dict(shot=0, t=dur, n_pass=int(len(pp)), pts=0.0, dur=dur, fails=int((pp['label'] == 0).sum()) if len(pp) and 'label' in pp else 0)


def summarise(shot, t, n_pass, pts, game=None):
    shot = np.asarray(shot, float)
    m = shot == 1
    return dict(p_shot=float(shot.mean()), t_first_shot=float(np.mean(np.asarray(t)[m])), passes_before_shot=float(np.mean(np.asarray(n_pass)[m])),
                pts_first_shot=float(np.mean(np.asarray(pts)[m])), passes_per_poss=float(np.mean(n_pass)), n=int(len(shot)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--split', default='val')
    ap.add_argument('--configs', default='1:1:1', help='comma list of shoot_scale:pass_scale:mask_ood(0/1)[:loss_rate]')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from api import PassValueModel
    t0 = time.time()
    fit_g, val_g = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, val_g if a.split == 'val' else split_games('holdout'))
    rng = np.random.default_rng(a.seed)
    starts = find_starts(G)
    if len(starts) > a.n:
        sel = np.sort(rng.choice(len(starts), a.n, replace=False))
        starts = [starts[i] for i in sel]
    os.makedirs(a.out, exist_ok=True)
    log(f'{len(starts)} start states from {len(G)} games ({time.time() - t0:.0f}s)')
    real = [real_outcome(G[g], k) for g, k in starts]
    R = summarise([r['shot'] for r in real], [r['t'] for r in real], [r['n_pass'] for r in real], [r['pts'] for r in real])
    R['p_noshot'] = 1 - R['p_shot']
    R['dur_mean'] = float(np.mean([r['dur'] for r in real]))
    log('real', json.dumps({k: round(v, 3) for k, v in R.items()}))
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    diff = PassValueModel()
    results = {}
    for cfg in a.configs.split(','):
        parts = cfg.split(':')
        ss, ps_, mk = parts[:3]
        lr_ = float(parts[3]) if len(parts) > 3 else 0.0
        sim = Sim(model, diff, tau=a.tau, seed=a.seed, shoot_scale=float(ss), pass_scale=float(ps_), mask_ood=bool(int(mk)), loss_rate=lr_).reset(G, table, starts)
        sim.run()
        log(f'[{cfg}] simulated ({time.time() - t0:.0f}s)')
        shot = (sim.end == 1).astype(int)
        S = summarise(shot, sim.shot_info[:, 3], sim.n_pass, np.nan_to_num(sim.shot_info[:, 1]))
        S['p_noshot'] = 1 - S['p_shot']
        S['dur_mean'] = float(sim.elapsed.mean())
        S['p_turnover'] = float((sim.end == 2).mean())
        S['p_violation'] = float((sim.end == 3).mean())
        S['p_timeout'] = float((sim.end == 4).mean())
        S['mean_reward'] = float(sim.reward.mean())
        S['masked_pass_share'] = float(sim.n_pass_fail_mask.sum() / max(sim.n_pass.sum(), 1))
        S['shot_dist_mean'] = float(np.nanmean(sim.shot_info[:, 0]))
        S['pass_len_mean'] = float(np.mean([x for l in sim.pass_len for x in l])) if any(sim.pass_len) else float('nan')
        log('sim ', json.dumps({k: round(v, 3) for k, v in S.items()}))
        results[cfg] = S
        np.savez_compressed(os.path.join(a.out, f'sim_{a.split}_{cfg.replace(":", "_")}.npz'), end=sim.end, reward=sim.reward, n_pass=sim.n_pass, shot_info=sim.shot_info, elapsed=sim.elapsed)
    # game-clustered bootstrap of the real side
    g = np.array([s[0] for s in starts])
    ug = np.unique(g)
    boots = []
    for _ in range(300):
        pick = rng.choice(ug, len(ug))
        ii = np.concatenate([np.flatnonzero(g == u) for u in pick])
        r = summarise([real[i]['shot'] for i in ii], [real[i]['t'] for i in ii], [real[i]['n_pass'] for i in ii], [real[i]['pts'] for i in ii])
        boots.append(r)
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in boots[0] if k != 'n'}
    json.dump(dict(real=R, sim=results, real_ci95=ci, args=vars(a), n_starts=len(starts)), open(os.path.join(a.out, f'validate_{a.split}.json'), 'w'), indent=1)
    for k, v_ in ci.items():
        log(f'  {k}: real {R.get(k, float("nan")):.3f} [{v_[0]:.3f}, {v_[1]:.3f}]  ' + '  '.join(f'{c}={results[c].get(k, float("nan")):.3f}' for c in results))


if __name__ == '__main__':
    main()
