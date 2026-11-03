"""Stage 6C/6D training driver: learning agents inside the quarter engine.

    # 6C: defence RL against the frozen imitation offence (both teams' defences use the learner)
    python rl/train_agents.py --learn def --symmetric --iters 60 --out ../reports/rl-stage6c/run1
    # 6D: league self-play: team A = learners (offence + defence), team B = an opponent drawn each iteration from the pool (+ the imitation engine)
    python rl/train_agents.py --learn off,def --pool --iters 120 --out ../reports/rl-stage6d/run1
    # best-response probe: fresh learner of one side against a frozen checkpoint
    python rl/train_agents.py --learn def --opp-ckpt X.pkl --iters 40 --out ...

Every iteration: n chains start from random quarter starts of the 401 RL-train games, play `blocks` engine blocks (0.4 s each), the learners are updated from the
per-possession returns. Checkpoints (policy weights) every --ckpt-every iterations, one-line json log per iteration, a short engine GIF every --gif-every.
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from engine_validate import quarter_starts   # noqa: E402
import league as L                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--learn', default='def', help='comma list of sides that learn: def, off')
    ap.add_argument('--symmetric', action='store_true', help='both roster teams use the learner (6C); otherwise team A learns and team B is the opponent')
    ap.add_argument('--pool', action='store_true', help='team B = an opponent drawn from the pool of past checkpoints (and the imitation engine) each iteration')
    ap.add_argument('--opp-ckpt', default=None, help='frozen opponent checkpoint (pickle) for team B')
    ap.add_argument('--init', default=None, help='checkpoint to start the learners from')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--blocks', type=int, default=300)
    ap.add_argument('--beta-def', type=float, default=0.05)
    ap.add_argument('--beta-off', type=float, default=0.02)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--rmax-off', type=float, default=None, help='bound of the offence logit residual (default 1.5)')
    ap.add_argument('--rmax-def', type=float, default=None, help='bound of the defence logit residual (default 1.5)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt-every', type=int, default=10)
    ap.add_argument('--gif-every', type=int, default=0)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras')
    ap.add_argument('--data', default='../data/rl')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    learn = set(a.learn.split(','))
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games('train'))
    qs = quarter_starts(G)
    print(f'{len(qs)} training quarter starts from {len(G)} games', flush=True)
    init = pickle.load(open(a.init, 'rb')) if a.init else None
    off_l, def_l = L.make_agents(st, init, learn=True, seed=a.seed, beta_def=a.beta_def, beta_off=a.beta_off, lr=a.lr, rmax_off=a.rmax_off, rmax_def=a.rmax_def)
    opp_fixed = pickle.load(open(a.opp_ckpt, 'rb')) if a.opp_ckpt else None
    pool = []
    rng = np.random.default_rng(a.seed + 99)
    logf = open(os.path.join(a.out, 'log.jsonl'), 'a')
    t_start = time.time()
    for it in range(a.iters + 1):
        t0 = time.time()
        pick = rng.choice(len(qs), a.n, replace=a.n > len(qs))
        starts = [(qs[j][0], qs[j][1]) for j in pick]
        A = (off_l if 'off' in learn else None, def_l if 'def' in learn else None)
        if a.symmetric:
            B, opp = A, 'self'
        elif opp_fixed is not None:
            ob, db = L.make_agents(st, opp_fixed, learn=False, seed=a.seed + 5)
            B, opp = (ob if opp_fixed.get('off') is not None else None, db if opp_fixed.get('def') is not None else None), 'frozen'
        elif a.pool:
            cands = ['imitation'] + list(range(len(pool)))
            c = cands[rng.integers(len(cands))]
            if c == 'imitation':
                B, opp = (None, None), 'imitation'
            else:
                ob, db = L.make_agents(st, pool[c], learn=False, seed=a.seed + 5)
                B, opp = (ob, db), f'pool{c}'
        else:
            B, opp = (None, None), 'imitation'
        eng = L.run_match(st, G, table, starts, seed=1000 * a.seed + it, blocks=a.blocks, A=A, B=B)
        rec = dict(it=it, opp=opp, seconds_run=time.time() - t0)
        if 'def' in learn:
            rec['def_usage'] = def_l.last.get('usage', np.zeros(0)).round(3).tolist()
            rec['def_kl_dec'] = def_l.last.get('kl')
        if it < a.iters:
            if 'def' in learn:
                rec['def'] = def_l.update_from(eng)
            if 'off' in learn:
                rec['off'] = off_l.update_from(eng)
        else:
            def_l.buf, off_l.buf = [], []
        m = L.match_metrics(eng)
        rec.update({k: m[k] for k in ('n_poss', 'ppp', 'ppp_team0_off', 'ppp_team1_off', 'diff_per_chain', 'fga_per_poss', 'fg3a_per_poss', 'tov_per_poss', 'stl_per_poss', 'fta_per_poss', 'rim_share', 'p_make_mean', 'def1_mean') if k in m})
        rec['seconds'] = time.time() - t0
        logf.write(json.dumps(rec, default=float) + '\n'); logf.flush()
        print(f"it {it:3d} opp {opp:9s} ppp {m['ppp']:.3f} (A off {m['ppp_team0_off']:.3f} / B off {m['ppp_team1_off']:.3f}) diff/chain {m['diff_per_chain']:+.2f} stl {m['stl_per_poss']:.3f} tov {m['tov_per_poss']:.3f} rim {m.get('rim_share', 0):.2f} "
              + (f"def kl {rec.get('def_kl_dec', 0):.3f} " if 'def' in learn else '') + f"{rec['seconds']:.0f}s", flush=True)
        if it % a.ckpt_every == 0 or it == a.iters:
            ck = dict(it=it, off=off_l.get_state() if 'off' in learn else None, rmax_off=off_l.rmax, rmax_def=def_l.rmax, **{'def': def_l.get_state() if 'def' in learn else None})
            pickle.dump(ck, open(os.path.join(a.out, f'ckpt_{it:04d}.pkl'), 'wb'))
            if a.pool and it > 0:
                pool.append(ck)
        if a.gif_every and it % a.gif_every == 0:
            try:
                from engine_gif import render
                gd = os.path.join(a.out, 'gifs'); os.makedirs(gd, exist_ok=True)
                s0 = starts[0]
                q = [x for x in qs if (x[0], x[1]) == s0][0][2]
                eng_g = L.run_match(st, G, table, [s0], seed=7, blocks=int(60 / 0.4) + 40, A=A, B=B)
                render(eng_g, 0, q, 60.0, os.path.join(gd, f'iter_{it:04d}.gif'), label=f'iter {it}  ')
            except Exception as e:                                       # a failed GIF must not stop a training run
                print('gif failed:', repr(e), flush=True)
    print(f'done in {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
