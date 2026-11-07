"""Executor v5g (Stage 6B.2): v4g plus defensive intent.

    cd Code && python rl/train_executor5.py --data ../data/rl --out ../reports/rl-stage6b/exec5 --save rl/model/executor_v5g.keras

New token features for the five defenders (zeros for attackers): (i) man features: vector to the attacker he is matched to (stateless min-distance
assignment, defense.assign), its length, whether he guards the ball handler; (ii) the scheme goal: the centroid (z-scored knob space) of the scheme the
defence is told to play. Hindsight, as for the plays: in training the told scheme is the library cluster of the knob vector the defence REALLY showed over the
next 3 s (states +1, +2, +3 s), dropped 30% of the time; context gets a 'defence told' flag. The offence keeps its play goals (v4g).
Evaluation (validation games, goals off unless stated): one-step error of defenders and attackers against v4g, with and without the told scheme.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
import executor_v2 as E                  # noqa: E402
from executor_data import build, F_DIM, G_DIM   # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from play_cond import TOK_EXTRA          # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from train_executor2 import per_sample   # noqa: E402
from train_executor4 import goal_features4, CTX4   # noqa: E402
import defense as DF                      # noqa: E402

DEF_TOK = DF.DEF_TOK       # man features (4) + told scheme (6) + told minus current (6)
CTX5 = CTX4 + 1
TOK5 = TOK_EXTRA + DEF_TOK


def log(*a):
    print(*a, flush=True)


def def_features(S, games, lib, seed, goals=True, drop=0.3, told=None):
    """-> tok (N,10,DEF_TOK), ctx (N,1). told: optional array (N,) of scheme ids to tell instead of the hindsight cluster (-1 = nothing)."""
    N = len(S['y'])
    tok = np.zeros((N, 10, DEF_TOK), np.float32)
    ctx = np.zeros((N, 1), np.float32)
    rng = np.random.default_rng(seed + 23)
    for gi, d in enumerate(games):
        m0 = np.flatnonzero(S['game_idx'] == gi)
        if len(m0) == 0:
            continue
        st = S['state'][m0]
        pos = d['pos'][st]
        man = DF.assign(pos)
        tok[m0, 5:, :4] = DF.man_features(pos, d['ball'][st, :2], man)
        if not goals:
            continue
        fcf = d['frontcourt'][st]
        nfr = len(d['frame'])
        # how many of the next 15 states (3 s) are in the same possession, contiguous, with the same ten players: the goal is the scheme shown over the part of the window that exists
        avail = np.zeros(len(st), int)
        alive = fcf.copy()
        for k_ in range(1, 16):
            j = np.minimum(st + k_, nfr - 1)
            alive &= (st + k_ < nfr) & (d['poss_id'][j] == d['poss_id'][st]) & (d['frame'][j] - d['frame'][st] == 5 * k_) & (d['ids'][j] == d['ids'][st]).all(1)
            avail += alive
        ok = fcf & (avail >= 5)
        ok &= rng.random(len(st)) >= drop
        o1, o2, o3 = np.minimum(avail, 5), np.minimum(avail, 10), np.minimum(avail, 15)
        stc = st
        if told is not None:
            tk = told[m0]
            ok = fcf & (tk >= 0)
            g_ = lib.goal(np.maximum(tk, 0))
        else:
            raw = np.mean([DF.knobs(d['pos'][np.minimum(stc + o, nfr - 1)], d['ball'][np.minimum(stc + o, nfr - 1), :2], man) for o in (o1, o2, o3)], axis=0)
            g_ = lib.goal(lib.nearest(raw))
        idx = np.flatnonzero(ok)
        z_now = lib.z(DF.knobs(pos, d['ball'][st, :2], man))
        gg = np.concatenate([g_, g_ - z_now], axis=1)
        tok[m0[idx][:, None], np.arange(5, 10)[None, :], 4:] = gg[idx][:, None, :]
        ctx[m0[idx], 0] = 1.0
    return tok, ctx


def attach5(S, games, lib, seed, goals=True, play_goals=True, told=None):
    tok4, ctx4 = goal_features4(S, games, seed, goals=play_goals)
    tokd, ctxd = def_features(S, games, lib, seed, goals=goals, told=told)
    fc = ctx4[:, 2] > 0
    return dict(S, tokens=np.concatenate([S['tokens'], tok4, tokd], axis=2), ctx=np.concatenate([S['ctx'], ctx4, ctxd], axis=1), fc=fc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--save', required=True)
    ap.add_argument('--max-n', type=int, default=400000)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--n-games', type=int, default=0)
    ap.add_argument('--dec-h', type=int, default=50, help='decision label horizon in frames (50 = 2 s as before, 10 = one engine block)')
    a = ap.parse_args()
    import executor_data
    executor_data.DEC_H = a.dec_h
    fit_g, val_g = fit_val_games()
    if a.n_games:
        fit_g, val_g = fit_g[:a.n_games], val_g[:max(4, a.n_games // 4)]
    lib = DF.SchemeLibrary()
    t0 = time.time()
    FIT, VAL = load_games(a.data, fit_g), load_games(a.data, val_g)
    table = build_profiles(FIT)
    S0 = build(FIT, table, 3, a.max_n, 0, 'real', mask_mode='all_live')
    V0 = build(VAL, table, 3, 150000, 1, 'real', mask_mode='all_live')
    log(f'data ready ({time.time() - t0:.0f}s): {len(S0["y"])} fit, {len(V0["y"])} val samples')
    S = attach5(S0, FIT, lib, 0)
    V_off = attach5(V0, VAL, lib, 1, goals=False, play_goals=False)           # no goals at all (man features only)
    V_def = attach5(V0, VAL, lib, 1, goals=True, play_goals=False)            # hindsight scheme told, no play goals
    log(f'features ready ({time.time() - t0:.0f}s), token dim {S["tokens"].shape[2]}')
    keras.utils.set_random_seed(0)
    model = E.build_model(F_DIM + TOK5, G_DIM + CTX5, k_modes=3)
    n_steps = (len(S['y']) // 512) * a.epochs
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(2e-3, n_steps, alpha=0.05), clipnorm=1.0),
                  loss={'mix': E.mix_nll, 'dec': 'sparse_categorical_crossentropy', 'tgt': E.tgt_loss}, loss_weights={'mix': 1.0, 'dec': 0.3, 'tgt': 0.3})
    sw = {'mix': np.ones(len(S['y']), 'float32'), 'dec': (S['action'] >= 0).astype('float32'), 'tgt': (S['target_slot'] >= 0).astype('float32')}
    model.fit([S['tokens'], S['ctx']], {'mix': S['y'], 'dec': np.maximum(S['action'], 0), 'tgt': np.maximum(S['target_slot'], 0)}, sample_weight=sw, batch_size=512,
              epochs=a.epochs, verbose=2)
    res = dict(n_fit=int(len(S['y'])), n_val=int(len(V_off['y'])))
    for nm, V in (('no_goals', V_off), ('scheme_told', V_def)):
        ps = per_sample(model, V)
        fc = V['fc']
        res[nm] = {}
        for name, m in (('frontcourt', fc), ('backcourt_transition', ~fc)):
            res[nm][name] = {k: float(np.nanmean(v[m])) for k, v in ps.items() if k in ('nll_off', 'nll_def', 'ade1_off', 'ade1_def', 'dec_ll', 'tgt_ll')}
    log(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'exec5_eval.json'), 'w'), indent=1)
    model.save(a.save)


if __name__ == '__main__':
    main()
