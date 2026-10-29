"""Stage 4b: decision-level play-caller (residual policy on top of the executor's own decisions), trained by REINFORCE inside the simulator.

What it controls: at every 0.4 s block in which an offensive player holds the ball, one of  keep | shoot | pass to teammate j (4 slots).
What it does NOT control: where the five players move (the executors do that; Stage 3 found play ids give them no useful signal), so this is
a shoot / pass / who-to-pass-to policy, not full play-calling. Reward = the simulator's (shot xPts, -1.16 turnover/violation).

Policy: logits = log(prior) + MLP(action features), softmax over valid actions. `prior` is the executor's decision hazards (what the simulator does by
itself), so zero residual reproduces the baseline simulator. Guard rails: pass actions whose difficulty features are out of the training range
(in_distribution False) or extrapolated are not available; shot xPts uses the guarded API rules (feature clamp, heave value), applied in batch here.
Trust region: residual bounded by RMAX (added after the first run, which drove the policy to shoot at once, see reports/rl-stage4b.md).
Learning: REINFORCE with a learned value baseline and a KL penalty to the prior (beta) so the policy cannot move far from observed behaviour.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import keras                             # noqa: E402
from executor_data import DT             # noqa: E402
from features import BASKET              # noqa: E402
from pass_features import shot_features  # noqa: E402

D_FEAT = 12
N_ACT = 6        # keep, shoot, pass x4
RMAX = 1.5         # the learned logit residual is bounded to +-1.5 (action probabilities can change by at most e^1.5 = 4.5x vs the executor's own)
HEAVE_CLOCK = 3.0   # seconds: a shot beyond the training distance range is only an available action in the last 3 s of the shot clock


def batch_xpts(diff, items):
    """items: list of (S, D, vel, shot_clock) -> (array of guarded xPts (same rules as PassValueModel.shot_xpts), bool array: valued as a heave)."""
    rows, ok = [], []
    out = np.zeros(len(items))
    heave = np.zeros(len(items), bool)
    lo_hi = diff.ranges['shot_features'] if diff.ranges is not None else None
    for n, (S, Dd, vel, sc) in enumerate(items):
        f = shot_features(S, Dd, BASKET, vel, sc)
        if f is None:
            continue
        if lo_hi is not None:
            if f['shot_dist'] > lo_hi['shot_dist'][1]:
                heave[n] = True
                continue
            f = {k: (min(max(v, lo_hi[k][0]), lo_hi[k][1]) if k in lo_hi and k != 'shot_dist' and np.isfinite(v) else v) for k, v in f.items()}
        rows.append(f)
        ok.append(n)
    if rows:
        out[ok] = diff._xpts(diff.shot, pd.DataFrame(rows))['xpts'].to_numpy()
    if lo_hi is not None:
        out[heave] = 3.0 * diff.ranges['heave_fg']
    return out, heave


class Caller:
    def __init__(self, diff, beta=0.02, lr=3e-3, seed=0, learn=True, c_fail=0.10, n_steps=8):
        self.diff, self.beta, self.learn, self.c_fail, self.n_steps = diff, beta, learn, c_fail, n_steps
        keras.utils.set_random_seed(seed)
        self.rng = np.random.default_rng(seed + 1)
        self.pi = keras.Sequential([keras.Input((D_FEAT,)), keras.layers.Dense(32, activation='gelu'), keras.layers.Dense(32, activation='gelu'),
                                    keras.layers.Dense(1, kernel_initializer='zeros')])
        self.v = keras.Sequential([keras.Input((6,)), keras.layers.Dense(32, activation='gelu'), keras.layers.Dense(1)])
        self.opt_pi = keras.optimizers.Adam(lr)
        self.opt_v = keras.optimizers.Adam(lr)
        self.buf = []
        self.pass_cache = {}

    # ------------------------------------------------------------------ features
    def act(self, sim, idx, can, o):
        """Returns shoot (n,) bool, pass_slot (n,) int (-1 none), and fills sim cache with the chosen pass's difficulty row."""
        n = len(idx)
        shoot = np.zeros(n, bool)
        pslot = np.full(n, -1)
        hold = np.flatnonzero(can)
        if len(hold) == 0:
            return shoot, pslot
        G = idx[hold]
        pos = sim.hist[2][G]
        vel = (sim.hist[2][G] - sim.hist[1][G]) / DT
        hz = 1.0 - (1.0 - np.clip(o['dec'][hold] * np.array([sim.shoot_scale, sim.pass_scale, 1.0]), 0, 0.999)) ** (0.4 / 2.0)
        hslot = np.array([int(np.flatnonzero(sim.ids[i, :5] == sim.holder[i])[0]) if (sim.ids[i, :5] == sim.holder[i]).any() else -1 for i in G])
        valid_h = hslot >= 0
        m = len(G)
        mates = np.zeros((m, 4), int)
        for r_, h in enumerate(hslot):
            mates[r_] = [s for s in range(5) if s != h][:4] if h >= 0 else [0, 1, 2, 3]
        # shot xPts for the holder and each teammate as shooter, one batch
        items = []
        for r_, i in enumerate(G):
            h = hslot[r_] if hslot[r_] >= 0 else 0
            Dd = pos[r_, 5:]
            items.append((pos[r_, h], Dd, vel[r_, h], sim.shot_clock[i]))
            for s in mates[r_]:
                items.append((pos[r_, s], Dd, vel[r_, s], sim.shot_clock[i]))
        xp, heave = batch_xpts(self.diff, items)
        xp = xp.reshape(m, 5)
        heave_self = heave.reshape(m, 5)[:, 0]
        states = []
        for r_, i in enumerate(G):
            h = hslot[r_] if hslot[r_] >= 0 else 0
            for s in mates[r_]:
                states.append(dict(P=pos[r_, h], T=pos[r_, s], defenders=pos[r_, 5:], basket=BASKET, shot_clock=sim.shot_clock[i], p_vel=vel[r_, h], t_vel=vel[r_, s]))
        pr = self.diff.predict_many(states)
        pc = pr['p_complete'].to_numpy().reshape(m, 4)
        okp = (pr['in_distribution'].to_numpy() & ~pr['extrapolated'].to_numpy()).reshape(m, 4)
        # prior over (keep, shoot, pass x4)
        tg = o['tgt'][hold]
        tp = np.take_along_axis(tg, mates, axis=1) * okp
        pass_p = hz[:, 1:2] * tp / np.maximum(tg.sum(1, keepdims=True), 1e-9)
        prior = np.concatenate([np.zeros((m, 1)), hz[:, 0:1], pass_p], axis=1)
        prior[:, 0] = np.clip(1.0 - prior[:, 1:].sum(1), 1e-6, 1.0)
        prior[~valid_h] = [1, 0, 0, 0, 0, 0]
        prior_logp = np.log(np.clip(prior, 1e-9, 1.0))
        shoot_ok = ~(heave_self & (sim.shot_clock[G] > HEAVE_CLOCK))
        mask = np.concatenate([np.ones((m, 1), bool), shoot_ok[:, None], okp], axis=1) & valid_h[:, None] | (~valid_h[:, None] & (np.arange(N_ACT) == 0)[None])
        # action features
        sc = np.clip(sim.shot_clock[G], 0, 24) / 24.0
        tin = sim.t_in[G] / 24.0
        nd_self = np.array([np.hypot(*(pos[r_, 5:] - pos[r_, max(hslot[r_], 0)]).T).min() for r_ in range(m)]) / 10.0
        F = np.zeros((m, N_ACT, D_FEAT), np.float32)
        F[:, :, 8] = sc[:, None]
        F[:, :, 9] = tin[:, None]
        F[:, :, 10] = xp[:, 0:1]
        F[:, :, 11] = nd_self[:, None]
        F[:, 1, 0] = 1
        F[:, 1, 2] = xp[:, 0]
        for j in range(4):
            F[:, 2 + j, 1] = 1
            F[:, 2 + j, 2] = xp[:, 1 + j]
            F[:, 2 + j, 3] = pc[:, j]
            F[:, 2 + j, 4] = pc[:, j] * xp[:, 1 + j] - (1 - pc[:, j]) * self.c_fail
            tpos = np.take_along_axis(pos, mates[:, j][:, None, None].repeat(2, 2), axis=1)[:, 0]
            F[:, 2 + j, 5] = np.hypot(*(tpos - BASKET).T) / 47.0
            F[:, 2 + j, 6] = np.array([np.hypot(*(pos[r_, 5:] - tpos[r_]).T).min() for r_ in range(m)]) / 10.0
            hp = np.array([pos[r_, max(hslot[r_], 0)] for r_ in range(m)])
            F[:, 2 + j, 7] = np.hypot(*(tpos - hp).T) / 30.0
        ctx = np.stack([sc, tin, xp[:, 0], nd_self, np.where(okp, F[:, 2:, 4], -2).max(1), np.where(okp, xp[:, 1:], 0).max(1)], axis=1).astype(np.float32)
        logits = prior_logp + RMAX * np.tanh(np.asarray(self.pi(F.reshape(-1, D_FEAT))).reshape(m, N_ACT) / RMAX)
        logits = np.where(mask, logits, -1e9)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        u = self.rng.random(m)[:, None]
        a = (np.cumsum(p, 1) < u).sum(1).clip(0, N_ACT - 1)
        if self.learn:
            self.buf.append(dict(F=F, mask=mask, prior=prior_logp, a=a, ctx=ctx, ep=G))
        sh = a == 1
        shoot[hold[sh]] = True
        for r_ in np.flatnonzero(a >= 2):
            pslot[hold[r_]] = mates[r_, a[r_] - 2]
            self.pass_cache[int(G[r_])] = (float(pc[r_, a[r_] - 2]), bool(okp[r_, a[r_] - 2]))
        self.last_stats = dict(p_shoot=float(p[:, 1].mean()), p_pass=float(p[:, 2:].sum(1).mean()), kl=float((p * (np.log(np.clip(p, 1e-9, 1)) - np.where(mask, prior_logp, 0))).sum(1).mean()))
        return shoot, pslot

    # ------------------------------------------------------------------ learning
    def update(self, returns):
        """returns: (N_episodes,) final reward of every episode; consumes the buffer."""
        if not self.buf:
            return {}
        F = np.concatenate([b['F'] for b in self.buf])
        mask = np.concatenate([b['mask'] for b in self.buf])
        prior = np.concatenate([b['prior'] for b in self.buf])
        a = np.concatenate([b['a'] for b in self.buf])
        ctx = np.concatenate([b['ctx'] for b in self.buf])
        G = returns[np.concatenate([b['ep'] for b in self.buf])].astype(np.float32)
        self.buf = []
        import tensorflow as tf
        Ft, mt, pt, at, ct, Gt = [tf.constant(x) for x in (F, mask, prior.astype(np.float32), a, ctx, G)]
        for _ in range(3):
            with tf.GradientTape() as tv:
                v = tf.squeeze(self.v(ct), -1)
                lv = tf.reduce_mean((v - Gt) ** 2)
            self.opt_v.apply_gradients(zip(tv.gradient(lv, self.v.trainable_variables), self.v.trainable_variables))
        adv = Gt - tf.squeeze(self.v(ct), -1)
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-6)
        for _ in range(self.n_steps):
            with tf.GradientTape() as t:
                res = RMAX * tf.tanh(tf.reshape(self.pi(tf.reshape(Ft, (-1, D_FEAT))), (-1, N_ACT)) / RMAX)
                lg = tf.where(mt, pt + res, -1e9)
                lp = tf.nn.log_softmax(lg, axis=-1)
                logp_a = tf.gather(lp, at, batch_dims=1)
                p = tf.exp(lp)
                kl = tf.reduce_sum(p * (lp - tf.where(mt, pt - tf.reduce_logsumexp(tf.where(mt, pt, -1e9), axis=-1, keepdims=True), 0.0)), axis=-1)
                loss = -tf.reduce_mean(tf.stop_gradient(adv) * logp_a) + self.beta * tf.reduce_mean(kl)
            self.opt_pi.apply_gradients(zip(t.gradient(loss, self.pi.trainable_variables), self.pi.trainable_variables))
        return dict(loss=float(loss), kl=float(tf.reduce_mean(kl)), v_loss=float(lv), n_dec=int(len(a)))
