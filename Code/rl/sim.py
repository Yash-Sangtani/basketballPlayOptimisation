"""Stage 4a: possession simulator (vectorised over N parallel possessions).

    players   executor v2 (movement mixture sampled with one correlated draw per player per 0.4 s block, holder decision head, pass-target head)
    ball      rl/ball.py (held at the holder, straight-line flight at 28 ft/s)
    pass      difficulty model: Bernoulli(p_complete) at the release state; passes with in_distribution == False or extrapolated -> see `mask_ood`
    shot      shot model xPts of the shooter at the decision instant (guarded API); the episode ends at the shot (no rebounds, spec section 2)
    clocks    shot clock runs 0.2 s per state; violation at 0
    reward    shot: xPts; turnover or violation: -c_fail (1.16)

Approximations (inferred, to be validated by sim_validate.py): the executor's decision head is a probability of the action in the next 2 s, so it
is applied as a hazard of 1-(1-p)^(0.4/2) per 0.4 s block; the pass target is drawn from the target head at the decision instant;
the receiver's catch outcome is decided at the release state (as the difficulty model is defined) and played out over the flight.
Decisions are taken only while a player holds the ball and no pass is in flight.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import ball as B                         # noqa: E402
from executor_data import make_tokens, make_ctx, pos_codes, Y_SCALE, DT   # noqa: E402
from profiles import profile_lookup      # noqa: E402
from train_executor2 import mixture_stats   # noqa: E402
from features import BASKET              # noqa: E402

BLOCK = 2                 # states per model call (0.4 s)
DEC_WINDOW_S = 2.0
MAX_S = 30.0
END_NAMES = {0: 'running', 1: 'shot', 2: 'turnover', 3: 'violation', 4: 'timeout'}


class Sim:
    def __init__(self, model, diff, c_fail=1.16, mask_ood=True, tau=1.0, seed=0, shoot_scale=1.0, pass_scale=1.0, loss_rate=0.0):
        self.model, self.diff, self.c_fail, self.mask_ood, self.tau = model, diff, c_fail, mask_ood, tau
        self.rng = np.random.default_rng(seed)
        self.shoot_scale, self.pass_scale, self.loss_rate = shoot_scale, pass_scale, loss_rate

    # ---------------------------------------------------------------- reset
    def reset(self, games, table, starts):
        """starts: list of (game_index, state index k). History needs k-2, k-1, k in the data (caller checks)."""
        gi = np.array([s[0] for s in starts])
        ks = np.array([s[1] for s in starts])
        N = len(ks)
        self.N = N
        take = lambda key: np.stack([games[g][key][k] for g, k in starts])
        self.ids = take('ids')
        self.pc = np.stack([pos_codes(games[g], games[g]['ids'][k][None])[0] for g, k in starts])
        self.prof = np.stack([profile_lookup(table, games[g]['ids'][k][None])[0] for g, k in starts])
        self.hist = [np.stack([games[g]['pos'][k - 2] for g, k in starts]), np.stack([games[g]['pos'][k - 1] for g, k in starts]), take('pos')]
        self.holder = take('holder').astype(np.int64)
        self.ball = take('ball').astype(np.float64)
        sc = take('shot_clock').astype(np.float64)
        t_in = take('t_in_poss').astype(np.float64)
        self.shot_clock = np.where(np.isnan(sc), np.clip(24.0 - t_in, 0.0, 24.0), sc)
        self.t_in = t_in
        self.t_front = take('t_front').astype(np.float64)
        self.qt = take('quarter_time').astype(np.float64)
        self.elapsed = np.zeros(N)
        self.alive = np.ones(N, bool)
        self.end = np.zeros(N, np.int8)
        self.reward = np.zeros(N)
        self.n_pass = np.zeros(N, np.int32)
        self.n_pass_fail_mask = np.zeros(N, np.int32)
        self.shot_info = np.full((N, 4), np.nan)           # dist, xpts, shot_clock, elapsed
        self.pass_len = [[] for _ in range(N)]
        # flight state
        self.fl_on = np.zeros(N, bool)
        self.fl_p0 = np.zeros((N, 2))
        self.fl_p1 = np.zeros((N, 2))
        self.fl_T = np.zeros(N)
        self.fl_t = np.zeros(N)
        self.fl_slot = np.zeros(N, np.int64)
        self.fl_ok = np.zeros(N, bool)
        self.traj = [self.hist[2].copy()]
        return self

    # ---------------------------------------------------------------- one block
    def _tokens(self, idx):
        tok = make_tokens(self.hist[2][idx], self.hist[1][idx], self.hist[0][idx], self.ids[idx], self.holder[idx], self.ball[idx], self.pc[idx], self.prof[idx])
        ctx = make_ctx(self.ball[idx, 2], self.shot_clock[idx], self.t_in[idx], self.t_front[idx], self.qt[idx])
        return tok, ctx

    def step_block(self):
        idx = np.flatnonzero(self.alive)
        if len(idx) == 0:
            return False
        tok, ctx = self._tokens(idx)
        o = self.model.predict([tok, ctx], verbose=0, batch_size=2048)
        w, mu, lv = mixture_stats(o['mix'])
        n = len(idx)
        u = self.rng.random((n, 10, 1))
        k = (np.cumsum(w, -1) < u).sum(-1).clip(0, w.shape[-1] - 1)
        m = np.take_along_axis(mu, k[:, :, None, None, None], axis=2)[:, :, 0]
        s = np.exp(0.5 * np.take_along_axis(lv, k[:, :, None, None, None], axis=2)[:, :, 0])
        step = m + self.rng.standard_normal((n, 10, 1, 2)) * self.tau * s
        # --- decisions for holders (offence, ball held)
        can = (~self.fl_on[idx]) & (self.holder[idx] != 0)
        hz = 1.0 - (1.0 - np.clip(o['dec'] * np.array([self.shoot_scale, self.pass_scale, 1.0]), 0, 0.999)) ** (BLOCK * DT / DEC_WINDOW_S)
        shoot = np.zeros(n, bool)
        passes = np.zeros(n, bool)
        r = self.rng.random(n)
        shoot = can & (r < hz[:, 0])
        passes = can & ~shoot & (self.rng.random(n) < hz[:, 1] / np.maximum(1.0 - hz[:, 0], 1e-6))
        self._shots(idx[shoot])
        self._passes(idx[passes], o['tgt'][passes], np.flatnonzero(passes))
        # --- advance positions over BLOCK states
        p0 = self.hist[2][idx]
        for s_ in range(BLOCK):
            pos = p0 + step[:, :, s_, :] * Y_SCALE
            pos[..., 0] = pos[..., 0].clip(0, 94)
            pos[..., 1] = pos[..., 1].clip(0, 50)
            full = self.hist[2].copy()
            full[idx] = pos
            self.hist = [self.hist[1], self.hist[2], full]
            self._advance_state()
            self.traj.append(self.hist[2].copy())
        return bool(self.alive.any())

    def _advance_state(self):
        self.elapsed[self.alive] += DT
        self.shot_clock[self.alive] -= DT
        self.t_in[self.alive] += DT
        self.t_front[self.alive] += DT
        self.qt[self.alive] -= DT
        # ball
        for i in np.flatnonzero(self.alive):
            if self.fl_on[i]:
                self.fl_t[i] += DT
                if self.fl_t[i] >= self.fl_T[i]:
                    self.fl_on[i] = False
                    if self.fl_ok[i]:
                        self.holder[i] = self.ids[i, self.fl_slot[i]]
                        self.ball[i] = B.held_position(self.hist[2][i, self.fl_slot[i]])
                    else:
                        self._finish(i, 2, -self.c_fail)
                        continue
                else:
                    self.ball[i] = B.flight_position(self.fl_p0[i], self.fl_p1[i], self.fl_t[i])
            else:
                j = np.flatnonzero(self.ids[i] == self.holder[i])
                if len(j):
                    self.ball[i] = B.held_position(self.hist[2][i, j[0]])
            if self.alive[i] and (not self.fl_on[i]) and self.loss_rate > 0 and self.rng.random() < 1.0 - np.exp(-self.loss_rate * DT):
                self._finish(i, 2, -self.c_fail)
            elif self.alive[i] and self.shot_clock[i] <= 0:
                self._finish(i, 3, -self.c_fail)
            elif self.alive[i] and self.elapsed[i] >= MAX_S:
                self._finish(i, 4, 0.0)

    def _finish(self, i, code, reward):
        self.alive[i] = False
        self.end[i] = code
        self.reward[i] = reward

    # ---------------------------------------------------------------- events
    def _shots(self, ii):
        for i in ii:
            j = np.flatnonzero(self.ids[i, :5] == self.holder[i])
            if len(j) == 0:
                continue
            S = self.hist[2][i, j[0]]
            D = self.hist[2][i, 5:]
            vel = (self.hist[2][i, j[0]] - self.hist[1][i, j[0]]) / DT
            xp = self.diff.shot_xpts(S, D, BASKET, vel, self.shot_clock[i])
            if not np.isfinite(xp):
                xp = 0.0
            self.shot_info[i] = [float(np.hypot(*(BASKET - S))), xp, self.shot_clock[i], self.elapsed[i]]
            self._finish(i, 1, float(xp))

    def _passes(self, ii, tgt_p, pos_in_idx):
        if len(ii) == 0:
            return
        states, meta = [], []
        for i, tp in zip(ii, tgt_p):
            hj = np.flatnonzero(self.ids[i, :5] == self.holder[i])
            if len(hj) == 0:
                continue
            tp = tp.copy()
            tp[hj[0]] = 0
            if tp.sum() <= 0:
                continue
            t = int(self.rng.choice(5, p=tp / tp.sum()))
            P = self.hist[2][i, hj[0]]
            T = self.hist[2][i, t]
            states.append(dict(P=P, T=T, defenders=self.hist[2][i, 5:], basket=BASKET, shot_clock=self.shot_clock[i],
                               p_vel=(P - self.hist[1][i, hj[0]]) / DT, t_vel=(T - self.hist[1][i, t]) / DT))
            meta.append((i, hj[0], t, P, T))
        if not states:
            return
        r = self.diff.predict_many(states)
        for (i, hj, t, P, T), (_, row) in zip(meta, r.iterrows()):
            self.n_pass[i] += 1
            masked = (not bool(row['extrapolated'])) and (not bool(row['in_distribution'])) if self.mask_ood else False
            ok = bool(self.rng.random() < row['p_complete']) and not masked
            if masked:
                self.n_pass_fail_mask[i] += 1
            self.pass_len[i].append(float(np.hypot(*(T - P))))
            self.fl_on[i] = True
            self.fl_p0[i], self.fl_p1[i] = P, T
            self.fl_T[i] = B.flight_time(P, T)
            self.fl_t[i] = 0.0
            self.fl_slot[i] = t
            self.fl_ok[i] = ok
            self.holder[i] = 0

    def run(self, max_blocks=200):
        for _ in range(max_blocks):
            if not self.step_block():
                break
        return self
