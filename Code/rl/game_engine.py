"""Stage 6A.3: the game engine. Vectorised over N parallel quarters; one movement model (executor v4g) drives all ten players through whole possessions,
possession changes and transition; events are resolved with fitted models.

What is modelled (and by what):
  movement          executor v4g (front court and back court), mixture sampling with one correlated draw per player per 0.4 s block
  shoot / pass      the executor's decision heads (calibrated scales) or an external `caller` (as in sim.py)
  pass outcome      pass-difficulty model at release; a failed pass is a turnover (a steal credited to the nearest defender 55% of the time, else dead ball)
  shot outcome      shot model: P(three), P(make) of the shooter in his state; shooting fouls (p by shot type) -> free throws (player FT%); and-ones
  rebounds          P(offensive) by shot type and distance (play-by-play), rebounder chosen by a conditional logit on positions at the rebound time
  other turnovers   state-dependent hazard (nearest defender distance, speed, ...) times a calibration multiplier
  clocks            game clock, 24-s shot clock (reset to 24 on any change of possession and on an offensive rebound), dead-ball time charged after made shots
                    and dead-ball turnovers; free throws stop the clock
  score / stats     per team: points, FGA/FGM, 3PA/3PM, FTA/FTM, ORB/DRB, TOV, STL, possessions, passes
Defender decisions (help, gambles, fouls) are not learned yet: the defenders are the imitation executors (Stage 6B/6C add intent and RL).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import ball as B                         # noqa: E402
from executor_data import make_tokens, make_ctx, pos_codes, Y_SCALE, DT   # noqa: E402
from profiles import profile_lookup      # noqa: E402
from train_executor2 import mixture_stats   # noqa: E402
from features import BASKET              # noqa: E402
from engine_models import feats          # noqa: E402
from pass_features import shot_features  # noqa: E402

BLOCK = 2
PARAMS_PATH = os.path.join(HERE, 'model', 'engine_params.json')
STATS = ['pts', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'orb', 'drb', 'tov', 'stl', 'poss', 'passes', 'fouled', 'viol']


def rot(x):
    """Rotate attack-normalised coordinates by 180 degrees (the other team's frame)."""
    x = np.asarray(x, float).copy()
    x[..., 0] = 94.0 - x[..., 0]
    x[..., 1] = 50.0 - x[..., 1]
    return x


class Engine:
    def __init__(self, model, diff, plays=None, seed=0, shoot_scale=1.0, pass_scale=1.0, tov_mult=1.0, dead_make=4.0, dead_oob=2.0, steal_share=0.55,
                 foul_mult=1.0, params=None, tau=1.0, mask_ood=False, fail_keep=0.0, dist_mult=None, orb_mult=1.0, force_clock=1.2, force_p=0.9):
        self.model, self.diff, self.pt = model, diff, plays
        self.P = json.load(open(params or PARAMS_PATH))
        self.rng = np.random.default_rng(seed)
        self.shoot_scale, self.pass_scale, self.tov_mult = shoot_scale, pass_scale, tov_mult
        self.dead_make, self.dead_oob, self.steal_share, self.foul_mult, self.tau, self.mask_ood = dead_make, dead_oob, steal_share, foul_mult, tau, mask_ood
        self.cond = model.inputs[0].shape[-1] > 32
        self.orb_mult = orb_mult
        self.force_clock, self.force_p = force_clock, force_p     # with <= force_clock s left on the shot clock the holder shoots with probability force_p (players get a shot up)
        self.fail_keep = fail_keep                      # share of failed passes that are only deflections the offence recovers (no turnover)
        self.dist_edges = np.array([0, 6, 12, 18, 22, 25, 28, 33, 200.0])
        self.dist_mult = np.ones(len(self.dist_edges) - 1) if dist_mult is None else np.asarray(dist_mult, float)   # multiplier of the shoot decision by distance bin (calibrated on the real shot mix)
        self.caller = None
        self.defender_policy = None          # hook for Stage 6C: callable(engine, idx) -> per-defender modifiers (unused)

    # ------------------------------------------------------------------ setup
    def reset(self, games, table, starts):
        """starts: list of (game index, state index): the first live state of a quarter; its ten players are the two rosters for the whole quarter."""
        N = len(starts)
        self.N = N
        g = [games[a] for a, _ in starts]
        ks = [k for _, k in starts]
        self.R_ids = np.stack([d['ids'][k] for d, k in zip(g, ks)])
        self.R_pc = np.stack([pos_codes(d, d['ids'][k][None])[0] for d, k in zip(g, ks)])
        self.R_prof = np.stack([profile_lookup(table, d['ids'][k][None])[0] for d, k in zip(g, ks)])
        ftp = self.P['ft']['player_ft']
        self.R_ft = np.vectorize(lambda p: ftp.get(str(int(p)), self.P['ft']['ft_pct']))(self.R_ids)
        self.order = np.tile(np.arange(10), (N, 1))
        self.hist = [np.stack([d['pos'][k - 2] for d, k in zip(g, ks)]).astype(float), np.stack([d['pos'][k - 1] for d, k in zip(g, ks)]).astype(float),
                     np.stack([d['pos'][k] for d, k in zip(g, ks)]).astype(float)]
        self.holder = np.stack([d['holder'][k] for d, k in zip(g, ks)]).astype(np.int64)
        self.ball = np.stack([d['ball'][k] for d, k in zip(g, ks)]).astype(float)
        self.game_clock = np.full(N, 720.0)
        self.shot_clock = np.full(N, 24.0)
        self.t_in = np.zeros(N)
        self.t_front = np.zeros(N)
        self.crossed = self.ball[:, 0] < 47.0
        self.off_team = np.zeros(N, int)         # 0 = roster slots 0-4 of R_ are on offence (A), 1 = B; A starts on offence
        self.dead = np.zeros(N)                  # seconds of dead ball left
        self.dead_kind = np.zeros(N, int)        # what happens when the dead time ends: 0 none, 1 inbound at baseline, 2 inbound at the ball spot
        self.reb_t = np.full(N, -1.0)            # >= 0: a missed shot is in the air / rebound pending
        self.reb_info = [None] * N
        self.play = np.full(N, -1, np.int32)
        self.alive = np.ones(N, bool)
        self.score = np.zeros((N, 2))
        self.stats = {k: np.zeros((N, 2)) for k in STATS}
        self.fl_on = np.zeros(N, bool)
        self.fl_p0 = np.zeros((N, 2)); self.fl_p1 = np.zeros((N, 2)); self.fl_T = np.zeros(N); self.fl_t = np.zeros(N)
        self.fl_slot = np.zeros(N, np.int64); self.fl_ok = np.zeros(N, bool); self.fl_target_sl = np.zeros(N, np.int64)
        self.traj = [self.hist[2].copy()]
        self.ball_traj = [self.ball.copy()]
        self.holder_traj = [self.holder.copy()]
        self.order_traj = [self.order.copy()]
        self.score_traj = [self.score.copy()]
        self.clock_traj = [self.game_clock.copy()]
        self.shot_log = []
        self.event_log = []
        for i in range(N):
            self.stats['poss'][i, 0] += 1
            self._new_possession_play(i)
        return self

    # ------------------------------------------------------------------ helpers
    def ids(self, i):
        return self.R_ids[i, self.order[i]]

    def team_of_slot(self, i, s):
        """0/1 roster team of a current slot s."""
        return int(self.order[i, s] >= 5)

    def _off(self, i):
        return int(self.order[i, 0] >= 5)

    def _new_possession_play(self, i):
        if self.pt is not None and self.cond:
            self.play[i] = int(self.rng.choice(self.pt.K, p=self.pt.freq / self.pt.freq.sum()))

    def _flip(self, i, holder_slot_new=None, ball_xy=None):
        """Change possession: the old defence becomes the offence (slots swap, coordinates rotate)."""
        self.order[i] = np.r_[self.order[i, 5:], self.order[i, :5]]
        for h in range(3):
            self.hist[h][i] = np.concatenate([rot(self.hist[h][i][5:]), rot(self.hist[h][i][:5])])
        self.ball[i, :2] = rot(self.ball[i, :2])
        self.shot_clock[i] = 24.0
        self.t_in[i] = 0.0
        self.t_front[i] = 0.0
        self.crossed[i] = self.ball[i, 0] < 47.0
        self.fl_on[i] = False
        self.reb_t[i] = -1.0
        self.stats['poss'][i, self._off(i)] += 1
        self._new_possession_play(i)

    def _holder_slot(self, i):
        j = np.flatnonzero(self.ids(i)[:5] == self.holder[i])
        return int(j[0]) if len(j) else -1

    def _set_holder(self, i, slot):
        self.holder[i] = self.ids(i)[slot]
        self.ball[i] = B.held_position(self.hist[2][i, slot])

    def _tokens(self, idx):
        ids = np.stack([self.ids(i) for i in idx])
        pc = np.take_along_axis(self.R_pc[idx], self.order[idx], axis=1)
        prof = np.take_along_axis(self.R_prof[idx], self.order[idx][:, :, None], axis=1)
        tok = make_tokens(self.hist[2][idx], self.hist[1][idx], self.hist[0][idx], ids, self.holder[idx], self.ball[idx], pc, prof)
        qt = self.game_clock[idx]
        ctx = make_ctx(self.ball[idx, 2], self.shot_clock[idx], self.t_in[idx], self.t_front[idx], qt)
        if self.cond:
            tx, cx = self.pt.features4(self.hist[2][idx], self.t_front[idx], self.play[idx], self.crossed[idx])
            tok, ctx = np.concatenate([tok, tx], axis=2), np.concatenate([ctx, cx], axis=1)
        return tok, ctx

    def _inbound(self, i, baseline):
        """Place the ball with an inbounder of the offence: under the basket just scored on (baseline) or at the spot where the ball went out."""
        pcs = np.take_along_axis(self.R_pc[i], self.order[i], axis=0)[:5]
        cand = np.flatnonzero(pcs <= 1)
        s = int(self.rng.choice(cand if len(cand) else np.arange(5)))
        if baseline:
            spot = np.array([92.5, 25.0 + self.rng.uniform(-3, 3)])
        else:
            spot = np.clip(self.ball[i, :2], [1, 1], [93, 49])
        for h in range(3):
            self.hist[h][i, s] = spot
        self._set_holder(i, s)
        self.crossed[i] = False
        self.t_front[i] = 0.0

    # ------------------------------------------------------------------ events
    def _team_stat(self, i, name, v=1.0, team=None):
        self.stats[name][i, self._off(i) if team is None else team] += v

    def _shot(self, i):
        s = self._holder_slot(i)
        if s < 0:
            return
        S = self.hist[2][i, s]
        D = self.hist[2][i, 5:]
        vel = (self.hist[2][i, s] - self.hist[1][i, s]) / DT
        f = shot_features(S, D, BASKET, vel, self.shot_clock[i])
        if f is None:
            return
        lo_hi = self.diff.ranges['shot_features'] if self.diff.ranges is not None else None
        heave = lo_hi is not None and f['shot_dist'] > lo_hi['shot_dist'][1]
        if lo_hi is not None:
            f = {k: (min(max(v, lo_hi[k][0]), lo_hi[k][1]) if k in lo_hi and k != 'shot_dist' and np.isfinite(v) else v) for k, v in f.items()}
        o = self.diff._xpts(self.diff.shot, pd.DataFrame([f]))
        pm, p3 = float(o['p_make'].iloc[0]), float(o['p_three'].iloc[0])
        if heave:
            pm, p3 = self.diff.ranges['heave_fg'], 1.0
        dist = float(np.hypot(*(BASKET - S)))
        is3 = bool(self.rng.random() < p3)
        shooter_ft = self.R_ft[i, self.order[i, s]]
        pf = (self.P['ft']['p_foul_3pt'] if is3 else self.P['ft']['p_foul_2pt']) * self.foul_mult
        off = self._off(i)
        self.shot_log.append(dict(i=i, dist=dist, is3=is3, clock=float(self.shot_clock[i]), p_make=pm, def1=f['def1']))
        if self.rng.random() < pf:                                     # fouled in the act of shooting: free throws, no field-goal attempt
            n = 3 if is3 else 2
            self.stats['fouled'][i, off] += 1
            self._free_throws(i, n, shooter_ft, final_rebound_dist=14.0)
            return
        self.stats['fga'][i, off] += 1
        self.stats['fg3a'][i, off] += int(is3)
        if self.rng.random() < pm:
            pts = 3 if is3 else 2
            self.stats['fgm'][i, off] += 1
            self.stats['fg3m'][i, off] += int(is3)
            self.score[i, off] += pts
            self.stats['pts'][i, off] += pts
            if self.rng.random() < self.P['ft']['and_one_per_make']:
                self._free_throws(i, 1, shooter_ft, final_rebound_dist=14.0)
            else:
                self._after_make(i)
        else:                                                         # miss: the ball is in the air; the rebound resolves in ~1.6 s
            self.holder[i] = 0
            self.reb_t[i] = 1.6
            self.reb_info[i] = (dist, is3)

    def _free_throws(self, i, n, p_ft, final_rebound_dist):
        off = self._off(i)
        made = 0
        last = False
        for k in range(n):
            self.stats['fta'][i, off] += 1
            last = bool(self.rng.random() < p_ft)
            if last:
                made += 1
                self.stats['ftm'][i, off] += 1
        self.score[i, off] += made
        self.stats['pts'][i, off] += made
        self.holder[i] = 0
        self.fl_on[i] = False
        if last:                                                      # the final free throw decides: made -> the other team inbounds
            self._after_make(i, charge=1.0)
        else:                                                         # missed -> live rebound
            self.reb_t[i] = 1.2
            self.reb_info[i] = (final_rebound_dist, False)

    def _after_make(self, i, charge=None):
        self.holder[i] = 0
        self.fl_on[i] = False
        self._flip(i)
        self.dead[i] = self.dead_make if charge is None else charge
        self.dead_kind[i] = 1
        self.game_clock[i] -= 0.0

    def _rebound(self, i):
        dist, is3 = self.reb_info[i]
        self.reb_t[i] = -1.0
        bins = self.P['orb_bins']
        b = int(np.clip(np.searchsorted(bins, dist, side='right') - 1, 0, len(bins) - 2))
        p_orb = self.P['orb']['1' if is3 else '0'][b]
        orb = bool(self.rng.random() < min(p_orb * self.orb_mult, 0.95))
        w = np.array(self.P['rebounder']['w'])
        pos = self.hist[2][i]
        pc = np.take_along_axis(self.R_pc[i], self.order[i], axis=0)
        z = feats(pos, pc) @ w
        sl = np.arange(5) if orb else np.arange(5, 10)
        p = np.exp(z[sl] - z[sl].max())
        p /= p.sum()
        s = int(sl[self.rng.choice(5, p=p)])
        off = self._off(i)
        self.stats['orb' if orb else 'drb'][i, off if orb else 1 - off] += 1
        if orb:
            self.shot_clock[i] = 24.0
            self.t_in[i] = 0.0
            self._set_holder(i, s)
            self.play[i] = -1
        else:
            self.holder[i] = 0
            self._flip(i)
            self._set_holder(i, s - 5)

    def _turnover(self, i, steal, nearest_slot=None):
        """The offence loses the ball. steal: the nearest defender takes it live; otherwise dead ball (out of bounds / violation) and an inbound."""
        off = self._off(i)
        self.stats['tov'][i, off] += 1
        self.holder[i] = 0
        self.fl_on[i] = False
        if steal and nearest_slot is not None:
            self.stats['stl'][i, 1 - off] += 1
            self._flip(i)
            self._set_holder(i, nearest_slot - 5)
        else:
            self._flip(i)
            self.dead[i] = self.dead_oob
            self.dead_kind[i] = 2

    # ------------------------------------------------------------------ passes
    def _pass(self, i, tgt_p):
        s = self._holder_slot(i)
        if s < 0:
            return
        tp = tgt_p.copy()
        tp[s] = 0
        if tp.sum() <= 0:
            return
        t = int(self.rng.choice(5, p=tp / tp.sum()))
        P, T = self.hist[2][i, s], self.hist[2][i, t]
        r = self.diff.predict_one(P, T, self.hist[2][i, 5:], self.shot_clock[i], (P - self.hist[1][i, s]) / DT, (T - self.hist[1][i, t]) / DT) if hasattr(self.diff, 'predict_one') else None
        if r is None:
            r = self.diff.predict(P, T, self.hist[2][i, 5:], BASKET, self.shot_clock[i], (P - self.hist[1][i, s]) / DT, (T - self.hist[1][i, t]) / DT)
        ok = bool(self.rng.random() < r['p_complete']) or bool(self.rng.random() < self.fail_keep)
        self.stats['passes'][i, self._off(i)] += 1
        vT = (T - self.hist[1][i, t]) / DT
        tgt = T.copy()
        tf = B.flight_time(P, T)
        for _ in range(3):
            disp = vT * tf
            n = np.hypot(*disp)
            if n > 18.0 * tf:
                disp = disp / n * 18.0 * tf
            tgt = np.array([np.clip(T[0] + disp[0], 0, 94), np.clip(T[1] + disp[1], 0, 50)])
            tf = B.flight_time(P, tgt)
        self.fl_on[i] = True
        self.fl_p0[i], self.fl_p1[i] = P, tgt
        self.fl_T[i], self.fl_t[i] = B.flight_time(P, tgt), 0.0
        self.fl_slot[i], self.fl_ok[i] = t, ok
        self.holder[i] = 0

    # ------------------------------------------------------------------ one block
    def step_block(self):
        live = self.alive & (self.dead <= 0)
        dead = self.alive & (self.dead > 0)
        for i in np.flatnonzero(dead):
            self._advance_clock(i, BLOCK * DT, running=self.dead_kind[i] != 0 and self.dead_charge(i))
            self.dead[i] -= BLOCK * DT
            if self.dead[i] <= 0:
                self._inbound(i, baseline=(self.dead_kind[i] == 1))
                self.dead_kind[i] = 0
        idx = np.flatnonzero(live)
        if len(idx):
            self._live_block(idx)
        for i in np.flatnonzero(self.alive):
            if self.game_clock[i] <= 0:
                self.alive[i] = False
        return bool(self.alive.any())

    def dead_charge(self, i):
        return True

    def _advance_clock(self, i, dt, running=True):
        if running:
            self.game_clock[i] -= dt

    def _live_block(self, idx):
        tok, ctx = self._tokens(idx)
        o = self.model.predict([tok, ctx], verbose=0, batch_size=2048)
        w, mu, lv = mixture_stats(o['mix'])
        n = len(idx)
        u = self.rng.random((n, 10, 1))
        k = (np.cumsum(w, -1) < u).sum(-1).clip(0, w.shape[-1] - 1)
        m = np.take_along_axis(mu, k[:, :, None, None, None], axis=2)[:, :, 0]
        s = np.exp(0.5 * np.take_along_axis(lv, k[:, :, None, None, None], axis=2)[:, :, 0])
        step = m + self.rng.standard_normal((n, 10, 1, 2)) * self.tau * s
        can = (~self.fl_on[idx]) & (self.holder[idx] != 0) & (self.reb_t[idx] < 0)
        hz = 1.0 - (1.0 - np.clip(o['dec'] * np.array([self.shoot_scale, self.pass_scale, 1.0]), 0, 0.999)) ** (BLOCK * DT / 2.0)
        for r_, i in enumerate(idx):
            if not can[r_]:
                continue
            if self.shot_clock[i] <= self.force_clock and self.rng.random() < self.force_p:
                self._shot(i)
                continue
            sl_ = self._holder_slot(i)
            hz0 = hz[r_, 0]
            if sl_ >= 0:
                dd_ = np.hypot(*(self.hist[2][i, sl_] - BASKET))
                hz0 = min(hz0 * self.dist_mult[int(np.clip(np.searchsorted(self.dist_edges, dd_, side='right') - 1, 0, len(self.dist_mult) - 1))], 0.999)
            if self.rng.random() < hz0:
                self._shot(i)
            elif self.rng.random() < hz[r_, 1] / max(1.0 - hz[r_, 0], 1e-6):
                self._pass(i, o['tgt'][r_])
            else:
                self._tov_hazard(i)
        p0 = self.hist[2].copy()
        for s_ in range(BLOCK):
            pos = p0[idx] + step[:, :, s_, :] * Y_SCALE
            for r_, i in enumerate(idx):
                if self.fl_on[i] and self.fl_ok[i] and self.alive[i]:
                    sl = self.fl_slot[i]
                    cur = self.hist[2][i, sl]
                    to = self.fl_p1[i] - cur
                    dist = float(np.hypot(*to))
                    if dist > 1e-6:
                        rem = max(self.fl_T[i] - self.fl_t[i], DT)
                        pos[r_, sl] = cur + to / dist * min(dist / rem, 18.0) * DT
            pos[..., 0] = pos[..., 0].clip(0, 94)
            pos[..., 1] = pos[..., 1].clip(0, 50)
            full = self.hist[2].copy()
            full[idx] = pos
            self.hist = [self.hist[1], self.hist[2], full]
            self._advance_state(idx)
            self.traj.append(self.hist[2].copy())
            self.ball_traj.append(self.ball.copy())
            self.holder_traj.append(self.holder.copy())
            self.order_traj.append(self.order.copy())
            self.score_traj.append(self.score.copy())
            self.clock_traj.append(self.game_clock.copy())

    def _tov_hazard(self, i):
        c = self.P['turnover']
        s = self._holder_slot(i)
        if s < 0:
            return
        hp = self.hist[2][i, s]
        dd = np.sort(np.hypot(*(self.hist[2][i, 5:] - hp).T))
        sp = np.hypot(*(self.hist[2][i, s] - self.hist[1][i, s])) / DT
        x = np.array([dd[0], dd[1], sp, self.t_in[i] / 24.0, float(self.crossed[i]), 1.0 / (0.5 + dd[0])])
        z = float(np.dot(c['coef'], x) + c['intercept'])
        p = 1.0 / (1.0 + np.exp(-z))
        p = 1.0 - (1.0 - p) ** BLOCK if False else p * BLOCK
        if self.rng.random() < p * self.tov_mult:
            steal = bool(self.rng.random() < self.steal_share)
            near = int(5 + np.argmin(np.hypot(*(self.hist[2][i, 5:] - hp).T)))
            self._turnover(i, steal, near)

    def _advance_state(self, idx):
        for i in idx:
            if not self.alive[i]:
                continue
            self.game_clock[i] -= DT
            self.shot_clock[i] -= DT
            self.t_in[i] += DT
            if self.crossed[i]:
                self.t_front[i] += DT
            if self.reb_t[i] >= 0:
                self.reb_t[i] -= DT
                self.ball[i] = np.r_[BASKET + np.array([0.0, 0.0]), 10.0] if self.reb_t[i] > 0.4 else self.ball[i]
                if self.reb_t[i] <= 0:
                    self._rebound(i)
                continue
            if self.fl_on[i]:
                self.fl_t[i] += DT
                if self.fl_t[i] >= self.fl_T[i]:
                    self.fl_on[i] = False
                    if self.fl_ok[i]:
                        self._set_holder(i, int(self.fl_slot[i]))
                    else:
                        P = self.fl_p1[i]
                        near = int(5 + np.argmin(np.hypot(*(self.hist[2][i, 5:] - P).T)))
                        self.ball[i, :2] = P
                        self._turnover(i, bool(self.rng.random() < self.steal_share), near)
                        continue
                else:
                    self.ball[i] = B.flight_position(self.fl_p0[i], self.fl_p1[i], self.fl_t[i])
            else:
                s = self._holder_slot(i)
                if s >= 0:
                    self.ball[i] = B.held_position(self.hist[2][i, s])
            if self.alive[i] and not self.crossed[i] and self.ball[i, 0] < 47.0:
                self.crossed[i] = True
            if self.shot_clock[i] <= 0 and self.holder[i] != 0:
                self.stats['viol'][i, self._off(i)] += 1
                self._turnover(i, False)

    def run(self, max_blocks=2400):
        for _ in range(max_blocks):
            if not self.step_block():
                break
        return self

    def summary(self):
        return {k: v.sum(0) for k, v in self.stats.items()}
