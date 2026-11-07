"""Stage 6C/6D: learning agents inside the game engine.

  DefAgent   the defence's scheme caller: every 3 s of the front court (and at each change of possession) it picks one of the K library schemes; the
             executor v5g moves the five defenders to play it. logits = log(library frequency) + RMAX*tanh(MLP(state)); the residual is bounded and a KL penalty to the
             library frequencies is the 'human-likeness budget'.
  OffAgent   the offence's shoot / pass / who-to-pass decision residual (the Stage 4b/5 Caller, with the same bounded residual, KL penalty and support guard),
             learning inside the engine from per-possession returns.
Returns are per possession: the offence gets the points its team scores in the possession, the defence the negative of that (a zero-sum game; the engine plays on,
so what a turnover or a rebound leads to is counted in the next possession of whoever gets the ball).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
import tensorflow as tf                  # noqa: E402
import executor_v2 as E                  # noqa: E402
from caller import Caller, RMAX          # noqa: E402
from game_engine import Engine           # noqa: E402
import defense as DF                     # noqa: E402

BASKET = DF.BASKET


def possession_returns(eng):
    """{(episode, possession number): points scored by the offence in it}"""
    return {(i, c): p for i, c, off, p in eng.poss_log}


class DefAgent:
    """Scheme caller. Prior = a behaviour-cloned model of what real defences play given the state (build_bc.py); logits = log prior + bounded residual."""

    def __init__(self, lib, beta=0.05, lr=3e-3, seed=0, learn=True, min_freq=0.01, n_steps=8, bc_path=None, rmax=None):
        self.lib, self.beta, self.learn, self.n_steps = lib, beta, learn, n_steps
        self.rmax = RMAX if rmax is None else rmax
        self.K = lib.K
        self.allowed = lib.freq >= min_freq
        self.bc = keras.saving.load_model(bc_path or os.path.join(HERE, 'model', 'scheme_bc.keras'), compile=False)
        self.D = DF.D_AGENT
        keras.utils.set_random_seed(seed)
        self.rng = np.random.default_rng(seed + 7)
        self.pi = keras.Sequential([keras.Input((self.D,)), keras.layers.Dense(32, activation='gelu'), keras.layers.Dense(32, activation='gelu'),
                                    keras.layers.Dense(self.K, kernel_initializer='zeros')])
        self.v = keras.Sequential([keras.Input((self.D,)), keras.layers.Dense(32, activation='gelu'), keras.layers.Dense(1, kernel_initializer='zeros')])
        self.opt_pi, self.opt_v = keras.optimizers.Adam(lr), keras.optimizers.Adam(lr)
        self.buf = []
        self.last = {}

    # state ------------------------------------------------------------------
    def get_state(self):
        return [w.copy() for w in self.pi.get_weights()], [w.copy() for w in self.v.get_weights()]

    def set_state(self, st):
        self.pi.set_weights(st[0])
        self.v.set_weights(st[1])

    def features(self, eng, idx, team):
        return DF.agent_features(self.lib, eng.hist[2][idx], eng.ball[idx, :2], eng.shot_clock[idx], eng.t_front[idx], eng.scheme[idx, team])

    def prior_logp(self, F):
        p = np.asarray(self.bc(F)) * self.allowed[None, :]
        p = p / p.sum(1, keepdims=True)
        return np.log(np.clip(p, 1e-9, 1.0)).astype(np.float32)

    def probs(self, F, logp0=None):
        logp0 = self.prior_logp(F) if logp0 is None else logp0
        lg = logp0 + self.rmax * np.tanh(np.asarray(self.pi(F)) / self.rmax)
        lg = np.where(self.allowed[None, :], lg, -1e9)
        p = np.exp(lg - lg.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def decide(self, eng, idx, team):
        F = self.features(eng, idx, team)
        logp0 = self.prior_logp(F)
        p = self.probs(F, logp0)
        u = self.rng.random(len(idx))[:, None]
        a = (np.cumsum(p, 1) < u).sum(1).clip(0, self.K - 1)
        if self.learn:
            self.buf.append(dict(F=F, a=a, ep=idx.copy(), pc=eng.poss_count[idx].copy(), logp0=logp0))
        self.last = dict(kl=float((p * (np.log(np.clip(p, 1e-9, 1)) - logp0)).sum(1).mean()), usage=np.bincount(a, minlength=self.K) / max(len(a), 1))
        return a

    def update_from(self, eng):
        pts = possession_returns(eng)
        Fs, As, Rs, Ls = [], [], [], []
        for b in self.buf:
            r = np.array([pts.get((int(e), int(c)), np.nan) for e, c in zip(b['ep'], b['pc'])])
            k = ~np.isnan(r)
            if k.any():
                Fs.append(b['F'][k]); As.append(b['a'][k]); Rs.append(-r[k]); Ls.append(b['logp0'][k])
        self.buf = []
        if not Fs:
            return {}
        F, a, G, L0 = np.concatenate(Fs), np.concatenate(As), np.concatenate(Rs).astype(np.float32), np.concatenate(Ls)
        Ft, at, Gt, L0t = tf.constant(F), tf.constant(a), tf.constant(G), tf.constant(L0)
        for _ in range(3):
            with tf.GradientTape() as t:
                lv = tf.reduce_mean((tf.squeeze(self.v(Ft), -1) - Gt) ** 2)
            self.opt_v.apply_gradients(zip(t.gradient(lv, self.v.trainable_variables), self.v.trainable_variables))
        adv = Gt - tf.squeeze(self.v(Ft), -1)
        adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-6)
        msk = tf.constant(self.allowed)
        for _ in range(self.n_steps):
            with tf.GradientTape() as t:
                lg = tf.where(msk[None, :], L0t + self.rmax * tf.tanh(self.pi(Ft) / self.rmax), -1e9)
                lp = tf.nn.log_softmax(lg, axis=-1)
                lpa = tf.gather(lp, at, batch_dims=1)
                kl = tf.reduce_sum(tf.exp(lp) * (lp - L0t) * tf.cast(msk[None, :], tf.float32), -1)
                loss = -tf.reduce_mean(tf.stop_gradient(adv) * lpa) + self.beta * tf.reduce_mean(kl)
            self.opt_pi.apply_gradients(zip(t.gradient(loss, self.pi.trainable_variables), self.pi.trainable_variables))
        return dict(n_dec=int(len(a)), loss=float(loss), kl=float(tf.reduce_mean(kl)), v_loss=float(lv), ret_mean=float(G.mean()))


class OffAgent(Caller):
    """The Stage-4/5 Caller inside the engine, trained on per-possession returns."""

    def get_state(self):
        return [w.copy() for w in self.pi.get_weights()], [w.copy() for w in self.v.get_weights()]

    def set_state(self, st):
        self.pi.set_weights(st[0])
        self.v.set_weights(st[1])

    def update_from(self, eng):
        pts = possession_returns(eng)
        newbuf, ret, off = [], [], 0
        for b in self.buf:
            r = np.array([pts.get((int(e), int(c)), np.nan) for e, c in zip(b['ep'], b['pc'])]) if b.get('pc') is not None else np.full(len(b['a']), np.nan)
            k = ~np.isnan(r)
            if not k.any():
                continue
            nb = {key: (val[k] if isinstance(val, np.ndarray) else val) for key, val in b.items() if key != 'pc'}
            nb['ep'] = np.arange(off, off + int(k.sum()))
            off += int(k.sum())
            ret.append(r[k]); newbuf.append(nb)
        self.buf = newbuf
        if not newbuf:
            return {}
        out = Caller.update(self, np.concatenate(ret))
        out['ret_mean'] = float(np.concatenate(ret).mean())
        return out


def load_stack(model_path='rl/model/executor_v5g.keras'):
    from api import PassValueModel
    from play_cond import PlayTargets
    from shot_audit import AuditedShotValue
    model = keras.saving.load_model(model_path, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    cal = json.load(open(os.path.join(HERE, 'model', 'engine_calibration.json')))
    return dict(model=model, diff=PassValueModel(), pt=PlayTargets(), cal=cal, audit=AuditedShotValue(), lib=DF.SchemeLibrary())


def make_engine(st, seed, off_agents=(None, None), def_agents=(None, None), audit=True, **over):
    kw = dict(st['cal'])
    kw.pop('audit', None)                    # the flag is handled below; Engine takes the table, not a flag
    kw.update(over)
    eng = Engine(st['model'], st['diff'], plays=st['pt'], seed=seed, **kw)
    if audit:
        eng.audit = st['audit']
    eng.off_agent, eng.def_agent = list(off_agents), list(def_agents)
    return eng


def summarize(eng):
    """per-team stats of a finished run: dict of (N,2) arrays plus completed-possession points."""
    out = {k: v.copy() for k, v in eng.stats.items()}
    out['score'] = eng.score.copy()
    return out


# ---------------------------------------------------------------------------------------------------- agents from checkpoints, matches, metrics
def make_agents(st, ckpt=None, learn=False, seed=0, beta_def=0.05, beta_off=0.02, lr=3e-3, rmax_off=None, rmax_def=None):
    """-> (OffAgent, DefAgent), optionally loading weights from a checkpoint dict {'off': state, 'def': state}. A checkpoint may carry its own bounds (rmax_off / rmax_def)."""
    if ckpt is not None:
        rmax_off = ckpt.get('rmax_off', rmax_off)
        rmax_def = ckpt.get('rmax_def', rmax_def)
    o = OffAgent(st['diff'], beta=beta_off, lr=lr, seed=seed + 1, learn=learn, support=st['audit'], rmax=rmax_off)
    d = DefAgent(st['lib'], beta=beta_def, lr=lr, seed=seed, learn=learn, rmax=rmax_def)
    if ckpt is not None:
        if ckpt.get('off') is not None:
            o.set_state(ckpt['off'])
        if ckpt.get('def') is not None:
            d.set_state(ckpt['def'])
    return o, d


def run_match(st, G, table, starts, seed, blocks, A=(None, None), B=(None, None), **over):
    """A, B = (offence agent, defence agent) of roster team 0 / 1 (None = the executor's own behaviour). Returns the finished engine."""
    eng = make_engine(st, seed, off_agents=(A[0], B[0]), def_agents=(A[1], B[1]), **over)
    eng.reset(G, table, starts)
    eng.run(blocks)
    return eng


def match_metrics(eng, game_of_episode=None):
    """possession-level and chain-level summaries of a finished engine."""
    log = np.array(eng.poss_log, dtype=float).reshape(-1, 4)          # episode, count, offence team, points
    n_poss = len(log)
    s = {k: v.sum(0) for k, v in eng.stats.items()}
    poss = max(float(s['poss'].sum()), 1.0)
    out = dict(n_chains=int(eng.N), n_poss=int(n_poss), ppp=float(log[:, 3].mean()) if n_poss else float('nan'),
               ppp_team0_off=float(log[log[:, 2] == 0, 3].mean()) if (log[:, 2] == 0).any() else float('nan'),
               ppp_team1_off=float(log[log[:, 2] == 1, 3].mean()) if (log[:, 2] == 1).any() else float('nan'),
               diff_per_chain=float((eng.score[:, 0] - eng.score[:, 1]).mean()),
               fga_per_poss=float(s['fga'].sum() / poss), fg3a_per_poss=float(s['fg3a'].sum() / poss), tov_per_poss=float(s['tov'].sum() / poss), stl_per_poss=float(s['stl'].sum() / poss),
               fta_per_poss=float(s['fta'].sum() / poss), orb_per_poss=float(s['orb'].sum() / poss), fouled_per_poss=float(s['fouled'].sum() / poss))
    sl = eng.shot_log
    if sl:
        d = np.array([x['dist'] for x in sl])
        out['rim_share'] = float((d < 6).mean())
        out['p_make_mean'] = float(np.mean([x['p_make'] for x in sl]))
        out['def1_mean'] = float(np.mean([x['def1'] for x in sl]))
    return out


def paired_ci(eng_a, eng_b, game_idx, n_boot=400, seed=0):
    """difference in points per completed possession between two runs (a - b), game-clustered bootstrap over the chains' source games."""
    rng = np.random.default_rng(seed)

    def per_chain(eng):
        log = np.array(eng.poss_log, dtype=float).reshape(-1, 4)
        s = np.zeros(eng.N); c = np.zeros(eng.N)
        np.add.at(s, log[:, 0].astype(int), log[:, 3]); np.add.at(c, log[:, 0].astype(int), 1)
        return s, c
    sa, ca = per_chain(eng_a); sb, cb = per_chain(eng_b)
    g = np.asarray(game_idx)
    ug = np.unique(g)
    ba = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug))
        w = np.concatenate([np.flatnonzero(g == u) for u in pick])
        ba.append(sa[w].sum() / ca[w].sum() - sb[w].sum() / cb[w].sum())
    d = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    return float(d), [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]


def formation_stats(eng, every=5):
    """mean of the six scheme knobs (raw units) over front-court live states of the run, and the mean defender / offence speed (ft/s) in those states."""
    T = np.stack(eng.traj)[::every]
    H = np.stack(eng.holder_traj)[::every]
    B = np.stack(eng.ball_traj)[::every]
    live = (H != 0) & (T[:, :, :5, 0].mean(2) < 47.0)
    ti, ci = np.nonzero(live)
    if len(ti) == 0:
        return {}
    kn = DF.knobs(T[ti, ci], B[ti, ci, :2])
    S = np.stack(eng.traj)
    sp = np.hypot(*(S[1:] - S[:-1]).transpose(3, 0, 1, 2)) / 0.2
    Hl = np.stack(eng.holder_traj)[1:]
    lv = (Hl != 0) & (S[1:, :, :5, 0].mean(2) < 47.0)
    return dict(knobs={n: float(v) for n, v in zip(DF.KNOB_NAMES, kn.mean(0))}, def_speed=float(sp[:, :, 5:][lv].mean()), off_speed=float(sp[:, :, :5][lv].mean()))
