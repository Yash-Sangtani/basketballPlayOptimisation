"""Stage 6B: defensive intent. Man assignment, scheme knobs, scheme library.

Everything works in the attack-normalised frame (offence = slots 0-4 attack the basket at (5.35, 25); defence = slots 5-9).

  assign(pos)      man-to-man matching: which attacker each defender is guarding (min total distance, stateless, vectorised over the 120 permutations)
  knobs(pos, ball) six numbers that describe how a defence is playing at one instant (z-scored with SCALE below):
                     gap     mean distance of a defender to the attacker he is matched to (ft)         tight vs loose
                     sag     mean (attacker's distance to the basket - defender's distance to it) (ft)  positive = defender between man and basket
                     press   distance of the ball handler's defender to the ball handler (ft)           ball pressure
                     paint   number of defenders within 10 ft of the basket                             paint protection
                     help    distance of the nearest off-ball defender to the ball (ft)                 help proximity
                     depth   mean distance of the five defenders to the basket (ft)                     how far back the line sits
  A scheme is a cluster of the window-averaged knob vector (k-means, library K); its centroid is what the executor is told (hindsight, as for plays).
"""
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASKET = np.array([5.35, 25.0])
KNOB_NAMES = ['gap', 'sag', 'press', 'paint', 'help', 'depth']
N_KNOB = 6
DEF_TOK = 4 + 2 * N_KNOB                                           # per-token extras for the v5g executor: man features (4) + told scheme (6) + what is still to change (told - current, 6)
PERMS = np.array(list(itertools.permutations(range(5))))          # (120,5): defender j guards attacker PERMS[p, j]
LIB_PATH = os.path.join(HERE, 'model', 'scheme_library.npz')


def assign(pos):
    """pos (N,10,2) -> man (N,5) int: attacker slot (0-4) guarded by defender j (slot 5+j)."""
    off, dfn = pos[:, :5], pos[:, 5:]
    D = np.hypot(dfn[:, :, None, 0] - off[:, None, :, 0], dfn[:, :, None, 1] - off[:, None, :, 1])     # (N,def,att)
    cost = D[:, np.arange(5)[None, :], PERMS].sum(-1)                                                 # (N,120)
    return PERMS[cost.argmin(1)]


def ball_man(pos, ball_xy):
    """attacker slot nearest to the ball."""
    return np.hypot(*(pos[:, :5] - ball_xy[:, None, :]).transpose(2, 0, 1)).argmin(1)


def knobs(pos, ball_xy, man=None):
    """-> (N,6) raw knob values (ft / counts)."""
    if man is None:
        man = assign(pos)
    N = len(pos)
    off, dfn = pos[:, :5], pos[:, 5:]
    mp = np.take_along_axis(off, man[:, :, None], axis=1)                       # attacker guarded by each defender (N,5,2)
    gap = np.hypot(*(dfn - mp).transpose(2, 0, 1))                              # (N,5)
    dm = np.hypot(*(mp - BASKET).transpose(2, 0, 1))
    dd = np.hypot(*(dfn - BASKET).transpose(2, 0, 1))
    sag = (dm - dd).mean(1)
    bm = ball_man(pos, ball_xy)
    is_bm = man == bm[:, None]                                                  # defender guarding the ball man
    press = np.where(is_bm, np.hypot(*(dfn - ball_xy[:, None, :]).transpose(2, 0, 1)), 0).sum(1)
    paint = (dd <= 10.0).sum(1)
    dball = np.hypot(*(dfn - ball_xy[:, None, :]).transpose(2, 0, 1))
    help_ = np.where(is_bm, 99.0, dball).min(1)
    return np.stack([gap.mean(1), sag, press, paint, help_, dd.mean(1)], axis=1).astype(np.float32)


def man_features(pos, ball_xy, man=None):
    """per-defender token extras (N,5,4): vector to the assigned attacker (/20), its length (/20), and whether he is the ball handler."""
    if man is None:
        man = assign(pos)
    mp = np.take_along_axis(pos[:, :5], man[:, :, None], axis=1)
    v = mp - pos[:, 5:]
    bm = ball_man(pos, ball_xy)
    return np.concatenate([v / 20.0, (np.hypot(v[..., 0], v[..., 1]) / 20.0)[..., None], (man == bm[:, None])[..., None].astype(np.float32)], axis=-1).astype(np.float32)


class SchemeLibrary:
    """K cluster centroids in z-scored knob space."""

    def __init__(self, path=LIB_PATH):
        z = np.load(path)
        self.mu, self.sd, self.C, self.freq = z['mu'], z['sd'], z['C'], z['freq']
        self.K = len(self.C)

    def z(self, raw):
        return ((raw - self.mu) / self.sd).astype(np.float32)

    def nearest(self, raw):
        d = ((self.z(raw)[:, None, :] - self.C[None]) ** 2).sum(-1)
        return d.argmin(1)

    def goal(self, k):
        """centroid(s) of scheme(s) k -> (len(k),6) float32, the vector the executor is told."""
        return self.C[np.asarray(k)].astype(np.float32)


# ---------------------------------------------------------------------------------------------------- features of the scheme caller (shared by the behaviour-cloned prior and the agent)
D_AGENT = N_KNOB + 2 + 3 + 8          # knobs (z), ball position (2), shot clock, front-court time, offence players within 15 ft of the basket, previous scheme (one-hot over K = 8)


def agent_features(lib, pos, ball_xy, shot_clock, t_front, prev):
    """pos (N,10,2) in the offence frame, ball_xy (N,2), shot_clock (N,), t_front (N,), prev (N,) int (-1 = no previous scheme) -> (N, D_AGENT) float32."""
    z = lib.z(knobs(pos, ball_xy))
    bxy = np.stack([ball_xy[:, 0] / 47.0 - 1.0, ball_xy[:, 1] / 25.0 - 1.0], 1)
    sc = np.clip(np.nan_to_num(shot_clock, nan=24.0), 0, 24) / 24.0
    tf_ = np.clip(t_front, 0, 24) / 24.0
    near = (np.hypot(*(pos[:, :5] - BASKET).transpose(2, 0, 1)) <= 15.0).sum(1) / 5.0
    oh = np.zeros((len(pos), lib.K), np.float32)
    m = prev >= 0
    oh[np.flatnonzero(m), prev[m]] = 1.0
    return np.concatenate([z, bxy, sc[:, None], tf_[:, None], near[:, None], oh], axis=1).astype(np.float32)
