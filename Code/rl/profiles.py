"""Player profiles for the behaviour models: 12 low-dimensional style statistics per player, shrunk toward the player's position group.

Stage 1 showed that a raw player id does not help but short style statistics do, and that a player's own shot-location mix is the
strongest signal. So a profile is a short vector, estimated ONLY from the games it is told to use (the fitting games), never from
the games a model is evaluated on:

  offence (half-court live states)  off_speed, off_basket_dist, off_abs_y (distance from the court's long axis), off_usage (share of
                                    his states holding the ball), off_shoot / off_pass (shots / passes per holder-state)
  shot locations                    shot_rim, shot_short, shot_mid, shot_three (shares of his matched shots by distance from the basket)
  defence                           def_speed, def_tight (mean distance to the nearest offensive player)

Each statistic is  w * own + (1 - w) * position_group_mean,  w = n / (n + K) with K pseudo-observations (a bench player with little
data falls back to his position group). Profiles are z-scored with the fitting players' means and spreads. Unknown players get zeros.
"""
import numpy as np
import pandas as pd

from features import BASKET, roster_positions

PROFILE_COLS = ['off_speed', 'off_basket_dist', 'off_abs_y', 'off_usage', 'off_shoot', 'off_pass',
                'shot_rim', 'shot_short', 'shot_mid', 'shot_three', 'def_speed', 'def_tight']
K_STATES = 2000.0        # pseudo-states for the per-state statistics (a starter has 100k+)
K_SHOTS = 50.0           # pseudo-shots for the shot-location mix
BINS = [0, 5, 14, 22, 100]


def half_court_mask(d):
    return d['live'] & d['frontcourt'] & (d['t_front'] >= 1.0) & (d['holder'] != 0)


def build_profiles(games, shrink=True):
    """games: list of dataset.py game dicts (the FITTING games). -> DataFrame indexed by player id, columns PROFILE_COLS (z-scored),
    plus 'pos' (position code) and 'n_states'."""
    acc = {}
    pos_of = {}
    shot_rows = []
    for d in games:
        pos_of.update(roster_positions(d['roster']))
        hc = half_court_mask(d)
        if not hc.any():
            continue
        P, V, ids, holder = d['pos'][hc], d['vel'][hc], d['ids'][hc], d['holder'][hc]
        N = len(P)
        spd = np.hypot(V[..., 0], V[..., 1])
        bd = np.hypot(P[..., 0] - BASKET[0], P[..., 1] - BASKET[1])
        ay = np.abs(P[..., 1] - 25.0)
        off, dfn = P[:, :5], P[:, 5:]
        dd = np.hypot(off[:, :, None, 0] - dfn[:, None, :, 0], off[:, :, None, 1] - dfn[:, None, :, 1])   # (N,5,5)
        tight_def = dd.min(axis=1)                                   # for each defender, nearest offensive player
        for j in range(10):
            pl = ids[:, j]
            if j < 5:
                cols = dict(off_n=np.ones(N), off_speed=spd[:, j], off_basket_dist=bd[:, j], off_abs_y=ay[:, j],
                            off_hold=(pl == holder).astype(float))
            else:
                cols = dict(def_n=np.ones(N), def_speed=spd[:, j], def_tight=tight_def[:, j - 5])
            for k, v in cols.items():
                s = pd.Series(v).groupby(pl).sum()
                for p, x in s.items():
                    acc.setdefault(int(p), {}).setdefault(k, 0.0)
                    acc[int(p)][k] += float(x)
        pas = d['passes']
        shots = d['shots']
        for pl, g in (pas[pas['live'] == 1].groupby('passer') if len(pas) else []):
            acc.setdefault(int(pl), {}).setdefault('n_pass', 0.0)
            acc[int(pl)]['n_pass'] += len(g)
        fr = d['frame']
        for s in shots.itertuples():
            k = int(np.searchsorted(fr, s.frame))
            cand = [c for c in (k - 1, k) if 0 <= c < len(fr) and abs(int(fr[c]) - int(s.frame)) <= 2]
            if not cand:
                continue
            c = cand[0]
            j = np.flatnonzero(d['ids'][c, :5] == s.shooter)
            if len(j) and d['ctrl_team'][c] == s.team:
                shot_rows.append((int(s.shooter), float(np.hypot(*(d['pos'][c, j[0]] - BASKET)))))
        for s in (shots.itertuples() if len(shots) else []):
            acc.setdefault(int(s.shooter), {}).setdefault('n_shot', 0.0)
            acc[int(s.shooter)]['n_shot'] += 1
    rows = []
    for p, a in acc.items():
        on, dn = a.get('off_n', 0.0), a.get('def_n', 0.0)
        rows.append(dict(player=p, pos=pos_of.get(p, 2), off_n=on, def_n=dn,
                         off_speed=a.get('off_speed', 0) / on if on else np.nan, off_basket_dist=a.get('off_basket_dist', 0) / on if on else np.nan,
                         off_abs_y=a.get('off_abs_y', 0) / on if on else np.nan, off_usage=a.get('off_hold', 0) / on if on else np.nan,
                         off_shoot=a.get('n_shot', 0) / max(a.get('off_hold', 0), 1.0) if on else np.nan,
                         off_pass=a.get('n_pass', 0) / max(a.get('off_hold', 0), 1.0) if on else np.nan,
                         def_speed=a.get('def_speed', 0) / dn if dn else np.nan, def_tight=a.get('def_tight', 0) / dn if dn else np.nan))
    T = pd.DataFrame(rows).set_index('player')
    S = pd.DataFrame(shot_rows, columns=['player', 'dist'])
    S['cat'] = pd.cut(S['dist'], BINS, labels=False, right=False)
    cnt = S.groupby(['player', 'cat']).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    for c in range(4):
        T[f'shot_n{c}'] = cnt[c].reindex(T.index).fillna(0)
    T['shot_n'] = T[[f'shot_n{c}' for c in range(4)]].sum(axis=1)
    for c, name in enumerate(('shot_rim', 'shot_short', 'shot_mid', 'shot_three')):
        T[name] = T[f'shot_n{c}'] / T['shot_n'].where(T['shot_n'] > 0)
    out = pd.DataFrame(index=T.index)
    out['pos'] = T['pos'].astype(int)
    out['n_states'] = (T['off_n'].fillna(0) + T['def_n'].fillna(0)).astype(int)
    weights = {'off_speed': 'off_n', 'off_basket_dist': 'off_n', 'off_abs_y': 'off_n', 'off_usage': 'off_n', 'off_shoot': 'off_n', 'off_pass': 'off_n',
               'shot_rim': 'shot_n', 'shot_short': 'shot_n', 'shot_mid': 'shot_n', 'shot_three': 'shot_n', 'def_speed': 'def_n', 'def_tight': 'def_n'}
    for c in PROFILE_COLS:
        nn = T[weights[c]].fillna(0)
        K = K_SHOTS if c.startswith('shot_') else K_STATES
        w = nn / (nn + K) if shrink else (nn > 0).astype(float)
        wsum = T[c].fillna(0)
        # position-group mean weighted by evidence
        g = (T[c].fillna(0) * nn).groupby(T['pos']).transform('sum') / nn.groupby(T['pos']).transform('sum').replace(0, np.nan)
        g = g.fillna(T[c].mean())
        out[c] = w * T[c].fillna(g) + (1 - w) * g
    mu, sd = out[PROFILE_COLS].mean(), out[PROFILE_COLS].std().replace(0, 1.0)
    out[PROFILE_COLS] = (out[PROFILE_COLS] - mu) / sd
    out.attrs['mu'], out.attrs['sd'] = mu, sd
    return out


def profile_lookup(table, ids):
    """ids: int array of any shape -> float32 array ids.shape + (12,); unknown players get zeros (the average profile)."""
    flat = ids.reshape(-1)
    idx = table.index.get_indexer(flat)
    arr = table[PROFILE_COLS].to_numpy(np.float32)
    out = np.zeros((len(flat), len(PROFILE_COLS)), np.float32)
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out.reshape(ids.shape + (len(PROFILE_COLS),))
