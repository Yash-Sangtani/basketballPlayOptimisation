"""Tiny synthetic data in the same layout as the real per-game pickles."""
import os

import numpy as np
import pandas as pd

import evaluation as ev


def game_names(n, prefix='01'):
    return [f"{prefix}.{i + 1:02d}.2016.AAA.at.B{i:02d}" for i in range(n)]


def make_game_frame(game, rng, n_rows=120, base_rate=0.06):
    """One game's (features, meta). Failures are more likely with a short shot clock and a long pass."""
    X = pd.DataFrame({c: rng.normal(size=n_rows) for c in ev.FEATURES})
    X['shot_clock'] = rng.uniform(0, 24, n_rows)
    X['dist'] = rng.uniform(5, 40, n_rows)
    X['backcourt'] = (rng.random(n_rows) < 0.3).astype(float)
    X['lane_n_def_5ft'] = rng.integers(0, 4, n_rows).astype(float)
    X.loc[rng.random(n_rows) < 0.05, 'passer_speed'] = np.nan
    logit = np.log(base_rate / (1 - base_rate)) + 0.06 * (X['dist'] - 22) - 0.08 * (X['shot_clock'] - 12)
    fail = rng.random(n_rows) < 1 / (1 + np.exp(-logit))
    X[ev.LABEL] = (~fail).astype(float)
    meta = pd.DataFrame({'game_id': 1, 'quarter': 1, 'game_time': 1.0, 'frame': np.arange(n_rows), 'passer': 1,
                         'target': 2.0, 'team_passer': 3, 'target_inferred': False, 'target_mode': 'actual',
                         'outcome': np.where(fail, 'intercepted', 'completed'), 'game': game}, index=X.index)
    return X, meta


def make_arrays(n_games=30, n_rows=150, seed=0):
    """(X, y, g, features) in the layout evaluation.py uses."""
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for game in game_names(n_games):
        df, _ = make_game_frame(game, rng, n_rows)
        Xs.append(df[ev.FEATURES].to_numpy(float)); ys.append((df[ev.LABEL] == 0).astype(np.int8).to_numpy())
        gs.append(np.full(n_rows, game))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), list(ev.FEATURES)


def write_pickle_dir(path, games, seed=0, n_rows=120, status='ok', version='abc123'):
    """Write features, meta/ and _status.csv for `games` under `path`."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(path, 'meta'), exist_ok=True)
    rows = []
    for game in games:
        df, meta = make_game_frame(game, rng, n_rows)
        fn = ev.game_to_pickle_name(game)
        df.to_pickle(os.path.join(path, fn)); meta.to_pickle(os.path.join(path, 'meta', fn))
        rows.append(f'{game},{status},{len(df)},1.0,{version},""')
    with open(os.path.join(path, '_status.csv'), 'w') as f:
        f.write('game,status,rows,seconds,version,error\n' + '\n'.join(rows) + '\n')
