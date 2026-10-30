"""Read the per-game files written by dataset.py."""
import io
import json
import os

import numpy as np
import pandas as pd


def load_game(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if not k.endswith('_json')}
    for k in ('passes', 'shots', 'roster'):
        d[k] = pd.read_json(io.StringIO(str(z[k + '_json'])), orient='split')
    d['report'] = json.loads(str(z['report_json']))
    d['game'] = d['report']['game']
    return d


def load_games(data_dir, games):
    out = []
    for g in games:
        p = os.path.join(data_dir, f'{g}.npz')
        if os.path.exists(p):
            out.append(load_game(p))
    return out
