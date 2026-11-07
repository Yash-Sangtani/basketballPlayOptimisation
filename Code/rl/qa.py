"""Dataset QA over built games: what the Stage 0 spec (reports/rl-spec.md) quotes.

    cd Code && python rl/qa.py --data ../data/rl [--out ../reports/rl-qa.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from load import load_game             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.data) if f.endswith('.npz'))
    rows, players = [], {}
    for f in files:
        d = load_game(os.path.join(a.data, f))
        r = d['report']
        pid = d['poss_id']
        g = pd.DataFrame({'p': pid, 't': d['t_in_poss'], 'front': d['frontcourt'], 'tf': d['t_front']}).groupby('p').agg(dur=('t', 'max'), tf=('tf', 'max'))
        sh = d['shots']
        sp = np.hypot(d['vel'][..., 0], d['vel'][..., 1])
        hc = d['live'] & d['frontcourt'] & (d['t_front'] >= 1)
        rows.append(dict(game=r['game'], states=r['n_states'], dropped=r['n_dropped'], poss=r['n_poss'], poss_ge4=int((g.dur >= 4).sum()),
                         poss_halfcourt=int((g.tf >= 4).sum()), live=float(d['live'].mean()), halfcourt_states=float(hc.mean()),
                         passes=r['n_passes'], shots=r['n_shots'], fg_pts=int((sh.made * np.where(sh.is3 == 1, 3, 2)).sum()) if len(sh) else 0,
                         mean_speed_off=float(sp[:, :5].mean()), mean_speed_def=float(sp[:, 5:].mean()), seconds=r['seconds']))
        for pl in np.unique(d['ids']):
            players[int(pl)] = players.get(int(pl), 0) + int((d['ids'] == pl).any(axis=1).sum())
    R = pd.DataFrame(rows)
    pc = np.array(sorted(players.values()))
    out = dict(n_games=len(R), mean=R.drop(columns='game').mean().round(3).to_dict(), total_states=int(R.states.sum()),
               n_players=len(players), player_states_pctl={str(q): int(np.percentile(pc, q)) for q in (10, 25, 50, 75, 90)},
               players_over_5000_states=int((pc >= 5000).sum()))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
