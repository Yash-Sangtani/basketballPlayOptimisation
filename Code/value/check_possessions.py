"""
Acceptance check for the possession table over a game list (pbp only, no tracking).

    python check_possessions.py ../splits/dev_games.txt ../../reports/ws5b-possession-check.csv

Needs the game-name -> game-id map (`--map`, default /private/tmp/ws5b/game_map.csv; built
from Tracking_Data/pass_features/meta/*.pkl of the shared dataset) because file names do
not carry the game id.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possessions import build_possessions, box_estimate  # noqa: E402

EVENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Tracking_Data', 'events')


def read_games(path):
    return [l.strip().replace('.7z', '') for l in open(path) if l.strip()]


def load_pbp(game_id):
    return pd.read_csv(os.path.join(EVENTS, f'{int(game_id):010d}.csv'))


def check_game(game, game_id):
    pbp = load_pbp(game_id)
    table, info = build_possessions(pbp, game_id=int(game_id))
    est = box_estimate(pbp)
    away, home = info['away'], info['home']
    n = {t: int((table.offense_team == t).sum()) for t in (away, home)}
    pts = {t: int(table.loc[table.offense_team == t, 'points'].sum()) for t in (away, home)}
    oth = info['other_points']
    final_away, final_home = info['final']
    est_sum = sum(est[t]['est'] for t in (away, home))
    est_ex = sum(est[t]['est_ex_tech'] for t in (away, home))
    row = dict(game=game, game_id=int(game_id), periods=info['periods'], n_poss=len(table),
               n_away=n[away], n_home=n[home], est=est_sum, est_ex_tech=est_ex,
               ratio=len(table) / est_sum, ratio_ex_tech=len(table) / est_ex,
               ratio_away=n[away] / est[away]['est'], ratio_home=n[home] / est[home]['est'],
               poss_diff=abs(n[away] - n[home]),
               pts_away=pts[away], pts_home=pts[home], other_away=oth[away], other_home=oth[home],
               final_away=final_away, final_home=final_home,
               strict_ok=(pts[away] == final_away and pts[home] == final_home),
               with_other_ok=(pts[away] + oth[away] == final_away and pts[home] + oth[home] == final_home),
               fga=sum(est[t]['fga'] for t in est), fta=sum(est[t]['fta'] for t in est),
               oreb=sum(est[t]['oreb'] for t in est), tov=sum(est[t]['tov'] for t in est),
               oreb_state=int(table.oreb.sum()))
    row.update({f'diag_{k}': v for k, v in info['diag'].items()})
    reasons = table.end_reason.value_counts().to_dict()
    row.update({f'end_{k}': v for k, v in reasons.items()})
    return row, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games')
    ap.add_argument('out')
    ap.add_argument('--map', default='/private/tmp/ws5b/game_map.csv')
    ap.add_argument('--save-tables', default=None, help='optional path for all possession tables (csv.gz)')
    a = ap.parse_args()
    gm = pd.read_csv(a.map).set_index('game').game_id
    games = read_games(a.games)
    t0 = time.perf_counter()
    rows, tables, errors = [], [], []
    for g in games:
        if g not in gm.index:
            errors.append((g, 'no game id')); continue
        try:
            r, t = check_game(g, gm[g])
            rows.append(r); tables.append(t)
        except Exception as ex:   # report, never hide
            errors.append((g, f'{type(ex).__name__}: {ex}'))
    df = pd.DataFrame(rows).fillna({c: 0 for c in []})
    end_cols = [c for c in df.columns if c.startswith('end_')]
    df[end_cols] = df[end_cols].fillna(0).astype(int)
    df.to_csv(a.out, index=False)
    if a.save_tables:
        pd.concat(tables).to_csv(a.save_tables, index=False)
    print(f'{len(df)} games ok, {len(errors)} errors, {time.perf_counter() - t0:.1f}s')
    for g, m in errors:
        print('ERROR', g, m)
    q = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    for col in ('ratio', 'ratio_ex_tech', 'ratio_away', 'ratio_home'):
        print(col, 'quantiles', dict(zip(q, np.round(df[col].quantile(q).values, 4))),
              'share within +-5%:', round(float(((df[col] - 1).abs() <= .05).mean()), 4),
              'within +-3%:', round(float(((df[col] - 1).abs() <= .03).mean()), 4))
    print('mean n_poss', df.n_poss.mean(), 'mean est', df.est.mean())
    print('score: strict (possession points only) ok', int(df.strict_ok.sum()), '/', len(df),
          '; possession + other points ok', int(df.with_other_ok.sum()), '/', len(df))
    print('other points per game: mean', (df.other_away + df.other_home).mean(), 'max', (df.other_away + df.other_home).max())
    bad = df[~df.with_other_ok]
    if len(bad):
        print('score exceptions:'); print(bad[['game', 'pts_away', 'other_away', 'final_away', 'pts_home', 'other_home', 'final_home']].to_string())
    print('|home - away| possessions: distribution', df.poss_diff.value_counts().sort_index().to_dict())
    print('games with |diff| > periods:', int((df.poss_diff > df.periods).sum()),
          '; > periods+1:', int((df.poss_diff > df.periods + 1).sum()))
    dcols = [c for c in df.columns if c.startswith('diag_')]
    print('diagnostics (sum over games / games with >=1):')
    for c in dcols:
        print(' ', c, int(df[c].sum()), int((df[c] > 0).sum()))
    print('OREB: independent estimate vs state machine, mean', df.oreb.mean(), df.oreb_state.mean(),
          'games differing', int((df.oreb != df.oreb_state).sum()))
    print('end reasons (total):', df[end_cols].sum().to_dict())


if __name__ == '__main__':
    main()
