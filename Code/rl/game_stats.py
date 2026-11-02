"""Stage 6A.1: real per-game statistics and event tables from the play-by-play, for the 401 train and 100 holdout games.

    cd Code && TRACKING_DATA_DIR=<raw data> python rl/game_stats.py --out ../reports/rl-stage6a

Outputs (private reports dir): `real_game_stats.csv` (one row per game and team), `events_shots.csv`, `events_ft.csv`, `events_reb.csv`, `events_tov.csv`.
Possessions per team = FGA + 0.44 FTA - ORB + TOV (the usual box-score estimate). Clock in seconds elapsed in the period.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from splits import games as split_games  # noqa: E402

RAW = os.environ.get('TRACKING_DATA_DIR', '/home/user/basketballPlayOptimisation/Tracking_Data')


def clock_s(t):
    m, s = t.split(':')
    return int(m) * 60 + int(s)


def desc(r):
    return ' '.join(str(x) for x in (r.HOMEDESCRIPTION, r.NEUTRALDESCRIPTION, r.VISITORDESCRIPTION) if isinstance(x, str))


def parse_game(game_id, split):
    df = pd.read_csv(os.path.join(RAW, 'events', f'{int(game_id):010d}.csv'))
    shots, fts, rebs, tovs, fouls = [], [], [], [], []
    last_shot = None
    for r in df.itertuples():
        t = int(r.EVENTMSGTYPE)
        d = desc(r)
        per = int(r.PERIOD)
        el = (per - 1) * 720 + (720 - clock_s(r.PCTIMESTRING)) if per <= 4 else 2880 + (per - 5) * 300 + (300 - clock_s(r.PCTIMESTRING))
        team = r.PLAYER1_TEAM_ID
        if t in (1, 2):
            m = re.search(r"(\d+)'", d)
            dist = float(m.group(1)) if m else 0.0
            is3 = '3PT' in d
            shots.append(dict(game=game_id, split=split, period=per, t=el, team=team, player=r.PLAYER1_ID, made=int(t == 1), is3=int(is3), dist=dist,
                              blocked=int(t == 2 and 'BLOCK' in d), assisted=int(t == 1 and 'AST' in d), pts=(3 if is3 else 2) * int(t == 1)))
            last_shot = (el, team, int(t == 2), is3, dist)
        elif t == 3:
            m = re.search(r'Free Throw (\d) of (\d)', d) or re.search(r'Free Throw Technical|Free Throw Flagrant', d)
            k, n = (int(m.group(1)), int(m.group(2))) if m and m.lastindex else (1, 1)
            fts.append(dict(game=game_id, split=split, period=per, t=el, team=team, player=r.PLAYER1_ID, made=int('MISS' not in d), k=k, n=n))
        elif t == 4:
            off = None
            if last_shot is not None and last_shot[2] == 1 and el - last_shot[0] <= 8:
                off = int(team == last_shot[1])
            rebs.append(dict(game=game_id, split=split, period=per, t=el, team=team, player=r.PLAYER1_ID, off=off, dist=last_shot[4] if last_shot else np.nan,
                             is3=last_shot[3] if last_shot else np.nan))
        elif t == 5:
            tovs.append(dict(game=game_id, split=split, period=per, t=el, team=team, player=r.PLAYER1_ID, stealer=r.PLAYER2_ID if int(r.PERSON2TYPE) in (4, 5) and r.PLAYER2_ID else 0,
                             action=int(r.EVENTMSGACTIONTYPE), text=d[:60]))
        elif t == 6:
            fouls.append(dict(game=game_id, split=split, period=per, t=el, team=team, action=int(r.EVENTMSGACTIONTYPE)))
    return shots, fts, rebs, tovs, fouls, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='../data/rl')
    a = ap.parse_args()
    import glob
    S, F, R, T, FO, rows = [], [], [], [], [], []
    for split in ('train', 'holdout'):
        for g in split_games(split):
            p = os.path.join(a.data, g + '.npz')
            if not os.path.exists(p):
                continue
            rep = json.loads(str(np.load(p, allow_pickle=True)['report_json']))
            s, f, r, t, fo, df = parse_game(rep['game_id'], split)
            S += s; F += f; R += r; T += t; FO += fo
            sh, ft, rb, tv = pd.DataFrame(s), pd.DataFrame(f), pd.DataFrame(r), pd.DataFrame(t)
            for team in sh['team'].unique():
                a_ = sh[sh.team == team]
                fa = ft[ft.team == team] if len(ft) else ft
                rb_t = rb[rb.team == team] if len(rb) else rb
                tv_t = tv[tv.team == team] if len(tv) else tv
                orb = int((rb_t['off'] == 1).sum()) if len(rb_t) else 0
                drb = int((rb_t['off'] == 0).sum()) if len(rb_t) else 0
                fta = len(fa)
                tov = len(tv_t)
                rows.append(dict(game=rep['game'], game_id=rep['game_id'], split=split, team=team, fga=len(a_), fgm=int(a_.made.sum()), fg3a=int(a_.is3.sum()), fg3m=int(a_[a_.is3 == 1].made.sum()),
                                 fta=fta, ftm=int(fa.made.sum()) if fta else 0, orb=orb, drb=drb, tov=tov, stl_by_opp=0,
                                 pts=int(a_.pts.sum() + (fa.made.sum() if fta else 0)), poss=len(a_) + 0.44 * fta - orb + tov,
                                 ast=int(a_.assisted.sum()), dist_mean=float(a_.dist.mean())))
            tvp = tv.copy()
            for team in sh['team'].unique():
                rows_t = [x for x in rows[-2:] if x['team'] == team]
                if rows_t:
                    rows_t[0]['stl_by_opp'] = int(((tvp.team != team) & (tvp.stealer > 0)).sum()) if len(tvp) else 0
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'real_game_stats.csv'), index=False)
    pd.DataFrame(S).to_csv(os.path.join(a.out, 'events_shots.csv'), index=False)
    pd.DataFrame(F).to_csv(os.path.join(a.out, 'events_ft.csv'), index=False)
    pd.DataFrame(R).to_csv(os.path.join(a.out, 'events_reb.csv'), index=False)
    pd.DataFrame(T).to_csv(os.path.join(a.out, 'events_tov.csv'), index=False)
    pd.DataFrame(FO).to_csv(os.path.join(a.out, 'events_foul.csv'), index=False)
    g = pd.DataFrame(rows)
    print(len(g), 'team-games; mean per team-game:')
    print(g[['fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'orb', 'drb', 'tov', 'pts', 'poss']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
