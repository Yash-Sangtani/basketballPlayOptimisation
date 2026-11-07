"""WS4: does any small change to the detector's thresholds improve assist recall without hurting the precision proxies?

DEV games only (a stride sample of splits/dev_games.txt; the test list is never read). One variant = overrides of
PassTracking class constants. For each variant and game it reports what validate_passes.py reports: assist recall,
completed/failed counts, matched pbp bad passes, the minority-team share and repeated-flip windows, and the precision
proxies. Nothing here edits passtracking.py: the constants are overridden on a subclass.

    cd Code && ../.venv/bin/python analysis/ws4_detect_sweep.py [--n 24] [--out reports/ws4-sweep.json]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
import passtracking as P                                      # noqa: E402
from validate_passes import assist_recall, mixed_team_share, precision_proxies, parse_game   # noqa: E402

VARIANTS = {
    'baseline': {},
    'hold_z_10': dict(HOLD_Z=10.0),
    'min_disp_3': dict(MIN_DISP=3.0),
    'min_hold_3': dict(MIN_HOLD_FRAMES=3),
    'hold_dist_3.5': dict(HOLD_DIST=3.5),
    'hold_z_10+min_disp_3': dict(HOLD_Z=10.0, MIN_DISP=3.0),
    'handoffs': dict(MIN_FLIGHT_S=0.08, MIN_DISP=2.0),
    'handoffs+hold_z_10': dict(MIN_FLIGHT_S=0.08, MIN_DISP=2.0, HOLD_Z=10.0),
    'all_four': dict(HOLD_Z=10.0, MIN_DISP=3.0, MIN_HOLD_FRAMES=3, HOLD_DIST=3.5),
}
SHAPING = {'11.20.2015.SAS.at.NOP', '12.12.2015.GSW.at.MIL', '01.12.2016.HOU.at.MEM', '12.30.2015.PHI.at.SAC'}


def games(n):
    lst = [l.strip().replace('.7z', '') for l in open(os.path.join(HERE, '..', 'splits', 'dev_games.txt')) if l.strip()]
    lst = [g for g in lst if g not in SHAPING]
    step = max(1, len(lst) // n)
    return lst[::step][:n]


def run(game, over):
    date, home, away = parse_game(game)
    cls = type('V', (P.PassTracking,), over)
    pt = cls(date, home, away)
    pt.track_passes()
    pa = pt.pass_attempt
    n_ast, hit, miss = assist_recall(pt)
    minor, both, one, many = mixed_team_share(pa)
    pp = precision_proxies(pt)
    fail = pa[pa['pass_successful'] == 0]
    return dict(n_comp=int(pa['pass_successful'].sum()), n_fail=len(fail), n_ast=n_ast, hit=hit,
                miss_assister=miss['assister_not_holding'], miss_scorer=miss['scorer_not_holding'], miss_other=miss['other'],
                minority=minor, mixed_one=one, mixed_many=many, **{k: v for k, v in pp.items() if k != 'n'},
                matched=int(pt.failure_report.get('matched', len(fail))) if isinstance(pt.failure_report, dict) else len(fail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--out', default=None)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    a = ap.parse_args()
    gs = games(a.n)
    print(f'{len(gs)} dev games, code version {P.CODE_VERSION}', flush=True)
    res = {}
    for v in a.variants.split(','):
        t0 = time.time()
        rows = [run(g, VARIANTS[v]) for g in gs]
        df = pd.DataFrame(rows)
        s = df.sum(numeric_only=True)
        w = df['n_comp']
        out = dict(n_comp=int(s['n_comp']), n_fail=int(s['n_fail']), assists=int(s['n_ast']), hit=int(s['hit']),
                   recall=float(s['hit'] / s['n_ast']), miss_assister=int(s['miss_assister']), miss_scorer=int(s['miss_scorer']),
                   miss_other=int(s['miss_other']), minority_share=float((df['minority'] * w).sum() / w.sum()),
                   windows_repeated_flips=int(s['mixed_many']), windows_one_switch=int(s['mixed_one']),
                   **{k: float((df[k] * w).sum() / w.sum()) for k in ('recv_ball_z_gt_9', 'flight_lt_0_4s',
                                                                      'next_release_within_0_5s', 'passer_target_lt_6ft')})
        res[v] = out
        print(f'[{time.time() - t0:.0f}s] {v:24s} comp {out["n_comp"]} fail {out["n_fail"]} | assist recall {out["hit"]}/{out["assists"]} = '
              f'{out["recall"]:.3f} (miss: assister {out["miss_assister"]}, scorer {out["miss_scorer"]}, other {out["miss_other"]}) | '
              f'minority {out["minority_share"]:.3f} flips {out["windows_repeated_flips"]} | z>9 {out["recv_ball_z_gt_9"]:.3f} '
              f'fl<0.4s {out["flight_lt_0_4s"]:.3f} next<0.5s {out["next_release_within_0_5s"]:.3f} <6ft {out["passer_target_lt_6ft"]:.3f}',
              flush=True)
    if a.out:
        json.dump(dict(games=gs, variants=res), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
