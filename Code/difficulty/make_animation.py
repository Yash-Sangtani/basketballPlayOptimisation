"""Render a short real-game clip with the pass-difficulty model drawn on top (GIF, no ffmpeg needed).

Thin lines: from the ball holder to each teammate, coloured by the model's difficulty for that candidate pass
(green = safe, red = likely to fail) and labelled with p_complete. Thick line: the pass actually thrown, with the
model's difficulty at the release instant and, at the end of the flight, what happened.

    cd Code && python difficulty/make_animation.py --out ../docs/pass_difficulty.gif [--game 11.20.2015.SAS.at.NOP]

Uses a TUNING game by default (never the held-out games). Positions are the raw SportVU coordinates (94 x 50 ft).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
os.environ.setdefault('TQDM_DISABLE', '1')

import matplotlib                       # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib import animation        # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle   # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import extract as X                     # noqa: E402
import train_eval as T                  # noqa: E402
from api import PassDifficultyModel     # noqa: E402
from pass_features import FEATURE_NAMES, pass_features   # noqa: E402

CMAP = plt.get_cmap('RdYlGn_r')
STEP = 2                 # draw every 2nd frame (12.5 fps)
WINDOW = 300             # frames (12 s)


def court(ax, xlim=(-2, 96)):
    ax.set_xlim(*xlim)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 94, 50, fill=False, lw=1.2, color='#555'))
    ax.plot([47, 47], [0, 50], color='#555', lw=1)
    ax.add_patch(Circle((47, 25), 6, fill=False, lw=1, color='#555'))
    for side, sgn in ((5.35, 1), (88.65, -1)):
        x0 = 0 if sgn == 1 else 94
        ax.add_patch(Rectangle((x0 if sgn == 1 else x0 - 19, 17), 19, 16, fill=False, lw=1, color='#555'))
        ax.add_patch(Circle((side, 25), 0.75, fill=False, lw=1.5, color='#d9534f'))
        ax.add_patch(Arc((side, 25), 47.5, 47.5, theta1=-68 if sgn == 1 else 112, theta2=68 if sgn == 1 else 248, lw=1, color='#555'))


def pick_window(pt, P, model):
    """A 12 s stretch of live play with >= 3 passes, at least one failure and at least one pass with difficulty > 15%."""
    fr = pt._fr
    P = P[(P.live == 1)].sort_values('frame')
    best = None
    for f0 in P.frame.values[::3]:
        w = P[(P.frame >= f0) & (P.frame < f0 + WINDOW)]
        if len(w) < 3 or (w.label == 0).sum() < 1:
            continue
        # one continuous block of tracking, no gaps
        if fr['block'][f0] != fr['block'][min(f0 + WINDOW, fr['n'] - 1)]:
            continue
        if not (fr['has_ball'][f0:f0 + WINDOW]).all():
            continue
        score = len(w) + 3 * (w.label == 0).sum()
        if best is None or score > best[0]:
            best = (score, int(f0))
    return None if best is None else best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--game', default=None)
    ap.add_argument('--start', type=int, default=None)
    a = ap.parse_args()
    model = PassDifficultyModel()
    tune, _ = T.split_games()
    games = [a.game] if a.game else tune[3::40]
    for g in games:
        date, away, home = X.split(g)
        pt = X.DiffPT(date, home, away)
        pt.build()
        f0 = a.start if a.start is not None else pick_window(pt, pt.pass_rows, model)
        if f0 is not None:
            break
    else:
        raise SystemExit('no suitable window found')
    fr, P = pt._fr, pt.pass_rows
    print(f'game {g}, frames {f0}..{f0 + WINDOW}, passes in window: {len(P[(P.frame >= f0) & (P.frame < f0 + WINDOW)])}')
    w = P[(P.frame >= f0) & (P.frame < f0 + WINDOW)].copy()
    w['diff'] = model.predict_features(w[FEATURE_NAMES])['difficulty'].values
    home_id, away_id = pt.home_id, pt.away_id

    bx = np.nanmedian(fr['ball'][f0:f0 + WINDOW, 0])
    xlim = (-2, 50) if bx < 47 else (44, 96)                  # zoom to the half court in use
    fig, ax = plt.subplots(figsize=(6.6, 4.9), dpi=80)
    fig.patch.set_facecolor('white')
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 0.4))
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('difficulty  P(pass fails)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4])
    cb.set_ticklabels(['0%', '10%', '20%', '30%', '40%+'])
    frames = list(range(f0, f0 + WINDOW, STEP))
    cache = {}

    def teammates_scores(i):
        h = int(fr['holder'][i])
        if h == 0:
            return None
        pid, xy, tm = fr['pid'][i], fr['xy'][i], fr['team'][i]
        ih = np.flatnonzero(pid == h)
        if len(ih) == 0:
            return None
        team = tm[ih[0]]
        opp = (tm != team) & (pid != 0)
        mates = np.flatnonzero((tm == team) & (pid != 0) & (pid != h))
        basket = pt._attack_basket(team, int(fr['quarter'][i]))
        if basket is None or opp.sum() < 3:
            return None
        sc = fr['qt'][i] if np.isnan(pt.moments['shot_clock'].iat[i]) else float(pt.moments['shot_clock'].iat[i])
        res = model.score_teammates(xy[ih[0]], xy[mates], xy[opp], [basket[0], 25.0], sc)
        return xy[ih[0]], xy[mates], res['difficulty'].values, res['p_complete'].values

    def draw(k):
        i = frames[k]
        ax.clear()
        court(ax, xlim)
        pid, xy, tm = fr['pid'][i], fr['xy'][i], fr['team'][i]
        for j in range(10):
            c = '#1f77b4' if tm[j] == home_id else '#ff7f0e'
            ax.add_patch(Circle(xy[j], 1.5, color=c, alpha=0.9, zorder=3))
        b = fr['ball'][i]
        ax.add_patch(Circle(b[:2], 0.9 + b[2] / 40, color='black', zorder=5))
        # candidate passes from the current holder
        if i not in cache:
            cache[i] = teammates_scores(i)
        s = cache[i]
        if s is not None:
            P0, M, dif, pc = s
            for m_, d_, p_ in zip(M, dif, pc):
                ax.plot([P0[0], m_[0]], [P0[1], m_[1]], color=CMAP(min(d_ / 0.4, 1)), lw=1.2, alpha=0.55, zorder=2)
                ax.text(*(P0 + 0.8 * (m_ - P0)), f'{p_ * 100:.0f}%', fontsize=6.5, color='#333', ha='center', va='center', zorder=6, clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))
        # the pass actually thrown, while the ball is in the air
        msg = ''
        for r in w.itertuples():
            end = r.frame + int(round(r.flight_s * 25))
            if r.frame <= i <= end + 12:
                pid_r, xy_r = fr['pid'][r.frame], fr['xy'][r.frame]
                ip, it = np.flatnonzero(pid_r == r.passer), np.flatnonzero(pid_r == r.heading_target)
                if len(ip) and len(it) and i <= end:
                    p0, t0 = xy_r[ip[0]], xy_r[it[0]]
                    ax.plot([p0[0], t0[0]], [p0[1], t0[1]], color=CMAP(min(r.diff / 0.4, 1)), lw=4, alpha=0.95, zorder=4, solid_capstyle='round')
                lab = {'completed': 'caught', 'intercepted': 'INTERCEPTED', 'out_of_bounds': 'OUT OF BOUNDS', 'fumbled': 'FUMBLED'}[r.outcome]
                if i <= end:
                    msg = f'pass thrown: model difficulty {r.diff * 100:.0f}%'
                else:
                    msg = f'pass thrown: model difficulty {r.diff * 100:.0f}%  ->  {lab}'
        ax.text(xlim[0] + 3, 52.6, msg or 'ball held: labels = P(catch) of each candidate pass, colour = difficulty', fontsize=8.5, va='bottom', clip_on=False,
                color='#b00020' if ('INTERCEPTED' in msg or 'OUT' in msg or 'FUMBLED' in msg) else '#222')
        ax.text(xlim[1] - 2, -1.8, f'{g}', fontsize=7, ha='right', va='top', color='#777', clip_on=False)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / 12.5)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ani.save(a.out, writer=animation.PillowWriter(fps=12.5))
    print(f'wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {len(frames)} frames)')


if __name__ == '__main__':
    main()
