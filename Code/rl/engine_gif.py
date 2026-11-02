"""Render the first minutes of a simulated quarter (both teams, scoreboard) as a GIF: the full-court view of the engine.
    cd Code && python rl/engine_gif.py --out ../reports/rl-stage6a/quarter_preview.gif --seconds 100
Absolute frame: team A (the roster of the first offence) attacks the left basket when it has the ball; positions of the other team's possessions are rotated back.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import matplotlib                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import animation         # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle   # noqa: E402
import numpy as np                       # noqa: E402
import keras                             # noqa: E402
import executor_v2 as E                  # noqa: E402
from load import load_games              # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from play_cond import PlayTargets        # noqa: E402
from game_engine import Engine, rot      # noqa: E402
from engine_validate import quarter_starts   # noqa: E402


def court(ax):
    c = '#666'
    ax.set_xlim(-2, 96)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 94, 50, fill=False, lw=1.2, color=c))
    ax.plot([47, 47], [0, 50], color=c, lw=1)
    ax.add_patch(Circle((47, 25), 6, fill=False, lw=1, color=c))
    for x0, sgn, bx in ((0, 1, 5.35), (94, -1, 88.65)):
        ax.add_patch(Rectangle((x0 if sgn == 1 else x0 - 19, 17), 19, 16, fill=False, lw=1, color=c))
        ax.add_patch(Circle((bx, 25), 0.75, fill=False, lw=1.5, color='#d9534f'))
        ax.add_patch(Arc((bx, 25), 47.5, 47.5, theta1=-68 if sgn == 1 else 112, theta2=68 if sgn == 1 else 248, lw=1, color=c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/rl')
    ap.add_argument('--model', default='rl/model/executor_v4g.keras')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=100.0)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    from api import PassValueModel
    cal = json.load(open(os.path.join(HERE, 'model', 'engine_calibration.json')))
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games('holdout'))
    qs = quarter_starts(G)
    g, k, q = qs[a.seed]
    model = keras.saving.load_model(a.model, compile=False, safe_mode=False, custom_objects=E.CUSTOM_OBJECTS)
    eng = Engine(model, PassValueModel(), plays=PlayTargets(), seed=a.seed, **cal).reset(G, table, [(g, k)])
    blocks = int(a.seconds / 0.4) + 40
    for _ in range(blocks):
        eng.step_block()
    T = np.stack(eng.traj)[:, 0]                  # (S,10,2) in the current offence frame
    O = np.stack(eng.order_traj)[:, 0]            # (S,10) roster index per slot
    Bt = np.stack(eng.ball_traj)[:, 0, :2]
    Sc = np.stack(eng.score_traj)[:, 0]
    Ck = np.stack(eng.clock_traj)[:, 0]
    abs_pos = np.zeros_like(T)
    abs_ball = np.zeros_like(Bt)
    for s in range(len(T)):
        b_off = O[s, 0] >= 5
        P = rot(T[s]) if b_off else T[s]
        abs_pos[s] = P
        abs_ball[s] = rot(Bt[s]) if b_off else Bt[s]
    keep = np.flatnonzero(Ck[0] - Ck <= a.seconds)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=60)

    def draw(f):
        s = keep[f]
        ax.clear()
        court(ax)
        team = (O[s] >= 5)
        for r in range(10):
            ax.scatter([abs_pos[s, r, 0]], [abs_pos[s, r, 1]], s=75, c='#1f77b4' if team[r] else '#d62728', zorder=3)
        ax.scatter([abs_ball[s, 0]], [abs_ball[s, 1]], s=36, c='#ff9f1c', ec='k', zorder=5)
        m, sec = divmod(max(Ck[s], 0), 60)
        ax.set_title(f'Q{q}  {int(m)}:{int(sec):02d}   red (A) {int(Sc[s, 0])} - {int(Sc[s, 1])} (B) blue   ({"A" if O[s, 0] < 5 else "B"} on offence)', fontsize=9)

    ani = animation.FuncAnimation(fig, draw, frames=len(keep), interval=1000 / 12)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ani.save(a.out, writer=animation.PillowWriter(fps=12))
    print(f'wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {len(keep)} frames, final score {int(Sc[keep[-1], 0])}-{int(Sc[keep[-1], 1])})')


if __name__ == '__main__':
    main()
