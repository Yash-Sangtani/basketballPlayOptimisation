"""Render one simulated possession (from a Sim after run()) as a GIF."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                      # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402
import numpy as np                                    # noqa: E402

UP = 3
MAX_S = 14.0


def _court(ax):
    ax.set_xlim(-2, 49)
    ax.set_ylim(-2, 52)
    ax.set_aspect('equal')
    ax.axis('off')
    c = '#666'
    ax.add_patch(Rectangle((0, 0), 47, 50, fill=False, lw=1.2, color=c))
    ax.add_patch(Rectangle((0, 17), 19, 16, fill=False, lw=1, color=c))
    ax.add_patch(Circle((19, 25), 6, fill=False, lw=1, color=c))
    ax.add_patch(Circle((5.35, 25), 0.75, fill=False, lw=1.5, color='#d9534f'))
    ax.add_patch(Arc((5.35, 25), 47.5, 47.5, theta1=-68, theta2=68, lw=1, color=c))


def render(sim, i, path, title, outcome_text):
    n = int(round(sim.elapsed[i] / 0.2))
    n = min(n, int(MAX_S / 0.2))
    T = np.stack(sim.traj)[:n + 1, i]                 # (S,10,2)
    Bt = np.stack(sim.ball_traj)[:n + 1, i, :2]
    Ht = np.stack(sim.holder_traj)[:n + 1, i]
    ids = sim.ids[i]
    tt = np.linspace(0, n, n * UP + 1)
    src = np.arange(n + 1)
    P = np.stack([np.stack([np.interp(tt, src, T[:, p, c]) for c in (0, 1)], -1) for p in range(10)], 1)
    Bi = np.stack([np.interp(tt, src, Bt[:, c]) for c in (0, 1)], -1)
    hi = np.minimum(np.floor(tt).astype(int), n)
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=60)

    def draw(f):
        ax.clear()
        _court(ax)
        ax.scatter(P[f, :5, 0], P[f, :5, 1], s=95, c='#d62728', zorder=3)
        ax.scatter(P[f, 5:, 0], P[f, 5:, 1], s=95, c='#1f77b4', zorder=3)
        h = Ht[hi[f]]
        for q in range(5):
            if h != 0 and ids[q] == h:
                ax.add_patch(Circle(P[f, q], 1.9, fill=False, ec='k', lw=1.5, zorder=4))
        ax.scatter([Bi[f, 0]], [Bi[f, 1]], s=40, c='#ff9f1c', ec='k', zorder=5)
        last = f >= len(tt) - 1 - 6
        ax.set_title(title + f'   t={tt[f] * 0.2:.1f}s', fontsize=7, loc='left')
        if last:
            ax.text(23.5, 49, outcome_text, ha='center', va='top', fontsize=8, color='#b00020', bbox=dict(fc='white', ec='none', alpha=0.8))

    ani = animation.FuncAnimation(fig, draw, frames=len(tt), interval=1000 / 15)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ani.save(path, writer=animation.PillowWriter(fps=15))
    plt.close(fig)
