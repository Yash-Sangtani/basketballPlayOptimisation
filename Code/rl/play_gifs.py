"""Stage 3: one prototype GIF per discovered play (cluster).

    cd Code && python rl/play_gifs.py --data ../data/rl --stage3 ../reports/rl-stage3 --out ../docs/plays

Prototype = the validation-game possession whose descriptor is closest (PCA space) to the cluster centre, i.e. a real possession of a
game the library was not fitted on. Attack-normalised: the offence attacks the basket on the left, red = offence, blue = defence,
orange = ball, ring around the ball handler. First 8 s after midcourt, real time, 3x interpolation.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import matplotlib                        # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import animation         # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle   # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from load import load_games              # noqa: E402
from plays import possessions, describe   # noqa: E402
from train_executor import fit_val_games   # noqa: E402

DUR = 8.0
UP = 3


def court(ax):
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
    ax.plot([14, 14], [3, 47], alpha=0)  # keep limits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--stage3', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lib = np.load(os.path.join(a.stage3, 'play_library.npz'))
    labs = np.load(os.path.join(a.stage3, 'labels.npz'), allow_pickle=True)
    summ = pd.read_csv(os.path.join(a.stage3, 'cluster_summary.csv'))
    K = int(lib['K'])
    _, val_g = fit_val_games()
    VAL = load_games(a.data, val_g)
    Z, ref = [], []
    for d in VAL:
        for p, idx in possessions(d):
            x = (describe(d, idx) - lib['mu']) / lib['sd']
            Z.append((x - lib['pca_mean']) @ lib['pca_components'].T)
            ref.append((d, idx, p))
    Z = np.array(Z)
    dist = ((Z[:, None, :] - lib['centers'][None]) ** 2).sum(-1)       # (N,K)
    assign = dist.argmin(1)
    os.makedirs(a.out, exist_ok=True)
    for k in range(K):
        m = np.flatnonzero(assign == k)
        if len(m) == 0:
            continue
        j = m[dist[m, k].argmin()]
        d, idx, p = ref[j]
        tf = d['t_front'][idx] - d['t_front'][idx[0]]
        sel = idx[tf <= DUR]
        t = tf[tf <= DUR]
        pos = np.concatenate([d['pos'][sel], d['ball'][sel, None, :2]], axis=1)          # (T,11,2)
        hold = d['holder'][sel]
        ids = d['ids'][sel]
        tt = np.linspace(t[0], t[-1], (len(t) - 1) * UP + 1)
        P = np.stack([np.stack([np.interp(tt, t, pos[:, i, c]) for c in (0, 1)], -1) for i in range(11)], 1)
        hi = np.searchsorted(t, tt, side='right') - 1
        s = summ[(summ.cluster == k) & (summ.split == 'val')].iloc[0]
        fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=60)

        def draw(f):
            ax.clear()
            court(ax)
            ax.scatter(P[f, :5, 0], P[f, :5, 1], s=95, c='#d62728', zorder=3)
            ax.scatter(P[f, 5:10, 0], P[f, 5:10, 1], s=95, c='#1f77b4', zorder=3)
            h = hold[hi[f]]
            for q in range(5):
                if ids[hi[f], q] == h and h != 0:
                    ax.add_patch(Circle(P[f, q], 1.9, fill=False, ec='k', lw=1.5, zorder=4))
            ax.scatter([P[f, 10, 0]], [P[f, 10, 1]], s=40, c='#ff9f1c', ec='k', zorder=5)
            ax.set_title(f'play {k}: {int(s.n)} val poss, {s.fg_pts:.2f} FG pts/poss, {s.three_share * 100:.0f}% of shots 3s   t={tt[f] - tt[0]:.1f}s', fontsize=7, loc='left')

        ani = animation.FuncAnimation(fig, draw, frames=len(tt), interval=1000 / 15)
        out = os.path.join(a.out, f'play_{k:02d}.gif')
        ani.save(out, writer=animation.PillowWriter(fps=15))
        plt.close(fig)
        print(f'{out} {os.path.getsize(out) / 1e6:.2f} MB  ({d["game"]} poss {p}, {len(tt)} frames)', flush=True)


if __name__ == '__main__':
    main()
