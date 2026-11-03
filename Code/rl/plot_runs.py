"""Learning curves of a train_agents.py run: python rl/plot_runs.py RUN_DIR [RUN_DIR ...] --out file.png"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(x, k=5):
    x = np.asarray(x, float)
    return np.array([np.nanmean(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in a.runs:
        L = [json.loads(l) for l in open(os.path.join(r, 'log.jsonl'))]
        it = [x['it'] for x in L]
        ax[0].plot(it, [x['ppp'] for x in L], alpha=0.25); ax[0].plot(it, smooth([x['ppp'] for x in L]), label=os.path.basename(r.rstrip('/')))
        ax[1].plot(it, [x.get('def_kl_dec', np.nan) if x.get('def_kl_dec') is not None else np.nan for x in L], label='defence KL per decision')
        ax[2].plot(it, smooth([x.get('stl_per_poss', np.nan) for x in L]), label='steals / poss'); ax[2].plot(it, smooth([x.get('tov_per_poss', np.nan) for x in L]), label='turnovers / poss')
    ax[0].set_title('points per completed possession (both teams, training chains)'); ax[1].set_title('KL to the behaviour-cloned prior (nats / decision)'); ax[2].set_title('steals and turnovers per possession')
    for x in ax: x.set_xlabel('iteration'); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(a.out, dpi=110)


if __name__ == '__main__':
    main()
