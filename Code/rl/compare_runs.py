"""Paired comparison of the profile modes from train_executor2.py output: seed-to-seed spread and game-clustered bootstrap intervals.

    cd Code && python rl/compare_runs.py ../reports/rl-stage2b
Per-sample losses are averaged over the seeds of a mode (same validation samples for every run), then the difference of means between two
modes is bootstrapped over validation GAMES (500 resamples).
"""
import glob
import json
import os
import sys

import numpy as np

METRICS = ['nll_off', 'nll_def', 'ade1_off', 'ade1_def', 'dec_ll', 'tgt_ll']


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, '*_s*.npz'))):
        mode, seed = os.path.basename(f)[:-4].rsplit('_s', 1)
        z = np.load(f)
        runs.setdefault(mode, {})[int(seed)] = {k: z[k] for k in z.files}
    return runs


def boot(game, a, b, n=500, seed=0):
    ug, gi = np.unique(game, return_inverse=True)
    ok = ~(np.isnan(a) | np.isnan(b))
    cnt = np.bincount(gi[ok], minlength=len(ug)).astype(float)
    sa = np.bincount(gi[ok], weights=a[ok], minlength=len(ug))
    sb = np.bincount(gi[ok], weights=b[ok], minlength=len(ug))
    rng = np.random.default_rng(seed)
    d0 = sa.sum() / cnt.sum() - sb.sum() / cnt.sum()
    out = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))
        out.append((w @ sa - w @ sb) / (w @ cnt))
    return float(d0), np.percentile(out, [2.5, 97.5]).tolist()


def main():
    d = sys.argv[1]
    R = load(d)
    res = {'spread': {}, 'paired': {}}
    for mode, seeds in R.items():
        res['spread'][mode] = {}
        for m in METRICS:
            v = [float(np.nanmean(s[m])) for s in seeds.values()]
            res['spread'][mode][m] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n_seeds=len(v))
        print(f"{mode:9s} seeds {sorted(seeds)}: " + '  '.join(f"{m} {res['spread'][mode][m]['mean']:.4f}" + (f"+-{res['spread'][mode][m]['sd']:.4f}" if res['spread'][mode][m]['sd'] is not None else '') for m in METRICS))
    avg = {mode: {m: np.mean([s[m] for s in seeds.values()], axis=0) for m in METRICS} for mode, seeds in R.items()}
    game = next(iter(next(iter(R.values())).values()))['game_idx']
    for a, b in (('real', 'none'), ('shuffled', 'none'), ('real', 'shuffled')):
        if a in avg and b in avg:
            for m in METRICS:
                d0, ci = boot(game, avg[a][m], avg[b][m])
                res['paired'][f'{a} - {b} | {m}'] = dict(diff=d0, ci=ci)
                print(f'  {a} - {b} | {m:9s}: {d0:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]   (negative = {a} better for losses/errors)')
    json.dump(res, open(os.path.join(d, 'comparison.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
