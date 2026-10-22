"""Grouped-by-game evaluation of failed-pass prediction (positive class = failed pass).

    python pass_model.py [--split dev|test] [--models logistic,gboost,mlp] [--pickles DIR]
                         [--seed N] [--folds 5] [--n-boot 1000] [--out DIR] [--plot PNG]

--split dev (default)  GroupKFold by game over the games in splits/dev_games.txt only. The frozen
                       test games are never loaded.
--split test           Fit on ALL dev games, score the frozen test games ONCE. Refuses to start
                       unless --confirm-frozen-test is also given. Do not use it while tuning or
                       comparing models; it exists for the single final number.

The logic lives in evaluation.py (see its docstring for the split/calibration/bootstrap design).
Models: a no-skill baseline and a shot_clock+dist logistic baseline are always reported; the rest
are chosen with --models. Heavy runs (gboost, mlp, full data) go through `heavy.sh`.

Removed from the old version of this file: the import-time read of ./allgames.txt (which did not
exist in Code/) and the NOO skip list (it held '.7z' names that could never match '.pkl' files),
the random 60/40 row split, Keras validation_split, accuracy/F1 on a 98%-positive label, the
printed weights, the backcourt==0 + dropna row filter (now an optional secondary population,
--population frontcourt, with NaNs imputed inside each fold), BackupAndRestore (a fixed directory
would resume fold k from another fold's weights) and the blocking plt.show().
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import evaluation as ev
from evaluation import stamp

TEST_WARNING = """
################################################################################
#  WARNING: --split test touches the FROZEN TEST GAMES.                        #
#  Fit on all dev games, score the test games once. Any result printed below   #
#  must not be used to tune, select or compare models. If you have already     #
#  looked at test results for this project, this run is no longer a fair       #
#  held-out estimate.                                                          #
################################################################################
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=['dev', 'test'], default='dev')
    p.add_argument('--confirm-frozen-test', action='store_true',
                   help='required for --split test: acknowledges the test games are frozen and this is the one final run')
    p.add_argument('--models', default='logistic,gboost,mlp', help='comma list from: logistic, gboost, mlp')
    p.add_argument('--pickles', default=ev.DEFAULT_PICKLES, help='folder with per-game feature pickles and meta/')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--n-boot', type=int, default=1000, help='game-clustered bootstrap replicates')
    p.add_argument('--cal-frac', type=float, default=0.25, help="share of each training fold's games held out to fit the isotonic calibrator")
    p.add_argument('--population', choices=['all', 'frontcourt'], default='all')
    p.add_argument('--dev-list', default=os.path.join(ev.SPLIT_DIR, 'dev_games.txt'))
    p.add_argument('--test-list', default=os.path.join(ev.SPLIT_DIR, 'test_games.txt'))
    p.add_argument('--ignore-status', action='store_true', help='do not require an ok row in _status.csv for each game')
    p.add_argument('--status-only', action='store_true', help='print dataset completeness for the dev games and exit')
    p.add_argument('--out', default=None, help='folder for CSV/markdown/npz outputs (off by default)')
    p.add_argument('--plot', default=None, help='save a reliability + precision-recall figure to this PNG path (off by default)')
    return p


def fmt_ci(v, lo, hi, nd=3) -> str:
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def md_table(df: pd.DataFrame, floatfmt: str = '{:.4f}') -> str:
    cols = list(df.columns)
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = ['' if _blank(v) else (floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)) for v in r]
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def tables(s: dict) -> dict[str, pd.DataFrame]:
    """Presentation tables built from evaluation.summarize()."""
    g = lambda r, name: getattr(r, name, np.nan)
    d = s['discrimination']
    t1 = pd.DataFrame({
        'model': d['model'],
        'ROC-AUC [95% CI]': [fmt_ci(r.roc, r.roc_lo, r.roc_hi) for r in d.itertuples()],
        'PR-AUC [95% CI]': [fmt_ci(r.pr, r.pr_lo, r.pr_hi, 4) for r in d.itertuples()],
        'fold ROC mean+-sd': ['' if _blank(g(r, 'fold_roc_mean')) else f"{r.fold_roc_mean:.3f}+-{r.fold_roc_sd:.3f}" for r in d.itertuples()],
        'fold PR mean+-sd': ['' if _blank(g(r, 'fold_pr_mean')) else f"{r.fold_pr_mean:.4f}+-{r.fold_pr_sd:.4f}" for r in d.itertuples()],
        'fit s (all folds)': ['' if _blank(g(r, 'fit_seconds')) else f"{r.fit_seconds:.0f}" for r in d.itertuples()],
    })
    out = {'discrimination': t1}
    if len(s['paired']):
        p = s['paired']
        out['paired'] = pd.DataFrame({
            'difference': p['model'],
            'ROC-AUC diff [95% CI]': [fmt_ci(r.roc, r.roc_lo, r.roc_hi) for r in p.itertuples()],
            'PR-AUC diff [95% CI]': [fmt_ci(r.pr, r.pr_lo, r.pr_hi, 4) for r in p.itertuples()]})
    c = s['calibration']
    out['calibration'] = pd.DataFrame({
        'model': c['model'],
        'mean pred raw / cal': ['' if _blank(g(r, 'mean_pred_raw')) else f"{r.mean_pred_raw:.4f} / {r.mean_pred_cal:.4f}" for r in c.itertuples()],
        'Brier raw': ['' if _blank(g(r, 'raw_brier')) else f"{r.raw_brier:.5f}" for r in c.itertuples()],
        'Brier cal [95% CI]': [f"{r.cal_brier:.5f}" if _blank(g(r, 'cal_brier_lo')) else fmt_ci(r.cal_brier, r.cal_brier_lo, r.cal_brier_hi, 5) for r in c.itertuples()],
        'BSS raw': ['' if _blank(g(r, 'raw_bss')) else f"{r.raw_bss:+.3f}" for r in c.itertuples()],
        'BSS cal [95% CI]': [f"{r.cal_bss:+.3f}" if _blank(g(r, 'cal_bss_lo')) else fmt_ci(r.cal_bss, r.cal_bss_lo, r.cal_bss_hi) for r in c.itertuples()],
        'logloss raw': ['' if _blank(g(r, 'raw_logloss')) else f"{r.raw_logloss:.4f}" for r in c.itertuples()],
        'logloss cal': [f"{r.cal_logloss:.4f}" for r in c.itertuples()],
        'ROC cal': ['' if _blank(g(r, 'cal_roc')) else f"{r.cal_roc:.3f}" for r in c.itertuples()],
        'PR raw -> cal': ['' if _blank(g(r, 'raw_pr')) else f"{r.raw_pr:.4f} -> {r.cal_pr:.4f}" for r in c.itertuples()],
    })
    return out


def save_plot(path: str, s: dict, res: ev.EvalResult, y: np.ndarray, model_names: list[str]) -> None:
    import matplotlib
    matplotlib.use('Agg')                       #never opens a window
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot([0, 0.2], [0, 0.2], 'k--', lw=1, label='perfect')
    for m in model_names:
        t = s['reliability'][f'{m}|cal']
        ax[0].plot(t['mean_pred'], t['observed'], 'o-', ms=3, label=f'{m} (calibrated)')
    ax[0].set_xlabel('mean predicted P(fail), decile'); ax[0].set_ylabel('observed failure rate')
    ax[0].legend(); ax[0].set_title('Reliability (isotonic, out of fold)')
    yy = y[res.scored]
    for m in model_names:
        pr, rc, _ = precision_recall_curve(yy, res.raw[m])
        ax[1].plot(rc, pr, label=m)
    ax[1].axhline(yy.mean(), color='k', ls='--', lw=1, label='no skill')
    ax[1].set_xlabel('recall'); ax[1].set_ylabel('precision'); ax[1].legend(); ax[1].set_title('Pooled out-of-fold precision-recall')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    #The frozen-test guard runs before anything is read.
    if args.split == 'test':
        if not args.confirm_frozen_test:
            sys.stderr.write("REFUSING to run --split test: the test games are FROZEN. This mode fits on all dev games and "
                             "scores the test games once, as the very last step. Re-run with --confirm-frozen-test only "
                             "if that is what you intend. Nothing was loaded.\n")
            return 2
        sys.stderr.write(TEST_WARNING)

    wanted = [m for m in args.models.split(',') if m]
    bad = [m for m in wanted if m not in ('logistic', 'gboost', 'mlp')]
    if bad:
        sys.stderr.write(f"unknown model(s) {bad}; choose from logistic, gboost, mlp\n")
        return 2
    model_names = ['simple'] + wanted

    dev, test = ev.read_game_list(args.dev_list), ev.read_game_list(args.test_list)
    ev.check_lists_disjoint(dev, test)
    check_status = not args.ignore_status

    if args.status_only:
        st = ev.load_status(args.pickles)
        pk = lambda sub: [g for g in dev if os.path.exists(os.path.join(args.pickles, sub, ev.game_to_pickle_name(g)))]
        ok = [] if st is None else [g for g in dev if g in st.index and st.loc[g, 'status'] == 'ok']
        print(f"dev games {len(dev)}: pickle {len(pk('.'))}, meta {len(pk('meta'))}, status ok {len(ok)}")
        return 0

    games = dev + (test if args.split == 'test' else [])
    stamp(f"loading {len(games)} listed games ({args.split}) from {args.pickles}")
    ds = ev.load_games(args.pickles, games, check_status=check_status)
    ds = ev.restrict_population(ds, args.population)
    sk = {k: len(v) for k, v in ds.info['skipped'].items()}
    unlisted = 0
    if os.path.isdir(args.pickles):
        listed = {ev.game_to_pickle_name(x) for x in dev + test}
        unlisted = sum(1 for f in os.listdir(args.pickles) if f.endswith('.pkl') and f not in listed)
    stamp(f"loaded: {ds.summary()} [{args.split}, population={args.population}]")
    print(f"requested {ds.info['requested']} games; skipped {sk}; pickles in neither list: {unlisted}; "
          f"code versions of loaded games: {ds.info['versions'] or 'unchecked'}; "
          f"label/outcome mismatches: {ds.info['label_outcome_mismatch']}")
    if len(ds.info['versions']) > 1:
        print("WARNING: games were produced by more than one code version (mixed dataset).")

    #Defence in depth: dev mode must never see a test game.
    if args.split == 'dev':
        leaked = set(ds.g) & set(test)
        if leaked:
            raise AssertionError(f"{len(leaked)} test games were loaded in dev mode: {sorted(leaked)[:3]}")
        splits = ev.outer_folds(ds.g, args.folds, args.seed)
    else:
        is_test = np.isin(ds.g, test)
        splits = [(np.flatnonzero(~is_test), np.flatnonzero(is_test))]

    diag = ev.dataset_diagnostics(ds)
    print("\nfeature NaN rates (non-zero only):", {k: round(float(v), 4) for k, v in diag['nan_rate'].items() if v > 0})
    print("near-duplicate feature pairs (|r| > 0.95):", diag['near_duplicates'] or 'none')
    print("single-feature AUC for failure (descriptive, pooled):",
          {k: round(v, 3) for k, v in diag['single_feature_auc'].head(5).items()})

    stamp(f"evaluating {model_names} on {len(splits)} split(s), calibration games {args.cal_frac:.0%} of each training fold")
    res = ev.evaluate_splits(ds.X, ds.y, ds.g, splits, model_names, ds.features, seed=args.seed,
                             cal_frac=args.cal_frac, log=stamp)
    stamp("fitting done; bootstrapping by game")
    s = ev.summarize(ds.y, ds.g, res, model_names, n_boot=args.n_boot, seed=args.seed)
    stamp("bootstrap done")

    label = ('FROZEN TEST (single final run)' if args.split == 'test'
             else f'DEV, {args.folds}-fold GroupKFold by game, seed {args.seed}')
    print(f"\n=== {label}: {s['n_games']} games, {s['n_rows']} scored passes, {s['n_failures']} failures, "
          f"prevalence {s['prevalence']:.4%}, population={args.population} ===")
    tb = tables(s)
    print("\nDiscrimination of the raw model (positive = failed pass; CI = game-clustered bootstrap, "
          f"{args.n_boot} replicates). PR-AUC of no skill = prevalence.")
    print(tb['discrimination'].to_string(index=False))
    if 'paired' in tb:
        print("\nPaired differences vs logistic (same game resamples):")
        print(tb['paired'].to_string(index=False))
    print("\nCalibration (isotonic fit on held-out calibration games inside each training fold; BSS = 1 - Brier / p(1-p)):")
    print(tb['calibration'].to_string(index=False))
    for key, t in s['reliability'].items():
        print(f"\nReliability deciles, {key} (equal-count bins of predicted P(fail)):")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for m, n in res.notes.items():
        if n:
            print(f"\n{m} training notes per fold:", n)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for name in ('discrimination', 'paired', 'calibration'):
            s[name].to_csv(os.path.join(args.out, f'{name}.csv'), index=False)
        for key, t in s['reliability'].items():
            t.to_csv(os.path.join(args.out, f"reliability_{key.replace('|', '_')}.csv"), index=False)
        with open(os.path.join(args.out, 'tables.md'), 'w') as f:
            f.write(f"{label}: {s['n_games']} games, {s['n_rows']} passes, {s['n_failures']} failures\n\n")
            for name, t in tb.items():
                f.write(f"### {name}\n\n{md_table(t)}\n\n")
        np.savez_compressed(os.path.join(args.out, 'oof.npz'), scored=res.scored, fold=res.fold, y=ds.y[res.scored],
                            game=ds.g[res.scored], **{f'raw_{m}': v for m, v in res.raw.items()},
                            **{f'cal_{m}': v for m, v in res.cal.items()})
        stamp(f"wrote outputs to {args.out}")
    if args.plot:
        save_plot(args.plot, s, res, ds.y, model_names)
        stamp(f"wrote figure {args.plot}")
    stamp("done")
    return 0


if __name__ == '__main__':
    sys.exit(main())
