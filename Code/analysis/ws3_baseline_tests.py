"""WS3 tests that run on the shared baseline pickles of the 501 DEV games (never the frozen test list):
   ref   dev reference (logit / hgb / sklearn-gb), GroupKFold(5) by game      (ties back to the 0.851/0.856 headline)
   t3    shuffle control: pass_successful shuffled WITHIN each game, 3 seeds
   t4    feature-subset baselines (shot_clock + dist, + backcourt, passer-only, ...)
   t5    failure types (intercepted / out_of_bounds / fumbled) vs completions, using OOF scores
   t6    matched negatives: completions of the SAME team in the SAME game within +-8 s of each failure
   t6b   target-side geometry only, scored within strata of the passer's context
   t7    permutation importance + univariate directions
Usage (from Code/, under heavy.sh):  python analysis/ws3_baseline_tests.py ref t3 t4 t5 t6 t7 [--limit N games]
Positive class = failed pass (fail == 1).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ws3_common import (FEATS, PASSER, PHASE, DEFENDER, SUBSETS, RES, boot_ci, dev_games, fmt_ci, fold_map,
                        load_baseline, log, make_model, oof_predict, paired_boot, auc_pr, save_json, _game_index,
                        shuffle_within_groups)


def prep(limit=None):
    games = dev_games()
    if limit:
        games = games[::max(1, len(games) // limit)][:limit]
    X, M = load_baseline(games)
    fm = fold_map(X['game'].unique())
    folds = X['game'].map(fm).values
    y = X['fail'].values
    log(f"{X['game'].nunique()} dev games, {len(X)} passes, {y.sum()} failures (prevalence {y.mean():.4f})")
    return X, M, folds, y


def oof_cached(name, X, cols, y, folds, model, force=False):
    path = f'{RES}/oof_{name}.npy'
    if os.path.exists(path) and not force:
        o = np.load(path)
        if len(o) == len(X):
            return o
    t0 = time.time()
    o = oof_predict(X[cols].values, y, folds, model)
    np.save(path, o)
    log(f'   [{name}: {time.time() - t0:.0f}s]')
    return o


def report_line(label, y, p, g, n_boot=200):
    roc, pr = auc_pr(y, p)
    ci = boot_ci(y, p, g, n=n_boot)
    return dict(label=label, roc=roc, roc_ci=ci[0], pr=pr, pr_ci=ci[1])


def show(r):
    log(f"  {r['label']:44s} ROC {fmt_ci(r['roc'], r['roc_ci'])}   PR {fmt_ci(r['pr'], r['pr_ci'], 4)}")


# ----------------------------------------------------------------------------------------
def stage_ref(X, M, folds, y):
    log('\n=== REF: all 15 features, GroupKFold(5) by game, dev games ===')
    out = []
    for model in ('logit', 'hgb', 'gb'):
        o = oof_cached(f'ref_{model}', X, FEATS, y, folds, model)
        r = report_line(model, y, o, X['game'])
        fr = [auc_pr(y[folds == k], o[folds == k]) for k in range(5)]
        r['fold_roc'] = [a for a, _ in fr]
        r['fold_pr'] = [b for _, b in fr]
        show(r)
        log(f"     per-fold ROC {np.round(r['fold_roc'], 3).tolist()}  PR {np.round(r['fold_pr'], 4).tolist()}")
        out.append(r)
    # frontcourt-only (backcourt == 0), the subset pass_model.py used
    fc = (X['backcourt'] == 0).values
    for model in ('logit', 'hgb'):
        o = np.load(f'{RES}/oof_ref_{model}.npy')
        r = report_line(f'{model}, scored on frontcourt rows only (same OOF)', y[fc], o[fc], X['game'][fc])
        show(r)
        out.append(r)
    log(f"  frontcourt share of passes {fc.mean():.3f}; failure rate frontcourt {y[fc].mean():.4f} vs backcourt {y[~fc].mean():.4f}")
    save_json('ref', out)


def stage_t3(X, M, folds, y):
    log('\n=== T3: shuffle pass_successful WITHIN each game (keeps each game base rate) ===')
    out = []
    g = X['game'].values
    for seed in (1, 2, 3):
        ys = shuffle_within_groups(y, g, seed)
        assert ys.sum() == y.sum() and all(ys[i].sum() == y[i].sum() for i in _game_index(g))   # per-game rate kept
        for model in ('logit', 'hgb'):
            o = oof_predict(X[FEATS].values, ys, folds, model)
            roc, pr = auc_pr(ys, o)
            fold_roc = float(np.mean([auc_pr(ys[folds == k], o[folds == k])[0] for k in range(5)]))
            out.append(dict(seed=seed, model=model, roc=roc, pr=pr, prevalence=ys.mean(), fold_mean_roc=fold_roc))
            log(f"  seed {seed} {model:6s}: pooled ROC {roc:.4f} (mean of per-fold ROC {fold_roc:.4f})  "
                f"PR-AUC {pr:.4f}  (base rate {ys.mean():.4f})")
    ok = all(abs(r['roc'] - 0.5) <= 0.015 and abs(r['pr'] - r['prevalence']) <= 0.002 for r in out)
    log(f"  pass bar (ROC within 0.5+-0.015 and PR within base rate +-0.002 on every seed/model): {'PASS' if ok else 'FAIL'}")
    save_json('t3', dict(rows=out, passed=bool(ok)))


def stage_t4(X, M, folds, y):
    log('\n=== T4: feature subsets (GroupKFold(5) by game, dev) ===')
    out, oofs = [], {}
    for name, cols in SUBSETS.items():
        for model in ('logit', 'hgb'):
            o = oof_cached(f"t4_{model}_{name.replace(' ', '_').replace('/', '-').replace('+', '_')}", X, cols, y, folds, model)
            oofs[(name, model)] = o
            r = report_line(f'{name} [{model}] ({len(cols)} f)', y, o, X['game'], n_boot=100)
            show(r)
            out.append(r)
    log('  paired differences (game-level bootstrap, same rows), ROC and PR-AUC:')
    pairs = []
    for model in ('logit', 'hgb'):
        for a, b in (('all 15', 'shot_clock+dist'), ('all 15', 'shot_clock+dist+backcourt'),
                     ('all 15', 'passer-only (no target)'), ('all 15', 'all minus phase'),
                     ('all 15', 'all minus defenders'), ('passer-only (no target)', 'shot_clock+dist+backcourt')):
            d = paired_boot(y, oofs[(a, model)], oofs[(b, model)], X['game'], n=100)
            log(f"   [{model}] {a} - {b}: dROC {d['roc'][0]:+.3f} [{d['roc'][1]:+.3f},{d['roc'][2]:+.3f}]  "
                f"dPR {d['pr'][0]:+.4f} [{d['pr'][1]:+.4f},{d['pr'][2]:+.4f}]")
            pairs.append(dict(model=model, a=a, b=b, **{k: list(v) for k, v in d.items()}))
    save_json('t4', dict(rows=out, pairs=pairs))


def stage_t5(X, M, folds, y):
    log('\n=== T5 (baseline part): how the model scores each failure type vs completions (OOF, all dev games) ===')
    out = []
    comp = (M['outcome'] == 'completed').values
    for model in ('logit', 'hgb'):
        o = oof_cached(f'ref_{model}', X, FEATS, y, folds, model)
        sc = o[comp]
        for oc in ('intercepted', 'out_of_bounds', 'fumbled'):
            m = (M['outcome'] == oc).values
            yy = np.r_[np.ones(m.sum()), np.zeros(comp.sum())]
            pp = np.r_[o[m], sc]
            roc, pr = auc_pr(yy, pp)
            pct = float(np.mean([(sc < v).mean() for v in o[m]]))
            out.append(dict(model=model, outcome=oc, n=int(m.sum()), roc_vs_completed=roc, mean_percentile_in_completions=pct,
                            median_score=float(np.median(o[m])), median_completed=float(np.median(sc))))
            log(f"  [{model}] {oc:14s} n={m.sum():5d}: ROC vs completions {roc:.3f}, mean percentile among completions {pct:.3f}")
    save_json('t5_types', out)


# ---------------------------------------------------------------------------- T6
def build_matches(X, M, window=8.0, cap=10, subset=None):
    """For every failure: completions of the same team in the same game within +-window s of game clock
    (the `cap` nearest if more). Returns dict failure_row -> list of completion rows, and no-match count."""
    df = pd.DataFrame({'game': X['game'].values, 'team': M['team_passer'].values, 't': M['game_time'].values,
                       'fail': X['fail'].values, 'row': np.arange(len(X))})
    if subset is not None:
        df = df[subset[df['row'].values]]
    comp = df[df['fail'] == 0]
    groups = {k: (g['t'].values, g['row'].values) for k, g in comp.sort_values('t').groupby(['game', 'team'])}
    matches, nomatch = {}, 0
    for _, f in df[df['fail'] == 1].iterrows():
        gk = groups.get((f['game'], f['team']))
        if gk is None:
            nomatch += 1
            continue
        t, rows = gk
        lo, hi = np.searchsorted(t, f['t'] - window), np.searchsorted(t, f['t'] + window, side='right')
        if hi <= lo:
            nomatch += 1
            continue
        cand = np.arange(lo, hi)
        if len(cand) > cap:
            cand = cand[np.argsort(np.abs(t[cand] - f['t']))[:cap]]
        matches[int(f['row'])] = rows[cand].tolist()
    return matches, nomatch


def concordance(matches, score, games, n_boot=200, seed=0):
    """Mean over failures of the share of its matched completions scoring LOWER than it (ties 0.5);
    95% CI from a game-level bootstrap."""
    fr = np.array(list(matches.keys()))
    c = np.array([np.mean([(score[f] > score[m]) + 0.5 * (score[f] == score[m]) for m in matches[f]]) for f in fr])
    g = games[fr]
    ug = pd.Series(c).groupby(g).agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        bs.append(ug['sum'].values[pick].sum() / ug['count'].values[pick].sum())
    return float(c.mean()), np.percentile(bs, [2.5, 97.5]).tolist(), len(fr)


def stage_t6(X, M, folds, y):
    log('\n=== T6: matched negatives (same team, same game, +-8 s game clock) ===')
    games = X['game'].values
    out = []
    sets = {'all passes': None, 'frontcourt only (backcourt==0)': (X['backcourt'] == 0).values}
    feat_sets = {'all 15': FEATS, 'all minus phase': SUBSETS['all minus phase']}
    for sname, sub in sets.items():
        matches, nomatch = build_matches(X, M, subset=sub)
        nf = int(y[sub].sum()) if sub is not None else int(y.sum())
        comp_rows = sorted({m for v in matches.values() for m in v})
        rows = np.array(sorted(set(matches) | set(comp_rows)))
        log(f"\n -- {sname}: {nf} failures, {len(matches)} matched ({nomatch} with no completion within +-8 s), "
            f"{len(comp_rows)} distinct matched completions, mean matches/failure "
            f"{np.mean([len(v) for v in matches.values()]):.1f}")
        for fname, cols in feat_sets.items():
            for model in ('logit', 'hgb'):
                # (a) scores from the GLOBAL OOF model (all dev rows), judged only within matched sets
                key = 'ref_' + model if fname == 'all 15' else f"t4_{model}_all_minus_phase"
                glob_oof = np.load(f'{RES}/oof_{key}.npy')
                conc, ci, nfm = concordance(matches, glob_oof, games)
                # pooled AUC on the matched rows (global OOF scores)
                yy = y[rows]
                roc_m, pr_m = auc_pr(yy, glob_oof[rows])
                # (b) retrain on the matched rows only (grouped CV by game)
                fo = folds[rows]
                om = oof_predict(X.iloc[rows][cols].values, yy, fo, model)
                roc_r, pr_r = auc_pr(yy, om)
                ci_r = boot_ci(yy, om, games[rows], n=100)
                rec = dict(set=sname, features=fname, model=model, n_fail=nfm, n_nomatch=nomatch, n_matched_comp=len(comp_rows),
                           concordance=conc, concordance_ci=ci, pooled_roc_global_model=roc_m, pooled_pr_global_model=pr_m,
                           retrained_roc=roc_r, retrained_roc_ci=ci_r[0], retrained_pr=pr_r, matched_prevalence=float(yy.mean()))
                out.append(rec)
                log(f"   {fname:16s} [{model:5s}] concordance {fmt_ci(conc, ci)} (0.5 = chance) | pooled ROC (global model) {roc_m:.3f} "
                    f"| retrained on matched: ROC {fmt_ci(roc_r, ci_r[0])}, PR {pr_r:.4f} (prev {yy.mean():.3f})")
    # descriptive: how different are failures from their matched completions on the context variables?
    matches, _ = build_matches(X, M)
    fr = np.array(list(matches.keys()))
    desc = {}
    for c in ['shot_clock', 'passer_basket_dist', 'backcourt', 'dist', 'closest_def_dist_passer', 'closest_def_trajectory',
              'lane_n_def_5ft', 'passer_speed']:
        v = X[c].values
        fm_ = np.array([np.nanmean(v[matches[f]]) for f in fr])
        desc[c] = dict(failures=float(np.nanmean(v[fr])), matched_completions=float(np.nanmean(fm_)))
    log('  mean value: failures vs their matched completions ->',
        {k: (round(v['failures'], 2), round(v['matched_completions'], 2)) for k, v in desc.items()})
    save_json('t6', dict(rows=out, descriptive=desc))


# --------------------------------------------------------------------------- T6b
TARGET_SIDE = ['dist', 'ball_end_basket_dist', 'basket_angle', 'closest_def_dist_ball_end', 'closest_def_trajectory',
               'closest_def_angle_passer', 'closest_def_angle_ball_end', 'closest_def_trajectory_angle',
               'lane_n_def_5ft', 'target_speed']


def stage_t6b(X, M, folds, y):
    """Stratified control: strata = passer-side context (backcourt x quintiles of passer_basket_dist,
    closest_def_dist_passer, shot_clock).  Inside a stratum the passer's situation is the same for failures and
    completions; does a model that sees ONLY target-side/lane geometry still rank failures above completions?"""
    log('\n=== T6b: target-side geometry within passer-context strata ===')
    games = X['game'].values
    o = oof_cached('t6b_hgb_target_side', X, TARGET_SIDE, y, folds, 'hgb')
    o_l = oof_cached('t6b_logit_target_side', X, TARGET_SIDE, y, folds, 'logit')
    out = []
    for nq in (3, 5):
        keys = [X['backcourt'].fillna(-1).astype(int).astype(str)]
        for c in ('passer_basket_dist', 'closest_def_dist_passer', 'shot_clock'):
            q = pd.qcut(X[c], nq, labels=False, duplicates='drop')
            keys.append(q.fillna(-1).astype(int).astype(str))
        strata = keys[0].str.cat(keys[1:], sep='_').values
        for name, sc in (('hgb target-side only', o), ('logit target-side only', o_l)):
            df = pd.DataFrame({'s': strata, 'sc': sc, 'y': y, 'g': games})
            vals, gg = [], []
            for _, d in df.groupby('s'):
                f, c = d[d['y'] == 1], d[d['y'] == 0]
                if len(f) == 0 or len(c) == 0:
                    continue
                cs = np.sort(c['sc'].values)
                lo = np.searchsorted(cs, f['sc'].values, side='left')
                hi = np.searchsorted(cs, f['sc'].values, side='right')
                vals.extend(((lo + hi) / 2 / len(cs)).tolist())
                gg.extend(f['g'].tolist())
            vals, gg = np.array(vals), np.array(gg)
            ug = pd.Series(vals).groupby(gg).agg(['sum', 'count'])
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(200):
                pick = rng.integers(0, len(ug), len(ug))
                bs.append(ug['sum'].values[pick].sum() / ug['count'].values[pick].sum())
            ci = np.percentile(bs, [2.5, 97.5])
            glob_roc = auc_pr(y, sc)[0]
            log(f"  {nq}-quantile strata ({df['s'].nunique()} strata, {len(vals)} failures with a non-empty stratum), {name}: "
                f"within-stratum concordance {vals.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  (global ROC of same scores {glob_roc:.3f})")
            out.append(dict(nq=nq, model=name, concordance=float(vals.mean()), ci=ci.tolist(), n_fail=len(vals), global_roc=glob_roc))
    save_json('t6b', out)


# ---------------------------------------------------------------------------- T7
def stage_t7(X, M, folds, y, repeats=5):
    log('\n=== T7: permutation importance on held-out folds (drop in metric when a column/group is shuffled) ===')
    groups = {'phase/location (shot_clock, backcourt, passer_basket_dist, ball_end_basket_dist, basket_angle)': PHASE,
              'defenders (all 7)': DEFENDER,
              'lane defenders (closest_def_trajectory[_angle], lane_n_def_5ft)':
                  ['closest_def_trajectory', 'closest_def_trajectory_angle', 'lane_n_def_5ft'],
              'speeds (passer_speed, target_speed)': ['passer_speed', 'target_speed'],
              'passer-side (shot_clock, backcourt, passer_basket_dist, closest_def_dist_passer, passer_speed)': PASSER}
    rng = np.random.default_rng(0)
    out = {}
    Xv = X[FEATS].values
    for model in ('hgb', 'logit'):
        res = {k: [] for k in list(FEATS) + list(groups)}
        for k in range(5):
            tr, te = folds != k, folds == k
            m = make_model(model).fit(Xv[tr], y[tr])
            Xt, yt = Xv[te].copy(), y[te]
            base_roc, base_pr = auc_pr(yt, m.predict_proba(Xt)[:, 1])
            for name, cols in list(((f, [f]) for f in FEATS)) + list(groups.items()):
                ci = [FEATS.index(c) for c in cols]
                d = []
                for _ in range(repeats):
                    Xp = Xt.copy()
                    perm = rng.permutation(len(Xp))
                    Xp[:, ci] = Xt[perm][:, ci]          # shuffle the columns jointly (keeps within-group correlation)
                    r, p = auc_pr(yt, m.predict_proba(Xp)[:, 1])
                    d.append((base_roc - r, base_pr - p))
                res[name].append(np.mean(d, axis=0))
        out[model] = {k: dict(d_roc=float(np.mean([a for a, _ in v])), d_roc_sd=float(np.std([a for a, _ in v])),
                              d_pr=float(np.mean([b for _, b in v]))) for k, v in res.items()}
        log(f"\n  [{model}] importance (mean over 5 folds x {repeats} shuffles; baseline ROC ~ see REF)")
        for k, v in sorted(out[model].items(), key=lambda kv: -kv[1]['d_roc']):
            log(f"    {k[:96]:96s} dROC {v['d_roc']:+.4f} (sd {v['d_roc_sd']:.4f})  dPR {v['d_pr']:+.4f}")
    # univariate direction check
    uni = {}
    log('\n  univariate: AUC for FAIL (>0.5 = larger values in failures), medians of failures vs completions')
    for c in FEATS:
        v = X[c].values
        ok = ~np.isnan(v)
        a = roc_auc_score(y[ok], v[ok])
        uni[c] = dict(auc=float(a), median_fail=float(np.nanmedian(v[y == 1])), median_comp=float(np.nanmedian(v[y == 0])))
        log(f"    {c:30s} AUC {a:.3f}   median fail {uni[c]['median_fail']:8.2f}   median completed {uni[c]['median_comp']:8.2f}")
    lg = make_model('logit').fit(Xv, y)
    coef = dict(zip(FEATS, lg[-1].coef_[0].round(3).tolist()))
    log('  standardised logistic coefficients (positive = raises P(fail)):', coef)
    save_json('t7', dict(perm=out, univariate=uni, logit_coef=coef))


STAGES = dict(ref=stage_ref, t3=stage_t3, t4=stage_t4, t5=stage_t5, t6=stage_t6, t6b=stage_t6b, t7=stage_t7)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stages', nargs='+', choices=list(STAGES))
    ap.add_argument('--limit', type=int, default=None, help='debug: use only N dev games (also set WS3_RES to a scratch dir)')
    a = ap.parse_args()
    X, M, folds, y = prep(a.limit)
    for s in a.stages:
        t0 = time.time()
        STAGES[s](X, M, folds, y)
        log(f'[{s} took {time.time() - t0:.0f}s]')
