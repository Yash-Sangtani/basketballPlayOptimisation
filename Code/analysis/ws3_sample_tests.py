"""WS3 tests on the per-game extraction of the dev SAMPLE (ws3_extract.py output; dev games only):
   t1    target-mode check: 'actual' vs 'heading' on identical games/folds (paired), plus the live-ball cut
   t2    tracking-only failure labels: counts, auto-tags, manual-look sheet, do the features predict them?
   t5    non-pass turnovers (lost ball / travel / off. foul / other) scored by the failure model
Usage (from Code/, under heavy.sh):  python analysis/ws3_sample_tests.py t1 t2 t5
Positive class = failed pass.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ws3_common import (FEATS, PASSER, RES, auc_pr, boot_ci, dev_games, fmt_ci, fold_map, load_baseline, load_sample,
                        log, make_model, oof_predict, paired_boot, read_list, save_json, CODE)

pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 400)
pd.set_option('display.max_columns', 60)

DEAD_OR_SHOT = ('dead_ball_or_inbound', 'shot_or_rebound_context', 'releaser_shot_context')


def tag_trk(r):
    tags = set(r.pbp_tags.split(',')) if r.pbp_tags else set()
    if r.clock_running == 0 or r.inbound == 1:
        return 'dead_ball_or_inbound'
    if 'shot_by_releaser' in tags:
        return 'releaser_shot_context'
    if r.pbp_to_by_passer == 2:
        return 'pbp_lost_ball_by_releaser'
    if r.pbp_to_by_passer in (1, 45):
        return 'pbp_bad_pass_by_releaser_other_release'
    if r.pbp_to_by_passer != 0:
        return 'pbp_other_turnover_by_releaser'
    if tags & {'shot_made', 'shot_miss', 'ft', 'block', 'rebound'}:
        return 'shot_or_rebound_context'
    if r.trk_outcome == 'out_of_bounds' and r.oob_next_same == 1:
        return 'oob_kept_by_offense'
    if tags & {'foul', 'violation', 'timeout', 'jumpball', 'period_end', 'period_start'}:
        return 'stoppage_context'
    if tags & {'to_badpass', 'to_lostball', 'to_other'}:
        return 'other_players_turnover_nearby'
    return 'no_pbp_event_live'


def prep():
    S = load_sample()
    ext = S['ext']
    log(f"sample: {ext['game'].nunique()} games, {len(ext)} release rows: {ext['src'].value_counts().to_dict()}")
    ext['trk_tag'] = ''
    m = ext['src'] == 'trk_only'
    ext.loc[m, 'trk_tag'] = ext[m].apply(tag_trk, axis=1)
    ext['live'] = ((ext['clock_running'] != 0) & (ext['inbound'] != 1))
    return S


def sample_folds(S):
    fm = fold_map(S['ext']['game'].unique())
    return fm


def baseline_other_games(sample_games):
    """Baseline rows (actual mode) of the dev games NOT in the sample: the training set for transfer tests."""
    others = [g for g in dev_games() if g not in set(sample_games)]
    if os.environ.get('WS3_TRAIN_LIMIT'):    # debugging only
        others = others[::max(1, len(others) // int(os.environ['WS3_TRAIN_LIMIT']))]
    X, M = load_baseline(others)
    log(f"transfer training set: {len(others)} other dev games, {len(X)} passes, {X['fail'].sum()} failures")
    return X


# ------------------------------------------------------------------------------- T1
def stage_t1(S):
    log('\n=== T1: target mode, identical games and folds ===')
    ext, fa, fh = S['ext'], S['feat']['actual'], S['feat']['heading']
    base = ext.index[ext['src'].isin(['completed', 'pbp_fail'])]
    A, H = fa.loc[fa.index.isin(base)], fh.loc[fh.index.isin(base)]
    fm = sample_folds(S)
    comp = ext[ext['src'] == 'completed']
    log(f"  base rows {len(base)}: actual keeps {len(A)}, heading keeps {len(H)}; "
        f"completions without a heading target (dropped in heading mode): "
        f"{(~comp.index.isin(H.index) & comp.index.isin(A.index)).sum()} of {len(comp)}")
    agree = (comp['target_heading'] == comp['target']).mean()
    nohead = comp['target_heading'].isna().mean()
    log(f"  heading target == true receiver on completions: {agree:.3f} (no heading target: {nohead:.3f})")
    out = {}

    def run(df, label, model, cols=FEATS):
        y = (df['pass_successful'] == 0).astype(int).values
        folds = df['game'].map(fm).values
        o = oof_predict(df[cols].values, y, folds, model)
        r = auc_pr(y, o)
        ci = boot_ci(y, o, df['game'], n=200)
        log(f"  {label:48s} [{model:5s}] n={len(df):6d} fail={y.sum():4d}  ROC {fmt_ci(r[0], ci[0])}  PR {fmt_ci(r[1], ci[1], 4)}")
        return o, y

    inter = A.index.intersection(H.index)
    for model in ('logit', 'hgb'):
        oa, ya = run(A, 'actual, all rows', model)
        oh, yh = run(H, 'heading, all rows', model)
        Ai, Hi = A.loc[inter], H.loc[inter]
        oai, yi = run(Ai, 'actual, intersection rows', model)
        ohi, yi2 = run(Hi, 'heading, intersection rows', model)
        assert (yi == yi2).all()
        d = paired_boot(yi, oai, ohi, Ai['game'].values, n=300)
        log(f"     PAIRED (intersection, same games/folds) actual - heading: dROC {d['roc'][0]:+.4f} "
            f"[{d['roc'][1]:+.4f}, {d['roc'][2]:+.4f}]   dPR {d['pr'][0]:+.4f} [{d['pr'][1]:+.4f}, {d['pr'][2]:+.4f}]")
        out[model] = dict(paired_roc=list(d['roc']), paired_pr=list(d['pr']), n_inter=len(inter), n_fail=int(yi.sum()))
    # target-free reference on the same games
    for model in ('logit', 'hgb'):
        run(A, 'passer-only features (no target at all)', model, PASSER)
    out['agree'] = float(agree)
    out['no_heading_share_comp'] = float(nohead)
    save_json('t1', out)

    # --- live-ball cut (dead-ball / inbound completions are part of how the completion class is built)
    log('\n--- T1b: dead-ball / inbound completions ---')
    e = ext.loc[A.index]
    comp_e = e[e['src'] == 'completed']
    dead = ~comp_e['live']
    log(f"  completions: {dead.mean():.3f} dead-ball or inbound ({dead.sum()} of {len(comp_e)}); "
        f"clock stopped {(comp_e['clock_running'] == 0).mean():.3f}, passer out of bounds {(comp_e['inbound'] == 1).mean():.3f}; "
        f"pbp failures dead/inbound: {(~e[e['src'] == 'pbp_fail']['live']).mean():.3f}")
    keep = ~((e['src'] == 'completed') & ~e['live'])
    out2 = dict(dead_share_completions=float(dead.mean()))
    for model in ('logit', 'hgb'):
        # model trained on all rows, judged on the live-only subset (same OOF scores)
        y = (A['pass_successful'] == 0).astype(int).values
        folds = A['game'].map(fm).values
        o = oof_predict(A[FEATS].values, y, folds, model)
        r_all = auc_pr(y, o)
        r_live = auc_pr(y[keep.values], o[keep.values])
        ci_live = boot_ci(y[keep.values], o[keep.values], A['game'].values[keep.values], n=200)
        # model retrained on live rows only
        Al = A[keep.values]
        yl = (Al['pass_successful'] == 0).astype(int).values
        ol = oof_predict(Al[FEATS].values, yl, Al['game'].map(fm).values, model)
        r_re = auc_pr(yl, ol)
        log(f"  [{model:5s}] all rows ROC {r_all[0]:.3f} PR {r_all[1]:.4f} | same OOF, dead/inbound completions removed: ROC "
            f"{fmt_ci(r_live[0], ci_live[0])} PR {r_live[1]:.4f} | retrained on live rows: ROC {r_re[0]:.3f} PR {r_re[1]:.4f} "
            f"(live prevalence {yl.mean():.4f})")
        out2[model] = dict(all=r_all, live_same_oof=r_live, retrained_live=r_re)
    # frontcourt + live
    fc = (A['backcourt'] == 0).values & keep.values
    for model in ('logit', 'hgb'):
        y = (A['pass_successful'] == 0).astype(int).values
        o = oof_predict(A[FEATS].values, y, A['game'].map(fm).values, model)
        log(f"  [{model:5s}] same OOF on frontcourt & live rows only (n={fc.sum()}, fail={y[fc].sum()}): ROC {auc_pr(y[fc], o[fc])[0]:.3f} "
            f"PR {auc_pr(y[fc], o[fc])[1]:.4f}")
    save_json('t1b', out2)


# ------------------------------------------------------------------------------- T2
def stage_t2(S):
    log('\n=== T2: tracking-only failure labels ===')
    ext, fa = S['ext'], S['feat']['actual']
    rep = S['report']
    ng = ext['game'].nunique()
    cnt = ext.groupby('game')['src'].value_counts().unstack(fill_value=0)
    log(f"  per game (mean over {ng} games): completed {cnt['completed'].mean():.1f}, pbp_fail {cnt['pbp_fail'].mean():.1f}, "
        f"trk_only {cnt['trk_only'].mean():.1f} (sd {cnt['trk_only'].std():.1f}), np-only rows {cnt.filter(like='np_').sum(axis=1).mean():.1f}; "
        f"trk-only candidates removed by _flight_ok: {rep['n_trk_filtered'].mean():.1f}/game")
    log(f"  completed rows relabelled failed by tracking (fumbled after receipt): {int(((ext['src'] == 'completed') & (ext['trk_fail'] == 1)).sum())}"
        f" of {int((ext['src'] == 'completed').sum())}")
    t = ext[ext['src'] == 'trk_only']
    log(f"  trk_only total {len(t)}: outcomes {t['trk_outcome'].value_counts().to_dict()}")
    tab = pd.crosstab(t['trk_tag'], t['trk_outcome'], margins=True)
    tab['share'] = (tab['All'] / len(t)).round(3)
    log('\n  auto-tags (priority order as listed in tag_trk), counts by flight outcome:\n' + tab.to_string())
    desc = t.groupby('trk_tag')[['zmax', 'ball_disp', 'init_speed', 'hold_frames']].median().round(1)
    desc['n'] = t.groupby('trk_tag').size()
    pbpf = ext[ext['src'] == 'pbp_fail']
    desc.loc['(ref) pbp_fail rows'] = list(pbpf[['zmax', 'ball_disp', 'init_speed', 'hold_frames']].median().round(1)) + [len(pbpf)]
    cm = ext[(ext['src'] == 'completed') & ext['live']]
    desc.loc['(ref) live completions'] = list(cm[['zmax', 'ball_disp', 'init_speed', 'hold_frames']].median().round(1)) + [len(cm)]
    log('\n  medians per tag (zmax ft, ball displacement ft, ball speed first 0.32 s ft/s, hold length in frames):\n' + desc.to_string())
    has_feat = t.index.isin(fa.index)
    log(f"\n  trk_only rows that have a heading target (usable for the model): {has_feat.sum()} of {len(t)} ({has_feat.mean():.2f}); "
        f"pbp_fail rows usable: {pbpf.index.isin(fa.index).mean():.2f}")
    out = dict(trk_only_per_game=float(cnt['trk_only'].mean()), pbp_fail_per_game=float(cnt['pbp_fail'].mean()),
               tags=t['trk_tag'].value_counts().to_dict())

    # --- do the same features predict them?
    sg = ext['game'].unique()
    Xtr = baseline_other_games(sg)
    ytr = Xtr['fail'].values
    E = ext.loc[fa.index, ['src', 'trk_fail', 'trk_tag', 'live', 'trk_outcome', 'outcome', 'game']].join(fa[FEATS])
    neg_all = E[(E['src'] == 'completed') & (E['trk_fail'] == 0)]
    neg_live = neg_all[neg_all['live']]
    groups = {'pbp-confirmed failures': E[E['src'] == 'pbp_fail'],
              'tracking-only failures (all)': E[E['src'] == 'trk_only'],
              'completed-then-fumbled (tracking)': E[(E['src'] == 'completed') & (E['trk_fail'] == 1)]}
    for tg in sorted(E['trk_tag'].unique()):
        if tg:
            groups[f'  trk_only: {tg}'] = E[(E['src'] == 'trk_only') & (E['trk_tag'] == tg)]
    live_nonshot = E[(E['src'] == 'trk_only') & ~E['trk_tag'].isin(DEFAULT_EXCL)]
    groups['tracking-only, live & non-shot (excl. dead_ball/shot tags)'] = live_nonshot
    res = {}
    log('\n  (i) TRANSFER: model trained on the pbp labels of the 401 OTHER dev games, scoring this sample.')
    log('      AUC of each positive group vs (a) all completions, (b) live completions only; mean percentile of the group among live completions')
    for model in ('logit', 'hgb'):
        m = make_model(model).fit(Xtr[FEATS].values, ytr)
        s_neg_all = m.predict_proba(neg_all[FEATS].values)[:, 1]
        s_neg_live = m.predict_proba(neg_live[FEATS].values)[:, 1]
        log(f"   [{model}]")
        for gname, g in groups.items():
            if len(g) == 0:
                continue
            sg_ = m.predict_proba(g[FEATS].values)[:, 1]
            a_all = roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(neg_all))], np.r_[sg_, s_neg_all])
            a_live = roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(neg_live))], np.r_[sg_, s_neg_live])
            pct = float(np.mean([(s_neg_live < v).mean() for v in sg_]))
            log(f"      {gname:62s} n={len(g):5d}  AUC vs all comp {a_all:.3f} | vs live comp {a_live:.3f} | mean pct {pct:.3f}")
            res.setdefault(model, {})[gname.strip()] = dict(n=len(g), auc_all=a_all, auc_live=a_live, pct=pct)
    out['transfer'] = res

    log('\n  (ii) WITHIN-SAMPLE grouped CV (100 games) with TRACKING labels: positives = pbp failures + tracking-only failures + fumbled completions')
    fm = sample_folds(S)
    pos = E[(E['trk_fail'] == 1)]
    for variant, neg, posv in (('all rows', neg_all, pos),
                               ('live non-inbound completions as negatives; positives without dead_ball/shot tags',
                                neg_live, pos[~pos['trk_tag'].isin(DEFAULT_EXCL)])):
        D = pd.concat([posv.assign(y=1), neg.assign(y=0)])
        y, folds = D['y'].values, D['game'].map(fm).values
        for model in ('logit', 'hgb'):
            o = oof_predict(D[FEATS].values, y, folds, model)
            parts = []
            for gname, mask in (('all positives', D['y'] == 1), ('pbp-confirmed', D['src'] == 'pbp_fail'),
                                ('tracking-only', D['src'] == 'trk_only')):
                sel = (mask | (D['y'] == 0)).values
                r = auc_pr(y[sel], o[sel])
                parts.append(f"{gname}: ROC {r[0]:.3f} PR {r[1]:.4f} (n+={int((mask & (D['y'] == 1)).sum())})")
            log(f"   [{variant}] [{model}] n={len(D)}, positives {y.sum()} ({y.mean():.3f}) | " + ' | '.join(parts))
    save_json('t2', out)
    manual_sheet(S, ext)


DEFAULT_EXCL = ('dead_ball_or_inbound', 'shot_or_rebound_context', 'releaser_shot_context')


def manual_sheet(S, ext, n_per=3, seed=7, path=None):
    """~20 random trk_only examples, stratified by auto-tag, with the pbp text near the event, for a human look."""
    rng = np.random.default_rng(seed)
    t = ext[ext['src'] == 'trk_only']
    rows = []
    tags = t['trk_tag'].value_counts()
    for tag in tags.index:
        k = min(len(t[t['trk_tag'] == tag]), n_per if tags[tag] >= 30 else 1)
        rows.append(t[t['trk_tag'] == tag].sample(k, random_state=int(rng.integers(1e6))))
    s = pd.concat(rows)
    if len(s) < 20:
        extra = t.drop(s.index).sample(20 - len(s), random_state=3)
        s = pd.concat([s, extra])
    lines = []
    for rid, r in s.iterrows():
        pb = S['pbp'][r['game']]
        nm = pb.loc[pb['PLAYER1_ID'] == r['passer'], 'PLAYER1_NAME'].dropna()
        lines.append(f"- {r['game']} Q{r['quarter']} clock {int(r['quarter_time'] // 60)}:{int(r['quarter_time'] % 60):02d} "
                     f"releaser={nm.iloc[0] if len(nm) else r['passer']} outcome={r['trk_outcome']} tag={r['trk_tag']} "
                     f"live={bool(r['live'])} zmax={r['zmax']:.1f} disp={r['ball_disp']:.0f}ft v0={r['init_speed']:.0f}ft/s "
                     f"hold={r['hold_frames']}fr\n    pbp(-3..+8s): {r['pbp_ctx'] or '(none)'}")
    txt = '\n'.join(lines)
    with open(path or f'{RES}/t2_manual_sheet.txt', 'w') as f:
        f.write(txt)
    log(f'\n  manual-look sheet ({len(s)} examples) written to {path or RES + "/t2_manual_sheet.txt"}')


# ------------------------------------------------------------------------------- T5
def stage_t5(S):
    log('\n=== T5 (non-pass turnovers): scored by the failure model trained on the 401 OTHER dev games ===')
    ext, fa, solo = S['ext'], S['feat']['actual'], S['solo']
    npe = S['np_events']
    log('  pbp non-pass turnover events matched to a hold and where they ended up in the release table:')
    tab = pd.crosstab(npe['kind'], npe['in_table'].replace('', 'unmatched'), margins=True)
    log(tab.to_string())
    log('  by kind: events, matched, matched with a flight to an opponent/OOB (flight_outcome) :')
    fo = npe.assign(flight=npe['flight_outcome'].str.startswith('none').map({True: 'no flight', False: 'flight'}))
    fo.loc[npe['matched'] == 0, 'flight'] = 'unmatched'
    log(pd.crosstab(fo['kind'], fo['flight'], margins=True).to_string())
    sg = ext['game'].unique()
    Xtr = baseline_other_games(sg)
    ytr = Xtr['fail'].values
    E = ext[['src', 'trk_fail', 'trk_tag', 'live', 'np_kind', 'game']].join(solo, how='left')
    E = E.join(fa[FEATS].add_prefix('f_'), how='left')
    live_comp = E[(E['src'] == 'completed') & (E['trk_fail'] == 0) & E['live']]
    groups = {'pbp bad-pass failures (reference)': E['src'] == 'pbp_fail',
              'tracking-only failures, live (reference)': (E['src'] == 'trk_only') & E['live']}
    for k in ('lost_ball', 'travel', 'off_foul', 'other_to'):
        groups[f'np {k}: new row (no flight to opp/OOB), live'] = (E['np_kind'] == k) & (E['src'] == f'np_{k}') & E['live']
        groups[f'np {k}: also a tracking-only flight, live'] = (E['np_kind'] == k) & (E['src'] == 'trk_only') & E['live']
    out = {}
    for label, cols, prefix in (('FULL model (15 features; rows need a heading target)', FEATS, 'f_'),
                                ('PASSER-ONLY model (5 features, every matched row)', PASSER, '')):
        log(f"\n  {label}")
        for model in ('logit', 'hgb'):
            m = make_model(model).fit(Xtr[cols].values, ytr)
            Xn = live_comp[[prefix + c for c in cols]]
            Xn = Xn.dropna(how='all')
            s_neg = m.predict_proba(Xn.values)[:, 1]
            log(f"   [{model}] reference = {len(Xn)} live completions")
            for gname, mask in groups.items():
                g = E[mask][[prefix + c for c in cols]].dropna(how='all')
                if len(g) < 3:
                    log(f"      {gname:58s} n={len(g)} (too few)")
                    continue
                sg_ = m.predict_proba(g.values)[:, 1]
                a = roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(Xn))], np.r_[sg_, s_neg])
                pct = float(np.mean([(s_neg < v).mean() for v in sg_]))
                log(f"      {gname:58s} n={len(g):5d}  AUC vs live completions {a:.3f}  mean percentile {pct:.3f}")
                out.setdefault(label, {}).setdefault(model, {})[gname] = dict(n=len(g), auc=a, pct=pct)
    save_json('t5_np', out)


STAGES = dict(t1=stage_t1, t2=stage_t2, t5=stage_t5)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stages', nargs='+', choices=list(STAGES))
    a = ap.parse_args()
    S = prep()
    for s in a.stages:
        t0 = time.time()
        STAGES[s](S)
        log(f'[{s} took {time.time() - t0:.0f}s]')
