"""
Acceptance checks for the pass pipeline (detection, labels, features).

Run from Code/ (it extracts each game into ./temp, so run games sequentially,
never two copies at once):

    python3 validate_passes.py 12.12.2015.GSW.at.MIL 12.30.2015.PHI.at.SAC
    python3 validate_passes.py --target-mode heading 12.12.2015.GSW.at.MIL

What it reports, per game and pooled:
  detection  - completed/failed counts, assist recall (pbp assist = passer
               PLAYER2 -> scorer PLAYER1), precision proxies, share of
               passes by the minority team inside a shot-clock window
  labels     - how many pbp bad-pass turnovers were matched, and why not
  heading    - coverage/accuracy of the receiver-heading rule on completions
  leakage    - single-feature AUC, class-constant features, backcourt x label,
               grouped cross-validated AUC (by game, or by quarter for one game)
  attack     - the two teams attack opposite baskets and swap at halftime

Thresholds are not tuned to hit a number; read the output, don't gate on it.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from passtracking import features


def parse_game(name):
    """'12.12.2015.GSW.at.MIL' -> (date, home, away)."""
    c = name.replace('.7z', '').split('.')
    return '.'.join(c[:3]), c[5], c[3]


def assist_recall(pt):
    pbp, pa, fr = pt.pbp, pt.pass_attempt, pt._fr
    ast = pbp.loc[(pbp['EVENTMSGTYPE'] == 1) & (pbp['PLAYER2_ID'] > 0)]
    comp = pa.loc[pa['pass_successful'] == 1]
    hit = exact_time = 0
    miss_reasons = {'assister_not_holding': 0, 'scorer_not_holding': 0, 'other': 0}
    for _, ev in ast.iterrows():
        passer, scorer, period, e = int(ev['PLAYER2_ID']), int(ev['PLAYER1_ID']), int(ev['PERIOD']), float(ev['Qtime'])
        c = comp.loc[(comp['passer'] == passer) & (comp['target'] == scorer) & (comp['quarter'] == period)]
        recv_clock = fr['qt'][c['recv_frame'].astype(int).values] if len(c) else np.array([])
        d = recv_clock - e
        if ((d >= -1.0) & (d <= 8.0)).any():
            hit += 1
            continue
        near = [s for s in pt.segments if fr['quarter'][s['start']] == period
                and -2.0 <= fr['qt'][s['end']] - e <= 10.0]
        if not any(s['holder'] == passer for s in near):
            miss_reasons['assister_not_holding'] += 1
        elif not any(s['holder'] == scorer for s in near):
            miss_reasons['scorer_not_holding'] += 1
        else:
            miss_reasons['other'] += 1
    return len(ast), hit, miss_reasons


def mixed_team_share(pa):
    """Completed passes grouped into shot-clock windows (a window ends when the
    clock jumps up). Returns (share of passes by the minority team, share of
    windows holding both teams, mixed windows with one team switch, mixed
    windows with several). A single switch is a possession change the clock
    check can't see; repeated switches mean detection is flipping teams.
    Passes with no shot clock (clock off late in a quarter) are excluded."""
    comp = pa.loc[(pa['pass_successful'] == 1) & pa['shot_clock'].notna()].sort_values('frame')
    win = ((comp['shot_clock'].diff() > 0.5) | (comp['quarter'].diff() != 0)).cumsum()
    g = comp.assign(win=win).groupby('win')['team_passer']
    size = g.size()
    minority = g.apply(lambda s: (len(s) - s.value_counts().max()) / len(s))
    switches = g.apply(lambda s: int((s.values[1:] != s.values[:-1]).sum()))
    mixed = g.nunique() > 1
    return ((minority * size).sum() / size.sum(), mixed.mean(),
            int((mixed & (switches == 1)).sum()), int((mixed & (switches > 1)).sum()))


def precision_proxies(pt):
    pa, fr = pt.pass_attempt, pt._fr
    comp = pa.loc[pa['pass_successful'] == 1].sort_values('frame')
    z_recv = fr['ball'][comp['recv_frame'].astype(int).values, 2]
    gaps = fr['ut'][comp['frame'].values[1:].astype(int)] - fr['ut'][comp['recv_frame'].values[:-1].astype(int)]
    dist = np.hypot(*(np.array([fr['xy'][r][fr['pid'][r] == p][0] - fr['xy'][r][fr['pid'][r] == t][0]
                                for r, p, t in zip(comp['frame'].astype(int), comp['passer'], comp['target'])]).T))
    return dict(n=len(comp),
                recv_ball_z_gt_9=float((z_recv > 9).mean()),
                flight_lt_0_4s=float((comp['flight_s'] < 0.4).mean()),
                next_release_within_0_5s=float((gaps < 0.5).mean()) if len(gaps) else np.nan,
                passer_target_lt_6ft=float((dist < 6).mean()))


def heading_quality(pa):
    comp = pa.loc[pa['pass_successful'] == 1]
    have = comp['target_heading'].notna()
    return float(have.mean()), float((comp.loc[have, 'target_heading'] == comp.loc[have, 'target']).mean())


def attack_check(pt):
    out = {}
    for q in sorted({q for (_, q) in pt.attack_basket}):
        out[q] = tuple(pt.attack_basket.get((t, q)) for t in (pt.home_id, pt.away_id))
    ok = all(a is not None and b is not None and a != b for a, b in out.values())
    flips = (1 not in out or 3 not in out) or (out[1][0] != out[3][0])
    return out, ok, flips


def leakage_report(F, M):
    X, y = F.drop(columns='pass_successful'), F['pass_successful'].astype(int)
    print(f"  rows={len(F)}  completed={int(y.sum())}  failed={int((1 - y).sum())}  failure rate={1 - y.mean():.1%}")
    print("  single-feature AUC for pass_successful (|auc-0.5|+0.5; >=0.85 = suspicious):")
    aucs = {}
    for c in X.columns:
        v = X[c]
        if v.nunique(dropna=True) <= 1 or y.nunique() < 2:
            aucs[c] = np.nan
            continue
        mask = v.notna()
        a = roc_auc_score(y[mask], v[mask])
        aucs[c] = max(a, 1 - a)
    for c, a in sorted(aucs.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0)):
        print(f"    {c:32s} {a:.3f}" + ("   <-- suspicious" if a >= 0.85 else ""))
    const = [c for c in X.columns for k in (0, 1)
             if (y == k).any() and X.loc[y == k, c].nunique(dropna=True) <= 1]
    print("  features constant within a class:", sorted(set(const)) or "none")
    print("  backcourt x label (rows; failure rate per flag value):")
    ct = pd.crosstab(F['backcourt'], F['pass_successful'])
    ct['failure_rate'] = ct.get(0.0, 0) / ct.sum(axis=1)
    print(ct.to_string().replace('\n', '\n    '))
    #Grouped CV: by game when pooled over several, else by quarter.
    groups = M['game_id'] if M['game_id'].nunique() > 1 else M['quarter']
    print(f"  grouped CV ({'game' if M['game_id'].nunique() > 1 else 'quarter'}), positive class = FAILED pass:")
    for name, model in [('logistic', make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                                    LogisticRegression(max_iter=1000, class_weight='balanced'))),
                        ('gboost', make_pipeline(SimpleImputer(strategy='median'),
                                                 GradientBoostingClassifier(random_state=0)))]:
        for label, cols in [('all features', list(X.columns)),
                            ('without backcourt', [c for c in X.columns if c != 'backcourt'])]:
            p = np.full(len(F), np.nan)
            for tr, te in LeaveOneGroupOut().split(X, 1 - y, groups):
                if (1 - y.iloc[tr]).nunique() < 2:
                    continue
                model.fit(X.iloc[tr][cols], 1 - y.iloc[tr])
                p[te] = model.predict_proba(X.iloc[te][cols])[:, 1]
            ok = ~np.isnan(p)
            if ok.sum() and (1 - y)[ok].nunique() == 2:
                print(f"    {name:9s} {label:18s} ROC {roc_auc_score((1 - y)[ok], p[ok]):.3f}"
                      f"  PR {average_precision_score((1 - y)[ok], p[ok]):.3f}  (PR baseline {(1 - y)[ok].mean():.3f})")


def main():
    args = sys.argv[1:]
    mode = 'actual'
    if args[:1] == ['--target-mode']:
        mode, args = args[1], args[2:]
    if not args:
        args = ['12.12.2015.GSW.at.MIL']
    all_f, all_m = [], []
    for name in args:
        date, home, away = parse_game(name)
        print(f"\n=== {name}  (target_mode={mode}) ===")
        f = features(date, home, away, target_mode=mode)
        f.features_extraction()
        pt, pa = f.passes, f.passes.pass_attempt
        comp, fail = pa['pass_successful'].sum(), (pa['pass_successful'] == 0).sum()
        print(f"detection: completed={comp} failed={fail}  completed per quarter={pa.loc[pa.pass_successful == 1].groupby('quarter').size().to_dict()}")
        n_ast, hit, miss = assist_recall(pt)
        print(f"assist recall: {hit}/{n_ast} = {hit / max(n_ast, 1):.0%}   misses by cause: {miss}")
        print("precision proxies:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in precision_proxies(pt).items()})
        minor, both, one, many = mixed_team_share(pa)
        print(f"mixed-team windows: minority-team share={minor:.1%}, windows with both teams={both:.0%} "
              f"(clean single switch={one}, repeated switches={many}; repeated = detection flipping teams)")
        print("failure matching (pbp bad-pass/OOB-bad-pass turnovers):", pt.failure_report,
              " outcomes:", pa.loc[pa.pass_successful == 0, 'outcome'].value_counts().to_dict())
        cov, acc = heading_quality(pa)
        print(f"heading rule on completed passes: coverage={cov:.0%}  agrees with actual receiver={acc:.0%}")
        print(f"failed passes with a heading target: {pa.loc[pa.pass_successful == 0, 'target'].notna().mean():.0%}")
        out, ok, flips = attack_check(pt)
        print(f"attack baskets (home,away) per quarter: {out}  opposite each quarter={ok}  swap at half={flips}")
        print(f"feature rows kept: {len(f.features)}/{len(pa)}")
        all_f.append(f.features)
        all_m.append(f.meta)
    F, M = pd.concat(all_f, ignore_index=True), pd.concat(all_m, ignore_index=True)
    print("\n=== leakage / signal checks (pooled over the games above) ===")
    leakage_report(F, M)


if __name__ == '__main__':
    main()
