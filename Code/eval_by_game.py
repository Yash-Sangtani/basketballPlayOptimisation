"""Grouped-by-game evaluation of failed-pass prediction. Usage: python eval_by_game.py <pass_features dir>
Positive class = failed pass (pass_successful == 0). GroupKFold(5), groups = game (pickle file name)."""
import sys, glob, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

d = sys.argv[1] if len(sys.argv) > 1 else '../Tracking_Data/pass_features'
parts = []
for f in sorted(glob.glob(os.path.join(d, '*.pkl'))):
    df = pd.read_pickle(f)
    df['game'] = os.path.basename(f)[:-4]
    parts.append(df)
data = pd.concat(parts, ignore_index=True)
data['fail'] = (data['pass_successful'] == 0).astype(int)
print(f"{len(parts)} games, {len(data)} passes, {data['fail'].sum()} failures")

models = {
    # constant score everywhere -> ROC exactly 0.5 and PR-AUC exactly = prevalence (no-skill reference)
    'no-skill (prevalence)': lambda: DummyClassifier(strategy='constant', constant=0),
    'logistic (balanced)': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                                 LogisticRegression(class_weight='balanced', max_iter=1000)),
    'gradient boosting': lambda: make_pipeline(SimpleImputer(strategy='median'),
                                               GradientBoostingClassifier(subsample=0.5, random_state=0)),
}

def boot_ci(y, p, g, metric, n=200, seed=0):
    """95% CI of a pooled metric, resampling whole games with replacement."""
    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(y))).groupby(g.values).apply(np.array)
    out = []
    for _ in range(n):
        pick = np.concatenate(idx.iloc[rng.integers(0, len(idx), len(idx))].values)
        if y[pick].min() == y[pick].max():
            continue
        out.append(metric(y[pick], p[pick]))
    return np.percentile(out, [2.5, 97.5])

for label, sub in [('ALL rows', data), ('backcourt == 0 rows only', data[data['backcourt'] == 0])]:
    sub = sub.reset_index(drop=True)
    feats = [c for c in sub.columns if c not in ('pass_successful', 'fail', 'game')]
    if label != 'ALL rows':
        feats.remove('backcourt')               # constant 0 in this subset
    X, y, g = sub[feats], sub['fail'].values, sub['game']
    prev = y.mean()
    print(f"\n=== {label}: {len(sub)} passes, {y.sum()} failures, prevalence {prev:.4f}, {g.nunique()} games, "
          f"{len(feats)} features ===")
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    print("failures per fold:", [int(y[te].sum()) for _, te in folds])
    print(f"{'model':22s} {'pooled ROC':>10s} {'ROC 95% CI':>14s} {'fold ROC mean+-sd':>19s} "
          f"{'pooled PR':>9s} {'PR 95% CI':>16s} {'fold PR mean+-sd':>18s}")
    for name, make in models.items():
        t0 = time.time()
        oof = np.zeros(len(sub)); roc_f, pr_f = [], []
        for tr, te in folds:
            m = make().fit(X.iloc[tr], y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            roc_f.append(roc_auc_score(y[te], oof[te])); pr_f.append(average_precision_score(y[te], oof[te]))
        roc, pr = roc_auc_score(y, oof), average_precision_score(y, oof)
        rl, rh = boot_ci(y, oof, g, roc_auc_score); pl, ph = boot_ci(y, oof, g, average_precision_score)
        print(f"{name:22s} {roc:10.3f} {f'[{rl:.3f},{rh:.3f}]':>14s} {f'{np.mean(roc_f):.3f}+-{np.std(roc_f):.3f}':>19s} "
              f"{pr:9.4f} {f'[{pl:.4f},{ph:.4f}]':>16s} {f'{np.mean(pr_f):.4f}+-{np.std(pr_f):.4f}':>18s}   ({time.time()-t0:.0f}s)")
    print(f"(PR-AUC of a no-skill model = prevalence = {prev:.4f})")
