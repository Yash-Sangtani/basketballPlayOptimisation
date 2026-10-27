# WS2: grouped-by-game evaluation of failed-pass prediction (dev games only)

Run 2026-10-27 on Linux (4 cores, 15 GB), baseline dataset code version `a846f643`.
Command: `cd Code && ./heavy.sh ../.venv/bin/python pass_model.py --split dev --models logistic,gboost,mlp --out ../reports/ws2-out --plot ../reports/ws2-out/ws2.png`
(10.5 min; log is `logs/ws2_dev.log`, git-ignored). Raw tables: `reports/ws2-out/` (`tables.md`, CSVs, `ws2.png`, `oof.npz` out-of-fold scores (not committed, 7 MB each)).
Two ablations, same command with `--population frontcourt` (logistic, gboost) and `--feature-set passer` (logistic, gboost, mlp): `reports/ws2-out-frontcourt/`, `reports/ws2-out-passer/`.
The frozen test games were not loaded (`requested 501 games`, dev list only). One seed (0), 5 folds; CIs are game-clustered bootstrap (1000 resamples, 300 for the ablations) and capture test-sampling noise only, not refit-to-refit variability.

## Data
501 dev games, 245,371 passes, 4,051 failures (1.651%). Every game had a pickle, meta file and `ok` status row, one code version. No label/outcome mismatches. NaN rates: `shot_clock` 4.85%, the angle and speed columns <= 0.2%. No feature pair with |r| > 0.95.
Positive class = failed pass; every score is P(fail). No row filtering (NaNs are imputed inside each fold; the old `backcourt == 0` + `dropna` filter is now the separate "frontcourt" population).

## Main result (all passes, 15 features)

| model | ROC-AUC [95% CI] | PR-AUC [95% CI] (no-skill 0.0165) |
|---|---|---|
| simple baseline (shot_clock + dist, logistic) | 0.684 [0.676, 0.693] | 0.0405 [0.0377, 0.0440] |
| logistic (class-balanced) | 0.851 [0.846, 0.857] | 0.1026 [0.0968, 0.1091] |
| gradient boosting | 0.854 [0.848, 0.859] | 0.1107 [0.1044, 0.1177] |
| MLP (Keras, early stopping on held-out games) | **0.869 [0.864, 0.874]** | **0.1303 [0.1226, 0.1394]** |

Paired differences vs logistic: gboost +0.002 ROC [-0.000, 0.005], +0.0081 PR [0.0032, 0.0129]; MLP +0.017 ROC [0.015, 0.020], +0.0277 PR [0.0229, 0.0328]. Per-fold ROC is stable (logistic 0.852 +- 0.004, MLP 0.869 +- 0.006). Measured: the numbers reproduce the earlier all-games result (0.851 / 0.856) and the MLP is the best model here. The MLP's advantage is the only new finding; the earlier 0.83 from `pass_model.py` was on a filtered, row-split population and is not comparable.

**By failure type** (against all completions; MLP ROC / PR, no-skill PR in brackets): intercepted (2,946) 0.872 / 0.102 [0.012]; out of bounds (617) 0.853 / 0.017 [0.003]; fumbled (488) 0.872 / 0.025 [0.002]. All three types are separable at ROC 0.84-0.87 for every model. Intercepted passes are 73% of failures and dominate the PR-AUC.

## Calibration
Raw outputs are not probabilities: the logistic and MLP use balanced class weights (mean prediction 0.32 and 0.29 against a 0.0165 base rate; Brier worse than predicting the base rate, BSS -9 and -8). Gradient boosting is unweighted and is roughly calibrated raw (mean 0.0164, BSS +0.036). After isotonic calibration on held-out calibration games inside each training fold, all models are well calibrated by decile (reliability tables in `tables.md` / the log) and the skill scores are:

| model | Brier skill score, calibrated [95% CI] | log loss (no-skill 0.0841) |
|---|---|---|
| simple | 0.009 [0.008, 0.011] | 0.0805 |
| logistic | 0.049 [0.045, 0.053] | 0.0689 |
| gboost | 0.055 [0.051, 0.059] | 0.0684 |
| MLP | 0.065 [0.060, 0.071] | 0.0666 |

So the models explain about 5-7% of the Brier variance of a rare event: real but small. Calibration costs about 0.002-0.004 PR-AUC (isotonic ties); use the raw scores for ranking and the calibrated ones for probabilities. In the top calibrated decile, 8.7-9.7% of passes are labelled failures (observed rate equals predicted), i.e. even the best-scored 10% of passes are about 90% completions.

## Ablations (measured)

| population / features | model | ROC-AUC [95% CI] | PR-AUC [95% CI] | no-skill PR |
|---|---|---|---|---|
| frontcourt only (173,157 passes, 3,703 failures) | logistic | 0.828 [0.821, 0.835] | 0.1148 [0.1075, 0.1230] | 0.0214 |
| | gboost | 0.831 [0.824, 0.838] | 0.1173 [0.1099, 0.1250] | |
| | simple | 0.621 [0.610, 0.630] | 0.0432 | |
| passer-side features only (5: shot_clock, backcourt, passer_basket_dist, closest_def_dist_passer, passer_speed) | logistic | 0.756 [0.748, 0.763] | 0.0483 [0.0457, 0.0517] | 0.0165 |
| | gboost | 0.761 [0.754, 0.768] | 0.0469 [0.0446, 0.0498] | |
| | MLP | 0.761 [0.754, 0.768] | 0.0491 [0.0465, 0.0523] | |

- Dropping the 5 features that do not depend on who the receiver is (i.e. keeping only the passer side) loses most of the signal: ROC 0.85 -> 0.76, PR-AUC 0.10-0.13 -> 0.048. The 10 target/lane-side features carry most of the discrimination.
- That matters for the selection caveat (WS3): in this dataset every failure has an *inferred* receiver (`target_inferred` is True for all failures and False for all completions, see `evaluation.py`), so the target-side features are built from a different kind of target in the two classes. This is inferred, not shown to explain the signal; `reports/ws3-selection.md` tests it.
- Frontcourt-only ROC is lower (0.83) because the backcourt rows are easy negatives with a different failure rate (0.5% vs 2.1%); the PR-AUC is about the same.

## What this does and does not show
- Measured: failed passes, as labelled by the detector, are ranked 0.85-0.87 ROC / 5-8x the base rate in PR-AUC on games the model never saw, with no leakage from labels in the features (whitelisted columns) and no game straddling a split.
- Not shown: that this is pass *difficulty*. The label is "pbp bad-pass turnover matched to a flight", the negatives are hold-to-hold completions. See WS3.
- Accuracy / F1 are deliberately not reported (always guessing "completed" scores 98.3%).
- Not run: the date-ordered fold mode (`--fold-mode date`), more seeds. Test split untouched.
