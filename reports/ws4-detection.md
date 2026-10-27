# WS4: detection and dataset changes, one regeneration, dev re-evaluation, and the single test run

Run 2026-10-27. Tuning and comparison used the 501 dev games only. The 130 test games were scored once, at the very end, with a configuration fixed beforehand.

## 1. What I tried to improve in the detector, and what I changed
**Threshold sweep (dev, no change made).** `Code/analysis/ws4_detect_sweep.py` re-runs detection on 24 dev games (a stride sample of the dev list, excluding the 4 games that shaped the original thresholds) with different values of `HOLD_Z`, `MIN_DISP`, `MIN_HOLD_FRAMES`, `HOLD_DIST`, `MIN_FLIGHT_S` (output `reports/ws4-sweep.json`, `logs` not kept). Baseline: 11,576 completed passes, 174 failures, assist recall 820/1057 = 77.6%.

| variant | completed | assist recall | passer-target < 6 ft | flights < 0.4 s |
|---|---|---|---|---|
| baseline | 11,576 | 77.6% | 0.7% | 8.4% |
| HOLD_Z 10 | 11,610 | 77.5% | 0.7% | 8.6% |
| MIN_DISP 3 | 12,093 | 79.1% | 1.3% | 11.3% |
| MIN_HOLD_FRAMES 3 | 11,674 | 77.8% | 0.7% | 8.7% |
| HOLD_DIST 3.5 | 11,481 | 77.7% | 0.6% | 9.1% |
| hand-offs (MIN_FLIGHT_S 0.08, MIN_DISP 2) | 12,551 | 80.2% | 2.6% | 14.0% |
| all four loosened | 12,088 | 79.2% | 1.3% | 12.7% |

No variant improves assist recall by more than 2.6 points, and the ones that gain do it by adding very short, hand-off-like flights (more flights under 0.4 s, more passer-target pairs under 6 ft), i.e. a different kind of event. Looking at 5 missed assists in one game by hand, the misses look like hand-offs (the next player's hold starts within 0.1-0.5 s of the release), passes with a teammate touch in between, and holds shorter than the minimum (inferred from hold timings, not video), not threshold effects. **I changed no detection threshold.** Assist recall stays at 67-81%, repeated-flip windows and the 2-of-15 unmatched pbp bad passes are unchanged. This is a negative result: parameter tweaks do not buy detection recall here.

**What I did change (it needed the one regeneration).** The WS3 findings pointed at how the dataset is defined, not at the detector:
1. `features` now defaults to the heading receiver rule for both classes (`target_mode='heading'`), because failures never have a real receiver and the true receiver for completions makes the classes differ in kind (WS3 T1: about 0.025 ROC and 40% of PR-AUC). The main pickles are heading-mode; the old behaviour is written to `Tracking_Data/pass_features/actual/` (+ `actual/meta/`) so both can be compared.
2. Each pass carries `clock_running`, `inbound` and `live` (live = clock running and the passer in bounds); 16.8% of the passes are not live (inbound and dead-ball passes), 17.6% of completions in the WS3 sample against 2.4% of failures. Rows are kept; `pass_model.py --population live|live_frontcourt` filters.
3. No change to which passes are detected.

**Regeneration (the one budgeted).** Code version `a846f643` -> `fc2ceb29`; 631 games ok, the same 5 known-bad games failed, one code version, 306,782 passes and 5,025 failures as before. The `actual/` copy equals the old baseline pickles in all 631 games (float tolerance rtol 1e-4; the old numbers were made on a Mac). Heading-mode rows: 299,995 (6,787 completions, 2.2%, have no heading target and are dropped), failures 5,025, live share 83.2%. Old dataset kept locally as `Tracking_Data/pass_features_baseline_a846f643/` (git-ignored; also on the `baseline-results` branch).

## 2. Dev re-evaluation (501 dev games, GroupKFold(5) by game, 500 bootstrap resamples)
| setting | model | ROC-AUC [95% CI] | PR-AUC [95% CI] | no-skill PR |
|---|---|---|---|---|
| old: actual receiver, all passes (`ws2-eval.md`) | logistic / gboost / MLP | 0.851 / 0.854 / 0.869 | 0.103 / 0.111 / 0.130 | 0.0165 |
| **new: heading rule, all passes** (239,894 passes, 4,051 failures) | logistic | 0.825 [0.819, 0.831] | 0.0753 [0.0709, 0.0803] | 0.0169 |
| | gboost | 0.828 [0.822, 0.834] | 0.0754 [0.0715, 0.0797] | |
| | MLP | 0.843 [0.837, 0.848] | 0.0833 [0.0784, 0.0887] | |
| **new: heading rule, live passes** (200,131 passes, 3,927 failures) | logistic | 0.808 [0.801, 0.814] | 0.0776 [0.0732, 0.0827] | 0.0196 |
| | gboost | 0.811 [0.805, 0.817] | 0.0792 [0.0749, 0.0840] | |
| | MLP | 0.828 [0.821, 0.834] | 0.0882 [0.0831, 0.0942] | |
| live, passer-side features only | logistic / gboost | 0.728 / 0.733 | 0.051 / 0.049 | 0.0196 |

Measured: removing the receiver asymmetry and the dead-ball/inbound passes lowers the headline from ROC 0.85-0.87 to 0.81-0.83 and PR-AUC from 0.10-0.13 to 0.078-0.088 (about 4-4.5x the live base rate rather than 6-8x), matching what WS3 T1 predicted on 100 games (ROC -0.025, PR -40%). Calibrated Brier skill drops from 0.05-0.065 to 0.03-0.04 (live). Model ranking is unchanged: MLP > gboost ~ logistic (MLP - logistic +0.020 ROC [0.017, 0.023], +0.011 PR [0.008, 0.014]). By kind of failure (live, MLP): intercepted 0.830, fumbled 0.833, out of bounds 0.816 ROC. The simple shot-clock + distance baseline falls from 0.68 to 0.65 (all) / 0.61 (live), i.e. it was partly exploiting the same asymmetry.

## 3. The single test run
Fixed in advance of the run (from the dev results, not from test): heading features, **live** population, logistic / gboost / MLP, fit on all 501 dev games, calibration games held out of the fit as in dev mode, 1000 bootstrap resamples. Command: `pass_model.py --split test --confirm-frozen-test --models logistic,gboost,mlp --population live`. It was run once. Output: `reports/ws4-test-live/`.

130 test games, 49,550 live passes, 938 failures (1.89%), all code version `fc2ceb29`:

| model | ROC-AUC [95% CI] | PR-AUC [95% CI] (no-skill 0.0189) | calibrated Brier skill [95% CI] |
|---|---|---|---|
| simple (shot clock + distance) | 0.594 [0.576, 0.611] | 0.0255 [0.0230, 0.0293] | 0.000 [-0.002, 0.002] |
| logistic | 0.804 [0.791, 0.817] | 0.0725 [0.0640, 0.0828] | 0.029 [0.022, 0.034] |
| gboost | 0.809 [0.795, 0.821] | 0.0752 [0.0662, 0.0862] | 0.032 [0.026, 0.038] |
| MLP | 0.820 [0.806, 0.831] | 0.0808 [0.0705, 0.0924] | 0.035 [0.028, 0.042] |

This agrees with dev within its intervals (dev live: 0.808 / 0.811 / 0.828 ROC, 0.078 / 0.079 / 0.088 PR), so the dev tuning did not overfit the test games. Caveats: `01.12.2016.HOU.at.MEM` is in the test list and was looked at early in the project (slightly contaminated, 1 of 130 games); one seed; intervals reflect test-set sampling only. The test games were not used to choose anything; the earlier all-passes / actual-receiver configuration was not scored on test.

## 4. What this means
- The detector-labelled "failed pass" is predictable from release-time geometry at ROC about 0.81-0.82 and PR-AUC about 0.075-0.08 on held-out games with the receiver rule symmetric: real but modest.
- It is not pass difficulty (WS3 T5: non-pass turnovers score as high), so the models are risk rankings for a release, not a pass-quality model. That claim stands after the regeneration; T5 was not repeated because the heading-mode features it used are identical to the regenerated ones (checked on one game).
- Not done: detection recall improvements (none found by threshold changes); multiple seeds; the date-ordered folds; a data-driven pass-vs-shoot value (see `ws5b-value-review.md`).
