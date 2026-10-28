# WS3: is the failed-pass signal real, or is it selection? (dev games only)

Run 2026-10-27, Linux, baseline dataset `a846f643`. Only `Code/splits/dev_games.txt` was read; the frozen test list was never opened.
Scripts: `Code/analysis/ws3_baseline_tests.py` (all 501 dev games, uses the baseline pickles), `Code/analysis/ws3_extract.py` + `ws3_sample_tests.py` (a 100-game dev sample, `Code/splits/ws3_sample_games.txt`, re-extracted with extra annotations and both target modes without editing `passtracking.py`). Outputs: `reports/ws3-out/` (JSON, full logs, the manual-look sheet).
Positive class = failed pass. "hgb" = sklearn HistGradientBoosting, "logit" = balanced logistic; GroupKFold(5) by game; CIs are game-clustered bootstraps (100-300 resamples, so slightly rough).

**Parity check.** The 100-game re-extraction reproduces the baseline feature rows for all 100 games, but only to float tolerance (rtol 1e-4), not bit for bit: the baseline was built on a Mac, this is Linux, and 16% of the `dist` values differ in the last float32 digit. I loosened the script's `check_exact=True` for that reason. Not a concern for the conclusions below.

## Bottom line

1. **The signal is not a statistical artifact.** Shuffling labels within each game gives ROC about 0.49-0.50 (T3); the model's ranking holds inside strata of identical passer context (T6b) and against same-team, same-game, +-8 s completions (T6).
2. **But it is not a clean measure of pass difficulty.** Three findings, all measured:
   - **It predicts non-pass turnovers as well as, or better than, bad passes** (T5). Lost balls, travels and offensive fouls, which are not passes at all, score ROC 0.85-0.89 under the model trained on bad-pass labels, against 0.825-0.848 for the bad passes it was trained on. A passer-only model (no receiver at all) separates them from completions at ROC 0.82-0.87 too, but separates the real bad passes only at 0.72-0.73. The model has learned "this release happens in a turnover-prone situation" (close to the basket, defender on the passer, moving ball handler), not a property of the pass.
   - **The receiver rule costs about 0.025 ROC (7-8% of the edge over chance) and about 40% of the PR-AUC** (T1). Failures never have a real receiver (the receiver is inferred from the ball heading); completions use the true one. Using the same heading rule for both classes drops ROC 0.857 -> 0.829 (hgb) and PR-AUC 0.121 -> 0.071 on identical rows and folds. The headline PR-AUC 0.10-0.14 is therefore overstated; a symmetric-target figure is about 0.07 (on the 100-game sample).
   - **Where on the court and when carries most of it** (T4, T7): five phase/location features alone give ROC 0.82 (hgb); the seven defender features alone give 0.83; the simple shot_clock + distance baseline gives 0.71.
3. **Tracking-only failure labels are not a usable alternative label** (T2): 3,546 candidate flights in the 100 games (35/game vs 8.6 pbp failures/game), but 75% sit in shot, rebound or dead-ball contexts (the other 25%, 892 rows, are the plausible candidates; 590 of them have a heading target and can be scored).
4. **Decision.** Read the metrics as "how well the features rank detector-labelled turnover-like releases", not as pass difficulty or pass quality. The model is useful as a risk ranking for a release, not as a pass-quality model, and any value model should not use it as the pass-success term without the changes in WS4 (same receiver rule for both classes, live-ball population, non-pass turnovers kept as a negative control).

## T3 - shuffle control (501 dev games)
Labels shuffled within each game (each game's base rate kept), 3 seeds, logit and hgb, out-of-fold:

| | pooled ROC | PR-AUC (base rate 0.0165) |
|---|---|---|
| logit (seeds 1, 2, 3) | 0.498, 0.485, 0.504 | 0.0165, 0.0159, 0.0169 |
| hgb (seeds 1, 2, 3) | 0.497, 0.492, 0.487 | 0.0163, 0.0162, 0.0159 |

The pre-set bar (every ROC within 0.5 +- 0.015 and PR within +-0.002 of the base rate) **fails by 0.0004 on one of six cells** (logit seed 2: 0.4846); the mean of the per-fold ROCs is 0.485-0.504 in all cells. Pooled out-of-fold ROC on random labels sits slightly below 0.5 in five of six cells (0.485-0.498), which is the known small negative bias of cross-validation on noise (inferred, not separately tested). Real signal is 0.85, so this does not change the conclusion; I report it as a marginal fail of the stated bar, not a pass.

## T4 - feature subsets (501 dev games, hgb / logit ROC, PR)
| features | hgb ROC | hgb PR | logit ROC |
|---|---|---|---|
| shot_clock | 0.651 | 0.028 | 0.647 |
| dist | 0.629 | 0.037 | 0.618 |
| shot_clock + dist | 0.707 | 0.046 | 0.684 |
| + backcourt | 0.718 | 0.048 | 0.692 |
| passer-only (5, no receiver) | 0.763 | 0.049 | 0.756 |
| phase / location only (5) | 0.822 | 0.088 | 0.791 |
| defenders only (7) | 0.834 | 0.091 | 0.822 |
| all minus defenders | 0.834 | 0.102 | 0.799 |
| all minus phase | 0.857 | 0.124 | 0.842 |
| all 15 | **0.871** | **0.140** | 0.851 |

Paired (game bootstrap): all 15 minus shot_clock+dist = +0.164 ROC [0.155, 0.171] and +0.094 PR [0.087, 0.101] (hgb). Removing the receiver-dependent features (passer-only) costs 0.108 ROC and 0.091 PR. Note the tuned hgb (0.871 / 0.140) is better than the sklearn GradientBoosting used in `reports/ws2-eval.md` (0.854 / 0.111, default settings) and than the MLP (0.869 / 0.130); the model ranking in WS2 is partly a tuning artifact and should not be over-read.

## T7 - what the model uses
Permutation importance on held-out folds (hgb, drop in ROC when shuffled): defender features as a group 0.227; lane defenders (`closest_def_trajectory`, its angle, `lane_n_def_5ft`) 0.137; `closest_def_trajectory` alone 0.073; phase/location group 0.048; `closest_def_dist_ball_end` 0.039. Univariate directions are sensible and consistent with a defender-on-the-line story: failures have median defender-to-pass-line distance 1.6 ft vs 4.3 ft for completions, median defender distance to the passer 3.8 vs 6.7 ft, release 19.7 vs 29.9 ft from the basket, shot clock 14.2 vs 17.1 s. Whether a defender on the line causes a failure or is the interceptor by construction is exactly what the data cannot separate: an intercepted flight needs a defender near the line by definition, and 73% of failures are interceptions.

## T6 - matched negatives
Each failure is compared with completions of the same team in the same game within +-8 s (3,087 of 4,051 failures have at least one; mean 1.8 matches). Concordance = share of matched completions scored lower than the failure (0.5 = chance): logit 0.889 [0.877, 0.899], hgb 0.905 [0.897, 0.915]; without the five phase features 0.856 and 0.877; frontcourt only 0.863 / 0.878 (0.810 / 0.844 without phase). Retrained on the matched rows only: ROC 0.877 / 0.891. So the ranking does not come from comparing different games, teams or periods. Caveat: the matched completions are very different passes (passer 44.5 ft from the basket vs 21.0 for the failures, 33% backcourt vs 4%), so this control is weak against the context explanation; T5 and T6b are the sharper tests.

## T6b - receiver-side geometry inside identical passer contexts
Strata = backcourt x quantiles of passer-to-basket distance, defender distance to the passer and shot clock (48 or 180 strata). A model that sees only receiver/lane-side features still ranks failures above completions within a stratum: concordance 0.800 [0.793, 0.807] (hgb) and 0.771 (logit) with 3 quantiles; 0.791 / 0.761 with 5. So there is information beyond where the passer is. This is also where the receiver-choice asymmetry (T1) can enter, so it is an upper bound on real pass-geometry signal.

## T1 - receiver rule (100-game dev sample, same games and folds)
Heading rule agrees with the true receiver on 75.9% of completions (2.3% have no heading target and are dropped in heading mode). On identical rows and folds:

| | logit ROC / PR | hgb ROC / PR |
|---|---|---|
| actual receiver | 0.840 / 0.0995 | 0.857 / 0.1208 |
| heading rule for both classes | 0.816 / 0.0692 | 0.829 / 0.0708 |
| paired difference (actual - heading) | +0.0235 [0.018, 0.029] / +0.0302 [0.023, 0.042] | +0.0275 [0.023, 0.033] / +0.0500 [0.038, 0.066] |
| passer-only, no receiver at all | 0.749 / 0.0457 | 0.746 / 0.0431 |

**T1b - live ball.** 17.6% of completions are dead-ball or inbound passes (clock stopped 12.4%, passer out of bounds 13.0%) against 2.4% of pbp failures. Scoring the same out-of-fold predictions on live rows only: ROC 0.815 (logit), 0.826 (hgb) (from 0.838 / 0.848), PR rises slightly because the failure rate is higher (2.0%) in that population. Frontcourt and live: 0.809 / 0.818.

## T2 - tracking-only failure labels (100-game sample)
Per game: 472.8 completions, 8.6 pbp failures, 35.5 tracking-only candidate failures (an opponent hold, out-of-bounds or fumble ends the flight, passing the same `_flight_ok` filter, no pbp bad pass matched), and 956 completions (2.0%) that the tracking relabels as fumbled after receipt. Auto-tags of the 3,546 tracking-only rows:

| tag | share |
|---|---|
| shot or rebound nearby in pbp | 33.6% |
| dead ball or inbound | 22.5% |
| releaser's own shot | 18.8% |
| live, no pbp event | 8.7% |
| the releaser's own lost ball / other turnover (not bad pass) | 7.3% |
| stoppage, other player's turnover, ball kept out of bounds by offense, bad pass credited to another release | 9.2% |

An informal look at the 28-example sheet (`reports/ws3-out/t2_manual_sheet.txt`; my reading, not a labelling study) agrees: most are missed-shot rebounds, free-throw rebounds, blocked shots, own-team dead balls and out-of-bounds lost balls, with a few real bad passes credited to a different player. Excluding the dead-ball, inbound and shot tags leaves 892 rows (25%), 590 with a heading target. The model (trained on pbp labels from the other 401 dev games) scores these 590 at AUC 0.828 (logit) / 0.813 (hgb) against live completions, close to its 0.825 / 0.848 on pbp failures. So the signal does **not** depend on the pbp matching step, but tracking-only labels are mostly not passes, so they cannot replace it.

## T5 - non-pass turnovers as a negative control
Releases of pbp lost balls, travels, offensive fouls and other turnovers, scored by the model trained on bad-pass labels (401 other dev games), AUC against 38,013 live completions:

| group | n | full model logit / hgb | passer-only logit / hgb |
|---|---|---|---|
| pbp bad-pass failures (the training label) | 807-855 | 0.825 / 0.848 | 0.717 / 0.729 |
| lost ball, no flight | 196-354 | 0.882 / 0.892 | 0.869 / 0.859 |
| travel, no flight | 46-176 | 0.855 / 0.873 | 0.847 / 0.851 |
| offensive foul, no flight | 84-205 | 0.868 / 0.885 | 0.861 / 0.854 |
| other turnover, no flight | 55-179 | 0.849 / 0.863 | 0.822 / 0.805 |

Of 1,633 pbp non-pass turnovers, 933 produced a new release row (no flight to an opponent) and 371 could not be matched to a hold. Events that are not passes look more "failed" than failed passes do. **This is the strongest evidence against the pass-difficulty reading.** (Counts differ between the full and passer-only columns because the full model needs a heading target.)

## What was not done
- No manual labelling study; the manual sheet is a spot check by me.
- A full 631-game symmetric-target dataset does not exist yet; T1 uses 100 games. WS4 regenerates once and re-evaluates (see reports/ws4-detection.md for the outcome).
- T3 used 3 seeds; the bar was marginally missed once (above).
- The causal question (does a defender in the lane make a pass fail, or is he the interceptor by construction?) is not answerable from these labels.
