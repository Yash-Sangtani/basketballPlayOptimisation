# Pass difficulty model (stage 1) and value layer (stage 2)

Built 2026-10-27 for the RL phase. Code: `Code/difficulty/`. Models: `Code/difficulty/model/difficulty_v1.joblib`, `shot_v1.joblib`.
Dev tuning used 401 dev games; 100 other dev games (`Code/splits/difficulty_final_games.txt`, drawn once with a fixed seed) were scored once at the end; **the frozen test split was not read** (it is still unused for this model).

## What "difficulty" is here
**difficulty = P(a pass thrown now is not caught by the intended teammate)** (intercepted, out of bounds, or fumbled), `p_complete = 1 - difficulty`. It is a calibrated probability for any (passer, teammate) pair in any court state, so it can be used directly as a transition probability or a penalty in RL, including for passes that were never thrown.
The label comes from **tracking alone**, not from the play-by-play turnover log that WS3 showed was selecting turnover-prone moments: every live hold-end is a release; the next thing that happens to the ball decides the label (a teammate gets it = 0, an opponent gets it / it goes out / a teammate touches and loses it = 1). Both outcomes pass the same flight filter. Play-by-play is used only to *exclude* non-pass releases (the releaser's own shot, a rebound/block just around the release, dead-ball contexts, the releaser's own lost ball/travel/offensive foul) and to cross-check.
Features are computed at the release instant for the intended receiver (heading rule, applied to both classes): the original 15 pass-geometry features plus interception geometry (`intercept_margin`: how much earlier the ball reaches points along the line than the best-placed defender could, `n_def_can_intercept`), second-nearest defenders and receiver/passer velocity along the pass. All from one court state (+ velocities), so a simulator can supply them.

## Results (positive class = failed pass; HistGradientBoosting + isotonic calibration; grouped by game)
Tuning games (401 games, 130,148 clean live passes, 8,096 failures = 6.2%), 5-fold GroupKFold:

| features | model | ROC [95% CI] | PR-AUC [95% CI] (base rate 0.062) | Brier skill (calibrated) |
|---|---|---|---|---|
| passer's situation only (6) | hgb | 0.744 [0.738, 0.750] | 0.259 [0.248, 0.271] | 0.106 |
| original 15 | hgb | 0.833 [0.827, 0.838] | 0.353 [0.342, 0.369] | 0.171 |
| **all 21 (the model)** | **hgb** | **0.840 [0.834, 0.845]** | **0.359 [0.347, 0.375]** | **0.176** |
| all 21 | logistic | 0.807 [0.801, 0.812] | 0.301 [0.290, 0.314] | 0.133 |

**Held-out 100 dev games, scored once** (model fit on the 401 tuning games): ROC 0.838 [0.828, 0.848], PR-AUC 0.365 [0.337, 0.391] against a 0.064 base rate, Brier skill 0.177; situation-only 0.744 / 0.262 / 0.100. This matches the tuning estimate. Calibration holds across deciles (predicted vs observed failure rate, top decile 30.6% vs 31.0%), with a small under-prediction at the low end (mean prediction 0.0604 vs 0.0637 observed).

**Does the pass itself matter, not just the situation?** Yes. Inside strata of identical passer situation (back/front court, quantiles of distance to basket, nearest-defender distance, shot clock), a model that sees only receiver/lane-side features ranks failures above completions 77% of the time (0.5 = no information); the full model on the held-out games 79%. The model also exposes `relative_difficulty` = log-odds of failure minus log-odds of what the situation alone predicts, which isolates the pass-specific part.

**Face validity (tuning games, observed rate = predicted rate in every bin):** failure rate falls monotonically with `intercept_margin` (8.1% when a defender can beat the ball to the line, 0.8% when none can), with the distance of the nearest defender to the pass line (12.8% within 2 ft, 0.7% beyond 9 ft), and with the nearest defender's distance to the receiver (11.8% -> 4.1%). It is U-shaped in pass length (23% for hand-off-length passes under 8 ft, 5% at 14-26 ft, 7.5% beyond 34 ft).

**Cross-label check.** A model that never saw a pbp-confirmed bad pass as a failure ranks those 2,620 passes above completions with AUC 0.817, so the tracking label agrees with the official turnover log where they overlap.

## Sensitivity (tuning games; reported, not hidden)
| setting | clean passes | failure rate | ROC / PR (hgb, all features) |
|---|---|---|---|
| primary: narrow exclusion windows, fumbles kept | 130,148 | 6.2% | 0.840 / 0.359 |
| wide exclusion windows (30% of live flights excluded) | 115,048 | 6.3% | 0.836 / 0.362 |
| wide windows and fumbled flights dropped | 112,864 | 4.5% | 0.864 / 0.359 |

The conclusions do not depend on these choices. The primary setting was chosen on the tuning games for excluding as little as possible with the same skill.

## What the label is and is not (read before using it in RL)
- 6.2% of live non-shot passes are labelled failures, not the 1.6% the turnover log gives: only about one third (2,620 of 8,096) coincide with a pbp bad-pass turnover. The rest are deflections, loose balls, fumbled catches (30% of failures end as a short teammate touch then a loss; only 15% of those are pbp bad passes) and hand-off strips. "Difficulty" therefore means "likely to not reach the intended teammate cleanly", not "likely to produce a turnover".
- The intended receiver is inferred from the ball's first 0.32 s (right for 76% of completions), so for about a quarter of completions the features describe a teammate other than the one who caught it. This adds label noise and is the price of treating both classes the same way.
- The model still gives non-pass turnovers (lost ball, travel, offensive foul, excluded from training) 2.3x the completion risk (mean 0.120 vs 0.051; failures 0.247): part of the signal is still "ball handler under pressure". `relative_difficulty` is the part of the score that does not come from the passer's situation.
- It describes observed NBA defenders in 2015-16 tracking (first half of the season). A policy trained against it can find states outside that distribution (for example passes through three defenders at 40 ft/s) where the model is extrapolating; passes shorter than 1 ft are returned as `p_complete = 1`, `extrapolated = True`.
- Association, not causation: a defender on the line is also how interceptions are defined.

## Stage 2: value layer (weaker; use with care)
- **Shot model** (`shot_v1`): P(make) from distance, angle, two nearest defenders, nearest-defender-between-shooter-and-basket, speed, shot clock, on 62,569 matched field-goal attempts. Grouped CV: ROC 0.633, Brier skill 0.078 over the league average (distance alone: 0.033), calibrated by decile, predicted 0.998 vs actual 0.994 points per shot. xPts = P(make) x (2 + P(three | distance, angle)).
- **Turnover cost** `c_fail = 1.16` points: possession mean 1.050 points plus the transition bonus (the next possession scores 1.137 after a turnover vs 1.01-1.04 otherwise), measured on the 501 dev games' possession table.
- `PassValueModel.evaluate` returns `ev_pass = p_complete * threat_receiver - difficulty * c_fail` and `pass_minus_shoot`, where `threat` is the xPts of a shot taken right now.
- **Not validated as a pass-value signal.** Among 122,052 clean completed passes, the receiver's shot threat identifies assisted passes only weakly (AUC 0.537; the pass's threat gain 0.602; plain distance of the receiver to the basket does better, 0.630 at release and 0.726 at reception on a 30-game check). Measuring threat at reception rather than release did not help the xPts score (0.538). The immediate-shot threat ignores drives and further passes, so `ev_pass` is a lower bound that is only meaningful near the basket. Recommendation: use `p_complete` / `difficulty` and `c_fail` from this work, and let the RL value function learn the continuation value.

## Use
```python
from difficulty.api import PassDifficultyModel, PassValueModel   # Code/ on the path
m = PassValueModel()
m.evaluate(P, T, defenders_xy, basket_xy, shot_clock, p_vel, t_vel)          # one candidate pass
m.score_teammates(holder_xy, mates_xy, defenders_xy, basket_xy, shot_clock)  # all teammates in one model pass (about 18 ms for 4)
m.predict_many([...states...])                                               # vectorised
```
Feet and ft/s, court 94 x 50, mirror-symmetric (pass the basket the offense attacks). Features are one pure function (`pass_features.py`, shared by training and the API); on one game it reproduces the existing pipeline's 14 shared features exactly (max absolute difference 0.0). 10 tests in `Code/difficulty/tests`.

## Reproduce
```
cd Code
# per-game dataset for dev games (about 35 min with 3 processes, one job): difficulty/extract.py <list> --out DIR
python difficulty/train_eval.py --data DIR --regime narrow --out ../reports/difficulty-dev-narrow    # tuning CV
python difficulty/train_eval.py --data DIR --regime narrow --fit-final                                # fit on 401 games
python difficulty/train_eval.py --data DIR --regime narrow --score-holdout --out ../reports/difficulty-holdout   # one look at 100 games
python difficulty/shot_value.py --data DIR --out ../reports/difficulty-dev-shots                      # stage 2 dev report
```
Not done: test-split scoring (kept unused), multiple seeds, hyper-parameter search (fixed settings), a model of what the receiver does after the catch, a trained policy-side check that the model is not exploited.
