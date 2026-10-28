# WS5b: review of the value-model MVP (possession table, V(s), pass delta-V)

Reviewed and run 2026-10-27 on the 120 dev games in `Code/value/value_games.txt` (none of them in `test_games.txt`; all in `dev_games.txt`).
Commands: `TRACKING_DATA_DIR=<repo>/Tracking_Data python value/extract_games.py value/value_games.txt <states dir>` (10.5 s/game, 0 failures) then `python value/fit_value.py <states dir> reports/ws5b-out` (39 s). Full output: `reports/ws5b-out/fit_value_output.txt`. All value tests pass (20 in `Code/value/tests`; the previously failing `test_feature_values` had a wrong expected value, see below).

## What was fixed
- `test_states.py::test_feature_values` expected `n_def_10 == 2`; the defender at (10, 26) is 10.05 ft from the holder at (20, 25), just outside the 10 ft radius, so the code's answer (1) is right. The test now states the distances. No change to `states.py`.
- `fit_value.py` wrote `pass_dv.csv.gz` to a hard-coded Mac path (`<scratch>`); it now goes to the reports folder. `check_possessions.py` (`--map`) and `run_extract.sh` still carry Mac paths (`<scratch>`, the Mac venv, a `gh` download of archives); they are not needed here because the archives are in the checkout.

## Measured results (120 dev games, 229,702 sampled 1 Hz states, 47,836 completed passes)
**V(s) predicts almost nothing.** Target = points scored on the possession the state belongs to (mean 1.03, sd 1.135). Out-of-fold, GroupKFold(5) by game:

| model | RMSE | R^2 |
|---|---|---|
| constant (training mean) | 1.1347 | -0.0001 |
| baseline: shot clock + ball position (5 features) | 1.1329 | 0.0030 |
| rich: 21 features | 1.1343 | 0.0006 |

Rich minus baseline: RMSE gain -0.0014 [-0.0030, +0.0003]; the 16 extra features do not help (better than baseline in 5% of game resamples). The only visible structure is distance to the basket (ball within 5 ft of the rim: observed 1.46 points vs 1.02 elsewhere) and a little shot-clock/period-time dependence. Calibration slope of outcome on prediction: 0.52 [0.43, 0.63] for the rich model (overconfident), 0.74 [0.66, 0.82] baseline.

**Delta-V on completed passes (V at reception minus V at release, out of fold).** Mean +0.0265 per pass (rich), +0.0034 (baseline). By group, rich model: assisted passes +0.088 [0.083, 0.093]; receiver took the next shot and missed +0.048; unassisted make +0.029; receiver did not shoot next +0.016. Assisted minus all others: +0.067 [0.062, 0.071], Cohen d 0.78. The same contrast with the baseline model is +0.020 [0.016, 0.023], d 0.23.

## Reading (measured vs inferred)
- Measured: the pipeline works end to end (pbp possession table with points matching final scores in all 501 dev games after counting technical/other points separately; frames matched to possessions; states, CV, bootstrap), and delta-V is higher on assisted passes than on other passes.
- Measured: V(s) as built has essentially no predictive power beyond distance to the rim (R^2 0.0006-0.003), and the rich model is worse calibrated than the baseline. The large rich-model delta-V difference for assists (d 0.78 vs 0.23) therefore cannot be read as the extra features valuing the pass better: those features add no out-of-fold skill. (Inferred, not tested: it comes from release-to-reception changes in features such as `closest_def` and `holder_speed` that are themselves correlated with the pass being followed by a shot. Assists are defined by the receiver then scoring, so an assist contrast is only a sanity check, not validation.)
- Not done: no completion-probability term (the failed-pass model), so there is no pass-versus-shoot value, no turnover value, and no comparison with Dhruv's shot value (the shot models do not exist in the repo, see `reports/ws5a-shots.md`).
- Design weakness (inferred from the numbers): training V on whole-possession points from any frame in the possession, with only one state per second and no information about what happens next, is a very noisy target. A usable V would need either a much larger set of games, a target restricted to "ball is in the frontcourt and the possession's remaining events", or a structured model (shot probability x shot value plus turnover rate); this MVP does not show any of those work.
- Not independently re-run this session: `check_possessions.py` (the earlier handoff numbers for the possession table: 501 games ok, count ratio median 1.013, points equal to the final score in 501/501 with technical/other points separated; only 216/501 strictly). Its 20+ unit tests pass.

## Recommendation
Keep it as a working MVP of the data plumbing, not as a result. Do not use delta-V for play recommendations. The next step that would change this is a better target and a shot-value model, not more features.
