# RL Stage 2b: executor v2 (mixture outputs, pass-target head, seeds, holdout)

Code: `Code/rl/executor_v2.py`, `train_executor2.py`, `ball.py`, `compare_runs.py`, `rollout_check2.py`, `score_holdout2.py`; model `Code/rl/model/executor_v2.keras` (real profiles, seed 0). Raw numbers: `reports/rl-stage2b/` (per-run json/npz, `compare.txt`, `rollout/`, `holdout.json`).
Setup: 320 fit / 81 validation games from the 401 RL train games, 300k fit samples, 5 epochs, 3 mixture modes, 278k parameters, CPU. Profiles from fit games only.
Runs: real profiles x3 seeds, no profiles x3 seeds, shuffled profiles x1. (A worker restart killed the first job after five runs; the last two, `none:2` and `shuffled:0`, were rerun with the same settings.)

## Results (validation, 1-s step; negative differences favour real profiles; intervals are paired game-clustered bootstrap)
| | real | none | real - none |
|---|---|---|---|
| offence 1-s error (ft) | 2.007 | 2.004 | +0.003 [+0.001, +0.004] |
| defence 1-s error (ft) | 1.771 | 1.766 | +0.006 [+0.005, +0.007] |
| decision loss | 0.771 | 0.779 | -0.009 [-0.010, -0.007] |
| pass-target loss | 0.870 | 0.876 | -0.007 [-0.011, -0.004] |

Shuffled profiles are worse than real on every metric (e.g. offence error +0.020 ft, decision loss +0.026), so the small decision gain is a property of the right player's profile, not extra inputs. Profiles do not help movement (differences of 0.003-0.006 ft, in the wrong direction). Pass-target head: 65% top-1 accuracy of who the holder passes to. Seed-to-seed spread is 0.001-0.009 ft, smaller than any real-vs-none gap that matters.

Holdout (100 games, scored once with the saved real-profile model): offence 2.005 ft, defence 1.771 ft, decision loss 0.774, target loss 0.870, accuracy 0.653: the same as validation, so nothing was overfit to the validation games.

## Roll-outs (3 s, replanning every 0.4 s, 7,513 roll-outs, 40 validation games; real speed 4.76 offence / 4.09 defence ft/s)
| policy | speed off / def (ft/s) | 3-s ADE off / def (ft) | spacing (ft) |
|---|---|---|---|
| constant velocity | 5.31 / 4.18 | 15.3 / 13.0 | 34.7 (leaves court 25%) |
| mean of mixture | 1.62 / 1.37 | 8.1 / 5.9 | 24.8 |
| most likely mode | 1.60 / 1.35 | 8.7 / 6.3 | 25.9 |
| sample | 4.05 / 3.33 | 10.3 / 7.2 | 24.6 |
| sample, noise x0.7 | 3.48 / 2.83 | 9.7 / 6.8 | 24.7 |
| real | 4.76 / 4.09 | | 24.9 |

The mixture with sampling raises speed from 3.6 (v1) to 4.05 ft/s offence, still 15% under real; deterministic policies (mean, top mode) stand nearly still, so use sampled roll-outs for realism and the mean only for point forecasts. Spacing matches real and nobody leaves the court. Sampling costs ADE versus the mean (that is expected: it commits to one future).

## Verdict
Done: mixture and sampling fix most of the under-speed, the pass-target head works, intervals and holdout are in. Not fixed: sampled players still move about 15% too slowly, and profiles do not improve movement. Treat the executor as adequate for Stage 4 prototyping, not as a faithful simulator of individual players' movement.
