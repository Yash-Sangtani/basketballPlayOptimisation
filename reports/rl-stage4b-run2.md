# RL Stage 4b, run 2: 500 more iterations with a play-run bonus and a GIF every 10 iterations

Code: `Code/rl/play_reward.py`, `build_play_thresholds.py`, `train_caller2.py`, `gif_episode.py`, `eval_run2.py`. Results and all 50 GIFs: `reports/rl-stage4b/run2/` (`gifs/iter_010.gif` ... `iter_500.gif`, `README.md` index with per-checkpoint statistics, `train_history.json`, `eval_run2_val.json`, `eval_run2_holdout.json`).

## What changed from run 1
- **Continued training** from the run-1 policy for 500 iterations of 800 possessions (about 2.5 hours), checkpoint and GIF every 10 iterations. The policy gets one new input feature (time left until the 6-s play window closes).
- **Play-run bonus (+0.25 points)** when the simulated possession (a) lasts at least 6 s after midcourt, (b) its first-6-s formation descriptor is assigned to one of the 24 Stage-3 plays and lies within that play's median in-cluster distance ("run cleanly"), and (c) it ends in a shot with xPts >= 1.0 ("successfully"). The thresholds were fitted on the 320 fitting games (by construction half of real possessions are "clean"); the descriptor built from simulated trajectories uses finite-difference speeds and assigns the same cluster as the library's own descriptor for 90.6% of real possessions.
- Everything else (turnover cost 0.10, heave mask, bounded residual, KL penalty) as in run 1.

## Results (same starts and seeds for every policy; paired game-clustered 95% intervals)
| | reward without bonus | play-bonus rate | clean-play rate | reached 6 s | duration (s) | passes |
|---|---|---|---|---|---|---|
| executor's own decisions (dev, 2,000 starts) | 0.646 | 0.072 | 0.252 | 0.60 | 6.3 | 1.29 |
| run 1 policy | 0.956 | 0.038 | 0.076 | 0.20 | 2.6 | 1.44 |
| **run 2 policy** | 0.948 | 0.056 | 0.108 | 0.29 | 3.3 | 0.97 |
| holdout: baseline / run 1 / run 2 | 0.657 / 0.948 / 0.961 | 0.080 / 0.041 / 0.058 | 0.257 / 0.082 / 0.101 | 0.60 / 0.21 / 0.26 | 6.4 / 2.7 / 3.0 | 1.36 / 1.55 / 0.91 |

Run 2 minus run 1: -0.008 [-0.027, +0.012] on development, +0.013 [-0.011, +0.035] on holdout (a second look at the holdout, which scored run 1 earlier; not a clean test), so no difference. Run 2 minus the executor's own decisions: +0.30 [0.27, 0.33] (dev), +0.30 [0.27, 0.34] (holdout).

## What the 500 iterations did and did not do
- **Performance plateaued by about iteration 100.** The first 50 iterations averaged 0.960 reward (play bonus rate 4.8%), the last 50 averaged 0.969 (4.3%). The extra 450 iterations did not improve the policy.
- **The play bonus did not make the agents run plays.** The policy still shoots about 3 s into the possession and only 22-29% of its possessions last 6 s, against 60% for the executors' own decisions. Relative to the executors' own behaviour, the policy now runs a clean play and scores in fewer possessions (4-6% vs 7-8%). The bonus was worth at most 0.25 x 0.06 = 0.015 points per possession, against about 0.3 for shooting early, so the optimiser had no reason to wait.
- **Run 2 differs from run 1 mostly in the passing.** It passes less (0.9-1.0 vs 1.4-1.6) and lasts about 0.4 s longer; reward and shot value are the same within noise.
- **Why the play reward is weak even if the weight were raised.** (1) The policy does not control formations: movement comes from the executors, which are not conditioned on plays, so the policy can only affect which formation arises by holding the ball longer or choosing passes. (2) The 24 clusters are a continuum (silhouette 0.06) and predict outcomes at the level of noise (Stage 3), so "ran a play" is mostly "was in a common formation for 6 s". (3) Every possession of 6 s or more lands in some cluster, and the "clean" test passes for half of them by construction. Raising the bonus would therefore mostly reward holding the ball for 6 s and then taking a good shot, not running a play. I did not raise it: that would be tuning the reward to produce a visible behaviour.

## The GIFs
`reports/rl-stage4b/run2/gifs/iter_NNN.gif`: red = offence, blue = defence, orange = ball, ring = ball handler, title shows the iteration and whether the possession earned the bonus and for which play id, final caption shows the outcome. Because the policy shoots early, most GIFs are short (2-3 s). A GIF that earned a bonus is chosen when one exists in the batch.

## Verdict
The requested training was run to completion and the play-run reward is in place and logged, but within this action space it does not change behaviour. Getting agents to run plays would need intent-conditioned executors (the policy picks a play and the executors move to run it) and a sharper play definition; Stage 3's current library does not supply either.
