# RL Stage 4c: ball fix, play-following executors, EPV, and a play-calling policy (500 iterations)

Code: `Code/rl/sim.py` (lead pass), `play_cond.py`, `build_play_protos.py`, `train_executor3.py`, `rollout_follow.py`, `sim_validate3.py`, `train_caller3.py`, `caller.py` (play head), `eval_run3.py`, `epv.py`, tests `tests/test_play.py`. Results and all 50 GIFs: `reports/rl-stage4c/` (`run/gifs/iter_010.gif` ... `iter_500.gif`, `run/README.md` index with per-checkpoint stats, `follow_goal/follow_test.json`, `sim_val_grid*/`, `epv/`, `run/eval_val.json`, `run/eval_holdout.json`). EPV details: `reports/rl-stage4c-epv.md`.

## 1. Why the ball looked magnetic (and the fix)
**Cause (measured).** A pass was aimed at where the receiver stood at release; during the 0.7 s flight he moved 3.8 ft on average (90th percentile 8.9 ft; 46% of caught passes more than 3 ft), and on arrival the ball jumped to him. The ball itself moved in a straight line at constant speed (it is a model of one, not a physics engine): the snap at the end made it look pulled.
**Fix.** The ball now flies straight to a lead point (receiver position plus his velocity times the flight time, receiver speed capped at 18 ft/s), and the receiver's movement is steered, within the same cap, to meet it. Measured after: ball-to-receiver gap at arrival 0.00 ft (was 3.8), receiver speed in flight unchanged (mean 4.5 vs 4.9 ft/s, 95th percentile 12.6 in both). The Stage-4a realism check with the new ball: 5 of 5 statistics inside the real interval (shot share 0.721 vs 0.699 [0.678, 0.723], time to shot 6.48 vs 6.61 s, passes before the shot 1.30 vs 1.36, shot value 0.98 vs 1.01, passes per possession 1.29 vs 1.36).
Caveat: it is still a kinematic ball: no bounce, no deflection, no in-flight interception animation (a failed pass just ends where it was aimed).

## 2. Executors that follow a chosen play
**What I tried first (trained as you chose): prototype-conditioned executor.** The chosen play's prototype formation (its five most occupied court cells at 1 s and 3 s ahead, matched to the five attackers by distance) was an input. Test that can fail (`rollout_follow.py`: 1,200 real validation possessions, executors alone, 6 s, does the resulting formation fall in the target play?): told a different play, 3.3% landed in it (3.0% for unconditioned executors): they ignored it. Told their own play 43.5% (39.8% free). Reason: in real data the prototype always agrees with what the players already do, so the network learned nothing it did not know from the history.
**What worked: goal-conditioned executors with hindsight goals (v3g).** Trained on the same real possessions, but the goal given is the court cell where each attacker actually is 1 s and 3 s later (the same coarse cells the prototypes use), with 30% of window samples given no goal. At run time the goals come from the chosen play's prototype. Following test:
| executors, told | formation lands in the target play | clean instance | mean offence speed (ft/s) | off court |
|---|---|---|---|---|
| nothing (v2) | 39.8% (their own play) / 2.9% (random other play) | 18% / 1.5% | 4.6 | 0% |
| **v3g, a random other play** | **37.2%** | 24% | 4.8 | 0% |
| **v3g, their own play** | **88.5%** | 63% | 4.5 | 0% |
Conditioning on a chosen different play raises the chance of producing it from 2.9% to 37%, and speeds and spacing stay realistic. Limits: rollouts start 1.2 s after midcourt from a real state (the first part of the formation is already real); 37% is far from always; the one-step prediction error of v3g in the play window is better than v2 only because the training goals are the real future (an oracle in evaluation), so those numbers are not comparable and are not claimed. The v3g decision head shifted: possessions got longer and passes more frequent, so the simulator was recalibrated on development games (shoot scale 1.05, pass scale 0.9, loss 0.03/s; five statistics: shot share 0.720 vs 0.699 [0.678, 0.723], time to shot 6.68 vs 6.61, passes before the shot 1.448 vs 1.363 [1.284, 1.437] (just above), shot value 0.97 vs 1.01, passes per possession 1.41 vs 1.36).

## 3. Pass difficulty and EPV
Pass difficulty has been used throughout: it decides every simulated catch and is an input to the policy. EPV: a new state-value network explains no more variance than shot clock plus ball position (R^2 0.0015 vs 0.0028; `rl-stage4c-epv.md`), so by the rule agreed beforehand it is not used in the reward; the reward is shot xPts and a turnover cost of 0.10.

## 4. The run (500 iterations, play choice learned, 14 s of GIFs every 10 iterations)
Policy: a play is chosen once per possession (prior = real play shares, residual bounded by 1.5 logits), the executors follow it, and the shoot/pass policy of Stage 4b (initialised from its last weights) acts on top. Bonus +0.25 if, up to the end of the possession (at least 3 s after midcourt), the formation followed the chosen play (mean cell-count distance to the play's prototype at or below the median of that play's real members over the same slices) and the possession ended in a shot with xPts >= 1.0. About half of real members meet the formation test, and in real data about 5% of the other plays' possessions do.

Evaluation (2,000 fixed starts per split, same random draws for every row, paired game-clustered 95% intervals):
| | reward without bonus | with bonus | shot share | duration (s) | shot distance (ft) | followed chosen play | bonus rate |
|---|---|---|---|---|---|---|---|
| dev: v2 executors, own decisions | 0.648 | 0.648 | 0.69 | 6.4 | 17.6 | - | - |
| dev: v3g, plays drawn from real shares, own decisions | 0.675 | 0.704 | 0.72 | 6.3 | 19.0 | 44% | 11.6% |
| dev: v3g, masked zero-residual policy | 0.645 | 0.669 | 0.71 | 6.3 | 18.8 | 45% | 9.5% |
| **dev: trained policy** | **0.970** | 1.000 | 0.90 | 2.5 | 22.9 | 41% | 12.0% |
| holdout: v2 / v3g own / trained | 0.666 / 0.647 / 0.988 | - / 0.671 / 1.014 | 0.71 / 0.71 / 0.91 | 6.4 / 6.0 / 2.4 | 17.9 / 18.8 / 22.8 | - / 44% / 37% | - / 9.8% / 10.2% |
Trained minus v2 baseline: +0.322 [0.288, 0.353] (dev), +0.323 [0.292, 0.354] (holdout). Trained minus v3g own decisions: +0.295 [0.266, 0.324] (dev), +0.342 [0.314, 0.370] (holdout). (The holdout was also used to score the earlier policies, so it is a third look, not a clean test.)

## What the policy actually learned
- **The reward gain is the shot policy, not the plays.** The xPts part is 0.97 against 0.65-0.68; the play bonus rate is the same as without training (12.0% vs 11.6% on dev). It shoots about 2.5 s into the possession, from 22.9 ft (many threes), at a shot value of 1.09 against 0.97-0.99, with turnovers at 9.5% against 25-27%. Same story as Stage 4b: shoot early and long, the part least supported by real data (the shot model was trained on shots real players chose).
- **The play head collapsed onto two plays.** Plays 11 and 19 are chosen 25.5% and 20.9% of the time (prior 5.1% and 4.1%, the maximum allowed by the bound is 4.5x; the observed ratios are 5.0 and 5.1 including the renormalisation), and the choice entropy fell from 2.98 to 2.43 nats. Because the policy shoots at 2.5 s, plays that are easy to "follow" for 3 s and end in a good shot are the cheapest way to collect the bonus. It is a reward-hacking pattern, not evidence that plays 11 and 19 are good plays.
- **It follows its chosen play less often than the baseline** (41% vs 44-45%) because it stops the possession sooner (56% of possessions last 3 s, against 88%).
- **Time cost, shot value and the shot model** carry the same caveats as Stage 4b (calibrated time cost; optimistic shot value for shots real players rarely take); the sensitivity runs of that stage were not repeated here.

## Verdict on your three requests
1. **Ball**: fixed and measured (no snap; gap 0.0 ft), with the limits above.
2. **Executors follow a chosen play**: yes after a redesign (hindsight goals); the test shows 37% vs 3% when told a different play, and the training shortcut I started with (prototype input) did not work.
3. **Pass difficulty and EPV**: difficulty is used; a new EPV does not beat simple features, so it is not in the reward.
The 500-iteration run shows the play-calling machinery works end to end (choose a play, executors follow, bonus scored), but with this reward the optimiser prefers shooting early to running a play: the bonus is worth at most 0.25 x 12% = 0.03 points against 0.3 for the early shot. Making plays matter needs a reward in which waiting for a better shot is worth something (a validated shot-quality or EPV signal for non-shot states, or longer possessions with rebounds), which this project does not have yet.
