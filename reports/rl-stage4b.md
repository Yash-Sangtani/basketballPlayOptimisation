# RL Stage 4b: decision-level play-caller trained in the simulator

Code: `Code/rl/caller.py` (policy, features, REINFORCE), `train_caller.py` (training and evaluation), `caller_sens.py` (robustness); results in `reports/rl-stage4b/` (`train_history.json`, `eval_val.json`, `eval_holdout.json`, `sensitivity_val.json`, `policy.weights.h5`, and `exploit_run1/` for the failed first run).

## What this stage is, and is not
The policy chooses, at every 0.4 s block in which an offensive player holds the ball: keep, shoot, or pass to one of four teammates. The five players' movement still comes from the executors. This is a shoot/pass/target policy, **not** full play-calling: Stage 3 found that play ids give the executors no usable signal, so there is no lever through which a "play" could change the movement. Full play-calling needs executors that respond to an intent; that is open work.
- **Policy:** logits = log(executor's own decision probabilities) + a bounded learned residual (|residual| <= 1.5, i.e. action probabilities can change by at most 4.5x versus the imitation model) from a small MLP over per-action features (shot xPts of the shooter, p_complete of the pass, expected value of the pass, distance and nearest defender of the receiver, clocks). Zero residual reproduces the simulator's own behaviour.
- **Guard rails:** passes the difficulty model flags out of distribution are not available actions; shooting beyond the shot model's training range (a "heave") is only available in the last 3 s of the shot clock.
- **Reward:** shot xPts; turnover or shot-clock violation -0.10 (the measured extra points the opponent scores after a live-ball turnover, 1.137 - ~1.03).
- **Learning:** REINFORCE with a learned value baseline and a KL penalty to the executor's behaviour, 45 iterations x 800 possessions from the 320 fitting games' start states (about 15 minutes on CPU).

## The first run was exploited (kept as evidence)
The first version used the Stage-0 spec reward (turnover -1.16) and an unbounded residual. Within six iterations the policy shot immediately in every possession (shot share 1.00, 0 passes, duration 0 s, reward 0.98 against 0.36 for the simulator's own play; log in `exploit_run1/`). Two causes: (1) -1.16 on a turnover on top of raw points double counts the loss (C_FAIL was defined as the swing relative to an average possession), which makes any quick shot look excellent; (2) the shot model values hypothetical long threes at about 1 point (a 30-33 ft shot is 33% x 3), but it was trained on shots real players chose to take, so it is optimistic for shots nobody would take at that moment. Fixes made afterwards (listed so the change is visible): turnover cost 0.10, heave masking, and the bounded residual. Fix (2) only limits the exploit, it does not remove the selection bias.

## Results (second run, fixed settings)
Development (81 validation games, 1,500 starts) and holdout (100 games, 3,000 starts, policy frozen, scored once). Same start states and random seed for each policy; difference is paired with a game-clustered 95% interval.

| holdout, 3,000 starts | reward (points) | shot share | turnover share | passes | duration (s) | shot distance (ft) | xPts per shot |
|---|---|---|---|---|---|---|---|
| executor's own decisions (baseline) | 0.657 | 0.70 | 0.27 | 1.34 | 6.4 | 17.6 | 0.99 |
| same with OOD passes masked (zero residual) | 0.691 | 0.74 | 0.23 | 1.21 | 6.7 | 17.7 | 0.97 |
| **trained policy** | **0.956** | 0.90 | 0.10 | 1.51 | 2.7 | 21.7 | 1.07 |

Trained minus baseline: +0.299 points per possession [0.279, 0.321] on holdout (+0.289 [0.257, 0.321] on development). Trained minus masked-prior: +0.265 [0.243, 0.288].

What the policy does differently (measured): shoots about 4 s sooner (mean 2.7 s after the start state), from farther out (21.7 vs 17.6 ft, more threes), takes shots the shot model values higher (1.07 vs 0.99), passes slightly more but longer (23.8 vs 20.4 ft), and loses the ball far less often (10% vs 27%).

## Robustness of the gain (development games, frozen policy)
| simulator assumption | baseline | trained | trained - baseline |
|---|---|---|---|
| default | 0.663 | 0.945 | +0.282 [0.247, 0.312] |
| no time cost of holding the ball (loss hazard 0) | 0.812 | 1.002 | +0.189 [0.163, 0.214] |
| double time cost (0.06/s) | 0.561 | 0.914 | +0.353 [0.319, 0.386] |
| every shot worth 10% less | 0.594 | 0.855 | +0.261 [0.237, 0.287] |

The gain stays well above zero in each scenario but changes by a factor of two: about a third of it comes from the time cost (the calibrated ball-loss hazard punishes slow possessions), the rest from selecting higher-value shots and safer passes.

## What this does and does not show
- It shows a learned shoot/pass policy beats imitation **inside this simulator**, with a margin that survives the three assumption changes tried, and that the learner finds exploits quickly if the reward or the guard rails are loose (the first run).
- It does **not** show these decisions would score more against real defences. The reward is an expected-points proxy; the shot model is optimistic for shots real players rarely take (selection), which is exactly what the policy moves toward (longer, earlier shots); the simulator has no rebounds, free throws or transition, and the time cost is a calibration constant rather than a measured effect. Shooting earlier is the policy's main lever and also the claim least supported by real data.
- Not done: seeds (one training run), policy classes more expressive than a 32-unit MLP, a check on real possessions that the policy's preferred early shots are ever taken and succeed, and tests for `caller.py` (the simulator has unit tests; the policy was checked through the training runs and the first-run exploit only).

## Verdict
Stage 4b is done as a decision-level policy with an honest caveat: the +0.30 is a simulator result, partly driven by the shot model's value of shots that real data cannot validate. Stage 5 (audit) should test exactly that: compare the policy's preferred shots against real outcomes of similar real shots, rerun with several training seeds and a stricter shot-value guard, and decide whether real-world claims are warranted.
