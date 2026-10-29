# RL Stage 2: behaviour models for all ten players (offence and defence), with player profiles

Run 2026-10-28 on CPU (4 cores, no GPU). Code: `Code/rl/{profiles,executor_data,executor_model,train_executor,rollout_check}.py`, tests in `Code/rl/tests/test_executor.py` (4 pass, including that the model is permutation-equivariant in the player slots). Trained model: `Code/rl/model/executor_v1.keras` (257,079 parameters). Raw outputs: `reports/rl-stage2/`.
Data discipline: fit on 320 of the 401 `train` games, validate on the other 81 (fixed split, `fit_val_games()`); player profiles are estimated from the 320 fitting games only; the holdout and test games were not read.

## What was built
- **Profiles** (`profiles.py`): 12 style numbers per player, shrunk toward his position group: off-ball speed, distance from the basket and from the court's long axis, usage, shoot rate, pass rate, the four shot-location shares (rim / short / mid / three), and defence: speed and how tightly he guards (distance to the nearest offensive player). Players with little data fall back to their position group; unknown players get the average profile.
- **Executor** (`executor_model.py`): a 3-layer transformer over the ten players plus a context token. For every player (offence and defence, one shared network, a role flag in each token) it outputs where he will be at +0.2, +0.4, ... +1.0 s as a Gaussian with learned uncertainty; for the ball handler it outputs a decision over the next 2 s (shoot / pass / keep). Inputs per player: position, velocities over 0.2 s and 0.4 s (backward differences, so a rollout can recompute them), role, position group, holder flag, ball and basket geometry, nearest opponent, and the profile. Context: ball height, shot clock, time in possession, time since midcourt, period clock.
- **Training set:** half-court live states with an unbroken 1.4 s window; 400,000 fit samples, 150,000 validation samples; 6 epochs, about 135 s per epoch.

## Result 1: one-second prediction (validation games)
Mean position error in feet at each horizon (the model, then a constant-velocity extrapolation, then "stand still"):

| horizon | 0.2 s | 0.4 s | 0.6 s | 0.8 s | 1.0 s |
|---|---|---|---|---|---|
| offence, model | 0.19 | 0.50 | 0.93 | 1.41 | **1.96** |
| offence, constant velocity | 0.37 | 0.93 | 1.63 | 2.45 | 3.37 |
| offence, stand still | 1.04 | 2.05 | 3.01 | 3.93 | 4.80 |
| defence, model | 0.19 | 0.48 | 0.87 | 1.28 | **1.71** |
| defence, constant velocity | 0.37 | 0.92 | 1.60 | 2.37 | 3.22 |

The predicted uncertainty is well calibrated (mean squared standardised error at 1 s: 1.02, where 1.0 is perfect). Decision head (holder states, n = 148,483): log loss **0.760** against 1.012 for the class prior, accuracy 63.6% (shoot 15.7% / pass 37.0% / keep 47.4%).

## Result 2: do profiles help? (same model, same data, three versions)
| profile input | offence 1 s error | defence 1 s error | movement NLL (offence) | decision log loss |
|---|---|---|---|---|
| own profile | 1.956 ft | 1.712 ft | -1.1305 | **0.7602** |
| none (position group only) | 1.954 ft | 1.711 ft | -1.1355 | 0.7655 |
| another player's profile (control) | 1.976 ft | 1.737 ft | -1.1248 | 0.7776 |

- **Movement:** the profile does **not** help (own profile = none; the shuffled control is slightly worse). This matches Stage 1: footwork is explained by the situation and position.
- **Decisions:** the own profile beats position-only by 0.7% and beats a wrong profile by 2.2%, so the style signal is real but small. Single run each (one seed): differences of this size are suggestive, not established; I did not measure seed-to-seed variation.
- So "players move like the real ones" is, in this data, mostly a matter of the situation; "players choose like the real ones" is where individual style appears. The profile pathway is in place and will matter most for the decision and play-selection layers.

## Result 3: multi-second rollouts (the check the simulator depends on)
7,513 rollouts of 3 s from 40 validation games: the model predicts a second, keeps the first 0.4 s, recomputes velocities from its own positions and repeats. Ball, holder and clocks come from the real data, so this tests movement given the ball, not decisions.

| | ADE at 1 / 2 / 3 s, offence (ft) | at 3 s, defence | mean speed off / def (ft/s) | 99th pct speed (off) | mean pairwise spacing (ft) | outside court | pairs closer than 1.5 ft |
|---|---|---|---|---|---|---|---|
| real | | | 4.76 / 4.09 | 14.8 | 24.9 | 0.2% | 0.4% |
| constant velocity | 3.4 / 8.9 / 15.3 | 13.0 | 5.31 / 4.18 | 16.3 | 34.7 | 25.4% | 0.3% |
| model, mean of the Gaussian | 2.0 / 5.0 / **7.8** | 5.8 | 2.22 / 1.64 | 10.5 | 24.9 | 0.0% | 0.7% |
| model, **correlated samples** (one draw per player, applied along the predicted second) | 2.6 / 6.2 / 9.4 | 6.7 | 3.62 / 2.95 | 13.6 | 25.2 | 0.2% | 0.6% |
| model, independent noise at every step | 3.8 / 10.5 / 18.7 | 16.8 | 29.0 / 28.0 | 123 | 30.8 | 20.7% | 0.3% |

- The model stays on the court, keeps realistic spacing (24.9 ft against 24.9 ft real) and halves the 3-second error of constant velocity.
- **Naive sampling is unstable:** independent noise at every step feeds into the recomputed velocities and speeds explode (29 ft/s). One consistent draw per player along each predicted second fixes it (speeds 3.6 ft/s, 0.2% outside the court). Use correlated sampling in the simulator.
- **Players still move too slowly** (mean offensive speed 3.6 ft/s sampled, 2.2 ft/s for the mean, against 4.8 real): each step predicts an average over possible moves, so rollouts under-move. This is the main realism gap and the first thing to fix (below).

## What this does not show / limits
- One seed, one fit/validation split, no confidence intervals; the profile differences in Result 2 are small.
- The decision head is three-way (shoot / pass / keep), not "pass to which teammate", and rollouts do not move the ball.
- One Gaussian per step cannot represent "either cut left or cut right": multimodality is the likely cause of the under-speed. A mixture output, or predicting the destination and then the path, is the next experiment.
- Trained on 400k of about 3 million available half-court samples, with a 257k-parameter model, on CPU; a GPU would allow the full data, a larger model and several seeds.
- Half-court live states only: transition play, inbounds and dead-ball sequences are not modelled.
- Defenders are conditioned on their own profile only (not on the opposing coach's scheme); the opposing team's identity is not an input.

## Next (Stage 2 hardening, before the simulator)
1. Mixture-of-Gaussians (or destination-then-path) outputs to remove the under-speed; re-run the rollout check.
2. A pass-target head and a ball model (flight time, interception), so decisions can drive the ball in rollouts.
3. Several seeds and bootstrap intervals for the profile comparison; the full 401-game data.
4. Final Stage 2 scoring on the 100 holdout games (not yet done).
