# RL Stage 4a: possession simulator

Code: `Code/rl/sim.py` (simulator), `Code/rl/sim_validate.py` (validation), tests `Code/rl/tests/test_sim.py`. Results: `reports/rl-stage4a/` (`dev2/validate_val.json` = development grid, `holdout/validate_holdout.json` = the single holdout scoring).

## What it is
A vectorised environment: N possessions run in parallel from real half-court start states (real positions, lineup and profiles, clocks), stepped 0.2 s with the executor called every 0.4 s.
- **Players:** executor v2, all ten players, sampled mixture mode with one correlated draw per player per call.
- **Holder decision:** the executor's decision head (shoot / pass / keep) applied as a per-block hazard; pass target from the target head.
- **Ball:** `rl/ball.py` (held at the holder, 28 ft/s straight-line flight).
- **Pass outcome:** Bernoulli(`p_complete`) from the difficulty model at the release state; a failed catch ends the possession as a turnover on arrival. Option `mask_ood`: passes the difficulty model flags as out of distribution fail (the RL guard rail).
- **Shot:** the episode ends; reward is the shot model's xPts of that shot (guarded API); no rebounds.
- **Terminals/reward:** shot xPts; turnover or shot-clock violation -1.16; 30 s timeout.

## Calibration (development games only) and what was fitted
Three constants were fitted on the 81 development validation games (1,500 start states): a shoot-hazard scale 0.85, a pass-hazard scale 1.0, and a constant ball-loss hazard 0.03/s for non-pass turnovers (steals, lost balls, offensive fouls: the difficulty model only covers passes). Reason for the loss hazard: with it at 0 the simulated possessions end in a shot 87% of the time against 70% real, because the sim had no way to lose the ball except on a pass. These are calibration constants, not learned behaviour. Grids: `reports/rl-stage4a/*`, selection logged in `logs` (not tracked). Chosen config: all five dev statistics inside the real 95% interval.

## Holdout (100 games, 3,000 start states, config frozen before this run, scored once)
| statistic (from the start state to the first shot / possession end) | real [95% game-clustered CI] | simulator |
|---|---|---|
| possession ends in a shot | 0.707 [0.695, 0.722] | 0.721 |
| time to first shot (s) | 6.62 [6.44, 6.85] | 6.76 |
| passes before the first shot | 1.44 [1.39, 1.51] | 1.36 (just below) |
| passes per possession | 1.39 [1.34, 1.45] | 1.35 (just below) |
| points of the first shot (real: made x value; sim: xPts) | 0.985 [0.935, 1.046] | 0.983 |

Four of five are inside the real interval; passes are about 5% low. Not compared (no real counterpart prepared): shot distance and pass-length distributions, spatial distribution of shots, rebounds, and score by player.

## The guard rail costs realism
With the OOD mask on (passes the difficulty model flags are failed), the same config gives a shot share of 0.569 and 1.12 passes before the shot. The model flags a large share of ordinary passes (its ranges are the 0.5-99.5 percentiles of each of about a dozen features, so the union is wide), so masking by `in_distribution` as a hard fail is too blunt for training. Use it as a penalty or a mask on policy-chosen unusual actions, not on every sampled pass. This needs a decision at Stage 4b.

## Limits (read before using the reward)
- **Episodes start from real states** and last on average 6-8 s; there is no full-possession simulation (transition, early offence) yet.
- **No rebounds**: a miss ends the possession at xPts, so a policy cannot gain from offensive rebounds or be hurt by transition.
- **Mean xPts as reward** rather than a sampled make; no free throws or fouls drawn.
- **Hazard approximation** for the decision head (inferred): trained as the action in the next 2 s, applied per 0.4 s block as 1-(1-p)^0.2. Calibration scales absorb part of the error.
- **Movement is 15% too slow** (Stage 2b); the play-caller currently has no way to act on the simulator except through the executor's own behaviour (Stage 4b adds that).
- **Passes the model never saw**: the executor's target head can choose passes the difficulty model scores out of range; see the guard rail above and `reports/difficulty-exploit.md`.
- Calibration used one set of 3 constants for all teams and lineups; per-team realism was not checked.

## Verdict
The simulator reproduces the first-order shape of a half-court possession on unseen games (when a shot happens, how many passes precede it, how often the ball is lost, shot value) to within a few percent, after calibrating three constants. That is enough to start Stage 4b; it is not enough to trust differences smaller than those few percent, and the reward is an expected-points proxy, not real scoring.
