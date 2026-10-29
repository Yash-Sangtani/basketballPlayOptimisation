# RL project: Stage 0 specification (problem definition and data layer)

Written 2026-10-28. Owner decisions that shape this: the agent is a **play-caller** for a full team (all five players move), the players
must **act like the real players** (their characteristics and how they play), it must **learn set plays**, and training goes **offline
first, simulator second**. Stage 1 (below) tests whether player-specific behaviour is learnable from this data at all.

## 1. Structure: a hierarchy, because "like the real players" and "learn set plays" are different problems
1. **Play-caller (the RL agent).** Every possession it chooses a *play* (an entry in a learned library, Stage 3) and may switch or deviate during the possession. It is the only part trained with RL.
2. **Player executors (imitation, not RL).** One model per player role, conditioned on a learned *player style* and on the current play call: given the state, it outputs that player's next move and, for the ball handler, the next decision (shoot / pass to k / keep). Trained by behaviour cloning on real tracking so they move and decide like the real players.
3. **Defence.** The same kind of model for the five defenders, conditioned on the defending team and players. It is the main source of risk (Stage 2 checks its realism).
4. **Outcome models (already built).** A pass reaches its target with probability `p_complete` (`Code/difficulty`, with its out-of-distribution mask); a shot is worth `xPts` (`shot_v1`, modest skill, guarded); a turnover costs `c_fail = 1.16` points.

## 2. The decision process
- **Time step** 0.2 s (5 Hz). **Episode** = one possession (starts when a team gains control, ends at a shot outcome, turnover, shot-clock violation, or a period end).
- **State** (attack-normalised: the offence always attacks the basket at (5.35, 25)): positions and velocities of the 10 players (slots 0-4 offence, 5-9 defence, each sorted by player id so a lineup keeps its slots), ball (x, y, z), holder, shot clock, period clock, time in possession, time since crossing midcourt, and the player ids and positions (G/F/C) behind every slot. Score margin is not in the data layer yet (needs the play-by-play join, Stage 3).
- **Actions.** Play-caller: choose play `c` from the library (+ a short list of overrides: "go early", "reset", "take the open shot"). Executors: per-player next displacement, limited by speed and acceleration measured from the data (99.9th percentile speed 21.6 ft/s; 0.1% of states above 30 ft/s are discarded as tracking glitches), and the holder's discrete decision.
- **Reward.** Points scored on the possession (sampled from `xPts`, or its mean for lower variance), minus `c_fail` on a turnover or shot-clock violation. No shaping by hand-made "good position" terms in Stage 1-3. Missed-shot rebounds are not modelled yet: the episode ends at the shot with its expected points; Stage 3 decides whether to add a rebound model.
- **Guard rails (from the exploit check):** the agent never places players (movement comes from the executors); passes with `in_distribution == False` or `extrapolated == True` are masked; `ev_pass` is not used.

## 3. Data layer (`Code/rl/dataset.py`, `load.py`, `splits.py`, `features.py`, `qa.py`)
One `.npz` per game under `data/rl/` (git-ignored; built from tracking, plus the pass and shot events that `difficulty/extract.py` already matches to frames, plus the roster with positions from the tracking JSON). Built for all 501 dev games on 2026-10-28 (about 10 s per game, 1.2 GB):

| quantity (mean per game unless stated) | value |
|---|---|
| states (5 Hz, ball tracked, a team in control) | 15,516 (7.77 million in total) |
| states dropped (missing player, no velocity, speed > 30 ft/s) | 226 (1.5%) |
| live-ball share of states (clock running) | 86.5% |
| half-court share (live, frontcourt, 1 s after midcourt) | 55.8% |
| tracking possessions / of which at least 4 s / reaching 4 s past midcourt | 275 / 228 / 202 |
| passes / shots matched to frames / field-goal points | 495 / 157 / 155 |
| mean speed offence / defence | 6.3 / 5.4 ft/s |
| players / players with at least 5,000 states / median player | 439 / 427 / 180,000 states |

**Known limits.** Tracking possessions are 1.3-1.4x the play-by-play count (about 193 per game) because dead-ball hand-offs, inbounds and loose-ball touches split possessions even after merging control runs shorter than 2 s; use `poss_id` to group states, not to count possessions. Some play-by-play shots cannot be matched to a tracking frame (2-25 unmatched per game were seen in individual games), so field-goal points are slightly below the true total; this was not measured across all games. Free throws are not in the reward yet. The data is the first half of one season; bench players have far less data than starters.

## 4. Splits (frozen; `Code/rl/splits.py`)
`train` = the 401 tuning games of the difficulty model; `holdout` = the 100 games held out there, **unused by any RL model until a stage's final evaluation**; `test` = `splits/test_games.txt` (130 games), reserved until the end of the RL project (the helper raises if asked for it). Stage-level numbers are cross-validated by game inside `train`.

## 5. Stage 1: is player-specific behaviour learnable here? (`Code/rl/identity_test.py`)
Two tasks on half-court live states, 1 Hz, `train` games only: **A** each offensive player's displacement over the next 1 s; **B** the ball handler's action in the next 2 s (shoot / pass / keep). Gradient-boosted models, GroupKFold by game, four variants: M0 situation only, M1 + position (G/F/C), M2 + player identity (categorical id, top 250 players), M3 control (the same ids assigned to random rows). Baselines for A: zero move and constant velocity.

**Decision rule, fixed before the 150-game result was read:** identity "helps" if M2 beats M1 by at least 1% (squared error for A, log loss for B) with a game-bootstrap 95% interval above zero **and** the M3 control does not show the same gain. If identity does not clear the bar on a task, the executors for that task use position-role conditioning only and the report says so; the full-data run (all 401 games) is the tie-breaker when the 150-game result is inside the noise. Results: `reports/rl-identity-results.md`.

## 6. What is not decided yet (Stage 2 onward)
How to represent a player's style (learned embedding versus explicit statistics such as usage, speed, shot profile); the movement-model architecture (set transformer or graph network over the 10 players); how plays are discovered and validated (Stage 3); a rebound model; whether the play-caller acts every possession or only at dead balls. Compute: this environment has 4 CPUs and no GPU; Stage 2 beyond a small prototype needs a GPU.

## 7. Stage status (2026-10-28)
| stage | what | status |
|---|---|---|
| 0 | problem definition, state/action/reward spec, data layer, splits | **completed** |
| 1 | is player-specific behaviour learnable? (identity, style statistics, shot profiles) | **completed** (`rl-identity-results.md`) |
| 2 | behaviour models for all ten players with profiles; multi-second rollout check | **completed as a CPU prototype** (`rl-stage2.md`); hardening items listed there are next |
| 2b | mixture outputs, pass-target head, seeds, holdout | **completed** (`rl-stage2b.md`) |
| 3 | set-play discovery and validation (possession clustering, play library) | **completed** (`rl-stage3.md`; clusters reproducible but weakly outcome-predictive; `docs/plays/`) |
| 4a | possession simulator (executors + difficulty and shot models + guard rails) | **completed** (`rl-stage4a.md`) |
| 4b | the play-caller RL inside the simulator | **completed as a decision-level policy** (`rl-stage4b.md`); full play-calling still open |
| 5 | audit and final evaluation (exploit check on visited states, holdout then test) | next |
