# RL Stage 1: can player-specific behaviour be learned from this data?

Run 2026-10-28 on the `train` games only (the holdout and test games were not read). Code: `Code/rl/identity_test.py`, `Code/rl/shot_profile_test.py`; raw outputs and logs in `reports/rl-identity*/`, `reports/rl-shot-profile/`. Setup and the decision rule fixed in advance: `reports/rl-spec.md` section 5.

## Answer in one paragraph
**Yes, players are individually identifiable, but their style lives in what they choose (where they shoot from, whether they shoot or pass), not in their second-by-second footwork.** A player's own shot-location profile predicts his held-out shots 6.5% better (log loss) than his position group does, and giving each player another player's profile makes the prediction worse, as it should. For 1-second movement and for the next-2-seconds ball-handler decision, the player-specific gain is real but small (about 0.6-0.7 points after subtracting a control), and a plain player-id feature does not help at all at this data size. So the executors should be conditioned on a compact **profile** (shot locations, shoot/pass rates, usage, preferred spots) at the level of intent, with generic kinematics underneath, rather than on a raw player embedding learned from per-step movement.

## Test 1 (movement): each offensive player's displacement over the next 1 s
150 train games, 867,040 player-states (half-court, live ball, someone holding it), 421 players; gradient boosting, GroupKFold by game. RMSE per axis in feet (zero-move baseline 4.31, constant-velocity baseline 2.22):

| variant | RMSE | squared-error change vs M1 (game-bootstrap 95% interval) |
|---|---|---|
| M0 situation only | 1.762 | position adds +0.82% [0.77, 0.86] over M0 |
| M1 + position (G/F/C) | 1.755 | (reference) |
| M2 + player id (categorical, top 250 players) | 1.759 | **-0.49%** [-0.57, -0.41] |
| M3 control: random player ids | 1.773 | -2.10% [-2.14, -2.06] |
| M4 + style statistics (8 per-player numbers, shrunk toward the position average, from training games only) | 1.751 | **+0.44%** [0.39, 0.49] |
| M5 control: the same statistics given to random players | 1.756 | -0.16% [-0.19, -0.13] |

The ids carry information (the control is 1.6 points worse than the real ids) but 250 categories overfit and the net effect is negative. Style statistics help a little: +0.44% against a control of -0.16%, i.e. about +0.6 points that are player-specific.

## Test 2 (decision): the ball handler's action in the next 2 s (shoot 15.7% / pass 36.0% / keep 48.3%)
150 train games, 178,048 holder states. Log loss (prior 1.010):

| variant | log loss | change vs M1 |
|---|---|---|
| M0 situation only | 0.8274 | position adds +0.39% [0.32, 0.46] |
| M1 + position | 0.8242 | (reference) |
| M2 + player id | 0.8291 | -0.59% [-0.84, -0.34] |
| M3 control: random ids | 0.8491 | -3.02% [-3.12, -2.91] |
| M4 + style statistics | 0.8141 | **+1.23%** [1.09, 1.37] |
| M5 control: random players' style | 0.8201 | +0.49% [0.39, 0.59] |

The control also gains +0.49%, so the player-specific part of the style gain is about +0.7 points, not 1.2.

## Test 3 (style where it should show): shot-location profile
401 train games, 60,984 matched shots by 434 players. Category of the shot's distance from the basket (rim < 5 ft 19%, short 5-14 ft 29%, mid 14-22 ft 22%, three 31%). Each held-out shot is predicted from the league rates, the shooter's position group, or the shooter's own training-game rates shrunk toward his position group with 50 pseudo-shots (GroupKFold by game):

| predictor | log loss | change |
|---|---|---|
| league rates | 1.3676 | |
| + position group | 1.3256 | +3.07% [2.86, 3.26] over league |
| **+ the player's own profile** | **1.2395** | **+6.50% [6.28, 6.74] over position** |
| control: profiles assigned to random players | 1.3543 | -2.16% [-2.34, -1.97] |

Players differ a lot: among the 119 shooters with at least 200 shots, the three-point share ranges from 3% (10th percentile) through 30% (median) to 49% (90th), and the rim share from 7% through 15% to 29%.

## Against the rule fixed in advance
- **Player-id categorical (M2 vs M1) must beat position by at least 1% with an interval above zero and a control that does not show it:** failed on both tasks (-0.49% and -0.59%).
- **Style statistics** are a different representation, tested afterwards because the control showed the ids do carry information. Net of the control they give about +0.6 (movement) and +0.7 (decision) points: real, below the 1% bar. I am reporting that as a pass for "learnable" and a fail for "large" at this granularity.
- The 150-game run was the tie-breaker I allowed; I did not run the full 401 games for tests 1-2 because the shot-profile result (on all 401) already answers the design question. A 401-game run would tighten the small gains but is unlikely to change the picture.

## What this means for the design
1. **Condition at the level of intent and choice.** Profile features (shot-location mix, shoot/pass rates as a handler, usage, preferred zones, speed and spacing habits) feed the *decision* models and the choice of destination; the per-step footwork can be generic plus position.
2. **Use low-dimensional, shrunk profiles estimated from training games only**, not a 250-way id. Bench players with little data fall back to their position group automatically.
3. **Set plays will carry much of the structure.** Because per-step movement is mostly explained by the situation (and position), the play call is likely a larger lever than individual style; Stage 3 tests this.
4. **Next step in Stage 2:** train neural executors with low-dimensional profile inputs and compare with the gradient-boosting results here; test destination-based targets (where he is 2-3 s later, where he shoots) instead of 1-second steps.

## Not done
No learned embedding (neural) tests; tests 1-2 on 150 of 401 train games; the control for the style statistics reuses one random permutation; defenders were not tested (all rows are offensive players); the half-court filter excludes transition play and set plays out of dead balls with the ball in the backcourt.
