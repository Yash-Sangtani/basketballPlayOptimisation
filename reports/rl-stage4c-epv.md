# EPV (expected possession value) from tracking: a second attempt, and why it is not used

Code `Code/rl/epv.py`; numbers `reports/rl-stage4c/epv/epv_eval.json`; log not tracked.

**Question.** Does a neural state-value model on the full court state (10 players' positions and velocities, ball, holder, clocks; 400,000 fitting states from 320 games) predict the field-goal points the team in control scores in the rest of the possession, better than the Stage-5b feature set (shot clock, time in possession, ball distance to the basket and position)? The earlier value model in this project had R^2 of 0.0006-0.003.

**Result (81 validation games, 150,000 states).**
| model | R^2 on points scored in the rest of the possession |
|---|---|
| constant (mean) | -0.0004 |
| gradient boosting on shot clock + ball position | 0.0028 |
| neural network, full state | 0.0015 |

The network does not beat the simple model (MSE difference vs boosting -0.0016 [-0.0043, +0.0009], i.e. not better; vs constant +0.0022 [-0.0007, +0.0054]). A first, longer training (6 epochs) overfit: validation loss rose from 1.197 to 1.398 while training loss fell, because the 5 Hz states of one possession share one target; the reported run uses 2 epochs and dropout 0.2 (that setting was chosen after seeing the first run's validation curve, so it is development, not a clean test).

**Reading.** The target is the make/miss outcome, whose variance (about 1.2 points^2 per possession) is almost entirely luck given any state we can measure: R^2 near 0.003 is the signal there is. So a better architecture is not the missing piece; a lower-variance target would be (for example the expected points of the shots actually taken, from the shot model, which needs the shot features joined to the state table: not done).

**Decision, as agreed beforehand.** The condition for using a new EPV as reward was that it clearly beats the old one. It does not, so the reward stays shot xPts (turnover -0.10); EPV is not used in Stage 4c. Pass difficulty is used throughout (it decides every simulated catch and is a policy input).
**Why EPV would not have changed the reward anyway.** A potential-based shaping term with no discounting telescopes to V(start) - V(end), a constant for the policy; its only effect would be as a baseline for variance reduction, which needs a state value that is accurate, which this is not.
