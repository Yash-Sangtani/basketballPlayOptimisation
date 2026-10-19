# basketballPlayOptimisation

Analyzes NBA SportVU player-tracking data from the 2015-16 season to model
passing quality and shot quality. Two independent pipelines built by
different contributors, sharing one raw data source:

- **`Code/`** (Yash) — passing pipeline: detects passes from tracking data
  and trains a model predicting pass difficulty/success.
- **`Shots/`** (Dhruv) — shooting pipeline: detects shot attempts and trains
  a model predicting shot quality/make probability, plus standalone spacing
  and velocity analysis.

The two pipelines don't share code — each has its own copy of the
tracking-data-loading logic (`Code/getting_data.py`'s `get_data` vs
`Shots/simulate.py`'s `Game`) — but both read from the same `Tracking_Data/`.

## Current status — read this first when resuming (last updated 2026-10-19)

**Bottom line:** on 2026-10-19 the passing pipeline (`Code/passtracking.py`) was
rewritten after an audit showed the old dataset did not measure pass
difficulty. It now yields ~450–550 completed passes and ~6–13 failed passes
per game (matched from 7–16 pbp bad-pass turnovers) with no label leak. It has
been run on four games only: `12.12.2015 GSW@MIL` and `12.30.2015 PHI@SAC` (used
while building, so in-sample) and `11.20.2015 SAS@NOP` and `01.12.2016 HOU@MEM`
(held out, not retuned). The failure signal is still weak (17–21 failures per
pair of games) and `pass_model.py` has not been run at all (TensorFlow is not
installed in the local Python 3.12 env; pandas/sklearn/scipy are; `7z` is at
`/opt/homebrew/bin/7z`).

**Git state:** all on `main`, nothing pushed (~10 commits ahead of
`origin/main`). `.gitignore` + status docs (`519d73a`), `origin/dhruv` merged
(`bf64535`), then the passtracking rewrite, its docs, and a follow-up (symmetric
flight filter + attack-direction fix + held-out validation). Run
`git log --oneline -8` to see where things are.

**How detection works now** (constants are class attributes of `PassTracking`):
ball-holder state machine over every frame (shot clock is a feature, not a
gate). Holder = nearest player within 3 ft, ball z ≤ 8 ft, ball speed relative
to the holder ≤ 15 ft/s (drops passes flying past a player), held ≥ 4 frames,
dropouts ≤ 6 frames filled, never across a time gap. *Completed pass* = hold by
A then (≤ 4 s later) hold by teammate B, ball travel ≥ 5 ft, in bounds, not a
shot. *Failed pass* = pbp bad-pass turnover (`EVENTMSGTYPE 5`, action 1 or 45)
matched to the committer's hold (window −3..+8 s because the pbp clock is often
2–5 s off tracking; a candidate whose next holder is the credited stealer wins);
flight ends in an interception, out of bounds, or a fumbled teammate touch. Lost
balls/travels/fouls are NOT passes. All features are computed at the **release**
frame (last frame of the passer's hold). A failed pass has no receiver, so its
`target` is `target_heading` = the teammate the ball is heading towards in its
first 8 frames (0.32 s) after release (≤ 50° off), so the *target choice* peeks
slightly past release even though every geometric feature is measured at release;
the same rule on completions agrees with the true receiver 75–85% of the time. `features(..., target_mode='actual'|'heading')`:
'actual' (default) uses the true receiver for completions, 'heading' uses the
heading rule for both classes (leak-safest; gives the same leakage picture).
`backcourt` = passer released in his own half (|x − attacked basket x| > 47 ft;
attacked basket voted per team/quarter and swaps at halftime), defined for both
classes, so `pass_model.py`'s `backcourt == 0` filter now keeps frontcourt passes.

**Validation (`Code/validate_passes.py`, measured; "before" = the old code on
12.12.2015). Games 1–2 shaped the design (thresholds, fumble rule, filters), so
only the held-out column is a fair out-of-sample look; it was run once, with
no tuning, after fixing the one bug it exposed (see below).**

| check | before | in-sample (2 games) | held-out (2 games) |
|---|---|---|---|
| assist recall (pbp assist = passer→scorer) | 3/53 (6%) | 72%, 67% | 81%, 81% |
| minority-team share of passes in a shot-clock window | 28% | 8%, 13% | 7%, 7% |
| windows with repeated team flips (detection error) | not measured | 1, 1 | 3, 5 |
| pbp bad-pass turnovers matched | 7/34 rows had the right passer | 9/15, 13/16 | 6/7, 11/16 |
| heading rule vs true receiver on completions | – | 78%, 75% | 80%, 85% |
| failures in feature rows (rate) | 34 (10%) | 21 (2.1%) | 17 (1.7%) |
| max single-feature AUC for `pass_successful` | 1.00 (sentinel 25.57) | 0.79 | 0.79 |
| leave-one-game-out failure ROC, logistic / gboost | 0.99 | 0.72–0.76 / 0.57 | 0.72 / 0.49 |

Reproduce: `cd Code && ./run_logged.sh validate python3 validate_passes.py
11.20.2015.SAS.at.NOP 01.12.2016.HOU.at.MEM` (add `--target-mode heading` to
compare). Read the output; thresholds are not tuned to hit a number. Assist
recall is 67–81%, below the 85% the audit proposed; the misses are mostly
hand-offs/lobs excluded by design or an assister with no ≥ 4-frame hold.

Bugs found by this process: (1) the held-out game exposed both teams being
assigned the same basket in Q4 (one team's own vote was wrong) — attack
direction is now scored jointly per quarter (home margin − away margin) so teams
always attack opposite baskets; (2) failed passes were selected by a looser
filter than completions (2 under 5 ft of ball travel, 3 lobs ending near a rim),
which would let the classes differ by construction — both classes now go through
the same `_flight_ok` filter, which cost ~5 of 27 matched failures.

**Known limits of the current dataset:**
1. ~2% of passes are failures (21 in 992 rows and 17 in 1010 rows, per pair of
   games) → hundreds of games needed; accuracy/F1 are meaningless, use PR-AUC grouped by game.
2. Assist misses are by design (hand-offs < 5 ft, lobs/alley-oops caught above
   8 ft) or the assister has no ≥ 4-frame hold. 2 of ~15 pbp bad passes per game
   have no committer hold in tracking and cannot be matched.
3. Residual mixed-team windows are almost all a single clean switch (possession
   changes the shot-clock check can't see), not flipping.
4. Thresholds were tuned on games 1–2 (`HOLD_REL_SPEED` on game 1;
   `HEAD_MAX_ANGLE`/`HEAD_FRAMES` on both). Failures skew frontcourt (about 2–3%
   of frontcourt passes vs 0.7% of backcourt passes): possibly real, watch it
   on more games. The gradient-boosted model is at chance leave-one-game-out
   (0.49–0.57) and the logistic one reaches ~0.72 on only 17–21 failures, so
   treat any model result as noise until there are hundreds of games.
   `HOU@MEM` still had 5 shot-clock windows with repeated team flips.
5. Features changed meaning: all at release; lane distance is to the pass
   *segment*; angles are interior angles at the vertex (small = defender in the
   passing direction); new `lane_n_def_5ft`, `passer_speed`, `target_speed`;
   `shot_clock` is filled with the game clock when the shot clock is off and
   ≤ 24 s remain (other NaNs remain, ~5%). `features.meta` (game_id, quarter,
   game_time, frame, passer, target, outcome, target_inferred) is NOT saved by
   `process_features` — save it next to the pickles before pooled training.
6. `Code/pass_model.py` itself is unchanged apart from the path-case fix: random
   row split (should be GroupKFold by game), accuracy/F1 on imbalanced data,
   printed weights, the `NOO` skip list can never match (`.7z` names vs `.pkl`),
   and it reads `./allgames.txt` at import (not in `Code/`; use
   `../Shots/allgames.txt`).

**Data facts (M):** 636 `.7z` archives (2015-10-27 → 2016-01-23, about half a
season) but 632 events CSVs. 4 archives are 32-byte stubs with no events:
`01.14.2016.CHI.at.PHI`, `01.14.2016.CLE.at.SAS`, `01.14.2016.DET.at.MEM`,
`01.23.2016.UTA.at.WAS`; `01.23.2016.CHI.at.CLE` has events but 0 moments →
~631 usable (the batch prints a traceback for the others and carries on). Speed
is not a bottleneck: ~3.2 s/game measured. File names don't carry the game id;
it comes from the JSON `gameid`. Never write per-game `moments` CSVs in a batch
(43 MB × 631).

**Direction (proposed by audit, NOT yet decided by owner):** `pass_successful`
is the wrong target for play optimisation. Proposed: pass value =
P(complete)·V(receiver state) + (1−P)·V(turnover) − V(passer state), built as
(1) possession table with points, (2) V(s) model, (3) detector (done),
(4) completion model, (5) ΔV per pass, then pass-vs-shoot. Minimum viable:
ΔxPts passer→receiver using Dhruv's per-player shot value, validated against
assists. Closest literature (from memory): Cervone et al. 2014/2016 (EPV),
Power et al. KDD 2017 "Not All Passes Are Created Equal".

**Next steps (priority order — confirm with owner before starting):**
- P0 run a dev batch (~20 games): `cd Code && python3 passtracking.py
  ../Shots/allgames.txt` writes `Tracking_Data/pass_features/*.pkl` (ignored by
  git); re-run `validate_passes.py` on a few new games to check the thresholds
  hold up; fix whatever breaks.
- P0 fix `pass_model.py` evaluation (GroupKFold by game, PR-AUC, class weights,
  calibration) and save `features.meta` with the pickles.
- P1 loader + batch: one loader module (stream `7z e -so`, game id from JSON, no
  shared `./temp`), manifest of ~631 games, resumable Pool of ~4 with a per-game
  status row.
- P1 clean up the merged `Shots/` code (see Known issues), then the possession
  table and V(s) model.

**Open questions for the owner:** (1) retrospective grading of decisions or live
recommendation? (2) 20-game dev set first, date-based test split? (3) keep 3.7 GB
of raw data in git or move to LFS/DVC/bucket (history rewrite acceptable)? (4)
does Dhruv still have the trained `.keras` models/scalers (none are tracked)?
**Planned changes the owner wanted to implement (not yet recorded — add here).**

**Working notes for a fresh session:**
- Run everything from `Code/` (relative paths). `get_data` extracts into the
  shared relative `./temp` and takes `game_id` from whatever JSON is there →
  never run two pipeline processes at once from the same directory.
- Logged runs: `./run_logged.sh <name> python3 <script>.py` → `../logs/`.
  `_regenerate_fixed.py` re-runs the sample game and writes `*_v2.csv` to
  `Tracking_Data/passes/` (ignored by git). The unsuffixed CSVs there are the
  pre-fix originals (tracked); `*_fixed.csv` are from the pre-rewrite code.
- `PassTracking.pass_attempt` (one row per pass, at the release frame):
  `frame, recv_frame, end_frame, quarter, universe_time, quarter_time,
  shot_clock, positions, game_time, pass_successful, passer, target,
  team_passer, team_target, backcourt, target_inferred, target_heading, outcome
  (completed/intercepted/out_of_bounds/fumbled), flight_s, basket_x`. Frame
  arrays live in `PassTracking._fr`; hold segments in `.segments`; pbp matching
  stats in `.failure_report`. `moments` frame columns by position: 0 quarter,
  1 universe_time, 2 quarter_time (s left), 3 shot_clock, 4 positions (list of
  `[team_id, player_id, x, y, z]`, ball first as `-1,-1`), 5 game_time. Court is
  94×50 ft; baskets at `[5.35,25]` and `[88.65,25]`.
- Audit scratch scripts were not saved; `validate_passes.py` is the durable check.

## Directory layout

- **`Tracking_Data/`** — raw data, shared by both pipelines.
  - `game logs/` — per-game SportVU tracking data as `.7z` archives, named
    `MM.DD.YYYY.AWAY.at.HOME.7z`. Each archive contains one JSON with
    player/ball x-y-z positions per frame ("moments").
  - `events/` — play-by-play CSVs per game, named `<game_id>.csv`.
  - `player/` — player ratings and shot percentages (used as features).
  - `shots/` — shot zone data (`shots.csv`).
  - `passes/` — outputs for the single sample game: tracked unsuffixed CSVs are
    the pre-fix originals; `*_fixed.csv` (old code) and `*_v2.csv` (current
    code, from `_regenerate_fixed.py`) are git-ignored.
  - `pass_features/` — per-game feature pickles written by the batch run
    (git-ignored; does not exist until you run it).
  - `setup.sh` — installs `p7zip` and extracts all `.7z` archives in place.
- **`Code/`** — passing pipeline.
  - `getting_data.py` — `get_data` class: extracts one game's `.7z` +
    matching play-by-play CSV, builds the `moments` DataFrame.
  - `passtracking.py` — `PassTracking` detects completed and failed passes
    from `moments` (see "Current status" for how); `features` extracts
    release-time pass-geometry features; `main()` runs the sample game, or
    batch-processes a game list given as argv[1].
  - `validate_passes.py` — acceptance checks (assist recall, precision proxies,
    failure matching, leakage tests, attack direction) for any games.
  - `pass_model.py` — the Keras trainer: loads per-game feature pickles from
    `Tracking_Data/pass_features/`, trains an MLP to predict `pass_successful`,
    reports ROC AUC / F1 / accuracy. Not yet run on real data (see status).
    (`pass_difficulty_model.py` is the dead draft — see Known issues.)
  - `_regenerate_fixed.py` — reruns the sample game and writes `*_v2.csv`
    to `Tracking_Data/passes/`; `run_logged.sh` — runs a command and saves a
    timestamped log to `logs/`. `temp/` is scratch used by `get_data`.
  - `allgames.txt` is **not present in `Code/`** (only in `Shots/`).
    `passtracking.py` takes a list path as argv[1] (e.g. `../Shots/allgames.txt`);
    `pass_model.py` still reads `./allgames.txt` at import.
- **`Shots/`** — shooting pipeline.
  - `simulate.py` — `Game` class: shooting-side equivalent of `get_data`,
    with added plotting/animation methods (`plot_frame`, `animate_play`,
    `watch_play`) for visualizing individual plays.
  - `shot_tracking.py` — extracts shot features (shooter/defender distance,
    angle, velocity, zone FG%, score margin, etc.) per shot event.
  - `shot_model.py` — trains a Keras MLP on those shot features.
  - `spacing.py` — convex-hull-based team spacing analysis (standalone).
  - `velocity.py` — player/team velocity analysis (standalone).
  - `allgames.txt` — list of games in `MM.DD.YYYY.AWAY.at.HOME` form, used
    by the batch-extraction loops in this directory.

## Core data flow

**Passing (`Code/`):**
`get_data(date, home, away)` extracts the game's `.7z` into a `moments`
DataFrame (one row per tracking frame: quarter, game clock, shot clock,
player/ball positions) plus the play-by-play CSV
→ `PassTracking.track_passes()` runs a ball-holder state machine over every
frame (nearest player, ball low and moving with him, held for a few frames) to
find completed passes between teammates, and matches play-by-play bad-pass
turnovers (`EVENTMSGTYPE 5`, action 1/45) to the committer's flight to get
failed passes → `features.features_extraction()` computes release-time
pass-geometry features per pass (distance, angle to the attacked basket,
closest-defender distance to passer/target, defender distance to the pass
segment, defenders in the lane, speeds) → `pass_model.py` trains a Keras model
on those features to predict `pass_successful` (details and validation numbers
in "Current status").

**Shooting (`Shots/`):**
`Game(date, team1, team2)` (`simulate.py`) is the shooting-side equivalent
of `get_data`, with the same `moments`/`pbp` structure plus visualization
methods → `shot_tracking.py` extracts per-shot features (shooter/defender
position, distance, angle, velocity, zone FG%) → `shot_model.py` trains a
Keras model to predict shot outcome. `spacing.py` and `velocity.py` are
standalone analyses over `Game` and don't feed the model.

## Dependencies

No `requirements.txt` exists yet. Inferred from imports across both
pipelines:

- `pandas`, `numpy`
- `scikit-learn` (`train_test_split`, `StandardScaler`, metrics)
- `tensorflow` (Keras — `Sequential`, `Dense`, `Dropout`, `Adam`)
- `matplotlib`, `seaborn`
- `scipy` (`ConvexHull` in `Shots/simulate.py` and `spacing.py`)
- `tqdm`

System binaries (not pip-installable): `7z`/`p7zip` (tracking data
extraction — see `Tracking_Data/setup.sh`), `ffmpeg` (play animation/video
export in `Shots/simulate.py`).

## Known issues

- **`Code/pass_difficulty_model.py` is dead** (earlier versions of this file
  said `pass_model.py` was — that was backwards). It is an abandoned draft of
  `PassTracking`, superseded by `Code/passtracking.py`: it references undefined
  names (`moments_df`, `find_ball_man` as a bare function) and will not run.
  Don't use it as a reference for how pass tracking works — use
  `passtracking.py`. The live Keras trainer is `Code/pass_model.py`.
- **Path case is fixed in `Code/`** (`Tracking_Data` everywhere) but check other
  files before running on Linux; macOS is case-insensitive and hides this.
- **`Shots/` has hardcoded Windows paths.** `C:/Users/Dhruv/PycharmProjects/...`
  appears in `simulate.py` (2), `shot_tracking.py` (3) and `shot_tracking_plot.py`
  (2) (data paths such as `ratings`, `zones`, `fg_percentage` at module load, and
  the tracking/pbp folders). Make them relative or configurable before running
  this pipeline on any other machine.
- **`origin/dhruv` is merged into `main`** (`bf64535`; adds EPV support to
  `Shots/simulate.py` — `show_epv` on `animate_play`/`watch_play`,
  `extract_shot_probability` — plus `Shots/zone_model.py` and
  `Shots/shot_tracking_plot.py`). Verified problems in the merged code (not yet
  fixed, nothing in `Shots/` was changed by the merge or since):
  - `simulate.py:14` imports `shot_tracking_plot`, which imports TensorFlow and
    loads `shot_model_v2.keras` / `zone_model_v2.keras` from the current dir;
    **no model files are tracked**, so EPV plotting cannot run until they are
    retrained or provided.
  - `simulate.py:16` forces `matplotlib.use('TkAgg')`; `shot_tracking.py:386`
    runs `write_features(extract_games())` at import time.
  - Coordinates: `simulate.py:484` shifts y by −50 for plotting while
    `shot_tracking.py:181` uses basket y = 25. Not yet checked whether the shot
    features consume shifted coordinates (a design-agent report said the committed
    `Shots/shot_features/` shows a layup at 48.9 ft, which suggests they do).
- **A `.passes.csv.swp` swap file is still tracked** in `Tracking_Data/passes/`
  (`git rm --cached` it); `.git` is 3.7 GB because raw `.7z`/CSV data are tracked.
