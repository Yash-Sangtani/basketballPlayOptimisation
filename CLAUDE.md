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

**Bottom line:** the passing pipeline runs end-to-end on ONE game
(`12.12.2015 GSW at MIL`, `get_data('12.12.2015','MIL','GSW')`;
GSW=1610612744, MIL=1610612749) but the dataset it produces does **not yet
measure pass difficulty**. Do not batch-run it or train on it until the
"Next steps" below are done. `pass_model.py` has never run on real
multi-game data (no `Tracking_Data/pass_features/`; TensorFlow is not
installed in the local Python 3.12 env — pandas/sklearn/scipy are; `7z` is at
`/opt/homebrew/bin/7z`).

**Uncommitted work (staged, nothing committed since `693ec3a`):** bug fixes to
`Code/passtracking.py` plus `Code/_regenerate_fixed.py`, `Code/run_logged.sh`
and the `*_fixed.csv` outputs in `Tracking_Data/passes/`.
- Good fixes in that diff: `iloc[j,...]`→`iloc[end,...]` when tagging
  unsuccessful passes; `features.__init__` ignored its args (hardcoded game);
  target-defender distance was measured from the passer; ball was counted as
  a defender; zero-length pass rows skipped; `Tracking_data`→`Tracking_Data`
  in `passtracking.py`.
- **Regression in that diff: the steal-following `elif temp2 != self.ball_team`
  branch (`passtracking.py:115-120`) should be reverted/replaced.** It flips the
  tracked team whenever the nearest player is briefly an opponent, so the
  state machine swallows real passes and credits ~40% of detections to the
  wrong team (in the sample game 60% of shot-clock windows contain detected
  passes from *both* teams). Replay without it: 401 vs 317 raw passes, assist
  recall 20/51 vs 3/51 (sub-agent replay, not saved).
- Don't commit as-is: staged `.DS_Store` x2 and 43 MB `moments_fixed.csv` (no
  `.gitignore` exists; `.git` is already 3.7 GB because raw `.7z`/CSV data are
  tracked and a `.passes.csv.swp` is committed).

**Findings of the 2026-10-19 audit (three sub-agent audits + inline checks,
all on the one sample game; "M" = measured):**
1. *Label leakage (M).* Failed passes have `target=None`, so
   `features_extraction` computes every target feature against (0,0)
   (`passtracking.py:200,207-209,249-272`). All 34 failure rows have
   `ball_end_basket_dist == 25.57` exactly; no success does. Leave-one-quarter-out
   AUC is 0.99 with all features but 0.61–0.66 once target-derived features
   and `backcourt` are removed — i.e. there is no real signal yet.
2. *`backcourt` is inverted and conditions on the label (M).* 1 means the
   receiver is in the half being attacked (227/230); failures never get the flag
   written (stays 0). `pass_model.py:40` (`backcourt == 0`) keeps 33/34
   failures but only 69/298 successes.
3. *Failure label is not "failed pass" (M).* Rule = "a pbp turnover
   (`EVENTMSGTYPE 5`) within ±20 s of a possession end" (`:123-143`). Game has
   36 turnovers: 13 bad pass, 9 lost ball, rest OOB/travel/foul/shot clock; only
   ~14-15 true failed passes. Of the 34 rows only 7 have the right passer and 13
   the wrong team; the rule fires at the same rate as for arbitrary game
   seconds. Class balance 298:34 (real completion rate probably ~97%), so
   accuracy/F1 are meaningless here.
4. *Detection precision/recall are poor (sub-agent proxy detector, M vs a
   proxy; treat as ranges).* Nearest-player switching has no distance
   threshold, hold time or ball-height check: 30% of detections occur with ball
   z ≥ 10 ft (shots/rebounds), false-positive rate ~36–55%; only 3 of 53 pbp
   assists match a detected passer→scorer pass. Shot-clock gating
   (`:57-58`,`:78`) drops ~42% of possessions — the 80 "live" resets
   (23.6–23.99, e.g. after steals) never become windows; windows cover ~50% of
   game time and are biased to dead-ball endings.
5. *Features are measured at the wrong time/with wrong geometry (M).*
   Snapshot is taken at the detection frame — ~0.22 s after release, ~0.16 s
   before reception, ball >6 ft high in 56%. Lane-defender distance uses the
   infinite line, not the pass segment (wrong for 34% by >2 ft);
   `calculate_angle` returns the turning angle (π − interior angle); basket is
   picked from the target not attack direction; passer/target default to (0,0)
   if not found in the snapshot; 3 snapshots have no ball (`positions[0]` is
   a player).
6. *Evaluation (R).* Random row split across correlated passes; weights printed
   instead of importances; `NOO` skip list in `pass_model.py:23-31` can never
   match (`.7z` names vs `.pkl`, home/away order).

**Data facts (M):** 636 `.7z` archives (2015-10-27 → 2016-01-23, ~half a
season) but 632 events CSVs. 4 archives are 32-byte stubs with no events:
`01.14.2016.CHI.at.PHI`, `01.14.2016.CLE.at.SAS`, `01.14.2016.DET.at.MEM`,
`01.23.2016.UTA.at.WAS`; `01.23.2016.CHI.at.CLE` has events but 0 moments
(sub-agent) → ~631 usable. File names don't carry the game id; it comes from the
JSON `gameid`. Speed is not a bottleneck: ~3-4 s/game, ~0.9 GB RSS (sub-agent
replica) — never write per-game `moments` CSVs in a batch (43 MB × 631).

**Direction (proposed by audit, NOT yet decided by owner):** `pass_successful`
is the wrong target for play optimisation. Proposed: pass value =
P(complete)·V(receiver state) + (1−P)·V(turnover) − V(passer state), built as
(1) possession table with points, (2) V(s) model, (3) rebuilt pass detector,
(4) completion model, (5) ΔV per pass, then pass-vs-shoot. Minimum viable:
ΔxPts passer→receiver using Dhruv's per-player shot value, validated against
assists. Closest literature (from memory): Cervone et al. 2014/2016 (EPV),
Power et al. KDD 2017 "Not All Passes Are Created Equal".

**Next steps (priority order — confirm with owner before starting):**
- P0 housekeeping: add `.gitignore` (`.DS_Store`, `*.swp`, `Code/temp/`, `logs/`,
  generated `Tracking_Data/passes/*.csv`, `pass_features/`); unstage generated files;
  revert the `elif` steal branch.
- P0 detector v2 (rewrite `track_passes`): run over all frames (shot clock is a
  feature, not a gate); holder = nearest player within ~3 ft, ball z ≤ ~8 ft,
  held ≥4 frames, fill dropouts ≤6 frames; pass = same-team A→B hold segments
  with ≥~5 ft ball displacement (A→A is a dribble); exclude shots (flight z_max
  ≥ ~10.5 ft ending within ~20 ft of a rim); take features at **release**
  (last frame ball within ~3.5 ft of passer). Regression test: the 53-54 pbp
  assists of the sample game (now 3/53 recalled; target ≥85%).
- P0 label v2: `completed`=1 if a teammate controls the ball ≥0.2 s before any
  opponent; 0 for interception/deflection/OOB confirmed by pbp
  `EVENTMSGACTIONTYPE` 1 or 45 with `PLAYER1_ID` = passer within ~±2 s; drop
  shots/fouls/travels/shot-clock/strips. Target for failures = interceptor or
  ball end-of-flight point (never (0,0)). All features use frames ≤ release.
- P1 loader + batch: one loader module (stream `7z e -so`, game id from JSON, no
  shared `./temp`), manifest of ~631 games, resumable Pool of ~4 with a per-game
  status row; fail loudly if `pass_features/` is missing (the current
  `process_features` try/except hides it).
- P1 evaluation: GroupKFold by game, class weights, PR-AUC vs prevalence,
  log-loss/Brier/reliability, baselines (prevalence, distance-only logistic).
- P1/P2: decide whether to merge `origin/dhruv` first (see Known issues); then
  the possession table and V(s) model.

**Open questions for the owner:** (1) retrospective grading of decisions or live
recommendation? (2) 20-game dev set first, date-based test split? (3) merge
`origin/dhruv` first, and who owns the shared loader? (4) keep 3.7 GB of raw data in
git or move to LFS/DVC/bucket (history rewrite acceptable?) (5) does Dhruv still
have the trained `.keras` models/scalers (none are on `origin/dhruv`)?
**Planned changes the owner wanted to implement (not yet recorded — add here).**

**Working notes for a fresh session:**
- Run everything from `Code/` (relative paths). `get_data` extracts into the
  shared relative `./temp` and takes `game_id` from whatever JSON is there →
  never run two pipeline processes at once from the same directory.
- Logged runs: `./run_logged.sh <name> python3 <script>.py` → `../logs/`.
  Regenerate the sample game: `./run_logged.sh regen python3 _regenerate_fixed.py`
  (writes `*_fixed.csv`; the unsuffixed CSVs in `Tracking_Data/passes/` are the
  pre-fix originals — keep them for comparison).
- `moments` after `track_passes()`, positional `iloc` columns: 0 quarter,
  1 universe_time, 2 quarter_time (s left), 3 shot_clock, 4 positions (list of
  `[team_id, player_id, x, y, z]`, ball first as `-1,-1`), 5 game_time,
  6 pass_successful, 7 passer, 8 target, 9 team_passer, 10 team_target,
  11 backcourt. Court is 94×50 ft; baskets at `[5.35,25]` and `[88.65,25]`.
- The audit's scratch scripts lived in a temp dir and were not saved; the
  numbers above are the record. Re-derive from `Tracking_Data/passes/*_fixed.csv`.

## Directory layout

- **`Tracking_Data/`** — raw data, shared by both pipelines.
  - `game logs/` — per-game SportVU tracking data as `.7z` archives, named
    `MM.DD.YYYY.AWAY.at.HOME.7z`. Each archive contains one JSON with
    player/ball x-y-z positions per frame ("moments").
  - `events/` — play-by-play CSVs per game, named `<game_id>.csv`.
  - `player/` — player ratings and shot percentages (used as features).
  - `shots/` — shot zone data (`shots.csv`).
  - `passes/` — output of `Code/passtracking.py`'s `main()`: extracted
    moments/pbp/pass-attempt/feature CSVs for a single sample game.
  - `setup.sh` — installs `p7zip` and extracts all `.7z` archives in place.
- **`Code/`** — passing pipeline.
  - `getting_data.py` — `get_data` class: extracts one game's `.7z` +
    matching play-by-play CSV, builds the `moments` DataFrame.
  - `passtracking.py` — `PassTracking` class detects individual passes from
    `moments`; `features` class extracts pass-geometry features from
    detected passes; `main()` runs this for one sample game.
  - `pass_model.py` — the Keras trainer: loads per-game feature pickles from
    `Tracking_Data/pass_features/`, trains an MLP to predict `pass_successful`,
    reports ROC AUC / F1 / accuracy. Not yet run on real data (see status).
    (`pass_difficulty_model.py` is the dead draft — see Known issues.)
  - `_regenerate_fixed.py` — reruns the sample game and writes `*_fixed.csv`
    to `Tracking_Data/passes/`; `run_logged.sh` — runs a command and saves a
    timestamped log to `logs/`. `temp/` is scratch used by `get_data`.
  - `allgames.txt` is **not present in `Code/`** (only in `Shots/`). The
    `open('./allgames.txt')` at the top of `passtracking.py:main()` is NOT
    commented out (only the per-game loop is), so `main()` raises
    `FileNotFoundError` from `Code/`; `pass_model.py` also reads it at import.
    Use `_regenerate_fixed.py` as the working entry point.
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
→ `PassTracking.track_passes()` walks `moments` possession-by-possession,
using the shot-clock reset pattern to find possession boundaries and
tracking who is closest to the ball each frame to detect pass events
(successful passes and turnovers, matched against play-by-play `EVENTMSGTYPE`
codes) → `features.features_extraction()` computes pass-geometry features
per detected pass (distance, angle to basket, closest-defender distance to
passer/target, distance from closest defender to the pass trajectory line)
→ `pass_model.py` trains a Keras model on those features to
predict `pass_successful` (see "Current status": the labels/features
produced by this flow are currently leaky and mostly mislabelled).

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
- **`Tracking_data` vs `Tracking_Data` path case.** Still lowercase at
  `Code/getting_data.py:114` (play-by-play folder) and `Code/pass_model.py:27`.
  Works on macOS (case-insensitive) but breaks on Linux.
- **`Shots/shot_tracking.py` has hardcoded Windows paths.** Data paths
  (`ratings`, `zones`, `fg_percentage` at module load, plus the tracking/pbp
  folders inside `simulate.py`) are absolute paths under
  `C:/Users/Dhruv/PycharmProjects/basketballPlayOptimisation/...`. These need
  to become relative or configurable before this pipeline runs on any other
  machine.
- **`origin/dhruv` has unmerged shot-pipeline work.** Diverged from `main`
  at commit `18611fa`. It adds EPV (expected possession value) support to
  `Shots/simulate.py` (`show_epv` param on `animate_play`/`watch_play`,
  `extract_shot_probability`), plus two new files not on `main`:
  `Shots/zone_model.py` and `Shots/shot_tracking_plot.py`. None of this has
  been merged into `main`.
