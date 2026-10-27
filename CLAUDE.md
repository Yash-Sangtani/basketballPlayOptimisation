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

> **2026-10-27: see `HANDOFF.md` first.** It holds the latest state (all branches merged into `main`, what is finished, what was never run, the next steps and the cloud setup). The text below is the 2026-10-19 status and is still accurate for the detection design, validation and data facts.

**Bottom line:** on 2026-10-19 the passing pipeline (`Code/passtracking.py`) was
rewritten after an audit showed the old dataset did not measure pass
difficulty. It now yields ~450–550 completed passes and ~6–13 failed passes
per game (matched from 7–16 pbp bad-pass turnovers) with no label leak. It has now
been run on **all 631 usable games** (306,782 passes, 5,025 failures = 1.64%; 32 min
44 s serial). Only the first four games shaped the design (two in-sample, two
held out), so the other 627 are untouched by tuning. Evaluated grouped by game
(`Code/eval_by_game.py`), failure prediction reaches ROC-AUC 0.85–0.86 and PR-AUC
0.10–0.12 against a 0.016 base rate: real signal, modest precision (details in
"Full batch" below; the earlier two-game 0.72/0.49 was small-sample noise).
`pass_model.py` ran on all games too, but its accuracy/F1 only equal always
guessing "completed", so ignore them. Python env: use the git-ignored venv
`.venv/bin/python` (Python 3.12, TensorFlow 2.21, Keras 3.15, pandas, sklearn);
the miniforge base python has no TensorFlow. `7z` is at `/opt/homebrew/bin/7z`.
Machine: Apple M1, 8 cores, **8 GB RAM**, disk ~98% full (free space swung between
2.8 and 14 GB on 2026-10-19).

**Git state:** all on `main`, and `main` is NOT pushed. The code up to `bf0110a`
is on GitHub as the branch `cloud-full-batch` (pushed 2026-10-19 for a cloud run;
GitHub `main` is still `693ec3a`). History: `.gitignore` + status docs
(`519d73a`), `origin/dhruv` merged (`bf64535`), the passtracking rewrite and
follow-ups, the resumable batch (`bf0110a`), then a docs + `eval_by_game.py`
commit. Run `git log --oneline -8` to see where things are. The full-batch
results (`Tracking_Data/pass_features/`, 42 MB) are local only; nothing from them
has been pushed.

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

**Full batch (2026-10-19, all games, M1 8 GB):** 631 games ok, 5 failed (the 4 stub
archives plus `01.23.2016.CHI.at.CLE`, as expected), 3.06 s/game mean (4.5 s max),
306,782 passes (mean 486/game), 5,025 failures (mean 8/game, max 23, one game 0),
42 MB of pickles, all stamped code version `d969a5a8`. A sub-agent ran it (locally,
see the cloud gotcha in Working notes); the lead session re-checked the totals
against the pickles and the table below against the agent's saved output.
`Code/eval_by_game.py`: GroupKFold(5) by game, positive class = failed pass,
imputer/scaler fit per fold, CIs from a game-clustered bootstrap (200 resamples).

| rows | model | pooled ROC (95% CI) | pooled PR-AUC (95% CI) | no-skill PR |
|---|---|---|---|---|
| all: 306,782 passes, 5,025 failures | logistic (balanced) | 0.851 [0.847, 0.855] | 0.102 [0.096, 0.108] | 0.0164 |
| | gradient boosting | 0.856 [0.852, 0.861] | 0.117 [0.111, 0.123] | |
| `backcourt == 0`: 216,292 passes, 4,583 failures | logistic | 0.826 [0.821, 0.832] | 0.113 [0.107, 0.120] | 0.0212 |
| | gradient boosting | 0.832 [0.827, 0.838] | 0.121 [0.115, 0.129] | |

Both models are well above chance (PR-AUC 5–7× the base rate), gradient boosting
slightly better. In absolute terms a PR-AUC of ~0.11 means most flagged passes are
not failures. Leak check: largest single-feature AUC 0.786 (`closest_def_trajectory`),
`lane_n_def_5ft` 0.785, nothing near the 0.85 alarm level, missing-value rates
similar across classes. **Caveat (inferred, not tested):** failures are labelled by
matching pbp turnovers to detected flights while completions come from hold-to-hold
detection, so a signal like "a defender is on the pass line" is plausible but part
of it may be selection. Read the AUC as "predicts what the detector labels as
failed", not as proven pass difficulty.

**Known limits of the current dataset:**
1. ~1.6% of passes are failures (5,025 of 306,782). Accuracy/F1 are meaningless
   (always guessing "completed" scores ~98.4%); use PR-AUC grouped by game.
2. Assist misses are by design (hand-offs < 5 ft, lobs/alley-oops caught above
   8 ft) or the assister has no ≥ 4-frame hold. 2 of ~15 pbp bad passes per game
   have no committer hold in tracking and cannot be matched.
3. Residual mixed-team windows are almost all a single clean switch (possession
   changes the shot-clock check can't see), not flipping.
4. Thresholds were tuned on games 1–2 (`HOLD_REL_SPEED` on game 1;
   `HEAD_MAX_ANGLE`/`HEAD_FRAMES` on both). Failures skew frontcourt (2.1% of
   frontcourt passes vs 0.5% of backcourt ones: 4,583 vs 442 failures), which may
   be real. Detection still flips teams in a few shot-clock windows per game
   (5 in `HOU@MEM`).
5. Features changed meaning: all at release; lane distance is to the pass
   *segment*; angles are interior angles at the vertex (small = defender in the
   passing direction); new `lane_n_def_5ft`, `passer_speed`, `target_speed`;
   `shot_clock` is filled with the game clock when the shot clock is off and
   ≤ 24 s remain (other NaNs remain, ~5%). `features.meta` (game_id, quarter,
   game_time, frame, passer, target, outcome, target_inferred) is NOT saved by
   `process_features` — save it next to the pickles before pooled training.
6. `Code/pass_model.py`: the model and evaluation are unchanged (random row
   split — `eval_by_game.py` is the honest evaluation; accuracy/F1 on imbalanced
   data; unweighted loss; printed weights; the `NOO` skip list can never match,
   `.7z` names vs `.pkl`). It now takes the game list as argv[1] (default `../Shots/allgames.txt`), skips games
   with no pickle, prints `[time]` stamps, and checkpoints each epoch.

**!! iCloud placeholders (M, 2026-10-19): most of the raw data is NOT on this Mac.**
The repo sits in `~/Documents`, which syncs to iCloud with storage optimisation:
589 of 636 archives and 577 of 632 events CSVs are "dataless" placeholders (check
with `ls -lO`; flag `dataless`). Reading one does not error — it silently returns
**0 bytes**, so 7z says "Can not open the file as archive" and Python reads
nothing. Only 47 archives are physically present. `get_data` now detects this and
raises `DatalessFileError`; the batch logs such games as `unavailable` (retried on
the next run, unlike `failed`). Two ways to get the data: (a) the real bytes are in
git (archives are tracked), so `Code/fetch_games.sh <list> [dest]` copies games
out of git history into `~/.cache/basketballPlayOptimisation/Tracking_Data`
(outside iCloud, works offline, 1 s for 7 games) — then `export
TRACKING_DATA_DIR=<dest>` before running; (b) download them in Finder
("Download Now" / "Keep Downloaded") or `brctl download` — about 3.8 GB, and the
disk is ~98% full so macOS may evict them again (it already evicted git's own pack file). Cleanest long-term fix: move
the repo out of iCloud-synced `~/Documents`.

On 2026-10-19, with free disk down to ~3 GB, macOS also evicted **git's own pack
file** (`.git/objects/pack/pack-*.pack`, 3.9 GB, now `dataless`). Git kept working
through it, slowly (one commit took ~8 min), but this is fragile. Nothing is lost:
iCloud still holds the file and every commit up to `bf0110a` is on GitHub in
`cloud-full-batch`. If git starts failing with pack errors, re-clone from GitHub
and re-apply any uncommitted work.

**Data facts (M):** 636 `.7z` archives (2015-10-27 → 2016-01-23, about half a
season) but 632 events CSVs. 4 archives are 32-byte stubs with no events:
`01.14.2016.CHI.at.PHI`, `01.14.2016.CLE.at.SAS`, `01.14.2016.DET.at.MEM`,
`01.23.2016.UTA.at.WAS`; `01.23.2016.CHI.at.CLE` has events but 0 moments →
~631 usable (the batch logs a failure for the others and carries on). Speed is not
a bottleneck: 2.8–3.9 s/game measured over 7 games (~23 s total; ~35 min serial
for all 631), 0.86 GB peak RSS per extraction process, so on 8 GB RAM at most ~3
parallel workers. File names don't carry the game id;
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
- P0 replace `pass_model.py`'s evaluation with grouped-by-game metrics (PR-AUC,
  class weights, calibration; start from `eval_by_game.py`) and save
  `features.meta` (game_id etc.) next to the pickles so pooled models can group
  by game.
- P0 move the repo out of iCloud-synced `~/Documents` and free disk (see the
  iCloud section): the disk is ~98% full and macOS has already evicted git's own
  3.9 GB pack file.
- P1 improve detection using the full-batch data (assist recall 67–81%, repeated
  team flips) and test the selection caveat (e.g. compare against turnovers by
  type, or label failures from tracking alone).
- P1 loader + batch: one loader module (stream `7z e -so`, game id from JSON),
  manifest of ~631 games, resumable Pool of ~3 with a per-game status row.
- P1 clean up the merged `Shots/` code (see Known issues), then the possession
  table and V(s) model.

**Open questions for the owner:** (1) retrospective grading of decisions or live
recommendation? (2) a date-based test split for the final evaluation? (3) keep 3.7 GB
of raw data in git or move to LFS/DVC/bucket (history rewrite acceptable)? (4)
does Dhruv still have the trained `.keras` models/scalers (none are tracked)? (5) push the
42 MB of results (and keep or delete the `cloud-full-batch` branch on GitHub)?
**Planned changes the owner wanted to implement (not yet recorded — add here).**

**Model runs (2026-10-19, `pass_model.py`, M1):** (1) first 7 games of
`Shots/allgames.txt`: extraction 23 s; script 8.4 s wall (TensorFlow import ~7 s
warm — **64 s on the very first import**; load + build + 5 epochs + predict ≈ 1.2 s),
340 MB peak; 3,470 passes → 2,449 after its `backcourt == 0` + `dropna` filters, 43
failures, AUC 0.47–0.48. (2) all 631 games: ~24 s wall including the TensorFlow
import (load 0.4 s, 5 epochs ≈ 7.5 s); 306,782 passes → 216,292 after
`backcourt == 0` → 209,946 after dropna, 4,484 failures (2.14%); random 60/40 split
(125,967 train / 83,979 test); F1 0.990, accuracy 0.979, ROC AUC 0.83 (0.829 and
0.833 over two unseeded runs). Accuracy equals always-guess-completed, so only the
AUC says anything, and it comes from a random-row split, not one grouped by game.
Use `eval_by_game.py` for honest numbers.

**Working notes for a fresh session:**
- Run from `Code/` (relative paths) with `../.venv/bin/python`. `get_data` extracts
  each game into its own temporary directory (no shared `./temp` any more), so a
  killed run can't leave a stale JSON and runs don't collide.
- **Resumable batch:** `python passtracking.py <list> [--retry-failed]
  [--allow-stale]`. Safe to stop any time (Ctrl-C, shutdown, power loss) and run
  the same command later: it skips games already done, writes each pickle
  atomically (temp file + rename), stops cleanly on SIGINT/SIGTERM after the
  current game (tested: SIGTERM and `kill -9` both resume correctly), and logs
  every game to `Tracking_Data/pass_features/_status.csv` (game, status
  ok/failed/unavailable, rows, seconds, code version, error). Results are stamped
  with a hash of `passtracking.py` + `getting_data.py`; pickles from older code are
  redone unless `--allow-stale`, so a resume can't silently reuse stale data.
  `pass_model.py <list>` also resumes training from the last finished epoch
  (Keras `BackupAndRestore`; written but not exercised — training takes ~1 s).
  Tip: `caffeinate -i <command>` stops the Mac sleeping mid-run.
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
- Durable checks: `validate_passes.py` (detection/labels/leakage on a few games) and
  `eval_by_game.py` (model signal over all games). The full-batch log is in
  `logs/full_batch_worktree.log` (git-ignored).
- **Cloud gotcha (2026-10-19):** `Agent(isolation: "remote")` did NOT start a cloud
  session here. It ran as an ordinary local sub-agent in a git worktree under
  `.claude/worktrees/` (a full 3.8 GB checkout inside the iCloud-synced folder,
  which also ate disk) with `caffeinate`; its own report said it was local. Before
  telling anyone something runs in the cloud, check `ListAgents` (cloud sessions
  are listed with type `cloud`) and `ps` / the process's working directory. The
  worktree was deleted after its results were verified identical and copied. A real
  cloud run needs a Claude Code web session started by the owner on the
  `cloud-full-batch` branch, or a scheduled cloud routine.

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
    (git-ignored; holds the 631 pickles + `_status.csv` from the 2026-10-19 full batch).
  - `setup.sh` — installs `p7zip` and extracts all `.7z` archives in place.
- **`Code/`** — passing pipeline.
  - `getting_data.py` — `get_data` class: extracts one game's `.7z` +
    matching play-by-play CSV, builds the `moments` DataFrame.
  - `passtracking.py` — `PassTracking` detects completed and failed passes
    from `moments` (see "Current status" for how); `features` extracts
    release-time pass-geometry features; `main()` runs the sample game, or
    batch-processes a game list given as argv[1].
  - `fetch_games.sh` — copies games out of git history into a local folder outside
    iCloud (see "iCloud placeholders" in Current status).
  - `validate_passes.py` — acceptance checks (assist recall, precision proxies,
    failure matching, leakage tests, attack direction) for any games.
  - `eval_by_game.py` — grouped-by-game evaluation of failed-pass prediction
    (GroupKFold(5), logistic + gradient boosting, bootstrap CIs) over
    `Tracking_Data/pass_features/`; ~8 min on all games.
  - `pass_model.py` — the Keras trainer: loads per-game feature pickles from
    `Tracking_Data/pass_features/`, trains an MLP to predict `pass_successful`,
    reports ROC AUC / F1 / accuracy. Run on all 631 games on 2026-10-19 (see status).
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
