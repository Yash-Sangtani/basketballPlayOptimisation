# WS5a: Shots/ cleanup and coordinate check

Branch `ws5a-shots` (worktree `/Users/yash/dev/wt/ws5a-shots`). Code-only: nothing under `Code/`, no CLAUDE.md edit,
no model trained or created, main clone untouched. Python: the shared venv (3.12, pandas 3.1.0, matplotlib 3.11.2).

## 1. Coordinate verdict

**The committed feature CSV is wrong, but the code at HEAD was already geometrically correct. What remained was that shot
features were computed in the plot frame (y - 50). That is now fixed: features use raw court coordinates.**

Evidence (measured). Game 01.02.2016 DEN@GSW (id 0021500504, from the fetch_games.sh cache). Scratch script
`/private/tmp/ws5a/scratch/coord_check.py` reads the tracking JSON and pbp directly (it does not use `Shots/`).
Event 2, Iguodala alley-oop dunk, Q1 11:47:

| quantity | value |
|---|---|
| shooter, raw tracking coordinates | (5.28, 26.13) |
| shooter, plot frame (y - 50) | (5.28, -23.87) |
| basket | (5.35, 25) raw |
| distance, raw coordinates, raw basket | **1.13 ft** |
| distance, plot-frame y against basket y=+25 | **48.87 ft** |
| `dist` in committed `Shots/shot_features/01.02.2016-DEN-GSW.csv` | 48.86918... |
| `dist` from the real `get_features` after this change | **1.133 ft** (x, y = 5.28, 26.13) |

So the committed CSV is exactly the "plot-frame y against a basket at y=+25" mismatch. History (git): commit `5e8b9b4` (2024-05-05) had
`basket=[5.35, 25]` with plot-frame y; `82a34a6` uploaded the CSV it generated; `3b4302b` (2024-05-11) changed the basket and court
centre to y=-25. Distances and angles are translation-invariant, so HEAD's y=-25 constants give the same `dist` and angles as raw
coordinates (checked: max difference 3.6e-15 over the 43 made layups/dunks, computed from the stored x, y; I could not run the old code
itself because it does not import here). Over those 43 shots the mismatched formulation gives a median 52.4 ft and 0% under 5 ft.
The committed CSV is therefore a stale artifact of a bug that was fixed upstream, and every row in it carries the error. I did not
delete it (owner's call); I recommend `git rm`.

**What I changed anyway.** At HEAD the model-facing `y` feature (and `[x, y, dist]` into the zone model) was still in the plot frame
(y in [-50, 0]), with the constants repeated by hand in two files. Features now use raw coordinates in both places that build them:
`Game._get_moment_details(..., raw=True)` (default stays the plot frame, so plotting, `spacing.py` and `velocity.py` are untouched);
baskets, court centre and the 50 ft offset live only in `Shots/court.py`; `extract_shot_probability` calls `assert_raw_y`, which raises
if handed plot-frame details. Real run after the change: shooter y over all 191 shots in the game spans 0.01..52.43 (raw).

**Compatibility warning.** Distances and angles do not change; the `y` column changes by +50, and the zone model's `y` input too.
Any `shot_model_v2.keras` / `zone_model_v2.keras`, scalers, or `sfeatures/` and `zones/` pickles that Dhruv produced at HEAD
expect the plot frame and are incompatible with this code; retrain. None are tracked, so nothing in the repo breaks.

### Second finding (measured, not fixed): features are measured at the wrong moment

`get_features` takes the shooter's position at the frame of the play-by-play clock, not at release. With the real code on DEN@GSW:

- 43 made layups/dunks: `dist` at the code's frame has median **6.95 ft**, mean 9.7, max 39.1; only **42%** are within 5 ft and 67% within
  10 ft. The NBA shot log (`Tracking_Data/shots/shots.csv`, `SHOT_DISTANCE`) gives a median of 1.0 ft for the same shots.
- Independent scratch check (my approximation of the frame choice, 42 shots): within +-4 s the shooter gets within 5 ft of the rim in 90%
  of cases (median 1.7 ft), and that moment is a median 0.9 s *before* the code's frame (69% earlier; p10 -2.5 s, p90 +0.9 s).
- All 191 shots: tracking `dist` minus `SHOT_DISTANCE` has median +4.5 ft and mean absolute error 7.6 ft; only 34% are within 3 ft and
  54% within 6 ft; correlation 0.685.

Inferred cause: the pbp clock has 1 s resolution and is often off the tracking clock (CLAUDE.md saw 2-5 s for passes), and the code only
walks back up to 5 *frames* (0.2 s) to match the clock string. Fixing frame selection changes behaviour, so I left it; it matters for
any reuse (section 6).

## 2. What changed

New files:
- `Shots/config.py`: all paths. `TRACKING_DATA_DIR` (default `<repo>/Tracking_Data`, resolved from the file, not the cwd; same variable as
  `Code/getting_data.py`), `SHOTS_OUTPUT_DIR` (default `Shots/`, what the scripts did when run from `Shots/`), `SHOTS_MODEL_DIR` (default `Shots/`).
  Reference tables (`player/`, `shots/`) fall back to the repo copy when `TRACKING_DATA_DIR` is the games-only cache.
  `require_model_files` raises `FileNotFoundError` naming every missing model file; `require_local` detects iCloud placeholders.
- `Shots/court.py`: raw-coordinate constants, `basket_for_x`, `assert_raw_y`.
- `Shots/.gitignore`: generated `sfeatures/`, `zones/`, `data/`, `labels.txt`, `errorlog*.txt`, models/scalers/encoders.
- `Shots/tests/` (section 3).

Edited files (line numbers are in the new files):
- `simulate.py`: removed `matplotlib.use('TkAgg')`, the import-time `mkdir temp`, the import-time `warnings.filterwarnings("ignore")`, the
  top-level `import seaborn` and the top-level `from shot_tracking_plot import ...` (now inside `plot_frame` when `show_epv`, line 571;
  seaborn is optional at line 607). `_use_file_backend()` (line 30) switches to Agg only when `MPLBACKEND` is unset, called from `plot_frame`
  (line 577). `_temp_dir()` (line 24) creates `./temp` where it is used. `_get_tracking_data` (line 119): config path, per-call temp folder,
  7z without `shell=True` (with a list plus `shell=True` on macOS/Linux this ran bare `7z` and extracted nothing), clear errors for a missing
  archive, an iCloud placeholder or a failed extraction. `_get_playbyplay_data`: config path. `_get_moment_details(raw=False)` (line 465).
  `plot_frame` passes `raw=True` details to `extract_shot_probability` (lines 573-574). pandas 3 fixes (both were hard `TypeError`s, verified with a
  one-liner): `fillna(method='ffill')` -> `.ffill()` (line 173), `int(<1-row Series>)` -> `.iloc[0]` (line 833); also the `drop(inplace=True)` warning.
  The ffmpeg call in `animate_play` lost `shell=True` too (same bug; not run).
- `shot_tracking.py`: the three CSV reads moved into a cached `_tables()` (line 20); `extract_games` uses config; `get_features` uses `raw=True`
  at all six `_get_moment_details` calls, `basket_for_x` (line 210) and `COURT_CENTER` (line 250), creates its output folders (it used to raise on a
  missing `./sfeatures`, and `write_features` swallowed that for every game), and writes into the config dirs; `write_features` writes
  `labels.txt`/`errorlog.txt` into the output dir and its bare `except:` became `except Exception` that logs the error text;
  `write_features(extract_games())` is under `if __name__ == '__main__':` (line 410).
- `shot_tracking_plot.py`: TensorFlow imported lazily (`_tensorflow`, line 33) after `require_model_files` checks every model, scaler and
  encoder file; CSVs lazy (`_tables`, line 21); model/scaler/encoder paths from config; basket constants from `court.py`; `assert_raw_y` at lines 130-131.
- `shot_model.py`, `zone_model.py`, `list_games.py`: whole-module scripts wrapped in `main()` with `if __name__ == '__main__'`; TensorFlow imported inside
  `main()`; paths and saved artifacts via config. Verified with `git diff -w` that only paths, the wrapper and the TF import moved; `main()` was not run
  (no training allowed).
- `spacing.py`: the two `C:/Users/Dhruv/...` write paths replaced, and the matching read paths (`data/spacing`, `data/score`, `data/game`) now use the
  same config dirs (before, writes and reads went to different places unless you ran from `Shots/`); `get_spacing_statistics` creates them; seaborn lazy.
- `velocity.py`: seaborn lazy only. Its cwd-relative `data/velocity`, `temp/`, `examples/` and `ag.txt` paths are unchanged (not Windows paths).

No `C:/Users` string remains in `Shots/*.py` (a test checks this).

## 3. Tests (`Shots/tests/`: 32 tests; 31 run by default, 1 opt-in real-game test; all 32 pass)

Run: `cd /private/tmp/ws5a/run1 && /Users/yash/dev/basketballPlayOptimisation/.venv/bin/python -m pytest <worktree>/Shots/tests -p no:cacheprovider -q -rs`
(about 13 s; with `SHOTS_REAL_GAME_TEST=1` about 18 s and ~1 GB RAM, I ran that through `Code/heavy.sh`).

- `test_imports.py`: each of the 10 modules is imported in a fresh subprocess with cwd in an empty temp folder, `MPLBACKEND=Agg`, and data/output/model
  dirs pointing at nonexistent paths. Asserts: import succeeds (a missing third-party package would skip with a visible reason, none did), no new files in
  the cwd, TensorFlow and seaborn not imported, backend still `agg` (a forced TkAgg fails it), output/model folders not created. Also: no `C:/Users` or
  `matplotlib.use(` in any `Shots/*.py`; the extraction call sits behind `__main__`; missing models raise a `FileNotFoundError` naming all nine files
  before TensorFlow is imported.
- `test_config.py`: defaults resolve from the repo and not the cwd; env overrides; a relative env value is cwd-relative; reference-table fallback;
  `require_model_files` message.
- `test_coordinates.py`: the dunk is 1.13 ft in raw coordinates; the plot-frame/raw-basket mix reproduces 48.87; distances and angles are frame-invariant when
  the basket moves with the frame (50 random points); `_get_moment_details(raw=True)` y equals the default y + 50 (fake game using the real method); the real
  `get_features` run on a synthetic one-shot game gives dist 1.13 and raw y, plus a right-basket case and the output dirs; `extract_shot_probability` rejects
  plot-frame details and builds the same raw `y`/distance as training; opt-in real-game test (event 2 = 1.13 ft not 48.87; all shooter y in raw range).
- Mutation check (measured): removing `raw=True` from `shot_tracking.py` fails 2 coordinate tests; re-adding a forced TkAgg and an unguarded extraction call
  fails 6 import/source tests.

Before the change none of the Shots modules imported in this venv: `simulate`, `shot_tracking`, `spacing`, `velocity` failed with
`ModuleNotFoundError: seaborn` (not installed); `shot_tracking_plot` imported TensorFlow and then failed on the Windows CSV path; `shot_model` and
`zone_model` imported TensorFlow and failed on `./sfeatures` / `./zones`; `list_games` failed on its relative path. After: all 10 import cleanly,
without TensorFlow or seaborn.

## 4. What I ran and what I did not

Ran: the pytest suite; the real `shot_tracking.get_features('01.02.2016','GSW','DEN')` on the cache game (5.9 s, 191 shots, via `heavy.sh`, output in
`/private/tmp/ws5a/out`); `Game.plot_frame` on that game (wrote a correct-looking PNG, switched to Agg; with `show_epv=True` and no models it raised the
clear missing-model error with TensorFlow not imported).

Did not run: any training (`shot_model.main`, `zone_model.main`); the EPV happy path (no models); the `spacing.py` / `velocity.py` analyses and any plot
function that uses seaborn (not installed; those modules are import-tested only); `animate_play` / `watch_play` / ffmpeg; `list_games.main`.

## 5. What still blocks running the shot pipeline

1. **No trained models.** `shot_model_v2.keras`, `zone_model_v2.keras`, `shot_scaler_{mean,scale}.npy`, `zone_scaler_{mean,scale}.npy`,
   `encoder_shot_{zone,area,range}.pkl` are not tracked. EPV plotting and `extract_shot_probability` cannot run until `shot_model.py` / `zone_model.py`
   are run on freshly generated `sfeatures/` + `zones/` pickles (old plot-frame ones are incompatible).
2. Generating those pickles took 5.9 s for one game, so roughly an hour for ~631 games serially (inferred); only the archives in the cache exist locally.
3. seaborn is not installed in the venv (spacing/velocity plots).
4. `spacing.py` / `velocity.py` read `ag.txt` (untracked) and per-game folders; unchanged.

### Bugs found by reading, not fixed (not run, so inferred)
- `shot_tracking_plot.extract_shot_probability`: for a player absent from `player_shot_percentages.csv`, `fg_percentage` stays the float 34.6 and the next
  line calls `.empty` on it, which raises `AttributeError`.
- `shot_tracking.get_features` score-margin fallback: `if temp in game.pbp['EVENTNUM'].values` presumably should be `temp_event` (frame index vs event number),
  and `score_margin = features[-1][-3]` takes the previous row's `seconds`, not its margin.
- `extract_shot_probability` reloads the shot model, zone model and encoders once per player per frame.
- Docs say "within 4 feet" but `num_close_defenders` counts defenders within 10 ft; the in-pickle header has 21 names for 24 columns.
- `shot_model.py` evaluates with a random row split (shots from one game land in train and test), fits the scaler on all data and leaves `x`, `y`,
  `fg_percentage_zone` and the count columns unscaled.

## 6. What a possession-value model could reuse from Shots/ and what it should not

Reuse (judged from `shot_tracking.py`, `shot_tracking_plot.py`, `zone_model.py`, `config.py`/`court.py`):
- **The reference tables**, especially `Tracking_Data/shots/shots.csv`: an NBA shot log with `LOC_X`/`LOC_Y`, `SHOT_DISTANCE`, zone, made flag, game and
  event id, i.e. a labelled shot set that needs no tracking. A location-only xFG baseline can be built from it, with tracking features added on top.
  (The `LOC_X/LOC_Y` origin and units are the NBA stats convention, tenths of feet from the basket; inferred, not checked here.)
- **The feature ideas** in `get_features`: distance and angle to the attacked basket, closest-defender distance and angle to the shot line, defenders within
  10 ft, shooter and defender speed, shot clock, score margin. Now in raw coordinates, and distances/angles are translation-invariant anyway.
- **The expected-points concept** in `extract_shot_probability`: P(make) times points attempted for every offensive player at a frame. This is the
  "delta xPts passer->receiver" building block named in CLAUDE.md. Reuse the idea and `court.py`, not the implementation (per-player model reloads, the
  all-11-players-on-one-half direction heuristic that drops transition, the `AttributeError` above).
- **`spacing.py` convex-hull area and `velocity.py`** as state features for V(s); hull area does not depend on the coordinate frame.
- `court.py` and `config.py` as the one coordinate and path convention (raw 94x50, baskets at (5.35, 25) and (88.65, 25), same as `Code/`).

Do not reuse:
- **The committed `shot_features/` CSV and any model, scaler or pickle produced before this change** (wrong distances in the CSV; plot-frame `y` elsewhere).
- **The shot-frame selection.** Features sit at the pbp-clock frame: median 6.95 ft from the rim for made layups/dunks against 1.0 ft in the shot log (section 1).
  A value model needs the state at release; `Code/passtracking.py`'s hold/release logic is the better basis.
- **`zone_model.py`.** It learns shot zone from `(x, y, dist)` where `dist` comes from the mistimed frame but the zone label from the shot log at the true spot.
  Zone is a deterministic function of court position, so compute it geometrically; a PV model mostly needs xFG at a location, not a zone label.
- **`fg_percentage_zone` and `player_ratings`** as features: season aggregates that include the shot being predicted (leakage). Use out-of-fold player priors.
- **The training/evaluation recipe in `shot_model.py`** (ungrouped split, scaler fit on all data, 4x512 MLP). Use grouped-by-game CV and a simple model first,
  as the passing work does.
- **Shot outcomes as the target for V(s).** They give P(make | a shot was taken); a possession-value model needs possession outcomes including turnovers.
- **The `Game` class** in the unified pipeline: it duplicates `Code/getting_data.py`, uses a "~90% accurate" direction heuristic, and its default coordinates are
  the plot frame. Use the `Code/` loader and call into Shots only for the pieces above.

## 7. Reproduce

```
cd /private/tmp/ws5a/run1    # any scratch dir; do not run from an iCloud folder
PY=/Users/yash/dev/basketballPlayOptimisation/.venv/bin/python
$PY -m pytest /Users/yash/dev/wt/ws5a-shots/Shots/tests -p no:cacheprovider -q -rs
SHOTS_REAL_GAME_TEST=1 /Users/yash/dev/wt/ws5a-shots/Code/heavy.sh $PY -m pytest /Users/yash/dev/wt/ws5a-shots/Shots/tests -p no:cacheprovider -q
```
Scratch scripts (not committed): `/private/tmp/ws5a/scratch/{coord_check,run_get_features,analyze_out,smoke_plot}.py`.
