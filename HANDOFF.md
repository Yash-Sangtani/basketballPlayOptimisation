# Handoff (2026-10-27): continue the passing-pipeline work in a Claude Code cloud session

Written when the local Mac session was stopped (API session limit + a 8 GB / nearly-full-disk machine).
Read `CLAUDE.md` first for the project and the pass-detection design; this file is the *state of the work*.
Everything below is measured unless marked (inferred) or (unreviewed).

## Where things are (GitHub: Yash-Sangtani/basketballPlayOptimisation)
**Everything is merged into `main` (pushed 2026-10-27); `main` is the branch to work from.** `cloud-full-batch` is kept in sync with it.
| branch | what it is |
|---|---|
| `main` | all work: detection rewrite, resumable batch, metadata, frozen dev/test split, chunked driver, Shots cleanup (ws5a), new evaluation (ws2), WIP selection scripts (ws3), possession table + WIP value scripts (ws5b), this file. |
| `baseline-results` | `cloud-full-batch` as of the handoff + the 65 MB baseline dataset force-added (`Tracking_Data/pass_features/`: 631 pickles, 631 meta files, `_status.csv`). Restore with `git checkout origin/baseline-results -- Tracking_Data/pass_features`. Saves ~40 min. |
| `ws5a-shots`, `ws2-eval`, `ws3-selection`, `ws5b-value` | the workstream branches (already merged into `main`; kept for history). |
| `dhruv`, `yash` | untouched. |

**Known failing test after the merge:** `Code/value/tests/test_states.py::test_feature_values` (WS5b's unreviewed WIP value-model code). All other tests pass (41 Code tests, 31 Shots tests + 1 opt-in skipped). The merged `passtracking.py`/`getting_data.py` hash is still `a846f643`, so the baseline dataset is still valid.

## Verified baseline dataset (code version `a846f643`)
631 games ok + the 5 known-bad failed (stubs `01.14.2016.CHI.at.PHI`, `CLE.at.SAS`, `DET.at.MEM`, `01.23.2016.UTA.at.WAS`, and `01.23.2016.CHI.at.CLE`), 306,782 passes, 5,025 failures (1.64%), one meta file per pickle. The acceptance gate (counts, 5 known failures, single code version, aligned meta, totals identical to the first full run) passed. Earlier result on this data (`Code/eval_by_game.py`, GroupKFold(5) by game): ROC ~0.85, PR-AUC 0.10-0.12 vs base rate 0.016. **Caveat still untested:** failures are labelled by pbp-turnover matching, so part of the signal may be selection (WS3's job).

## Frozen split (do not break)
`Code/splits/dev_games.txt` (501) / `test_games.txt` (130, 01.04.2016-01.23.2016). Tune, select features, compare models on dev ONLY; run test once at the very end. `01.12.2016.HOU.at.MEM` is in test but was looked at early (slightly contaminated).

## Status per workstream
- **WS5a Shots cleanup: done.** Import side effects removed, paths via `Shots/config.py` (`TRACKING_DATA_DIR`), shot features now use raw court coordinates. Found: committed `Shots/shot_features/01.02.2016-DEN-GSW.csv` is wrong (dunk 48.87 ft vs real 1.13 ft; mixed plot-frame y). Shot features read the shooter at the pbp clock frame, not release (layups median 6.95 ft from rim vs 1.0 ft in the shot log): do not reuse them for a value model. No trained shot/zone models exist (9 artifacts missing). Full report: `reports/ws5a-shots.md` on that branch.
- **WS2 evaluation: code done, nothing run at full size.** `python pass_model.py --split dev --models logistic,gboost,mlp` (see its `--help`); `--split test` needs an explicit confirmation flag. Tests in `Code/tests/`. Partial results from a 265-game snapshot were lost (scratch wiped). TODO: run on all 501 dev games, write `reports/ws2-eval.md`.
- **WS3 selection tests: not run.** Planned tests: (T1) `target_mode='heading'` vs `'actual'` on ~100 dev games, (T2) tracking-only failure labels, (T3) shuffle labels within game (ROC must be ~0.5), (T4) shot_clock+dist baseline, (T5) failure-type breakdown, (T6) matched negatives (same team/game, +-8 s), (T7) permutation importance.
- **WS5b value model: possession table done (unreviewed by the lead).** `Code/value/possessions.py` + tests: on all 501 dev games the possession count / (FGA+0.44*FTA-OREB+TOV) ratio has median 1.013, min 0.988; possession points equal the final score in all 501 games only after counting technical/other points separately (216/501 strictly). V(s) and pass-value scripts are WIP/unvalidated.
- **WS4 detection improvements: not started** (depends on WS3 findings). Known issues: assist recall 67-81%, a few shot-clock windows still flip teams, 2 of ~15 pbp bad passes per game unmatched. Any change to `passtracking.py`/`getting_data.py` changes the code hash and needs a full regeneration (budget exactly one more).

## Next steps (in this order)
1. Set up (see below), restore the baseline, run the acceptance gate logic yourself (counts above).
2. Run WS2's full dev evaluation; commit `reports/ws2-eval.md`.
3. Finish and run WS3's tests; decide from them whether the signal is real; commit `reports/ws3-selection.md`.
4. WS4: improve detection on dev games only, regenerate once, re-evaluate with WS2's tooling, then run the test split exactly once.
5. Review/finish WS5b (V(s) grouped by game vs shot-clock+ball-location baseline; pass delta-V vs assists), keeping it an MVP.
6. Work directly on `main` (the owner approved merging everything into it); keep `CLAUDE.md` updated at the end as the single writer.

## Cloud setup (Linux; the Mac's iCloud/low-disk problems do not apply)
```
git clone https://github.com/Yash-Sangtani/basketballPlayOptimisation.git && cd basketballPlayOptimisation
git checkout main
git checkout origin/baseline-results -- Tracking_Data/pass_features   # restores the verified baseline
sudo apt-get install -y p7zip-full    # needs the `7z` binary
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn scipy matplotlib tqdm tensorflow pytest
cd Code && ../.venv/bin/python passtracking.py ../Shots/allgames.txt --print-todo | wc -l   # expect 0 if the baseline restored correctly
```
Full raw data is in the normal checkout (`Tracking_Data/game logs/*.7z`), so you can regenerate with `../.venv/bin/python passtracking.py ../Shots/allgames.txt` (resumable, ~35-40 min serial, ~0.9 GB RAM per process). Use `Code/heavy.sh <cmd>` if several heavy jobs could overlap. `Code/run_chunked.sh` is only for the low-disk Mac.

## Gotchas learned the hard way
- Results are stamped with a hash of `passtracking.py` + `getting_data.py`; editing either makes every pickle stale (the batch redoes them). Do not edit them mid-run.
- Never tune on `test_games.txt`.
- On a partial clone, `git log --follow`/`-p`/`-S`, `git grep <tree-ish>` download every historical blob (6 MB archives); use `GIT_NO_LAZY_FETCH=1`. A normal full clone does not have this problem.
- `Agent(isolation:"remote")` did not give a cloud session in the earlier local session; verify where a process actually runs.
- The AUC predicts what the detector labels as failed; do not call it "pass difficulty" until WS3 says so.
- Open owner questions: retrospective grading vs live recommendation (decides whether a value model is worth building); whether to merge into `main`; Dhruv's trained shot models do not exist in the repo; the owner's "planned changes" were never recorded.
