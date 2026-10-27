# Handoff (updated 2026-10-27, end of the cloud session)

Read `CLAUDE.md` first ("State on 2026-10-27"); this file only says what to do next and what is where.

## Where things are
Everything is on `main` (GitHub: Yash-Sangtani/basketballPlayOptimisation). `baseline-results` holds the OLD dataset (code `a846f643`); the regenerated one (`fc2ceb29`, heading receiver rule + live flags) is local only (git-ignored, 65 MB; regenerate in about 12 min, command in CLAUDE.md).

| report | content |
|---|---|
| `reports/ws2-eval.md` | full 501-game dev evaluation (old receiver rule), calibration, frontcourt and passer-only ablations |
| `reports/ws3-selection.md` | selection-caveat tests T1-T7: signal real but not pass difficulty |
| `reports/ws4-detection.md` | detector sweep (no change), the one regeneration, new dev numbers, the single test run |
| `reports/ws5b-value-review.md` | value-model MVP run on 120 dev games: works end to end, V(s) has almost no skill |
| `reports/difficulty-results.md` | the pass-difficulty model for RL (stage 1 completion probability, stage 2 shot value), usage, caveats |
| `reports/ws5a-shots.md` | Shots cleanup findings (no trained shot models exist) |

## Frozen split
The test split has been scored once (WS4). Treat it as used: do not tune or compare models on it again.

## Done in this session / not done
Done: steps 1-5 of the task (WS2 run, WS3 run, failing test fixed, detection review + one regeneration, test run once, CLAUDE.md updated). Not done: any recall improvement in the detector; a pass-quality label that is not a turnover; a useful V(s); several seeds; date-ordered folds.

## Open owner questions
Retrospective grading vs live recommendation (decides if the value model is worth building); whether to keep the failed-pass model given it ranks turnover risk rather than pass difficulty; Dhruv's trained shot models do not exist in the repo; the owner's "planned changes" were never recorded.
