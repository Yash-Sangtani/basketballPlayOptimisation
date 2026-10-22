# WS5a: Shots/ cleanup and coordinate check

Status: IN PROGRESS (skeleton committed early; sections below are filled in as work lands).

## Coordinate verdict (measured, independent of the Shots/ code)

Scratch script (not committed): `/private/tmp/ws5a/scratch/coord_check.py`, game 01.02.2016 DEN@GSW (game id 0021500504),
42 made layups/dunks from the play-by-play.

- Event 2 (Iguodala alley-oop dunk, Q1 11:47): raw tracking coordinates (5.28, 26.13); the basket is at (5.35, 25).
  Distance with raw coordinates = 1.13 ft. Plot-frame y (y-50 = -23.87) against a basket at y=+25 = 48.87 ft, which is exactly
  the number in the committed `Shots/shot_features/01.02.2016-DEN-GSW.csv` (48.86918...).
- That mismatch existed in commit 5e8b9b4 (2024-05-05, the commit that produced the CSV). Commit 3b4302b (2024-05-11) moved the
  basket and court centre to y=-25, consistent with the shifted y. HEAD's distance/angle maths is therefore correct; the committed
  CSV is a stale artifact of the buggy version.

(more detail below once the work is complete)
