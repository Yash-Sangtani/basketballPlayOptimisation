# RL Stage 3: set-play discovery (first pass)

Code: `Code/rl/plays.py` (descriptor), `Code/rl/cluster_plays.py` (clustering and tests). Output: `reports/rl-stage3/` (`cluster_tests.json`, `cluster_summary.csv`, `labels.npz`, `play_library.npz` for K=24). Fit on the 320 fitting games (45,566 half-court possessions of at least 6 s), tested on the 81 validation games (11,330). Holdout and test games were not read.

**Descriptor.** First 6 s of each possession, 1-s steps: a 6x5 court-grid count of the five attackers (player identities ignored), ball position, mean offensive speed; PCA to 24 components (32% of the variance), k-means.

## Tests (measured)
| K | stability ARI (two game halves) | silhouette | team-mix stat vs null (95th pct) | outcome eta^2, validation (null 95th pct) |
|---|---|---|---|---|
| 6 | 0.63 | 0.059 | 0.066 (0.020) | 0.0016 (0.0009) |
| 10 | 0.58 | 0.064 | 0.086 (0.027) | 0.0015 (0.0015) |
| 16 | 0.58 | 0.062 | 0.134 (0.045) | 0.0021 (0.0022) |
| 24 | 0.52 | 0.062 | 0.186 (0.063) | 0.0042 (0.0030) |

Selection rule (fixed beforehand): highest validation eta^2 among K with ARI >= 0.4, which gives K=24.

## What this says
- **Teams run different mixes of these clusters**: the team-by-cluster statistic is about 3x the null at every K (null shuffles possessions between the two sides within each game). Caveat: this null does not remove roster, pace or position effects, so it shows teams differ in shape of play, not that they call named plays.
- **The clusters are reproducible but not separated**: ARI 0.5-0.6 between independent halves, but silhouette is about 0.06, so possessions form a continuum and the clusters cut it into regions rather than finding distinct plays.
- **Clusters barely predict outcome**: at K=24 the cluster explains 0.4% of the variance in field-goal points per possession on validation games, only just above the shuffled null (0.30%); for K up to 16 it is at the null. Per-cluster mean points per possession span roughly 0.68-1.0 (`cluster_summary.csv`), but most of that is sampling noise at these sizes, and points exclude free throws and turnovers.

## Verdict
The descriptor finds a stable vocabulary of formations and team tendencies, not evidence of plays that change results. It is a usable starting library for conditioning, but the claim "set plays discovered" is not supported yet.

## Not done
Conditioning the executor on cluster id (does it lower 1-3 s movement error?), prototype animations, a finer descriptor (player-role-aware trajectories, screens and cuts), and outcome measures including turnovers and free throws. These are the next Stage 3 steps.

## Executor conditioning test (added)
Question: does knowing the play (the 24-way cluster of the possession's first 6 s) lower the executor's error afterwards? Code `Code/rl/train_play_cond.py`, raw numbers `reports/rl-stage3/cond/` (`compare.txt`, `paired.txt`).
Design: executor v2, real profiles, the Stage 2b fit/validation games. To avoid leakage (the play id is computed from t_front 0-6 s) only samples with t_front >= 6 s are used, for every condition: 270,890 fit and 66,198 validation samples. Conditions: no play id (2 seeds), one-hot play id appended to the context (2 seeds), play ids permuted across possessions (1 seed). Differences are paired and game-clustered (play minus none, negative = play better):

| metric | play - none (95% interval) | seed spread (sd) |
|---|---|---|
| offence 1-s error (ft) | -0.0009 [-0.0014, -0.0002] | 0.005 |
| defence 1-s error (ft) | -0.0002 [-0.0007, +0.0003] | 0.003 |
| offence NLL | -0.0002 [-0.0004, +0.0002] | 0.000 |
| decision loss | +0.0017 [+0.0007, +0.0027] (worse) | 0.001 |
| pass-target loss | +0.0009 [-0.0012, +0.0031] | 0.000 |

**Result: no useful gain.** The one nominally significant movement difference is 0.0009 ft (0.04% of a 2.02 ft error), smaller than the seed-to-seed spread of 0.005 ft, and decisions get slightly worse. The shuffled-id control is as good as the real one (offence error 2.022 vs 2.022 base). After 6 s the executor already sees the last 0.4 s of motion, which carries what a formation label would add.
Limits: this tests only the later part of the possession (the early part is where a play id would help most, but there the descriptor would leak the target); one-hot ids of this weak clustering (outcome eta^2 0.4%) are a weak test of whether better-defined plays would help; 2 seeds. Reading: at this stage play ids are not worth feeding to the executor; they may still serve as the play-caller's discrete action vocabulary, which is a different use and is untested.

## Prototype GIFs (added)
`docs/plays/play_00.gif` ... `play_23.gif` (index with per-play stats: `docs/plays/README.md`), made by `Code/rl/play_gifs.py`: one real validation possession per cluster, nearest the centre. Watching them is the quickest sanity check of what a cluster means; they show formation families (e.g. corner/wing spacing, early-clock drives), not coached plays. Stage 3 is closed at this level: a library exists and is documented, and the honest verdict above stands (reproducible, team-specific, not outcome-predictive, no executor gain). A role-aware trajectory descriptor remains the one untried improvement and is deferred unless the play-caller needs a richer vocabulary.
