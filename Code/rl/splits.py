"""Game splits for the RL project. Frozen on 2026-10-28; never move a game between sets.

  train    the 401 'tuning' dev games of the difficulty model (dev minus the 100 held-out ones): fit, cross-validate, choose
  holdout  the 100 dev games held out for the difficulty model; untouched by any RL model until the final evaluation of a stage
  test     splits/test_games.txt (130 games); reserved, do not read until the very end of the RL project
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, '..', 'splits')


def _read(name):
    return [l.strip().replace('.7z', '') for l in open(os.path.join(SPLITS, name)) if l.strip()]


def games(which):
    dev, hold = _read('dev_games.txt'), _read('difficulty_final_games.txt')
    if which == 'train':
        return [g for g in dev if g not in set(hold)]
    if which == 'holdout':
        return hold
    if which == 'test':
        raise PermissionError('the test split is reserved until the end of the RL project; read splits/test_games.txt deliberately if you mean it')
    raise ValueError(which)
