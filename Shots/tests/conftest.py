"""Shared pytest setup for the Shots tests. Run from anywhere:

    /path/to/.venv/bin/python -m pytest Shots/tests -p no:cacheprovider

(``-p no:cacheprovider`` keeps pytest from writing a .pytest_cache folder into the repo.)
"""
import os
import sys
from pathlib import Path

import pytest

SHOTS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SHOTS_DIR.parent

# The Shots modules import each other by bare name (``from simulate import Game``).
if str(SHOTS_DIR) not in sys.path:
    sys.path.insert(0, str(SHOTS_DIR))

MODULES = ['config', 'court', 'simulate', 'shot_tracking', 'shot_tracking_plot', 'shot_model',
           'zone_model', 'spacing', 'velocity', 'list_games']


@pytest.fixture
def shots_env(tmp_path):
    """Environment for a subprocess that imports Shots modules from a scratch folder, with every
    data/output/model location pointing at folders that do not exist (so any import-time read
    fails loudly and any import-time write is visible)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(SHOTS_DIR)
    env['TRACKING_DATA_DIR'] = str(tmp_path / 'no_such_tracking_data')
    env['SHOTS_OUTPUT_DIR'] = str(tmp_path / 'out')
    env['SHOTS_MODEL_DIR'] = str(tmp_path / 'models')
    env['MPLBACKEND'] = 'Agg'
    return env
