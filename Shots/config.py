"""
Paths for the Shots pipeline.

Importing this module has no side effects (no file is read, no directory is created).
Everything is a function so the environment is read when a path is asked for, not when the
module is first imported.

Environment variables
---------------------
TRACKING_DATA_DIR   Where Tracking_Data lives (``game logs/<MM.DD.YYYY.AWAY.at.HOME>.7z`` and
                    ``events/<game_id>.csv``). Same variable as ``Code/getting_data.py``.
                    Default: ``<repo>/Tracking_Data``, resolved from this file, not from the
                    current directory. A relative value is taken relative to the current directory.
SHOTS_OUTPUT_DIR    Where generated files go (``sfeatures/``, ``zones/``, ``data/``, ``labels.txt``,
                    error logs). Default: this ``Shots/`` folder (what the scripts did when run
                    from ``Shots/``).
SHOTS_MODEL_DIR     Where the trained models, scalers and encoders are saved and loaded.
                    Default: this ``Shots/`` folder. None are tracked in git.
"""

import os
from pathlib import Path

SHOTS_DIR = Path(__file__).resolve().parent
REPO_DIR = SHOTS_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / 'Tracking_Data'

SF_DATALESS = 0x40000000  # macOS file flag: listed in the folder, contents not stored on this Mac


def _env_path(name, default):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return Path(default)


def tracking_data_dir():
    return _env_path('TRACKING_DATA_DIR', DEFAULT_DATA_DIR)


def game_logs_dir():
    return tracking_data_dir() / 'game logs'


def events_dir():
    return tracking_data_dir() / 'events'


def table_path(subfolder, filename):
    """
    Small reference tables that ship with the repo (``player/player_ratings.csv``,
    ``shots/shots.csv``, ...). Looked up under TRACKING_DATA_DIR first; if that folder was
    pointed at a copy that only holds game logs and events (e.g. the fetch_games.sh cache),
    fall back to the copy in the repo.
    """
    primary = tracking_data_dir() / subfolder / filename
    if primary.exists():
        return primary
    fallback = DEFAULT_DATA_DIR / subfolder / filename
    if fallback.exists():
        return fallback
    return primary  # neither exists: let the caller's error message name the preferred path


def ratings_path():
    return table_path('player', 'player_ratings.csv')


def player_shot_percentages_path():
    return table_path('player', 'player_shot_percentages.csv')


def zones_path():
    return table_path('shots', 'shots.csv')


def output_dir():
    return _env_path('SHOTS_OUTPUT_DIR', SHOTS_DIR)


def sfeatures_dir():
    return output_dir() / 'sfeatures'


def zones_out_dir():
    return output_dir() / 'zones'


def spacing_dir():
    return output_dir() / 'data' / 'spacing'


def score_dir():
    return output_dir() / 'data' / 'score'


def game_pickle_dir():
    return output_dir() / 'data' / 'game'


def model_dir():
    return _env_path('SHOTS_MODEL_DIR', SHOTS_DIR)


# The model, scaler and encoder files written by shot_model.py / zone_model.py and read by
# shot_tracking_plot.py.
SHOT_MODEL_FILES = ('shot_model_v2.keras', 'shot_scaler_mean.npy', 'shot_scaler_scale.npy')
ZONE_MODEL_FILES = ('zone_model_v2.keras', 'zone_scaler_mean.npy', 'zone_scaler_scale.npy',
                    'encoder_shot_zone.pkl', 'encoder_shot_area.pkl', 'encoder_shot_range.pkl')


def model_path(filename):
    return model_dir() / filename


def require_model_files(filenames, trained_by):
    """Raise FileNotFoundError naming every missing model artifact (none are tracked in git)."""
    missing = [str(model_path(f)) for f in filenames if not model_path(f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained-model files (they are not tracked in git; train them with "
            f"{trained_by} or point SHOTS_MODEL_DIR at a folder that has them):\n  "
            + "\n  ".join(missing))


def game_list_path():
    """The list of games, one ``MM.DD.YYYY.AWAY.at.HOME`` per line."""
    return SHOTS_DIR / 'allgames.txt'


def require_local(path):
    """
    iCloud placeholders ("dataless" files) are listed in the folder but read as 0 bytes
    instead of raising. Fail loudly instead of silently reading nothing.
    """
    st = os.stat(path)
    if getattr(st, 'st_flags', 0) & SF_DATALESS:
        raise OSError(
            f"{path} is an iCloud placeholder (not downloaded to this Mac). Download it in "
            "Finder, or copy the games out of git with Code/fetch_games.sh and set "
            "TRACKING_DATA_DIR.")
