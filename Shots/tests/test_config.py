"""config.py resolves paths from the repo layout (not the current directory) and from env vars."""
import os
from pathlib import Path

import pytest

import config
from conftest import REPO_DIR, SHOTS_DIR


@pytest.fixture
def clean_env(monkeypatch):
    for name in ('TRACKING_DATA_DIR', 'SHOTS_OUTPUT_DIR', 'SHOTS_MODEL_DIR'):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_relative_to_the_repo_not_the_cwd(clean_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.SHOTS_DIR == SHOTS_DIR
    assert config.REPO_DIR == REPO_DIR
    assert config.tracking_data_dir() == REPO_DIR / 'Tracking_Data'
    assert config.game_logs_dir() == REPO_DIR / 'Tracking_Data' / 'game logs'
    assert config.events_dir() == REPO_DIR / 'Tracking_Data' / 'events'
    assert config.output_dir() == SHOTS_DIR
    assert config.sfeatures_dir() == SHOTS_DIR / 'sfeatures'
    assert config.model_dir() == SHOTS_DIR
    assert config.game_list_path() == SHOTS_DIR / 'allgames.txt'
    for path in (config.ratings_path(), config.zones_path(), config.player_shot_percentages_path()):
        assert path.is_absolute()


def test_reference_tables_exist_in_the_repo(clean_env):
    for path in (config.ratings_path(), config.zones_path(), config.player_shot_percentages_path()):
        assert path.exists(), path


def test_tracking_data_dir_env_var(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv('TRACKING_DATA_DIR', str(tmp_path))
    assert config.tracking_data_dir() == tmp_path.resolve()
    assert config.game_logs_dir() == tmp_path.resolve() / 'game logs'
    assert config.events_dir() == tmp_path.resolve() / 'events'


def test_relative_env_value_is_relative_to_cwd(clean_env, tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRACKING_DATA_DIR', 'data')
    assert config.tracking_data_dir() == (tmp_path / 'data').resolve()


def test_reference_tables_fall_back_to_the_repo_copy(clean_env, tmp_path, monkeypatch):
    """TRACKING_DATA_DIR may point at a cache holding only game logs and events."""
    monkeypatch.setenv('TRACKING_DATA_DIR', str(tmp_path))
    assert config.ratings_path() == REPO_DIR / 'Tracking_Data' / 'player' / 'player_ratings.csv'
    assert config.zones_path() == REPO_DIR / 'Tracking_Data' / 'shots' / 'shots.csv'
    # a table present in the chosen folder wins
    (tmp_path / 'player').mkdir()
    (tmp_path / 'player' / 'player_ratings.csv').write_text('PLAYER_NAME\n')
    assert config.ratings_path() == tmp_path.resolve() / 'player' / 'player_ratings.csv'


def test_output_and_model_dirs_are_overridable(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv('SHOTS_OUTPUT_DIR', str(tmp_path / 'o'))
    monkeypatch.setenv('SHOTS_MODEL_DIR', str(tmp_path / 'm'))
    assert config.sfeatures_dir() == (tmp_path / 'o').resolve() / 'sfeatures'
    assert config.spacing_dir() == (tmp_path / 'o').resolve() / 'data' / 'spacing'
    assert config.model_path('shot_model_v2.keras') == (tmp_path / 'm').resolve() / 'shot_model_v2.keras'


def test_require_model_files_names_every_missing_file(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv('SHOTS_MODEL_DIR', str(tmp_path))
    (tmp_path / 'shot_scaler_mean.npy').write_bytes(b'')
    with pytest.raises(FileNotFoundError) as info:
        config.require_model_files(config.SHOT_MODEL_FILES, 'Shots/shot_model.py')
    message = str(info.value)
    assert 'shot_model_v2.keras' in message and 'shot_scaler_scale.npy' in message
    assert 'shot_scaler_mean.npy' not in message  # present, so not listed
    assert 'Shots/shot_model.py' in message and 'SHOTS_MODEL_DIR' in message
