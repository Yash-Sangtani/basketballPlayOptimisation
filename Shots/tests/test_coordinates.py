"""Shot features must be computed in RAW court coordinates (x in [0, 94], y in [0, 50], baskets
at (5.35, 25) and (88.65, 25)), not in the plot frame (y - 50) that Game._get_moment_details
returns by default.

Background: commit 5e8b9b4 computed features from plot-frame y against a basket at y=+25, so
every shot was ~50 ft from the rim. The committed shot_features/01.02.2016-DEN-GSW.csv still has
those numbers: event 2 (an alley-oop dunk at raw (5.28, 26.13)) has dist 48.87 there, the true
distance is 1.13 ft.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import court
import shot_tracking
import shot_tracking_plot
from simulate import Game

# ---------------------------------------------------------------- known example from DEN@GSW
DUNK_RAW_XY = (5.28373, 26.13086)          # Iguodala, event 2, raw SportVU coordinates
COMMITTED_CSV_DIST = 48.86918493337596    # what the committed CSV says for that shot


def test_basket_for_x():
    assert court.basket_for_x(10.0) == [5.35, 25.0]
    assert court.basket_for_x(80.0) == [88.65, 25.0]
    assert court.basket_for_x(46.99) == [5.35, 25.0] and court.basket_for_x(47.0) == [88.65, 25.0]


def test_known_dunk_is_about_one_foot_from_the_rim_in_raw_coordinates():
    x, y = DUNK_RAW_XY
    dist = np.linalg.norm(np.array([x, y]) - np.array(court.basket_for_x(x)))
    assert dist == pytest.approx(1.13, abs=0.01)
    assert dist < 5


def test_mixing_plot_frame_y_with_raw_basket_reproduces_the_committed_csv_bug():
    """Documents the bug: plot-frame y (y - 50) against the raw basket y=25."""
    x, y = DUNK_RAW_XY
    wrong = np.linalg.norm(np.array([x, y - court.PLOT_Y_OFFSET]) - np.array(court.basket_for_x(x)))
    assert wrong == pytest.approx(COMMITTED_CSV_DIST, abs=1e-6)


def test_distances_and_angles_do_not_depend_on_the_frame_if_the_basket_moves_with_it():
    """Why HEAD's y=-25 constants were also correct, and why only the 'y' column differs."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = rng.uniform(0, 94), rng.uniform(0, 50)
        defender = np.array([x, y]) + rng.normal(0, 3, 2)
        shift = np.array([0.0, -court.PLOT_Y_OFFSET])

        def geometry(shooter, defender, basket, centre):
            to_basket = basket - shooter
            to_centre = basket - centre
            angle = np.arccos(np.dot(to_basket, to_centre) / (np.linalg.norm(to_basket) * np.linalg.norm(to_centre)))
            return np.linalg.norm(to_basket), angle, np.linalg.norm(defender - shooter)

        basket = np.array(court.basket_for_x(x))
        centre = np.array(court.COURT_CENTER)
        raw = geometry(np.array([x, y]), defender, basket, centre)
        plot = geometry(np.array([x, y]) + shift, defender + shift, basket + shift, centre + shift)
        assert raw == pytest.approx(plot)


def test_assert_raw_y():
    court.assert_raw_y([5.0, 25.0, 40.0], 'x')
    with pytest.raises(ValueError, match='plot frame'):
        court.assert_raw_y([-5.0, -25.0, -40.0], 'x')


# ------------------------------------------------------------- Game._get_moment_details(raw=...)
HOME, AWAY = 1610612744, 1610612743
HOME_IDS = list(range(101, 114))      # 13 home players, then 13 away (shot_tracking splits on 13)
AWAY_IDS = list(range(201, 214))


def make_positions(shooter_xy=(6.0, 26.0)):
    """Ball first, then 5 home and 5 away players. All x values are distinct (the feature code
    finds players by their x value)."""
    positions = [[-1, -1, shooter_xy[0] + 0.1, shooter_xy[1], 7.0],
                 [HOME, 101, shooter_xy[0], shooter_xy[1], 0.0]]
    home_rest = [(20.0, 10.0), (25.0, 40.0), (30.0, 25.0), (35.0, 30.0)]
    for pid, (x, y) in zip(HOME_IDS[1:5], home_rest):
        positions.append([HOME, pid, x, y, 0.0])
    away = [(8.0, 27.5), (14.0, 22.0), (22.0, 33.0), (28.0, 18.0), (33.0, 8.0)]
    for pid, (x, y) in zip(AWAY_IDS[:5], away):
        positions.append([AWAY, pid, x, y, 0.0])
    return positions


class FakeGame:
    """Just enough of Game for shot_tracking.get_features, with the REAL _get_moment_details."""
    _get_moment_details = Game._get_moment_details

    def __init__(self, shooter_xy=(6.0, 26.0)):
        times = [709.0, 708.0, 707.0, 706.0]            # quarter_time (s left): 11:49 .. 11:46
        self.moments = pd.DataFrame({
            'quarter': [1] * 4,
            'universe_time': [1_000_000 + 40 * i for i in range(4)],
            'quarter_time': times,
            'shot_clock': [20.0] * 4,
            'positions': [make_positions(shooter_xy) for _ in times],
            'game_time': [720 - t for t in times],
        })
        self.team_colors = {-1: 'orange', AWAY: 'blue', HOME: 'red'}
        self.player_ids = {**{f'Home {i}': i for i in HOME_IDS}, **{f'Away {i}': i for i in AWAY_IDS}}
        self.game_id = '0021500504'
        self.pbp = pd.DataFrame({
            'EVENTNUM': [2], 'EVENTMSGTYPE': [1], 'PCTIMESTRING': ['11:47'],
            'PLAYER1_ID': [101], 'PLAYER1_NAME': ['Home 101'], 'SCOREMARGIN': ['TIE'],
        })

    def get_play_frames(self, event_num, play_type='offense'):
        return 0, 2                                      # shot at frame 2 (quarter_time 707 -> 11:47)


def test_moment_details_raw_is_the_plot_frame_plus_50():
    game = FakeGame()
    plot = game._get_moment_details(2)
    raw = game._get_moment_details(2, raw=True)
    assert np.allclose(np.asarray(raw[2]) - np.asarray(plot[2]), court.PLOT_Y_OFFSET)
    assert np.all(np.asarray(plot[2]) < 0)                      # default: plot frame, y in [-50, 0]
    assert np.all((np.asarray(raw[2]) >= 0) & (np.asarray(raw[2]) <= 50))
    assert raw[1] == plot[1]                                    # x unchanged
    assert raw[10] == plot[10]                                  # ids unchanged
    assert game._get_moment_details(2, raw=False)[2].tolist() == plot[2].tolist()


@pytest.fixture
def feature_run(tmp_path, monkeypatch):
    """Run the real shot_tracking.get_features on a synthetic one-shot game."""
    monkeypatch.setenv('SHOTS_OUTPUT_DIR', str(tmp_path / 'out'))
    ratings = pd.DataFrame({'PLAYER_NAME': ['Home 101', 'Away 201'],
                            'OFFENSIVE_RATING': [110.0, 100.0], 'DEFENSIVE_RATING': [105.0, 108.0]})
    zones = pd.DataFrame(columns=['PLAYER_NAME', 'GAME_ID', 'GAME_EVENT_ID', 'PERIOD', 'MINUTES_REMAINING',
                                  'SECONDS_REMAINING', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE'])
    fg = pd.DataFrame(columns=['PLAYER_NAME', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE',
                               'SHOT_PERCENTAGE'])
    monkeypatch.setattr(shot_tracking, '_TABLES', (ratings, zones, fg))

    def run(shooter_xy):
        game = FakeGame(shooter_xy)
        monkeypatch.setattr(shot_tracking, 'Game', lambda *args: game)
        features = shot_tracking.get_features('01.02.2016', 'GSW', 'DEN')
        header, row = features[0], features[1]
        names = ['event', 'shooter', 'shooter_name', 'shooter_offensive_rating', 'dist', 'x', 'y', 'shot_angle',
                 'shooter_velocity', 'fg_percentage_zone', 'closest_defender', 'closest_defender_name',
                 'closest_defender_defensive_rating', 'closest_defender_dist', 'closest_defender_angle',
                 'closest_defender_velocity', 'num_close_defenders', 'shot_clock', 'score_margin', 'quarter',
                 'minutes', 'seconds', 'points_attempted', 'make']
        assert len(row) == len(names)
        return dict(zip(names, row))

    return run


def test_get_features_uses_raw_coordinates(feature_run):
    row = feature_run((5.28373, 26.13086))
    assert row['x'] == pytest.approx(5.28373)
    assert row['y'] == pytest.approx(26.13086)                       # raw y, not -23.87
    assert row['dist'] == pytest.approx(1.13, abs=0.01)              # not 48.87
    assert row['dist'] < 5
    assert np.isfinite(row['shot_angle'])
    # closest defender (8.0, 27.5): 2.7 ft x, 1.4 ft y from the shooter
    assert row['closest_defender'] == 201
    assert row['closest_defender_dist'] == pytest.approx(np.hypot(8.0 - 5.28373, 27.5 - 26.13086))
    assert row['make'] == 1


def test_get_features_right_basket(feature_run):
    # shooter near the right basket (88.65, 25): the basket is chosen by x, still raw
    row = feature_run((87.0, 24.0))
    assert row['dist'] == pytest.approx(np.hypot(88.65 - 87.0, 25.0 - 24.0))
    assert row['y'] == pytest.approx(24.0)


def test_get_features_writes_into_the_configured_output_dir(feature_run, tmp_path):
    feature_run((6.0, 26.0))
    assert (tmp_path / 'out' / 'sfeatures' / '01.02.2016-DEN-GSW.pkl').exists()
    assert (tmp_path / 'out' / 'zones' / '01.02.2016-DEN-GSW.pkl').exists()
    with open(tmp_path / 'out' / 'sfeatures' / '01.02.2016-DEN-GSW.pkl', 'rb') as f:
        assert len(pickle.load(f)) == 2                              # header + one shot


# ------------------------------------------------- inference side (shot_tracking_plot)
def test_extract_shot_probability_rejects_plot_frame_details():
    game = FakeGame()
    plot = game._get_moment_details(2)            # plot frame: what plot_frame uses for drawing
    previous = game._get_moment_details(1)
    with pytest.raises(ValueError, match='plot frame'):
        shot_tracking_plot.extract_shot_probability(game, plot, previous)


def test_inference_features_use_the_same_raw_frame_as_training(monkeypatch):
    """extract_shot_probability (used for EPV plotting) must build the same 'y' and distance as
    get_features, i.e. raw coordinates, since the models are trained on get_features output."""
    game = FakeGame((6.0, 26.0))
    game.flip_direction = False
    game.pbp['game_time'] = [13]
    raw, raw_previous = game._get_moment_details(2, raw=True), game._get_moment_details(1, raw=True)

    names = [f'Home {i}' for i in HOME_IDS[:5]]
    ratings = pd.DataFrame({'PLAYER_NAME': names, 'OFFENSIVE_RATING': 100.0, 'DEFENSIVE_RATING': 100.0})
    fg = pd.DataFrame({'PLAYER_NAME': names, 'SHOT_ZONE_BASIC': 'Restricted Area', 'SHOT_ZONE_AREA': 'Center(C)',
                       'SHOT_ZONE_RANGE': 'Less Than 8 ft.', 'SHOT_PERCENTAGE': 60.0})
    monkeypatch.setattr(shot_tracking_plot, '_TABLES', (ratings, fg))
    monkeypatch.setattr(shot_tracking_plot, 'predict_output',
                        lambda vector: ['Restricted Area', 'Center(C)', 'Less Than 8 ft.'])
    captured = []
    monkeypatch.setattr(shot_tracking_plot, 'load_model_and_predict',
                        lambda vector: captured.append(list(vector)) or 0.5)

    result = shot_tracking_plot.extract_shot_probability(game, raw, raw_previous)

    assert set(result) == set(HOME_IDS[:5]) and len(captured) == 5
    shooter_vector = captured[0]                     # home 101 is first on the court
    # feature order: rating, dist, x, y, angle, velocity, fg%, def rating, def dist, ...
    assert shooter_vector[2] == pytest.approx(6.0)
    assert shooter_vector[3] == pytest.approx(26.0)                   # raw y (the plot frame would give -24)
    basket = court.basket_for_x(80.0)                                 # this fake half-court attacks the right basket
    assert basket == [88.65, 25.0]
    assert shooter_vector[1] == pytest.approx(np.hypot(6.0 - basket[0], 26.0 - basket[1]))
    # and the zone model sees raw coordinates too
    assert all(v[3] >= 0 for v in captured)


# ------------------------------------------------- real game (opt-in)
REAL_ARCHIVE = '01.02.2016.DEN.at.GSW.7z'


def _real_data_dir():
    candidates = [os.environ.get('TRACKING_DATA_DIR'),
                  str(Path.home() / '.cache' / 'basketballPlayOptimisation' / 'Tracking_Data')]
    for candidate in candidates:
        if candidate and (Path(candidate) / 'game logs' / REAL_ARCHIVE).exists():
            return candidate
    return None


@pytest.mark.skipif(os.environ.get('SHOTS_REAL_GAME_TEST') != '1',
                    reason='opt-in: set SHOTS_REAL_GAME_TEST=1 (needs 7z and the DEN@GSW archive, ~1 GB RAM, ~10 s)')
def test_real_game_event_2_is_one_foot_from_the_rim(tmp_path, monkeypatch):
    data_dir = _real_data_dir()
    if data_dir is None:
        pytest.skip(f'{REAL_ARCHIVE} not found under TRACKING_DATA_DIR or the fetch_games.sh cache')
    monkeypatch.setenv('TRACKING_DATA_DIR', data_dir)
    monkeypatch.setenv('SHOTS_OUTPUT_DIR', str(tmp_path / 'out'))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shot_tracking, '_TABLES', None)
    features = shot_tracking.get_features('01.02.2016', 'GSW', 'DEN')
    rows = [r for r in features[1:] if r[0] == 2]
    assert len(rows) == 1
    event, dist, x, y = rows[0][0], rows[0][4], rows[0][5], rows[0][6]
    assert (x, y) == pytest.approx((5.28373, 26.13086), abs=0.01)
    assert dist == pytest.approx(1.13, abs=0.02)                    # the committed CSV says 48.87
    ys = [r[6] for r in features[1:]]
    assert min(ys) > -3 and max(ys) < 55                            # raw court coordinates
