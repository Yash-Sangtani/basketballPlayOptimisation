"""
Scripts for extracting features from the NBA tracking data
"""

import os
import pickle
import numpy as np
import pandas as pd
from simulate import Game


def extract_games():
    """
    Extract games from allgames.txt

    Returns:
        list: list of games.  Each element is list is
            [date, home_team, away_team]
            an example element: ['01.01.2016', 'TOR', 'CHI']
    """

    games = []
    with open('ag.txt', 'r') as game_file:
        for line in game_file:
            game = line.strip().split('.')
            date = "{game[0]}.{game[1]}.{game[2]}".format(game=game)
            away = game[3]
            home = game[5]
            games.append([date, home, away])
    return games


def get_features(date, home_team, away_team, write_file=True,
                 write_score=False, write_game=False):
    """
    Calculates features for each frame in game

    Args:
        date (str): date of game in form 'MM.DD.YYYY'.  Example: '01.01.2016'
        home_team (str): home team in form 'XXX'. Example: 'TOR'
        away_team (str): away team in form 'XXX'. Example: 'CHI'
        write_file (bool): If True, write pickle file of spacing
            statistics into data/spacing directory
        write_score (bool): If True, write pickle file of game score
            into data/score directory
        write_game (bool): If True, write pickle file of tracking data
            into data/game directory
            Note: This file is ~100MB.

        Returns:
        tuple: tuple of data:
            (
            dist : distance of the shot which is basically shooter to basket distance -> float,
            x : x coordinate of the shot -> float,
            y : y coordinate of the shot -> float,
            make : 1 if the shot was made, 0 if missed -> int,
            shot_angle : angle of the shot w.r.t court center -> float,
            shooter_velocity : velocity of the shooter in ft/msec -> float,
            closest_defender : distance of the closest defender to the shooter -> float,
            closest_defender_angle : angle of the closest defender w.r.t shot trajectory -> float,
            closest_defender_velocity : velocity of the closest defender in ft/msec -> float,
            num_close_defenders : number of defenders within 4 feet of the shooter -> int,
            shot_clock : time left on the shot clock -> float,
            score_margin : score margin of the game -> int)
            quarter : quarter of the game -> int
    """
    filename = ("{date}-{away_team}-"
                "{home_team}.pkl").format(date=date,
                                          away_team=away_team,
                                          home_team=home_team)
    # Do not recalculate spacing data if already saved to disk
    if filename in os.listdir('./data/features'):
        return
    game = Game(date, home_team, away_team)

    # Extract features for each frame
    features = []

    # Get all shot frames where EVENTMSGTYPE == 1 (shot made) or 2 (shot missed)
    shot_frames = sorted(game.pbp[game.pbp['EVENTMSGTYPE'] == 1]['EVENTNUM'].tolist() +
                         game.pbp[game.pbp['EVENTMSGTYPE'] == 2]['EVENTNUM'].tolist())

    for frame in shot_frames:
        # Get the frame of the shot
        shot_frame = game.get_play_frames(event_num=frame)[1]

        # Get the moment details as per the frame
        details = game._get_moment_details(shot_frame)

        # Shot clock time
        shot_clock = int(details[6])
        # Quarter of the game
        quarter = details[5]

        # Get the id of the shooter
        shooter = game.pbp[game.pbp['EVENTNUM'] == frame]['PLAYER1_ID'].values[0]
        shooter_index = details[10].index(shooter)

        # Get the x and y coordinates of the shooter
        shooter_coords = [details[1][shooter_index], details[2][shooter_index]]

        home_team_ids = [i for i in details[10] if i in list(game.player_ids.values())[:13]]
        away_team_ids = [i for i in details[10] if i in list(game.player_ids.values())[13:]]

        # Get the offensive team
        if shooter in home_team_ids:
            offensive_team = home_team_ids
            defensive_team = away_team_ids
        else:
            offensive_team = away_team_ids
            defensive_team = home_team_ids

        # Get the x and y coordinates of the defenders
        defender_coords = [[details[1][details[10].index(defender)],
                            details[2][details[10].index(defender)]]
                           for defender in defensive_team]

        # Get the distance of the closest defender
        closest_defender_coord = defender_coords[
            np.argmin([np.linalg.norm(np.array(shooter_coords) - np.array(defender))
                       for defender in defender_coords])]
        closest_defender_dist = np.linalg.norm(np.array(shooter_coords) - np.array(closest_defender_coord))

        # Determine the basket coordinates based on which basket is closer to the shooter [5.35, 25] or [88.65, 25]
        basket = [5.35, 25] if shooter_coords[0] < 47 else [88.65, 25]

        # Get the distance of the shot
        dist = np.linalg.norm(np.array(shooter_coords) - np.array(basket))

        # Calculate the angle between the shooter, the basket, and the center of the court
        court_center = [47, 25]  # Assuming this is the center of the court along the x-axis
        shooter_basket_vector = np.array(basket) - np.array(shooter_coords)
        center_basket_vector = np.array(basket) - np.array(court_center)
        shot_angle = np.arccos(np.dot(shooter_basket_vector, center_basket_vector) /
                               (np.linalg.norm(shooter_basket_vector) * np.linalg.norm(center_basket_vector)))

        # Calculate the angle between the shooter and the closest defender and the basket
        shooter_defender_vector = np.array(closest_defender_coord) - np.array(shooter_coords)
        shot_defender_angle = np.arccos(np.dot(shooter_basket_vector, shooter_defender_vector) /
                                        (np.linalg.norm(shooter_basket_vector) * np.linalg.norm(
                                            shooter_defender_vector)))

        # Get the velocity of the shooter
        previous_details = game._get_moment_details(shot_frame - 1)

        delta_x = np.array(details[1]) - np.array(previous_details[1])
        delta_y = np.array(details[2]) - np.array(previous_details[2])
        delta_time = details[9] - previous_details[9]
        shooter_velocity = np.linalg.norm([delta_x[shooter_index], delta_y[shooter_index]]) / delta_time

        # Get the velocity of the closest defender
        closest_defender_index = details[1].index(closest_defender_coord[0])
        defender_velocity = np.linalg.norm(
            [delta_x[closest_defender_index], delta_y[closest_defender_index]]) / delta_time

        # Get the number of defenders within 4 feet of the shooter
        num_close_defenders = sum([1 for defender in defender_coords
                                   if np.linalg.norm(np.array(shooter_coords) - np.array(defender)) < 4])

        # Get the score margin of t`he game
        score_margin = game.pbp[game.pbp['EVENTNUM'] == frame]['SCOREMARGIN'].values[0]

        # if the score margin is float nan and not found, go back to the previous frames until a score margin is found

        temp = frame
        if score_margin == pd.isna(score_margin):
            while score_margin == pd.isna(score_margin) and temp > 0:
                temp -= 1
                score_margin = game.pbp[game.pbp['EVENTNUM'] == temp]['SCOREMARGIN'].values[0]

        if score_margin == 'TIE' or pd.isna(score_margin):
            score_margin = 0
        elif offensive_team == away_team_ids:
            score_margin = -int(score_margin)
        else:
            score_margin = int(score_margin)

        # Get the make -> 1 if the shot was made, 0 if missed
        make = 1 if game.pbp[game.pbp['EVENTNUM'] == frame]['EVENTMSGTYPE'].values[0] == 1 else 0

        # Append the features to the list
        features.append((dist, shooter_coords[0], shooter_coords[1], make, shot_angle,
                         shooter_velocity, closest_defender_dist, shot_defender_angle,
                         defender_velocity, num_close_defenders, shot_clock, score_margin, quarter))

    # Write features to disk
    if write_file:
        with open('./data/features/{filename}'.format(filename=filename), 'wb') as myfile:
            pickle.dump(features, myfile)

    # Also write the features in csv format
    with open('./data/features/{filename}.csv'.format(filename=filename), 'w') as myfile:
        myfile.write('dist, x, y, make, shot_angle, shooter_velocity, closest_defender, closest_defender_angle, '
                     'closest_defender_velocity, num_close_defenders, shot_clock, score_margin, quarter\n')
        for feature in features:
            myfile.write(','.join([str(f) for f in feature]) + '\n')

    print("Features extracted for {filename}".format(filename=filename))

    return features


def write_features(gamelist):
    """
    Writes all spacing statistics to data/spacing directory for each game
    """
    for game in gamelist:
        try:
            get_features(game[0], game[1], game[2],
                         write_file=True, write_score=False)
        except:
            with open('errorlog.txt', 'a') as myfile:
                myfile.write("{game} Could not extract spacing data\n"
                             .format(game=game))


write_features(extract_games())
