import pandas as pd
import numpy as np
import os
import json
import subprocess
import tempfile
"""
Reads the trscking data in a more readable format.
This can help us make our models more easily.
"""

#Where Tracking_Data lives. Override with the TRACKING_DATA_DIR environment variable
#(e.g. a local copy made by fetch_games.sh, outside any iCloud-synced folder).
DATA_DIR = os.environ.get('TRACKING_DATA_DIR', '../Tracking_Data')
SF_DATALESS = 0x40000000   #macOS file flag: listed in the folder, contents not stored here


class DatalessFileError(RuntimeError):
    """The file is an iCloud placeholder. Reading it does NOT fail, it silently
    returns nothing, so check before reading."""


def require_local(path):
    st = os.stat(path)
    if getattr(st, 'st_flags', 0) & SF_DATALESS:
        raise DatalessFileError(
            f"{path} is an iCloud placeholder (not downloaded to this Mac). Download it "
            f"(Finder: Download Now) or copy the games out of git with fetch_games.sh and "
            f"set TRACKING_DATA_DIR.")

class get_data:
    def __init__(self, date, home, away):
        """
        Args:
            date (str): 'MM.DD.YYYY', date of game
            home (str): 'XXX', abbreviation of home team in data
                tracking file name
            away (str): 'XXX', abbreviation of away team in data
                tracking file name
 
        Attributes:
            date (str): 'MM.DD.YYYY', date of game
            home (str): 'XXX', abbreviation of home team in data
                tracking file name
            away (str): 'XXX', abbreviation of away team in data
                tracking file name
            tracking_id (str): id to access player tracking data
                Due to the way the SportVU data is stored, game_id is
                complicated: 'MM.DD.YYYY.AWAYTEAM.at.HOMETEAM'
                For Example: 01.13.2016.GSW.at.DEN
            tracking_data (dict): Dictionary of unstructured tracking
                data scraped from GitHub.
            game_id (str): ID for game. Luckily, SportVU and play by
                play use the same game ID
            pbp (pd.DataFrame): Play by play data.  33 columns per pbp
                instance.
            moments (pd.DataFrame): DataFrame of player tracking data.
                Each entry is a single snapshot of where the players
                are at a given time on the court.
                Columns: ['quarter', 'universe_time', 'quarter_time',
                'shot_clock', 'positions', 'game_time'].
                moments['positions'] contains a list of where each player
                and the ball are located.
            player_ids (dict): dictionary of {player: player_id} for
                all players in game.
            away_id (int): ID of away team
            home_id (int): ID of home team
            team_colors (dict): dictionary of colors for each team and
                ball. Used for plotting.
            home_team (str): 'XXX', abbreviation of home team
            away_team (str): 'XXX', abbreviation of the away team
        """
 
        self.date = date
        self.home = home
        self.away = away
        self.tracking_id = f'{self.date}.{self.away}.at.{self.home}'
        self.tracking_data = None
        self.game_id = None
        self.pbp = None
        self.moments = None
        self.player_ids = None
        self._get_tracking_data()
        self._get_play_by_play_data()
        self.data_frame_tracking()
        self.away_id = self.tracking_data['events'][0]['visitor']['teamid']
        self.home_id = self.tracking_data['events'][0]['home']['teamid']
        self.home_team = (self.tracking_data['events'][0]['home']
        ['abbreviation'])
        self.away_team = (self.tracking_data['events'][0]['visitor']
        ['abbreviation'])
 
    def _get_tracking_data(self):
        """
        Retrieves tracking data from basketballPlay/Optimisation/Tracking_Data/'game logs' folder
        The tracking data is in .7z format with in 'date.team1.at.team2' format. It is extracted
        into a private temporary directory that is removed afterwards, so a run that is killed
        part-way can't leave a stale JSON for a later run to mistake for its own game, and
        several runs can go side by side. Raises if the archive can't be read.
        """
        tracking_folder = os.path.join(DATA_DIR, 'game logs')
        tracking_file_path = os.path.join(tracking_folder, f'{self.tracking_id}.7z')
        require_local(tracking_file_path)

        with tempfile.TemporaryDirectory(prefix='tracking_') as output_path:
            result = subprocess.run(['7z', 'e', tracking_file_path, f'-o{output_path}', '-y'],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"7z could not extract {tracking_file_path}: "
                                   f"{(result.stderr or result.stdout).strip()[-200:]}")
            jsons = [f for f in os.listdir(output_path) if f.endswith('.json')]
            if len(jsons) != 1:
                raise RuntimeError(f"expected one JSON in {tracking_file_path}, found {jsons}")
            self.game_id = jsons[0][:-5]   #The json is named after the game id.
            with open(os.path.join(output_path, jsons[0]), 'r') as data_file:
                self.tracking_data = json.load(data_file)

    def _get_play_by_play_data(self):
        path = os.path.join(DATA_DIR, 'events')
        file_path = f'{path}/{self.game_id}.csv'
        require_local(file_path)
        self.pbp = pd.read_csv(file_path)
        #print('Read the events data for this match.')

    def data_frame_tracking(self):
        events = pd.DataFrame(self.tracking_data['events'])
        moments = []
        for row in events['moments']:
            for inner_row in row:
                moments.append(inner_row)
        moments = pd.DataFrame(moments)
        moments = moments.drop_duplicates(subset=[1])
        moments = moments.reset_index()

        moments.columns = ['index', 'quarter', 'universe_time', 'quarter_time',
                           'shot_clock', 'unknown', 'positions']
        moments['game_time'] = (moments.quarter - 1) * 720 + \
                               (720 - moments.quarter_time)
        moments.drop(['index', 'unknown'], axis=1, inplace=True)

        self.moments = moments


if __name__ == '__main__':
    data = get_data('12.02.2015', 'WAS', 'LAL')
    #12.02.2015.LAL.at.WAS.7z

