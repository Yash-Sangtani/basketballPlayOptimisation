import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from getting_data import get_data

"""
features to calculate:
    dist: distance of pass
    passer_basket_dist
    ball_end_basket_dist:

first we need to find the pass attempts. Miss can be indentified easily beacause of the turnover
in events or any of the first 7 event type.
"""


class PassTracking:

    def __init__(self, date, home, away):
        self.ball_team = None   # Whose ball.
        self.home = home
        self.away = away
        data = get_data(date, home, away)
        self.home_id = data.home_id
        self.away_id = data.away_id
        self.game_id = data.game_id
        self.ball_team = int(data.pbp.loc[data.pbp['EVENTMSGTYPE'] == 10]['PLAYER3_TEAM_ID'].iloc[0])
        self.ball_man = int(data.pbp.loc[data.pbp['EVENTMSGTYPE'] == 10]['PLAYER3_ID'].iloc[0])
        self.pass_attempt = None
        self.moments = data.moments
        self.pbp = data.pbp

    def _dist_calculate(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def find_ball_man(self, positions):
        ball_x, ball_y = positions[0][2], positions[0][3]
        ball_man_id = None
        min_distance = float('inf')

        for player in positions[1:]:  # Skip the first entry as it's the ball
            distance = self._dist_calculate(ball_x, ball_y, player[2], player[3])
            if distance < min_distance:
                min_distance = distance
                player_id_with_ball = (player[0], player[1])  # team_id, player_id

        return player_id_with_ball[0], player_id_with_ball[1]    

    def track_passes(self):

        """While the shot cloc is less than twenty four, and the player id changes, pass successful. 
        If the shot clock resets to twenty-four and it wasn't a shot attempt, shot made, foul, violation, etc. (will get to knoow this from the event_num), 
        PASS UNSUCCESSFUL.
        

        # Modify getting_data.py to have event_num and event_type.
        Write the code to use the logic written above.
        Run the code.
        Calculate the features.
        Run the model.
        """
        possessions = self.moments.loc[((self.moments['shot_clock']==24.0) & (self.moments['shot_clock'] - self.moments['shot_clock'].shift(1) != 0.0)) |
                                       ((self.moments['shot_clock'] < 24.0) & (self.moments['shot_clock'].shift(-1) == 24.0))]
        possessions = possessions.loc[(possessions['shot_clock']==24.0) |(possessions['shot_clock'] <23.0)]
        possessions = possessions.loc[((possessions['shot_clock']==24.0) &(possessions['shot_clock'].shift(-1) != 24.0))|
                                      (possessions['shot_clock']!=24.0)]

        
        self.pass_attempt = possessions
        
        possessions = possessions['shot_clock']
        print(possessions)

        """
        self.moments[['ball_team', 'ball_man']] = self.moments['positions'].apply(self.find_ball_man).apply(pd.Series)
        self.moments['prev_ball_man'] = self.moments['ball_man'].shift(1)
        self.moments['prev_ball_team'] = self.moments['ball_team'].shift(1)
        self.moments['prev_2_ball_man'] = self.moments['ball_man'].shift(2)
        self.moments['prev_2_ball_team'] = self.moments['ball_team'].shift(2)
        self.moments['prev_3_ball_man'] = self.moments['ball_man'].shift(3)
        self.moments['prev_3_ball_team'] = self.moments['ball_team'].shift(3)
        required_columns = ['ball_team', 'ball_man', 'prev_ball_man', 'prev_ball_team', 
                            'prev_2_ball_man', 'prev_2_ball_team', 'prev_3_ball_man', 'prev_3_ball_team']
        self.moments.dropna(subset=required_columns, inplace=True)
        self.moments['pass_successful'] = np.where(self.moments['ball_team'] != self.moments['prev_2_ball_team'], 0, 1)
        print(len(self.moments.loc[self.moments['pass_successful'] == 1]))
        condition1 = self.moments['ball_man'].ne(self.moments['prev_ball_man'])
        condition2 = self.moments['ball_man'].ne(self.moments['prev_2_ball_man'])

        # Detecting pass attempts using bitwise OR on boolean arrays
        self.pass_attempt = self.moments.loc[condition1 | condition2]

        #self.pass_attempt = self.moments.loc[self.moments['ball_man'] != 
        #                                     self.moments['prev_ball_man'] | self.moments['prev_2_ball_man']]

        """
if __name__ == '__main__':
    passes = PassTracking('12.25.2015', 'GSW', 'CLE')
    passes.track_passes()
    passes.moments.to_csv('../Tracking_data/passes/passes.csv')
    passes.pass_attempt.to_csv('../Tracking_data/passes/possessions.csv')
    passes.pbp.to_csv('../Tracking_data/passes/pbp.csv')
    #passes.pass_attempt.to_csv('../Tracking_data/passes/passes.csv')
    print("CSV file saved and ready to use.")
    print(passes.pass_attempt)
    print(passes.pbp)
    #12.30.2015.PHI.at.SAC.7z
