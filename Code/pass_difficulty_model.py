import pandas as pd
import numpy as np
from getting_data import get_data

"""
features to calculate:
    dist: distance of pass
    passer_basket_dist
    ball_end_basket_dist:

first we need to find the pass attempts. Miss can be indentified easily beacause of the turnover
in events or any of the first 7 event type.
"""


class passes:

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
        print(self.moments.iloc[0])

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

        return player_id_with_ball
    
    def track_passes(self):
        self.moments['ball_man'] = self.moments['positions'].apply(find_ball_man)
        self.moments['prev_ball_man'] = self.moments['ball_man'].shift(1)
        self.moments['prev_2_ball_man'] = self.moments['ball_man'].shift(2)
        self.moments['prev_3_ball_man'] = self.moments['ball_man'].shift(3)
        moments_df['pass_occurred'] = moments_df['player_with_ball'] != moments_df['previous_player_with_ball']
        
        """
        separate out the ball_man to ball_man and ball_team. Separate columns would be perfect.
        Successful pass = if  prev_3's team_id == ball_man's team_id.
        Unsuccessful pass = if not equal.
        """
        # Optional: Add a condition to ignore changes that are too fast to be realistic passes
        # For instance, exclude changes if they happen within less than 1 second if your data resolution allows
        return moments_df

if __name__ == '__main__':
    passs = passes('12.30.2015', 'SAC', 'PHI')
    print(passs.game_id)
    print(passs.find_ball_man(passs.moments['positions'].iloc[0]))
    #12.30.2015.PHI.at.SAC.7z
