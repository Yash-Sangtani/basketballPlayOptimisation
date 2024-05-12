import pandas as pd
import numpy as np
from getting_data import get_data
from tqdm import tqdm

"""
features to calculate:
    dist: distance of pass
    passer_basket_dist
    ball_end_basket_dist:

first we need to find the pass attempts. Miss can be indentified easily beacause of the turnover
in events or any of the first 7 event type.
"""

np.seterr(divide='ignore', invalid='ignore')

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
        self.basket1 = [5.35, 25]
        self.basket2 = [88.65, 25]
        self.pass_attempt = None
        self.moments = data.moments
        self.pbp = data.pbp



    def _dist_calculate(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def find_ball_man(self, positions):
        ball_x, ball_y = positions[0][2], positions[0][3]
        ball_man_id = None
        min_distance = float('inf')
        player_distance = None
        for player in positions[1:]:  # Skip the first entry as it's the ball
            distance = self._dist_calculate(ball_x, ball_y, player[2], player[3])
            if distance < min_distance:
                min_distance = distance
                player_id_with_ball = (player[0], player[1])  # team_id, player_id
                player_distance = [player[2], player[3]]

        return player_id_with_ball[1], player_id_with_ball[0], player_distance[0], player_distance[1]

    def track_passes(self):

        possessions = self.moments.loc[(self.moments['shot_clock']>=23.60) |
                                       ((self.moments['shot_clock'] < 23.0) & (self.moments['shot_clock'].shift(-1) == 24.0))]
        

        self.pbp['Qmin'] = (self.pbp['PCTIMESTRING'].str.split(':', expand=True)[0]).astype(int)
        self.pbp['Qsec'] = (self.pbp['PCTIMESTRING'].str.split(':', expand=True)[1]).astype(int)
        self.pbp['Qtime'] = (self.pbp['Qmin'].astype(int) * 60 + self.pbp['Qsec'].astype(int))
        index = self.moments.index.tolist()
        self.moments['pass_successful'] = None
        self.moments['passer'] = None
        self.moments['target'] = None
        self.moments['team_passer'] = None
        self.moments['team_target'] = None
        self.moments['backcourt'] = 0


        index = possessions.index.tolist()

        for i in range(len(index)-1):
            start = index[i]
            end = index[i+1]
            if self.moments.iloc[start, 3] <= 23.0 or self.moments.iloc[end, 3] >=23.0:
                continue
            basket = None
            self.ball_man, self.ball_team, target_x, target_y = self.find_ball_man(self.moments.iloc[end-5, 4])
            if (self._dist_calculate(target_x, target_y, self.basket1[0], self.basket1[1]) 
                < self._dist_calculate(target_x, target_y, self.basket2[0], self.basket2[1])):
                basket = self.basket1
            else:
                basket = self.basket2
            
            self.ball_man, self.ball_team, _, _ = self.find_ball_man(self.moments.iloc[start, 4]) #start or i?

            for j in range(start, end+1, 2):
                #Find the ball man before this loop.
                #Check if the ball_man changes inside this loop.
                #If yes, pass succcessful.
                temp, temp2, target_x, target_y = self.find_ball_man(self.moments.iloc[j, 4])  #player_id, team_id
                #print(temp)
                if temp != self.ball_man and temp2 == self.ball_team:
                    self.moments.iloc[j, 6] = 1 #Successful
                    self.moments.iloc[j, 7] = self.ball_man #passer
                    self.moments.iloc[j, 8] = temp #target
                    self.moments.iloc[j, 9] = self.ball_team #team
                    self.moments.iloc[j, 10] = temp2 #team
                    if (self._dist_calculate(target_x, target_y, self.basket1[0], self.basket1[1]) 
                        < self._dist_calculate(target_x, target_y, self.basket2[0], self.basket2[1])):
                        if basket == self.basket1:
                            self.moments.iloc[j, 11] = 1 #BACKCOURT PASS
                        else:
                            self.moments.iloc[j, 11] = 0
                    else:
                        if basket == self.basket2:
                            self.moments.iloc[j, 11] = 1
                        else:
                            self.moments.iloc[j, 11] = 0

                    self.ball_man=temp

            #Converting quater_time to minute and seconds to compare to PCTIMESTRING in pbp data.
            time_end_str = str(self.moments.iloc[end, 2])
            time_end_min = int(time_end_str.split('.')[0])
            time_end_sec = time_end_min % 60  
            time_end_min = time_end_min // 60
            #time_end_sec = [time_end_sec-1, time_end_sec, time_end_sec+1]
            quater = self.moments.iloc[end, 0]

            #match = self.pbp.loc[(self.pbp['Qmin'] == time_end_min) & (self.pbp['Qsec']==time_end_sec) & (self.pbp['PERIOD'] == quater) & (self.pbp['EVENTMSGTYPE'] == 5)]

            bad_passes = self.pbp.loc[self.pbp['EVENTMSGTYPE'] == 5]

            bad_passes = bad_passes.loc[bad_passes['PERIOD'] == quater]

            bad_passes = bad_passes.loc[bad_passes['Qmin'] == time_end_min]
            bad_passes = bad_passes.loc[(bad_passes['Qsec'] <= time_end_sec+20) & (bad_passes['Qsec'] >= time_end_sec-20)]
            if len(bad_passes) == 1:
                self.moments.iloc[end, 6] = 0 #unsuccessful pass.
                self.moments.iloc[j, 7] = self.ball_man #passer
                self.moments.iloc[j, 8] = None
                self.moments.iloc[j, 9] = self.ball_team #team
                self.moments.iloc[j, 10] = None

            #Checking for who took the rebound is unnecessary since the possesion is ending so, it must be the other team.
            i=i+1

        #print("Passes Extracted successfully.")
        output = self.moments.loc[(self.moments['pass_successful'] == 1) | (self.moments['pass_successful'] == 0)]
        change_possession = possessions.loc[possessions['shot_clock'] != 24.0]

        output = output.loc[output['shot_clock'].shift(1) != output['shot_clock']]

        #print("moments")
        #print(self.moments)
        #print()
        #output.dropna(how='any', inplace=True)
        #print("passes_attempt")
        #print(output)
        self.pass_attempt = output
        self.pass_attempt.reset_index(inplace=True)


class features:

    def __init__(self, date, home, away):
        self.passes = PassTracking('12.12.2015', 'MIL', 'GSW')
        self.passes.track_passes()
        self.features = pd.DataFrame(self.passes.pass_attempt[['shot_clock', 'pass_successful', 'backcourt']])
        
    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
        vector1 = [x2 - x1, y2 - y1]
        vector2 = [x3 - x2, y3 - y2]
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        dot_product = np.dot(unit_vector1, unit_vector2)
        clipped_dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(clipped_dot_product)
        return angle

    def features_extraction(self):
        #dist
        dist = []
        dist_basket = []
        target_basket = []
        basket_angle = []
        passer_closest_def = []
        target_closest_def = []
        closest_def_trajectory = []
        closest_def_angle_passer = []
        closest_def_angle_ball_end = []
        closest_def_trajectory_angle = []
        for index, row in self.passes.pass_attempt.iterrows():
            passer = row['passer']
            team = row['team_passer']
            target = row['target']
            def_dist = 0
            def_dist_target = 0
            passer_x = passer_y = target_x = target_y = 0
            def_passer_x = def_passer_y = def_target_x = def_target_y = 0
            positions = row['positions']
            for pos in positions:
                if pos[1] == passer:
                    passer_x = pos[2]
                    passer_y = pos[3]
                elif pos[1] == target:
                    target_x = pos[2]
                    target_y = pos[3]
            passer_def = [10000000, None, 0.0, 0.0]
            target_def = [10000000, None, 0.0, 0.0]
            A = target_y - passer_y
            B = passer_x - target_x
            C = target_x * passer_y - passer_x * target_y
            traj_def = [100000, None, 0.0, 0.0]
            for pos in positions:
                #passer
                if pos[0] != team and self.passes._dist_calculate(passer_x, passer_y, pos[2], pos[3]) < passer_def[0]:
                    passer_def[0] = self.passes._dist_calculate(passer_x, passer_y, pos[2], pos[3])
                    passer_def[1] = pos[1]
                    passer_def[2] = pos[2]
                    passer_def[3] = pos[3]
                #target
                if pos[0] != team and self.passes._dist_calculate(passer_x, passer_y, pos[2], pos[3]) < target_def[0]:
                    target_def[0] = self.passes._dist_calculate(target_x, target_y, pos[2], pos[3])
                    target_def[1] = pos[1]
                    target_def[2] = pos[2]
                    target_def[3] = pos[3]

                if pos[0] != team and np.abs(A * pos[2] + B * pos[3] + C) / np.sqrt(A**2 + B**2) < traj_def[0]:
                    traj_def[0] = np.abs(A * pos[2] + B * pos[3] + C) / np.sqrt(A**2 + B**2)
                    traj_def[1] = pos[1]
                    traj_def[2] = pos[2]
                    traj_def[3] = pos[3]


            
            basket = None
            temp_basket = None
            if (self.passes._dist_calculate(target_x, target_y, self.passes.basket1[0], self.passes.basket1[1]) 
                < self.passes._dist_calculate(target_x, target_y, self.passes.basket2[0], self.passes.basket2[1])):
                basket = self.passes.basket1
            else:
                basket = self.passes.basket2
            """
            if (self.passes._dist_calculate(passer_x, passer_y, self.passes.basket1[0], self.passes.basket1[1]) 
                < self.passes._dist_calculate(passer_x, passer_y, self.passes.basket2[0], self.passes.basket2[1])):
                temp_basket = self.passes.basket1
            else:
                temp_basket = self.passes.basket2
            """
            
            #Using the basket closest to the target. 
            dist.append(self.passes._dist_calculate(passer_x, passer_y, target_x, target_y))
            dist_basket.append(self.passes._dist_calculate(passer_x, passer_y, basket[0], basket[1]))
            target_basket.append(self.passes._dist_calculate(target_x, target_y, basket[0], basket[1]))
            basket_angle.append(self.calculate_angle(passer_x, passer_y, basket[0], basket[1], target_x, target_y))
            passer_closest_def.append(passer_def[0])
            target_closest_def.append(target_def[0])
            closest_def_trajectory.append(traj_def[0])
            closest_def_angle_passer.append(self.calculate_angle(target_x, target_y, passer_x, passer_y, passer_def[2], passer_def[3]))
            closest_def_angle_ball_end.append(self.calculate_angle(passer_x, passer_y, target_x, target_y, target_def[2], target_def[3]))
            closest_def_trajectory_angle.append(self.calculate_angle(passer_x, passer_y, target_x, target_y, traj_def[2], traj_def[3]))
            """
            dist_basket.append(min(self.passes._dist_calculate(passer_x, passer_y, self.passes.basket1[0], self.passes.basket1[1]),
                                   self.passes._dist_calculate(passer_x, passer_y, self.passes.basket2[0], self.passes.basket2[1])))
            target_basket.append(min(self.passes._dist_calculate(target_x, target_y, self.passes.basket1[0], self.passes.basket1[1]),
                                   self.passes._dist_calculate(target_x, target_y, self.passes.basket2[0], self.passes.basket2[1])))
            """

        self.features['dist'] = dist
        self.features['passer_basket_dist'] = dist_basket
        self.features['ball_end_basket_dist'] = target_basket
        self.features['basket_angle'] = basket_angle
        self.features['closest_def_dist_passer'] = passer_closest_def
        self.features['closest_def_dist_ball_end'] = target_closest_def
        self.features['closest_def_trajectory'] = closest_def_trajectory
        self.features['closest_def_angle_passer'] = closest_def_angle_passer
        self.features['closest_def_angle_ball_end'] = closest_def_angle_ball_end
        self.features['closest_def_trajectory_angle'] = closest_def_trajectory_angle        
        #print(self.features)
        #backcourt

        """
        I will check the dist to basket for both, the passer and target. if the basket is different,
        I will use the target's basket as the pass cannot go the other way or else it would have been back court.
        """
        pass



def process_features(date_away_home):
    date, away, home = date_away_home
    try:
        feature = features(date, home, away)
        feature.features_extraction()
        feature.features.to_pickle(f'./../Tracking_data/pass_features/{date}.{away}.{home}.pkl')
    except Exception as e:
        print(f"Error processing {date}.{home}.{away}: {e}")




def main():

    """passes = PassTracking('12.12.2015', 'MIL', 'GSW')
    # passes.moments.to_csv('../Tracking_data/passes/moments.csv')
    passes.track_passes()
    passes.pbp.to_csv('../Tracking_data/passes/pbp.csv')
    passes.pass_attempt.to_csv('../Tracking_data/passes/passes.csv')
    """
    dates = []
    aways = []
    homes = []
    with open('./allgames.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            components = line.split('.')
            dates.append('.'.join(components[:3]))
            aways.append(components[3])
            homes.append(components[5].split()[0])
    
    """
    for date, away, home in tqdm(zip(dates, aways, homes), total=len(dates)):
        try:
            feature = features(date, home, away)
            feature.features_extraction()
            feature.features.to_pickle(f'./../Tracking_data/pass_features/{date+"."+away+"."+home}.pkl')
        except:
            print(date, home, away, sep='.', end='.7z')
    """
    feature = features('12.12.2015', 'MIL', 'GSW')
    feature.features_extraction()
    feature.features.to_csv('./../Tracking_data/passes/features.csv')
    print("CSV file saved and ready to use.")
    #12.30.2015.PHI.at.SAC.7z
        
if __name__ == '__main__':
    main()
