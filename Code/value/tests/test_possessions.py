"""Tests for possessions.py: one small synthetic event sequence per rule, plus a real game."""
import os

import numpy as np
import pandas as pd
import pytest

from possessions import build_possessions, box_estimate, elapsed, clock_seconds

A, H = 1610612737, 1610612738          # away, home team ids
PA, PH = 1001, 2001                     # a player on each team
COLS = ['GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE', 'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION',
        'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION', 'SCORE', 'SCOREMARGIN', 'PLAYER1_ID', 'PLAYER1_TEAM_ID',
        'PLAYER2_ID', 'PLAYER3_TEAM_ID']


class Game:
    """Builds a pbp DataFrame event by event (file order = call order)."""

    def __init__(self):
        self.rows, self.n = [], 0

    def ev(self, etype, team=None, clock='11:00', period=1, action=0, desc='', score=None, team_event=False,
           p3_team=None):
        """team_event: the event is recorded against the team itself (team rebound, shot clock turnover):
        PLAYER1_ID = team id and PLAYER1_TEAM_ID is empty."""
        self.n += 1
        player = PA if team == A else PH
        self.rows.append({'GAME_ID': 1, 'EVENTNUM': self.n, 'EVENTMSGTYPE': etype, 'EVENTMSGACTIONTYPE': action,
                          'PERIOD': period, 'PCTIMESTRING': clock, 'HOMEDESCRIPTION': desc if team == H else np.nan,
                          'NEUTRALDESCRIPTION': np.nan, 'VISITORDESCRIPTION': desc if team == A else np.nan,
                          'SCORE': score if score else np.nan, 'SCOREMARGIN': np.nan,
                          'PLAYER1_ID': team if team_event else (player if team else 0),
                          'PLAYER1_TEAM_ID': np.nan if (team_event or not team) else team,
                          'PLAYER2_ID': np.nan, 'PLAYER3_TEAM_ID': p3_team if p3_team else np.nan})
        return self

    # shorthands
    def made(self, team, score, clock='11:00', period=1, desc='Jump Shot (2 PTS)'):
        return self.ev(1, team, clock, period, 1, desc, score)

    def miss(self, team, clock='11:00', period=1):
        return self.ev(2, team, clock, period, 1, 'MISS Jump Shot')

    def reb(self, team, clock='11:00', period=1, team_event=False):
        return self.ev(4, team, clock, period, 0, 'REBOUND (Off:0 Def:1)', team_event=team_event)

    def tov(self, team, clock='11:00', period=1, team_event=False, action=1):
        return self.ev(5, team, clock, period, action, 'Turnover', team_event=team_event)

    def ft(self, team, k, n, made, score=None, clock='11:00', period=1, kind='Free Throw'):
        d = f'{"" if made else "MISS "}{kind} {k} of {n}' if kind != 'Free Throw Technical' else f'{"" if made else "MISS "}Free Throw Technical'
        action = 16 if 'Technical' in kind else 10 + k
        if 'Flagrant' in kind:
            action = 17 + k
        return self.ev(3, team, clock, period, action, d, score)

    def foul(self, team, clock='11:00', period=1):
        return self.ev(6, team, clock, period, 1, 'FOUL')

    def start_period(self, period=1, clock=None):
        clock = clock or ('12:00' if period <= 4 else '5:00')
        return self.ev(12, None, clock, period)

    def end_period(self, period=1):
        return self.ev(13, None, '0:00', period)

    def build(self):
        return build_possessions(pd.DataFrame(self.rows, columns=COLS), game_id=1)


def with_scores():
    """Game that starts with one made basket for each team so home/away can be inferred."""
    g = Game().start_period(1)
    g.made(A, '2 - 0', clock='11:50').made(H, '2 - 2', clock='11:40')
    return g


# ----------------------------------------------------------------------------- rules
def test_clock_helpers():
    assert clock_seconds('11:41') == 701
    assert elapsed(1, 720) == 0 and elapsed(4, 0) == 2880
    assert elapsed(5, 300) == 2880 and elapsed(5, 0) == 3180   #overtime periods are 300 s


def test_made_fg_ends_possession_and_points_from_score_column():
    g = Game().start_period(1)
    g.made(A, '3 - 0', clock='11:30', desc='25 ft 3PT (3 PTS)')    #a three: points come from SCORE, not the text
    g.miss(H, '11:10').reb(A, '11:08').made(A, '5 - 0', clock='10:50')
    t, info = g.build()
    assert list(t.offense_team) == [A, H, A]                     #A scores, H misses (A rebounds), A scores
    assert t.iloc[0].points == 3 and t.iloc[0].end_reason == 'made_fg'
    assert info['final'] == (5, 0) and info['other_points'] == {A: 0, H: 0}
    assert t.points.sum() == 5


def test_and_one_made_free_throw_belongs_to_the_possession():
    g = with_scores()
    g.made(A, '4 - 2', clock='10:30').foul(H, '10:30').ft(A, 1, 1, True, '5 - 2', clock='10:30')
    t, info = g.build()
    poss = t[t.offense_team == A]
    assert len(poss) == 2                                   #the 2-point make + this one (not 3)
    assert poss.iloc[-1].points == 3 and poss.iloc[-1].end_reason == 'made_ft'
    assert poss.iloc[-1].fga == 1 and poss.iloc[-1].fta == 1
    assert info['diag']['and_one_reopens'] == 1


def test_and_one_missed_free_throw_waits_for_the_rebound():
    g = with_scores()
    g.made(A, '4 - 2', clock='10:30').foul(H, '10:30').ft(A, 1, 1, False, clock='10:30').reb(H, '10:29')
    t, info = g.build()
    last_a = t[t.offense_team == A].iloc[-1]
    assert last_a.points == 2 and last_a.end_reason == 'def_rebound'
    assert t.iloc[-1].offense_team == H                      #the rebounder's possession is open at the end of data


def test_dead_ball_team_rebound_between_free_throws_is_ignored():
    g = with_scores()
    g.foul(H, '10:30').ft(A, 1, 2, False, clock='10:30')
    g.reb(A, '10:30', team_event=True)                      #"Team Rebound": placeholder, not an OREB
    g.ft(A, 2, 2, True, '3 - 2', clock='10:30')
    t, info = g.build()
    p = t[(t.offense_team == A)].iloc[-1]
    assert p.points == 1 and p.end_reason == 'made_ft' and p.oreb == 0 and p.fta == 2
    assert info['diag']['dead_ball_rebounds'] == 1
    assert box_estimate(pd.DataFrame(g.rows, columns=COLS))[A]['oreb'] == 0


def test_missed_final_free_throw_offensive_rebound_continues_defensive_ends():
    g = with_scores()
    g.ft(A, 1, 2, True, '3 - 2', clock='10:30').ft(A, 2, 2, False, clock='10:30').reb(A, '10:29').made(A, '5 - 2', clock='10:20')
    g.ft(H, 1, 2, False, clock='9:30').ft(H, 2, 2, False, clock='9:30').reb(A, '9:29')
    t, info = g.build()
    p = t[t.offense_team == A].iloc[1]
    assert p.oreb == 1 and p.points == 3 and p.end_reason == 'made_fg'
    q = t[t.offense_team == H].iloc[1]
    assert q.end_reason == 'def_rebound' and q.points == 0


def test_defensive_technical_free_throw_does_not_change_possession():
    g = with_scores()
    g.miss(A, '10:30')                                       #A shoots, rebound pending
    g.ev(3, H, '10:30', 1, 16, 'Free Throw Technical (3 PTS)', '2 - 3')
    g.reb(A, '10:29').made(A, '4 - 3', clock='10:10')
    t, info = g.build()
    assert info['other_points'][H] == 1 and info['diag']['other_tech_points'] == 1
    a = t[t.offense_team == A].iloc[1]
    assert a.fga == 2 and a.oreb == 1 and a.points == 2     #possession 2 continued across the technical
    assert t.points.sum() + sum(info['other_points'].values()) == sum(info['final'])


def test_flagrant_free_throws_keep_the_ball_with_the_fouled_team():
    g = with_scores()                                        #H made last, so A gets the ball next
    g.ft(A, 1, 2, True, '3 - 2', clock='10:30', kind='Free Throw Flagrant')
    g.ft(A, 2, 2, False, clock='10:30', kind='Free Throw Flagrant')
    g.reb(H, '10:30', team_event=True)                      #dead-ball placeholder
    g.made(A, '5 - 2', clock='10:10')
    t, info = g.build()
    p = t[t.offense_team == A].iloc[1]
    assert p.points == 3 and p.fta == 2 and p.fga == 1       #free throws + the basket are one possession
    assert len(t[t.offense_team == A]) == 2 and info['diag']['dead_ball_rebounds'] == 1


def test_team_turnover_shot_clock_uses_player1_id_as_the_team():
    g = with_scores()
    g.tov(A, '10:30', team_event=True, action=11)
    g.made(H, '4 - 2', clock='10:10')
    t, info = g.build()
    p = t.iloc[2]
    assert p.offense_team == A and p.end_reason == 'turnover' and p.points == 0
    assert t.iloc[3].offense_team == H


def test_offensive_rebound_continues_the_possession():
    g = with_scores()
    g.miss(A, '10:30').reb(A, '10:28').miss(A, '10:20').reb(A, '10:18').made(A, '4 - 2', clock='10:10')
    t, info = g.build()
    p = t.iloc[2]
    assert p.offense_team == A and p.fga == 3 and p.oreb == 2 and p.points == 2 and p.end_reason == 'made_fg'
    assert len(t) == 3
    assert (p.start_clock, p.end_clock) == (700, 610)            #starts when H scored (11:40), ends at 10:10


def test_jump_ball_to_other_team_ends_the_possession_to_same_team_does_not():
    g = with_scores()
    g.miss(A, '10:30')
    g.ev(10, A, '10:29', 1, 0, 'Jump Ball', p3_team=A)       #held ball, tipped to A: possession continues
    g.ev(4, A, '10:29', 1, 1, 'Rebound', team_event=True)     #placeholder after the jump ball
    g.made(A, '4 - 2', clock='10:10')
    g.miss(A, '9:50').ev(10, H, '9:49', 1, 0, 'Jump Ball', p3_team=H).tov(H, '9:40')
    t, info = g.build()
    a = t[t.offense_team == A]
    assert a.iloc[1].fga == 2 and a.iloc[1].points == 2
    assert a.iloc[2].end_reason == 'jump_ball'
    assert t.iloc[-1].offense_team == H and t.iloc[-1].end_reason == 'turnover'


def test_period_boundaries_and_overtime_length():
    g = Game().start_period(1).made(A, '2 - 0', clock='11:00').end_period(1)
    g.start_period(2).tov(H, '11:30', period=2).made(A, '4 - 0', clock='11:00', period=2)
    g.end_period(2)
    g.start_period(5).miss(H, '4:00', period=5).reb(A, '3:58', period=5).end_period(5)
    t, info = g.build()
    assert list(t.period) == [1, 2, 2, 5, 5]
    assert t.iloc[1].offense_team == H and t.iloc[1].start_clock == 720   #offense of period 2 comes from its first event
    assert t.iloc[3].start_clock == 300 and t.iloc[3].offense_team == H   #OT has 300 s
    assert t.iloc[3].end_reason == 'def_rebound' and t.iloc[4].end_reason == 'end_of_period'
    assert t.iloc[4].end_elapsed == elapsed(5, 0)
    assert t.start_elapsed.is_monotonic_increasing


def test_file_order_is_used_not_eventnum():
    """The rebound has a larger EVENTNUM than the putback but sits before it in the file, as in the data."""
    g = with_scores()
    g.miss(A, '10:30')
    g.ev(4, A, '10:20', 1, 0, 'REBOUND (Off:1 Def:0)')
    g.made(A, '4 - 2', clock='10:20')
    g.rows[-1]['EVENTNUM'], g.rows[-2]['EVENTNUM'] = 100, 101   #EVENTNUM disagrees with file order
    t, info = g.build()
    assert len(t) == 3 and t.iloc[2].oreb == 1 and t.iloc[2].fga == 2


def test_possession_points_plus_other_points_equal_final_score():
    g = with_scores()
    g.ev(3, A, '10:30', 1, 16, 'Free Throw Technical (3 PTS)', '3 - 2')
    g.made(H, '3 - 4', clock='10:10').tov(A, '9:50')
    t, info = g.build()
    for team, final in zip((A, H), info['final']):
        assert t.loc[t.offense_team == team, 'points'].sum() + info['other_points'][team] == final
    assert info['final'] == (3, 4)


# ----------------------------------------------------------------------------- real game
GAME_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'Tracking_Data', 'events',
                        '0021500003.csv')   #10.27.2015 NOP at GSW, a dev game


@pytest.mark.skipif(not os.path.exists(GAME_CSV), reason='events CSV not present')
def test_real_dev_game():
    pbp = pd.read_csv(GAME_CSV)
    t, info = build_possessions(pbp, game_id=21500003)
    est = box_estimate(pbp)
    estimate = sum(v['est'] for v in est.values())
    assert abs(len(t) / estimate - 1) < 0.05
    for team, final in zip((info['away'], info['home']), info['final']):
        assert t.loc[t.offense_team == team, 'points'].sum() + info['other_points'][team] == final
    assert info['final'] == (95, 111)                       #NOP 95 - GSW 111
    assert (t.end_clock <= t.start_clock).all() and t.start_elapsed.is_monotonic_increasing
    assert abs((t.offense_team == info['away']).sum() - (t.offense_team == info['home']).sum()) <= 4
    assert info['diag']['score_team_mismatch'] == 0 and info['diag']['unexplained_score_rows'] == 0
