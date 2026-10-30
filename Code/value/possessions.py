"""
Possession table built from the play-by-play (pbp) alone.

One row per possession: who had the ball, the clock at its start and end, how it
ended, the points the offense scored on it, and the first/last pbp EVENTNUM.

Rules (all of them are encoded in `build_possessions`; tests/test_possessions.py
has one synthetic sequence per rule)
-----
Event order     File (row) order is used. It is clock-monotone in every game, and
                breaks clock ties causally (a rebound sits before the putback that
                follows it). EVENTNUM order is NOT used: it is out of clock order.
Points          Taken from the SCORE column ("VISITOR - HOME"), as the per-event
                change in each team's total, not parsed from descriptions. That makes
                3-pointers, and-ones and goaltending come out right. Points that
                cannot be tied to a possession (technical free throws, a score
                change on an event that is not a shot or free throw, a score change
                for the team that is not the event's team) go into `other_points`,
                so  possession points + other points = final score.
Team            Player events: PLAYER1_TEAM_ID. Team events (team rebound, shot clock
                turnover): PLAYER1_ID is the team id and PLAYER1_TEAM_ID is empty.
                Jump ball: PLAYER3_TEAM_ID (the player the ball is tipped to).
Opening         A possession is opened lazily by the first event that gives the ball
                to a team: a shot, a turnover, a (non-dead-ball) free throw or a
                defensive rebound. Its start clock is the end clock of the previous
                possession in the same period, or the period length (720 s, 300 s in
                overtime) for the first one. So the offense of the first possession in
                periods 2-4 comes from the first event with an offensive meaning, not
                from a fouling team.
Ending          made field goal; defensive rebound; any turnover (type 5, including
                team turnovers); a made final free throw of a trip; a jump ball won by
                the other team; end of period.
And-one         A made field goal closes the possession. If the next possession-
                relevant event (fouls, subs, timeouts, violations in between are
                ignored) is a free throw by the same team, the possession is re-opened
                and the free throws belong to it.
Missed shot     Waits for a rebound. Offensive rebound (same team): the possession
                continues (`oreb` +1). Defensive rebound: it ends and the rebounder's
                possession starts. A free throw by the shooting team cancels the wait
                (shooting foul on a missed shot).
Free throws     "k of n": the last one decides (made: ends; missed: wait for a
                rebound). A rebound recorded right after a missed non-final free
                throw is a dead-ball placeholder and is dropped (not an OREB, not a
                change of possession). Technical free throws (action 16) never open or
                close a possession; their points go to `other_points`. Flagrant and
                clear-path free throws (actions 18-20, 25-26; the fouled team keeps
                the ball) never close a possession, and a missed one is followed by a
                dead-ball placeholder rebound. If the shooter's team has no possession
                open, one is opened for it (it gets the ball), so the points belong to
                the possession that follows. A rebound recorded right after a missed
                technical or flagrant free throw, or right after a jump ball, is also
                a dead-ball placeholder.
Inconsistent    If an event says team T has the ball but the open possession belongs
pbp             to the other team (a missing rebound or turnover event), the open one
                is closed with reason `inferred_change` and one for T is opened.
                These are counted in `diag['inferred_changes']`.
Jump ball       Held-ball jump ball: if the tip goes to the other team the possession
                ends (`jump_ball`); if it goes to the offense it continues.
"""
import re

import numpy as np
import pandas as pd

TEAM_ID_MIN, TEAM_ID_MAX = 1610612737, 1610612766
SCORE_RE = re.compile(r'(\d+)\s*-\s*(\d+)')
FT_RE = re.compile(r'(\d+)\s+of\s+(\d+)')
MISS_RE = re.compile(r'\bMISS\b')
# Event types that carry no possession information and never reset the and-one /
# dead-ball-rebound bookkeeping: fouls, violations, subs, timeouts, replay, ejections.
IGNORABLE = {6, 7, 8, 9, 11, 18, 20}


def period_length(period):
    return 720 if period <= 4 else 300


def elapsed(period, clock_left):
    """Seconds since tip-off, with 300 s overtimes (the tracking `game_time` assumes 720)."""
    if period <= 4:
        return (period - 1) * 720 + 720 - clock_left
    return 4 * 720 + (period - 5) * 300 + 300 - clock_left


def clock_seconds(s):
    m, sec = str(s).split(':')[:2]
    return int(m) * 60 + int(float(sec))


def _as_int(x):
    return None if pd.isna(x) else int(x)


def _event_team(row):
    """Team id of an event: the player's team, or PLAYER1_ID itself for team events."""
    p1 = _as_int(row['PLAYER1_ID'])
    if p1 is not None and TEAM_ID_MIN <= p1 <= TEAM_ID_MAX:
        return p1
    return _as_int(row['PLAYER1_TEAM_ID'])


def _desc(row):
    return ' '.join(str(row.get(c)) for c in ('HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION')
                    if pd.notna(row.get(c)))


def prepare_events(pbp):
    """Normalise the pbp rows (kept in file order) into plain Python records."""
    scores = pbp['SCORE'].astype('string').str.extract(SCORE_RE)
    away = pd.to_numeric(scores[0], errors='coerce').ffill().fillna(0).astype(int).values
    home = pd.to_numeric(scores[1], errors='coerce').ffill().fillna(0).astype(int).values
    d_away = np.diff(away, prepend=0)
    d_home = np.diff(home, prepend=0)
    evs = []
    for i, row in enumerate(pbp.to_dict('records')):
        period = int(row['PERIOD'])
        desc = _desc(row)
        etype = int(row['EVENTMSGTYPE'])
        action = int(row['EVENTMSGACTIONTYPE']) if pd.notna(row['EVENTMSGACTIONTYPE']) else 0
        team = _as_int(row.get('PLAYER3_TEAM_ID')) if etype == 10 else _event_team(row)
        evs.append(dict(i=i, num=int(row['EVENTNUM']), period=period, clock=clock_seconds(row['PCTIMESTRING']),
                        etype=etype, action=action, team=team, desc=desc,
                        d_away=int(d_away[i]), d_home=int(d_home[i]),
                        away=int(away[i]), home=int(home[i]),
                        has_score=pd.notna(row.get('SCORE')),
                        p1=_as_int(row.get('PLAYER1_ID')), p2=_as_int(row.get('PLAYER2_ID'))))
    return evs


def infer_home_away(evs):
    """(away_id, home_id) from which score component moves when a team scores."""
    votes = {}
    for e in evs:
        if e['etype'] in (1, 3) and e['team'] is not None:
            if e['d_away'] > 0 and e['d_home'] == 0:
                votes[(e['team'], 'away')] = votes.get((e['team'], 'away'), 0) + 1
            elif e['d_home'] > 0 and e['d_away'] == 0:
                votes[(e['team'], 'home')] = votes.get((e['team'], 'home'), 0) + 1
    teams = {e['team'] for e in evs if e['team'] is not None}
    away = max((t for t in teams), key=lambda t: votes.get((t, 'away'), 0), default=None)
    home = max((t for t in teams if t != away), key=lambda t: votes.get((t, 'home'), 0), default=None)
    return away, home


def free_throw_kind(e):
    """('technical'|'bonus'|'regular', k, n)."""
    d = e['desc']
    m = FT_RE.search(d)
    k, n = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
    if e['action'] == 16 or 'Technical' in d:
        return 'technical', k, n
    if e['action'] in (18, 19, 20, 25, 26) or 'Flagrant' in d or 'Clear Path' in d:
        return 'bonus', k, n
    return 'regular', k, n


def build_possessions(pbp, game_id=None):
    """Build the possession table for one game.

    Returns (table, info). `table` has one row per possession. `info` has `away`/`home`
    team ids, `final` (away, home), `other_points` {team: pts}, and `diag` counters."""
    evs = prepare_events(pbp)
    away_id, home_id = infer_home_away(evs)
    if away_id is None or home_id is None:
        raise ValueError('cannot tell the two teams apart (no scoring events?)')
    other = {away_id: 0, home_id: 0}
    notes = []   #(kind, eventnum) of every anomaly, for reading the raw pbp around it
    diag = dict(other_tech_points=0, other_misc_points=0, inferred_changes=0, dead_ball_rebounds=0, stray_rebounds=0, jump_ball_changes=0,
                and_one_reopens=0, score_team_mismatch=0, unexplained_score_rows=0,
                made_without_score=0, ft_made_without_score=0, ft_disorder=0)
    rows, closed_ids = [], []
    S = dict(cur=None, awaiting=False, dead_reb=False, last_closed=None, prev_end=None,
             period=None, away=0, home=0)

    def score_of(team):
        return S['away'] if team == away_id else S['home']

    def other_team(team):
        return home_id if team == away_id else away_id

    def start(team, e):
        prev = S['prev_end']
        start_clock = prev[1] if prev is not None and prev[0] == e['period'] else period_length(e['period'])
        mine, theirs = score_of(team), score_of(other_team(team))
        S['cur'] = dict(game_id=game_id, period=e['period'], offense_team=team, defense_team=other_team(team),
                        start_clock=start_clock, first_event_num=e['num'], points=0, fga=0, fta=0, oreb=0,
                        off_margin_start=mine - theirs, last_event_num=e['num'])
        S['awaiting'] = False
        S['last_closed'] = None

    def close(reason, e, by_made_fg=False):
        c = S['cur']
        c['end_clock'], c['end_reason'], c['last_event_num'] = e['clock'], reason, e['num']
        rows.append(c)
        S['prev_end'] = (e['period'], e['clock'])
        S['last_closed'] = c if by_made_fg else None
        S['cur'], S['awaiting'] = None, False

    def ensure_offense(team, e):
        if S['cur'] is not None and S['cur']['offense_team'] != team:
            close('inferred_change', e)
            diag['inferred_changes'] += 1
            notes.append(('inferred_change', e['num']))
        if S['cur'] is None:
            start(team, e)

    def credit(e, team, pts):
        """Give `pts` scored by `team` on this event to the open possession if it is
        that team's, else to other_points."""
        if pts <= 0:
            return
        c = S['cur']
        if c is not None and c['offense_team'] == team:
            c['points'] += pts
        else:
            other[team] += pts
            diag['other_misc_points'] += pts
            notes.append(('points_to_other', e['num']))

    last_ft = None
    for e in evs:
        # keep running scores current (used for the score margin at possession starts)
        if S['period'] is not None and e['period'] != S['period'] and S['cur'] is not None:
            close('end_of_period', dict(e, clock=0, period=S['period']))   #period-end event missing
        if e['period'] != S['period']:
            S['period'], S['prev_end'], S['last_closed'], S['dead_reb'] = e['period'], None, None, False
        etype, team = e['etype'], e['team']
        d = {away_id: e['d_away'], home_id: e['d_home']}

        # points not attributable to a shot or a free throw by the event's team
        explained = {away_id: 0, home_id: 0}

        if etype == 12:
            if S['cur'] is not None:
                close('end_of_period', e)
            S['prev_end'], S['last_closed'], S['dead_reb'] = None, None, False
        elif etype == 13:
            if S['cur'] is not None:
                close('end_of_period', e)
            S['last_closed'], S['dead_reb'] = None, False
        elif etype == 10:
            S['last_closed'], S['dead_reb'] = None, True    #a team rebound recorded after a jump ball is a placeholder
            if S['cur'] is not None and team is not None:
                if team != S['cur']['offense_team']:
                    close('jump_ball', e)
                    diag['jump_ball_changes'] += 1
                S['awaiting'] = False
        elif etype in (1, 2) and team is not None:
            S['last_closed'], S['dead_reb'] = None, False
            ensure_offense(team, e)
            c = S['cur']
            c['fga'] += 1
            c['last_event_num'] = e['num']
            S['awaiting'] = False
            made = etype == 1
            if made:
                if d[team] <= 0:
                    diag['made_without_score'] += 1
                credit(e, team, d[team])
                explained[team] = d[team]
                close('made_fg', e, by_made_fg=True)
            else:
                S['awaiting'] = True
        elif etype == 3 and team is not None:
            kind, k, n = free_throw_kind(e)
            made_ft = MISS_RE.search(e['desc']) is None
            if made_ft and d[team] <= 0 and kind != 'technical':
                diag['ft_made_without_score'] += 1
            if last_ft is not None and last_ft[0] == e['period'] and last_ft[1] == e['clock'] and last_ft[2] == n and k < last_ft[3]:
                diag['ft_disorder'] += 1
            last_ft = (e['period'], e['clock'], n, k)
            if kind == 'technical':
                other[team] += d[team]
                diag['other_tech_points'] += d[team]
                explained[team] = d[team]
                if not made_ft:
                    S['dead_reb'] = True       #a rebound recorded after a missed technical is a placeholder
            elif kind == 'bonus':
                S['last_closed'] = None
                ensure_offense(team, e)        #the fouled team keeps (or gets) the ball: these free throws
                S['cur']['last_event_num'] = e['num']   #belong to its possession, which carries on afterwards
                S['cur']['fta'] += 1
                credit(e, team, d[team])
                explained[team] = d[team]
                S['dead_reb'] = not made_ft
            else:
                reopened = False
                if S['cur'] is None and S['last_closed'] is not None and S['last_closed']['offense_team'] == team:
                    c = S['last_closed']          #and-one: the free throw belongs to the made field goal
                    assert rows[-1] is c
                    rows.pop()
                    S['cur'], S['last_closed'] = c, None
                    S['cur'].pop('end_clock', None)
                    S['cur'].pop('end_reason', None)
                    diag['and_one_reopens'] += 1
                    reopened = True
                S['last_closed'] = None
                if not reopened:
                    ensure_offense(team, e)
                c = S['cur']
                S['awaiting'] = False
                c['fta'] += 1
                c['last_event_num'] = e['num']
                credit(e, team, d[team])
                explained[team] = d[team]
                S['dead_reb'] = False
                if k >= n:
                    if made_ft:
                        close('made_ft', e)
                    else:
                        S['awaiting'] = True
                elif not made_ft:
                    S['dead_reb'] = True     #the next rebound, if any, is a dead-ball placeholder
        elif etype == 4 and team is not None:
            S['last_closed'] = None
            if S['dead_reb']:
                diag['dead_ball_rebounds'] += 1
                S['dead_reb'] = False
            elif S['cur'] is not None and S['awaiting']:
                if team == S['cur']['offense_team']:
                    S['cur']['oreb'] += 1
                    S['cur']['last_event_num'] = e['num']
                    S['awaiting'] = False
                else:
                    close('def_rebound', e)
                    start(team, e)
            elif S['cur'] is None:
                diag['stray_rebounds'] += 1
                notes.append(('stray_rebound_no_possession', e['num']))
                start(team, e)
            elif team != S['cur']['offense_team']:
                diag['stray_rebounds'] += 1
                notes.append(('stray_rebound_other_team', e['num']))
                close('inferred_change', e)
                diag['inferred_changes'] += 1
                start(team, e)
            else:
                diag['stray_rebounds'] += 1
                notes.append(('stray_rebound_same_team', e['num']))
        elif etype == 5 and team is not None:
            S['last_closed'], S['dead_reb'] = None, False
            ensure_offense(team, e)
            close('turnover', e)
        # else: ignorable event (foul, sub, timeout, violation, replay ...)

        # points that moved the score but were not credited above
        for t in (away_id, home_id):
            extra = d[t] - explained[t]
            if extra > 0:
                if etype in (1, 3) and team is not None and t != team:
                    diag['score_team_mismatch'] += extra
                else:
                    diag['unexplained_score_rows'] += 1
                diag['other_misc_points'] += extra
                notes.append(('unexplained_score', e['num']))
                other[t] += extra
        S['away'], S['home'] = e['away'], e['home']

    if S['cur'] is not None:
        last = evs[-1]
        close('end_of_data', dict(last, clock=0) if last['etype'] != 13 else last)

    table = pd.DataFrame(rows)
    if len(table):
        table.insert(1, 'poss_idx', np.arange(len(table)))   #rows are appended in closing order = time order
        table['start_elapsed'] = [elapsed(p, c) for p, c in zip(table.period, table.start_clock)]
        table['end_elapsed'] = [elapsed(p, c) for p, c in zip(table.period, table.end_clock)]
        table = table[['game_id', 'poss_idx', 'period', 'offense_team', 'defense_team', 'start_clock', 'end_clock',
                       'start_elapsed', 'end_elapsed', 'end_reason', 'points', 'first_event_num', 'last_event_num',
                       'fga', 'fta', 'oreb', 'off_margin_start']]
    final = (evs[-1]['away'], evs[-1]['home'])
    info = dict(away=away_id, home=home_id, final=final, other_points=other, diag=diag, notes=notes,
                periods=int(pbp['PERIOD'].max()))
    return table, info


def box_estimate(pbp):
    """Independent possession estimate per team, FGA + 0.44*FTA - OREB + TOV, from raw
    event counts only (not the state machine). OREB: a rebound by the team of the most
    recent shot or free throw, excluding a rebound right after a missed non-final free
    throw (dead-ball placeholder). Returns {team: dict(fga, fta, fta_ex_tech, oreb, tov, est)}."""
    evs = prepare_events(pbp)
    out = {}

    def row(t):
        return out.setdefault(t, dict(fga=0, fta=0, fta_ex_tech=0, oreb=0, tov=0))
    last_shot_team, dead = None, False
    for e in evs:
        t = e['team']
        if t is None:
            continue
        if e['etype'] in (1, 2):
            row(t)['fga'] += 1
            last_shot_team, dead = t, False
        elif e['etype'] == 3:
            kind, k, n = free_throw_kind(e)
            row(t)['fta'] += 1
            if kind != 'technical':
                row(t)['fta_ex_tech'] += 1
                last_shot_team = t
                dead = (k < n) and MISS_RE.search(e['desc']) is not None
        elif e['etype'] == 4:
            if dead:
                dead = False
            elif last_shot_team is not None and t == last_shot_team:
                row(t)['oreb'] += 1
            dead = False
        elif e['etype'] == 5:
            row(t)['tov'] += 1
            last_shot_team, dead = None, False
    for t, r in out.items():
        r['est'] = r['fga'] + 0.44 * r['fta'] - r['oreb'] + r['tov']
        r['est_ex_tech'] = r['fga'] + 0.44 * r['fta_ex_tech'] - r['oreb'] + r['tov']
    return out
