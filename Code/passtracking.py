import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
from getting_data import get_data
from tqdm import tqdm

"""
Pass detection and pass-geometry features.

Detection (PassTracking.track_passes) is a ball-holder state machine run over
every frame (the shot clock is a feature, not a gate):

  * a player "holds" the ball when he is the nearest player to it, within
    HOLD_DIST ft, with the ball no higher than HOLD_Z ft;
  * a hold must last MIN_HOLD_FRAMES frames; dropouts of up to MAX_FILL_FRAMES
    frames inside a hold are filled; holds never span a time gap;
  * a COMPLETED pass is a hold by player A followed (within MAX_FLIGHT_S) by a
    hold of a different teammate B, with the ball travelling >= MIN_DISP ft,
    staying in bounds, and not being a shot (high arc ending near a rim);
  * a FAILED pass is a play-by-play bad-pass turnover (EVENTMSGTYPE 5, action
    type 1 or 45) matched to the committer's last hold before the event, whose
    flight ends in an opponent's hold (interception) or out of bounds.
    Lost balls, travels, fouls and shot-clock turnovers are not passes.

Every feature is computed at the RELEASE frame (the last frame of the passer's
hold), identically for both classes. The target of a completed pass is the
receiver; the target of a failed pass is the teammate the ball was heading
towards (`target_heading`, from the ball's first HEAD_FRAMES frames of flight).
The heading rule is also run on completed passes so its accuracy is known.
"""

np.seterr(divide='ignore', invalid='ignore')


class PassTracking:

    FPS = 25
    HOLD_DIST = 3.0          # ft, nearest player to ball within this = holder
    HOLD_Z = 8.0             # ft, ball above this is in the air, not held
    HOLD_REL_SPEED = 15.0    # ft/s, ball speed relative to the holder (drops passes flying past a player)
    MIN_HOLD_FRAMES = 4
    MAX_FILL_FRAMES = 6
    MAX_FRAME_DT = 0.12      # s, larger step between frames = time gap
    MIN_DISP = 5.0           # ft, ball displacement release -> reception
    MIN_FLIGHT_S = 0.2
    MAX_FLIGHT_S = 4.0
    SHOT_Z = 10.5            # ft, flight arc this high and ending near a rim = shot
    SHOT_RIM_DIST = 20.0
    OOB_MARGIN = 1.5
    OOB_SKIP_FRAMES = 4      # inbound passes start out of bounds
    FAIL_ACTION_TYPES = (1, 45)   # bad pass, out-of-bounds bad pass
    FAIL_WINDOW = (-3.0, 8.0)     # s, release clock minus pbp clock (pbp is often seconds off)
    FAIL_TYPICAL_LAG = 1.5        # s, tie-break between candidate releases
    FUMBLE_MAX_FRAMES = 15        # a teammate touch this short, then lost, is a fumbled pass
    FUMBLE_FLIGHT_FRAMES = 37
    HEAD_FRAMES = 8
    HEAD_MIN_MOVE = 1.5
    HEAD_MIN_MATE_DIST = 3.0
    HEAD_MAX_ANGLE = 50.0    # degrees
    ATTACK_VOTE_DIST = 30.0

    def __init__(self, date, home, away):
        self.home = home
        self.away = away
        data = get_data(date, home, away)
        self.home_id = data.home_id
        self.away_id = data.away_id
        self.game_id = data.game_id
        self.basket1 = [5.35, 25]
        self.basket2 = [88.65, 25]
        self.pass_attempt = None
        self.attack_basket = {}
        self.failure_report = {}
        self.moments = data.moments
        self.pbp = data.pbp

    def _dist_calculate(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    # ------------------------------------------------------------------ frames
    def _build_frames(self):
        """Per-frame numpy arrays. Players are matched by id, never by slot."""
        m = self.moments
        n = len(m)
        ball = np.full((n, 3), np.nan)
        pid = np.zeros((n, 10), dtype=np.int64)
        team = np.zeros((n, 10), dtype=np.int64)
        xy = np.full((n, 10, 2), np.nan)
        has_ball = np.zeros(n, dtype=bool)
        for i, p in enumerate(m['positions'].values):
            if len(p) != 11 or p[0][0] != -1:
                continue   #Frames without a ball (or an odd roster) are unusable.
            a = np.asarray(p, dtype=float)
            ball[i] = a[0, 2:5]
            team[i] = a[1:, 0]
            pid[i] = a[1:, 1]
            xy[i] = a[1:, 2:4]
            has_ball[i] = True

        ut = m['universe_time'].values / 1000.0
        quarter = m['quarter'].values
        dt = np.diff(ut, prepend=-np.inf)
        cont = (np.diff(quarter, prepend=-1) == 0) & (dt > 0) & (dt <= self.MAX_FRAME_DT)
        block = np.cumsum(~cont)   #Frames in one block are gap-free.
        block_last = np.append(np.flatnonzero(~cont)[1:] - 1, n - 1)

        d = np.hypot(xy[:, :, 0] - ball[:, None, 0], xy[:, :, 1] - ball[:, None, 1])
        d = np.where(np.isnan(d), np.inf, d)
        rows = np.arange(n)
        near = d.argmin(axis=1)

        #Ball speed relative to the nearest player (central difference over +-K
        #frames, players matched by slot only if the same id holds that slot).
        #A ball that is really held moves with its holder; a pass flying within
        #HOLD_DIST of a player does not. Unknown (edges, gaps) is not penalised.
        K = 2
        rel = np.full(n, np.nan)
        if n > 2 * K:
            lo, hi = np.arange(0, n - 2 * K), np.arange(2 * K, n)
            mid = lo + K
            ok = (block[lo] == block[hi]) & has_ball[lo] & has_ball[hi] & has_ball[mid]
            dtk = ut[hi] - ut[lo]
            slot = near[mid]
            same = (pid[lo, slot] == pid[mid, slot]) & (pid[hi, slot] == pid[mid, slot])
            vb = (ball[hi, :2] - ball[lo, :2]) / dtk[:, None]
            vp = (xy[hi, slot] - xy[lo, slot]) / dtk[:, None]
            r_ = np.hypot(*(vb - vp).T)
            rel[mid] = np.where(ok & same, r_, np.nan)
        slow = np.isnan(rel) | (rel <= self.HOLD_REL_SPEED)

        held = (has_ball & (d[rows, near] <= self.HOLD_DIST) & (ball[:, 2] <= self.HOLD_Z) & slow)
        holder = np.where(held, pid[rows, near], 0)
        holder_team = np.where(held, team[rows, near], 0)

        self._fr = dict(n=n, ball=ball, pid=pid, team=team, xy=xy, has_ball=has_ball,
                        ut=ut, quarter=quarter, qt=m['quarter_time'].values,
                        cont=cont, block=block, block_last=block_last,
                        holder=holder, holder_team=holder_team)

    def _out_of_bounds(self, idx):
        b = self._fr['ball'][idx]
        mg = self.OOB_MARGIN
        return (b[:, 0] < -mg) | (b[:, 0] > 94 + mg) | (b[:, 1] < -mg) | (b[:, 1] > 50 + mg)

    # ----------------------------------------------------------- hold segments
    def _hold_segments(self):
        fr = self._fr
        n, holder, block = fr['n'], fr['holder'], fr['block']
        change = np.ones(n, dtype=bool)
        change[1:] = (holder[1:] != holder[:-1]) | (block[1:] != block[:-1])
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], n) - 1
        segs = []
        for s, e in zip(starts, ends):
            h = holder[s]
            if h == 0 or e - s + 1 < self.MIN_HOLD_FRAMES:
                continue
            last = segs[-1] if segs else None
            if (last is not None and last['holder'] == h and last['block'] == block[s]
                    and s - last['end'] - 1 <= self.MAX_FILL_FRAMES):
                last['end'] = int(e)   #Fill a short dropout inside one hold.
            else:
                segs.append(dict(holder=int(h), team=int(fr['holder_team'][s]),
                                 block=int(block[s]), start=int(s), end=int(e)))
        return segs

    def _compute_attack_baskets(self, segs):
        """Which basket each team attacks in each quarter, voted from where the
        team holds the ball near a rim. Teams swap baskets at halftime."""
        fr = self._fr
        votes = defaultdict(lambda: [0, 0])
        for s in segs:
            b = fr['ball'][s['start']:s['end'] + 1]
            k = (s['team'], int(fr['quarter'][s['start']]))
            votes[k][0] += int((np.hypot(b[:, 0] - self.basket1[0], b[:, 1] - self.basket1[1]) <= self.ATTACK_VOTE_DIST).sum())
            votes[k][1] += int((np.hypot(b[:, 0] - self.basket2[0], b[:, 1] - self.basket2[1]) <= self.ATTACK_VOTE_DIST).sum())
        attack = {}
        for k, (v1, v2) in votes.items():
            if v1 != v2:
                attack[k] = 1 if v1 > v2 else 2
        for (t, q), side in list(attack.items()):
            #A team with no votes in a quarter attacks the opposite of the other team.
            for t2 in (self.home_id, self.away_id):
                if t2 != t and (t2, q) not in attack:
                    attack[(t2, q)] = 2 if side == 1 else 1
        self.attack_basket = attack

    def _attack_basket(self, team, quarter):
        side = self.attack_basket.get((int(team), int(quarter)))
        if side is None:
            return None
        return self.basket1 if side == 1 else self.basket2

    # --------------------------------------------------------- target heading
    def _heading_target(self, r, end, passer, team):
        """Teammate the ball is heading towards over its first frames of flight."""
        fr = self._fr
        e = min(r + self.HEAD_FRAMES, end)
        while e >= r + 3 and not fr['has_ball'][e]:
            e -= 1
        if e < r + 3 or not fr['has_ball'][r]:
            return None
        h = fr['ball'][e, :2] - fr['ball'][r, :2]
        hn = np.hypot(*h)
        if hn < self.HEAD_MIN_MOVE:
            return None
        mates = np.flatnonzero((fr['team'][r] == team) & (fr['pid'][r] != passer) & (fr['pid'][r] != 0))
        if len(mates) == 0:
            return None
        v = fr['xy'][r][mates] - fr['ball'][r, :2]
        vn = np.hypot(v[:, 0], v[:, 1])
        ang = np.degrees(np.arccos(np.clip((v @ h) / (vn * hn), -1, 1)))
        ang = np.where(vn >= self.HEAD_MIN_MATE_DIST, ang, np.inf)
        best = int(np.nanargmin(ang)) if np.isfinite(ang).any() else None
        if best is None or ang[best] > self.HEAD_MAX_ANGLE:
            return None
        return int(fr['pid'][r][mates[best]])

    # ------------------------------------------------------------------ passes
    def _row(self, r, recv, end, passer, team, target, success, outcome):
        fr, m = self._fr, self.moments
        basket = self._attack_basket(team, fr['quarter'][r])
        passer_x = fr['xy'][r][fr['pid'][r] == passer][:, 0]
        if basket is None or len(passer_x) == 0:
            backcourt = np.nan
        else:
            backcourt = int(abs(passer_x[0] - basket[0]) > 47)
        heading = self._heading_target(r, end, passer, team)
        if success:
            tgt = target
        else:
            tgt = heading
        team_target = None
        if tgt is not None:
            t_team = fr['team'][r][fr['pid'][r] == tgt]
            team_target = int(t_team[0]) if len(t_team) else None
        return dict(index=r, frame=r, recv_frame=recv, end_frame=end,
                    quarter=int(fr['quarter'][r]), universe_time=int(m['universe_time'].iat[r]),
                    quarter_time=float(fr['qt'][r]), shot_clock=m['shot_clock'].iat[r],
                    positions=m['positions'].iat[r], game_time=m['game_time'].iat[r],
                    pass_successful=int(success), passer=passer, target=tgt,
                    team_passer=team, team_target=team_target, backcourt=backcourt,
                    target_inferred=not success, target_heading=heading,
                    outcome=outcome, flight_s=float(fr['ut'][end] - fr['ut'][r]),
                    basket_x=None if basket is None else basket[0])

    def _completed_passes(self, segs):
        fr = self._fr
        rows = []
        for a, b in zip(segs[:-1], segs[1:]):
            if a['block'] != b['block'] or a['team'] != b['team'] or a['holder'] == b['holder']:
                continue
            r, recv = a['end'], b['start']
            flight_s = fr['ut'][recv] - fr['ut'][r]
            if not (self.MIN_FLIGHT_S <= flight_s <= self.MAX_FLIGHT_S):
                continue
            if np.hypot(*(fr['ball'][recv, :2] - fr['ball'][r, :2])) < self.MIN_DISP:
                continue
            flight = np.arange(r + 1, recv)
            if len(flight) == 0:
                continue
            if self._out_of_bounds(flight[self.OOB_SKIP_FRAMES:]).any():
                continue
            zmax = np.nanmax(fr['ball'][flight, 2]) if np.isfinite(fr['ball'][flight, 2]).any() else 0.0
            end_xy = fr['ball'][recv, :2]
            near_rim = min(np.hypot(*(end_xy - self.basket1)), np.hypot(*(end_xy - self.basket2))) <= self.SHOT_RIM_DIST
            if zmax >= self.SHOT_Z and near_rim:
                continue   #A shot (and offensive rebound), not a pass.
            rows.append(self._row(r, recv, recv, a['holder'], a['team'], b['holder'], True, 'completed'))
        return rows

    def _flight_outcome(self, segs, i, min_flight=2):
        """How the flight after hold segs[i] ends: (end_frame, outcome, next
        segment) for an interception or out-of-bounds, else (None, reason, next)."""
        fr = self._fr
        s = segs[i]
        r = s['end']
        limit = min(r + int(self.MAX_FLIGHT_S * self.FPS), int(fr['block_last'][s['block'] - 1]))
        nxt = segs[i + 1] if i + 1 < len(segs) and segs[i + 1]['block'] == s['block'] else None
        if nxt is not None and nxt['start'] <= limit:
            if nxt['team'] == s['team']:
                #A brief teammate touch (fumbled catch, deflection) that is then
                #lost to an opponent or out of bounds is still a failed pass.
                if nxt['end'] - nxt['start'] + 1 <= self.FUMBLE_MAX_FRAMES:
                    end, outcome, nxt2 = self._flight_outcome(segs, i + 1, min_flight=1)
                    if end is not None and end - nxt['end'] <= self.FUMBLE_FLIGHT_FRAMES:
                        return end, 'fumbled', nxt2
                return None, 'teammate_holds_next', nxt
            end, outcome = nxt['start'], 'intercepted'
        else:
            window = np.arange(r + 1, limit + 1)[self.OOB_SKIP_FRAMES:]
            oob = np.flatnonzero(self._out_of_bounds(window)) if len(window) else []
            if len(oob) == 0:
                return None, 'no_interception_or_oob', nxt
            end, outcome = int(window[oob[0]]), 'out_of_bounds'
        if end - r < min_flight:
            return None, 'flight_too_short', nxt
        return end, outcome, nxt

    def _failed_passes(self, segs):
        """Match pbp bad-pass turnovers to the committer's flight. The pbp clock
        can be a few seconds off the tracking clock, so the window is wide and a
        candidate whose next holder is the credited stealer (PLAYER2_ID) wins."""
        fr = self._fr
        pbp = self.pbp
        events = pbp.loc[(pbp['EVENTMSGTYPE'] == 5) & (pbp['EVENTMSGACTIONTYPE'].isin(self.FAIL_ACTION_TYPES))]
        by_holder = defaultdict(list)
        for i, s in enumerate(segs):
            by_holder[s['holder']].append(i)
        rows, unmatched, used = [], defaultdict(int), set()
        for _, ev in events.iterrows():
            committer, period = int(ev['PLAYER1_ID']), int(ev['PERIOD'])
            stealer = int(ev['PLAYER2_ID']) if pd.notna(ev['PLAYER2_ID']) else 0
            e = float(ev['Qtime'])
            cands = [i for i in by_holder[committer]
                     if fr['quarter'][segs[i]['end']] == period and i not in used
                     and self.FAIL_WINDOW[0] <= fr['qt'][segs[i]['end']] - e <= self.FAIL_WINDOW[1]]
            if not cands:
                unmatched['no_hold_by_committer'] += 1
                continue
            scored, reasons = [], []
            for i in cands:
                end, outcome, nxt = self._flight_outcome(segs, i)
                if end is None:
                    reasons.append(outcome)
                    continue
                confirmed = outcome == 'intercepted' and stealer != 0 and nxt['holder'] == stealer
                scored.append((not confirmed, abs(fr['qt'][segs[i]['end']] - e - self.FAIL_TYPICAL_LAG), i, end, outcome))
            if not scored:
                unmatched[sorted(reasons)[0]] += 1
                continue
            _, _, i, end, outcome = min(scored)
            used.add(i)
            rows.append(self._row(segs[i]['end'], end, end, committer, segs[i]['team'], None, False, outcome))
        self.failure_report = dict(pbp_events=len(events), matched=len(rows), unmatched=dict(unmatched))
        return rows

    def track_passes(self):
        #Quarter-clock seconds for the pbp events (kept as columns, as before).
        self.pbp['Qmin'] = (self.pbp['PCTIMESTRING'].str.split(':', expand=True)[0]).astype(int)
        self.pbp['Qsec'] = (self.pbp['PCTIMESTRING'].str.split(':', expand=True)[1]).astype(int)
        self.pbp['Qtime'] = (self.pbp['Qmin'].astype(int) * 60 + self.pbp['Qsec'].astype(int))

        self._build_frames()
        segs = self._hold_segments()
        self.segments = segs
        self._compute_attack_baskets(segs)
        failed = self._failed_passes(segs)
        failed_frames = {row['frame'] for row in failed}
        #A pbp-confirmed failure overrides a 'completed' pass from the same release
        #(e.g. the receiver touched it for a few frames and lost it).
        rows = [row for row in self._completed_passes(segs) if row['frame'] not in failed_frames] + failed
        rows.sort(key=lambda row: row['frame'])

        for col in ['pass_successful', 'passer', 'target', 'team_passer', 'team_target', 'backcourt']:
            self.moments[col] = None
        self.pass_attempt = pd.DataFrame(rows)
        for row in rows:
            for col in ['pass_successful', 'passer', 'target', 'team_passer', 'team_target', 'backcourt']:
                self.moments.iat[row['frame'], self.moments.columns.get_loc(col)] = row[col]


class features:

    def __init__(self, date, home, away, target_mode='actual'):
        """target_mode: 'actual' uses the real receiver of completed passes
        (and the heading-inferred teammate for failed ones); 'heading' uses the
        heading-inferred teammate for both classes."""
        self.passes = PassTracking(date, home, away)
        self.passes.track_passes()
        self.target_mode = target_mode
        self.features = pd.DataFrame(self.passes.pass_attempt[['shot_clock', 'pass_successful', 'backcourt']])
        self.meta = None

    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
        """Interior angle (radians, 0..pi) at the vertex (x2, y2) between the
        rays to (x1, y1) and (x3, y3). NaN if either ray has zero length."""
        v1 = np.array([x1 - x2, y1 - y2], dtype=float)
        v2 = np.array([x3 - x2, y3 - y2], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return np.nan
        return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

    def _speed(self, r, player):
        """Speed (ft/s) over the 5 frames before release, matched by player id."""
        fr = self.passes._fr
        r0 = r - 5
        if r0 < 0 or fr['block'][r0] != fr['block'][r]:
            return np.nan
        j0 = np.flatnonzero(fr['pid'][r0] == player)
        j1 = np.flatnonzero(fr['pid'][r] == player)
        if len(j0) == 0 or len(j1) == 0:
            return np.nan
        return float(np.hypot(*(fr['xy'][r][j1[0]] - fr['xy'][r0][j0[0]])) / (fr['ut'][r] - fr['ut'][r0]))

    def features_extraction(self):
        fr = self.passes._fr
        tcol = 'target_heading' if self.target_mode == 'heading' else 'target'
        feats, metas, valid_index = [], [], []
        for index, row in tqdm(self.passes.pass_attempt.iterrows(),
                               total=len(self.passes.pass_attempt),
                               desc="Extracting features"):
            r = int(row['frame'])
            passer, target, team = row['passer'], row[tcol], row['team_passer']
            if pd.isna(target) or pd.isna(row['basket_x']):
                continue
            pid, xy = fr['pid'][r], fr['xy'][r]
            ip, it = np.flatnonzero(pid == passer), np.flatnonzero(pid == target)
            if len(ip) == 0 or len(it) == 0:
                continue
            P, T = xy[ip[0]], xy[it[0]]
            if np.isnan(P).any() or np.isnan(T).any():
                continue
            pass_vec = T - P
            pass_len = np.hypot(*pass_vec)
            if pass_len < 1.0:
                continue   #Passer and target on top of each other: lane geometry undefined.
            opp = (fr['team'][r] != team) & (pid != 0)
            D = xy[opp]
            if len(D) == 0:
                continue
            basket = np.array([row['basket_x'], 25.0])

            #Distance from each defender to the pass SEGMENT (not the infinite line).
            u = np.clip(((D - P) @ pass_vec) / pass_len**2, 0, 1)
            lane_d = np.hypot(*(D - (P + u[:, None] * pass_vec)).T)
            d_passer = np.hypot(*(D - P).T)
            d_target = np.hypot(*(D - T).T)
            dp, dt_, dl = D[d_passer.argmin()], D[d_target.argmin()], D[lane_d.argmin()]

            sc = row['shot_clock']
            if pd.isna(sc) and row['quarter_time'] <= 24:
                sc = row['quarter_time']   #Shot clock is off late in the quarter.

            feats.append(dict(
                shot_clock=sc,
                pass_successful=row['pass_successful'],
                backcourt=row['backcourt'],
                dist=pass_len,
                passer_basket_dist=np.hypot(*(P - basket)),
                ball_end_basket_dist=np.hypot(*(T - basket)),
                basket_angle=self.calculate_angle(P[0], P[1], basket[0], basket[1], T[0], T[1]),
                closest_def_dist_passer=d_passer.min(),
                closest_def_dist_ball_end=d_target.min(),
                closest_def_trajectory=lane_d.min(),
                closest_def_angle_passer=self.calculate_angle(T[0], T[1], P[0], P[1], dp[0], dp[1]),
                closest_def_angle_ball_end=self.calculate_angle(P[0], P[1], T[0], T[1], dt_[0], dt_[1]),
                closest_def_trajectory_angle=self.calculate_angle(T[0], T[1], P[0], P[1], dl[0], dl[1]),
                lane_n_def_5ft=int((lane_d <= 5).sum()),
                passer_speed=self._speed(r, passer),
                target_speed=self._speed(r, target)))
            metas.append(dict(game_id=int(self.passes.game_id), quarter=row['quarter'],
                              game_time=row['game_time'], frame=r, passer=passer, target=target,
                              team_passer=team, target_inferred=bool(row['target_inferred']),
                              target_mode=self.target_mode, outcome=row['outcome']))
            valid_index.append(index)

        self.features = pd.DataFrame(feats, index=valid_index).astype(float)
        self.meta = pd.DataFrame(metas, index=valid_index)


def process_features(date_away_home):
    date, away, home = date_away_home
    out_dir = './../Tracking_Data/pass_features'
    try:
        os.makedirs(out_dir, exist_ok=True)
        feature = features(date, home, away)
        feature.features_extraction()
        feature.features.to_pickle(f'{out_dir}/{date}.{away}.{home}.pkl')
    except Exception:
        print(f"Error processing {date}.{away}.at.{home}:")
        traceback.print_exc()


def read_game_list(path):
    """Parse lines like 'MM.DD.YYYY.AWAY.at.HOME.7z' into (date, away, home)."""
    games = []
    with open(path, 'r') as file:
        for line in file:
            components = line.split('.')
            if len(components) < 6:
                continue
            games.append(('.'.join(components[:3]), components[3], components[5].split()[0]))
    return games


def main():
    """`python passtracking.py` runs the single sample game and writes
    Tracking_Data/passes/features_v2.csv (the committed features.csv is the
    pre-fix original and is left alone). `python passtracking.py <list file>`
    batch-processes the games in <list file> (e.g. ../Shots/allgames.txt) into
    Tracking_Data/pass_features/*.pkl."""
    if len(sys.argv) > 1:
        for game in tqdm(read_game_list(sys.argv[1]), desc="Games"):
            process_features(game)
        return
    feature = features('12.12.2015', 'MIL', 'GSW')
    feature.features_extraction()
    feature.features.to_csv('./../Tracking_Data/passes/features_v2.csv')
    print("CSV file saved and ready to use.")


if __name__ == '__main__':
    main()
