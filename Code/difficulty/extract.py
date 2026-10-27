"""Build the pass-difficulty dataset: every live flight with a TRACKING-ONLY outcome label, plus shot attempts.

Label (no pbp turnover involved): at every hold-end (release) the next thing that happens to the ball decides it:
  1 = a teammate gets it (hold of a different teammate, same gap-free block, <= MAX_FLIGHT_S later);
  0 = an opponent gets it (intercepted), it leaves the court (out_of_bounds), or a teammate touches it
      briefly and it is then lost (fumbled)  -- the detector's own _flight_outcome rules.
Both classes go through the SAME _flight_ok filter. Play-by-play is used ONLY to exclude releases that are
not passes (the releaser's own shot, a rebound / block scramble, the releaser's own lost ball / travel /
offensive foul) and to flag the old pbp bad-pass label (`pbp_badpass`) for cross-checking; the same
exclusions apply to both classes. Dead-ball / inbound releases are flagged (`live`), not dropped.

Features are computed at the release frame for the INTENDED receiver = the heading rule for both classes
(the true receiver is stored as `actual_target` for diagnosis only).

    cd Code && python difficulty/extract.py <game list> --out DIR     (resumable, one pickle per game)
Does NOT modify passtracking.py (subclass), so the existing dataset stays valid.
"""
import argparse
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
import passtracking as P                    # noqa: E402
from passtracking import PassTracking       # noqa: E402
from pass_features import pass_features, shot_features, FEATURE_NAMES   # noqa: E402

SHOT_LAG = (-1.5, 3.5)       # s, release clock minus pbp clock: the releaser's own FGA
REB_LAG = (-3.0, 1.0)        # s: a rebound / block just before the release (loose-ball scramble)
TO_LAG = (-3.0, 8.0)         # s: releaser's own non-pass turnover (lost ball, travel, off. foul, ...)
SHOT_WINDOW = (-3.0, 8.0)    # s: matching a pbp FGA to the shooter's release (same window as bad passes)
SHOT_TYPICAL_LAG = 1.5


class DiffPT(PassTracking):

    # ---------------------------------------------------------- pbp context
    # For each class of pbp event, an integer mask: bit k is set if such an event happened at lag in [-5 + k, -4 + k) s,
    # lag = release clock - event clock (negative: the event is BEFORE the release; positive: after), k = 0..12 (-5 s .. +8 s).
    # Exclusion windows are chosen at analysis time (train_eval.py) so they can be tuned on the tuning games without
    # re-extracting.
    LAG_MIN, LAG_MAX = -5, 8
    DEAD_TYPES = (1, 3, 6, 7, 9, 10)        # made FG, free throw, foul, violation, timeout, jump ball: ball is (about to be) dead

    def _pbp_index(self):
        p = self.pbp
        txt = (p['HOMEDESCRIPTION'].fillna('') + p['VISITORDESCRIPTION'].fillna('') + p['NEUTRALDESCRIPTION'].fillna(''))
        self._ev = {
            'fga_any': p[p['EVENTMSGTYPE'].isin([1, 2])],
            'fga_miss': p[p['EVENTMSGTYPE'] == 2],
            'reb': p[p['EVENTMSGTYPE'] == 4],
            'blk': p[(p['EVENTMSGTYPE'] == 2) & txt.str.contains('BLOCK')],
            'dead': p[p['EVENTMSGTYPE'].isin(self.DEAD_TYPES)],
            'to_any': p[p['EVENTMSGTYPE'] == 5],
        }

    def _mask(self, ev, period, qt, player=None, action_in=None, action_not_in=None):
        if ev.empty:
            return 0
        lag = qt - ev['Qtime']
        m = (ev['PERIOD'] == period) & (lag >= self.LAG_MIN) & (lag < self.LAG_MAX)
        if player is not None:
            m &= ev['PLAYER1_ID'] == player
        if action_in is not None:
            m &= ev['EVENTMSGACTIONTYPE'].isin(action_in)
        if action_not_in is not None:
            m &= ~ev['EVENTMSGACTIONTYPE'].isin(action_not_in)
        out = 0
        for k in np.floor(lag[m].to_numpy() - self.LAG_MIN).astype(int):
            out |= 1 << int(k)
        return out

    def _exclusions(self, period, qt, passer):
        e = self._ev
        return dict(m_shot_by_releaser=self._mask(e['fga_any'], period, qt, passer),
                    m_miss_any=self._mask(e['fga_miss'], period, qt),
                    m_reb=self._mask(e['reb'], period, qt), m_blk=self._mask(e['blk'], period, qt),
                    m_dead=self._mask(e['dead'], period, qt),
                    m_to_np_by_releaser=self._mask(e['to_any'], period, qt, passer, action_not_in=self.FAIL_ACTION_TYPES),
                    m_to_badpass_by_releaser=self._mask(e['to_any'], period, qt, passer, action_in=self.FAIL_ACTION_TYPES))

    # ------------------------------------------------------------- features
    def _vel(self, r, pid):
        fr = self._fr
        r0 = r - 5
        if r0 < 0 or fr['block'][r0] != fr['block'][r]:
            return np.array([0.0, 0.0]), False
        a, b = np.flatnonzero(fr['pid'][r0] == pid), np.flatnonzero(fr['pid'][r] == pid)
        if len(a) == 0 or len(b) == 0:
            return np.array([0.0, 0.0]), False
        dt = fr['ut'][r] - fr['ut'][r0]
        if dt <= 0:
            return np.array([0.0, 0.0]), False
        return (fr['xy'][r][b[0]] - fr['xy'][r0][a[0]]) / dt, True

    def _features_at(self, r, passer, team, target, basket_x, shot_clock):
        fr = self._fr
        pid, xy = fr['pid'][r], fr['xy'][r]
        ip, it = np.flatnonzero(pid == passer), np.flatnonzero(pid == target)
        if len(ip) == 0 or len(it) == 0 or basket_x is None or np.isnan(basket_x):
            return None
        opp = (fr['team'][r] != team) & (pid != 0)
        pv, _ = self._vel(r, passer)
        tv, _ = self._vel(r, target)
        bk = np.array([basket_x, 25.0])
        f = pass_features(xy[ip[0]], xy[it[0]], xy[opp], bk, shot_clock, pv, tv)
        if f is None:
            return None
        rs = shot_features(xy[it[0]], xy[opp], bk, tv, shot_clock)           # what a shot by the intended receiver would look like now
        f.update({'rs_' + k: v for k, v in rs.items() if k != 'shot_clock'})
        ps = shot_features(xy[ip[0]], xy[opp], bk, pv, shot_clock)           # and by the passer
        f.update({'ps_' + k: v for k, v in ps.items() if k != 'shot_clock'})
        return f

    def _shot_clock(self, row):
        sc = row['shot_clock']
        if pd.isna(sc) and row['quarter_time'] <= 24:
            sc = row['quarter_time']
        return np.nan if pd.isna(sc) else float(sc)

    # ----------------------------------------------------------------- build
    def build(self):
        self.track_passes()                    # hold segments, attack baskets, frame arrays
        self._pbp_index()
        fr, segs = self._fr, self.segments
        rows, consumed = [], set()
        n_skip = defaultdict(int)
        for i, s in enumerate(segs):
            if i in consumed:
                continue
            r = s['end']
            end, outcome, nxt = self._flight_outcome(segs, i)
            if end is not None:                                    # intercepted / out of bounds / fumbled
                if outcome == 'fumbled':
                    consumed.add(i + 1)
                if not self._flight_ok(r, end, check_oob=False):
                    n_skip['fail_flight_filtered'] += 1
                    continue
                label, recv, actual = 0, end, None
            elif outcome == 'teammate_holds_next':
                if (s['block'] != nxt['block'] or s['team'] != nxt['team'] or s['holder'] == nxt['holder']
                        or not self._flight_ok(r, nxt['start'])):
                    n_skip['comp_filtered'] += 1
                    continue
                label, recv, actual, end = 1, nxt['start'], nxt['holder'], nxt['start']
            else:
                n_skip[outcome] += 1
                continue
            row = self._row(r, recv, end, s['holder'], s['team'], actual, bool(label), 'completed' if label else outcome)
            tgt = row['target_heading']
            if tgt is None:
                n_skip['no_heading_target'] += 1
                continue
            sc = self._shot_clock(row)
            flight = fr['ball'][r:end + 1]
            zmax = float(np.nanmax(flight[:, 2])) if np.isfinite(flight[:, 2]).any() else np.nan
            f = self._features_at(r, s['holder'], s['team'], tgt, row['basket_x'], sc)
            if f is None:
                n_skip['no_features'] += 1
                continue
            rows.append(dict(
                game_id=int(self.game_id), frame=int(r), quarter=row['quarter'], quarter_time=row['quarter_time'],
                game_time=row['game_time'], passer=int(s['holder']), team=int(s['team']), actual_target=actual,
                heading_target=int(tgt), label=label, outcome=row['outcome'], flight_s=row['flight_s'],
                clock_running=row['clock_running'], inbound=row['inbound'], live=int(row['live']), zmax=zmax,
                hold_frames=int(s['end'] - s['start'] + 1),
                **self._exclusions(row['quarter'], row['quarter_time'], s['holder']), **f))
        self.pass_rows = pd.DataFrame(rows)
        self.n_skip = dict(n_skip)

    # ----------------------------------------------------------------- shots
    def build_shots(self):
        """One row per pbp field-goal attempt matched to the shooter's last hold before it (features at release)."""
        fr, segs = self._fr, self.segments
        by_holder = defaultdict(list)
        for i, s in enumerate(segs):
            by_holder[s['holder']].append(i)
        out, unmatched = [], 0
        for _, ev in self._ev['fga_any'].iterrows():
            shooter, period, e = int(ev['PLAYER1_ID']), int(ev['PERIOD']), float(ev['Qtime'])
            cands = [i for i in by_holder[shooter] if fr['quarter'][segs[i]['end']] == period
                     and SHOT_WINDOW[0] <= fr['qt'][segs[i]['end']] - e <= SHOT_WINDOW[1]]
            if not cands:
                unmatched += 1
                continue
            i = min(cands, key=lambda k: abs(fr['qt'][segs[k]['end']] - e - SHOT_TYPICAL_LAG))
            s = segs[i]
            r = s['end']
            basket = self._attack_basket(s['team'], period)
            if basket is None:
                unmatched += 1
                continue
            pid, xy = fr['pid'][r], fr['xy'][r]
            ip = np.flatnonzero(pid == shooter)
            opp = (fr['team'][r] != s['team']) & (pid != 0)
            if len(ip) == 0 or not opp.any() or np.isnan(xy[ip[0]]).any():
                unmatched += 1
                continue
            pv, _ = self._vel(r, shooter)
            sf = shot_features(xy[ip[0]], xy[opp], np.array([basket[0], 25.0]), pv)
            if sf is None:
                unmatched += 1
                continue
            desc = ' '.join(str(ev[c]) for c in ('HOMEDESCRIPTION', 'VISITORDESCRIPTION', 'NEUTRALDESCRIPTION') if pd.notna(ev[c]))
            sc = fr['qt'][r]
            smom = self.moments['shot_clock'].iat[r]
            if pd.isna(smom) and sc <= 24:
                smom = sc
            out.append(dict(game_id=int(self.game_id), frame=int(r), quarter=period, quarter_time=float(sc), shooter=shooter,
                            team=int(s['team']), made=int(ev['EVENTMSGTYPE'] == 1), is3=int('3PT' in desc),
                            **{k: v for k, v in sf.items() if k != 'shot_clock'}, shot_clock=float(smom) if not pd.isna(smom) else np.nan,
                            live=int(self._live_flags(r, shooter)['live']), assisted=int(pd.notna(ev['PLAYER2_ID']) and ev['PLAYER2_ID'] != 0 and ev['EVENTMSGTYPE'] == 1),
                            assister=int(ev['PLAYER2_ID']) if pd.notna(ev['PLAYER2_ID']) and ev['EVENTMSGTYPE'] == 1 else 0,
                            lag=float(fr['qt'][r] - e), desc=desc[:60]))
        self.shot_rows = pd.DataFrame(out)
        self.n_shot_unmatched = unmatched


def read_games(path):
    return [l.strip().replace('.7z', '') for l in open(path) if l.strip()]


def split(game):
    c = game.split('.')
    return '.'.join(c[:3]), c[3], c[5]


def process(game, out_dir):
    date, away, home = split(game)
    t0 = time.perf_counter()
    pt = DiffPT(date, home, away)
    pt.build()
    pt.build_shots()
    rep = dict(game=game, passes=len(pt.pass_rows), fails=int((pt.pass_rows['label'] == 0).sum()) if len(pt.pass_rows) else 0,
               shots=len(pt.shot_rows), shots_unmatched=pt.n_shot_unmatched, skipped=pt.n_skip,
               seconds=time.perf_counter() - t0, code=P.CODE_VERSION)
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/{game}.pkl'
    with open(path + '.tmp', 'wb') as fh:
        pickle.dump(dict(passes=pt.pass_rows.assign(game=game), shots=pt.shot_rows.assign(game=game), report=rep), fh, protocol=4)
    os.replace(path + '.tmp', path)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game_list')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    games = read_games(a.game_list)
    todo = [g for g in games if not os.path.exists(f'{a.out}/{g}.pkl')]
    print(f'{len(games)} games, {len(todo)} to do, passtracking {P.CODE_VERSION}', flush=True)
    for k, g in enumerate(todo):
        try:
            rep = process(g, a.out)
            print(f"[{k + 1}/{len(todo)}] {g}: passes {rep['passes']} fails {rep['fails']} shots {rep['shots']} "
                  f"(unmatched {rep['shots_unmatched']}) {rep['seconds']:.1f}s", flush=True)
        except Exception as ex:
            print(f'[{k + 1}/{len(todo)}] {g}: FAILED {type(ex).__name__}: {ex}', flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
