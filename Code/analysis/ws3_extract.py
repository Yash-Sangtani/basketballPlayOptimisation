"""WS3 extraction: one pass per game that records everything the selection-caveat tests need.

Does NOT modify passtracking.py: it subclasses PassTracking and re-uses `features.features_extraction`
unchanged (so the 'actual'-mode rows are bit-identical to the shared baseline pickles; checked by
--check-baseline), running it twice, once per target mode ('actual' and 'heading').  Because both modes
come out of one pass over each game, no PASS_TARGET_MODE switch in passtracking.py is needed.

Per game it writes <out>/<game>.pkl with
  ext          one row per release: the baseline rows (src 'completed' / 'pbp_fail') plus
               - src 'trk_only'  : same-team release whose flight ends in an opponent hold / out of bounds /
                                   fumbled teammate touch, passing the SAME _flight_ok filter, with NO pbp bad-pass
                                   event matched (T2);
               - src 'np_<kind>' : releases of non-pass pbp turnovers (lost ball, travel, off. foul, ...) that
                                   have no row yet (T5);
               plus annotation columns (trk_fail, clock_running, inbound, zmax, pbp_tags, pbp_ctx, ...).
  feats        {'actual': (features, meta), 'heading': (features, meta)}, same index as ext
  solo         passer-side features for EVERY row, even those with no target (T5)
  np_events    one row per non-pass pbp turnover with how it matched
  pbp          compact play-by-play for manual inspection
  report       counts

Usage (from Code/, TRACKING_DATA_DIR set BEFORE the import):
  python analysis/ws3_extract.py <game list> [--out DIR] [--print-todo] [--check-baseline DIR]
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
os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import passtracking as P                # noqa: E402
from passtracking import PassTracking, features   # noqa: E402

OUT = os.environ.get('WS3_OUT', '/private/tmp/ws3/out')
CTX_WINDOW = (-3.0, 8.0)     # s, release clock minus pbp clock; same window the baseline uses for bad passes
EVENT_TAG = {1: 'shot_made', 2: 'shot_miss', 3: 'ft', 4: 'rebound', 6: 'foul', 7: 'violation', 8: 'sub',
             9: 'timeout', 10: 'jumpball', 12: 'period_start', 13: 'period_end'}
NP_KIND = {2: 'lost_ball', 4: 'travel', 5: 'off_foul'}   # pbp turnover action types; everything else 'other_to'
FIX_MIN = 5.0


class ExtPT(PassTracking):

    # ------------------------------------------------------------ pbp context
    def _ctx(self, period, qt, passer, with_text):
        p = self.pbp
        lag = qt - p['Qtime']
        sel = p[(p['PERIOD'] == period) & (lag >= CTX_WINDOW[0]) & (lag <= CTX_WINDOW[1])]
        tags, text, to_by_passer = [], [], 0
        for _, e in sel.iterrows():
            t, a = int(e['EVENTMSGTYPE']), int(e['EVENTMSGACTIONTYPE'])
            desc = ' / '.join(str(e[c])[:70] for c in ('HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION')
                              if pd.notna(e[c]))
            if t in (1, 2) and int(e['PLAYER1_ID']) == passer:
                tags.append('shot_by_releaser')
            if t == 5:
                tag = 'to_badpass' if a in self.FAIL_ACTION_TYPES else ('to_lostball' if a == 2 else 'to_other')
                if int(e['PLAYER1_ID']) == passer:
                    to_by_passer = a
            else:
                tag = EVENT_TAG.get(t)
            if tag:
                tags.append(tag)
            if 'BLOCK' in desc:
                tags.append('block')
            if 'STEAL' in desc:
                tags.append('steal')
            if with_text:
                text.append(f"[{e['PCTIMESTRING']} T{t}/{a}] {desc}")
        return ','.join(sorted(set(tags))), to_by_passer, (' | '.join(text) if with_text else '')

    # --------------------------------------------------------- live/dead ball
    def _clock_running(self, r):
        fr = self._fr
        a, b = max(r - 12, 0), min(r + 12, fr['n'] - 1)
        while a < r and fr['block'][a] != fr['block'][r]:
            a += 1
        while b > r and fr['block'][b] != fr['block'][r]:
            b -= 1
        dt = fr['ut'][b] - fr['ut'][a]
        if dt < 0.3:
            return np.nan
        return float((fr['qt'][a] - fr['qt'][b]) >= 0.5 * dt)

    def _annotate(self, row, seg, with_text):
        fr = self._fr
        r, end, passer = int(row['frame']), int(row['end_frame']), int(row['passer'])
        flight = fr['ball'][r:end + 1]
        z = flight[:, 2]
        x0 = fr['xy'][r][fr['pid'][r] == passer]
        inbound = np.nan
        if len(x0) and not np.isnan(x0[0]).any():
            px, py = x0[0]
            inbound = float(px < -0.5 or px > 94.5 or py < -0.5 or py > 50.5)
        e8 = min(r + self.HEAD_FRAMES, end)
        dt8 = fr['ut'][e8] - fr['ut'][r]
        init_speed = float(np.hypot(*(fr['ball'][e8, :2] - fr['ball'][r, :2])) / dt8) if dt8 > 0 and e8 > r else np.nan
        oob_same = np.nan
        if row['outcome'] == 'out_of_bounds':
            nxt = next((s for s in self.segments if s['start'] >= end and fr['ut'][s['start']] - fr['ut'][end] <= 15), None)
            oob_same = np.nan if nxt is None else float(nxt['team'] == row['team_passer'])
        tags, to_by_passer, txt = self._ctx(int(row['quarter']), float(row['quarter_time']), passer, with_text)
        # dribbling evidence: ball touching the floor while held (last 25 frames of the hold)
        hz = fr['ball'][max(seg['start'], r - 25):r + 1, 2]
        hz = hz[np.isfinite(hz)]
        bounces = int(((hz[1:-1] < hz[:-2]) & (hz[1:-1] <= hz[2:]) & (hz[1:-1] < 1.5)).sum()) if len(hz) > 2 else 0
        # what happens after an interception: does the other team keep the ball, or does it come straight back?
        opp_first, regain = np.nan, np.nan
        if row['outcome'] in ('intercepted', 'fumbled'):
            j = next((k for k, sg in enumerate(self.segments) if sg['start'] >= end and sg['team'] != row['team_passer']), None)
            if j is not None and self.segments[j]['start'] - end <= 3:
                sj = self.segments[j]
                opp_first = float(sj['end'] - sj['start'] + 1)
                back = next((sg for sg in self.segments[j + 1:j + 6] if sg['team'] == row['team_passer'] and sg['block'] == sj['block']), None)
                regain = float(fr['ut'][back['start']] - fr['ut'][sj['start']]) if back is not None else np.nan
        return dict(clock_running=self._clock_running(r), inbound=inbound,
                    hold_zmin=float(hz.min()) if len(hz) else np.nan, hold_bounces=bounces,
                    opp_first_hold_frames=opp_first, regain_s=regain,
                    zmax=float(np.nanmax(z)) if np.isfinite(z).any() else np.nan,
                    ball_disp=float(np.hypot(*(fr['ball'][end, :2] - fr['ball'][r, :2]))),
                    init_speed=init_speed, hold_frames=int(seg['end'] - seg['start'] + 1),
                    oob_next_same=oob_same, pbp_tags=tags, pbp_to_by_passer=to_by_passer, pbp_ctx=txt)

    # ------------------------------------------------------- non-pass events
    def _match_np(self, segs, row_by_frame, rows):
        """Non-pass turnovers: match each to the committer's hold (same window as the bad passes).
        If that release already has a row, annotate it; else create an 'np_<kind>' row."""
        fr, pbp = self._fr, self.pbp
        ev = pbp[(pbp['EVENTMSGTYPE'] == 5) & ~pbp['EVENTMSGACTIONTYPE'].isin(self.FAIL_ACTION_TYPES)]
        by_holder = defaultdict(list)
        for i, s in enumerate(segs):
            by_holder[s['holder']].append(i)
        out, new_rows = [], []
        for _, e in ev.iterrows():
            a = int(e['EVENTMSGACTIONTYPE'])
            kind = NP_KIND.get(a, 'other_to')
            committer, period, qev = int(e['PLAYER1_ID']), int(e['PERIOD']), float(e['Qtime'])
            stealer = int(e['PLAYER2_ID']) if pd.notna(e['PLAYER2_ID']) else 0
            rec = dict(kind=kind, action=a, committer=committer, period=period, qtime=qev, steal=int(stealer != 0),
                       desc=' / '.join(str(e[c])[:70] for c in ('HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION')
                                       if pd.notna(e[c])),
                       matched=0, reason='', frame=-1, in_table='', flight_outcome='')
            cands = [i for i in by_holder[committer] if fr['quarter'][segs[i]['end']] == period
                     and CTX_WINDOW[0] <= fr['qt'][segs[i]['end']] - qev <= CTX_WINDOW[1]]
            if not cands:
                rec['reason'] = 'no_hold_by_committer'
                out.append(rec)
                continue
            scored = []
            for i in cands:
                end, outcome, nxt = self._flight_outcome(segs, i)
                conf = outcome == 'intercepted' and stealer != 0 and nxt is not None and nxt['holder'] == stealer
                scored.append((not conf, abs(fr['qt'][segs[i]['end']] - qev - self.FAIL_TYPICAL_LAG), i, end, outcome))
            _, _, i, end, outcome = min(scored, key=lambda t: t[:3])
            r = segs[i]['end']
            rec.update(matched=1, frame=r, flight_outcome=outcome if end is not None else f'none:{outcome}')
            if r in row_by_frame:
                k = row_by_frame[r]
                tgt_row = rows[k] if k < len(rows) else new_rows[k - len(rows)]
                rec['in_table'] = tgt_row['src']
                tgt_row['np_kind'] = tgt_row['np_kind'] or kind
            else:
                blk_last = int(fr['block_last'][segs[i]['block'] - 1])
                row = self._row(r, r, min(r + self.HEAD_FRAMES, blk_last), committer, segs[i]['team'], None, False, 'np_' + kind)
                row.update(src='np_' + kind, trk_fail=0, trk_outcome='', np_kind=kind, _seg=i)
                row_by_frame[r] = len(rows) + len(new_rows)
                new_rows.append(row)
                rec['in_table'] = 'np_new'
            out.append(rec)
        return out, new_rows

    # ------------------------------------------------------------------ build
    def build(self):
        self.track_passes()
        fr, segs = self._fr, self.segments
        rows = self.pass_attempt.to_dict('records')
        row_by_frame = {int(r_['frame']): k for k, r_ in enumerate(rows)}
        seg_by_end = {s['end']: i for i, s in enumerate(segs)}
        for r_ in rows:
            r_.update(src='completed' if r_['pass_successful'] == 1 else 'pbp_fail', trk_fail=0, trk_outcome='',
                      np_kind='', _seg=seg_by_end[int(r_['frame'])])
        # tracking-only failures: every hold whose flight ends in an opponent hold / out of bounds / fumble
        consumed, n_filtered, new = set(), 0, []
        for i, s in enumerate(segs):
            if i in consumed:
                continue
            end, outcome, nxt = self._flight_outcome(segs, i)
            if end is None:
                continue
            if outcome == 'fumbled':
                consumed.add(i + 1)       # the teammate's brief touch belongs to this flight
            r = s['end']
            if r in row_by_frame:
                rows[row_by_frame[r]].update(trk_fail=1, trk_outcome=outcome)
                continue
            if not self._flight_ok(r, end, check_oob=False):
                n_filtered += 1
                continue
            row = self._row(r, end, end, s['holder'], s['team'], None, False, outcome)
            row.update(src='trk_only', trk_fail=1, trk_outcome=outcome, np_kind='', _seg=i)
            row_by_frame[r] = len(rows) + len(new)
            new.append(row)
        rows += new
        np_events, np_rows = self._match_np(segs, row_by_frame, rows)
        rows += np_rows
        for k, r_ in enumerate(rows):
            r_.update(self._annotate(r_, segs[r_['_seg']], with_text=r_['src'] != 'completed'))
        ext = pd.DataFrame(rows)
        ext.index = np.arange(len(ext))
        self.n_base = len(self.pass_attempt)
        self.report = dict(n_base=self.n_base, n_trk_only=len(new), n_trk_filtered=n_filtered, n_np_new=len(np_rows),
                           n_np_events=len(np_events), n_np_matched=sum(e['matched'] for e in np_events),
                           failure_report=self.failure_report)
        self.ext = ext
        self.np_events = pd.DataFrame(np_events)

    def features_both(self):
        self.pass_attempt = self.ext      # features_extraction reads self.passes.pass_attempt
        out = {}
        for mode in ('actual', 'heading'):
            f = features.__new__(features)
            f.passes, f.target_mode, f.features, f.meta = self, mode, None, None
            f.features_extraction()
            out[mode] = (f.features, f.meta)
            self._f = f
        # passer-side features for every row, target or not (same formulas as features_extraction)
        solo = []
        for idx, row in self.ext.iterrows():
            r, passer, team = int(row['frame']), row['passer'], row['team_passer']
            sd = dict(shot_clock=np.nan, backcourt=row['backcourt'], passer_basket_dist=np.nan,
                      closest_def_dist_passer=np.nan, passer_speed=np.nan)
            sc = row['shot_clock']
            if pd.isna(sc) and row['quarter_time'] <= 24:
                sc = row['quarter_time']
            sd['shot_clock'] = sc
            pid, xy = self._fr['pid'][r], self._fr['xy'][r]
            ip = np.flatnonzero(pid == passer)
            if len(ip) and not np.isnan(xy[ip[0]]).any() and not pd.isna(row['basket_x']):
                Pp = xy[ip[0]]
                opp = (self._fr['team'][r] != team) & (pid != 0)
                sd['passer_basket_dist'] = np.hypot(*(Pp - np.array([row['basket_x'], 25.0])))
                if opp.any():
                    sd['closest_def_dist_passer'] = np.hypot(*(xy[opp] - Pp).T).min()
                sd['passer_speed'] = self._f._speed(r, passer)
            solo.append(sd)
        return out, pd.DataFrame(solo, index=self.ext.index).astype(float)


def read_games(path):
    return [l.strip().replace('.7z', '') for l in open(path) if l.strip()]


def split(game):
    c = game.split('.')
    return '.'.join(c[:3]), c[3], c[5]    # date, away, home


def process(game, out_dir, baseline_dir=None):
    date, away, home = split(game)
    t0 = time.perf_counter()
    pt = ExtPT(date, home, away)
    pt.build()
    feats, solo = pt.features_both()
    ext = pt.ext.drop(columns=['positions', '_seg'])
    ext['game'] = game
    pbp = pt.pbp[['EVENTNUM', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE', 'PERIOD', 'PCTIMESTRING', 'Qtime', 'HOMEDESCRIPTION',
                  'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER2_ID']].copy()
    if baseline_dir:      # parity with the shared baseline (T1 'actual' mode must be identical)
        base_f = pd.read_pickle(f'{baseline_dir}/{date}.{away}.{home}.pkl')
        base_m = pd.read_pickle(f'{baseline_dir}/meta/{date}.{away}.{home}.pkl')
        f_act, m_act = feats['actual']
        sub = f_act.loc[f_act.index < pt.n_base]
        # last-ulp float32 differences between the Mac that built the baseline and other platforms are tolerated
        pd.testing.assert_frame_equal(sub, base_f, check_exact=False, rtol=1e-4, atol=1e-5)
        # meta: same values; dtype may differ (target is float when a row without a target sits in the table)
        pd.testing.assert_frame_equal(m_act.loc[m_act.index < pt.n_base], base_m.drop(columns=['game']),
                                      check_exact=False, rtol=1e-4, atol=1e-5, check_dtype=False)
    rep = dict(pt.report, game=game, game_id=int(pt.game_id), home_id=int(pt.home_id), away_id=int(pt.away_id),
               seconds=time.perf_counter() - t0)
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/{game}.pkl'
    with open(path + '.tmp', 'wb') as fh:
        pickle.dump(dict(ext=ext, feats=feats, solo=solo, np_events=pt.np_events.assign(game=game), pbp=pbp,
                         report=rep), fh, protocol=4)
    os.replace(path + '.tmp', path)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game_list')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--print-todo', action='store_true')
    ap.add_argument('--check-baseline', metavar='DIR', help='assert actual-mode features equal the baseline pickles in DIR')
    a = ap.parse_args()
    games = read_games(a.game_list)
    todo = [g for g in games if not os.path.exists(f'{a.out}/{g}.pkl')]
    if a.print_todo:
        print('\n'.join(todo))
        return
    print(f'{len(games)} games, {len(todo)} to do, code version {P.CODE_VERSION}', flush=True)
    for k, g in enumerate(todo):
        try:
            rep = process(g, a.out, a.check_baseline)
            print(f'[{k + 1}/{len(todo)}] {g}: base {rep["n_base"]} trk_only {rep["n_trk_only"]} '
                  f'np_new {rep["n_np_new"]} {rep["seconds"]:.1f}s', flush=True)
        except Exception as e:      # a failed game is logged, never silently skipped
            print(f'[{k + 1}/{len(todo)}] {g}: FAILED {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
