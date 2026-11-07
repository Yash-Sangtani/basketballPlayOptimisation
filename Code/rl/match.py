"""Stage 6E: match mode. A game is four quarter chains played by the engine; the viewer is a self-contained HTML page (score timeline, quarter table, team and player box score,
a real-time GIF excerpt).

    # replay real games: each real quarter's ten players and start formation, four quarters, then compare with the real score (calibration over many games)
    python rl/match.py replay --split holdout --reps 2 --out ../reports/rl-stage6e/replay_holdout.json [--ckpt league.pkl]
    # a named matchup of two teams (real lineups composed from different games) with a viewer page
    python rl/match.py versus --home 1610612744 --away 1610612759 --seed 3 --out ../reports/rl-stage6e/versus_GSW_SAS [--ckpt ...]

Roster teams are the team ids of the players. Substitutions happen at quarter boundaries only (each quarter starts from a real lineup of that team); no timeouts; the engine's
quarter clock is 12:00. Player points are from the engine's own box counters (field goals and free throws).
"""
import argparse
import base64
import io
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'difficulty'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import league as L                       # noqa: E402
from load import load_games             # noqa: E402
from profiles import build_profiles      # noqa: E402
from splits import games as split_games  # noqa: E402
from train_executor import fit_val_games  # noqa: E402
from engine_validate import quarter_starts, real_quarter_stats   # noqa: E402


def team_of(d):
    return {int(r.player): int(r.team) for r in d['roster'].itertuples()}


def name_of(G):
    out = {}
    for d in G:
        for r in d['roster'].itertuples():
            out[int(r.player)] = f'{r.first} {r.last}'.strip()
    return out


def quarter_team_ids(eng, G, starts):
    """-> (N,2) team ids of roster A / B of every chain."""
    out = np.zeros((eng.N, 2), np.int64)
    for i, (g, k) in enumerate(starts):
        tm = team_of(G[g])
        out[i] = [tm.get(int(eng.R_ids[i, 0]), 0), tm.get(int(eng.R_ids[i, 5]), 0)]
    return out


def agents_from(st, ckpt, seed=0):
    if ckpt in (None, 'imitation'):
        return (None, None)
    ck = pickle.load(open(ckpt, 'rb'))
    o, d = L.make_agents(st, ck, learn=False, seed=seed)
    return (o if ck.get('off') is not None else None, d if ck.get('def') is not None else None)


def compose_start(G, home, away, rng):
    """A synthetic quarter start for two named teams: the five of `home` on offence from a real state of one game, the five of `away` on defence from a real state of another.
    Returns a one-state game dict (3 identical states) that Engine.reset can use."""
    def find(team, side):
        cands = []
        for gi, d in enumerate(G):
            tm = team_of(d)
            ids = d['ids']
            ok = d['live'] & d['frontcourt'] & (d['holder'] != 0)
            ks = np.flatnonzero(ok)[::50]
            if side == 'off':
                good = np.array([all(tm.get(int(p), -1) == team for p in ids[k, :5]) for k in ks], bool)
            else:
                good = np.array([all(tm.get(int(p), -1) == team for p in ids[k, 5:]) for k in ks], bool)
            cands += [(gi, int(k)) for k in ks[good]]
        if not cands:
            raise SystemExit(f'no state with the five of team {team} on {"offence" if side == "off" else "defence"} in the loaded games')
        return cands[rng.integers(len(cands))]
    ga, ka = find(home, 'off')
    gb, kb = find(away, 'def')
    A, Bd = G[ga], G[gb]
    pos = np.concatenate([A['pos'][ka][:5], Bd['pos'][kb][5:]], axis=0)
    ids = np.concatenate([A['ids'][ka][:5], Bd['ids'][kb][5:]])
    roster = pd.concat([A['roster'], Bd['roster']]).drop_duplicates('player')
    n = 3
    return dict(pos=np.repeat(pos[None], n, 0).astype(np.float32), ids=np.repeat(ids[None], n, 0), holder=np.repeat(A['holder'][ka][None], n, 0),
                ball=np.repeat(A['ball'][ka][None], n, 0).astype(np.float32), roster=roster, frame=np.arange(n) * 5), (ga, ka, gb, kb)


def box_score(eng, i, names):
    rows = []
    for (ep, pid), b in eng.box.items():
        if ep == i:
            j = int(np.flatnonzero(eng.R_ids[i] == pid)[0]) if (eng.R_ids[i] == pid).any() else -1
            rows.append(dict(team='A' if 0 <= j < 5 else 'B', player=names.get(pid, str(pid)), **b))
    return pd.DataFrame(rows)


def replay(a):
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    names_g = split_games(a.split) if a.split in ('train', 'holdout') else None
    G = load_games(a.data if a.split != 'test' else '../data/rl_test', names_g)
    qs = quarter_starts(G)
    if a.max_games:
        keep = set(range(min(a.max_games, len(G))))
        qs = [x for x in qs if x[0] in keep]
    A = agents_from(st, a.ckpt, 1)
    out = []
    t0 = time.time()
    batch = a.batch
    for rep in range(a.reps):
        for b0 in range(0, len(qs), batch):
            sub = qs[b0:b0 + batch]
            starts = [(g, k) for g, k, q in sub]
            eng = L.run_match(st, G, table, starts, seed=500 + 17 * rep + b0, blocks=a.blocks, A=A, B=A)
            tid = quarter_team_ids(eng, G, starts)
            for i, (g, k, q) in enumerate(sub):
                out.append(dict(rep=rep, game=G[g]['report']['game_id'], gname=G[g]['game'], period=int(q), teamA=int(tid[i, 0]), teamB=int(tid[i, 1]), ptsA=float(eng.score[i, 0]), ptsB=float(eng.score[i, 1]),
                                unfinished=bool(eng.alive[i]), poss=[float(eng.stats['poss'][i, 0]), float(eng.stats['poss'][i, 1])]))
            print(f'rep {rep} batch {b0 // batch + 1}/{(len(qs) + batch - 1) // batch}  {time.time() - t0:.0f}s', flush=True)
            json.dump(dict(rows=out), open(a.out, 'w'))
    # real quarter scores
    R = real_quarter_stats(os.path.join(HERE, '..', '..', 'reports', 'rl-stage6a'), a.split if a.split != 'test' else 'holdout', None) if a.split != 'test' else None
    if R is not None:
        json.dump(dict(rows=out, real=R.to_dict('records')), open(a.out, 'w'))


def versus(a):
    st = L.load_stack(a.model)
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(a.data, fit_g))
    G = load_games(a.data, split_games('holdout') + split_games('train')[:150])
    rng = np.random.default_rng(a.seed)
    names = name_of(G)
    A = agents_from(st, a.ckpt, 1)
    synth, meta = [], []
    for q in range(4):
        d, m = compose_start(G, a.home, a.away, rng)
        synth.append(d)
        meta.append(m)
    games = synth
    starts = [(i, 2) for i in range(4)]
    eng = L.make_engine(st, a.seed)
    eng.off_agent, eng.def_agent = [A[0], None], [A[1], None]       # the checkpoint (if any) plays for the home side only
    eng.reset(games, table, starts)
    eng.run(a.blocks)
    return eng, names, G, meta


def html_page(eng, names, title, out_dir, excerpt_gif=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    N = eng.N
    sc = np.stack(eng.score_traj)            # (S, N, 2)
    ck = np.stack(eng.clock_traj)            # (S, N)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    off = 0.0
    cum = np.zeros(2)
    tx, ya, yb = [], [], []
    for q in range(N):
        t = (720.0 - ck[:, q]) + 720.0 * q
        ok = ck[:, q] >= 0
        tx.append(t[ok]); ya.append(sc[ok, q, 0] + cum[0]); yb.append(sc[ok, q, 1] + cum[1])
        cum += sc[-1, q]
        ax.axvline(720.0 * q, color='#bbb', lw=.6)
    tx, ya, yb = np.concatenate(tx), np.concatenate(ya), np.concatenate(yb)
    ax.plot(tx / 60, ya, label='home / roster A', color='#d62728'); ax.plot(tx / 60, yb, label='away / roster B', color='#1f77b4')
    ax.set_xlabel('game minute'); ax.set_ylabel('points'); ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=100); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode()
    q_rows = ''.join(f'<tr><td>Q{q + 1}</td><td>{int(eng.score[q, 0])}</td><td>{int(eng.score[q, 1])}</td></tr>' for q in range(N))
    tot = eng.score.sum(0)
    keys = ['pts', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm', 'orb', 'drb', 'tov', 'stl', 'poss']
    team_rows = ''.join(f'<tr><td>{k}</td><td>{eng.stats[k][:, 0].sum():.0f}</td><td>{eng.stats[k][:, 1].sum():.0f}</td></tr>' for k in keys)
    boxes = []
    for i in range(N):
        b = box_score(eng, i, names)
        if len(b):
            b['q'] = i + 1
            boxes.append(b)
    pb = pd.concat(boxes).groupby(['team', 'player'], as_index=False)[['pts', 'fga', 'fgm', 'fg3a', 'fg3m', 'fta', 'ftm']].sum().sort_values(['team', 'pts'], ascending=[True, False])
    pb_html = pb.to_html(index=False, border=0, classes='box')
    gif_html = ''
    if excerpt_gif and os.path.exists(excerpt_gif):
        gif_html = f'<h2>Excerpt (real time)</h2><img src="data:image/gif;base64,{base64.b64encode(open(excerpt_gif, "rb").read()).decode()}" style="max-width:100%">'
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 16px;color:#222}}table{{border-collapse:collapse;margin:.5rem 0}}td,th{{padding:3px 10px;border-bottom:1px solid #ddd;text-align:right}}h1{{font-size:1.3rem}}h2{{font-size:1.05rem;margin-top:1.4rem}}
.score{{font-size:2rem;font-weight:600}}.note{{color:#666;font-size:.85rem}}</style></head><body>
<h1>{title}</h1><div class="score">A {int(tot[0])} &ndash; {int(tot[1])} B</div>
<p class="note">Simulated by the quarter engine (Stage 6E): four quarters, substitutions only at quarter starts, no timeouts. Roster A attacks first in every quarter.</p>
<img src="data:image/png;base64,{img}" style="max-width:100%"><h2>By quarter</h2><table><tr><th></th><th>A</th><th>B</th></tr>{q_rows}</table>
<h2>Team totals</h2><table><tr><th></th><th>A</th><th>B</th></tr>{team_rows}</table><h2>Player box score (field goals and free throws)</h2>{pb_html}{gif_html}</body></html>"""
    path = os.path.join(out_dir, 'game.html')
    open(path, 'w').write(html)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['replay', 'versus'])
    ap.add_argument('--split', default='holdout'); ap.add_argument('--reps', type=int, default=2); ap.add_argument('--batch', type=int, default=95); ap.add_argument('--max-games', type=int, default=0)
    ap.add_argument('--blocks', type=int, default=1800); ap.add_argument('--ckpt', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--home', type=int); ap.add_argument('--away', type=int); ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--model', default='rl/model/executor_v5g.keras'); ap.add_argument('--data', default='../data/rl')
    a = ap.parse_args()
    if a.mode == 'replay':
        replay(a)
    else:
        eng, names, G, meta = versus(a)
        gif = os.path.join(a.out, 'excerpt.gif')
        os.makedirs(a.out, exist_ok=True)
        from engine_gif import render
        render(eng, 1, 2, 60.0, gif, label='versus  ')
        p = html_page(eng, names, f'Team {a.home} vs team {a.away} (simulated)', a.out, gif)
        print('wrote', p, 'final', eng.score.sum(0))


if __name__ == '__main__':
    main()
