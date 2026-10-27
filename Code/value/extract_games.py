"""
Run state extraction (states.extract_game) for the games in a list; one pickle per game in OUT.
Resumable: games with a pickle are skipped. TRACKING_DATA_DIR must point at a folder with
`game logs/<game>.7z` and `events/<id>.csv` (set BEFORE python starts).

    TRACKING_DATA_DIR=/private/tmp/ws5b/Tracking_Data python extract_games.py list.txt /private/tmp/ws5b/states
"""
import os
import pickle
import sys
import time
import traceback

if 'TRACKING_DATA_DIR' not in os.environ:
    sys.exit('set TRACKING_DATA_DIR first')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import states  # noqa: E402


def main():
    games = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for g in games:
        path = os.path.join(out, f'{g}.pkl')
        if os.path.exists(path):
            continue
        t0 = time.perf_counter()
        try:
            res = states.extract_game(g)
        except Exception as ex:
            print(f'FAILED {g}: {type(ex).__name__}: {ex}', flush=True)
            traceback.print_exc()
            continue
        res['seconds'] = time.perf_counter() - t0
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(res, f)
        os.replace(path + '.tmp', path)
        d = res['diag']
        print(f"{g}: {res['seconds']:.1f}s samples={d['kept']}/{d['candidates']} passes ok={d['pass_ok']}/{d['passes_completed']}", flush=True)


if __name__ == '__main__':
    main()
