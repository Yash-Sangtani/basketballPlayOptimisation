"""
Freeze the dev/test split of the usable games (run once; the output files are tracked).

Test = the last ~20% of games by DATE (whole dates stay together), so evaluation
looks forward in time. Anything that tunes thresholds, picks features or compares
models may only use the dev games. Run the test games once, at the end.

Usage (from Code/):  python splits/make_splits.py
Reads ../Shots/allgames.txt, writes splits/dev_games.txt and splits/test_games.txt.
"""
import os
from datetime import datetime

#Archives with no usable tracking data (see CLAUDE.md "Data facts").
KNOWN_BAD = {
    '01.14.2016.CHI.at.PHI', '01.14.2016.CLE.at.SAS', '01.14.2016.DET.at.MEM',
    '01.23.2016.UTA.at.WAS', '01.23.2016.CHI.at.CLE',
}
TEST_FRACTION = 0.20

here = os.path.dirname(os.path.abspath(__file__))
games = []
with open(os.path.join(here, '..', '..', 'Shots', 'allgames.txt')) as f:
    for line in f:
        name = line.strip().replace('.7z', '')
        if name and name not in KNOWN_BAD:
            games.append(name)
games = sorted(set(games), key=lambda g: (datetime.strptime('.'.join(g.split('.')[:3]), '%m.%d.%Y'), g))

#Walk back from the end, taking whole dates until we have >= TEST_FRACTION of the games.
by_date = {}
for g in games:
    by_date.setdefault('.'.join(g.split('.')[:3]), []).append(g)
dates = sorted(by_date, key=lambda d: datetime.strptime(d, '%m.%d.%Y'))
test, n_test = [], 0
for d in reversed(dates):
    if n_test >= TEST_FRACTION * len(games):
        break
    test = by_date[d] + test
    n_test += len(by_date[d])
test_set = set(test)
dev = [g for g in games if g not in test_set]

for name, items in (('dev_games.txt', dev), ('test_games.txt', test)):
    with open(os.path.join(here, name), 'w') as f:
        f.write('\n'.join(items) + '\n')
print(f"{len(games)} usable games -> dev {len(dev)} ({dev[0].split('.')[0:3]} .. {dev[-1].split('.')[0:3]}), "
      f"test {len(test)} (from {'.'.join(test[0].split('.')[:3])} to {'.'.join(test[-1].split('.')[:3])})")
