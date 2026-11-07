"""Pick the ~100-game dev sample for WS3 (T1/T2/T5): sort the 501 dev games by date, cut them
into 100 contiguous blocks and take one game per block with a fixed seed, so the sample is spread
over the whole dev period.  Usage: python ws3_sample.py > splits/ws3_sample_games.txt (run from Code/).
Reads ONLY splits/dev_games.txt (never the test list)."""
import random
from datetime import datetime

N, SEED = 100, 0
games = [l.strip() for l in open('splits/dev_games.txt') if l.strip()]
games.sort(key=lambda g: (datetime.strptime('.'.join(g.split('.')[:3]), '%m.%d.%Y'), g))
rng = random.Random(SEED)
for k in range(N):
    lo, hi = k * len(games) // N, (k + 1) * len(games) // N
    print(rng.choice(games[lo:hi]))
