#!/bin/bash
# Resumable full-batch driver for a machine with little disk. For each chunk of games:
#   1. work out which games still need processing (passtracking.py --print-todo),
#   2. download just those archives from GitHub (4 in parallel) into
#      $TRACKING_DATA_DIR/game logs, outside the repo, so .git never grows,
#   3. run the resumable batch on the chunk (holding the heavy-job lock),
#   4. delete the archives it downloaded.
# Stop it any time (Ctrl-C / shutdown) and run the same command again.
#
# Usage:  ./run_chunked.sh <game list> [chunk size, default 100]
# Needs:  gh (authenticated), 7z, the venv at ../.venv, network.
# Env:    TRACKING_DATA_DIR (default ~/.cache/basketballPlayOptimisation/Tracking_Data),
#         GH_REPO, GH_REF (default cloud-full-batch), MIN_FREE_GB (default 3), PYTHON.
set -u
list="${1:?Usage: $0 <game list> [chunk size]}"; chunk="${2:-100}"
here="$(cd "$(dirname "$0")" && pwd)"; cd "$here"
export TRACKING_DATA_DIR="${TRACKING_DATA_DIR:-$HOME/.cache/basketballPlayOptimisation/Tracking_Data}"
GH_REPO="${GH_REPO:-Yash-Sangtani/basketballPlayOptimisation}"; GH_REF="${GH_REF:-cloud-full-batch}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"; PY="${PYTHON:-$here/../.venv/bin/python}"
GL="$TRACKING_DATA_DIR/game logs"; mkdir -p "$GL"
[ -d "$TRACKING_DATA_DIR/events" ] || { mkdir -p "$TRACKING_DATA_DIR/events" && cp "$here/../Tracking_Data/events/"*.csv "$TRACKING_DATA_DIR/events/"; }

free_gb() { df -g / | awk 'NR==2{print $4}'; }
fetch_one() {   # download one archive; leave nothing behind on failure
    out="$GL/$1.7z"
    if gh api -H "Accept: application/vnd.github.raw" \
        "repos/$GH_REPO/contents/Tracking_Data/game%20logs/$1.7z?ref=$GH_REF" > "$out.tmp" 2>/dev/null && [ -s "$out.tmp" ]; then
        mv "$out.tmp" "$out"
    else
        rm -f "$out.tmp"; echo "download failed: $1" >&2
    fi
}
export -f fetch_one; export GL GH_REPO GH_REF

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tr -d '\r' < "$list" | grep -v '^$' | sed 's/\.7z$//' > "$work/all.txt"
split -l "$chunk" "$work/all.txt" "$work/chunk_"
n_chunks=$(ls "$work"/chunk_* | wc -l | tr -d ' '); i=0
for c in "$work"/chunk_*; do
    i=$((i + 1))
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then echo "STOP: only $(free_gb) GB free (< $MIN_FREE_GB). Free some disk and re-run." >&2; exit 1; fi
    "$PY" passtracking.py "$c" --print-todo > "$c.todo"
    todo=$(wc -l < "$c.todo" | tr -d ' ')
    echo "[chunk $i/$n_chunks] $todo of $(wc -l < "$c" | tr -d ' ') games to do; free disk $(free_gb) GB"
    [ "$todo" -eq 0 ] && continue
    : > "$c.got"
    # download the ones we do not already have; remember which ones WE fetched so we can delete them
    while read -r g; do [ -s "$GL/$g.7z" ] && echo "$g" >> "$c.have" || echo "$g" >> "$c.need"; done < "$c.todo"
    [ -s "$c.need" ] && { xargs -P4 -I{} bash -c 'fetch_one {}' < "$c.need"; }
    # only hand the batch the games whose archive is present (a failed download is retried next run)
    while read -r g; do [ -s "$GL/$g.7z" ] && echo "$g" >> "$c.run"; done < "$c.todo"
    [ -s "$c.run" ] && "$here/heavy.sh" "$PY" passtracking.py "$c.run"
    [ -s "$c.need" ] && while read -r g; do rm -f "$GL/$g.7z"; done < "$c.need"
done
echo "done. Status: $("$PY" - <<'PYEOF'
import pandas as pd
s = pd.read_csv('../Tracking_Data/pass_features/_status.csv', keep_default_na=False).drop_duplicates('game', keep='last')
print(s.status.value_counts().to_dict())
PYEOF
)"
