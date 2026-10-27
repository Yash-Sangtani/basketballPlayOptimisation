#!/bin/bash
# Chunked, resumable state extraction for the value-function games. Per chunk: download the
# archives that still lack a result pickle (GitHub, 4 in parallel) into a scratch folder outside
# the repo, run extract_games.py under the machine-wide heavy-job lock, delete the archives.
#   usage: run_extract.sh <game list> [chunk size, default 30]
# Needs gh (authenticated), 7z, network. Env: TRACKING_DATA_DIR (scratch Tracking_Data with an
# events/ folder holding the games' pbp CSVs), OUT (pickle folder), MIN_FREE_GB (default 3).
set -u
list="${1:?usage: $0 <game list> [chunk]}"; chunk="${2:-30}"
here="$(cd "$(dirname "$0")" && pwd)"; cd "$here/.."
export TRACKING_DATA_DIR="${TRACKING_DATA_DIR:-/private/tmp/ws5b/Tracking_Data}"
OUT="${OUT:-/private/tmp/ws5b/states}"; MIN_FREE_GB="${MIN_FREE_GB:-3}"
PY="${PYTHON:-/Users/yash/dev/basketballPlayOptimisation/.venv/bin/python}"
GH_REPO="${GH_REPO:-Yash-Sangtani/basketballPlayOptimisation}"; GH_REF="${GH_REF:-cloud-full-batch}"
GL="$TRACKING_DATA_DIR/game logs"; mkdir -p "$GL" "$OUT"
free_gb() { df -g / | awk 'NR==2{print $4}'; }
fetch_one() {
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
for c in "$work"/chunk_*; do
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then echo "STOP: only $(free_gb) GB free (< $MIN_FREE_GB)." >&2; exit 1; fi
    : > "$c.need"; : > "$c.run"
    while read -r g; do
        [ -s "$OUT/$g.pkl" ] && continue
        echo "$g" >> "$c.run"
        [ -s "$GL/$g.7z" ] || echo "$g" >> "$c.need"
    done < "$c"
    echo "[chunk $(basename "$c")] $(wc -l < "$c.run" | tr -d ' ') to extract, $(wc -l < "$c.need" | tr -d ' ') to download; free disk $(free_gb) GB"
    [ -s "$c.run" ] || continue
    [ -s "$c.need" ] && xargs -P4 -I{} bash -c 'fetch_one {}' < "$c.need"
    # only games whose archive is present
    : > "$c.go"; while read -r g; do [ -s "$GL/$g.7z" ] && echo "$g" >> "$c.go"; done < "$c.run"
    [ -s "$c.go" ] && "$here/../heavy.sh" "$PY" -W ignore "$here/extract_games.py" "$c.go" "$OUT"
    while read -r g; do rm -f "$GL/$g.7z"; done < "$c.need"
done
echo "done: $(ls "$OUT"/*.pkl 2>/dev/null | wc -l | tr -d ' ') pickles in $OUT"
