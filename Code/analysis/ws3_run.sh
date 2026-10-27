#!/bin/bash
# Chunked, resumable driver for ws3_extract.py (modelled on ../run_chunked.sh): per chunk, download the
# missing archives from GitHub into $TRACKING_DATA_DIR/game logs (outside the repo), run the extraction
# under the machine-wide heavy lock, delete the archives it downloaded.
# Usage: ws3_run.sh <game list> [chunk size, default 50]   (run from anywhere)
set -u
list="${1:?Usage: $0 <game list> [chunk]}"; chunk="${2:-50}"
here="$(cd "$(dirname "$0")" && pwd)"; code="$here/.."; cd "$code"
export TRACKING_DATA_DIR="${TRACKING_DATA_DIR:-/private/tmp/ws3/Tracking_Data}"
export WS3_OUT="${WS3_OUT:-/private/tmp/ws3/out}"
GH_REPO="${GH_REPO:-Yash-Sangtani/basketballPlayOptimisation}"; GH_REF="${GH_REF:-cloud-full-batch}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"; PY="${PYTHON:-/Users/yash/dev/basketballPlayOptimisation/.venv/bin/python}"
GL="$TRACKING_DATA_DIR/game logs"; mkdir -p "$GL" "$WS3_OUT"
free_gb() { df -g / | awk 'NR==2{print $4}'; }
fetch_one() {
    out="$GL/$1.7z"
    if gh api -H "Accept: application/vnd.github.raw" \
        "repos/$GH_REPO/contents/Tracking_Data/game%20logs/$1.7z?ref=$GH_REF" > "$out.tmp" 2>/dev/null && [ -s "$out.tmp" ]; then
        mv "$out.tmp" "$out"
    else rm -f "$out.tmp"; echo "download failed: $1" >&2; fi
}
export -f fetch_one; export GL GH_REPO GH_REF
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
tr -d '\r' < "$list" | grep -v '^$' | sed 's/\.7z$//' > "$work/all.txt"
split -l "$chunk" "$work/all.txt" "$work/chunk_"
n=$(ls "$work"/chunk_* | wc -l | tr -d ' '); i=0
for c in "$work"/chunk_*; do
    i=$((i + 1))
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then echo "STOP: only $(free_gb) GB free" >&2; exit 1; fi
    "$PY" analysis/ws3_extract.py "$c" --print-todo > "$c.todo"
    echo "[chunk $i/$n] $(wc -l < "$c.todo" | tr -d ' ') of $(wc -l < "$c" | tr -d ' ') to do; free disk $(free_gb) GB"
    [ -s "$c.todo" ] || continue
    while read -r g; do [ -s "$GL/$g.7z" ] || echo "$g" >> "$c.need"; done < "$c.todo"
    [ -s "$c.need" ] && xargs -P4 -I{} bash -c 'fetch_one {}' < "$c.need"
    while read -r g; do [ -s "$GL/$g.7z" ] && echo "$g" >> "$c.run"; done < "$c.todo"
    [ -s "$c.run" ] && "$code/heavy.sh" "$PY" analysis/ws3_extract.py "$c.run" \
        --check-baseline /Users/yash/dev/basketballPlayOptimisation/Tracking_Data/pass_features
    [ -s "$c.need" ] && while read -r g; do rm -f "$GL/$g.7z"; done < "$c.need"
done
echo "done: $(ls "$WS3_OUT"/*.pkl 2>/dev/null | wc -l) game files in $WS3_OUT"
