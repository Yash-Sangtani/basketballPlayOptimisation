#!/bin/sh
# Copies games out of git history into a local Tracking_Data folder that is NOT
# inside an iCloud-synced directory (iCloud "Optimize Mac Storage" turns the
# working-tree .7z/CSV files into empty placeholders; git always has the bytes).
#
# Usage:  ./fetch_games.sh <game list> [dest]        run from anywhere in the repo
#   <game list>  lines like MM.DD.YYYY.AWAY.at.HOME[.7z]  (e.g. ../Shots/allgames.txt)
#   [dest]       default $HOME/.cache/basketballPlayOptimisation/Tracking_Data
# Then:   export TRACKING_DATA_DIR=<dest>   before running passtracking.py.
# Safe to re-run: existing files are skipped. Needs no network.
set -e
list="$1"; dest="${2:-$HOME/.cache/basketballPlayOptimisation/Tracking_Data}"
[ -n "$list" ] || { echo "Usage: $0 <game list> [dest]" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
mkdir -p "$dest/game logs" "$dest/events"
if [ ! -f "$dest/events/.complete" ]; then
    git archive HEAD Tracking_Data/events | tar -x -C "$dest" --strip-components=1
    touch "$dest/events/.complete"
fi
n=0
while IFS= read -r line; do
    name="${line%.7z}"; name="$(printf '%s' "$name" | tr -d '\r')"
    [ -n "$name" ] || continue
    out="$dest/game logs/$name.7z"
    [ -s "$out" ] && continue
    git show "HEAD:Tracking_Data/game logs/$name.7z" > "$out.tmp" && mv "$out.tmp" "$out"
    n=$((n + 1))
done < "$list"
echo "copied $n new archive(s) into $dest"
echo "next: export TRACKING_DATA_DIR=\"$dest\""
