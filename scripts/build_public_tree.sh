#!/bin/bash
# Build the curated PUBLIC copy of this (private) working repo into a directory.
#   scripts/build_public_tree.sh <output dir>
# Includes tracked files only. Leaves out: working notes (CLAUDE.md, HANDOFF.md), the raw data (Tracking_Data/),
# machine-specific helper scripts, and anything not under version control. Rewrites internal references in the copy.
set -euo pipefail
out="${1:?usage: $0 <output dir>}"
cd "$(dirname "$0")/.."
rm -rf "$out"; mkdir -p "$out"
EXCLUDE='^(Tracking_Data/|CLAUDE\.md$|HANDOFF\.md$|Code/analysis/ws3_run\.sh$|Code/value/run_extract\.sh$|Code/run_chunked\.sh$|Code/fetch_games\.sh$|scripts/publish_public\.sh$|scripts/build_public_tree\.sh$|logs/|\.claude/)'
git ls-files | grep -v -E "$EXCLUDE" | while IFS= read -r f; do
    mkdir -p "$out/$(dirname "$f")"; cp -p "$f" "$out/$f"
done
# rewrite internal references in text files of the copy
find "$out" -type f \( -name '*.md' -o -name '*.py' -o -name '*.txt' \) -print0 | xargs -0 sed -i \
    -e 's#/Users/yash/dev/wt/ws5a-shots#<repo>#g' -e 's#/Users/yash/dev/basketballPlayOptimisation#<repo>#g' -e 's#/private/tmp/[A-Za-z0-9_/.]*#<scratch>#g' \
    -e 's#(see CLAUDE\.md for the outcome)#(see reports/ws4-detection.md for the outcome)#' \
    -e 's#see CLAUDE\.md "Data facts"#see reports/ws4-detection.md#' \
    -e 's#no CLAUDE\.md edit#no project-notes edit#' -e 's#CLAUDE\.md saw#the project notes saw#' \
    -e 's#named in CLAUDE\.md#named in the project notes#' -e 's#the HANDOFF numbers#the earlier handoff numbers#'
# the working repo's raw data does not ship: say so in the copy's README
cat >> "$out/README.md" <<'EOF'

## Data

The raw SportVU archives (`Tracking_Data/game logs/*.7z`) and play-by-play CSVs (`Tracking_Data/events/<game_id>.csv`) are **not** part of this repository.
Put your own copy under `Tracking_Data/` with that layout to run the extraction scripts. The trained difficulty models, the frozen game splits and every report are included.
EOF
echo "built $(find "$out" -type f | wc -l) files, $(du -sh "$out" | cut -f1) in $out"
