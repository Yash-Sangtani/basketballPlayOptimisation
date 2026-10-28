#!/bin/bash
# Publish the curated copy to the PUBLIC repo (a clean single-history repo: no notes, no raw data in any commit).
#   scripts/publish_public.sh [commit message]      needs a git remote URL in $PUBLIC_REMOTE (default: the repo below)
set -euo pipefail
PUBLIC_REMOTE="${PUBLIC_REMOTE:-https://github.com/Yash-Sangtani/basketball-pass-difficulty.git}"
here="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git clone -q "$PUBLIC_REMOTE" "$tmp/pub"
"$here/scripts/build_public_tree.sh" "$tmp/tree" >/dev/null
rsync -a --delete --exclude '.git' "$tmp/tree/" "$tmp/pub/"
cd "$tmp/pub"
git add -A
if git diff --cached --quiet; then echo "public repo already up to date"; exit 0; fi
git commit -q -m "${1:-Sync from working repo ($(git -C "$here" rev-parse --short HEAD))}"
git push -q origin HEAD
echo "published: $(git rev-parse --short HEAD)"
