#!/bin/sh
# Runs a command from Code/ and saves its combined stdout/stderr to
# ../logs/<UTC timestamp>_<name>.log, with a header recording exactly
# what was run, from where, and when.
#
# Usage: ./run_logged.sh <short-name> <command> [args...]
# Example: ./run_logged.sh regenerate_fixed python3 _regenerate_fixed.py

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <short-name> <command> [args...]" >&2
    exit 1
fi

name="$1"
shift

script_dir="$(cd "$(dirname "$0")" && pwd)"
log_dir="$script_dir/../logs"
mkdir -p "$log_dir"

timestamp="$(date -u +%Y%m%d-%H%M%S)"
logfile="$log_dir/${timestamp}_${name}.log"

{
    echo "# command : $*"
    echo "# cwd     : $(pwd)"
    echo "# started : $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "# ---"
    "$@"
    status=$?
    echo "# ---"
    echo "# finished: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "# exit code: $status"
} > "$logfile" 2>&1

cat "$logfile"
echo "(log saved to $logfile)"
exit "$status"
