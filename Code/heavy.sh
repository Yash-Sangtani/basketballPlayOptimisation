#!/bin/sh
# Run a command while holding the machine-wide "heavy job" lock. This Mac has 8 GB RAM, so only
# one heavy job (batch extraction, TensorFlow training, large model fits) should run at a time.
# Light work (editing, reading, small scripts) does not need the lock.
#
# Usage: ./heavy.sh <command> [args...]
# Waits for the lock if another heavy job holds it. A lock left by a killed job is detected and
# cleared. Override the lock path with HEAVY_LOCK.
LOCK="${HEAVY_LOCK:-/tmp/basketball_heavy.lock}"
while ! mkdir "$LOCK" 2>/dev/null; do
    holder="$(cat "$LOCK/pid" 2>/dev/null)"
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
        rm -rf "$LOCK"; continue          # holder died without cleaning up
    fi
    sleep 5
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
trap 'exit 130' INT TERM
"$@"
