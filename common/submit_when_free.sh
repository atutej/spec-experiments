#!/bin/bash
# Wait until you have fewer than MAX jobs submitted in a partition, then run a command (normally a submit script). Slurm itself cannot do this: a
# submission over the QOS limit (gb: 3 submitted jobs per user, running + pending, idev sessions included) is REJECTED at once, not queued.
#
#   bash common/submit_when_free.sh [--max 3] [--partition gb] [--interval 60] [--timeout SECONDS] -- COMMAND [ARGS...]
#   e.g. AFTER='1069270?1069271' bash common/submit_when_free.sh -- bash <experiment>/vista/submit_chain.sh prepare,train
#
# Run it on a LOGIN node (sbatch is refused elsewhere), inside tmux/screen or with `nohup ... &`: it ends with the session that runs it. It polls
# `squeue` once per --interval seconds (light), prints a line when the count changes, and runs the command once, as soon as a slot is free (the
# command's exit status is the script's). --timeout gives up (exit 1) after that many seconds; default: wait forever.
# Note: a login-node process that runs for hours is tolerated only because this one is nearly idle; do not use it for anything heavy.
set -uo pipefail
MAX=3 PARTITION=gb INTERVAL=60 TIMEOUT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
    --max) MAX=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --interval) INTERVAL=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option: $1 (use -- before the command)" >&2; exit 2 ;;
    esac
done
[[ $# -gt 0 ]] || { echo "no command given after --" >&2; exit 2; }
if [[ -z "${SKIP_LOGIN_CHECK:-}" && "$(type -t sbatch)" != file ]]; then
    echo "sbatch is not a real command here (a compute node?): run this on a login node" >&2; exit 2
fi
start=$SECONDS last=-1
while true; do
    n=$(squeue -u "$USER" -h -p "$PARTITION" 2>/dev/null | wc -l)
    if (( n < MAX )); then
        echo "$(date +%H:%M:%S) $n of $MAX $PARTITION slots used: running: $*"
        exec "$@"
    fi
    if (( n != last )); then echo "$(date +%H:%M:%S) $n of $MAX $PARTITION slots used: waiting (checking every ${INTERVAL}s)"; last=$n; fi
    if (( TIMEOUT > 0 && SECONDS - start >= TIMEOUT )); then echo "gave up after ${TIMEOUT}s" >&2; exit 1; fi
    sleep "$INTERVAL"
done
