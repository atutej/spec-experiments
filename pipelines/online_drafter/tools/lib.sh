# Helpers shared by the online-drafter run scripts (source it). Everything a script starts is a process group recorded under $RUN/pids; cleanup kills only those groups
# (never pkill/pgrep patterns: they also match other people's and your own processes).
ROOT=/scratch/09749/atutej/marin_speculator
local_ip() { python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0])" 2>/dev/null || hostname -i | awk '{print $1}'; }
port_free() { python3 - "$1" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(1)
sys.exit(1 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 0)
PY
}
init_run() {   # init_run <new run dir>
    RUN=${1:?usage: <new run dir>}; [[ ! -e "$RUN" ]] || { echo "refusing: $RUN exists (use a new directory)" >&2; exit 1; }
    mkdir -p "$RUN/logs" "$RUN/pids"; echo "run dir: $RUN"
}
run_in() {   # run_in <conda env> <name> <env assignments...> -- <command...>   (own session, log in $RUN/logs/<name>.log, pid in $RUN/pids/<name>)
    local env=$1 name=$2; shift 2; local assigns=(); while [[ "$1" != "--" ]]; do assigns+=("$1"); shift; done; shift
    setsid env "${assigns[@]}" bash -c 'set +u; source '"$ROOT"'/spec-experiments/env.sh >/dev/null 2>&1; conda activate '"$env"'; set -u; unset TRANSFORMERS_CACHE; exec "$@"' _ "$@" > "$RUN/logs/$name.log" 2>&1 &
    echo $! > "$RUN/pids/$name"; echo "started $name (pid $!, env $env, log $RUN/logs/$name.log)"
}
alive() { [[ -f "$RUN/pids/$1" ]] && kill -0 "$(cat "$RUN/pids/$1")" 2>/dev/null; }
stop_one() {
    [[ -f "$RUN/pids/$1" ]] || return 0; local pg; pg=$(cat "$RUN/pids/$1")
    kill -TERM -- "-$pg" 2>/dev/null || true
    for _ in $(seq 20); do kill -0 -- "-$pg" 2>/dev/null || return 0; sleep 1; done
    kill -KILL -- "-$pg" 2>/dev/null || true
}
wait_http() {   # wait_http <url> <timeout s> [<name of a process that must stay alive>]
    local url=$1 limit=$2 who=${3:-} waited=0
    until curl -s -o /dev/null -f "$url"; do
        sleep 3; waited=$((waited + 3))
        [[ -n "$who" ]] && ! alive "$who" && { echo "ERROR: $who died while waiting for $url (see $RUN/logs/$who.log)" >&2; return 1; }
        [[ $waited -ge $limit ]] && { echo "ERROR: $url not up after ${limit}s" >&2; return 1; }
    done
    echo "$url is up after ${waited}s"
}
forever_watch() {   # forever_watch: the FOREVER=1 main loop of the run scripts; returns (non-zero) only when the trainer, server or client has died
    # After every successful draft sync, copy the trainer's sync dir ($RUN/out/draft_sync, overwritten in place, so it can be torn by a kill) to $RUN/drafts/step_N through a
    # temp dir + rename; keep the last KEEP_DRAFTS and every 1000th step.
    mkdir -p "$RUN/drafts"; local seen=0 n step
    while alive training && alive server && alive client; do
        n=$(grep -ac "External sglang weight sync: {'success': True" "$RUN/logs/training.log" 2>/dev/null || true)
        if [[ ${n:-0} -gt $seen ]]; then   # a sync finished: the trainer does not touch the sync dir again until the next interval
            step=$(grep -a "updating external sglang" "$RUN/logs/training.log" | tail -1 | sed -E 's/.*Step ([0-9]+):.*/\1/')
            rm -rf "$RUN/drafts/.tmp" && cp -r "$RUN/out/draft_sync" "$RUN/drafts/.tmp" && mv "$RUN/drafts/.tmp" "$RUN/drafts/step_$(printf %08d "$step")" && seen=$n
            ls -d "$RUN"/drafts/step_* | head -n -"${KEEP_DRAFTS:-5}" | grep -v '000$' | xargs -r rm -rf
        fi
        sleep 30
    done
    echo "ERROR: a process stopped (training $(alive training && echo up || echo down), server $(alive server && echo up || echo down), client $(alive client && echo up || echo down)); stopping" >&2
    return 1
}
