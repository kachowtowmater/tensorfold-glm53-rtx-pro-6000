#!/usr/bin/env bash
# Run tfprobe2 only when no other request is running; print server-side tok/s + acceptance for the 4 probes.
# Usage: PORT=8090 LOG=~/tf/recipe/logs/current/rank0.log ./quietprobe.sh
PORT="${PORT:-8090}"; LOG="${LOG:-$HOME/tf/recipe/logs/current/rank0.log}"
idle(){ for i in $(seq 1 60); do r=$(curl -s -m 3 "localhost:$PORT/metrics" | awk '/^tensorfold:requests_running /{print $2}'); [ "${r:-1}" = "0" ] && return 0; sleep 2; done; return 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
idle; PORT=$PORT python3 "$HERE/tfprobe2.py" >/dev/null 2>&1   # warm
idle; PORT=$PORT python3 "$HERE/tfprobe2.py" >/dev/null 2>&1
grep "done req-" "$LOG" | grep -E "prompt=(26|33|39) " | tail -4 | grep -oE "tok/s=[0-9.]+|accepted=[0-9/]+" | paste - - | tr "\n" " "; echo
