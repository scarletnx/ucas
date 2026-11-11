#!/bin/sh
# ucas.sh [ARG...]: run ucas with ARG, from whichever binary this machine has; `ucas.sh which` prints the binary.
# The binary, first found: $UCAS; `ucas` on PATH; UCAS_BIN in the nearest .claude/skills.env above the working
# folder; the ucas this skill sits in (its repository's ./ucas, after make). None: it says so, exit 2.
set -u
here=$(cd "$(dirname "$0")" && pwd)

from_env() {  # UCAS_BIN of the nearest .claude/skills.env, ~/ expanded
    d=$(pwd)
    while :; do
        f=$d/.claude/skills.env
        if [ -f "$f" ]; then
            v=$(sed -n 's/^UCAS_BIN=//p' "$f" | tail -n 1)
            case $v in "~/"*) v=$HOME/${v#\~/} ;; esac
            [ -n "$v" ] && { echo "$v"; return 0; }
        fi
        up=$(dirname "$d") || return 1
        # at the root, or a dirname that does not move: nowhere left to look
        [ -n "$up" ] && [ "$up" != "$d" ] || return 1
        d=$up
    done
}

bin=
if [ -n "${UCAS:-}" ]; then
    bin=$UCAS
elif command -v ucas >/dev/null 2>&1; then
    bin=$(command -v ucas)
elif v=$(from_env); then
    bin=$v
elif [ -x "$here/../../../ucas" ]; then
    bin=$(cd "$here/../../.." && pwd)/ucas
fi
if [ -z "$bin" ] || [ ! -x "$bin" ]; then
    echo "ucas.sh: no ucas binary${bin:+ at $bin}: build it (make in the ucas repository), or set UCAS, or UCAS_BIN in .claude/skills.env" >&2
    exit 2
fi
if [ "${1:-}" = which ]; then
    echo "$bin"
    exit 0
fi
exec "$bin" "$@"
