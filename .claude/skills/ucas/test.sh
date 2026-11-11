#!/bin/sh
# The ucas skill's test: the recipes of SKILL.md run against the binary ucas.sh finds, each answer checked, and
# ucas.sh's refusal with no binary. Offline, a few seconds. Exit 0 all passed, 1 a check failed.
set -u
here=$(cd "$(dirname "$0")" && pwd)
U=$here/ucas.sh
t=$(mktemp -d "${TMPDIR:-/tmp}/ucas-skill-test.XXXXXX") || exit 1
trap 'rm -rf "$t"' EXIT INT TERM
cd "$t" || exit 1
fails=0
check() {  # NAME OUTPUT PATTERN: the output must hold the pattern (grep -E)
    if printf '%s\n' "$2" | grep -Eq -- "$3"; then echo "ok   $1"; else echo "FAIL $1"; printf '%s\n' "$2" | sed 's/^/     | /' | head -8; fails=$((fails + 1)); fi
}

bin=$("$U" which) || { echo "FAIL no ucas binary (see the line above)"; exit 1; }
echo "ucas: $bin"

out=$("$U" -e 'avg(x: u32, y: u32) = (x & y) + ((x xor y) >> 1)' \
    -e 'avg2(x: u32, y: u32) = u32((u64(x) + u64(y)) >> 1)' -e 'equiv avg avg2' 2>&1)
check "equiv: two forms of a mean proved equal by SAT" "$out" 'avg and avg2 are equal .*proved by SAT'

out=$("$U" -e 'avg2(x: u32, y: u32) = u32((u64(x) + u64(y)) >> 1)' -e 'bad(x: u32, y: u32) = (x + y) >> 1' \
    -e 'equiv avg2 bad' 2>&1)
check "equiv: a wrapping form differs, with an input" "$out" 'differ at .* e\.g\. x=[0-9]+, y=[0-9]+'

out=$("$U" -e 'r(c: u32 in 0..65535) = (c * 255 + 32895) >> 16' -e 'range r' 2>&1)
check "range: exact over every point, the smaller type" "$out" '0\.\.255'
check "range: exhaustive" "$out" 'over all 65536 points'

out=$("$U" -e 'bad(x: u32, y: u32) = (x + y) >> 1' -e 'wrap bad' 2>&1)
check "wrap: an unsigned wrap with its witness" "$out" 'nuw fails: unsigned wrap'

out=$("$U" -e 'f(x: i32) = (x * 2) / 2' -e 'domain f' 2>&1)
check "domain: C's signed overflow assumption" "$out" 'C assumes 2x does not overflow i32'

out=$("$U" -e 'cost (x * 3) >> 1 model zen4' 2>&1)
check "cost: under a CPU model" "$out" 'under zen4'

out=$("$U" --json -e '{"id": 1, "cmd": "equiv", "args": {"f": {"name": "avg", "body": "(x & y) + ((x xor y) >> 1)", "params": [{"name": "x", "type": "u32"}, {"name": "y", "type": "u32"}]}, "g": "u32((u64(x) + u64(y)) >> 1)"}}' 2>&1)
check "json: ok, with the proof's kind" "$out" '"ok": true.*"proof": \{"kind": "sat"'

# the system's tools stay on PATH (ucas.sh needs sed and dirname), but no folder that holds a ucas
p=; for dir in /usr/local/bin /usr/bin /bin; do [ -x "$dir/ucas" ] || p=${p:+$p:}$dir; done
out=$(cd "$t" && env -u UCAS PATH="$p" /bin/sh "$U" which 2>&1); rc=$?
# with no UCAS, no ucas on PATH and no skills.env above $t, only the repository's own ./ucas can answer
if [ -x "$here/../../../ucas" ]; then
    check "which: the repository's ./ucas when nothing else names one" "$out" '/ucas$'
else
    check "which: no binary is refused, saying how to get one" "$out:$rc" 'no ucas binary.*:2$'
fi

out=$(env UCAS=/nonexistent/ucas /bin/sh "$U" which 2>&1); rc=$?
check "which: a UCAS that is not there is refused, naming it" "$out:$rc" 'no ucas binary at /nonexistent/ucas.*:2$'

if [ "$fails" -eq 0 ]; then echo "all passed"; else echo "$fails failed"; exit 1; fi
