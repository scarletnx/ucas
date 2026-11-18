#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# A small set of tests whose expected output is portable across the C
# libraries: the exact numeric lines of the full tests/*.out suite are not
# (last-digit libm differences, SAT conflict counts), so that suite stays on
# the BSD make/FreeBSD path.  This is what CTest and the Linux workflow run.
set -eu

bin=${1:?usage: tools/smoke.sh PATH/TO/ucas}
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tmp=$(mktemp -d 2>/dev/null || mktemp -d -t ucas-smoke)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

fail=0

"$bin" --sat-selftest

for n in cost find; do
	if (cd "$root/tests" && UCAS_SPIRV_CROSS=none "$bin" "$n.ucas") \
	    > "$tmp/$n.got" 2>&1 &&
	    diff -u "$root/tests/$n.out" "$tmp/$n.got"; then
		echo "ok   $n"
	else
		echo "FAIL $n"
		fail=1
	fi
done

if (cd "$root/tests/json" && UCAS_SPIRV_CROSS=none "$bin" --json \
    --files none < schema.jsonl) > "$tmp/schema.got" 2>&1 &&
    diff -u "$root/tests/json/schema.out" "$tmp/schema.got"; then
	echo "ok   json/schema"
else
	echo "FAIL json/schema"
	fail=1
fi

exit "$fail"
