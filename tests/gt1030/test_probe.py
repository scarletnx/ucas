#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks of probe.py that need no GPU: what its steps compute, the tables it walks, the analysis of
the values it reads back.  `./test_probe.py`; it exits 1 and names each check that fails."""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import probe as PR     # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def steps():
    return [p for p in PR.P if p['kind'] == 'step']


# subgroup operations whose result is the same in every lane
UNIFORM = ('subgroupBroadcast', 'subgroupBroadcastFirst', 'subgroupAdd', 'subgroupMul', 'subgroupMin',
           'subgroupMax', 'subgroupAnd', 'subgroupOr', 'subgroupXor', 'subgroupBallot', 'subgroupAll',
           'subgroupAny', 'subgroupAllEqual')


def outer_call(expr):
    """the function a whole expression is one call of, else None"""
    m = re.fullmatch(r'\s*(\w+)\((.*)\)\s*', expr)
    if not m:
        return None
    depth = 0
    for ch in m.group(2):
        depth += ch == '('
        depth -= ch == ')'
        if depth < 0:
            return None    # f(x) + g(y): the first call closes before the end
    return m.group(1)


@check
def uniform_steps_fold():
    """a step that is a subgroup operation with a uniform result alone feeds the same value to every lane
    of the next step, where the operation is the identity: the driver keeps one round of the ring"""
    bad = [p['name'] for p in steps() if p['step'] and outer_call(p['step']) in UNIFORM]
    assert not bad, 'steps whose result is uniform alone: %s' % ', '.join(bad)


def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print('ok    %s' % fn.__name__)
        except AssertionError as e:
            failed += 1
            print('FAIL  %s: %s' % (fn.__name__, e))
    print('%d of %d checks failed' % (failed, len(CHECKS)) if failed else 'all %d checks pass' % len(CHECKS))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
