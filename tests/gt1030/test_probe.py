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



def split_args(text):
    """the arguments of a call's text, split at the commas outside parentheses"""
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
            continue
        depth += ch == '('
        depth -= ch == ')'
        cur += ch
    return out + [cur]


def glsl_value(expr, env):
    """a bound's GLSL text (chain values, masks, abs, negation) evaluated in Python"""
    py = re.sub(r'\{(\w)\}', r'\1', expr)
    py = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', py)
    return eval(py, {'abs': abs}, dict(env))


@check
def clamp_bounds_ordered():
    """clamp(x, lo, hi) is undefined for lo > hi: every clamp of chain values keeps lo <= hi"""
    import random
    rnd = random.Random(7)
    bad = []
    for p in steps():
        m = re.search(r'clamp\((.*)\)', p['step'] or '')
        if not m or not re.search(r'\{[a-d]\}', m.group(1)):
            continue
        x, lo, hi = split_args(m.group(1))[:3]
        for _ in range(1000):
            if p['ty'] == 'float':
                env = {k: rnd.uniform(-4.0, 4.0) for k in 'abcd'}
            else:
                env = {k: rnd.getrandbits(32) for k in 'abcd'}
            if glsl_value(lo, env) > glsl_value(hi, env):
                bad.append('%s (lo %s, hi %s)' % (p['name'], lo.strip(), hi.strip()))
                break
    assert not bad, 'clamps with a low bound above the high one: %s' % ', '.join(bad)


def calls(text, prefix):
    """(name, argument text) of every call whose name starts with prefix"""
    out = []
    for m in re.finditer(r'\b(%s\w*)\(' % prefix, text):
        depth, j = 1, m.end()
        while depth and j < len(text):
            depth += text[j] == '('
            depth -= text[j] == ')'
            j += 1
        out.append((m.group(1), text[m.end():j - 1]))
    return out


@check
def subgroup_operands_vary():
    """a subgroup operation none of whose operands is a chain value takes the same input at every step, and
    the driver computes it once"""
    bad = [p['name'] for p in steps() for name, args in calls((p['step'] or '') + (p['body'] or ''), 'subgroup')
           if '{' not in args]
    assert not bad, 'subgroup operations with no chain value among their operands: %s' % ', '.join(bad)

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
