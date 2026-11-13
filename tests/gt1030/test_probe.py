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


def noise_words(rng, n):
    """snx-shaderperf's fill=noise: floats k / 2^24, k uniform below 2^24, as their bits"""
    import numpy as np
    k = rng.integers(0, 1 << 24, size=n, dtype=np.uint64)
    return (k.astype(np.float64) / float(1 << 24)).astype(np.float32).view(np.uint32).astype(np.uint64)


def walk_table(p, rng):
    """the table a walking step reads, as an array, and its size"""
    import numpy as np
    text = p['step']
    if 'tab[' in text:
        vals = [int(v, 16) for v in re.findall(r'0x([0-9a-f]{8})u', p['decls'])]
        return np.array(vals, dtype=np.uint64)
    for fn, data in p.get('files', {}).items():
        if fn in ' '.join(p['args']):
            return np.frombuffer(data, dtype='<u4').astype(np.uint64)
    # inb.data, or a texture made by pattern=noise: the buffer's noise
    return noise_words(rng, 1 << 16)


@check
def table_walks_keep_lanes_apart():
    """a step whose table index is moved by the value it loads keeps the warp's lanes on as many entries
    as they start on, through the loop of both builds: a table that merges them measures fewer and fewer
    addresses a warp, and its A and 2A builds read differently"""
    import numpy as np
    rng = np.random.default_rng(3)
    M = (1 << 32) - 1
    bad, seen = [], 0
    for p in steps():
        text = p['step'] or ''
        m = re.search(r'\{a\}( \+ \{u\}u)?\)*\s*&\s*(\d+)u\]?\)*', text.split(' + ', 1)[1]) \
            if text.startswith('{a} + ') else None
        if not m or 'ib' in text:
            continue
        seen += 1
        tab = walk_table(p, rng)
        mask = len(tab) - 1 if 'tab[' in text else int(m.group(2))
        warps = 64
        s = noise_words(rng, 32 * warps) ^ 1
        for U in (p['A'], 2 * p['A']):
            a = (s * 0x9e3779b1 + 1) & M
            start = None
            for i in range(111):
                for u in range(U):
                    idx = (a + (u if m.group(1) else 0)) & mask
                    if start is None:
                        start = [len(set(idx[w * 32:(w + 1) * 32].tolist())) for w in range(warps)]
                    a = (a + tab[idx]) & M
            end = [len(set(idx[w * 32:(w + 1) * 32].tolist())) for w in range(warps)]
            if end != start:
                bad.append('%s (A=%d: %.1f entries a warp at the start, %.1f at the end)' % (
                    p['name'], U, sum(start) / warps, sum(end) / warps))
                break
    assert seen >= 5, 'found %d walking steps, expected the tables of #166 and #168' % seen
    assert not bad, 'walks that merge lanes: %s' % '; '.join(bad)


@check
def constant_tables_bracket_the_refusal():
    """the card refused a 64 KiB constant array and took a 1 KiB one: #166 needs the sizes between"""
    kib = sorted(int(m.group(1)) for p in steps() for m in [re.fullmatch(r'ctab(\d+)k_random', p['name'])] if m)
    between = [k for k in kib if 1 < k < 64]
    assert len(between) >= 2, 'constant tables of %s KiB: none or one between 1 and 64' % kib
    for k in kib:
        assert any(p['name'] == 'ssbo%dk_random' % k for p in steps()), 'no storage buffer beside ctab%dk' % k

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
