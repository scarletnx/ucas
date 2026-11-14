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
            return np.frombuffer(data() if callable(data) else data, dtype='<u4').astype(np.uint64)
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
        if not m or 'ib' in text or int(m.group(2)) >= 1 << 16:
            continue    # past 256 KiB: perm_table as below, and the walk takes minutes in numpy
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


def card(bits, mode, quant=None):
    """the results a filter with a weight grid of bits (None: exact weights) would give the probe's reads;
    quant: the result rounded to that many fractional bits, as a fixed-point blend"""
    from fractions import Fraction
    out = []
    for t, x, u in PR.filter_coords():
        a, b = Fraction(PR.RAMP[t], 255), Fraction(PR.RAMP[t + 1], 255)
        w = PR.grid_weight(x, bits, mode) if bits else Fraction(x)
        v = a + w * (b - a)
        if quant:
            v = Fraction(round(v * 2 ** quant), 2 ** quant)
        out.append(PR.round_f32(v, 'even'))
    return out


@check
def filter_grid_from_its_reads():
    """the filter analysis finds the weight grid and its rounding from what the card returns, for grids
    coarser and finer than the coarse sweep's 4096 points, and tells an exact blend from a fixed-point one"""
    for bits, mode in ((8, 'truncate'), (8, 'nearest'), (12, 'truncate'), (14, 'nearest'), (16, 'truncate')):
        r = PR.filter_analysis(card(bits, mode))
        assert (r['grid_bits'], r['weights_round']) == (bits, mode), \
            'a %d-bit grid, %s: read as %s bits, %s (%s)' % (bits, mode, r['grid_bits'], r['weights_round'],
                                                              r.get('grid_note', ''))
        assert r['blend_exact'], 'a %d-bit grid, %s: an exact blend read as off by %g' % (bits, mode,
                                                                                         r['max_error_abs'])
    r = PR.filter_analysis(card(12, 'truncate', quant=16))
    assert r['grid_bits'] == 12 and not r['blend_exact'] and 0 < r['max_error_abs'] <= 2.0 ** -17, \
        'a blend in 16 fractional bits: %s' % r
    r = PR.filter_analysis(card(None, None))
    assert r['grid_bits'] is None and 'every sample' in r['grid_note'], 'exact weights: %s' % r


@check
def tex_within_only_from_an_exact_blend():
    """tex_within is ulps of the result: written when every result is the f32 nearest the weighted sum at its
    grid weight, not from an absolute error, which no count of ulps bounds near 0"""
    st = dict(subtexel=8, device='a card')
    for res_fw, want in ((PR.filter_analysis(card(12, 'truncate')), 'tex_within: 0.5'),
                         (PR.filter_analysis(card(12, 'truncate', quant=16)), None)):
        lines = [ln for ln, cm in PR.model_lines(dict(filter_weights=res_fw, subtexel_bits=st))[0] if ln]
        tw = [ln for ln in lines if ln.startswith('tex_within')]
        assert tw == ([want] if want else []), 'tex_within lines %s, expected %s' % (tw, want)


@check
def uncached_reads_scatter():
    """a read meant out of the cache moves each step's coordinate by the step ({u}) and not through
    textureLodOffset, whose offsets of at most 8 texels keep the step's reads in the line the first brought"""
    bad = [p['name'] for p in steps() if p['name'].endswith('_uncached') and
           ('textureLodOffset' in p['step'] or '{u}' not in p['step'])]
    assert not bad, 'reads meant out of the cache that stay in a line: %s' % ', '.join(bad)


@check
def units_convert_by_their_own_fadd():
    """a draw or a dispatch timed whole is in fadd steps by a fadd timed just before it, not by the fadd of
    whichever step ran last: the SM clock moves between runs"""
    import argparse
    saved = PR.A, PR.OUT, PR.FADD_NS, PR.sp, PR.compile_glsl
    fadd = PR.fadd_entry()
    per_step = 1e-3                            # ns a lane-step, the fadd of this test's card

    def fake_sp(args, log, timeout=None):
        shaders = [a for a in args if a.endswith('.spv')]
        iters = next((int(a.split(':')[1]) for a in args if a.startswith('iters=')), 0)
        if shaders and shaders[0].endswith(fadd['files']['a']):
            n = 65536 * iters * fadd['chains']
            ts = [n * fadd['unroll'][0] * per_step, n * fadd['unroll'][1] * per_step]
        else:
            ts = [2870.0]
        return dict(shaders=[dict(time_ns_per_rep=dict(median=t, median_lo=t, median_hi=t),
                                  invocations_per_rep=65536) for t in ts])
    try:
        PR.A = argparse.Namespace(quick=False, runs=20, timeout=600)
        PR.OUT = '/nonexistent'
        PR.FADD_NS = 4.4e-3                    # a step run's fadd at another clock
        PR.sp = fake_sp
        PR.compile_glsl = lambda name, src, stage: name + '.spv'
        r = PR.unit_draw_empty(next(p for p in PR.P if p['name'] == 'draw_empty'))
    finally:
        PR.A, PR.OUT, PR.FADD_NS, PR.sp, PR.compile_glsl = saved
    want = 2870.0 / per_step
    assert abs(r['fadd_steps'] - want) < 1e-6 * want, 'an empty draw of 2870 ns: %.6g fadd steps, %.6g by the ' \
        'fadd timed beside it' % (r['fadd_steps'], want)


def compiles(src, stage):
    """GLSL into SPIR-V that spirv-val takes, through pipes; the errors when not"""
    import subprocess
    g = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '--stdin', '-S', stage, '-o',
                        '/dev/stdout', '--quiet'], input=src.encode(), capture_output=True, timeout=300)
    if g.returncode:
        return (g.stdout + g.stderr).decode(errors='replace').strip().splitlines()[-3:]
    v = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', '-'], input=g.stdout, capture_output=True,
                       timeout=300)
    return v.stderr.decode(errors='replace').strip().splitlines()[-3:] if v.returncode else None


@check
def every_step_compiles():
    """every step's A build and every latency's U build is GLSL that glslangValidator compiles and spirv-val
    takes"""
    bad = []
    for p in PR.P:
        if p['kind'] == 'step':
            src = PR.G.gen_compute(dict(p), p['A'], False)
        elif p['kind'] == 'lat':
            src = PR.G.gen_compute(dict(p), PR.LAT_U, True) if p['how'] == 'alu' else PR.lat_shader(p, PR.LAT_U)
        else:
            continue
        err = compiles(src, 'comp')
        if err:
            bad.append('%s: %s' % (p['name'], ' / '.join(err)))
    assert not bad, '; '.join(bad)


@check
def u64_division_by_a_constant():
    """#18 and #19 compare a u64 division and remainder by a constant with their rewrites: both are probed"""
    have = {p['name']: p for p in steps()}
    for n in ('u64_udiv_const', 'u64_umod_const'):
        assert n in have and have[n]['ty'] == 'u64' and {18, 19} <= set(have[n]['issues']), \
            '%s: not a u64 step of #18 and #19' % n


def lats():
    return [p for p in PR.P if p.get('kind') == 'lat']


@check
def latencies_of_the_model_classes():
    """one warp's dependent chain for the classes the model prices: the ALU, the loads, the texture reads"""
    want = {'iadd', 'logic', 'shift', 'imul', 'fadd', 'fmul', 'fdiv', 'fsqrt', 'ftrans', 'conv', 'f64', 'load',
            'load_stream', 'load_shared', 'load_ubo', 'tex_fetch', 'tex_sample'}
    have = {p['cls'] for p in lats()}
    assert want <= have, 'no latency for %s' % ', '.join(sorted(want - have))


@check
def chases_follow_one_cycle():
    """a chase's table is one cycle through all its entries, so the chain never settles in a short loop
    that a cache holds; the texture's coordinates lead to the centre of the next entry"""
    import struct
    for p in lats():
        if p['how'] != 'chase' or p['n'] > 65536:
            continue
        for fn, data in p['files'].items():
            b = data() if callable(data) else data
            if fn.startswith('uv'):
                xy = struct.unpack('<%df' % (2 * p['n']), b)
                side = int(round(p['n'] ** 0.5))
                t = [int(xy[2 * j] * side) + side * int(xy[2 * j + 1] * side) for j in range(p['n'])]
            else:
                words = struct.unpack('<%dI' % (len(b) // 4), b)
                t = list(words[::len(words) // p['n']])
            j, seen = 0, set()
            while j not in seen:
                seen.add(j)
                j = t[j]
            assert len(seen) == p['n'], '%s (%s): a cycle of %d of %d entries' % (p['name'], fn, len(seen), p['n'])


@check
def only_takes_patterns():
    """--only lat_* picks every latency probe and nothing else"""
    got = {p['name'] for p in PR.choose('lat_*', None)}
    assert got == {p['name'] for p in lats()}, 'lat_* picked %s' % sorted(got)


@check
def every_unit_shader_compiles():
    """the unit and value probes' own shaders (CS_*, VS_*, FS_*) compile and pass spirv-val"""
    stage = {'CS': 'comp', 'VS': 'vert', 'FS': 'frag'}
    bad = []
    for name in dir(PR):
        if name[:3] in ('CS_', 'VS_', 'FS_') and isinstance(getattr(PR, name), str):
            err = compiles(getattr(PR, name), stage[name[:2]])
            if err:
                bad.append('%s: %s' % (name, ' / '.join(err)))
    assert not bad, '; '.join(bad)


@check
def cache_sweep_crosses_l2():
    """#65's cache size: random walks over footprints on both sides of GP108's 512 KiB L2, and the model line
    takes the last footprint below twice the smallest one's cost"""
    kib = sorted(p['kib'] for p in steps() if p['name'].startswith('cache_'))
    assert len([k for k in kib if k < 512]) >= 2 and len([k for k in kib if k > 512]) >= 2, 'footprints %s' % kib
    costs = dict(zip(kib, [8.0, 8.1, 9.0, 15.9, 30.0, 60.0, 80.0, 90.0][:len(kib)]))
    res = {p['name']: dict(cost=costs[p['kib']], err=0.01) for p in steps() if p['name'].startswith('cache_')}
    lines = [ln for ln, cm in PR.model_lines(res)[0] if ln.startswith('cache:')]
    want = 'cache: %d' % (max(k for k in kib if costs[k] < 2 * costs[kib[0]]) * 1024)
    assert lines == [want], 'cache lines %s, expected %s' % (lines, want)


@check
def what_the_tool_cannot_do_is_named():
    """#52's blits and #53's indirect dispatch: snx-shaderperf times neither, so the report names them"""
    have = {p['name']: p for p in PR.P}
    for n, i in (('blit_mip', 52), ('dispatch_indirect', 53)):
        assert n in have and have[n]['kind'] == 'absent' and i in have[n]['issues'], '%s: not named for #%d' % (n, i)
    for n, i in (('dispatch_empty', 53), ('export', 50)):
        assert n in have and i in have[n]['issues'], '%s: no probe for #%d' % (n, i)

def main():
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            print('ok    %s' % fn.__name__)
        except Exception as e:    # an assert, or probe.py without what the check reads
            failed += 1
            print('FAIL  %s: %s%s' % (fn.__name__, '' if isinstance(e, AssertionError) else
                                      type(e).__name__ + ': ', e))
    print('%d of %d checks failed' % (failed, len(CHECKS)) if failed else 'all %d checks pass' % len(CHECKS))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
