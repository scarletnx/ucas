#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""The GT 1030 measurements ucas still waits for, in one run on the card.

Each probe answers one issue of the "calibrate later" list (gt1030: #115-#122, #144,
#145, #147, #156, #166, #168, #184-#188, #207-#212) with the method the issue names:

- step: one step written A and 2A times in a loop, both builds timed beside the
  corpus's fadd in one snx-shaderperf run (run.py's method); the step's cost is its
  time over fadd's, in the unit of the ucas model.
- unit: a draw or a dispatch timed as a whole (a point, a vertex, a dispatch); its time
  over the fadd lane-step of the step run just before it gives fadd steps.
- value: chosen inputs and the results read back (--check --dump): the filter's weight
  grid and precision, the blender's and the UNORM conversion's rounding, the BC4
  palette, REPEAT at large coordinates.
- info: vulkaninfo's subTexelPrecisionBits.

What snx-shaderperf cannot set up (logic ops, stencil operations, MIN/MAX and advanced
blending) is named in the report with what the probe needs, not skipped silently.

It writes to --out (default ~/ucas/work/gt1030/probe-TIME): probe.json (every number),
report.md (by issue: what was measured and what it sets), model.ucas (`model
gt1030_probed : gt1030 { ... }`, the prices and parameters the probes set, and the
measures that have no class yet as comments) and raw/ (each run's command, log and
JSON).  ucas reads model.ucas back at the end (--ucas, else ucas beside this repo or on
PATH) and the run says whether it took it: `ucas model.ucas`, then `use gt1030_probed`.

--quick runs every probe on a small grid with few samples: the numbers mean nothing,
it shows the run goes through (lavapipe: VK_ICD_FILENAMES=.../lvp_icd.x86_64.json).

Usage: probe.py [--sp PATH] [--out DIR] [--only NAME,...] [--issue N,...] [--runs N]
                [--tries N] [--quick] [--list] [--ucas PATH] [--timeout S]
"""
import argparse, json, math, os, random, re, shutil, struct, subprocess, sys, time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True     # no __pycache__ left in the corpus by the imports below
import gen as G     # noqa: E402  the corpus's shader text (a step written UNROLL times)
import run as R     # noqa: E402  its timing (snx-shaderperf, fadd beside) and the prelude

PRELUDE = R.PRELUDE
A = None            # the options
OUT = None          # the output folder
FADD_NS = None      # fadd's ns a lane-step, from the last fadd timed


class ProbeError(Exception):
    pass


# ---------------------------------------------------------------------------
# the probes

P = []


def probe(kind, name, issues, **kw):
    P.append(dict(kind=kind, name=name, issues=list(issues), **kw))


def step(name, issues, stp, ops, **kw):
    """a step of the corpus's shape (gen.py's t()), timed against fadd"""
    x = dict(step=stp, ops=ops, ty='uint', exts=[], decls='', pre='', post='', C=8, ring=None, iters=None,
             note='', sub='alu', A=8, wg=128, vkext=[], args=[], qual='', pre_iter='', body=None, prestep='',
             poststep='', files={})
    x.update(kw)
    if x['ring'] is None:
        x['ring'] = any(p in (x['step'] or x['body'] or '') for p in ('{b}', '{c}', '{d}'))
    x['latC'] = 1
    probe('step', name, issues, **x)


SG = ['sg']
TEX = 'layout(set = 0, binding = 5) uniform sampler2D tex;\n'
UTEX = 'layout(set = 0, binding = 5) uniform usampler2D utex;\n'
UV = 'vec2 uv = vec2(float(gid & 1023u) + 0.5, float((gid >> 10) + i * 16u) + 0.5) / 1024.0;'
TEXSTEP = '{a} + textureLodOffset(tex, uv, 0.0, ivec2({u} & 7, {u} >> 3)).x'


def texarg(fmt, size, filt, wrap='repeat'):
    return ['--bind', '0.5=format=%s,size=%s,pattern=noise,filter=%s,wrap=%s' % (fmt, size, filt, wrap)]


# --- #184, #185 (and the queue's 16-bit multiply): XMAD against IMUL
step('mul16_mad', [184], '({a} & 0xffffu) * ({b} & 0xffffu) + {c}', {'logic': 2, 'mul16': 1, 'iadd': 1},
     queue='FILA 109 (#10)', note='(a & 0xffff) * (b & 0xffff) + c: one XMAD on sm_61 when the masks and the add '
     'fold')
step('imad_full', [184, 185], '{a} * {b} + {c}', {'imul': 1, 'iadd': 1},
     note='the same with 32-bit operands (the corpus: imad 3.0)')
step('mul32x16_mad', [185], '{a} * ({b} & 0xffffu) + {c}', {'logic': 1, 'mul16': 1, 'iadd': 1},
     note='a * (b & 0xffff) + c: one XMAD or two')
# --- #186: the saturate modifier
step('fsat_fma', [186], 'clamp({a} * {b} + {c}, 0.0, 1.0)', {'fmul': 1, 'fadd': 1, 'minmax': 2}, ty='float',
     note='clamp(a*b + c, 0, 1): FFMA.SAT when the clamp folds')
step('fma_contract', [186], '{a} * {b} + {c}', {'fmul': 1, 'fadd': 1}, ty='float',
     note='the same without the clamp (the corpus: fmad_contract 1.0)')
step('fsat_fma_precise', [186], 'clamp({a} * {b} + {c}, 0.0, 1.0)', {'fmul': 1, 'fadd': 1, 'minmax': 2},
     ty='float', qual='precise', note='precise: a separate FMUL and FADD, the clamp on the add')
step('fma_precise', [186], '{a} * {b} + {c}', {'fmul': 1, 'fadd': 1}, ty='float', qual='precise',
     note='precise without the clamp (the corpus: fmad_precise 2.0)')
# --- #187: an integer to float conversion by the bits
step('i2f_bits', [187], 'floatBitsToUint(uintBitsToFloat(({a} & 0x7fffffu) | 0x4b000000u) - 8388608.0) + {b}',
     {'logic': 1, 'bitcast': 2, 'fadd': 1, 'iadd': 1}, note='asfloat(i | 0x4B000000) - 2^23 on i < 2^23')
step('i2f_conv', [187], 'floatBitsToUint(float({a} & 0x7fffffu)) + {b}', {'logic': 1, 'conv': 1, 'bitcast': 1, 'iadd': 1},
     note='the same through I2F')
# --- #188: an add on the exponent field
step('exp_add_bits', [188], 'uintBitsToFloat(floatBitsToUint({a}) + ((floatBitsToUint({b}) & 7u) << 23u))',
     {'bitcast': 3, 'logic': 1, 'shift': 1}, ty='float',
     note='asfloat(asint(x) + (k << 23)): ucas prices the shift and add as one shl_add')
step('exp_mul_conv', [188], '{a} * float(1u << (floatBitsToUint({b}) & 7u))',
     {'bitcast': 1, 'logic': 1, 'shift': 1, 'conv': 1, 'fmul': 1}, ty='float',
     note='the same as a shift, a conversion and a multiply')
# --- #18, #19: a 64-bit division and remainder by a constant (multiply-high and shifts in 64 bits)
step('u64_udiv_const', [18, 19], '({a} / 7ul) + {b}', {'x_u64_udivk': 1, 'x_i64add': 1}, ty='u64', exts=['i64'],
     note='x / 7 on u64, x from the noise buffer (the corpus: u64_add 2.02, u64_mul 15.52)')
step('u64_umod_const', [18, 19], '({a} % 7ul) + {b}', {'x_u64_umodk': 1, 'x_i64add': 1}, ty='u64', exts=['i64'],
     note='x % 7 on u64')
# --- #156: the catalogue's classes, each beside what it replaces
step('bfe_builtin', [156], 'bitfieldExtract({a}, 7, 9) + {b}', {'bfe': 1, 'iadd': 1}, note='BFE, constant offset')
step('bfe_shift_mask', [156], '(({a} >> 7u) & 0x1ffu) + {b}', {'shift': 1, 'logic': 1, 'iadd': 1},
     note='the same as a shift and a mask')
step('funnel', [156], '(({a} << sh) | ({b} >> (32u - sh))) + {c}', {'funnel': 1, 'iadd': 1},
     pre='uint sh = (k2 & 31u) | 1u;', note='a funnel shift by a variable amount (SHF)')
step('rotate_var', [156], '(({a} << sh) | ({a} >> (32u - sh))) + {b}', {'shift': 2, 'logic': 1, 'iadd': 1},
     pre='uint sh = (k2 & 31u) | 1u;', note='a rotate by a variable amount (the corpus: rotl 4.41)')
step('byteswap', [156], '(({a} >> 24u) | (({a} >> 8u) & 0xff00u) | (({a} << 8u) & 0xff0000u) | ({a} << 24u)) + {b}',
     {'byteperm': 1, 'iadd': 1}, note='a byte swap with shifts and masks (PRMT)')
step('byte_extract', [156], '(({a} >> 16u) & 0xffu) + {b}', {'shift': 1, 'logic': 1, 'iadd': 1},
     note='a byte extract with a shift and a mask')
step('mulhi', [156], 'umulhi({a}, {b}) + k', {'mulhi': 1, 'iadd': 1}, decls=G.UMULHI,
     note="umulExtended's high word (the corpus: umulhi 5.11)")
step('mulhi_u64', [156], 'uint((uint64_t({a}) * uint64_t({b})) >> 32) + k', {'x_i64mul': 1, 'shift': 1, 'iadd': 1},
     exts=['i64'], note='the same through a u64 multiply and a shift')
step('brev', [156], 'bitfieldReverse({a}) + {b}', {'brev': 1, 'iadd': 1}, note='BREV (the corpus: bitreverse 2.26)')
# clamp is undefined when its low bound is above its high one, so the bounds are kept apart: by their bits
# (below 2^16 and from 2^16), in float by their signs (free source modifiers on sm_61)
step('med3_clamp', [156], 'clamp({a}, {b} & 0xffffu, {c} | 0x10000u)', {'med3': 1, 'logic': 2}, lat=False,
     note='clamp of three chains, the bounds ordered by two masks')
step('med3_minmax', [156], 'min(max({a}, {b} & 0xffffu), {c} | 0x10000u)', {'minmax': 2, 'logic': 2},
     note='the same as a max and a min')
step('med3_fclamp', [156], 'clamp({a}, -abs({b}), abs({c}))', {'med3': 1}, ty='float',
     note='a float clamp of three chains, the bounds ordered by their signs')
step('sad4x8', [156], 'uint(abs(int({a} & 0xffu) - int({b} & 0xffu)) + abs(int(({a} >> 8u) & 0xffu) - '
     'int(({b} >> 8u) & 0xffu)) + abs(int(({a} >> 16u) & 0xffu) - int(({b} >> 16u) & 0xffu)) + '
     'abs(int({a} >> 24u) - int({b} >> 24u))) + {c}', {'sad4x8': 1, 'iadd': 1},
     note='four byte differences and their sum')
# --- #144: a ballot and a vote with their compare
step('sg_ballot_eq', [144], 'subgroupBallot({a} == {b}).x + {c}', {'sg_ballot': 1, 'cmp': 1, 'iadd': 1}, exts=SG)
step('sg_ballot_ult', [144], 'subgroupBallot({a} < {b}).x + {c}', {'sg_ballot': 1, 'cmp': 1, 'iadd': 1}, exts=SG)
step('sg_ballot_slt', [144], 'int(subgroupBallot({a} < {b}).x) + {c}', {'sg_ballot': 1, 'cmp': 1, 'iadd': 1},
     ty='int', exts=SG)
step('sg_ballot_flt', [144], 'uintBitsToFloat(subgroupBallot({a} < {b}).x | 0x3f800000u) + {c}',
     {'sg_ballot': 1, 'cmp': 1, 'logic': 1, 'fadd': 1}, ty='float', exts=SG)
probe('none', 'sg_ballot_bool', [144],
      why='a bool that changes at every step without a compare does not exist in SPIR-V: a logical not of the '
          'last one alternates, the driver makes both ballots once and the step is its add alone; the ballot of '
          'a compare is measured above')
step('sg_any_eq', [144], 'subgroupAny({a} == {b}) ? {c} : {d}', {'sg_vote': 1, 'cmp': 1, 'select': 1}, exts=SG)
step('sg_all_ult', [144], 'subgroupAll({a} < {b}) ? {c} : {d}', {'sg_vote': 1, 'cmp': 1, 'select': 1}, exts=SG)
step('sg_allequal', [144], 'subgroupAllEqual({a}) ? {b} : {c}', {'sg_vote': 1, 'select': 1}, exts=SG)
# --- #145: the shuffle's forms without an add
step('sg_shuffle_only', [145], 'subgroupShuffle({b}, {c} & 31u)', {'sg_shuffle': 1, 'logic': 1}, exts=SG)
step('sg_shuffle_xor_only', [145], 'subgroupShuffleXor({b}, 1u)', {'x_shfl_xor': 1}, exts=SG)
# a broadcast's result is the same in every lane, so a broadcast of it is its input: a ring of broadcasts
# alone folds after one round; they keep the add of the corpus's sg_broadcast (beside a shuffle the add
# costs nothing: sg_shuffle_xor_only against the corpus's sg_shuffle_xor)
step('sg_broadcast_add', [145], 'subgroupBroadcast({a}, 5u) + {b}', {'x_shfl_bcast': 1, 'iadd': 1}, exts=SG,
     note='with its add, the corpus sg_broadcast: a step of broadcasts alone folds')
step('sg_broadcast_first_add', [145], 'subgroupBroadcastFirst({a}) + {b}', {'x_shfl_bcast_first': 1, 'iadd': 1},
     exts=SG, note='with its add, the corpus sg_broadcast_first: a step of broadcasts alone folds')
step('sg_rotate_only', [145], 'subgroupRotate({b}, 3u)', {'x_shfl_rotate': 1}, exts=SG + ['sgrot'])
step('sg_quad_swap_only', [145], 'subgroupQuadSwapHorizontal({b})', {'x_shfl_quad': 1}, exts=SG)
# --- #147: reductions, the exclusive scan, a table in a register
for _op in ('Min', 'Max', 'And', 'Or', 'Xor'):
    step('sg_%s' % _op.lower(), [147], 'subgroup%s({a}) + {b}' % _op, {'x_sg_' + _op.lower(): 1, 'iadd': 1}, exts=SG)
step('sg_exclusive_add', [147], 'subgroupExclusiveAdd({a}) + {b}', {'x_sg_exscan': 1, 'iadd': 1}, exts=SG)
step('sg_regtab_read', [147], 'subgroupShuffle(rt, {a} & 31u) + {b}', {'sg_shuffle': 1, 'logic': 1, 'iadd': 1},
     exts=SG, pre='uint rt = inb.data[(gid & 31u) + 4096u];',
     note='a 32-entry table one entry a lane, read by a shuffle (set up by one load before the loop)')
step('ubo_tab_read', [147], 'ubo.d[{a} & 31u].x + {b}', {'x_load_ubo_div': 1, 'logic': 1, 'iadd': 1},
     decls='layout(std140, set = 0, binding = 3) uniform U { uvec4 d[1024]; } ubo;\n',
     note='the same table in constant memory, a different entry in each lane')
# --- #166: a constant table's load by its size
def perm_table(n, seed=0x5eed):
    """n u32 whose low bits move an index along a permutation: entry j holds pi(j) - j in its low log2(n)
    bits (random above), so a walk i -> (i + t[i] + d) mod n keeps every lane on an entry of its own.  A
    table that does not do this merges lanes: one affine in j (j k + c, k odd) sends every lane to one
    entry in log2(n) steps, a random one leaves a few entries of 32 after a few hundred"""
    rnd = random.Random(seed + n)
    pi = list(range(n))
    rnd.shuffle(pi)
    sh = n.bit_length() - 1
    return [((pi[j] - j) % n) | ((rnd.getrandbits(32 - sh) << sh) & 0xffffffff) for j in range(n)]


def table_bytes(n):
    return struct.pack('<%dI' % n, *perm_table(n))


def _ctab(n):
    """a constant table of n u32 and how to read entry I: one OpConstantComposite holds at most
    65532 operands (the word count is 16 bits), so 256 KiB is 4 rows of 16384"""
    v = ['0x%08xu' % x for x in perm_table(n)]
    if n <= 16384:
        return 'const uint tab[%d] = uint[](%s);\n' % (n, ', '.join(v)), 'tab[(I) & %du]' % (n - 1)
    rows = n // 16384
    return ('const uint tab[%d][16384] = uint[%d][16384](%s);\n' % (rows, rows, ', '.join(
        'uint[16384](%s)' % ', '.join(v[r * 16384:(r + 1) * 16384]) for r in range(rows))),
        'tab[((I) >> 14) & %du][(I) & 16383u]' % (rows - 1))


TB = 'layout(std430, set = 0, binding = 2) readonly buffer T { uint t[]; } tb;\n'


def tbarg(n):
    """the storage-buffer table: n entries of perm_table at binding 2"""
    return dict(files={'table%d.bin' % n: table_bytes(n)}, args=['--bind', '0.2=file={data}/table%d.bin' % n])


# NVIDIA's driver refuses a 64 KiB constant array (VK_ERROR_UNKNOWN): the sizes between show where it stops
for _n, _kib in ((256, 1), (1024, 4), (4096, 16), (8192, 32), (16384, 64), (65536, 256)):
    _tab, _rd = _ctab(_n)
    _x = {'x_ctab%dk' % _kib: 1, 'logic': 1, 'iadd': 1}
    if _n > 16384:
        _x.update(logic=2, shift=1)
    # Mesa (lavapipe, RADV) compiles a 64 KiB constant array for more than 10 minutes: those run last
    _slow = '; run last: a large constant array takes Mesa minutes to compile, so a timeout here is an answer ' \
        'too' if _n > 256 else ''
    step('ctab%dk_random' % _kib, [166], '{a} + ' + _rd.replace('I', '{a} + {u}u'), _x, sub='mem', C=1,
         decls=_tab, slow=_n > 256, note='a %d KiB constant array, an index that the loaded value moves (random)%s%s'
         % (_kib, '; 4 rows of 64 KiB, as SPIR-V needs' if _n > 16384 else '', _slow))
    step('ctab%dk_seq' % _kib, [166], '{a} + ' + _rd.replace('I', 'ib + {u}u'), dict(_x), sub='mem', C=1,
         decls=_tab, slow=_n > 256, pre_iter='uint ib = gid + i * 64u;',
         note='the same, consecutive lanes on consecutive entries' + _slow)
    step('ssbo%dk_random' % _kib, [166], '{a} + tb.t[({a} + {u}u) & %du]' % (_n - 1),
         {'load': 1, 'logic': 1, 'iadd': 1}, sub='mem', C=1, decls=TB, **tbarg(_n),
         note='the same table in a storage buffer (random)')
    step('ssbo%dk_seq' % _kib, [166], '{a} + tb.t[(ib + {u}u) & %du]' % (_n - 1),
         {'load': 1, 'logic': 1, 'iadd': 1}, sub='mem', C=1, pre_iter='uint ib = gid + i * 64u;', decls=TB,
         **tbarg(_n), note='the same table in a storage buffer (consecutive)')
# --- #168: a small texture as a table
step('tex_table_r32ui', [168], '{a} + texelFetch(utex, ivec2(int({a} & 255u), 0), 0).x',
     {'x_tex_fetch_r32ui': 1, 'logic': 1, 'iadd': 1}, sub='mem', C=1, decls=UTEX, files={'table256.bin': table_bytes(256)},
     args=['--bind', '0.5=format=r32_uint,size=256x1,raw={data}/table256.bin,filter=nearest'],
     note='texelFetch of a 256-entry R32UI texture')
step('ssbo_table_256', [168], '{a} + tb.t[{a} & 255u]', {'load': 1, 'logic': 1, 'iadd': 1}, sub='mem', C=1,
     decls=TB, **tbarg(256), note='the same 256 entries in a storage buffer')
step('tex_lerp_r8', [168], '{a} + textureLod(tex, vec2(fract({a}), 0.5), 0.0).x',
     {'x_tex_bilinear_r8': 1, 'conv': 1, 'fadd': 1}, ty='float', sub='mem', C=1, decls=TEX,
     args=texarg('r8_unorm', '256x1', 'linear'), note='a bilinear R8 read as a 256-entry interpolating table')
step('ssbo_lerp', [168], '{a} + mix(uintBitsToFloat(inb.data[uint({a} * 255.0) & 255u] & 0x3fffffffu), '
     'uintBitsToFloat(inb.data[(uint({a} * 255.0) + 1u) & 255u] & 0x3fffffffu), fract({a} * 255.0))',
     {'load': 2, 'conv': 2, 'fmul': 2, 'logic': 4, 'iadd': 1, 'fadd': 2}, ty='float', sub='mem', C=1,
     note='the same interpolation from a storage buffer: two loads and a mix')
# --- #115, #118, #121: sampled reads
step('tex_bilinear_ref', [115, 118, 121], TEXSTEP, {'tex_sample': 1, 'fadd': 1}, ty='float', sub='mem', C=1,
     decls=TEX, pre_iter=UV, args=texarg('rgba8_unorm', '1024x1024', 'linear'),
     note='the corpus tex_bilinear (rgba8, bilinear), the reference')
step('tex_nearest', [115], TEXSTEP, {'tex_nearest': 1, 'fadd': 1}, ty='float', sub='mem', C=1, decls=TEX,
     pre_iter=UV, args=texarg('rgba8_unorm', '1024x1024', 'nearest'), note='the same with a NEAREST sampler')
step('tex_nearest_repeat', [115], '{a} + textureLod(tex, vec2((float(ix + {u}) + 0.5) * (1.0 / 1024.0), vy), 0.0).x',
     {'iadd': 1, 'conv': 1, 'fadd': 2, 'fmul': 1, 'tex_nearest': 1}, ty='float', sub='mem', C=1, decls=TEX,
     pre_iter='int ix = int(gid) + int(i) * 16; float vy = (float((gid >> 10) + i * 16u) + 0.5) / 1024.0;',
     args=texarg('rgba8_unorm', '1024x1024', 'nearest', 'repeat'),
     note='NEAREST with REPEAT, the coordinate (i + 0.5) / n made in the step, past 1')
step('tex_bilinear_r8', [118], TEXSTEP, {'x_tex_bilinear_r8': 1, 'fadd': 1}, ty='float', sub='mem', C=1,
     decls=TEX, pre_iter=UV, args=texarg('r8_unorm', '1024x1024', 'linear'), note='an R8 texture of the same size')
# in and out of the cache with the same instructions: each read at a place of its own over the whole texture
# (a hash of the lane and the iteration, then the step's offset of a golden-ratio fraction of the texture),
# so a 4096x4096 texture (16 MiB of R8, 8 of BC4, over a 512 KiB L2) comes from DRAM and a 64x64 one (4 KiB)
# from the cache; textureLodOffset's offsets of at most 8 texels would keep a step's reads in one line
SCATTER = ('uint hs = (gid ^ (i * 0x9e3779b9u)) * 0x85ebca6bu; hs ^= hs >> 13; '
           'vec2 uv = vec2(float(hs & 0xffffu), float(hs >> 16)) * (1.0 / 65536.0);')
SCATSTEP = '{a} + textureLod(tex, uv + vec2(float({u}) * 0.618034, float({u}) * 0.381966), 0.0).x'
for _f in ('bc4_unorm', 'r8_unorm'):
    for _filt in ('linear', 'nearest'):
        for _where, _size in (('cached', '64x64'), ('uncached', '4096x4096')):
            step('%s_%s_%s' % (_f.split('_')[0], _filt, _where), [121], SCATSTEP,
                 {'x_%s_%s' % (_f.split('_')[0], _filt): 1, 'fadd': 3}, ty='float', sub='mem', C=1, decls=TEX,
                 pre_iter=SCATTER, args=texarg(_f, _size, _filt),
                 note='%s, %s, %s, a different place each read (%s); the step has 3 fadd' % (
                     _f, _filt, _size, 'in the cache' if _where == 'cached' else 'out of the cache'))
# --- #188 and #187 come with their alternatives above; #120, #207, #212 are units; the rest values

probe('unit', 'setup_binding', [120], how='dispatch',
      note='a 128-invocation dispatch with and without one more combined image sampler (read once)')
probe('absent', 'setup_upload', [120], needs='a timed upload: snx-shaderperf times dispatches and draws, '
      'not the copy of a texture')
probe('unit', 'draw_empty', [212], how='draw', note='a draw of one point outside the target: no fragment')
probe('unit', 'points', [212], how='points', note='points, one fragment each, into rgba8, and outside the target')
probe('unit', 'rop_blend', [207], how='rop',
      note='points into R32_SFLOAT with no blending, ADD ONE ONE and SRC_ALPHA, ONE_MINUS_SRC_ALPHA (factors), '
           'against a float atomicAdd on the same cells; 256x256 (L2) and 4096x4096 (DRAM, points 61 cells apart)')
probe('absent', 'rop_minmax', [207, 211], needs='MIN and MAX blend ops (snx-shaderperf --blend has none, '
      'alpha and add): the MIN/MAX price, and whether they keep a NaN and the sign of a zero',
      key=r'--blend [^\n]*\bmax\b')
probe('absent', 'rop_logic', [208], needs='logicOpEnable with OR and XOR on R32_UINT (snx-shaderperf has no '
      'logic op state)', key=r'--logic')
probe('absent', 'rop_stencil', [209], needs='a stencil test with INCREMENT_AND_WRAP on an S8 attachment '
      '(snx-shaderperf has --depth but no stencil state)', key=r'--stencil')
probe('absent', 'rop_adv', [210], needs='VK_EXT_blend_operation_advanced DIFFERENCE and EXCLUSION '
      '(snx-shaderperf has no advanced blend state)', key=r'(?i)advanced')
probe('value', 'blend_rounding', [211], how='blend',
      note='two full-screen triangles into R32_SFLOAT, ADD ONE ONE: (0 + a) + b against round to nearest; '
           'and SRC_ALPHA, ONE_MINUS_SRC_ALPHA into RGBA32F: s a + d (1 - a), fused or not')
probe('value', 'unorm_rounding', [211], how='unorm', queue='FILA 107 (#10)',
      note='k/255 + e for small e of both signs, written into rgba8_unorm and read back')
probe('value', 'filter_weights', [116, 117], how='filter',
      note='a 4x1 R8 ramp (0, 255, 37, 200) read bilinear at 4096 points across two texel pairs, and every '
           '2^-20 of a texel over the first 1/16 of one')
probe('info', 'subtexel_bits', [117], note='vulkaninfo: VkPhysicalDeviceLimits.subTexelPrecisionBits')
probe('value', 'repeat_large', [119], how='repeat',
      note='NEAREST REPEAT on a 256x1 R32UI table at (i + 0.5) / 256 for i near 0 and near +-2^22')
probe('value', 'bc4_palette', [122], how='bc4',
      note='texelFetch of every palette entry of 12 endpoint pairs, both BC4 modes')
probe('none', 'f16_arith', [], queue='FILA 108 (#10)',
      why='measured on 2026-11-11: the f32 -> f16 -> f32 round trip is dropped (folds in the prelude), '
          'GP108 has no f16 arithmetic')
probe('none', 'stage_io', [], queue='FILA 110-112 (#12)',
      why='measured by the corpus (results/2026-11-11-graphics.md); run.py --io-only measures them again')
probe('none', 'sg_operations', [146],
      why='measured (clustered add 19.44, partition 54.54, partitioned add 65.16, subgroup barrier 1.34); what '
          'is missing is ucas operations for them, not a measure')


# ---------------------------------------------------------------------------
# running

def log_path(name):
    return os.path.join(OUT, 'raw', name)


def run_tool(cmd, log, timeout=None, cwd=None):
    """a command with its log; the exit, the output and what went wrong are kept"""
    with open(log, 'w') as fh:
        fh.write(' '.join(cmd) + '\n')
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           timeout=timeout or A.timeout, cwd=cwd or HERE)
    except subprocess.TimeoutExpired:
        raise ProbeError('%s: no answer in %d s (raw/%s)' % (os.path.basename(cmd[0]), timeout or A.timeout,
                                                             os.path.basename(log)))
    except OSError as e:
        raise ProbeError('%s: %s' % (cmd[0], e))
    with open(log, 'a') as fh:
        fh.write(p.stderr)
    if p.returncode:
        tail = p.stderr.strip().splitlines()[-3:]
        raise ProbeError('%s exited %d: %s (raw/%s)' % (os.path.basename(cmd[0]), p.returncode,
                                                        ' / '.join(tail), os.path.basename(log)))
    return p.stdout


def sp(args, log, timeout=None):
    out = run_tool([A.sp] + args, log, timeout)
    try:
        d = json.loads(out)
    except ValueError:
        raise ProbeError('snx-shaderperf gave no JSON (raw/%s)' % os.path.basename(log))
    with open(log + '.json', 'w') as fh:
        json.dump(d, fh)
    return d


def compile_glsl(name, src, stage):
    """GLSL into OUT/spv/NAME.spv, checked by spirv-val; the path relative to nothing (absolute)"""
    sp_ = os.path.join(OUT, 'src', '%s.%s' % (name, stage))
    out = os.path.join(OUT, 'spv', name + '.spv')
    with open(sp_, 'w') as fh:
        fh.write(src)
    run_tool(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '--quiet', '-o', out, sp_],
             log_path('glslang-%s.log' % name), 120)
    run_tool(['spirv-val', '--target-env', 'vulkan1.3', out], log_path('spirv-val-%s.log' % name), 120)
    return out


def timing_args(runs=None):
    # one pipeline creation: compile times are not what a probe measures (a 64 KiB constant table
    # takes minutes to compile on some drivers)
    if A.quick:
        return ['--runs', '3', '--warmup', '1', '--warmup-ms', '1', '--compile-runs', '1', '--json', '--quiet']
    return ['--runs', str(runs or A.runs), '--compile-runs', '1', '--json', '--quiet']


def per_rep(sh):
    t = sh['time_ns_per_rep']
    return t['median'], (t['median_hi'] - t['median_lo']) / 2


def fadd_entry():
    man = json.load(open(os.path.join(HERE, 'manifest.json')))
    for t in man['tests']:
        if t['name'] == 'fadd':
            return t
    raise ProbeError('manifest.json has no fadd')


def run_step(p):
    """the A and 2A builds beside fadd.a and fadd.b in one run: the step's cost over fadd's"""
    global FADD_NS
    x = dict(p, name=p['name'])
    Aun = x['A']
    lanes = 4096 if A.quick else 65536
    cost_guess = G.model_guess(x['ops'])
    iters = 2 if A.quick else (x['iters'] or max(2, int(round(5.8e8 / (lanes * Aun * x['C'] * cost_guess)))))
    files = []
    for tag, U in (('a', Aun), ('b', 2 * Aun)):
        files.append(compile_glsl('%s.%s' % (p['name'], tag), G.gen_compute(x, U, False), 'comp'))
    ref = fadd_entry()
    shaders = [os.path.join(HERE, ref['files']['a']), os.path.join(HERE, ref['files']['b'])] + files
    test = dict(chains=x['C'], unroll=[Aun, 2 * Aun], iters=iters)
    for fn, data in x['files'].items():
        write(fn, data)
    extra = timing_args() + ['--size', str(lanes)] + [a.replace('{data}', os.path.join(OUT, 'data'))
                                                      for a in x['args']]
    for v in x['vkext']:
        extra += ['--ext', v]
    best = None
    for attempt in range(1 if A.quick else A.tries):
        d = sp(R.sp_cmd('', shaders, iters, extra)[1:], log_path('%s.%d.log' % (p['name'], attempt)))
        tp = R.throughput(test, d, ref)
        if tp['ref']['ns'] <= 0 or tp['x']['ns'] == 0:
            if not A.quick:
                raise ProbeError('b ran no slower than a (ref %.3g ns, step %.3g ns): no step to time'
                                 % (tp['ref']['ns'], tp['x']['ns']))
            # a smoke run's grid is too small to time: keep going with a placeholder of 1 ns
            tp['ref']['ns'] = tp['ref']['ns'] if tp['ref']['ns'] > 0 else 1.0
            tp['x']['ns'] = tp['x']['ns'] or 1.0
        cost = tp['x']['ns'] / tp['ref']['ns']
        err = abs(cost) * ((tp['x']['err'] / tp['x']['ns']) ** 2 + (tp['ref']['err'] / tp['ref']['ns']) ** 2) ** 0.5
        if best is None or err / abs(cost) < best['err'] / abs(best['cost']):
            best = dict(cost=cost, err=err, tries=attempt + 1, fadd_ns=tp['ref']['ns'],
                        bytes_per_step=tp['x']['bytes_per_step'], iters=iters, lanes=lanes, clock_mhz=R.sm_clock(d))
        if err <= A.max_rel_err * abs(cost):
            break
    FADD_NS = best['fadd_ns']
    known = price(p['ops'])
    best['ucas_price'] = known
    return best


def price(ops):
    """the step as the prelude prices it, None when a class is not in it"""
    if any(c not in PRELUDE for c in ops):
        return None
    return sum(n * PRELUDE[c] for c, n in ops.items())


def in_fadd(ns):
    if FADD_NS is None:
        raise ProbeError('no fadd run before this unit: nothing to convert ns into fadd steps')
    return ns / FADD_NS


def fadd_now(tag):
    """fadd's ns a lane-step from a run of its own, just before a unit's runs: the SM clock moves between
    runs (1.2 to 1.9 GHz), so a unit converts into fadd steps by the fadd timed beside it"""
    global FADD_NS
    ref = fadd_entry()
    lanes = 4096 if A.quick else 65536
    iters = 2 if A.quick else ref['iters']
    shaders = [os.path.join(HERE, ref['files']['a']), os.path.join(HERE, ref['files']['b'])]
    d = sp(R.sp_cmd('', shaders, iters, timing_args() + ['--size', str(lanes)])[1:], log_path('fadd_%s.log' % tag))
    (ta, ea), (tb, eb) = per_rep(d['shaders'][0]), per_rep(d['shaders'][1])
    steps = d['shaders'][0]['invocations_per_rep'] * iters * ref['unroll'][0] * ref['chains']
    if tb <= ta:
        if not A.quick:
            raise ProbeError('fadd.b ran no slower than fadd.a (%.3g ns, %.3g ns): no fadd to convert by' % (tb, ta))
        tb = ta + steps * 1e-3    # a smoke run's grid is too small to time: a placeholder of 1 ps
    FADD_NS = (tb - ta) / steps
    return dict(fadd_ns=FADD_NS, fadd_err_ns=(ea * ea + eb * eb) ** 0.5 / steps, fadd_clock_mhz=R.sm_clock(d))


# --- shaders of the unit and value probes ----------------------------------------

VS_POINTS = '''#version 460
// point i at cell (i * stride) % (w * h) of a w x h target; off moves every point outside it
layout(push_constant) uniform PC { uint w; uint h; uint stride; float off; } pc;
void main() {
    uint i = uint(gl_VertexIndex), k = (i * pc.stride) % (pc.w * pc.h);
    vec2 c = (vec2(float(k % pc.w), float(k / pc.w)) + 0.5) / vec2(float(pc.w), float(pc.h));
    gl_Position = vec4(c * 2.0 - 1.0 + pc.off, 0.0, 1.0);
    gl_PointSize = 1.0;
}
'''
FS_ONE = '''#version 460
layout(location = 0) out vec4 c;
void main() { c = vec4(1.0, 1.0, 1.0, 0.5); }
'''
CS_ATOMIC = '''#version 460
#extension GL_EXT_shader_atomic_float : require
// the same cells as the points, each a float atomicAdd into a buffer
layout(local_size_x = 128) in;
layout(push_constant) uniform PC { uint w; uint h; uint stride; } pc;
layout(std430, set = 0, binding = 2) buffer OF { float f[]; } of;
void main() {
    uint i = gl_GlobalInvocationID.x;
    atomicAdd(of.f[(i * pc.stride) % (pc.w * pc.h)], 1.0);
}
'''
CS_BIND0 = '''#version 460
layout(local_size_x = 128) in;
layout(std430, set = 0, binding = 1) buffer Out { uint data[]; } outb;
void main() { outb.data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 3u; }
'''
CS_BIND1 = '''#version 460
layout(local_size_x = 128) in;
layout(std430, set = 0, binding = 1) buffer Out { uint data[]; } outb;
layout(set = 0, binding = 5) uniform sampler2D tex;
void main() { outb.data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 3u + uint(texelFetch(tex, ivec2(0), 0).x); }
'''
VS_TWO = '''#version 460
// two full-screen triangles, one after the other: which says which
layout(location = 0) flat out uint which;
void main() {
    uint v = uint(gl_VertexIndex), k = v % 3u;
    gl_Position = vec4(k == 1u ? 3.0 : -1.0, k == 2u ? 3.0 : -1.0, 0.0, 1.0);
    which = v / 3u;
}
'''
FS_PAIR = '''#version 460
// the first triangle writes a, the second b (pixel i's pair at 2i, 2i + 1)
layout(location = 0) flat in uint which;
layout(location = 0) out vec4 c;
layout(std430, set = 0, binding = 1) readonly buffer V { float v[]; } vals;
layout(push_constant) uniform PC { uint w; } pc;
void main() {
    uint i = uint(gl_FragCoord.y) * pc.w + uint(gl_FragCoord.x);
    c = vec4(vals.v[2u * i + which]);
}
'''
FS_ALPHA = '''#version 460
// the first triangle writes d with alpha 1, the second s with alpha a (pixel i's at 3i)
layout(location = 0) flat in uint which;
layout(location = 0) out vec4 c;
layout(std430, set = 0, binding = 1) readonly buffer V { float v[]; } vals;
layout(push_constant) uniform PC { uint w; } pc;
void main() {
    uint i = uint(gl_FragCoord.y) * pc.w + uint(gl_FragCoord.x);
    c = which == 0u ? vec4(vec3(vals.v[3u * i]), 1.0) : vec4(vec3(vals.v[3u * i + 1u]), vals.v[3u * i + 2u]);
}
'''
FS_VALUE = '''#version 460
// pixel i writes value i as it is: the target's conversion rounds it
layout(location = 0) out vec4 c;
layout(std430, set = 0, binding = 1) readonly buffer V { float v[]; } vals;
layout(push_constant) uniform PC { uint w; } pc;
void main() {
    uint i = uint(gl_FragCoord.y) * pc.w + uint(gl_FragCoord.x);
    c = vec4(vals.v[i]);
}
'''
CS_SAMPLE = '''#version 460
// lane i reads the texture at u[i] (row 0.5), explicit LOD 0, and stores the bits
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer In { float u[]; } inb;
layout(std430, set = 0, binding = 1) buffer Out { uint r[]; } outb;
layout(set = 0, binding = 5) uniform sampler2D tex;
void main() {
    uint i = gl_GlobalInvocationID.x;
    outb.r[i] = floatBitsToUint(textureLod(tex, vec2(inb.u[i], 0.5), 0.0).x);
}
'''
CS_REPEAT = '''#version 460
// lane i reads a 256-entry table at (idx[i] + 0.5) / 256, NEAREST and REPEAT
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer In { int idx[]; } inb;
layout(std430, set = 0, binding = 1) buffer Out { uint r[]; } outb;
layout(set = 0, binding = 5) uniform usampler2D tab;
void main() {
    uint i = gl_GlobalInvocationID.x;
    outb.r[i] = textureLod(tab, vec2((float(inb.idx[i]) + 0.5) * (1.0 / 256.0), 0.5), 0.0).x;
}
'''
CS_FETCH = '''#version 460
// lane i fetches texel (i % w, i / w) and stores the bits
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 1) buffer Out { uint r[]; } outb;
layout(set = 0, binding = 5) uniform sampler2D tex;
layout(push_constant) uniform PC { uint w; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    outb.r[i] = floatBitsToUint(texelFetch(tex, ivec2(int(i % pc.w), int(i / pc.w)), 0).x);
}
'''


def write(name, data):
    path = os.path.join(OUT, 'data', name)
    with open(path, 'wb') as fh:
        fh.write(data)
    return path


def dump(name):
    return os.path.join(OUT, 'data', name)


def read_dump(path, n):
    """--dump of several shaders writes FILE-N; of one, FILE"""
    for cand in (path, path.replace('.bin', '-0.bin')):
        if os.path.exists(cand):
            data = open(cand, 'rb').read()
            if len(data) < n:
                raise ProbeError('%s: %d bytes, %d expected' % (os.path.basename(cand), len(data), n))
            return data[:n]
    raise ProbeError('snx-shaderperf wrote no %s' % os.path.basename(path))


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def f32bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def f32_step(v, up):
    """the f32 next to v, up or down"""
    if v == 0:
        return math.copysign(2.0 ** -149, 1 if up else -1)
    b = f32bits(v)
    b += 1 if (v > 0) == up else -1
    return struct.unpack('<f', struct.pack('<I', b))[0]


def round_f32(x, mode):
    """the rational x rounded to f32, to nearest even or toward zero (no overflow here)"""
    if x == 0:
        return 0.0
    v = f32(float(x))
    lo, hi = (v, f32_step(v, True)) if Fraction(v) <= x else (f32_step(v, False), v)
    if Fraction(lo) == x:
        return lo
    if mode == 'zero':
        return lo if abs(Fraction(lo)) < abs(Fraction(hi)) else hi
    dl, dh = x - Fraction(lo), Fraction(hi) - x
    if dl != dh:
        return lo if dl < dh else hi
    return lo if f32bits(lo) % 2 == 0 else hi


def ulp32(x):
    """the spacing of f32 at x"""
    x = abs(x)
    if x < 2.0 ** -126:
        return 2.0 ** -149
    return 2.0 ** (math.floor(math.log2(x)) - 23)


def ulps(got, want):
    """how far got is from want, in f32 ulps at the larger of the two"""
    return abs(got - want) / ulp32(max(abs(got), abs(want), 2.0 ** -126))


# --- units ------------------------------------------------------------------------

def unit_setup_binding(p):
    f = fadd_now(p['name'])
    a = compile_glsl('setup_bind0', CS_BIND0, 'comp')
    b = compile_glsl('setup_bind1', CS_BIND1, 'comp')
    d = sp([a, b, '--size', '128', '--bind', '0.5=format=rgba8_unorm,size=4x4,pattern=noise'] + timing_args(),
           log_path('setup_binding.log'))
    (ta, ea), (tb, eb) = per_rep(d['shaders'][0]), per_rep(d['shaders'][1])
    return dict(f, ns_base=ta, ns_one_more=tb, ns_binding=tb - ta, err_ns=(ea * ea + eb * eb) ** 0.5,
                fadd_steps=in_fadd(tb - ta), clock_mhz=R.sm_clock(d))


def unit_draw_empty(p):
    f = fadd_now(p['name'])
    vs = compile_glsl('vs_points', VS_POINTS, 'vert')
    fs = compile_glsl('fs_one', FS_ONE, 'frag')
    d = sp([vs, '--fs', fs, '--topology', 'points', '--vertices', '1', '--size', '256x256', '--push', 'w=u32:256',
            '--push', 'h=u32:256', '--push', 'stride=u32:1', '--push', 'off=f32:4.0'] + timing_args(),
           log_path('draw_empty.log'))
    t, e = per_rep(d['shaders'][0])
    return dict(f, ns_draw=t, err_ns=e, fadd_steps=in_fadd(t), clock_mhz=R.sm_clock(d))


def unit_points(p):
    vs = compile_glsl('vs_points', VS_POINTS, 'vert')
    fs = compile_glsl('fs_one', FS_ONE, 'frag')
    n = 4096 if A.quick else 262144
    r = fadd_now(p['name'])
    for tag, off in (('onscreen', '0.0'), ('offscreen', '4.0')):
        d = sp([vs, '--fs', fs, '--topology', 'points', '--vertices', str(n), '--size', '256x256', '--push',
                'w=u32:256', '--push', 'h=u32:256', '--push', 'stride=u32:1', '--push', 'off=f32:' + off] +
               timing_args(),
               log_path('points_%s.log' % tag))
        t, e = per_rep(d['shaders'][0])
        r[tag] = dict(ns_vertex=t / n, err_ns=e / n, fadd_steps=in_fadd(t / n), clock_mhz=R.sm_clock(d))
    r['vertices'] = n
    return r


def unit_rop(p):
    vs = compile_glsl('vs_points', VS_POINTS, 'vert')
    fs = compile_glsl('fs_one', FS_ONE, 'frag')
    cs = compile_glsl('cs_atomic', CS_ATOMIC, 'comp')
    n = 4096 if A.quick else 262144
    r = dict(points=n)
    # L2: 256 KiB, every cell four times in order; DRAM: 64 MiB, the points 61 cells apart over all of it
    for where, w, stride in (('l2', 256, 1), ('dram', 4096, 61)):
        if A.quick and where == 'dram':
            w = 512
        r['fadd_' + where] = fadd_now('%s_%s' % (p['name'], where))
        for blend in ('none', 'add', 'alpha'):
            d = sp([vs, '--fs', fs, '--topology', 'points', '--vertices', str(n), '--size', '%dx%d' % (w, w),
                    '--rt', '0=r32_sfloat', '--blend', blend, '--push', 'w=u32:%d' % w, '--push', 'h=u32:%d' % w,
                    '--push', 'stride=u32:%d' % stride, '--push', 'off=f32:0.0'] + timing_args(),
                   log_path('rop_%s_%s.log' % (blend, where)))
            t, e = per_rep(d['shaders'][0])
            r['%s_%s' % (blend, where)] = dict(ns_point=t / n, err_ns=e / n, fadd_steps=in_fadd(t / n),
                                               clock_mhz=R.sm_clock(d))
        d = sp([cs, '--size', str(n), '--buffer-size', '%dM' % max(1, w * w * 4 >> 20), '--ext',
                'VK_EXT_shader_atomic_float', '--push', 'w=u32:%d' % w, '--push', 'h=u32:%d' % w,
                '--push', 'stride=u32:%d' % stride] + timing_args(), log_path('rop_atomic_%s.log' % where))
        t, e = per_rep(d['shaders'][0])
        r['atomic_%s' % where] = dict(ns_atomic=t / n, err_ns=e / n, fadd_steps=in_fadd(t / n),
                                      clock_mhz=R.sm_clock(d))
    return r


# --- values -----------------------------------------------------------------------

def value_blend(p):
    vs = compile_glsl('vs_two', VS_TWO, 'vert')
    W = 64
    # pairs whose sum rounds: a tie to even each way, a sticky bit, a sum that cancels, random
    pairs = [(1.0, 2.0 ** -24), (1.0 + 2.0 ** -23, 2.0 ** -24), (1.0, 3 * 2.0 ** -25), (1.0, 2.0 ** -25),
             (-1.0, 2.0 ** -24), (16777216.0, 1.0), (16777218.0, 1.0), (1.0, -1.0 + 2.0 ** -24)]
    seed = 12345
    while len(pairs) < W * W:
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xffffffffffffffff
        a = struct.unpack('<f', struct.pack('<I', 0x3f800000 | (seed >> 41)))[0] * 2.0 ** ((seed >> 8) % 8)
        b = struct.unpack('<f', struct.pack('<I', 0x3f800000 | ((seed >> 17) & 0x7fffff)))[0] * \
            2.0 ** -((seed >> 4) % 30) * (-1 if seed & 1 else 1)
        pairs.append((f32(a), f32(b)))
    vals = write('blend_pairs.bin', b''.join(struct.pack('<ff', a, b) for a, b in pairs))
    fs = compile_glsl('fs_pair', FS_PAIR, 'frag')
    out = dump('blend_add.bin')
    sp([fs, '--vs', vs, '--vertices', '6', '--size', '%dx%d' % (W, W), '--rt', '0=r32_sfloat', '--blend', 'add',
        '--clear', '--bind', '0.1=file=' + vals, '--push', 'w=u32:%d' % W, '--check', '--dump', 'rt0=' + out,
        '--runs', '2', '--warmup', '1', '--warmup-ms', '1', '--compile-runs', '1', '--json', '--quiet'], log_path('blend_add.log'))
    got = struct.unpack('<%df' % (W * W), read_dump(out, 4 * W * W))
    exact = [Fraction(a) + Fraction(b) for a, b in pairs]
    rne = [round_f32(x, 'even') for x in exact]
    rtz = [round_f32(x, 'zero') for x in exact]
    bad = [(a, b, g, e) for (a, b), g, e in zip(pairs, got, rne) if f32bits(g) != f32bits(e)][:8]
    # snx-shaderperf leaves blending off, saying nothing, on a format the device cannot blend: then
    # the second triangle's b is all the target holds
    moved = [(b, g) for (a, b), g, e in zip(pairs, got, rne) if f32bits(e) != f32bits(b)]
    r = dict(add_pairs=len(pairs), add_unblended=sum(f32bits(g) == f32bits(b) for b, g in moved),
             add_moved=len(moved),
             add_round_nearest=sum(f32bits(g) == f32bits(e) for g, e in zip(got, rne)),
             add_toward_zero=sum(f32bits(g) == f32bits(e) for g, e in zip(got, rtz)),
             add_differ=[dict(a=a, b=b, got=g, nearest=e) for a, b, g, e in bad])
    # factors: s a + d (1 - a) against the roundings a blender may do
    tri = []
    seed = 777
    while len(tri) < W * W:
        seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xffffffffffffffff
        d_ = f32(1.0 + ((seed >> 40) & 0xffffff) / 2.0 ** 24)
        s_ = f32(((seed >> 16) & 0xffffff) / 2.0 ** 24)
        al = f32(((seed >> 3) & 0x1fff) / 8192.0 + 2.0 ** -14)
        tri.append((d_, s_, al))
    vals = write('blend_alpha.bin', b''.join(struct.pack('<fff', d_, s_, al) for d_, s_, al in tri))
    fs = compile_glsl('fs_alpha', FS_ALPHA, 'frag')
    out = dump('blend_alpha.bin.out')
    sp([fs, '--vs', vs, '--vertices', '6', '--size', '%dx%d' % (W, W), '--rt', '0=rgba32_sfloat', '--blend',
        'alpha', '--clear', '--bind', '0.1=file=' + vals, '--push', 'w=u32:%d' % W, '--check', '--dump',
        'rt0=' + out, '--runs', '2', '--warmup', '1', '--warmup-ms', '1', '--compile-runs', '1', '--json', '--quiet'],
       log_path('blend_alpha.log'))
    got = struct.unpack('<%df' % (4 * W * W), read_dump(out, 16 * W * W))[0::4]
    F = Fraction

    def rn(x):
        return round_f32(x, 'even')
    forms = {
        'fl(fl(s a) + fl(d fl(1 - a)))': lambda d_, s_, al: rn(F(rn(F(s_) * F(al))) + F(rn(F(d_) * F(rn(1 - F(al)))))),
        'fl(s a + fl(d fl(1 - a))) (an fma on s a)': lambda d_, s_, al: rn(F(s_) * F(al) + F(rn(F(d_) * F(rn(1 - F(al)))))),
        'fl(d fl(1 - a) + fl(s a)) (an fma on d)': lambda d_, s_, al: rn(F(d_) * F(rn(1 - F(al))) + F(rn(F(s_) * F(al)))),
        'fl(s a + d (1 - a)) (exact, one rounding)': lambda d_, s_, al: rn(F(s_) * F(al) + F(d_) * (1 - F(al))),
    }
    r['alpha_triples'] = len(tri)
    r['alpha_forms'] = {k: sum(1 for (d_, s_, al), g in zip(tri, got) if f32bits(g) == f32bits(fn(d_, s_, al)))
                        for k, fn in forms.items()}
    return r


def value_unorm(p):
    W = 64
    vals = []
    for k in range(255):
        for e in (-2.0 ** -20, -2.0 ** -24, 0.0, 2.0 ** -24, 2.0 ** -20):
            vals.append(f32((k + 0.5) / 255.0 + e))
    for k in range(256):
        vals.append(f32(k / 255.0))
    while len(vals) < W * W:
        vals.append(f32((len(vals) % 4096) / 4096.0))
    data = write('unorm_vals.bin', b''.join(struct.pack('<f', v) for v in vals))
    fs = compile_glsl('fs_value', FS_VALUE, 'frag')
    out = dump('unorm_rt.bin')
    sp([fs, '--size', '%dx%d' % (W, W), '--rt', '0=rgba8_unorm', '--bind', '0.1=file=' + data, '--push',
        'w=u32:%d' % W, '--check', '--dump', 'rt0=' + out, '--runs', '2', '--warmup', '1', '--warmup-ms', '1',
        '--json', '--quiet'], log_path('unorm.log'))
    got = read_dump(out, 4 * W * W)[0::4]
    near = other = 0
    off = []
    for v, g in zip(vals, got):
        x = v * 255.0
        n = int(math.floor(x + 0.5))       # ties up: the exact midpoints are not f32 values here
        if g == n:
            near += 1
        elif g in (math.floor(x), math.ceil(x)):
            other += 1
            if len(off) < 8:
                off.append(dict(value=v, times_255=x, got=g, nearest=n))
        else:
            off.append(dict(value=v, times_255=x, got=g, nearest=n, neither_neighbour=True))
    return dict(values=len(vals), round_nearest=near, other_neighbour=other, examples=off)


RAMP = (0, 255, 37, 200)      # the 4x1 R8 texture the filter probe reads
FILTER_N = 4096               # samples across each of two texel pairs
FILTER_FINE_BITS = 20         # the fine sweep's spacing, 2^-20 of a texel
FILTER_FINE = 1 << 16         # its samples: the first 1/16 of the pair (0, 255)


def filter_coords():
    """(first texel, x, u) of every read of the filter probe: x the position between the pair's two texels
    in texel units, u the coordinate on the 4-wide texture (4 u - 1/2 = texel + x, every u exact in f32).
    The pairs (0, 255) and (37, 200) at FILTER_N points, then (0, 255) every 2^-20 over its first 1/16,
    finer than any weight grid the result can show, so a change lands on a grid point or between two"""
    pts = [(0, j / FILTER_N) for j in range(FILTER_N)] + [(2, j / FILTER_N) for j in range(FILTER_N)] + \
        [(0, j / 2.0 ** FILTER_FINE_BITS) for j in range(FILTER_FINE)]
    return [(t, x, (t + 0.5 + x) / 4.0) for t, x in pts]


def grid_weight(x, bits, mode):
    """x on a grid of 2^bits steps, truncated or to the nearest (ties up, as a fixed-point add of a half)"""
    g = 1 << bits
    k = math.floor(x * g) if mode == 'truncate' else math.floor(x * g + 0.5)
    return Fraction(k, g)


def filter_analysis(got):
    """what the reads of filter_coords() say: the weight grid, its rounding, the blend's error"""
    n, F = FILTER_N, FILTER_FINE
    coarse, fine = got[:2 * n], got[2 * n:2 * n + F]
    r = dict(distinct=len(set(coarse[:n])), samples=n)
    changes = [j for j in range(1, F) if fine[j] != fine[j - 1]]
    gaps = sorted({b - a for a, b in zip(changes, changes[1:])})
    r['fine_changes'] = len(changes)
    r['grid_bits'] = r['weights_round'] = None
    if len(changes) < 2:
        r['grid_note'] = '%s in the first 1/16 of a texel: a grid of 5 bits or coarser' % (
            'one change' if changes else 'no change')
    elif len(gaps) > 1 or gaps[0] & (gaps[0] - 1):
        r['grid_note'] = 'irregular: the changes are %s to %s samples of 2^-%d apart' % (
            gaps[0], gaps[-1], FILTER_FINE_BITS)
    else:
        step = gaps[0]
        if step == 1:
            r['grid_note'] = 'a change at every sample: a grid of %d bits or finer' % FILTER_FINE_BITS
        else:
            r['grid_bits'] = FILTER_FINE_BITS - step.bit_length() + 1
            offs = sorted({c % step for c in changes})
            r['change_offsets'] = [o / step for o in offs]
            r['weights_round'] = 'truncate' if offs == [0] else 'nearest' if offs == [step // 2] else 'other'
    # the blend against the weighted sum at the weight the grid gives (the exact weight when unseen)
    worst, exact = 0.0, True
    for k, (t, x, u) in enumerate(filter_coords()[:2 * n]):
        a, b = Fraction(RAMP[t], 255), Fraction(RAMP[t + 1], 255)
        w = grid_weight(x, r['grid_bits'], r['weights_round']) if r['weights_round'] in ('truncate', 'nearest') \
            else Fraction(x)
        want = a + w * (b - a)
        worst = max(worst, abs(Fraction(coarse[k]) - want))
        exact = exact and f32bits(coarse[k]) == f32bits(round_f32(want, 'even'))
    r['error_against'] = 'its grid weight' if r['weights_round'] in ('truncate', 'nearest') else 'the exact weight'
    r['max_error_abs'] = float(worst)
    r['max_error_255'] = float(worst * 255)
    r['blend_exact'] = exact
    return r


def value_filter(p):
    pgm = write('ramp.pgm', b'P5\n4 1\n255\n' + bytes(RAMP))
    co = filter_coords()
    ub = write('filter_u.bin', b''.join(struct.pack('<f', u) for t, x, u in co))
    cs = compile_glsl('cs_sample', CS_SAMPLE, 'comp')
    out = dump('filter_r.bin')
    sp([cs, '--size', str(len(co)), '--bind', '0.0=file=' + ub, '--bind',
        '0.5=format=r8_unorm,file=%s,filter=linear,wrap=clamp' % pgm, '--check', '--dump', '0.1=' + out,
        '--runs', '2', '--warmup', '1', '--warmup-ms', '1', '--compile-runs', '1', '--json', '--quiet'], log_path('filter.log'))
    return filter_analysis(struct.unpack('<%df' % len(co), read_dump(out, 4 * len(co))))


def value_repeat(p):
    tab = write('repeat_tab.bin', b''.join(struct.pack('<I', j) for j in range(256)))
    idx = list(range(0, 1024)) + list(range(2 ** 22 - 1024, 2 ** 22 + 1024)) + \
        list(range(-2 ** 22 - 1024, -2 ** 22 + 1024))
    ib = write('repeat_idx.bin', b''.join(struct.pack('<i', i) for i in idx))
    cs = compile_glsl('cs_repeat', CS_REPEAT, 'comp')
    out = dump('repeat_r.bin')
    sp([cs, '--size', str(len(idx)), '--bind', '0.0=file=' + ib, '--bind',
        '0.5=format=r32_uint,size=256x1,raw=%s,filter=nearest,wrap=repeat' % tab, '--check', '--dump',
        '0.1=' + out, '--runs', '2', '--warmup', '1', '--warmup-ms', '1', '--compile-runs', '1', '--json', '--quiet'],
       log_path('repeat.log'))
    got = struct.unpack('<%dI' % len(idx), read_dump(out, 4 * len(idx)))
    r = {}
    for name, lo, hi in (('near_0', 0, 1024), ('near_plus_2p22', 1024, 3072), ('near_minus_2p22', 3072, 5120)):
        bad = [(idx[j], got[j], idx[j] % 256) for j in range(lo, hi) if got[j] != idx[j] % 256]
        r[name] = dict(reads=hi - lo, wrong=len(bad), examples=[dict(i=i, got=g, want=w) for i, g, w in bad[:6]])
    return r


def bc4_block(r0, r1):
    idx = 0
    for t in range(16):
        idx |= (t % 8) << (3 * t)
    return bytes([r0, r1]) + idx.to_bytes(6, 'little')


def bc4_palette(r0, r1):
    if r0 > r1:
        return [r0, r1] + [((7 - i) * r0 + i * r1) / 7.0 for i in range(1, 7)]
    return [r0, r1] + [((5 - i) * r0 + i * r1) / 5.0 for i in range(1, 5)] + [0, 255]


def value_bc4(p):
    pairs = [(255, 0), (0, 255), (200, 37), (37, 200), (128, 127), (127, 128), (1, 0), (0, 1), (254, 1), (90, 90),
             (17, 241), (241, 17)]
    nb = len(pairs)
    W, H = 4 * nb, 4
    # the blocks in a row: block b's texel (x, y) at (4b + x, y)
    raw = write('bc4.bin', b''.join(bc4_block(a, b) for a, b in pairs))
    cs = compile_glsl('cs_fetch', CS_FETCH, 'comp')
    out = dump('bc4_r.bin')
    sp([cs, '--size', str(W * H), '--push', 'w=u32:%d' % W, '--bind',
        '0.5=format=bc4_unorm,size=%dx%d,raw=%s,filter=nearest' % (W, H, raw), '--check', '--dump',
        '0.1=' + out, '--runs', '2', '--warmup', '1', '--warmup-ms', '1', '--compile-runs', '1', '--json', '--quiet'],
       log_path('bc4.log'))
    got = struct.unpack('<%df' % (W * H), read_dump(out, 4 * W * H))
    worst_ulp = worst_abs = 0.0
    entries = []
    for bi, (a, b) in enumerate(pairs):
        pal = bc4_palette(a, b)
        for t in range(16):
            x, y = 4 * bi + t % 4, t // 4
            g = got[y * W + x]
            exact = pal[t % 8] / 255.0
            e = f32(exact)
            d_abs = abs(g - exact)
            d_ulp = ulps(g, e)
            worst_abs, worst_ulp = max(worst_abs, d_abs), max(worst_ulp, d_ulp)
            if t < 8 and f32bits(g) != f32bits(e):
                entries.append(dict(r0=a, r1=b, index=t, got=g, exact_f32=e, ulps=d_ulp))
    return dict(texels=W * H, max_error_ulps=worst_ulp, max_error_abs=worst_abs, max_error_255=worst_abs * 255,
                off_entries=len(entries), examples=entries[:10])


def info_subtexel(p):
    out = run_tool(['vulkaninfo'], log_path('vulkaninfo.log'), 120)
    gpus = []
    cur = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith('deviceName') and '=' in s:
            cur = dict(name=s.split('=', 1)[1].strip())
            gpus.append(cur)
        elif s.startswith('subTexelPrecisionBits') and '=' in s and cur is not None and 'subtexel' not in cur:
            cur['subtexel'] = int(s.split('=', 1)[1].strip())
    if not gpus or not any('subtexel' in g for g in gpus):
        raise ProbeError('vulkaninfo printed no subTexelPrecisionBits (raw/vulkaninfo.log)')
    pick = next((g for g in gpus if '1030' in g['name'] and 'subtexel' in g), None) or \
        next(g for g in gpus if 'subtexel' in g)
    return dict(device=pick['name'], subtexel=pick['subtexel'], devices=gpus)


UNITS = dict(dispatch=unit_setup_binding, draw=unit_draw_empty, points=unit_points, rop=unit_rop)
VALUES = dict(blend=value_blend, unorm=value_unorm, filter=value_filter, repeat=value_repeat, bc4=value_bc4)


# ---------------------------------------------------------------------------
# what the measures set in the model

def q8(v):
    """a price on the prelude's 1/8 grid, never below 0"""
    return max(0.0, round(v * 8) / 8)


def model_lines(res):
    """(line, comment) pairs for the model block, and the measures with no key"""
    L, extra, used = [], [], set()

    def cost(n):
        r = res.get(n)
        return r['cost'] if r and 'cost' in r else None

    def note(n):
        used.add(n)
        r = res[n]
        b = r.get('bytes_per_step')
        return '%s %.2f +- %.2f%s' % (n, r['cost'], r['err'], (', B/step %.0f' % b) if b is not None else '')

    c = cost('mul16_mad')
    if c is not None:
        L.append(('mul16: %g' % q8(c), '#184 ' + note('mul16_mad') + ' (the masks and the add in the step, as '
                  'imad has its add)'))
        L.append(('fuse xmad(a: u16, b: u16, c: int) = a * b + c : mul16', None))
        L.append(('fuse mul16(a: u16, b: u16) = a * b : mul16', None))
    c = cost('mul32x16_mad')
    if c is not None:
        L.append(('fuse xmad32(a: int, b: u16, c: int) = a * b + c : %g' % q8(c), '#185 ' + note('mul32x16_mad')))
    c, f = cost('fsat_fma'), cost('fma_contract')
    if c is not None and f is not None:
        L.append(('fuse fsat(a: float) = min(max(a, 0.0), 1.0) : %g on fadd fmul' % q8(c - f),
                  '#186 %s against %s: %.2f' % (note('fsat_fma'), note('fma_contract'), c - f)))
    for cls, n, minus, what in (('bfe', 'bfe_builtin', 1, 'its add'), ('brev', 'brev', 1, 'its add'),
                                ('mulhi', 'mulhi', 1, 'its add'), ('funnel', 'funnel', 1, 'its add'),
                                ('byteperm', 'byteswap', 1, 'its add'), ('med3', 'med3_clamp', 2, 'its two masks'),
                                ('sad4x8', 'sad4x8', 1, 'its add')):
        c = cost(n)
        if c is not None:
            L.append(('%s: %g' % (cls, q8(c - minus)), '#156 %s less %s' % (note(n), what)))
    c = cost('tex_nearest')
    if c is not None:
        L.append(('tex_nearest: %g' % q8(c - 1), '#115 %s less its add' % note('tex_nearest')))
    st = res.get('subtexel_bits', {})
    fw = res.get('filter_weights', {})
    if 'subtexel' in st:
        L.append(('subtexel: %d' % st['subtexel'], '#117 vulkaninfo on %s%s' % (
            st['device'], '; the filter probe sees a weight grid of %s bits' % fw['grid_bits']
            if fw.get('grid_bits') else '; the filter probe: %s' % fw['grid_note'] if fw.get('grid_note') else '')))
    if 'blend_exact' in fw:
        grid = ('weights on a grid of %d bits, %s' % (fw['grid_bits'], 'truncated' if fw['weights_round'] ==
                'truncate' else 'to the nearest' if fw['weights_round'] == 'nearest' else 'neither truncated nor '
                'to the nearest')) if fw['grid_bits'] else fw['grid_note']
        if fw['blend_exact'] and fw['weights_round'] in ('truncate', 'nearest'):
            L.append(('tex_within: 0.5', '#116 every bilinear R8 result is the f32 nearest the weighted sum at its '
                      'grid weight; %s' % grid))
        else:
            L.append(('', '#116 tex_within (ulps of the result) not written: the bilinear R8 result is at most %.3g '
                      '(%.3g of an 8-bit step) from the weighted sum at %s, an absolute error; %s' % (
                          fw['max_error_abs'], fw['max_error_255'], fw['error_against'], grid)))
    rb = res.get('rop_blend', {})
    br = res.get('blend_rounding', {})
    if br.get('add_moved') and br.get('add_unblended') == br['add_moved']:
        L.append(('', '#207 rop_blend not written: blend_rounding shows R32_SFLOAT was not blended'))
    elif 'add_l2' in rb:
        L.append(('rop_blend: %g' % q8(rb['add_l2']['fadd_steps']),
                  '#207 a point blended ADD ONE ONE into R32_SFLOAT 256x256: %.3g ns, %.2f fadd steps (no blend '
                  '%.2f; a float atomicAdd on the same cells %.2f)' % (
                      rb['add_l2']['ns_point'], rb['add_l2']['fadd_steps'], rb['none_l2']['fadd_steps'],
                      rb['atomic_l2']['fadd_steps'])))
    # measured, with no class or parameter yet
    for n, r in res.items():
        if 'cost' in r and n not in used:
            p = next(p for p in P if p['name'] == n)
            extra.append('%s %s: %.2f +- %.2f' % (', '.join('#%d' % i for i in p['issues']), n, r['cost'], r['err']))
    for n in ('setup_binding', 'draw_empty'):
        if 'fadd_steps' in res.get(n, {}):
            p = next(p for p in P if p['name'] == n)
            extra.append('%s %s: %.3g fadd steps (%.3g ns)' % (', '.join('#%d' % i for i in p['issues']), n,
                         res[n]['fadd_steps'], res[n].get('ns_binding', res[n].get('ns_draw'))))
    return L, extra


def model_text(res, dev):
    L, extra = model_lines(res)
    out = ['# measured on %s (%s) by tests/gt1030/probe.py, %s%s\n' % (
        dev.get('name'), dev.get('driver_info'), time.strftime('%Y-%m-%d'), ' (--quick: the numbers mean nothing)'
        if A.quick else ''),
        '# ucas model.ucas, then `use gt1030_probed`\n', 'model gt1030_probed : gt1030 {\n']
    for line, cm in L:
        if cm:
            out.append('  # %s\n' % cm)
        if line:
            out.append('  %s\n' % line)
    if extra:
        out.append('  # measured, with no class or parameter in ucas yet (cost a step over fadd):\n')
        out += ['  #   %s\n' % e for e in extra]
    out.append('}\n')
    return ''.join(out)


# ---------------------------------------------------------------------------
# the report

def summary(p, r):
    """what a unit or value probe found, in a line"""
    n = p['name']
    if n == 'setup_binding':
        return 'one more sampler: %.3g ns a dispatch (%.3g +- %.2g), %.3g fadd steps' % (
            r['ns_binding'], r['ns_one_more'], r['err_ns'], r['fadd_steps'])
    if n == 'draw_empty':
        return 'a draw with no fragment: %.3g ns, %.3g fadd steps' % (r['ns_draw'], r['fadd_steps'])
    if n == 'points':
        return '; '.join('%s: %.3g ns a point, %.3g fadd steps' % (k, r[k]['ns_vertex'], r[k]['fadd_steps'])
                         for k in ('onscreen', 'offscreen'))
    if n == 'rop_blend':
        return '; '.join('%s: %.3g fadd steps' % (k, v['fadd_steps']) for k, v in r.items()
                         if isinstance(v, dict) and 'fadd_steps' in v)
    if n == 'blend_rounding':
        if r['add_moved'] and r['add_unblended'] == r['add_moved']:
            return 'NOT BLENDED: the target holds the second value alone at every pixel (the device cannot blend ' \
                   'R32_SFLOAT, or snx-shaderperf did not ask it to)'
        return ('ADD: %d of %d sums are round to nearest, %d toward zero%s; factors: %s' % (
            r['add_round_nearest'], r['add_pairs'], r['add_toward_zero'],
            (' (first off: %s)' % r['add_differ'][0]) if r['add_differ'] else '',
            ', '.join('%s %d of %d' % (k, v, r['alpha_triples']) for k, v in r['alpha_forms'].items())))
    if n == 'unorm_rounding':
        return '%d of %d to the nearest, %d to the other neighbour%s' % (
            r['round_nearest'], r['values'], r['other_neighbour'], (' (e.g. %s)' % r['examples'][0])
            if r['examples'] else '')
    if n == 'filter_weights':
        grid = ('a weight grid of %d bits, the weight %s (the results change %s)' % (
            r['grid_bits'], {'truncate': 'truncated', 'nearest': 'rounded to nearest'}.get(
                r['weights_round'], 'neither truncated nor rounded to nearest'),
            'at a grid point' if r['weights_round'] == 'truncate' else 'half a grid step past one'
            if r['weights_round'] == 'nearest' else 'at %s of a grid step' % ', '.join(
                '%.3g' % o for o in r['change_offsets']))) if r['grid_bits'] else r['grid_note']
        return '%s; %d distinct results of %d from 0 to 1; the blend %s the weighted sum at %s' % (
            grid, r['distinct'], r['samples'], 'is the f32 nearest' if r['blend_exact'] else
            'is at most %.3g (%.3g of an 8-bit step) from' % (r['max_error_abs'], r['max_error_255']),
            r['error_against'])
    if n == 'subtexel_bits':
        return 'subTexelPrecisionBits %d on %s' % (r['subtexel'], r['device'])
    if n == 'repeat_large':
        return '; '.join('%s: %d of %d wrong%s' % (k, v['wrong'], v['reads'], (' (e.g. %s)' % v['examples'][0])
                                                   if v['examples'] else '') for k, v in r.items())
    if n == 'bc4_palette':
        return 'at most %.3g/255 (%.3g ulps) from the exact palette; %d of %d entries not the exact f32%s' % (
            r['max_error_255'], r['max_error_ulps'], r['off_entries'], 8 * 12,
            (' (e.g. %s)' % r['examples'][0]) if r['examples'] else '')
    return json.dumps(r)[:600]


def fadd_note(p, r):
    """a unit's fadd, timed just before it (rop's: before its L2 runs)"""
    f = r.get('fadd_l2', r) if p['kind'] == 'unit' else {}
    if not f.get('fadd_ns'):
        return ''
    return ' (fadd timed just before: %.3g ps a lane-step%s)' % (
        f['fadd_ns'] * 1e3, (', SM %d MHz' % f['fadd_clock_mhz']) if f.get('fadd_clock_mhz') else '')


def report(res, dev, model, ucas_check):
    L = ['# GT 1030 probes\n\n',
         'Device: %s (%s). %s\n\n' % (dev.get('name'), dev.get('driver_info'),
                                      '**--quick: a smoke run, the numbers mean nothing.**' if A.quick else
                                      'Step costs are ratios to fadd in the same run; units are in fadd steps by '
                                      'the fadd lane-step of the step run before them.'),
         'model.ucas: %s\n\n' % ucas_check]
    issues = sorted({i for p in P for i in p['issues']})
    for i in issues + [None]:
        ps = [p for p in P if (i in p['issues'] if i else not p['issues']) and p['name'] in res]
        if not ps:
            continue
        L.append('## %s\n\n' % ('#%d' % i if i else 'The queue\'s lines with no issue'))
        for p in ps:
            r = res.get(p['name'], {})
            head = '- `%s`%s' % (p['name'], (' (and %s)' % p['queue']) if p.get('queue') and i else '')
            if p['kind'] == 'step':
                if 'cost' in r:
                    pr = r.get('ucas_price')
                    L.append('%s: %.2f +- %.2f fadd%s%s%s. %s\n' % (
                        head, r['cost'], r['err'], (', B/step %.0f' % r['bytes_per_step'])
                        if r.get('bytes_per_step') is not None else '',
                        (', gt1030 prices it %g' % pr) if pr is not None else '',
                        (', SM %d MHz' % r['clock_mhz']) if r.get('clock_mhz') else '', p['note']))
                else:
                    L.append('%s: FAILED: %s. %s\n' % (head, r.get('error', 'not run'), p['note']))
            elif p['kind'] == 'absent':
                L.append('%s: not run: needs %s%s\n' % (head, p['needs'], ('; this snx-shaderperf has something '
                         'named %s now, so the probe can be written' % p['key']) if r.get('tool_has_it') else ''))
            elif p['kind'] == 'none':
                L.append('%s%s: no probe: %s\n' % (head, (' (%s)' % p['queue']) if p.get('queue') and not i else '',
                                                    p['why']))
            else:
                if 'error' in r:
                    L.append('%s: FAILED: %s\n' % (head, r['error']))
                else:
                    L.append('%s: %s. %s%s\n' % (head, p['note'], summary(p, r), fadd_note(p, r)))
        L.append('\n')
    L.append('## Model\n\n```\n%s```\n' % model)
    with open(os.path.join(OUT, 'report.md'), 'w') as fh:
        fh.write(''.join(L))


def find_ucas():
    if A.ucas:
        return A.ucas
    for cand in (os.path.join(HERE, '..', '..', 'ucas'), shutil.which('ucas')):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    return None


def ucas_reads(path):
    u = find_ucas()
    if u is None:
        return 'not read back: no ucas binary (give --ucas PATH)'
    p = subprocess.run([u, os.path.basename(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       cwd=os.path.dirname(path), timeout=60)
    bad = [line for line in p.stdout.splitlines() if 'error' in line]
    if p.returncode or bad:
        return 'ucas (%s) REFUSED it: %s' % (u, ' / '.join(bad[:3]) or 'exit %d' % p.returncode)
    return 'read back by ucas (%s) without an error' % u


# ---------------------------------------------------------------------------

def main():
    global A, OUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sp', default=os.path.expanduser('~/ScarletNX/build/rel/snx-shaderperf'))
    ap.add_argument('--out', default=os.path.expanduser('~/ucas/work/gt1030/probe-' + time.strftime('%Y%m%d-%H%M%S')))
    ap.add_argument('--only', help='probes by name, comma-separated')
    ap.add_argument('--issue', help='probes of these issues, comma-separated numbers')
    ap.add_argument('--runs', type=int, default=20)
    ap.add_argument('--tries', type=int, default=3, help='runs of a step whose interval is wide')
    ap.add_argument('--max-rel-err', type=float, default=0.03)
    ap.add_argument('--timeout', type=int, default=600, help='seconds one snx-shaderperf run may take')
    ap.add_argument('--quick', action='store_true', help='a small grid and few samples: it runs, nothing more')
    ap.add_argument('--list', action='store_true', help='the probes and their issues, and stop')
    ap.add_argument('--ucas', help='the ucas binary that reads model.ucas back')
    A = ap.parse_args()
    if A.list:
        for p in P:
            print('%-26s %-6s %s' % (p['name'], p['kind'], ', '.join('#%d' % i for i in p['issues']) or
                                     p.get('queue', '')))
        return 0
    only = set(A.only.split(',')) if A.only else None
    if only:
        unknown = only - {p['name'] for p in P}
        if unknown:
            sys.exit('probe.py: no probe named %s (--list shows them)' % ', '.join(sorted(unknown)))
    issues = None
    if A.issue:
        try:
            issues = {int(s.strip().lstrip('#')) for s in A.issue.split(',')}
        except ValueError:
            sys.exit('probe.py: --issue %s: issue numbers, comma-separated' % A.issue)
        unknown = issues - {i for p in P for i in p['issues']}
        if unknown:
            sys.exit('probe.py: no probe for issue %s (--list shows them)' % ', '.join('#%d' % i for i in unknown))
    for tool in (A.sp, 'glslangValidator', 'spirv-val'):
        if not shutil.which(tool) and not os.access(tool, os.X_OK):
            sys.exit('probe.py: %s not found' % tool)
    OUT = os.path.abspath(A.out)
    for d in ('raw', 'src', 'spv', 'data'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    helptext = subprocess.run([A.sp, '--help'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    res, dev = {}, {}
    chosen = [p for p in P if (not only or p['name'] in only) and (not issues or set(p['issues']) & issues)]
    order = {'step': 0, 'unit': 1, 'value': 2, 'info': 3, 'absent': 4, 'none': 5}
    chosen.sort(key=lambda p: (bool(p.get('slow')), order[p['kind']]))
    t0 = time.time()
    for k, p in enumerate(chosen):
        t1 = time.time()
        r = res.setdefault(p['name'], {})
        try:
            if p['kind'] == 'step':
                r.update(run_step(p))
            elif p['kind'] == 'unit':
                r.update(UNITS[p['how']](p))
            elif p['kind'] == 'value':
                r.update(VALUES[p['how']](p))
            elif p['kind'] == 'info':
                r.update(info_subtexel(p))
            elif p['kind'] == 'absent':
                r['tool_has_it'] = bool(p.get('key') and re.search(p['key'], helptext))
        except ProbeError as e:
            r['error'] = str(e)
        if not dev:
            for f in sorted(os.listdir(os.path.join(OUT, 'raw'))):
                if f.endswith('.json'):
                    dev = json.load(open(os.path.join(OUT, 'raw', f))).get('device', {})
                    break
        state = ('%.2f +- %.2f' % (r['cost'], r['err'])) if 'cost' in r else \
            ('FAILED: ' + r['error']) if 'error' in r else p['kind']
        print('[%d/%d %4.0fs] %-26s %s (%.1fs)' % (k + 1, len(chosen), time.time() - t0, p['name'], state,
                                                   time.time() - t1), flush=True)
        with open(os.path.join(OUT, 'probe.json'), 'w') as fh:
            json.dump(dict(device=dev, quick=A.quick, results=res), fh, indent=1)
    model = model_text(res, dev)
    path = os.path.join(OUT, 'model.ucas')
    with open(path, 'w') as fh:
        fh.write(model)
    check = ucas_reads(path)
    report(res, dev, model, check)
    failed = [n for n, r in res.items() if 'error' in r]
    print('\n' + model)
    print('model.ucas: %s' % check)
    print('report: %s' % os.path.join(OUT, 'report.md'))
    if failed:
        print('%d probe%s failed: %s' % (len(failed), 's' if len(failed) > 1 else '', ', '.join(failed)))
    return 1 if failed or 'REFUSED' in check else 0


if __name__ == '__main__':
    sys.exit(main())
