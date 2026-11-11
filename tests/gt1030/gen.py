#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the GT 1030 calibration corpus: GLSL in src/, SPIR-V in spv/, manifest.json.

Every test is one step expression, written out UNROLL times inside a loop of pc.iters
iterations.  Each test is built twice, with UNROLL = A and UNROLL = 2A, so that the
difference of the two times is the cost of A*C steps alone (the loop, the prologue and
the dispatch cancel).  run.py times both builds of every test beside the fadd reference
in one snx-shaderperf run and turns the differences into ucas model costs.

Step placeholders: {a} this chain, {b} {c} {d} the next chains of the ring (chain j
reads chains j+1, j+2, j+3 mod C, updated in order, so integer chains cannot be folded
by reassociation), {u} the unroll index, {j} the chain index.  Invariants k, k2 (uint),
kf, kf2 (float), kd, kd2 (double), ki (int) come from push constants, unknown to the
driver's compiler.

Usage: gen.py [--no-build]
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# extensions: GLSL #extension names per feature tag
EXT = {
    'i64': ['GL_EXT_shader_explicit_arithmetic_types_int64'],
    'i16': ['GL_EXT_shader_explicit_arithmetic_types_int16'],
    'i8': ['GL_EXT_shader_explicit_arithmetic_types_int8'],
    'f64': [],
    'clock': ['GL_ARB_shader_clock'],
    'rtclock': ['GL_EXT_shader_realtime_clock'],
    'sm': ['GL_NV_shader_sm_builtins'],
    'sg': ['GL_KHR_shader_subgroup_basic', 'GL_KHR_shader_subgroup_arithmetic',
           'GL_KHR_shader_subgroup_ballot', 'GL_KHR_shader_subgroup_vote',
           'GL_KHR_shader_subgroup_shuffle', 'GL_KHR_shader_subgroup_shuffle_relative',
           'GL_KHR_shader_subgroup_clustered', 'GL_KHR_shader_subgroup_quad'],
    'sgrot': ['GL_KHR_shader_subgroup_rotate'],
    'part': ['GL_NV_shader_subgroup_partitioned'],
    'afloat': ['GL_EXT_shader_atomic_float'],
    'a64': ['GL_EXT_shader_atomic_int64'],
    'dot': ['GL_EXT_spirv_intrinsics'],
    'cf': ['GL_EXT_control_flow_attributes'],
    'expect': ['GL_EXT_expect_assume'],
}

# SPIR-V instructions that GLSL has no built-in for (SPV_KHR_integer_dot_product,
# 4x8-bit packed format = 0).  Pascal (sm_61) has DP4A, and the driver reports the
# 4x8 packed forms as accelerated.
DOT_DECLS = '''
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4451)
uint udot4x8(uint a, uint b, spirv_literal int fmt);
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4450)
int sdot4x8(uint a, uint b, spirv_literal int fmt);
spirv_instruction(extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4454)
uint udot4x8sat(uint a, uint b, uint acc, spirv_literal int fmt);
'''

UMULHI = 'uint umulhi(uint x, uint y) { uint h, l; umulExtended(x, y, h, l); return h; }\n'

SHARED = 'shared uint sh[4096 + 64];\n'
SHARED_F = 'shared float shf[4096 + 64];\n'

# kinds of chain values: GLSL type, initial value of chain {i} from seed s, sink to uint
TYPES = {
    'uint': ('uint', 's * 0x9e3779b1u + {i}u * 0x85ebca6bu + 1u', '{x}'),
    'int': ('int', 'int(s * 0x9e3779b1u + {i}u * 0x85ebca6bu) | 1', 'uint({x})'),
    'float': ('float', '1.0 + float(s & 255u) * 1e-4 + float({i}) * 1e-3', 'floatBitsToUint({x})'),
    'double': ('double', '1.0lf + double(s & 255u) * 1e-4lf + double({i}) * 1e-3lf',
               '(unpackDouble2x32({x}).x ^ unpackDouble2x32({x}).y)'),
    'u64': ('uint64_t', 'uint64_t(s) * 0x9e3779b97f4a7c15ul + uint64_t({i}) + 1ul',
            'uint({x} ^ ({x} >> 32))'),
    'u16': ('uint16_t', 'uint16_t(s + {i}u * 977u)', 'uint({x})'),
    'u8': ('uint8_t', 'uint8_t(s + {i}u * 13u) | uint8_t(1)', 'uint({x})'),
}

# ---------------------------------------------------------------------------
# The tests.  ops: the ucas cost classes one step costs (as the model prices the step's
# SPIR-V).  fit: the primary test of a class (the solve uses these); the rest are checks,
# priced by the fitted model and compared.  Classes not in the ucas model are "extra"
# (named x_...): each gets its own cost from its test, for a model that grows them.

T = []


def t(name, step, ops, ty='uint', fit=False, exts=(), decls='', pre='', post='', C=8,
      lat=True, latC=None, ring=None, iters=None, note='', kind='alu', A=8, wg=128,
      vkext=(), args=(), qual='', **extra):
    if ring is None:
        ring = any(p in (step or '') for p in ('{b}', '{c}', '{d}'))
    if latC is None and any(p in (step or '') for p in ('{c}', '{d}')):
        lat = False     # in a 2-chain ring {c} is {a}: the driver simplifies
    T.append(dict(name=name, step=step, ops=ops, ty=ty, fit=fit, exts=list(exts), decls=decls,
                  pre=pre, post=post, C=C, lat=lat, latC=latC or (2 if ring else 1), ring=ring,
                  iters=iters, note=note, kind=kind, A=A, wg=wg, vkext=list(vkext),
                  args=list(args), qual=qual, **extra))


# --- 32-bit integer ---------------------------------------------------------
t('iadd', '{a} + {b}', {'iadd': 1}, fit=True)
t('isub', '{b} - {a}', {'iadd': 1}, lat=False, note='latency: b - (b - a) folds in a 2-chain ring')
t('xor_add', '({a} ^ {b}) + k', {'logic': 1, 'iadd': 1}, fit=True)
t('and_add', '({a} & {b}) + k', {'logic': 1, 'iadd': 1})
t('or_add', '({a} | {b}) + k', {'logic': 1, 'iadd': 1})
t('not_add', '~{a} + {b}', {'logic': 1, 'iadd': 1},
  note='~a + b = b - a - 1: the not can become a negated add operand')
t('lop3_select', '(({a} & {b}) | (~{a} & {c})) + k', {'logic': 4, 'iadd': 1},
  note='a 3-input function: one LOP3 on sm_61')
t('lop3_xor3', '({a} ^ {b} ^ {c}) + k', {'logic': 2, 'iadd': 1}, note='one LOP3')
t('shl_var', '{a} << {b}', {'shift': 1}, fit=True,
  note='counts past 31 are undefined in SPIR-V; the hardware clamps')
t('shr_var', '{a} >> {b}', {'shift': 1})
t('sar_var', '{a} >> {b}', {'shift': 1}, ty='int')
t('shr_const_add', '({a} >> 7) + {b}', {'shift': 1, 'iadd': 1})
t('shl_const_add', '({a} << 3) + {b}', {'shift': 1, 'iadd': 1}, note='ISCADD: shift and add in one')
t('rotl', '({a} << 7) | ({a} >> 25) ^ {b}', {'shift': 2, 'logic': 2},
  note='no funnel shift on sm_61: two shifts and a LOP3')
t('imul', '{a} * {b}', {'imul': 1}, fit=True, note='XMAD sequence on sm_61')
t('imad', '{a} * {b} + {c}', {'imul': 1, 'iadd': 1}, note='the add folds into the XMAD chain')
t('imul_const', '{a} * 0x9e3779b1u + {b}', {'imul': 1, 'iadd': 1})
t('imul_pow2', '{a} * 8u + {b}', {'shift': 1, 'iadd': 1}, note='ucas prices x*2^n as a shift')
t('umulhi', 'umulhi({a}, {b}) + k', {'x_imulhi': 1, 'iadd': 1}, decls=UMULHI, fit=True)
t('udiv', '({a} / {b}) + k', {'idiv': 1, 'iadd': 1}, fit=True, note='emulated: no divide unit')
t('umod', '({a} % {b}) + k', {'idiv': 1, 'iadd': 1})
t('sdiv', '({a} / {b}) + ki', {'idiv': 1, 'iadd': 1}, ty='int')
t('udiv_const', '({a} / 7u) + {b}', {'idivk': 1, 'iadd': 1}, fit=True, note='multiply-high and shift')
t('umod_const', '({a} % 7u) + {b}', {'idivk': 1, 'iadd': 1})
t('sdiv_const', '({a} / 7) + {b}', {'idivk': 1, 'iadd': 1}, ty='int')
t('udiv_pow2', '({a} / 16u) + {b}', {'idivk': 1, 'iadd': 1}, note='a shift; ucas prices it as idivk')
t('cmp_select', '({a} < {b}) ? {c} : {d}', {'cmp': 1, 'select': 1}, fit=True, lat=False,
  note='the values only move between chains, nothing to fold')
t('select', 'p{u4} ? {b} : {c}', {'select': 1}, fit=True, lat=False,
  pre='bool p0 = (s & 1u) != 0u, p1 = (s & 2u) != 0u, p2 = (s & 4u) != 0u, p3 = (s & 8u) != 0u;',
  note='predicates computed before the loop')
t('cmp_select_eq', '({a} == {b}) ? {c} : {d}', {'cmp': 1, 'select': 1}, lat=False)
t('umin_add', 'min({a} + k, {b})', {'minmax': 1, 'iadd': 1}, fit=True)
t('smax_add', 'max({a} + ki, {b})', {'minmax': 1, 'iadd': 1}, ty='int')
t('uclamp_add', 'clamp({a} + k, {b}, {c})', {'minmax': 2, 'iadd': 1}, note='ucas: a clamp is a min and a max')
t('bitcount', 'uint(bitCount({a})) + {b}', {'bitcnt': 1, 'iadd': 1}, fit=True)
t('findmsb', 'uint(findMSB({a})) + {b}', {'bitcnt': 1, 'iadd': 1})
t('findlsb', 'uint(findLSB({a})) + {b}', {'bitcnt': 1, 'iadd': 1})
t('bitreverse', 'bitfieldReverse({a}) + {b}', {'bitcnt': 1, 'iadd': 1})
t('bfe', 'bitfieldExtract({a}, int(k2 & 15u), 9) + {b}', {'shift': 1, 'iadd': 1},
  note='ucas prices OpBitField* as a shift')
t('bfi', 'bitfieldInsert({a}, {b}, int(k2 & 15u), 9)', {'shift': 1})
# --- conversions and bitcasts -----------------------------------------------
t('f2u2f', 'float(uint({a} * 1000.0)) + {b}', {'conv': 2, 'fmul': 1, 'fadd': 1}, ty='float', fit=True)
t('i2f_add', 'float(int({b} * 1000.0)) + {a}', {'conv': 2, 'fmul': 1, 'fadd': 1}, ty='float')
t('floor_add', 'floor({a} * 3.0) + {b}', {'conv': 1, 'fmul': 1, 'fadd': 1}, ty='float',
  note='ucas prices floor/ceil/fract/round as conv')
t('fract_add', 'fract({a} * 3.0) + {b}', {'conv': 1, 'fmul': 1, 'fadd': 1}, ty='float')
t('round_add', 'roundEven({a} * 3.0) + {b}', {'conv': 1, 'fmul': 1, 'fadd': 1}, ty='float')
t('f16_roundtrip', 'unpackHalf2x16(packHalf2x16(vec2({a}, 0.0))).x + {b}', {'conv': 2, 'fadd': 1},
  ty='float', note='f32->f16->f32: the driver drops the round trip; shaderFloat16 is false, only conversions')
t('f16_pack_add', 'unpackHalf2x16(packHalf2x16(vec2({a}, {b})) + k2).x', {'conv': 3, 'iadd': 1}, ty='float',
  note='two F2F to f16 and a pack, an integer add on the halves, one F2F back')
t('f32_f64_roundtrip', 'float(double({a}) + double({b}))', {'conv': 3, 'f64': 1}, ty='float', exts=['f64'])
t('bitcast', 'uintBitsToFloat(floatBitsToUint({a}) + floatBitsToUint({b}))', {'bitcast': 3, 'iadd': 1},
  ty='float', fit=True)
t('u16_add', '{a} + {b}', {'iadd': 1}, ty='u16', exts=['i16'], note='shaderInt16: no 16-bit ALU on sm_61')
t('u8_mul', '{a} * {b} + uint8_t(3)', {'imul': 1, 'iadd': 1}, ty='u8', exts=['i8'])
# --- 32-bit float -------------------------------------------------------------
t('fadd', '{a} + {b}', {'fadd': 1}, ty='float', fit=True,
  note='the reference: every cost is a ratio to it')
t('fmul', '{a} * {b}', {'fmul': 1}, ty='float', fit=True)
t('ffma', 'fma({a}, {b}, {c})', {'fmul': 1}, ty='float', note='ucas prices Fma as fmul')
t('fmad_contract', '{a} * {b} + {c}', {'fmul': 1, 'fadd': 1}, ty='float',
  note='no NoContraction: the driver may fuse into FFMA')
t('fmad_precise', '{a} * {b} + {c}', {'fmul': 1, 'fadd': 1}, ty='float', qual='precise',
  note='precise: NoContraction, a separate FMUL and FADD')
t('fneg_add', '-{a} + {b}', {'fadd': 2}, ty='float', lat=False, note='a negation is a free source modifier')
t('fabs_add', 'abs({a}) - {b}', {'minmax': 1, 'fadd': 1}, ty='float', note='abs is a free source modifier')
t('fmin_add', 'min({a} + kf, {b})', {'minmax': 1, 'fadd': 1}, ty='float', fit=False)
t('fclamp_add', 'clamp({a} + kf, {b}, {c})', {'minmax': 2, 'fadd': 1}, ty='float')
t('fcmp_select', '({a} < {b}) ? {c} : {d}', {'cmp': 1, 'select': 1}, ty='float', lat=False,
  note='sm_61 FCMP compares and selects in one')
t('fmix', 'mix({a}, {b}, {c})', {'fmul': 1}, ty='float', note='ucas prices FMix as fmul')
t('length3', 'length(vec3({a}, {b}, kf))', {'fmul': 3}, ty='float', note='ucas prices Length as 3 fmul')
t('fdiv', 'kf2 / ({a} + {b})', {'fdiv': 1, 'fadd': 1}, ty='float', fit=True)
t('frcp', '1.0 / ({a} + {b})', {'fdiv': 1, 'fadd': 1}, ty='float')
t('fdiv_const', '({a} + {b}) / 2.0001', {'fdiv': 1, 'fadd': 1}, ty='float')
t('fsqrt', 'sqrt({a})', {'fsqrt': 1}, ty='float', fit=True)
t('frsqrt', 'inversesqrt({a})', {'ftrans': 1}, ty='float')
t('exp2', 'exp2({a})', {'ftrans': 1}, ty='float', fit=True)
t('log2', 'log2({a})', {'ftrans': 1}, ty='float')
t('exp', 'exp({a})', {'ftrans': 1}, ty='float')
t('log', 'log({a})', {'ftrans': 1}, ty='float')
t('sin', 'sin({a})', {'ftrans': 1}, ty='float')
t('cos', 'cos({a})', {'ftrans': 1}, ty='float')
t('tan', 'tan({a})', {'ftrans': 2, 'fdiv': 1}, ty='float', note='ucas: 2 SFU and a divide')
t('asin', 'asin({a} * 0.5)', {'ftrans': 4, 'fmul': 1}, ty='float', note='ucas: a library call, 4 ftrans')
t('acos', 'acos({a} * 0.5)', {'ftrans': 4, 'fmul': 1}, ty='float')
t('atan', 'atan({a})', {'ftrans': 4}, ty='float')
t('atan2', 'atan({a}, {b})', {'ftrans': 4}, ty='float')
t('sinh', 'sinh({a})', {'ftrans': 4}, ty='float')
t('tanh', 'tanh({a})', {'ftrans': 4}, ty='float')
t('pow', 'pow({a}, {b})', {'fpow': 1}, ty='float', fit=True)
# --- 64-bit ---------------------------------------------------------------------
t('dadd', '{a} + {b}', {'f64': 1}, ty='double', exts=['f64'], fit=True, note='fp64 is 1/32 rate on GP108')
t('dmul', '{a} * {b}', {'f64': 1}, ty='double', exts=['f64'])
t('dfma', 'fma({a}, {b}, {c})', {'f64': 1}, ty='double', exts=['f64'])
t('ddiv', 'kd2 / ({a} + {b})', {'x_f64div': 1, 'f64': 1}, ty='double', exts=['f64'], fit=True)
t('dsqrt', 'sqrt({a})', {'x_f64sqrt': 1}, ty='double', exts=['f64'], fit=True)
t('u64_add', '{a} + {b}', {'x_i64add': 1}, ty='u64', exts=['i64'], fit=True)
t('u64_mul', '{a} * {b}', {'x_i64mul': 1}, ty='u64', exts=['i64'], fit=True)
t('u64_shl', '{a} << ({b} & 63ul)', {'x_i64shl': 1, 'x_i64and': 1}, ty='u64', exts=['i64'])
# --- integer dot product (DP4A) ---------------------------------------------------
t('udot4x8', 'udot4x8({a}, {b}, 0) + k', {'x_dp4a': 1, 'iadd': 1}, exts=['dot'], decls=DOT_DECLS, fit=True)
t('sdot4x8', 'uint(sdot4x8({a}, {b}, 0)) + k', {'x_dp4a': 1, 'iadd': 1}, exts=['dot'], decls=DOT_DECLS)
t('udot4x8sat', 'udot4x8sat({a}, {b}, k, 0)', {'x_dp4a': 1}, exts=['dot'], decls=DOT_DECLS)
t('udot4x8_emul',
  'bitfieldExtract({a}, 0, 8) * bitfieldExtract({b}, 0, 8) + bitfieldExtract({a}, 8, 8) * bitfieldExtract({b}, 8, 8)'
  ' + bitfieldExtract({a}, 16, 8) * bitfieldExtract({b}, 16, 8) + ({a} >> 24) * ({b} >> 24) + k',
  {'shift': 8, 'imul': 4, 'iadd': 4}, note='the same dot product without DP4A, as ucas prices it')
# --- subgroup (warp) operations ----------------------------------------------------
SG = dict(exts=['sg'])
t('sg_add', 'subgroupAdd({a}) + {b}', {'x_sg_reduce': 1, 'iadd': 1}, fit=True, **SG)
t('sg_inclusive_add', 'subgroupInclusiveAdd({a}) + {b}', {'x_sg_scan': 1, 'iadd': 1}, fit=True, **SG)
t('sg_clustered_add4', 'subgroupClusteredAdd({a}, 4u) + {b}', {'x_sg_cluster4': 1, 'iadd': 1}, fit=True, **SG)
t('sg_shuffle', 'subgroupShuffle({a}, {b})', {'x_shfl': 1}, fit=True, **SG,
  note='index out of range reads an undefined value; timing is the same')
t('sg_shuffle_xor', 'subgroupShuffleXor({a}, 1u) + {b}', {'x_shfl': 1, 'iadd': 1}, **SG)
t('sg_broadcast', 'subgroupBroadcast({a}, 5u) + {b}', {'x_shfl': 1, 'iadd': 1}, **SG)
t('sg_broadcast_first', 'subgroupBroadcastFirst({a}) + {b}', {'x_shfl': 1, 'iadd': 1}, **SG)
t('sg_quad_swap', 'subgroupQuadSwapHorizontal({a}) + {b}', {'x_shfl': 1, 'iadd': 1}, **SG)
t('sg_ballot', 'subgroupBallot(({a} & 1u) != 0u).x + {b}', {'x_ballot': 1, 'logic': 1, 'cmp': 1, 'iadd': 1},
  fit=True, **SG)
t('sg_any', 'subgroupAny(({a} & 1u) != 0u) ? {b} : {c}', {'x_vote': 1, 'logic': 1, 'cmp': 1, 'select': 1},
  fit=True, lat=False, **SG)
t('sg_rotate', 'subgroupRotate({a}, 3u) + {b}', {'x_shfl': 1, 'iadd': 1}, exts=['sg', 'sgrot'])
t('sg_partition', 'subgroupPartitionNV({a} & 3u).x + {b}', {'x_match': 1, 'logic': 1, 'iadd': 1}, fit=True,
  exts=['sg', 'part'], note='NV: MATCH.ANY')
t('sg_partitioned_add', 'subgroupPartitionedAddNV({a}, pm) + {b}', {'x_part_reduce': 1, 'iadd': 1}, fit=True,
  exts=['sg', 'part'], pre='uvec4 pm = subgroupPartitionNV(s & 3u);', note='NV: a reduction within each partition')
# --- clocks (KHR_shader_clock, NV builtins) -------------------------------------
t('clock_read', 'uint(clockARB()) + {a}', {'x_clock': 1, 'iadd': 1}, exts=['clock'], fit=True, lat=False)
t('clock_realtime_read', 'uint(clockRealtimeEXT()) + {a}', {'x_rtclock': 1, 'iadd': 1}, exts=['rtclock'],
  fit=True, lat=False)

# --- memory ------------------------------------------------------------------------
# One chain (C=1): the warps give the parallelism.  {u} offsets fold into the
# instruction's immediate, so a step is the access and one add.
M = dict(C=1, kind='mem', A=8)
GI = 'uint ib = gid + ((i & 3u) << 20);'          # 16 MiB of 4 MiB uints: DRAM


def mem(name, step, ops, **kw):
    kw2 = dict(M)
    kw2.update(kw)
    t(name, step, ops, **kw2)


mem('load_dram', '{a} + inb.data[ib + {u}u * 65536u]', {'x_load_dram': 1, 'iadd': 1}, pre_iter=GI, fit=True,
    note='coalesced, 16 MiB working set (L2 is 512 KiB): DRAM bound')
mem('load_l2', '{a} + inb.data[ib + {u}u * 4096u]', {'load': 1, 'iadd': 1},
    pre_iter='uint ib = (gid + i * 32u) & 0xfffu;', fit=True,
    note='256 KiB working set, coalesced: the ucas load (an instruction, not a bandwidth)')
mem('load_bcast', '{a} + inb.data[ib + {u}u]', {'x_load_bcast': 1, 'iadd': 1},
    pre_iter='uint ib = (i & 255u) * 16u;', fit=True, note='every lane the same address')
mem('load_ubo', '{a} + ubo.d[ib + {u}u].x', {'x_load_ubo': 1, 'iadd': 1}, pre_iter='uint ib = (i & 31u) * 16u;',
    decls='layout(std140, set = 0, binding = 3) uniform U { uvec4 d[1024]; } ubo;\n', fit=True,
    note='uniform address: the constant cache')
mem('load_ubo_divergent', '{a} + ubo.d[ib + {u}u].x', {'x_load_ubo_div': 1, 'iadd': 1},
    pre_iter='uint ib = (gid + i) & 511u;',
    decls='layout(std140, set = 0, binding = 3) uniform U { uvec4 d[1024]; } ubo;\n', fit=True,
    note='a different address in each lane: the constant cache serialises')
SH_PRE = ('uint lid = gl_LocalInvocationID.x;\n'
          '    for (uint q = lid; q < 4096u + 64u; q += 128u) sh[q] = s + q;\n'
          '    barrier();')
mem('load_shared', '{a} + sh[ib + {u}u * 128u]', {'x_load_shared': 1, 'iadd': 1},
    pre_iter='uint ib = (lid + i) & 127u;', decls=SHARED, pre=SH_PRE, fit=True)
mem('load_shared_conflict', '{a} + sh[ib + {u}u]', {'x_load_shared_conflict': 1, 'iadd': 1},
    pre_iter='uint ib = (lid * 32u + i) & 4095u;', decls=SHARED, pre=SH_PRE, fit=True,
    note='32 lanes on one bank: 32-way conflict')
mem('store_dram', '{a} + k', {'x_store_dram': 1, 'iadd': 1}, pre_iter='uint ob = gid + ((i & 3u) << 20);',
    post='outb.data[ob + {u}u * 65536u] = {a};', fit=True, note='coalesced, 16 MiB: DRAM bound')
mem('store_l2', '{a} + k', {'store': 1, 'iadd': 1}, pre_iter='uint ob = (gid + i * 32u) & 0xfffu;',
    post='outb.data[ob + {u}u * 4096u] = {a};', fit=True, note='coalesced, 256 KiB: the ucas store')
mem('store_shared', '{a} + k', {'x_store_shared': 1, 'iadd': 1}, pre_iter='uint ib = (lid + i) & 127u;',
    post='sh[ib + {u}u * 128u] = {a};', decls=SHARED, pre=SH_PRE, fit=True)
mem('atomic_global_dram', '{a} + atomicAdd(outb.data[ob + {u}u * 65536u], {a})', {'x_atomic_global_dram': 1, 'iadd': 1},
    pre_iter='uint ob = gid + ((i & 3u) << 20);', fit=True, note='distinct addresses over 16 MiB, value returned')
mem('atomic_global', '{a} + atomicAdd(outb.data[ob + {u}u * 4096u], {a})', {'atomic_global': 1, 'iadd': 1},
    pre_iter='uint ob = (gid + i * 32u) & 0xfffu;', fit=True,
    note='distinct addresses over 256 KiB (L2 atomics), value returned: the ucas atomic_global')
mem('atomic_global_red', '{a} + k', {'x_red_global': 1, 'iadd': 1}, pre_iter='uint ob = gid + ((i & 3u) << 20);',
    post='atomicAdd(outb.data[ob + {u}u * 65536u], {a});', fit=True, note='value unused: RED')
mem('atomic_global_same', '{a} + atomicAdd(outb.data[{u}u], 1u)', {'x_atomic_global_same': 1, 'iadd': 1},
    fit=True, note='every lane of every warp on one address per step')
mem('atomic_global_cas', 'atomicCompSwap(outb.data[ob + {u}u * 65536u], {a}, {a} + k)', {'x_cas_global': 1, 'iadd': 1},
    pre_iter='uint ob = gid + ((i & 3u) << 20);', fit=True)
mem('atomic_global_f32', '{a} + atomicAdd(of.f[ob + {u}u * 65536u], {a})', {'x_atomic_global_f32': 1, 'fadd': 1},
    ty='float', exts=['afloat'], decls='layout(std430, set = 0, binding = 2) buffer OF { float f[]; } of;\n',
    pre_iter='uint ob = gid + ((i & 3u) << 20);', fit=True, vkext=['VK_EXT_shader_atomic_float'])
mem('atomic_global_u64', '{a} + atomicAdd(oq.q[ob + {u}u * 32768u], {a})', {'x_atomic_global_u64': 1, 'x_i64add': 1},
    ty='u64', exts=['i64', 'a64'], decls='layout(std430, set = 0, binding = 4) buffer OQ { uint64_t q[]; } oq;\n',
    pre_iter='uint ob = gid + ((i & 3u) << 19);', fit=True)
mem('atomic_shared', '{a} + atomicAdd(sh[ib + {u}u * 128u], {a})', {'atomic_shared': 1, 'iadd': 1},
    pre_iter='uint ib = (lid + i) & 127u;', decls=SHARED, pre=SH_PRE, fit=True, note='distinct addresses')
mem('atomic_shared_same', '{a} + atomicAdd(sh[{u}u], {a})', {'x_atomic_shared_same': 1, 'iadd': 1},
    decls=SHARED, pre=SH_PRE, fit=True, note='every lane on one address')
mem('atomic_shared_f32', '{a} + atomicAdd(shf[ib + {u}u * 128u], {a})', {'x_atomic_shared_f32': 1, 'fadd': 1},
    ty='float', exts=['afloat'], decls=SHARED_F, pre_iter='uint ib = (lid + i) & 127u;',
    pre='uint lid = gl_LocalInvocationID.x;', fit=True, vkext=['VK_EXT_shader_atomic_float'],
    note='no native shared float add on sm_61: a CAS loop')
mem('barrier', '{a} + sh[(lid ^ 1u) + {u}u * 128u]', {'barrier': 2, 'x_store_shared': 1, 'x_load_shared': 1, 'iadd': 1},
    decls=SHARED, pre=SH_PRE, fit=True, prestep='sh[lid + {u}u * 128u] = {a}; barrier();', poststep='barrier();',
    note='a store, a barrier, a load from the neighbour, a barrier (128 invocations)')
mem('barrier_wg1024', '{a} + sh[(lid ^ 1u) + {u}u * 4u]', {'x_barrier1024': 2, 'x_store_shared': 1, 'x_load_shared': 1, 'iadd': 1},
    decls=SHARED, wg=1024, pre='uint lid = gl_LocalInvocationID.x;', fit=True,
    prestep='sh[lid + {u}u * 4u] = {a}; barrier();', poststep='barrier();',
    note='the same with 1024 invocations: 32 warps meet')
mem('subgroup_barrier', '{a} + sh[(lid ^ 1u) + {u}u * 128u]', {'x_sg_barrier': 2, 'x_store_shared': 1, 'x_load_shared': 1, 'iadd': 1},
    decls=SHARED, pre=SH_PRE, fit=True, exts=['sg'], prestep='sh[lid + {u}u * 128u] = {a}; subgroupMemoryBarrierShared(); subgroupBarrier();',
    poststep='subgroupBarrier();', note='the neighbour is in the same warp')
TEX = 'layout(set = 0, binding = 5) uniform sampler2D tex;\n'
mem('tex_fetch', '{a} + texelFetch(tex, ivec2(px, py + {u}), 0).x', {'x_tex_fetch': 1, 'fadd': 1}, ty='float',
    decls=TEX, pre_iter='int px = int(gid & 1023u), py = int((gid >> 10) + i * 16u) & 1007;', fit=True,
    args=['--bind', '0.5=format=rgba8_unorm,size=1024x1024,pattern=noise,filter=nearest'],
    note='rgba8 1024x1024, nearest, one texel a lane in a row')
mem('tex_bilinear', '{a} + textureLodOffset(tex, uv, 0.0, ivec2({u} & 7, {u} >> 3)).x', {'x_tex_bilinear': 1, 'fadd': 1},
    ty='float', decls=TEX, pre_iter='vec2 uv = vec2(float(gid & 1023u) + 0.5, float((gid >> 10) + i * 16u) + 0.5) / 1024.0;',
    fit=True, args=['--bind', '0.5=format=rgba8_unorm,size=1024x1024,pattern=noise,filter=linear'],
    note='rgba8, bilinear, explicit LOD')
mem('tex_bilinear_rgba32f', '{a} + textureLodOffset(tex, uv, 0.0, ivec2({u} & 7, {u} >> 3)).x', {'x_tex_bilinear_rgba32f': 1, 'fadd': 1},
    ty='float', decls=TEX, pre_iter='vec2 uv = vec2(float(gid & 1023u) + 0.5, float((gid >> 10) + i * 16u) + 0.5) / 1024.0;',
    fit=True, args=['--bind', '0.5=format=rgba32_sfloat,size=1024x1024,pattern=noise,filter=linear'],
    note='rgba32f, bilinear: four times the texel bytes')
mem('image_load', '{a} + imageLoad(img, ivec2(px, py + {u})).x', {'x_image_load': 1, 'iadd': 1},
    decls='layout(set = 0, binding = 6, r32ui) uniform readonly uimage2D img;\n',
    pre_iter='int px = int(gid & 1023u), py = int((gid >> 10) + i * 16u) & 1007;', fit=True,
    args=['--bind', '0.6=format=r32_uint,size=1024x1024,pattern=noise'])

# --- branches ------------------------------------------------------------------------
# One float chain.  Each arm adds kf M times (cannot fold); [[dont_flatten]] asks for a
# real branch.  The condition is a bit of a per-iteration hash: uniform when it does not
# depend on the lane, divergent when it does.
def arms(m, x='kf'):
    # fma(a, a, k): one FFMA, nonlinear, so the driver cannot sum the arm up
    return ' '.join(['{a} = fma({a}, {a}, %s);' % x] * m)


B = dict(C=1, kind='branch', A=8, ty='float', exts=['cf'], lat=False)
# condition: one bit of a per-iteration word; bi is the same in every lane, bd = bi ^ a
# per-lane hash (bit u of gid * K differs between lanes for every u), so both tests
# have the same instructions and only the divergence differs.
for mm in (2, 8, 32):
    t('branch_uniform_m%d' % mm, None, {'branch': 1, 'fmul': mm, 'logic': 1, 'cmp': 1}, fit=(mm == 32), **B,
      body='[[dont_flatten]] if ((bi & (1u << {u}u)) != 0u) {{ %s }} else {{ %s }}' % (arms(mm), arms(mm, 'kf2')),
      note='the warp takes one arm of %d FFMAs' % mm)
    t('branch_divergent_m%d' % mm, None, {'branch': 1, 'diverge': 1, 'fmul': 2 * mm, 'logic': 1, 'cmp': 1},
      fit=(mm == 32), **B,
      body='[[dont_flatten]] if ((bd & (1u << {u}u)) != 0u) {{ %s }} else {{ %s }}' % (arms(mm), arms(mm, 'kf2')),
      note='lanes split: the warp runs both arms of %d FFMAs (keep-up)' % mm)
for mm in (2, 8):
    t('branch_flatten_m%d' % mm, None, {'select': 1, 'fmul': 2 * mm, 'logic': 1, 'cmp': 1}, **B,
      body='[[flatten]] if ((bd & (1u << {u}u)) != 0u) {{ %s }} else {{ %s }}' % (arms(mm), arms(mm, 'kf2')),
      note='the divergent branch flattened: both arms and a select')
t('branch_expect_m8', None, {'branch': 1, 'fmul': 8, 'logic': 1, 'cmp': 1},
  **dict(B, exts=['cf', 'expect']),
  body='[[dont_flatten]] if (expectEXT((bi & (1u << {u}u)) != 0u, false)) {{ %s }} else {{ %s }}' % (arms(8), arms(8, 'kf2')),
  note='KHR_shader_expect_assume hint on a uniform branch')

for x in T:
    x.setdefault('pre_iter', '')
    x.setdefault('body', None)
    x.setdefault('prestep', '')
    x.setdefault('poststep', '')
for x in T:
    if x['kind'] == 'branch':
        x['pre_iter'] = 'uint bi = i * 0x9e3779b9u + k, bd = bi ^ (gid * 0x9e3779b9u);'

# --- fragment stage -----------------------------------------------------------------
FRAG = []


def f(name, step, ops, fit=False, exts=(), decls='', pre='', note='', args=(), A=8, C=4):
    FRAG.append(dict(name=name, step=step, ops=ops, ty='float', fit=fit, exts=list(exts), decls=decls,
                     pre=pre, note=note, args=list(args), kind='frag', A=A, C=C, lat=False, ring=False,
                     vkext=[], qual='', iters=None))


f('frag_fadd', '{a} + {b}', {'fadd': 1}, note='the fragment reference')
f('frag_dfdx', 'dFdx({a}) + {a}', {'x_deriv': 1, 'fadd': 1}, fit=True, note='a quad shuffle and a subtract')
f('frag_dfdy_fine', 'dFdyFine({a}) + {a}', {'x_deriv': 1, 'fadd': 1})
f('frag_fwidth', 'fwidth({a}) + {a}', {'x_deriv': 2, 'minmax': 2, 'fadd': 2})
f('frag_tex_implicit', '{a} + texture(tex, uv + vec2({a} * 1e-3, 0.0)).x', {'x_tex_implicit': 1, 'fmul': 1, 'fadd': 2},
  fit=True, decls='layout(set = 0, binding = 5) uniform sampler2D tex;\n', C=1,
  args=['--bind', '0.5=format=rgba8_unorm,size=1024x1024,pattern=noise,filter=linear,mips=full'],
  note='implicit LOD: the derivatives come from the quad')

# --- probes (not timed per op): occupancy and SM placement --------------------------
PROBE = r'''#version 460
#extension GL_NV_shader_sm_builtins : require
#extension GL_ARB_shader_clock : require
#extension GL_EXT_shader_realtime_clock : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// One record of 8 uints a warp: SM id, warp slot, SM count, warps per SM, workgroup,
// realtime clock at start (lo, hi), SM clocks spent.  run.py reads it back (--dump).
layout(local_size_x = 128) in;
layout(push_constant) uniform PC { uint iters; uint seed; uint k; uint k2; float kf; float kf2; } pc;
layout(std430, set = 0, binding = 1) buffer Out { uint data[]; } outb;
void main() {
    uint64_t t0 = clockRealtimeEXT();
    uint64_t c0 = clockARB();
    float a = float(gl_GlobalInvocationID.x);
    for (uint i = 0; i < pc.iters; i++)
        a = a * pc.kf + pc.kf2;
    uint64_t c1 = clockARB();
    if (subgroupElect()) {
        uint w = gl_WorkGroupID.x * (gl_WorkGroupSize.x / gl_SubgroupSize) + gl_SubgroupID;
        uint o = w * 8u;
        outb.data[o + 0u] = gl_SMIDNV;
        outb.data[o + 1u] = gl_WarpIDNV;
        outb.data[o + 2u] = gl_SMCountNV;
        outb.data[o + 3u] = gl_WarpsPerSMNV;
        outb.data[o + 4u] = gl_WorkGroupID.x;
        outb.data[o + 5u] = uint(t0);
        outb.data[o + 6u] = uint(t0 >> 32);
        outb.data[o + 7u] = uint(c1 - c0) + (floatBitsToUint(a) & 0u);
    }
}
'''


# ---------------------------------------------------------------------------
def fill(s, **kw):
    for k, v in kw.items():
        s = s.replace('{' + k + '}', v)
    return s


def step_text(x, j, u, C, chain):
    def v(n):
        return chain(n % C)
    rep = dict(a=v(j), b=v(j + 1), c=v(j + 2), d=v(j + 3), u=str(u), u4=str(u % 4), j=str(j))
    out = []
    if x.get('prestep'):
        out.append(fill(x['prestep'], **rep))
    if x.get('body'):
        out.append(fill(x['body'], **rep))
    else:
        out.append('%s = %s;' % (v(j), fill(x['step'], **rep)))
    if x.get('post'):
        out.append(fill(x['post'], **rep))
    if x.get('poststep'):
        out.append(fill(x['poststep'], **rep))
    return ' '.join(out)


HEADER_PC = ('layout(push_constant) uniform PC { uint iters; uint seed; uint k; uint k2; float kf; float kf2; } pc;\n')


def extlines(x, extra=()):
    names = []
    for tag in list(x['exts']) + list(extra):
        for e in EXT[tag]:
            if e not in names:
                names.append(e)
    if x['ty'] in ('u64',) and 'GL_EXT_shader_explicit_arithmetic_types_int64' not in names:
        names.append('GL_EXT_shader_explicit_arithmetic_types_int64')
    if x['ty'] == 'u16' and EXT['i16'][0] not in names:
        names += EXT['i16']
    if x['ty'] == 'u8' and EXT['i8'][0] not in names:
        names += EXT['i8']
    return ''.join('#extension %s : require\n' % e for e in names)


def gen_compute(x, U, latency):
    ty, init, sink = TYPES[x['ty']]
    C = x['latC'] if latency else x['C']
    wg = 32 if latency else x['wg']
    extra = ['clock'] if latency else []
    if x['ty'] == 'u64' or 'clock' in extra:
        extra.append('i64')
    lines = ['#version 460\n', extlines(x, extra)]
    lines.append('// generated by gen.py: test %s, %s, UNROLL %d, %d chain(s)\n'
                 % (x['name'], 'latency (one warp)' if latency else 'throughput', U, C))
    lines.append('// one step: %s\n' % json.dumps(x['ops']))
    if x.get('note'):
        lines.append('// %s\n' % x['note'])
    lines.append('layout(local_size_x = %d) in;\n' % wg)
    lines.append(HEADER_PC)
    lines.append('layout(std430, set = 0, binding = 0) readonly buffer In { uint data[]; } inb;\n')
    lines.append('layout(std430, set = 0, binding = 1) buffer Out { uint data[]; } outb;\n')
    if x['decls']:
        lines.append(x['decls'])
    lines.append('void main() {\n')
    lines.append('    uint gid = gl_GlobalInvocationID.x;\n')
    lines.append('    uint s = inb.data[gid & 0xfffffu] ^ pc.seed;\n')
    lines.append('    uint k = pc.k, k2 = pc.k2; int ki = int(pc.k); float kf = pc.kf, kf2 = pc.kf2;\n')
    if x['ty'] == 'double' or 'f64' in x['exts']:
        lines.append('    double kd = double(pc.kf), kd2 = double(pc.kf2);\n')
    if x['pre']:
        lines.append('    %s\n' % x['pre'])
    q = (x['qual'] + ' ') if x['qual'] else ''
    chain = lambda n: 'v%d' % n
    for n in range(C):
        lines.append('    %s%s v%d = %s;\n' % (q, ty, n, fill(init, i=str(n))))
    if latency:
        lines.append('    uint64_t c0 = clockARB();\n')
    lines.append('    for (uint i = 0u; i < pc.iters; i++) {\n')
    if x['pre_iter']:
        lines.append('        %s\n' % x['pre_iter'])
    for u in range(U):
        for j in range(C):
            lines.append('        %s\n' % step_text(x, j, u, C, chain))
    lines.append('    }\n')
    if latency:
        lines.append('    uint64_t c1 = clockARB();\n')
    r = ' ^ '.join('(%s)' % fill(sink, x=chain(n)) for n in range(C))
    if latency:
        lines.append('    outb.data[2u * gid] = %s;\n' % r)
        lines.append('    outb.data[2u * gid + 1u] = uint(c1 - c0);\n')
    else:
        # one word a lane at 4 MiB+: away from the words the memory tests write
        lines.append('    outb.data[0x3f0000u + (gid & 0xffffu)] = %s;\n' % r)
    lines.append('}\n')
    return ''.join(lines)


def gen_frag(x, U):
    C = x['C']
    lines = ['#version 460\n', extlines(x)]
    lines.append('// generated by gen.py: test %s, fragment, UNROLL %d, %d chain(s)\n' % (x['name'], U, C))
    lines.append('// one step: %s\n' % json.dumps(x['ops']))
    if x.get('note'):
        lines.append('// %s\n' % x['note'])
    lines.append(HEADER_PC)
    if x['decls']:
        lines.append(x['decls'])
    lines.append('layout(location = 0) in vec4 vin;\nlayout(location = 0) out vec4 col;\n')
    lines.append('void main() {\n    vec2 uv = vin.xy;\n    float kf = pc.kf, kf2 = pc.kf2;\n')
    for n in range(C):
        lines.append('    float v%d = uv.x * %d.0 + uv.y + 1.0;\n' % (n, n + 1))
    lines.append('    for (uint i = 0u; i < pc.iters; i++) {\n')
    for u in range(U):
        for j in range(C):
            lines.append('        %s\n' % step_text(x, j, u, C, lambda n: 'v%d' % n))
    lines.append('    }\n')
    lines.append('    col = vec4(%s, 0.0, 0.0, 1.0);\n' % ' + '.join('v%d' % n for n in range(C)))
    lines.append('}\n')
    return ''.join(lines)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(' '.join(cmd) + '\n' + p.stdout + p.stderr)
        raise SystemExit(1)


def model_guess(ops):
    g = dict(iadd=1, logic=1, shift=2, imul=3, idiv=24, idivk=5, cmp=2, select=1, minmax=2, bitcnt=4,
             conv=4, bitcast=0, fadd=1, fmul=1, fdiv=10, fsqrt=8, ftrans=4, fpow=10, f64=32, load=8,
             branch=2, diverge=6, barrier=20, atomic_shared=8, atomic_global=64, store=8)
    return max(1.0, sum(g.get(k, 8) * n for k, n in ops.items()))


def main():
    build = '--no-build' not in sys.argv
    os.makedirs(os.path.join(HERE, 'src'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'spv'), exist_ok=True)
    man = dict(device='NVIDIA GeForce GT 1030 (GP108, sm_61)', tests=[])
    files = []
    for x in T + FRAG:
        A = x['A']
        cost = model_guess(x['ops'])
        C = x['C']
        # about 1 ms of GPU time for the A build at the model's guess
        lanes = 65536 if x['kind'] != 'frag' else 1920 * 1080
        iters = x['iters'] or max(2, int(round(5.8e8 / (lanes * A * C * cost))))
        ent = dict(name=x['name'], kind=x['kind'], ops=x['ops'], fit=x['fit'], ty=x['ty'], chains=C,
                   unroll=[A, 2 * A], iters=iters, note=x.get('note', ''), vkext=x['vkext'],
                   args=x['args'], files={})
        builds = [('a', A, False), ('b', 2 * A, False)]
        if x['lat'] and x['kind'] == 'alu':
            builds += [('la', 32, True), ('lb', 64, True)]
            ent['latency'] = dict(chains=x['latC'], unroll=[32, 64], iters=64)
        for tag, U, lat in builds:
            if x['kind'] == 'frag':
                src, stage = gen_frag(x, U), 'frag'
            else:
                src, stage = gen_compute(x, U, lat), 'comp'
            base = '%s.%s' % (x['name'], tag)
            sp = os.path.join('src', base + '.' + stage)
            with open(os.path.join(HERE, sp), 'w') as fh:
                fh.write(src)
            ent['files'][tag] = 'spv/%s.spv' % base
            files.append((sp, 'spv/%s.spv' % base))
        man['tests'].append(ent)
    probe = os.path.join('src', 'probe_sm.comp')
    with open(os.path.join(HERE, probe), 'w') as fh:
        fh.write(PROBE)
    files.append((probe, 'spv/probe_sm.spv'))
    man['probe'] = 'spv/probe_sm.spv'
    with open(os.path.join(HERE, 'manifest.json'), 'w') as fh:
        json.dump(man, fh, indent=1)
        fh.write('\n')
    if not build:
        return
    for sp, out in files:
        run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '--quiet', '-o',
             os.path.join(HERE, out), os.path.join(HERE, sp)])
        run(['spirv-val', '--target-env', 'vulkan1.3', os.path.join(HERE, out)])
    print('%d tests, %d SPIR-V modules' % (len(man['tests']), len(files)))


if __name__ == '__main__':
    main()
