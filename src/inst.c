/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * inst.c: a catalogue of single instructions that targets have and the
 * base operations do not name.
 *
 * Each one is an operation of the language whose value is defined by an
 * expression over the base operations (inst_expand): that expression is
 * its exact semantics, and it is how the operation is evaluated, bounded
 * and bit-blasted (sg_lower spells it out, as it does the subgroup
 * operations).  The node keeps the operation for printing and for the
 * price: one instruction of its cost class.  Like the dot forms, a model
 * that does not list the class cannot use the operation, and no built-in
 * model lists one yet: none is measured.
 *
 *   bfe(x, p, n)       bits p..p+n-1 of x, zero-extended (u32); n = 0 is
 *                      0; undefined when p + n > 32: OpBitFieldUExtract
 *   sbfe(x, p, n)      the same field sign-extended (i32):
 *                      OpBitFieldSExtract
 *   brev(x)            bit i is bit 31 - i of x: OpBitReverse
 *   mulhi(a, b)        the high 32 bits of the 64-bit product of two
 *                      u32; smulhi(a, b) of two i32: Op[US]MulExtended
 *   funl(hi, lo, k)    the high word of (hi:lo) << k, k in 0..31:
 *                      (hi << k) | (lo >> (32 - k)); funl(x, x, k) is x
 *                      rotated left by k
 *   funr(hi, lo, k)    the low word of (hi:lo) >> k, k in 0..31
 *   med3(a, b, c)      the median of three integers: max(min(a, b),
 *                      min(max(a, b), c)); with lo <= hi, med3(x, lo, hi)
 *                      is x clamped to lo..hi
 *   usad4x8(a, b, c)   c plus the sum of |a.byte[i] - b.byte[i]| over the
 *                      four bytes of two u32 (u32, wrapping)
 *   byteperm(a, b, s)  each byte of the result is one of the eight bytes
 *                      of b:a (byte 0 the low byte of a, byte 4 the low
 *                      byte of b): result byte i is chosen by nibble i of
 *                      s, whose low three bits are the byte's number and
 *                      whose top bit, when set, replaces the byte by
 *                      eight copies of its sign bit
 *   bfi(base, ins, p, n)
 *                      base with bits p..p+n-1 replaced by the low n bits
 *                      of ins (u32), p and n constant (a node has three
 *                      operands, so they ride in one, bfi_field):
 *                      OpBitFieldInsert
 *   mulwide(a, b)      the 64-bit product of two u32 (u64)
 *   madwide(a, b, c)   that product plus the u64 c (wrapping): IMAD.WIDE,
 *                      v_mad_u64_u32
 *   bytesel(hi, lo, s) each byte of the result chosen by a byte of s from
 *                      the eight bytes of hi:lo (byte 0 the low byte of
 *                      lo, byte 4 the low byte of hi): selector 0..7 that
 *                      byte, 8..11 eight copies of the sign bit of byte
 *                      1, 3, 5 or 7, 12 the byte 0, 13 and up 255:
 *                      v_perm_b32 (hi its S0, lo its S1)
 *   add4x8, sub4x8, absdiff4x8, min4x8, max4x8 (a, b)
 *                      the four bytes of two u32 lane by lane, unsigned:
 *                      the sum and the difference mod 256, |a - b|, the
 *                      smaller, the larger (the video SIMD forms with no
 *                      merge or saturation: vadd4, vsub4, vabsdiff4,
 *                      vmin4, vmax4)
 *   msad4x8(a, b, c)   c plus |a.byte[i] - b.byte[i]| over the bytes
 *                      where a's byte is not 0 (u32, wrapping):
 *                      v_msad_u8, DXIL's MSAD
 *   add2x16, sub2x16, mul2x16, min2x16, max2x16, smin2x16, smax2x16
 *   (a, b)             the two 16-bit halves lane by lane: the sum, the
 *                      difference and the low half of the product mod
 *                      2^16, the smaller and the larger unsigned or
 *                      signed: v_pk_add_u16, v_pk_sub_u16,
 *                      v_pk_mul_lo_u16, v_pk_min/max_u16/i16 (no clamp)
 *   lerp4x8(a, b, c)   byte i is (a.byte[i] + b.byte[i] + bit 0 of
 *                      c.byte[i]) >> 1, the average rounded up where
 *                      that bit is set: v_lerp_u8
 *   qsad4x8(s, r, acc) four sums of byte differences against the bytes
 *                      of r (u32), sum k over the four bytes of s (u64)
 *                      from byte k, plus the 16-bit lane k of acc
 *                      (u64), each cut to 16 bits and packed (u64):
 *                      v_qsad_pk_u16_u8; mqsad4x8 the same leaving out
 *                      the bytes where r's is 0: v_mqsad_pk_u16_u8
 *   min3, max3 (a, b, c)
 *                      the least and the greatest of three integers;
 *                      min_max(a, b, c) is max(min(a, b), c),
 *                      max_min(a, b, c) min(max(a, b), c), min_add(a,
 *                      b, c) min(a, b) + c and max_add max(a, b) + c
 *                      (wrapping), in the operands' common type:
 *                      v_min3, v_max3, v_minmax, v_maxmin; VMNMX and
 *                      PTX vmin, vmax with a .min, .max or .add
 *                      secondary operation
 *   sad32(a, b, c)     |a - b| + c, u32 (wrapping): v_sad_u32; PTX
 *                      vabsdiff.add
 *   addsat, subsat (a, b)
 *                      a + b and a - b clamped to the range of u32, or
 *                      of i32 when the operands are signed (narrower
 *                      ones widened first): PTX vadd.sat, vsub.sat;
 *                      v_add_u32 and v_add_nc_i32 with clamp
 *
 * Of floats (f16, f32, f64), defined on their bits as LLVM's AMDGPU
 * constant folder reads the instructions, NaNs and signed zeros included:
 *
 *   med3(a, b, c)      of three floats: the median in the order where -0
 *                      is below +0; when one is NaN, the least of the
 *                      others (v_min3), and when all three are, c:
 *                      v_med3_f32, v_med3_f16
 *   ldexp(x, n)        x times 2^n rounded once (to nearest even,
 *                      subnormals kept), n any integer: v_ldexp_f16/32/64
 *   frexp_mant(x)      the significand in [0.5, 1) with x's sign; 0, an
 *                      infinity and a NaN are x: v_frexp_mant
 *   frexp_exp(x)       the e with x = frexp_mant(x) * 2^e, 0 for 0, an
 *                      infinity and a NaN (i32; i16 for an f16):
 *                      v_frexp_exp_i32_f32, _i32_f64, _i16_f16
 *   cubeid, cubesc, cubetc, cubema (x, y, z)
 *                      of three f32, the cube map face of the major
 *                      axis (z, then y, then x on ties of magnitude; a
 *                      negative one is the odd face, -0 and NaN are not
 *                      negative) as 0..5, its s and t before the
 *                      division, and twice the major axis: v_cubeid_f32,
 *                      v_cubesc_f32, v_cubetc_f32, v_cubema_f32
 *   trig_preop(x, s)   53 bits of 2/pi from bit 53 * (s mod 32), moved
 *                      further by x's exponent past 1077, scaled by
 *                      2^(-53 - shift) (2^128 more when the exponent is
 *                      1968 or more); 0 past the 1184 bits the folder
 *                      holds: v_trig_preop_f64 (x is an f64; its sign
 *                      and significand are not read)
 *
 * The channels of a packed texel format, from its 32-bit word w as a
 * texel fetch decodes them (c a constant channel, r g b a as 0 1 2 3):
 *
 *   unpack_b10g11r11(w, c)
 *                      r (bits 0..10) and g (11..21) are unsigned 11-bit
 *                      floats, b (22..31) an unsigned 10-bit float: 5
 *                      exponent bits e over 6 or 5 significand bits m,
 *                      2^(e-15) (1 + m/2^k), 2^-14 m/2^k when e is 0,
 *                      Inf (m 0) or NaN when e is 31 (the Khronos Data
 *                      Format 10.2, 10.3): B10G11R11_UFLOAT_PACK32
 *   unpack_e5b9g9r9(w, c)
 *                      r, g, b the 9-bit significands at bits 0, 9 and
 *                      18 times 2^(e - 24), e the 5 bits from 27, no
 *                      implicit one (EXT_texture_shared_exponent, Mesa's
 *                      rgb9e5_to_float3): E5B9G9R9_UFLOAT_PACK32
 *   unpack_a2b10g10r10(w, c)
 *                      the 10-bit r, g, b at bits 0, 10 and 20 over
 *                      1023 and the 2-bit a at 30 over 3, each rounded
 *                      once to f32: A2B10G10R10_UNORM_PACK32
 *
 * They are priced by tex_packed: what the unit does when the word is a
 * texel (texalu.c offers them for code that decodes a word by hand).
 *
 * find (inst_find) offers them for the patterns they are, each kept only
 * when the active model prices the whole function lower, and checked on
 * the function's domain.
 */
#include "calc.h"
#include "fuse.h"
#include "sg.h"

#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#define U32 TY(K_UINT, 32)
#define U64 TY(K_UINT, 64)
#define I32 TY(K_SINT, 32)
#define I64 TY(K_SINT, 64)

int
inst_class(int op)
{
	switch (op) {
	case O_BFE: case O_SBFE:
		return C_BFE;
	case O_BREV:
		return C_BREV;
	case O_MULHI: case O_SMULHI:
		return C_MULHI;
	case O_FUNL: case O_FUNR:
		return C_FUNNEL;
	case O_MED3:
		return C_MED3;
	case O_USAD4X8:
		return C_SAD4X8;
	case O_BFI:
		return C_BFI;
	case O_MULWIDE: case O_MADWIDE:
		return C_MULWIDE;
	case O_BYTESEL:
		return C_BYTESEL;
	case O_ADD4X8: case O_SUB4X8: case O_ABSDIFF4X8: case O_MIN4X8:
	case O_MAX4X8:
		return C_SIMD4X8;
	case O_MSAD4X8:
		return C_MSAD;
	case O_ADD2X16: case O_SUB2X16: case O_MUL2X16: case O_MIN2X16:
	case O_MAX2X16: case O_SMIN2X16: case O_SMAX2X16:
		return C_PK16;
	case O_LERP4X8:
		return C_LERP8;
	case O_QSAD4X8: case O_MQSAD4X8:
		return C_QSAD;
	case O_MIN3: case O_MAX3: case O_MINMAX: case O_MAXMIN: case O_MINADD:
	case O_MAXADD:
		return C_VMNMX;
	case O_SAD:
		return C_SAD32;
	case O_ADDSAT: case O_SUBSAT:
		return C_ADDSAT;
	case O_LDEXP:
		return C_LDEXP;
	case O_FREXP_MANT: case O_FREXP_EXP:
		return C_FREXP;
	case O_CUBEID: case O_CUBESC: case O_CUBETC: case O_CUBEMA:
		return C_CUBE;
	case O_TRIG_PREOP:
		return C_TRIGPREOP;
	case O_UNPACK_B10G11R11: case O_UNPACK_E5B9G9R9:
	case O_UNPACK_A2B10G10R10:
		return C_TEX_PACKED;
	}
	return C_BYTEPERM;
}

static nid
ku(ty_t t, uint64_t v)
{
	return mk_const_i(t, (int64_t)v);
}

nid
bfi_field(int p, int n)
{
	return mk_const_i(TY_LIT, (int64_t)p | (int64_t)n << 6);
}

/* the mask of bits p..p+n-1 */
static uint32_t
field_mask(int p, int n)
{
	return n == 0 ? 0 : (uint32_t)(0xffffffffu >> (32 - n)) << p;
}

static nid
sum_type(nid a, nid b, nid c)
{
	nid s = mk_bin(O_ADD, mk_bin(O_ADD, a, b), c);

	return s;
}

static ty_t
inst_type(int op, nid a, nid b, nid c)
{
	switch (op) {
	case O_SBFE: case O_SMULHI:
		return I32;
	case O_MED3: case O_MIN3: case O_MAX3: case O_MINMAX: case O_MAXMIN:
	case O_MINADD: case O_MAXADD:
		/* the operands' common type, as their sum would have */
		return N.ty[sum_type(a, b, c)];
	case O_MULWIDE: case O_MADWIDE: case O_QSAD4X8: case O_MQSAD4X8:
		return U64;
	case O_ADDSAT: case O_SUBSAT:
		/* u32 or i32: mk_inst has widened the operands */
		return N.ty[a] == I32 ? I32 : U32;
	case O_LDEXP: case O_FREXP_MANT:
		return N.ty[a];
	case O_FREXP_EXP:
		return TYW(N.ty[a]) == 16 ? TY(K_SINT, 16) : I32;
	case O_CUBEID: case O_CUBESC: case O_CUBETC: case O_CUBEMA:
		return TY_F32;
	case O_TRIG_PREOP:
		return TY_F64;
	case O_UNPACK_B10G11R11: case O_UNPACK_E5B9G9R9:
	case O_UNPACK_A2B10G10R10:
		return TY_F32;
	}
	return U32;
}

static int
float_inst(int op)
{
	return op >= O_LDEXP && op <= O_TRIG_PREOP;
}

/* ---- the float instructions, by their operands' bits ---- */

static nid fof(ty_t t, nid u);

/* the f32 2^k from its bits, k an exponent of a normal f32 (any u32
 * node) */
static nid
pow2_of(nid k)
{
	return fof(TY_F32, mk_bin(O_SHL, mk_bin(O_ADD, k, ku(U32, 127)),
	    ku(U32, 23)));
}

/* the unsigned float of 5 exponent bits over mb significand bits at
 * bit p of w (Data Format 10.2, 10.3): every value is an f32 */
static nid
ufloat_of(nid w, int p, int mb)
{
	nid f = mk_bin(O_AND, mk_bin(O_SHR, w, ku(U32, p)), ku(U32,
	    ((uint64_t)1 << (mb + 5)) - 1));
	nid e = mk_bin(O_SHR, f, ku(U32, mb));
	nid m = mk_bin(O_AND, f, ku(U32, ((uint64_t)1 << mb) - 1));
	nid den = mk_bin(O_MUL, mk_cast(TY_F32, m), pow2_of(ku(U32,
	    (uint64_t)(-14 - mb) & 0xffffffff)));
	/* 2^(e-15) (1 + m/2^mb) = (2^mb + m) 2^(e - 15 - mb), e in 1..30 */
	nid nrm = mk_bin(O_MUL, mk_cast(TY_F32, mk_bin(O_ADD, m, ku(U32,
	    (uint64_t)1 << mb))), pow2_of(mk_bin(O_SUB, e, ku(U32,
	    (uint64_t)(15 + mb)))));
	nid big = mk_sel(mk_bin(O_EQ, m, ku(U32, 0)), fof(TY_F32, ku(U32,
	    0x7f800000)), fof(TY_F32, ku(U32, 0x7fc00000)));

	return mk_sel(mk_bin(O_EQ, e, ku(U32, 0)), den, mk_sel(mk_bin(O_EQ, e,
	    ku(U32, 31)), big, nrm));
}

/* channel c of the packed word w */
static nid
unpack_ch(int op, nid w, int c)
{
	nid k;

	w = mk_cast(U32, w);
	switch (op) {
	case O_UNPACK_B10G11R11:
		return ufloat_of(w, 11 * c, c == 2 ? 5 : 6);
	case O_UNPACK_E5B9G9R9:
		/* m 2^(e - 24): exact, the significand has 9 bits */
		k = mk_bin(O_AND, mk_bin(O_SHR, w, ku(U32, 9 * c)), ku(U32,
		    511));
		return mk_bin(O_MUL, mk_cast(TY_F32, k), pow2_of(mk_bin(O_SUB,
		    mk_bin(O_SHR, w, ku(U32, 27)), ku(U32, 24))));
	}
	/* k/1023 (or k/3) as one f32 division: correctly rounded */
	k = c == 3 ? mk_bin(O_SHR, w, ku(U32, 30)) : mk_bin(O_AND,
	    mk_bin(O_SHR, w, ku(U32, 10 * c)), ku(U32, 1023));
	return mk_bin(O_DIV, mk_cast(TY_F32, k), mk_cast(TY_F32, ku(U32,
	    c == 3 ? 3 : 1023)));
}

static int
fsig(ty_t t)	/* the significand's stored bits */
{
	return TYW(t) == 16 ? 10 : TYW(t) == 32 ? 23 : 52;
}

static ty_t
fbitty(ty_t t)
{
	return TY(K_UINT, TYW(t));
}

static nid
fl_bits(nid x)
{
	return mk_un(O_ASUINT, x);
}

static nid
fof(ty_t t, nid u)	/* the float of t whose bits are u */
{
	return mk_un(O_ASFLOAT, mk_cast(fbitty(t), u));
}

static uint64_t
fl_mag(ty_t t)	/* every bit but the sign */
{
	return ((uint64_t)1 << (TYW(t) - 1)) - 1;
}

/* x is a NaN: its bits past the exponent's all ones */
static nid
fnan(nid x)
{
	ty_t t = N.ty[x];
	uint64_t m = fl_mag(t);

	return mk_bin(O_GT, mk_bin(O_AND, fl_bits(x), ku(fbitty(t), m)),
	    ku(fbitty(t), m >> fsig(t) << fsig(t)));
}

/* a signed integer ordered as x is, -0 below +0 (x not a NaN) */
static nid
fl_key(nid x)
{
	ty_t t = N.ty[x], st = TY(K_SINT, TYW(t));
	nid s = mk_un(O_ASINT, x);

	return mk_sel(mk_bin(O_LT, s, mk_const_i(st, 0)), mk_bin(O_XOR, s,
	    mk_const_i(st, (int64_t)fl_mag(t))), s);
}

/* v_min_f32: the other when one is NaN, -0 below +0 */
static nid
fminnum(nid x, nid y)
{
	return mk_sel(fnan(x), y, mk_sel(fnan(y), x, mk_sel(mk_bin(O_LE,
	    fl_key(x), fl_key(y)), x, y)));
}

static nid
fmed3(nid a, nid b, nid c)
{
	nid ka = fl_key(a), kb = fl_key(b), kc = fl_key(c), med;

	med = mk_sel(mk_bin(O_LE, ka, kb), mk_sel(mk_bin(O_LE, kb, kc), b,
	    mk_sel(mk_bin(O_LE, ka, kc), c, a)), mk_sel(mk_bin(O_LE, ka, kc), a,
	    mk_sel(mk_bin(O_LE, kb, kc), c, b)));
	/* with a NaN, v_min3: c when all three are */
	return mk_sel(mk_bin(O_LOR, mk_bin(O_LOR, fnan(a), fnan(b)), fnan(c)),
	    fminnum(fminnum(a, b), c), med);
}

static nid
ki64(int64_t v)
{
	return mk_const_i(I64, v);
}

/* the f64 2^k, k an i64 in -1074..1023 (subnormal below -1022); the shift
 * of the branch not taken is masked to stay in range */
static nid
pow2_64(nid k)
{
	nid nrm = fof(TY_F64, mk_bin(O_SHL, mk_bin(O_AND, mk_cast(U64,
	    mk_bin(O_ADD, k, ki64(1023))), ku(U64, 0x7ff)), ku(U64, 52)));
	nid sub = fof(TY_F64, mk_bin(O_SHL, ku(U64, 1), mk_bin(O_AND,
	    mk_cast(U64, mk_bin(O_ADD, k, ki64(1074))), ku(U64, 63))));

	return mk_sel(mk_bin(O_GE, k, ki64(-1022)), nrm, sub);
}

/* n clamped to -c..c as an i64, whatever n's integer type */
static nid
clamp_exp(nid n, int64_t c)
{
	ty_t t = N.ty[n];

	if (is_const(n))
		return ki64(N.ki[n] > c ? c : N.ki[n] < -c ? -c : N.ki[n]);
	if (TYK(t) == K_UINT || t == TY_BOOL)
		return mk_cast(I64, mk_bin(O_MIN, n, mk_const_i(t, c)));
	if (TYW(t) < 64)
		n = mk_cast(I64, n);
	return mk_bin(O_MAX, mk_bin(O_MIN, mk_cast(I64, n), ki64(c)), ki64(-c));
}

/*
 * x * 2^n, one rounding.  Below f64 the product is exact in f64 (every
 * f32 times 2^k for |k| <= 300 is an f64 normal; past 300 the result is
 * what 300 gives) and the cast rounds it.  An f64 is first scaled
 * exactly (up by at most 2^1023 a step, down only while it stays normal:
 * three steps cover |n| <= 2200, past which every f64 gives 0 or an
 * infinity), then the last factor rounds once.
 */
static nid
fldexp(nid x, nid n)
{
	ty_t t = N.ty[x];
	nid k, y, e, e0, lo, kk, r;
	int i;

	if (TYW(t) < 64)
		return mk_cast(t, mk_bin(O_MUL, mk_cast(TY_F64, x),
		    pow2_64(clamp_exp(n, 300))));
	k = clamp_exp(n, 2200);
	y = x;
	e0 = 0;
	for (i = 0; i < 3; i++) {
		e = mk_cast(I64, mk_bin(O_AND, mk_bin(O_SHR, fl_bits(y), ku(U64,
		    52)), ku(U64, 0x7ff)));
		if (i == 0)
			e0 = e;
		/* down to the least normal exponent, never up */
		lo = mk_bin(O_MIN, mk_bin(O_MAX, mk_bin(O_SUB, ki64(1), e),
		    ki64(-1022)), ki64(0));
		kk = mk_sel(mk_bin(O_GE, k, ki64(0)), mk_bin(O_MIN, k,
		    ki64(1023)), mk_bin(O_MAX, k, lo));
		y = mk_bin(O_MUL, y, pow2_64(kk));
		k = mk_bin(O_SUB, k, kk);
	}
	/* what is left is 0 or y below 2^-1021: below -1074 it is under half
	 * the least subnormal */
	r = mk_sel(mk_bin(O_LT, k, ki64(-1074)), fof(t, mk_bin(O_AND,
	    fl_bits(y), ku(U64, (uint64_t)1 << 63))), mk_bin(O_MUL, y,
	    pow2_64(k)));
	/* an infinity, and a NaN quieted by the steps' products */
	return mk_sel(mk_bin(O_EQ, e0, ki64(0x7ff)), y, r);
}

static nid
ffrexp(int op, nid x)
{
	ty_t t = N.ty[x], ut = fbitty(t), wt = TYW(t) == 64 ? U64 : U32;
	int sig = fsig(t), ew = TYW(t) - 1 - sig;
	int64_t bias = ((int64_t)1 << (ew - 1)) - 1;
	uint64_t emax = ((uint64_t)1 << ew) - 1, mm = ((uint64_t)1 << sig) - 1;
	nid u = fl_bits(x), e, m, len, special, sub, r;

	e = mk_bin(O_AND, mk_bin(O_SHR, u, ku(ut, sig)), ku(ut, emax));
	m = mk_bin(O_AND, u, ku(ut, mm));
	/* a subnormal's significand is len bits long, 1..sig */
	len = mk_bin(O_SUB, ku(wt, TYW(wt)), mk_un(O_CLZ, mk_cast(wt, m)));
	special = mk_bin(O_LOR, mk_bin(O_EQ, e, ku(ut, emax)), mk_bin(O_EQ,
	    mk_bin(O_AND, u, ku(ut, fl_mag(t))), ku(ut, 0)));
	sub = mk_bin(O_EQ, e, ku(ut, 0));
	if (op == O_FREXP_EXP) {
		r = mk_sel(sub, mk_bin(O_SUB, mk_cast(I32, len), mk_const_i(I32,
		    bias - 1 + sig)), mk_bin(O_SUB, mk_cast(I32, e),
		    mk_const_i(I32, bias - 1)));
		return mk_cast(inst_type(op, x, 0, 0), mk_sel(special,
		    mk_const_i(I32, 0), r));
	}
	/* the sign, the exponent of [0.5, 1), and the significand moved up
	 * to its top bit when subnormal */
	r = mk_sel(sub, mk_bin(O_AND, mk_bin(O_SHL, m, mk_cast(ut,
	    mk_bin(O_SUB, ku(wt, sig + 1), len))), ku(ut, mm)), m);
	r = mk_bin(O_OR, mk_bin(O_OR, mk_bin(O_AND, u, ku(ut, (uint64_t)1 <<
	    (TYW(t) - 1))), ku(ut, (uint64_t)(bias - 1) << sig)), r);
	return mk_sel(special, x, fof(t, r));
}

static nid
fl_f32(uint32_t bits)
{
	return mk_un(O_ASFLOAT, ku(U32, bits));
}

static nid
fcube(int op, nid x, nid y, nid z)
{
	nid zero = fl_f32(0), ax = mk_un(O_ABS, x), ay = mk_un(O_ABS, y);
	nid az = mk_un(O_ABS, z), zmaj, ymaj, xn, yn, zn, m;

	/* z is the major axis on ties, then y */
	zmaj = mk_bin(O_LAND, mk_bin(O_GE, az, ax), mk_bin(O_GE, az, ay));
	ymaj = mk_bin(O_GE, ay, ax);
	xn = mk_bin(O_LT, x, zero);
	yn = mk_bin(O_LT, y, zero);
	zn = mk_bin(O_LT, z, zero);
	switch (op) {
	case O_CUBEID:
		/* 5.0, 4.0, 3.0, 2.0, 1.0 */
		return mk_sel(zmaj, mk_sel(zn, fl_f32(0x40a00000),
		    fl_f32(0x40800000)), mk_sel(ymaj, mk_sel(yn,
		    fl_f32(0x40400000), fl_f32(0x40000000)), mk_sel(xn,
		    fl_f32(0x3f800000), zero)));
	case O_CUBESC:
		return mk_sel(zmaj, mk_sel(zn, mk_un(O_NEG, x), x), mk_sel(ymaj,
		    x, mk_sel(xn, z, mk_un(O_NEG, z))));
	case O_CUBETC:
		return mk_sel(zmaj, mk_un(O_NEG, y), mk_sel(ymaj, mk_sel(yn,
		    mk_un(O_NEG, z), z), mk_un(O_NEG, y)));
	}
	m = mk_sel(zmaj, z, mk_sel(ymaj, y, x));
	return mk_bin(O_ADD, m, m);
}

/* 2/pi's bits as LLVM's AMDGPU folder holds them, 37 words */
static const uint32_t two_by_pi[37] = {
	0xa2f9836e, 0x4e441529, 0xfc2757d1, 0xf534ddc0, 0xdb629599, 0x3c439041,
	0xfe5163ab, 0xdebbc561, 0xb7246e3a, 0x424dd2e0, 0x06492eea, 0x09d1921c,
	0xfe1deb1c, 0xb129a73e, 0xe88235f5, 0x2ebb4484, 0xe99c7026, 0xb45f7e41,
	0x3991d639, 0x835339f4, 0x9c845f8b, 0xbdf9283b, 0x1ff897ff, 0xde05980f,
	0xef2f118b, 0x5a0a6d1f, 0x6d367ecf, 0x27cb09b7, 0x4f463f66, 0x9e5fea2d,
	0x7527bac7, 0xebe5f17b, 0x3d0739f7, 0x8a5292ea, 0x6bfb5fb1, 0x1f8d5d08,
	0x56033046,
};

static nid
ftrig_preop(nid x, nid s)
{
	nid w[37], tab, e, sh, idx, out, hi, lo, bs, th, sc, v, r;
	int i;

	for (i = 0; i < 37; i++)
		w[i] = ku(U64, two_by_pi[i]);
	tab = mk_table(w, 37);
	e = mk_bin(O_AND, mk_bin(O_SHR, fl_bits(x), ku(U64, 52)), ku(U64,
	    0x7ff));
	sh = mk_bin(O_ADD, mk_bin(O_MUL, mk_bin(O_AND, mk_cast(U64, s),
	    ku(U64, 31)), ku(U64, 53)), mk_sel(mk_bin(O_GT, e, ku(U64, 1077)),
	    mk_bin(O_SUB, e, ku(U64, 1077)), ku(U64, 0)));
	idx = mk_bin(O_SHR, sh, ku(U64, 5));
	out = mk_bin(O_GE, idx, ku(U64, 35));
	idx = mk_bin(O_MIN, idx, ku(U64, 34));
	hi = mk_bin(O_OR, mk_bin(O_SHL, mk_index(tab, idx), ku(U64, 32)),
	    mk_index(tab, mk_bin(O_ADD, idx, ku(U64, 1))));
	lo = mk_bin(O_SHL, mk_index(tab, mk_bin(O_ADD, idx, ku(U64, 2))),
	    ku(U64, 32));
	bs = mk_bin(O_AND, sh, ku(U64, 31));
	/* the 53 bits from bit sh: the low word's part shifted in by two
	 * steps, so a shift of 0 brings nothing */
	th = mk_bin(O_SHR, mk_bin(O_OR, mk_bin(O_SHL, hi, bs), mk_bin(O_SHR,
	    mk_bin(O_SHR, lo, mk_bin(O_SUB, ku(U64, 63), bs)), ku(U64, 1))),
	    ku(U64, 11));
	sc = mk_bin(O_ADD, mk_bin(O_SUB, ki64(-53), mk_cast(I64, sh)),
	    mk_sel(mk_bin(O_GE, e, ku(U64, 1968)), ki64(128), ki64(0)));
	v = mk_cast(TY_F64, th);
	/* th * 2^sc rounded once: below 2^-1074 as (th * 2^-53) * 2^(sc +
	 * 53), the first product exact; below 2^-1127 it is under half the
	 * least subnormal */
	r = mk_sel(mk_bin(O_GE, sc, ki64(-1074)), mk_bin(O_MUL, v, pow2_64(sc)),
	    mk_sel(mk_bin(O_GE, sc, ki64(-1127)), mk_bin(O_MUL, mk_bin(O_MUL, v,
	    pow2_64(ki64(-53))), pow2_64(mk_bin(O_ADD, sc, ki64(53)))),
	    fof(TY_F64, ku(U64, 0))));
	return mk_sel(out, fof(TY_F64, ku(U64, 0)), r);
}

/* byte k of w (u32 or u64), as a u32 */
static nid
byte_of(nid w, nid k8)
{
	return mk_cast(U32, mk_bin(O_AND, mk_bin(O_SHR, w, k8), ku(N.ty[w],
	    255)));
}

/* lane l of w (u32), wd bits wide, as a u32 */
static nid
lane_of(nid w, int l, int wd)
{
	return mk_bin(O_AND, mk_bin(O_SHR, w, ku(U32, (uint64_t)(l * wd))),
	    ku(U32, (1u << wd) - 1));
}

/* the lane-wise ops: lane l of the result from lanes p, q of a, b */
static nid
lanewise(int op, nid a, nid b)
{
	int wd = op <= O_MSAD4X8 ? 8 : 16, n = 32 / wd, l;
	uint32_t m = (1u << wd) - 1;
	nid r = ku(U32, 0);

	for (l = 0; l < n; l++) {
		nid p = lane_of(a, l, wd), q = lane_of(b, l, wd), v = 0;
		switch (op) {
		case O_ADD4X8: case O_ADD2X16:
			v = mk_bin(O_ADD, p, q);
			break;
		case O_SUB4X8: case O_SUB2X16:
			v = mk_bin(O_SUB, p, q);
			break;
		case O_MUL2X16:
			v = mk_bin(O_MUL, p, q);
			break;
		case O_ABSDIFF4X8:
			v = mk_bin(O_SUB, mk_bin(O_MAX, p, q), mk_bin(O_MIN, p, q));
			break;
		case O_MIN4X8: case O_MIN2X16:
			v = mk_bin(O_MIN, p, q);
			break;
		case O_MAX4X8: case O_MAX2X16:
			v = mk_bin(O_MAX, p, q);
			break;
		case O_SMIN2X16: case O_SMAX2X16:
			/* the halves as i16: x ^ 0x8000 - 0x8000 in i32 */
			p = mk_bin(O_SUB, mk_cast(I32, mk_bin(O_XOR, p, ku(U32, 0x8000))),
			    ku(I32, 0x8000));
			q = mk_bin(O_SUB, mk_cast(I32, mk_bin(O_XOR, q, ku(U32, 0x8000))),
			    ku(I32, 0x8000));
			v = mk_cast(U32, mk_bin(op == O_SMIN2X16 ? O_MIN : O_MAX, p, q));
			break;
		}
		r = mk_bin(O_OR, r, mk_bin(O_SHL, mk_bin(O_AND, v, ku(U32, m)),
		    ku(U32, (uint64_t)(l * wd))));
	}
	return r;
}

nid
inst_expand(int op, nid a, nid b, nid c)
{
	ty_t t = inst_type(op, a, b, c);
	nid r = 0, x, w, ok;
	int i;

	switch (op) {
	case O_BFE:
		/* in 64 bits, so a field of all 32 bits has its mask */
		x = mk_cast(U64, a);
		r = mk_bin(O_AND, mk_bin(O_SHR, x, mk_cast(U64, b)), mk_bin(O_SUB,
		    mk_bin(O_SHL, ku(U64, 1), mk_cast(U64, c)), ku(U64, 1)));
		ok = mk_bin(O_LE, mk_bin(O_ADD, mk_cast(U64, b), mk_cast(U64, c)),
		    ku(U64, 32));
		r = mk_sel(ok, mk_cast(t, r), mk_undef(t));
		break;
	case O_SBFE:
		/* the field, then copies of its top bit above it: all in
		 * unsigned 64 bits, the low 32 read as i32 */
		x = mk_cast(U64, a);
		w = mk_bin(O_SUB, mk_bin(O_SHL, ku(U64, 1), mk_cast(U64, c)),
		    ku(U64, 1));
		r = mk_bin(O_AND, mk_bin(O_SHR, x, mk_cast(U64, b)), w);
		r = mk_bin(O_OR, r, mk_bin(O_SHL, mk_bin(O_SUB, ku(U64, 0),
		    mk_bin(O_AND, mk_bin(O_SHR, r, mk_bin(O_SUB, mk_cast(U64, c),
		    ku(U64, 1))), ku(U64, 1))), mk_cast(U64, c)));
		ok = mk_bin(O_LE, mk_bin(O_ADD, mk_cast(U64, b), mk_cast(U64, c)),
		    ku(U64, 32));
		r = mk_sel(mk_bin(O_EQ, mk_cast(U64, c), ku(U64, 0)), ku(t, 0),
		    mk_sel(ok, mk_cast(t, mk_cast(U32, r)), mk_undef(t)));
		break;
	case O_BREV: {
		static const uint32_t m[5] = {0x55555555u, 0x33333333u,
		    0x0f0f0f0fu, 0x00ff00ffu, 0x0000ffffu};
		r = a;
		for (i = 0; i < 5; i++)
			r = mk_bin(O_OR, mk_bin(O_AND, mk_bin(O_SHR, r, ku(U32,
			    1u << i)), ku(U32, m[i])), mk_bin(O_SHL, mk_bin(O_AND,
			    r, ku(U32, m[i])), ku(U32, 1u << i)));
		break;
	}
	case O_MULHI:
		r = mk_bin(O_SHR, mk_bin(O_MUL, mk_cast(U64, a), mk_cast(U64, b)),
		    ku(U64, 32));
		break;
	case O_SMULHI:
		r = mk_bin(O_SHR, mk_bin(O_MUL, mk_cast(I64, a), mk_cast(I64, b)),
		    ku(I64, 32));
		break;
	case O_FUNL: case O_FUNR:
		w = mk_bin(O_OR, mk_bin(O_SHL, mk_cast(U64, a), ku(U64, 32)),
		    mk_cast(U64, b));
		r = op == O_FUNL ? mk_bin(O_SHR, mk_bin(O_SHL, w, mk_cast(U64, c)),
		    ku(U64, 32)) : mk_bin(O_SHR, w, mk_cast(U64, c));
		r = mk_sel(mk_bin(O_LT, mk_cast(U64, c), ku(U64, 32)),
		    mk_cast(t, r), mk_undef(t));
		break;
	case O_MED3:
		if (TYK(t) == K_FLOAT)
			return fmed3(a, b, c);
		r = mk_bin(O_MAX, mk_bin(O_MIN, a, b), mk_bin(O_MIN,
		    mk_bin(O_MAX, a, b), c));
		break;
	case O_LDEXP:
		return fldexp(a, b);
	case O_FREXP_MANT: case O_FREXP_EXP:
		return ffrexp(op, a);
	case O_CUBEID: case O_CUBESC: case O_CUBETC: case O_CUBEMA:
		return fcube(op, a, b, c);
	case O_TRIG_PREOP:
		return ftrig_preop(a, b);
	case O_UNPACK_B10G11R11: case O_UNPACK_E5B9G9R9:
	case O_UNPACK_A2B10G10R10:
		return unpack_ch(op, a, (int)N.ki[b]);
	case O_USAD4X8:
		r = c;
		for (i = 0; i < 4; i++) {
			nid p = byte_of(a, ku(U32, 8 * i));
			nid q = byte_of(b, ku(U32, 8 * i));
			r = mk_bin(O_ADD, r, mk_bin(O_SUB, mk_bin(O_MAX, p, q),
			    mk_bin(O_MIN, p, q)));
		}
		break;
	case O_MULWIDE: case O_MADWIDE:
		r = mk_bin(O_MUL, mk_cast(U64, a), mk_cast(U64, b));
		if (op == O_MADWIDE)
			r = mk_bin(O_ADD, r, c);
		break;
	case O_BFI: {
		uint32_t m = field_mask(BFI_P(N.ki[c]), BFI_N(N.ki[c]));
		r = mk_bin(O_OR, mk_bin(O_AND, a, ku(U32, ~m)), mk_bin(O_AND,
		    mk_bin(O_SHL, b, ku(U32, (uint64_t)BFI_P(N.ki[c]) & 31)),
		    ku(U32, m)));
		break;
	}
	case O_BYTEPERM:
		w = mk_bin(O_OR, mk_bin(O_SHL, mk_cast(U64, b), ku(U64, 32)),
		    mk_cast(U64, a));
		r = ku(U32, 0);
		for (i = 0; i < 4; i++) {
			nid s = mk_bin(O_AND, mk_bin(O_SHR, c, ku(U32, 4 * i)),
			    ku(U32, 15));
			nid by = byte_of(w, mk_cast(U64, mk_bin(O_SHL, mk_bin(O_AND,
			    s, ku(U32, 7)), ku(U32, 3))));
			by = mk_sel(mk_bin(O_NE, mk_bin(O_AND, s, ku(U32, 8)),
			    ku(U32, 0)), mk_sel(mk_bin(O_NE, mk_bin(O_AND, by,
			    ku(U32, 128)), ku(U32, 0)), ku(U32, 255), ku(U32, 0)),
			    by);
			r = mk_bin(O_OR, r, mk_bin(O_SHL, by, ku(U32, 8 * i)));
		}
		break;
	case O_ADD4X8: case O_SUB4X8: case O_ABSDIFF4X8: case O_MIN4X8:
	case O_MAX4X8: case O_ADD2X16: case O_SUB2X16: case O_MUL2X16:
	case O_MIN2X16: case O_MAX2X16: case O_SMIN2X16: case O_SMAX2X16:
		r = lanewise(op, a, b);
		break;
	case O_MSAD4X8:
		r = c;
		for (i = 0; i < 4; i++) {
			nid p = byte_of(a, ku(U32, 8 * i));
			nid q = byte_of(b, ku(U32, 8 * i));
			r = mk_bin(O_ADD, r, mk_sel(mk_bin(O_NE, p, ku(U32, 0)),
			    mk_bin(O_SUB, mk_bin(O_MAX, p, q), mk_bin(O_MIN, p, q)),
			    ku(U32, 0)));
		}
		break;
	case O_LERP4X8:
		r = ku(U32, 0);
		for (i = 0; i < 4; i++) {
			nid p = byte_of(a, ku(U32, 8 * i));
			nid q = byte_of(b, ku(U32, 8 * i));
			nid h = mk_bin(O_AND, mk_bin(O_SHR, c, ku(U32, 8 * i)),
			    ku(U32, 1));
			r = mk_bin(O_OR, r, mk_bin(O_SHL, mk_bin(O_SHR, mk_bin(O_ADD,
			    mk_bin(O_ADD, p, q), h), ku(U32, 1)), ku(U32, 8 * i)));
		}
		break;
	case O_QSAD4X8: case O_MQSAD4X8: {
		int k;
		r = ku(U64, 0);
		for (k = 0; k < 4; k++) {
			nid win = mk_cast(U32, mk_bin(O_SHR, a, ku(U64, 8 * k)));
			nid sum = mk_cast(U32, mk_bin(O_AND, mk_bin(O_SHR, c, ku(U64,
			    16 * k)), ku(U64, 0xffff)));
			for (i = 0; i < 4; i++) {
				nid p = byte_of(win, ku(U32, 8 * i));
				nid q = byte_of(b, ku(U32, 8 * i));
				nid d = mk_bin(O_SUB, mk_bin(O_MAX, p, q), mk_bin(O_MIN,
				    p, q));
				if (op == O_MQSAD4X8)
					d = mk_sel(mk_bin(O_NE, q, ku(U32, 0)), d, ku(U32,
					    0));
				sum = mk_bin(O_ADD, sum, d);
			}
			r = mk_bin(O_OR, r, mk_bin(O_SHL, mk_cast(U64, mk_bin(O_AND,
			    sum, ku(U32, 0xffff))), ku(U64, 16 * k)));
		}
		break;
	}
	case O_MIN3:
		r = mk_bin(O_MIN, mk_bin(O_MIN, a, b), c);
		break;
	case O_MAX3:
		r = mk_bin(O_MAX, mk_bin(O_MAX, a, b), c);
		break;
	case O_MINMAX:
		r = mk_bin(O_MAX, mk_bin(O_MIN, a, b), c);
		break;
	case O_MAXMIN:
		r = mk_bin(O_MIN, mk_bin(O_MAX, a, b), c);
		break;
	case O_MINADD:
		r = mk_bin(O_ADD, mk_bin(O_MIN, a, b), c);
		break;
	case O_MAXADD:
		r = mk_bin(O_ADD, mk_bin(O_MAX, a, b), c);
		break;
	case O_SAD:
		r = mk_bin(O_ADD, mk_bin(O_SUB, mk_bin(O_MAX, a, b), mk_bin(O_MIN,
		    a, b)), c);
		break;
	case O_ADDSAT: case O_SUBSAT:
		/* in 64 bits, where neither overflows, then clamped */
		x = mk_bin(op == O_ADDSAT ? O_ADD : O_SUB, mk_cast(I64, a),
		    mk_cast(I64, b));
		r = t == U32 ? mk_bin(O_MAX, mk_bin(O_MIN, x, mk_const_i(I64,
		    0xffffffffLL)), mk_const_i(I64, 0)) : mk_bin(O_MAX,
		    mk_bin(O_MIN, x, mk_const_i(I64, INT32_MAX)), mk_const_i(I64,
		    INT32_MIN));
		break;
	case O_BYTESEL:
		w = mk_bin(O_OR, mk_bin(O_SHL, mk_cast(U64, a), ku(U64, 32)),
		    mk_cast(U64, b));
		r = ku(U32, 0);
		for (i = 0; i < 4; i++) {
			nid s = mk_bin(O_AND, mk_bin(O_SHR, c, ku(U32, 8 * i)),
			    ku(U32, 255));
			/* the byte's number masked to 0..7, the sign's (byte 2k + 1)
			 * to 0..3: every shift in range whichever is taken */
			nid by = byte_of(w, mk_cast(U64, mk_bin(O_SHL, mk_bin(O_AND,
			    s, ku(U32, 7)), ku(U32, 3))));
			nid sg = mk_bin(O_AND, mk_bin(O_SHR, w, mk_cast(U64,
			    mk_bin(O_ADD, mk_bin(O_SHL, mk_bin(O_AND, s, ku(U32, 3)),
			    ku(U32, 4)), ku(U32, 15)))), ku(U64, 1));
			sg = mk_sel(mk_bin(O_NE, sg, ku(U64, 0)), ku(U32, 255),
			    ku(U32, 0));
			by = mk_sel(mk_bin(O_GE, s, ku(U32, 13)), ku(U32, 255),
			    mk_sel(mk_bin(O_EQ, s, ku(U32, 12)), ku(U32, 0),
			    mk_sel(mk_bin(O_GE, s, ku(U32, 8)), sg, by)));
			r = mk_bin(O_OR, r, mk_bin(O_SHL, by, ku(U32, 8 * i)));
		}
		break;
	}
	return mk_cast(t, r);
}

/* x as an operand of type t: a float of t, or a constant rounded to t */
static nid
float_operand(const char *nm, ty_t t, nid x)
{
	if (TYK(N.ty[x]) != K_FLOAT && !is_const(x))
		fail("%s takes %s operands, not %s", nm, ty_str(t),
		    ty_str(N.ty[x]));
	return mk_cast(t, x);
}

static nid
mk_finst(int op, nid a, nid b, nid c)
{
	int ar = opinfo[op].arity;
	const char *nm = opinfo[op].name;
	ty_t t;

	switch (op) {
	case O_MED3:
		t = inst_type(op, a, b, c);
		if (TYK(t) != K_FLOAT)
			fail("med3 takes integers or floats (f16, f32, f64), "
			    "not %s", ty_str(t));
		a = float_operand(nm, t, a);
		b = float_operand(nm, t, b);
		c = float_operand(nm, t, c);
		break;
	case O_LDEXP: case O_FREXP_MANT: case O_FREXP_EXP:
		if (TYK(N.ty[a]) != K_FLOAT)
			fail("%s takes a float (f16, f32 or f64), not %s", nm,
			    ty_str(N.ty[a]));
		if (op == O_LDEXP && (!TY_ISINT(N.ty[b]) || N.ty[b] == TY_BOOL))
			fail("ldexp: the power of two is an integer, not %s",
			    ty_str(N.ty[b]));
		break;
	case O_CUBEID: case O_CUBESC: case O_CUBETC: case O_CUBEMA:
		/* v_cube* are f32 only: another float is not converted */
		if ((!is_const(a) && N.ty[a] != TY_F32) || (!is_const(b) &&
		    N.ty[b] != TY_F32) || (!is_const(c) && N.ty[c] != TY_F32))
			fail("%s takes f32 operands, not %s, %s, %s", nm,
			    ty_str(N.ty[a]), ty_str(N.ty[b]), ty_str(N.ty[c]));
		a = mk_cast(TY_F32, a);
		b = mk_cast(TY_F32, b);
		c = mk_cast(TY_F32, c);
		break;
	case O_TRIG_PREOP:
		if (N.ty[a] != TY_F64 && !is_const(a))
			fail("trig_preop takes an f64, not %s",
			    ty_str(N.ty[a]));
		a = mk_cast(TY_F64, a);
		if (!TY_ISINT(N.ty[b]) || N.ty[b] == TY_BOOL)
			fail("trig_preop: the segment is an integer, not %s",
			    ty_str(N.ty[b]));
		break;
	}
	t = inst_type(op, a, b, c);
	if (N.op[a] == O_UNDEF || (ar >= 2 && N.op[b] == O_UNDEF) ||
	    (ar == 3 && N.op[c] == O_UNDEF))
		return mk_undef(t);
	if (is_const(a) && (ar < 2 || is_const(b)) && (ar < 3 || is_const(c))) {
		nid r = inst_expand(op, a, b, c);
		if (is_const(r) || N.op[r] == O_UNDEF)
			return r;
	}
	return mk_raw(op, t, a, ar >= 2 ? b : 0, ar == 3 ? c : 0, 0, NF_SG |
	    ((N.fl[a] | (ar >= 2 ? N.fl[b] : 0) | (ar == 3 ? N.fl[c] : 0)) &
	    NF_PARTIAL));
}

nid
mk_inst(int op, nid a, nid b, nid c)
{
	int ar = opinfo[op].arity, fl = NF_SG;
	const char *nm = opinfo[op].name;
	ty_t t;
	nid r;

	if (IS_ASTC(op))
		fail("%s cannot be rebuilt from its operands alone (its "
		    "constant rides in the node: mk_like)", nm);
	if (float_inst(op) || (op == O_MED3 && TY_FP(inst_type(op, a, b, c))))
		return mk_finst(op, a, b, c);
	if (!TY_ISINT(N.ty[a]) || (ar >= 2 && !TY_ISINT(N.ty[b])) ||
	    (ar == 3 && !TY_ISINT(N.ty[c])))
		fail("%s takes integer operands", nm);
	switch (op) {
	case O_BFE: case O_SBFE: case O_BREV:
		a = mk_cast(U32, a);
		break;
	case O_MULHI: case O_USAD4X8: case O_BYTEPERM: case O_BYTESEL:
	case O_ADD4X8: case O_SUB4X8: case O_ABSDIFF4X8: case O_MIN4X8:
	case O_MAX4X8: case O_MSAD4X8: case O_ADD2X16: case O_SUB2X16:
	case O_MUL2X16: case O_MIN2X16: case O_MAX2X16: case O_SMIN2X16:
	case O_SMAX2X16: case O_LERP4X8: case O_SAD:
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		if (ar == 3)
			c = mk_cast(U32, c);
		break;
	case O_QSAD4X8: case O_MQSAD4X8:
		a = mk_cast(U64, a);
		b = mk_cast(U32, b);
		c = mk_cast(U64, c);
		break;
	case O_ADDSAT: case O_SUBSAT:
		/* two literals: u32 unless one is negative */
		t = N.ty[mk_bin(O_ADD, a, b)];
		if (TYK(t) == K_LIT)
			t = (is_const(a) && N.ki[a] < 0) || (is_const(b) &&
			    N.ki[b] < 0) ? I32 : U32;
		else if (t != TY_BOOL && TYW(t) < 32)
			t = TYK(t) == K_SINT ? I32 : U32;
		if (t != U32 && t != I32)
			fail("%s works on 32-bit words, u32 or i32, not %s", nm,
			    ty_str(t));
		a = mk_cast(t, a);
		b = mk_cast(t, b);
		break;
	case O_MULWIDE: case O_MADWIDE:
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		if (op == O_MADWIDE)
			c = mk_cast(U64, c);
		break;
	case O_BFI:
		if (!is_const(c) || BFI_P(N.ki[c]) + BFI_N(N.ki[c]) > 32 ||
		    N.ki[c] < 0)
			fail("bfi: its field is a constant offset and count of a "
			    "32-bit word, not %s", node_str(c));
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		break;
	case O_SMULHI:
		a = mk_cast(I32, a);
		b = mk_cast(I32, b);
		break;
	case O_UNPACK_B10G11R11: case O_UNPACK_E5B9G9R9:
	case O_UNPACK_A2B10G10R10:
		if (!is_const(b) || N.ki[b] < 0 || N.ki[b] > (op ==
		    O_UNPACK_A2B10G10R10 ? 3 : 2))
			fail("%s takes a constant channel, 0..%d (%s), not %s",
			    nm, op == O_UNPACK_A2B10G10R10 ? 3 : 2, op ==
			    O_UNPACK_A2B10G10R10 ? "r g b a" : "r g b",
			    node_str(b));
		a = mk_cast(U32, a);
		break;
	case O_FUNL: case O_FUNR:
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		break;
	case O_MED3: case O_MIN3: case O_MAX3: case O_MINMAX: case O_MAXMIN:
	case O_MINADD: case O_MAXADD:
		t = inst_type(op, a, b, c);
		if (t == TY_BOOL)
			fail("%s takes numbers", nm);
		a = mk_cast(t, a);
		b = mk_cast(t, b);
		c = mk_cast(t, c);
		break;
	}
	t = inst_type(op, a, b, c);
	if ((op == O_BFE || op == O_SBFE) && is_const(b) && is_const(c) &&
	    (N.ki[b] < 0 || N.ki[c] < 0 || N.ki[b] + N.ki[c] > 32))
		fail("%s: bits %lld..%lld are not a field of a 32-bit word", nm,
		    (long long)N.ki[b], (long long)(N.ki[b] + N.ki[c] - 1));
	if ((op == O_FUNL || op == O_FUNR) && is_const(c) && (N.ki[c] < 0 ||
	    N.ki[c] > 31))
		fail("%s: a shift of %lld is not in 0..31", nm, (long long)N.ki[c]);
	if (N.op[a] == O_UNDEF || (ar >= 2 && N.op[b] == O_UNDEF) ||
	    (ar == 3 && N.op[c] == O_UNDEF))
		return mk_undef(t);
	if (is_const(a) && (ar < 2 || is_const(b)) && (ar < 3 || is_const(c))) {
		r = inst_expand(op, a, b, c);
		if (is_const(r) || N.op[r] == O_UNDEF)
			return r;
	}
	fl |= (N.fl[a] | (ar >= 2 ? N.fl[b] : 0) | (ar == 3 ? N.fl[c] : 0)) &
	    NF_PARTIAL;
	if ((op == O_BFE || op == O_SBFE) && !(is_const(b) && is_const(c)))
		fl |= NF_PARTIAL;
	if ((op == O_FUNL || op == O_FUNR) && !is_const(c))
		fl |= NF_PARTIAL;
	return mk_raw(op, t, a, ar >= 2 ? b : 0, ar == 3 ? c : 0, 0, fl);
}

/* ---- find: the instructions for the patterns they are ---- */

/*
 * Each pattern below is what one instruction computes, written with the
 * base operations.  Where the active model prices the instruction, the
 * pattern is offered as that instruction, kept when the whole function
 * gets cheaper, and the result is checked against the function on its
 * domain (the instruction's definition is the pattern, so the check is
 * usually the same circuit twice).  Nothing here knows a target.
 */

static int
klog2(int64_t v)	/* v + 1 a power of two: its exponent, else -1 */
{
	int n = 0;

	if (v <= 0 || (v & (v + 1)))
		return -1;
	while (v) {
		v >>= 1;
		n++;
	}
	return n;
}

static nid
kint(int64_t v)
{
	return mk_const_i(TY_LIT, v);
}

static int
cshift(nid n, int op, ty_t t, nid *x, int *k)
{
	if (N.op[n] != op || N.ty[n] != t || !is_const(N.b[n]) ||
	    N.ki[N.b[n]] < 0 || N.ki[N.b[n]] > 31)
		return 0;
	*x = N.a[n];
	*k = (int)N.ki[N.b[n]];
	return 1;
}

/* the constant operand of an and, and the other one */
static int
and_const(nid n, nid *other, uint32_t *k)
{
	if (N.op[n] != O_AND || N.ty[n] != U32)
		return 0;
	if (is_const(N.b[n])) {
		*k = (uint32_t)N.ki[N.b[n]];
		*other = N.a[n];
		return 1;
	}
	if (is_const(N.a[n])) {
		*k = (uint32_t)N.ki[N.a[n]];
		*other = N.b[n];
		return 1;
	}
	return 0;
}

/*
 * r as the field part of a bit field insert at bits p..p+n-1 (mask m):
 * (ins << p) & m, (ins & (2^n - 1)) << p, or ins << p (ins itself when
 * p is 0) where ins's range keeps it in n bits; the inserted value, or 0
 */
static nid
bfi_ins(const struct func *f, nid r, int p, int n, uint32_t m,
    struct sbuf *how)
{
	nid t, ins;
	uint32_t k;
	int s;
	int64_t lo, hi;

	if (and_const(r, &t, &k) && k == m && (p == 0 || (cshift(t, O_SHL, U32,
	    &ins, &s) && s == p)))
		return p == 0 ? t : ins;
	if (p > 0 && cshift(r, O_SHL, U32, &t, &s) && s == p) {
		if (and_const(t, &ins, &k) && k == (m >> p))
			return ins;
		r = t;
	} else if (p > 0)
		return 0;
	if (sg_range_of(f, r, 32, &lo, &hi) && lo >= 0 && hi <= (int64_t)(m >>
	    p)) {
		sb_printf(how, " (%s <= %lld, by intervals)", node_str(r),
		    (long long)hi);
		return r;
	}
	return 0;
}

/* q, a u64, as the u32 it widens: a word or narrower unsigned widened, a
 * constant below 2^32, or a value whose range stays below it; 0 if not */
static nid
narrow32(const struct func *f, nid q, struct sbuf *how)
{
	int64_t lo, hi;

	if (N.op[q] == O_CAST && TYK(N.ty[N.a[q]]) == K_UINT &&
	    TYW(N.ty[N.a[q]]) <= 32)
		return mk_cast(U32, N.a[q]);
	if (is_const(q))
		return (uint64_t)N.ki[q] <= 0xffffffffULL ? ku(U32,
		    (uint64_t)N.ki[q]) : 0;
	if (sg_range_of(f, q, 32, &lo, &hi) && lo >= 0 && hi <= 0xffffffffLL) {
		sb_printf(how, " (%s <= %lld, by intervals)", node_str(q),
		    (long long)hi);
		return mk_cast(U32, q);
	}
	return 0;
}

/* x, a u64 product of two words: *a and *b those words */
static int
wide_product(const struct func *f, nid x, nid *a, nid *b, struct sbuf *how)
{
	if (N.op[x] == O_MULWIDE) {
		*a = N.a[x];
		*b = N.b[x];
		return 1;
	}
	if (N.op[x] != O_MUL || N.ty[x] != U64)
		return 0;
	return (*a = narrow32(f, N.a[x], how)) != 0 &&
	    (*b = narrow32(f, N.b[x], how)) != 0;
}

/* x, a word's field replaced, as bfi: (base & ~m) | field, with or, xor
 * or +, m the field's bits */
static nid
bfi_of(const struct func *f, nid x, struct sbuf *why)
{
	int j, n, q, op = N.op[x];

	if ((op != O_OR && op != O_XOR && op != O_ADD) || N.ty[x] != U32)
		return 0;
	for (j = 0; j < 2; j++) {
		nid l = j ? N.b[x] : N.a[x], r = j ? N.a[x] : N.b[x];
		nid base, ins;
		uint32_t kk, m;
		struct sbuf how = {0};
		if (!and_const(l, &base, &kk) || (m = ~kk) == 0 ||
		    m == 0xffffffffu)
			continue;
		for (q = 0; !((m >> q) & 1); q++)
			;
		for (n = 0; q + n < 32 && ((m >> (q + n)) & 1); n++)
			;
		if (m != field_mask(q, n) ||
		    (ins = bfi_ins(f, r, q, n, m, &how)) == 0) {
			free(how.p);
			continue;
		}
		sb_printf(why, "%s: bits %d..%d of %s replaced by the low %d "
		    "bits of %s%s", shared_str(x, 80), q, q + n - 1,
		    node_str(base), n, node_str(ins), sb_str(&how));
		free(how.p);
		return mk_inst(O_BFI, base, ins, bfi_field(q, n));
	}
	return 0;
}

/* x, a wide product plus a u64, as madwide */
static nid
madwide_of(const struct func *f, nid x, struct sbuf *why)
{
	nid a, b;
	int j;

	if (N.op[x] != O_ADD || N.ty[x] != U64)
		return 0;
	for (j = 0; j < 2; j++) {
		nid pr = j ? N.b[x] : N.a[x], c = j ? N.a[x] : N.b[x];
		struct sbuf how = {0};
		if (wide_product(f, pr, &a, &b, &how)) {
			sb_printf(why, "%s: the 64-bit product of two words "
			    "plus %s%s", shared_str(x, 80), node_str(c),
			    sb_str(&how));
			free(how.p);
			return mk_inst(O_MADWIDE, a, b, c);
		}
		free(how.p);
	}
	return 0;
}

/* n without its conversions */
static nid
bare(nid n)
{
	while (N.op[n] == O_CAST && TY_ISINT(N.ty[n]))
		n = N.a[n];
	return n;
}

/* p as byte i of a u32 word w: (w >> 8i) & 255, w & 255, w >> 24 or
 * bfe(w, 8i, 8) */
static int
byte_field(nid p, nid *w, int *i)
{
	nid t;
	uint32_t k;
	int s;

	p = bare(p);
	if (N.op[p] == O_BFE && is_const(N.b[p]) && is_const(N.c[p]) &&
	    N.ki[N.c[p]] == 8 && N.ki[N.b[p]] % 8 == 0) {
		*w = N.a[p];
		*i = (int)N.ki[N.b[p]] / 8;
		return 1;
	}
	if (cshift(p, O_SHR, U32, &t, &s) && s == 24) {
		*w = t;
		*i = 3;
		return 1;
	}
	if (!and_const(p, &t, &k) || k != 255)
		return 0;
	if (cshift(t, O_SHR, U32, w, &s) && s % 8 == 0 && s <= 24) {
		*i = s / 8;
		return 1;
	}
	if (N.ty[t] != U32)
		return 0;
	*w = t;
	*i = 0;
	return 1;
}

/* t as |a.byte[i] - b.byte[i]|: max(p, q) - min(p, q) or abs(p - q) */
static int
byte_absdiff(nid t, nid *a, nid *b, int *i)
{
	nid p, q, x, y;
	int i2;

	t = bare(t);
	if (N.op[t] == O_SUB && N.op[N.a[t]] == O_MAX && N.op[N.b[t]] ==
	    O_MIN) {
		x = N.a[t];
		y = N.b[t];
		if (!((N.a[x] == N.a[y] && N.b[x] == N.b[y]) ||
		    (N.a[x] == N.b[y] && N.b[x] == N.a[y])))
			return 0;
		p = N.a[x];
		q = N.b[x];
	} else if (N.op[t] == O_ABS && N.op[bare(N.a[t])] == O_SUB) {
		p = N.a[bare(N.a[t])];
		q = N.b[bare(N.a[t])];
	} else
		return 0;
	return byte_field(p, a, i) && byte_field(q, b, &i2) && i2 == *i;
}

#define MAXTERM 16

static int
sum_terms(nid x, ty_t t, nid *out, int n)
{
	if (N.op[x] == O_ADD && N.ty[x] == t) {
		n = sum_terms(N.a[x], t, out, n);
		return n < 0 ? n : sum_terms(N.b[x], t, out, n);
	}
	if (n == MAXTERM)
		return -1;
	out[n++] = x;
	return n;
}

/* x, a sum with the four bytes' differences of two words in it: usad4x8,
 * the rest of the sum its accumulator */
static nid
usad_of(nid x, struct sbuf *why)
{
	nid tm[MAXTERM], a, b, a0 = 0, b0 = 0, c = 0;
	int n, k, i, seen = 0;
	uint8_t used[MAXTERM] = {0};

	if (N.op[x] != O_ADD || !TY_ISINT(N.ty[x]) || N.ty[x] == TY_BOOL ||
	    (n = sum_terms(x, N.ty[x], tm, 0)) < 4)
		return 0;
	for (k = 0; k < n; k++) {
		if (!byte_absdiff(tm[k], &a, &b, &i))
			continue;
		if (!a0) {
			a0 = a;
			b0 = b;
		}
		if (!((a == a0 && b == b0) || (a == b0 && b == a0)) ||
		    ((seen >> i) & 1))
			continue;
		seen |= 1 << i;
		used[k] = 1;
	}
	if (seen != 15)
		return 0;
	for (k = 0; k < n; k++)
		if (!used[k])
			c = c ? mk_bin(O_ADD, c, tm[k]) : tm[k];
	sb_printf(why, "%s: the sum of the four bytes' absolute differences "
	    "of %s and %s%s%s", shared_str(x, 80), node_str(a0), node_str(b0),
	    c ? ", plus " : "", c ? node_str(c) : "");
	return mk_cast(N.ty[x], mk_inst(O_USAD4X8, a0, b0, c ? c : ku(U32, 0)));
}

/*
 * x, a u32 of one or two u32 words and constants, run at random points of
 * the words: its slots, how many, and at sample k the words w0[k], w1[k]
 * (0 with one word) and x's value r[k].  0 when x is not of that kind or
 * is undefined at a point.
 */
#define BP_SAMPLES 48

static int
word_samples_t(const struct func *f, nid x, ty_t t, int *slot, int *ns,
    uint32_t *w0, uint32_t *w1, uint32_t *r)
{
	uint64_t sl = node_slots(x), rs = 0x9e3779b97f4a7c15ULL;
	int j, smp, ok = 1;
	struct plan pl;
	int64_t *in_i;
	double *in_r;
	nid roots[1] = {x};
	uint32_t loc;

	*ns = 0;
	if (N.ty[x] != t || N.op[x] == O_VAR || IS_INST(N.op[x]) ||
	    (sl >> MAXP) || (N.fl[x] & NF_SG))
		return 0;
	for (j = 0; j < f->np; j++)
		if ((sl >> j) & 1) {
			if (*ns == 2 || f->p[j].ty != t)
				return 0;
			slot[(*ns)++] = j;
		}
	if (*ns == 0)
		return 0;
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	for (smp = 0; smp < BP_SAMPLES; smp++)
		for (j = 0; j < *ns; j++) {
			rs ^= rs << 13;
			rs ^= rs >> 7;
			rs ^= rs << 17;
			in_i[(size_t)slot[j] * CH + smp] = t == I32 ?
			    (int64_t)(int32_t)(uint32_t)rs :
			    (int64_t)(uint32_t)rs;
		}
	plan_build(&pl, roots, 1, 0);
	loc = plan_loc(&pl, x);
	plan_run(&pl, BP_SAMPLES, in_i, in_r);
	for (smp = 0; smp < BP_SAMPLES; smp++) {
		if (COLF(&pl, loc)[smp] & F_UNDEF)
			ok = 0;
		r[smp] = (uint32_t)COLI(&pl, loc)[smp];
		w0[smp] = (uint32_t)in_i[(size_t)slot[0] * CH + smp];
		w1[smp] = *ns == 2 ? (uint32_t)in_i[(size_t)slot[1] * CH + smp] : 0;
	}
	plan_free(&pl);
	free(in_i);
	free(in_r);
	return ok;
}

static int
word_samples(const struct func *f, nid x, int *slot, int *ns, uint32_t *w0,
    uint32_t *w1, uint32_t *r)
{
	return word_samples_t(f, x, U32, slot, ns, w0, w1, r);
}

/*
 * x, a u32 of one or two u32 words and constants, told byte by byte at
 * random points of the words, then checked with the function: what each
 * result byte can be among byteperm's choices (bp: bit j for byte j of
 * b:a, j + 8 for that byte's sign; with one word b is 0) and bytesel's
 * (bs: byte j of hi:lo, 8 + k the sign of byte 2k + 1, 12 the byte 0, 13
 * the byte 255; with one word hi is the word again).  For what the GF(2)
 * matrix of one word does not say: two words, a byte's sign, a constant
 * byte.  0 when x is not of that kind.
 */
static int
byte_choices(const struct func *f, nid x, int *slot, int *ns, uint16_t *bp,
    uint16_t *bs)
{
	uint32_t w0[BP_SAMPLES], w1[BP_SAMPLES], r[BP_SAMPLES];
	int j, i, smp;

	if (!word_samples(f, x, slot, ns, w0, w1, r))
		return 0;
	for (i = 0; i < 4; i++)
		bp[i] = bs[i] = 0xffff;
	for (smp = 0; smp < BP_SAMPLES; smp++) {
		uint64_t ab = w0[smp] | (uint64_t)w1[smp] << 32;
		uint64_t hl = *ns == 2 ? ab : w0[smp] | (uint64_t)w0[smp] << 32;
		for (i = 0; i < 4; i++) {
			uint32_t rb = (r[smp] >> (8 * i)) & 255;
			for (j = 0; j < 16; j++) {
				uint32_t by = (uint32_t)(ab >> (8 * (j & 7))) & 255;
				if (j & 8)
					by = (by & 128) ? 255 : 0;
				if (rb != by)
					bp[i] &= (uint16_t)~(1u << j);
			}
			for (j = 0; j < 14; j++) {
				uint32_t by = j < 8 ? (uint32_t)(hl >> (8 * j)) & 255 :
				    j < 12 ? ((hl >> (16 * (j - 8) + 15)) & 1) * 255 :
				    j == 12 ? 0 : 255;
				if (rb != by)
					bs[i] &= (uint16_t)~(1u << j);
			}
		}
	}
	for (i = 0; i < 4; i++)
		bs[i] &= 0x3fff;
	return 1;
}

/* the lane-wise ops in C, for find's samples (inst_expand defines them) */
static uint32_t
lanewise_c(int op, uint32_t a, uint32_t b)
{
	int wd = op <= O_MSAD4X8 ? 8 : 16, l;
	uint32_t m = (1u << wd) - 1, r = 0;

	for (l = 0; l < 32 / wd; l++) {
		uint32_t p = (a >> (l * wd)) & m, q = (b >> (l * wd)) & m, v = 0;
		int32_t sp = (int32_t)(p ^ 0x8000) - 0x8000;
		int32_t sq = (int32_t)(q ^ 0x8000) - 0x8000;
		switch (op) {
		case O_ADD4X8: case O_ADD2X16: v = p + q; break;
		case O_SUB4X8: case O_SUB2X16: v = p - q; break;
		case O_MUL2X16: v = p * q; break;
		case O_ABSDIFF4X8: v = p > q ? p - q : q - p; break;
		case O_MIN4X8: case O_MIN2X16: v = p < q ? p : q; break;
		case O_MAX4X8: case O_MAX2X16: v = p > q ? p : q; break;
		case O_SMIN2X16: v = (uint32_t)(sp < sq ? sp : sq); break;
		case O_SMAX2X16: v = (uint32_t)(sp > sq ? sp : sq); break;
		case O_MSAD4X8:
			r += p ? (p > q ? p - q : q - p) : 0;
			continue;
		}
		r |= (v & m) << (l * wd);
	}
	return r;
}

/* lerp4x8 in C, for find's samples */
static uint32_t
lerp_c(uint32_t a, uint32_t b, uint32_t c)
{
	uint32_t r = 0;
	int i;

	for (i = 0; i < 32; i += 8)
		r |= ((((a >> i) & 255) + ((b >> i) & 255) + ((c >> i) & 1)) >> 1)
		    << i;
	return r;
}

/* addsat and subsat of two u32 in C, for find's samples */
static uint32_t
sat_c(int op, uint32_t a, uint32_t b)
{
	if (op == O_ADDSAT)
		return a + b < a ? 0xffffffffu : a + b;
	return a < b ? 0 : a - b;
}

/* of two i32 (their bits), clamped to the range of i32 */
static uint32_t
ssat_c(int op, uint32_t a, uint32_t b)
{
	int64_t v = op == O_ADDSAT ? (int64_t)(int32_t)a + (int32_t)b :
	    (int64_t)(int32_t)a - (int32_t)b;

	return (uint32_t)(int32_t)(v > INT32_MAX ? INT32_MAX : v < INT32_MIN ?
	    INT32_MIN : v);
}

/*
 * x, an i32 of two i32, as a saturating add or subtract (either order of
 * the subtract): told at random points, then checked with the function
 */
static nid
ssat_of(const struct func *f, nid x, struct sbuf *why)
{
	uint32_t w0[BP_SAMPLES], w1[BP_SAMPLES], r[BP_SAMPLES];
	int slot[2] = {-1, -1}, ns, k, o, smp;

	if (!cclass_available(model_active(), C_ADDSAT) ||
	    !word_samples_t(f, x, I32, slot, &ns, w0, w1, r) || ns != 2)
		return 0;
	for (k = 0; k < 3; k++) {
		int op = k ? O_SUBSAT : O_ADDSAT;
		o = k == 2;
		for (smp = 0; smp < BP_SAMPLES; smp++)
			if (r[smp] != ssat_c(op, o ? w1[smp] : w0[smp],
			    o ? w0[smp] : w1[smp]))
				break;
		if (smp == BP_SAMPLES) {
			nid a = mk_var((uint32_t)slot[o], f->p[slot[o]].name,
			    I32);
			nid b = mk_var((uint32_t)slot[1 - o], f->p[slot[1 - o]].name,
			    I32);
			sb_printf(why, "%s: %s(%s, %s), %s clamped to the range of "
			    "i32, told at %d random points", shared_str(x, 80),
			    opinfo[op].name, node_str(a), node_str(b), op == O_ADDSAT ?
			    "the sum" : "the difference", BP_SAMPLES);
			return mk_inst(op, a, b, 0);
		}
	}
	return 0;
}

/*
 * x, a u32 of two words, as a lane-wise op of them (either order) that
 * the model has, the four bytes' rounded or truncated average, or a
 * saturating add or subtract: its value at random points told against
 * each, then checked with the function
 */
static nid
lanewise_of(const struct func *f, nid x, struct sbuf *why)
{
	static const int ops[] = {O_ADD4X8, O_SUB4X8, O_ABSDIFF4X8, O_MIN4X8,
	    O_MAX4X8, O_MSAD4X8, O_ADD2X16, O_SUB2X16, O_MUL2X16, O_MIN2X16,
	    O_MAX2X16, O_SMIN2X16, O_SMAX2X16};
	const struct model *m = model_active();
	uint32_t w0[BP_SAMPLES], w1[BP_SAMPLES], r[BP_SAMPLES];
	int slot[2] = {-1, -1}, ns, k, o, smp;

	if (!word_samples(f, x, slot, &ns, w0, w1, r) || ns != 2)
		return 0;
	for (k = 0; k < (int)(sizeof ops / sizeof ops[0]); k++) {
		if (!cclass_available(m, inst_class(ops[k])))
			continue;
		for (o = 0; o < 2; o++) {
			nid a, b;
			for (smp = 0; smp < BP_SAMPLES; smp++)
				if (r[smp] != lanewise_c(ops[k], o ? w1[smp] : w0[smp],
				    o ? w0[smp] : w1[smp]))
					break;
			if (smp < BP_SAMPLES)
				continue;
			a = mk_var((uint32_t)slot[o], f->p[slot[o]].name, U32);
			b = mk_var((uint32_t)slot[1 - o], f->p[slot[1 - o]].name, U32);
			sb_printf(why, "%s: %s(%s, %s%s), lane by lane, told at %d "
			    "random points", shared_str(x, 80), opinfo[ops[k]].name,
			    node_str(a), node_str(b), ops[k] == O_MSAD4X8 ? ", 0" : "",
			    BP_SAMPLES);
			return mk_inst(ops[k], a, b, ops[k] == O_MSAD4X8 ?
			    ku(U32, 0) : 0);
		}
	}
	/* lerp4x8(a, b, 0x01010101) rounds the average up, 0 truncates */
	for (k = 0; k < 2 && cclass_available(m, C_LERP8); k++) {
		uint32_t c = k ? 0 : 0x01010101u;
		for (smp = 0; smp < BP_SAMPLES; smp++)
			if (r[smp] != lerp_c(w0[smp], w1[smp], c))
				break;
		if (smp == BP_SAMPLES) {
			nid a = mk_var((uint32_t)slot[0], f->p[slot[0]].name, U32);
			nid b = mk_var((uint32_t)slot[1], f->p[slot[1]].name, U32);
			sb_printf(why, "%s: the four bytes' averages, %s, told at %d "
			    "random points", shared_str(x, 80), k ? "rounded down" :
			    "a half rounded up", BP_SAMPLES);
			return mk_inst(O_LERP4X8, a, b, ku(U32, c));
		}
	}
	for (k = 0; k < 3 && cclass_available(m, C_ADDSAT); k++) {
		int op = k ? O_SUBSAT : O_ADDSAT;
		o = k == 2;
		for (smp = 0; smp < BP_SAMPLES; smp++)
			if (r[smp] != sat_c(op, o ? w1[smp] : w0[smp], o ? w0[smp] :
			    w1[smp]))
				break;
		if (smp == BP_SAMPLES) {
			nid a = mk_var((uint32_t)slot[o], f->p[slot[o]].name, U32);
			nid b = mk_var((uint32_t)slot[1 - o], f->p[slot[1 - o]].name,
			    U32);
			sb_printf(why, "%s: %s(%s, %s), %s clamped to the range of "
			    "u32, told at %d random points", shared_str(x, 80),
			    opinfo[op].name, node_str(a), node_str(b), op == O_ADDSAT ?
			    "the sum" : "the difference", BP_SAMPLES);
			return mk_inst(op, a, b, 0);
		}
	}
	return 0;
}

/*
 * x, a min or max with a min or max of the same type for an operand:
 * min3, max3, min_max (a max of a min), max_min (a min of a max), where
 * the model has vmnmx
 */
static nid
minmax3_of(nid x, struct sbuf *why)
{
	int op = N.op[x], j;

	if ((op != O_MIN && op != O_MAX) || !TY_ISINT(N.ty[x]) ||
	    N.ty[x] == TY_BOOL || !cclass_available(model_active(), C_VMNMX))
		return 0;
	for (j = 0; j < 2; j++) {
		nid in = j ? N.b[x] : N.a[x], out = j ? N.a[x] : N.b[x];
		int r;
		if ((N.op[in] != O_MIN && N.op[in] != O_MAX) || N.ty[in] !=
		    N.ty[x])
			continue;
		r = N.op[in] == op ? (op == O_MIN ? O_MIN3 : O_MAX3) :
		    op == O_MAX ? O_MINMAX : O_MAXMIN;
		sb_printf(why, "%s: %s of %s and %s", shared_str(x, 80),
		    op == O_MIN ? "the min" : "the max", node_str(in),
		    node_str(out));
		return mk_inst(r, N.a[in], N.b[in], out);
	}
	return 0;
}

/* x as |a - b| for words: max(a, b) - min(a, b), or a select of a - b
 * and b - a on a compare of a and b; *a, *b */
static int
absdiff32(nid x, nid *a, nid *b)
{
	nid p, q, c;

	if (N.ty[x] != U32)
		return 0;
	if (N.op[x] == O_SUB && N.op[N.a[x]] == O_MAX && N.op[N.b[x]] ==
	    O_MIN) {
		p = N.a[x];
		q = N.b[x];
		if ((N.a[p] == N.a[q] && N.b[p] == N.b[q]) || (N.a[p] == N.b[q] &&
		    N.b[p] == N.a[q])) {
			*a = N.a[p];
			*b = N.b[p];
			return 1;
		}
		return 0;
	}
	if (N.op[x] != O_SEL || N.op[N.b[x]] != O_SUB || N.op[N.c[x]] !=
	    O_SUB)
		return 0;
	c = N.a[x];
	p = N.b[x];	/* the value where the compare holds */
	q = N.c[x];
	if (N.a[p] != N.b[q] || N.b[p] != N.a[q])
		return 0;
	/* p = u - v is the right one when u >= v: u > v, u >= v, v < u,
	 * v <= u (at u == v both are 0) */
	if (((N.op[c] == O_GT || N.op[c] == O_GE) && N.a[c] == N.a[p] &&
	    N.b[c] == N.b[p]) || ((N.op[c] == O_LT || N.op[c] == O_LE) &&
	    N.a[c] == N.b[p] && N.b[c] == N.a[p])) {
		*a = N.a[p];
		*b = N.b[p];
		return 1;
	}
	return 0;
}

/*
 * x, a sum with a min or max of the same type, or a word's absolute
 * difference, for a term: min_add, max_add (vmnmx), sad32 (sad32); an
 * absolute difference alone is sad32(a, b, 0)
 */
static nid
vidadd_of(nid x, struct sbuf *why)
{
	const struct model *m = model_active();
	nid a, b;
	int j;

	if (cclass_available(m, C_SAD32) && absdiff32(x, &a, &b)) {
		sb_printf(why, "%s: |%s - %s|", shared_str(x, 80), node_str(a),
		    node_str(b));
		return mk_inst(O_SAD, a, b, ku(U32, 0));
	}
	if (N.op[x] != O_ADD || !TY_ISINT(N.ty[x]) || N.ty[x] == TY_BOOL)
		return 0;
	for (j = 0; j < 2; j++) {
		nid in = j ? N.b[x] : N.a[x], out = j ? N.a[x] : N.b[x];
		if (cclass_available(m, C_SAD32) && absdiff32(in, &a, &b)) {
			sb_printf(why, "%s: |%s - %s| + %s", shared_str(x, 80),
			    node_str(a), node_str(b), node_str(out));
			return mk_inst(O_SAD, a, b, out);
		}
		if ((N.op[in] == O_MIN || N.op[in] == O_MAX) && N.ty[in] ==
		    N.ty[x] && cclass_available(m, C_VMNMX)) {
			sb_printf(why, "%s: %s plus %s", shared_str(x, 80),
			    node_str(in), node_str(out));
			return mk_inst(N.op[in] == O_MIN ? O_MINADD : O_MAXADD,
			    N.a[in], N.b[in], out);
		}
	}
	return 0;
}

/* x as byteperm(a, b, s) */
static nid
byteperm_of(const struct func *f, nid x, struct sbuf *why)
{
	int slot[2] = {-1, -1}, ns, j, i, nib[4];
	uint16_t cand[4], bs[4];
	nid w[2];
	uint32_t sel = 0;

	if (!byte_choices(f, x, slot, &ns, cand, bs))
		return 0;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 16 && !((cand[i] >> j) & 1); j++)
			;
		if (j == 16)
			return 0;
		nib[i] = j;
		sel |= (uint32_t)j << (4 * i);
	}
	/* one word moving its whole bytes is the GF(2) pattern's */
	if (ns == 1 && nib[0] < 4 && nib[1] < 4 && nib[2] < 4 && nib[3] < 4)
		return 0;
	w[0] = mk_var((uint32_t)slot[0], f->p[slot[0]].name, U32);
	w[1] = ns == 2 ? mk_var((uint32_t)slot[1], f->p[slot[1]].name, U32) :
	    ku(U32, 0);
	sb_printf(why, "%s: bytes %x, %x, %x, %x of %s:%s (8 and up: a byte's "
	    "sign), told at %d random points", shared_str(x, 80), nib[0],
	    nib[1], nib[2], nib[3], node_str(w[1]), node_str(w[0]),
	    BP_SAMPLES);
	return mk_inst(O_BYTEPERM, w[0], w[1], ku(U32, sel));
}

/* x as bytesel(hi, lo, s): whole bytes of one word moved too, which only
 * byteperm's GF(2) pattern offers otherwise */
static nid
bytesel_of(const struct func *f, nid x, struct sbuf *why)
{
	int slot[2] = {-1, -1}, ns, j, i, by[4];
	uint16_t bp[4], cand[4];
	nid lo, hi;
	uint32_t sel = 0;

	if (!byte_choices(f, x, slot, &ns, bp, cand))
		return 0;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 14 && !((cand[i] >> j) & 1); j++)
			;
		if (j == 14)
			return 0;
		by[i] = j;
		sel |= (uint32_t)j << (8 * i);
	}
	if (ns == 1 && sel == 0x03020100u)
		return 0;	/* the word itself */
	lo = mk_var((uint32_t)slot[0], f->p[slot[0]].name, U32);
	hi = ns == 2 ? mk_var((uint32_t)slot[1], f->p[slot[1]].name, U32) : lo;
	sb_printf(why, "%s: selectors %d, %d, %d, %d over %s:%s (8 to 11: the "
	    "sign of byte 1, 3, 5, 7; 12: 0; 13: 255), told at %d random "
	    "points", shared_str(x, 80), by[0], by[1], by[2], by[3],
	    node_str(hi), node_str(lo), BP_SAMPLES);
	return mk_inst(O_BYTESEL, hi, lo, ku(U32, sel));
}

/* x, a word of a 64-bit shift of hi:lo: u32((hi:lo) >> k) is funr,
 * u32(((hi:lo) << k) >> 32) funl, k in 1..31 */
static nid
funnel64_of(nid x, struct sbuf *why)
{
	nid s, w, hi, lo, sh;
	int k, j, left = 0;

	if (N.op[x] != O_CAST || N.ty[x] != U32)
		return 0;
	s = N.a[x];
	if (N.op[s] != O_SHR || N.ty[s] != U64 || !is_const(N.b[s]))
		return 0;
	k = (int)N.ki[N.b[s]];
	w = N.a[s];
	if (k == 32 && N.op[w] == O_SHL && N.ty[w] == U64 &&
	    is_const(N.b[w]) && N.ki[N.b[w]] >= 1 && N.ki[N.b[w]] <= 31) {
		left = 1;
		k = (int)N.ki[N.b[w]];
		w = N.a[w];
	} else if (k < 1 || k > 31)
		return 0;
	if (N.op[w] != O_OR && N.op[w] != O_ADD && N.op[w] != O_XOR)
		return 0;
	for (j = 0; j < 2; j++) {
		sh = j ? N.b[w] : N.a[w];
		lo = j ? N.a[w] : N.b[w];
		if (N.op[sh] != O_SHL || !is_const(N.b[sh]) ||
		    N.ki[N.b[sh]] != 32)
			continue;
		hi = N.a[sh];
		if (N.op[hi] != O_CAST || N.op[lo] != O_CAST ||
		    TYK(N.ty[N.a[hi]]) != K_UINT || TYW(N.ty[N.a[hi]]) > 32 ||
		    TYK(N.ty[N.a[lo]]) != K_UINT || TYW(N.ty[N.a[lo]]) > 32)
			continue;
		hi = mk_cast(U32, N.a[hi]);
		lo = mk_cast(U32, N.a[lo]);
		sb_printf(why, "%s: the %s word of the 64-bit (%s:%s) %s %d",
		    shared_str(x, 80), left ? "high" : "low", node_str(hi),
		    node_str(lo), left ? "<<" : ">>", k);
		return mk_inst(left ? O_FUNL : O_FUNR, hi, lo, kint(k));
	}
	return 0;
}

/* x as the instruction of pattern p; 0 when it is not that pattern */
static nid
pattern(const struct func *f, int p, nid x, struct sbuf *why)
{
	nid a, b, s, m;
	int k, j, op = N.op[x];
	int64_t lo, hi, lo2, hi2;

	switch (p) {
	case 0:		/* (w >> k) & (2^n - 1); (w << j) >> k */
		if (op == O_AND && N.ty[x] == U32) {
			m = is_const(N.b[x]) ? N.b[x] : N.a[x];
			s = m == N.b[x] ? N.a[x] : N.b[x];
			if (is_const(m) && (j = klog2(N.ki[m])) > 0 &&
			    cshift(s, O_SHR, U32, &a, &k) && k + j <= 32 && k) {
				sb_printf(why, "%s: bits %d..%d of %s", shared_str(x, 80), k,
				    k + j - 1, shared_str(a, 40));
				return mk_inst(O_BFE, a, kint(k), kint(j));
			}
		}
		if (cshift(x, O_SHR, U32, &s, &k) && cshift(s, O_SHL, U32, &a,
		    &j) && k >= j && j) {
			sb_printf(why, "%s: bits %d..%d of %s", shared_str(x, 80), k - j,
			    31 - j, node_str(a));
			return mk_inst(O_BFE, a, kint(k - j), kint(32 - k));
		}
		break;
	case 1:		/* i32: (w << j) >> k keeps the field's sign */
		if (cshift(x, O_SHR, I32, &s, &k) && cshift(s, O_SHL, I32, &a,
		    &j) && k >= j && j) {
			sb_printf(why, "%s: bits %d..%d of %s with their sign",
			    shared_str(x, 80), k - j, 31 - j, node_str(a));
			return mk_inst(O_SBFE, a, kint(k - j), kint(32 - k));
		}
		break;
	case 2:		/* (a << k) | (b >> (32 - k)); or, xor or + alike */
		if ((op == O_OR || op == O_XOR || op == O_ADD) && N.ty[x] == U32)
			for (j = 0; j < 2; j++) {
				nid l = j ? N.b[x] : N.a[x], r = j ? N.a[x] : N.b[x];
				int kl, kr;
				if (cshift(l, O_SHL, U32, &a, &kl) && cshift(r, O_SHR,
				    U32, &b, &kr) && kl + kr == 32 && kl) {
					sb_printf(why, "%s: the high word of (%s:%s) "
					    "<< %d (the two fields share no bit)",
					    shared_str(x, 80), node_str(a), node_str(b), kl);
					return mk_inst(O_FUNL, a, b, kint(kl));
				}
			}
		break;
	case 3:		/* u32((u64(a) * u64(b)) >> 32), and the signed one */
		if (op == O_CAST && (N.ty[x] == U32 || N.ty[x] == I32)) {
			ty_t wide = N.ty[x] == U32 ? U64 : I64;
			s = N.a[x];
			if (N.op[s] != O_SHR || N.ty[s] != wide || !is_const(N.b[s]) ||
			    N.ki[N.b[s]] < 32 || N.ki[N.b[s]] > 63 ||
			    N.op[N.a[s]] != O_MUL)
				break;
			k = (int)N.ki[N.b[s]] - 32;
			a = N.a[N.a[s]];
			b = N.b[N.a[s]];
			for (j = 0; j < 2; j++) {
				nid *q = j ? &b : &a;
				if (is_const(*q) && (wide == U64 ? (uint64_t)N.ki[*q] <=
				    0xffffffffULL : N.ki[*q] >= INT32_MIN &&
				    N.ki[*q] <= INT32_MAX))
					*q = mk_const_i(N.ty[x], N.ki[*q]);
				else if (N.op[*q] == O_CAST && N.ty[N.a[*q]] == N.ty[x])
					*q = N.a[*q];
				else
					return 0;
			}
			sb_printf(why, "%s: the high half of the 64-bit product%s",
			    shared_str(x, 80), k ? ", shifted by what is left" : "");
			a = mk_inst(wide == U64 ? O_MULHI : O_SMULHI, a, b, 0);
			return k ? mk_bin(O_SHR, a, mk_const_i(N.ty[x], k)) : a;
		}
		break;
	case 4:		/* max(lo, min(v, hi)), min(hi, max(v, lo)), lo <= hi */
		if ((op == O_MAX || op == O_MIN) && TY_ISINT(N.ty[x]) &&
		    N.ty[x] != TY_BOOL)
			for (j = 0; j < 2; j++) {
				nid in = j ? N.b[x] : N.a[x], out = j ? N.a[x] : N.b[x];
				int in_op = op == O_MAX ? O_MIN : O_MAX, i;
				if (N.op[in] != in_op)
					continue;
				for (i = 0; i < 2; i++) {
					nid v = i ? N.b[in] : N.a[in];
					nid o2 = i ? N.a[in] : N.b[in];
					nid l = op == O_MAX ? out : o2;
					nid h = op == O_MAX ? o2 : out;
					if (!sg_range_of(f, l, 32, &lo, &hi) ||
					    !sg_range_of(f, h, 32, &lo2, &hi2) ||
					    hi > lo2)
						continue;
					sb_printf(why, "%s: %s clamped to %s..%s, which "
					    "is their median since %s <= %s (%lld <= "
					    "%lld, by intervals)", shared_str(x, 80),
					    node_str(v), node_str(l), node_str(h),
					    node_str(l), node_str(h), (long long)hi,
					    (long long)lo2);
					return mk_inst(O_MED3, v, l, h);
				}
			}
		break;
	case 5: {	/* a permutation of one word's bits */
		uint64_t slots = node_slots(x);
		struct aff *m2;
		int slot = -1, i, rev = 1, rot, perm[4], byp = 1;
		nid r = 0, w;

		if (N.ty[x] != U32 || op == O_VAR || IS_INST(op) ||
		    (slots >> MAXP))
			break;
		for (j = 0; j < f->np; j++)
			if ((slots >> j) & 1) {
				if (slot >= 0)
					return 0;
				slot = j;
			}
		if (slot < 0 || f->p[slot].ty != U32)
			break;
		m2 = xmalloc(sizeof *m2);
		if (!gf2_affine(x, slot, m2, 0) || m2->k || m2->w != 32) {
			free(m2);
			break;
		}
		w = mk_var((uint32_t)slot, f->p[slot].name, U32);
		for (i = 0; i < 32; i++)
			rev &= m2->row[i] == 1ULL << (31 - i);
		for (rot = 1; rot < 32; rot++) {
			for (i = 0; i < 32 && m2->row[i] == 1ULL << ((i - rot) & 31);
			    i++)
				;
			if (i == 32)
				break;
		}
		for (i = 0; i < 32 && byp; i++) {
			uint64_t rw = m2->row[i];
			int src = 0;
			if (!rw || (rw & (rw - 1))) {
				byp = 0;
				break;
			}
			while (!(rw & 1)) {
				rw >>= 1;
				src++;
			}
			if ((src & 7) != (i & 7) || ((i & 7) && perm[i >> 3] !=
			    src >> 3))
				byp = 0;
			perm[i >> 3] = src >> 3;
		}
		if (byp && perm[0] == 0 && perm[1] == 1 && perm[2] == 2 &&
		    perm[3] == 3)
			byp = 0;
		if (rev) {
			sb_printf(why, "%s: bit i is bit 31 - i of %s (linear over "
			    "GF(2) by its structure, every row one bit)",
			    shared_str(x, 80), node_str(w));
			r = mk_inst(O_BREV, w, 0, 0);
		} else if (rot < 32) {
			sb_printf(why, "%s: %s rotated left by %d (linear over "
			    "GF(2) by its structure, every row one bit): the "
			    "high word of (%s:%s) << %d", shared_str(x, 80),
			    node_str(w), rot, node_str(w), node_str(w), rot);
			r = mk_inst(O_FUNL, w, w, kint(rot));
		} else if (byp) {
			sb_printf(why, "%s: the bytes %d, %d, %d, %d of %s (linear "
			    "over GF(2) by its structure, every row one bit, "
			    "whole bytes moved)", shared_str(x, 80), perm[0], perm[1],
			    perm[2], perm[3], node_str(w));
			r = mk_inst(O_BYTEPERM, w, w, kint(perm[0] | perm[1] << 4 |
			    perm[2] << 8 | perm[3] << 12));
		}
		free(m2);
		return r;
	}
	case 6:		/* (base & ~m) | field, m the field's bits p..p+n-1 */
		return bfi_of(f, x, why);
	case 7:		/* u64(a) * u64(b), a and b words */
		if (op == O_MUL && N.ty[x] == U64) {
			struct sbuf how = {0};
			if (wide_product(f, x, &a, &b, &how)) {
				sb_printf(why, "%s: the 64-bit product of two "
				    "words%s", shared_str(x, 80), sb_str(&how));
				free(how.p);
				return mk_inst(O_MULWIDE, a, b, 0);
			}
			free(how.p);
		}
		break;
	case 8:		/* that product plus a u64 */
		return madwide_of(f, x, why);
	case 9:		/* the four bytes' |a - b| summed */
		return usad_of(x, why);
	case 10:	/* bytes of two words, signs of bytes */
		return byteperm_of(f, x, why);
	case 11:	/* a word of a 64-bit shift of two words */
		return funnel64_of(x, why);
	case 12:	/* bytes, signs of bytes 1, 3, 5, 7, 0 and 255 */
		return bytesel_of(f, x, why);
	case 13:	/* two words' bytes or halves lane by lane; i32 saturation */
		return N.ty[x] == I32 ? ssat_of(f, x, why) : lanewise_of(f, x,
		    why);
	case 14:	/* a min or max of a min or max */
		return minmax3_of(x, why);
	case 15:	/* a min, max or |a - b| plus a third */
		return vidadd_of(x, why);
	}
	return 0;
}

#define NPAT 16

/*
 * The low word of a product, a * b in u32, where the function also reads
 * its high word (mulhi(a, b), or the pattern it is): both from one wide
 * product.  The candidate, or 0.
 */
static nid
wide_pair(const struct func *f, nid cur, const nid *nd, int n, nid x,
    struct sbuf *why)
{
	nid w, hi, cand;
	int i;
	struct sbuf scratch = {0};

	if (N.op[x] != O_MUL || N.ty[x] != U32)
		return 0;
	for (i = 0; i < n; i++) {
		hi = nd[i];
		if (N.op[hi] != O_MULHI) {
			nid h = pattern(f, 3, hi, &scratch);
			sb_clear(&scratch);
			if (!h || N.op[h] != O_MULHI)
				continue;
			if (!((N.a[h] == N.a[x] && N.b[h] == N.b[x]) ||
			    (N.a[h] == N.b[x] && N.b[h] == N.a[x])))
				continue;
		} else if (!((N.a[hi] == N.a[x] && N.b[hi] == N.b[x]) ||
		    (N.a[hi] == N.b[x] && N.b[hi] == N.a[x])))
			continue;
		w = mk_inst(O_MULWIDE, N.a[x], N.b[x], 0);
		cand = sg_replaced(cur, x, mk_cast(U32, w));
		if (cand)
			cand = sg_replaced(cand, hi, mk_cast(N.ty[hi],
			    mk_bin(O_SHR, w, ku(U64, 32))));
		if (!cand)
			continue;
		sb_printf(why, "%s and %s: the low and the high word of one "
		    "64-bit product", shared_str(x, 60), shared_str(hi, 60));
		free(scratch.p);
		return cand;
	}
	free(scratch.p);
	return 0;
}
#define MAXNODE 512

static int
all_nodes(nid e, nid *out, int n, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || n == MAXNODE)
		return n;
	seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_UNDEF)
		return n;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			n = all_nodes(N.lst[N.a[e] + j], out, n, seen);
		return n;
	}
	out[n++] = e;
	n = all_nodes(N.a[e], out, n, seen);
	n = all_nodes(N.b[e], out, n, seen);
	return all_nodes(N.c[e], out, n, seen);
}

void
inst_find(const struct func *f)
{
	const struct model *m = model_active();
	const char *mn = sym_str(m->name), *s, *e, *chk;
	struct sbuf why = {0}, one = {0}, best_why = {0}, w = {0};
	struct eqproof pr;
	nid cur = f->body, *nd;
	float before, cost;
	int k, iter, bad;

	for (k = C_BFE; k <= C_ADDSAT && !cclass_available(m, k); k++)
		;
	if (k > C_ADDSAT || !TY_ISINT(N.ty[cur]))
		return;
	nd = xmalloc(MAXNODE * sizeof *nd);
	before = cost = fuse_expr_cost(m, sg_priced(f, cur));
	for (iter = 0; iter < 32; iter++) {
		uint8_t *seen = xcalloc(N.n, 1);
		int n = all_nodes(cur, nd, 0, seen), i, p;
		nid best = 0;
		float bc = cost;

		free(seen);
		for (i = 0; i < n; i++) {
			nid cand;
			float c;
			sb_clear(&one);
			cand = wide_pair(f, cur, nd, n, nd[i], &one);
			if (cand && (c = fuse_expr_cost(m, sg_priced(f, cand))) <
			    bc) {
				bc = c;
				best = cand;
				sb_clear(&best_why);
				sb_puts(&best_why, sb_str(&one));
			}
			for (p = 0; p < NPAT; p++) {
				nid to, cand;
				float c;
				sb_clear(&one);
				to = pattern(f, p, nd[i], &one);
				if (!to || to == nd[i])
					continue;
				cand = sg_replaced(cur, nd[i], mk_cast(N.ty[nd[i]], to));
				if (!cand)
					continue;
				c = fuse_expr_cost(m, sg_priced(f, cand));
				if (c < bc) {
					bc = c;
					best = cand;
					sb_clear(&best_why);
					sb_puts(&best_why, sb_str(&one));
				}
			}
		}
		if (!best)
			break;
		sb_printf(&why, "%s\n", sb_str(&best_why));
		cur = best;
		cost = bc;
	}
	free(nd);
	if (cur == f->body)
		goto out;
	bad = equiv_decide(f, f->body, cur, &w, &pr);
	chk = pr.how == EQ_EXHAUSTIVE ? "checked on every point of the domain" :
	    pr.how == EQ_SAT ? "checked by SAT" :
	    "checked on a sample of the domain";
	if (bad) {
		printf("  instructions under %s: %s (cost %g) does not hold: %s\n",
		    mn, shared_str(cur, 300), (double)cost, sb_str(&w));
		rep_obj("instructions");
		rep_str("model", mn);
		rep_num("cost_before", before);
		rep_expr("candidate", cur);
		rep_str("not_proved", sb_str(&w));
		rep_end();
		goto out;
	}
	printf("  instructions under %s (each for the pattern it is; cost %g as "
	    "written):\n", mn, (double)before);
	printf("  cost %-3g %s\n", (double)cost, shared_str(cur, 400));
	for (s = sb_str(&why); *s; s = *e ? e + 1 : e) {
		e = strchr(s, '\n');
		if (!e)
			e = s + strlen(s);
		printf("           %.*s\n", (int)(e - s), s);
	}
	printf("           %s\n", chk);
	rep_obj("instructions");
	rep_str("model", mn);
	rep_num("cost_before", before);
	rep_num("cost", cost);
	rep_expr("expr", cur);
	rep_str("derivation", sb_str(&why));
	rep_str("check", pr.how == EQ_EXHAUSTIVE ? "exhaustive" : pr.how ==
	    EQ_SAT ? "sat" : "sampled");
	rep_end();
out:
	free(why.p);
	free(one.p);
	free(best_why.p);
	free(w.p);
}
