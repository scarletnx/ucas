/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * astc.c: steps of the ASTC decoder (Khronos Data Format Specification,
 * "ASTC Compressed Texture Image Formats"), 2D blocks of the LDR
 * profile, as operations of the language.
 *
 * Each one is defined by an expression over the base integer operations
 * (astc_expand), like the catalogue's instructions (inst.c): that
 * expression is its exact semantics, and it is how the operation is
 * evaluated, bounded and bit-blasted, so equiv proves the step a decoder
 * writes against it.  A constant of the operation rides in the node's ki
 * (a node has three operands).  No target has one of them as an
 * instruction (the texture unit decodes a whole block), so an operation
 * is priced as its definition: the operations it spells out over its
 * operands, constants folded.
 *
 *   astc_partition(seed, x, y, n)
 *       the partition, 0..n-1, of texel (x, y) in a block of n
 *       partitions (n 1..4, n = 1 is 0; a count that is not a constant
 *       is a select of the four, undefined for any other) whose
 *       partition index is seed (its low 10 bits, the block's field): the
 *       specification's hash of the seed, its twelve squared nibbles
 *       shifted by the seed's bits, four scores (s1 x + s2 y + p >> 14)
 *       & 63 ... and the highest of the first n, the lowest partition
 *       on a tie.  z is 0 (2D); a block of fewer than 31 texels takes
 *       the hash at (2x, 2y).  x and y are any integers (u32 arithmetic,
 *       of which the scores keep the low 6 bits).
 *   astc_unquant(levels, v)
 *       a colour endpoint value v, 0..levels-1 as the integer sequence
 *       encodes it (a trit or quint times 2^m plus m bits), unquantised
 *       to 0..255: the bits repeated, or the trit or quint's formula with
 *       its bit-scrambled constants and the low bit as a mirror.  levels
 *       is a colour range's count, 6 to 256; anything else, and v past
 *       levels - 1, is undefined
 *   astc_unquantw(levels, v)
 *       a weight the same way, to 0..64 (the 0..63 result above 32 moved
 *       up one), levels a weight range's count, 2 to 32
 *   astc_endpoint(cem, v, k)
 *       byte k of an LDR colour endpoint mode's two endpoints (0..3 the
 *       first's R, G, B, A, 4..7 the second's) from its unquantised
 *       values, v0..v7 the bytes of the u64 v (v0 the low byte): the
 *       direct, base and offset (the offset's top bit moved into the
 *       base), base and scale forms, the blue contraction when the second
 *       endpoint's sum is the smaller.  The HDR modes (2, 3, 7, 11, 14,
 *       15) are the error colour, magenta (255, 0, 255, 255), as the LDR
 *       profile decodes them; cem past 15 and k past 7 are undefined
 *   astc_bilerp(G, gw, s, t)
 *       the weight grid's infill at (s, t), in sixteenths of the grid
 *       (the specification's gs and gt): G a table of weights read as
 *       rows of gw (gw a constant), its four entries around (s/16, t/16)
 *       weighed by w11 = (fs ft + 8) >> 4, w10 = ft - w11, w01 = fs -
 *       w11, w00 = 16 - fs - ft + w11 (fs = s & 15, ft = t & 15), plus 8,
 *       over 16.  An entry past the grid reads 0 (its weight is 0 where
 *       a block's texel lands).  A second plane is a second grid
 *   astc_lerp(e0, e1, w)
 *       the endpoints' channel at weight w, as UNORM16: e0 and e1 bytes
 *       widened to e * 257, (c0 (64 - w) + c1 w + 32) >> 6; w past 64 or
 *       a byte past 255 is undefined.  The UNORM8 decode mode is its top
 *       byte, >> 8
 *
 * The HDR profile (23.15, 23.19):
 *
 *   astc_hdr(cem, v, k)
 *       component k (0..3 the first endpoint's R, G, B, A, 4..7 the
 *       second's) of HDR endpoint mode cem (2, 3, 7, 11, 14, 15) from its
 *       unquantised values, v0..v7 the bytes of the u64 v: the
 *       specification's 12-bit pseudo-logarithmic values, mode 7's and
 *       11's bit layouts selected by their mode bits, 0x780 (1.0) for the
 *       alpha of modes 2, 3, 7 and 11, the bytes v6, v7 for mode 14's
 *       (an LDR alpha), mode 15's alphas by their own mode bits;
 *       anything else is undefined
 *   astc_lerph(e0, e1, w)
 *       two 12-bit values at weight w as the 16-bit C: ((e0 << 4) (64 -
 *       w) + (e1 << 4) w + 32) >> 6
 *   astc_lns(c)
 *       C to FP16 bits: (c >> 11 << 10) + (Mt >> 3), Mt the low 11 bits
 *       m times 3 below 512, 5 m - 2048 from 1536, 4 m - 512 between;
 *       Inf and NaN (0x7C00 and up) are 0x7BFF
 *   astc_f16(c)
 *       an LDR channel's UNORM16 to FP16 bits (decode_float16): 65535 is
 *       1.0 (0x3C00), any other c / 65536 rounded toward zero
 */
#include "calc.h"
#include "fuse.h"

#include <stdint.h>
#include <stdlib.h>

#define U32 TY(K_UINT, 32)

static nid
ku(uint32_t v)
{
	return mk_const_i(U32, (int64_t)v);
}

static nid
shr(nid x, int k)
{
	return mk_bin(O_SHR, x, ku((uint32_t)k));
}

static nid
shl(nid x, int k)
{
	return mk_bin(O_SHL, x, ku((uint32_t)k));
}

static nid
band(nid x, uint32_t m)
{
	return mk_bin(O_AND, x, ku(m));
}

static nid
bit_set(nid x, uint32_t m)
{
	return mk_bin(O_NE, band(x, m), ku(0));
}

/* the partition of (x, y) among n (2..4) for the 10-bit seed */
static nid
partition(int n, nid seed, nid x, nid y)
{
	nid p, sd[13], b0, b1, b4, v[4], ge01, ge02, ge12;
	int i, k3 = n == 3 ? 6 : 5;

	/* the hash of the seed and the count: seed + 1024 (n - 1) */
	p = mk_bin(O_ADD, band(seed, 1023), ku(1024u * (uint32_t)(n - 1)));
	p = mk_bin(O_XOR, p, shr(p, 15));
	p = mk_bin(O_SUB, p, shl(p, 17));
	p = mk_bin(O_ADD, p, shl(p, 7));
	p = mk_bin(O_ADD, p, shl(p, 4));
	p = mk_bin(O_XOR, p, shr(p, 5));
	p = mk_bin(O_ADD, p, shl(p, 16));
	p = mk_bin(O_XOR, p, shr(p, 7));
	p = mk_bin(O_XOR, p, shr(p, 3));
	p = mk_bin(O_XOR, p, shl(p, 6));
	p = mk_bin(O_XOR, p, shr(p, 17));
	for (i = 1; i <= 8; i++)
		sd[i] = band(shr(p, 4 * (i - 1)), 15);
	sd[9] = band(shr(p, 18), 15);
	sd[10] = band(shr(p, 22), 15);
	sd[11] = band(shr(p, 26), 15);
	sd[12] = band(mk_bin(O_OR, shr(p, 30), shl(p, 2)), 15);
	/* bits 0, 1 and 4 of the seed choose the shifts, which adding a
	 * multiple of 1024 leaves as they are */
	b0 = bit_set(seed, 1);
	b1 = bit_set(seed, 2);
	b4 = bit_set(seed, 16);
	for (i = 1; i <= 12; i++) {
		nid q = mk_bin(O_MUL, sd[i], sd[i]);
		nid b45 = mk_sel(b1, shr(q, 4), shr(q, 5));
		/* sh1: bit 0 ? (bit 1 ? 4 : 5) : (n == 3 ? 6 : 5); sh2 the
		 * other way round; the last four by sh1 when bit 4 is set: a
		 * select of constant shifts, each in range */
		nid s1 = mk_sel(b0, b45, shr(q, k3));
		nid s2 = mk_sel(b0, shr(q, k3), b45);
		sd[i] = i <= 8 ? (i & 1 ? s1 : s2) : mk_sel(b4, s1, s2);
	}
	for (i = 0; i < 4; i++) {
		nid sx = mk_bin(O_MUL, sd[2 * i + 1], x);
		nid sy = mk_bin(O_MUL, sd[2 * i + 2], y);
		v[i] = band(mk_bin(O_ADD, mk_bin(O_ADD, sx, sy), shr(p,
		    14 - 4 * i)), 63);
	}
	ge01 = mk_bin(O_GE, v[0], v[1]);
	if (n == 2)
		return mk_sel(ge01, ku(0), ku(1));
	ge02 = mk_bin(O_GE, v[0], v[2]);
	ge12 = mk_bin(O_GE, v[1], v[2]);
	if (n == 3)
		return mk_sel(mk_bin(O_LAND, ge01, ge02), ku(0), mk_sel(ge12,
		    ku(1), ku(2)));
	return mk_sel(mk_bin(O_LAND, mk_bin(O_LAND, ge01, ge02), mk_bin(O_GE,
	    v[0], v[3])), ku(0), mk_sel(mk_bin(O_LAND, ge12, mk_bin(O_GE, v[1],
	    v[3])), ku(1), mk_sel(mk_bin(O_GE, v[2], v[3]), ku(2), ku(3))));
}

/* ---- unquantisation ---- */

/*
 * A level count as the integer sequence codes it: 1 (bits only), 3 (a
 * trit) or 5 (a quint), and its m low bits; 0 when it is not one
 */
static int
level_kind(int64_t L, int *m)
{
	int k;

	for (k = 0; k <= 8; k++) {
		if (L == (int64_t)1 << k) {
			*m = k;
			return 1;
		}
		if (L == (int64_t)3 << k) {
			*m = k;
			return 3;
		}
		if (L == (int64_t)5 << k) {
			*m = k;
			return 5;
		}
	}
	return 0;
}

/* a count of colour (w = 0) or weight (w = 1) values */
static int
level_ok(int64_t L, int w)
{
	int m = 0, q = level_kind(L, &m);

	if (w)
		return (q == 1 && m >= 1 && m <= 5) || (q == 3 && m <= 3) ||
		    (q == 5 && m <= 2);
	return (q == 1 && m >= 3 && m <= 8) || (q == 3 && m >= 1 && m <= 6) ||
	    (q == 5 && m >= 1 && m <= 5);
}

static nid
bit(nid v, int i)
{
	return band(shr(v, i), 1);
}

/* v's m bits repeated from the top down to `to` bits */
static nid
replicate(nid v, int m, int to)
{
	nid r = ku(0);
	int n = 0;

	while (n < to) {
		r = mk_bin(O_OR, shl(r, m), v);
		n += m;
	}
	return band(shr(r, n - to), (1u << to) - 1);
}

/*
 * The scrambled constant B of a trit or quint value with m bits, b of
 * them: sh[j] the shift of bit j + 1 (b, c, d, e, f), or -1, repeated as
 * the table lists them
 */
static nid
scramble(nid v, const int (*sh)[3], int nb)
{
	nid r = ku(0);
	int j, k;

	for (j = 0; j < nb; j++)
		for (k = 0; k < 3; k++)
			if (sh[j][k] >= 0)
				r = mk_bin(O_OR, r, shl(bit(v, j + 1), sh[j][k]));
	return r;
}

/* a colour value of a count's kind q with m bits, to 0..255 */
static nid
color_unq(int q, int m, nid v)
{
	/* per m: C, then the shifts of b, c, d, e, f into B */
	static const int tc[7] = {0, 204, 93, 44, 22, 11, 5};
	static const int tb[7][5][3] = {
		{{-1}}, {{-1, -1, -1}},
		{{8, 4, 2}, {-1, -1, -1}},		/* and b << 1 below */
		{{7, 2, 0}, {8, 3, 1}},
		{{6, 0, -1}, {7, 1, -1}, {8, 2, -1}},
		{{5, -1, -1}, {6, -1, -1}, {7, 0, -1}, {8, 1, -1}},
		{{4, -1, -1}, {5, -1, -1}, {6, -1, -1}, {7, -1, -1},
		    {8, 0, -1}},
	};
	static const int qc[6] = {0, 113, 54, 26, 13, 6};
	static const int qb[6][4][3] = {
		{{-1}}, {{-1, -1, -1}},
		{{8, 3, 2}},
		{{7, 1, -1}, {8, 2, 0}},
		{{6, -1, -1}, {7, 0, -1}, {8, 1, -1}},
		{{5, -1, -1}, {6, -1, -1}, {7, -1, -1}, {8, 0, -1}},
	};
	nid A, B, T;

	if (q == 1)
		return replicate(v, m, 8);
	A = mk_bin(O_MUL, bit(v, 0), ku(0x1ff));
	if (q == 3) {
		B = scramble(v, tb[m], m - 1);
		if (m == 2)
			B = mk_bin(O_OR, B, shl(bit(v, 1), 1));
	} else
		B = scramble(v, qb[m], m - 1);
	T = mk_bin(O_ADD, mk_bin(O_MUL, shr(v, m), ku((uint32_t)(q == 3 ?
	    tc[m] : qc[m]))), B);
	T = mk_bin(O_XOR, T, A);
	return mk_bin(O_OR, band(A, 0x80), shr(T, 2));
}

/* a weight of a count's kind q with m bits, to 0..64 */
static nid
weight_unq(int q, int m, nid v)
{
	static const int tc[4] = {0, 50, 23, 11};
	static const int tb[4][2][3] = {
		{{-1}}, {{-1, -1, -1}},
		{{6, 2, 0}},
		{{5, 0, -1}, {6, 1, -1}},
	};
	static const int qc[3] = {0, 28, 13};
	static const int qb[3][1][3] = {
		{{-1}}, {{-1, -1, -1}},
		{{6, 1, -1}},
	};
	nid A, B, T;

	if (q == 1)
		T = replicate(v, m, 6);
	else if (m == 0)
		/* a trit or quint alone: 0, 32, 63 and 0, 16, 32, 47, 63 */
		T = q == 3 ? mk_sel(mk_bin(O_EQ, v, ku(0)), ku(0),
		    mk_sel(mk_bin(O_EQ, v, ku(1)), ku(32), ku(63))) :
		    mk_sel(mk_bin(O_EQ, v, ku(0)), ku(0), mk_sel(mk_bin(O_EQ,
		    v, ku(1)), ku(16), mk_sel(mk_bin(O_EQ, v, ku(2)), ku(32),
		    mk_sel(mk_bin(O_EQ, v, ku(3)), ku(47), ku(63)))));
	else {
		A = mk_bin(O_MUL, bit(v, 0), ku(0x7f));
		B = q == 3 ? scramble(v, tb[m], m - 1) : scramble(v, qb[m],
		    m - 1);
		T = mk_bin(O_ADD, mk_bin(O_MUL, shr(v, m), ku((uint32_t)(q == 3 ?
		    tc[m] : qc[m]))), B);
		T = mk_bin(O_XOR, T, A);
		T = mk_bin(O_OR, band(A, 0x20), shr(T, 2));
	}
	return mk_sel(mk_bin(O_GT, T, ku(32)), mk_bin(O_ADD, T, ku(1)), T);
}

/* v at one level count L, undefined past it */
static nid
unquant_at(int w, int64_t L, nid v)
{
	int m = 0, q = level_kind(L, &m);
	nid r = w ? weight_unq(q, m, v) : color_unq(q, m, v);

	return mk_sel(mk_bin(O_LT, v, ku((uint32_t)L)), r, mk_undef(U32));
}

/* levels a value: that count; otherwise a select of every count */
static nid
unquant(int w, nid levels, nid v)
{
	nid r = mk_undef(U32);
	int64_t L;

	if (is_const(levels))
		return level_ok(N.ki[levels], w) ? unquant_at(w, N.ki[levels], v) :
		    mk_undef(U32);
	for (L = 256; L >= 2; L--)
		if (level_ok(L, w))
			r = mk_sel(mk_bin(O_EQ, levels, ku((uint32_t)L)),
			    unquant_at(w, L, v), r);
	return r;
}

/* ---- the endpoint modes ---- */

#define I32 TY(K_SINT, 32)

static nid
ks(int32_t v)
{
	return mk_const_i(I32, v);
}

static nid
iadd(nid a, nid b)
{
	return mk_bin(O_ADD, a, b);
}

static nid
clamp8(nid x)
{
	return mk_bin(O_MIN, mk_bin(O_MAX, x, ks(0)), ks(255));
}

/*
 * bit_transfer_signed: the offset a gives its top bit to the base b and
 * keeps six bits, read as -32..31
 */
static void
transfer(nid *a, nid *b)
{
	nid t = mk_bin(O_AND, mk_bin(O_SHR, *a, ks(1)), ks(0x3f));

	*b = mk_bin(O_OR, mk_bin(O_SHR, *b, ks(1)), mk_bin(O_AND, *a, ks(0x80)));
	*a = mk_bin(O_SUB, mk_bin(O_XOR, t, ks(0x20)), ks(0x20));
}

/* blue_contract: red and green halfway to blue */
static void
contract(nid *e)
{
	e[0] = mk_bin(O_SHR, iadd(e[0], e[2]), ks(1));
	e[1] = mk_bin(O_SHR, iadd(e[1], e[2]), ks(1));
}

static void
set4(nid *e, nid r, nid g, nid b, nid a)
{
	e[0] = r;
	e[1] = g;
	e[2] = b;
	e[3] = a;
}

/* e[0..7]: the two endpoints of mode cem from the bytes of v (i32) */
static void
endpoints(int cem, nid v64, nid *e)
{
	nid v[8], c[8], t, swap;
	int i, alpha = cem == 12 || cem == 13;

	for (i = 0; i < 8; i++)
		v[i] = mk_cast(I32, mk_bin(O_AND, mk_bin(O_SHR, v64, mk_const_i(
		    TY(K_UINT, 64), 8 * i)), mk_const_i(TY(K_UINT, 64), 255)));
	switch (cem) {
	case 0:
		set4(e, v[0], v[0], v[0], ks(255));
		set4(e + 4, v[1], v[1], v[1], ks(255));
		return;
	case 1:
		t = mk_bin(O_OR, mk_bin(O_SHR, v[0], ks(2)), mk_bin(O_AND, v[1],
		    ks(0xc0)));
		set4(e, t, t, t, ks(255));
		t = mk_bin(O_MIN, iadd(t, mk_bin(O_AND, v[1], ks(0x3f))), ks(255));
		set4(e + 4, t, t, t, ks(255));
		return;
	case 4:
		set4(e, v[0], v[0], v[0], v[2]);
		set4(e + 4, v[1], v[1], v[1], v[3]);
		return;
	case 5:
		transfer(&v[1], &v[0]);
		transfer(&v[3], &v[2]);
		set4(e, v[0], v[0], v[0], v[2]);
		t = clamp8(iadd(v[0], v[1]));
		set4(e + 4, t, t, t, clamp8(iadd(v[2], v[3])));
		return;
	case 6:
	case 10:
		for (i = 0; i < 3; i++)
			e[i] = mk_bin(O_SHR, mk_bin(O_MUL, v[i], v[3]), ks(8));
		e[3] = cem == 10 ? v[4] : ks(255);
		set4(e + 4, v[0], v[1], v[2], cem == 10 ? v[5] : ks(255));
		return;
	case 8:
	case 12:
		swap = mk_bin(O_LT, iadd(iadd(v[1], v[3]), v[5]), iadd(iadd(v[0],
		    v[2]), v[4]));
		set4(c, v[1], v[3], v[5], alpha ? v[7] : ks(255));
		set4(c + 4, v[0], v[2], v[4], alpha ? v[6] : ks(255));
		contract(c);
		contract(c + 4);
		set4(e, v[0], v[2], v[4], alpha ? v[6] : ks(255));
		set4(e + 4, v[1], v[3], v[5], alpha ? v[7] : ks(255));
		for (i = 0; i < 8; i++)
			e[i] = mk_sel(swap, c[i], e[i]);
		return;
	case 9:
	case 13:
		transfer(&v[1], &v[0]);
		transfer(&v[3], &v[2]);
		transfer(&v[5], &v[4]);
		if (alpha)
			transfer(&v[7], &v[6]);
		else {
			v[6] = ks(255);
			v[7] = ks(0);
		}
		swap = mk_bin(O_LT, iadd(iadd(v[1], v[3]), v[5]), ks(0));
		set4(c, iadd(v[0], v[1]), iadd(v[2], v[3]), iadd(v[4], v[5]),
		    iadd(v[6], v[7]));
		set4(c + 4, v[0], v[2], v[4], v[6]);
		contract(c);
		contract(c + 4);
		set4(e, v[0], v[2], v[4], v[6]);
		set4(e + 4, iadd(v[0], v[1]), iadd(v[2], v[3]), iadd(v[4], v[5]),
		    iadd(v[6], v[7]));
		for (i = 0; i < 8; i++)
			e[i] = clamp8(mk_sel(swap, c[i], e[i]));
		return;
	}
	/* an HDR mode: the error colour */
	for (i = 0; i < 8; i++)
		e[i] = ks((i & 3) == 1 ? 0 : 255);
}

/* byte k of mode cem's endpoints: each a select when it is not a value */
static nid
endpoint(nid cem, nid v, nid k)
{
	nid r = mk_undef(I32), e[8];
	int c, i;

	for (c = 15; c >= 0; c--) {
		nid b = mk_undef(I32);
		if (is_const(cem) && N.ki[cem] != c)
			continue;
		endpoints(c, v, e);
		if (is_const(k))
			b = N.ki[k] >= 0 && N.ki[k] < 8 ? e[N.ki[k]] : b;
		else
			for (i = 7; i >= 0; i--)
				b = mk_sel(mk_bin(O_EQ, k, ku((uint32_t)i)), e[i], b);
		r = is_const(cem) ? b : mk_sel(mk_bin(O_EQ, cem, ku((uint32_t)c)),
		    b, r);
	}
	return mk_cast(U32, r);
}

/* ---- the weight grid's infill, and the interpolation ---- */

static nid
bilerp(nid G, int64_t gw, nid s, nid t)
{
	uint32_t n = N.b[G], gh = (uint32_t)(n / gw), i;
	nid js = shr(s, 4), fs = band(s, 15), jt = shr(t, 4), ft = band(t, 15);
	nid w[4], p[4], r = ku(8);

	w[3] = shr(mk_bin(O_ADD, mk_bin(O_MUL, fs, ft), ku(8)), 4);
	w[2] = mk_bin(O_SUB, ft, w[3]);
	w[1] = mk_bin(O_SUB, fs, w[3]);
	w[0] = mk_bin(O_ADD, mk_bin(O_SUB, mk_bin(O_SUB, ku(16), fs), ft), w[3]);
	for (i = 0; i < 4; i++) {
		nid x = mk_bin(O_ADD, js, ku(i & 1)), y = mk_bin(O_ADD, jt, ku(i >> 1));
		nid in = mk_bin(O_LAND, mk_bin(O_LT, x, ku((uint32_t)gw)),
		    mk_bin(O_LT, y, ku(gh)));
		nid at = mk_sel(in, mk_bin(O_ADD, x, mk_bin(O_MUL, y,
		    ku((uint32_t)gw))), ku(0));
		p[i] = mk_sel(in, mk_cast(U32, mk_index(G, at)), ku(0));
		r = mk_bin(O_ADD, r, mk_bin(O_MUL, p[i], w[i]));
	}
	return shr(r, 4);
}

static nid
lerp(nid e0, nid e1, nid w)
{
	nid c0 = mk_bin(O_MUL, e0, ku(257)), c1 = mk_bin(O_MUL, e1, ku(257));
	nid ok = mk_bin(O_LAND, mk_bin(O_LAND, mk_bin(O_LE, e0, ku(255)),
	    mk_bin(O_LE, e1, ku(255))), mk_bin(O_LE, w, ku(64)));
	nid r = shr(mk_bin(O_ADD, mk_bin(O_ADD, mk_bin(O_MUL, c0,
	    mk_bin(O_SUB, ku(64), w)), mk_bin(O_MUL, c1, w)), ku(32)), 6);

	return mk_sel(ok, r, mk_undef(U32));
}

/* ---- the HDR endpoint modes and interpolation (#109) ---- */

/* bit i of x, 0 or 1 */
static nid
xbit(nid x, int i)
{
	return mk_bin(O_AND, mk_bin(O_SHR, x, ks(i)), ks(1));
}

/* acc with bit b (0 or 1) placed at k */
static nid
place(nid acc, nid b, int k)
{
	return mk_bin(O_OR, acc, mk_bin(O_SHL, b, ks(k)));
}

static nid
clamp12(nid x)
{
	return mk_bin(O_MIN, mk_bin(O_MAX, x, ks(0)), ks(0xFFF));
}

/* the low n bits of x read as signed */
static nid
sext(nid x, int n)
{
	return mk_bin(O_SUB, mk_bin(O_XOR, mk_bin(O_AND, x, ks((1 << n) - 1)),
	    ks(1 << (n - 1))), ks(1 << (n - 1)));
}

/* x times 2^k: a left shift that stays defined for a negative x */
static nid
times2k(nid x, int k)
{
	return mk_bin(O_MUL, x, ks(1 << k));
}

/* swap of two channels in both endpoints where c holds */
static void
swap_where(nid c, nid *e, int i, int j)
{
	nid a, b;
	int k;

	for (k = 0; k < 8; k += 4) {
		a = mk_sel(c, e[k + j], e[k + i]);
		b = mk_sel(c, e[k + i], e[k + j]);
		e[k + i] = a;
		e[k + j] = b;
	}
}

/* mode 11's RGB (23.15.4), e[3] and e[7] 0x780 */
static void
hdr11(const nid *v, nid *e)
{
	static const int dbits[8] = {7, 6, 7, 6, 5, 6, 5, 6};
	nid maj = mk_bin(O_OR, xbit(v[4], 7), mk_bin(O_SHL, xbit(v[5], 7),
	    ks(1)));
	nid mode = mk_bin(O_OR, mk_bin(O_OR, xbit(v[1], 7), mk_bin(O_SHL,
	    xbit(v[2], 7), ks(1))), mk_bin(O_SHL, xbit(v[3], 7), ks(2)));
	nid x[6], r[8], d[8];
	int m, i;

	x[0] = xbit(v[2], 6);
	x[1] = xbit(v[3], 6);
	x[2] = xbit(v[4], 6);
	x[3] = xbit(v[5], 6);
	x[4] = xbit(v[4], 5);
	x[5] = xbit(v[5], 5);
	for (m = 7; m >= 0; m--) {
		int ohm = 1 << m, sh = (m >> 1) ^ 3;
		nid va = mk_bin(O_OR, v[0], mk_bin(O_SHL, mk_bin(O_AND, v[1],
		    ks(0x40)), ks(2)));
		nid vb0 = mk_bin(O_AND, v[2], ks(0x3f)), vb1 = mk_bin(O_AND, v[3],
		    ks(0x3f)), vc = mk_bin(O_AND, v[1], ks(0x3f));
		nid vd0 = sext(v[4], dbits[m]), vd1 = sext(v[5], dbits[m]), e1[8];
		if (ohm & 0xA4)
			va = place(va, x[0], 9);
		if (ohm & 0x08)
			va = place(va, x[2], 9);
		if (ohm & 0x50)
			va = place(va, x[4], 9);
		if (ohm & 0x50)
			va = place(va, x[5], 10);
		if (ohm & 0xA0)
			va = place(va, x[1], 10);
		if (ohm & 0xC0)
			va = place(va, x[2], 11);
		if (ohm & 0x04)
			vc = place(vc, x[1], 6);
		if (ohm & 0xE8)
			vc = place(vc, x[3], 6);
		if (ohm & 0x20)
			vc = place(vc, x[2], 7);
		if (ohm & 0x5B)
			vb0 = place(vb0, x[0], 6);
		if (ohm & 0x5B)
			vb1 = place(vb1, x[1], 6);
		if (ohm & 0x12)
			vb0 = place(vb0, x[2], 7);
		if (ohm & 0x12)
			vb1 = place(vb1, x[3], 7);
		va = times2k(va, sh);
		vb0 = times2k(vb0, sh);
		vb1 = times2k(vb1, sh);
		vc = times2k(vc, sh);
		vd0 = times2k(vd0, sh);
		vd1 = times2k(vd1, sh);
		e1[4] = clamp12(va);
		e1[5] = clamp12(mk_bin(O_SUB, va, vb0));
		e1[6] = clamp12(mk_bin(O_SUB, va, vb1));
		e1[0] = clamp12(mk_bin(O_SUB, va, vc));
		e1[1] = clamp12(mk_bin(O_SUB, mk_bin(O_SUB, mk_bin(O_SUB, va,
		    vb0), vc), vd0));
		e1[2] = clamp12(mk_bin(O_SUB, mk_bin(O_SUB, mk_bin(O_SUB, va,
		    vb1), vc), vd1));
		for (i = 0; i < 8; i++) {
			if ((i & 3) == 3)
				continue;
			r[i] = m == 7 ? e1[i] : mk_sel(mk_bin(O_EQ, mode, ks(m)),
			    e1[i], r[i]);
		}
	}
	r[3] = r[7] = ks(0x780);
	swap_where(mk_bin(O_EQ, maj, ks(1)), r, 0, 1);
	swap_where(mk_bin(O_EQ, maj, ks(2)), r, 0, 2);
	/* both major bits set: the values directly */
	d[0] = mk_bin(O_SHL, v[0], ks(4));
	d[1] = mk_bin(O_SHL, v[2], ks(4));
	d[2] = mk_bin(O_SHL, mk_bin(O_AND, v[4], ks(0x7f)), ks(5));
	d[4] = mk_bin(O_SHL, v[1], ks(4));
	d[5] = mk_bin(O_SHL, v[3], ks(4));
	d[6] = mk_bin(O_SHL, mk_bin(O_AND, v[5], ks(0x7f)), ks(5));
	d[3] = d[7] = ks(0x780);
	for (i = 0; i < 8; i++)
		e[i] = mk_sel(mk_bin(O_EQ, maj, ks(3)), d[i], r[i]);
}

/* mode 7 (23.15.3): a base RGB, a scale and the mode bits */
static void
hdr7(const nid *v, nid *e)
{
	static const int shamts[6] = {1, 1, 2, 3, 4, 5};
	nid modeval = mk_bin(O_OR, mk_bin(O_OR, mk_bin(O_SHR, mk_bin(O_AND,
	    v[0], ks(0xC0)), ks(6)), mk_bin(O_SHR, mk_bin(O_AND, v[1], ks(0x80)),
	    ks(5))), mk_bin(O_SHR, mk_bin(O_AND, v[2], ks(0x80)), ks(4)));
	nid x[7];
	int mv, i;

	x[0] = xbit(v[1], 6);
	x[1] = xbit(v[1], 5);
	x[2] = xbit(v[2], 6);
	x[3] = xbit(v[2], 5);
	x[4] = xbit(v[3], 7);
	x[5] = xbit(v[3], 6);
	x[6] = xbit(v[3], 5);
	for (mv = 15; mv >= 0; mv--) {
		int maj, mode, ohm, sh;
		nid red = mk_bin(O_AND, v[0], ks(0x3f));
		nid green = mk_bin(O_AND, v[1], ks(0x1f));
		nid blue = mk_bin(O_AND, v[2], ks(0x1f));
		nid scale = mk_bin(O_AND, v[3], ks(0x1f)), c[8], t;
		if ((mv & 0xC) != 0xC) {
			maj = mv >> 2;
			mode = mv & 3;
		} else if (mv != 0xF) {
			maj = mv & 3;
			mode = 4;
		} else {
			maj = 0;
			mode = 5;
		}
		ohm = 1 << mode;
		if (ohm & 0x30)
			green = place(green, x[0], 6);
		if (ohm & 0x3A)
			green = place(green, x[1], 5);
		if (ohm & 0x30)
			blue = place(blue, x[2], 6);
		if (ohm & 0x3A)
			blue = place(blue, x[3], 5);
		if (ohm & 0x3D)
			scale = place(scale, x[6], 5);
		if (ohm & 0x2D)
			scale = place(scale, x[5], 6);
		if (ohm & 0x04)
			scale = place(scale, x[4], 7);
		if (ohm & 0x3B)
			red = place(red, x[4], 6);
		if (ohm & 0x04)
			red = place(red, x[3], 6);
		if (ohm & 0x10)
			red = place(red, x[5], 7);
		if (ohm & 0x0F)
			red = place(red, x[2], 7);
		if (ohm & 0x05)
			red = place(red, x[1], 8);
		if (ohm & 0x0A)
			red = place(red, x[0], 8);
		if (ohm & 0x05)
			red = place(red, x[0], 9);
		if (ohm & 0x02)
			red = place(red, x[6], 9);
		if (ohm & 0x01)
			red = place(red, x[3], 10);
		if (ohm & 0x02)
			red = place(red, x[5], 10);
		sh = shamts[mode];
		red = times2k(red, sh);
		green = times2k(green, sh);
		blue = times2k(blue, sh);
		scale = times2k(scale, sh);
		if (mode != 5) {
			green = mk_bin(O_SUB, red, green);
			blue = mk_bin(O_SUB, red, blue);
		}
		if (maj == 1) {
			t = red;
			red = green;
			green = t;
		} else if (maj == 2) {
			t = red;
			red = blue;
			blue = t;
		}
		c[4] = clamp12(red);
		c[5] = clamp12(green);
		c[6] = clamp12(blue);
		c[0] = clamp12(mk_bin(O_SUB, red, scale));
		c[1] = clamp12(mk_bin(O_SUB, green, scale));
		c[2] = clamp12(mk_bin(O_SUB, blue, scale));
		for (i = 0; i < 8; i++)
			if ((i & 3) != 3)
				e[i] = mv == 15 ? c[i] : mk_sel(mk_bin(O_EQ,
				    modeval, ks(mv)), c[i], e[i]);
	}
	e[3] = e[7] = ks(0x780);
}

/* mode 15's alphas (23.15.6) */
static void
hdr15a(const nid *v, nid *e)
{
	nid mode = mk_bin(O_OR, xbit(v[6], 7), mk_bin(O_SHL, xbit(v[7], 7),
	    ks(1)));
	nid a6 = mk_bin(O_AND, v[6], ks(0x7F)), a7 = mk_bin(O_AND, v[7],
	    ks(0x7F)), r3 = mk_bin(O_SHL, a6, ks(5)), r7 = mk_bin(O_SHL, a7,
	    ks(5));
	int m;

	for (m = 2; m >= 0; m--) {
		nid b6 = mk_bin(O_OR, a6, mk_bin(O_AND, mk_bin(O_SHL, a7, ks(m +
		    1)), ks(0x780)));
		nid b7 = mk_bin(O_SUB, mk_bin(O_XOR, mk_bin(O_AND, a7, ks(0x3F >>
		    m)), ks(0x20 >> m)), ks(0x20 >> m));
		b6 = mk_bin(O_SHL, b6, ks(4 - m));
		b7 = clamp12(mk_bin(O_ADD, times2k(b7, 4 - m), b6));
		r3 = mk_sel(mk_bin(O_EQ, mode, ks(m)), b6, r3);
		r7 = mk_sel(mk_bin(O_EQ, mode, ks(m)), b7, r7);
	}
	e[3] = r3;
	e[7] = r7;
}

/* e[0..7]: the 12-bit values of HDR mode cem (2, 3, 7, 11, 14, 15) from
 * the bytes of v (i32); mode 14's alphas are its bytes */
static int
hdr_endpoints(int cem, nid v64, nid *e)
{
	nid v[8], y0, y1, d;
	int i;

	for (i = 0; i < 8; i++)
		v[i] = mk_cast(I32, mk_bin(O_AND, mk_bin(O_SHR, v64, mk_const_i(
		    TY(K_UINT, 64), 8 * i)), mk_const_i(TY(K_UINT, 64), 255)));
	switch (cem) {
	case 2:
		d = mk_bin(O_GE, v[1], v[0]);
		y0 = mk_sel(d, mk_bin(O_SHL, v[0], ks(4)), mk_bin(O_ADD,
		    mk_bin(O_SHL, v[1], ks(4)), ks(8)));
		y1 = mk_sel(d, mk_bin(O_SHL, v[1], ks(4)), mk_bin(O_SUB,
		    mk_bin(O_SHL, v[0], ks(4)), ks(8)));
		set4(e, y0, y0, y0, ks(0x780));
		set4(e + 4, y1, y1, y1, ks(0x780));
		return 1;
	case 3:
		d = mk_bin(O_NE, mk_bin(O_AND, v[0], ks(0x80)), ks(0));
		y0 = mk_sel(d, mk_bin(O_OR, mk_bin(O_SHL, mk_bin(O_AND, v[1],
		    ks(0xE0)), ks(4)), mk_bin(O_SHL, mk_bin(O_AND, v[0], ks(0x7F)),
		    ks(2))), mk_bin(O_OR, mk_bin(O_SHL, mk_bin(O_AND, v[1],
		    ks(0xF0)), ks(4)), mk_bin(O_SHL, mk_bin(O_AND, v[0], ks(0x7F)),
		    ks(1))));
		y1 = mk_bin(O_MIN, mk_bin(O_ADD, y0, mk_sel(d, mk_bin(O_SHL,
		    mk_bin(O_AND, v[1], ks(0x1F)), ks(2)), mk_bin(O_SHL,
		    mk_bin(O_AND, v[1], ks(0x0F)), ks(1)))), ks(0xFFF));
		set4(e, y0, y0, y0, ks(0x780));
		set4(e + 4, y1, y1, y1, ks(0x780));
		return 1;
	case 7:
		hdr7(v, e);
		return 1;
	case 11: case 14: case 15:
		hdr11(v, e);
		if (cem == 14) {
			e[3] = v[6];
			e[7] = v[7];
		} else if (cem == 15)
			hdr15a(v, e);
		return 1;
	}
	return 0;
}

/* component k of HDR mode cem: each a select when it is not a value */
static nid
hdr_endpoint(nid cem, nid v, nid k)
{
	static const int modes[6] = {2, 3, 7, 11, 14, 15};
	nid r = mk_undef(I32), e[8];
	int c, i;

	for (c = 5; c >= 0; c--) {
		nid b = mk_undef(I32);
		if (is_const(cem) && N.ki[cem] != modes[c])
			continue;
		hdr_endpoints(modes[c], v, e);
		if (is_const(k))
			b = N.ki[k] >= 0 && N.ki[k] < 8 ? e[N.ki[k]] : b;
		else
			for (i = 7; i >= 0; i--)
				b = mk_sel(mk_bin(O_EQ, k, ku((uint32_t)i)), e[i],
				    b);
		r = is_const(cem) ? b : mk_sel(mk_bin(O_EQ, cem,
		    ku((uint32_t)modes[c])), b, r);
	}
	return mk_cast(U32, r);
}

/* two 12-bit values at weight w, as the 16-bit C of 23.19 */
static nid
lerph(nid e0, nid e1, nid w)
{
	nid ok = mk_bin(O_LAND, mk_bin(O_LAND, mk_bin(O_LE, e0, ku(0xFFF)),
	    mk_bin(O_LE, e1, ku(0xFFF))), mk_bin(O_LE, w, ku(64)));
	nid r = shr(mk_bin(O_ADD, mk_bin(O_ADD, mk_bin(O_MUL, shl(e0, 4),
	    mk_bin(O_SUB, ku(64), w)), mk_bin(O_MUL, shl(e1, 4), w)), ku(32)), 6);

	return mk_sel(ok, r, mk_undef(U32));
}

/* the 16-bit C to FP16 bits: the top five bits the exponent, the low 11
 * through the piecewise map closer to a logarithm, Inf and NaN as 0x7BFF */
static nid
lns(nid c)
{
	nid m = band(c, 0x7FF), mt, cf;

	mt = mk_sel(mk_bin(O_LT, m, ku(512)), mk_bin(O_MUL, m, ku(3)),
	    mk_sel(mk_bin(O_GE, m, ku(1536)), mk_bin(O_SUB, mk_bin(O_MUL, m,
	    ku(5)), ku(2048)), mk_bin(O_SUB, mk_bin(O_MUL, m, ku(4)), ku(512))));
	cf = mk_bin(O_ADD, shl(shr(band(c, 0xF800), 11), 10), shr(mt, 3));
	return mk_sel(mk_bin(O_LE, c, ku(0xFFFF)), mk_bin(O_MIN, cf, ku(0x7BFF)),
	    mk_undef(U32));
}

/* a UNORM16 to FP16 bits (decode_float16): 65535 is 1.0, the rest c/65536
 * rounded toward zero */
static nid
f16of(nid c)
{
	nid r = ku(0);
	int e;

	/* c in [2^e, 2^(e+1)): exponent e - 1 (biased e - 16 + 15), the
	 * significand's 10 bits below the top one */
	for (e = 0; e < 16; e++) {
		nid in = mk_bin(O_GE, c, ku(1u << e)), v;
		if (e < 2)	/* c 2^-16 is subnormal: c 2^8 units of 2^-24 */
			v = shl(c, 8);
		else
			v = mk_bin(O_OR, ku((uint32_t)(e - 1) << 10), e >= 10 ?
			    band(shr(c, e - 10), 0x3FF) : band(shl(c, 10 - e),
			    0x3FF));
		r = mk_sel(in, v, r);
	}
	r = mk_sel(mk_bin(O_EQ, c, ku(65535)), ku(0x3C00), r);
	return mk_sel(mk_bin(O_LE, c, ku(0xFFFF)), r, mk_undef(U32));
}

nid
astc_expand(int op, int64_t ki, nid a, nid b, nid c)
{
	switch (op) {
	case O_ASTC_PART:
		return partition((int)ki, a, b, c);
	case O_ASTC_UNQ:
	case O_ASTC_UNQW:
		return unquant(op == O_ASTC_UNQW, a, b);
	case O_ASTC_ENDPOINT:
		return endpoint(a, b, c);
	case O_ASTC_BILERP:
		return bilerp(a, ki, b, c);
	case O_ASTC_LERP:
		return lerp(a, b, c);
	case O_ASTC_HDR:
		return hdr_endpoint(a, b, c);
	case O_ASTC_LERPH:
		return lerph(a, b, c);
	case O_ASTC_LNS:
		return lns(a);
	case O_ASTC_F16:
		return f16of(a);
	}
	fail("%s: not an ASTC step", opinfo[op].name);
	return 0;
}

/* a table whose entries are all constants */
static int
const_table(nid t)
{
	uint32_t j;

	if (N.op[t] != O_TABLE)
		return 0;
	for (j = 0; j < N.b[t]; j++)
		if (!is_const(N.lst[N.a[t] + j]))
			return 0;
	return 1;
}

nid
mk_astc(int op, int64_t ki, nid a, nid b, nid c)
{
	const char *nm = opinfo[op].name;
	int ar = opinfo[op].arity, fl = NF_SG, i;
	/* a step of one operand keeps no other (the caller's may be unset) */
	nid o[3] = {a, ar >= 2 ? b : 0, ar == 3 ? c : 0}, r;

	if (op == O_ASTC_BILERP) {
		if (N.op[a] != O_TABLE)
			fail("astc_bilerp takes a table of weights first, not %s",
			    node_str(a));
		if (ki < 1 || N.b[a] % (uint64_t)ki)
			fail("astc_bilerp: a grid of %u weights is not rows of %lld",
			    N.b[a], (long long)ki);
	}
	for (i = 0; i < ar; i++)
		if (!TY_ISINT(N.ty[o[i]]))
			fail("%s takes integer operands", nm);
	if (op == O_ASTC_PART && (ki < 1 || ki > 4))
		fail("astc_partition: a block has 1 to 4 partitions, not %lld",
		    (long long)ki);
	/* a constant out of its step's range is refused; a value is
	 * undefined there (domain says where) */
	if ((op == O_ASTC_UNQ || op == O_ASTC_UNQW) && is_const(a) &&
	    !level_ok(N.ki[a], op == O_ASTC_UNQW))
		fail("%s: %lld is not a %s range's count (%s)", nm,
		    (long long)N.ki[a], op == O_ASTC_UNQ ? "colour" : "weight",
		    op == O_ASTC_UNQ ? "6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, "
		    "80, 96, 128, 160, 192, 256" : "2, 3, 4, 5, 6, 8, 10, 12, 16, "
		    "20, 24, 32");
	if (op == O_ASTC_ENDPOINT && is_const(a) && (N.ki[a] < 0 ||
	    N.ki[a] > 15))
		fail("astc_endpoint: an endpoint mode is 0..15, not %lld",
		    (long long)N.ki[a]);
	if (op == O_ASTC_ENDPOINT && is_const(c) && (N.ki[c] < 0 ||
	    N.ki[c] > 7))
		fail("astc_endpoint: the two endpoints have bytes 0..7, not %lld",
		    (long long)N.ki[c]);
	if ((op == O_ASTC_LERP || op == O_ASTC_LERPH) && is_const(c) &&
	    (N.ki[c] < 0 || N.ki[c] > 64))
		fail("%s: a weight is 0..64, not %lld", nm, (long long)N.ki[c]);
	if (op == O_ASTC_HDR && is_const(a) && N.ki[a] != 2 && N.ki[a] != 3 &&
	    N.ki[a] != 7 && N.ki[a] != 11 && N.ki[a] != 14 && N.ki[a] != 15)
		fail("astc_hdr: the HDR endpoint modes are 2, 3, 7, 11, 14 and "
		    "15, not %lld", (long long)N.ki[a]);
	if (op == O_ASTC_HDR && is_const(c) && (N.ki[c] < 0 || N.ki[c] > 7))
		fail("astc_hdr: the two endpoints have components 0..7, not %lld",
		    (long long)N.ki[c]);
	if (op == O_ASTC_LERPH)
		for (i = 0; i < 2; i++)
			if (is_const(o[i]) && (N.ki[o[i]] < 0 || N.ki[o[i]] >
			    0xFFF))
				fail("astc_lerph: an endpoint is a 12-bit value, "
				    "not %lld", (long long)N.ki[o[i]]);
	if ((op == O_ASTC_LNS || op == O_ASTC_F16) && is_const(a) &&
	    (N.ki[a] < 0 || N.ki[a] > 0xFFFF))
		fail("%s: C is a 16-bit value, not %lld", nm, (long long)N.ki[a]);
	if (op == O_ASTC_LERP)
		for (i = 0; i < 2; i++)
			if (is_const(o[i]) && (N.ki[o[i]] < 0 || N.ki[o[i]] > 255))
				fail("astc_lerp: an endpoint is a byte, not %lld",
				    (long long)N.ki[o[i]]);
	for (i = 0; i < ar; i++) {
		if (i == 0 && op == O_ASTC_BILERP)
			continue;
		o[i] = mk_cast((op == O_ASTC_ENDPOINT || op == O_ASTC_HDR) && i ==
		    1 ? TY(K_UINT, 64) : U32, o[i]);
		if (N.op[o[i]] == O_UNDEF)
			return mk_undef(U32);
	}
	if (op == O_ASTC_PART && ki == 1)
		return ku(0);
	for (i = 0; i < ar; i++)
		if (!(i == 0 && op == O_ASTC_BILERP ? const_table(o[i]) :
		    is_const(o[i])))
			break;
	if (i == ar) {
		r = astc_expand(op, ki, o[0], o[1], o[2]);
		if (is_const(r) || N.op[r] == O_UNDEF)
			return r;
	}
	for (i = 0; i < ar; i++)
		fl |= N.fl[o[i]] & NF_PARTIAL;
	/* outside its operands' ranges it is undefined */
	if (op == O_ASTC_UNQ || op == O_ASTC_UNQW || op == O_ASTC_ENDPOINT ||
	    op == O_ASTC_LERP || op == O_ASTC_HDR || op == O_ASTC_LERPH ||
	    op == O_ASTC_LNS || op == O_ASTC_F16)
		fl |= NF_PARTIAL;
	return mk_raw(op, U32, o[0], o[1], o[2], ki, fl);
}

/*
 * A call of the language: astc_partition's count rides in the node's ki,
 * and one that is not a constant (a decoder reads it from the block) is
 * a select of the four counts' nodes, undefined for any other;
 * astc_bilerp's row width rides there too and is a constant
 */
nid
astc_call(int op, const nid *args)
{
	nid n = args[3], r;
	int k;

	if (op == O_ASTC_BILERP) {
		n = args[1];
		if (!is_const(n) || !TY_ISINT(N.ty[n]))
			fail("astc_bilerp takes a constant row width (the grid's "
			    "width), not %s", node_str(n));
		return mk_astc(op, N.ki[n], args[0], args[2], args[3]);
	}
	if (op != O_ASTC_PART)
		return mk_astc(op, 0, args[0], opinfo[op].arity >= 2 ? args[1] :
		    0, opinfo[op].arity == 3 ? args[2] : 0);
	if (!TY_ISINT(N.ty[n]))
		fail("astc_partition takes an integer partition count 1..4, not "
		    "%s", node_str(n));
	if (is_const(n))
		return mk_astc(op, N.ki[n], args[0], args[1], args[2]);
	r = mk_undef(U32);
	for (k = 4; k >= 1; k--)
		r = mk_sel(mk_bin(O_EQ, n, mk_const_i(N.ty[n], k)), mk_astc(op, k,
		    args[0], args[1], args[2]), r);
	return r;
}

/*
 * The node's price under m: its definition's operations, over its
 * operands as they are (a constant seed folds the hash, a grid stays a
 * table), each other operand a free value.
 */
float
astc_cost(const struct model *m, nid n)
{
	static const char *const nm[3] = {"_a", "_b", "_c"};
	nid op[3] = {N.a[n], N.b[n], N.c[n]};
	int j;

	for (j = 0; j < opinfo[N.op[n]].arity; j++)
		if (!is_const(op[j]) && N.op[op[j]] != O_TABLE)
			op[j] = mk_var((uint32_t)j, sym_cstr(nm[j]),
			    N.ty[op[j]]);
	return fuse_expr_cost(m, astc_expand(N.op[n], N.ki[n], op[0], op[1],
	    op[2]));
}
