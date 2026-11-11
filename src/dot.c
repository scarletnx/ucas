/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Packed integer dot products, as SPV_KHR_integer_dot_product defines
 * them (OpUDot, OpSDot, OpSUDot and their AccSat forms), with the lanes
 * packed in a 32-bit word:
 *
 *   4 x 8 bits   PackedVectorFormat4x8Bit (DP4A on NVIDIA sm_61 and later,
 *                vpdpbusd on x86 with VNNI: signed x unsigned)
 *   2 x 16 bits  a 2-component vector of 16-bit integers (vpdpwssd)
 *   8 x 4 bits   int4 lanes (AMD v_dot8_u32_u4 / v_dot8_i32_i4); SPIR-V
 *                has no form for these, so they stay in expressions
 *
 * Lane k of a word w is bits [k*bits, (k+1)*bits), zero-extended for an
 * unsigned operand, sign-extended for a signed one; su: the first
 * operand's lanes are signed, the second's unsigned.  The result is u32
 * when both operands are unsigned, i32 otherwise.
 *
 *   udot4x8(a, b)        = sum_k a_k * b_k, its low 32 bits ("the
 *                          low-order N bits of the correct result")
 *   udot4x8sat(a, b, c)  = the exact sum plus c, clamped to the result's
 *                          range (c has the result's type)
 *
 * so for 4x8 and 8x4 the plain forms never wrap (|sum| <= 4 * 255^2 =
 * 260100 and 8 * 15^2); 2 x 16 can (2 * 65535^2 > 2^32).
 *
 * The second half is the search: sums of products laid into the lanes of
 * the forms the active cost model offers (dot_find).
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const struct dotform forms[O_DOT_LAST - O_DOT_FIRST + 1] = {
	{"udot4x8", 4, 8, 0, 0, 0}, {"sdot4x8", 4, 8, 1, 1, 0},
	{"sudot4x8", 4, 8, 1, 0, 0}, {"udot2x16", 2, 16, 0, 0, 0},
	{"sdot2x16", 2, 16, 1, 1, 0}, {"sudot2x16", 2, 16, 1, 0, 0},
	{"udot8x4", 8, 4, 0, 0, 0}, {"sdot8x4", 8, 4, 1, 1, 0},
	{"sudot8x4", 8, 4, 1, 0, 0},
	{"udot4x8sat", 4, 8, 0, 0, 1}, {"sdot4x8sat", 4, 8, 1, 1, 1},
	{"sudot4x8sat", 4, 8, 1, 0, 1}, {"udot2x16sat", 2, 16, 0, 0, 1},
	{"sdot2x16sat", 2, 16, 1, 1, 1}, {"sudot2x16sat", 2, 16, 1, 0, 1},
	{"udot8x4sat", 8, 4, 0, 0, 1}, {"sdot8x4sat", 8, 4, 1, 1, 1},
	{"sudot8x4sat", 8, 4, 1, 0, 1},
};

const struct dotform *
dot_form(int op)
{
	return IS_DOT(op) ? &forms[op - O_DOT_FIRST] : NULL;
}

/* the op for a form; s x u has no op of its own (swap the operands) */
int
dot_op(int lanes, int bits, int sa, int sb, int sat)
{
	int k;

	for (k = 0; k <= O_DOT_LAST - O_DOT_FIRST; k++)
		if (forms[k].lanes == lanes && forms[k].bits == bits &&
		    forms[k].sa == sa && forms[k].sb == sb && forms[k].sat == sat)
			return O_DOT_FIRST + k;
	return 0;
}

static int64_t
lane(uint64_t w, int k, int bits, int s)
{
	uint64_t m = (1ULL << bits) - 1, v = (w >> (k * bits)) & m;

	if (s && (v >> (bits - 1)))
		return (int64_t)v - (int64_t)(1ULL << bits);
	return (int64_t)v;
}

int64_t
dot_scalar(int op, int ideal, int64_t a, int64_t b, int64_t acc,
    uint8_t *fl)
{
	const struct dotform *df = dot_form(op);
	uint64_t ua = (uint64_t)a & 0xffffffffULL, ub = (uint64_t)b & 0xffffffffULL;
	int64_t sum = 0, lo, hi;
	int k, sres = df->sa || df->sb;

	for (k = 0; k < df->lanes; k++)
		sum += lane(ua, k, df->bits, df->sa) * lane(ub, k, df->bits, df->sb);
	lo = sres ? -(int64_t)0x80000000LL : 0;
	hi = sres ? 0x7fffffffLL : 0xffffffffLL;
	if (df->sat) {
		sum += acc;
		return sum < lo ? lo : sum > hi ? hi : sum;
	}
	if (ideal || (sum >= lo && sum <= hi))
		return sum;
	*fl |= sres ? F_SW : F_UW;
	sum &= 0xffffffffLL;
	return sres && sum > hi ? sum - 0x100000000LL : sum;
}

/*
 * Lane k's range from the word's [lo, hi] (as a u32): the top lane is
 * monotone in the word; a lower one is when the bits above it are the
 * same at lo and hi, otherwise it can be anything.
 */
static void
lane_range(int64_t lo, int64_t hi, int k, int lanes, int bits, int s,
    int64_t *l, int64_t *h)
{
	int64_t m = ((int64_t)1 << bits) - 1, half = (int64_t)1 << (bits - 1);
	int sh = k * bits;

	if (lo < 0 || hi > 0xffffffffLL || lo > hi) {
		*l = 0;
		*h = m;
	} else if (k == lanes - 1 || (hi >> (sh + bits)) == (lo >> (sh + bits))) {
		*l = (lo >> sh) & m;
		*h = (hi >> sh) & m;
	} else {
		*l = 0;
		*h = m;
	}
	if (!s)
		return;
	if (*h < half)
		return;
	if (*l >= half) {
		*l -= m + 1;
		*h -= m + 1;
		return;
	}
	*l = -half;
	*h = half - 1;
}

int
dot_ival(int op, int64_t alo, int64_t ahi, int64_t blo, int64_t bhi,
    int64_t clo, int64_t chi, int64_t *lo, int64_t *hi)
{
	const struct dotform *df = dot_form(op);
	int64_t slo = 0, shi = 0, rlo, rhi;
	int k, j, sres = df->sa || df->sb;

	for (k = 0; k < df->lanes; k++) {
		int64_t al, ah, bl, bh, x[4], pl, ph;
		lane_range(alo, ahi, k, df->lanes, df->bits, df->sa, &al, &ah);
		lane_range(blo, bhi, k, df->lanes, df->bits, df->sb, &bl, &bh);
		x[0] = al * bl;
		x[1] = al * bh;
		x[2] = ah * bl;
		x[3] = ah * bh;
		pl = ph = x[0];
		for (j = 1; j < 4; j++) {
			if (x[j] < pl)
				pl = x[j];
			if (x[j] > ph)
				ph = x[j];
		}
		slo += pl;
		shi += ph;
	}
	rlo = sres ? -(int64_t)0x80000000LL : 0;
	rhi = sres ? 0x7fffffffLL : 0xffffffffLL;
	if (df->sat) {
		slo += clo;
		shi += chi;
		*lo = slo < rlo ? rlo : slo > rhi ? rhi : slo;
		*hi = shi < rlo ? rlo : shi > rhi ? rhi : shi;
		return 1;
	}
	*lo = slo;
	*hi = shi;
	return slo >= rlo && shi <= rhi;
}

void
dot_find(const struct func *f)
{
	(void)f;
}
