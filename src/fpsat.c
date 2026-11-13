/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * IEEE 754 binary floating point as circuits of sat.c's and-inverter
 * graph, for nodesat.c's proofs about float formulas: binary16, binary32
 * and binary64 (any exponent and significand width), rounded to nearest
 * even, toward zero, up or down, with subnormals, signed zeros,
 * infinities and NaN as IEEE 754 has them.
 *
 * Values stay unpacked between operations (struct fpv, sat.h): the
 * significand is an integer of whatever width the operation left, not
 * normalized, and the exponent is that of its bit 0.  An exact product
 * is then the product of the integers, and a conversion to a wider
 * format changes nothing.  Rounding (round_core) finds the bits it drops
 * from the leading one (the precision) and from the exponent (the
 * subnormal floor), one literal per position; where the exponent's
 * static bounds and the significand's width show that no position can be
 * dropped and nothing can overflow, all of them are the constant false
 * and the rounding folds away in the graph.  So f32(f64(x) * f64(y)) and
 * x * y build the same circuit, not two that the solver must show equal.
 *
 * Packing (fp_pack) is for what reads bits: a bit cast, a comparison,
 * the miter.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "sat.h"

#define ELIM 16000	/* exponent bounds kept well inside 16 bits */

const char *fp_err;
static int mut;		/* fp_selftest's deliberate defect, 0 none */

static const struct fpfmt fmt16 = {5, 11}, fmt32 = {8, 24}, fmt64 = {11, 53};

const struct fpfmt *
fp_fmt(int w)
{
	return w == 16 ? &fmt16 : w == 32 ? &fmt32 : &fmt64;
}

static int
bias(const struct fpfmt *F)
{
	return (1 << (F->eb - 1)) - 1;
}

/* the least subnormal is 2^qmin */
static int
qmin(const struct fpfmt *F)
{
	return 2 - bias(F) - F->p;
}

static int
fwidth(const struct fpfmt *F)
{
	return F->eb + F->p;
}

static void
toolarge(const char *why)
{
	if (!fp_err)
		fp_err = why;
}

static void
set_bounds(struct fpv *r, long lo, long hi)
{
	if (lo < -ELIM || hi > ELIM) {
		toolarge("a float exponent beyond the circuit's 16 bits");
		lo = lo < -ELIM ? -ELIM : lo;
		hi = hi > ELIM ? ELIM : hi;
	}
	r->elo = (int)lo;
	r->ehi = (int)hi;
}

/* e < k for e within [lo, hi]: a constant where the bounds decide it */
static lit
e_lt(struct aig *g, const struct word *e, long lo, long hi, long k)
{
	struct word c;

	if (hi < k)
		return L_TRUE;
	if (lo >= k)
		return L_FALSE;
	w_const(&c, FPV_EW, (uint64_t)k);
	return w_slt(g, e, &c);
}

/* a < b for words within [alo, ahi] and [blo, bhi] */
static lit
ee_lt(struct aig *g, const struct word *a, long alo, long ahi,
    const struct word *b, long blo, long bhi)
{
	if (ahi < blo)
		return L_TRUE;
	if (alo >= bhi)
		return L_FALSE;
	return w_slt(g, a, b);
}

/* drop constant zero bits from the top (one bit stays) */
static void
trim(struct fpv *a)
{
	while (a->n > 1 && a->m[a->n - 1] == L_FALSE)
		a->n--;
}

/* ab[k]: some bit of m above k is set, k = 0 .. n - 1 */
static void
above(struct aig *g, lit *ab, const lit *m, int n)
{
	int k;

	ab[n - 1] = L_FALSE;
	for (k = n - 2; k >= 0; k--)
		ab[k] = aig_or(g, ab[k + 1], m[k + 1]);
}

/* the position of the one true literal of h[0 .. n-1] (0 if none) */
static void
onehot_bin(struct aig *g, struct word *r, const lit *h, int n)
{
	int i, k;

	w_const(r, FPV_EW, 0);
	for (i = 1; i < n; i++)
		for (k = 0; k < FPV_EW && (i >> k); k++)
			if (i >> k & 1)
				r->b[k] = aig_or(g, r->b[k], h[i]);
}

/* the leading one of m: L one-hot, [*jlo, *jhi] where it can be */
static void
lead(struct aig *g, lit *L, const lit *m, int n, int *jlo, int *jhi)
{
	lit ab[FPV_MW];
	int i;

	above(g, ab, m, n);
	*jlo = n;
	*jhi = 0;
	for (i = 0; i < n; i++) {
		L[i] = aig_and(g, m[i], L_NOT(ab[i]));
		if (L[i] != L_FALSE) {
			if (i < *jlo)
				*jlo = i;
			*jhi = i;
		}
	}
	if (*jlo > *jhi)
		*jlo = *jhi = 0;
}

void
fp_unpack(struct aig *g, struct fpv *r, const struct word *x,
    const struct fpfmt *F)
{
	int fb = F->p - 1, i;
	lit ez = L_TRUE, e1 = L_TRUE, fz = L_TRUE;
	struct word ef, c;

	for (i = 0; i < fb; i++)
		fz = aig_and(g, fz, L_NOT(x->b[i]));
	for (i = 0; i < F->eb; i++) {
		ez = aig_and(g, ez, L_NOT(x->b[fb + i]));
		e1 = aig_and(g, e1, x->b[fb + i]);
	}
	r->s = x->b[fb + F->eb];
	r->nan = aig_and(g, e1, L_NOT(fz));
	r->inf = aig_and(g, e1, fz);
	r->z = aig_and(g, ez, fz);
	r->n = F->p;
	for (i = 0; i < fb; i++)
		r->m[i] = x->b[i];
	r->m[fb] = L_NOT(ez);
	/* e = max(field, 1) - bias - (p - 1) */
	w_const(&ef, FPV_EW, 0);
	for (i = 0; i < F->eb; i++)
		ef.b[i] = x->b[fb + i];
	ef.b[0] = aig_or(g, ef.b[0], ez);
	w_const(&c, FPV_EW, (uint64_t)(bias(F) + fb));
	w_sub(g, &r->e, &ef, &c);
	set_bounds(r, qmin(F), bias(F) - fb);
	r->nb = *x;
	trim(r);
}

/* a NaN's quiet bit set, as a binary32 value held in a double reads it */
void
fp_quiet(struct aig *g, struct word *x, const struct fpfmt *F)
{
	int fb = F->p - 1, i;
	lit e1 = L_TRUE, fz = L_TRUE;

	for (i = 0; i < fb; i++)
		fz = aig_and(g, fz, L_NOT(x->b[i]));
	for (i = 0; i < F->eb; i++)
		e1 = aig_and(g, e1, x->b[fb + i]);
	x->b[fb - 1] = aig_or(g, x->b[fb - 1], aig_and(g, e1, L_NOT(fz)));
}

/* a constant: its significand odd (or 0), so 2^k is m = 1 */
void
fp_const(struct aig *g, struct fpv *r, uint64_t bits, const struct fpfmt *F)
{
	struct word x;
	uint64_t v = 0;
	int t = 0, i, e;

	w_const(&x, fwidth(F), bits);
	fp_unpack(g, r, &x, F);
	if (r->z != L_FALSE || r->nan != L_FALSE || r->inf != L_FALSE)
		return;
	for (i = 0; i < FPV_EW; i++)
		v |= (uint64_t)(r->e.b[i] == L_TRUE) << i;
	e = v >= 0x8000 ? (int)v - 0x10000 : (int)v;
	while (t < r->n - 1 && r->m[t] == L_FALSE)
		t++;
	for (i = 0; i + t < r->n; i++)
		r->m[i] = r->m[i + t];
	r->n -= t;
	w_const(&r->e, FPV_EW, (uint64_t)(e + t));
	set_bounds(r, e + t, e + t);
}

/* any NaN of format F: a free sign and fraction, the fraction not 0 */
void
fp_anynan(struct aig *g, struct word *r, const struct fpfmt *F)
{
	int fb = F->p - 1, i;
	lit fz = L_TRUE;

	r->w = fwidth(F);
	for (i = 0; i < fb; i++) {
		r->b[i] = aig_input(g);
		fz = aig_and(g, fz, L_NOT(r->b[i]));
	}
	r->b[fb - 1] = aig_or(g, r->b[fb - 1], fz);
	for (i = 0; i < F->eb; i++)
		r->b[fb + i] = L_TRUE;
	r->b[r->w - 1] = aig_input(g);
}

/*
 * a's finite value, representable in F, as F's bits.  The leading one at
 * j puts the value's top at E = e + j: a normal number when E >= emin
 * (field E + bias), a subnormal otherwise.  With ml = m << (p - 1), the
 * fraction is ml >> sh: sh = j for a normal number (the bit below the
 * leading one becomes the fraction's top), qmin + p - 1 - e for a
 * subnormal (m's bit 0 lands at 2^e).  A NaN is nb with `payload`, else
 * the quiet NaN of sign 0: comparisons and the miter read no payload,
 * and two values that differ only there pack to the same literals.
 */
void
fp_pack(struct aig *g, struct word *r, const struct fpv *a,
    const struct fpfmt *F, int payload)
{
	int n = a->n, fb = F->p - 1, w = fwidth(F), i, k, jlo, jhi, nb = n + fb;
	lit L[FPV_MW], ml[2 * FPV_MW], nz = L_FALSE, normal, big = L_FALSE;
	struct word j, E, c, t, sh, field;

	lead(g, L, a->m, n, &jlo, &jhi);
	for (i = 0; i < n; i++)
		nz = aig_or(g, nz, a->m[i]);
	onehot_bin(g, &j, L, n);
	w_add(g, &E, &a->e, &j);
	normal = L_NOT(e_lt(g, &E, (long)a->elo + jlo, (long)a->ehi + jhi,
	    1 - bias(F) + (mut == 7)));
	w_const(&c, FPV_EW, (uint64_t)bias(F));
	w_add(g, &field, &E, &c);
	w_const(&c, FPV_EW, (uint64_t)(qmin(F) + fb));
	w_sub(g, &t, &c, &a->e);
	w_mux(g, &sh, normal, &j, &t);
	for (i = 0; i < nb; i++)
		ml[i] = i < fb ? L_FALSE : a->m[i - fb];
	for (k = 0; (1 << k) < nb; k++) {
		int d = 1 << k;
		for (i = 0; i < nb; i++)
			ml[i] = aig_mux(g, sh.b[k], i + d < nb ? ml[i + d] :
			    L_FALSE, ml[i]);
	}
	for (; k < FPV_EW; k++)
		big = aig_or(g, big, sh.b[k]);
	r->w = w;
	for (i = 0; i < fb; i++)
		r->b[i] = aig_and(g, ml[i], L_NOT(big));
	for (i = 0; i < F->eb; i++)
		r->b[fb + i] = aig_and(g, aig_and(g, field.b[i], normal), nz);
	r->b[w - 1] = a->s;
	for (i = 0; i < w - 1; i++)
		r->b[i] = aig_mux(g, a->inf, i >= fb ? L_TRUE : L_FALSE, r->b[i]);
	for (i = 0; i < w; i++)
		r->b[i] = aig_mux(g, a->nan, payload ? a->nb.b[i] : i >= fb - 1 &&
		    i < w - 1 ? L_TRUE : L_FALSE, r->b[i]);
}

void
fp_mux(struct aig *g, struct fpv *r, lit s, const struct fpv *t,
    const struct fpv *e)
{
	struct fpv x;
	int i, n = t->n > e->n ? t->n : e->n;

	if (s == L_TRUE || s == L_FALSE) {
		*r = s == L_TRUE ? *t : *e;
		return;
	}
	x.s = aig_mux(g, s, t->s, e->s);
	x.nan = aig_mux(g, s, t->nan, e->nan);
	x.inf = aig_mux(g, s, t->inf, e->inf);
	x.z = aig_mux(g, s, t->z, e->z);
	w_mux(g, &x.e, s, &t->e, &e->e);
	x.elo = t->elo < e->elo ? t->elo : e->elo;
	x.ehi = t->ehi > e->ehi ? t->ehi : e->ehi;
	x.n = n;
	for (i = 0; i < n; i++)
		x.m[i] = aig_mux(g, s, i < t->n ? t->m[i] : L_FALSE,
		    i < e->n ? e->m[i] : L_FALSE);
	w_mux(g, &x.nb, s, &t->nb, &e->nb);
	trim(&x);
	*r = x;
}

/*
 * a's finite value (-1)^s m 2^e, plus `sticky`: something nonzero lies
 * below m's bit 0 (the caller makes sure that two bits of m then fall
 * below the rounding position), rounded by mode to p significant bits (p
 * == 0: no limit) and to a multiple of 2^q, overflowing at 2^(emax + 1)
 * (when p > 0).
 *
 * dr[i], i = 0 .. n + 1: position i falls, for the precision (a set bit
 * above i + p - 1) or for the floor (e + i < q); once true for i, true
 * for every lower i.  The first kept position r has lk[r]; past n + 1
 * every bit fell, and the value is a multiple of 2^q again.  The guard
 * is bit r - 1, the sticky bits are those below it.  The kept bits move
 * down to bit 0 and e grows by r, so the result has at most p + 1 bits,
 * the top one a carry of the increment.  Nothing that can fall leaves
 * every dr[i] the constant false: lk[0] is true, r is 0, and the result
 * is the operand's own literals.
 */
static void
round_core(struct aig *g, struct fpv *r, const struct fpv *a, lit sticky,
    int p, long q, long emax, int mode)
{
	int n = a->n, K, i, j, rlo = -1, rhi = 0;
	lit ab[FPV_MW], dbuf[FPV_MW + 3], *dr = dbuf + 1, lk[FPV_MW + 2];
	lit mk[FPV_MW + 1], G = L_FALSE, S = sticky, de = L_TRUE, inc, all, c;
	lit exact, nz = L_FALSE, ovf = L_FALSE, up = L_FALSE, toinf, lsb, tomax;
	struct word rp, e2, k;
	struct fpv o;
	long lo, hi;

	above(g, ab, a->m, n);
	dr[-1] = L_TRUE;
	for (i = 0; i <= n + 1; i++) {
		lit dp = L_FALSE;
		if (p > 0 && i + p - 1 <= n - 2)
			dp = ab[i + p - 1];
		if (mut != 3)
			de = aig_and(g, de, e_lt(g, &a->e, a->elo, a->ehi,
			    q - i));
		else
			de = L_FALSE;
		dr[i] = aig_or(g, dp, de);
	}
	for (i = 0; i < n; i++) {
		G = aig_or(g, G, aig_and(g, aig_and(g, a->m[i], dr[i]),
		    L_NOT(dr[i + 1])));
		if (mut != 2)
			S = aig_or(g, S, aig_and(g, a->m[i], dr[i + 1]));
	}
	all = dr[n + 1];
	for (i = 0; i <= n + 1; i++) {
		lk[i] = aig_and(g, L_NOT(dr[i]), dr[i - 1]);
		if (lk[i] != L_FALSE) {
			if (rlo < 0)
				rlo = i;
			rhi = i;
		}
	}
	K = p > 0 && p < n ? p : n;
	if (K >= FPV_MW) {
		toolarge("too wide a float significand");
		K = FPV_MW - 1;
	}
	for (j = 0; j < K; j++) {
		lit v = L_FALSE;
		for (i = 0; i + j < n; i++)
			v = aig_or(g, v, aig_and(g, lk[i], a->m[i + j]));
		mk[j] = v;
	}
	lsb = mk[0];
	switch (mode) {
	case FR_RNE:
		inc = aig_and(g, G, mut == 1 ? L_TRUE : aig_or(g, S, lsb));
		break;
	case FR_RTZ:
		inc = L_FALSE;
		break;
	case FR_RUP:
		inc = aig_and(g, L_NOT(a->s), aig_or(g, G, S));
		break;
	default:
		inc = aig_and(g, a->s, aig_or(g, G, S));
		break;
	}
	o.s = a->s;
	o.nan = a->nan;
	o.nb = a->nb;
	c = inc;
	for (j = 0; j < K; j++) {
		o.m[j] = aig_xor(g, mk[j], c);
		c = aig_and(g, mk[j], c);
	}
	o.m[K] = c;
	o.n = K + 1;
	/* the exponent: e plus the first kept position, or q when every
	 * bit fell */
	onehot_bin(g, &rp, lk, n + 2);
	w_add(g, &e2, &a->e, &rp);
	if (rlo < 0) {
		lo = hi = q;
		w_const(&e2, FPV_EW, (uint64_t)q);
	} else {
		lo = a->elo + rlo;
		hi = a->ehi + rhi;
		if (all != L_FALSE) {
			w_const(&k, FPV_EW, (uint64_t)q);
			w_mux(g, &e2, all, &k, &e2);
			lo = q < lo ? q : lo;
			hi = q > hi ? q : hi;
		}
	}
	o.e = e2;
	tomax = L_FALSE;
	toinf = L_FALSE;
	if (p > 0) {
		/* the value reaches 2^(emax + 1): some bit j with e2 + j >
		 * emax */
		for (j = 0; j <= K; j++) {
			up = aig_or(g, up, L_NOT(e_lt(g, &e2, lo, hi, emax - j +
			    1)));
			ovf = aig_or(g, ovf, aig_and(g, o.m[j], up));
		}
		toinf = mode == FR_RNE ? L_TRUE : mode == FR_RTZ ? L_FALSE :
		    mode == FR_RUP ? L_NOT(a->s) : a->s;
		tomax = aig_and(g, ovf, L_NOT(toinf));
		if (tomax != L_FALSE) {
			/* the largest finite value, (2^p - 1) 2^(emax - p + 1) */
			for (j = o.n; j < p; j++)
				o.m[j] = L_FALSE;
			if (o.n < p)
				o.n = p;
			for (j = 0; j < o.n; j++)
				o.m[j] = aig_mux(g, tomax, j < p ? L_TRUE :
				    L_FALSE, o.m[j]);
			w_const(&k, FPV_EW, (uint64_t)(emax - p + 1));
			w_mux(g, &o.e, tomax, &k, &o.e);
			lo = emax - p + 1 < lo ? emax - p + 1 : lo;
		}
		/* a finite result is below 2^(emax + 1) */
		hi = hi > emax ? emax : hi;
		if (lo > hi)
			lo = hi;
	}
	o.inf = aig_or(g, a->inf, aig_and(g, ovf, toinf));
	for (j = 0; j < o.n; j++)
		nz = aig_or(g, nz, o.m[j]);
	exact = aig_and(g, L_NOT(dr[0]), L_NOT(sticky));
	o.z = aig_mux(g, exact, a->z, L_NOT(nz));
	set_bounds(&o, lo, hi);
	trim(&o);
	*r = o;
}

void
fp_round(struct aig *g, struct fpv *r, const struct fpv *a,
    const struct fpfmt *F, int mode)
{
	struct word nb;

	round_core(g, r, a, L_FALSE, F->p, qmin(F), bias(F), mode);
	fp_anynan(g, &nb, F);
	r->nb = nb;
}

/*
 * A total order on unpacked values by their literals (the NaN payload
 * aside): operands that commute are put in it, so a + b and b + a, a * b
 * and b * a build one circuit, not two the solver must show equal.
 */
static int
after(const struct fpv *a, const struct fpv *b)
{
	int i;

	if (a->n != b->n)
		return a->n > b->n;
	for (i = 0; i < a->n; i++)
		if (a->m[i] != b->m[i])
			return a->m[i] > b->m[i];
	for (i = 0; i < FPV_EW; i++)
		if (a->e.b[i] != b->e.b[i])
			return a->e.b[i] > b->e.b[i];
	if (a->s != b->s)
		return a->s > b->s;
	if (a->nan != b->nan)
		return a->nan > b->nan;
	if (a->inf != b->inf)
		return a->inf > b->inf;
	return a->z > b->z;
}

/* m moved up to put its leading one at bit W - 1 (W >= n), e down as much */
static void
normalize(struct aig *g, struct fpv *r, const struct fpv *a, int W)
{
	lit L[FPV_MW], x[FPV_MW];
	struct word j, c, sh;
	int i, k, jlo, jhi;

	lead(g, L, a->m, a->n, &jlo, &jhi);
	onehot_bin(g, &j, L, a->n);
	w_const(&c, FPV_EW, (uint64_t)(W - 1));
	w_sub(g, &sh, &c, &j);
	for (i = 0; i < W; i++)
		x[i] = i < a->n ? a->m[i] : L_FALSE;
	for (k = 0; (1 << k) < W; k++) {
		int d = 1 << k;
		for (i = W - 1; i >= 0; i--)
			x[i] = aig_mux(g, sh.b[k], i >= d ? x[i - d] : L_FALSE,
			    x[i]);
	}
	*r = *a;
	r->n = W;
	memcpy(r->m, x, (size_t)W * sizeof *x);
	w_sub(g, &r->e, &a->e, &sh);
	set_bounds(r, (long)a->elo - (W - 1 - jlo), (long)a->ehi - (W - 1 -
	    jhi));
}

/*
 * a + b exact, then rounded: the classic adder with a guard, a round and
 * a sticky position.  Both are normalized to W >= p bits; X has the
 * larger exponent (a zero is never X unless both are), Y moves right by
 * d = eX - eY with what falls off ORed into its bit 0.  When bits fell
 * (d >= 4), |X| > 2^(W + 2) and |Y| < 2^(W + 1) in units of the frame, so
 * the sum's leading one is at W + 1 or above and rounding to p <= W bits
 * drops at least two positions: the ORed bit is below the guard, and the
 * sum's floor and parity give the correct rounding (Y's dropped part is
 * strictly between 0 and one unit).  An exact zero sum is +0, or -0
 * rounding down, unless both operands have the same sign.
 */
static void
addc(struct aig *g, struct fpv *r, const struct fpv *a, const struct fpv *b,
    const struct fpfmt *F, int mode)
{
	struct fpv A, B, X, Y, s;
	int W = a->n, i, k, n3;
	lit swap, sticky = L_FALSE, big, same, neg, zero, zs, sign, c, cb;
	lit nanr, infr, sinf;
	lit x3[FPV_MW], y3[FPV_MW], sum[FPV_MW], dif[FPV_MW];
	struct word d;
	long dlo, dhi;

	if (after(a, b)) {
		const struct fpv *t = a;
		a = b;
		b = t;
	}
	W = a->n;
	if (b->n > W)
		W = b->n;
	if (F->p > W)
		W = F->p;
	if (W + 4 > FPV_MW) {
		toolarge("too wide a float significand");
		W = FPV_MW - 4;
	}
	normalize(g, &A, a, W);
	normalize(g, &B, b, W);
	swap = aig_or(g, a->z, aig_and(g, L_NOT(b->z), ee_lt(g, &A.e, A.elo,
	    A.ehi, &B.e, B.elo, B.ehi)));
	fp_mux(g, &X, swap, &B, &A);
	fp_mux(g, &Y, swap, &A, &B);
	w_sub(g, &d, &X.e, &Y.e);
	dlo = (long)X.elo - Y.ehi;
	dhi = (long)X.ehi - Y.elo;
	n3 = W + 3;
	for (i = 0; i < n3; i++) {
		x3[i] = i < 3 ? L_FALSE : X.m[i - 3];
		y3[i] = i < 3 || i - 3 >= Y.n ? L_FALSE : Y.m[i - 3];
	}
	for (k = 0; (1 << k) < n3; k++) {
		int sh = 1 << k;
		lit out = L_FALSE;
		for (i = 0; i < sh; i++)
			out = aig_or(g, out, y3[i]);
		sticky = aig_or(g, sticky, aig_and(g, d.b[k], out));
		for (i = 0; i < n3; i++)
			y3[i] = aig_mux(g, d.b[k], i + sh < n3 ? y3[i + sh] :
			    L_FALSE, y3[i]);
	}
	/* d past the stages: all of Y falls */
	big = L_NOT(e_lt(g, &d, dlo, dhi, 1L << k));
	if (big != L_FALSE) {
		lit any = L_FALSE;
		for (i = 0; i < Y.n; i++)
			any = aig_or(g, any, Y.m[i]);
		sticky = aig_mux(g, big, any, sticky);
		for (i = 0; i < n3; i++)
			y3[i] = aig_and(g, y3[i], L_NOT(big));
	}
	if (mut != 4)
		y3[0] = aig_or(g, y3[0], sticky);
	same = L_NOT(aig_xor(g, X.s, Y.s));
	c = L_FALSE;
	cb = L_TRUE;
	for (i = 0; i <= n3; i++) {
		lit xi = i < n3 ? x3[i] : L_FALSE, yi = i < n3 ? y3[i] : L_FALSE;
		lit t = aig_xor(g, xi, yi), u = L_NOT(t), p, q;
		sum[i] = aig_xor(g, t, c);
		p = aig_and(g, xi, yi);
		q = aig_and(g, t, c);
		c = aig_or(g, p, q);
		dif[i] = aig_xor(g, u, cb);
		p = aig_and(g, xi, L_NOT(yi));
		q = aig_and(g, u, cb);
		cb = aig_or(g, p, q);
	}
	neg = dif[n3];
	s.n = n3 + 1;
	c = L_TRUE;
	zero = L_TRUE;
	for (i = 0; i <= n3; i++) {
		lit t = L_NOT(dif[i]), nd = aig_xor(g, t, c);
		c = aig_and(g, t, c);
		s.m[i] = aig_mux(g, same, sum[i], aig_mux(g, neg, nd, dif[i]));
		zero = aig_and(g, zero, L_NOT(s.m[i]));
	}
	sign = aig_mux(g, same, X.s, aig_mux(g, neg, Y.s, X.s));
	zs = mut == 5 ? L_TRUE : aig_mux(g, aig_xor(g, a->s, b->s),
	    mode == FR_RDN ? L_TRUE : L_FALSE, a->s);
	s.s = aig_mux(g, zero, zs, sign);
	s.z = zero;
	s.nan = L_FALSE;
	s.inf = L_FALSE;
	w_const(&s.nb, fwidth(F), 0);
	{
		struct word three;
		w_const(&three, FPV_EW, 3);
		w_sub(g, &s.e, &X.e, &three);
	}
	set_bounds(&s, (long)X.elo - 3, (long)X.ehi - 3);
	trim(&s);
	/* r may be a or b */
	nanr = aig_or(g, a->nan, b->nan);
	infr = aig_and(g, a->inf, b->inf);
	sinf = aig_xor(g, a->s, b->s);
	nanr = aig_or(g, nanr, aig_and(g, infr, sinf));
	infr = aig_or(g, a->inf, b->inf);
	sinf = aig_mux(g, a->inf, a->s, b->s);
	round_core(g, r, &s, L_FALSE, F->p, qmin(F), bias(F), mode);
	r->nan = nanr;
	r->s = aig_mux(g, infr, sinf, r->s);
	r->inf = aig_or(g, infr, r->inf);
	fp_anynan(g, &r->nb, F);
}

void
fp_add(struct aig *g, struct fpv *r, const struct fpv *a, const struct fpv *b,
    const struct fpfmt *F, int mode)
{
	addc(g, r, a, b, F, mode);
}

/* a zero that is not an infinity's leftover */
static lit
fzero(struct aig *g, const struct fpv *a)
{
	return aig_and(g, a->z, L_NOT(a->inf));
}

/* the exact product: the integers' product, the exponents' sum */
static void
mulx(struct aig *g, struct fpv *r, const struct fpv *a, const struct fpv *b)
{
	int na, nb, n, i, j;
	lit acc[FPV_MW];

	if (after(a, b)) {
		const struct fpv *t = a;
		a = b;
		b = t;
	}
	na = a->n;
	nb = b->n;
	n = na + nb;
	if (n > FPV_MW) {
		toolarge("too wide a float significand");
		n = FPV_MW;
		if (na >= n)
			na = n - 1;
		nb = n - na;
	}
	for (i = 0; i < n; i++)
		acc[i] = L_FALSE;
	for (i = 0; i < nb; i++) {
		lit c = L_FALSE;
		if (b->m[i] == L_FALSE)
			continue;
		for (j = 0; j < na && i + j < n; j++) {
			lit pp = aig_and(g, a->m[j], b->m[i]), x = acc[i + j];
			lit t = aig_xor(g, x, pp), u, v;
			acc[i + j] = aig_xor(g, t, c);
			u = aig_and(g, x, pp);
			v = aig_and(g, t, c);
			c = aig_or(g, u, v);
		}
		for (j = i + na; j < n && c != L_FALSE; j++) {
			lit x = acc[j];
			acc[j] = aig_xor(g, x, c);
			c = aig_and(g, x, c);
		}
	}
	r->s = aig_xor(g, a->s, b->s);
	memcpy(r->m, acc, (size_t)n * sizeof *acc);
	r->n = n;
	w_add(g, &r->e, &a->e, &b->e);
	set_bounds(r, (long)a->elo + b->elo, (long)a->ehi + b->ehi);
	r->z = aig_or(g, a->z, b->z);
	r->nan = aig_or(g, a->nan, b->nan);
	if (mut != 6) {
		lit u = aig_and(g, a->inf, fzero(g, b)), v;
		v = aig_and(g, fzero(g, a), b->inf);
		r->nan = aig_or(g, r->nan, aig_or(g, u, v));
	}
	r->inf = aig_or(g, a->inf, b->inf);
	r->nb = a->nb;
	trim(r);
}

void
fp_mul(struct aig *g, struct fpv *r, const struct fpv *a, const struct fpv *b,
    const struct fpfmt *F, int mode)
{
	struct fpv p;

	mulx(g, &p, a, b);
	fp_round(g, r, &p, F, mode);
}

/*
 * a * b + c rounded once: the exact product (its width the significands'
 * sum, which the adder's frame holds, so the adder's argument for its
 * sticky bit holds with W the product's width) into the adder.  NaN for
 * inf * 0 and for an infinite product plus the opposite infinity, as
 * the adder and the product have them.
 */
void
fp_fma(struct aig *g, struct fpv *r, const struct fpv *a, const struct fpv *b,
    const struct fpv *c, const struct fpfmt *F, int mode)
{
	struct fpv p, q;

	mulx(g, &p, a, b);
	if (mut == 9) {
		fp_round(g, &q, &p, F, mode);
		p = q;
	}
	addc(g, r, &p, c, F, mode);
}

/*
 * a rounded to an integral value by mode (floor: down, ceil: up, a
 * conversion to an integer: toward zero): the precision has no limit and
 * the floor is 2^0.  A value of 2^(p-1) or more is already integral, and
 * what rounds up stays below 2^p, so the result is one of F's.
 */
void
fp_rint(struct aig *g, struct fpv *r, const struct fpv *a,
    const struct fpfmt *F, int mode)
{
	if (mut == 12 && mode == FR_RDN)
		mode = FR_RTZ;
	round_core(g, r, a, L_FALSE, 0, 0, 0, mode);
	fp_anynan(g, &r->nb, F);
}

/*
 * a truncated toward zero, as a 64-bit two's complement word; the
 * literal returned is the evaluator's "defined": a number in [-2^63,
 * 2^63) (eval.c, real_to_int).  The integral value m 2^e (e >= 0 unless
 * it is 0) is m moved up by e; past bit 63, or e >= 64, it is out.
 */
lit
fp_to_int(struct aig *g, struct word *r, const struct fpv *a)
{
	struct fpv t;
	struct word mag, neg;
	lit x[FPV_MW + 64], big, over = L_FALSE, low = L_TRUE, nz = L_FALSE, ok;
	int n, i, k, W;

	round_core(g, &t, a, L_FALSE, 0, 0, 0, FR_RTZ);
	n = t.n;
	W = n + 64;
	for (i = 0; i < W; i++)
		x[i] = i < n ? t.m[i] : L_FALSE;
	for (i = 0; i < n; i++)
		nz = aig_or(g, nz, t.m[i]);
	for (k = 0; k < 6; k++) {
		int d = 1 << k;
		for (i = W - 1; i >= 0; i--)
			x[i] = aig_mux(g, t.e.b[k], i >= d ? x[i - d] : L_FALSE,
			    x[i]);
	}
	big = L_NOT(e_lt(g, &t.e, t.elo, t.ehi, 64));
	for (i = 64; i < W; i++)
		over = aig_or(g, over, x[i]);
	over = aig_or(g, over, aig_and(g, big, nz));
	mag.w = 64;
	for (i = 0; i < 64; i++) {
		mag.b[i] = x[i];
		if (i < 63)
			low = aig_and(g, low, L_NOT(x[i]));
	}
	/* |v| < 2^63, or v = -2^63 */
	ok = aig_or(g, L_NOT(x[63]), aig_and(g, t.s, mut == 10 ? L_TRUE : low));
	ok = aig_and(g, L_NOT(over), ok);
	w_neg(g, &neg, &mag);
	w_mux(g, r, t.s, &neg, &mag);
	ok = aig_and(g, ok, L_NOT(a->inf));
	return aig_and(g, ok, L_NOT(a->nan));
}

/* an integer word v (two's complement when sgn) rounded to F by mode */
void
fp_from_int(struct aig *g, struct fpv *r, const struct word *v, int sgn,
    const struct fpfmt *F, int mode)
{
	struct fpv x;
	struct word mag, neg;
	lit nz = L_FALSE;
	int i;

	x.s = sgn ? v->b[v->w - 1] : L_FALSE;
	w_neg(g, &neg, v);
	w_mux(g, &mag, x.s, &neg, v);
	x.nan = L_FALSE;
	x.inf = L_FALSE;
	x.n = v->w;
	for (i = 0; i < v->w; i++) {
		x.m[i] = mag.b[i];
		nz = aig_or(g, nz, mag.b[i]);
	}
	x.z = L_NOT(nz);
	w_const(&x.e, FPV_EW, 0);
	x.elo = x.ehi = 0;
	w_const(&x.nb, fwidth(F), 0);
	trim(&x);
	round_core(g, r, &x, L_FALSE, F->p, qmin(F), bias(F), mut == 11 ?
	    FR_RTZ : mode);
	fp_anynan(g, &r->nb, F);
}

static void
classify(struct aig *g, const struct word *x, const struct fpfmt *F,
    lit *nan, lit *zero, struct word *mag)
{
	int fb = F->p - 1, i;
	lit e1 = L_TRUE, fz = L_TRUE;

	for (i = 0; i < fb; i++)
		fz = aig_and(g, fz, L_NOT(x->b[i]));
	for (i = 0; i < F->eb; i++)
		e1 = aig_and(g, e1, x->b[fb + i]);
	*nan = aig_and(g, e1, L_NOT(fz));
	mag->w = x->w - 1;
	*zero = L_TRUE;
	for (i = 0; i < mag->w; i++) {
		mag->b[i] = x->b[i];
		*zero = aig_and(g, *zero, L_NOT(x->b[i]));
	}
}

lit
fp_lt(struct aig *g, const struct word *a, const struct word *b,
    const struct fpfmt *F)
{
	struct word ma, mb;
	lit na, nb, za, zb, sa = a->b[a->w - 1], sb = b->b[b->w - 1], lt, neg;
	lit pos;

	classify(g, a, F, &na, &za, &ma);
	classify(g, b, F, &nb, &zb, &mb);
	neg = aig_and(g, sa, L_NOT(sb));
	if (mut != 8)
		neg = aig_and(g, neg, L_NOT(aig_and(g, za, zb)));
	/* one node at a time: C leaves the order of arguments open, and the
	 * graph's numbering (so the solver's search) would follow the
	 * compiler */
	pos = aig_and(g, L_NOT(sa), L_NOT(sb));
	pos = aig_and(g, pos, w_ult(g, &ma, &mb));
	lt = aig_and(g, sa, sb);
	lt = aig_and(g, lt, w_ult(g, &mb, &ma));
	lt = aig_or(g, neg, aig_or(g, pos, lt));
	return aig_and(g, lt, L_NOT(aig_or(g, na, nb)));
}

lit
fp_eq(struct aig *g, const struct word *a, const struct word *b,
    const struct fpfmt *F)
{
	struct word ma, mb;
	lit na, nb, za, zb, any, eq;

	classify(g, a, F, &na, &za, &ma);
	classify(g, b, F, &nb, &zb, &mb);
	any = aig_or(g, na, nb);
	eq = w_eq(g, a, b);

	eq = aig_or(g, eq, aig_and(g, za, zb));
	return aig_and(g, L_NOT(any), eq);
}

lit
fp_same(struct aig *g, const struct word *a, const struct word *b,
    const struct fpfmt *F)
{
	struct word ma, mb;
	lit na, nb, za, zb, both;

	classify(g, a, F, &na, &za, &ma);
	classify(g, b, F, &nb, &zb, &mb);
	both = aig_and(g, na, nb);
	return aig_or(g, both, fp_eq(g, a, b, F));
}

void
fp_minmax(struct aig *g, struct fpv *r, const struct fpv *a,
    const struct word *pa, const struct fpv *b, const struct word *pb,
    int max, const struct fpfmt *F)
{
	struct word ma, mb, nb;
	lit na, nbn, za, zb, better, tie, pickb, sig;
	int q;

	classify(g, pa, F, &na, &za, &ma);
	classify(g, pb, F, &nbn, &zb, &mb);
	better = max ? fp_lt(g, pa, pb, F) : fp_lt(g, pb, pa, F);
	tie = aig_and(g, za, zb);
	tie = aig_and(g, tie, aig_xor(g, pa->b[pa->w - 1], pb->b[pb->w - 1]));
	tie = aig_and(g, tie, aig_input(g));
	pickb = aig_or(g, better, tie);
	pickb = aig_or(g, na, aig_and(g, L_NOT(nbn), pickb));
	fp_mux(g, r, pickb, b, a);
	fp_anynan(g, &nb, F);
	w_mux(g, &r->nb, aig_and(g, na, nbn), &nb, &r->nb);
	/* a signaling NaN operand: glibc's fmin gives a NaN, FreeBSD's the
	 * other operand */
	q = F->p - 2;
	sig = aig_and(g, na, L_NOT(pa->b[q]));
	sig = aig_or(g, sig, aig_and(g, nbn, L_NOT(pb->b[q])));
	if (sig != L_FALSE) {
		struct fpv x = *r;
		x.nan = L_TRUE;
		fp_anynan(g, &x.nb, F);
		fp_mux(g, r, aig_and(g, sig, aig_input(g)), &x, r);
	}
}

/* ---- self test ---- */

static uint64_t
rng(uint64_t *s)
{
	*s ^= *s << 13;
	*s ^= *s >> 7;
	*s ^= *s << 17;
	return *s;
}

/* the value of F's bits (any format a double holds exactly) */
static double
ref_val(uint64_t b, const struct fpfmt *F)
{
	int fb = F->p - 1, eb = F->eb;
	uint64_t frac = b & ((1ULL << fb) - 1), ef = b >> fb & ((1ULL << eb) - 1);
	double v;

	if (ef == (1ULL << eb) - 1)
		v = frac ? NAN : INFINITY;
	else if (ef)
		v = ldexp((double)(frac | 1ULL << fb), (int)ef - bias(F) - fb);
	else
		v = ldexp((double)frac, qmin(F));
	return b >> (fb + eb) & 1 ? -v : v;
}

/*
 * The exact value v rounded to F by mode, as F's bits (NaN: the quiet
 * one).  As fround16: q = |v| / u, u the spacing of F's values around
 * |v|, is exact, so are floor(q) and the rest d.
 */
static uint64_t
ref_bits(double v, const struct fpfmt *F, int mode)
{
	int fb = F->p - 1, eb = F->eb, e, emin = 1 - bias(F), neg = signbit(v) != 0;
	uint64_t sign = (uint64_t)neg << (fb + eb), inf = ((1ULL << eb) - 1) << fb;
	double a = fabs(v), u, q, f, d, top = ldexp(2.0, bias(F));

	if (isnan(v))
		return inf | 1ULL << (fb - 1);
	if (isinf(a))
		return sign | inf;
	if (a == 0)
		return sign;
	(void)frexp(a, &e);
	e = e - 1 < emin ? emin : e - 1;
	u = ldexp(1.0, e - fb);
	q = a / u;
	f = floor(q);
	d = q - f;
	if ((mode == FR_RNE && (d > 0.5 || (d == 0.5 && fmod(f, 2.0) != 0))) ||
	    (mode == FR_RUP && !neg && d > 0) || (mode == FR_RDN && neg && d > 0))
		f += 1;
	a = f * u;
	if (a >= top) {
		int toinf = mode == FR_RNE || (mode == FR_RUP && !neg) ||
		    (mode == FR_RDN && neg);
		if (toinf)
			return sign | inf;
		return sign | (inf - (1ULL << fb)) | ((1ULL << fb) - 1);
	}
	if (a < ldexp(1.0, emin))
		return sign | (uint64_t)ldexp(a, -qmin(F));
	(void)frexp(a, &e);
	return sign | (uint64_t)(e - 1 + bias(F)) << fb |
	    ((uint64_t)ldexp(a, fb - (e - 1)) & ((1ULL << fb) - 1));
}

/* the exact sum's rounding, with IEEE's sign of an exact zero */
static uint64_t
ref_add(double x, double y, const struct fpfmt *F, int mode)
{
	double s = x + y;

	if (s == 0 && !isnan(s))
		s = signbit(x) == signbit(y) ? x : mode == FR_RDN ? -0.0 : 0.0;
	return ref_bits(s, F, mode);
}

/* the constant a word folded to; 0 when some bit is not constant */
static int
wval(const struct word *x, uint64_t *v)
{
	int i;

	*v = 0;
	for (i = 0; i < x->w; i++) {
		if (x->b[i] != L_FALSE && x->b[i] != L_TRUE)
			return 0;
		*v |= (uint64_t)(x->b[i] == L_TRUE) << i;
	}
	return 1;
}

static int
isnanb(uint64_t b, const struct fpfmt *F)
{
	int fb = F->p - 1;
	uint64_t inf = ((1ULL << F->eb) - 1) << fb;

	return (b & inf) == inf && (b & ((1ULL << fb) - 1));
}

enum { FT_ADD, FT_SUB, FT_MUL, FT_MIN, FT_MAX, FT_LT, FT_EQ, FT_CVT, FT_MA,
    FT_AM, FT_FMA, FT_NOPS };

/* the bits a value of F packs to: a NaN as the quiet one, sign 0 */
static uint64_t
canon(uint64_t v, const struct fpfmt *F)
{
	int fb = F->p - 1;

	return isnanb(v, F) ? ((1ULL << F->eb) - 1) << fb | 1ULL << (fb - 1) : v;
}

/*
 * Can op on constant operands give `want`?  The circuits fold to
 * constants but for their free inputs (a NaN's payload, the zero that
 * min(+0, -0) is, a signaling NaN's fmin): the solver looks for them.
 * FT_CVT converts a to format G; a comparison gives 0 or 1.  FT_MA is
 * a * b + a and FT_AM (a + b) * b, each operation's result fed to the
 * next unpacked, as nodesat.c chains them.
 */
static uint64_t fma_c;	/* FT_FMA's addend */

static int
cop_can(int op, uint64_t xa, uint64_t xb, const struct fpfmt *F,
    const struct fpfmt *G, int mode, uint64_t want)
{
	struct aig *g = aig_new();
	struct word wa, wb, wr, k;
	struct fpv a, b, r, c;
	int ok;

	w_const(&wa, fwidth(F), xa);
	w_const(&wb, fwidth(F), xb);
	fp_unpack(g, &a, &wa, F);
	fp_unpack(g, &b, &wb, F);
	switch (op) {
	case FT_FMA:
		w_const(&wr, fwidth(F), fma_c);
		fp_unpack(g, &c, &wr, F);
		fp_fma(g, &r, &a, &b, &c, F, mode);
		break;
	case FT_ADD:
	case FT_SUB:
		if (op == FT_SUB)
			b.s = L_NOT(b.s);
		fp_add(g, &r, &a, &b, F, mode);
		break;
	case FT_MUL:
		fp_mul(g, &r, &a, &b, F, mode);
		break;
	case FT_MA:
		fp_mul(g, &r, &a, &b, F, mode);
		fp_add(g, &r, &r, &a, F, mode);
		break;
	case FT_AM:
		fp_add(g, &r, &a, &b, F, mode);
		fp_mul(g, &r, &r, &b, F, mode);
		break;
	case FT_MIN:
	case FT_MAX:
		fp_minmax(g, &r, &a, &wa, &b, &wb, op == FT_MAX, F);
		break;
	case FT_LT:
	case FT_EQ:
		w_const(&wr, 1, 0);
		wr.b[0] = op == FT_LT ? fp_lt(g, &wa, &wb, F) : fp_eq(g, &wa, &wb,
		    F);
		w_const(&k, 1, want);
		ok = aig_sat(g, w_eq(g, &wr, &k), 0) == 1;
		aig_free(g);
		return ok;
	default:
		fp_round(g, &r, &a, G, mode);
		F = G;
		break;
	}
	fp_pack(g, &wr, &r, F, 0);
	w_const(&k, fwidth(F), canon(want, F));
	ok = aig_sat(g, w_eq(g, &wr, &k), 0) == 1;
	aig_free(g);
	return ok;
}

/* to an integral value: can a constant's rounding give want? */
static int
rint_can(uint64_t x, const struct fpfmt *F, int mode, uint64_t want)
{
	struct aig *g = aig_new();
	struct word wx, wr, k;
	struct fpv a, r;
	int ok;

	w_const(&wx, fwidth(F), x);
	fp_unpack(g, &a, &wx, F);
	fp_rint(g, &r, &a, F, mode);
	fp_pack(g, &wr, &r, F, 0);
	w_const(&k, fwidth(F), canon(want, F));
	ok = aig_sat(g, w_eq(g, &wr, &k), 0) == 1;
	aig_free(g);
	return ok;
}

/* to an integer: a constant's conversion is undefined (def 0) or v */
static int
toint_can(uint64_t x, const struct fpfmt *F, int def, int64_t v)
{
	struct aig *g = aig_new();
	struct word wx, wr, k;
	struct fpv a;
	lit ok, goal;
	int r;

	w_const(&wx, fwidth(F), x);
	fp_unpack(g, &a, &wx, F);
	ok = fp_to_int(g, &wr, &a);
	w_const(&k, 64, (uint64_t)v);
	goal = def ? aig_and(g, ok, w_eq(g, &wr, &k)) : L_NOT(ok);
	r = aig_sat(g, goal, 0) == 1;
	aig_free(g);
	return r;
}

/* from an integer of w bits (two's complement when sgn) */
static int
fromint_can(uint64_t v, int w, int sgn, const struct fpfmt *F, int mode,
    uint64_t want)
{
	struct aig *g = aig_new();
	struct word wv, wr, k;
	struct fpv r;
	int ok;

	w_const(&wv, w, v);
	fp_from_int(g, &r, &wv, sgn, F, mode);
	fp_pack(g, &wr, &r, F, 0);
	w_const(&k, fwidth(F), canon(want, F));
	ok = aig_sat(g, w_eq(g, &wr, &k), 0) == 1;
	aig_free(g);
	return ok;
}

/*
 * Every operation of a 6-bit format (3 exponent bits, 3 significand
 * bits: all 4096 pairs, every mode) against exact arithmetic in double
 * rounded by ref_bits.  Returns the mismatches.
 */
static int
tiny_exhaustive(int verbose)
{
	static const struct fpfmt T = {3, 3}, T2 = {4, 5};
	int bad = 0, op, mode;
	uint64_t x, y, want;

	for (x = 0; x < 64; x++)
		for (y = 0; y < 64; y++) {
			double a = ref_val(x, &T), b = ref_val(y, &T);
			for (op = FT_ADD; op <= FT_EQ; op++)
				for (mode = 0; mode < 4; mode++) {
					if (op >= FT_MIN && mode)
						continue;
					switch (op) {
					case FT_ADD: want = ref_add(a, b, &T, mode); break;
					case FT_SUB: want = ref_add(a, -b, &T, mode); break;
					case FT_MUL: want = ref_bits(a * b, &T, mode); break;
					case FT_MIN: want = ref_bits(fmin(a, b), &T, mode);
						break;
					case FT_MAX: want = ref_bits(fmax(a, b), &T, mode);
						break;
					case FT_LT: want = a < b; break;
					default: want = a == b; break;
					}
					if (cop_can(op, x, y, &T, &T, mode, want))
						continue;
					if (verbose && bad < 8)
						fprintf(stderr, "fp selftest: 6-bit op %d "
						    "mode %d on %02llx, %02llx: not %llx\n",
						    op, mode, (unsigned long long)x,
						    (unsigned long long)y,
						    (unsigned long long)want);
					bad++;
				}
			/* conversions to a wider format */
			for (mode = 0; mode < 4 && y == 0; mode++) {
				want = ref_bits(a, &T2, mode);
				if (!cop_can(FT_CVT, x, 0, &T, &T2, mode, want)) {
					if (verbose && bad < 8)
						fprintf(stderr, "fp selftest: widen %02llx: "
						    "not %llx\n", (unsigned long long)x,
						    (unsigned long long)want);
					bad++;
				}
			}
		}
	/* fma on random triples: a * b exact in a double, so is + c */
	{
		uint64_t seed = 99;
		int i;
		for (i = 0; i < 6000; i++) {
			uint64_t z = rng(&seed) & 63;
			double a, b, c, p;
			x = rng(&seed) & 63;
			y = rng(&seed) & 63;
			a = ref_val(x, &T);
			b = ref_val(y, &T);
			c = ref_val(z, &T);
			p = a * b;
			mode = i & 3;
			want = ref_add(p, c, &T, mode);
			fma_c = z;
			if (!cop_can(FT_FMA, x, y, &T, &T, mode, want)) {
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: 6-bit fma mode %d "
					    "on %02llx, %02llx, %02llx: not %llx\n", mode,
					    (unsigned long long)x, (unsigned long long)y,
					    (unsigned long long)z,
					    (unsigned long long)want);
				bad++;
			}
		}
	}
	/* to integral values, to and from integers */
	for (x = 0; x < 64; x++) {
		double a = ref_val(x, &T), v;
		for (mode = 0; mode < 4; mode++) {
			v = mode == FR_RNE ? nearbyint(a) : mode == FR_RTZ ? trunc(a) :
			    mode == FR_RUP ? ceil(a) : floor(a);
			if (!rint_can(x, &T, mode, ref_bits(v, &T, FR_RNE))) {
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: 6-bit rint %02llx "
					    "mode %d\n", (unsigned long long)x, mode);
				bad++;
			}
		}
		if (!toint_can(x, &T, isfinite(a), isfinite(a) ? (int64_t)a : 0)) {
			if (verbose && bad < 8)
				fprintf(stderr, "fp selftest: 6-bit to int %02llx\n",
				    (unsigned long long)x);
			bad++;
		}
	}
	for (y = 0; y < 601; y++)
		for (mode = 0; mode < 4; mode++) {
			int64_t v = (int64_t)y - 300;
			if (!fromint_can((uint64_t)v & 0xffff, 16, 1, &T, mode,
			    ref_bits((double)v, &T, mode)) || !fromint_can(y, 16, 0,
			    &T, mode, ref_bits((double)y, &T, mode))) {
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: 6-bit from int %lld "
					    "mode %d\n", (long long)v, mode);
				bad++;
			}
		}
	/* and to a narrower one */
	for (x = 0; x < 512; x++)
		for (mode = 0; mode < 4; mode++) {
			want = ref_bits(ref_val(x, &T2), &T, mode);
			if (!cop_can(FT_CVT, x, 0, &T2, &T, mode, want)) {
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: narrow %03llx mode "
					    "%d: not %llx\n", (unsigned long long)x, mode,
					    (unsigned long long)want);
				bad++;
			}
		}
	return bad;
}

/* edge values of format F: zeros, subnormals, normals, the top, inf, NaN */
static int
edges(uint64_t *e, const struct fpfmt *F)
{
	int fb = F->p - 1, n = 0, s;
	uint64_t one = (uint64_t)bias(F) << fb, inf = ((1ULL << F->eb) - 1) << fb;
	uint64_t mx = inf - 1, mn = 1ULL << fb;

	for (s = 0; s < 2; s++) {
		uint64_t sg = (uint64_t)s << (fb + F->eb);
		e[n++] = sg;
		e[n++] = sg | 1;
		e[n++] = sg | (mn - 1);
		e[n++] = sg | mn;
		e[n++] = sg | (mn + 1);
		e[n++] = sg | one;
		e[n++] = sg | (one + 1);
		e[n++] = sg | (one - 1);
		e[n++] = sg | mx;
		e[n++] = sg | (mx - 1);
		e[n++] = sg | inf;
		e[n++] = sg | inf | 1ULL << (fb - 1);
		e[n++] = sg | inf | 5;
		e[n++] = sg | (one + (3ULL << fb));
		e[n++] = sg | (one - (2ULL << fb)) | 3;
	}
	return n;
}

/*
 * The circuits against eval.c's arithmetic (scalar_float, fround) in each
 * width: edge pairs and random ones, the random ones near each other too
 * (cancellation) and far apart (alignment).  The evaluator holds a value
 * as a double, so an input is the bits the evaluator would give back.
 */
static int
against_eval(int verbose)
{
	static const int ws[3] = {16, 32, 64};
	int wi, bad = 0;
	uint64_t seed = 0x5eed1234abcdULL;

	for (wi = 0; wi < 3; wi++) {
		int w = ws[wi], ne, i, op, npairs;
		const struct fpfmt *F = fp_fmt(w);
		uint64_t ed[32], mask = w == 64 ? ~0ULL : (1ULL << w) - 1;

		ne = edges(ed, F);
		npairs = ne * ne + (w == 64 ? 300 : 600);
		for (i = 0; i < npairs; i++) {
			uint64_t x, y, want;
			double a, b, r;
			uint8_t fl = 0;
			if (i < ne * ne) {
				x = ed[i / ne];
				y = ed[i % ne];
			} else {
				x = rng(&seed) & mask;
				y = (i & 3) == 0 ? (x + (rng(&seed) & 7) - 3) & mask :
				    (i & 3) == 1 ? x ^ (rng(&seed) & (mask >> 3)) :
				    rng(&seed) & mask;
			}
			a = frombits(w, x);
			b = frombits(w, y);
			x = fbits(w, a);
			y = fbits(w, b);
			{
				/* fma: the evaluator's fma16, fma32, fma64 */
				uint64_t z = i < ne * ne ? ed[(i * 7) % ne] :
				    rng(&seed) & mask;
				double c = frombits(w, z), fr;
				z = fbits(w, c);
				fr = w == 16 ? fma16(a, b, c) : w == 32 ? fma32(a, b, c) :
				    fma64(a, b, c);
				want = fbits(w, fr);
				fma_c = z;
				if (!cop_can(FT_FMA, x, y, F, F, FR_RNE, isnan(fr) ?
				    ((1ULL << F->eb) - 1) << (F->p - 1) | 1 : want)) {
					if (verbose && bad < 8)
						fprintf(stderr, "fp selftest: f%d fma on %llx, "
						    "%llx, %llx: not %llx\n", w,
						    (unsigned long long)x, (unsigned long long)y,
						    (unsigned long long)z,
						    (unsigned long long)want);
					bad++;
				}
			}
			{
				/* floor, ceil, roundeven; to an integer */
				static const int rop[3] = {O_FLOOR, O_CEIL, O_ROUNDEVEN};
				static const int rmd[3] = {FR_RDN, FR_RUP, FR_RNE};
				int q, def = !isnan(a) && a < 9.2233720368547758e18 &&
				    a >= -9.2233720368547758e18;
				for (q = 0; q < 3; q++) {
					double rv = scalar_float(rop[q], w, a, 0, &fl);
					if (!rint_can(x, F, rmd[q], isnan(rv) ? canon(((1ULL <<
					    F->eb) - 1) << (F->p - 1) | 1, F) :
					    fbits(w, rv))) {
						if (verbose && bad < 8)
							fprintf(stderr, "fp selftest: f%d rint %d on "
							    "%llx\n", w, q, (unsigned long long)x);
						bad++;
					}
				}
				if (!toint_can(x, F, def, def ? (int64_t)a : 0)) {
					if (verbose && bad < 8)
						fprintf(stderr, "fp selftest: f%d to int on "
						    "%llx\n", w, (unsigned long long)x);
					bad++;
				}
			}
			for (op = FT_ADD; op <= FT_AM; op++) {
				static const int eop[] = {O_ADD, O_SUB, O_MUL, O_MIN,
				    O_MAX, O_LT, O_EQ};
				int w2 = w, ok;
				if (op == FT_MA || op == FT_AM) {
					double t = scalar_float(op == FT_MA ? O_MUL : O_ADD,
					    w, a, b, &fl);
					r = op == FT_MA ? scalar_float(O_ADD, w, t, a, &fl) :
					    scalar_float(O_MUL, w, t, b, &fl);
					want = fbits(w, r);
					ok = cop_can(op, x, y, F, F, FR_RNE, want);
				} else if (op == FT_CVT) {
					w2 = ws[(wi + 1 + (i & 1)) % 3];
					r = fround(w2, a);
					want = fbits(w2, r);
					ok = cop_can(op, x, 0, F, fp_fmt(w2), FR_RNE, want);
				} else {
					r = scalar_float(eop[op], w, a, b, &fl);
					want = op >= FT_LT ? (uint64_t)r : fbits(w, r);
					ok = cop_can(op, x, y, F, F, FR_RNE, want);
				}
				if (ok)
					continue;
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: f%d op %d (to f%d) on "
					    "%llx, %llx: not %llx\n", w, op, w2,
					    (unsigned long long)x, (unsigned long long)y,
					    (unsigned long long)want);
				bad++;
			}
		}
	}
	/* the ends of int64 for to_int; integers into each width */
	{
		static const uint64_t b32[] = {0x5f000000, 0x5effffff, 0xdf000000,
		    0xdf000001}, b64[] = {0x43e0000000000000ULL,
		    0x43dfffffffffffffULL, 0xc3e0000000000000ULL,
		    0xc3e0000000000001ULL};
		static const int ts[] = {8, 16, 32, 64};
		int i;
		for (i = 0; i < 4; i++) {
			double a = frombits(32, b32[i]), c = frombits(64, b64[i]);
			int da = a < 9.2233720368547758e18 && a >= -9.2233720368547758e18;
			int dc = c < 9.2233720368547758e18 && c >= -9.2233720368547758e18;
			if (!toint_can(b32[i], &fmt32, da, da ? (int64_t)a : 0) ||
			    !toint_can(b64[i], &fmt64, dc, dc ? (int64_t)c : 0)) {
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: to int at 2^63 (%d)\n",
					    i);
				bad++;
			}
		}
		for (i = 0; i < 400; i++) {
			int bits = ts[i % 4], sg = (i >> 2) & 1, wi = (i >> 3) % 3;
			int w = wi == 0 ? 16 : wi == 1 ? 32 : 64;
			uint64_t m = bits == 64 ? ~0ULL : (1ULL << bits) - 1;
			uint64_t v = i < 12 ? (1ULL << 60) + (1ULL << 36) + 1 :
			    rng(&seed) >> (rng(&seed) % 64);
			ty_t ti = TY(sg ? K_SINT : K_UINT, bits);
			int64_t sv;
			v &= m;
			sv = sg && bits < 64 && (v >> (bits - 1) & 1) ?
			    (int64_t)(v | ~m) : (int64_t)v;
			if (!fromint_can(v, bits, sg, fp_fmt(w), FR_RNE,
			    fbits(w, ifloat(ti, sv, w)))) {
				if (verbose && bad < 8)
					fprintf(stderr, "fp selftest: %s%d %llx to f%d\n",
					    sg ? "i" : "u", bits, (unsigned long long)v, w);
				bad++;
			}
		}
	}
	return bad;
}

/*
 * The solver on float miters: random expressions of the 6-bit format in
 * x and y and a near relative of each (one operation or its rounding
 * changed); every one of the 4096 inputs decides by exact arithmetic,
 * and the solver must agree.  Then two proofs that need no search: an
 * f32 product is the f64 one rounded once, and a conversion to a wider
 * format and back changes nothing.
 */
static int
sat_floats(int verbose)
{
	static const struct fpfmt T = {3, 3};
	uint64_t seed = 777;
	int trial, bad = 0;

	for (trial = 0; trial < 40; trial++) {
		int ops[4], modes[4], k, mut_at = (int)(rng(&seed) % 4), v, r;
		int differs = 0;
		uint64_t cs[4], xv, yv;
		struct aig *g;
		struct word wx, wy, out[2];
		struct fpv x, y;

		for (k = 0; k < 4; k++) {
			ops[k] = (int)(rng(&seed) % 3);
			modes[k] = (int)(rng(&seed) % 4);
			cs[k] = rng(&seed) & 63;
		}
		g = aig_new();
		w_input(g, &wx, 6);
		w_input(g, &wy, 6);
		fp_unpack(g, &x, &wx, &T);
		fp_unpack(g, &y, &wy, &T);
		for (v = 0; v < 2; v++) {
			struct fpv acc = x, c;
			for (k = 0; k < 4; k++) {
				int op = ops[k], md = modes[k];
				if (v && k == mut_at) {
					if (trial % 2)
						md = (md + 1) % 4;
					else
						op = (op + 1) % 3;
				}
				if (k % 2)
					fp_const(g, &c, cs[k], &T);
				else
					c = y;
				if (op == 2)
					fp_mul(g, &acc, &acc, &c, &T, md);
				else {
					if (op == 1)
						c.s = L_NOT(c.s);
					fp_add(g, &acc, &acc, &c, &T, md);
				}
			}
			fp_pack(g, &out[v], &acc, &T, 0);
		}
		for (xv = 0; xv < 64 && !differs; xv++)
			for (yv = 0; yv < 64 && !differs; yv++) {
				uint64_t res[2];
				for (v = 0; v < 2; v++) {
					uint64_t acc = xv;
					for (k = 0; k < 4; k++) {
						int op = ops[k], md = modes[k];
						double a, b;
						if (v && k == mut_at) {
							if (trial % 2)
								md = (md + 1) % 4;
							else
								op = (op + 1) % 3;
						}
						a = ref_val(acc, &T);
						b = ref_val(k % 2 ? cs[k] : yv, &T);
						acc = op == 2 ? ref_bits(a * b, &T, md) :
						    ref_add(a, op == 1 ? -b : b, &T, md);
					}
					res[v] = acc;
				}
				if (!(isnanb(res[0], &T) && isnanb(res[1], &T)) &&
				    !(ref_val(res[0], &T) == ref_val(res[1], &T)))
					differs = 1;
			}
		r = aig_sat(g, L_NOT(fp_same(g, &out[0], &out[1], &T)), 0);
		if (r != differs) {
			if (verbose)
				fprintf(stderr, "fp selftest: miter %d: solver %d, "
				    "exhaustive %d\n", trial, r, differs);
			bad++;
		}
		aig_free(g);
	}
	{
		/* L2: f32(u * v) is f32(f64(u) * f64(v)): no conflict at all */
		struct aig *g = aig_new();
		struct word wu, wv, p1, p2;
		struct fpv u, w, U, V, a, b, c;
		lit ok;
		w_input(g, &wu, 32);
		w_input(g, &wv, 32);
		fp_unpack(g, &u, &wu, &fmt32);
		fp_unpack(g, &w, &wv, &fmt32);
		fp_mul(g, &a, &u, &w, &fmt32, FR_RNE);
		fp_round(g, &U, &u, &fmt64, FR_RNE);
		fp_round(g, &V, &w, &fmt64, FR_RNE);
		fp_mul(g, &b, &U, &V, &fmt64, FR_RNE);
		fp_round(g, &c, &b, &fmt32, FR_RNE);
		fp_pack(g, &p1, &a, &fmt32, 0);
		fp_pack(g, &p2, &c, &fmt32, 0);
		if (aig_sat(g, L_NOT(fp_same(g, &p1, &p2, &fmt32)), 1) != 0 ||
		    aig_conflicts != 0) {
			if (verbose)
				fprintf(stderr, "fp selftest: f32 product vs f64: "
				    "not proved without search\n");
			bad++;
		}
		/* a + b is b + a, a * b is b * a: one circuit each */
		fp_add(g, &a, &u, &w, &fmt32, FR_RNE);
		fp_add(g, &b, &w, &u, &fmt32, FR_RNE);
		fp_pack(g, &p1, &a, &fmt32, 0);
		fp_pack(g, &p2, &b, &fmt32, 0);
		fp_mul(g, &a, &w, &u, &fmt32, FR_RNE);
		fp_mul(g, &b, &u, &w, &fmt32, FR_RNE);
		fp_pack(g, &wu, &a, &fmt32, 0);
		fp_pack(g, &wv, &b, &fmt32, 0);
		ok = fp_same(g, &p1, &p2, &fmt32);
		ok = aig_and(g, ok, fp_same(g, &wu, &wv, &fmt32));
		if (aig_sat(g, L_NOT(ok), 1) != 0 ||
		    aig_conflicts != 0) {
			if (verbose)
				fprintf(stderr, "fp selftest: commuted operands make "
				    "another circuit\n");
			bad++;
		}
		/* f16 -> f64 -> f16, bit for bit */
		w_input(g, &wu, 16);
		fp_unpack(g, &u, &wu, &fmt16);
		fp_round(g, &U, &u, &fmt64, FR_RNE);
		fp_round(g, &a, &U, &fmt16, FR_RNE);
		fp_pack(g, &p1, &u, &fmt16, 0);
		fp_pack(g, &p2, &a, &fmt16, 0);
		if (aig_sat(g, L_NOT(fp_same(g, &p1, &p2, &fmt16)), 1) != 0) {
			if (verbose)
				fprintf(stderr, "fp selftest: f16 round trip\n");
			bad++;
		}
		aig_free(g);
	}
	return bad;
}

/*
 * The float circuits: exhaustively on a 6-bit format against exact
 * arithmetic, against the evaluator in binary16, 32 and 64, and through
 * the solver.  Then each deliberate defect (a rounding, a sticky bit, the
 * subnormal floor, the adder's sticky, an exact zero's sign, inf * 0, the
 * packed subnormal, -0 < +0, fma rounding its product, 2^63 taken as an
 * integer, an integer truncated, floor as a truncation) must be caught by
 * those same checks.
 */
int
fp_selftest(void)
{
	int bad = 0, m;

	fp_err = NULL;
	mut = 0;
	bad += tiny_exhaustive(1);
	bad += against_eval(1);
	bad += sat_floats(1);
	for (m = 1; m <= 12; m++) {
		int caught;
		mut = m;
		caught = tiny_exhaustive(0);
		if (!caught)
			caught = against_eval(0);
		mut = 0;
		if (!caught) {
			fprintf(stderr, "fp selftest: defect %d not caught\n", m);
			bad++;
		}
	}
	if (fp_err) {
		fprintf(stderr, "fp selftest: %s\n", fp_err);
		bad++;
	}
	fp_err = NULL;
	return bad;
}
