/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Integers as floats and floats as integers: identities between the two
 * number systems of a 32-bit word that seed find (through irw.c), each
 * with its derivation and the conditions it holds under.  As in irw.c,
 * nothing here decides: find keeps one only when the active cost model
 * prices it lower.  f32 only; every condition is proved from the
 * parameters' ranges and the where clause (integer intervals from
 * irw.c, float ranges and integrality from the small analysis below).
 *
 *   - f32(i) for 0 <= i < 2^23 is asfloat(i | 0x4B000000) - 2^23 (the
 *     bits of 2^23 + i, then an exact subtraction); for -2^22 <= i <
 *     2^22, asfloat(i + 0x4B400000) - 1.5*2^23;
 *   - iN(x) for an integer-valued x in [-2^22, 2^22) is asint(x +
 *     1.5*2^23) - 0x4B400000 (the add is exact; it reads the integer in
 *     the mantissa);
 *   - floor(x), ceil(x) for |x| <= 2^22 by adding and subtracting
 *     1.5*2^23 (round to nearest even) and one compare; of an
 *     integer-valued x, x itself;
 *   - roundeven(x) for |x| <= 2^22 is that add and subtraction alone,
 *     iN(roundeven(x)) for -2^22 <= x < 2^22 - 1/2 the add read as bits,
 *     and roundeven(x * 2^-k) * 2^k, x to the nearest multiple of 2^k,
 *     the add and subtraction of 1.5*2^(23+k);
 *   - f32(iN(x)) of an integer-valued x is x; iN(f32(i)) below 2^24 is
 *     i;
 *   - f32(1 << k) is asfloat((k + 127) << 23): the exponent field;
 *   - x * f32(1 << k) for a normal x whose product stays normal is
 *     asfloat(asint(x) + (k << 23)): ldexp by an add on the bits;
 *   - W-1 - clz(x), floor(log2(x)), for x >= 1 is (asint(f32(x)) >> 23)
 *     - 127, the exponent field of the conversion, where the conversion
 *     is exact (below 2^24) or does not round up into the next power of
 *     two (x below 2^(e+1) - 2^(e-24) in each binade [2^e, 2^(e+1))
 *     the range meets: up to 2^25 - 2 from 1);
 *   - integer +, -, * whose every value stays inside (-2^24, 2^24) run
 *     on floats without rounding: the multiply-adds become a*b + c on
 *     f32 (an FFMA where the model contracts), conversions at the ends;
 *   - x / k for a constant k and 0 <= x < 2^24 is iN(f32(x) * c), c = 1/k
 *     rounded up to an f32, where the product never reaches the next
 *     integer (a margin, or every x tried); x % k is x - k q, or an fma;
 *   - two products with a common factor, one of them moved up by a
 *     constant, x*c + ((y*c) << k) or x*c + K*(y*c), are one product
 *     (x + (y << k))*c: two small integers packed in one word, or in
 *     one mantissa, take one multiply.  In a sized integer type that is
 *     distributivity modulo 2^w and holds always (and for | or xor
 *     where the ranges keep the two fields apart); in f32 it holds
 *     when the operands are integers and every value on both sides
 *     stays below 2^24, so that nothing rounds;
 *   - (a * b) % m for a, b in [0, 2^24), the product past 2^24, m a
 *     constant below 2^22: the product as p + e by 2Prod, q =
 *     floor(p * fl(1/m)) within one of the quotient, fma(q, -m, p) + e
 *     exact, then one add and one subtract of m (rw_modmul).
 */
#include "calc.h"
#include "irw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define P23 8388608.0		/* 2^23 */
#define P24 16777216.0		/* 2^24 */
#define MAGIC 12582912.0	/* 1.5 * 2^23 */
#define MAGIC_BITS 0x4B400000LL	/* its bits */
#define P23_BITS 0x4B000000LL	/* the bits of 2^23 */
#define P22 4194304.0		/* 2^22 */

static const ty_t U32 = TY(K_UINT, 32), I32 = TY(K_SINT, 32);

/* ---- what is known of an f32 subterm on the domain ---- */

struct fval {
	double lo, hi;		/* its values lie in [lo, hi], finite */
	int ok;			/* the bounds hold, and it is never NaN */
	int integral;		/* every value is an integer */
};

static void fval_of(struct ivc *c, nid n, struct fval *v, int depth);

static int
is_f32(nid n)
{
	return N.ty[n] == TY_F32;
}

/* one f32 step outward: a bound of a rounded result from the real one */
static double
out_lo(double v)
{
	return (double)nextafterf((float)v, -INFINITY);
}

static double
out_hi(double v)
{
	return (double)nextafterf((float)v, INFINITY);
}

/* does [lo, hi] fit the integer type t */
static int
fits_ty(ty_t t, int64_t lo, int64_t hi)
{
	struct ity it;

	if (TYK(t) == K_INT || TYK(t) == K_LIT)
		return 1;
	ity_of(&it, t, 0);
	if (it.k == K_UINT || it.k == K_BOOL)
		return lo >= 0 && (it.w >= 63 || (uint64_t)hi <= it.mask);
	return lo >= it.smin && hi <= it.smax;
}

/* an integer subterm's range on the domain, when it does not wrap */
static int
irange(struct ivc *c, nid a, int64_t *lo, int64_t *hi)
{
	char why[200];
	uint32_t k;

	if (!TY_ISINT(N.ty[a]) || TYK(N.ty[a]) == K_LIT)
		return 0;
	if (is_const(a)) {
		*lo = *hi = N.ki[a];
		return 1;
	}
	/* a float truncated: from the float's range, when it fits */
	if (N.op[a] == O_CAST && is_f32(N.a[a])) {
		struct fval X;
		fval_of(c, N.a[a], &X, 1);
		if (!X.ok || fabs(X.lo) > 9e18 || fabs(X.hi) > 9e18)
			return 0;
		*lo = (int64_t)trunc(X.lo);
		*hi = (int64_t)trunc(X.hi);
		return fits_ty(N.ty[a], *lo, *hi);
	}
	k = plan_loc(&c->p, a);
	if (k == NOTOK || c->iv[k].top || c->iv[k].wrap)
		return 0;
	return irw_range(c, a, INT64_MIN, lo, hi, why, sizeof why);
}

static void
fval_of(struct ivc *c, nid n, struct fval *v, int depth)
{
	struct fval A, B, C;
	int op = N.op[n];
	double x[4], lo, hi;
	int j;

	memset(v, 0, sizeof *v);
	if (depth > 48)
		return;
	if (op == O_CONST && TY_FP(N.ty[n])) {
		double k = TYK(N.ty[n]) == K_FLOAT ? N.kr[n] : fround(32, N.kr[n]);
		if (!isfinite(k))
			return;
		v->lo = v->hi = k;
		v->ok = 1;
		v->integral = k == floor(k);
		return;
	}
	if (!is_f32(n))
		return;
	switch (op) {
	case O_VAR: {
		const struct param *pp = &c->g.p[N.ki[n]];
		if (!pp->ranged || !isfinite(pp->rlo) || !isfinite(pp->rhi))
			return;
		v->lo = pp->rlo;
		v->hi = pp->rhi;
		v->ok = 1;
		return;
	}
	case O_CAST: {
		int64_t ilo, ihi;
		if (!irange(c, N.a[n], &ilo, &ihi))
			return;
		/* the conversion rounds to nearest, which is monotone; an
		 * f32 that is a rounded integer is an integer */
		v->lo = fround(32, (double)ilo);
		v->hi = fround(32, (double)ihi);
		v->ok = 1;
		v->integral = 1;
		return;
	}
	case O_FLOOR: case O_CEIL: case O_ROUNDEVEN:
		fval_of(c, N.a[n], &A, depth + 1);
		if (!A.ok)
			return;
		v->lo = op == O_FLOOR ? floor(A.lo) : op == O_CEIL ? ceil(A.lo) :
		    nearbyint(A.lo);
		v->hi = op == O_FLOOR ? floor(A.hi) : op == O_CEIL ? ceil(A.hi) :
		    nearbyint(A.hi);
		v->ok = v->integral = 1;
		return;
	case O_NEG:
		fval_of(c, N.a[n], &A, depth + 1);
		if (!A.ok)
			return;
		v->lo = -A.hi;
		v->hi = -A.lo;
		v->ok = 1;
		v->integral = A.integral;
		return;
	case O_ABS:
		fval_of(c, N.a[n], &A, depth + 1);
		if (!A.ok)
			return;
		v->lo = A.lo > 0 ? A.lo : A.hi < 0 ? -A.hi : 0;
		v->hi = fabs(A.lo) > fabs(A.hi) ? fabs(A.lo) : fabs(A.hi);
		v->ok = 1;
		v->integral = A.integral;
		return;
	case O_MIN: case O_MAX:
		fval_of(c, N.a[n], &A, depth + 1);
		fval_of(c, N.b[n], &B, depth + 1);
		if (!A.ok || !B.ok)
			return;
		if (op == O_MIN) {
			v->lo = A.lo < B.lo ? A.lo : B.lo;
			v->hi = A.hi < B.hi ? A.hi : B.hi;
		} else {
			v->lo = A.lo > B.lo ? A.lo : B.lo;
			v->hi = A.hi > B.hi ? A.hi : B.hi;
		}
		v->ok = 1;
		v->integral = A.integral && B.integral;
		return;
	case O_SEL:
		fval_of(c, N.b[n], &A, depth + 1);
		fval_of(c, N.c[n], &B, depth + 1);
		if (!A.ok || !B.ok)
			return;
		v->lo = A.lo < B.lo ? A.lo : B.lo;
		v->hi = A.hi > B.hi ? A.hi : B.hi;
		v->ok = 1;
		v->integral = A.integral && B.integral;
		return;
	case O_ADD: case O_SUB: case O_MUL: case O_FMA:
		fval_of(c, N.a[n], &A, depth + 1);
		fval_of(c, N.b[n], &B, depth + 1);
		if (!A.ok || !B.ok)
			return;
		if (op == O_ADD) {
			lo = A.lo + B.lo;
			hi = A.hi + B.hi;
		} else if (op == O_SUB) {
			lo = A.lo - B.hi;
			hi = A.hi - B.lo;
		} else {
			x[0] = A.lo * B.lo;
			x[1] = A.lo * B.hi;
			x[2] = A.hi * B.lo;
			x[3] = A.hi * B.hi;
			lo = hi = x[0];
			for (j = 1; j < 4; j++) {
				lo = x[j] < lo ? x[j] : lo;
				hi = x[j] > hi ? x[j] : hi;
			}
		}
		v->integral = A.integral && B.integral;
		if (op == O_FMA) {
			fval_of(c, N.c[n], &C, depth + 1);
			if (!C.ok)
				return;
			lo += C.lo;
			hi += C.hi;
			v->integral = v->integral && C.integral;
		}
		/*
		 * integers below 2^24 in magnitude: the exact result is an
		 * f32 and the bounds are exact (the doubles above hold
		 * integers below 2^49 exactly); otherwise one f32 step
		 * outward covers the double's and the f32's rounding
		 */
		if (!(v->integral && lo >= -P24 && hi <= P24)) {
			if (!(fabs(lo) < 1e38 && fabs(hi) < 1e38))
				return;
			lo = out_lo(lo);
			hi = out_hi(hi);
		}
		v->lo = lo;
		v->hi = hi;
		v->ok = 1;
		return;
	default:
		return;
	}
}

/* ---- helpers ---- */

static const char *
sstr(nid e)
{
	return shared_str(e, 60);
}

/* the same, in parentheses when it is an operation written infix */
static const char *
pstr(nid e)
{
	static char buf[4][80];
	static int at;
	char *b = buf[at++ & 3];
	int op = N.op[e];

	if (op == O_CONST || op == O_VAR || opinfo[op].arity == 1 ||
	    op == O_CAST || op == O_MIN || op == O_MAX || op == O_FMA)
		return sstr(e);
	snprintf(b, sizeof buf[0], "(%s)", sstr(e));
	return b;
}

static nid
kf(double v)
{
	return mk_const_i(TY_F32, (int64_t)v);
}

/* f32(a) by the bits, when a's range allows it (0: it does not) */
static nid
magic_i2f(nid a, int64_t lo, int64_t hi)
{
	/* a 32-bit integer read as another 32-bit type has the same bits,
	 * and the same value inside these ranges */
	while (N.op[a] == O_CAST && TY_SIZED(N.ty[a]) && TYW(N.ty[a]) == 32 &&
	    TY_SIZED(N.ty[N.a[a]]) && TYW(N.ty[N.a[a]]) == 32 &&
	    N.ty[N.a[a]] != TY_BOOL)
		a = N.a[a];
	if (lo >= 0 && hi < (int64_t)P23)
		return mk_bin(O_SUB, mk_un(O_ASFLOAT, mk_bin(O_OR, mk_cast(U32,
		    a), mk_const_i(U32, P23_BITS))), kf(P23));
	if (lo >= -(int64_t)P22 && hi < (int64_t)P22)
		return mk_bin(O_SUB, mk_un(O_ASFLOAT, mk_bin(O_ADD, mk_cast(I32,
		    a), mk_const_i(I32, MAGIC_BITS))), kf(MAGIC));
	return 0;
}

/* t(x) by the bits, for an integer-valued x in [-2^22, 2^22) */
static nid
magic_f2i(ty_t t, nid x)
{
	return mk_cast(t, mk_bin(O_SUB, mk_un(O_ASINT, mk_bin(O_ADD, x,
	    kf(MAGIC))), mk_const_i(I32, MAGIC_BITS)));
}

/* e as text for a derivation (kept: shared_str's buffers rotate) */
struct nm {
	char s[96];
};

static const char *
nm(struct nm *b, nid e)
{
	snprintf(b->s, sizeof b->s, "%s", pstr(e));
	return b->s;
}

static void
i2f_proof(struct sbuf *pf, nid a, int64_t lo, int64_t hi)
{
	struct nm A;
	const char *i = nm(&A, a);

	if (lo >= 0 && hi < (int64_t)P23)
		sb_printf(pf, "0 <= %s < 2^23: %s | 0x4B000000 is the bit "
		    "pattern with sign 0, exponent field 150 and mantissa %s, the "
		    "f32 2^23 + %s exactly (one unit in the last place there is "
		    "1)\n2^23 + %s - 2^23 = %s is an integer below 2^24, an f32: "
		    "the subtraction does not round", i, i, i, i, i, i);
	else
		sb_printf(pf, "-2^22 <= %s < 2^22: 0x4B400000 are the bits of "
		    "1.5*2^23 (exponent field 150, mantissa 2^22), and adding %s "
		    "moves the mantissa inside [0, 2^23) without reaching the "
		    "exponent: the bits of the f32 1.5*2^23 + %s exactly\n"
		    "subtracting 1.5*2^23 leaves %s, an integer below 2^24, an "
		    "f32: the subtraction does not round", i, i, i, i);
}

/* name: what to call x in the text (NULL: x as it is written) */
static void
f2i_proof(struct sbuf *pf, nid x, const char *name)
{
	struct nm X;
	const char *f = name ? name : nm(&X, x);

	sb_printf(pf, "%s + 1.5*2^23 is an integer in [2^23, 2^24), where "
	    "every integer is an f32: the add does not round\nits bits are "
	    "0x4B000000 + (%s + 2^22) = 0x4B400000 + %s (exponent field 150, "
	    "the mantissa the integer less 2^23): subtracting 0x4B400000 "
	    "leaves %s", f, f, f, f);
}

/* ---- the candidates ---- */

/* f32(a) of an integer a */
static void
rw_i2f(struct ivc *c, nid root, nid n, struct irw **out, int *cnt, int *cap)
{
	nid a = N.a[n], to;
	int64_t lo, hi;
	char what[200], cond[300];
	struct sbuf pf = {0};
	struct fval X;
	struct nm A;

	if (!is_f32(n) || TY_FP(N.ty[a]))
		return;
	/* f32(iN(x)) of an integer-valued x that fits: x itself */
	if (N.op[a] == O_CAST && is_f32(N.a[a])) {
		nid x = N.a[a];
		fval_of(c, x, &X, 0);
		if (X.ok && X.integral && fabs(X.lo) < 9e18 && fabs(X.hi) < 9e18 &&
		    fits_ty(N.ty[a], (int64_t)X.lo, (int64_t)X.hi)) {
			snprintf(what, sizeof what, "%s as %s (an integer-valued "
			    "float converted and back)", sstr(n), sstr(x));
			snprintf(cond, sizeof cond, "%s is an integer in [%.9g, "
			    "%.9g], inside %s", sstr(x), X.lo, X.hi,
			    ty_str(N.ty[a]));
			sb_printf(&pf, "%s(%s) truncates toward zero: of an integer "
			    "inside the type it is the same integer\nthat integer "
			    "came from an f32, so it is one: converting back does "
			    "not round", ty_str(N.ty[a]), sstr(x));
			irw_add(out, cnt, cap, root, n, x, what, cond, sb_str(&pf),
			    1);
			sb_clear(&pf);
		}
	}
	if (!irange(c, a, &lo, &hi) || (to = magic_i2f(a, lo, hi)) == 0) {
		free(pf.p);
		return;
	}
	snprintf(what, sizeof what, "the conversion %s by the bits of %s",
	    sstr(n), lo >= 0 ? "2^23" : "1.5*2^23");
	snprintf(cond, sizeof cond, "%s in [%lld, %lld], inside %s", sstr(a),
	    (long long)lo, (long long)hi, lo >= 0 ? "[0, 2^23)" :
	    "[-2^22, 2^22)");
	i2f_proof(&pf, a, lo, hi);
	sb_printf(&pf, "\nf32(%s) is exact below 2^24: both are %s", sstr(a),
	    nm(&A, a));
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

/* iN(x) of an f32 x */
static void
rw_f2i(struct ivc *c, nid root, nid n, struct irw **out, int *cnt, int *cap)
{
	nid x = N.a[n], to;
	ty_t t = N.ty[n];
	struct fval X;
	char what[200], cond[300];
	struct sbuf pf = {0};

	if (!TY_ISINT(t) || t == TY_BOOL || !is_f32(x))
		return;
	/* iN(f32(i)) below 2^24: i */
	if (N.op[x] == O_CAST && TY_ISINT(N.ty[N.a[x]])) {
		nid i = N.a[x];
		int64_t lo, hi;
		if (irange(c, i, &lo, &hi) && lo >= -(int64_t)P24 &&
		    hi <= (int64_t)P24 && fits_ty(t, lo, hi)) {
			to = mk_cast(t, i);
			snprintf(what, sizeof what, "%s as %s (an integer below "
			    "2^24 converted and back)", sstr(n), sstr(to));
			snprintf(cond, sizeof cond, "%s in [%lld, %lld], inside "
			    "[-2^24, 2^24] and %s", sstr(i), (long long)lo,
			    (long long)hi, ty_str(t));
			sb_printf(&pf, "f32 holds every integer up to 2^24 in "
			    "magnitude: f32(%s) is %s, and truncating an integer "
			    "leaves it", sstr(i), sstr(i));
			irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf),
			    1);
			sb_clear(&pf);
		}
	}
	/* iN(roundeven(y)): the add rounds y itself */
	if (N.op[x] == O_ROUNDEVEN && is_f32(N.a[x])) {
		nid y = N.a[x];
		struct fval Y;
		fval_of(c, y, &Y, 0);
		if (Y.ok && Y.lo >= -P22 && Y.hi < P22 - 0.5 && fits_ty(t,
		    (int64_t)nearbyint(Y.lo), (int64_t)nearbyint(Y.hi))) {
			struct nm YS;
			const char *ys = nm(&YS, y);
			to = magic_f2i(t, y);
			snprintf(what, sizeof what, "%s by the bits of 1.5*2^23, the "
			    "add rounding %s", sstr(n), sstr(y));
			snprintf(cond, sizeof cond, "%s in [%.9g, %.9g], inside "
			    "[-2^22, 2^22 - 1/2) and %s", sstr(y), Y.lo, Y.hi,
			    ty_str(t));
			sb_printf(&pf, "%s + 1.5*2^23 lies in [2^23, 2^24), where the "
			    "f32 are the integers: the add rounds %s to the nearest "
			    "integer, a tie to the even one, giving roundeven(%s) + "
			    "1.5*2^23 exactly (below 2^22 - 1/2, roundeven(%s) stays "
			    "below 2^22 and the sum below 2^24)\nits bits are 0x4B000000 "
			    "+ (roundeven(%s) + 2^22) = 0x4B400000 + roundeven(%s) "
			    "(exponent field 150): subtracting 0x4B400000 leaves "
			    "roundeven(%s)\n%s of an integer is that integer", ys, ys, ys,
			    ys, ys, ys, ys, ty_str(t));
			irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf),
			    1);
			sb_clear(&pf);
		}
	}
	fval_of(c, x, &X, 0);
	if (!X.ok || !X.integral || X.lo < -P22 || X.hi >= P22 ||
	    !fits_ty(t, (int64_t)X.lo, (int64_t)X.hi)) {
		free(pf.p);
		return;
	}
	to = magic_f2i(t, x);
	snprintf(what, sizeof what, "the conversion %s by the bits of 1.5*2^23",
	    sstr(n));
	snprintf(cond, sizeof cond, "%s is an integer in [%.9g, %.9g], inside "
	    "[-2^22, 2^22) and %s", sstr(x), X.lo, X.hi, ty_str(t));
	f2i_proof(&pf, x, NULL);
	sb_printf(&pf, "\n%s(%s) truncates toward zero: of an integer it is "
	    "that integer", ty_str(t), sstr(x));
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

/* roundeven(x) of an f32 x: the magic add and subtraction */
static void
rw_roundeven(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid x = N.a[n], to;
	struct fval X;
	char what[200], cond[300];
	struct sbuf pf = {0};
	struct nm XS;
	const char *xs;

	if (!is_f32(n) || !is_f32(x))
		return;
	fval_of(c, x, &X, 0);
	if (!X.ok)
		return;
	if (X.integral) {
		snprintf(what, sizeof what, "%s as %s (it is an integer already)",
		    sstr(n), sstr(x));
		snprintf(cond, sizeof cond, "%s is an integer in [%.9g, %.9g]",
		    sstr(x), X.lo, X.hi);
		sb_puts(&pf, "roundeven of an integer is that integer");
		irw_add(out, cnt, cap, root, n, x, what, cond, sb_str(&pf), 1);
		free(pf.p);
		return;
	}
	if (X.lo < -P22 || X.hi > P22)
		return;
	to = mk_bin(O_SUB, mk_bin(O_ADD, x, kf(MAGIC)), kf(MAGIC));
	snprintf(what, sizeof what, "%s by adding and subtracting 1.5*2^23",
	    sstr(n));
	snprintf(cond, sizeof cond, "%s in [%.9g, %.9g], inside [-2^22, 2^22]; "
	    "where the result is zero its sign may differ (allowed by Vulkan's "
	    "default float controls, without SignedZeroInfNanPreserve)",
	    sstr(x), X.lo, X.hi);
	xs = nm(&XS, x);
	sb_printf(&pf, "%s + 1.5*2^23 lies in [2^23, 2^24], where the f32 are "
	    "the integers: the add rounds %s to the nearest integer, a tie to "
	    "the even one, giving roundeven(%s) + 1.5*2^23 exactly\nsubtracting "
	    "1.5*2^23 leaves roundeven(%s), an integer up to 2^22 in magnitude, "
	    "an f32: no rounding", xs, xs, xs, xs);
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

/* k when e is the f32 constant 2^k, |k| <= 20; else 0 with *ok 0 */
static int
pow2_const(nid e, int *ok)
{
	double v, m;
	int k;

	*ok = 0;
	if (!is_const(e) || !is_f32(e))
		return 0;
	v = N.kr[e];
	if (!(v > 0))
		return 0;
	m = frexp(v, &k);
	if (m != 0.5 || k - 1 < -20 || k - 1 > 20)
		return 0;
	*ok = 1;
	return k - 1;
}

/* roundeven(x * 2^-k) * 2^k: x to the nearest multiple of 2^k */
static void
rw_roundmul(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid r, z, x, to;
	int i, j, k, kk, ok, ok2;
	struct fval X;
	double C, lim;
	char what[200], cond[300], cs[48];
	struct sbuf pf = {0};
	struct nm XS;
	const char *xs;

	if (N.op[n] != O_MUL || !is_f32(n))
		return;
	for (i = 0; i < 2; i++) {
		r = i ? N.b[n] : N.a[n];
		k = pow2_const(i ? N.a[n] : N.b[n], &ok);
		if (!ok || N.op[r] != O_ROUNDEVEN || N.op[N.a[r]] != O_MUL)
			continue;
		z = N.a[r];
		for (j = 0; j < 2; j++) {
			x = j ? N.b[z] : N.a[z];
			kk = pow2_const(j ? N.a[z] : N.b[z], &ok2);
			if (!ok2 || kk != -k || !is_f32(x))
				continue;
			fval_of(c, x, &X, 0);
			lim = ldexp(1, 22 + k);
			if (!X.ok || X.lo < -lim || X.hi > lim)
				continue;
			C = 1.5 * ldexp(1, 23 + k);
			to = mk_bin(O_SUB, mk_bin(O_ADD, x, mk_cast(TY_F32,
			    mk_const_r(C))), mk_cast(TY_F32, mk_const_r(C)));
			snprintf(cs, sizeof cs, "1.5*2^%d", 23 + k);
			snprintf(what, sizeof what, "%s, %s to the nearest multiple of "
			    "2^%d, by adding and subtracting %s", sstr(n), sstr(x), k,
			    cs);
			snprintf(cond, sizeof cond, "%s in [%.9g, %.9g], inside "
			    "[-2^%d, 2^%d]; where the result is zero its sign may differ",
			    sstr(x), X.lo, X.hi, 22 + k, 22 + k);
			xs = nm(&XS, x);
			sb_printf(&pf, "%s * 2^%d rounds only below 2^-126 in "
			    "magnitude, far under 1/2, where roundeven gives 0 either "
			    "way, and the integer it gives times 2^%d is exact: "
			    "roundeven(%s * 2^%d) * 2^%d is %s rounded to the nearest "
			    "multiple of 2^%d, a tie to the even multiple\n"
			    "%s + %s lies in [2^%d, 2^%d], where the f32 are the multiples "
			    "of 2^%d: the add rounds %s the same way, exactly\nsubtracting "
			    "%s leaves that multiple, at most 2^%d in magnitude and a "
			    "multiple of 2^%d, an f32: no rounding", xs, -k, k, xs, -k, k,
			    xs, k, xs, cs, 23 + k, 24 + k, k, xs, cs, 22 + k, k);
			irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
			sb_clear(&pf);
		}
	}
	free(pf.p);
}

/* floor(x), ceil(x) of an f32 x */
static void
rw_floor(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid x = N.a[n], r, to;
	struct fval X;
	int fl = N.op[n] == O_FLOOR;
	char what[200], cond[300];
	struct sbuf pf = {0};
	struct nm XS;
	const char *xs;

	if (!is_f32(n) || !is_f32(x))
		return;
	fval_of(c, x, &X, 0);
	if (!X.ok)
		return;
	if (X.integral) {
		snprintf(what, sizeof what, "%s as %s (it is an integer already)",
		    sstr(n), sstr(x));
		snprintf(cond, sizeof cond, "%s is an integer in [%.9g, %.9g]",
		    sstr(x), X.lo, X.hi);
		sb_printf(&pf, "%s of an integer is that integer",
		    fl ? "floor" : "ceil");
		irw_add(out, cnt, cap, root, n, x, what, cond, sb_str(&pf), 1);
		free(pf.p);
		return;
	}
	if (X.lo < -P22 || X.hi > P22)
		return;
	r = mk_bin(O_SUB, mk_bin(O_ADD, x, kf(MAGIC)), kf(MAGIC));
	to = fl ? mk_sel(mk_bin(O_GT, r, x), mk_bin(O_SUB, r, kf(1)), r) :
	    mk_sel(mk_bin(O_LT, r, x), mk_bin(O_ADD, r, kf(1)), r);
	snprintf(what, sizeof what, "%s by adding and subtracting 1.5*2^23, "
	    "and one compare", sstr(n));
	snprintf(cond, sizeof cond, "%s in [%.9g, %.9g], inside [-2^22, 2^22]; "
	    "where the result is zero its sign may differ (allowed by Vulkan's "
	    "default float controls, without SignedZeroInfNanPreserve)",
	    sstr(x), X.lo, X.hi);
	xs = nm(&XS, x);
	sb_printf(&pf, "%s + 1.5*2^23 lies in [2^23, 2^24], where the f32 are "
	    "the integers: the add rounds it to the nearest integer (ties to "
	    "even), r + 1.5*2^23 with r = round(%s)\nsubtracting 1.5*2^23 from "
	    "an integer in [2^23, 2^24] gives r, an integer up to 2^22 in "
	    "magnitude, an f32: no rounding\n|r - %s| <= 1/2, so %s(%s) is r "
	    "where r %s %s and r %s 1 otherwise (an exact f32 subtraction of "
	    "integers)", xs, sstr(x), xs, fl ? "floor" : "ceil", sstr(x),
	    fl ? "<=" : ">=", xs, fl ? "-" : "+");
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

/*
 * x / k and x % k, k a constant > 2 and no power of two, for an integer
 * 0 <= x <= X < 2^24, on the f32 pipe: q = iN(f32(x) * c), c = 1/k
 * rounded up to an f32, and x % k = x - k q, or all in f32 as
 * fma(-floor(f32(x) c), k, f32(x)) read back by the bits (the remainder
 * is an integer below k, exact in the fma).  floor(x/k) <= x c, so the
 * product rounded to nearest is not below that integer; it stays below
 * the next one while X (c - 1/k) plus half an ulp of X c is below 1/k
 * (the fraction of x/k is at most 1 - 1/k).  Where that margin does not
 * hold, every x of the range is tried; a range where some x fails gets
 * the rewrite as not applied, with that x.
 */
/*
 * The first x in [lo, hi] where trunc(f32(x) * cf) is not x / k, or -1:
 * every x tried (up to 2^24 of them, the quotient kept by counting), and
 * the answer kept for the next asker of the same range (find asks once a
 * node, and a module asks for each of its nodes).
 */
static int64_t
divk_tried(int64_t k, float cf, int64_t lo, int64_t hi)
{
	static struct {
		int64_t k, lo, hi, bad;
	} memo[16];
	static int nmemo, next;
	int64_t v, q = lo / k, r = lo % k, bad = -1;
	int i;

	for (i = 0; i < nmemo; i++)
		if (memo[i].k == k && memo[i].lo == lo && memo[i].hi == hi)
			return memo[i].bad;
	for (v = lo; v <= hi; v++) {
		if ((int64_t)((float)v * cf) != q) {
			bad = v;
			break;
		}
		if (++r == k) {
			r = 0;
			q++;
		}
	}
	i = next;
	next = (next + 1) % 16;
	if (nmemo < 16)
		nmemo++;
	memo[i].k = k;
	memo[i].lo = lo;
	memo[i].hi = hi;
	memo[i].bad = bad;
	return bad;
}

static void
rw_divk(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid x = N.a[n], d = N.b[n], q, to, xf, p, r;
	ty_t t = N.ty[n];
	int64_t lo, hi, k, bad = -1;
	int rem = N.op[n] == O_MOD, e, m;
	size_t at;
	float cf;
	double ex, hulp;
	char what[200], cond[400], cs[40];
	struct sbuf pf = {0};
	struct nm XS;
	const char *xs;

	if (!TY_ISINT(t) || t == TY_BOOL || TYK(t) == K_LIT || !is_const(d) ||
	    N.ty[x] != t)
		return;
	k = N.ki[d];
	if (k <= 2 || (k & (k - 1)) == 0 || k >= (int64_t)P24)
		return;
	if (!irange(c, x, &lo, &hi) || lo < 0 || hi >= (int64_t)P24 || hi < k)
		return;
	/* the f32 at or above 1/k nearest to it (cf k is exact in double) */
	cf = (float)(1.0 / (double)k);
	while ((double)cf * (double)k < 1.0)
		cf = nextafterf(cf, INFINITY);
	while ((double)nextafterf(cf, 0.0f) * (double)k >= 1.0)
		cf = nextafterf(cf, 0.0f);
	/* hi (cf - 1/k), and half an ulp where hi cf lies */
	ex = (double)hi * ((double)cf * (double)k - 1.0) / (double)k;
	(void)frexp((double)hi * (double)cf, &e);
	hulp = ldexp(1.0, e - 1 - 24);
	m = ex + hulp < 1.0 / (double)k;
	if (!m)
		bad = divk_tried(k, cf, lo, hi);
	xf = mk_cast(TY_F32, x);
	p = mk_bin(O_MUL, xf, mk_cast(TY_F32, mk_const_r(cf)));
	q = mk_cast(t, p);
	to = rem ? mk_bin(O_SUB, x, mk_bin(O_MUL, q, mk_const_i(t, k))) : q;
	xs = nm(&XS, x);
	snprintf(cs, sizeof cs, "%.9g", (double)cf);
	if (rem)
		snprintf(what, sizeof what, "%s as %s - %lld q on the f32 pipe, q "
		    "the conversion of f32(%s) * %s (1/%lld rounded up to an f32)",
		    sstr(n), xs, (long long)k, xs, cs, (long long)k);
	else
		snprintf(what, sizeof what, "%s on the f32 pipe: the conversion of "
		    "f32(%s) * %s (1/%lld rounded up to an f32)", sstr(n), xs, cs,
		    (long long)k);
	if (m)
		snprintf(cond, sizeof cond, "%s in [%lld, %lld], inside [0, 2^24), "
		    "and %lld (%s - 1/%lld) + ulp/2 < 1/%lld: %.3g + %.3g < %.3g",
		    xs, (long long)lo, (long long)hi, (long long)hi, cs,
		    (long long)k, (long long)k, ex, hulp, 1.0 / (double)k);
	else if (bad < 0)
		snprintf(cond, sizeof cond, "%s in [%lld, %lld], inside [0, 2^24), "
		    "every value of it tried (past the margin)", xs, (long long)lo,
		    (long long)hi);
	else
		snprintf(cond, sizeof cond, "%s in [%lld, %lld]: at %s = %lld the "
		    "product rounds up to %lld, not %lld", xs, (long long)lo,
		    (long long)hi, xs, (long long)bad, (long long)(bad / k + 1),
		    (long long)(bad / k));
	sb_printf(&pf, "%s < 2^24 converts to f32 exactly\n%s = %a is 1/%lld "
	    "rounded up to an f32, above it by %.3g\nq0 = floor(%s/%lld) is an "
	    "integer below 2^24, an f32, and f32(%s) * %s >= %s/%lld >= q0: the "
	    "product rounded to nearest is not below q0", xs, cs, (double)cf,
	    (long long)k, ex / (double)hi, xs, (long long)k, xs, cs, xs,
	    (long long)k);
	if (m)
		sb_printf(&pf, "\n%s/%lld <= q0 + %lld/%lld, so the exact product "
		    "is at most q0 + %lld/%lld + %.3g (%lld times the excess); it "
		    "is below %.9g, where rounding adds at most %.3g: it stays "
		    "below q0 + 1, as %.3g + %.3g < 1/%lld", xs, (long long)k,
		    (long long)(k - 1), (long long)k, (long long)(k - 1),
		    (long long)k, ex, (long long)hi, ldexp(1.0, e), hulp, ex, hulp,
		    (long long)k);
	else
		sb_printf(&pf, "\nthe margin does not cover the range (%.3g + %.3g "
		    "against 1/%lld): every %s in [%lld, %lld] was tried", ex,
		    hulp, (long long)k, xs, (long long)lo, (long long)hi);
	sb_printf(&pf, "\nthe conversion to %s truncates toward zero: of a "
	    "value in [q0, q0 + 1) it is q0 = %s / %lld", ty_str(t), xs,
	    (long long)k);
	if (!rem) {
		irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf),
		    bad < 0);
		free(pf.p);
		return;
	}
	at = pf.n;
	sb_printf(&pf, "\n%s - %lld q0 = %s %% %lld, in [0, %lld): the "
	    "integer operations do not wrap", xs, (long long)k, xs,
	    (long long)k, (long long)k);
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), bad < 0);
	if (k > (int64_t)P22) {
		free(pf.p);
		return;
	}
	/* all in f32: floor where the conversion truncated, then the fma */
	r = mk_fma(mk_un(O_FLOOR, p), kf((double)-k), xf);
	to = magic_f2i(t, r);
	snprintf(what, sizeof what, "%s as fma(-floor(f32(%s) * %s), %lld, "
	    "f32(%s)) on the f32 pipe, read back by the bits of 1.5*2^23",
	    sstr(n), xs, cs, (long long)k, xs);
	pf.n = at;
	pf.p[at] = 0;
	sb_printf(&pf, ", and so is its floor\nthe fma rounds f32(%s) - %lld q0 "
	    "= %s %% %lld once: an integer in [0, %lld), an f32, so it does not "
	    "round\n", xs, (long long)k, xs, (long long)k, (long long)k);
	f2i_proof(&pf, r, "that remainder");
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), bad < 0);
	free(pf.p);
}

/* a widening integer cast's operand, where a's range [0, 2^24) says the
 * value goes through unchanged */
static nid
unwiden(nid a)
{
	while (N.op[a] == O_CAST && TY_ISINT(N.ty[a]) && TY_SIZED(N.ty[a]) &&
	    TY_ISINT(N.ty[N.a[a]]) && TY_SIZED(N.ty[N.a[a]]) &&
	    N.ty[N.a[a]] != TY_BOOL && TYW(N.ty[N.a[a]]) <= TYW(N.ty[a]))
		a = N.a[a];
	return a;
}

/* f32(a) of an integer a in [lo, hi] inside [0, 2^24): exact */
static nid
f32_of(nid a, int64_t lo, int64_t hi)
{
	nid f;

	if (is_const(a))
		return kf((double)N.ki[a]);
	a = unwiden(a);
	f = magic_i2f(a, lo, hi);
	return f ? f : mk_cast(TY_F32, a);
}

/*
 * (a * b) % m for integers a, b in [0, 2^24) and a constant m below 2^22
 * (Barrett on the f32 pipe, #180): the product as two floats p + e
 * (2Prod), q = floor(p * fl(1/m)) within one of floor(a b / m), and
 * a b - q m = fma(-q, m, p) + e, exact, put back in [0, m) by one add
 * and one subtract of m.  Where a b passes 2^24 nothing else on the f32
 * pipe holds it; the integer path is a wide multiply and a remainder.
 */
static void
rw_modmul(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid x = N.a[n], d = N.b[n], a, b, fa, fb, p, e, t, q, r, to;
	ty_t ty = N.ty[n];
	int64_t m, alo, ahi, blo, bhi;
	double P, emax, g, E, ulp = ldexp(1.0, -24);
	float cf;
	int ex;
	char what[200], cond[400], cs[40];
	struct sbuf pf = {0};
	struct nm AS, BS;
	const char *as, *bs;
	struct irw *rw;

	if (N.op[n] != O_MOD || !TY_ISINT(ty) || ty == TY_BOOL ||
	    TYK(ty) == K_LIT || !is_const(d) || N.op[x] != O_MUL ||
	    N.ty[x] != ty)
		return;
	m = N.ki[d];
	if (m < 3 || (m & (m - 1)) == 0 || m >= (int64_t)P22)
		return;
	a = N.a[x];
	b = N.b[x];
	if (!irange(c, a, &alo, &ahi) || !irange(c, b, &blo, &bhi) || alo < 0 ||
	    blo < 0 || ahi >= (int64_t)P24 || bhi >= (int64_t)P24)
		return;
	P = (double)ahi * (double)bhi;	/* exact: below 2^48 */
	/* below 2^24 the product is one f32 (rw_divk); the type must hold
	 * it, or the multiply wraps first */
	if (P < P24 || !fits_ty(ty, 0, (int64_t)P))
		return;
	cf = (float)(1.0 / (double)m);
	g = fabs((double)cf * (double)m - 1.0);	/* exact in double */
	E = P / (double)m * ((1.0 + ulp) * (1.0 + ulp) * (1.0 + g) - 1.0);
	(void)frexp(P, &ex);
	emax = ldexp(1.0, ex - 1 - 24);	/* half an ulp at P */
	if (E >= 1.0 || 2.0 * (double)m + emax >= P24)
		return;
	fa = f32_of(a, alo, ahi);
	fb = f32_of(b, blo, bhi);
	p = mk_bin(O_MUL, fa, fb);
	e = mk_fma(fa, fb, mk_un(O_NEG, p));
	t = mk_bin(O_MUL, p, mk_cast(TY_F32, mk_const_r(cf)));
	q = mk_un(O_FLOOR, t);
	r = mk_bin(O_ADD, mk_fma(q, kf((double)-m), p), e);
	r = mk_sel(mk_bin(O_LT, r, kf(0)), mk_bin(O_ADD, r, kf((double)m)), r);
	r = mk_sel(mk_bin(O_GE, r, kf((double)m)), mk_bin(O_SUB, r,
	    kf((double)m)), r);
	to = magic_f2i(ty, r);
	as = nm(&AS, a);
	bs = nm(&BS, b);
	snprintf(cs, sizeof cs, "%.9g", (double)cf);
	snprintf(what, sizeof what, "%s on the f32 pipe: %s*%s as two floats "
	    "(2Prod), the quotient by %s (1/%lld) floored, the remainder put "
	    "in [0, %lld)", sstr(n), as, bs, cs, (long long)m, (long long)m);
	snprintf(cond, sizeof cond, "%s in [%lld, %lld] and %s in [%lld, %lld], "
	    "below 2^24; their product at most %.17g, and %.17g/%lld times the "
	    "three roundings is %.3g < 1", as, (long long)alo, (long long)ahi,
	    bs, (long long)blo, (long long)bhi, P, P, (long long)m, E);
	sb_printf(&pf, "%s and %s are integers below 2^24: their f32 are exact"
	    "\np = f32(%s)*f32(%s) rounded and e = fma(f32(%s), f32(%s), -p) "
	    "make %s*%s = p + e exactly (2Prod: the error of a rounded product "
	    "is an f32, and the fma rounds it once, to itself); |e| <= %.17g, "
	    "half an ulp at %.17g", as, bs, as, bs, as, bs, as, bs, emax, P);
	sb_printf(&pf, "\n%s = %a is 1/%lld within %.3g relative; t = p * %s "
	    "rounded: p, the constant and t each round by at most 2^-24, so |t "
	    "- %s*%s/%lld| <= %.17g/%lld ((1 + 2^-24)^2 (1 + %.3g) - 1) = %.3g "
	    "< 1", cs, (double)cf, (long long)m, g, cs, as, bs, (long long)m, P,
	    (long long)m, g, E);
	sb_printf(&pf, "\nq = floor(t) is then floor(%s*%s/%lld) - 1, + 0 or + "
	    "1: %s*%s - q*%lld is in (-%lld, %lld)", as, bs, (long long)m, as,
	    bs, (long long)m, (long long)m, 2 * (long long)m);
	sb_printf(&pf, "\nfma(q, -%lld, p) = p - q*%lld rounded once: an integer "
	    "of magnitude below 2*%lld + %.17g < 2^24, an f32, so it does not "
	    "round; adding e gives %s*%s - q*%lld, an integer in (-%lld, %lld), "
	    "exactly", (long long)m, (long long)m, (long long)m, emax, as, bs,
	    (long long)m, (long long)m, 2 * (long long)m);
	sb_printf(&pf, "\nadding %lld where it is negative and subtracting it "
	    "where it is %lld or more (exact: integers below 2^24) leaves the "
	    "one value in [0, %lld): %s*%s %% %lld\n", (long long)m,
	    (long long)m, (long long)m, as, bs, (long long)m);
	f2i_proof(&pf, r, "that remainder");
	sb_puts(&pf, "\nVulkan rounds an fma once only where it is "
	    "NoContraction (`precise` in GLSL): written into a shader, the two "
	    "fmas go under it");
	rw = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	if (rw)
		rw->nosat = rw->derived = 1;
	free(pf.p);
}

/* x / k and x % k by a constant on the f32 pipe (irw.c calls it after its
 * own candidates for the node, so theirs keep their places) */
void
ifx_divk(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	rw_divk(c, root, n, out, cnt, cap);
	rw_modmul(c, root, n, out, cnt, cap);
}

/* n is f32(1 << k): k, and its range inside the exponents that hold */
static int
pow2_of(struct ivc *c, nid n, nid *k, int64_t *klo, int64_t *khi)
{
	nid s;
	ty_t t;
	int w;

	if (N.op[n] != O_CAST || !is_f32(n))
		return 0;
	s = N.a[n];
	if (N.op[s] != O_SHL || !is_const(N.a[s]) || N.ki[N.a[s]] != 1 ||
	    TY_FP(N.ty[s]))
		return 0;
	*k = N.b[s];
	if (!irange(c, *k, klo, khi))
		return 0;
	t = N.ty[s];
	w = TY_SIZED(t) ? TYW(t) : 64;
	/* 1 << k must be 2^k in its type (no wrap, not the sign bit) */
	if (*klo < 0 || *khi > (TYK(t) == K_UINT ? w - 1 : w - 2) || *khi > 127)
		return 0;
	return 1;
}

/* f32(1 << k): the exponent field */
static void
rw_pow2(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid k, to;
	int64_t klo, khi;
	char what[200], cond[300];
	struct sbuf pf = {0};
	struct nm K;
	const char *ks;

	if (!pow2_of(c, n, &k, &klo, &khi))
		return;
	to = mk_un(O_ASFLOAT, mk_bin(O_SHL, mk_bin(O_ADD, mk_cast(U32, k),
	    mk_const_i(U32, 127)), mk_const_i(U32, 23)));
	ks = nm(&K, k);
	snprintf(what, sizeof what, "%s as its bits, the exponent field %s + "
	    "127", sstr(n), ks);
	snprintf(cond, sizeof cond, "%s in [%lld, %lld]: 1 << %s is 2^%s, not "
	    "wrapped", sstr(k), (long long)klo, (long long)khi, ks, ks);
	sb_printf(&pf, "2^%s is a power of two below 2^128: an f32, so the "
	    "conversion is exact\nits bits are sign 0, exponent field %s + 127 "
	    "(in [%lld, %lld], a normal number) and mantissa 0: (%s + 127) << "
	    "23", ks, ks, (long long)klo + 127, (long long)khi + 127, ks);
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

/* x * f32(1 << k): an add on x's exponent field */
static void
rw_ldexp(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid x, k = 0, to;
	int64_t klo = 0, khi = 0;
	struct fval X;
	double amin, amax;
	char what[200], cond[400];
	struct sbuf pf = {0};
	struct nm XS, K;
	const char *xs, *ks;
	int j;

	if (!is_f32(n))
		return;
	for (j = 0; j < 2; j++)
		if (pow2_of(c, j ? N.a[n] : N.b[n], &k, &klo, &khi))
			break;
	if (j == 2)
		return;
	x = j ? N.b[n] : N.a[n];
	fval_of(c, x, &X, 0);
	if (!X.ok || !(X.lo > 0 || X.hi < 0))
		return;
	amin = X.lo > 0 ? X.lo : -X.hi;
	amax = X.lo > 0 ? X.hi : -X.lo;
	/* x normal, and x * 2^k below 2^128 */
	if (amin < ldexp(1.0, -126) || amax * ldexp(1.0, (int)khi) >=
	    ldexp(1.0, 128))
		return;
	to = mk_un(O_ASFLOAT, mk_bin(O_ADD, mk_un(O_ASINT, x), mk_bin(O_SHL,
	    mk_cast(I32, k), mk_const_i(I32, 23))));
	xs = nm(&XS, x);
	ks = nm(&K, k);
	snprintf(what, sizeof what, "%s as an add of %s to the exponent field "
	    "of %s", sstr(n), ks, xs);
	snprintf(cond, sizeof cond, "%s in [%.9g, %.9g] (normal, never zero), "
	    "%s in [%lld, %lld], and %.9g * 2^%lld < 2^128", sstr(x), X.lo,
	    X.hi, sstr(k), (long long)klo, (long long)khi, amax,
	    (long long)khi);
	sb_printf(&pf, "%s = m * 2^e with 1 <= |m| < 2 and e >= -126: its "
	    "bits hold e + 127 in the exponent field (bits 23..30)\n%s * 2^%s "
	    "= m * 2^(e + %s) has the same mantissa and sign, and e + %s <= "
	    "127: it is an f32, so the multiply does not round, and its "
	    "exponent field is e + 127 + %s\nadding %s << 23 to the bits adds "
	    "%s to that field; the field stays in [1, 254], so nothing "
	    "carries into the sign", xs, xs, ks, ks, ks, ks, ks, ks);
	irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

/* W-1 - clz(x): the exponent field of f32(x) */
static void
rw_ilog2(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid z = N.b[n], x, f, to;
	ty_t t = N.ty[n], tx;
	int64_t lo, hi, bad = -1;
	int v, e, be = 0, nzone = 0;
	char what[200], cond[300];
	struct sbuf pf = {0};
	struct nm XS;
	const char *xs;

	if (N.op[n] != O_SUB || !TY_ISINT(t) || !is_const(N.a[n]) ||
	    N.op[z] != O_CLZ)
		return;
	x = N.a[z];
	tx = N.ty[x];
	if (!TY_SIZED(tx) || N.ki[N.a[n]] != TYW(tx) - 1 ||
	    !irange(c, x, &lo, &hi) || lo < 1 || hi >= ((int64_t)1 << 62))
		return;
	/*
	 * Past 2^24 f32(x) rounds to nearest even: in [2^e, 2^(e+1)) the f32
	 * are 2^(e-23) apart, so x goes up into 2^(e+1) from half a step
	 * below it on (2^(e+1) - 2^(e-24), a tie going to the even 2^(e+1)).
	 * The range must miss every such zone, binade by binade.
	 */
	for (e = 24; e < 62 && bad < 0; e++) {
		int64_t zlo = ((int64_t)1 << (e + 1)) - ((int64_t)1 << (e - 24));
		int64_t zhi = ((int64_t)1 << (e + 1)) - 1;
		if (zlo > hi)
			break;
		if (lo <= zhi && hi >= zlo) {
			bad = lo > zlo ? lo : zlo;
			be = e;
		}
		nzone++;
	}
	xs = nm(&XS, x);
	for (v = 0; v < 2; v++) {
		f = v ? magic_i2f(x, lo, hi) : mk_cast(TY_F32, x);
		if (!f)
			continue;
		to = mk_cast(t, mk_bin(O_SUB, mk_bin(O_SHR, mk_un(O_ASINT, f),
		    mk_const_i(I32, 23)), mk_const_i(I32, 127)));
		snprintf(what, sizeof what, "%s, floor(log2(%s)), as the exponent "
		    "field of f32(%s)%s", sstr(n), sstr(x), sstr(x), v ?
		    ", converted by the bits of 2^23" : "");
		sb_clear(&pf);
		sb_printf(&pf, "%d - clz(%s) is the index e of the highest set "
		    "bit of %s: 2^e <= %s < 2^(e+1)\n", TYW(tx) - 1, sstr(x), xs,
		    xs);
		if (hi <= (int64_t)P24) {
			snprintf(cond, sizeof cond, "%s in [%lld, %lld], inside [1, "
			    "2^24]", sstr(x), (long long)lo, (long long)hi);
			sb_printf(&pf, "%s <= 2^24: f32(%s) is exact, so its exponent "
			    "field is e + 127 and its sign bit 0", xs, sstr(x));
		} else {
			if (v)
				continue;	/* the bits of 2^23 hold below 2^23 */
			snprintf(cond, sizeof cond, "%s in [%lld, %lld]%s", sstr(x),
			    (long long)lo, (long long)hi, bad < 0 ? ", which misses "
			    "every x that rounds up into the next power of two" : "");
			sb_printf(&pf, "past 2^24 f32(%s) rounds to nearest even: in "
			    "[2^e, 2^(e+1)) the f32 are 2^(e-23) apart, so it rounds "
			    "up to 2^(e+1) only from 2^(e+1) - 2^(e-24) on (half a step "
			    "below; the tie goes to the even 2^(e+1)), and elsewhere "
			    "stays in [2^e, 2^(e+1))\n", xs);
			if (bad >= 0) {
				sb_printf(&pf, "but %s = %lld is in [2^%d - 2^%d, 2^%d): "
				    "f32(%s) is 2^%d, whose exponent field is %d + 127, not "
				    "%d + 127", xs, (long long)bad, be + 1, be - 24, be + 1,
				    xs, be + 1, be + 1, be);
				irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf),
				    0);
				continue;
			}
			if (nzone == 0)
				sb_printf(&pf, "[%lld, %lld] stays below 2^25 - 1, the "
				    "first x that rounds up (e = 24)", (long long)lo,
				    (long long)hi);
			else
				sb_printf(&pf, "[%lld, %lld] meets none of those, binade "
				    "by binade (e = 24 to %d: from 2^25 - 1, 2^26 - 2, ..., "
				    "2^(e+1) - 2^(e-24))", (long long)lo, (long long)hi, 23 +
				    nzone);
			sb_printf(&pf, ", so f32(%s) keeps the exponent e, its field e + "
			    "127, its sign bit 0", sstr(x));
		}
		sb_printf(&pf, "\nthe bits shifted right by 23 are e + 127");
		if (v) {
			sb_putc(&pf, '\n');
			i2f_proof(&pf, x, lo, hi);
		}
		irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	}
	free(pf.p);
}

/* ---- integer arithmetic on the fp32 pipe ---- */

struct fcone {
	struct ivc *c;
	int magic;		/* conversions by the bits where they hold */
	int nmul, nconv, nmagic, nback;
	int64_t lo, hi;		/* the widest range met */
	int bad;
};

static int
cone_op(nid a)
{
	int op = N.op[a];

	return (op == O_ADD || op == O_SUB || op == O_MUL || op == O_NEG) &&
	    TY_ISINT(N.ty[a]) && N.ty[a] != TY_BOOL;
}

/*
 * The f32 that holds integer a's value exactly, and that value's range
 * [*lo, *hi]: intervals of its own (the plan's do not see through a
 * float converted to an integer), each inside (-2^24, 2^24) and inside
 * its node's type, so the machine integers are the ideal ones.  0 and
 * bad when there is none.
 */
static nid
cone_fl(struct fcone *k, nid a, int depth, int64_t *lo, int64_t *hi)
{
	int64_t alo, ahi, blo = 0, bhi = 0, x[4];
	nid fa, fb = 0, m;
	int j;

	if (k->bad || depth > 40)
		goto bad;
	if (cone_op(a)) {
		fa = cone_fl(k, N.a[a], depth + 1, &alo, &ahi);
		if (N.op[a] != O_NEG)
			fb = cone_fl(k, N.b[a], depth + 1, &blo, &bhi);
		if (k->bad)
			return 0;
		switch (N.op[a]) {
		case O_NEG:
			*lo = -ahi;
			*hi = -alo;
			break;
		case O_ADD:
			*lo = alo + blo;
			*hi = ahi + bhi;
			break;
		case O_SUB:
			*lo = alo - bhi;
			*hi = ahi - blo;
			break;
		default:	/* below 2^24 each: the products fit 2^48 */
			x[0] = alo * blo;
			x[1] = alo * bhi;
			x[2] = ahi * blo;
			x[3] = ahi * bhi;
			*lo = *hi = x[0];
			for (j = 1; j < 4; j++) {
				*lo = x[j] < *lo ? x[j] : *lo;
				*hi = x[j] > *hi ? x[j] : *hi;
			}
			k->nmul++;
		}
		if (*lo <= -(int64_t)P24 || *hi >= (int64_t)P24 ||
		    !fits_ty(N.ty[a], *lo, *hi))
			goto bad;
		if (*lo < k->lo)
			k->lo = *lo;
		if (*hi > k->hi)
			k->hi = *hi;
		return N.op[a] == O_NEG ? mk_un(O_NEG, fa) : mk_bin(N.op[a], fa,
		    fb);
	}
	if (!irange(k->c, a, lo, hi) || *lo <= -(int64_t)P24 ||
	    *hi >= (int64_t)P24)
		goto bad;
	if (*lo < k->lo)
		k->lo = *lo;
	if (*hi > k->hi)
		k->hi = *hi;
	if (is_const(a))
		return kf((double)N.ki[a]);
	/* an integer-valued float converted: the float itself */
	if (N.op[a] == O_CAST && is_f32(N.a[a])) {
		struct fval X;
		fval_of(k->c, N.a[a], &X, 0);
		if (X.ok && X.integral) {
			k->nback++;
			return N.a[a];
		}
		goto bad;	/* its truncation is not exact arithmetic */
	}
	if (k->magic && (m = magic_i2f(a, *lo, *hi)) != 0) {
		k->nmagic++;
		return m;
	}
	k->nconv++;
	return mk_cast(TY_F32, a);
bad:
	k->bad = 1;
	return 0;
}

static void
rw_fpipe(struct ivc *c, nid root, nid n, uint8_t *inner, uint32_t lim,
    struct irw **out, int *cnt, int *cap)
{
	ty_t t = N.ty[n];
	int v;
	nid prev = 0;
	char what[200], cond[400];
	struct sbuf pf = {0};

	if (!cone_op(n) || N.op[n] == O_NEG)
		return;
	/* the operands that are part of this cone are not cones of their
	 * own: the largest one is the candidate */
	if (N.a[n] < lim && cone_op(N.a[n]))
		inner[N.a[n]] = 1;
	if (N.b[n] && N.b[n] < lim && cone_op(N.b[n]))
		inner[N.b[n]] = 1;
	if (n < lim && inner[n])
		return;
	for (v = 0; v < 2; v++) {
		struct fcone k;
		nid f, to;
		int64_t lo, hi;
		struct irw *r;
		int om;

		memset(&k, 0, sizeof k);
		k.c = c;
		k.magic = v;
		k.lo = INT64_MAX;
		k.hi = INT64_MIN;
		f = cone_fl(&k, n, 0, &lo, &hi);
		if (k.bad || !f || !k.nmul)
			break;
		om = v && lo >= -(int64_t)P22 && hi < (int64_t)P22;
		to = om ? magic_f2i(t, f) : mk_cast(t, f);
		if (to == prev)
			break;
		prev = to;
		snprintf(what, sizeof what, "the integer arithmetic %s on f32%s",
		    sstr(n), v ? ", converted by the bits" : "");
		snprintf(cond, sizeof cond, "every value in it is an integer in "
		    "[%lld, %lld], inside (-2^24, 2^24) (interval arithmetic over "
		    "the parameters' ranges)", (long long)k.lo, (long long)k.hi);
		sb_clear(&pf);
		sb_printf(&pf, "f32 holds every integer below 2^24 in magnitude: "
		    "each operand converts exactly, and each +, -, * has an exact "
		    "result that is such an integer, so no operation rounds (a*b "
		    "+ c contracted into one fma rounds a*b + c once: the same "
		    "integer)");
		if (k.nback)
			sb_printf(&pf, "\n%d operand%s converted to an integer: "
			    "the float is used as it is", k.nback, k.nback == 1 ?
			    " is an integer-valued float" :
			    "s are integer-valued floats");
		if (k.nmagic)
			sb_printf(&pf, "\n%d operand%s converted by the bits: i | "
			    "0x4B000000 (0 <= i < 2^23) and i + 0x4B400000 (-2^22 "
			    "<= i < 2^22) are the bits of 2^23 + i and 1.5*2^23 + "
			    "i, and subtracting that constant is exact", k.nmagic,
			    k.nmagic == 1 ? "" : "s");
		if (om) {
			sb_putc(&pf, '\n');
			f2i_proof(&pf, f, "the float result");
		} else
			sb_printf(&pf, "\nthe result converts back exactly: it is "
			    "an integer inside %s", ty_str(t));
		/* every integer product has these: said only when cheaper */
		r = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
		if (r)
			r->quiet = 1;
	}
	free(pf.p);
}

/*
 * The candidates at node n of root (irw_cands calls it for every
 * operation, parents first); inner marks the nodes below lim that are
 * part of a larger integer cone.
 */
/* ---- two products with a common factor ---- */

struct fterm {
	nid x, y;	/* the product x*y */
	double scale;	/* times this constant (1: none) */
	int shift;	/* the constant written as a shift by this (-1: not) */
};

/* e as scale * (x * y), the scale a constant factor or a left shift */
static int
fterm_of(nid e, struct fterm *t)
{
	nid m = e;
	double k;
	int64_t sh;

	t->scale = 1;
	t->shift = -1;
	if (N.op[e] == O_SHL && TY_ISINT(N.ty[e]) && int_of(N.b[e], &sh) &&
	    sh > 0 && sh < 31) {
		m = N.a[e];
		t->scale = (double)((int64_t)1 << sh);
		t->shift = (int)sh;
	} else if (N.op[e] == O_MUL && num_of(N.a[e], &k) &&
	    N.op[N.b[e]] == O_MUL) {
		m = N.b[e];
		t->scale = k;
	} else if (N.op[e] == O_MUL && num_of(N.b[e], &k) &&
	    N.op[N.a[e]] == O_MUL) {
		m = N.a[e];
		t->scale = k;
	}
	if (N.op[m] != O_MUL || N.ty[m] != N.ty[e])
		return 0;
	t->x = N.a[m];
	t->y = N.b[m];
	return t->scale == floor(t->scale) && t->scale != 0 &&
	    fabs(t->scale) < P24;
}

/* scale * u, as it was written */
static nid
fterm_rest(const struct fterm *t, nid u)
{
	ty_t ty = N.ty[u];

	if (t->scale == 1)
		return u;
	if (t->shift >= 0)
		return mk_bin(O_SHL, u, mk_const_i(ty, t->shift));
	return mk_bin(O_MUL, ty == TY_F32 ? kf(t->scale) : mk_const_i(ty,
	    (int64_t)t->scale), u);
}

static double
fmag(const struct fval *v)
{
	return fmax(fabs(v->lo), fabs(v->hi));
}

static void
rw_factor(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	struct fterm A, B;
	ty_t t = N.ty[n];
	int op = N.op[n], i, j, sized = TYK(t) == K_UINT || TYK(t) == K_SINT;
	nid f = 0, ua = 0, ub = 0, to;
	char what[200], cond[400], why[160];
	struct sbuf pf = {0};
	struct nm FS;
	const char *fs;
	struct irw *r;

	if ((!sized && t != TY_F32) || !fterm_of(N.a[n], &A) ||
	    !fterm_of(N.b[n], &B))
		return;
	if (op != O_ADD && op != O_SUB && !(sized && (op == O_OR ||
	    op == O_XOR)))
		return;
	for (i = 0; i < 2 && !f; i++)
		for (j = 0; j < 2 && !f; j++)
			if ((i ? A.y : A.x) == (j ? B.y : B.x)) {
				f = i ? A.y : A.x;
				ua = i ? A.x : A.y;
				ub = j ? B.x : B.y;
			}
	if (!f || is_const(f))
		return;
	cond[0] = 0;
	if (op == O_OR || op == O_XOR) {
		/* the fields apart: one below 2^k, the other a multiple of it */
		const struct fterm *lo_t = A.shift >= 0 ? &B : &A;
		const struct fterm *hi_t = A.shift >= 0 ? &A : &B;
		nid lo_e = A.shift >= 0 ? N.b[n] : N.a[n];
		int64_t lo, hi;

		if (hi_t->shift < 0 || lo_t->shift >= 0 || TYK(t) != K_UINT ||
		    !irw_range(c, lo_e, 0, &lo, &hi, why, sizeof why) || lo < 0 ||
		    hi >= ((int64_t)1 << hi_t->shift))
			return;
		snprintf(cond, sizeof cond, "%s in [%lld, %lld] (%s), below 2^%d, "
		    "and the other side a multiple of 2^%d: no bit in both, so "
		    "the %s is their sum", sstr(lo_e), (long long)lo,
		    (long long)hi, why, hi_t->shift, hi_t->shift, op == O_OR ?
		    "or" : "xor");
		op = O_ADD;
	}
	if (t == TY_F32) {
		struct fval X, Y, F;
		double ma, mb, mf;

		fval_of(c, ua, &X, 0);
		fval_of(c, ub, &Y, 0);
		fval_of(c, f, &F, 0);
		if (!X.ok || !Y.ok || !F.ok || !X.integral || !Y.integral ||
		    !F.integral)
			return;
		ma = fabs(A.scale) * fmag(&X);
		mb = fabs(B.scale) * fmag(&Y);
		mf = fmag(&F);
		if ((ma + mb) * mf >= P24 || ma + mb >= P24)
			return;
		snprintf(cond, sizeof cond, "%s, %s and %s are integers, and "
		    "every value on both sides is at most %.9g in magnitude, "
		    "below 2^24", sstr(ua), sstr(ub), sstr(f), (ma + mb) * mf);
		sb_printf(&pf, "f32 holds every integer below 2^24 in magnitude, "
		    "so no product or sum on either side rounds, and in the "
		    "integers ");
	} else {
		if (!cond[0])
			snprintf(cond, sizeof cond, "any operands: %s arithmetic "
			    "is modulo 2^%d, a ring", ty_str(t), TYW(t));
		sb_printf(&pf, "modulo 2^%d ", TYW(t));
	}
	to = mk_bin(O_MUL, mk_bin(op, fterm_rest(&A, ua), fterm_rest(&B, ub)),
	    f);
	fs = nm(&FS, f);
	sb_printf(&pf, "multiplication distributes over %s, and a shift left "
	    "by k or a constant factor moves inside the product: p*%s %s "
	    "K*(q*%s) = (p %s K*q)*%s", op == O_SUB ? "subtraction" : "addition",
	    fs, op == O_SUB ? "-" : "+", fs, op == O_SUB ? "-" : "+", fs);
	snprintf(what, sizeof what, "%s as one product by the common factor %s",
	    sstr(n), fs);
	r = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	if (r && sized)
		r->nosat = r->factor = 1;	/* multipliers on both sides */
	free(pf.p);
}

/*
 * A factor shared across a sum of any number of terms, each a product of
 * any number of factors: f p1 + f p2 + ... + rest = f (p1 + p2 + ...) +
 * rest, the factor most terms share taken out once.  A constant factor
 * and a shift left by one are a term's constant; the rest are its
 * factors.  Distributivity modulo 2^w in a sized integer type, so any
 * operands; rw_factor has two products of two factors (and the float
 * case), so this says only what that one does not.
 */
#define FS_TERMS 16
#define FS_FACS 8

struct sterm {
	int neg;
	uint64_t k;		/* the constant part */
	nid f[FS_FACS];
	int nf;
};

static int
prod_facs(nid e, ty_t t, struct sterm *s, int depth)
{
	if (depth > 64)
		return 0;
	if (N.ty[e] == t && N.op[e] == O_MUL)
		return prod_facs(N.a[e], t, s, depth + 1) &&
		    prod_facs(N.b[e], t, s, depth + 1);
	if (N.ty[e] == t && N.op[e] == O_SHL && is_const(N.b[e]) &&
	    N.ki[N.b[e]] >= 0 && N.ki[N.b[e]] < TYW(t)) {
		s->k *= (uint64_t)1 << N.ki[N.b[e]];
		return prod_facs(N.a[e], t, s, depth + 1);
	}
	if (is_const(e)) {
		s->k *= (uint64_t)N.ki[e];
		return 1;
	}
	if (s->nf == FS_FACS)
		return 0;
	s->f[s->nf++] = e;
	return 1;
}

static int
sum_terms(nid e, ty_t t, int neg, struct sterm *s, int *n, int depth)
{
	if (depth > 64)
		return 0;
	if (N.ty[e] == t && (N.op[e] == O_ADD || N.op[e] == O_SUB))
		return sum_terms(N.a[e], t, neg, s, n, depth + 1) &&
		    sum_terms(N.b[e], t, N.op[e] == O_SUB ? !neg : neg, s, n,
		    depth + 1);
	if (N.ty[e] == t && N.op[e] == O_NEG)
		return sum_terms(N.a[e], t, !neg, s, n, depth + 1);
	if (*n == FS_TERMS)
		return 0;
	s[*n].neg = neg;
	s[*n].k = 1;
	s[*n].nf = 0;
	if (!prod_facs(e, t, &s[*n], 0))
		return 0;
	(*n)++;
	return 1;
}

/* term s without its factor at skip (-1: none), its constant as written:
 * a shift for a power of two, a multiply otherwise */
static nid
sterm_node(const struct sterm *s, int skip, ty_t t)
{
	uint64_t m = TYW(t) >= 64 ? ~0ULL : (1ULL << TYW(t)) - 1;
	uint64_t k = s->k & m;
	nid r = 0;
	int i;

	for (i = 0; i < s->nf; i++)
		if (i != skip)
			r = r ? mk_bin(O_MUL, r, s->f[i]) : s->f[i];
	if (!r)
		return mk_const_i(t, (int64_t)k);
	if (k == 1)
		return r;
	if (k && !(k & (k - 1)))
		return mk_bin(O_SHL, r, mk_const_i(t, __builtin_ctzll(k)));
	return mk_bin(O_MUL, mk_const_i(t, (int64_t)k), r);
}

/* acc + (or -) e, acc 0 for none */
static nid
sum_add(nid acc, nid e, int neg, ty_t t)
{
	if (!acc)
		return neg ? mk_un(O_NEG, e) : e;
	return mk_bin(neg ? O_SUB : O_ADD, acc, e);
}

static void
rw_factor_sum(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	ty_t t = N.ty[n];
	struct sterm s[FS_TERMS];
	int ns = 0, i, j, q, bestc = 0, has[FS_TERMS], in1 = 0, plain2 = 1;
	nid f = 0, in = 0, rest = 0, to;
	char what[200], cond[200];
	struct sbuf pf = {0};
	struct irw *r;

	(void)c;
	if ((TYK(t) != K_UINT && TYK(t) != K_SINT) || !sum_terms(n, t, 0, s,
	    &ns, 0) || ns < 2)
		return;
	/* the factor most terms have */
	for (i = 0; i < ns; i++)
		for (j = 0; j < s[i].nf; j++) {
			int k2 = 0;
			for (q = 0; q < ns; q++) {
				int z;
				for (z = 0; z < s[q].nf && s[q].f[z] != s[i].f[j]; z++)
					;
				k2 += z < s[q].nf;
			}
			if (k2 > bestc) {
				bestc = k2;
				f = s[i].f[j];
			}
		}
	if (bestc < 2)
		return;
	/* rw_factor's case: two terms, each two factors */
	for (i = 0; i < ns; i++)
		if (s[i].nf != 2)
			plain2 = 0;
	if (ns == 2 && plain2)
		return;
	for (i = 0; i < ns; i++) {
		for (has[i] = -1, j = 0; j < s[i].nf && has[i] < 0; j++)
			if (s[i].f[j] == f)
				has[i] = j;
		if (has[i] >= 0) {
			in = sum_add(in, sterm_node(&s[i], has[i], t), s[i].neg, t);
			in1++;
		} else
			rest = sum_add(rest, sterm_node(&s[i], -1, t), s[i].neg, t);
	}
	to = mk_bin(O_MUL, in, f);
	if (rest)
		to = mk_bin(O_ADD, rest, to);
	snprintf(what, sizeof what, "%s: the factor %s of %d terms taken out "
	    "once", sstr(n), sstr(f), in1);
	snprintf(cond, sizeof cond, "any operands: %s arithmetic is modulo "
	    "2^%d, a ring", ty_str(t), TYW(t));
	sb_printf(&pf, "modulo 2^%d multiplication distributes over addition "
	    "and subtraction, and a product commutes and associates (a shift "
	    "left by k is a product by 2^k): the %d terms with the factor %s "
	    "are %s times the sum of what each has besides it", TYW(t), in1,
	    sstr(f), sstr(f));
	r = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	if (r)
		r->nosat = r->derived = r->factor = 1;
	free(pf.p);
}

void
ifx_cands(struct ivc *c, nid root, nid n, uint8_t *inner, uint32_t lim,
    struct irw **out, int *cnt, int *cap)
{
	switch (N.op[n]) {
	case O_CAST:
		rw_pow2(c, root, n, out, cnt, cap);
		rw_i2f(c, root, n, out, cnt, cap);
		rw_f2i(c, root, n, out, cnt, cap);
		break;
	case O_FLOOR: case O_CEIL:
		rw_floor(c, root, n, out, cnt, cap);
		break;
	case O_ROUNDEVEN:
		rw_roundeven(c, root, n, out, cnt, cap);
		break;
	case O_MUL:
		rw_roundmul(c, root, n, out, cnt, cap);
		rw_ldexp(c, root, n, out, cnt, cap);
		rw_fpipe(c, root, n, inner, lim, out, cnt, cap);
		break;
	case O_SUB:
		rw_ilog2(c, root, n, out, cnt, cap);
		rw_fpipe(c, root, n, inner, lim, out, cnt, cap);
		rw_factor(c, root, n, out, cnt, cap);
		rw_factor_sum(c, root, n, out, cnt, cap);
		break;
	case O_ADD:
		rw_fpipe(c, root, n, inner, lim, out, cnt, cap);
		rw_factor(c, root, n, out, cnt, cap);
		rw_factor_sum(c, root, n, out, cnt, cap);
		break;
	case O_OR: case O_XOR:
		rw_factor(c, root, n, out, cnt, cap);
		break;
	}
}
