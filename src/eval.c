/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Evaluation.
 *
 * A plan is the slice of the node DAG reachable from a few roots, copied
 * into dense local arrays in topological (= id) order.  plan_run pushes a
 * chunk of up to CH domain points through it one node at a time, so every
 * operation is a loop over a contiguous column.  Each value carries a flag
 * byte (F_*) saying whether it is defined and whether the operation that
 * made it wrapped.
 *
 * Two semantics share the code: machine (each value in its declared width,
 * wrapping) and ideal (every integer type is an unbounded integer, checked
 * against the int64 range).  Comparing the two is how `wrap` works.
 */
#include "ucas.h"

#include "texalu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void
ity_of(struct ity *it, ty_t t, int ideal)
{
	int k = TYK(t), w = TYW(t);

	memset(it, 0, sizeof *it);
	if (k == K_LIT)
		k = K_INT;
	if (ideal && (k == K_UINT || k == K_SINT))
		k = K_INT;
	if (ideal && k == K_FLOAT)
		k = K_REAL;
	if (k == K_INT || k == K_REAL)
		w = 64;
	if (k == K_BOOL)
		w = 1;
	it->k = (uint8_t)k;
	it->w = (uint8_t)w;
	it->mask = w >= 64 ? ~0ULL : (1ULL << w) - 1;
	if (k == K_SINT || k == K_INT) {
		it->smin = w >= 64 ? INT64_MIN : -(int64_t)(1ULL << (w - 1));
		it->smax = w >= 64 ? INT64_MAX : (int64_t)((1ULL << (w - 1)) - 1);
	} else {
		/* signed view of an unsigned width, for nsw checks */
		it->smin = w >= 64 ? INT64_MIN : -(int64_t)(1ULL << (w - 1));
		it->smax = w >= 64 ? INT64_MAX : (int64_t)((1ULL << (w - 1)) - 1);
		if (w == 1)
			it->smin = -1, it->smax = 0;
	}
}

static inline uint64_t
asu(const struct ity *t, int64_t v)
{
	return (uint64_t)v & t->mask;
}

static inline int64_t
ass(const struct ity *t, int64_t v)
{
	uint64_t m, x;

	if (t->w >= 64)
		return v;
	m = 1ULL << (t->w - 1);
	x = (uint64_t)v & t->mask;
	return (int64_t)((x ^ m) - m);
}

static inline int64_t
inorm(const struct ity *t, uint64_t v)
{
	switch (t->k) {
	case K_UINT:
	case K_BOOL:
		return (int64_t)(v & t->mask);
	case K_SINT:
		return ass(t, (int64_t)v);
	}
	return (int64_t)v;
}

static int
add_ovf(int64_t a, int64_t b, int64_t *r)
{
	if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b))
		return 1;
	*r = a + b;
	return 0;
}

static int
sub_ovf(int64_t a, int64_t b, int64_t *r)
{
	if ((b < 0 && a > INT64_MAX + b) || (b > 0 && a < INT64_MIN + b))
		return 1;
	*r = a - b;
	return 0;
}

static int
mul_ovf(int64_t a, int64_t b, int64_t *r)
{
	if (a == 0 || b == 0) {
		*r = 0;
		return 0;
	}
	if (a > 0) {
		if (b > 0 ? a > INT64_MAX / b : b < INT64_MIN / a)
			return 1;
	} else {
		if (b > 0 ? a < INT64_MIN / b : b < INT64_MAX / a)
			return 1;
	}
	*r = a * b;
	return 0;
}

static uint64_t
isqrt64(uint64_t v)
{
	uint64_t r = (uint64_t)sqrt((double)v);

	while (r > 0 && (r > 0xffffffffULL || r * r > v))
		r--;
	while (r < 0xffffffffULL && (r + 1) * (r + 1) <= v)
		r++;
	return r;
}

/* +, -, *, <<, unary - with wrap detection in both interpretations */
static int64_t
arith(int op, const struct ity *t, int64_t a, int64_t b, uint8_t *fl)
{
	uint64_t ua = asu(t, a), ub = asu(t, b), raw = 0;
	int64_t sa = ass(t, a), sb = ass(t, b), sr = 0;
	int uw = 0, so = 0, sh;

	switch (op) {
	case O_ADD:
		raw = ua + ub;
		uw = raw < ua || raw > t->mask;
		so = add_ovf(sa, sb, &sr);
		break;
	case O_SUB:
		raw = ua - ub;
		uw = ua < ub;
		so = sub_ovf(sa, sb, &sr);
		break;
	case O_MUL:
		raw = ua * ub;
		uw = ub != 0 && ua > t->mask / ub;
		so = mul_ovf(sa, sb, &sr);
		break;
	case O_NEG:
		raw = 0 - ua;
		uw = ua != 0;
		so = sub_ovf(0, sa, &sr);
		break;
	case O_SHL:
		if (b < 0 || b >= t->w) {
			*fl |= F_UNDEF | F_UB;
			return 0;
		}
		sh = (int)b;
		raw = sh ? ua << sh : ua;
		uw = sh && (ua >> (t->w - sh)) != 0;
		if (t->k == K_INT) {
			so = sh && (sa > (INT64_MAX >> sh) || sa < (INT64_MIN >> sh));
			sr = so ? 0 : (int64_t)((uint64_t)sa << sh);
		} else {
			sr = ass(t, (int64_t)raw);
			so = (sr >> sh) != sa;
			sr = so ? 0 : sr;
			if (!so)
				return inorm(t, raw);
			*fl |= (uw ? F_UW : 0) | F_SW;
			if (t->k == K_SINT)
				*fl |= F_UB;
			return inorm(t, raw);
		}
		break;
	}
	if (t->k == K_INT) {
		if (so) {
			*fl |= F_OVF | F_UNDEF;
			return 0;
		}
		return sr;
	}
	if (!so && (sr < t->smin || sr > t->smax))
		so = 1;
	if (uw)
		*fl |= F_UW;
	if (so) {
		*fl |= F_SW;
		if (t->k == K_SINT)
			*fl |= F_UB;
	}
	return inorm(t, raw);
}

static int
cmp_lt(const struct ity *t, int64_t a, int64_t b)
{
	if (t->k == K_UINT || t->k == K_BOOL)
		return asu(t, a) < asu(t, b);
	return ass(t, a) < ass(t, b);
}

int64_t
scalar_int(int op, const struct ity *t, const struct ity *ta, int64_t a,
    int64_t b, int64_t c, uint8_t *fl)
{
	uint64_t ua, ub, u;
	int64_t sa, sb, r;
	int i, neg, rneg;

	(void)c;
	switch (op) {
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_SHL:
		return arith(op, t, a, b, fl);
	case O_NEG:
		return arith(op, t, a, 0, fl);
	case O_DIV:
	case O_MOD:
		if (asu(t, b) == 0) {
			*fl |= F_UNDEF | F_UB;
			return 0;
		}
		if (t->k == K_UINT || t->k == K_BOOL) {
			ua = asu(t, a);
			ub = asu(t, b);
			return inorm(t, op == O_DIV ? ua / ub : ua % ub);
		}
		sa = ass(t, a);
		sb = ass(t, b);
		if (sa == t->smin && sb == -1) {
			if (t->k == K_INT) {
				*fl |= F_OVF | F_UNDEF;
				return 0;
			}
			*fl |= F_SW | F_UB;
			return op == O_DIV ? sa : 0;
		}
		return op == O_DIV ? sa / sb : sa % sb;
	case O_AND:
		return inorm(t, (uint64_t)a & (uint64_t)b);
	case O_OR:
		return inorm(t, (uint64_t)a | (uint64_t)b);
	case O_XOR:
		return inorm(t, (uint64_t)a ^ (uint64_t)b);
	case O_NOT:
		return inorm(t, ~(uint64_t)a);
	case O_SHR:
		if (b < 0 || b >= t->w) {
			*fl |= F_UNDEF | F_UB;
			return 0;
		}
		if (t->k == K_UINT || t->k == K_BOOL)
			return inorm(t, asu(t, a) >> b);
		sa = ass(t, a);
		return sa < 0 ? ~(~sa >> b) : sa >> b;
	case O_POW:
		if (b < 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = inorm(t, 1);
		if (b > 4096) {
			/* square and multiply; flags are approximate */
			int64_t base = a;
			while (b) {
				if (b & 1)
					r = arith(O_MUL, t, r, base, fl);
				b >>= 1;
				if (b)
					base = arith(O_MUL, t, base, base, fl);
				if (*fl & F_UNDEF)
					return 0;
			}
			return r;
		}
		for (i = 0; i < b; i++) {
			r = arith(O_MUL, t, r, a, fl);
			if (*fl & F_UNDEF)
				return 0;
		}
		return r;
	case O_MIN:
		return cmp_lt(t, b, a) ? b : a;
	case O_MAX:
		return cmp_lt(t, a, b) ? b : a;
	case O_ABS:
		if (t->k == K_UINT || t->k == K_BOOL)
			return a;
		sa = ass(t, a);
		if (sa >= 0)
			return sa;
		return arith(O_NEG, t, sa, 0, fl);
	case O_SIGN:
		if (t->k == K_UINT || t->k == K_BOOL)
			return asu(t, a) != 0;
		sa = ass(t, a);
		return (sa > 0) - (sa < 0);
	case O_ISQRT:
		if (t->k == K_UINT || t->k == K_BOOL)
			return (int64_t)isqrt64(asu(t, a));
		if (ass(t, a) < 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		return (int64_t)isqrt64((uint64_t)ass(t, a));
	case O_CLZ:
	case O_CTZ:
		ua = asu(t, a);
		if (ua == 0) {
			*fl |= F_UB;
			return t->w;
		}
		r = 0;
		if (op == O_CLZ)
			while (!(ua >> (t->w - 1 - r) & 1))
				r++;
		else
			while (!(ua >> r & 1))
				r++;
		return r;
	case O_POPCNT:
		ua = asu(t, a);
		for (r = 0; ua; ua &= ua - 1)
			r++;
		return r;
	case O_EQ: return asu(ta, a) == asu(ta, b);
	case O_NE: return asu(ta, a) != asu(ta, b);
	case O_LT: return cmp_lt(ta, a, b);
	case O_LE: return !cmp_lt(ta, b, a);
	case O_GT: return cmp_lt(ta, b, a);
	case O_GE: return !cmp_lt(ta, a, b);
	case O_LNOT: return a == 0;
	case O_LAND: return a && b;
	case O_LOR: return a || b;
	case O_CAST:
		if (ta->k == K_UINT || ta->k == K_BOOL) {
			neg = 0;
			u = asu(ta, a);
		} else {
			neg = ass(ta, a) < 0;
			u = (uint64_t)ass(ta, a);
		}
		if (t->k == K_INT) {
			if (!neg && u > (uint64_t)INT64_MAX) {
				*fl |= F_OVF | F_UNDEF;
				return 0;
			}
			return (int64_t)u;
		}
		r = inorm(t, u);
		rneg = (t->k == K_SINT) && r < 0;
		if (rneg != neg || (uint64_t)r != (t->k == K_SINT ? (uint64_t)r :
		    (uint64_t)r) || (t->k != K_SINT && (uint64_t)r != u) ||
		    (t->k == K_SINT && (uint64_t)r != u))
			*fl |= F_TRUNC;
		return r;
	}
	*fl |= F_UNDEF;
	return 0;
}

double
scalar_real(int op, double a, double b, uint8_t *fl)
{
	double r;

	switch (op) {
	case O_ADD: r = a + b; break;
	case O_SUB: r = a - b; break;
	case O_MUL: r = a * b; break;
	case O_DIV:
		if (b == 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = a / b;
		break;
	case O_MOD:
		if (b == 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = fmod(a, b);
		break;
	case O_POW:
		if ((a < 0 && b != floor(b)) || (a == 0 && b < 0)) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = pow(a, b);
		break;
	case O_NEG: r = -a; break;
	case O_ABS: r = fabs(a); break;
	case O_SIGN: r = (a > 0) - (a < 0); break;
	case O_SQRT:
		if (a < 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = sqrt(a);
		break;
	case O_EXP: r = exp(a); break;
	case O_LOG:
		if (a <= 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = log(a);
		break;
	case O_LAMBERTW:
	case O_LAMBERTWM1:
		r = lambert_w(a, op == O_LAMBERTWM1);
		if (isnan(r)) {
			*fl |= F_UNDEF;
			return 0;
		}
		break;
	case O_SIN: r = sin(a); break;
	case O_COS: r = cos(a); break;
	case O_TAN:
		if (fabs(cos(a)) < 1e-15) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = tan(a);
		break;
	case O_SEC:
		if (fabs(cos(a)) < 1e-15) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = 1 / cos(a);
		break;
	case O_CSC:
	case O_COT:
		if (sin(a) == 0) {
			*fl |= F_UNDEF;
			return 0;
		}
		r = op == O_CSC ? 1 / sin(a) : cos(a) / sin(a);
		break;
	case O_ATAN: r = atan(a); break;
	case O_ASIN: case O_ACOS: case O_ATAN2: case O_SINH: case O_COSH:
	case O_TANH: case O_ASINH: case O_ACOSH: case O_ATANH:   /* forms.c */
		if (!forms_scalar(op, a, b, &r)) {
			*fl |= F_UNDEF;
			return 0;
		}
		break;
	case O_CONJ: case O_RE: case O_IM: case O_ARG:   /* hyper.c */
		if (!hyper_scalar(op, a, &r)) {
			*fl |= F_UNDEF;
			return 0;
		}
		break;
	case O_COPYSIGN: r = copysign(a, b); break;
	case O_FLOOR: r = floor(a); break;
	case O_CEIL: r = ceil(a); break;
	case O_ROUNDEVEN: r = nearbyint(a); break;
	case O_MIN: r = a < b ? a : b; break;
	case O_MAX: r = a > b ? a : b; break;
	case O_EQ: return a == b;
	case O_NE: return a != b;
	case O_LT: return a < b;
	case O_LE: return a <= b;
	case O_GT: return a > b;
	case O_GE: return a >= b;
	default:
		*fl |= F_UNDEF;
		return 0;
	}
	if (isnan(r) || isinf(r)) {
		*fl |= F_UNDEF;
		return 0;
	}
	return r;
}

/*
 * Lambert W: the w with w e^w = x.  Branch 0 (principal) on [-1/e, inf),
 * branch -1 on [-1/e, 0).  Halley's iteration from a series near the
 * branch point, an asymptotic guess for large |log|, log1p otherwise.
 */
double
lambert_w(double x, int branch)
{
	const double em1 = 0.36787944117144233;	/* 1/e */
	double w, p, ew, f, d, wn;
	int i;

	if (isnan(x) || x < -em1 * (1 + 1e-15) || (branch && x >= 0))
		return NAN;
	if (x <= -em1)
		return -1;
	if (x == 0 && !branch)
		return 0;
	if (isinf(x))
		return x;
	p = sqrt(2 * (2.718281828459045 * x + 1));
	if (branch) {
		if (x < -0.25)
			w = -1 - p - p * p / 3 - 11 * p * p * p / 72;
		else {
			double l1 = log(-x), l2 = log(-l1);
			w = l1 - l2 + l2 / l1;
		}
	} else if (x < -0.3)
		w = -1 + p - p * p / 3 + 11 * p * p * p / 72;
	else if (x < 3)
		w = log1p(x) * (1 - log1p(log1p(x)) / (2 + log1p(x)));
	else {
		double l1 = log(x), l2 = log(l1);
		w = l1 - l2 + l2 / l1;
	}
	for (i = 0; i < 64; i++) {
		ew = exp(w);
		f = w * ew - x;
		if (w == -1)
			break;
		d = ew * (w + 1) - (w + 2) * f / (2 * w + 2);
		wn = w - f / d;
		if (!isfinite(wn))
			break;
		if (fabs(wn - w) <= 1e-16 * (1 + fabs(wn))) {
			w = wn;
			break;
		}
		w = wn;
	}
	return w;
}

/* ---- IEEE helpers ---- */

/*
 * binary16: a sign bit, 5 exponent bits (bias 15) and 10 fraction bits.
 * Its finite values are m 2^(e - 10) with an integer |m| < 2^11 and
 * -14 <= e <= 15: the subnormals are the multiples of 2^-24 below 2^-14,
 * the largest finite value is 65504 = (2^11 - 1) 2^5.  The host has no
 * such type in C99, so the rounding is done here, on the double.
 *
 * fround16(v): v rounded to nearest, ties to even, for any double v.
 *
 * Derivation.  Let a = |v|, finite and nonzero.  At or above the overflow
 * threshold 65520 = 2^16 (1 - 2^-12), the midpoint of 65504 and 2^16 (which
 * ties to the even 2^16, out of range), the result is inf.  Below it, let
 * a lie in [2^e, 2^(e + 1)) (frexp), e raised to -14 for the subnormals: the
 * binary16 values around a are the multiples of u = 2^(e - 10).  q = a / u
 * is exact (a power of two, and 2^-1050 <= q < 2^11), f = floor(q) and
 * d = q - f are exact (d is a multiple of q's last place below 1), so
 * comparing d with 1/2 decides the rounding with no error, f's parity the
 * tie; (f or f + 1) u is exact, at most 2^11 u, the next binade's first
 * value, and below 2^16 as a < 65520 gives q < 2047.5.  A result of 0 keeps
 * v's sign.
 */
static double
fround16(double v)
{
	double a = fabs(v), u, q, f, d;
	int e;

	if (isnan(v) || isinf(v) || a == 0)
		return v;
	if (a >= 65520.0)
		return v < 0 ? -INFINITY : INFINITY;
	(void)frexp(a, &e);	/* a in [2^(e - 1), 2^e) */
	e = e - 1 < -14 ? -14 : e - 1;
	u = ldexp(1.0, e - 10);
	q = a / u;
	f = floor(q);
	d = q - f;
	if (d > 0.5 || (d == 0.5 && fmod(f, 2.0) != 0))
		f += 1;
	return copysign(f * u, v);
}

double
fround(int w, double v)
{
	return w == 32 ? (double)(float)v : w == 16 ? fround16(v) : v;
}

/*
 * The bits of a binary16 value (v is rounded first).  A NaN keeps its sign
 * and its fraction's top 10 bits, which is where frombits puts a binary16
 * NaN's fraction: asuint(asfloat(b)) is b.  A NaN whose top 10 fraction bits
 * are 0 (one made in double with a low payload) is the quiet NaN 0x7e00.
 */
static uint64_t
fbits16(double v)
{
	uint64_t u, sign;
	double a;
	int e;

	memcpy(&u, &v, 8);
	sign = u >> 63 << 15;
	if (isnan(v)) {
		uint64_t frac = u >> 42 & 0x3ff;
		return sign | 0x7c00 | (frac ? frac : 0x200);
	}
	a = fabs(fround16(v));
	if (isinf(a))
		return sign | 0x7c00;
	if (a < 6.103515625e-05)	/* 2^-14: subnormal, or 0 */
		return sign | (uint64_t)(a * 16777216.0);
	(void)frexp(a, &e);
	return sign | (uint64_t)(e - 1 + 15) << 10 |
	    ((uint64_t)ldexp(a, 10 - (e - 1)) & 0x3ff);
}

static double
frombits16(uint64_t b)
{
	uint64_t frac = b & 0x3ff, u;
	int e = (int)(b >> 10 & 0x1f);
	double v;

	if (e == 31) {
		u = (b >> 15 & 1) << 63 | 0x7ff0000000000000ULL | frac << 42;
		memcpy(&v, &u, 8);
		return v;
	}
	v = e ? ldexp((double)(frac | 0x400), e - 25) :
	    ldexp((double)frac, -24);
	return b >> 15 & 1 ? -v : v;
}

uint64_t
fbits(int w, double v)
{
	if (w == 16)
		return fbits16(v);
	if (w == 32) {
		float f = (float)v;
		uint32_t u;
		memcpy(&u, &f, 4);
		return u;
	} else {
		uint64_t u;
		memcpy(&u, &v, 8);
		return u;
	}
}

double
frombits(int w, uint64_t b)
{
	if (w == 16)
		return frombits16(b);
	if (w == 32) {
		uint32_t u = (uint32_t)b;
		float f;
		memcpy(&f, &u, 4);
		return f;
	} else {
		double d;
		memcpy(&d, &b, 8);
		return d;
	}
}

/* the decimal digits that tell every value of a float of w bits apart */
int
fdigits(int w)
{
	return w == 16 ? 5 : w == 32 ? 9 : 17;
}

/* the sign bit of a float of w bits */
static uint64_t
fsign(int w)
{
	return 1ULL << (w - 1);
}

/* a total order on the non-NaN values: consecutive keys are adjacent floats */
int64_t
fkey(int w, double v)
{
	uint64_t u = fbits(w, v);
	uint64_t sign = fsign(w);

	if (u & sign)
		return -(int64_t)(u & (sign - 1));
	return (int64_t)u;
}

double
fromkey(int w, int64_t k)
{
	uint64_t sign = fsign(w);

	if (k < 0)
		return frombits(w, (uint64_t)(-k) | sign);
	return frombits(w, (uint64_t)k);
}

/*
 * fma64(a, b, c): the f64 fused multiply-add, correctly rounded (round to
 * nearest even), in integer arithmetic.  The host libm's fma() is not
 * trusted: FreeBSD 15's software fma rounds
 * fma(-0x1.5af1d78b58c4p+66, -0x1.b428a02b5f643p-121, -1) to -1, where
 * the exact value -1 + 6.41e-17 is nearer to -1 + 2^-53 (x86check
 * found it against vfmadd).
 *
 * Derivation.  Finite a = ma 2^ea, b = mb 2^eb, c = mc 2^ec with integer
 * significands below 2^53 (subnormals included).
 * 1. If a*b = 0 the exact result is 0 + c with IEEE's sign rule for a
 *    zero sum; double p + c with p = a*b (an exact signed zero) is that.
 * 2. Otherwise X = ma mb (below 2^106, exact in 128 bits) times 2^(ea+eb)
 *    and Y = mc 2^ec.  Both are shifted left so their leading bit is bit
 *    125 (exact), the larger exponent first; the smaller is shifted right
 *    by d, the bits shifted out ORed into bit 0 (sticky).  If d <= 1
 *    nothing is shifted out: a significand has at most 106 bits, so its
 *    lowest set bit is at 20 or above, and the sum or difference R is
 *    exact.  If d >= 2 and bits were shifted out, the larger operand's
 *    bit 0 is 0 (lowest set bit >= 20), so R is odd and the exact value
 *    lies strictly between R - 1 and R + 1 on the side of R it was
 *    truncated from: R is the exact value rounded to odd at bit 0.  R is
 *    at least 2^125 - 2^124, so its leading bit is at 124 or higher and
 *    rounding keeps bits s >= 72 and up.  As in fma32 (step 5), every
 *    rounding boundary (multiples of 2^(s-1)) is an even integer, and
 *    round to nearest of R equals round to nearest of the exact value.
 * 3. Rounding: the kept bits q = R >> s and the remainder r = R mod 2^s,
 *    with s chosen for 53 bits, or for the 2^-1074 grid when the value
 *    is subnormal; q + 1 when r > 2^(s-1), or r = 2^(s-1) and q is odd.
 *    q * 2^(e+s) is then exact in double by ldexp (q <= 2^53, and on the
 *    subnormal grid when below 2^-1022), or inf: the overflow of RN.
 * Inf and NaN operands: no rounding, a*b + c is IEEE's fma except when
 * a and b are finite and c is not (a*b may overflow): then c.
 */
struct u128 {
	uint64_t hi, lo;
};

static struct u128
mul64(uint64_t a, uint64_t b)
{
	uint64_t al = a & 0xffffffffu, ah = a >> 32, bl = b & 0xffffffffu,
	    bh = b >> 32, ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh,
	    mid = (ll >> 32) + (lh & 0xffffffffu) + (hl & 0xffffffffu);
	struct u128 r;

	r.lo = (mid << 32) | (ll & 0xffffffffu);
	r.hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
	return r;
}

static int
lead128(struct u128 x)	/* the leading bit's index; -1 for 0 */
{
	int k;

	for (k = 127; k >= 0; k--)
		if (k >= 64 ? (x.hi >> (k - 64)) & 1 : (x.lo >> k) & 1)
			return k;
	return -1;
}

static struct u128
shl128(struct u128 x, int n)
{
	struct u128 r;

	if (n <= 0)
		return x;
	if (n >= 64) {
		r.hi = x.lo << (n - 64);
		r.lo = 0;
	} else {
		r.hi = x.hi << n | x.lo >> (64 - n);
		r.lo = x.lo << n;
	}
	return r;
}

/* x >> n; *rest: whether nonzero bits were shifted out */
static struct u128
shr128(struct u128 x, int n, int *rest)
{
	struct u128 r;

	*rest = 0;
	if (n <= 0)
		return x;
	if (n >= 128) {
		*rest = (x.hi | x.lo) != 0;
		r.hi = r.lo = 0;
		return r;
	}
	if (n >= 64) {
		*rest = x.lo != 0 || (n > 64 && (x.hi << (128 - n)) != 0);
		r.lo = x.hi >> (n - 64);
		r.hi = 0;
	} else {
		*rest = (x.lo << (64 - n)) != 0;
		r.lo = x.lo >> n | x.hi << (64 - n);
		r.hi = x.hi >> n;
	}
	return r;
}

static void
split_double(double x, uint64_t *m, int *e)	/* finite x != 0: |x| = m 2^e */
{
	uint64_t u;
	int be;

	memcpy(&u, &x, sizeof u);
	be = (int)((u >> 52) & 0x7ff);
	*m = u & 0xfffffffffffffULL;
	if (be) {
		*m |= 1ULL << 52;
		*e = be - 1075;
	} else
		*e = -1074;
}

double
fma64(double a, double b, double c)
{
	uint64_t ma, mb, mc;
	int ea, eb, ec, sx, sy, ex, ey, rest, L, s, sticky = 0;
	struct u128 X, Y, R;
	double r;

	if (!isfinite(a) || !isfinite(b))
		return a * b + c;
	if (!isfinite(c))
		return c;
	if (a == 0 || b == 0)
		return a * b + c;
	split_double(a, &ma, &ea);
	split_double(b, &mb, &eb);
	X = mul64(ma, mb);
	ex = ea + eb;
	sx = (a < 0) != (b < 0);
	if (c == 0)
		Y.hi = Y.lo = 0, ey = ex, sy = sx;
	else {
		split_double(c, &mc, &ec);
		Y.hi = 0;
		Y.lo = mc;
		ey = ec;
		sy = c < 0;
	}
	/* leading bits at 125 */
	L = lead128(X);
	X = shl128(X, 125 - L);
	ex -= 125 - L;
	if (Y.lo) {
		L = lead128(Y);
		Y = shl128(Y, 125 - L);
		ey -= 125 - L;
	} else
		ey = ex;
	if (ey > ex || (ey == ex && (Y.hi > X.hi || (Y.hi == X.hi &&
	    Y.lo > X.lo)))) {
		struct u128 t = X;
		int ti = ex;
		X = Y, Y = t, ex = ey, ey = ti;
		ti = sx, sx = sy, sy = ti;
	}
	Y = shr128(Y, ex - ey, &rest);
	if (rest)
		Y.lo |= 1;
	if (sx == sy) {
		R.lo = X.lo + Y.lo;
		R.hi = X.hi + Y.hi + (R.lo < X.lo);
	} else {
		R.lo = X.lo - Y.lo;
		R.hi = X.hi - Y.hi - (X.lo < Y.lo);
	}
	if (!R.hi && !R.lo)
		return 0.0;	/* exact cancellation: +0 in round to nearest */
	L = lead128(R);
	/* the kept bits: 53, or down to the 2^-1074 grid */
	s = L - 52;
	if (ex + L < -1022)
		s = -1074 - ex;
	if (s <= 0)
		return (sx ? -1.0 : 1.0) * ldexp((double)R.lo, ex);
	if (s > L + 1)		/* below half the smallest subnormal */
		return sx ? -0.0 : 0.0;
	{
		struct u128 q = shr128(R, s, &sticky), back = shl128(q, s);
		/* r = R - back, compared with 2^(s-1) */
		struct u128 rr, half;
		rr.lo = R.lo - back.lo;
		rr.hi = R.hi - back.hi - (R.lo < back.lo);
		half.hi = s - 1 >= 64 ? 1ULL << (s - 1 - 64) : 0;
		half.lo = s - 1 < 64 ? 1ULL << (s - 1) : 0;
		if (rr.hi > half.hi || (rr.hi == half.hi && rr.lo > half.lo) ||
		    (rr.hi == half.hi && rr.lo == half.lo && (q.lo & 1)))
			q.lo++;
		r = ldexp((double)q.lo, ex + s);
	}
	return sx ? -r : r;
}

/*
 * fma32(a, b, c): the f32 fused multiply-add, correctly rounded (round to
 * nearest even), for f32 operands a, b, c held in doubles.
 *
 * Derivation.  Let D be the binary64 grid, F the binary32 grid
 * (including its overflow threshold 2^128 (1 - 2^-25), where rounding
 * changes to inf), and RN_F rounding to nearest even on F.
 *
 * 1. p = a * b is exact in double: a and b have 24-bit significands, so
 *    the product has at most 48 <= 53 significant bits, and |p| lies in
 *    [2^-298, 2^256] or is 0, inside binary64's normal range (no
 *    overflow, no underflow).
 *
 * 2. s = p + c rounded to D, and e = (p + c) - s by Knuth's TwoSum (six
 *    operations, exact in binary64 round to nearest for any finite
 *    operands without overflow; there is none, |p + c| < 2^257).  So
 *    the exact sum is x = s + e.
 *
 * 3. Round to odd: y = s if e = 0 or the last significand bit of s is 1;
 *    otherwise y is the neighbour of s in D on the side of x (bits of s
 *    plus or minus 1, by the signs of s and e; s != 0 here, since a sum
 *    that rounds to 0 is exact).  Then y is the point of D nearest x
 *    whose last bit is odd among the two D-points d1 < x < d2 enclosing
 *    x when x is not in D, and y = x when x is in D.
 *
 * 4. Every point of F and every midpoint between neighbouring points of
 *    F (the decision boundaries of RN_F, and the overflow threshold) is
 *    an even point of D: in a binade [2^k, 2^(k+1)) with k >= -126,
 *    F-midpoints are multiples of 2^(k-24), and D's spacing there is
 *    2^(k-52), so they are multiples of 2^(k-51) = 2 ulp_D; below
 *    2^-126 they are multiples of 2^-150, and there ulp_D <= 2^-179, a
 *    power of two, so 2^-150 is a multiple of 2 ulp_D.  (An "even" point
 *    of D is a multiple of 2 ulp_D in its binade: last bit 0.)
 *
 * 5. RN_F is constant on every interval between consecutive boundaries
 *    (4).  If x is in D, y = x.  Otherwise x is strictly inside (d1, d2),
 *    which holds no point of D and so no boundary; y is d1 or d2 and is
 *    odd, so it is not a boundary either.  Hence the closed segment
 *    between x and y holds no boundary, and RN_F(y) = RN_F(x): rounding
 *    y to float (one rounding, the cast) is the correctly rounded fma.
 *    (Boldo and Melquiond's round-to-odd theorem, here with 29 spare
 *    bits where 2 suffice.)
 *
 * Signed zeros: x = 0 gives e = 0 and s = p + c with IEEE's sign rule,
 * as fma's.  Infinite or NaN operands: no rounding question, s is the
 * IEEE result (inf * 0 and inf - inf give NaN) and step 3 is skipped.
 *
 * fma16(a, b, c), the binary16 one, is the same derivation with F the
 * binary16 grid and fround's rounding to it in place of the cast:
 * 1. the product has at most 22 significant bits and |p| lies in
 *    [2^-48, 2^32) or is 0; 2. |p + c| < 2^33, and a nonzero sum is a
 *    multiple of 2^-48, far above binary64's subnormals; 4. in a binade
 *    [2^k, 2^(k+1)) with k >= -14, F-midpoints (and the overflow threshold
 *    2^16 (1 - 2^-12)) are multiples of 2^(k-11) = 2^41 ulp_D; below 2^-14
 *    they are multiples of 2^-25, and there ulp_D <= 2^-67.  42 spare
 *    bits where 2 suffice.
 */
static double
fma_narrow(int w, double a, double b, double c)
{
	double p = a * b, s = p + c, bb, e;
	uint64_t u;

	if (!isfinite(s))
		return fround(w, s);
	bb = s - p;
	e = (p - (s - bb)) + (c - bb);
	if (e != 0) {
		memcpy(&u, &s, sizeof u);
		if (!(u & 1)) {
			/* away from zero when e has s's sign */
			if ((e > 0) == (s > 0))
				u++;
			else
				u--;
			memcpy(&s, &u, sizeof s);
		}
	}
	return fround(w, s);
}

double
fma32(double a, double b, double c)
{
	return fma_narrow(32, a, b, c);
}

double
fma16(double a, double b, double c)
{
	return fma_narrow(16, a, b, c);
}

/*
 * IEEE semantics: computed in double, then rounded to the width (for f32
 * and f16 this is the correctly rounded result for + - * / sqrt: binary64's
 * 53 bits are at least twice the width's 24 or 11, plus 2, so rounding the
 * rounded double is rounding the exact value).  NaN and inf are
 * ordinary values; the flags only record where they first appear.
 */
/*
 * float_ftz (equiv's `ftz`): every float operation but a bit cast reads a
 * subnormal operand as the zero of its sign and gives one for a
 * subnormal result.
 */
int float_ftz;

double
fflush0(int w, double v)
{
	double mn = w == 16 ? 6.103515625e-05 : w == 32 ? 1.1754943508222875e-38 :
	    2.2250738585072014e-308;

	return v != 0 && fabs(v) < mn ? copysign(0.0, v) : v;
}

double
scalar_float(int op, int w, double a, double b, uint8_t *fl)
{
	double r;

	if (float_ftz) {
		a = fflush0(w, a);
		b = fflush0(w, b);
	}
	switch (op) {
	case O_ADD: r = a + b; break;
	case O_SUB: r = a - b; break;
	case O_MUL: r = a * b; break;
	case O_DIV: r = a / b; break;
	case O_MOD: r = fmod(a, b); break;
	case O_POW: r = pow(a, b); break;
	case O_NEG: r = -a; break;
	case O_ABS: r = fabs(a); break;
	case O_SIGN: r = isnan(a) ? a : (a > 0) - (a < 0); break;
	case O_SQRT: r = sqrt(a); break;
	case O_EXP: r = exp(a); break;
	case O_LOG: r = log(a); break;
	case O_SIN: r = sin(a); break;
	case O_COS: r = cos(a); break;
	case O_TAN: r = tan(a); break;
	case O_LAMBERTW: r = lambert_w(a, 0); break;
	case O_LAMBERTWM1: r = lambert_w(a, 1); break;
	case O_SEC: r = 1 / cos(a); break;
	case O_CSC: r = 1 / sin(a); break;
	case O_COT: r = cos(a) / sin(a); break;
	case O_ATAN: r = atan(a); break;
	case O_ASIN: r = asin(a); break;
	case O_ACOS: r = acos(a); break;
	case O_ATAN2: r = atan2(a, b); break;
	case O_SINH: r = sinh(a); break;
	case O_COSH: r = cosh(a); break;
	case O_TANH: r = tanh(a); break;
	case O_ASINH: r = asinh(a); break;
	case O_ACOSH: r = acosh(a); break;
	case O_ATANH: r = atanh(a); break;
	case O_CONJ: case O_RE: r = a; break;
	case O_IM: r = 0; break;
	case O_ARG: r = a > 0 ? 0 : a < 0 ? M_PI : NAN; break;
	case O_COPYSIGN: r = copysign(a, b); break;
	case O_FLOOR: r = floor(a); break;
	case O_CEIL: r = ceil(a); break;
	case O_ROUNDEVEN: r = nearbyint(a); break;
	case O_MIN: r = fmin(a, b); break;
	case O_MAX: r = fmax(a, b); break;
	case O_EQ: return a == b;
	case O_NE: return a != b;
	case O_LT: return a < b;
	case O_LE: return a <= b;
	case O_GT: return a > b;
	case O_GE: return a >= b;
	default:
		*fl |= F_UNDEF;
		return 0;
	}
	r = fround(w, r);
	if (float_ftz)
		r = fflush0(w, r);
	if (isnan(r) && !isnan(a) && !isnan(b))
		*fl |= F_NAN;
	if (isinf(r) && !isinf(a) && !isinf(b))
		*fl |= F_INF;
	return r;
}

static double
int_to_real(const struct ity *t, int64_t v)
{
	if (t->k == K_UINT || t->k == K_BOOL)
		return (double)asu(t, v);
	return (double)ass(t, v);
}

/*
 * v as a float of w bits, rounded once.  Through a double, a 64-bit
 * integer would be rounded twice (2^60 + 2^36 + 1 to the midpoint 2^60 +
 * 2^36, then to the even 2^60); C's own conversion to float rounds once.
 * Into f16 every integer past 2^53 is past 65520 too, so its double
 * rounding is harmless.
 */
static double
int_to_float(const struct ity *t, int64_t v, int w)
{
	if (w == 32) {
		if (t->k == K_UINT || t->k == K_BOOL)
			return (float)(uint64_t)asu(t, v);
		return (float)ass(t, v);
	}
	return fround(w, int_to_real(t, v));
}

double
ifloat(ty_t ti, int64_t v, int w)
{
	struct ity it;

	ity_of(&it, ti, 0);
	return int_to_float(&it, v, w);
}

static int64_t
real_to_int(const struct ity *t, double x, uint8_t *fl)
{
	int64_t v;

	if (isnan(x) || x >= 9.2233720368547758e18 || x < -9.2233720368547758e18) {
		*fl |= F_UNDEF | F_UB;
		return 0;
	}
	v = (int64_t)x;
	if (t->k != K_INT && inorm(t, (uint64_t)v) != v)
		*fl |= F_TRUNC | F_UB;
	return inorm(t, (uint64_t)v);
}

/* used by the node constructors to fold constant operands */
int
fold_const(int op, ty_t t, nid a, nid b, nid c, int64_t *ri, double *rr)
{
	struct ity it, ita;
	uint8_t fl = 0;
	ty_t ta = N.ty[a];

	if (op == O_SEL || op == O_TABLE || op == O_INDEX || op == O_UNDEF)
		return 0;
	if (op == O_BARRIER || op == O_ATOMIC)
		return 0;
	if (IS_DOT(op)) {
		*ri = dot_scalar(op, 0, N.ki[a], N.ki[b], c ? N.ki[c] : 0, &fl);
		return !(fl & F_UNDEF);
	}
	if (IS_TEX(op))		/* mk_tex folds its own */
		return 0;
	if (op == O_FMA) {
		if (TYK(t) == K_FLOAT && TYW(t) == 32)
			*rr = fma32(N.kr[a], N.kr[b], N.kr[c]);
		else if (TYK(t) == K_FLOAT && TYW(t) == 16)
			*rr = fma16(N.kr[a], N.kr[b], N.kr[c]);
		else
			*rr = fma64(N.kr[a], N.kr[b], N.kr[c]);
		return isfinite(*rr);
	}
	ity_of(&it, t, 0);
	ity_of(&ita, ta, 0);
	if (TY_FP(t)) {
		if (op == O_ASFLOAT)
			*rr = frombits(TYW(t), (uint64_t)N.ki[a]);
		else if (op == O_CAST)
			*rr = TY_FP(ta) ? N.kr[a] : TYK(t) == K_FLOAT ?
			    int_to_float(&ita, N.ki[a], TYW(t)) :
			    int_to_real(&ita, N.ki[a]);
		else if (TYK(t) == K_FLOAT)
			*rr = scalar_float(op, TYW(t), N.kr[a], b ? N.kr[b] : 0, &fl);
		else
			*rr = scalar_real(op, N.kr[a], b ? N.kr[b] : 0, &fl);
		if (TYK(t) == K_FLOAT)
			*rr = fround(TYW(t), *rr);
	} else if (TY_FP(ta)) {
		if (op == O_ASUINT || op == O_ASINT)
			*ri = int_norm(t, (int64_t)fbits(TYW(ta), N.kr[a]));
		else if (op == O_CAST)
			*ri = real_to_int(&it, N.kr[a], &fl);
		else
			*ri = scalar_real(op, N.kr[a], b ? N.kr[b] : 0, &fl) != 0;
	} else {
		*ri = scalar_int(op, &it, &ita, N.ki[a], b ? N.ki[b] : 0, 0, &fl);
	}
	return !(fl & (F_UNDEF | F_UB | F_OVF | F_NAN | F_INF));
}

/* ---- plans ---- */

void
plan_build(struct plan *p, const nid *roots0, uint32_t nr, int ideal)
{
	uint8_t *mark;
	uint32_t *map;
	nid *stack, g, *roots = xmalloc((nr + 1) * sizeof *roots);
	uint32_t sp = 0, i, k, nl = 0, j;

	/* subgroup operations spelled out over their lanes (sg.c): new
	 * nodes, so before anything is sized by N.n */
	for (i = 0; i < nr; i++) {
		if (roots0[i] && (N.fl[roots0[i]] & NF_LANES)) {
			const char *nm = sg_lanes_name(roots0[i]);
			free(roots);
			fail("%s has its own value in each lane (a lanes "
			    "parameter) and is read across the lanes: there is no "
			    "one point to evaluate; equiv and find decide by SAT",
			    nm ? nm : "a parameter");
		}
		roots[i] = sg_lower(roots0[i]);
		if (roots[i] && sg_has_lane(roots[i]))
			fail("lane is a value per lane: outside a subgroup "
			    "operation it has to be a parameter, f(..., lane: "
			    "u32 in 0..N-1)");
	}
	mark = xcalloc(N.n, 1);
	map = xcalloc(N.n, sizeof *map);
	stack = xmalloc(N.n * sizeof *stack);
	memset(p, 0, sizeof *p);
	for (i = 0; i < nr; i++)
		if (roots[i] && !mark[roots[i]])
			mark[roots[i]] = 1, stack[sp++] = roots[i];
	while (sp) {
		g = stack[--sp];
#define VISIT(x) do { nid x_ = (x); \
		if (x_ && !mark[x_]) { mark[x_] = 1; stack[sp++] = x_; } } while (0)
		switch (N.op[g]) {
		case O_CONST:
		case O_VAR:
		case O_UNDEF:
			break;
		case O_TABLE:
			for (j = 0; j < N.b[g]; j++)
				VISIT(N.lst[N.a[g] + j]);
			nl += N.b[g];
			break;
		default:
			VISIT(N.a[g]);
			VISIT(N.b[g]);
			VISIT(N.c[g]);
		}
#undef VISIT
	}
	for (g = 1; g < N.n; g++)
		if (mark[g])
			p->n++;
	k = p->n;
	p->ideal = ideal;
	p->op = xmalloc(k);
	p->ty = xmalloc(k * sizeof *p->ty);
	p->it = xmalloc(k * sizeof *p->it);
	p->a = xcalloc(k, sizeof *p->a);
	p->b = xcalloc(k, sizeof *p->b);
	p->c = xcalloc(k, sizeof *p->c);
	p->ki = xcalloc(k, sizeof *p->ki);
	p->kr = xcalloc(k, sizeof *p->kr);
	p->gid = xmalloc(k * sizeof *p->gid);
	p->lst = xmalloc((nl + 1) * sizeof *p->lst);
	p->vi = xcalloc((size_t)k * CH, sizeof *p->vi);
	p->vr = xcalloc((size_t)k * CH, sizeof *p->vr);
	p->vf = xcalloc((size_t)k * CH, 1);
	nl = 0;
	for (g = 1, k = 0; g < N.n; g++) {
		if (!mark[g])
			continue;
		map[g] = k;
		p->gid[k] = g;
		p->op[k] = N.op[g];
		p->ty[k] = N.ty[g];
		ity_of(&p->it[k], N.ty[g], ideal);
		p->ki[k] = N.ki[g];
		p->kr[k] = N.kr[g];
		switch (N.op[g]) {
		case O_CONST:
			for (i = 0; i < CH; i++) {
				COLI(p, k)[i] = N.ki[g];
				COLR(p, k)[i] = N.kr[g];
			}
			if (ideal && TYK(N.ty[g]) == K_UINT)
				for (i = 0; i < CH; i++)
					COLI(p, k)[i] = N.ki[g];
			break;
		case O_VAR:
		case O_UNDEF:
			break;
		case O_TABLE:
			p->a[k] = nl;
			p->b[k] = N.b[g];
			for (j = 0; j < N.b[g]; j++)
				p->lst[nl++] = map[N.lst[N.a[g] + j]];
			break;
		default:
			p->a[k] = N.a[g] ? map[N.a[g]] : 0;
			p->b[k] = N.b[g] ? map[N.b[g]] : 0;
			p->c[k] = N.c[g] ? map[N.c[g]] : 0;
		}
		k++;
	}
	p->nroot = nr;
	p->root = xmalloc((nr + 1) * sizeof *p->root);
	for (i = 0; i < nr; i++)
		p->root[i] = roots[i] ? map[roots[i]] : NOTOK;
	free(mark);
	free(map);
	free(stack);
	free(roots);
}

void
plan_free(struct plan *p)
{
	free(p->op);
	free(p->ty);
	free(p->it);
	free(p->a);
	free(p->b);
	free(p->c);
	free(p->ki);
	free(p->kr);
	free(p->gid);
	free(p->lst);
	free(p->root);
	free(p->vi);
	free(p->vr);
	free(p->vf);
	memset(p, 0, sizeof *p);
}

uint32_t
plan_loc(const struct plan *p, nid g)
{
	uint32_t lo = 0, hi = p->n;

	g = sg_lower(g);	/* as plan_build took it */
	if (sg_has_lane(g))	/* part of a value per lane: no column */
		return NOTOK;
	while (lo < hi) {
		uint32_t m = (lo + hi) / 2;
		if (p->gid[m] < g)
			lo = m + 1;
		else
			hi = m;
	}
	return lo < p->n && p->gid[lo] == g ? lo : NOTOK;
}

/* texel j of a table in the plan, at point i (tex_filter's callback) */
struct texcol {
	const struct plan *p;
	uint32_t st, i;
	uint8_t fl;
};

static double
tex_col_texel(void *ctx, int64_t j)
{
	struct texcol *t = ctx;
	uint32_t e = t->p->lst[t->st + (uint32_t)j];

	t->fl |= COLF(t->p, e)[t->i];
	return COLR(t->p, e)[t->i];
}

void
plan_run(struct plan *p, uint32_t np, const int64_t *in_i, const double *in_r)
{
	uint32_t k, i, n = p->n;

	for (k = 0; k < n; k++) {
		int op = p->op[k];
		int real = TY_FP(p->ty[k]);
		const struct ity *t = &p->it[k];
		int64_t *o = COLI(p, k);
		double *orr = COLR(p, k);
		uint8_t *of = COLF(p, k);
		uint32_t a = p->a[k], b = p->b[k], c = p->c[k];

		switch (op) {
		case O_CONST:
			memset(of, 0, np);
			continue;
		case O_TABLE:
			continue;
		case O_TEXARGS:
			/* no value: its read takes the two operands apart */
			for (i = 0; i < np; i++) {
				o[i] = 0;
				orr[i] = 0;
				of[i] = (COLF(p, a)[i] | COLF(p, b)[i]) &
				    F_UNDEF;
			}
			continue;
		case O_UNDEF:
			memset(of, F_UNDEF, np);
			continue;
		case O_LOOP:
			loop_run(p, k, np, in_i, in_r);
			continue;
		case O_BARRIER:
		case O_ATOMIC:
			for (i = 0; i < np; i++) {
				o[i] = COLI(p, a)[i];
				orr[i] = COLR(p, a)[i];
				of[i] = (uint8_t)((COLF(p, a)[i] | (op == O_ATOMIC ?
				    COLF(p, b)[i] : 0)) & F_UNDEF);
			}
			continue;
		case O_VAR:
			if (real) {
				memcpy(orr, in_r + (size_t)p->ki[k] * CH,
				    np * sizeof *orr);
				if (t->k == K_FLOAT && t->w != 64)
					for (i = 0; i < np; i++)
						orr[i] = fround(t->w, orr[i]);
			} else
				for (i = 0; i < np; i++)
					o[i] = inorm(t, (uint64_t)
					    in_i[(size_t)p->ki[k] * CH + i]);
			memset(of, 0, np);
			continue;
		case O_SEL: {
			const int64_t *cv = COLI(p, a);
			const uint8_t *cf = COLF(p, a);
			for (i = 0; i < np; i++) {
				uint32_t s = cv[i] ? b : c;
				o[i] = COLI(p, s)[i];
				orr[i] = COLR(p, s)[i];
				of[i] = (uint8_t)((cf[i] | COLF(p, s)[i]) & F_UNDEF);
			}
			continue;
		}
		case O_LAND:
		case O_LOR: {
			const int64_t *av = COLI(p, a), *bv = COLI(p, b);
			const uint8_t *af = COLF(p, a), *bf = COLF(p, b);
			int stop = op == O_LOR;
			for (i = 0; i < np; i++) {
				if (af[i] & F_UNDEF) {
					o[i] = 0;
					of[i] = F_UNDEF;
				} else if ((av[i] != 0) == stop) {
					o[i] = stop;
					of[i] = 0;
				} else {
					o[i] = bv[i] != 0;
					of[i] = bf[i] & F_UNDEF;
				}
			}
			continue;
		}
		case O_FMA: {
			/*
			 * fma64, fma32 and fma16, correctly rounded (the
			 * derivations are at fma64 and fma32).
			 */
			const double *av = COLR(p, a), *bv = COLR(p, b);
			const double *cv = COLR(p, c);
			int fw = t->k == K_FLOAT ? t->w : 0;
			for (i = 0; i < np; i++) {
				uint8_t f = 0;
				if ((COLF(p, a)[i] | COLF(p, b)[i] | COLF(p, c)[i]) &
				    F_UNDEF) {
					of[i] = F_UNDEF;
					orr[i] = 0;
					continue;
				}
				if (fw && float_ftz) {
					double x = fflush0(fw, av[i]), y = fflush0(fw, bv[i]);
					double z = fflush0(fw, cv[i]);
					orr[i] = fflush0(fw, fw == 32 ? fma32(x, y, z) :
					    fw == 16 ? fma16(x, y, z) : fma64(x, y, z));
				} else
					orr[i] = fw == 32 ? fma32(av[i], bv[i], cv[i]) :
					    fw == 16 ? fma16(av[i], bv[i], cv[i]) :
					    fma64(av[i], bv[i], cv[i]);
				if (fw && isnan(orr[i]) && !isnan(av[i]) &&
				    !isnan(bv[i]) && !isnan(cv[i]))
					f |= F_NAN;
				if (fw && isinf(orr[i]) && !isinf(av[i]) &&
				    !isinf(bv[i]) && !isinf(cv[i]))
					f |= F_INF;
				if (!fw && !isfinite(orr[i])) {
					f |= F_UNDEF;
					orr[i] = 0;
				}
				of[i] = f;
			}
			continue;
		}
		case O_UDOT4X8: case O_SDOT4X8: case O_SUDOT4X8:
		case O_UDOT2X16: case O_SDOT2X16: case O_SUDOT2X16:
		case O_UDOT8X4: case O_SDOT8X4: case O_SUDOT8X4:
		case O_UDOT4X8S: case O_SDOT4X8S: case O_SUDOT4X8S:
		case O_UDOT2X16S: case O_SDOT2X16S: case O_SUDOT2X16S:
		case O_UDOT8X4S: case O_SDOT8X4S: case O_SUDOT8X4S: {
			const int64_t *av = COLI(p, a), *bv = COLI(p, b);
			int sat = opinfo[op].arity == 3;
			for (i = 0; i < np; i++) {
				uint8_t f = (COLF(p, a)[i] | COLF(p, b)[i] |
				    (sat ? COLF(p, c)[i] : 0)) & F_UNDEF;
				if (f) {
					o[i] = 0;
					of[i] = F_UNDEF;
					continue;
				}
				o[i] = dot_scalar(op, p->ideal, av[i], bv[i],
				    sat ? COLI(p, c)[i] : 0, &f);
				of[i] = f;
			}
			continue;
		}
		case O_FETCH_REPEAT: case O_FETCH_MIRROR: case O_FETCH_CLAMP:
		case O_FETCH_MCLAMP: case O_FETCH_BORDER: {
			/* a nearest read: the address mode makes it total */
			const int64_t *iv = COLI(p, b);
			const uint8_t *ifl = COLF(p, b);
			uint32_t st = p->a[a], cnt = p->b[a];
			for (i = 0; i < np; i++) {
				int64_t j = iv[i];
				uint32_t e;
				if (TYK(p->ty[b]) == K_UINT)
					j = (int64_t)asu(&p->it[b], j);
				if (ifl[i] & F_UNDEF) {
					o[i] = 0;
					orr[i] = 0;
					of[i] = F_UNDEF;
					continue;
				}
				j = tex_wrap(op, j + TEX_OFF(p->ki[k]), cnt);
				e = j < 0 ? c : p->lst[st + (uint32_t)j];
				o[i] = COLI(p, e)[i];
				orr[i] = COLR(p, e)[i];
				of[i] = COLF(p, e)[i] & F_UNDEF;
			}
			continue;
		}
		case O_FETCH_TEXEL: case O_FETCH_BUFFER: {
			/* texelFetch: the texel, undefined outside the table */
			const int64_t *iv = COLI(p, b);
			uint32_t st = p->a[a], cnt = p->b[a];
			for (i = 0; i < np; i++) {
				int64_t j = iv[i];
				uint32_t e;
				if (TYK(p->ty[b]) == K_UINT)
					j = (int64_t)asu(&p->it[b], j);
				if ((COLF(p, b)[i] & F_UNDEF) || j < 0 || j >=
				    (int64_t)cnt) {
					o[i] = 0;
					orr[i] = 0;
					of[i] = F_UNDEF;
					continue;
				}
				e = p->lst[st + (uint32_t)j];
				o[i] = COLI(p, e)[i];
				orr[i] = COLR(p, e)[i];
				of[i] = COLF(p, e)[i] & F_UNDEF;
			}
			continue;
		}
		case O_FETCH_LAYER: {
			/* a layer of an array texture: the coordinate rounded
			 * to nearest even and clamped; NaN selects none */
			uint32_t st = p->a[a], cnt = p->b[a];
			for (i = 0; i < np; i++) {
				int64_t j = tex_layer(COLR(p, b)[i], cnt);
				uint32_t e;
				if ((COLF(p, b)[i] & F_UNDEF) || j < 0) {
					o[i] = 0;
					orr[i] = 0;
					of[i] = F_UNDEF;
					continue;
				}
				e = p->lst[st + (uint32_t)j];
				o[i] = COLI(p, e)[i];
				orr[i] = COLR(p, e)[i];
				of[i] = COLF(p, e)[i] & F_UNDEF;
			}
			continue;
		}
		case O_TEX_LINEAR: case O_TEX_BILINEAR: case O_TEX_TRILINEAR:
		case O_TEX_COMPARE: {
			struct texcol tc;
			uint32_t cnt = p->b[a], y = c, z = c;
			int hz = TEX_HAS_ARGS(op);
			tc.p = p;
			tc.st = p->a[a];
			if (hz) {
				/* (y, z) from the pair (plan index 0 is a node,
				 * not "none": hz says whether there is a z) */
				y = p->a[c];
				z = p->b[c];
			}
			for (i = 0; i < np; i++) {
				uint8_t f = COLF(p, b)[i] & F_UNDEF;
				if (op != O_TEX_LINEAR)
					f |= COLF(p, y)[i] & F_UNDEF;
				if (hz)
					f |= COLF(p, z)[i] & F_UNDEF;
				tc.i = i;
				tc.fl = 0;
				o[i] = 0;
				orr[i] = f ? 0 : tex_filter(op, p->ki[k], cnt,
				    COLR(p, b)[i], op != O_TEX_LINEAR ?
				    COLR(p, y)[i] : 0, hz ? COLR(p, z)[i] : 0,
				    tex_col_texel, &tc);
				of[i] = (f | tc.fl) & F_UNDEF;
			}
			continue;
		}
		case O_INDEX: {
			const int64_t *iv = COLI(p, b);
			const uint8_t *ifl = COLF(p, b);
			uint32_t st = p->a[a], cnt = p->b[a];
			for (i = 0; i < np; i++) {
				int64_t j = iv[i];
				uint32_t e;
				if (TYK(p->ty[b]) == K_UINT)
					j = (int64_t)asu(&p->it[b], j);
				if ((ifl[i] & F_UNDEF) || j < 0 || (uint64_t)j >= cnt) {
					o[i] = 0;
					orr[i] = 0;
					of[i] = F_UNDEF;
					continue;
				}
				e = p->lst[st + (uint32_t)j];
				o[i] = COLI(p, e)[i];
				orr[i] = COLR(p, e)[i];
				of[i] = COLF(p, e)[i] & F_UNDEF;
			}
			continue;
		}
		}

		/* plain operations: one loop per column */
		{
			const uint8_t *af = COLF(p, a);
			const uint8_t *bf = opinfo[op].arity > 1 ? COLF(p, b) : af;
			int areal = TY_FP(p->ty[a]);
			const struct ity *ta = &p->it[a];
			int fw = t->k == K_FLOAT ? t->w : 0;

			if (real && (op == O_CAST || op == O_ASFLOAT) && !areal) {
				const int64_t *av = COLI(p, a);
				for (i = 0; i < np; i++) {
					if (op == O_ASFLOAT)
						orr[i] = frombits(TYW(p->ty[k]),
						    asu(ta, av[i]));
					else
						orr[i] = fw ? int_to_float(ta, av[i], fw) :
						    int_to_real(ta, av[i]);
					of[i] = af[i] & F_UNDEF;
				}
			} else if (real && op == O_CAST) {
				const double *av = COLR(p, a);
				for (i = 0; i < np; i++) {
					orr[i] = fw ? fround(fw, av[i]) : av[i];
					if (fw && float_ftz)
						orr[i] = fflush0(fw, fround(fw,
						    TYK(p->ty[a]) == K_FLOAT ?
						    fflush0(TYW(p->ty[a]), av[i]) : av[i]));
					of[i] = af[i] & F_UNDEF;
				}
			} else if (real && fw) {
				const double *av = COLR(p, a), *bv = COLR(p, b);
				for (i = 0; i < np; i++) {
					uint8_t f = 0;
					if ((af[i] | bf[i]) & F_UNDEF) {
						of[i] = F_UNDEF;
						orr[i] = 0;
						continue;
					}
					orr[i] = scalar_float(op, fw, av[i], bv[i], &f);
					of[i] = f;
				}
			} else if (real) {
				const double *av = COLR(p, a), *bv = COLR(p, b);
				for (i = 0; i < np; i++) {
					uint8_t f = 0;
					if ((af[i] | bf[i]) & F_UNDEF) {
						of[i] = F_UNDEF;
						orr[i] = 0;
						continue;
					}
					orr[i] = scalar_real(op, av[i], bv[i], &f);
					of[i] = f;
				}
			} else if (areal) {
				const double *av = COLR(p, a), *bv = COLR(p, b);
				for (i = 0; i < np; i++) {
					uint8_t f = 0;
					if ((af[i] | bf[i]) & F_UNDEF) {
						of[i] = F_UNDEF;
						o[i] = 0;
						continue;
					}
					if (op == O_ASUINT || op == O_ASINT)
						o[i] = inorm(t, fbits(TYW(p->ty[a]), av[i]));
					else if (op == O_CAST)
						o[i] = real_to_int(t, av[i], &f);
					else if (float_ftz && TYK(p->ty[a]) == K_FLOAT)
						o[i] = scalar_real(op, fflush0(TYW(p->ty[a]),
						    av[i]), fflush0(TYW(p->ty[a]), bv[i]), &f) != 0;
					else
						o[i] = scalar_real(op, av[i], bv[i], &f) != 0;
					of[i] = f;
				}
			} else {
				const int64_t *av = COLI(p, a), *bv = COLI(p, b);
				for (i = 0; i < np; i++) {
					uint8_t f = 0;
					if ((af[i] | bf[i]) & F_UNDEF) {
						of[i] = F_UNDEF;
						o[i] = 0;
						continue;
					}
					o[i] = scalar_int(op, t, ta, av[i], bv[i], 0, &f);
					of[i] = f;
				}
			}
		}
		(void)c;
	}
}

/*
 * Which nodes are actually evaluated at each point, C-style: only the
 * chosen arm of ?:, the right side of && / || when it is reached, the
 * selected table entry.  Flags from nodes that are not live are ignored.
 */
void
plan_live(const struct plan *p, uint32_t np, uint8_t *lv)
{
	uint32_t k, i;

	memset(lv, 0, (size_t)p->n * CH);
	for (i = 0; i < p->nroot; i++)
		if (p->root[i] != NOTOK)
			memset(lv + (size_t)p->root[i] * CH, 1, np);
	for (k = p->n; k-- > 0;) {
		const uint8_t *l = lv + (size_t)k * CH;
		uint32_t a = p->a[k], b = p->b[k], c = p->c[k];
		uint8_t *la = lv + (size_t)a * CH, *lb = lv + (size_t)b * CH;
		uint8_t *lc = lv + (size_t)c * CH;
		const int64_t *av = COLI(p, a);
		const uint8_t *af = COLF(p, a);

		switch (p->op[k]) {
		case O_CONST:
		case O_VAR:
		case O_UNDEF:
		case O_TABLE:
			break;
		case O_SEL:
			for (i = 0; i < np; i++) {
				if (!l[i])
					continue;
				la[i] = 1;
				if (af[i] & F_UNDEF)
					continue;
				if (av[i])
					lb[i] = 1;
				else
					lc[i] = 1;
			}
			break;
		case O_LAND:
		case O_LOR:
			for (i = 0; i < np; i++) {
				if (!l[i])
					continue;
				la[i] = 1;
				if (!(af[i] & F_UNDEF) &&
				    (av[i] != 0) == (p->op[k] == O_LAND))
					lb[i] = 1;
			}
			break;
		case O_FETCH_REPEAT: case O_FETCH_MIRROR: case O_FETCH_CLAMP:
		case O_FETCH_MCLAMP: case O_FETCH_BORDER: case O_TEX_LINEAR:
		case O_TEX_BILINEAR: case O_FETCH_LAYER: case O_FETCH_TEXEL:
		case O_FETCH_BUFFER: case O_TEX_TRILINEAR: case O_TEX_COMPARE: {
			/* the coordinates (a pair's: its own case), and any
			 * texel */
			uint32_t st = p->a[a], cnt = p->b[a], j;
			for (i = 0; i < np; i++) {
				if (!l[i])
					continue;
				lb[i] = 1;
				if (p->op[k] == O_FETCH_BORDER ||
				    p->op[k] == O_TEX_BILINEAR ||
				    TEX_HAS_ARGS(p->op[k]))
					lc[i] = 1;
				for (j = 0; j < cnt; j++)
					lv[(size_t)p->lst[st + j] * CH + i] = 1;
			}
			break;
		}
		case O_INDEX: {
			const int64_t *iv = COLI(p, b);
			uint32_t st = p->a[a], cnt = p->b[a];
			for (i = 0; i < np; i++) {
				int64_t j = iv[i];
				if (!l[i])
					continue;
				lb[i] = 1;
				if (TYK(p->ty[b]) == K_UINT)
					j = (int64_t)asu(&p->it[b], j);
				if (j >= 0 && (uint64_t)j < cnt)
					lv[(size_t)p->lst[st + (uint32_t)j] * CH + i] = 1;
			}
			break;
		}
		default:
			for (i = 0; i < np; i++)
				if (l[i]) {
					la[i] = 1;
					if (opinfo[p->op[k]].arity > 1)
						lb[i] = 1;
					if (opinfo[p->op[k]].arity > 2)
						lc[i] = 1;
				}
		}
	}
}

/* ---- domains ---- */

void
param_bounds(const struct param *p, int64_t *lo, int64_t *hi)
{
	struct ity it;

	ity_of(&it, p->ty, 0);
	switch (it.k) {
	case K_UINT:
	case K_BOOL:
		*lo = 0;
		*hi = it.w >= 63 ? INT64_MAX : (int64_t)it.mask;
		break;
	case K_SINT:
	case K_INT:
		*lo = it.smin;
		*hi = it.smax;
		break;
	default:
		*lo = *hi = 0;
	}
	if (p->ranged && TYK(p->ty) != K_REAL) {
		if (p->lo > *lo)
			*lo = p->lo;
		if (p->hi < *hi)
			*hi = p->hi;
	}
}

/*
 * The where clause's conjuncts that compare a parameter (through casts
 * that keep its value) with an expression bound the parameter by the
 * expression's interval: `v < 1 << n` with n in 5..8 gives v <= 255.
 * The intervals are sound for machine values (plan_ival), so every point
 * cut off fails the where clause; a few rounds reach a fixpoint.
 */
static int
cast_keeps(ty_t s, ty_t t)
{
	int ks = TYK(s), kt = TYK(t), ws = TYW(s), wt = TYW(t);

	if (ks == K_UINT || ks == K_BOOL) {
		if (ks == K_BOOL)
			ws = 1;
		return (kt == K_UINT && wt >= ws) || (kt == K_SINT && wt > ws) ||
		    (kt == K_INT && ws < 64);
	}
	if (ks == K_SINT)
		return (kt == K_SINT && wt >= ws) || kt == K_INT;
	if (ks == K_INT)
		return kt == K_INT;
	return 0;
}

static int
var_under_casts(nid e)
{
	while (N.op[e] == O_CAST && cast_keeps(N.ty[N.a[e]], N.ty[e]))
		e = N.a[e];
	return N.op[e] == O_VAR && N.ki[e] < MAXP ? (int)N.ki[e] : -1;
}

static int
conjuncts(nid w, nid *out, int n, int max)
{
	if (N.op[w] == O_LAND) {
		n = conjuncts(N.a[w], out, n, max);
		return conjuncts(N.b[w], out, n, max);
	}
	if (n < max && (opinfo[N.op[w]].fl & OF_CMP) &&
	    TY_ISINT(N.ty[N.a[w]]) && N.ty[N.a[w]] != TY_BOOL)
		out[n++] = w;
	return n;
}

int
where_bounds(const struct func *f, int64_t *lo, int64_t *hi,
    uint8_t *narrowed)
{
	nid cj[64], roots[128];
	int nc, i, j, round, any = 0;
	struct func g;
	struct plan p;
	struct ival *iv;

	if (!f->where)
		return 0;
	nc = conjuncts(f->where, cj, 0, 64);
	if (nc == 0)
		return 0;
	for (i = 0; i < nc; i++) {
		roots[2 * i] = N.a[cj[i]];
		roots[2 * i + 1] = N.b[cj[i]];
	}
	plan_build(&p, roots, (uint32_t)(2 * nc), 0);
	iv = xcalloc(p.n ? p.n : 1, sizeof *iv);
	g = *f;
	for (round = 0; round < 4; round++) {
		int changed = 0;
		for (j = 0; j < f->np; j++)
			if (TY_ISINT(g.p[j].ty) && !(TYK(g.p[j].ty) == K_UINT &&
			    TYW(g.p[j].ty) == 64)) {
				g.p[j].ranged = 1;
				g.p[j].lo = lo[j];
				g.p[j].hi = hi[j];
			}
		plan_ival(&p, &g, iv);
		for (i = 0; i < nc; i++) {
			int op = N.op[cj[i]], k;
			for (k = 0; k < 2; k++) {
				nid x = k ? N.b[cj[i]] : N.a[cj[i]];
				nid e = k ? N.a[cj[i]] : N.b[cj[i]];
				int s = var_under_casts(x), o = op;
				const struct ival *b;
				int64_t nlo, nhi;
				if (s < 0 || s >= f->np || !TY_ISINT(f->p[s].ty) ||
				    (TYK(f->p[s].ty) == K_UINT &&
				    TYW(f->p[s].ty) == 64))
					continue;
				b = &iv[p.root[2 * i + (k ? 0 : 1)]];
				if (b->top)
					continue;
				if (k)   /* e op x: x op' e */
					o = o == O_LT ? O_GT : o == O_GT ? O_LT :
					    o == O_LE ? O_GE : o == O_GE ? O_LE : o;
				nlo = lo[s];
				nhi = hi[s];
				switch (o) {
				case O_LT:
					if (b->hi > INT64_MIN && b->hi - 1 < nhi)
						nhi = b->hi - 1;
					break;
				case O_LE:
					if (b->hi < nhi)
						nhi = b->hi;
					break;
				case O_GT:
					if (b->lo < INT64_MAX && b->lo + 1 > nlo)
						nlo = b->lo + 1;
					break;
				case O_GE:
					if (b->lo > nlo)
						nlo = b->lo;
					break;
				case O_EQ:
					if (b->hi < nhi)
						nhi = b->hi;
					if (b->lo > nlo)
						nlo = b->lo;
					break;
				}
				(void)e;
				if (nlo != lo[s] || nhi != hi[s]) {
					lo[s] = nlo;
					hi[s] = nhi;
					if (narrowed)
						narrowed[s] = 1;
					changed = any = 1;
				}
			}
		}
		if (!changed)
			break;
	}
	free(iv);
	plan_free(&p);
	return any;
}

void
diter_init(struct diter *d, const struct func *f, uint64_t maxexh,
    uint32_t rsteps, uint64_t nsample)
{
	int j;
	uint64_t sz;
	int64_t ilo[MAXP], ihi[MAXP];

	memset(d, 0, sizeof *d);
	d->f = f;
	d->np = f->np;
	d->rsteps = rsteps < 2 ? 2 : rsteps;
	d->total = 1;
	d->rng = 0x9e3779b97f4a7c15ULL;
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &ilo[j], &ihi[j]);
		else
			ilo[j] = ihi[j] = 0;
	where_bounds(f, ilo, ihi, d->narrowed);
	for (j = 0; j < f->np; j++) {
		const struct param *p = &f->p[j];
		if (TYK(p->ty) == K_REAL) {
			d->real[j] = 1;
			d->rlo[j] = p->ranged ? p->rlo : -10;
			d->rhi[j] = p->ranged ? p->rhi : 10;
			sz = d->rsteps;
		} else if (TYK(p->ty) == K_FLOAT) {
			/* every float in the range, by ordered key */
			int w = TYW(p->ty);
			double mx = w == 32 ? 3.4028234663852886e38 :
			    w == 16 ? 65504.0 : 1.7976931348623157e308;
			d->real[j] = w == 32 ? 2 : w == 16 ? 4 : 3;
			d->rlo[j] = p->ranged ? p->rlo : -mx;
			d->rhi[j] = p->ranged ? p->rhi : mx;
			d->lo[j] = fkey(w, d->rlo[j]);
			d->hi[j] = fkey(w, d->rhi[j]);
			if (fromkey(w, d->lo[j]) < d->rlo[j])
				d->lo[j]++;
			if (fromkey(w, d->hi[j]) > d->rhi[j])
				d->hi[j]--;
			sz = d->hi[j] < d->lo[j] ? 0 :
			    (uint64_t)d->hi[j] - (uint64_t)d->lo[j] + 1;
		} else {
			d->lo[j] = ilo[j];
			d->hi[j] = ihi[j];
			if (d->hi[j] < d->lo[j])
				sz = 0;
			else {
				sz = (uint64_t)d->hi[j] - (uint64_t)d->lo[j];
				sz = sz == UINT64_MAX ? UINT64_MAX : sz + 1;
			}
			if (TYK(p->ty) == K_UINT && TYW(p->ty) == 64)
				sz = UINT64_MAX;
		}
		if (sz && d->total > UINT64_MAX / sz)
			d->total = UINT64_MAX;
		else
			d->total *= sz;
	}
	d->exhaustive = d->total <= maxexh;
	d->limit = d->exhaustive ? d->total : nsample;
}

static uint64_t
rnd(struct diter *d)
{
	d->rng ^= d->rng << 13;
	d->rng ^= d->rng >> 7;
	d->rng ^= d->rng << 17;
	return d->rng;
}

uint32_t
diter_next(struct diter *d, int64_t *in_i, double *in_r)
{
	uint32_t i;
	int j;

	for (i = 0; i < CH && d->pos < d->limit; i++, d->pos++) {
		uint64_t idx = d->pos;
		for (j = 0; j < d->np; j++) {
			size_t at = (size_t)j * CH + i;
			if (d->real[j] >= 2) {
				int w = d->real[j] == 2 ? 32 : d->real[j] == 4 ? 16 : 64;
				uint64_t sz = (uint64_t)d->hi[j] - (uint64_t)d->lo[j] + 1;
				uint64_t r;
				if (d->exhaustive) {
					r = idx % sz;
					idx /= sz;
				} else if (d->pos < 2)
					r = d->pos ? sz - 1 : 0;
				else
					r = sz ? rnd(d) % sz : 0;
				in_r[at] = fromkey(w, (int64_t)((uint64_t)d->lo[j] + r));
				in_i[at] = 0;
			} else if (d->exhaustive) {
				uint64_t sz = d->real[j] ? d->rsteps :
				    (uint64_t)d->hi[j] - (uint64_t)d->lo[j] + 1;
				uint64_t r = sz ? idx % sz : 0;
				idx = sz ? idx / sz : 0;
				if (d->real[j])
					in_r[at] = d->rlo[j] + (d->rhi[j] - d->rlo[j]) *
					    (double)r / (double)(d->rsteps - 1);
				else
					in_i[at] = (int64_t)((uint64_t)d->lo[j] + r);
			} else if (d->real[j]) {
				double u = (double)(rnd(d) >> 11) / 9007199254740992.0;
				in_r[at] = d->rlo[j] + (d->rhi[j] - d->rlo[j]) * u;
			} else {
				/* edge cases first, then uniform */
				int64_t lo = d->lo[j], hi = d->hi[j], v;
				uint64_t sz = (uint64_t)hi - (uint64_t)lo;
				static const int64_t sp[] = {0, 1, -1, 2};
				if (d->pos < 8) {
					uint64_t pick = (d->pos + (uint64_t)j * 3) % 8;
					v = pick == 0 ? lo : pick == 1 ? hi :
					    pick == 2 ? lo + 1 : pick == 3 ? hi - 1 :
					    sp[pick - 4];
					if (v < lo || v > hi)
						v = lo;
				} else if (sz == UINT64_MAX)
					v = (int64_t)rnd(d);
				else
					v = (int64_t)((uint64_t)lo + rnd(d) % (sz + 1));
				in_i[at] = v;
			}
		}
	}
	return i;
}

void
valid_mask(const struct plan *p, uint32_t wloc, uint32_t np, uint8_t *ok)
{
	uint32_t i;

	if (wloc == NOTOK) {
		memset(ok, 1, np);
		return;
	}
	for (i = 0; i < np; i++)
		ok[i] = !(COLF(p, wloc)[i] & F_UNDEF) && COLI(p, wloc)[i] != 0;
}

void
fmt_val(struct sbuf *b, ty_t t, int64_t vi, double vr)
{
	switch (TYK(t)) {
	case K_REAL:
		sb_printf(b, "%.6g", vr);
		break;
	case K_FLOAT:
		sb_printf(b, "%.*g", fdigits(TYW(t)), vr);
		break;
	case K_UINT:
		if (TYW(t) == 64)
			sb_printf(b, "%llu", (unsigned long long)vi);
		else
			sb_printf(b, "%lld", (long long)vi);
		break;
	default:
		sb_printf(b, "%lld", (long long)vi);
	}
}

void
fmt_point(struct sbuf *b, const struct func *f, const int64_t *in_i,
    const double *in_r, uint32_t i)
{
	int j;

	for (j = 0; j < f->np; j++) {
		if (j)
			sb_puts(b, ", ");
		sb_printf(b, "%s=", sym_str(f->p[j].name));
		fmt_val(b, f->p[j].ty, in_i[(size_t)j * CH + i],
		    in_r[(size_t)j * CH + i]);
	}
}

/* ---- interval bounds (ideal integers; reals where cheap) ---- */

/*
 * The least and the greatest x | y (x ^ y when xor) for x in A and y in
 * B, both non-negative, exact (Hacker's Delight, 4-3): from the top bit
 * down, a bound moves to where its bit gives way to the other's, with
 * all ones (a maximum) or zeros (a minimum) below it.
 */
static int64_t
min_or(const struct ival *A, const struct ival *B, int isxor)
{
	uint64_t a = (uint64_t)A->lo, b = (uint64_t)A->hi, c = (uint64_t)B->lo,
	    d = (uint64_t)B->hi, m, t;

	for (m = (uint64_t)1 << 62; m; m >>= 1)
		if (~a & c & m) {
			t = (a | m) & -m;
			if (t <= b) {
				a = t;
				if (!isxor)
					break;
			}
		} else if (a & ~c & m) {
			t = (c | m) & -m;
			if (t <= d) {
				c = t;
				if (!isxor)
					break;
			}
		}
	return (int64_t)(isxor ? a ^ c : a | c);
}

static int64_t
max_or(const struct ival *A, const struct ival *B, int isxor)
{
	uint64_t a = (uint64_t)A->lo, b = (uint64_t)A->hi, c = (uint64_t)B->lo,
	    d = (uint64_t)B->hi, m, t;

	for (m = (uint64_t)1 << 62; m; m >>= 1)
		if (b & d & m) {
			t = (b - m) | (m - 1);
			if (t >= a) {
				b = t;
				if (!isxor)
					break;
				continue;
			}
			t = (d - m) | (m - 1);
			if (t >= c) {
				d = t;
				if (!isxor)
					break;
			}
		}
	return (int64_t)(isxor ? b ^ d : b | d);
}

static int
ival_fits(const struct ity *t, int64_t lo, int64_t hi)
{
	if (t->k == K_UINT || t->k == K_BOOL)
		return lo >= 0 && (t->w >= 63 || (uint64_t)hi <= t->mask);
	if (t->k == K_SINT)
		return lo >= t->smin && hi <= t->smax;
	return 1;
}

static void
ival_full(const struct ity *t, struct ival *v)
{
	v->top = 0;
	if (t->k == K_UINT || t->k == K_BOOL) {
		v->lo = 0;
		v->hi = t->w >= 63 ? INT64_MAX : (int64_t)t->mask;
		if (t->w >= 63)
			v->top = 1;
	} else if (t->k == K_SINT) {
		v->lo = t->smin;
		v->hi = t->smax;
	} else
		v->top = 1;
}

void
plan_ival(const struct plan *p, const struct func *f, struct ival *out)
{
	uint32_t k, j;

	for (k = 0; k < p->n; k++) {
		struct ival *v = &out[k], *A = &out[p->a[k]], *B = &out[p->b[k]];
		const struct ity *t = &p->it[k];
		int op = p->op[k], ok = 1;
		int64_t x[4], r;

		memset(v, 0, sizeof *v);
		if (op == O_BARRIER || op == O_ATOMIC) {
			*v = *A;
			continue;
		}
		if (TYK(p->ty[k]) == K_FLOAT) {
			v->top = 1;
			continue;
		}
		if (TYK(p->ty[k]) == K_REAL) {
			switch (op) {
			case O_CONST:
				v->rlo = v->rhi = p->kr[k];
				break;
			case O_VAR: {
				const struct param *pp = &f->p[p->ki[k]];
				if (pp->ranged)
					v->rlo = pp->rlo, v->rhi = pp->rhi;
				else
					v->top = 1;
				break;
			}
			case O_ADD:
				v->rlo = A->rlo + B->rlo;
				v->rhi = A->rhi + B->rhi;
				v->top = A->top | B->top;
				break;
			case O_SUB:
				v->rlo = A->rlo - B->rhi;
				v->rhi = A->rhi - B->rlo;
				v->top = A->top | B->top;
				break;
			case O_NEG:
				v->rlo = -A->rhi;
				v->rhi = -A->rlo;
				v->top = A->top;
				break;
			case O_SIN:
			case O_COS:
				v->rlo = -1;
				v->rhi = 1;
				break;
			case O_EXP:
				v->rlo = exp(A->rlo);
				v->rhi = exp(A->rhi);
				v->top = A->top;
				break;
			default:
				v->top = 1;
			}
			continue;
		}
		if (TYK(p->ty[k]) == K_BOOL || (opinfo[op].fl & OF_CMP) ||
		    op == O_LAND || op == O_LOR || op == O_LNOT) {
			v->lo = 0;
			v->hi = 1;
			continue;
		}
		switch (op) {
		case O_CONST:
			v->lo = v->hi = p->ki[k];
			break;
		case O_VAR:
			param_bounds(&f->p[p->ki[k]], &v->lo, &v->hi);
			break;
		case O_ADD:
			ok = !A->top && !B->top && !add_ovf(A->lo, B->lo, &v->lo) &&
			    !add_ovf(A->hi, B->hi, &v->hi);
			break;
		case O_SUB:
			ok = !A->top && !B->top && !sub_ovf(A->lo, B->hi, &v->lo) &&
			    !sub_ovf(A->hi, B->lo, &v->hi);
			break;
		case O_NEG:
			ok = !A->top && A->lo != INT64_MIN;
			v->lo = ok ? -A->hi : 0;
			v->hi = ok ? -A->lo : 0;
			break;
		case O_MUL:
			ok = !A->top && !B->top &&
			    !mul_ovf(A->lo, B->lo, &x[0]) && !mul_ovf(A->lo, B->hi, &x[1]) &&
			    !mul_ovf(A->hi, B->lo, &x[2]) && !mul_ovf(A->hi, B->hi, &x[3]);
			if (ok) {
				v->lo = v->hi = x[0];
				for (j = 1; j < 4; j++) {
					if (x[j] < v->lo)
						v->lo = x[j];
					if (x[j] > v->hi)
						v->hi = x[j];
				}
			}
			break;
		case O_AND:
			ok = !A->top && !B->top && (A->lo >= 0 || B->lo >= 0);
			v->lo = 0;
			v->hi = A->lo >= 0 && B->lo >= 0 ?
			    (A->hi < B->hi ? A->hi : B->hi) :
			    A->lo >= 0 ? A->hi : B->hi;
			break;
		case O_OR:
		case O_XOR:
			ok = !A->top && !B->top && A->lo >= 0 && B->lo >= 0;
			if (ok) {
				v->lo = min_or(A, B, op == O_XOR);
				v->hi = max_or(A, B, op == O_XOR);
			}
			break;
		case O_SHL:
			ok = !A->top && !B->top && A->lo >= 0 && B->lo >= 0 &&
			    B->hi < 63 && A->hi <= (INT64_MAX >> B->hi);
			v->lo = ok ? A->lo << B->lo : 0;
			v->hi = ok ? A->hi << B->hi : 0;
			break;
		case O_SHR:
			ok = !A->top && !B->top && A->lo >= 0 && B->lo >= 0 && B->hi < 64;
			v->lo = ok ? A->lo >> B->hi : 0;
			v->hi = ok ? A->hi >> B->lo : 0;
			break;
		case O_DIV:
			ok = !A->top && !B->top && B->lo > 0 && A->lo >= 0;
			v->lo = ok ? A->lo / B->hi : 0;
			v->hi = ok ? A->hi / B->lo : 0;
			break;
		case O_MOD:
			ok = !A->top && !B->top && B->lo > 0 && A->lo >= 0;
			v->lo = 0;
			v->hi = ok ? (A->hi < B->hi - 1 ? A->hi : B->hi - 1) : 0;
			break;
		case O_MIN:
		case O_MAX:
			ok = !A->top && !B->top;
			if (op == O_MIN) {
				v->lo = A->lo < B->lo ? A->lo : B->lo;
				v->hi = A->hi < B->hi ? A->hi : B->hi;
			} else {
				v->lo = A->lo > B->lo ? A->lo : B->lo;
				v->hi = A->hi > B->hi ? A->hi : B->hi;
			}
			break;
		case O_ISQRT:
			ok = !A->top && A->lo >= 0;
			v->lo = ok ? (int64_t)isqrt64((uint64_t)A->lo) : 0;
			v->hi = ok ? (int64_t)isqrt64((uint64_t)A->hi) : 0;
			break;
		case O_POPCNT:
		case O_CLZ:
		case O_CTZ:
			v->lo = 0;
			v->hi = t->w;
			break;
		case O_SEL: {
			struct ival *C = &out[p->c[k]];
			ok = !B->top && !C->top;
			v->lo = B->lo < C->lo ? B->lo : C->lo;
			v->hi = B->hi > C->hi ? B->hi : C->hi;
			break;
		}
		case O_FETCH_BORDER:
			/* a texel or the border's value */
			if (out[p->c[k]].top)
				ok = 0;
			/* FALLTHROUGH */
		case O_FETCH_REPEAT: case O_FETCH_MIRROR: case O_FETCH_CLAMP:
		case O_FETCH_MCLAMP: case O_FETCH_LAYER: case O_FETCH_TEXEL:
		case O_FETCH_BUFFER:
		case O_INDEX: {
			uint32_t st = p->a[p->a[k]], cnt = p->b[p->a[k]];
			if (op == O_FETCH_BORDER) {
				v->lo = out[p->c[k]].lo;
				v->hi = out[p->c[k]].hi;
			}
			for (j = 0; j < cnt; j++) {
				struct ival *e = &out[p->lst[st + j]];
				if (e->top)
					ok = 0;
				if ((!j && op != O_FETCH_BORDER) || e->lo < v->lo)
					v->lo = e->lo;
				if ((!j && op != O_FETCH_BORDER) || e->hi > v->hi)
					v->hi = e->hi;
			}
			break;
		}
		case O_CAST:
			if (TY_FP(p->ty[p->a[k]])) {
				ok = 0;
				break;
			}
			*v = *A;
			ok = !A->top;
			break;
		case O_UDOT4X8: case O_SDOT4X8: case O_SUDOT4X8:
		case O_UDOT2X16: case O_SDOT2X16: case O_SUDOT2X16:
		case O_UDOT8X4: case O_SDOT8X4: case O_SUDOT8X4:
		case O_UDOT4X8S: case O_SDOT4X8S: case O_SUDOT4X8S:
		case O_UDOT2X16S: case O_SDOT2X16S: case O_SUDOT2X16S:
		case O_UDOT8X4S: case O_SDOT8X4S: case O_SUDOT8X4S: {
			struct ival *C = &out[p->c[k]];
			int sat = opinfo[op].arity == 3;
			ok = !A->top && !B->top && (!sat || !C->top) &&
			    dot_ival(op, A->lo, A->hi, B->lo, B->hi,
			    sat ? C->lo : 0, sat ? C->hi : 0, &v->lo, &v->hi);
			break;
		}
		case O_ABS:
			ok = !A->top && A->lo != INT64_MIN;
			if (ok) {
				int64_t m1 = A->lo < 0 ? -A->lo : A->lo;
				int64_t m2 = A->hi < 0 ? -A->hi : A->hi;
				v->hi = m1 > m2 ? m1 : m2;
				v->lo = (A->lo <= 0 && A->hi >= 0) ? 0 :
				    (m1 < m2 ? m1 : m2);
			}
			break;
		case O_TABLE:
			break;
		default:
			ok = 0;
		}
		(void)r;
		if (!ok) {
			ival_full(t, v);
			if (t->k == K_INT)
				v->top = 1;
			v->wrap = t->k != K_INT;
			continue;
		}
		if (!ival_fits(t, v->lo, v->hi)) {
			v->wrap = 1;
			ival_full(t, v);
		}
	}
}
