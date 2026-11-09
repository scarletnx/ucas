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

double
fround(int w, double v)
{
	return w == 32 ? (double)(float)v : v;
}

uint64_t
fbits(int w, double v)
{
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

/* a total order on the non-NaN values: consecutive keys are adjacent floats */
int64_t
fkey(int w, double v)
{
	uint64_t u = fbits(w, v);
	uint64_t sign = w == 32 ? 0x80000000ULL : 0x8000000000000000ULL;

	if (u & sign)
		return -(int64_t)(u & (sign - 1));
	return (int64_t)u;
}

double
fromkey(int w, int64_t k)
{
	uint64_t sign = w == 32 ? 0x80000000ULL : 0x8000000000000000ULL;

	if (k < 0)
		return frombits(w, (uint64_t)(-k) | sign);
	return frombits(w, (uint64_t)k);
}

/*
 * IEEE semantics: computed in double, then rounded to the width (for f32
 * this is the correctly rounded result for + - * / sqrt).  NaN and inf are
 * ordinary values; the flags only record where they first appear.
 */
double
scalar_float(int op, int w, double a, double b, uint8_t *fl)
{
	double r;

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
	if (op == O_FMA) {
		if (TYK(t) == K_FLOAT && TYW(t) == 32)
			*rr = (double)(float)((long double)N.kr[a] * N.kr[b] +
			    N.kr[c]);
		else
			*rr = fma(N.kr[a], N.kr[b], N.kr[c]);
		return isfinite(*rr);
	}
	ity_of(&it, t, 0);
	ity_of(&ita, ta, 0);
	if (TY_FP(t)) {
		if (op == O_ASFLOAT)
			*rr = frombits(TYW(t), (uint64_t)N.ki[a]);
		else if (op == O_CAST)
			*rr = TY_FP(ta) ? N.kr[a] : int_to_real(&ita, N.ki[a]);
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
plan_build(struct plan *p, const nid *roots, uint32_t nr, int ideal)
{
	uint8_t *mark = xcalloc(N.n, 1);
	uint32_t *map = xcalloc(N.n, sizeof *map);
	nid *stack = xmalloc(N.n * sizeof *stack), g;
	uint32_t sp = 0, i, k, nl = 0, j;

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

	while (lo < hi) {
		uint32_t m = (lo + hi) / 2;
		if (p->gid[m] < g)
			lo = m + 1;
		else
			hi = m;
	}
	return lo < p->n && p->gid[lo] == g ? lo : NOTOK;
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
				if (t->k == K_FLOAT && t->w == 32)
					for (i = 0; i < np; i++)
						orr[i] = (float)orr[i];
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
			 * f64: fma().  f32: the product is exact in long
			 * double and the sum rounds there first (64 bits),
			 * so a double rounding is possible but very rare.
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
				orr[i] = fw == 32 ? (double)(float)((long double)av[i] *
				    bv[i] + cv[i]) : fma(av[i], bv[i], cv[i]);
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
						orr[i] = fw ? fround(fw,
						    int_to_real(ta, av[i])) :
						    int_to_real(ta, av[i]);
					of[i] = af[i] & F_UNDEF;
				}
			} else if (real && op == O_CAST) {
				const double *av = COLR(p, a);
				for (i = 0; i < np; i++) {
					orr[i] = fw ? fround(fw, av[i]) : av[i];
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

void
diter_init(struct diter *d, const struct func *f, uint64_t maxexh,
    uint32_t rsteps, uint64_t nsample)
{
	int j;
	uint64_t sz;

	memset(d, 0, sizeof *d);
	d->f = f;
	d->np = f->np;
	d->rsteps = rsteps < 2 ? 2 : rsteps;
	d->total = 1;
	d->rng = 0x9e3779b97f4a7c15ULL;
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
			    1.7976931348623157e308;
			d->real[j] = w == 32 ? 2 : 3;
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
			param_bounds(p, &d->lo[j], &d->hi[j]);
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
				int w = d->real[j] == 2 ? 32 : 64;
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
		sb_printf(b, TYW(t) == 32 ? "%.9g" : "%.17g", vr);
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
				uint64_t m = (uint64_t)(A->hi > B->hi ? A->hi : B->hi);
				int bb = bits_for(m);
				v->lo = op == O_OR ? (A->lo > B->lo ? A->lo : B->lo) : 0;
				v->hi = bb >= 63 ? INT64_MAX : (int64_t)((1ULL << bb) - 1);
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
		case O_INDEX: {
			uint32_t st = p->a[p->a[k]], cnt = p->b[p->a[k]];
			for (j = 0; j < cnt; j++) {
				struct ival *e = &out[p->lst[st + j]];
				if (e->top)
					ok = 0;
				if (!j || e->lo < v->lo)
					v->lo = e->lo;
				if (!j || e->hi > v->hi)
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
