/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * dword: fp64 arithmetic without fp64.
 *
 * A double computation is rewritten over f32 in two ways: with f32 alone,
 * and as double-word ("float-float") arithmetic, where a value is an
 * unevaluated sum hi + lo of two f32 with |lo| <= u |hi|, u = 2^-24, built
 * on the error-free transformations 2Sum, Fast2Sum and 2Prod (with fma).
 * Each candidate carries a bound on its error against the exact value,
 * derived here from the standard model fl(a op b) = (a op b)(1 + d),
 * |d| <= u, and the exactness of the transformations; the candidates are
 * priced under the cost model, and the cheapest one that meets the
 * required accuracy replaces the f64 operations when it costs less than
 * they do.  Nothing here names a machine: under a model whose f64 is as
 * cheap as its f32 no candidate is cheaper and nothing is rewritten.
 *
 * The algorithms are the ones analysed by Joldes, Muller and Popescu,
 * "Tight and rigorous error bounds for basic building blocks of
 * double-word arithmetic" (ACM TOMS 44(2), 2017): DWPlusFP, SloppyDWPlusDW,
 * AccurateDWPlusDW, DWTimesFP3, DWTimesDW3, DWDivFP3, DWDivDW2 with
 * DWTimesFP3, and the Newton square root.  Their paper proves tighter
 * constants by case analyses; the bounds used here are the ones the
 * standard model gives line by line (printed with the derivation), with
 * the subtractions Sterbenz makes exact taken as exact and DWPlusFP's
 * and AccurateDWPlusDW's cases on their high words; the sloppy
 * addition's needs its cancellation bounded.
 *
 * A bound is relative where the signs allow it and absolute always:
 * every value carries the interval of its exact value (from the
 * parameters' ranges), so an addition's amplification kappa = (|x| +
 * |y|) / |x + y| is the largest over the box, and overflow, underflow of
 * an error term, a zero divisor and Fast2Sum's precondition are either
 * derived from the ranges or stated as conditions.
 */
#include "dword.h"
#include "fuse.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define U 0x1p-24L		/* f32 unit roundoff */
#define U64 0x1p-53L		/* f64 unit roundoff */
#define LINF ((long double)INFINITY)
#define SLACK (1 + 0x1p-40L)	/* over the rounding of the bounds themselves */

struct iv {
	double lo, hi;
	int top;
};

struct err {
	long double rel, abs_;	/* against the exact value; LINF: none */
};

struct dv {
	nid hi, lo;		/* f32; lo == 0: the value is hi alone */
	struct iv iv;		/* the exact value */
	struct err e, e64;	/* this form's error, the f64 program's */
};

/* the building blocks a derivation used */
enum {
	DB_2SUM = 1, DB_F2SUM = 2, DB_2PROD = 4, DB_ADD_FP = 8,
	DB_ADD_SLOPPY = 16, DB_ADD_ACC = 32, DB_MUL_FP = 64, DB_MUL_DD = 128,
	DB_DIV_FP = 256, DB_DIV_DD = 512, DB_SQRT = 1024, DB_SPLIT = 2048,
	DB_SINGLE = 4096, DB_MINMAX = 8192
};

#define MAXLEAF 48
static struct {
	const struct func *f;
	int mode;		/* 0 f32 alone, 1 pairs */
	uint32_t lim;		/* nodes that existed before */
	struct dv *memo;
	uint8_t *done;
	uint8_t *acc;		/* an add of the source: the accurate form */
	uint8_t *eft;		/* nodes of the form that need NoContraction */
	uint32_t eftcap;
	unsigned used;
	long double maxmag;	/* the largest magnitude of a value */
	int mag_unknown;	/* a value has no interval */
	int uflow;		/* a product's error term may underflow */
	int nprod;		/* error-free products */
	int div0;		/* a divisor's interval holds 0 */
	int sqrt0;		/* a square root's operand may be <= 0 */
	int pre_cases;		/* Fast2Sum's precondition by an addition's cases */
	int nsloppy;
	nid worst;		/* the sloppy add with the largest kappa */
	long double worstk;
	int opaque;		/* f64 values kept in f64 and split */
	int splitvar;		/* f64 parameters split in the shader */
	nid leaf[MAXLEAF];	/* values the check reads from the plan */
	int nleaf;
	struct sbuf table;	/* the composition, a row per value */
	char unsup[120];
	int accv;		/* a loop's sum: 1 + its f64 parameter, else 0 */
	nid acch, accl;		/* the pair its passes carry (f32 parameters) */
	struct iv acciv;	/* where the carried value stays */
} L;

static long double
ldmax(long double a, long double b)
{
	return a > b ? a : b;
}

static long double
ldmin(long double a, long double b)
{
	return a < b ? a : b;
}

/* a * b with 0 * inf = 0: a zero error stays zero */
static long double
lmul(long double a, long double b)
{
	if (a == 0 || b == 0)
		return 0;
	return a * b;
}

/* ---- intervals of the exact value ---- */

static struct iv
iv_make(double lo, double hi)
{
	struct iv r;

	r.top = !(lo <= hi) || isinf(lo) || isinf(hi);
	/* outward by one step; an end at 0 is exact (0 times anything) */
	r.lo = lo == 0 ? 0 : nextafter(lo, -INFINITY);
	r.hi = hi == 0 ? 0 : nextafter(hi, INFINITY);
	return r;
}

static struct iv
iv_top(void)
{
	struct iv r = {0, 0, 1};

	return r;
}

static struct iv
iv_point(double v)
{
	struct iv r;

	r.lo = r.hi = v;
	r.top = isnan(v) || isinf(v);
	return r;
}

static long double
iv_mag(const struct iv *a)
{
	if (a->top)
		return LINF;
	return ldmax(fabsl((long double)a->lo), fabsl((long double)a->hi));
}

/* the smallest |v| in the interval: 0 when it holds 0 */
static long double
iv_mig(const struct iv *a)
{
	if (a->top || (a->lo <= 0 && a->hi >= 0))
		return 0;
	return a->lo > 0 ? (long double)a->lo : -(long double)a->hi;
}

static struct iv
iv_neg(struct iv a)
{
	struct iv r = a;

	r.lo = -a.hi;
	r.hi = -a.lo;
	return r;
}

static struct iv
iv_add(struct iv a, struct iv b)
{
	if (a.top || b.top)
		return iv_top();
	return iv_make(a.lo + b.lo, a.hi + b.hi);
}

static struct iv
iv_mul(struct iv a, struct iv b)
{
	double p[4], lo, hi;
	int i;

	if (a.top || b.top)
		return iv_top();
	p[0] = a.lo * b.lo;
	p[1] = a.lo * b.hi;
	p[2] = a.hi * b.lo;
	p[3] = a.hi * b.hi;
	lo = hi = p[0];
	for (i = 1; i < 4; i++) {
		if (p[i] < lo)
			lo = p[i];
		if (p[i] > hi)
			hi = p[i];
	}
	return iv_make(lo, hi);
}

static struct iv
iv_div(struct iv a, struct iv b)
{
	struct iv r;

	if (a.top || b.top || (b.lo <= 0 && b.hi >= 0))
		return iv_top();
	r.top = 0;
	r.lo = 1 / b.hi;
	r.hi = 1 / b.lo;
	r = iv_make(r.lo, r.hi);
	return iv_mul(a, r);
}

static struct iv
iv_sqrt(struct iv a)
{
	if (a.top || a.hi < 0)
		return iv_top();
	return iv_make(sqrt(a.lo < 0 ? 0 : a.lo), sqrt(a.hi));
}

static struct iv
iv_abs(struct iv a)
{
	if (a.top)
		return a;
	if (a.lo >= 0)
		return a;
	if (a.hi <= 0)
		return iv_neg(a);
	return iv_make(0, a.hi > -a.lo ? a.hi : -a.lo);
}

/* an f32, integer or real leaf expression's interval (its exact value) */
static struct iv
leaf_iv(const struct func *f, nid n, int depth)
{
	ty_t t = N.ty[n];
	struct iv a, b;

	if (depth > 24)
		return iv_top();
	switch (N.op[n]) {
	case O_CONST:
		return iv_point(TY_FP(t) ? N.kr[n] : (double)N.ki[n]);
	case O_VAR: {
		const struct param *p;
		if (N.ki[n] < 0 || N.ki[n] >= f->np)
			return iv_top();
		p = &f->p[N.ki[n]];
		if (TY_FP(t)) {
			if (!p->ranged)
				return iv_top();
			a.lo = p->rlo;
			a.hi = p->rhi;
			a.top = 0;
			return a;
		} else {
			int64_t lo, hi;
			param_bounds(p, &lo, &hi);
			a.lo = (double)lo;
			a.hi = (double)hi;
			a.top = TYK(t) == K_INT && !p->ranged;
			return a;
		}
	}
	case O_CAST:
		a = leaf_iv(f, N.a[n], depth + 1);
		if (TY_FP(t) || a.top)
			return a;
		{
			/* to an integer type: its own range when a's leaves it */
			struct ity it;
			ity_of(&it, t, 0);
			if (TYK(t) == K_UINT && (a.lo < 0 || a.hi >
			    (double)it.mask))
				return iv_make(0, (double)it.mask);
			if (TYK(t) == K_SINT && (a.lo < (double)it.smin ||
			    a.hi > (double)it.smax))
				return iv_make((double)it.smin, (double)it.smax);
		}
		return a;
	case O_NEG:
		return iv_neg(leaf_iv(f, N.a[n], depth + 1));
	case O_ABS:
		return iv_abs(leaf_iv(f, N.a[n], depth + 1));
	case O_ADD: case O_SUB: case O_MUL: case O_DIV:
		if (!TY_FP(t))
			break;
		a = leaf_iv(f, N.a[n], depth + 1);
		b = leaf_iv(f, N.b[n], depth + 1);
		if (N.op[n] == O_ADD)
			return iv_add(a, b);
		if (N.op[n] == O_SUB)
			return iv_add(a, iv_neg(b));
		if (N.op[n] == O_MUL)
			return iv_mul(a, b);
		return iv_div(a, b);
	case O_SQRT:
		return iv_sqrt(leaf_iv(f, N.a[n], depth + 1));
	}
	if (TY_SIZED(t)) {
		struct ity it;
		ity_of(&it, t, 0);
		if (TYW(t) > 52)
			return iv_top();
		if (TYK(t) == K_SINT)
			return iv_make((double)it.smin, (double)it.smax);
		return iv_make(0, (double)it.mask);
	}
	return iv_top();
}

/*
 * The amplification of x + y: the largest (|x| + |y|) / |x + y| over the
 * box.  In each sign quadrant the ratio is a quotient of two linear
 * functions, monotone along both axes, so the largest value is at a
 * corner of the box cut by the axes.
 */
static long double
kappa_of(const struct iv *x, const struct iv *y)
{
	double xs[3], ys[3];
	int nx = 0, ny = 0, i, j;
	long double k = 1;

	if (x->top || y->top)
		return LINF;
	if (x->lo + y->lo <= 0 && x->hi + y->hi >= 0) {
		if (x->lo == 0 && x->hi == 0 && y->lo == 0 && y->hi == 0)
			return 1;
		return LINF;
	}
	xs[nx++] = x->lo;
	xs[nx++] = x->hi;
	if (x->lo < 0 && x->hi > 0)
		xs[nx++] = 0;
	ys[ny++] = y->lo;
	ys[ny++] = y->hi;
	if (y->lo < 0 && y->hi > 0)
		ys[ny++] = 0;
	for (i = 0; i < nx; i++)
		for (j = 0; j < ny; j++) {
			long double s = fabsl((long double)xs[i] + ys[j]);
			long double a = fabsl((long double)xs[i]) +
			    fabsl((long double)ys[j]);
			if (s == 0)
				return LINF;
			if (a / s > k)
				k = a / s;
		}
	return k * SLACK;
}

/* ---- the local errors, line by line ---- */

/*
 * An addition's local error |z - (x~ + y~)| <= L(S, A) for S >= |x~ + y~|
 * and A >= |xh| + |yh|; linear in (S, A), so L(1, kappa~ / (1 - u)) is the
 * error relative to x~ + y~.  *pre: Fast2Sum's precondition |a| >= |b|
 * follows from the same magnitudes.
 */
enum { AK_EXACT, AK_ACC, AK_SLOPPY, AK_FP, AK_SINGLE, AK_F64 };

static long double
add_local(int kind, long double S, long double A, int *pre)
{
	long double sh, sl, th, tl, c, vh, vl, v, w;

	*pre = 1;
	if (isinf(S) || isinf(A))
		return kind == AK_EXACT ? 0 : LINF;
	switch (kind) {
	case AK_EXACT:
		return 0;
	case AK_SINGLE:
		return U * S;
	case AK_F64:
		return U64 * S;
	case AK_ACC:
		sh = (1 + U) * (S + U * A);	/* |xh + yh| <= S + u A */
		sl = U * sh;
		th = (1 + U) * U * A;		/* |xl + yl| <= u A */
		tl = U * th;
		c = (1 + U) * (sl + th);
		vh = (1 + U) * (sh + c);
		vl = U * vh;
		w = (1 + U) * (tl + vl);
		*pre = (1 - U) * (S - U * A) >= c &&
		    (1 - U) * ((1 - U) * (S - U * A) - c) >= w;
		return (U * (sl + th) + U * (tl + vl)) * SLACK;
	case AK_SLOPPY:
		sh = (1 + U) * (S + U * A);
		sl = U * sh;
		v = (1 + U) * U * A;
		w = (1 + U) * (sl + v);
		*pre = (1 - U) * (S - U * A) >= w;
		return (U * (U * A) + U * (sl + v)) * SLACK;
	case AK_FP:
		sh = (1 + U) * (S + U * A);
		sl = U * sh;
		v = (1 + U) * (U * A + sl);
		*pre = (1 - U) * (S - U * A) >= v;
		return U * (U * A + sl) * SLACK;
	}
	return LINF;
}

/* DWTimesDW3: (xh + xl)(yh + yl), relative to the exact product */
static long double
eta_mul_dd(void)
{
	long double cl1 = U, tl0 = (1 + U) * U * U, e0 = U * U * U;
	long double t1 = U + tl0, e1 = U * t1, tl1 = (1 + U) * t1;
	long double t2 = tl1 + U, e2 = U * t2, cl2 = (1 + U) * t2;
	long double e3 = U * (cl1 + cl2);

	return (e0 + e1 + e2 + e3) / ((1 - U) * (1 - U)) * SLACK;
}

/* DWTimesFP3: (xh + xl) y */
static long double
eta_mul_fp(void)
{
	return U * (U + U) / (1 - U) * SLACK;
}

/*
 * DWDivFP3: (xh + xl) / y; single: xl = 0 and its addition is skipped.
 * th = RN(xh / y) and (ph, pl) = 2Prod(th, y): th y is within a factor
 * 1 +- u of xh and ph within (1 +- u)^2, so xh - ph is a float (Sterbenz);
 * and xh - th y, a multiple of ulp(th) ulp(y) at most ulp(th) |y| / 2 in
 * magnitude (fewer than 2^23 of them, 2Prod's own condition keeping
 * them above underflow), is one too, so RN(RN(xh - ph) - pl) is exact.
 * The residual x~ - th y is at most 2u |xh| (u |xh| when xl = 0): its sum
 * with xl rounds once, the division once.
 */
static long double
eta_div_fp(int single)
{
	long double rho = single ? U : 2 * U;
	long double eps = single ? 0 : U * rho;

	return (eps + U * (rho + eps)) / (1 - U) * SLACK;
}

/* DWDivDW2 with DWTimesFP3 for the product */
static long double
eta_div_dd(void)
{
	long double emf = eta_mul_fp();
	long double rho = 3 * U + U * U;
	long double er = emf * (1 + U) * (1 + U);
	long double rh = (1 + U) * (1 + U) * (1 + emf) / (1 - U), rl = U * rh;
	/* rh is within a factor 1 +- 3u of xh: xh - rh is a float
	 * (Sterbenz), its RN exact */
	long double err1 = 0;
	long double d2 = U + rl, err2 = U * d2;
	long double d3 = rho + er + err1 + err2, err3 = U * d3;
	long double eps = er + err1 + err2 + err3;

	return (U * rho + (1 + U) * (eps + U * (rho + eps))) / (1 - U) * SLACK;
}

/* the Newton square root of xh + xl */
static long double
eta_sqrt(void)
{
	long double r1 = 2 * U + U * U, err1 = U * r1;
	long double d2 = U + r1 + err1, err2 = U * d2;
	long double rho = 3 * U + U * U, eps = err1 + err2;
	long double first = rho * rho / (8 * (1 - U) * (1 - U) * (1 - U));
	long double second = (eps + U * (rho + eps)) / (2 * (1 - U));

	return (first + second) / (1 - U) * SLACK;
}

/* ---- composing the errors ---- */

static void
comp_mul(struct err *z, const struct err *x, const struct err *y,
    const struct iv *ix, const struct iv *iy, long double eta)
{
	long double X = iv_mag(ix), Y = iv_mag(iy);

	z->rel = (1 + x->rel) * (1 + y->rel) * (1 + eta) - 1;
	z->abs_ = lmul(X, y->abs_) + lmul(Y, x->abs_) + lmul(x->abs_, y->abs_) +
	    lmul(eta, lmul(X + x->abs_, Y + y->abs_));
	if (isnan(z->abs_))
		z->abs_ = LINF;
}

static void
comp_div(struct err *z, const struct err *x, const struct err *y,
    const struct iv *ix, const struct iv *iy, long double eta)
{
	long double X = iv_mag(ix), ym = iv_mig(iy);

	if (y->rel < 1)
		z->rel = (1 + x->rel) * (1 + eta) / (1 - y->rel) - 1;
	else
		z->rel = LINF;
	if (ym > y->abs_ && !isinf(X)) {
		long double den = ym - y->abs_;
		z->abs_ = (x->abs_ + lmul(X / ym, y->abs_)) / den +
		    lmul(eta, (X + x->abs_) / den);
	} else
		z->abs_ = LINF;
}

static void
comp_sqrt(struct err *z, const struct err *x, const struct iv *ix,
    long double eta)
{
	long double xm = iv_mig(ix), X = iv_mag(ix);

	if (x->rel < 1) {
		long double g = 1 - sqrtl(1 - x->rel);
		z->rel = (1 + g) * (1 + eta) - 1;
	} else
		z->rel = LINF;
	if (ix->lo > 0 && xm > x->abs_ && !isinf(X))
		z->abs_ = x->abs_ / (2 * sqrtl(xm - x->abs_)) +
		    lmul(eta, sqrtl(X + x->abs_));
	else
		z->abs_ = x->abs_ == 0 && eta == 0 ? 0 : LINF;
}

/*
 * DWPlusFP's local error whatever cancels, by two cases on its high
 * words xh and y (s = xh + y, (sh, sl) = 2Sum(xh, y), v = RN(xl + sl)):
 *   opposite signs and |y| in [|xh|/2, 2|xh|]: s is a float (Sterbenz),
 *   so sl = 0 and v = xl, and the result is x~ + y exactly; Fast2Sum(sh,
 *   v) is exact as sh's exponent is at least v's (a nonzero s is a
 *   multiple of ulp(xh)/2, and |xl| <= ulp(xh)/2);
 *   otherwise |s| > |xh|/2: the one rounding errs by u |xl + sl| <= u^2
 *   (|xh| + |s|) < 3 u^2 |s|, |x~ + y| >= |s| - u |xh| > (1 - 2u) |s|,
 *   and |v| < |sh|, so Fast2Sum is exact.
 * So 3 u^2 / (1 - 2u) |x~ + y| for every kappa (the paper: 2 u^2).
 */
static long double
eta_add_fp_split(void)
{
	return 3 * U * U / (1 - 2 * U) * SLACK;
}

/*
 * AccurateDWPlusDW's local error whatever cancels, with |xh| >= |yh|
 * (it is symmetric), s = xh + yh, t = xl + yl, A = |xh| + |yh|; the
 * error is u |sl + th| + u |tl + vl| while both Fast2Sums are exact:
 *   opposite signs and |yh| >= |xh|/2: s is a float (Sterbenz), sl = 0
 *   and c = th; s is a multiple of q = ulp(xh)/2, |xl|, |yl| <= q and
 *   |th| <= 2q, so q is a multiple of ulp(th) and s + th of ulp(th): it
 *   is a float (vl = 0, w = tl, the result exact) unless it is 2^p
 *   ulp(th) or more, where |tl| <= ulp(th)/2 <= u/2 |s + th| and the
 *   error is below u^2 (1.5 + u) / (1 - u/2); Fast2Sum(s, th) is exact as
 *   s's exponent is th's or more, or s = +-q, th = +-2q and the sum is a
 *   float; Fast2Sum(vh, w) as |w| < |vh|;
 *   otherwise A <= 3 |s|: |sl + th| <= 4u (1 + u) |s|, |vh| <= (1 + u)^2
 *   (1 + 4u (1 + u)) |s|, |x~ + y~| >= (1 - 3u) |s|, and |c| < |sh|, |w| <
 *   |vh|, so both Fast2Sums are exact.
 * The larger, about 5 u^2 (the paper: 3 u^2 / (1 - 4u)).
 */
static long double
eta_add_acc_split(void)
{
	long double e1 = U * U * (4 * (1 + U) + 3 * U * (1 + U) * (1 + U) +
	    (1 + U) * (1 + U) * (1 + 4 * U * (1 + U))) / (1 - 3 * U);
	long double e2 = U * U * (1.5L + U) / (1 - U / 2);

	return ldmax(e1, e2) * SLACK;
}

/* the bound whatever cancels, for the additions that have one: 0 none */
static long double
eta_add_split(int kind)
{
	return kind == AK_FP ? eta_add_fp_split() : kind == AK_ACC ?
	    eta_add_acc_split() : 0;
}

/* y's interval already negated for a subtraction */
static void
comp_add(struct err *z, const struct err *x, const struct err *y,
    const struct iv *ix, const struct iv *iy, const struct iv *iz, int kind,
    long double kappa, int *pre)
{
	long double rmax = ldmax(x->rel, y->rel), S, A;
	int p2;

	*pre = 1;
	if (rmax == 0 && (kind == AK_EXACT || kind == AK_SINGLE ||
	    kind == AK_F64)) {
		/* exact operands, one rounding: whatever cancels */
		z->rel = kind == AK_EXACT ? 0 : kind == AK_SINGLE ? U : U64;
	} else if (rmax == 0 && eta_add_split(kind) > 0) {
		/* exact operands: the split bound, whatever cancels */
		z->rel = eta_add_split(kind);
		if (!isinf(kappa))
			z->rel = ldmin(z->rel, add_local(kind, 1, kappa / (1 - U),
			    pre));
		else
			*pre = 0;
	} else if (!isinf(kappa) && !isinf(rmax) && kappa * rmax < 0.25L) {
		long double kt = kappa * (1 + rmax) / (1 - kappa * rmax);
		long double eta = add_local(kind, 1, kt / (1 - U), pre);
		if (eta_add_split(kind) > 0)
			eta = ldmin(eta, eta_add_split(kind));
		z->rel = kappa * rmax + eta * (1 + kappa * rmax);
	} else {
		z->rel = LINF;
		*pre = kind == AK_EXACT || kind == AK_SINGLE || kind == AK_F64;
	}
	/* where the magnitudes do not give it, the split's cases do: 2 */
	if (eta_add_split(kind) > 0 && !*pre)
		*pre = 2;
	S = iv_mag(iz) + x->abs_ + y->abs_;
	A = (iv_mag(ix) + x->abs_ + iv_mag(iy) + y->abs_) / (1 - U);
	z->abs_ = x->abs_ + y->abs_ + add_local(kind, S, A, &p2);
	if (eta_add_split(kind) > 0)
		z->abs_ = ldmin(z->abs_, x->abs_ + y->abs_ +
		    lmul(eta_add_split(kind), S));
	if (isnan(z->abs_))
		z->abs_ = LINF;
}

/* ---- the forms: f32 nodes ---- */

static nid
f32k(double v)
{
	return mk_cast(TY_F32, mk_const_r(v));
}

static void
mark_eft(nid n)
{
	if (n >= L.eftcap) {
		uint32_t nc = newcap(L.eftcap, n + 1);
		L.eft = xrealloc(L.eft, nc);
		memset(L.eft + L.eftcap, 0, nc - L.eftcap);
		L.eftcap = nc;
	}
	L.eft[n] = 1;
}

static nid
op2(int op, nid a, nid b, int eft)
{
	nid r = mk_bin(op, a, b);

	if (eft && N.op[r] != O_CONST && N.op[r] != O_VAR)
		mark_eft(r);
	return r;
}

/* s + t = a + b (sub: a - b) exactly; six operations */
static void
two_sum(nid a, nid b, int sub, nid *s, nid *t)
{
	nid a1, b1, da, db;

	L.used |= DB_2SUM;
	if (!sub) {
		*s = op2(O_ADD, a, b, 1);
		a1 = op2(O_SUB, *s, b, 1);
		b1 = op2(O_SUB, *s, a1, 1);
		da = op2(O_SUB, a, a1, 1);
		db = op2(O_SUB, b, b1, 1);
		*t = op2(O_ADD, da, db, 1);
	} else {
		*s = op2(O_SUB, a, b, 1);
		a1 = op2(O_ADD, *s, b, 1);
		b1 = op2(O_SUB, *s, a1, 1);
		da = op2(O_SUB, a, a1, 1);
		db = op2(O_ADD, b, b1, 1);
		*t = op2(O_SUB, da, db, 1);
	}
}

/* s + t = a + b exactly when |a| >= |b|; three operations */
static void
fast_two_sum(nid a, nid b, nid *s, nid *t)
{
	nid z;

	L.used |= DB_F2SUM;
	*s = op2(O_ADD, a, b, 1);
	z = op2(O_SUB, *s, a, 1);
	*t = op2(O_SUB, b, z, 1);
}

/* p + e = a b exactly (no underflow of e); a multiply and an fma */
static void
two_prod(nid a, nid b, nid *p, nid *e)
{
	nid r;

	L.used |= DB_2PROD;
	L.nprod++;
	*p = op2(O_MUL, a, b, 1);
	r = mk_fma(a, b, mk_un(O_NEG, *p));
	mark_eft(r);
	*e = r;
}

static struct dv lower(nid n);

static void
row(nid n, const struct dv *z, const char *how)
{
	char ex[44], rg[64], eb[40];
	const char *s = node_str(n);

	if (strlen(s) > 40)
		snprintf(ex, sizeof ex, "%.37s...", s);
	else
		snprintf(ex, sizeof ex, "%s", s);
	if (z->iv.top)
		snprintf(rg, sizeof rg, "(no range)");
	else
		snprintf(rg, sizeof rg, "[%.4g, %.4g]", z->iv.lo, z->iv.hi);
	if (!isinf(z->e.rel))
		snprintf(eb, sizeof eb, "%.4Lg %s", z->e.rel / (L.mode ? U * U : U),
		    L.mode ? "u^2" : "u");
	else if (!isinf(z->e.abs_))
		snprintf(eb, sizeof eb, "%.3Lg abs", z->e.abs_);
	else
		snprintf(eb, sizeof eb, "unbounded");
	sb_printf(&L.table, "      %-40s %-24s %-14s %s\n", ex, rg, eb, how);
}

static void
track(const struct dv *z)
{
	long double m = iv_mag(&z->iv);

	if (isinf(m))
		L.mag_unknown = 1;
	else if (m > L.maxmag)
		L.maxmag = m;
}

/* the error term of a product a b underflows below 2^-102 */
static void
track_prod(const struct iv *p)
{
	long double m = iv_mig(p);

	if (p->top || (m < 0x1p-100L && !(p->lo == 0 && p->hi == 0)))
		L.uflow = 1;
}

static void
add_leaf(nid n)
{
	int i;

	for (i = 0; i < L.nleaf; i++)
		if (L.leaf[i] == n)
			return;
	if (L.nleaf < MAXLEAF)
		L.leaf[L.nleaf++] = n;
}

/* an f64 value kept as it is: hi = f32(v), lo = f32(v - f64(hi)) */
static struct dv
split64(nid n, struct iv iv)
{
	struct dv z;
	long double m = iv_mag(&iv);

	memset(&z, 0, sizeof z);
	add_leaf(n);
	z.hi = mk_cast(TY_F32, n);
	z.iv = iv;
	if (L.mode) {
		z.lo = mk_cast(TY_F32, mk_bin(O_SUB, n, mk_cast(TY_F64, z.hi)));
		L.used |= DB_SPLIT;
		z.e.rel = U * U * SLACK;
		z.e.abs_ = lmul(U * U * SLACK, m);
	} else {
		z.e.rel = U;
		z.e.abs_ = lmul(U, m);
	}
	return z;
}

static int
pow2_const(nid n, double *v)
{
	double a;
	int e;

	if (N.op[n] != O_CONST || !TY_FP(N.ty[n]))
		return 0;
	a = fabs(N.kr[n]);
	if (a == 0 || isinf(a) || isnan(a) || frexp(a, &e) != 0.5)
		return 0;
	if (a < 0x1p-60 || a > 0x1p60)
		return 0;
	*v = N.kr[n];
	return 1;
}

/* x scaled by a power of two: exact */
static struct dv
scale(const struct dv *x, double k)
{
	struct dv z = *x;
	nid kk = f32k(k);

	z.hi = mk_bin(O_MUL, x->hi, kk);
	z.lo = x->lo ? mk_bin(O_MUL, x->lo, kk) : 0;
	z.iv = iv_mul(x->iv, iv_point(k));
	z.e.abs_ = lmul(x->e.abs_, fabsl((long double)k));
	z.e64.abs_ = lmul(x->e64.abs_, fabsl((long double)k));
	return z;
}

/*
 * min(x~, y~), max(x~, y~): one of the two pairs, chosen by the sign of
 * d = RN(RN(xh - yh) + RN(xl - yl)).  With xh and yh within a factor of
 * two xh - yh is exact (Sterbenz) and d has the sign of
 * (x~ - y~) + d1 (xl - yl), |d1| <= u: the wrong pair is chosen only when
 * |x~ - y~| <= u |xl - yl| <= u^2 (|xh| + |yh|), and then it is within
 * that of the right one.  Farther apart, or of opposite signs,
 * |xh - yh| >= max(|xh|, |yh|) / 2 and the low words cannot turn the
 * sign.  So the choice adds at most 3 u^2 (relative) to the larger of the
 * operands' bounds: min and max move no farther than their operands do.
 * Two single f32 are compared exactly by the instruction itself.
 */
#define DW_PICK (3 * U * U * SLACK)

static struct dv
lower_minmax(nid n)
{
	struct dv x = lower(N.a[n]), y = lower(N.b[n]), z;
	int mx = N.op[n] == O_MAX;
	long double er, ea, mag;
	nid dh, dl, d, c, zero = f32k(0);

	memset(&z, 0, sizeof z);
	if (x.iv.top || y.iv.top)
		z.iv = iv_top();
	else if (mx)
		z.iv = iv_make(x.iv.lo > y.iv.lo ? x.iv.lo : y.iv.lo,
		    x.iv.hi > y.iv.hi ? x.iv.hi : y.iv.hi);
	else
		z.iv = iv_make(x.iv.lo < y.iv.lo ? x.iv.lo : y.iv.lo,
		    x.iv.hi < y.iv.hi ? x.iv.hi : y.iv.hi);
	er = x.e.rel > y.e.rel ? x.e.rel : y.e.rel;
	ea = x.e.abs_ > y.e.abs_ ? x.e.abs_ : y.e.abs_;
	z.e64.rel = x.e64.rel > y.e64.rel ? x.e64.rel : y.e64.rel;
	z.e64.abs_ = x.e64.abs_ > y.e64.abs_ ? x.e64.abs_ : y.e64.abs_;
	if (!x.lo && !y.lo) {
		z.hi = mk_bin(N.op[n], x.hi, y.hi);
		z.e.rel = er;
		z.e.abs_ = ea;
		row(n, &z, mx ? "max of two f32 (the choice is exact)" :
		    "min of two f32 (the choice is exact)");
		track(&z);
		return z;
	}
	L.used |= DB_MINMAX;
	dh = mk_bin(O_SUB, x.hi, y.hi);
	dl = x.lo && y.lo ? mk_bin(O_SUB, x.lo, y.lo) : x.lo ? x.lo :
	    mk_un(O_NEG, y.lo);
	d = mk_bin(O_ADD, dh, dl);
	c = mx ? mk_bin(O_LT, zero, d) : mk_bin(O_LT, d, zero);
	z.hi = mk_sel(c, x.hi, y.hi);
	z.lo = mk_sel(c, x.lo ? x.lo : zero, y.lo ? y.lo : zero);
	mag = iv_mag(&x.iv) > iv_mag(&y.iv) ? iv_mag(&x.iv) : iv_mag(&y.iv);
	z.e.rel = er + DW_PICK * (1 + er);
	z.e.abs_ = ea + lmul(DW_PICK, mag + ea);
	row(n, &z, mx ? "the larger pair, by the sign of (xh - yh) + (xl - yl)"
	    : "the smaller pair, by the sign of (xh - yh) + (xl - yl)");
	track(&z);
	return z;
}

static struct dv
lower_add(nid n, int sub)
{
	struct dv x = lower(N.a[n]), y = lower(N.b[n]), z;
	struct iv iy = sub ? iv_neg(y.iv) : y.iv;
	long double kappa;
	int kind, pre = 1, pre64;
	const char *how;
	nid sh, sl, th, tl, c, vh, vl, w, v;

	memset(&z, 0, sizeof z);
	z.iv = iv_add(x.iv, iy);
	kappa = kappa_of(&x.iv, &iy);
	comp_add(&z.e64, &x.e64, &y.e64, &x.iv, &iy, &z.iv, AK_F64, kappa,
	    &pre64);
	if (!L.mode) {
		z.hi = mk_bin(sub ? O_SUB : O_ADD, x.hi, y.hi);
		comp_add(&z.e, &x.e, &y.e, &x.iv, &iy, &z.iv, AK_SINGLE, kappa,
		    &pre);
		L.used |= DB_SINGLE;
		row(n, &z, "one f32 add");
		track(&z);
		return z;
	}
	if (!x.lo && !y.lo) {
		two_sum(x.hi, y.hi, sub, &z.hi, &z.lo);
		kind = AK_EXACT;
		how = "2Sum (exact)";
	} else if (!x.lo || !y.lo) {
		/* DWPlusFP: the single operand joins the high parts */
		const struct dv *d = x.lo ? &x : &y;
		kind = AK_FP;
		L.used |= DB_ADD_FP;
		two_sum(x.hi, y.hi, sub, &sh, &sl);
		if (x.lo)
			v = op2(O_ADD, d->lo, sl, 0);
		else
			v = op2(sub ? O_SUB : O_ADD, sl, d->lo, 0);
		fast_two_sum(sh, v, &z.hi, &z.lo);
		how = "DWPlusFP";
	} else {
		int p_s = 1;
		/* sloppy while its precondition follows from the ranges */
		{
			struct err t;
			comp_add(&t, &x.e, &y.e, &x.iv, &iy, &z.iv, AK_SLOPPY,
			    kappa, &p_s);
			if (isinf(t.rel))
				p_s = 0;
		}
		if (p_s && !(n < L.lim && L.acc[n])) {
			kind = AK_SLOPPY;
			L.used |= DB_ADD_SLOPPY;
			two_sum(x.hi, y.hi, sub, &sh, &sl);
			v = op2(sub ? O_SUB : O_ADD, x.lo, y.lo, 0);
			w = op2(O_ADD, sl, v, 0);
			fast_two_sum(sh, w, &z.hi, &z.lo);
			how = "SloppyDWPlusDW";
			if (n < L.lim) {
				L.nsloppy++;
				if (kappa > L.worstk) {
					L.worstk = kappa;
					L.worst = n;
				}
			}
		} else {
			kind = AK_ACC;
			L.used |= DB_ADD_ACC;
			two_sum(x.hi, y.hi, sub, &sh, &sl);
			two_sum(x.lo, y.lo, sub, &th, &tl);
			c = op2(O_ADD, sl, th, 0);
			fast_two_sum(sh, c, &vh, &vl);
			w = op2(O_ADD, tl, vl, 0);
			fast_two_sum(vh, w, &z.hi, &z.lo);
			how = "AccurateDWPlusDW";
		}
	}
	comp_add(&z.e, &x.e, &y.e, &x.iv, &iy, &z.iv, kind, kappa, &pre);
	if (pre == 2)
		L.pre_cases = 1;
	row(n, &z, how);
	track(&z);
	return z;
}

static struct dv
lower_mul2(nid n, struct dv x, struct dv y)
{
	struct dv z;
	long double eta;
	const char *how;
	nid ch, cl1, cl2, cl3, tl0, tl1;

	memset(&z, 0, sizeof z);
	z.iv = iv_mul(x.iv, y.iv);
	comp_mul(&z.e64, &x.e64, &y.e64, &x.iv, &y.iv, U64);
	if (!L.mode) {
		z.hi = mk_bin(O_MUL, x.hi, y.hi);
		comp_mul(&z.e, &x.e, &y.e, &x.iv, &y.iv, U);
		L.used |= DB_SINGLE;
		row(n, &z, "one f32 multiply");
		track(&z);
		return z;
	}
	track_prod(&z.iv);
	if (!x.lo && !y.lo) {
		two_prod(x.hi, y.hi, &z.hi, &z.lo);
		eta = 0;
		how = "2Prod (exact)";
	} else if (!x.lo || !y.lo) {
		const struct dv *d = x.lo ? &x : &y, *s = x.lo ? &y : &x;
		L.used |= DB_MUL_FP;
		two_prod(d->hi, s->hi, &ch, &cl1);
		cl3 = mk_fma(d->lo, s->hi, cl1);
		fast_two_sum(ch, cl3, &z.hi, &z.lo);
		eta = eta_mul_fp();
		how = "DWTimesFP3";
	} else {
		L.used |= DB_MUL_DD;
		two_prod(x.hi, y.hi, &ch, &cl1);
		tl0 = mk_bin(O_MUL, x.lo, y.lo);
		tl1 = mk_fma(x.hi, y.lo, tl0);
		cl2 = mk_fma(x.lo, y.hi, tl1);
		cl3 = mk_bin(O_ADD, cl1, cl2);
		fast_two_sum(ch, cl3, &z.hi, &z.lo);
		eta = eta_mul_dd();
		how = "DWTimesDW3";
	}
	comp_mul(&z.e, &x.e, &y.e, &x.iv, &y.iv, eta);
	row(n, &z, how);
	track(&z);
	return z;
}

static struct dv
lower_div(nid n)
{
	struct dv x = lower(N.a[n]), y = lower(N.b[n]), z;
	long double eta;
	const char *how;
	nid th, ph, pl, dh, d, tl, rh, rl, ch, cl1, cl3;

	memset(&z, 0, sizeof z);
	z.iv = iv_div(x.iv, y.iv);
	comp_div(&z.e64, &x.e64, &y.e64, &x.iv, &y.iv, U64);
	if (y.iv.top || (y.iv.lo <= 0 && y.iv.hi >= 0))
		L.div0 = 1;
	if (!L.mode) {
		z.hi = mk_bin(O_DIV, x.hi, y.hi);
		comp_div(&z.e, &x.e, &y.e, &x.iv, &y.iv, U);
		L.used |= DB_SINGLE;
		row(n, &z, "one f32 divide");
		track(&z);
		return z;
	}
	track_prod(&x.iv);	/* th y is about x */
	th = mk_bin(O_DIV, x.hi, y.hi);
	if (!y.lo) {
		L.used |= DB_DIV_FP;
		two_prod(th, y.hi, &ph, &pl);
		dh = mk_bin(O_SUB, x.hi, ph);
		d = mk_bin(O_SUB, dh, pl);
		if (x.lo)
			d = mk_bin(O_ADD, d, x.lo);
		tl = mk_bin(O_DIV, d, y.hi);
		eta = eta_div_fp(!x.lo);
		how = x.lo ? "DWDivFP3" : "FP / FP as a pair";
	} else {
		L.used |= DB_DIV_DD | DB_MUL_FP;
		two_prod(y.hi, th, &ch, &cl1);
		cl3 = mk_fma(y.lo, th, cl1);
		fast_two_sum(ch, cl3, &rh, &rl);
		dh = mk_bin(O_SUB, x.hi, rh);
		if (x.lo)
			d = mk_bin(O_ADD, dh, mk_bin(O_SUB, x.lo, rl));
		else
			d = mk_bin(O_SUB, dh, rl);
		tl = mk_bin(O_DIV, d, y.hi);
		eta = eta_div_dd();
		how = "DWDivDW2";
	}
	fast_two_sum(th, tl, &z.hi, &z.lo);
	comp_div(&z.e, &x.e, &y.e, &x.iv, &y.iv, eta);
	row(n, &z, how);
	track(&z);
	return z;
}

static struct dv
lower_sqrt(nid n)
{
	struct dv x = lower(N.a[n]), z;
	nid sh, r, sl;

	memset(&z, 0, sizeof z);
	z.iv = iv_sqrt(x.iv);
	comp_sqrt(&z.e64, &x.e64, &x.iv, U64);
	if (x.iv.top || x.iv.lo <= 0)
		L.sqrt0 = 1;
	if (!L.mode) {
		z.hi = mk_un(O_SQRT, x.hi);
		comp_sqrt(&z.e, &x.e, &x.iv, U);
		L.used |= DB_SINGLE;
		row(n, &z, "one f32 sqrt");
		track(&z);
		return z;
	}
	L.used |= DB_SQRT;
	L.nprod++;
	track_prod(&x.iv);
	sh = mk_un(O_SQRT, x.hi);
	r = mk_fma(mk_un(O_NEG, sh), sh, x.hi);
	mark_eft(r);
	if (x.lo)
		r = mk_bin(O_ADD, x.lo, r);
	sl = mk_bin(O_DIV, r, mk_bin(O_ADD, sh, sh));
	fast_two_sum(sh, sl, &z.hi, &z.lo);
	comp_sqrt(&z.e, &x.e, &x.iv, eta_sqrt());
	row(n, &z, "Newton sqrt");
	track(&z);
	return z;
}

/* fma(a, b, c): one rounding in f64; here a product, then its sum */
static struct dv
lower_fma(nid n)
{
	struct dv x = lower(N.a[n]), y, c, p, s;
	struct iv ip;
	struct err pe;
	long double kappa;
	int kind, pre = 1, p64;
	nid sh, sl, th, tl, cc, vh, vl, w, v;

	y = N.b[n] == N.a[n] ? x : lower(N.b[n]);
	c = lower(N.c[n]);
	memset(&s, 0, sizeof s);
	ip = iv_mul(x.iv, y.iv);
	s.iv = iv_add(ip, c.iv);
	kappa = kappa_of(&ip, &c.iv);
	/* the f64 fma: the product is exact there */
	comp_mul(&pe, &x.e64, &y.e64, &x.iv, &y.iv, 0);
	comp_add(&s.e64, &pe, &c.e64, &ip, &c.iv, &s.iv, AK_F64, kappa, &p64);
	if (!L.mode) {
		s.hi = mk_fma(x.hi, y.hi, c.hi);
		comp_mul(&pe, &x.e, &y.e, &x.iv, &y.iv, 0);
		comp_add(&s.e, &pe, &c.e, &ip, &c.iv, &s.iv, AK_SINGLE, kappa,
		    &pre);
		L.used |= DB_SINGLE;
		row(n, &s, "one f32 fma");
		track(&s);
		return s;
	}
	p = lower_mul2(n, x, y);
	if (!p.lo && !c.lo) {
		two_sum(p.hi, c.hi, 0, &s.hi, &s.lo);
		kind = AK_EXACT;
	} else if (!p.lo || !c.lo) {
		const struct dv *d = p.lo ? &p : &c;
		kind = AK_FP;
		L.used |= DB_ADD_FP;
		two_sum(p.hi, c.hi, 0, &sh, &sl);
		v = op2(O_ADD, d->lo, sl, 0);
		fast_two_sum(sh, v, &s.hi, &s.lo);
	} else {
		kind = AK_ACC;
		L.used |= DB_ADD_ACC;
		two_sum(p.hi, c.hi, 0, &sh, &sl);
		two_sum(p.lo, c.lo, 0, &th, &tl);
		cc = op2(O_ADD, sl, th, 0);
		fast_two_sum(sh, cc, &vh, &vl);
		w = op2(O_ADD, tl, vl, 0);
		fast_two_sum(vh, w, &s.hi, &s.lo);
	}
	comp_add(&s.e, &p.e, &c.e, &p.iv, &c.iv, &s.iv, kind, kappa, &pre);
	if (pre == 2)
		L.pre_cases = 1;
	row(n, &s, "the product's pair, then its sum");
	track(&s);
	return s;
}

/* an integer converted to f64 */
static struct dv
lower_int(nid n, nid a)
{
	struct dv z;
	struct iv iv = leaf_iv(L.f, a, 0);
	ty_t t = N.ty[a];

	memset(&z, 0, sizeof z);
	z.iv = iv;
	add_leaf(a);
	if (!iv.top && iv_mag(&iv) <= 0x1p24L) {
		z.hi = mk_cast(TY_F32, a);	/* exact: at most 24 bits */
		return z;
	}
	if (TY_SIZED(t) && TYW(t) <= 32 && L.mode) {
		/* the low 12 bits apart: both parts exact in f32 */
		nid lo = mk_bin(O_AND, a, mk_const_i(t, 0xfff));
		nid hi = mk_bin(O_SUB, a, lo);
		fast_two_sum(mk_cast(TY_F32, hi), mk_cast(TY_F32, lo), &z.hi,
		    &z.lo);
		return z;
	}
	if (!L.mode && !iv.top) {
		z.hi = mk_cast(TY_F32, a);
		z.e.rel = U;
		z.e.abs_ = lmul(U, iv_mag(&iv));
		return z;
	}
	/* wider: converted in f64, then split */
	L.opaque++;
	return split64(n, iv);
}

static struct dv
lower1(nid n)
{
	struct dv z, x, y;
	ty_t t = N.ty[n];
	double k;

	memset(&z, 0, sizeof z);
	if (t != TY_F64) {
		/* an f32 value: itself */
		if (t == TY_F32) {
			z.hi = n;
			z.iv = leaf_iv(L.f, n, 0);
			add_leaf(n);
			return z;
		}
		snprintf(L.unsup, sizeof L.unsup, "%s is %s, not f64",
		    node_str(n), ty_str(t));
		z.hi = mk_cast(TY_F32, n);
		z.iv = iv_top();
		return z;
	}
	switch (N.op[n]) {
	case O_CONST: {
		double c = N.kr[n], hi = (double)(float)c, lo;
		if (isinf(hi) || isnan(c))
			break;
		lo = (double)(float)(c - hi);
		z.iv = iv_point(c);
		z.hi = f32k(L.mode ? hi : c);
		if (L.mode) {
			double r = (c - hi) - lo;
			if (lo != 0)
				z.lo = f32k(lo);
			z.e.abs_ = fabsl((long double)r);
		} else
			z.e.abs_ = fabsl((long double)c - (long double)hi);
		z.e.rel = c != 0 ? z.e.abs_ / fabsl((long double)c) : 0;
		return z;
	}
	case O_VAR:
		if (L.accv && (int)N.ki[n] == L.accv - 1) {
			/* a loop's sum: the pair the passes carry is its value */
			z.hi = L.acch;
			z.lo = L.mode ? L.accl : 0;
			z.iv = L.acciv;
			track(&z);
			return z;
		}
		L.splitvar++;
		return split64(n, leaf_iv(L.f, n, 0));
	case O_CAST:
		if (N.ty[N.a[n]] == TY_F32) {
			z.hi = N.a[n];
			z.iv = leaf_iv(L.f, N.a[n], 0);
			add_leaf(N.a[n]);
			return z;
		}
		if (TY_ISINT(N.ty[N.a[n]]))
			return lower_int(n, N.a[n]);
		break;
	case O_NEG:
		x = lower(N.a[n]);
		z = x;
		z.hi = mk_un(O_NEG, x.hi);
		z.lo = x.lo ? mk_un(O_NEG, x.lo) : 0;
		z.iv = iv_neg(x.iv);
		return z;
	case O_ABS:
		x = lower(N.a[n]);
		z = x;
		z.iv = iv_abs(x.iv);
		if (!x.iv.top && x.iv.lo >= 0)
			return z;
		if (!x.iv.top && x.iv.hi <= 0) {
			z.hi = mk_un(O_NEG, x.hi);
			z.lo = x.lo ? mk_un(O_NEG, x.lo) : 0;
			return z;
		}
		z.hi = mk_un(O_ABS, x.hi);
		if (x.lo)
			z.lo = mk_sel(mk_bin(O_LT, x.hi, f32k(0)),
			    mk_un(O_NEG, x.lo), x.lo);
		return z;
	case O_ADD:
		return lower_add(n, 0);
	case O_SUB:
		return lower_add(n, 1);
	case O_MUL:
		if (pow2_const(N.a[n], &k)) {
			x = lower(N.b[n]);
			z = scale(&x, k);
			row(n, &z, "scaling by a power of two (exact)");
			track(&z);
			return z;
		}
		x = lower(N.a[n]);
		y = N.b[n] == N.a[n] ? x : lower(N.b[n]);
		return lower_mul2(n, x, y);
	case O_DIV:
		if (pow2_const(N.b[n], &k)) {
			x = lower(N.a[n]);
			z = scale(&x, 1 / k);
			row(n, &z, "scaling by a power of two (exact)");
			track(&z);
			return z;
		}
		return lower_div(n);
	case O_POW:
		if (N.op[N.b[n]] == O_CONST && N.kr[N.b[n]] == 2) {
			x = lower(N.a[n]);
			return lower_mul2(n, x, x);
		}
		break;
	case O_SQRT:
		return lower_sqrt(n);
	case O_FMA:
		return lower_fma(n);
	case O_MIN:
	case O_MAX:
		if (N.b[n] && TY_FP(N.ty[N.a[n]]) && TY_FP(N.ty[N.b[n]]))
			return lower_minmax(n);
		break;
	}
	/* no double-word form: the value stays in f64 and is split */
	L.opaque++;
	return split64(n, iv_top());
}

static struct dv
lower(nid n)
{
	struct dv z;

	if (n < L.lim && L.done[n])
		return L.memo[n];
	z = lower1(n);
	if (n < L.lim) {
		L.memo[n] = z;
		L.done[n] = 1;
	}
	return z;
}

/* ---- pricing ---- */

/* the price under the model, its fusions covering what they can (an
 * operation of an error-free transformation is read more than once, so
 * no fusion absorbs one); *nops: the operations */
static double
dag_cost(const struct model *m, nid root, int *nops)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc((size_t)N.n * sizeof *st), n;
	uint32_t sp = 0;
	int cls, cnt = 0;

	st[sp++] = root;
	mark[root] = 1;
	while (sp) {
		n = st[--sp];
		(void)node_cost(m, n, &cls);
		if (cls >= 0)
			cnt++;
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF
		    || N.op[n] == O_TABLE)
			continue;
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
	}
	free(mark);
	free(st);
	if (nops)
		*nops = cnt;
	return (double)fuse_expr_cost(m, root);
}

/* ---- one candidate ---- */

struct cand {
	int mode;
	struct dv r;
	nid body;
	nid body2;		/* a pair's low word, when both are the result */
	double cost;
	int nops;
	long double extra;	/* f32 sink: ulp beyond its own rounding */
	int meets;
	unsigned used;
	long double maxmag;
	int mag_unknown, uflow, nprod, div0, sqrt0, pre_cases, opaque,
	    splitvar, nsloppy;
	char unsup[120];
	char *table;
	uint8_t *eft;
	uint32_t neft;
	nid leaf[MAXLEAF];
	int nleaf;
};

static void
lower_reset(const struct func *f, int mode, uint32_t lim, uint8_t *acc)
{
	free(L.memo);
	free(L.done);
	free(L.eft);
	free(L.table.p);
	memset(&L, 0, sizeof L);
	L.f = f;
	L.mode = mode;
	L.lim = lim;
	L.memo = xcalloc(lim, sizeof *L.memo);
	L.done = xcalloc(lim, 1);
	L.acc = acc;
	L.worstk = 0;
}

/* the f64 value the function's result is made of, and its sink */
static nid
island_root(const struct func *f, int *sink32)
{
	nid b = f->body;

	*sink32 = 0;
	if (N.ty[b] == TY_F32 && N.op[b] == O_CAST && N.ty[N.a[b]] == TY_F64) {
		*sink32 = 1;
		return N.a[b];
	}
	if (N.ty[b] == TY_F64)
		return b;
	return 0;
}

static int
meets(const struct dwreq *rq, int sink32, const struct dv *r,
    long double *extra)
{
	long double mig = iv_mig(&r->iv);

	*extra = LINF;
	if (!isinf(r->e.rel))
		*extra = r->e.rel / U;
	else if (!isinf(r->e.abs_) && mig > 0) {
		/* ulp(v) >= u |v| */
		*extra = r->e.abs_ / (U * mig);
	}
	switch (rq->kind) {
	case DW_TOL:
		return r->e.rel <= (long double)rq->v;
	case DW_ATOL:
		return r->e.abs_ <= (long double)rq->v;
	case DW_ULP:
		return sink32 && *extra <= (long double)rq->v;
	}
	/* an f64 result with no tolerance: the exact value only */
	if (!sink32)
		return r->e.rel == 0 || r->e.abs_ == 0;
	/* a final f32: ten guard bits past its own rounding */
	return *extra <= 0x1p-10L;
}

static int f64_bits(const struct func *f, nid n, int depth);

/*
 * A pair recombined into an f64 result: hi + lo needs more than 53 bits
 * when lo is far below hi (hi = 1, lo = 2^-40 (1 + 2^-23)), so the f64 add
 * rounds, by u64 of the sum at most; not when the pair is the exact value
 * and that fits in 53 bits
 */
static struct err
recombined(const struct dv *r)
{
	struct err e = r->e;
	long double mx = iv_mag(&r->iv);

	if (!isinf(e.rel))
		e.rel += U64 * (1 + e.rel);
	if (!isinf(e.abs_))
		e.abs_ += U64 * (mx + e.abs_);
	return e;
}

static void
run_cand(const struct func *f, const struct model *m, const struct dwreq *rq,
    int mode, uint32_t lim, struct cand *c)
{
	uint8_t *acc = xcalloc(lim, 1);
	int sink32, guard;
	nid root = island_root(f, &sink32);

	memset(c, 0, sizeof *c);
	c->mode = mode;
	for (guard = 0; guard < 64; guard++) {
		lower_reset(f, mode, lim, acc);
		c->r = lower(root);
		if (!sink32 && c->r.lo && !((c->r.e.rel == 0 || c->r.e.abs_ == 0) &&
		    f64_bits(f, root, 0) >= 0 && f64_bits(f, root, 0) <= 53))
			c->r.e = recombined(&c->r);
		c->meets = meets(rq, sink32, &c->r, &c->extra);
		/* a sloppy add with cancellation: the accurate one */
		if (c->meets || !L.nsloppy || !L.worst || L.worstk <= 1.001L)
			break;
		acc[L.worst] = 1;
	}
	if (sink32)
		c->body = c->r.hi;
	else if (c->r.lo)
		c->body = mk_bin(O_ADD, mk_cast(TY_F64, c->r.hi),
		    mk_cast(TY_F64, c->r.lo));
	else
		c->body = mk_cast(TY_F64, c->r.hi);
	c->cost = dag_cost(m, c->body, &c->nops);
	if (!L.mag_unknown && L.maxmag >= 0x1p126L) {
		c->meets = 0;	/* an f32 would overflow where the f64 does not */
		snprintf(L.unsup, sizeof L.unsup, "a value reaches %.3Lg: an "
		    "f32 overflows there", L.maxmag);
	}
	c->used = L.used;
	c->maxmag = L.maxmag;
	c->mag_unknown = L.mag_unknown;
	c->uflow = L.uflow;
	c->nprod = L.nprod;
	c->div0 = L.div0;
	c->sqrt0 = L.sqrt0;
	c->pre_cases = L.pre_cases;
	c->opaque = L.opaque;
	c->splitvar = L.splitvar;
	c->nsloppy = L.nsloppy;
	snprintf(c->unsup, sizeof c->unsup, "%s", L.unsup);
	c->table = L.table.p ? strdup(L.table.p) : strdup("");
	c->neft = N.n;
	c->eft = xcalloc(c->neft, 1);
	if (L.eft)
		memcpy(c->eft, L.eft, L.eftcap < c->neft ? L.eftcap : c->neft);
	memcpy(c->leaf, L.leaf, sizeof c->leaf);
	c->nleaf = L.nleaf;
	free(acc);
	L.acc = NULL;
}

static void
cand_free(struct cand *c)
{
	free(c->table);
	free(c->eft);
}

/* ---- the report ---- */

static const char *
err_text(const struct err *e, long double unit, const char *uname, char *b,
    size_t n)
{
	if (!isinf(e->rel))
		snprintf(b, n, "%.4Lg (%.4Lg %s) relative", e->rel, e->rel / unit,
		    uname);
	else if (!isinf(e->abs_))
		snprintf(b, n, "%.4Lg absolute (no relative bound: a sum may "
		    "cancel)", e->abs_);
	else
		snprintf(b, n, "unbounded (give the parameters ranges)");
	return b;
}

static void
derivations(struct sbuf *o, unsigned used)
{
	sb_puts(o, "  derivation (u = 2^-24; RN rounds to nearest; a pair x~ = "
	    "xh + xl has |xl| <= u |xh|):\n");
	if (used & DB_MINMAX)
		sb_puts(o, "    min, max(x~, y~): d = RN(RN(xh - yh) + RN(xl - yl)), "
		    "the pair with d < 0 (max: 0 < d), else the other.  With\n"
		    "      xh, yh within a factor of two xh - yh is exact "
		    "(Sterbenz) and d has the sign of x~ - y~ unless\n"
		    "      |x~ - y~| <= u^2 (|xh| + |yh|), where either pair is "
		    "that close to the other; farther apart the high\n"
		    "      words decide.  The choice adds at most 3.000 u^2 to "
		    "the larger of the operands' bounds\n");
	if (used & DB_SPLIT)
		sb_puts(o, "    an f64 value v as a pair: hi = RN32(v), lo = "
		    "RN32(v - hi); |v - hi| <= u |v| and its rounding\n"
		    "      loses at most u of that: |v - hi - lo| <= u^2 |v|\n");
	if (used & DB_2SUM)
		sb_puts(o, "    2Sum(a, b): s = RN(a + b), a' = RN(s - b), b' = "
		    "RN(s - a'), t = RN((a - a') + (b - b')):\n"
		    "      s + t = a + b exactly (Knuth), |t| <= u |s|\n");
	if (used & DB_F2SUM)
		sb_puts(o, "    Fast2Sum(a, b): s = RN(a + b), t = RN(b - RN(s - "
		    "a)): s + t = a + b exactly when |a| >= |b| (Dekker)\n");
	if (used & DB_2PROD)
		sb_puts(o, "    2Prod(a, b): p = RN(a b), e = fma(a, b, -p): p + e "
		    "= a b exactly while |a b| >= 2^-102\n"
		    "      (a b - p is a multiple of ulp(a) ulp(b) below ulp(p): "
		    "representable), |e| <= u |a b|\n");
	if (used & DB_ADD_FP) {
		int pre;
		sb_printf(o, "    DWPlusFP(x~, y): (sh, sl) = 2Sum(xh, y); v = "
		    "RN(xl + sl); (zh, zl) = Fast2Sum(sh, v):\n"
		    "      zh + zl = x~ + y + d (xl + sl), |d| <= u: the error is "
		    "at most u (u |xh| + u |sh|),\n"
		    "      %.3Lf u^2 |x~ + y| when no cancellation (kappa = 1), "
		    "(1 + kappa) u^2 in general;\n"
		    "      whatever cancels, by cases on xh and y: of opposite "
		    "signs with |y| in [|xh|/2, 2|xh|],\n"
		    "      xh + y is a float (Sterbenz), so sl = 0, v = xl and the "
		    "sum is exact; otherwise\n"
		    "      |xh + y| > |xh|/2 and the error is below 3 u^2 / (1 - "
		    "2u) = %.3Lf u^2 |x~ + y| (the paper:\n"
		    "      2 u^2); Fast2Sum's precondition holds in both\n",
		    add_local(AK_FP, 1, 1 / (1 - U), &pre) / (U * U),
		    eta_add_fp_split() / (U * U));
	}
	if (used & DB_ADD_SLOPPY) {
		int pre;
		sb_printf(o, "    SloppyDWPlusDW(x~, y~): (sh, sl) = 2Sum(xh, yh); "
		    "v = RN(xl + yl); w = RN(sl + v);\n"
		    "      (zh, zl) = Fast2Sum(sh, w): zh + zl = x~ + y~ + d1 (xl "
		    "+ yl) + d2 (sl + v): at most\n"
		    "      u (u A) + u (u |sh| + (1 + u) u A), A = |xh| + |yh|: "
		    "%.3Lf u^2 |x~ + y~| at kappa = 1,\n"
		    "      (1 + 2 kappa) u^2 in general, kappa = (|x| + |y|) / |x "
		    "+ y| the largest on the ranges\n",
		    add_local(AK_SLOPPY, 1, 1 / (1 - U), &pre) / (U * U));
	}
	if (used & DB_ADD_ACC) {
		int pre;
		sb_printf(o, "    AccurateDWPlusDW(x~, y~): (sh, sl) = 2Sum(xh, "
		    "yh); (th, tl) = 2Sum(xl, yl); c = RN(sl + th);\n"
		    "      (vh, vl) = Fast2Sum(sh, c); w = RN(tl + vl); (zh, zl) "
		    "= Fast2Sum(vh, w):\n"
		    "      zh + zl = x~ + y~ + d1 (sl + th) + d2 (tl + vl): %.3Lf "
		    "u^2 |x~ + y~| at kappa = 1,\n"
		    "      (2 + kappa) u^2 in general; whatever cancels, with |xh| "
		    ">= |yh|: of opposite signs with\n"
		    "      |yh| >= |xh|/2, xh + yh is a float (Sterbenz), a "
		    "multiple of ulp(xh)/2 as is th's ulp, so\n"
		    "      sl = 0 and xh + yh + th is a float, the result exact, "
		    "or 2^p ulp(th) or more, the error\n"
		    "      then below 1.5 u^2; otherwise |xh| + |yh| <= 3 |xh + "
		    "yh| and the error is below %.3Lf u^2\n"
		    "      (the paper: 3 u^2 / (1 - 4u)); both Fast2Sums are exact "
		    "in each case\n",
		    add_local(AK_ACC, 1, 1 / (1 - U), &pre) / (U * U),
		    eta_add_acc_split() / (U * U));
	}
	if (used & DB_MUL_FP)
		sb_printf(o, "    DWTimesFP3(x~, y): (ch, cl1) = 2Prod(xh, y); cl3 "
		    "= fma(xl, y, cl1); (zh, zl) = Fast2Sum(ch, cl3):\n"
		    "      zh + zl = x~ y + d (xl y + cl1), |xl y| and |cl1| <= u "
		    "|xh y|: %.3Lf u^2 |x~ y|\n", eta_mul_fp() / (U * U));
	if (used & DB_MUL_DD)
		sb_printf(o, "    DWTimesDW3(x~, y~): (ch, cl1) = 2Prod(xh, yh); "
		    "tl0 = RN(xl yl); tl1 = fma(xh, yl, tl0);\n"
		    "      cl2 = fma(xl, yh, tl1); cl3 = RN(cl1 + cl2); (zh, zl) "
		    "= Fast2Sum(ch, cl3): the four\n"
		    "      roundings err by u^3, u^2, 2 u^2 and 3 u^2 times |xh "
		    "yh|: %.3Lf u^2 |x~ y~| (the paper: 5 u^2)\n",
		    eta_mul_dd() / (U * U));
	if (used & DB_DIV_FP)
		sb_printf(o, "    DWDivFP3(x~, y): th = RN(xh / y); (ph, pl) = "
		    "2Prod(th, y); d = RN(RN(RN(xh - ph) - pl) + xl);\n"
		    "      tl = RN(d / y); (zh, zl) = Fast2Sum(th, tl): ph is "
		    "within (1 +- u)^2 of xh, so xh - ph\n"
		    "      is exact (Sterbenz), and xh - th y, a multiple of "
		    "ulp(th) ulp(y) at most ulp(th) |y| / 2,\n"
		    "      is a float, so less pl it is exact too; the residual "
		    "x~ - y th, at most 2u |xh|, rounds\n"
		    "      once with xl and once divided: %.3Lf u^2 |x~ / y| "
		    "(%.3Lf u^2 when xl = 0; the paper: 3 u^2)\n",
		    eta_div_fp(0) / (U * U), eta_div_fp(1) / (U * U));
	if (used & DB_DIV_DD)
		sb_printf(o, "    DWDivDW2(x~, y~): th = RN(xh / yh); r~ = "
		    "DWTimesFP3(y~, th); d = RN(RN(xh - rh) + RN(xl - rl));\n"
		    "      tl = RN(d / yh); (zh, zl) = Fast2Sum(th, tl): the "
		    "residual x~ - y~ th is at most 3u |xh|;\n"
		    "      rh is within 1 +- 3u of xh, so xh - rh is exact "
		    "(Sterbenz); r~, the two other roundings,\n"
		    "      the division and yh for y~ give %.3Lf u^2 |x~ / y~| "
		    "(the paper: 15 u^2)\n",
		    eta_div_dd() / (U * U));
	if (used & DB_SQRT)
		sb_printf(o, "    sqrt(x~): sh = RN(sqrt(xh)); r = fma(-sh, sh, "
		    "xh) (exact); sl = RN(RN(xl + r) / (2 sh));\n"
		    "      (zh, zl) = Fast2Sum(sh, sl): one Newton step, sqrt(x~) "
		    "= sh + (x~ - sh^2) / (sh + sqrt(x~));\n"
		    "      2 sh for the denominator and the roundings give %.3Lf "
		    "u^2 sqrt(x~)\n", eta_sqrt() / (U * U));
	sb_puts(o, "    composition: a product's errors multiply (1 + ex)(1 + "
	    "ey)(1 + e) - 1, a quotient's (1 + ex)(1 + e) / (1 - ey) - 1,\n"
	    "      a sum's are kappa max(ex, ey) + e (1 + kappa max(ex, ey)), "
	    "a square root halves its operand's\n");
}

static int cand_leaf(const struct cand *c, nid n);

/* the form as a program: one line per operation, ! where NoContraction */
static void
listing(struct sbuf *o, const struct cand *c)
{
	uint8_t *mark = xcalloc(N.n, 1);
	uint32_t *name = xcalloc(N.n, sizeof *name);
	nid *st = xmalloc((size_t)N.n * sizeof *st), n;
	uint32_t sp = 0, g, k = 0;
	int neft = 0;

	/* the form's own operations: down to the source's values */
	st[sp++] = c->body;
	mark[c->body] = 1;
	if (c->body2 && !mark[c->body2]) {
		st[sp++] = c->body2;
		mark[c->body2] = 1;
	}
	while (sp) {
		n = st[--sp];
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || cand_leaf(c, n)) {
			mark[n] = 0;
			continue;
		}
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
	}
	for (g = 1; g < N.n; g++)
		if (mark[g])
			name[g] = ++k;
#define OPND(x) do { \
	if (name[x]) \
		sb_printf(o, "t%u", name[x]); \
	else if (N.op[x] == O_CONST || N.op[x] == O_VAR) \
		sb_puts(o, node_str(x)); \
	else \
		sb_printf(o, "(%s)", node_str(x)); \
} while (0)
	for (g = 1; g < N.n; g++) {
		const char *sym = NULL;
		size_t at;
		int eft = g < c->neft && c->eft[g];
		if (!name[g])
			continue;
		at = o->n;
		sb_printf(o, "    t%u = ", name[g]);
		switch (N.op[g]) {
		case O_ADD: sym = " + "; break;
		case O_SUB: sym = " - "; break;
		case O_MUL: sym = " * "; break;
		case O_DIV: sym = " / "; break;
		case O_LT: sym = " < "; break;
		case O_LE: sym = " <= "; break;
		case O_GT: sym = " > "; break;
		case O_GE: sym = " >= "; break;
		case O_EQ: sym = " == "; break;
		case O_NE: sym = " != "; break;
		case O_LAND: sym = " && "; break;
		case O_LOR: sym = " || "; break;
		case O_AND: sym = " & "; break;
		}
		if (sym) {
			OPND(N.a[g]);
			sb_puts(o, sym);
			OPND(N.b[g]);
		} else if (N.op[g] == O_FMA) {
			sb_puts(o, "fma(");
			OPND(N.a[g]);
			sb_puts(o, ", ");
			OPND(N.b[g]);
			sb_puts(o, ", ");
			OPND(N.c[g]);
			sb_putc(o, ')');
		} else if (N.op[g] == O_NEG) {
			sb_putc(o, '-');
			OPND(N.a[g]);
		} else if (N.op[g] == O_SQRT || N.op[g] == O_ABS) {
			sb_puts(o, N.op[g] == O_SQRT ? "sqrt(" : "abs(");
			OPND(N.a[g]);
			sb_putc(o, ')');
		} else if (N.op[g] == O_CAST) {
			sb_printf(o, "%s(", ty_str(N.ty[g]));
			OPND(N.a[g]);
			sb_putc(o, ')');
		} else if (N.op[g] == O_SEL) {
			OPND(N.a[g]);
			sb_puts(o, " ? ");
			OPND(N.b[g]);
			sb_puts(o, " : ");
			OPND(N.c[g]);
		} else
			sb_puts(o, node_str(g));
		if (eft) {
			size_t len = o->n - at;
			while (len++ < 44)
				sb_putc(o, ' ');
			sb_puts(o, " !");
			neft++;
		}
		sb_putc(o, '\n');
	}
#undef OPND
	sb_puts(o, c->body2 ? "    result: the pair (" : "    result: ");
	if (name[c->body])
		sb_printf(o, "t%u", name[c->body]);
	else
		sb_puts(o, node_str(c->body));
	if (c->body2) {
		sb_puts(o, ", ");
		if (name[c->body2])
			sb_printf(o, "t%u", name[c->body2]);
		else
			sb_puts(o, node_str(c->body2));
		sb_putc(o, ')');
	}
	sb_putc(o, '\n');
	if (neft)
		sb_printf(o, "    ! %d operations of the error-free "
		    "transformations: exact only as written, so they are\n"
		    "      NoContraction in SPIR-V, `precise` in GLSL (no fused "
		    "multiply-add, no reassociation)\n", neft);
	free(mark);
	free(name);
	free(st);
}

static void
conditions(struct sbuf *o, const struct cand *c)
{
	sb_puts(o, "  conditions:\n");
	sb_puts(o, "    finite inputs (an infinity or a NaN gives NaN where "
	    "f64 would carry it)\n");
	if (c->mag_unknown)
		sb_puts(o, "    no overflow: every value below 2^127 in "
		    "magnitude: not derived (a parameter has no range), to be\n"
		    "      ensured by the caller\n");
	else
		sb_printf(o, "    no overflow: the largest value is %.4Lg, below "
		    "2^127 (derived from the ranges)%s\n", c->maxmag,
		    c->maxmag < 0x1p126L ? "" : ": NOT met");
	if (c->nprod) {
		if (c->uflow)
			sb_printf(o, "    no underflow: %d error-free product%s, "
			    "exact while the product is 0 or at least 2^-102 in\n"
			    "      magnitude; below that each adds at most 2^-149 "
			    "(absolute): not excluded by the ranges\n", c->nprod,
			    c->nprod == 1 ? "" : "s");
		else
			sb_printf(o, "    no underflow: the %d error-free product%s "
			    "stay%s above 2^-102 (derived from the ranges)\n",
			    c->nprod, c->nprod == 1 ? "" : "s",
			    c->nprod == 1 ? "s" : "");
	}
	if (c->div0)
		sb_puts(o, "    a divisor's range holds 0: the quotient is "
		    "undefined there, in f64 too\n");
	if (c->sqrt0)
		sb_puts(o, "    a square root's operand must be positive: at 0 "
		    "the Newton step divides 0 by 0 (f64 gives 0)\n");
	if (c->mode && c->pre_cases)
		sb_puts(o, "    Fast2Sum's precondition |a| >= |b|: derived in "
		    "each use from the magnitudes above, and in an addition\n"
		    "      whose cancellation the ranges do not bound, from the "
		    "cases of its derivation\n");
	else if (c->mode)
		sb_puts(o, "    Fast2Sum's precondition |a| >= |b|: derived in "
		    "each use from the magnitudes above\n");
}

/* ---- a sampled check against the exact value (long double) ---- */

static struct {
	const struct plan *p;
	uint32_t i;
	long double *v;
	uint32_t *stamp, cur, lim;
	const struct func *f;
} X;

static long double
leafval(nid n)
{
	uint32_t loc = plan_loc(X.p, n);

	if (loc == NOTOK)
		return (long double)NAN;
	return TY_FP(N.ty[n]) ? (long double)COLR(X.p, loc)[X.i] :
	    (long double)COLI(X.p, loc)[X.i];
}

static int
cand_leaf(const struct cand *c, nid n)
{
	int i;

	for (i = 0; i < c->nleaf; i++)
		if (c->leaf[i] == n)
			return 1;
	return 0;
}

static int
is_leaf(const struct cand *c, nid n)
{
	int i;

	for (i = 0; i < c->nleaf; i++)
		if (c->leaf[i] == n)
			return 1;
	return 0;
}

static const struct cand *xc;

static long double
exact(nid n)
{
	long double r, a, b;

	if (n < X.lim && X.stamp[n] == X.cur)
		return X.v[n];
	if (is_leaf(xc, n))
		r = leafval(n);
	else
		switch (N.op[n]) {
		case O_CONST:
			r = TY_FP(N.ty[n]) ? (long double)N.kr[n] :
			    (long double)N.ki[n];
			break;
		case O_CAST:
			r = is_leaf(xc, N.a[n]) ? leafval(N.a[n]) : exact(N.a[n]);
			break;
		case O_NEG:
			r = -exact(N.a[n]);
			break;
		case O_ABS:
			r = fabsl(exact(N.a[n]));
			break;
		case O_ADD:
			a = exact(N.a[n]);
			r = a + exact(N.b[n]);
			break;
		case O_SUB:
			a = exact(N.a[n]);
			r = a - exact(N.b[n]);
			break;
		case O_MUL:
			a = exact(N.a[n]);
			r = a * exact(N.b[n]);
			break;
		case O_DIV:
			a = exact(N.a[n]);
			r = a / exact(N.b[n]);
			break;
		case O_POW:
			a = exact(N.a[n]);
			r = a * a;
			break;
		case O_SQRT:
			r = sqrtl(exact(N.a[n]));
			break;
		case O_FMA:
			a = exact(N.a[n]);
			b = exact(N.b[n]);
			r = a * b + exact(N.c[n]);
			break;
		case O_MIN:
			a = exact(N.a[n]);
			b = exact(N.b[n]);
			r = a < b ? a : b;
			break;
		case O_MAX:
			a = exact(N.a[n]);
			b = exact(N.b[n]);
			r = a > b ? a : b;
			break;
		default:
			r = leafval(n);
		}
	if (n < X.lim) {
		X.v[n] = r;
		X.stamp[n] = X.cur;
	}
	return r;
}

#define DW_SAMPLE 4096

static void
sampled_check(struct sbuf *o, const struct func *f, const struct cand *c,
    uint32_t lim)
{
	struct plan p;
	struct diter d;
	nid roots[MAXLEAF + 4], root;
	int nr = 0, i, sink32;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint8_t ok[CH];
	uint32_t np, k, lhi, llo = NOTOK, l64, wloc;
	long double worst = 0, worst64 = 0;
	long n = 0;

	root = island_root(f, &sink32);
	roots[nr++] = c->r.hi;
	if (c->r.lo)
		roots[nr++] = c->r.lo;
	roots[nr++] = root;
	for (i = 0; i < c->nleaf; i++)
		roots[nr++] = c->leaf[i];
	roots[nr] = f->where;
	plan_build(&p, roots, (uint32_t)nr + 1, 0);
	wloc = f->where ? plan_loc(&p, f->where) : NOTOK;
	lhi = plan_loc(&p, c->r.hi);
	if (c->r.lo)
		llo = plan_loc(&p, c->r.lo);
	l64 = plan_loc(&p, root);
	diter_init(&d, f, DW_SAMPLE, 65, DW_SAMPLE);
	X.p = &p;
	X.lim = lim;
	X.v = xcalloc(lim, sizeof *X.v);
	X.stamp = xcalloc(lim, sizeof *X.stamp);
	X.cur = 0;
	X.f = f;
	xc = c;
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wloc, np, ok);
		for (k = 0; k < np; k++) {
			long double ex, got, e;
			double v64;
			if (!ok[k])
				continue;
			X.i = k;
			X.cur++;
			ex = exact(root);
			got = (long double)COLR(&p, lhi)[k];
			if (llo != NOTOK)
				got += (long double)COLR(&p, llo)[k];
			v64 = COLR(&p, l64)[k];
			if (isnan(ex) || isinf(ex) || ex == 0 || isnan(got) ||
			    isinf(got) || fabsl(ex) > 0x1p120L ||
			    fabsl(ex) < 0x1p-90L)
				continue;
			n++;
			e = fabsl(got - ex) / fabsl(ex);
			if (e > worst)
				worst = e;
			e = fabsl((long double)v64 - ex) / fabsl(ex);
			if (e > worst64)
				worst64 = e;
		}
	}
	sb_printf(o, "  check (sampled, not a proof): %ld points of the domain, "
	    "the largest error against the exact value\n"
	    "    (long double) %.2Lg %s", n, worst / (c->mode ? U * U : U),
	    c->mode ? "u^2" : "u");
	if (!isinf(c->r.e.rel))
		sb_printf(o, " (the bound: %.4Lg)", c->r.e.rel /
		    (c->mode ? U * U : U));
	sb_printf(o, "; the f64 program's %.2Lg u64\n", worst64 / U64);
	rep_int("check_points", n);
	rep_num("check_max_rel", (double)worst);
	plan_free(&p);
	free(in_i);
	free(in_r);
	free(X.v);
	free(X.stamp);
	memset(&X, 0, sizeof X);
}

/* ---- the entry ---- */

static int
has_f64(nid root)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc((size_t)N.n * sizeof *st), n;
	uint32_t sp = 0;
	int found = 0;

	st[sp++] = root;
	mark[root] = 1;
	while (sp && !found) {
		n = st[--sp];
		if (N.ty[n] == TY_F64 && N.op[n] != O_CONST && N.op[n] != O_VAR
		    && N.op[n] != O_CAST)
			found = 1;
		if (N.op[n] == O_CONST || N.op[n] == O_VAR ||
		    N.op[n] == O_UNDEF || N.op[n] == O_TABLE)
			continue;
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
	}
	free(mark);
	free(st);
	return found;
}

void
dword_res_free(struct dwres *r)
{
	free(r->eft);
	r->eft = NULL;
}

/* ---- a comparison of f64 values ---- */

/*
 * The result is a bool, not a value with an error to spend: the f32
 * forms of the two sides replace the f64 arithmetic only where the
 * answer cannot change.  Either both sides are exact, in f64 and in the
 * form (products and sums of f32 values that the ranges keep within the
 * formats), so the f64 comparison is the exact one and so is the form's:
 * pairs compared high word first, then low (a normalized pair's high
 * word is its value rounded, and rounding keeps order); or the ranges of
 * the two sides, widened by the f64 program's error, do not meet, and
 * the comparison is a constant.
 */
struct cmpc {
	struct dv a, b;
	int exact, konst, kval;
	int bits[2];		/* each side's significant bits in f64; -1 */
	nid body;
	double cost;
	int nops;
	struct cand c;		/* the lowering's facts, for the report */
};

/*
 * The significant bits n needs as an f64 when no operation on the way
 * rounds: an f32 converted 24, an integer its width (or its range's), a
 * constant its mantissa's, a product the sum of its factors' (two f32 make
 * 48: exact in f64), a negation or a scaling by a power of two its
 * operand's.  -1: not known to be exact (a sum needs its operands'
 * exponents, a quotient rounds).
 */
static int
f64_bits(const struct func *f, nid n, int depth)
{
	int a, b;
	double k;

	if (depth > 32)
		return -1;
	switch (N.op[n]) {
	case O_CAST:
		if (N.ty[N.a[n]] == TY_F32)
			return 24;
		if (TY_ISINT(N.ty[N.a[n]])) {
			struct iv iv = leaf_iv(f, N.a[n], 0);
			long double m = iv.top ? 0 : iv_mag(&iv);
			int w = 0;
			if (iv.top)
				return TY_SIZED(N.ty[N.a[n]]) && TYW(N.ty[N.a[n]]) <= 53 ?
				    TYW(N.ty[N.a[n]]) : -1;
			while (w < 64 && m >= 1)
				m /= 2, w++;
			return w <= 53 ? w : -1;
		}
		return -1;
	case O_CONST: {
		double v = fabs(N.kr[n]), mnt;
		int e, bits = 0;
		if (isnan(v) || isinf(v))
			return -1;
		if (v == 0)
			return 0;
		mnt = frexp(v, &e);
		while (mnt != floor(mnt) && bits < 60)
			mnt *= 2, bits++;
		return bits <= 53 ? bits : -1;
	}
	case O_NEG:
		return f64_bits(f, N.a[n], depth + 1);
	case O_MUL:
		if (pow2_const(N.a[n], &k))
			return f64_bits(f, N.b[n], depth + 1);
		if (pow2_const(N.b[n], &k))
			return f64_bits(f, N.a[n], depth + 1);
		a = f64_bits(f, N.a[n], depth + 1);
		b = N.b[n] == N.a[n] ? a : f64_bits(f, N.b[n], depth + 1);
		return a >= 0 && b >= 0 && a + b <= 53 ? a + b : -1;
	case O_DIV:
		if (pow2_const(N.b[n], &k))
			return f64_bits(f, N.a[n], depth + 1);
		return -1;
	}
	return -1;
}

static int
is_cmp_op(int op)
{
	return op == O_LT || op == O_LE || op == O_GT || op == O_GE ||
	    op == O_EQ || op == O_NE;
}

/* the comparison body of f, both sides f64 (0: f has none) */
static nid
cmp_root(const struct func *f)
{
	nid b = f->body;

	if (is_cmp_op(N.op[b]) && N.ty[N.a[b]] == TY_F64 &&
	    N.ty[N.b[b]] == TY_F64)
		return b;
	return 0;
}

/* a op b on pairs: the high words first, the low words where they tie */
static nid
cmp_form(int op, const struct dv *a, const struct dv *b)
{
	nid ah = a->hi, bh = b->hi, al, bl;

	if (!a->lo && !b->lo)
		return mk_bin(op, ah, bh);
	al = a->lo ? a->lo : f32k(0);
	bl = b->lo ? b->lo : f32k(0);
	switch (op) {
	case O_EQ:
		return mk_bin(O_LAND, mk_bin(O_EQ, ah, bh), mk_bin(O_EQ, al, bl));
	case O_NE:
		return mk_bin(O_LOR, mk_bin(O_NE, ah, bh), mk_bin(O_NE, al, bl));
	}
	/* a < b: ah < bh, or ah == bh and al < bl; likewise <=, >, >= with
	 * the strict high-word test */
	return mk_bin(O_LOR, mk_bin(op == O_LT || op == O_LE ? O_LT : O_GT, ah,
	    bh), mk_bin(O_LAND, mk_bin(O_EQ, ah, bh), mk_bin(op, al, bl)));
}

/* the comparison's value when the widened ranges decide it: 1 true, 0
 * false, -1 not decided */
static int
cmp_decided(int op, const struct dv *a, const struct dv *b)
{
	long double alo, ahi, blo, bhi;

	if (a->iv.top || b->iv.top || isinf(a->e64.abs_) || isinf(b->e64.abs_))
		return -1;
	alo = (long double)a->iv.lo - a->e64.abs_;
	ahi = (long double)a->iv.hi + a->e64.abs_;
	blo = (long double)b->iv.lo - b->e64.abs_;
	bhi = (long double)b->iv.hi + b->e64.abs_;
	switch (op) {
	case O_LT: return ahi < blo ? 1 : alo >= bhi ? 0 : -1;
	case O_LE: return ahi <= blo ? 1 : alo > bhi ? 0 : -1;
	case O_GT: return alo > bhi ? 1 : ahi <= blo ? 0 : -1;
	case O_GE: return alo >= bhi ? 1 : ahi < blo ? 0 : -1;
	case O_EQ: return ahi < blo || bhi < alo ? 0 : -1;
	case O_NE: return ahi < blo || bhi < alo ? 1 : -1;
	}
	return -1;
}

static void
cmp_cand(const struct func *f, const struct model *m, int mode,
    uint32_t lim, struct cmpc *r)
{
	uint8_t *acc = xcalloc(lim, 1);
	nid root = cmp_root(f);
	int op = N.op[root];

	memset(r, 0, sizeof *r);
	lower_reset(f, mode, lim, acc);
	r->a = lower(N.a[root]);
	r->b = lower(N.b[root]);
	r->kval = cmp_decided(op, &r->a, &r->b);
	r->konst = r->kval >= 0;
	r->bits[0] = f64_bits(f, N.a[root], 0);
	r->bits[1] = f64_bits(f, N.b[root], 0);
	r->exact = r->a.e.abs_ == 0 && r->b.e.abs_ == 0 && r->bits[0] >= 0 &&
	    r->bits[1] >= 0 && !L.opaque && !L.splitvar;
	if (r->konst)
		r->body = mk_const_i(TY_BOOL, r->kval);
	else
		r->body = cmp_form(op, &r->a, &r->b);
	r->cost = dag_cost(m, r->body, &r->nops);
	r->c.mode = mode;
	r->c.body = r->body;
	r->c.used = L.used;
	r->c.maxmag = L.maxmag;
	r->c.mag_unknown = L.mag_unknown;
	r->c.uflow = L.uflow;
	r->c.nprod = L.nprod;
	r->c.div0 = L.div0;
	r->c.sqrt0 = L.sqrt0;
	r->c.pre_cases = L.pre_cases;
	r->c.opaque = L.opaque;
	r->c.splitvar = L.splitvar;
	r->c.table = L.table.p ? strdup(L.table.p) : strdup("");
	r->c.neft = N.n;
	r->c.eft = xcalloc(r->c.neft, 1);
	if (L.eft)
		memcpy(r->c.eft, L.eft, L.eftcap < r->c.neft ? L.eftcap :
		    r->c.neft);
	free(acc);
	L.acc = NULL;
}

/* the form against the f64 comparison at sampled points, where the
 * conditions hold (each side 0 or between 2^-90 and 2^120 in magnitude) */
static void
cmp_check(struct sbuf *o, const struct func *f, nid form)
{
	struct plan p;
	struct diter d;
	nid roots[5];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint8_t ok[CH];
	uint32_t np, k, lf, lr, la, lb, wloc;
	long n = 0, bad = 0, out = 0;

	roots[0] = form;
	roots[1] = f->body;
	roots[2] = N.a[f->body];
	roots[3] = N.b[f->body];
	roots[4] = f->where;
	plan_build(&p, roots, 5, 0);
	wloc = f->where ? plan_loc(&p, f->where) : NOTOK;
	lf = plan_loc(&p, form);
	lr = plan_loc(&p, f->body);
	la = plan_loc(&p, N.a[f->body]);
	lb = plan_loc(&p, N.b[f->body]);
	diter_init(&d, f, DW_SAMPLE, 65, DW_SAMPLE);
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wloc, np, ok);
		for (k = 0; k < np; k++) {
			double va = fabs(COLR(&p, la)[k]), vb = fabs(COLR(&p, lb)[k]);
			if (!ok[k] || (COLF(&p, lr)[k] & F_UNDEF))
				continue;
			if (va > 0x1p120 || vb > 0x1p120 || (va != 0 && va < 0x1p-90) ||
			    (vb != 0 && vb < 0x1p-90)) {
				out++;
				continue;
			}
			n++;
			bad += (COLI(&p, lf)[k] != 0) != (COLI(&p, lr)[k] != 0);
		}
	}
	sb_printf(o, "  check (sampled, not a proof): %ld points of the domain, "
	    "%ld where the form and the f64 comparison differ", n, bad);
	if (out)
		sb_printf(o, " (%ld more outside the conditions: a side past "
		    "2^120 or below 2^-90)", out);
	sb_putc(o, '\n');
	rep_int("check_points", n);
	rep_int("check_differ", bad);
	plan_free(&p);
	free(in_i);
	free(in_r);
}

static int
cmp_lower(const struct func *f, const struct model *m,
    const struct dwreq *rq, struct dwres *out, struct sbuf *o)
{
	struct cmpc r[2];
	uint32_t lim = N.n;
	int best = -1, i, n64;
	double c64 = dag_cost(m, f->body, &n64);
	const char *fn = f->anon ? shared_str(f->body, 120) : sym_str(f->name);
	char ea[120], eb[120];

	cmp_cand(f, m, 0, lim, &r[0]);
	cmp_cand(f, m, 1, lim, &r[1]);
	for (i = 0; i < 2; i++)
		if ((r[i].exact || r[i].konst) && (rq->anyprice || r[i].cost <
		    c64) && (best < 0 || r[i].cost < r[best].cost))
			best = i;
	if (o) {
		sb_printf(o, "dword %s under %s: a comparison of f64 values over "
		    "f32 (u = 2^-24, u64 = 2^-53)\n", fn, sym_str(m->name));
		if (r[1].bits[0] >= 0 && r[1].bits[1] >= 0)
			sb_printf(o, "  as written  cost %-7.6g both sides exact in "
			    "f64 (%d and %d significant bits, no operation "
			    "rounds)\n", c64, r[1].bits[0], r[1].bits[1]);
		else
			sb_printf(o, "  as written  cost %-7.6g the sides' errors "
			    "against their exact values <= %s and %s\n", c64,
			    r[1].bits[0] >= 0 ? "0 (exact)" : err_text(&r[1].a.e64,
			    U64, "u64", ea, sizeof ea), r[1].bits[1] >= 0 ?
			    "0 (exact)" : err_text(&r[1].b.e64, U64, "u64", eb,
			    sizeof eb));
		for (i = 0; i < 2; i++)
			sb_printf(o, "  %-11s cost %-7.6g sides %s and %s: %s\n", i ?
			    "f32 pairs" : "f32 alone", r[i].cost, err_text(&r[i].a.e,
			    i ? U * U : U, i ? "u^2" : "u", ea, sizeof ea),
			    err_text(&r[i].b.e, i ? U * U : U, i ? "u^2" : "u", eb,
			    sizeof eb), r[i].konst ? "decided by the ranges" :
			    r[i].exact ? "both exact, in f64 too" : "not both exact");
		sb_puts(o, "  required: the same answer as the f64 comparison for "
		    "every input: both sides exact (in f64 and in the\n    form), "
		    "or ranges that, widened by the f64 errors, decide it\n");
	}
	if (best < 0) {
		int able = r[0].exact || r[0].konst || r[1].exact || r[1].konst;
		if (able)
			snprintf(out->why, sizeof out->why, "not rewritten: the "
			    "form that gives the same answer costs %g, the f64 "
			    "comparison %g under %s", r[r[1].exact || r[1].konst ?
			    1 : 0].cost, c64, sym_str(m->name));
		else
			snprintf(out->why, sizeof out->why, "not rewritten: a "
			    "side is not exact (in f64 or as f32 pairs) and the "
			    "ranges do not decide the comparison");
		if (o)
			sb_printf(o, "  -> %s\n", out->why);
		out->cost64 = c64;
		out->cost = r[1].cost;
		out->rel = out->abs_ = LINF;
		cand_free(&r[0].c);
		cand_free(&r[1].c);
		return 0;
	}
	out->fired = 1;
	out->mode = best;
	out->hi = r[best].body;
	out->lo = 0;
	out->body = r[best].body;
	out->cost64 = c64;
	out->cost = r[best].cost;
	out->rel = out->abs_ = 0;
	if (o) {
		const struct cmpc *b = &r[best];
		if (b->konst)
			sb_printf(o, "  the sides lie in [%.6g, %.6g] and [%.6g, "
			    "%.6g], each within its f64 error: %s for every "
			    "input\n", b->a.iv.lo, b->a.iv.hi, b->b.iv.lo, b->b.iv.hi,
			    b->kval ? "true" : "false");
		sb_printf(o, "  -> %s: cost %g against %g (%+.1f%%), %d f32 "
		    "operations for %d\n", b->konst ? b->kval ? "true" : "false" :
		    best ? "f32 pairs" : "f32 alone", b->cost, c64, 100 * (b->cost -
		    c64) / c64, b->nops, n64);
		if (!b->konst) {
			if (best)
				sb_puts(o, "     the pairs compared high word first, "
				    "then low where the high words tie: a normalized "
				    "pair's\n     high word is its value rounded, and "
				    "rounding keeps order\n");
			sb_puts(o, "  the form:\n");
			listing(o, &b->c);
			sb_puts(o, "  each value: its range, the derived error "
			    "bound, the algorithm:\n");
			sb_puts(o, b->c.table);
			if (best)
				derivations(o, b->c.used);
			conditions(o, &b->c);
		}
		cmp_check(o, f, b->body);
	}
	out->eft = r[best].c.eft;
	out->neft = r[best].c.neft;
	r[best].c.eft = NULL;
	cand_free(&r[0].c);
	cand_free(&r[1].c);
	return 1;
}

int
dword_lower(const struct func *f, const struct model *m,
    const struct dwreq *rq, struct dwres *out, struct sbuf *o)
{
	struct cand c[2];
	uint32_t lim = N.n;
	int sink32, best = -1, i, n64;
	nid root = island_root(f, &sink32);
	double c64;
	char eb[120];
	const char *fn = f->anon ? shared_str(f->body, 120) : sym_str(f->name);

	memset(out, 0, sizeof *out);
	if (cmp_root(f) && has_f64(f->body))
		return cmp_lower(f, m, rq, out, o);
	if (root == 0 || !has_f64(f->body)) {
		snprintf(out->why, sizeof out->why, "%s has no f64 arithmetic "
		    "whose result is its value (f64, or f32(...) of it)", fn);
		return 0;
	}
	c64 = dag_cost(m, f->body, &n64);
	run_cand(f, m, rq, 0, lim, &c[0]);
	run_cand(f, m, rq, 1, lim, &c[1]);
	for (i = 0; i < 2; i++)
		if (c[i].meets && (rq->anyprice || c[i].cost < c64) && (best < 0 ||
		    c[i].cost < c[best].cost))
			best = i;
	if (o) {
		sb_printf(o, "dword %s under %s: the f64 arithmetic over f32 "
		    "(u = 2^-24, u64 = 2^-53)\n", fn, sym_str(m->name));
		sb_printf(o, "  as written  cost %-7.6g error against the exact "
		    "value <= %s\n", c64, err_text(&c[1].r.e64, U64, "u64", eb,
		    sizeof eb));
		for (i = 0; i < 2; i++) {
			sb_printf(o, "  %-11s cost %-7.6g error <= %s", i ? "f32 "
			    "pairs" : "f32 alone", c[i].cost, err_text(&c[i].r.e,
			    i ? U * U : U, i ? "u^2" : "u", eb, sizeof eb));
			if (sink32 && !isinf(c[i].extra))
				sb_printf(o, ": as f32 within 0.5 + %.3Lg ulp",
				    c[i].extra);
			sb_putc(o, '\n');
		}
		sb_puts(o, "  required: ");
		switch (rq->kind) {
		case DW_TOL:
			sb_printf(o, "relative error <= %g (tol)\n", rq->v);
			break;
		case DW_ATOL:
			sb_printf(o, "absolute error <= %g (atol)\n", rq->v);
			break;
		case DW_ULP:
			sb_printf(o, "the f32 result within 0.5 + %g ulp of the "
			    "exact value (ulp)\n", rq->v);
			break;
		default:
			if (sink32)
				sb_puts(o, "the result feeds an f32: within 0.5 + "
				    "2^-10 ulp of the exact value (ten guard bits; "
				    "`ulp X`, `tol R` or `atol T` to change)\n");
			else
				sb_puts(o, "the result stays f64 and no tolerance "
				    "was given: the exact value (`tol R` or `atol T` "
				    "says how much error is allowed)\n");
		}
	}
	if (best < 0) {
		int cheaper = c[0].cost < c64 || c[1].cost < c64;
		int met = c[0].meets || c[1].meets;
		if (!cheaper)
			snprintf(out->why, sizeof out->why, "not rewritten: under "
			    "%s the f64 operations cost %g, f32 alone %g, the "
			    "pairs %g", sym_str(m->name), c64, c[0].cost,
			    c[1].cost);
		else if (!met)
			snprintf(out->why, sizeof out->why, "not rewritten: no "
			    "f32 form's derived bound meets the requirement");
		else
			snprintf(out->why, sizeof out->why, "not rewritten: the "
			    "form that meets the requirement costs %g, the f64 "
			    "operations %g under %s", c[c[1].meets ? 1 : 0].cost,
			    c64, sym_str(m->name));
		if (o) {
			sb_printf(o, "  -> %s\n", out->why);
			/* the bound and its conditions, for the form nearest */
			if (!met && sink32 == 0 && rq->kind == DW_DEFAULT &&
			    !isinf(c[1].r.e.rel) && c[1].cost < c64)
				sb_printf(o, "     the pairs would be taken for "
				    "tol >= %.3Lg\n", c[1].r.e.rel);
			if (c[1].unsup[0])
				sb_printf(o, "     %s\n", c[1].unsup);
		}
		out->cost64 = c64;
		out->cost = c[1].cost;
		out->rel = c[1].r.e.rel;
		out->abs_ = c[1].r.e.abs_;
		cand_free(&c[0]);
		cand_free(&c[1]);
		return 0;
	}
	out->fired = 1;
	out->mode = best;
	out->hi = c[best].r.hi;
	out->lo = c[best].r.lo;
	out->body = c[best].body;
	out->cost64 = c64;
	out->cost = c[best].cost;
	out->rel = c[best].r.e.rel;
	out->abs_ = c[best].r.e.abs_;
	if (o) {
		const struct cand *b = &c[best];
		sb_printf(o, "  -> %s: cost %g against %g (%+.1f%%), %d f32 "
		    "operations for %d\n", best ? "f32 pairs" : "f32 alone",
		    b->cost, c64, 100 * (b->cost - c64) / c64, b->nops, n64);
		if (b->opaque)
			sb_printf(o, "     %d value%s without a pair form stay%s "
			    "in f64 and %s split (hi = f32(v), lo = f32(v - "
			    "hi))\n", b->opaque, b->opaque == 1 ? "" : "s",
			    b->opaque == 1 ? "s" : "", b->opaque == 1 ? "is" :
			    "are");
		if (b->splitvar)
			sb_printf(o, "     %d f64 parameter%s split here with an "
			    "f64 subtraction each; a caller that passes (hi, "
			    "lo) as two\n     f32 saves that and the fp64 "
			    "capability: an interface change, not made here\n",
			    b->splitvar, b->splitvar == 1 ? " is" : "s are");
		if (!sink32)
			sb_puts(o, "     the result is recombined as f64(hi) + "
			    "f64(lo), its rounding (u64 at most) in the bound; a "
			    "consumer\n     that takes the pair saves that f64 "
			    "addition: an interface change, not made here\n");
		sb_puts(o, "  the form:\n");
		listing(o, b);
		sb_puts(o, "  each value: its range, the derived error bound, "
		    "the algorithm:\n");
		sb_puts(o, b->table);
		if (best)
			derivations(o, b->used);
		conditions(o, b);
	}
	if (o)
		sampled_check(o, f, &c[best], lim);
	out->eft = c[best].eft;
	out->neft = c[best].neft;
	c[best].eft = NULL;
	cand_free(&c[0]);
	cand_free(&c[1]);
	return 1;
}

static struct func *
define_form(const struct func *f, nid body)
{
	char nm[96];
	struct func tmp = *f, *g;

	snprintf(nm, sizeof nm, "%s_dw", f->anon ? "" : sym_str(f->name));
	tmp.name = sym_cstr(nm);
	tmp.anon = 0;
	tmp.body = body;
	g = func_new(tmp.name);
	*g = tmp;
	return g;
}

static void
cmd_dword(struct func *f, const struct model *m, const struct dwreq *rq)
{
	struct dwres r;
	struct sbuf o = {0};
	struct func *g;

	rep_begin("dword");
	rep_str("model", sym_str(m->name));
	dword_lower(f, m, rq, &r, &o);
	if (o.p == NULL)
		sb_printf(&o, "dword: %s\n", r.why);
	fputs(o.p, stdout);
	rep_bool("rewritten", r.fired);
	rep_num("cost_f64", r.cost64);
	rep_num("cost_form", r.cost);
	if (!isinf(r.rel))
		rep_num("bound_relative", (double)r.rel);
	if (!isinf(r.abs_))
		rep_num("bound_absolute", (double)r.abs_);
	if (r.fired) {
		/* f may have moved: func_new can grow the table */
		uint32_t fname = f->name;
		int anon = f->anon;
		struct func copy = *f;
		g = define_form(&copy, r.body);
		rep_str("form", r.mode ? "pairs" : "f32");
		rep_expr("body", g->body);
		rep_str("defined", sym_str(g->name));
		rep_proof("derivation", "the standard model of rounding and "
		    "the error-free transformations, composed over the "
		    "parameters' ranges");
		printf("  defined %s: the form as a function (cost %s, error %s "
		    "~ %s)\n", sym_str(g->name), sym_str(g->name),
		    sym_str(g->name), anon ? "F" : sym_str(fname));
	} else
		rep_str("why", r.why);
	dword_res_free(&r);
	free(o.p);
}

int
dword_command(struct P *p)
{
	static const char *const stop[] = {"tol", "atol", "ulp", "model", NULL};
	struct func *f;
	struct model *m = model_active();
	struct dwreq rq = {DW_DEFAULT, 0};
	char w[48];
	uint32_t at = cmdp_pos(p);

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (strcmp(w, "dword") != 0 || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	f = cmdp_func(p, stop);
	while (!cmdp_eos(p)) {
		if (cmdp_kw(p, "tol")) {
			rq.kind = DW_TOL;
			rq.v = cmdp_number(p, stop);
		} else if (cmdp_kw(p, "atol")) {
			rq.kind = DW_ATOL;
			rq.v = cmdp_number(p, stop);
		} else if (cmdp_kw(p, "ulp")) {
			rq.kind = DW_ULP;
			rq.v = cmdp_number(p, stop);
		} else if (cmdp_kw(p, "model")) {
			if (!cmdp_word(p, w, sizeof w) ||
			    (m = model_find(sym_cstr(w))) == NULL)
				fail("dword: unknown model (see 'models')");
		} else {
			cmdp_fail_here(p);
			fail("dword F [tol R | atol T | ulp X] [model M]");
		}
	}
	if (rq.kind != DW_DEFAULT && !(rq.v > 0))
		fail("dword: the tolerance must be positive, not %g", rq.v);
	notes_flush();
	cmd_dword(f, m, &rq);
	return 1;
}

/* find: one line when a cheaper f32 form meets the default requirement */
int
dword_find(struct func *f)
{
	struct dwres r;
	struct dwreq rq = {DW_DEFAULT, 0};
	const struct model *m = model_active();
	int said;

	if (!has_f64(f->body))
		return 0;
	if ((said = dword_lower(f, m, &rq, &r, NULL)) != 0)
		printf("find %s: its f64 arithmetic as %s costs %g, the f64 "
		    "operations %g under %s;\n  `dword %s` derives the error "
		    "bound and prints the form\n", f->anon ? shared_str(f->body,
		    120) : sym_str(f->name), r.mode ? "f32 pairs (double-word "
		    "arithmetic)" : "f32 alone", r.cost, r.cost64,
		    sym_str(m->name), f->anon ? "F" : sym_str(f->name));
	dword_res_free(&r);
	return said;
}

/* ---- English ---- */

static int
word_eq(const char *s, size_t n, const char *w)
{
	return strlen(w) == n && !strncasecmp(s, w, n);
}

/*
 * avoid fp64 | doubles | double precision in F; F without fp64 | doubles;
 * float-float F; double-word F; emulate fp64 | doubles in F [with float
 * pairs]; rewrite F with | as float pairs; each with [within N ulp] [to
 * within R | tolerance R] [under | on MODEL].  The driver's folds:
 * make F fold-proof, fold-proof F, does the driver fold F, keep the half
 * rounding in F.
 */
const char *
dword_nlp_raw(const char *sent)
{
	static char out[1200];
	char buf[1000], tail[200] = "", *s, *e, *q;
	size_t n;
	const char *cmd = NULL;
	static const char *const pre[] = {"avoid fp64 in ", "avoid doubles in ",
		"avoid double precision in ", "avoid double in ",
		"float-float ", "float float ", "double-word ", "double word ",
		"emulate fp64 in ", "emulate doubles in ", "emulate double in ",
		"emulate double precision in ", "remove fp64 from ",
		"remove doubles from ", NULL};
	static const char *const post[] = {" without fp64", " without doubles",
		" without double precision", " with float pairs",
		" as float pairs", " with pairs of floats", " as float-float",
		" in double-word arithmetic", NULL};
	static const char *const fpre[] = {"fold-proof ", "fold proof ",
		"does the driver fold ", "is the round trip folded in ",
		"keep the half rounding in ", "keep the f16 rounding in ",
		"which conversions does the driver fold in ", NULL};
	static const char *const fpost[] = {" fold-proof", " fold proof", NULL};
	int i, hit = 0;

	while (*sent == ' ' || *sent == '\t')
		sent++;
	if (strlen(sent) >= sizeof buf)
		return NULL;
	strcpy(buf, sent);
	n = strlen(buf);
	while (n && (buf[n - 1] == ' ' || buf[n - 1] == '?' || buf[n - 1] == '.'
	    || buf[n - 1] == '\n'))
		buf[--n] = 0;
	s = buf;
	if (!strncasecmp(s, "please ", 7))
		s += 7;
	if (!strncasecmp(s, "can you ", 8))
		s += 8;
	/* the tails, from the right */
	for (;;) {
		int again = 0;
		q = NULL;
		for (e = s; (e = strstr(e, " under ")) != NULL; e++)
			q = e;
		if (q == NULL)
			for (e = s; (e = strstr(e, " on ")) != NULL; e++)
				q = e;
		if (q != NULL) {
			const char *w = strrchr(q, ' ') + 1;
			size_t wn = strlen(w);
			if (wn && wn < 40 && strchr(q + 1, ' ') == strrchr(q, ' ') &&
			    model_find(sym_intern(w, wn)) != NULL) {
				snprintf(tail + strlen(tail), sizeof tail -
				    strlen(tail), " model %s", w);
				*q = 0;
				again = 1;
			}
		}
		if ((q = strstr(s, " within ")) != NULL || (q = strstr(s,
		    " to within ")) != NULL || (q = strstr(s, " tolerance "))
		    != NULL) {
			char num[64];
			const char *v = strchr(q + 1, ' ') + 1;
			size_t k = 0;
			if (!strncmp(v, "within ", 7))
				v += 7;
			while (v[k] && v[k] != ' ' && k + 1 < sizeof num) {
				num[k] = v[k];
				k++;
			}
			num[k] = 0;
			if (k && (isdigit((unsigned char)num[0]) || num[0] == '.')) {
				const char *unit = v + k;
				while (*unit == ' ')
					unit++;
				snprintf(tail + strlen(tail), sizeof tail -
				    strlen(tail), " %s %s", !strncmp(unit, "ulp", 3) ?
				    "ulp" : "tol", num);
				*q = 0;
				again = 1;
			}
		}
		if (!again)
			break;
	}
	n = strlen(s);
	while (n && (s[n - 1] == ',' || s[n - 1] == ' '))
		s[--n] = 0;
	for (i = 0; pre[i] && !hit; i++)
		if (!strncasecmp(s, pre[i], strlen(pre[i]))) {
			s += strlen(pre[i]);
			cmd = "dword";
			hit = 1;
		}
	for (i = 0; fpre[i] && !hit; i++)
		if (!strncasecmp(s, fpre[i], strlen(fpre[i]))) {
			s += strlen(fpre[i]);
			cmd = "foldproof";
			hit = 1;
		}
	if (!strncasecmp(s, "rewrite ", 8) || !strncasecmp(s, "compute ", 8) ||
	    !strncasecmp(s, "make ", 5))
		s = strchr(s, ' ') + 1;
	n = strlen(s);
	for (i = 0; post[i]; i++) {
		size_t k = strlen(post[i]);
		if (n > k && !strcasecmp(s + n - k, post[i])) {
			s[n - k] = 0;
			n -= k;
			cmd = "dword";
			hit = 1;
		}
	}
	for (i = 0; fpost[i]; i++) {
		size_t k = strlen(fpost[i]);
		if (n > k && !strcasecmp(s + n - k, fpost[i])) {
			s[n - k] = 0;
			n -= k;
			cmd = "foldproof";
			hit = 1;
		}
	}
	if (!hit || cmd == NULL || *s == 0)
		return NULL;
	if (word_eq(s, 2, "it") && s[2] == 0)
		return NULL;	/* a pronoun: nlp.c's own parser knows it */
	snprintf(out, sizeof out, "%s %s%s", cmd, s, tail);
	return out;
}

/* fpfold.c: the interval of an f32 or integer expression, 0 when unknown */
int
dword_interval(const struct func *f, nid n, double *lo, double *hi)
{
	struct iv v = leaf_iv(f, n, 0);

	if (v.top)
		return 0;
	*lo = v.lo;
	*hi = v.hi;
	return 1;
}

/* ---- a loop's f64 sum: its pair carried across the passes ---- */

/*
 * The passes: S(0) = s0, S(k+1) = S(k) + T(k), the computed s(k) within
 * E(k) of S(k).  A pass's form is derived like any result, s being the
 * pair the pass is handed: r bounds its error against the exact
 * s(k) + T(k), so E(k+1) <= E(k) + r |s(k) + T(k)| <= (1 + r) E(k) +
 * r |S(k+1)|, and E(n) <= (1 + r)^n E(0) + ((1 + r)^n - 1) max |S(k)|.
 * When s0 and every term have one sign the partial sums only grow,
 * |S(k)| <= |S(n)|, and that is a relative bound.  With only an absolute
 * bound A for a pass, E(n) <= E(0) + n A.  A pass is derived on the
 * range of the partial sums widened by w (outward only when they have
 * one sign), and the bound holds only when E(n) <= w: the computed sums
 * then stay inside it, so each pass's bound applies (by induction on k,
 * E(k) growing with k).
 */
struct acand {
	struct cand c;
	struct err tot;		/* after the passes, against the exact sum */
	struct iv fin, hull;	/* the sum after the passes; every partial */
	long double grow, wid;	/* (1 + r)^n - 1; the widening */
	int same, inside, sub;
	double pass, all;	/* cost: one pass, the whole loop and after */
};

/* the range of every partial sum, and of the last */
static int
acc_ranges(struct acand *A, const struct dwacc *a, struct iv t)
{
	long double n = (long double)a->n;

	if (t.top || isinf(t.lo) || isinf(t.hi) || isnan(a->s0lo) ||
	    isnan(a->s0hi))
		return 0;
	if (A->sub)
		t = iv_neg(t);
	A->fin = iv_make((double)(a->s0lo + n * t.lo), (double)(a->s0hi +
	    n * t.hi));
	A->hull = iv_make((double)(a->s0lo + (t.lo < 0 ? n * t.lo : 0)),
	    (double)(a->s0hi + (t.hi > 0 ? n * t.hi : 0)));
	A->same = (a->s0lo >= 0 && t.lo >= 0) || (a->s0hi <= 0 && t.hi <= 0);
	return !isinf(A->hull.lo) && !isinf(A->hull.hi);
}

static void
acc_passes(struct acand *A, const struct dwacc *a)
{
	const struct err *r = &A->c.r.e;
	long double n = (long double)a->n, M = iv_mag(&A->hull), e0 = a->e0;
	long double m0 = iv_mig(&(struct iv){a->s0lo, a->s0hi, 0}), rel0;

	A->tot.rel = A->tot.abs_ = A->grow = LINF;
	rel0 = e0 == 0 ? 0 : m0 > 0 ? e0 / m0 : LINF;
	if (!isinf(r->rel) && r->rel < 0.5L) {
		long double g = expm1l(n * log1pl(r->rel)) * SLACK;
		A->grow = g;
		A->tot.abs_ = (1 + g) * e0 + lmul(g, M);
		if (A->same && !isinf(rel0))
			A->tot.rel = ((1 + g) * rel0 + g) * SLACK;
	} else if (!isinf(r->abs_))
		A->tot.abs_ = (e0 + n * r->abs_) * SLACK;
	A->inside = !isinf(A->tot.abs_) && A->tot.abs_ <= A->wid;
}

static void
run_acc(const struct func *f, const struct dwacc *a, const struct model *m,
    const struct dwreq *rq, int mode, uint32_t lim, struct acand *A)
{
	uint8_t *acc = xcalloc(lim, 1);
	nid root = f->body, t = N.b[root];
	nid hv = mk_var((uint32_t)a->ph, f->p[a->ph].name, TY_F32);
	nid lv = mk_var((uint32_t)a->pl, f->p[a->pl].name, TY_F32);
	struct dv tv;
	int guard, ok = 0, it;
	long double M;

	memset(A, 0, sizeof *A);
	A->c.mode = mode;
	A->sub = N.op[root] == O_SUB;
	A->wid = 0;
	for (it = 0; it < 4; it++) {
		for (guard = 0; guard < 64; guard++) {
			lower_reset(f, mode, lim, acc);
			tv = lower(t);
			if (!(ok = acc_ranges(A, a, tv.iv)))
				break;
			M = iv_mag(&A->hull);
			if (it == 0)
				A->wid = M * 0x1p-20L;
			L.accv = a->ps + 1;
			L.acch = hv;
			L.accl = lv;
			L.acciv = A->same && A->hull.lo >= 0 ?
			    iv_make((double)ldmax(0, A->hull.lo - A->wid),
			    (double)(A->hull.hi + A->wid)) : A->same ?
			    iv_make((double)(A->hull.lo - A->wid),
			    (double)ldmin(0, A->hull.hi + A->wid)) :
			    iv_make((double)(A->hull.lo - A->wid),
			    (double)(A->hull.hi + A->wid));
			A->c.r = lower(root);
			acc_passes(A, a);
			{
				/* an f32 reads the pair's high word; a store
				 * the pair recombined */
				struct dv fin;
				long double x64;
				memset(&fin, 0, sizeof fin);
				fin.iv = A->fin;
				fin.e = A->tot;
				fin.hi = hv;
				fin.lo = mode ? lv : 0;
				A->c.extra = LINF;
				A->c.meets = A->inside && (a->nconv == 0 ||
				    meets(rq, 1, &fin, &A->c.extra));
				if (a->nstore && mode)
					fin.e = recombined(&fin);
				if (a->nstore)
					A->c.meets &= meets(rq->kind == DW_ULP ?
					    &(struct dwreq){DW_DEFAULT, 0, 0} : rq, 0,
					    &fin, &x64);
			}
			/* a sloppy add with cancellation: the accurate one */
			if (A->c.meets || !L.nsloppy || !L.worst ||
			    L.worstk <= 1.001L)
				break;
			acc[L.worst] = 1;
		}
		if (!ok || A->inside || isinf(A->tot.abs_))
			break;
		/* the computed sums may leave the range: wider, again */
		A->wid = 2 * A->tot.abs_;
	}
	if (!ok) {
		snprintf(A->c.unsup, sizeof A->c.unsup, "the term %s has no "
		    "range: no bound across the passes", node_str(t));
		A->c.body = A->c.r.hi = hv;
		A->c.cost = LINF;
		A->c.table = strdup("");
		free(acc);
		L.acc = NULL;
		L.accv = 0;
		return;
	}
	A->c.body = A->c.r.hi;
	A->c.body2 = A->c.r.lo;
	if (A->c.r.lo) {
		nid two[2];
		two[0] = A->c.r.hi;
		two[1] = A->c.r.lo;
		A->pass = (double)fuse_roots_cost(m, two, 2);
	} else
		A->pass = dag_cost(m, A->c.r.hi, NULL);
	(void)dag_cost(m, A->c.r.hi, &A->c.nops);
	/* the sum's f32 is the pair's high word, RN(hi + lo); a store pays
	 * the recombination */
	A->all = (double)a->n * A->pass;
	if (a->nstore)
		A->all += (double)a->nstore * dag_cost(m, mode ? mk_bin(O_ADD,
		    mk_cast(TY_F64, hv), mk_cast(TY_F64, lv)) : mk_cast(TY_F64,
		    hv), NULL);
	A->c.cost = A->all;
	if (!L.mag_unknown && L.maxmag >= 0x1p126L) {
		A->c.meets = 0;
		snprintf(L.unsup, sizeof L.unsup, "a value reaches %.3Lg: an "
		    "f32 overflows there", L.maxmag);
	}
	A->c.used = L.used;
	A->c.maxmag = L.maxmag;
	A->c.mag_unknown = L.mag_unknown;
	A->c.uflow = L.uflow;
	A->c.nprod = L.nprod;
	A->c.div0 = L.div0;
	A->c.sqrt0 = L.sqrt0;
	A->c.pre_cases = L.pre_cases;
	A->c.opaque = L.opaque;
	A->c.splitvar = L.splitvar;
	A->c.nsloppy = L.nsloppy;
	snprintf(A->c.unsup, sizeof A->c.unsup, "%s", L.unsup);
	A->c.table = L.table.p ? strdup(L.table.p) : strdup("");
	A->c.neft = N.n;
	A->c.eft = xcalloc(A->c.neft, 1);
	if (L.eft)
		memcpy(A->c.eft, L.eft, L.eftcap < A->c.neft ? L.eftcap :
		    A->c.neft);
	memcpy(A->c.leaf, L.leaf, sizeof A->c.leaf);
	A->c.nleaf = L.nleaf;
	free(acc);
	L.acc = NULL;
	L.accv = 0;
}

#define DW_ACC_PASSES 4096	/* the sampled check's passes at most */

/* the loop run on sampled inputs, each pass with another point's terms */
static void
acc_check(struct sbuf *o, const struct func *f, const struct acand *A,
    const struct dwacc *a, uint32_t lim)
{
	struct plan p;
	struct diter d;
	nid roots[MAXLEAF + 8], root = f->body, t = N.b[f->body];
	int nr = 0, i, j;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i), pass;
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int64_t *k_i = xcalloc((size_t)MAXP * CH, sizeof *k_i);
	double *k_r = xcalloc((size_t)MAXP * CH, sizeof *k_r);
	double h[CH], l[CH], v[CH];
	long double ex[CH], worst = 0, worst64 = 0, wabs = 0, unit = A->c.mode ?
	    U * U : U;
	uint32_t np, k, lhi, llo = NOTOK, l64, l0h, l0l = NOTOK;
	long cnt = 0;

	if (a->n > DW_ACC_PASSES) {
		sb_printf(o, "  check: not sampled (%lld passes, past the %d "
		    "the check runs)\n", (long long)a->n, DW_ACC_PASSES);
		free(in_i);
		free(in_r);
		free(k_i);
		free(k_r);
		return;
	}
	roots[nr++] = A->c.r.hi;
	if (A->c.r.lo)
		roots[nr++] = A->c.r.lo;
	roots[nr++] = root;
	roots[nr++] = a->s0h;
	if (a->s0l)
		roots[nr++] = a->s0l;
	for (i = 0; i < A->c.nleaf; i++)
		roots[nr++] = A->c.leaf[i];
	plan_build(&p, roots, (uint32_t)nr, 0);
	lhi = plan_loc(&p, A->c.r.hi);
	if (A->c.r.lo)
		llo = plan_loc(&p, A->c.r.lo);
	l64 = plan_loc(&p, root);
	l0h = plan_loc(&p, a->s0h);
	if (a->s0l)
		l0l = plan_loc(&p, a->s0l);
	diter_init(&d, f, DW_SAMPLE, 65, DW_SAMPLE);
	X.p = &p;
	X.lim = lim;
	X.v = xcalloc(lim, sizeof *X.v);
	X.stamp = xcalloc(lim, sizeof *X.stamp);
	X.cur = 0;
	X.f = f;
	xc = &A->c;
	np = diter_next(&d, k_i, k_r);
	if (np > 0) {
		memcpy(in_i, k_i, (size_t)MAXP * CH * sizeof *in_i);
		memcpy(in_r, k_r, (size_t)MAXP * CH * sizeof *in_r);
		plan_run(&p, np, in_i, in_r);
		for (k = 0; k < np; k++) {
			h[k] = COLR(&p, l0h)[k];
			l[k] = l0l != NOTOK ? COLR(&p, l0l)[k] : 0;
			ex[k] = (long double)h[k] + l[k];
			v[k] = (double)ex[k];
		}
	}
	for (pass = 0; np > 0 && pass < a->n; pass++) {
		for (j = 0; j < f->np; j++)
			for (k = 0; k < np; k++) {
				uint32_t q = (uint32_t)((k + (uint64_t)pass) % np);
				in_i[(size_t)j * CH + k] = k_i[(size_t)j * CH + q];
				in_r[(size_t)j * CH + k] = k_r[(size_t)j * CH + q];
			}
		for (k = 0; k < np; k++) {
			in_r[(size_t)a->ph * CH + k] = h[k];
			in_r[(size_t)a->pl * CH + k] = l[k];
			in_r[(size_t)a->ps * CH + k] = v[k];
		}
		plan_run(&p, np, in_i, in_r);
		for (k = 0; k < np; k++) {
			long double te;
			X.i = k;
			X.cur++;
			te = exact(t);
			ex[k] += N.op[root] == O_SUB ? -te : te;
			h[k] = COLR(&p, lhi)[k];
			l[k] = llo != NOTOK ? COLR(&p, llo)[k] : 0;
			v[k] = COLR(&p, l64)[k];
		}
	}
	for (k = 0; k < np; k++) {
		long double got = (long double)h[k] + l[k], e;
		if (isnan(ex[k]) || isinf(ex[k]) || ex[k] == 0 || isnan(got) ||
		    isinf(got) || fabsl(ex[k]) > 0x1p120L)
			continue;
		cnt++;
		if (fabsl(got - ex[k]) > wabs)
			wabs = fabsl(got - ex[k]);
		e = fabsl(got - ex[k]) / fabsl(ex[k]);
		if (e > worst)
			worst = e;
		e = fabsl((long double)v[k] - ex[k]) / fabsl(ex[k]);
		if (e > worst64)
			worst64 = e;
	}
	sb_printf(o, "  check (sampled, not a proof): the loop run on %ld "
	    "points, each pass reading another point's\n    terms; the "
	    "largest error against the exact sum (long double) %.2Lg %s",
	    cnt, worst / unit, A->c.mode ? "u^2" : "u");
	if (!isinf(A->tot.rel))
		sb_printf(o, " (the bound: %.4Lg)", A->tot.rel / unit);
	else
		sb_printf(o, ", %.3Lg absolute (the bound: %.4Lg)", wabs,
		    A->tot.abs_);
	sb_printf(o, "; the f64 program's %.2Lg u64\n", worst64 / U64);
	rep_int("check_points", cnt);
	rep_num("check_max_rel", (double)worst);
	plan_free(&p);
	free(in_i);
	free(in_r);
	free(k_i);
	free(k_r);
	free(X.v);
	free(X.stamp);
	memset(&X, 0, sizeof X);
}

static void
acc_derivation(struct sbuf *o, const struct acand *A, const struct dwacc *a)
{
	long double unit = A->c.mode ? U * U : U;
	const char *un = A->c.mode ? "u^2" : "u";

	sb_printf(o, "    the passes: S(0) = s0, S(k+1) = S(k) %c T(k); the "
	    "computed s(k) within E(k) of S(k), and a pass\n      within r "
	    "of the exact s(k) %c T(k), so E(k+1) <= (1 + r) E(k) + r "
	    "|S(k+1)|,\n      E(n) <= (1 + r)^n E(0) + ((1 + r)^n - 1) "
	    "max |S(k)|\n", A->sub ? '-' : '+', A->sub ? '-' : '+');
	if (!isinf(A->c.r.e.rel))
		sb_printf(o, "      r = %.4Lg %s (the pass, above), n = %lld: "
		    "(1 + r)^n - 1 <= %.4Lg %s\n", A->c.r.e.rel / unit, un,
		    (long long)a->n, A->grow / unit, un);
	else
		sb_printf(o, "      the pass has no relative bound: E(n) <= E(0) "
		    "+ n A, A = %.4Lg its absolute bound\n", A->c.r.e.abs_);
	if (a->e0 > 0)
		sb_printf(o, "      E(0) <= %.4Lg: s0 as the pair it starts "
		    "from\n", a->e0);
	else
		sb_puts(o, "      E(0) = 0: s0 is its pair exactly\n");
	if (A->same)
		sb_printf(o, "      s0 and every term of one sign: |S(k)| <= "
		    "|S(n)|, so the bound is relative: E(n) <= %.4Lg %s |S(n)|\n",
		    A->tot.rel / unit, un);
	else
		sb_printf(o, "      terms of either sign: max |S(k)| <= %.4g, "
		    "E(n) <= %.4Lg (absolute)\n", (double)iv_mag(&A->hull),
		    A->tot.abs_);
	sb_printf(o, "      each pass derived on the partial sums' range [%.6g, "
	    "%.6g] widened by w = %.3Lg; E(n) <= w%s,\n      so every "
	    "computed sum stays inside it (E(k) grows with k)\n",
	    A->hull.lo, A->hull.hi, A->wid, A->inside ? "" : ": NOT met");
}

int
dword_accum(const struct func *f, const struct dwacc *a,
    const struct model *m, const struct dwreq *rq, struct dwres *out,
    struct sbuf *o)
{
	struct acand c[2];
	uint32_t lim = N.n;
	nid root = f->body, sv;
	double pass64, once64 = 0, end64, all64;
	int best = -1, i, n64;
	char eb[120];

	memset(out, 0, sizeof *out);
	if (N.ty[root] != TY_F64 || (N.op[root] != O_ADD && N.op[root] !=
	    O_SUB) || N.op[N.a[root]] != O_VAR || (int)N.ki[N.a[root]] !=
	    a->ps) {
		snprintf(out->why, sizeof out->why, "not a sum's pass (s + T or "
		    "s - T)");
		return 0;
	}
	sv = N.a[root];
	pass64 = dag_cost(m, root, &n64);
	if (a->nonce > 0) {
		once64 = (double)fuse_roots_cost(m, a->once, a->nonce);
		pass64 -= once64;
	}
	end64 = a->nconv * dag_cost(m, mk_cast(TY_F32, sv), NULL);
	all64 = (double)a->n * pass64 + once64 + end64;
	for (i = 0; i < 2; i++)
		run_acc(f, a, m, rq, i, lim, &c[i]);
	for (i = 0; i < 2; i++)
		if (c[i].c.meets && (rq->anyprice || c[i].all < all64) &&
		    (best < 0 || c[i].all < c[best].all))
			best = i;
	if (o) {
		sb_printf(o, "dword: a loop's f64 sum under %s, s = s0, then %lld "
		    "passes of s = s %c %s (u = 2^-24, u64 = 2^-53)\n",
		    sym_str(m->name), (long long)a->n, N.op[root] == O_SUB ? '-' :
		    '+', node_str(N.b[root]));
		sb_printf(o, "  as written  cost %g a pass, %g in all", pass64,
		    all64);
		if (once64 > 0 && end64 > 0)
			sb_printf(o, " (%g computed before the loop, %g the f32 after "
			    "it)", once64, end64);
		else if (once64 > 0)
			sb_printf(o, " (%g computed before the loop)", once64);
		else if (end64 > 0)
			sb_printf(o, " (%g the f32 after it)", end64);
		sb_putc(o, '\n');
		if (a->nstore)
			sb_printf(o, "  the sum is stored as f64 (%d store%s): the "
			    "pair recombined, f64(hi) + f64(lo), its rounding (u64 at "
			    "most) in the bound\n", a->nstore, a->nstore == 1 ? "" :
			    "s");
		for (i = 0; i < 2; i++) {
			sb_printf(o, "  %-11s cost %g a pass, %g in all", i ? "f32 "
			    "pairs" : "f32 alone", c[i].pass, c[i].all);
			if (c[i].c.cost == LINF) {
				sb_printf(o, ": %s\n", c[i].c.unsup);
				continue;
			}
			sb_printf(o, "; a pass within %s,\n              after the "
			    "%lld passes %s", err_text(&c[i].c.r.e, i ? U * U : U,
			    i ? "u^2" : "u", eb, sizeof eb), (long long)a->n, "");
			sb_printf(o, "%s", err_text(&c[i].tot, i ? U * U : U, i ?
			    "u^2" : "u", eb, sizeof eb));
			if (!c[i].inside)
				sb_puts(o, " (leaves the range it was derived on)");
			else if (!isinf(c[i].c.extra))
				sb_printf(o, ": as f32 within 0.5 + %.3Lg ulp",
				    c[i].c.extra);
			sb_putc(o, '\n');
		}
		sb_puts(o, "  required: ");
		switch (rq->kind) {
		case DW_TOL:
			sb_printf(o, "relative error <= %g (tol)\n", rq->v);
			break;
		case DW_ATOL:
			sb_printf(o, "absolute error <= %g (atol)\n", rq->v);
			break;
		case DW_ULP:
			sb_printf(o, "the f32 result within 0.5 + %g ulp of the "
			    "exact value (ulp)\n", rq->v);
			break;
		default:
			if (a->nconv)
				sb_puts(o, "the sum feeds an f32: within 0.5 + 2^-10 ulp "
				    "of the exact value (ten guard bits; `ulp X`, `tol R` "
				    "or `atol T` to change)\n");
			if (a->nstore)
				sb_printf(o, "%sthe sum is stored as f64 and no "
				    "tolerance was given: the exact value (`tol R` or "
				    "`atol T` says how much error is allowed)\n",
				    a->nconv ? "  and " : "");
		}
	}
	out->cost64 = all64;
	if (best < 0) {
		int cheaper = c[0].all < all64 || c[1].all < all64;
		int met = c[0].c.meets || c[1].c.meets;
		if (!cheaper)
			snprintf(out->why, sizeof out->why, "not rewritten: under "
			    "%s the f64 loop costs %g, f32 alone %g, the pairs %g",
			    sym_str(m->name), all64, c[0].all, c[1].all);
		else if (!met)
			snprintf(out->why, sizeof out->why, "not rewritten: no "
			    "f32 form's derived bound meets the requirement after "
			    "the %lld passes", (long long)a->n);
		else
			snprintf(out->why, sizeof out->why, "not rewritten: the "
			    "form that meets the requirement costs %g, the f64 loop "
			    "%g under %s", c[c[1].c.meets ? 1 : 0].all, all64,
			    sym_str(m->name));
		if (o) {
			sb_printf(o, "  -> %s\n", out->why);
			if (c[1].c.unsup[0] && c[1].c.cost != LINF)
				sb_printf(o, "     %s\n", c[1].c.unsup);
			/* where the requirement is relative and the sum can be 0 */
			if (!met && cheaper && c[1].inside && isinf(c[1].tot.rel) &&
			    !isinf(c[1].tot.abs_) && rq->kind != DW_ATOL &&
			    c[1].fin.lo <= 0 && c[1].fin.hi >= 0)
				sb_printf(o, "     the sum's range [%.6g, %.6g] holds 0: no "
				    "relative bound; the pairs would be taken for "
				    "atol >= %.3Lg\n", c[1].fin.lo, c[1].fin.hi,
				    c[1].tot.abs_);
		}
		out->cost = c[1].all;
		out->rel = c[1].tot.rel;
		out->abs_ = c[1].tot.abs_;
		for (i = 0; i < 2; i++)
			cand_free(&c[i].c);
		return 0;
	}
	out->fired = 1;
	out->mode = best;
	out->hi = c[best].c.r.hi;
	out->lo = c[best].c.r.lo;
	out->body = c[best].c.r.hi;
	out->cost = c[best].all;
	out->rel = c[best].tot.rel;
	out->abs_ = c[best].tot.abs_;
	if (o) {
		const struct acand *b = &c[best];
		sb_printf(o, "  -> %s: cost %g against %g (%+.1f%%), the sum "
		    "carried in %s across the passes\n", best ? "f32 pairs" :
		    "f32 alone", b->all, all64, 100 * (b->all - all64) / all64,
		    best ? "two f32 (hi, lo)" : "one f32");
		sb_printf(o, "  a pass (%s%s: the sum the pass is handed):\n",
		    f->p[a->ph].name ? sym_str(f->p[a->ph].name) : "hi", best ?
		    ", and its low word" : "");
		listing(o, &b->c);
		sb_puts(o, "  each value: its range, the derived error bound, "
		    "the algorithm:\n");
		sb_puts(o, b->c.table);
		if (best)
			derivations(o, b->c.used);
		else
			sb_puts(o, "  derivation (u = 2^-24; RN rounds to "
			    "nearest):\n");
		acc_derivation(o, b, a);
		conditions(o, &b->c);
		acc_check(o, f, b, a, lim);
	}
	out->eft = c[best].c.eft;
	out->neft = c[best].c.neft;
	c[best].c.eft = NULL;
	for (i = 0; i < 2; i++)
		cand_free(&c[i].c);
	return 1;
}
