/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * acert: a bound on an approximation's error that holds for every x in
 * its interval, derived, for what approx's searches keep (approx.c).
 *
 * The candidate (a program in f32 or f64) and the target (a body of one
 * real variable) are cut into pieces X = [l, r] with float ends (even in
 * the floats' order, so the pieces are geometric in x, and bisected where
 * a bound is loose or a branch undecided).  On each piece:
 *
 *   1. the approximation: E = R - f, R the candidate in exact arithmetic
 *	with its rounded constants.  Taylor's theorem at the piece's middle
 *	m, D = X - m,
 *
 *	    E(x) = sum_{k<K} E_k(m) (x - m)^k + E_K(xi) (x - m)^K,
 *
 *	with E_k(m) enclosed by Taylor arithmetic on intervals at the point
 *	m, and E_K(xi), xi in X, by the same recurrences on X itself (the
 *	order-K coefficient of every point of X lies in the result).  K = 2
 *	is float simplify's |E(m)| + |E'(m)| h/2 + max|E''| h^2/8 (fsimp.c);
 *	f32 takes K = 3, f64 K = 6.  A relative bound divides by the least
 *	|f| on X.  When f has a zero of order k0 at z (the fit's factor
 *	(x - z)^k0), E and f are divided by (x - z)^k0 first: by Taylor at z
 *	on the piece that holds z (E's first k0 coefficients there must be
 *	exactly zero), by the jets' division elsewhere;
 *   2. the rounding: every operation of the candidate returns
 *	fl(v) = v (1 + d) (+ e, |e| <= half the least subnormal, where the
 *	result can be subnormal), |d| <= u, and each value's error is
 *	carried as rho |v| + eta through the program, with |v| bounded on X
 *	by intervals; values are kept as t^p w (t = x - z) so that a sum
 *	near the zero is divided by its cofactor, not by 0, and a sum that
 *	varies more than twice over the piece carries its error absolute.
 *	The last rounding adds half an ulp of f where its operand and f
 *	share a binade, at most their distance where they do not (a power
 *	of two lies between them);
 *   3. the piece's bound in the metric: ulp of the exact value (relative)
 *	or absolute; the certificate is the largest over the pieces.
 *
 * A range reduction's candidate is taken piece by piece where its integer
 * side is one value: a node whose computed value is one number on the
 * whole piece is frozen to it (ucas's own constant folding gives it, so
 * it is what the program computes), R uses it as a constant, and its
 * error is 0.  Integers are followed as intervals where the operation is
 * monotone (asuint of a float of one sign, a constant added, shifts, a
 * mask of high bits); a float's bits moved by a frozen multiple of 2^mant
 * and read back are the float times a power of two, and a frozen sign
 * bit xor-ed in its negation.  An operation is exact when every result it
 * can have is a multiple of its quantum q within 2^prec q (Cody-Waite's
 * k c1 step, Sterbenz), or a product by a frozen power of two that cannot
 * underflow.  A select's untaken side is not evaluated.
 *
 * Interval arithmetic is in long double with outward rounding: +, -, *,
 * / and sqrt round in the needed direction by their exact error terms
 * (TwoSum, Dekker's product), so an exact result stays a point; the
 * long double functions of libm (sinl, expl, logl, atanl...) are taken
 * within ACERT_SLACK of the true value (relative, 32 to 64 of long
 * double's ulps), except at their exact points
 * (sin 0 = 0, cos 0 = 1, log 1 = 0, ...).  That is the one assumption,
 * and the report states it.
 */
#include "approx.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI_L 3.14159265358979323846264338327950288L

/* the libm functions' error, relative: 32 to 64 of long double's ulps
 * (2^-58 where long double is x87's; 2^-47 where it is double, whose
 * libm is off by 2^-53 and more) */
#define ACERT_SLACK_EXP (6 - LDBL_MANT_DIG)
#define ACERT_SLACK ldexpl(1, ACERT_SLACK_EXP)
/* a piece is refined while its coefficients' bound exceeds what was
 * reached anywhere by more than this fraction of the largest total */
#define ACERT_TAU (1.0L / 128)
#define ACERT_MAXPIECES 24576
#define ACERT_K_MAX 8

/* ---- intervals with outward rounding ---- */

struct iv {
	ld lo, hi;
};

static ld split_k;	/* 2^ceil(p/2) + 1, Veltkamp's constant */

static ld
nx_dn(ld v)
{
	return nextafterl(v, -INFINITY);
}

static ld
nx_up(ld v)
{
	return nextafterl(v, INFINITY);
}

/* a + b rounded down (up = 0) or up: TwoSum's error says which way */
static ld
add_r(ld a, ld b, int up)
{
	ld s = a + b, bb, e;

	if (!isfinite(s))
		return s;
	bb = s - a;
	e = (a - (s - bb)) + (b - bb);
	if (e == 0)
		return s;
	return up ? (e > 0 ? nx_up(s) : s) : (e < 0 ? nx_dn(s) : s);
}

static void
vsplit(ld a, ld *h, ld *l)
{
	ld t = split_k * a;

	*h = t - (t - a);
	*l = a - *h;
}

/* a b - p exactly, p = fl(a b) (Dekker) */
static ld
prod_err(ld a, ld b, ld p)
{
	ld ah, al, bh, bl;

	vsplit(a, &ah, &al);
	vsplit(b, &bh, &bl);
	return al * bl - (((p - ah * bh) - al * bh) - ah * bl);
}

static ld
mul_r(ld a, ld b, int up)
{
	ld p, e;

	if (a == 0 || b == 0)
		return 0;
	p = a * b;
	if (!isfinite(p))
		return p;
	e = prod_err(a, b, p);
	if (e == 0)
		return p;
	return up ? (e > 0 ? nx_up(p) : p) : (e < 0 ? nx_dn(p) : p);
}

static ld
div_r(ld a, ld b, int up)
{
	ld q, p, e, r;
	int s;

	if (a == 0)
		return 0;
	q = a / b;
	if (!isfinite(q) || q == 0)
		return up ? (q == 0 ? (a > 0) == (b > 0) ? nx_up(0) : 0 : q) :
		    (q == 0 ? (a > 0) != (b > 0) ? nx_dn(0) : 0 : q);
	p = q * b;
	e = prod_err(q, b, p);
	r = (a - p) - e;	/* a - q b, its sign exact */
	if (r == 0)
		return q;
	s = (r > 0) == (b > 0) ? 1 : -1;	/* a / b - q */
	return up ? (s > 0 ? nx_up(q) : q) : (s < 0 ? nx_dn(q) : q);
}

static ld
sqrt_r(ld a, int up)
{
	ld s, p, e, r;

	if (a <= 0)
		return 0;
	s = sqrtl(a);
	p = s * s;
	e = prod_err(s, s, p);
	r = (a - p) - e;
	if (r == 0)
		return s;
	return up ? (r > 0 ? nx_up(s) : s) : (r < 0 ? nx_dn(s) : s);
}

static struct iv
ivp(ld v)
{
	struct iv r;

	r.lo = r.hi = v;
	return r;
}

static struct iv
iv2(ld lo, ld hi)
{
	struct iv r;

	r.lo = lo;
	r.hi = hi;
	return r;
}

static ld
mag(struct iv a)
{
	return fmaxl(fabsl(a.lo), fabsl(a.hi));
}

static ld
mig(struct iv a)
{
	if (a.lo > 0)
		return a.lo;
	if (a.hi < 0)
		return -a.hi;
	return 0;
}

static int
has0(struct iv a)
{
	return a.lo <= 0 && a.hi >= 0;
}

static int
is0(struct iv a)
{
	return a.lo == 0 && a.hi == 0;
}

static struct iv
iadd(struct iv a, struct iv b)
{
	return iv2(add_r(a.lo, b.lo, 0), add_r(a.hi, b.hi, 1));
}

static struct iv
isub(struct iv a, struct iv b)
{
	return iv2(add_r(a.lo, -b.hi, 0), add_r(a.hi, -b.lo, 1));
}

static struct iv
ineg(struct iv a)
{
	return iv2(-a.hi, -a.lo);
}

static struct iv
imul(struct iv a, struct iv b)
{
	ld lo, hi, t;

	if (is0(a) || is0(b))
		return ivp(0);
	if (a.lo >= 0 && b.lo >= 0)
		return iv2(mul_r(a.lo, b.lo, 0), mul_r(a.hi, b.hi, 1));
	lo = mul_r(a.lo, b.lo, 0);
	t = mul_r(a.lo, b.hi, 0);
	lo = fminl(lo, t);
	t = mul_r(a.hi, b.lo, 0);
	lo = fminl(lo, t);
	t = mul_r(a.hi, b.hi, 0);
	lo = fminl(lo, t);
	hi = mul_r(a.lo, b.lo, 1);
	t = mul_r(a.lo, b.hi, 1);
	hi = fmaxl(hi, t);
	t = mul_r(a.hi, b.lo, 1);
	hi = fmaxl(hi, t);
	t = mul_r(a.hi, b.hi, 1);
	hi = fmaxl(hi, t);
	return iv2(lo, hi);
}

/* a / b, b without 0 */
static struct iv
idiv(struct iv a, struct iv b)
{
	ld lo, hi, t;

	if (is0(a))
		return ivp(0);
	lo = div_r(a.lo, b.lo, 0);
	t = div_r(a.lo, b.hi, 0);
	lo = fminl(lo, t);
	t = div_r(a.hi, b.lo, 0);
	lo = fminl(lo, t);
	t = div_r(a.hi, b.hi, 0);
	lo = fminl(lo, t);
	hi = div_r(a.lo, b.lo, 1);
	t = div_r(a.lo, b.hi, 1);
	hi = fmaxl(hi, t);
	t = div_r(a.hi, b.lo, 1);
	hi = fmaxl(hi, t);
	t = div_r(a.hi, b.hi, 1);
	hi = fmaxl(hi, t);
	return iv2(lo, hi);
}

/* a times a whole number (or its reciprocal) */
static struct iv
iscale(struct iv a, ld k)
{
	return imul(a, ivp(k));
}

static struct iv
iabs(struct iv a)
{
	if (a.lo >= 0)
		return a;
	if (a.hi <= 0)
		return ineg(a);
	return iv2(0, fmaxl(-a.lo, a.hi));
}

/* |v|^e rounded down or up, e >= 0 */
static ld
pow_mag(ld v, int e, int up)
{
	ld r = 1;
	int i;

	v = fabsl(v);
	for (i = 0; i < e; i++)
		r = mul_r(r, v, up);
	return r;
}

/* a^e, e >= 0 whole: monotone for odd e, through |a| for even e */
static struct iv
ipowi(struct iv a, int e)
{
	struct iv b;

	if (e == 0)
		return ivp(1);
	if (e % 2 == 0) {
		b = iabs(a);
		return iv2(pow_mag(b.lo, e, 0), pow_mag(b.hi, e, 1));
	}
	return iv2(a.lo >= 0 ? pow_mag(a.lo, e, 0) : -pow_mag(a.lo, e, 1),
	    a.hi >= 0 ? pow_mag(a.hi, e, 1) : -pow_mag(a.hi, e, 0));
}

/* ---- libm's functions on intervals ---- */

/* the exact values at the points where they are known */
static int
el_exact(int op, ld x, ld *v)
{
	switch (op) {
	case O_SIN: case O_TAN: case O_ATAN: case O_ASIN: case O_SINH:
	case O_TANH: case O_ASINH: case O_ATANH:
		if (x == 0) {
			*v = 0;
			return 1;
		}
		break;
	case O_COS: case O_COSH: case O_EXP:
		if (x == 0) {
			*v = 1;
			return 1;
		}
		break;
	case O_LOG: case O_ACOS: case O_ACOSH:
		if (x == 1) {
			*v = 0;
			return 1;
		}
		break;
	}
	return 0;
}

static ld
el_val(int op, ld x)
{
	switch (op) {
	case O_SIN: return sinl(x);
	case O_COS: return cosl(x);
	case O_TAN: return tanl(x);
	case O_EXP: return expl(x);
	case O_LOG: return logl(x);
	case O_ATAN: return atanl(x);
	case O_ASIN: return asinl(x);
	case O_ACOS: return acosl(x);
	case O_SINH: return sinhl(x);
	case O_COSH: return coshl(x);
	case O_TANH: return tanhl(x);
	case O_ASINH: return asinhl(x);
	case O_ACOSH: return acoshl(x);
	case O_ATANH: return atanhl(x);
	}
	return NAN;
}

static ld
el_r(int op, ld x, int up)
{
	ld v;

	if (el_exact(op, x, &v))
		return v;
	v = el_val(op, x);
	if (v == 0)
		return up ? LDBL_MIN * LDBL_EPSILON : -LDBL_MIN * LDBL_EPSILON;
	return up ? nx_up(v + fabsl(v) * ACERT_SLACK) :
	    nx_dn(v - fabsl(v) * ACERT_SLACK);
}

/* f over [a.lo, a.hi], f monotone increasing (dir 1) or decreasing; an
 * exact point p of f on one side of a bounds f there by f(p) (exp(x) >= 1
 * for x >= 0: the slack must not cross it) */
static struct iv
el_mono(int op, struct iv a, int dir)
{
	struct iv r;
	ld p, fp;

	if (dir > 0)
		r = iv2(el_r(op, a.lo, 0), el_r(op, a.hi, 1));
	else
		r = iv2(el_r(op, a.hi, 0), el_r(op, a.lo, 1));
	p = op == O_LOG || op == O_ACOS || op == O_ACOSH ? 1 : 0;
	if (!el_exact(op, p, &fp))
		return r;
	if (a.lo >= p) {
		if (dir > 0)
			r.lo = fmaxl(r.lo, fp);
		else
			r.hi = fminl(r.hi, fp);
	}
	if (a.hi <= p) {
		if (dir > 0)
			r.hi = fminl(r.hi, fp);
		else
			r.lo = fmaxl(r.lo, fp);
	}
	return r;
}

/* is some c0 + k pi (k of the parity par, -1 any) within a, give or take
 * the error of computing it? */
static int
crit_in(struct iv a, ld c0, int par)
{
	ld k, kl = floorl((a.lo - c0) / PI_L) - 1, kh = ceill((a.hi - c0) /
	    PI_L) + 1;

	for (k = kl; k <= kh; k++) {
		ld c = c0 + k * PI_L, eps = 1e-15L * (1 + fabsl(c));
		if (par >= 0 && fabsl(fmodl(k, 2)) != par)
			continue;
		if (c >= a.lo - eps && c <= a.hi + eps)
			return 1;
	}
	return 0;
}

/* sin (cos: shift 1) over a */
static struct iv
el_sincos(struct iv a, int cosine)
{
	int op = cosine ? O_COS : O_SIN;
	ld c0 = cosine ? 0 : PI_L / 2;
	struct iv r;

	if (a.hi - a.lo >= 6.3L)
		return iv2(-1, 1);
	r.lo = fminl(el_r(op, a.lo, 0), el_r(op, a.hi, 0));
	r.hi = fmaxl(el_r(op, a.lo, 1), el_r(op, a.hi, 1));
	if (crit_in(a, c0, 0))
		r.hi = 1;
	if (crit_in(a, c0, 1))
		r.lo = -1;
	if (r.hi > 1)
		r.hi = 1;
	if (r.lo < -1)
		r.lo = -1;
	return r;
}

/* ---- programs ---- */

struct prog {
	int n, root;
	uint8_t *op;
	int *a, *b, *c;
	ld *k;
	uint8_t *fw;		/* a float node's width (32, 64); 0: exact */
	signed char *dec;	/* the branch a node took on the piece */
	int slot;
	nid *g;			/* the node it was made from */
	/* on the piece: the nodes that could not be evaluated (each with
	 * why), and those whose computed value is one number there (fz,
	 * fv) or another's times +-2^xs exactly (xsrc >= 0: a bit trick) */
	uint8_t *bad, *fz, *xneg;
	char (*why)[120];
	ld *fv;
	int64_t *fi;		/* a frozen integer's value, in its type */
	int *xsrc, *xs;
};

/* the operand k reads for its value on the piece (-1 none): a select
 * reads its taken side, a frozen node nothing, a bit trick its source */
static int
uses(const struct prog *P, int k, int i)
{
	int op = P->op[k], ar = opinfo[op].arity;

	if (P->fz[k])
		return -1;
	if (op == O_ASFLOAT && P->xsrc[k] >= 0)
		return i == 0 ? P->xsrc[k] : -1;
	if (op == O_SEL) {
		if (i == 0)
			return P->a[k];
		if (i == 1 && P->dec[P->a[k]] >= 0)
			return P->dec[P->a[k]] ? P->b[k] : P->c[k];
		return -1;
	}
	if (op == O_CONST || op == O_VAR || i >= ar)
		return -1;
	return i == 0 ? P->a[k] : i == 1 ? P->b[k] : P->c[k];
}

static int
opbad(const struct prog *P, int k)
{
	int i, q;

	for (i = 0; i < 3; i++)
		if ((q = uses(P, k, i)) >= 0 && P->bad[q])
			return 1;
	return 0;
}

/* why the root could not be evaluated: the first node on its way down
 * whose operands could */
static void
bad_why(const struct prog *P, char *out, size_t on)
{
	int k = P->root, i, q, again = 1;

	while (again) {
		again = 0;
		for (i = 0; i < 3; i++)
			if ((q = uses(P, k, i)) >= 0 && P->bad[q]) {
				k = q;
				again = 1;
				break;
			}
	}
	snprintf(out, on, "%s", P->why[k]);
}

static int
cand_op(int op)
{
	switch (op) {
	case O_CONST: case O_VAR: case O_CAST: case O_BARRIER: case O_NEG:
	case O_ADD: case O_SUB: case O_MUL: case O_DIV: case O_FMA:
	case O_ABS: case O_COPYSIGN: case O_EQ: case O_NE: case O_LT:
	case O_LE: case O_GT: case O_GE: case O_LAND: case O_LOR:
	case O_LNOT: case O_SEL: case O_MIN: case O_MAX:
	/* a range reduction's integer side and bit tricks, the machine exp */
	case O_ASUINT: case O_ASINT: case O_ASFLOAT: case O_SHL: case O_SHR:
	case O_AND: case O_OR: case O_XOR: case O_EXP:
		return 1;
	}
	return 0;
}

static int
target_op(int op)
{
	switch (op) {
	case O_SQRT: case O_EXP: case O_LOG: case O_SIN: case O_COS:
	case O_TAN: case O_SEC: case O_CSC: case O_COT: case O_ATAN:
	case O_ASIN: case O_ACOS: case O_SINH: case O_COSH: case O_TANH:
	case O_ASINH: case O_ACOSH: case O_ATANH: case O_POW: case O_SIGN:
	case O_FLOOR: case O_CEIL:
		return 1;
	}
	return cand_op(op);
}

/* body as a program; 0 and why when an operation has no rule here */
static int
prog_make(struct prog *P, nid body, int slot, int cand, char *why,
    size_t wn)
{
	uint8_t *mark = xcalloc(N.n, 1);
	uint32_t *map = xcalloc(N.n, sizeof *map);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0;
	int k, ok = 1;

	memset(P, 0, sizeof *P);
	P->slot = slot;
	st[sp++] = body;
	mark[body] = 1;
	while (sp) {
		g = st[--sp];
		if (ok && !(cand ? cand_op(N.op[g]) : target_op(N.op[g]))) {
			snprintf(why, wn, "%s %s has no interval rule here",
			    cand ? "the candidate's" : "the target's",
			    opinfo[N.op[g]].name);
			ok = 0;
		}
		if (ok && N.op[g] == O_VAR && N.ki[g] != slot) {
			snprintf(why, wn, "the target has another variable");
			ok = 0;
		}
		if (ok && N.op[g] == O_POW && (!N.b[g] || N.op[N.b[g]] !=
		    O_CONST)) {
			snprintf(why, wn, "the target's pow has an exponent that "
			    "is not a constant");
			ok = 0;
		}
		if (ok && cand && TYK(N.ty[g]) == K_FLOAT && TYW(N.ty[g]) != 32 &&
		    TYW(N.ty[g]) != 64) {
			snprintf(why, wn, "the candidate computes in %s",
			    ty_str(N.ty[g]));
			ok = 0;
		}
		/* a machine candidate's real operation runs in double, rounded
		 * by no rule here */
		if (ok && cand == 2 && TYK(N.ty[g]) == K_REAL && N.op[g] != O_CONST) {
			snprintf(why, wn, "the candidate has a real-valued %s (computed "
			    "in double)", opinfo[N.op[g]].name);
			ok = 0;
		}
		if (N.op[g] == O_CONST || N.op[g] == O_VAR ||
		    N.op[g] == O_UNDEF || N.op[g] == O_TABLE)
			continue;
		if (N.a[g] && !mark[N.a[g]])
			mark[N.a[g]] = 1, st[sp++] = N.a[g];
		if (N.b[g] && !mark[N.b[g]])
			mark[N.b[g]] = 1, st[sp++] = N.b[g];
		if (N.c[g] && !mark[N.c[g]])
			mark[N.c[g]] = 1, st[sp++] = N.c[g];
	}
	if (ok) {
		for (g = 1; g < N.n; g++)
			if (mark[g])
				P->n++;
		P->op = xmalloc(P->n);
		P->fw = xcalloc(P->n, 1);
		P->dec = xcalloc(P->n, 1);
		P->a = xcalloc(P->n, sizeof *P->a);
		P->b = xcalloc(P->n, sizeof *P->b);
		P->c = xcalloc(P->n, sizeof *P->c);
		P->k = xcalloc(P->n, sizeof *P->k);
		P->g = xcalloc(P->n, sizeof *P->g);
		P->bad = xcalloc(P->n, 1);
		P->fz = xcalloc(P->n, 1);
		P->xneg = xcalloc(P->n, 1);
		P->why = xcalloc(P->n, sizeof *P->why);
		P->fv = xcalloc(P->n, sizeof *P->fv);
		P->fi = xcalloc(P->n, sizeof *P->fi);
		P->xsrc = xcalloc(P->n, sizeof *P->xsrc);
		P->xs = xcalloc(P->n, sizeof *P->xs);
		for (g = 1, k = 0; g < N.n; g++) {
			if (!mark[g])
				continue;
			map[g] = k;
			P->g[k] = g;
			P->xsrc[k] = -1;
			P->op[k] = N.op[g];
			P->k[k] = TY_FP(N.ty[g]) ? (ld)N.kr[g] : (ld)N.ki[g];
			if (cand == 2 && TYK(N.ty[g]) == K_FLOAT)
				P->fw[k] = (uint8_t)TYW(N.ty[g]);
			if (N.op[g] != O_CONST && N.op[g] != O_VAR) {
				P->a[k] = N.a[g] ? (int)map[N.a[g]] : 0;
				P->b[k] = N.b[g] ? (int)map[N.b[g]] : 0;
				P->c[k] = N.c[g] ? (int)map[N.c[g]] : 0;
			}
			k++;
		}
		P->root = (int)map[body];
	}
	free(mark);
	free(map);
	free(st);
	return ok;
}

static void
prog_free(struct prog *P)
{
	free(P->g);
	free(P->bad);
	free(P->fz);
	free(P->xneg);
	free(P->why);
	free(P->fv);
	free(P->fi);
	free(P->xsrc);
	free(P->xs);
	free(P->op);
	free(P->fw);
	free(P->dec);
	free(P->a);
	free(P->b);
	free(P->c);
	free(P->k);
	memset(P, 0, sizeof *P);
}

/* ---- Taylor arithmetic on intervals ---- */

static void
jmul(const struct iv *a, const struct iv *b, struct iv *c, int n)
{
	int k, i;

	for (k = 0; k < n; k++) {
		struct iv s = ivp(0);
		for (i = 0; i <= k; i++)
			s = iadd(s, imul(a[i], b[k - i]));
		c[k] = s;
	}
	if (n > 0 && a == b)
		c[0] = ipowi(a[0], 2);	/* a square, not a product */
}

static int
jdiv(const struct iv *a, const struct iv *b, struct iv *c, int n)
{
	int k, i;

	if (has0(b[0]))
		return 0;
	for (k = 0; k < n; k++) {
		struct iv s = a[k];
		for (i = 1; i <= k; i++)
			s = isub(s, imul(b[i], c[k - i]));
		c[k] = idiv(s, b[0]);
	}
	return 1;
}

static void
jsincos(const struct iv *a, struct iv *s, struct iv *c, int n)
{
	int k, i;

	s[0] = el_sincos(a[0], 0);
	c[0] = el_sincos(a[0], 1);
	for (k = 1; k < n; k++) {
		struct iv ss = ivp(0), cc = ivp(0);
		for (i = 1; i <= k; i++) {
			ss = iadd(ss, iscale(imul(a[i], c[k - i]), i));
			cc = iadd(cc, iscale(imul(a[i], s[k - i]), i));
		}
		s[k] = idiv(ss, ivp(k));
		c[k] = ineg(idiv(cc, ivp(k)));
	}
}

static void
jsinhcosh(const struct iv *a, struct iv *s, struct iv *c, int n)
{
	int k, i;

	s[0] = el_mono(O_SINH, a[0], 1);
	c[0] = a[0].lo >= 0 ? el_mono(O_COSH, a[0], 1) : a[0].hi <= 0 ?
	    el_mono(O_COSH, a[0], -1) : iv2(1, fmaxl(el_r(O_COSH, a[0].lo, 1),
	    el_r(O_COSH, a[0].hi, 1)));
	for (k = 1; k < n; k++) {
		struct iv ss = ivp(0), cc = ivp(0);
		for (i = 1; i <= k; i++) {
			ss = iadd(ss, iscale(imul(a[i], c[k - i]), i));
			cc = iadd(cc, iscale(imul(a[i], s[k - i]), i));
		}
		s[k] = idiv(ss, ivp(k));
		c[k] = idiv(cc, ivp(k));
	}
}

/* o = F(a), F' = 1/sqrt(q) (sq) or 1/q, o[0] = f0 */
static int
jinv(const struct iv *a, const struct iv *q, int sq, struct iv f0,
    struct iv *o, int n, struct iv *t1, struct iv *t2, struct iv *t3)
{
	int i, j;

	if (sq) {
		if (!(q[0].lo > 0))
			return 0;
		t1[0] = iv2(sqrt_r(q[0].lo, 0), sqrt_r(q[0].hi, 1));
		for (j = 1; j < n; j++) {
			struct iv s = q[j];
			for (i = 1; i < j; i++)
				s = isub(s, imul(t1[i], t1[j - i]));
			t1[j] = idiv(s, iscale(t1[0], 2));
		}
	} else
		memcpy(t1, q, n * sizeof *t1);
	for (i = 0; i + 1 < n; i++)
		t2[i] = iscale(a[i + 1], i + 1);
	t2[n - 1] = ivp(0);
	if (!jdiv(t2, t1, t3, n))
		return 0;
	o[0] = f0;
	for (i = 1; i < n; i++)
		o[i] = idiv(t3[i - 1], ivp(i));
	return 1;
}

static int
cmp_dec(int op, struct iv a, struct iv b)
{
	switch (op) {
	case O_LT:
		return a.hi < b.lo ? 1 : a.lo >= b.hi ? 0 : -1;
	case O_LE:
		return a.hi <= b.lo ? 1 : a.lo > b.hi ? 0 : -1;
	case O_GT:
		return a.lo > b.hi ? 1 : a.hi <= b.lo ? 0 : -1;
	case O_GE:
		return a.lo >= b.hi ? 1 : a.hi < b.lo ? 0 : -1;
	case O_EQ:
		return a.lo == a.hi && b.lo == b.hi && a.lo == b.lo ? 1 :
		    a.hi < b.lo || b.hi < a.lo ? 0 : -1;
	case O_NE:
		return a.lo == a.hi && b.lo == b.hi && a.lo == b.lo ? 0 :
		    a.hi < b.lo || b.hi < a.lo ? 1 : -1;
	}
	return -1;
}

/* the sign of a on the piece: 1, -1, or 0 undecided; an end at -0 or
 * +0 counts with its sign (x = -0 and +0 are the pieces' ends) */
static int
sign_dec(struct iv a)
{
	if (a.lo > 0 || (a.lo == 0 && !signbit(a.lo) && a.hi >= 0 &&
	    !(a.hi == 0 && signbit(a.hi))))
		return 1;
	if (a.hi < 0 || (a.hi == 0 && signbit(a.hi)))
		return -1;
	return 0;
}

struct jctx {
	struct iv *J;		/* n per node */
	struct iv *t[5];	/* scratch, ACERT_K_MAX + 1 each */
	char why[200];
};

#define JN(P, q) (C->J + (size_t)(q) * (ACERT_K_MAX + 1))

/*
 * The first n Taylor coefficients of every node of P at x (an interval:
 * of every point of it).  decide: the branches are decided here (from
 * the values on x), else P->dec holds them.  0 (why) when a branch is
 * undecided on x or an operation is undefined somewhere on it.
 */
static int
prog_jets(struct prog *P, struct jctx *C, struct iv x, int n, int decide)
{
	int k, i, j;
	struct iv *t1 = C->t[0], *t2 = C->t[1], *t3 = C->t[2], *t4 = C->t[3];

	for (k = 0; k < P->n; k++) {
		struct iv *o = JN(P, k), *a = JN(P, P->a[k]), *b = JN(P, P->b[k]);
		int op = P->op[k], d;
		for (i = 0; i < n; i++)
			o[i] = ivp(0);
		P->bad[k] = 0;
		if (P->fz[k]) {
			/* its computed value is that number on the whole piece */
			o[0] = ivp(P->fv[k]);
			continue;
		}
		if (op == O_ASFLOAT && P->xsrc[k] >= 0) {
			/* +-2^xs times its source, exactly */
			const struct iv *s = JN(P, P->xsrc[k]);
			ld sc = ldexpl(P->xneg[k] ? -1 : 1, P->xs[k]);
			if (P->bad[P->xsrc[k]]) {
				P->bad[k] = 1;
				continue;
			}
			for (i = 0; i < n; i++)
				o[i] = iscale(s[i], sc);
			continue;
		}
		if (opbad(P, k)) {
			P->bad[k] = 1;
			continue;
		}
		switch (op) {
		case O_CONST:
			o[0] = ivp(P->k[k]);
			if (decide)
				P->dec[k] = P->k[k] != 0;	/* a constant condition */
			break;
		case O_VAR:
			o[0] = x;
			if (n > 1)
				o[1] = ivp(1);
			break;
		case O_CAST:
		case O_BARRIER:
			memcpy(o, a, n * sizeof *o);
			break;
		case O_NEG:
			for (i = 0; i < n; i++)
				o[i] = ineg(a[i]);
			break;
		case O_ADD:
			for (i = 0; i < n; i++)
				o[i] = iadd(a[i], b[i]);
			break;
		case O_SUB:
			for (i = 0; i < n; i++)
				o[i] = isub(a[i], b[i]);
			break;
		case O_MUL:
			jmul(a, b, o, n);
			break;
		case O_FMA:
			jmul(a, b, o, n);
			for (i = 0; i < n; i++)
				o[i] = iadd(o[i], JN(P, P->c[k])[i]);
			break;
		case O_DIV:
			if (!jdiv(a, b, o, n)) {
				snprintf(P->why[k], sizeof P->why[k], "a divisor can be 0");
				goto bad;
			}
			break;
		case O_EXP:
			o[0] = el_mono(O_EXP, a[0], 1);
			for (j = 1; j < n; j++) {
				struct iv s = ivp(0);
				for (i = 1; i <= j; i++)
					s = iadd(s, iscale(imul(a[i], o[j - i]), i));
				o[j] = idiv(s, ivp(j));
			}
			break;
		case O_LOG:
			if (!(a[0].lo > 0)) {
				snprintf(P->why[k], sizeof P->why[k], "log's argument can "
				    "be 0 or less");
				goto bad;
			}
			o[0] = el_mono(O_LOG, a[0], 1);
			for (j = 1; j < n; j++) {
				struct iv s = ivp(0);
				for (i = 1; i < j; i++)
					s = iadd(s, iscale(imul(o[i], a[j - i]), i));
				o[j] = idiv(isub(a[j], idiv(s, ivp(j))), a[0]);
			}
			break;
		case O_SQRT:
			if (!(a[0].lo > 0 || (n == 1 && a[0].lo >= 0))) {
				snprintf(P->why[k], sizeof P->why[k], "sqrt's argument can "
				    "be 0 or less");
				goto bad;
			}
			o[0] = iv2(sqrt_r(a[0].lo, 0), sqrt_r(a[0].hi, 1));
			for (j = 1; j < n; j++) {
				struct iv s = a[j];
				for (i = 1; i < j; i++)
					s = isub(s, imul(o[i], o[j - i]));
				o[j] = idiv(s, iscale(o[0], 2));
			}
			break;
		case O_SIN: case O_COS: case O_TAN: case O_SEC: case O_CSC:
		case O_COT:
			jsincos(a, t1, t2, n);
			if (op == O_SIN)
				memcpy(o, t1, n * sizeof *o);
			else if (op == O_COS)
				memcpy(o, t2, n * sizeof *o);
			else {
				int ok;
				if (op == O_TAN && crit_in(a[0], PI_L / 2, -1)) {
					snprintf(P->why[k], sizeof P->why[k], "tan has a pole "
					    "there");
					goto bad;
				}
				for (i = 0; i < n; i++)
					t3[i] = ivp(i == 0);
				ok = op == O_TAN ? jdiv(t1, t2, o, n) : op == O_COT ?
				    jdiv(t2, t1, o, n) : jdiv(t3, op == O_SEC ? t2 : t1,
				    o, n);
				if (!ok) {
					snprintf(P->why[k], sizeof P->why[k], "%s has a pole "
					    "there", opinfo[op].name);
					goto bad;
				}
				if (op == O_TAN)
					o[0] = el_mono(O_TAN, a[0], 1);
			}
			break;
		case O_ATAN:
			jmul(a, a, t1, n);
			t1[0] = iadd(t1[0], ivp(1));
			for (i = 0; i + 1 < n; i++)
				t2[i] = iscale(a[i + 1], i + 1);
			t2[n - 1] = ivp(0);
			jdiv(t2, t1, t3, n);
			o[0] = el_mono(O_ATAN, a[0], 1);
			for (i = 1; i < n; i++)
				o[i] = idiv(t3[i - 1], ivp(i));
			break;
		case O_SINH: case O_COSH: case O_TANH:
			jsinhcosh(a, t1, t2, n);
			if (op == O_SINH)
				memcpy(o, t1, n * sizeof *o);
			else if (op == O_COSH)
				memcpy(o, t2, n * sizeof *o);
			else {
				jdiv(t1, t2, o, n);
				o[0] = el_mono(O_TANH, a[0], 1);
			}
			break;
		case O_ASIN: case O_ACOS: case O_ASINH: case O_ACOSH:
		case O_ATANH: {
			struct iv f0;
			int ok = 1;
			if (op == O_ASIN || op == O_ACOS)
				ok = a[0].lo >= -1 && a[0].hi <= 1 && (n == 1 ||
				    (a[0].lo > -1 && a[0].hi < 1));
			else if (op == O_ATANH)
				ok = a[0].lo > -1 && a[0].hi < 1;
			else if (op == O_ACOSH)
				ok = a[0].lo >= 1 && (n == 1 || a[0].lo > 1);
			if (!ok) {
				snprintf(P->why[k], sizeof P->why[k], "%s's argument leaves "
				    "its domain there", opinfo[op].name);
				goto bad;
			}
			f0 = el_mono(op, a[0], op == O_ACOS ? -1 : 1);
			if (n == 1) {
				o[0] = f0;
				break;
			}
			jmul(a, a, t4, n);
			if (op == O_ASINH)
				t4[0] = iadd(t4[0], ivp(1));
			else if (op == O_ACOSH)
				t4[0] = isub(t4[0], ivp(1));
			else {
				for (i = 0; i < n; i++)
					t4[i] = ineg(t4[i]);
				t4[0] = iadd(t4[0], ivp(1));
			}
			if (!jinv(a, t4, op != O_ATANH, f0, o, n, t1, t2, t3)) {
				snprintf(P->why[k], sizeof P->why[k], "%s's derivative is "
				    "unbounded there", opinfo[op].name);
				goto bad;
			}
			if (op == O_ACOS)
				for (i = 1; i < n; i++)
					o[i] = ineg(o[i]);
			break;
		}
		case O_POW: {
			ld p = P->k[P->b[k]];
			if (p == floorl(p) && p >= 0 && p <= 64) {
				int e = (int)p;
				for (i = 0; i < n; i++)
					t1[i] = ivp(i == 0);
				for (j = 0; j < e; j++) {
					jmul(t1, a, t2, n);
					memcpy(t1, t2, n * sizeof *t1);
				}
				memcpy(o, t1, n * sizeof *o);
				o[0] = ipowi(a[0], e);
			} else if (a[0].lo > 0) {
				struct iv lg = el_mono(O_LOG, a[0], 1);
				struct iv pl = iscale(lg, p);
				o[0] = el_mono(O_EXP, pl, 1);
				for (j = 1; j < n; j++) {
					struct iv s = ivp(0);
					for (i = 1; i <= j; i++)
						s = iadd(s, iscale(imul(a[i], o[j - i]),
						    (p + 1) * i - j));
					o[j] = idiv(s, iscale(a[0], j));
				}
			} else {
				snprintf(P->why[k], sizeof P->why[k], "pow's base can be 0 "
				    "or less");
				goto bad;
			}
			break;
		}
		case O_ABS:
		case O_COPYSIGN:
		case O_SIGN:
		case O_FLOOR:
		case O_CEIL:
		case O_MIN:
		case O_MAX:
			if (decide) {
				if (op == O_ABS)
					d = sign_dec(a[0]);
				else if (op == O_COPYSIGN) {
					int sa = sign_dec(a[0]), sb = sign_dec(b[0]);
					d = sa && sb ? sa * sb : 0;
					if (is0(a[0]) && n == 1)
						d = 1;
				} else if (op == O_SIGN)
					d = is0(a[0]) ? 2 : a[0].lo > 0 ? 1 :
					    a[0].hi < 0 ? -1 : 0;
				else if (op == O_FLOOR || op == O_CEIL) {
					ld f1 = op == O_FLOOR ? floorl(a[0].lo) :
					    ceill(a[0].lo), f2 = op == O_FLOOR ?
					    floorl(a[0].hi) : ceill(a[0].hi);
					d = f1 == f2 && !(n > 1 && f1 == a[0].lo &&
					    f1 == a[0].hi) ? 1 : 0;
				} else {
					d = a[0].hi <= b[0].lo ? 1 : b[0].hi <= a[0].lo ?
					    -1 : 0;
					if (op == O_MAX && d)
						d = -d;
				}
				P->dec[k] = (signed char)d;
			}
			d = P->dec[k];
			if (d == 0) {
				snprintf(P->why[k], sizeof P->why[k], "%s is undecided "
				    "there", opinfo[op].name);
				goto bad;
			}
			if (op == O_ABS || op == O_COPYSIGN)
				for (i = 0; i < n; i++)
					o[i] = d > 0 ? a[i] : ineg(a[i]);
			else if (op == O_SIGN)
				o[0] = ivp(d == 2 ? 0 : d);
			else if (op == O_FLOOR || op == O_CEIL)
				o[0] = ivp(op == O_FLOOR ? floorl(a[0].lo) :
				    ceill(a[0].lo));
			else
				memcpy(o, d > 0 ? a : b, n * sizeof *o);
			break;
		case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT:
		case O_GE: case O_LAND: case O_LOR: case O_LNOT:
			if (decide) {
				if (op == O_LAND)
					d = P->dec[P->a[k]] == 0 || P->dec[P->b[k]] == 0 ?
					    0 : P->dec[P->a[k]] == 1 && P->dec[P->b[k]] ==
					    1 ? 1 : -1;
				else if (op == O_LOR)
					d = P->dec[P->a[k]] == 1 || P->dec[P->b[k]] == 1 ?
					    1 : P->dec[P->a[k]] == 0 && P->dec[P->b[k]] ==
					    0 ? 0 : -1;
				else if (op == O_LNOT)
					d = P->dec[P->a[k]] < 0 ? -1 : !P->dec[P->a[k]];
				else
					d = cmp_dec(op, a[0], b[0]);
				P->dec[k] = (signed char)d;
			}
			d = P->dec[k];
			if (d < 0) {
				snprintf(P->why[k], sizeof P->why[k], "a comparison is "
				    "undecided there");
				goto bad;
			}
			o[0] = ivp(d);
			break;
		case O_SEL:
			d = P->dec[P->a[k]];
			if (d < 0) {
				snprintf(P->why[k], sizeof P->why[k], "a select's condition "
				    "is undecided there");
				goto bad;
			}
			memcpy(o, d ? b : JN(P, P->c[k]), n * sizeof *o);
			break;
		default:
			snprintf(P->why[k], sizeof P->why[k], "%s has no interval rule",
			    opinfo[op].name);
			goto bad;
		}
		for (i = 0; i < n; i++)
			if (!isfinite(o[i].lo) || !isfinite(o[i].hi)) {
				snprintf(P->why[k], sizeof P->why[k], "a value is unbounded "
				    "there");
				goto bad;
			}
		continue;
bad:
		P->bad[k] = 1;
	}
	if (P->bad[P->root]) {
		bad_why(P, C->why, sizeof C->why);
		return 0;
	}
	return 1;
}

/* ---- the candidate's rounding ---- */

struct rnode {
	struct iv v;		/* the exact value on the piece */
	struct iv w;		/* v = t^p w */
	int p;
	ld rho, eta;		/* |computed - v| <= rho |v| + eta */
	ld rho0, eta0;		/* before the node's own rounding */
	int rounds;		/* this node rounds (not exact on the piece) */
	ld q;			/* every computed value is a multiple of q
				 * (INFINITY: it is 0) */
	int iok;		/* an integer: its values are ilo..ihi, in its
				 * type's order */
	int64_t ilo, ihi;
};

struct rctx {
	struct rnode *R;
	struct iv T;		/* x - z on the piece */
	int hasz;
	ld z;
	int mexp;		/* the candidate rounds a machine exp */
	char why[200];
};

/* |t|^e on the piece, and t^e (signed) */
static struct iv
tpow_abs(const struct rctx *c, int e)
{
	return ipowi(iabs(c->T), e);
}

static struct iv
tpow(const struct rctx *c, int e)
{
	return ipowi(c->T, e);
}

static ld
rmag(const struct rctx *c, const struct rnode *r)
{
	return fminl(mag(r->v), mul_r(mag(tpow_abs(c, r->p)), mag(r->w), 1));
}

static ld
rmig(const struct rctx *c, const struct rnode *r)
{
	return fmaxl(mig(r->v), mul_r(mig(tpow_abs(c, r->p)), mig(r->w), 0));
}

static int
fmt_prec(int w)
{
	return w == 32 ? 24 : 53;
}

static int
fmt_emin(int w)
{
	return w == 32 ? -126 : -1022;
}

/* the format's unit roundoff, least normal, half the least subnormal */
static void
fmt_consts(int w, ld *u, ld *nmin, ld *eta)
{
	*u = ldexpl(1, -fmt_prec(w));
	*nmin = ldexpl(1, fmt_emin(w));
	*eta = ldexpl(1, fmt_emin(w) - fmt_prec(w));
}

static ld
fmt_max(int w)
{
	return w == 32 ? (ld)FLT_MAX : (ld)DBL_MAX;
}

/* v rounded to the format (to nearest) */
static ld
rn_fmt(int w, ld v)
{
	return w == 32 ? (ld)(float)v : (ld)(double)v;
}

/* the spacing of the format's floats at |v| (the least subnormal's at 0) */
static ld
ulp_w(int w, ld v)
{
	int e = v == 0 ? fmt_emin(w) : ilogbl(fabsl(v));

	if (e < fmt_emin(w))
		e = fmt_emin(w);
	return ldexpl(1, e - fmt_prec(w) + 1);
}

/* the largest power of two a float v of the format is a multiple of */
static ld
quant_of(int w, ld v)
{
	ld q, m;

	if (v == 0)
		return INFINITY;
	q = ulp_w(w, v);
	m = fabsl(v) / q;	/* a whole number below 2^prec */
	while (fmodl(m, 2) == 0) {
		m /= 2;
		q *= 2;
	}
	return q;
}

/* the computed values of r: its exact range widened by its error */
static struct iv
computed(const struct rnode *r)
{
	ld e = add_r(mul_r(r->rho, mag(r->v), 1), r->eta, 1);

	return iv2(add_r(r->v.lo, -e, 0), add_r(r->v.hi, e, 1));
}

/* (1 + a)(1 + b) - 1, rounded up */
static ld
rel_comb(ld a, ld b)
{
	return add_r(add_r(a, b, 1), mul_r(a, b, 1), 1);
}

/* the sum of a and b (each t^p w, errors rho, eta) before rounding:
 * value, cofactor, and the error relative to the sum */
static void
sum_step(const struct rctx *c, const struct rnode *a, const struct rnode *b,
    int neg, struct rnode *o)
{
	struct iv wa, wb;
	ld ra, rb;
	int sa, sb;

	o->v = neg ? isub(a->v, b->v) : iadd(a->v, b->v);
	o->p = a->p < b->p ? a->p : b->p;
	wa = imul(a->w, tpow(c, a->p - o->p));
	wb = imul(b->w, tpow(c, b->p - o->p));
	if (neg)
		wb = ineg(wb);
	o->w = iadd(wa, wb);
	o->eta0 = add_r(a->eta, b->eta, 1);
	o->q = fminl(a->q, b->q);
	/* the same sign: |a| + |b| = |a + b| */
	sa = sign_dec(a->v);
	sb = sign_dec(b->v);
	if (neg)
		sb = -sb;
	ra = mul_r(a->rho, mul_r(mag(tpow_abs(c, a->p - o->p)), mag(a->w), 1),
	    1);
	rb = mul_r(b->rho, mul_r(mag(tpow_abs(c, b->p - o->p)), mag(b->w), 1),
	    1);
	if (mig(o->w) > 0 && mag(o->w) <= 2 * mig(o->w)) {
		/* (rho_a |a| + rho_b |b|) / |a + b|; of one sign, also at most
		 * the larger rho */
		o->rho0 = div_r(add_r(ra, rb, 1), mig(o->w), 1);
		if (sa && sa == sb)
			o->rho0 = fminl(o->rho0, fmaxl(a->rho, b->rho));
	} else if (sa && sa == sb)
		o->rho0 = fmaxl(a->rho, b->rho);
	else {
		/* a sum that varies more than twice over the piece (near a
		 * cancellation, or 0 in it): absolute, or its least value would
		 * scale the error of its largest */
		o->rho0 = 0;
		o->eta0 = add_r(o->eta0, add_r(mul_r(a->rho, rmag(c, a), 1),
		    mul_r(b->rho, rmag(c, b), 1), 1), 1);
	}
}

/* a constant node of q's type holding its frozen value */
static nid
const_of(const struct prog *P, int q)
{
	ty_t t = N.ty[P->g[q]];

	if (!TY_FP(t))
		return mk_const_i(t, P->fi[q]);
	if (TYK(t) == K_FLOAT)
		return mk_cast(t, mk_const_r((double)P->fv[q]));
	return mk_const_r((double)P->fv[q]);
}

/*
 * Node k's value from its operands' frozen values, by ucas's own
 * constant folding, so exactly what the program computes there; 0 when
 * it does not fold (undefined, or not a constant).
 */
static int
fold_node(struct prog *P, int k)
{
	nid g = P->g[k], r, o[3] = {0, 0, 0};
	int op = P->op[k], i, ar = opinfo[op].arity;

	if (op == O_CONST)
		r = g;
	else if (op == O_VAR)
		return 0;
	else {
		int src[3] = {P->a[k], P->b[k], P->c[k]};
		for (i = 0; i < ar && i < 3; i++) {
			if (!P->fz[src[i]])
				return 0;
			o[i] = const_of(P, src[i]);
		}
		r = op == O_CAST ? mk_cast(N.ty[g], o[0]) : mk_like(g, o[0], o[1],
		    o[2]);
	}
	if (N.op[r] != O_CONST)
		return 0;
	P->fz[k] = 1;
	if (TY_FP(N.ty[r])) {
		P->fv[k] = N.kr[r];
		P->fi[k] = 0;
	} else {
		P->fi[k] = N.ki[r];
		P->fv[k] = (ld)N.ki[r];
	}
	return 1;
}

/* an integer type's range and order: 0 when it is not one of u8..u64,
 * i8..i64, bool */
static int
int_range(ty_t t, int64_t *lo, int64_t *hi)
{
	int w = TYW(t);

	if (TYK(t) == K_BOOL) {
		*lo = 0;
		*hi = 1;
		return 1;
	}
	if (TYK(t) == K_UINT && w <= 63) {
		*lo = 0;
		*hi = (int64_t)(((uint64_t)1 << w) - 1);
		return 1;
	}
	if (TYK(t) == K_UINT && w == 64) {
		*lo = 0;		/* the half below 2^63 */
		*hi = INT64_MAX;
		return 1;
	}
	if (TYK(t) == K_SINT && w <= 64) {
		*lo = w == 64 ? INT64_MIN : -((int64_t)1 << (w - 1));
		*hi = w == 64 ? INT64_MAX : ((int64_t)1 << (w - 1)) - 1;
		return 1;
	}
	return 0;
}

/* the interval [lo, hi] of mathematical values in type t: as is, or
 * moved by 2^w when all of it wrapped the same way (unsigned) */
static int
int_fit(ty_t t, ld lo, ld hi, int64_t *ilo, int64_t *ihi)
{
	int64_t tlo, thi;

	if (!int_range(t, &tlo, &thi) || lo > hi)
		return 0;
	if (TYK(t) == K_UINT && TYW(t) <= 32) {
		ld m = ldexpl(1, TYW(t));
		if (hi < 0) {
			lo += m;
			hi += m;
		} else if (lo > thi) {
			lo -= m;
			hi -= m;
		}
	}
	if (lo < (ld)tlo || hi > (ld)thi)
		return 0;
	*ilo = (int64_t)lo;
	*ihi = (int64_t)hi;
	return 1;
}

/* the bits of a float node's computed values, all of one sign, as an
 * interval of its unsigned type */
static int
bits_range(int w, struct iv cv, int64_t *lo, int64_t *hi)
{
	ld a, b;

	if (cv.lo >= 0) {
		a = rn_fmt(w, cv.lo);
		if (a < cv.lo)
			a = w == 32 ? (ld)nextafterf((float)a, INFINITY) :
			    (ld)nextafter((double)a, INFINITY);
		b = rn_fmt(w, cv.hi);
		if (b > cv.hi)
			b = w == 32 ? (ld)nextafterf((float)b, -INFINITY) :
			    (ld)nextafter((double)b, -INFINITY);
		if (a > b || isinf(b))
			return 0;
		*lo = (int64_t)fbits(w, (double)a);
		*hi = (int64_t)fbits(w, (double)b);
		return w == 32 || *hi >= 0;
	}
	return 0;
}

/* an integer node's interval from its operands', where the operation is
 * monotone; 0 when it is not followed */
static int
int_step(const struct prog *P, const struct rctx *c, int k, struct rnode *o)
{
	const struct rnode *a = &c->R[P->a[k]];
	ty_t t = N.ty[P->g[k]];
	int op = P->op[k];
	int64_t bv = P->fz[P->b[k]] ? P->fi[P->b[k]] : 0;

	switch (op) {
	case O_ASUINT:
	case O_ASINT: {
		int w = P->fw[P->a[k]];
		return w && bits_range(w, computed(a), &o->ilo, &o->ihi);
	}
	case O_ADD:
	case O_SUB:
		if (!a->iok || !P->fz[P->b[k]])
			return 0;
		return op == O_ADD ? int_fit(t, (ld)a->ilo + bv, (ld)a->ihi + bv,
		    &o->ilo, &o->ihi) : int_fit(t, (ld)a->ilo - bv, (ld)a->ihi -
		    bv, &o->ilo, &o->ihi);
	case O_SHR:
		if (!a->iok || !P->fz[P->b[k]] || bv < 0 || bv >= TYW(t))
			return 0;
		/* unsigned: logical; signed: arithmetic, both monotone */
		o->ilo = a->ilo >> bv;
		o->ihi = a->ihi >> bv;
		return 1;
	case O_SHL:
		if (!a->iok || !P->fz[P->b[k]] || bv < 0 || bv >= 62)
			return 0;
		return int_fit(t, ldexpl((ld)a->ilo, (int)bv), ldexpl((ld)a->ihi,
		    (int)bv), &o->ilo, &o->ihi);
	case O_AND:
		/* x & m is one value where every x shares the bits m covers
		 * from its lowest up */
		if (a->iok && P->fz[P->b[k]] && bv > 0) {
			int tz = 0;
			while (!((bv >> tz) & 1))
				tz++;
			if ((a->ilo >> tz) == (a->ihi >> tz)) {
				o->ilo = o->ihi = int_norm(t, a->ilo & bv);
				return 1;
			}
		}
		return 0;
	case O_CAST:
		if (!a->iok || !TY_ISINT(t))
			return 0;
		return int_fit(t, (ld)a->ilo, (ld)a->ihi, &o->ilo, &o->ihi) ||
		    (TYK(t) == K_SINT && int_fit(t, (ld)a->ilo - ldexpl(1,
		    TYW(t)), (ld)a->ihi - ldexpl(1, TYW(t)), &o->ilo, &o->ihi));
	}
	return 0;
}

/*
 * The rounding errors of the candidate's nodes on the piece (x in X),
 * and its branches decided on the computed values.  A node whose
 * computed value is one number on the whole piece is frozen to it (the
 * reduction's integer side, the shifter's round, the 2^k); a float's
 * bits moved by a frozen multiple of 2^mant and read back are that float
 * times a power of two.  0 (why) when a node the result needs cannot be
 * evaluated.
 */
static int
prog_round(struct prog *P, struct rctx *c, struct iv X)
{
	int k, i;

	for (k = 0; k < P->n; k++) {
		struct rnode *o = &c->R[k], *a = &c->R[P->a[k]], *b = &c->R[P->b[k]];
		int op = P->op[k], w = P->fw[k], d, intk;
		ld u = 0, nmin = 0, eta = 0;
		ty_t t = N.ty[P->g[k]];

		if (w)
			fmt_consts(w, &u, &nmin, &eta);
		memset(o, 0, sizeof *o);
		P->bad[k] = 0;
		P->fz[k] = 0;
		P->xsrc[k] = -1;
		P->xs[k] = 0;
		P->xneg[k] = 0;
		P->why[k][0] = 0;
		intk = !TY_FP(t);
		/* every operand one number: the node is one number too */
		if (op != O_VAR && op != O_SEL) {
			int all = 1, src[3] = {P->a[k], P->b[k], P->c[k]};
			for (i = 0; i < opinfo[op].arity && i < 3; i++)
				all &= P->fz[src[i]];
			if (all && fold_node(P, k)) {
				o->v = o->w = ivp(P->fv[k]);
				o->q = w ? quant_of(w, P->fv[k]) : 0;
				o->iok = intk;
				o->ilo = o->ihi = P->fi[k];
				if (op == O_CONST || TYK(t) == K_BOOL || (op >= O_EQ &&
				    op <= O_LOR) || op == O_LNOT)
					P->dec[k] = (signed char)(P->fv[k] != 0);
				continue;
			}
		}
		if (opbad(P, k)) {
			P->bad[k] = 1;
			continue;
		}
		if (intk && op != O_EQ && op != O_NE && op != O_LT && op != O_LE &&
		    op != O_GT && op != O_GE && op != O_LAND && op != O_LOR &&
		    op != O_LNOT) {
			/* the integer side: an interval, frozen when a point; the
			 * bits of a float keep where they came from */
			o->iok = int_step(P, c, k, o);
			if ((op == O_ASUINT || op == O_ASINT) && P->fw[P->a[k]])
				P->xsrc[k] = P->a[k];
			else if ((op == O_ADD || op == O_SUB || op == O_XOR) &&
			    P->xsrc[P->a[k]] >= 0 && P->fz[P->b[k]]) {
				int fwv = P->fw[P->xsrc[P->a[k]]], mant = fmt_prec(fwv) - 1;
				int64_t cv = P->fi[P->b[k]];
				int64_t cs = TYW(t) < 64 && cv >= ((int64_t)1 << (TYW(t) -
				    1)) ? cv - ((int64_t)1 << TYW(t)) : cv;
				P->xsrc[k] = P->xsrc[P->a[k]];
				P->xs[k] = P->xs[P->a[k]];
				P->xneg[k] = P->xneg[P->a[k]];
				if (op == O_XOR && cv == 0)
					;
				else if (op == O_XOR && TYW(t) == fwv &&
				    (uint64_t)cv == (uint64_t)1 << (fwv - 1))
					P->xneg[k] ^= 1;
				else if (op != O_XOR && cs % ((int64_t)1 << mant) == 0)
					P->xs[k] += (int)((op == O_ADD ? cs : -cs) >>
					    mant);
				else
					P->xsrc[k] = -1;
			}
			if (o->iok && o->ilo == o->ihi) {
				P->fz[k] = 1;
				P->fi[k] = o->ilo;
				P->fv[k] = (ld)o->ilo;
				o->v = o->w = ivp((ld)o->ilo);
			}
			if (!o->iok && P->xsrc[k] < 0) {
				snprintf(P->why[k], sizeof P->why[k], "the integers' `%s` "
				    "is not one value there, nor a float's bits moved by a "
				    "power of two", opinfo[op].name);
				goto bad;
			}
			continue;
		}
		switch (op) {
		case O_CONST:
			o->v = o->w = ivp(P->k[k]);
			o->q = w ? quant_of(w, P->k[k]) : 0;
			P->dec[k] = P->k[k] != 0;	/* a constant condition */
			break;
		case O_VAR:
			o->v = X;
			if (c->hasz && c->z == 0) {
				o->p = 1;
				o->w = ivp(1);
			} else
				o->w = X;
			o->q = w ? (mig(X) > 0 ? ulp_w(w, mig(X)) : ulp_w(w, 0)) : 0;
			break;
		case O_ASFLOAT: {
			/* a float's bits moved by a frozen multiple of 2^mant and
			 * read back: +-2^xs times it, while every computed value
			 * keeps a normal exponent */
			int src = P->xsrc[P->a[k]], ebits, e1, e2;
			const struct rnode *sr;
			struct iv cv;
			ld sc;
			if (src < 0 || P->fw[src] != w) {
				snprintf(P->why[k], sizeof P->why[k], "asfloat of bits "
				    "that are not a float's moved by a power of two");
				goto bad;
			}
			sr = &c->R[src];
			cv = computed(sr);
			ebits = w == 32 ? 8 : 11;
			if (P->xs[P->a[k]] != 0) {
				if (!(mig(cv) >= ldexpl(1, fmt_emin(w)))) {
					snprintf(P->why[k], sizeof P->why[k], "a float moved by "
					    "2^%d can be 0 or subnormal there", P->xs[P->a[k]]);
					goto bad;
				}
				e1 = ilogbl(mig(cv)) + (1 << (ebits - 1)) - 1;
				e2 = ilogbl(mag(cv)) + (1 << (ebits - 1)) - 1;
				if (e1 + P->xs[P->a[k]] < 1 || e2 + P->xs[P->a[k]] >
				    (1 << ebits) - 2) {
					snprintf(P->why[k], sizeof P->why[k], "a float moved by "
					    "2^%d leaves the normal exponents there",
					    P->xs[P->a[k]]);
					goto bad;
				}
			}
			P->xsrc[k] = src;
			P->xs[k] = P->xs[P->a[k]];
			P->xneg[k] = P->xneg[P->a[k]];
			sc = ldexpl(P->xneg[k] ? -1 : 1, P->xs[k]);
			*o = *sr;
			o->v = iscale(sr->v, sc);
			o->w = iscale(sr->w, sc);
			o->eta = mul_r(sr->eta, fabsl(sc), 1);
			o->rho0 = o->rho;
			o->eta0 = o->eta;
			o->q = mul_r(sr->q, fabsl(sc), 0);
			o->rounds = 0;
			break;
		}
		case O_CAST:
		case O_BARRIER:
			if (op == O_CAST && !TY_FP(N.ty[P->g[P->a[k]]])) {
				snprintf(P->why[k], sizeof P->why[k], "an integer that is "
				    "not one value there, read as a float");
				goto bad;
			}
			*o = *a;
			o->rho0 = a->rho;
			o->eta0 = a->eta;
			/* a cast to a narrower float rounds */
			o->rounds = op == O_CAST && w && w < 64 && (P->fw[P->a[k]] ==
			    0 || P->fw[P->a[k]] > w);
			break;
		case O_NEG:
			*o = *a;
			o->v = ineg(a->v);
			o->w = ineg(a->w);
			o->rounds = 0;
			break;
		case O_ABS:
		case O_COPYSIGN:
			*o = *a;
			o->rounds = 0;
			d = sign_dec(a->v);
			if (op == O_COPYSIGN) {
				int sb = sign_dec(b->v);
				/* the sign operand is exact in approx's candidates */
				if (b->rho != 0 || b->eta != 0)
					sb = 0;
				if (is0(a->v) && sb)
					d = 1;
				d = d && sb ? d * sb : 0;
			}
			if (d == 0) {
				snprintf(P->why[k], sizeof P->why[k], "%s is undecided "
				    "there", opinfo[op].name);
				goto bad;
			}
			P->dec[k] = (signed char)d;
			if (d < 0) {
				o->v = ineg(a->v);
				o->w = ineg(a->w);
			}
			break;
		case O_ADD:
		case O_SUB:
			sum_step(c, a, b, op == O_SUB, o);
			if (c->hasz && c->z != 0 && a->p == 0 && b->p == 0 &&
			    (P->op[P->a[k]] == O_VAR || (P->op[P->a[k]] == O_ASFLOAT &&
			    P->xsrc[P->a[k]] >= 0 && P->op[P->xsrc[P->a[k]]] == O_VAR &&
			    P->xs[P->a[k]] == 0 && !P->xneg[P->a[k]])) &&
			    P->fz[P->b[k]] && P->fv[P->b[k]] == (op == O_SUB ? c->z :
			    -c->z)) {
				/* x - z: t itself */
				o->p = 1;
				o->w = ivp(1);
			}
			o->rounds = w != 0;
			break;
		case O_MUL:
		case O_FMA:
			o->v = imul(a->v, b->v);
			o->p = a->p + b->p;
			o->w = imul(a->w, b->w);
			o->rho0 = rel_comb(a->rho, b->rho);
			o->eta0 = add_r(add_r(mul_r(mul_r(rmag(c, a), 1 + a->rho, 1),
			    b->eta, 1), mul_r(mul_r(rmag(c, b), 1 + b->rho, 1), a->eta,
			    1), 1), mul_r(a->eta, b->eta, 1), 1);
			o->q = mul_r(a->q, b->q, 0);
			o->rounds = w != 0;
			if (op == O_FMA) {
				struct rnode ab = *o, s;
				ab.rho = ab.rho0;
				ab.eta = ab.eta0;
				sum_step(c, &ab, &c->R[P->c[k]], 0, &s);
				s.rounds = o->rounds;
				*o = s;
			} else if (w && (is0(a->v) || is0(b->v)))
				o->rounds = 0;
			else if (w) {
				/* times a frozen power of two: exact, unless the result
				 * can be subnormal */
				int pa = P->fz[P->a[k]] && P->fv[P->a[k]] != 0 &&
				    frexpl(P->fv[P->a[k]], &d) == 0.5L * (P->fv[P->a[k]] <
				    0 ? -1 : 1);
				int pb = P->fz[P->b[k]] && P->fv[P->b[k]] != 0 &&
				    frexpl(P->fv[P->b[k]], &d) == 0.5L * (P->fv[P->b[k]] <
				    0 ? -1 : 1);
				ld em = add_r(mul_r(o->rho0, rmag(c, o), 1), o->eta0, 1);
				if ((pa || pb) && add_r(rmig(c, o), -em, 0) >= nmin)
					o->rounds = 0;
			}
			break;
		case O_DIV: {
			ld mb = rmig(c, b), rb;
			struct iv bw = b->w, bv;
			/* a divisor with the zero's factor, away from the zero: as
			 * it is (t^p w, t without 0 here) */
			if (b->p > 0 && !has0(c->T))
				bw = imul(b->w, tpow(c, b->p));
			/* then b = bw: its range is in both intervals */
			bv = iv2(fmaxl(b->v.lo, bw.lo), fminl(b->v.hi, bw.hi));
			if ((b->p != 0 && has0(c->T)) || !(mb > 0) || has0(bv)) {
				snprintf(P->why[k], sizeof P->why[k], "a divisor can be 0 "
				    "there");
				goto bad;
			}
			rb = add_r(b->rho, div_r(b->eta, mb, 1), 1);
			if (!(rb < 0.5L)) {
				snprintf(P->why[k], sizeof P->why[k], "a divisor's error is "
				    "as large as it there");
				goto bad;
			}
			o->v = idiv(a->v, bv);
			o->p = a->p;
			o->w = idiv(a->w, bv);
			o->rho0 = div_r(add_r(a->rho, rb, 1), 1 - rb, 1);
			o->eta0 = div_r(a->eta, mul_r(mb, 1 - rb, 0), 1);
			o->rounds = w != 0;
			break;
		}
		case O_EXP: {
			/* the machine exp: libm's double exp (taken within 2^-52),
			 * then rounded to the format; the operand's error e moves it
			 * by at most a factor exp(e) */
			ld ea = add_r(mul_r(a->rho, mag(a->v), 1), a->eta, 1);
			o->v = el_mono(O_EXP, a->v, 1);
			o->w = o->v;
			o->p = 0;
			o->rho0 = add_r(el_r(O_EXP, ea, 1), -1, 1);
			o->eta0 = 0;
			if (!(o->rho0 < 1)) {
				snprintf(P->why[k], sizeof P->why[k], "exp's argument "
				    "error is too large there");
				goto bad;
			}
			o->rho0 = rel_comb(o->rho0, ldexpl(1, -52));
			o->rounds = w != 0;
			c->mexp = 1;
			break;
		}
		case O_MIN:
		case O_MAX:
		case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT:
		case O_GE: {
			/* decided on the computed values: exact +- the error */
			struct iv ca = a->iok ? iv2((ld)a->ilo, (ld)a->ihi) :
			    computed(a);
			struct iv cb = b->iok ? iv2((ld)b->ilo, (ld)b->ihi) :
			    computed(b);
			if (op == O_MIN || op == O_MAX) {
				d = ca.hi <= cb.lo ? 1 : cb.hi <= ca.lo ? -1 : 0;
				if (op == O_MAX)
					d = -d;
				if (d == 0) {
					snprintf(P->why[k], sizeof P->why[k], "%s is undecided "
					    "there", opinfo[op].name);
					goto bad;
				}
				P->dec[k] = (signed char)d;
				*o = d > 0 ? *a : *b;
				o->rounds = 0;
				break;
			}
			d = cmp_dec(op, ca, cb);
			P->dec[k] = (signed char)d;
			o->v = o->w = ivp(d < 0 ? 0 : d);
			break;
		}
		case O_LAND:
		case O_LOR:
		case O_LNOT: {
			int da = P->dec[P->a[k]], db = op == O_LNOT ? 0 :
			    P->dec[P->b[k]];
			if (op == O_LAND)
				d = da == 0 || db == 0 ? 0 : da == 1 && db == 1 ? 1 : -1;
			else if (op == O_LOR)
				d = da == 1 || db == 1 ? 1 : da == 0 && db == 0 ? 0 : -1;
			else
				d = da < 0 ? -1 : !da;
			P->dec[k] = (signed char)d;
			o->v = o->w = ivp(d < 0 ? 0 : d);
			break;
		}
		case O_SEL:
			d = P->dec[P->a[k]];
			if (d < 0) {
				snprintf(P->why[k], sizeof P->why[k], "a select's "
				    "condition is undecided there");
				goto bad;
			}
			*o = d ? *b : c->R[P->c[k]];
			o->rounds = 0;
			{
				int s2 = d ? P->b[k] : P->c[k];
				if (P->fz[s2]) {
					P->fz[k] = 1;
					P->fv[k] = P->fv[s2];
					P->fi[k] = P->fi[s2];
				}
				P->xsrc[k] = P->xsrc[s2];
				P->xs[k] = P->xs[s2];
				P->xneg[k] = P->xneg[s2];
			}
			break;
		default:
			snprintf(P->why[k], sizeof P->why[k], "the candidate's %s "
			    "has no rounding rule here", opinfo[op].name);
			goto bad;
		}
		if (op == O_ADD || op == O_SUB || op == O_MUL || op == O_FMA ||
		    op == O_DIV || op == O_EXP || (op == O_CAST && o->rounds)) {
			ld vm, vn, em;
			if (op == O_CAST && o->rounds) {
				o->rho0 = a->rho;
				o->eta0 = a->eta;
			}
			vm = rmag(c, o);
			vn = rmig(c, o);
			em = add_r(mul_r(o->rho0, vm, 1), o->eta0, 1);
			if (w && add_r(vm, em, 1) >= fmt_max(w)) {
				snprintf(P->why[k], sizeof P->why[k], "a value can "
				    "overflow there");
				goto bad;
			}
			if (o->rounds && op != O_EXP && op != O_DIV &&
			    !(op == O_CAST)) {
				/* exact when every possible result is a float: a
				 * multiple of q no larger than 2^prec q (Sterbenz's
				 * subtractions, Cody-Waite's k c1 steps, sums with 0) */
				if (o->q >= ulp_w(w, 0) && add_r(vm, em, 1) <=
				    ldexpl(o->q, fmt_prec(w)))
					o->rounds = 0;
			}
			if (o->rounds) {
				/* fl(s) = s (1 + d), plus e where s can be subnormal
				 * (a sum that small is exact); one value on the piece
				 * when every s rounds to the same float */
				struct iv sr = iv2(add_r(o->v.lo, -em, 0), add_r(o->v.hi,
				    em, 1));
				/* a GPU's f32 quotient: Vulkan's 2.5 ulp, at most 5u,
				 * and not one value even where the rounding is */
				int gd = op == O_DIV && float_div_gpu && w == 32;
				if (!gd && rn_fmt(w, sr.lo) == rn_fmt(w, sr.hi)) {
					P->fz[k] = 1;
					P->fv[k] = rn_fmt(w, sr.lo);
					P->fi[k] = 0;
					memset(o, 0, sizeof *o);
					o->v = o->w = ivp(P->fv[k]);
					o->q = quant_of(w, P->fv[k]);
					continue;
				}
				o->rho = rel_comb(o->rho0, gd ? 5 * u : u);
				o->eta = mul_r(o->eta0, 1 + (gd ? 5 * u : u), 1);
				if (op != O_ADD && op != O_SUB && add_r(vn, -em, 0) <
				    nmin)
					o->eta = add_r(o->eta, eta, 1);
				vn = add_r(vn, -em, 0);
				o->q = vn > 0 ? ulp_w(w, vn) : ulp_w(w, 0);
			} else {
				o->rho = o->rho0;
				o->eta = o->eta0;
			}
		} else if (op != O_ABS && op != O_COPYSIGN && op != O_NEG &&
		    op != O_SEL && op != O_MIN && op != O_MAX && op != O_CAST &&
		    op != O_BARRIER && op != O_ASFLOAT) {
			o->rho = o->rho0 = 0;
			o->eta = o->eta0 = 0;
		}
		continue;
bad:
		P->bad[k] = 1;
	}
	if (P->bad[P->root]) {
		bad_why(P, c->why, sizeof c->why);
		return 0;
	}
	return 1;
}

/* ---- the certificate ---- */

struct piece {
	int64_t ka, kb;
	ld total, approx, reached;	/* -1: not certified there */
	ld worst;
};

struct acrun {
	struct prog F, C;
	struct jctx jf, jc;
	struct rctx rc;
	int w, kw, rel, k0, K, prec, emin;
	ld z;
	int hasz;
	struct iv sj[ACERT_K_MAX + 1], ej[ACERT_K_MAX + 1], fj[ACERT_K_MAX + 1];
	struct iv q1[ACERT_K_MAX + 1], q2[ACERT_K_MAX + 1];
	char why[200];
};

static ld
ulp_fmt(const struct acrun *A, ld y)
{
	int e;

	y = fabsl(y);
	e = y == 0 ? A->emin : ilogbl(y);
	if (e < A->emin)
		e = A->emin;
	return ldexpl(1, e - (A->prec - 1));
}

static int
binade(const struct acrun *A, ld y)
{
	int e = y == 0 ? A->emin : ilogbl(fabsl(y));

	return e < A->emin ? A->emin : e;
}

/* the jets of E = R - f and of f at x (n coefficients), each divided by
 * (x - z)^k0 when div (that factor is not 0 on x); 0 when it is */
static int
quot_jets(struct acrun *A, struct iv x, int n, int div)
{
	const struct iv *jr = A->jc.J + (size_t)A->C.root * (ACERT_K_MAX + 1);
	const struct iv *jf = A->jf.J + (size_t)A->F.root * (ACERT_K_MAX + 1);
	struct iv t[ACERT_K_MAX + 1];
	int i, j;

	for (i = 0; i < n; i++) {
		A->ej[i] = isub(jr[i], jf[i]);
		A->fj[i] = jf[i];
	}
	if (!div || A->k0 == 0)
		return 1;
	for (i = 0; i < n; i++)
		A->sj[i] = ivp(0);
	A->sj[0] = isub(x, ivp(A->z));
	if (n > 1)
		A->sj[1] = ivp(1);
	memcpy(t, A->sj, n * sizeof *t);
	for (j = 1; j < A->k0; j++) {
		jmul(t, A->sj, A->q1, n);
		memcpy(t, A->q1, n * sizeof *t);
	}
	if (!jdiv(A->ej, t, A->q1, n) || !jdiv(A->fj, t, A->q2, n))
		return 0;
	memcpy(A->ej, A->q1, n * sizeof *A->q1);
	memcpy(A->fj, A->q2, n * sizeof *A->q2);
	return 1;
}

/*
 * c[0] + c[1] d + ... + c[n-1] d^(n-1) + rem d^K over |d| <= dm: its
 * largest magnitude, and its least (0 when the constant does not
 * dominate)
 */
static void
poly_mag(const struct iv *c, int n, ld dm, struct iv rem, int K, ld *mx,
    ld *mn)
{
	ld s = 0, p = 1, pk = 1;
	int k;

	for (k = 1; k < n; k++) {
		p = mul_r(p, dm, 1);
		s = add_r(s, mul_r(mag(c[k]), p, 1), 1);
	}
	for (k = 0; k < K; k++)
		pk = mul_r(pk, dm, 1);
	s = add_r(s, mul_r(mag(rem), pk, 1), 1);
	*mx = add_r(mag(c[0]), s, 1);
	*mn = add_r(mig(c[0]), -s, 0);
	if (*mn < 0)
		*mn = 0;
}

/* the last rounding of the candidate: its root under exact sign changes,
 * selects, min and max, and exact scalings by powers of two (*scale
 * gets theirs: the root is that node's value times it) */
static int
last_round(const struct acrun *A, ld *scale)
{
	const struct prog *C = &A->C;
	int q = C->root, d;

	*scale = 1;
	for (;;) {
		int op = C->op[q];
		if (C->fz[q])
			return q;
		if (op == O_NEG || op == O_ABS || op == O_COPYSIGN ||
		    op == O_BARRIER || (op == O_CAST && !A->rc.R[q].rounds))
			q = C->a[q];
		else if (op == O_SEL)
			q = C->dec[C->a[q]] ? C->b[q] : C->c[q];
		else if (op == O_MIN || op == O_MAX)
			q = C->dec[q] > 0 ? C->a[q] : C->b[q];
		else if (op == O_ASFLOAT && C->xsrc[q] >= 0) {
			*scale = ldexpl(*scale, C->xs[q]);
			q = C->xsrc[q];
		} else if (op == O_MUL && !A->rc.R[q].rounds && (C->fz[C->a[q]] ||
		    C->fz[C->b[q]]) && frexpl(C->fz[C->a[q]] ? C->fv[C->a[q]] :
		    C->fv[C->b[q]], &d) == (C->fz[C->a[q]] ? C->fv[C->a[q]] :
		    C->fv[C->b[q]]) / fabsl(C->fz[C->a[q]] ? C->fv[C->a[q]] :
		    C->fv[C->b[q]]) * 0.5L) {
			ld p2 = C->fz[C->a[q]] ? C->fv[C->a[q]] : C->fv[C->b[q]];
			*scale *= fabsl(p2);
			q = C->fz[C->a[q]] ? C->b[q] : C->a[q];
		} else
			return q;
	}
}

/* an error e at x in the metric, against f in F: its bound (lo 0) or a
 * lower bound of it (lo 1); -1 when there is none */
static ld
metric_at(const struct acrun *A, ld e, struct iv F, int lo)
{
	if (A->rel && A->w)
		return lo ? e / ulp_fmt(A, mag(F)) : div_r(e, ulp_fmt(A, mig(F)),
		    1);
	if (A->rel) {
		if (e == 0)
			return 0;
		if (lo)
			return mag(F) > 0 ? e / mag(F) : 0;
		return mig(F) > 0 ? div_r(e, mig(F), 1) : -1;
	}
	return e;
}

/*
 * A piece of one float x: the candidate's and the target's values there,
 * their distance and the root's rounding directly (no Taylor, nor a
 * division by (x - z): at x = z both are 0, the candidate exactly).
 */
static int
point_piece(struct acrun *A, struct piece *pc, struct iv X)
{
	struct iv R0, F0, E0;
	ld rho = 0, eta = 0, err;

	if (!prog_jets(&A->F, &A->jf, X, 1, 1)) {
		snprintf(A->why, sizeof A->why, "%s", A->jf.why);
		return 0;
	}
	if (A->w) {
		A->rc.T = isub(X, ivp(A->z));
		if (!prog_round(&A->C, &A->rc, X)) {
			snprintf(A->why, sizeof A->why, "%s", A->rc.why);
			return 0;
		}
		rho = A->rc.R[A->C.root].rho;
		eta = A->rc.R[A->C.root].eta;
	} else if (!prog_jets(&A->C, &A->jc, X, 1, 1)) {
		snprintf(A->why, sizeof A->why, "%s", A->jc.why);
		return 0;
	}
	if (!prog_jets(&A->C, &A->jc, X, 1, 0)) {
		snprintf(A->why, sizeof A->why, "%s", A->jc.why);
		return 0;
	}
	R0 = A->jc.J[(size_t)A->C.root * (ACERT_K_MAX + 1)];
	F0 = A->jf.J[(size_t)A->F.root * (ACERT_K_MAX + 1)];
	E0 = isub(R0, F0);
	err = add_r(add_r(mag(E0), mul_r(rho, mag(R0), 1), 1), eta, 1);
	pc->total = metric_at(A, err, F0, 0);
	pc->approx = metric_at(A, mag(E0), F0, 0);
	pc->reached = metric_at(A, mig(E0), F0, 1);
	pc->worst = X.lo;
	if (!(pc->total >= 0) || !(pc->approx >= 0) || !isfinite(pc->total)) {
		snprintf(A->why, sizeof A->why, "the target is 0 there and the "
		    "candidate is not");
		return 0;
	}
	return 1;
}

/*
 * One piece's bounds: the total, the approximation's part, and what
 * that part reaches at the piece's middle (a lower bound of its largest
 * value there), in the metric; 0 with why.
 */
static int
piece_eval(struct acrun *A, struct piece *pc)
{
	ld l = (ld)fromkey(A->kw, pc->ka), r = (ld)fromkey(A->kw, pc->kb);
	ld m = l * 0.5L + r * 0.5L, dm, qemx, qemn, qfmx, qfmn, fmx, fmn;
	ld eps = 0, ab, tot, rch;
	struct iv X = iv2(l, r), M = ivp(m), remE, remF, fX, qX;
	int K = A->K, atz, n0 = 0, i;

	A->why[0] = 0;
	if (l == r)
		return point_piece(A, pc, X);
	atz = A->k0 > 0 && l <= A->z && A->z <= r;
	/* the branches on X, and the order-K coefficients on it */
	if (!prog_jets(&A->F, &A->jf, X, K + 1, 1)) {
		snprintf(A->why, sizeof A->why, "%s", A->jf.why);
		return 0;
	}
	if (A->w) {
		A->rc.T = isub(X, ivp(A->z));
		if (!prog_round(&A->C, &A->rc, X)) {
			snprintf(A->why, sizeof A->why, "%s", A->rc.why);
			return 0;
		}
	} else if (!prog_jets(&A->C, &A->jc, X, 1, 1)) {
		snprintf(A->why, sizeof A->why, "%s", A->jc.why);
		return 0;
	}
	if (!prog_jets(&A->C, &A->jc, X, K + 1, 0)) {
		snprintf(A->why, sizeof A->why, "%s", A->jc.why);
		return 0;
	}
	if (!quot_jets(A, X, K + 1, !atz)) {
		snprintf(A->why, sizeof A->why, "the target is 0 there, beyond "
		    "the fit's factor");
		return 0;
	}
	remE = A->ej[K];
	remF = A->fj[K];
	/* f on X directly, and f / (x - z)^k0 where it was divided: |f|'s
	 * bounds take the tighter of these and Taylor's */
	fX = A->jf.J[(size_t)A->F.root * (ACERT_K_MAX + 1)];
	qX = A->fj[0];
	/* the coefficients at the expansion point: z on its piece, else
	 * the middle */
	if (!prog_jets(&A->F, &A->jf, atz ? ivp(A->z) : M, K, 0) ||
	    !prog_jets(&A->C, &A->jc, atz ? ivp(A->z) : M, K, 0)) {
		snprintf(A->why, sizeof A->why, "%s", A->jf.why[0] ? A->jf.why :
		    A->jc.why);
		return 0;
	}
	if (!quot_jets(A, atz ? ivp(A->z) : M, K, !atz)) {
		snprintf(A->why, sizeof A->why, "the target is 0 there, beyond "
		    "the fit's factor");
		return 0;
	}
	if (atz) {
		/* (x - z)^k0 divides E and f: their first k0 coefficients at
		 * z are exactly 0 */
		for (i = 0; i < A->k0; i++) {
			if (!is0(A->fj[i])) {
				snprintf(A->why, sizeof A->why, "the target's zero at "
				    "x = %.9Lg is not exact in interval arithmetic",
				    A->z);
				return 0;
			}
			if (!is0(A->ej[i])) {
				snprintf(A->why, sizeof A->why, "the candidate is not 0 "
				    "at x = %.9Lg, where the target is: no relative "
				    "bound there", A->z);
				return 0;
			}
		}
		n0 = A->k0;
		dm = mag(isub(X, ivp(A->z)));
	} else
		dm = fmaxl(add_r(m, -l, 1), add_r(r, -m, 1));
	poly_mag(A->ej + n0, K - n0, dm, remE, K - n0, &qemx, &qemn);
	poly_mag(A->fj + n0, K - n0, dm, remF, K - n0, &qfmx, &qfmn);
	if (!atz) {
		qfmx = fminl(qfmx, mag(qX));
		qfmn = fmaxl(qfmn, mig(qX));
	}
	/* |f| on X */
	fmx = qfmx;
	fmn = qfmn;
	if (A->k0 > 0) {
		struct iv ta = ipowi(iabs(isub(X, ivp(A->z))), A->k0);
		fmn = mul_r(fmn, ta.lo, 0);
		fmx = mul_r(fmx, ta.hi, 1);
	}
	fmx = fminl(fmx, mag(fX));
	fmn = fmaxl(fmn, mig(fX));
	/* the approximation's part: eps = |E| / |f| (relative) or |E| */
	if (A->rel) {
		if (!(qfmn > 0)) {
			snprintf(A->why, sizeof A->why, "the target can be 0 there "
			    "(a relative bound needs it away from 0)");
			return 0;
		}
		eps = div_r(qemx, qfmn, 1);
	}
	/* what E reaches at the middle */
	if (!prog_jets(&A->F, &A->jf, M, 1, 0) ||
	    !prog_jets(&A->C, &A->jc, M, 1, 0)) {
		snprintf(A->why, sizeof A->why, "%s", A->jf.why);
		return 0;
	}
	{
		struct iv fm = A->jf.J[(size_t)A->F.root * (ACERT_K_MAX + 1)];
		struct iv em = isub(A->jc.J[(size_t)A->C.root *
		    (ACERT_K_MAX + 1)], fm);
		if (A->rel && A->w)
			rch = mig(em) / ulp_fmt(A, mag(fm));
		else if (A->rel)
			rch = mag(fm) > 0 ? mig(em) / mag(fm) : 0;
		else
			rch = mig(em);
	}
	if (!A->w) {
		ab = tot = A->rel ? eps : qemx;
	} else {
		ld scale;
		const struct rnode *rt = &A->rc.R[last_round(A, &scale)];
		ld ulpmin = ldexpl(1, A->emin - A->prec + 1);
		ld rho0 = rt->rounds ? rt->rho0 : rt->rho;
		ld eta0 = mul_r(rt->rounds ? rt->eta0 : rt->eta, scale, 1), fin = 0;
		if (A->rel) {
			/* in ulps of f: |y| / ulp(y) < 2^prec, and on X at most
			 * max|f| / ulp(min|f|) */
			ld Q = ldexpl(1, A->prec), delta, lo_s, hi_s, ratio;
			if (fmn > 0)
				Q = fminl(Q, div_r(fmx, ulp_fmt(A, fmn), 1));
			ab = mul_r(eps, Q, 1);
			/* before the last rounding: rho0 |R| + eta0 + |E|, |R| <=
			 * |f| (1 + eps) */
			delta = add_r(mul_r(rho0, 1 + eps, 1), eps, 1);
			/* eta0 is absolute: over ulp(f) >= ulp(min |f|) */
			tot = add_r(mul_r(delta, Q, 1), div_r(eta0, ulp_fmt(A, fmn), 1),
			    1);
			/* tot is D, s's distance from f in ulps of f.  The last
			 * rounding: half an ulp of s, so half of f's when they share
			 * a binade; when they do not, a power of two (a float) lies
			 * between them and |fl(s) - s| <= |s - f|: at most max(1/2,
			 * D) ulp of f either way */
			if (rt->rounds) {
				/* half an ulp of the result s, in ulps of f */
				lo_s = add_r(mul_r(fmn, 1 - delta, 0), -eta0, 0);
				hi_s = add_r(mul_r(fmx, 1 + delta, 1), eta0, 1);
				if (fmn > 0 && lo_s > 0 && binade(A, lo_s) ==
				    binade(A, hi_s) && binade(A, fmn) == binade(A, lo_s))
					ratio = 1;
				else if (fmn > 0)
					ratio = div_r(ulp_fmt(A, hi_s), ulp_fmt(A, fmn), 1);
				else
					ratio = fmaxl(1, add_r(mul_r(2, 1 + delta, 1),
					    div_r(mul_r(ldexpl(1, 1 - A->prec), eta0, 1),
					    ulpmin, 1), 1));
				fin = fminl(ratio / 2, fmaxl(0.5L, tot));
			}
		} else {
			ld rmx = add_r(fmx, qemx, 1);
			ld pre = add_r(mul_r(rho0, rmx, 1), eta0, 1);
			ab = qemx;
			tot = add_r(pre, qemx, 1);
			if (rt->rounds)
				fin = ulp_fmt(A, add_r(rmx, pre, 1)) / 2;
		}
		tot = add_r(tot, fin, 1);
	}
	if (!isfinite(tot) || !isfinite(ab)) {
		snprintf(A->why, sizeof A->why, "a bound is unbounded there");
		return 0;
	}
	pc->total = tot;
	pc->approx = ab;
	pc->reached = rch;
	pc->worst = m;
	return 1;
}
void
acert_run(nid target, nid cand, ld lo, ld hi, int w, int rel, int k0, ld z,
    struct acert *res)
{
	struct acrun A;
	struct piece *pc = NULL;
	int np = 0, cap = 0, i, nb = 0;
	int64_t brk[72];

	memset(res, 0, sizeof *res);
	memset(&A, 0, sizeof A);
	split_k = ldexpl(1, (LDBL_MANT_DIG + 1) / 2) + 1;
	A.w = w;
	A.kw = w ? w : 64;
	A.rel = rel;
	A.k0 = rel ? k0 : 0;
	A.z = A.k0 ? z : 0;
	A.hasz = A.k0 > 0;
	A.K = w == 32 ? 3 : 6;
	A.prec = w == 32 ? 24 : 53;
	A.emin = w == 32 ? -126 : -1022;
	res->order = A.K;
	res->slack = ACERT_SLACK_EXP;
	if (A.k0 >= A.K) {
		snprintf(res->why, sizeof res->why, "the target's zero has order "
		    "%d, the Taylor order here is %d", A.k0, A.K);
		return;
	}
	if (!prog_make(&A.F, target, 0, 0, res->why, sizeof res->why) ||
	    !prog_make(&A.C, cand, 0, w != 0 ? 2 : 1, res->why,
	    sizeof res->why)) {
		prog_free(&A.F);
		prog_free(&A.C);
		return;
	}
	A.jf.J = xcalloc((size_t)A.F.n * (ACERT_K_MAX + 1), sizeof *A.jf.J);
	A.jc.J = xcalloc((size_t)A.C.n * (ACERT_K_MAX + 1), sizeof *A.jc.J);
	for (i = 0; i < 5; i++) {
		A.jf.t[i] = xcalloc(ACERT_K_MAX + 1, sizeof(struct iv));
		A.jc.t[i] = xcalloc(ACERT_K_MAX + 1, sizeof(struct iv));
	}
	A.rc.R = xcalloc((size_t)A.C.n, sizeof *A.rc.R);
	A.rc.hasz = A.hasz;
	A.rc.z = A.z;
	/* the floats in [lo, hi] are the keys brk[0] .. brk[1] - 1 */
	{
		int64_t ka = fkey(A.kw, (double)lo), kb = fkey(A.kw, (double)hi);
		if ((ld)fromkey(A.kw, ka) < lo)
			ka++;
		if ((ld)fromkey(A.kw, kb) > hi)
			kb--;
		if (ka > kb) {
			snprintf(res->why, sizeof res->why, "no %s lies in the "
			    "interval", w ? (w == 32 ? "f32" : "f64") : "double");
			goto out;
		}
		brk[nb++] = ka;
		brk[nb++] = kb + 1;
	}
	/* break points: z, 0 and every constant the programs compare x (or
	 * |x|) with, where a branch changes */
	{
		struct prog *Ps[2] = {&A.F, &A.C};
		ld pts[64];
		int p, q, npts = 0;
		pts[npts++] = 0;
		if (A.hasz)
			pts[npts++] = A.z;
		for (p = 0; p < 2; p++)
			for (q = 0; q < Ps[p]->n && npts < 62; q++) {
				int op = Ps[p]->op[q], a = Ps[p]->a[q], b = Ps[p]->b[q];
				if (op < O_EQ || op > O_GE || Ps[p]->op[b] != O_CONST)
					continue;
				if (Ps[p]->op[a] == O_VAR)
					pts[npts++] = Ps[p]->k[b];
				else if (Ps[p]->op[a] == O_ABS &&
				    Ps[p]->op[Ps[p]->a[a]] == O_VAR) {
					pts[npts++] = Ps[p]->k[b];
					pts[npts++] = -Ps[p]->k[b];
				}
			}
		for (q = 0; q < npts; q++) {
			int64_t kk = fkey(A.kw, (double)pts[q]);
			if (kk > brk[0] && kk < brk[1])
				brk[nb++] = kk;
		}
	}
	/* sorted, distinct */
	for (i = 1; i < nb; i++) {
		int64_t t = brk[i];
		int j = i - 1;
		while (j >= 0 && brk[j] > t) {
			brk[j + 1] = brk[j];
			j--;
		}
		brk[j + 1] = t;
	}
	{
		int j = 0;
		for (i = 0; i < nb; i++)
			if (j == 0 || brk[i] != brk[j - 1])
				brk[j++] = brk[i];
		nb = j;
	}
	/* 32 pieces between consecutive break points, even in the keys */
	for (i = 0; i + 1 < nb; i++) {
		int64_t a = brk[i], len = brk[i + 1] - a, j, parts;
		parts = len < 32 ? len : 32;
		for (j = 0; j < parts; j++) {
			if (np == cap) {
				cap = cap ? cap * 2 : 256;
				pc = xrealloc(pc, (size_t)cap * sizeof *pc);
			}
			pc[np].ka = a + (int64_t)((ld)len * j / parts);
			pc[np].kb = a + (int64_t)((ld)len * (j + 1) / parts) - 1;
			pc[np].total = -2;	/* not evaluated */
			np++;
		}
	}
	/*
	 * Evaluate, then bisect each piece not certified, or whose
	 * approximation's part may exceed what was reached anywhere by more
	 * than ACERT_TAU of the largest total bound.
	 */
	for (;;) {
		ld reached = 0, tmax = 0;
		int nref = 0, nnew;
		for (i = 0; i < np; i++)
			if (pc[i].total == -2) {
				res->evals++;
				if (!piece_eval(&A, &pc[i])) {
					pc[i].total = -1;
					snprintf(res->why, sizeof res->why, "%s", A.why);
					if (pc[i].ka == pc[i].kb) {
						snprintf(res->why + strlen(res->why),
						    sizeof res->why - strlen(res->why), " (at x = "
						    "%.9g)", fromkey(A.kw, pc[i].ka));
						goto out;
					}
				}
			}
		for (i = 0; i < np; i++)
			if (pc[i].total >= 0) {
				reached = fmaxl(reached, pc[i].reached);
				tmax = fmaxl(tmax, pc[i].total);
			}
		for (i = 0; i < np; i++)
			if (pc[i].ka < pc[i].kb && (pc[i].total < 0 ||
			    pc[i].approx > reached + ACERT_TAU * tmax))
				nref++;
		if (nref == 0)
			break;
		if (np + nref > ACERT_MAXPIECES) {
			/* out of pieces: the bounds stand where every piece has
			 * one */
			for (i = 0; i < np; i++)
				if (pc[i].total < 0) {
					snprintf(res->why + strlen(res->why), sizeof res->why -
					    strlen(res->why), " (after %d pieces)", np);
					goto out;
				}
			break;
		}
		nnew = np;
		for (i = 0; i < np; i++)
			if (pc[i].ka < pc[i].kb && (pc[i].total < 0 ||
			    pc[i].approx > reached + ACERT_TAU * tmax)) {
				int64_t mid = pc[i].ka + (pc[i].kb - pc[i].ka) / 2;
				if (nnew == cap) {
					cap *= 2;
					pc = xrealloc(pc, (size_t)cap * sizeof *pc);
				}
				pc[nnew].ka = mid + 1;
				pc[nnew].kb = pc[i].kb;
				pc[nnew].total = -2;
				nnew++;
				pc[i].kb = mid;
				pc[i].total = -2;
			}
		np = nnew;
	}
	res->ok = 1;
	res->why[0] = 0;
	res->pieces = np;
	res->mexp = A.rc.mexp;
	for (i = 0; i < np; i++) {
		if (pc[i].total > res->bound) {
			res->bound = (double)pc[i].total;
			res->worst = (double)pc[i].worst;
		}
		res->approx = fmax(res->approx, (double)pc[i].approx);
		res->reached = fmax(res->reached, (double)pc[i].reached);
	}
out:
	free(pc);
	free(A.jf.J);
	free(A.jc.J);
	for (i = 0; i < 5; i++) {
		free(A.jf.t[i]);
		free(A.jc.t[i]);
	}
	free(A.rc.R);
	prog_free(&A.F);
	prog_free(&A.C);
}
