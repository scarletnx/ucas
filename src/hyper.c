/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * hyper.c - complex, quaternion, octonion and Cayley-Dickson numbers.
 *
 * The node DAG stays real-valued; a hypercomplex expression is an
 * ordinary DAG in which the units (i; i j k; e1..e7; e1..e15 ...) are
 * variables in reserved slots (UB + u) and the complex variables (z, w,
 * q: quat ...) are ordinary parameter slots.  The constructors keep the
 * order of the operands of a product (only real constants move), so the
 * DAG is faithful for non-commutative and non-associative algebras: a
 * product is evaluated exactly as it is grouped.  This file evaluates such
 * a DAG with hypercomplex values (a small plan: the reachable nodes in id
 * order, one value of dimension d per node), and the symbolic work is
 * done by the existing code: diff_node for derivatives, forms.c for the
 * real and imaginary parts with z = x + iy.
 *
 * Multiplication is the Cayley-Dickson doubling
 *     (p, q)(s, t) = (p s - conj(t) q, t p + q conj(s)),
 * with e_{m + d/2} = (0, e_m): i j = k, e1 e2 = e3, e1 e4 = e5.  Every
 * element of every Cayley-Dickson algebra lies in the complex subalgebra
 * spanned by 1 and its vector part, so an analytic function with real
 * Taylor coefficients is evaluated through C's complex function at
 * a + |v| i and mapped back: f(a + v) = Re f(a + |v|i) + (v/|v|) Im f(...).
 * That is exact for quaternions and octonions as well (a single element
 * generates an associative subalgebra).
 */
#include "calc.h"
#include "gif.h"

#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#define HM 32		/* the largest algebra: cd(32) */
#define UB 64		/* unit e_u is the variable in slot UB + u */
#define MAXLG 5

typedef double complex cplx;

static const char *const alg_name[] = {"real", "complex", "quaternion",
	"octonion", "sedenion", "cd(32)"};
static const char *const alg_pl[] = {"reals", "complex numbers",
	"quaternions", "octonions", "sedenions", "cd(32) numbers"};

/* ---- the algebra ---- */

static void
h_conj(const double *a, double *r, int d)
{
	int k;

	r[0] = a[0];
	for (k = 1; k < d; k++)
		r[k] = -a[k];
}

/* r = a b; r may alias a or b */
static void
h_mul(const double *a, const double *b, double *r, int d)
{
	double t1[HM], t2[HM], cx[HM], out[HM];
	int h = d / 2, k;

	if (d == 1) {
		r[0] = a[0] * b[0];
		return;
	}
	if (d == 2) {
		double re = a[0] * b[0] - a[1] * b[1];
		double im = a[0] * b[1] + a[1] * b[0];
		r[0] = re;
		r[1] = im;
		return;
	}
	/* (p, q)(s, t) = (p s - conj(t) q, t p + q conj(s)) */
	h_conj(b + h, cx, h);
	h_mul(a, b, t1, h);
	h_mul(cx, a + h, t2, h);
	for (k = 0; k < h; k++)
		out[k] = t1[k] - t2[k];
	h_conj(b, cx, h);
	h_mul(b + h, a, t1, h);
	h_mul(a + h, cx, t2, h);
	for (k = 0; k < h; k++)
		out[h + k] = t1[k] + t2[k];
	memcpy(r, out, (size_t)d * sizeof *r);
}

static double
h_n2(const double *a, int d)
{
	double s = 0;
	int k;

	for (k = 0; k < d; k++)
		s += a[k] * a[k];
	return s;
}

static double
h_vabs(const double *a, int d)
{
	double s = 0;
	int k;

	for (k = 1; k < d; k++)
		s += a[k] * a[k];
	return sqrt(s);
}

static int
h_isreal(const double *a, int d)
{
	int k;
	double m = fabs(a[0]);

	for (k = 1; k < d; k++)
		if (fabs(a[k]) > 1e-13 * (m + 1e-300) && a[k] != 0)
			return 0;
	return 1;
}

static int
h_inv(const double *a, double *r, int d)
{
	double n = h_n2(a, d);
	int k;

	if (n == 0)
		return 0;
	h_conj(a, r, d);
	for (k = 0; k < d; k++)
		r[k] /= n;
	return 1;
}

static double
h_dist(const double *a, const double *b, int d)
{
	double s = 0;
	int k;

	for (k = 0; k < d; k++)
		s += (a[k] - b[k]) * (a[k] - b[k]);
	return sqrt(s);
}

static int
h_close(const double *a, const double *b, int d)
{
	double m = sqrt(h_n2(a, d)) + sqrt(h_n2(b, d));

	return h_dist(a, b, d) <= 1e-9 * (m > 1 ? m : 1);
}

/* ---- complex functions: principal values, as forms.c takes them ---- */

static cplx
nz(cplx w)
{
	return cimag(w) == 0 ? (cplx)creal(w) : w;
}

static int cut_op;	/* the last function evaluated on a branch cut */

static cplx
cfun(int op, cplx a)
{
	double x = creal(a);
	int re = cimag(a) == 0;
	cplx r;

	if (re)
		a = x;		/* no -0: the cuts belong to the upper side */
	switch (op) {
	case O_EXP: return cexp(a);
	case O_LOG:
		if (a == 0)
			return NAN;
		if (re && x < 0)
			cut_op = op;
		return re && x > 0 ? log(x) : clog(a);
	case O_SQRT:
		if (re && x < 0)
			cut_op = op;
		return re && x >= 0 ? sqrt(x) : csqrt(a);
	case O_SIN: return re ? sin(x) : csin(a);
	case O_COS: return re ? cos(x) : ccos(a);
	case O_TAN:
		r = ccos(a);
		if (cabs(r) < 1e-15)
			return NAN;
		return re ? tan(x) : ctan(a);
	case O_SEC:
		r = ccos(a);
		return cabs(r) < 1e-15 ? NAN : 1 / r;
	case O_CSC:
		r = csin(a);
		return cabs(r) < 1e-15 ? NAN : 1 / r;
	case O_COT:
		r = csin(a);
		return cabs(r) < 1e-15 ? NAN : ccos(a) / r;
	case O_ATAN:
		if (x == 0 && fabs(cimag(a)) == 1)
			return NAN;
		if (x == 0 && fabs(cimag(a)) > 1)
			cut_op = op;
		return re ? atan(x) : catan(a);
	case O_ASIN:
	case O_ACOS:
		if (re && x >= -1 && x <= 1)
			return op == O_ASIN ? asin(x) : acos(x);
		if (!re)
			return op == O_ASIN ? casin(a) : cacos(a);
		cut_op = op;
		r = -I * clog(nz(I * a + csqrt(nz(1 - a * a))));
		return op == O_ASIN ? r : M_PI / 2 - r;
	case O_SINH: return re ? sinh(x) : csinh(a);
	case O_COSH: return re ? cosh(x) : ccosh(a);
	case O_TANH:
		r = ccosh(a);
		if (cabs(r) < 1e-15)
			return NAN;
		return re ? tanh(x) : ctanh(a);
	case O_ASINH:
		if (x == 0 && fabs(cimag(a)) > 1)
			cut_op = op;
		return re ? asinh(x) : casinh(a);
	case O_ACOSH:
		if (re && x >= 1)
			return acosh(x);
		if (!re)
			return cacosh(a);
		cut_op = op;
		return clog(nz(a + csqrt(nz(a + 1)) * csqrt(nz(a - 1))));
	case O_ATANH:
		if (re && x > -1 && x < 1)
			return atanh(x);
		if (!re)
			return catanh(a);
		if (fabs(x) == 1)
			return NAN;
		cut_op = op;
		return (clog(nz(1 + a)) - clog(nz(1 - a))) / 2;
	}
	return NAN;
}

static int
is_slice_op(int op)
{
	switch (op) {
	case O_EXP: case O_LOG: case O_SQRT: case O_SIN: case O_COS:
	case O_TAN: case O_SEC: case O_CSC: case O_COT: case O_ATAN:
	case O_ASIN: case O_ACOS: case O_SINH: case O_COSH: case O_TANH:
	case O_ASINH: case O_ACOSH: case O_ATANH:
		return 1;
	}
	return 0;
}

/* notes raised while evaluating, printed once per command */
enum {
	HN_RDIV = 1, HN_DIR = 2, HN_REAL = 4, HN_POW = 8, HN_ASSOC = 16,
	HN_CUT = 32
};
static unsigned hn;
static int hn_realop, hn_cutop;

/* f(a + v) through the complex plane spanned by 1 and v */
static int
h_slice(int op, const double *x0, double *r, int d)
{
	cplx w;
	double s, x[HM];
	int k;

	memcpy(x, x0, (size_t)d * sizeof *x);	/* r may alias x0 */

	cut_op = 0;
	if (d == 2) {
		w = cfun(op, x[0] + I * x[1]);
		r[0] = creal(w);
		r[1] = cimag(w);
	} else {
		s = h_vabs(x, d);
		memset(r, 0, (size_t)d * sizeof *r);
		if (s == 0) {
			w = cfun(op, x[0]);
			r[0] = creal(w);
			if (cimag(w) != 0) {
				r[1] = cimag(w);	/* any unit would do: i */
				hn |= HN_DIR;
			}
		} else {
			w = cfun(op, x[0] + I * s);
			r[0] = creal(w);
			for (k = 1; k < d; k++)
				r[k] = x[k] / s * cimag(w);
		}
	}
	if (cut_op) {
		hn |= HN_CUT;
		hn_cutop = cut_op;
	}
	return isfinite(creal(w)) && isfinite(cimag(w));
}

static int
h_ipow(const double *a, int64_t n, double *r, int d)
{
	double b[HM], acc[HM];
	int neg = n < 0;

	memset(acc, 0, sizeof acc);
	acc[0] = 1;
	memcpy(b, a, (size_t)d * sizeof *b);
	if (neg) {
		if (!h_inv(a, b, d))
			return 0;
		n = -n;
	}
	while (n) {
		if (n & 1)
			h_mul(acc, b, acc, d);
		n >>= 1;
		if (n)
			h_mul(b, b, b, d);
	}
	memcpy(r, acc, (size_t)d * sizeof *r);
	return 1;
}

static int
h_commute(const double *a, const double *b, int d)
{
	double x[HM], y[HM];

	h_mul(a, b, x, d);
	h_mul(b, a, y, d);
	return h_close(x, y, d);
}

/* a^b: integer powers by multiplication, else principal exp(log(a) b) */
static int
h_pow(const double *a0, const double *b0, double *r, int d)
{
	double l[HM], t[HM], a[HM], b[HM];
	int k;

	memcpy(a, a0, (size_t)d * sizeof *a);	/* r may alias a or b */
	memcpy(b, b0, (size_t)d * sizeof *b);

	if (h_isreal(b, d) && b[0] == floor(b[0]) && fabs(b[0]) <= 1e6) {
		if (h_n2(a, d) == 0) {
			if (b[0] < 0)
				return 0;
			memset(r, 0, (size_t)d * sizeof *r);
			r[0] = b[0] == 0 ? 1 : 0;
			return 1;
		}
		return h_ipow(a, (int64_t)b[0], r, d);
	}
	if (h_n2(a, d) == 0) {
		if (h_isreal(b, d) && b[0] > 0) {
			memset(r, 0, (size_t)d * sizeof *r);
			return 1;
		}
		return 0;
	}
	if (h_isreal(b, d)) {
		/* a real exponent: in the plane of a */
		cplx w, z;
		double s = d == 2 ? 0 : h_vabs(a, d);
		cut_op = 0;
		if (d == 2)
			z = a[0] + I * (a[1] == 0 ? 0.0 : a[1]);
		else
			z = a[0] + I * s;
		if (cimag(z) == 0 && creal(z) < 0) {
			hn |= HN_CUT;
			hn_cutop = O_POW;
			if (d > 2)
				hn |= HN_DIR;
		}
		w = cexp(b[0] * clog(z));
		memset(r, 0, (size_t)d * sizeof *r);
		r[0] = creal(w);
		if (d == 2)
			r[1] = cimag(w);
		else if (s == 0)
			r[1] = cimag(w);
		else
			for (k = 1; k < d; k++)
				r[k] = a[k] / s * cimag(w);
		return isfinite(creal(w)) && isfinite(cimag(w));
	}
	if (!h_slice(O_LOG, a, l, d))
		return 0;
	if (d > 2 && !h_commute(l, b, d))
		hn |= HN_POW;
	h_mul(l, b, t, d);
	return h_slice(O_EXP, t, r, d);
}

/* ---- the plan: reachable nodes in id order, a value each ---- */

struct hp {
	int d, f32, np;
	uint32_t n;
	nid *g;
	uint32_t *a, *b, *c;
	double *v;
	uint8_t *bad;
	nid *roots;
	uint32_t *rloc;
	int nroot;
};

#define HV(P, k) (&(P)->v[(size_t)(k) * (P)->d])

static int
nid_cmp(const void *x, const void *y)
{
	nid a = *(const nid *)x, b = *(const nid *)y;

	return a < b ? -1 : a > b;
}

static uint32_t
hp_find(const struct hp *P, nid g)
{
	uint32_t lo = 0, hi = P->n;

	while (lo < hi) {
		uint32_t m = (lo + hi) / 2;
		if (P->g[m] < g)
			lo = m + 1;
		else
			hi = m;
	}
	return lo < P->n && P->g[lo] == g ? lo : NOTOK;
}

static void
hp_build(struct hp *P, const nid *roots, int nroot, int d, int np)
{
	uint8_t *seen = xcalloc(N.n, 1);
	nid *st = xmalloc((size_t)N.n * sizeof *st);
	uint32_t sp = 0, cap = 64, k;
	int j;

	memset(P, 0, sizeof *P);
	P->d = d;
	P->np = np;
	P->g = xmalloc(cap * sizeof *P->g);
	for (j = 0; j < nroot; j++)
		if (!seen[roots[j]]) {
			seen[roots[j]] = 1;
			st[sp++] = roots[j];
		}
	while (sp) {
		nid e = st[--sp];
		int op = N.op[e], ar = opinfo[op].arity;
		nid ch[3];
		int c;
		if (op == O_LOOP || op == O_TABLE || op == O_INDEX ||
		    op == O_ATOMIC)
			fail("%s is not supported with complex or hypercomplex "
			    "values", op == O_LOOP ? "a loop or sum" : opinfo[op].name);
		if (P->n == cap) {
			cap *= 2;
			P->g = xrealloc(P->g, cap * sizeof *P->g);
		}
		P->g[P->n++] = e;
		if (op == O_CONST || op == O_VAR || op == O_UNDEF)
			continue;
		ch[0] = N.a[e];
		ch[1] = N.b[e];
		ch[2] = N.c[e];
		for (c = 0; c < ar; c++)
			if (!seen[ch[c]]) {
				seen[ch[c]] = 1;
				st[sp++] = ch[c];
			}
	}
	free(seen);
	free(st);
	qsort(P->g, P->n, sizeof *P->g, nid_cmp);
	P->a = xcalloc(P->n, sizeof *P->a);
	P->b = xcalloc(P->n, sizeof *P->b);
	P->c = xcalloc(P->n, sizeof *P->c);
	for (k = 0; k < P->n; k++) {
		nid e = P->g[k];
		int ar = opinfo[N.op[e]].arity;
		if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_UNDEF)
			continue;
		if (ar >= 1)
			P->a[k] = hp_find(P, N.a[e]);
		if (ar >= 2)
			P->b[k] = hp_find(P, N.b[e]);
		if (ar >= 3)
			P->c[k] = hp_find(P, N.c[e]);
	}
	P->v = xcalloc((size_t)P->n * d, sizeof *P->v);
	P->bad = xcalloc(P->n, 1);
	P->nroot = nroot;
	P->roots = xmalloc((size_t)nroot * sizeof *P->roots);
	P->rloc = xmalloc((size_t)nroot * sizeof *P->rloc);
	for (j = 0; j < nroot; j++) {
		P->roots[j] = roots[j];
		P->rloc[j] = hp_find(P, roots[j]);
	}
}

static void
hp_free(struct hp *P)
{
	free(P->g);
	free(P->a);
	free(P->b);
	free(P->c);
	free(P->v);
	free(P->bad);
	free(P->roots);
	free(P->rloc);
	memset(P, 0, sizeof *P);
}

static void
real_only(int op)
{
	hn |= HN_REAL;
	hn_realop = op;
}

static void
hp_node(struct hp *P, uint32_t k, const double *in)
{
	nid e = P->g[k];
	int op = N.op[e], d = P->d, j;
	double *r = HV(P, k), t[HM];
	const double *A = NULL, *B = NULL, *C = NULL;
	int ar = opinfo[op].arity;
	double v;

	P->bad[k] = 0;
	switch (op) {
	case O_CONST:
		memset(r, 0, (size_t)d * sizeof *r);
		num_of(e, &v);
		r[0] = v;
		return;
	case O_VAR: {
		int64_t s = N.ki[e];
		memset(r, 0, (size_t)d * sizeof *r);
		if (s >= UB) {
			if (s - UB >= d)
				P->bad[k] = 1;
			else
				r[s - UB] = 1;
		} else if (s < P->np)
			memcpy(r, in + (size_t)s * d, (size_t)d * sizeof *r);
		else
			P->bad[k] = 1;
		return;
	}
	case O_UNDEF:
		P->bad[k] = 1;
		return;
	}
	if (ar >= 1)
		A = HV(P, P->a[k]);
	if (ar >= 2)
		B = HV(P, P->b[k]);
	if (ar >= 3)
		C = HV(P, P->c[k]);
	if (op == O_SEL) {
		uint32_t pick;
		if (P->bad[P->a[k]]) {
			P->bad[k] = 1;
			return;
		}
		if (!h_isreal(A, d)) {
			real_only(op);
			P->bad[k] = 1;
			return;
		}
		pick = A[0] != 0 ? P->b[k] : P->c[k];
		memcpy(r, HV(P, pick), (size_t)d * sizeof *r);
		P->bad[k] = P->bad[pick];
		return;
	}
	if (op == O_LAND || op == O_LOR) {
		int x, y;
		if (P->bad[P->a[k]] || !h_isreal(A, d)) {
			P->bad[k] = 1;
			return;
		}
		x = A[0] != 0;
		memset(r, 0, (size_t)d * sizeof *r);
		if ((op == O_LAND && !x) || (op == O_LOR && x)) {
			r[0] = x;
			return;
		}
		if (P->bad[P->b[k]] || !h_isreal(B, d)) {
			P->bad[k] = 1;
			return;
		}
		y = B[0] != 0;
		r[0] = y;
		return;
	}
	if ((ar >= 1 && P->bad[P->a[k]]) || (ar >= 2 && P->bad[P->b[k]]) ||
	    (ar >= 3 && P->bad[P->c[k]])) {
		P->bad[k] = 1;
		return;
	}
	switch (op) {
	case O_NEG:
		for (j = 0; j < d; j++)
			r[j] = -A[j];
		break;
	case O_ADD:
		for (j = 0; j < d; j++)
			r[j] = A[j] + B[j];
		break;
	case O_SUB:
		for (j = 0; j < d; j++)
			r[j] = A[j] - B[j];
		break;
	case O_MUL:
		h_mul(A, B, r, d);
		/* octonions: is the grouping what makes the value? */
		if (d >= 8 && !(hn & HN_ASSOC)) {
			nid ea = N.a[e], eb = N.b[e];
			double u[HM];
			if (N.op[ea] == O_MUL) {
				const double *x = HV(P, P->a[P->a[k]]),
				    *y = HV(P, P->b[P->a[k]]);
				h_mul(y, B, t, d);
				h_mul(x, t, u, d);
				if (!h_close(u, r, d))
					hn |= HN_ASSOC;
			} else if (N.op[eb] == O_MUL) {
				const double *x = HV(P, P->a[P->b[k]]),
				    *y = HV(P, P->b[P->b[k]]);
				h_mul(A, x, t, d);
				h_mul(t, y, u, d);
				if (!h_close(u, r, d))
					hn |= HN_ASSOC;
			}
		}
		break;
	case O_DIV:
		if (!h_inv(B, t, d)) {
			P->bad[k] = 1;
			return;
		}
		if (d > 2 && !h_commute(A, B, d))
			hn |= HN_RDIV;
		h_mul(A, t, r, d);
		break;
	case O_POW:
		if (!h_pow(A, B, r, d))
			P->bad[k] = 1;
		break;
	case O_ABS:
		memset(r, 0, (size_t)d * sizeof *r);
		r[0] = sqrt(h_n2(A, d));
		break;
	case O_CONJ:
		h_conj(A, r, d);
		break;
	case O_RE:
		memset(r, 0, (size_t)d * sizeof *r);
		r[0] = A[0];
		break;
	case O_IM:
		memset(r, 0, (size_t)d * sizeof *r);
		if (d == 2)
			r[0] = A[1];
		else
			for (j = 1; j < d; j++)
				r[j] = A[j];
		break;
	case O_ARG:
		if (h_n2(A, d) == 0) {
			P->bad[k] = 1;
			return;
		}
		memset(r, 0, (size_t)d * sizeof *r);
		r[0] = d == 2 ? atan2(A[1] == 0 ? 0.0 : A[1], A[0]) :
		    atan2(h_vabs(A, d), A[0]);
		break;
	case O_CAST:
	case O_BARRIER:
		memcpy(r, A, (size_t)d * sizeof *r);
		break;
	case O_FMA:
		h_mul(A, B, t, d);
		for (j = 0; j < d; j++)
			r[j] = t[j] + C[j];
		break;
	default:
		if (is_slice_op(op)) {
			if (!h_slice(op, A, r, d))
				P->bad[k] = 1;
			break;
		}
		/* real-only operations: every operand must be real */
		if ((A && !h_isreal(A, d)) || (B && !h_isreal(B, d))) {
			real_only(op);
			P->bad[k] = 1;
			return;
		}
		{
			uint8_t fl = 0;
			double y = scalar_real(op, A ? A[0] : 0, B ? B[0] : 0, &fl);
			memset(r, 0, (size_t)d * sizeof *r);
			if (fl & F_UNDEF) {
				P->bad[k] = 1;
				return;
			}
			r[0] = y;
		}
	}
	if (P->f32)
		for (j = 0; j < d; j++)
			r[j] = fround(32, r[j]);
	for (j = 0; j < d; j++)
		if (!isfinite(r[j])) {
			P->bad[k] = 1;
			return;
		}
}

/* in: np values of dimension d; 1 when root j is defined */
static void
hp_run(struct hp *P, const double *in)
{
	uint32_t k;

	for (k = 0; k < P->n; k++)
		hp_node(P, k, in);
}

static int
hp_get(struct hp *P, int j, double *out)
{
	uint32_t l = P->rloc[j];

	memcpy(out, HV(P, l), (size_t)P->d * sizeof *out);
	return !P->bad[l];
}

/* ---- printing values ---- */

static const char *
unit_str(int u, int lg)
{
	static char b[8][8];
	static int k;
	char *s = b[k++ & 7];

	if (u == 0)
		return "1";
	if (lg <= 1)
		return "i";
	if (lg == 2) {
		static const char *const q[] = {"1", "i", "j", "k"};
		return q[u];
	}
	snprintf(s, 8, "e%d", u);
	return s;
}

static int
digits_only(const char *s)
{
	if (*s == '-')
		s++;
	for (; *s; s++)
		if (!isdigit((unsigned char)*s) && *s != '.')
			return 0;
	return 1;
}

static int
plain_num(const char *s)
{
	if (*s == '-')
		s++;
	for (; *s; s++)
		if (!isdigit((unsigned char)*s) && *s != '.' && *s != 'e' &&
		    *s != '-' && *s != '+')
			return 0;
	return 1;
}

/* "(-1 + sqrt(3))/3" -> "(1 - sqrt(3))/3": the negated text, or NULL */
static const char *
flip_sum(const char *s)
{
	static char b[4][128];
	static int k;
	char *o = b[k++ & 3];
	const char *p, *e;
	int depth = 0;

	if (s[0] != '(' || s[1] != '-')
		return NULL;
	for (p = s + 1; *p; p++) {
		if (*p == '(')
			depth++;
		if (*p == ')') {
			if (depth == 0)
				break;
			depth--;
		}
		if (depth == 0 && (!strncmp(p, " + ", 3) || !strncmp(p, " - ", 3)))
			break;
	}
	if (*p != ' ')
		return NULL;
	e = strchr(p + 3, ')');
	if (e == NULL)
		return NULL;
	snprintf(o, 128, "(%.*s %c %.*s%s", (int)(p - s - 2), s + 2,
	    p[1] == '+' ? '-' : '+', (int)(e - p - 3), p + 3, e);
	return o;
}

/* drop parts that are rounding noise next to the largest one */
static void
h_clean(double *x, int d)
{
	double m = 0;
	int k;

	for (k = 0; k < d; k++)
		if (fabs(x[k]) > m)
			m = fabs(x[k]);
	for (k = 0; k < d; k++)
		if (fabs(x[k]) <= 1e-13 * m || fabs(x[k]) < 1e-300)
			x[k] = 0;
}

static void
hv_text(struct sbuf *b, const double *x0, int d, int lg)
{
	double x[HM];
	char nb[128];
	int k, first = 1;

	memcpy(x, x0, (size_t)d * sizeof *x);
	h_clean(x, d);
	if (d == 2) {
		int e1 = forms_num(x[0], nb, sizeof nb) || x[0] == 0;
		int e2 = forms_num(x[1], nb, sizeof nb) || x[1] == 0;
		if (e1 && e2) {
			/* forms writes i*2pi: here 2pi*i */
			struct sbuf z = {0};
			const char *p, *q;
			forms_ztext(x[0], x[1], &z);
			p = strstr(z.p, "i*");
			q = p ? p + 2 : NULL;
			while (q && isdigit((unsigned char)*q))
				q++;
			if (p && (p == z.p || p[-1] == ' ' || p[-1] == '-') &&
			    !strncmp(q, "pi", 2) && !isalnum((unsigned char)q[2])) {
				sb_printf(b, "%.*s%.*spi*i%s", (int)(p - z.p), z.p,
				    (int)(q - p - 2), p + 2, q + 2);
			} else
				sb_puts(b, z.p);
			free(z.p);
			return;
		}
		if (x[1] == 0) {
			sb_printf(b, "%.10g", x[0]);
			return;
		}
		if (x[0] != 0)
			sb_printf(b, "%.10g %s ", x[0], x[1] < 0 ? "-" : "+");
		else if (x[1] < 0)
			sb_puts(b, "-");
		sb_printf(b, "%.10gi", fabs(x[1]));
		return;
	}
	for (k = 0; k < d; k++) {
		double c = x[k];
		const char *u = unit_str(k, lg);
		if (c == 0)
			continue;
		forms_num(fabs(c), nb, sizeof nb);
		if (c < 0 && k > 0 && flip_sum(nb)) {
			snprintf(nb, sizeof nb, "%s", flip_sum(nb));
			c = -c;
		}
		if (!first)
			sb_puts(b, c < 0 ? " - " : " + ");
		else if (c < 0)
			sb_puts(b, "-");
		first = 0;
		if (k == 0)
			sb_puts(b, nb);
		else if (!strcmp(nb, "1"))
			sb_puts(b, u);
		else if (plain_num(nb))
			sb_printf(b, "%s%s%s", nb, lg >= 3 ? " " : "", u);
		else if (strchr(nb, '/') && digits_only(strchr(nb, '/') + 1)) {
			/* 3/2 -> 3i/2 */
			const char *sl = strchr(nb, '/');
			char num[64];
			snprintf(num, sizeof num, "%.*s", (int)(sl - nb), nb);
			if (plain_num(num))
				sb_printf(b, "%s%s%s%s", strcmp(num, "1") ? num : "",
				    lg >= 3 && strcmp(num, "1") ? " " : "", u, sl);
			else
				sb_printf(b, "%s*%s", u, nb);
		} else
			sb_printf(b, "%s*%s", u, nb);
	}
	if (first)
		sb_puts(b, "0");
}

static const char *
hv_str(const double *x, int d, int lg)
{
	static struct sbuf b[6];
	static int k;
	struct sbuf *s = &b[k++ % 6];

	sb_clear(s);
	hv_text(s, x, d, lg);
	return sb_str(s);
}

static const char *const cut_desc_ops[][2] = {
	{"log", "log z = log|z| + i arg z with -pi < arg z <= pi"},
	{"sqrt", "sqrt z = e^(log(z)/2): i sqrt|z| on the negative reals (the "
	    "other root is its negative)"},
	{"^", "z^w = e^(w log z) with the principal log"},
	{"asin", "asin z = -i log(iz + sqrt(1 - z^2)) (cuts (-inf, -1) and "
	    "(1, inf))"},
	{"acos", "acos z = pi/2 - asin z (cuts (-inf, -1) and (1, inf))"},
	{"acosh", "acosh z = log(z + sqrt(z + 1) sqrt(z - 1)) (cut (-inf, 1))"},
	{"atanh", "atanh z = (log(1 + z) - log(1 - z))/2 (cuts (-inf, -1] and "
	    "[1, inf))"},
	{"atan", "atan z = (i/2)(log(1 - iz) - log(1 + iz)) (cuts on the "
	    "imaginary axis beyond +-i)"},
	{"asinh", "asinh z = log(z + sqrt(z^2 + 1)) (cuts on the imaginary "
	    "axis beyond +-i)"},
	{NULL, NULL}
};

/* the notes for what the evaluation met; d the dimension */
static void
hn_flush(int d)
{
	if (hn & HN_CUT) {
		const char *nm = opinfo[hn_cutop].name;
		int k;
		for (k = 0; cut_desc_ops[k][0]; k++)
			if (!strcmp(cut_desc_ops[k][0], nm))
				break;
		note("on the branch cut of %s: the principal value, %s",
		    hn_cutop == O_POW ? "a power" : nm,
		    cut_desc_ops[k][1] ? cut_desc_ops[k][1] : "");
	}
	if (hn & HN_DIR)
		note("a function of a negative real number in the %s: every unit "
		    "u with u^2 = -1 gives a value; u = i was taken",
		    alg_pl[d >= 32 ? 5 : ilog2u((uint64_t)d)]);
	if (hn & HN_RDIV)
		note("a/b is a b^-1 (right division); b^-1 a differs here, "
		    "since a b != b a");
	if (hn & HN_POW)
		note("a^b with a non-real exponent is e^(log(a) b) (log(a) on "
		    "the left); e^(b log a) differs here");
	if (hn & HN_ASSOC)
		note("octonion products are not associative: each product is "
		    "taken as grouped, (a b) c != a (b c) here");
	if (hn & HN_REAL)
		note("%s needs real operands: undefined for non-real values",
		    opinfo[hn_realop].name);
	hn = 0;
}

/* ---- classifying a statement ---- */

/* the reading of one identifier */
enum { HU_NONE, HU_I, HU_J, HU_K, HU_E };

struct hscan {
	int lg;			/* the algebra, log2 of its dimension */
	int f32;		/* c64: f32 parts */
	int unit_i, unit_j, unit_k, emax;	/* units used */
	int xy;			/* x or y appears */
	int oct_num;		/* 2e1 written for 2*e1 */
	/* names with a kind: annotated, defaulted or a hyper function's */
	int nn;
	char nm[16][24];
	int kind[16];		/* -1 real (said), else lg */
	int how[16];		/* 0 annotated, 1 default complex */
	uint32_t params[MAXP];	/* explicit parameters (not units) */
	int np;
	int hfun;		/* calls a hyper function */
	int keepk;		/* k is a unit even alone (a list of quaternions) */
};

static int
name_kind(const struct hscan *S, const char *s)
{
	int k;

	for (k = 0; k < S->nn; k++)
		if (!strcmp(S->nm[k], s))
			return k;
	return -1;
}

static void
add_name(struct hscan *S, const char *s, int kind, int how)
{
	int k = name_kind(S, s);

	if (k >= 0) {
		if (how == 0) {
			S->kind[k] = kind;
			S->how[k] = 0;
		}
		return;
	}
	if (S->nn == 16)
		return;
	snprintf(S->nm[S->nn], sizeof S->nm[0], "%s", s);
	S->kind[S->nn] = kind;
	S->how[S->nn++] = how;
}

/* a type word: hyper kind (lg >= 1), -1 real, -2 not a type */
static int
hyper_type(const char *s, int *f32)
{
	ty_t t;

	*f32 = 0;
	if (!strcmp(s, "complex") || !strcmp(s, "cplx") || !strcmp(s, "c128") ||
	    !strcmp(s, "C"))
		return 1;
	if (!strcmp(s, "c64")) {
		*f32 = 1;
		return 1;
	}
	if (!strcmp(s, "quat") || !strcmp(s, "quaternion") || !strcmp(s, "H"))
		return 2;
	if (!strcmp(s, "oct") || !strcmp(s, "octonion") || !strcmp(s, "O"))
		return 3;
	if (!strcmp(s, "sed") || !strcmp(s, "sedenion") || !strcmp(s, "cd16") ||
	    !strcmp(s, "S"))
		return 4;
	if (!strcmp(s, "cd32"))
		return 5;
	if (!strcmp(s, "cd2"))
		return 1;
	if (!strcmp(s, "cd4"))
		return 2;
	if (!strcmp(s, "cd8"))
		return 3;
	if (!strcmp(s, "cd1"))
		return -1;
	if (ty_parse(s, strlen(s), &t))
		return -1;
	return -2;
}

static int
default_complex(const char *s)
{
	if (!strcmp(s, "z") || !strcmp(s, "w") || !strcmp(s, "zeta"))
		return 1;
	if ((s[0] == 'z' || s[0] == 'w') && isdigit((unsigned char)s[1]) &&
	    s[2] == 0)
		return 1;
	if ((s[0] == 'z' || s[0] == 'w') && s[1] == '_' &&
	    isdigit((unsigned char)s[2]) && s[3] == 0)
		return 1;
	return 0;
}

/* e1 .. e31 (e0 is 1) */
static int
e_unit(const char *s)
{
	int n;

	if (s[0] != 'e' || !isdigit((unsigned char)s[1]))
		return -1;
	if (s[2] && (!isdigit((unsigned char)s[2]) || s[3]))
		return -1;
	n = atoi(s + 1);
	if (s[1] == '0' && s[2])
		return -1;
	return n < 32 ? n : -1;
}

static const char *const no_hyper_words[] = {"sum", "prod", "for", "while",
	"match", "switch", "loop", "if", "basis", "model", "pde", "where",
	"\xce\xa3", "\xe2\x88\x8f", "return", NULL};

static int
in_list(const char *const *l, const char *s)
{
	int k;

	for (k = 0; l[k]; k++)
		if (!strcmp(l[k], s))
			return 1;
	return 0;
}

static void
tok_text(const struct toks *t, uint32_t i, char *buf, size_t n)
{
	snprintf(buf, n, "%.*s", (int)t->len[i], t->src + t->off[i]);
}

static int
is_param(const struct hscan *S, const char *s)
{
	int k;

	for (k = 0; k < S->np; k++)
		if (!strcmp(sym_str(S->params[k]), s))
			return 1;
	return 0;
}

static int
known_fn(const char *s)
{
	static const char *const fns[] = {"sin", "cos", "tan", "exp", "log", "ln",
		"sqrt", "abs", "sec", "csc", "cot", "atan", "asin", "acos", "atan2",
		"sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "arcsin",
		"arccos", "arctan", "arsinh", "arcosh", "artanh", "conj", "re",
		"im", "arg", "norm", "cis", "floor", "ceil", "sign", "min", "max",
		"real", "int", "f32", "f64", NULL};
	return in_list(fns, s);
}

static uint32_t shadow;	/* a defined function named i, j or e1 */

/*
 * Scans tokens [a, b) of t.  Returns 0 when the statement must not be
 * read as hypercomplex (sums, loops, code blocks, ODEs).
 */
static int
hy_scan(const struct toks *t, uint32_t a, uint32_t b, struct hscan *S)
{
	uint32_t i;
	char w[64], w2[64];
	int k, f32;

	for (i = a; i < b; i++) {
		int kd = t->kind[i];
		if (kd == T_LB || kd == T_LC)
			return 0;
		if (kd == T_NUM && S) {
			/* 2e1: an octonion coefficient or a number? */
			tok_text(t, i, w, sizeof w);
			{
				char *e = strchr(w, 'e');
				if (e && isdigit((unsigned char)e[1]) && !strchr(w, '.') &&
				    e_unit(e) > 0)
					S->oct_num = 1;
			}
		}
		if (kd != T_ID)
			continue;
		tok_text(t, i, w, sizeof w);
		if (strchr(w, '\''))
			return 0;
		if (in_list(no_hyper_words, w))
			return 0;
		/* name: type */
		if (i + 2 < b && t->kind[i + 1] == T_COLON &&
		    t->kind[i + 2] == T_ID) {
			tok_text(t, i + 2, w2, sizeof w2);
			if (!strcmp(w2, "cd") && i + 5 < b && t->kind[i + 3] == T_LP &&
			    t->kind[i + 4] == T_NUM) {
				char nb[16];
				int n;
				tok_text(t, i + 4, nb, sizeof nb);
				n = atoi(nb);
				k = n <= 1 ? -1 : ilog2u((uint64_t)n);
				if (n > 1 && (1 << k) != n)
					fail("cd(%d): the Cayley-Dickson algebras have "
					    "dimension 1, 2, 4, 8, 16 or 32", n);
				if (k > MAXLG)
					fail("cd(%d) is too large (at most cd(32))", n);
				add_name(S, w, k, 0);
				i += 5;
				continue;
			}
			k = hyper_type(w2, &f32);
			if (k != -2) {
				add_name(S, w, k, 0);
				if (f32)
					S->f32 = 1;
				i += 2;
				continue;
			}
		}
		/* a call of a function */
		if (i + 1 < b && t->kind[i + 1] == T_LP && known_fn(w))
			continue;
		{
			struct func *g = func_find(sym_cstr(w));
			if (g && !g->halg && !is_param(S, w) && (!strcmp(w, "i") ||
			    !strcmp(w, "j") || e_unit(w) > 0))
				shadow = g->name;
			if (g && !is_param(S, w)) {
				if (g->halg) {
					S->hfun = 1;
					if (g->halg > S->lg)
						S->lg = g->halg;
				}
				continue;
			}
		}
		if (!strcmp(w, "x") || !strcmp(w, "y"))
			S->xy = 1;
		if (is_param(S, w))
			continue;
		if (!strcmp(w, "i") || !strcmp(w, "I") || !strcmp(w, "im"))
			S->unit_i = 1;
		else if (!strcmp(w, "j"))
			S->unit_j = 1;
		else if (!strcmp(w, "k"))
			S->unit_k = 1;
		else if ((k = e_unit(w)) >= 0) {
			if (k > S->emax)
				S->emax = k;
			if (k == 0 && !S->emax)
				S->emax = 0;
		}
	}
	return 1;
}

/* the algebra and the units; notes */
static void
hy_decide(struct hscan *S, const struct toks *t, uint32_t a, uint32_t b)
{
	uint32_t i;
	char w[64];
	int k;

	/* k alone is a variable; with i or j a quaternion unit */
	if (S->unit_k && !S->unit_i && !S->unit_j && !S->keepk)
		S->unit_k = 0;
	if (S->emax > 0) {
		int lg = 1;
		while ((1 << lg) <= S->emax)
			lg++;
		if (lg < 3)
			lg = 3;	/* e1, e2, e3 alone are still octonions */
		if (lg > S->lg)
			S->lg = lg;
	} else if ((S->unit_i && (S->unit_j || S->unit_k)) ||
	    (S->unit_j && S->unit_k)) {
		if (S->lg < 2)
			S->lg = 2;
	} else if (S->unit_i || S->unit_j) {
		if (S->lg < 1)
			S->lg = 1;
	}
	for (k = 0; k < S->nn; k++)
		if (S->kind[k] > S->lg)
			S->lg = S->kind[k];
	/* z, w: complex unless x or y is there (a point, a surface z = f(x, y)) */
	if (!S->xy)
		for (i = a; i < b; i++) {
			if (t->kind[i] != T_ID)
				continue;
			tok_text(t, i, w, sizeof w);
			if (default_complex(w) && name_kind(S, w) < 0 &&
			    !func_find(sym_cstr(w))) {
				add_name(S, w, 1, 1);
				if (S->lg < 1)
					S->lg = 1;
			}
		}
}

static void
hy_notes(const struct hscan *S)
{
	struct sbuf b = {0};
	int k, m = 0;

	for (k = 0; k < S->nn; k++)
		if (S->how[k] == 1)
			sb_printf(&b, "%s%s", m++ ? ", " : "", S->nm[k]);
	if (m)
		note("%s %s taken as complex (%s; say %s: real for a real "
		    "variable)", b.p, m > 1 ? "are" : "is",
		    m > 1 ? "no x or y here" : "no x or y here", S->nm[0]);
	if (S->lg == 1 && S->unit_j && !S->unit_i)
		note("j is read as the imaginary unit (j^2 = -1)");
	if (S->lg >= 3 && (S->unit_i || S->unit_j || S->unit_k))
		note("i, j, k are read as e1, e2, e3 in the %s", alg_pl[S->lg]);
	if (S->lg >= 3 && S->oct_num)
		note("2e1 is read as 2*e1 (a unit), not the number 20; write "
		    "20 or 2.0e1 for the number");
	if (S->lg == 2 && S->unit_k)
		note("i, j, k are the quaternion units: i^2 = j^2 = k^2 = ijk = -1, "
		    "ij = k = -ji (say k: real for a variable k)");
	else if (S->lg == 2 && (S->unit_i && S->unit_j))
		note("i, j are quaternion units: ij = k = -ji");
	if (S->lg >= 4)
		note("the %s have zero divisors: a b = 0 with a, b != 0 is "
		    "possible, and |ab| != |a||b|", alg_pl[S->lg]);
	free(b.p);
}

/* ---- text: units, annotations and aliases rewritten for the parser ---- */

static int
ident_ch(int c)
{
	return isalnum((unsigned char)c) || c == '_';
}

static size_t
paren_end(const char *s, size_t i)
{
	int d = 0;

	for (; s[i]; i++) {
		if (s[i] == '(')
			d++;
		else if (s[i] == ')' && --d == 0)
			return i + 1;
	}
	return i;
}

static const char *
canon_unit(const struct hscan *S, const char *w)
{
	static char b[8];

	if (S->lg == 0)
		return NULL;
	if (!strcmp(w, "i") || !strcmp(w, "I") || !strcmp(w, "im"))
		return S->unit_i ? (S->lg >= 3 ? "e1" : "i") : NULL;
	if (!strcmp(w, "j") && S->unit_j)
		return S->lg == 1 ? "i" : S->lg == 2 ? "j" : "e2";
	if (!strcmp(w, "k") && S->unit_k)
		return S->lg == 2 ? "k" : "e3";
	if (e_unit(w) >= 0 && S->lg >= 3) {
		snprintf(b, sizeof b, "%s", w);
		return b;
	}
	return NULL;
}

static void
hy_prep(const char *in, struct sbuf *o, const struct hscan *S)
{
	size_t i = 0, j, n;

	while (in[i]) {
		unsigned char c = in[i];
		if (isdigit(c) || (c == '.' && isdigit((unsigned char)in[i + 1]))) {
			while (isdigit((unsigned char)in[i]) || in[i] == '.' ||
			    in[i] == '_')
				sb_putc(o, in[i++]);
			if ((in[i] == 'e' || in[i] == 'E') &&
			    isdigit((unsigned char)in[i + 1])) {
				if (S->lg >= 3) {
					sb_putc(o, ' ');	/* 2e1 = 2*e1 */
					continue;
				}
				sb_putc(o, in[i++]);
				while (isdigit((unsigned char)in[i]))
					sb_putc(o, in[i++]);
			} else if ((in[i] == 'e' || in[i] == 'E') &&
			    (in[i + 1] == '-' || in[i + 1] == '+') &&
			    isdigit((unsigned char)in[i + 2])) {
				sb_putc(o, in[i++]);
				sb_putc(o, in[i++]);
				while (isdigit((unsigned char)in[i]))
					sb_putc(o, in[i++]);
			}
			continue;
		}
		if (!(isalpha(c) || c == '_')) {
			sb_putc(o, in[i++]);
			continue;
		}
		for (j = i; ident_ch(in[j]); j++)
			;
		n = j - i;
		{
			char w[64];
			const char *cu;
			size_t k = j;
			snprintf(w, sizeof w, "%.*s", (int)n, in + i);
			while (in[k] == ' ')
				k++;
			/* name: complex (a hyper annotation goes) */
			if (in[k] == ':') {
				size_t a = k + 1, e;
				char tw[32];
				int f32, ty;
				while (in[a] == ' ')
					a++;
				for (e = a; ident_ch(in[e]); e++)
					;
				snprintf(tw, sizeof tw, "%.*s", (int)(e - a), in + a);
				ty = hyper_type(tw, &f32);
				if (!strcmp(tw, "cd") || ty >= 1 ||
				    (ty == -1 && name_kind(S, w) >= 0 &&
				    S->kind[name_kind(S, w)] == -1 &&
				    (!strcmp(tw, "real") || !strcmp(tw, "cd1")))) {
					sb_puts(o, w);
					i = e;
					if (!strcmp(tw, "cd")) {
						while (in[i] == ' ')
							i++;
						if (in[i] == '(')
							i = paren_end(in, i);
					}
					continue;
				}
			}
			if ((cu = canon_unit(S, w)) != NULL &&
			    !(in[k] == '(' && !strcmp(w, "im"))) {
				sb_puts(o, cu);
				i = j;
				continue;
			}
			if (!strcmp(w, "e0") && S->lg >= 3) {
				sb_puts(o, "1");
				i = j;
				continue;
			}
			if (!strcmp(w, "norm") && in[k] == '(') {
				sb_puts(o, "abs");
				i = j;
				continue;
			}
			if (!strcmp(w, "cis") && in[k] == '(') {
				size_t e = paren_end(in, k);
				struct sbuf t = {0};
				char *arg = xmalloc(e - k + 1);
				memcpy(arg, in + k, e - k);
				arg[e - k] = 0;
				hy_prep(arg, &t, S);
				sb_printf(o, "exp(%s*%s)", S->lg >= 3 ? "e1" : "i",
				    sb_str(&t));
				free(arg);
				free(t.p);
				i = e;
				continue;
			}
			sb_puts(o, w);
			i = j;
		}
	}
}

/* ---- parsing into a function ---- */

static int
unit_index(const char *s, int lg)
{
	if (lg == 1)
		return !strcmp(s, "i") ? 1 : -1;
	if (lg == 2) {
		if (!strcmp(s, "i"))
			return 1;
		if (!strcmp(s, "j"))
			return 2;
		if (!strcmp(s, "k"))
			return 3;
		return -1;
	}
	{
		int u = e_unit(s);
		return u >= 1 && u < (1 << lg) ? u : -1;
	}
}

/* nm (canonical) is a unit of this statement, not a variable */
static int
canon_ok(const struct hscan *S, const char *nm)
{
	if (!strcmp(nm, "i"))
		return S->unit_i || (S->lg == 1 && S->unit_j);
	if (!strcmp(nm, "j"))
		return S->unit_j;
	if (!strcmp(nm, "k"))
		return S->unit_k;
	if (!strcmp(nm, "e1"))
		return S->lg >= 3;
	if (!strcmp(nm, "e2"))
		return S->lg >= 3;
	if (!strcmp(nm, "e3"))
		return S->lg >= 3;
	return e_unit(nm) >= 1 && S->lg >= 3;
}

/*
 * Parse body text into f (whose first nexp parameters are given),
 * moving the units to their slots and giving each parameter its kind.
 * defname: the function being defined (free variables are an error
 * when it has an explicit parameter list).
 */
static void
hy_parse(const char *body, struct func *f, int nexp, struct hscan *S,
    int explicit_list)
{
	struct sbuf t = {0};
	struct func tmp;
	nid map[MAXP], e;
	int k, np = 0;
	struct sbuf fv = {0};

	hy_prep(body, &t, S);
	tmp = *f;
	e = calc_parse_text(sb_str(&t), &tmp, TY_REAL);
	free(t.p);
	memset(f->p, 0, sizeof f->p);
	for (k = 0; k < tmp.np; k++) {
		const char *nm = sym_str(tmp.p[k].name);
		int u = k >= nexp ? unit_index(nm, S->lg) : -1;
		if (u >= 1 && canon_ok(S, nm)) {
			map[k] = mk_var(UB + u, tmp.p[k].name, TY_REAL);
			continue;
		}
		if (k >= nexp && explicit_list)
			fail("'%s' is not defined", nm);
		if (k >= nexp)
			sb_printf(&fv, "%s%s", fv.n ? ", " : "", nm);
		f->p[np] = tmp.p[k];
		if (TY_ISINT(f->p[np].ty) && name_kind(S, nm) >= 0 &&
		    S->kind[name_kind(S, nm)] >= 1)
			f->p[np].ty = TY_REAL;
		{
			int nk = name_kind(S, nm);
			f->hk[np] = nk >= 0 && S->kind[nk] > 0 ? (uint8_t)S->kind[nk] : 0;
		}
		map[k] = mk_var((uint32_t)np, tmp.p[k].name, f->p[np].ty);
		np++;
	}
	f->np = np;
	f->body = subst(e, map, tmp.np);
	f->halg = (uint8_t)S->lg;
	f->hf32 = (uint8_t)S->f32;
	if (fv.n && !f->anon && nexp == 0)
		note("%s has free variables: defined as %s(%s)", sym_str(f->name),
		    sym_str(f->name), fv.p);
	free(fv.p);
}

/* ---- statements ---- */

static const char *const ucas_cmds[] = {"show", "table", "range", "domain",
	"wrap", "crit", "diff", "equiv", "gates", "find", "fit", "error",
	"cost", "plot", "plot3d", "neighbourhood", "neighborhood", "animate",
	"approx", "help", "funcs", "models", "use", "more", "parse", "closed",
	"scheme", "continuous", "integrate", "dsolve", "monotonic", "monotone",
	"period", "periodic", "identify", "classify", "parity", "symmetry",
	"bounded", "injective", "surjective", "bijective", "continuity",
	"discontinuities", "convexity", "convex", "concave", "asymptotes",
	"zeros", "roots", "complex", "rect", "polar", "conj", "re", "im",
	"modulus", "arg", "nthroots", "unity", "trig", "spvsimplify", "limit",
	"lim", "taylor", "maclaurin", "implicit", "antiderivative",
	"primitive", "grad", "gradient", "hessian", "laplacian", "jacobian",
	"ddir", "wronskian", "partial", "basis", "model", "assume", "forget",
	/* alg.c */
	"expand", "factor", "collect", "simplify", "together", "apart",
	"cancel", "gcd", "lcm", "resultant", "discriminant", "quo", "rem",
	"divide", "solve", "substitute", "subst",
	/* ring.c */
	"ring", "inverse", "order", "phi", "carmichael", "factorint",
	"congruence", "crt", "legendre", "jacobi", "sqrtmod", "qr", "primroot",
	"dlog", "magic", "irreducible", "det", "rank", "nullspace", "matinv",
	"matsolve", "gf2matrix", "matrix", "norm", "poly",
	"zmod", "gf", "GF", "totient", "multorder", "invmod",
	"modinv", "primroots", "gf2map", "kernel", "inv", "ord",
	"derivative", "series",
	/* xform.c (z, s, w are the transforms' own variables there) */
	"laplace", "ilaplace", "ztrans", "iztrans", "zsolve", "tf", "fourier",
	"ifourier", "fseries", "dft", "idft", "fft", "ifft", "dtft", "genfunc",
	"ogf", "egf", "converge", "converges", "radius", "powerseries",
	"coeffs", "asymptotic",
	/* viz.c */
	"area", "revolve", "volume", "surfacearea", "arclength", "centroid",
	/* dword.c, fpfold.c */
	"dword", "foldproof",
	NULL};

static const char *
kind_str(const struct func *f, int j)
{
	int k = j < 0 ? f->halg : f->hk[j];

	if (k == 0)
		return j < 0 ? "real" : ty_str(f->p[j].ty);
	if (k == 1 && f->hf32)
		return "c64";
	return alg_name[k];
}

static void
hy_show(const struct func *f)
{
	struct sbuf b = {0};
	int j;

	sb_printf(&b, "%s(", f->anon ? "_" : sym_str(f->name));
	for (j = 0; j < f->np; j++)
		sb_printf(&b, "%s%s: %s", j ? ", " : "", sym_str(f->p[j].name),
		    kind_str(f, j));
	sb_puts(&b, ") = ");
	print_node(&b, f->body, NULL);
	printf("%s\n  -> %s\n", b.p, kind_str(f, -1));
	free(b.p);
}

/* sample points for a parameter of kind k (lg) */
static int
sample_points(const struct func *f, int j, double (*pt)[HM], int max)
{
	int k = f->hk[j], n = 0, a, b, u;

	memset(pt, 0, (size_t)max * sizeof *pt);
	if (k == 0) {
		const struct param *p = &f->p[j];
		double lo = -2, hi = 2;
		if (p->ranged) {
			lo = TY_FP(p->ty) ? p->rlo : (double)p->lo;
			hi = TY_FP(p->ty) ? p->rhi : (double)p->hi;
		}
		for (a = 0; a < 5 && n < max; a++)
			pt[n++][0] = lo + (hi - lo) * a / 4;
		return n;
	}
	if (k == 1) {
		static const int ord[][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1},
			{0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {2, 0}, {0, 2}};
		for (a = 0; a < 11 && n < max; a++) {
			pt[n][0] = ord[a][0];
			pt[n++][1] = ord[a][1];
		}
		return n;
	}
	/* 0, 1, each unit, 1 + each unit, the sum of all over 2 */
	pt[n++][0] = 0;
	pt[n++][0] = 1;
	for (u = 1; u < (1 << k) && n < max; u++)
		pt[n++][u] = 1;
	for (u = 1; u < (1 << k) && u < 4 && n < max; u++) {
		pt[n][0] = 1;
		pt[n++][u] = 1;
	}
	if (n < max) {
		for (b = 0; b < (1 << k) && b < 4; b++)
			pt[n][b] = 0.5;
		n++;
	}
	return n;
}

static void
hy_table(const struct func *f, int max)
{
	struct hp P;
	static double pts[MAXP][24][HM];
	int cnt[MAXP], idx[MAXP], j, shown = 0, d = 1 << f->halg;
	double in[MAXP * HM], out[HM];
	nid root = f->body;

	if (f->np == 0)
		fail("table: %s has no parameters", sym_str(f->name));
	for (j = 0; j < f->np; j++) {
		cnt[j] = sample_points(f, j, pts[j], 24);
		idx[j] = 0;
	}
	hp_build(&P, &root, 1, d, f->np);
	P.f32 = f->hf32 == 1;
	for (;;) {
		struct sbuf b = {0};
		memset(in, 0, sizeof in);
		for (j = 0; j < f->np; j++)
			memcpy(in + j * d, pts[j][idx[j]], (size_t)d * sizeof(double));
		hp_run(&P, in);
		for (j = 0; j < f->np; j++)
			sb_printf(&b, "%s%s = %s", j ? ", " : "", sym_str(f->p[j].name),
			    hv_str(in + j * d, d, f->halg));
		if (hp_get(&P, 0, out))
			printf("  %s  ->  %s\n", b.p, hv_str(out, d, f->halg));
		else
			printf("  %s  ->  undefined\n", b.p);
		free(b.p);
		if (++shown >= max)
			break;
		for (j = 0; j < f->np; j++) {
			if (++idx[j] < cnt[j])
				break;
			idx[j] = 0;
		}
		if (j == f->np)
			break;
	}
	if (shown >= max)
		printf("  ... (table max %d)\n", max);
	hp_free(&P);
	hn_flush(d);
}

/* evaluate a function with no parameters; 1 when defined */
static int
hy_value(const struct func *f, double *out)
{
	struct hp P;
	nid root = f->body;
	int ok;

	hp_build(&P, &root, 1, 1 << f->halg, 0);
	P.f32 = f->hf32 == 1;
	hp_run(&P, NULL);
	ok = hp_get(&P, 0, out);
	hp_free(&P);
	return ok;
}

/* the token after the matching ')' of the '(' at i */
static uint32_t
tok_close(const struct toks *t, uint32_t i, uint32_t b)
{
	int d = 0;

	for (; i < b; i++) {
		if (t->kind[i] == T_LP)
			d++;
		else if (t->kind[i] == T_RP && --d == 0)
			return i + 1;
	}
	return b;
}

static const char *
tok_src(const struct toks *t, uint32_t i)
{
	return t->src + t->off[i];
}

/* text of tokens [a, b) */
static char *
toks_text(const struct toks *t, uint32_t a, uint32_t b)
{
	size_t n;
	char *s;

	if (b <= a)
		return strdup("");
	n = t->off[b - 1] + t->len[b - 1] - t->off[a];
	s = xmalloc(n + 1);
	memcpy(s, t->src + t->off[a], n);
	s[n] = 0;
	return s;
}

/*
 * A definition: NAME [ (params) ] = body, hypercomplex.  Returns 0 when
 * it is an ordinary one.
 */
static int
hy_def(const struct toks *t, uint32_t b)
{
	struct hscan S;
	struct func tmp, *nf;
	uint32_t eq, i, bodyat;
	char name[64], *body;
	struct sbuf clean = {0};
	int nexp = 0, explicit_list = 0;

	memset(&S, 0, sizeof S);
	tok_text(t, 0, name, sizeof name);
	if (t->kind[1] == T_ASSIGN)
		eq = 1;
	else if (t->kind[1] == T_LP) {
		uint32_t a = 2, c = tok_close(t, 1, b);
		if (c >= b || t->kind[c] != T_ASSIGN)
			return 0;
		eq = c;
		explicit_list = 1;
		/* the parameters: NAME [: type] [in ...], ... */
		while (a < c - 1) {
			uint32_t e = a;
			int depth = 0;
			char pn[64];
			while (e < c - 1 && !(depth == 0 && t->kind[e] == T_COMMA)) {
				if (t->kind[e] == T_LP)
					depth++;
				if (t->kind[e] == T_RP)
					depth--;
				e++;
			}
			if (t->kind[a] != T_ID)
				return 0;	/* y(0) = 3: an initial value */
			tok_text(t, a, pn, sizeof pn);
			/*
			 * The translator's it(i) = i^i: an untyped i of a translated
			 * definition is the unit, not a parameter.
			 */
			if (e == a + 1 && (!strcmp(name, "it") || !strcmp(name, "that") ||
			    !strcmp(name, "lhs") || !strcmp(name, "rhs")) &&
			    (!strcmp(pn, "i") || !strcmp(pn, "j") || !strcmp(pn, "k") ||
			    e_unit(pn) > 0)) {
				a = e + 1;
				continue;
			}
			if (nexp == MAXP)
				return 0;
			S.params[S.np++] = sym_cstr(pn);
			nexp++;
			if (e > a + 2 && t->kind[a + 1] == T_COLON &&
			    t->kind[a + 2] == T_ID) {
				char tw[32];
				int f32;
				tok_text(t, a + 2, tw, sizeof tw);
				if (!strcmp(tw, "cd") || hyper_type(tw, &f32) >= 1) {
					sb_printf(&clean, "%s%s", clean.n ? ", " : "", pn);
					a = e + 1;
					continue;
				}
			}
			{
				char *s = toks_text(t, a, e);
				sb_printf(&clean, "%s%s", clean.n ? ", " : "", s);
				free(s);
			}
			a = e + 1;
		}
	} else
		return 0;
	bodyat = eq + 1;
	if (bodyat >= b)
		return 0;
	if (!hy_scan(t, 1, b, &S))
		return 0;
	hy_decide(&S, t, bodyat, b);
	/* parameters named like complex variables get their default too */
	if (!S.xy)
		for (i = 0; i < (uint32_t)S.np; i++) {
			const char *pn = sym_str(S.params[i]);
			if (default_complex(pn) && name_kind(&S, pn) < 0) {
				add_name(&S, pn, 1, 1);
				if (S.lg < 1)
					S.lg = 1;
			}
		}
	if (S.lg == 0) {
		free(clean.p);
		return 0;
	}
	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr(name);
	if (nexp) {
		/* the plain parameters (types, ranges) as an ordinary
		 * definition reads them */
		struct sbuf d = {0};
		struct func *pf;
		sb_printf(&d, "__hyper_params(%s) = 0\n", clean.p);
		run_source(d.p, "<hyper>", 0);
		free(d.p);
		pf = func_find(sym_cstr("__hyper_params"));
		if (pf == NULL || pf->np != nexp)
			fail("%s: the parameter list does not parse", name);
		memcpy(tmp.p, pf->p, sizeof tmp.p);
		tmp.np = nexp;
	}
	free(clean.p);
	body = toks_text(t, bodyat, b);
	hy_notes(&S);
	hy_parse(body, &tmp, nexp, &S, explicit_list);
	free(body);
	nf = func_new(tmp.name);
	*nf = tmp;
	return 1;
}

/* a bare expression: its value, or the function it defines */
static int
hy_bare(const struct toks *t, uint32_t b)
{
	struct hscan S;
	struct func tmp, *nf;
	double v[HM];
	char *text;

	memset(&S, 0, sizeof S);
	if (!hy_scan(t, 0, b, &S))
		return 0;
	hy_decide(&S, t, 0, b);
	text = toks_text(t, 0, b);
	if (S.lg == 0) {
		/* sqrt(-4), log(-1), asin(2): undefined in the reals */
		struct func rf;
		nid e;
		int ok = 0;
		jmp_buf saved;
		memset(&rf, 0, sizeof rf);
		memcpy(saved, fail_jmp, sizeof saved);
		if (setjmp(fail_jmp) == 0) {
			e = calc_parse_text(text, &rf, TY_REAL);
			ok = 1;
		}
		memcpy(fail_jmp, saved, sizeof saved);
		notes_drop();
		if (!ok || rf.np || is_const(e) || !(N.fl[e] & NF_PARTIAL)) {
			free(text);
			return 0;
		}
		eval_const_at(e, 0, 0, &ok);
		if (ok) {
			free(text);
			return 0;
		}
		S.lg = 1;
		memset(&tmp, 0, sizeof tmp);
		tmp.name = sym_cstr("_");
		tmp.anon = 1;
		hy_parse(text, &tmp, 0, &S, 0);
		if (!hy_value(&tmp, v)) {
			free(text);
			hn = 0;
			return 0;
		}
		note("%s is undefined in the reals; its complex principal value "
		    "follows", text);
		hn_flush(2);
		notes_flush();
		printf("%s\n", hv_str(v, 2, 1));
		free(text);
		return 1;
	}
	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr("_");
	tmp.anon = 1;
	hy_notes(&S);
	hy_parse(text, &tmp, 0, &S, 0);
	free(text);
	notes_flush();
	if (tmp.np == 0) {
		int d = 1 << tmp.halg;
		if (hy_value(&tmp, v)) {
			hn_flush(d);
			notes_flush();
			printf("%s\n", hv_str(v, d, tmp.halg));
		} else {
			hn_flush(d);
			notes_flush();
			printf("undefined\n");
		}
		return 1;
	}
	nf = func_new(tmp.name);
	*nf = tmp;
	hy_show(nf);
	return 1;
}

static int hy_command(const struct toks *t, uint32_t b, const char *w0);

int
hyper_statement(const char *stmt)
{
	struct toks *t = xcalloc(1, sizeof *t);
	char *text = strdup(stmt), w0[32] = "";
	uint32_t b;
	int r = 0;

	lex(t, text);
	shadow = 0;
	for (b = 0; b < t->n && t->kind[b] != T_EOF && t->kind[b] != T_NL; b++)
		;
	if (b == 0)
		goto out;
	if (t->kind[0] == T_ID)
		tok_text(t, 0, w0, sizeof w0);
	if (!strcmp(w0, "__hyper_params") || strchr(w0, '\''))
		goto out;	/* y' = ...: an ODE (calc) */
	if (t->kind[0] == T_ID && (r = hy_command(t, b, w0)) != 0) {
		r = r > 0;
		goto out;
	}
	if (t->kind[0] == T_ID && (t->kind[1] == T_ASSIGN ||
	    t->kind[1] == T_LP)) {
		uint32_t c = t->kind[1] == T_LP ? tok_close(t, 1, b) : 1;
		if (c < b && t->kind[c] == T_ASSIGN) {
			r = hy_def(t, b);
			goto out;
		}
	}
	if (t->kind[0] == T_ID && in_list(ucas_cmds, w0) && b > 1 &&
	    t->kind[1] != T_PLUS && t->kind[1] != T_STAR &&
	    t->kind[1] != T_SLASH && t->kind[1] != T_CARET &&
	    t->kind[1] != T_ASSIGN)
		goto out;
	r = hy_bare(t, b);
out:
	if (!r && shadow)
		note("%s is defined (%s), so it is that here, not a unit; rename "
		    "it to use %s as the imaginary unit", sym_str(shadow),
		    func_find(shadow) ? "a function or value" : "", sym_str(shadow));
	shadow = 0;
	toks_free(t);
	free(t);
	free(text);
	return r;
}

/* ---- the ops this file adds, on reals ---- */

int
hyper_scalar(int op, double a, double *r)
{
	switch (op) {
	case O_CONJ:
	case O_RE:
		*r = a;
		return 1;
	case O_IM:
		*r = 0;
		return 1;
	case O_ARG:
		if (a == 0 || isnan(a))
			return 0;
		*r = a > 0 ? 0 : M_PI;
		return 1;
	}
	return 0;
}

nid
hyper_diff(nid e, uint32_t slot)
{
	switch (N.op[e]) {
	case O_CONJ:
	case O_RE:
		return diff_node(N.a[e], slot);
	}
	return mk_const_i(TY_LIT, 0);	/* im, arg: constant on the reals */
}

/* ---- commands ---- */

/*
 * The function argument of a command, tokens [a, b): a hypercomplex
 * function's name, or an expression (an equation L = R is L - R).
 * NULL when it is not hypercomplex (the command is not ours then).
 */
static struct func *
hy_funarg(const struct toks *t, uint32_t a, uint32_t b, int force)
{
	static unsigned turn;
	struct hscan S;
	struct func tmp, *g;
	char w[64], *text;
	uint32_t i, eq = 0;
	int depth = 0;

	if (a >= b)
		return NULL;
	if (b == a + 1 && t->kind[a] == T_ID) {
		tok_text(t, a, w, sizeof w);
		if ((g = func_find(sym_cstr(w))) != NULL)
			return g->halg ? g : NULL;
	}
	memset(&S, 0, sizeof S);
	if (!hy_scan(t, a, b, &S))
		return NULL;
	hy_decide(&S, t, a, b);
	if (S.lg == 0 && force)
		S.lg = 1;
	if (S.lg == 0)
		return NULL;
	for (i = a; i < b; i++) {
		if (t->kind[i] == T_LP)
			depth++;
		if (t->kind[i] == T_RP)
			depth--;
		if (depth == 0 && t->kind[i] == T_ASSIGN)
			eq = i;
	}
	if (eq) {
		char *l = toks_text(t, a, eq), *r = toks_text(t, eq + 1, b);
		text = xmalloc(strlen(l) + strlen(r) + 16);
		sprintf(text, "(%s) - (%s)", l, r);
		free(l);
		free(r);
	} else
		text = toks_text(t, a, b);
	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr(turn++ & 1 ? "_2" : "_");
	tmp.anon = 1;
	hy_notes(&S);
	hy_parse(text, &tmp, 0, &S, 0);
	free(text);
	if (force && tmp.np == 1 && tmp.hk[0] == 0 && tmp.halg <= 1) {
		note("%s is taken as a complex variable here", sym_str(tmp.p[0].name));
		tmp.hk[0] = 1;
		tmp.halg = 1;
	}
	g = func_new(tmp.name);
	*g = tmp;
	return g;
}

/* the first token in [a, b) at depth 0 that is one of words, else b */
static uint32_t
find_word(const struct toks *t, uint32_t a, uint32_t b,
    const char *const *words)
{
	uint32_t i;
	int depth = 0;
	char w[32];

	for (i = a; i < b; i++) {
		if (t->kind[i] == T_LP || t->kind[i] == T_LB)
			depth++;
		if (t->kind[i] == T_RP || t->kind[i] == T_RB)
			depth--;
		if (depth == 0 && t->kind[i] == T_ID) {
			tok_text(t, i, w, sizeof w);
			if (in_list(words, w))
				return i;
		}
	}
	return b;
}

/* a constant (hypercomplex) expression in tokens [a, b) */
static int
hy_const(const struct toks *t, uint32_t a, uint32_t b, int lgmin,
    double *v, int *lgout)
{
	struct hscan S;
	struct func tmp;
	char *text;
	int ok;

	memset(&S, 0, sizeof S);
	S.keepk = lgmin >= 2;
	if (a >= b)
		fail("a number is missing");
	if (!hy_scan(t, a, b, &S))
		fail("not a number");
	hy_decide(&S, t, a, b);
	if (S.lg < lgmin)
		S.lg = lgmin;
	text = toks_text(t, a, b);
	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr("_");
	tmp.anon = 1;
	hy_parse(text, &tmp, 0, &S, 0);
	if (tmp.np)
		fail("%s: a constant is needed (%s is a variable)", text,
		    sym_str(tmp.p[0].name));
	free(text);
	memset(v, 0, HM * sizeof *v);
	ok = hy_value(&tmp, v);
	if (lgout)
		*lgout = tmp.halg;
	return ok;
}

/* ---- one complex variable ---- */

static struct hp FP;
static int fp_slot, fp_np;
static double fp_in[MAXP * HM];
static struct func *fp_f;

static void
fz_open(struct func *f, const char *cmd)
{
	int j, slot = -1;

	if (f->halg != 1)
		fail("%s works on complex functions; %s is %s-valued", cmd,
		    f->anon ? "this" : sym_str(f->name), alg_name[f->halg]);
	for (j = 0; j < f->np; j++)
		if (f->hk[j] == 1) {
			if (slot >= 0)
				fail("%s: one complex variable is needed (%s has %s and "
				    "%s)", cmd, f->anon ? "the expression" :
				    sym_str(f->name), sym_str(f->p[slot].name),
				    sym_str(f->p[j].name));
			slot = j;
		} else
			fail("%s: %s is a real parameter; give it a value (%s has "
			    "to be a function of one complex variable)", cmd,
			    sym_str(f->p[j].name), f->anon ? "the expression" :
			    sym_str(f->name));
	if (slot < 0)
		fail("%s: no complex variable", cmd);
	hp_build(&FP, &f->body, 1, 2, f->np);
	FP.f32 = f->hf32 == 1;
	fp_slot = slot;
	fp_np = f->np;
	fp_f = f;
	memset(fp_in, 0, sizeof fp_in);
}

static int
fz(cplx z, cplx *w)
{
	double o[2];

	fp_in[fp_slot * 2] = creal(z);
	fp_in[fp_slot * 2 + 1] = cimag(z);
	hp_run(&FP, fp_in);
	if (!hp_get(&FP, 0, o))
		return 0;
	*w = o[0] + I * o[1];
	return 1;
}

static void
fz_close(void)
{
	hp_free(&FP);
	hn = 0;
}

static const char *
zs(cplx z)
{
	double v[2];

	v[0] = creal(z);
	v[1] = cimag(z);
	return hv_str(v, 2, 1);
}

static const char *
fname(const struct func *f)
{
	static char b[4][600];
	static int k;
	char *s = b[k++ & 3];

	if (f->anon) {
		struct sbuf t = {0};
		print_node(&t, f->body, NULL);
		snprintf(s, 600, "%s", t.p);
		free(t.p);
	} else
		snprintf(s, 600, "%s", sym_str(f->name));
	return s;
}

static const char *
vname(void)
{
	return sym_str(fp_f->p[fp_slot].name);
}

static uint64_t hrng = 0x9e3779b97f4a7c15ULL;
static cplx snap(cplx z);

static double
hrand(void)
{
	hrng = hrng * 6364136223846793005ULL + 1442695040888963407ULL;
	return (double)(hrng >> 11) / 9007199254740992.0;
}

/* f' at z by central differences along 1 and along i */
static int
fz_cr(cplx z, cplx *fx, cplx *fy)
{
	cplx a, b, c, d;
	double h = 1e-5 * (cabs(z) > 1 ? cabs(z) : 1);

	if (!fz(z + h, &a) || !fz(z - h, &b) || !fz(z + I * h, &c) ||
	    !fz(z - I * h, &d))
		return 0;
	*fx = (a - b) / (2 * h);
	*fy = (c - d) / (2 * h * I);
	return 1;
}

static int
has_op(nid e, int op, uint8_t *seen)
{
	int ar, r = 0;

	if (seen[e])
		return 0;
	seen[e] = 1;
	if (N.op[e] == op)
		return 1;
	ar = opinfo[N.op[e]].arity;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return 0;
	if (ar >= 1)
		r |= has_op(N.a[e], op, seen);
	if (ar >= 2)
		r |= has_op(N.b[e], op, seen);
	if (ar >= 3)
		r |= has_op(N.c[e], op, seen);
	return r;
}

static int
uses(nid e, int op)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int r = has_op(e, op, seen);

	free(seen);
	return r;
}

/* fractional powers: z^(1/2) has a cut, z^3 does not */
static int
frac_pow(nid e, uint8_t *seen)
{
	int ar, r = 0;
	double v;

	if (seen[e])
		return 0;
	seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return 0;
	if (N.op[e] == O_POW && (N.fl[N.a[e]] & NF_VAR) &&
	    !(num_of(N.b[e], &v) && v == floor(v)))
		return 1;
	ar = opinfo[N.op[e]].arity;
	if (ar >= 1)
		r |= frac_pow(N.a[e], seen);
	if (ar >= 2)
		r |= frac_pow(N.b[e], seen);
	if (ar >= 3)
		r |= frac_pow(N.c[e], seen);
	return r;
}

/* z^-2: a pole */
static int
neg_pow(nid e, uint8_t *seen)
{
	int ar, r = 0;
	double v;

	if (seen[e])
		return 0;
	seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return 0;
	if (N.op[e] == O_POW && (N.fl[N.a[e]] & NF_VAR) &&
	    !(num_of(N.b[e], &v) && v >= 0))
		return 1;
	ar = opinfo[N.op[e]].arity;
	if (ar >= 1)
		r |= neg_pow(N.a[e], seen);
	if (ar >= 2)
		r |= neg_pow(N.b[e], seen);
	if (ar >= 3)
		r |= neg_pow(N.c[e], seen);
	return r;
}

/* where holomorphy can fail: "the branch cut of log", "poles" */
static void
where_not(nid body, struct sbuf *b)
{
	static const int cuts[] = {O_LOG, O_SQRT, O_ASIN, O_ACOS, O_ATAN,
		O_ASINH, O_ACOSH, O_ATANH, 0};
	int k, m = 0;
	uint8_t *seen;

	for (k = 0; cuts[k]; k++)
		if (uses(body, cuts[k]))
			sb_printf(b, "%s%s", m++ ? ", " : "", opinfo[cuts[k]].name);
	seen = xcalloc(N.n, 1);
	if (frac_pow(body, seen))
		sb_printf(b, "%s%s", m++ ? ", " : "", "non-integer powers");
	free(seen);
	if (m)
		sb_puts(b, " (the branch cuts of the principal values)");
	seen = xcalloc(N.n, 1);
	k = neg_pow(body, seen);
	free(seen);
	if (k || uses(body, O_DIV) || uses(body, O_TAN) || uses(body, O_SEC) ||
	    uses(body, O_CSC) || uses(body, O_COT) || uses(body, O_TANH))
		sb_printf(b, "%sthe points where it is undefined (poles)",
		    m ? "; " : "");
}

/* nonholomorphic parts */
static int
nonhol_ops(nid body, struct sbuf *b)
{
	static const int ops[] = {O_CONJ, O_RE, O_IM, O_ABS, O_ARG, 0};
	int k, m = 0;

	for (k = 0; ops[k]; k++)
		if (uses(body, ops[k]))
			sb_printf(b, "%s%s", m++ ? ", " : "", ops[k] == O_ABS ? "|.|" :
			    opinfo[ops[k]].name);
	return m;
}

/*
 * The Cauchy-Riemann test at sample points in [-2, 2]^2: 1 holomorphic
 * there, 0 not (witness in *wz); *n the points used.
 */
static int
cr_test(int *n, cplx *wz, cplx *wx, cplx *wy, double *maxd)
{
	int j, bad = 0;

	*n = 0;
	*maxd = 0;
	hrng = 0x9e3779b97f4a7c15ULL;
	for (j = 0; j < 200 && *n < 64; j++) {
		cplx z = (4 * hrand() - 2) + I * (4 * hrand() - 2), fx, fy, w;
		double d, s;
		if (j < 2)
			z = j == 0 ? 0.5 + 0.25 * I : -0.75 + 1.25 * I;
		if (!fz(z, &w) || !fz_cr(z, &fx, &fy))
			continue;
		s = cabs(fx) + cabs(fy);
		d = cabs(fx - fy) / (s > 1 ? s : 1);
		if (!isfinite(d))
			continue;
		/* near a cut or a pole the difference quotient is meaningless */
		(*n)++;
		if (d > 1e-5) {
			if (!bad++) {
				*wz = z;
				*wx = fx;
				*wy = fy;
			}
		} else if (d > *maxd)
			*maxd = d;
	}
	return *n > 0 && bad * 8 < *n ? (bad == 0 ? 1 : 2) : 0;
}

static void
cmd_holo(struct func *f, int deriv)
{
	struct sbuf b = {0}, nh = {0};
	cplx wz = 0, wx = 0, wy = 0;
	int n, r;
	double maxd;
	nid d = 0;

	fz_open(f, deriv ? "diff" : "holomorphic");
	nonhol_ops(f->body, &nh);
	r = cr_test(&n, &wz, &wx, &wy, &maxd);
	rep_begin(deriv ? "diff" : "holomorphic");
	rep_str("var", vname());
	rep_bool("holomorphic", r == 1 || r == 2);
	if (deriv) {
		struct sbuf t = {0};
		d = tidy(diff_node(f->body, (uint32_t)fp_slot));
		print_node(&t, d, NULL);
		if (r)
			printf("d/d%s %s = %s\n", vname(), fname(f), t.p);
		if (r == 1 || r == 2)
			rep_str("derivative", t.p);
		free(t.p);
	}
	if (r == 1 || r == 2) {
		where_not(f->body, &b);
		printf("  %s: the Cauchy-Riemann equations hold at %d sample "
		    "points of [-2, 2] x [-2, 2] (df/dx = df/(i dy)%s)\n",
		    deriv ? "the complex derivative" : "holomorphic", n,
		    maxd > 0 ? "" : "");
		if (r == 2)
			printf("  (except a few points next to a cut or a pole)\n");
		rep_check("cauchy_riemann", n, maxd, 1, "df/dx = df/(i dy) at "
		    "sample points of [-2, 2] x [-2, 2]");
		rep_bool("entire", !b.n);
		if (b.n)
			rep_str("except", b.p);
		if (b.n)
			printf("  holomorphic except at %s\n", b.p);
		else
			printf("  entire: holomorphic on the whole plane\n");
		if (deriv) {
			/* the symbolic derivative against the difference quotient */
			int j, ok = 0, tot = 0;
			double md = 0;
			struct hp DP;
			hp_build(&DP, &d, 1, 2, f->np);
			hrng = 12345;
			for (j = 0; j < 64; j++) {
				cplx z = (4 * hrand() - 2) + I * (4 * hrand() - 2), fx, fy;
				double o[2], s, dd;
				memcpy(fp_in, fp_in, 0);
				if (!fz_cr(z, &fx, &fy))
					continue;
				fp_in[fp_slot * 2] = creal(z);
				fp_in[fp_slot * 2 + 1] = cimag(z);
				hp_run(&DP, fp_in);
				if (!hp_get(&DP, 0, o))
					continue;
				tot++;
				s = cabs(fx) > 1 ? cabs(fx) : 1;
				dd = cabs(fx - (o[0] + I * o[1])) / s;
				if (dd < 1e-6) {
					ok++;
					if (dd > md)
						md = dd;
				}
			}
			hp_free(&DP);
			printf("  check: equals the difference quotient at %d of %d "
			    "sample points (max relative difference %.0e)\n", ok, tot,
			    md);
			rep_check("difference_quotient", tot, md, ok == tot, NULL);
		}
	} else if (n == 0)
		printf("%s: no sample point where %s is defined\n", fname(f),
		    fname(f));
	else {
		if (deriv)
			printf("d/d%s %s does not exist: %s is not holomorphic\n",
			    vname(), fname(f), fname(f));
		else
			printf("%s is not holomorphic\n", fname(f));
		printf("  Cauchy-Riemann fails: at %s = %s, df/dx = %s but "
		    "df/(i dy) = %s\n", vname(), zs(wz), zs(wx), zs(wy));
		rep_obj("counterexample");
		rep_str("at", zs(wz));
		rep_str("df_dx", zs(wx));
		rep_str("df_idy", zs(wy));
		rep_end();
		if (nh.n)
			rep_str("non_holomorphic_parts", nh.p);
		if (nh.n)
			printf("  (it uses %s, which %s not holomorphic)\n", nh.p,
			    strchr(nh.p, ',') ? "are" : "is");
		{
			cplx a = (wx + wy) / 2, b = (wx - wy) / 2;
			double m = cabs(a) + cabs(b);
			if (cabs(a) < 1e-8 * m)
				a = 0;
			if (cabs(b) < 1e-8 * m)
				b = 0;
			printf("  Wirtinger derivatives there: df/dz = %s, df/dconj(z) "
			    "= %s (holomorphic means df/dconj(z) = 0)\n", zs(snap(a)),
			    zs(snap(b)));
		}
	}
	free(b.p);
	free(nh.p);
	fz_close();
}

/* ---- zeros ---- */

#define MAXZ 64

/* polynomial coefficients from samples on a circle: the degree, or -1 */
static int
poly_of(cplx *c, int maxdeg)
{
	enum { M = 64 };
	cplx v[M], w;
	double R = 1.25, mx = 0;
	int k, j, deg = -1;

	for (k = 0; k < M; k++)
		if (!fz(R * cexp(2 * M_PI * I * k / M), &v[k]))
			return -1;
	for (j = 0; j < M; j++) {
		cplx s = 0;
		for (k = 0; k < M; k++)
			s += v[k] * cexp(-2 * M_PI * I * j * k / M);
		c[j] = s / M / cpow(R, j);
		if (cabs(c[j]) * pow(R, j) > mx)
			mx = cabs(c[j]) * pow(R, j);
	}
	for (j = 0; j < M; j++)
		if (cabs(c[j]) * pow(R, j) > 1e-11 * mx)
			deg = j;
	if (deg < 0 || deg > maxdeg)
		return -1;
	/* round tiny parts, then check at other points */
	for (j = 0; j <= deg; j++) {
		if (fabs(creal(c[j])) < 1e-12 * mx)
			c[j] = I * cimag(c[j]);
		if (fabs(cimag(c[j])) < 1e-12 * mx)
			c[j] = creal(c[j]);
		{
			double re = creal(c[j]), im = cimag(c[j]);
			if (fabs(re - nearbyint(re)) < 1e-9)
				re = nearbyint(re);
			if (fabs(im - nearbyint(im)) < 1e-9)
				im = nearbyint(im);
			c[j] = re + I * im;
		}
	}
	hrng = 777;
	for (k = 0; k < 8; k++) {
		cplx z = (4.5 * hrand() + 0.5) * cexp(2 * M_PI * I * hrand()), p = 0;
		double sc = 0;
		if (!fz(z, &w))
			return -1;
		for (j = deg; j >= 0; j--) {
			p = p * z + c[j];
			sc = sc * cabs(z) + cabs(c[j]);
		}
		if (cabs(p - w) > 1e-9 * (cabs(w) + sc))
			return -1;
	}
	return deg;
}

static cplx
peval(const cplx *c, int deg, cplx z)
{
	cplx p = 0;
	int j;

	for (j = deg; j >= 0; j--)
		p = p * z + c[j];
	return p;
}

/* Durand-Kerner on a monic copy */
static void
dk(const cplx *c0, int deg, cplx *z)
{
	cplx c[MAXZ + 1];
	int j, k, it;

	for (j = 0; j <= deg; j++)
		c[j] = c0[j] / c0[deg];
	for (k = 0; k < deg; k++)
		z[k] = cpow(0.4 + 0.9 * I, k);
	for (it = 0; it < 2000; it++) {
		double mv = 0;
		for (k = 0; k < deg; k++) {
			cplx q = 1, nz_;
			for (j = 0; j < deg; j++)
				if (j != k)
					q *= z[k] - z[j];
			if (q == 0)
				q = 1e-12;
			nz_ = z[k] - peval(c, deg, z[k]) / q;
			if (cabs(nz_ - z[k]) > mv)
				mv = cabs(nz_ - z[k]);
			z[k] = nz_;
		}
		if (mv < 1e-15)
			break;
	}
}

struct zr {
	cplx z;
	int mult;
};

static int
zr_cmp(const void *x, const void *y)
{
	const struct zr *a = x, *b = y;

	if (fabs(creal(a->z) - creal(b->z)) > 1e-9)
		return creal(a->z) < creal(b->z) ? -1 : 1;
	if (fabs(cimag(a->z) - cimag(b->z)) > 1e-9)
		return cimag(a->z) < cimag(b->z) ? -1 : 1;
	return 0;
}

static cplx
snap(cplx z)
{
	double re = creal(z), im = cimag(z), s = cabs(z) > 1 ? cabs(z) : 1;
	char nb[64];

	if (fabs(re) < 1e-10 * s)
		re = 0;
	if (fabs(im) < 1e-10 * s)
		im = 0;
	/* a value near an exact constant becomes it */
	if (re != 0 && !forms_num(re, nb, sizeof nb)) {
		int k;
		for (k = 0; k < 2; k++) {
			double r = nearbyint(re * 1e6) / 1e6;
			if (fabs(r - re) < 1e-9 && forms_num(r, nb, sizeof nb))
				re = r;
		}
	}
	return re + I * im;
}

/* Newton from a grid: zeros of f (or of 1/f) in |Re|, |Im| <= R */
static int
newton_zeros(double R, int inv, struct zr *out, int max)
{
	int n = 0, a, b, it;
	const int G = R > 5 ? 21 : 13;

	for (a = 0; a < G; a++)
		for (b = 0; b < G; b++) {
			cplx z = -R + 2 * R * (a + 0.37) / G + I * (-R + 2 * R *
			    (b + 0.61) / G), w, w1;
			int ok = 0, k;
			for (it = 0; it < 60; it++) {
				double h = 1e-7 * (cabs(z) > 1 ? cabs(z) : 1);
				cplx d;
				if (!fz(z, &w) || !fz(z + h, &w1))
					break;
				if (inv) {
					if (w == 0 || w1 == 0) {
						ok = w == 0;
						break;
					}
					w = 1 / w;
					w1 = 1 / w1;
				}
				d = (w1 - w) / h;
				if (d == 0)
					break;
				z -= w / d;
				if (cabs(w / d) < 1e-14 * (cabs(z) + 1)) {
					ok = 1;
					break;
				}
				if (cabs(z) > 10 * R)
					break;
			}
			if (!ok || fabs(creal(z)) > R || fabs(cimag(z)) > R)
				continue;
			if (!inv) {
				if (!fz(z, &w) || cabs(w) > 1e-8)
					continue;
			} else if (fz(z, &w) && cabs(w) < 1e8)
				continue;
			z = snap(z);
			for (k = 0; k < n; k++)
				if (cabs(out[k].z - z) < 1e-7 * (cabs(z) + 1))
					break;
			if (k == n && n < max) {
				out[n].z = z;
				out[n++].mult = 1;
			}
		}
	qsort(out, (size_t)n, sizeof *out, zr_cmp);
	return n;
}

static void
cmd_zeros(struct func *f)
{
	cplx c[64], z[MAXZ];
	struct zr r[MAXZ];
	int deg, n = 0, k, j;

	fz_open(f, "zeros");
	printf("zeros of %s (%s complex):\n", fname(f), vname());
	deg = poly_of(c, MAXZ);
	if (deg == 0) {
		printf("  none: a nonzero constant\n");
		fz_close();
		return;
	}
	if (deg > 0) {
		dk(c, deg, z);
		for (k = 0; k < deg; k++)
			if (!isfinite(creal(z[k])) || !isfinite(cimag(z[k])))
				deg = -1;
	}
	if (deg > 0) {
		double used[MAXZ] = {0};
		/* group near-equal roots: multiplicity */
		for (k = 0; k < deg; k++) {
			cplx s;
			int m = 0;
			if (used[k])
				continue;
			s = 0;
			for (j = k; j < deg; j++)
				if (!used[j] && cabs(z[j] - z[k]) < 1e-4 * (cabs(z[k]) + 1)) {
					used[j] = 1;
					s += z[j];
					m++;
				}
			s /= m;
			if (m > 1) {
				/* Newton on the (m-1)-th derivative, where the root is
				 * simple */
				cplx dc[MAXZ + 1];
				int dd = deg - (m - 1), q, it;
				for (q = 0; q <= dd; q++) {
					double fac = 1;
					int u;
					for (u = 1; u < m; u++)
						fac *= q + u;
					dc[q] = c[q + m - 1] * fac;
				}
				for (it = 0; it < 20; it++) {
					cplx pv = 0, pd = 0;
					for (q = dd; q >= 0; q--) {
						pd = pd * s + pv;
						pv = pv * s + dc[q];
					}
					if (pd == 0)
						break;
					s -= pv / pd;
				}
			}
			r[n].z = snap(s);
			r[n++].mult = m;
		}
		qsort(r, (size_t)n, sizeof *r, zr_cmp);
		printf("  %d root%s (a polynomial of degree %d):\n", deg,
		    deg == 1 ? "" : "s", deg);
		rep_begin("zeros");
		rep_str("var", vname());
		rep_bool("polynomial", 1);
		rep_int("degree", deg);
		rep_list("zeros");
		for (k = 0; k < n; k++) {
			rep_obj(NULL);
			rep_str("value", zs(r[k].z));
			rep_num("re", creal(r[k].z));
			rep_num("im", cimag(r[k].z));
			rep_int("multiplicity", r[k].mult);
			rep_end();
		}
		rep_end();
		rep_proof("numeric", "Durand-Kerner on the polynomial's "
		    "coefficients (from 64 samples, checked at 8 other points); "
		    "the degree counts the roots");
		for (k = 0; k < n; k++)
			if (r[k].mult > 1)
				printf("    %s = %s  (%s)\n", vname(), zs(r[k].z),
				    r[k].mult == 2 ? "double" : r[k].mult == 3 ? "triple" :
				    "multiple");
			else
				printf("    %s = %s\n", vname(), zs(r[k].z));
		{
			double mx = 0;
			cplx w;
			for (k = 0; k < n; k++)
				if (fz(r[k].z, &w) && cabs(w) > mx)
					mx = cabs(w);
			if (mx > 0)
				printf("  check: |%s| <= %.0e at the roots", fname(f), mx);
			else
				printf("  check: %s = 0 at the roots", fname(f));
			printf("; the polynomial's coefficients come from 64 samples and "
			    "were checked at 8 other points\n");
		}
		printf("  a polynomial of degree %d has %d complex roots counted "
		    "with multiplicity (the fundamental theorem of algebra)\n",
		    deg, deg);
		fz_close();
		return;
	}
	n = newton_zeros(7, 0, r, MAXZ);
	rep_begin("zeros");
	rep_str("var", vname());
	rep_bool("polynomial", 0);
	rep_list("zeros");
	for (k = 0; k < n; k++) {
		rep_obj(NULL);
		rep_str("value", zs(r[k].z));
		rep_num("re", creal(r[k].z));
		rep_num("im", cimag(r[k].z));
		rep_end();
	}
	rep_end();
	rep_str("searched", "|Re|, |Im| <= 7");
	rep_proof("numeric", "Newton's method from a 21x21 grid; not a proof "
	    "that there are no others");
	if (n == 0)
		printf("  none found in |Re %s|, |Im %s| <= 7\n", vname(), vname());
	else
		printf("  %d zero%s in |Re %s|, |Im %s| <= 7:\n", n, n == 1 ? "" :
		    "s", vname(), vname());
	for (k = 0; k < n; k++)
		printf("    %s = %s\n", vname(), zs(r[k].z));
	printf("  numerically: Newton's method from a 21x21 grid, each zero "
	    "checked (|f| < 1e-8); not a proof that there are no others\n");
	fz_close();
}

/* ---- limits, residues, poles, contour integrals ---- */

static int
is_inf_word(const struct toks *t, uint32_t a, uint32_t b, int *sign)
{
	char w[32];

	*sign = 1;
	if (b == a + 2 && t->kind[a] == T_MINUS) {
		*sign = -1;
		a++;
	}
	if (b != a + 1 || t->kind[a] != T_ID)
		return 0;
	tok_text(t, a, w, sizeof w);
	return !strcmp(w, "inf") || !strcmp(w, "oo") || !strcmp(w, "infinity") ||
	    !strcmp(w, "\xe2\x88\x9e");
}

static cplx
tok_point(const struct toks *t, uint32_t a, uint32_t b, int *inf)
{
	double v[HM];
	int sg;

	*inf = 0;
	if (is_inf_word(t, a, b, &sg)) {
		*inf = 1;
		return 0;
	}
	if (!hy_const(t, a, b, 1, v, NULL))
		fail("the point is undefined");
	return v[0] + I * v[1];
}

static void
cmd_limit(struct func *f, cplx a, int inf)
{
	static const char *const dn[] = {"0", "pi/4", "pi/2", "3pi/4", "pi",
		"-3pi/4", "-pi/2", "-pi/4"};
	cplx L[8];
	int ok[8], k, j, ninf = 0, nok = 0, agree = 1;
	double big[8];

	fz_open(f, "limit");
	for (k = 0; k < 8; k++) {
		cplx u = cexp(I * M_PI * k / 4), v1 = 0, v2 = 0, w;
		int got = 0;
		big[k] = 0;
		ok[k] = 0;
		for (j = 0; j < 14; j++) {
			double r = 0.1 * pow(0.25, j);
			cplx z = inf ? u / r : a + r * u;
			if (!fz(z, &w))
				continue;
			v1 = v2;
			v2 = w;
			got++;
			if (cabs(w) > big[k])
				big[k] = cabs(w);
		}
		if (got < 3)
			continue;
		if (cabs(v2) > 1e6 && cabs(v2) > 2 * cabs(v1)) {
			ninf++;
			continue;
		}
		if (cabs(v2 - v1) <= 1e-6 * (cabs(v2) + 1)) {
			L[k] = v2 + (v2 - v1) / 3;
			ok[k] = 1;
			nok++;
		}
	}
	printf("limit of %s as %s -> %s:\n", fname(f), vname(), inf ? "infinity" :
	    zs(a));
	if (ninf == 8) {
		printf("  infinity: |%s| grows without bound from every direction "
		    "(a pole%s)\n", fname(f), inf ? " at infinity" : "");
		fz_close();
		return;
	}
	for (k = 1; k < 8; k++)
		if (!ok[k] || !ok[0] || cabs(L[k] - L[0]) > 1e-5 * (cabs(L[0]) + 1))
			agree = 0;
	if (nok == 8 && agree) {
		cplx v = snap(L[0]), w;
		printf("  = %s\n", zs(v));
		printf("  checked numerically along 8 directions (arg(%s%s) = 0, "
		    "pi/4, ..., -pi/4), %s -> %s by factors of 4: all agree\n",
		    vname(), inf ? "" : " - a", vname(), inf ? "infinity" : "a");
		if (!inf && fz(a, &w) && cabs(w - v) < 1e-9 * (cabs(v) + 1))
			printf("  %s is continuous there: f(%s) = %s\n", fname(f),
			    zs(a), zs(w));
		else if (!inf && !fz(a, &w))
			printf("  (%s is undefined at %s itself: a removable "
			    "singularity)\n", fname(f), zs(a));
	} else {
		printf("  does not exist: the values along different directions "
		    "differ\n");
		for (k = 0; k < 8; k++) {
			if (ok[k])
				printf("    along arg(%s%s) = %s: %s\n", vname(),
				    inf ? "" : " - a", dn[k], zs(snap(L[k])));
			else
				printf("    along arg(%s%s) = %s: %s\n", vname(),
				    inf ? "" : " - a", dn[k], big[k] > 1e8 ? "infinity" :
				    "no limit (oscillates)");
		}
	}
	fz_close();
}

/* (1/2 pi i) times the integral on |z - a| = r, n points */
static int
circle_int(cplx a, double r, int n, cplx *res)
{
	cplx s = 0, w;
	int k;

	for (k = 0; k < n; k++) {
		cplx u = cexp(2 * M_PI * I * (k + 0.5) / n);
		if (!fz(a + r * u, &w))
			return 0;
		s += w * r * u;
	}
	*res = s / n;	/* (1/2pi i) sum f dz, dz = i r u dtheta */
	return isfinite(creal(*res)) && isfinite(cimag(*res));
}

static double
circle_max(cplx a, double r)
{
	double m = 0;
	cplx w;
	int k;

	for (k = 0; k < 32; k++)
		if (fz(a + r * cexp(2 * M_PI * I * (k + 0.3) / 32), &w) &&
		    cabs(w) > m)
			m = cabs(w);
	return m;
}

/* kind: 0 regular or removable, m > 0 a pole of order m, -1 essential */
static int
sing_order(cplx a)
{
	double m1 = circle_max(a, 1e-2), m2 = circle_max(a, 1e-3),
	    m3 = circle_max(a, 1e-4), e1, e2;

	if (m1 == 0 || m2 == 0 || m3 == 0)
		return 0;
	e1 = log10(m2 / m1);
	e2 = log10(m3 / m2);
	if (e1 < 0.2 && e2 < 0.2)
		return 0;
	if (fabs(e1 - e2) > 0.3 || e2 > 30)
		return -1;
	return (int)nearbyint(e2) > 0 ? (int)nearbyint(e2) : 0;
}

static int
residue_at(cplx a, cplx *res)
{
	cplx r1, r2;

	if (!circle_int(a, 1e-2, 512, &r1) || !circle_int(a, 5e-3, 512, &r2))
		return 0;
	if (cabs(r1 - r2) > 1e-7 * (cabs(r1) + 1))
		return -1;
	*res = snap(r2);
	return 1;
}

static void
cmd_residue(struct func *f, cplx a)
{
	cplx r;
	int o, ok;

	fz_open(f, "residue");
	o = sing_order(a);
	ok = residue_at(a, &r);
	printf("residue of %s at %s = %s:\n", fname(f), vname(), zs(a));
	rep_begin("residue");
	rep_str("var", vname());
	rep_str("at", zs(a));
	rep_bool("found", ok > 0);
	if (ok > 0) {
		rep_str("residue", zs(r));
		rep_num("re", creal(r));
		rep_num("im", cimag(r));
	}
	rep_str("singularity", o > 0 ? "pole" : o == 0 ? "regular" :
	    "essential");
	if (o > 0)
		rep_int("order", o);
	if (ok > 0)
		rep_check("circle_integrals", 1024, -1, 1, "radius 0.01 and "
		    "0.005 agree");
	if (ok <= 0) {
		printf("  the integral around small circles does not settle (an "
		    "essential singularity or a branch point)\n");
		fz_close();
		return;
	}
	printf("  = %s\n", zs(r));
	if (o > 0)
		printf("  a pole of order %d (|f| grows like |%s - a|^-%d)\n", o,
		    vname(), o);
	else if (o == 0)
		printf("  %s is regular there (or the singularity is removable): "
		    "the residue is 0\n", fname(f));
	else
		printf("  an essential singularity (|f| grows faster than any "
		    "power)\n");
	printf("  computed as (1/(2 pi i)) times the integral of %s on |%s - a| "
	    "= 0.01 and 0.005 (512 points each): they agree\n", fname(f),
	    vname());
	fz_close();
}

static void
cmd_poles(struct func *f)
{
	struct zr p[MAXZ];
	int n, k;

	fz_open(f, "poles");
	n = newton_zeros(7, 1, p, MAXZ);
	printf("poles of %s:\n", fname(f));
	rep_begin("poles");
	rep_str("var", vname());
	rep_list("poles");
	if (n == 0)
		printf("  none in |Re %s|, |Im %s| <= 7\n", vname(), vname());
	for (k = 0; k < n; k++) {
		cplx r;
		int o = sing_order(p[k].z), ok = residue_at(p[k].z, &r);
		printf("    %s = %s", vname(), zs(p[k].z));
		rep_obj(NULL);
		rep_str("at", zs(p[k].z));
		if (o > 0)
			rep_int("order", o);
		else if (o < 0)
			rep_str("singularity", "essential");
		if (ok > 0)
			rep_str("residue", zs(r));
		rep_end();
		if (o > 0)
			printf("  order %d", o);
		else if (o < 0)
			printf("  essential");
		if (ok > 0)
			printf(", residue %s", zs(r));
		printf("\n");
	}
	rep_end();
	rep_proof("numeric", "zeros of 1/f by Newton's method from a grid");
	printf("  numerically: zeros of 1/f by Newton's method from a grid in "
	    "|Re %s|, |Im %s| <= 7; orders from the growth of |f| on circles "
	    "of radius 1e-2..1e-4, residues by circle integrals\n", vname(),
	    vname());
	fz_close();
}

static void
cmd_contour(struct func *f, cplx c, double r, int cw)
{
	cplx v1, v2, sum = 0, val;
	struct zr p[MAXZ];
	int n, k, np = 0;

	fz_open(f, "integrate");
	if (!circle_int(c, r, 1024, &v1) || !circle_int(c, r, 2048, &v2)) {
		fz_close();
		fail("%s is undefined on the contour |%s - %s| = %g", fname(f),
		    vname(), zs(c), r);
	}
	val = snap(2 * M_PI * I * v2 * (cw ? -1 : 1));
	printf("integral of %s around |%s%s%s| = %g (%s):\n", fname(f), vname(),
	    c == 0 ? "" : " - ", c == 0 ? "" : zs(c), r, cw ? "clockwise" :
	    "counterclockwise");
	printf("  = %s\n", zs(val));
	printf("  numerically: the trapezoid rule on the circle with 1024 and "
	    "2048 points (difference %.0e)\n", cabs(v1 - v2) * 2 * M_PI);
	/* the residue theorem as a check */
	n = newton_zeros(fabs(creal(c)) + fabs(cimag(c)) + r + 0.5, 1, p, MAXZ);
	for (k = 0; k < n; k++) {
		cplx rr;
		if (cabs(p[k].z - c) >= r)
			continue;
		if (residue_at(p[k].z, &rr) > 0) {
			sum += rr;
			np++;
		}
	}
	if (np) {
		cplx rt = 2 * M_PI * I * sum * (cw ? -1 : 1);
		printf("  residue theorem: 2 pi i times the sum of the residues at "
		    "the %d pole%s inside = %s: %s\n", np, np == 1 ? "" : "s",
		    zs(snap(rt)), cabs(rt - val) < 1e-6 * (cabs(val) + 1) ? "agrees" :
		    "DISAGREES");
	} else if (cabs(val) < 1e-9)
		printf("  no pole inside: 0, as Cauchy's theorem says for a "
		    "function holomorphic on the disc\n");
	fz_close();
}

/* the integral along the segment from a to b (Simpson) */
static void
cmd_segment(struct func *f, cplx a, cplx b)
{
	int n = 4096, k;
	cplx s = 0, s2 = 0, w, d = b - a;

	fz_open(f, "integrate");
	for (k = 0; k <= n; k++) {
		int wt = k == 0 || k == n ? 1 : (k & 1 ? 4 : 2);
		if (!fz(a + d * k / n, &w)) {
			fz_close();
			fail("%s is undefined on the segment, at %s", fname(f),
			    zs(a + d * k / n));
		}
		s += wt * w;
		if (!(k & 1))
			s2 += (k == 0 || k == n ? 1 : ((k / 2) & 1 ? 4 : 2)) * w;
	}
	s *= d / (3.0 * n);
	s2 *= d / (3.0 * n / 2);
	printf("integral of %s along the segment from %s to %s:\n", fname(f),
	    zs(a), zs(b));
	printf("  = %s\n", zs(snap(s)));
	printf("  numerically: Simpson's rule with 4096 and 2048 steps "
	    "(difference %.0e); the straight path; a holomorphic integrand on "
	    "a simply connected region containing it gives the same value on "
	    "every path\n", cabs(s - s2));
	fz_close();
}

/* ---- z = x + iy: the real picture ---- */

/* f's body with each complex parameter written x + i*y, as text */
static char *
xy_text(const struct func *f, struct sbuf *say)
{
	nid map[MAXP], iu = mk_var(UB + 1, sym_cstr("i"), TY_REAL), e;
	int j, m = 0;
	struct sbuf b = {0};

	for (j = 0; j < f->np; j++) {
		const char *n = sym_str(f->p[j].name);
		char rx[32], ry[32];
		int k, clash = 0;
		if (f->hk[j] != 1) {
			map[j] = mk_var((uint32_t)j, f->p[j].name, TY_REAL);
			continue;
		}
		if (!strcmp(n, "z"))
			strcpy(rx, "x"), strcpy(ry, "y");
		else if (!strcmp(n, "w"))
			strcpy(rx, "u"), strcpy(ry, "v");
		else if ((n[0] == 'z' || n[0] == 'w') && isdigit((unsigned char)n[1]))
			snprintf(rx, sizeof rx, "%c%s", n[0] == 'z' ? 'x' : 'u', n + 1),
			    snprintf(ry, sizeof ry, "%c%s", n[0] == 'z' ? 'y' : 'v', n + 1);
		else
			snprintf(rx, sizeof rx, "%s_re", n),
			    snprintf(ry, sizeof ry, "%s_im", n);
		for (k = 0; k < f->np; k++)
			if (!strcmp(sym_str(f->p[k].name), rx) ||
			    !strcmp(sym_str(f->p[k].name), ry))
				clash = 1;
		if (clash)
			snprintf(rx, sizeof rx, "%s_re", n),
			    snprintf(ry, sizeof ry, "%s_im", n);
		map[j] = mk_bin(O_ADD, mk_var(40 + 2 * j, sym_cstr(rx), TY_REAL),
		    mk_bin(O_MUL, iu, mk_var(41 + 2 * j, sym_cstr(ry), TY_REAL)));
		sb_printf(say, "%s%s = %s + i%s", m++ ? ", " : "", n, rx, ry);
	}
	e = subst(f->body, map, f->np);
	print_node(&b, e, NULL);
	return b.p;
}

static void
cmd_forms(const char *cmd, const struct func *f)
{
	struct sbuf say = {0}, st = {0};
	char *text;

	if (f->halg != 1)
		fail("%s works on complex expressions; this is %s-valued", cmd,
		    alg_name[f->halg]);
	text = xy_text(f, &say);
	if (say.n)
		note("%s (real and imaginary parts real)", say.p);
	notes_flush();
	sb_printf(&st, "%s %s\n", cmd, text);
	run_source(st.p, "<hyper>", 0);
	free(st.p);
	free(say.p);
	free(text);
}

/* ---- domain colouring ---- */

#define DCH 24		/* hues */
#define DCL 10		/* brightness levels */

static void
hsv(double h, double s, double v, unsigned char *c)
{
	double r, g, b, f, p, q, t;
	int i;

	h = fmod(h, 1.0);
	if (h < 0)
		h += 1;
	i = (int)(h * 6);
	f = h * 6 - i;
	p = v * (1 - s);
	q = v * (1 - f * s);
	t = v * (1 - (1 - f) * s);
	switch (i % 6) {
	case 0: r = v, g = t, b = p; break;
	case 1: r = q, g = v, b = p; break;
	case 2: r = p, g = v, b = t; break;
	case 3: r = p, g = q, b = v; break;
	case 4: r = t, g = p, b = v; break;
	default: r = v, g = p, b = q; break;
	}
	c[0] = (unsigned char)(255 * r);
	c[1] = (unsigned char)(255 * g);
	c[2] = (unsigned char)(255 * b);
}

static void
cmd_dcolour(struct func *f, double x0, double x1, double y0, double y1,
    const char *name, int nofile)
{
	enum { W = 400, H = 400 };
	unsigned char pal[256][3], *px;
	struct gif g;
	struct zr zz[MAXZ], pp[MAXZ];
	int i, j, nz_, np_, k, undef = 0;
	double R;

	fz_open(f, "plot");
	for (i = 0; i < DCH; i++)
		for (j = 0; j < DCL; j++)
			hsv((double)i / DCH, 0.85, 0.45 + 0.55 * j / (DCL - 1),
			    pal[i * DCL + j]);
	for (i = DCH * DCL; i < 256; i++)
		pal[i][0] = pal[i][1] = pal[i][2] = 0;
	pal[241][0] = pal[241][1] = pal[241][2] = 128;
	px = xmalloc((size_t)W * H);
	for (j = 0; j < H; j++)
		for (i = 0; i < W; i++) {
			cplx z = x0 + (x1 - x0) * (i + 0.5) / W +
			    I * (y1 - (y1 - y0) * (j + 0.5) / H), w;
			int c;
			if (!fz(z, &w) || !isfinite(cabs(w))) {
				px[j * W + i] = 240;
				undef++;
				continue;
			}
			{
				double a = carg(w) / (2 * M_PI), m = cabs(w), l;
				int hi = (int)floor(a * DCH + 0.5);
				hi = ((hi % DCH) + DCH) % DCH;
				l = m > 0 ? log2(m) : -50;
				l = l - floor(l);
				c = hi * DCL + (int)(l * DCL);
				if (c >= DCH * DCL)
					c = DCH * DCL - 1;
			}
			px[j * W + i] = (unsigned char)c;
		}
	/* the axes */
	if (x0 < 0 && x1 > 0) {
		i = (int)(-x0 / (x1 - x0) * W);
		for (j = 0; j < H; j += 2)
			px[j * W + i] = 241;
	}
	if (y0 < 0 && y1 > 0) {
		j = (int)(y1 / (y1 - y0) * H);
		for (i = 0; i < W; i += 2)
			px[j * W + i] = 241;
	}
	printf("domain colouring of %s, %s in [%g, %g] x [%g, %g]i\n", fname(f),
	    vname(), x0, x1, y0, y1);
	printf("  hue = arg f (red: positive real, yellow-green: arg 2pi/3 "
	    "side, cyan: negative real, blue-magenta: arg < 0); brightness "
	    "rises from dark to light each time |f| doubles\n");
	R = fmax(fmax(fabs(x0), fabs(x1)), fmax(fabs(y0), fabs(y1)));
	nz_ = newton_zeros(R, 0, zz, MAXZ);
	np_ = newton_zeros(R, 1, pp, MAXZ);
	if (nz_) {
		printf("  zeros (every hue meets, counterclockwise red, yellow, "
		    "green, blue):");
		for (k = 0; k < nz_; k++)
			if (creal(zz[k].z) >= x0 && creal(zz[k].z) <= x1 &&
			    cimag(zz[k].z) >= y0 && cimag(zz[k].z) <= y1)
				printf(" %s", zs(zz[k].z));
		printf("\n");
	}
	if (np_) {
		printf("  poles (the hues in the opposite order):");
		for (k = 0; k < np_; k++)
			if (creal(pp[k].z) >= x0 && creal(pp[k].z) <= x1 &&
			    cimag(pp[k].z) >= y0 && cimag(pp[k].z) <= y1)
				printf(" %s", zs(pp[k].z));
		printf("\n");
	}
	if (undef)
		printf("  undefined at %d of %d pixels (black)\n", undef, W * H);
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, 256);
	gif_frame(&g, px, 0);
	gif_end(&g);
	printf("  image: %dx%d, %zu bytes\n", W, H, g.out.n);
	gif_deliver(&g, name, nofile);
	free(g.out.p);
	free(g.prev);
	free(px);
	fz_close();
}

/* ---- the algebras themselves ---- */

static int
alg_word(const struct toks *t, uint32_t a, uint32_t b, uint32_t *next)
{
	char w[32];

	if (a >= b || t->kind[a] != T_ID)
		return -1;
	tok_text(t, a, w, sizeof w);
	*next = a + 1;
	if (!strcmp(w, "the") && a + 1 < b)
		return alg_word(t, a + 1, b, next);
	if (!strcmp(w, "complex") || !strcmp(w, "C"))
		return 1;
	if (!strncmp(w, "quaternion", 10) || !strcmp(w, "quat") ||
	    !strcmp(w, "quats") || !strcmp(w, "H"))
		return 2;
	if (!strncmp(w, "octonion", 8) || !strcmp(w, "oct") || !strcmp(w, "O"))
		return 3;
	if (!strncmp(w, "sedenion", 8) || !strcmp(w, "sed") || !strcmp(w, "S"))
		return 4;
	if (!strcmp(w, "pathion") || !strcmp(w, "pathions") ||
	    !strcmp(w, "trigintaduonion") || !strcmp(w, "trigintaduonions"))
		return 5;
	if (!strncmp(w, "cd", 2)) {
		int n = 0;
		if (w[2])
			n = atoi(w + 2);
		else if (a + 1 < b && t->kind[a + 1] == T_NUM) {
			char nb[16];
			tok_text(t, a + 1, nb, sizeof nb);
			n = atoi(nb);
			*next = a + 2;
		} else if (a + 3 < b && t->kind[a + 1] == T_LP &&
		    t->kind[a + 2] == T_NUM) {
			char nb[16];
			tok_text(t, a + 2, nb, sizeof nb);
			n = atoi(nb);
			*next = a + 4;
		}
		if (n < 2 || n > 32 || (n & (n - 1)))
			fail("cd(N): N is 2, 4, 8, 16 or 32");
		return ilog2u((uint64_t)n);
	}
	if (!strcmp(w, "real") || !strcmp(w, "reals"))
		fail("the reals are a field: commutative, associative, ordered");
	return -1;
}

static void
unit_vec(int u, double *x, int d)
{
	memset(x, 0, (size_t)d * sizeof *x);
	x[u] = 1;
}

/* e_a e_b = s e_c */
static void
unit_prod(int a, int b, int d, int *c, int *s)
{
	double x[HM], y[HM], r[HM];
	int k;

	unit_vec(a, x, d);
	unit_vec(b, y, d);
	h_mul(x, y, r, d);
	for (k = 0; k < d; k++)
		if (r[k] != 0) {
			*c = k;
			*s = r[k] > 0 ? 1 : -1;
			return;
		}
	*c = 0;
	*s = 0;
}

static void
cmd_cayley(int lg)
{
	int d = 1 << lg, a, b, c, s, w = lg >= 3 ? (lg >= 4 ? 5 : 4) : 3;
	char cell[16];

	printf("multiplication table of the %s (row times column)%s:\n",
	    alg_pl[lg], lg >= 2 ? ", not symmetric: the product does not "
	    "commute" : "");
	printf("  %*s |", w, "");
	for (b = 0; b < d; b++)
		printf(" %*s", w, unit_str(b, lg));
	printf("\n  %.*s-+", w, "------");
	for (b = 0; b < d; b++)
		printf("%.*s", w + 1, "-------");
	printf("\n");
	for (a = 0; a < d; a++) {
		printf("  %*s |", w, unit_str(a, lg));
		for (b = 0; b < d; b++) {
			unit_prod(a, b, d, &c, &s);
			snprintf(cell, sizeof cell, "%s%s", s < 0 ? "-" : "",
			    unit_str(c, lg));
			printf(" %*s", w, cell);
		}
		printf("\n");
	}
	if (lg == 2)
		printf("  i^2 = j^2 = k^2 = ijk = -1 (Hamilton); ij = k, jk = i, "
		    "ki = j, and reversing a product changes its sign\n");
	if (lg == 3) {
		printf("  the Fano plane: the 7 lines (a, b, c) with e_a e_b = e_c, "
		    "e_b e_c = e_a, e_c e_a = e_b:\n   ");
		for (a = 1; a < 8; a++)
			for (b = a + 1; b < 8; b++) {
				unit_prod(a, b, d, &c, &s);
				if (c > b) {
					/* oriented, smallest first: e_p e_q = e_r */
					int p = s > 0 ? a : b, q = s > 0 ? b : a, r = c;
					if (q < p && q < r)
						printf(" (%d, %d, %d)", q, r, p);
					else if (r < p && r < q)
						printf(" (%d, %d, %d)", r, p, q);
					else
						printf(" (%d, %d, %d)", p, q, r);
				}
			}
		printf("\n  every pair of units lies on one line; e_a^2 = -1 and "
		    "e_a e_b = -e_b e_a for a != b\n");
	}
	if (lg >= 4)
		printf("  (Cayley-Dickson doubling of the %s: (p, q)(s, t) = (ps - "
		    "conj(t) q, tp + q conj(s)))\n", alg_name[lg - 1]);
}

static void
h_rand(double *x, int d)
{
	int k;

	for (k = 0; k < d; k++)
		x[k] = 2 * hrand() - 1;
}

/* a zero divisor pair (e_a + s e_b)(e_c + t e_d) = 0; 1 when found */
static int
zero_div(int d, int *A, int *B, int *s1, int *C, int *D, int *s2)
{
	int a, b, c, e, s, t, k;
	double x[HM], y[HM], r[HM];

	for (a = 1; a < d; a++)
		for (b = a + 1; b < d; b++)
			for (c = 1; c < d; c++)
				for (e = c + 1; e < d; e++)
					for (s = -1; s <= 1; s += 2)
						for (t = -1; t <= 1; t += 2) {
							memset(x, 0, sizeof x);
							memset(y, 0, sizeof y);
							x[a] = 1;
							x[b] = s;
							y[c] = 1;
							y[e] = t;
							h_mul(x, y, r, d);
							for (k = 0; k < d; k++)
								if (r[k] != 0)
									break;
							if (k == d) {
								*A = a, *B = b, *s1 = s;
								*C = c, *D = e, *s2 = t;
								return 1;
							}
						}
	return 0;
}

static const char *
pair_str(int a, int b, int s, int lg)
{
	static char buf[4][32];
	static int k;
	char *p = buf[k++ & 3];

	snprintf(p, 32, "%s %s %s", unit_str(a, lg), s > 0 ? "+" : "-",
	    unit_str(b, lg));
	return p;
}

/* which: NULL all, else one property name */
static void
cmd_algebra(int lg, const char *which)
{
	int d = 1 << lg, j, ok, a, b, c, s, c2, s2;
	double x[HM], y[HM], z[HM], u[HM], v[HM], t[HM];
	int A, B, S1, C, D, S2;
	int all = which == NULL;
#define WANT(w) (all || !strcmp(which, w))

	printf("the %s: dimension %d over the reals, basis %s", alg_pl[lg], d,
	    unit_str(0, lg));
	for (j = 1; j < d && j < 8; j++)
		printf(", %s", unit_str(j, lg));
	printf("%s\n", d > 8 ? ", ..." : "");
	hrng = 4242;
	if (WANT("commutative")) {
		if (lg <= 1)
			printf("  commutative: yes (checked on 200 random pairs)\n");
		else {
			unit_prod(1, 2, d, &c, &s);
			printf("  commutative: no: %s %s = %s%s but %s %s = %s%s\n",
			    unit_str(1, lg), unit_str(2, lg), s < 0 ? "-" : "",
			    unit_str(c, lg), unit_str(2, lg), unit_str(1, lg),
			    s > 0 ? "-" : "", unit_str(c, lg));
		}
	}
	if (WANT("associative")) {
		for (ok = 1, j = 0; j < 200 && ok; j++) {
			h_rand(x, d), h_rand(y, d), h_rand(z, d);
			h_mul(x, y, t, d), h_mul(t, z, u, d);
			h_mul(y, z, t, d), h_mul(x, t, v, d);
			ok = h_close(u, v, d);
		}
		if (ok)
			printf("  associative: yes ((ab)c = a(bc) at 200 random "
			    "triples)%s\n", lg == 2 ? "; by Frobenius, R, C and H are "
			    "the only associative real division algebras" : "");
		else {
			for (a = 1; a < d; a++)
				for (b = 1; b < d; b++)
					for (c = 1; c < d; c++) {
						int p, q, r1, r2;
						unit_prod(a, b, d, &p, &s);
						unit_prod(p, c, d, &r1, &s2);
						s2 *= s;
						unit_prod(b, c, d, &q, &s);
						unit_prod(a, q, d, &r2, &c2);
						c2 *= s;
						if (r1 != r2 || s2 != c2) {
							printf("  associative: no: (%s %s) %s = %s%s but "
							    "%s (%s %s) = %s%s\n", unit_str(a, lg),
							    unit_str(b, lg), unit_str(c, lg),
							    s2 < 0 ? "-" : "", unit_str(r1, lg),
							    unit_str(a, lg), unit_str(b, lg),
							    unit_str(c, lg), c2 < 0 ? "-" : "",
							    unit_str(r2, lg));
							goto assoc_done;
						}
					}
		}
assoc_done:;
	}
	if (WANT("alternative")) {
		double worst = 0;
		for (ok = 1, j = 0; j < 200; j++) {
			h_rand(x, d), h_rand(y, d);
			h_mul(x, x, t, d), h_mul(t, y, u, d);
			h_mul(x, y, t, d), h_mul(x, t, v, d);
			if (!h_close(u, v, d)) {
				ok = 0;
				worst = h_dist(u, v, d);
			}
		}
		if (ok)
			printf("  alternative: yes ((aa)b = a(ab), (ba)a = b(aa) at "
			    "200 random pairs)%s\n", lg == 3 ? "; by Artin's theorem "
			    "any two elements generate an associative subalgebra" : "");
		else
			printf("  alternative: no: (aa)b != a(ab) for random a, b (by "
			    "%.3g)\n", worst);
	}
	if (WANT("flexible") || WANT("power-associative") || all) {
		for (ok = 1, j = 0; j < 200 && ok; j++) {
			h_rand(x, d), h_rand(y, d);
			h_mul(x, y, t, d), h_mul(t, x, u, d);
			h_mul(y, x, t, d), h_mul(x, t, v, d);
			ok = h_close(u, v, d);
		}
		printf("  flexible and power-associative: %s ((ab)a = a(ba); "
		    "powers a^n are well defined), as in every Cayley-Dickson "
		    "algebra\n", ok ? "yes" : "NO (unexpected)");
	}
	if (WANT("normed") || WANT("composition") || WANT("division") || all) {
		double worst = 0;
		for (ok = 1, j = 0; j < 200; j++) {
			h_rand(x, d), h_rand(y, d);
			h_mul(x, y, t, d);
			if (fabs(sqrt(h_n2(t, d)) - sqrt(h_n2(x, d) * h_n2(y, d))) >
			    1e-9) {
				ok = 0;
				worst = fabs(sqrt(h_n2(t, d)) - sqrt(h_n2(x, d) *
				    h_n2(y, d)));
			}
		}
		if (WANT("normed") || WANT("composition") || all) {
			if (ok)
				printf("  normed (a composition algebra): yes, |ab| = |a||b| "
				    "at 200 random pairs%s\n", lg >= 2 ? (lg == 2 ?
				    " (Euler's four-square identity)" :
				    " (Degen's eight-square identity)") : "");
			else
				printf("  normed: no: |ab| != |a||b| (off by %.3g at a random "
				    "pair)\n", worst);
		}
		if (WANT("division") || all) {
			if (lg <= 3)
				printf("  a division algebra: yes, ab = 0 only when a = 0 "
				    "or b = 0 (it is normed); by Hurwitz, R, C, H and O "
				    "are the only normed division algebras\n");
			else if (zero_div(d, &A, &B, &S1, &C, &D, &S2))
				printf("  a division algebra: no, it has zero divisors: "
				    "(%s)(%s) = 0\n", pair_str(A, B, S1, lg),
				    pair_str(C, D, S2, lg));
		}
	}
	if (WANT("inverse") || all) {
		h_rand(x, d);
		h_inv(x, t, d);
		h_mul(x, t, u, d);
		printf("  inverses: every a != 0 has a^-1 = conj(a)/|a|^2 (a a^-1 "
		    "= 1%s)%s\n", h_isreal(u, d) && fabs(u[0] - 1) < 1e-12 ?
		    ", checked" : "", lg >= 4 ? "; but a^-1(ab) = b can fail, so "
		    "ax = b need not be solvable" : lg == 3 ? "; a^-1(ab) = b "
		    "holds (alternativity)" : "");
	}
	if (WANT("conjugation") || all) {
		h_rand(x, d), h_rand(y, d);
		h_mul(x, y, t, d), h_conj(t, u, d);
		h_conj(x, t, d), h_conj(y, z, d), h_mul(z, t, v, d);
		printf("  conjugation: conj(ab) = conj(b) conj(a)%s; a conj(a) = "
		    "|a|^2\n", h_close(u, v, d) ? " (checked)" : " FAILS");
	}
#undef WANT
}

static void
cmd_zerodiv(int lg)
{
	int A, B, S1, C, D, S2, d = 1 << lg;
	double x[HM] = {0}, y[HM] = {0}, r[HM];

	if (lg <= 3) {
		printf("the %s have no zero divisors: |ab| = |a||b|, so ab = 0 "
		    "only when a = 0 or b = 0\n", alg_pl[lg]);
		return;
	}
	if (!zero_div(d, &A, &B, &S1, &C, &D, &S2)) {
		printf("no zero divisor of the form (e_a +- e_b)(e_c +- e_d)\n");
		return;
	}
	x[A] = 1, x[B] = S1, y[C] = 1, y[D] = S2;
	h_mul(x, y, r, d);
	printf("zero divisors in the %s:\n", alg_pl[lg]);
	printf("  (%s)(%s) = %s, although |%s| = |%s| = sqrt(2)\n",
	    pair_str(A, B, S1, lg), pair_str(C, D, S2, lg), hv_str(r, d, lg),
	    pair_str(A, B, S1, lg), pair_str(C, D, S2, lg));
	printf("  so |ab| = |a||b| fails and the %s are not a division "
	    "algebra (from dimension 16 on, every Cayley-Dickson algebra has "
	    "them)\n", alg_pl[lg]);
}

/* comma-separated constants in [a, b) */
static int
const_list(const struct toks *t, uint32_t a, uint32_t b, double v[][HM],
    int max, int *lg, int lgmin)
{
	int n = 0, depth = 0, l, lgall = 1;
	uint32_t s = a, i;

	if (a < b && t->kind[a] == T_LP && tok_close(t, a, b) == b) {
		a++;
		b--;
		s = a;
	}
	*lg = 0;
	{
		/* the algebra of the whole list: "i, j" are quaternions */
		struct hscan S;
		memset(&S, 0, sizeof S);
		S.keepk = lgmin >= 2;
		if (hy_scan(t, a, b, &S)) {
			hy_decide(&S, t, a, b);
			lgall = S.lg > lgmin ? S.lg : lgmin;
		}
	}
	for (i = a; i <= b; i++) {
		if (i < b && (t->kind[i] == T_LP || t->kind[i] == T_LB))
			depth++;
		if (i < b && (t->kind[i] == T_RP || t->kind[i] == T_RB))
			depth--;
		if (i == b || (depth == 0 && (t->kind[i] == T_COMMA ||
		    (t->kind[i] == T_ID && tok_is(t, i, "and"))))) {
			if (n == max)
				fail("too many values");
			if (!hy_const(t, s, i, lgall > 1 ? lgall : 1, v[n], &l))
				fail("value %d is undefined", n + 1);
			if (l > *lg)
				*lg = l;
			n++;
			s = i + 1;
		}
	}
	return n;
}

static void
cmd_assoc(const struct toks *t, uint32_t a, uint32_t b, int comm)
{
	double v[3][HM], p[HM], q[HM], r1[HM], r2[HM], r[HM];
	int n, lg, d, k;

	n = const_list(t, a, b, v, 3, &lg, 2);
	if (n != (comm ? 2 : 3))
		fail("%s takes %d values", comm ? "commutator" : "associator",
		    comm ? 2 : 3);
	if (lg < 2)
		lg = 2;
	d = 1 << lg;
	if (comm) {
		h_mul(v[0], v[1], r1, d);
		h_mul(v[1], v[0], r2, d);
		for (k = 0; k < d; k++)
			r[k] = r1[k] - r2[k];
		printf("[a, b] = ab - ba = %s - (%s) = %s\n", hv_str(r1, d, lg),
		    hv_str(r2, d, lg), hv_str(r, d, lg));
		printf("  %s\n", h_n2(r, d) == 0 ? "they commute" :
		    "they do not commute");
		return;
	}
	h_mul(v[0], v[1], p, d);
	h_mul(p, v[2], r1, d);
	h_mul(v[1], v[2], q, d);
	h_mul(v[0], q, r2, d);
	for (k = 0; k < d; k++)
		r[k] = r1[k] - r2[k];
	printf("[a, b, c] = (ab)c - a(bc) = %s - (%s) = %s\n", hv_str(r1, d, lg),
	    hv_str(r2, d, lg), hv_str(r, d, lg));
	printf("  %s (in the %s)\n", h_n2(r, d) < 1e-24 ? "the product of "
	    "these three is associative" : "not associative: the grouping "
	    "changes the product", alg_pl[lg]);
}

/* ---- rotations ---- */

static void
q_matrix(const double *q, double m[3][3])
{
	double w = q[0], x = q[1], y = q[2], z = q[3];

	m[0][0] = 1 - 2 * (y * y + z * z);
	m[0][1] = 2 * (x * y - w * z);
	m[0][2] = 2 * (x * z + w * y);
	m[1][0] = 2 * (x * y + w * z);
	m[1][1] = 1 - 2 * (x * x + z * z);
	m[1][2] = 2 * (y * z - w * x);
	m[2][0] = 2 * (x * z - w * y);
	m[2][1] = 2 * (y * z + w * x);
	m[2][2] = 1 - 2 * (x * x + y * y);
}

static const char *
rnum(double v)
{
	static char b[12][64];
	static int k;
	char *s = b[k++ % 12];

	if (fabs(v) < 1e-13)
		v = 0;
	forms_num(v, s, 64);
	if (s[0] == '-' && flip_sum(s + 1))
		snprintf(s, 64, "%s", flip_sum(s + 1));
	return s;
}

static void
print_matrix(double m[3][3], const char *ind)
{
	int r;

	for (r = 0; r < 3; r++)
		printf("%s[%s, %s, %s]\n", ind, rnum(m[r][0]), rnum(m[r][1]),
		    rnum(m[r][2]));
}

static void
q_normalize(double *q)
{
	double n = sqrt(h_n2(q, 4));
	int k;

	if (n == 0)
		fail("the zero quaternion is not a rotation");
	if (fabs(n - 1) > 1e-12)
		note("q is not a unit quaternion (|q| = %s): normalised",
		    rnum(n));
	for (k = 0; k < 4; k++)
		q[k] /= n;
}

static void
axis_angle_of(const double *q, double *ax, double *ang)
{
	double s = h_vabs(q, 4);

	*ang = 2 * atan2(s, q[0]);
	if (s == 0) {
		ax[0] = 0, ax[1] = 0, ax[2] = 1;
		return;
	}
	ax[0] = q[1] / s, ax[1] = q[2] / s, ax[2] = q[3] / s;
}

static void
do_rotate(const double *v3, const double *q0)
{
	double q[HM] = {0}, v[HM] = {0}, qi[HM], t[HM], r[HM], m[3][3], mv[3],
	    ax[3], ang;
	int k, ok = 1;

	memcpy(q, q0, 4 * sizeof *q);
	q_normalize(q);
	notes_flush();
	v[1] = v3[0], v[2] = v3[1], v[3] = v3[2];
	h_inv(q, qi, 4);
	h_mul(q, v, t, 4);
	h_mul(t, qi, r, 4);
	axis_angle_of(q, ax, &ang);
	printf("  q = %s: a rotation by %s about (%s, %s, %s)\n",
	    hv_str(q, 4, 2), rnum(ang), rnum(ax[0]), rnum(ax[1]), rnum(ax[2]));
	printf("  v = %s (the vector as a pure quaternion)\n", hv_str(v, 4, 2));
	printf("  q v q^-1 = %s\n", hv_str(r, 4, 2));
	printf("  rotated: (%s, %s, %s)\n", rnum(r[1]), rnum(r[2]), rnum(r[3]));
	q_matrix(q, m);
	for (k = 0; k < 3; k++) {
		mv[k] = m[k][0] * v3[0] + m[k][1] * v3[1] + m[k][2] * v3[2];
		if (fabs(mv[k] - r[k + 1]) > 1e-9)
			ok = 0;
	}
	printf("  check: |v'| = |v| = %s, the real part is %s, and the "
	    "rotation matrix gives %s\n", rnum(sqrt(h_n2(v, 4))),
	    fabs(r[0]) < 1e-12 ? "0" : rnum(r[0]), ok ? "the same vector" :
	    "a DIFFERENT vector");
	printf("  rotation matrix:\n");
	print_matrix(m, "    ");
	printf("  (right-handed: counterclockwise looking down the axis "
	    "towards the origin; -q gives the same rotation)\n");
}

static int
axis_of(const struct toks *t, uint32_t a, uint32_t b, double *ax)
{
	char w[16];
	double v[3][HM];
	int n, lg, k;

	while (a < b && t->kind[a] == T_ID && (tok_is(t, a, "the") ||
	    tok_is(t, a, "an") || tok_is(t, a, "a")))
		a++;
	if (a < b && t->kind[a] == T_ID) {
		tok_text(t, a, w, sizeof w);
		if ((!strcmp(w, "x") || !strcmp(w, "y") || !strcmp(w, "z")) &&
		    (a + 1 == b || tok_is(t, a + 1, "axis") ||
		    (t->kind[a + 1] == T_MINUS && a + 2 < b &&
		    tok_is(t, a + 2, "axis")))) {
			ax[0] = w[0] == 'x', ax[1] = w[0] == 'y', ax[2] = w[0] == 'z';
			return 1;
		}
	}
	while (b > a && t->kind[b - 1] == T_ID && tok_is(t, b - 1, "axis"))
		b--;
	n = const_list(t, a, b, v, 3, &lg, 1);
	if (n != 3)
		fail("the axis is a vector (x, y, z), or x, y, z");
	for (k = 0; k < 3; k++) {
		if (!h_isreal(v[k], 2))
			fail("the axis needs real components");
		ax[k] = v[k][0];
	}
	return 1;
}

/* an angle: radians, or degrees with a word or when it is large */
static double
angle_of(const struct toks *t, uint32_t a, uint32_t b)
{
	double v[HM];
	int deg = 0;

	while (b > a && t->kind[b - 1] == T_ID && (tok_is(t, b - 1, "degrees") ||
	    tok_is(t, b - 1, "degree") || tok_is(t, b - 1, "deg") ||
	    tok_is(t, b - 1, "radians") || tok_is(t, b - 1, "rad"))) {
		deg = tok_is(t, b - 1, "degrees") || tok_is(t, b - 1, "degree") ||
		    tok_is(t, b - 1, "deg") ? 1 : -1;
		b--;
	}
	if (!hy_const(t, a, b, 1, v, NULL) || !h_isreal(v, 2))
		fail("the angle must be a real number");
	if (deg == 0 && fabs(v[0]) > 2 * M_PI && v[0] == floor(v[0])) {
		note("the angle %s is read as degrees (say %s rad for radians)",
		    rnum(v[0]), rnum(v[0]));
		deg = 1;
	}
	return deg == 1 ? v[0] * M_PI / 180 : v[0];
}

static void
quat_axis_angle(const double *ax, double ang, double *q)
{
	double n = sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
	int k;

	if (n == 0)
		fail("the axis is the zero vector");
	memset(q, 0, HM * sizeof *q);
	q[0] = cos(ang / 2);
	for (k = 0; k < 3; k++)
		q[k + 1] = sin(ang / 2) * ax[k] / n;
	if (fabs(q[0]) < 1e-15)
		q[0] = 0;
}

static void
cmd_rotate(const struct toks *t, uint32_t a, uint32_t b)
{
	static const char *const by_w[] = {"by", NULL};
	static const char *const ab_w[] = {"about", "around", "round", NULL};
	uint32_t by = find_word(t, a, b, by_w), ab, i;
	double v[3][HM], v3[3], q[HM], ax[3], ang;
	int n, lg, k;

	/* "... using quaternions", "with a quaternion" */
	for (i = by; i < b; i++)
		if (tok_is(t, i, "using") || tok_is(t, i, "via") ||
		    (tok_is(t, i, "with") && i + 1 < b && (tok_is(t, i + 1, "a") ||
		    tok_is(t, i + 1, "quaternions") ||
		    tok_is(t, i + 1, "quaternion")))) {
			b = i;
			break;
		}

	if (by == b)
		fail("rotate V by Q, or rotate V by ANGLE about AXIS");
	n = const_list(t, a, by, v, 3, &lg, 1);
	if (n != 3)
		fail("rotate: the vector has 3 components (x, y, z)");
	for (k = 0; k < 3; k++) {
		if (!h_isreal(v[k], 2))
			fail("rotate: the vector needs real components");
		v3[k] = v[k][0];
	}
	ab = find_word(t, by + 1, b, ab_w);
	printf("rotate (%s, %s, %s)", rnum(v3[0]), rnum(v3[1]), rnum(v3[2]));
	if (ab < b) {
		ang = angle_of(t, by + 1, ab);
		axis_of(t, ab + 1, b, ax);
		quat_axis_angle(ax, ang, q);
		printf(" by %s about (%s, %s, %s):\n", rnum(ang), rnum(ax[0]),
		    rnum(ax[1]), rnum(ax[2]));
		printf("  q = cos(theta/2) + sin(theta/2) u, u the unit axis\n");
	} else {
		if (!hy_const(t, by + 1, b, 2, q, &lg))
			fail("rotate: the quaternion is undefined");
		if (lg > 2)
			fail("rotate: q must be a quaternion");
		printf(" by the quaternion %s:\n", hv_str(q, 4, 2));
	}
	notes_flush();
	do_rotate(v3, q);
}

static void
cmd_slerp(const struct toks *t, uint32_t a, uint32_t b)
{
	double v[3][HM], q1[HM] = {0}, q2[HM] = {0}, r[HM] = {0}, d, th, tt,
	    chk[HM], qi[HM], m[HM];
	int n, lg, k;

	n = const_list(t, a, b, v, 3, &lg, 2);
	if (n != 3 || !h_isreal(v[2], 4))
		fail("slerp Q1, Q2, T (T real, 0..1)");
	memcpy(q1, v[0], 4 * sizeof(double));
	memcpy(q2, v[1], 4 * sizeof(double));
	tt = v[2][0];
	q_normalize(q1);
	q_normalize(q2);
	for (d = 0, k = 0; k < 4; k++)
		d += q1[k] * q2[k];
	if (d < 0) {
		note("q1 . q2 < 0: q2 replaced by -q2 (the same rotation), so the "
		    "path is the shorter arc");
		for (k = 0; k < 4; k++)
			q2[k] = -q2[k];
		d = -d;
	}
	th = acos(d > 1 ? 1 : d);
	for (k = 0; k < 4; k++)
		r[k] = th < 1e-12 ? q1[k] : (sin((1 - tt) * th) * q1[k] +
		    sin(tt * th) * q2[k]) / sin(th);
	notes_flush();
	printf("slerp(%s, %s, %s) = %s\n", hv_str(q1, 4, 2), hv_str(q2, 4, 2),
	    rnum(tt), hv_str(r, 4, 2));
	/* q1 (q1^-1 q2)^t */
	h_inv(q1, qi, 4);
	h_mul(qi, q2, m, 4);
	{
		double e[HM] = {0};
		e[0] = tt;
		h_pow(m, e, m, 4);
	}
	h_mul(q1, m, chk, 4);
	printf("  the arc between them on the unit sphere: a = %s (a rotation "
	    "by %s); slerp = (sin((1-t)a) q1 + sin(ta) q2)/sin(a)\n", rnum(th),
	    rnum(2 * th));
	printf("  check: |slerp| = %s, and q1 (q1^-1 q2)^t = %s: %s\n",
	    rnum(sqrt(h_n2(r, 4))), hv_str(chk, 4, 2), h_close(chk, r, 4) ?
	    "the same" : "DIFFERENT");
}

static void
cmd_axisangle(const struct toks *t, uint32_t a, uint32_t b)
{
	double v[4][HM], q[HM], ax[3], ang;
	int n, lg, k;

	n = const_list(t, a, b, v, 4, &lg, 2);
	if (n == 1) {
		memcpy(q, v[0], sizeof q);
		q_normalize(q);
		notes_flush();
		axis_angle_of(q, ax, &ang);
		printf("%s = cos(theta/2) + sin(theta/2) u:\n", hv_str(q, 4, 2));
		printf("  theta = %s (%s degrees), axis u = (%s, %s, %s)\n",
		    rnum(ang), rnum(ang * 180 / M_PI), rnum(ax[0]), rnum(ax[1]),
		    rnum(ax[2]));
		return;
	}
	if (n != 2 || lg > 1)
		fail("axisangle AXIS, ANGLE  or  axisangle Q");
	/* (x, y, z), angle: the axis list was split; read again */
	fail("axisangle: write the axis in parentheses: axisangle (0, 0, 1), "
	    "pi/2");
	(void)k;
}

static void
cmd_axisangle2(const struct toks *t, uint32_t a, uint32_t b)
{
	uint32_t c;
	double ax[3], ang, q[HM], m[3][3];

	if (a < b && t->kind[a] == T_LP && (c = tok_close(t, a, b)) < b &&
	    t->kind[c] == T_COMMA) {
		axis_of(t, a, c, ax);
		ang = angle_of(t, c + 1, b);
		quat_axis_angle(ax, ang, q);
		notes_flush();
		printf("the rotation by %s about (%s, %s, %s):\n", rnum(ang),
		    rnum(ax[0]), rnum(ax[1]), rnum(ax[2]));
		printf("  q = cos(theta/2) + sin(theta/2) u = %s\n", hv_str(q, 4, 2));
		q_matrix(q, m);
		printf("  rotation matrix:\n");
		print_matrix(m, "    ");
		return;
	}
	cmd_axisangle(t, a, b);
}

static void
cmd_rotmatrix(const struct toks *t, uint32_t a, uint32_t b)
{
	double q[HM], m[3][3];
	int lg;

	if (!hy_const(t, a, b, 2, q, &lg) || lg > 2)
		fail("rotmatrix Q: Q a quaternion");
	q_normalize(q);
	notes_flush();
	q_matrix(q, m);
	printf("the rotation matrix of %s (v -> q v q^-1):\n", hv_str(q, 4, 2));
	print_matrix(m, "  ");
}

static void
cmd_frommatrix(const struct toks *t, uint32_t a, uint32_t b)
{
	double v[9][HM], m[3][3], q[HM] = {0}, s, det, chk[3][3], e = 0;
	int n, lg, r, c;

	n = const_list(t, a, b, v, 9, &lg, 1);
	if (n != 9)
		fail("frommatrix: 9 entries, row by row");
	for (r = 0; r < 3; r++)
		for (c = 0; c < 3; c++)
			m[r][c] = v[r * 3 + c][0];
	det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
	    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
	    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	for (r = 0; r < 3; r++)
		for (c = 0; c < 3; c++) {
			double d = m[0][r] * m[0][c] + m[1][r] * m[1][c] +
			    m[2][r] * m[2][c] - (r == c);
			if (fabs(d) > e)
				e = fabs(d);
		}
	if (e > 1e-9 || fabs(det - 1) > 1e-9)
		fail("frommatrix: not a rotation (M^T M = I and det M = 1 needed; "
		    "det = %s)", rnum(det));
	s = m[0][0] + m[1][1] + m[2][2];
	if (s > 0) {
		double k = sqrt(s + 1) * 2;
		q[0] = k / 4;
		q[1] = (m[2][1] - m[1][2]) / k;
		q[2] = (m[0][2] - m[2][0]) / k;
		q[3] = (m[1][0] - m[0][1]) / k;
	} else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
		double k = sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
		q[0] = (m[2][1] - m[1][2]) / k;
		q[1] = k / 4;
		q[2] = (m[0][1] + m[1][0]) / k;
		q[3] = (m[0][2] + m[2][0]) / k;
	} else if (m[1][1] > m[2][2]) {
		double k = sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
		q[0] = (m[0][2] - m[2][0]) / k;
		q[1] = (m[0][1] + m[1][0]) / k;
		q[2] = k / 4;
		q[3] = (m[1][2] + m[2][1]) / k;
	} else {
		double k = sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
		q[0] = (m[1][0] - m[0][1]) / k;
		q[1] = (m[0][2] + m[2][0]) / k;
		q[2] = (m[1][2] + m[2][1]) / k;
		q[3] = k / 4;
	}
	printf("the unit quaternion of the rotation matrix: q = %s (or -q)\n",
	    hv_str(q, 4, 2));
	q_matrix(q, chk);
	for (e = 0, r = 0; r < 3; r++)
		for (c = 0; c < 3; c++)
			if (fabs(chk[r][c] - m[r][c]) > e)
				e = fabs(chk[r][c] - m[r][c]);
	printf("  check: q's matrix equals the given one (max difference "
	    "%.0e)\n", e);
}

/* ---- the dispatcher ---- */

static const char *const opt_words[] = {"max", "nofile", "to", "at", "as",
	"around", "radius", "from", "along", "clockwise", NULL};

static void
cmd_hplot(const struct toks *t, uint32_t a, uint32_t b)
{
	uint32_t e = find_word(t, a, b, opt_words), i;
	struct func *f;
	double x0 = -2, x1 = 2, y0 = -2, y1 = 2;
	char name[64] = "plot";
	int nofile = 0;

	for (i = e; i < b; i++) {
		if (tok_is(t, i, "nofile"))
			nofile = 1;
		else if (tok_is(t, i, "to") && i + 1 < b && t->kind[i + 1] == T_ID)
			tok_text(t, ++i, name, sizeof name);
		else if (tok_is(t, i, "radius") && i + 1 < b) {
			double v[HM];
			uint32_t j = i + 1;
			while (j < b && t->kind[j] != T_ID)
				j++;
			if (!hy_const(t, i + 1, j, 1, v, NULL) || v[0] <= 0)
				fail("plot: radius R > 0");
			x0 = y0 = -v[0];
			x1 = y1 = v[0];
			i = j - 1;
		}
	}
	f = hy_funarg(t, a, e, 0);
	if (f == NULL)
		fail("plot: not a complex function");
	notes_flush();
	cmd_dcolour(f, x0, x1, y0, y1, name, nofile);
}

/* |f| or re/im of f as a surface over (x, y), by plot.c */
static int
plot_part(const struct toks *t, uint32_t a, uint32_t b)
{
	uint32_t e, inner0, inner1;
	struct func *f;
	struct sbuf re = {0}, im = {0}, st = {0}, say = {0};
	char *text, *opts;
	const char *which;

	e = find_word(t, a, b, opt_words);
	if (t->kind[a] == T_BAR && t->kind[e - 1] == T_BAR && e - 1 > a + 1) {
		which = "abs";
		inner0 = a + 1;
		inner1 = e - 1;
	} else if (t->kind[a] == T_ID && (tok_is(t, a, "abs") ||
	    tok_is(t, a, "re") || tok_is(t, a, "im") || tok_is(t, a, "modulus")) &&
	    a + 1 < e) {
		which = tok_is(t, a, "re") ? "re" : tok_is(t, a, "im") ? "im" :
		    "abs";
		inner0 = a + 1;
		inner1 = e;
		if (t->kind[inner0] == T_LP && tok_close(t, inner0, e) == e)
			inner0++, inner1--;
	} else
		return 0;
	f = hy_funarg(t, inner0, inner1, 0);
	if (f == NULL || f->halg != 1)
		return 0;
	text = xy_text(f, &say);
	if (!forms_reim(text, &re, &im))
		fail("plot: could not split %s into real and imaginary parts", text);
	if (say.n)
		note("%s", say.p);
	notes_flush();
	opts = toks_text(t, e, b);
	if (!strcmp(which, "abs"))
		sb_printf(&st, "plot sqrt((%s)^2 + (%s)^2) %s\n", re.p, im.p, opts);
	else
		sb_printf(&st, "plot %s %s\n", !strcmp(which, "re") ? re.p : im.p,
		    opts);
	printf("  %s(%s) as a surface over the plane (%s):\n",
	    !strcmp(which, "abs") ? "|f|" : which, fname(f), say.p);
	run_source(st.p, "<hyper>", 0);
	free(opts);
	free(text);
	free(re.p);
	free(im.p);
	free(st.p);
	free(say.p);
	return 1;
}

static int
mentions_hyper(const struct toks *t, uint32_t a, uint32_t b)
{
	uint32_t i;
	char w[64];
	struct func *g;

	for (i = a; i < b; i++)
		if (t->kind[i] == T_ID) {
			tok_text(t, i, w, sizeof w);
			if ((g = func_find(sym_cstr(w))) != NULL && g->halg)
				return 1;
		}
	return 0;
}

/* 0: not ours; 1 done */
static int
hy_command(const struct toks *t, uint32_t b, const char *w0)
{
	struct func *f;
	uint32_t e, n;
	int lg;

	if (b < 2 || t->kind[1] == T_ASSIGN)
		return 0;
	/* the algebras */
	if (!strcmp(w0, "table") && (lg = alg_word(t, 1, b, &n)) >= 1 &&
	    n == b && !func_find(sym_cstr("quaternion"))) {
		rep_begin("multiplication_table");
		rep_text_fields();
		cmd_cayley(lg);
		return 1;
	}
	if (!strcmp(w0, "algebra") || !strcmp(w0, "zerodivisors")) {
		char prop[32] = "";
		if ((lg = alg_word(t, 1, b, &n)) < 1)
			lg = w0[0] == 'z' ? 4 : -1;
		if (lg < 1)
			fail("%s complex|quaternion|octonion|sedenion|cd N", w0);
		if (n < b && t->kind[n] == T_ID)
			tok_text(t, n, prop, sizeof prop);
		rep_begin(w0[0] == 'z' ? "zerodivisors" : "algebra");
		rep_text_fields();
		if (w0[0] == 'z')
			cmd_zerodiv(lg);
		else
			cmd_algebra(lg, prop[0] ? prop : NULL);
		return 1;
	}
	if (!strcmp(w0, "associator") || !strcmp(w0, "commutator")) {
		rep_begin(w0[0] == 'c' ? "commutator" : "associator");
		rep_text_fields();
		cmd_assoc(t, 1, b, w0[0] == 'c');
		hn = 0;
		return 1;
	}
	if (!strcmp(w0, "rotate") && !func_find(sym_cstr("rotate")) &&
	    find_word(t, 1, b, (const char *const[]){"by", NULL}) < b) {
		/* without "by": a solid of revolution, English (viz.c) */
		rep_begin("rotate");
		rep_text_fields();
		cmd_rotate(t, 1, b);
		hn = 0;
		return 1;
	}
	if (!strcmp(w0, "slerp")) {
		rep_begin("slerp");
		rep_text_fields();
		cmd_slerp(t, 1, b);
		hn = 0;
		return 1;
	}
	if (!strcmp(w0, "axisangle")) {
		rep_begin("axisangle");
		rep_text_fields();
		cmd_axisangle2(t, 1, b);
		hn = 0;
		return 1;
	}
	if (!strcmp(w0, "rotmatrix")) {
		rep_begin("rotmatrix");
		rep_text_fields();
		cmd_rotmatrix(t, 1, b);
		hn = 0;
		return 1;
	}
	if (!strcmp(w0, "frommatrix")) {
		rep_begin("frommatrix");
		rep_text_fields();
		cmd_frommatrix(t, 1, b);
		hn = 0;
		return 1;
	}
	if (func_find(sym_cstr(w0)))
		return 0;
	/* commands on a complex function */
	if (!strcmp(w0, "show")) {
		if ((f = hy_funarg(t, 1, b, 0)) == NULL)
			return 0;
		notes_flush();
		hy_show(f);
		return 1;
	}
	if (!strcmp(w0, "table")) {
		static const char *const mx[] = {"max", NULL};
		int max = 24;
		e = find_word(t, 1, b, mx);
		if ((f = hy_funarg(t, 1, e, 0)) == NULL)
			return 0;
		if (e + 1 < b) {
			char nb[16];
			tok_text(t, e + 1, nb, sizeof nb);
			max = atoi(nb);
		}
		notes_flush();
		hy_table(f, max > 0 ? max : 24);
		return 1;
	}
	if (!strcmp(w0, "zeros") || !strcmp(w0, "roots") ||
	    !strcmp(w0, "solve")) {
		uint32_t a = 1;
		if (tok_is(t, 1, "of"))
			a = 2;
		if ((f = hy_funarg(t, a, b, 0)) == NULL)
			return 0;
		notes_flush();
		cmd_zeros(f);
		return 1;
	}
	if (!strcmp(w0, "holomorphic") || !strcmp(w0, "analytic") ||
	    !strcmp(w0, "diff") || !strcmp(w0, "derivative")) {
		int dv = w0[0] == 'd';
		uint32_t bb = b;
		/* diff F z: the variable named */
		if (dv && b > 2 && t->kind[b - 1] == T_ID &&
		    (t->fl[b - 1] & TF_SPACE)) {
			char w[32];
			tok_text(t, b - 1, w, sizeof w);
			if (default_complex(w))
				bb = b - 1;
		}
		if ((f = hy_funarg(t, 1, bb, !dv)) == NULL)
			return 0;
		notes_flush();
		cmd_holo(f, dv);
		return 1;
	}
	if (!strcmp(w0, "residue") || !strcmp(w0, "residues") ||
	    !strcmp(w0, "poles")) {
		static const char *const at_w[] = {"at", NULL};
		e = find_word(t, 1, b, at_w);
		if ((f = hy_funarg(t, 1, e, 1)) == NULL)
			return 0;
		notes_flush();
		if (w0[0] == 'p' || e == b) {
			cmd_poles(f);
			return 1;
		}
		{
			int inf;
			cplx a = tok_point(t, e + 1, b, &inf);
			if (inf)
				fail("residue at infinity is not supported");
			notes_flush();
			cmd_residue(f, a);
		}
		return 1;
	}
	if (!strcmp(w0, "limit") || !strcmp(w0, "lim")) {
		uint32_t i, ar = b, fa = 1, fb = b, pa = b, pb = b;
		int inf;
		cplx a;
		for (i = 1; i < b; i++)
			if (t->kind[i] == T_ARROW) {
				ar = i;
				break;
			}
		if (ar < b) {
			if (ar == 2 || (ar == 3 && t->kind[1] == T_LP)) {
				/* lim z -> a F, lim(z -> a) F */
				if (t->kind[1] == T_LP) {
					pa = ar + 1;
					pb = tok_close(t, 1, b) - 1;
					fa = pb + 1;
				} else {
					pa = ar + 1;
					for (pb = pa + 1; pb < b && !(t->fl[pb] & TF_SPACE); pb++)
						;
					fa = pb;
				}
				fb = b;
			} else {
				fb = ar - 1;
				if (fb > 1 && (tok_is(t, fb - 1, "as") ||
				    t->kind[fb - 1] == T_COMMA))
					fb--;
				pa = ar + 1;
				pb = b;
			}
		} else {
			static const char *const at_w[] = {"at", NULL};
			fb = find_word(t, 1, b, at_w);
			pa = fb + 1;
			pb = b;
			if (fb == b)
				return 0;
		}
		if ((f = hy_funarg(t, fa, fb, 0)) == NULL)
			return 0;
		a = tok_point(t, pa, pb, &inf);
		notes_flush();
		cmd_limit(f, a, inf);
		return 1;
	}
	if (!strcmp(w0, "integrate") || !strcmp(w0, "\xe2\x88\xab")) {
		static const char *const ar_w[] = {"around", "over", "along",
			"on", "from", NULL};
		uint32_t i;
		e = find_word(t, 1, b, ar_w);
		if (e == b)
			return 0;
		if (tok_is(t, e, "from")) {
			static const char *const to_w[] = {"to", NULL};
			uint32_t to = find_word(t, e + 1, b, to_w);
			int inf;
			cplx za, zb;
			if (to == b || (f = hy_funarg(t, 1, e, 0)) == NULL)
				return 0;
			za = tok_point(t, e + 1, to, &inf);
			zb = tok_point(t, to + 1, b, &inf);
			notes_flush();
			rep_begin("segment_integral");
			rep_text_fields();
			cmd_segment(f, za, zb);
			return 1;
		}
		if ((f = hy_funarg(t, 1, e, 1)) == NULL)
			return 0;
		{
			cplx c = 0;
			double r = 1, v[HM];
			int cw = 0;
			uint32_t s = e + 1, bb = b;
			for (i = s; i < b; i++)
				if (tok_is(t, i, "clockwise"))
					cw = 1, bb = i;
			while (s < bb && (tok_is(t, s, "the") || tok_is(t, s, "a")))
				s++;
			if (s < bb && tok_is(t, s, "circle") && s + 1 < bb &&
			    t->kind[s + 1] == T_BAR)
				s++;
			if (s < bb && tok_is(t, s, "unit"))
				;	/* the unit circle */
			else if (s < bb && tok_is(t, s, "circle")) {
				/* circle of radius R [about | around | at | centred at C] */
				static const char *const cw_[] = {"about", "around", "at",
					"centered", "centred", "center", "centre", NULL};
				uint32_t r0 = s + 1, r1;
				int inf;
				if (r0 < bb && tok_is(t, r0, "of"))
					r0++;
				if (r0 < bb && tok_is(t, r0, "radius"))
					r0++;
				r1 = find_word(t, r0, bb, cw_);
				if (!hy_const(t, r0, r1, 1, v, NULL) || v[0] <= 0 || v[1] != 0)
					fail("the radius must be a positive real");
				r = v[0];
				if (r1 < bb) {
					uint32_t c0 = r1 + 1;
					if (c0 < bb && tok_is(t, c0, "at"))
						c0++;
					c = tok_point(t, c0, bb, &inf);
				}
			} else if (s < bb && t->kind[s] == T_BAR) {
				uint32_t c2;
				struct func *g;
				cplx g0, g1;
				for (c2 = s + 1; c2 < bb && t->kind[c2] != T_BAR; c2++)
					;
				if (c2 + 1 >= bb || t->kind[c2 + 1] != T_ASSIGN)
					fail("integrate F around |z - c| = r");
				g = hy_funarg(t, s + 1, c2, 1);
				if (!hy_const(t, c2 + 2, bb, 1, v, NULL) || v[0] <= 0 ||
				    v[1] != 0)
					fail("the radius must be a positive real");
				r = v[0];
				{
					struct func *keep = f;
					fz_open(g, "integrate");
					if (!fz(0, &g0) || !fz(1, &g1) || cabs(g1 - g0 - 1) > 1e-12)
						fail("integrate F around |z - c| = r: z - c must be "
						    "linear in z");
					fz_close();
					c = -g0;
					f = keep;
				}
			} else
				fail("integrate F around |z - c| = r (or the unit circle)");
			notes_flush();
			rep_begin("contour_integral");
			rep_text_fields();
			cmd_contour(f, c, r, cw);
		}
		return 1;
	}
	if (!strcmp(w0, "plot") || !strcmp(w0, "graph")) {
		if (plot_part(t, 1, b))
			return 1;
		e = find_word(t, 1, b, opt_words);
		if ((f = hy_funarg(t, 1, e, 0)) == NULL)
			return 0;
		cmd_hplot(t, 1, b);
		return 1;
	}
	if (!strcmp(w0, "complex") || !strcmp(w0, "rect") ||
	    !strcmp(w0, "re") || !strcmp(w0, "im") || !strcmp(w0, "conj") ||
	    !strcmp(w0, "modulus") || !strcmp(w0, "arg") ||
	    !strcmp(w0, "polar")) {
		int j, cv = 0;
		if ((f = hy_funarg(t, 1, b, 0)) == NULL)
			return 0;
		if (f->halg >= 2 && f->np == 0) {
			notes_drop();
			return hy_bare(t, b);	/* conj(1 + 2i + 3j): a value */
		}
		for (j = 0; j < f->np; j++)
			if (f->hk[j])
				cv = 1;
		if (f->halg == 1 && !cv && f->anon) {
			notes_drop();
			return 0;	/* constants: forms.c as before */
		}
		cmd_forms(w0, f);
		return 1;
	}
	if (in_list(ucas_cmds, w0) && mentions_hyper(t, 1, b))
		fail("%s works on real functions; %s is complex (use re, im, "
		    "modulus or arg to get a real one, or define it with real "
		    "parameters)", w0, "this one");
	return 0;
}

/* nlp.c: does text need complex (1), quaternion (2) ... numbers? */
int
hyper_text_lg(const char *text)
{
	struct toks *t = xcalloc(1, sizeof *t);
	struct hscan S;
	jmp_buf saved;
	uint32_t b;
	int lg = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0) {
		lex(t, text);
		for (b = 0; b < t->n && t->kind[b] != T_EOF; b++)
			;
		memset(&S, 0, sizeof S);
		if (hy_scan(t, 0, b, &S)) {
			hy_decide(&S, t, 0, b);
			lg = S.lg;
		}
	}
	memcpy(fail_jmp, saved, sizeof saved);
	toks_free(t);
	free(t);
	return lg;
}
