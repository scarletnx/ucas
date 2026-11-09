/*
 * forms.c - complex numbers, polar form, trig rewrites.
 *
 * The node DAG is real-valued.  Inside these commands `i` is the imaginary
 * unit (also `j`, `I`, `im`); every other variable is taken as real.  An
 * expression is mapped to a pair of real nodes (Re, Im) by the rules of
 * complex arithmetic with principal branches (sqrt, log, non-integer
 * powers, the inverse functions), then each part is normalised: a sum of
 * monomials with numeric coefficients over atoms (variables and function
 * calls with normalised arguments), sin^2 + cos^2 = 1 and cosh^2 - sinh^2
 * = 1 folded.  Constants are printed exactly where they are recognised
 * (rationals, rational multiples of pi, square roots, (p + k sqrt(m))/q,
 * logarithms of small integers).
 *
 * Every printed result is parsed back as ucas text and compared with the
 * original at sample points, the original evaluated by C's complex
 * arithmetic (cexp, clog, cpow, csin ...), so a rewrite is never trusted
 * without a check.  Identities that need a condition carry it as an
 * expression; the check is made where it holds, and the points outside
 * where the rewrite differs are counted to show that the condition is
 * needed.
 *
 *   complex F | rect F     a + bi                    re F, im F, conj F
 *   polar F                r e^(i theta), r(cos theta + i sin theta)
 *   modulus F | arg F      |z|, the principal argument in (-pi, pi]
 *   nthroots N F           the N complex N-th roots;  unity N
 *   trig F [mode]          equivalent trig forms, each checked
 *
 * It also adds the functions asin acos atan2 sinh cosh tanh asinh acosh
 * atanh to the language (the evaluator calls forms_scalar).
 */
#include "calc.h"

#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

/* ---- the functions this file adds ---- */

int
forms_scalar(int op, double a, double b, double *r)
{
	switch (op) {
	case O_ASIN:
	case O_ACOS:
		if (!(a >= -1 && a <= 1))
			return 0;
		*r = op == O_ASIN ? asin(a) : acos(a);
		break;
	case O_ATAN2:
		if (a == 0 && b == 0)
			return 0;
		if (a == 0)
			a = 0;	/* -0: the negative real axis has angle pi */
		*r = atan2(a, b);
		break;
	case O_SINH: *r = sinh(a); break;
	case O_COSH: *r = cosh(a); break;
	case O_TANH: *r = tanh(a); break;
	case O_ASINH: *r = asinh(a); break;
	case O_ACOSH:
		if (!(a >= 1))
			return 0;
		*r = acosh(a);
		break;
	case O_ATANH:
		if (!(a > -1 && a < 1))
			return 0;
		*r = atanh(a);
		break;
	default:
		return 0;
	}
	return isfinite(*r);
}

const char *
forms_cond(nid n)
{
	static char buf[4][256];
	static int k;
	char *b = buf[k++ & 3];
	const char *a = node_str(N.a[n]);

	switch (N.op[n]) {
	case O_ASIN:
	case O_ACOS:
		snprintf(b, 256, "-1 <= %s <= 1", a);
		break;
	case O_ATAN2:
		snprintf(b, 256, "(%s, %s) != (0, 0)  (atan2: the angle of the "
		    "origin is undefined)", a, node_str(N.b[n]));
		break;
	case O_ACOSH:
		snprintf(b, 256, "%s >= 1", a);
		break;
	case O_ATANH:
		snprintf(b, 256, "-1 < %s < 1", a);
		break;
	default:
		return NULL;
	}
	return b;
}

#define K(v) mk_const_i(TY_LIT, (int64_t)(v))
#define U1(op, a) mk_un((op), (a))
#define B2(op, a, b) mk_bin((op), (a), (b))

nid
forms_diff(nid e, uint32_t slot)
{
	nid u = N.a[e], du = diff_node(u, slot), r;

	switch (N.op[e]) {
	case O_ASIN:	/* u'/sqrt(1 - u^2) */
	case O_ACOS:
		r = B2(O_DIV, du, U1(O_SQRT, B2(O_SUB, K(1), B2(O_POW, u, K(2)))));
		return N.op[e] == O_ACOS ? U1(O_NEG, r) : r;
	case O_ATAN2: {	/* (x y' - y x')/(x^2 + y^2) */
		nid x = N.b[e], dx = diff_node(x, slot);
		return B2(O_DIV, B2(O_SUB, B2(O_MUL, x, du), B2(O_MUL, u, dx)),
		    B2(O_ADD, B2(O_POW, x, K(2)), B2(O_POW, u, K(2))));
	}
	case O_SINH:
		return B2(O_MUL, U1(O_COSH, u), du);
	case O_COSH:
		return B2(O_MUL, U1(O_SINH, u), du);
	case O_TANH:
		return B2(O_DIV, du, B2(O_POW, U1(O_COSH, u), K(2)));
	case O_ASINH:
		return B2(O_DIV, du, U1(O_SQRT, B2(O_ADD, B2(O_POW, u, K(2)),
		    K(1))));
	case O_ACOSH:
		return B2(O_DIV, du, U1(O_SQRT, B2(O_SUB, B2(O_POW, u, K(2)),
		    K(1))));
	case O_ATANH:
		return B2(O_DIV, du, B2(O_SUB, K(1), B2(O_POW, u, K(2))));
	}
	fail("no derivative for %s", opinfo[N.op[e]].name);
	return 0;
}

/* ---- numbers: exact-looking forms of a double ---- */

struct cf {
	char num[96], den[24];
	int neg;
	int sum;	/* num is a parenthesised sum */
	int unit;	/* num is "1" (only a denominator) */
	int exact;	/* recognised; otherwise %.10g */
};

static int
near(double a, double b)
{
	return fabs(a - b) <= 1e-11 * (fabs(b) > 1 ? fabs(b) : 1);
}

static int
small_q(double v, int64_t maxd, int64_t maxn, int64_t *pn, int64_t *pd)
{
	struct q q;

	if (!q_from_double(v, &q) || q.d > maxd || llabs(q.n) > maxn)
		return 0;
	if (!near((double)q.n / (double)q.d, v))
		return 0;
	*pn = q.n;
	*pd = q.d;
	return 1;
}

/* n = k^2 m with m squarefree */
static void
sqfree(int64_t n, int64_t *k, int64_t *m)
{
	int64_t f;

	*k = 1;
	for (f = 2; f * f <= n; f++)
		while (n % (f * f) == 0) {
			n /= f * f;
			*k *= f;
		}
	*m = n;
}

static void
cfmt(double v, struct cf *c)
{
	double a = fabs(v);
	int64_t n, d, k, m;
	static const int ms[] = {2, 3, 5, 6, 7, 10, 11, 13, 15, 0};
	static const int lg[] = {2, 3, 5, 7, 10, 0};
	int i, q, kk;

	memset(c, 0, sizeof *c);
	c->neg = v < 0;
	c->exact = 1;
	if (a == 0 || a < 1e-14) {
		strcpy(c->num, "0");
		c->neg = 0;
		return;
	}
	if (small_q(a, 100000, 1000000000000LL, &n, &d)) {
		snprintf(c->num, sizeof c->num, "%lld", (long long)n);
		if (d > 1)
			snprintf(c->den, sizeof c->den, "%lld", (long long)d);
		c->unit = n == 1 && d > 1;
		return;
	}
	if (small_q(a / M_PI, 48, 480, &n, &d)) {
		if (n == 1)
			strcpy(c->num, "pi");
		else
			snprintf(c->num, sizeof c->num, "%lldpi", (long long)n);
		if (d > 1)
			snprintf(c->den, sizeof c->den, "%lld", (long long)d);
		return;
	}
	if (near(a, M_E)) {
		strcpy(c->num, "e");
		return;
	}
	/* a = sqrt(n/d) = k sqrt(m)/d' */
	if (small_q(a * a, 10000, 10000000, &n, &d)) {
		int64_t g;
		sqfree(n * d, &k, &m);
		g = gcd64(k, d);
		k /= g;
		d /= g;
		if (k == 1)
			snprintf(c->num, sizeof c->num, "sqrt(%lld)", (long long)m);
		else
			snprintf(c->num, sizeof c->num, "%lldsqrt(%lld)",
			    (long long)k, (long long)m);
		if (d > 1)
			snprintf(c->den, sizeof c->den, "%lld", (long long)d);
		return;
	}
	/* (p + k sqrt(m))/q */
	for (i = 0; ms[i]; i++)
		for (q = 1; q <= 16; q++)
			for (kk = -8; kk <= 8; kk++) {
				double p;
				int64_t pi, g;
				if (kk == 0)
					continue;
				p = a * q - kk * sqrt((double)ms[i]);
				pi = llround(p);
				if (llabs(pi) > 200 || pi == 0 ||
				    fabs(p - (double)pi) > 1e-10 * q)
					continue;
				g = gcd64(gcd64(llabs(pi), llabs(kk)), q);
				if (g != 1)
					continue;
				if (llabs(kk) == 1)
					snprintf(c->num, sizeof c->num, "(%lld %s sqrt(%d))",
					    (long long)pi, kk < 0 ? "-" : "+", ms[i]);
				else
					snprintf(c->num, sizeof c->num,
					    "(%lld %s %dsqrt(%d))", (long long)pi,
					    kk < 0 ? "-" : "+", abs(kk), ms[i]);
				if (q > 1)
					snprintf(c->den, sizeof c->den, "%d", q);
				c->sum = 1;
				return;
			}
	/* sqrt((p + k sqrt(m))/q): sin(2pi/5) = sqrt((5 + sqrt(5))/8) */
	{
		struct cf c2;
		static int depth;
		if (depth == 0) {
			depth++;
			cfmt(a * a, &c2);
			depth--;
			if (c2.sum) {
				snprintf(c->num, sizeof c->num, "sqrt(%s%s%s)", c2.num,
				    c2.den[0] ? "/" : "", c2.den);
				return;
			}
		}
	}
	/* q log(n) */
	for (i = 0; lg[i]; i++)
		if (small_q(a / log((double)lg[i]), 12, 48, &n, &d)) {
			if (n == 1)
				snprintf(c->num, sizeof c->num, "log(%d)", lg[i]);
			else
				snprintf(c->num, sizeof c->num, "%lldlog(%d)",
				    (long long)n, lg[i]);
			if (d > 1)
				snprintf(c->den, sizeof c->den, "%lld", (long long)d);
			return;
		}
	/* exp(q), exp(q pi): i^i = exp(-pi/2) */
	{
		double l = log(a);
		if (small_q(l, 12, 48, &n, &d)) {
			snprintf(c->num, sizeof c->num, d > 1 ? "exp(%lld/%lld)" :
			    "exp(%lld)", (long long)n, (long long)d);
			return;
		}
		if (small_q(l / M_PI, 12, 48, &n, &d)) {
			char t[48];
			if (llabs(n) == 1)
				snprintf(t, sizeof t, "%spi", n < 0 ? "-" : "");
			else
				snprintf(t, sizeof t, "%lldpi", (long long)n);
			if (d > 1)
				snprintf(c->num, sizeof c->num, "exp(%s/%lld)", t,
				    (long long)d);
			else
				snprintf(c->num, sizeof c->num, "exp(%s)", t);
			return;
		}
	}
	/* log(p + k sqrt(m)): asin(2) = pi/2 - i log(2 + sqrt(3)) */
	{
		double b = exp(a);
		for (i = 0; ms[i] && b < 1e6; i++)
			for (kk = 1; kk <= 4; kk++) {
				double p = b - kk * sqrt((double)ms[i]);
				int64_t pi = llround(p);
				if (pi < 1 || pi > 50 || fabs(p - (double)pi) > 1e-10 * b)
					continue;
				if (kk == 1)
					snprintf(c->num, sizeof c->num, "log(%lld + sqrt(%d))",
					    (long long)pi, ms[i]);
				else
					snprintf(c->num, sizeof c->num, "log(%lld + %dsqrt(%d))",
					    (long long)pi, kk, ms[i]);
				return;
			}
		if (small_q(b, 12, 100, &n, &d) && d > 1) {
			snprintf(c->num, sizeof c->num, "log(%lld/%lld)", (long long)n,
			    (long long)d);
			return;
		}
	}
	c->exact = 0;
	snprintf(c->num, sizeof c->num, "%.10g", a);
}

/* the whole number as text */
static const char *
cstr(double v)
{
	static char buf[8][128];
	static int k;
	char *b = buf[k++ & 7];
	struct cf c;

	cfmt(v, &c);
	snprintf(b, 128, "%s%s%s%s", c.neg ? "-" : "", c.num, c.den[0] ? "/" : "",
	    c.den);
	return b;
}

/* ---- the imaginary unit ---- */

static uint32_t sym_i;
static uint64_t unit_slots;	/* slots of the current function that are i */

static int
is_unit_name(const char *s)
{
	return !strcmp(s, "i") || !strcmp(s, "j") || !strcmp(s, "I") ||
	    !strcmp(s, "im");
}

static int
is_unit(nid e)
{
	return N.op[e] == O_VAR && N.ki[e] < 64 &&
	    (unit_slots >> N.ki[e] & 1);
}

static nid
unit_node(void)
{
	return mk_var(63, sym_cstr("i"), TY_REAL);
}

/* ---- printing (ucas syntax that parses back; i for the unit) ---- */

static void pr(struct sbuf *b, nid e, int prec);

enum { PS_SUM = 1, PS_PROD, PS_NEG, PS_POW, PS_ATOM };

static int
num_val(nid e, double *v)
{
	return num_of(e, v);
}

/* a sum's terms with their signs */
static void
flat_sum(nid e, int sg, nid *t, int *s, int *n, int max)
{
	double v;

	if (N.op[e] == O_ADD || N.op[e] == O_SUB) {
		flat_sum(N.a[e], sg, t, s, n, max);
		flat_sum(N.b[e], N.op[e] == O_SUB ? -sg : sg, t, s, n, max);
		return;
	}
	if (N.op[e] == O_NEG) {
		flat_sum(N.a[e], -sg, t, s, n, max);
		return;
	}
	if (*n >= max)
		return;
	/* a negative leading coefficient becomes the sign */
	if (num_val(e, &v) && v < 0) {
		t[*n] = mk_const_r(-v);
		s[(*n)++] = -sg;
		return;
	}
	if ((N.op[e] == O_MUL || N.op[e] == O_DIV) && num_val(N.a[e], &v) &&
	    v < 0) {
		nid a = mk_const_r(-v);
		t[*n] = v == -1 && N.op[e] == O_MUL ? N.b[e] :
		    mk_bin(N.op[e], a, N.b[e]);
		s[(*n)++] = -sg;
		return;
	}
	t[*n] = e;
	s[(*n)++] = sg;
}

/* a product's factors: numeric coefficient, numerator and denominator */
static void
flat_prod(nid e, int inv, double *c, nid *nu, int *nn, nid *de, int *nd)
{
	double v;

	if (N.op[e] == O_MUL || (N.op[e] == O_DIV)) {
		flat_prod(N.a[e], inv, c, nu, nn, de, nd);
		flat_prod(N.b[e], N.op[e] == O_DIV ? !inv : inv, c, nu, nn, de,
		    nd);
		return;
	}
	if (N.op[e] == O_NEG) {
		*c = -*c;
		flat_prod(N.a[e], inv, c, nu, nn, de, nd);
		return;
	}
	if (num_val(e, &v)) {
		if (inv)
			*c /= v;
		else
			*c *= v;
		return;
	}
	if (inv) {
		if (*nd < 32)
			de[(*nd)++] = e;
	} else if (*nn < 32)
		nu[(*nn)++] = e;
}

static int
starts_name(nid e)
{
	int op = N.op[e];

	if (op == O_POW)
		return starts_name(N.a[e]);
	return op == O_VAR || (opinfo[op].arity >= 1 && op != O_NEG &&
	    op != O_ADD && op != O_SUB && op != O_MUL && op != O_DIV &&
	    op != O_POW && op != O_SEL && !(opinfo[op].fl & OF_CMP) &&
	    op != O_LAND && op != O_LOR && op != O_LNOT && op != O_CAST);
}

static void
pr_factors(struct sbuf *b, nid *f, int n)
{
	int k;

	for (k = 0; k < n; k++) {
		if (k)
			sb_putc(b, '*');
		pr(b, f[k], PS_POW);
	}
}

static void
pr_prod(struct sbuf *b, nid e, int prec)
{
	double c = 1;
	nid nu[32], de[32];
	int nn = 0, nd = 0, paren;
	struct cf cn;
	struct sbuf t = {0};

	flat_prod(e, 0, &c, nu, &nn, de, &nd);
	cfmt(c, &cn);
	paren = prec > PS_PROD;
	if (cn.neg)
		sb_putc(&t, '-');
	if (nn == 0) {
		sb_puts(&t, cn.num);
	} else if (!strcmp(cn.num, "1") || cn.unit) {
		pr_factors(&t, nu, nn);
	} else {
		sb_puts(&t, cn.num);
		if (!(isdigit((unsigned char)cn.num[strlen(cn.num) - 1]) &&
		    !cn.sum && starts_name(nu[0])))
			sb_putc(&t, '*');
		pr_factors(&t, nu, nn);
	}
	if (cn.den[0] || nd) {
		sb_putc(&t, '/');
		if (cn.den[0] && nd == 0)
			sb_puts(&t, cn.den);
		else if (!cn.den[0] && nd == 1)
			pr(&t, de[0], PS_POW);
		else {
			sb_putc(&t, '(');
			if (cn.den[0]) {
				sb_puts(&t, cn.den);
				sb_putc(&t, '*');
			}
			pr_factors(&t, de, nd);
			sb_putc(&t, ')');
		}
	}
	if (paren || (cn.neg && prec > PS_SUM))
		sb_printf(b, "(%s)", t.p);
	else
		sb_puts(b, t.p);
	free(t.p);
}

static void
pr(struct sbuf *b, nid e, int prec)
{
	int op = N.op[e], k;
	double v;

	switch (op) {
	case O_CONST: {
		struct cf c;
		num_val(e, &v);
		cfmt(v, &c);
		if ((c.neg && prec > PS_SUM) || ((c.den[0] || c.sum) &&
		    prec > PS_PROD) || (prec >= PS_POW && !c.exact && c.neg))
			sb_printf(b, "(%s)", cstr(v));
		else
			sb_puts(b, cstr(v));
		return;
	}
	case O_VAR:
		sb_puts(b, is_unit(e) || N.a[e] == sym_i ? "i" : sym_str(N.a[e]));
		return;
	case O_ADD:
	case O_SUB:
	case O_NEG: {
		nid t[64];
		int s[64], n = 0;
		flat_sum(e, 1, t, s, &n, 64);
		if (n == 1 && op == O_NEG) {
			if (prec > PS_SUM)
				sb_putc(b, '(');
			sb_putc(b, '-');
			pr(b, t[0], PS_PROD);
			if (prec > PS_SUM)
				sb_putc(b, ')');
			return;
		}
		if (prec > PS_SUM)
			sb_putc(b, '(');
		for (k = 0; k < n; k++) {
			if (k)
				sb_puts(b, s[k] < 0 ? " - " : " + ");
			else if (s[k] < 0)
				sb_putc(b, '-');
			pr(b, t[k], PS_PROD);
		}
		if (prec > PS_SUM)
			sb_putc(b, ')');
		return;
	}
	case O_MUL:
	case O_DIV:
		pr_prod(b, e, prec);
		return;
	case O_POW:
		if (prec > PS_POW)
			sb_putc(b, '(');
		pr(b, N.a[e], PS_ATOM);
		sb_putc(b, '^');
		pr(b, N.b[e], PS_ATOM);
		if (prec > PS_POW)
			sb_putc(b, ')');
		return;
	case O_SEL:
		sb_putc(b, '(');
		pr(b, N.a[e], PS_SUM);
		sb_puts(b, " ? ");
		pr(b, N.b[e], PS_SUM);
		sb_puts(b, " : ");
		pr(b, N.c[e], PS_SUM);
		sb_putc(b, ')');
		return;
	case O_CAST:
		pr(b, N.a[e], prec);
		return;
	case O_LAND:
	case O_LOR:
		if (prec > PS_SUM)
			sb_putc(b, '(');
		pr(b, N.a[e], PS_ATOM);
		sb_puts(b, op == O_LAND ? " && " : " || ");
		pr(b, N.b[e], PS_ATOM);
		if (prec > PS_SUM)
			sb_putc(b, ')');
		return;
	case O_LNOT:
		sb_puts(b, "!");
		pr(b, N.a[e], PS_ATOM);
		return;
	}
	if (opinfo[op].fl & OF_CMP) {
		if (prec > PS_SUM)
			sb_putc(b, '(');
		pr(b, N.a[e], PS_SUM);
		sb_printf(b, " %s ", opinfo[op].name);
		pr(b, N.b[e], PS_SUM);
		if (prec > PS_SUM)
			sb_putc(b, ')');
		return;
	}
	if (op == O_ABS) {
		sb_puts(b, "abs(");
		pr(b, N.a[e], 0);
		sb_putc(b, ')');
		return;
	}
	if (opinfo[op].arity >= 1 && opinfo[op].arity <= 3 && op < O_NOPS &&
	    op != O_TABLE && op != O_INDEX && op != O_LOOP && op != O_ATOMIC) {
		sb_printf(b, "%s(", opinfo[op].name);
		pr(b, N.a[e], 0);
		if (opinfo[op].arity >= 2) {
			sb_puts(b, ", ");
			pr(b, N.b[e], 0);
		}
		if (opinfo[op].arity >= 3) {
			sb_puts(b, ", ");
			pr(b, N.c[e], 0);
		}
		sb_putc(b, ')');
		return;
	}
	sb_puts(b, node_str(e));
}

static const char *
pstr(nid e)
{
	static struct sbuf buf[8];
	static int k;
	struct sbuf *b = &buf[k++ & 7];

	sb_clear(b);
	pr(b, e, 0);
	return sb_str(b);
}

/* the same at a precedence (an operand of * or ^) */
static const char *
pstr_at(nid e, int prec)
{
	static struct sbuf buf[8];
	static int k;
	struct sbuf *b = &buf[k++ & 7];

	sb_clear(b);
	pr(b, e, prec);
	return sb_str(b);
}

/* ---- numeric evaluation in complex arithmetic (the checks) ---- */

typedef double complex cplx;

static cplx *cv;
static uint32_t *cvgen, cvcap, cvg;
static cplx cv_slot[64];
static int cv_real;	/* real semantics where the operands are real */

static int
creal_ok(cplx z)
{
	return cimag(z) == 0;
}

static cplx
cnan(void)
{
	return NAN;
}

static cplx
cipow(cplx z, long n)
{
	cplx r = 1;
	int neg = n < 0;

	if (neg)
		n = -n;
	while (n) {
		if (n & 1)
			r *= z;
		z *= z;
		n >>= 1;
	}
	return neg ? 1 / r : r;
}

static cplx ce(nid e);

/* no negative zero imaginary part */
static cplx
nz(cplx w)
{
	return cimag(w) == 0 ? (cplx)creal(w) : w;
}

static cplx
ce1(nid e)
{
	int op = N.op[e];
	cplx a = 0, b = 0, r;
	double x;
	int re;

	switch (op) {
	case O_CONST:
		num_of(e, &x);
		return x;
	case O_VAR:
		return N.ki[e] < 64 ? cv_slot[N.ki[e]] : cnan();
	case O_UNDEF:
		return cnan();
	case O_SEL:
		a = ce(N.a[e]);
		if (isnan(creal(a)))
			return cnan();
		return creal(a) != 0 ? ce(N.b[e]) : ce(N.c[e]);
	case O_LAND:
		a = ce(N.a[e]);
		return creal(a) != 0 && creal(ce(N.b[e])) != 0;
	case O_LOR:
		a = ce(N.a[e]);
		return creal(a) != 0 || creal(ce(N.b[e])) != 0;
	}
	if (opinfo[op].arity >= 1)
		a = ce(N.a[e]);
	if (opinfo[op].arity >= 2)
		b = ce(N.b[e]);
	if (isnan(creal(a)) || isnan(cimag(a)) || isnan(creal(b)) ||
	    isnan(cimag(b)))
		return cnan();
	re = creal_ok(a) && creal_ok(b);
	x = creal(a);
	switch (op) {
	case O_NEG: return -a;
	case O_ADD: return a + b;
	case O_SUB: return a - b;
	case O_MUL: return a * b;
	case O_DIV:
		return b == 0 ? cnan() : a / b;
	case O_POW:
		if (creal_ok(b) && creal(b) == floor(creal(b)) &&
		    fabs(creal(b)) <= 64) {
			if (a == 0 && creal(b) < 0)
				return cnan();
			return cipow(a, (long)creal(b));
		}
		if (re && cv_real) {
			if (x < 0 || (x == 0 && creal(b) <= 0))
				return cnan();
			return pow(x, creal(b));
		}
		if (a == 0)
			return creal(b) > 0 ? 0 : cnan();
		return cexp(b * clog(a));
	case O_EXP: return cexp(a);
	case O_LOG:
		if (a == 0)
			return cnan();
		if (re && cv_real)
			return x > 0 ? log(x) : cnan();
		return re && x > 0 ? log(x) : clog(a);
	case O_SQRT:
		if (re && cv_real)
			return x >= 0 ? sqrt(x) : cnan();
		return re && x >= 0 ? sqrt(x) : csqrt(a);
	case O_ABS: return cabs(a);
	case O_SIN: return re ? sin(x) : csin(a);
	case O_COS: return re ? cos(x) : ccos(a);
	case O_TAN:
		r = re ? cos(x) : ccos(a);
		if (cabs(r) < 1e-15)
			return cnan();
		return re ? tan(x) : ctan(a);
	case O_SEC:
		r = ccos(a);
		return cabs(r) < 1e-15 ? cnan() : 1 / r;
	case O_CSC:
		r = csin(a);
		return cabs(r) < 1e-15 ? cnan() : 1 / r;
	case O_COT:
		r = csin(a);
		return cabs(r) < 1e-15 ? cnan() : ccos(a) / r;
	case O_ATAN: return re ? atan(x) : catan(a);
	case O_ASIN:
	case O_ACOS:
		if (re && (x >= -1 && x <= 1))
			return op == O_ASIN ? asin(x) : acos(x);
		if (cv_real && re)
			return cnan();
		/* the defining formulas, also on the cuts (C's casin uses the
		 * sign of a zero imaginary part there) */
		if (!re)
			return op == O_ASIN ? casin(a) : cacos(a);
		r = -I * clog(nz(I * a + csqrt(nz(1 - a * a))));
		return op == O_ASIN ? r : M_PI / 2 - r;
	case O_SINH: return re ? sinh(x) : csinh(a);
	case O_COSH: return re ? cosh(x) : ccosh(a);
	case O_TANH: return re ? tanh(x) : ctanh(a);
	case O_ASINH: return re ? asinh(x) : casinh(a);
	case O_ACOSH:
		if (re && x >= 1)
			return acosh(x);
		if (cv_real && re)
			return cnan();
		return re ? clog(nz(a + csqrt(nz(a + 1)) * csqrt(nz(a - 1)))) : cacosh(a);
	case O_ATANH:
		if (re && x > -1 && x < 1)
			return atanh(x);
		if (cv_real && re)
			return cnan();
		return re ? (clog(nz(1 + a)) - clog(nz(1 - a))) / 2 : catanh(a);
	case O_CAST:
	case O_BARRIER:
		return a;
	case O_FMA:
		return a * b + ce(N.c[e]);
	}
	/* real-only operations */
	if (!re || cimag(a) != 0)
		return cnan();
	{
		uint8_t fl = 0;
		double y = scalar_real(op, x, creal(b), &fl);
		if (fl & F_UNDEF)
			return cnan();
		return y;
	}
}

static cplx
ce(nid e)
{
	cplx r;

	if (e < cvcap && cvgen[e] == cvg)
		return cv[e];
	r = ce1(e);
	if (cimag(r) == 0)
		r = creal(r);	/* no -0: the cuts belong to the upper side */
	if (e < cvcap) {
		cvgen[e] = cvg;
		cv[e] = r;
	}
	return r;
}

static cplx
ceval(nid e)
{
	if (cvcap < N.n) {
		uint32_t c = N.n + N.n / 2 + 64;
		cv = xrealloc(cv, c * sizeof *cv);
		cvgen = xrealloc(cvgen, c * sizeof *cvgen);
		memset(cvgen, 0, c * sizeof *cvgen);
		cvcap = c;
		cvg = 0;
	}
	if (++cvg == 0) {
		memset(cvgen, 0, cvcap * sizeof *cvgen);
		cvg = 1;
	}
	return ce(e);
}

/* ---- normalised polynomials over atoms ---- */

#define MF 12
struct term {
	double c;
	int nf;
	nid f[MF];
	int p[MF];
};
struct poly {
	int n, cap;
	struct term *t;
};

static int lc_overflow;		/* a poly got too big: keep the atom */
static char lc_conds[512];	/* x/x cancelled: "x != 0" */

static void
p_free(struct poly *p)
{
	free(p->t);
	memset(p, 0, sizeof *p);
}

static int
t_same(const struct term *a, const struct term *b)
{
	int k;

	if (a->nf != b->nf)
		return 0;
	for (k = 0; k < a->nf; k++)
		if (a->f[k] != b->f[k] || a->p[k] != b->p[k])
			return 0;
	return 1;
}

static void
p_add_term(struct poly *p, const struct term *t, double s)
{
	int k;

	if (t->c * s == 0)
		return;
	for (k = 0; k < p->n; k++)
		if (t_same(&p->t[k], t)) {
			p->t[k].c += t->c * s;
			if (fabs(p->t[k].c) < 1e-12) {
				p->t[k] = p->t[p->n - 1];
				p->n--;
			}
			return;
		}
	if (p->n >= 400) {
		lc_overflow = 1;
		return;
	}
	if (p->n == p->cap) {
		p->cap = p->cap ? 2 * p->cap : 8;
		p->t = xrealloc(p->t, p->cap * sizeof *p->t);
	}
	p->t[p->n] = *t;
	p->t[p->n].c *= s;
	p->n++;
}

static void
p_add(struct poly *p, const struct poly *q, double s)
{
	int k;

	for (k = 0; k < q->n; k++)
		p_add_term(p, &q->t[k], s);
}

static void
p_const(struct poly *p, double c)
{
	struct term t;

	memset(p, 0, sizeof *p);
	memset(&t, 0, sizeof t);
	t.c = c;
	if (fabs(c) >= 1e-14)
		p_add_term(p, &t, 1);
}

static void
p_atom(struct poly *p, nid a, int pw)
{
	struct term t;

	memset(p, 0, sizeof *p);
	memset(&t, 0, sizeof t);
	t.c = 1;
	t.nf = 1;
	t.f[0] = a;
	t.p[0] = pw;
	p_add_term(p, &t, 1);
}

/* never zero: exp, cosh, a positive constant power ... */
static int
nonzero_atom(nid a)
{
	int op = N.op[a];

	return op == O_EXP || op == O_COSH;
}

static const char *
S_cond(nid f)
{
	static char b[160];

	snprintf(b, sizeof b, "%s != 0", pstr(f));
	return b;
}

static int lc_nocond;	/* t_bump: moving factors around, not cancelling */

static void
cond_add(const char *s)
{
	if (lc_nocond)
		return;
	if (strstr(lc_conds, s))
		return;
	if (strlen(lc_conds) + strlen(s) + 8 >= sizeof lc_conds)
		return;
	if (lc_conds[0])
		strcat(lc_conds, ", ");
	strcat(lc_conds, s);
}

static int
t_mul(const struct term *a, const struct term *b, struct term *r)
{
	int i = 0, j = 0;

	memset(r, 0, sizeof *r);
	r->c = a->c * b->c;
	while (i < a->nf || j < b->nf) {
		nid f;
		int p;
		if (j >= b->nf || (i < a->nf && a->f[i] < b->f[j])) {
			f = a->f[i];
			p = a->p[i++];
		} else if (i >= a->nf || b->f[j] < a->f[i]) {
			f = b->f[j];
			p = b->p[j++];
		} else {
			f = a->f[i];
			p = a->p[i] + b->p[j];
			if (p == 0 && (a->p[i] < 0 || b->p[j] < 0) &&
			    !nonzero_atom(f))
				cond_add(S_cond(f));
			i++, j++;
		}
		if (p == 0)
			continue;
		if (r->nf >= MF)
			return 0;
		r->f[r->nf] = f;
		r->p[r->nf++] = p;
	}
	return 1;
}

static void
p_mul(struct poly *r, const struct poly *a, const struct poly *b)
{
	int i, j;
	struct term t;

	memset(r, 0, sizeof *r);
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			if (!t_mul(&a->t[i], &b->t[j], &t)) {
				lc_overflow = 1;
				continue;
			}
			p_add_term(r, &t, 1);
		}
}

static void
p_copy(struct poly *r, const struct poly *a)
{
	memset(r, 0, sizeof *r);
	p_add(r, a, 1);
}

static int
t_deg(const struct term *t)
{
	int k, d = 0;

	for (k = 0; k < t->nf; k++)
		if (t->p[k] > 0)
			d += t->p[k];
	return d;
}

static int
t_cmp(const void *x, const void *y)
{
	const struct term *a = x, *b = y;
	int k;

	if (t_deg(a) != t_deg(b))
		return t_deg(b) - t_deg(a);
	for (k = 0; k < a->nf && k < b->nf; k++) {
		if (a->f[k] != b->f[k])
			return a->f[k] < b->f[k] ? -1 : 1;
		if (a->p[k] != b->p[k])
			return b->p[k] - a->p[k];
	}
	return a->nf - b->nf;
}

static void
p_sort(struct poly *p)
{
	if (p->n > 1)
		qsort(p->t, p->n, sizeof *p->t, t_cmp);
}

static int
p_isconst(const struct poly *p, double *c)
{
	if (p->n == 0) {
		*c = 0;
		return 1;
	}
	if (p->n == 1 && p->t[0].nf == 0) {
		*c = p->t[0].c;
		return 1;
	}
	return 0;
}

/* the exponent of atom f in t (0 if absent) */
static int
t_pow(const struct term *t, nid f)
{
	int k;

	for (k = 0; k < t->nf; k++)
		if (t->f[k] == f)
			return t->p[k];
	return 0;
}

/* t with atom f's exponent changed by d */
static int
t_bump(const struct term *t, nid f, int d, struct term *r)
{
	struct term u;

	memset(&u, 0, sizeof u);
	u.c = 1;
	u.nf = 1;
	u.f[0] = f;
	u.p[0] = d;
	lc_nocond++;
	if (!t_mul(t, &u, r)) {
		lc_nocond--;
		return 0;
	}
	lc_nocond--;
	return 1;
}

/* c M sin^2 u + c M cos^2 u -> c M;  c M cosh^2 u - c M sinh^2 u -> c M */
static void
p_pyth(struct poly *p)
{
	int i, j, again = 1, k;

	while (again) {
		again = 0;
		for (i = 0; i < p->n && !again; i++)
			for (k = 0; k < p->t[i].nf && !again; k++) {
				nid f = p->t[i].f[k], g;
				int op = N.op[f], sg;
				struct term want, rest;
				if (p->t[i].p[k] < 2)
					continue;
				if (op == O_SIN)
					g = mk_un(O_COS, N.a[f]), sg = 1;
				else if (op == O_SINH)
					g = mk_un(O_COSH, N.a[f]), sg = -1;
				else
					continue;
				if (!t_bump(&p->t[i], f, -2, &rest) ||
				    !t_bump(&rest, g, 2, &want))
					continue;
				/* sin: same coefficient; sinh: cosh's is the negation */
				for (j = 0; j < p->n; j++)
					if (j != i && t_same(&p->t[j], &want) &&
					    near(p->t[j].c, sg == 1 ? p->t[i].c :
					    -p->t[i].c))
						break;
				if (j == p->n)
					continue;
				rest.c = p->t[j].c;
				/* remove both, add the rest */
				if (i > j) {
					int x = i;
					i = j;
					j = x;
				}
				p->t[j] = p->t[p->n - 1];
				p->n--;
				p->t[i] = p->t[p->n - 1];
				p->n--;
				p_add_term(p, &rest, 1);
				again = 1;
			}
	}
}

/* memo: the normal form of each node seen in this command */
static struct poly *lmemo;
static uint8_t *lmset;
static uint32_t lmcap;

static void
lc_reset(void)
{
	uint32_t k;

	for (k = 0; k < lmcap; k++)
		if (lmset[k])
			p_free(&lmemo[k]);
	free(lmemo);
	free(lmset);
	lmemo = NULL;
	lmset = NULL;
	lmcap = 0;
}

static void lc_of(nid e, struct poly *r);
static nid lc_build(struct poly *p);

static nid
lc_node(nid e)
{
	struct poly p;
	nid r;

	lc_of(e, &p);
	r = lc_build(&p);
	p_free(&p);
	return r;
}

static int odd_fn(int op)
{
	return op == O_SIN || op == O_TAN || op == O_CSC || op == O_COT ||
	    op == O_SINH || op == O_TANH || op == O_ASIN || op == O_ATAN ||
	    op == O_ASINH || op == O_ATANH;
}

static int even_fn(int op)
{
	return op == O_COS || op == O_SEC || op == O_COSH;
}

/* known to be >= 0 */
static int
nonneg_atom(nid f)
{
	int op = N.op[f];

	return op == O_EXP || op == O_SQRT || op == O_ABS || op == O_COSH;
}

static void
lc_atom(nid e, struct poly *r)
{
	int op = N.op[e], k;
	struct poly a, b;
	double c;
	nid x, y;

	if (op == O_CONST || op == O_VAR || op == O_UNDEF ||
	    opinfo[op].arity == 0 || op == O_TABLE || op == O_INDEX ||
	    op == O_LOOP || op == O_ATOMIC || op == O_BARRIER) {
		p_atom(r, e, 1);
		return;
	}
	if (opinfo[op].arity == 1 && op != O_CAST) {
		lc_of(N.a[e], &a);
		if (p_isconst(&a, &c)) {
			x = mk_un(op, mk_const_r(c));
			p_free(&a);
			if (num_of(x, &c)) {
				p_const(r, c);
				return;
			}
			p_atom(r, x, 1);
			return;
		}
		p_sort(&a);
		if ((odd_fn(op) || even_fn(op)) && a.t[0].c < 0) {
			struct poly m;
			p_copy(&m, &a);
			for (k = 0; k < m.n; k++)
				m.t[k].c = -m.t[k].c;
			x = mk_un(op, lc_build(&m));
			p_free(&m);
			p_free(&a);
			p_atom(r, x, 1);
			if (odd_fn(op))
				r->t[0].c = -1;
			return;
		}
		if (op == O_EXP) {
			/* exp(a + b) = exp(a) exp(b): e^x e^-x cancels */
			memset(r, 0, sizeof *r);
			p_const(r, 1);
			for (k = 0; k < a.n; k++) {
				struct poly t, f, m;
				memset(&t, 0, sizeof t);
				p_add_term(&t, &a.t[k], 1);
				if (t.n == 1 && t.t[0].nf == 0)
					x = mk_un(O_EXP, mk_const_r(t.t[0].c));
				else
					x = mk_un(O_EXP, lc_build(&t));
				if (num_of(x, &c))
					p_const(&f, c);
				else
					p_atom(&f, x, 1);
				p_mul(&m, r, &f);
				p_free(r);
				*r = m;
				p_free(&t);
				p_free(&f);
			}
			p_free(&a);
			return;
		}
		if (op == O_LOG && a.n == 1 && a.t[0].c == 1 && a.t[0].nf == 1 &&
		    a.t[0].p[0] == 1 && N.op[a.t[0].f[0]] == O_EXP) {
			lc_of(N.a[a.t[0].f[0]], r);
			p_free(&a);
			return;
		}
		if ((op == O_SQRT || op == O_ABS) && a.n == 1) {
			/* sqrt(c x^2 y^4) = sqrt(c) |x| y^2 */
			struct term t = a.t[0];
			int ok = op == O_ABS || t.c > 0;
			for (k = 0; k < t.nf && op == O_SQRT; k++)
				if (t.p[k] % 2)
					ok = 0;
			if (ok) {
				struct term u;
				memset(&u, 0, sizeof u);
				u.c = op == O_SQRT ? sqrt(t.c) : fabs(t.c);
				memset(r, 0, sizeof *r);
				for (k = 0; k < t.nf; k++) {
					int pw = op == O_SQRT ? t.p[k] / 2 : t.p[k];
					nid f = t.f[k];
					if ((pw % 2) && !nonneg_atom(f))
						f = mk_un(O_ABS, f);
					u.f[u.nf] = f;
					u.p[u.nf++] = pw;
				}
				/* the factors may be out of order now */
				{
					struct poly one, prod;
					struct term v;
					p_const(&one, u.c);
					for (k = 0; k < u.nf; k++) {
						struct poly at;
						p_atom(&at, u.f[k], u.p[k]);
						p_mul(&prod, &one, &at);
						p_free(&one);
						p_free(&at);
						one = prod;
					}
					(void)v;
					*r = one;
				}
				p_free(&a);
				return;
			}
		}
		x = lc_build(&a);
		p_free(&a);
		p_atom(r, op == O_ABS && nonneg_atom(x) ? x : mk_un(op, x), 1);
		return;
	}
	if (op == O_CAST) {
		lc_of(N.a[e], r);
		return;
	}
	if (op == O_SEL) {
		p_atom(r, mk_sel(lc_node(N.a[e]), lc_node(N.b[e]),
		    lc_node(N.c[e])), 1);
		return;
	}
	if (op == O_FMA) {
		p_atom(r, mk_fma(lc_node(N.a[e]), lc_node(N.b[e]),
		    lc_node(N.c[e])), 1);
		return;
	}
	x = lc_node(N.a[e]);
	y = lc_node(N.b[e]);
	x = mk_bin(op, x, y);
	if (num_of(x, &c)) {
		p_const(r, c);
		return;
	}
	(void)b;
	p_atom(r, x, 1);
}

static void
lc_of1(nid e, struct poly *r)
{
	struct poly a, b, t;
	double c;
	int64_t n;
	int k;

	switch (N.op[e]) {
	case O_CONST:
		num_of(e, &c);
		p_const(r, c);
		return;
	case O_NEG:
		lc_of(N.a[e], &a);
		memset(r, 0, sizeof *r);
		p_add(r, &a, -1);
		p_free(&a);
		return;
	case O_ADD:
	case O_SUB:
		lc_of(N.a[e], &a);
		lc_of(N.b[e], &b);
		memset(r, 0, sizeof *r);
		p_add(r, &a, 1);
		p_add(r, &b, N.op[e] == O_SUB ? -1 : 1);
		p_free(&a);
		p_free(&b);
		p_pyth(r);
		return;
	case O_MUL:
		lc_of(N.a[e], &a);
		lc_of(N.b[e], &b);
		p_mul(r, &a, &b);
		p_free(&a);
		p_free(&b);
		p_pyth(r);
		return;
	case O_DIV:
		lc_of(N.a[e], &a);
		lc_of(N.b[e], &b);
		if (p_isconst(&b, &c) && c != 0) {
			memset(r, 0, sizeof *r);
			p_add(r, &a, 1 / c);
		} else if (b.n == 1) {
			struct poly inv;
			memset(&inv, 0, sizeof inv);
			t = b;
			p_copy(&inv, &b);
			inv.t[0].c = 1 / inv.t[0].c;
			for (k = 0; k < inv.t[0].nf; k++)
				inv.t[0].p[k] = -inv.t[0].p[k];
			p_mul(r, &a, &inv);
			p_free(&inv);
		} else {
			struct poly d;
			p_sort(&b);
			/* signs: fewer minus signs; on a tie a positive leading
			 * coefficient in the denominator */
			int neg = 0, tot = a.n + b.n;
			for (k = 0; k < a.n; k++)
				neg += a.t[k].c < 0;
			for (k = 0; k < b.n; k++)
				neg += b.t[k].c < 0;
			if (2 * neg > tot || (2 * neg == tot && b.t[0].c < 0)) {
				for (k = 0; k < b.n; k++)
					b.t[k].c = -b.t[k].c;
				for (k = 0; k < a.n; k++)
					a.t[k].c = -a.t[k].c;
			}
			p_atom(&d, lc_build(&b), -1);
			p_mul(r, &a, &d);
			p_free(&d);
		}
		p_free(&a);
		p_free(&b);
		p_pyth(r);
		return;
	case O_POW:
		if (int_of(N.b[e], &n) && llabs(n) <= 12) {
			lc_of(N.a[e], &a);
			if (a.n == 1 || (n >= 0 && n <= 6 && a.n <= 6)) {
				struct poly acc;
				int m = (int)llabs(n);
				if (n < 0) {
					a.t[0].c = 1 / a.t[0].c;
					for (k = 0; k < a.t[0].nf; k++)
						a.t[0].p[k] = -a.t[0].p[k];
				}
				p_const(&acc, 1);
				for (k = 0; k < m; k++) {
					p_mul(&t, &acc, &a);
					p_free(&acc);
					acc = t;
				}
				*r = acc;
				p_free(&a);
				p_pyth(r);
				return;
			}
			p_atom(r, lc_build(&a), (int)n);
			p_free(&a);
			return;
		}
		p_atom(r, mk_bin(O_POW, lc_node(N.a[e]), lc_node(N.b[e])), 1);
		return;
	}
	lc_atom(e, r);
}

static void
lc_of(nid e, struct poly *r)
{
	if (e < lmcap && lmset[e]) {
		p_copy(r, &lmemo[e]);
		return;
	}
	lc_of1(e, r);
	if (lmcap < N.n) {
		uint32_t c = N.n + N.n / 2 + 64;
		lmemo = xrealloc(lmemo, c * sizeof *lmemo);
		lmset = xrealloc(lmset, c);
		memset(lmset + lmcap, 0, c - lmcap);
		lmcap = c;
	}
	if (e < lmcap) {
		p_copy(&lmemo[e], r);
		lmset[e] = 1;
	}
}

/* ---- back to nodes ---- */

static nid
t_factors(const struct term *t, int sign)
{
	nid r = 0, f;
	int k;

	for (k = 0; k < t->nf; k++) {
		if ((t->p[k] > 0) != (sign > 0))
			continue;
		f = t->f[k];
		if (abs(t->p[k]) != 1)
			f = mk_bin(O_POW, f, K(abs(t->p[k])));
		r = r ? mk_bin(O_MUL, r, f) : f;
	}
	return r;
}

static nid
num_node(double c)
{
	if (c == floor(c) && fabs(c) < 1e15)
		return K((int64_t)c);
	return mk_const_r(c);
}

/* sum of terms, positive factors only, all with the same denominator */
static nid
build_plain(struct term *t, int n)
{
	int64_t L = 1, qn, qd;
	int k, ok = n > 1, cfirst = 0, nneg = 0;
	nid r = 0;

	for (k = 0; k < n && ok; k++) {
		if (!small_q(fabs(t[k].c), 1000, 1000000000, &qn, &qd)) {
			ok = 0;
			break;
		}
		L = L / gcd64(L, qd) * qd;
		if (L > 1000)
			ok = 0;
	}
	if (!ok)
		L = 1;
	/* the constant first when every other term is negative: 1 - cos(2x) */
	for (k = 0; k < n; k++)
		if (t_factors(&t[k], 1) && t[k].c < 0)
			nneg++;
	if (n > 1 && t_factors(&t[n - 1], 1) == 0 && t[n - 1].c > 0 &&
	    nneg == n - 1)
		cfirst = 1;
	for (k = 0; k < n; k++) {
		int j = cfirst ? (k == 0 ? n - 1 : k - 1) : k;
		double c = t[j].c * (double)L;
		nid f = t_factors(&t[j], 1), x;
		double a = fabs(c);
		if (f == 0)
			x = num_node(a);
		else if (near(a, 1))
			x = f;
		else
			x = mk_bin(O_MUL, num_node(a), f);
		if (r == 0)
			r = c < 0 ? mk_un(O_NEG, x) : x;
		else
			r = mk_bin(c < 0 ? O_SUB : O_ADD, r, x);
	}
	if (r == 0)
		r = K(0);
	if (L > 1)
		r = mk_bin(O_DIV, r, K(L));
	return r;
}

static nid
lc_build(struct poly *p)
{
	int k, j, ng = 0, *g;
	nid r = 0;
	struct term *tt;

	if (p->n == 0)
		return K(0);
	p_sort(p);
	g = xcalloc(p->n, sizeof *g);
	tt = xmalloc(p->n * sizeof *tt);
	for (k = 0; k < p->n; k++)
		g[k] = -1;
	/* group the terms by their denominator */
	for (k = 0; k < p->n; k++) {
		nid dk, num, x;
		int n = 0, neg = 0;
		if (g[k] >= 0)
			continue;
		dk = t_factors(&p->t[k], -1);
		for (j = k; j < p->n; j++)
			if (g[j] < 0 && t_factors(&p->t[j], -1) == dk) {
				g[j] = ng;
				tt[n++] = p->t[j];
			}
		ng++;
		num = build_plain(tt, n);
		if (dk) {
			/* -a/d: the sign in front */
			if (N.op[num] == O_NEG) {
				num = N.a[num];
				neg = 1;
			}
			x = mk_bin(O_DIV, num, dk);
		} else
			x = num;
		if (r == 0)
			r = neg ? mk_un(O_NEG, x) : x;
		else
			r = mk_bin(neg ? O_SUB : O_ADD, r, x);
	}
	free(g);
	free(tt);
	return r;
}

static nid
norm(nid e)
{
	return lc_node(e);
}

/* ---- the complex layer: an expression as a pair of real nodes ---- */

struct cz {
	nid re, im;
};

static int
zero(nid e)
{
	return is_const(e) && const_is(e, 0);
}

static int
one(nid e)
{
	return is_const(e) && const_is(e, 1);
}

static nid
nadd(nid a, nid b)
{
	if (zero(a))
		return b;
	if (zero(b))
		return a;
	return mk_bin(O_ADD, a, b);
}

static nid
nsub(nid a, nid b)
{
	if (zero(b))
		return a;
	if (zero(a))
		return mk_un(O_NEG, b);
	return mk_bin(O_SUB, a, b);
}

static nid
nmul(nid a, nid b)
{
	if (zero(a) || zero(b))
		return K(0);
	if (one(a))
		return b;
	if (one(b))
		return a;
	return mk_bin(O_MUL, a, b);
}

static nid
ndiv(nid a, nid b)
{
	if (zero(a))
		return K(0);
	if (one(b))
		return a;
	return mk_bin(O_DIV, a, b);
}

static nid
nneg(nid a)
{
	return zero(a) ? a : mk_un(O_NEG, a);
}

static nid
nsq(nid a)
{
	return mk_bin(O_POW, a, K(2));
}

static struct cz
Z(nid re, nid im)
{
	struct cz z;

	z.re = re;
	z.im = im;
	return z;
}

static struct cz
zadd(struct cz a, struct cz b)
{
	return Z(nadd(a.re, b.re), nadd(a.im, b.im));
}

static struct cz
zsub(struct cz a, struct cz b)
{
	return Z(nsub(a.re, b.re), nsub(a.im, b.im));
}

/* sg = -1: i^2 = -1 (complex); sg = 1: j^2 = 1 (split-complex, cosh/sinh) */
static struct cz
zmul_s(struct cz a, struct cz b, int sg)
{
	nid bd = nmul(a.im, b.im);
	return Z(sg < 0 ? nsub(nmul(a.re, b.re), bd) : nadd(nmul(a.re, b.re), bd),
	    nadd(nmul(a.re, b.im), nmul(a.im, b.re)));
}

static struct cz
zmul(struct cz a, struct cz b)
{
	return zmul_s(a, b, -1);
}

static struct cz
zdiv(struct cz a, struct cz b)
{
	nid d;

	if (zero(b.im))
		return Z(ndiv(a.re, b.re), ndiv(a.im, b.re));
	/* a conj(b) / |b|^2 */
	d = norm(nadd(nsq(b.re), nsq(b.im)));
	return Z(ndiv(nadd(nmul(a.re, b.re), nmul(a.im, b.im)), d),
	    ndiv(nsub(nmul(a.im, b.re), nmul(a.re, b.im)), d));
}

static struct cz
zipow(struct cz z, int64_t n, int sg)
{
	struct cz r = Z(K(1), K(0));
	int neg = n < 0;

	if (neg)
		n = -n;
	while (n) {
		if (n & 1)
			r = zmul_s(r, z, sg);
		n >>= 1;
		if (n)
			z = zmul_s(z, z, sg);
	}
	if (neg)
		r = zdiv(Z(K(1), K(0)), r);
	return r;
}

static int
known_pos(nid e)
{
	double v;

	if (num_of(e, &v))
		return v > 0;
	return N.op[e] == O_EXP || N.op[e] == O_COSH;
}

static int
known_neg(nid e)
{
	double v;

	return num_of(e, &v) && v < 0;
}

/* |z|, normalised */
static nid
zabs(struct cz z)
{
	if (zero(z.im))
		return norm(mk_un(O_ABS, z.re));
	return norm(mk_un(O_SQRT, nadd(nsq(z.re), nsq(z.im))));
}

/* principal argument; the conditions go to cond (may be NULL) */
static nid
zarg(struct cz z)
{
	if (zero(z.im)) {
		if (known_pos(z.re))
			return K(0);
		if (known_neg(z.re))
			return mk_const_r(M_PI);
	}
	if (zero(z.re)) {
		if (known_pos(z.im))
			return mk_const_r(M_PI / 2);
		if (known_neg(z.im))
			return mk_const_r(-M_PI / 2);
	}
	return mk_bin(O_ATAN2, z.im, z.re);
}

static struct cz
zexp(struct cz z)
{
	nid r = mk_un(O_EXP, z.re);

	if (zero(z.im))
		return Z(r, K(0));
	return Z(nmul(r, mk_un(O_COS, z.im)), nmul(r, mk_un(O_SIN, z.im)));
}

static struct cz
zlog(struct cz z)
{
	if (zero(z.im)) {
		if (known_pos(z.re))
			return Z(mk_un(O_LOG, z.re), K(0));
		if (known_neg(z.re))
			return Z(mk_un(O_LOG, nneg(z.re)), mk_const_r(M_PI));
		return Z(mk_un(O_LOG, mk_un(O_ABS, z.re)),
		    mk_sel(mk_bin(O_LT, z.re, K(0)), mk_const_r(M_PI), K(0)));
	}
	return Z(ndiv(mk_un(O_LOG, nadd(nsq(z.re), nsq(z.im))), K(2)), zarg(z));
}

static struct cz
zsqrt(struct cz z)
{
	nid m, re, im;

	if (zero(z.im)) {
		if (known_pos(z.re) || zero(z.re))
			return Z(mk_un(O_SQRT, z.re), K(0));
		if (known_neg(z.re))
			return Z(K(0), mk_un(O_SQRT, nneg(z.re)));
		return Z(mk_sel(mk_bin(O_GE, z.re, K(0)), mk_un(O_SQRT, z.re), K(0)),
		    mk_sel(mk_bin(O_LT, z.re, K(0)), mk_un(O_SQRT, nneg(z.re)),
		    K(0)));
	}
	m = zabs(z);
	re = norm(mk_un(O_SQRT, ndiv(nadd(m, z.re), K(2))));
	im = norm(mk_un(O_SQRT, ndiv(nsub(m, z.re), K(2))));
	if (known_neg(z.im))
		im = nneg(im);
	else if (!known_pos(z.im))
		im = mk_bin(O_COPYSIGN, im, z.im);
	return Z(re, im);
}

static struct cz zof(nid e);

static struct cz
zpow(struct cz a, struct cz b)
{
	int64_t n;

	if (zero(b.im) && int_of(b.re, &n) && llabs(n) <= 64)
		return zipow(a, n, -1);
	if (zero(a.im) && zero(b.im))
		return Z(mk_bin(O_POW, a.re, b.re), K(0));
	if (zero(a.im) && is_const(a.re) && const_is(a.re, M_E))
		return zexp(b);
	if (zero(b.im) && is_const(b.re) && const_is(b.re, 0.5))
		return zsqrt(a);
	/* principal: exp(b log a) */
	if (zero(a.im) && known_pos(a.re))
		return zexp(Z(nmul(b.re, mk_un(O_LOG, a.re)),
		    nmul(b.im, mk_un(O_LOG, a.re))));
	return zexp(zmul(b, zlog(a)));
}

static const struct cz ZI = {0, 0};

static struct cz
zi(void)
{
	return Z(K(0), K(1));
}

static struct cz
zre(nid r)
{
	return Z(r, K(0));
}

/* sin(a + bi) = sin a cosh b + i cos a sinh b ... */
static struct cz
ztrig(int op, struct cz z)
{
	nid a = z.re, b = z.im;
	struct cz s, c;

	if (zero(b) && op != O_ASIN && op != O_ACOS && op != O_ACOSH &&
	    op != O_ATANH)
		return zre(mk_un(op, a));
	switch (op) {
	case O_SIN:
		return Z(nmul(mk_un(O_SIN, a), mk_un(O_COSH, b)),
		    nmul(mk_un(O_COS, a), mk_un(O_SINH, b)));
	case O_COS:
		return Z(nmul(mk_un(O_COS, a), mk_un(O_COSH, b)),
		    nneg(nmul(mk_un(O_SIN, a), mk_un(O_SINH, b))));
	case O_SINH:
		return Z(nmul(mk_un(O_SINH, a), mk_un(O_COS, b)),
		    nmul(mk_un(O_COSH, a), mk_un(O_SIN, b)));
	case O_COSH:
		return Z(nmul(mk_un(O_COSH, a), mk_un(O_COS, b)),
		    nmul(mk_un(O_SINH, a), mk_un(O_SIN, b)));
	case O_TAN: case O_SEC: case O_CSC: case O_COT:
		s = ztrig(O_SIN, z);
		c = ztrig(O_COS, z);
		if (op == O_TAN)
			return zdiv(s, c);
		if (op == O_COT)
			return zdiv(c, s);
		return zdiv(zre(K(1)), op == O_SEC ? c : s);
	case O_TANH:
		return zdiv(ztrig(O_SINH, z), ztrig(O_COSH, z));
	}
	return ZI;
}

/* the inverse functions through log and sqrt (principal values) */
static struct cz
zinv(int op, struct cz z)
{
	struct cz one_ = zre(K(1)), i = zi(), r;
	double v;

	/* a real argument inside the real domain stays a real function */
	if (zero(z.im)) {
		int inside = op == O_ATAN || op == O_ASINH;
		if (num_of(z.re, &v))
			inside = op == O_ATAN || op == O_ASINH ||
			    ((op == O_ASIN || op == O_ACOS) && fabs(v) <= 1) ||
			    (op == O_ACOSH && v >= 1) ||
			    (op == O_ATANH && fabs(v) < 1);
		else if (op != O_ATAN && op != O_ASINH) {
			note("%s(%s) is taken as the real function: %s", opinfo[op].name,
			    pstr(z.re), forms_cond(mk_un(op, z.re)));
			inside = 1;
		}
		if (inside)
			return zre(mk_un(op, z.re));
		note("%s(%s) is on the branch cut; the value is that of %s",
		    opinfo[op].name, pstr(z.re), op == O_ASIN ?
		    "-i log(iz + sqrt(1 - z^2))" : op == O_ACOS ?
		    "pi/2 - asin(z) = pi/2 + i log(iz + sqrt(1 - z^2))" :
		    op == O_ACOSH ? "log(z + sqrt(z + 1) sqrt(z - 1))" :
		    "(log(1 + z) - log(1 - z))/2");
	}
	switch (op) {
	case O_ASIN:	/* -i log(iz + sqrt(1 - z^2)) */
		r = zlog(zadd(zmul(i, z), zsqrt(zsub(one_, zmul(z, z)))));
		return Z(r.im, nneg(r.re));
	case O_ACOS:	/* pi/2 - asin z */
		r = zinv(O_ASIN, z);
		return Z(nsub(mk_const_r(M_PI / 2), r.re), nneg(r.im));
	case O_ATAN:	/* (i/2)(log(1 - iz) - log(1 + iz)) */
		r = zsub(zlog(zsub(one_, zmul(i, z))), zlog(zadd(one_,
		    zmul(i, z))));
		return Z(ndiv(nneg(r.im), K(2)), ndiv(r.re, K(2)));
	case O_ASINH:	/* log(z + sqrt(z^2 + 1)) */
		return zlog(zadd(z, zsqrt(zadd(zmul(z, z), one_))));
	case O_ACOSH:	/* log(z + sqrt(z + 1) sqrt(z - 1)) */
		return zlog(zadd(z, zmul(zsqrt(zadd(z, one_)),
		    zsqrt(zsub(z, one_)))));
	case O_ATANH:	/* (log(1 + z) - log(1 - z))/2 */
		r = zsub(zlog(zadd(one_, z)), zlog(zsub(one_, z)));
		return Z(ndiv(r.re, K(2)), ndiv(r.im, K(2)));
	}
	return ZI;
}

static struct cz *zmemo;
static uint8_t *zmset;
static uint32_t zmcap;

static void
z_reset(void)
{
	free(zmemo);
	free(zmset);
	zmemo = NULL;
	zmset = NULL;
	zmcap = 0;
}

static struct cz
zof1(nid e)
{
	int op = N.op[e];
	struct cz a, b;

	switch (op) {
	case O_CONST:
		return zre(e);
	case O_VAR:
		return is_unit(e) ? zi() : zre(e);
	case O_UNDEF:
		fail("the expression is undefined");
	case O_NEG:
		a = zof(N.a[e]);
		return Z(nneg(a.re), nneg(a.im));
	case O_ADD:
		return zadd(zof(N.a[e]), zof(N.b[e]));
	case O_SUB:
		return zsub(zof(N.a[e]), zof(N.b[e]));
	case O_MUL:
		return zmul(zof(N.a[e]), zof(N.b[e]));
	case O_DIV:
		return zdiv(zof(N.a[e]), zof(N.b[e]));
	case O_POW:
		return zpow(zof(N.a[e]), zof(N.b[e]));
	case O_EXP:
		return zexp(zof(N.a[e]));
	case O_LOG:
		return zlog(zof(N.a[e]));
	case O_SQRT:
		return zsqrt(zof(N.a[e]));
	case O_ABS:
		return zre(zabs(zof(N.a[e])));
	case O_SIN: case O_COS: case O_TAN: case O_SEC: case O_CSC:
	case O_COT: case O_SINH: case O_COSH: case O_TANH:
		return ztrig(op, zof(N.a[e]));
	case O_ASIN: case O_ACOS: case O_ATAN: case O_ASINH: case O_ACOSH:
	case O_ATANH:
		return zinv(op, zof(N.a[e]));
	case O_CAST:
		return zof(N.a[e]);
	case O_SEL:
		a = zof(N.b[e]);
		b = zof(N.c[e]);
		{
			struct cz c = zof(N.a[e]);
			if (!zero(c.im))
				fail("a condition must be real");
			return Z(mk_sel(c.re, a.re, b.re), mk_sel(c.re, a.im, b.im));
		}
	}
	/* real-only operations: every operand must be real */
	if (opinfo[op].arity >= 1) {
		a = zof(N.a[e]);
		b = opinfo[op].arity >= 2 ? zof(N.b[e]) : zre(K(0));
		if (!zero(a.im) || !zero(b.im))
			fail("%s of a complex value is not defined here (it needs "
			    "real operands)", opinfo[op].name);
		if (opinfo[op].arity == 1)
			return zre(mk_un(op, a.re));
		if (opinfo[op].arity == 2)
			return zre(mk_bin(op, a.re, b.re));
	}
	fail("%s is not supported by the complex forms", opinfo[op].name);
	return ZI;
}

static struct cz
zof(nid e)
{
	struct cz r;

	if (e < zmcap && zmset[e])
		return zmemo[e];
	r = zof1(e);
	if (zmcap < N.n) {
		uint32_t c = N.n + N.n / 2 + 64;
		zmemo = xrealloc(zmemo, c * sizeof *zmemo);
		zmset = xrealloc(zmset, c);
		memset(zmset + zmcap, 0, c - zmcap);
		zmcap = c;
	}
	if (e < zmcap) {
		zmemo[e] = r;
		zmset[e] = 1;
	}
	return r;
}

/* display-only constant expressions: atan(2) stays atan(2) */
static nid
raw1(int op, nid a)
{
	return mk_raw(op, TY_REAL, a, 0, 0, 0, 0);
}

static nid
raw2(int op, nid a, nid b)
{
	return mk_raw(op, TY_REAL, a, b, 0, 0, 0);
}

/* ---- the input: text with i, sin^2 x, cis, 45°, r∠theta ---- */

static const char *const trig_names[] = {"sin", "cos", "tan", "sec", "csc",
	"cot", "sinh", "cosh", "tanh", "asin", "acos", "atan", "arcsin",
	"arccos", "arctan", "asinh", "acosh", "atanh", NULL};

static int
in_names(const char *s, size_t n, const char *const *l)
{
	int k;

	for (k = 0; l[k]; k++)
		if (strlen(l[k]) == n && !strncmp(l[k], s, n))
			return 1;
	return 0;
}

static int
ident_ch(int c)
{
	return isalnum((unsigned char)c) || c == '_';
}

/* the end of a balanced (...) starting at s[i] == '(' */
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

/* a juxtaposed operand: (...), or a run like 2x, x, 3.5, pi */
static size_t
operand_end(const char *s, size_t i)
{
	if (s[i] == '(')
		return paren_end(s, i);
	if (s[i] == '-' || s[i] == '+')
		i++;
	while (s[i] && (ident_ch(s[i]) || s[i] == '.'))
		i++;
	if (s[i] == '(')	/* a call: f(x) */
		return paren_end(s, i);
	return i;
}

static const char *unit_name = "i";
static int saw_j, saw_unit;

static void
prep(const char *in, struct sbuf *o)
{
	size_t i = 0, j, n;

	while (in[i]) {
		unsigned char c = in[i];
		/* degrees: 45°, 45 deg */
		if (c == 0xc2 && (unsigned char)in[i + 1] == 0xb0) {
			sb_puts(o, "*(pi/180)");
			i += 2;
			continue;
		}
		if (c == 0xe2 && (unsigned char)in[i + 1] == 0x88 &&
		    (unsigned char)in[i + 2] == 0xa0) {
			/* r∠theta: r*exp(i*theta) */
			struct sbuf t = {0};
			char *arg;
			i += 3;
			while (in[i] == ' ')
				i++;
			j = operand_end(in, i);
			arg = xmalloc(j - i + 1);
			memcpy(arg, in + i, j - i);
			arg[j - i] = 0;
			i = j;
			/* a trailing ° or deg belongs to the angle */
			while (in[i] == ' ')
				i++;
			prep(arg, &t);
			if ((unsigned char)in[i] == 0xc2 &&
			    (unsigned char)in[i + 1] == 0xb0) {
				sb_puts(&t, "*(pi/180)");
				i += 2;
			} else if (!strncmp(in + i, "deg", 3) && !ident_ch(in[i + 3])) {
				sb_puts(&t, "*(pi/180)");
				i += 3;
			} else if (!strncmp(in + i, "degrees", 7)) {
				sb_puts(&t, "*(pi/180)");
				i += 7;
			}
			sb_printf(o, "*exp(%s*(%s))", unit_name, sb_str(&t));
			free(t.p);
			free(arg);
			saw_unit = 1;
			continue;
		}
		if (!(isalpha(c) || c == '_')) {
			/* a number: copy it with its exponent (2e5) */
			if (isdigit(c) || (c == '.' && isdigit((unsigned char)in[i + 1]))) {
				while (isdigit((unsigned char)in[i]) || in[i] == '.')
					sb_putc(o, in[i++]);
				if ((in[i] == 'e' || in[i] == 'E') &&
				    (isdigit((unsigned char)in[i + 1]) ||
				    ((in[i + 1] == '-' || in[i + 1] == '+') &&
				    isdigit((unsigned char)in[i + 2])))) {
					sb_putc(o, in[i++]);
					sb_putc(o, in[i++]);
					while (isdigit((unsigned char)in[i]))
						sb_putc(o, in[i++]);
				}
				/* 45 deg */
				j = i;
				while (in[j] == ' ')
					j++;
				if ((!strncmp(in + j, "deg", 3) && !ident_ch(in[j + 3])) ||
				    (!strncmp(in + j, "degrees", 7) &&
				    !ident_ch(in[j + 7]))) {
					sb_puts(o, "*(pi/180)");
					i = j + (in[j + 3] == 'r' ? 7 : 3);
				}
				continue;
			}
			sb_putc(o, in[i++]);
			continue;
		}
		for (j = i; ident_ch(in[j]); j++)
			;
		n = j - i;
		if ((n == 1 && (in[i] == 'i' || in[i] == 'j' || in[i] == 'I')) ||
		    (n == 2 && !strncmp(in + i, "im", 2))) {
			if (in[i] == 'j')
				saw_j = 1;
			saw_unit = 1;
			sb_puts(o, unit_name);
			i = j;
			continue;
		}
		if (n == 3 && !strncmp(in + i, "cis", 3)) {
			size_t k = j, e;
			while (in[k] == ' ')
				k++;
			e = operand_end(in, k);
			if (e > k) {
				struct sbuf t = {0};
				char *arg = xmalloc(e - k + 1);
				memcpy(arg, in + k, e - k);
				arg[e - k] = 0;
				prep(arg, &t);
				sb_printf(o, "exp(%s*(%s))", unit_name, sb_str(&t));
				free(t.p);
				free(arg);
				saw_unit = 1;
				i = e;
				continue;
			}
		}
		/* sin^2 x, sin²(x), sin^-1 x */
		if (in_names(in + i, n, trig_names)) {
			size_t k = j;
			int pw = 0, neg = 0, ok = 0;
			if (in[k] == '^') {
				k++;
				if (in[k] == '(' && in[k + 1] == '-') {
					neg = 1;
					k += 2;
				} else if (in[k] == '-') {
					neg = 1;
					k++;
				}
				while (isdigit((unsigned char)in[k]))
					pw = pw * 10 + in[k++] - '0', ok = 1;
				if (neg && in[k] == ')')
					k++;
			} else if ((unsigned char)in[k] == 0xc2 &&
			    ((unsigned char)in[k + 1] == 0xb2 ||
			    (unsigned char)in[k + 1] == 0xb3)) {
				pw = (unsigned char)in[k + 1] == 0xb2 ? 2 : 3;
				k += 2;
				ok = 1;
			}
			if (ok && pw > 0) {
				size_t a = k, e;
				struct sbuf t = {0};
				char *arg;
				while (in[a] == ' ')
					a++;
				e = operand_end(in, a);
				if (e > a) {
					arg = xmalloc(e - a + 1);
					memcpy(arg, in + a, e - a);
					arg[e - a] = 0;
					prep(arg, &t);
					free(arg);
					if (neg && pw == 1) {
						if (in[i] == 'a')
							fail("%.*s^-1 is ambiguous", (int)n, in + i);
						note("%.*s^-1 read as the inverse function a%.*s, "
						    "not 1/%.*s", (int)n, in + i, (int)n, in + i,
						    (int)n, in + i);
						sb_printf(o, "a%.*s(%s)", (int)n, in + i,
						    sb_str(&t));
					} else
						sb_printf(o, "%.*s(%s)^%s%d", (int)n, in + i,
						    sb_str(&t), neg ? "-" : "", pw);
					free(t.p);
					i = e;
					continue;
				}
				free(t.p);
			}
		}
		sb_printf(o, "%.*s", (int)n, in + i);
		i = j;
	}
}

/* the function to work on: a name, or an expression in text */
struct inp {
	struct func fn;
	nid body;
	char text[1024];
	int nreal;		/* real parameters */
};

static void
mark_units(struct func *f)
{
	int k;

	unit_slots = 0;
	for (k = 0; k < f->np; k++)
		if (!strcmp(sym_str(f->p[k].name), unit_name) ||
		    (f->p[k].name == sym_i))
			unit_slots |= 1ull << k;
}

static void
get_input(const char *text, struct inp *in, int trig)
{
	struct sbuf t = {0};
	struct func *g;
	size_t n = strlen(text);
	int k;

	memset(in, 0, sizeof *in);
	sym_i = sym_cstr("i");
	unit_name = "i";
	saw_j = saw_unit = 0;
	while (n > 0 && isspace((unsigned char)text[n - 1]))
		n--;
	snprintf(in->text, sizeof in->text, "%.*s", (int)n, text);
	if (n == 0)
		fail("which expression?");
	/* a defined function, by name */
	for (k = 0; k < (int)n && ident_ch(text[k]); k++)
		;
	if (k == (int)n && (g = func_find(sym_intern(text, n))) != NULL &&
	    !is_unit_name(in->text)) {
		in->fn = *g;
		in->body = g->body;
		unit_slots = 0;
		for (k = 0; k < g->np && !trig; k++)
			if (is_unit_name(sym_str(g->p[k].name))) {
				/* a declared parameter named i: which is meant? */
				static const char *const o[] = {"the imaginary unit",
					"a real parameter"};
				char what[160];
				snprintf(what, sizeof what, "%s's parameter %s", in->text,
				    sym_str(g->p[k].name));
				if (nlp_ask_choice(what, o, 2, 0) == 0) {
					unit_slots |= 1ull << k;
					saw_unit = 1;
				}
			}
	} else {
		if (func_find(sym_i)) {
			unit_name = "i_";
			note("i is a defined function; here it is the imaginary unit");
		}
		prep(in->text, &t);
		in->fn.name = sym_cstr("_");
		in->fn.anon = 1;
		in->body = calc_parse_text(sb_str(&t), &in->fn, TY_REAL);
		free(t.p);
		in->fn.body = in->body;
		unit_slots = 0;
		for (k = 0; k < in->fn.np; k++)
			if (!strcmp(sym_str(in->fn.p[k].name), unit_name))
				unit_slots |= 1ull << k;
		if (saw_j)
			note("j is read as the imaginary unit (j^2 = -1)");
	}
	if (trig && unit_slots)
		fail("trig works on real expressions; for i use complex or polar");
	for (k = 0; k < in->fn.np; k++)
		if (!(unit_slots >> k & 1)) {
			in->nreal++;
			if (!TY_FP(in->fn.p[k].ty))
				fail("%s is an integer parameter; these forms need real "
				    "ones", sym_str(in->fn.p[k].name));
		}
}

/* "x and y are taken as real" */
static void
real_note(const struct inp *in, int units)
{
	struct sbuf b = {0};
	int k, m = 0;

	for (k = 0; k < in->fn.np; k++)
		if (!(unit_slots >> k & 1))
			sb_printf(&b, "%s%s", m++ ? ", " : "", sym_str(in->fn.p[k].name));
	if (units && m)
		note("i is the imaginary unit; %s %s taken as real", b.p,
		    m > 1 ? "are" : "is");
	else if (m)
		note("%s %s taken as real", b.p, m > 1 ? "are" : "is");
	free(b.p);
}

/* ---- checking a printed result against the original ---- */

struct chk {
	int n, bad, nout, badout, pts;
	double maxd;
	char wit[200];
	cplx one;	/* the value, when there are no variables */
};

static uint64_t rng_state;

static double
urand(void)
{
	rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
	return (double)(rng_state >> 11) / 9007199254740992.0;
}

static double
sample_val(const struct param *p)
{
	double u = urand();

	if (p->ranged && isfinite(p->rlo) && isfinite(p->rhi) && p->rhi > p->rlo)
		return p->rlo + (p->rhi - p->rlo) * (0.02 + 0.96 * u);
	u = -3 + 6 * u;
	if (fabs(u) < 1e-3)
		u = 0.37;
	return u;
}

static int
parse_guard(const char *text, struct func *fn, nid *out)
{
	jmp_buf saved;
	int ok = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0) {
		*out = calc_parse_text(text, fn, TY_REAL);
		ok = 1;
	}
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return ok;
}

static int
same(cplx a, cplx b, double *d)
{
	double s = cabs(a) > 1 ? cabs(a) : 1;

	*d = cabs(a - b) / s;
	return *d <= 1e-9;
}

static void
fmt_c(char *b, size_t n, cplx z)
{
	if (cimag(z) == 0 || fabs(cimag(z)) < 1e-14 * (fabs(creal(z)) + 1))
		snprintf(b, n, "%.10g", creal(z));
	else if (fabs(creal(z)) < 1e-14 * fabs(cimag(z)))
		snprintf(b, n, "%.10gi", cimag(z));
	else
		snprintf(b, n, "%.10g %s %.10gi", creal(z), cimag(z) < 0 ? "-" : "+",
		    fabs(cimag(z)));
}

/*
 * text (ucas, with i) against the original body of in; cond (or NULL) is
 * where the identity is claimed.  orig_real: evaluate the original with
 * real semantics (trig) instead of principal complex values.
 */
enum { PT_ALL, PT_RE, PT_IM, PT_CONJ, PT_ABS, PT_ARG };
static int check_part;	/* compare this part of the original's value */

static cplx
part_of(cplx a)
{
	switch (check_part) {
	case PT_RE: return creal(a);
	case PT_IM: return cimag(a);
	case PT_CONJ: return conj(a);
	case PT_ABS: return cabs(a);
	case PT_ARG:
		if (a == 0)
			return NAN;
		return atan2(cimag(a) == 0 ? 0.0 : cimag(a), creal(a));
	}
	return a;
}

static int
check_text(const struct inp *in, int orig_real, const char *text,
    const char *cond, struct chk *r)
{
	struct func fr;
	nid e, ce_ = 0;
	int j, k, npts;
	double vals[MAXP];
	struct sbuf t = {0};

	memset(r, 0, sizeof *r);
	memset(&fr, 0, sizeof fr);
	fr.name = sym_cstr("_");
	fr.anon = 1;
	/* the original's parameters first, so names and slots agree */
	for (k = 0; k < in->fn.np; k++) {
		fr.p[k] = in->fn.p[k];
		if (unit_slots >> k & 1)
			fr.p[k].name = sym_cstr(unit_name);
	}
	fr.np = in->fn.np;
	prep(text, &t);
	if (!parse_guard(sb_str(&t), &fr, &e)) {
		free(t.p);
		snprintf(r->wit, sizeof r->wit, "the result does not parse");
		return -1;
	}
	free(t.p);
	if (cond && !parse_guard(cond, &fr, &ce_)) {
		snprintf(r->wit, sizeof r->wit, "the condition does not parse");
		return -1;
	}
	npts = in->nreal ? 64 : 1;
	rng_state = 0x9e3779b97f4a7c15ULL;
	for (j = 0; j < npts; j++) {
		cplx a, b, c = 1;
		double d;
		int inside;
		for (k = 0; k < in->fn.np; k++)
			vals[k] = sample_val(&in->fn.p[k]);
		/* a few points at the special values 0 and +-1 */
		if (j < 3 && in->nreal)
			for (k = 0; k < in->fn.np; k++)
				vals[k] = j == 0 ? 0.5 : j == 1 ? -0.5 : 1.25;
		for (k = 0; k < 64; k++)
			cv_slot[k] = NAN;
		for (k = 0; k < fr.np; k++)
			cv_slot[k] = !strcmp(sym_str(fr.p[k].name), unit_name) ||
			    (k < in->fn.np && (unit_slots >> k & 1)) ? I :
			    k < in->fn.np ? vals[k] : NAN;
		cv_slot[63] = I;
		cv_real = orig_real;
		a = part_of(ceval(in->body));
		cv_real = 0;
		b = ceval(e);
		if (cond) {
			cv_real = 1;
			c = ceval(ce_);
			cv_real = 0;
		}
		if (!isfinite(creal(a)) || !isfinite(cimag(a)))
			continue;
		inside = !cond || (isfinite(creal(c)) && creal(c) != 0);
		if (npts == 1)
			r->one = a;
		if (inside) {
			r->n++;
			if (!isfinite(creal(b)) || !isfinite(cimag(b)) ||
			    !same(a, b, &d)) {
				if (r->bad++ == 0) {
					char x1[64], x2[64];
					struct sbuf w = {0};
					for (k = 0; k < in->fn.np; k++)
						if (!(unit_slots >> k & 1))
							sb_printf(&w, "%s%s = %.6g",
							    w.n ? ", " : "",
							    sym_str(in->fn.p[k].name), vals[k]);
					fmt_c(x1, sizeof x1, a);
					fmt_c(x2, sizeof x2, b);
					snprintf(r->wit, sizeof r->wit, "%s%s%s vs %s",
					    w.n ? "at " : "", w.n ? w.p : "", w.n ? ": " : "",
					    x1);
					strncat(r->wit, "", 1);
					snprintf(r->wit + strlen(r->wit),
					    sizeof r->wit - strlen(r->wit), " (rewrite %s)", x2);
					free(w.p);
				}
			} else if (d > r->maxd)
				r->maxd = d;
		} else {
			r->nout++;
			if (!isfinite(creal(b)) || !same(a, b, &d))
				r->badout++;
		}
	}
	r->pts = npts;
	return r->bad == 0 && r->n > 0;
}

static int
print_check(const struct inp *in, const struct chk *r, const char *cond,
    const char *ind)
{
	char v[96];

	if (r->n == 0) {
		printf("%scheck: %s\n", ind, r->wit[0] ? r->wit :
		    "no sample point where both sides are defined");
		return 0;
	}
	if (r->bad) {
		printf("%scheck: FAILED at %d of %d sample points, %s\n", ind, r->bad,
		    r->n, r->wit);
		return 0;
	}
	if (r->pts == 1) {
		fmt_c(v, sizeof v, r->one);
		printf("%scheck: equal numerically (%s, complex arithmetic)\n", ind,
		    v);
		return 1;
	}
	printf("%scheck: equal at %d of %d sample points%s%s", ind, r->n, r->pts,
	    cond ? " where " : "", cond ? cond : "");
	if (r->maxd > 0)
		printf(" (max relative difference %.0e)", r->maxd);
	printf("\n");
	if (cond && r->badout)
		printf("%s  (outside: differs at %d of %d points, so the condition "
		    "is needed)\n", ind, r->badout, r->nout);
	return 1;
}

/* ---- a + bi ---- */

static int
all_neg(nid e)
{
	struct poly p;
	int k, neg = 1;

	lc_of(e, &p);
	for (k = 0; k < p.n; k++)
		if (p.t[k].c > 0)
			neg = 0;
	if (p.n == 0)
		neg = 0;
	p_free(&p);
	return neg;
}

/* "7i/10", "i*sqrt(2)/2", "i*sin(x)"; *neg gets the sign */
static void
im_text(struct sbuf *b, nid im, int *neg)
{
	double v;
	struct cf c;
	size_t k;
	int digits = 1;

	*neg = 0;
	if (num_of(im, &v)) {
		cfmt(v, &c);
		*neg = c.neg;
		for (k = 0; c.num[k]; k++)
			if (!isdigit((unsigned char)c.num[k]))
				digits = 0;
		if (!strcmp(c.num, "1"))
			sb_puts(b, "i");
		else if (digits)
			sb_printf(b, "%si", c.num);
		else
			sb_printf(b, "i*%s", c.num);
		if (c.den[0])
			sb_printf(b, "/%s", c.den);
		return;
	}
	if (all_neg(im)) {
		*neg = 1;
		im = norm(mk_un(O_NEG, im));
	}
	{
		/* an integer coefficient goes in front: 2i*x*y */
		struct poly p;
		double c1 = 0;
		lc_of(im, &p);
		if (p.n == 1 && p.t[0].c == floor(p.t[0].c) && p.t[0].c > 1 &&
		    p.t[0].c < 1e6)
			c1 = p.t[0].c;
		p_free(&p);
		if (c1) {
			sb_printf(b, "%.0fi*%s", c1, pstr_at(norm(ndiv(im,
			    mk_const_r(c1))), PS_PROD));
			return;
		}
	}
	sb_printf(b, "i*%s", pstr_at(im, PS_PROD));
}

static void
zfmt(struct sbuf *b, nid re, nid im)
{
	int neg;
	struct sbuf t = {0};

	if (zero(im)) {
		pr(b, re, 0);
		return;
	}
	im_text(&t, im, &neg);
	if (zero(re))
		sb_printf(b, "%s%s", neg ? "-" : "", t.p);
	else {
		pr(b, re, PS_SUM);
		sb_printf(b, " %s %s", neg ? "-" : "+", t.p);
	}
	free(t.p);
}

static const char *
zstr(nid re, nid im)
{
	static struct sbuf b[4];
	static int k;
	struct sbuf *s = &b[k++ & 3];

	sb_clear(s);
	zfmt(s, re, im);
	return sb_str(s);
}

/* ---- polar form ---- */

struct polar {
	nid r, th;		/* modulus and argument */
	const char *cond;	/* where th is the principal argument */
	int zero;		/* z = 0: no argument */
	int approx;		/* th is not exact: also print its value */
	double thv;
};

/* R cos(u) + i R sin(u) with R > 0: the argument is u reduced */
static int
cis_pattern(nid re, nid im, nid *R, nid *u)
{
	struct poly a, b;
	struct term ta, tb;
	int k, ok = 0;

	lc_of(re, &a);
	lc_of(im, &b);
	if (a.n == 1 && b.n == 1) {
		for (k = 0; k < a.t[0].nf; k++) {
			nid f = a.t[0].f[k], g;
			if (N.op[f] != O_COS || a.t[0].p[k] != 1)
				continue;
			g = mk_un(O_SIN, N.a[f]);
			if (t_pow(&b.t[0], g) != 1)
				continue;
			if (!t_bump(&a.t[0], f, -1, &ta) ||
			    !t_bump(&b.t[0], g, -1, &tb))
				continue;
			if (!t_same(&ta, &tb) || !near(ta.c, tb.c) || ta.c <= 0)
				continue;
			for (ok = 1, k = 0; k < ta.nf; k++)
				if (!nonneg_atom(ta.f[k]) && ta.p[k] % 2)
					ok = 0;
			if (ok) {
				struct poly rp;
				memset(&rp, 0, sizeof rp);
				p_add_term(&rp, &ta, 1);
				*R = lc_build(&rp);
				p_free(&rp);
				*u = N.a[f];
			}
			break;
		}
	}
	p_free(&a);
	p_free(&b);
	return ok;
}

static void
polar_of(nid re, nid im, struct polar *P)
{
	double a, b;
	static char cond[256];
	nid R, u;

	memset(P, 0, sizeof *P);
	if (num_of(re, &a) && num_of(im, &b)) {
		struct cf c;
		if (a == 0 && b == 0) {
			P->zero = 1;
			P->r = K(0);
			return;
		}
		P->r = mk_const_r(hypot(a, b));
		P->thv = atan2(b == 0 ? 0.0 : b, a);
		cfmt(P->thv, &c);
		if (c.exact)
			P->th = mk_const_r(P->thv);
		else {
			nid q = raw1(O_ATAN, mk_const_r(b / a));
			P->approx = 1;
			if (a > 0)
				P->th = q;
			else if (a < 0 && b >= 0)
				P->th = raw2(O_ADD, q, mk_const_r(M_PI));
			else
				P->th = raw2(O_SUB, q, mk_const_r(M_PI));
		}
		return;
	}
	if (cis_pattern(re, im, &R, &u)) {
		P->r = R;
		P->th = norm(u);
		snprintf(cond, sizeof cond, "-pi < %s <= pi", pstr(P->th));
		P->cond = cond;
		return;
	}
	P->r = norm(mk_un(O_SQRT, nadd(nsq(re), nsq(im))));
	P->th = norm(zarg(Z(re, im)));
	if (zero(im) || zero(re)) {
		/* a real or imaginary z of unknown sign */
		snprintf(cond, sizeof cond, "%s != 0", pstr(zero(im) ? re : im));
		P->cond = NULL;
	}
}

/* "sqrt(2)*e^(i*pi/4)" and "sqrt(2)*(cos(pi/4) + i*sin(pi/4))" */
static void
polar_text(const struct polar *P, struct sbuf *ex, struct sbuf *tr)
{
	double v;
	int r1 = one(P->r);
	const char *rs = pstr_at(P->r, PS_PROD);
	char rbuf[256];

	snprintf(rbuf, sizeof rbuf, "%s", rs);
	if (num_of(P->th, &v) && v < 0 && !P->approx)
		sb_printf(ex, "%s%se^(-i*%s)", r1 ? "" : rbuf, r1 ? "" : "*",
		    pstr_at(mk_const_r(-v), PS_PROD));
	else
		sb_printf(ex, "%s%se^(i*%s)", r1 ? "" : rbuf, r1 ? "" : "*",
		    pstr_at(P->th, PS_PROD));
	if (r1)
		sb_printf(tr, "cos(%s) + i*sin(%s)", pstr(P->th), pstr(P->th));
	else
		sb_printf(tr, "%s*(cos(%s) + i*sin(%s))", rbuf, pstr(P->th),
		    pstr(P->th));
}

/* ---- the complex commands ---- */

static int
can_be_zero(nid re, nid im)
{
	double v;

	return !((num_of(re, &v) && v != 0) || (num_of(im, &v) && v != 0));
}

enum { FC_RECT, FC_RE, FC_IM, FC_CONJ, FC_MOD, FC_ARG, FC_POLAR };

static const char *
lhs_text(const struct inp *in)
{
	return in->text;	/* as written */
}

static void
show_conds(const char *ind)
{
	if (lc_conds[0])
		printf("%s(for %s: a factor was cancelled)\n", ind, lc_conds);
}

static void
cmd_cform(const char *text, int which)
{
	struct inp in;
	struct cz z;
	nid re, im;
	struct chk ck;
	struct sbuf b = {0};
	const char *lhs;

	lc_conds[0] = 0;
	get_input(text, &in, 0);
	real_note(&in, unit_slots != 0);
	notes_flush();
	z = zof(in.body);
	re = norm(z.re);
	im = norm(z.im);
	notes_flush();
	lhs = lhs_text(&in);
	check_part = PT_ALL;
	switch (which) {
	case FC_RECT:
		zfmt(&b, re, im);
		printf("%s = %s\n", lhs, b.p);
		printf("  Re = %s,  Im = %s\n", pstr(re), pstr(im));
		break;
	case FC_RE:
		pr(&b, re, 0);
		printf("Re(%s) = %s\n", lhs, b.p);
		check_part = PT_RE;
		break;
	case FC_IM:
		pr(&b, im, 0);
		printf("Im(%s) = %s\n", lhs, b.p);
		check_part = PT_IM;
		break;
	case FC_CONJ:
		zfmt(&b, re, norm(nneg(im)));
		printf("conj(%s) = %s\n", lhs, b.p);
		check_part = PT_CONJ;
		break;
	case FC_MOD:
		pr(&b, zabs(Z(re, im)), 0);
		printf("|%s| = %s\n", lhs, b.p);
		check_part = PT_ABS;
		break;
	case FC_ARG:
	case FC_POLAR: {
		struct polar P;
		struct sbuf ex = {0}, tr = {0};
		polar_of(re, im, &P);
		if (P.zero) {
			printf("%s = 0: the modulus is 0 and the argument is undefined\n",
			    lhs);
			free(b.p);
			return;
		}
		if (which == FC_ARG) {
			pr(&b, P.th, 0);
			printf("arg(%s) = %s", lhs, b.p);
			if (P.approx)
				printf("  (about %.10g)", P.thv);
			printf("\n  the principal argument, -pi < arg <= pi");
			if (P.cond)
				printf("; this form for %s (otherwise reduce it by a "
				    "multiple of 2pi)", P.cond);
			else if (!num_of(P.th, &P.thv) && !P.approx &&
			    can_be_zero(re, im))
				printf("; undefined where %s = %s = 0", pstr(re),
				    pstr(im));
			printf("\n");
			check_part = PT_ARG;
			break;
		}
		polar_text(&P, &ex, &tr);
		printf("%s = %s\n", lhs, ex.p);
		printf("  = %s\n", tr.p);
		printf("  r = |z| = %s,  theta = arg z = %s", pstr(P.r), pstr(P.th));
		if (P.approx)
			printf("  (about %.10g)", P.thv);
		printf("\n  theta is the principal argument, -pi < theta <= pi");
		if (P.cond)
			printf(";\n  theta = %s for %s (otherwise reduce it by a "
			    "multiple of 2pi)", pstr(P.th), P.cond);
		else if (!num_of(P.th, &P.thv) && !P.approx && can_be_zero(re, im))
			printf("; undefined where %s = %s = 0", pstr(re), pstr(im));
		printf("\n");
		show_conds("  ");
		check_text(&in, 0, ex.p, P.cond, &ck);
		print_check(&in, &ck, P.cond, "  ");
		check_text(&in, 0, tr.p, P.cond, &ck);
		print_check(&in, &ck, P.cond, "  ");
		free(ex.p);
		free(tr.p);
		free(b.p);
		return;
	}
	}
	show_conds("  ");
	check_text(&in, 0, b.p, NULL, &ck);
	print_check(&in, &ck, NULL, "  ");
	check_part = PT_ALL;
	free(b.p);
}

/* the n complex n-th roots */
static void
cmd_nthroots(int n, const char *text)
{
	struct inp in;
	struct cz z;
	nid re, im;
	struct polar P;
	struct chk ck;
	int k;
	static const char *const ord[] = {"", "", "square", "cube", "fourth",
		"fifth", "sixth", "seventh", "eighth", "ninth", "tenth"};
	char nm[32];

	if (n < 1 || n > 64)
		fail("nthroots: n must be in 1..64");
	lc_conds[0] = 0;
	get_input(text, &in, 0);
	real_note(&in, unit_slots != 0);
	notes_flush();
	z = zof(in.body);
	re = norm(z.re);
	im = norm(z.im);
	polar_of(re, im, &P);
	if (n <= 10)
		snprintf(nm, sizeof nm, "%s", ord[n]);
	else
		snprintf(nm, sizeof nm, "%dth", n);
	if (P.zero) {
		printf("the %s roots of 0: w = 0 (%d-fold)\n", nm, n);
		return;
	}
	if (!(num_of(re, &P.thv) && num_of(im, &P.thv))) {
		/* symbolic: the formula, checked for k = 0 and k = 1 */
		struct sbuf w = {0};
		printf("the %d %s roots of %s: w^%d = z\n", n, nm, lhs_text(&in), n);
		printf("  w_k = r^(1/%d) e^(i (theta + 2 pi k)/%d),  k = 0..%d\n", n,
		    n, n - 1);
		printf("  r = |z| = %s,  theta = arg z = %s\n", pstr(P.r),
		    pstr(P.th));
		for (k = 0; k < 2 && k < n; k++) {
			sb_clear(&w);
			sb_printf(&w, "(%s^(1/%d)*e^(i*(%s + 2pi*%d)/%d))^%d",
			    pstr_at(P.r, PS_ATOM), n, pstr(P.th), k, n, n);
			check_text(&in, 0, w.p, P.cond, &ck);
			printf("  k = %d: ", k);
			print_check(&in, &ck, P.cond, "");
		}
		free(w.p);
		return;
	}
	{
		double a, b, r, th;
		int nbad = 0;
		num_of(re, &a);
		num_of(im, &b);
		r = hypot(a, b);
		th = atan2(b == 0 ? 0.0 : b, a);
		printf("the %d %s roots of %s (w^%d = %s)\n", n, nm, lhs_text(&in), n,
		    zstr(re, im));
		for (k = 0; k < n; k++) {
			double t = (th + 2 * M_PI * k) / n, rr = pow(r, 1.0 / n);
			double wr = rr * cos(t), wi = rr * sin(t);
			struct cf c1, c2, c3;
			struct sbuf ex = {0}, chk = {0};
			nid rn;
			if (t > M_PI + 1e-12)
				t -= 2 * M_PI;
			if (fabs(wr) < 1e-13 * rr)
				wr = 0;
			if (fabs(wi) < 1e-13 * rr)
				wi = 0;
			cfmt(wr, &c1);
			cfmt(wi, &c2);
			cfmt(rr, &c3);
			rn = c3.exact ? mk_const_r(rr) : raw2(O_POW, mk_const_r(r),
			    raw2(O_DIV, K(1), K(n)));
			if (fabs(t) < 1e-15)
				sb_printf(&ex, "%s", pstr(rn));
			else
				sb_printf(&ex, "%s%se^(%si*%s)", one(rn) ? "" :
				    pstr_at(rn, PS_PROD), one(rn) ? "" : "*",
				    t < 0 ? "-" : "", pstr_at(mk_const_r(fabs(t)),
				    PS_PROD));
			printf("  w%d = ", k);
			if (c1.exact && c2.exact) {
				const char *s = zstr(mk_const_r(wr), mk_const_r(wi));
				printf("%-22s = %s", s, ex.p);
				sb_printf(&chk, "(%s)^%d", s, n);
			} else {
				char v[96];
				fmt_c(v, sizeof v, wr + I * wi);
				printf("%s  (about %s)", ex.p, v);
				sb_printf(&chk, "(%s)^%d", ex.p, n);
			}
			if (k == 0)
				printf("  (principal)");
			printf("\n");
			check_text(&in, 0, chk.p, NULL, &ck);
			if (!ck.n || ck.bad) {
				print_check(&in, &ck, NULL, "      ");
				nbad++;
			}
			free(ex.p);
			free(chk.p);
		}
		if (nbad == 0)
			printf("  check: each w^%d, parsed back as printed, equals %s "
			    "numerically\n", n, zstr(re, im));
	}
}

/* ---- trig rewrites ---- */

enum {
	M_SIMP = 1, M_SINCOS = 2, M_EXP = 4, M_REDUCE = 8, M_PRODUCT = 16,
	M_SUM = 32, M_EXPAND = 64, M_HARM = 128, M_HALF = 256, M_INV = 512,
	M_ALL = 1023, M_TSUB = 1024	/* t = tan(u/2) only when asked */
};

struct cand {
	char label[64];
	char ident[200];
	char text[2048];
	char cond[400];
	int ok;
};
#define MAXCAND 40
static struct cand cands[MAXCAND];
static int ncand;

static void
add_cand(const char *label, const char *ident, nid e, const char *cond)
{
	struct cand *c;
	struct sbuf b = {0};
	int k;

	pr(&b, e, 0);
	for (k = 0; k < ncand; k++)
		if (!strcmp(cands[k].text, b.p) && !strcmp(cands[k].cond,
		    cond ? cond : "")) {
			free(b.p);
			return;
		}
	if (ncand >= MAXCAND || b.n >= sizeof cands[0].text) {
		free(b.p);
		return;
	}
	c = &cands[ncand++];
	memset(c, 0, sizeof *c);
	snprintf(c->label, sizeof c->label, "%s", label);
	snprintf(c->ident, sizeof c->ident, "%s", ident ? ident : "");
	snprintf(c->text, sizeof c->text, "%s", b.p);
	snprintf(c->cond, sizeof c->cond, "%s", cond ? cond : "");
	free(b.p);
}

/* a generic rebuild with one rule per op */
typedef nid (*rule_fn)(nid e, nid a, nid b);
static rule_fn rw_rule;
static nid *rwmemo;
static uint32_t rwn;
static int rw_hit;
static char rw_cond[400];

static void
rcond(const char *s)
{
	if (strstr(rw_cond, s))
		return;
	if (strlen(rw_cond) + strlen(s) + 6 >= sizeof rw_cond)
		return;
	if (rw_cond[0])
		strcat(rw_cond, " && ");
	strcat(rw_cond, s);
}

static nid
remake(nid e, nid a, nid b, nid c)
{
	int op = N.op[e];

	if (a == N.a[e] && b == N.b[e] && c == N.c[e])
		return e;
	if (op == O_CAST)
		return mk_cast(N.ty[e], a);
	if (op == O_SEL)
		return mk_sel(a, b, c);
	if (op == O_FMA)
		return mk_fma(a, b, c);
	if (opinfo[op].arity == 1)
		return mk_un(op, a);
	return mk_bin(op, a, b);
}

static nid
rw1(nid e)
{
	int op = N.op[e];
	nid a = 0, b = 0, c = 0, r;

	if (e < rwn && rwmemo[e])
		return rwmemo[e];
	if (op == O_CONST || op == O_VAR || op == O_UNDEF || op == O_TABLE ||
	    op == O_LOOP || op == O_ATOMIC || op == O_INDEX ||
	    opinfo[op].arity == 0)
		r = e;
	else {
		if (opinfo[op].arity >= 1)
			a = rw1(N.a[e]);
		if (opinfo[op].arity >= 2)
			b = rw1(N.b[e]);
		if (opinfo[op].arity >= 3)
			c = rw1(N.c[e]);
		r = rw_rule(e, a, b);
		if (r == 0)
			r = remake(e, a, b, c);
		else
			rw_hit = 1;
	}
	if (e < rwn)
		rwmemo[e] = r;
	return r;
}

static nid
rewrite(nid e, rule_fn f)
{
	nid r, *saved = rwmemo;
	uint32_t sn = rwn;
	rule_fn sf = rw_rule;

	rw_rule = f;
	rwn = N.n;
	rwmemo = xcalloc(rwn, sizeof *rwmemo);
	r = rw1(e);
	free(rwmemo);
	rwmemo = saved;
	rwn = sn;
	rw_rule = sf;
	return r;
}

static nid
ki(int64_t v)
{
	return K(v);
}

/* tan sec csc cot tanh in sin and cos */
static nid
rule_sincos(nid e, nid a, nid b)
{
	switch (N.op[e]) {
	case O_TAN: return mk_bin(O_DIV, mk_un(O_SIN, a), mk_un(O_COS, a));
	case O_SEC: return mk_bin(O_DIV, ki(1), mk_un(O_COS, a));
	case O_CSC: return mk_bin(O_DIV, ki(1), mk_un(O_SIN, a));
	case O_COT: return mk_bin(O_DIV, mk_un(O_COS, a), mk_un(O_SIN, a));
	case O_TANH: return mk_bin(O_DIV, mk_un(O_SINH, a), mk_un(O_COSH, a));
	}
	return 0;
}

/* Euler: everything through exp and log */
static nid
rule_exp(nid e, nid a, nid b)
{
	nid i = unit_node(), p, m, ia = nmul(i, a);
	nid one_ = ki(1), two = ki(2);

	p = mk_un(O_EXP, ia);
	m = mk_un(O_EXP, nneg(ia));
	switch (N.op[e]) {
	case O_SIN: return ndiv(nsub(p, m), nmul(two, i));
	case O_COS: return ndiv(nadd(p, m), two);
	case O_TAN: return ndiv(nsub(p, m), nmul(i, nadd(p, m)));
	case O_SEC: return ndiv(two, nadd(p, m));
	case O_CSC: return ndiv(nmul(two, i), nsub(p, m));
	case O_COT: return ndiv(nmul(i, nadd(p, m)), nsub(p, m));
	}
	p = mk_un(O_EXP, a);
	m = mk_un(O_EXP, nneg(a));
	switch (N.op[e]) {
	case O_SINH: return ndiv(nsub(p, m), two);
	case O_COSH: return ndiv(nadd(p, m), two);
	case O_TANH: return ndiv(nsub(p, m), nadd(p, m));
	case O_ASIN:	/* -i log(i u + sqrt(1 - u^2)) */
		return nmul(nneg(i), mk_un(O_LOG, nadd(ia, mk_un(O_SQRT,
		    nsub(one_, nsq(a))))));
	case O_ACOS:	/* -i log(u + i sqrt(1 - u^2)) */
		return nmul(nneg(i), mk_un(O_LOG, nadd(a, nmul(i, mk_un(O_SQRT,
		    nsub(one_, nsq(a)))))));
	case O_ATAN:	/* i/2 log((1 - i u)/(1 + i u)) */
		return nmul(ndiv(i, two), mk_un(O_LOG, ndiv(nsub(one_, ia),
		    nadd(one_, ia))));
	case O_ASINH:
		return mk_un(O_LOG, nadd(a, mk_un(O_SQRT, nadd(nsq(a), one_))));
	case O_ACOSH:
		return mk_un(O_LOG, nadd(a, mk_un(O_SQRT, nsub(nsq(a), one_))));
	case O_ATANH:
		return ndiv(mk_un(O_LOG, ndiv(nadd(one_, a), nsub(one_, a))), two);
	}
	return 0;
}

/* t = tan(u/2) (Weierstrass) */
static nid
rule_half(nid e, nid a, nid b)
{
	nid t, t2, one_ = ki(1), two = ki(2);
	int op = N.op[e];

	if (op != O_SIN && op != O_COS && op != O_TAN && op != O_SEC &&
	    op != O_CSC && op != O_COT)
		return 0;
	t = mk_un(O_TAN, norm(ndiv(a, two)));
	t2 = nsq(t);
	rcond(S_cond(mk_un(O_COS, norm(ndiv(a, two)))));
	switch (op) {
	case O_SIN: return ndiv(nmul(two, t), nadd(one_, t2));
	case O_COS: return ndiv(nsub(one_, t2), nadd(one_, t2));
	case O_TAN: return ndiv(nmul(two, t), nsub(one_, t2));
	case O_SEC: return ndiv(nadd(one_, t2), nsub(one_, t2));
	case O_CSC: return ndiv(nadd(one_, t2), nmul(two, t));
	}
	return ndiv(nsub(one_, t2), nmul(two, t));
}

/* sin(asin u) = u, asin(sin u) = u for |u| <= pi/2 ... */
static nid
rule_compose(nid e, nid a, nid b)
{
	int op = N.op[e], ia = N.op[a];
	nid u = N.a[a], one_ = ki(1), s1, s2;
	char c[300];

	if (opinfo[ia].arity != 1)
		return 0;
	s1 = mk_un(O_SQRT, nsub(one_, nsq(u)));	/* sqrt(1 - u^2) */
	s2 = mk_un(O_SQRT, nadd(one_, nsq(u)));	/* sqrt(1 + u^2) */
	switch (op * 1000 + ia) {
	case O_SIN * 1000 + O_ASIN: return u;
	case O_COS * 1000 + O_ASIN: return s1;
	case O_TAN * 1000 + O_ASIN: return ndiv(u, s1);
	case O_SIN * 1000 + O_ACOS: return s1;
	case O_COS * 1000 + O_ACOS: return u;
	case O_TAN * 1000 + O_ACOS: return ndiv(s1, u);
	case O_SIN * 1000 + O_ATAN: return ndiv(u, s2);
	case O_COS * 1000 + O_ATAN: return ndiv(one_, s2);
	case O_TAN * 1000 + O_ATAN: return u;
	case O_SEC * 1000 + O_ATAN: return s2;
	case O_SINH * 1000 + O_ASINH: return u;
	case O_COSH * 1000 + O_ASINH: return s2;
	case O_COSH * 1000 + O_ACOSH: return u;
	case O_SINH * 1000 + O_ACOSH:
		return mk_un(O_SQRT, nsub(nsq(u), one_));
	case O_TANH * 1000 + O_ATANH: return u;
	case O_ASINH * 1000 + O_SINH: return u;
	case O_ATANH * 1000 + O_TANH: return u;
	case O_ACOSH * 1000 + O_COSH: return mk_un(O_ABS, u);
	case O_LOG * 1000 + O_EXP: return u;
	case O_EXP * 1000 + O_LOG: return u;
	case O_ASIN * 1000 + O_SIN:
		snprintf(c, sizeof c, "-pi/2 <= %s <= pi/2", pstr(u));
		rcond(c);
		return u;
	case O_ACOS * 1000 + O_COS:
		snprintf(c, sizeof c, "0 <= %s <= pi", pstr(u));
		rcond(c);
		return u;
	case O_ATAN * 1000 + O_TAN:
		snprintf(c, sizeof c, "-pi/2 < %s < pi/2", pstr(u));
		rcond(c);
		return u;
	}
	return 0;
}

static int
has_op(nid e, int op, uint8_t *seen)
{
	int r = 0;

	if (seen[e])
		return 0;
	seen[e] = 1;
	if (N.op[e] == op)
		return 1;
	if (opinfo[N.op[e]].arity >= 1 && N.op[e] != O_TABLE &&
	    N.op[e] != O_LOOP)
		r |= has_op(N.a[e], op, seen);
	if (opinfo[N.op[e]].arity >= 2 && N.op[e] != O_TABLE &&
	    N.op[e] != O_LOOP)
		r |= has_op(N.b[e], op, seen);
	if (opinfo[N.op[e]].arity >= 3)
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

/* ---- linear forms of angles ---- */

#define MA 8
struct ang {
	double c0;		/* constant part */
	int na;
	nid atom[MA];		/* monomials, coefficient 1 */
	double k[MA];
};

static nid
mono_node(const struct term *t)
{
	struct poly p;
	struct term u = *t;
	nid r;

	u.c = 1;
	memset(&p, 0, sizeof p);
	p_add_term(&p, &u, 1);
	r = lc_build(&p);
	p_free(&p);
	return r;
}

static int
ang_of(nid u, struct ang *A)
{
	struct poly p;
	int k, j;

	memset(A, 0, sizeof *A);
	lc_of(u, &p);
	for (k = 0; k < p.n; k++) {
		nid m;
		if (p.t[k].nf == 0) {
			A->c0 += p.t[k].c;
			continue;
		}
		m = mono_node(&p.t[k]);
		for (j = 0; j < A->na; j++)
			if (A->atom[j] == m)
				break;
		if (j == A->na) {
			if (A->na == MA) {
				p_free(&p);
				return 0;
			}
			A->atom[A->na++] = m;
		}
		A->k[j] += p.t[k].c;
	}
	p_free(&p);
	return 1;
}

/* ---- linearising products and powers (power reduction, product to
 * sum): every sin, cos, sinh, cosh as exponentials, multiplied out,
 * then paired back into cos/sin of combined angles ---- */

#define MFT 600
struct fterm {
	cplx c;
	nid mono;
	double v[2 * MA];	/* angle coefficients: trig atoms, hyp atoms */
};
static struct fterm *ft;
static int nft;
static nid fatom[2 * MA];
static int nfatom, fbad;

static int
fatom_ix(nid m, int hyp)
{
	int k;

	for (k = 0; k < nfatom; k++)
		if (fatom[k] == m && (k & 1) == hyp)
			return k;
	/* trig atoms at even slots, hyperbolic at odd */
	for (k = hyp; k < 2 * MA; k += 2)
		if (fatom[k] == 0) {
			fatom[k] = m;
			if (k + 1 > nfatom)
				nfatom = k + 1;
			return k;
		}
	fbad = 1;
	return 0;
}

static int
fsame(const struct fterm *a, const struct fterm *b)
{
	int k;

	if (a->mono != b->mono)
		return 0;
	for (k = 0; k < 2 * MA; k++)
		if (!near(a->v[k], b->v[k]))
			return 0;
	return 1;
}

static void
fadd(struct fterm *set, int *n, const struct fterm *t)
{
	int k;

	if (cabs(t->c) < 1e-14)
		return;
	for (k = 0; k < *n; k++)
		if (fsame(&set[k], t)) {
			set[k].c += t->c;
			return;
		}
	if (*n >= MFT) {
		fbad = 1;
		return;
	}
	set[(*n)++] = *t;
}

/* set *= (sum of e terms) */
static void
fmul(struct fterm *set, int *n, const struct fterm *e, int ne)
{
	struct fterm *out = xmalloc(MFT * sizeof *out);
	int no = 0, i, j, k;

	for (i = 0; i < *n; i++)
		for (j = 0; j < ne; j++) {
			struct fterm t = set[i];
			t.c *= e[j].c;
			for (k = 0; k < 2 * MA; k++)
				t.v[k] += e[j].v[k];
			fadd(out, &no, &t);
		}
	memcpy(set, out, no * sizeof *out);
	*n = no;
	free(out);
}

/* sin, cos, sinh, cosh of u as two exponentials */
static int
fexpand_atom(nid f, struct fterm *e)
{
	struct ang A;
	int op = N.op[f], hyp = op == O_SINH || op == O_COSH, k, ix;
	cplx ph, mh;

	if (!ang_of(N.a[f], &A))
		return 0;
	memset(e, 0, 2 * sizeof *e);
	for (k = 0; k < A.na; k++) {
		ix = fatom_ix(A.atom[k], hyp);
		e[0].v[ix] = A.k[k];
		e[1].v[ix] = -A.k[k];
	}
	if (hyp) {
		ph = exp(A.c0);
		mh = exp(-A.c0);
	} else {
		ph = cexp(I * A.c0);
		mh = cexp(-I * A.c0);
	}
	switch (op) {
	case O_SIN: e[0].c = ph / (2 * I); e[1].c = -mh / (2 * I); break;
	case O_COS: e[0].c = ph / 2; e[1].c = mh / 2; break;
	case O_SINH: e[0].c = ph / 2; e[1].c = -mh / 2; break;
	case O_COSH: e[0].c = ph / 2; e[1].c = mh / 2; break;
	}
	return 1;
}

static int
is_sc(nid f)
{
	int op = N.op[f];

	return op == O_SIN || op == O_COS || op == O_SINH || op == O_COSH;
}

static nid
ang_node(const double *v, int sign)
{
	struct poly p;
	int k;
	nid r;

	memset(&p, 0, sizeof p);
	for (k = 0; k < 2 * MA; k++)
		if (fatom[k] && fabs(v[k]) > 1e-12) {
			struct poly a;
			p_atom(&a, fatom[k], 1);
			p_add(&p, &a, sign * v[k]);
			p_free(&a);
		}
	{
		/* the atoms are monomials: renormalise x*y as a product */
		r = lc_build(&p);
		p_free(&p);
		return norm(r);
	}
}

/* the linearised form, or 0; *pw and *prod say what was there */
static nid
linearize(nid body, int *pw, int *prod)
{
	struct poly p;
	int k, j, i;
	nid r = K(0);

	*pw = *prod = 0;
	lc_of(body, &p);
	ft = xmalloc(MFT * sizeof *ft);
	nft = 0;
	memset(fatom, 0, sizeof fatom);
	nfatom = 0;
	fbad = 0;
	for (k = 0; k < p.n && !fbad; k++) {
		struct fterm *set = xmalloc(MFT * sizeof *set), e[2];
		int n = 1, ntr = 0;
		struct term mono;
		memset(set, 0, sizeof *set);
		set[0].c = p.t[k].c;
		memset(&mono, 0, sizeof mono);
		mono.c = 1;
		for (j = 0; j < p.t[k].nf; j++) {
			nid f = p.t[k].f[j];
			int e_ = p.t[k].p[j];
			if (is_sc(f) && e_ > 0 && e_ <= 12 && fexpand_atom(f, e)) {
				if (e_ > 1)
					*pw = 1;
				ntr++;
				for (i = 0; i < e_; i++)
					fmul(set, &n, e, 2);
			} else {
				mono.f[mono.nf] = f;
				mono.p[mono.nf++] = e_;
			}
		}
		if (ntr > 1)
			*prod = 1;
		{
			nid mn = mono.nf ? mono_node(&mono) : 0;
			for (i = 0; i < n; i++) {
				set[i].mono = mn;
				fadd(ft, &nft, &set[i]);
			}
		}
		free(set);
	}
	p_free(&p);
	if (fbad || (!*pw && !*prod)) {
		free(ft);
		return 0;
	}
	/* pair each angle with its negation */
	{
		uint8_t *used = xcalloc(nft, 1);
		for (k = 0; k < nft && !fbad; k++) {
			struct fterm neg;
			cplx c = ft[k].c, d = 0;
			int first = -1, trig = 0, hypn = 0, s = 1;
			nid th, mono = ft[k].mono ? ft[k].mono : K(1), t;
			if (used[k])
				continue;
			used[k] = 1;
			neg = ft[k];
			for (j = 0; j < 2 * MA; j++) {
				neg.v[j] = -neg.v[j];
				if (fabs(ft[k].v[j]) > 1e-12) {
					if (first < 0)
						first = j;
					if (j & 1)
						hypn++;
					else
						trig++;
				}
			}
			if (first < 0) {
				r = nadd(r, nmul(mk_const_r(creal(c)), mono));
				continue;
			}
			if (trig && hypn) {
				fbad = 1;
				break;
			}
			for (j = k + 1; j < nft; j++)
				if (!used[j] && fsame(&ft[j], &neg)) {
					used[j] = 1;
					d = ft[j].c;
					break;
				}
			/* the angle with a positive first coefficient */
			if (ft[k].v[first] < 0) {
				cplx x = c;
				c = d;
				d = x;
				s = -1;
			}
			th = ang_node(ft[k].v, s);
			if (trig) {
				double A = creal(c + d), B = creal(I * (c - d));
				t = nadd(nmul(mk_const_r(A), mk_un(O_COS, th)),
				    nmul(mk_const_r(B), mk_un(O_SIN, th)));
			} else {
				double A = creal(c + d), B = creal(c - d);
				t = nadd(nmul(mk_const_r(A), mk_un(O_COSH, th)),
				    nmul(mk_const_r(B), mk_un(O_SINH, th)));
			}
			r = nadd(r, nmul(t, mono));
		}
		free(used);
	}
	free(ft);
	if (fbad)
		return 0;
	return norm(r);
}

/* ---- expanding compound and multiple angles ---- */

static int
q_of(double v, int64_t *n, int64_t *d)
{
	return small_q(v, 1000, 1000000, n, d);
}

/* rational gcd */
static double
rgcd(double a, double b)
{
	int64_t an, ad, bn, bd, g, l;

	if (a == 0)
		return fabs(b);
	if (b == 0)
		return fabs(a);
	if (!q_of(fabs(a), &an, &ad) || !q_of(fabs(b), &bn, &bd))
		return 0;
	g = gcd64(an, bn);
	l = ad / gcd64(ad, bd) * bd;
	return (double)g / (double)l;
}

static struct {
	nid atom[MA];
	double g[MA];
	int n;
} bases;

static int
compound(nid f, struct ang *A)
{
	if (!ang_of(N.a[f], A))
		return 0;
	if (A->na == 0)
		return 0;
	return A->na > 1 || !near(fabs(A->k[0]), 1) || A->c0 != 0;
}

static int xp_ops[] = {O_SIN, O_COS, O_TAN, O_SINH, O_COSH, O_TANH, 0};

static int
xp_op(int op)
{
	int k;

	for (k = 0; xp_ops[k]; k++)
		if (xp_ops[k] == op)
			return 1;
	return 0;
}

static void
collect_bases(nid e, uint8_t *seen)
{
	struct ang A;
	int k, j;

	if (seen[e])
		return;
	seen[e] = 1;
	if (xp_op(N.op[e]) && compound(e, &A))
		for (k = 0; k < A.na; k++) {
			for (j = 0; j < bases.n; j++)
				if (bases.atom[j] == A.atom[k])
					break;
			if (j == bases.n) {
				if (bases.n == MA)
					continue;
				bases.atom[bases.n] = A.atom[k];
				/* 3x has base x, x/2 base x/2 */
				bases.g[bases.n++] = rgcd(1, A.k[k]);
			} else
				bases.g[j] = rgcd(bases.g[j], A.k[k]);
		}
	if (N.op[e] == O_TABLE || N.op[e] == O_LOOP)
		return;
	if (opinfo[N.op[e]].arity >= 1)
		collect_bases(N.a[e], seen);
	if (opinfo[N.op[e]].arity >= 2)
		collect_bases(N.b[e], seen);
	if (opinfo[N.op[e]].arity >= 3)
		collect_bases(N.c[e], seen);
}

static nid
base_arg(int j)
{
	return norm(nmul(mk_const_r(bases.g[j]), bases.atom[j]));
}

static nid
rule_expand(nid e, nid a, nid b)
{
	int op = N.op[e], hyp, tanf, k, j;
	struct ang A;
	struct cz acc;
	int sg;

	if (!xp_op(op))
		return 0;
	if (!compound(mk_un(op, a), &A))
		return 0;
	hyp = op == O_SINH || op == O_COSH || op == O_TANH;
	tanf = op == O_TAN || op == O_TANH;
	sg = hyp ? 1 : -1;
	/* the constant part's phase */
	if (tanf) {
		double c = hyp ? 1 : cos(A.c0), s = hyp ? tanh(A.c0) : sin(A.c0);
		if (!hyp && fabs(c) > 1e-12) {
			s = s / c;
			c = 1;
		}
		acc = Z(mk_const_r(fabs(c) < 1e-15 ? 0 : c), mk_const_r(fabs(s) <
		    1e-15 ? 0 : s));
	} else {
		double c = hyp ? cosh(A.c0) : cos(A.c0),
		    s = hyp ? sinh(A.c0) : sin(A.c0);
		acc = Z(mk_const_r(fabs(c) < 1e-15 ? 0 : c),
		    mk_const_r(fabs(s) < 1e-15 ? 0 : s));
	}
	for (k = 0; k < A.na; k++) {
		double n;
		int64_t ni;
		struct cz f;
		nid ba;
		for (j = 0; j < bases.n; j++)
			if (bases.atom[j] == A.atom[k])
				break;
		if (j == bases.n || bases.g[j] == 0)
			return 0;
		n = A.k[k] / bases.g[j];
		ni = llround(n);
		if (!near(n, (double)ni) || llabs(ni) > 12)
			return 0;
		ba = base_arg(j);
		if (tanf)
			f = Z(K(1), mk_un(hyp ? O_TANH : O_TAN, ba));
		else
			f = Z(mk_un(hyp ? O_COSH : O_COS, ba),
			    mk_un(hyp ? O_SINH : O_SIN, ba));
		if (ni < 0) {
			f.im = nneg(f.im);
			ni = -ni;
		}
		acc = zmul_s(acc, zipow(f, ni, sg), sg);
		if (tanf)
			rcond(S_cond(mk_un(hyp ? O_COSH : O_COS, ba)));
	}
	switch (op) {
	case O_SIN: case O_SINH: return acc.im;
	case O_COS: case O_COSH: return acc.re;
	}
	return ndiv(acc.im, acc.re);
}

/* p with c^2 replaced by 1 - s^2 (trig) or 1 + s^2 (cosh^2 = 1 + sinh^2) */
static int
p_subst_sq(const struct poly *p, nid c, nid s, int plus, struct poly *r)
{
	int k, j, hit = 0;
	struct poly rep, one_;

	/* rep = 1 -+ s^2 */
	p_const(&rep, 1);
	p_atom(&one_, s, 2);
	p_add(&rep, &one_, plus ? 1 : -1);
	p_free(&one_);
	memset(r, 0, sizeof *r);
	for (k = 0; k < p->n; k++) {
		int e = t_pow(&p->t[k], c);
		struct poly t, acc;
		struct term rest;
		if (e % 2) {
			p_free(r);
			p_free(&rep);
			return 0;
		}
		if (e == 0) {
			p_add_term(r, &p->t[k], 1);
			continue;
		}
		hit = 1;
		t_bump(&p->t[k], c, -e, &rest);
		memset(&acc, 0, sizeof acc);
		p_add_term(&acc, &rest, 1);
		for (j = 0; j < e / 2; j++) {
			p_mul(&t, &acc, &rep);
			p_free(&acc);
			acc = t;
		}
		p_add(r, &acc, 1);
		p_free(&acc);
	}
	p_free(&rep);
	p_pyth(r);
	return hit;
}

/* ---- sum to product ---- */

static nid
sum_to_product(nid body)
{
	struct poly p;
	struct term r1, r2;
	nid f1 = 0, f2 = 0, a, b, s, d, res = 0;
	int k, o1, o2, hyp, neg;

	lc_of(body, &p);
	if (p.n != 2)
		goto out;
	for (k = 0; k < p.t[0].nf; k++)
		if (p.t[0].p[k] == 1 && is_sc(p.t[0].f[k]))
			f1 = p.t[0].f[k];
	for (k = 0; k < p.t[1].nf; k++)
		if (p.t[1].p[k] == 1 && is_sc(p.t[1].f[k]))
			f2 = p.t[1].f[k];
	if (!f1 || !f2 || !t_bump(&p.t[0], f1, -1, &r1) ||
	    !t_bump(&p.t[1], f2, -1, &r2))
		goto out;
	r2.c = r1.c;
	if (!t_same(&r1, &r2) || !near(fabs(p.t[0].c), fabs(p.t[1].c)))
		goto out;
	o1 = N.op[f1];
	o2 = N.op[f2];
	hyp = o1 == O_SINH || o1 == O_COSH;
	if (hyp != (o2 == O_SINH || o2 == O_COSH))
		goto out;
	a = N.a[f1];
	b = N.a[f2];
	neg = (p.t[0].c < 0) != (p.t[1].c < 0);
	if (o1 != o2) {
		if (hyp)
			goto out;
		/* cos b = sin(b + pi/2) */
		if (o1 == O_COS) {
			a = nadd(a, mk_const_r(M_PI / 2));
			o1 = O_SIN;
		} else {
			b = nadd(b, mk_const_r(M_PI / 2));
			o2 = O_SIN;
		}
	}
	s = norm(ndiv(nadd(a, b), K(2)));
	d = norm(ndiv(nsub(a, b), K(2)));
	if (!hyp) {
		if (o1 == O_SIN && !neg)
			res = nmul(K(2), nmul(mk_un(O_SIN, s), mk_un(O_COS, d)));
		else if (o1 == O_SIN)
			res = nmul(K(2), nmul(mk_un(O_COS, s), mk_un(O_SIN, d)));
		else if (!neg)
			res = nmul(K(2), nmul(mk_un(O_COS, s), mk_un(O_COS, d)));
		else
			res = nmul(K(-2), nmul(mk_un(O_SIN, s), mk_un(O_SIN, d)));
	} else {
		if (o1 == O_SINH && !neg)
			res = nmul(K(2), nmul(mk_un(O_SINH, s), mk_un(O_COSH, d)));
		else if (o1 == O_SINH)
			res = nmul(K(2), nmul(mk_un(O_COSH, s), mk_un(O_SINH, d)));
		else if (!neg)
			res = nmul(K(2), nmul(mk_un(O_COSH, s), mk_un(O_COSH, d)));
		else
			res = nmul(K(2), nmul(mk_un(O_SINH, s), mk_un(O_SINH, d)));
	}
	r1.c = p.t[0].c;
	res = norm(nmul(res, mono_node(&r1) == K(1) ? K(1) : mono_node(&r1)));
	if (!near(p.t[0].c, 1))
		res = norm(nmul(mk_const_r(p.t[0].c), res));
out:
	p_free(&p);
	return res;
}

/* ---- a sin u + b cos u = R sin(u + phi) ---- */

static nid
angle_const(double a, double b)	/* atan2(b, a), exact-looking */
{
	double t = atan2(b, a);
	struct cf c;
	nid q;

	cfmt(t, &c);
	if (c.exact)
		return mk_const_r(t);
	q = raw1(O_ATAN, mk_const_r(b / a));
	if (a > 0)
		return q;
	return raw2(a < 0 && b >= 0 ? O_ADD : O_SUB, q, mk_const_r(M_PI));
}

static int
harmonic(nid body, nid *f1, nid *f2, char *cond, size_t cn)
{
	struct poly p;
	int k, j, ok = 1;
	nid th = 0, A = K(0), B = K(0), C = K(0), R, ph, ps;
	double a, b;
	int hyp = 0;

	lc_of(body, &p);
	for (k = 0; k < p.n && ok; k++) {
		struct term t = p.t[k], rest;
		nid f = 0;
		int nsc = 0;
		for (j = 0; j < t.nf; j++)
			if (N.op[t.f[j]] == O_SIN || N.op[t.f[j]] == O_COS) {
				nsc++;
				f = t.f[j];
				if (t.p[j] != 1)
					ok = 0;
			}
		if (nsc == 0) {
			struct poly q;
			memset(&q, 0, sizeof q);
			p_add_term(&q, &t, 1);
			C = nadd(C, lc_build(&q));
			p_free(&q);
			continue;
		}
		if (nsc > 1 || !ok) {
			ok = 0;
			break;
		}
		if (th == 0)
			th = N.a[f];
		else if (N.a[f] != th) {
			ok = 0;
			break;
		}
		t_bump(&t, f, -1, &rest);
		{
			struct poly q;
			nid x;
			memset(&q, 0, sizeof q);
			p_add_term(&q, &rest, 1);
			x = lc_build(&q);
			p_free(&q);
			if (N.op[f] == O_SIN)
				A = nadd(A, x);
			else
				B = nadd(B, x);
		}
	}
	p_free(&p);
	(void)hyp;
	if (!ok || th == 0 || zero(A) || zero(B))
		return 0;
	A = norm(A);
	B = norm(B);
	if (num_of(A, &a) && num_of(B, &b)) {
		R = mk_const_r(hypot(a, b));
		ph = angle_const(a, b);
		ps = angle_const(b, a);
		cond[0] = 0;
	} else {
		R = norm(mk_un(O_SQRT, nadd(nsq(A), nsq(B))));
		ph = mk_bin(O_ATAN2, B, A);
		ps = mk_bin(O_ATAN2, A, B);
		snprintf(cond, cn, "%s != 0 || %s != 0", pstr_at(A, PS_PROD), pstr_at(B, PS_PROD));
	}
	*f1 = nadd(nmul(R, mk_un(O_SIN, mk_bin(O_ADD, th, ph))), C);
	*f2 = nadd(nmul(R, mk_un(O_COS, mk_bin(O_SUB, th, ps))), C);
	return 1;
}

/* ---- inverse functions: alternative forms and their conditions ---- */

static nid inv_from, inv_to;

static nid
rule_subst(nid e, nid a, nid b)
{
	return e == inv_from ? inv_to : 0;
}

static nid
parse_alt(const char *text, const struct inp *in)
{
	struct func fr = in->fn;
	nid e;

	if (!parse_guard(text, &fr, &e))
		return 0;
	if (fr.np != in->fn.np)
		return 0;	/* a new name: not ours */
	return e;
}

struct ialt {
	const char *label, *form, *cond;	/* U, Y, X: the arguments */
};

static const struct ialt alt_atan[] = {
	{"atan via asin", "asin(U/sqrt(1 + U^2))", NULL},
	{"atan via acos", "sign(U)*acos(1/sqrt(1 + U^2))", NULL},
	{"atan reflection", "pi/2 - atan(1/U)", "U > 0"},
	{"atan reflection", "-pi/2 - atan(1/U)", "U < 0"},
	{"atan half-angle", "2atan(U/(1 + sqrt(1 + U^2)))", NULL},
	{"atan as atan2", "atan2(U, 1)", NULL},
	{NULL, NULL, NULL}
};
static const struct ialt alt_asin[] = {
	{"asin via atan", "atan(U/sqrt(1 - U^2))", "-1 < U < 1"},
	{"asin via acos", "pi/2 - acos(U)", NULL},
	{"asin via acos", "acos(sqrt(1 - U^2))", "0 <= U <= 1"},
	{"asin via acos", "-acos(sqrt(1 - U^2))", "-1 <= U <= 0"},
	{"asin half-angle", "2atan(U/(1 + sqrt(1 - U^2)))", NULL},
	{"asin as atan2", "atan2(U, sqrt(1 - U^2))", NULL},
	{NULL, NULL, NULL}
};
static const struct ialt alt_acos[] = {
	{"acos via asin", "pi/2 - asin(U)", NULL},
	{"acos via asin", "asin(sqrt(1 - U^2))", "0 <= U <= 1"},
	{"acos via asin", "2asin(sqrt((1 - U)/2))", NULL},
	{"acos via atan", "atan(sqrt(1 - U^2)/U)", "0 < U <= 1"},
	{"acos via atan", "pi + atan(sqrt(1 - U^2)/U)", "-1 <= U < 0"},
	{"acos half-angle", "2atan(sqrt((1 - U)/(1 + U)))", "-1 < U <= 1"},
	{"acos as atan2", "atan2(sqrt(1 - U^2), U)", NULL},
	{NULL, NULL, NULL}
};
static const struct ialt alt_atan2[] = {
	{"atan2 half-angle", "2atan(Y/(sqrt(X^2 + Y^2) + X))", "X > 0 || Y != 0"},
	{"atan2 via atan", "atan(Y/X)", "X > 0"},
	{"atan2 via atan", "atan(Y/X) + pi", "X < 0 && Y >= 0"},
	{"atan2 via atan", "atan(Y/X) - pi", "X < 0 && Y < 0"},
	{"atan2 via atan", "pi/2 - atan(X/Y)", "Y > 0"},
	{"atan2 via atan", "-pi/2 - atan(X/Y)", "Y < 0"},
	{NULL, NULL, NULL}
};
static const struct ialt alt_asinh[] = {
	{"asinh as a log", "log(U + sqrt(U^2 + 1))", NULL},
	{"asinh via atanh", "atanh(U/sqrt(1 + U^2))", NULL},
	{NULL, NULL, NULL}
};
static const struct ialt alt_acosh[] = {
	{"acosh as a log", "log(U + sqrt(U^2 - 1))", NULL},
	{"acosh via asinh", "asinh(sqrt(U^2 - 1))", NULL},
	{NULL, NULL, NULL}
};
static const struct ialt alt_atanh[] = {
	{"atanh as a log", "log((1 + U)/(1 - U))/2", NULL},
	{"atanh via asinh", "asinh(U/sqrt(1 - U^2))", NULL},
	{NULL, NULL, NULL}
};

/* U, Y, X replaced by the (parenthesised) arguments */
static void
fill(struct sbuf *b, const char *form, const char *u, const char *x)
{
	const char *s;

	for (s = form; *s; s++) {
		if (*s == 'U' || *s == 'Y')
			sb_printf(b, "%s", u);
		else if (*s == 'X')
			sb_printf(b, "%s", x);
		else
			sb_putc(b, *s);
	}
}

static void
collect_inv(nid e, uint8_t *seen, nid *out, int *n)
{
	int op = N.op[e];

	if (seen[e] || *n >= 6)
		return;
	seen[e] = 1;
	if (op == O_ATAN || op == O_ASIN || op == O_ACOS || op == O_ATAN2 ||
	    op == O_ASINH || op == O_ACOSH || op == O_ATANH)
		out[(*n)++] = e;
	if (op == O_TABLE || op == O_LOOP)
		return;
	if (opinfo[op].arity >= 1)
		collect_inv(N.a[e], seen, out, n);
	if (opinfo[op].arity >= 2)
		collect_inv(N.b[e], seen, out, n);
	if (opinfo[op].arity >= 3)
		collect_inv(N.c[e], seen, out, n);
}

static const char *
S_fn(int op)
{
	static char b[32];

	snprintf(b, sizeof b, "%s(u)", opinfo[op].name);
	return b;
}

static void
inverse_cands(const struct inp *in, nid body)
{
	nid atoms[6];
	uint8_t *seen = xcalloc(N.n, 1);
	int n = 0, k, j;

	collect_inv(body, seen, atoms, &n);
	free(seen);
	for (k = 0; k < n; k++) {
		nid f = atoms[k];
		const struct ialt *t;
		char u[300], x[300], id[200];
		switch (N.op[f]) {
		case O_ATAN: t = alt_atan; break;
		case O_ASIN: t = alt_asin; break;
		case O_ACOS: t = alt_acos; break;
		case O_ATAN2: t = alt_atan2; break;
		case O_ASINH: t = alt_asinh; break;
		case O_ACOSH: t = alt_acosh; break;
		default: t = alt_atanh; break;
		}
		snprintf(u, sizeof u, "(%s)", pstr(N.a[f]));
		if (N.op[N.a[f]] == O_VAR)
			snprintf(u, sizeof u, "%s", pstr(N.a[f]));
		x[0] = 0;
		if (N.op[f] == O_ATAN2) {
			snprintf(x, sizeof x, "(%s)", pstr(N.b[f]));
			if (N.op[N.b[f]] == O_VAR)
				snprintf(x, sizeof x, "%s", pstr(N.b[f]));
		}
		for (j = 0; t[j].label; j++) {
			struct sbuf fs = {0}, cs = {0}, gen = {0};
			nid rep, whole;
			fill(&fs, t[j].form, u, x);
			if (t[j].cond)
				fill(&cs, t[j].cond, u, x);
			rep = parse_alt(fs.p, in);
			if (rep) {
				inv_from = f;
				inv_to = rep;
				whole = rewrite(body, rule_subst);
				/* the identity in general terms */
				sb_printf(&gen, "%s", N.op[f] == O_ATAN2 ? "atan2(y, x)" :
				    S_fn(N.op[f]));
				sb_puts(&gen, " = ");
				fill(&gen, t[j].form, N.op[f] == O_ATAN2 ? "y" : "u", "x");
				snprintf(id, sizeof id, "%s", gen.p);
				add_cand(t[j].label, id, whole, cs.p);
			}
			free(fs.p);
			free(cs.p);
			free(gen.p);
		}
	}
}

/* ---- one fraction: a/b + c/d -> (ad + bc)/(bd) ---- */

static int
p_equal_scaled(struct poly *a, struct poly *b, double *k)
{
	int i;

	if (a->n != b->n || a->n == 0)
		return 0;
	p_sort(a);
	p_sort(b);
	*k = a->t[0].c / b->t[0].c;
	for (i = 0; i < a->n; i++) {
		struct term t = b->t[i];
		t.c = a->t[i].c;
		if (!t_same(&a->t[i], &t) || !near(a->t[i].c, *k * b->t[i].c))
			return 0;
	}
	return 1;
}

static void
tog(nid e, struct poly *n, struct poly *d)
{
	struct poly n1, d1, n2, d2, t1, t2;
	int64_t k;
	int op = N.op[e], j;
	double c;

	switch (op) {
	case O_NEG:
		tog(N.a[e], &n1, d);
		memset(n, 0, sizeof *n);
		p_add(n, &n1, -1);
		p_free(&n1);
		return;
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_DIV:
		tog(N.a[e], &n1, &d1);
		tog(N.b[e], &n2, &d2);
		if (op == O_MUL) {
			p_mul(n, &n1, &n2);
			p_mul(d, &d1, &d2);
		} else if (op == O_DIV) {
			p_mul(n, &n1, &d2);
			p_mul(d, &d1, &n2);
		} else if (p_equal_scaled(&d1, &d2, &c) && near(c, 1)) {
			p_copy(n, &n1);
			p_add(n, &n2, op == O_SUB ? -1 : 1);
			p_copy(d, &d1);
		} else {
			p_mul(&t1, &n1, &d2);
			p_mul(&t2, &n2, &d1);
			p_copy(n, &t1);
			p_add(n, &t2, op == O_SUB ? -1 : 1);
			p_mul(d, &d1, &d2);
			p_free(&t1);
			p_free(&t2);
		}
		p_free(&n1);
		p_free(&d1);
		p_free(&n2);
		p_free(&d2);
		p_pyth(n);
		p_pyth(d);
		return;
	case O_POW:
		if (int_of(N.b[e], &k) && llabs(k) <= 8) {
			tog(N.a[e], &n1, &d1);
			p_const(n, 1);
			p_const(d, 1);
			for (j = 0; j < llabs(k); j++) {
				p_mul(&t1, n, k > 0 ? &n1 : &d1);
				p_mul(&t2, d, k > 0 ? &d1 : &n1);
				p_free(n);
				p_free(d);
				*n = t1;
				*d = t2;
			}
			p_free(&n1);
			p_free(&d1);
			return;
		}
	}
	lc_of(e, n);
	p_const(d, 1);
}

static nid
together(nid e)
{
	struct poly n, d;
	double c;
	nid r;

	tog(e, &n, &d);
	if (p_isconst(&d, &c) && c != 0) {
		struct poly m;
		memset(&m, 0, sizeof m);
		p_add(&m, &n, 1 / c);
		r = lc_build(&m);
		p_free(&m);
	} else if (p_equal_scaled(&n, &d, &c))
		r = mk_const_r(c);
	else
		r = norm(ndiv(lc_build(&n), lc_build(&d)));
	p_free(&n);
	p_free(&d);
	return r;
}

/* sin(u/2) = sqrt((1 - cos u)/2) where sin(u/2) >= 0 ... */
static nid
rule_halfexp(nid e, nid a, nid b)
{
	int op = N.op[e], k, half = 0;
	struct ang A;
	nid v, one_ = ki(1), two = ki(2);
	char c[300];

	if (op != O_SIN && op != O_COS && op != O_TAN)
		return 0;
	if (!ang_of(a, &A) || A.na == 0)
		return 0;
	for (k = 0; k < A.na; k++) {
		double d = 2 * A.k[k];
		if (!near(d, floor(d + 0.5)))
			return 0;
		if (!near(A.k[k], floor(A.k[k] + 0.5)))
			half = 1;
	}
	if (!half)
		return 0;
	v = norm(nmul(two, a));
	if (op == O_TAN)
		return ndiv(mk_un(O_SIN, v), nadd(one_, mk_un(O_COS, v)));
	snprintf(c, sizeof c, "%s(%s) >= 0", op == O_SIN ? "sin" : "cos", pstr(a));
	rcond(c);
	return mk_un(O_SQRT, ndiv(op == O_SIN ? nsub(one_, mk_un(O_COS, v)) :
	    nadd(one_, mk_un(O_COS, v)), two));
}

/* atan a + atan b = atan((a + b)/(1 - a b)) for a b < 1 */
static void
atan_sum(nid body)
{
	struct poly p;
	nid a, b, r;
	int neg;
	char c[400];

	lc_of(body, &p);
	if (p.n == 2 && p.t[0].nf == 1 && p.t[1].nf == 1 &&
	    p.t[0].p[0] == 1 && p.t[1].p[0] == 1 &&
	    N.op[p.t[0].f[0]] == O_ATAN && N.op[p.t[1].f[0]] == O_ATAN &&
	    near(fabs(p.t[0].c), fabs(p.t[1].c))) {
		a = N.a[p.t[0].f[0]];
		b = N.a[p.t[1].f[0]];
		neg = (p.t[0].c < 0) != (p.t[1].c < 0);
		r = mk_un(O_ATAN, norm(ndiv(neg ? nsub(a, b) : nadd(a, b),
		    neg ? nadd(K(1), nmul(a, b)) : nsub(K(1), nmul(a, b)))));
		r = nmul(mk_const_r(p.t[0].c), r);
		snprintf(c, sizeof c, "%s %s %s", pstr(norm(nmul(a, b))),
		    neg ? ">" : "<", neg ? "-1" : "1");
		add_cand("atan addition", neg ? "atan a - atan b = atan((a - b)/"
		    "(1 + a b)) for a b > -1" : "atan a + atan b = atan((a + b)/"
		    "(1 - a b)) for a b < 1", r, c);
	}
	p_free(&p);
}

static const char *const mode_words[] = {"simplify", "sincos", "exp",
	"reduce", "product", "sum", "expand", "harmonic", "half", "inverse",
	"all", "double", "euler", "power", "weierstrass", NULL};
static const int mode_bits[] = {M_SIMP, M_SINCOS, M_EXP, M_REDUCE,
	M_PRODUCT, M_SUM, M_EXPAND, M_HARM, M_HALF | M_TSUB, M_INV, M_ALL,
	M_EXPAND | M_REDUCE, M_EXP, M_REDUCE, M_TSUB};

/* simpler: shorter, with fewer function calls (each counts 6) */
static size_t
score(const char *s)
{
	size_t n = strlen(s), k;

	for (k = 1; s[k]; k++)
		if (s[k] == '(' && isalpha((unsigned char)s[k - 1]))
			n += 6;
	return n;
}

static void
cmd_trig(const char *text, int modes)
{
	struct inp in;
	nid body, x, sc;
	struct chk ck;
	int k, pw, prod, best = -1, anyexp = 0;
	char lhs[1100];
	size_t blen;

	lc_conds[0] = 0;
	ncand = 0;
	get_input(text, &in, 1);
	real_note(&in, 0);
	notes_flush();
	body = in.body;
	snprintf(lhs, sizeof lhs, "%s", lhs_text(&in));
	printf("trig forms of %s\n", lhs);

	if (modes & M_SIMP) {
		rw_cond[0] = 0;
		rw_hit = 0;
		x = norm(rewrite(body, rule_compose));
		add_cand("simplified", "sin^2 + cos^2 = 1, cosh^2 - sinh^2 = 1, "
		    "f(f^-1(u)) = u", x, rw_cond[0] ? rw_cond : NULL);
		/* everything else starts from the simpler form */
		if (modes == M_ALL && !rw_cond[0] && (rw_hit ||
		    strlen(pstr(x)) < strlen(pstr(body))))
			body = x;
	}
	rw_hit = 0;
	sc = rewrite(body, rule_sincos);
	if ((modes & M_SINCOS) && rw_hit)
		add_cand("in sin and cos", "tan = sin/cos, sec = 1/cos, csc = 1/sin, "
		    "cot = cos/sin, tanh = sinh/cosh", norm(sc), NULL);
	if (modes & (M_REDUCE | M_PRODUCT)) {
		x = linearize(sc, &pw, &prod);
		if (x && ((pw && (modes & M_REDUCE)) || (prod && (modes &
		    M_PRODUCT)))) {
			int tr = uses(sc, O_SIN) || uses(sc, O_COS);
			const char *id = pw && !prod ? (tr ? "sin^2 u = (1 - cos 2u)/2, "
			    "cos^2 u = (1 + cos 2u)/2 (higher powers through e^(iu) = "
			    "cos u + i sin u)" : "sinh^2 u = (cosh 2u - 1)/2, cosh^2 u = "
			    "(cosh 2u + 1)/2") : tr ? "sin a cos b = (sin(a + b) + "
			    "sin(a - b))/2, cos a cos b = (cos(a - b) + cos(a + b))/2, "
			    "sin a sin b = (cos(a - b) - cos(a + b))/2" :
			    "sinh a cosh b = (sinh(a + b) + sinh(a - b))/2, cosh a cosh b"
			    " = (cosh(a + b) + cosh(a - b))/2";
			add_cand(pw && prod ? "power reduction, product to sum" :
			    pw ? "power reduction" : "product to sum", id, x, NULL);
		}
	}
	if (modes & M_EXPAND) {
		uint8_t *seen = xcalloc(N.n, 1);
		memset(&bases, 0, sizeof bases);
		collect_bases(body, seen);
		free(seen);
		if (bases.n) {
			nid ex;
			struct poly p, q;
			int j;
			rw_cond[0] = 0;
			ex = norm(rewrite(body, rule_expand));
			{
				char id[200] = "";
				if (uses(body, O_SIN) || uses(body, O_COS))
					strcat(id, "sin(a + b) = sin a cos b + cos a sin b, "
					    "cos(a + b) = cos a cos b - sin a sin b, ");
				if (uses(body, O_TAN))
					strcat(id, "tan(a + b) = (tan a + tan b)/(1 - tan a "
					    "tan b), ");
				if (uses(body, O_SINH) || uses(body, O_COSH) ||
				    uses(body, O_TANH))
					strcat(id, "sinh(a + b) = sinh a cosh b + cosh a sinh b, "
					    "cosh(a + b) = cosh a cosh b + sinh a sinh b, ");
				if (strlen(id) > 2)
					id[strlen(id) - 2] = 0;
				add_cand("compound and multiple angles", id, ex,
				    rw_cond[0] ? rw_cond : NULL);
			}
			/* in sin only / in cos only, base by base */
			lc_of(ex, &p);
			for (j = 0; j < bases.n; j++) {
				nid ba = base_arg(j);
				nid c = mk_un(O_COS, ba), s = mk_un(O_SIN, ba);
				nid ch = mk_un(O_COSH, ba), sh = mk_un(O_SINH, ba);
				if (p_subst_sq(&p, c, s, 0, &q)) {
					add_cand("compound angles, in sin", "cos^2 u = 1 - "
					    "sin^2 u", lc_build(&q), rw_cond[0] ? rw_cond : NULL);
					p_free(&q);
				}
				if (p_subst_sq(&p, s, c, 0, &q)) {
					add_cand("compound angles, in cos", "sin^2 u = 1 - "
					    "cos^2 u", lc_build(&q), rw_cond[0] ? rw_cond : NULL);
					p_free(&q);
				}
				if (p_subst_sq(&p, ch, sh, 1, &q)) {
					add_cand("compound angles, in sinh", "cosh^2 u = 1 + "
					    "sinh^2 u", lc_build(&q), NULL);
					p_free(&q);
				}
			}
			p_free(&p);
		}
	}
	if (modes & M_SUM) {
		x = sum_to_product(body);
		if (x)
			add_cand("sum to product", "sin a + sin b = 2 sin((a + b)/2) "
			    "cos((a - b)/2), cos a + cos b = 2 cos((a + b)/2) "
			    "cos((a - b)/2) ...", x, NULL);
	}
	if (modes & M_HARM) {
		nid f1, f2;
		char cond[300];
		int got = harmonic(body, &f1, &f2, cond, sizeof cond);
		if (!got) {
			x = linearize(sc, &pw, &prod);
			if (x)
				got = harmonic(x, &f1, &f2, cond, sizeof cond);
		}
		if (got) {
			add_cand("amplitude-phase", "a sin u + b cos u = R sin(u + phi), "
			    "R = sqrt(a^2 + b^2), phi = atan2(b, a)", f1,
			    cond[0] ? cond : NULL);
			add_cand("amplitude-phase", "a sin u + b cos u = R cos(u - psi), "
			    "psi = atan2(a, b)", f2, cond[0] ? cond : NULL);
		}
	}
	if (modes & M_TSUB) {
		rw_cond[0] = 0;
		rw_hit = 0;
		x = rewrite(body, rule_half);
		if (rw_hit)
			add_cand("tan half-angle", "t = tan(u/2): sin u = 2t/(1 + t^2), "
			    "cos u = (1 - t^2)/(1 + t^2), tan u = 2t/(1 - t^2)", together(x),
			    rw_cond);
	}
	if (modes & M_HALF) {
		rw_cond[0] = 0;
		rw_hit = 0;
		x = rewrite(body, rule_halfexp);
		if (rw_hit)
			add_cand("half angle", "sin(u/2) = sqrt((1 - cos u)/2) where "
			    "sin(u/2) >= 0, cos(u/2) = sqrt((1 + cos u)/2) where "
			    "cos(u/2) >= 0, tan(u/2) = sin u/(1 + cos u)", x,
			    rw_cond[0] ? rw_cond : NULL);
	}
	if (modes & M_INV) {
		inverse_cands(&in, body);
		atan_sum(body);
	}
	if (modes & M_EXP) {
		rw_hit = 0;
		x = rewrite(body, rule_exp);
		if (rw_hit) {
			static const struct {
				int op;
				const char *id;
			} ids[] = {
				{O_SIN, "sin u = (e^(iu) - e^(-iu))/(2i)"},
				{O_COS, "cos u = (e^(iu) + e^(-iu))/2"},
				{O_TAN, "tan u = (e^(iu) - e^(-iu))/(i(e^(iu) + e^(-iu)))"},
				{O_SEC, "sec u = 2/(e^(iu) + e^(-iu))"},
				{O_CSC, "csc u = 2i/(e^(iu) - e^(-iu))"},
				{O_COT, "cot u = i(e^(iu) + e^(-iu))/(e^(iu) - e^(-iu))"},
				{O_SINH, "sinh u = (e^u - e^(-u))/2"},
				{O_COSH, "cosh u = (e^u + e^(-u))/2"},
				{O_TANH, "tanh u = (e^u - e^(-u))/(e^u + e^(-u))"},
				{O_ASIN, "asin u = -i log(iu + sqrt(1 - u^2))"},
				{O_ACOS, "acos u = -i log(u + i sqrt(1 - u^2))"},
				{O_ATAN, "atan u = (i/2) log((1 - iu)/(1 + iu))"},
				{O_ASINH, "asinh u = log(u + sqrt(u^2 + 1))"},
				{O_ACOSH, "acosh u = log(u + sqrt(u^2 - 1))"},
				{O_ATANH, "atanh u = log((1 + u)/(1 - u))/2"},
			};
			char id[200] = "";
			int j;
			for (j = 0; j < (int)(sizeof ids / sizeof ids[0]); j++)
				if (uses(body, ids[j].op) && strlen(id) +
				    strlen(ids[j].id) + 3 < sizeof id) {
					if (id[0])
						strcat(id, ", ");
					strcat(id, ids[j].id);
				}
			add_cand(uses(body, O_ASIN) || uses(body, O_ACOS) ||
			    uses(body, O_ATAN) || uses(body, O_ASINH) ||
			    uses(body, O_ACOSH) || uses(body, O_ATANH) ?
			    "logarithmic and exponential (Euler)" : "exponential (Euler)",
			    id, x, NULL);
			anyexp = 1;
		}
	}
	if (ncand == 0 || (ncand == 1 && !strcmp(cands[0].text, pstr(in.body)))) {
		printf("  no rewrite of this kind applies (try: trig F all)\n");
		return;
	}
	blen = score(pstr(in.body));
	for (k = 0; k < ncand; k++) {
		struct cand *c = &cands[k];
		if (!strcmp(c->text, pstr(in.body)) && !c->cond[0])
			continue;	/* nothing changed */
		printf("  %s: %s\n", c->label, c->ident);
		printf("    = %s\n", c->text);
		if (c->cond[0])
			printf("    for %s\n", c->cond);
		check_text(&in, 1, c->text, c->cond[0] ? c->cond : NULL, &ck);
		c->ok = print_check(&in, &ck, c->cond[0] ? c->cond : NULL, "    ");
		if (c->ok && !c->cond[0] && score(c->text) < blen &&
		    (best < 0 || score(c->text) < score(cands[best].text)))
			best = k;
	}
	(void)anyexp;
	if (best >= 0)
		printf("  simplest: %s  (%s)\n", cands[best].text, cands[best].label);
	else
		printf("  simplest: %s  (as given)\n", lhs);
}

/* ---- the commands ---- */

static int
has_assign(const char *s)
{
	int d = 0;

	for (; *s; s++) {
		if (*s == '(' || *s == '[')
			d++;
		else if (*s == ')' || *s == ']')
			d--;
		else if (*s == '=' && d == 0 && s[1] != '=' && (s[-1] != '=' &&
		    s[-1] != '<' && s[-1] != '>' && s[-1] != '!'))
			return 1;
	}
	return 0;
}

int
forms_command(struct P *p)
{
	static const char *const words[] = {"complex", "rect", "polar", "conj",
		"re", "im", "modulus", "arg", "nthroots", "unity", "trig", NULL};
	char w[32], rest[4096];
	uint32_t at = cmdp_pos(p);
	int k, n = 0, modes = 0;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; words[k]; k++)
		if (!strcmp(words[k], w))
			break;
	if (!words[k] || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	rest[0] = 0;
	if (!cmdp_eos(p))
		cmdp_rest(p, rest, sizeof rest, NULL);
	if (has_assign(rest)) {
		cmdp_seek(p, at);
		return 0;
	}
	notes_flush();
	lc_reset();	/* the memos depend on which slots are i */
	z_reset();
	if (!strcmp(w, "nthroots") || !strcmp(w, "unity")) {
		char *s = rest, *e;
		while (isspace((unsigned char)*s))
			s++;
		n = (int)strtol(s, &e, 10);
		if (e == s)
			fail("%s needs a count: %s N%s", w, w,
			    w[0] == 'n' ? " EXPR" : "");
		s = e;
		while (isspace((unsigned char)*s))
			s++;
		if (!strncmp(s, "of ", 3))
			s += 3;
		if (w[0] == 'u') {
			if (*s)
				fail("unity N: the N-th roots of 1");
			cmd_nthroots(n, "1");
		} else
			cmd_nthroots(n, s);
		lc_reset();
		z_reset();
		return 1;
	}
	if (!strcmp(w, "trig")) {
		/* mode words at the end */
		for (;;) {
			size_t len = strlen(rest);
			char *sp;
			int j;
			while (len && isspace((unsigned char)rest[len - 1]))
				rest[--len] = 0;
			sp = strrchr(rest, ' ');
			if (!sp)
				break;
			for (j = 0; mode_words[j]; j++)
				if (!strcmp(sp + 1, mode_words[j]))
					break;
			if (!mode_words[j])
				break;
			modes |= mode_bits[j];
			*sp = 0;
		}
		cmd_trig(rest, modes ? modes : M_ALL);
	} else if (!strcmp(w, "complex") || !strcmp(w, "rect"))
		cmd_cform(rest, FC_RECT);
	else if (!strcmp(w, "polar"))
		cmd_cform(rest, FC_POLAR);
	else if (!strcmp(w, "conj"))
		cmd_cform(rest, FC_CONJ);
	else if (!strcmp(w, "re"))
		cmd_cform(rest, FC_RE);
	else if (!strcmp(w, "im"))
		cmd_cform(rest, FC_IM);
	else if (!strcmp(w, "modulus"))
		cmd_cform(rest, FC_MOD);
	else
		cmd_cform(rest, FC_ARG);
	lc_reset();
	z_reset();
	return 1;
}
