/*
 * xfrat.c: the shared machinery of the transforms (xform.h): numbers as
 * text, a printer for results, polynomials and rational functions with
 * double coefficients, partial fractions, the input rewriting and
 * evaluation helpers.
 *
 * Coefficients are doubles.  Rational and quadratic-surd roots come out
 * exact (cnum.c's poly_roots on integer polynomials), so the printed
 * results are exact where the input was; everything printed is also
 * checked numerically by the commands that use it.
 */
#include "xform.h"
#include "calc2.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int xf_discrete;

/* ---- numbers ---- */

static char *
rbuf(void)
{
	static char buf[32][96];
	static int k;

	return buf[k++ & 31];
}

static int
near_int(double v, double *r)
{
	double n = nearbyint(v);

	if (fabs(v - n) <= 1e-11 * fmax(1, fabs(v)) && fabs(v) < 1e15) {
		*r = n;
		return 1;
	}
	return 0;
}

/* v = n/d with small n, d */
static int
small_q(double v, int64_t maxd, int64_t maxn, int64_t *pn, int64_t *pd)
{
	struct q q;
	double r;

	if (near_int(v, &r) && fabs(r) <= (double)maxn) {
		*pn = (int64_t)r;
		*pd = 1;
		return 1;
	}
	if (!q_from_double(v, &q) || q.d > maxd || llabs(q.n) > maxn)
		return 0;
	if (fabs((double)q.n / (double)q.d - v) > 1e-11 * fabs(v))
		return 0;
	*pn = q.n;
	*pd = q.d;
	return 1;
}

/* the bases a constant may be a rational multiple of */
static const struct {
	const char *name;
	double v;
	int64_t maxd, maxn;
} bases[] = {
	{"pi", M_PI, 360, 2000},
	{"pi^2", M_PI * M_PI, 2000, 2000},
	{"pi^3", M_PI * M_PI * M_PI, 2000, 500},
	{"pi^4", M_PI * M_PI * M_PI * M_PI, 10000, 500},
	{"pi^6", M_PI * M_PI * M_PI * M_PI * M_PI * M_PI, 1000000, 100},
	{"sqrt(2)", 1.41421356237309504880, 100, 200},
	{"sqrt(3)", 1.73205080756887729353, 100, 200},
	{"sqrt(5)", 2.23606797749978969641, 100, 200},
	{"sqrt(6)", 2.44948974278317809820, 100, 100},
	{"sqrt(7)", 2.64575131106459059050, 100, 100},
	{"sqrt(10)", 3.16227766016837933200, 100, 100},
	{"sqrt(pi)", 1.77245385090551602730, 100, 100},
	{"log(2)", 0.69314718055994530942, 100, 100},
	{"log(3)", 1.09861228866810969140, 100, 100},
	{"e", M_E, 100, 100},
	{"pi*sqrt(3)", M_PI * 1.73205080756887729353, 100, 100},
	{"pi*sqrt(2)", M_PI * 1.41421356237309504880, 100, 100},
	{NULL, 0, 0, 0}
};

/*
 * v = (n/d) * base: *bn the base's name or "" for a rational; 0 when v
 * is none of those
 */
static int
num_parts(double v, int64_t *n, int64_t *d, const char **bn)
{
	int k;

	*bn = "";
	if (small_q(v, 1000, 100000000, n, d))
		return 1;
	for (k = 0; bases[k].name; k++)
		if (small_q(v / bases[k].v, bases[k].maxd, bases[k].maxn, n, d)) {
			*bn = bases[k].name;
			return 1;
		}
	/* 1/pi, 1/sqrt(pi), 1/pi^2: small numerators only */
	{
		static const struct {
			const char *name;
			double v;
		} inv[] = {{"/pi", M_PI}, {"/sqrt(pi)", 1.77245385090551602730},
			{"/pi^2", M_PI * M_PI}, {"/e", M_E}, {NULL, 0}};
		for (k = 0; inv[k].name; k++)
			if (small_q(v * inv[k].v, 100, 100, n, d) && *n != 0) {
				*bn = inv[k].name;
				return 1;
			}
	}
	return 0;
}

static const char *
parts_text(int64_t n, int64_t d, const char *bn)
{
	char *b = rbuf();

	if (!bn[0]) {
		if (d == 1)
			snprintf(b, 96, "%lld", (long long)n);
		else
			snprintf(b, 96, "%lld/%lld", (long long)n, (long long)d);
		return b;
	}
	if (bn[0] == '/') {
		if (d == 1)
			snprintf(b, 96, "%lld%s", (long long)n, bn);
		else
			snprintf(b, 96, "%lld/(%lld%s)", (long long)n, (long long)d,
			    bn + 1);
		return b;
	}
	{
		char nb[48];
		if (n == 1)
			snprintf(nb, sizeof nb, "%s", bn);
		else if (n == -1)
			snprintf(nb, sizeof nb, "-%s", bn);
		else if (isalpha((unsigned char)bn[0]))
			snprintf(nb, sizeof nb, "%lld%s", (long long)n, bn);
		else
			snprintf(nb, sizeof nb, "%lld*%s", (long long)n, bn);
		if (d == 1)
			snprintf(b, 96, "%s", nb);
		else
			snprintf(b, 96, "%s/%lld", nb, (long long)d);
	}
	return b;
}

const char *
xf_num(double v)
{
	int64_t n, d;
	const char *bn;
	char *b;

	if (isnan(v))
		return "undefined";
	if (isinf(v))
		return v > 0 ? "inf" : "-inf";
	if (v == 0 || fabs(v) < 1e-300)
		return "0";
	if (num_parts(v, &n, &d, &bn))
		return parts_text(n, d, bn);
	{
		/* (a + b sqrt(m))/d */
		static const int ms[] = {2, 3, 5, 6, 7, 10, 0};
		int mi, dd, bb;
		for (mi = 0; ms[mi]; mi++)
			for (dd = 1; dd <= 12; dd++)
				for (bb = -12; bb <= 12; bb++) {
					double a, r;
					if (bb == 0)
						continue;
					a = v * dd - bb * sqrt((double)ms[mi]);
					if (!near_int(a, &r) || fabs(r) > 200 || r == 0)
						continue;
					if (fabs(r + bb * sqrt((double)ms[mi]) - v * dd) >
					    1e-11 * fmax(1, fabs(v * dd)))
						continue;
					{
						int64_t g = gcd64(gcd64((int64_t)fabs(r), abs(bb)), dd);
						char *o = rbuf();
						int64_t A = (int64_t)r / g, B = bb / g, D = dd / g;
						char sb[32];
						if (B == 1 || B == -1)
							snprintf(sb, sizeof sb, "sqrt(%d)", ms[mi]);
						else
							snprintf(sb, sizeof sb, "%lldsqrt(%d)",
							    (long long)(B < 0 ? -B : B), ms[mi]);
						if (D == 1)
							snprintf(o, 96, "%lld %c %s", (long long)A,
							    B < 0 ? '-' : '+', sb);
						else
							snprintf(o, 96, "(%lld %c %s)/%lld", (long long)A,
							    B < 0 ? '-' : '+', sb, (long long)D);
						return o;
					}
				}
	}
	b = rbuf();
	snprintf(b, 96, "%.10g", v);
	return b;
}

const char *
xf_cnum(double complex z)
{
	char *b = rbuf();
	double re = creal(z), im = cimag(z);
	double sc = fmax(fabs(re), fabs(im));

	if (fabs(im) <= 1e-12 * sc)
		im = 0;
	if (fabs(re) <= 1e-12 * sc)
		re = 0;
	if (im == 0)
		return xf_num(re);
	{
		const char *is = xf_num(fabs(im));
		char ib[64];
		if (!strcmp(is, "1"))
			snprintf(ib, sizeof ib, "i");
		else if (strchr(is, '/')) {
			/* 1/2 -> i/2 */
			const char *sl = strchr(is, '/');
			if (sl - is == 1 && is[0] == '1')
				snprintf(ib, sizeof ib, "i%s", sl);
			else
				snprintf(ib, sizeof ib, "%.*si%s", (int)(sl - is), is, sl);
		} else if (isdigit((unsigned char)is[strlen(is) - 1]) &&
		    !strchr(is, 'e'))
			snprintf(ib, sizeof ib, "%si", is);
		else
			snprintf(ib, sizeof ib, "%s*i", is);
		if (re == 0)
			snprintf(b, 96, "%s%s", im < 0 ? "-" : "", ib);
		else
			snprintf(b, 96, "%s %s %s", xf_num(re), im < 0 ? "-" : "+", ib);
	}
	return b;
}

/* ---- nodes ---- */

nid
xk(double v)
{
	if (v == 0)
		v = 0;   /* no -0 */
	return mk_const_r(v);
}

int
xnum(nid e, double *v)
{
	if (e == 0)
		return 0;
	if (N.op[e] == O_CAST)
		return xnum(N.a[e], v);
	return num_of(e, v);
}

static nid
real(nid a)
{
	return TY_FP(N.ty[a]) ? a : mk_cast(TY_REAL, a);
}

nid
xadd(nid a, nid b)
{
	double x, y;

	a = real(a);
	b = real(b);
	if (xnum(a, &x) && x == 0)
		return b;
	if (xnum(b, &y) && y == 0)
		return a;
	if (xnum(b, &y) && y < 0)
		return mk_bin(O_SUB, a, xk(-y));
	if (N.op[b] == O_NEG)
		return mk_bin(O_SUB, a, N.a[b]);
	return mk_bin(O_ADD, a, b);
}

nid
xsub(nid a, nid b)
{
	double x, y;

	a = real(a);
	b = real(b);
	if (xnum(b, &y) && y == 0)
		return a;
	if (xnum(a, &x) && x == 0)
		return xneg(b);
	if (xnum(b, &y) && y < 0)
		return mk_bin(O_ADD, a, xk(-y));
	if (N.op[b] == O_NEG)
		return mk_bin(O_ADD, a, N.a[b]);
	if (a == b)
		return xk(0);
	return mk_bin(O_SUB, a, b);
}

nid
xneg(nid a)
{
	double x;

	a = real(a);
	if (xnum(a, &x))
		return xk(-x);
	if (N.op[a] == O_NEG)
		return N.a[a];
	if (N.op[a] == O_MUL && xnum(N.a[a], &x))
		return xmul(xk(-x), N.b[a]);
	return mk_un(O_NEG, a);
}

nid
xmul(nid a, nid b)
{
	double x, y;

	a = real(a);
	b = real(b);
	if (xnum(a, &x)) {
		if (x == 0)
			return xk(0);
		if (x == 1)
			return b;
		if (xnum(b, &y))
			return xk(x * y);
		if (x == -1)
			return xneg(b);
		if (N.op[b] == O_MUL && xnum(N.a[b], &y))
			return xmul(xk(x * y), N.b[b]);
		if (N.op[b] == O_NEG)
			return xmul(xk(-x), N.a[b]);
		return mk_bin(O_MUL, a, b);
	}
	if (xnum(b, &y))
		return xmul(b, a);
	if (N.op[a] == O_NEG)
		return xneg(xmul(N.a[a], b));
	if (N.op[b] == O_NEG)
		return xneg(xmul(a, N.a[b]));
	if (N.op[a] == O_MUL && xnum(N.a[a], &x))
		return xmul(N.a[a], xmul(N.b[a], b));
	if (N.op[b] == O_MUL && xnum(N.a[b], &y))
		return xmul(N.a[b], xmul(a, N.b[b]));
	return mk_bin(O_MUL, a, b);
}

nid
xdiv(nid a, nid b)
{
	double x, y;

	a = real(a);
	b = real(b);
	if (xnum(b, &y)) {
		if (y == 1)
			return a;
		if (xnum(a, &x))
			return xk(x / y);
		return xmul(xk(1 / y), a);
	}
	if (xnum(a, &x) && x == 0)
		return xk(0);
	if (N.op[a] == O_NEG)
		return xneg(xdiv(N.a[a], b));
	return mk_bin(O_DIV, a, b);
}

nid
xpow(nid a, nid b)
{
	double x, y;

	a = real(a);
	b = real(b);
	if (xnum(b, &y)) {
		if (y == 0)
			return xk(1);
		if (y == 1)
			return a;
		if (xnum(a, &x))
			return xk(pow(x, y));
	}
	if (xnum(a, &x) && x == 1)
		return xk(1);
	return mk_bin(O_POW, a, b);
}

nid
xexp(nid u)
{
	double x;

	u = real(u);
	if (xnum(u, &x))
		return xk(exp(x));
	return mk_un(O_EXP, u);
}

nid
xheav(nid u)
{
	return mk_sel(mk_bin(O_GE, real(u), xk(0)), xk(1), xk(0));
}

nid
xkron(nid u)
{
	return mk_sel(mk_bin(O_EQ, real(u), xk(0)), xk(1), xk(0));
}

nid
xdirac(nid u)
{
	return mk_sel(mk_bin(O_EQ, real(u), xk(0)), xk(INFINITY), xk(0));
}

static nid
uncast(nid n)
{
	while (N.op[n] == O_CAST)
		n = N.a[n];
	return n;
}

/* SEL(cmp(x, 0), A, B) with the numbers given */
static int
sel_is(nid n, int cmp, double A, double B, nid *arg)
{
	double a, b, z;
	nid c;

	n = uncast(n);
	if (N.op[n] != O_SEL)
		return 0;
	c = N.a[n];
	if (!xnum(N.b[n], &a) || !xnum(N.c[n], &b))
		return 0;
	if (N.op[c] == cmp && xnum(N.b[c], &z) && z == 0 && a == A && b == B) {
		*arg = uncast(N.a[c]);
		return 1;
	}
	return 0;
}

int
xis_heav(nid n, nid *arg)
{
	return sel_is(n, O_GE, 1, 0, arg);
}

int
xis_kron(nid n, nid *arg)
{
	return sel_is(n, O_EQ, 1, 0, arg);
}

int
xis_dirac(nid n, nid *arg)
{
	return sel_is(n, O_EQ, INFINITY, 0, arg);
}

#define FSLOT 60   /* the factorial loop's index; 61 its product */

nid
xfact(nid m)
{
	struct loopdef d;
	nid iv, s;
	double v;

	if (xnum(m, &v) && v >= 0 && v <= 170 && v == floor(v)) {
		double r = 1;
		int k;
		for (k = 2; k <= (int)v; k++)
			r *= k;
		return xk(r);
	}
	memset(&d, 0, sizeof d);
	d.kind = LK_FOR;
	d.orig = LO_PROD;
	d.name = sym_cstr("j");
	d.islot = FSLOT;
	d.ity = TY_INT;
	d.nst = 1;
	d.sslot[0] = FSLOT + 1;
	d.sname[0] = sym_cstr("prod");
	d.sty[0] = TY_REAL;
	d.init[0] = xk(1);
	iv = var_of(FSLOT, d.name, TY_INT);
	s = var_of(FSLOT + 1, d.sname[0], TY_REAL);
	d.next[0] = mk_bin(O_MUL, s, mk_cast(TY_REAL, iv));
	d.aux = mk_cast(TY_REAL, iv);
	return mk_loop(loop_intern(&d), 0, KI(1), m, 0);
}

int
xis_fact(nid n, nid *arg)
{
	struct loopdef d;

	n = uncast(n);
	if (N.op[n] != O_LOOP)
		return 0;
	loop_get(loop_ent(n), &d);
	if (d.orig != LO_PROD || d.islot != FSLOT)
		return 0;
	*arg = uncast(N.b[n]);
	return 1;
}

int
xis_exp(nid n, nid *u)
{
	double c;

	n = uncast(n);
	if (N.op[n] == O_EXP) {
		*u = N.a[n];
		return 1;
	}
	if (N.op[n] == O_POW && xnum(N.a[n], &c) && c > 0 && !is_const(N.b[n])) {
		*u = c == M_E ? N.b[n] : xmul(xk(log(c)), N.b[n]);
		return 1;
	}
	return 0;
}

nid
xsubst1(nid e, int slot, nid v)
{
	nid map[NSLOT];

	memset(map, 0, sizeof map);
	map[slot] = v;
	return subst(e, map, NSLOT);
}

nid
xv(int slot, const char *name)
{
	return mk_var((uint32_t)slot, sym_cstr(name), TY_REAL);
}

/* ---- printing ---- */

enum { PS_SUM = 1, PS_PROD, PS_NEG, PS_POW, PS_ATOM };

static void pr(struct sbuf *b, nid n, int minp);

static void
pr_str(struct sbuf *b, nid n, int minp)
{
	struct sbuf t = {0};

	pr(&t, n, minp);
	sb_puts(b, sb_str(&t));
	free(t.p);
}

/* the text of a constant and its precedence */
static int
const_prec(const char *s)
{
	if (strstr(s, " + ") || strstr(s, " - "))
		return s[0] == '(' && strchr(s, '/') ? PS_PROD : PS_SUM;
	if (s[0] == '-')
		return PS_NEG;
	if (strchr(s, '/') || strchr(s, '*'))
		return PS_PROD;
	if (strchr(s, '^'))
		return PS_POW;
	if (isdigit((unsigned char)s[0]) && isalpha((unsigned char)s[strlen(s) - 1]))
		return PS_PROD;   /* 3pi */
	if (isdigit((unsigned char)s[0]) && strchr(s, '('))
		return PS_PROD;
	return PS_ATOM;
}

#define MAXF 32
struct flat {
	double k;
	nid num[MAXF], den[MAXF];
	int nn, nd;
};

static void
flatten(nid n, struct flat *f, int inv)
{
	double v;

	n = uncast(n);
	if (xnum(n, &v) && isfinite(v) && v != 0) {
		f->k = inv ? f->k / v : f->k * v;
		return;
	}
	switch (N.op[n]) {
	case O_NEG:
		f->k = -f->k;
		flatten(N.a[n], f, inv);
		return;
	case O_MUL:
		flatten(N.a[n], f, inv);
		flatten(N.b[n], f, inv);
		return;
	case O_DIV:
		flatten(N.a[n], f, inv);
		flatten(N.b[n], f, !inv);
		return;
	case O_POW:
		if (xnum(N.b[n], &v) && v < 0) {
			nid p = v == -1 ? N.a[n] : xpow(N.a[n], xk(-v));
			if (inv ? f->nn < MAXF : f->nd < MAXF) {
				if (inv)
					f->num[f->nn++] = p;
				else
					f->den[f->nd++] = p;
			}
			return;
		}
		break;
	}
	if (inv ? f->nd < MAXF : f->nn < MAXF) {
		if (inv)
			f->den[f->nd++] = n;
		else
			f->num[f->nn++] = n;
	}
}

static int
atomic_text(const char *s)
{
	return const_prec(s) == PS_ATOM;
}

/* factors joined: 2t, 3(s + 1), t*e^(-t) */
static void
pr_factors(struct sbuf *b, const char *coef, nid *fs, int n)
{
	int i, lastnum = 0;

	if (coef) {
		sb_puts(b, coef);
		lastnum = isdigit((unsigned char)coef[strlen(coef) - 1]) &&
		    atomic_text(coef);
		if (!lastnum && n)
			sb_putc(b, '*');
	}
	for (i = 0; i < n; i++) {
		struct sbuf t = {0};
		pr(&t, fs[i], PS_POW);
		if (i > 0 || (coef && !lastnum))
			if (i > 0)
				sb_putc(b, '*');
		/* 2t, 2(s + 1), 2sin(t), but 2*3, 2*e^t */
		if (i == 0 && coef && lastnum && (isdigit((unsigned char)t.p[0]) ||
		    t.p[0] == '.' || (t.p[0] == 'e' && t.p[1] == '^') ||
		    t.p[0] == '-'))
			sb_putc(b, '*');
		sb_puts(b, sb_str(&t));
		free(t.p);
	}
}

static void
pr_prod(struct sbuf *b, nid n, int minp)
{
	struct flat f;
	int64_t qn, qd;
	const char *bn, *coef = NULL, *dcoef = NULL;
	int neg, prec = PS_PROD;

	memset(&f, 0, sizeof f);
	f.k = 1;
	flatten(n, &f, 0);
	neg = f.k < 0;
	if (neg)
		f.k = -f.k;
	if (num_parts(f.k, &qn, &qd, &bn)) {
		if (!bn[0] || bn[0] != '/') {
			if (!bn[0])
				coef = qn == 1 ? NULL : parts_text(qn, 1, "");
			else
				coef = parts_text(qn, 1, bn);
			dcoef = qd == 1 ? NULL : parts_text(qd, 1, "");
		} else {
			coef = qn == 1 ? NULL : parts_text(qn, 1, "");
			dcoef = qd == 1 ? bn + 1 : parts_text(qd, 1, bn + 1);
		}
	} else if (f.k != 1)
		coef = xf_num(f.k);
	if (neg)
		prec = PS_NEG;
	if (!neg && !coef && !dcoef && f.nd == 0 && f.nn == 1) {
		pr(b, f.num[0], minp);
		return;
	}
	if (prec < minp)
		sb_putc(b, '(');
	if (neg)
		sb_putc(b, '-');
	if (f.nn == 0)
		sb_puts(b, coef ? coef : "1");
	else
		pr_factors(b, coef, f.num, f.nn);
	if (dcoef || f.nd) {
		int many = f.nd > 1 || (f.nd == 1 && dcoef);
		sb_putc(b, '/');
		if (f.nd == 0) {
			if (!atomic_text(dcoef))
				sb_printf(b, "(%s)", dcoef);
			else
				sb_puts(b, dcoef);
		} else if (many) {
			sb_putc(b, '(');
			pr_factors(b, dcoef, f.den, f.nd);
			sb_putc(b, ')');
		} else
			pr(b, f.den[0], PS_POW);
	}
	if (prec < minp)
		sb_putc(b, ')');
}

static void
pr_call(struct sbuf *b, const char *name, nid a)
{
	sb_printf(b, "%s(", name);
	pr(b, a, 0);
	sb_putc(b, ')');
}

static void
pr(struct sbuf *b, nid n, int minp)
{
	double v;
	nid arg, u;
	int op;

	n = uncast(n);
	op = N.op[n];
	if (xnum(n, &v)) {
		const char *s = xf_num(v);
		int p = const_prec(s);
		if (p < minp)
			sb_printf(b, "(%s)", s);
		else
			sb_puts(b, s);
		return;
	}
	if (xis_heav(n, &arg)) {
		if (xf_discrete) {
			sb_puts(b, "u[");
			pr(b, arg, 0);
			sb_putc(b, ']');
		} else
			pr_call(b, "u", arg);
		return;
	}
	if (xis_kron(n, &arg)) {
		sb_puts(b, "delta[");
		pr(b, arg, 0);
		sb_putc(b, ']');
		return;
	}
	if (xis_dirac(n, &arg)) {
		pr_call(b, "delta", arg);
		return;
	}
	if (xis_fact(n, &arg)) {
		pr(b, arg, PS_ATOM);
		sb_putc(b, '!');
		return;
	}
	if (xis_exp(n, &u) && (op == O_EXP || (N.op[n] == O_POW &&
	    xnum(N.a[n], &v) && v == M_E))) {
		struct sbuf t = {0};
		int p = PS_POW;
		pr(&t, u, PS_ATOM);
		if (p < minp)
			sb_putc(b, '(');
		sb_printf(b, "e^%s", sb_str(&t));
		if (p < minp)
			sb_putc(b, ')');
		free(t.p);
		return;
	}
	switch (op) {
	case O_VAR:
		sb_puts(b, sym_str(N.a[n]));
		return;
	case O_ADD:
	case O_SUB: {
		struct sbuf t = {0};
		if (PS_SUM < minp)
			sb_putc(b, '(');
		pr(b, N.a[n], PS_SUM);
		pr(&t, N.b[n], op == O_ADD ? PS_SUM : PS_PROD);
		if (t.p[0] == '-')
			sb_printf(b, " %c %s", op == O_ADD ? '-' : '+', t.p + 1);
		else
			sb_printf(b, " %c %s", op == O_ADD ? '+' : '-', t.p);
		free(t.p);
		if (PS_SUM < minp)
			sb_putc(b, ')');
		return;
	}
	case O_NEG:
	case O_MUL:
	case O_DIV:
		pr_prod(b, n, minp);
		return;
	case O_POW:
		if (xnum(N.b[n], &v) && v == 0.5) {
			pr_call(b, "sqrt", N.a[n]);
			return;
		}
		if (xnum(N.b[n], &v) && v < 0) {
			pr_prod(b, n, minp);
			return;
		}
		{
			struct sbuf t = {0};
			if (PS_POW < minp)
				sb_putc(b, '(');
			pr(b, N.a[n], PS_ATOM);
			pr(&t, N.b[n], PS_ATOM);
			sb_printf(b, "^%s", sb_str(&t));
			free(t.p);
			if (PS_POW < minp)
				sb_putc(b, ')');
		}
		return;
	case O_SEL:
		if (PS_SUM < minp)
			sb_putc(b, '(');
		sb_printf(b, "%s ? ", node_str(N.a[n]));
		pr(b, N.b[n], PS_SUM);
		sb_puts(b, " : ");
		pr(b, N.c[n], PS_SUM);
		if (PS_SUM < minp)
			sb_putc(b, ')');
		return;
	}
	if (op > O_UNDEF && op < O_NOPS && opinfo[op].arity == 1 &&
	    op != O_LOOP) {
		pr_call(b, opinfo[op].name, N.a[n]);
		return;
	}
	if (op > O_UNDEF && op < O_NOPS && opinfo[op].arity == 2 &&
	    (op == O_MIN || op == O_MAX || op == O_ATAN2 ||
	    op == O_COPYSIGN)) {
		sb_printf(b, "%s(", opinfo[op].name);
		pr(b, N.a[n], 0);
		sb_puts(b, ", ");
		pr(b, N.b[n], 0);
		sb_putc(b, ')');
		return;
	}
	if (minp > PS_SUM)
		sb_printf(b, "(%s)", node_str(n));
	else
		sb_puts(b, node_str(n));
}

void
xf_print(struct sbuf *b, nid n)
{
	pr(b, n, 0);
}

const char *
xf_str(nid n)
{
	static struct sbuf buf[16];
	static int k;
	struct sbuf *b = &buf[k++ & 15];

	sb_clear(b);
	pr(b, n, 0);
	return sb_str(b);
}

/* ---- evaluation ---- */

static nid
realroot(nid e)
{
	return TY_FP(N.ty[e]) ? e : mk_cast(TY_REAL, e);
}

double
xf_eval(nid e, const int *slots, const double *x, int nx, int *ok)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r), v;
	uint32_t l;
	int k;

	e = realroot(e);
	plan_build(&p, &e, 1, 0);
	for (k = 0; k < nx; k++)
		if (slots[k] >= 0 && slots[k] < MAXP) {
			in_r[(size_t)slots[k] * CH] = x[k];
			in_i[(size_t)slots[k] * CH] = (int64_t)x[k];
		}
	plan_run(&p, 1, in_i, in_r);
	l = p.root[0];
	*ok = !(COLF(&p, l)[0] & F_UNDEF);
	v = COLR(&p, l)[0];
	if (isnan(v))
		*ok = 0;
	plan_free(&p);
	free(in_i);
	free(in_r);
	return v;
}

double
xf_eval1(nid e, int slot, double x, int *ok)
{
	return xf_eval(e, &slot, &x, 1, ok);
}

void
xf_evalv(nid e, int slot, const double *xs, int n, const int *fs,
    const double *fx, int nf, double *out, uint8_t *ok)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t l;
	int k, i, done = 0;

	e = realroot(e);
	plan_build(&p, &e, 1, 0);
	l = p.root[0];
	while (done < n) {
		int m = n - done > CH ? CH : n - done;
		for (k = 0; k < nf; k++)
			if (fs[k] >= 0 && fs[k] < MAXP)
				for (i = 0; i < m; i++) {
					in_r[(size_t)fs[k] * CH + i] = fx[k];
					in_i[(size_t)fs[k] * CH + i] = (int64_t)fx[k];
				}
		for (i = 0; i < m; i++) {
			in_r[(size_t)slot * CH + i] = xs[done + i];
			in_i[(size_t)slot * CH + i] = (int64_t)xs[done + i];
		}
		plan_run(&p, (uint32_t)m, in_i, in_r);
		for (i = 0; i < m; i++) {
			out[done + i] = COLR(&p, l)[i];
			ok[done + i] = !(COLF(&p, l)[i] & F_UNDEF) &&
			    !isnan(out[done + i]);
		}
		done += m;
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
}

/* a node without variables, evaluated */
static int
const_eval(nid e, double *v)
{
	int ok;

	if (xnum(e, v))
		return isfinite(*v);
	if (node_slots(e) != 0)
		return 0;
	*v = xf_eval(e, NULL, NULL, 0, &ok);
	return ok && isfinite(*v);
}

/* ---- polynomials ---- */

void
xp_zero(struct xpoly *p)
{
	p->deg = -1;
	memset(p->c, 0, sizeof p->c);
}

void
xp_const(struct xpoly *p, double c)
{
	xp_zero(p);
	p->c[0] = c;
	p->deg = c != 0 ? 0 : -1;
}

void
xp_trim(struct xpoly *p)
{
	double mx = 0;
	int i;

	for (i = 0; i <= p->deg; i++)
		mx = fmax(mx, fabs(p->c[i]));
	for (i = 0; i <= p->deg; i++)
		if (fabs(p->c[i]) <= 1e-13 * mx)
			p->c[i] = 0;
	while (p->deg >= 0 && p->c[p->deg] == 0)
		p->deg--;
	for (i = p->deg + 1; i <= XD; i++)
		p->c[i] = 0;
}

int
xp_add(const struct xpoly *a, const struct xpoly *b, double sb,
    struct xpoly *r)
{
	struct xpoly t;
	int i, d = a->deg > b->deg ? a->deg : b->deg;

	xp_zero(&t);
	for (i = 0; i <= d; i++)
		t.c[i] = (i <= a->deg ? a->c[i] : 0) +
		    sb * (i <= b->deg ? b->c[i] : 0);
	t.deg = d;
	{
		/* cancellation: trim relative to the operands */
		double mx = 0;
		for (i = 0; i <= a->deg; i++)
			mx = fmax(mx, fabs(a->c[i]));
		for (i = 0; i <= b->deg; i++)
			mx = fmax(mx, fabs(sb * b->c[i]));
		for (i = 0; i <= d; i++)
			if (fabs(t.c[i]) <= 1e-14 * mx)
				t.c[i] = 0;
	}
	while (t.deg >= 0 && t.c[t.deg] == 0)
		t.deg--;
	*r = t;
	return 1;
}

int
xp_mul(const struct xpoly *a, const struct xpoly *b, struct xpoly *r)
{
	struct xpoly t;
	int i, j;

	xp_zero(&t);
	if (a->deg < 0 || b->deg < 0) {
		*r = t;
		return 1;
	}
	if (a->deg + b->deg > XD)
		return 0;
	for (i = 0; i <= a->deg; i++)
		for (j = 0; j <= b->deg; j++)
			t.c[i + j] += a->c[i] * b->c[j];
	t.deg = a->deg + b->deg;
	xp_trim(&t);
	*r = t;
	return 1;
}

void
xp_scale(struct xpoly *p, double k)
{
	int i;

	for (i = 0; i <= p->deg; i++)
		p->c[i] *= k;
	if (k == 0)
		xp_zero(p);
}

void
xp_deriv(const struct xpoly *a, struct xpoly *r)
{
	struct xpoly t;
	int i;

	xp_zero(&t);
	for (i = 1; i <= a->deg; i++)
		t.c[i - 1] = i * a->c[i];
	t.deg = a->deg - 1;
	if (t.deg < -1)
		t.deg = -1;
	xp_trim(&t);
	*r = t;
}

double
xp_eval(const struct xpoly *p, double x)
{
	double v = 0;
	int i;

	for (i = p->deg; i >= 0; i--)
		v = v * x + p->c[i];
	return v;
}

double complex
xp_ceval(const struct xpoly *p, double complex x)
{
	double complex v = 0;
	int i;

	for (i = p->deg; i >= 0; i--)
		v = v * x + p->c[i];
	return v;
}

int
xp_divmod(const struct xpoly *a, const struct xpoly *b, struct xpoly *q,
    struct xpoly *r)
{
	struct xpoly R = *a, Q;
	int i, j;

	if (b->deg < 0)
		return 0;
	xp_zero(&Q);
	for (i = R.deg - b->deg; i >= 0; i--) {
		double c = R.c[i + b->deg] / b->c[b->deg];
		Q.c[i] = c;
		for (j = 0; j <= b->deg; j++)
			R.c[i + j] -= c * b->c[j];
		R.c[i + b->deg] = 0;
	}
	Q.deg = a->deg - b->deg;
	if (Q.deg < -1)
		Q.deg = -1;
	R.deg = b->deg - 1 < R.deg ? b->deg - 1 : R.deg;
	{
		double mx = 0;
		for (i = 0; i <= a->deg; i++)
			mx = fmax(mx, fabs(a->c[i]));
		for (i = 0; i <= R.deg; i++)
			if (fabs(R.c[i]) <= 1e-12 * mx)
				R.c[i] = 0;
	}
	for (i = R.deg + 1; i <= XD; i++)
		R.c[i] = 0;
	while (R.deg >= 0 && R.c[R.deg] == 0)
		R.deg--;
	xp_trim(&Q);
	if (q)
		*q = Q;
	if (r)
		*r = R;
	return 1;
}

/* a node as a polynomial in slot with numeric coefficients */
static int
xq_rec(nid e, int slot, struct xrat *r, int depth);

int
xp_of(nid e, int slot, struct xpoly *p)
{
	struct xrat r;

	if (!xq_of(e, slot, &r) || r.d.deg != 0)
		return 0;
	*p = r.n;
	xp_scale(p, 1 / r.d.c[0]);
	return 1;
}

static nid
mono(double c, nid x, int k)
{
	nid m = k == 0 ? xk(1) : k == 1 ? x : xpow(x, xk(k));

	return k == 0 ? xk(c) : xmul(xk(c), m);
}

nid
xp_node(const struct xpoly *p, nid x)
{
	nid acc = 0;
	int i;

	for (i = p->deg; i >= 0; i--) {
		if (p->c[i] == 0)
			continue;
		if (acc == 0)
			acc = mono(p->c[i], x, i);
		else if (p->c[i] < 0)
			acc = xsub(acc, mono(-p->c[i], x, i));
		else
			acc = xadd(acc, mono(p->c[i], x, i));
	}
	return acc ? acc : xk(0);
}

/* ---- rational functions ---- */

static void
xq_one(struct xrat *r, double c)
{
	xp_const(&r->n, c);
	xp_const(&r->d, 1);
}

int
xq_add(const struct xrat *a, const struct xrat *b, double sb,
    struct xrat *r)
{
	struct xpoly x, y, d;
	struct xrat t;

	/* the same denominator: add the numerators */
	if (a->d.deg == b->d.deg && !memcmp(a->d.c, b->d.c, sizeof a->d.c)) {
		t.d = a->d;
		xp_add(&a->n, &b->n, sb, &t.n);
		*r = t;
		return 1;
	}
	if (!xp_mul(&a->n, &b->d, &x) || !xp_mul(&b->n, &a->d, &y) ||
	    !xp_mul(&a->d, &b->d, &d))
		return 0;
	xp_add(&x, &y, sb, &t.n);
	t.d = d;
	xq_reduce(&t);
	*r = t;
	return 1;
}

int
xq_mul(const struct xrat *a, const struct xrat *b, struct xrat *r)
{
	struct xrat t;

	if (!xp_mul(&a->n, &b->n, &t.n) || !xp_mul(&a->d, &b->d, &t.d))
		return 0;
	xq_reduce(&t);
	*r = t;
	return 1;
}

int
xq_div(const struct xrat *a, const struct xrat *b, struct xrat *r)
{
	struct xrat t;

	if (b->n.deg < 0)
		return 0;
	if (!xp_mul(&a->n, &b->d, &t.n) || !xp_mul(&a->d, &b->n, &t.d))
		return 0;
	xq_reduce(&t);
	*r = t;
	return 1;
}

int
xq_deriv(const struct xrat *a, struct xrat *r)
{
	struct xpoly dn, dd, x, y;
	struct xrat t;

	xp_deriv(&a->n, &dn);
	xp_deriv(&a->d, &dd);
	if (!xp_mul(&dn, &a->d, &x) || !xp_mul(&a->n, &dd, &y) ||
	    !xp_mul(&a->d, &a->d, &t.d))
		return 0;
	xp_add(&x, &y, -1, &t.n);
	xq_reduce(&t);
	*r = t;
	return 1;
}

double
xq_eval(const struct xrat *r, double x)
{
	return xp_eval(&r->n, x) / xp_eval(&r->d, x);
}

double complex
xq_ceval(const struct xrat *r, double complex x)
{
	return xp_ceval(&r->n, x) / xp_ceval(&r->d, x);
}

static int
xq_rec(nid e, int slot, struct xrat *r, int depth)
{
	struct xrat a, b;
	double v;
	int64_t k;
	int i;

	if (depth > 200)
		return 0;
	if (!has_slot(e, slot)) {
		if (!const_eval(e, &v))
			return 0;
		xq_one(r, v);
		return 1;
	}
	switch (N.op[e]) {
	case O_VAR:
		xp_zero(&r->n);
		r->n.c[1] = 1;
		r->n.deg = 1;
		xp_const(&r->d, 1);
		return 1;
	case O_CAST:
		if (!TY_FP(N.ty[e]))
			return 0;
		return xq_rec(N.a[e], slot, r, depth + 1);
	case O_NEG:
		if (!xq_rec(N.a[e], slot, r, depth + 1))
			return 0;
		xp_scale(&r->n, -1);
		return 1;
	case O_ADD:
	case O_SUB:
		if (!xq_rec(N.a[e], slot, &a, depth + 1) ||
		    !xq_rec(N.b[e], slot, &b, depth + 1))
			return 0;
		return xq_add(&a, &b, N.op[e] == O_ADD ? 1 : -1, r);
	case O_MUL:
		if (!xq_rec(N.a[e], slot, &a, depth + 1) ||
		    !xq_rec(N.b[e], slot, &b, depth + 1))
			return 0;
		return xq_mul(&a, &b, r);
	case O_DIV:
		if (!xq_rec(N.a[e], slot, &a, depth + 1) ||
		    !xq_rec(N.b[e], slot, &b, depth + 1))
			return 0;
		return xq_div(&a, &b, r);
	case O_POW:
		if (has_slot(N.b[e], slot) || !xnum(N.b[e], &v) ||
		    v != floor(v) || fabs(v) > XD)
			return 0;
		k = (int64_t)v;
		if (!xq_rec(N.a[e], slot, &a, depth + 1))
			return 0;
		xq_one(&b, 1);
		for (i = 0; i < (k < 0 ? -k : k); i++)
			if (!xq_mul(&b, &a, &b))
				return 0;
		if (k < 0) {
			struct xrat one;
			xq_one(&one, 1);
			return xq_div(&one, &b, r);
		}
		*r = b;
		return 1;
	}
	return 0;
}

int
xq_of(nid e, int slot, struct xrat *r)
{
	if (!xq_rec(e, slot, r, 0))
		return 0;
	xq_reduce(r);
	return 1;
}

/* ---- roots ---- */

/* the polynomial scaled to integers, when its coefficients are rational */
static int
int_coeffs(const struct xpoly *p, double *c)
{
	int64_t L = 1;
	struct q q;
	int i;

	for (i = 0; i <= p->deg; i++) {
		int64_t g;
		if (p->c[i] == 0)
			continue;
		if (!q_from_double(p->c[i], &q) || q.d > 100000)
			return 0;
		g = gcd64(L, q.d);
		if (__builtin_mul_overflow(L / g, q.d, &L) || L > 100000000)
			return 0;
	}
	for (i = 0; i <= p->deg; i++) {
		double v = nearbyint(p->c[i] * (double)L);
		if (fabs(v - p->c[i] * (double)L) > 1e-6 * fmax(1, fabs(v)) ||
		    fabs(v) > 1e13)
			return 0;
		c[i] = v;
	}
	return 1;
}

/* Newton polish of a simple root */
static double complex
polish(const struct xpoly *p, double complex z)
{
	struct xpoly d;
	int it;

	xp_deriv(p, &d);
	for (it = 0; it < 8; it++) {
		double complex f = xp_ceval(p, z), fp = xp_ceval(&d, z);
		if (cabs(fp) == 0)
			break;
		{
			double complex nz = z - f / fp;
			if (cabs(xp_ceval(p, nz)) >= cabs(f))
				break;
			z = nz;
		}
	}
	return z;
}

/* monic gcd of two polynomials (Euclid with a relative tolerance) */
static void
xp_gcd(const struct xpoly *a0, const struct xpoly *b0, struct xpoly *g)
{
	struct xpoly a = *a0, b = *b0, r;
	double sc = 0;
	int i, it;

	for (i = 0; i <= a.deg; i++)
		sc = fmax(sc, fabs(a.c[i]));
	for (it = 0; it < 64 && b.deg >= 0; it++) {
		double bs = 0;
		xp_scale(&b, 1 / b.c[b.deg]);
		xp_divmod(&a, &b, NULL, &r);
		for (i = 0; i <= b.deg; i++)
			bs = fmax(bs, fabs(b.c[i]));
		for (i = 0; i <= r.deg; i++)
			if (fabs(r.c[i]) <= 1e-9 * fmax(bs, 1))
				r.c[i] = 0;
		while (r.deg >= 0 && r.c[r.deg] == 0)
			r.deg--;
		a = b;
		b = r;
	}
	if (a.deg >= 0)
		xp_scale(&a, 1 / a.c[a.deg]);
	for (i = 0; i <= a.deg; i++) {
		struct q q;
		if (q_from_double(a.c[i], &q) && q.d <= 10000 &&
		    fabs(q_val(q) - a.c[i]) <= 1e-9 * fmax(1, fabs(a.c[i])))
			a.c[i] = q_val(q);
	}
	*g = a;
}

/* how often (x - z) (or its conjugate pair) divides p */
static int
root_mult(const struct xpoly *p0, double complex z)
{
	struct xpoly p = *p0, f, q, r;
	int m = 0, i;

	xp_zero(&f);
	if (cimag(z) != 0) {
		f.c[0] = creal(z) * creal(z) + cimag(z) * cimag(z);
		f.c[1] = -2 * creal(z);
		f.c[2] = 1;
		f.deg = 2;
	} else {
		f.c[0] = -creal(z);
		f.c[1] = 1;
		f.deg = 1;
	}
	while (p.deg >= f.deg && m < XD) {
		double sc = 0, rs = 0;
		xp_divmod(&p, &f, &q, &r);
		for (i = 0; i <= p.deg; i++)
			sc = fmax(sc, fabs(p.c[i]));
		for (i = 0; i <= r.deg; i++)
			rs = fmax(rs, fabs(r.c[i]));
		if (rs > 1e-7 * sc)
			break;
		m++;
		p = q;
	}
	return m ? m : 1;
}

/*
 * Distinct roots, one per conjugate pair (im > 0 stands for the pair);
 * returns the count, or -1 when the degree is too high.  The square-free
 * part p/gcd(p, p') is solved, so repeated roots come out as accurately
 * as simple ones (exactly when they are rational or quadratic surds).
 */
int
xf_roots(const struct xpoly *p0, double complex *z, int *mult)
{
	struct root r[16];
	double c[XD + 1];
	struct xpoly p = *p0, d, g, sq;
	int n, i, nz = 0, zm = 0;

	xp_trim(&p);
	/* roots at 0 first: degree drops */
	while (p.deg > 0 && p.c[0] == 0) {
		memmove(p.c, p.c + 1, (size_t)p.deg * sizeof p.c[0]);
		p.c[p.deg--] = 0;
		zm++;
	}
	if (zm) {
		z[nz] = 0;
		mult[nz++] = zm;
	}
	if (p.deg <= 0)
		return nz;
	sq = p;
	if (p.deg >= 2) {
		xp_deriv(&p, &d);
		xp_gcd(&p, &d, &g);
		if (g.deg >= 1)
			xp_divmod(&p, &g, &sq, NULL);
	}
	xp_scale(&sq, 1 / sq.c[sq.deg]);
	{
		/* snap: the square-free part of an integer polynomial is rational */
		for (i = 0; i <= sq.deg; i++) {
			struct q q;
			if (q_from_double(sq.c[i], &q) && q.d <= 10000 &&
			    fabs(q_val(q) - sq.c[i]) <= 1e-8 * fmax(1, fabs(sq.c[i])))
				sq.c[i] = q_val(q);
		}
	}
	if (sq.deg > 14)
		return -1;
	if (!int_coeffs(&sq, c))
		memcpy(c, sq.c, sizeof c);
	n = poly_roots(c, sq.deg, r);
	for (i = 0; i < n; i++) {
		double complex w = r[i].re + I * r[i].im;
		if (!r[i].exact && !r[i].quad)
			w = polish(&sq, w);
		if (fabs(cimag(w)) < 1e-12 * (1 + cabs(w)))
			w = creal(w);
		if (cimag(w) != 0 && fabs(creal(w)) < 1e-13 * cabs(w))
			w = I * cimag(w);
		if (cimag(w) < 0)
			w = conj(w);
		z[nz] = w;
		mult[nz++] = sq.deg < p.deg ? root_mult(&p, w) : 1;
	}
	return nz;
}

/* (x - z)^m multiplied into a complex polynomial */
static int
cmul_lin(double complex *c, int *deg, double complex z, int m)
{
	int k, i;

	for (k = 0; k < m; k++) {
		if (*deg + 1 > XD)
			return 0;
		c[*deg + 1] = 0;
		for (i = *deg + 1; i >= 1; i--)
			c[i] = c[i - 1] - z * c[i];
		c[0] = -z * c[0];
		(*deg)++;
	}
	return 1;
}

/* coefficients of p(z + u) in u */
static void
ctaylor(const double complex *c, int deg, double complex z, double complex *o)
{
	double complex t[XD + 1];
	int i, j;

	for (i = 0; i <= deg; i++)
		t[i] = c[i];
	for (i = 0; i <= deg; i++) {
		/* synthetic division by (x - z), deg - i times */
		for (j = deg - 1; j >= i; j--)
			t[j] += z * t[j + 1];
		o[i] = t[i];
	}
}

int
xf_apart(const struct xrat *r0, struct xpf *pf)
{
	struct xrat r = *r0;
	struct xpoly rem;
	double complex z[XD];
	int mult[XD], nz, i, j, k, total = 0;

	memset(pf, 0, sizeof *pf);
	xq_reduce(&r);
	if (r.d.deg < 0)
		return 0;
	if (!xp_divmod(&r.n, &r.d, &pf->poly, &rem))
		return 0;
	if (r.d.deg == 0)
		return 1;
	nz = xf_roots(&r.d, z, mult);
	if (nz < 0)
		return 0;
	for (i = 0; i < nz; i++)
		total += mult[i] * (cimag(z[i]) != 0 ? 2 : 1);
	if (total != r.d.deg)
		return 0;
	for (i = 0; i < nz; i++) {
		double complex dt[XD + 1], dn[XD + 1], nn[XD + 1], ns[XD + 1],
		    ds[XD + 1], g[XD + 2];
		int dd = 0, m = mult[i];
		dt[0] = r.d.c[r.d.deg];
		for (j = 0; j < nz; j++) {
			if (j != i && !cmul_lin(dt, &dd, z[j], mult[j]))
				return 0;
			if (cimag(z[j]) != 0 && !cmul_lin(dt, &dd, conj(z[j]),
			    mult[j]))
				return 0;
		}
		for (j = 0; j <= rem.deg; j++)
			nn[j] = rem.c[j];
		if (rem.deg < 0)
			nn[0] = 0;
		ctaylor(nn, rem.deg < 0 ? 0 : rem.deg, z[i], ns);
		for (j = 0; j <= dd; j++)
			dn[j] = dt[j];
		ctaylor(dn, dd, z[i], ds);
		/* g = ns / ds as power series, m terms */
		for (k = 0; k < m; k++) {
			double complex s = k <= (rem.deg < 0 ? 0 : rem.deg) ? ns[k] : 0;
			for (j = 1; j <= k; j++)
				if (j <= dd)
					s -= ds[j] * g[k - j];
			g[k] = s / ds[0];
		}
		pf->pole[pf->np].p = z[i];
		pf->pole[pf->np].m = m;
		for (k = 1; k <= m; k++) {
			double complex a = g[m - k];
			double sc = cabs(a);
			if (fabs(cimag(a)) <= 1e-12 * sc)
				a = creal(a);
			if (fabs(creal(a)) <= 1e-12 * sc)
				a = I * cimag(a);
			pf->pole[pf->np].a[k] = a;
		}
		pf->np++;
	}
	return 1;
}

void
xq_reduce(struct xrat *r)
{
	double complex z[XD];
	int mult[XD], nz, i, k;
	double lc;

	xp_trim(&r->n);
	xp_trim(&r->d);
	if (r->d.deg < 0)
		return;
	if (r->n.deg < 0) {
		xp_const(&r->d, 1);
		return;
	}
	if (r->d.deg > 0 && r->n.deg > 0) {
		nz = xf_roots(&r->d, z, mult);
		for (i = 0; i < nz; i++)
			for (k = 0; k < mult[i] && r->n.deg > 0; k++) {
				double complex w = z[i];
				double sc = 0, aw = cabs(w), pw = 1;
				int j;
				struct xpoly f, q1, q2, rm;
				for (j = 0; j <= r->n.deg; j++, pw *= fmax(aw, 1e-300))
					sc += fabs(r->n.c[j]) * pw;
				if (cabs(xp_ceval(&r->n, w)) > 1e-9 * sc)
					break;
				xp_zero(&f);
				if (cimag(w) != 0) {
					f.c[0] = creal(w) * creal(w) + cimag(w) * cimag(w);
					f.c[1] = -2 * creal(w);
					f.c[2] = 1;
					f.deg = 2;
				} else {
					f.c[0] = -creal(w);
					f.c[1] = 1;
					f.deg = 1;
				}
				if (r->n.deg < f.deg)
					break;
				xp_divmod(&r->n, &f, &q1, &rm);
				xp_divmod(&r->d, &f, &q2, NULL);
				r->n = q1;
				r->d = q2;
			}
	}
	lc = r->d.c[r->d.deg];
	if (lc != 1 && lc != 0) {
		xp_scale(&r->n, 1 / lc);
		xp_scale(&r->d, 1 / lc);
	}
	/* snap coefficients that are rationals within rounding */
	for (i = 0; i <= r->n.deg; i++) {
		struct q q;
		if (q_from_double(r->n.c[i], &q) && q.d <= 10000 &&
		    fabs(q_val(q) - r->n.c[i]) <= 1e-12 * fabs(r->n.c[i]))
			r->n.c[i] = q_val(q);
	}
	for (i = 0; i <= r->d.deg; i++) {
		struct q q;
		if (q_from_double(r->d.c[i], &q) && q.d <= 10000 &&
		    fabs(q_val(q) - r->d.c[i]) <= 1e-12 * fabs(r->d.c[i]))
			r->d.c[i] = q_val(q);
	}
}

/* a polynomial scaled to coprime integers with a positive lead: the scale */
static double
primitive(struct xpoly *p)
{
	double c[XD + 1], g = 0;
	int i;

	if (p->deg < 0 || !int_coeffs(p, c))
		return 1;
	for (i = 0; i <= p->deg; i++)
		if (c[i] != 0)
			g = g == 0 ? fabs(c[i]) : (double)gcd64((int64_t)g,
			    (int64_t)fabs(c[i]));
	if (g == 0)
		return 1;
	if (c[p->deg] < 0)
		g = -g;
	{
		double k = p->c[p->deg] / (c[p->deg] / g);
		for (i = 0; i <= p->deg; i++)
			p->c[i] = c[i] / g;
		return k;
	}
}

static nid
pow_node(nid f, int m)
{
	return m == 1 ? f : xpow(f, xk(m));
}

nid
xq_node(const struct xrat *r0, nid x)
{
	struct xrat r = *r0;
	struct xpoly rest, fac[XD];
	int fm[XD], nf = 0, i, j, used[XD] = {0};
	double complex z[XD];
	int mult[XD], nz;
	double K = 1;
	nid num, den = 0;

	xq_reduce(&r);
	if (r.n.deg < 0)
		return xk(0);
	if (r.d.deg <= 0)
		return xp_node(&r.n, x);
	rest = r.d;   /* monic */
	nz = xf_roots(&r.d, z, mult);
	for (i = 0; i < nz && nf < XD; i++) {
		struct xpoly f, q, rm;
		double a = creal(z[i]), b = cimag(z[i]);
		struct q qa, qb;
		if (used[i])
			continue;
		xp_zero(&f);
		if (b == 0 && q_from_double(a, &qa) && qa.d <= 1000) {
			f.c[0] = -a;
			f.c[1] = 1;
			f.deg = 1;
		} else if (b != 0 && q_from_double(-2 * a, &qa) && qa.d <= 1000 &&
		    q_from_double(a * a + b * b, &qb) && qb.d <= 1000) {
			f.c[0] = q_val(qb);
			f.c[1] = q_val(qa);
			f.c[2] = 1;
			f.deg = 2;
		} else if (b == 0) {
			/* irrational real: pair it with a partner, a ± sqrt(d) */
			int j2;
			for (j2 = i + 1; j2 < nz; j2++) {
				double s = a + creal(z[j2]), p2 = a * creal(z[j2]);
				struct q qs, qp;
				if (used[j2] || cimag(z[j2]) != 0 || mult[j2] != mult[i] ||
				    !q_from_double(s, &qs) || qs.d > 1000 ||
				    !q_from_double(p2, &qp) || qp.d > 1000)
					continue;
				f.c[0] = q_val(qp);
				f.c[1] = -q_val(qs);
				f.c[2] = 1;
				f.deg = 2;
				used[j2] = 1;
				break;
			}
			if (f.deg < 0)
				continue;
		} else
			continue;
		for (j = 0; j < mult[i]; j++) {
			xp_divmod(&rest, &f, &q, &rm);
			rest = q;
		}
		{
			/* f monic = k g, g primitive */
			double k = primitive(&f);
			for (j = 0; j < mult[i]; j++)
				K /= k;
		}
		fac[nf] = f;
		fm[nf++] = mult[i];
	}
	xp_trim(&rest);
	if (rest.deg > 0) {
		K /= primitive(&rest);
		den = xp_node(&rest, x);
	} else if (rest.deg == 0)
		K /= rest.c[0];
	for (i = 0; i < nf; i++) {
		nid f = pow_node(xp_node(&fac[i], x), fm[i]);
		den = den ? xmul(den, f) : f;
	}
	{
		struct xpoly n = r.n;
		if (n.deg == 0) {
			K *= n.c[0];
			num = xk(1);
		} else {
			K *= primitive(&n);
			num = xp_node(&n, x);
		}
	}
	{
		int64_t qn, qd;
		if (small_q(K, 100000, 100000000, &qn, &qd)) {
			if (qn != 1)
				num = xmul(xk((double)qn), num);
			if (qd != 1)
				den = den ? xmul(xk((double)qd), den) : xk((double)qd);
		} else
			num = xmul(xk(K), num);
	}
	if (!den)
		return num;
	return mk_bin(O_DIV, real(num), real(den));
}

/* ---- special functions ---- */

double
xf_gamma(double x)
{
	return tgamma(x);
}

double complex
xf_cgamma(double complex z)
{
	static const double g = 7, c[9] = {
		0.99999999999980993, 676.5203681218851, -1259.1392167224028,
		771.32342877765313, -176.61502916214059, 12.507343278686905,
		-0.13857109526572012, 9.9843695780195716e-6,
		1.5056327351493116e-7};
	double complex x, t;
	int i;

	if (creal(z) < 0.5)
		return M_PI / (csin(M_PI * z) * xf_cgamma(1 - z));
	z -= 1;
	x = c[0];
	for (i = 1; i < 9; i++)
		x += c[i] / (z + i);
	t = z + g + 0.5;
	return sqrt(2 * M_PI) * cpow(t, z + 0.5) * cexp(-t) * x;
}

double complex
xf_cdigamma(double complex z)
{
	double complex r = 0, w, w2;

	if (creal(z) < 0.5)
		return xf_cdigamma(1 - z) - M_PI / ctan(M_PI * z);
	while (cabs(z) < 12) {
		r -= 1 / z;
		z += 1;
	}
	w = 1 / z;
	w2 = w * w;
	r += clog(z) - 0.5 * w - w2 * (1.0 / 12 - w2 * (1.0 / 120 -
	    w2 * (1.0 / 252 - w2 * (1.0 / 240 - w2 / 132))));
	return r;
}

/* Bernoulli numbers B_2j for Euler-Maclaurin */
static const double B2[] = {1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30,
	5.0 / 66, -691.0 / 2730, 7.0 / 6, -3617.0 / 510};

double
xf_hurwitz(double s, double a)
{
	int Nn = 16, k, j;
	double r = 0, x, fac;

	for (k = 0; k < Nn; k++)
		r += pow(k + a, -s);
	x = Nn + a;
	r += pow(x, 1 - s) / (s - 1) + 0.5 * pow(x, -s);
	fac = s;   /* s (s+1) ... (s + 2j - 2) / (2j)! */
	for (j = 1; j <= 8; j++) {
		double tgm = 1;
		int i;
		for (i = 1; i <= 2 * j; i++)
			tgm *= i;
		r += B2[j - 1] / tgm * fac * pow(x, -s - 2 * j + 1);
		fac *= (s + 2 * j - 1) * (s + 2 * j);
	}
	return r;
}

double
xf_zeta(double s)
{
	return xf_hurwitz(s, 1);
}

/* ---- input ---- */

static int
idch(int c)
{
	return isalnum(c) || c == '_' || (c & 0x80);
}

/* the start, in o, of the operand that ends at o's end */
static size_t
operand_start(const char *o, size_t n)
{
	size_t i = n;

	while (i > 0 && o[i - 1] == ' ')
		i--;
	if (i > 0 && (o[i - 1] == ')' || o[i - 1] == ']')) {
		int depth = 0;
		while (i > 0) {
			i--;
			if (o[i] == ')' || o[i] == ']')
				depth++;
			else if ((o[i] == '(' || o[i] == '[') && --depth == 0)
				break;
		}
		/* a call: sin(x)! */
		while (i > 0 && idch((unsigned char)o[i - 1]))
			i--;
		return i;
	}
	while (i > 0 && (idch((unsigned char)o[i - 1]) || o[i - 1] == '.'))
		i--;
	return i;
}

void
xf_prep(const char *in, struct sbuf *out)
{
	size_t i = 0, n = strlen(in);
	int close[64], nclose = 0, depth = 0;

	sb_clear(out);
	while (i < n) {
		unsigned char c = (unsigned char)in[i];
		if (c == 0xe2 && i + 2 < n && (unsigned char)in[i + 1] == 0x88 &&
		    (unsigned char)in[i + 2] == 0x9e) {
			sb_puts(out, "inf");
			i += 3;
			continue;
		}
		if (idch(c) && (i == 0 || !idch((unsigned char)in[i - 1]))) {
			size_t j = i, k;
			char w[40];
			while (j < n && idch((unsigned char)in[j]))
				j++;
			snprintf(w, sizeof w, "%.*s", (int)(j - i), in + i);
			for (k = j; k < n && in[k] == ' '; k++)
				;
			if (!strcmp(w, "oo") || !strcmp(w, "infinity") ||
			    !strcmp(w, "Infinity")) {
				sb_puts(out, "inf");
				i = j;
				continue;
			}
			if (k < n && (in[k] == '(' || in[k] == '[') &&
			    !func_find(sym_cstr(w))) {
				const char *rep = NULL;
				int sq = in[k] == '[';
				if (!strcmp(w, "u") || !strcmp(w, "H") ||
				    !strcmp(w, "heaviside") || !strcmp(w, "Heaviside") ||
				    !strcmp(w, "step") || !strcmp(w, "theta") ||
				    !strcmp(w, "unitstep") || !strcmp(w, "\xce\xb8"))
					rep = "heaviside";
				else if (!strcmp(w, "delta") || !strcmp(w, "dirac") ||
				    !strcmp(w, "Delta") || !strcmp(w, "DiracDelta") ||
				    !strcmp(w, "\xce\xb4") || !strcmp(w, "impulse"))
					rep = sq ? "kron" : "dirac";
				else if (!strcmp(w, "kron") || !strcmp(w, "kronecker"))
					rep = "kron";
				else if (!strcmp(w, "factorial") || !strcmp(w, "fact"))
					rep = "fact";
				else if (!strcmp(w, "binom") || !strcmp(w, "choose") ||
				    !strcmp(w, "binomial") || (!strcmp(w, "C") &&
				    !sq))
					rep = "binom";
				if (rep) {
					sb_puts(out, rep);
					sb_putc(out, '(');
					if (nclose < 64)
						close[nclose++] = depth;
					depth++;
					if (sq) {
						/* the matching ] becomes ) */
						size_t m = k + 1;
						int d2 = 1;
						char *cp;
						for (; m < n; m++) {
							if (in[m] == '[')
								d2++;
							if (in[m] == ']' && --d2 == 0)
								break;
						}
						cp = xmalloc(n + 1);
						memcpy(cp, in, n + 1);
						if (m < n)
							cp[m] = ')';
						in = cp;   /* leaked: small and rare */
					}
					i = k + 1;
					continue;
				}
			}
			sb_printf(out, "%.*s", (int)(j - i), in + i);
			i = j;
			continue;
		}
		if (c == '(' || c == '[')
			depth++;
		if (c == ')' || c == ']') {
			depth--;
			if (nclose && close[nclose - 1] == depth)
				nclose--;
		}
		if (c == '!' && (i + 1 >= n || in[i + 1] != '=') && out->n > 0) {
			size_t s = operand_start(out->p, out->n);
			size_t len = out->n - s;
			char *op = xmalloc(len + 1);
			memcpy(op, out->p + s, len);
			op[len] = 0;
			out->n = s;
			out->p[s] = 0;
			sb_printf(out, "fact(%s)", op);
			free(op);
			i++;
			continue;
		}
		sb_putc(out, (char)c);
		i++;
	}
}

static const char *const helper_names[] = {"heaviside", "kron", "dirac",
	"fact", "binom", NULL};

void
xf_helpers(void)
{
	static const char *const defs[] = {
		"heaviside(x) = x >= 0 ? 1.0 : 0.0",
		"kron(x) = x == 0 ? 1.0 : 0.0",
		"dirac(x) = x == 0 ? inf : 0.0",
		"fact(x) = Wm1(x)",
		"binom(n, k) = fact(n)/(fact(k)*fact(n - k))",
	};
	int k;

	for (k = 0; helper_names[k]; k++) {
		struct func *f = func_find(sym_cstr(helper_names[k]));
		if (f)
			continue;
		run_source(defs[k], "<xform>", 0);
		notes_drop();
		if ((f = func_find(sym_cstr(helper_names[k]))) != NULL)
			f->anon = 1;
	}
}

static nid fix_rec(nid e, nid *memo, uint32_t lim);
static int fix_mode;   /* 0: fact placeholders, 1: dirac to 0 */

static nid
remake(nid n, nid a, nid b, nid c)
{
	int op = N.op[n];

	if (a == N.a[n] && b == N.b[n] && c == N.c[n])
		return n;
	switch (op) {
	case O_CAST:
		return mk_cast(N.ty[n], a);
	case O_SEL:
		return mk_sel(a, b, c);
	case O_FMA:
		return mk_fma(a, b, c);
	}
	if (opinfo[op].arity == 1)
		return mk_un(op, a);
	if (opinfo[op].arity == 2)
		return mk_bin(op, a, b);
	return n;
}

static nid
fix_rec(nid e, nid *memo, uint32_t lim)
{
	nid a, b, c, r;
	int op;

	if (e == 0 || e >= lim)
		return e;
	if (memo[e])
		return memo[e];
	op = N.op[e];
	if (fix_mode == 1 && xis_dirac(e, &a)) {
		memo[e] = xk(0);
		return memo[e];
	}
	if (op == O_CONST || op == O_VAR || op == O_UNDEF || op == O_LOOP ||
	    op == O_TABLE || op == O_INDEX || op == O_ATOMIC) {
		memo[e] = e;
		return e;
	}
	a = fix_rec(N.a[e], memo, lim);
	if (op == O_LAMBERTWM1)
		r = xfact(a);
	else {
		b = opinfo[op].arity >= 2 || op == O_SEL ? fix_rec(N.b[e], memo,
		    lim) : N.b[e];
		c = op == O_SEL || op == O_FMA ? fix_rec(N.c[e], memo, lim) :
		    N.c[e];
		r = remake(e, a, b, c);
	}
	memo[e] = r;
	return r;
}

nid
xf_fix(nid e)
{
	uint32_t lim = N.n;
	nid *memo = xcalloc(lim, sizeof *memo), r;

	fix_mode = 0;
	r = fix_rec(e, memo, lim);
	free(memo);
	return r;
}

nid
xf_nodirac(nid e)
{
	uint32_t lim = N.n;
	nid *memo = xcalloc(lim, sizeof *memo), r;

	fix_mode = 1;
	r = fix_rec(e, memo, lim);
	fix_mode = 0;
	free(memo);
	return r;
}

/*
 * The parsed function is copied out of the function table: cmd_func
 * reuses the anonymous entries "_" and "_2", and func_new may move the
 * table, so a pointer into it does not survive the next parse.
 */
struct func *
xf_parse(const char *text)
{
	static struct func pool[32];
	static int k;
	struct sbuf s = {0};
	struct func *f, *r = &pool[k++ & 31];

	xf_prep(text, &s);
	f = cmd_func_text(sb_str(&s));
	free(s.p);
	*r = *f;
	r->body = xf_fix(r->body);
	notes_flush();
	return r;
}

int
xf_slot(const struct func *f, const char *name)
{
	int k;

	for (k = 0; k < f->np; k++)
		if (!strcmp(sym_str(f->p[k].name), name))
			return k;
	return -1;
}
