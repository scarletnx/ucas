/*
 * series.c: infinite sums and products (xform.h).
 *
 *   sum(n, lo, inf, a)        convergence (the test named), a closed form
 *   sum(n in lo..inf, a)      when one is found, a numeric value with
 *                             acceleration, and the two compared
 *   prod(n, lo, inf, a)       infinite products (Gamma quotients)
 *   converge A | sum(...)     the convergence tests alone
 *   radius A | sum(...)       radius and interval of convergence of a
 *                             power series, endpoints tested
 *   powerseries F [x] [at a]  the general term of F's power series
 *   asymptotic sum(k, 1, n, f)   Euler-Maclaurin: log(n) + gamma + ...
 *
 * Closed forms: polynomial times r^n (Eulerian numbers), rational bodies
 * by partial fractions (zeta(k), eta(k), Dirichlet beta, digamma at
 * integers and half-integers, pi coth(pi) for n^2 + a^2), a library of
 * power series (exp, sin, cos, sinh, cosh, log, atan, atanh, asin,
 * geometric, binomial, central binomial, Catalan) matched against the
 * terms with a scale and a shift, telescoping g(n) - g(n + k).
 */
#include "xform.h"
#include "calc2.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- small helpers ---- */

/* a + or - between terms at the top level of printed text */
static int
top_level_sum(const char *s)
{
	int depth = 0;
	size_t i;

	for (i = 1; s[i]; i++) {
		if (s[i] == '(')
			depth++;
		else if (s[i] == ')')
			depth--;
		else if (!depth && s[i - 1] == ' ' && (s[i] == '+' || s[i] == '-') &&
		    s[i + 1] == ' ')
			return 1;
	}
	return 0;
}

static char *
sdup(const char *s)
{
	char *r = xmalloc(strlen(s) + 1);

	strcpy(r, s);
	return r;
}

static void
strim(char *s)
{
	size_t n = strlen(s), i = 0;

	while (n > 0 && isspace((unsigned char)s[n - 1]))
		s[--n] = 0;
	while (isspace((unsigned char)s[i]))
		i++;
	if (i)
		memmove(s, s + i, n - i + 1);
}

static int
is_inf_text(const char *s)
{
	while (*s == ' ' || *s == '+')
		s++;
	return !strcmp(s, "inf") || !strcmp(s, "oo") || !strcmp(s, "infinity") ||
	    !strcmp(s, "\xe2\x88\x9e") || !strcmp(s, "Infinity");
}

static double
eval_n(nid e, int slot, double n, int *ok)
{
	return xf_eval1(e, slot, n, ok);
}

/* ---- constants recognised ---- */

static const double EG = 0.57721566490153286061;    /* Euler's gamma */
static const double CAT = 0.91596559417721901505;   /* Catalan's G */

static double ident_tol = 1e-12;

const char *
xf_ident_tol(double v, double tol)
{
	const char *r;

	ident_tol = tol;
	r = xf_ident(v);
	ident_tol = 1e-12;
	return r;
}

const char *
xf_ident(double v)
{
	static char buf[8][96];
	static int k;
	char *b = buf[k++ & 7];
	const char *t = xf_num(v);
	struct q q;
	int i;
	static const struct {
		const char *name;
		double v;
	} more[] = {
		{"log(2pi)", 1.83787706640934548356},
		{"gamma", 0.57721566490153286061},
		{"zeta(3)", 1.20205690315959428540},
		{"zeta(5)", 1.03692775514336992633},
		{"catalan", 0.91596559417721901505},
		{"log(2pi)", 1.83787706640934548356},
		{"pi*cosh(pi)/sinh(pi)", 3.15334809493716},
		{"sinh(pi)/pi", 3.67607791037497945},
		{"e^pi", 23.1406926327792690},
		{"log(10)", 2.30258509299404568402},
		{NULL, 0}
	};

	{
		/* a decimal: digits with '.' or an exponent 1e-05 */
		int dec = strchr(t, '.') != NULL;
		const char *e = t;
		while ((e = strchr(e, 'e')) != NULL) {
			if (e > t && isdigit((unsigned char)e[-1]) &&
			    (e[1] == '-' || e[1] == '+' || isdigit((unsigned char)e[1])))
				dec = 1;
			e++;
		}
		if (!dec)
			return t;
	}
	for (i = 0; more[i].name; i++) {
		int dd;
		q.d = 0;
		for (dd = 1; dd <= 48; dd++) {
			double nn = nearbyint(v / more[i].v * dd);
			if (nn != 0 && fabs(nn) <= 48 && fabs(nn / dd * more[i].v - v) <=
			    ident_tol * fabs(v)) {
				int64_t g = gcd64((int64_t)fabs(nn), dd);
				q.n = (int64_t)nn / g;
				q.d = dd / g;
				break;
			}
		}
		if (q.d) {
			if (q.n == 1 && q.d == 1)
				snprintf(b, 96, "%s", more[i].name);
			else if (q.d == 1)
				snprintf(b, 96, "%lld*%s", (long long)q.n, more[i].name);
			else
				snprintf(b, 96, "%s%s/%lld", q.n == 1 ? "" : q.n == -1 ? "-" :
				    xf_num((double)q.n), more[i].name, (long long)q.d);
			if (q.n != 1 && q.n != -1 && q.d != 1)
				snprintf(b, 96, "%lld*%s/%lld", (long long)q.n, more[i].name,
				    (long long)q.d);
			return b;
		}
	}
	/* r + q B with a small rational r, for B = pi^k, log(2), gamma */
	{
		static const struct {
			const char *name;
			double v;
		} B[] = {{"pi", M_PI}, {"pi^2", M_PI * M_PI},
			{"log(2)", 0.69314718055994530942}, {"gamma", 0.57721566490153286061},
			{"pi*cosh(pi)/sinh(pi)", 3.15334809493716}, {NULL, 0}};
		int d2, n2;
		for (i = 0; B[i].name; i++)
			for (d2 = 1; d2 <= 24; d2++)
				for (n2 = -24; n2 <= 24; n2++) {
					double r;
					if (n2 == 0)
						continue;
					r = v - (double)n2 / d2 * B[i].v;
					if (q_from_double(r, &q) && q.d <= 24 && llabs(q.n) <= 100 &&
					    fabs(q_val(q) - r) <= 1e-12 * fmax(1, fabs(v)) &&
					    q.n != 0) {
						char bt[64];
						int64_t g = gcd64(llabs((int64_t)n2), d2);
						int64_t nn = n2 / g, dd = d2 / g;
						if (nn == 1 || nn == -1)
							snprintf(bt, sizeof bt, "%s", B[i].name);
						else
							snprintf(bt, sizeof bt, "%lld%s%s", (long long)llabs(nn),
							    isalpha((unsigned char)B[i].name[0]) ? "" : "*",
							    B[i].name);
						if (dd != 1)
							snprintf(bt + strlen(bt), sizeof bt - strlen(bt),
							    "/%lld", (long long)dd);
						snprintf(b, 96, "%s %c %s", xf_num(q_val(q)),
						    nn < 0 ? '-' : '+', bt);
						return b;
					}
				}
	}
	return NULL;
}

/* ---- parsing sum(...) ---- */

struct ssum {
	char idx[32];
	char lo[64], hi[64];
	char *body;
	int isprod;
};

/* sum(n, lo, hi, body) | sum(n in lo..hi, body) | sum(n = lo..hi, body) */
static int
parse_sum(const char *text, struct ssum *S)
{
	const char *p = text, *open;
	char *in, *part[8];
	int n, depth = 0, i;
	size_t len;

	memset(S, 0, sizeof *S);
	while (*p == ' ')
		p++;
	if (!strncmp(p, "sum", 3))
		p += 3;
	else if (!strncmp(p, "prod", 4))
		p += 4, S->isprod = 1;
	else if (!strncmp(p, "\xce\xa3", 2))
		p += 2;
	else if (!strncmp(p, "\xe2\x88\x8f", 3))
		p += 3, S->isprod = 1;
	else
		return 0;
	while (*p == ' ')
		p++;
	if (*p != '(')
		return 0;
	open = p;
	for (i = 0; p[i]; i++) {
		if (p[i] == '(')
			depth++;
		if (p[i] == ')' && --depth == 0)
			break;
	}
	if (!p[i])
		return 0;
	{
		const char *r = p + i + 1;
		while (*r == ' ')
			r++;
		if (*r)
			return 0;   /* sum(...) + more: not ours */
	}
	len = (size_t)i - 1;
	in = xmalloc(len + 1);
	memcpy(in, open + 1, len);
	in[len] = 0;
	{
		/* split at top-level commas */
		char *q = in, *start = in;
		int d = 0;
		n = 0;
		for (; *q; q++) {
			if (*q == '(' || *q == '[')
				d++;
			if (*q == ')' || *q == ']')
				d--;
			if (d == 0 && *q == ',' && n < 7) {
				*q = 0;
				part[n++] = start;
				start = q + 1;
			}
		}
		part[n++] = start;
	}
	for (i = 0; i < n; i++)
		strim(part[i]);
	if (n == 4) {
		snprintf(S->idx, sizeof S->idx, "%s", part[0]);
		snprintf(S->lo, sizeof S->lo, "%s", part[1]);
		snprintf(S->hi, sizeof S->hi, "%s", part[2]);
		S->body = sdup(part[3]);
	} else if (n == 2) {
		char *dd = strstr(part[0], ".."), *sep;
		if (!dd)
			goto bad;
		sep = strstr(part[0], " in ");
		if (!sep)
			sep = strchr(part[0], '=');
		if (!sep)
			goto bad;
		*dd = 0;
		snprintf(S->hi, sizeof S->hi, "%s", dd + 2);
		*sep = 0;
		snprintf(S->idx, sizeof S->idx, "%s", part[0]);
		snprintf(S->lo, sizeof S->lo, "%s", sep + (sep[1] == 'i' ? 4 : 1));
		strim(S->idx);
		strim(S->lo);
		strim(S->hi);
		S->body = sdup(part[1]);
	} else
		goto bad;
	free(in);
	return 1;
bad:
	free(in);
	return 0;
}

/* ---- terms ---- */

struct sctx {
	struct func *f;
	nid body;
	int ns;                   /* index slot */
	nid n;
	int64_t lo;
	char idx[32];
};

/* a(n) for n = from .. from + cnt - 1 */
static void
terms(struct sctx *C, int64_t from, int cnt, double *a, uint8_t *ok)
{
	double *xs = xmalloc((size_t)cnt * sizeof *xs);
	int i;

	for (i = 0; i < cnt; i++)
		xs[i] = (double)(from + i);
	xf_evalv(C->body, C->ns, xs, cnt, NULL, NULL, 0, a, ok);
	free(xs);
}

/* ---- convergence ---- */

enum { CV_UNKNOWN, CV_ABS, CV_COND, CV_DIV };

struct verdict {
	int v;
	char test[240];
	double ratio;             /* the ratio-test limit when known */
};

static int
alternates(struct sctx *C)
{
	double a[64];
	uint8_t ok[64];
	int i, flips = 0;

	terms(C, C->lo + 20, 64, a, ok);
	for (i = 1; i < 64; i++)
		if (ok[i] && ok[i - 1] && a[i] * a[i - 1] < 0)
			flips++;
	return flips >= 60;
}

/* a limit seen numerically: a nearby small rational, 1/e, e; else v */
static double
snap_limit(double v)
{
	int d;
	static const double sp[] = {M_E, 1 / M_E, 0};
	int i;

	if (!isfinite(v))
		return v;
	for (d = 1; d <= 100; d++) {
		double t = nearbyint(v * d) / d;
		if (fabs(t - v) <= 2e-6 * fmax(fabs(v), 1e-3))
			return t;
	}
	for (i = 0; sp[i] != 0; i++)
		if (fabs(v - sp[i]) <= 2e-6 * sp[i])
			return sp[i];
	return v;
}

/* lim |a(n+1)/a(n)|, Richardson in 1/n: NaN when unclear */
static double
ratio_limit(struct sctx *C)
{
	double r[12], N = 16;
	int k, ok1, ok2, m = 0;

	for (k = 0; k < 8; k++, N *= 2) {
		double a = eval_n(C->body, C->ns, C->lo + N, &ok1);
		double b = eval_n(C->body, C->ns, C->lo + N + 1, &ok2);
		if (!ok1 || !ok2 || !isfinite(a) || !isfinite(b) || a == 0 ||
		    fabs(a) < 1e-290 || fabs(b) < 1e-290)
			break;
		r[m++] = fabs(b / a);
	}
	if (m == 0)
		return NAN;
	{
		/* a steady ratio: r(N), r(N+1), r(N+2) close together */
		double Nl = 16 * pow(2, m - 1), a0, a1, a2, a3;
		int o0, o1, o2, o3;
		a0 = eval_n(C->body, C->ns, C->lo + Nl, &o0);
		a1 = eval_n(C->body, C->ns, C->lo + Nl + 1, &o1);
		a2 = eval_n(C->body, C->ns, C->lo + Nl + 2, &o2);
		a3 = eval_n(C->body, C->ns, C->lo + Nl + 3, &o3);
		if (o0 && o1 && o2 && o3 && a0 != 0 && a1 != 0 && a2 != 0) {
			double q1 = fabs(a1 / a0), q2 = fabs(a2 / a1), q3 = fabs(a3 / a2);
			double sp = fmax(fmax(q1, q2), q3) - fmin(fmin(q1, q2), q3);
			if (sp > 0.1 * fmax(q2, 1e-300) && sp > 1e-6)
				return NAN;   /* oscillating: the ratio test does not apply */
		}
	}
	if (m >= 3 && r[m - 1] < 0.6 * r[m - 2] && r[m - 2] < 0.6 * r[m - 3])
		return 0;   /* the ratio tends to 0 (factorials) */
	if (m >= 3 && r[m - 1] > 1.6 * r[m - 2] && r[m - 2] > 1.6 * r[m - 3])
		return INFINITY;
	if (m < 3)
		return snap_limit(r[m - 1]);
	{
		/* two Richardson steps (errors ~ 1/N, 1/N^2) */
		double r1 = 2 * r[m - 1] - r[m - 2], r0 = 2 * r[m - 2] - r[m - 3];
		double L = (4 * r1 - r0) / 3;
		if (fabs(L - r[m - 1]) > 0.05 || L < 0)
			L = r[m - 1];
		return snap_limit(L);
	}
}

/* the largest |a(n)| for n in [N, N + 300) */
static double
envelope(struct sctx *C, double N)
{
	double a[300], m = 0;
	uint8_t ok[300];
	int i;

	terms(C, (int64_t)N, 300, a, ok);
	for (i = 0; i < 300; i++)
		if (ok[i] && isfinite(a[i]))
			m = fmax(m, fabs(a[i]));
	return m;
}

/* the exponent p of the envelope |a(n)| ~ C n^-p (NaN when not one) */
static double
power_exponent(struct sctx *C, double *pfine)
{
	double e3 = envelope(C, 1e4), e4 = envelope(C, 1e5), e5 = envelope(C, 1e6);

	if (e3 == 0 || e4 == 0 || e5 == 0)
		return NAN;
	*pfine = -log10(e5 / e4);
	if (fabs(-log10(e4 / e3) - *pfine) > 0.03)
		return NAN;   /* drifting: log factors */
	return nearbyint(*pfine * 2) / 2;
}

/* the integral test numerically: int f(e^u) e^u du, increments over
 * doublings of u: shrinking (converges) or not (diverges) */
static int
integral_numeric(struct sctx *C, double *ratio)
{
	double U[5] = {20, 40, 80, 160, 320}, Iu[5];
	nid g = xmul(xsubst1(C->body, C->ns, xexp(C->n)), xexp(C->n));
	int k, nev;
	double err, a = log(fmax(C->lo, 2));

	for (k = 0; k < 5; k++) {
		if (!gk_integrate(NULL, g, C->ns, a, U[k], &Iu[k], &err, &nev))
			return 0;
	}
	{
		double d1 = Iu[3] - Iu[2], d2 = Iu[4] - Iu[3];
		if (d1 <= 0 || d2 < 0)
			return 0;
		*ratio = d2 / d1;
	}
	return 1;
}

/* n^p as text: n, n^2, n^(1/2) */
static const char *
npow(const char *idx, double p)
{
	static char buf[4][48];
	static int k;
	char *b = buf[k++ & 3];
	const char *t = xf_num(p);

	if (p == 1)
		snprintf(b, 48, "%s", idx);
	else if (strchr(t, '/') || strchr(t, '.'))
		snprintf(b, 48, "%s^(%s)", idx, t);
	else
		snprintf(b, 48, "%s^%s", idx, t);
	return b;
}

static void
verdict(struct sctx *C, struct verdict *V, int quiet_abs)
{
	struct xrat r;
	double L, a6, a5, pf = 0, p;
	int ok1, ok2, alt;

	memset(V, 0, sizeof *V);
	V->ratio = NAN;
	a5 = eval_n(C->body, C->ns, 1e5, &ok1);
	a6 = eval_n(C->body, C->ns, 1e6, &ok2);
	alt = alternates(C);
	/* exact: a rational body */
	if (xq_of(C->body, C->ns, &r) && r.n.deg >= 0) {
		int d = r.d.deg - r.n.deg;
		if (d <= 0) {
			V->v = CV_DIV;
			snprintf(V->test, sizeof V->test, "nth-term test: a_%s -> %s, not 0",
			    C->idx, d < 0 ? "infinity" : xf_num(r.n.c[r.n.deg] /
			    r.d.c[r.d.deg]));
			return;
		}
		if (d >= 2) {
			V->v = CV_ABS;
			if (r.n.deg == 0 && r.d.deg == d && r.d.c[d] != 0) {
				int pure = 1, i;
				for (i = 0; i < d; i++)
					if (r.d.c[i] != 0)
						pure = 0;
				if (pure) {
					snprintf(V->test, sizeof V->test, "p-series 1/%s^%d with "
					    "p = %d > 1", C->idx, d, d);
					return;
				}
			}
			snprintf(V->test, sizeof V->test, "limit comparison with the "
			    "p-series sum 1/%s^%d (p = %d > 1): a_%s %s^%d -> %s", C->idx, d,
			    d, C->idx, C->idx, d, xf_num(r.n.c[r.n.deg] / r.d.c[r.d.deg]));
			return;
		}
		V->v = CV_DIV;
		snprintf(V->test, sizeof V->test, "limit comparison with the harmonic "
		    "series sum 1/%s (p = 1): a_%s %s -> %s", C->idx, C->idx, C->idx,
		    xf_num(r.n.c[r.n.deg] / r.d.c[r.d.deg]));
		return;
	}
	/* the n-th term */
	if (ok1 && ok2 && (!isfinite(a6) || (fabs(a6) > 1e-3 && fabs(a6) >=
	    0.5 * fabs(a5)))) {
		L = ratio_limit(C);
		if (!(isfinite(L) && L < 1)) {
			V->v = CV_DIV;
			snprintf(V->test, sizeof V->test, "nth-term test: a_%s does not "
			    "tend to 0 (a_1000000 = %.4g)", C->idx, a6);
			return;
		}
	}
	/* ratio test */
	L = ratio_limit(C);
	V->ratio = L;
	if (isfinite(L) && L < 1 - 1e-4) {
		V->v = CV_ABS;
		snprintf(V->test, sizeof V->test, "ratio test: |a_(%s+1)/a_%s| -> %s "
		    "< 1", C->idx, C->idx, xf_num(L));
		return;
	}
	if ((isfinite(L) && L > 1 + 1e-4) || isinf(L)) {
		V->v = CV_DIV;
		snprintf(V->test, sizeof V->test, "ratio test: |a_(%s+1)/a_%s| -> %s "
		    "> 1", C->idx, C->idx, isinf(L) ? "infinity" : xf_num(L));
		return;
	}
	/* algebraic decay: compare with a p-series */
	p = power_exponent(C, &pf);
	if (alt) {
		/* alternating series test, then absolute convergence */
		int dec = 1, k, okk;
		double prev = INFINITY;
		for (k = 0; k < 8; k++) {
			double v = fabs(eval_n(C->body, C->ns, 1000 * pow(4, k), &okk));
			if (!okk || v > prev * (1 + 1e-12))
				dec = 0;
			prev = v;
		}
		if (dec && prev < 1e-3) {
			if (isfinite(p) && p > 1.05) {
				V->v = CV_ABS;
				snprintf(V->test, sizeof V->test, "|a_%s| ~ C/%s: comparison "
				    "with a p-series (p > 1)", C->idx, npow(C->idx, p));
			} else {
				V->v = CV_COND;
				snprintf(V->test, sizeof V->test, "alternating series test "
				    "(|a_%s| decreases to 0); not absolutely: |a_%s| ~ C/%s "
				    "(p <= 1)", C->idx, C->idx, npow(C->idx, isfinite(p) ? p :
				    nearbyint(pf * 100) / 100));
			}
			return;
		}
	}
	if (isfinite(p) && pf > 0) {
		if (p > 1.05 && fabs(pf - p) < 0.02) {
			V->v = CV_ABS;
			snprintf(V->test, sizeof V->test, "comparison with the p-series "
			    "sum 1/%s (|a_%s| ~ C/%s, p > 1)", npow(C->idx, p), C->idx,
			    npow(C->idx, p));
			return;
		}
		if (p <= 1 && fabs(pf - p) < 0.02 && !alt) {
			V->v = CV_DIV;
			snprintf(V->test, sizeof V->test, "comparison with the p-series "
			    "sum 1/%s (a_%s ~ C/%s, p <= 1)", npow(C->idx, p), C->idx,
			    npow(C->idx, p));
			return;
		}
	}
	/* integral test on the body as a function of a real variable */
	if (!alt) {
		const char *how;
		nid F = antideriv(C->body, C->ns, &how);
		if (F) {
			/* the limit of F at infinity, symbolically (calc2) */
			double L;
			int k = c2_limit_value(F, C->ns, INFINITY, -1, &L);
			if (k == 1) {
				V->v = CV_ABS;
				snprintf(V->test, sizeof V->test, "integral test: int f = %s "
				    "has the finite limit %s as %s -> inf", xf_str(F),
				    c2_fnum(L), C->idx);
				return;
			}
			if (k == 2 || k == -2) {
				V->v = CV_DIV;
				snprintf(V->test, sizeof V->test, "integral test: int f = %s "
				    "-> %sinf as %s -> inf", xf_str(F), k > 0 ? "+" : "-",
				    C->idx);
				return;
			}
		}
	}
	if (!alt) {
		double q;
		if (integral_numeric(C, &q)) {
			if (q < 0.75) {
				V->v = CV_ABS;
				snprintf(V->test, sizeof V->test, "integral test (numeric): "
				    "int f(x) dx converges (over x = e^u its pieces shrink by "
				    "%.2g per doubling of u)", q);
				return;
			}
			if (q > 0.9) {
				V->v = CV_DIV;
				snprintf(V->test, sizeof V->test, "integral test (numeric): "
				    "int f(x) dx diverges (over x = e^u its pieces do not "
				    "shrink: %.2g per doubling of u)", q);
				return;
			}
		}
	}
	/* root test */
	{
		double r1 = pow(fabs(a5), 1e-5), r2 = pow(fabs(a6), 1e-6);
		if (ok1 && ok2 && isfinite(r2) && fabs(r1 - r2) < 1e-3) {
			if (r2 < 0.999) {
				V->v = CV_ABS;
				snprintf(V->test, sizeof V->test, "root test: |a_%s|^(1/%s) -> "
				    "%.6g < 1", C->idx, C->idx, r2);
				return;
			}
			if (r2 > 1.001) {
				V->v = CV_DIV;
				snprintf(V->test, sizeof V->test, "root test: |a_%s|^(1/%s) -> "
				    "%.6g > 1", C->idx, C->idx, r2);
				return;
			}
		}
	}
	V->v = CV_UNKNOWN;
	snprintf(V->test, sizeof V->test, "the ratio, root, comparison and "
	    "integral tests are inconclusive here");
	(void)quiet_abs;
}

static const char *
verdict_word(int v)
{
	switch (v) {
	case CV_ABS: return "converges absolutely";
	case CV_COND: return "converges conditionally";
	case CV_DIV: return "diverges";
	}
	return "convergence unknown";
}

/* ---- numeric value with acceleration ---- */

struct nval {
	double v, err;
	char how[200];
	int ok;
};

static const double BN[] = {1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30};

static void
numeric_sum(struct sctx *C, struct verdict *V, struct nval *R)
{
	double a[4096], S = 0;
	uint8_t ok[4096];
	int i, n = 4096;

	memset(R, 0, sizeof *R);
	terms(C, C->lo, n, a, ok);
	if (isfinite(V->ratio) && V->ratio < 0.95) {
		/* geometric-like: sum until negligible, Aitken as the check */
		double s0 = 0, s1 = 0, s2 = 0;
		int used = 0;
		for (i = 0; i < n; i++) {
			if (!ok[i] || !isfinite(a[i])) {
				if (used > 8 && fabs(s2 - s1) <= 1e-15 * fabs(s2))
					break;
				snprintf(R->how, sizeof R->how, "a_%s is undefined at %s = "
				    "%lld", C->idx, C->idx, (long long)(C->lo + i));
				return;
			}
			S += a[i];
			s0 = s1;
			s1 = s2;
			s2 = S;
			used = i + 1;
			if (fabs(a[i]) <= 1e-18 * fabs(S) && i > 8)
				break;
		}
		{
			double d = (s2 - s1) - (s1 - s0), ait = d != 0 ? s2 - (s2 - s1) *
			    (s2 - s1) / d : s2;
			R->v = S;
			R->err = fmax(fabs(ait - S), 1e-16 * fabs(S));
		}
		snprintf(R->how, sizeof R->how, "%d terms (the ratio of terms -> %s; "
		    "Aitken's delta^2 agrees)", used, xf_num(V->ratio));
		R->ok = 1;
		return;
	}
	for (i = 0; i < n; i++)
		if (!ok[i] || !isfinite(a[i])) {
			snprintf(R->how, sizeof R->how, "a_%s is undefined at %s = %lld",
			    C->idx, C->idx, (long long)(C->lo + i));
			return;
		}
	if (alternates(C)) {
		/* Cohen-Rodriguez Villegas-Zagier on b_k = |a(lo + k)| */
		int m, k, pass;
		double res[2];
		double sgn = a[0] < 0 ? -1 : 1;
		for (pass = 0; pass < 2; pass++) {
			double d, b, c, s = 0;
			m = pass ? 30 : 40;
			d = pow(3 + sqrt(8), m);
			d = (d + 1 / d) / 2;
			b = -1;
			c = -d;
			for (k = 0; k < m; k++) {
				c = b - c;
				s += c * fabs(a[k]);
				b = (k + m) * (k - m) * b / ((k + 0.5) * (k + 1));
			}
			res[pass] = sgn * s / d;
		}
		R->v = res[0];
		R->err = fmax(fabs(res[0] - res[1]), 1e-16 * fabs(res[0]));
		snprintf(R->how, sizeof R->how, "alternating: Cohen-Villegas-Zagier "
		    "acceleration (an Euler-type transform), 40 terms");
		R->ok = 1;
		return;
	}
	{
		/* Euler-Maclaurin for the tail from N */
		int Ns[2] = {64, 128}, pass;
		double res[2];
		nid d1 = 0, d3 = 0, d5 = 0;
		jmp_buf saved;
		int good = 1;
		memcpy(saved, fail_jmp, sizeof saved);
		if (setjmp(fail_jmp) == 0) {
			d1 = diff_node(C->body, (uint32_t)C->ns);
			d3 = diff_node(diff_node(d1, (uint32_t)C->ns), (uint32_t)C->ns);
			d5 = diff_node(diff_node(d3, (uint32_t)C->ns), (uint32_t)C->ns);
		} else
			good = 0;
		memcpy(fail_jmp, saved, sizeof saved);
		notes_drop();
		for (pass = 0; good && pass < 2; pass++) {
			int N = Ns[pass], o1, o2, o3, o4, nev;
			double head = 0, tail, err, fN, f1, f3, f5;
			for (i = 0; i < N; i++)
				head += a[i];
			int gko = gk_integrate(NULL, C->body, C->ns, (double)(C->lo + N),
			    INFINITY, &tail, &err, &nev);
			if (!gko || err > 1e-10 * fabs(tail)) {
				/* x = e^u: int f(e^u) e^u du from log N */
				nid g = xmul(xsubst1(C->body, C->ns, xexp(C->n)),
				    xexp(C->n));
				double t2, e2;
				int gk2 = gk_integrate(NULL, g, C->ns, log((double)(C->lo + N)),
				    600, &t2, &e2, &nev);
				if (gk2 && e2 < 1e-9 * fabs(t2)) {
					/* beyond u = 700 (x = 1e304): g ~ u^-p, its tail is
					 * g(700) 700/(p - 1) */
					int q1, q2;
					double g1 = xf_eval1(g, C->ns, 300, &q1);
					double g2 = xf_eval1(g, C->ns, 600, &q2);
					tail = t2;
					if (q1 && q2 && g2 > 0 && g1 > 0) {
						double p = log(g1 / g2) / log(2);
						if (p > 1.05)
							tail += g2 * 600 / (p - 1);
						else {
							good = 0;
							break;
						}
					}
				} else {
					good = 0;
					break;
				}
			}
			fN = eval_n(C->body, C->ns, C->lo + N, &o1);
			f1 = eval_n(d1, C->ns, C->lo + N, &o2);
			f3 = eval_n(d3, C->ns, C->lo + N, &o3);
			f5 = eval_n(d5, C->ns, C->lo + N, &o4);
			if (!o1 || !o2 || !o3 || !o4) {
				good = 0;
				break;
			}
			res[pass] = head + tail + fN / 2 - BN[0] / 2 * f1 - BN[1] / 24 *
			    f3 - BN[2] / 720 * f5;
		}
		if (good) {
			R->v = res[1];
			R->err = fmax(fabs(res[1] - res[0]), 1e-16 * fabs(res[1]));
			snprintf(R->how, sizeof R->how, "the first 128 terms plus an "
			    "Euler-Maclaurin tail (integral + 3 corrections)");
			R->ok = 1;
			return;
		}
	}
	{
		/* Richardson on partial sums S_N, S_2N, S_4N with error ~ 1/N */
		double s1 = 0, s2 = 0, s4 = 0;
		for (i = 0; i < 1024; i++)
			s1 += a[i];
		s2 = s1;
		for (; i < 2048; i++)
			s2 += a[i];
		s4 = s2;
		for (; i < 4096; i++)
			s4 += a[i];
		R->v = 2 * s4 - s2;
		R->err = fabs((2 * s4 - s2) - (2 * s2 - s1));
		snprintf(R->how, sizeof R->how, "Richardson extrapolation of the "
		    "partial sums S_1024, S_2048, S_4096");
		R->ok = 1;
	}
}

/* ---- closed forms ---- */

/* exact constants: rational + sum c_k zeta(k) + c log 2 + ... */
struct xconst {
	double r, z[17], ln2, cat, beta[9], gam, pi1;
	int bad;
	char extra[160];
	double extrav;
};

static double
zeta_even_over_pik(int k)
{
	/* zeta(2m)/pi^(2m) */
	switch (k) {
	case 2: return 1.0 / 6;
	case 4: return 1.0 / 90;
	case 6: return 1.0 / 945;
	case 8: return 1.0 / 9450;
	case 10: return 1.0 / 93555;
	case 12: return 691.0 / 638512875;
	}
	return 0;
}

/* sum_{m>=0} 1/(m + a)^k for a = M or M + 1/2 (k >= 2): into X */
static int
hurwitz_exact(int k, double a, double c, struct xconst *X)
{
	int M, j;

	if (k > 16)
		return 0;
	if (a == floor(a) && a >= 1) {
		M = (int)a;
		X->z[k] += c;
		for (j = 1; j < M; j++)
			X->r -= c * pow(j, -k);
		return 1;
	}
	if (a - floor(a) == 0.5 && a > 0) {
		M = (int)floor(a);
		X->z[k] += c * (pow(2, k) - 1);
		for (j = 0; j < M; j++)
			X->r -= c * pow(2, k) * pow(2 * j + 1, -k);
		return 1;
	}
	return 0;
}

/* -psi(a): -(-gamma + H_(M-1)) or -(-gamma - 2 log 2 + sum 2/(2j-1)) */
static int
mpsi_exact(double a, double c, struct xconst *X)
{
	int M, j;

	if (a == floor(a) && a >= 1) {
		M = (int)a;
		X->gam += c;
		for (j = 1; j < M; j++)
			X->r -= c / j;
		return 1;
	}
	if (a - floor(a) == 0.5 && a > 0) {
		M = (int)floor(a);
		X->gam += c;
		X->ln2 += 2 * c;
		for (j = 1; j <= M; j++)
			X->r -= c * 2.0 / (2 * j - 1);
		return 1;
	}
	return 0;
}

/* phi_k(a) = sum_{m>=0} (-1)^m/(m + a)^k */
static int
alt_exact(int k, double a, double c, struct xconst *X)
{
	int M, j;

	if (k > 8)
		return 0;
	if (a == floor(a) && a >= 1) {
		/* (-1)^(M-1) [eta(k) - sum_{j<M} (-1)^(j-1)/j^k] */
		double s = ((int)a - 1) & 1 ? -1 : 1;
		M = (int)a;
		if (k == 1)
			X->ln2 += s * c;
		else
			X->z[k] += s * c * (1 - pow(2, 1 - k));
		for (j = 1; j < M; j++)
			X->r -= s * c * ((j - 1) & 1 ? -1 : 1) * pow(j, -k);
		return 1;
	}
	if (a - floor(a) == 0.5 && a > 0) {
		/* 2^k (-1)^M [beta(k) - sum_{j<M} (-1)^j/(2j+1)^k] */
		double s = ((int)floor(a)) & 1 ? -1 : 1, f = pow(2, k);
		M = (int)floor(a);
		X->beta[k] += s * f * c;
		for (j = 0; j < M; j++)
			X->r -= s * f * c * (j & 1 ? -1 : 1) * pow(2 * j + 1, -k);
		return 1;
	}
	return 0;
}

static double
xc_value(const struct xconst *X)
{
	static const double beta[] = {0, 0.78539816339744830962,
		0.91596559417721901505, 0.96894614625936938048,
		0.98894455174110533610, 0.99615782807708806400,
		0.99868522221843813544, 0.99955450789053990949,
		0.99984999024682965634};
	double v = X->r + X->ln2 * M_LN2 + X->gam * EG + X->extrav;
	int k;

	for (k = 2; k <= 16; k++)
		v += X->z[k] * xf_zeta(k);
	for (k = 1; k <= 8; k++)
		v += X->beta[k] * beta[k];
	return v;
}

static int
snapq(double v, struct q *q)
{
	if (fabs(v) < 1e-13) {
		q->n = 0;
		q->d = 1;
		return 1;
	}
	return q_from_double(v, q) && q->d <= 1000000 &&
	    fabs(q_val(*q) - v) <= 1e-10 * fmax(1, fabs(v));
}

/* the exact text of X; 0 when a coefficient is not rational */
static int
xc_text(const struct xconst *X, struct sbuf *b)
{
	struct q q;
	double pik[17] = {0};
	int k, first = 1;
	double rest = 0;

	if (fabs(X->gam) > 1e-9)
		return 0;   /* gamma must cancel */
	/* zeta(even) -> pi^k; beta(odd) -> pi^k */
	for (k = 2; k <= 16; k++)
		if (k % 2 == 0 && zeta_even_over_pik(k) != 0)
			pik[k] += X->z[k] * zeta_even_over_pik(k);
	pik[1] += X->beta[1] / 4;
	pik[3] += X->beta[3] / 32;
	pik[5] += X->beta[5] * 5 / 1536;
	pik[7] += X->beta[7] * 61 / 184320;
	if (!snapq(X->r, &q))
		return 0;
	if (q.n != 0) {
		sb_puts(b, xf_num(q_val(q)));
		first = 0;
	}
#define TERM(val, name) do { \
		if (!snapq(val, &q)) \
			return 0; \
		if (q.n != 0) { \
			int64_t an = q.n < 0 ? -q.n : q.n; \
			if (!first) \
				sb_puts(b, q.n < 0 ? " - " : " + "); \
			else if (q.n < 0) \
				sb_putc(b, '-'); \
			if (an != 1) \
				sb_printf(b, "%lld%s", (long long)an, \
				    isalpha((unsigned char)(name)[0]) ? "" : "*"); \
			sb_puts(b, name); \
			if (q.d != 1) \
				sb_printf(b, "/%lld", (long long)q.d); \
			first = 0; \
		} \
	} while (0)
	for (k = 1; k <= 16; k++) {
		char nm[16];
		if (k == 1)
			snprintf(nm, sizeof nm, "pi");
		else
			snprintf(nm, sizeof nm, "pi^%d", k);
		TERM(pik[k], nm);
	}
	TERM(X->ln2, "log(2)");
	for (k = 3; k <= 16; k += 2) {
		char nm[16];
		snprintf(nm, sizeof nm, "zeta(%d)", k);
		TERM(X->z[k], nm);
	}
	TERM(X->beta[2], "catalan");
	TERM(X->beta[4], "beta(4)");
	TERM(X->beta[6], "beta(6)");
	TERM(X->beta[8], "beta(8)");
#undef TERM
	if (X->extra[0]) {
		sb_printf(b, "%s%s", first ? "" : " + ", X->extra);
		first = 0;
	}
	if (first)
		sb_puts(b, "0");
	(void)rest;
	return 1;
}

/*
 * sum_{n>=lo} R(n) or (-1)^n R(n) for rational R: partial fractions with
 * poles at integers or half-integers (and n^2 + a^2): exact
 */
static int
sum_rational(struct sctx *C, int alt, struct sbuf *txt, double *val,
    const char **how)
{
	struct xrat r;
	struct xpf pf;
	struct xconst X;
	nid body = C->body;
	int i, k;

	memset(&X, 0, sizeof X);
	if (!xq_of(body, C->ns, &r))
		return 0;
	if (r.d.deg - r.n.deg < (alt ? 1 : 2))
		return 0;
	if (!xf_apart(&r, &pf) || pf.poly.deg >= 0)
		return 0;
	{
		/* first-order residues must cancel (plain sums) */
		double s1 = 0;
		for (i = 0; i < pf.np; i++)
			s1 += creal(pf.pole[i].a[1]) * (cimag(pf.pole[i].p) != 0 ? 2 : 1);
		if (!alt && fabs(s1) > 1e-9)
			return 0;
	}
	for (i = 0; i < pf.np; i++) {
		double complex p = pf.pole[i].p;
		if (cimag(p) != 0) {
			/* n^2 + a^2 only: (pi a coth(pi a) - 1)/(2 a^2) from n = 1 */
			double a = cimag(p);
			double A1 = creal(pf.pole[i].a[1]), B1 = cimag(pf.pole[i].a[1]);
			int j;
			if (alt || creal(p) != 0 || pf.pole[i].m != 1 || fabs(A1) > 1e-12)
				return 0;
			/* B1 (1/(n - ia)i ... ) : c/(n^2 + a^2) with c = -2 a B1 */
			{
				double c = -2 * a * B1, ca = M_PI * a / tanh(M_PI * a);
				double v = c * (ca - 1) / (2 * a * a);
				for (j = 0; j < C->lo && C->lo > 1; j++)
					;
				/* from lo: subtract n = 1..lo-1, or add n = 0 */
				for (j = 1; j < C->lo; j++)
					v -= c / (j * (double)j + a * a);
				if (C->lo <= 0)
					v += c / (a * a);
				X.extrav += v;
				{
					struct q qc, qa2;
					if (!snapq(c / (2 * a * a), &qc) || !snapq(a * a, &qa2))
						return 0;
					if (a == 1)
						snprintf(X.extra, sizeof X.extra, "%s%spi*cosh(pi)/sinh(pi)",
						    qc.n == 1 && qc.d == 1 ? "" : xf_num(q_val(qc)),
						    qc.n == 1 && qc.d == 1 ? "" : "*");
					else
						snprintf(X.extra, sizeof X.extra,
						    "%s*pi*%s*cosh(pi*%s)/sinh(pi*%s)", xf_num(q_val(qc)),
						    xf_num(a), xf_num(a), xf_num(a));
					/* the rest is rational: -c/(2a^2) and the shift terms */
					X.r += v - (c / (2 * a * a)) * ca;
					X.extrav -= v;
					X.extrav += (c / (2 * a * a)) * ca;
				}
			}
			continue;
		}
		{
			double a = C->lo - creal(p);
			if (a <= 0)
				return 0;   /* a pole at or after lo */
			for (k = 1; k <= pf.pole[i].m; k++) {
				double c = creal(pf.pole[i].a[k]);
				if (c == 0)
					continue;
				if (alt) {
					/* (-1)^n = (-1)^lo (-1)^m */
					double s = C->lo & 1 ? -1 : 1;
					if (!alt_exact(k, a, s * c, &X))
						return 0;
				} else if (k == 1) {
					if (!mpsi_exact(a, c, &X))
						return 0;
				} else if (!hurwitz_exact(k, a, c, &X))
					return 0;
			}
		}
	}
	*val = xc_value(&X);
	if (!xc_text(&X, txt)) {
		sb_clear(txt);
		return 0;
	}
	*how = alt ? "partial fractions; eta(k) = (1 - 2^(1-k)) zeta(k), "
	    "Dirichlet beta" : "partial fractions; zeta(k), digamma at integers";
	return 1;
}

/* body = (-1)^n R(n) (also (-1)^(n + c), cos(pi n)): R, else 0 */
static nid
strip_alt(nid e, int ns)
{
	double v, c1, c0;
	nid r;

	while (N.op[e] == O_CAST)
		e = N.a[e];
	switch (N.op[e]) {
	case O_NEG:
		r = strip_alt(N.a[e], ns);
		return r ? xneg(r) : 0;
	case O_MUL:
	case O_DIV:
		if ((r = strip_alt(N.a[e], ns)) != 0)
			return N.op[e] == O_MUL ? xmul(r, N.b[e]) : xdiv(r, N.b[e]);
		if (N.op[e] == O_MUL && (r = strip_alt(N.b[e], ns)) != 0)
			return xmul(N.a[e], r);
		return 0;
	case O_POW:
		if (xnum(N.a[e], &v) && v == -1) {
			struct npoly p;
			if (npoly_of(N.b[e], ns, &p, 1) && p.deg == 1 &&
			    xnum(xadd(xk(0), p.c[1]), &c1) && c1 == 1 &&
			    xnum(xadd(xk(0), p.c[0]), &c0) && c0 == floor(c0))
				return xk(fmod(fabs(c0), 2) == 1 ? -1 : 1);
		}
		return 0;
	case O_COS: {
		struct npoly p;
		if (npoly_of(N.a[e], ns, &p, 1) && p.deg == 1 &&
		    xnum(xadd(xk(0), p.c[1]), &c1) && fabs(c1 - M_PI) < 1e-15 &&
		    xnum(xadd(xk(0), p.c[0]), &c0) && c0 == 0)
			return xk(1);
		return 0;
	}
	}
	return 0;
}

/* poly(n) r^n: sum_{n>=lo} = r^lo sum_m p(m + lo) r^m */
static int
sum_geometric(struct sctx *C, struct sbuf *txt, double *val, const char **how)
{
	double a[24], r;
	uint8_t ok[24];
	int i, d, k;

	terms(C, C->lo, 24, a, ok);
	for (i = 0; i < 24; i++)
		if (!ok[i] || a[i] == 0)
			return 0;
	r = a[1] / a[0];
	/* find the degree d: a(n)/r^n is a polynomial of degree d */
	for (d = 0; d <= 6; d++) {
		/* r from consecutive ratios of the (d+1)-th differences... simpler:
		 * try r = lim a(n+1)/a(n) */
		double rr = a[23] / a[22], rq;
		struct q q;
		if (!q_from_double(rr, &q))
			continue;
		(void)rq;
		r = rr;
		break;
	}
	{
		/* r exactly: from a(n+1)/a(n) -> r with polynomial corrections:
		 * fit a(n) = r^n p(n) for candidate rationals near a[23]/a[22] */
		double cand[4];
		int nc = 0, ci;
		struct q q;
		double rr = a[23] / a[22];
		for (k = 1; k <= 12 && nc < 4; k++) {
			double t = nearbyint(rr * k) / k;
			if (t != 0 && fabs(t - rr) < 0.2 && (nc == 0 || cand[nc - 1] != t))
				cand[nc++] = t;
		}
		for (ci = 0; ci < nc; ci++) {
			double p[24];
			int dd;
			r = cand[ci];
			if (fabs(r) >= 1)
				continue;
			for (i = 0; i < 24; i++)
				p[i] = a[i] / pow(r, C->lo + i);
			/* p polynomial of degree dd: (dd+1)-th differences vanish */
			for (dd = 0; dd <= 6; dd++) {
				double df[24], mx = 0, sc = 0;
				int j, m;
				for (j = 0; j < 24; j++)
					df[j] = p[j];
				for (m = 0; m <= dd; m++)
					for (j = 0; j + m + 1 < 24; j++)
						df[j] = df[j + 1] - df[j];
				for (j = 0; j + dd + 1 < 24; j++)
					mx = fmax(mx, fabs(df[j]));
				for (j = 0; j < 24; j++)
					sc = fmax(sc, fabs(p[j]));
				if (mx <= 1e-9 * sc)
					break;
			}
			if (dd > 6)
				continue;
			{
				/* coefficients of p(n + lo) in m: fit, then sum
				 * sum_m m^k r^m = r A_k(r)/(1 - r)^(k+1) */
				double M[64], rhs[8], sol[8], S = 0, rl;
				int j;
				for (j = 0; j <= dd; j++) {
					int k2;
					for (k2 = 0; k2 <= dd; k2++)
						M[j * (dd + 1) + k2] = pow(j, k2);
					rhs[j] = p[j];
				}
				if (!solve_lin(M, rhs, dd + 1))
					continue;
				memcpy(sol, rhs, sizeof sol);
				for (k = 0; k <= dd; k++) {
					/* sum_{m>=0} m^k r^m */
					double s;
					if (k == 0)
						s = 1 / (1 - r);
					else {
						/* Eulerian */
						static double E[8][8];
						int x, y;
						double num = 0;
						memset(E, 0, sizeof E);
						E[0][0] = 1;
						for (x = 1; x < 8; x++)
							for (y = 0; y < x; y++)
								E[x][y] = (y + 1) * E[x - 1][y] + (x - y) *
								    (y > 0 ? E[x - 1][y - 1] : 0);
						for (y = 0; y < k; y++)
							num += E[k][y] * pow(r, y + 1);
						s = num / pow(1 - r, k + 1);
					}
					S += sol[k] * s;
				}
				rl = pow(r, C->lo);
				*val = rl * S;
				if (snapq(*val, &q))
					sb_puts(txt, xf_num(q_val(q)));
				else
					sb_puts(txt, xf_num(*val));
				*how = dd == 0 ? "geometric series a r^lo/(1 - r)" :
				    "arithmetico-geometric: sum m^k r^m = r A_k(r)/(1 - r)^(k+1) "
				    "(Eulerian numbers)";
				return 1;
			}
		}
	}
	return 0;
}

/* ---- the power-series library ---- */

enum { L_EXP, L_COSHD, L_SINHD, L_GEOM, L_GEOM2, L_GEOM3, L_MLOG,
	L_ATANHD, L_CBIN, L_CAT, L_SQRT, L_ISQRT, L_NLIB };

/* l(m) of sum_m l(m) u^m (dense in u) */
static double
lib_coef(int e, int m)
{
	double f = 1;
	int i;

	switch (e) {
	case L_EXP:
		for (i = 2; i <= m; i++)
			f *= i;
		return 1 / f;
	case L_COSHD:   /* 1/(2m)!: cosh(sqrt u) */
		for (i = 2; i <= 2 * m; i++)
			f *= i;
		return 1 / f;
	case L_SINHD:   /* 1/(2m+1)!: sinh(sqrt u)/sqrt u */
		for (i = 2; i <= 2 * m + 1; i++)
			f *= i;
		return 1 / f;
	case L_GEOM: return 1;
	case L_GEOM2: return m + 1;
	case L_GEOM3: return (m + 1.0) * (m + 2) / 2;
	case L_MLOG: return m ? 1.0 / m : 0;   /* -log(1 - u) */
	case L_ATANHD: return 1.0 / (2 * m + 1);   /* atanh(sqrt u)/sqrt u */
	case L_CBIN:
		for (i = 1; i <= m; i++)
			f = f * (m + i) / i;
		return f;
	case L_CAT:
		for (i = 1; i <= m; i++)
			f = f * (m + i) / i;
		return f / (m + 1);
	case L_SQRT:   /* (1 + u)^(1/2) */
		for (i = 0; i < m; i++)
			f = f * (0.5 - i) / (i + 1);
		return f;
	case L_ISQRT:  /* (1 + u)^(-1/2) */
		for (i = 0; i < m; i++)
			f = f * (-0.5 - i) / (i + 1);
		return f;
	}
	return 0;
}

static double
lib_radius(int e)
{
	switch (e) {
	case L_EXP: case L_COSHD: case L_SINHD:
		return INFINITY;
	case L_CBIN: case L_CAT:
		return 0.25;
	}
	return 1;
}

static const char *const lib_name[] = {"e^u", "cosh(sqrt(u))",
	"sinh(sqrt(u))/sqrt(u)", "1/(1 - u)", "1/(1 - u)^2", "1/(1 - u)^3",
	"-log(1 - u)", "atanh(sqrt(u))/sqrt(u)", "1/sqrt(1 - 4u)",
	"(1 - sqrt(1 - 4u))/(2u)", "sqrt(1 + u)", "1/sqrt(1 + u)"};

/* L(u) as a node; the sign of u picks cos/atan for the sqrt forms */
static nid
lib_node(int e, nid u, int neg)
{
	nid su;

	switch (e) {
	case L_EXP:
		return xexp(u);
	case L_COSHD:
		if (neg)
			return mk_un(O_COS, mk_un(O_SQRT, xneg(u)));
		return mk_un(O_COSH, mk_un(O_SQRT, u));
	case L_SINHD:
		if (neg) {
			su = mk_un(O_SQRT, xneg(u));
			return xdiv(mk_un(O_SIN, su), su);
		}
		su = mk_un(O_SQRT, u);
		return xdiv(mk_un(O_SINH, su), su);
	case L_GEOM:
		return xdiv(xk(1), xsub(xk(1), u));
	case L_GEOM2:
		return xdiv(xk(1), xpow(xsub(xk(1), u), xk(2)));
	case L_GEOM3:
		return xdiv(xk(1), xpow(xsub(xk(1), u), xk(3)));
	case L_MLOG:
		return xneg(mk_un(O_LOG, xsub(xk(1), u)));
	case L_ATANHD:
		if (neg) {
			su = mk_un(O_SQRT, xneg(u));
			return xdiv(mk_un(O_ATAN, su), su);
		}
		su = mk_un(O_SQRT, u);
		return xdiv(mk_un(O_ATANH, su), su);
	case L_CBIN:
		return xdiv(xk(1), mk_un(O_SQRT, xsub(xk(1), xmul(xk(4), u))));
	case L_CAT:
		return xdiv(xsub(xk(1), mk_un(O_SQRT, xsub(xk(1), xmul(xk(4), u)))),
		    xmul(xk(2), u));
	case L_SQRT:
		return mk_un(O_SQRT, xadd(xk(1), u));
	case L_ISQRT:
		return xdiv(xk(1), mk_un(O_SQRT, xadd(xk(1), u)));
	}
	return 0;
}

/*
 * c[m] (m = 0..nm-1, the coefficients of u^m of a series starting at m0)
 * as K lam^m l(m - j): the closed form sum_{m>=m0} c_m u^m as a node in u,
 * the radius in u, the entry
 */
struct libm {
	int e, j, ok;
	double K, lam, R;
};

static int
lib_match(const double *c, int nm, int m0, struct libm *L)
{
	int e, j, s;
	double sc = 0;
	int m;

	for (m = 0; m < nm; m++)
		sc = fmax(sc, fabs(c[m]));
	if (sc == 0)
		return 0;
	for (e = 0; e < L_NLIB; e++)
		for (j = 0; j <= 3; j++) {
			int m1 = -1, m2 = -1;
			for (m = m0; m < nm; m++)
				if (m >= j && c[m] != 0 && lib_coef(e, m - j) != 0) {
					if (m1 < 0)
						m1 = m;
					else {
						m2 = m;
						break;
					}
				}
			if (m1 < 0 || m2 < 0)
				continue;
			for (s = 0; s < 2; s++) {
				double rat = c[m2] * lib_coef(e, m1 - j) /
				    (c[m1] * lib_coef(e, m2 - j));
				double lam, K;
				int good = 1, dm = m2 - m1;
				if (rat < 0 && dm % 2 == 0)
					continue;
				lam = rat < 0 ? -pow(-rat, 1.0 / dm) : pow(rat, 1.0 / dm);
				if (s)
					lam = -lam;
				if (s && dm % 2)
					continue;
				K = c[m1] / (pow(lam, m1) * lib_coef(e, m1 - j));
				for (m = 0; m < nm && good; m++) {
					double want = m < m0 ? 0 : m < j ? 0 :
					    K * pow(lam, m) * lib_coef(e, m - j);
					if (m < j && m >= m0 && c[m] != 0)
						good = 0;
					if (fabs(want - c[m]) > 1e-10 * sc)
						good = 0;
				}
				if (!good)
					continue;
				{
					/* snap lam and K to nice values when they are */
					int d;
					for (d = 1; d <= 12; d++) {
						double t = nearbyint(lam * d) / d;
						if (fabs(t - lam) <= 1e-12 * fmax(1, fabs(lam))) {
							lam = t;
							break;
						}
					}
					for (d = 1; d <= 1000; d++) {
						double t = nearbyint(K * d) / d;
						if (fabs(t - K) <= 1e-12 * fmax(1, fabs(K))) {
							K = t;
							break;
						}
					}
				}
				L->e = e;
				L->j = j;
				L->K = K;
				L->lam = lam;
				L->R = lib_radius(e) / fabs(lam);
				L->ok = 1;
				return 1;
			}
		}
	return 0;
}

/* the closed form of sum_{m>=m0} c_m u^m given the match: a node; with
 * lu_given, that node is lam*u already */
static nid
lib_closed2(const struct libm *L, int m0, nid u, int neg, nid lu_given)
{
	nid lu = lu_given ? lu_given : xmul(xk(L->lam), u), r;
	int i;

	/* K lam^m l(m - j) u^m = K (lam u)^j l(i) (lam u)^i, i = m - j */
	r = lib_node(L->e, lu, neg);
	for (i = 0; i < m0 - L->j; i++)
		r = xsub(r, xmul(xk(lib_coef(L->e, i)), xpow(lu, xk(i))));
	if (L->j)
		r = xmul(xpow(lu, xk(L->j)), r);
	return xmul(xk(L->K), r);
}

static nid
lib_closed(const struct libm *L, int m0, nid u, int neg)
{
	return lib_closed2(L, m0, u, neg, 0);
}

/* for xform.c's genfunc: sum_{m>=m0} c_m u^m from the library */
nid
xf_libsum(const double *c, int nm, int m0, nid u, double *R,
    const char **name)
{
	struct libm L;

	memset(&L, 0, sizeof L);
	if (!lib_match(c, nm, m0, &L))
		return 0;
	*R = L.R;
	*name = lib_name[L.e];
	return lib_closed(&L, m0, u, L.lam < 0);
}

/* the library on a numeric sum: sum_{n>=lo} a(n) = closed form at u = 1 */
static int
sum_library(struct sctx *C, struct sbuf *txt, double *val, const char **how)
{
	double c[30];
	uint8_t ok[30];
	struct libm L;
	int i;
	nid u, r;

	if (C->lo < 0 || C->lo > 3)
		return 0;
	memset(c, 0, sizeof c);
	terms(C, C->lo, 30 - (int)C->lo, c + C->lo, ok);
	for (i = 0; i < 30 - C->lo; i++)
		if (!ok[i] || !isfinite(c[C->lo + i]))
			return 0;
	memset(&L, 0, sizeof L);
	if (!lib_match(c, 30, (int)C->lo, &L) || L.R <= 1)
		return 0;
	/* lam u is a variable named by lam's value, so that the text shows
	 * e.g. cos(1), e^2 - 1 */
	u = mk_var(MAXP - 1, sym_cstr(xf_num(L.lam)), TY_REAL);
	r = lib_closed2(&L, (int)C->lo, u, L.lam < 0, u);
	{
		int s = MAXP - 1, okk;
		double lam = L.lam;
		*val = xf_eval(r, &s, &lam, 1, &okk);
		if (!okk)
			return 0;
	}
	{
		const char *nice = xf_ident(*val);
		if (nice)
			sb_puts(txt, nice);
		else
			sb_puts(txt, xf_str(r));
	}
	*how = lib_name[L.e];
	return 1;
}

/* g(n) - g(n + k): telescoping */
static int
sum_telescoping(struct sctx *C, struct sbuf *txt, double *val,
    const char **how)
{
	nid b = C->body;
	int k;

	while (N.op[b] == O_CAST)
		b = N.a[b];
	if (N.op[b] != O_SUB)
		return 0;
	for (k = 1; k <= 4; k++) {
		nid g = N.a[b], h = N.b[b];
		nid gk = xsubst1(g, C->ns, xadd(C->n, xk(k)));
		int i, ok1, ok2, same = 1;
		for (i = 0; i < 6 && same; i++) {
			double x = C->lo + 0.37 + 1.3 * i;
			double v1 = xf_eval1(gk, C->ns, x, &ok1), v2 = xf_eval1(h, C->ns, x,
			    &ok2);
			if (!ok1 || !ok2 || fabs(v1 - v2) > 1e-12 * fmax(1, fabs(v1)))
				same = 0;
		}
		if (!same)
			continue;
		{
			/* sum = g(lo) + ... + g(lo + k - 1) - k lim g */
			double s = 0, lim1, lim2;
			int ok;
			for (i = 0; i < k; i++)
				s += xf_eval1(g, C->ns, C->lo + i, &ok);
			lim1 = xf_eval1(g, C->ns, 1e8, &ok);
			lim2 = xf_eval1(g, C->ns, 1e12, &ok);
			if (fabs(lim1 - lim2) > 1e-6 * fmax(1, fabs(lim2)))
				return 0;
			lim2 = nearbyint(lim2 * 1e6) / 1e6;
			*val = s - k * lim2;
			sb_puts(txt, xf_ident(*val) ? xf_ident(*val) : xf_num(*val));
			*how = "telescoping: g(n) - g(n + k) sums to g(lo) + ... + "
			    "g(lo + k - 1) - k lim g";
			return 1;
		}
	}
	return 0;
}

/* ---- context from text ---- */

static const char *const idx_names[] = {"n", "k", "i", "m", "j", NULL};

static int
setup(struct sctx *C, const char *bodytext, const char *idx, const char *lo,
    int need_alone)
{
	struct func *f = xf_parse(bodytext);
	int k;

	memset(C, 0, sizeof *C);
	C->f = f;
	C->body = f->body;
	if (idx && idx[0])
		snprintf(C->idx, sizeof C->idx, "%s", idx);
	else {
		for (k = 0; idx_names[k]; k++)
			if (xf_slot(f, idx_names[k]) >= 0)
				break;
		if (!idx_names[k])
			fail("which variable is the index?  (sum(n, 1, inf, ...))");
		snprintf(C->idx, sizeof C->idx, "%s", idx_names[k]);
	}
	C->ns = xf_slot(f, C->idx);
	if (C->ns < 0) {
		/* the body does not depend on the index */
		C->ns = f->np;
		if (C->ns >= MAXP)
			fail("too many variables");
	}
	C->n = xv(C->ns, C->idx);
	if (need_alone && f->np > (xf_slot(f, C->idx) >= 0 ? 1 : 0))
		return 0;
	if (lo && lo[0]) {
		struct func *fl = xf_parse(lo);
		double v;
		if (!xnum(fl->body, &v) || v != floor(v))
			fail("the lower bound must be an integer");
		C->lo = (int64_t)v;
	} else
		C->lo = 1;
	return 1;
}

/* ---- sum(...) with an infinite bound ---- */

static int sum_parts(struct sctx *C);

static void
infinite_sum(struct ssum *S)
{
	struct sctx C;
	struct verdict V;
	struct nval R;
	struct sbuf txt = {0};
	double cv = 0;
	const char *how = NULL;
	int have = 0;

	if (!setup(&C, S->body, S->idx, S->lo, 1))
		fail("sum: the terms have other variables: for a power series use "
		    "radius, for a value give them numbers");
	printf("sum(%s, %lld, inf, %s)\n", C.idx, (long long)C.lo,
	    xf_str(C.body));
	if (sum_parts(&C)) {
		free(txt.p);
		return;
	}
	verdict(&C, &V, 0);
	printf("  %s: %s\n", verdict_word(V.v), V.test);
	if (V.v == CV_DIV) {
		free(txt.p);
		return;
	}
	if (sum_geometric(&C, &txt, &cv, &how) ||
	    sum_rational(&C, 0, &txt, &cv, &how) ||
	    sum_library(&C, &txt, &cv, &how) ||
	    sum_telescoping(&C, &txt, &cv, &how))
		have = 1;
	if (!have) {
		/* (-1)^n R(n) */
		nid R = strip_alt(C.body, C.ns), saved = C.body;
		if (R) {
			sb_clear(&txt);
			C.body = R;
			if (sum_rational(&C, 1, &txt, &cv, &how))
				have = 1;
			C.body = saved;
		}
	}
	if (have)
		printf("  = %s\n  method: %s\n", txt.p, how);
	numeric_sum(&C, &V, &R);
	if (R.ok) {
		printf("  numeric: %.16g (%s; estimated error %.2g)\n", R.v, R.how,
		    R.err);
		if (have)
			printf("  check: the closed form and the numeric value %s "
			    "(difference %.2g)\n", fabs(cv - R.v) <= 1e3 * R.err + 1e-12 *
			    fabs(cv) ? "agree" : "DISAGREE", fabs(cv - R.v));
		else {
			const char *id = xf_ident(R.v);
			if (id && fabs(R.err) < 1e-11 * fmax(1, fabs(R.v)))
				printf("  recognised numerically (not proved): %s\n", id);
		}
	} else
		printf("  numeric: %s\n", R.how);
	free(txt.p);
}

/*
 * A sum of terms whose sums all converge: each on its own, the results
 * added (linearity).  0 when the body is one term or a part diverges.
 */
static int
sum_parts(struct sctx *C)
{
	nid parts[8];
	int sg[8], np = 0, i, all_abs = 1;
	nid stack[16];
	int ssg[16], sp = 0;
	struct sbuf txt = {0};
	double total = 0, nsum = 0, nerr = 0;
	int exact = 1;

	stack[sp] = C->body;
	ssg[sp++] = 1;
	while (sp) {
		nid e = stack[--sp];
		int g = ssg[sp];
		while (N.op[e] == O_CAST)
			e = N.a[e];
		if ((N.op[e] == O_ADD || N.op[e] == O_SUB) && sp < 14) {
			stack[sp] = N.b[e];
			ssg[sp++] = N.op[e] == O_SUB ? -g : g;
			stack[sp] = N.a[e];
			ssg[sp++] = g;
			continue;
		}
		if (np == 8)
			return 0;
		parts[np] = e;
		sg[np++] = g;
	}
	if (np < 2)
		return 0;
	{
		/* a rational body as a whole is better: 1/n - 1/(n + 1) */
		struct xrat r;
		if (xq_of(C->body, C->ns, &r))
			return 0;
	}
	for (i = 0; i < np; i++) {
		struct sctx P = *C;
		struct verdict V;
		P.body = parts[i];
		verdict(&P, &V, 0);
		if (V.v != CV_ABS && V.v != CV_COND)
			return 0;
		if (V.v != CV_ABS)
			all_abs = 0;
	}
	printf("  by linearity: the sum of %d series, each convergent:\n", np);
	for (i = 0; i < np; i++) {
		struct sctx P = *C;
		struct verdict V;
		struct nval R;
		struct sbuf t = {0};
		double cv = 0;
		const char *how = NULL;
		int have;
		P.body = parts[i];
		verdict(&P, &V, 0);
		have = sum_geometric(&P, &t, &cv, &how) ||
		    sum_rational(&P, 0, &t, &cv, &how) ||
		    sum_library(&P, &t, &cv, &how) ||
		    sum_telescoping(&P, &t, &cv, &how);
		if (!have) {
			nid Ra = strip_alt(P.body, P.ns), saved = P.body;
			if (Ra) {
				sb_clear(&t);
				P.body = Ra;
				have = sum_rational(&P, 1, &t, &cv, &how);
				P.body = saved;
			}
		}
		numeric_sum(&P, &V, &R);
		printf("    %ssum %s: %s%s%s\n", sg[i] < 0 ? "- " : "",
		    xf_str(parts[i]), verdict_word(V.v), have ? ", = " : "",
		    have ? t.p : "");
		if (have) {
			if (txt.n)
				sb_puts(&txt, sg[i] < 0 ? " - " : " + ");
			else if (sg[i] < 0)
				sb_puts(&txt, "-");
			sb_printf(&txt, top_level_sum(t.p) ? "(%s)" : "%s", t.p);
			total += sg[i] * cv;
		} else
			exact = 0;
		if (R.ok) {
			nsum += sg[i] * R.v;
			nerr += R.err;
		} else
			nerr = INFINITY;
		free(t.p);
	}
	printf("  %s\n", all_abs ? "converges absolutely" :
	    "converges (some parts only conditionally)");
	if (exact) {
		printf("  = %s\n", txt.p);
	}
	if (isfinite(nerr)) {
		printf("  numeric: %.16g (each part accelerated; estimated error "
		    "%.2g)\n", nsum, nerr);
		if (exact)
			printf("  check: the closed form and the numeric value %s "
			    "(difference %.2g)\n", fabs(total - nsum) <= 1e3 * nerr +
			    1e-12 * fabs(total) ? "agree" : "DISAGREE",
			    fabs(total - nsum));
	}
	free(txt.p);
	return 1;
}

/* ---- infinite products ---- */

static void
infinite_prod(struct ssum *S)
{
	struct sctx C;
	struct xrat r;
	double val = NAN, err = 0;
	int exact = 0, i;

	if (!setup(&C, S->body, S->idx, S->lo, 1))
		fail("prod: the factors have other variables");
	printf("prod(%s, %lld, inf, %s)\n", C.idx, (long long)C.lo,
	    xf_str(C.body));
	if (xq_of(C.body, C.ns, &r) && r.n.deg == r.d.deg && r.n.deg > 0) {
		double complex za[XD], zb[XD];
		int ma[XD], mb[XD], na = xf_roots(&r.n, za, ma), nb = xf_roots(&r.d,
		    zb, mb);
		double complex sa = 0, sb = 0, v = 1;
		double lead = r.n.c[r.n.deg] / r.d.c[r.d.deg];
		int zero = 0;
		if (na >= 0 && nb >= 0) {
			for (i = 0; i < na; i++) {
				int c = cimag(za[i]) != 0;
				sa += ma[i] * (c ? 2 * creal(za[i]) : za[i]);
			}
			for (i = 0; i < nb; i++) {
				int c = cimag(zb[i]) != 0;
				sb += mb[i] * (c ? 2 * creal(zb[i]) : zb[i]);
			}
			if (fabs(lead - 1) > 1e-12) {
				printf("  diverges (to %s): the factors tend to %s, not 1\n",
				    lead > 1 ? "infinity" : "0", xf_num(lead));
				return;
			}
			if (cabs(sa - sb) > 1e-9) {
				printf("  diverges: the factors are 1 + %s/%s + O(1/%s^2) and "
				    "sum %s/%s diverges\n", xf_num(creal(sa - sb) * -1 + 0),
				    C.idx, C.idx, xf_num(creal(sb - sa)), C.idx);
				return;
			}
			printf("  converges: the factors are 1 + O(1/%s^2) "
			    "(sum |a_%s - 1| converges)\n", C.idx, C.idx);
			/* prod Gamma(lo - b)/Gamma(lo - a) */
			for (i = 0; i < na; i++) {
				int k, c = cimag(za[i]) != 0;
				for (k = 0; k < ma[i]; k++) {
					double complex x = C.lo - za[i];
					if (cimag(x) == 0 && creal(x) <= 0 &&
					    creal(x) == floor(creal(x)))
						zero = 1;
					else {
						v /= xf_cgamma(x);
						if (c)
							v /= xf_cgamma(conj(x));
					}
				}
			}
			for (i = 0; i < nb; i++) {
				int k, c = cimag(zb[i]) != 0;
				for (k = 0; k < mb[i]; k++) {
					double complex x = C.lo - zb[i];
					v *= xf_cgamma(x);
					if (c)
						v *= xf_cgamma(conj(x));
				}
			}
			val = zero ? 0 : creal(v);
			exact = 1;
			{
				const char *id = xf_ident(val);
				printf("  = %s\n  method: the factors as (n - a_i)/(n - b_i): "
				    "prod Gamma(%lld - b_i)/Gamma(%lld - a_i)%s\n", id ? id :
				    xf_num(val), (long long)C.lo, (long long)C.lo,
				    zero ? " (a factor is 0)" : "");
			}
		}
	}
	{
		/* numeric: partial products, Richardson in 1/N */
		double p1 = 1, p2, p4;
		double a[4096];
		uint8_t ok[4096];
		terms(&C, C.lo, 4096, a, ok);
		for (i = 0; i < 1024; i++)
			p1 *= a[i];
		p2 = p1;
		for (; i < 2048; i++)
			p2 *= a[i];
		p4 = p2;
		for (; i < 4096; i++)
			p4 *= a[i];
		{
			double e1 = 2 * p4 - p2, e0 = 2 * p2 - p1;
			double ext = (4 * e1 - e0) / 3;
			err = fabs(e1 - e0) / 3;
			printf("  numeric: %.15g (partial products to %lld, Richardson in "
			    "1/N twice; estimated error %.2g)\n", ext,
			    (long long)(C.lo + 4095), err);
			if (exact)
				printf("  check: %s (difference %.2g)\n", fabs(ext - val) <= 100 *
				    err + 1e-12 ? "agree" : "DISAGREE", fabs(ext - val));
			else if (xf_ident(ext) && err < 1e-10)
				printf("  recognised numerically (not proved): %s\n",
				    xf_ident(ext));
		}
	}
}

/* ---- converge ---- */

static void
split_body(const char *text, struct ssum *S)
{
	if (!parse_sum(text, S)) {
		memset(S, 0, sizeof *S);
		S->body = sdup(text);
	}
}

static void
cmd_converge(char *text)
{
	struct ssum S;
	struct sctx C;
	struct verdict V;

	split_body(text, &S);
	if (!setup(&C, S.body, S.idx, S.lo, 1))
		fail("converge: the terms have other variables (radius for a power "
		    "series)");
	if (!S.lo[0]) {
		/* start where the terms are defined */
		int ok;
		C.lo = 1;
		xf_eval1(C.body, C.ns, 0, &ok);
		if (ok)
			C.lo = 0;
		xf_eval1(C.body, C.ns, 1, &ok);
		if (!ok)
			C.lo = 2;
	}
	printf("sum(%s, %lld, inf, %s)\n", C.idx, (long long)C.lo, xf_str(C.body));
	verdict(&C, &V, 0);
	printf("  %s: %s\n", verdict_word(V.v), V.test);
	{
		double s = 0, a[1000];
		uint8_t ok[1000];
		int i;
		terms(&C, C.lo, 1000, a, ok);
		for (i = 0; i < 1000; i++)
			s += ok[i] ? a[i] : 0;
		printf("  partial sums: S_10 = %.10g, S_1000 = %.10g\n",
		    a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8] + a[9],
		    s);
	}
	free(S.body);
}

/* ---- radius of convergence ---- */

/* the power: a node POW(base(x), e(n)) with e linear in n */
static nid
find_pow(nid e, int ns, int xs, nid *base, nid *ex)
{
	nid r;

	if (e == 0)
		return 0;
	while (N.op[e] == O_CAST)
		e = N.a[e];
	if (N.op[e] == O_POW && has_slot(N.a[e], xs) && !has_slot(N.a[e], ns) &&
	    has_slot(N.b[e], ns) && !has_slot(N.b[e], xs)) {
		*base = N.a[e];
		*ex = N.b[e];
		return e;
	}
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_LOOP)
		return 0;
	if ((r = find_pow(N.a[e], ns, xs, base, ex)) != 0)
		return r;
	if (opinfo[N.op[e]].arity >= 2 && (r = find_pow(N.b[e], ns, xs, base, ex)))
		return r;
	return 0;
}

/*
 * c1^(k1 n + j1) c2^(k2 n + j2) ... with constant bases combined into one
 * power, so that 3^n/3^n does not overflow at large n
 */
static nid
merge_pows(nid e, int ns, nid n)
{
	double lg = 0, cst = 1, v, sg = 1;
	nid rest = xk(1);
	nid stack[64];
	double pw[64];
	int sp = 0, any = 0;

	stack[sp] = e;
	pw[sp++] = 1;
	while (sp) {
		nid f = stack[--sp];
		double p = pw[sp];
		while (N.op[f] == O_CAST)
			f = N.a[f];
		if ((N.op[f] == O_MUL || N.op[f] == O_DIV) && sp < 62) {
			stack[sp] = N.a[f];
			pw[sp++] = p;
			stack[sp] = N.b[f];
			pw[sp++] = N.op[f] == O_DIV ? -p : p;
			continue;
		}
		if (N.op[f] == O_POW && xnum(N.a[f], &v) && v != 0) {
			struct npoly q;
			double k1, k0;
			if (npoly_of(N.b[f], ns, &q, 1) && q.deg == 1 &&
			    xnum(xadd(xk(0), q.c[1]), &k1) && xnum(xadd(xk(0), q.c[0]),
			    &k0) && k1 == floor(k1)) {
				lg += p * k1 * log(fabs(v));
				cst *= pow(fabs(v), p * k0);
				if (v < 0 && fmod(fabs(k1), 2) == 1)
					sg = -sg;
				if (v < 0 && fmod(fabs(k0), 2) == 1)
					cst = -cst;
				any = 1;
				continue;
			}
		}
		rest = p == 1 ? xmul(rest, f) : xdiv(rest, f);
	}
	if (!any)
		return e;
	{
		double base = exp(lg);
		nid r = xmul(xk(cst), rest);
		if (fabs(base - nearbyint(base)) < 1e-12 * base)
			base = nearbyint(base);
		if (base != 1)
			r = xmul(r, xpow(xk(base), n));
		if (sg < 0)
			r = xmul(r, xpow(xk(-1), n));
		return r;
	}
}

static void
cmd_radius(char *text)
{
	struct ssum S;
	struct sctx C;
	struct func *f;
	int xs = -1, k;
	nid P, base, ex, a;
	double c1, c0, k1, k0, R, x0, scale;
	struct verdict V;

	split_body(text, &S);
	setup(&C, S.body, S.idx, S.lo[0] ? S.lo : "0", 0);
	f = C.f;
	if (!S.lo[0]) {
		/* start where the coefficients are defined */
		int ok, j;
		for (j = 0; j < 3; j++) {
			{
				/* evaluate with every other slot at 0.5 */
				int sl[MAXP], k2, nn = 0;
				double xv2[MAXP];
				for (k2 = 0; k2 < f->np; k2++) {
					sl[nn] = k2;
					xv2[nn++] = k2 == C.ns ? j : 0.5;
				}
				xf_eval(C.body, sl, xv2, nn, &ok);
			}
			if (ok)
				break;
		}
		C.lo = j;
	}
	for (k = 0; k < f->np; k++)
		if (k != C.ns)
			xs = xs < 0 ? k : -2;
	if (xs < 0)
		fail("radius: expected a power series in one variable, like "
		    "x^n/n!");
	P = find_pow(C.body, C.ns, xs, &base, &ex);
	if (!P)
		fail("radius: no power (x - c)^(k n + j) found in %s",
		    xf_str(C.body));
	{
		struct npoly np;
		if (!npoly_of(base, xs, &np, 1) || np.deg != 1 ||
		    !xnum(xadd(xk(0), np.c[1]), &c1) || !xnum(xadd(xk(0), np.c[0]),
		    &c0))
			fail("radius: the base %s is not linear in %s", xf_str(base),
			    sym_str(f->p[xs].name));
		if (!npoly_of(ex, C.ns, &np, 1) || np.deg != 1 ||
		    !xnum(xadd(xk(0), np.c[1]), &k1) || !xnum(xadd(xk(0), np.c[0]),
		    &k0) || k1 <= 0)
			fail("radius: the exponent %s is not k n + j", xf_str(ex));
	}
	x0 = -c0 / c1;
	scale = fabs(c1);
	a = xreplace(C.body, P, xk(1));
	{
		struct sctx A = C;
		A.body = a;
		{
			double L = ratio_limit(&A);
			const char *test = "ratio test";
			if (isnan(L)) {
				/* root test */
				int ok;
				double v = fabs(eval_n(a, C.ns, 400, &ok));
				L = pow(v, 1.0 / 400);
				test = "root test";
			}
			if (L == 0)
				R = INFINITY;
			else if (isinf(L))
				R = 0;
			else
				R = snap_limit(pow(1 / L, 1 / k1) / scale);
			printf("power series sum(%s, %lld, inf, %s) in %s\n", C.idx,
			    (long long)C.lo, xf_str(C.body), sym_str(f->p[xs].name));
			printf("  center %s = %s; coefficients a_%s = %s of %s%s%s^(%s)\n",
			    sym_str(f->p[xs].name), xf_num(x0), C.idx, xf_str(a),
			    strchr(xf_str(base), ' ') ? "(" : "", xf_str(base),
			    strchr(xf_str(base), ' ') ? ")" : "", xf_str(ex));
			printf("  radius of convergence R = %s (%s: |a_(%s+1)/a_%s| -> %s",
			    isinf(R) ? "inf" : xf_num(R), test, C.idx, C.idx, xf_num(L));
			rep_exact("radius", R, isinf(R) ? "inf" : xf_num(R));
			rep_exact("center", x0, xf_num(x0));
			rep_str("coefficient", xf_str(a));
			rep_str("test", test);
			if (k1 != 1)
				printf(", the power steps by %s", xf_num(k1));
			printf(")\n");
		}
	}
	if (R == 0) {
		printf("  converges only at %s = %s\n", sym_str(f->p[xs].name),
		    xf_num(x0));
	} else if (isinf(R)) {
		printf("  interval of convergence: every %s (absolutely)\n",
		    sym_str(f->p[xs].name));
	} else {
		/* the endpoints */
		int e, res[2];
		const char *tst[2];
		char tbuf[2][240];
		for (e = 0; e < 2; e++) {
			double xe = x0 + (e ? R : -R);
			struct sctx E = C;
			E.body = merge_pows(xsubst1(C.body, xs, xk(xe)), C.ns, C.n);
			verdict(&E, &V, 0);
			res[e] = V.v;
			snprintf(tbuf[e], sizeof tbuf[e], "%s", V.test);
			tst[e] = tbuf[e];
			printf("  at %s = %s: %s (%s)\n", sym_str(f->p[xs].name),
			    xf_num(xe), verdict_word(V.v), tst[e]);
		}
		printf("  interval of convergence: %c%s, %s%c\n",
		    res[0] == CV_ABS || res[0] == CV_COND ? '[' : '(',
		    xf_num(x0 - R), xf_num(x0 + R),
		    res[1] == CV_ABS || res[1] == CV_COND ? ']' : ')');
	}
	{
		/* a closed form from the library: c_m for powers m of the base */
		double c[40];
		int m, ok, m0 = -1;
		struct libm L;
		memset(c, 0, sizeof c);
		for (m = 0; m < 40; m++) {
			double nn = (m - k0) / k1;
			if (nn != floor(nn) || nn < C.lo)
				continue;
			c[m] = eval_n(a, C.ns, nn, &ok);
			if (!ok)
				c[m] = 0;
			if (m0 < 0)
				m0 = m;
		}
		memset(&L, 0, sizeof L);
		if (m0 >= 0 && lib_match(c, 40, m0, &L)) {
			nid u = base, r = lib_closed(&L, m0, u, L.lam < 0);
			printf("  sum = %s    for |%s| < %s (%s)\n", xf_str(r),
			    xf_str(base), isinf(L.R) ? "inf" : xf_num(L.R),
			    lib_name[L.e]);
		}
	}
	free(S.body);
}

/* ---- power series of a function ---- */

static void
cmd_powerseries(char *text)
{
	static const char *const opts[] = {"at", "about", "around", NULL};
	char at[64] = "", *xname = NULL;
	struct func *f;
	double a = 0, c[17];
	int xs, m, nm = 0, ok;
	nid d, x, u;
	struct libm L;

	{
		/* "at a" */
		char *p = strstr(text, " at ");
		if (!p)
			p = strstr(text, " about ");
		if (p) {
			snprintf(at, sizeof at, "%s", strchr(p + 1, ' ') + 1);
			*p = 0;
		}
	}
	(void)opts;
	f = xf_parse(text);
	if (f->np != 1)
		fail("powerseries: one variable, please");
	xs = 0;
	xname = (char *)sym_str(f->p[0].name);
	if (at[0]) {
		struct func *fa = xf_parse(at);
		if (!xnum(fa->body, &a))
			fail("powerseries: the center must be a number");
	}
	x = xv(xs, xname);
	u = a == 0 ? x : xsub(x, xk(a));
	printf("power series of %s about %s = %s\n", xf_str(f->body), xname,
	    xf_num(a));
	/* rational: exact coefficients from the poles */
	{
		struct xrat r;
		if (xq_of(f->body, xs, &r) && r.d.deg > 0) {
			struct xpf pf;
			if (a != 0)
				goto taylor;
			/* G(x) = F; the coefficient of x^n: from X(z) = F(1/z) */
			if (xf_apart(&r, &pf) && pf.poly.deg < 1) {
				nid n = xv(1, "n"), an = xk(0);
				double R = INFINITY;
				int i;
				/* 1/(x - p)^k = (-1)^k/p^k sum C(n+k-1, k-1) (x/p)^n */
				for (i = 0; i < pf.np; i++) {
					double complex p = pf.pole[i].p;
					int k;
					R = fmin(R, cabs(p));
					if (cimag(p) != 0)
						goto taylor;
					for (k = 1; k <= pf.pole[i].m; k++) {
						double A = creal(pf.pole[i].a[k]);
						nid bin = xk(1);
						int j;
						if (A == 0)
							continue;
						for (j = 1; j < k; j++)
							bin = xmul(bin, xdiv(xadd(n, xk(j)), xk(j)));
						an = xadd(an, xmul(xmul(xk(A * (k & 1 ? -1 : 1) /
						    pow(creal(p), k)), bin), xpow(xk(1 / creal(p)), n)));
					}
				}
				if (pf.poly.deg == 0)
					printf("  (plus %s at n = 0)\n", xf_num(pf.poly.c[0]));
				printf("  = sum_{n>=0} a_n %s^n with a_n = %s\n", xname,
				    xf_str(an));
				printf("  radius of convergence %s (the nearest pole)\n",
				    xf_num(R));
				{
					double xs2 = R / 3, s = 0, xp = 1;
					int okk, i2;
					for (i2 = 0; i2 < 200; i2++, xp *= xs2) {
						double v = xf_eval1(an, 1, i2, &okk);
						s += v * xp + (i2 == 0 && pf.poly.deg == 0 ?
						    pf.poly.c[0] : 0);
					}
					printf("  check: the series at %s = %s sums to %.12g, f = "
					    "%.12g\n", xname, xf_num(xs2), s,
					    xf_eval1(f->body, xs, xs2, &okk));
				}
				return;
			}
		}
	}
taylor:
	d = f->body;
	for (m = 0; m <= 16; m++) {
		double v = xf_eval1(d, xs, a, &ok), fm = 1;
		int i;
		if (!ok || !isfinite(v))
			break;
		for (i = 2; i <= m; i++)
			fm *= i;
		c[m] = v / fm;
		if (fabs(c[m]) < 1e-14)
			c[m] = 0;
		nm = m + 1;
		{
			jmp_buf saved;
			int bad = 0;
			memcpy(saved, fail_jmp, sizeof saved);
			if (setjmp(fail_jmp) == 0)
				d = diff_node(d, (uint32_t)xs);
			else
				bad = 1;
			memcpy(fail_jmp, saved, sizeof saved);
			notes_drop();
			if (bad)
				break;
		}
	}
	if (nm > 13)
		nm = 13;   /* high derivatives lose accuracy */
	if (nm < 6)
		fail("powerseries: %s is not differentiable enough at %s = %s",
		    xf_str(f->body), xname, xf_num(a));
	printf("  Taylor coefficients c_0..c_7: ");
	for (m = 0; m < 8 && m < nm; m++)
		printf("%s%s", m ? ", " : "", xf_num(c[m]));
	printf("\n");
	memset(&L, 0, sizeof L);
	if (lib_match(c, nm, 0, &L)) {
		/* the general term: K lam^m l(m - j) u^m */
		nid mm = xv(1, "m"), term;
		switch (L.e) {
		case L_EXP: term = xdiv(xk(1), xfact(xsub(mm, xk(L.j)))); break;
		case L_COSHD: term = xdiv(xk(1), xfact(xmul(xk(2), xsub(mm,
		    xk(L.j))))); break;
		case L_SINHD: term = xdiv(xk(1), xfact(xadd(xmul(xk(2), xsub(mm,
		    xk(L.j))), xk(1)))); break;
		case L_GEOM: term = xk(1); break;
		case L_GEOM2: term = xadd(xsub(mm, xk(L.j)), xk(1)); break;
		case L_MLOG: term = xdiv(xk(1), xsub(mm, xk(L.j))); break;
		case L_ATANHD: term = xdiv(xk(1), xadd(xmul(xk(2), xsub(mm,
		    xk(L.j))), xk(1))); break;
		default: term = 0;
		}
		if (term) {
			term = xmul(xmul(xk(L.K), xpow(xk(L.lam), mm)), term);
			printf("  = sum_{m>=%d} c_m %s^m, c_m = %s\n", L.j + (L.e == L_MLOG),
			    xf_str(u), xf_str(term));
		} else
			printf("  = %s with u = %s %s\n", lib_name[L.e], xf_num(L.lam),
			    xf_str(u));
		printf("  matches %s: radius of convergence %s\n", lib_name[L.e],
		    isinf(L.R) ? "inf" : xf_num(L.R));
	} else {
		/* sparse in u^2: try c_(2k+1) and c_(2k) */
		double co[9], ce[9];
		int k;
		for (k = 0; k < 8 && 2 * k + 1 < nm; k++) {
			co[k] = c[2 * k + 1];
			ce[k] = c[2 * k];
		}
		memset(&L, 0, sizeof L);
		if (k >= 6 && (lib_match(co, k, 0, &L) || lib_match(ce, k, 0, &L))) {
			int odd = L.ok && lib_match(co, k, 0, &L);
			nid kk = xv(1, "k"), term = 0;
			if (!odd)
				lib_match(ce, k, 0, &L);
			switch (L.e) {
			case L_COSHD: term = xdiv(xk(1), xfact(xmul(xk(2), kk))); break;
			case L_SINHD: term = xdiv(xk(1), xfact(xadd(xmul(xk(2), kk),
			    xk(1)))); break;
			case L_ATANHD: term = xdiv(xk(1), xadd(xmul(xk(2), kk), xk(1)));
				break;
			case L_EXP: term = xdiv(xk(1), xfact(kk)); break;
			case L_GEOM: term = xk(1); break;
			}
			if (term && L.j == 0) {
				nid t = xmul(xmul(xk(L.K), xpow(xk(L.lam), kk)), term);
				printf("  = sum_{k>=0} %s * %s^(2k%s)\n", xf_str(t), xf_str(u),
				    odd ? " + 1" : "");
				printf("  radius of convergence %s\n", isinf(L.R) ? "inf" :
				    xf_num(sqrt(L.R)));
			} else
				printf("  no general term recognised\n");
		} else
			printf("  no general term recognised (the library: exp, sin, cos, "
			    "sinh, cosh, log, atan, atanh, geometric, binomial)\n");
	}
}

/* ---- asymptotic expansion of a partial sum ---- */

static void
cmd_asymptotic(char *text)
{
	struct ssum S;
	struct sctx C;
	nid F, f1, f3, nn, Fn, approx;
	const char *how;
	struct func *fh;
	int okk, k;
	double Cst;

	if (!parse_sum(text, &S) || S.isprod)
		fail("asymptotic: expected sum(k, 1, n, f(k))");
	setup(&C, S.body, S.idx, S.lo, 0);
	fh = xf_parse(S.hi);
	if (fh->np != 1)
		fail("asymptotic: the upper bound should be a variable, like n");
	if (C.f->np > 1)
		fail("asymptotic: the terms have other variables");
	nn = xv(C.ns + 1, sym_str(fh->p[0].name));
	F = antideriv(C.body, C.ns, &how);
	if (!F)
		fail("asymptotic: no antiderivative of %s", xf_str(C.body));
	/* |k| -> k: k >= lo >= 1 */
	F = xreplace(F, mk_un(O_ABS, C.n), C.n);
	f1 = diff_node(C.body, (uint32_t)C.ns);
	f3 = diff_node(diff_node(f1, (uint32_t)C.ns), (uint32_t)C.ns);
	Fn = xsubst1(F, C.ns, nn);
	{
		/* C = lim S(N) - [F(N) + f(N)/2 + B2/2 f'(N) + B4/24 f'''(N)] */
		double res[2];
		int Ns[2] = {2000, 4000}, p;
		for (p = 0; p < 2; p++) {
			double s = 0, comp = 0, a[4000];
			uint8_t ok[4000];
			int N = Ns[p], i;
			terms(&C, C.lo, N - C.lo + 1, a, ok);
			for (i = 0; i <= N - C.lo; i++) {
				/* Neumaier summation */
				double t = s + a[i];
				if (fabs(s) >= fabs(a[i]))
					comp += (s - t) + a[i];
				else
					comp += (a[i] - t) + s;
				s = t;
			}
			s += comp;
			res[p] = s - (xf_eval1(F, C.ns, N, &okk) + xf_eval1(C.body, C.ns,
			    N, &okk) / 2 + xf_eval1(f1, C.ns, N, &okk) / 12 -
			    xf_eval1(f3, C.ns, N, &okk) / 720);
		}
		Cst = res[1];
	}
	{
		nid t1 = xf_polish(xdiv(xsubst1(C.body, C.ns, nn), xk(2)), N.ki[nn], nn);
		nid t2 = xf_polish(xdiv(xsubst1(f1, C.ns, nn), xk(12)), N.ki[nn], nn);
		nid t3 = xf_polish(xdiv(xsubst1(f3, C.ns, nn), xk(-720)), N.ki[nn], nn);
		const char *cs = xf_ident_tol(Cst, 1e-10);
		struct sbuf b = {0};
		sb_printf(&b, "%s", xf_str(Fn));
		{
			const char *ts[3] = {xf_str(t1), xf_str(t2), xf_str(t3)};
			int grows;
			{
				int o1, o2;
				double v1 = fabs(xf_eval1(t1, N.ki[nn], 1e3, &o1));
				double v2 = fabs(xf_eval1(t1, N.ki[nn], 1e6, &o2));
				grows = o1 && o2 && v2 > v1;
			}
			if (!grows) {
				if (cs)
					sb_printf(&b, cs[0] == '-' ? " - %s" : " + %s",
					    cs[0] == '-' ? cs + 1 : cs);
				else
					sb_printf(&b, " + %.15g", Cst);
			}
			for (k = 0; k < 3; k++) {
				if (strcmp(ts[k], "0"))
					sb_printf(&b, ts[k][0] == '-' ? " - %s" : " + %s",
					    ts[k][0] == '-' ? ts[k] + 1 : ts[k]);
				if (k == 0 && (grows || !strcmp(ts[0], "0"))) {
					if (cs)
						sb_printf(&b, cs[0] == '-' ? " - %s" : " + %s",
						    cs[0] == '-' ? cs + 1 : cs);
					else
						sb_printf(&b, " + %.15g", Cst);
				}
			}
		}
		printf("sum(%s, %lld, %s, %s) ~ %s + ...\n", C.idx, (long long)C.lo,
		    sym_str(N.a[nn]), xf_str(C.body), b.p);
		printf("  Euler-Maclaurin: int f + C + f(n)/2 + B2/2! f'(n) + "
		    "B4/4! f'''(n), B2 = 1/6, B4 = -1/30\n");
		printf("  the constant C = %.15g%s%s\n", Cst, cs ? " = " : "",
		    cs ? cs : " (not recognised)");
		approx = xadd(xadd(xadd(Fn, xk(Cst)), t1), xadd(t2, t3));
		{
			int Ns[3] = {10, 100, 1000}, i;
			printf("  check:");
			for (i = 0; i < 3; i++) {
				double s = 0, a[1000];
				uint8_t ok[1000];
				int j;
				terms(&C, C.lo, Ns[i] - (int)C.lo + 1, a, ok);
				for (j = 0; j <= Ns[i] - C.lo; j++)
					s += a[j];
				printf("%s n = %d: sum %.12g, expansion off by %.2g", i ? ";" : "",
				    Ns[i], s, fabs(s - xf_eval1(approx, N.ki[nn], Ns[i], &okk)));
			}
			printf("\n");
		}
		free(b.p);
	}
}

/* ---- the commands ---- */

int
series_command(struct P *p, const char *w)
{
	uint32_t at = cmdp_pos(p);
	char *text;
	size_t cap = 4096;
	char word[32];

	text = xmalloc(cap);
	if (!strcmp(w, "sum") || !strcmp(w, "prod") || !strcmp(w, "Σ") ||
	    !strcmp(w, "∏")) {
		struct ssum S;
		if (!cmdp_rest(p, text, cap, NULL) || !parse_sum(text, &S) ||
		    !is_inf_text(S.hi)) {
			cmdp_seek(p, at);
			free(text);
			return 0;
		}
		xf_helpers();
		notes_flush();
		rep_begin(S.isprod ? "infinite_product" : "infinite_sum");
		rep_text_fields();
		if (S.isprod)
			infinite_prod(&S);
		else
			infinite_sum(&S);
		free(S.body);
		free(text);
		return 1;
	}
	if (!cmdp_word(p, word, sizeof word)) {
		free(text);
		return 0;
	}
	if (!cmdp_rest(p, text, cap, NULL))
		text[0] = 0;
	xf_helpers();
	notes_flush();
	rep_begin(!strcmp(word, "converges") ? "converge" : !strcmp(word,
	    "coeffs") ? "powerseries" : word);
	rep_text_fields();
	if (!strcmp(word, "converge") || !strcmp(word, "converges"))
		cmd_converge(text);
	else if (!strcmp(word, "radius"))
		cmd_radius(text);
	else if (!strcmp(word, "powerseries") || !strcmp(word, "coeffs"))
		cmd_powerseries(text);
	else if (!strcmp(word, "asymptotic"))
		cmd_asymptotic(text);
	else {
		cmdp_seek(p, at);
		free(text);
		return 0;
	}
	free(text);
	return 1;
}
