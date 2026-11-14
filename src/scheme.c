/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Cheap evaluation schemes, priced with the active cost model:
 *
 *   polynomials    naive (pow), powers by multiplication, Horner, Estrin
 *                  (cost and critical path); from degree 4 the forms with
 *                  fewer multiplications (Knuth's adapted coefficients,
 *                  Paterson and Stockmeyer's blocks and splitting) and the
 *                  cheapest once the model's fusions are counted; for f32
 *                  compensated Horner, measured against the exact value
 *                  and against its bound; forward
 *                  differences for tabulating at consecutive integers (d
 *                  additions a point)
 *   powers         x^n by squaring: the addition chain against pow()
 *   loops          iteration against the closed form (and the trip count
 *                  where the closed form wins), the matrix power of a
 *                  linear recurrence (O(log n) by squaring), and strength
 *                  reduction of a polynomial loop body by differences
 *
 * Every rewritten form is evaluated against the original on the domain.
 */
#include "calc.h"
#include "fuse.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- pricing a DAG: each node once; depth is the critical path ---- */

static float
dc1(const struct model *m, nid n, uint8_t *seen, float *dep, int *nops)
{
	float c = 0, d = 0, t;
	int cls;
	uint32_t j;

	if (n == 0)
		return 0;
	if (seen[n])
		return 0;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
		dep[n] = 0;
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			c += dc1(m, N.lst[N.a[n] + j], seen, dep, nops);
		dep[n] = 0;
		return c;
	case O_LOOP:
		c = loop_iter_cost(m, n);
		break;
	default:
		c = node_cost(m, n, &cls);
	}
	t = c;
	c += dc1(m, N.a[n], seen, dep, nops) + dc1(m, N.b[n], seen, dep, nops) +
	    dc1(m, N.c[n], seen, dep, nops);
	if (N.a[n] && dep[N.a[n]] > d)
		d = dep[N.a[n]];
	if (N.b[n] && dep[N.b[n]] > d)
		d = dep[N.b[n]];
	if (N.c[n] && N.op[n] != O_LOOP && dep[N.c[n]] > d)
		d = dep[N.c[n]];
	dep[n] = d + t;
	if (t > 0)
		(*nops)++;
	return c;
}

float
dag_cost(const struct model *m, nid root, int *nops, float *depth)
{
	uint8_t *seen = xcalloc(N.n, 1);
	float *dep = xcalloc(N.n, sizeof *dep), c;

	*nops = 0;
	c = dc1(m, root, seen, dep, nops);
	*depth = dep[root];
	free(seen);
	free(dep);
	return c;
}

/* ---- comparing two forms on the domain ---- */

struct cmpres {
	uint64_t n, bad;
	double maxrel, atzero;	/* atzero: the form's size where the value is 0 */
	int exact;
};

static void
compare(const struct func *f, nid a, nid b, struct cmpres *r)
{
	struct vres v;

	memset(r, 0, sizeof *r);
	if (!TY_FP(N.ty[a])) {
		verify_nodes(f, a, b, &v);
		r->n = v.n;
		r->bad = v.bad + v.covf;
		r->exact = 1;
		vres_free(&v);
		return;
	}
	{
		struct plan p;
		struct diter d;
		nid roots[3] = {a, mk_cast(N.ty[a], b), f->where};
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		uint8_t ok[CH];
		uint32_t np, i;
		plan_build(&p, roots, 3, 0);
		diter_init(&d, f, 1u << 16, 201, 4096);
		while ((np = diter_next(&d, in_i, in_r)) != 0) {
			plan_run(&p, np, in_i, in_r);
			valid_mask(&p, f->where ? p.root[2] : NOTOK, np, ok);
			for (i = 0; i < np; i++) {
				double x = COLR(&p, p.root[0])[i], y = COLR(&p, p.root[1])[i];
				if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF) ||
				    !isfinite(x))
					continue;
				r->n++;
				if (!isfinite(y)) {
					r->bad++;
					continue;
				}
				if (x == 0)
					r->atzero = fmax(r->atzero, fabs(y));
				else if (x != y)
					r->maxrel = fmax(r->maxrel, fabs(x - y) /
					    fabs(x));
			}
		}
		plan_free(&p);
		free(in_i);
		free(in_r);
	}
}

static void
print_cmp(const struct cmpres *r)
{
	if (r->exact)
		printf(r->bad ? "differs at %llu of %llu points" :
		    "equal at all %llu points checked", r->bad ? (unsigned long long)
		    r->bad : (unsigned long long)r->n, (unsigned long long)r->n);
	else if (r->bad)
		printf("NOT FINITE at %llu of %llu points",
		    (unsigned long long)r->bad, (unsigned long long)r->n);
	else if (r->maxrel == 0 && r->atzero == 0)
		printf("bit-identical at %llu points", (unsigned long long)r->n);
	else {
		printf("max relative difference %.2g (rounding) at %llu points",
		    r->maxrel, (unsigned long long)r->n);
		if (r->atzero != 0)
			printf("; %.2g where the value is 0", r->atzero);
	}
}

static void
line(const struct model *m, const char *name, nid e, const struct func *f,
    nid ref)
{
	int nops;
	float depth, c = dag_cost(m, e, &nops, &depth);
	struct cmpres r;

	printf("  %-10s cost %-6g depth %-5g %2d ops  %s\n", name, (double)c,
	    (double)depth, nops, node_str(e));
	if (ref && ref != e) {
		compare(f, ref, e, &r);
		printf("  %-10s ", "");
		print_cmp(&r);
		printf("\n");
	}
}

/* ---- polynomials ---- */

static nid
mkc(ty_t t, double v)
{
	if (!TY_FP(t))
		return mk_const_i(t, (int64_t)v);
	if (TYK(t) == K_FLOAT)
		return mk_cast(t, KR(v));
	return KR(v);
}

/*
 * ---- polynomials in fewer multiplications ----
 *
 * Three forms that trade multiplications for additions, so they pay where
 * a multiplication is dearer than an addition and nothing fuses the two:
 *
 *   Knuth      coefficients adapted beforehand (TAOCP 4.6.4): a quartic
 *              u4 ((y + x + a2) y + a3), y = (x + a0) x + a1, in three
 *              multiplications, a sextic in four (one root of a cubic picks
 *              its coefficients); higher degrees take Horner steps below one
 *   P-S blocks Paterson and Stockmeyer: the powers up to x^s once, a block
 *              of s coefficients over them, Horner in x^s; about 2 sqrt(n)
 *              multiplications of two variables, the rest by a coefficient
 *   P-S split  their monic splitting: p = (x^m + c) q + r with q and r monic
 *              of degree m - 1, over x^2, x^4, ...; n/2 + log2 n in all
 *
 * A form is built as a node and, beside it, as the polynomial it expands
 * to with its constants as the type rounds them: in whole numbers exactly
 * (an integer form is listed only when it is the polynomial itself, which
 * holds for every x of a wrapping type too), in floats to tell how far the
 * rounded adapted coefficients are from the polynomial.
 */
struct pe {
	nid e;
	int deg;
	double c[13];
	int64_t ci[13];
};

static struct {
	ty_t t;
	nid x;
	int fp, fail;
	int64_t ci0[13];	/* an integer polynomial's coefficients, exact */
} A;

static int
iadd_ok(int64_t a, int64_t b, int64_t *r)
{
	if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b))
		return 0;
	*r = a + b;
	return 1;
}

static int
imul_ok(int64_t a, int64_t b, int64_t *r)
{
	if (a == 0 || b == 0) {
		*r = 0;
		return 1;
	}
	if (a == INT64_MIN || b == INT64_MIN ||
	    llabs(a) > INT64_MAX / llabs(b))
		return 0;
	*r = a * b;
	return 1;
}

/* |v| as a constant of the type, and the value the type gives it */
static nid
pe_k(double av, double *r)
{
	nid k;

	if (!A.fp && (av != floor(av) || av > 9e15)) {
		A.fail = 1;
		av = 0;
	}
	k = mkc(A.t, av);
	if (!num_of(k, r))
		*r = av;
	return k;
}

static struct pe
pe_var(void)
{
	struct pe r;

	memset(&r, 0, sizeof r);
	r.e = A.x;
	r.deg = 1;
	r.c[1] = 1;
	r.ci[1] = 1;
	return r;
}

static struct pe
pe_neg(struct pe a)
{
	int i;

	a.e = mk_un(O_NEG, a.e);
	for (i = 0; i <= a.deg; i++) {
		a.c[i] = -a.c[i];
		a.ci[i] = -a.ci[i];
	}
	return a;
}

static struct pe
pe_addc(struct pe a, double v)
{
	double r;
	nid k;

	if (v == 0)
		return a;
	k = pe_k(fabs(v), &r);
	a.e = v < 0 ? SUB(a.e, k) : ADD(a.e, k);
	a.c[0] += v < 0 ? -r : r;
	if (!A.fp && !iadd_ok(a.ci[0], (int64_t)(v < 0 ? -r : r), &a.ci[0]))
		A.fail = 1;
	return a;
}

static struct pe
pe_const(double v)
{
	struct pe r;
	double q;

	memset(&r, 0, sizeof r);
	r.e = pe_k(fabs(v), &q);
	r.c[0] = q;
	r.ci[0] = A.fp ? 0 : (int64_t)q;
	return v < 0 ? pe_neg(r) : r;
}

static struct pe
pe_add2(struct pe a, struct pe b, int sub)
{
	int i;

	a.e = sub ? SUB(a.e, b.e) : ADD(a.e, b.e);
	if (b.deg > a.deg)
		a.deg = b.deg;
	for (i = 0; i <= a.deg; i++) {
		a.c[i] += sub ? -b.c[i] : b.c[i];
		if (!A.fp && !iadd_ok(a.ci[i], sub ? -b.ci[i] : b.ci[i],
		    &a.ci[i]))
			A.fail = 1;
	}
	return a;
}

static struct pe
pe_mul(struct pe a, struct pe b)
{
	struct pe r;
	int i, j;
	int64_t v;

	memset(&r, 0, sizeof r);
	r.e = MUL(a.e, b.e);
	r.deg = a.deg + b.deg;
	if (r.deg > 12) {
		A.fail = 1;
		r.deg = 12;
		return r;
	}
	for (i = 0; i <= a.deg; i++)
		for (j = 0; j <= b.deg; j++) {
			r.c[i + j] += a.c[i] * b.c[j];
			if (!A.fp && (!imul_ok(a.ci[i], b.ci[j], &v) ||
			    !iadd_ok(r.ci[i + j], v, &r.ci[i + j])))
				A.fail = 1;
		}
	return r;
}

/* v a: nothing for 1, a negation for -1, the constant first */
static struct pe
pe_mulc(double v, struct pe a)
{
	double r;
	nid k;
	int i;

	if (fabs(v) != 1) {
		k = pe_k(fabs(v), &r);
		a.e = MUL(k, a.e);
		for (i = 0; i <= a.deg; i++) {
			a.c[i] *= r;
			if (!A.fp && !imul_ok(a.ci[i], (int64_t)r, &a.ci[i]))
				A.fail = 1;
		}
	}
	return v < 0 ? pe_neg(a) : a;
}

/* a x + v */
static struct pe
pe_step(struct pe a, double v)
{
	return pe_addc(pe_mul(a, pe_var()), v);
}

/*
 * How far a form's expansion is from the polynomial c[0..d], as a part of
 * its largest coefficient; a whole-number form is 0 (the polynomial, modulo
 * the type's width) or -1.
 */
static double
pe_resid(const struct pe *a, const double *c, int d)
{
	double big = 0, worst = 0;
	int i, n = a->deg > d ? a->deg : d;

	if (A.fail)
		return -1;
	if (!A.fp) {
		uint64_t mask = TY_SIZED(A.t) && TYW(A.t) < 64 ?
		    ((uint64_t)1 << TYW(A.t)) - 1 : ~(uint64_t)0;
		for (i = 0; i <= n; i++) {
			uint64_t want = i <= d ? (uint64_t)A.ci0[i] : 0;
			if (((uint64_t)a->ci[i] - want) & mask)
				return -1;
		}
		return 0;
	}
	for (i = 0; i <= d; i++)
		big = fmax(big, fabs(c[i]));
	for (i = 0; i <= n; i++)
		worst = fmax(worst, fabs(a->c[i] - (i <= d ? c[i] : 0)));
	return isfinite(worst) ? worst / big : -1;
}

/* Knuth's quartic, u[0..4] */
static struct pe
knuth4(const double *u)
{
	double a0 = (u[3] / u[4] - 1) / 2, b = u[2] / u[4] - a0 * (a0 + 1),
	    a1 = u[1] / u[4] - a0 * b, a2 = b - 2 * a1,
	    a3 = u[0] / u[4] - a1 * (a1 + a2);
	struct pe x = pe_var(), y = pe_addc(pe_mul(pe_addc(x, a0), x), a1);

	return pe_mulc(u[4], pe_addc(pe_mul(pe_addc(pe_add2(y, x, 0), a2), y),
	    a3));
}

/*
 * Knuth's sextic, u[0..6]: z = (x + a0) x + a1, w = (x + a2) z + a3,
 * u6 ((w + z + a4) w + a5).  With v = u/u6, A = (v5 - 1)/2 and
 * b0 = (v4 - A^2 - A)/2, matching the powers of x leaves a0 = t a root of
 *   t^3 - (2A + 1) t^2 + (4 c0 + 2 (b0 - g0)) t + 4 (b0 g0 - v1),
 * c0 = v2 - b0^2 - A v3 + (2A^2 + A) b0, g0 = v3 - (2A + 1) b0, and the
 * other five follow from it.  A cubic has a real root; of up to three the
 * one with the smallest coefficients is taken, a whole one for integers.
 */
static struct pe
knuth6(const double *u)
{
	double v[6], Av, b0, c0, g0, p, q, r, lo, hi, t[3], bm = -1;
	double best[6] = {0};
	struct pe x = pe_var(), z, w;
	int i, k, nt = 1;

	for (i = 0; i < 6; i++)
		v[i] = u[i] / u[6];
	Av = (v[5] - 1) / 2;
	b0 = (v[4] - Av * Av - Av) / 2;
	c0 = v[2] - b0 * b0 - Av * v[3] + (2 * Av * Av + Av) * b0;
	g0 = v[3] - (2 * Av + 1) * b0;
	p = -(2 * Av + 1);
	q = 4 * c0 + 2 * (b0 - g0);
	r = 4 * (b0 * g0 - v[1]);
	hi = 1 + fmax(fabs(p), fmax(fabs(q), fabs(r)));
	lo = -hi;
	for (i = 0; i < 200; i++) {
		double mid = (lo + hi) / 2;
		if (((mid + p) * mid + q) * mid + r > 0)
			hi = mid;
		else
			lo = mid;
	}
	t[0] = (lo + hi) / 2;
	{
		/* the other two, of t^2 + (p + t0) t + (q + t0 (p + t0)) */
		double b1 = p + t[0], c1 = q + t[0] * b1, disc = b1 * b1 - 4 * c1;
		if (disc >= 0) {
			double s = -(b1 + (b1 < 0 ? -sqrt(disc) : sqrt(disc))) / 2;
			t[nt++] = s;
			if (s != 0)
				t[nt++] = c1 / s;
		}
	}
	for (k = 0; k < nt; k++) {
		double tt = A.fp ? t[k] : floor(t[k] + 0.5), a[6], B, C, mx = 0;
		B = b0 - tt / 2;
		a[0] = tt;
		a[2] = Av - tt;
		a[1] = B - tt * a[2];
		C = c0 - Av / 2 * tt + tt * tt / 4;
		a[4] = v[3] - 2 * Av * B - 2 * C - B - tt * Av - a[1];
		a[3] = C - a[1] * a[2];
		a[5] = v[0] - C * C - a[1] * C - a[4] * C;
		for (i = 0; i < 6; i++)
			mx = fmax(mx, fabs(a[i]));
		if (!isfinite(mx))
			continue;
		if (!A.fp)
			for (i = 0; i < 6; i++)
				if (a[i] != floor(a[i]))
					mx = -1;
		if (mx >= 0 && (bm < 0 || mx < bm)) {
			bm = mx;
			memcpy(best, a, sizeof best);
		}
	}
	if (bm < 0) {
		A.fail = 1;
		return x;
	}
	z = pe_addc(pe_mul(pe_addc(x, best[0]), x), best[1]);
	w = pe_addc(pe_mul(pe_addc(x, best[2]), z), best[3]);
	return pe_mulc(u[6], pe_addc(pe_mul(pe_addc(pe_add2(w, z, 0), best[4]),
	    w), best[5]));
}

/* the adapted form of the top core + 1 coefficients, Horner below it */
static struct pe
knuth(const double *c, int d, int core)
{
	struct pe a = core == 6 ? knuth6(c + d - 6) : knuth4(c + d - 4);
	int k;

	for (k = d - core - 1; k >= 0; k--)
		a = pe_step(a, c[k]);
	return a;
}

/* blocks of s coefficients over 1, x, ..., x^(s-1), Horner in x^s */
static struct pe
ps_blocks(const double *c, int d, int s)
{
	struct pe pw[13], acc, blk;
	int i, j, have = 0;

	pw[1] = pe_var();
	for (i = 2; i <= s; i++)
		pw[i] = pe_mul(pw[i / 2], pw[i - i / 2]);
	memset(&acc, 0, sizeof acc);
	for (j = d / s; j >= 0; j--) {
		int hb = 0;
		memset(&blk, 0, sizeof blk);
		for (i = s - 1; i >= 0; i--) {
			double v = j * s + i <= d ? c[j * s + i] : 0;
			struct pe term;
			if (v == 0)
				continue;
			if (i == 0) {
				blk = hb ? pe_addc(blk, v) : pe_const(v);
				hb = 1;
				continue;
			}
			term = pe_mulc(fabs(v), pw[i]);
			blk = !hb ? (v < 0 ? pe_neg(term) : term) :
			    pe_add2(blk, term, v < 0);
			hb = 1;
		}
		if (!have) {
			acc = blk;	/* the top block holds c[d] */
			have = 1;
			continue;
		}
		acc = acc.deg == 0 ? pe_mulc(A.fp ? acc.c[0] :
		    (double)acc.ci[0], pw[s]) : pe_mul(acc, pw[s]);
		if (hb && blk.deg == 0)
			acc = pe_addc(acc, A.fp ? blk.c[0] : (double)blk.ci[0]);
		else if (hb)
			acc = pe_add2(acc, blk, 0);
	}
	return acc;
}

/* p[0..n] monic, n + 1 a power of two; xp[m] = x^m for m a power of two */
static struct pe
ps_monic(const double *p, int n, const struct pe *xp)
{
	double q[8], r[8], cc;
	int m = (n + 1) / 2, i;

	if (n == 1)
		return pe_addc(xp[1], p[0]);
	cc = p[m - 1] - 1;
	for (i = 0; i < m; i++) {
		q[i] = p[m + i];
		r[i] = p[i] - cc * q[i];
	}
	r[m - 1] = 1;
	return pe_add2(pe_mul(pe_addc(xp[m], cc), ps_monic(q, m - 1, xp)),
	    ps_monic(r, m - 1, xp), 0);
}

/*
 * The splitting on the top n + 1 coefficients, n the largest 2^k - 1 up
 * to d, made monic by the leading one; Horner below it.
 */
static struct pe
ps_split(const double *c, int d)
{
	struct pe xp[9], a;
	double p[13];
	int n = 1, i, k;

	while (2 * n + 1 <= d)
		n = 2 * n + 1;
	for (i = 0; i <= n; i++) {
		p[i] = c[d - n + i] / c[d];
		if (!A.fp && p[i] != floor(p[i]))
			A.fail = 1;
	}
	p[n] = 1;
	xp[1] = pe_var();
	for (i = 2; i <= (n + 1) / 2; i *= 2)
		xp[i] = pe_mul(xp[i / 2], xp[i / 2]);
	a = pe_mulc(c[d], ps_monic(p, n, xp));
	for (k = d - n - 1; k >= 0; k--)
		a = pe_step(a, c[k]);
	return a;
}

/*
 * ---- compensated Horner ----
 *
 * Horner's rule with the rounding error of every product (2Prod: an fma)
 * and every sum (2Sum, Knuth) kept and run through a second Horner, then
 * added back: s + c.  Graillat, Langlois and Louvet ("Algorithms for
 * accurate, validated and fast polynomial evaluation"): where nothing
 * underflows the result is as good as Horner's in twice the precision
 * rounded once,
 *   |comp - p(x)| <= u |p(x)| + gamma_2n^2 p~(|x|),
 * against Horner's gamma_2n p~(|x|), with p~ the polynomial of the
 * coefficients' absolute values and gamma_k = k u / (1 - k u).  Both are
 * measured against the exact value and checked against these bounds.
 */
static nid
comp_horner(const double *c, int d)
{
	nid s = mkc(A.t, c[d]), cc = 0, p, pi, sg, a1, b1, k, e;
	int i;

	for (i = d - 1; i >= 0; i--) {
		if (i == d - 1 && c[d] == 1) {
			p = A.x;	/* 1 x is exact */
			pi = 0;
		} else {
			p = MUL(s, A.x);
			pi = mk_fma(s, A.x, mk_un(O_NEG, p));
		}
		if (c[i] != 0) {
			k = mkc(A.t, c[i]);
			s = ADD(p, k);
			a1 = SUB(s, k);
			b1 = SUB(s, a1);
			sg = ADD(SUB(p, a1), SUB(k, b1));
			e = pi ? ADD(pi, sg) : sg;
		} else {
			s = p;
			e = pi;
		}
		if (cc && e)
			cc = ADD(MUL(cc, A.x), e);
		else if (cc)
			cc = MUL(cc, A.x);
		else
			cc = e;
	}
	return cc ? ADD(s, cc) : s;
}

/* the f32 polynomial c[0..d] at x in doubles, compensated: its error is
 * far under an f32 form's, so it stands for the exact value */
static double
poly_ref(const double *c, int d, double x)
{
	double s = c[d], cc = 0, p, pi, t, bb, sg;
	int i;

	for (i = d - 1; i >= 0; i--) {
		p = s * x;
		pi = fma(s, x, -p);
		t = p + c[i];
		bb = t - p;
		sg = (p - (t - bb)) + (c[i] - bb);
		s = t;
		cc = cc * x + (pi + sg);
	}
	return s + cc;
}

/*
 * The compensated form's line for an f32 polynomial: what it costs, and
 * its error and Horner's against the exact value on the domain, each
 * with its bound.
 */
static void
comp_scheme(const struct model *m, const struct func *f, const double *c,
    int d, nid horner)
{
	const double u = 0x1p-24, g = 2 * d * u / (1 - 2 * d * u);
	nid comp = comp_horner(c, d), h64, roots[4];
	double eh = 0, ec = 0, bc = 0;
	uint64_t n = 0, over = 0;
	struct plan p;
	struct diter it;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint8_t ok[CH];
	uint32_t np, i;
	int nops, k;
	float depth, cost;

	roots[0] = horner;
	roots[1] = comp;
	roots[2] = A.x;
	roots[3] = f->where;
	plan_build(&p, roots, 4, 0);
	diter_init(&it, f, 1u << 16, 201, 4096);
	while ((np = diter_next(&it, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[3] : NOTOK, np, ok);
		for (i = 0; i < np; i++) {
			double x = COLR(&p, p.root[2])[i], ref, pt = 0, dh, dc;
			if (!ok[i] || !isfinite(x))
				continue;
			ref = poly_ref(c, d, x);
			for (k = d; k >= 0; k--)
				pt = pt * fabs(x) + fabs(c[k]);
			dh = fabs(COLR(&p, p.root[0])[i] - ref);
			dc = fabs(COLR(&p, p.root[1])[i] - ref);
			if (!isfinite(ref) || !isfinite(pt) || !isfinite(dh) ||
			    !isfinite(dc))
				continue;
			if (pt == 0)
				continue;
			n++;
			eh = fmax(eh, dh / pt);
			ec = fmax(ec, dc / pt);
			bc = fmax(bc, u * fabs(ref) / pt + g * g);
			if (dh > g * pt || dc > u * fabs(ref) + g * g * pt)
				over++;
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	cost = dag_cost(m, comp, &nops, &depth);
	printf("  %-10s cost %-6g depth %-5g %2d ops  compensated Horner, s + "
	    "c: Horner's s, and the rounding error of each product (an fma) "
	    "and each sum (2Sum) run through a second Horner c\n", "comp.",
	    (double)cost, (double)depth, nops);
	if (n == 0) {
		printf("  %-10s no finite point of the domain to measure it "
		    "on\n", "");
		return;
	}
	printf("  %-10s error against the exact polynomial at %llu points, as "
	    "a part of p~(|x|) (the polynomial of the coefficients' sizes): "
	    "%.2g, Horner's %.2g; the bounds are %.2g = max u |p|/p~ + "
	    "gamma_%d^2 and %.2g = gamma_%d\n", "", (unsigned long long)n, ec,
	    eh, bc, 2 * d, g, 2 * d);
	if (over)
		printf("  %-10s a bound is exceeded at %llu points (it holds "
		    "where nothing underflows)\n", "", (unsigned long long)over);
	else
		printf("  %-10s it gains where the polynomial cancels (|p| far "
		    "under p~): its first term is the last rounding alone\n",
		    "");
	/* the other way to that accuracy: Horner in f64 */
	h64 = mk_cast(TY_F64, mk_const_r(c[d]));
	for (k = d - 1; k >= 0; k--) {
		h64 = MUL(h64, mk_cast(TY_F64, A.x));
		if (c[k] != 0)
			h64 = ADD(h64, mk_cast(TY_F64, mk_const_r(c[k])));
	}
	fuse_domain(f);
	printf("  %-10s with %s's fusions counted: %g, against %g for Horner "
	    "in f64 and %g in f32; its error-free products and sums must "
	    "stay as written (NoContraction)\n", "", sym_str(m->name),
	    (double)fuse_expr_cost(m, comp), (double)fuse_expr_cost(m, h64),
	    (double)fuse_expr_cost(m, horner));
	fuse_domain(NULL);
}

/* a form's line, and under it what its expansion is */
static void
line_pe(const struct model *m, const char *name, const struct pe *a,
    double resid, const struct func *f)
{
	line(m, name, a->e, f, f->body);
	if (resid == 0)
		printf("  %-10s expands to the polynomial itself%s\n", "",
		    A.fp ? "" : ": equal for every x");
	else
		printf("  %-10s expands to the polynomial within %.2g of its "
		    "largest coefficient (the adapted ones are rounded)\n", "",
		    resid);
}

/* coefficient k as a constant of the type, or its magnitude (mag) */
static nid
coef_k(ty_t t, const double *c, const int64_t *ki, int k, int mag)
{
	if (TY_FP(t))
		return mkc(t, mag ? fabs(c[k]) : c[k]);
	/* an unsigned one is never negative: added as it is (an IMAD) */
	return mk_const_i(t, mag && ki[k] < 0 && TYK(t) != K_UINT ?
	    (int64_t)(0 - (uint64_t)ki[k]) : ki[k]);
}

/* -v of a constant node k whose value is hv */
static nid
neg_k(ty_t t, nid k, double hv)
{
	if (TY_FP(t) || !is_const(k))
		return mkc(t, -hv);
	return mk_const_i(t, (int64_t)(0 - (uint64_t)N.ki[k]));
}

static int
poly_scheme(const struct model *m, struct func *f, int more)
{
	struct npoly P;
	double c[13];
	int s, d, k;
	ty_t t = N.ty[f->body];
	nid x, naive = 0, mulp = 0, horner, estrin, pw[13], forms[8];
	const char *names[8];
	double cr[13], tol;
	int64_t ki[13];
	int nf = 3, shown = 0;

	for (s = 0; s < f->np; s++)
		if (npoly_of(f->body, s, &P, 12) && P.deg >= 2)
			break;
	if (s == f->np)
		return 0;
	for (k = 0; k <= P.deg; k++)
		if (!num_of(P.c[k], &c[k]))
			return 0;
	/*
	 * an integer type's coefficients exactly, as its constants hold them
	 * (a double holds them only to 2^53); c keeps their sign and size
	 */
	for (k = 0; k <= P.deg && !TY_FP(t); k++) {
		nid q = P.c[k];
		int w = TY_SIZED(t) ? TYW(t) : 64;
		uint64_t u = 0;
		if (is_const(q) && !TY_FP(N.ty[q]))
			u = (uint64_t)N.ki[q];
		else if (fabs(c[k]) < 0x1p63)
			u = (uint64_t)(int64_t)c[k];
		else
			fail("scheme: the coefficient of x^%d, %.17g, is past what "
			    "it reads exactly (|c| < 2^63, or a constant of the "
			    "type)", k, c[k]);
		if (w < 64) {
			u &= ((uint64_t)1 << w) - 1;
			if (TYK(t) == K_SINT && u >> (w - 1))
				u -= (uint64_t)1 << w;
		}
		ki[k] = (int64_t)u;
		c[k] = TYK(t) == K_UINT ? (double)u : (double)ki[k];
	}
	d = P.deg;
	x = var_of(s, f->p[s].name, f->p[s].ty);
	if (N.ty[x] != t)
		x = mk_cast(t, x);
	printf("polynomial of degree %d in %s:\n", d, sym_str(f->p[s].name));
	/* naive with pow, and with a chain of multiplications */
	pw[1] = x;
	for (k = 2; k <= d; k++)
		pw[k] = MUL(pw[k - 1], x);
	for (k = d; k >= 0; k--) {
		nid a, b;
		double ac = fabs(c[k]);
		if (c[k] == 0)
			continue;
		a = k == 0 ? coef_k(t, c, ki, k, 1) : ac == 1 ? POW(x, KI(k)) :
		    MUL(coef_k(t, c, ki, k, 1), POW(x, KI(k)));
		b = k == 0 ? coef_k(t, c, ki, k, 1) : ac == 1 ? pw[k] :
		    MUL(coef_k(t, c, ki, k, 1), pw[k]);
		if (k == 1)
			a = ac == 1 ? x : MUL(coef_k(t, c, ki, k, 1), x);
		if (c[k] < 0) {
			naive = naive ? SUB(naive, a) : mk_un(O_NEG, a);
			mulp = mulp ? SUB(mulp, b) : mk_un(O_NEG, b);
		} else {
			naive = naive ? ADD(naive, a) : a;
			mulp = mulp ? ADD(mulp, b) : b;
		}
	}
	horner = coef_k(t, c, ki, d, 0);
	for (k = d - 1; k >= 0; k--) {
		horner = MUL(horner, x);
		if (c[k] > 0)
			horner = ADD(horner, coef_k(t, c, ki, k, 0));
		else if (c[k] < 0)
			horner = SUB(horner, coef_k(t, c, ki, k, 1));
	}
	/* Estrin: pairs a + b x, then combine with x^2, x^4, ... */
	{
		nid lev[13], xp = x;
		int n = d + 1, j;
		for (j = 0; j < n; j++)
			lev[j] = coef_k(t, c, ki, j, 0);
		while (n > 1) {
			int nn = 0;
			for (j = 0; j < n; j += 2) {
				double hv;
				nid hi;
				if (j + 1 >= n) {
					lev[nn++] = lev[j];
					continue;
				}
				/* lev[j] + lev[j+1] * xp, written without -1* and + -c */
				if (num_of(lev[j + 1], &hv) && fabs(hv) == 1)
					hi = xp;
				else if (num_of(lev[j + 1], &hv) && hv < 0)
					hi = MUL(neg_k(t, lev[j + 1], hv), xp);
				else
					hi = MUL(lev[j + 1], xp), hv = 1;
				if (const_is(lev[j], 0))
					lev[nn++] = hv < 0 ? mk_un(O_NEG, hi) : hi;
				else if (hv > 0 && num_of(lev[j], &hv) && hv < 0)
					lev[nn++] = SUB(hi, neg_k(t, lev[j], hv));
				else
					lev[nn++] = hv < 0 ? SUB(lev[j], hi) : ADD(lev[j], hi);
			}
			n = nn;
			xp = MUL(xp, xp);
		}
		estrin = lev[0];
	}
	line(m, "as written", f->body, f, 0);
	line(m, "pow()", naive, f, f->body);
	line(m, "products", mulp, f, f->body);
	line(m, "Horner", horner, f, f->body);
	line(m, "Estrin", estrin, f, f->body);
	names[0] = "products";
	forms[0] = mulp;
	names[1] = "Horner";
	forms[1] = horner;
	names[2] = "Estrin";
	forms[2] = estrin;
	/* fewer multiplications: adapted coefficients, blocks, the splitting */
	A.t = t;
	A.x = x;
	A.fp = TY_FP(t);
	if (!A.fp)
		memcpy(A.ci0, ki, (size_t)(d + 1) * sizeof *ki);
	for (k = 0; k <= d; k++)
		if (!num_of(mkc(t, c[k]), &cr[k]))
			cr[k] = c[k];
	/* an adapted form may be off by 64 roundings of a coefficient: past
	 * that its coefficients have cancelled and it is not the polynomial */
	tol = !A.fp ? 0 : TYK(t) == K_FLOAT && TYW(t) == 32 ? 0x1p-18 :
	    TYK(t) == K_FLOAT && TYW(t) == 16 ? 0x1p-5 : 0x1p-47;
	if (more && d >= 4) {
		struct pe a, best;
		double r = -1, rb = 0;
		float cb = -1, cc;
		int core, sb = 0;

		memset(&a, 0, sizeof a);
		for (core = d >= 6 ? 6 : 4; core >= 4; core -= 2) {
			A.fail = 0;
			a = knuth(cr, d, core);
			r = pe_resid(&a, cr, d);
			if (r >= 0 && r <= tol)
				break;
		}
		if (core >= 4) {
			line_pe(m, "Knuth", &a, r, f);
			names[nf] = "Knuth";
			forms[nf++] = a.e;
			shown |= 1;
		} else if (A.fp && r >= 0)
			printf("  %-10s not listed: its adapted coefficients "
			    "cancel (the expansion is off by %.2g of the largest "
			    "coefficient)\n", "Knuth", r);
		else
			printf("  %-10s not listed: its adapted coefficients are "
			    "not %s here\n", "Knuth", A.fp ? "finite" :
			    "whole numbers");
		fuse_domain(f);		/* the block size that prices lowest */
		for (k = 2; k <= (d + 1) / 2; k++) {
			A.fail = 0;
			a = ps_blocks(cr, d, k);
			r = pe_resid(&a, cr, d);
			if (r < 0 || r > tol)
				continue;
			cc = fuse_expr_cost(m, a.e);
			if (cb < 0 || cc < cb) {
				cb = cc;
				best = a;
				rb = r;
				sb = k;
			}
		}
		fuse_domain(NULL);
		if (sb) {
			line_pe(m, "P-S blocks", &best, rb, f);
			names[nf] = "P-S blocks";
			forms[nf++] = best.e;
			shown |= 2;
		}
		if (d >= 7) {
			A.fail = 0;
			a = ps_split(cr, d);
			r = pe_resid(&a, cr, d);
			if (r >= 0 && r <= tol) {
				line_pe(m, "P-S split", &a, r, f);
				names[nf] = "P-S split";
				forms[nf++] = a.e;
				shown |= 4;
			} else if (A.fp)
				printf("  %-10s not listed: its coefficients cancel "
				    "(the expansion is off by %.2g of the largest "
				    "coefficient)\n", "P-S split", r);
			else
				printf("  %-10s not listed: the leading coefficient "
				    "does not divide the others\n", "P-S split");
		}
	}
	printf("  (depth is the critical path: Estrin trades operations for "
	    "parallelism)\n");
	if (shown & 1)
		printf("  (Knuth: coefficients adapted beforehand, fewer "
		    "multiplications and more additions than Horner%s)\n",
		    A.fp ? "; its sums cancel, so it is less exact near a zero "
		    "of the polynomial" : "");
	if (shown & 6)
		printf("  (P-S: Paterson and Stockmeyer;%s%s%s)\n", shown & 2 ?
		    " blocks of coefficients over the first powers, Horner in "
		    "the next" : "", (shown & 6) == 6 ? ";" : "", shown & 4 ?
		    " the splitting (x^m + c)*q + r into monic halves" : "");
	if (more && TYK(t) == K_FLOAT && TYW(t) == 32)
		comp_scheme(m, f, cr, d, horner);
	if (more && d >= 4) {
		/* what the model's fusions make of each: Horner is all FFMAs */
		float cost[8];
		int bi = 0;

		fuse_domain(f);
		for (k = 0; k < nf; k++) {
			cost[k] = fuse_expr_cost(m, forms[k]);
			if (cost[k] < cost[bi])
				bi = k;
		}
		fuse_domain(NULL);
		printf("  cheapest with %s's fusions counted: %s %g;", sym_str(
		    m->name), names[bi], (double)cost[bi]);
		for (k = 0; k < nf; k++)
			if (k != bi)
				printf("%s %s %g", k == (bi == 0) ? "" : ",",
				    names[k], (double)cost[k]);
		printf("\n");
	}
	/* forward differences for consecutive integers */
	if (!TY_FP(f->p[s].ty) && f->np == 1) {
		int64_t lo, hi, D[13], v, j, n, bad = 0;
		int64_t vals[14];
		param_bounds(&f->p[s], &lo, &hi);
		if (hi - lo > 100000)
			hi = lo + 100000;
		for (j = 0; j <= d; j++) {
			double acc = 0, xv = (double)(lo + j);
			for (k = d; k >= 0; k--)
				acc = acc * xv + c[k];
			vals[j] = (int64_t)acc;
		}
		for (j = 0; j <= d; j++)
			D[j] = vals[j];
		for (k = 1; k <= d; k++)
			for (j = d; j >= k; j--)
				D[j] -= D[j - 1];
		printf("  differences, for %s = %lld, %lld, ...: start", sym_str(
		    f->p[s].name), (long long)lo, (long long)lo + 1);
		for (j = 0; j <= d; j++)
			printf(" d%lld = %lld", (long long)j, (long long)D[j]);
		printf("\n    per point: ");
		for (j = 0; j < d; j++)
			printf("d%lld += d%lld; ", (long long)j, (long long)j + 1);
		printf("(%d additions, cost %g a point after the start)\n", d,
		    (double)(d * m->c[TY_FP(t) ? C_FADD : C_IADD]));
		/* check: run the difference engine against the polynomial */
		n = 0;
		for (v = lo; v <= hi; v++, n++) {
			double acc = 0;
			for (k = d; k >= 0; k--)
				acc = acc * (double)v + c[k];
			if (fabs(acc) < 9e15 && (int64_t)acc != D[0])
				bad++;
			for (j = 0; j < d; j++)
				D[j] += D[j + 1];
		}
		if (bad)
			printf("    check: differs at %lld of %lld points\n",
			    (long long)bad, (long long)n);
		else
			printf("    check: equals the polynomial at all %lld points "
			    "%lld..%lld\n", (long long)n, (long long)lo, (long long)hi);
	}
	return 1;
}

/*
 * ---- inner products: Winograd ----
 *
 * sum a_i b_i over n = 2k pairs is
 *   sum_j (a_2j + b_2j+1)(a_2j+1 + b_2j) - A - B,
 *   A = sum_j a_2j a_2j+1,  B = sum_j b_2j b_2j+1:
 * an identity of the ring (each product's cross terms are a_2j b_2j +
 * a_2j+1 b_2j+1, the rest is A and B), so it holds for every x of an
 * integer type, through wrapping too.  k multiplications of two values
 * once A and B are known: in a matrix product A is a row's, B a column's,
 * each computed once and shared by every dot product that row or column
 * takes part in.  Alone it costs 3k for the plain form's 2k.
 */
#define WINO_MAX 16

/* the signed terms of e, each a product of two parameters */
static int
wino_terms(nid e, int neg, nid *a, nid *b, int *n)
{
	switch (N.op[e]) {
	case O_ADD:
		return wino_terms(N.a[e], neg, a, b, n) &&
		    wino_terms(N.b[e], neg, a, b, n);
	case O_SUB:
		return wino_terms(N.a[e], neg, a, b, n) &&
		    wino_terms(N.b[e], !neg, a, b, n);
	case O_NEG:
		return wino_terms(N.a[e], !neg, a, b, n);
	case O_MUL:
		if (*n == WINO_MAX || N.op[N.a[e]] != O_VAR ||
		    N.op[N.b[e]] != O_VAR)
			return 0;
		a[*n] = N.a[e];
		b[*n] = neg ? mk_un(O_NEG, N.b[e]) : N.b[e];
		(*n)++;
		return 1;
	default:
		return 0;
	}
}

static int
prod_scheme(const struct model *m, struct func *f, int more)
{
	nid a[WINO_MAX], b[WINO_MAX], w = 0, A = 0, B = 0, wg, va, vb, last = 0;
	int n = 0, j, k;
	float p0, p1, p2;
	ty_t t = N.ty[f->body];

	if (!more || !wino_terms(f->body, 0, a, b, &n) || n < 2)
		return 0;
	k = n / 2;
	for (j = 0; j < k; j++) {
		nid pr = MUL(ADD(a[2 * j], b[2 * j + 1]), ADD(a[2 * j + 1],
		    b[2 * j]));
		w = w ? ADD(w, pr) : pr;
		A = A ? ADD(A, MUL(a[2 * j], a[2 * j + 1])) :
		    MUL(a[2 * j], a[2 * j + 1]);
		B = B ? ADD(B, MUL(b[2 * j], b[2 * j + 1])) :
		    MUL(b[2 * j], b[2 * j + 1]);
	}
	if (n & 1)
		last = MUL(a[n - 1], b[n - 1]);
	va = mk_var(0, sym_cstr("A"), t);
	vb = mk_var(0, sym_cstr("B"), t);
	wg = SUB(SUB(w, va), vb);
	w = SUB(SUB(w, A), B);
	if (last) {
		w = ADD(w, last);
		wg = ADD(wg, last);
	}
	printf("inner product of %d pairs (Winograd's form from 2):\n", n);
	line(m, "as written", f->body, f, 0);
	line(m, "Winograd", w, f, f->body);
	if (!TY_FP(t))
		printf("  %-10s an identity of the ring: equal for every x, "
		    "through wrapping\n", "");
	printf("  %-10s A = %s, B = %s\n", "", node_str(A), node_str(B));
	fuse_domain(f);
	p0 = fuse_expr_cost(m, f->body);
	p1 = fuse_expr_cost(m, w);
	p2 = fuse_expr_cost(m, wg);
	fuse_domain(NULL);
	printf("  %-10s with A and B given: %s (cost %g with %s's fusions: "
	    "%d multiplication%s of two values for %d)\n", "", node_str(wg),
	    (double)p2, sym_str(m->name), k + (n & 1), k + (n & 1) == 1 ? "" :
	    "s", n);
	printf("  (Winograd: A is a row's and B a column's, each computed "
	    "once in a matrix product and shared by its dot products; alone "
	    "it is dearer%s)\n", TY_FP(t) ? "; in floats its sums cancel, "
	    "and the line above says by how much" : "");
	printf("  cheapest with %s's fusions counted: ", sym_str(m->name));
	if (p2 < p0)
		printf("Winograd with A and B given %g; as written %g, Winograd "
		    "%g\n", (double)p2, (double)p0, (double)p1);
	else
		printf("as written %g; Winograd %g, with A and B given %g\n",
		    (double)p0, (double)p1, (double)p2);
	return 1;
}

/*
 * ---- complex and quaternion products ----
 *
 * (a + bi)(c + di) in Gauss's three products: k1 = c (a + b), k2 = a (d -
 * c), k3 = b (c + d), re = k1 - k3, im = k1 + k2; with d - c and c + d
 * computed once for a w that many z meet, three multiplications and three
 * additions.  A quaternion product in 8 multiplications for 16 (and one
 * halving), from the identity
 *   A1 = (d1 + b1)(b2 + c2), A3 = (a1 - c1)(a2 + d2), A4 = (a1 + c1)(a2 - d2),
 *   A5 = (A1 + A3 + A4 + (d1 - b1)(b2 - c2))/2,
 *   1: A5 - A1 + (d1 - c1)(c2 - d2), i: A5 - A1 - A3 - A4 + (b1 + a1)(b2 +
 *   a2), j: A5 - A3 + (a1 - b1)(c2 + d2), k: A5 - A4 + (d1 + c1)(a2 - b2).
 * Both are identities of the reals: the parts are checked against the
 * componentwise product on [-2, 2], and each set of parts is priced as one
 * evaluation (what the parts share counted once).
 */
static void
hparts(struct func *t, const struct func *f, int pi, int dim, ty_t ty,
    nid *v)
{
	static const char *const sx[2][4] = {{"_re", "_im"}, {"_1", "_i",
	    "_j", "_k"}};
	char nm[48];
	int k;

	for (k = 0; k < dim; k++) {
		int s = t->np;
		snprintf(nm, sizeof nm, "%s%s", sym_str(f->p[pi].name),
		    sx[dim == 4][k]);
		t->p[s].name = sym_cstr(nm);
		t->p[s].ty = ty;
		t->p[s].ranged = 1;
		t->p[s].rlo = -2;
		t->p[s].rhi = 2;
		v[k] = mk_var((uint32_t)s, t->p[s].name, ty);
		t->np++;
	}
}

static void
hline(const struct model *m, const char *name, const nid *r, const nid *ref,
    int dim, const struct func *t)
{
	static const char *const pn[2][4] = {{"re", "im"}, {"1", "i", "j",
	    "k"}};
	struct cmpres cr;
	int k;

	printf("  %-10s cost %g with %s's fusions (the parts together)\n", name,
	    (double)fuse_roots_cost(m, r, dim), sym_str(m->name));
	for (k = 0; k < dim; k++) {
		printf("    %-3s %s\n", pn[dim == 4][k], node_str(r[k]));
		if (ref && ref[k] != r[k]) {
			compare(t, ref[k], r[k], &cr);
			printf("        ");
			print_cmp(&cr);
			printf("\n");
		}
	}
}

void
cmd_hscheme(struct func *f, struct model *mm, int more)
{
	const struct model *m = mm ? mm : model_active();
	nid z, w, p[4], q[4], cw[4], g[4];
	struct func t;
	int dim = 1 << f->halg, zi = -1, wi = -1, j;
	ty_t ty = f->hf32 ? TY_F32 : TY_REAL;

	printf("evaluation schemes for %s under %s\n", f->anon ?
	    node_str(f->body) : sym_str(f->name), sym_str(m->name));
	z = N.a[f->body];
	w = N.b[f->body];
	if (N.op[f->body] == O_MUL && N.op[z] == O_VAR && N.op[w] == O_VAR)
		for (j = 0; j < f->np; j++) {
			if (f->p[j].name == N.a[z] && f->hk[j] == f->halg)
				zi = j;
			if (f->p[j].name == N.a[w] && f->hk[j] == f->halg)
				wi = j;
		}
	if ((dim != 2 && dim != 4) || zi < 0 || wi < 0 || zi == wi) {
		printf("  no product of two different %s parameters to rewrite "
		    "(scheme takes F = z*w)\n", dim == 2 ? "complex" :
		    dim == 4 ? "quaternion" : "complex or quaternion");
		return;
	}
	memset(&t, 0, sizeof t);
	hparts(&t, f, zi, dim, ty, p);
	hparts(&t, f, wi, dim, ty, q);
	t.name = f->name;
	t.anon = f->anon;
	if (dim == 2) {
		nid k1, k2, k3, e1, e2;
		cw[0] = SUB(MUL(p[0], q[0]), MUL(p[1], q[1]));
		cw[1] = ADD(MUL(p[0], q[1]), MUL(p[1], q[0]));
		printf("complex product, as its parts:\n");
		hline(m, "4 products", cw, NULL, 2, &t);
		if (!more)
			return;
		k1 = MUL(q[0], ADD(p[0], p[1]));
		k2 = MUL(p[0], SUB(q[1], q[0]));
		k3 = MUL(p[1], ADD(q[0], q[1]));
		g[0] = SUB(k1, k3);
		g[1] = ADD(k1, k2);
		hline(m, "Gauss", g, cw, 2, &t);
		e1 = mk_var(0, sym_cstr("e1"), ty);
		e2 = mk_var(0, sym_cstr("e2"), ty);
		k2 = MUL(p[0], e1);
		k3 = MUL(p[1], e2);
		g[0] = SUB(k1, k3);
		g[1] = ADD(k1, k2);
		printf("  %-10s with e1 = %s and e2 = %s given (a w that many z "
		    "meet): cost %g\n", "", node_str(SUB(q[1], q[0])),
		    node_str(ADD(q[0], q[1])), (double)fuse_roots_cost(m, g, 2));
	} else {
		nid A1, A3, A4, A5;
		cw[0] = SUB(SUB(SUB(MUL(p[0], q[0]), MUL(p[1], q[1])), MUL(p[2],
		    q[2])), MUL(p[3], q[3]));
		cw[1] = SUB(ADD(ADD(MUL(p[0], q[1]), MUL(p[1], q[0])), MUL(p[2],
		    q[3])), MUL(p[3], q[2]));
		cw[2] = ADD(ADD(SUB(MUL(p[0], q[2]), MUL(p[1], q[3])), MUL(p[2],
		    q[0])), MUL(p[3], q[1]));
		cw[3] = ADD(SUB(ADD(MUL(p[0], q[3]), MUL(p[1], q[2])), MUL(p[2],
		    q[1])), MUL(p[3], q[0]));
		printf("quaternion product, as its parts:\n");
		hline(m, "16 products", cw, NULL, 4, &t);
		if (!more)
			return;
		A1 = MUL(ADD(p[3], p[1]), ADD(q[1], q[2]));
		A3 = MUL(SUB(p[0], p[2]), ADD(q[0], q[3]));
		A4 = MUL(ADD(p[0], p[2]), SUB(q[0], q[3]));
		A5 = MUL(mkc(ty, 0.5), ADD(ADD(ADD(A1, A3), A4),
		    MUL(SUB(p[3], p[1]), SUB(q[1], q[2]))));
		g[0] = ADD(SUB(A5, A1), MUL(SUB(p[3], p[2]), SUB(q[2], q[3])));
		g[1] = ADD(SUB(A5, ADD(ADD(A1, A3), A4)), MUL(ADD(p[1], p[0]),
		    ADD(q[1], q[0])));
		g[2] = ADD(SUB(A5, A3), MUL(SUB(p[0], p[1]), ADD(q[2], q[3])));
		g[3] = ADD(SUB(A5, A4), MUL(ADD(p[3], p[2]), SUB(q[0], q[1])));
		hline(m, "8 products", g, cw, 4, &t);
	}
	printf("  (%s; its sums of products cancel, and each part's line "
	    "says by how much)\n", dim == 2 ? "Gauss: three multiplications "
	    "for four, three more additions" : "eight multiplications for "
	    "sixteen, more additions and a halving");
}

/* ---- powers by squaring ---- */

static nid
pow_chain(nid x, int64_t e)
{
	nid r = 0, sq = x;

	while (e) {
		if (e & 1)
			r = r ? MUL(r, sq) : sq;
		e >>= 1;
		if (e)
			sq = MUL(sq, sq);
	}
	return r;
}

static void
find_pows(nid e, nid *out, int *n, int max, uint8_t *seen)
{
	if (e == 0 || seen[e] || *n >= max)
		return;
	seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_TABLE ||
	    N.op[e] == O_LOOP)
		return;
	if (N.op[e] == O_POW)
		out[(*n)++] = e;
	find_pows(N.a[e], out, n, max, seen);
	find_pows(N.b[e], out, n, max, seen);
	find_pows(N.c[e], out, n, max, seen);
}

static void
pow_scheme(const struct model *m, struct func *f)
{
	nid pows[8];
	int n = 0, i;
	uint8_t *seen = xcalloc(N.n, 1);

	find_pows(f->body, pows, &n, 8, seen);
	free(seen);
	for (i = 0; i < n; i++) {
		nid p = pows[i], ch;
		int64_t e;
		int cls, nops;
		float cp, cc, dep;
		if (!int_of(N.b[p], &e)) {
			nid x = N.b[p];
			while (N.op[x] == O_CAST)
				x = N.a[x];
			if (!TY_FP(N.ty[x]))
				printf("%s: an integer power by squaring takes at most "
				    "2 log2(%s) multiplications\n", node_str(p),
				    node_str(x));
			continue;
		}
		if (e < 3 || e > 1000000)
			continue;
		ch = pow_chain(N.a[p], e);
		cp = node_cost(m, p, &cls);
		cc = dag_cost(m, ch, &nops, &dep);
		printf("%s by squaring: %d multiplications (%d squarings), cost %g "
		    "against %g for the power\n", node_str(p), nops,
		    ilog2u((uint64_t)e), (double)cc, (double)cp);
		printf("  %s\n", node_str(ch));
		{
			struct func g = *f;
			struct cmpres r;
			g.body = p;
			compare(&g, p, ch, &r);
			printf("  ");
			print_cmp(&r);
			printf("\n");
		}
	}
}

/* ---- loops ---- */

static void
find_loops2(nid e, nid *out, int *n, int max, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || *n >= max)
		return;
	seen[e] = 1;
	if (N.op[e] == O_LOOP) {
		if ((node_slots(e) >> MAXP) == 0)
			out[(*n)++] = e;
		return;
	}
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			find_loops2(N.lst[N.a[e] + j], out, n, max, seen);
		return;
	}
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return;
	find_loops2(N.a[e], out, n, max, seen);
	find_loops2(N.b[e], out, n, max, seen);
	find_loops2(N.c[e], out, n, max, seen);
}

/* the largest trip count over the domain */
static double
max_trips(const struct func *f, nid n)
{
	struct plan p;
	struct diter d;
	nid T = ADD(SUB(N.b[n], N.a[n]), KI(1)), roots[2] = {T, f->where};
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r), mx = 0;
	uint32_t np, i;
	uint8_t ok[CH];

	if (loops.kind[loop_ent(n)] != LK_FOR)
		return -1;
	plan_build(&p, roots, 2, 0);
	diter_init(&d, f, 1u << 16, 101, 4096);
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[1] : NOTOK, np, ok);
		for (i = 0; i < np; i++)
			if (ok[i] && !(COLF(&p, p.root[0])[i] & F_UNDEF))
				mx = fmax(mx, (double)COLI(&p, p.root[0])[i]);
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return mx;
}

/* integer matrix products in int64; 0 on overflow */
static int
imatmul(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
	int i, j, l;
	int64_t s, t;

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {
			s = 0;
			for (l = 0; l < n; l++)
				if (__builtin_mul_overflow(a[i * n + l], b[l * n + j], &t) ||
				    __builtin_add_overflow(s, t, &s))
					return 0;
			c[i * n + j] = s;
		}
	return 1;
}

/* M^k v by squaring; exactly in int64 when everything is an integer */
static int
matpow(const double *M, const double *v, int n, int64_t k, int res,
    double *out, int exact, int64_t *iout)
{
	double A[256], R[256], T[256], x[16];
	int i, j, l;

	if (exact) {
		int64_t IA[256], IR[256], IT[256], s, t;
		for (i = 0; i < n * n; i++) {
			IA[i] = (int64_t)M[i];
			IR[i] = (i / n == i % n);
		}
		while (k > 0) {
			if (k & 1) {
				if (!imatmul(IR, IA, IT, n))
					return 0;
				memcpy(IR, IT, (size_t)n * n * sizeof *IR);
			}
			k >>= 1;
			if (k) {
				if (!imatmul(IA, IA, IT, n))
					return 0;
				memcpy(IA, IT, (size_t)n * n * sizeof *IA);
			}
		}
		s = 0;
		for (l = 0; l < n; l++)
			if (__builtin_mul_overflow(IR[res * n + l], (int64_t)v[l], &t) ||
			    __builtin_add_overflow(s, t, &s))
				return 0;
		*iout = s;
		return 1;
	}

	for (i = 0; i < n * n; i++) {
		A[i] = M[i];
		R[i] = (i / n == i % n);
	}
	while (k > 0) {
		if (k & 1) {
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++) {
					double s = 0;
					for (l = 0; l < n; l++)
						s += R[i * n + l] * A[l * n + j];
					T[i * n + j] = s;
				}
			memcpy(R, T, sizeof T);
		}
		k >>= 1;
		if (k) {
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++) {
					double s = 0;
					for (l = 0; l < n; l++)
						s += A[i * n + l] * A[l * n + j];
					T[i * n + j] = s;
				}
			memcpy(A, T, sizeof T);
		}
	}
	for (i = 0; i < n; i++) {
		double s = 0;
		for (l = 0; l < n; l++)
			s += R[i * n + l] * v[l];
		x[i] = s;
	}
	*out = x[res];
	return 1;
}

/* does e have a loop in it */
static int
has_loop(nid e, int depth)
{
	uint32_t j;

	if (e == 0 || depth > 64)
		return 0;
	switch (N.op[e]) {
	case O_CONST: case O_VAR: case O_UNDEF:
		return 0;
	case O_LOOP:
		return 1;
	case O_TABLE:
		for (j = 0; j < N.b[e]; j++)
			if (has_loop(N.lst[N.a[e] + j], depth + 1))
				return 1;
		return 0;
	}
	return has_loop(N.a[e], depth + 1) || has_loop(N.b[e], depth + 1) ||
	    has_loop(N.c[e], depth + 1);
}

/* a state for d that no variable of the loop has: its index, -1 if none */
static int
add_state(struct loopdef *d, uint64_t *used, const char *name, ty_t t,
    nid init)
{
	int slot;

	if (d->nst == MAXST)
		return -1;
	for (slot = MAXP; slot < KSLOT && (*used >> slot & 1); slot++)
		;
	if (slot >= KSLOT)
		return -1;
	*used |= 1ULL << slot;
	d->sslot[d->nst] = slot;
	d->sname[d->nst] = sym_cstr(name);
	d->sty[d->nst] = t;
	d->init[d->nst] = init;
	d->next[d->nst] = 0;
	return d->nst++;
}

/*
 * Kahan's compensated sum for a counted loop's f32 accumulator, s = s + t
 * with t free of s: y = t - c, w = s + y, c = (w - s) - y, s = w, the
 * rounding error of each add carried into the next (#47).  It is run on
 * the domain beside the loop as written, and both are compared with the
 * same f32 terms summed in f64, as a part of the sum of the terms' sizes:
 * the plain sum's error there is at most gamma_n (n adds), Kahan's
 * 2u + O(n u^2) (Knuth; Higham, Accuracy and Stability, 4.3).  The price
 * is the model's, an iteration, beside the accumulator in f64.
 */
static void
kahan_scheme(const struct model *m, const struct func *f, nid n)
{
	const double u = 0x1p-24;
	struct loopdef d, k, r, w;
	uint64_t used, pts = 0, over = 0;
	int res = loop_res(n), j, ic, ir, ia, sub = 0;
	nid sv, t, cv, y, wv, nk, nr, na, n64, roots[6], trips;
	struct plan p;
	struct diter it;
	int64_t *in_i;
	double *in_r, ep = 0, ek = 0, gmax = 0, tmax = 0;
	uint8_t ok[CH];
	uint32_t np, i;
	float per, perk, per64;

	loop_get(loop_ent(n), &d);
	if (d.kind != LK_FOR || res < 0 || res >= d.nst || d.sty[res] != TY_F32)
		return;
	sv = var_of(d.sslot[res], d.sname[res], d.sty[res]);
	t = d.next[res];
	if (N.op[t] == O_ADD && N.a[t] == sv)
		t = N.b[t];
	else if (N.op[t] == O_ADD && N.b[t] == sv)
		t = N.a[t];
	else if (N.op[t] == O_SUB && N.a[t] == sv) {
		t = N.b[t];
		sub = 1;
	} else
		return;
	if (N.ty[t] != TY_F32 || has_slot(t, d.sslot[res]))
		return;
	/* nothing else reads the accumulator, and no loop inside holds a
	 * variable the new states could take */
	for (j = 0; j < d.nst; j++)
		if ((j != res && has_slot(d.next[j], d.sslot[res])) ||
		    has_loop(d.next[j], 0) || has_loop(d.init[j], 0))
			return;
	used = loops.bound[loop_ent(n)] | loops.free[loop_ent(n)] |
	    node_slots(n);
	if (sub)
		t = mk_un(O_NEG, t);
	/* the compensated loop: s and c */
	k = d;
	if ((ic = add_state(&k, &used, "c", TY_F32, mk_cast(TY_F32,
	    mk_const_r(0)))) < 0)
		return;
	cv = var_of(k.sslot[ic], k.sname[ic], TY_F32);
	y = SUB(t, cv);
	wv = ADD(sv, y);
	k.next[res] = wv;
	k.next[ic] = SUB(SUB(wv, sv), y);
	nk = mk_loop(loop_intern(&k), res, N.a[n], N.b[n], N.c[n]);
	/* the reference: the same f32 terms summed in f64, and their sizes */
	r = d;
	if ((ir = add_state(&r, &used, "ref", TY_F64, mk_cast(TY_F64,
	    d.init[res]))) < 0 || (ia = add_state(&r, &used, "size", TY_F64,
	    mk_un(O_ABS, mk_cast(TY_F64, d.init[res])))) < 0)
		return;
	r.next[ir] = ADD(var_of(r.sslot[ir], r.sname[ir], TY_F64),
	    mk_cast(TY_F64, t));
	r.next[ia] = ADD(var_of(r.sslot[ia], r.sname[ia], TY_F64),
	    mk_un(O_ABS, mk_cast(TY_F64, t)));
	nr = mk_loop(loop_intern(&r), ir, N.a[n], N.b[n], N.c[n]);
	na = mk_loop(loop_intern(&r), ia, N.a[n], N.b[n], N.c[n]);
	/* the accumulator in f64, for its price */
	w = d;
	w.sty[res] = TY_F64;
	w.init[res] = mk_cast(TY_F64, d.init[res]);
	w.next[res] = ADD(var_of(d.sslot[res], d.sname[res], TY_F64),
	    mk_cast(TY_F64, t));
	n64 = mk_loop(loop_intern(&w), res, N.a[n], N.b[n], N.c[n]);

	trips = ADD(SUB(N.b[n], N.a[n]), KI(1));
	roots[0] = n;
	roots[1] = nk;
	roots[2] = nr;
	roots[3] = na;
	roots[4] = trips;
	roots[5] = f->where;
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	plan_build(&p, roots, 6, 0);
	diter_init(&it, f, 1u << 16, 101, 4096);
	while ((np = diter_next(&it, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[5] : NOTOK, np, ok);
		for (i = 0; i < np; i++) {
			double ref = COLR(&p, p.root[2])[i], a = COLR(&p,
			    p.root[3])[i], dp, dk, g, T;
			if (!ok[i] || ((COLF(&p, p.root[0])[i] | COLF(&p,
			    p.root[1])[i] | COLF(&p, p.root[2])[i] | COLF(&p,
			    p.root[4])[i]) & F_UNDEF))
				continue;
			T = (double)COLI(&p, p.root[4])[i];
			dp = fabs(COLR(&p, p.root[0])[i] - ref);
			dk = fabs(COLR(&p, p.root[1])[i] - ref);
			if (!isfinite(ref) || !isfinite(dp) || !isfinite(dk) ||
			    !(a > 0) || !isfinite(a) || T < 1)
				continue;
			pts++;
			g = T * u / (1 - T * u);
			ep = fmax(ep, dp / a);
			ek = fmax(ek, dk / a);
			gmax = fmax(gmax, g);
			tmax = fmax(tmax, T);
			if (dp > g * a)
				over++;
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	per = loop_iter_cost(m, n);
	perk = loop_iter_cost(m, nk);
	per64 = loop_iter_cost(m, n64);
	printf("  %-10s Kahan's sum of the f32 accumulator %s: y = t - c, w = "
	    "%s + y, c = (w - %s) - y, %s = w, the rounding error c of each "
	    "add carried into the next: %g per iteration against %g, and %g "
	    "with %s in f64\n", "comp.", sym_str(d.sname[res]),
	    sym_str(d.sname[res]), sym_str(d.sname[res]), sym_str(d.sname[res]),
	    (double)perk, (double)per, (double)per64, sym_str(d.sname[res]));
	if (pts == 0) {
		printf("  %-10s no finite point of the domain to measure it "
		    "on\n", "");
		return;
	}
	printf("  %-10s error against the same f32 terms summed in f64 at %llu "
	    "points, as a part of the sum of the terms' sizes: %.2g, the "
	    "plain sum's %.2g; the bounds are 2u + O(n u^2) (2u = %.2g) and "
	    "gamma_n = n u/(1 - n u), %.2g for the %.0f adds at most here\n",
	    "", (unsigned long long)pts, ek, ep, 2 * u, gmax, tmax);
	if (over)
		printf("  %-10s the plain sum is past its bound at %llu points "
		    "(it holds where nothing underflows)\n", "",
		    (unsigned long long)over);
	else if (ek < ep)
		printf("  %-10s it gains where the plain sum's errors add up "
		    "(many terms, or terms that cancel); its own error is the "
		    "last rounding\n", "");
	else
		printf("  %-10s it gains nothing here: its error is not below "
		    "the plain sum's\n", "");
	printf("  %-10s its adds must stay as written: reassociated, c is "
	    "0\n", "");
}

static void
loop_scheme(const struct model *m, struct func *f, nid n, int more)
{
	float per = loop_iter_cost(m, n), cc;
	double T = max_trips(f, n), Mx[256], v0[16];
	nid cf;
	int nops, dim, res, j;
	float dep;

	printf("%s\n", node_str(n));
	if (T >= 0)
		printf("  iterating: %g per iteration, up to %g iterations: %g\n",
		    (double)per, T, (double)per * T);
	else
		printf("  iterating: %g per iteration (a while loop)\n",
		    (double)per);
	cf = loop_closed(n, NULL, NULL);
	if (cf) {
		struct vres v;
		cf = tidy(cf);
		cc = dag_cost(m, cf, &nops, &dep);
		verify_nodes(f, n, cf, &v);
		printf("  closed form: %g (%d ops): %s\n", (double)cc, nops,
		    node_str(cf));
		if (v.bad || !v.n)
			printf("    but it differs from the loop at %llu of %llu points "
			    "(e.g. %s): not used\n", (unsigned long long)v.bad,
			    (unsigned long long)v.n, sb_str(&v.wit));
		else {
			printf("    matches the loop at %llu points; ",
			    (unsigned long long)v.n);
			if (per > 0)
				printf("cheaper than iterating from %.0f iterations on\n",
				    ceil(cc / per));
			else
				printf("\n");
		}
		vres_free(&v);
	}
	if (lin_rec_matrix(n, Mx, v0, &dim, &res) && dim >= 2) {
		int exact = 1, bad = 0, chk = 0, skipped = 0;
		float mc = dim * dim * dim * m->c[C_IMUL] + dim * dim * (dim - 1) *
		    m->c[C_IADD];
		double steps = T > 1 ? ceil(log2(T)) : 1;
		ty_t st = N.ty[n];
		if (TY_FP(st))
			mc = dim * dim * dim * m->c[C_FMUL] + dim * dim * (dim - 1) *
			    m->c[C_FADD];
		for (j = 0; j < dim * dim; j++)
			if (Mx[j] != floor(Mx[j]) || fabs(Mx[j]) > 1e15)
				exact = 0;
		for (j = 0; j < dim; j++)
			if (v0[j] != floor(v0[j]) || fabs(v0[j]) > 1e15)
				exact = 0;
		if (TY_FP(st))
			exact = 0;
		printf("  matrix form: state' = M state, M = [");
		for (j = 0; j < dim * dim; j++)
			printf("%s%g", j == 0 ? "" : j % dim ? " " : "; ", Mx[j]);
		printf("]; the answer is (M^k v0)[%d]\n", res);
		printf("    M^k by squaring: at most %g matrix products of %g "
		    "each: %g (O(log k))\n", 2 * steps, (double)mc,
		    2 * steps * mc);
		/* check against the loop at sample points */
		{
			struct plan p;
			struct diter d;
			nid Tn = ADD(SUB(N.b[n], N.a[n]), KI(1));
			nid roots[3] = {n, Tn, f->where};
			int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
			double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
			uint8_t ok[CH];
			uint32_t np, i;
			plan_build(&p, roots, 3, 0);
			diter_init(&d, f, 1u << 12, 51, 512);
			while ((np = diter_next(&d, in_i, in_r)) != 0) {
				plan_run(&p, np, in_i, in_r);
				valid_mask(&p, f->where ? p.root[2] : NOTOK, np, ok);
				for (i = 0; i < np; i++) {
					double want, got = 0;
					int64_t k = COLI(&p, p.root[1])[i], ig = 0;
					if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF) ||
					    k < 0)
						continue;
					want = TY_FP(st) ? COLR(&p, p.root[0])[i] :
					    (double)COLI(&p, p.root[0])[i];
					if (!matpow(Mx, v0, dim, k, res, &got, exact, &ig)) {
						skipped++;
						continue;
					}
					chk++;
					if (exact ? ig != COLI(&p, p.root[0])[i] :
					    fabs(got - want) > 1e-9 * fmax(1, fabs(want)))
						bad++;
				}
			}
			plan_free(&p);
			free(in_i);
			free(in_r);
		}
		if (bad)
			printf("    check: differs from the loop at %d of %d points\n",
			    bad, chk);
		else
			printf("    check: equals the loop at %d points%s\n", chk,
			    exact ? " (in exact int64 arithmetic: no rounding, "
			    "unlike the closed form's real powers)" : "");
		if (skipped)
			printf("    (%d points not checked: the matrix power overflows "
			    "int64)\n", skipped);
		if (per > 0)
			printf("    cheaper than iterating once k > about %.0f\n",
			    ceil(2 * steps * mc / per));
	}
	if (more)
		kahan_scheme(m, f, n);
	/* strength reduction of a polynomial sum body */
	{
		struct loopdef d;
		struct npoly P;
		loop_get(loop_ent(n), &d);
		if (d.orig == LO_SUM && npoly_of(d.aux, d.islot, &P, 12) &&
		    P.deg >= 1) {
			ty_t t = N.ty[d.aux];
			int dg = P.deg;
			float hc = dg * (m->c[TY_FP(t) ? C_FMUL : C_IMUL] +
			    m->c[TY_FP(t) ? C_FADD : C_IADD]);
			float dc = dg * m->c[TY_FP(t) ? C_FADD : C_IADD];
			printf("  strength reduction: the body is a degree-%d "
			    "polynomial in %s; keep its %d differences and update "
			    "them by additions\n", dg, sym_str(d.name), dg);
			printf("    per iteration: Horner %g, differences %g\n",
			    (double)hc, (double)dc);
		}
	}
}

void
cmd_scheme(struct func *f, struct model *mm, int more)
{
	const struct model *m = mm ? mm : model_active();
	nid loops[8];
	int nl = 0, i, any;
	uint8_t *seen;

	printf("evaluation schemes for %s under %s\n", f->anon ?
	    node_str(f->body) : sym_str(f->name), sym_str(m->name));
	any = poly_scheme(m, f, more);
	if (!any)
		any = prod_scheme(m, f, more);
	if (!any)
		pow_scheme(m, f);
	seen = xcalloc(N.n, 1);
	find_loops2(f->body, loops, &nl, 8, seen);
	free(seen);
	for (i = 0; i < nl; i++)
		loop_scheme(m, f, loops[i], more);
	if (!any && nl == 0) {
		int nops;
		float dep, c = dag_cost(m, f->body, &nops, &dep);
		printf("  as written: cost %g, depth %g, %d operations (no "
		    "polynomial, power or loop to rewrite)\n", (double)c,
		    (double)dep, nops);
	}
	notes_drop();
}
