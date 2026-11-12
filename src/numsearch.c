/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * numsearch: stochastic searches over integer offsets.
 *
 * approx rounds a fit's coefficients to the machine format and walks
 * each a few ulps while the measured error drops (a coordinate descent:
 * it stops at the first point where no single small step helps).  The
 * two searches here move several coefficients at once, from that point:
 *
 * anneal   simulated annealing: one coefficient moves by a few ulps; a
 *          worse point is accepted with probability exp(-dE / T), T
 *          falling geometrically from t0 to t1 times the starting
 *          objective over the budget.
 *
 * cmaes    CMA-ES (Hansen): a Gaussian over the offsets whose mean,
 *          step and covariance follow the better half of each
 *          generation; a sample is rounded to whole ulps before it is
 *          measured.  The recombination weights are linear, not the
 *          usual logarithmic ones, the normal deviates are sums of
 *          twelve uniforms, and exp is computed here: nothing in the
 *          loop comes from libm but sqrt, which IEEE rounds correctly,
 *          so a seed gives the same run everywhere.
 *
 * Both only ever report a point they measured, and the best one found
 * (offset 0, the walk's result, included): the objective is the
 * caller's, here the largest error over approx's sample.
 */
#include "numsearch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void
ns_seed(struct nsrng *g, uint64_t seed)
{
	g->s = seed * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
	if (g->s == 0)
		g->s = 1;
}

uint64_t
ns_u64(struct nsrng *g)
{
	g->s ^= g->s >> 12;
	g->s ^= g->s << 25;
	g->s ^= g->s >> 27;
	return g->s * 0x2545F4914F6CDD1DULL;
}

double
ns_unif(struct nsrng *g)
{
	return (double)(ns_u64(g) >> 11) * (1.0 / 9007199254740992.0);
}

/* e^x by scaling and squaring of a Taylor polynomial: + * / only */
double
ns_exp(double x)
{
	double t = 1, s = 1, y;
	int k, neg = x < 0;

	if (neg)
		x = -x;
	if (x > 700)
		return neg ? 0 : INFINITY;
	y = x / 1024;
	for (k = 1; k <= 12; k++) {
		t *= y / k;
		s += t;
	}
	for (k = 0; k < 10; k++)
		s *= s;
	return neg ? 1 / s : s;
}

/* a standard normal deviate: twelve uniforms less six (Irwin-Hall) */
static double
ns_normal(struct nsrng *g)
{
	double s = 0;
	int i;

	for (i = 0; i < 12; i++)
		s += ns_unif(g);
	return s - 6;
}

void
ns_defaults(struct nsopt *o)
{
	memset(o, 0, sizeof *o);
	o->budget = 2000;
	o->t0 = 0.1;
	o->t1 = 0.001;
	o->pop = 0;
	o->sigma = 2;
	o->radius = 64;
	o->seed = 1;
}

const char *
ns_name(int kind)
{
	return kind == NS_ANNEAL ? "simulated annealing" : kind == NS_CMAES ?
	    "CMA-ES" : "none";
}

static void
anneal(int n, ns_obj f, void *ctx, const struct nsopt *o, int *best,
    struct nsres *r)
{
	struct nsrng g;
	int *cur = xcalloc((size_t)n, sizeof *cur);
	double e0, e, eb, ratio, T;
	long k, steps = o->budget - 1;

	ns_seed(&g, o->seed);
	e0 = e = eb = f(ctx, cur);
	r->evals = 1;
	r->before = e0;
	memset(best, 0, (size_t)n * sizeof *best);
	/* T_k = t0 (t1 / t0)^(k / steps), as a constant ratio per step */
	ratio = 1;
	if (steps > 1 && o->t0 > 0 && o->t1 > 0) {
		/* (t1/t0)^(1/steps) = exp(ln(t1/t0) / steps); ln by Newton on
		 * ns_exp, so that no libm function is used */
		double q = o->t1 / o->t0, l = 0;
		int it;
		for (it = 0; it < 60; it++)
			l -= 1 - q / ns_exp(l);
		ratio = ns_exp(l / (double)steps);
	}
	T = o->t0 * (e0 > 0 ? e0 : 1);
	for (k = 0; k < steps && isfinite(e0); k++, T *= ratio) {
		int i = (int)(ns_u64(&g) % (uint64_t)n), old = cur[i];
		int mag = 1 + (int)(ns_u64(&g) % 3), d;
		double en;
		/* one in four moves is a longer jump */
		if ((ns_u64(&g) & 3) == 0)
			mag *= 1 + (int)(ns_u64(&g) % 4);
		d = (ns_u64(&g) & 1) ? mag : -mag;
		cur[i] = old + d;
		if (cur[i] > o->radius || cur[i] < -o->radius) {
			cur[i] = old;
			continue;
		}
		en = f(ctx, cur);
		r->evals++;
		if (en <= e || (T > 0 && ns_unif(&g) < ns_exp(-(en - e) / T))) {
			e = en;
			r->accepted++;
			if (e < eb) {
				eb = e;
				memcpy(best, cur, (size_t)n * sizeof *best);
			}
		} else
			cur[i] = old;
	}
	r->after = eb;
	free(cur);
}

/* the eigenvectors (columns of B) and eigenvalues (d) of the symmetric
 * C, by cyclic Jacobi rotations */
static void
jacobi(int n, const double *C, double *B, double *d)
{
	double *a = xmalloc((size_t)n * n * sizeof *a);
	int i, j, k, sweep;

	memcpy(a, C, (size_t)n * n * sizeof *a);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			B[i * n + j] = i == j;
	for (sweep = 0; sweep < 30; sweep++) {
		double off = 0;
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				off += a[i * n + j] * a[i * n + j];
		if (off < 1e-30)
			break;
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++) {
				double apq = a[i * n + j], th, t, c, s;
				if (apq == 0)
					continue;
				th = (a[j * n + j] - a[i * n + i]) / (2 * apq);
				t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1));
				c = 1 / sqrt(t * t + 1);
				s = t * c;
				for (k = 0; k < n; k++) {
					double akp = a[k * n + i], akq = a[k * n + j];
					a[k * n + i] = c * akp - s * akq;
					a[k * n + j] = s * akp + c * akq;
				}
				for (k = 0; k < n; k++) {
					double apk = a[i * n + k], aqk = a[j * n + k];
					a[i * n + k] = c * apk - s * aqk;
					a[j * n + k] = s * apk + c * aqk;
				}
				for (k = 0; k < n; k++) {
					double bkp = B[k * n + i], bkq = B[k * n + j];
					B[k * n + i] = c * bkp - s * bkq;
					B[k * n + j] = s * bkp + c * bkq;
				}
			}
	}
	for (i = 0; i < n; i++)
		d[i] = a[i * n + i] > 1e-20 ? a[i * n + i] : 1e-20;
	free(a);
}

struct ranked {
	double f;
	int k;
};

static int
cmp_ranked(const void *a, const void *b)
{
	const struct ranked *x = a, *y = b;

	if (x->f != y->f)
		return x->f < y->f ? -1 : 1;
	return x->k - y->k;
}

static void
cmaes(int n, ns_obj f, void *ctx, const struct nsopt *o, int *best,
    struct nsres *r)
{
	struct nsrng g;
	int lam = o->pop > 3 ? o->pop : 8 + n, mu = lam / 2, i, j, k, gen;
	double *w = xcalloc((size_t)mu, sizeof *w);
	double *m = xcalloc((size_t)n, sizeof *m);
	double *C = xcalloc((size_t)n * n, sizeof *C);
	double *B = xcalloc((size_t)n * n, sizeof *B);
	double *D = xcalloc((size_t)n, sizeof *D);
	double *ps = xcalloc((size_t)n, sizeof *ps);
	double *pc = xcalloc((size_t)n, sizeof *pc);
	double *z = xcalloc((size_t)lam * n, sizeof *z);
	double *y = xcalloc((size_t)lam * n, sizeof *y);
	double *yw = xcalloc((size_t)n, sizeof *yw);
	double *tmp = xcalloc((size_t)n, sizeof *tmp);
	int *xi = xcalloc((size_t)n, sizeof *xi);
	struct ranked *rk = xcalloc((size_t)lam, sizeof *rk);
	double sw = 0, sw2 = 0, mueff, cc, cs, c1, cmu, damps, chiN;
	double sigma = o->sigma > 0 ? o->sigma : 2, eb, decay = 1;

	ns_seed(&g, o->seed);
	for (i = 0; i < mu; i++) {
		w[i] = mu - i;		/* linear weights */
		sw += w[i];
	}
	for (i = 0; i < mu; i++) {
		w[i] /= sw;
		sw2 += w[i] * w[i];
	}
	mueff = 1 / sw2;
	cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
	cs = (mueff + 2) / (n + mueff + 5);
	c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
	cmu = 2 * (mueff - 2 + 1 / mueff) / ((n + 2.0) * (n + 2.0) + mueff);
	if (cmu > 1 - c1)
		cmu = 1 - c1;
	damps = 1 + cs + 2 * (sqrt((mueff - 1) / (n + 1)) > 1 ?
	    sqrt((mueff - 1) / (n + 1)) - 1 : 0);
	chiN = sqrt((double)n) * (1 - 1 / (4.0 * n) + 1 / (21.0 * n * n));
	for (i = 0; i < n; i++)
		C[i * n + i] = 1;
	memset(best, 0, (size_t)n * sizeof *best);
	eb = f(ctx, best);
	r->evals = 1;
	r->before = eb;
	for (gen = 0; r->evals + lam <= o->budget && isfinite(eb); gen++) {
		double psn = 0, hs;
		jacobi(n, C, B, D);
		for (k = 0; k < lam; k++) {
			double e;
			for (i = 0; i < n; i++)
				z[k * n + i] = ns_normal(&g);
			for (i = 0; i < n; i++) {
				double s = 0;
				for (j = 0; j < n; j++)
					s += B[i * n + j] * sqrt(D[j]) * z[k * n + j];
				y[k * n + i] = s;
				s = m[i] + sigma * s;
				xi[i] = (int)(s < 0 ? s - 0.5 : s + 0.5);
				if (xi[i] > o->radius)
					xi[i] = o->radius;
				if (xi[i] < -o->radius)
					xi[i] = -o->radius;
			}
			e = f(ctx, xi);
			r->evals++;
			rk[k].f = e;
			rk[k].k = k;
			if (e < eb) {
				eb = e;
				memcpy(best, xi, (size_t)n * sizeof *best);
				r->accepted++;
			}
		}
		qsort(rk, (size_t)lam, sizeof *rk, cmp_ranked);
		for (i = 0; i < n; i++) {
			yw[i] = 0;
			for (k = 0; k < mu; k++)
				yw[i] += w[k] * y[rk[k].k * n + i];
			m[i] += sigma * yw[i];
		}
		/* ps: C^(-1/2) yw = B D^(-1/2) B^T yw */
		for (j = 0; j < n; j++) {
			double s = 0;
			for (i = 0; i < n; i++)
				s += B[i * n + j] * yw[i];
			tmp[j] = s / sqrt(D[j]);
		}
		for (i = 0; i < n; i++) {
			double s = 0;
			for (j = 0; j < n; j++)
				s += B[i * n + j] * tmp[j];
			ps[i] = (1 - cs) * ps[i] + sqrt(cs * (2 - cs) * mueff) * s;
			psn += ps[i] * ps[i];
		}
		psn = sqrt(psn);
		decay *= (1 - cs) * (1 - cs);
		hs = psn / sqrt(1 - decay) / chiN < 1.4 + 2 / (n + 1.0);
		for (i = 0; i < n; i++)
			pc[i] = (1 - cc) * pc[i] + hs * sqrt(cc * (2 - cc) * mueff) *
			    yw[i];
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++) {
				double rmu = 0;
				for (k = 0; k < mu; k++)
					rmu += w[k] * y[rk[k].k * n + i] * y[rk[k].k * n + j];
				C[i * n + j] = (1 - c1 - cmu) * C[i * n + j] + c1 *
				    (pc[i] * pc[j] + (1 - hs) * cc * (2 - cc) *
				    C[i * n + j]) + cmu * rmu;
			}
		sigma *= ns_exp(cs / damps * (psn / chiN - 1));
		if (sigma > o->radius)
			sigma = o->radius;
		if (sigma < 0.05)
			break;		/* the steps round to nothing */
	}
	r->gens = gen;
	r->after = eb;
	free(w);
	free(m);
	free(C);
	free(B);
	free(D);
	free(ps);
	free(pc);
	free(z);
	free(y);
	free(yw);
	free(tmp);
	free(xi);
	free(rk);
}

void
ns_run(int n, ns_obj f, void *ctx, const struct nsopt *o, int *best,
    struct nsres *r)
{
	memset(r, 0, sizeof *r);
	if (n < 1) {
		r->before = r->after = f(ctx, best);
		r->evals = 1;
		return;
	}
	if (o->kind == NS_CMAES)
		cmaes(n, f, ctx, o, best, r);
	else
		anneal(n, f, ctx, o, best, r);
}
