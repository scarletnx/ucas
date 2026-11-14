/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * qmc: quasi-Monte Carlo integration on Sobol points.
 *
 * The points are the Sobol sequence in Gray-code order, with the
 * direction numbers of Joe and Kuo ("Constructing Sobol sequences with
 * better two-dimensional projections", SIAM J. Sci. Comput. 30, 2008;
 * the first eight dimensions of new-joe-kuo-6.21201).  One sequence
 * gives no error estimate, so by default R replicates are run, each the
 * same points under a random digital shift (every coordinate xored with
 * a random word: the simplest scrambling, and unbiased); the estimate is
 * the mean of the R integrals and its error the standard error of that
 * mean, which is an estimate from R numbers and not a bound.
 *
 * `owen` scrambles each replicate's points by a nested uniform (Owen)
 * scrambling instead: every bit of a coordinate flipped by a random
 * function of the bits above it, so each elementary interval's points
 * are permuted within it.  Its variance falls faster than a shift's for
 * smooth integrands (n^-3 against n^-2 for the mean square, Owen 1997),
 * and the replicates' spread estimates it from the same R.  The bits
 * come from Burley's hash ("Practical Hash-based Owen Scrambling", JCGT
 * 9(4), 2020): Laine and Karras's permutation with his constants on the
 * bit-reversed coordinate, a different seed per replicate and dimension.
 *
 * A coordinate is taken at the centre of its 2^-32 cell, so the ends of
 * the box are never evaluated.  An infinite side is a change of variable
 * onto [0, 1): x = a + t/(1 - t) for [a, inf), x = b - t/(1 - t) for
 * (-inf, b], x = s/(1 - s^2) with s = 2t - 1 for the whole line, the
 * integrand times the derivative; a parameter with no range runs over
 * the whole line.  Points where the integrand is undefined
 * are left out and counted.
 */
#include "numsearch.h"
#include "stoch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* a dimension for each parameter a function can have: a larger MAXP
 * needs more rows below */
typedef char qmc_dims_are_maxp[QMC_MAXDIM == MAXP ? 1 : -1];

/* dimension 2..8: degree s, coefficients a, initial m_1..m_s */
static const struct {
	int s, a, m[5];
} jk[QMC_MAXDIM - 1] = {
	{1, 0, {1}}, {2, 1, {1, 3}}, {3, 1, {1, 3, 1}}, {3, 2, {1, 1, 1}},
	{4, 1, {1, 1, 3, 3}}, {4, 4, {1, 3, 5, 13}}, {5, 2, {1, 1, 5, 5, 17}},
};

static uint32_t V[QMC_MAXDIM][32];
static int vready;

static void
sobol_init(void)
{
	int d, i, k;

	if (vready)
		return;
	for (i = 0; i < 32; i++)
		V[0][i] = (uint32_t)1 << (31 - i);
	for (d = 1; d < QMC_MAXDIM; d++) {
		int s = jk[d - 1].s, a = jk[d - 1].a;
		for (i = 0; i < s; i++)
			V[d][i] = (uint32_t)jk[d - 1].m[i] << (31 - i);
		for (i = s; i < 32; i++) {
			V[d][i] = V[d][i - s] ^ (V[d][i - s] >> s);
			for (k = 1; k < s; k++)
				if ((a >> (s - 1 - k)) & 1)
					V[d][i] ^= V[d][i - k];
		}
	}
	vready = 1;
}

void
sobol_point(uint32_t k, int dim, uint32_t *x)
{
	uint32_t g = k ^ (k >> 1);
	int d, i;

	sobol_init();
	for (d = 0; d < dim; d++) {
		uint32_t v = 0;
		for (i = 0; i < 32; i++)
			if (g >> i & 1)
				v ^= V[d][i];
		x[d] = v;
	}
}

static uint32_t
rev32(uint32_t x)
{
	x = (x >> 1 & 0x55555555u) | (x & 0x55555555u) << 1;
	x = (x >> 2 & 0x33333333u) | (x & 0x33333333u) << 2;
	x = (x >> 4 & 0x0f0f0f0fu) | (x & 0x0f0f0f0fu) << 4;
	x = (x >> 8 & 0x00ff00ffu) | (x & 0x00ff00ffu) << 8;
	return x >> 16 | x << 16;
}

/*
 * Owen's nested uniform scrambling, hash-based: reversed, a product by an
 * even constant only carries into higher bits, so in the coordinate's
 * own order each bit is flipped by a function of the bits above it
 */
static uint32_t
owen32(uint32_t x, uint32_t seed)
{
	x = rev32(x);
	x += seed;
	x ^= x * 0x6c50b47cu;
	x ^= x * 0xb82f1e52u;
	x ^= x * 0xc7afe638u;
	x ^= x * 0x8d22f6e6u;
	return rev32(x);
}

void
qmc_defaults(struct qmcopt *o)
{
	o->points = 4096;
	o->reps = 8;
	o->seed = stoch_seed();
	o->fallback = 0;
	o->owen = 0;
}

int
qmc_options(struct P *p, struct qmcopt *o)
{
	int64_t v;

	if (cmdp_kw(p, "points")) {
		if (!cmdp_int(p, &v) || v < 2 || v > (1 << 22) || (v & (v - 1)))
			fail("qmc: points N, a power of two from 2 to 4194304");
		o->points = (long)v;
		return 1;
	}
	if (cmdp_kw(p, "replicates")) {
		if (!cmdp_int(p, &v) || v < 2 || v > 64)
			fail("qmc: replicates R, from 2 to 64 (noscramble for one "
			    "unshifted sequence)");
		o->reps = (int)v;
		return 1;
	}
	if (cmdp_kw(p, "seed")) {
		if (!cmdp_int(p, &v))
			fail("qmc: seed K");
		o->seed = (uint64_t)v;
		return 1;
	}
	if (cmdp_kw(p, "noscramble")) {
		o->reps = 0;
		return 1;
	}
	if (cmdp_kw(p, "owen")) {
		o->owen = 1;
		return 1;
	}
	return 0;
}

/* the changes of variable an infinite side took */
static void
inf_line(const int *side, int dim, const char *ind)
{
	static const char *const how[] = {"", "x = a + t/(1 - t) for [a, inf)",
		"x = b - t/(1 - t) for (-inf, b]", "x = s/(1 - s^2), s = 2t - 1, "
		"for the whole line"};
	int k, j, n = 0;

	printf("%s  infinite sides onto t in [0, 1), the integrand times the "
	    "derivative:", ind);
	for (k = 1; k <= 3; k++)
		for (j = 0; j < dim; j++)
			if (side[j] == k) {
				printf("%s %s", n++ ? ";" : "", how[k]);
				break;
			}
	printf("\n");
}

/* Student's t, two-sided 95%, for 1..30 degrees of freedom */
static double
t95(int df)
{
	static const double t[] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447,
		2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
		2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
		2.060, 2.056, 2.052, 2.048, 2.045, 2.042};

	return df < 1 ? 0 : df <= 30 ? t[df - 1] : 1.96;
}

void
qmc_integrate(const struct func *f, nid e, int slot, double a, double b,
    const struct qmcopt *o, const char *ind)
{
	struct plan p;
	struct nsrng g;
	nid roots[1];
	int64_t *in_i;
	double *in_r, lo[MAXP], hi[MAXP], vol = 1, mean = 0, se = 0;
	double rep[64];
	int dim = 0, map[MAXP], j, r, nrep = o->reps > 0 ? o->reps : 1;
	int side[MAXP], anyinf = 0;	/* 0 finite, 1 to inf, 2 from -inf, 3 both */
	double jac[CH];
	uint32_t loc, x[QMC_MAXDIM], shift[QMC_MAXDIM];
	long k, undef = 0, total = 0;
	double amax = 0, asum = 0;
	int sgn = 1;

	if (slot >= 0) {
		if (f->np != 1)
			fail("qmc: an integral over one variable needs the other "
			    "parameters fixed (without from..to it runs over the box "
			    "of every parameter's range)");
		if (isnan(a) || isnan(b))
			fail("qmc: the bounds %g..%g are not numbers", a, b);
		map[dim] = slot;
		lo[dim] = a < b ? a : b;
		hi[dim] = a < b ? b : a;
		sgn = a <= b ? 1 : -1;
		dim++;
	} else {
		for (j = 0; j < f->np; j++) {
			const struct param *pp = &f->p[j];
			if (!TY_FP(pp->ty))
				fail("qmc: parameter %s is %s; the box is over real "
				    "(or float) parameters", sym_str(pp->name),
				    ty_str(pp->ty));
			map[dim] = j;
			lo[dim] = pp->ranged ? pp->rlo : -INFINITY;
			hi[dim] = pp->ranged ? pp->rhi : INFINITY;
			if (isnan(lo[dim]) || isnan(hi[dim]))
				fail("qmc: parameter %s's range %g..%g is not numbers",
				    sym_str(pp->name), lo[dim], hi[dim]);
			dim++;
		}
	}
	if (o->owen && o->reps <= 0)
		fail("qmc: owen scrambles each replicate and noscramble runs one "
		    "unscrambled sequence: not both");
	if (dim < 1 || dim > QMC_MAXDIM)
		fail("qmc: 1 to %d dimensions, not %d", QMC_MAXDIM, dim);
	for (j = 0; j < dim; j++) {
		side[j] = (isfinite(hi[j]) ? 0 : 1) | (isfinite(lo[j]) ? 0 : 2);
		anyinf |= side[j] != 0;
		if (!side[j])
			vol *= hi[j] - lo[j];
	}
	roots[0] = mk_cast(TY_REAL, e);
	plan_build(&p, roots, 1, 0);
	loc = plan_loc(&p, roots[0]);
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	ns_seed(&g, o->seed);
	for (r = 0; r < nrep; r++) {
		double sum = 0;
		long cnt = 0;
		for (j = 0; j < dim; j++)
			shift[j] = o->reps > 0 ? (uint32_t)(ns_u64(&g) >> 32) : 0;
		for (k = 0; k < o->points;) {
			uint32_t np = 0, q;
			for (; np < CH && k < o->points; np++, k++) {
				sobol_point((uint32_t)k, dim, x);
				jac[np] = 1;
				for (j = 0; j < dim; j++) {
					uint32_t y = o->reps <= 0 ? x[j] : o->owen ?
					    owen32(x[j], shift[j]) : x[j] ^ shift[j];
					double t = ((double)y + 0.5) / 4294967296.0, v, s;
					switch (side[j]) {
					case 0: v = lo[j] + (hi[j] - lo[j]) * t; break;
					case 1:
						v = lo[j] + t / (1 - t);
						jac[np] /= (1 - t) * (1 - t);
						break;
					case 2:
						v = hi[j] - t / (1 - t);
						jac[np] /= (1 - t) * (1 - t);
						break;
					default:
						s = 2 * t - 1;
						v = s / (1 - s * s);
						jac[np] *= 2 * (1 + s * s) / ((1 - s * s) * (1 - s *
						    s));
					}
					in_r[(size_t)map[j] * CH + np] = v;
				}
			}
			plan_run(&p, np, in_i, in_r);
			for (q = 0; q < np; q++) {
				double v = COLR(&p, loc)[q] * jac[q];
				total++;
				if ((COLF(&p, loc)[q] & F_UNDEF) || !isfinite(v)) {
					undef++;
					continue;
				}
				sum += v;
				cnt++;
				asum += fabs(v);
				if (fabs(v) > amax)
					amax = fabs(v);
			}
		}
		rep[r] = cnt ? sgn * vol * sum / (double)cnt : 0;
		mean += rep[r];
	}
	mean /= nrep;
	if (nrep > 1) {
		for (r = 0; r < nrep; r++)
			se += (rep[r] - mean) * (rep[r] - mean);
		se = sqrt(se / ((double)nrep * (nrep - 1)));
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	rep_obj("qmc");
	rep_int("dimensions", dim);
	rep_int("points", o->points);
	rep_int("replicates", o->reps);
	rep_int("seed", (long long)o->seed);
	rep_str("scrambling", o->reps <= 0 ? "none" : o->owen ? "owen" :
	    "digital shift");
	rep_bool("infinite_box", anyinf);
	rep_num("estimate", mean);
	if (undef == total) {
		printf("%squasi-Monte Carlo: the integrand is undefined at all %ld "
		    "points: no estimate\n", ind, total);
		rep_null("value");
		rep_end();
		return;
	}
	if (o->fallback && o->reps > 0 && !(se <= 0.05 * fabs(mean))) {
		/* nobody asked for it and it settles nothing: say only that */
		printf("%squasi-Monte Carlo does not settle either: %d replicates "
		    "of %ld Sobol points give %.3g with a standard error of %.2g: "
		    "no value (a pole, or an integral that does not exist)\n", ind,
		    o->reps, o->points, mean, se);
		rep_bool("converged", 0);
		rep_null("value");
		rep_end();
		return;
	}
	if (o->reps > 0) {
		printf("%squasi-Monte Carlo (an estimate, not a proof): %.10g, "
		    "standard error %.2g\n", ind, mean, se);
		printf("%s  %d randomised replicates of %ld Sobol points in %d "
		    "dimension%s (Joe-Kuo direction numbers, %s each%s; seed "
		    "%llu)\n", ind, o->reps, o->points, dim, dim == 1 ? "" : "s",
		    o->owen ? "a nested uniform (Owen) scrambling" :
		    "a random digital shift", o->owen ? ", from a hash" : "",
		    (unsigned long long)o->seed);
		if (anyinf)
			inf_line(side, dim, ind);
		printf("%s  the error is estimated from the %d replicates' spread: "
		    "+-%.2g at 95%% if their means are normal (t = %.3f), which "
		    "nothing proves\n", ind, o->reps, t95(o->reps - 1) * se,
		    t95(o->reps - 1));
		rep_num("standard_error", se);
		rep_num("interval95", t95(o->reps - 1) * se);
	} else {
		printf("%squasi-Monte Carlo (an estimate, not a proof): %.10g\n",
		    ind, mean);
		printf("%s  %ld Sobol points in %d dimension%s (Joe-Kuo direction "
		    "numbers), not randomised: no error estimate (drop noscramble "
		    "for one)\n", ind, o->points, dim, dim == 1 ? "" : "s");
		if (anyinf)
			inf_line(side, dim, ind);
	}
	if (total > undef && asum > 0 && amax > 50 * asum / (double)(total -
	    undef))
		printf("%s  a spike: the largest value met is %.0f times the mean "
		    "size, so a few points carry the integral; the replicates "
		    "then agree on a value that is off (usually too small) and "
		    "their spread understates the error\n", ind, amax / (asum /
		    (double)(total - undef)));
	rep_bool("spike", total > undef && asum > 0 && amax > 50 * asum /
	    (double)(total - undef));
	if (o->reps > 0 && se > 0.05 * fabs(mean))
		printf("%s  not converged: the replicates differ by %.0f%% of the "
		    "value (standard error); more points, or the integral does "
		    "not exist (a pole between the points looks like this)\n", ind,
		    100 * se / (fabs(mean) > 0 ? fabs(mean) : 1));
	rep_bool("converged", !(o->reps > 0 && se > 0.05 * fabs(mean)));
	if (undef)
		printf("%s  the integrand is undefined at %ld of the %ld points, "
		    "which are left out: if it is not integrable there (a pole), "
		    "this number means nothing\n", ind, undef, total);
	rep_int("undefined_points", undef);
	rep_str("label", "estimate");
	rep_end();
}
