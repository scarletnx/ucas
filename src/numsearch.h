/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Numerical search, probabilistic error and quasi-Monte Carlo.
 *
 * numsearch.c  simulated annealing and CMA-ES over integer offsets (a
 *              rounded coefficient's ulps), for approx's `search`
 * rounderr.c   first-order rounding error of a float expression: the
 *              worst case beside the high-probability and expected ones
 * qmc.c        Sobol points (Joe-Kuo direction numbers), randomised
 *              replicates; integrate ... numerically qmc
 *
 * Everything random is seeded (`seed K`, else UCAS_SEED, else 1) and
 * uses + - * / and sqrt only, so a run is the same on every libm.
 */
#ifndef NUMSEARCH_H
#define NUMSEARCH_H

#include "ucas.h"

/* ---- numsearch.c ---- */
enum { NS_NONE, NS_ANNEAL, NS_CMAES };
struct nsopt {
	int kind;
	long budget;		/* objective evaluations (2000) */
	double t0, t1;		/* anneal: the temperature's ends, as fractions
				 * of the starting objective (0.1 to 0.001) */
	int pop;		/* cmaes: the population (8 + n) */
	double sigma;		/* cmaes: the first step, in ulps (2) */
	int radius;		/* offsets stay within +-radius ulps (64) */
	uint64_t seed;
};
struct nsres {
	long evals, accepted;
	int gens;
	double before, after;	/* the objective at offset 0, at the best */
};
typedef double (*ns_obj)(void *ctx, const int *off);
void ns_defaults(struct nsopt *o);
/* minimise f over off[0..n-1]; best[] gets the best offsets found */
void ns_run(int n, ns_obj f, void *ctx, const struct nsopt *o, int *best,
    struct nsres *r);
const char *ns_name(int kind);

/* a seeded generator of its own (xorshift64*) */
struct nsrng {
	uint64_t s;
};
void ns_seed(struct nsrng *g, uint64_t seed);
uint64_t ns_u64(struct nsrng *g);
double ns_unif(struct nsrng *g);	/* [0, 1) */
double ns_exp(double x);		/* e^x from + * / */

/* ---- rounderr.c ---- */
struct rounderr {
	int ok;			/* analysed */
	int nops;		/* rounded operations */
	int w;			/* the float width */
	uint64_t points;
	double worst;		/* max over the points of u sum |c_i| / |g| */
	double prob;		/* ... of lambda u sqrt(sum c_i^2) / |g| */
	double rms;		/* ... of u sqrt(sum c_i^2 / 3) / |g| */
	double lambda, conf;	/* conf = 1 - 2 exp(-lambda^2 / 2) */
	char why[120];		/* not analysed: the operation */
};
/* g: a float function; lambda <= 0: 4 */
void rounderr_measure(const struct func *g, double lambda,
    struct rounderr *r);
/* the lines `error` and approx print, each starting with ind */
void rounderr_print(const struct rounderr *r, const char *ind, int rep);
/* the same in one line, for approx's table; name: the candidate's */
void rounderr_line(const struct rounderr *r, const char *ind,
    const char *name);

/* ---- qmc.c ---- */
struct P;
struct qmcopt {
	long points;		/* per replicate, a power of two (4096) */
	int reps;		/* randomised replicates (8); 0: unshifted */
	uint64_t seed;
	int fallback;		/* not asked for by name: one line when the
				 * replicates do not agree */
	int owen;		/* nested uniform scrambling, not a shift */
};
void qmc_defaults(struct qmcopt *o);
/* the options after `numerically`: qmc [points N] [replicates R] [seed K]
 * [noscramble] [owen]; 0 when the next word is none of them */
int qmc_options(struct P *p, struct qmcopt *o);
/* the integral of e over slot in [a, b] when slot >= 0, else over the
 * box of f's parameters' ranges; prints the estimate and its error */
void qmc_integrate(const struct func *f, nid e, int slot, double a,
    double b, const struct qmcopt *o, const char *ind);
/* point k (from 0) of the Sobol sequence in dim dimensions, as 32-bit
 * fractions; dim <= QMC_MAXDIM */
#define QMC_MAXDIM 8
void sobol_point(uint32_t k, int dim, uint32_t *x);

#endif
