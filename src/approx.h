/*
 * approx: polynomial and rational approximation (approx.c, remez.c).
 * Internal interface between the command and the numerics.
 */
#ifndef APPROX_H
#define APPROX_H

#include "ucas.h"

typedef long double ld;

#define AMAXC 40	/* coefficients per polynomial */
#define AMAXK 16	/* exact (keep) points */
#define AMAXJ 48	/* Taylor terms */

typedef ld (*ldfn)(void *ctx, ld x);

/*
 * Reference evaluator: a function body evaluated in long double, one point
 * at a time (machine types are read as their ideal real twins).  Bodies
 * with integer or bit-level operations fall back to the double-precision
 * ideal plan.
 */
struct lref {
	uint32_t n, root;
	uint8_t *op;
	uint32_t *a, *b, *c;
	ld *k;
	ld *v;
	int slot;
	int full;	/* every node has a long double rule */
	struct plan dp;	/* fallback */
	int64_t *in_i;
	double *in_r;
};
void lref_init(struct lref *r, nid body, int slot);
void lref_free(struct lref *r);
ld lref_eval(struct lref *r, ld x);	/* NaN where undefined */
ld lref_fn(void *ctx, ld x);
int lref_taylor(struct lref *r, ld c, int n, ld *a);

/*
 * A fit: f(x) ~ t^k0 * N(v) / D(v), t = x - c, v = t^step, on [xa, xb]
 * (step 2 needs c = 0 <= xa).  N has nt terms, D has nd terms after its
 * constant 1.  keep[] are points where the fit must be exact.
 */
enum { FM_CHEB, FM_ECON, FM_REMEZ, FM_PADE };
struct fitprob {
	ldfn f;
	void *fctx;
	ldfn w;		/* extra weight, or NULL */
	void *wctx;
	ld xa, xb, c;
	int k0, step, rel;
	int nkeep;
	ld keep[AMAXK];
	int nt, nd;
	int method;
	const ld *taylor;	/* f's Taylor coefficients in t (ECON, PADE) */
	int ntaylor;
};
struct fitres {
	int ok;
	int nn, nd;		/* num terms; den terms including den[0] = 1 */
	ld num[AMAXC];		/* coefficient of t^(k0 + step j) */
	ld den[AMAXC];		/* coefficient of t^(step j) */
	ld err;			/* max weighted error (real arithmetic) */
	ld eabs, erel;
	ld xworst;
	int iters;
	const char *why;
};
int fit_run(const struct fitprob *pb, struct fitres *r);
ld fit_eval(const struct fitprob *pb, const struct fitres *r, ld x);
void fit_measure(const struct fitprob *pb, struct fitres *r);

#endif
