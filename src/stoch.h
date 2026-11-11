/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * stoch.c: stochastic search and probabilistic answers.
 *
 *   stoke F [steps N] [restarts R] [len L] [seed K] [model M]
 *       MCMC (Metropolis-Hastings) over straight-line programs, priced
 *       by the model; what reaches distance 0 is verified
 *   pit F G [trials N] [seed K]
 *       polynomial identity testing (Schwartz-Zippel), with its bound
 *   phash F [seed K] [model M]
 *       a switch's keys hashed into a table when the model prices the
 *       lookup below the compare chain
 *
 * Every search is seeded: `seed K`, else UCAS_SEED, else 1.
 */
#ifndef STOCH_H
#define STOCH_H

#include "ucas.h"

struct P;
int stoch_command(struct P *p);			/* parse.c */
const char *stoch_nlp_raw(const char *sent);	/* nlp.c */

/* find F stoke: the search with its defaults */
void stoke_find(struct func *f);

/*
 * equiv's last resort: a and b as polynomials over the integers in f's
 * parameters, compared at random points mod 2^61 - 1.  -1: not both
 * polynomials; 0: they differ over Z (not a verdict on wrapping types);
 * 1: equal over Z except with probability at most *bound, hence equal as
 * wrapping functions.  text: the line to print.
 */
int pit_check(const struct func *f, nid a, nid b, int trials, uint64_t seed,
    double *bound, struct sbuf *text);

/*
 * How a ?: on cond behaves across a warp of `warp` lanes: *p the
 * probability that it holds for one lane (uniform over f's domain),
 * *all_t, *all_f that the whole warp agrees, *split = 1 - *all_t -
 * *all_f.  The warps are the aligned runs of `warp` consecutive values
 * of f's one varying integer parameter when it has one (returns 2:
 * counted, *exact when every case was), independent lanes otherwise
 * (returns 1: split = 1 - p^warp - (1 - p)^warp).  0: not measured.
 */
int diverge_stats(const struct func *f, nid cond, int warp, double *p,
    double *all_t, double *all_f, int *exact, uint64_t *points);

/* cost F consecutive NAME: the parameter slot whose values run along the
 * lanes of a warp (-1: none, the lanes are independent) */
void diverge_consecutive(int slot);

uint64_t stoch_seed(void);	/* UCAS_SEED, or 1 */

#endif
