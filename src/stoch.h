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
 * M, A and s with ((k[j] M + A) >> s) & (2^b - 1) = t[j] for every j, in
 * u32 (s + b <= 32): every multiplier for the small shifts, then maxtries
 * drawn from seed for the others, A solved for each.  1 when found; *exh
 * the largest shift every multiplier was tried at (-1: none), *tries the
 * multipliers drawn.  ms_none says what a failed search covered.
 */
int ms_solve(const uint32_t *k, const uint32_t *t, int n, int b,
    uint64_t seed, long maxtries, uint32_t *M, uint32_t *A, int *s,
    long *tries, int *exh);
void ms_none(char *txt, size_t n, int b, int exh, long tries);

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
 * a and b (a sized integer type) as polynomials mod 2^w in the
 * parameters and in subterms taken whole: 1 when their difference is 0
 * at every point, so b equals a on every input (how: the proof's line);
 * 0 when that is not shown.  *deg: the higher total degree of the two.
 */
int poly_identity(nid a, nid b, int *deg, struct sbuf *how);

/*
 * How a ?: on cond behaves across a warp of `warp` lanes: *p the
 * probability that it holds for one lane (uniform over f's domain),
 * *all_t, *all_f that the whole warp agrees, *split = 1 - *all_t -
 * *all_f.  The warps are the aligned runs of `warp` consecutive values
 * of f's one varying integer parameter when it has one (returns 2:
 * counted, *exact when every case was), independent lanes otherwise
 * (returns 1: split = 1 - p^warp - (1 - p)^warp).  0: not measured.
 */
int diverge_stats(const struct func *f, nid cond, nid reach, uint32_t loop,
    int warp, double *p, double *all_t, double *all_f, double *reached,
    int *exact, uint64_t *points);
#define LOOP_NONE 0xffffffffu

/* loop.c: a ?: in loop e's body, counted at each iteration over the lanes
 * that run it, in windows of warp; reading the counts turns it off */
void loop_probe(uint32_t e, nid cond, int warp);
void loop_probe_counts(uint64_t *nw, uint64_t *nat, uint64_t *naf,
    uint64_t *n, uint64_t *nt);

/*
 * A loop across a warp of `warp` lanes, for cost: the warp runs until its
 * last lane leaves, so it takes the most of its lanes' trip counts
 * (*emax, on average), and its lanes leave together only when their
 * counts are equal (*split: the probability that they do not).  The
 * count is `trips`, an expression of f's parameters, or, when trips is
 * 0, hi - lo + 1 (none when negative).  The warps are the aligned runs of
 * the index named consecutive (returns 2: counted, *exact when every
 * case was), independent draws from the domain otherwise (1: from the
 * counts at its points, every one when *exact).  0: not measured.
 */
int warp_trips(const struct func *f, nid trips, nid lo, nid hi, int warp,
    double *emax, double *split, int *exact, uint64_t *points);

/* cost F consecutive NAME: the parameter slot whose values run along the
 * lanes of a warp (-1: none, the lanes are independent) */
void diverge_consecutive(int slot);

uint64_t stoch_seed(void);	/* UCAS_SEED, or 1 */

#endif
