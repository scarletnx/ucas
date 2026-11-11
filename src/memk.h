/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * memk.c: the memory side of a cost model, and the rewrites it makes
 * worth doing.
 *
 *   banks F [, G ...] [size N] [elem B] [model M]
 *       the bank-conflict degree of shared-memory accesses over a warp,
 *       and an index swizzle that lowers it, proved a bijection
 *   intatomic F [count N] [tol T] [global] [model M]
 *       a float atomic add as an integer atomic add on fixed point
 *
 * The memory kinds are cost classes like the others; a model that lists
 * none of them prices each as its fallback (load, store, atomic_shared,
 * atomic_global), and one without `banks` has no bank conflicts.
 */
#ifndef MEMK_H
#define MEMK_H

#include "ucas.h"

struct P;
int memk_command(struct P *p);			/* parse.c */
const char *memk_nlp_raw(const char *sent);	/* nlp.c */

/* a class's price under m: its own, or its fallback's when m lists none */
float model_cost(const struct model *m, int cls);
int model_lists(const struct model *m, int cls);

/* the conflict degree of index e over a warp of f's lane parameter (the
 * first integer parameter that is not uniform): the largest number of
 * distinct addresses on one bank; *exact: every case was enumerated or
 * the index is affine in the lane; *affine: the stride when it is */
struct bankres {
	int degree;
	int exact, affine;
	int64_t stride;
	uint64_t cases;
};
int bank_degree(const struct func *f, nid e, const struct model *m,
    int elem, struct bankres *r);
/* the price of one access with that degree: load or store */
double bank_price(const struct model *m, int store, int degree);

/* banks' report for the accesses named (functions); 1 and the swizzle
 * when one is cheaper and proved: i xor (i >> s & banks - 1), the row
 * added into the bank bits, or i + (i >> s) with the array grown to size */
enum { SW_XOR, SW_ADD, SW_PAD };
struct bankswz {
	int kind, s;
	int64_t size;
};
int memk_banks(const uint32_t *names, int nf, int64_t size, int elem,
    const struct model *m, struct bankswz *out);
/* intatomic's report for f (a float atomic add in it, or the addend) */
void memk_intatomic(struct func *f, long count, double tol, int global,
    const struct model *m);

/* the fixed-point grid of a float expression: its values are k 2^g for
 * integers |k| <= *kmax, and its f32 evaluation is exact; 0: none */
int fix_grid(const struct func *f, nid e, int *g, double *kmax,
    struct sbuf *why);

#endif
