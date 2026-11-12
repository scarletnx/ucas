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
int model_class(const struct model *m, int cls);
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
 * added into the bank bits, or i + (i >> s) with the array grown to size
 * (a padded row of row elements, when row > 0: a nested array's own);
 * extra, what writing any swizzle adds to each access (a nested array's
 * indices laid flat and taken apart again) */
enum { SW_XOR, SW_ADD, SW_PAD };
struct bankswz {
	int kind, s;
	int64_t size;
};
int memk_banks(const uint32_t *names, int nf, int64_t size, int elem,
    int64_t row, double extra, const struct model *m, struct bankswz *out);
/* intatomic's report for f (a float atomic add in it, or the addend),
 * count the adds into one cell (0: not known, and the most that fit is
 * said instead); *out, when given, the decision */
struct intatom {
	int ok;			/* cheaper, and exact (or within tol) */
	int exact;		/* on the grid: no rounding at all */
	int g;			/* the cell holds the sum times 2^-g */
	double kmax;		/* each addend is k 2^g with |k| <= kmax */
	double fits;		/* the most adds whose sum fits 32 bits */
	double cost_f, cost_i;	/* addend and atomic: float, integer */
	nid form;		/* the integer addend, over f's parameters */
};
void memk_intatomic(struct func *f, long count, double tol, int global,
    const struct model *m, struct intatom *out);
/* v 2^-g as an i32 expression: exact when fix_grid puts v on the grid
 * 2^g with |v| 2^-g below 2^31 (its bound holds every step on the way;
 * the conversions dropped where v's structure allows, else v scaled by
 * a power of two and converted) */
nid memk_fixform(nid v, int g);

/* the fixed-point grid of a float expression: its values are k 2^g for
 * integers |k| <= *kmax, and its f32 evaluation is exact; 0: none */
int fix_grid(const struct func *f, nid e, int *g, double *kmax,
    struct sbuf *why);

#endif
