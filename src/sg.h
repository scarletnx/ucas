/* SPDX-License-Identifier: GPL-3.0-or-later */
/* sg.c's helpers that inst.c's find section uses too */
#ifndef UCAS_SG_H
#define UCAS_SG_H

#include "ucas.h"

/* root with from replaced by to; 0 when the result cannot be built */
nid sg_replaced(nid root, nid from, nid to);
/*
 * The proved range of e (the parameters' ranges narrowed by the where
 * clause, then intervals); a bound lane in it runs over 0..W-1.
 */
int sg_range_of(const struct func *f, nid e, int W, int64_t *lo, int64_t *hi);

/*
 * A GF(2)-affine map of one word: bit i of the value is the parity of
 * x & row[i], xor bit i of k; w its width.  A bit of bad is one that is
 * not affine (two bits and-ed, a carry): a mask or a shift may drop it.
 */
struct aff {
	uint64_t row[64];
	uint64_t k, bad;
	int w;
};
/* e as an affine map of parameter slot's bits, by its structure; 0 when
 * some operation is not linear */
int gf2_affine(nid e, int slot, struct aff *r, int depth);

#endif
