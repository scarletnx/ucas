/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * bound.c: bounds of a function proved by monotonicity, the where
 * clause's relations and exact integer-quotient algebra.
 */
#ifndef UCAS_BOUND_H
#define UCAS_BOUND_H

#include "ucas.h"

struct pbound {
	int have_lo, have_hi;   /* 1 exact (a constant), 2 an interval at the end */
	double lo, hi;
	struct sbuf why_lo, why_hi;   /* the steps, one per line */
};

int prove_bounds(const struct func *f, struct pbound *pb);
void pbound_free(struct pbound *pb);

#endif
