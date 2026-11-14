/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * nodesat.c: SAT proofs about expression DAGs on machine integers.
 */
#ifndef UCAS_NODESAT_H
#define UCAS_NODESAT_H

#include "ucas.h"

/* struct nsat_res is in ucas.h (equiv_decide's result carries one) */

/* 1 when every parameter is a machine integer (or int, bool) */
int nsat_supported(const struct func *f);
/*
 * b refines a on f's domain (ranges, where clause, a defined): b is
 * defined and equal there.  The conflict budget bounds the solver.
 */
int nsat_equiv(const struct func *f, nid a, nid b, long budget,
    struct nsat_res *out);

extern int nsat_narrow;	/* parameters' bits beyond their ranges are fixed */
extern int nsat_ftz;	/* float operations flush subnormals (equiv ftz) */
extern int nsat_floats;	/* floats are blasted (equiv's own question) */
extern int nsat_nanbits;	/* NaN results equal only bit for bit (equiv nanbits) */
int nsat_has_float(const struct func *f, nid e);	/* or f's parameters */
/* a counterexample's words of the `lanes` parameters, by slot and lane,
 * and which of them the circuit read */
extern int64_t nsat_lanes[MAXP][SG_MAXW];
extern uint8_t nsat_lanes_set[MAXP][SG_MAXW];
#define NSAT_BUDGET 100000   /* conflicts */
/* and steps, the watched clauses the solver visits, counted so that an answer
 * is the same on every machine and under any load: about 1 s of CPU where a
 * second visits 100 million (glibc, gcc 16, x86-64), 2 s on a machine half
 * as fast, where a give-up must not eat the C frontend's own CPU budget; the
 * largest a test's decided call takes is 94 million */
#define NSAT_WORK 100000000L

#endif
