/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecan.h - the analytic side of elliptic curves over Q (ecan.c).
 * Floating point, with the error bounds stated next to each value.
 */
#ifndef ECAN_H
#define ECAN_H

#include "ecq.h"
#include <complex.h>

struct ecan_per {
	double complex w1, w2;   /* w1 real; Im(w2 / w1) > 0 */
	double omega;            /* the real period times the components */
	double err;
	int ncomp;               /* components of E(R) */
};

struct ecan_L {
	int w;                   /* the root number */
	double wfit, wtest;      /* the functional equation's mismatch for
				  * w and for -w */
	double L[6];             /* L^(r)(E, 1) for the r computed */
	double err;              /* bound on the truncation error */
	int rank;                /* analytic rank (first r with |L^(r)| >
				  * 1e-8); 4, 5: "at least" */
	int64_t nterms;
	int64_t an[31];
};

int ecan_periods(const struct ec *E, struct ecan_per *P);
void ecan_wp(const struct ec *E, const struct ecan_per *P, double complex z,
    double complex *x, double complex *y, double *check);
void ecan_an(const struct ec *E, const struct ecq_local *L, int nl,
    int64_t M, double *an);
/* 0 when the conductor is too large (> 10^8) */
int ecan_lseries(const struct ec *E, const struct ecq_local *L, int nl,
    zint N, struct ecan_L *R);

#endif
