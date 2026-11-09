/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecw.c - short Weierstrass curves over F_p (ecp_*, Montgomery limbs)
 * and over F_{p^k} (eck_*), from one template (ecw_impl.inc).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

void
mf_rand(const struct mf *F, u64 *r)
{
	nn v;

	nn_rand_below(&v, &F->P);
	mf_from_nn(F, r, &v);
}

void
fk_small(const struct fk *K, u64 *r, int64_t v)
{
	fk_zero(K, r);
	mf_from_i64(K->F, r, v);
}

void
fk_rand(const struct fk *K, u64 *r)
{
	int i;

	for (i = 0; i < K->k; i++)
		mf_rand(K->F, r + (size_t)i * FW);
}

static int
mf_inv1(const struct mf *F, u64 *r, const u64 *a)
{
	return mf_inv(F, r, a, NULL);
}

/* ---- F_p ---- */
#define FIELD struct mf
#define FE_N FW
#define PFX(x) ecp_##x
#define FE_ADD mf_add
#define FE_SUB mf_sub
#define FE_MUL mf_mul
#define FE_SQR mf_sqr
#define FE_NEG mf_neg
#define FE_COPY mf_copy
#define FE_ZERO mf_zero
#define FE_ONE mf_one
#define FE_ISZERO mf_is_zero
#define FE_ISONE mf_is_one
#define FE_EQ mf_eq
#define FE_INV mf_inv1
#define FE_SQRT mf_sqrt
#define FE_SMALL mf_from_i64
#define FE_RAND mf_rand
#include "ecw_impl.inc"
#undef FIELD
#undef FE_N
#undef PFX
#undef FE_ADD
#undef FE_SUB
#undef FE_MUL
#undef FE_SQR
#undef FE_NEG
#undef FE_COPY
#undef FE_ZERO
#undef FE_ONE
#undef FE_ISZERO
#undef FE_ISONE
#undef FE_EQ
#undef FE_INV
#undef FE_SQRT
#undef FE_SMALL
#undef FE_RAND

/* ---- F_{p^k} ---- */
#define FIELD struct fk
#define FE_N FKE
#define PFX(x) eck_##x
#define FE_ADD fk_add
#define FE_SUB fk_sub
#define FE_MUL fk_mul
#define FE_SQR fk_sqr
#define FE_NEG fk_neg
#define FE_COPY fk_copy
#define FE_ZERO fk_zero
#define FE_ONE fk_one
#define FE_ISZERO fk_is_zero
#define FE_ISONE fk_is_one
#define FE_EQ fk_eq
#define FE_INV fk_inv
#define FE_SQRT fk_sqrt
#define FE_SMALL fk_small
#define FE_RAND fk_rand
#include "ecw_impl.inc"

static char ptbuf[4][2400];
static int ptslot;

const char *
ecp_pt_str(const struct ecp_curve *E, const struct ecp_pt *P)
{
	char *o = ptbuf[ptslot++ & 3];

	if (P->inf)
		snprintf(o, sizeof ptbuf[0], "O");
	else if (E->K->bits > 64)
		snprintf(o, sizeof ptbuf[0], "(%s, %s)", mf_hex(E->K, P->x),
		    mf_hex(E->K, P->y));
	else
		snprintf(o, sizeof ptbuf[0], "(%s, %s)", mf_str(E->K, P->x),
		    mf_str(E->K, P->y));
	return o;
}

const char *
eck_pt_str(const struct eck_curve *E, const struct eck_pt *P)
{
	char *o = ptbuf[ptslot++ & 3], a[1100], b[1100];

	if (P->inf) {
		snprintf(o, sizeof ptbuf[0], "O");
		return o;
	}
	fk_print(E->K, a, sizeof a, P->x);
	fk_print(E->K, b, sizeof b, P->y);
	snprintf(o, sizeof ptbuf[0], "(%s, %s)", a, b);
	return o;
}
