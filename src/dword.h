/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * dword.c: fp64 arithmetic as pairs of f32 (double-word arithmetic), with
 * a derived error bound, offered when the active cost model prices the
 * pairs below its f64 operations.  fpfold.c: the conversion round trips a
 * model says the driver folds, and a fold-proof form of their rounding.
 *
 *   dword F [tol R | atol T | ulp X] [model M]
 *   foldproof F [model M]
 */
#ifndef DWORD_H
#define DWORD_H

#include "ucas.h"

struct P;
int dword_command(struct P *p);		/* parse.c: 1 when it was ours */
const char *dword_nlp_raw(const char *sent);	/* nlp.c: English, or NULL */

/* what the result must meet: 0 the default for its sink, or a bound */
enum { DW_DEFAULT, DW_TOL, DW_ATOL, DW_ULP };
struct dwreq {
	int kind;
	double v;
	int anyprice;		/* the cheapest form that meets it, whatever the
				 * f64 operations cost: the caller prices */
};

/* the cheapest f32 form of f's f64 arithmetic under m that meets rq */
struct dwres {
	int fired;		/* a form is cheaper and meets rq */
	int mode;		/* 0 f32 alone, 1 pairs */
	nid hi, lo;		/* the result: hi (+ lo, 0 when it is exact) */
	nid body;		/* the function's new body */
	double cost64, cost;	/* under m: as written, the form */
	long double rel, abs_;	/* derived bounds against the exact value */
	uint8_t *eft;		/* per node of the form: needs NoContraction */
	uint32_t neft;
	char why[200];		/* not fired: the reason */
};
int dword_lower(const struct func *f, const struct model *m,
    const struct dwreq *rq, struct dwres *out, struct sbuf *report);
void dword_res_free(struct dwres *r);
/*
 * spirv.c: a loop's f64 sum, s = s0, then n passes of s = s + T (or
 * s - T): f's body is that pass, s its f64 parameter ps, and the pair the
 * passes carry instead is its f32 parameters ph and pl.  s0 is the pair
 * (s0h, s0l: f32 constants or parameters), whose value is within e0 of
 * s0's exact value in [s0lo, s0hi].  once[] are the f64 values of the pass
 * computed before the loop (paid once as written).  A store takes the
 * pair recombined, f64(hi) + f64(lo), its rounding in the bound.  out->hi
 * and lo are the pass's new pair, out->rel and abs_ the bound after the n
 * passes.
 */
struct dwacc {
	int ps, ph, pl;
	int64_t n;
	nid s0h, s0l;
	double s0lo, s0hi;
	long double e0;
	nid once[16];
	int nonce;
	int nconv, nstore;	/* its readers: conversions to f32, f64 stores */
};
int dword_accum(const struct func *f, const struct dwacc *a,
    const struct model *m, const struct dwreq *rq, struct dwres *out,
    struct sbuf *report);
/* find.c: what a cheaper f32 form of f's f64 arithmetic costs, when there
 * is one (1), in a line */
int dword_find(struct func *f);
/* the interval of an f32 or integer expression of f; 0: not known */
int dword_interval(const struct func *f, nid n, double *lo, double *hi);

/* ---- fpfold.c ---- */
int fold_command(struct P *p);
/* m folds the round trip from -> mid -> from (a conversion chain) */
int model_folds(const struct model *m, ty_t from, ty_t mid);
/* cost: the nodes under root that cost nothing because m's driver folds
 * them (out[n] = 1), their count; and what cmd_cost says about them */
int fold_marks(const struct model *m, nid root, uint8_t *out, uint32_t lim);
void fold_cost_notes(const struct func *f, const struct model *m);
/* root as m's driver runs it: each folded round trip is its operand */
nid fold_apply(const struct model *m, nid root);
/* the fold-proof form of f32(f16(x)) for x: f32 with values in [lo, hi]
 * (known: 0 none): integer rounding of the mantissa bits; *full: the form
 * that also takes subnormal results, infinities and NaN */
nid fold_proof_form(nid x, int known, double lo, double hi, int *full);
/* one folded round trip of f, reported (its rounding relied on or not,
 * the form, the proof); the form, the operand when it is the identity
 * there (*identity), 0 when a step is not proved */
nid fold_one(const struct func *f, nid trip, const struct model *m,
    int *identity);
/* spirv.c: simplify ... foldproof writes those forms into the module */
void spirv_dword_opts(int foldproof);
/* spirv.c: simplify ... tol R or atol T, the f64 results' requirement */
void spirv_dword_req(int kind, double v);

#endif
