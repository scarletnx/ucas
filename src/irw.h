/* SPDX-License-Identifier: GPL-3.0-or-later */
/* irw.c: integer rewrites with their derivations (see there) */
#ifndef UCAS_IRW_H
#define UCAS_IRW_H

#include "ucas.h"

struct irw {
	nid e;			/* the whole expression, rewritten */
	nid at, to;		/* the subterm and what replaces it */
	char what[200];
	char cond[400];		/* what it holds under, and why that holds */
	char *proof;		/* the derivation, one step a line */
	int nearmiss;		/* not proved: the derivation says why */
	int quiet;		/* not worth a "derived, but not cheaper" line
				 * (ifx.c: every integer product has one) */
	int loose;		/* relies on a GPU's float precision (2.5 ulp
				 * division), looser than ucas's correctly
				 * rounded f32: checks by evaluation do not
				 * cover it, and nothing may be dropped from it
				 * on their word */
	int nosat;		/* a multiplier against a divider: SAT does not
				 * finish on it; the derivation is the proof
				 * and the second opinion is the domain's
				 * points (all of them, or a sample) */
	int derived;		/* the derivation is the whole proof: the
				 * divisor's algebra, or the map's value read
				 * back at every point of its key's range
				 * (simplify writes these into a module) */
	int factor;		/* a common factor taken out of a sum, in a
				 * sized integer type (ifx.c): what simplify
				 * reads a sum for */
};

/* ranges of root's subterms on f's domain (g: f with the parameters'
 * ranges narrowed by the where clause) */
struct ivc {
	const struct func *f;
	struct func g;
	struct plan p;
	struct ival *iv;
	nid root;
};
int irw_range(struct ivc *c, nid s, int64_t need_lo, int64_t *lo, int64_t *hi,
    char *why, size_t nwhy);
struct irw *irw_add(struct irw **out, int *n, int *cap, nid root, nid from,
    nid to, const char *what, const char *cond, const char *proof, int keep);
/* idrw.c: identities solved exactly or found by search (see there) */
void idrw_node(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap);
void idrw_maps(struct ivc *c, nid root, struct irw **out, int *cnt, int *cap);
/* x / d for x in [0, n] as (x M) >> s in u32: s, or 0 when none serves */
int idrw_divk(uint64_t n, uint64_t d, uint64_t *m);
int irw_cands(const struct func *f, nid root, struct irw **out);
/* ifx.c: integers as floats and floats as integers, at node n of root */
void ifx_cands(struct ivc *c, nid root, nid n, uint8_t *inner, uint32_t lim,
    struct irw **out, int *cnt, int *cap);
/* ifx.c: x / k and x % k by a constant on the f32 pipe */
void ifx_divk(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap);
/* npn.c: a bitwise cone of 4 leaves as its minimum network of 3-input
 * tables, where the model has a lut fusion */
void npn_cands(struct ivc *c, nid root, struct irw **out, int *cnt, int *cap);
void irw_free(struct irw *r, int n);
void irw_print(const struct irw *r, const char *ind);

#endif
