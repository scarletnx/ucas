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
	int loose;		/* relies on a GPU's float precision (2.5 ulp
				 * division), looser than ucas's correctly
				 * rounded f32: checks by evaluation do not
				 * cover it, and nothing may be dropped from it
				 * on their word */
};
int irw_cands(const struct func *f, nid root, struct irw **out);
void irw_free(struct irw *r, int n);
void irw_print(const struct irw *r, const char *ind);

#endif
