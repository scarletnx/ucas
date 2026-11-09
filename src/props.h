/*
 * props.c: properties of functions and equations.
 *
 *   monotonic F [x] [in a..b]      increasing/decreasing, else the pieces
 *   period F [x], periodic F [x]   the fundamental period
 *   identify EXPR | LHS = RHS | F  classify, with a list of properties
 *   parity F, symmetry F           even/odd; axes and centres of symmetry
 *   bounded F                      bounded above/below
 *   injective/surjective/bijective F [onto a..b]
 *   continuity F, discontinuities F  removable, jump, infinite, oscillating
 *   convexity F (convex, concave)  intervals and inflection points
 *   asymptotes F                   vertical, horizontal, oblique
 *   zeros F (roots)                the zeros, with multiplicity
 */
#ifndef PROPS_H
#define PROPS_H

#include "ucas.h"

struct P;
int props_command(struct P *p);	/* parse.c: 1 when it was one of ours */
/* nlp.c: an equation or ODE in an "identify"/"classify" sentence, as the
 * ucas statement "identify ..." (static buffer), else NULL */
const char *props_nlp_raw(const char *sent);

/*
 * The fundamental period of f in parameter slot, quietly: 1 when f is
 * periodic (*p set: found symbolically or by an exhaustive search over an
 * integer domain, and checked at sample points), 0 when not periodic or
 * no period was found, -1 when f does not depend on the parameter (every
 * p is a period).
 */
int fn_period(struct func *f, int slot, double *p);
/*
 * Monotonicity of a one-parameter f on [lo, hi] (lo > hi: its domain):
 * 2 strictly increasing, 1 non-decreasing, 3 constant, -1 non-increasing,
 * -2 strictly decreasing, 0 neither.
 */
int fn_monotonic(struct func *f, double lo, double hi);
/* the zeros of a one-parameter real f in [lo, hi], at most max */
int fn_zeros(struct func *f, double lo, double hi, double *x, int max);
/*
 * Discontinuities of a one-parameter real f in [lo, hi]: their x and kind
 * (1 removable, 2 jump, 3 infinite, 4 oscillating), at most max.
 */
int fn_discontinuities(struct func *f, double lo, double hi, double *x,
    int *kind, int max);

/* ode.c: the right-hand side of var's equation var^(order) = rhs */
const char *ode_rhs(uint32_t var, int *order);

#endif
