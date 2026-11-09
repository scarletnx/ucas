/*
 * calc2: derivatives of any order (also partial and mixed) and where they
 * exist, the gradient, Hessian, Jacobian, Laplacian and directional
 * derivatives, Wronskians, limits, Taylor series and implicit
 * differentiation (calc2.c); the command syntax is calc2_syntax.inc.
 * Every symbolic answer is checked numerically and the check printed.
 */
#ifndef CALC2_H
#define CALC2_H

#include "ucas.h"

#define C2MAXF 8          /* functions in a list (wronskian, jacobian) */
#define C2MAXO 12         /* derivative order */

/* diff F [x y ... | x n]: slots[0] first */
void c2_deriv(struct func *f, const int *slots, int n);
/* grad (0), hessian (1), laplacian (2) of f, optionally at a point */
void c2_grad(struct func *f, int what, const double *at, int hasat);
/* jacobian of the bodies fs[] (in f's parameters) wrt vars[] */
void c2_jacobian(struct func *f, const nid *fs, int nf, const int *vars,
    int nv, const double *at, int hasat);
/* directional derivative of f along v (normalized unless raw) */
void c2_ddir(struct func *f, const double *v, int nv, const double *at,
    int hasat, int raw);
/* wronskian of the bodies fs[] in variable slot */
void c2_wronskian(struct func *f, const nid *fs, int nf, int slot);
/* limit of f as slot -> a (side -1 left, +1 right, 0 both; a may be inf) */
void c2_limit(struct func *f, int slot, double a, int side);
/* Taylor series of f in slot at a, through order n */
void c2_taylor(struct func *f, int slot, double a, int n);
/* implicit differentiation of F = 0: dy/dx (and d2y/dx2 when order 2) */
void c2_implicit(struct func *f, nid F, int xs, int ys, int order);
/* more antiderivative rules for integ.c's ad (0: none) */
nid c2_antideriv(nid f, int slot, nid x);
/* does the ord-th derivative of f exist at x?  NULL or why not */
const char *c2_point_check(struct func *f, int ord, double x);
/* a number as text: 3, -1/2, pi/4, else %.10g (rotating buffers) */
const char *c2_fnum(double v);
/* the n-th derivative for a symbolic n, from a pattern (0: none) */
int c2_nth(struct func *f, int slot, uint32_t nname);

#endif
