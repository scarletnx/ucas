/*
 * homalg.h - exact homological algebra over Z (part B of absalg.md).
 *
 * Layers, each usable on its own:
 *
 *   spaces   (simp.c)  finite cell complexes: simplicial complexes (simplices
 *                      in SoA vertex columns) and cellular complexes given by
 *                      their boundary matrices; the standard models and the
 *                      constructions (product, wedge, suspension).
 *
 * Handles are small integers into arenas that live for the process; nothing
 * is freed one object at a time.  Every constructor checks what it claims
 * (d o d = 0, a simplicial quotient is a 2-to-1 covering ...) and fail()s
 * (util.c) otherwise.
 */
#ifndef HOMALG_H
#define HOMALG_H

#include <stdint.h>

#define HNONE UINT32_MAX
#define HMAXDIM 30		/* cells of dimension 0..HMAXDIM */

/* ---- spaces (simp.c) ---- */

typedef uint32_t hspace;	/* a finite cell complex */

enum { HS_CELLULAR, HS_SIMPLICIAL };

/*
 * A space is its cellular chain complex: cells in dimensions 0..top and the
 * boundaries d_k: C_k -> C_{k-1} (k = 1..top) as sparse triplets (row = a
 * (k-1)-cell, col = a k-cell, val != 0), sorted by (col, row).  A simplicial
 * complex keeps its simplices too: dimension k as k+1 vertex columns, the
 * rows sorted lexicographically, vertices increasing within a simplex; the
 * simplicial boundary is sum_i (-1)^i [v0 .. ^vi .. vk].
 */
int hs_top(hspace s);				/* -1 for the empty space */
int hs_kind(hspace s);
const char *hs_name(hspace s);
uint32_t hs_ncells(hspace s, int k);		/* 0 outside 0..top */
uint32_t hs_bd(hspace s, int k, const uint32_t **row,
    const uint32_t **col, const int64_t **val);	/* entries of d_k */
/* simplicial: the j-th vertex column of the k-simplices (NULL otherwise) */
const uint32_t *hs_simplex_col(hspace s, int k, int j);
uint32_t hs_nvert(hspace s);
long long hs_euler(hspace s);

/* simplicial complex from facets: facet f has vertices v[off[f]..off[f+1]) */
hspace hs_simplicial(const char *name, const uint32_t *v, const uint32_t *off,
    uint32_t nfacets);
/* cellular complex from boundary matrices: d[k-1] is ncells[k-1] x ncells[k]
 * (row-major), k = 1..top.  Checks d_{k-1} d_k = 0. */
hspace hs_cellular(const char *name, int top, const uint32_t *ncells,
    const int64_t *const *d);

/* standard models */
hspace hs_sphere(int n, int simplicial);	/* S^n: d(Delta^{n+1}) or e0 u en */
hspace hs_torus(int n);				/* T^n = (S^1)^n, cellular */
hspace hs_torus_simplicial(void);		/* 7-vertex Moebius torus */
hspace hs_rp(int n);				/* RP^n, one cell per dimension */
hspace hs_rp_simplicial(int n);		/* RP^2: 6 vertices; n <= 3: S^n/+-1
						 * on sd(cross-polytope) */
hspace hs_klein(int simplicial);		/* Klein bottle */
hspace hs_lens(uint64_t m, uint64_t q);		/* L(m;q), cellular */
hspace hs_cp(int n);				/* CP^n, cellular */
hspace hs_point(void);
hspace hs_product(hspace a, hspace b);		/* cells e x f, Koszul signs */
hspace hs_wedge(hspace a, hspace b);		/* at the first 0-cells */
hspace hs_suspension(hspace a);		/* unreduced: N, S, e x (0,1) */

/* "S^2 x RP^3", "L(9;1)", "T^2 v S^1" ...: a space by its name; HNONE and
 * *err set when the text is not one (grammar in simp.c) */
hspace hs_parse(const char *text, const char **err);

#endif
