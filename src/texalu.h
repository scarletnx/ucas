/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * texalu.c: the texture unit's fixed function as operations of the
 * language, and the rewrites into them a cost model makes worthwhile.
 *
 * T is a table [t0, t1, ...] of n texels.  A read with an address mode
 * takes an integer texel index i (any integer type) and is total:
 *
 *   fetch_repeat(T, i)        T[i mod n]                (floored mod)
 *   fetch_mirror(T, i)        T[min(r, 2n - 1 - r)], r = i mod 2n
 *   fetch_clamp(T, i)         T[min(max(i, 0), n - 1)]
 *   fetch_mirror_clamp(T, i)  T[min(max(i, -1 - i), n - 1)]
 *   fetch_border(T, i, k)     0 <= i < n ? T[i] : k
 *
 * (Vulkan's wrapping operations on the integer texel coordinate of a
 * nearest read.)  A linear filter takes a texel coordinate, texel j's
 * centre at j + 1/2, clamps to the edge, and has its weights on the
 * sub-texel grid of B bits:
 *
 *   tex_linearB(T, x)         s = x - 1/2, i0 = floor(s),
 *                             a = floor((s - i0) 2^B + 1/2) / 2^B,
 *                             (1 - a) T[c(i0)] + a T[c(i0 + 1)]
 *   tex_bilinearB(T, W, x, y) the same on both axes of the table read as
 *                             rows of W texels: four texels, their
 *                             weights the products of the two axes'
 *   tex_minB(T, x), tex_maxB  the same footprint with a MIN or MAX
 *   tex_biminB(T, W, x, y),   reduction sampler: the least or the
 *   tex_bimaxB                greatest of its texels whose weight is
 *                             not 0, exactly
 *
 * An array texture's layer: fetch_layer(T, z) is layer clamp(RNE(z), 0,
 * n - 1) of n layers, each holding one entry of T.  The specification's
 * RNE is roundTiesToEven (preferred) or floor(z + 1/2) (allowed), so the
 * two differ at z = k + 1/2 for an even k; the operation is the first and
 * a rewrite into it holds only where the unit's choice does not matter.
 * A NaN selects no layer: undefined.
 *
 * with c the clamp to the table (the row, the column).  The texels are
 * f32, the weighted sum is exact and rounded once to f32 (to nearest,
 * ties to even).  B is 1..14 (tex_linear8: weights in 1/256).
 *
 * A cost model prices them by class: tex_nearest (a fetch_ read; its
 * coordinate from the index costs a conversion and a multiply more),
 * tex_sample (a filtered read of an 8-bit normalised texture: every
 * texel k/255) and tex_sample_f32 (of a 32-bit float texture).  A model
 * that does not list a class cannot use the operation.  Two model
 * parameters say what the hardware's unit does beyond the definition
 * above: subtexel, the bits of a texel it resolves (coordinates and
 * weights snap to that grid: by less than 2^-subtexel), and tex_within, how
 * many ulps of the result its filter may be from the exact weighted sum.
 * A rewrite is offered only where it holds whichever way the unit snaps
 * and rounds within those; without them, nothing filtered is offered.
 *
 *   texalu F [tol E] [model M]   the reads and weighted sums of table
 *       entries in F as sampled reads, where the model prices them
 *       lower: each with what it holds under, its derivation, and what
 *       the pipeline needs for it (a proposal, never applied)
 */
#ifndef UCAS_TEXALU_H
#define UCAS_TEXALU_H

#include "ucas.h"

#define TEX_MAXBITS 14
#define TEX_KI(bits, width) ((int64_t)(bits) | (int64_t)(width) << 8)
#define TEX_BITS(ki) ((int)((ki) & 31))
#define TEX_WIDTH(ki) ((int64_t)((ki) >> 8 & 0xffffff))
/* a filter's reduction in ki: the weighted sum (0), or the least or the
 * greatest of the texels with a non-zero weight (tex_min, tex_max) */
#define TEX_REDMIN (1 << 5)
#define TEX_REDMAX (2 << 5)
#define TEX_RED(ki) ((int)((ki) & (3 << 5)))
/* a fetch_ op's ki: an R8_SRGB texture of the bytes, read by texelFetch
 * (fetch_srgb: its table is the 256 values the sRGB EOTF decodes) */
#define TEX_SRGB 1

/* is tab the 256 f32 values of the sRGB EOTF of k/255, rounded once */
int tex_srgb_table(nid tab);

/* a fetch_ op's texel for index i of n: 0..n-1, or -1 the border */
int64_t tex_wrap(int op, int64_t i, int64_t n);
/* fetch_layer's layer for coordinate z of n layers: clamp(RNE(z), 0, n - 1)
 * with RNE to nearest, a tie to the even one; -1 for a NaN */
int64_t tex_layer(double z, int64_t n);
/* a linear filter's first texel floor(x - 1/2) and the weight of the
 * next one, on the grid of bits; NaN for a coordinate that is not finite */
double tex_weight(double x, int bits, int64_t *i0);
/* the correctly rounded f32 of sum w[k] t[k] (n <= 4; t f32 values, w
 * with at most 29 significant bits) */
double tex_blend(const double *t, const double *w, int n);
/* tex_linear / tex_bilinear at one point: tv(ctx, j) is texel j */
double tex_filter(int op, int64_t ki, int64_t n, double x, double y,
    double (*tv)(void *ctx, int64_t j), void *ctx);
/* a fetch_ node as plain operations (its definition): for SAT */
nid tex_expand(nid n);
/* the node's cost class, and its price under m (infinity: unavailable) */
int tex_class(nid n);
float tex_cost(const struct model *m, nid n, int *cls);
/* name -> op and, for the filters, the bits; 0 if not one */
int tex_lookup(const char *s, size_t n, int *bits);
void tex_print_name(struct sbuf *b, nid n);

struct P;
int texalu_command(struct P *p);		/* parse.c */
const char *texalu_nlp_raw(const char *sent);	/* nlp.c */
/* find's section: the sampled reads the model prices lower */
int tex_find(const struct func *f);

enum { TEXR_NONE, TEXR_PROVED, TEXR_NOTCHEAPER, TEXR_UNAVAILABLE,
	TEXR_OVERTOL };
struct texres {
	int status;		/* TEXR_ */
	nid body;		/* the rewritten body (TEXR_PROVED) */
	float before, after;	/* costs under the active model */
	double err, tol;	/* the derived bound, the budget; 0: exact */
	int exact;		/* an integer rewrite: equal everywhere */
	struct sbuf what;	/* one line per read: what became what */
	struct sbuf holds;	/* the conditions and why each holds */
	struct sbuf proof;	/* the derivation */
	struct sbuf pipeline;	/* what the pipeline needs */
	struct sbuf why;	/* not PROVED: why */
};
int tex_search(const struct func *f, double tol, struct texres *r);
void texres_free(struct texres *r);

#endif
