/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * reduce.c - shrink the chain complex of a space by cancelling unit
 * pivots (elementary collapses / algebraic Morse theory), keeping the
 * chain homotopy equivalence (homalg.h, "reductions").
 *
 * One step: a k-cell s whose boundary meets the (k-1)-cell t with
 * coefficient c = +-1.  With C' = C without s and t:
 *   d'(r) = d r - a c d s   (a = <d r, t>; then <d' r, t> = 0)
 *   g: C' -> C   g(r) = r - a c s on k-cells, the identity elsewhere
 *   f: C -> C'   f(y) = y - y_t c d s on (k-1)-chains (t dropped), f(s) = 0
 *   h: C -> C    h(y) = y_t c s, and 1 - g f = d h + h d, f g = 1.
 * Composites: f = f2 f1, g = g1 g2, h = h1 + g1 h2 f1, so the state is,
 * per surviving cell, g(cell) (a chain of the original complex), the row
 * of f for the cell (a cochain of the original: f*(cell^*)), and one
 * rank-one term c g1(s) (x) f1^*(t^*) of h per step.
 *
 * Everything is sparse (sorted index/value vectors, int64 with i128
 * accumulation and an overflow check).  The certificates are checked on
 * the original complex, sparsely: f and g are chain maps, f g = 1, and
 * 1 - g f = d h + h d on every original cell.
 */
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "homalg.h"

typedef __int128 i128_t;

struct sv {			/* a sparse vector, indices increasing */
	uint32_t *i;
	int64_t *v;
	uint32_t n, cap;
};

static void
sv_reserve(struct sv *a, uint32_t n)
{
	if (n > a->cap) {
		a->cap = newcap(a->cap, n);
		a->i = xrealloc(a->i, a->cap * sizeof *a->i);
		a->v = xrealloc(a->v, a->cap * sizeof *a->v);
	}
}

static void
sv_free(struct sv *a)
{
	free(a->i);
	free(a->v);
	memset(a, 0, sizeof *a);
}

static void
sv_copy(struct sv *d, const struct sv *s)
{
	sv_reserve(d, s->n);
	memcpy(d->i, s->i, s->n * sizeof *s->i);
	memcpy(d->v, s->v, s->n * sizeof *s->v);
	d->n = s->n;
}

static int64_t
sv_get(const struct sv *a, uint32_t k)
{
	uint32_t lo = 0, hi = a->n;

	while (lo < hi) {
		uint32_t m = lo + (hi - lo) / 2;
		if (a->i[m] == k)
			return a->v[m];
		if (a->i[m] < k)
			lo = m + 1;
		else
			hi = m;
	}
	return 0;
}

#define VLIM ((int64_t)1 << 62)

static int64_t
fit(i128_t x)
{
	if (x >= VLIM || x <= -VLIM)
		fail("homalg: the reduction's coefficients overflow 2^62 (use "
		    "\"unreduced\")");
	return (int64_t)x;
}

/* d += m s (merge); the new indices are reported to fresh[] if given */
static struct sv tmp;

static void
sv_axpy(struct sv *d, int64_t m, const struct sv *s)
{
	uint32_t a = 0, b = 0, n = 0;

	if (m == 0 || s->n == 0)
		return;
	sv_reserve(&tmp, d->n + s->n);
	while (a < d->n || b < s->n) {
		if (b >= s->n || (a < d->n && d->i[a] < s->i[b])) {
			tmp.i[n] = d->i[a];
			tmp.v[n++] = d->v[a++];
		} else if (a >= d->n || s->i[b] < d->i[a]) {
			tmp.i[n] = s->i[b];
			tmp.v[n++] = fit((i128_t)m * s->v[b++]);
		} else {
			int64_t v = fit((i128_t)d->v[a] + (i128_t)m * s->v[b]);
			if (v) {
				tmp.i[n] = d->i[a];
				tmp.v[n++] = v;
			}
			a++;
			b++;
		}
	}
	sv_reserve(d, n);
	memcpy(d->i, tmp.i, n * sizeof *d->i);
	memcpy(d->v, tmp.v, n * sizeof *d->v);
	d->n = n;
}

static void
sv_drop(struct sv *a, uint32_t k)
{
	uint32_t j, n = 0;

	for (j = 0; j < a->n; j++)
		if (a->i[j] != k) {
			a->i[n] = a->i[j];
			a->v[n++] = a->v[j];
		}
	a->n = n;
}

struct u32l {
	uint32_t *p;
	uint32_t n, cap;
};

static void
ul_push(struct u32l *l, uint32_t x)
{
	if (l->n == l->cap) {
		l->cap = newcap(l->cap, l->n + 1);
		l->p = xrealloc(l->p, l->cap * sizeof *l->p);
	}
	l->p[l->n++] = x;
}

/* the working state, per dimension */
struct dimw {
	uint32_t n;
	uint8_t *alive;
	struct sv *col;		/* boundary columns (k >= 1), current cells */
	struct u32l *occ;	/* (k+1)-cells whose boundary may meet this */
	struct sv *g;		/* g(cell): a chain of the original */
	struct sv *f;		/* row of f: a cochain of the original */
};

struct hterm {			/* h += c g(s) (x) f^*(t^*) */
	int k;			/* the degree of s */
	int64_t c;
	struct sv gs, ft;
};

struct hred_impl {
	int top;
	struct dimw w[HMAXDIM + 1];
	struct hterm *t;
	uint32_t nt, ct;
};

static void
eliminate(struct hred_impl *R, int k, uint32_t s, uint32_t t, int64_t c)
{
	struct dimw *W = &R->w[k], *L = &R->w[k - 1];
	struct sv B = {0};
	uint32_t j;

	sv_copy(&B, &W->col[s]);
	/* h's term, from the state before this step */
	if (R->nt == R->ct) {
		R->ct = newcap(R->ct, R->nt + 1);
		R->t = xrealloc(R->t, R->ct * sizeof *R->t);
	}
	memset(&R->t[R->nt], 0, sizeof R->t[0]);
	R->t[R->nt].k = k;
	R->t[R->nt].c = c;
	sv_copy(&R->t[R->nt].gs, &W->g[s]);
	sv_copy(&R->t[R->nt].ft, &L->f[t]);
	R->nt++;
	/* columns meeting t: r -= a c (d s), and g(r) -= a c g(s) */
	for (j = 0; j < L->occ[t].n; j++) {
		uint32_t r = L->occ[t].p[j], q;
		int64_t a;
		if (r == s || !W->alive[r])
			continue;
		a = sv_get(&W->col[r], t);
		if (a == 0)
			continue;
		sv_axpy(&W->col[r], fit(-(i128_t)a * c), &B);
		sv_axpy(&W->g[r], fit(-(i128_t)a * c), &W->g[s]);
		for (q = 0; q < B.n; q++)
			if (B.i[q] != t)
				ul_push(&L->occ[B.i[q]], r);
	}
	/* rows of f on the support of d s: f(u) -= (d s)_u c f(t) */
	for (j = 0; j < B.n; j++)
		if (B.i[j] != t)
			sv_axpy(&L->f[B.i[j]], fit(-(i128_t)B.v[j] * c),
			    &L->f[t]);
	/* s and t go; s leaves the boundaries of the (k+1)-cells */
	W->alive[s] = 0;
	L->alive[t] = 0;
	if (k + 1 <= R->top)
		for (j = 0; j < W->occ[s].n; j++)
			sv_drop(&R->w[k + 1].col[W->occ[s].p[j]], s);
	sv_free(&B);
}

/* ---- the public side ---- */

static struct {
	uint32_t n, cap;
	struct hred_impl **r;
	hspace *sp;
	uint32_t **keep;	/* per handle: offsets into kp (top+1 each) */
} H;

struct hred_impl *
hred_impl(hred r)
{
	if (r >= H.n)
		fail("homalg: no reduction %u", (unsigned)r);
	return H.r[r];
}

/* check, sparsely, on the original complex: f d = d' f, d g = g d',
 * f g = 1 and 1 - g f = d h + h d */
static void reduce_check(struct hred_impl *R, hspace s, uint32_t **idx);

hred
hred_space(hspace s)
{
	struct hred_impl *R = xcalloc(1, sizeof *R);
	int top = hs_top(s), k, changed;
	uint32_t c, **idx;
	hred h;

	R->top = top;
	for (k = 0; k <= top; k++) {
		struct dimw *W = &R->w[k];
		W->n = hs_ncells(s, k);
		W->alive = xmalloc(W->n + 1);
		memset(W->alive, 1, W->n + 1);
		W->col = xcalloc(W->n + 1, sizeof *W->col);
		W->occ = xcalloc(W->n + 1, sizeof *W->occ);
		W->g = xcalloc(W->n + 1, sizeof *W->g);
		W->f = xcalloc(W->n + 1, sizeof *W->f);
		for (c = 0; c < W->n; c++) {
			sv_reserve(&W->g[c], 1);
			W->g[c].i[0] = c;
			W->g[c].v[0] = 1;
			W->g[c].n = 1;
			sv_copy(&W->f[c], &W->g[c]);
		}
	}
	for (k = 1; k <= top; k++) {
		const uint32_t *r, *cc;
		const int64_t *v;
		uint32_t m = hs_bd(s, k, &r, &cc, &v), e;
		/* triplets are sorted by (col, row): append in order */
		for (e = 0; e < m; e++) {
			struct sv *col = &R->w[k].col[cc[e]];
			sv_reserve(col, col->n + 1);
			col->i[col->n] = r[e];
			col->v[col->n++] = v[e];
			ul_push(&R->w[k - 1].occ[r[e]], cc[e]);
		}
	}
	/* cancel unit pivots until none is left */
	do {
		changed = 0;
		for (k = top; k >= 1; k--) {
			struct dimw *W = &R->w[k], *L = &R->w[k - 1];
			for (c = 0; c < W->n; c++) {
				uint32_t j, best = HNONE, bo = UINT32_MAX;
				int64_t bc = 0;
				if (!W->alive[c])
					continue;
				for (j = 0; j < W->col[c].n; j++) {
					uint32_t t = W->col[c].i[j];
					int64_t x = W->col[c].v[j];
					if ((x == 1 || x == -1) && L->alive[t] &&
					    L->occ[t].n < bo) {
						best = t;
						bo = L->occ[t].n;
						bc = x;
					}
				}
				if (best != HNONE) {
					eliminate(R, k, c, best, bc);
					changed = 1;
				}
			}
		}
	} while (changed);
	/* the surviving cells, numbered */
	idx = xcalloc((size_t)top + 1, sizeof *idx);
	for (k = 0; k <= top; k++) {
		uint32_t n = 0;
		idx[k] = xmalloc((R->w[k].n + 1) * sizeof **idx);
		for (c = 0; c < R->w[k].n; c++)
			idx[k][c] = R->w[k].alive[c] ? n++ : HNONE;
	}
	reduce_check(R, s, idx);
	if (H.n == H.cap) {
		H.cap = newcap(H.cap, H.n + 1);
		H.r = xrealloc(H.r, H.cap * sizeof *H.r);
		H.sp = xrealloc(H.sp, H.cap * sizeof *H.sp);
		H.keep = xrealloc(H.keep, H.cap * sizeof *H.keep);
	}
	h = H.n++;
	H.r[h] = R;
	H.sp[h] = s;
	H.keep[h] = (uint32_t *)idx;
	return h;
}

static uint32_t **
idx_of(hred r)
{
	hred_impl(r);
	return (uint32_t **)H.keep[r];
}

hspace
hred_space_of(hred r)
{
	hred_impl(r);
	return H.sp[r];
}

uint32_t
hred_ncells(hred r, int k)
{
	struct hred_impl *R = hred_impl(r);
	uint32_t c, n = 0;

	if (k < 0 || k > R->top)
		return 0;
	for (c = 0; c < R->w[k].n; c++)
		n += R->w[k].alive[c];
	return n;
}

/* the reduced boundary d'_k as a dense zmat (ncells(k-1) x ncells(k)) */
static void
red_bd(struct zmat *M, hred r, int k, int transpose)
{
	struct hred_impl *R = hred_impl(r);
	uint32_t **idx = idx_of(r), c, j;
	int nr = (int)hred_ncells(r, k - 1), nc = (int)hred_ncells(r, k);

	if (transpose)
		zm_init(M, nc, nr);
	else
		zm_init(M, nr, nc);
	if (k < 1 || k > R->top)
		return;
	for (c = 0; c < R->w[k].n; c++) {
		const struct sv *col = &R->w[k].col[c];
		if (!R->w[k].alive[c])
			continue;
		for (j = 0; j < col->n; j++) {
			uint32_t rr = idx[k - 1][col->i[j]], cc = idx[k][c];
			if (rr == HNONE)
				fail("internal: homalg: a reduced boundary meets "
				    "a cancelled cell");
			if (transpose)
				ZM(M, cc, rr) = z_from_i64(col->v[j]);
			else
				ZM(M, rr, cc) = z_from_i64(col->v[j]);
		}
	}
}

hcx
hred_cochains(hred r)
{
	struct hred_impl *R = hred_impl(r);
	int top = R->top, n, *g = xmalloc((size_t)(top + 1) * sizeof *g);
	struct zmat *d = xcalloc((size_t)top + 1, sizeof *d);
	char name[300];
	hcx c;

	for (n = 0; n <= top; n++)
		g[n] = (int)hred_ncells(r, n);
	for (n = 0; n <= top; n++)
		red_bd(&d[n], r, n + 1, 1);	/* d^n = d'_{n+1}^T */
	snprintf(name, sizeof name, "C^*(%s) reduced", hs_name(H.sp[r]));
	c = hcx_new(name, 0, top, g, d, NULL);
	for (n = 0; n <= top; n++)
		zm_free(&d[n]);
	free(d);
	free(g);
	return c;
}

hcx
hred_chains(hred r)
{
	struct hred_impl *R = hred_impl(r);
	int top = R->top, n, *g = xmalloc((size_t)(top + 1) * sizeof *g);
	struct zmat *d = xcalloc((size_t)top + 1, sizeof *d);
	char name[300];
	hcx c;

	for (n = 0; n <= top; n++)
		g[top - n] = (int)hred_ncells(r, n);
	for (n = 0; n <= top; n++)
		red_bd(&d[top - n], r, n, 0);
	snprintf(name, sizeof name, "C_*(%s) reduced", hs_name(H.sp[r]));
	c = hcx_new(name, -top, 0, g, d, NULL);
	for (n = 0; n <= top; n++)
		zm_free(&d[n]);
	free(d);
	free(g);
	return c;
}

/* f^*: a reduced n-cochain x to the original one, y = sum_u x_u f-row(u) */
void
hred_cochain_back(hred r, int n, const zint *x, zint *y)
{
	struct hred_impl *R = hred_impl(r);
	uint32_t **idx = idx_of(r), c, j;

	for (c = 0; c < R->w[n].n; c++)
		y[c] = Z_ZERO;
	for (c = 0; c < R->w[n].n; c++) {
		const struct sv *fr = &R->w[n].f[c];
		if (!R->w[n].alive[c] || z_is_zero(x[idx[n][c]]))
			continue;
		for (j = 0; j < fr->n; j++)
			y[fr->i[j]] = z_addmul(y[fr->i[j]], x[idx[n][c]],
			    z_from_i64(fr->v[j]));
	}
}

/* g: a reduced n-chain to the original one */
void
hred_chain_back(hred r, int n, const zint *x, zint *y)
{
	struct hred_impl *R = hred_impl(r);
	uint32_t **idx = idx_of(r), c, j;

	for (c = 0; c < R->w[n].n; c++)
		y[c] = Z_ZERO;
	for (c = 0; c < R->w[n].n; c++) {
		const struct sv *gc = &R->w[n].g[c];
		if (!R->w[n].alive[c] || z_is_zero(x[idx[n][c]]))
			continue;
		for (j = 0; j < gc->n; j++)
			y[gc->i[j]] = z_addmul(y[gc->i[j]], x[idx[n][c]],
			    z_from_i64(gc->v[j]));
	}
}

/* ---- the certificate ---- */

/* acc (dense, i128, over original k-cells) += m * v */
static void
acc_add(i128_t *acc, int64_t m, const struct sv *v)
{
	uint32_t j;

	for (j = 0; j < v->n; j++)
		acc[v->i[j]] += (i128_t)m * v->v[j];
}

static void
bd_orig(i128_t *acc, hspace s, int k, uint32_t cell, int64_t m,
    uint32_t *start)
{
	const uint32_t *r, *c;
	const int64_t *v;
	uint32_t e;

	if (k < 1)
		return;
	hs_bd(s, k, &r, &c, &v);
	for (e = start[cell]; e < start[cell + 1]; e++)
		acc[r[e]] += (i128_t)m * v[e];
}

static void
reduce_check(struct hred_impl *R, hspace s, uint32_t **idx)
{
	int top = R->top, k;
	uint32_t **cs = xcalloc((size_t)top + 2, sizeof *cs);
	/* frow inverted: for original k-cell y, the (u, coef) with f(y)_u */
	struct u32l **fin = xcalloc((size_t)top + 1, sizeof *fin);
	struct u32l **tin = xcalloc((size_t)top + 1, sizeof *tin);
	uint32_t maxn = 1, c, j;
	i128_t *acc, *acc2;

	for (k = 0; k <= top; k++)
		if (R->w[k].n + 1 > maxn)
			maxn = R->w[k].n + 1;
	acc = xcalloc(maxn, sizeof *acc);
	acc2 = xcalloc(maxn, sizeof *acc2);
	for (k = 1; k <= top; k++) {
		const uint32_t *r, *cc;
		const int64_t *v;
		uint32_t m = hs_bd(s, k, &r, &cc, &v), e = 0;
		cs[k] = xmalloc((R->w[k].n + 1) * sizeof **cs);
		for (c = 0; c <= R->w[k].n; c++) {
			while (e < m && cc[e] < c)
				e++;
			cs[k][c] = e;
		}
	}
	for (k = 0; k <= top; k++) {
		fin[k] = xcalloc(R->w[k].n + 1, sizeof **fin);
		tin[k] = xcalloc(R->w[k].n + 1, sizeof **tin);
		for (c = 0; c < R->w[k].n; c++)
			if (R->w[k].alive[c])
				for (j = 0; j < R->w[k].f[c].n; j++)
					ul_push(&fin[k][R->w[k].f[c].i[j]], c);
	}
	for (j = 0; j < R->nt; j++) {
		uint32_t q;
		/* h: (k-1)-cells -> k-cells; index by the (k-1)-cell */
		for (q = 0; q < R->t[j].ft.n; q++)
			ul_push(&tin[R->t[j].k - 1][R->t[j].ft.i[q]], j);
	}
	for (k = 0; k <= top; k++) {
		uint32_t nk = R->w[k].n;
		/* f g = 1: for a surviving x, f(g(x)) = x */
		for (c = 0; c < nk; c++) {
			const struct sv *gx = &R->w[k].g[c];
			if (!R->w[k].alive[c])
				continue;
			for (j = 0; j < nk; j++)
				acc[j] = 0;
			for (j = 0; j < gx->n; j++) {
				uint32_t y = gx->i[j], q;
				for (q = 0; q < fin[k][y].n; q++) {
					uint32_t u = fin[k][y].p[q];
					acc[u] += (i128_t)gx->v[j] *
					    sv_get(&R->w[k].f[u], y);
				}
			}
			for (j = 0; j < nk; j++)
				if (acc[j] != (j == c))
					fail("internal: homalg: reduction: f g != "
					    "1 in degree %d", k);
		}
		/* g is a chain map: d g(x) = g(d' x), as original chains */
		if (k >= 1)
			for (c = 0; c < nk; c++) {
				const struct sv *gx = &R->w[k].g[c];
				const struct sv *dx = &R->w[k].col[c];
				uint32_t nl = R->w[k - 1].n;
				if (!R->w[k].alive[c])
					continue;
				for (j = 0; j < nl; j++)
					acc[j] = 0;
				for (j = 0; j < gx->n; j++)
					bd_orig(acc, s, k, gx->i[j], gx->v[j],
					    cs[k]);
				for (j = 0; j < dx->n; j++)
					acc_add(acc, -dx->v[j],
					    &R->w[k - 1].g[dx->i[j]]);
				for (j = 0; j < nl; j++)
					if (acc[j] != 0)
						fail("internal: homalg: reduction: "
						    "g is not a chain map");
			}
		/* f is a chain map, as cochains: d^T(f-row u) = sum over
		 * (k+1)-cells w of <d' w, u> f-row(w), checked on every
		 * original (k+1)-cell y: sum_u' ... done as f(d y) = d' f(y) */
		if (k + 1 <= top) {
			uint32_t ny = R->w[k + 1].n;
			for (c = 0; c < ny; c++) {
				uint32_t q;
				/* acc over current k-cells: f(d y) */
				for (j = 0; j < nk; j++)
					acc[j] = 0;
				{
					const uint32_t *r, *cc;
					const int64_t *v;
					uint32_t e;
					hs_bd(s, k + 1, &r, &cc, &v);
					for (e = cs[k + 1][c]; e < cs[k + 1][c + 1];
					    e++)
						for (q = 0; q < fin[k][r[e]].n; q++) {
							uint32_t u = fin[k][r[e]].p[q];
							acc[u] += (i128_t)v[e] *
							    sv_get(&R->w[k].f[u],
							    r[e]);
						}
				}
				/* minus d'(f(y)) */
				for (q = 0; q < fin[k + 1][c].n; q++) {
					uint32_t w = fin[k + 1][c].p[q];
					int64_t fy = sv_get(&R->w[k + 1].f[w], c);
					acc_add(acc, -fy, &R->w[k + 1].col[w]);
				}
				for (j = 0; j < nk; j++)
					if (acc[j] != 0)
						fail("internal: homalg: reduction: "
						    "f is not a chain map");
			}
		}
		/* 1 - g f = d h + h d on every original k-cell y */
		for (c = 0; c < nk; c++) {
			uint32_t q;
			for (j = 0; j < nk; j++)
				acc[j] = 0;
			acc[c] += 1;
			for (q = 0; q < fin[k][c].n; q++) {	/* - g f(y) */
				uint32_t u = fin[k][c].p[q];
				acc_add(acc, -sv_get(&R->w[k].f[u], c),
				    &R->w[k].g[u]);
			}
			/* - d h(y): h(y) in k+1 */
			if (k + 1 <= top) {
				uint32_t n1 = R->w[k + 1].n;
				for (j = 0; j < n1; j++)
					acc2[j] = 0;
				for (q = 0; q < tin[k][c].n; q++) {
					const struct hterm *T = &R->t[tin[k][c].p[q]];
					int64_t fy = sv_get(&T->ft, c);
					acc_add(acc2, fit((i128_t)T->c * fy),
					    &T->gs);
				}
				for (j = 0; j < n1; j++)
					if (acc2[j] != 0) {
						int64_t m = fit(acc2[j]);
						bd_orig(acc, s, k + 1, j, -m,
						    cs[k + 1]);
					}
			}
			/* - h(d y) */
			if (k >= 1) {
				const uint32_t *r, *cc;
				const int64_t *v;
				uint32_t e;
				hs_bd(s, k, &r, &cc, &v);
				for (e = cs[k][c]; e < cs[k][c + 1]; e++)
					for (q = 0; q < tin[k - 1][r[e]].n; q++) {
						const struct hterm *T =
						    &R->t[tin[k - 1][r[e]].p[q]];
						int64_t fy = sv_get(&T->ft, r[e]);
						acc_add(acc, fit(-(i128_t)v[e] *
						    T->c * fy), &T->gs);
					}
			}
			for (j = 0; j < nk; j++)
				if (acc[j] != 0)
					fail("internal: homalg: reduction: 1 - g f "
					    "!= d h + h d in degree %d", k);
		}
	}
	hal_checks[HK_REDUCE]++;
	for (k = 0; k <= top; k++) {
		for (c = 0; c <= R->w[k].n; c++) {
			free(fin[k][c].p);
			free(tin[k][c].p);
		}
		free(fin[k]);
		free(tin[k]);
		free(cs[k]);
	}
	free(cs[top + 1]);
	free(cs);
	free(fin);
	free(tin);
	free(acc);
	free(acc2);
	(void)idx;
}
