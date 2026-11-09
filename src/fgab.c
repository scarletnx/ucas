/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * fgab.c - finitely generated abelian groups as subquotients A/B of Z^n.
 *
 * A group keeps A as a basis in column Hermite form (so coordinates come
 * from a triangular solve), B as a basis in the same form, R = the
 * A-coordinates of B (G = Z^a / R Z^b) and the Smith form U R V = D with
 * U^-1, which gives the invariant factors, canonical coordinates and
 * generators.  Everything else is lattice algebra in A-coordinates:
 *
 *   x = 0 in G        U c has u_i = 0 mod d_i (and 0 where d_i is free)
 *   ker f             {c : M c in R_H Z^*} = the first block of ker [M | R_H]
 *   im f, coker f     A_H M + B_H, and A_H / (A_H M + B_H)
 *   G[m]              {c : m c in R Z^b}
 *   Hom(G, H)         {M : M R_G in R_H Z^*} / {M : M Z^a in R_H Z^*}
 *   G (x) H           Z^{a a'} / (R (x) 1, 1 (x) R')
 *   Tor, Ext          from 0 -> Z^b -R-> Z^a -> G -> 0 (R injective)
 *
 * Results that claim something carry a certificate that is checked here
 * (a preimage, an inverse map, a well-definedness check).  Each top-level
 * function compacts the arena down to its outputs before returning.
 */
#include "absalg.h"
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

/* ---- lattices in column Hermite form ---- */

/* the pivot row of column j: its first nonzero entry */
static int
piv_row(const struct zmat *A, int j)
{
	int i;

	for (i = 0; i < A->r; i++)
		if (ZM(A, i, j) != 0)
			return i;
	return -1;
}

/* x = A c for A a column-HNF basis; 0 when x is not in the lattice */
static int
lat_coords(const struct zmat *A, const zint *x, zint *c)
{
	zint *res = xmalloc(((size_t)A->r + 1) * sizeof *res);
	int i, j, ok = 1;

	memcpy(res, x, (size_t)A->r * sizeof *res);
	for (j = 0; j < A->c; j++) {
		int p = piv_row(A, j);
		zint q, r;
		z_divmod_trunc(res[p], ZM(A, p, j), &q, &r);
		if (r != 0) {
			ok = 0;
			break;
		}
		c[j] = q;
		if (q != 0)
			for (i = p; i < A->r; i++)
				if (ZM(A, i, j) != 0)
					res[i] = z_submul(res[i], q, ZM(A, i, j));
	}
	for (i = 0; ok && i < A->r; i++)
		if (res[i] != 0)
			ok = 0;
	free(res);
	return ok;
}

/* C = coordinates of the columns of X in the basis A */
static int
lat_coords_mat(const struct zmat *A, const struct zmat *X, struct zmat *C)
{
	zint *x = xmalloc(((size_t)X->r + 1) * sizeof *x);
	zint *c = xmalloc(((size_t)A->c + 1) * sizeof *c);
	struct zmat t;
	int i, j, ok = 1;

	if (X->r != A->r)
		fail("internal: coordinates of vectors in Z^%d in a lattice of "
		    "Z^%d", X->r, A->r);
	zm_init(&t, A->c, X->c);
	for (j = 0; j < X->c && ok; j++) {
		zm_col(x, X, j);
		if (!lat_coords(A, x, c))
			ok = 0;
		else
			for (i = 0; i < A->c; i++)
				ZM(&t, i, j) = c[i];
	}
	free(x);
	free(c);
	if (ok)
		zm_move(C, &t);
	else
		zm_free(&t);
	return ok;
}

/* ---- keeping outputs ---- */

static void
keep_all(zmark m, struct fgab **gs, int ng, struct zmat **ms, int nm)
{
	size_t n = 0, o = 0;
	zint *buf;
	int i, k;

#define GM(g, k) ((k) == 0 ? &(g)->A : (k) == 1 ? &(g)->B : (k) == 2 ? \
	&(g)->R : (k) == 3 ? &(g)->S.D : (k) == 4 ? &(g)->S.U : (k) == 5 ? \
	&(g)->S.Ui : (k) == 6 ? &(g)->S.V : &(g)->S.Vi)
#define SZ(m) ((m)->a ? (size_t)(m)->r * (size_t)(m)->c : 0)
	for (i = 0; i < ng; i++) {
		if (!gs[i])
			continue;
		for (k = 0; k < 8; k++)
			n += SZ(GM(gs[i], k));
		n += (size_t)gs[i]->a;
	}
	for (i = 0; i < nm; i++)
		if (ms[i])
			n += SZ(ms[i]);
	buf = xmalloc((n + 1) * sizeof *buf);
	for (i = 0; i < ng; i++) {
		if (!gs[i])
			continue;
		for (k = 0; k < 8; k++) {
			struct zmat *mm = GM(gs[i], k);
			memcpy(buf + o, mm->a, SZ(mm) * sizeof *buf);
			o += SZ(mm);
		}
		memcpy(buf + o, gs[i]->d, (size_t)gs[i]->a * sizeof *buf);
		o += (size_t)gs[i]->a;
	}
	for (i = 0; i < nm; i++)
		if (ms[i]) {
			memcpy(buf + o, ms[i]->a, SZ(ms[i]) * sizeof *buf);
			o += SZ(ms[i]);
		}
	z_release_keep(m, buf, n);
	o = 0;
	for (i = 0; i < ng; i++) {
		if (!gs[i])
			continue;
		for (k = 0; k < 8; k++) {
			struct zmat *mm = GM(gs[i], k);
			memcpy(mm->a, buf + o, SZ(mm) * sizeof *buf);
			o += SZ(mm);
		}
		memcpy(gs[i]->d, buf + o, (size_t)gs[i]->a * sizeof *buf);
		o += (size_t)gs[i]->a;
	}
	for (i = 0; i < nm; i++)
		if (ms[i]) {
			memcpy(ms[i]->a, buf + o, SZ(ms[i]) * sizeof *buf);
			o += SZ(ms[i]);
		}
	free(buf);
#undef GM
#undef SZ
}

static void
keep1(zmark m, struct fgab *g, struct zmat *m1, struct zmat *m2)
{
	struct fgab *gs[1];
	struct zmat *ms[2];

	gs[0] = g;
	ms[0] = m1;
	ms[1] = m2;
	keep_all(m, gs, 1, ms, 2);
}

/* ---- construction ---- */

void
ab_free(struct fgab *G)
{
	zm_free(&G->A);
	zm_free(&G->B);
	zm_free(&G->R);
	zsnf_free(&G->S);
	free(G->d);
	memset(G, 0, sizeof *G);
}

/* G from generators of A and B in Z^n (both consumed); 0 if B is not in A */
static int
build(struct fgab *G, int n, struct zmat *Ag, struct zmat *Bg)
{
	struct fgab T;
	int i;

	memset(&T, 0, sizeof T);
	T.n = n;
	if (Ag->r != n || Bg->r != n)
		fail("internal: generators in Z^%d and Z^%d for a group in Z^%d",
		    Ag->r, Bg->r, n);
	zm_image(Ag, &T.A);
	zm_image(Bg, &T.B);
	zm_free(Ag);
	zm_free(Bg);
	T.a = T.A.c;
	T.b = T.B.c;
	if (!lat_coords_mat(&T.A, &T.B, &T.R)) {
		zm_free(&T.A);
		zm_free(&T.B);
		return 0;
	}
	zm_snf(&T.R, &T.S);
	T.d = xcalloc((size_t)T.a + 1, sizeof *T.d);
	for (i = 0; i < T.a; i++)
		T.d[i] = i < T.S.rank ? ZM(&T.S.D, i, i) : 0;
	for (i = 0; i < T.a && z_is_one(T.d[i]); i++)
		;
	T.ntriv = i;
	T.ninv = T.a - T.ntriv;
	T.rank = T.a - T.S.rank;
	ab_free(G);
	*G = T;
	return 1;
}

void
ab_present(struct fgab *G, const struct zmat *rel)
{
	struct zmat A, B;
	zmark m = z_mark();

	zm_identity(&A, rel->r);
	zm_copy(&B, rel);
	build(G, rel->r, &A, &B);
	keep1(m, G, NULL, NULL);
}

int
ab_subquotient(struct fgab *G, int n, const struct zmat *A,
    const struct zmat *B)
{
	struct zmat a, b;
	zmark m = z_mark();
	int ok;

	if (A)
		zm_copy(&a, A);
	else
		zm_identity(&a, n);
	if (B)
		zm_copy(&b, B);
	else
		zm_init(&b, n, 0);
	ok = build(G, n, &a, &b);
	if (ok)
		keep1(m, G, NULL, NULL);
	else
		z_release(m);
	return ok;
}

void
ab_cyclic(struct fgab *G, zint m)
{
	struct zmat r;

	zm_init(&r, 1, 1);
	ZM(&r, 0, 0) = z_abs(m);
	ab_present(G, &r);
	zm_free(&r);
}

void
ab_free_abelian(struct fgab *G, int rk)
{
	struct zmat r;

	zm_init(&r, rk, 0);
	ab_present(G, &r);
	zm_free(&r);
}

void
ab_from_invariants(struct fgab *G, const zint *d, int k)
{
	struct zmat r;
	int i;

	zm_init(&r, k, k);
	for (i = 0; i < k; i++)
		ZM(&r, i, i) = z_abs(d[i]);
	ab_present(G, &r);
	zm_free(&r);
}

void
ab_copy(struct fgab *dst, const struct fgab *src)
{
	struct fgab T = *src;

	zm_copy(&T.A, &src->A);
	zm_copy(&T.B, &src->B);
	zm_copy(&T.R, &src->R);
	zm_copy(&T.S.D, &src->S.D);
	zm_copy(&T.S.U, &src->S.U);
	zm_copy(&T.S.Ui, &src->S.Ui);
	zm_copy(&T.S.V, &src->S.V);
	zm_copy(&T.S.Vi, &src->S.Vi);
	T.d = xcalloc((size_t)src->a + 1, sizeof *T.d);
	memcpy(T.d, src->d, (size_t)src->a * sizeof *T.d);
	if (dst != src)
		ab_free(dst);
	*dst = T;
}

/* "Z/4 + Z/6", "Z^2 (+) Z/3", "(Z/2)^3", "Z/2^3", "Z", "0" */
int
ab_parse(struct fgab *G, const char *s, const char **end)
{
	zint *d = NULL;
	int nd = 0, cap = 0, ok = 1;
	const char *p = s;

#define PUSH(v) do { \
		if (nd == cap) { \
			cap = cap ? cap * 2 : 8; \
			d = xrealloc(d, (size_t)cap * sizeof *d); \
		} \
		d[nd++] = (v); \
	} while (0)
	for (;;) {
		zint m = 0, e = 0;
		int paren = 0, rep = 1;
		const char *q;
		while (*p == ' ')
			p++;
		if (*p == '0' && !(p[1] >= '0' && p[1] <= '9')) {
			p++;
			goto sep;
		}
		if (*p == '(') {
			paren = 1;
			p++;
			while (*p == ' ')
				p++;
		}
		if (*p != 'Z') {
			ok = 0;
			break;
		}
		p++;
		while (*p == ' ')
			p++;
		if (*p == '/' || *p == '_') {
			p++;
			while (*p == ' ')
				p++;
			if (*p == '(') {
				if (!z_parse(p + 1, &m, &q)) {
					ok = 0;
					break;
				}
				p = q;
				while (*p == ' ')
					p++;
				if (*p == '^') {
					if (!z_parse(p + 1, &e, &q)) {
						ok = 0;
						break;
					}
					m = z_pow(m, (uint64_t)z_to_i64(e));
					p = q;
				}
				while (*p == ' ')
					p++;
				if (*p != ')') {
					ok = 0;
					break;
				}
				p++;
			} else {
				if (!z_parse(p, &m, &q)) {
					ok = 0;
					break;
				}
				p = q;
				if (*p == '^' && !paren) {
					if (!z_parse(p + 1, &e, &q)) {
						ok = 0;
						break;
					}
					if (z_sign(e) < 0 || !z_fits_i64(e) ||
					    z_to_i64(e) > 100000) {
						ok = 0;
						break;
					}
					m = z_pow(m, (uint64_t)z_to_i64(e));
					p = q;
				}
			}
			m = z_abs(m);
		} else
			m = 0;
		while (*p == ' ')
			p++;
		if (paren) {
			if (*p != ')') {
				ok = 0;
				break;
			}
			p++;
		}
		if (*p == '^') {
			zint r;
			if (!z_parse(p + 1, &r, &q) || z_sign(r) < 0 ||
			    z_cmp(r, z_from_i64(100000)) > 0) {
				ok = 0;
				break;
			}
			rep = (int)z_to_i64(r);
			p = q;
		}
		while (rep-- > 0)
			if (!z_is_one(m))
				PUSH(m);
sep:
		while (*p == ' ')
			p++;
		if (*p == '+')
			p++;
		else if (!strncmp(p, "(+)", 3))
			p += 3;
		else if (!strncmp(p, "\xe2\x8a\x95", 3))   /* U+2295 */
			p += 3;
		else if (!strncmp(p, "\xc3\x97", 2))       /* U+00D7 */
			p += 2;
		else if (*p == 'x' || *p == '*')
			p++;
		else
			break;
	}
#undef PUSH
	if (end)
		*end = p;
	if (!ok) {
		free(d);
		return 0;
	}
	ab_from_invariants(G, d, nd);
	free(d);
	return 1;
}

/* ---- invariants ---- */

int
ab_is_trivial(const struct fgab *G)
{
	return G->ninv == 0;
}

int
ab_is_finite(const struct fgab *G)
{
	return G->rank == 0;
}

int
ab_is_free(const struct fgab *G)
{
	return G->ninv == G->rank;
}

int
ab_is_cyclic(const struct fgab *G)
{
	return G->ninv <= 1;
}

zint
ab_order(const struct fgab *G)
{
	zint o = Z_ONE;
	int i;

	if (G->rank)
		return 0;
	for (i = G->ntriv; i < G->a; i++)
		o = z_mul(o, G->d[i]);
	return o;
}

zint
ab_exponent(const struct fgab *G)
{
	if (G->rank)
		return 0;
	return G->ninv ? G->d[G->a - 1] : Z_ONE;
}

int
ab_invariants(const struct fgab *G, zint *out)
{
	int i;

	for (i = 0; i < G->ninv; i++)
		out[i] = G->d[G->ntriv + i];
	return G->ninv;
}

int
ab_primary(const struct fgab *G, zint *out, int max)
{
	int i, j, k, n = 0, ok = 1;

	for (i = G->ntriv; i < G->a; i++) {
		zint p[64];
		int e[64], c, np;
		if (G->d[i] == 0)
			continue;
		np = z_factor(G->d[i], p, e, 64, &c);
		if (!c)
			ok = 0;
		for (k = 0; k < np; k++) {
			if (n >= max)
				fail("too many primary factors");
			out[n++] = z_pow(p[k], (uint64_t)e[k]);
		}
	}
	/* sort by prime, then by power: insertion sort keyed on (p, q) */
	for (i = 1; i < n; i++)
		for (j = i; j > 0; j--) {
			zint a = out[j - 1], b = out[j];
			zint pa[4], pb[4];
			int ea[4], eb[4], c;
			z_factor(a, pa, ea, 4, &c);
			z_factor(b, pb, eb, 4, &c);
			if (z_cmp(pa[0], pb[0]) > 0 || (z_eq(pa[0], pb[0]) &&
			    ea[0] > eb[0])) {
				out[j - 1] = b;
				out[j] = a;
			} else
				break;
		}
	for (i = 0; i < G->rank; i++) {
		if (n >= max)
			fail("too many primary factors");
		out[n++] = 0;
	}
	return ok ? n : -1;
}

static void
print_list(struct sbuf *b, const zint *d, int n)
{
	int i, free_r = 0, any = 0;

	for (i = 0; i < n; i++) {
		if (d[i] == 0) {
			free_r++;
			continue;
		}
		if (any)
			sb_puts(b, " + ");
		sb_puts(b, "Z/");
		z_print(b, d[i]);
		any = 1;
	}
	if (free_r) {
		if (any)
			sb_puts(b, " + ");
		sb_puts(b, "Z");
		if (free_r > 1)
			sb_printf(b, "^%d", free_r);
		any = 1;
	}
	if (!any)
		sb_puts(b, "0");
}

void
ab_print(struct sbuf *b, const struct fgab *G)
{
	print_list(b, G->d + G->ntriv, G->ninv);
}

void
ab_print_primary(struct sbuf *b, const struct fgab *G)
{
	zint *out = xcalloc((size_t)G->a * 64 + 1, sizeof *out);
	int n = ab_primary(G, out, G->a * 64);

	print_list(b, out, n < 0 ? 0 : n);
	if (n < 0)
		sb_puts(b, " (not fully factored)");
	free(out);
}

const char *
ab_str(const struct fgab *G)
{
	static struct sbuf rb[8];
	static int k;

	k = (k + 1) & 7;
	sb_clear(&rb[k]);
	ab_print(&rb[k], G);
	return sb_str(&rb[k]);
}

/* ---- elements ---- */

int
ab_coords(const struct fgab *G, const zint *x, zint *c)
{
	return lat_coords(&G->A, x, c);
}

void
ab_from_coords(const struct fgab *G, const zint *c, zint *x)
{
	zm_mul_vec(x, &G->A, c);
}

/* c (A-coordinates) in R Z^b?  w (b entries, may be NULL): c = R w */
static int
rel_member(const struct fgab *G, const zint *c, zint *w)
{
	zint *u = xcalloc((size_t)G->a + 1, sizeof *u);
	zint *t = xcalloc((size_t)G->b + 1, sizeof *t);
	int i, ok = 1;

	zm_mul_vec(u, &G->S.U, c);
	for (i = 0; i < G->a && ok; i++) {
		if (i < G->S.rank) {
			zint q, r;
			z_divmod_trunc(u[i], ZM(&G->S.D, i, i), &q, &r);
			if (r != 0)
				ok = 0;
			t[i] = q;
		} else if (u[i] != 0)
			ok = 0;
	}
	if (ok && w)
		zm_mul_vec(w, &G->S.V, t);
	free(u);
	free(t);
	return ok;
}

static void
canon_of_coords(const struct fgab *G, const zint *c, zint *y)
{
	zint *u = xcalloc((size_t)G->a + 1, sizeof *u);
	int i;

	zm_mul_vec(u, &G->S.U, c);
	for (i = 0; i < G->ninv; i++) {
		zint d = G->d[G->ntriv + i];
		y[i] = d == 0 ? u[G->ntriv + i] : z_mod(u[G->ntriv + i], d);
	}
	free(u);
}

int
ab_canon(const struct fgab *G, const zint *x, zint *y)
{
	zint *c = xcalloc((size_t)G->a + 1, sizeof *c);
	int ok = lat_coords(&G->A, x, c);

	if (ok)
		canon_of_coords(G, c, y);
	free(c);
	return ok;
}

void
ab_from_canon(const struct fgab *G, const zint *y, zint *x)
{
	zint *u = xcalloc((size_t)G->a + 1, sizeof *u);
	zint *c = xcalloc((size_t)G->a + 1, sizeof *c);
	int i;

	for (i = 0; i < G->ninv; i++)
		u[G->ntriv + i] = y[i];
	zm_mul_vec(c, &G->S.Ui, u);
	zm_mul_vec(x, &G->A, c);
	free(u);
	free(c);
}

void
ab_gen(const struct fgab *G, int i, zint *x)
{
	zint *c = xcalloc((size_t)G->a + 1, sizeof *c);

	if (i < 0 || i >= G->ninv)
		fail("internal: generator %d of %d", i, G->ninv);
	zm_col(c, &G->S.Ui, G->ntriv + i);
	zm_mul_vec(x, &G->A, c);
	free(c);
}

static void
need_coords(const struct fgab *G, const zint *x, zint *c)
{
	if (!lat_coords(&G->A, x, c))
		fail("the vector is not an element of the group (not in its "
		    "generator lattice)");
}

int
ab_elem_is_zero(const struct fgab *G, const zint *x)
{
	zint *c = xcalloc((size_t)G->a + 1, sizeof *c);
	int z;

	need_coords(G, x, c);
	z = rel_member(G, c, NULL);
	free(c);
	return z;
}

int
ab_elem_eq(const struct fgab *G, const zint *x, const zint *y)
{
	zint *t = xcalloc((size_t)G->n + 1, sizeof *t);
	int i, r;

	for (i = 0; i < G->n; i++)
		t[i] = z_sub(x[i], y[i]);
	r = ab_elem_is_zero(G, t);
	free(t);
	return r;
}

void
ab_reduce(const struct fgab *G, zint *x)
{
	int i, j;

	for (j = 0; j < G->B.c; j++) {
		int p = piv_row(&G->B, j);
		zint q = z_div_floor(x[p], ZM(&G->B, p, j));
		if (q != 0)
			for (i = p; i < G->n; i++)
				if (ZM(&G->B, i, j) != 0)
					x[i] = z_submul(x[i], q, ZM(&G->B, i, j));
	}
}

int
ab_elem_zero_witness(const struct fgab *G, const zint *x, zint *w)
{
	return lat_coords(&G->B, x, w);
}

zint
ab_elem_order(const struct fgab *G, const zint *x)
{
	zint *y = xcalloc((size_t)G->ninv + 1, sizeof *y);
	zint o = Z_ONE;
	int i;

	if (!ab_canon(G, x, y))
		fail("the vector is not an element of the group");
	for (i = 0; i < G->ninv; i++) {
		zint d = G->d[G->ntriv + i];
		if (y[i] == 0)
			continue;
		if (d == 0) {
			o = 0;
			break;
		}
		o = z_lcm(o, z_divexact(d, z_gcd(d, y[i])));
	}
	free(y);
	/* certificate: o x = 0 */
	if (o != 0) {
		zint *t = xcalloc((size_t)G->n + 1, sizeof *t);
		for (i = 0; i < G->n; i++)
			t[i] = z_mul(o, x[i]);
		if (!ab_elem_is_zero(G, t))
			fail("internal: element order check failed");
		free(t);
	}
	return o;
}

/* ---- subgroups ---- */

static void
same_ambient(const struct fgab *S, const struct fgab *T)
{
	if (S->n != T->n)
		fail("internal: subgroups of Z^%d and Z^%d", S->n, T->n);
}

void
ab_sub(struct fgab *S, const struct fgab *G, const struct zmat *gens)
{
	struct zmat a, b, c;
	zmark m = z_mark();

	if (gens->r != G->n)
		fail("subgroup: generators have %d entries, the group lives in "
		    "Z^%d", gens->r, G->n);
	memset(&c, 0, sizeof c);
	if (!lat_coords_mat(&G->A, gens, &c))
		fail("subgroup: a generator is not an element of the group");
	zm_free(&c);
	zm_init(&a, 0, 0);
	zm_hcat(&a, gens, &G->B);
	zm_copy(&b, &G->B);
	build(S, G->n, &a, &b);
	keep1(m, S, NULL, NULL);
}

int
ab_member(const struct fgab *S, const zint *x, zint *w)
{
	zint *c = xcalloc((size_t)S->a + 1, sizeof *c);
	int ok = lat_coords(&S->A, x, c);

	if (ok && w)
		memcpy(w, c, (size_t)S->a * sizeof *c);
	free(c);
	return ok;
}

int
ab_sub_le(const struct fgab *S, const struct fgab *T)
{
	struct zmat c;
	int ok;

	same_ambient(S, T);
	memset(&c, 0, sizeof c);
	ok = lat_coords_mat(&T->A, &S->A, &c);
	zm_free(&c);
	return ok;
}

int
ab_sub_eq(const struct fgab *S, const struct fgab *T)
{
	same_ambient(S, T);
	/* both A in column HNF: the canonical basis decides */
	return zm_eq(&S->A, &T->A) && zm_eq(&S->B, &T->B);
}

void
ab_sub_sum(struct fgab *r, const struct fgab *S, const struct fgab *T)
{
	struct zmat a, b;
	zmark m = z_mark();

	same_ambient(S, T);
	zm_init(&a, 0, 0);
	zm_hcat(&a, &S->A, &T->A);
	zm_init(&b, 0, 0);
	zm_lattice_meet(&b, &S->B, &T->B);
	build(r, S->n, &a, &b);
	keep1(m, r, NULL, NULL);
}

void
ab_sub_meet(struct fgab *r, const struct fgab *S, const struct fgab *T)
{
	struct zmat a, b;
	zmark m = z_mark();

	same_ambient(S, T);
	zm_init(&a, 0, 0);
	zm_init(&b, 0, 0);
	zm_lattice_meet(&a, &S->A, &T->A);
	zm_lattice_meet(&b, &S->B, &T->B);
	if (!build(r, S->n, &a, &b))
		fail("internal: subgroups with different relations");
	keep1(m, r, NULL, NULL);
}

void
ab_mulsub(struct fgab *r, const struct fgab *G, zint k)
{
	struct zmat a, ka, b;
	zmark m = z_mark();

	zm_init(&ka, 0, 0);
	zm_scale(&ka, &G->A, k);
	zm_init(&a, 0, 0);
	zm_hcat(&a, &ka, &G->B);
	zm_copy(&b, &G->B);
	build(r, G->n, &a, &b);
	zm_free(&ka);
	keep1(m, r, NULL, NULL);
}

void
ab_torsion_m(struct fgab *r, const struct fgab *G, zint k)
{
	struct zmat mi, cat, ker, top, gens, a, b;
	zmark m = z_mark();

	/* {c : k c = R w} = first block of ker [k I | -R] */
	zm_identity(&mi, G->a);
	zm_scale(&mi, &mi, k);
	{
		struct zmat nr;
		zm_init(&nr, 0, 0);
		zm_scale(&nr, &G->R, z_from_i64(-1));
		zm_init(&cat, 0, 0);
		zm_hcat(&cat, &mi, &nr);
		zm_free(&nr);
	}
	memset(&ker, 0, sizeof ker);
	zm_kernel(&cat, &ker);
	zm_init(&top, 0, 0);
	zm_sub_rows(&top, &ker, 0, G->a);
	zm_init(&gens, 0, 0);
	zm_mul(&gens, &G->A, &top);
	zm_init(&a, 0, 0);
	zm_hcat(&a, &gens, &G->B);
	zm_copy(&b, &G->B);
	build(r, G->n, &a, &b);
	zm_free(&mi);
	zm_free(&cat);
	zm_free(&ker);
	zm_free(&top);
	zm_free(&gens);
	keep1(m, r, NULL, NULL);
}

/* the subgroup generated by B and the columns A Ui[:, i] * mult[i] for the
 * canonical generators with mult[i] != 0 */
static void
sub_of_gens(struct fgab *r, const struct fgab *G, const zint *mult)
{
	struct zmat g, a, b;
	zint *x = xcalloc((size_t)G->n + 1, sizeof *x);
	int i, j, k = 0;

	for (i = 0; i < G->ninv; i++)
		if (mult[i] != 0)
			k++;
	zm_init(&g, G->n, k);
	for (i = 0, k = 0; i < G->ninv; i++) {
		if (mult[i] == 0)
			continue;
		ab_gen(G, i, x);
		for (j = 0; j < G->n; j++)
			ZM(&g, j, k) = z_mul(mult[i], x[j]);
		k++;
	}
	zm_init(&a, 0, 0);
	zm_hcat(&a, &g, &G->B);
	zm_copy(&b, &G->B);
	build(r, G->n, &a, &b);
	zm_free(&g);
	free(x);
}

void
ab_torsion(struct fgab *r, const struct fgab *G)
{
	zint *mult = xcalloc((size_t)G->ninv + 1, sizeof *mult);
	zmark m = z_mark();
	int i;

	for (i = 0; i < G->ninv; i++)
		mult[i] = G->d[G->ntriv + i] != 0 ? Z_ONE : 0;
	sub_of_gens(r, G, mult);
	free(mult);
	keep1(m, r, NULL, NULL);
}

void
ab_ptorsion(struct fgab *r, const struct fgab *G, zint p)
{
	zint *mult = xcalloc((size_t)G->ninv + 1, sizeof *mult);
	zmark m = z_mark();
	int i;

	if (!z_isprime(p))
		fail("T_p: %s is not a prime", z_str(p));
	for (i = 0; i < G->ninv; i++) {
		zint d = G->d[G->ntriv + i], rest;
		if (d == 0 || !z_divides(p, d))
			continue;
		z_val(d, p, &rest);
		mult[i] = rest;
	}
	sub_of_gens(r, G, mult);
	free(mult);
	keep1(m, r, NULL, NULL);
}

void
ab_quotient(struct fgab *r, const struct fgab *G, const struct fgab *S)
{
	struct zmat a, b;
	zmark m = z_mark();

	same_ambient(G, S);
	zm_copy(&a, &G->A);
	zm_copy(&b, &S->A);
	if (!build(r, G->n, &a, &b))
		fail("quotient: the subgroup is not inside the group");
	keep1(m, r, NULL, NULL);
}

void
ab_inclusion(struct zmat *M, const struct fgab *S, const struct fgab *G)
{
	same_ambient(S, G);
	if (!lat_coords_mat(&G->A, &S->A, M))
		fail("inclusion: not a subgroup");
}

/* ---- homomorphisms ---- */

static void
hom_size(const struct fgab *G, const struct fgab *H, const struct zmat *M)
{
	if (M->r != H->a || M->c != G->a)
		fail("a homomorphism from a group with %d generators to one "
		    "with %d is a %dx%d coordinate matrix, not %dx%d", G->a,
		    H->a, H->a, G->a, M->r, M->c);
}

int
ab_hom_check(const struct fgab *G, const struct fgab *H,
    const struct zmat *M)
{
	struct zmat mr;
	zint *c = xcalloc((size_t)H->a + 1, sizeof *c);
	zmark m = z_mark();
	int j, ok = 1;

	hom_size(G, H, M);
	zm_init(&mr, 0, 0);
	zm_mul(&mr, M, &G->R);
	for (j = 0; j < mr.c && ok; j++) {
		zm_col(c, &mr, j);
		if (!rel_member(H, c, NULL))
			ok = 0;
	}
	zm_free(&mr);
	free(c);
	z_release(m);
	return ok;
}

int
ab_hom_from_images(struct zmat *M, const struct fgab *G,
    const struct fgab *H, const struct zmat *imgs)
{
	struct zmat t;

	if (imgs->r != H->n || imgs->c != G->a)
		fail("homomorphism: %d images in Z^%d expected, got %dx%d",
		    G->a, H->n, imgs->r, imgs->c);
	memset(&t, 0, sizeof t);
	if (!lat_coords_mat(&H->A, imgs, &t))
		return 0;
	if (!ab_hom_check(G, H, &t)) {
		zm_free(&t);
		return 0;
	}
	zm_move(M, &t);
	return 1;
}

int
ab_hom_induced(struct zmat *M, const struct fgab *G, const struct fgab *H,
    const struct zmat *F)
{
	struct zmat imgs;
	int ok;

	if (F->r != H->n || F->c != G->n)
		fail("induced map: Z^%d -> Z^%d needs a %dx%d matrix, not %dx%d",
		    G->n, H->n, H->n, G->n, F->r, F->c);
	zm_init(&imgs, 0, 0);
	zm_mul(&imgs, F, &G->A);
	ok = ab_hom_from_images(M, G, H, &imgs);
	zm_free(&imgs);
	return ok;
}

int
ab_hom_from_gens(struct zmat *M, const struct fgab *G, const struct fgab *H,
    const struct zmat *imgs)
{
	struct zmat hc, ur, t;
	int ok;

	if (imgs->r != H->n || imgs->c != G->ninv)
		fail("homomorphism: images of the %d generators in Z^%d expected",
		    G->ninv, H->n);
	memset(&hc, 0, sizeof hc);
	if (!lat_coords_mat(&H->A, imgs, &hc))
		return 0;
	zm_init(&ur, 0, 0);
	zm_sub_rows(&ur, &G->S.U, G->ntriv, G->a);
	zm_init(&t, 0, 0);
	zm_mul(&t, &hc, &ur);
	zm_free(&hc);
	zm_free(&ur);
	ok = ab_hom_check(G, H, &t);
	if (ok)
		zm_move(M, &t);
	else
		zm_free(&t);
	return ok;
}

void
ab_hom_apply(const struct fgab *G, const struct fgab *H,
    const struct zmat *M, const zint *x, zint *y)
{
	zint *c = xcalloc((size_t)G->a + 1, sizeof *c);
	zint *e = xcalloc((size_t)H->a + 1, sizeof *e);

	hom_size(G, H, M);
	need_coords(G, x, c);
	zm_mul_vec(e, M, c);
	zm_mul_vec(y, &H->A, e);
	free(c);
	free(e);
}

void
ab_hom_compose(struct zmat *r, const struct zmat *g, const struct zmat *f)
{
	zm_mul(r, g, f);
}

int
ab_hom_is_zero(const struct fgab *G, const struct fgab *H,
    const struct zmat *M)
{
	zint *c = xcalloc((size_t)H->a + 1, sizeof *c);
	int j, ok = 1;

	hom_size(G, H, M);
	for (j = 0; j < M->c && ok; j++) {
		zm_col(c, M, j);
		ok = rel_member(H, c, NULL);
	}
	free(c);
	return ok;
}

int
ab_hom_eq(const struct fgab *G, const struct fgab *H,
    const struct zmat *M1, const struct zmat *M2)
{
	struct zmat d;
	zmark m = z_mark();
	int ok;

	zm_init(&d, 0, 0);
	zm_sub(&d, M1, M2);
	ok = ab_hom_is_zero(G, H, &d);
	zm_free(&d);
	z_release(m);
	return ok;
}

void
ab_hom_identity(struct zmat *M, const struct fgab *G)
{
	struct zmat t;

	zm_identity(&t, G->a);
	zm_move(M, &t);
}

void
ab_hom_zero(struct zmat *M, const struct fgab *G, const struct fgab *H)
{
	struct zmat t;

	zm_init(&t, H->a, G->a);
	zm_move(M, &t);
}

/* the first n rows of a basis of ker [X | Y] */
static void
ker_top(struct zmat *r, const struct zmat *X, const struct zmat *Y)
{
	struct zmat cat, k;

	zm_init(&cat, 0, 0);
	zm_hcat(&cat, X, Y);
	memset(&k, 0, sizeof k);
	zm_kernel(&cat, &k);
	zm_sub_rows(r, &k, 0, X->c);
	zm_free(&cat);
	zm_free(&k);
}

void
ab_kernel(struct fgab *K, struct zmat *inc, const struct fgab *G,
    const struct fgab *H, const struct zmat *M)
{
	struct zmat L, gens, a, b;
	zmark m = z_mark();

	hom_size(G, H, M);
	zm_init(&L, 0, 0);
	ker_top(&L, M, &H->R);
	zm_init(&gens, 0, 0);
	zm_mul(&gens, &G->A, &L);
	zm_init(&a, 0, 0);
	zm_hcat(&a, &gens, &G->B);
	zm_copy(&b, &G->B);
	build(K, G->n, &a, &b);
	zm_free(&L);
	zm_free(&gens);
	if (inc)
		ab_inclusion(inc, K, G);
	keep1(m, K, inc, NULL);
}

void
ab_image(struct fgab *I, struct zmat *inc, const struct fgab *G,
    const struct fgab *H, const struct zmat *M)
{
	struct zmat gens, a, b;
	zmark m = z_mark();

	hom_size(G, H, M);
	zm_init(&gens, 0, 0);
	zm_mul(&gens, &H->A, M);
	zm_init(&a, 0, 0);
	zm_hcat(&a, &gens, &H->B);
	zm_copy(&b, &H->B);
	build(I, H->n, &a, &b);
	zm_free(&gens);
	if (inc)
		ab_inclusion(inc, I, H);
	keep1(m, I, inc, NULL);
}

void
ab_cokernel(struct fgab *C, struct zmat *proj, const struct fgab *G,
    const struct fgab *H, const struct zmat *M)
{
	struct zmat gens, a, b;
	zmark m = z_mark();

	hom_size(G, H, M);
	zm_init(&gens, 0, 0);
	zm_mul(&gens, &H->A, M);
	zm_init(&b, 0, 0);
	zm_hcat(&b, &gens, &H->B);
	zm_copy(&a, &H->A);
	build(C, H->n, &a, &b);
	zm_free(&gens);
	if (proj && !lat_coords_mat(&C->A, &H->A, proj))
		fail("internal: cokernel projection");
	keep1(m, C, proj, NULL);
}

int
ab_hom_injective(const struct fgab *G, const struct fgab *H,
    const struct zmat *M)
{
	struct fgab K;
	int r;

	memset(&K, 0, sizeof K);
	ab_kernel(&K, NULL, G, H, M);
	r = ab_is_trivial(&K);
	ab_free(&K);
	return r;
}

int
ab_hom_surjective(const struct fgab *G, const struct fgab *H,
    const struct zmat *M)
{
	struct fgab C;
	int r;

	memset(&C, 0, sizeof C);
	ab_cokernel(&C, NULL, G, H, M);
	r = ab_is_trivial(&C);
	ab_free(&C);
	return r;
}

/* P (M.c x T.c) with M P = T (mod R_H), all columns with one Smith form;
 * 0 if some column has no preimage */
static int
preimages(const struct fgab *H, const struct zmat *M, const struct zmat *T,
    struct zmat *P)
{
	struct zmat cat, X;
	int ok;

	zm_init(&cat, 0, 0);
	zm_hcat(&cat, M, &H->R);
	memset(&X, 0, sizeof X);
	ok = zm_solve(&cat, T, &X);
	if (ok)
		zm_sub_rows(P, &X, 0, M->c);
	zm_free(&cat);
	zm_free(&X);
	return ok;
}

int
ab_hom_iso(const struct fgab *G, const struct fgab *H,
    const struct zmat *M, struct zmat *inv)
{
	struct zmat N, id, comp;
	zint *e, *c;
	zmark m = z_mark();
	int j, ok = 1;

	if (!ab_hom_check(G, H, M))
		return 0;
	if (!ab_hom_injective(G, H, M) || !ab_hom_surjective(G, H, M))
		return 0;
	(void)e;
	(void)c;
	(void)j;
	zm_identity(&id, H->a);
	zm_init(&N, 0, 0);
	ok = preimages(H, M, &id, &N);
	zm_free(&id);
	if (!ok)
		fail("internal: a bijective homomorphism without a preimage");
	/* the certificate: N well defined, N M = 1 on G, M N = 1 on H */
	zm_identity(&id, G->a);
	zm_init(&comp, 0, 0);
	zm_mul(&comp, &N, M);
	if (!ab_hom_check(H, G, &N) || !ab_hom_eq(G, G, &comp, &id))
		fail("internal: inverse check failed");
	zm_free(&id);
	zm_identity(&id, H->a);
	zm_mul(&comp, M, &N);
	if (!ab_hom_eq(H, H, &comp, &id))
		fail("internal: inverse check failed");
	zm_free(&id);
	zm_free(&comp);
	if (inv) {
		zm_move(inv, &N);
		keep1(m, NULL, inv, NULL);
	} else {
		zm_free(&N);
		z_release(m);
	}
	return 1;
}

/* the map sending G's canonical generators to H's, in coordinates */
static void
canon_map(struct zmat *M, const struct fgab *G, const struct fgab *H)
{
	struct zmat hi, ur;

	zm_init(&hi, 0, 0);
	zm_sub_cols(&hi, &H->S.Ui, H->ntriv, H->a);
	zm_init(&ur, 0, 0);
	zm_sub_rows(&ur, &G->S.U, G->ntriv, G->a);
	zm_mul(M, &hi, &ur);
	zm_free(&hi);
	zm_free(&ur);
}

int
ab_isomorphic(const struct fgab *G, const struct fgab *H,
    struct zmat *phi, struct zmat *psi)
{
	struct zmat f, g, comp, id;
	zmark m = z_mark();
	int i;

	if (G->ninv != H->ninv)
		return 0;
	for (i = 0; i < G->ninv; i++)
		if (!z_eq(G->d[G->ntriv + i], H->d[H->ntriv + i]))
			return 0;
	zm_init(&f, 0, 0);
	zm_init(&g, 0, 0);
	canon_map(&f, G, H);
	canon_map(&g, H, G);
	if (!ab_hom_check(G, H, &f) || !ab_hom_check(H, G, &g))
		fail("internal: isomorphism not well defined");
	zm_init(&comp, 0, 0);
	zm_mul(&comp, &g, &f);
	zm_identity(&id, G->a);
	if (!ab_hom_eq(G, G, &comp, &id))
		fail("internal: isomorphism check failed");
	zm_free(&id);
	zm_mul(&comp, &f, &g);
	zm_identity(&id, H->a);
	if (!ab_hom_eq(H, H, &comp, &id))
		fail("internal: isomorphism check failed");
	zm_free(&id);
	zm_free(&comp);
	if (phi)
		zm_move(phi, &f);
	else
		zm_free(&f);
	if (psi)
		zm_move(psi, &g);
	else
		zm_free(&g);
	keep1(m, NULL, phi, psi);
	return 1;
}

/* ---- constructions ---- */

void
ab_direct_sum(struct fgab *r, const struct fgab *G, const struct fgab *H,
    struct zmat *i1, struct zmat *i2, struct zmat *p1, struct zmat *p2)
{
	struct zmat a, b, t, top;
	struct fgab T;
	zmark m = z_mark();
	struct fgab *gs[1];
	struct zmat *ms[4];

	memset(&T, 0, sizeof T);
	zm_init(&a, 0, 0);
	zm_init(&b, 0, 0);
	zm_block_diag(&a, &G->A, &H->A);
	zm_block_diag(&b, &G->B, &H->B);
	build(&T, G->n + H->n, &a, &b);
	if (i1 || i2) {
		zm_init(&t, 0, 0);
		if (i1) {
			struct zmat z;
			zm_init(&z, H->n, G->a);
			zm_vcat(&t, &G->A, &z);
			if (!lat_coords_mat(&T.A, &t, i1))
				fail("internal: direct sum inclusion");
			zm_free(&z);
		}
		if (i2) {
			struct zmat z;
			zm_init(&z, G->n, H->a);
			zm_vcat(&t, &z, &H->A);
			if (!lat_coords_mat(&T.A, &t, i2))
				fail("internal: direct sum inclusion");
			zm_free(&z);
		}
		zm_free(&t);
	}
	if (p1) {
		zm_init(&top, 0, 0);
		zm_sub_rows(&top, &T.A, 0, G->n);
		if (!lat_coords_mat(&G->A, &top, p1))
			fail("internal: direct sum projection");
		zm_free(&top);
	}
	if (p2) {
		zm_init(&top, 0, 0);
		zm_sub_rows(&top, &T.A, G->n, G->n + H->n);
		if (!lat_coords_mat(&H->A, &top, p2))
			fail("internal: direct sum projection");
		zm_free(&top);
	}
	ab_free(r);
	*r = T;
	gs[0] = r;
	ms[0] = i1;
	ms[1] = i2;
	ms[2] = p1;
	ms[3] = p2;
	keep_all(m, gs, 1, ms, 4);
}

void
ab_hom_group(struct fgab *r, const struct fgab *G, const struct fgab *H)
{
	int ha = H->a, ga = G->a, gb = G->b, hb = H->b;
	int N = ha * ga, i, j, k, l;
	struct zmat phi, L, gens, B;
	zmark m = z_mark();

	/* (M, W) -> M R_G - R_H W, rows (i, k), columns M[i][j] then W[l][k] */
	zm_init(&phi, ha * gb, N + hb * gb);
	for (i = 0; i < ha; i++)
		for (k = 0; k < gb; k++) {
			int row = i * gb + k;
			for (j = 0; j < ga; j++)
				ZM(&phi, row, i * ga + j) = ZM(&G->R, j, k);
			for (l = 0; l < hb; l++)
				ZM(&phi, row, N + l * gb + k) = z_neg(ZM(&H->R, i,
				    l));
		}
	memset(&L, 0, sizeof L);
	zm_kernel(&phi, &L);
	zm_init(&gens, 0, 0);
	zm_sub_rows(&gens, &L, 0, N);
	/* the zero maps: M = R_H e_l e_j^T */
	zm_init(&B, N, hb * ga);
	for (l = 0; l < hb; l++)
		for (j = 0; j < ga; j++)
			for (i = 0; i < ha; i++)
				ZM(&B, i * ga + j, l * ga + j) = ZM(&H->R, i, l);
	{
		struct zmat a;
		zm_init(&a, 0, 0);
		zm_hcat(&a, &gens, &B);
		if (!build(r, N, &a, &B))
			fail("internal: Hom group");
	}
	zm_free(&phi);
	zm_free(&L);
	zm_free(&gens);
	keep1(m, r, NULL, NULL);
}

void
ab_hom_unflatten(struct zmat *M, const struct fgab *G, const struct fgab *H,
    const zint *x)
{
	struct zmat t;
	int i, j;

	zm_init(&t, H->a, G->a);
	for (i = 0; i < H->a; i++)
		for (j = 0; j < G->a; j++)
			ZM(&t, i, j) = x[i * G->a + j];
	zm_move(M, &t);
}

void
ab_tensor(struct fgab *r, const struct fgab *G, const struct fgab *H)
{
	int ga = G->a, ha = H->a, i, j, k, c = 0;
	struct zmat rel;
	zmark m = z_mark();

	zm_init(&rel, ga * ha, G->b * ha + H->b * ga);
	for (k = 0; k < G->b; k++)
		for (j = 0; j < ha; j++, c++)
			for (i = 0; i < ga; i++)
				ZM(&rel, i * ha + j, c) = ZM(&G->R, i, k);
	for (k = 0; k < H->b; k++)
		for (i = 0; i < ga; i++, c++)
			for (j = 0; j < ha; j++)
				ZM(&rel, i * ha + j, c) = ZM(&H->R, j, k);
	ab_present(r, &rel);
	zm_free(&rel);
	keep1(m, r, NULL, NULL);
}

/* H^k as a subquotient of Z^{k H.n} */
static void
power(struct fgab *r, const struct fgab *H, int k)
{
	struct zmat a, b, t;
	int i;

	zm_init(&a, 0, 0);
	zm_init(&b, 0, 0);
	for (i = 0; i < k; i++) {
		zm_init(&t, 0, 0);
		zm_block_diag(&t, &a, &H->A);
		zm_move(&a, &t);
		zm_init(&t, 0, 0);
		zm_block_diag(&t, &b, &H->B);
		zm_move(&b, &t);
	}
	if (k == 0) {
		zm_free(&a);
		zm_free(&b);
		zm_init(&a, 0, 0);
		zm_init(&b, 0, 0);
	}
	build(r, k * H->n, &a, &b);
}

/* the ambient map R (x) 1 : Z^{cols R.c * n} -> Z^{R.r * n} */
static void
kron_id(struct zmat *F, const struct zmat *R, int n)
{
	int i, k, t;

	zm_init(F, R->r * n, R->c * n);
	for (i = 0; i < R->r; i++)
		for (k = 0; k < R->c; k++)
			if (ZM(R, i, k) != 0)
				for (t = 0; t < n; t++)
					ZM(F, i * n + t, k * n + t) = ZM(R, i, k);
}

void
ab_tor(struct fgab *r, const struct fgab *G, const struct fgab *H)
{
	struct fgab Hb, Ha;
	struct zmat F, M;
	zmark m = z_mark();

	memset(&Hb, 0, sizeof Hb);
	memset(&Ha, 0, sizeof Ha);
	power(&Hb, H, G->b);
	power(&Ha, H, G->a);
	kron_id(&F, &G->R, H->n);
	memset(&M, 0, sizeof M);
	if (!ab_hom_induced(&M, &Hb, &Ha, &F))
		fail("internal: Tor map");
	ab_kernel(r, NULL, &Hb, &Ha, &M);
	zm_free(&F);
	zm_free(&M);
	ab_free(&Hb);
	ab_free(&Ha);
	keep1(m, r, NULL, NULL);
}

void
ab_ext(struct fgab *r, const struct fgab *G, const struct fgab *H)
{
	struct fgab Hb, Ha;
	struct zmat F, M, Rt;
	zmark m = z_mark();

	memset(&Hb, 0, sizeof Hb);
	memset(&Ha, 0, sizeof Ha);
	power(&Hb, H, G->b);
	power(&Ha, H, G->a);
	zm_init(&Rt, 0, 0);
	zm_transpose(&Rt, &G->R);
	kron_id(&F, &Rt, H->n);
	memset(&M, 0, sizeof M);
	if (!ab_hom_induced(&M, &Ha, &Hb, &F))
		fail("internal: Ext map");
	ab_cokernel(r, NULL, &Ha, &Hb, &M);
	zm_free(&F);
	zm_free(&M);
	zm_free(&Rt);
	ab_free(&Hb);
	ab_free(&Ha);
	keep1(m, r, NULL, NULL);
}

/* ---- exactness ---- */

int
ab_exact_at(const struct fgab *G1, const struct fgab *G2,
    const struct fgab *G3, const struct zmat *f, const struct zmat *g,
    struct zmat *pre, const char **why)
{
	struct fgab K;
	struct zmat inc, gf, P;
	zint *col, *c;
	zmark m = z_mark();
	int j, ok = 1;
	const char *w = NULL;

	if (!ab_hom_check(G1, G2, f)) {
		w = "the first map is not well defined";
		ok = 0;
	} else if (!ab_hom_check(G2, G3, g)) {
		w = "the second map is not well defined";
		ok = 0;
	}
	if (ok) {
		zm_init(&gf, 0, 0);
		zm_mul(&gf, g, f);
		if (!ab_hom_is_zero(G1, G3, &gf)) {
			w = "the composite is not zero (im f is not inside ker g)";
			ok = 0;
		}
		zm_free(&gf);
	}
	if (!ok) {
		z_release(m);
		if (why)
			*why = w;
		return 0;
	}
	memset(&K, 0, sizeof K);
	memset(&inc, 0, sizeof inc);
	ab_kernel(&K, &inc, G2, G3, g);
	col = xcalloc((size_t)G2->a + 1, sizeof *col);
	c = xcalloc((size_t)G1->a + 1, sizeof *c);
	zm_init(&P, 0, 0);
	if (!preimages(G2, f, &inc, &P)) {
		ok = 0;
		w = "ker g is not inside im f";
	}
	if (ok) {
		/* the certificate: f(P) = inc modulo R_G2 */
		struct zmat fp, d;
		zm_init(&fp, 0, 0);
		zm_init(&d, 0, 0);
		zm_mul(&fp, f, &P);
		zm_sub(&d, &fp, &inc);
		for (j = 0; j < d.c; j++) {
			zm_col(col, &d, j);
			if (!rel_member(G2, col, NULL))
				fail("internal: exactness certificate failed");
		}
		zm_free(&fp);
		zm_free(&d);
	}
	free(col);
	free(c);
	zm_free(&inc);
	ab_free(&K);
	if (ok && pre) {
		zm_move(pre, &P);
		keep1(m, NULL, pre, NULL);
	} else {
		zm_free(&P);
		z_release(m);
	}
	if (!ok && why)
		*why = w;
	return ok;
}

int
ab_exact_seq(const struct fgab *G, int k, const struct zmat *f, int *bad)
{
	int i;

	if (bad)
		*bad = -1;
	for (i = 1; i + 1 < k; i++)
		if (!ab_exact_at(&G[i - 1], &G[i], &G[i + 1], &f[i - 1], &f[i],
		    NULL, NULL)) {
			if (bad)
				*bad = i;
			return 0;
		}
	return 1;
}
