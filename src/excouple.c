/*
 * excouple.c - exact couples, derived couples, spectral sequences, the
 * Bockstein spectral sequence, element tracing and the concrete check of
 * Proposition 0.1 of bockstein.pdf (homalg.h).
 *
 * A couple is a few parallel arrays indexed by degree - lo (D, E, i, j, k,
 * d); every page of a spectral sequence keeps its groups as subquotients
 * of the same ambient Z^g as page 1, so an element is the same vector on
 * every page.
 */
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "homalg.h"

static struct fgab zero_grp;
static int zero_made;

static const struct fgab *
zero(void)
{
	if (!zero_made) {
		ab_free_abelian(&zero_grp, 0);
		zero_made = 1;
	}
	return &zero_grp;
}

static int
inr(const struct hcouple *C, int n)
{
	return n >= C->lo && n <= C->hi;
}

const struct fgab *
hc_D(const struct hcouple *C, int n)
{
	return inr(C, n) ? &C->D[n - C->lo] : zero();
}

const struct fgab *
hc_E(const struct hcouple *C, int n)
{
	return inr(C, n) ? &C->E[n - C->lo] : zero();
}

void
hc_init(struct hcouple *C, int lo, int hi, int di, int dj, int dk)
{
	size_t len = (size_t)(hi - lo + 1);

	memset(C, 0, sizeof *C);
	C->lo = lo;
	C->hi = hi;
	C->di = di;
	C->dj = dj;
	C->dk = dk;
	C->page = 1;
	C->D = xcalloc(len, sizeof *C->D);
	C->E = xcalloc(len, sizeof *C->E);
	C->i = xcalloc(len, sizeof *C->i);
	C->j = xcalloc(len, sizeof *C->j);
	C->k = xcalloc(len, sizeof *C->k);
	C->d = xcalloc(len, sizeof *C->d);
}

void
hc_free(struct hcouple *C)
{
	int n, len = C->hi - C->lo + 1;

	if (C->D == NULL)
		return;
	for (n = 0; n < len; n++) {
		ab_free(&C->D[n]);
		ab_free(&C->E[n]);
		zm_free(&C->i[n]);
		zm_free(&C->j[n]);
		zm_free(&C->k[n]);
		zm_free(&C->d[n]);
	}
	free(C->D);
	free(C->E);
	free(C->i);
	free(C->j);
	free(C->k);
	free(C->d);
	memset(C, 0, sizeof *C);
}

/* d = j k on every degree (k: E^n -> D^{n+dk}, then j) */
static void
make_d(struct hcouple *C)
{
	int n;

	for (n = C->lo; n <= C->hi; n++) {
		int m = n + C->dk;
		struct zmat *d = &C->d[n - C->lo];
		zm_free(d);
		if (!inr(C, m) || !inr(C, m + C->dj))
			zm_init(d, hc_E(C, m + C->dj)->a, hc_E(C, n)->a);
		else
			ab_hom_compose(d, &C->j[m - C->lo], &C->k[n - C->lo]);
	}
}

/* exactness of X -f-> Y -g-> Z (the maps may be NULL: from/to zero) */
static void
exact3(const struct fgab *X, const struct fgab *Y, const struct fgab *Z,
    const struct zmat *f, const struct zmat *g, const char *what, int n,
    int page)
{
	struct zmat f0 = {0}, g0 = {0};
	const char *why = NULL;

	if (f == NULL) {
		zm_init(&f0, Y->a, X->a);
		f = &f0;
	}
	if (g == NULL) {
		zm_init(&g0, Z->a, Y->a);
		g = &g0;
	}
	if (!ab_exact_at(X, Y, Z, f, g, NULL, &why))
		fail("internal: homalg: the couple (page %d) is not exact at "
		    "%s^%d: %s", page, what, n, why ? why : "");
	hal_checks[HK_EXACT]++;
	zm_free(&f0);
	zm_free(&g0);
}

void
hc_check(const struct hcouple *C)
{
	int n;

	for (n = C->lo; n <= C->hi; n++) {
		int a;
		/* at D^n: E^{n-dk} -k-> D^n -i-> D^{n+di} */
		a = n - C->dk;
		exact3(hc_E(C, a), hc_D(C, n), hc_D(C, n + C->di),
		    inr(C, a) ? &C->k[a - C->lo] : NULL, &C->i[n - C->lo],
		    "D", n, C->page);
		/* at D^n: D^{n-di} -i-> D^n -j-> E^{n+dj} */
		a = n - C->di;
		exact3(hc_D(C, a), hc_D(C, n), hc_E(C, n + C->dj),
		    inr(C, a) ? &C->i[a - C->lo] : NULL, &C->j[n - C->lo],
		    "D", n, C->page);
		/* at E^n: D^{n-dj} -j-> E^n -k-> D^{n+dk} */
		a = n - C->dj;
		exact3(hc_D(C, a), hc_E(C, n), hc_D(C, n + C->dk),
		    inr(C, a) ? &C->j[a - C->lo] : NULL, &C->k[n - C->lo],
		    "E", n, C->page);
	}
	/* d d = 0 */
	for (n = C->lo; n <= C->hi; n++) {
		int m = n + C->dj + C->dk;
		if (inr(C, m)) {
			struct zmat dd = {0};
			ab_hom_compose(&dd, &C->d[m - C->lo], &C->d[n - C->lo]);
			if (!ab_hom_is_zero(hc_E(C, n), hc_E(C, m + C->dj +
			    C->dk), &dd))
				fail("internal: homalg: d d != 0 on page %d",
				    C->page);
			zm_free(&dd);
			hal_checks[HK_DD]++;
		}
	}
}

/* the coordinate matrix of a map on S's A-basis given ambient images */
static void
from_images(struct zmat *M, const struct fgab *S, const struct fgab *T,
    const struct zmat *imgs, const char *what, int page)
{
	if (S->a == 0) {
		zm_init(M, T->a, 0);
		return;
	}
	if (!ab_hom_from_images(M, S, T, imgs))
		fail("internal: homalg: page %d: an image of %s is not in "
		    "its group", page, what);
	if (!ab_hom_check(S, T, M))
		fail("internal: homalg: page %d: %s is not well defined",
		    page, what);
	hal_checks[HK_WELLDEF]++;
}

/* apply a coordinate map to every A-basis column of S (as an element of
 * G, S <= G): imgs is T.n x S.a */
static void
apply_cols(struct zmat *imgs, const struct fgab *S, const struct fgab *G,
    const struct fgab *T, const struct zmat *M)
{
	zint *x = xcalloc((size_t)S->n + 1, sizeof *x);
	zint *y = xcalloc((size_t)T->n + 1, sizeof *y);
	int c, r;

	zm_init(imgs, T->n, S->a);
	for (c = 0; c < S->a; c++) {
		zm_col(x, &S->A, c);
		if (T->n > 0)
			ab_hom_apply(G, T, M, x, y);
		for (r = 0; r < T->n; r++)
			ZM(imgs, r, c) = y[r];
	}
	free(x);
	free(y);
}

void
hc_derive(struct hcouple *R, const struct hcouple *C)
{
	int n, lo = C->lo, hi = C->hi, dd = C->dj + C->dk;
	struct fgab *Im = xcalloc((size_t)(hi - lo + 1), sizeof *Im);

	hc_init(R, lo, hi, C->di, C->dj - C->di, C->dk);
	R->page = C->page + 1;
	/* D' = i(D), E' = ker d / im d */
	for (n = lo; n <= hi; n++) {
		int a = n - C->di, b = n - dd;
		struct fgab K = {0};
		if (inr(C, a))
			ab_image(&R->D[n - lo], NULL, hc_D(C, a), hc_D(C, n),
			    &C->i[a - lo]);
		else {
			struct zmat z = {0};
			zm_init(&z, C->D[n - lo].n, 0);
			ab_sub(&R->D[n - lo], &C->D[n - lo], &z);
			zm_free(&z);
		}
		ab_kernel(&K, NULL, hc_E(C, n), hc_E(C, n + dd),
		    &C->d[n - lo]);
		if (inr(C, b))
			ab_image(&Im[n - lo], NULL, hc_E(C, b), hc_E(C, n),
			    &C->d[b - lo]);
		else {
			struct zmat z = {0};
			zm_init(&z, C->E[n - lo].n, 0);
			ab_sub(&Im[n - lo], &C->E[n - lo], &z);
			zm_free(&z);
		}
		if (!ab_sub_le(&Im[n - lo], &K))
			fail("internal: homalg: page %d: im d is not in ker d "
			    "in degree %d", C->page, n);
		hal_checks[HK_DD]++;
		ab_quotient(&R->E[n - lo], &K, &Im[n - lo]);
		ab_free(&K);
	}
	for (n = lo; n <= hi; n++) {
		const struct fgab *Dn = &R->D[n - lo];
		struct zmat imgs = {0};
		int t;

		/* i' = i on D'^n */
		t = n + C->di;
		apply_cols(&imgs, Dn, hc_D(C, n), hc_D(C, t), &C->i[n - lo]);
		from_images(&R->i[n - lo], Dn, hc_D(R, t), &imgs, "i'",
		    R->page);
		zm_free(&imgs);

		/* j'(i x) = [j x]: x by solving i x = y on D'^n's basis */
		t = n - C->di + C->dj;
		if (!inr(C, n - C->di) || Dn->a == 0)
			zm_init(&R->j[n - lo], hc_E(R, t)->a, Dn->a);
		else {
			const struct fgab *Ds = hc_D(C, n - C->di);
			const struct fgab *Dt = hc_D(C, n);
			const struct zmat *Mi = &C->i[n - C->di - lo];
			struct zmat MR = {0}, X = {0}, Mj = {0};
			zint *y = xcalloc((size_t)Dt->n + 1, sizeof *y);
			zint *cy = xcalloc((size_t)Dt->a + 1, sizeof *cy);
			zint *sol = xcalloc((size_t)(Mi->c + Dt->b) + 1,
			    sizeof *sol);
			int c, r;

			if (Dt->R.c > 0)
				zm_hcat(&MR, Mi, &Dt->R);
			else
				zm_copy(&MR, Mi);
			zm_init(&X, Ds->n, Dn->a);
			for (c = 0; c < Dn->a; c++) {
				zint *x = xcalloc((size_t)Ds->n + 1, sizeof *x);
				zm_col(y, &Dn->A, c);
				if (!ab_coords(Dt, y, cy) ||
				    !zm_solve_vec(&MR, cy, sol))
					fail("internal: homalg: page %d: an "
					    "element of i(D) has no preimage",
					    R->page);
				ab_from_coords(Ds, sol, x);
				for (r = 0; r < Ds->n; r++)
					ZM(&X, r, c) = x[r];
				free(x);
			}
			/* j on those preimages, as elements of E^t */
			{
				zm_init(&Mj, hc_E(C, t)->n, Dn->a);
				for (c = 0; c < Dn->a; c++) {
					zint *x = xcalloc((size_t)Ds->n + 1,
					    sizeof *x);
					zint *e = xcalloc((size_t)hc_E(C, t)->n +
					    1, sizeof *e);
					zm_col(x, &X, c);
					if (hc_E(C, t)->n > 0)
						ab_hom_apply(Ds, hc_E(C, t),
						    &C->j[n - C->di - lo], x, e);
					for (r = 0; r < hc_E(C, t)->n; r++)
						ZM(&Mj, r, c) = e[r];
					free(x);
					free(e);
				}
			}
			from_images(&R->j[n - lo], Dn, hc_E(R, t), &Mj, "j'",
			    R->page);
			/* independent of the preimage: j(ker i) <= im d */
			{
				struct fgab Ki = {0}, Ji = {0};
				struct zmat ji = {0};
				ab_kernel(&Ki, NULL, Ds, Dt, Mi);
				apply_cols(&ji, &Ki, Ds, hc_E(C, t),
				    &C->j[n - C->di - lo]);
				ab_sub(&Ji, hc_E(C, t), &ji);
				if (inr(C, t) && !ab_sub_le(&Ji, &Im[t - lo]))
					fail("internal: homalg: page %d: j' "
					    "depends on the preimage", R->page);
				hal_checks[HK_WELLDEF]++;
				ab_free(&Ki);
				ab_free(&Ji);
				zm_free(&ji);
			}
			zm_free(&MR);
			zm_free(&X);
			zm_free(&Mj);
			free(y);
			free(cy);
			free(sol);
		}

		/* k'[e] = k e on E'^n */
		t = n + C->dk;
		apply_cols(&imgs, &R->E[n - lo], hc_E(C, n), hc_D(C, t),
		    &C->k[n - lo]);
		from_images(&R->k[n - lo], &R->E[n - lo], hc_D(R, t), &imgs,
		    "k'", R->page);
		zm_free(&imgs);
	}
	for (n = lo; n <= hi; n++)
		ab_free(&Im[n - lo]);
	free(Im);
	make_d(R);
	hc_check(R);
}

int
hc_stable(const struct hcouple *C)
{
	int n;

	for (n = C->lo; n <= C->hi; n++)
		if (!ab_hom_injective(hc_D(C, n), hc_D(C, n + C->di),
		    &C->i[n - C->lo]))
			return 0;
	return 1;
}

void
hc_from_les(struct hcouple *C, const struct hles *L)
{
	int n, len = L->hi - L->lo + 1;

	if (L->A != L->B)
		fail("homalg: an exact couple from 0 -> A -> A -> C -> 0 "
		    "needs the same complex twice");
	hc_init(C, L->lo, L->hi, 0, 0, 1);
	for (n = 0; n < len; n++) {
		ab_copy(&C->D[n], &L->HA[n]);
		ab_copy(&C->E[n], &L->HC[n]);
		zm_copy(&C->i[n], &L->fs[n]);
		zm_copy(&C->j[n], &L->gs[n]);
		zm_copy(&C->k[n], &L->delta[n]);
	}
	make_d(C);
	hc_check(C);
}

/* ---- spectral sequences ---- */

/*
 * The concrete side of hc_stable: d = 0 on the stable page, and the next
 * page, derived, has the same E (the same lattices A and B in every
 * degree) and D_{r+1} = i(D_r) isomorphic to D_r (i injective).
 */
static void
stable_check(const struct hcouple *C)
{
	struct hcouple N;
	int n;

	for (n = C->lo; n <= C->hi; n++)
		if (!ab_hom_is_zero(hc_E(C, n), hc_E(C, n + C->dj + C->dk),
		    &C->d[n - C->lo]))
			fail("internal: homalg: page %d is stable but d != 0",
			    C->page);
	hc_derive(&N, C);
	for (n = C->lo; n <= C->hi; n++) {
		const struct fgab *a = hc_E(C, n), *b = hc_E(&N, n);
		if (a->a != b->a || (a->a > 0 && !zm_lattice_eq(&a->A, &b->A)) ||
		    !ab_isomorphic(a, b, NULL, NULL) ||
		    !ab_isomorphic(hc_D(C, n), hc_D(&N, n), NULL, NULL))
			fail("internal: homalg: page %d is stable but page %d "
			    "differs in degree %d", C->page, N.page, n);
		hal_checks[HK_STABLE]++;
	}
	hc_free(&N);
}

void
hss_run(struct hss *S, const struct hcouple *C1, int maxpage)
{
	int r;

	S->pg = xcalloc((size_t)maxpage + 1, sizeof *S->pg);
	/* page 1 is a copy of C1 */
	hc_init(&S->pg[0], C1->lo, C1->hi, C1->di, C1->dj, C1->dk);
	for (r = 0; r <= C1->hi - C1->lo; r++) {
		ab_copy(&S->pg[0].D[r], &C1->D[r]);
		ab_copy(&S->pg[0].E[r], &C1->E[r]);
		zm_copy(&S->pg[0].i[r], &C1->i[r]);
		zm_copy(&S->pg[0].j[r], &C1->j[r]);
		zm_copy(&S->pg[0].k[r], &C1->k[r]);
	}
	make_d(&S->pg[0]);
	S->npages = 1;
	S->stable = 0;
	for (r = 1; ; r++) {
		if (hc_stable(&S->pg[r - 1])) {
			S->stable = r;
			stable_check(&S->pg[r - 1]);
			break;
		}
		if (r >= maxpage)
			break;
		hc_derive(&S->pg[r], &S->pg[r - 1]);
		S->npages = r + 1;
	}
}

void
hss_free(struct hss *S)
{
	int r;

	for (r = 0; r < S->npages; r++)
		hc_free(&S->pg[r]);
	free(S->pg);
	hles_free(&S->les);
	memset(S, 0, sizeof *S);
}

void
hss_bockstein(struct hss *S, hcx c, zint p, int maxpage)
{
	struct fgab Zp = {0};
	struct hcouple C1;
	hcx cp;
	hmap f, g;
	int n;

	if (!z_isprime(p))
		fail("homalg: the Bockstein spectral sequence needs a prime, "
		    "not %s", z_str(p));
	for (n = hcx_lo(c); n <= hcx_hi(c); n++)
		if (hcx_rel(c, n)->c > 0)
			fail("homalg: the Bockstein spectral sequence of a "
			    "complex of free groups");
	memset(S, 0, sizeof *S);
	S->p = p;
	S->cx = c;
	ab_cyclic(&Zp, p);
	cp = hcx_tensor(c, &Zp);
	f = hmap_scalar(c, p);
	g = hmap_identity(c, cp);
	hles_compute(&S->les, f, g);
	hc_from_les(&C1, &S->les);
	hss_run(S, &C1, maxpage);
	hc_free(&C1);
	ab_free(&Zp);
}

/* ---- tracing ---- */

void
hss_trace(struct htrace *T, const struct hss *S, int n, const zint *a, int r)
{
	const struct hcouple *P1 = &S->pg[0], *Pr;
	const struct fgab *G;
	zint *x, *y;
	int s, deg = n, m, dd;

	memset(T, 0, sizeof *T);
	if (r < 1)
		fail("homalg: pages start at 1");
	if (r > S->npages) {
		if (!S->stable)
			fail("homalg: page %d was not computed", r);
		r = S->npages;	/* stable: E_r = E_stable for r beyond */
	}
	Pr = &S->pg[r - 1];
	G = hc_D(P1, n);
	if (G->n > 0) {
		zint *c = xcalloc((size_t)G->a + 1, sizeof *c);
		int ok = ab_coords(G, a, c);
		free(c);
		if (!ok)
			fail("homalg: the element is not a cocycle of degree %d",
			    n);
	}
	x = xcalloc((size_t)G->n + 1, sizeof *x);
	memcpy(x, a, (size_t)G->n * sizeof *x);
	/* x = i^{r-1} a, one page-1 i at a time */
	for (s = 1; s < r; s++) {
		const struct fgab *Gs = hc_D(P1, deg), *Gt = hc_D(P1, deg +
		    P1->di);
		y = xcalloc((size_t)Gt->n + 1, sizeof *y);
		if (Gt->n > 0)
			ab_hom_apply(Gs, Gt, &P1->i[deg - P1->lo], x, y);
		free(x);
		x = y;
		deg += P1->di;
	}
	T->x = x;
	T->nx = hc_D(P1, deg)->n;
	if (!ab_member(hc_D(Pr, deg), x, NULL))
		fail("internal: homalg: i^%d(a) is not in D_%d", r - 1, r);
	m = deg + Pr->dj;
	T->m = m;
	T->ne = hc_E(Pr, m)->n;
	T->e = xcalloc((size_t)T->ne + 1, sizeof *T->e);
	if (inr(Pr, deg) && T->ne > 0)
		ab_hom_apply(hc_D(Pr, deg), hc_E(Pr, m), &Pr->j[deg - Pr->lo], x,
		    T->e);
	T->zero = T->ne == 0 || ab_elem_is_zero(hc_E(Pr, m), T->e);
	dd = Pr->dj + Pr->dk;
	/* in ker d_r */
	{
		const struct fgab *Et = hc_E(Pr, m + dd);
		zint *de = xcalloc((size_t)Et->n + 1, sizeof *de);
		if (inr(Pr, m) && Et->n > 0)
			ab_hom_apply(hc_E(Pr, m), Et, &Pr->d[m - Pr->lo], T->e,
			    de);
		T->in_ker = Et->n == 0 || ab_elem_is_zero(Et, de);
		free(de);
	}
	/* in im d_r: e = d u + (a boundary), u found by solving */
	T->in_im = 0;
	if (T->zero)
		T->in_im = 1;
	else if (inr(Pr, m - dd)) {
		const struct fgab *Es = hc_E(Pr, m - dd), *Et = hc_E(Pr, m);
		const struct zmat *Md = &Pr->d[m - dd - Pr->lo];
		struct zmat MR = {0};
		zint *ce = xcalloc((size_t)Et->a + 1, sizeof *ce);
		zint *sol = xcalloc((size_t)(Md->c + Et->R.c) + 1, sizeof *sol);

		if (Et->R.c > 0)
			zm_hcat(&MR, Md, &Et->R);
		else
			zm_copy(&MR, Md);
		if (!ab_coords(Et, T->e, ce))
			fail("internal: homalg: j_r(x) is not in E_r");
		if (MR.c > 0 && zm_solve_vec(&MR, ce, sol)) {
			zint *de = xcalloc((size_t)Et->n + 1, sizeof *de);
			T->nw = Es->n;
			T->wit = xcalloc((size_t)T->nw + 1, sizeof *T->wit);
			ab_from_coords(Es, sol, T->wit);
			/* the certificate: d u = e in E_r */
			ab_hom_apply(Es, Et, Md, T->wit, de);
			if (!ab_elem_eq(Et, de, T->e))
				fail("internal: homalg: the preimage under "
				    "d_r does not map to the element");
			free(de);
			T->in_im = 1;
		}
		zm_free(&MR);
		free(ce);
		free(sol);
	}
}

void
hss_trace_free(struct htrace *T)
{
	free(T->x);
	free(T->e);
	free(T->wit);
	memset(T, 0, sizeof *T);
}

/* ---- Proposition 0.1, concretely ---- */

void
hss_prop01(struct hprop01 *R, const struct hss *S, int n, int r)
{
	const struct hcouple *P1 = &S->pg[0], *Pr;
	const struct fgab *H = hc_D(P1, n);
	struct fgab T = {0}, F = {0}, Q = {0}, Im = {0}, K = {0}, Kb = {0};
	struct zmat imgs = {0}, Mi = {0}, Mpsi = {0}, Md = {0}, z = {0};
	int c, s, m, dd;
	zint *x, *y, g = Z_ZERO;

	memset(R, 0, sizeof *R);
	R->n = n;
	R->r = r;
	if (r > S->npages)
		r = S->npages;		/* stable beyond: the same page */
	Pr = &S->pg[r - 1];
	if (P1->di != 0 || Pr->dj != 0 || P1->dk != 1)
		fail("homalg: Proposition 0.1 is about the Bockstein couple "
		    "(degrees 0, 0, 1)");
	/* p^{r-1}: D_1^n -> D_r^n */
	zm_init(&imgs, H->n, H->a);
	x = xcalloc((size_t)H->n + 1, sizeof *x);
	y = xcalloc((size_t)H->n + 1, sizeof *y);
	for (c = 0; c < H->a; c++) {
		int i;
		zm_col(x, &H->A, c);
		for (s = 1; s < r; s++) {
			ab_hom_apply(H, H, &P1->i[n - P1->lo], x, y);
			memcpy(x, y, (size_t)H->n * sizeof *x);
		}
		for (i = 0; i < H->n; i++)
			ZM(&imgs, i, c) = x[i];
	}
	free(x);
	free(y);
	from_images(&Mi, H, hc_D(Pr, n), &imgs, "p^(r-1)", r);
	m = n;
	dd = Pr->dj + Pr->dk;
	/* psi' = j_r p^{r-1}: D_1^n -> E_r^n; first half: d_r psi' = 0 */
	if (inr(Pr, n))
		ab_hom_compose(&Mpsi, &Pr->j[n - Pr->lo], &Mi);
	else
		zm_init(&Mpsi, 0, H->a);
	if (inr(Pr, m)) {
		ab_hom_compose(&Md, &Pr->d[m - Pr->lo], &Mpsi);
		R->ker_ok = ab_hom_is_zero(H, hc_E(Pr, m + dd), &Md);
	} else
		R->ker_ok = 1;
	/* Q = E_r^n / im d_r^{n-1}; the projection is the identity */
	if (inr(Pr, m - dd))
		ab_image(&Im, NULL, hc_E(Pr, m - dd), hc_E(Pr, m),
		    &Pr->d[m - dd - Pr->lo]);
	else {
		zm_init(&z, hc_E(Pr, m)->n, 0);
		ab_sub(&Im, hc_E(Pr, m), &z);
		zm_free(&z);
	}
	ab_quotient(&Q, hc_E(Pr, m), &Im);
	ab_kernel(&K, NULL, H, &Q, &Mpsi);
	/* the image of K in F = H / T(H), and the gcd of its coordinates */
	ab_torsion(&T, H);
	ab_quotient(&F, H, &T);
	R->free_rank = F.rank;
	ab_sub(&Kb, &F, &K.A);
	y = xcalloc((size_t)F.ninv + 1, sizeof *y);
	x = xcalloc((size_t)F.n + 1, sizeof *x);
	for (c = 0; c < Kb.a; c++) {
		int i;
		zm_col(x, &Kb.A, c);
		if (!ab_canon(&F, x, y))
			fail("internal: homalg: ker psi is not in H");
		for (i = 0; i < F.ninv; i++)
			g = z_gcd(g, y[i]);
	}
	free(x);
	free(y);
	R->content = g;
	R->im_ok = !z_is_one(g);
	R->holds = R->ker_ok && R->im_ok;
	ab_free(&T);
	ab_free(&F);
	ab_free(&Q);
	ab_free(&Im);
	ab_free(&K);
	ab_free(&Kb);
	zm_free(&imgs);
	zm_free(&Mi);
	zm_free(&Mpsi);
	zm_free(&Md);
}
