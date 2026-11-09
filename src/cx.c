/*
 * cx.c - cochain complexes of presented f.g. abelian groups, chain maps,
 * cohomology as fgab subquotients, short exact sequences of complexes and
 * their long exact sequences (homalg.h).
 *
 * Storage (struct of arrays, process-lifetime arenas):
 *   complexes   lo[] hi[] off[] name[]
 *   degrees     g[] d[] R[]      (complex c, degree n at off[c] + n - lo[c])
 *   chain maps  src[] dst[] moff[] and F[] (by n - lo of src)
 */
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "homalg.h"

long hal_checks[HK_NKIND];
const char *const hal_check_name[HK_NKIND] = {
	"d o d = 0", "d maps relations to relations", "chain map commutes with d",
	"induced map well defined", "exact", "short exact (degree)",
	"universal coefficients", "snake lemma lift",
	"stable page: d = 0, next page equal",
	"reduction: f g = 1, 1 - g f = dh + hd, chain maps"
};

void
hal_checks_reset(void)
{
	memset(hal_checks, 0, sizeof hal_checks);
}

/* ---- lattice helpers ---- */

/* M Z^c <= L Z^k (L may have no columns) */
static int
lat_le(const struct zmat *M, const struct zmat *L)
{
	if (M->c == 0 || zm_is_zero(M))
		return 1;
	if (L == NULL || L->c == 0)
		return 0;
	return zm_lattice_le(M, L);
}

/* generators of {x in Z^c : F x in L Z^k} (F: r x c) */
static void
lat_preimage(struct zmat *out, const struct zmat *F, const struct zmat *L)
{
	struct zmat FL = {0}, K = {0};
	int i, j;

	if (F->r == 0 || zm_is_zero(F)) {
		zm_identity(out, F->c);
		return;
	}
	if (L && L->c > 0)
		zm_hcat(&FL, F, L);
	else
		zm_copy(&FL, F);
	zm_kernel(&FL, &K);
	zm_init(out, F->c, K.c);
	for (i = 0; i < F->c; i++)
		for (j = 0; j < K.c; j++)
			ZM(out, i, j) = ZM(&K, i, j);
	zm_free(&FL);
	zm_free(&K);
}

/* a (x) b, the Kronecker product: (i a.r... ) entry (i*b.r + k, j*b.c + l) */
static void
kron(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	int i, j, k, l;

	zm_init(r, a->r * b->r, a->c * b->c);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < a->c; j++)
			if (!z_is_zero(ZM(a, i, j)))
				for (k = 0; k < b->r; k++)
					for (l = 0; l < b->c; l++)
						ZM(r, i * b->r + k, j * b->c + l) =
						    z_mul(ZM(a, i, j), ZM(b, k, l));
}

/* ---- complexes ---- */

static struct {
	uint32_t n, cap;
	int *lo, *hi;
	uint32_t *off, *name;
} X;

static struct {
	uint32_t n, cap;
	int *g;
	struct zmat *d, *R;
} XD;

static void
xchk(hcx c)
{
	if (c >= X.n)
		fail("homalg: no complex %u", (unsigned)c);
}

int
hcx_lo(hcx c)
{
	xchk(c);
	return X.lo[c];
}

int
hcx_hi(hcx c)
{
	xchk(c);
	return X.hi[c];
}

const char *
hcx_name(hcx c)
{
	xchk(c);
	return sym_str(X.name[c]);
}

int
hcx_rank(hcx c, int n)
{
	xchk(c);
	if (n < X.lo[c] || n > X.hi[c])
		return 0;
	return XD.g[X.off[c] + (uint32_t)(n - X.lo[c])];
}

const struct zmat *
hcx_d(hcx c, int n)
{
	xchk(c);
	if (n < X.lo[c] || n > X.hi[c])
		return NULL;
	return &XD.d[X.off[c] + (uint32_t)(n - X.lo[c])];
}

const struct zmat *
hcx_rel(hcx c, int n)
{
	xchk(c);
	if (n < X.lo[c] || n > X.hi[c])
		return NULL;
	return &XD.R[X.off[c] + (uint32_t)(n - X.lo[c])];
}

hcx
hcx_new(const char *name, int lo, int hi, const int *g, const struct zmat *d,
    const struct zmat *R)
{
	hcx c;
	int n, len = hi - lo + 1;

	if (len < 1)
		fail("homalg: a complex needs a degree");
	for (n = 0; n < len; n++) {
		int gn1 = n + 1 < len ? g[n + 1] : 0;
		if (g[n] < 0 || d[n].c != g[n] || d[n].r != gn1)
			fail("homalg: %s: d^%d is %d x %d, not %d x %d", name,
			    lo + n, d[n].r, d[n].c, gn1, g[n]);
		if (R && R[n].c > 0 && R[n].r != g[n])
			fail("homalg: %s: the relations in degree %d have %d "
			    "rows, not %d", name, lo + n, R[n].r, g[n]);
	}
	if (X.n == X.cap) {
		X.cap = newcap(X.cap, X.n + 1);
		X.lo = xrealloc(X.lo, X.cap * sizeof *X.lo);
		X.hi = xrealloc(X.hi, X.cap * sizeof *X.hi);
		X.off = xrealloc(X.off, X.cap * sizeof *X.off);
		X.name = xrealloc(X.name, X.cap * sizeof *X.name);
	}
	if (XD.n + (uint32_t)len > XD.cap) {
		XD.cap = newcap(XD.cap, XD.n + (uint32_t)len);
		XD.g = xrealloc(XD.g, XD.cap * sizeof *XD.g);
		XD.d = xrealloc(XD.d, XD.cap * sizeof *XD.d);
		XD.R = xrealloc(XD.R, XD.cap * sizeof *XD.R);
	}
	c = X.n;
	X.lo[c] = lo;
	X.hi[c] = hi;
	X.off[c] = XD.n;
	X.name[c] = sym_cstr(name);
	for (n = 0; n < len; n++) {
		uint32_t o = XD.n + (uint32_t)n;
		XD.g[o] = g[n];
		memset(&XD.d[o], 0, sizeof XD.d[o]);
		memset(&XD.R[o], 0, sizeof XD.R[o]);
		zm_copy(&XD.d[o], &d[n]);
		if (R && R[n].c > 0)
			zm_copy(&XD.R[o], &R[n]);
		else
			zm_init(&XD.R[o], g[n], 0);
	}
	XD.n += (uint32_t)len;
	X.n++;
	/* the certificates: d(R_n) <= R_{n+1}, d d (Z^g) <= R_{n+2} */
	for (n = lo; n <= hi; n++) {
		const struct zmat *dn = hcx_d(c, n), *Rn = hcx_rel(c, n);
		struct zmat t = {0};
		if (n + 1 <= hi && Rn->c > 0) {
			zm_mul(&t, dn, Rn);
			if (!lat_le(&t, hcx_rel(c, n + 1)))
				fail("homalg: %s: d^%d does not map the relations "
				    "into the relations", name, n);
			zm_free(&t);
			hal_checks[HK_REL]++;
		}
		if (n + 2 <= hi) {
			zm_mul(&t, hcx_d(c, n + 1), dn);
			if (!lat_le(&t, hcx_rel(c, n + 2)))
				fail("homalg: %s: d^%d d^%d != 0", name, n + 1, n);
			zm_free(&t);
			hal_checks[HK_DD]++;
		}
	}
	return c;
}

hcx
hcx_cochains(hspace s)
{
	int top = hs_top(s), n, *g;
	struct zmat *d;
	char name[300];
	hcx c;

	g = xmalloc((size_t)(top + 1) * sizeof *g);
	d = xcalloc((size_t)(top + 1), sizeof *d);
	for (n = 0; n <= top; n++)
		g[n] = (int)hs_ncells(s, n);
	for (n = 0; n <= top; n++) {
		const uint32_t *r, *cc;
		const int64_t *v;
		uint32_t m, e;
		zm_init(&d[n], n < top ? g[n + 1] : 0, g[n]);
		m = hs_bd(s, n + 1, &r, &cc, &v);
		/* d^n = (d_{n+1})^T: the (n+1)-cell cc[e] gets v * its face */
		for (e = 0; e < m; e++)
			ZM(&d[n], cc[e], r[e]) = z_from_i64(v[e]);
	}
	snprintf(name, sizeof name, "C^*(%s)", hs_name(s));
	c = hcx_new(name, 0, top, g, d, NULL);
	for (n = 0; n <= top; n++)
		zm_free(&d[n]);
	free(d);
	free(g);
	return c;
}

hcx
hcx_chains(hspace s)
{
	int top = hs_top(s), n, *g;
	struct zmat *d;
	char name[300];
	hcx c;

	/* degree -n (index top - n) holds C_n; d^{-n} = d_n */
	g = xmalloc((size_t)(top + 1) * sizeof *g);
	d = xcalloc((size_t)(top + 1), sizeof *d);
	for (n = 0; n <= top; n++)
		g[top - n] = (int)hs_ncells(s, n);
	for (n = 0; n <= top; n++) {
		const uint32_t *r, *cc;
		const int64_t *v;
		uint32_t m, e;
		zm_init(&d[top - n], n > 0 ? (int)hs_ncells(s, n - 1) : 0,
		    (int)hs_ncells(s, n));
		m = hs_bd(s, n, &r, &cc, &v);
		for (e = 0; e < m; e++)
			ZM(&d[top - n], r[e], cc[e]) = z_from_i64(v[e]);
	}
	snprintf(name, sizeof name, "C_*(%s)", hs_name(s));
	c = hcx_new(name, -top, 0, g, d, NULL);
	for (n = 0; n <= top; n++)
		zm_free(&d[n]);
	free(d);
	free(g);
	return c;
}

hcx
hcx_tensor(hcx c, const struct fgab *G)
{
	int lo = hcx_lo(c), hi = hcx_hi(c), len = hi - lo + 1, n, a = G->a;
	int *g = xmalloc((size_t)len * sizeof *g);
	struct zmat *d = xcalloc((size_t)len, sizeof *d);
	struct zmat *R = xcalloc((size_t)len, sizeof *R);
	struct zmat Ia = {0};
	char name[300];
	hcx t;

	zm_identity(&Ia, a);
	for (n = 0; n < len; n++) {
		const struct zmat *Rn = hcx_rel(c, lo + n);
		struct zmat Ig = {0}, r1 = {0}, r2 = {0};
		g[n] = hcx_rank(c, lo + n) * a;
		kron(&d[n], hcx_d(c, lo + n), &Ia);
		zm_identity(&Ig, hcx_rank(c, lo + n));
		kron(&r2, &Ig, &G->R);
		if (Rn->c > 0) {
			kron(&r1, Rn, &Ia);
			zm_hcat(&R[n], &r1, &r2);
		} else
			zm_copy(&R[n], &r2);
		zm_free(&Ig);
		zm_free(&r1);
		zm_free(&r2);
	}
	snprintf(name, sizeof name, "%s (x) %s", hcx_name(c), ab_str(G));
	t = hcx_new(name, lo, hi, g, d, R);
	for (n = 0; n < len; n++) {
		zm_free(&d[n]);
		zm_free(&R[n]);
	}
	zm_free(&Ia);
	free(d);
	free(R);
	free(g);
	return t;
}

void
hcx_cohomology(struct fgab *H, hcx c, int n)
{
	struct zmat Z = {0}, Bd = {0};
	int g = hcx_rank(c, n);

	if (g == 0) {
		ab_free_abelian(H, 0);
		return;
	}
	lat_preimage(&Z, hcx_d(c, n), n + 1 <= hcx_hi(c) ? hcx_rel(c, n + 1)
	    : NULL);
	if (n - 1 >= hcx_lo(c) && hcx_rank(c, n - 1) > 0)
		zm_hcat(&Bd, hcx_d(c, n - 1), hcx_rel(c, n));
	else
		zm_copy(&Bd, hcx_rel(c, n));
	if (!ab_subquotient(H, g, &Z, Bd.c > 0 ? &Bd : NULL))
		fail("internal: homalg: %s: the boundaries in degree %d are "
		    "not cycles", hcx_name(c), n);
	zm_free(&Z);
	zm_free(&Bd);
}

/* ---- chain maps ---- */

static struct {
	uint32_t n, cap;
	hcx *src, *dst;
	uint32_t *off;
} M;
static struct {
	uint32_t n, cap;
	struct zmat *F;
} MF;

hcx
hmap_src(hmap f)
{
	if (f >= M.n)
		fail("homalg: no chain map %u", (unsigned)f);
	return M.src[f];
}

hcx
hmap_dst(hmap f)
{
	if (f >= M.n)
		fail("homalg: no chain map %u", (unsigned)f);
	return M.dst[f];
}

const struct zmat *
hmap_F(hmap f, int n)
{
	hcx s = hmap_src(f);

	if (n < hcx_lo(s) || n > hcx_hi(s))
		return NULL;
	return &MF.F[M.off[f] + (uint32_t)(n - hcx_lo(s))];
}

hmap
hmap_new(hcx src, hcx dst, const struct zmat *F)
{
	int lo = hcx_lo(src), hi = hcx_hi(src), n, len = hi - lo + 1;
	hmap f;

	if (hcx_lo(dst) != lo || hcx_hi(dst) != hi)
		fail("homalg: a chain map between complexes with the same "
		    "degrees");
	for (n = lo; n <= hi; n++)
		if (F[n - lo].r != hcx_rank(dst, n) || F[n - lo].c !=
		    hcx_rank(src, n))
			fail("homalg: the chain map in degree %d is %d x %d, not "
			    "%d x %d", n, F[n - lo].r, F[n - lo].c,
			    hcx_rank(dst, n), hcx_rank(src, n));
	if (M.n == M.cap) {
		M.cap = newcap(M.cap, M.n + 1);
		M.src = xrealloc(M.src, M.cap * sizeof *M.src);
		M.dst = xrealloc(M.dst, M.cap * sizeof *M.dst);
		M.off = xrealloc(M.off, M.cap * sizeof *M.off);
	}
	if (MF.n + (uint32_t)len > MF.cap) {
		MF.cap = newcap(MF.cap, MF.n + (uint32_t)len);
		MF.F = xrealloc(MF.F, MF.cap * sizeof *MF.F);
	}
	f = M.n;
	M.src[f] = src;
	M.dst[f] = dst;
	M.off[f] = MF.n;
	for (n = 0; n < len; n++) {
		memset(&MF.F[MF.n + (uint32_t)n], 0, sizeof MF.F[0]);
		zm_copy(&MF.F[MF.n + (uint32_t)n], &F[n]);
	}
	MF.n += (uint32_t)len;
	M.n++;
	/* F(R) <= R', d' F - F d in R' */
	for (n = lo; n <= hi; n++) {
		struct zmat t = {0}, u = {0}, w = {0};
		const struct zmat *Rn = hcx_rel(src, n);
		if (Rn->c > 0) {
			zm_mul(&t, hmap_F(f, n), Rn);
			if (!lat_le(&t, hcx_rel(dst, n)))
				fail("homalg: the chain map does not map the "
				    "relations of degree %d into the relations",
				    n);
			zm_free(&t);
		}
		if (n < hi) {
			zm_mul(&t, hcx_d(dst, n), hmap_F(f, n));
			zm_mul(&u, hmap_F(f, n + 1), hcx_d(src, n));
			zm_sub(&w, &t, &u);
			if (!lat_le(&w, hcx_rel(dst, n + 1)))
				fail("homalg: not a chain map: d F != F d in "
				    "degree %d", n);
			zm_free(&t);
			zm_free(&u);
			zm_free(&w);
		}
		hal_checks[HK_CHAIN]++;
	}
	return f;
}

hmap
hmap_scalar(hcx c, zint m)
{
	int lo = hcx_lo(c), hi = hcx_hi(c), n;
	struct zmat *F = xcalloc((size_t)(hi - lo + 1), sizeof *F);
	hmap f;

	for (n = lo; n <= hi; n++) {
		struct zmat I = {0};
		zm_identity(&I, hcx_rank(c, n));
		zm_scale(&F[n - lo], &I, m);
		zm_free(&I);
	}
	f = hmap_new(c, c, F);
	for (n = lo; n <= hi; n++)
		zm_free(&F[n - lo]);
	free(F);
	return f;
}

hmap
hmap_identity(hcx src, hcx dst)
{
	int lo = hcx_lo(src), hi = hcx_hi(src), n;
	struct zmat *F = xcalloc((size_t)(hi - lo + 1), sizeof *F);
	hmap f;

	for (n = lo; n <= hi; n++) {
		if (hcx_rank(src, n) != hcx_rank(dst, n))
			fail("homalg: the identity between ranks %d and %d",
			    hcx_rank(src, n), hcx_rank(dst, n));
		zm_identity(&F[n - lo], hcx_rank(src, n));
	}
	f = hmap_new(src, dst, F);
	for (n = lo; n <= hi; n++)
		zm_free(&F[n - lo]);
	free(F);
	return f;
}

void
hmap_induced(struct zmat *Mo, hmap f, int n, const struct fgab *H,
    const struct fgab *Hp)
{
	const struct zmat *F = hmap_F(f, n);

	if (F == NULL || H->a == 0) {
		zm_init(Mo, Hp->a, H->a);
		return;
	}
	if (!ab_hom_induced(Mo, H, Hp, F))
		fail("internal: homalg: H^%d of the chain map %s -> %s is not "
		    "well defined", n, hcx_name(hmap_src(f)),
		    hcx_name(hmap_dst(f)));
	if (!ab_hom_check(H, Hp, Mo))
		fail("internal: homalg: H^%d(f) fails ab_hom_check", n);
	hal_checks[HK_WELLDEF]++;
}

/* ---- short and long exact sequences ---- */

/* the presented group Z^g / R of a complex in degree n */
static void
presented(struct fgab *G, hcx c, int n)
{
	const struct zmat *R = hcx_rel(c, n);
	int g = hcx_rank(c, n);

	if (g == 0) {
		ab_free_abelian(G, 0);
		return;
	}
	if (!ab_subquotient(G, g, NULL, R->c > 0 ? R : NULL))
		fail("internal: homalg: relations outside Z^%d", g);
}

static void
ses_check(hmap f, hmap g, int n)
{
	struct fgab Ga = {0}, Gb = {0}, Gc = {0};
	struct zmat Mf = {0}, Mg = {0};
	const char *why = NULL;
	hcx A = hmap_src(f), B = hmap_dst(f), C = hmap_dst(g);

	presented(&Ga, A, n);
	presented(&Gb, B, n);
	presented(&Gc, C, n);
	if (Ga.a == 0)
		zm_init(&Mf, Gb.a, 0);
	else if (!ab_hom_induced(&Mf, &Ga, &Gb, hmap_F(f, n)))
		fail("homalg: f is not well defined in degree %d", n);
	if (Gb.a == 0)
		zm_init(&Mg, Gc.a, 0);
	else if (!ab_hom_induced(&Mg, &Gb, &Gc, hmap_F(g, n)))
		fail("homalg: g is not well defined in degree %d", n);
	if (!ab_hom_injective(&Ga, &Gb, &Mf))
		fail("homalg: not short exact: f is not injective in degree "
		    "%d", n);
	if (!ab_exact_at(&Ga, &Gb, &Gc, &Mf, &Mg, NULL, &why))
		fail("homalg: not short exact in degree %d: %s", n,
		    why ? why : "im f != ker g");
	if (!ab_hom_surjective(&Gb, &Gc, &Mg))
		fail("homalg: not short exact: g is not onto in degree %d", n);
	hal_checks[HK_SES]++;
	ab_free(&Ga);
	ab_free(&Gb);
	ab_free(&Gc);
	zm_free(&Mf);
	zm_free(&Mg);
}

/* delta: H^n(C) -> H^{n+1}(A) by the snake lemma, on HC's basis */
static void
connecting(struct zmat *D, hmap f, hmap g, int n, const struct fgab *HC,
    const struct fgab *HA1)
{
	hcx A = hmap_src(f), B = hmap_dst(f), C = hmap_dst(g);
	int hi = hcx_hi(A), j, gb = hcx_rank(B, n), ga1 = hcx_rank(A, n + 1);
	struct zmat GR = {0}, FR = {0}, imgs = {0};
	zint *z, *b, *u, *sol, *chk;

	if (n + 1 > hi || HC->a == 0) {
		zm_init(D, HA1->a, HC->a);
		return;
	}
	/* [g | R_C] (b; y) = z and [f | R_B] (a; y) = d b */
	if (hcx_rel(C, n)->c > 0)
		zm_hcat(&GR, hmap_F(g, n), hcx_rel(C, n));
	else
		zm_copy(&GR, hmap_F(g, n));
	if (hcx_rel(B, n + 1)->c > 0)
		zm_hcat(&FR, hmap_F(f, n + 1), hcx_rel(B, n + 1));
	else
		zm_copy(&FR, hmap_F(f, n + 1));
	zm_init(&imgs, ga1, HC->a);
	z = xcalloc((size_t)HC->n + 1, sizeof *z);
	sol = xcalloc((size_t)(GR.c + FR.c) + 1, sizeof *sol);
	b = xcalloc((size_t)gb + 1, sizeof *b);
	u = xcalloc((size_t)hcx_rank(B, n + 1) + 1, sizeof *u);
	chk = xcalloc((size_t)hcx_rank(B, n + 1) + (size_t)GR.r + 1,
	    sizeof *chk);
	for (j = 0; j < HC->a; j++) {
		int i;
		zm_col(z, &HC->A, j);
		if (!zm_solve_vec(&GR, z, sol))
			fail("internal: homalg: snake: a cycle of %s has no "
			    "lift in degree %d", hcx_name(C), n);
		for (i = 0; i < gb; i++)
			b[i] = sol[i];
		/* verify g b = z mod R_C: done by zm_solve_vec; d b */
		zm_mul_vec(u, hcx_d(B, n), b);
		if (!zm_solve_vec(&FR, u, sol))
			fail("internal: homalg: snake: d b is not in f(A) + "
			    "R_B in degree %d", n + 1);
		for (i = 0; i < ga1; i++)
			ZM(&imgs, i, j) = sol[i];
		/* re-check f a + R y = d b */
		zm_mul_vec(chk, &FR, sol);
		for (i = 0; i < FR.r; i++)
			if (!z_eq(chk[i], u[i]))
				fail("internal: homalg: snake: f a != d b");
		hal_checks[HK_SNAKE]++;
	}
	if (!ab_hom_from_images(D, HC, HA1, &imgs))
		fail("internal: homalg: snake: delta of a cycle is not a cycle "
		    "(degree %d)", n);
	if (!ab_hom_check(HC, HA1, D))
		fail("internal: homalg: delta^%d is not well defined", n);
	hal_checks[HK_WELLDEF]++;
	zm_free(&GR);
	zm_free(&FR);
	zm_free(&imgs);
	free(z);
	free(sol);
	free(b);
	free(u);
	free(chk);
}

void
hles_compute(struct hles *L, hmap f, hmap g)
{
	hcx A = hmap_src(f), B = hmap_dst(f), C = hmap_dst(g);
	int lo = hcx_lo(A), hi = hcx_hi(A), len = hi - lo + 1, n, k, bad;
	struct fgab *G, zero = {0};
	struct zmat *Ms;

	if (hmap_src(g) != B)
		fail("homalg: a short exact sequence A -f-> B -g-> C");
	if (hcx_lo(C) != lo || hcx_hi(C) != hi || hcx_lo(B) != lo ||
	    hcx_hi(B) != hi)
		fail("homalg: A, B and C need the same degrees");
	memset(L, 0, sizeof *L);
	L->lo = lo;
	L->hi = hi;
	L->A = A;
	L->B = B;
	L->C = C;
	for (n = lo; n <= hi; n++)
		ses_check(f, g, n);
	L->HA = xcalloc((size_t)len, sizeof *L->HA);
	L->HB = xcalloc((size_t)len, sizeof *L->HB);
	L->HC = xcalloc((size_t)len, sizeof *L->HC);
	L->fs = xcalloc((size_t)len, sizeof *L->fs);
	L->gs = xcalloc((size_t)len, sizeof *L->gs);
	L->delta = xcalloc((size_t)len, sizeof *L->delta);
	for (n = 0; n < len; n++) {
		hcx_cohomology(&L->HA[n], A, lo + n);
		if (B == A)
			ab_copy(&L->HB[n], &L->HA[n]);
		else
			hcx_cohomology(&L->HB[n], B, lo + n);
		hcx_cohomology(&L->HC[n], C, lo + n);
	}
	for (n = 0; n < len; n++) {
		hmap_induced(&L->fs[n], f, lo + n, &L->HA[n], &L->HB[n]);
		hmap_induced(&L->gs[n], g, lo + n, &L->HB[n], &L->HC[n]);
	}
	ab_free_abelian(&zero, 0);
	for (n = 0; n < len; n++)
		connecting(&L->delta[n], f, g, lo + n, &L->HC[n], n + 1 < len ?
		    &L->HA[n + 1] : &zero);
	/* 0 -> HA^lo -> HB^lo -> HC^lo -> HA^lo+1 -> ... -> HC^hi -> 0 */
	G = xcalloc((size_t)(3 * len + 2), sizeof *G);
	Ms = xcalloc((size_t)(3 * len + 1), sizeof *Ms);
	G[0] = zero;
	zm_init(&Ms[0], L->HA[0].a, 0);
	for (n = 0, k = 1; n < len; n++) {
		G[k] = L->HA[n];
		Ms[k] = L->fs[n];
		k++;
		G[k] = L->HB[n];
		Ms[k] = L->gs[n];
		k++;
		G[k] = L->HC[n];
		if (n + 1 < len)
			Ms[k] = L->delta[n];
		else
			zm_init(&Ms[k], 0, L->HC[n].a);
		k++;
	}
	G[k] = zero;
	k++;
	if (!ab_exact_seq(G, k, Ms, &bad))
		fail("internal: homalg: the long exact sequence is not exact "
		    "at spot %d", bad);
	hal_checks[HK_EXACT] += k - 2;
	zm_free(&Ms[0]);
	zm_free(&Ms[k - 2]);
	free(G);
	free(Ms);
	ab_free(&zero);
}

void
hles_free(struct hles *L)
{
	int n, len = L->hi - L->lo + 1;

	if (L->HA == NULL)
		return;
	for (n = 0; n < len; n++) {
		ab_free(&L->HA[n]);
		ab_free(&L->HB[n]);
		ab_free(&L->HC[n]);
		zm_free(&L->fs[n]);
		zm_free(&L->gs[n]);
		zm_free(&L->delta[n]);
	}
	free(L->HA);
	free(L->HB);
	free(L->HC);
	free(L->fs);
	free(L->gs);
	free(L->delta);
	memset(L, 0, sizeof *L);
}
