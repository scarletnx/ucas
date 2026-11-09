/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ideal.c - polynomials in several variables over Q or F_p, Groebner
 * bases with cofactors (Buchberger with the product and chain criteria),
 * ideal membership, inclusion and equality with certificates, and the
 * Smith form of matrices over F[x] (modules over a PID: rational
 * canonical form invariants).
 *
 * A polynomial is struct-of-arrays: exponent vectors (n * nv, one flat
 * array) and the coefficient's numerator and denominator columns, terms
 * in decreasing monomial order.  Over F_p a coefficient is its residue
 * (denominator 1).  Every Groebner basis element carries its cofactors,
 * g_i = sum_j c_ij f_j, which are updated through every S-polynomial and
 * reduction; gb_check re-verifies them, re-reduces every S-pair of the
 * final basis to 0 and every input to 0.  Membership answers h = sum
 * c_j f_j, multiplied out and compared.
 */
#include "absalg.h"
#include "ucas.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

/* ---- coefficients ---- */

static struct zq
c_norm(const struct mring *R, struct zq a)
{
	if (R->p) {
		zint p = z_from_u64(R->p);
		zint d = z_invmod(a.d, p);
		if (d == 0)
			fail("division by a multiple of %llu in F_%llu",
			    (unsigned long long)R->p, (unsigned long long)R->p);
		a.n = z_mod(z_mul(a.n, d), p);
		a.d = Z_ONE;
	}
	return a;
}

static struct zq
c_add(const struct mring *R, struct zq a, struct zq b)
{
	if (R->p) {
		struct zq r;
		r.n = z_mod(z_add(a.n, b.n), z_from_u64(R->p));
		r.d = Z_ONE;
		return r;
	}
	return zq_add(a, b);
}

static struct zq
c_neg(const struct mring *R, struct zq a)
{
	if (R->p) {
		a.n = z_mod(z_neg(a.n), z_from_u64(R->p));
		return a;
	}
	return zq_neg(a);
}

static struct zq
c_mul(const struct mring *R, struct zq a, struct zq b)
{
	if (R->p) {
		struct zq r;
		r.n = z_mod(z_mul(a.n, b.n), z_from_u64(R->p));
		r.d = Z_ONE;
		return r;
	}
	return zq_mul(a, b);
}

static struct zq
c_inv(const struct mring *R, struct zq a)
{
	if (R->p) {
		struct zq r;
		r.n = z_invmod(a.n, z_from_u64(R->p));
		if (r.n == 0)
			fail("division by zero in F_%llu", (unsigned long long)R->p);
		r.d = Z_ONE;
		return r;
	}
	return zq_inv(a);
}

static int
c_is_one(struct zq a)
{
	return z_is_one(a.n) && z_is_one(a.d);
}

/* ---- monomials ---- */

static int
mono_cmp(const struct mring *R, const uint32_t *a, const uint32_t *b)
{
	int i, nv = R->nv;
	uint64_t da = 0, db = 0;

	if (R->order != MPL_LEX) {
		for (i = 0; i < nv; i++) {
			da += a[i];
			db += b[i];
		}
		if (da != db)
			return da > db ? 1 : -1;
	}
	if (R->order == MPL_GREVLEX) {
		for (i = nv - 1; i >= 0; i--)
			if (a[i] != b[i])
				return a[i] < b[i] ? 1 : -1;
		return 0;
	}
	for (i = 0; i < nv; i++)
		if (a[i] != b[i])
			return a[i] > b[i] ? 1 : -1;
	return 0;
}

static int
mono_divides(int nv, const uint32_t *a, const uint32_t *b)
{
	int i;

	for (i = 0; i < nv; i++)
		if (a[i] > b[i])
			return 0;
	return 1;
}

/* ---- polynomials ---- */

#define EXP(f, R, k) ((f)->e + (size_t)(k) * (size_t)(R)->nv)

void
mring_init(struct mring *R, int nv, const char *const *names, int order,
    u64 p)
{
	int i;

	memset(R, 0, sizeof *R);
	if (nv < 0 || nv > MPL_MAXV)
		fail("polynomials: at most %d variables", MPL_MAXV);
	if (p && !nt_isprime(p))
		fail("polynomials over F_%llu: %llu is not a prime",
		    (unsigned long long)p, (unsigned long long)p);
	R->nv = nv;
	R->order = order;
	R->p = p;
	for (i = 0; i < nv; i++)
		snprintf(R->name[i], sizeof R->name[i], "%s", names[i]);
}

void
mpl_init(struct mpl *f)
{
	memset(f, 0, sizeof *f);
}

void
mpl_free(struct mpl *f)
{
	free(f->e);
	free(f->cn);
	free(f->cd);
	memset(f, 0, sizeof *f);
}

static void
reserve(struct mpl *f, uint32_t n, int nv)
{
	f->nv = nv;
	if (n <= f->cap)
		return;
	f->cap = n < 8 ? 8 : n * 2;
	f->e = xrealloc(f->e, (size_t)f->cap * (nv ? nv : 1) * sizeof *f->e);
	f->cn = xrealloc(f->cn, (size_t)f->cap * sizeof *f->cn);
	f->cd = xrealloc(f->cd, (size_t)f->cap * sizeof *f->cd);
}

static void
push(const struct mring *R, struct mpl *f, const uint32_t *e, struct zq c)
{
	reserve(f, f->n + 1, R->nv);
	memcpy(EXP(f, R, f->n), e, (size_t)R->nv * sizeof *e);
	f->cn[f->n] = c.n;
	f->cd[f->n] = c.d;
	f->n++;
}

static struct zq
coef(const struct mpl *f, uint32_t k)
{
	struct zq c;

	c.n = f->cn[k];
	c.d = f->cd[k];
	return c;
}

static void
move(struct mpl *dst, struct mpl *src)
{
	if (dst == src)
		return;
	mpl_free(dst);
	*dst = *src;
	memset(src, 0, sizeof *src);
}

void
mpl_copy(struct mpl *dst, const struct mpl *src)
{
	struct mpl t;
	int nv = src->nv;

	memset(&t, 0, sizeof t);
	t.n = t.cap = src->n;
	t.nv = nv;
	t.e = xmalloc(((size_t)src->n * nv + 1) * sizeof *t.e);
	t.cn = xmalloc(((size_t)src->n + 1) * sizeof *t.cn);
	t.cd = xmalloc(((size_t)src->n + 1) * sizeof *t.cd);
	memcpy(t.e, src->e, (size_t)src->n * nv * sizeof *t.e);
	memcpy(t.cn, src->cn, (size_t)src->n * sizeof *t.cn);
	memcpy(t.cd, src->cd, (size_t)src->n * sizeof *t.cd);
	move(dst, &t);
}

static void
copy_r(const struct mring *R, struct mpl *dst, const struct mpl *src)
{
	struct mpl t;
	uint32_t k;

	memset(&t, 0, sizeof t);
	reserve(&t, src->n, R->nv);
	for (k = 0; k < src->n; k++)
		push(R, &t, EXP(src, R, k), coef(src, k));
	move(dst, &t);
}

void
mpl_const(const struct mring *R, struct mpl *f, struct zq c)
{
	struct mpl t;
	uint32_t e[MPL_MAXV] = {0};

	memset(&t, 0, sizeof t);
	c = c_norm(R, c);
	if (c.n != 0)
		push(R, &t, e, c);
	move(f, &t);
}

void
mpl_var(const struct mring *R, struct mpl *f, int v)
{
	struct mpl t;
	uint32_t e[MPL_MAXV] = {0};

	memset(&t, 0, sizeof t);
	e[v] = 1;
	push(R, &t, e, zq_int(Z_ONE));
	move(f, &t);
}

int
mpl_is_zero(const struct mpl *f)
{
	return f->n == 0;
}

int
mpl_eq(const struct mring *R, const struct mpl *f, const struct mpl *g)
{
	uint32_t k;

	if (f->n != g->n)
		return 0;
	for (k = 0; k < f->n; k++)
		if (mono_cmp(R, EXP(f, R, k), EXP(g, R, k)) != 0 ||
		    !zq_eq(coef(f, k), coef(g, k)))
			return 0;
	return 1;
}

/* r = a + s b (s = +1 or -1), by merging */
static void
addsub(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b, int neg)
{
	struct mpl t;
	uint32_t i = 0, j = 0;

	memset(&t, 0, sizeof t);
	reserve(&t, a->n + b->n, R->nv);
	while (i < a->n || j < b->n) {
		int c = i >= a->n ? -1 : j >= b->n ? 1 : mono_cmp(R, EXP(a, R, i),
		    EXP(b, R, j));
		if (c > 0) {
			push(R, &t, EXP(a, R, i), coef(a, i));
			i++;
		} else if (c < 0) {
			struct zq v = coef(b, j);
			push(R, &t, EXP(b, R, j), neg ? c_neg(R, v) : v);
			j++;
		} else {
			struct zq v = coef(b, j);
			v = c_add(R, coef(a, i), neg ? c_neg(R, v) : v);
			if (v.n != 0)
				push(R, &t, EXP(a, R, i), v);
			i++;
			j++;
		}
	}
	move(r, &t);
}

void
mpl_add(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b)
{
	addsub(R, r, a, b, 0);
}

void
mpl_sub(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b)
{
	addsub(R, r, a, b, 1);
}

/* r = c x^e a (monomial orders are multiplicative: no sort needed) */
static void
mul_term(const struct mring *R, struct mpl *r, const struct mpl *a,
    const uint32_t *e, struct zq c)
{
	struct mpl t;
	uint32_t k, m[MPL_MAXV];
	int i;

	memset(&t, 0, sizeof t);
	if (c.n == 0) {
		move(r, &t);
		return;
	}
	reserve(&t, a->n, R->nv);
	for (k = 0; k < a->n; k++) {
		const uint32_t *x = EXP(a, R, k);
		for (i = 0; i < R->nv; i++)
			m[i] = x[i] + e[i];
		push(R, &t, m, c_mul(R, coef(a, k), c));
	}
	move(r, &t);
}

void
mpl_mul(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b)
{
	struct mpl acc, t;
	uint32_t k;

	memset(&acc, 0, sizeof acc);
	memset(&t, 0, sizeof t);
	for (k = 0; k < b->n; k++) {
		mul_term(R, &t, a, EXP(b, R, k), coef(b, k));
		mpl_add(R, &acc, &acc, &t);
	}
	mpl_free(&t);
	move(r, &acc);
}

void
mpl_scale(const struct mring *R, struct mpl *r, const struct mpl *a,
    struct zq c)
{
	uint32_t e[MPL_MAXV] = {0};

	mul_term(R, r, a, e, c_norm(R, c));
}

void
mpl_monic(const struct mring *R, struct mpl *f)
{
	if (f->n == 0 || c_is_one(coef(f, 0)))
		return;
	mpl_scale(R, f, f, c_inv(R, coef(f, 0)));
}

int
mpl_deg(const struct mring *R, const struct mpl *f)
{
	uint32_t k;
	int i, d = -1;

	for (k = 0; k < f->n; k++) {
		int s = 0;
		for (i = 0; i < R->nv; i++)
			s += (int)EXP(f, R, k)[i];
		if (s > d)
			d = s;
	}
	return d;
}

/* ---- text ---- */

static const char *pp;
static const struct mring *PR;

static void
pskip(void)
{
	while (*pp == ' ' || *pp == '\t')
		pp++;
}

static int pexpr(struct mpl *f);

static int
pbase(struct mpl *f)
{
	pskip();
	if (*pp == '(') {
		pp++;
		if (!pexpr(f))
			return 0;
		pskip();
		if (*pp != ')')
			return 0;
		pp++;
		return 1;
	}
	if (isdigit((unsigned char)*pp)) {
		struct zq c;
		zint n;
		const char *e;
		if (!z_parse(pp, &n, &e))
			return 0;
		pp = e;
		c = zq_int(n);
		mpl_const(PR, f, c);
		return 1;
	}
	if (isalpha((unsigned char)*pp) || *pp == '_') {
		const char *b = pp;
		size_t n;
		int v;
		while (isalnum((unsigned char)*pp) || *pp == '_')
			pp++;
		n = (size_t)(pp - b);
		for (v = 0; v < PR->nv; v++)
			if (strlen(PR->name[v]) == n && !strncmp(PR->name[v], b, n))
				break;
		if (v == PR->nv) {
			/* x y run together: the longest variable prefix */
			for (v = 0; v < PR->nv; v++) {
				size_t l = strlen(PR->name[v]);
				if (l < n && !strncmp(PR->name[v], b, l)) {
					pp = b + l;
					break;
				}
			}
			if (v == PR->nv)
				fail("polynomial: unknown variable \"%.*s\"", (int)n, b);
		}
		mpl_var(PR, f, v);
		return 1;
	}
	return 0;
}

static int
ppow(struct mpl *f)
{
	if (!pbase(f))
		return 0;
	pskip();
	if (*pp == '^' || (pp[0] == '*' && pp[1] == '*')) {
		zint e;
		const char *q;
		int64_t k;
		struct mpl r, b;
		pp += *pp == '^' ? 1 : 2;
		pskip();
		if (!z_parse(pp, &e, &q) || z_sign(e) < 0 || !z_fits_i64(e) ||
		    (k = z_to_i64(e)) > 100000)
			fail("polynomial: a power needs an exponent 0 .. 100000");
		pp = q;
		memset(&r, 0, sizeof r);
		memset(&b, 0, sizeof b);
		mpl_const(PR, &r, zq_int(Z_ONE));
		copy_r(PR, &b, f);
		while (k) {
			if (k & 1)
				mpl_mul(PR, &r, &r, &b);
			k >>= 1;
			if (k)
				mpl_mul(PR, &b, &b, &b);
		}
		move(f, &r);
		mpl_free(&b);
	}
	return 1;
}

static int
punary(struct mpl *f)
{
	pskip();
	if (*pp == '-') {
		pp++;
		if (!punary(f))
			return 0;
		mpl_scale(PR, f, f, zq_int(z_from_i64(-1)));
		return 1;
	}
	if (*pp == '+') {
		pp++;
		return punary(f);
	}
	return ppow(f);
}

static int
pterm(struct mpl *f)
{
	struct mpl g;

	if (!punary(f))
		return 0;
	memset(&g, 0, sizeof g);
	for (;;) {
		pskip();
		if (*pp == '*' && pp[1] != '*') {
			pp++;
			if (!punary(&g))
				return 0;
			mpl_mul(PR, f, f, &g);
		} else if (*pp == '/') {
			struct mpl d;
			pp++;
			memset(&d, 0, sizeof d);
			if (!ppow(&d))
				return 0;
			if (d.n != 1 || mpl_deg(PR, &d) != 0)
				fail("polynomial: division only by a nonzero constant");
			mpl_scale(PR, f, f, c_inv(PR, coef(&d, 0)));
			mpl_free(&d);
		} else if (isalnum((unsigned char)*pp) || *pp == '(' ||
		    *pp == '_') {
			/* juxtaposition: 3x y */
			if (!ppow(&g))
				return 0;
			mpl_mul(PR, f, f, &g);
		} else
			break;
	}
	mpl_free(&g);
	return 1;
}

static int
pexpr(struct mpl *f)
{
	struct mpl g;

	if (!pterm(f))
		return 0;
	memset(&g, 0, sizeof g);
	for (;;) {
		pskip();
		if (*pp == '+' || *pp == '-') {
			int neg = *pp == '-';
			pp++;
			if (!pterm(&g))
				return 0;
			addsub(PR, f, f, &g, neg);
		} else
			break;
	}
	mpl_free(&g);
	return 1;
}

int
mpl_parse(const struct mring *R, const char *s, struct mpl *f,
    const char **end)
{
	struct mpl t;
	int ok;

	memset(&t, 0, sizeof t);
	pp = s;
	PR = R;
	ok = pexpr(&t);
	if (end)
		*end = pp;
	if (ok)
		move(f, &t);
	else
		mpl_free(&t);
	return ok;
}

void
mpl_print(struct sbuf *b, const struct mring *R, const struct mpl *f)
{
	uint32_t k;
	int i;

	if (f->n == 0) {
		sb_puts(b, "0");
		return;
	}
	for (k = 0; k < f->n; k++) {
		struct zq c = coef(f, k);
		const uint32_t *e = EXP(f, R, k);
		int deg = 0, neg = z_sign(c.n) < 0, first = 1;
		for (i = 0; i < R->nv; i++)
			deg += e[i] != 0;
		if (k)
			sb_puts(b, neg ? " - " : " + ");
		else if (neg)
			sb_puts(b, "-");
		if (neg)
			c = zq_neg(c);
		if (!c_is_one(c) || deg == 0) {
			zq_print(b, c);
			first = 0;
		}
		for (i = 0; i < R->nv; i++) {
			if (!e[i])
				continue;
			if (!first)
				sb_putc(b, '*');
			sb_puts(b, R->name[i]);
			if (e[i] > 1)
				sb_printf(b, "^%u", e[i]);
			first = 0;
		}
	}
}

const char *
mpl_str(const struct mring *R, const struct mpl *f)
{
	static struct sbuf b[8];
	static int k;

	k = (k + 1) & 7;
	sb_clear(&b[k]);
	mpl_print(&b[k], R, f);
	return sb_str(&b[k]);
}

/* ---- division ---- */

void
mpl_divide(const struct mring *R, const struct mpl *f, const struct mpl *g,
    int k, struct mpl *q, struct mpl *rem)
{
	struct mpl p, r, t;
	uint32_t m[MPL_MAXV];
	int i, v;

	memset(&p, 0, sizeof p);
	memset(&r, 0, sizeof r);
	memset(&t, 0, sizeof t);
	copy_r(R, &p, f);
	if (q)
		for (i = 0; i < k; i++)
			mpl_free(&q[i]);
	while (p.n) {
		const uint32_t *lp = EXP(&p, R, 0);
		for (i = 0; i < k; i++)
			if (g[i].n && mono_divides(R->nv, EXP(&g[i], R, 0), lp))
				break;
		if (i < k) {
			struct zq c = c_mul(R, coef(&p, 0), c_inv(R, coef(&g[i], 0)));
			for (v = 0; v < R->nv; v++)
				m[v] = lp[v] - EXP(&g[i], R, 0)[v];
			if (q) {
				struct mpl mono;
				memset(&mono, 0, sizeof mono);
				push(R, &mono, m, c);
				mpl_add(R, &q[i], &q[i], &mono);
				mpl_free(&mono);
			}
			mul_term(R, &t, &g[i], m, c);
			mpl_sub(R, &p, &p, &t);
		} else {
			push(R, &r, lp, coef(&p, 0));
			/* drop the leading term */
			memmove(p.e, p.e + R->nv, (size_t)(p.n - 1) * R->nv *
			    sizeof *p.e);
			memmove(p.cn, p.cn + 1, (size_t)(p.n - 1) * sizeof *p.cn);
			memmove(p.cd, p.cd + 1, (size_t)(p.n - 1) * sizeof *p.cd);
			p.n--;
		}
	}
	mpl_free(&p);
	mpl_free(&t);
	if (rem)
		move(rem, &r);
	else
		mpl_free(&r);
}

/* ---- Groebner bases ---- */

struct gbw {
	const struct mring *R;
	int k;            /* input count */
	int n, cap;
	struct mpl *g;    /* basis */
	struct mpl *cof;  /* n * k cofactors */
	uint8_t *dead;
};

static void
gbw_add(struct gbw *w, struct mpl *g, struct mpl *cof)
{
	int j;

	if (w->n == w->cap) {
		w->cap = w->cap ? w->cap * 2 : 8;
		w->g = xrealloc(w->g, (size_t)w->cap * sizeof *w->g);
		w->cof = xrealloc(w->cof, (size_t)w->cap * w->k * sizeof *w->cof);
		w->dead = xrealloc(w->dead, (size_t)w->cap);
	}
	memset(&w->g[w->n], 0, sizeof w->g[w->n]);
	move(&w->g[w->n], g);
	for (j = 0; j < w->k; j++) {
		memset(&w->cof[(size_t)w->n * w->k + j], 0, sizeof *w->cof);
		move(&w->cof[(size_t)w->n * w->k + j], &cof[j]);
	}
	w->dead[w->n] = 0;
	w->n++;
}

/* h (with cofactors hc) reduced fully by the live basis elements */
static void
reduce_full(struct gbw *w, struct mpl *h, struct mpl *hc, int skip)
{
	const struct mring *R = w->R;
	struct mpl *gs, *q, rem, t;
	int *map, m = 0, i, j;

	gs = xcalloc((size_t)w->n + 1, sizeof *gs);
	q = xcalloc((size_t)w->n + 1, sizeof *q);
	map = xcalloc((size_t)w->n + 1, sizeof *map);
	for (i = 0; i < w->n; i++)
		if (!w->dead[i] && i != skip) {
			gs[m] = w->g[i];	/* shallow: read only */
			map[m++] = i;
		}
	memset(&rem, 0, sizeof rem);
	memset(&t, 0, sizeof t);
	mpl_divide(R, h, gs, m, q, &rem);
	for (i = 0; i < m; i++) {
		if (q[i].n == 0)
			continue;
		for (j = 0; j < w->k; j++) {
			mpl_mul(R, &t, &q[i], &w->cof[(size_t)map[i] * w->k + j]);
			mpl_sub(R, &hc[j], &hc[j], &t);
		}
		mpl_free(&q[i]);
	}
	move(h, &rem);
	mpl_free(&t);
	free(gs);
	free(q);
	free(map);
}

static void
lcm_mono(int nv, uint32_t *r, const uint32_t *a, const uint32_t *b)
{
	int i;

	for (i = 0; i < nv; i++)
		r[i] = a[i] > b[i] ? a[i] : b[i];
}

static void
gb_finish(struct gbw *w, struct gbasis *gb)
{
	const struct mring *R = w->R;
	int i, j, n = 0, *ord;

	/* minimal: drop g_i when another live lt divides lt(g_i) */
	for (i = 0; i < w->n; i++)
		for (j = 0; j < w->n && !w->dead[i]; j++)
			if (j != i && !w->dead[j] && mono_divides(R->nv,
			    EXP(&w->g[j], R, 0), EXP(&w->g[i], R, 0)) &&
			    (mono_cmp(R, EXP(&w->g[j], R, 0), EXP(&w->g[i], R,
			    0)) != 0 || j < i))
				w->dead[i] = 1;
	/* reduced: each element by the others (the leading terms stay) */
	for (i = 0; i < w->n; i++) {
		if (w->dead[i])
			continue;
		reduce_full(w, &w->g[i], &w->cof[(size_t)i * w->k], i);
		if (!c_is_one(coef(&w->g[i], 0)))
			fail("internal: Groebner reduction changed a leading term");
	}
	/* sorted by decreasing leading monomial */
	ord = xcalloc((size_t)w->n + 1, sizeof *ord);
	for (i = 0; i < w->n; i++)
		if (!w->dead[i])
			ord[n++] = i;
	for (i = 1; i < n; i++)
		for (j = i; j > 0 && mono_cmp(R, EXP(&w->g[ord[j - 1]], R, 0),
		    EXP(&w->g[ord[j]], R, 0)) < 0; j--) {
			int t = ord[j];
			ord[j] = ord[j - 1];
			ord[j - 1] = t;
		}
	gb->n = n;
	gb->k = w->k;
	gb->g = xcalloc((size_t)n + 1, sizeof *gb->g);
	gb->cof = xcalloc((size_t)n * w->k + 1, sizeof *gb->cof);
	for (i = 0; i < n; i++) {
		move(&gb->g[i], &w->g[ord[i]]);
		for (j = 0; j < w->k; j++)
			move(&gb->cof[(size_t)i * w->k + j],
			    &w->cof[(size_t)ord[i] * w->k + j]);
	}
	for (i = 0; i < w->n; i++) {
		mpl_free(&w->g[i]);
		for (j = 0; j < w->k; j++)
			mpl_free(&w->cof[(size_t)i * w->k + j]);
	}
	free(w->g);
	free(w->cof);
	free(w->dead);
	free(ord);
}

int
mpl_groebner(const struct mring *R, const struct mpl *f, int k,
    struct gbasis *gb)
{
	struct gbw w;
	struct mpl h, *hc, t;
	int i, j;
	/* pairs: (i, j), pending flags in a growing triangle */
	int *pi = NULL, *pj = NULL, np = 0, pcap = 0;
	uint8_t *pend = NULL;
	int pendcap = 0;

	memset(&w, 0, sizeof w);
	memset(gb, 0, sizeof *gb);
	w.R = R;
	w.k = k;
	hc = xcalloc((size_t)k + 1, sizeof *hc);
	memset(&h, 0, sizeof h);
	memset(&t, 0, sizeof t);
#define PEND(a, b) pend[(size_t)(a) * pendcap + (b)]
	for (i = 0; i < k; i++) {
		struct zq inv;
		if (f[i].n == 0)
			continue;
		copy_r(R, &h, &f[i]);
		for (j = 0; j < k; j++)
			mpl_free(&hc[j]);
		inv = c_inv(R, coef(&h, 0));
		mpl_const(R, &hc[i], inv);
		mpl_scale(R, &h, &h, inv);
		gbw_add(&w, &h, hc);
	}
	/* the pair list, with a pending flag per pair */
	pendcap = w.n * 2 + 8;
	pend = xcalloc((size_t)pendcap * pendcap, 1);
	for (i = 0; i < w.n; i++)
		for (j = i + 1; j < w.n; j++) {
			if (np == pcap) {
				pcap = pcap ? pcap * 2 : 64;
				pi = xrealloc(pi, (size_t)pcap * sizeof *pi);
				pj = xrealloc(pj, (size_t)pcap * sizeof *pj);
			}
			pi[np] = i;
			pj[np++] = j;
			PEND(i, j) = PEND(j, i) = 1;
		}
	while (np > 0) {
		uint32_t L[MPL_MAXV], mi[MPL_MAXV], mj[MPL_MAXV];
		int best = 0, a, b, kk, coprime = 1, v, chain = 0;
		/* the normal strategy: the smallest lcm first */
		for (kk = 1; kk < np; kk++) {
			uint32_t l1[MPL_MAXV], l2[MPL_MAXV];
			lcm_mono(R->nv, l1, EXP(&w.g[pi[kk]], R, 0),
			    EXP(&w.g[pj[kk]], R, 0));
			lcm_mono(R->nv, l2, EXP(&w.g[pi[best]], R, 0),
			    EXP(&w.g[pj[best]], R, 0));
			if (mono_cmp(R, l1, l2) < 0)
				best = kk;
		}
		a = pi[best];
		b = pj[best];
		pi[best] = pi[np - 1];
		pj[best] = pj[np - 1];
		np--;
		PEND(a, b) = PEND(b, a) = 0;
		gb->pairs++;
		if (w.dead[a] || w.dead[b]) {
			gb->skipped++;
			continue;
		}
		lcm_mono(R->nv, L, EXP(&w.g[a], R, 0), EXP(&w.g[b], R, 0));
		for (v = 0; v < R->nv; v++)
			if (EXP(&w.g[a], R, 0)[v] && EXP(&w.g[b], R, 0)[v])
				coprime = 0;
		/* Buchberger's first criterion: coprime leading terms */
		if (coprime) {
			gb->skipped++;
			continue;
		}
		/* the chain criterion: some g_c with lt | lcm, and the pairs
		 * (a, c), (b, c) already treated */
		for (kk = 0; kk < w.n && !chain; kk++)
			if (kk != a && kk != b && !w.dead[kk] && !PEND(a, kk) &&
			    !PEND(b, kk) && mono_divides(R->nv, EXP(&w.g[kk], R, 0),
			    L))
				chain = 1;
		if (chain) {
			gb->skipped++;
			continue;
		}
		/* S = (L / lt a) g_a - (L / lt b) g_b, with cofactors */
		for (v = 0; v < R->nv; v++) {
			mi[v] = L[v] - EXP(&w.g[a], R, 0)[v];
			mj[v] = L[v] - EXP(&w.g[b], R, 0)[v];
		}
		mul_term(R, &h, &w.g[a], mi, zq_int(Z_ONE));
		mul_term(R, &t, &w.g[b], mj, zq_int(Z_ONE));
		mpl_sub(R, &h, &h, &t);
		for (j = 0; j < k; j++) {
			mul_term(R, &hc[j], &w.cof[(size_t)a * k + j], mi,
			    zq_int(Z_ONE));
			mul_term(R, &t, &w.cof[(size_t)b * k + j], mj, zq_int(Z_ONE));
			mpl_sub(R, &hc[j], &hc[j], &t);
		}
		gb->reductions++;
		reduce_full(&w, &h, hc, -1);
		if (h.n == 0)
			continue;
		{
			struct zq inv = c_inv(R, coef(&h, 0));
			int nn;
			mpl_scale(R, &h, &h, inv);
			for (j = 0; j < k; j++)
				mpl_scale(R, &hc[j], &hc[j], inv);
			gbw_add(&w, &h, hc);
			nn = w.n - 1;
			if (w.n > pendcap) {
				uint8_t *np2;
				int nc = w.n * 2 + 8, q;
				np2 = xcalloc((size_t)nc * nc, 1);
				for (q = 0; q < pendcap; q++)
					memcpy(np2 + (size_t)q * nc, pend + (size_t)q *
					    pendcap, (size_t)pendcap);
				free(pend);
				pend = np2;
				pendcap = nc;
			}
			for (i = 0; i < nn; i++) {
				if (w.dead[i])
					continue;
				if (np == pcap) {
					pcap = pcap ? pcap * 2 : 64;
					pi = xrealloc(pi, (size_t)pcap * sizeof *pi);
					pj = xrealloc(pj, (size_t)pcap * sizeof *pj);
				}
				pi[np] = i;
				pj[np++] = nn;
				PEND(i, nn) = PEND(nn, i) = 1;
			}
			if (w.n > 4000)
				fail("Groebner basis: more than 4000 elements; giving up");
		}
	}
#undef PEND
	free(pi);
	free(pj);
	free(pend);
	mpl_free(&h);
	mpl_free(&t);
	for (j = 0; j < k; j++)
		mpl_free(&hc[j]);
	free(hc);
	gb_finish(&w, gb);
	if (!gb_check(R, f, gb))
		fail("internal: Groebner basis check failed");
	return gb->n;
}

void
gb_free(struct gbasis *gb)
{
	int i;

	for (i = 0; i < gb->n; i++)
		mpl_free(&gb->g[i]);
	for (i = 0; i < gb->n * gb->k; i++)
		mpl_free(&gb->cof[i]);
	free(gb->g);
	free(gb->cof);
	memset(gb, 0, sizeof *gb);
}

static int
reduces_to_zero(const struct mring *R, const struct gbasis *gb,
    const struct mpl *h)
{
	struct mpl r;
	int z;

	memset(&r, 0, sizeof r);
	mpl_divide(R, h, gb->g, gb->n, NULL, &r);
	z = r.n == 0;
	mpl_free(&r);
	return z;
}

int
gb_check(const struct mring *R, const struct mpl *f, const struct gbasis *gb)
{
	struct mpl s, t;
	int i, j, ok = 1, v;

	memset(&s, 0, sizeof s);
	memset(&t, 0, sizeof t);
	/* the cofactors: g_i = sum_j c_ij f_j */
	for (i = 0; i < gb->n && ok; i++) {
		struct mpl acc;
		memset(&acc, 0, sizeof acc);
		for (j = 0; j < gb->k; j++) {
			mpl_mul(R, &t, &gb->cof[(size_t)i * gb->k + j], &f[j]);
			mpl_add(R, &acc, &acc, &t);
		}
		if (!mpl_eq(R, &acc, &gb->g[i]))
			ok = 0;
		mpl_free(&acc);
	}
	/* every input reduces to 0 */
	for (j = 0; j < gb->k && ok; j++)
		if (!reduces_to_zero(R, gb, &f[j]))
			ok = 0;
	/* Buchberger's criterion on the final basis, every pair */
	for (i = 0; i < gb->n && ok; i++)
		for (j = i + 1; j < gb->n && ok; j++) {
			uint32_t L[MPL_MAXV], mi[MPL_MAXV], mj[MPL_MAXV];
			lcm_mono(R->nv, L, EXP(&gb->g[i], R, 0), EXP(&gb->g[j], R, 0));
			for (v = 0; v < R->nv; v++) {
				mi[v] = L[v] - EXP(&gb->g[i], R, 0)[v];
				mj[v] = L[v] - EXP(&gb->g[j], R, 0)[v];
			}
			mul_term(R, &s, &gb->g[i], mi, c_inv(R, coef(&gb->g[i], 0)));
			mul_term(R, &t, &gb->g[j], mj, c_inv(R, coef(&gb->g[j], 0)));
			mpl_sub(R, &s, &s, &t);
			if (!reduces_to_zero(R, gb, &s))
				ok = 0;
		}
	mpl_free(&s);
	mpl_free(&t);
	return ok;
}

int
gb_member(const struct mring *R, const struct gbasis *gb, const struct mpl *f,
    const struct mpl *h, struct mpl *cof)
{
	struct mpl *q, r, t, acc;
	int i, j, ok;

	q = xcalloc((size_t)gb->n + 1, sizeof *q);
	memset(&r, 0, sizeof r);
	memset(&t, 0, sizeof t);
	memset(&acc, 0, sizeof acc);
	mpl_divide(R, h, gb->g, gb->n, q, &r);
	ok = r.n == 0;
	if (ok && cof) {
		for (j = 0; j < gb->k; j++) {
			struct mpl c;
			memset(&c, 0, sizeof c);
			for (i = 0; i < gb->n; i++) {
				mpl_mul(R, &t, &q[i], &gb->cof[(size_t)i * gb->k + j]);
				mpl_add(R, &c, &c, &t);
			}
			mpl_free(&cof[j]);
			move(&cof[j], &c);
		}
		/* the certificate: h = sum cof_j f_j */
		for (j = 0; j < gb->k; j++) {
			mpl_mul(R, &t, &cof[j], &f[j]);
			mpl_add(R, &acc, &acc, &t);
		}
		if (!mpl_eq(R, &acc, h))
			fail("internal: ideal membership certificate failed");
	}
	for (i = 0; i < gb->n; i++)
		mpl_free(&q[i]);
	free(q);
	mpl_free(&r);
	mpl_free(&t);
	mpl_free(&acc);
	return ok;
}

int
gb_ideal_le(const struct mring *R, const struct mpl *f, int kf,
    const struct gbasis *gbh, const struct mpl *h)
{
	int i;
	struct mpl *cof = xcalloc((size_t)gbh->k + 1, sizeof *cof);

	for (i = 0; i < kf; i++)
		if (!gb_member(R, gbh, h, &f[i], cof)) {
			int j;
			for (j = 0; j < gbh->k; j++)
				mpl_free(&cof[j]);
			free(cof);
			return 0;
		}
	for (i = 0; i < gbh->k; i++)
		mpl_free(&cof[i]);
	free(cof);
	return 1;
}

int
gb_ideal_eq(const struct mring *R, const struct mpl *f, int kf,
    const struct mpl *h, int kh)
{
	struct gbasis gf, gh;
	int ok;

	mpl_groebner(R, f, kf, &gf);
	mpl_groebner(R, h, kh, &gh);
	ok = gb_ideal_le(R, f, kf, &gh, h) && gb_ideal_le(R, h, kh, &gf, f);
	gb_free(&gf);
	gb_free(&gh);
	return ok;
}

/* ---- matrices over F[x] ---- */

void
pm_init(struct pmat *m, int r, int c)
{
	m->r = r;
	m->c = c;
	m->a = xcalloc((size_t)r * c + 1, sizeof *m->a);
}

void
pm_free(struct pmat *m)
{
	int i;

	for (i = 0; i < m->r * m->c; i++)
		mpl_free(&m->a[i]);
	free(m->a);
	memset(m, 0, sizeof *m);
}

void
pm_mul(const struct mring *R, struct pmat *r, const struct pmat *a,
    const struct pmat *b)
{
	struct pmat t;
	struct mpl p;
	int i, j, k;

	if (a->c != b->r)
		fail("internal: polynomial matrix product sizes");
	pm_init(&t, a->r, b->c);
	memset(&p, 0, sizeof p);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < b->c; j++)
			for (k = 0; k < a->c; k++) {
				mpl_mul(R, &p, PM(a, i, k), PM(b, k, j));
				mpl_add(R, PM(&t, i, j), PM(&t, i, j), &p);
			}
	mpl_free(&p);
	if (r->a)
		pm_free(r);
	*r = t;
}

static void
pm_identity(const struct mring *R, struct pmat *m, int n)
{
	int i;

	pm_init(m, n, n);
	for (i = 0; i < n; i++)
		mpl_const(R, PM(m, i, i), zq_int(Z_ONE));
}

/* row i -= q row k (on a matrix) */
static void
prow_sub(const struct mring *R, struct pmat *m, int i, int k,
    const struct mpl *q)
{
	struct mpl t;
	int j;

	memset(&t, 0, sizeof t);
	for (j = 0; j < m->c; j++) {
		mpl_mul(R, &t, q, PM(m, k, j));
		mpl_sub(R, PM(m, i, j), PM(m, i, j), &t);
	}
	mpl_free(&t);
}

static void
pcol_sub(const struct mring *R, struct pmat *m, int i, int k,
    const struct mpl *q)
{
	struct mpl t;
	int j;

	memset(&t, 0, sizeof t);
	for (j = 0; j < m->r; j++) {
		mpl_mul(R, &t, PM(m, j, k), q);
		mpl_sub(R, PM(m, j, i), PM(m, j, i), &t);
	}
	mpl_free(&t);
}

static void
prow_swap(struct pmat *m, int a, int b)
{
	int j;

	for (j = 0; j < m->c && a != b; j++) {
		struct mpl t = *PM(m, a, j);
		*PM(m, a, j) = *PM(m, b, j);
		*PM(m, b, j) = t;
	}
}

static void
pcol_swap(struct pmat *m, int a, int b)
{
	int j;

	for (j = 0; j < m->r && a != b; j++) {
		struct mpl t = *PM(m, j, a);
		*PM(m, j, a) = *PM(m, j, b);
		*PM(m, j, b) = t;
	}
}

static void
prow_scale(const struct mring *R, struct pmat *m, int i, struct zq c)
{
	int j;

	for (j = 0; j < m->c; j++)
		mpl_scale(R, PM(m, i, j), PM(m, i, j), c);
}

static void
pcol_scale(const struct mring *R, struct pmat *m, int i, struct zq c)
{
	int j;

	for (j = 0; j < m->r; j++)
		mpl_scale(R, PM(m, j, i), PM(m, j, i), c);
}

int
pm_snf(const struct mring *R, const struct pmat *a, struct pmat *D,
    struct pmat *U, struct pmat *V)
{
	struct pmat d, u, v, ui, vi, chk, chk2;
	struct mpl q, rem, mq;
	int k, i, j, n = a->r < a->c ? a->r : a->c, rank = 0;

	if (R->nv != 1)
		fail("internal: matrices over F[x] need one variable");
	pm_init(&d, a->r, a->c);
	for (i = 0; i < a->r * a->c; i++)
		copy_r(R, &d.a[i], &a->a[i]);
	pm_identity(R, &u, a->r);
	pm_identity(R, &ui, a->r);
	pm_identity(R, &v, a->c);
	pm_identity(R, &vi, a->c);
	memset(&q, 0, sizeof q);
	memset(&rem, 0, sizeof rem);
	memset(&mq, 0, sizeof mq);
	for (k = 0; k < n; k++) {
		int pi = -1, pj = -1;
		for (;;) {
			int dirty = 0;
			pi = pj = -1;
			for (i = k; i < d.r; i++)
				for (j = k; j < d.c; j++)
					if (PM(&d, i, j)->n && (pi < 0 || mpl_deg(R,
					    PM(&d, i, j)) < mpl_deg(R, PM(&d, pi, pj)))) {
						pi = i;
						pj = j;
					}
			if (pi < 0)
				break;
			/* the pivot to (k, k): row swap (ui: column swap) and column
			 * swap (vi: row swap) */
			prow_swap(&d, k, pi);
			prow_swap(&u, k, pi);
			pcol_swap(&ui, k, pi);
			pcol_swap(&d, k, pj);
			pcol_swap(&v, k, pj);
			prow_swap(&vi, k, pj);
			/* reduce the column and the row by division */
			for (i = k + 1; i < d.r; i++) {
				if (!PM(&d, i, k)->n)
					continue;
				mpl_divide(R, PM(&d, i, k), PM(&d, k, k), 1, &q, &rem);
				prow_sub(R, &d, i, k, &q);
				prow_sub(R, &u, i, k, &q);
				/* ui <- ui E^-1: column k += q column i */
				mpl_scale(R, &mq, &q, zq_int(z_from_i64(-1)));
				pcol_sub(R, &ui, k, i, &mq);
				if (rem.n)
					dirty = 1;
			}
			for (j = k + 1; j < d.c; j++) {
				if (!PM(&d, k, j)->n)
					continue;
				mpl_divide(R, PM(&d, k, j), PM(&d, k, k), 1, &q, &rem);
				pcol_sub(R, &d, j, k, &q);
				pcol_sub(R, &v, j, k, &q);
				/* vi <- F^-1 vi: row k += q row j */
				mpl_scale(R, &mq, &q, zq_int(z_from_i64(-1)));
				prow_sub(R, &vi, k, j, &mq);
				if (rem.n)
					dirty = 1;
			}
			if (dirty)
				continue;
			/* divisibility of the rest by the pivot */
			for (i = k + 1; i < d.r && !dirty; i++)
				for (j = k + 1; j < d.c && !dirty; j++) {
					if (!PM(&d, i, j)->n)
						continue;
					mpl_divide(R, PM(&d, i, j), PM(&d, k, k), 1, &q,
					    &rem);
					if (rem.n) {
						/* row k += row i */
						struct mpl one;
						memset(&one, 0, sizeof one);
						mpl_const(R, &one, zq_int(z_from_i64(-1)));
						prow_sub(R, &d, k, i, &one);
						prow_sub(R, &u, k, i, &one);
						mpl_const(R, &one, zq_int(Z_ONE));
						pcol_sub(R, &ui, i, k, &one);
						mpl_free(&one);
						dirty = 1;
					}
				}
			if (!dirty)
				break;
		}
		if (pi < 0)
			break;
		/* monic pivot: scale row k by 1/lc */
		{
			struct zq lc, inv;
			lc.n = PM(&d, k, k)->cn[0];
			lc.d = PM(&d, k, k)->cd[0];
			inv = c_inv(R, lc);
			prow_scale(R, &d, k, inv);
			prow_scale(R, &u, k, inv);
			pcol_scale(R, &ui, k, lc);
		}
		rank++;
	}
	mpl_free(&q);
	mpl_free(&rem);
	mpl_free(&mq);
	/* the certificate: U A V = D, U Ui = I, V Vi = I */
	memset(&chk, 0, sizeof chk);
	memset(&chk2, 0, sizeof chk2);
	pm_mul(R, &chk, &u, a);
	pm_mul(R, &chk2, &chk, &v);
	for (i = 0; i < d.r * d.c; i++)
		if (!mpl_eq(R, &chk2.a[i], &d.a[i]))
			fail("internal: Smith form over F[x]: U A V != D");
	pm_mul(R, &chk, &u, &ui);
	for (i = 0; i < chk.r; i++)
		for (j = 0; j < chk.c; j++)
			if (PM(&chk, i, j)->n != (i == j) || (i == j &&
			    (mpl_deg(R, PM(&chk, i, j)) != 0 || !c_is_one(coef(PM(&chk,
			    i, j), 0)))))
				fail("internal: Smith form over F[x]: U not invertible");
	pm_mul(R, &chk, &v, &vi);
	for (i = 0; i < chk.r; i++)
		for (j = 0; j < chk.c; j++)
			if (PM(&chk, i, j)->n != (i == j) || (i == j &&
			    (mpl_deg(R, PM(&chk, i, j)) != 0 || !c_is_one(coef(PM(&chk,
			    i, j), 0)))))
				fail("internal: Smith form over F[x]: V not invertible");
	pm_free(&chk);
	pm_free(&chk2);
	pm_free(&ui);
	pm_free(&vi);
	if (D) {
		if (D->a)
			pm_free(D);
		*D = d;
	} else
		pm_free(&d);
	if (U) {
		if (U->a)
			pm_free(U);
		*U = u;
	} else
		pm_free(&u);
	if (V) {
		if (V->a)
			pm_free(V);
		*V = v;
	} else
		pm_free(&v);
	return rank;
}

int
pm_charmodule(const struct mring *R, const struct qmat *m, struct mpl *inv,
    int max)
{
	struct pmat A, D;
	int n = m->r, i, j, cnt = 0;

	if (m->r != m->c)
		fail("the matrix is %dx%d, not square", m->r, m->c);
	pm_init(&A, n, n);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {
			struct mpl x;
			mpl_const(R, PM(&A, i, j), zq_neg(qm_get(m, i, j)));
			if (i == j) {
				memset(&x, 0, sizeof x);
				mpl_var(R, &x, 0);
				mpl_add(R, PM(&A, i, j), PM(&A, i, j), &x);
				mpl_free(&x);
			}
		}
	memset(&D, 0, sizeof D);
	pm_snf(R, &A, &D, NULL, NULL);
	for (i = 0; i < n; i++)
		if (mpl_deg(R, PM(&D, i, i)) >= 1) {
			if (cnt >= max)
				fail("too many invariant factors");
			memset(&inv[cnt], 0, sizeof inv[cnt]);
			copy_r(R, &inv[cnt++], PM(&D, i, i));
		}
	pm_free(&A);
	pm_free(&D);
	return cnt;
}
