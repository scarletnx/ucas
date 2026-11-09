/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecpoly.c - polynomials over F_p (px): coefficient i of a at a->c + i*n
 * (n = F->n limbs, Montgomery form), one flat array per polynomial.
 * Karatsuba multiplication over a schoolbook base that accumulates raw
 * double-width products and reduces once per coefficient; reduction mod a
 * fixed modulus through a Newton inverse of its reversal (pxmod); gcd,
 * inverses, powers, roots (Cantor-Zassenhaus).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

#define CO(a, i) ((a)->c + (size_t)(i) * (size_t)F->n)

void
px_init(struct px *a)
{
	memset(a, 0, sizeof *a);
	a->d = -1;
}

void
px_free(struct px *a)
{
	free(a->c);
	px_init(a);
}

void
px_reserve(const struct mf *F, struct px *a, int d)
{
	if (d + 1 > a->cap) {
		int nc = a->cap ? a->cap : 8;
		while (nc < d + 1)
			nc *= 2;
		a->c = xrealloc(a->c, sizeof(u64) * (size_t)F->n * (size_t)nc);
		memset(a->c + (size_t)a->cap * F->n, 0, sizeof(u64) *
		    (size_t)F->n * (size_t)(nc - a->cap));
		a->cap = nc;
	}
}

void
px_norm(const struct mf *F, struct px *a)
{
	while (a->d >= 0 && mf_is_zero(F, CO(a, a->d)))
		a->d--;
}

void
px_zero(const struct mf *F, struct px *a)
{
	a->d = -1;
}

void
px_set_c(const struct mf *F, struct px *a, const u64 *c)
{
	px_reserve(F, a, 0);
	mf_copy(F, CO(a, 0), c);
	a->d = 0;
	px_norm(F, a);
}

void
px_set_u64(const struct mf *F, struct px *a, int64_t v)
{
	u64 t[FW];

	mf_from_i64(F, t, v);
	px_set_c(F, a, t);
}

void
px_set_x(const struct mf *F, struct px *a)
{
	px_reserve(F, a, 1);
	mf_zero(F, CO(a, 0));
	mf_one(F, CO(a, 1));
	a->d = 1;
}

void
px_copy(const struct mf *F, struct px *r, const struct px *a)
{
	if (r == a)
		return;
	px_reserve(F, r, a->d < 0 ? 0 : a->d);
	if (a->d >= 0)
		memcpy(r->c, a->c, sizeof(u64) * (size_t)F->n *
		    (size_t)(a->d + 1));
	r->d = a->d;
}

u64 *
px_coef(const struct mf *F, const struct px *a, int i)
{
	return CO(a, i);
}

int
px_eq(const struct mf *F, const struct px *a, const struct px *b)
{
	int i;

	if (a->d != b->d)
		return 0;
	for (i = 0; i <= a->d; i++)
		if (!mf_eq(F, CO(a, i), CO(b, i)))
			return 0;
	return 1;
}

int
px_is_one(const struct mf *F, const struct px *a)
{
	return a->d == 0 && mf_is_one(F, CO(a, 0));
}

void
px_add(const struct mf *F, struct px *r, const struct px *a,
    const struct px *b)
{
	int i, d = a->d > b->d ? a->d : b->d;

	px_reserve(F, r, d < 0 ? 0 : d);
	for (i = 0; i <= d; i++) {
		if (i <= a->d && i <= b->d)
			mf_add(F, CO(r, i), CO(a, i), CO(b, i));
		else if (i <= a->d)
			mf_copy(F, CO(r, i), CO(a, i));
		else
			mf_copy(F, CO(r, i), CO(b, i));
	}
	r->d = d;
	px_norm(F, r);
}

void
px_sub(const struct mf *F, struct px *r, const struct px *a,
    const struct px *b)
{
	int i, d = a->d > b->d ? a->d : b->d;

	px_reserve(F, r, d < 0 ? 0 : d);
	for (i = 0; i <= d; i++) {
		if (i <= a->d && i <= b->d)
			mf_sub(F, CO(r, i), CO(a, i), CO(b, i));
		else if (i <= a->d)
			mf_copy(F, CO(r, i), CO(a, i));
		else
			mf_neg(F, CO(r, i), CO(b, i));
	}
	r->d = d;
	px_norm(F, r);
}

void
px_scale(const struct mf *F, struct px *r, const struct px *a, const u64 *c)
{
	int i;

	px_reserve(F, r, a->d < 0 ? 0 : a->d);
	for (i = 0; i <= a->d; i++)
		mf_mul(F, CO(r, i), CO(a, i), c);
	r->d = a->d;
	px_norm(F, r);
}

/* ---- multiplication ---- */

/* r[0 .. na+nb-2] = a * b, raw coefficient arrays (stride n), schoolbook
 * with one reduction per output coefficient */
static void
mul_base(const struct mf *F, u64 *r, const u64 *a, int na, const u64 *b,
    int nb)
{
	int n = F->n, i, j, k;
	u64 acc[2 * FW + 2], red[FW];

	for (k = 0; k < na + nb - 1; k++) {
		int lo = k - (nb - 1) > 0 ? k - (nb - 1) : 0;
		int hi = k < na - 1 ? k : na - 1;
		int len = 2 * n + 1;
		memset(acc, 0, sizeof(u64) * (size_t)len);
		for (i = lo; i <= hi; i++) {
			const u64 *x = a + (size_t)i * n, *y = b + (size_t)(k - i) * n;
			/* acc += x * y */
			for (j = 0; j < n; j++) {
				u64 c = 0;
				int m;
				if (x[j] == 0)
					continue;
				for (m = 0; m < n; m++) {
					u128 s = (u128)x[j] * y[m] + acc[j + m] + c;
					acc[j + m] = (u64)s;
					c = (u64)(s >> 64);
				}
				for (m = j + n; c && m < len; m++) {
					u128 s = (u128)acc[m] + c;
					acc[m] = (u64)s;
					c = (u64)(s >> 64);
				}
			}
		}
		/* acc = sum of (aR)(bR): reduce mod p, then one R^-1 */
		l_divmod(NULL, red, acc, len, F->p, n);
		{
			u64 one[FW];
			memset(one, 0, sizeof one);
			one[0] = 1;
			mf_mul(F, r + (size_t)k * n, red, one);
		}
	}
}

#define KARA 24

static void
addv(const struct mf *F, u64 *r, const u64 *a, int na, const u64 *b, int nb)
{
	int i, n = F->n, m = na > nb ? na : nb;

	for (i = 0; i < m; i++) {
		if (i < na && i < nb)
			mf_add(F, r + (size_t)i * n, a + (size_t)i * n,
			    b + (size_t)i * n);
		else if (i < na)
			mf_copy(F, r + (size_t)i * n, a + (size_t)i * n);
		else
			mf_copy(F, r + (size_t)i * n, b + (size_t)i * n);
	}
}

/* r (2m - 1 coefficients) = a * b, both m coefficients; scratch t */
static void
kara(const struct mf *F, u64 *r, const u64 *a, const u64 *b, int m, u64 *t)
{
	int n = F->n, h, l, i;
	u64 *s1, *s2, *mid, *t2;

	if (m <= KARA) {
		mul_base(F, r, a, m, b, m);
		return;
	}
	h = m / 2;
	l = m - h;   /* low part l >= high part h */
	/* r = a0 b0 + x^l ((a0 + a1)(b0 + b1) - a0 b0 - a1 b1) + x^2l a1 b1 */
	s1 = t;
	s2 = s1 + (size_t)l * n;
	mid = s2 + (size_t)l * n;
	t2 = mid + (size_t)(2 * l - 1) * n;
	memset(r, 0, sizeof(u64) * (size_t)n * (size_t)(2 * m - 1));
	kara(F, r, a, b, l, t2);                         /* a0 b0 at 0 */
	if (h == l)
		kara(F, r + (size_t)(2 * l) * n, a + (size_t)l * n,
		    b + (size_t)l * n, h, t2);
	else {
		/* pad the high halves to l */
		u64 *ah = t2, *bh = ah + (size_t)l * n, *t3 = bh + (size_t)l * n;
		memset(ah, 0, sizeof(u64) * (size_t)n * (size_t)l * 2);
		memcpy(ah, a + (size_t)l * n, sizeof(u64) * (size_t)n * h);
		memcpy(bh, b + (size_t)l * n, sizeof(u64) * (size_t)n * h);
		kara(F, mid, ah, bh, l, t3);
		memcpy(r + (size_t)(2 * l) * n, mid, sizeof(u64) * (size_t)n *
		    (size_t)(2 * h - 1));
	}
	addv(F, s1, a, l, a + (size_t)l * n, h);
	addv(F, s2, b, l, b + (size_t)l * n, h);
	kara(F, mid, s1, s2, l, t2);
	for (i = 0; i < 2 * l - 1; i++)
		mf_sub(F, mid + (size_t)i * n, mid + (size_t)i * n,
		    r + (size_t)i * n);
	for (i = 0; i < 2 * h - 1; i++)
		mf_sub(F, mid + (size_t)i * n, mid + (size_t)i * n,
		    r + (size_t)(2 * l + i) * n);
	for (i = 0; i < 2 * l - 1; i++)
		mf_add(F, r + (size_t)(l + i) * n, r + (size_t)(l + i) * n,
		    mid + (size_t)i * n);
}

void
px_mul(const struct mf *F, struct px *r, const struct px *a,
    const struct px *b)
{
	int n = F->n, na = a->d + 1, nb = b->d + 1, d;
	u64 *out;

	if (na <= 0 || nb <= 0) {
		r->d = -1;
		return;
	}
	d = na + nb - 2;
	out = xmalloc(sizeof(u64) * (size_t)n * (size_t)(d + 1));
	if (na <= KARA || nb <= KARA)
		mul_base(F, out, a->c, na, b->c, nb);
	else {
		int m = na > nb ? na : nb;
		u64 *A = xcalloc((size_t)n * (size_t)m * 2, sizeof(u64));
		u64 *B = A + (size_t)n * m;
		u64 *R = xmalloc(sizeof(u64) * (size_t)n * (size_t)(2 * m));
		u64 *T = xmalloc(sizeof(u64) * (size_t)n * (size_t)(8 * m + 64));
		memcpy(A, a->c, sizeof(u64) * (size_t)n * na);
		memcpy(B, b->c, sizeof(u64) * (size_t)n * nb);
		kara(F, R, A, B, m, T);
		memcpy(out, R, sizeof(u64) * (size_t)n * (size_t)(d + 1));
		free(A);
		free(R);
		free(T);
	}
	px_reserve(F, r, d);
	memcpy(r->c, out, sizeof(u64) * (size_t)n * (size_t)(d + 1));
	r->d = d;
	free(out);
	px_norm(F, r);
}

void
px_sqr(const struct mf *F, struct px *r, const struct px *a)
{
	px_mul(F, r, a, a);
}

/* ---- division ---- */

/* q, r = a divmod b (b != 0); q or r may be NULL */
void
px_divmod(const struct mf *F, struct px *q, struct px *r, const struct px *a,
    const struct px *b)
{
	struct px t;
	u64 il[FW], c[FW], x[FW];
	int i, j;

	if (b->d < 0)
		fail("ecff: polynomial division by zero");
	int da = a->d, db = b->d;
	px_init(&t);
	px_copy(F, &t, a);
	if (q) {
		px_reserve(F, q, da - db < 0 ? 0 : da - db);
		q->d = da - db;
		for (i = 0; i <= q->d; i++)
			mf_zero(F, CO(q, i));
	}
	if (!mf_inv(F, il, CO(b, db), NULL))
		fail("ecff: leading coefficient not invertible");
	for (i = da; i >= db; i--) {
		if (mf_is_zero(F, CO(&t, i)))
			continue;
		mf_mul(F, c, CO(&t, i), il);
		if (q)
			mf_copy(F, CO(q, i - db), c);
		for (j = 0; j <= db; j++) {
			mf_mul(F, x, c, CO(b, j));
			mf_sub(F, CO(&t, i - db + j), CO(&t, i - db + j), x);
		}
	}
	if (q)
		px_norm(F, q);
	if (r) {
		t.d = t.d < db - 1 ? t.d : db - 1;
		px_norm(F, &t);
		px_copy(F, r, &t);
	}
	px_free(&t);
}

void
px_rem(const struct mf *F, struct px *r, const struct px *a,
    const struct px *b)
{
	px_divmod(F, NULL, r, a, b);
}

/* truncate to degree < k */
static void
px_trunc(const struct mf *F, struct px *a, int k)
{
	if (a->d >= k)
		a->d = k - 1;
	px_norm(F, a);
}

static void
px_reverse(const struct mf *F, struct px *r, const struct px *a, int d)
{
	struct px t;
	int i;

	px_init(&t);
	px_reserve(F, &t, d < 0 ? 0 : d);
	for (i = 0; i <= d; i++) {
		if (d - i <= a->d)
			mf_copy(F, CO(&t, i), CO(a, d - i));
		else
			mf_zero(F, CO(&t, i));
	}
	t.d = d;
	px_norm(F, &t);
	px_copy(F, r, &t);
	px_free(&t);
}

/* the inverse of a (a(0) != 0) mod x^k by Newton */
static void
px_inv_series(const struct mf *F, struct px *r, const struct px *a, int k)
{
	struct px g, t, u;
	u64 c[FW];
	int m = 1;

	px_init(&g);
	px_init(&t);
	px_init(&u);
	if (!mf_inv(F, c, CO(a, 0), NULL))
		fail("ecff: series not invertible");
	px_set_c(F, &g, c);
	while (m < k) {
		m *= 2;
		/* g = g (2 - a g) mod x^m */
		px_copy(F, &t, a);
		px_trunc(F, &t, m);
		px_mul(F, &u, &t, &g);
		px_trunc(F, &u, m);
		{
			struct px two;
			px_init(&two);
			px_set_u64(F, &two, 2);
			px_sub(F, &u, &two, &u);
			px_free(&two);
		}
		px_mul(F, &t, &g, &u);
		px_trunc(F, &t, m);
		px_copy(F, &g, &t);
	}
	px_trunc(F, &g, k);
	px_copy(F, r, &g);
	px_free(&g);
	px_free(&t);
	px_free(&u);
}

void
pxmod_init(const struct mf *F, struct pxmod *M, const struct px *m)
{
	struct px rv;
	int d = m->d;

	px_init(&M->m);
	px_init(&M->inv);
	px_copy(F, &M->m, m);
	if (d < 1)
		return;
	px_init(&rv);
	px_reverse(F, &rv, m, d);
	px_inv_series(F, &M->inv, &rv, d);
	px_free(&rv);
}

void
pxmod_free(struct pxmod *M)
{
	px_free(&M->m);
	px_free(&M->inv);
}

/* r = a mod M (deg a <= 2 deg m - 2) */
void
px_reduce(const struct mf *F, const struct pxmod *M, struct px *r,
    const struct px *a)
{
	struct px ar, q, t;
	int d = M->m.d, k;

	if (a->d < d) {
		px_copy(F, r, a);
		return;
	}
	if (d < 1 || a->d > 2 * d - 2 || d < 2 * KARA) {
		px_rem(F, r, a, &M->m);
		return;
	}
	k = a->d - d + 1;   /* quotient size */
	px_init(&ar);
	px_init(&q);
	px_init(&t);
	px_reverse(F, &ar, a, a->d);
	px_trunc(F, &ar, k);
	px_mul(F, &q, &ar, &M->inv);
	px_trunc(F, &q, k);
	px_reverse(F, &q, &q, k - 1);
	px_mul(F, &t, &q, &M->m);
	px_sub(F, r, a, &t);
	px_free(&ar);
	px_free(&q);
	px_free(&t);
	if (r->d >= d)
		px_rem(F, r, r, &M->m);
}

void
px_mulmod(const struct mf *F, const struct pxmod *M, struct px *r,
    const struct px *a, const struct px *b)
{
	struct px t;

	px_init(&t);
	px_mul(F, &t, a, b);
	px_reduce(F, M, r, &t);
	px_free(&t);
}

void
px_powmod(const struct mf *F, const struct pxmod *M, struct px *r,
    const struct px *a, const nn *e)
{
	struct px x, b;
	int i;

	px_init(&x);
	px_init(&b);
	px_set_u64(F, &x, 1);
	px_reduce(F, M, &b, a);
	for (i = nn_bits(e) - 1; i >= 0; i--) {
		px_mulmod(F, M, &x, &x, &x);
		if (nn_bit(e, i))
			px_mulmod(F, M, &x, &x, &b);
	}
	px_copy(F, r, &x);
	px_free(&x);
	px_free(&b);
}

void
px_monic(const struct mf *F, struct px *a)
{
	u64 il[FW];

	if (a->d < 0)
		return;
	mf_inv(F, il, CO(a, a->d), NULL);
	px_scale(F, a, a, il);
}

void
px_gcd(const struct mf *F, struct px *g, const struct px *a,
    const struct px *b)
{
	struct px x, y, t;

	px_init(&x);
	px_init(&y);
	px_init(&t);
	px_copy(F, &x, a);
	px_copy(F, &y, b);
	while (y.d >= 0) {
		px_rem(F, &t, &x, &y);
		px_copy(F, &x, &y);
		px_copy(F, &y, &t);
	}
	px_monic(F, &x);
	px_copy(F, g, &x);
	px_free(&x);
	px_free(&y);
	px_free(&t);
}

/* 1 and r = a^-1 mod m, or 0 and g = gcd(a, m) (monic) */
int
px_invmod(const struct mf *F, struct px *r, const struct px *a,
    const struct px *m, struct px *g)
{
	struct px r0, r1, s0, s1, q, t, u;
	int ok;

	px_init(&r0);
	px_init(&r1);
	px_init(&s0);
	px_init(&s1);
	px_init(&q);
	px_init(&t);
	px_init(&u);
	px_copy(F, &r0, m);
	px_rem(F, &r1, a, m);
	px_set_u64(F, &s1, 1);
	while (r1.d > 0) {
		px_divmod(F, &q, &t, &r0, &r1);
		px_copy(F, &r0, &r1);
		px_copy(F, &r1, &t);
		px_mul(F, &u, &q, &s1);
		px_sub(F, &u, &s0, &u);
		px_copy(F, &s0, &s1);
		px_copy(F, &s1, &u);
	}
	if (r1.d == 0) {
		u64 il[FW];
		mf_inv(F, il, CO(&r1, 0), NULL);
		px_scale(F, &s1, &s1, il);
		px_rem(F, r, &s1, m);
		ok = 1;
	} else {
		if (g) {
			px_copy(F, g, &r0);
			px_monic(F, g);
		}
		ok = 0;
	}
	px_free(&r0);
	px_free(&r1);
	px_free(&s0);
	px_free(&s1);
	px_free(&q);
	px_free(&t);
	px_free(&u);
	return ok;
}

void
px_deriv(const struct mf *F, struct px *r, const struct px *a)
{
	struct px t;
	int i;

	px_init(&t);
	if (a->d <= 0) {
		px_copy(F, r, &t);
		px_free(&t);
		return;
	}
	px_reserve(F, &t, a->d - 1);
	for (i = 1; i <= a->d; i++)
		mf_mul_u64(F, CO(&t, i - 1), CO(a, i), (u64)i);
	t.d = a->d - 1;
	px_norm(F, &t);
	px_copy(F, r, &t);
	px_free(&t);
}

void
px_eval(const struct mf *F, u64 *r, const struct px *a, const u64 *x)
{
	u64 acc[FW];
	int i;

	mf_zero(F, acc);
	for (i = a->d; i >= 0; i--) {
		mf_mul(F, acc, acc, x);
		mf_add(F, acc, acc, CO(a, i));
	}
	mf_copy(F, r, acc);
}

/* r = a(b) mod M */
void
px_compose_mod(const struct mf *F, const struct pxmod *M, struct px *r,
    const struct px *a, const struct px *b)
{
	struct px acc, t;
	int i;

	px_init(&acc);
	px_init(&t);
	for (i = a->d; i >= 0; i--) {
		px_mulmod(F, M, &acc, &acc, b);
		px_set_c(F, &t, CO(a, i));
		px_add(F, &acc, &acc, &t);
	}
	px_copy(F, r, &acc);
	px_free(&acc);
	px_free(&t);
}

/* ---- roots ---- */

/* split a squarefree product of linear factors f (monic) */
static int
split_linear(const struct mf *F, const struct px *f, u64 *roots, int max)
{
	struct px h, g, t, q;
	struct pxmod M;
	nn e;
	int n = 0, tries;

	if (f->d <= 0)
		return 0;
	if (f->d == 1) {
		if (max < 1)
			return 0;
		mf_neg(F, roots, CO(f, 0));
		return 1;
	}
	px_init(&h);
	px_init(&g);
	px_init(&t);
	px_init(&q);
	pxmod_init(F, &M, f);
	nn_sub_u64(&e, &F->P, 1);
	nn_shr(&e, &e, 1);
	for (tries = 0; tries < 200; tries++) {
		u64 c[FW];
		/* gcd((x + c)^((p-1)/2) - 1, f) */
		mf_rand(F, c);
		px_set_x(F, &t);
		mf_copy(F, CO(&t, 0), c);
		px_powmod(F, &M, &h, &t, &e);
		px_set_u64(F, &t, 1);
		px_sub(F, &h, &h, &t);
		px_gcd(F, &g, &h, f);
		if (g.d > 0 && g.d < f->d) {
			px_divmod(F, &q, NULL, f, &g);
			px_monic(F, &q);
			n = split_linear(F, &g, roots, max);
			n += split_linear(F, &q, roots + (size_t)n * FW, max - n);
			break;
		}
	}
	pxmod_free(&M);
	px_free(&h);
	px_free(&g);
	px_free(&t);
	px_free(&q);
	return n;
}

/* the distinct roots of f in F_p (returns how many; at most max) */
int
px_roots(const struct mf *F, const struct px *f0, u64 *roots, int max)
{
	struct px f, h, x, g;
	struct pxmod M;
	int n;

	if (f0->d <= 0)
		return 0;
	px_init(&f);
	px_init(&h);
	px_init(&x);
	px_init(&g);
	px_copy(F, &f, f0);
	px_monic(F, &f);
	pxmod_init(F, &M, &f);
	px_set_x(F, &x);
	px_powmod(F, &M, &h, &x, &F->P);
	px_sub(F, &h, &h, &x);
	px_gcd(F, &g, &h, &f);     /* the product of the distinct (x - r) */
	pxmod_free(&M);
	n = split_linear(F, &g, roots, max);
	px_free(&f);
	px_free(&h);
	px_free(&x);
	px_free(&g);
	return n;
}

void
px_print(const struct mf *F, char *buf, size_t sz, const struct px *a,
    const char *var)
{
	size_t o = 0;
	int i, any = 0;

	buf[0] = 0;
	for (i = a->d; i >= 0 && o + 16 < sz; i--) {
		nn v, w;
		int neg = 0;
		const char *s;
		if (mf_is_zero(F, CO(a, i)))
			continue;
		mf_to_nn(F, &v, CO(a, i));
		nn_sub(&w, &F->P, &v);
		if (nn_bits(&w) + 8 < nn_bits(&v)) {
			neg = 1;
			v = w;
		}
		s = nn_str(&v);
		if (any)
			o += (size_t)snprintf(buf + o, sz - o, " %c ", neg ? '-' :
			    '+');
		else if (neg)
			o += (size_t)snprintf(buf + o, sz - o, "-");
		if (i == 0 || !nn_is_one(&v))
			o += (size_t)snprintf(buf + o, sz - o, "%s", s);
		if (i > 0 && o < sz)
			o += (size_t)snprintf(buf + o, sz - o, "%s%s", (i == 0 ||
			    !nn_is_one(&v)) ? "*" : "", var);
		if (i > 1 && o < sz)
			o += (size_t)snprintf(buf + o, sz - o, "^%d", i);
		any = 1;
	}
	if (!any)
		snprintf(buf, sz, "0");
	if (o + 16 >= sz && sz > 8)
		snprintf(buf + sz - 8, 8, " ...");
}
