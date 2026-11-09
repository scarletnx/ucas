/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecfp.c - Z/n in Montgomery form (mf), up to FW limbs, and the
 * extension fields F_{p^k} = F_p[x]/(m) (fk).  See ecff.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- mf ---- */

int
mf_init(struct mf *F, const nn *p)
{
	u64 t[2 * FW + 1], inv;
	int n = l_len(p->w, NW), i;

	memset(F, 0, sizeof *F);
	if (n == 0 || n > FW || !(p->w[0] & 1))
		return 0;
	F->n = n;
	F->P = *p;
	F->bits = nn_bits(p);
	memcpy(F->p, p->w, sizeof(u64) * (size_t)n);
	inv = p->w[0];          /* Newton: inv = p^-1 mod 2^64 */
	for (i = 0; i < 6; i++)
		inv *= 2 - p->w[0] * inv;
	F->pinv = (u64)0 - inv;
	memset(t, 0, sizeof t);
	t[n] = 1;
	l_divmod(NULL, F->one, t, n + 1, F->p, n);
	memset(t, 0, sizeof t);
	t[2 * n] = 1;
	l_divmod(NULL, F->r2, t, 2 * n + 1, F->p, n);
	F->prime = -1;
	return 1;
}

/* p - 1 = 2^s q and a non-residue, for square roots; whether p is prime */
int
mf_prime_setup(struct mf *F)
{
	nn pm1, a;
	u64 v;

	F->prime = nn_isprime(&F->P);
	if (!F->prime)
		return 0;
	nn_sub_u64(&pm1, &F->P, 1);
	F->s = 0;
	F->q = pm1;
	while (!(F->q.w[0] & 1)) {
		nn_shr(&F->q, &F->q, 1);
		F->s++;
	}
	for (v = 2;; v++) {
		nn_set(&a, v);
		if (nn_cmp(&a, &F->P) >= 0)
			break;
		if (nn_jacobi(&a, &F->P) == -1) {
			mf_from_u64(F, F->nonres, v);
			break;
		}
	}
	return 1;
}

static void
mont(const struct mf *F, u64 *r, const u64 *a, const u64 *b)
{
	u64 t[FW + 2];
	int n = F->n, i, j;

	memset(t, 0, sizeof(u64) * (size_t)(n + 2));
	for (i = 0; i < n; i++) {
		u64 c = 0, m;
		u128 s;
		for (j = 0; j < n; j++) {
			s = (u128)a[j] * b[i] + t[j] + c;
			t[j] = (u64)s;
			c = (u64)(s >> 64);
		}
		s = (u128)t[n] + c;
		t[n] = (u64)s;
		t[n + 1] = (u64)(s >> 64);
		m = t[0] * F->pinv;
		s = (u128)m * F->p[0] + t[0];
		c = (u64)(s >> 64);
		for (j = 1; j < n; j++) {
			s = (u128)m * F->p[j] + t[j] + c;
			t[j - 1] = (u64)s;
			c = (u64)(s >> 64);
		}
		s = (u128)t[n] + c;
		t[n - 1] = (u64)s;
		t[n] = t[n + 1] + (u64)(s >> 64);
	}
	if (t[n] || l_cmp(t, F->p, n) >= 0)
		l_sub(t, t, F->p, n);
	memcpy(r, t, sizeof(u64) * (size_t)n);
}

/* n = 4: unrolled (256-bit fields) */
static void
mont4(const struct mf *F, u64 *r, const u64 *a, const u64 *b)
{
	u64 t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5, c, m;
	const u64 *p = F->p;
	u128 s;
	int i;

	for (i = 0; i < 4; i++) {
		u64 bi = b[i];
		s = (u128)a[0] * bi + t0; t0 = (u64)s; c = (u64)(s >> 64);
		s = (u128)a[1] * bi + t1 + c; t1 = (u64)s; c = (u64)(s >> 64);
		s = (u128)a[2] * bi + t2 + c; t2 = (u64)s; c = (u64)(s >> 64);
		s = (u128)a[3] * bi + t3 + c; t3 = (u64)s; c = (u64)(s >> 64);
		s = (u128)t4 + c; t4 = (u64)s; t5 = (u64)(s >> 64);
		m = t0 * F->pinv;
		s = (u128)m * p[0] + t0; c = (u64)(s >> 64);
		s = (u128)m * p[1] + t1 + c; t0 = (u64)s; c = (u64)(s >> 64);
		s = (u128)m * p[2] + t2 + c; t1 = (u64)s; c = (u64)(s >> 64);
		s = (u128)m * p[3] + t3 + c; t2 = (u64)s; c = (u64)(s >> 64);
		s = (u128)t4 + c; t3 = (u64)s; t4 = t5 + (u64)(s >> 64);
	}
	{
		u64 t[4] = {t0, t1, t2, t3};
		if (t4 || l_cmp(t, p, 4) >= 0)
			l_sub(t, t, p, 4);
		r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
	}
}

void
mf_mul(const struct mf *F, u64 *r, const u64 *a, const u64 *b)
{
	if (F->n == 4)
		mont4(F, r, a, b);
	else
		mont(F, r, a, b);
}

void
mf_sqr(const struct mf *F, u64 *r, const u64 *a)
{
	mf_mul(F, r, a, a);
}

void
mf_from_nn(const struct mf *F, u64 *r, const nn *a)
{
	u64 t[FW];

	l_divmod(NULL, t, a->w, NW, F->p, F->n);
	mf_mul(F, r, t, F->r2);
}

void
mf_from_u64(const struct mf *F, u64 *r, u64 v)
{
	nn a;

	nn_set(&a, v);
	mf_from_nn(F, r, &a);
}

void
mf_from_i64(const struct mf *F, u64 *r, int64_t v)
{
	mf_from_u64(F, r, v < 0 ? (u64)0 - (u64)v : (u64)v);
	if (v < 0)
		mf_neg(F, r, r);
}

void
mf_from_zz(const struct mf *F, u64 *r, const zz *a)
{
	mf_from_nn(F, r, &a->m);
	if (a->neg)
		mf_neg(F, r, r);
}

void
mf_to_nn(const struct mf *F, nn *r, const u64 *a)
{
	u64 one[FW];

	memset(one, 0, sizeof one);
	one[0] = 1;
	nn_zero(r);
	mf_mul(F, r->w, a, one);
}

void
mf_copy(const struct mf *F, u64 *r, const u64 *a)
{
	memmove(r, a, sizeof(u64) * (size_t)F->n);
}

void
mf_zero(const struct mf *F, u64 *r)
{
	memset(r, 0, sizeof(u64) * (size_t)F->n);
}

void
mf_one(const struct mf *F, u64 *r)
{
	memcpy(r, F->one, sizeof(u64) * (size_t)F->n);
}

int
mf_is_zero(const struct mf *F, const u64 *a)
{
	return l_len(a, F->n) == 0;
}

int
mf_is_one(const struct mf *F, const u64 *a)
{
	return !memcmp(a, F->one, sizeof(u64) * (size_t)F->n);
}

int
mf_eq(const struct mf *F, const u64 *a, const u64 *b)
{
	return !memcmp(a, b, sizeof(u64) * (size_t)F->n);
}

void
mf_add(const struct mf *F, u64 *r, const u64 *a, const u64 *b)
{
	u64 c = l_add(r, a, b, F->n);

	if (c || l_cmp(r, F->p, F->n) >= 0)
		l_sub(r, r, F->p, F->n);
}

void
mf_sub(const struct mf *F, u64 *r, const u64 *a, const u64 *b)
{
	if (l_sub(r, a, b, F->n))
		l_add(r, r, F->p, F->n);
}

void
mf_neg(const struct mf *F, u64 *r, const u64 *a)
{
	if (mf_is_zero(F, a))
		mf_zero(F, r);
	else
		l_sub(r, F->p, a, F->n);
}

void
mf_dbl(const struct mf *F, u64 *r, const u64 *a)
{
	mf_add(F, r, a, a);
}

void
mf_mul_u64(const struct mf *F, u64 *r, const u64 *a, u64 k)
{
	u64 t[FW];

	mf_from_u64(F, t, k);
	mf_mul(F, r, a, t);
}

void
mf_pow_l(const struct mf *F, u64 *r, const u64 *a, const u64 *e, int ne)
{
	u64 x[FW], tab[16][FW];
	int b = l_bits(e, ne), i, j;

	mf_one(F, x);
	if (b == 0) {
		mf_copy(F, r, x);
		return;
	}
	/* fixed 4-bit windows */
	mf_one(F, tab[0]);
	for (j = 1; j < 16; j++)
		mf_mul(F, tab[j], tab[j - 1], a);
	for (i = ((b + 3) / 4) * 4 - 4; i >= 0; i -= 4) {
		int w = (int)((e[i / 64] >> (i % 64)) & 15);
		mf_sqr(F, x, x);
		mf_sqr(F, x, x);
		mf_sqr(F, x, x);
		mf_sqr(F, x, x);
		if (w)
			mf_mul(F, x, x, tab[w]);
	}
	mf_copy(F, r, x);
}

void
mf_pow(const struct mf *F, u64 *r, const u64 *a, const nn *e)
{
	mf_pow_l(F, r, a, e->w, NW);
}

/* binary extended Euclid on the plain residue */
static int
binv(const struct mf *F, u64 *r, const u64 *a, nn *g)
{
	u64 u[FW], v[FW], x1[FW + 1], x2[FW + 1];
	int n = F->n, i;

	memcpy(u, a, sizeof(u64) * (size_t)n);
	memcpy(v, F->p, sizeof(u64) * (size_t)n);
	memset(x1, 0, sizeof x1);
	memset(x2, 0, sizeof x2);
	x1[0] = 1;
	for (;;) {
		int ulen = l_len(u, n), vlen = l_len(v, n);
		if (ulen == 0 || vlen == 0) {
			if (g) {
				nn_zero(g);
				memcpy(g->w, ulen ? u : v, sizeof(u64) * (size_t)n);
			}
			return 0;
		}
		if (ulen == 1 && u[0] == 1) {
			memcpy(r, x1, sizeof(u64) * (size_t)n);
			return 1;
		}
		if (vlen == 1 && v[0] == 1) {
			memcpy(r, x2, sizeof(u64) * (size_t)n);
			return 1;
		}
		while (!(u[0] & 1)) {
			for (i = 0; i < n - 1; i++)
				u[i] = (u[i] >> 1) | (u[i + 1] << 63);
			u[n - 1] >>= 1;
			x1[n] = 0;
			if (x1[0] & 1)
				x1[n] = l_add(x1, x1, F->p, n);
			for (i = 0; i < n; i++)
				x1[i] = (x1[i] >> 1) | (x1[i + 1] << 63);
		}
		while (!(v[0] & 1)) {
			for (i = 0; i < n - 1; i++)
				v[i] = (v[i] >> 1) | (v[i + 1] << 63);
			v[n - 1] >>= 1;
			x2[n] = 0;
			if (x2[0] & 1)
				x2[n] = l_add(x2, x2, F->p, n);
			for (i = 0; i < n; i++)
				x2[i] = (x2[i] >> 1) | (x2[i + 1] << 63);
		}
		if (l_cmp(u, v, n) >= 0) {
			l_sub(u, u, v, n);
			if (l_sub(x1, x1, x2, n))
				l_add(x1, x1, F->p, n);
		} else {
			l_sub(v, v, u, n);
			if (l_sub(x2, x2, x1, n))
				l_add(x2, x2, F->p, n);
		}
	}
}

int
mf_inv(const struct mf *F, u64 *r, const u64 *a, nn *g)
{
	u64 x[FW];

	if (!binv(F, x, a, g))
		return 0;
	/* a = A R: x = A^-1 R^-1; two Montgomery products by R^2 give A^-1 R */
	mf_mul(F, x, x, F->r2);
	mf_mul(F, r, x, F->r2);
	return 1;
}

int
mf_legendre(const struct mf *F, const u64 *a)
{
	nn v;

	mf_to_nn(F, &v, a);
	return nn_jacobi(&v, &F->P);
}

int
mf_sqrt(const struct mf *F, u64 *r, const u64 *a)
{
	u64 x[FW], b[FW], g[FW], t[FW], chk[FW];
	nn e;
	int m, i;

	if (mf_is_zero(F, a)) {
		mf_zero(F, r);
		return 1;
	}
	if (F->s == 0)
		fail("ecff: square root mod a modulus not set up as prime");
	if (mf_legendre(F, a) != 1)
		return 0;
	if (F->s == 1) {
		nn_add_u64(&e, &F->P, 1);
		nn_shr(&e, &e, 2);
		mf_pow(F, x, a, &e);
	} else {
		/* Tonelli-Shanks */
		nn_add_u64(&e, &F->q, 1);
		nn_shr(&e, &e, 1);
		mf_pow(F, x, a, &e);
		mf_pow(F, b, a, &F->q);
		mf_pow(F, g, F->nonres, &F->q);
		m = F->s;
		while (!mf_is_one(F, b)) {
			mf_copy(F, t, b);
			for (i = 0; i < m && !mf_is_one(F, t); i++)
				mf_sqr(F, t, t);
			if (i == m)
				return 0;
			mf_copy(F, t, g);
			for (int j = 0; j < m - i - 1; j++)
				mf_sqr(F, t, t);
			mf_mul(F, x, x, t);
			mf_sqr(F, g, t);
			mf_mul(F, b, b, g);
			m = i;
		}
	}
	mf_sqr(F, chk, x);
	if (!mf_eq(F, chk, a))
		return 0;
	/* the smaller root, so answers are canonical */
	{
		nn v, w;
		mf_to_nn(F, &v, x);
		nn_sub(&w, &F->P, &v);
		if (nn_cmp(&w, &v) < 0)
			mf_neg(F, x, x);
	}
	mf_copy(F, r, x);
	return 1;
}

const char *
mf_str(const struct mf *F, const u64 *a)
{
	nn v;

	mf_to_nn(F, &v, a);
	return nn_str(&v);
}

const char *
mf_hex(const struct mf *F, const u64 *a)
{
	nn v;

	mf_to_nn(F, &v, a);
	return nn_hex(&v);
}

void
mfv_get(const struct mf *F, u64 *e, const u64 *col, size_t stride, size_t i)
{
	int j;

	for (j = 0; j < F->n; j++)
		e[j] = col[(size_t)j * stride + i];
}

void
mfv_put(const struct mf *F, u64 *col, size_t stride, size_t i, const u64 *e)
{
	int j;

	for (j = 0; j < F->n; j++)
		col[(size_t)j * stride + i] = e[j];
}

int
mfv_inv(const struct mf *F, u64 *col, size_t stride, size_t cnt, nn *g)
{
	u64 *pre, acc[FW], inv[FW], x[FW], t[FW];
	size_t i;

	if (cnt == 0)
		return 1;
	pre = xmalloc(sizeof(u64) * FW * cnt);
	mf_one(F, acc);
	for (i = 0; i < cnt; i++) {
		memcpy(pre + i * FW, acc, sizeof acc);
		mfv_get(F, x, col, stride, i);
		mf_mul(F, acc, acc, x);
	}
	if (!mf_inv(F, inv, acc, g)) {
		free(pre);
		return 0;
	}
	for (i = cnt; i-- > 0;) {
		mfv_get(F, x, col, stride, i);
		mf_mul(F, t, inv, pre + i * FW);
		mf_mul(F, inv, inv, x);
		mfv_put(F, col, stride, i, t);
	}
	free(pre);
	return 1;
}

/* ---- fk: F_{p^k}; coefficient i of an element at a + i * FW ---- */

#define C(a, i) ((a) + (size_t)(i) * FW)

void
fk_init(struct fk *K, const struct mf *F, int k, const u64 *mod)
{
	int i;

	memset(K, 0, sizeof *K);
	K->F = F;
	K->k = k;
	K->n = F->n;
	memcpy(K->m, mod, sizeof(u64) * FW * (size_t)(k + 1));
	K->binom = 1;
	for (i = 1; i < k; i++)
		if (!mf_is_zero(F, C(K->m, i)))
			K->binom = 0;
}

void
fk_zero(const struct fk *K, u64 *r)
{
	memset(r, 0, sizeof(u64) * FW * (size_t)K->k);
}

void
fk_one(const struct fk *K, u64 *r)
{
	fk_zero(K, r);
	mf_one(K->F, r);
}

void
fk_copy(const struct fk *K, u64 *r, const u64 *a)
{
	memmove(r, a, sizeof(u64) * FW * (size_t)K->k);
}

void
fk_from_mf(const struct fk *K, u64 *r, const u64 *a)
{
	u64 t[FW];

	mf_copy(K->F, t, a);
	fk_zero(K, r);
	mf_copy(K->F, r, t);
}

int
fk_is_zero(const struct fk *K, const u64 *a)
{
	int i;

	for (i = 0; i < K->k; i++)
		if (!mf_is_zero(K->F, C(a, i)))
			return 0;
	return 1;
}

int
fk_is_one(const struct fk *K, const u64 *a)
{
	int i;

	if (!mf_is_one(K->F, a))
		return 0;
	for (i = 1; i < K->k; i++)
		if (!mf_is_zero(K->F, C(a, i)))
			return 0;
	return 1;
}

int
fk_eq(const struct fk *K, const u64 *a, const u64 *b)
{
	int i;

	for (i = 0; i < K->k; i++)
		if (!mf_eq(K->F, C(a, i), C(b, i)))
			return 0;
	return 1;
}

void
fk_add(const struct fk *K, u64 *r, const u64 *a, const u64 *b)
{
	int i;

	for (i = 0; i < K->k; i++)
		mf_add(K->F, C(r, i), C(a, i), C(b, i));
}

void
fk_sub(const struct fk *K, u64 *r, const u64 *a, const u64 *b)
{
	int i;

	for (i = 0; i < K->k; i++)
		mf_sub(K->F, C(r, i), C(a, i), C(b, i));
}

void
fk_neg(const struct fk *K, u64 *r, const u64 *a)
{
	int i;

	for (i = 0; i < K->k; i++)
		mf_neg(K->F, C(r, i), C(a, i));
}

/* reduce t (2k - 1 coefficients) mod m into r */
static void
fk_reduce(const struct fk *K, u64 *r, u64 *t)
{
	const struct mf *F = K->F;
	u64 x[FW];
	int i, j, k = K->k;

	for (i = 2 * k - 2; i >= k; i--) {
		if (mf_is_zero(F, C(t, i)))
			continue;
		if (K->binom) {
			/* x^k = -m0 */
			mf_mul(F, x, C(t, i), C(K->m, 0));
			mf_sub(F, C(t, i - k), C(t, i - k), x);
			continue;
		}
		for (j = 0; j < k; j++) {
			if (mf_is_zero(F, C(K->m, j)))
				continue;
			mf_mul(F, x, C(t, i), C(K->m, j));
			mf_sub(F, C(t, i - k + j), C(t, i - k + j), x);
		}
	}
	memcpy(r, t, sizeof(u64) * FW * (size_t)k);
}

void
fk_mul(const struct fk *K, u64 *r, const u64 *a, const u64 *b)
{
	const struct mf *F = K->F;
	u64 t[(2 * FKMAX) * FW], x[FW];
	int i, j, k = K->k;

	if (k == 1) {
		mf_mul(F, r, a, b);
		return;
	}
	memset(t, 0, sizeof(u64) * FW * (size_t)(2 * k - 1));
	for (i = 0; i < k; i++) {
		if (mf_is_zero(F, C(a, i)))
			continue;
		for (j = 0; j < k; j++) {
			mf_mul(F, x, C(a, i), C(b, j));
			mf_add(F, C(t, i + j), C(t, i + j), x);
		}
	}
	fk_reduce(K, r, t);
}

void
fk_sqr(const struct fk *K, u64 *r, const u64 *a)
{
	fk_mul(K, r, a, a);
}

/* small polynomials over F_p (degree <= FKMAX) for inverses and gcds */
static int
sp_deg(const struct mf *F, const u64 *a, int d)
{
	while (d >= 0 && mf_is_zero(F, C(a, d)))
		d--;
	return d;
}

/* r = a mod b (b of degree db, any leading coefficient); returns deg r;
 * q (optional) receives the quotient */
static int
sp_divmod(const struct mf *F, u64 *q, u64 *a, int da, const u64 *b, int db)
{
	u64 il[FW], c[FW], x[FW];
	int i, j;

	if (q)
		memset(q, 0, sizeof(u64) * FW * (FKMAX + 1));
	mf_inv(F, il, C(b, db), NULL);
	for (i = da; i >= db; i--) {
		if (mf_is_zero(F, C(a, i)))
			continue;
		mf_mul(F, c, C(a, i), il);
		if (q)
			mf_copy(F, C(q, i - db), c);
		for (j = 0; j <= db; j++) {
			mf_mul(F, x, c, C(b, j));
			mf_sub(F, C(a, i - db + j), C(a, i - db + j), x);
		}
	}
	return sp_deg(F, a, db - 1);
}

int
fk_inv(const struct fk *K, u64 *r, const u64 *a)
{
	const struct mf *F = K->F;
	u64 r0[(FKMAX + 2) * FW], r1[(FKMAX + 2) * FW], s0[(FKMAX + 2) * FW],
	    s1[(FKMAX + 2) * FW], q[(FKMAX + 2) * FW], t[(FKMAX + 2) * FW],
	    x[FW], il[FW];
	int k = K->k, d0, d1, i, j, dq;

	if (k == 1)
		return mf_inv(F, r, a, NULL);
	memset(r0, 0, sizeof r0);
	memset(r1, 0, sizeof r1);
	memset(s0, 0, sizeof s0);
	memset(s1, 0, sizeof s1);
	memcpy(r0, K->m, sizeof(u64) * FW * (size_t)(k + 1));
	memcpy(r1, a, sizeof(u64) * FW * (size_t)k);
	d0 = k;
	d1 = sp_deg(F, r1, k - 1);
	if (d1 < 0)
		return 0;
	mf_one(F, s1);
	/* invariant: r_i = s_i a mod m */
	while (d1 > 0) {
		int dr;
		memcpy(t, r0, sizeof t);
		dr = sp_divmod(F, q, t, d0, r1, d1);
		dq = d0 - d1;
		/* s0 - q s1 */
		for (i = 0; i <= dq; i++) {
			if (mf_is_zero(F, C(q, i)))
				continue;
			for (j = 0; j + i <= k; j++) {
				if (mf_is_zero(F, C(s1, j)))
					continue;
				mf_mul(F, x, C(q, i), C(s1, j));
				mf_sub(F, C(s0, i + j), C(s0, i + j), x);
			}
		}
		memcpy(r0, r1, sizeof r0);
		memcpy(r1, t, sizeof r1);
		memcpy(t, s0, sizeof t);
		memcpy(s0, s1, sizeof s0);
		memcpy(s1, t, sizeof s1);
		d0 = d1;
		d1 = dr;
		if (d1 < 0)
			return 0;   /* m not irreducible: a shares a factor */
	}
	/* r1 = c constant: a^-1 = s1 / c */
	if (!mf_inv(F, il, r1, NULL))
		return 0;
	for (i = 0; i < k; i++)
		mf_mul(F, C(r, i), C(s1, i), il);
	return 1;
}

void
fk_pow_l(const struct fk *K, u64 *r, const u64 *a, const u64 *e, int ne)
{
	u64 x[FKE], b[FKE];
	int nb = l_bits(e, ne), i;

	fk_copy(K, b, a);
	fk_one(K, x);
	for (i = nb - 1; i >= 0; i--) {
		fk_sqr(K, x, x);
		if ((e[i / 64] >> (i % 64)) & 1)
			fk_mul(K, x, x, b);
	}
	fk_copy(K, r, x);
}

void
fk_frob(const struct fk *K, u64 *r, const u64 *a)
{
	fk_pow_l(K, r, a, K->F->P.w, K->F->n);
}

int
fk_qm1(const struct fk *K, u64 *e)
{
	u64 t[FKMAX * FW + 2], u[FKMAX * FW + 2];
	int len = 1, i, n = K->n, one = 1;

	memset(t, 0, sizeof t);
	t[0] = 1;
	for (i = 0; i < K->k; i++) {
		memset(u, 0, sizeof u);
		l_mul(u, t, len, K->F->p, n);
		len += n;
		memcpy(t, u, sizeof(u64) * (size_t)len);
		len = l_len(t, len);
	}
	/* minus 1 */
	for (i = 0; one && i < len; i++) {
		if (t[i]) {
			t[i]--;
			one = 0;
		} else
			t[i] = ~0ULL;
	}
	memcpy(e, t, sizeof(u64) * (size_t)len);
	return l_len(e, len);
}

void
fk_print(const struct fk *K, char *buf, size_t sz, const u64 *a)
{
	size_t o = 0;
	int i, any = 0;

	buf[0] = 0;
	for (i = K->k - 1; i >= 0; i--) {
		const char *s;
		if (mf_is_zero(K->F, C(a, i)))
			continue;
		s = mf_str(K->F, C(a, i));
		if (o + 8 >= sz)
			break;
		o += (size_t)snprintf(buf + o, sz - o, "%s", any ? " + " : "");
		if (i == 0 || strcmp(s, "1"))
			o += (size_t)snprintf(buf + o, sz - o, "%s%s", s,
			    i ? "*" : "");
		if (i == 1)
			o += (size_t)snprintf(buf + o, sz - o, "x");
		else if (i > 1)
			o += (size_t)snprintf(buf + o, sz - o, "x^%d", i);
		any = 1;
	}
	if (!any)
		snprintf(buf, sz, "0");
}

/* is m (the context's modulus) irreducible: x^(p^k) = x mod m and
 * gcd(x^(p^(k/r)) - x, m) = 1 for primes r | k */
int
fk_irreducible(const struct fk *K)
{
	const struct mf *F = K->F;
	u64 X[FKE], Y[FKE], t[(FKMAX + 2) * FW], g0[(FKMAX + 2) * FW],
	    g1[(FKMAX + 2) * FW];
	int k = K->k, i, r;

	fk_zero(K, X);
	if (k == 1)
		return 1;
	mf_one(F, C(X, 1));
	fk_copy(K, Y, X);
	for (i = 0; i < k; i++)
		fk_frob(K, Y, Y);
	if (!fk_eq(K, X, Y))
		return 0;
	for (r = 2; r <= k; r++) {
		int d0, d1, isp = 1, j;
		for (j = 2; j < r; j++)
			if (r % j == 0)
				isp = 0;
		if (!isp || k % r)
			continue;
		fk_copy(K, Y, X);
		for (i = 0; i < k / r; i++)
			fk_frob(K, Y, Y);
		fk_sub(K, Y, Y, X);
		memset(g0, 0, sizeof g0);
		memset(g1, 0, sizeof g1);
		memcpy(g0, K->m, sizeof(u64) * FW * (size_t)(k + 1));
		memcpy(g1, Y, sizeof(u64) * FW * (size_t)k);
		d0 = k;
		d1 = sp_deg(F, g1, k - 1);
		while (d1 >= 0) {
			int dr;
			memcpy(t, g0, sizeof t);
			dr = sp_divmod(F, NULL, t, d0, g1, d1);
			memcpy(g0, g1, sizeof g0);
			memcpy(g1, t, sizeof g1);
			d0 = d1;
			d1 = dr;
		}
		if (d0 > 0)
			return 0;
	}
	return 1;
}

int
fk_init_auto(struct fk *K, const struct mf *F, int k)
{
	u64 m[(FKMAX + 1) * FW];
	int c, tries;

	if (k < 1 || k > FKMAX)
		return 0;
	memset(m, 0, sizeof m);
	mf_one(F, C(m, k));
	if (k == 1) {
		fk_init(K, F, 1, m);
		return 1;
	}
	/* binomials x^k - c, then trinomials x^k + x + c, then random */
	for (c = 1; c < 200; c++) {
		mf_from_i64(F, C(m, 0), -(int64_t)(c / 2 + 1) *
		    (c & 1 ? 1 : -1));
		fk_init(K, F, k, m);
		if (fk_irreducible(K))
			return 1;
	}
	mf_one(F, C(m, 1));
	for (c = 1; c < 400; c++) {
		mf_from_i64(F, C(m, 0), c);
		fk_init(K, F, k, m);
		if (fk_irreducible(K))
			return 1;
	}
	for (tries = 0; tries < 2000; tries++) {
		int i;
		for (i = 0; i < k; i++) {
			nn v;
			nn_rand_below(&v, &F->P);
			mf_from_nn(F, C(m, i), &v);
		}
		fk_init(K, F, k, m);
		if (fk_irreducible(K))
			return 1;
	}
	return 0;
}

int
fk_sqrt(const struct fk *K, u64 *r, const u64 *a)
{
	u64 e[FKMAX * FW + 2], q[FKMAX * FW + 2], x[FKE], b[FKE], g[FKE],
	    t[FKE], z[FKE], chk[FKE];
	int ne, nq, s = 0, m, i, j;

	if (K->k == 1)
		return mf_sqrt(K->F, r, a);
	if (fk_is_zero(K, a)) {
		fk_zero(K, r);
		return 1;
	}
	ne = fk_qm1(K, e);
	/* q - 1 = 2^s q' */
	memcpy(q, e, sizeof(u64) * (size_t)ne);
	nq = ne;
	while (!(q[0] & 1)) {
		for (i = 0; i < nq; i++)
			q[i] = (q[i] >> 1) | (i + 1 < nq ? q[i + 1] << 63 : 0);
		s++;
	}
	/* Euler's criterion: a^((q-1)/2) */
	{
		u64 h[FKMAX * FW + 2];
		memcpy(h, e, sizeof(u64) * (size_t)ne);
		for (i = 0; i < ne; i++)
			h[i] = (h[i] >> 1) | (i + 1 < ne ? h[i + 1] << 63 : 0);
		fk_pow_l(K, t, a, h, ne);
		if (!fk_is_one(K, t))
			return 0;
		/* a non-residue z */
		for (;;) {
			for (i = 0; i < K->k; i++) {
				nn v;
				nn_rand_below(&v, &K->F->P);
				mf_from_nn(K->F, C(z, i), &v);
			}
			if (fk_is_zero(K, z))
				continue;
			fk_pow_l(K, t, z, h, ne);
			if (!fk_is_one(K, t))
				break;
		}
	}
	/* Tonelli-Shanks */
	{
		u64 q1[FKMAX * FW + 2];
		u64 c = 1;
		memcpy(q1, q, sizeof(u64) * (size_t)nq);
		for (i = 0; c && i < nq; i++) {
			u128 sm = (u128)q1[i] + c;
			q1[i] = (u64)sm;
			c = (u64)(sm >> 64);
		}
		for (i = 0; i < nq; i++)
			q1[i] = (q1[i] >> 1) | (i + 1 < nq ? q1[i + 1] << 63 : 0);
		fk_pow_l(K, x, a, q1, nq);
	}
	fk_pow_l(K, b, a, q, nq);
	fk_pow_l(K, g, z, q, nq);
	m = s;
	while (!fk_is_one(K, b)) {
		fk_copy(K, t, b);
		for (i = 0; i < m && !fk_is_one(K, t); i++)
			fk_sqr(K, t, t);
		if (i == m)
			return 0;
		fk_copy(K, t, g);
		for (j = 0; j < m - i - 1; j++)
			fk_sqr(K, t, t);
		fk_mul(K, x, x, t);
		fk_sqr(K, g, t);
		fk_mul(K, b, b, g);
		m = i;
	}
	fk_sqr(K, chk, x);
	if (!fk_eq(K, chk, a))
		return 0;
	fk_copy(K, r, x);
	return 1;
}

/* ---- misc ---- */

u64
ecff_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (u64)ts.tv_sec * 1000 + (u64)ts.tv_nsec / 1000000;
}
