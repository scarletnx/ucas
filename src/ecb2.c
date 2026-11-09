/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecb2.c - binary fields GF(2^m) = GF(2)[x]/(f), m <= 571 (f a trinomial
 * or pentanomial: its exponents), elements as bit vectors of FW limbs;
 * binary curves y^2 + x y = x^3 + a x^2 + b: the group law, scalar
 * multiples, counting (naive for small m, Koblitz curves by the Frobenius
 * trace recurrence), named NIST curves checked on load.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

void
b2_init(struct b2f *K, int m, const int *ex, int nex)
{
	int i;

	memset(K, 0, sizeof *K);
	if (m < 2 || m > 64 * FW - 4)
		fail("GF(2^m): 2 <= m <= %d", 64 * FW - 4);
	K->m = m;
	K->n = (m + 63) / 64;
	K->nex = nex;
	for (i = 0; i < nex; i++)
		K->ex[i] = ex[i];
}

static void
b2_reduce(const struct b2f *K, u64 *r, u64 *t, int tl)
{
	int i, k, m = K->m;

	for (i = 64 * tl - 1; i >= m; i--) {
		if (!((t[i / 64] >> (i % 64)) & 1))
			continue;
		t[i / 64] ^= 1ULL << (i % 64);
		for (k = 0; k < K->nex; k++) {
			int b = i - m + K->ex[k];
			t[b / 64] ^= 1ULL << (b % 64);
		}
	}
	memcpy(r, t, sizeof(u64) * (size_t)K->n);
	for (i = K->n; i < FW; i++)
		r[i] = 0;
}

void
b2_mul(const struct b2f *K, u64 *r, const u64 *a, const u64 *b)
{
	u64 t[2 * FW + 1], tab[16][FW + 1];
	int n = K->n, i, j, s;

	memset(t, 0, sizeof t);
	memset(tab, 0, sizeof tab);
	/* tab[k] = k(x) * b for the 16 nibbles k */
	for (s = 1; s < 16; s++) {
		for (j = 0; j < 4; j++)
			if ((s >> j) & 1) {
				u64 c = 0;
				for (i = 0; i < n; i++) {
					u64 v = j ? (b[i] << j) | c : b[i];
					c = j ? b[i] >> (64 - j) : 0;
					tab[s][i] ^= v;
				}
				tab[s][n] ^= c;
			}
	}
	for (i = n - 1; i >= 0; i--)
		for (s = 60; s >= 0; s -= 4) {
			int nib = (int)((a[i] >> s) & 15), sh = 64 * i + s, w = sh / 64,
			    bs = sh % 64;
			if (!nib)
				continue;
			for (j = 0; j <= n; j++) {
				t[w + j] ^= tab[nib][j] << bs;
				if (bs)
					t[w + j + 1] ^= tab[nib][j] >> (64 - bs);
			}
		}
	b2_reduce(K, r, t, 2 * n + 1);
}

void
b2_sqr(const struct b2f *K, u64 *r, const u64 *a)
{
	b2_mul(K, r, a, a);
}

int
b2_is_zero(const struct b2f *K, const u64 *a)
{
	return l_len(a, K->n) == 0;
}

/* a^-1 = a^(2^m - 2) */
int
b2_inv(const struct b2f *K, u64 *r, const u64 *a)
{
	u64 x[FW], s[FW];
	int i;

	if (b2_is_zero(K, a))
		return 0;
	/* 2^m - 2 = 2 + 4 + ... + 2^(m-1) */
	memset(x, 0, sizeof x);
	x[0] = 1;
	memcpy(s, a, sizeof s);
	for (i = 1; i < K->m; i++) {
		b2_sqr(K, s, s);
		b2_mul(K, x, x, s);
	}
	memcpy(r, x, sizeof x);
	return 1;
}

/* the absolute trace (0 or 1) */
int
b2_trace(const struct b2f *K, const u64 *a)
{
	u64 t[FW], s[FW];
	int i;

	memcpy(t, a, sizeof t);
	memcpy(s, a, sizeof s);
	for (i = 1; i < K->m; i++) {
		b2_sqr(K, s, s);
		for (int j = 0; j < FW; j++)
			t[j] ^= s[j];
	}
	return (int)(t[0] & 1);
}

/* ---- curves y^2 + x y = x^3 + a x^2 + b ---- */

int
b2_on(const struct b2c *E, const struct b2pt *P)
{
	const struct b2f *K = E->K;
	u64 l[FW], r[FW], t[FW];
	int i;

	if (P->inf)
		return 1;
	b2_sqr(K, l, P->y);
	b2_mul(K, t, P->x, P->y);
	for (i = 0; i < FW; i++)
		l[i] ^= t[i];
	b2_sqr(K, t, P->x);
	b2_mul(K, r, t, P->x);
	b2_mul(K, t, t, E->a);
	for (i = 0; i < FW; i++)
		r[i] ^= t[i] ^ E->b[i];
	return !memcmp(l, r, sizeof l);
}

void
b2_add(const struct b2c *E, struct b2pt *R, const struct b2pt *P,
    const struct b2pt *Q)
{
	const struct b2f *K = E->K;
	u64 l[FW], t[FW], x3[FW], y3[FW], d[FW];
	int i;

	if (P->inf) {
		*R = *Q;
		return;
	}
	if (Q->inf) {
		*R = *P;
		return;
	}
	if (!memcmp(P->x, Q->x, sizeof P->x)) {
		/* -P = (x, x + y) */
		for (i = 0; i < FW; i++)
			t[i] = P->x[i] ^ P->y[i];
		if (!memcmp(t, Q->y, sizeof t) || b2_is_zero(K, P->x)) {
			R->inf = 1;
			return;
		}
		/* doubling: l = x + y/x, x3 = l^2 + l + a, y3 = x^2 + (l + 1) x3 */
		b2_inv(K, d, P->x);
		b2_mul(K, l, P->y, d);
		for (i = 0; i < FW; i++)
			l[i] ^= P->x[i];
		b2_sqr(K, x3, l);
		for (i = 0; i < FW; i++)
			x3[i] ^= l[i] ^ E->a[i];
		b2_sqr(K, y3, P->x);
		l[0] ^= 1;
		b2_mul(K, t, l, x3);
		for (i = 0; i < FW; i++)
			y3[i] ^= t[i];
	} else {
		/* l = (y1 + y2)/(x1 + x2), x3 = l^2 + l + x1 + x2 + a,
		 * y3 = l (x1 + x3) + x3 + y1 */
		for (i = 0; i < FW; i++) {
			t[i] = P->y[i] ^ Q->y[i];
			d[i] = P->x[i] ^ Q->x[i];
		}
		b2_inv(K, d, d);
		b2_mul(K, l, t, d);
		b2_sqr(K, x3, l);
		for (i = 0; i < FW; i++)
			x3[i] ^= l[i] ^ P->x[i] ^ Q->x[i] ^ E->a[i];
		for (i = 0; i < FW; i++)
			t[i] = P->x[i] ^ x3[i];
		b2_mul(K, y3, l, t);
		for (i = 0; i < FW; i++)
			y3[i] ^= x3[i] ^ P->y[i];
	}
	memcpy(R->x, x3, sizeof x3);
	memcpy(R->y, y3, sizeof y3);
	R->inf = 0;
}

void
b2_mul_pt(const struct b2c *E, struct b2pt *R, const struct b2pt *P,
    const nn *k)
{
	struct b2pt A, B = *P;
	int i;

	A.inf = 1;
	memset(A.x, 0, sizeof A.x);
	memset(A.y, 0, sizeof A.y);
	for (i = nn_bits(k) - 1; i >= 0; i--) {
		b2_add(E, &A, &A, &A);
		if (nn_bit(k, i))
			b2_add(E, &A, &A, &B);
	}
	*R = A;
}

/* naive count (m <= 24): 2 points over x = 0 ... no: x = 0 gives y^2 = b,
 * one point; x != 0: z^2 + z = x + a + b/x^2 has 2 or 0 solutions */
int
b2_count_naive(const struct b2c *E, nn *N)
{
	const struct b2f *K = E->K;
	u64 x, cnt = 2;

	if (K->m > 24)
		return 0;
	for (x = 1; x < (1ULL << K->m); x++) {
		u64 X[FW], c[FW], t[FW];
		int i;
		memset(X, 0, sizeof X);
		X[0] = x;
		b2_sqr(K, t, X);
		b2_inv(K, t, t);
		b2_mul(K, c, t, E->b);
		for (i = 0; i < FW; i++)
			c[i] ^= X[i] ^ E->a[i];
		if (!b2_trace(K, c))
			cnt += 2;
	}
	nn_set(N, cnt);
	return 1;
}

/* a, b in GF(2) (Koblitz): #E(F_2^m) = 2^m + 1 - t_m, t_1 = 2 + 1 -
 * #E(F_2), t_k+1 = t_1 t_k - 2 t_k-1 */
int
b2_count_koblitz(const struct b2c *E, nn *N, zz *tm)
{
	const struct b2f *K = E->K;
	int a, b, x, y, n1 = 1, k;
	zz t0, t1, tc, two, T;

	if (l_len(E->a + 1, FW - 1) || (E->a[0] >> 1) || l_len(E->b + 1, FW - 1)
	    || E->b[0] != 1)
		return 0;
	a = (int)(E->a[0] & 1);
	b = 1;
	for (x = 0; x < 2; x++)
		for (y = 0; y < 2; y++)
			if (((y * y + x * y) & 1) == ((x * x * x + a * x * x + b) & 1))
				n1++;
	zz_set_i64(&t0, 2);
	zz_set_i64(&t1, 3 - n1);
	T = t1;
	zz_set_i64(&two, 2);
	for (k = 2; k <= K->m; k++) {
		zz u, v;
		zz_mul(&u, &T, &t1);
		zz_mul(&v, &two, &t0);
		zz_sub(&tc, &u, &v);
		t0 = t1;
		t1 = tc;
	}
	{
		nn pm;
		zz P, one;
		nn_set(&pm, 1);
		nn_shl(&pm, &pm, K->m);
		zz_from_nn(&P, &pm, 0);
		zz_set_i64(&one, 1);
		zz_add(&P, &P, &one);
		zz_sub(&P, &P, &t1);
		*N = P.m;
	}
	*tm = t1;
	return 1;
}

/* ---- named binary curves (FIPS 186-4) ---- */

const struct b2named b2named_tab[] = {
	{"K-163", "k163,sect163k1", 163, {7, 6, 3, 0}, 4, "1", "1",
	    "0x02FE13C0537BBC11ACAA07D793DE4E6D5E5C94EEE8",
	    "0x0289070FB05D38FF58321F2E800536D538CCDAA3D9",
	    "0x04000000000000000000020108A2E0CC0D99F8A5EF", "2"},
	{"B-163", "b163,sect163r2", 163, {7, 6, 3, 0}, 4, "1",
	    "0x020A601907B8C953CA1481EB10512F78744A3205FD",
	    "0x03F0EBA16286A2D57EA0991168D4994637E8343E36",
	    "0x00D51FBC6C71A0094FA2CDD545B11C5C0C797324F1",
	    "0x040000000000000000000292FE77E70C12A4234C33", "2"},
	{"K-233", "k233,sect233k1", 233, {74, 0}, 2, "0", "1",
	    "0x017232BA853A7E731AF129F22FF4149563A419C26BF50A4C9D6EEFAD6126",
	    "0x01DB537DECE819B7F70F555A67C427A8CD9BF18AEB9B56E0C11056FAE6A3",
	    "0x8000000000000000000000000000069D5BB915BCD46EFB1AD5F173ABDF", "4"},
	{NULL, NULL, 0, {0}, 0, NULL, NULL, NULL, NULL, NULL, NULL}
};

const struct b2named *
b2named_find(const char *s)
{
	int i;

	for (i = 0; b2named_tab[i].name; i++) {
		const char *a = b2named_tab[i].aliases;
		size_t L = strlen(s);
		if (!strcasecmp(b2named_tab[i].name, s))
			return &b2named_tab[i];
		while (a && *a) {
			const char *c = strchr(a, ',');
			size_t n = c ? (size_t)(c - a) : strlen(a);
			if (n == L && !strncasecmp(a, s, n))
				return &b2named_tab[i];
			a = c ? c + 1 : NULL;
		}
	}
	return NULL;
}

void
b2_from_nn(const struct b2f *K, u64 *r, const nn *v)
{
	memset(r, 0, sizeof(u64) * FW);
	memcpy(r, v->w, sizeof(u64) * (size_t)K->n);
}
