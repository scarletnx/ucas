/*
 * ecmp.c - fixed-width naturals (nn), signed (zz), parsing and printing,
 * BPSW primality and factorisation, on top of length-generic limb
 * routines.  See ecff.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- limb routines ---- */

u64
l_add(u64 *r, const u64 *a, const u64 *b, int n)
{
	u64 c = 0;
	int i;

	for (i = 0; i < n; i++) {
		u128 s = (u128)a[i] + b[i] + c;
		r[i] = (u64)s;
		c = (u64)(s >> 64);
	}
	return c;
}

u64
l_sub(u64 *r, const u64 *a, const u64 *b, int n)
{
	u64 br = 0;
	int i;

	for (i = 0; i < n; i++) {
		u128 d = (u128)a[i] - b[i] - br;
		r[i] = (u64)d;
		br = (u64)(d >> 64) ? 1 : 0;
	}
	return br;
}

int
l_cmp(const u64 *a, const u64 *b, int n)
{
	int i;

	for (i = n - 1; i >= 0; i--)
		if (a[i] != b[i])
			return a[i] < b[i] ? -1 : 1;
	return 0;
}

int
l_len(const u64 *a, int n)
{
	while (n > 0 && a[n - 1] == 0)
		n--;
	return n;
}

int
l_bits(const u64 *a, int n)
{
	n = l_len(a, n);
	if (n == 0)
		return 0;
	return 64 * (n - 1) + 64 - __builtin_clzll(a[n - 1]);
}

void
l_mul(u64 *r, const u64 *a, int na, const u64 *b, int nb)
{
	int i, j;
	u64 t[2 * 320];
	u64 *o = t;

	if (na + nb > (int)(sizeof t / sizeof t[0]))
		fail("ecff: number too large (%d limbs)", na + nb);
	memset(o, 0, sizeof(u64) * (size_t)(na + nb));
	for (i = 0; i < na; i++) {
		u64 c = 0;
		if (a[i] == 0)
			continue;
		for (j = 0; j < nb; j++) {
			u128 s = (u128)a[i] * b[j] + o[i + j] + c;
			o[i + j] = (u64)s;
			c = (u64)(s >> 64);
		}
		o[i + nb] = c;
	}
	memcpy(r, o, sizeof(u64) * (size_t)(na + nb));
}

#define LDMAX 320

void
l_divmod(u64 *q, u64 *r, const u64 *a, int na0, const u64 *b, int nb0)
{
	u64 un[LDMAX + 2], vn[LDMAX + 1], qq[LDMAX + 1];
	int na = l_len(a, na0), nb = l_len(b, nb0), s, i, j;

	if (nb == 0)
		fail("division by zero");
	if (na > LDMAX)
		fail("ecff: number too large (%d limbs)", na);
	if (q)
		memset(q, 0, sizeof(u64) * (size_t)na0);
	if (na < nb) {
		if (r) {
			memset(r, 0, sizeof(u64) * (size_t)nb0);
			memcpy(r, a, sizeof(u64) * (size_t)na);
		}
		return;
	}
	if (nb == 1) {
		u64 rem = 0;
		for (i = na - 1; i >= 0; i--) {
			u128 cur = ((u128)rem << 64) | a[i];
			u64 qd = (u64)(cur / b[0]);
			rem = (u64)(cur - (u128)qd * b[0]);
			if (q)
				q[i] = qd;
		}
		if (r) {
			memset(r, 0, sizeof(u64) * (size_t)nb0);
			r[0] = rem;
		}
		return;
	}
	s = __builtin_clzll(b[nb - 1]);
	for (i = nb - 1; i > 0; i--)
		vn[i] = s ? (b[i] << s) | (b[i - 1] >> (64 - s)) : b[i];
	vn[0] = b[0] << s;
	un[na] = s ? a[na - 1] >> (64 - s) : 0;
	for (i = na - 1; i > 0; i--)
		un[i] = s ? (a[i] << s) | (a[i - 1] >> (64 - s)) : a[i];
	un[0] = a[0] << s;
	for (j = na - nb; j >= 0; j--) {
		u128 num = ((u128)un[j + nb] << 64) | un[j + nb - 1];
		u128 qhat = num / vn[nb - 1];
		u128 rhat = num - qhat * vn[nb - 1];
		i128 t;
		u64 k, br;

		while (qhat >> 64 || qhat * vn[nb - 2] >
		    ((rhat << 64) | un[j + nb - 2])) {
			qhat--;
			rhat += vn[nb - 1];
			if (rhat >> 64)
				break;
		}
		/* multiply and subtract */
		k = 0;
		br = 0;
		for (i = 0; i < nb; i++) {
			u128 p = qhat * vn[i] + k;
			u64 pl = (u64)p;
			k = (u64)(p >> 64);
			t = (i128)un[i + j] - pl - br;
			un[i + j] = (u64)t;
			br = t < 0 ? 1 : 0;
		}
		t = (i128)un[j + nb] - k - br;
		un[j + nb] = (u64)t;
		if (t < 0) {
			u64 c = 0;
			qhat--;
			for (i = 0; i < nb; i++) {
				u128 sm = (u128)un[i + j] + vn[i] + c;
				un[i + j] = (u64)sm;
				c = (u64)(sm >> 64);
			}
			un[j + nb] += c;
		}
		qq[j] = (u64)qhat;
	}
	if (q)
		for (j = 0; j <= na - nb; j++)
			q[j] = qq[j];
	if (r) {
		memset(r, 0, sizeof(u64) * (size_t)nb0);
		for (i = 0; i < nb; i++)
			r[i] = s ? (un[i] >> s) | (un[i + 1] << (64 - s)) :
			    un[i];
	}
}

/* ---- nn ---- */

void
nn_zero(nn *a)
{
	memset(a, 0, sizeof *a);
}

void
nn_set(nn *a, u64 v)
{
	nn_zero(a);
	a->w[0] = v;
}

int
nn_is_zero(const nn *a)
{
	return l_len(a->w, NW) == 0;
}

int
nn_is_one(const nn *a)
{
	return a->w[0] == 1 && l_len(a->w + 1, NW - 1) == 0;
}

int
nn_cmp(const nn *a, const nn *b)
{
	return l_cmp(a->w, b->w, NW);
}

int
nn_cmp_u64(const nn *a, u64 v)
{
	if (l_len(a->w + 1, NW - 1))
		return 1;
	return a->w[0] < v ? -1 : a->w[0] > v;
}

int
nn_eq(const nn *a, const nn *b)
{
	return !memcmp(a, b, sizeof *a);
}

int
nn_bits(const nn *a)
{
	return l_bits(a->w, NW);
}

int
nn_bit(const nn *a, int i)
{
	if (i < 0 || i >= 64 * NW)
		return 0;
	return (int)((a->w[i / 64] >> (i % 64)) & 1);
}

int
nn_fits_u64(const nn *a)
{
	return l_len(a->w + 1, NW - 1) == 0;
}

u64
nn_add(nn *r, const nn *a, const nn *b)
{
	return l_add(r->w, a->w, b->w, NW);
}

u64
nn_sub(nn *r, const nn *a, const nn *b)
{
	return l_sub(r->w, a->w, b->w, NW);
}

void
nn_add_u64(nn *r, const nn *a, u64 v)
{
	nn t;

	nn_set(&t, v);
	if (nn_add(r, a, &t))
		fail("ecff: integer overflow (more than %d bits)", 64 * NW);
}

void
nn_sub_u64(nn *r, const nn *a, u64 v)
{
	nn t;

	nn_set(&t, v);
	nn_sub(r, a, &t);
}

int
nn_mul(nn *r, const nn *a, const nn *b)
{
	u64 t[2 * NW];
	int na = l_len(a->w, NW), nb = l_len(b->w, NW);

	if (na == 0 || nb == 0) {
		nn_zero(r);
		return 1;
	}
	memset(t, 0, sizeof t);
	l_mul(t, a->w, na, b->w, nb);
	memcpy(r->w, t, sizeof r->w);
	return l_len(t + NW, NW) == 0;
}

int
nn_mul_u64(nn *r, const nn *a, u64 v)
{
	nn t;

	nn_set(&t, v);
	return nn_mul(r, a, &t);
}

void
nn_shl(nn *r, const nn *a, int k)
{
	nn t;
	int i, wsh = k / 64, bsh = k % 64;

	nn_zero(&t);
	for (i = NW - 1; i >= wsh; i--) {
		t.w[i] = a->w[i - wsh] << bsh;
		if (bsh && i - wsh - 1 >= 0)
			t.w[i] |= a->w[i - wsh - 1] >> (64 - bsh);
	}
	*r = t;
}

void
nn_shr(nn *r, const nn *a, int k)
{
	nn t;
	int i, wsh = k / 64, bsh = k % 64;

	nn_zero(&t);
	for (i = 0; i + wsh < NW; i++) {
		t.w[i] = a->w[i + wsh] >> bsh;
		if (bsh && i + wsh + 1 < NW)
			t.w[i] |= a->w[i + wsh + 1] << (64 - bsh);
	}
	*r = t;
}

void
nn_divmod(nn *q, nn *r, const nn *a, const nn *b)
{
	nn qq, rr;

	l_divmod(qq.w, rr.w, a->w, NW, b->w, NW);
	if (q)
		*q = qq;
	if (r)
		*r = rr;
}

u64
nn_div_u64(nn *q, const nn *a, u64 m)
{
	nn qq;
	u64 rem = 0;
	int i;

	if (m == 0)
		fail("division by zero");
	for (i = NW - 1; i >= 0; i--) {
		u128 cur = ((u128)rem << 64) | a->w[i];
		qq.w[i] = (u64)(cur / m);
		rem = (u64)(cur % m);
	}
	if (q)
		*q = qq;
	return rem;
}

u64
nn_mod_u64(const nn *a, u64 m)
{
	u64 rem = 0;
	int i;

	if (m == 0)
		fail("division by zero");
	for (i = l_len(a->w, NW) - 1; i >= 0; i--)
		rem = (u64)((((u128)rem << 64) | a->w[i]) % m);
	return rem;
}

void
nn_mod(nn *r, const nn *a, const nn *m)
{
	nn_divmod(NULL, r, a, m);
}

void
nn_gcd(nn *g, const nn *a, const nn *b)
{
	nn x = *a, y = *b, t;

	while (!nn_is_zero(&y)) {
		nn_mod(&t, &x, &y);
		x = y;
		y = t;
	}
	*g = x;
}

/* extended Euclid with cofactors kept reduced mod m */
int
nn_invmod(nn *r, const nn *a, const nn *m)
{
	nn r0 = *m, r1, s0, s1, q, t, u;

	nn_mod(&r1, a, m);
	nn_zero(&s0);
	nn_set(&s1, 1);
	if (nn_is_zero(&r1))
		return nn_is_one(m) ? (nn_zero(r), 1) : 0;
	while (!nn_is_zero(&r1)) {
		nn_divmod(&q, &t, &r0, &r1);
		r0 = r1;
		r1 = t;
		/* s0 - q s1 mod m */
		nn_mulmod(&u, &q, &s1, m);
		nn_submod(&t, &s0, &u, m);
		s0 = s1;
		s1 = t;
	}
	if (!nn_is_one(&r0))
		return 0;
	*r = s0;
	return 1;
}

void
nn_mulmod(nn *r, const nn *a, const nn *b, const nn *m)
{
	u64 t[2 * NW], rr[NW];
	int na = l_len(a->w, NW), nb = l_len(b->w, NW);

	if (na == 0 || nb == 0) {
		nn_zero(r);
		return;
	}
	memset(t, 0, sizeof t);
	l_mul(t, a->w, na, b->w, nb);
	l_divmod(NULL, rr, t, na + nb, m->w, NW);
	memcpy(r->w, rr, sizeof rr);
}

void
nn_addmod(nn *r, const nn *a, const nn *b, const nn *m)
{
	nn t;
	u64 c = nn_add(&t, a, b);

	if (c || nn_cmp(&t, m) >= 0)
		nn_sub(&t, &t, m);
	*r = t;
}

void
nn_submod(nn *r, const nn *a, const nn *b, const nn *m)
{
	nn t;

	if (nn_sub(&t, a, b))
		nn_add(&t, &t, m);
	*r = t;
}

void
nn_powmod(nn *r, const nn *a, const nn *e, const nn *m)
{
	nn x, b;
	int i;

	if (m->w[0] & 1 && nn_bits(m) <= 64 * FW) {
		struct mf F;
		u64 xa[FW], xr[FW];
		mf_init(&F, m);
		mf_from_nn(&F, xa, a);
		mf_pow(&F, xr, xa, e);
		mf_to_nn(&F, r, xr);
		return;
	}
	nn_set(&x, 1);
	nn_mod(&x, &x, m);
	nn_mod(&b, a, m);
	for (i = nn_bits(e) - 1; i >= 0; i--) {
		nn_mulmod(&x, &x, &x, m);
		if (nn_bit(e, i))
			nn_mulmod(&x, &x, &b, m);
	}
	*r = x;
}

void
nn_pow_u64(nn *r, const nn *a, u64 e)
{
	nn x, b = *a;

	nn_set(&x, 1);
	while (e) {
		if (e & 1)
			if (!nn_mul(&x, &x, &b))
				fail("ecff: power too large");
		e >>= 1;
		if (e && !nn_mul(&b, &b, &b))
			fail("ecff: power too large");
	}
	*r = x;
}

void
nn_sqrt(nn *r, const nn *a)
{
	nn x, y, t;
	int b = nn_bits(a);

	if (b == 0) {
		nn_zero(r);
		return;
	}
	nn_set(&x, 1);
	nn_shl(&x, &x, (b + 1) / 2);   /* x >= sqrt(a) */
	for (;;) {
		nn_divmod(&t, NULL, a, &x);
		nn_add(&y, &x, &t);
		nn_shr(&y, &y, 1);
		if (nn_cmp(&y, &x) >= 0)
			break;
		x = y;
	}
	*r = x;
}

int
nn_is_square(const nn *a, nn *root)
{
	nn s, t;
	u64 m = a->w[0] & 63;

	/* squares mod 64 are 0, 1, 4, 9, 16, 17, 25, 33, 36, 41, 49, 57 */
	if (!((0x202021202030213ULL >> m) & 1))
		return 0;
	nn_sqrt(&s, a);
	nn_mul(&t, &s, &s);
	if (!nn_eq(&t, a))
		return 0;
	if (root)
		*root = s;
	return 1;
}

/* floor of the k-th root */
static void
nn_root(nn *r, const nn *a, int k)
{
	nn x, y, t, km1x, xk1;
	int b = nn_bits(a);

	if (b == 0 || k == 1) {
		*r = *a;
		return;
	}
	nn_set(&x, 1);
	nn_shl(&x, &x, (b + k - 1) / k);
	for (;;) {
		/* y = ((k-1) x + a / x^(k-1)) / k */
		nn_pow_u64(&xk1, &x, (u64)k - 1);
		nn_divmod(&t, NULL, a, &xk1);
		nn_mul_u64(&km1x, &x, (u64)k - 1);
		nn_add(&y, &km1x, &t);
		nn_div_u64(&y, &y, (u64)k);
		if (nn_cmp(&y, &x) >= 0)
			break;
		x = y;
	}
	*r = x;
}

int
nn_jacobi(const nn *a0, const nn *n0)
{
	nn a, n, t;
	int j = 1, z;

	nn_mod(&a, a0, n0);
	n = *n0;
	while (!nn_is_zero(&a)) {
		z = 0;
		while (!(a.w[0] & 1)) {
			nn_shr(&a, &a, 1);
			z++;
		}
		if (z & 1) {
			u64 r = n.w[0] & 7;
			if (r == 3 || r == 5)
				j = -j;
		}
		if ((a.w[0] & 3) == 3 && (n.w[0] & 3) == 3)
			j = -j;
		t = a;
		a = n;
		n = t;
		nn_mod(&a, &a, &n);
	}
	return nn_is_one(&n) ? j : 0;
}

/* ---- PRNG (deterministic: tests reproduce) ---- */

static u64 rs = 0x9e3779b97f4a7c15ULL;

void
nn_seed(u64 s)
{
	rs = s ^ 0x9e3779b97f4a7c15ULL;
}

u64
nn_rand64(void)
{
	u64 z = (rs += 0x9e3779b97f4a7c15ULL);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

void
nn_rand_below(nn *r, const nn *bound)
{
	nn t;
	int i, b = nn_bits(bound);

	if (b == 0) {
		nn_zero(r);
		return;
	}
	do {
		nn_zero(&t);
		for (i = 0; i < (b + 63) / 64; i++)
			t.w[i] = nn_rand64();
		if (b % 64)
			t.w[(b - 1) / 64] &= (~0ULL) >> (64 - b % 64);
	} while (nn_cmp(&t, bound) >= 0);
	*r = t;
}

/* ---- primality: BPSW ---- */

static const unsigned short smallp[] = {
	2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
	67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
	139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
	223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
	293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
	383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
	463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 0
};

static void
mf_half(const struct mf *F, u64 *r, const u64 *a)
{
	u64 t[FW + 1];
	int i, n = F->n;

	memcpy(t, a, sizeof(u64) * (size_t)n);
	t[n] = 0;
	if (t[0] & 1)
		t[n] = l_add(t, t, F->p, n);
	for (i = 0; i < n; i++)
		r[i] = (t[i] >> 1) | (t[i + 1] << 63);
}

static int
strong_lucas(const nn *n)
{
	struct mf F;
	nn d, np1, t;
	int64_t D = 5;
	int s, i, b;
	u64 U[FW], V[FW], Qk[FW], Q[FW], Dm[FW], P[FW], t1[FW], t2[FW], two[FW];

	if (nn_is_square(n, NULL))
		return 0;
	for (;;) {
		zz z;
		nn m;
		int j;
		zz_set_i64(&z, D);
		zz_mod(&m, &z, n);
		j = nn_jacobi(&m, n);
		if (j == -1)
			break;
		if (j == 0) {
			/* gcd(D, n) > 1: n composite unless n = |D| */
			return nn_cmp_u64(n, (u64)(D < 0 ? -D : D)) == 0;
		}
		D = D > 0 ? -(D + 2) : -D + 2;
	}
	mf_init(&F, n);
	mf_from_i64(&F, Dm, D);
	mf_one(&F, P);
	mf_from_i64(&F, Q, (1 - D) / 4);
	mf_from_u64(&F, two, 2);
	nn_add_u64(&np1, n, 1);
	s = 0;
	d = np1;
	while (!(d.w[0] & 1)) {
		nn_shr(&d, &d, 1);
		s++;
	}
	mf_one(&F, U);
	mf_copy(&F, V, P);
	mf_copy(&F, Qk, Q);
	for (b = nn_bits(&d) - 2; b >= 0; b--) {
		mf_mul(&F, U, U, V);
		mf_sqr(&F, V, V);
		mf_dbl(&F, t1, Qk);
		mf_sub(&F, V, V, t1);
		mf_sqr(&F, Qk, Qk);
		if (nn_bit(&d, b)) {
			/* U' = (P U + V)/2, V' = (D U + P V)/2 */
			mf_mul(&F, t1, P, U);
			mf_add(&F, t1, t1, V);
			mf_mul(&F, t2, Dm, U);
			mf_mul(&F, V, P, V);
			mf_add(&F, t2, t2, V);
			mf_half(&F, U, t1);
			mf_half(&F, V, t2);
			mf_mul(&F, Qk, Qk, Q);
		}
	}
	if (mf_is_zero(&F, U) || mf_is_zero(&F, V))
		return 1;
	for (i = 1; i < s; i++) {
		mf_sqr(&F, V, V);
		mf_dbl(&F, t1, Qk);
		mf_sub(&F, V, V, t1);
		if (mf_is_zero(&F, V))
			return 1;
		mf_sqr(&F, Qk, Qk);
	}
	(void)t;
	(void)two;
	return 0;
}

static int
mr_base(const nn *n, u64 base)
{
	struct mf F;
	nn d, nm1, a;
	int s = 0, i;
	u64 x[FW], m1[FW];

	nn_sub_u64(&nm1, n, 1);
	d = nm1;
	while (!(d.w[0] & 1)) {
		nn_shr(&d, &d, 1);
		s++;
	}
	mf_init(&F, n);
	nn_set(&a, base);
	mf_from_nn(&F, x, &a);
	mf_pow(&F, x, x, &d);
	mf_one(&F, m1);
	mf_neg(&F, m1, m1);
	if (mf_is_one(&F, x) || mf_eq(&F, x, m1))
		return 1;
	for (i = 1; i < s; i++) {
		mf_sqr(&F, x, x);
		if (mf_eq(&F, x, m1))
			return 1;
		if (mf_is_one(&F, x))
			return 0;
	}
	return 0;
}

int
nn_isprime(const nn *a)
{
	int i;

	if (nn_fits_u64(a))
		return nt_isprime(a->w[0]);
	if (!(a->w[0] & 1))
		return 0;
	for (i = 1; smallp[i]; i++)
		if (nn_mod_u64(a, smallp[i]) == 0)
			return 0;
	if (nn_bits(a) > 64 * FW)
		fail("ecff: primality of numbers above %d bits", 64 * FW);
	return mr_base(a, 2) && strong_lucas(a);
}

void
nn_nextprime(nn *r, const nn *a)
{
	nn t = *a;

	if (nn_cmp_u64(&t, 2) <= 0) {
		nn_set(r, 2);
		return;
	}
	if (!(t.w[0] & 1))
		nn_add_u64(&t, &t, 1);
	while (!nn_isprime(&t))
		nn_add_u64(&t, &t, 2);
	*r = t;
}

/* ---- bytes ---- */

void
nn_from_bytes_be(nn *r, const unsigned char *b, size_t n)
{
	size_t i;

	nn_zero(r);
	for (i = 0; i < n; i++) {
		size_t k = n - 1 - i;    /* byte significance */
		if (k / 8 < NW)
			r->w[k / 8] |= (u64)b[i] << (8 * (k % 8));
	}
}

void
nn_to_bytes_be(unsigned char *b, size_t n, const nn *a)
{
	size_t i;

	for (i = 0; i < n; i++) {
		size_t k = n - 1 - i;
		b[i] = k / 8 < NW ? (unsigned char)(a->w[k / 8] >> (8 * (k % 8)))
		    : 0;
	}
}

void
nn_from_bytes_le(nn *r, const unsigned char *b, size_t n)
{
	size_t i;

	nn_zero(r);
	for (i = 0; i < n && i / 8 < NW; i++)
		r->w[i / 8] |= (u64)b[i] << (8 * (i % 8));
}

void
nn_to_bytes_le(unsigned char *b, size_t n, const nn *a)
{
	size_t i;

	for (i = 0; i < n; i++)
		b[i] = i / 8 < NW ? (unsigned char)(a->w[i / 8] >> (8 * (i % 8)))
		    : 0;
}

/* ---- zz ---- */

void
zz_set_i64(zz *a, int64_t v)
{
	nn_zero(&a->m);
	a->neg = v < 0;
	a->m.w[0] = v < 0 ? (u64)0 - (u64)v : (u64)v;
}

void
zz_from_nn(zz *a, const nn *m, int neg)
{
	a->m = *m;
	a->neg = neg && !nn_is_zero(m);
}

int
zz_is_zero(const zz *a)
{
	return nn_is_zero(&a->m);
}

static void
zz_addsub(zz *r, const zz *a, const zz *b, int bneg)
{
	zz t;

	if (a->neg == bneg) {
		if (nn_add(&t.m, &a->m, &b->m))
			fail("ecff: integer overflow");
		t.neg = a->neg;
	} else if (nn_cmp(&a->m, &b->m) >= 0) {
		nn_sub(&t.m, &a->m, &b->m);
		t.neg = a->neg;
	} else {
		nn_sub(&t.m, &b->m, &a->m);
		t.neg = bneg;
	}
	if (nn_is_zero(&t.m))
		t.neg = 0;
	*r = t;
}

void
zz_add(zz *r, const zz *a, const zz *b)
{
	zz_addsub(r, a, b, b->neg);
}

void
zz_sub(zz *r, const zz *a, const zz *b)
{
	zz_addsub(r, a, b, !b->neg && !nn_is_zero(&b->m));
}

int
zz_mul(zz *r, const zz *a, const zz *b)
{
	int neg = a->neg != b->neg, ok = nn_mul(&r->m, &a->m, &b->m);

	r->neg = neg && !nn_is_zero(&r->m);
	return ok;
}

void
zz_mod(nn *r, const zz *a, const nn *m)
{
	nn t;

	nn_mod(&t, &a->m, m);
	if (a->neg && !nn_is_zero(&t))
		nn_sub(&t, m, &t);
	*r = t;
}

int
zz_cmp(const zz *a, const zz *b)
{
	if (a->neg != b->neg)
		return a->neg ? -1 : 1;
	return a->neg ? -nn_cmp(&a->m, &b->m) : nn_cmp(&a->m, &b->m);
}

int64_t
zz_to_i64(const zz *a, int *ok)
{
	*ok = nn_fits_u64(&a->m) && a->m.w[0] <= (u64)INT64_MAX;
	if (!*ok)
		return 0;
	return a->neg ? -(int64_t)a->m.w[0] : (int64_t)a->m.w[0];
}

/* ---- parsing: + - * / ^ ( ) and numbers ---- */

static const char *ps;
static int perr;

static void
pskip(void)
{
	while (*ps == ' ' || *ps == '\t')
		ps++;
}

static void pexpr(zz *r);

static void
patom(zz *r)
{
	pskip();
	if (*ps == '(') {
		ps++;
		pexpr(r);
		pskip();
		if (*ps != ')')
			perr = 1;
		else
			ps++;
		return;
	}
	if (*ps == '-') {
		ps++;
		patom(r);
		r->neg = !r->neg && !nn_is_zero(&r->m);
		return;
	}
	nn_zero(&r->m);
	r->neg = 0;
	if (ps[0] == '0' && (ps[1] == 'x' || ps[1] == 'X') &&
	    isxdigit((unsigned char)ps[2])) {
		ps += 2;
		while (isxdigit((unsigned char)*ps) || *ps == '_') {
			int d = *ps == '_' ? -1 : isdigit((unsigned char)*ps) ?
			    *ps - '0' : (tolower((unsigned char)*ps) - 'a' + 10);
			ps++;
			if (d < 0)
				continue;
			if (r->m.w[NW - 1] >> 60)
				fail("ecff: number too large");
			nn_shl(&r->m, &r->m, 4);
			r->m.w[0] |= (u64)d;
		}
		return;
	}
	if (!isdigit((unsigned char)*ps)) {
		perr = 1;
		return;
	}
	while (isdigit((unsigned char)*ps)) {
		if (!nn_mul_u64(&r->m, &r->m, 10))
			fail("ecff: number too large");
		nn_add_u64(&r->m, &r->m, (u64)(*ps - '0'));
		ps++;
	}
}

static void
ppow(zz *r)
{
	patom(r);
	pskip();
	if (*ps == '^' || (ps[0] == '*' && ps[1] == '*')) {
		zz e;
		int ok;
		int64_t k;
		ps += *ps == '^' ? 1 : 2;
		ppow(&e);
		k = zz_to_i64(&e, &ok);
		if (!ok || k < 0 || k > 64 * NW) {
			perr = 1;
			return;
		}
		nn_pow_u64(&r->m, &r->m, (u64)k);
		r->neg = r->neg && (k & 1);
	}
}

static void
pterm(zz *r)
{
	ppow(r);
	for (;;) {
		zz b;
		pskip();
		if (*ps == '*' && ps[1] != '*') {
			ps++;
			ppow(&b);
			if (!zz_mul(r, r, &b))
				fail("ecff: number too large");
		} else if (*ps == '/') {
			nn q, rem;
			ps++;
			ppow(&b);
			if (nn_is_zero(&b.m))
				fail("division by zero");
			nn_divmod(&q, &rem, &r->m, &b.m);
			if (!nn_is_zero(&rem))
				fail("ecff: %s / %s is not an integer",
				    zz_str(r), zz_str(&b));
			zz_from_nn(r, &q, r->neg != b.neg);
		} else
			return;
	}
}

static void
pexpr(zz *r)
{
	pterm(r);
	for (;;) {
		zz b;
		pskip();
		if (*ps == '+') {
			ps++;
			pterm(&b);
			zz_add(r, r, &b);
		} else if (*ps == '-' && !perr) {
			ps++;
			pterm(&b);
			zz_sub(r, r, &b);
		} else
			return;
	}
}

int
zz_parse(const char *s, zz *out, const char **end)
{
	ps = s;
	perr = 0;
	pskip();
	if (!*ps)
		return 0;
	pexpr(out);
	if (end)
		*end = ps;
	return !perr;
}

int
nn_parse(const char *s, nn *out, const char **end)
{
	zz z;

	if (!zz_parse(s, &z, end) || z.neg)
		return 0;
	*out = z.m;
	return 1;
}

/* ---- printing ---- */

static char pbuf[8][400];
static int pslot;

const char *
nn_str(const nn *a)
{
	char tmp[400], *o = pbuf[pslot++ & 7];
	nn t = *a;
	int k = 0, i;

	if (nn_is_zero(&t)) {
		strcpy(o, "0");
		return o;
	}
	while (!nn_is_zero(&t)) {
		u64 r = nn_div_u64(&t, &t, 10000000000000000000ULL);
		for (i = 0; i < 19; i++) {
			tmp[k++] = (char)('0' + r % 10);
			r /= 10;
			if (nn_is_zero(&t) && r == 0)
				break;
		}
	}
	while (k > 1 && tmp[k - 1] == '0')
		k--;
	for (i = 0; i < k; i++)
		o[i] = tmp[k - 1 - i];
	o[k] = 0;
	return o;
}

const char *
nn_hex(const nn *a)
{
	char *o = pbuf[pslot++ & 7];
	int n = l_len(a->w, NW), i, k;

	if (n == 0) {
		strcpy(o, "0x0");
		return o;
	}
	k = sprintf(o, "0x%llx", (unsigned long long)a->w[n - 1]);
	for (i = n - 2; i >= 0; i--)
		k += sprintf(o + k, "%016llx", (unsigned long long)a->w[i]);
	return o;
}

const char *
zz_str(const zz *a)
{
	char *o = pbuf[pslot++ & 7];

	snprintf(o, sizeof pbuf[0], "%s%s", a->neg ? "-" : "", nn_str(&a->m));
	return o;
}

/* ---- factorisation ---- */

static int
rho(const nn *n, nn *f, u64 c, u64 iters)
{
	struct mf F;
	u64 x[FW], y[FW], ys[FW], q[FW], cc[FW], t[FW];
	nn g, d;
	u64 r = 1, k, i, m = 64, done = 0;

	mf_init(&F, n);
	mf_from_u64(&F, cc, c);
	mf_from_u64(&F, y, 2);
	mf_one(&F, q);
	nn_set(&g, 1);
	while (nn_is_one(&g) && done < iters) {
		mf_copy(&F, x, y);
		for (i = 0; i < r; i++) {
			mf_sqr(&F, y, y);
			mf_add(&F, y, y, cc);
		}
		k = 0;
		while (k < r && nn_is_one(&g)) {
			mf_copy(&F, ys, y);
			for (i = 0; i < m && i < r - k; i++) {
				mf_sqr(&F, y, y);
				mf_add(&F, y, y, cc);
				mf_sub(&F, t, x, y);
				mf_mul(&F, q, q, t);
			}
			mf_to_nn(&F, &d, q);
			nn_gcd(&g, &d, n);
			k += m;
			done += m;
		}
		r *= 2;
	}
	if (nn_eq(&g, n)) {
		do {
			mf_sqr(&F, ys, ys);
			mf_add(&F, ys, ys, cc);
			mf_sub(&F, t, x, ys);
			mf_to_nn(&F, &d, t);
			nn_gcd(&g, &d, n);
		} while (nn_is_one(&g));
	}
	if (nn_is_one(&g) || nn_eq(&g, n))
		return 0;
	*f = g;
	return 1;
}

int ecm_find(const nn *n, nn *f, int effort);   /* ecalg.c */

static void
fac_add(struct nnfac *F, const nn *p, int e, int prime)
{
	int i, j;

	for (i = 0; i < F->k; i++)
		if (nn_eq(&F->p[i], p)) {
			F->e[i] += e;
			return;
		}
	if (F->k >= NNFMAX)
		fail("ecff: too many prime factors");
	/* sorted */
	for (i = 0; i < F->k && nn_cmp(&F->p[i], p) < 0; i++)
		;
	for (j = F->k; j > i; j--) {
		F->p[j] = F->p[j - 1];
		F->e[j] = F->e[j - 1];
		F->prime[j] = F->prime[j - 1];
	}
	F->p[i] = *p;
	F->e[i] = e;
	F->prime[i] = prime;
	F->k++;
}

static void
fac_rec(struct nnfac *F, const nn *n, int e, int effort)
{
	nn f, q;
	int k;

	if (nn_is_one(n))
		return;
	if (nn_isprime(n)) {
		fac_add(F, n, e, 1);
		return;
	}
	/* perfect powers */
	for (k = 2; k <= nn_bits(n); k++) {
		nn r, t;
		nn_root(&r, n, k);
		nn_pow_u64(&t, &r, (u64)k);
		if (nn_eq(&t, n)) {
			fac_rec(F, &r, e * k, effort);
			return;
		}
	}
	if (nn_fits_u64(n)) {
		struct nt_fac nf;
		int i;
		nt_factor(n->w[0], &nf);
		for (i = 0; i < nf.k; i++) {
			nn p;
			nn_set(&p, nf.p[i]);
			fac_add(F, &p, e * nf.e[i], 1);
		}
		return;
	}
	if (rho(n, &f, 1, effort >= 2 ? 400000 : 100000) ||
	    rho(n, &f, 3, 50000) || ecm_find(n, &f, effort)) {
		nn_divmod(&q, NULL, n, &f);
		fac_rec(F, &f, e, effort);
		fac_rec(F, &q, e, effort);
		return;
	}
	fac_add(F, n, e, 0);
	F->complete = 0;
}

void
nn_factor(const nn *n0, struct nnfac *F, int effort)
{
	nn n = *n0, q;
	int i;

	F->k = 0;
	F->complete = 1;
	if (nn_is_zero(&n))
		fail("ecff: factoring 0");
	for (i = 0; smallp[i]; i++) {
		int e = 0;
		while (nn_div_u64(&q, &n, smallp[i]) == 0) {
			n = q;
			e++;
		}
		if (e) {
			nn p;
			nn_set(&p, smallp[i]);
			fac_add(F, &p, e, 1);
		}
	}
	fac_rec(F, &n, 1, effort);
	if (!nnfac_check(F, n0))
		fail("ecff: internal error: factorisation does not multiply back");
}

int
nnfac_check(const struct nnfac *F, const nn *n)
{
	nn t, pe;
	int i;

	nn_set(&t, 1);
	for (i = 0; i < F->k; i++) {
		nn_pow_u64(&pe, &F->p[i], (u64)F->e[i]);
		if (!nn_mul(&t, &t, &pe))
			return 0;
	}
	return nn_eq(&t, n);
}

void
nnfac_print(char *buf, size_t sz, const struct nnfac *F)
{
	size_t k = 0;
	int i;

	buf[0] = 0;
	for (i = 0; i < F->k && k + 1 < sz; i++) {
		k += (size_t)snprintf(buf + k, sz - k, "%s%s%s", i ? " * " : "",
		    nn_str(&F->p[i]), F->prime[i] ? "" : "(c)");
		if (F->e[i] > 1 && k < sz)
			k += (size_t)snprintf(buf + k, sz - k, "^%d", F->e[i]);
	}
	if (F->k == 0)
		snprintf(buf, sz, "1");
}
