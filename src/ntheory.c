/*
 * Number theory on u64: modular arithmetic through 128-bit products,
 * Miller-Rabin (deterministic for 64 bits), Pollard-Brent factoring,
 * Euler phi, Carmichael lambda, multiplicative order, Jacobi symbols,
 * square roots (Tonelli-Shanks, Hensel lifting, CRT), primitive roots and
 * the Chinese remainder theorem.
 */
#include "ring.h"

#include <stdlib.h>
#include <string.h>

u64
nt_mulmod(u64 a, u64 b, u64 m)
{
	return (u64)((u128)a * b % m);
}

u64
nt_addmod(u64 a, u64 b, u64 m)
{
	u128 s = (u128)a + b;

	return (u64)(s >= m ? s - m : s);
}

u64
nt_submod(u64 a, u64 b, u64 m)
{
	return a >= b ? a - b : a + (m - b);
}

u64
nt_powmod(u64 a, u64 e, u64 m)
{
	u64 r = 1 % m;

	a %= m;
	while (e) {
		if (e & 1)
			r = nt_mulmod(r, a, m);
		a = nt_mulmod(a, a, m);
		e >>= 1;
	}
	return r;
}

u64
nt_gcd(u64 a, u64 b)
{
	while (b) {
		u64 t = a % b;
		a = b;
		b = t;
	}
	return a;
}

u64
nt_inv(u64 a, u64 m, u64 *g)
{
	i128 t0 = 0, t1 = 1, r0 = m, r1 = a % m, q, t;

	while (r1) {
		q = r0 / r1;
		t = r0 - q * r1;
		r0 = r1;
		r1 = t;
		t = t0 - q * t1;
		t0 = t1;
		t1 = t;
	}
	if (g)
		*g = (u64)r0;
	if (r0 != 1)
		return 0;
	if (t0 < 0)
		t0 += m;
	return (u64)t0 % m;
}

u64
nt_isqrt(u64 n)
{
	u64 r = 0, b;

	for (b = (u64)1 << 31; b; b >>= 1) {
		u64 t = r | b;
		if ((u128)t * t <= n)
			r = t;
	}
	return r;
}

static int
mr_witness(u64 n, u64 a, u64 d, int s)
{
	u64 x = nt_powmod(a, d, n);
	int i;

	if (x == 1 || x == n - 1)
		return 0;
	for (i = 1; i < s; i++) {
		x = nt_mulmod(x, x, n);
		if (x == n - 1)
			return 0;
	}
	return 1;
}

int
nt_isprime(u64 n)
{
	static const u64 base[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
	u64 d;
	int s = 0, i;

	if (n < 2)
		return 0;
	for (i = 0; i < 12; i++) {
		if (n == base[i])
			return 1;
		if (n % base[i] == 0)
			return 0;
	}
	for (d = n - 1; !(d & 1); d >>= 1)
		s++;
	for (i = 0; i < 12; i++)
		if (mr_witness(n, base[i], d, s))
			return 0;
	return 1;
}

/* Pollard-Brent: a nontrivial factor of composite odd n */
static u64
rho(u64 n, u64 c)
{
	u64 y = 2, x, q = 1, g = 1, ys = 2, r = 1, k, i, m = 64;

	while (g == 1) {
		x = y;
		for (i = 0; i < r; i++)
			y = nt_addmod(nt_mulmod(y, y, n), c, n);
		for (k = 0; k < r && g == 1; k += m) {
			ys = y;
			for (i = 0; i < m && i < r - k; i++) {
				y = nt_addmod(nt_mulmod(y, y, n), c, n);
				q = nt_mulmod(q, x > y ? x - y : y - x, n);
			}
			g = nt_gcd(q, n);
		}
		r <<= 1;
	}
	if (g == n) {
		do {
			ys = nt_addmod(nt_mulmod(ys, ys, n), c, n);
			g = nt_gcd(x > ys ? x - ys : ys - x, n);
		} while (g == 1);
	}
	return g;
}

static void
fac_add(struct nt_fac *f, u64 p, int e)
{
	int i;

	for (i = 0; i < f->k; i++)
		if (f->p[i] == p) {
			f->e[i] += e;
			return;
		}
	f->p[f->k] = p;
	f->e[f->k++] = e;
}

static void
fac_rec(u64 n, struct nt_fac *f)
{
	u64 c, d;

	if (n == 1)
		return;
	if (nt_isprime(n)) {
		fac_add(f, n, 1);
		return;
	}
	for (c = 1;; c++) {
		d = rho(n, c);
		if (d != n && d != 1)
			break;
	}
	fac_rec(d, f);
	fac_rec(n / d, f);
}

void
nt_factor(u64 n, struct nt_fac *f)
{
	static const int sp[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
		43, 47};
	int i, j;

	f->k = 0;
	if (n < 2)
		return;
	for (i = 0; i < 15; i++)
		if (n % sp[i] == 0) {
			int e = 0;
			while (n % sp[i] == 0) {
				n /= sp[i];
				e++;
			}
			fac_add(f, sp[i], e);
		}
	fac_rec(n, f);
	/* sort */
	for (i = 1; i < f->k; i++)
		for (j = i; j > 0 && f->p[j - 1] > f->p[j]; j--) {
			u64 t = f->p[j];
			int te = f->e[j];
			f->p[j] = f->p[j - 1];
			f->e[j] = f->e[j - 1];
			f->p[j - 1] = t;
			f->e[j - 1] = te;
		}
}

static u64
ipow(u64 p, int e)
{
	u64 r = 1;

	while (e-- > 0)
		r *= p;
	return r;
}

u64
nt_phi(u64 n)
{
	struct nt_fac f;
	u64 r = n;
	int i;

	if (n == 0)
		return 0;
	nt_factor(n, &f);
	for (i = 0; i < f.k; i++)
		r = r / f.p[i] * (f.p[i] - 1);
	return r;
}

static u64
lcm64(u64 a, u64 b)
{
	return a / nt_gcd(a, b) * b;
}

u64
nt_lambda(u64 n)
{
	struct nt_fac f;
	u64 r = 1, l;
	int i;

	nt_factor(n, &f);
	for (i = 0; i < f.k; i++) {
		if (f.p[i] == 2)
			l = f.e[i] == 1 ? 1 : f.e[i] == 2 ? 2 : ipow(2, f.e[i] - 2);
		else
			l = ipow(f.p[i], f.e[i] - 1) * (f.p[i] - 1);
		r = lcm64(r, l);
	}
	return r;
}

u64
nt_order(u64 a, u64 n)
{
	struct nt_fac f;
	u64 o;
	int i;

	if (n == 1)
		return 1;
	a %= n;
	if (nt_gcd(a, n) != 1)
		return 0;
	o = nt_lambda(n);
	nt_factor(o, &f);
	for (i = 0; i < f.k; i++)
		while (o % f.p[i] == 0 && nt_powmod(a, o / f.p[i], n) == 1)
			o /= f.p[i];
	return o;
}

int
nt_jacobi(u64 a, u64 n)
{
	int s = 1;

	a %= n;
	while (a) {
		while (!(a & 1)) {
			a >>= 1;
			if ((n & 7) == 3 || (n & 7) == 5)
				s = -s;
		}
		{
			u64 t = a;
			a = n;
			n = t;
		}
		if ((a & 3) == 3 && (n & 3) == 3)
			s = -s;
		a %= n;
	}
	return n == 1 ? s : 0;
}

int
nt_sqrt_prime(u64 a, u64 p, u64 *r)
{
	u64 q, z, m, c, t, R, b;
	int s, i;

	a %= p;
	if (p == 2 || a == 0) {
		*r = a;
		return 1;
	}
	if (nt_powmod(a, (p - 1) / 2, p) != 1)
		return 0;
	if (p % 4 == 3) {
		*r = nt_powmod(a, (p + 1) / 4, p);
		return 1;
	}
	for (q = p - 1, s = 0; !(q & 1); q >>= 1)
		s++;
	for (z = 2; nt_powmod(z, (p - 1) / 2, p) != p - 1; z++)
		;
	m = (u64)s;
	c = nt_powmod(z, q, p);
	t = nt_powmod(a, q, p);
	R = nt_powmod(a, (q + 1) / 2, p);
	while (t != 1) {
		u64 tt = t;
		for (i = 0; tt != 1; i++)
			tt = nt_mulmod(tt, tt, p);
		b = c;
		for (; (u64)i + 1 < m; m--)
			b = nt_mulmod(b, b, p);
		m = (u64)i;
		c = nt_mulmod(b, b, p);
		t = nt_mulmod(t, c, p);
		R = nt_mulmod(R, b, p);
	}
	*r = R;
	return 1;
}

static int
cmpu64(const void *a, const void *b)
{
	u64 x = *(const u64 *)a, y = *(const u64 *)b;

	return x < y ? -1 : x > y;
}

/* roots of x^2 = a mod p^e: from mod p, lifted one power at a time */
static int
sqrt_pp(u64 a, u64 p, int e, u64 *out, int max)
{
	u64 *cur = malloc(sizeof(u64) * (size_t)max), pk = p, r;
	int n = 0, k, i, m;

	if (p == 2) {
		for (r = 0; r < 2; r++)
			if ((r * r - a) % 2 == 0)
				cur[n++] = r;
	} else if (nt_sqrt_prime(a % p, p, &r)) {
		cur[n++] = r;
		if (r != 0)
			cur[n++] = p - r;
	}
	for (k = 1; k < e && n > 0; k++) {
		u64 pk1 = pk * p;
		u64 *nx = malloc(sizeof(u64) * (size_t)max);
		m = 0;
		for (i = 0; i < n; i++) {
			u64 r0 = cur[i], am = a % pk1;
			if (p != 2 && r0 % p != 0) {
				/* Hensel: t = (a - r^2)/p^k / (2r) mod p */
				u64 d = nt_submod(am, nt_mulmod(r0, r0, pk1), pk1) / pk;
				u64 t = nt_mulmod(d % p, nt_inv(2 * r0 % p, p, NULL), p);
				if (m >= max)
					goto over;
				nx[m++] = r0 + t * pk;
			} else if (nt_mulmod(r0, r0, pk1) == am) {
				/* every lift is a root, or none is */
				u64 t;
				if ((u64)m + p > (u64)max)
					goto over;
				for (t = 0; t < p; t++)
					nx[m++] = r0 + t * pk;
			}
			continue;
over:
			free(nx);
			free(cur);
			return -1;
		}
		free(cur);
		cur = nx;
		n = m;
		pk = pk1;
	}
	memcpy(out, cur, sizeof(u64) * (size_t)n);
	free(cur);
	return n;
}

int
nt_sqrt_all(u64 a, u64 n, u64 *roots, int max)
{
	struct nt_fac f;
	u64 *acc, *part, *nx, m = 1;
	int na = 1, np, i, j, k, nn;

	if (n == 1) {
		roots[0] = 0;
		return 1;
	}
	a %= n;
	nt_factor(n, &f);
	acc = malloc(sizeof(u64) * (size_t)max);
	part = malloc(sizeof(u64) * (size_t)max);
	nx = malloc(sizeof(u64) * (size_t)max);
	acc[0] = 0;
	for (k = 0; k < f.k; k++) {
		u64 pe = ipow(f.p[k], f.e[k]), r, mm;
		np = sqrt_pp(a % pe, f.p[k], f.e[k], part, max);
		if (np < 0 || (u128)na * np > (u128)max) {
			na = -1;
			goto done;
		}
		nn = 0;
		for (i = 0; i < na; i++)
			for (j = 0; j < np; j++) {
				nt_crt(acc[i], m, part[j], pe, &r, &mm);
				nx[nn++] = r;
			}
		memcpy(acc, nx, sizeof(u64) * (size_t)nn);
		na = nn;
		m *= pe;
		if (na == 0)
			break;
	}
	qsort(acc, (size_t)na, sizeof(u64), cmpu64);
	memcpy(roots, acc, sizeof(u64) * (size_t)na);
done:
	free(acc);
	free(part);
	free(nx);
	return na;
}

int
nt_has_primroot(u64 n)
{
	struct nt_fac f;

	if (n == 1 || n == 2 || n == 4)
		return 1;
	if (n % 4 == 0)
		return 0;
	if (n % 2 == 0)
		n /= 2;
	nt_factor(n, &f);
	return f.k == 1 && f.p[0] != 2;
}

u64
nt_primroot(u64 n)
{
	struct nt_fac f;
	u64 ph, g;
	int i;

	if (!nt_has_primroot(n))
		return 0;
	if (n <= 2)
		return n - 1;
	ph = nt_phi(n);
	nt_factor(ph, &f);
	for (g = 2; g < n; g++) {
		if (nt_gcd(g, n) != 1)
			continue;
		for (i = 0; i < f.k; i++)
			if (nt_powmod(g, ph / f.p[i], n) == 1)
				break;
		if (i == f.k)
			return g;
	}
	return 0;
}

int
nt_crt(u64 r1, u64 m1, u64 r2, u64 m2, u64 *r, u64 *m)
{
	u64 g = nt_gcd(m1, m2), l2, d, t, inv;
	u128 L;

	r1 %= m1;
	r2 %= m2;
	if ((r1 > r2 ? r1 - r2 : r2 - r1) % g)
		return 0;
	L = (u128)(m1 / g) * m2;
	if (L >> 64)
		return -1;
	l2 = m2 / g;
	d = (nt_submod(r2 % m2, r1 % m2, m2)) / g % l2;
	inv = nt_inv((m1 / g) % l2, l2, NULL);
	t = nt_mulmod(d, inv, l2 ? l2 : 1);
	*m = (u64)L;
	*r = (u64)(((u128)m1 * t + r1) % L);
	return 1;
}
