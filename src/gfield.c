/*
 * Z/n and Z/n[x]/(f) as rings of u64 codes, and polynomials and matrices
 * over them.  For a field GF(q): irreducibility (Rabin, Ben-Or, and
 * Berlekamp's count of factors as an independent check), factoring
 * (square-free split, distinct-degree, Cantor-Zassenhaus equal-degree
 * split), the order of x mod f, discrete logs (Pohlig-Hellman with
 * baby-step giant-step), and Gaussian elimination.  Over Z/n with n
 * composite only unimodular row operations are used (Euclid on a column),
 * so determinants and inverses are still right there.
 */
#include "ring.h"
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

/* ---- the ring ---- */

void
ring_zn(struct ring *R, u64 n)
{
	memset(R, 0, sizeof *R);
	R->deg = 1;
	R->n = n;
	R->nprime = nt_isprime(n);
	R->field = R->nprime;
	R->qm1 = n - 1;
}

static int rp_from_mod(const struct ring *B, struct rpoly *f, const u64 *c,
    int deg);

int
ring_ext(struct ring *R, u64 n, int deg, const u64 *f)
{
	u128 q = 1;
	int i;

	memset(R, 0, sizeof *R);
	for (i = 0; i < deg; i++) {
		q *= n;
		if (q > ((u128)1 << 64))
			return 0;
	}
	R->deg = deg;
	R->n = n;
	R->nprime = nt_isprime(n);
	R->qm1 = (u64)(q - 1);
	for (i = 0; i <= deg; i++)
		R->f[i] = f[i] % n;
	R->f[deg] = 1;
	if (deg == 1) {
		/* Z/n[x]/(x - c) is Z/n itself */
		R->field = R->nprime;
		return 1;
	}
	if (R->nprime) {
		struct ring B;
		struct rpoly fp;
		ring_zn(&B, n);
		rp_from_mod(&B, &fp, R->f, deg);
		R->field = rp_irreducible(&B, &fp);
	}
	return 1;
}

void
ring_free(struct ring *R)
{
	free(R->lg);
	free(R->ex);
	R->lg = R->ex = NULL;
}

u64
r_charp(const struct ring *R)
{
	return R->n;
}

void
r_digits(const struct ring *R, u64 a, u64 *d)
{
	int i;

	if (R->deg == 1) {
		d[0] = a;
		return;
	}
	if (R->n == 2) {
		for (i = 0; i < R->deg; i++)
			d[i] = a >> i & 1;
		return;
	}
	for (i = 0; i < R->deg; i++) {
		d[i] = a % R->n;
		a /= R->n;
	}
}

u64
r_pack(const struct ring *R, const u64 *d)
{
	u64 a = 0;
	int i;

	if (R->deg == 1)
		return d[0];
	if (R->n == 2) {
		for (i = 0; i < R->deg; i++)
			a |= (d[i] & 1) << i;
		return a;
	}
	for (i = R->deg - 1; i >= 0; i--)
		a = a * R->n + d[i];
	return a;
}

u64
r_add(const struct ring *R, u64 a, u64 b)
{
	u64 x[RMAXF], y[RMAXF];
	int i;

	if (R->deg == 1)
		return nt_addmod(a, b, R->n);
	if (R->n == 2)
		return a ^ b;
	r_digits(R, a, x);
	r_digits(R, b, y);
	for (i = 0; i < R->deg; i++)
		x[i] = nt_addmod(x[i], y[i], R->n);
	return r_pack(R, x);
}

u64
r_neg(const struct ring *R, u64 a)
{
	u64 x[RMAXF];
	int i;

	if (R->deg == 1)
		return a ? R->n - a : 0;
	if (R->n == 2)
		return a;
	r_digits(R, a, x);
	for (i = 0; i < R->deg; i++)
		x[i] = x[i] ? R->n - x[i] : 0;
	return r_pack(R, x);
}

u64
r_sub(const struct ring *R, u64 a, u64 b)
{
	return r_add(R, a, r_neg(R, b));
}

/* GF(2)[x]/(f): carry-less product, then reduction */
static u64
mul2(const struct ring *R, u64 a, u64 b)
{
	u128 p = 0, fm = 0;
	int i, d = R->deg;

	for (i = 0; i < 64; i++)
		if (b >> i & 1)
			p ^= (u128)a << i;
	for (i = 0; i <= d; i++)
		if (R->f[i])
			fm |= (u128)1 << i;
	for (i = 2 * d - 2; i >= d; i--)
		if (p >> i & 1)
			p ^= fm << (i - d);
	return (u64)p;
}

u64
r_mul(const struct ring *R, u64 a, u64 b)
{
	u64 x[RMAXF], y[RMAXF], z[2 * RMAXF];
	int i, j, d = R->deg;

	if (d == 1)
		return nt_mulmod(a, b, R->n);
	if (R->n == 2)
		return mul2(R, a, b);
	r_digits(R, a, x);
	r_digits(R, b, y);
	memset(z, 0, sizeof z);
	for (i = 0; i < d; i++)
		if (x[i])
			for (j = 0; j < d; j++)
				z[i + j] = nt_addmod(z[i + j],
				    nt_mulmod(x[i], y[j], R->n), R->n);
	for (i = 2 * d - 2; i >= d; i--)
		if (z[i]) {
			u64 c = z[i];
			for (j = 0; j <= d; j++)
				z[i - d + j] = nt_submod(z[i - d + j],
				    nt_mulmod(c, R->f[j], R->n), R->n);
		}
	return r_pack(R, z);
}

/*
 * The same product another way, for checks: shift-and-add (the "xtime"
 * loop: b's digits from the top, multiplying the accumulator by x and
 * reducing at each step), or exp(log a + log b) when tables exist.
 */
u64
r_mul_alt(const struct ring *R, u64 a, u64 b)
{
	u64 y[RMAXF], acc = 0, xx;
	int i;

	if (R->lg && a && b) {
		u64 s = (u64)R->lg[a] + R->lg[b];
		return R->ex[s % R->qm1];
	}
	if (R->deg == 1) {
		/* double-and-add */
		u64 r = 0;
		a %= R->n;
		while (b) {
			if (b & 1)
				r = nt_addmod(r, a, R->n);
			a = nt_addmod(a, a, R->n);
			b >>= 1;
		}
		return r;
	}
	r_digits(R, b, y);
	xx = r_x(R);
	for (i = R->deg - 1; i >= 0; i--) {
		u64 k;
		acc = r_mul(R, acc, xx);
		for (k = 0; k < y[i]; k++)
			acc = r_add(R, acc, a);
	}
	return acc;
}

u64
r_pow(const struct ring *R, u64 a, u64 e)
{
	u64 r = r_const(R, 1);

	while (e) {
		if (e & 1)
			r = r_mul(R, r, a);
		a = r_mul(R, a, a);
		e >>= 1;
	}
	return r;
}

u64
r_const(const struct ring *R, u64 k)
{
	u64 d[RMAXF];

	if (R->deg == 1)
		return k % R->n;
	memset(d, 0, sizeof d);
	d[0] = k % R->n;
	return r_pack(R, d);
}

u64
r_x(const struct ring *R)
{
	u64 d[RMAXF];

	if (R->deg == 1)
		return nt_submod(0, R->f[0] % R->n, R->n);  /* x = -f0 */
	memset(d, 0, sizeof d);
	d[1] = 1;
	return r_pack(R, d);
}

u64
r_frob(const struct ring *R, u64 a)
{
	return r_pow(R, a, R->n);
}

static void
code_poly(const struct ring *R, u64 a, struct rpoly *p)
{
	u64 d[RMAXF];
	int i;

	r_digits(R, a, d);
	for (i = 0; i < R->deg; i++)
		p->c[i] = d[i];
	p->d = R->deg - 1;
	rp_norm(p);
}

int
r_inv(const struct ring *R, u64 a, u64 *out)
{
	u64 g;

	if (R->deg == 1) {
		*out = nt_inv(a, R->n, &g);
		return g == 1;
	}
	if (R->field) {
		if (a == 0)
			return 0;
		*out = r_pow(R, a, R->qm1 - 1);
		return 1;
	}
	if (R->nprime) {
		/* extended Euclid in GF(p)[x]: s a + t f = 1 */
		struct ring B;
		struct rpoly A, F, G, S, T;
		u64 d[RMAXF];
		int i;
		ring_zn(&B, R->n);
		code_poly(R, a, &A);
		rp_from_mod(&B, &F, R->f, R->deg);
		rp_xgcd(&B, &G, &S, &T, &A, &F);
		if (G.d != 0)
			return 0;
		memset(d, 0, sizeof d);
		for (i = 0; i <= S.d; i++)
			d[i] = S.c[i];
		*out = r_pack(R, d);
		return 1;
	}
	/* Z/n[x]/(f), n composite: small rings by search */
	if (R->qm1 < (1u << 20)) {
		u64 b, one = r_const(R, 1);
		for (b = 1; b <= R->qm1; b++)
			if (r_mul(R, a, b) == one) {
				*out = b;
				return 1;
			}
	}
	return 0;
}

int
r_isunit(const struct ring *R, u64 a)
{
	u64 t;

	return r_inv(R, a, &t);
}

/* the order of a unit: divides the group exponent */
u64
r_order(const struct ring *R, u64 a)
{
	struct nt_fac f;
	u64 o, one = r_const(R, 1);
	int i;

	if (R->deg == 1)
		return nt_order(a, R->n);
	if (!r_isunit(R, a))
		return 0;
	if (!R->field) {
		/* the unit group's order is not known in general: count */
		u64 k, x = a;
		for (k = 1; k <= R->qm1; k++) {
			if (x == one)
				return k;
			x = r_mul(R, x, a);
		}
		return 0;
	}
	o = R->qm1;
	nt_factor(o, &f);
	for (i = 0; i < f.k; i++)
		while (o % f.p[i] == 0 && r_pow(R, a, o / f.p[i]) == one)
			o /= f.p[i];
	return o;
}

void
ring_tables(struct ring *R)
{
	u64 g, x;
	uint32_t i;

	if (!R->field || R->gen)
		return;
	if (R->deg == 1 && R->n == 2)
		g = 1;
	else {
		/* x first (a primitive modulus), then the codes in order */
		g = r_x(R);
		if (R->deg == 1 || r_order(R, g) != R->qm1)
			for (g = 2; g <= R->qm1; g++)
				if (r_order(R, g) == R->qm1)
					break;
	}
	R->gen = g;
	if (R->qm1 >= (1u << 16))
		return;
	R->lg = xcalloc((size_t)R->qm1 + 1, sizeof *R->lg);
	R->ex = xcalloc((size_t)R->qm1 + 1, sizeof *R->ex);
	x = r_const(R, 1);
	for (i = 0; i < R->qm1; i++) {
		R->ex[i] = (uint32_t)x;
		R->lg[x] = i;
		x = r_mul(R, x, g);
	}
}

/* ---- discrete logarithm: Pohlig-Hellman over BSGS ---- */

struct bs {
	u64 *k;
	u64 *v;
	uint8_t *u;
	size_t cap;
};

static size_t
bs_h(u64 k, size_t cap)
{
	k ^= k >> 33;
	k *= 0xff51afd7ed558ccdULL;
	k ^= k >> 33;
	return (size_t)k & (cap - 1);
}

/* g^x = h with x < n (g of order n); 0 if none */
static int
bsgs(const struct ring *R, u64 g, u64 h, u64 n, u64 *x)
{
	struct bs t;
	u64 m = nt_isqrt(n) + 1, j, i, e, gm, ginv;
	size_t cap = 16, s;

	while (cap < 2 * m)
		cap <<= 1;
	t.cap = cap;
	t.k = xcalloc(cap, sizeof *t.k);
	t.v = xcalloc(cap, sizeof *t.v);
	t.u = xcalloc(cap, 1);
	e = r_const(R, 1);
	for (j = 0; j < m; j++) {
		for (s = bs_h(e, cap); t.u[s] && t.k[s] != e; s = (s + 1) & (cap - 1))
			;
		if (!t.u[s]) {
			t.u[s] = 1;
			t.k[s] = e;
			t.v[s] = j;
		}
		e = r_mul(R, e, g);
	}
	r_inv(R, g, &ginv);
	gm = r_pow(R, ginv, m);
	e = h;
	for (i = 0; i <= m; i++) {
		for (s = bs_h(e, cap); t.u[s]; s = (s + 1) & (cap - 1))
			if (t.k[s] == e) {
				u128 r = (u128)i * m + t.v[s];
				free(t.k);
				free(t.v);
				free(t.u);
				*x = (u64)(r % n);
				return 1;
			}
		e = r_mul(R, e, gm);
	}
	free(t.k);
	free(t.v);
	free(t.u);
	return 0;
}

/*
 * x with g^x = a; *ord is the order of g.  0: a is not a power of g;
 * -1: a prime factor of the order is too large for BSGS.
 */
int
r_dlog(const struct ring *R, u64 g, u64 a, u64 *x, u64 *ord)
{
	struct nt_fac f;
	u64 n = r_order(R, g), res = 0, mod = 1, one = r_const(R, 1);
	int i, k;

	*ord = n;
	if (n == 0 || !r_isunit(R, a))
		return 0;
	if (r_pow(R, a, n) != one)
		return 0;
	nt_factor(n, &f);
	for (i = 0; i < f.k; i++) {
		u64 p = f.p[i], pe = 1, xi = 0, gp, ginv, pk = 1;
		if (p > ((u64)1 << 44))
			return -1;
		for (k = 0; k < f.e[i]; k++)
			pe *= p;
		/* x mod p^e, one digit at a time: g0 = g^(n/p) has order p */
		gp = r_pow(R, g, n / p);
		r_inv(R, g, &ginv);
		for (k = 0; k < f.e[i]; k++) {
			u64 hk, d, t;
			/* (a g^-xi)^(n/p^(k+1)) = gp^d */
			t = r_mul(R, a, r_pow(R, ginv, xi));
			hk = r_pow(R, t, n / (pk * p));
			if (!bsgs(R, gp, hk, p, &d))
				return 0;
			xi += d * pk;
			pk *= p;
		}
		if (nt_crt(res, mod, xi, pe, &res, &mod) != 1)
			return 0;
	}
	*x = res % n;
	return r_pow(R, g, *x) == a;
}

/* ---- polynomials over a ring ---- */

void
rp_zero(struct rpoly *a)
{
	a->d = -1;
}

void
rp_norm(struct rpoly *a)
{
	while (a->d >= 0 && a->c[a->d] == 0)
		a->d--;
}

void
rp_const(const struct ring *R, struct rpoly *a, u64 c)
{
	a->d = 0;
	a->c[0] = c;
	rp_norm(a);
}

void
rp_x(struct rpoly *a)
{
	a->d = 1;
	a->c[0] = 0;
	a->c[1] = 1;
}

/* a modulus f[0..deg] (codes of the base ring) as a polynomial */
static int
rp_from_mod(const struct ring *B, struct rpoly *f, const u64 *c, int deg)
{
	int i;

	for (i = 0; i <= deg; i++)
		f->c[i] = c[i];
	f->d = deg;
	rp_norm(f);
	return 1;
}

int
rp_eq(const struct rpoly *a, const struct rpoly *b)
{
	int i;

	if (a->d != b->d)
		return 0;
	for (i = 0; i <= a->d; i++)
		if (a->c[i] != b->c[i])
			return 0;
	return 1;
}

void
rp_add(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b)
{
	int i, d = a->d > b->d ? a->d : b->d;

	for (i = 0; i <= d; i++)
		r->c[i] = r_add(R, i <= a->d ? a->c[i] : 0, i <= b->d ? b->c[i] : 0);
	r->d = d;
	rp_norm(r);
}

void
rp_sub(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b)
{
	int i, d = a->d > b->d ? a->d : b->d;

	for (i = 0; i <= d; i++)
		r->c[i] = r_sub(R, i <= a->d ? a->c[i] : 0, i <= b->d ? b->c[i] : 0);
	r->d = d;
	rp_norm(r);
}

int
rp_mul(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b)
{
	static struct rpoly t;
	int i, j;

	if (a->d < 0 || b->d < 0) {
		r->d = -1;
		return 1;
	}
	if (a->d + b->d >= RMAXD)
		return 0;
	t.d = a->d + b->d;
	for (i = 0; i <= t.d; i++)
		t.c[i] = 0;
	for (i = 0; i <= a->d; i++)
		if (a->c[i])
			for (j = 0; j <= b->d; j++)
				if (b->c[j])
					t.c[i + j] = r_add(R, t.c[i + j],
					    r_mul(R, a->c[i], b->c[j]));
	rp_norm(&t);
	*r = t;
	return 1;
}

void
rp_scale(const struct ring *R, struct rpoly *r, const struct rpoly *a, u64 k)
{
	int i;

	for (i = 0; i <= a->d; i++)
		r->c[i] = r_mul(R, a->c[i], k);
	r->d = a->d;
	rp_norm(r);
}

int
rp_divmod(const struct ring *R, const struct rpoly *a, const struct rpoly *b,
    struct rpoly *q, struct rpoly *r)
{
	struct rpoly rr = *a, qq;
	u64 li;
	int i, j;

	if (b->d < 0 || !r_inv(R, b->c[b->d], &li))
		return 0;
	qq.d = a->d - b->d;
	if (qq.d < 0)
		qq.d = -1;
	for (i = 0; i <= qq.d; i++)
		qq.c[i] = 0;
	for (i = rr.d; i >= b->d; i--) {
		u64 c = rr.c[i];
		if (c == 0)
			continue;
		c = r_mul(R, c, li);
		qq.c[i - b->d] = c;
		for (j = 0; j <= b->d; j++)
			rr.c[i - b->d + j] = r_sub(R, rr.c[i - b->d + j],
			    r_mul(R, c, b->c[j]));
	}
	rr.d = b->d - 1 < rr.d ? b->d - 1 : rr.d;
	rp_norm(&rr);
	rp_norm(&qq);
	if (q)
		*q = qq;
	if (r)
		*r = rr;
	return 1;
}

int
rp_monic(const struct ring *R, struct rpoly *a)
{
	u64 li;

	if (a->d < 0)
		return 1;
	if (!r_inv(R, a->c[a->d], &li))
		return 0;
	rp_scale(R, a, a, li);
	return 1;
}

int
rp_gcd(const struct ring *R, struct rpoly *g, const struct rpoly *a,
    const struct rpoly *b)
{
	struct rpoly x = *a, y = *b, r;

	while (y.d >= 0) {
		if (!rp_divmod(R, &x, &y, NULL, &r))
			return 0;
		x = y;
		y = r;
	}
	*g = x;
	return rp_monic(R, g);
}

int
rp_xgcd(const struct ring *R, struct rpoly *g, struct rpoly *s,
    struct rpoly *t, const struct rpoly *a, const struct rpoly *b)
{
	struct rpoly r0 = *a, r1 = *b, s0, s1, t0, t1, q, tmp, m;
	u64 li;

	rp_const(R, &s0, r_const(R, 1));
	rp_zero(&s1);
	rp_zero(&t0);
	rp_const(R, &t1, r_const(R, 1));
	while (r1.d >= 0) {
		if (!rp_divmod(R, &r0, &r1, &q, &tmp))
			return 0;
		r0 = r1;
		r1 = tmp;
		rp_mul(R, &m, &q, &s1);
		rp_sub(R, &tmp, &s0, &m);
		s0 = s1;
		s1 = tmp;
		rp_mul(R, &m, &q, &t1);
		rp_sub(R, &tmp, &t0, &m);
		t0 = t1;
		t1 = tmp;
	}
	if (r0.d >= 0 && r_inv(R, r0.c[r0.d], &li)) {
		rp_scale(R, &r0, &r0, li);
		rp_scale(R, &s0, &s0, li);
		rp_scale(R, &t0, &t0, li);
	}
	*g = r0;
	*s = s0;
	*t = t0;
	return 1;
}

void
rp_deriv(const struct ring *R, struct rpoly *r, const struct rpoly *a)
{
	int i;

	for (i = 1; i <= a->d; i++)
		r->c[i - 1] = r_mul(R, a->c[i], r_const(R, (u64)i % R->n));
	r->d = a->d - 1;
	rp_norm(r);
}

void
rp_mulmod(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b, const struct rpoly *f)
{
	struct rpoly t;

	if (!rp_mul(R, &t, a, b))
		fail("a polynomial product has degree above %d", RMAXD - 1);
	rp_divmod(R, &t, f, NULL, r);
}

void
rp_powmod(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    u64 e, const struct rpoly *f)
{
	struct rpoly b, acc;

	rp_divmod(R, a, f, NULL, &b);
	rp_const(R, &acc, r_const(R, 1));
	if (f->d == 0)
		rp_zero(&acc);
	while (e) {
		if (e & 1)
			rp_mulmod(R, &acc, &acc, &b, f);
		rp_mulmod(R, &b, &b, &b, f);
		e >>= 1;
	}
	*r = acc;
}

/* a^q mod f, q = qm1 + 1 (which may be 2^64) */
void
rp_frobmod(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *f)
{
	struct rpoly t, b;

	rp_divmod(R, a, f, NULL, &b);
	rp_powmod(R, &t, &b, R->qm1, f);
	rp_mulmod(R, r, &t, &b, f);
}

u64
rp_eval(const struct ring *R, const struct rpoly *a, u64 x)
{
	u64 v = 0;
	int i;

	for (i = a->d; i >= 0; i--)
		v = r_add(R, r_mul(R, v, x), a->c[i]);
	return v;
}

/* x^(q^k) mod f */
static void
xqk(const struct ring *R, struct rpoly *r, int k, const struct rpoly *f)
{
	struct rpoly x;
	int i;

	rp_x(&x);
	rp_divmod(R, &x, f, NULL, r);
	for (i = 0; i < k; i++)
		rp_frobmod(R, r, r, f);
}

/*
 * Rabin: f of degree n is irreducible iff x^(q^n) = x mod f and
 * gcd(x^(q^(n/r)) - x, f) = 1 for each prime r | n.
 */
int
rp_irreducible(const struct ring *R, const struct rpoly *f0)
{
	struct rpoly f = *f0, h, x, g;
	struct nt_fac nf;
	int i, n;

	if (!rp_monic(R, &f) || f.d < 1)
		return 0;
	n = f.d;
	if (n == 1)
		return 1;
	rp_x(&x);
	nt_factor((u64)n, &nf);
	for (i = 0; i < nf.k; i++) {
		xqk(R, &h, n / (int)nf.p[i], &f);
		rp_sub(R, &h, &h, &x);
		rp_gcd(R, &g, &h, &f);
		if (g.d != 0)
			return 0;
	}
	xqk(R, &h, n, &f);
	return rp_eq(&h, &x) || (rp_sub(R, &h, &h, &x), h.d < 0);
}

/* Ben-Or: gcd(x^(q^i) - x, f) = 1 for i = 1..n/2 */
int
rp_benor(const struct ring *R, const struct rpoly *f0)
{
	struct rpoly f = *f0, h, t, x, g;
	int i;

	if (!rp_monic(R, &f) || f.d < 1)
		return 0;
	rp_x(&x);
	rp_divmod(R, &x, &f, NULL, &h);
	for (i = 1; i <= f.d / 2; i++) {
		rp_frobmod(R, &h, &h, &f);
		rp_sub(R, &t, &h, &x);
		rp_gcd(R, &g, &t, &f);
		if (g.d != 0)
			return 0;
	}
	return 1;
}

/* ---- matrices ---- */

void
rm_init(struct rmat *m, int r, int c)
{
	m->r = r;
	m->c = c;
	m->a = xcalloc((size_t)(r > 0 ? r : 1) * (c > 0 ? c : 1), sizeof *m->a);
}

void
rm_free(struct rmat *m)
{
	free(m->a);
	m->a = NULL;
}

static void
row_swap(struct rmat *m, int i, int j)
{
	int k;

	for (k = 0; k < m->c; k++) {
		u64 t = RM(m, i, k);
		RM(m, i, k) = RM(m, j, k);
		RM(m, j, k) = t;
	}
}

/* row i -= c * row j */
static void
row_sub(const struct ring *R, struct rmat *m, int i, int j, u64 c)
{
	int k;

	if (c == 0)
		return;
	for (k = 0; k < m->c; k++)
		RM(m, i, k) = r_sub(R, RM(m, i, k), r_mul(R, c, RM(m, j, k)));
}

/* fields: reduced row echelon form in place; piv[r] = pivot column */
int
rm_rref(const struct ring *R, struct rmat *m, int *piv)
{
	int r = 0, c, i, k;

	for (c = 0; c < m->c && r < m->r; c++) {
		u64 li;
		for (i = r; i < m->r && RM(m, i, c) == 0; i++)
			;
		if (i == m->r)
			continue;
		row_swap(m, i, r);
		r_inv(R, RM(m, r, c), &li);
		for (k = 0; k < m->c; k++)
			RM(m, r, k) = r_mul(R, RM(m, r, k), li);
		for (i = 0; i < m->r; i++)
			if (i != r)
				row_sub(R, m, i, r, RM(m, i, c));
		if (piv)
			piv[r] = c;
		r++;
	}
	return r;
}

/*
 * Z/n, n composite: make column c below row r zero by Euclid on the
 * entries (as integers 0..n-1), with swaps and row subtractions only.
 * Returns the number of swaps.
 */
static int
euclid_col(const struct ring *R, struct rmat *m, int r, int c)
{
	int swaps = 0, i, best;

	for (;;) {
		best = -1;
		for (i = r; i < m->r; i++)
			if (RM(m, i, c) && (best < 0 || RM(m, i, c) < RM(m, best, c)))
				best = i;
		if (best < 0)
			return swaps;
		if (best != r) {
			row_swap(m, best, r);
			swaps++;
		}
		for (i = r + 1; i < m->r; i++)
			if (RM(m, i, c))
				break;
		if (i == m->r)
			return swaps;
		for (i = r + 1; i < m->r; i++)
			row_sub(R, m, i, r, RM(m, i, c) / RM(m, r, c));
	}
}

int
rm_det(const struct ring *R, const struct rmat *m0, u64 *det)
{
	struct rmat m;
	u64 d = r_const(R, 1);
	int i, j, swaps = 0;

	if (m0->r != m0->c)
		return 0;
	rm_init(&m, m0->r, m0->c);
	memcpy(m.a, m0->a, sizeof(u64) * (size_t)m.r * m.c);
	for (j = 0; j < m.c; j++) {
		if (R->field) {
			for (i = j; i < m.r && RM(&m, i, j) == 0; i++)
				;
			if (i == m.r) {
				d = 0;
				break;
			}
			if (i != j) {
				row_swap(&m, i, j);
				swaps++;
			}
			{
				u64 li;
				r_inv(R, RM(&m, j, j), &li);
				for (i = j + 1; i < m.r; i++)
					row_sub(R, &m, i, j, r_mul(R, RM(&m, i, j), li));
			}
		} else if (R->deg == 1)
			swaps += euclid_col(R, &m, j, j);
		else {
			rm_free(&m);
			return 0;
		}
		d = r_mul(R, d, RM(&m, j, j));
	}
	if (swaps & 1)
		d = r_neg(R, d);
	*det = d;
	rm_free(&m);
	return 1;
}

/* the inverse: Gauss-Jordan on [A | I]; over Z/n Euclid first */
int
rm_inverse(const struct ring *R, const struct rmat *a, struct rmat *inv)
{
	struct rmat m;
	int n = a->r, i, j, k;

	if (a->r != a->c || (!R->field && R->deg != 1))
		return 0;
	rm_init(&m, n, 2 * n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			RM(&m, i, j) = RM(a, i, j);
		RM(&m, i, n + i) = r_const(R, 1);
	}
	for (j = 0; j < n; j++) {
		u64 li;
		if (!R->field)
			euclid_col(R, &m, j, j);
		else {
			for (i = j; i < n && RM(&m, i, j) == 0; i++)
				;
			if (i < n)
				row_swap(&m, i, j);
		}
		if (!r_inv(R, RM(&m, j, j), &li)) {
			rm_free(&m);
			return 0;
		}
		for (k = 0; k < 2 * n; k++)
			RM(&m, j, k) = r_mul(R, RM(&m, j, k), li);
		for (i = 0; i < n; i++)
			if (i != j)
				row_sub(R, &m, i, j, RM(&m, i, j));
	}
	rm_init(inv, n, n);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			RM(inv, i, j) = RM(&m, i, n + j);
	rm_free(&m);
	return 1;
}

/* fields: a basis of {v : m v = 0}, one vector per row of ns */
int
rm_nullspace(const struct ring *R, const struct rmat *m0, struct rmat *ns)
{
	struct rmat m;
	int *piv, rank, i, j, k, free_n = 0, *isp;

	rm_init(&m, m0->r, m0->c);
	memcpy(m.a, m0->a, sizeof(u64) * (size_t)m.r * m.c);
	piv = xcalloc((size_t)m.r + 1, sizeof *piv);
	isp = xcalloc((size_t)m.c + 1, sizeof *isp);
	rank = rm_rref(R, &m, piv);
	for (i = 0; i < rank; i++)
		isp[piv[i]] = 1;
	rm_init(ns, m.c - rank, m.c);
	for (j = 0; j < m.c; j++) {
		if (isp[j])
			continue;
		RM(ns, free_n, j) = r_const(R, 1);
		for (k = 0; k < rank; k++)
			RM(ns, free_n, piv[k]) = r_neg(R, RM(&m, k, j));
		free_n++;
	}
	free(piv);
	free(isp);
	rm_free(&m);
	return free_n;
}

int
rm_mul(const struct ring *R, struct rmat *r, const struct rmat *a,
    const struct rmat *b)
{
	int i, j, k;

	if (a->c != b->r)
		return 0;
	rm_init(r, a->r, b->c);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < b->c; j++) {
			u64 s = 0;
			for (k = 0; k < a->c; k++)
				s = r_add(R, s, r_mul(R, RM(a, i, k), RM(b, k, j)));
			RM(r, i, j) = s;
		}
	return 1;
}

/*
 * Characteristic polynomial det(xI - M) over a field: reduce to upper
 * Hessenberg form by similarity, then the standard recurrence.
 */
void
rm_charpoly(const struct ring *R, const struct rmat *m0, struct rpoly *cp)
{
	static struct rpoly p[RMAXF + 1];
	struct rmat h;
	int n = m0->r, i, j, k, mm;

	rm_init(&h, n, n);
	memcpy(h.a, m0->a, sizeof(u64) * (size_t)n * n);
	for (mm = 1; mm < n - 1; mm++) {
		u64 li;
		for (i = mm; i < n && RM(&h, i, mm - 1) == 0; i++)
			;
		if (i == n)
			continue;
		if (i != mm) {
			row_swap(&h, i, mm);
			for (k = 0; k < n; k++) {
				u64 t = RM(&h, k, i);
				RM(&h, k, i) = RM(&h, k, mm);
				RM(&h, k, mm) = t;
			}
		}
		r_inv(R, RM(&h, mm, mm - 1), &li);
		for (i = mm + 1; i < n; i++) {
			u64 u = r_mul(R, RM(&h, i, mm - 1), li);
			if (u == 0)
				continue;
			/* row i -= u row mm, then column mm += u column i */
			for (k = 0; k < n; k++)
				RM(&h, i, k) = r_sub(R, RM(&h, i, k), r_mul(R, u,
				    RM(&h, mm, k)));
			for (k = 0; k < n; k++)
				RM(&h, k, mm) = r_add(R, RM(&h, k, mm), r_mul(R, u,
				    RM(&h, k, i)));
		}
	}
	/* p[k]: charpoly of the leading k x k block */
	rp_const(R, &p[0], r_const(R, 1));
	for (k = 1; k <= n; k++) {
		struct rpoly t, xm;
		u64 prod = r_const(R, 1);
		rp_x(&xm);
		xm.c[0] = r_neg(R, RM(&h, k - 1, k - 1));
		rp_norm(&xm);
		rp_mul(R, &p[k], &xm, &p[k - 1]);
		for (i = 1; i < k; i++) {
			prod = r_mul(R, prod, RM(&h, k - i, k - i - 1));
			rp_scale(R, &t, &p[k - i - 1], r_mul(R, prod,
			    RM(&h, k - i - 1, k - 1)));
			rp_sub(R, &p[k], &p[k], &t);
		}
	}
	*cp = p[n];
	(void)j;
	rm_free(&h);
}

/*
 * Berlekamp: for square-free f, the number of distinct irreducible
 * factors is the nullity of Q - I, Q's rows x^(q i) mod f.
 */
int
rp_berlekamp_count(const struct ring *R, const struct rpoly *f0)
{
	struct rpoly f = *f0, xq, row;
	struct rmat Q;
	int n, i, j, rank;

	if (!rp_monic(R, &f) || f.d < 1)
		return 0;
	n = f.d;
	rm_init(&Q, n, n);
	xqk(R, &xq, 1, &f);
	rp_const(R, &row, r_const(R, 1));
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			RM(&Q, j, i) = j <= row.d ? row.c[j] : 0;   /* transposed */
		RM(&Q, i, i) = r_sub(R, RM(&Q, i, i), 0);
		rp_mulmod(R, &row, &row, &xq, &f);
	}
	for (i = 0; i < n; i++)
		RM(&Q, i, i) = r_sub(R, RM(&Q, i, i), r_const(R, 1));
	rank = rm_rref(R, &Q, NULL);
	rm_free(&Q);
	return n - rank;
}

/* ---- factoring over GF(q) ---- */

static u64 rng_state = 0x9e3779b97f4a7c15ULL;

static u64
rng(void)
{
	rng_state ^= rng_state << 13;
	rng_state ^= rng_state >> 7;
	rng_state ^= rng_state << 17;
	return rng_state;
}

static u64
r_random(const struct ring *R)
{
	if (R->qm1 == ~(u64)0)
		return rng();
	return rng() % (R->qm1 + 1);
}

/* the p-th root of an element: a^(q/p) = a^(p^(deg-1)) */
static u64
r_proot(const struct ring *R, u64 a)
{
	int i;

	for (i = 1; i < R->deg; i++)
		a = r_frob(R, a);
	return a;
}

static int nfac;
static struct rpoly *facv;
static int *multv, facmax;

static void
fac_push(const struct ring *R, const struct rpoly *g, int m)
{
	int i;

	for (i = 0; i < nfac; i++)
		if (rp_eq(&facv[i], g)) {
			multv[i] += m;
			return;
		}
	if (nfac < facmax) {
		facv[nfac] = *g;
		multv[nfac++] = m;
	}
}

/* Cantor-Zassenhaus: f is a product of distinct irreducibles of degree d */
static void
edf(const struct ring *R, const struct rpoly *f, int d, int m)
{
	struct rpoly a, b, g, one, t, q;
	int tries, i;

	if (f->d <= d) {
		if (f->d > 0)
			fac_push(R, f, m);
		return;
	}
	rp_const(R, &one, r_const(R, 1));
	for (tries = 0; tries < 400; tries++) {
		a.d = f->d - 1;
		for (i = 0; i <= a.d; i++)
			a.c[i] = r_random(R);
		rp_norm(&a);
		if (a.d < 1)
			continue;
		if (R->n == 2) {
			/* the trace a + a^2 + ... + a^(2^(k d - 1)), q = 2^k */
			int k = R->deg * d;
			t = a;
			b = a;
			for (i = 1; i < k; i++) {
				rp_mulmod(R, &t, &t, &t, f);
				rp_add(R, &b, &b, &t);
			}
		} else {
			/* a^((q^d - 1)/2) = (a^(1 + q + ... + q^(d-1)))^((q-1)/2) */
			t = a;
			b = a;
			for (i = 1; i < d; i++) {
				rp_frobmod(R, &t, &t, f);
				rp_mulmod(R, &b, &b, &t, f);
			}
			rp_powmod(R, &b, &b, R->qm1 / 2, f);
			rp_sub(R, &b, &b, &one);
		}
		rp_gcd(R, &g, &b, f);
		if (g.d > 0 && g.d < f->d) {
			rp_divmod(R, f, &g, &q, NULL);
			edf(R, &g, d, m);
			edf(R, &q, d, m);
			return;
		}
	}
	fac_push(R, f, m);   /* not split: reported as is */
}

/* distinct-degree split of a square-free monic f */
static void
ddf(const struct ring *R, const struct rpoly *f0, int m)
{
	struct rpoly f = *f0, h, x, t, g, q;
	int i;

	rp_x(&x);
	rp_divmod(R, &x, &f, NULL, &h);
	for (i = 1; 2 * i <= f.d; i++) {
		rp_frobmod(R, &h, &h, &f);
		rp_sub(R, &t, &h, &x);
		rp_gcd(R, &g, &t, &f);
		if (g.d > 0) {
			edf(R, &g, i, m);
			rp_divmod(R, &f, &g, &q, NULL);
			f = q;
			rp_divmod(R, &h, &f, NULL, &h);
		}
	}
	if (f.d > 0)
		fac_push(R, &f, m);
}

/* square-free factorisation over GF(q), then ddf/edf of each part */
static void
sff(const struct ring *R, const struct rpoly *f0, int mult)
{
	struct rpoly f = *f0, c, w, y, z, fd, q;
	int i = 1, j;

	rp_deriv(R, &fd, &f);
	if (fd.d < 0) {
		/* f = g(x^p): take the p-th root of the coefficients */
		struct rpoly g;
		u64 p = R->n;
		g.d = f.d / (int)p;
		for (j = 0; j <= g.d; j++)
			g.c[j] = r_proot(R, f.c[j * p]);
		rp_norm(&g);
		sff(R, &g, mult * (int)p);
		return;
	}
	rp_gcd(R, &c, &f, &fd);
	rp_divmod(R, &f, &c, &w, NULL);
	while (w.d > 0) {
		rp_gcd(R, &y, &w, &c);
		rp_divmod(R, &w, &y, &z, NULL);
		if (z.d > 0)
			ddf(R, &z, i * mult);
		i++;
		w = y;
		rp_divmod(R, &c, &y, &q, NULL);
		c = q;
	}
	if (c.d > 0) {
		/* c = g(x^p) */
		struct rpoly g;
		u64 p = R->n;
		g.d = c.d / (int)p;
		for (j = 0; j <= g.d; j++)
			g.c[j] = r_proot(R, c.c[j * p]);
		rp_norm(&g);
		sff(R, &g, mult * (int)p);
	}
}

int
rp_factor(const struct ring *R, const struct rpoly *f0, struct rpoly *fac,
    int *mult, int max, u64 *lead)
{
	struct rpoly f = *f0;
	int i, j;

	nfac = 0;
	facv = fac;
	multv = mult;
	facmax = max;
	rng_state = 0x9e3779b97f4a7c15ULL;
	*lead = f.d >= 0 ? f.c[f.d] : 0;
	if (f.d < 1 || !rp_monic(R, &f))
		return 0;
	sff(R, &f, 1);
	/* by degree, then lexicographically by coefficients from the top */
	for (i = 1; i < nfac; i++)
		for (j = i; j > 0; j--) {
			struct rpoly *a = &fac[j - 1], *b = &fac[j];
			int k, sw = a->d > b->d;
			if (a->d == b->d)
				for (k = a->d; k >= 0; k--)
					if (a->c[k] != b->c[k]) {
						sw = a->c[k] > b->c[k];
						break;
					}
			if (!sw)
				break;
			{
				struct rpoly t = *a;
				int tm = mult[j - 1];
				*a = *b;
				*b = t;
				mult[j - 1] = mult[j];
				mult[j] = tm;
			}
		}
	return nfac;
}

/*
 * The order of x mod f: for irreducible g of degree d it divides
 * q^d - 1; for g^e it is ord(g) p^t with p^t >= e the least such power;
 * for a product the lcm.
 */
u64
rp_xorder(const struct ring *R, const struct rpoly *f, int *overflow)
{
	static struct rpoly fac[RMAXDEG + 1];
	int mult[RMAXDEG + 1], k, i, j;
	u64 lead, res = 1;

	*overflow = 0;
	if (f->d < 1 || f->c[0] == 0)
		return 0;
	k = rp_factor(R, f, fac, mult, RMAXDEG + 1, &lead);
	for (i = 0; i < k; i++) {
		struct nt_fac nf;
		struct rpoly x, h, one;
		u128 qd = 1;
		u64 o, pt = 1;
		for (j = 0; j < fac[i].d; j++) {
			qd *= R->qm1 + (u128)1;
			if (qd > ((u128)1 << 64)) {
				*overflow = 1;
				return 0;
			}
		}
		o = (u64)(qd - 1);
		rp_x(&x);
		rp_const(R, &one, r_const(R, 1));
		nt_factor(o, &nf);
		for (j = 0; j < nf.k; j++)
			while (o % nf.p[j] == 0) {
				rp_powmod(R, &h, &x, o / nf.p[j], &fac[i]);
				if (!rp_eq(&h, &one))
					break;
				o /= nf.p[j];
			}
		while (pt < (u64)mult[i])
			pt *= R->n;
		if ((u128)o * pt > ~(u64)0) {
			*overflow = 1;
			return 0;
		}
		o *= pt;
		{
			u128 l = (u128)(res / nt_gcd(res, o)) * o;
			if (l >> 64) {
				*overflow = 1;
				return 0;
			}
			res = (u64)l;
		}
	}
	return res;
}
