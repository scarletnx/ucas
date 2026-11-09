/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ec.c - the shared elliptic-curve core (ec.h): fields behind an
 * operation table, curves in general Weierstrass form with their
 * invariants, the group law, the isomorphisms [u, r, s, t], other models
 * with their maps both ways, plane cubics with a flex to Weierstrass
 * form, and the integer helpers of ecq.h.
 */
#include "ecq.h"
#include "ucas.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/* integer helpers                                                     */
/* ================================================================== */

zint
zx_isqrt(zint a)
{
	zmark m;
	zint x, y;
	size_t b;

	if (z_sign(a) < 0)
		fail("isqrt of a negative number");
	if (z_is_zero(a))
		return 0;
	if (z_fits_i64(a)) {
		int64_t v = z_to_i64(a);
		int64_t r = (int64_t)sqrt((double)v);
		while (r > 0 && (i128)r * r > v)
			r--;
		while ((i128)(r + 1) * (r + 1) <= v)
			r++;
		return z_from_i64(r);
	}
	m = z_mark();
	b = z_bits(a);
	x = z_pow(z_from_i64(2), (uint64_t)((b + 1) / 2 + 1));
	for (;;) {
		y = z_div_floor(z_add(x, z_div_floor(a, x)), z_from_i64(2));
		if (z_cmp(y, x) >= 0)
			break;
		x = y;
	}
	z_release_keep(m, &x, 1);
	return x;
}

int
zx_is_square(zint a, zint *root)
{
	zint r;

	if (z_sign(a) < 0)
		return 0;
	/* quick residue filter mod 64 and 63 */
	{
		static const char sq64[64] = {1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,
		    1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,
		    0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0};
		if (!sq64[z_mod_u64(a, 64)])
			return 0;
	}
	r = zx_isqrt(a);
	if (!z_eq(z_mul(r, r), a))
		return 0;
	if (root)
		*root = r;
	return 1;
}

zint
zx_iroot(zint a, int k, int *exact)
{
	zmark m;
	zint x, y, kk, km1;
	int neg = 0;
	size_t b;

	if (k < 1)
		fail("iroot: k < 1");
	if (z_sign(a) < 0) {
		if (!(k & 1))
			fail("an even root of a negative number");
		neg = 1;
		a = z_neg(a);
	}
	if (k == 1 || z_is_zero(a) || z_is_one(a)) {
		if (exact)
			*exact = 1;
		return neg ? z_neg(a) : a;
	}
	m = z_mark();
	b = z_bits(a);
	kk = z_from_i64(k);
	km1 = z_from_i64(k - 1);
	x = z_pow(z_from_i64(2), (uint64_t)(b / (size_t)k + 1));
	for (;;) {
		/* y = ((k-1) x + a / x^(k-1)) / k */
		y = z_div_floor(z_add(z_mul(km1, x),
		    z_div_floor(a, z_pow(x, (uint64_t)(k - 1)))), kk);
		if (z_cmp(y, x) >= 0)
			break;
		x = y;
	}
	if (exact)
		*exact = z_eq(z_pow(x, (uint64_t)k), a);
	if (neg) {
		/* floor of a negative real root: -ceil(root of |a|) */
		if (!z_eq(z_pow(x, (uint64_t)k), a))
			x = z_add(x, Z_ONE);
		x = z_neg(x);
	}
	z_release_keep(m, &x, 1);
	return x;
}

int
zq_is_square(struct zq a, struct zq *root)
{
	zint rn, rd;

	if (z_sign(a.n) < 0)
		return 0;
	if (!zx_is_square(a.n, &rn) || !zx_is_square(a.d, &rd))
		return 0;
	if (root)
		*root = zq_make(rn, rd);
	return 1;
}

zint
zx_squarefree_part(zint a, zint *sq)
{
	zint p[64], part, s;
	int e[64], k, i, complete;

	if (z_is_zero(a))
		fail("squarefree part of 0");
	k = z_factor(a, p, e, 64, &complete);
	if (k < 0 || !complete)
		fail("squarefree part: %s could not be factored", z_str(a));
	part = z_from_i64(z_sign(a));
	s = Z_ONE;
	for (i = 0; i < k; i++) {
		if (e[i] & 1)
			part = z_mul(part, p[i]);
		s = z_mul(s, z_pow(p[i], (uint64_t)(e[i] / 2)));
	}
	if (sq)
		*sq = s;
	return part;
}

static zint
cub_val(zint a, zint b, zint c, zint x)
{
	/* ((x + a) x + b) x + c */
	return z_add(z_mul(z_add(z_mul(z_add(x, a), x), b), x), c);
}

/* an integer root of the monotone f on [lo, hi] (inclusive), if any */
static int
cub_bisect(zint a, zint b, zint c, zint lo, zint hi, zint *root)
{
	int slo, shi;

	if (z_cmp(lo, hi) > 0)
		return 0;
	slo = z_sign(cub_val(a, b, c, lo));
	shi = z_sign(cub_val(a, b, c, hi));
	if (slo == 0) {
		*root = lo;
		return 1;
	}
	if (shi == 0) {
		*root = hi;
		return 1;
	}
	if (slo == shi)
		return 0;
	while (z_cmp(z_sub(hi, lo), Z_ONE) > 0) {
		zint mid = z_div_floor(z_add(lo, hi), z_from_i64(2));
		int sm = z_sign(cub_val(a, b, c, mid));
		if (sm == 0) {
			*root = mid;
			return 1;
		}
		if (sm == slo)
			lo = mid;
		else
			hi = mid;
	}
	return 0;
}

int
zx_cubic_int_roots(zint a, zint b, zint c, zint *roots)
{
	zint bound, cp[4], d, s, r;
	int n = 0, ncp = 0, i, k;

	/* Cauchy: |root| <= 1 + max |coefficient| */
	bound = z_abs(a);
	if (z_cmpabs(b, bound) > 0)
		bound = z_abs(b);
	if (z_cmpabs(c, bound) > 0)
		bound = z_abs(c);
	bound = z_add(bound, Z_ONE);
	cp[ncp++] = z_neg(bound);
	/* f' = 3x^2 + 2a x + b: roots (-a +- sqrt(a^2 - 3b)) / 3 */
	d = z_sub(z_mul(a, a), z_mul(z_from_i64(3), b));
	if (z_sign(d) > 0) {
		s = zx_isqrt(d);
		/* floor of (-a - sqrt d)/3 and floor of (-a + sqrt d)/3; the
		 * monotone pieces are split at these integers (and +1) */
		cp[ncp++] = z_div_floor(z_sub(z_neg(a), z_add(s, Z_ONE)),
		    z_from_i64(3));
		cp[ncp++] = z_div_floor(z_add(z_neg(a), s), z_from_i64(3));
	}
	cp[ncp++] = bound;
	for (i = 0; i + 1 < ncp; i++) {
		zint lo = i == 0 ? cp[i] : z_add(cp[i], Z_ONE);
		zint hi = cp[i + 1];
		/* the integer at a piece's border may hold a root */
		if (i > 0 && z_is_zero(cub_val(a, b, c, cp[i]))) {
			for (k = 0; k < n && !z_eq(roots[k], cp[i]); k++)
				;
			if (k == n)
				roots[n++] = cp[i];
		}
		if (cub_bisect(a, b, c, lo, hi, &r)) {
			for (k = 0; k < n && !z_eq(roots[k], r); k++)
				;
			if (k == n)
				roots[n++] = r;
		}
	}
	/* sort */
	for (i = 1; i < n; i++)
		for (k = i; k > 0 && z_cmp(roots[k - 1], roots[k]) > 0; k--) {
			r = roots[k];
			roots[k] = roots[k - 1];
			roots[k - 1] = r;
		}
	return n;
}

int
zq_cubic_roots(struct zq a, struct zq b, struct zq c, struct zq d,
    struct zq *roots)
{
	/* monic: x^3 + B x^2 + C x + D; with L = lcm of denominators,
	 * X = L x gives X^3 + B L X^2 + C L^2 X + D L^3, integral */
	struct zq B, C, D;
	zint L, ib, ic, id, r[3];
	int n, i;

	if (zq_is_zero(a))
		fail("cubic roots: leading coefficient 0");
	B = zq_div(b, a);
	C = zq_div(c, a);
	D = zq_div(d, a);
	L = z_lcm(B.d, z_lcm(C.d, D.d));
	ib = z_divexact(z_mul(B.n, L), B.d);
	ic = z_divexact(z_mul(C.n, z_mul(L, L)), C.d);
	id = z_divexact(z_mul(D.n, z_pow(L, 3)), D.d);
	n = zx_cubic_int_roots(ib, ic, id, r);
	for (i = 0; i < n; i++)
		roots[i] = zq_make(r[i], L);
	return n;
}

double
zx_to_double(zint a)
{
	size_t b;

	if (z_fits_i64(a))
		return (double)z_to_i64(a);
	b = z_bits(a);
	if (b > 1000) {
		/* scale down: ldexp of the top 60 bits */
		zint t = z_div_floor(z_abs(a), z_pow(z_from_i64(2),
		    (uint64_t)(b - 60)));
		double v = ldexp((double)z_to_i64(t), (int)(b - 60));
		return z_sign(a) < 0 ? -v : v;
	} else {
		zint t = z_div_floor(z_abs(a), z_pow(z_from_i64(2),
		    (uint64_t)(b - 60)));
		double v = ldexp((double)z_to_i64(t), (int)(b - 60));
		return z_sign(a) < 0 ? -v : v;
	}
}

double
zq_to_double(struct zq a)
{
	size_t bn = z_bits(a.n), bd = z_bits(a.d);

	if (bn < 1000 && bd < 1000)
		return zx_to_double(a.n) / zx_to_double(a.d);
	{
		/* both scaled to about 60 bits */
		double ln = zx_log(a.n) - zx_log(a.d);
		return (z_sign(a.n) < 0 ? -1 : 1) * exp(ln);
	}
}

double
zx_log(zint a)
{
	size_t b = z_bits(a);
	zint t;

	if (b <= 60)
		return log(fabs((double)z_to_i64(a)));
	t = z_div_floor(z_abs(a), z_pow(z_from_i64(2), (uint64_t)(b - 60)));
	return log((double)z_to_i64(t)) + (double)(b - 60) * log(2.0);
}

/* ================================================================== */
/* fields                                                              */
/* ================================================================== */

#define ZW(a, i) ((zint)(a)->w[i])

static void
fe_clear(fe *r)
{
	memset(r, 0, sizeof *r);
}

/* ---- Q ---- */

struct zq
fe_q(const fe *a)
{
	struct zq q;

	q.n = ZW(a, 0);
	q.d = ZW(a, 1);
	return q;
}

fe
fe_from_q(struct zq a)
{
	fe r;

	fe_clear(&r);
	r.w[0] = (uint64_t)a.n;
	r.w[1] = (uint64_t)a.d;
	return r;
}

static void
q_from_i64(const struct ecf *F, fe *r, int64_t v)
{
	*r = fe_from_q(zq_int(z_from_i64(v)));
}

static int
q_from_zq(const struct ecf *F, fe *r, struct zq a)
{
	*r = fe_from_q(a);
	return 1;
}

static void
q_add(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	*r = fe_from_q(zq_add(fe_q(a), fe_q(b)));
}

static void
q_sub(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	*r = fe_from_q(zq_sub(fe_q(a), fe_q(b)));
}

static void
q_neg(const struct ecf *F, fe *r, const fe *a)
{
	*r = fe_from_q(zq_neg(fe_q(a)));
}

static void
q_mul(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	*r = fe_from_q(zq_mul(fe_q(a), fe_q(b)));
}

static int
q_inv(const struct ecf *F, fe *r, const fe *a)
{
	if (z_is_zero(ZW(a, 0)))
		return 0;
	*r = fe_from_q(zq_inv(fe_q(a)));
	return 1;
}

static int
q_is_zero(const struct ecf *F, const fe *a)
{
	return z_is_zero(ZW(a, 0));
}

static int
q_eq(const struct ecf *F, const fe *a, const fe *b)
{
	return zq_eq(fe_q(a), fe_q(b));
}

static int
q_sqrt(const struct ecf *F, fe *r, const fe *a)
{
	struct zq s;

	if (!zq_is_square(fe_q(a), &s))
		return 0;
	*r = fe_from_q(s);
	return 1;
}

static int
q_to_zq(const struct ecf *F, struct zq *r, const fe *a)
{
	*r = fe_q(a);
	return 1;
}

static void
q_print(const struct ecf *F, struct sbuf *b, const fe *a)
{
	zq_print(b, fe_q(a));
}

static int
q_parse(const struct ecf *F, fe *r, const char *s, const char **end)
{
	struct zq q;

	while (*s == ' ')
		s++;
	if (!zq_parse(s, &q, end))
		return 0;
	*r = fe_from_q(q);
	return 1;
}

static void
keep_words(zmark m, fe *a, size_t n, int words)
{
	static zint *tmp;
	static size_t cap;
	size_t i, k = 0;
	int w;

	if (n * (size_t)words > cap) {
		cap = n * (size_t)words + 16;
		tmp = xrealloc(tmp, cap * sizeof *tmp);
	}
	for (i = 0; i < n; i++)
		for (w = 0; w < words; w++)
			tmp[k++] = ZW(&a[i], w);
	z_release_keep(m, tmp, k);
	k = 0;
	for (i = 0; i < n; i++)
		for (w = 0; w < words; w++)
			a[i].w[w] = (uint64_t)tmp[k++];
}

static void
q_keep(const struct ecf *F, zmark m, fe *a, size_t n)
{
	keep_words(m, a, n, 2);
}

static const struct ecf_ops q_ops = {
	"Q", q_from_i64, q_from_zq, q_add, q_sub, q_neg, q_mul, q_inv,
	q_is_zero, q_eq, q_sqrt, q_to_zq, q_print, q_parse, q_keep
};

void
ecf_q(struct ecf *F)
{
	memset(F, 0, sizeof *F);
	F->kind = ECF_Q;
	F->deg = 1;
	F->op = &q_ops;
}

/* ---- F_p and Z/n with zint residues ---- */

static void
zp_set(fe *r, zint v)
{
	fe_clear(r);
	r->w[0] = (uint64_t)v;
}

static void
zp_from_i64(const struct ecf *F, fe *r, int64_t v)
{
	zp_set(r, z_mod(z_from_i64(v), F->p));
}

static int zp_inv(const struct ecf *F, fe *r, const fe *a);

static int
zp_from_zq(const struct ecf *F, fe *r, struct zq a)
{
	fe d;

	zp_set(&d, z_mod(a.d, F->p));
	if (!zp_inv(F, &d, &d))
		return 0;
	zp_set(r, z_mod(z_mul(z_mod(a.n, F->p), ZW(&d, 0)), F->p));
	return 1;
}

static void
zp_add(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	zint s = z_add(ZW(a, 0), ZW(b, 0));

	if (z_cmp(s, F->p) >= 0)
		s = z_sub(s, F->p);
	zp_set(r, s);
}

static void
zp_sub(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	zint s = z_sub(ZW(a, 0), ZW(b, 0));

	if (z_sign(s) < 0)
		s = z_add(s, F->p);
	zp_set(r, s);
}

static void
zp_neg(const struct ecf *F, fe *r, const fe *a)
{
	zp_set(r, z_is_zero(ZW(a, 0)) ? 0 : z_sub(F->p, ZW(a, 0)));
}

static void
zp_mul(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	zp_set(r, z_mod(z_mul(ZW(a, 0), ZW(b, 0)), F->p));
}

static int
zp_inv(const struct ecf *F, fe *r, const fe *a)
{
	zint g, s, t;

	if (z_is_zero(ZW(a, 0)))
		return 0;
	g = z_xgcd(ZW(a, 0), F->p, &s, &t);
	if (!z_is_one(g)) {
		if (F->kind == ECF_ZN && !z_eq(g, F->p))
			((struct ecf *)F)->factor = g;
		return 0;
	}
	zp_set(r, z_mod(s, F->p));
	return 1;
}

static int
zp_is_zero(const struct ecf *F, const fe *a)
{
	return z_is_zero(ZW(a, 0));
}

static int
zp_eq(const struct ecf *F, const fe *a, const fe *b)
{
	return z_eq(ZW(a, 0), ZW(b, 0));
}

/* Tonelli-Shanks over zint */
static int
zp_sqrt(const struct ecf *F, fe *r, const fe *a)
{
	zint p = F->p, x = ZW(a, 0), q, z, c, t, R, b, e1;
	int64_t s, M, i;
	zmark m;

	if (F->kind != ECF_FP)
		return 0;
	if (z_is_zero(x)) {
		zp_set(r, 0);
		return 1;
	}
	if (z_eq(p, z_from_i64(2))) {
		zp_set(r, x);
		return 1;
	}
	m = z_mark();
	e1 = z_div_floor(z_sub(p, Z_ONE), z_from_i64(2));
	if (!z_is_one(z_powmod(x, e1, p))) {
		z_release(m);
		return 0;
	}
	if (z_mod_u64(p, 4) == 3) {
		R = z_powmod(x, z_div_floor(z_add(p, Z_ONE), z_from_i64(4)), p);
		z_release_keep(m, &R, 1);
		zp_set(r, R);
		return 1;
	}
	q = z_sub(p, Z_ONE);
	s = 0;
	while (z_mod_u64(q, 2) == 0) {
		q = z_div_floor(q, z_from_i64(2));
		s++;
	}
	z = z_from_i64(2);
	while (z_is_one(z_powmod(z, e1, p)))
		z = z_add(z, Z_ONE);
	M = s;
	c = z_powmod(z, q, p);
	t = z_powmod(x, q, p);
	R = z_powmod(x, z_div_floor(z_add(q, Z_ONE), z_from_i64(2)), p);
	while (!z_is_one(t)) {
		zint tt = t;
		for (i = 0; i < M && !z_is_one(tt); i++)
			tt = z_mod(z_mul(tt, tt), p);
		if (i == M) {
			z_release(m);
			return 0;
		}
		b = c;
		{
			int64_t k;
			for (k = 0; k < M - i - 1; k++)
				b = z_mod(z_mul(b, b), p);
		}
		M = i;
		c = z_mod(z_mul(b, b), p);
		t = z_mod(z_mul(t, c), p);
		R = z_mod(z_mul(R, b), p);
	}
	z_release_keep(m, &R, 1);
	zp_set(r, R);
	return 1;
}

static int
zp_to_zq(const struct ecf *F, struct zq *r, const fe *a)
{
	*r = zq_int(ZW(a, 0));
	return 1;
}

static void
zp_print(const struct ecf *F, struct sbuf *b, const fe *a)
{
	z_print(b, ZW(a, 0));
}

static int
zp_parse(const struct ecf *F, fe *r, const char *s, const char **end)
{
	struct zq q;

	while (*s == ' ')
		s++;
	if (!zq_parse(s, &q, end))
		return 0;
	if (!zp_from_zq(F, r, q))
		fail("%s is not invertible in %s", z_str(q.d), ecf_name(F));
	return 1;
}

static void
zp_keep(const struct ecf *F, zmark m, fe *a, size_t n)
{
	keep_words(m, a, n, 1);
}

static const struct ecf_ops fp_ops = {
	"F_p", zp_from_i64, zp_from_zq, zp_add, zp_sub, zp_neg, zp_mul, zp_inv,
	zp_is_zero, zp_eq, zp_sqrt, zp_to_zq, zp_print, zp_parse, zp_keep
};

void
ecf_fp(struct ecf *F, zint p)
{
	if (z_cmp(p, z_from_i64(2)) < 0 || !z_isprime(p))
		fail("F_p: %s is not a prime", z_str(p));
	memset(F, 0, sizeof *F);
	F->kind = ECF_FP;
	F->p = p;
	F->deg = 1;
	F->op = &fp_ops;
}

void
ecf_zn(struct ecf *F, zint n)
{
	if (z_cmp(n, z_from_i64(2)) < 0)
		fail("Z/n: n = %s < 2", z_str(n));
	memset(F, 0, sizeof *F);
	F->kind = ECF_ZN;
	F->p = n;
	F->deg = 1;
	F->op = &fp_ops;
}

/* ---- GF(p^k) through ring.h ---- */

static void
fq_set(fe *r, u64 v)
{
	fe_clear(r);
	r->w[0] = v;
}

static void
fq_from_i64(const struct ecf *F, fe *r, int64_t v)
{
	u64 n = F->R->n;
	int64_t m = v % (int64_t)n;

	if (m < 0)
		m += (int64_t)n;
	fq_set(r, r_const(F->R, (u64)m));
}

static int
fq_from_zq(const struct ecf *F, fe *r, struct zq a)
{
	u64 n = F->R->n, num = z_mod_u64(a.n, n), den = z_mod_u64(a.d, n), di;

	if (!r_inv(F->R, r_const(F->R, den), &di))
		return 0;
	fq_set(r, r_mul(F->R, r_const(F->R, num), di));
	return 1;
}

static void
fq_add(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	fq_set(r, r_add(F->R, a->w[0], b->w[0]));
}

static void
fq_sub(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	fq_set(r, r_sub(F->R, a->w[0], b->w[0]));
}

static void
fq_neg(const struct ecf *F, fe *r, const fe *a)
{
	fq_set(r, r_neg(F->R, a->w[0]));
}

static void
fq_mul(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	fq_set(r, r_mul(F->R, a->w[0], b->w[0]));
}

static int
fq_inv(const struct ecf *F, fe *r, const fe *a)
{
	u64 v;

	if (!r_inv(F->R, a->w[0], &v))
		return 0;
	fq_set(r, v);
	return 1;
}

static int
fq_is_zero(const struct ecf *F, const fe *a)
{
	return a->w[0] == 0;
}

static int
fq_eq(const struct ecf *F, const fe *a, const fe *b)
{
	return a->w[0] == b->w[0];
}

static int
fq_sqrt(const struct ecf *F, fe *r, const fe *a)
{
	const struct ring *R = F->R;
	u64 x = a->w[0], q1 = R->qm1, Q, z, c, t, Rr, b;
	int s, M, i, k;

	if (x == 0) {
		fq_set(r, 0);
		return 1;
	}
	if (R->n == 2) {
		/* a^(q/2), q = qm1 + 1 */
		fq_set(r, r_pow(R, x, q1 / 2 + 1));
		return 1;
	}
	if (r_pow(R, x, q1 / 2) != r_const(R, 1))
		return 0;
	Q = q1;
	s = 0;
	while (!(Q & 1)) {
		Q >>= 1;
		s++;
	}
	for (z = 2; z <= q1; z++)
		if (r_pow(R, z, q1 / 2) != r_const(R, 1) && z != 0)
			break;
	M = s;
	c = r_pow(R, z, Q);
	t = r_pow(R, x, Q);
	Rr = r_pow(R, x, (Q + 1) / 2);
	while (t != r_const(R, 1)) {
		u64 tt = t;
		for (i = 0; i < M && tt != r_const(R, 1); i++)
			tt = r_mul(R, tt, tt);
		if (i == M)
			return 0;
		b = c;
		for (k = 0; k < M - i - 1; k++)
			b = r_mul(R, b, b);
		M = i;
		c = r_mul(R, b, b);
		t = r_mul(R, t, c);
		Rr = r_mul(R, Rr, b);
	}
	fq_set(r, Rr);
	return 1;
}

static int
fq_to_zq(const struct ecf *F, struct zq *r, const fe *a)
{
	if (F->R->deg > 1 && a->w[0] >= F->R->n)
		return 0;
	*r = zq_int(z_from_u64(a->w[0]));
	return 1;
}

static void
fq_print(const struct ecf *F, struct sbuf *b, const fe *a)
{
	if (F->R->deg > 1)
		sb_printf(b, "0x%llx", (unsigned long long)a->w[0]);
	else
		sb_printf(b, "%llu", (unsigned long long)a->w[0]);
}

static int
fq_parse(const struct ecf *F, fe *r, const char *s, const char **end)
{
	char *e;
	unsigned long long v;
	int neg = 0;

	while (*s == ' ')
		s++;
	if (*s == '-') {
		neg = 1;
		s++;
	}
	if (!isdigit((unsigned char)*s))
		return 0;
	v = strtoull(s, &e, 0);
	if (end)
		*end = e;
	if (F->R->deg == 1)
		v %= F->R->n;
	fq_set(r, (u64)v);
	if (neg)
		fq_neg(F, r, r);
	return 1;
}

static const struct ecf_ops fq_ops = {
	"GF", fq_from_i64, fq_from_zq, fq_add, fq_sub, fq_neg, fq_mul, fq_inv,
	fq_is_zero, fq_eq, fq_sqrt, fq_to_zq, fq_print, fq_parse, NULL
};

void
ecf_fq(struct ecf *F, struct ring *R)
{
	if (!R->field)
		fail("GF: the ring is not a field");
	memset(F, 0, sizeof *F);
	F->kind = ECF_FQ;
	F->p = z_from_u64(R->n);
	F->deg = R->deg;
	F->R = R;
	F->op = &fq_ops;
}

const char *
ecf_name(const struct ecf *F)
{
	static char buf[4][96];
	static int k;
	char *b = buf[k++ & 3];

	switch (F->kind) {
	case ECF_Q:
		return "Q";
	case ECF_FP:
		snprintf(b, 96, "F_%s", z_str(F->p));
		return b;
	case ECF_ZN:
		snprintf(b, 96, "Z/%s", z_str(F->p));
		return b;
	case ECF_FQ:
		if (F->deg == 1)
			snprintf(b, 96, "F_%llu", (unsigned long long)F->R->n);
		else
			snprintf(b, 96, "GF(%llu^%d)",
			    (unsigned long long)F->R->n, F->deg);
		return b;
	}
	return F->op && F->op->name ? F->op->name : "?";
}

/* ---- element helpers ---- */

void
fe_zero(const struct ecf *F, fe *r)
{
	F->op->from_i64(F, r, 0);
}

void
fe_one(const struct ecf *F, fe *r)
{
	F->op->from_i64(F, r, 1);
}

void
fe_set_i64(const struct ecf *F, fe *r, int64_t v)
{
	F->op->from_i64(F, r, v);
}

int
fe_set_zq(const struct ecf *F, fe *r, struct zq a)
{
	return F->op->from_zq(F, r, a);
}

int
fe_set_z(const struct ecf *F, fe *r, zint a)
{
	return F->op->from_zq(F, r, zq_int(a));
}

void
fe_add(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	F->op->add(F, r, a, b);
}

void
fe_sub(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	F->op->sub(F, r, a, b);
}

void
fe_neg(const struct ecf *F, fe *r, const fe *a)
{
	F->op->neg(F, r, a);
}

void
fe_mul(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	F->op->mul(F, r, a, b);
}

void
fe_sqr(const struct ecf *F, fe *r, const fe *a)
{
	F->op->mul(F, r, a, a);
}

void
fe_mul_i64(const struct ecf *F, fe *r, const fe *a, int64_t k)
{
	fe t;

	F->op->from_i64(F, &t, k);
	F->op->mul(F, r, a, &t);
}

int
fe_inv(const struct ecf *F, fe *r, const fe *a)
{
	return F->op->inv(F, r, a);
}

int
fe_div(const struct ecf *F, fe *r, const fe *a, const fe *b)
{
	fe t;

	if (!F->op->inv(F, &t, b))
		return 0;
	F->op->mul(F, r, a, &t);
	return 1;
}

void
fe_pow(const struct ecf *F, fe *r, const fe *a, uint64_t e)
{
	fe acc, b = *a;

	fe_one(F, &acc);
	while (e) {
		if (e & 1)
			fe_mul(F, &acc, &acc, &b);
		e >>= 1;
		if (e)
			fe_mul(F, &b, &b, &b);
	}
	*r = acc;
}

int
fe_is_zero(const struct ecf *F, const fe *a)
{
	return F->op->is_zero(F, a);
}

int
fe_is_one(const struct ecf *F, const fe *a)
{
	fe one;

	fe_one(F, &one);
	return F->op->eq(F, a, &one);
}

int
fe_eq(const struct ecf *F, const fe *a, const fe *b)
{
	return F->op->eq(F, a, b);
}

int
fe_sqrt(const struct ecf *F, fe *r, const fe *a)
{
	return F->op->sqrt ? F->op->sqrt(F, r, a) : 0;
}

int
fe_to_zq(const struct ecf *F, struct zq *r, const fe *a)
{
	return F->op->to_zq ? F->op->to_zq(F, r, a) : 0;
}

void
fe_print(const struct ecf *F, struct sbuf *b, const fe *a)
{
	F->op->print(F, b, a);
}

const char *
fe_str(const struct ecf *F, const fe *a)
{
	static struct sbuf b[8];
	static int k;
	struct sbuf *s = &b[k++ & 7];

	sb_clear(s);
	fe_print(F, s, a);
	return sb_str(s);
}

int
fe_parse(const struct ecf *F, fe *r, const char *s, const char **end)
{
	return F->op->parse(F, r, s, end);
}

void
fe_keep(const struct ecf *F, zmark m, fe *a, size_t n)
{
	if (F->op->keep)
		F->op->keep(F, m, a, n);
}

/* ================================================================== */
/* curves                                                              */
/* ================================================================== */

static void
ec_invariants(struct ec *E)
{
	const struct ecf *F = E->F;
	fe t, u, b2sq;

	/* b2 = a1^2 + 4 a2 */
	fe_mul(F, &t, &E->a1, &E->a1);
	fe_mul_i64(F, &u, &E->a2, 4);
	fe_add(F, &E->b2, &t, &u);
	/* b4 = 2 a4 + a1 a3 */
	fe_mul(F, &t, &E->a1, &E->a3);
	fe_mul_i64(F, &u, &E->a4, 2);
	fe_add(F, &E->b4, &t, &u);
	/* b6 = a3^2 + 4 a6 */
	fe_mul(F, &t, &E->a3, &E->a3);
	fe_mul_i64(F, &u, &E->a6, 4);
	fe_add(F, &E->b6, &t, &u);
	/* b8 = a1^2 a6 + 4 a2 a6 - a1 a3 a4 + a2 a3^2 - a4^2 */
	{
		fe s;
		fe_mul(F, &t, &E->a1, &E->a1);
		fe_mul(F, &s, &t, &E->a6);
		fe_mul(F, &t, &E->a2, &E->a6);
		fe_mul_i64(F, &t, &t, 4);
		fe_add(F, &s, &s, &t);
		fe_mul(F, &t, &E->a1, &E->a3);
		fe_mul(F, &t, &t, &E->a4);
		fe_sub(F, &s, &s, &t);
		fe_mul(F, &t, &E->a3, &E->a3);
		fe_mul(F, &t, &t, &E->a2);
		fe_add(F, &s, &s, &t);
		fe_mul(F, &t, &E->a4, &E->a4);
		fe_sub(F, &E->b8, &s, &t);
	}
	/* c4 = b2^2 - 24 b4 */
	fe_mul(F, &b2sq, &E->b2, &E->b2);
	fe_mul_i64(F, &t, &E->b4, 24);
	fe_sub(F, &E->c4, &b2sq, &t);
	/* c6 = -b2^3 + 36 b2 b4 - 216 b6 */
	{
		fe s;
		fe_mul(F, &s, &b2sq, &E->b2);
		fe_neg(F, &s, &s);
		fe_mul(F, &t, &E->b2, &E->b4);
		fe_mul_i64(F, &t, &t, 36);
		fe_add(F, &s, &s, &t);
		fe_mul_i64(F, &t, &E->b6, 216);
		fe_sub(F, &E->c6, &s, &t);
	}
	/* disc = -b2^2 b8 - 8 b4^3 - 27 b6^2 + 9 b2 b4 b6 */
	{
		fe s;
		fe_mul(F, &s, &b2sq, &E->b8);
		fe_neg(F, &s, &s);
		fe_mul(F, &t, &E->b4, &E->b4);
		fe_mul(F, &t, &t, &E->b4);
		fe_mul_i64(F, &t, &t, 8);
		fe_sub(F, &s, &s, &t);
		fe_mul(F, &t, &E->b6, &E->b6);
		fe_mul_i64(F, &t, &t, 27);
		fe_sub(F, &s, &s, &t);
		fe_mul(F, &t, &E->b2, &E->b4);
		fe_mul(F, &t, &t, &E->b6);
		fe_mul_i64(F, &t, &t, 9);
		fe_add(F, &E->disc, &s, &t);
	}
	E->singular = fe_is_zero(F, &E->disc);
	E->jok = 0;
	fe_zero(F, &E->j);
	if (!E->singular) {
		fe di;
		if (fe_inv(F, &di, &E->disc)) {
			fe_mul(F, &t, &E->c4, &E->c4);
			fe_mul(F, &t, &t, &E->c4);
			fe_mul(F, &E->j, &t, &di);
			E->jok = 1;
		}
	}
}

void
ec_init(struct ec *E, const struct ecf *F, const fe a[5])
{
	memset(E, 0, sizeof *E);
	E->F = F;
	E->a1 = a[0];
	E->a2 = a[1];
	E->a3 = a[2];
	E->a4 = a[3];
	E->a6 = a[4];
	ec_invariants(E);
}

void
ec_init_i64(struct ec *E, const struct ecf *F, const int64_t a[5])
{
	fe c[5];
	int i;

	for (i = 0; i < 5; i++)
		fe_set_i64(F, &c[i], a[i]);
	ec_init(E, F, c);
}

void
ec_init_short(struct ec *E, const struct ecf *F, const fe *a4, const fe *a6)
{
	fe c[5];

	fe_zero(F, &c[0]);
	c[1] = c[2] = c[0];
	c[3] = *a4;
	c[4] = *a6;
	ec_init(E, F, c);
}

void
ec_coeffs(const struct ec *E, fe a[5])
{
	a[0] = E->a1;
	a[1] = E->a2;
	a[2] = E->a3;
	a[3] = E->a4;
	a[4] = E->a6;
}

int
ec_reduce(struct ec *r, const struct ec *Eq, const struct ecf *F)
{
	fe a[5], c[5];
	int i;

	ec_coeffs(Eq, a);
	for (i = 0; i < 5; i++)
		if (!fe_set_zq(F, &c[i], fe_q(&a[i])))
			return 0;
	ec_init(r, F, c);
	return 1;
}

void
ec_copy(struct ec *dst, const struct ec *src)
{
	*dst = *src;
}

void
ec_keep(struct ec *E, zmark m)
{
	fe_keep(E->F, m, &E->a1, 1);
	fe_keep(E->F, m, &E->a2, 1);
	fe_keep(E->F, m, &E->a3, 1);
	fe_keep(E->F, m, &E->a4, 1);
	fe_keep(E->F, m, &E->a6, 1);
	/* the invariants: recompute rather than keep 10 more */
	ec_invariants(E);
}

void
ec_eval(const struct ec *E, fe *r, const fe *x, const fe *y)
{
	const struct ecf *F = E->F;
	fe l, rr, t;

	/* y^2 + a1 x y + a3 y */
	fe_mul(F, &t, &E->a1, x);
	fe_add(F, &t, &t, y);
	fe_add(F, &t, &t, &E->a3);
	fe_mul(F, &l, &t, y);
	/* ((x + a2) x + a4) x + a6 */
	fe_add(F, &rr, x, &E->a2);
	fe_mul(F, &rr, &rr, x);
	fe_add(F, &rr, &rr, &E->a4);
	fe_mul(F, &rr, &rr, x);
	fe_add(F, &rr, &rr, &E->a6);
	fe_sub(F, r, &l, &rr);
}

/* " + c*name" with the sign pulled out where the field has one (Q) */
static void
put_term(struct sbuf *b, const struct ecf *F, const fe *c, const char *name,
    int first)
{
	struct zq q;

	if (fe_is_zero(F, c))
		return;
	if (F->kind == ECF_Q && fe_to_zq(F, &q, c)) {
		int neg = z_sign(q.n) < 0;
		struct zq a = neg ? zq_neg(q) : q;
		if (first)
			sb_puts(b, neg ? "-" : "");
		else
			sb_puts(b, neg ? " - " : " + ");
		if (!*name) {
			zq_print(b, a);
			return;
		}
		if (!(z_is_one(a.n) && z_is_one(a.d))) {
			zq_print(b, a);
			sb_putc(b, '*');
		}
		sb_puts(b, name);
		return;
	}
	if (!first)
		sb_puts(b, " + ");
	if (!*name) {
		fe_print(F, b, c);
		return;
	}
	if (!fe_is_one(F, c)) {
		fe_print(F, b, c);
		sb_putc(b, '*');
	}
	sb_puts(b, name);
}

void
ec_print(struct sbuf *b, const struct ec *E)
{
	const struct ecf *F = E->F;
	fe one;

	fe_one(F, &one);
	sb_puts(b, "y^2");
	put_term(b, F, &E->a1, "x*y", 0);
	put_term(b, F, &E->a3, "y", 0);
	sb_puts(b, " = x^3");
	put_term(b, F, &E->a2, "x^2", 0);
	put_term(b, F, &E->a4, "x", 0);
	put_term(b, F, &E->a6, "", 0);
}

void
ec_print_ainv(struct sbuf *b, const struct ec *E)
{
	sb_printf(b, "[%s, %s, %s, %s, %s]", fe_str(E->F, &E->a1),
	    fe_str(E->F, &E->a2), fe_str(E->F, &E->a3), fe_str(E->F, &E->a4),
	    fe_str(E->F, &E->a6));
}

/* ================================================================== */
/* points                                                              */
/* ================================================================== */

void
ec_O(struct ecpt *P)
{
	memset(P, 0, sizeof *P);
	P->inf = 1;
}

void
ec_pt(const struct ec *E, struct ecpt *P, const fe *x, const fe *y)
{
	P->x = *x;
	P->y = *y;
	P->inf = 0;
}

int
ec_on(const struct ec *E, const struct ecpt *P)
{
	fe r;

	if (P->inf)
		return 1;
	ec_eval(E, &r, &P->x, &P->y);
	return fe_is_zero(E->F, &r);
}

int
ec_pt_eq(const struct ec *E, const struct ecpt *P, const struct ecpt *Q)
{
	if (P->inf || Q->inf)
		return P->inf && Q->inf;
	return fe_eq(E->F, &P->x, &Q->x) && fe_eq(E->F, &P->y, &Q->y);
}

void
ec_neg(const struct ec *E, struct ecpt *R, const struct ecpt *P)
{
	const struct ecf *F = E->F;
	fe t;

	if (P->inf) {
		ec_O(R);
		return;
	}
	/* -(x, y) = (x, -y - a1 x - a3) */
	fe_mul(F, &t, &E->a1, &P->x);
	fe_add(F, &t, &t, &E->a3);
	fe_add(F, &t, &t, &P->y);
	fe_neg(F, &R->y, &t);
	R->x = P->x;
	R->inf = 0;
}

/* the chord or tangent through P, Q with slope lam and intercept nu */
static int
ec_add_core(const struct ec *E, struct ecpt *R, const struct ecpt *P,
    const struct ecpt *Q)
{
	const struct ecf *F = E->F;
	fe lam, nu, num, den, t, x3, y3;

	if (P->inf) {
		*R = *Q;
		return 1;
	}
	if (Q->inf) {
		*R = *P;
		return 1;
	}
	if (fe_eq(F, &P->x, &Q->x)) {
		/* y1 + y2 + a1 x2 + a3 == 0: Q = -P */
		fe_mul(F, &t, &E->a1, &Q->x);
		fe_add(F, &t, &t, &E->a3);
		fe_add(F, &t, &t, &P->y);
		fe_add(F, &t, &t, &Q->y);
		if (fe_is_zero(F, &t)) {
			ec_O(R);
			return 1;
		}
		/* doubling (P = Q) */
		/* lam = (3x^2 + 2 a2 x + a4 - a1 y) / (2y + a1 x + a3) */
		fe_mul(F, &num, &P->x, &P->x);
		fe_mul_i64(F, &num, &num, 3);
		fe_mul(F, &t, &E->a2, &P->x);
		fe_mul_i64(F, &t, &t, 2);
		fe_add(F, &num, &num, &t);
		fe_add(F, &num, &num, &E->a4);
		fe_mul(F, &t, &E->a1, &P->y);
		fe_sub(F, &num, &num, &t);
		fe_mul_i64(F, &den, &P->y, 2);
		fe_mul(F, &t, &E->a1, &P->x);
		fe_add(F, &den, &den, &t);
		fe_add(F, &den, &den, &E->a3);
		if (!fe_inv(F, &t, &den)) {
			ec_O(R);
			return 0;
		}
		fe_mul(F, &lam, &num, &t);
	} else {
		fe_sub(F, &num, &Q->y, &P->y);
		fe_sub(F, &den, &Q->x, &P->x);
		if (!fe_inv(F, &t, &den)) {
			ec_O(R);
			return 0;
		}
		fe_mul(F, &lam, &num, &t);
	}
	/* nu = y1 - lam x1 */
	fe_mul(F, &t, &lam, &P->x);
	fe_sub(F, &nu, &P->y, &t);
	/* x3 = lam^2 + a1 lam - a2 - x1 - x2 */
	fe_add(F, &t, &lam, &E->a1);
	fe_mul(F, &x3, &t, &lam);
	fe_sub(F, &x3, &x3, &E->a2);
	fe_sub(F, &x3, &x3, &P->x);
	fe_sub(F, &x3, &x3, &Q->x);
	/* y3 = -(lam + a1) x3 - nu - a3 */
	fe_mul(F, &y3, &t, &x3);
	fe_add(F, &y3, &y3, &nu);
	fe_add(F, &y3, &y3, &E->a3);
	fe_neg(F, &y3, &y3);
	R->x = x3;
	R->y = y3;
	R->inf = 0;
	return 1;
}

int
ec_add(const struct ec *E, struct ecpt *R, const struct ecpt *P,
    const struct ecpt *Q)
{
	return ec_add_core(E, R, P, Q);
}

int
ec_dbl(const struct ec *E, struct ecpt *R, const struct ecpt *P)
{
	return ec_add_core(E, R, P, P);
}

int
ec_sub(const struct ec *E, struct ecpt *R, const struct ecpt *P,
    const struct ecpt *Q)
{
	struct ecpt n;

	ec_neg(E, &n, Q);
	return ec_add_core(E, R, P, &n);
}

int
ec_mul(const struct ec *E, struct ecpt *R, zint k, const struct ecpt *P)
{
	struct ecpt acc, base, t[2];
	zmark m;
	unsigned char *bits;
	size_t nb, i;
	zint kk;
	int ok = 1;

	if (P->inf || z_is_zero(k)) {
		ec_O(R);
		return 1;
	}
	m = z_mark();
	if (z_sign(k) < 0) {
		ec_neg(E, &base, P);
		kk = z_neg(k);
	} else {
		base = *P;
		kk = k;
	}
	nb = z_bits(kk);
	bits = xmalloc(nb);
	{
		/* bits by 30 at a time */
		zint q = kk, r, b30 = z_from_i64((int64_t)1 << 30);
		size_t at = 0;
		while (!z_is_zero(q)) {
			int64_t v;
			int j;
			z_divmod_floor(q, b30, &q, &r);
			v = z_to_i64(r);
			for (j = 0; j < 30 && at < nb; j++)
				bits[at++] = (unsigned char)((v >> j) & 1);
		}
	}
	ec_O(&acc);
	{
		zmark m2 = z_mark();
		for (i = nb; i-- > 0;) {
			if (!ec_dbl(E, &acc, &acc)) {
				ok = 0;
				break;
			}
			if (bits[i] && !ec_add(E, &acc, &acc, &base)) {
				ok = 0;
				break;
			}
			if (E->F->op->keep) {
				t[0] = acc;
				t[1] = base;
				ec_pt_keep(E, m2, t, 2);
				acc = t[0];
				base = t[1];
			}
		}
	}
	free(bits);
	if (!ok)
		ec_O(&acc);
	ec_pt_keep(E, m, &acc, 1);
	*R = acc;
	return ok;
}

int
ec_mul_i64(const struct ec *E, struct ecpt *R, int64_t k, const struct ecpt *P)
{
	return ec_mul(E, R, z_from_i64(k), P);
}

int64_t
ec_order_upto(const struct ec *E, const struct ecpt *P, int64_t max)
{
	struct ecpt Q = *P;
	int64_t n;

	for (n = 1; n <= max; n++) {
		if (Q.inf)
			return n;
		if (!ec_add(E, &Q, &Q, P))
			return 0;
	}
	return 0;
}

int
ec_lift_x(const struct ec *E, const fe *x, struct ecpt *out)
{
	const struct ecf *F = E->F;
	fe b, c, d, s, t, two, i2;

	/* y^2 + b y - c = 0, b = a1 x + a3, c = x^3 + a2 x^2 + a4 x + a6 */
	fe_mul(F, &b, &E->a1, x);
	fe_add(F, &b, &b, &E->a3);
	fe_add(F, &c, x, &E->a2);
	fe_mul(F, &c, &c, x);
	fe_add(F, &c, &c, &E->a4);
	fe_mul(F, &c, &c, x);
	fe_add(F, &c, &c, &E->a6);
	fe_set_i64(F, &two, 2);
	if (fe_is_zero(F, &two)) {
		/* characteristic 2 */
		if (fe_is_zero(F, &b)) {
			if (!fe_sqrt(F, &s, &c))
				return 0;
			out[0].x = *x;
			out[0].y = s;
			out[0].inf = 0;
			return 1;
		}
		/* y = b z, z^2 + z = c / b^2: brute force over small fields */
		if (F->kind == ECF_FQ && F->R->qm1 < (1u << 20)) {
			fe bi, cc, z, zz;
			u64 v;
			int n = 0;
			fe_inv(F, &bi, &b);
			fe_mul(F, &cc, &c, &bi);
			fe_mul(F, &cc, &cc, &bi);
			for (v = 0; v <= F->R->qm1 && n < 2; v++) {
				memset(&z, 0, sizeof z);
				z.w[0] = v;
				fe_mul(F, &zz, &z, &z);
				fe_add(F, &zz, &zz, &z);
				if (fe_eq(F, &zz, &cc)) {
					out[n].x = *x;
					fe_mul(F, &out[n].y, &b, &z);
					out[n].inf = 0;
					n++;
				}
			}
			return n;
		}
		fail("lift_x: characteristic 2 with a1 x + a3 != 0 needs a "
		    "field of at most 2^20 elements here");
	}
	/* d = b^2 + 4c, y = (-b +- sqrt d) / 2 */
	fe_mul(F, &d, &b, &b);
	fe_mul_i64(F, &t, &c, 4);
	fe_add(F, &d, &d, &t);
	if (!fe_sqrt(F, &s, &d))
		return 0;
	if (!fe_inv(F, &i2, &two))
		return 0;
	fe_sub(F, &t, &s, &b);
	fe_mul(F, &out[0].y, &t, &i2);
	out[0].x = *x;
	out[0].inf = 0;
	if (fe_is_zero(F, &s))
		return 1;
	fe_neg(F, &t, &s);
	fe_sub(F, &t, &t, &b);
	fe_mul(F, &out[1].y, &t, &i2);
	out[1].x = *x;
	out[1].inf = 0;
	return 2;
}

void
ec_pt_print(struct sbuf *b, const struct ec *E, const struct ecpt *P)
{
	if (P->inf) {
		sb_puts(b, "O");
		return;
	}
	sb_putc(b, '(');
	fe_print(E->F, b, &P->x);
	sb_puts(b, ", ");
	fe_print(E->F, b, &P->y);
	sb_putc(b, ')');
}

const char *
ec_pt_str(const struct ec *E, const struct ecpt *P)
{
	static struct sbuf b[8];
	static int k;
	struct sbuf *s = &b[k++ & 7];

	sb_clear(s);
	ec_pt_print(s, E, P);
	return sb_str(s);
}

static const char *
skipws(const char *s)
{
	while (*s == ' ' || *s == '\t')
		s++;
	return s;
}

int
ec_pt_parse(const struct ec *E, struct ecpt *P, const char *s,
    const char **end)
{
	const struct ecf *F = E->F;
	fe c[3];
	int n = 0, close;

	s = skipws(s);
	if (*s == 'O' && !isalnum((unsigned char)s[1])) {
		ec_O(P);
		if (end)
			*end = s + 1;
		return 1;
	}
	if (*s != '(' && *s != '[')
		return 0;
	close = *s == '(' ? ')' : ']';
	s++;
	for (;;) {
		const char *e;
		if (n == 3 || !fe_parse(F, &c[n], s, &e))
			return 0;
		n++;
		s = skipws(e);
		if (*s == ',' || *s == ':') {
			s++;
			continue;
		}
		if (*s == close) {
			s++;
			break;
		}
		return 0;
	}
	if (end)
		*end = s;
	if (n == 2) {
		ec_pt(E, P, &c[0], &c[1]);
		return 1;
	}
	if (n == 3) {
		fe zi;
		if (fe_is_zero(F, &c[2])) {
			ec_O(P);
			return 1;
		}
		if (!fe_inv(F, &zi, &c[2]))
			return 0;
		fe_mul(F, &P->x, &c[0], &zi);
		fe_mul(F, &P->y, &c[1], &zi);
		P->inf = 0;
		return 1;
	}
	return 0;
}

void
ec_pt_keep(const struct ec *E, zmark m, struct ecpt *P, size_t n)
{
	static fe *tmp;
	static size_t cap;
	size_t i;

	if (!E->F->op->keep) 
		return;
	if (2 * n > cap) {
		cap = 2 * n + 8;
		tmp = xrealloc(tmp, cap * sizeof *tmp);
	}
	for (i = 0; i < n; i++) {
		tmp[2 * i] = P[i].x;
		tmp[2 * i + 1] = P[i].y;
	}
	fe_keep(E->F, m, tmp, 2 * n);
	for (i = 0; i < n; i++) {
		P[i].x = tmp[2 * i];
		P[i].y = tmp[2 * i + 1];
	}
}

/* ---- point tables ---- */

void
ecpts_init(struct ecpts *T)
{
	memset(T, 0, sizeof *T);
}

void
ecpts_free(struct ecpts *T)
{
	free(T->x);
	free(T->y);
	free(T->inf);
	memset(T, 0, sizeof *T);
}

uint32_t
ecpts_push(struct ecpts *T, const struct ecpt *P)
{
	if (T->n == T->cap) {
		T->cap = newcap(T->cap, T->n + 1);
		T->x = xrealloc(T->x, T->cap * sizeof *T->x);
		T->y = xrealloc(T->y, T->cap * sizeof *T->y);
		T->inf = xrealloc(T->inf, T->cap * sizeof *T->inf);
	}
	T->x[T->n] = P->x;
	T->y[T->n] = P->y;
	T->inf[T->n] = (uint8_t)P->inf;
	return T->n++;
}

void
ecpts_get(const struct ecpts *T, uint32_t i, struct ecpt *P)
{
	P->x = T->x[i];
	P->y = T->y[i];
	P->inf = T->inf[i];
}

void
ecpts_set(struct ecpts *T, uint32_t i, const struct ecpt *P)
{
	T->x[i] = P->x;
	T->y[i] = P->y;
	T->inf[i] = (uint8_t)P->inf;
}

uint32_t
ecpts_find(const struct ec *E, const struct ecpts *T, const struct ecpt *P)
{
	uint32_t i;
	struct ecpt Q;

	for (i = 0; i < T->n; i++) {
		ecpts_get(T, i, &Q);
		if (ec_pt_eq(E, &Q, P))
			return i;
	}
	return UINT32_MAX;
}

void
ecpts_keep(const struct ec *E, zmark m, struct ecpts *T)
{
	static fe *tmp;
	static size_t cap;
	uint32_t i;

	if (!E->F->op->keep)
		return;
	if (2 * (size_t)T->n > cap) {
		cap = 2 * (size_t)T->n + 8;
		tmp = xrealloc(tmp, cap * sizeof *tmp);
	}
	for (i = 0; i < T->n; i++) {
		tmp[2 * i] = T->x[i];
		tmp[2 * i + 1] = T->y[i];
	}
	fe_keep(E->F, m, tmp, 2 * (size_t)T->n);
	for (i = 0; i < T->n; i++) {
		T->x[i] = tmp[2 * i];
		T->y[i] = tmp[2 * i + 1];
	}
}

/* ================================================================== */
/* isomorphisms                                                        */
/* ================================================================== */

void
eciso_identity(const struct ecf *F, struct eciso *m)
{
	fe_one(F, &m->u);
	fe_zero(F, &m->r);
	m->s = m->t = m->r;
}

void
eciso_apply(const struct ec *E, const struct eciso *m, struct ec *Ep)
{
	const struct ecf *F = E->F;
	fe ui, u2, u3, u4, u6, a[5], t, s;

	if (!fe_inv(F, &ui, &m->u))
		fail("isomorphism: u is not a unit");
	fe_mul(F, &u2, &ui, &ui);
	fe_mul(F, &u3, &u2, &ui);
	fe_mul(F, &u4, &u2, &u2);
	fe_mul(F, &u6, &u4, &u2);
	/* u a1' = a1 + 2s */
	fe_mul_i64(F, &t, &m->s, 2);
	fe_add(F, &t, &t, &E->a1);
	fe_mul(F, &a[0], &t, &ui);
	/* u^2 a2' = a2 - s a1 + 3r - s^2 */
	fe_mul(F, &t, &m->s, &E->a1);
	fe_sub(F, &s, &E->a2, &t);
	fe_mul_i64(F, &t, &m->r, 3);
	fe_add(F, &s, &s, &t);
	fe_mul(F, &t, &m->s, &m->s);
	fe_sub(F, &s, &s, &t);
	fe_mul(F, &a[1], &s, &u2);
	/* u^3 a3' = a3 + r a1 + 2t */
	fe_mul(F, &t, &m->r, &E->a1);
	fe_add(F, &s, &E->a3, &t);
	fe_mul_i64(F, &t, &m->t, 2);
	fe_add(F, &s, &s, &t);
	fe_mul(F, &a[2], &s, &u3);
	/* u^4 a4' = a4 - s a3 + 2 r a2 - (t + r s) a1 + 3 r^2 - 2 s t */
	fe_mul(F, &t, &m->s, &E->a3);
	fe_sub(F, &s, &E->a4, &t);
	fe_mul(F, &t, &m->r, &E->a2);
	fe_mul_i64(F, &t, &t, 2);
	fe_add(F, &s, &s, &t);
	{
		fe w;
		fe_mul(F, &w, &m->r, &m->s);
		fe_add(F, &w, &w, &m->t);
		fe_mul(F, &w, &w, &E->a1);
		fe_sub(F, &s, &s, &w);
		fe_mul(F, &w, &m->r, &m->r);
		fe_mul_i64(F, &w, &w, 3);
		fe_add(F, &s, &s, &w);
		fe_mul(F, &w, &m->s, &m->t);
		fe_mul_i64(F, &w, &w, 2);
		fe_sub(F, &s, &s, &w);
	}
	fe_mul(F, &a[3], &s, &u4);
	/* u^6 a6' = a6 + r a4 + r^2 a2 + r^3 - t a3 - t^2 - r t a1 */
	{
		fe r2, w;
		fe_mul(F, &r2, &m->r, &m->r);
		fe_mul(F, &w, &m->r, &E->a4);
		fe_add(F, &s, &E->a6, &w);
		fe_mul(F, &w, &r2, &E->a2);
		fe_add(F, &s, &s, &w);
		fe_mul(F, &w, &r2, &m->r);
		fe_add(F, &s, &s, &w);
		fe_mul(F, &w, &m->t, &E->a3);
		fe_sub(F, &s, &s, &w);
		fe_mul(F, &w, &m->t, &m->t);
		fe_sub(F, &s, &s, &w);
		fe_mul(F, &w, &m->r, &m->t);
		fe_mul(F, &w, &w, &E->a1);
		fe_sub(F, &s, &s, &w);
	}
	fe_mul(F, &a[4], &s, &u6);
	ec_init(Ep, F, a);
}

void
eciso_pt(const struct ec *E, const struct eciso *m, struct ecpt *R,
    const struct ecpt *P)
{
	const struct ecf *F = E->F;
	fe ui, u2, xr, t;

	if (P->inf) {
		ec_O(R);
		return;
	}
	fe_inv(F, &ui, &m->u);
	fe_mul(F, &u2, &ui, &ui);
	/* x' = (x - r)/u^2, y' = (y - s (x - r) - t)/u^3 */
	fe_sub(F, &xr, &P->x, &m->r);
	fe_mul(F, &t, &m->s, &xr);
	fe_sub(F, &t, &P->y, &t);
	fe_sub(F, &t, &t, &m->t);
	fe_mul(F, &t, &t, &u2);
	fe_mul(F, &R->y, &t, &ui);
	fe_mul(F, &R->x, &xr, &u2);
	R->inf = 0;
}

void
eciso_pt_inv(const struct ec *E, const struct eciso *m, struct ecpt *R,
    const struct ecpt *Pp)
{
	const struct ecf *F = E->F;
	fe u2, u3, x, y, t;

	if (Pp->inf) {
		ec_O(R);
		return;
	}
	fe_mul(F, &u2, &m->u, &m->u);
	fe_mul(F, &u3, &u2, &m->u);
	/* x = u^2 x' + r, y = u^3 y' + s u^2 x' + t */
	fe_mul(F, &x, &u2, &Pp->x);
	fe_mul(F, &t, &m->s, &x);
	fe_add(F, &x, &x, &m->r);
	fe_mul(F, &y, &u3, &Pp->y);
	fe_add(F, &y, &y, &t);
	fe_add(F, &y, &y, &m->t);
	R->x = x;
	R->y = y;
	R->inf = 0;
}

void
eciso_compose(const struct ecf *F, struct eciso *m, const struct eciso *m1,
    const struct eciso *m2)
{
	struct eciso r;
	fe u12, t;

	/* u = u1 u2, r = u1^2 r2 + r1, s = s1 + u1 s2,
	 * t = t1 + u1^3 t2 + s1 u1^2 r2 */
	fe_mul(F, &r.u, &m1->u, &m2->u);
	fe_mul(F, &u12, &m1->u, &m1->u);
	fe_mul(F, &t, &u12, &m2->r);
	fe_add(F, &r.r, &t, &m1->r);
	fe_mul(F, &t, &m1->u, &m2->s);
	fe_add(F, &r.s, &m1->s, &t);
	fe_mul(F, &t, &u12, &m1->u);
	fe_mul(F, &t, &t, &m2->t);
	fe_add(F, &r.t, &m1->t, &t);
	fe_mul(F, &t, &m1->s, &u12);
	fe_mul(F, &t, &t, &m2->r);
	fe_add(F, &r.t, &r.t, &t);
	*m = r;
}

void
eciso_inverse(const struct ecf *F, struct eciso *r, const struct eciso *m)
{
	struct eciso o;
	fe ui, u2, u3, t;

	if (!fe_inv(F, &ui, &m->u))
		fail("isomorphism: u is not a unit");
	fe_mul(F, &u2, &ui, &ui);
	fe_mul(F, &u3, &u2, &ui);
	o.u = ui;
	fe_mul(F, &o.r, &m->r, &u2);
	fe_neg(F, &o.r, &o.r);
	fe_mul(F, &o.s, &m->s, &ui);
	fe_neg(F, &o.s, &o.s);
	fe_mul(F, &t, &m->r, &m->s);
	fe_sub(F, &t, &t, &m->t);
	fe_mul(F, &o.t, &t, &u3);
	*r = o;
}

void
eciso_to_short(const struct ec *E, struct eciso *m, struct ec *Es)
{
	const struct ecf *F = E->F;
	fe six, t, h;

	fe_set_i64(F, &six, 6);
	if (fe_is_zero(F, &six))
		fail("short Weierstrass form: characteristic 2 or 3");
	/* u = 1/6, r = -b2/12, s = -a1/2, t = -(a3 + r a1)/2 */
	fe_inv(F, &m->u, &six);
	fe_set_i64(F, &t, 12);
	fe_inv(F, &t, &t);
	fe_mul(F, &m->r, &E->b2, &t);
	fe_neg(F, &m->r, &m->r);
	fe_set_i64(F, &h, 2);
	fe_inv(F, &h, &h);
	fe_mul(F, &m->s, &E->a1, &h);
	fe_neg(F, &m->s, &m->s);
	fe_mul(F, &t, &m->r, &E->a1);
	fe_add(F, &t, &t, &E->a3);
	fe_mul(F, &m->t, &t, &h);
	fe_neg(F, &m->t, &m->t);
	if (Es)
		eciso_apply(E, m, Es);
}

/* a k-th root of a in F (k = 2, 3, 6): Q exactly, others by sqrt only */
static int
fe_root(const struct ecf *F, fe *r, const fe *a, int k)
{
	if (k == 2)
		return fe_sqrt(F, r, a);
	if (F->kind == ECF_Q) {
		struct zq q = fe_q(a);
		int e1, e2;
		zint n = zx_iroot(q.n, k, &e1), d;
		if (!e1 || z_sign(q.d) <= 0)
			return 0;
		d = zx_iroot(q.d, k, &e2);
		if (!e2)
			return 0;
		*r = fe_from_q(zq_make(n, d));
		return 1;
	}
	if (k == 6) {
		fe s;
		/* a sixth root: a square root of a cube root; cube roots by
		 * search in small fields only */
		if (F->kind == ECF_FQ && F->R->qm1 < (1u << 20)) {
			u64 v;
			for (v = 0; v <= F->R->qm1; v++) {
				fe z, z6;
				memset(&z, 0, sizeof z);
				z.w[0] = v;
				fe_pow(F, &z6, &z, 6);
				if (fe_eq(F, &z6, a)) {
					*r = z;
					return 1;
				}
			}
		}
		(void)s;
		return 0;
	}
	if (k == 3 && F->kind == ECF_FQ && F->R->qm1 < (1u << 20)) {
		u64 v;
		for (v = 0; v <= F->R->qm1; v++) {
			fe z, z3;
			memset(&z, 0, sizeof z);
			z.w[0] = v;
			fe_pow(F, &z3, &z, 3);
			if (fe_eq(F, &z3, a)) {
				*r = z;
				return 1;
			}
		}
	}
	return 0;
}

int
eciso_find(const struct ec *E1, const struct ec *E2, struct eciso *m)
{
	const struct ecf *F = E1->F;
	struct eciso s1, s2, sc, s2i, t;
	struct ec S1, S2;
	fe u, u2, w, six;

	if (E1->singular || E2->singular)
		return 0;
	fe_set_i64(F, &six, 6);
	if (fe_is_zero(F, &six))
		return 0;   /* characteristic 2, 3: not handled here */
	if (!fe_eq(F, &E1->j, &E2->j))
		return 0;
	eciso_to_short(E1, &s1, &S1);
	eciso_to_short(E2, &s2, &S2);
	/* S1 -> S2 by [u, 0, 0, 0]: A2 = A1 / u^4, B2 = B1 / u^6 */
	if (fe_is_zero(F, &S1.a4)) {
		/* j = 0: u^6 = B1 / B2 */
		fe_div(F, &w, &S1.a6, &S2.a6);
		if (!fe_root(F, &u, &w, 6))
			return 0;
	} else if (fe_is_zero(F, &S1.a6)) {
		/* j = 1728: u^4 = A1 / A2 */
		fe_div(F, &w, &S1.a4, &S2.a4);
		if (!fe_sqrt(F, &u2, &w) || !fe_sqrt(F, &u, &u2)) {
			fe_neg(F, &u2, &u2);
			if (!fe_sqrt(F, &u, &u2))
				return 0;
		}
	} else {
		/* u^2 = (B1 / B2) / (A1 / A2) */
		fe x, y;
		fe_div(F, &x, &S1.a6, &S2.a6);
		fe_div(F, &y, &S1.a4, &S2.a4);
		fe_div(F, &u2, &x, &y);
		if (!fe_sqrt(F, &u, &u2))
			return 0;
	}
	eciso_identity(F, &sc);
	sc.u = u;
	/* check */
	{
		struct ec T;
		eciso_apply(&S1, &sc, &T);
		if (!fe_eq(F, &T.a4, &S2.a4) || !fe_eq(F, &T.a6, &S2.a6))
			return 0;
	}
	eciso_compose(F, &t, &s1, &sc);
	eciso_inverse(F, &s2i, &s2);
	eciso_compose(F, m, &t, &s2i);
	{
		struct ec T;
		eciso_apply(E1, m, &T);
		if (!fe_eq(F, &T.a1, &E2->a1) || !fe_eq(F, &T.a2, &E2->a2) ||
		    !fe_eq(F, &T.a3, &E2->a3) || !fe_eq(F, &T.a4, &E2->a4) ||
		    !fe_eq(F, &T.a6, &E2->a6))
			fail("internal: eciso_find composed to another curve");
	}
	return 1;
}

void
eciso_print(struct sbuf *b, const struct ecf *F, const struct eciso *m)
{
	sb_printf(b, "[u, r, s, t] = [%s, %s, %s, %s]", fe_str(F, &m->u),
	    fe_str(F, &m->r), fe_str(F, &m->s), fe_str(F, &m->t));
}

/* ================================================================== */
/* ternary cubics and the flex algorithm                               */
/* ================================================================== */

const unsigned char ec_cub_mono[10][3] = {
	{3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2},
	{0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3}
};

static int
mono_index(int e0, int e1, int e2)
{
	int i;

	for (i = 0; i < 10; i++)
		if (ec_cub_mono[i][0] == e0 && ec_cub_mono[i][1] == e1 &&
		    ec_cub_mono[i][2] == e2)
			return i;
	return -1;
}

void
ec_cub_eval(const struct ecf *F, const fe *c, const fe *P, fe *r)
{
	fe acc, t;
	int i, k;

	fe_zero(F, &acc);
	for (i = 0; i < 10; i++) {
		if (fe_is_zero(F, &c[i]))
			continue;
		t = c[i];
		for (k = 0; k < 3; k++) {
			int e;
			for (e = 0; e < ec_cub_mono[i][k]; e++)
				fe_mul(F, &t, &t, &P[k]);
		}
		fe_add(F, &acc, &acc, &t);
	}
	*r = acc;
}

void
ec_cub_grad(const struct ecf *F, const fe *c, const fe *P, fe *g)
{
	int v, i, k;

	for (v = 0; v < 3; v++) {
		fe acc, t;
		fe_zero(F, &acc);
		for (i = 0; i < 10; i++) {
			int ev = ec_cub_mono[i][v];
			if (!ev || fe_is_zero(F, &c[i]))
				continue;
			fe_mul_i64(F, &t, &c[i], ev);
			for (k = 0; k < 3; k++) {
				int e, ek = ec_cub_mono[i][k] - (k == v);
				for (e = 0; e < ek; e++)
					fe_mul(F, &t, &t, &P[k]);
			}
			fe_add(F, &acc, &acc, &t);
		}
		g[v] = acc;
	}
}

void
ec_cub_transform(const struct ecf *F, const fe *c, const fe *Minv, fe *out)
{
	/* X_k = sum_j Minv[k][j] U_j; expand each monomial as a product of
	 * three linear forms in a dense 4x4x4 cube */
	fe res[10];
	int i, j;

	for (i = 0; i < 10; i++)
		fe_zero(F, &res[i]);
	for (i = 0; i < 10; i++) {
		fe cube[4][4][4], nc[4][4][4];
		int a, b, d, k, f, e;
		if (fe_is_zero(F, &c[i]))
			continue;
		for (a = 0; a < 4; a++)
			for (b = 0; b < 4; b++)
				for (d = 0; d < 4; d++)
					fe_zero(F, &cube[a][b][d]);
		cube[0][0][0] = c[i];
		for (k = 0; k < 3; k++)
			for (e = 0; e < ec_cub_mono[i][k]; e++) {
				for (a = 0; a < 4; a++)
					for (b = 0; b < 4; b++)
						for (d = 0; d < 4; d++)
							fe_zero(F, &nc[a][b][d]);
				for (a = 0; a < 3; a++)
					for (b = 0; b + a < 3; b++)
						for (d = 0; d + a + b < 3; d++) {
							fe t;
							if (fe_is_zero(F, &cube[a][b][d]))
								continue;
							for (f = 0; f < 3; f++) {
								const fe *l = &Minv[k * 3 + f];
								fe *dst;
								if (fe_is_zero(F, l))
									continue;
								fe_mul(F, &t, &cube[a][b][d], l);
								dst = &nc[a + (f == 0)][b + (f == 1)][d + (f == 2)];
								fe_add(F, dst, dst, &t);
							}
						}
				memcpy(cube, nc, sizeof cube);
			}
		for (j = 0; j < 10; j++) {
			fe *v = &cube[ec_cub_mono[j][0]][ec_cub_mono[j][1]]
			    [ec_cub_mono[j][2]];
			fe_add(F, &res[j], &res[j], v);
		}
	}
	memcpy(out, res, sizeof res);
}

static int
mat3_inv(const struct ecf *F, const fe *m, fe *inv)
{
	fe det, t, u, di;
	int i, j;

	/* adjugate */
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++) {
			int r0 = (j + 1) % 3, r1 = (j + 2) % 3;
			int c0 = (i + 1) % 3, c1 = (i + 2) % 3;
			fe_mul(F, &t, &m[r0 * 3 + c0], &m[r1 * 3 + c1]);
			fe_mul(F, &u, &m[r0 * 3 + c1], &m[r1 * 3 + c0]);
			fe_sub(F, &inv[i * 3 + j], &t, &u);
		}
	fe_zero(F, &det);
	for (j = 0; j < 3; j++) {
		fe_mul(F, &t, &m[j], &inv[j * 3]);
		fe_add(F, &det, &det, &t);
	}
	if (!fe_inv(F, &di, &det))
		return 0;
	for (i = 0; i < 9; i++)
		fe_mul(F, &inv[i], &inv[i], &di);
	return 1;
}

static void
mat3_vec(const struct ecf *F, const fe *m, const fe *x, fe *y)
{
	int i, j;
	fe r[3], t;

	for (i = 0; i < 3; i++) {
		fe_zero(F, &r[i]);
		for (j = 0; j < 3; j++) {
			fe_mul(F, &t, &m[i * 3 + j], &x[j]);
			fe_add(F, &r[i], &r[i], &t);
		}
	}
	memcpy(y, r, sizeof r);
}

int
ec_cub_flex_to_w(const struct ecf *F, const fe *c, const fe *P, struct ec *E,
    struct ec_flexmap *m)
{
	fe g[3], val, forms[3][3], G[10], a[5], lam, mu, K, Ki, t;
	int k, pick = -1, vk = -1;

	ec_cub_eval(F, c, P, &val);
	if (!fe_is_zero(F, &val))
		return 0;
	ec_cub_grad(F, c, P, g);
	if (fe_is_zero(F, &g[0]) && fe_is_zero(F, &g[1]) && fe_is_zero(F, &g[2]))
		return 0;   /* singular point */
	/* linear forms vanishing at P: P x e_k */
	fe_zero(F, &forms[0][0]);
	forms[0][1] = P[2];
	fe_neg(F, &forms[0][2], &P[1]);
	fe_neg(F, &forms[1][0], &P[2]);
	fe_zero(F, &forms[1][1]);
	forms[1][2] = P[0];
	forms[2][0] = P[1];
	fe_neg(F, &forms[2][1], &P[0]);
	fe_zero(F, &forms[2][2]);
	for (k = 0; k < 3 && pick < 0; k++) {
		/* not proportional to g: the cross product is nonzero */
		fe c0, c1, c2, x, y;
		fe_mul(F, &x, &forms[k][1], &g[2]);
		fe_mul(F, &y, &forms[k][2], &g[1]);
		fe_sub(F, &c0, &x, &y);
		fe_mul(F, &x, &forms[k][2], &g[0]);
		fe_mul(F, &y, &forms[k][0], &g[2]);
		fe_sub(F, &c1, &x, &y);
		fe_mul(F, &x, &forms[k][0], &g[1]);
		fe_mul(F, &y, &forms[k][1], &g[0]);
		fe_sub(F, &c2, &x, &y);
		if (!fe_is_zero(F, &c0) || !fe_is_zero(F, &c1) ||
		    !fe_is_zero(F, &c2))
			pick = k;
	}
	for (k = 0; k < 3 && vk < 0; k++)
		if (!fe_is_zero(F, &P[k]))
			vk = k;
	if (pick < 0 || vk < 0)
		return 0;
	for (k = 0; k < 3; k++) {
		m->M[k] = forms[pick][k];
		fe_set_i64(F, &m->M[3 + k], k == vk);
		m->M[6 + k] = g[k];
	}
	if (!mat3_inv(F, m->M, m->Minv))
		return 0;
	ec_cub_transform(F, c, m->Minv, G);
	/* V^3, U V^2, U^2 V vanish at a flex (0:1:0) with tangent W = 0 */
	if (!fe_is_zero(F, &G[mono_index(0, 3, 0)]) ||
	    !fe_is_zero(F, &G[mono_index(1, 2, 0)]))
		fail("internal: the cubic's transform at its point");
	if (!fe_is_zero(F, &G[mono_index(2, 1, 0)]))
		return 0;   /* not a flex */
	{
		fe al = G[mono_index(3, 0, 0)], be = G[mono_index(0, 2, 1)];
		fe ga = G[mono_index(1, 1, 1)], de = G[mono_index(0, 1, 2)];
		fe ep = G[mono_index(2, 0, 1)], ze = G[mono_index(1, 0, 2)];
		fe et = G[mono_index(0, 0, 3)];
		if (fe_is_zero(F, &al) || fe_is_zero(F, &be))
			return 0;   /* singular (a line is a component) */
		/* lam = -al be, mu = al^2 be, K = al^4 be^3 */
		fe_mul(F, &lam, &al, &be);
		fe_neg(F, &lam, &lam);
		fe_mul(F, &mu, &al, &al);
		fe_mul(F, &mu, &mu, &be);
		fe_mul(F, &K, &mu, &mu);
		fe_mul(F, &K, &K, &be);
		if (!fe_inv(F, &Ki, &K))
			return 0;
		fe_mul(F, &t, &ga, &lam);
		fe_mul(F, &t, &t, &mu);
		fe_mul(F, &a[0], &t, &Ki);
		fe_mul(F, &t, &ep, &lam);
		fe_mul(F, &t, &t, &lam);
		fe_mul(F, &t, &t, &Ki);
		fe_neg(F, &a[1], &t);
		fe_mul(F, &t, &de, &mu);
		fe_mul(F, &a[2], &t, &Ki);
		fe_mul(F, &t, &ze, &lam);
		fe_mul(F, &t, &t, &Ki);
		fe_neg(F, &a[3], &t);
		fe_mul(F, &t, &et, &Ki);
		fe_neg(F, &a[4], &t);
	}
	ec_init(E, F, a);
	if (E->singular)
		return 0;
	m->lam = lam;
	m->mu = mu;
	m->P[0] = P[0];
	m->P[1] = P[1];
	m->P[2] = P[2];
	return 1;
}

int
ec_cub_is_flex(const struct ecf *F, const fe *c, const fe *P)
{
	struct ec E;
	struct ec_flexmap m;

	return ec_cub_flex_to_w(F, c, P, &E, &m);
}

int
ec_flexmap_fwd(const struct ec *E, const struct ec_flexmap *m, const fe *X,
    struct ecpt *R)
{
	const struct ecf *F = E->F;
	fe U[3], d;

	mat3_vec(F, m->M, X, U);
	if (fe_is_zero(F, &U[2])) {
		ec_O(R);
		return 1;
	}
	fe_mul(F, &d, &U[2], &m->lam);
	if (!fe_div(F, &R->x, &U[0], &d))
		return 0;
	fe_mul(F, &d, &U[2], &m->mu);
	if (!fe_div(F, &R->y, &U[1], &d))
		return 0;
	R->inf = 0;
	return 1;
}

void
ec_flexmap_back(const struct ec *E, const struct ec_flexmap *m,
    const struct ecpt *R, fe *X)
{
	const struct ecf *F = E->F;
	fe U[3];

	if (R->inf) {
		X[0] = m->P[0];
		X[1] = m->P[1];
		X[2] = m->P[2];
		return;
	}
	fe_mul(F, &U[0], &R->x, &m->lam);
	fe_mul(F, &U[1], &R->y, &m->mu);
	fe_one(F, &U[2]);
	mat3_vec(F, m->Minv, U, X);
}

/* ================================================================== */
/* models                                                              */
/* ================================================================== */

static const char *const mkind[] = {NULL, "weierstrass", "short",
	"montgomery", "edwards", "twisted_edwards", "legendre", "hessian"};

const char *
ecm_kind_name(int kind)
{
	return kind >= ECM_WEIERSTRASS && kind <= ECM_HESSIAN ? mkind[kind] :
	    "?";
}

int
ecm_kind_parse(const char *s)
{
	int k;

	for (k = ECM_WEIERSTRASS; k <= ECM_HESSIAN; k++)
		if (!strcmp(s, mkind[k]))
			return k;
	if (!strcmp(s, "twisted") || !strcmp(s, "tedwards"))
		return ECM_TWISTED_EDWARDS;
	return 0;
}

/* the twisted-Edwards parameters (a, d) of a model of kind Edwards */
static void
ted_params(const struct ecmodel *M, fe *a, fe *d)
{
	if (M->kind == ECM_EDWARDS) {
		fe_one(M->F, a);
		*d = M->c[0];
	} else {
		*a = M->c[0];
		*d = M->c[1];
	}
}

void
ecm_print(struct sbuf *b, const struct ecmodel *M)
{
	const struct ecf *F = M->F;
	struct ec E;

	switch (M->kind) {
	case ECM_WEIERSTRASS:
		ec_init(&E, F, M->c);
		ec_print(b, &E);
		return;
	case ECM_SHORT:
		sb_puts(b, "y^2 = x^3");
		put_term(b, F, &M->c[0], "x", 0);
		put_term(b, F, &M->c[1], "", 0);
		return;
	case ECM_MONTGOMERY:
		if (!fe_is_one(F, &M->c[1]))
			sb_printf(b, "%s*", fe_str(F, &M->c[1]));
		sb_puts(b, "y^2 = x^3");
		put_term(b, F, &M->c[0], "x^2", 0);
		sb_puts(b, " + x");
		return;
	case ECM_EDWARDS:
	case ECM_TWISTED_EDWARDS: {
		fe a, d;
		ted_params(M, &a, &d);
		if (!fe_is_one(F, &a))
			sb_printf(b, "%s*", fe_str(F, &a));
		sb_puts(b, "x^2 + y^2 = 1");
		put_term(b, F, &d, "x^2*y^2", 0);
		return;
	}
	case ECM_LEGENDRE:
		sb_printf(b, "y^2 = x*(x - 1)*(x - %s)", fe_str(F, &M->c[0]));
		return;
	case ECM_HESSIAN:
		if (!fe_is_one(F, &M->c[0]))
			sb_printf(b, "%s*", fe_str(F, &M->c[0]));
		sb_printf(b, "x^3 + y^3 + 1 = %s*x*y", fe_str(F, &M->c[1]));
		return;
	}
}

int
ecm_on(const struct ecmodel *M, const struct ecmpt *P)
{
	const struct ecf *F = M->F;
	fe l, r, t, x2, y2;

	switch (M->kind) {
	case ECM_WEIERSTRASS:
	case ECM_SHORT:
	case ECM_LEGENDRE: {
		struct ec E;
		struct ecpt Q;
		if (!ecm_to_weierstrass(M, &E))
			return 0;
		Q.x = P->x;
		Q.y = P->y;
		Q.inf = P->inf;
		return ec_on(&E, &Q);
	}
	case ECM_MONTGOMERY:
		if (P->inf)
			return 1;
		fe_mul(F, &l, &P->y, &P->y);
		fe_mul(F, &l, &l, &M->c[1]);
		fe_add(F, &r, &P->x, &M->c[0]);
		fe_mul(F, &r, &r, &P->x);
		fe_set_i64(F, &t, 1);
		fe_add(F, &r, &r, &t);
		fe_mul(F, &r, &r, &P->x);
		return fe_eq(F, &l, &r);
	case ECM_EDWARDS:
	case ECM_TWISTED_EDWARDS: {
		fe a, d;
		if (P->inf)
			return 0;
		ted_params(M, &a, &d);
		fe_mul(F, &x2, &P->x, &P->x);
		fe_mul(F, &y2, &P->y, &P->y);
		fe_mul(F, &l, &a, &x2);
		fe_add(F, &l, &l, &y2);
		fe_mul(F, &r, &x2, &y2);
		fe_mul(F, &r, &r, &d);
		fe_one(F, &t);
		fe_add(F, &r, &r, &t);
		return fe_eq(F, &l, &r);
	}
	case ECM_HESSIAN: {
		/* a X^3 + Y^3 + Z^3 = d X Y Z with Z = 1, or 0 when inf */
		fe X3, Y3, Z;
		if (P->inf)
			fe_zero(F, &Z);
		else
			fe_one(F, &Z);
		fe_pow(F, &X3, &P->x, 3);
		fe_mul(F, &X3, &X3, &M->c[0]);
		fe_pow(F, &Y3, &P->y, 3);
		fe_add(F, &l, &X3, &Y3);
		if (!P->inf)
			fe_add(F, &l, &l, &Z);
		fe_mul(F, &r, &P->x, &P->y);
		fe_mul(F, &r, &r, &Z);
		fe_mul(F, &r, &r, &M->c[1]);
		if (P->inf && fe_is_zero(F, &P->x) && fe_is_zero(F, &P->y))
			return 0;
		return fe_eq(F, &l, &r);
	}
	}
	return 0;
}

void
ecm_neutral(const struct ecmodel *M, struct ecmpt *P)
{
	const struct ecf *F = M->F;

	memset(P, 0, sizeof *P);
	switch (M->kind) {
	case ECM_EDWARDS:
	case ECM_TWISTED_EDWARDS:
		fe_zero(F, &P->x);
		fe_one(F, &P->y);
		return;
	case ECM_HESSIAN:
		fe_zero(F, &P->x);
		fe_one(F, &P->y);
		fe_neg(F, &P->y, &P->y);
		return;
	default:
		P->inf = 1;
	}
}

/* the Hessian a X^3 + Y^3 + Z^3 - d X Y Z as a ternary cubic */
static void
hess_cubic(const struct ecmodel *M, fe *c)
{
	const struct ecf *F = M->F;
	int i;

	for (i = 0; i < 10; i++)
		fe_zero(F, &c[i]);
	c[mono_index(3, 0, 0)] = M->c[0];
	fe_one(F, &c[mono_index(0, 3, 0)]);
	fe_one(F, &c[mono_index(0, 0, 3)]);
	fe_neg(F, &c[mono_index(1, 1, 1)], &M->c[1]);
}

/* Montgomery parameters (A, B) of a twisted Edwards model:
 * A = 2 (a + d) / (a - d), B = 4 / (a - d) */
static int
ted_to_mont(const struct ecmodel *M, fe *A, fe *B)
{
	const struct ecf *F = M->F;
	fe a, d, s, df, t;

	ted_params(M, &a, &d);
	fe_sub(F, &df, &a, &d);
	if (!fe_inv(F, &t, &df))
		return 0;
	fe_add(F, &s, &a, &d);
	fe_mul_i64(F, &s, &s, 2);
	fe_mul(F, A, &s, &t);
	fe_mul_i64(F, B, &t, 4);
	return 1;
}

int
ecm_to_weierstrass(const struct ecmodel *M, struct ec *E)
{
	const struct ecf *F = M->F;
	fe a[5], A, B, Bi, t;
	int i;

	for (i = 0; i < 5; i++)
		fe_zero(F, &a[i]);
	switch (M->kind) {
	case ECM_WEIERSTRASS:
		ec_init(E, F, M->c);
		break;
	case ECM_SHORT:
		a[3] = M->c[0];
		a[4] = M->c[1];
		ec_init(E, F, a);
		break;
	case ECM_LEGENDRE:
		/* x^3 - (1 + l) x^2 + l x */
		fe_one(F, &t);
		fe_add(F, &t, &t, &M->c[0]);
		fe_neg(F, &a[1], &t);
		a[3] = M->c[0];
		ec_init(E, F, a);
		break;
	case ECM_MONTGOMERY:
	case ECM_EDWARDS:
	case ECM_TWISTED_EDWARDS:
		if (M->kind == ECM_MONTGOMERY) {
			A = M->c[0];
			B = M->c[1];
		} else if (!ted_to_mont(M, &A, &B))
			return 0;
		if (!fe_inv(F, &Bi, &B))
			return 0;
		/* a2 = A / B, a4 = 1 / B^2 */
		fe_mul(F, &a[1], &A, &Bi);
		fe_mul(F, &a[3], &Bi, &Bi);
		ec_init(E, F, a);
		break;
	case ECM_HESSIAN: {
		fe c[10], P[3];
		struct ec_flexmap fm;
		hess_cubic(M, c);
		fe_zero(F, &P[0]);
		fe_one(F, &P[1]);
		fe_neg(F, &P[1], &P[1]);
		fe_one(F, &P[2]);
		if (!ec_cub_flex_to_w(F, c, P, E, &fm))
			return 0;
		break;
	}
	default:
		return 0;
	}
	return !E->singular;
}

int
ecm_pt_to_w(const struct ecmodel *M, const struct ec *E,
    const struct ecmpt *P, struct ecpt *R)
{
	const struct ecf *F = M->F;
	fe A, B, Bi, u, v, t, one;

	switch (M->kind) {
	case ECM_WEIERSTRASS:
	case ECM_SHORT:
	case ECM_LEGENDRE:
		R->x = P->x;
		R->y = P->y;
		R->inf = P->inf;
		return 1;
	case ECM_MONTGOMERY:
		if (P->inf) {
			ec_O(R);
			return 1;
		}
		/* u = x / B, v = y / B */
		if (!fe_inv(F, &Bi, &M->c[1]))
			return 0;
		fe_mul(F, &R->x, &P->x, &Bi);
		fe_mul(F, &R->y, &P->y, &Bi);
		R->inf = 0;
		return 1;
	case ECM_EDWARDS:
	case ECM_TWISTED_EDWARDS:
		if (P->inf)
			return 0;
		if (!ted_to_mont(M, &A, &B) || !fe_inv(F, &Bi, &B))
			return 0;
		fe_one(F, &one);
		if (fe_is_zero(F, &P->x)) {
			if (fe_eq(F, &P->y, &one)) {
				ec_O(R);    /* the neutral (0, 1) */
				return 1;
			}
			/* (0, -1) -> Montgomery (0, 0) */
			fe_zero(F, &R->x);
			fe_zero(F, &R->y);
			R->inf = 0;
			return 1;
		}
		/* u = (1 + y) / (1 - y), v = u / x (Montgomery), then / B */
		fe_sub(F, &t, &one, &P->y);
		if (!fe_inv(F, &t, &t))
			return 0;
		fe_add(F, &u, &one, &P->y);
		fe_mul(F, &u, &u, &t);
		if (!fe_div(F, &v, &u, &P->x))
			return 0;
		fe_mul(F, &R->x, &u, &Bi);
		fe_mul(F, &R->y, &v, &Bi);
		R->inf = 0;
		return 1;
	case ECM_HESSIAN: {
		fe c[10], X[3], P0[3];
		struct ec_flexmap fm;
		struct ec E2;
		hess_cubic(M, c);
		fe_zero(F, &P0[0]);
		fe_one(F, &P0[1]);
		fe_neg(F, &P0[1], &P0[1]);
		fe_one(F, &P0[2]);
		if (!ec_cub_flex_to_w(F, c, P0, &E2, &fm))
			return 0;
		X[0] = P->x;
		X[1] = P->y;
		if (P->inf)
			fe_zero(F, &X[2]);
		else
			fe_one(F, &X[2]);
		return ec_flexmap_fwd(&E2, &fm, X, R);
	}
	}
	return 0;
}

int
ecm_pt_from_w(const struct ecmodel *M, const struct ec *E,
    const struct ecpt *P, struct ecmpt *R)
{
	const struct ecf *F = M->F;
	fe A, B, x, y, t, one;

	memset(R, 0, sizeof *R);
	switch (M->kind) {
	case ECM_WEIERSTRASS:
	case ECM_SHORT:
	case ECM_LEGENDRE:
		R->x = P->x;
		R->y = P->y;
		R->inf = P->inf;
		return 1;
	case ECM_MONTGOMERY:
		if (P->inf) {
			R->inf = 1;
			return 1;
		}
		fe_mul(F, &R->x, &P->x, &M->c[1]);
		fe_mul(F, &R->y, &P->y, &M->c[1]);
		return 1;
	case ECM_EDWARDS:
	case ECM_TWISTED_EDWARDS:
		fe_one(F, &one);
		if (P->inf) {
			fe_zero(F, &R->x);
			R->y = one;
			return 1;
		}
		if (!ted_to_mont(M, &A, &B))
			return 0;
		/* Montgomery (x, y) = B (u, v) */
		fe_mul(F, &x, &P->x, &B);
		fe_mul(F, &y, &P->y, &B);
		if (fe_is_zero(F, &x) && fe_is_zero(F, &y)) {
			fe_zero(F, &R->x);
			fe_neg(F, &R->y, &one);
			return 1;
		}
		/* X = x / y, Y = (x - 1) / (x + 1); none when y = 0 or
		 * x = -1 (points at infinity of the Edwards curve) */
		if (!fe_div(F, &R->x, &x, &y))
			return 0;
		fe_add(F, &t, &x, &one);
		if (!fe_inv(F, &t, &t))
			return 0;
		fe_sub(F, &R->y, &x, &one);
		fe_mul(F, &R->y, &R->y, &t);
		return 1;
	case ECM_HESSIAN: {
		fe c[10], X[3], P0[3];
		struct ec_flexmap fm;
		struct ec E2;
		hess_cubic(M, c);
		fe_zero(F, &P0[0]);
		fe_one(F, &P0[1]);
		fe_neg(F, &P0[1], &P0[1]);
		fe_one(F, &P0[2]);
		if (!ec_cub_flex_to_w(F, c, P0, &E2, &fm))
			return 0;
		ec_flexmap_back(&E2, &fm, P, X);
		if (fe_is_zero(F, &X[2])) {
			/* at infinity: (X : Y : 0), scaled so X = 1 when X != 0 */
			R->inf = 1;
			if (!fe_is_zero(F, &X[0])) {
				fe_div(F, &R->y, &X[1], &X[0]);
				fe_one(F, &R->x);
			} else {
				fe_zero(F, &R->x);
				fe_one(F, &R->y);
			}
			return 1;
		}
		fe_div(F, &R->x, &X[0], &X[2]);
		fe_div(F, &R->y, &X[1], &X[2]);
		return 1;
	}
	}
	return 0;
}

int
ecm_add(const struct ecmodel *M, struct ecmpt *R, const struct ecmpt *P,
    const struct ecmpt *Q)
{
	struct ec E;
	struct ecpt a, b, c;

	if (!ecm_to_weierstrass(M, &E) || !ecm_pt_to_w(M, &E, P, &a) ||
	    !ecm_pt_to_w(M, &E, Q, &b))
		return 0;
	if (!ec_add(&E, &c, &a, &b))
		return 0;
	return ecm_pt_from_w(M, &E, &c, R);
}

/* the roots in F of the cubic x^3 + a x^2 + b x + c */
static int
fe_cubic_roots(const struct ecf *F, const fe *a, const fe *b, const fe *c,
    fe *roots)
{
	int n = 0;

	if (F->kind == ECF_Q) {
		struct zq r[3];
		int i;
		n = zq_cubic_roots(zq_int(Z_ONE), fe_q(a), fe_q(b), fe_q(c), r);
		for (i = 0; i < n; i++)
			roots[i] = fe_from_q(r[i]);
		return n;
	}
	{
		/* small fields: try every element */
		u64 q, v;
		fe x, y;
		if (F->kind == ECF_FQ)
			q = F->R->qm1 + 1;
		else if (z_fits_i64(F->p) && z_to_i64(F->p) < (1 << 20))
			q = (u64)z_to_i64(F->p);
		else
			return 0;
		if (q > (1u << 20))
			return 0;
		for (v = 0; v < q && n < 3; v++) {
			if (F->kind == ECF_FQ) {
				memset(&x, 0, sizeof x);
				x.w[0] = v;
			} else
				fe_set_i64(F, &x, (int64_t)v);
			fe_add(F, &y, &x, a);
			fe_mul(F, &y, &y, &x);
			fe_add(F, &y, &y, b);
			fe_mul(F, &y, &y, &x);
			fe_add(F, &y, &y, c);
			if (fe_is_zero(F, &y))
				roots[n++] = x;
		}
	}
	return n;
}

int
ecm_from_weierstrass(int kind, const struct ec *E, struct ecmodel *M,
    struct eciso *iso)
{
	const struct ecf *F = E->F;
	struct ec W, Wm;
	struct eciso m1;
	fe two, r[3], t;
	int n, i, j;

	memset(M, 0, sizeof *M);
	M->kind = kind;
	M->F = F;
	if (E->singular)
		return 0;
	if (kind == ECM_WEIERSTRASS) {
		ec_coeffs(E, M->c);
		eciso_identity(F, iso);
		return 1;
	}
	if (kind == ECM_SHORT) {
		struct ec S;
		fe six;
		fe_set_i64(F, &six, 6);
		if (fe_is_zero(F, &six))
			return 0;
		eciso_to_short(E, iso, &S);
		M->c[0] = S.a4;
		M->c[1] = S.a6;
		return 1;
	}
	if (kind == ECM_HESSIAN)
		return 0;   /* needs a rational 3-torsion point: not done */
	fe_set_i64(F, &two, 2);
	if (fe_is_zero(F, &two))
		return 0;
	/* complete the square: y -> y - (a1 x + a3)/2 */
	eciso_identity(F, &m1);
	fe_inv(F, &t, &two);
	fe_mul(F, &m1.s, &E->a1, &t);
	fe_neg(F, &m1.s, &m1.s);
	fe_mul(F, &m1.t, &E->a3, &t);
	fe_neg(F, &m1.t, &m1.t);
	eciso_apply(E, &m1, &W);
	n = fe_cubic_roots(F, &W.a2, &W.a4, &W.a6, r);
	if (!n)
		return 0;
	for (i = 0; i < n; i++) {
		fe x0 = r[i], b, a, s;
		/* x -> x + x0: y^2 = x^3 + a x^2 + b x, a = 3 x0 + a2,
		 * b = 3 x0^2 + 2 a2 x0 + a4 */
		fe_mul_i64(F, &a, &x0, 3);
		fe_add(F, &a, &a, &W.a2);
		fe_mul(F, &b, &x0, &x0);
		fe_mul_i64(F, &b, &b, 3);
		fe_mul(F, &t, &W.a2, &x0);
		fe_mul_i64(F, &t, &t, 2);
		fe_add(F, &b, &b, &t);
		fe_add(F, &b, &b, &W.a4);
		if (kind == ECM_LEGENDRE) {
			/* y^2 = x (x - e1)(x - e2): need e1 a square (scale) */
			fe e[2];
			int k = 0;
			for (j = 0; j < n; j++)
				if (j != i)
					fe_sub(F, &e[k++], &r[j], &x0);
			if (k < 2)
				continue;
			for (j = 0; j < 2; j++) {
				fe sq, l;
				if (!fe_sqrt(F, &sq, &e[j]))
					continue;
				fe_div(F, &l, &e[1 - j], &e[j]);
				M->c[0] = l;
				if (!ecm_to_weierstrass(M, &Wm))
					continue;
				if (eciso_find(E, &Wm, iso))
					return 1;
			}
			continue;
		}
		if (!fe_sqrt(F, &s, &b) || fe_is_zero(F, &s))
			continue;
		/* Montgomery (1/s) Y^2 = X^3 + (a/s) X^2 + X */
		{
			fe A, B;
			fe_inv(F, &B, &s);
			fe_mul(F, &A, &a, &B);
			if (kind == ECM_MONTGOMERY) {
				M->c[0] = A;
				M->c[1] = B;
			} else {
				/* twisted Edwards a = (A + 2)/B, d = (A - 2)/B */
				fe ea, ed, Bi, sq;
				fe_inv(F, &Bi, &B);
				fe_add(F, &ea, &A, &two);
				fe_mul(F, &ea, &ea, &Bi);
				fe_sub(F, &ed, &A, &two);
				fe_mul(F, &ed, &ed, &Bi);
				if (fe_is_zero(F, &ea) || fe_is_zero(F, &ed) ||
				    fe_eq(F, &ea, &ed))
					continue;
				if (kind == ECM_EDWARDS) {
					/* x -> x sqrt(a): d / a */
					if (!fe_sqrt(F, &sq, &ea))
						continue;
					fe_div(F, &M->c[0], &ed, &ea);
				} else {
					M->c[0] = ea;
					M->c[1] = ed;
				}
			}
		}
		if (!ecm_to_weierstrass(M, &Wm))
			continue;
		if (eciso_find(E, &Wm, iso))
			return 1;
	}
	return 0;
}

/* ec_command lives in ec_cmd.c */
