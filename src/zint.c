/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * zint.c - arbitrary-precision integers in one arena, and rationals.
 *
 * A zint is a 64-bit handle (absalg.h): even handles hold a small value
 * inline (v = h >> 1, |v| < 2^61), odd handles index the arena, whose
 * entries are three columns (sign, limb count, limb offset) over one
 * array of 32-bit limbs, little endian, normalised (no leading zero
 * limb, and never a value that would fit inline).  Arithmetic works on
 * views of the operands (a small value is spread into two limbs on the
 * stack), computes into scratch arrays and interns the result at the end,
 * so a growing arena never moves memory an operation is reading.
 * Compound operations (gcd, powers, factoring) mark the arena and keep
 * only their result.
 */
#include "absalg.h"
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

/* ---- the arena ---- */

static struct {
	uint64_t n, cap;
	int8_t *sg;
	uint32_t *len;
	uint64_t *off;
	uint64_t nl, lcap;
	uint32_t *limb;
} Z;

#define SMALL_OK(v) ((v) > -Z_SMALL_LIM && (v) < Z_SMALL_LIM)

static zint
mk_small(int64_t v)
{
	return (zint)((uint64_t)v << 1);
}

static int64_t
sval(zint h)
{
	return h >> 1;
}

static uint64_t
zidx(zint h)
{
	return (uint64_t)h >> 1;
}

/* a read-only view of an operand's magnitude */
struct zv {
	int sg;
	uint32_t n;
	const uint32_t *d;
	uint32_t buf[2];
};

static void
view(zint h, struct zv *v)
{
	if (Z_IS_SMALL(h)) {
		int64_t x = sval(h);
		uint64_t m;
		v->sg = x < 0 ? -1 : x > 0;
		m = x < 0 ? (uint64_t)(-x) : (uint64_t)x;
		v->buf[0] = (uint32_t)m;
		v->buf[1] = (uint32_t)(m >> 32);
		v->n = v->buf[1] ? 2 : v->buf[0] ? 1 : 0;
		v->d = v->buf;
	} else {
		uint64_t i = zidx(h);
		v->sg = Z.sg[i];
		v->n = Z.len[i];
		v->d = Z.limb + Z.off[i];
	}
}

/* sign and magnitude (n limbs, may have leading zeros) -> handle */
static zint
intern(int sg, const uint32_t *d, uint64_t n)
{
	uint64_t i;

	while (n > 0 && d[n - 1] == 0)
		n--;
	if (n == 0)
		return 0;
	if (n <= 2) {
		uint64_t m = d[0] | (n == 2 ? (uint64_t)d[1] << 32 : 0);
		if (m < (uint64_t)Z_SMALL_LIM)
			return mk_small(sg < 0 ? -(int64_t)m : (int64_t)m);
	}
	if (Z.n == Z.cap) {
		Z.cap = Z.cap ? Z.cap * 2 : 1024;
		Z.sg = xrealloc(Z.sg, Z.cap);
		Z.len = xrealloc(Z.len, Z.cap * sizeof *Z.len);
		Z.off = xrealloc(Z.off, Z.cap * sizeof *Z.off);
	}
	if (Z.nl + n > Z.lcap) {
		while (Z.nl + n > Z.lcap)
			Z.lcap = Z.lcap ? Z.lcap * 2 : 8192;
		Z.limb = xrealloc(Z.limb, Z.lcap * sizeof *Z.limb);
	}
	if (n > UINT32_MAX)
		fail("integer too large");
	i = Z.n++;
	Z.sg[i] = (int8_t)(sg < 0 ? -1 : 1);
	Z.len[i] = (uint32_t)n;
	Z.off[i] = Z.nl;
	memcpy(Z.limb + Z.nl, d, n * sizeof *d);
	Z.nl += n;
	return (zint)(i * 2 + 1);
}

zmark
z_mark(void)
{
	zmark m;

	m.n = Z.n;
	m.limbs = Z.nl;
	return m;
}

void
z_release(zmark m)
{
	if (m.n <= Z.n) {
		Z.n = m.n;
		Z.nl = m.limbs;
	}
}

void
z_release_keep(zmark m, zint *keep, size_t n)
{
	static uint32_t *tmp;
	static uint64_t tcap;
	static uint64_t *toff;
	static size_t tocap;
	uint64_t tn = 0;
	size_t k;

	if (n > tocap) {
		tocap = n;
		toff = xrealloc(toff, tocap * sizeof *toff);
	}
	for (k = 0; k < n; k++) {
		uint64_t i;
		if (Z_IS_SMALL(keep[k]) || (i = zidx(keep[k])) < m.n)
			continue;
		if (tn + Z.len[i] > tcap) {
			while (tn + Z.len[i] > tcap)
				tcap = tcap ? tcap * 2 : 1024;
			tmp = xrealloc(tmp, tcap * sizeof *tmp);
		}
		toff[k] = tn;
		memcpy(tmp + tn, Z.limb + Z.off[i], Z.len[i] * sizeof *tmp);
		tn += Z.len[i];
	}
	/* the signs and lengths are read before the release overwrites */
	{
		static int8_t *ts;
		static uint32_t *tl;
		static size_t tscap;
		if (n > tscap) {
			tscap = n;
			ts = xrealloc(ts, tscap);
			tl = xrealloc(tl, tscap * sizeof *tl);
		}
		for (k = 0; k < n; k++) {
			uint64_t i;
			if (Z_IS_SMALL(keep[k]) || (i = zidx(keep[k])) < m.n)
				continue;
			ts[k] = Z.sg[i];
			tl[k] = Z.len[i];
		}
		z_release(m);
		for (k = 0; k < n; k++) {
			uint64_t i;
			if (Z_IS_SMALL(keep[k]) || (i = zidx(keep[k])) < m.n)
				continue;
			keep[k] = intern(ts[k], tmp + toff[k], tl[k]);
		}
	}
}

size_t
z_arena_bytes(void)
{
	return (size_t)(Z.cap * (1 + 4 + 8) + Z.lcap * 4);
}

void
z_arena_reset(void)
{
	Z.n = 0;
	Z.nl = 0;
}

/* ---- scratch ---- */

static uint32_t *
scratch(int k, uint64_t n)
{
	static uint32_t *s[6];
	static uint64_t cap[6];

	if (n + 2 > cap[k]) {
		cap[k] = (n + 2) * 2;
		s[k] = xrealloc(s[k], cap[k] * sizeof **s);
	}
	return s[k];
}

/* ---- naturals: little-endian 32-bit limbs ---- */

static int
nat_cmp(const uint32_t *a, uint32_t an, const uint32_t *b, uint32_t bn)
{
	uint32_t i;

	while (an && a[an - 1] == 0)
		an--;
	while (bn && b[bn - 1] == 0)
		bn--;
	if (an != bn)
		return an < bn ? -1 : 1;
	for (i = an; i-- > 0;)
		if (a[i] != b[i])
			return a[i] < b[i] ? -1 : 1;
	return 0;
}

static uint32_t
nat_add(uint32_t *r, const uint32_t *a, uint32_t an, const uint32_t *b,
    uint32_t bn)
{
	uint64_t c = 0;
	uint32_t i;

	if (an < bn) {
		const uint32_t *t = a;
		uint32_t tn = an;
		a = b;
		an = bn;
		b = t;
		bn = tn;
	}
	for (i = 0; i < an; i++) {
		c += (uint64_t)a[i] + (i < bn ? b[i] : 0);
		r[i] = (uint32_t)c;
		c >>= 32;
	}
	r[an] = (uint32_t)c;
	return an + 1;
}

/* a >= b */
static uint32_t
nat_sub(uint32_t *r, const uint32_t *a, uint32_t an, const uint32_t *b,
    uint32_t bn)
{
	int64_t c = 0;
	uint32_t i;

	for (i = 0; i < an; i++) {
		c += (int64_t)a[i] - (i < bn ? (int64_t)b[i] : 0);
		r[i] = (uint32_t)c;
		c = c < 0 ? -1 : 0;
	}
	return an;
}

static uint32_t
nat_mul(uint32_t *r, const uint32_t *a, uint32_t an, const uint32_t *b,
    uint32_t bn)
{
	uint32_t i, j;

	memset(r, 0, (size_t)(an + bn) * sizeof *r);
	for (i = 0; i < an; i++) {
		uint64_t c = 0, ai = a[i];
		if (ai == 0)
			continue;
		for (j = 0; j < bn; j++) {
			c += ai * b[j] + r[i + j];
			r[i + j] = (uint32_t)c;
			c >>= 32;
		}
		r[i + bn] = (uint32_t)c;
	}
	return an + bn;
}

static int
nlz32(uint32_t x)
{
	int n = 0;

	if (x == 0)
		return 32;
	while (!(x & 0x80000000u)) {
		x <<= 1;
		n++;
	}
	return n;
}

/* q = u / d, returns the remainder; q may be u */
static uint32_t
nat_divsmall(uint32_t *q, const uint32_t *u, uint32_t un, uint32_t d)
{
	uint64_t r = 0;
	uint32_t i;

	for (i = un; i-- > 0;) {
		uint64_t t = (r << 32) | u[i];
		q[i] = (uint32_t)(t / d);
		r = t % d;
	}
	return (uint32_t)r;
}

/*
 * Knuth's algorithm D (Hacker's Delight divmnu): u (m limbs) / v (n
 * limbs, v[n-1] != 0, m >= n >= 2); q gets m - n + 1 limbs, r n limbs.
 */
static void
nat_divmod_knuth(uint32_t *q, uint32_t *r, const uint32_t *u, uint32_t m,
    const uint32_t *v, uint32_t n)
{
	const uint64_t b = (uint64_t)1 << 32;
	uint32_t *un = scratch(4, m + 1), *vn = scratch(5, n);
	int s = nlz32(v[n - 1]);
	int64_t t, k;
	uint64_t qhat, rhat, p;
	uint32_t i;
	int64_t j;

	for (i = n - 1; i > 0; i--)
		vn[i] = (uint32_t)(((uint64_t)v[i] << s) |
		    ((uint64_t)v[i - 1] >> (32 - s)));
	vn[0] = (uint32_t)((uint64_t)v[0] << s);
	un[m] = (uint32_t)((uint64_t)u[m - 1] >> (32 - s));
	for (i = m - 1; i > 0; i--)
		un[i] = (uint32_t)(((uint64_t)u[i] << s) |
		    ((uint64_t)u[i - 1] >> (32 - s)));
	un[0] = (uint32_t)((uint64_t)u[0] << s);
	for (j = (int64_t)m - n; j >= 0; j--) {
		uint64_t num = ((uint64_t)un[j + n] << 32) | un[j + n - 1];
		qhat = num / vn[n - 1];
		rhat = num % vn[n - 1];
		while (qhat >= b || qhat * vn[n - 2] > ((rhat << 32) |
		    un[j + n - 2])) {
			qhat--;
			rhat += vn[n - 1];
			if (rhat >= b)
				break;
		}
		k = 0;
		for (i = 0; i < n; i++) {
			p = qhat * vn[i];
			t = (int64_t)un[i + j] - k - (int64_t)(p & 0xffffffffu);
			un[i + j] = (uint32_t)t;
			k = (int64_t)(p >> 32) - (t >> 32);
		}
		t = (int64_t)un[j + n] - k;
		un[j + n] = (uint32_t)t;
		q[j] = (uint32_t)qhat;
		if (t < 0) {
			uint64_t c = 0;
			q[j]--;
			for (i = 0; i < n; i++) {
				c += (uint64_t)un[i + j] + vn[i];
				un[i + j] = (uint32_t)c;
				c >>= 32;
			}
			un[j + n] = (uint32_t)((uint64_t)un[j + n] + c);
		}
	}
	for (i = 0; i + 1 < n; i++)
		r[i] = (uint32_t)(((uint64_t)un[i] >> s) |
		    ((uint64_t)un[i + 1] << (32 - s)));
	r[n - 1] = (uint32_t)((uint64_t)un[n - 1] >> s);
}

/* ---- conversions ---- */

zint
z_from_i64(int64_t v)
{
	if (SMALL_OK(v))
		return mk_small(v);
	{
		uint64_t m = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
		uint32_t d[2];
		d[0] = (uint32_t)m;
		d[1] = (uint32_t)(m >> 32);
		return intern(v < 0 ? -1 : 1, d, 2);
	}
}

zint
z_from_u64(uint64_t v)
{
	uint32_t d[2];

	if (v < (uint64_t)Z_SMALL_LIM)
		return mk_small((int64_t)v);
	d[0] = (uint32_t)v;
	d[1] = (uint32_t)(v >> 32);
	return intern(1, d, 2);
}

zint
z_from_i128(i128 v)
{
	u128 m;
	uint32_t d[4];
	int i;

	if (v > -(i128)Z_SMALL_LIM && v < (i128)Z_SMALL_LIM)
		return mk_small((int64_t)v);
	m = v < 0 ? (u128)0 - (u128)v : (u128)v;
	for (i = 0; i < 4; i++) {
		d[i] = (uint32_t)m;
		m >>= 32;
	}
	return intern(v < 0 ? -1 : 1, d, 4);
}

static int
mag_u64(const struct zv *v, uint64_t *m)
{
	uint32_t n = v->n;

	while (n && v->d[n - 1] == 0)
		n--;
	if (n > 2)
		return 0;
	*m = (n > 0 ? v->d[0] : 0) | (n > 1 ? (uint64_t)v->d[1] << 32 : 0);
	return 1;
}

int
z_fits_i64(zint a)
{
	struct zv v;
	uint64_t m;

	if (Z_IS_SMALL(a))
		return 1;
	view(a, &v);
	if (!mag_u64(&v, &m))
		return 0;
	return v.sg < 0 ? m <= (uint64_t)1 << 63 : m < (uint64_t)1 << 63;
}

int64_t
z_to_i64(zint a)
{
	struct zv v;
	uint64_t m;

	if (Z_IS_SMALL(a))
		return sval(a);
	if (!z_fits_i64(a))
		fail("integer %s does not fit 64 bits", z_str(a));
	view(a, &v);
	mag_u64(&v, &m);
	return v.sg < 0 ? (int64_t)(0 - m) : (int64_t)m;
}

int
z_to_u64(zint a, u64 *out)
{
	struct zv v;
	uint64_t m;

	if (Z_IS_SMALL(a)) {
		if (sval(a) < 0)
			return 0;
		*out = (u64)sval(a);
		return 1;
	}
	view(a, &v);
	if (v.sg < 0 || !mag_u64(&v, &m))
		return 0;
	*out = m;
	return 1;
}

/* ---- comparison ---- */

int
z_sign(zint a)
{
	if (Z_IS_SMALL(a))
		return sval(a) < 0 ? -1 : sval(a) > 0;
	return Z.sg[zidx(a)];
}

int
z_cmp(zint a, zint b)
{
	struct zv x, y;
	int c;

	if (Z_IS_SMALL(a) && Z_IS_SMALL(b))
		return sval(a) < sval(b) ? -1 : sval(a) > sval(b);
	view(a, &x);
	view(b, &y);
	if (x.sg != y.sg)
		return x.sg < y.sg ? -1 : 1;
	c = nat_cmp(x.d, x.n, y.d, y.n);
	return x.sg < 0 ? -c : c;
}

int
z_cmpabs(zint a, zint b)
{
	struct zv x, y;

	view(a, &x);
	view(b, &y);
	return nat_cmp(x.d, x.n, y.d, y.n);
}

int
z_eq(zint a, zint b)
{
	if (Z_IS_SMALL(a) || Z_IS_SMALL(b))
		return a == b;
	return z_cmp(a, b) == 0;
}

int
z_is_zero(zint a)
{
	return a == 0;
}

int
z_is_one(zint a)
{
	return a == Z_ONE;
}

int
z_is_pm1(zint a)
{
	return a == Z_ONE || a == mk_small(-1);
}

/* ---- ring operations ---- */

static zint
addsub(zint a, zint b, int neg)
{
	struct zv x, y;
	int ys, c;
	uint32_t *r, rn;

	view(a, &x);
	view(b, &y);
	ys = neg ? -y.sg : y.sg;
	if (y.sg == 0)
		return a;
	if (x.sg == 0)
		return neg ? z_neg(b) : b;
	r = scratch(0, (x.n > y.n ? x.n : y.n) + 1);
	if (x.sg == ys) {
		rn = nat_add(r, x.d, x.n, y.d, y.n);
		return intern(x.sg, r, rn);
	}
	c = nat_cmp(x.d, x.n, y.d, y.n);
	if (c == 0)
		return 0;
	if (c > 0) {
		rn = nat_sub(r, x.d, x.n, y.d, y.n);
		return intern(x.sg, r, rn);
	}
	rn = nat_sub(r, y.d, y.n, x.d, x.n);
	return intern(ys, r, rn);
}

zint
z_add(zint a, zint b)
{
	if (Z_IS_SMALL(a) && Z_IS_SMALL(b))
		return z_from_i64(sval(a) + sval(b));
	return addsub(a, b, 0);
}

zint
z_sub(zint a, zint b)
{
	if (Z_IS_SMALL(a) && Z_IS_SMALL(b))
		return z_from_i64(sval(a) - sval(b));
	return addsub(a, b, 1);
}

zint
z_neg(zint a)
{
	struct zv x;

	if (Z_IS_SMALL(a))
		return mk_small(-sval(a));
	view(a, &x);
	{
		uint32_t *r = scratch(0, x.n);
		memcpy(r, x.d, x.n * sizeof *r);
		return intern(-x.sg, r, x.n);
	}
}

zint
z_abs(zint a)
{
	return z_sign(a) < 0 ? z_neg(a) : a;
}

zint
z_mul(zint a, zint b)
{
	struct zv x, y;
	uint32_t *r, rn;

	if (Z_IS_SMALL(a) && Z_IS_SMALL(b))
		return z_from_i128((i128)sval(a) * sval(b));
	view(a, &x);
	view(b, &y);
	if (x.sg == 0 || y.sg == 0)
		return 0;
	r = scratch(0, x.n + y.n);
	rn = nat_mul(r, x.d, x.n, y.d, y.n);
	return intern(x.sg * y.sg, r, rn);
}

zint
z_addmul(zint acc, zint a, zint b)
{
	if (Z_IS_SMALL(acc) && Z_IS_SMALL(a) && Z_IS_SMALL(b))
		return z_from_i128((i128)sval(acc) + (i128)sval(a) * sval(b));
	return z_add(acc, z_mul(a, b));
}

zint
z_submul(zint acc, zint a, zint b)
{
	if (Z_IS_SMALL(acc) && Z_IS_SMALL(a) && Z_IS_SMALL(b))
		return z_from_i128((i128)sval(acc) - (i128)sval(a) * sval(b));
	return z_sub(acc, z_mul(a, b));
}

void
z_divmod_trunc(zint a, zint b, zint *q, zint *r)
{
	struct zv x, y;
	uint32_t *qq, *rr;
	zint qv, rv;

	if (b == 0)
		fail("division by zero");
	if (Z_IS_SMALL(a) && Z_IS_SMALL(b)) {
		int64_t u = sval(a), v = sval(b);
		if (q)
			*q = z_from_i64(u / v);
		if (r)
			*r = mk_small(u % v);
		return;
	}
	view(a, &x);
	view(b, &y);
	if (nat_cmp(x.d, x.n, y.d, y.n) < 0) {
		if (q)
			*q = 0;
		if (r)
			*r = a;
		return;
	}
	{
		uint32_t xn = x.n, yn = y.n;
		while (yn && y.d[yn - 1] == 0)
			yn--;
		while (xn && x.d[xn - 1] == 0)
			xn--;
		qq = scratch(1, xn);
		rr = scratch(2, yn);
		if (yn == 1) {
			rr[0] = nat_divsmall(qq, x.d, xn, y.d[0]);
		} else
			nat_divmod_knuth(qq, rr, x.d, xn, y.d, yn);
		/* intern both from scratch before anything reuses it */
		qv = intern(x.sg * y.sg, qq, xn - yn + 1);
		rv = intern(x.sg, scratch(2, yn), yn);
	}
	if (q)
		*q = qv;
	if (r)
		*r = rv;
}

void
z_divmod_floor(zint a, zint b, zint *q, zint *r)
{
	zint qq, rr;

	z_divmod_trunc(a, b, &qq, &rr);
	if (rr != 0 && z_sign(rr) != z_sign(b)) {
		qq = z_sub(qq, Z_ONE);
		rr = z_add(rr, b);
	}
	if (q)
		*q = qq;
	if (r)
		*r = rr;
}

void
z_divmod_euclid(zint a, zint b, zint *q, zint *r)
{
	zint qq, rr;

	z_divmod_trunc(a, b, &qq, &rr);
	if (z_sign(rr) < 0) {
		if (z_sign(b) > 0) {
			qq = z_sub(qq, Z_ONE);
			rr = z_add(rr, b);
		} else {
			qq = z_add(qq, Z_ONE);
			rr = z_sub(rr, b);
		}
	}
	if (q)
		*q = qq;
	if (r)
		*r = rr;
}

zint
z_div_floor(zint a, zint b)
{
	zint q;

	z_divmod_floor(a, b, &q, NULL);
	return q;
}

zint
z_mod(zint a, zint b)
{
	zint r;

	if (Z_IS_SMALL(a) && Z_IS_SMALL(b) && b != 0) {
		int64_t u = sval(a), v = sval(b), w = u % v;
		if (w < 0)
			w += v < 0 ? -v : v;
		return mk_small(w);
	}
	z_divmod_euclid(a, b, NULL, &r);
	return r;
}

zint
z_divexact(zint a, zint b)
{
	zint q, r;

	z_divmod_trunc(a, b, &q, &r);
	if (r != 0)
		fail("internal: %s is not divisible by %s", z_str(a), z_str(b));
	return q;
}

int
z_divides(zint d, zint a)
{
	if (d == 0)
		return a == 0;
	return z_mod(a, d) == 0;
}

zint
z_div_round(zint a, zint b)
{
	zint q, r;
	int c;

	/* r = a - q b in [0, b) for b > 0, (b, 0] for b < 0; moving q up by
	 * one brings r into (-|b|/2, |b|/2] */
	z_divmod_floor(a, b, &q, &r);
	if (r == 0)
		return q;
	c = z_cmp(z_mul(z_abs(r), mk_small(2)), z_abs(b));
	if (c > 0 || (c == 0 && z_sign(b) < 0))
		q = z_add(q, Z_ONE);
	return q;
}

zint
z_gcd(zint a, zint b)
{
	zmark m;
	zint t;

	if (Z_IS_SMALL(a) && Z_IS_SMALL(b)) {
		uint64_t x = (uint64_t)(sval(a) < 0 ? -sval(a) : sval(a));
		uint64_t y = (uint64_t)(sval(b) < 0 ? -sval(b) : sval(b));
		while (y) {
			uint64_t r = x % y;
			x = y;
			y = r;
		}
		return mk_small((int64_t)x);
	}
	m = z_mark();
	a = z_abs(a);
	b = z_abs(b);
	while (b != 0) {
		if (Z_IS_SMALL(a) && Z_IS_SMALL(b)) {
			a = z_gcd(a, b);
			break;
		}
		t = z_mod(a, b);
		a = b;
		b = t;
	}
	z_release_keep(m, &a, 1);
	return a;
}

zint
z_xgcd(zint a, zint b, zint *s, zint *t)
{
	zmark m = z_mark();
	zint r0 = a, r1 = b, s0 = Z_ONE, s1 = 0, t0 = 0, t1 = Z_ONE, q, x;
	zint keep[3];

	while (r1 != 0) {
		z_divmod_floor(r0, r1, &q, &x);
		r0 = r1;
		r1 = x;
		x = z_submul(s0, q, s1);
		s0 = s1;
		s1 = x;
		x = z_submul(t0, q, t1);
		t0 = t1;
		t1 = x;
	}
	if (z_sign(r0) < 0) {
		r0 = z_neg(r0);
		s0 = z_neg(s0);
		t0 = z_neg(t0);
	}
	keep[0] = r0;
	keep[1] = s0;
	keep[2] = t0;
	z_release_keep(m, keep, 3);
	if (s)
		*s = keep[1];
	if (t)
		*t = keep[2];
	return keep[0];
}

zint
z_lcm(zint a, zint b)
{
	zint g;

	if (a == 0 || b == 0)
		return 0;
	g = z_gcd(a, b);
	return z_abs(z_mul(z_divexact(a, g), b));
}

zint
z_pow(zint a, uint64_t e)
{
	zmark m = z_mark();
	zint r = Z_ONE;

	while (e) {
		if (e & 1)
			r = z_mul(r, a);
		e >>= 1;
		if (e)
			a = z_mul(a, a);
	}
	z_release_keep(m, &r, 1);
	return r;
}

zint
z_powmod(zint a, zint e, zint mod)
{
	zmark m = z_mark();
	zint r = Z_ONE;
	size_t nb, i;
	struct zv v;

	if (z_sign(mod) <= 0 || z_sign(e) < 0)
		fail("powmod: needs e >= 0 and m > 0");
	a = z_mod(a, mod);
	nb = z_bits(e);
	for (i = nb; i-- > 0;) {
		r = z_mod(z_mul(r, r), mod);
		view(e, &v);
		if ((v.d[i / 32] >> (i % 32)) & 1)
			r = z_mod(z_mul(r, a), mod);
	}
	r = z_mod(r, mod);
	z_release_keep(m, &r, 1);
	return r;
}

zint
z_invmod(zint a, zint mod)
{
	zint s, g;

	g = z_xgcd(z_mod(a, mod), mod, &s, NULL);
	if (!z_is_one(g))
		return 0;
	return z_mod(s, mod);
}

int
z_val(zint a, zint p, zint *rest)
{
	int v = 0;
	zint q, r;

	if (a == 0)
		fail("valuation of 0");
	if (z_cmp(p, mk_small(2)) < 0)
		fail("valuation: p must be at least 2");
	for (;;) {
		z_divmod_trunc(a, p, &q, &r);
		if (r != 0)
			break;
		a = q;
		v++;
	}
	if (rest)
		*rest = a;
	return v;
}

u64
z_mod_u64(zint a, u64 m)
{
	struct zv v;
	u128 r = 0;
	uint32_t i;

	if (m == 0)
		fail("division by zero");
	view(a, &v);
	for (i = v.n; i-- > 0;)
		r = ((r << 32) | v.d[i]) % m;
	if (v.sg < 0 && r != 0)
		r = m - r;
	return (u64)r;
}

size_t
z_bits(zint a)
{
	struct zv v;
	uint32_t n;

	view(a, &v);
	n = v.n;
	while (n && v.d[n - 1] == 0)
		n--;
	if (n == 0)
		return 0;
	return (size_t)(n - 1) * 32 + (size_t)(32 - nlz32(v.d[n - 1]));
}

/* ---- primes and factoring ---- */

int
z_isprime(zint a)
{
	static const int bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
	    37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89};
	u64 u;
	zmark m;
	zint d, nm1, x;
	int s, i, k, ok = 1;

	if (z_to_u64(a, &u))
		return nt_isprime(u);
	if (z_sign(a) < 0)
		return 0;
	for (i = 0; i < (int)(sizeof bases / sizeof bases[0]); i++)
		if (z_mod_u64(a, (u64)bases[i]) == 0)
			return 0;
	m = z_mark();
	nm1 = z_sub(a, Z_ONE);
	s = z_val(nm1, mk_small(2), &d);
	for (i = 0; ok && i < (int)(sizeof bases / sizeof bases[0]); i++) {
		x = z_powmod(mk_small(bases[i]), d, a);
		if (z_is_one(x) || z_eq(x, nm1))
			continue;
		for (k = 1; k < s; k++) {
			x = z_mod(z_mul(x, x), a);
			if (z_eq(x, nm1))
				break;
		}
		if (k >= s)
			ok = 0;
	}
	z_release(m);
	return ok;
}

/* Pollard rho (Brent) on a composite n > 2^64; 0 if no factor found */
static zint
rho(zint n, int64_t c)
{
	zint x = mk_small(2), y = x, q = Z_ONE, g = Z_ONE, ys = x, cc;
	long r = 1, k, i, it = 0;
	const long lim = 1L << 21;

	cc = mk_small(c);
	while (z_is_one(g) && it < lim) {
		x = y;
		for (i = 0; i < r; i++)
			y = z_mod(z_add(z_mul(y, y), cc), n);
		k = 0;
		while (k < r && z_is_one(g)) {
			long lim2 = r - k < 64 ? r - k : 64;
			ys = y;
			for (i = 0; i < lim2; i++) {
				y = z_mod(z_add(z_mul(y, y), cc), n);
				q = z_mod(z_mul(q, z_abs(z_sub(x, y))), n);
			}
			g = z_gcd(q, n);
			k += lim2;
			it += lim2;
		}
		r *= 2;
	}
	if (z_eq(g, n)) {
		do {
			ys = z_mod(z_add(z_mul(ys, ys), cc), n);
			g = z_gcd(z_abs(z_sub(x, ys)), n);
		} while (z_is_one(g));
	}
	if (z_is_one(g) || z_eq(g, n))
		return 0;
	return g;
}

int
z_factor(zint a, zint *p, int *e, int max, int *complete)
{
	zint stack[128];
	int ns = 0, np = 0, i, j, comp = 1;
	static const int small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
	    37, 41, 43, 47};

	zmark mk;

	if (a == 0)
		fail("factor: 0");
	mk = z_mark();
	a = z_abs(a);
#define ADDP(q, k) do { \
		for (j = 0; j < np && !z_eq(p[j], (q)); j++) \
			; \
		if (j < np) \
			e[j] += (k); \
		else { \
			if (np >= max) \
				fail("factor: too many prime factors"); \
			p[np] = (q); \
			e[np++] = (k); \
		} \
	} while (0)
	for (i = 0; i < (int)(sizeof small / sizeof small[0]); i++) {
		int k = 0;
		while (z_mod_u64(a, (u64)small[i]) == 0) {
			a = z_divexact(a, mk_small(small[i]));
			k++;
		}
		if (k)
			ADDP(mk_small(small[i]), k);
	}
	if (!z_is_one(a))
		stack[ns++] = a;
	while (ns > 0) {
		zint n = stack[--ns], f = 0;
		u64 u;
		if (z_is_one(n))
			continue;
		if (z_to_u64(n, &u)) {
			struct nt_fac F;
			nt_factor(u, &F);
			for (i = 0; i < F.k; i++)
				ADDP(z_from_u64(F.p[i]), F.e[i]);
			continue;
		}
		if (z_isprime(n)) {
			ADDP(n, 1);
			continue;
		}
		for (i = 1; i < 6 && f == 0; i++)
			f = rho(n, i);
		if (f == 0 || ns + 2 > 128) {
			ADDP(n, 1);
			comp = 0;
			continue;
		}
		stack[ns++] = f;
		stack[ns++] = z_divexact(n, f);
	}
#undef ADDP
	/* sort by prime */
	for (i = 1; i < np; i++)
		for (j = i; j > 0 && z_cmp(p[j - 1], p[j]) > 0; j--) {
			zint t = p[j];
			int te = e[j];
			p[j] = p[j - 1];
			e[j] = e[j - 1];
			p[j - 1] = t;
			e[j - 1] = te;
		}
	z_release_keep(mk, p, (size_t)np);
	if (complete)
		*complete = comp;
	return np;
}

/* ---- text ---- */

int
z_parse(const char *s, zint *out, const char **end)
{
	int neg = 0, base = 10, any = 0;
	zint r = 0, bb;
	const char *s0 = s;

	while (*s == ' ' || *s == '\t')
		s++;
	if (*s == '-' || *s == '+') {
		neg = *s == '-';
		s++;
		while (*s == ' ')
			s++;
	}
	if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
		base = 16;
		s += 2;
	}
	bb = mk_small(base);
	/* chunks of 9 decimal digits (7 hex) at a time */
	for (;;) {
		int64_t chunk = 0, mul = 1;
		int k = 0;
		while (k < (base == 10 ? 9 : 7)) {
			int d;
			if (*s >= '0' && *s <= '9')
				d = *s - '0';
			else if (base == 16 && *s >= 'a' && *s <= 'f')
				d = *s - 'a' + 10;
			else if (base == 16 && *s >= 'A' && *s <= 'F')
				d = *s - 'A' + 10;
			else if (*s == '_' && any)
				{ s++; continue; }
			else
				break;
			chunk = chunk * base + d;
			mul *= base;
			s++;
			k++;
			any = 1;
		}
		if (k == 0)
			break;
		r = z_add(z_mul(r, mk_small(mul)), mk_small(chunk));
		if (k < (base == 10 ? 9 : 7))
			break;
	}
	(void)bb;
	if (!any) {
		if (end)
			*end = s0;
		return 0;
	}
	*out = neg ? z_neg(r) : r;
	if (end)
		*end = s;
	return 1;
}

void
z_print(struct sbuf *b, zint a)
{
	struct zv v;
	uint32_t *t, n, i;
	uint32_t *chunks;
	uint32_t nc = 0;

	if (Z_IS_SMALL(a)) {
		sb_printf(b, "%lld", (long long)sval(a));
		return;
	}
	view(a, &v);
	n = v.n;
	t = scratch(3, n);
	memcpy(t, v.d, n * sizeof *t);
	chunks = xmalloc(((size_t)n * 10 / 9 + 2) * sizeof *chunks);
	while (n > 0) {
		chunks[nc++] = nat_divsmall(t, t, n, 1000000000u);
		while (n > 0 && t[n - 1] == 0)
			n--;
	}
	if (v.sg < 0)
		sb_putc(b, '-');
	sb_printf(b, "%u", chunks[nc - 1]);
	for (i = nc - 1; i-- > 0;)
		sb_printf(b, "%09u", chunks[i]);
	free(chunks);
}

static char *
rot_buf(const char *s)
{
	static char *buf[8];
	static int k;
	char *r;

	k = (k + 1) & 7;
	free(buf[k]);
	r = buf[k] = xmalloc(strlen(s) + 1);
	strcpy(r, s);
	return r;
}

const char *
z_str(zint a)
{
	struct sbuf b = {0};
	const char *r;

	z_print(&b, a);
	r = rot_buf(sb_str(&b));
	free(b.p);
	return r;
}

/* ---- rationals ---- */

struct zq
zq_make(zint n, zint d)
{
	struct zq r;
	zint g;

	if (d == 0)
		fail("division by zero");
	if (z_sign(d) < 0) {
		n = z_neg(n);
		d = z_neg(d);
	}
	if (n == 0) {
		r.n = 0;
		r.d = Z_ONE;
		return r;
	}
	if (z_is_one(d)) {
		r.n = n;
		r.d = d;
		return r;
	}
	g = z_gcd(n, d);
	if (!z_is_one(g)) {
		n = z_divexact(n, g);
		d = z_divexact(d, g);
	}
	r.n = n;
	r.d = d;
	return r;
}

struct zq
zq_int(zint n)
{
	struct zq r;

	r.n = n;
	r.d = Z_ONE;
	return r;
}

struct zq
zq_add(struct zq a, struct zq b)
{
	if (z_is_one(a.d) && z_is_one(b.d))
		return zq_int(z_add(a.n, b.n));
	if (z_eq(a.d, b.d))
		return zq_make(z_add(a.n, b.n), a.d);
	return zq_make(z_add(z_mul(a.n, b.d), z_mul(b.n, a.d)),
	    z_mul(a.d, b.d));
}

struct zq
zq_neg(struct zq a)
{
	a.n = z_neg(a.n);
	return a;
}

struct zq
zq_sub(struct zq a, struct zq b)
{
	return zq_add(a, zq_neg(b));
}

struct zq
zq_mul(struct zq a, struct zq b)
{
	if (z_is_one(a.d) && z_is_one(b.d))
		return zq_int(z_mul(a.n, b.n));
	return zq_make(z_mul(a.n, b.n), z_mul(a.d, b.d));
}

struct zq
zq_inv(struct zq a)
{
	if (a.n == 0)
		fail("division by zero");
	return zq_make(a.d, a.n);
}

struct zq
zq_div(struct zq a, struct zq b)
{
	return zq_mul(a, zq_inv(b));
}

int
zq_cmp(struct zq a, struct zq b)
{
	return z_cmp(z_mul(a.n, b.d), z_mul(b.n, a.d));
}

int
zq_is_zero(struct zq a)
{
	return a.n == 0;
}

int
zq_eq(struct zq a, struct zq b)
{
	return z_eq(a.n, b.n) && z_eq(a.d, b.d);
}

int
zq_parse(const char *s, struct zq *out, const char **end)
{
	zint n, d = Z_ONE;
	const char *e;

	if (!z_parse(s, &n, &e))
		return 0;
	{
		const char *t = e;
		while (*t == ' ')
			t++;
		if (*t == '/') {
			zint dd;
			const char *e2;
			if (z_parse(t + 1, &dd, &e2)) {
				d = dd;
				e = e2;
			}
		}
	}
	if (d == 0)
		fail("division by zero");
	*out = zq_make(n, d);
	if (end)
		*end = e;
	return 1;
}

void
zq_print(struct sbuf *b, struct zq a)
{
	z_print(b, a.n);
	if (!z_is_one(a.d)) {
		sb_putc(b, '/');
		z_print(b, a.d);
	}
}

const char *
zq_str(struct zq a)
{
	struct sbuf b = {0};
	const char *r;

	zq_print(&b, a);
	r = rot_buf(sb_str(&b));
	free(b.p);
	return r;
}
