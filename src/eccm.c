/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * eccm.c - complex multiplication: reduced binary quadratic forms of
 * discriminant D < 0 (the class group), Hilbert class polynomials H_D
 * from j((-b + sqrt D) / 2a) evaluated in multiprecision fixed point
 * (floats where the mathematics is analytic: the coefficients are rounded
 * to integers with the rounding error reported and bounded), and CM
 * curves over F_p: a root of H_D mod p, the twist with the trace from
 * Cornacchia, certified by point orders.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- fixed point: value = sign * w / 2^(64 FR) ---- */

#define MPL 160
static int FR = 8;          /* fractional limbs, set per computation */
static int ML = 40;         /* live limbs */

typedef struct {
	int neg;
	u64 w[MPL];
} mpf;

typedef struct {
	mpf re, im;
} mpc;

static void
f_zero(mpf *a)
{
	memset(a, 0, sizeof *a);
}

static int
f_is_zero(const mpf *a)
{
	return l_len(a->w, ML) == 0;
}

static void
f_set_i(mpf *a, int64_t v)
{
	f_zero(a);
	a->neg = v < 0;
	a->w[FR] = v < 0 ? (u64)-v : (u64)v;
}

static void
f_addsub(mpf *r, const mpf *a, const mpf *b, int bneg)
{
	mpf t;

	f_zero(&t);
	if (a->neg == bneg) {
		if (l_add(t.w, a->w, b->w, ML))
			fail("cm: fixed-point overflow");
		t.neg = a->neg;
	} else if (l_cmp(a->w, b->w, ML) >= 0) {
		l_sub(t.w, a->w, b->w, ML);
		t.neg = a->neg;
	} else {
		l_sub(t.w, b->w, a->w, ML);
		t.neg = bneg;
	}
	if (f_is_zero(&t))
		t.neg = 0;
	*r = t;
}

static void
f_add(mpf *r, const mpf *a, const mpf *b)
{
	f_addsub(r, a, b, b->neg);
}

static void
f_sub(mpf *r, const mpf *a, const mpf *b)
{
	f_addsub(r, a, b, !b->neg && !f_is_zero(b));
}

static void
f_mul(mpf *r, const mpf *a, const mpf *b)
{
	u64 t[2 * MPL];
	int na = l_len(a->w, ML), nb = l_len(b->w, ML);
	mpf o;

	f_zero(&o);
	if (na && nb) {
		memset(t, 0, sizeof(u64) * (size_t)(na + nb + 1));
		l_mul(t, a->w, na, b->w, nb);
		if (l_len(t, na + nb) > ML + FR)
			fail("cm: fixed-point overflow (precision too low)");
		memcpy(o.w, t + FR, sizeof(u64) * (size_t)(na + nb > FR ?
		    (na + nb - FR < ML ? na + nb - FR : ML) : 0));
		o.neg = a->neg != b->neg && !f_is_zero(&o);
	}
	*r = o;
}

static void
f_div(mpf *r, const mpf *a, const mpf *b)
{
	u64 num[2 * MPL + 2], q[2 * MPL + 2];
	int na = l_len(a->w, ML), nb = l_len(b->w, ML);
	mpf o;

	if (nb == 0)
		fail("cm: division by zero");
	f_zero(&o);
	memset(num, 0, sizeof num);
	memset(q, 0, sizeof q);
	memcpy(num + FR, a->w, sizeof(u64) * (size_t)na);
	l_divmod(q, NULL, num, na + FR, b->w, nb);
	if (l_len(q, na + FR) > ML)
		fail("cm: fixed-point overflow in a division");
	memcpy(o.w, q, sizeof(u64) * (size_t)ML);
	o.neg = a->neg != b->neg && !f_is_zero(&o);
	*r = o;
}

static void
f_div_i(mpf *r, const mpf *a, int64_t d)
{
	mpf b;

	f_set_i(&b, d);
	f_div(r, a, &b);
}

static void
f_mul_i(mpf *r, const mpf *a, int64_t d)
{
	mpf b;

	f_set_i(&b, d);
	f_mul(r, a, &b);
}

/* 2^-k */
static void
f_shr(mpf *r, const mpf *a, int k)
{
	nn dummy;
	int ws = k / 64, bs = k % 64, i;
	mpf o;

	(void)dummy;
	f_zero(&o);
	for (i = 0; i + ws < ML; i++) {
		o.w[i] = a->w[i + ws] >> bs;
		if (bs && i + ws + 1 < ML)
			o.w[i] |= a->w[i + ws + 1] << (64 - bs);
	}
	o.neg = a->neg && !f_is_zero(&o);
	*r = o;
}

static void
f_sqrt(mpf *r, const mpf *a)
{
	/* Newton from a double estimate */
	mpf x, t, h;
	int i;
	double d = 0;

	for (i = ML - 1; i >= 0; i--)
		d = d * 18446744073709551616.0 + (double)a->w[i];
	d = ldexp(d, -64 * FR);
	d = sqrt(d);
	f_zero(&x);
	{
		/* x = d as fixed point (53 bits) */
		int e;
		double m = frexp(d, &e);
		u64 mm = (u64)ldexp(m, 53);
		f_set_i(&x, (int64_t)mm);
		if (e - 53 >= 0) {
			for (i = 0; i < e - 53; i++)
				f_mul_i(&x, &x, 2);
		} else
			f_shr(&x, &x, 53 - e);
	}
	for (i = 0; i < 12; i++) {
		f_div(&t, a, &x);
		f_add(&h, &x, &t);
		f_shr(&x, &h, 1);
	}
	*r = x;
}

static void
f_exp(mpf *r, const mpf *a)
{
	/* exp(a) = exp(a / 2^k)^(2^k), Taylor */
	mpf x, s, term;
	int k = 24, n;

	f_shr(&x, a, k);
	f_set_i(&s, 1);
	f_set_i(&term, 1);
	for (n = 1; n < 400; n++) {
		f_mul(&term, &term, &x);
		f_div_i(&term, &term, n);
		if (f_is_zero(&term))
			break;
		f_add(&s, &s, &term);
	}
	for (n = 0; n < k; n++)
		f_mul(&s, &s, &s);
	*r = s;
}

/* cos and sin of a (|a| <= 4) */
static void
f_cossin(mpf *c, mpf *s, const mpf *a)
{
	mpf x, cc, ss, term, x2, t1, t2;
	int k = 16, n;

	f_shr(&x, a, k);
	f_mul(&x2, &x, &x);
	/* Taylor for cos and sin of x */
	f_set_i(&cc, 1);
	f_set_i(&term, 1);
	for (n = 1; n < 300; n++) {
		f_mul(&term, &term, &x2);
		f_div_i(&term, &term, (2 * n - 1) * (2 * n));
		if (f_is_zero(&term))
			break;
		if (n & 1)
			f_sub(&cc, &cc, &term);
		else
			f_add(&cc, &cc, &term);
	}
	ss = x;
	term = x;
	for (n = 1; n < 300; n++) {
		f_mul(&term, &term, &x2);
		f_div_i(&term, &term, (2 * n) * (2 * n + 1));
		if (f_is_zero(&term))
			break;
		if (n & 1)
			f_sub(&ss, &ss, &term);
		else
			f_add(&ss, &ss, &term);
	}
	/* double the angle k times */
	for (n = 0; n < k; n++) {
		f_mul(&t1, &cc, &cc);
		f_mul(&t2, &ss, &ss);
		f_mul(&ss, &ss, &cc);
		f_mul_i(&ss, &ss, 2);
		f_sub(&cc, &t1, &t2);
	}
	*c = cc;
	*s = ss;
}

/* atan(1/m) for an integer m > 1 */
static void
f_atan_inv(mpf *r, int64_t m)
{
	mpf s, p, t;
	int64_t n;

	f_set_i(&p, 1);
	f_div_i(&p, &p, m);
	s = p;
	for (n = 1; n < 100000; n++) {
		f_div_i(&p, &p, m * m);
		f_div_i(&t, &p, 2 * n + 1);
		if (f_is_zero(&t))
			break;
		if (n & 1)
			f_sub(&s, &s, &t);
		else
			f_add(&s, &s, &t);
	}
	*r = s;
}

static void
f_pi(mpf *r)
{
	mpf a, b;

	f_atan_inv(&a, 5);
	f_atan_inv(&b, 239);
	f_mul_i(&a, &a, 16);
	f_mul_i(&b, &b, 4);
	f_sub(r, &a, &b);
}

static void
c_mul(mpc *r, const mpc *a, const mpc *b)
{
	mpf t1, t2, t3, t4;

	f_mul(&t1, &a->re, &b->re);
	f_mul(&t2, &a->im, &b->im);
	f_mul(&t3, &a->re, &b->im);
	f_mul(&t4, &a->im, &b->re);
	f_sub(&r->re, &t1, &t2);
	f_add(&r->im, &t3, &t4);
}

static void
c_div(mpc *r, const mpc *a, const mpc *b)
{
	mpf d, t1, t2, nr, ni;

	f_mul(&t1, &b->re, &b->re);
	f_mul(&t2, &b->im, &b->im);
	f_add(&d, &t1, &t2);
	f_mul(&t1, &a->re, &b->re);
	f_mul(&t2, &a->im, &b->im);
	f_add(&nr, &t1, &t2);
	f_mul(&t1, &a->im, &b->re);
	f_mul(&t2, &a->re, &b->im);
	f_sub(&ni, &t1, &t2);
	f_div(&r->re, &nr, &d);
	f_div(&r->im, &ni, &d);
}

/* ---- forms and class numbers ---- */

/* reduced forms (a, b, c), b^2 - 4ac = D: |b| <= a <= c, b >= 0 when
 * |b| = a or a = c; returns h (up to max stored) */
int
cm_forms(int64_t D, int64_t *fa, int64_t *fb, int max)
{
	int64_t a, b, c, aD = -D;
	int h = 0;

	if (D >= 0 || ((D % 4) + 4) % 4 > 1)
		return 0;
	for (a = 1; 3 * a * a <= aD; a++)
		for (b = -a + 1; b <= a; b++) {
			int64_t num = b * b - D;
			if (num % (4 * a))
				continue;
			c = num / (4 * a);
			if (c < a)
				continue;
			if (b < 0 && a == c)
				continue;
			/* primitive */
			{
				int64_t g = a, x = b < 0 ? -b : b, y = c, t;
				while (x) {
					t = g % x;
					g = x;
					x = t;
				}
				while (y) {
					t = g % y;
					g = y;
					y = t;
				}
				if (g != 1)
					continue;
			}
			if (h < max) {
				fa[h] = a;
				fb[h] = b;
			}
			h++;
		}
	return h;
}

/* j(tau) by E4^3 / Delta as q-series (|q| small) */
static void
j_of_q(mpc *j, const mpc *q, int nterms)
{
	mpc E4, D, qn, t, one, prod, f;
	int n, k;

	/* E4 = 1 + 240 sum sigma3(n) q^n; Delta = q prod (1 - q^n)^24 */
	f_set_i(&E4.re, 1);
	f_zero(&E4.im);
	f_set_i(&one.re, 1);
	f_zero(&one.im);
	prod = one;
	qn = one;
	for (n = 1; n <= nterms; n++) {
		int64_t s3 = 0;
		for (k = 1; k <= n; k++)
			if (n % k == 0)
				s3 += (int64_t)k * k * k;
		c_mul(&qn, &qn, q);
		f_mul_i(&t.re, &qn.re, 240 * s3);
		f_mul_i(&t.im, &qn.im, 240 * s3);
		f_add(&E4.re, &E4.re, &t.re);
		f_add(&E4.im, &E4.im, &t.im);
		/* (1 - q^n)^24 */
		f_sub(&f.re, &one.re, &qn.re);
		f_sub(&f.im, &one.im, &qn.im);
		for (k = 0; k < 24; k++)
			c_mul(&prod, &prod, &f);
	}
	c_mul(&D, &prod, q);
	c_mul(&t, &E4, &E4);
	c_mul(&t, &t, &E4);
	c_div(j, &t, &D);
}

/*
 * H_D(X) = prod over reduced forms (X - j((-b + sqrt D)/(2a))), integer
 * coefficients c[0..h] (c[h] = 1).  maxerr: the largest distance of a
 * computed coefficient from its integer, as -log2.
 */
int
cm_hilbert(int64_t D, zz *coef, int max, int *errbits)
{
	int64_t fa[256], fb[256];
	int h = cm_forms(D, fa, fb, 256), i, k, nterms, bits;
	double est = 0, sq = sqrt((double)-D);
	mpc *pc, *roots;
	mpf pi, sD;

	if (h <= 0 || h > max - 1 || h > 255)
		return 0;
	{
		double lmax = 0;
		for (i = 0; i < h; i++) {
			double l = M_PI * sq / (double)fa[i] / M_LN2 + 1;
			est += l;
			if (l > lmax)
				lmax = l;
		}
		bits = (int)est + h + 16;
		FR = (int)((2 * lmax + est + 128) / 64) + 1;
		ML = (bits + 63) / 64 + FR + 2;
	}
	if (ML + FR + 2 > MPL)
		fail("hilbert: D = %lld needs %d limbs of precision (at most %d "
		    "here)", (long long)D, ML + FR, MPL);
	pc = xcalloc((size_t)h + 2, sizeof *pc);
	roots = xcalloc((size_t)h + 1, sizeof *roots);
	f_pi(&pi);
	f_set_i(&sD, -D);
	f_sqrt(&sD, &sD);
	for (i = 0; i < h; i++) {
		/* q = exp(-pi sqrt|D| / a) (cos(pi b / a) - i sin(pi b / a)) */
		mpf r, ang, c, s;
		mpc q;
		double lq = M_PI * sq / (double)fa[i] / M_LN2;   /* -log2 |q| */
		f_mul(&r, &pi, &sD);
		f_div_i(&r, &r, fa[i]);
		r.neg = 1;
		f_exp(&r, &r);
		f_mul_i(&ang, &pi, fb[i]);
		f_div_i(&ang, &ang, fa[i]);
		f_cossin(&c, &s, &ang);
		f_mul(&q.re, &r, &c);
		f_mul(&q.im, &r, &s);
		q.im.neg = !q.im.neg && !f_is_zero(&q.im);
		nterms = (int)((64.0 * (FR + 2)) / lq) + 4;
		j_of_q(&roots[i], &q, nterms);
	}
	/* prod (X - j_i) */
	f_set_i(&pc[0].re, 1);
	for (i = 0; i < h; i++) {
		/* pc = pc * (X - r): new[k] = old[k-1] - r old[k] */
		for (k = i + 1; k >= 0; k--) {
			mpc a, b;
			if (k > 0)
				a = pc[k - 1];
			else {
				f_zero(&a.re);
				f_zero(&a.im);
			}
			if (k <= i)
				c_mul(&b, &pc[k], &roots[i]);
			else {
				f_zero(&b.re);
				f_zero(&b.im);
			}
			f_sub(&pc[k].re, &a.re, &b.re);
			f_sub(&pc[k].im, &a.im, &b.im);
		}
	}
	/* pc[k]: the coefficient of X^k; round to integers */
	*errbits = 100000;
	for (k = 0; k <= h; k++) {
		mpf v = pc[k].re, a, half, iz, fr;
		nn m;
		int eb;
		a = v;
		a.neg = 0;
		f_zero(&half);
		half.w[FR - 1] = 1ULL << 63;
		f_add(&a, &a, &half);
		if (l_len(a.w + FR, ML - FR) > NW)
			fail("hilbert: a coefficient exceeds %d bits", 64 * NW);
		nn_zero(&m);
		memcpy(m.w, a.w + FR, sizeof(u64) * (size_t)(ML - FR < NW ?
		    ML - FR : NW));
		zz_from_nn(&coef[k], &m, v.neg);
		f_zero(&iz);
		memcpy(iz.w + FR, m.w, sizeof(u64) * (size_t)(ML - FR < NW ?
		    ML - FR : NW));
		iz.neg = v.neg && !f_is_zero(&iz);
		f_sub(&fr, &v, &iz);
		eb = 64 * FR - l_bits(fr.w, ML);
		if (eb < *errbits)
			*errbits = eb;
		eb = 64 * FR - l_bits(pc[k].im.w, ML);
		if (eb < *errbits)
			*errbits = eb;
	}
	free(pc);
	free(roots);
	return h;
}

/* the class number by counting reduced forms */
int
cm_classno(int64_t D)
{
	int64_t a[1], b[1];

	return cm_forms(D, a, b, 0);
}

/* H_D mod p as a polynomial */
int
cm_hilbert_mod(const struct mf *F, int64_t D, struct px *H)
{
	zz c[260];
	int h, e, i;

	if (D == -3 || D == -4) {
		/* H_{-3} = X, H_{-4} = X - 1728 */
		px_set_x(F, H);
		if (D == -4) {
			mf_from_i64(F, px_coef(F, H, 0), -1728);
		}
		return 1;
	}
	h = cm_hilbert(D, c, 260, &e);
	if (h <= 0 || e < 16)
		return 0;
	px_init(H);
	px_reserve(F, H, h);
	for (i = 0; i <= h; i++)
		mf_from_zz(F, px_coef(F, H, i), &c[i]);
	H->d = h;
	px_norm(F, H);
	return h;
}

/* a curve with j and the given trace t (over F_p): y^2 = x^3 + a x + b
 * with j, or its twist; 0 if neither has p + 1 - t points (by test) */
int
cm_curve_for(const struct mf *F, const u64 *j, const zz *t,
    struct ecp_curve *E)
{
	u64 a[FW], b[FW], k[FW], c[FW];
	nn N, p1;
	zz P1, Nz;
	int tw, it;

	nn_add_u64(&p1, &F->P, 1);
	zz_from_nn(&P1, &p1, 0);
	zz_sub(&Nz, &P1, t);
	if (Nz.neg)
		return 0;
	N = Nz.m;
	if (mf_is_zero(F, j)) {
		mf_zero(F, a);
		mf_one(F, b);
	} else {
		mf_from_u64(F, c, 1728);
		if (mf_eq(F, j, c)) {
			mf_one(F, a);
			mf_zero(F, b);
		} else {
			mf_sub(F, c, c, j);
			mf_inv(F, c, c, NULL);
			mf_mul(F, k, j, c);
			mf_mul_u64(F, a, k, 3);
			mf_mul_u64(F, b, k, 2);
		}
	}
	/* j = 0 and 1728 have 6 and 4 twists: try twists by powers of a
	 * non-residue g (sextic/quartic act through g^2, g^3) */
	for (tw = 0; tw < 6; tw++) {
		struct ecp_pt P, R;
		int ok = 1;
		ecp_set(E, F, a, b);
		for (it = 0; it < 4 && ok; it++) {
			ecp_random_point(E, &P);
			ecp_mul(E, &R, &P, &N);
			ok = R.inf;
		}
		if (ok)
			return 1;
		/* next twist: (a, b) -> (a g^2, b g^3) for quadratic; for j = 0
		 * b -> b g, for j = 1728 a -> a g (a non-residue g) */
		{
			u64 g[FW], g2[FW], g3[FW];
			mf_copy(F, g, F->nonres);
			if (mf_is_zero(F, a))
				mf_mul(F, b, b, g);
			else if (mf_is_zero(F, b))
				mf_mul(F, a, a, g);
			else {
				mf_sqr(F, g2, g);
				mf_mul(F, g3, g2, g);
				mf_mul(F, a, a, g2);
				mf_mul(F, b, b, g3);
			}
		}
	}
	return 0;
}
