/*
 * Numbers for the calc files: exact rationals (int64, overflow checked),
 * small dense linear algebra, polynomial roots (exact rational roots and
 * quadratic surds first, Durand-Kerner for what is left), Faulhaber
 * polynomials, and polynomials with rational coefficients as nodes in
 * factored form.
 */
#include "calc.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- rationals ---- */

int64_t
gcd64(int64_t a, int64_t b)
{
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	while (b) {
		int64_t t = a % b;
		a = b;
		b = t;
	}
	return a;
}

int
q_make(int64_t n, int64_t d, struct q *r)
{
	int64_t g;

	if (d == 0 || n == INT64_MIN || d == INT64_MIN)
		return 0;
	if (d < 0)
		n = -n, d = -d;
	g = gcd64(n, d);
	if (g > 1)
		n /= g, d /= g;
	if (n == 0)
		d = 1;
	r->n = n;
	r->d = d;
	return 1;
}

int
q_mul(struct q a, struct q b, struct q *r)
{
	int64_t g1 = gcd64(a.n, b.d), g2 = gcd64(b.n, a.d), n, d;

	if (g1 > 1)
		a.n /= g1, b.d /= g1;
	if (g2 > 1)
		b.n /= g2, a.d /= g2;
	if (__builtin_mul_overflow(a.n, b.n, &n) ||
	    __builtin_mul_overflow(a.d, b.d, &d))
		return 0;
	return q_make(n, d, r);
}

int
q_div(struct q a, struct q b, struct q *r)
{
	struct q inv;

	if (b.n == 0)
		return 0;
	inv.n = b.d;
	inv.d = b.n;
	if (inv.d < 0)
		inv.n = -inv.n, inv.d = -inv.d;
	return q_mul(a, inv, r);
}

int
q_add(struct q a, struct q b, struct q *r)
{
	int64_t g = gcd64(a.d, b.d), x, y, n, d;

	if (__builtin_mul_overflow(a.n, b.d / g, &x) ||
	    __builtin_mul_overflow(b.n, a.d / g, &y) ||
	    __builtin_add_overflow(x, y, &n) ||
	    __builtin_mul_overflow(a.d / g, b.d, &d))
		return 0;
	return q_make(n, d, r);
}

int
q_sub(struct q a, struct q b, struct q *r)
{
	if (b.n == INT64_MIN)
		return 0;
	b.n = -b.n;
	return q_add(a, b, r);
}

double
q_val(struct q a)
{
	return (double)a.n / (double)a.d;
}

/* the simplest fraction within 1e-12 (relative) of v, denominator <= 1e6 */
int
q_from_double(double v, struct q *r)
{
	double x = fabs(v), h0 = 0, h1 = 1, k0 = 1, k1 = 0, a;
	int i;

	if (!isfinite(v) || fabs(v) > 9e15)
		return 0;
	if (v == floor(v))
		return q_make((int64_t)v, 1, r);
	for (i = 0; i < 40; i++) {
		double h, k;
		a = floor(x);
		h = a * h1 + h0;
		k = a * k1 + k0;
		if (k > 1e6)
			break;
		h0 = h1, h1 = h, k0 = k1, k1 = k;
		if (fabs(h1 / k1 - fabs(v)) <= 1e-12 * fabs(v))
			return q_make((int64_t)(v < 0 ? -h1 : h1), (int64_t)k1, r);
		if (x - a < 1e-300)
			break;
		x = 1 / (x - a);
	}
	return 0;
}

nid
q_node(struct q a)
{
	if (a.d == 1)
		return KI(a.n);
	return DIV(KI(a.n), KI(a.d));
}

/* ---- linear algebra ---- */

int
solve_lin(double *a, double *b, int n)
{
	int i, j, k, piv;
	double t, mx = 0;

	for (i = 0; i < n * n; i++)
		if (fabs(a[i]) > mx)
			mx = fabs(a[i]);
	for (k = 0; k < n; k++) {
		piv = k;
		for (i = k + 1; i < n; i++)
			if (fabs(a[i * n + k]) > fabs(a[piv * n + k]))
				piv = i;
		if (fabs(a[piv * n + k]) <= 1e-13 * (mx > 0 ? mx : 1))
			return 0;
		if (piv != k) {
			for (j = 0; j < n; j++) {
				t = a[k * n + j];
				a[k * n + j] = a[piv * n + j];
				a[piv * n + j] = t;
			}
			t = b[k];
			b[k] = b[piv];
			b[piv] = t;
		}
		for (i = k + 1; i < n; i++) {
			double f = a[i * n + k] / a[k * n + k];
			for (j = k; j < n; j++)
				a[i * n + j] -= f * a[k * n + j];
			b[i] -= f * b[k];
		}
	}
	for (i = n - 1; i >= 0; i--) {
		t = b[i];
		for (j = i + 1; j < n; j++)
			t -= a[i * n + j] * b[j];
		b[i] = t / a[i * n + i];
	}
	return 1;
}

/* least squares by Householder QR: a is m x n (m >= n), row major */
int
solve_lsq(const double *a0, const double *b0, int m, int n, double *x)
{
	double *a = xmalloc((size_t)m * n * sizeof *a);
	double *b = xmalloc((size_t)m * sizeof *b), *v;
	int i, j, k, ok = 1;

	memcpy(a, a0, (size_t)m * n * sizeof *a);
	memcpy(b, b0, (size_t)m * sizeof *b);
	v = xmalloc((size_t)m * sizeof *v);
	for (k = 0; k < n; k++) {
		double nrm = 0, s, vv = 0;
		for (i = k; i < m; i++)
			nrm += a[i * n + k] * a[i * n + k];
		nrm = sqrt(nrm);
		if (nrm == 0) {
			ok = 0;
			break;
		}
		s = a[k * n + k] > 0 ? -nrm : nrm;
		for (i = k; i < m; i++)
			v[i] = a[i * n + k];
		v[k] -= s;
		for (i = k; i < m; i++)
			vv += v[i] * v[i];
		if (vv == 0)
			continue;
		for (j = k; j < n; j++) {
			double d = 0;
			for (i = k; i < m; i++)
				d += v[i] * a[i * n + j];
			d = 2 * d / vv;
			for (i = k; i < m; i++)
				a[i * n + j] -= d * v[i];
		}
		{
			double d = 0;
			for (i = k; i < m; i++)
				d += v[i] * b[i];
			d = 2 * d / vv;
			for (i = k; i < m; i++)
				b[i] -= d * v[i];
		}
	}
	if (ok) {
		double mx = 0;
		for (k = 0; k < n; k++)
			if (fabs(a[k * n + k]) > mx)
				mx = fabs(a[k * n + k]);
		for (i = n - 1; i >= 0; i--) {
			double t = b[i];
			if (fabs(a[i * n + i]) <= 1e-12 * mx) {
				ok = 0;
				break;
			}
			for (j = i + 1; j < n; j++)
				t -= a[i * n + j] * x[j];
			x[i] = t / a[i * n + i];
		}
	}
	free(a);
	free(b);
	free(v);
	return ok;
}

int
solve_q(struct q *a, struct q *b, int n)
{
	int i, j, k, piv;
	struct q t, f, u;

	for (k = 0; k < n; k++) {
		for (piv = k; piv < n && a[piv * n + k].n == 0; piv++)
			;
		if (piv == n)
			return 0;
		if (piv != k) {
			for (j = 0; j < n; j++) {
				t = a[k * n + j];
				a[k * n + j] = a[piv * n + j];
				a[piv * n + j] = t;
			}
			t = b[k];
			b[k] = b[piv];
			b[piv] = t;
		}
		for (i = 0; i < n; i++) {
			if (i == k || a[i * n + k].n == 0)
				continue;
			if (!q_div(a[i * n + k], a[k * n + k], &f))
				return 0;
			for (j = k; j < n; j++) {
				if (!q_mul(f, a[k * n + j], &u) ||
				    !q_sub(a[i * n + j], u, &a[i * n + j]))
					return 0;
			}
			if (!q_mul(f, b[k], &u) || !q_sub(b[i], u, &b[i]))
				return 0;
		}
	}
	for (i = 0; i < n; i++)
		if (!q_div(b[i], a[i * n + i], &b[i]))
			return 0;
	return 1;
}

/* ---- polynomial roots ---- */

/* A = (q x - p) B exactly? A has degree r, B gets degree r - 1 */
static int
deflate(const int64_t *A, int r, int64_t p, int64_t q, int64_t *B)
{
	int j;
	int64_t t;

	if (A[r] % q)
		return 0;
	B[r - 1] = A[r] / q;
	for (j = r - 1; j >= 1; j--) {
		if (__builtin_mul_overflow(p, B[j], &t) ||
		    __builtin_add_overflow(A[j], t, &t) || t % q)
			return 0;
		B[j - 1] = t / q;
	}
	if (__builtin_mul_overflow(-p, B[0], &t) || t != A[0])
		return 0;
	return 1;
}

static int
divisors(int64_t v, int64_t *out, int max)
{
	int64_t i;
	int n = 0;

	if (v < 0)
		v = -v;
	if (v == 0)
		return 0;
	for (i = 1; i <= v && i <= 1000000 && n < max; i++)
		if (v % i == 0)
			out[n++] = i;
	return n;
}

/*
 * Rational roots of the integer polynomial A (degree *r), removing them;
 * returns the number of distinct roots stored in rr/rm.
 */
static int
rational_roots(int64_t *A, int *r, struct q *rr, int *rm, int max)
{
	int64_t dp[256], dq[256], B[16];
	int np, nq, i, j, s, nr = 0;

	while (*r > 0 && A[0] == 0) {
		for (j = 0; j < *r; j++)
			A[j] = A[j + 1];
		(*r)--;
		if (nr && rr[nr - 1].n == 0)
			rm[nr - 1]++;
		else if (nr < max) {
			rr[nr].n = 0;
			rr[nr].d = 1;
			rm[nr++] = 1;
		}
	}
	if (*r <= 0)
		return nr;
	np = divisors(A[0], dp, 256);
	nq = divisors(A[*r], dq, 256);
	for (i = 0; i < np && *r > 0; i++)
		for (j = 0; j < nq && *r > 0; j++)
			for (s = -1; s <= 1 && *r > 0; s += 2) {
				int64_t p = s * dp[i], q = dq[j];
				if (gcd64(p, q) != 1)
					continue;
				while (*r > 0 && deflate(A, *r, p, q, B)) {
					memcpy(A, B, (size_t)*r * sizeof *A);
					(*r)--;
					if (nr && rr[nr - 1].n == p && rr[nr - 1].d == q)
						rm[nr - 1]++;
					else if (nr < max) {
						rr[nr].n = p;
						rr[nr].d = q;
						rm[nr++] = 1;
					}
				}
			}
	return nr;
}

static int
dk_roots(const double *c, int deg, double complex *z)
{
	double complex a[16];
	int i, j, it;

	for (i = 0; i <= deg; i++)
		a[i] = c[i] / c[deg];
	for (i = 0; i < deg; i++)
		z[i] = cpow(0.4 + 0.9 * I, i);
	for (it = 0; it < 2000; it++) {
		double mv = 0;
		for (i = 0; i < deg; i++) {
			double complex num = 1, den = 1, w;
			num = a[deg];
			for (j = deg - 1; j >= 0; j--)
				num = num * z[i] + a[j];
			for (j = 0; j < deg; j++)
				if (j != i)
					den *= z[i] - z[j];
			if (cabs(den) == 0)
				den = 1e-12;
			w = num / den;
			z[i] -= w;
			if (cabs(w) > mv)
				mv = cabs(w);
		}
		if (mv < 1e-15)
			break;
	}
	return deg;
}

int
poly_roots(const double *c, int deg, struct root *r)
{
	int64_t A[16];
	int i, j, isint = 1, nr = 0, rd = deg, rm[16];
	struct q rr[16];

	while (deg > 0 && c[deg] == 0)
		deg--;
	if (deg <= 0 || deg > 14)
		return 0;
	for (i = 0; i <= deg; i++) {
		if (c[i] != floor(c[i]) || fabs(c[i]) > 1e15)
			isint = 0;
		A[i] = (int64_t)c[i];
	}
	rd = deg;
	if (isint) {
		int k = rational_roots(A, &rd, rr, rm, 16);
		for (i = 0; i < k; i++) {
			memset(&r[nr], 0, sizeof r[nr]);
			r[nr].re = q_val(rr[i]);
			r[nr].mult = rm[i];
			r[nr].exact = 1;
			r[nr].qr = rr[i];
			nr++;
		}
		if (rd == 2) {
			/* a x^2 + b x + c with an irrational or complex pair */
			int64_t a = A[2], b = A[1], cc = A[0], D;
			if (!__builtin_mul_overflow(b, b, &D) &&
			    !__builtin_sub_overflow(D, 4 * a * cc, &D)) {
				double sq = sqrt(fabs((double)D));
				memset(&r[nr], 0, sizeof r[nr]);
				r[nr].quad = 1;
				q_make(-b, 1, &r[nr].qb);
				q_make(D, 1, &r[nr].qd);
				q_make(2 * a, 1, &r[nr].qa);
				if (D > 0) {
					r[nr].re = (-(double)b + sq) / (2.0 * a);
					r[nr].mult = 1;
					r[nr + 1] = r[nr];
					r[nr + 1].re = (-(double)b - sq) / (2.0 * a);
					r[nr + 1].quad = -1;
					nr += 2;
				} else {
					r[nr].re = -(double)b / (2.0 * a);
					r[nr].im = fabs(sq / (2.0 * a));
					r[nr].mult = 1;
					nr++;
				}
				return nr;
			}
		}
		if (rd == 1) {
			memset(&r[nr], 0, sizeof r[nr]);
			q_make(-A[0], A[1], &r[nr].qr);
			r[nr].re = q_val(r[nr].qr);
			r[nr].exact = 1;
			r[nr].mult = 1;
			return nr + 1;
		}
		if (rd == 0)
			return nr;
	}
	{
		double cc[16];
		double complex z[16];
		int used[16] = {0};
		if (isint && rd < deg)
			for (i = 0; i <= rd; i++)
				cc[i] = (double)A[i];
		else {
			for (i = 0; i <= deg; i++)
				cc[i] = c[i];
			rd = deg;
		}
		dk_roots(cc, rd, z);
		for (i = 0; i < rd; i++) {
			double tol;
			if (used[i])
				continue;
			if (cimag(z[i]) < -1e-9 * (1 + cabs(z[i])))
				continue;   /* the conjugate is reported */
			tol = 1e-5 * (1 + cabs(z[i]));
			memset(&r[nr], 0, sizeof r[nr]);
			r[nr].mult = 1;
			{
				double complex s = z[i];
				int cnt = 1;
				for (j = i + 1; j < rd; j++)
					if (!used[j] && cabs(z[j] - z[i]) < tol) {
						used[j] = 1;
						s += z[j];
						cnt++;
					}
				s /= cnt;
				r[nr].mult = cnt;
				r[nr].re = creal(s);
				r[nr].im = fabs(cimag(s)) < 1e-9 * (1 + cabs(s)) ? 0 :
				    fabs(cimag(s));
			}
			if (r[nr].im == 0 && q_from_double(r[nr].re, &r[nr].qr) &&
			    r[nr].qr.d <= 64 && fabs(q_val(r[nr].qr) - r[nr].re) <
			    1e-9 * (1 + fabs(r[nr].re))) {
				r[nr].re = q_val(r[nr].qr);
				r[nr].exact = 1;
			}
			nr++;
		}
	}
	return nr;
}

/* ---- Faulhaber ---- */

static const int64_t bern[13][2] = {
	{1, 1}, {1, 2}, {1, 6}, {0, 1}, {-1, 30}, {0, 1}, {1, 42}, {0, 1},
	{-1, 30}, {0, 1}, {5, 66}, {0, 1}, {-691, 2730},
};

static int64_t
binom(int n, int k)
{
	int64_t r = 1;
	int i;

	for (i = 1; i <= k; i++)
		r = r * (n - k + i) / i;
	return r;
}

/* sum_{i=1}^{n} i^p = sum_j c[j] n^j, j = 0..p+1 */
int
faulhaber(int p, struct q *c)
{
	int j;
	struct q t, b;

	if (p < 0 || p > 11)
		return 0;
	for (j = 0; j <= p + 1; j++)
		c[j].n = 0, c[j].d = 1;
	for (j = 0; j <= p; j++) {
		q_make(bern[j][0], bern[j][1], &b);
		if (!q_mul(b, (struct q){binom(p + 1, j), 1}, &t) ||
		    !q_div(t, (struct q){p + 1, 1}, &t))
			return 0;
		c[p + 1 - j] = t;
	}
	return 1;
}

/* P(a v + b) for P with coefficients c[0..deg] */
int
qpoly_compose_lin(const struct q *c, int deg, struct q a, struct q b,
    struct q *out)
{
	struct q pw[16][16], t;
	int i, j, k;

	/* pw[i] = (a v + b)^i */
	for (i = 0; i <= deg; i++)
		for (j = 0; j <= deg; j++)
			pw[i][j] = (struct q){0, 1};
	pw[0][0] = (struct q){1, 1};
	for (i = 1; i <= deg; i++)
		for (j = 0; j <= i; j++) {
			struct q x = {0, 1}, y = {0, 1};
			if (j > 0 && !q_mul(pw[i - 1][j - 1], a, &x))
				return 0;
			if (!q_mul(pw[i - 1][j], b, &y))
				return 0;
			if (!q_add(x, y, &pw[i][j]))
				return 0;
		}
	for (k = 0; k <= deg; k++)
		out[k] = (struct q){0, 1};
	for (i = 0; i <= deg; i++)
		for (k = 0; k <= i; k++) {
			if (!q_mul(c[i], pw[i][k], &t) || !q_add(out[k], t, &out[k]))
				return 0;
		}
	return 1;
}

static nid
ipoly_node(const int64_t *A, int r, nid x)
{
	nid acc = 0, term;
	int j;

	for (j = r; j >= 0; j--) {
		int64_t a = A[j], m = a < 0 ? -a : a;
		if (a == 0)
			continue;
		term = j == 0 ? KI(m) : j == 1 ? x : POW(x, KI(j));
		if (j > 0 && m != 1)
			term = MUL(KI(m), term);
		if (acc == 0)
			acc = a < 0 ? mk_un(O_NEG, term) : term;
		else
			acc = a < 0 ? SUB(acc, term) : ADD(acc, term);
	}
	return acc ? acc : KI(0);
}

/* P(x) with rational coefficients as u * (factors) / v */
nid
qpoly_node(const struct q *c, int deg, nid x)
{
	int64_t A[16], D = 1, g = 0, u, v;
	struct q rr[16];
	int rm[16], r, j, k, nr;
	nid f = 0, fac;

	while (deg >= 0 && c[deg].n == 0)
		deg--;
	if (deg < 0)
		return KI(0);
	if (deg == 0)
		return q_node(c[0]);
	for (j = 0; j <= deg; j++) {
		int64_t l = c[j].d / gcd64(D, c[j].d);
		if (__builtin_mul_overflow(D, l, &D))
			return 0;
	}
	for (j = 0; j <= deg; j++) {
		if (__builtin_mul_overflow(c[j].n, D / c[j].d, &A[j]))
			return 0;
		g = gcd64(g, A[j]);
	}
	if (A[deg] < 0)
		g = -g;
	for (j = 0; j <= deg; j++)
		A[j] /= g;
	{
		struct q m;
		q_make(g, D, &m);
		u = m.n;
		v = m.d;
	}
	r = deg;
	nr = rational_roots(A, &r, rr, rm, 16);
	/* what is left is A[0..r]; a constant residual goes into u */
	if (r == 0) {
		if (__builtin_mul_overflow(u, A[0], &u))
			return 0;
		{
			struct q m;
			q_make(u, v, &m);
			u = m.n;
			v = m.d;
		}
	}
	for (k = 0; k < nr; k++) {
		int64_t p = rr[k].n, q = rr[k].d;
		if (p == 0)
			fac = x;
		else if (q == 1)
			fac = ADD(x, KI(-p));
		else if (p < 0)
			fac = ADD(MUL(KI(q), x), KI(-p));
		else
			fac = SUB(MUL(KI(q), x), KI(p));
		if (rm[k] > 1)
			fac = POW(fac, KI(rm[k]));
		f = f ? MUL(f, fac) : fac;
	}
	if (r > 0) {
		nid rest = ipoly_node(A, r, x);
		f = f ? MUL(f, rest) : rest;
	}
	if (f == 0)
		f = KI(1);
	if (u == -1)
		f = mk_un(O_NEG, f);
	else if (u != 1)
		f = MUL(KI(u), f);
	if (v != 1)
		f = DIV(f, KI(v));
	return f;
}
