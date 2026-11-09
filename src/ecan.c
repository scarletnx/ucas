/*
 * ecan.c - the analytic side of curves over Q (floating point; every
 * value comes with the error bound it is stated with):
 *
 *   periods        AGM (Cohen, Algorithm 7.4.7), the lattice (w1, w2),
 *                  the real period Omega (times the components)
 *   Weierstrass p  q-expansion on the lattice; x(z), y(z) on E
 *   a_n            a_p by Legendre sums at good p, the local data at bad
 *                  p, Hecke recursion and multiplicativity
 *   L-values       L(E, 1) = (1 + w) sum a_n/n exp(-2 pi n / sqrt N),
 *                  the root number w from the functional equation (the
 *                  value must not depend on the splitting parameter),
 *                  L^(r)(E, 1) = 2 r! sum a_n/n G_r(2 pi n / sqrt N)
 *   rank, BSD      the analytic rank as the first nonvanishing derivative
 *                  (|value| above the error bound), the BSD quotient
 *                  #Sha_an = L^(r)(1)/r! #T^2 / (Omega R prod c_p)
 */
#include "ecq.h"
#include "ecan.h"
#include "ucas.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double complex
cagm(double complex a, double complex b)
{
	int i;

	for (i = 0; i < 100; i++) {
		double complex an = (a + b) / 2, bn = csqrt(a * b);
		/* the right choice of square root: |an - bn| <= |an + bn| */
		if (cabs(an - bn) > cabs(an + bn))
			bn = -bn;
		a = an;
		b = bn;
		if (cabs(a - b) <= 1e-16 * cabs(a))
			break;
	}
	return (a + b) / 2;
}

int
ecan_periods(const struct ec *E, struct ecan_per *P)
{
	double b2 = zq_to_double(fe_q(&E->b2)), b4 = zq_to_double(fe_q(&E->b4)),
	    b6 = zq_to_double(fe_q(&E->b6));
	long double r[3];
	int n;

	memset(P, 0, sizeof *P);
	n = ecq_real_roots(b2, b4, b6, r);
	if (z_sign(fe_q(&E->disc).n) > 0) {
		double e1 = (double)r[2], e2 = (double)r[1], e3 = (double)r[0];
		if (n != 3)
			fail("internal: periods: 3 real roots expected");
		P->w1 = M_PI / creal(cagm(sqrt(e1 - e3), sqrt(e1 - e2)));
		P->w2 = I * M_PI / creal(cagm(sqrt(e1 - e3), sqrt(e2 - e3)));
		P->ncomp = 2;
	} else {
		double e1 = (double)r[0];
		double a = 3 * e1 + b2 / 4;
		double b = sqrt(3 * e1 * e1 + b2 / 2 * e1 + b4 / 2);
		P->w1 = 2 * M_PI / creal(cagm(2 * sqrt(b), sqrt(2 * b + a)));
		P->w2 = -P->w1 / 2 + I * M_PI / creal(cagm(2 * sqrt(b),
		    sqrt(2 * b - a)));
		P->ncomp = 1;
	}
	P->omega = creal(P->w1) * P->ncomp;
	P->err = 1e-14 * P->omega;
	/* the lattice's tau in the upper half plane */
	{
		double complex tau = P->w2 / P->w1;
		if (cimag(tau) < 0)
			P->w2 = -P->w2;
	}
	return 1;
}

/* p(z) and p'(z) for the lattice (w1, w2) */
static void
wp_series(double complex w1, double complex w2, double complex z,
    double complex *wp, double complex *wpd)
{
	double complex tau = w2 / w1, q = cexp(2 * M_PI * I * tau),
	    u = cexp(2 * M_PI * I * z / w1), c = 2 * M_PI * I / w1;
	double complex s = 1.0 / 12, sd = 0, qn = 1;
	int n;

	s += u / ((1 - u) * (1 - u));
	sd += u * (1 + u) / ((1 - u) * (1 - u) * (1 - u));
	for (n = 1; n < 200; n++) {
		double complex a, b;
		qn *= q;
		if (cabs(qn) < 1e-20)
			break;
		a = qn * u;
		b = qn / u;
		s += a / ((1 - a) * (1 - a)) + b / ((1 - b) * (1 - b)) -
		    2 * qn / ((1 - qn) * (1 - qn));
		sd += a * (1 + a) / ((1 - a) * (1 - a) * (1 - a)) -
		    b * (1 + b) / ((1 - b) * (1 - b) * (1 - b));
	}
	*wp = c * c * s;
	*wpd = c * c * c * sd;
}

void
ecan_wp(const struct ec *E, const struct ecan_per *P, double complex z,
    double complex *x, double complex *y, double *check)
{
	double complex wp, wpd;
	double b2 = zq_to_double(fe_q(&E->b2)), a1 = zq_to_double(fe_q(&E->a1)),
	    a3 = zq_to_double(fe_q(&E->a3)), c4 = zq_to_double(fe_q(&E->c4)),
	    c6 = zq_to_double(fe_q(&E->c6));

	wp_series(P->w1, P->w2, z, &wp, &wpd);
	/* p'^2 = 4 p^3 - g2 p - g3, g2 = c4/12, g3 = c6/216 */
	if (check)
		*check = cabs(wpd * wpd - (4 * wp * wp * wp - c4 / 12 * wp -
		    c6 / 216)) / (1 + cabs(wpd * wpd));
	*x = wp - b2 / 12;
	*y = (wpd - a1 * *x - a3) / 2;
}

/* a_n for n <= M (an[0] unused) */
void
ecan_an(const struct ec *E, const struct ecq_local *L, int nl, int64_t M,
    double *an)
{
	int64_t *spf = xcalloc((size_t)M + 1, sizeof *spf), n, p, k;
	double *ap = xcalloc((size_t)M + 1, sizeof *ap);

	for (n = 2; n <= M; n++)
		if (!spf[n])
			for (k = n; k <= M; k += n)
				if (!spf[k])
					spf[k] = n;
	for (p = 2; p <= M; p++) {
		int bad = -1, j;
		if (spf[p] != p)
			continue;
		for (j = 0; j < nl; j++)
			if (z_fits_i64(L[j].p) && z_to_i64(L[j].p) == p)
				bad = j;
		ap[p] = bad >= 0 ? L[bad].ap : (double)ecq_ap(E, p);
	}
	an[1] = 1;
	for (n = 2; n <= M; n++) {
		int64_t q = spf[n], m = n, pk = 1, e = 0;
		while (m % q == 0) {
			m /= q;
			pk *= q;
			e++;
		}
		if (m > 1) {
			an[n] = an[m] * an[pk];
			continue;
		}
		/* a prime power q^e */
		if (e == 1)
			an[n] = ap[q];
		else {
			int good = 1, j;
			for (j = 0; j < nl; j++)
				if (z_fits_i64(L[j].p) && z_to_i64(L[j].p) == q)
					good = 0;
			if (good)
				an[n] = ap[q] * an[n / q] - (double)q * an[n / q / q];
			else
				an[n] = ap[q] * an[n / q];
		}
	}
	free(spf);
	free(ap);
}

/* E1(x) for x > 0 */
static double
expint1(double x)
{
	if (x < 1) {
		double s = -0.57721566490153286061 - log(x), t = 1;
		int k;
		for (k = 1; k < 60; k++) {
			t *= -x / k;
			s -= t / k;
		}
		return s;
	} else {
		/* continued fraction (Lentz) */
		double b = x + 1, c = 1e300, d = 1 / b, h = d;
		int i;
		for (i = 1; i < 300; i++) {
			double an = -(double)i * i, del;
			b += 2;
			d = 1 / (an * d + b);
			c = b + an / c;
			del = c * d;
			h *= del;
			if (fabs(del - 1) < 1e-16)
				break;
		}
		return h * exp(-x);
	}
}

/* G_r(x) = 1/(r-1)! int_0^inf exp(-x e^u) u^(r-1) du */
static double
Gr(int r, double x)
{
	double U, h, s = 0, f;
	int n = 4000, i;

	if (r == 1)
		return expint1(x);
	U = log(80 / x);
	if (U < 1)
		U = 1;
	h = U / n;
	for (i = 0; i <= n; i++) {
		double u = i * h, w = (i == 0 || i == n) ? 1 : (i & 1) ? 4 : 2;
		s += w * exp(-x * exp(u)) * pow(u, r - 1);
	}
	s *= h / 3;
	for (f = 1, i = 2; i < r; i++)
		f *= i;
	return s / f;
}

int
ecan_lseries(const struct ec *E, const struct ecq_local *L, int nl,
    zint Nz, struct ecan_L *R)
{
	double N, sq, *an, s1a, s2a, s1b, s2b, A = 1.2;
	int64_t M, n;

	memset(R, 0, sizeof *R);
	if (!z_fits_i64(Nz) || z_to_i64(Nz) > 100000000)
		return 0;
	N = (double)z_to_i64(Nz);
	sq = sqrt(N);
	M = (int64_t)(8 * A * sq) + 20;
	an = xcalloc((size_t)M + 1, sizeof *an);
	ecan_an(E, L, nl, M, an);
	R->nterms = M;
	for (n = 1; n <= 30 && n <= M; n++)
		R->an[n] = (int64_t)an[n];
	/* the root number: L(1) = S1(A) + w S2(A) must not depend on A */
	s1a = s2a = s1b = s2b = 0;
	for (n = 1; n <= M; n++) {
		double c = an[n] / (double)n;
		s1a += c * exp(-2 * M_PI * n / (A * sq));
		s2a += c * exp(-2 * M_PI * n * A / sq);
		s1b += c * exp(-2 * M_PI * n / sq);
		s2b += c * exp(-2 * M_PI * n / sq);
	}
	{
		double dp = fabs((s1a + s2a) - (s1b + s2b));
		double dm = fabs((s1a - s2a) - (s1b - s2b));
		R->w = dp <= dm ? 1 : -1;
		R->wtest = R->w == 1 ? dm : dp;   /* the other sign's mismatch */
		R->wfit = R->w == 1 ? dp : dm;
	}
	/* the tail: |a_n| <= n, so sum_{n > M} e^{-cn} bounds each sum */
	{
		double c = 2 * M_PI / (A * sq);
		R->err = 4 * exp(-c * (double)M) / (1 - exp(-c)) + 1e-13;
	}
	if (R->w == 1) {
		double s = 0;
		for (n = 1; n <= M; n++)
			s += an[n] / (double)n * exp(-2 * M_PI * n / sq);
		R->L[0] = 2 * s;
		R->rank = 0;
		if (fabs(R->L[0]) <= 1e-8) {
			double s2 = 0;
			for (n = 1; n <= M; n++)
				s2 += an[n] / (double)n * Gr(2, 2 * M_PI * n / sq);
			R->L[2] = 2 * 2 * s2;
			R->rank = fabs(R->L[2]) > 1e-8 ? 2 : 4;
		}
	} else {
		double s = 0;
		for (n = 1; n <= M; n++)
			s += an[n] / (double)n * Gr(1, 2 * M_PI * n / sq);
		R->L[1] = 2 * s;
		R->rank = 1;
		if (fabs(R->L[1]) <= 1e-8) {
			double s3 = 0;
			for (n = 1; n <= M; n++)
				s3 += an[n] / (double)n * Gr(3, 2 * M_PI * n / sq);
			R->L[3] = 2 * 6 * s3;
			R->rank = fabs(R->L[3]) > 1e-8 ? 3 : 5;
		}
	}
	free(an);
	return 1;
}
