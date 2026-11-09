/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecalg.c - algorithms that use elliptic curves: Lenstra's ECM (stage 1
 * and the standard stage 2 continuation) on Montgomery curves with
 * Suyama's parametrisation.  ECPP and prime-order curve generation are in
 * ecpp.c.  See ecff.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- primes ---- */

static uint32_t *ptab;
static size_t nptab;
static u64 ptab_lim;

size_t
ecff_primes(u64 limit, const uint32_t **out)
{
	if (limit > ptab_lim) {
		unsigned char *s;
		u64 i, j, lim = limit < 1000000 ? 1000000 : limit;
		size_t cap = 0;
		if (lim > 400000000)
			fail("ecff: prime table limit too high");
		s = xcalloc((size_t)lim + 1, 1);
		free(ptab);
		ptab = NULL;
		nptab = 0;
		for (i = 2; i <= lim; i++) {
			if (s[i])
				continue;
			if (nptab == cap) {
				cap = cap ? cap * 2 : 1 << 16;
				ptab = xrealloc(ptab, cap * sizeof *ptab);
			}
			ptab[nptab++] = (uint32_t)i;
			for (j = i * i; j <= lim; j += i)
				s[j] = 1;
		}
		free(s);
		ptab_lim = lim;
	}
	*out = ptab;
	{
		size_t lo = 0, hi = nptab;
		while (lo < hi) {
			size_t mid = (lo + hi) / 2;
			if (ptab[mid] <= limit)
				lo = mid + 1;
			else
				hi = mid;
		}
		return lo;
	}
}

/* ---- Montgomery x-only arithmetic over Z/n ---- */

struct mpt {
	u64 X[FW], Z[FW];
};

static void
xdbl(const struct mf *F, struct mpt *r, const struct mpt *p, const u64 *a24)
{
	u64 s[FW], d[FW], t[FW], u[FW];

	mf_add(F, s, p->X, p->Z);
	mf_sqr(F, s, s);
	mf_sub(F, d, p->X, p->Z);
	mf_sqr(F, d, d);
	mf_sub(F, t, s, d);             /* 4XZ */
	mf_mul(F, r->X, s, d);
	mf_mul(F, u, a24, t);
	mf_add(F, u, u, d);
	mf_mul(F, r->Z, t, u);
}

static void
xadd(const struct mf *F, struct mpt *r, const struct mpt *p,
    const struct mpt *q, const struct mpt *diff)
{
	u64 a[FW], b[FW], c[FW], d[FW], s[FW], t[FW];

	mf_sub(F, a, p->X, p->Z);
	mf_add(F, b, q->X, q->Z);
	mf_mul(F, a, a, b);
	mf_add(F, c, p->X, p->Z);
	mf_sub(F, d, q->X, q->Z);
	mf_mul(F, c, c, d);
	mf_add(F, s, a, c);
	mf_sqr(F, s, s);
	mf_sub(F, t, a, c);
	mf_sqr(F, t, t);
	mf_mul(F, s, s, diff->Z);
	mf_mul(F, t, t, diff->X);
	mf_copy(F, r->X, s);
	mf_copy(F, r->Z, t);
}

static void
xmul_u64(const struct mf *F, struct mpt *r, const struct mpt *p, u64 k,
    const u64 *a24)
{
	struct mpt R0, R1;
	int i;

	if (k == 0) {
		mf_one(F, r->X);
		mf_zero(F, r->Z);
		return;
	}
	if (k == 1) {
		*r = *p;
		return;
	}
	R0 = *p;
	xdbl(F, &R1, p, a24);
	for (i = 62 - __builtin_clzll(k); i >= 0; i--) {
		if ((k >> i) & 1) {
			xadd(F, &R0, &R1, &R0, p);
			xdbl(F, &R1, &R1, a24);
		} else {
			xadd(F, &R1, &R1, &R0, p);
			xdbl(F, &R0, &R0, a24);
		}
	}
	*r = R0;
}

/* ---- ECM ---- */

/* one curve; 1: a factor (or n itself when every prime split at once) */
static int
ecm_curve(const struct mf *F, const nn *n, u64 sigma, u64 B1, u64 B2,
    nn *f, int *stage)
{
	u64 u[FW], v[FW], t[FW], w[FW], a24[FW], acc[FW];
	struct mpt P;
	const uint32_t *pr;
	size_t np, i;
	nn g, z;

	/* Suyama: u = s^2 - 5, v = 4 s; P = (u^3 : v^3);
	 * (A + 2)/4 = (v - u)^3 (3u + v) / (16 u^3 v) */
	mf_from_u64(F, u, sigma);
	mf_sqr(F, u, u);
	mf_from_u64(F, t, 5);
	mf_sub(F, u, u, t);
	mf_from_u64(F, v, 4 * sigma);
	mf_sqr(F, P.X, u);
	mf_mul(F, P.X, P.X, u);
	mf_sqr(F, P.Z, v);
	mf_mul(F, P.Z, P.Z, v);
	mf_sub(F, t, v, u);
	mf_sqr(F, w, t);
	mf_mul(F, t, t, w);              /* (v - u)^3 */
	mf_add(F, w, u, u);
	mf_add(F, w, w, u);
	mf_add(F, w, w, v);              /* 3u + v */
	mf_mul(F, t, t, w);
	mf_mul(F, w, P.X, v);
	mf_mul_u64(F, w, w, 16);         /* 16 u^3 v */
	if (!mf_inv(F, w, w, &g)) {
		if (nn_is_zero(&g) || nn_eq(&g, n))
			return 0;
		*f = g;
		*stage = 0;
		return 1;
	}
	mf_mul(F, a24, t, w);
	/* stage 1 */
	np = ecff_primes(B1, &pr);
	for (i = 0; i < np; i++) {
		u64 q = pr[i], qe = q;
		while (qe <= B1 / q)
			qe *= q;
		xmul_u64(F, &P, &P, qe, a24);
	}
	mf_to_nn(F, &z, P.Z);
	nn_gcd(&g, &z, n);
	if (!nn_is_one(&g)) {
		if (nn_eq(&g, n))
			return 0;
		*f = g;
		*stage = 1;
		return 1;
	}
	if (B2 <= B1)
		return 0;
	/* stage 2: q = m D +- j; acc *= x(m D P) - x(j P) (projectively) */
	{
		enum { D = 2310 };
		static u64 *cx;
		static int jidx[D / 2 + 1];
		int nj = 0, j;
		struct mpt Pj, P2, Pjm2, DP, Rm, Rm1, Tm;
		u64 m, mlo, mhi;
		size_t stride = D / 4 + 1;

		if (!cx)
			cx = xmalloc(sizeof(u64) * FW * 2 * stride);
		xdbl(F, &P2, &P, a24);
		Pj = P;
		Pjm2 = P;   /* (j - 2) P for j = 3: P - 2P = -P, same x */
		for (j = 1; j < D / 2; j += 2) {
			if (j > 1) {
				struct mpt nx;
				if (j == 3)
					xadd(F, &nx, &Pj, &P2, &P);
				else
					xadd(F, &nx, &Pj, &P2, &Pjm2);
				Pjm2 = Pj;
				Pj = nx;
			}
			if (nt_gcd((u64)j, D) != 1)
				continue;
			jidx[j] = nj;
			mfv_put(F, cx, 2 * stride, (size_t)nj, Pj.X);
			mfv_put(F, cx + stride, 2 * stride, (size_t)nj, Pj.Z);
			nj++;
		}
		/* normalise the table: x_j = X_j / Z_j (batch inverse) */
		if (!mfv_inv(F, cx + stride, 2 * stride, (size_t)nj, &g)) {
			if (nn_eq(&g, n) || nn_is_zero(&g))
				return 0;
			*f = g;
			*stage = 2;
			return 1;
		}
		for (j = 0; j < nj; j++) {
			u64 X[FW], Zi[FW];
			mfv_get(F, X, cx, 2 * stride, (size_t)j);
			mfv_get(F, Zi, cx + stride, 2 * stride, (size_t)j);
			mf_mul(F, X, X, Zi);
			mfv_put(F, cx, 2 * stride, (size_t)j, X);
		}
		xmul_u64(F, &DP, &P, D, a24);
		mlo = B1 / D;
		if (mlo < 1)
			mlo = 1;
		mhi = B2 / D + 1;
		xmul_u64(F, &Rm, &P, mlo * D, a24);
		xmul_u64(F, &Rm1, &P, (mlo - 1) * D, a24);
		np = ecff_primes(B2, &pr);
		mf_one(F, acc);
		i = 0;
		while (i < np && pr[i] <= B1)
			i++;
		for (m = mlo; m <= mhi && i < np; m++) {
			u64 c = m * D, X[FW];
			/* primes in (c - D/2, c + D/2) */
			while (i < np && pr[i] < c + D / 2) {
				u64 q = pr[i++];
				int jj = (int)(q > c ? q - c : c - q);
				if (q + D / 2 <= c || nt_gcd((u64)jj, D) != 1)
					continue;
				mfv_get(F, X, cx, 2 * stride, (size_t)jidx[jj]);
				mf_mul(F, X, X, Rm.Z);
				mf_sub(F, X, Rm.X, X);
				mf_mul(F, acc, acc, X);
			}
			if (m == 1)
				xdbl(F, &Tm, &DP, a24);
			else
				xadd(F, &Tm, &Rm, &DP, &Rm1);
			Rm1 = Rm;
			Rm = Tm;
		}
		mf_to_nn(F, &z, acc);
		nn_gcd(&g, &z, n);
		if (!nn_is_one(&g) && !nn_eq(&g, n)) {
			*f = g;
			*stage = 2;
			return 1;
		}
	}
	return 0;
}

int
ecm_run(const nn *n, u64 B1, u64 B2, int maxcurves, u64 sigma0,
    struct ecm_res *r)
{
	struct mf F;
	int c;
	u64 t0 = ecff_ms();

	memset(r, 0, sizeof *r);
	r->B1 = B1;
	r->B2 = B2;
	if (!(n->w[0] & 1)) {
		nn_set(&r->f, 2);
		r->found = 1;
		return 1;
	}
	if (!mf_init(&F, n))
		fail("ecm: n is too large (at most %d bits)", 64 * FW);
	for (c = 0; c < maxcurves; c++) {
		u64 sigma = sigma0 + (u64)c;
		int stage = 0;
		r->curves = c + 1;
		if (ecm_curve(&F, n, sigma, B1, B2, &r->f, &stage)) {
			r->found = 1;
			r->sigma = sigma;
			r->stage = stage;
			break;
		}
	}
	r->ms = ecff_ms() - t0;
	return r->found;
}

/* for nn_factor: a nontrivial factor of composite n */
int
ecm_find(const nn *n, nn *f, int effort)
{
	static const struct {
		u64 B1;
		int curves;
	} sched[] = {{2000, 25}, {11000, 90}, {50000, 200}, {250000, 400},
	    {1000000, 900}};
	int lv, maxlv = effort <= 0 ? 2 : effort == 1 ? 3 : 5;
	struct ecm_res r;

	for (lv = 0; lv < maxlv; lv++) {
		if (ecm_run(n, sched[lv].B1, sched[lv].B1 * 50, sched[lv].curves,
		    6 + 1000 * (u64)lv, &r)) {
			*f = r.f;
			return 1;
		}
	}
	return 0;
}
