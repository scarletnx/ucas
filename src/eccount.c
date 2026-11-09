/*
 * eccount.c - #E(F_p): naive counting, baby-step giant-step with Mestre's
 * twist trick, CM for j = 0 and 1728 (Cornacchia), and the certificate
 * every count carries: points whose orders (computed exactly from a
 * factorisation of the count) leave a single multiple in the Hasse
 * interval.  Group structure, twists, embedding degrees.  Schoof and SEA
 * are in ecschoof.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* [p + 1 - 2 sqrt p, p + 1 + 2 sqrt p] (integer bounds) */
void
ec_hasse(const nn *p, nn *lo, nn *hi, nn *w2)
{
	nn t, s;

	nn_shl(&t, p, 2);
	nn_sqrt(&s, &t);        /* floor(2 sqrt p) */
	nn_add_u64(&t, p, 1);
	nn_sub(lo, &t, &s);
	nn_add(hi, &t, &s);
	if (w2)
		*w2 = s;
}

/* the quadratic twist by a non-residue d: a d^2, b d^3 */
void
ec_twist(const struct ecp_curve *E, struct ecp_curve *T, u64 *dout)
{
	const struct mf *F = E->K;
	u64 d[FW], a[FW], b[FW], t[FW];

	mf_copy(F, d, F->nonres);
	mf_sqr(F, t, d);
	mf_mul(F, a, E->a, t);
	mf_mul(F, t, t, d);
	mf_mul(F, b, E->b, t);
	ecp_set(T, F, a, b);
	if (dout)
		mf_copy(F, dout, d);
}

/* ---- naive ---- */

int
ec_count_naive(const struct ecp_curve *E, nn *N)
{
	const struct mf *F = E->K;
	u64 p, x, cnt = 1, a, b;
	unsigned char *sq;

	if (!nn_fits_u64(&F->P) || F->P.w[0] > (1ULL << 26))
		return 0;
	p = F->P.w[0];
	{
		nn t;
		mf_to_nn(F, &t, E->a);
		a = t.w[0];
		mf_to_nn(F, &t, E->b);
		b = t.w[0];
	}
	sq = xcalloc((size_t)p, 1);
	for (x = 0; x < p; x++)
		sq[(x * x) % p]++;
	for (x = 0; x < p; x++) {
		u64 r = ((x * x % p + a) % p * x % p + b) % p;
		cnt += sq[r];
	}
	free(sq);
	nn_set(N, cnt);
	return 1;
}

/* ---- point orders ---- */

/* the exact order of P given a multiple N (factored); 0: N P != O */
int
ec_point_order(const struct ecp_curve *E, const struct ecp_pt *P,
    const nn *N, const struct nnfac *f, nn *ord)
{
	struct ecp_pt T;
	nn m = *N, q;
	int i;

	ecp_mul(E, &T, P, &m);
	if (!T.inf)
		return 0;
	for (i = 0; i < f->k; i++) {
		int e;
		for (e = 0; e < f->e[i]; e++) {
			nn r;
			nn_divmod(&q, &r, &m, &f->p[i]);
			if (!nn_is_zero(&r))
				break;
			ecp_mul(E, &T, P, &q);
			if (!T.inf)
				break;
			m = q;
		}
	}
	*ord = m;
	return 1;
}

static void
nn_lcm(nn *r, const nn *a, const nn *b)
{
	nn g, q;

	nn_gcd(&g, a, b);
	nn_divmod(&q, NULL, a, &g);
	if (!nn_mul(r, &q, b))
		fail("ecff: lcm overflow");
}

/* multiples of L in [lo, hi] satisfying also 2p+2-N = 0 mod L2 */
static int
count_candidates(const nn *lo, const nn *hi, const nn *L, const nn *L2,
    const nn *p, nn *first, int max)
{
	nn N, r, s2;
	int c = 0;

	nn_add_u64(&s2, p, 1);
	nn_shl(&s2, &s2, 1);
	/* first multiple of L >= lo */
	nn_divmod(NULL, &r, lo, L);
	N = *lo;
	if (!nn_is_zero(&r)) {
		nn_sub(&N, &N, &r);
		nn_add(&N, &N, L);
	}
	for (; nn_cmp(&N, hi) <= 0 && c <= max; nn_add(&N, &N, L)) {
		nn t;
		nn_sub(&t, &s2, &N);
		nn_divmod(NULL, &r, &t, L2);
		if (nn_is_zero(&r)) {
			if (c == 0)
				*first = N;
			c++;
		}
	}
	return c;
}

/*
 * Certify #E = N: random points on E (and on the twist, whose count is
 * 2p + 2 - N); their exact orders' lcm L (L' on the twist) must leave N
 * as the only candidate in the Hasse interval.  Returns 1 and fills cert.
 */
int
ec_certify(const struct ecp_curve *E, const nn *N, struct eccert *C)
{
	const struct mf *F = E->K;
	struct ecp_curve T;
	struct nnfac fN, fT;
	nn lo, hi, L, L2, NT, s2, first, ord;
	int it, c;

	memset(C, 0, sizeof *C);
	ec_hasse(&F->P, &lo, &hi, NULL);
	if (nn_cmp(N, &lo) < 0 || nn_cmp(N, &hi) > 0)
		return 0;
	ec_twist(E, &T, NULL);
	nn_add_u64(&s2, &F->P, 1);
	nn_shl(&s2, &s2, 1);
	nn_sub(&NT, &s2, N);
	nn_factor(N, &fN, 1);
	fT.k = 0;
	fT.complete = 1;
	nn_set(&L, 1);
	nn_set(&L2, 1);
	for (it = 0; it < 40; it++) {
		struct ecp_pt P;
		/* points of E first; the twist (Mestre) only when E's group
		 * exponent leaves several candidates */
		int tw = it >= 8 && it % 3 == 2;
		if (tw && fT.k == 0)
			nn_factor(&NT, &fT, 1);
		ecp_random_point(tw ? &T : E, &P);
		if (!ec_point_order(tw ? &T : E, &P, tw ? &NT : N,
		    tw ? &fT : &fN, &ord))
			return 0;     /* N (or 2p+2-N) does not kill P: wrong N */
		if (tw)
			nn_lcm(&L2, &L2, &ord);
		else {
			nn_lcm(&L, &L, &ord);
			if (C->npts < 4) {
				C->P[C->npts] = P;
				C->ord[C->npts] = ord;
				C->npts++;
			}
		}
		c = count_candidates(&lo, &hi, &L, &L2, &F->P, &first, 2);
		if (c == 1 && nn_eq(&first, N)) {
			C->L = L;
			C->L2 = L2;
			C->ok = 1;
			C->complete = fN.complete && (nn_is_one(&L2) || fT.complete);
			C->iters = it + 1;
			return 1;
		}
	}
	return 0;
}

/* ---- baby-step giant-step: all k in [k0, k1] with k Q = O ---- */

static u64
hkey(const u64 *x)
{
	u64 h = x[0] * 0x9e3779b97f4a7c15ULL;

	return h ^ (h >> 29);
}

/* returns the number of solutions written (up to max), or -1 when Q has
 * a small order (then *small = ord(Q)) */
static int
bsgs_all(const struct ecp_curve *E, const struct ecp_pt *Q,
    const struct ecp_pt *Tg, const nn *k0, const nn *k1, nn *sol, int max,
    u64 *small)
{
	const struct mf *F = E->K;
	nn w, sN, step, c;
	u64 s, j, hsize, *X, *Y, *Zc, i;
	uint32_t *ht;
	struct ecp_jpt J, Qj;
	struct ecp_pt R, G;
	int nsol = 0, n = F->n;

	if (Q->inf) {
		*small = 1;
		return -1;
	}
	nn_sub(&w, k1, k0);
	nn_shr(&sN, &w, 1);
	nn_sqrt(&sN, &sN);
	s = nn_fits_u64(&sN) ? sN.w[0] + 1 : (1ULL << 40);
	if (s > (1ULL << 22))
		fail("ecff: baby-step giant-step range too large (%d-bit p); "
		    "use schoof or sea", F->bits);
	/* babies jQ, j = 0..s, in columns: X | Y | Z */
	X = xmalloc(sizeof(u64) * (size_t)n * (size_t)(s + 1) * 3);
	Y = X + (size_t)n * (s + 1);
	Zc = Y + (size_t)n * (s + 1);
	ecp_to_jac(E, &Qj, Q);
	ecp_to_jac(E, &J, Q);
	mfv_put(F, X, s + 1, 0, F->one);
	mfv_put(F, Y, s + 1, 0, F->one);
	mfv_put(F, Zc, s + 1, 0, F->one);   /* j = 0: a placeholder */
	for (j = 1; j <= s; j++) {
		if (ecp_jis_inf(E, &J)) {
			*small = j;
			free(X);
			return -1;
		}
		mfv_put(F, X, s + 1, j, J.X);
		mfv_put(F, Y, s + 1, j, J.Y);
		mfv_put(F, Zc, s + 1, j, J.Z);
		ecp_jadd(E, &J, &J, &Qj);
	}
	if (!mfv_inv(F, Zc + 1, s + 1, s, NULL))
		fail("ecff: internal error (bsgs normalise)");
	for (j = 1; j <= s; j++) {
		u64 zi[FW], z2[FW], t[FW];
		mfv_get(F, zi, Zc, s + 1, j);
		mf_sqr(F, z2, zi);
		mfv_get(F, t, X, s + 1, j);
		mf_mul(F, t, t, z2);
		mfv_put(F, X, s + 1, j, t);
		mf_mul(F, z2, z2, zi);
		mfv_get(F, t, Y, s + 1, j);
		mf_mul(F, t, t, z2);
		mfv_put(F, Y, s + 1, j, t);
	}
	for (hsize = 1; hsize < 2 * (s + 1); hsize <<= 1)
		;
	ht = xcalloc((size_t)hsize, sizeof *ht);
	for (j = 1; j <= s; j++) {
		u64 x[FW];
		mfv_get(F, x, X, s + 1, j);
		for (i = hkey(x) & (hsize - 1); ht[i]; i = (i + 1) & (hsize - 1))
			;
		ht[i] = (uint32_t)j;
	}
	/* giants: c = k0 + s + i (2s + 1) */
	nn_add_u64(&c, k0, s);
	nn_set(&step, 2 * s + 1);
	ecp_mul(E, &R, Q, &c);
	if (Tg)
		ecp_sub(E, &R, &R, Tg);     /* giants c Q - T */
	ecp_mul(E, &G, Q, &step);
	for (;;) {
		nn lo_c;
		/* stop once c - s > k1 */
		nn_sub_u64(&lo_c, &c, s);
		if (nn_cmp(&lo_c, k1) > 0)
			break;
		if (R.inf) {
			if (nsol < max)
				sol[nsol] = c;
			nsol++;
		} else {
			for (i = hkey(R.x) & (hsize - 1); ht[i];
			    i = (i + 1) & (hsize - 1)) {
				u64 x[FW], y[FW];
				nn kk;
				j = ht[i];
				mfv_get(F, x, X, s + 1, j);
				if (!mf_eq(F, x, R.x))
					continue;
				mfv_get(F, y, Y, s + 1, j);
				/* R = jQ: (c - j) Q = O; R = -jQ: (c + j) Q = O */
				if (mf_eq(F, y, R.y))
					nn_sub_u64(&kk, &c, j);
				else
					nn_add_u64(&kk, &c, j);
				if (nn_cmp(&kk, k0) >= 0 && nn_cmp(&kk, k1) <= 0) {
					if (nsol < max)
						sol[nsol] = kk;
					nsol++;
				}
			}
		}
		nn_add(&c, &c, &step);
		ecp_add(E, &R, &R, &G);
	}
	free(ht);
	free(X);
	return nsol;
}

/*
 * BSGS + Mestre: candidates N = 0 mod L in the Hasse interval with N P = O
 * for random P on E, and likewise on the twist, until one remains.
 */
int
ec_count_bsgs(const struct ecp_curve *E, nn *N, struct eccert *C)
{
	const struct mf *F = E->K;
	struct ecp_curve T;
	nn lo, hi, L, L2, s2, first;
	int it;

	ec_hasse(&F->P, &lo, &hi, NULL);
	ec_twist(E, &T, NULL);
	nn_add_u64(&s2, &F->P, 1);
	nn_shl(&s2, &s2, 1);
	nn_set(&L, 1);
	nn_set(&L2, 1);
	for (it = 0; it < 60; it++) {
		int tw = it & 1, ns, k;
		const struct ecp_curve *EE = tw ? &T : E;
		struct ecp_pt P, Q;
		nn k0, k1, r, sol[8], *LL = tw ? &L2 : &L, lo2, hi2;
		u64 small = 0;

		ecp_random_point(EE, &P);
		ecp_mul(EE, &Q, &P, LL);
		/* N = L k with N in [lo, hi] (the twist's interval is the same) */
		lo2 = lo;
		hi2 = hi;
		nn_divmod(&k0, &r, &lo2, LL);
		if (!nn_is_zero(&r))
			nn_add_u64(&k0, &k0, 1);
		nn_divmod(&k1, NULL, &hi2, LL);
		ns = bsgs_all(EE, &Q, NULL, &k0, &k1, sol, 8, &small);
		if (ns < 0) {
			nn t;
			if (!nn_mul_u64(&t, LL, small))
				fail("ecff: overflow");
			*LL = t;
		} else if (ns >= 2) {
			/* consecutive solutions differ by lcm(L, ord P) / L */
			nn d, t;
			nn_sub(&d, &sol[1], &sol[0]);
			nn_mul(&t, LL, &d);
			*LL = t;
		} else if (ns == 1) {
			nn t;
			nn_mul(&t, LL, &sol[0]);
			/* a single multiple: L k is the count of EE */
			if (tw)
				nn_sub(&t, &s2, &t);
			*N = t;
			return C ? ec_certify(E, N, C) : 1;
		} else
			return 0;   /* no solution: not an elliptic curve? */
		(void)k;
		if (count_candidates(&lo, &hi, &L, &L2, &F->P, &first, 2) == 1) {
			*N = first;
			return C ? ec_certify(E, N, C) : 1;
		}
	}
	return 0;
}

/* N = c mod M in the Hasse interval with N P = O for random points P */
int
ec_match_bsgs(const struct ecp_curve *E, const nn *c, const nn *M, nn *N)
{
	const struct mf *F = E->K;
	nn lo, hi, c0, r, k1, kk0, cand[8];
	int nc = -1, it, i;

	ec_hasse(&F->P, &lo, &hi, NULL);
	/* c0: the least N >= lo with N = c mod M */
	nn_mod(&r, c, M);
	{
		nn lr;
		nn_mod(&lr, &lo, M);
		c0 = lo;
		nn_sub(&c0, &c0, &lr);
		nn_add(&c0, &c0, &r);
		if (nn_cmp(&c0, &lo) < 0)
			nn_add(&c0, &c0, M);
	}
	if (nn_cmp(&c0, &hi) > 0)
		return 0;
	nn_sub(&k1, &hi, &c0);
	nn_divmod(&k1, NULL, &k1, M);
	nn_zero(&kk0);
	for (it = 0; it < 20; it++) {
		struct ecp_pt P, Q, T;
		nn sol[8];
		u64 small = 0;
		int ns;
		ecp_random_point(E, &P);
		if (nc >= 0) {
			/* filter the candidates */
			int m = 0;
			for (i = 0; i < nc; i++) {
				ecp_mul(E, &T, &P, &cand[i]);
				if (T.inf)
					cand[m++] = cand[i];
			}
			nc = m;
			if (nc <= 1)
				break;
			continue;
		}
		ecp_mul(E, &Q, &P, M);
		ecp_mul(E, &T, &P, &c0);
		ecp_neg(E, &T, &T);
		if (Q.inf)
			continue;
		ns = bsgs_all(E, &Q, &T, &kk0, &k1, sol, 8, &small);
		if (ns < 0 || ns > 8)
			continue;
		nc = ns;
		for (i = 0; i < ns; i++) {
			nn t;
			nn_mul(&t, &sol[i], M);
			nn_add(&cand[i], &c0, &t);
		}
		if (nc <= 1)
			break;
	}
	if (nc != 1)
		return 0;
	*N = cand[0];
	return 1;
}

/* ---- CM: Cornacchia ---- */

/* x^2 + d y^2 = 4p (d = 3, 4, 7, ...): x and y, 0 if none */
int
cornacchia4(const nn *p, u64 d, nn *x, nn *y)
{
	struct mf F;
	nn r0, r1, t, b, fourp, lim, md;
	u64 s[FW];

	if (!mf_init(&F, p) || !mf_prime_setup(&F))
		return 0;
	mf_from_u64(&F, s, d);
	mf_neg(&F, s, s);
	if (!mf_sqrt(&F, s, s))
		return 0;
	mf_to_nn(&F, &r1, s);
	/* r1 = d mod 2 */
	if ((r1.w[0] & 1) != (d & 1))
		nn_sub(&r1, p, &r1);
	nn_shl(&r0, p, 1);
	nn_shl(&fourp, p, 2);
	nn_sqrt(&lim, &fourp);
	while (nn_cmp(&r1, &lim) > 0) {
		nn_mod(&t, &r0, &r1);
		r0 = r1;
		r1 = t;
	}
	/* (4p - r1^2) / d must be a square */
	nn_mul(&t, &r1, &r1);
	nn_sub(&t, &fourp, &t);
	nn_set(&md, d);
	nn_divmod(&b, &r0, &t, &md);
	if (!nn_is_zero(&r0) || !nn_is_square(&b, y))
		return 0;
	*x = r1;
	return 1;
}

/* the traces of curves with CM by D in {-3, -4} (all twists) */
static int
cm_traces(const nn *p, int D, zz *t, int max)
{
	nn x, y;
	int n = 0;

	if (D == -4) {
		/* 4p = x^2 + 4 y^2: traces +-x, +-2y */
		if (!cornacchia4(p, 4, &x, &y))
			return 0;
		zz_from_nn(&t[n++], &x, 0);
		zz_from_nn(&t[n++], &x, 1);
		nn_shl(&y, &y, 1);
		zz_from_nn(&t[n++], &y, 0);
		zz_from_nn(&t[n++], &y, 1);
	} else {
		/* 4p = x^2 + 3 y^2: traces +-x, +-(x + 3y)/2, +-(x - 3y)/2 */
		zz X, Y3, s;
		nn h;
		if (!cornacchia4(p, 3, &x, &y))
			return 0;
		zz_from_nn(&X, &x, 0);
		nn_mul_u64(&h, &y, 3);
		zz_from_nn(&Y3, &h, 0);
		zz_from_nn(&t[n++], &x, 0);
		zz_from_nn(&t[n++], &x, 1);
		zz_add(&s, &X, &Y3);
		nn_shr(&s.m, &s.m, 1);
		t[n++] = s;
		s.neg = !s.neg && !zz_is_zero(&s);
		t[n++] = s;
		zz_sub(&s, &X, &Y3);
		nn_shr(&s.m, &s.m, 1);
		t[n++] = s;
		s.neg = !s.neg && !zz_is_zero(&s);
		t[n++] = s;
	}
	(void)max;
	return n;
}

/* j = 0 or 1728: the count is one of the CM candidates; the certificate
 * picks it */
int
ec_count_cm01(const struct ecp_curve *E, nn *N, struct eccert *C)
{
	const struct mf *F = E->K;
	zz t[6];
	int n, i, D;
	nn p1;

	if (E->a_zero)
		D = -3;
	else if (mf_is_zero(F, E->b))
		D = -4;
	else
		return 0;
	nn_add_u64(&p1, &F->P, 1);
	if ((D == -3 && nn_mod_u64(&F->P, 3) == 2) ||
	    (D == -4 && nn_mod_u64(&F->P, 4) == 3)) {
		/* supersingular: #E = p + 1 */
		*N = p1;
		return ec_certify(E, N, C);
	}
	n = cm_traces(&F->P, D, t, 6);
	for (i = 0; i < n; i++) {
		zz z, P1;
		struct ecp_pt P;
		struct ecp_pt R;
		int k, ok = 1;
		zz_from_nn(&P1, &p1, 0);
		zz_sub(&z, &P1, &t[i]);
		for (k = 0; k < 3 && ok; k++) {
			ecp_random_point(E, &P);
			ecp_mul(E, &R, &P, &z.m);
			ok = R.inf;
		}
		if (ok && ec_certify(E, &z.m, C)) {
			*N = z.m;
			return 1;
		}
	}
	return 0;
}

/* ---- group structure Z/n1 x Z/n2 ---- */

/*
 * n2 | n1, n1 n2 = N, n2 | p - 1.  For each prime q | gcd(N, p - 1), the
 * q-part of the exponent is the largest order among random points pushed
 * into the q-Sylow subgroup; the certificate (two points whose Weil
 * pairing has full order) is added by ecpair.c when asked.
 */
void
ec_structure(const struct ecp_curve *E, const nn *N, const struct nnfac *f,
    nn *n1, nn *n2)
{
	const struct mf *F = E->K;
	nn pm1, g, cof, qe, t;
	int i, it;

	nn_sub_u64(&pm1, &F->P, 1);
	nn_gcd(&g, N, &pm1);
	*n1 = *N;
	nn_set(n2, 1);
	for (i = 0; i < f->k; i++) {
		int e = f->e[i], amax = 0;
		nn_divmod(NULL, &t, &g, &f->p[i]);
		if (!nn_is_zero(&t))
			continue;     /* q does not divide p - 1: cyclic */
		if (e < 2)
			continue;
		nn_pow_u64(&qe, &f->p[i], (u64)e);
		nn_divmod(&cof, NULL, N, &qe);
		for (it = 0; it < 30 && amax < e; it++) {
			struct ecp_pt P, Q;
			int a = 0;
			ecp_random_point(E, &P);
			ecp_mul(E, &Q, &P, &cof);
			while (!Q.inf) {
				ecp_mul(E, &Q, &Q, &f->p[i]);
				a++;
			}
			if (a > amax)
				amax = a;
		}
		if (amax < e) {
			nn qb;
			nn_pow_u64(&qb, &f->p[i], (u64)(e - amax));
			nn_mul(n2, n2, &qb);
		}
	}
	nn_divmod(n1, NULL, N, n2);
}

/* the embedding degree of r: the order of p mod r (0: r | p or > max) */
int
ec_embedding_degree(const nn *p, const nn *r, int max)
{
	nn x, pm;
	int k;

	nn_mod(&pm, p, r);
	if (nn_is_zero(&pm))
		return 0;
	x = pm;
	for (k = 1; k <= max; k++) {
		if (nn_is_one(&x))
			return k;
		nn_mulmod(&x, &x, &pm, r);
	}
	return -1;
}
