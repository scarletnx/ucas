/*
 * ecpair.c - pairings and discrete logarithms.
 *
 * Miller's algorithm on curves over F_q (eck, q = p^k): the Weil pairing
 * (randomised by an auxiliary point, so no divisor meets a zero or pole)
 * and the reduced Tate pairing; the pairing-friendly curves BN254 and
 * BLS12-381 with G2 untwisted into E(F_p^12).  Discrete logs: baby-step
 * giant-step, Pollard rho (r-adding walk, Floyd), Pohlig-Hellman, the MOV
 * transfer through a pairing, and Smart's p-adic attack on anomalous
 * curves.  Every logarithm is checked by multiplying back.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- Miller ---- */

/* the line through A and B (tangent when equal, vertical when A = -B)
 * at R, and the vertical through A + B at R: num / den; C = A + B */
static void
line_eval(const struct eck_curve *E, const struct eck_pt *A,
    const struct eck_pt *B, const struct eck_pt *R, u64 *num, u64 *den,
    struct eck_pt *Cout)
{
	const struct fk *K = E->K;
	u64 l[FKE], t[FKE], u[FKE];
	struct eck_pt C;

	if (A->inf || B->inf) {
		fk_one(K, num);
		fk_one(K, den);
		*Cout = A->inf ? *B : *A;
		return;
	}
	if (fk_eq(K, A->x, B->x)) {
		fk_add(K, t, A->y, B->y);
		if (fk_is_zero(K, t)) {
			/* vertical x - x_A; A + B = O */
			fk_sub(K, num, R->x, A->x);
			fk_one(K, den);
			eck_set_inf(E, Cout);
			return;
		}
		/* tangent */
		fk_sqr(K, l, A->x);
		fk_add(K, t, l, l);
		fk_add(K, l, l, t);
		fk_add(K, l, l, E->a);
		fk_add(K, t, A->y, A->y);
	} else {
		fk_sub(K, l, B->y, A->y);
		fk_sub(K, t, B->x, A->x);
	}
	if (!fk_inv(K, t, t))
		fail("pairing: a non-invertible slope");
	fk_mul(K, l, l, t);
	/* C = A + B */
	fk_sqr(K, C.x, l);
	fk_sub(K, C.x, C.x, A->x);
	fk_sub(K, C.x, C.x, B->x);
	fk_sub(K, t, A->x, C.x);
	fk_mul(K, C.y, l, t);
	fk_sub(K, C.y, C.y, A->y);
	C.inf = 0;
	/* num = y_R - y_A - l (x_R - x_A); den = x_R - x_C */
	fk_sub(K, t, R->x, A->x);
	fk_mul(K, t, l, t);
	fk_sub(K, u, R->y, A->y);
	fk_sub(K, num, u, t);
	fk_sub(K, den, R->x, C.x);
	*Cout = C;
}

/* f_{n,P} at R1 and R2 (divisor n(P) - n(O)): out1 = f(R1), out2 =
 * f(R2) (R2 may be NULL); 0 when an evaluation hits a zero or pole */
static int
miller(const struct eck_curve *E, const struct eck_pt *P, const nn *n,
    const struct eck_pt *R1, const struct eck_pt *R2, u64 *out1, u64 *out2)
{
	const struct fk *K = E->K;
	u64 n1[FKE], d1[FKE], n2[FKE], d2[FKE], a[FKE], b[FKE];
	struct eck_pt T, C;
	int i;

	fk_one(K, n1);
	fk_one(K, d1);
	fk_one(K, n2);
	fk_one(K, d2);
	T = *P;
	for (i = nn_bits(n) - 2; i >= 0; i--) {
		fk_sqr(K, n1, n1);
		fk_sqr(K, d1, d1);
		line_eval(E, &T, &T, R1, a, b, &C);
		fk_mul(K, n1, n1, a);
		fk_mul(K, d1, d1, b);
		if (R2) {
			struct eck_pt C2;
			fk_sqr(K, n2, n2);
			fk_sqr(K, d2, d2);
			line_eval(E, &T, &T, R2, a, b, &C2);
			fk_mul(K, n2, n2, a);
			fk_mul(K, d2, d2, b);
		}
		T = C;
		if (nn_bit(n, i)) {
			line_eval(E, &T, P, R1, a, b, &C);
			fk_mul(K, n1, n1, a);
			fk_mul(K, d1, d1, b);
			if (R2) {
				struct eck_pt C2;
				line_eval(E, &T, P, R2, a, b, &C2);
				fk_mul(K, n2, n2, a);
				fk_mul(K, d2, d2, b);
			}
			T = C;
		}
	}
	if (fk_is_zero(K, n1) || fk_is_zero(K, d1))
		return 0;
	if (!fk_inv(K, d1, d1))
		return 0;
	fk_mul(K, out1, n1, d1);
	if (R2) {
		if (fk_is_zero(K, n2) || fk_is_zero(K, d2) || !fk_inv(K, d2, d2))
			return 0;
		fk_mul(K, out2, n2, d2);
	}
	return 1;
}

/* e_n(P, Q) = [f_P(Q + S) / f_P(S)] / [f_Q(P - S) / f_Q(-S)] */
int
eck_weil(const struct eck_curve *E, const struct eck_pt *P,
    const struct eck_pt *Q, const nn *n, u64 *out)
{
	const struct fk *K = E->K;
	int tries;

	if (P->inf || Q->inf || eck_eq(E, P, Q)) {
		fk_one(K, out);
		return 1;
	}
	for (tries = 0; tries < 30; tries++) {
		struct eck_pt S, QS, PS, mS;
		u64 a[FKE], b[FKE], c[FKE], d[FKE];
		eck_random_point(E, &S);
		eck_add(E, &QS, Q, &S);
		eck_sub(E, &PS, P, &S);
		eck_neg(E, &mS, &S);
		if (QS.inf || PS.inf)
			continue;
		if (!miller(E, P, n, &QS, &S, a, b) ||
		    !miller(E, Q, n, &PS, &mS, c, d))
			continue;
		/* (a / b) / (c / d) = a d / (b c) */
		fk_mul(K, a, a, d);
		fk_mul(K, b, b, c);
		if (!fk_inv(K, b, b))
			continue;
		fk_mul(K, out, a, b);
		return 1;
	}
	return 0;
}

/* the reduced Tate pairing f_P(Q + S) / f_P(S), to the (q - 1)/n */
int
eck_tate(const struct eck_curve *E, const struct eck_pt *P,
    const struct eck_pt *Q, const nn *n, u64 *out, int direct)
{
	const struct fk *K = E->K;
	u64 e[FKMAX * FW + 2], q[FKMAX * FW + 2], f[FKE], g[FKE];
	int ne, tries;

	ne = fk_qm1(K, e);
	{
		u64 nl[NW];
		memcpy(nl, n->w, sizeof nl);
		l_divmod(q, NULL, e, ne, nl, NW);
	}
	if (direct && miller(E, P, n, Q, NULL, f, NULL)) {
		fk_pow_l(K, out, f, q, ne);
		return 1;
	}
	for (tries = 0; tries < 30; tries++) {
		struct eck_pt S, QS;
		eck_random_point(E, &S);
		eck_add(E, &QS, Q, &S);
		if (QS.inf)
			continue;
		if (!miller(E, P, n, &QS, &S, f, g))
			continue;
		if (!fk_inv(K, g, g))
			continue;
		fk_mul(K, f, f, g);
		fk_pow_l(K, out, f, q, ne);
		return 1;
	}
	return 0;
}

/* lift a curve over F_p into F_p^k */
void
eck_from_ecp(struct eck_curve *Ek, const struct fk *K,
    const struct ecp_curve *E)
{
	u64 a[FKE], b[FKE];

	fk_from_mf(K, a, E->a);
	fk_from_mf(K, b, E->b);
	eck_set(Ek, K, a, b);
}

void
eck_pt_from_ecp(const struct fk *K, struct eck_pt *R, const struct ecp_pt *P)
{
	fk_from_mf(K, R->x, P->x);
	fk_from_mf(K, R->y, P->y);
	R->inf = P->inf;
}

/* ---- discrete logs in F_q^* (subgroup of prime order r) ---- */

static u64
fk_hkey(const struct fk *K, const u64 *a)
{
	u64 h = a[0] * 0x9e3779b97f4a7c15ULL;
	int i;

	for (i = 1; i < K->k; i++)
		h ^= a[(size_t)i * FW] * 0xbf58476d1ce4e5b9ULL;
	return h ^ (h >> 31);
}

/* x with g^x = h, 0 <= x < r (r <= 2^44); 0: none */
int
fk_dlog_bsgs(const struct fk *K, const u64 *g, const u64 *h, const nn *r,
    nn *x)
{
	u64 m, j, hs, i, *tab, gm[FKE], cur[FKE], t[FKE], gi[FKE];
	uint32_t *ht;
	size_t esz = (size_t)K->k * FW;
	nn mm;

	if (nn_bits(r) > 44)
		fail("dlog in F_q^*: the subgroup order has %d bits (at most 44 "
		    "here)", nn_bits(r));
	nn_sqrt(&mm, r);
	m = mm.w[0] + 1;
	for (hs = 1; hs < 2 * m; hs <<= 1)
		;
	tab = xmalloc(sizeof(u64) * esz * (size_t)m);
	ht = xcalloc((size_t)hs, sizeof *ht);
	fk_one(K, cur);
	for (j = 0; j < m; j++) {
		/* store limbs of coefficient i contiguously per element */
		for (i = 0; i < (u64)K->k; i++)
			memcpy(tab + j * esz + i * FW, cur + i * FW,
			    sizeof(u64) * FW);
		for (i = fk_hkey(K, cur) & (hs - 1); ht[i]; i = (i + 1) & (hs - 1))
			;
		ht[i] = (uint32_t)(j + 1);
		fk_mul(K, cur, cur, g);
	}
	/* giant: h g^(-m i) */
	if (!fk_inv(K, gi, g))
		return 0;
	fk_pow_l(K, gm, gi, &m, 1);
	fk_copy(K, cur, h);
	for (i = 0; i <= m; i++) {
		u64 s;
		for (s = fk_hkey(K, cur) & (hs - 1); ht[s]; s = (s + 1) & (hs - 1)) {
			u64 jj = ht[s] - 1;
			if (fk_eq(K, tab + jj * esz, cur)) {
				nn a;
				nn_set(&a, i);
				nn_mul_u64(&a, &a, m);
				nn_add_u64(x, &a, jj);
				nn_mod(x, x, r);
				free(tab);
				free(ht);
				return 1;
			}
		}
		fk_mul(K, cur, cur, gm);
	}
	(void)t;
	free(tab);
	free(ht);
	return 0;
}

/* ---- discrete logs on E(F_p) ---- */

/* BSGS: k with k P = Q, 0 <= k < n */
int
ec_dlog_bsgs(const struct ecp_curve *E, const struct ecp_pt *P,
    const struct ecp_pt *Q, const nn *n, nn *k)
{
	const struct mf *F = E->K;
	nn mm;
	u64 m, j, i, hs, *X, *Y;
	uint32_t *ht;
	struct ecp_pt T, G, mP;
	int nl = F->n;

	if (nn_bits(n) > 46)
		fail("bsgs: the order has %d bits (at most 46 here; rho or "
		    "Pohlig-Hellman)", nn_bits(n));
	if (Q->inf) {
		nn_zero(k);
		return 1;
	}
	nn_sqrt(&mm, n);
	m = mm.w[0] + 1;
	for (hs = 1; hs < 2 * m; hs <<= 1)
		;
	X = xmalloc(sizeof(u64) * (size_t)nl * (size_t)m * 2);
	Y = X + (size_t)nl * m;
	ht = xcalloc((size_t)hs, sizeof *ht);
	T = *P;
	for (j = 1; j <= m - 1 + 1 && j < m; j++) {
		if (T.inf)
			break;
		mfv_put(F, X, m, j, T.x);
		mfv_put(F, Y, m, j, T.y);
		for (i = (T.x[0] * 0x9e3779b97f4a7c15ULL) >> 7 & (hs - 1); ht[i];
		    i = (i + 1) & (hs - 1))
			;
		ht[i] = (uint32_t)j;
		ecp_add(E, &T, &T, P);
	}
	ecp_mul_u64(E, &mP, P, m);
	ecp_neg(E, &mP, &mP);
	G = *Q;
	for (i = 0; i <= m; i++) {
		u64 s;
		if (G.inf) {
			nn_set(k, i);
			nn_mul_u64(k, k, m);
			nn_mod(k, k, n);
			free(X);
			free(ht);
			return 1;
		}
		for (s = (G.x[0] * 0x9e3779b97f4a7c15ULL) >> 7 & (hs - 1); ht[s];
		    s = (s + 1) & (hs - 1)) {
			u64 x[FW], y[FW];
			j = ht[s];
			mfv_get(F, x, X, m, j);
			mfv_get(F, y, Y, m, j);
			if (mf_eq(F, x, G.x) && mf_eq(F, y, G.y)) {
				nn_set(k, i);
				nn_mul_u64(k, k, m);
				nn_add_u64(k, k, j);
				nn_mod(k, k, n);
				free(X);
				free(ht);
				return 1;
			}
		}
		ecp_add(E, &G, &G, &mP);
	}
	free(X);
	free(ht);
	return 0;
}

/* Pollard rho, r-adding walk with 20 steps a_i P + b_i Q, Floyd; n prime */
int
ec_dlog_rho(const struct ecp_curve *E, const struct ecp_pt *P,
    const struct ecp_pt *Q, const nn *n, nn *k, u64 *iters)
{
	enum { R = 20 };
	struct ecp_pt M[R], X1, X2;
	nn ma[R], mb[R], a1, b1, a2, b2;
	u64 it, lim;
	int i, attempt;

	if (nn_bits(n) > 64)
		fail("rho: n has %d bits (at most 64 here)", nn_bits(n));
	lim = 64ULL << (nn_bits(n) / 2);
	for (attempt = 0; attempt < 8; attempt++) {
		for (i = 0; i < R; i++) {
			struct ecp_pt A, B;
			nn_rand_below(&ma[i], n);
			nn_rand_below(&mb[i], n);
			ecp_mul(E, &A, P, &ma[i]);
			ecp_mul(E, &B, Q, &mb[i]);
			ecp_add(E, &M[i], &A, &B);
		}
		nn_rand_below(&a1, n);
		nn_set(&b1, 0);
		ecp_mul(E, &X1, P, &a1);
		X2 = X1;
		a2 = a1;
		b2 = b1;
		for (it = 1; it < lim; it++) {
			int h;
			/* tortoise one step, hare two */
			h = X1.inf ? 0 : (int)((X1.x[0] * 0x9e3779b97f4a7c15ULL) >>
			    59) % R;
			ecp_add(E, &X1, &X1, &M[h]);
			nn_addmod(&a1, &a1, &ma[h], n);
			nn_addmod(&b1, &b1, &mb[h], n);
			for (int s = 0; s < 2; s++) {
				h = X2.inf ? 0 : (int)((X2.x[0] *
				    0x9e3779b97f4a7c15ULL) >> 59) % R;
				ecp_add(E, &X2, &X2, &M[h]);
				nn_addmod(&a2, &a2, &ma[h], n);
				nn_addmod(&b2, &b2, &mb[h], n);
			}
			if (ecp_eq(E, &X1, &X2)) {
				/* a1 P + b1 Q = a2 P + b2 Q: k = (a1 - a2)/(b2 - b1) */
				nn db, da, inv;
				nn_submod(&db, &b2, &b1, n);
				nn_submod(&da, &a1, &a2, n);
				if (iters)
					*iters = it;
				if (nn_is_zero(&db) || !nn_invmod(&inv, &db, n))
					break;   /* a useless collision: restart */
				nn_mulmod(k, &da, &inv, n);
				return 1;
			}
		}
	}
	return 0;
}

/* Pohlig-Hellman over the factorisation of n = ord(P); each prime power
 * digit by BSGS (or rho for large primes); CRT */
int
ec_dlog_ph(const struct ecp_curve *E, const struct ecp_pt *P,
    const struct ecp_pt *Q, const nn *n, const struct nnfac *f, nn *k,
    char *log, size_t ln)
{
	nn x, M;
	size_t lo = 0;
	int i;

	nn_zero(&x);
	nn_set(&M, 1);
	log[0] = 0;
	for (i = 0; i < f->k; i++) {
		const nn *q = &f->p[i];
		int e = f->e[i], j;
		nn qe, cof, xi, qj;
		struct ecp_pt P0, Qi, T;
		if (!f->prime[i])
			return 0;
		nn_pow_u64(&qe, q, (u64)e);
		nn_divmod(&cof, NULL, n, &qe);
		/* P0 = (n/q) P has order q */
		{
			nn nq;
			nn_divmod(&nq, NULL, n, q);
			ecp_mul(E, &P0, P, &nq);
		}
		nn_zero(&xi);
		nn_set(&qj, 1);
		for (j = 0; j < e; j++) {
			nn d, c, ex;
			/* Q_j = (n / q^(j+1)) (Q - xi P) */
			ecp_mul(E, &T, P, &xi);
			ecp_sub(E, &Qi, Q, &T);
			nn_pow_u64(&ex, q, (u64)j + 1);
			nn_divmod(&c, NULL, n, &ex);
			ecp_mul(E, &Qi, &Qi, &c);
			if (nn_bits(q) <= 40) {
				if (!ec_dlog_bsgs(E, &P0, &Qi, q, &d))
					return 0;
			} else if (!ec_dlog_rho(E, &P0, &Qi, q, &d, NULL))
				return 0;
			nn_mul(&d, &d, &qj);
			nn_add(&xi, &xi, &d);
			nn_mul(&qj, &qj, q);
		}
		if (lo + 80 < ln)
			lo += (size_t)snprintf(log + lo, ln - lo, "%sk = %s mod %s",
			    i ? ", " : "", nn_str(&xi), nn_str(&qe));
		/* CRT x mod M, xi mod qe */
		{
			nn inv, t, u, Mn;
			nn_mod(&t, &M, &qe);
			if (!nn_invmod(&inv, &t, &qe))
				return 0;
			nn_submod(&u, &xi, &x, &qe);    /* x < M, so reduce x */
			{
				nn xm;
				nn_mod(&xm, &x, &qe);
				nn_submod(&u, &xi, &xm, &qe);
			}
			nn_mulmod(&u, &u, &inv, &qe);
			nn_mul(&t, &u, &M);
			nn_add(&x, &x, &t);
			nn_mul(&Mn, &M, &qe);
			M = Mn;
		}
		(void)cof;
	}
	nn_mod(k, &x, n);
	return 1;
}

/* the order of P from a multiple (the group order), factored */
static int
point_order(const struct ecp_curve *E, const struct ecp_pt *P, const nn *N,
    nn *ord, struct nnfac *fo)
{
	struct nnfac f;

	nn_factor(N, &f, 1);
	if (!ec_point_order(E, P, N, &f, ord))
		return 0;
	nn_factor(ord, fo, 1);
	return 1;
}

/* ---- MOV ---- */

/* t_k: the trace of Frobenius over F_p^k (t_1 = t, t_2 = t^2 - 2p, ...),
 * and #E(F_p^k) = p^k + 1 - t_k (as zz) */
static void
count_ext(const nn *p, const zz *t, int k, zz *Nk)
{
	zz a, b, c, P, two, pk;
	int i;

	zz_from_nn(&P, p, 0);
	zz_set_i64(&two, 2);
	zz_set_i64(&a, 2);      /* t_0 = 2 */
	b = *t;                 /* t_1 */
	for (i = 2; i <= k; i++) {
		zz u, v;
		zz_mul(&u, t, &b);
		zz_mul(&v, &P, &a);
		zz_sub(&c, &u, &v);
		a = b;
		b = c;
	}
	zz_from_nn(&pk, p, 0);
	for (i = 1; i < k; i++)
		zz_mul(&pk, &pk, &P);
	{
		zz one;
		zz_set_i64(&one, 1);
		zz_add(&pk, &pk, &one);
		zz_sub(Nk, &pk, k == 1 ? t : &b);
	}
	(void)two;
}

/* MOV: Q = k P with ord(P) = r prime and embedding degree kk; the
 * pairing maps both to mu_r in F_p^kk, where BSGS finds k */
int
ec_mov(const struct ecp_curve *E, const struct ecp_pt *P,
    const struct ecp_pt *Q, const nn *r, const zz *trace, nn *k,
    int *kdeg, char *info, size_t in)
{
	const struct mf *F = E->K;
	static struct fk K;
	struct eck_curve Ek;
	struct eck_pt Pk, Qk, R;
	zz Nk;
	nn cof, rr;
	u64 e1[FKE], e2[FKE];
	int kk = ec_embedding_degree(&F->P, r, FKMAX), tries;

	if (kk <= 0)
		return 0;
	*kdeg = kk;
	if (!fk_init_auto(&K, F, kk))
		fail("mov: no irreducible polynomial of degree %d found", kk);
	eck_from_ecp(&Ek, &K, E);
	eck_pt_from_ecp(&K, &Pk, P);
	eck_pt_from_ecp(&K, &Qk, Q);
	count_ext(&F->P, trace, kk, &Nk);
	/* R: a point of order r not in <P>: (N_k / r^v) R0 */
	cof = Nk.m;
	for (;;) {
		nn q, rem;
		nn_divmod(&q, &rem, &cof, r);
		if (!nn_is_zero(&rem))
			break;
		cof = q;
	}
	for (tries = 0; tries < 40; tries++) {
		struct eck_pt R0, T;
		eck_random_point(&Ek, &R0);
		eck_mul(&Ek, &R, &R0, &cof);
		if (R.inf)
			continue;
		/* push R to order exactly r */
		for (;;) {
			eck_mul(&Ek, &T, &R, r);
			if (T.inf)
				break;
			R = T;
		}
		if (!eck_weil(&Ek, &Pk, &R, r, e1))
			continue;
		if (fk_is_one(&K, e1))
			continue;       /* R in <P>: degenerate */
		if (!eck_weil(&Ek, &Qk, &R, r, e2))
			continue;
		rr = *r;
		if (!fk_dlog_bsgs(&K, e1, e2, &rr, k))
			return 0;
		snprintf(info, in, "#E(F_p^%d) = %s; R of order r in E(F_p^%d), "
		    "e_r(P, R) a primitive r-th root of unity", kk, zz_str(&Nk),
		    kk);
		return 1;
	}
	return 0;
}

/* ---- Smart's attack on anomalous curves (#E = p) ---- */

/* over Z/p^2 (Montgomery mod p^2): affine points; the lift is on the
 * curve y^2 = x^3 + A x + B with A = a + p s, B chosen so P lifts */
static int
smart_psi(const struct mf *G, const nn *p, const u64 *A,
    const u64 *x, const u64 *y, nn *psi)
{
	/* (p-1) P' by double-and-add over Z/p^2, then p P' = (p-1)P' + P':
	 * x2 - x1 = p u, psi = u / (y2 - y1) mod p */
	nn e, u, dy;
	u64 X[FW], Y[FW], t[FW], l[FW], d[FW];
	int inf = 1, i;

	nn_sub_u64(&e, p, 1);
	for (i = nn_bits(&e) - 1; i >= 0; i--) {
		if (!inf) {
			/* double */
			mf_sqr(G, l, X);
			mf_mul_u64(G, l, l, 3);
			mf_add(G, l, l, A);
			mf_dbl(G, d, Y);
			if (!mf_inv(G, d, d, NULL))
				return 0;
			mf_mul(G, l, l, d);
			mf_sqr(G, t, l);
			mf_sub(G, t, t, X);
			mf_sub(G, t, t, X);
			mf_sub(G, d, X, t);
			mf_mul(G, d, l, d);
			mf_sub(G, Y, d, Y);
			mf_copy(G, X, t);
		}
		if (nn_bit(&e, i)) {
			if (inf) {
				mf_copy(G, X, x);
				mf_copy(G, Y, y);
				inf = 0;
			} else {
				mf_sub(G, l, y, Y);
				mf_sub(G, d, x, X);
				if (!mf_inv(G, d, d, NULL))
					return 0;
				mf_mul(G, l, l, d);
				mf_sqr(G, t, l);
				mf_sub(G, t, t, X);
				mf_sub(G, t, t, x);
				mf_sub(G, d, X, t);
				mf_mul(G, d, l, d);
				mf_sub(G, Y, d, Y);
				mf_copy(G, X, t);
			}
		}
	}
	/* X, Y = (p-1) P'; P' = (x, y) */
	{
		nn xa, xb, ya, yb, r;
		mf_to_nn(G, &xa, X);
		mf_to_nn(G, &xb, x);
		mf_to_nn(G, &ya, Y);
		mf_to_nn(G, &yb, y);
		nn_submod(&u, &xb, &xa, &G->P);
		nn_divmod(&u, &r, &u, p);
		if (!nn_is_zero(&r))
			return 0;          /* not in the kernel of reduction */
		nn_submod(&dy, &yb, &ya, &G->P);
		nn_mod(&dy, &dy, p);
		nn_mod(&u, &u, p);
		if (nn_is_zero(&u))
			return 0;          /* the canonical lift: try another */
		if (!nn_invmod(&r, &dy, p))
			return 0;
		nn_mulmod(psi, &u, &r, p);
	}
	return 1;
}

/* y' with y'^2 = rhs mod p^2 and y' = y mod p (Hensel) */
static void
hensel_y(const struct mf *G, const nn *p, const u64 *rhs, const nn *y,
    u64 *out)
{
	/* y' = y + p d: y^2 + 2 p y d = rhs  =>  d = (rhs - y^2)/p / (2y) */
	nn r, y2, diff, q, inv, d, t;

	mf_to_nn(G, &r, rhs);
	nn_mulmod(&y2, y, y, &G->P);
	nn_submod(&diff, &r, &y2, &G->P);
	nn_divmod(&q, &t, &diff, p);
	nn_add(&t, y, y);
	nn_mod(&t, &t, p);
	nn_invmod(&inv, &t, p);
	nn_mulmod(&d, &q, &inv, p);
	nn_mul(&d, &d, p);
	nn_add(&d, &d, y);
	mf_from_nn(G, out, &d);
}

int
ec_smart(const struct ecp_curve *E, const struct ecp_pt *P,
    const struct ecp_pt *Q, nn *k)
{
	const struct mf *F = E->K;
	struct mf G;
	nn p = F->P, p2, xs, ys, as, bs;
	int s;

	if (!nn_mul(&p2, &p, &p) || nn_bits(&p2) > 64 * FW)
		fail("smart: p has %d bits (at most %d here)", F->bits,
		    64 * FW / 2);
	mf_init(&G, &p2);
	mf_to_nn(F, &as, E->a);
	mf_to_nn(F, &bs, E->b);
	for (s = 1; s < 20; s++) {
		u64 A[FW], B[FW], xp[FW], yp[FW], xq[FW], yq[FW], t[FW], r[FW];
		nn psiP, psiQ, inv, sp, a2;
		/* A = a + p s; B = y_P^2 - x_P^3 - A x_P (P lifts as it is) */
		nn_set(&sp, (u64)s);
		nn_mul(&sp, &sp, &p);
		nn_add(&a2, &as, &sp);
		mf_from_nn(&G, A, &a2);
		mf_to_nn(F, &xs, P->x);
		mf_to_nn(F, &ys, P->y);
		mf_from_nn(&G, xp, &xs);
		mf_from_nn(&G, yp, &ys);
		mf_sqr(&G, B, yp);
		mf_sqr(&G, t, xp);
		mf_add(&G, t, t, A);
		mf_mul(&G, t, t, xp);
		mf_sub(&G, B, B, t);
		/* Q lifted onto it */
		mf_to_nn(F, &xs, Q->x);
		mf_to_nn(F, &ys, Q->y);
		mf_from_nn(&G, xq, &xs);
		mf_sqr(&G, r, xq);
		mf_add(&G, r, r, A);
		mf_mul(&G, r, r, xq);
		mf_add(&G, r, r, B);
		hensel_y(&G, &p, r, &ys, yq);
		if (!smart_psi(&G, &p, A, xp, yp, &psiP) ||
		    !smart_psi(&G, &p, A, xq, yq, &psiQ))
			continue;
		if (!nn_invmod(&inv, &psiP, &p))
			continue;
		nn_mulmod(k, &psiQ, &inv, &p);
		(void)bs;
		return 1;
	}
	return 0;
}
