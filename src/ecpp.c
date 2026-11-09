/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecpp.c - elliptic curve primality proving (Atkin-Morain) and its
 * independent checker.
 *
 * For a probable prime N: a discriminant D with 4N = t^2 + |D| v^2, an
 * order m = N + 1 - t (some twist's) = k q with q a probable prime above
 * (N^(1/4) + 1)^2, a CM curve E mod N with m points (a root of H_D mod N)
 * and a point P with k P != O, q (k P) = O.  Goldwasser-Kilian: then N is
 * prime if q is, and the chain goes down to q < 2^64 (deterministic
 * Miller-Rabin).  The checker (ecpp_check) redoes every step with plain
 * affine arithmetic on nn residues (not the Montgomery code that built
 * the certificate), refusing any non-invertible denominator.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* discriminants by class number then |D| (fundamental, h <= 12) */
static int64_t dlist[400];
static int ndl;

static int
fundamental(int64_t D)
{
	int64_t a = -D, p;

	if (a % 4 == 3) {
		/* D = 1 mod 4: squarefree */
		for (p = 2; p * p <= a; p++)
			if (a % (p * p) == 0)
				return 0;
		return 1;
	}
	if (a % 4 == 0) {
		int64_t m = a / 4;
		if (m % 4 != 1 && m % 4 != 2)
			return 0;
		for (p = 2; p * p <= m; p++)
			if (m % (p * p) == 0)
				return 0;
		return 1;
	}
	return 0;
}

static void
build_dlist(void)
{
	int64_t D;
	int h, i, j;
	int hs[400];

	if (ndl)
		return;
	for (D = -3; D > -3000 && ndl < 400; D--) {
		if (!fundamental(D))
			continue;
		h = cm_classno(D);
		if (h > 10)
			continue;
		dlist[ndl] = D;
		hs[ndl++] = h;
	}
	/* stable sort by h */
	for (i = 1; i < ndl; i++)
		for (j = i; j > 0 && hs[j - 1] > hs[j]; j--) {
			int64_t t = dlist[j];
			int th = hs[j];
			dlist[j] = dlist[j - 1];
			hs[j] = hs[j - 1];
			dlist[j - 1] = t;
			hs[j - 1] = th;
		}
}

/* (sqrt q - 1)^4 > N: then q > (N^(1/4) + 1)^2 */
static int
big_enough(const nn *q, const nn *N)
{
	nn s, t;

	nn_sqrt(&s, q);
	if (nn_cmp_u64(&s, 2) < 0)
		return 0;
	nn_sub_u64(&s, &s, 1);
	nn_mul(&t, &s, &s);
	if (!nn_mul(&t, &t, &t))
		return 1;
	return nn_cmp(&t, N) > 0;
}

/* the traces of the twists for D: +-t (and the extra ones for -3, -4) */
static int
traces(int64_t D, const nn *t, const nn *v, zz *out)
{
	int n = 0;

	zz_from_nn(&out[n++], t, 0);
	zz_from_nn(&out[n++], t, 1);
	if (D == -4) {
		nn w;
		nn_shl(&w, v, 1);
		zz_from_nn(&out[n++], &w, 0);
		zz_from_nn(&out[n++], &w, 1);
	} else if (D == -3) {
		zz T, V3, s;
		nn h;
		zz_from_nn(&T, t, 0);
		nn_mul_u64(&h, v, 3);
		zz_from_nn(&V3, &h, 0);
		zz_add(&s, &T, &V3);
		nn_shr(&s.m, &s.m, 1);
		out[n++] = s;
		s.neg = !s.neg && !zz_is_zero(&s);
		out[n++] = s;
		zz_sub(&s, &T, &V3);
		nn_shr(&s.m, &s.m, 1);
		out[n++] = s;
		s.neg = !s.neg && !zz_is_zero(&s);
		out[n++] = s;
	}
	return n;
}

/* m = k q with k smooth (primes < 2^17) and q a probable prime */
static int
split_order(const nn *m, nn *k, nn *q)
{
	const uint32_t *pr;
	size_t np = ecff_primes(1 << 17, &pr), i;
	nn r = *m, t;

	nn_set(k, 1);
	for (i = 0; i < np; i++) {
		while (nn_div_u64(&t, &r, pr[i]) == 0) {
			r = t;
			nn_mul_u64(k, k, pr[i]);
		}
		if (nn_cmp_u64(&r, (u64)pr[i] * pr[i]) < 0)
			break;
	}
	if (nn_is_one(k) || nn_cmp_u64(&r, 2) <= 0)
		return 0;
	*q = r;
	return nn_isprime(q);
}

int
ecpp_prove(const nn *N0, struct ecppcert *C, int maxsteps)
{
	nn N = *N0;
	int s = 0, di;

	build_dlist();
	memset(C, 0, sizeof *C);
	if (!nn_isprime(&N))
		return 0;
	while (!nn_fits_u64(&N)) {
		int found = 0;
		struct mf F;
		if (s >= maxsteps || s >= ECPPMAX)
			return 0;
		mf_init(&F, &N);
		mf_prime_setup(&F);
		for (di = 0; di < ndl && !found; di++) {
			int64_t D = dlist[di];
			nn t, v, Dm;
			zz tr[6];
			int nt, i;
			zz Dz;
			zz_set_i64(&Dz, D);
			zz_mod(&Dm, &Dz, &N);
			if (nn_jacobi(&Dm, &N) != 1)
				continue;
			if (!cornacchia4(&N, (u64)-D, &t, &v))
				continue;
			nt = traces(D, &t, &v, tr);
			for (i = 0; i < nt && !found; i++) {
				nn m, k, q, p1;
				zz P1, Mz;
				struct px H;
				u64 roots[16 * FW];
				int nr, r;
				nn_add_u64(&p1, &N, 1);
				zz_from_nn(&P1, &p1, 0);
				zz_sub(&Mz, &P1, &tr[i]);
				m = Mz.m;
				if (!split_order(&m, &k, &q) || !big_enough(&q, &N))
					continue;
				/* the curve: a root of H_D mod N, the right twist */
				px_init(&H);
				if (!cm_hilbert_mod(&F, D, &H)) {
					px_free(&H);
					continue;
				}
				nr = px_roots(&F, &H, roots, 16);
				px_free(&H);
				for (r = 0; r < nr && !found; r++) {
					struct ecp_curve E;
					struct ecp_pt P, Q, T;
					int tries;
					if (!cm_curve_for(&F, roots + (size_t)r * FW, &tr[i],
					    &E))
						continue;
					for (tries = 0; tries < 20; tries++) {
						ecp_random_point(&E, &P);
						ecp_mul(&E, &Q, &P, &k);
						if (Q.inf)
							continue;
						ecp_mul(&E, &T, &Q, &q);
						if (!T.inf)
							break;
						C->step[s].N = N;
						C->step[s].D = D;
						mf_to_nn(&F, &C->step[s].a, E.a);
						mf_to_nn(&F, &C->step[s].b, E.b);
						C->step[s].m = m;
						C->step[s].q = q;
						mf_to_nn(&F, &C->step[s].x, P.x);
						mf_to_nn(&F, &C->step[s].y, P.y);
						found = 1;
						break;
					}
				}
				if (found)
					N = q;
			}
		}
		if (!found)
			return 0;
		s++;
	}
	C->n = s;
	C->last = N;
	return 1;
}

/* ---- the checker: plain affine arithmetic mod N ---- */

struct apt {
	nn x, y;
	int inf;
};

/* R = P + Q on y^2 = x^3 + a x + b mod N; 0 when a denominator is not
 * invertible (N composite, or a bad certificate) */
static int
aadd(const nn *N, const nn *a, struct apt *R, const struct apt *P,
    const struct apt *Q)
{
	nn l, d, t, inv, x3, y3;

	if (P->inf) {
		*R = *Q;
		return 1;
	}
	if (Q->inf) {
		*R = *P;
		return 1;
	}
	if (nn_eq(&P->x, &Q->x)) {
		nn_addmod(&t, &P->y, &Q->y, N);
		if (nn_is_zero(&t)) {
			R->inf = 1;
			return 1;
		}
		nn_mulmod(&l, &P->x, &P->x, N);
		nn_mulmod(&t, &l, &(nn){{3}}, N);
		nn_addmod(&l, &t, a, N);
		nn_addmod(&d, &P->y, &P->y, N);
	} else {
		nn_submod(&l, &Q->y, &P->y, N);
		nn_submod(&d, &Q->x, &P->x, N);
	}
	if (!nn_invmod(&inv, &d, N))
		return 0;
	nn_mulmod(&l, &l, &inv, N);
	nn_mulmod(&x3, &l, &l, N);
	nn_submod(&x3, &x3, &P->x, N);
	nn_submod(&x3, &x3, &Q->x, N);
	nn_submod(&t, &P->x, &x3, N);
	nn_mulmod(&y3, &l, &t, N);
	nn_submod(&y3, &y3, &P->y, N);
	R->x = x3;
	R->y = y3;
	R->inf = 0;
	return 1;
}

static int
amul(const nn *N, const nn *a, struct apt *R, const struct apt *P,
    const nn *k)
{
	struct apt A, B = *P;
	int i;

	A.inf = 1;
	for (i = nn_bits(k) - 1; i >= 0; i--) {
		if (!aadd(N, a, &A, &A, &A))
			return 0;
		if (nn_bit(k, i) && !aadd(N, a, &A, &A, &B))
			return 0;
	}
	*R = A;
	return 1;
}

/* every step; why says what failed */
int
ecpp_check(const struct ecppcert *C, const nn *N0, char *why, size_t wn)
{
	int s;
	nn expect = *N0;

	for (s = 0; s < C->n; s++) {
		const struct ecppstep *S = &C->step[s];
		const nn *N = &S->N;
		nn t, u, k, r, g;
		struct apt P, Q, T;
		if (!nn_eq(N, &expect)) {
			snprintf(why, wn, "step %d: N is not the previous q", s + 1);
			return 0;
		}
		nn_gcd(&g, N, &(nn){{6}});
		if (!nn_is_one(&g)) {
			snprintf(why, wn, "step %d: gcd(N, 6) != 1", s + 1);
			return 0;
		}
		/* 4a^3 + 27b^2 a unit */
		nn_mulmod(&t, &S->a, &S->a, N);
		nn_mulmod(&t, &t, &S->a, N);
		nn_mulmod(&t, &t, &(nn){{4}}, N);
		nn_mulmod(&u, &S->b, &S->b, N);
		nn_mulmod(&u, &u, &(nn){{27}}, N);
		nn_addmod(&t, &t, &u, N);
		nn_gcd(&g, &t, N);
		if (!nn_is_one(&g)) {
			snprintf(why, wn, "step %d: the curve is singular mod N", s + 1);
			return 0;
		}
		/* P on the curve */
		nn_mulmod(&t, &S->y, &S->y, N);
		nn_mulmod(&u, &S->x, &S->x, N);
		nn_addmod(&u, &u, &S->a, N);
		nn_mulmod(&u, &u, &S->x, N);
		nn_addmod(&u, &u, &S->b, N);
		if (!nn_eq(&t, &u)) {
			snprintf(why, wn, "step %d: P is not on the curve", s + 1);
			return 0;
		}
		/* m = k q, q > (N^(1/4) + 1)^2 */
		nn_divmod(&k, &r, &S->m, &S->q);
		if (!nn_is_zero(&r) || nn_cmp_u64(&k, 2) < 0) {
			snprintf(why, wn, "step %d: q does not divide m", s + 1);
			return 0;
		}
		if (!big_enough(&S->q, N)) {
			snprintf(why, wn, "step %d: q <= (N^(1/4) + 1)^2", s + 1);
			return 0;
		}
		/* Q = k P != O, q Q = O, every denominator invertible */
		P.x = S->x;
		P.y = S->y;
		P.inf = 0;
		if (!amul(N, &S->a, &Q, &P, &k) || Q.inf) {
			snprintf(why, wn, "step %d: k P is O or not computable", s + 1);
			return 0;
		}
		if (!amul(N, &S->a, &T, &Q, &S->q) || !T.inf) {
			snprintf(why, wn, "step %d: q (k P) != O", s + 1);
			return 0;
		}
		expect = S->q;
	}
	if (!nn_eq(&C->last, &expect) || !nn_fits_u64(&expect) ||
	    !nt_isprime(expect.w[0])) {
		snprintf(why, wn, "the last q is not a prime below 2^64");
		return 0;
	}
	snprintf(why, wn, "%d steps: each curve nonsingular, P on it, q > (N^(1/4) "
	    "+ 1)^2, k P != O and q k P = O with every denominator invertible "
	    "mod N; the last q = %s < 2^64 is prime (deterministic Miller-Rabin)",
	    C->n, nn_str(&expect));
	return 1;
}
