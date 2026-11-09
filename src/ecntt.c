/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecntt.c - multi-modular number-theoretic transform for polynomials over
 * F_p: the coefficients (Montgomery residues, as integers) are reduced
 * mod several 62-bit NTT primes q = c 2^26 + 1, multiplied by NTT there,
 * recombined by Garner's CRT (the integer product is below the product
 * of the primes) and reduced mod p.  Exact: the primes' product exceeds
 * the largest possible coefficient of the integer product.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

#define NQ 12
#define LOGMAX 26

static struct nq {
	u64 q, qinv, r2, g;       /* g: a 2^LOGMAX-th root of unity */
	u64 w64[FW + 1];          /* 2^(64 j) mod q (Montgomery form) */
} Q[NQ];
static int nq_ready;
static u64 garner_inv[NQ][NQ];   /* q_k^-1 mod q_i (plain) */

static inline u64
redc(u128 T, u64 q, u64 qinv)
{
	u64 m = (u64)T * qinv;
	u128 t = (T + (u128)m * q) >> 64;

	return (u64)t >= q ? (u64)t - q : (u64)t;
}

static inline u64
mmul(u64 a, u64 b, const struct nq *P)
{
	return redc((u128)a * b, P->q, P->qinv);
}

static u64
tomont(u64 a, const struct nq *P)
{
	return mmul(a % P->q, P->r2, P);
}

static u64
frommont(u64 a, const struct nq *P)
{
	return redc(a, P->q, P->qinv);
}

static u64
mpow(u64 a, u64 e, const struct nq *P)
{
	u64 r = tomont(1, P);

	while (e) {
		if (e & 1)
			r = mmul(r, a, P);
		a = mmul(a, a, P);
		e >>= 1;
	}
	return r;
}

static void
setup(void)
{
	u64 c = ((1ULL << 62) >> LOGMAX) - 1;
	int k = 0, i, j;

	while (k < NQ) {
		u64 q = (c << LOGMAX) + 1, inv;
		c--;
		if (!nt_isprime(q))
			continue;
		Q[k].q = q;
		inv = q;
		for (i = 0; i < 6; i++)
			inv *= 2 - q * inv;
		Q[k].qinv = (u64)0 - inv;
		Q[k].r2 = (u64)(((((u128)1) << 64) % q) * (((u128)1) << 64) % q);
		{
			u128 r = ((u128)1 << 64) % q;
			Q[k].r2 = (u64)(r * r % q);
		}
		/* g = h^((q-1)/2^LOGMAX) for a primitive root h */
		{
			u64 h = nt_primroot(q);
			Q[k].g = mpow(tomont(h, &Q[k]), (q - 1) >> LOGMAX, &Q[k]);
		}
		Q[k].w64[0] = tomont(1, &Q[k]);
		for (j = 1; j <= FW; j++) {
			u64 two64 = (u64)((((u128)1) << 64) % q);
			Q[k].w64[j] = mmul(Q[k].w64[j - 1], tomont(two64, &Q[k]),
			    &Q[k]);
		}
		k++;
	}
	for (i = 0; i < NQ; i++)
		for (j = 0; j < NQ; j++)
			if (i != j) {
				u64 g;
				garner_inv[i][j] = tomont(nt_inv(Q[j].q % Q[i].q, Q[i].q,
				    &g), &Q[i]);
			}
	nq_ready = 1;
}

/* roots: T[j] = w^j, j < n/2, w a primitive n-th root (or its inverse) */
static void
roots(u64 *T, int logn, const struct nq *P, int inverse)
{
	size_t n = (size_t)1 << logn, j;
	u64 w = mpow(P->g, (u64)1 << (LOGMAX - logn), P);

	if (inverse)
		w = mpow(w, P->q - 2, P);
	T[0] = tomont(1, P);
	for (j = 1; j < n / 2; j++)
		T[j] = mmul(T[j - 1], w, P);
}

static void
ntt(u64 *a, int logn, const struct nq *P, const u64 *T, int inverse)
{
	size_t n = (size_t)1 << logn, i, j, len;
	u64 q = P->q;

	for (i = 1, j = 0; i < n; i++) {
		size_t bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;
		if (i < j) {
			u64 t = a[i];
			a[i] = a[j];
			a[j] = t;
		}
	}
	for (len = 2; len <= n; len <<= 1) {
		size_t half = len / 2, step = n / len;
		for (i = 0; i < n; i += len)
			for (j = 0; j < half; j++) {
				u64 u = a[i + j], v = mmul(a[i + j + half], T[j * step], P);
				u64 s = u + v;
				a[i + j] = s >= q ? s - q : s;
				a[i + j + half] = u >= v ? u - v : u + q - v;
			}
	}
	if (inverse) {
		u64 ninv = mpow(tomont(n % q, P), q - 2, P);
		for (i = 0; i < n; i++)
			a[i] = mmul(a[i], ninv, P);
	}
}

/* r[0 .. na+nb-2] = a * b (raw coefficient rows, stride F->n) */
void
px_mul_ntt(const struct mf *F, u64 *r, const u64 *a, int na, const u64 *b,
    int nb)
{
	int n = F->n, logn = 0, nprimes, bits, i, k, j;
	size_t L, outn = (size_t)(na + nb - 1);
	u64 *A, *B, *res;

	if (!nq_ready)
		setup();
	while (((size_t)1 << logn) < outn)
		logn++;
	if (logn > LOGMAX)
		fail("ntt: product too long");
	L = (size_t)1 << logn;
	bits = 2 * 64 * n + 64 - __builtin_clzll((u64)(na < nb ? na : nb)) + 2;
	nprimes = (bits + 60) / 61;
	if (nprimes > NQ)
		fail("ntt: modulus too large");
	A = xmalloc(sizeof(u64) * L * 3);
	B = A + L;
	{
	u64 *T = B + L, *Ti = xmalloc(sizeof(u64) * L / 2 + 8);
	res = xmalloc(sizeof(u64) * outn * (size_t)nprimes);
	for (k = 0; k < nprimes; k++) {
		const struct nq *P = &Q[k];
		memset(A, 0, sizeof(u64) * L * 2);
		/* the residue of each coefficient (an n-limb integer) mod q */
		for (i = 0; i < na; i++) {
			u64 s = 0;
			for (j = 0; j < n; j++) {
				u64 t = mmul(tomont(a[(size_t)i * n + j], P), P->w64[j], P);
				s = s + t >= P->q ? s + t - P->q : s + t;
			}
			A[i] = s;
		}
		for (i = 0; i < nb; i++) {
			u64 s = 0;
			for (j = 0; j < n; j++) {
				u64 t = mmul(tomont(b[(size_t)i * n + j], P), P->w64[j], P);
				s = s + t >= P->q ? s + t - P->q : s + t;
			}
			B[i] = s;
		}
		roots(T, logn, P, 0);
		roots(Ti, logn, P, 1);
		ntt(A, logn, P, T, 0);
		ntt(B, logn, P, T, 0);
		for (i = 0; (size_t)i < L; i++)
			A[i] = mmul(A[i], B[i], P);
		ntt(A, logn, P, Ti, 1);
		for (i = 0; (size_t)i < outn; i++)
			res[(size_t)i * nprimes + k] = frommont(A[i], P);
	}
	free(Ti);
	}
	/* Garner and the reduction mod p, then one R^-1 */
	for (i = 0; (size_t)i < outn; i++) {
		u64 c[NQ], v[NQ + 2], red[FW], one[FW];
		int vl = 1;
		u64 *x = res + (size_t)i * nprimes;
		for (k = 0; k < nprimes; k++) {
			u64 t = x[k];
			for (j = 0; j < k; j++) {
				/* t = (t - c_j) * q_j^-1 mod q_k */
				u64 cj = c[j] >= Q[k].q ? c[j] - Q[k].q : c[j];
				t = t >= cj ? t - cj : t + Q[k].q - cj;
				t = mmul(t, garner_inv[k][j], &Q[k]);
			}
			c[k] = t;
		}
		/* v = c_0 + q_0 (c_1 + q_1 (c_2 + ...)) */
		memset(v, 0, sizeof v);
		v[0] = c[nprimes - 1];
		for (k = nprimes - 2; k >= 0; k--) {
			u64 carry = 0;
			int z;
			for (z = 0; z < vl; z++) {
				u128 s = (u128)v[z] * Q[k].q + carry;
				v[z] = (u64)s;
				carry = (u64)(s >> 64);
			}
			if (carry)
				v[vl++] = carry;
			{
				u128 s = (u128)v[0] + c[k];
				u64 cc;
				v[0] = (u64)s;
				cc = (u64)(s >> 64);
				for (z = 1; cc && z < NQ + 2; z++) {
					s = (u128)v[z] + cc;
					v[z] = (u64)s;
					cc = (u64)(s >> 64);
					if (z >= vl)
						vl = z + 1;
				}
			}
		}
		l_divmod(NULL, red, v, vl, F->p, n);
		memset(one, 0, sizeof one);
		one[0] = 1;
		mf_mul(F, r + (size_t)i * n, red, one);
	}
	free(A);
	free(res);
}
