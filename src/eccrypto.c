/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * eccrypto.c - elliptic-curve cryptography as mathematics: ECDSA with
 * RFC 6979 deterministic nonces, ECDH, X25519 and X448 (RFC 7748),
 * Ed25519 (RFC 8032), SEC 1 point encodings and public-key validation.
 * Variable-time code: for checking test vectors and explaining the
 * algorithms, not for protecting secrets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- RFC 6979 ---- */

/* the leftmost qlen bits of b as an integer */
static void
bits2int(nn *r, const unsigned char *b, size_t n, int qlen)
{
	nn_from_bytes_be(r, b, n);
	if ((int)(8 * n) > qlen)
		nn_shr(r, r, (int)(8 * n) - qlen);
}

static void
int2octets(unsigned char *o, size_t rlen, const nn *x)
{
	nn_to_bytes_be(o, rlen, x);
}

void
rfc6979_k(const nn *q, const nn *x, int alg, const unsigned char *h1,
    size_t hlen, nn *k)
{
	int qlen = nn_bits(q);
	size_t rlen = (size_t)(qlen + 7) / 8, ol = (size_t)sha_outlen(alg);
	unsigned char V[64], K[64], buf[64 + 1 + 2 * 80], T[160];
	nn z;
	size_t o, tl;

	memset(V, 1, ol);
	memset(K, 0, ol);
	/* bits2octets(h1) = int2octets(bits2int(h1) mod q) */
	bits2int(&z, h1, hlen, qlen);
	if (nn_cmp(&z, q) >= 0)
		nn_sub(&z, &z, q);
	for (int round = 0; round < 2; round++) {
		o = 0;
		memcpy(buf, V, ol);
		o += ol;
		buf[o++] = (unsigned char)round;
		int2octets(buf + o, rlen, x);
		o += rlen;
		int2octets(buf + o, rlen, &z);
		o += rlen;
		hmac(alg, K, ol, buf, o, K);
		hmac(alg, K, ol, V, ol, V);
	}
	for (;;) {
		tl = 0;
		while (tl * 8 < (size_t)qlen) {
			hmac(alg, K, ol, V, ol, V);
			memcpy(T + tl, V, ol);
			tl += ol;
		}
		bits2int(k, T, tl, qlen);
		if (!nn_is_zero(k) && nn_cmp(k, q) < 0)
			return;
		memcpy(buf, V, ol);
		buf[ol] = 0;
		hmac(alg, K, ol, buf, ol + 1, K);
		hmac(alg, K, ol, V, ol, V);
	}
}

/* ---- ECDSA ---- */

static void
hash_z(const struct ecload *L, int alg, const unsigned char *m, size_t mn,
    unsigned char *h, size_t *hl, nn *z)
{
	*hl = (size_t)sha_outlen(alg);
	sha_hash(alg, m, mn, h);
	bits2int(z, h, *hl, nn_bits(&L->n));
}

int
ecdsa_sign(struct ecload *L, int alg, const nn *d, const unsigned char *m,
    size_t mn, const nn *kfix, nn *r, nn *s, nn *kout)
{
	struct mf N;
	unsigned char h[64];
	size_t hl;
	nn z, k;
	struct ecp_pt R;
	u64 kk[FW], rr[FW], dd[FW], zz_[FW], t[FW];

	if (!mf_init(&N, &L->n))
		fail("ecdsa: the group order must be odd");
	if (nn_is_zero(d) || nn_cmp(d, &L->n) >= 0)
		fail("ecdsa: the private key must be in [1, n - 1]");
	hash_z(L, alg, m, mn, h, &hl, &z);
	if (kfix)
		k = *kfix;
	else
		rfc6979_k(&L->n, d, alg, h, hl, &k);
	*kout = k;
	ecp_mul(&L->E, &R, &L->G, &k);
	if (R.inf)
		return 0;
	mf_to_nn(&L->F, r, R.x);
	nn_mod(r, r, &L->n);
	if (nn_is_zero(r))
		return 0;
	/* s = k^-1 (z + r d) mod n */
	mf_from_nn(&N, kk, &k);
	mf_from_nn(&N, rr, r);
	mf_from_nn(&N, dd, d);
	mf_from_nn(&N, zz_, &z);
	mf_mul(&N, t, rr, dd);
	mf_add(&N, t, t, zz_);
	if (!mf_inv(&N, kk, kk, NULL))
		return 0;
	mf_mul(&N, t, t, kk);
	mf_to_nn(&N, s, t);
	return !nn_is_zero(s);
}

int
ecdsa_verify(struct ecload *L, int alg, const struct ecp_pt *Q,
    const unsigned char *m, size_t mn, const nn *r, const nn *s)
{
	struct mf N;
	unsigned char h[64];
	size_t hl;
	nn z, u1, u2, v;
	struct ecp_pt A, B, X;
	u64 w[FW], t[FW], ss[FW];

	if (nn_is_zero(r) || nn_is_zero(s) || nn_cmp(r, &L->n) >= 0 ||
	    nn_cmp(s, &L->n) >= 0)
		return 0;
	mf_init(&N, &L->n);
	hash_z(L, alg, m, mn, h, &hl, &z);
	mf_from_nn(&N, ss, s);
	if (!mf_inv(&N, w, ss, NULL))
		return 0;
	mf_from_nn(&N, t, &z);
	mf_mul(&N, t, t, w);
	mf_to_nn(&N, &u1, t);
	mf_from_nn(&N, t, r);
	mf_mul(&N, t, t, w);
	mf_to_nn(&N, &u2, t);
	ecp_mul(&L->E, &A, &L->G, &u1);
	ecp_mul(&L->E, &B, Q, &u2);
	ecp_add(&L->E, &X, &A, &B);
	if (X.inf)
		return 0;
	mf_to_nn(&L->F, &v, X.x);
	nn_mod(&v, &v, &L->n);
	return nn_eq(&v, r);
}

/* ---- SEC 1 encodings and validation ---- */

int
sec1_encode(const struct ecload *L, const struct ecp_pt *P, int compressed,
    unsigned char *out)
{
	size_t fl = (size_t)(L->F.bits + 7) / 8;
	nn x, y;

	if (P->inf) {
		out[0] = 0;
		return 1;
	}
	mf_to_nn(&L->F, &x, P->x);
	mf_to_nn(&L->F, &y, P->y);
	if (compressed) {
		out[0] = (unsigned char)(2 + (y.w[0] & 1));
		nn_to_bytes_be(out + 1, fl, &x);
		return (int)(1 + fl);
	}
	out[0] = 4;
	nn_to_bytes_be(out + 1, fl, &x);
	nn_to_bytes_be(out + 1 + fl, fl, &y);
	return (int)(1 + 2 * fl);
}

int
sec1_decode(const struct ecload *L, struct ecp_pt *P, const unsigned char *in,
    size_t n)
{
	size_t fl = (size_t)(L->F.bits + 7) / 8;
	nn x, y;

	if (n == 1 && in[0] == 0) {
		ecp_set_inf(&L->E, P);
		return 1;
	}
	if ((in[0] == 2 || in[0] == 3) && n == 1 + fl) {
		u64 X[FW];
		nn_from_bytes_be(&x, in + 1, fl);
		if (nn_cmp(&x, &L->F.P) >= 0)
			return 0;
		mf_from_nn(&L->F, X, &x);
		if (!ecp_lift_x(&L->E, P, X))
			return 0;
		mf_to_nn(&L->F, &y, P->y);
		if ((int)(y.w[0] & 1) != in[0] - 2)
			mf_neg(&L->F, P->y, P->y);
		return 1;
	}
	if (in[0] == 4 && n == 1 + 2 * fl) {
		nn_from_bytes_be(&x, in + 1, fl);
		nn_from_bytes_be(&y, in + 1 + fl, fl);
		if (nn_cmp(&x, &L->F.P) >= 0 || nn_cmp(&y, &L->F.P) >= 0)
			return 0;
		mf_from_nn(&L->F, P->x, &x);
		mf_from_nn(&L->F, P->y, &y);
		P->inf = 0;
		return ecp_on_curve(&L->E, P);
	}
	return 0;
}

/* SEC 1 3.2.2.1: Q != O, coordinates in range, on the curve, n Q = O */
int
ec_validate_pub(struct ecload *L, const struct ecp_pt *Q, char *why,
    size_t wn)
{
	struct ecp_pt T;

	if (Q->inf) {
		snprintf(why, wn, "the point at infinity");
		return 0;
	}
	if (!ecp_on_curve(&L->E, Q)) {
		snprintf(why, wn, "not on the curve (an invalid-curve attack "
		    "point?)");
		return 0;
	}
	ecp_mul(&L->E, &T, Q, &L->n);
	if (!T.inf) {
		snprintf(why, wn, "n Q != O: not in the prime-order subgroup "
		    "(small-subgroup attack point?)");
		return 0;
	}
	snprintf(why, wn, "Q != O, coordinates reduced, on the curve, n Q = O");
	return 1;
}

/* ---- X25519 / X448 (RFC 7748) ---- */

static struct mf F25519, F448;
static int f25519_ok, f448_ok;

static const struct mf *
f25519(void)
{
	if (!f25519_ok) {
		nn p;
		nn_parse("2^255 - 19", &p, NULL);
		mf_init(&F25519, &p);
		mf_prime_setup(&F25519);
		f25519_ok = 1;
	}
	return &F25519;
}

static const struct mf *
f448(void)
{
	if (!f448_ok) {
		nn p;
		nn_parse("2^448 - 2^224 - 1", &p, NULL);
		mf_init(&F448, &p);
		mf_prime_setup(&F448);
		f448_ok = 1;
	}
	return &F448;
}

static void
xladder(const struct mf *F, nn *out, const nn *k, const nn *u, u64 a24v,
    int bits)
{
	u64 x1[FW], x2[FW], z2[FW], x3[FW], z3[FW], A[FW], AA[FW], B[FW],
	    BB[FW], E[FW], Cc[FW], D[FW], DA[FW], CB[FW], a24[FW], t[FW];
	int i;

	mf_from_nn(F, x1, u);
	mf_one(F, x2);
	mf_zero(F, z2);
	mf_copy(F, x3, x1);
	mf_one(F, z3);
	mf_from_u64(F, a24, a24v);
	for (i = bits - 1; i >= 0; i--) {
		int b = nn_bit(k, i);
		if (b) {
			/* swap (x2, z2) and (x3, z3): the ladder's two points */
			u64 s[FW];
			mf_copy(F, s, x2); mf_copy(F, x2, x3); mf_copy(F, x3, s);
			mf_copy(F, s, z2); mf_copy(F, z2, z3); mf_copy(F, z3, s);
		}
		mf_add(F, A, x2, z2);
		mf_sqr(F, AA, A);
		mf_sub(F, B, x2, z2);
		mf_sqr(F, BB, B);
		mf_sub(F, E, AA, BB);
		mf_add(F, Cc, x3, z3);
		mf_sub(F, D, x3, z3);
		mf_mul(F, DA, D, A);
		mf_mul(F, CB, Cc, B);
		mf_add(F, t, DA, CB);
		mf_sqr(F, x3, t);
		mf_sub(F, t, DA, CB);
		mf_sqr(F, t, t);
		mf_mul(F, z3, x1, t);
		mf_mul(F, x2, AA, BB);
		mf_mul(F, t, a24, E);
		mf_add(F, t, AA, t);
		mf_mul(F, z2, E, t);
		if (b) {
			u64 s[FW];
			mf_copy(F, s, x2); mf_copy(F, x2, x3); mf_copy(F, x3, s);
			mf_copy(F, s, z2); mf_copy(F, z2, z3); mf_copy(F, z3, s);
		}
	}
	/* x2 / z2 (z2 = 0 gives 0, as z2^(p-2) does) */
	if (mf_is_zero(F, z2))
		nn_zero(out);
	else {
		mf_inv(F, z2, z2, NULL);
		mf_mul(F, x2, x2, z2);
		mf_to_nn(F, out, x2);
	}
}

void
x25519(unsigned char *out, const unsigned char *k, const unsigned char *u)
{
	unsigned char kb[32], ub[32];
	nn K, U, R;

	memcpy(kb, k, 32);
	kb[0] &= 248;
	kb[31] &= 127;
	kb[31] |= 64;
	memcpy(ub, u, 32);
	ub[31] &= 127;
	nn_from_bytes_le(&K, kb, 32);
	nn_from_bytes_le(&U, ub, 32);
	nn_mod(&U, &U, &f25519()->P);
	xladder(f25519(), &R, &K, &U, 121665, 255);
	nn_to_bytes_le(out, 32, &R);
}

void
x448(unsigned char *out, const unsigned char *k, const unsigned char *u)
{
	unsigned char kb[56];
	nn K, U, R;

	memcpy(kb, k, 56);
	kb[0] &= 252;
	kb[55] |= 128;
	nn_from_bytes_le(&K, kb, 56);
	nn_from_bytes_le(&U, u, 56);
	nn_mod(&U, &U, &f448()->P);
	xladder(f448(), &R, &K, &U, 39081, 448);
	nn_to_bytes_le(out, 56, &R);
}

/* ---- Ed25519 (RFC 8032): extended coordinates, complete formulas ---- */

struct edpt {
	u64 X[FW], Y[FW], Z[FW], T[FW];
};

static u64 ed_d[FW], ed_d2[FW];
static struct edpt ed_B;
static nn ed_L;
static int ed_ok;

static void
ed_add(const struct mf *F, struct edpt *R, const struct edpt *P,
    const struct edpt *Q)
{
	/* a = -1: A = (Y1 - X1)(Y2 - X2), B = (Y1 + X1)(Y2 + X2),
	 * C = 2d T1 T2, D = 2 Z1 Z2, E = B - A, F = D - C, G = D + C,
	 * H = B + A; X3 = E F, Y3 = G H, T3 = E H, Z3 = F G */
	u64 A[FW], B[FW], C[FW], D[FW], E[FW], Fv[FW], G[FW], H[FW], t[FW];

	mf_sub(F, A, P->Y, P->X);
	mf_sub(F, t, Q->Y, Q->X);
	mf_mul(F, A, A, t);
	mf_add(F, B, P->Y, P->X);
	mf_add(F, t, Q->Y, Q->X);
	mf_mul(F, B, B, t);
	mf_mul(F, C, P->T, Q->T);
	mf_mul(F, C, C, ed_d2);
	mf_mul(F, D, P->Z, Q->Z);
	mf_dbl(F, D, D);
	mf_sub(F, E, B, A);
	mf_sub(F, Fv, D, C);
	mf_add(F, G, D, C);
	mf_add(F, H, B, A);
	mf_mul(F, R->X, E, Fv);
	mf_mul(F, R->Y, G, H);
	mf_mul(F, R->T, E, H);
	mf_mul(F, R->Z, Fv, G);
}

static void
ed_zero(const struct mf *F, struct edpt *P)
{
	mf_zero(F, P->X);
	mf_one(F, P->Y);
	mf_one(F, P->Z);
	mf_zero(F, P->T);
}

static void
ed_mul(const struct mf *F, struct edpt *R, const struct edpt *P, const nn *k)
{
	struct edpt A;
	int i;

	ed_zero(F, &A);
	for (i = nn_bits(k) - 1; i >= 0; i--) {
		ed_add(F, &A, &A, &A);
		if (nn_bit(k, i))
			ed_add(F, &A, &A, P);
	}
	*R = A;
}

static void
ed_encode(const struct mf *F, unsigned char *out, const struct edpt *P)
{
	u64 zi[FW], x[FW], y[FW];
	nn X, Y;

	mf_inv(F, zi, P->Z, NULL);
	mf_mul(F, x, P->X, zi);
	mf_mul(F, y, P->Y, zi);
	mf_to_nn(F, &X, x);
	mf_to_nn(F, &Y, y);
	nn_to_bytes_le(out, 32, &Y);
	out[31] |= (unsigned char)((X.w[0] & 1) << 7);
}

static int
ed_decode(const struct mf *F, struct edpt *P, const unsigned char *in)
{
	unsigned char b[32];
	int sign = in[31] >> 7;
	nn Y, X;
	u64 y[FW], y2[FW], u[FW], v[FW], x[FW], one[FW];

	memcpy(b, in, 32);
	b[31] &= 127;
	nn_from_bytes_le(&Y, b, 32);
	if (nn_cmp(&Y, &F->P) >= 0)
		return 0;
	mf_from_nn(F, y, &Y);
	mf_one(F, one);
	mf_sqr(F, y2, y);
	mf_sub(F, u, y2, one);           /* y^2 - 1 */
	mf_mul(F, v, ed_d, y2);
	mf_add(F, v, v, one);            /* d y^2 + 1 */
	if (!mf_inv(F, v, v, NULL))
		return 0;
	mf_mul(F, u, u, v);
	if (!mf_sqrt(F, x, u))
		return 0;
	mf_to_nn(F, &X, x);
	if (nn_is_zero(&X) && sign)
		return 0;
	if ((int)(X.w[0] & 1) != sign)
		mf_neg(F, x, x);
	mf_copy(F, P->X, x);
	mf_copy(F, P->Y, y);
	mf_one(F, P->Z);
	mf_mul(F, P->T, x, y);
	return 1;
}

static void
ed_setup(void)
{
	const struct mf *F = f25519();
	const struct ecnamed *N;
	zz d;
	nn bx, by;

	if (ed_ok)
		return;
	N = ecnamed_find("ed25519");
	zz_parse(N->b, &d, NULL);
	mf_from_zz(F, ed_d, &d);
	mf_dbl(F, ed_d2, ed_d);
	nn_parse(N->gx, &bx, NULL);
	nn_parse(N->gy, &by, NULL);
	mf_from_nn(F, ed_B.X, &bx);
	mf_from_nn(F, ed_B.Y, &by);
	mf_one(F, ed_B.Z);
	mf_mul(F, ed_B.T, ed_B.X, ed_B.Y);
	nn_parse(N->n, &ed_L, NULL);
	ed_ok = 1;
}

/* SHA-512(parts) mod L, little-endian */
static void
ed_hash_mod_l(nn *r, const unsigned char *a, size_t an,
    const unsigned char *b, size_t bn, const unsigned char *c, size_t cn)
{
	struct sha s;
	unsigned char h[64];

	sha_init(&s, 512);
	sha_update(&s, a, an);
	sha_update(&s, b, bn);
	sha_update(&s, c, cn);
	sha_final(&s, h);
	nn_from_bytes_le(r, h, 64);
	nn_mod(r, r, &ed_L);
}

static void
ed_expand(const unsigned char *sk, nn *a, unsigned char *prefix)
{
	unsigned char h[64];

	sha_hash(512, sk, 32, h);
	h[0] &= 248;
	h[31] &= 127;
	h[31] |= 64;
	nn_from_bytes_le(a, h, 32);
	memcpy(prefix, h + 32, 32);
}

void
ed25519_pub(unsigned char *pk, const unsigned char *sk)
{
	const struct mf *F = f25519();
	nn a;
	unsigned char pre[32];
	struct edpt A;

	ed_setup();
	ed_expand(sk, &a, pre);
	ed_mul(F, &A, &ed_B, &a);
	ed_encode(F, pk, &A);
}

void
ed25519_sign(unsigned char *sig, const unsigned char *sk,
    const unsigned char *m, size_t mn)
{
	const struct mf *F = f25519();
	nn a, r, k, S;
	unsigned char pre[32], pk[32];
	struct edpt R, A;

	ed_setup();
	ed_expand(sk, &a, pre);
	ed_mul(F, &A, &ed_B, &a);
	ed_encode(F, pk, &A);
	ed_hash_mod_l(&r, pre, 32, m, mn, NULL, 0);
	ed_mul(F, &R, &ed_B, &r);
	ed_encode(F, sig, &R);
	ed_hash_mod_l(&k, sig, 32, pk, 32, m, mn);
	/* S = r + k a mod L */
	nn_mulmod(&S, &k, &a, &ed_L);
	nn_addmod(&S, &S, &r, &ed_L);
	nn_to_bytes_le(sig + 32, 32, &S);
}

int
ed25519_verify(const unsigned char *pk, const unsigned char *m, size_t mn,
    const unsigned char *sig)
{
	const struct mf *F = f25519();
	struct edpt A, R, SB, kA, RkA;
	nn S, k;
	unsigned char e1[32], e2[32];

	ed_setup();
	if (!ed_decode(F, &A, pk) || !ed_decode(F, &R, sig))
		return 0;
	nn_from_bytes_le(&S, sig + 32, 32);
	if (nn_cmp(&S, &ed_L) >= 0)
		return 0;
	ed_hash_mod_l(&k, sig, 32, pk, 32, m, mn);
	ed_mul(F, &SB, &ed_B, &S);
	ed_mul(F, &kA, &A, &k);
	ed_add(F, &RkA, &R, &kA);
	ed_encode(F, e1, &SB);
	ed_encode(F, e2, &RkA);
	return !memcmp(e1, e2, 32);
}

/* the Edwards base point B is on -x^2 + y^2 = 1 + d x^2 y^2 and L B = 0 */
int
ed25519_check_base(void)
{
	const struct mf *F = f25519();
	struct edpt T;
	u64 x2[FW], y2[FW], l[FW], r[FW], one[FW];
	unsigned char e[32], z[32];

	ed_setup();
	mf_sqr(F, x2, ed_B.X);
	mf_sqr(F, y2, ed_B.Y);
	mf_sub(F, l, y2, x2);
	mf_mul(F, r, x2, y2);
	mf_mul(F, r, r, ed_d);
	mf_one(F, one);
	mf_add(F, r, r, one);
	if (!mf_eq(F, l, r))
		return 0;
	ed_mul(F, &T, &ed_B, &ed_L);
	ed_encode(F, e, &T);
	memset(z, 0, sizeof z);
	z[0] = 1;                        /* the identity (0, 1) */
	return !memcmp(e, z, 32);
}
