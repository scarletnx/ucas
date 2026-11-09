/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecnamed.c - named curves (SEC 2, FIPS 186-4, RFC 7748/8032, pairing
 * curves), as data.  Every entry is checked when used: the generator is
 * on the curve, n is prime (BPSW) and n G = O.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

const struct ecnamed ecnamed_tab[] = {
	{"secp256k1", "secp256k1,k256,bitcoin", "SEC 2 (Koblitz, j = 0)",
	    ECN_W, "2^256 - 2^32 - 977", "0", "7",
	    "0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
	    "0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
	    "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
	    "1", NULL, NULL, NULL, NULL},
	{"P-256", "p256,p-256,secp256r1,prime256v1,nistp256",
	    "FIPS 186-4 / SEC 2 secp256r1", ECN_W,
	    "2^256 - 2^224 + 2^192 + 2^96 - 1", "-3",
	    "0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
	    "0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
	    "0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
	    "0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
	    "1", NULL, NULL, NULL, NULL},
	{"P-384", "p384,p-384,secp384r1,nistp384", "FIPS 186-4 / SEC 2 secp384r1",
	    ECN_W, "2^384 - 2^128 - 2^96 + 2^32 - 1", "-3",
	    "0xB3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE8141120314088F5013875A"
	    "C656398D8A2ED19D2A85C8EDD3EC2AEF",
	    "0xAA87CA22BE8B05378EB1C71EF320AD746E1D3B628BA79B9859F741E082542A38"
	    "5502F25DBF55296C3A545E3872760AB7",
	    "0x3617DE4A96262C6F5D9E98BF9292DC29F8F41DBD289A147CE9DA3113B5F0B8C0"
	    "0A60B1CE1D7E819D7A431D7C90EA0E5F",
	    "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF"
	    "581A0DB248B0A77AECEC196ACCC52973",
	    "1", NULL, NULL, NULL, NULL},
	{"P-521", "p521,p-521,secp521r1,nistp521", "FIPS 186-4 / SEC 2 secp521r1",
	    ECN_W, "2^521 - 1", "-3",
	    "0x0051953EB9618E1C9A1F929A21A0B68540EEA2DA725B99B315F3B8B489918EF1"
	    "09E156193951EC7E937B1652C0BD3BB1BF073573DF883D2C34F1EF451FD46B503F00",
	    "0x00C6858E06B70404E9CD9E3ECB662395B4429C648139053FB521F828AF606B4D"
	    "3DBAA14B5E77EFE75928FE1DC127A2FFA8DE3348B3C1856A429BF97E7E31C2E5BD66",
	    "0x011839296A789A3BC0045C8A5FB42C7D1BD998F54449579B446817AFBD17273E"
	    "662C97EE72995EF42640C550B9013FAD0761353C7086A272C24088BE94769FD16650",
	    "0x01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
	    "FA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386409",
	    "1", NULL, NULL, NULL, NULL},
	{"Curve25519", "curve25519,x25519,25519",
	    "RFC 7748: v^2 = u^3 + 486662 u^2 + u", ECN_MONT, "2^255 - 19",
	    "486662", "1", "9",
	    "14781619447589544791020593568409986887264606134616475288964881837755586237401",
	    "2^252 + 27742317777372353535851937790883648493", "8", NULL, NULL,
	    NULL, NULL},
	{"Ed25519", "ed25519,edwards25519",
	    "RFC 8032: -x^2 + y^2 = 1 + d x^2 y^2, d = -121665/121666", ECN_ED,
	    "2^255 - 19", "-1",
	    "37095705934669439343138083508754565189542113879843219016388785533085940283555",
	    "15112221349535400772501151409588531511454012693041857206046113283949847762202",
	    "46316835694926478169428394003475163141307993866256225615783033603165251855960",
	    "2^252 + 27742317777372353535851937790883648493", "8", NULL, NULL,
	    NULL, NULL},
	{"Curve448", "curve448,x448,448",
	    "RFC 7748: v^2 = u^3 + 156326 u^2 + u", ECN_MONT,
	    "2^448 - 2^224 - 1", "156326", "1", "5",
	    "355293926785568175264127502063783334808976399387714271831880898435169088786967410002932673765864550910142774147268105838985595290606362",
	    "2^446 - 13818066809895115352007386748515426880336692474882178609894547503885",
	    "4", NULL, NULL, NULL, NULL},
	{"BN254", "bn254,alt_bn128,bn128,bn256",
	    "Barreto-Naehrig, embedding degree 12 (Ethereum alt_bn128)", ECN_W,
	    "21888242871839275222246405745257275088696311157297823662689037894645226208583",
	    "0", "3", "1", "2",
	    "21888242871839275222246405745257275088548364400416034343698204186575808495617",
	    "1",
	    /* G2 on y^2 = x^3 + 3/(9 + u) over F_p2 = F_p[u]/(u^2 + 1) */
	    "10857046999023057135944570762232829481370756359578518086990519993285655852781,"
	    "11559732032986387107991004021392285783925812861821192530917403151452391805634",
	    "8495653923123431417604973247489272438418190587263600148770280649306958101930,"
	    "4082367875863433681332203403145435568316851327593401208105741076214120093531",
	    "4965661367192848881", "9"},
	{"BLS12-381", "bls12_381,bls12-381,bls12381",
	    "Barreto-Lynn-Scott, embedding degree 12 (Zcash, Ethereum 2)", ECN_W,
	    "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f624"
	    "1eabfffeb153ffffb9feffffffffaaab", "0", "4",
	    "0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac58"
	    "6c55e83ff97a1aeffb3af00adb22c6bb",
	    "0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3ed"
	    "d03cc744a2888ae40caa232946c5e7e1",
	    "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
	    "0x396c8c005555e1568c00aaab0000aaab",
	    "0x024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d177"
	    "0bac0326a805bbefd48056c8c121bdb8,"
	    "0x13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049"
	    "334cf11213945d57e5ac7d055d042b7e",
	    "0x0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c"
	    "923ac9cc3baca289e193548608b82801,"
	    "0x0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab"
	    "3f370d275cec1da1aaa9075ff05f79be",
	    "-0xd201000000010000", "1"},
	/* small textbook curves */
	{"toy97", "toy97", "y^2 = x^3 + 2x + 3 over F_97 (a textbook example)",
	    ECN_W, "97", "2", "3", "3", "6", "5", "20", NULL, NULL, NULL, NULL},
	{NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
	    NULL, NULL, NULL}
};

const struct ecnamed *
ecnamed_find(const char *name)
{
	int i;
	char low[64];
	size_t k;

	for (k = 0; name[k] && k + 1 < sizeof low; k++)
		low[k] = (char)(name[k] >= 'A' && name[k] <= 'Z' ?
		    name[k] - 'A' + 'a' : name[k]);
	low[k] = 0;
	for (i = 0; ecnamed_tab[i].name; i++) {
		const char *a = ecnamed_tab[i].aliases;
		size_t L = strlen(low);
		if (!strcasecmp(ecnamed_tab[i].name, name))
			return &ecnamed_tab[i];
		while (a && *a) {
			const char *c = strchr(a, ',');
			size_t n = c ? (size_t)(c - a) : strlen(a);
			if (n == L && !strncmp(a, low, n))
				return &ecnamed_tab[i];
			a = c ? c + 1 : NULL;
		}
	}
	return NULL;
}

static void
num(const char *s, nn *out)
{
	zz z;

	if (!zz_parse(s, &z, NULL) || z.neg)
		fail("ecff: bad constant %s in the named-curve table", s);
	*out = z.m;
}

static void
fnum(const struct mf *F, const char *s, u64 *out)
{
	zz z;

	if (!zz_parse(s, &z, NULL))
		fail("ecff: bad constant %s", s);
	mf_from_zz(F, out, &z);
}

/* the Weierstrass model of a named curve (Montgomery and Edwards curves
 * through their maps to v^2 = u^3 + A u^2 + u, then u = x - A/3) */
void
ecnamed_load(const struct ecnamed *N, struct ecload *L)
{
	nn p;
	u64 a[FW], b[FW], t[FW], A[FW], i3[FW], u[FW], v[FW];

	memset(L, 0, sizeof *L);
	L->N = N;
	num(N->p, &p);
	if (!mf_init(&L->F, &p))
		fail("ecff: bad modulus for %s", N->name);
	mf_prime_setup(&L->F);
	num(N->n, &L->n);
	num(N->h, &L->h);
	nn_mul(&L->order, &L->n, &L->h);
	if (N->model == ECN_W) {
		fnum(&L->F, N->a, a);
		fnum(&L->F, N->b, b);
		ecp_set(&L->E, &L->F, a, b);
		fnum(&L->F, N->gx, L->G.x);
		fnum(&L->F, N->gy, L->G.y);
		L->G.inf = 0;
		return;
	}
	mf_from_u64(&L->F, i3, 3);
	mf_inv(&L->F, i3, i3, NULL);
	if (N->model == ECN_MONT) {
		/* v^2 = u^3 + A u^2 + u (B = 1) */
		fnum(&L->F, N->a, A);
		fnum(&L->F, N->gx, u);
		fnum(&L->F, N->gy, v);
	} else {
		/* Ed25519 (a = -1): u = (1 + y)/(1 - y), v = sqrt(-486664) u / x
		 * is Curve25519; A = 2(a + d)/(a - d), B = 4/(a - d) */
		u64 d[FW], ae[FW], x[FW], y[FW], one[FW], num_[FW], den[FW],
		    Bm[FW], s[FW];
		fnum(&L->F, N->a, ae);
		fnum(&L->F, N->b, d);
		fnum(&L->F, N->gx, x);
		fnum(&L->F, N->gy, y);
		mf_one(&L->F, one);
		mf_add(&L->F, num_, ae, d);
		mf_dbl(&L->F, num_, num_);
		mf_sub(&L->F, den, ae, d);
		mf_inv(&L->F, den, den, NULL);
		mf_mul(&L->F, A, num_, den);
		mf_from_u64(&L->F, Bm, 4);
		mf_mul(&L->F, Bm, Bm, den);
		mf_add(&L->F, u, one, y);
		mf_sub(&L->F, t, one, y);
		mf_inv(&L->F, t, t, NULL);
		mf_mul(&L->F, u, u, t);
		mf_inv(&L->F, t, x, NULL);
		mf_mul(&L->F, v, u, t);
		/* B v^2 = u^3 + A u^2 + u: scale v by sqrt(B) */
		if (!mf_sqrt(&L->F, s, Bm))
			fail("ecff: Ed25519 map: B is not a square");
		mf_mul(&L->F, v, v, s);
	}
	/* x = u + A/3, y = v: y^2 = x^3 + (1 - A^2/3) x + (2A^3/27 - A/3) */
	mf_sqr(&L->F, t, A);
	mf_mul(&L->F, t, t, i3);
	mf_one(&L->F, a);
	mf_sub(&L->F, a, a, t);
	mf_mul(&L->F, t, t, A);
	mf_mul(&L->F, t, t, i3);
	mf_mul(&L->F, t, t, i3);
	mf_dbl(&L->F, b, t);
	mf_mul(&L->F, t, A, i3);
	mf_sub(&L->F, b, b, t);
	ecp_set(&L->E, &L->F, a, b);
	mf_add(&L->F, L->G.x, u, t);
	mf_copy(&L->F, L->G.y, v);
	L->G.inf = 0;
	mf_copy(&L->F, L->A, A);
	L->has_A = 1;
}

/* the checks every use of a named curve repeats */
int
ecnamed_check(struct ecload *L, char *why, size_t n)
{
	struct ecp_pt T;

	if (!L->F.prime) {
		snprintf(why, n, "p is not prime");
		return 0;
	}
	if (!ecp_on_curve(&L->E, &L->G)) {
		snprintf(why, n, "the generator is not on the curve");
		return 0;
	}
	if (!nn_isprime(&L->n)) {
		snprintf(why, n, "n is not prime");
		return 0;
	}
	ecp_mul(&L->E, &T, &L->G, &L->n);
	if (!T.inf) {
		snprintf(why, n, "n G != O");
		return 0;
	}
	snprintf(why, n, "G on the curve, n prime (BPSW), n G = O");
	return 1;
}
