/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecsha.c - SHA-256, SHA-384/512 (FIPS 180-4) and HMAC (RFC 2104), for
 * ECDSA, RFC 6979 nonces and EdDSA.
 */
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "ecff.h"

static const uint32_t K256[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
	0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
	0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
	0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
	0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
	0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define ROR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

static void
sha256_block(uint32_t *h, const unsigned char *p)
{
	uint32_t w[64], a, b, c, d, e, f, g, hh, t1, t2;
	int i;

	for (i = 0; i < 16; i++)
		w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
		    (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
	for (i = 16; i < 64; i++) {
		uint32_t s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^
		    (w[i - 15] >> 3);
		uint32_t s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^
		    (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = h[0]; b = h[1]; c = h[2]; d = h[3];
	e = h[4]; f = h[5]; g = h[6]; hh = h[7];
	for (i = 0; i < 64; i++) {
		t1 = hh + (ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25)) +
		    ((e & f) ^ (~e & g)) + K256[i] + w[i];
		t2 = (ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22)) +
		    ((a & b) ^ (a & c) ^ (b & c));
		hh = g; g = f; f = e; e = d + t1;
		d = c; c = b; b = a; a = t1 + t2;
	}
	h[0] += a; h[1] += b; h[2] += c; h[3] += d;
	h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

static const uint64_t K512[80] = {
	0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
	0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
	0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
	0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
	0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
	0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
	0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
	0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
	0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
	0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
	0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
	0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
	0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
	0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
	0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
	0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
	0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
	0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
	0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
	0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
	0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
	0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
	0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
	0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
	0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
	0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
	0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

static void
sha512_block(uint64_t *h, const unsigned char *p)
{
	uint64_t w[80], a, b, c, d, e, f, g, hh, t1, t2;
	int i, j;

	for (i = 0; i < 16; i++) {
		w[i] = 0;
		for (j = 0; j < 8; j++)
			w[i] = w[i] << 8 | p[8 * i + j];
	}
	for (i = 16; i < 80; i++) {
		uint64_t s0 = ROR64(w[i - 15], 1) ^ ROR64(w[i - 15], 8) ^
		    (w[i - 15] >> 7);
		uint64_t s1 = ROR64(w[i - 2], 19) ^ ROR64(w[i - 2], 61) ^
		    (w[i - 2] >> 6);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = h[0]; b = h[1]; c = h[2]; d = h[3];
	e = h[4]; f = h[5]; g = h[6]; hh = h[7];
	for (i = 0; i < 80; i++) {
		t1 = hh + (ROR64(e, 14) ^ ROR64(e, 18) ^ ROR64(e, 41)) +
		    ((e & f) ^ (~e & g)) + K512[i] + w[i];
		t2 = (ROR64(a, 28) ^ ROR64(a, 34) ^ ROR64(a, 39)) +
		    ((a & b) ^ (a & c) ^ (b & c));
		hh = g; g = f; f = e; e = d + t1;
		d = c; c = b; b = a; a = t1 + t2;
	}
	h[0] += a; h[1] += b; h[2] += c; h[3] += d;
	h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

/* incremental interface */
void
sha_init(struct sha *s, int alg)
{
	static const uint32_t i256[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
		0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
	static const uint64_t i512[8] = {0x6a09e667f3bcc908ULL,
		0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
		0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL,
		0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL,
		0x5be0cd19137e2179ULL};
	static const uint64_t i384[8] = {0xcbbb9d5dc1059ed8ULL,
		0x629a292a367cd507ULL, 0x9159015a3070dd17ULL,
		0x152fecd8f70e5939ULL, 0x67332667ffc00b31ULL,
		0x8eb44a8768581511ULL, 0xdb0c2e0d64f98fa7ULL,
		0x47b5481dbefa4fa4ULL};
	int i;

	memset(s, 0, sizeof *s);
	s->alg = alg;
	if (alg == 256) {
		for (i = 0; i < 8; i++)
			s->h32[i] = i256[i];
		s->bsz = 64;
		s->outlen = 32;
	} else {
		for (i = 0; i < 8; i++)
			s->h64[i] = alg == 384 ? i384[i] : i512[i];
		s->bsz = 128;
		s->outlen = alg == 384 ? 48 : 64;
	}
}

void
sha_update(struct sha *s, const void *data, size_t n)
{
	const unsigned char *p = data;

	s->total += n;
	while (n > 0) {
		size_t take = (size_t)s->bsz - s->blen;
		if (take > n)
			take = n;
		memcpy(s->buf + s->blen, p, take);
		s->blen += take;
		p += take;
		n -= take;
		if (s->blen == (size_t)s->bsz) {
			if (s->bsz == 64)
				sha256_block(s->h32, s->buf);
			else
				sha512_block(s->h64, s->buf);
			s->blen = 0;
		}
	}
}

void
sha_final(struct sha *s, unsigned char *out)
{
	unsigned char pad[256];
	size_t lenb = s->bsz == 64 ? 8 : 16, padn, i;
	uint64_t bits = s->total * 8;

	padn = (size_t)s->bsz - ((s->blen + 1 + lenb) % (size_t)s->bsz);
	if (padn == (size_t)s->bsz)
		padn = 0;
	memset(pad, 0, sizeof pad);
	pad[0] = 0x80;
	for (i = 0; i < 8; i++)
		pad[1 + padn + lenb - 1 - i] = (unsigned char)(bits >> (8 * i));
	sha_update(s, pad, 1 + padn + lenb);
	if (s->bsz == 64)
		for (i = 0; i < 8; i++) {
			out[4 * i] = (unsigned char)(s->h32[i] >> 24);
			out[4 * i + 1] = (unsigned char)(s->h32[i] >> 16);
			out[4 * i + 2] = (unsigned char)(s->h32[i] >> 8);
			out[4 * i + 3] = (unsigned char)s->h32[i];
		}
	else
		for (i = 0; i < (size_t)s->outlen; i++)
			out[i] = (unsigned char)(s->h64[i / 8] >> (56 - 8 * (i % 8)));
}

void
sha_hash(int alg, const void *data, size_t n, unsigned char *out)
{
	struct sha s;

	sha_init(&s, alg);
	sha_update(&s, data, n);
	sha_final(&s, out);
}

int
sha_outlen(int alg)
{
	return alg == 256 ? 32 : alg == 384 ? 48 : 64;
}

/* HMAC with the given SHA-2; out has outlen bytes */
void
hmac(int alg, const unsigned char *key, size_t kn, const unsigned char *msg,
    size_t mn, unsigned char *out)
{
	struct sha s;
	unsigned char k[128], ip[128], op[128], inner[64];
	int bsz = alg == 256 ? 64 : 128, ol = sha_outlen(alg), i;

	memset(k, 0, sizeof k);
	if (kn > (size_t)bsz)
		sha_hash(alg, key, kn, k);
	else
		memcpy(k, key, kn);
	for (i = 0; i < bsz; i++) {
		ip[i] = k[i] ^ 0x36;
		op[i] = k[i] ^ 0x5c;
	}
	sha_init(&s, alg);
	sha_update(&s, ip, (size_t)bsz);
	sha_update(&s, msg, mn);
	sha_final(&s, inner);
	sha_init(&s, alg);
	sha_update(&s, op, (size_t)bsz);
	sha_update(&s, inner, (size_t)ol);
	sha_final(&s, out);
}
