/* The bridge (src/cf_bridge.c, cf_synth.c): kernels for ucas's engines
 * behind the C frontend.  tests/cfront.ucas runs them per target. */
#include <stdint.h>
#include <stddef.h>

/* ring: a linear congruence, 3x + 5 = 2 (mod 7) <=> x = 6 (mod 7) */
int cong(uint16_t x) { return (3 * x + 5) % 7 == 2; }

/* ring: x * 7 == 21 (mod 2^32) <=> x == 21 * 7^-1 = 3 */
int inv_eq(uint32_t x) { return x * 7u == 21u; }

/* ring: two congruences joined by the CRT */
int crt2(uint32_t x) { return x % 3 == 1 && x % 5 == 2; }

/* dio: x*x = 3 (mod 4) has no solution */
int never_sq(uint32_t x) { return x * x % 4 == 3; }

/* dio: a^2 + b^2 = 21 has none: 3 divides 21 to an odd power */
int two_sq(int16_t a, int16_t b) { return a * a + b * b == 21; }

/* dio: y^2 = x^3 - x has rank 0; its rational points have y = 0 */
int ec_pts(int16_t x, int16_t y) { return y != 0 && y * y == x * x * x - x; }

/* alg: a difference of squares, one multiply instead of two */
uint32_t dsq(uint32_t a, uint32_t b) { return a * a - b * b; }

/* alg: (x^2 - 1)/(x - 1) cancelled: nothing wraps */
int32_t cancel_q(int16_t x) { return (x * x - 1) / (x - 1); }

/* loops: a sum of degree 4 */
uint64_t sum4(uint64_t n)
{
	uint64_t s = 0;
	for (uint64_t i = 0; i < n; i++)
		s += i * i * i * i;
	return s;
}

/* loops: a geometric sum */
uint32_t geo3(uint32_t n)
{
	uint32_t s = 0, p = 1;
	for (uint32_t i = 0; i < n; i++) {
		s += p;
		p *= 3;
	}
	return s;
}

/* loops into integrals: a left Riemann sum of x^2 on [0, 1] */
double riemann(int n)
{
	double h = 1.0 / n, s = 0;
	for (int i = 0; i < n; i++) {
		double x = i * h;
		s += x * x * h;
	}
	return s;
}

/* synth: maps of the bytes of a word */
uint64_t rev_each(uint64_t x)
{
	x = (x & 0xF0F0F0F0F0F0F0F0ull) >> 4 | (x & 0x0F0F0F0F0F0F0F0Full) << 4;
	x = (x & 0xCCCCCCCCCCCCCCCCull) >> 2 | (x & 0x3333333333333333ull) << 2;
	return (x & 0xAAAAAAAAAAAAAAAAull) >> 1 | (x & 0x5555555555555555ull) << 1;
}

uint64_t pop_each(uint64_t x)
{
	x = x - ((x >> 1) & 0x5555555555555555ull);
	x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
	return (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
}

/* synth in the vectoriser: per-byte shift, rotate, bit reversal, and a
 * bitwise mix of a rotate with the byte (no 8-bit shifts in SSE/AVX) */
void shl3_bytes(uint8_t *restrict d, const uint8_t *restrict s, size_t n)
{
	for (size_t i = 0; i < n; i++)
		d[i] = (uint8_t)(s[i] << 3);
}

void rotr2_bytes(uint8_t *restrict d, const uint8_t *restrict s, size_t n)
{
	for (size_t i = 0; i < n; i++)
		d[i] = (uint8_t)((s[i] >> 2) | (s[i] << 6));
}

void rev_bytes(uint8_t *restrict d, const uint8_t *restrict s, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		uint8_t x = s[i];
		x = (uint8_t)((x & 0xF0) >> 4 | (x & 0x0F) << 4);
		x = (uint8_t)((x & 0xCC) >> 2 | (x & 0x33) << 2);
		d[i] = (uint8_t)((x & 0xAA) >> 1 | (x & 0x55) << 1);
	}
}

void mixrot(uint8_t *restrict d, const uint8_t *restrict s, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		uint8_t x = s[i], r = (uint8_t)((x >> 2) | (x << 6));
		d[i] = (uint8_t)((x & r) | (~x & 0x5a));
	}
}

/* gates: a 3-input bit function (majority) */
uint32_t maj(uint32_t a, uint32_t b, uint32_t c)
{
	return (a & b) | (a & c) | (b & c) | (a & b & c);
}

/* gf2: a bit gather (pext) */
uint32_t gather_bits(uint32_t x) { return ((x >> 3) & 7) | ((x >> 6) & 0x38); }

/* gf2: one byte of the reflected CRC-32 (0xEDB88320), bit by bit */
uint32_t crc8step(uint32_t crc, uint8_t b)
{
	crc ^= b;
	for (int k = 0; k < 8; k++)
		crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1)));
	return crc;
}

/* gf2: a 32-bit word of the same CRC: pclmul */
uint32_t crc32step(uint32_t crc, uint32_t w)
{
	crc ^= w;
	for (int k = 0; k < 32; k++)
		crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1)));
	return crc;
}

/* ring: magic numbers for a 64-bit division by a constant */
uint64_t div10(uint64_t x) { return x / 10; }
