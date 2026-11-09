#include <stdint.h>
#include <stddef.h>

#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))

uint32_t fmix32(uint32_t h)
{
	h ^= h >> 16;
	h *= 0x85ebca6bu;
	h ^= h >> 13;
	h *= 0xc2b2ae35u;
	h ^= h >> 16;
	return h;
}

uint32_t oaat_step(uint32_t h, uint8_t c)
{
	h += c;
	h += h << 10;
	h ^= h >> 6;
	return h;
}

uint32_t rotmix(uint32_t a, uint32_t b)
{
	return ROTL32(a ^ b, 13) * 5u + 0xe6546b64u;
}

int ilog2_loop(uint32_t x)
{
	int r = 0;
	while (x >>= 1)
		r++;
	return r;
}

int clz_loop(uint32_t x)
{
	int n = 0;
	if (x == 0)
		return 32;
	while (!(x & 0x80000000u)) {
		x <<= 1;
		n++;
	}
	return n;
}

uint32_t lowest_bit_loop(uint32_t x)
{
	uint32_t m = 1;
	if (x == 0)
		return 0;
	while (!(x & m))
		m <<= 1;
	return m;
}

int parity_loop(uint32_t x)
{
	int p = 0;
	while (x) {
		p ^= 1;
		x &= x - 1;
	}
	return p;
}

uint32_t next_pow2(uint32_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

uint8_t sat_add_u8(uint8_t a, uint8_t b)
{
	unsigned s = a + b;
	return s > 255 ? 255 : (uint8_t)s;
}

int clamp_byte(int v)
{
	if (v < 0)
		return 0;
	if (v > 255)
		return 255;
	return v;
}

unsigned digit_sum(unsigned x)
{
	x &= 1023;
	return x / 100 + (x / 10) % 10 + x % 10;
}

int half_nonneg(int x)
{
	if (x < 0)
		return 0;
	return x / 4 + x % 4;
}

static const uint8_t days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

int month_days(int m)
{
	switch (m) {
	case 1: return 31;
	case 2: return 28;
	case 3: return 31;
	case 4: return 30;
	case 5: return 31;
	case 6: return 30;
	case 7: return 31;
	case 8: return 31;
	case 9: return 30;
	case 10: return 31;
	case 11: return 30;
	case 12: return 31;
	}
	return 0;
}

int opcode_len(int op)
{
	switch (op) {
	case 0: return 1;
	case 1: return 3;
	case 2: return 5;
	case 3: return 7;
	case 4: return 9;
	case 5: return 11;
	default: return -1;
	}
}

int is_hex(int c)
{
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
	    (c >= 'A' && c <= 'F');
}

uint32_t tri(uint8_t n)
{
	uint32_t s = 0;
	for (uint32_t i = 0; i < n && i < 64; i++)
		s += i;
	return s;
}

void brighten(uint8_t *dst, const uint8_t *src, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned v = src[i] * 3u / 2u + 10u;
		dst[i] = v > 255 ? 255 : (uint8_t)v;
	}
}

size_t my_strlen(const char *s)
{
	size_t n = 0;
	while (s[n])
		n++;
	return n;
}

int table_month(int m)
{
	return days[(m - 1) & 15 & 11];
}
