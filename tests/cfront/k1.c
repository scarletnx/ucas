#include <stdint.h>

/* count bits the slow way */
uint32_t popcount_loop(uint32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 32; i++)
		c += (x >> i) & 1;
	return c;
}

int classify(int c)
{
	switch (c) {
	case 'a': case 'e': case 'i': case 'o': case 'u':
		return 1;
	case 'y':
		return 1;
	default:
		return 0;
	}
}

uint32_t clamp_avg(uint8_t a, uint8_t b)
{
	uint32_t s = ((uint32_t)a + b + 1) >> 1;
	if (s > 255)
		s = 255;
	return s & 0xff;
}

uint64_t div64(uint64_t x, uint64_t y)
{
	return (x & 0xffff) / ((y & 0xff) + 1);
}
