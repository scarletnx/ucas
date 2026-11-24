/* Kernels of tests/cfpop.ucas. */
#include <stdint.h>

/* the bits of x, counted in a loop: popcount(x) */
uint32_t pop_loop(uint32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 32; i++)
		c += (x >> i) & 1;
	return c;
}

/* the same sum written out */
uint32_t pop_sum(uint32_t x)
{
	return (x & 1) + (x >> 1 & 1) + (x >> 2 & 1) + (x >> 3 & 1) +
	    (x >> 4 & 1) + (x >> 5 & 1) + (x >> 6 & 1) + (x >> 7 & 1) +
	    (x >> 8 & 1) + (x >> 9 & 1) + (x >> 10 & 1) + (x >> 11 & 1) +
	    (x >> 12 & 1) + (x >> 13 & 1) + (x >> 14 & 1) + (x >> 15 & 1) +
	    (x >> 16 & 1) + (x >> 17 & 1) + (x >> 18 & 1) + (x >> 19 & 1) +
	    (x >> 20 & 1) + (x >> 21 & 1) + (x >> 22 & 1) + (x >> 23 & 1) +
	    (x >> 24 & 1) + (x >> 25 & 1) + (x >> 26 & 1) + (x >> 27 & 1) +
	    (x >> 28 & 1) + (x >> 29 & 1) + (x >> 30 & 1) + (x >> 31 & 1);
}

/* the bits of a signed x: of its pattern, the sign's included */
uint32_t pop_signed(int32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 32; i++)
		c += (x >> i) & 1;
	return c;
}

/* two bits a step, not one: no popcount of x */
uint32_t pop_wide(uint32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 32; i++)
		c += (x >> i) & 3;
	return c;
}

/* the two lowest bits alone */
uint32_t pop_two(uint32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 2; i++)
		c += (x >> i) & 1;
	return c;
}

/* it leaves at the first zero: the ones below it, not all of them */
uint32_t pop_until(uint32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 8; i++) {
		if (!((x >> i) & 1))
			break;
		c += (x >> i) & 1;
	}
	return c;
}
