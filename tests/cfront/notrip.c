/* Kernels of tests/cfnotrip.ucas: summands that are undefined in C at some
 * inputs, in loops that may run no iteration. */
#include <stdint.h>

/* 7 with m = 0, whatever d is */
uint32_t div_sum(uint32_t x, uint32_t d, uint32_t m)
{
	uint32_t s = 7;
	for (uint32_t i = 0; i < m; i++)
		s += x / d;
	return s;
}

/* 7 with m = 0, whatever n is */
uint32_t shr_sum(uint32_t x, uint32_t n, uint32_t m)
{
	uint32_t s = 7;
	for (uint32_t i = 0; i < m; i++)
		s += (x >> n) * i;
	return s;
}

/* no condition on the summand: no test of the trip count */
uint32_t plain_sum(uint32_t x, uint32_t m)
{
	uint32_t s = 7;
	for (uint32_t i = 0; i < m; i++)
		s += x * i;
	return s;
}

/* the division before the loop is no condition of the summand */
uint32_t div_then_sum(uint32_t x, uint32_t d, uint32_t m)
{
	uint32_t q = x / d, s = 7;
	for (uint32_t i = 0; i < m; i++)
		s += q * i;
	return s;
}
