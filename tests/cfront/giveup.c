/* Kernels of tests/cfpop.ucas: loops with a summand that has a closed form
 * beside one that has none.  They are unrolled, each sum added once. */
#include <stdint.h>

/* s is 6 */
uint32_t sum_and_bits(uint32_t x)
{
	uint32_t s = 0, c = 0;
	for (int i = 0; i < 4; i++) {
		s += i;
		c += (x >> i) & 1;
	}
	return s * 100 + c;
}

/* p is 81 */
uint32_t power_and_bits(uint32_t x)
{
	uint32_t p = 1, c = 0;
	for (int i = 0; i < 4; i++) {
		p *= 3;
		c += (x >> i) & 1;
	}
	return p * 100 + c;
}

/* p is 81, s is 120 */
uint32_t geometric_and_bits(uint32_t x)
{
	uint32_t p = 1, s = 0, c = 0;
	for (int i = 0; i < 4; i++) {
		p *= 3;
		s += p;
		c += (x >> i) & 1;
	}
	return (p + s) * 100 + c;
}

/* the first loop keeps its closed form, the second is unrolled */
uint32_t sum_then_bits(uint32_t x, uint32_t m)
{
	uint32_t s = 0, c = 0;
	for (uint32_t i = 0; i < m; i++)
		s += i;
	for (int i = 0; i < 4; i++) {
		s += i;
		c += (x >> i) & 1;
	}
	return s * 100 + c;
}

/* the sum and the power are parameters, read for the first time there */
uint32_t params_and_bits(uint32_t s, uint32_t p, uint32_t x)
{
	uint32_t c = 0;
	for (int i = 0; i < 4; i++) {
		s += i;
		p *= 3;
		c += (x >> i) & 1;
	}
	return s * 100 + p + c;
}
