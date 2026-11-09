/* cfront test: loops with symbolic trip counts, summed in closed form */
#include <stdint.h>

uint32_t tri32(uint32_t n)
{
	uint32_t s = 0;
	for (uint32_t i = 0; i < n; i++)
		s += i;
	return s;
}

uint64_t sumsq(int n)
{
	uint64_t s = 0;
	for (int i = 1; i < n; i++)
		s += (uint64_t)i * i;
	return s;
}

int lin(int a, int b, int k)
{
	int s = 7;
	for (int i = a; i < b; i++)
		s += 3 * i + k;
	return s;
}

long two(long n)
{
	long a = 0, b = 0;
	for (long i = 0; n > i; i += 1) {
		a += i;
		b -= 2 * i * i;
	}
	return a + b;
}

/* not a polynomial summand: stays a loop (not lowered: too many trips) */
uint32_t bits_below(uint32_t n)
{
	uint32_t s = 0;
	for (uint32_t i = 0; i < n; i++)
		s += i & 1;
	return s;
}
