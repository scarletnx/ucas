#include <stdint.h>
static uint32_t scale(uint32_t x, uint32_t d)
{
	return x / d;
}
uint32_t use(uint8_t a, uint8_t b)
{
	return scale(a, 3) + scale(b, 5);
}
static int bucket(int v)
{
	if (v < 0)
		return 0;
	if (v > 1000)
		return 9;
	return v / 100;
}
int use2(uint8_t x)
{
	return bucket(x * 2) + bucket(x);
}
