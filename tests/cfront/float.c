/* cfront test: float and double */
#include <stddef.h>

float half(float x)
{
	return x * 0.5;
}

float quarter(float x)
{
	return x / 4.0f;
}

double dscale(double x, double y)
{
	return (x + y) / 8.0;
}

float mixf(float a, float b)
{
	double t = (double)a * b;
	return (float)t;
}

int roundtrip(unsigned char c)
{
	float f = c;
	return (int)f;
}

int cmpf(short a, short b)
{
	return (float)a < (float)b;
}

/* not exact: 1/3 is not a power of two, (b - a) * t + a is not fused */
float lerp(float a, float b, float t)
{
	return a + (b - a) * t / 3.0f;
}

int to_int(float x)
{
	return (int)x;
}

void scale(float *restrict d, const float *restrict s, size_t n)
{
	for (size_t i = 0; i < n; i++)
		d[i] = s[i] / 2.0f;
}
