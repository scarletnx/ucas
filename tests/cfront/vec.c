/* cfront test: loops for the x86 vectoriser (cfx86.c): element-wise,
 * reductions, widening and narrowing, and loops it must refuse */
#include <stdint.h>
#include <stddef.h>

void saxpy(float *restrict y, const float *restrict x, float a, int n)
{
	for (int i = 0; i < n; i++)
		y[i] = a * x[i] + y[i];
}

void dscale(double *restrict y, const double *restrict x, int n)
{
	for (int i = 0; i < n; i++)
		y[i] = x[i] * 0.5 - 1.0;
}

void add32(int32_t *restrict c, const int32_t *restrict a, const int32_t *restrict b, int n)
{
	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i] * 3;
}

void avg8(uint8_t *restrict d, const uint8_t *restrict a, const uint8_t *restrict b, size_t n)
{
	for (size_t i = 0; i < n; i++)
		d[i] = (uint8_t)((a[i] + b[i] + 1) >> 1);
}

void sat16(int16_t *d, const int16_t *s, int n)
{
	for (int i = 0; i < n; i++) {
		int v = s[i] * 2;
		d[i] = (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
	}
}

uint32_t xorsum(const uint32_t *a, int n)
{
	uint32_t x = 0;
	for (int i = 0; i < n; i++)
		x ^= a[i] >> 3;
	return x;
}

int maxv(const int *a, int n)
{
	int m = -2147483647 - 1;
	for (int i = 0; i < n; i++)
		m = a[i] > m ? a[i] : m;
	return m;
}

int64_t sum64(const int64_t *a, int n)
{
	int64_t s = 0;
	for (int i = 0; i < n; i++)
		s += a[i];
	return s;
}

void iota(uint32_t *d, int n)
{
	for (int i = 0; i < n; i++)
		d[i] = (uint32_t)i * 2u + 1u;
}

void shift_left(int *a, int n)
{
	for (int i = 0; i < n - 1; i++)
		a[i] = a[i + 1];
}

void prefix(int *a, int n)
{
	for (int i = 1; i < n; i++)
		a[i] = a[i - 1] + a[i];
}

float fsum(const float *a, int n)
{
	float s = 0;
	for (int i = 0; i < n; i++)
		s += a[i];
	return s;
}

void after(int *a, int *b, int n)
{
	for (int i = 0; i < n; i++) {
		a[i] = b[i] + 1;
		b[i] = a[i] * 2;
	}
}
void inplace2(int *restrict a, int n)
{
	for (int i = 0; i < n; i++) {
		a[i] = a[i] + 1;
		int t = a[i];
		(void)t;
	}
}

/* refused: a gather (not vectorised yet), a float sum (reassociation
 * rounds differently) */
void gath(int *restrict d, const int *restrict t, const int *restrict idx,
    int n)
{
	for (int i = 0; i < n; i++)
		d[i] = t[idx[i] & 63];
}
