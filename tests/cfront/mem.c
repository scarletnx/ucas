/* cfront test: functions with memory: element-wise loop bodies, strlen
 * scans, the loop IR */
#include <stdint.h>
#include <stddef.h>

void brighten(uint8_t *dst, const uint8_t *src, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		unsigned v = src[i] * 3u / 2u + 10u;
		dst[i] = v > 255 ? 255 : (uint8_t)v;
	}
}

/* in place: the load and the store are the same element */
void invert_low(uint8_t *p, size_t n)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (uint8_t)(p[i] < 16 ? 15 - p[i] : p[i]) & 0xff;
}

/* two stores; the second reads a load from before the first */
void split(uint8_t *a, uint8_t *b, const uint8_t *s, int n)
{
	for (int i = 0; i < n; i++) {
		uint8_t x = s[i];
		a[i] = (uint8_t)(x / 16);
		b[i] = (uint8_t)(x % 16 + x / 16 * 0);
	}
}

size_t my_strlen(const char *s)
{
	size_t n = 0;
	while (s[n])
		n++;
	return n;
}

const char *skip(const char *p)
{
	while (*p != '\0')
		++p;
	return p;
}

/* an unsigned 32-bit counter could wrap: not a strlen */
unsigned len32(const char *s)
{
	unsigned n = 0;
	while (s[n])
		n++;
	return n;
}

void rgb2gray(uint8_t *restrict g, const uint8_t *restrict rgb, int n)
{
	for (int i = 0; i < n / 3; i++) {
		unsigned r = rgb[3 * i], gg = rgb[3 * i + 1], b = rgb[3 * i + 2];
		g[i] = (uint8_t)((r * 77 + gg * 150 + b * 29) >> 8);
	}
}

uint32_t sum8(const uint8_t *a, int n)
{
	uint32_t s = 0;
	for (int i = 0; i < n; i++)
		s += a[i] & 0xff;
	return s;
}

void copy_odd(int *d, const int *s, int n)
{
	for (int i = 0; i < n; i++) {
		if (s[i] & 1)
			d[i] = s[i];
	}
}
