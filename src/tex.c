/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Textures for animate exec's image instructions: formats, texel storage,
 * conversions, mipmaps and the sampler, following the Vulkan
 * specification's "Image Operations" chapter.
 *
 * Storage: every texel is 4 32-bit words, one per channel (the format's
 * missing channels unused): the float bits of the stored value for the
 * normalized and float formats (the value already quantized to the
 * format, so a texel holds exactly what the GPU's memory would), the
 * integer for the integer formats.  An sRGB texel keeps its encoded
 * value and is decoded on every read, before filtering, as Vulkan says.
 *
 * Conversions (Vulkan "Fixed-Point Data Conversions", "Floating Point"):
 *   unorm   c / (2^b - 1) on read; a write clamps to [0, 1] and rounds
 *           to nearest (ties away from zero: floor(v * max + 0.5))
 *   snorm   max(c / (2^(b-1) - 1), -1); a write clamps to [-1, 1]
 *   sRGB    c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4 on the
 *           colour channels, alpha linear
 *   16-bit float: round to nearest even, overflow to infinity; 11- and
 *           10-bit floats (R11G11B10F): no sign, round to nearest even
 *   integers: a write keeps the low b bits (two's complement)
 *   missing channels read as 0, 0, 0 and 1
 *
 * Sampling: the level of detail lambda = log2(rho) + bias, rho the larger
 * of the lengths of the coordinate's x and y derivatives in texels
 * (Vulkan's "scale factor"), clamped to [minLod, maxLod]; lambda <= 0 is
 * magnification.  Mip selection: nearest level ceil(d + 0.5) - 1, or a
 * linear blend of floor(d) and floor(d) + 1.  Nearest filtering takes
 * floor(u); linear takes the 2 (4, 8) texels around u - 0.5 with weights
 * from its fraction, in double precision (GPUs use 8 bits of sub-texel
 * precision for the weights, so their results differ in the last bits
 * of an 8-bit value).  Addressing per axis: repeat, mirrored repeat,
 * clamp to edge, clamp to border, mirror clamp to edge.  A cube map
 * picks its face by the major axis (Vulkan's table) and clamps to the
 * face's edge (no seamless filtering across faces).  Mipmaps are made by
 * a 2x2 (2x2x2) box filter of the level above, in linear space for sRGB.
 */
#include "ucas.h"
#include "img.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define F(n, s, c, cl, b0, b1, b2, b3) {n, s, c, cl, {b0, b1, b2, b3}}
const struct texfmt tex_fmts[] = {
	F("RGBA8", "Rgba8", 4, TC_UNORM, 8, 8, 8, 8),
	F("RGB8", "", 3, TC_UNORM, 8, 8, 8, 0),
	F("RG8", "Rg8", 2, TC_UNORM, 8, 8, 0, 0),
	F("R8", "R8", 1, TC_UNORM, 8, 0, 0, 0),
	F("RGBA8_SNORM", "Rgba8Snorm", 4, TC_SNORM, 8, 8, 8, 8),
	F("RG8_SNORM", "Rg8Snorm", 2, TC_SNORM, 8, 8, 0, 0),
	F("R8_SNORM", "R8Snorm", 1, TC_SNORM, 8, 0, 0, 0),
	F("SRGB8_A8", "", 4, TC_SRGB, 8, 8, 8, 8),
	F("SRGB8", "", 3, TC_SRGB, 8, 8, 8, 0),
	F("RGBA16", "Rgba16", 4, TC_UNORM, 16, 16, 16, 16),
	F("RG16", "Rg16", 2, TC_UNORM, 16, 16, 0, 0),
	F("R16", "R16", 1, TC_UNORM, 16, 0, 0, 0),
	F("RGBA16_SNORM", "Rgba16Snorm", 4, TC_SNORM, 16, 16, 16, 16),
	F("RGB10A2", "Rgb10A2", 4, TC_UNORM, 10, 10, 10, 2),
	F("RGB565", "", 3, TC_UNORM, 5, 6, 5, 0),
	F("RGBA16F", "Rgba16f", 4, TC_FLOAT, 16, 16, 16, 16),
	F("RG16F", "Rg16f", 2, TC_FLOAT, 16, 16, 0, 0),
	F("R16F", "R16f", 1, TC_FLOAT, 16, 0, 0, 0),
	F("RGBA32F", "Rgba32f", 4, TC_FLOAT, 32, 32, 32, 32),
	F("RG32F", "Rg32f", 2, TC_FLOAT, 32, 32, 0, 0),
	F("R32F", "R32f", 1, TC_FLOAT, 32, 0, 0, 0),
	F("R11G11B10F", "R11fG11fB10f", 3, TC_FLOAT, 11, 11, 10, 0),
	F("RGBA8UI", "Rgba8ui", 4, TC_UINT, 8, 8, 8, 8),
	F("RG8UI", "Rg8ui", 2, TC_UINT, 8, 8, 0, 0),
	F("R8UI", "R8ui", 1, TC_UINT, 8, 0, 0, 0),
	F("RGBA16UI", "Rgba16ui", 4, TC_UINT, 16, 16, 16, 16),
	F("RG16UI", "Rg16ui", 2, TC_UINT, 16, 16, 0, 0),
	F("R16UI", "R16ui", 1, TC_UINT, 16, 0, 0, 0),
	F("RGBA32UI", "Rgba32ui", 4, TC_UINT, 32, 32, 32, 32),
	F("RG32UI", "Rg32ui", 2, TC_UINT, 32, 32, 0, 0),
	F("R32UI", "R32ui", 1, TC_UINT, 32, 0, 0, 0),
	F("RGB10A2UI", "Rgb10a2ui", 4, TC_UINT, 10, 10, 10, 2),
	F("RGBA8I", "Rgba8i", 4, TC_SINT, 8, 8, 8, 8),
	F("RG8I", "Rg8i", 2, TC_SINT, 8, 8, 0, 0),
	F("R8I", "R8i", 1, TC_SINT, 8, 0, 0, 0),
	F("RGBA16I", "Rgba16i", 4, TC_SINT, 16, 16, 16, 16),
	F("RG16I", "Rg16i", 2, TC_SINT, 16, 16, 0, 0),
	F("R16I", "R16i", 1, TC_SINT, 16, 0, 0, 0),
	F("RGBA32I", "Rgba32i", 4, TC_SINT, 32, 32, 32, 32),
	F("RG32I", "Rg32i", 2, TC_SINT, 32, 32, 0, 0),
	F("R32I", "R32i", 1, TC_SINT, 32, 0, 0, 0),
};
#undef F
const int tex_nfmts = sizeof tex_fmts / sizeof tex_fmts[0];

int
tex_fmt_find(const char *name)
{
	static const struct { const char *a, *n; } alias[] = {
		{"RGBA", "RGBA8"}, {"RGB", "RGB8"}, {"RG", "RG8"}, {"R", "R8"},
		{"SRGB", "SRGB8_A8"}, {"RGBA8_SRGB", "SRGB8_A8"},
		{"SRGBA8", "SRGB8_A8"}, {"SRGB8A8", "SRGB8_A8"},
		{"RGB8_SRGB", "SRGB8"}, {"RGBA8UNORM", "RGBA8"},
		{"RGBA16FLOAT", "RGBA16F"}, {"RGBA32FLOAT", "RGBA32F"},
		{"HALF", "RGBA16F"}, {"FLOAT", "RGBA32F"}, {"GRAY", "R8"},
		{"GREY", "R8"}, {"L8", "R8"}, {"RGBA8888", "RGBA8"},
		{"RGB888", "RGB8"}, {"R11F_G11F_B10F", "R11G11B10F"},
		{"RGB10_A2", "RGB10A2"}, {"RGBA8SNORM", "RGBA8_SNORM"},
	};
	char u[32];
	size_t i;
	int k;

	for (i = 0; name[i] && i + 1 < sizeof u; i++)
		u[i] = (char)toupper((unsigned char)name[i]);
	u[i] = 0;
	for (i = 0; i < sizeof alias / sizeof alias[0]; i++)
		if (!strcmp(u, alias[i].a)) {
			snprintf(u, sizeof u, "%s", alias[i].n);
			break;
		}
	for (k = 0; k < tex_nfmts; k++)
		if (!strcmp(u, tex_fmts[k].name) || (tex_fmts[k].spv[0] &&
		    !strcasecmp(name, tex_fmts[k].spv)))
			return k;
	return -1;
}

int
tex_fmt_rgba8(void)
{
	return 0;
}

/* ---- small floats ---- */

static uint32_t
f2u(float f)
{
	uint32_t u;

	memcpy(&u, &f, 4);
	return u;
}

static float
u2f(uint32_t u)
{
	float f;

	memcpy(&f, &u, 4);
	return f;
}

/*
 * v rounded to a float with e exponent bits (bias 2^(e-1) - 1) and m
 * mantissa bits, round to nearest even; sign unless nosign (then
 * negatives are 0, as R11G11B10F)
 */
static double
small_float(double v, int e, int m, int nosign)
{
	int bias = (1 << (e - 1)) - 1, ex;
	double a = fabs(v), maxv, q;

	if (isnan(v))
		return v;
	if (nosign && v < 0)
		return 0;
	maxv = ldexp(2 - ldexp(1, -m), bias);
	if (a > maxv) {
		/* above the largest finite value: ties to the largest even */
		if (a >= ldexp(2 - ldexp(1, -m - 1), bias))
			return v < 0 ? -INFINITY : INFINITY;
		return v < 0 ? -maxv : maxv;
	}
	if (a == 0)
		return v;
	frexp(a, &ex);		/* a = f * 2^ex, f in [0.5, 1) */
	ex -= 1;
	if (ex < 1 - bias)
		ex = 1 - bias;	/* subnormal: a fixed step */
	q = ldexp(nearbyint(ldexp(a, m - ex)), ex - m);
	return v < 0 ? -q : q;
}

static double
srgb_decode(double c)
{
	return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

static double
srgb_encode(double l)
{
	return l <= 0.0031308 ? l * 12.92 : 1.055 * pow(l, 1 / 2.4) - 0.055;
}

/* one channel's value to its stored word */
static uint32_t
quantize(const struct texfmt *f, int ch, double v)
{
	int b = f->b[ch];
	double mx;

	switch (f->cls) {
	case TC_SRGB:
		if (ch < 3)
			v = srgb_encode(v < 0 ? 0 : v > 1 ? 1 : v);
		/* FALLTHROUGH */
	case TC_UNORM:
		mx = ldexp(1, b) - 1;
		if (isnan(v))
			v = 0;
		v = v < 0 ? 0 : v > 1 ? 1 : v;
		return f2u((float)(floor(v * mx + 0.5) / mx));
	case TC_SNORM:
		mx = ldexp(1, b - 1) - 1;
		if (isnan(v))
			v = 0;
		v = v < -1 ? -1 : v > 1 ? 1 : v;
		v = (v < 0 ? -floor(-v * mx + 0.5) : floor(v * mx + 0.5)) / mx;
		return f2u((float)v);
	case TC_FLOAT:
		if (b == 32)
			return f2u((float)v);
		if (b == 16)
			return f2u((float)small_float(v, 5, 10, 0));
		return f2u((float)small_float(v, 5, b - 5, 1));
	case TC_UINT: case TC_SINT: {
		int64_t iv = isfinite(v) ? (int64_t)v : 0;
		uint32_t m = b >= 32 ? 0xffffffffu : (1u << b) - 1;
		return (uint32_t)iv & m;
	}
	}
	return 0;
}

static double
unquantize(const struct texfmt *f, int ch, uint32_t w)
{
	int b = f->b[ch];

	switch (f->cls) {
	case TC_SRGB:
		return ch < 3 ? srgb_decode(u2f(w)) : u2f(w);
	case TC_UNORM: case TC_SNORM: case TC_FLOAT:
		return u2f(w);
	case TC_UINT:
		return (double)w;
	case TC_SINT:
		if (b >= 32)
			return (double)(int32_t)w;
		return (w >> (b - 1) & 1) ? (double)w - ldexp(1, b) : (double)w;
	}
	return 0;
}

/* ---- textures ---- */

void
tex_alloc(struct tex *t, int dim, int w, int h, int d, int layers, int fmt,
    int mips)
{
	int k;
	size_t n = 0;

	memset(t, 0, sizeof *t);
	t->dim = dim;
	t->w = w < 1 ? 1 : w;
	t->h = h < 1 ? 1 : h;
	t->d = d < 1 ? 1 : d;
	t->layers = layers < 1 ? 1 : layers;
	t->fmt = fmt < 0 ? 0 : fmt;
	t->levels = 1;
	if (mips) {
		int m = t->w > t->h ? t->w : t->h;
		if (dim == TD_3D && t->d > m)
			m = t->d;
		while ((m >>= 1) > 0 && t->levels < TEX_MAXLV)
			t->levels++;
	}
	for (k = 0; k < t->levels; k++) {
		t->lw[k] = t->w >> k ? t->w >> k : 1;
		t->lh[k] = t->h >> k ? t->h >> k : 1;
		t->ld[k] = dim == TD_3D ? (t->d >> k ? t->d >> k : 1) : t->d;
		t->lo[k] = n;
		n += (size_t)t->lw[k] * t->lh[k] * t->ld[k] * t->layers;
	}
	t->ntexel = n;
	t->t = xcalloc(n * 4, sizeof *t->t);
}

void
tex_free(struct tex *t)
{
	free(t->t);
	t->t = NULL;
}

void
tex_copy(struct tex *dst, const struct tex *src)
{
	*dst = *src;
	dst->t = xmalloc(src->ntexel * 4 * sizeof *dst->t);
	memcpy(dst->t, src->t, src->ntexel * 4 * sizeof *dst->t);
}

uint32_t *
tex_word(struct tex *t, int lv, int x, int y, int z, int layer)
{
	return t->t + 4 * (t->lo[lv] + (((size_t)layer * t->ld[lv] + z) *
	    t->lh[lv] + y) * t->lw[lv] + x);
}

void
tex_load(const struct tex *t, int lv, int x, int y, int z, int layer,
    double v[4])
{
	const struct texfmt *f = &tex_fmts[t->fmt];
	const uint32_t *w = tex_word((struct tex *)t, lv, x, y, z, layer);
	int c;

	for (c = 0; c < 4; c++)
		v[c] = c < f->nc ? unquantize(f, c, w[c]) : c == 3 ? 1 : 0;
}

void
tex_store(struct tex *t, int lv, int x, int y, int z, int layer,
    const double v[4])
{
	const struct texfmt *f = &tex_fmts[t->fmt];
	uint32_t *w = tex_word(t, lv, x, y, z, layer);
	int c;

	for (c = 0; c < f->nc; c++)
		w[c] = quantize(f, c, v[c]);
}

void
tex_from_rgba8(struct tex *t, const unsigned char *rgba, int w, int h)
{
	const struct texfmt *f = &tex_fmts[t->fmt];
	int x, y, z, l, c;

	for (l = 0; l < t->layers; l++)
		for (z = 0; z < t->ld[0]; z++)
			for (y = 0; y < t->h; y++)
				for (x = 0; x < t->w; x++) {
					const unsigned char *p = rgba + 4 * ((size_t)(y % h) * w +
					    x % w);
					uint32_t *o = tex_word(t, 0, x, y, z, l);
					for (c = 0; c < f->nc; c++) {
						double v = p[c] / 255.0;
						if (f->cls == TC_SRGB)	/* the bytes are encoded */
							o[c] = f2u((float)v);
						else if (f->cls == TC_UINT || f->cls == TC_SINT)
							o[c] = quantize(f, c, p[c]);
						else if (f->cls == TC_SNORM)
							o[c] = quantize(f, c, p[c] / 127.5 - 1);
						else
							o[c] = quantize(f, c, v);
					}
				}
}

void
tex_mips(struct tex *t)
{
	int k, x, y, z, l, c, i;
	int int_fmt = tex_fmts[t->fmt].cls == TC_UINT ||
	    tex_fmts[t->fmt].cls == TC_SINT;

	for (k = 1; k < t->levels; k++)
		for (l = 0; l < t->layers; l++)
			for (z = 0; z < t->ld[k]; z++)
				for (y = 0; y < t->lh[k]; y++)
					for (x = 0; x < t->lw[k]; x++) {
						double s[4] = {0, 0, 0, 0}, v[4];
						int n = 0, dz = t->dim == TD_3D ? 2 : 1;
						for (i = 0; i < 4 * dz; i++) {
							int sx = 2 * x + (i & 1), sy = 2 * y + (i >> 1 & 1),
							    sz = t->dim == TD_3D ? 2 * z + (i >> 2) : z;
							if (sx >= t->lw[k - 1])
								sx = t->lw[k - 1] - 1;
							if (sy >= t->lh[k - 1])
								sy = t->lh[k - 1] - 1;
							if (sz >= t->ld[k - 1])
								sz = t->ld[k - 1] - 1;
							tex_load(t, k - 1, sx, sy, sz, l, v);
							for (c = 0; c < 4; c++)
								s[c] += v[c];
							n++;
						}
						for (c = 0; c < 4; c++)
							s[c] = int_fmt ? floor(s[c] / n + 0.5) : s[c] / n;
						tex_store(t, k, x, y, z, l, s);
					}
}

int
tex_level_size(const struct tex *t, int lv, int *w, int *h, int *d)
{
	if (lv < 0 || lv >= t->levels)
		return 0;
	*w = t->lw[lv];
	*h = t->lh[lv];
	*d = t->ld[lv];
	return 1;
}

void
tex_to_rgba8(const struct tex *t, unsigned char *out)
{
	const struct texfmt *f = &tex_fmts[t->fmt];
	int x, y, c;

	for (y = 0; y < t->h; y++)
		for (x = 0; x < t->w; x++) {
			const uint32_t *w = tex_word((struct tex *)t, 0, x, y, 0, 0);
			unsigned char *o = out + 4 * ((size_t)y * t->w + x);
			for (c = 0; c < 4; c++) {
				double v;
				if (c >= f->nc) {
					o[c] = c == 3 ? 255 : 0;
					continue;
				}
				if (f->cls == TC_UINT || f->cls == TC_SINT) {
					double mx = ldexp(1, f->b[c]) - 1;
					v = unquantize(f, c, w[c]);
					v = f->cls == TC_SINT ? (v + mx / 2) / mx : v / mx;
				} else
					v = u2f(w[c]);	/* sRGB: shown as encoded */
				if (f->cls == TC_SNORM)
					v = (v + 1) / 2;
				v = isnan(v) ? 0 : v < 0 ? 0 : v > 1 ? 1 : v;
				o[c] = (unsigned char)floor(v * 255 + 0.5);
			}
		}
}

/* ---- sampling ---- */

void
samp_default(struct samp *s)
{
	memset(s, 0, sizeof *s);
	s->mag = s->min = s->mip = SF_LINEAR;
	s->addr[0] = s->addr[1] = s->addr[2] = SA_REPEAT;
	s->border = SB_TRANSPARENT;
	s->cmpop = 3;		/* less or equal */
	s->minlod = 0;
	s->maxlod = 1000;
}

double
tex_lambda(const struct tex *t, int nc, const double *dx, const double *dy)
{
	double size[3] = {t->w, t->h, t->d}, rx = 0, ry = 0, rho;
	int i;

	if (t->dim == TD_CUBE)
		size[2] = size[0];
	for (i = 0; i < nc && i < 3; i++) {
		rx += dx[i] * size[i] * dx[i] * size[i];
		ry += dy[i] * size[i] * dy[i] * size[i];
	}
	rho = sqrt(rx > ry ? rx : ry);
	return rho > 0 ? log2(rho) : -1000;
}

static int
mirror(int n)
{
	return n >= 0 ? n : -(1 + n);
}

/* an integer texel coordinate wrapped; -1 for the border */
static int
wrap(int i, int size, int mode)
{
	switch (mode) {
	case SA_REPEAT:
		i %= size;
		return i < 0 ? i + size : i;
	case SA_MIRROR: {
		int m = i % (2 * size);
		if (m < 0)
			m += 2 * size;
		return (size - 1) - mirror(m - size);
	}
	case SA_CLAMP:
		return i < 0 ? 0 : i >= size ? size - 1 : i;
	case SA_BORDER:
		return i < 0 || i >= size ? -1 : i;
	case SA_MIRROR_CLAMP:
		i = mirror(i);
		return i >= size ? size - 1 : i;
	}
	return 0;
}

static int
compare(int op, double ref, double v)
{
	switch (op) {
	case 0: return 0;
	case 1: return ref < v;
	case 2: return ref == v;
	case 3: return ref <= v;
	case 4: return ref > v;
	case 5: return ref != v;
	case 6: return ref >= v;
	}
	return 1;
}

/* one texel, wrapped, the border or compared */
static void
texel(const struct tex *t, const struct samp *s, const struct texreq *r,
    int lv, const int *ic, int axes, int layer, int cube, double v[4])
{
	int xyz[3] = {0, 0, 0}, size[3], a, c;

	size[0] = t->lw[lv];
	size[1] = t->lh[lv];
	size[2] = t->ld[lv];
	for (a = 0; a < axes; a++) {
		xyz[a] = wrap(ic[a], size[a], cube ? SA_CLAMP : s->addr[a]);
		if (xyz[a] < 0) {
			for (c = 0; c < 4; c++)
				v[c] = s->border == SB_WHITE ? 1 : c == 3 &&
				    s->border == SB_BLACK ? 1 : 0;
			goto cmp;
		}
	}
	tex_load(t, lv, xyz[0], xyz[1], xyz[2], layer, v);
cmp:
	if (r->dref) {
		double d = compare(s->cmpop, r->ref, v[0]);
		v[0] = v[1] = v[2] = d;
		v[3] = 1;
	}
}

static void
sample_level(const struct tex *t, const struct samp *s, const struct texreq *r,
    int lv, int filt, const double *u, int axes, int layer, int cube,
    double out[4])
{
	int i0[3], k, a, c, n;
	double fr[3];

	for (a = 0; a < axes; a++) {
		double sz = a == 0 ? t->lw[lv] : a == 1 ? t->lh[lv] : t->ld[lv];
		double x = s->unnorm ? u[a] : u[a] * sz;
		int off = cube ? 0 : r->off[a];
		if (filt == SF_NEAREST && r->gather < 0) {
			i0[a] = (int)floor(x) + off;
			fr[a] = 0;
		} else {
			double xx = x - 0.5;
			i0[a] = (int)floor(xx) + off;
			fr[a] = xx - floor(xx);
		}
		/* far outside: keep the integers sane */
		if (i0[a] > 1 << 28)
			i0[a] = 1 << 28;
		if (i0[a] < -(1 << 28))
			i0[a] = -(1 << 28);
	}
	if (r->gather >= 0) {
		/* the 2x2 footprint, (i0,j1) (i1,j1) (i1,j0) (i0,j0) */
		static const int ox[4] = {0, 1, 1, 0}, oy[4] = {1, 1, 0, 0};
		for (k = 0; k < 4; k++) {
			int ic[3] = {i0[0] + ox[k], i0[1] + oy[k], axes > 2 ? i0[2] : 0};
			double v[4];
			texel(t, s, r, lv, ic, axes, layer, cube, v);
			out[k] = r->dref ? v[0] : v[r->gather & 3];
		}
		return;
	}
	for (c = 0; c < 4; c++)
		out[c] = 0;
	n = filt == SF_NEAREST ? 1 : 1 << axes;
	for (k = 0; k < n; k++) {
		int ic[3];
		double wgt = 1, v[4];
		for (a = 0; a < axes; a++) {
			int hi = k >> a & 1;
			ic[a] = i0[a] + hi;
			if (filt != SF_NEAREST)
				wgt *= hi ? fr[a] : 1 - fr[a];
		}
		if (wgt == 0)
			continue;
		texel(t, s, r, lv, ic, axes, layer, cube, v);
		for (c = 0; c < 4; c++)
			out[c] += wgt * v[c];
	}
}

/* cube: the face and its (s, t) from a direction (Vulkan's table) */
static int
cube_face(const double *d, double *s, double *t)
{
	double ax = fabs(d[0]), ay = fabs(d[1]), az = fabs(d[2]), sc, tc, ma;
	int f;

	if (ax >= ay && ax >= az) {
		f = d[0] >= 0 ? 0 : 1;
		sc = d[0] >= 0 ? -d[2] : d[2];
		tc = -d[1];
		ma = ax;
	} else if (ay >= az) {
		f = d[1] >= 0 ? 2 : 3;
		sc = d[0];
		tc = d[1] >= 0 ? d[2] : -d[2];
		ma = ay;
	} else {
		f = d[2] >= 0 ? 4 : 5;
		sc = d[2] >= 0 ? d[0] : -d[0];
		tc = -d[1];
		ma = az;
	}
	if (ma == 0)
		ma = 1;
	*s = 0.5 * (sc / ma + 1);
	*t = 0.5 * (tc / ma + 1);
	return f;
}

void
tex_sample(const struct tex *t, const struct samp *s, const struct texreq *r,
    double out[4])
{
	double u[3] = {0, 0, 0}, lambda = r->lod, dl, a[4], b[4];
	int axes, layer = 0, cube = t->dim == TD_CUBE, filt, lv0, lv1, c;
	int q = t->levels - 1;

	axes = t->dim == TD_1D || t->dim == TD_BUFFER ? 1 : t->dim == TD_3D ?
	    3 : 2;
	if (cube) {
		int f = cube_face(r->c, &u[0], &u[1]);
		int arr = r->arrayed ? (int)floor(r->c[3] + 0.5) : 0;
		int n = t->layers / 6;
		arr = arr < 0 ? 0 : arr >= n ? n - 1 : arr;
		layer = arr * 6 + f;
	} else {
		for (c = 0; c < axes; c++)
			u[c] = r->c[c];
		if (r->arrayed) {
			layer = (int)floor(r->c[r->nc] + 0.5);
			layer = layer < 0 ? 0 : layer >= t->layers ? t->layers - 1 :
			    layer;
		}
	}
	if (r->gather >= 0) {
		sample_level(t, s, r, 0, SF_LINEAR, u, axes, layer, cube, out);
		return;
	}
	if (lambda > s->maxlod)
		lambda = s->maxlod;
	if (lambda < s->minlod)
		lambda = s->minlod;
	filt = lambda <= 0 ? s->mag : s->min;
	if (s->unnorm || lambda <= 0 || q == 0) {
		sample_level(t, s, r, 0, filt, u, axes, layer, cube, out);
		return;
	}
	dl = lambda > q ? q : lambda;
	if (s->mip == SF_NEAREST) {
		lv0 = dl <= 0.5 ? 0 : (int)ceil(dl + 0.5) - 1;
		if (lv0 > q)
			lv0 = q;
		sample_level(t, s, r, lv0, filt, u, axes, layer, cube, out);
		return;
	}
	lv0 = (int)floor(dl);
	lv1 = lv0 + 1 > q ? q : lv0 + 1;
	dl -= lv0;
	sample_level(t, s, r, lv0, filt, u, axes, layer, cube, a);
	if (dl == 0 || lv1 == lv0) {
		memcpy(out, a, sizeof a);
		return;
	}
	sample_level(t, s, r, lv1, filt, u, axes, layer, cube, b);
	for (c = 0; c < 4; c++)
		out[c] = a[c] * (1 - dl) + b[c] * dl;
}

/* ---- inputs: patterns and colours ---- */

static const struct { const char *n; unsigned char c[4]; } colors[] = {
	{"black", {0, 0, 0, 255}}, {"white", {255, 255, 255, 255}},
	{"red", {255, 0, 0, 255}}, {"green", {0, 255, 0, 255}},
	{"blue", {0, 0, 255, 255}}, {"yellow", {255, 255, 0, 255}},
	{"cyan", {0, 255, 255, 255}}, {"magenta", {255, 0, 255, 255}},
	{"purple", {128, 0, 128, 255}}, {"orange", {255, 165, 0, 255}},
	{"gray", {128, 128, 128, 255}}, {"grey", {128, 128, 128, 255}},
	{"lime", {0, 255, 0, 255}}, {"aqua", {0, 255, 255, 255}},
	{"fuchsia", {255, 0, 255, 255}}, {"violet", {238, 130, 238, 255}},
	{"pink", {255, 192, 203, 255}}, {"hotpink", {255, 105, 180, 255}},
	{"brown", {165, 42, 42, 255}}, {"maroon", {128, 0, 0, 255}},
	{"navy", {0, 0, 128, 255}}, {"teal", {0, 128, 128, 255}},
	{"olive", {128, 128, 0, 255}}, {"silver", {192, 192, 192, 255}},
	{"lightgray", {211, 211, 211, 255}}, {"lightgrey", {211, 211, 211, 255}},
	{"darkgray", {169, 169, 169, 255}}, {"darkgrey", {169, 169, 169, 255}},
	{"darkgreen", {0, 100, 0, 255}}, {"darkred", {139, 0, 0, 255}},
	{"darkblue", {0, 0, 139, 255}}, {"gold", {255, 215, 0, 255}},
	{"indigo", {75, 0, 130, 255}}, {"crimson", {220, 20, 60, 255}},
	{"coral", {255, 127, 80, 255}}, {"salmon", {250, 128, 114, 255}},
	{"turquoise", {64, 224, 208, 255}}, {"beige", {245, 245, 220, 255}},
	{"khaki", {240, 230, 140, 255}}, {"lavender", {230, 230, 250, 255}},
	{"chocolate", {210, 105, 30, 255}}, {"tan", {210, 180, 140, 255}},
	{"skyblue", {135, 206, 235, 255}}, {"lightblue", {173, 216, 230, 255}},
	{"lightgreen", {144, 238, 144, 255}}, {"tomato", {255, 99, 71, 255}},
	{"plum", {221, 160, 221, 255}}, {"orchid", {218, 112, 214, 255}},
	{"amber", {255, 191, 0, 255}}, {"mint", {152, 255, 152, 255}},
	{"transparent", {0, 0, 0, 0}},
};

static int
hexv(int c)
{
	return isdigit(c) ? c - '0' : isxdigit(c) ? tolower(c) - 'a' + 10 : -1;
}

int
img_color(const char *name, unsigned char rgba[4])
{
	size_t i, n = strlen(name);

	if (name[0] == '#' || (n == 6 && strspn(name, "0123456789abcdefABCDEF")
	    == 6 && strpbrk(name, "0123456789") != NULL)) {
		const char *h = name + (name[0] == '#');
		size_t hn = strlen(h);
		int v[8], k;
		if (hn != 3 && hn != 4 && hn != 6 && hn != 8)
			return 0;
		for (k = 0; k < (int)hn; k++)
			if ((v[k] = hexv((unsigned char)h[k])) < 0)
				return 0;
		rgba[3] = 255;
		if (hn <= 4)
			for (k = 0; k < (int)hn; k++)
				rgba[k] = (unsigned char)(v[k] * 17);
		else
			for (k = 0; k < (int)hn / 2; k++)
				rgba[k] = (unsigned char)(v[2 * k] * 16 + v[2 * k + 1]);
		return 1;
	}
	if (!strncasecmp(name, "rgb", 3)) {
		const char *p = strchr(name, '(');
		double c[4] = {0, 0, 0, -1};
		int k = 0, fl = 0, alpha = name[3] == 'a' || name[3] == 'A';
		char *e;
		if (p == NULL)
			return 0;
		for (p++; k < 4; k++) {
			while (*p == ' ' || *p == ',')
				p++;
			if (*p == ')' || *p == 0)
				break;
			c[k] = strtod(p, &e);
			if (e == p)
				return 0;
			if (memchr(p, '.', (size_t)(e - p)) && k < 3)
				fl = 1;
			p = e;
		}
		if (k < 3)
			return 0;
		if (!fl && c[0] <= 1 && c[1] <= 1 && c[2] <= 1)
			fl = 1;	/* rgba(1, 0, 0, 1): fractions */
		for (k = 0; k < 3; k++) {
			double v = fl ? c[k] * 255 : c[k];
			rgba[k] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5);
		}
		if (c[3] < 0 || !alpha)
			rgba[3] = 255;
		else {
			double a = c[3] <= 1 ? c[3] * 255 : c[3];
			rgba[3] = (unsigned char)(a < 0 ? 0 : a > 255 ? 255 : a + 0.5);
		}
		return 1;
	}
	for (i = 0; i < sizeof colors / sizeof colors[0]; i++)
		if (!strcasecmp(name, colors[i].n)) {
			memcpy(rgba, colors[i].c, 4);
			return 1;
		}
	/* "light gray" spelt as one word or with a dash */
	if (n < 32 && strchr(name, '-')) {
		char b[32];
		size_t j = 0;
		for (i = 0; i < n; i++)
			if (name[i] != '-')
				b[j++] = name[i];
		b[j] = 0;
		return img_color(b, rgba);
	}
	return 0;
}

const char *
img_color_name(const unsigned char rgba[4])
{
	size_t i;

	for (i = 0; i < sizeof colors / sizeof colors[0]; i++)
		if (!memcmp(rgba, colors[i].c, 4))
			return colors[i].n;
	return NULL;
}

static uint64_t
mix64(uint64_t z)
{
	z += 0x9e3779b97f4a7c15ULL;
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

void
img_pattern(unsigned char *rgba, int w, int h, int kind,
    const unsigned char col[4], uint64_t seed)
{
	int x, y, cell = w / 8 > 0 ? w / 8 : 1;

	for (y = 0; y < h; y++)
		for (x = 0; x < w; x++) {
			unsigned char *p = rgba + 4 * ((size_t)y * w + x);
			switch (kind) {
			case IK_COLOR:
				memcpy(p, col, 4);
				break;
			case IK_GRADIENT: {
				int r = w > 1 ? 255 * x / (w - 1) : 0;
				int g = h > 1 ? 255 * y / (h - 1) : 0;
				p[0] = (unsigned char)r;
				p[1] = (unsigned char)g;
				p[2] = (unsigned char)(255 - (r + g) / 2);
				p[3] = 255;
				break;
			}
			case IK_CHECKER: {
				unsigned char v = ((x / cell) + (y / cell)) & 1 ? 255 : 0;
				p[0] = p[1] = p[2] = v;
				p[3] = 255;
				break;
			}
			default: {	/* noise: every channel, alpha too */
				uint64_t r = mix64(seed * 0x100000001b3ULL ^
				    ((uint64_t)y << 32 | (uint32_t)x));
				p[0] = (unsigned char)r;
				p[1] = (unsigned char)(r >> 8);
				p[2] = (unsigned char)(r >> 16);
				p[3] = (unsigned char)(r >> 24);
			}
			}
		}
}

void
img_downscale(const unsigned char *src, int w, int h, unsigned char *dst,
    int nw, int nh)
{
	int x, y, c;

	for (y = 0; y < nh; y++) {
		int y0 = (int)((int64_t)y * h / nh), y1 = (int)((int64_t)(y + 1) *
		    h / nh);
		if (y1 <= y0)
			y1 = y0 + 1;
		for (x = 0; x < nw; x++) {
			int x0 = (int)((int64_t)x * w / nw), x1 = (int)((int64_t)(x + 1) *
			    w / nw), i, j;
			double s[4] = {0, 0, 0, 0}, n = 0;
			if (x1 <= x0)
				x1 = x0 + 1;
			for (j = y0; j < y1; j++)
				for (i = x0; i < x1; i++) {
					const unsigned char *p = src + 4 * ((size_t)j * w + i);
					/* premultiplied, so clear pixels add no colour */
					double a = p[3] / 255.0;
					for (c = 0; c < 3; c++)
						s[c] += p[c] * a;
					s[3] += p[3];
					n++;
				}
			for (c = 0; c < 4; c++) {
				double v = c < 3 ? (s[3] > 0 ? s[c] * 255 / s[3] : 0) :
				    s[3] / n;
				dst[4 * ((size_t)y * nw + x) + c] = (unsigned char)(v + 0.5);
			}
		}
	}
}

/* ---- the request's choices ---- */

struct imgopts img_opt;

void
img_opt_reset(void)
{
	struct imggal gal[IMG_MAXGAL];
	int k, ng = img_opt.ngal;

	memcpy(gal, img_opt.gal, sizeof gal);
	memset(&img_opt, 0, sizeof img_opt);
	memcpy(img_opt.gal, gal, sizeof gal);
	img_opt.ngal = ng;
	img_opt.outfmt = -1;
	img_opt.filt = img_opt.mipf = img_opt.addr = img_opt.border = -1;
	for (k = 0; k < IMG_MAXIN; k++) {
		img_opt.in[k].fmt = -1;
		img_opt.in[k].filt = img_opt.in[k].addr = -1;
	}
}

void
img_gallery_clear(void)
{
	int k;

	for (k = 0; k < img_opt.ngal; k++)
		free(img_opt.gal[k].data);
	img_opt.ngal = 0;
}

int
img_gallery_add(const char *name, const unsigned char *data, size_t n)
{
	struct imggal *g;

	if (img_opt.ngal == IMG_MAXGAL || n > IMG_MAXFILE)
		return 0;
	g = &img_opt.gal[img_opt.ngal++];
	snprintf(g->name, sizeof g->name, "%s", name);
	g->data = xmalloc(n ? n : 1);
	memcpy(g->data, data, n);
	g->n = n;
	return 1;
}

int
img_in_word(struct imgin *in, const char *w)
{
	int f;

	if (!strcasecmp(w, "noise") || !strcasecmp(w, "random") ||
	    !strcasecmp(w, "static")) {
		in->kind = IK_NOISE;
		return 1;
	}
	if (!strcasecmp(w, "gradient") || !strcasecmp(w, "ramp")) {
		in->kind = IK_GRADIENT;
		return 1;
	}
	if (!strncasecmp(w, "checker", 7) || !strcasecmp(w, "checkered")) {
		in->kind = IK_CHECKER;
		return 1;
	}
	if (!strcasecmp(w, "nearest") || !strcasecmp(w, "point")) {
		in->filt = SF_NEAREST;
		return 1;
	}
	if (!strcasecmp(w, "linear") || !strcasecmp(w, "bilinear")) {
		in->filt = SF_LINEAR;
		return 1;
	}
	if (!strcasecmp(w, "repeat") || !strcasecmp(w, "wrap")) {
		in->addr = SA_REPEAT;
		return 1;
	}
	if (!strcasecmp(w, "clamp")) {
		in->addr = SA_CLAMP;
		return 1;
	}
	if (!strcasecmp(w, "mirror") || !strcasecmp(w, "mirrored")) {
		in->addr = SA_MIRROR;
		return 1;
	}
	if (!strcasecmp(w, "border")) {
		in->addr = SA_BORDER;
		return 1;
	}
	if ((f = tex_fmt_find(w)) >= 0 && strcasecmp(w, "float") &&
	    strcasecmp(w, "half") && strcasecmp(w, "r") && strcasecmp(w, "gray")
	    && strcasecmp(w, "grey")) {
		in->fmt = f;
		return 1;
	}
	if (img_color(w, in->rgba)) {
		in->kind = IK_COLOR;
		return 1;
	}
	return 0;
}

size_t
img_b64(const char *s, size_t n, unsigned char *out)
{
	uint32_t acc = 0;
	int bits = 0;
	size_t i, o = 0;

	for (i = 0; i < n; i++) {
		int c = (unsigned char)s[i], v;
		if (c >= 'A' && c <= 'Z')
			v = c - 'A';
		else if (c >= 'a' && c <= 'z')
			v = c - 'a' + 26;
		else if (c >= '0' && c <= '9')
			v = c - '0' + 52;
		else if (c == '+' || c == '-')
			v = 62;
		else if (c == '/' || c == '_')
			v = 63;
		else if (c == '=')
			break;
		else
			continue;
		acc = acc << 6 | (uint32_t)v;
		bits += 6;
		if (bits >= 8) {
			bits -= 8;
			out[o++] = (unsigned char)(acc >> bits);
		}
	}
	return o;
}
