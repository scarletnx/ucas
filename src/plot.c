/*
 * plot: functions as GIF images (a still plot is a one-frame GIF).
 *
 *   plot F, G, ... [x in a..b] [y in c..d] [options] [to NAME]
 *   plot3d F ...              a surface z = f(x, y)
 *   neighbourhood F [at a]    plot with bounds chosen from F's features
 *
 * The function list is read as text: pieces separated by commas at the
 * top level, each one a function argument as other commands take it
 * (a name, or an expression whose free variables become parameters).
 * A piece with '=' is an equation: y = ..., r = ... (polar), z = ...
 * (a surface), y' = ... (a slope field), else lhs - rhs = 0 is an
 * implicit curve.  A parenthesised tuple (x(t), y(t)[, z(t)]) is a
 * parametric curve, or (P, Q) of two variables a vector field.
 *
 * What is drawn follows the number of free variables: one, a curve;
 * two, a surface (or a contour/heatmap view); three or more, frames
 * over one variable and small multiples over another.  Bounds not given
 * come from the parameters' declared ranges, else from the function
 * itself (its zeros, extrema, inflections, poles, domain edges and
 * period), the same search `neighbourhood` reports in full.
 *
 * Rendering is in software into an indexed canvas: anti-aliased curves
 * (a coverage buffer composited into four tints per series), a z-buffer
 * for surfaces, marching squares for implicit curves and contours, and
 * the 6x10 font for labels.  The image goes out as animate's GIFs do:
 * NAME.gif on the command line, an attachment in the bot.
 */
#include "ucas.h"
#include "calc.h"
#include "gif.h"
#include "font6x10.h"
#include "props.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- palette ---- */

#define NSER 8		/* series colours, four tints each */
#define NCM 32		/* colour map levels */
#define NSH 6		/* shades per level: 5 dark to full, 1 light */
enum {
	P_BG, P_TEXT, P_GRID, P_AXIS, P_FRAME, P_DIM, P_ZERO, P_MARK,
	P_PANE, P_PANEGRID, P_LEGBG, P_LEGBORDER, P_UNDEF, P_ASYM, P_WIRE,
	P_SPARE,
	P_SER = 16, P_CM = P_SER + NSER * 4, P_TINT = P_CM + NCM * NSH,
	NPAL = P_TINT + 9	/* three tints of P_AXIS, P_WIRE, P_MARK */
};
static unsigned char pal[256][3];

static const unsigned char sercol[NSER][3] = {
	{31, 119, 180}, {255, 127, 14}, {44, 160, 44}, {214, 39, 40},
	{148, 103, 189}, {140, 86, 75}, {227, 119, 194}, {23, 190, 207},
};
/* viridis, at ten points */
static const unsigned char vir[10][3] = {
	{68, 1, 84}, {72, 40, 120}, {62, 74, 137}, {49, 104, 142},
	{38, 130, 142}, {31, 158, 137}, {53, 183, 121}, {109, 205, 89},
	{180, 222, 44}, {253, 231, 37},
};

static void
pal_init(void)
{
	static const unsigned char fixed[16][3] = {
		{255, 255, 255}, {28, 28, 34}, {234, 235, 240}, {128, 130, 140},
		{88, 90, 100}, {96, 98, 108}, {150, 152, 162}, {205, 35, 35},
		{243, 244, 247}, {222, 224, 230}, {255, 255, 255}, {180, 182, 190},
		{218, 218, 222}, {160, 162, 172}, {50, 50, 58}, {0, 0, 0},
	};
	static const double tint[4] = {1, 0.7, 0.44, 0.2};
	static const double shade[5] = {0.42, 0.57, 0.71, 0.86, 1.0};
	int i, j, k;

	memset(pal, 0, sizeof pal);
	memcpy(pal, fixed, sizeof fixed);
	for (i = 0; i < NSER; i++)
		for (j = 0; j < 4; j++)
			for (k = 0; k < 3; k++)
				pal[P_SER + i * 4 + j][k] = (unsigned char)(255 +
				    (sercol[i][k] - 255) * tint[j] + 0.5);
	{
		static const int base[3] = {P_AXIS, P_WIRE, P_MARK};
		for (i = 0; i < 3; i++)
			for (j = 0; j < 3; j++)
				for (k = 0; k < 3; k++)
					pal[P_TINT + i * 3 + j][k] = (unsigned char)(255 +
					    (pal[base[i]][k] - 255) * tint[j + 1] + 0.5);
	}
	for (i = 0; i < NCM; i++) {
		double t = (i + 0.5) / NCM * 9, c[3];
		int a = (int)t;
		if (a > 8)
			a = 8;
		t -= a;
		for (k = 0; k < 3; k++)
			c[k] = vir[a][k] + (vir[a + 1][k] - vir[a][k]) * t;
		for (j = 0; j < NSH; j++)
			for (k = 0; k < 3; k++)
				pal[P_CM + i * NSH + j][k] = (unsigned char)(j < 5 ?
				    c[k] * shade[j] + 0.5 : c[k] + (255 - c[k]) * 0.62 + 0.5);
	}
}

/* colour map entry for v in [0, 1], shade 0..5 */
static int
cmap(double v, int sh)
{
	int l = (int)floor(v * NCM);

	if (!(v == v))
		l = 0;
	if (l < 0)
		l = 0;
	if (l >= NCM)
		l = NCM - 1;
	return P_CM + l * NSH + sh;
}

/* ---- canvas ---- */

static struct {
	int W, H;
	unsigned char *px;
	float *cov, *zb;
	int cx0, cy0, cx1, cy1;	/* clip rectangle for curves */
} C;

static void
cv_open(int w, int h)
{
	C.W = w;
	C.H = h;
	C.px = xcalloc((size_t)w * h, 1);
	C.cov = xcalloc((size_t)w * h, sizeof *C.cov);
	C.zb = xcalloc((size_t)w * h, sizeof *C.zb);
	C.cx0 = C.cy0 = 0;
	C.cx1 = w - 1;
	C.cy1 = h - 1;
}

static void
cv_close(void)
{
	free(C.px);
	free(C.cov);
	free(C.zb);
	memset(&C, 0, sizeof C);
}

static void
cv_clip(int x0, int y0, int x1, int y1)
{
	C.cx0 = x0 < 0 ? 0 : x0;
	C.cy0 = y0 < 0 ? 0 : y0;
	C.cx1 = x1 >= C.W ? C.W - 1 : x1;
	C.cy1 = y1 >= C.H ? C.H - 1 : y1;
}

static void
put(int x, int y, int c)
{
	if (x >= 0 && y >= 0 && x < C.W && y < C.H)
		C.px[(size_t)y * C.W + x] = (unsigned char)c;
}

static int
get(int x, int y)
{
	if (x >= 0 && y >= 0 && x < C.W && y < C.H)
		return C.px[(size_t)y * C.W + x];
	return P_BG;
}

static void
fill(int x, int y, int w, int h, int c)
{
	int i, j;

	for (j = y; j < y + h; j++)
		for (i = x; i < x + w; i++)
			put(i, j, c);
}

static void
rect(int x, int y, int w, int h, int c)
{
	fill(x, y, w, 1, c);
	fill(x, y + h - 1, w, 1, c);
	fill(x, y, 1, h, c);
	fill(x + w - 1, y, 1, h, c);
}

/* a crisp one-pixel line (axes, ticks, arrows) */
static void
line1(int x0, int y0, int x1, int y1, int c)
{
	int dx = abs(x1 - x0), dy = -abs(y1 - y0);
	int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, e = dx + dy, e2;

	for (;;) {
		put(x0, y0, c);
		if (x0 == x1 && y0 == y1)
			break;
		e2 = 2 * e;
		if (e2 >= dy) {
			e += dy;
			x0 += sx;
		}
		if (e2 <= dx) {
			e += dx;
			y0 += sy;
		}
	}
}

/* text in the 6x10 font; returns the x after it */
static int
text(int x, int y, const char *s, int c)
{
	int r, b;

	for (; *s; s++, x += FONT_6X10_W) {
		unsigned ch = (unsigned char)*s;
		if (ch < 32 || ch > 126)
			ch = '?';
		for (r = 0; r < FONT_6X10_H; r++) {
			unsigned bits = font_6x10[ch - 32][r];
			for (b = 0; b < FONT_6X10_W; b++)
				if (bits & (0x80 >> b))
					put(x + b, y + r, c);
		}
	}
	return x;
}

static int
textw(const char *s)
{
	return (int)strlen(s) * FONT_6X10_W;
}

/* text on a background box, so it stays readable over a plot */
static void
text_bg(int x, int y, const char *s, int c, int bg)
{
	fill(x - 2, y - 1, textw(s) + 4, FONT_6X10_H + 1, bg);
	text(x, y, s, c);
}

/* the expression as printable ASCII: pi for π, ^2 for ², ... */
static void
ascii_of(char *dst, size_t n, const char *s)
{
	static const struct {
		const char *u, *a;
	} map[] = {
		{"\xcf\x80", "pi"}, {"\xce\xb8", "theta"}, {"\xc2\xb2", "^2"},
		{"\xc2\xb3", "^3"}, {"\xe2\x88\x9a", "sqrt"}, {"\xc2\xb7", "*"},
		{"\xe2\x89\x88", "~"}, {"\xe2\x8a\x95", " xor "},
		{"\xe2\x89\xa4", "<="}, {"\xe2\x89\xa5", ">="},
		{"\xe2\x89\xa0", "!="}, {"\xce\xb1", "alpha"},
		{"\xce\xb2", "beta"}, {"\xcf\x86", "phi"}, {"\xcf\x89", "omega"},
		{"\xce\xbb", "lambda"}, {"\xce\xbc", "mu"}, {"\xcf\x84", "tau"},
	};
	size_t k = 0, m, j;

	while (*s && k + 1 < n) {
		if ((unsigned char)*s < 0x80) {
			dst[k++] = *s++;
			continue;
		}
		for (j = 0; j < sizeof map / sizeof map[0]; j++) {
			m = strlen(map[j].u);
			if (!strncmp(s, map[j].u, m))
				break;
		}
		if (j < sizeof map / sizeof map[0]) {
			const char *a = map[j].a;
			while (*a && k + 1 < n)
				dst[k++] = *a++;
			s += strlen(map[j].u);
		} else {
			dst[k++] = '?';
			s++;
			while ((*s & 0xc0) == 0x80)
				s++;
		}
	}
	dst[k] = 0;
}

/* at most w pixels of s, with "..." when cut */
static void
fit_text(char *dst, size_t n, const char *s, int w)
{
	int maxc = w / FONT_6X10_W;

	ascii_of(dst, n, s);
	if (maxc > 3 && (int)strlen(dst) > maxc) {
		dst[maxc - 3] = 0;
		strcat(dst, "...");
	}
}

/* ---- anti-aliased curves: coverage, then tints ---- */

static int
soft(int c)
{
	return c == P_BG || c == P_GRID || c == P_PANE || c == P_PANEGRID ||
	    c == P_LEGBG || c == P_ZERO || c == P_UNDEF ||
	    (c >= P_CM && (c - P_CM) % NSH == 5);
}

/* clip a segment to [x0, x1] x [y0, y1] (Liang-Barsky); 0 if outside */
static int
clipseg(double *ax, double *ay, double *bx, double *by, double x0, double y0,
    double x1, double y1)
{
	double t0 = 0, t1 = 1, dx = *bx - *ax, dy = *by - *ay;
	double p[4] = {-dx, dx, -dy, dy};
	double q[4] = {*ax - x0, x1 - *ax, *ay - y0, y1 - *ay};
	int i;

	for (i = 0; i < 4; i++) {
		if (p[i] == 0) {
			if (q[i] < 0)
				return 0;
		} else {
			double r = q[i] / p[i];
			if (p[i] < 0) {
				if (r > t1)
					return 0;
				if (r > t0)
					t0 = r;
			} else {
				if (r < t0)
					return 0;
				if (r < t1)
					t1 = r;
			}
		}
	}
	*bx = *ax + t1 * dx;
	*by = *ay + t1 * dy;
	*ax = *ax + t0 * dx;
	*ay = *ay + t0 * dy;
	return 1;
}

static void
cov_seg(double x0, double y0, double x1, double y1, double w)
{
	double r = w * 0.5 + 0.6, dx, dy, l2;
	int i, j, ia, ib, ja, jb;

	if (!isfinite(x0) || !isfinite(y0) || !isfinite(x1) || !isfinite(y1))
		return;
	if (!clipseg(&x0, &y0, &x1, &y1, C.cx0 - r, C.cy0 - r, C.cx1 + r,
	    C.cy1 + r))
		return;
	dx = x1 - x0;
	dy = y1 - y0;
	l2 = dx * dx + dy * dy;
	ia = (int)floor((x0 < x1 ? x0 : x1) - r);
	ib = (int)ceil((x0 > x1 ? x0 : x1) + r);
	ja = (int)floor((y0 < y1 ? y0 : y1) - r);
	jb = (int)ceil((y0 > y1 ? y0 : y1) + r);
	if (ia < C.cx0)
		ia = C.cx0;
	if (ja < C.cy0)
		ja = C.cy0;
	if (ib > C.cx1)
		ib = C.cx1;
	if (jb > C.cy1)
		jb = C.cy1;
	for (j = ja; j <= jb; j++)
		for (i = ia; i <= ib; i++) {
			double t = l2 > 0 ? ((i - x0) * dx + (j - y0) * dy) / l2 : 0, ex,
			    ey, d, c;
			float *cp = &C.cov[(size_t)j * C.W + i];
			if (t < 0)
				t = 0;
			if (t > 1)
				t = 1;
			ex = i - (x0 + t * dx);
			ey = j - (y0 + t * dy);
			d = sqrt(ex * ex + ey * ey);
			c = w * 0.5 + 0.5 - d;
			if (c > *cp)
				*cp = (float)(c > 1 ? 1 : c);
		}
}

static void
cov_disc(double x, double y, double rad)
{
	int i, j;

	for (j = (int)floor(y - rad - 1); j <= (int)ceil(y + rad + 1); j++)
		for (i = (int)floor(x - rad - 1); i <= (int)ceil(x + rad + 1); i++) {
			double d = sqrt((i - x) * (i - x) + (j - y) * (j - y)), c;
			float *cp;
			if (i < C.cx0 || j < C.cy0 || i > C.cx1 || j > C.cy1)
				continue;
			cp = &C.cov[(size_t)j * C.W + i];
			c = rad + 0.5 - d;
			if (c > *cp)
				*cp = (float)(c > 1 ? 1 : c);
		}
}

/* the coverage into the canvas with tints t[0] (full) .. t[3] (faint) */
static void
cov_put(const int *t)
{
	int i, j;

	for (j = C.cy0; j <= C.cy1; j++)
		for (i = C.cx0; i <= C.cx1; i++) {
			float *cp = &C.cov[(size_t)j * C.W + i];
			unsigned char *pp = &C.px[(size_t)j * C.W + i];
			float c = *cp;
			int l;
			if (c <= 0.07f)
				continue;
			*cp = 0;
			l = c > 0.8f ? 0 : c > 0.55f ? 1 : c > 0.3f ? 2 : 3;
			if (l == 0 || soft(*pp))
				*pp = (unsigned char)t[l];
		}
}

static void
cov_ser(int s)
{
	int t[4], k;

	for (k = 0; k < 4; k++)
		t[k] = P_SER + (s % NSER) * 4 + k;
	cov_put(t);
}

static void
cov_solid(int c)
{
	int t[4] = {c, c, P_AXIS, P_LEGBORDER}, b = -1, k;

	if (c == P_AXIS)
		b = 0;
	else if (c == P_WIRE || c == P_TEXT)
		b = 1;
	else if (c == P_MARK)
		b = 2;
	if (b >= 0)
		for (k = 1; k < 4; k++)
			t[k] = P_TINT + b * 3 + k - 1;
	cov_put(t);
}

/* a polyline in pixels; NAN in x breaks it */
static void
poly(const double *x, const double *y, int n, double w)
{
	int i;

	for (i = 0; i + 1 < n; i++)
		if (!isnan(x[i]) && !isnan(x[i + 1]))
			cov_seg(x[i], y[i], x[i + 1], y[i + 1], w);
	for (i = 0; i < n; i++)  /* isolated points still show */
		if (!isnan(x[i]) && (i == 0 || isnan(x[i - 1])) &&
		    (i + 1 == n || isnan(x[i + 1])))
			cov_disc(x[i], y[i], w * 0.6);
}

/* ---- numbers and ticks ---- */

static int
dcmp(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

static const char *
g4(double v)
{
	static char buf[8][32];
	static int k;
	char *b = buf[k++ & 7];

	if (v == 0)
		v = 0;		/* no -0 */
	snprintf(b, 32, "%.4g", v);
	return b;
}

static void
fmt_tick(char *b, size_t n, double v, double step)
{
	double a;
	char *e;

	if (fabs(v) < fabs(step) * 1e-9)
		v = 0;
	a = fabs(v);
	if (a != 0 && (a >= 1e6 || a < 1e-4 || step < 1e-4)) {
		snprintf(b, n, "%.3g", v);
		if ((e = strchr(b, 'e')) != NULL) {	/* 1e+06 -> 1e6 */
			char *d = e + 1;
			if (*d == '+')
				memmove(d, d + 1, strlen(d));
			if (*d == '-')
				d++;
			while (*d == '0' && d[1])
				memmove(d, d + 1, strlen(d));
		}
		return;
	}
	{	/* enough decimals to tell the ticks apart: 2.5, 0.25 */
		int dec = 0;
		while (dec < 8 && fabs(step * pow(10, dec) - floor(step * pow(10, dec) +
		    0.5)) > 1e-6 * step * pow(10, dec))
			dec++;
		snprintf(b, n, "%.*f", dec, v);
	}
}

/* m quarters of pi, as text: -3pi/4, pi, 2pi */
static void
fmt_pi(char *b, size_t n, long m)
{
	long num = m, den = 4, a;

	if (m == 0) {
		snprintf(b, n, "0");
		return;
	}
	while (den > 1 && num % 2 == 0) {
		num /= 2;
		den /= 2;
	}
	a = labs(num);
	if (a == 1)
		snprintf(b, n, "%spi%s", num < 0 ? "-" : "",
		    den == 1 ? "" : den == 2 ? "/2" : "/4");
	else
		snprintf(b, n, "%s%ldpi%s", num < 0 ? "-" : "", a,
		    den == 1 ? "" : den == 2 ? "/2" : "/4");
}

/* lo and hi are multiples of pi/4 (and the span at least pi) */
static int
pi_axis(double lo, double hi)
{
	double q = M_PI / 4, a = lo / q, b = hi / q;

	return hi - lo >= M_PI * 0.99 && hi - lo <= 64 * M_PI &&
	    fabs(a - floor(a + 0.5)) < 1e-6 && fabs(b - floor(b + 0.5)) < 1e-6 &&
	    (fabs(lo) > 1e-9 || fabs(hi) > 1e-9);
}

#define MAXTICK 24
struct ticks {
	int n;
	double v[MAXTICK];
	char lab[MAXTICK][20];
};

static double
nice_step(double span, int maxn)
{
	double raw = span / (maxn > 1 ? maxn : 1), mag, r;

	if (!(raw > 0))
		return 1;
	mag = pow(10, floor(log10(raw)));
	r = raw / mag;
	return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 2.5 ? 2.5 : r <= 5 ? 5 : 10) * mag;
}

/* integer axes: [0] horizontal, [1] vertical, [2] values (z, colour bar) */
static int tick_int[3];

static void
mkticks(struct ticks *t, double lo, double hi, int lg, int maxn, int isint)
{
	double step, v;
	int k;

	t->n = 0;
	if (!(hi > lo))
		return;
	if (lg) {
		int a = (int)ceil(log10(lo) - 1e-9), b = (int)floor(log10(hi) + 1e-9);
		int every = 1;
		while ((b - a) / every + 1 > maxn)
			every++;
		for (k = a; k <= b && t->n < MAXTICK; k += every) {
			v = pow(10, k);
			t->v[t->n] = v;
			if (k >= -3 && k <= 4)
				fmt_tick(t->lab[t->n], sizeof t->lab[0], v, k < 0 ? v : 1);
			else
				snprintf(t->lab[t->n], sizeof t->lab[0], "1e%d", k);
			t->n++;
		}
		if (b - a < 2) {	/* a short log axis: 2 and 5 too */
			for (k = a - 1; k <= b; k++) {
				int m;
				for (m = 2; m <= 5; m += 3) {
					v = m * pow(10, k);
					if (v >= lo && v <= hi && t->n < MAXTICK) {
						t->v[t->n] = v;
						fmt_tick(t->lab[t->n], sizeof t->lab[0], v,
						    k < 0 ? pow(10, k) : 1);
						t->n++;
					}
				}
			}
		}
		return;
	}
	if (!isint && pi_axis(lo, hi)) {
		long q = 1, m, ma, mb;
		while ((hi - lo) / (q * M_PI / 4) > maxn)
			q *= 2;
		ma = (long)ceil(lo / (M_PI / 4) / q - 1e-6) * q;
		mb = (long)floor(hi / (M_PI / 4) / q + 1e-6) * q;
		for (m = ma; m <= mb && t->n < MAXTICK; m += q) {
			t->v[t->n] = m * M_PI / 4;
			fmt_pi(t->lab[t->n], sizeof t->lab[0], m);
			t->n++;
		}
		return;
	}
	step = nice_step(hi - lo, maxn);
	if (isint && step < 1)
		step = 1;
	else if (isint && step != floor(step))
		step = floor(step);	/* 2.5 is 2 */
	for (k = (int)ceil(lo / step - 1e-9); t->n < MAXTICK; k++) {
		v = k * step;
		if (v > hi + step * 1e-9)
			break;
		t->v[t->n] = v;
		fmt_tick(t->lab[t->n], sizeof t->lab[0], v, step);
		t->n++;
	}
}

/* ---- evaluation ---- */

/* ok codes: 0 undefined (outside the domain, NaN), 1 a value, 2 infinite */
struct ev {
	struct func f;
	struct plan p;
	uint32_t loc, wloc;
	nid root;
	ty_t ty;
	int64_t *ii;
	double *ir;
	uint8_t ok[CH];
};

static void
ev_open(struct ev *e, const struct func *f, nid root)
{
	nid r[2];
	uint32_t n = 0;

	memset(e, 0, sizeof *e);
	e->f = *f;
	r[n++] = root;
	if (f->where)
		r[n++] = f->where;
	plan_build(&e->p, r, n, 0);
	e->loc = plan_loc(&e->p, root);
	e->wloc = f->where ? plan_loc(&e->p, f->where) : NOTOK;
	e->ty = N.ty[root];
	e->root = root;
	e->ii = xcalloc((size_t)MAXP * CH, sizeof *e->ii);
	e->ir = xcalloc((size_t)MAXP * CH, sizeof *e->ir);
}

static void
ev_close(struct ev *e)
{
	plan_free(&e->p);
	free(e->ii);
	free(e->ir);
	memset(e, 0, sizeof *e);
}

/* n points, in[i * np + j] for parameter j */
static void
ev_run(struct ev *e, int n, const double *in, double *out, uint8_t *ok)
{
	int np = e->f.np, base, m, i, j;

	for (base = 0; base < n; base += CH) {
		m = n - base < CH ? n - base : CH;
		for (j = 0; j < np; j++) {
			ty_t t = e->f.p[j].ty;
			for (i = 0; i < m; i++) {
				double v = in[(size_t)(base + i) * np + j];
				if (TY_FP(t)) {
					if (TYK(t) == K_FLOAT)
						v = fround(TYW(t), v);
					e->ir[j * CH + i] = v;
					e->ii[j * CH + i] = 0;
				} else {
					e->ii[j * CH + i] = (int64_t)llround(v);
					e->ir[j * CH + i] = v;
				}
			}
		}
		plan_run(&e->p, (uint32_t)m, e->ii, e->ir);
		valid_mask(&e->p, e->wloc, (uint32_t)m, e->ok);
		for (i = 0; i < m; i++) {
			uint8_t fl = COLF(&e->p, e->loc)[i];
			double v;
			if (TY_FP(e->ty))
				v = COLR(&e->p, e->loc)[i];
			else if (TYK(e->ty) == K_UINT)
				v = (double)(uint64_t)COLI(&e->p, e->loc)[i];
			else
				v = (double)COLI(&e->p, e->loc)[i];
			out[base + i] = v;
			if (!e->ok[i] || (fl & F_UNDEF) || isnan(v))
				ok[base + i] = 0;
			else if (isinf(v))
				ok[base + i] = 2;
			else
				ok[base + i] = 1;
		}
	}
}

static int
ev1(struct ev *e, const double *in, double *v)
{
	uint8_t ok;

	ev_run(e, 1, in, v, &ok);
	return ok;
}

/* ---- the request ---- */

#define MAXSER 8
enum {
	S_FX, S_IMPL, S_REGION, S_PARAM, S_POLAR, S_SPACE, S_VFIELD, S_SLOPE,
	S_SURF, S_ND, S_DERIV, S_TANGENT
};
enum { V_AUTO, V_SURF, V_CONTOUR, V_HEAT, V_WIRE, V_FLAT };

struct ser {
	int kind;
	struct func f[3];
	int nf;
	int sx, sy;			/* the slots on the two axes (or the curve's t) */
	double fix[3][MAXP];		/* the other slots' values */
	char label[200];
	int col;
	double tm, tc, ta;		/* tangent: y = tm*x + tc at x = ta */
	int has_ic;
	double ic_t, ic_y;
	uint32_t xname, yname;		/* axis names this series implies */
	double *cx, *cy;		/* a curve's points (data), NAN breaks */
	int cn;
	double t0, t1;			/* a curve's parameter range */
	nid bound;			/* a region's boundary function */
};

struct rng {
	uint32_t name;
	double lo, hi;
};

static struct {
	int mode;			/* 0 plot, 1 plot3d, 2 neighbourhood */
	struct ser s[MAXSER];
	int ns;
	struct rng r[MAXP + 4];
	int nr;
	char name[64];
	int nofile, view, rotate, fps, frames, logx, logy, deriv, tangent;
	int has_ic;			/* with y(0) = 1 */
	uint32_t ic_name;
	double ic_t, ic_y;
	double seconds, tan_at;
	int nat;
	double at[2];
	int npts, polar, field, slices, parallel, anim;
	uint32_t animvar;
	char text[4096];
} O;

static struct rng *
rng_find(uint32_t name)
{
	int k;

	for (k = 0; k < O.nr; k++)
		if (O.r[k].name == name)
			return &O.r[k];
	return NULL;
}

static int
slot_named(const struct func *f, const char *nm)
{
	uint32_t s = sym_cstr(nm);
	int k;

	for (k = 0; k < f->np; k++)
		if (f->p[k].name == s)
			return k;
	return -1;
}

/* ---- the statement's text: tokens with their nesting depth ---- */

struct tk {
	int k;		/* 'w' word, 'n' number, 'D' "..", 'E' a two-character
			 * operator, '=' assignment, else the character */
	int a, b, d;
};
static struct tk T[512];
static int nt;

static void
tk_lex(const char *s)
{
	int i = 0, d = 0;

	nt = 0;
	while (s[i] && nt < 511) {
		struct tk *t = &T[nt];
		unsigned char c = (unsigned char)s[i];
		if (isspace(c)) {
			i++;
			continue;
		}
		t->a = i;
		if (isdigit(c) && (s[i + 1] == 'd' || s[i + 1] == 'D') &&
		    !isalnum((unsigned char)s[i + 2]) && s[i + 2] != '_') {
			t->k = 'w';	/* 3d, 2d */
			i += 2;
		} else if (isalpha(c) || c == '_' || c >= 0x80) {
			t->k = 'w';
			while (isalnum((unsigned char)s[i]) || s[i] == '_' ||
			    (unsigned char)s[i] >= 0x80)
				i++;
			while (s[i] == '\'')
				i++;
		} else if (isdigit(c) || (c == '.' && isdigit((unsigned char)s[i + 1]))) {
			t->k = 'n';
			while (isdigit((unsigned char)s[i]) || (s[i] == '.' &&
			    s[i + 1] != '.'))
				i++;
			if ((s[i] == 'e' || s[i] == 'E') && (isdigit((unsigned char)
			    s[i + 1]) || ((s[i + 1] == '-' || s[i + 1] == '+') &&
			    isdigit((unsigned char)s[i + 2])))) {
				i += 2;
				while (isdigit((unsigned char)s[i]))
					i++;
			}
		} else if (c == '.' && s[i + 1] == '.') {
			t->k = 'D';
			i += 2;
		} else if ((c == '=' || c == '<' || c == '>' || c == '!') &&
		    s[i + 1] == '=') {
			t->k = 'E';
			i += 2;
		} else if (c == '-' && s[i + 1] == '>') {
			t->k = 'E';
			i += 2;
		} else {
			t->k = c;
			i++;
		}
		t->b = i;
		if (t->k == ')' || t->k == ']' || t->k == '}')
			d--;
		t->d = d;
		if (t->k == '(' || t->k == '[' || t->k == '{')
			d++;
		nt++;
	}
}

static int
tw(int i, const char *w)
{
	size_t n = strlen(w);

	return i < nt && T[i].k == 'w' && (size_t)(T[i].b - T[i].a) == n &&
	    !strncmp(O.text + T[i].a, w, n);
}

static void
tk_str(int a, int b, char *buf, size_t n)
{
	size_t len;

	if (a >= b) {
		buf[0] = 0;
		return;
	}
	len = (size_t)(T[b - 1].b - T[a].a);
	if (len >= n)
		len = n - 1;
	memcpy(buf, O.text + T[a].a, len);
	buf[len] = 0;
}

static const char *const optkw[] = {
	"to", "nofile", "3d", "2d", "surface", "surf", "contour", "contours",
	"heatmap", "heat", "density", "wireframe", "wire", "mesh", "rotate",
	"rotating", "spin", "fps", "frames", "seconds", "logx", "logy",
	"loglog", "deriv", "derivative", "derivatives", "tangent", "at",
	"around", "near", "points", "samples", "polar", "field", "slices",
	"parallel", "animate", "animated", "over", "for", "with",
	"flat", "nd", "Nd", "ND", "4d", "5d", "6d", NULL
};

static int
optstart(int i)
{
	int k;

	if (i >= nt || T[i].d != 0 || T[i].k != 'w')
		return 0;
	for (k = 0; optkw[k]; k++)
		if (tw(i, optkw[k]))
			return 1;
	/* with y(0) = 1: an initial value */
	if (i > 0 && (tw(i - 1, "with") || tw(i - 1, "and")) && i + 1 < nt &&
	    T[i + 1].k == '(')
		return 1;
	if (tw(i + 1, "in") || tw(i + 1, "from"))
		return 1;
	if (tw(i, "in") && (tw(i + 1, "3d") || tw(i + 1, "2d") || tw(i + 1, "polar")
	    || tw(i + 1, "parallel")))
		return 1;	/* in 3d */
	if (i + 1 < nt && T[i + 1].k == '=') {	/* x = 0..5 */
		int j;
		for (j = i + 2; j < nt && !(T[j].d == 0 && T[j].k == ','); j++)
			if (T[j].k == 'D' && T[j].d == 0)
				return 1;
			else if (T[j].k == 'w' && !tw(j, "pi") && !tw(j, "e"))
				break;	/* y' = y - y^2 t in 0..8: not a range */
	}
	return 0;
}

/* a constant written in tokens a..b-1 */
static double
const_tok(int a, int b)
{
	char buf[256];
	struct func tmp;
	nid e;

	tk_str(a, b, buf, sizeof buf);
	if (!buf[0])
		fail("plot: expected a number");
	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr("_");
	e = calc_parse_text(buf, &tmp, TY_REAL);
	if (tmp.np || !is_const(e))
		fail("plot: expected a number, found '%s'", buf);
	return TY_FP(N.ty[e]) ? N.kr[e] : (double)N.ki[e];
}

/* the end of a value starting at i: before '..', ',' or an option */
static int
val_end(int i, int stop_to)
{
	int j = i;

	while (j < nt) {
		if (T[j].d == 0 && (T[j].k == 'D' || T[j].k == ',' ||
		    (j > i && optstart(j)) || (stop_to && tw(j, "to"))))
			break;
		if (T[j].d < 0)
			break;
		j++;
	}
	return j;
}

static double
val_at(int *i, int stop_to)
{
	int e = val_end(*i, stop_to);
	double v;

	if (e == *i)
		fail("plot: expected a number after '%.*s'", T[*i - 1].b -
		    T[*i - 1].a, O.text + T[*i - 1].a);
	v = const_tok(*i, e);
	*i = e;
	return v;
}

/* [a, b] / (a, b) / a..b / a to b, from token *i */
static void
range_at(int *i, double *lo, double *hi)
{
	int j = *i;

	if (j < nt && (T[j].k == '[' || T[j].k == '(')) {
		int e, d0 = T[j].d + 1;
		for (e = j + 1; e < nt && !(T[e].d == d0 && T[e].k == ','); e++)
			;
		if (e >= nt)
			fail("plot: expected [a, b]");
		*lo = const_tok(j + 1, e);
		j = e + 1;
		for (e = j; e < nt && !(T[e].d == d0 - 1 && (T[e].k == ']' ||
		    T[e].k == ')')); e++)
			;
		if (e >= nt)
			fail("plot: expected ']' after the interval");
		*hi = const_tok(j, e);
		*i = e + 1;
	} else {
		*lo = val_at(&j, 1);
		if (j < nt && (T[j].k == 'D' || tw(j, "to")))
			j++;
		else
			fail("plot: expected a..b");
		*hi = val_at(&j, 0);
		*i = j;
	}
	if (*lo > *hi) {
		double t = *lo;
		*lo = *hi;
		*hi = t;
	}
	if (!(*hi > *lo))
		fail("plot: an empty interval [%s, %s]", g4(*lo), g4(*hi));
}

static void
add_range(uint32_t name, double lo, double hi)
{
	struct rng *r = rng_find(name);

	if (r == NULL) {
		if (O.nr == MAXP + 4)
			fail("plot: too many ranges");
		r = &O.r[O.nr++];
	}
	r->name = name;
	r->lo = lo;
	r->hi = hi;
}

/* options from token i on; returns the first token that is not one */
static int
parse_opts(int i, int leading)
{
	char w[64];

	while (i < nt) {
		if (T[i].k == ',' && T[i].d == 0) {
			i++;
			continue;
		}
		if (T[i].k == 'n' && T[i].d == 0 && tw(i + 1, "fps")) {	/* 30 fps */
			O.fps = (int)const_tok(i, i + 1);
			i += 2;
			continue;
		}
		if (!optstart(i))
			break;
		tk_str(i, i + 1, w, sizeof w);
		if (leading && (tw(i + 1, "in") || tw(i + 1, "from")))
			break;	/* plot x in ... has no function */
		if (!strcmp(w, "to")) {
			if (i + 1 >= nt)
				fail("plot: to NAME");
			tk_str(i + 1, i + 2, O.name, sizeof O.name);
			i += 2;
		} else if (!strcmp(w, "nofile")) {
			O.nofile = 1;
			i++;
		} else if (!strcmp(w, "3d") || !strcmp(w, "surface") ||
		    !strcmp(w, "surf")) {
			O.view = V_SURF;
			i++;
		} else if (!strcmp(w, "contour") || !strcmp(w, "contours")) {
			O.view = V_CONTOUR;
			i++;
		} else if (!strcmp(w, "heatmap") || !strcmp(w, "heat") ||
		    !strcmp(w, "density")) {
			O.view = V_HEAT;
			i++;
		} else if (!strcmp(w, "wireframe") || !strcmp(w, "wire") ||
		    !strcmp(w, "mesh")) {
			O.view = V_WIRE;
			i++;
		} else if (!strcmp(w, "2d") || !strcmp(w, "flat")) {
			O.view = V_FLAT;
			i++;
		} else if (!strcmp(w, "nd") || !strcmp(w, "Nd") ||
		    !strcmp(w, "ND") || !strcmp(w, "4d") || !strcmp(w, "5d") ||
		    !strcmp(w, "6d")) {
			i++;	/* N-d: the view follows the number of variables */
		} else if (!strcmp(w, "rotate") || !strcmp(w, "rotating") ||
		    !strcmp(w, "spin")) {
			O.rotate = 1;
			i++;
		} else if (!strcmp(w, "fps") || !strcmp(w, "frames") ||
		    !strcmp(w, "seconds") || !strcmp(w, "points") ||
		    !strcmp(w, "samples")) {
			double v;
			i++;
			v = val_at(&i, 0);
			if (!strcmp(w, "fps"))
				O.fps = (int)v;
			else if (!strcmp(w, "frames"))
				O.frames = (int)v;
			else if (!strcmp(w, "seconds"))
				O.seconds = v;
			else
				O.npts = (int)v;
		} else if (!strcmp(w, "logx") || !strcmp(w, "logy") ||
		    !strcmp(w, "loglog")) {
			if (w[3] != 'y')
				O.logx = 1;
			if (w[3] != 'x')
				O.logy = 1;
			i++;
		} else if (!strcmp(w, "deriv") || !strcmp(w, "derivative") ||
		    !strcmp(w, "derivatives")) {
			O.deriv = 1;
			i++;
		} else if (!strcmp(w, "tangent")) {
			i++;
			if (tw(i, "at"))
				i++;
			O.tangent = 1;
			O.tan_at = val_at(&i, 0);
		} else if (!strcmp(w, "at") || !strcmp(w, "around") ||
		    !strcmp(w, "near")) {
			i++;
			if (i < nt && T[i].k == '(') {	/* at (a, b) */
				int e, d0 = T[i].d + 1;
				for (e = i + 1; e < nt && !(T[e].d == d0 && T[e].k == ',');
				    e++)
					;
				if (e < nt) {
					int f;
					for (f = e + 1; f < nt && !(T[f].d == d0 - 1 &&
					    T[f].k == ')'); f++)
						;
					O.at[0] = const_tok(i + 1, e);
					O.at[1] = const_tok(e + 1, f);
					O.nat = 2;
					i = f + 1;
					continue;
				}
			}
			if (tw(val_end(i, 0), "fps")) {	/* at 30 fps */
				O.fps = (int)val_at(&i, 0);
				i++;
				continue;
			}
			O.at[0] = val_at(&i, 0);
			O.nat = 1;
		} else if (!strcmp(w, "polar")) {
			O.polar = 1;
			i++;
		} else if (!strcmp(w, "field")) {
			O.field = 1;
			i++;
		} else if (!strcmp(w, "slices")) {
			O.slices = 1;
			i++;
		} else if (!strcmp(w, "parallel")) {
			O.parallel = 1;
			i++;
		} else if (!strcmp(w, "animate") || !strcmp(w, "animated") ||
		    !strcmp(w, "over")) {
			O.anim = 1;
			i++;
			if (tw(i, "over"))
				i++;
			if (i < nt && T[i].k == 'w' && (!optstart(i) ||
			    tw(i + 1, "in") || tw(i + 1, "from"))) {
				char v[64];
				tk_str(i, i + 1, v, sizeof v);
				O.animvar = sym_cstr(v);
				if (!tw(i + 1, "in") && !tw(i + 1, "from"))
					i++;
			}
		} else if (!strcmp(w, "for") || !strcmp(w, "with") ||
		    !strcmp(w, "and") || !strcmp(w, "in")) {
			i++;
		} else if (i + 1 < nt && T[i + 1].k == '(') {
			/* NAME(t0) = v: the initial value of a slope field */
			int j = i + 2, e;
			while (j < nt && !(T[j].k == ')' && T[j].d == 0))
				j++;
			if (j >= nt || j + 1 >= nt || T[j + 1].k != '=')
				fail("plot: an initial value is NAME(t0) = value");
			tk_str(i, i + 1, w, sizeof w);
			O.ic_name = sym_cstr(w);
			O.ic_t = const_tok(i + 2, j);
			e = val_end(j + 2, 0);
			O.ic_y = const_tok(j + 2, e);
			O.has_ic = 1;
			i = e;
		} else {
			/* NAME in a..b, NAME from a to b, NAME = a..b */
			double lo, hi;
			char v[64];
			tk_str(i, i + 1, v, sizeof v);
			i += 2;
			range_at(&i, &lo, &hi);
			add_range(sym_cstr(v), lo, hi);
		}
	}
	return i;
}

/* ---- features of a function of one variable ---- */

/* a slice: the function of slot s, the other slots at base[] */
static int
f1(struct ev *e, const double *base, int s, double x, double *v)
{
	double in[MAXP];
	int np = e->f.np;

	memcpy(in, base, sizeof in);
	if (s >= 0 && s < np)
		in[s] = x;
	return ev1(e, in, v);
}

static void
f1n(struct ev *e, const double *base, int s, int n, const double *xs,
    double *out, uint8_t *ok)
{
	int np = e->f.np, i, j;
	double *in;

	if (np == 0) {
		ev_run(e, n, NULL, out, ok);
		return;
	}
	in = xmalloc((size_t)n * np * sizeof *in);
	for (i = 0; i < n; i++)
		for (j = 0; j < np; j++)
			in[(size_t)i * np + j] = j == s ? xs[i] : base[j];
	ev_run(e, n, in, out, ok);
	free(in);
}

static double
pi_frac(double v, char *buf, size_t n)
{
	static const int den[] = {1, 2, 3, 4, 6};
	int k, m;

	buf[0] = 0;
	for (k = 0; k < 5; k++) {
		double q = v / M_PI * den[k];
		m = (int)floor(q + 0.5);
		if (m != 0 && abs(m) <= 64 && fabs(q - m) < 1e-7) {
			char num[24];
			if (abs(m) == 1)
				snprintf(num, sizeof num, "%spi", m < 0 ? "-" : "");
			else
				snprintf(num, sizeof num, "%dpi", m);
			if (den[k] == 1)
				snprintf(buf, n, "%s", num);
			else
				snprintf(buf, n, "%s/%d", num, den[k]);
			return 1;
		}
	}
	return 0;
}

/* "6.283 (2pi)" */
static const char *
gpi(double v)
{
	static char buf[4][48];
	static int k;
	char *b = buf[k++ & 3], f[24];

	if (pi_frac(v, f, sizeof f))
		snprintf(b, 48, "%s (%s)", g4(v), f);
	else
		snprintf(b, 48, "%s", g4(v));
	return b;
}

/* f(x + P) = f(x) at the test points */
static int
match_p(struct ev *e, const double *base, int s, const double *xt,
    const double *yt, const uint8_t *ot, double *xs, double *ys, uint8_t *os,
    int n, double P, double tol)
{
	int i;

	for (i = 0; i < n; i++)
		xs[i] = xt[i] + P;
	f1n(e, base, s, n, xs, ys, os);
	for (i = 0; i < n; i++) {
		if ((ot[i] == 1) != (os[i] == 1))
			return 0;
		if (ot[i] == 1 && fabs(ys[i] - yt[i]) > tol)
			return 0;
	}
	return 1;
}

/*
 * Simple period detection along slot s: the smallest P with
 * f(x + P) = f(x) at 48 spread points (defined at the same points, equal
 * within 1e-6 of the range).  Multiples of pi/6, pi/4 and quarter units
 * first, then a scan of the mismatch over P refined by golden section.
 * (A branch adds a symbolic fn_period; this is the numeric fallback.)
 */
static int
guess_period(struct ev *e, const double *base, int s, double *period)
{
	enum { NT = 48, NG = 16001 };
	double xt[NT], yt[NT], xs[NT], ys[NT], lo = INFINITY, hi = -INFINITY;
	double tol, *g, *gx, best = 0;
	uint8_t ot[NT], os[NT], *og;
	int i, j, nd = 0, found = 0;
	static double cand[512];
	static int ncand;

	if (s < 0)
		return 0;
	/* props.c's period first: proved from the periodic subterms */
	{
		struct func pf = e->f;
		double pp;
		pf.body = e->root;
		if (fn_period(&pf, s, &pp) == 1 && isfinite(pp) && pp > 0) {
			*period = pp;
			return 1;
		}
	}
	for (i = 0; i < NT; i++)
		xt[i] = -7.3125 + i * 0.4173;
	f1n(e, base, s, NT, xt, yt, ot);
	for (i = 0; i < NT; i++)
		if (ot[i] == 1) {
			nd++;
			if (yt[i] < lo)
				lo = yt[i];
			if (yt[i] > hi)
				hi = yt[i];
		}
	if (nd < NT / 3 || !(hi - lo > 1e-12 * (1 + fabs(hi))))
		return 0;
	tol = (hi - lo) * (TYK(e->ty) == K_FLOAT || TYK(e->f.p[s].ty) ==
	    K_FLOAT ? 1e-4 : 1e-6);
	if (!TY_FP(e->f.p[s].ty))
		return 0;	/* integer inputs: not looked for */
	if (ncand == 0) {
		static const int den[] = {1, 2, 3, 4, 6};
		int k, m;
		for (k = 0; k < 5; k++)
			for (m = 1; m <= 24; m++)
				cand[ncand++] = m * M_PI / den[k];
		for (m = 1; m <= 96; m++)
			cand[ncand++] = m * 0.25;
		for (k = 0; k < ncand; k++)	/* sort */
			for (m = k + 1; m < ncand; m++)
				if (cand[m] < cand[k]) {
					double t = cand[k];
					cand[k] = cand[m];
					cand[m] = t;
				}
	}
	for (i = 0; i < ncand; i++)
		if (match_p(e, base, s, xt, yt, ot, xs, ys, os, NT, cand[i], tol)) {
			*period = cand[i];
			return 1;
		}
	/* the mismatch on a grid of step h over shifts up to 40 */
	{
		const double h = 0.005, x0 = -3.71;
		int stride = 79, npt = 100;
		g = xmalloc(NG * sizeof *g);
		gx = xmalloc(NG * sizeof *gx);
		og = xmalloc(NG);
		for (i = 0; i < NG; i++)
			gx[i] = x0 + i * h;
		f1n(e, base, s, NG, gx, g, og);
		for (j = 40; j + (npt - 1) * stride < NG && !found; j++) {
			double m = 0, mp = 0, mn = 0;
			int k, jj;
			for (jj = j - 1; jj <= j + 1; jj++) {
				double *mm = jj == j - 1 ? &mp : jj == j ? &m : &mn;
				for (k = 0; k < npt; k++) {
					int a = k * stride / 2, b = a + jj;
					if (b >= NG)
						break;
					if (og[a] != og[b])
						*mm += hi - lo;
					else if (og[a] == 1)
						*mm += fabs(g[b] - g[a]);
				}
			}
			if (m < mp && m <= mn && m / npt < (hi - lo) * 0.02) {
				/* golden section on the 48 points around j*h */
				double a = (j - 1) * h, b = (j + 1) * h, r = 0.6180339887;
				int it;
				for (it = 0; it < 50; it++) {
					double c = b - r * (b - a), d = a + r * (b - a), ec = 0,
					    ed = 0;
					int q;
					for (q = 0; q < NT; q++)
						xs[q] = xt[q] + c;
					f1n(e, base, s, NT, xs, ys, os);
					for (q = 0; q < NT; q++)
						if (ot[q] == 1 && os[q] == 1)
							ec += (ys[q] - yt[q]) * (ys[q] - yt[q]);
					for (q = 0; q < NT; q++)
						xs[q] = xt[q] + d;
					f1n(e, base, s, NT, xs, ys, os);
					for (q = 0; q < NT; q++)
						if (ot[q] == 1 && os[q] == 1)
							ed += (ys[q] - yt[q]) * (ys[q] - yt[q]);
					if (ec < ed)
						b = d;
					else
						a = c;
				}
				best = (a + b) / 2;
				if (match_p(e, base, s, xt, yt, ot, xs, ys, os, NT, best,
				    tol))
					found = 1;
			}
		}
		free(g);
		free(gx);
		free(og);
	}
	if (found)
		*period = best;
	return found;
}

enum { FT_ZERO, FT_MIN, FT_MAX, FT_INFL, FT_POLE, FT_JUMP, FT_EDGE, FT_HOLE,
	FT_N };
static const char *const ftname[FT_N] = {"zeros", "local minima",
	"local maxima", "inflection points", "poles", "jumps",
	"domain edges", "undefined points"};
static const char *const ftsing[FT_N] = {"zero", "local minimum",
	"local maximum", "inflection point", "pole", "jump", "domain edge",
	"undefined point"};
#define MAXFEAT 128
struct feats {
	int n;
	double x[MAXFEAT];
	int k[MAXFEAT];
};

static void
feat_add(struct feats *F, double x, int k, double tol)
{
	int i;

	for (i = 0; i < F->n; i++)
		if (fabs(F->x[i] - x) <= tol && (F->k[i] == k || k == FT_INFL ||
		    F->k[i] == FT_POLE || F->k[i] == FT_EDGE))
			return;
	if (F->n < MAXFEAT) {
		F->x[F->n] = x;
		F->k[F->n] = k;
		F->n++;
	}
}

/* where definedness (by_def) or the sign changes between a and b */
static double
bisect1(struct ev *e, const double *base, int s, double a, double b,
    int by_def)
{
	double fa, fm, m, a0 = a, b0 = b;
	int oka = f1(e, base, s, a, &fa) == 1, okm, it;

	for (it = 0; it < 60; it++) {
		m = (a + b) / 2;
		if (m == a || m == b)
			break;
		okm = f1(e, base, s, m, &fm) == 1;
		if (by_def ? okm == oka : (okm && oka && (fm > 0) == (fa > 0))) {
			a = m;
			fa = fm;
		} else
			b = m;
	}
	m = (a + b) / 2;
	return fabs(m) < 1e-9 * (b0 - a0) ? 0 : m;
}

/*
 * Does the jump between a and b stay when the interval shrinks?  0
 * continuous (steep), 1 a jump, 2 a pole (the values grow without
 * bound), with the place in *at.
 */
static int
jump_kind(struct ev *e, const double *base, int s, double a, double fa,
    double b, double fb, double *at)
{
	double d0 = fabs(fb - fa), m, fm, a0 = a, b0 = b;
	int it, ok;

	if (d0 <= 1e-12 * (fabs(fa) + fabs(fb)))
		return 0;	/* no jump at all (a hole between equal values) */

	for (it = 0; it < 64; it++) {
		m = (a + b) / 2;
		if (m == a || m == b)
			break;
		ok = f1(e, base, s, m, &fm);
		if (ok != 1) {	/* 1/x at exactly 0: look beside it */
			double m2 = m + (b - a) * 0.0381966;
			if (ok == 0 && f1(e, base, s, m2, &fm) == 1 && m2 < b)
				m = m2;
			else {
				*at = fabs(m) < 1e-9 * (b0 - a0) ? 0 : m;
				return ok == 2 || fabs(fa) + fabs(fb) > 1e3 * (d0 + 1) ? 2 : 1;
			}
		}
		if (fabs(fm - fa) > fabs(fb - fm)) {
			b = m;
			fb = fm;
		} else {
			a = m;
			fa = fm;
		}
		if (fabs(fb - fa) < d0 * 0.2)
			return 0;
	}
	*at = (a + b) / 2;
	if (fabs(*at) < 1e-9 * (b0 - a0))
		*at = 0;
	if (fabs(fb - fa) < d0 * 0.2)
		return 0;
	return fabs(fb - fa) > d0 * 4 || fabs(fa) > 1e6 * (d0 + 1) ? 2 : 1;
}

/* f with its body replaced by its derivative in slot 0 (0 if none) */
static int
deriv_func(const struct func *f, struct func *g)
{
	jmp_buf saved;
	int ok = 0;
	nid d = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0) {
		d = diff_node(f->body, 0);
		ok = 1;
	}
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	if (ok) {
		*g = *f;
		g->body = d;
	}
	return ok;
}

/* the features of slot s on [lo, hi] */
static void
find_feats(struct ev *e, const double *base, int s, double lo, double hi,
    struct feats *F)
{
	enum { NS = 2001 };
	double *xs = xmalloc(NS * sizeof *xs), *ys = xmalloc(NS * sizeof *ys);
	uint8_t *ok = xmalloc(NS);
	double ymin = INFINITY, ymax = -INFINITY, span, tol = (hi - lo) * 1e-6;
	double medd = 0;
	int i, isint = s >= 0 && !TY_FP(e->f.p[s].ty), allzero = 1;

	F->n = 0;
	for (i = 0; i < NS; i++)
		xs[i] = lo + (hi - lo) * i / (NS - 1);
	if (isint)
		for (i = 0; i < NS; i++)
			xs[i] = floor(xs[i] + 0.5);
	f1n(e, base, s, NS, xs, ys, ok);
	for (i = 0; i < NS; i++)
		if (ok[i] == 1) {
			if (ys[i] < ymin)
				ymin = ys[i];
			if (ys[i] > ymax)
				ymax = ys[i];
			if (ys[i] != 0)
				allzero = 0;
		}
	span = ymax > ymin ? ymax - ymin : 1;
	{	/* the typical step between samples */
		double *d = xmalloc(NS * sizeof *d);
		int m = 0;
		for (i = 0; i + 1 < NS; i++)
			if (ok[i] == 1 && ok[i + 1] == 1)
				d[m++] = fabs(ys[i + 1] - ys[i]);
		if (m) {
			qsort(d, m, sizeof *d, dcmp);
			medd = d[m / 2];
		}
		free(d);
	}
	for (i = 0; i + 1 < NS; i++) {
		double a = xs[i], b = xs[i + 1], c, at;
		int jk;
		if (a == b)
			continue;
		if (ok[i] == 2)
			feat_add(F, a, FT_POLE, tol);
		if (ok[i] == 0 && i > 0 && i + 1 < NS && ok[i - 1] == 1 &&
		    ok[i + 1] == 1 && !isint) {
			/* one undefined sample: a hole (sin(x)/x) or a pole */
			double l, r, d = (b - a) * 1e-3;
			if (f1(e, base, s, a - d, &l) == 1 && f1(e, base, s, a + d, &r) == 1) {
				if (jump_kind(e, base, s, xs[i - 1], ys[i - 1], b, ys[i + 1],
				    &at) == 2)
					feat_add(F, at, FT_POLE, tol);
				else
					feat_add(F, a, FT_HOLE, tol);
				continue;
			}
		}
		if ((ok[i] == 0) != (ok[i + 1] == 0) && ok[i] != 2 && ok[i + 1] != 2) {
			if ((ok[i] == 0 && i > 0 && ok[i - 1] == 1) || (ok[i + 1] == 0 &&
			    i + 2 < NS && ok[i + 2] == 1))
				continue;	/* the hole above */
			feat_add(F, isint ? (ok[i] ? a : b) : bisect1(e, base, s, a, b, 1),
			    FT_EDGE, tol);
			continue;
		}
		if (ok[i] != 1 || ok[i + 1] != 1)
			continue;
		if (ys[i] == 0 && !allzero && (i == 0 || ys[i - 1] != 0 ||
		    ok[i - 1] != 1))
			feat_add(F, a, FT_ZERO, tol);
		if ((ys[i] < 0 && ys[i + 1] > 0) || (ys[i] > 0 && ys[i + 1] < 0)) {
			if (isint) {
				feat_add(F, fabs(ys[i]) < fabs(ys[i + 1]) ? a : b, FT_ZERO,
				    tol);
				continue;
			}
			jk = jump_kind(e, base, s, a, ys[i], b, ys[i + 1], &at);
			if (jk == 0) {
				c = bisect1(e, base, s, a, b, 0);
				feat_add(F, c, FT_ZERO, tol);
			} else
				feat_add(F, at, jk == 2 ? FT_POLE : FT_JUMP, tol);
		} else if (!isint && fabs(ys[i + 1] - ys[i]) > span * 1e-3 &&
		    fabs(ys[i + 1] - ys[i]) > 20 * medd) {
			jk = jump_kind(e, base, s, a, ys[i], b, ys[i + 1], &at);
			if (jk)
				feat_add(F, at, jk == 2 ? FT_POLE : FT_JUMP, tol);
		}
	}
	/* extrema and inflections: from the derivative when there is one */
	if (!isint && e->f.np == 1 && s == 0) {
		struct func f = e->f, g;
		double r[64], v, vl, vr, h = (hi - lo) * 1e-5;
		int n = crit_roots(&f, lo, hi, r, 64), k, okc;
		if (n >= 48)
			n = 0;	/* a derivative that is zero on whole stretches */
		for (k = 0; k < n; k++) {
			okc = f1(e, base, s, r[k], &v);
			if (okc != 1 || fabs(v) > 1e3 * span + fabs(ymax) + fabs(ymin)) {
				feat_add(F, r[k], FT_POLE, tol);
				continue;
			}
			if (f1(e, base, s, r[k] - h, &vl) == 1 &&
			    f1(e, base, s, r[k] + h, &vr) == 1) {
				if (vl > v && vr > v)
					feat_add(F, r[k], FT_MIN, tol);
				else if (vl < v && vr < v)
					feat_add(F, r[k], FT_MAX, tol);
			}
		}
		if (deriv_func(&f, &g)) {
			n = crit_roots(&g, lo, hi, r, 64);
			if (n >= 48)
				n = 0;
			for (k = 0; k < n; k++) {
				/* the curvature changes sign there */
				double d = (hi - lo) * 1e-4, q[5];
				int j2, good = 1;
				for (j2 = 0; j2 < 5; j2++)
					good &= f1(e, base, s, r[k] + (j2 - 2) * d, &q[j2]) == 1;
				if (!good || fabs(q[2]) > 1e3 * span + fabs(ymax) + fabs(ymin))
					continue;
				if ((q[0] - 2 * q[1] + q[2]) * (q[2] - 2 * q[3] + q[4]) < 0)
					feat_add(F, r[k], FT_INFL, tol * 100);
			}
		}
	} else {
		/* sampled: strict local extrema of the samples */
		for (i = 1; i + 1 < NS; i++)
			if (ok[i - 1] == 1 && ok[i] == 1 && ok[i + 1] == 1 &&
			    xs[i] != xs[i - 1] && xs[i] != xs[i + 1]) {
				if (ys[i] < ys[i - 1] && ys[i] <= ys[i + 1])
					feat_add(F, xs[i], FT_MIN, tol);
				else if (ys[i] > ys[i - 1] && ys[i] >= ys[i + 1])
					feat_add(F, xs[i], FT_MAX, tol);
			}
	}
	free(xs);
	free(ys);
	free(ok);
}

/* is f undefined everywhere on (a, b) (sampled)? */
static int
undef_on(struct ev *e, const double *base, int s, double a, double b)
{
	double xs[33], ys[33];
	uint8_t ok[33];
	int i;

	for (i = 0; i < 33; i++)
		xs[i] = a + (b - a) * (i + 0.5) / 33;
	f1n(e, base, s, 33, xs, ys, ok);
	for (i = 0; i < 33; i++)
		if (ok[i] == 1)
			return 0;
	return 1;
}

static void
print_feats(const struct feats *F, const char *vn)
{
	int k, i, c;

	for (k = 0; k < FT_N; k++) {
		struct sbuf b = {0};
		for (i = c = 0; i < F->n; i++)
			if (F->k[i] == k) {
				if (c < 8)
					sb_printf(&b, "%s%s", c ? ", " : "", gpi(F->x[i]));
				c++;
			}
		if (c > 8)
			sb_printf(&b, ", ... (%d in all)", c);
		if (c)
			printf("    %s: %s ≈ %s\n", ftname[k], vn, b.p);
		free(b.p);
	}
}

/*
 * Bounds for slot s: declared range, small machine type, period, else
 * the features' hull with a margin (at: a window around that point
 * scaled to the nearest feature or the curvature).  why gets the reason.
 */
static void
auto1(struct ev *e, const double *base, int s, int at_set, double at,
    double *plo, double *phi, struct sbuf *why, int verbose)
{
	const struct param *pp = s >= 0 ? &e->f.p[s] : NULL;
	const char *vn = pp ? sym_str(pp->name) : "x";
	struct feats F;
	double P, lo = 0, hi = 0, w;
	int k, isint = pp && !TY_FP(pp->ty), periodic = 0;

	F.n = 0;
	if (pp && pp->ranged && !at_set) {
		*plo = TY_FP(pp->ty) ? pp->rlo : (double)pp->lo;
		*phi = TY_FP(pp->ty) ? pp->rhi : (double)pp->hi;
		sb_printf(why, "%s's declared range", vn);
		if (*phi > *plo)
			return;
	}
	if (pp && TY_SIZED(pp->ty) && !at_set) {
		int64_t a, b;
		param_bounds(pp, &a, &b);
		if ((double)b - (double)a <= 4096) {
			*plo = (double)a;
			*phi = (double)b;
			sb_printf(why, "every %s value", ty_str(pp->ty));
			return;
		}
	}
	periodic = !isint && guess_period(e, base, s, &P);
	if (periodic && !at_set) {
		lo = -P;
		hi = P;
		sb_printf(why, "periodic with period %s: two periods", gpi(P));
		if (verbose) {
			find_feats(e, base, s, 0, P, &F);
			printf("  %s: one period [0, %s]:\n", vn, gpi(P));
			print_feats(&F, vn);
		}
		*plo = lo;
		*phi = hi;
		return;
	}
	if (at_set) {
		double bl = INFINITY, br = INFINITY, v0, vl, vr, h;
		int kl = -1, kr = -1, kb;
		find_feats(e, base, s, at - 10, at + 10, &F);
		for (k = 0; k < F.n; k++) {
			double d = F.x[k] - at;
			if (d < -1e-7 * (1 + fabs(at)) && -d < bl) {
				bl = -d;
				kl = k;
			}
			if (d > 1e-7 * (1 + fabs(at)) && d < br) {
				br = d;
				kr = k;
			}
		}
		/* the nearest feature on each side; the window takes both in */
		kb = kl < 0 ? kr : kr < 0 ? kl : bl > br ? kl : kr;
		if (kb >= 0) {
			double best = fabs(F.x[kb] - at);
			w = best * 1.4;
			sb_printf(why, "around %s = %s: features on %s within %s (%s %s "
			    "at %s)", vn, g4(at), kl >= 0 && kr >= 0 ? "both sides" :
			    "one side", g4(best), strchr("aeiou", ftsing[F.k[kb]][0]) ?
			    "an" : "a", ftsing[F.k[kb]], gpi(F.x[kb]));
		} else {
			/* the radius of curvature */
			h = 1e-3 * (1 + fabs(at));
			if (f1(e, base, s, at, &v0) == 1 && f1(e, base, s, at - h, &vl) ==
			    1 && f1(e, base, s, at + h, &vr) == 1 &&
			    fabs(vl - 2 * v0 + vr) > 1e-12) {
				double d1 = (vr - vl) / (2 * h), d2 = (vr - 2 * v0 + vl) /
				    (h * h);
				w = pow(1 + d1 * d1, 1.5) / fabs(d2);
				if (w > 10)
					w = 10;
				if (w < 1e-3)
					w = 1e-3;
				sb_printf(why, "around %s = %s: no feature within 10, the "
				    "radius of curvature is %s", vn, g4(at), g4(w));
			} else {
				w = 1;
				sb_printf(why, "around %s = %s: no feature within 10 and no "
				    "curvature: a unit window", vn, g4(at));
			}
		}
		lo = at - w;
		hi = at + w;
		if (verbose && F.n) {
			printf("  %s near %s (within 10):\n", vn, g4(at));
			print_feats(&F, vn);
		}
	} else {
		static const double win[] = {10, 100};
		int wi;
		for (wi = 0; wi < 2; wi++) {
			find_feats(e, base, s, -win[wi], win[wi], &F);
			if (F.n)
				break;
		}
		if (F.n == 0) {
			lo = -5;
			hi = 5;
			sb_printf(why, "no zeros, extrema, poles or domain edges in "
			    "[-100, 100]: a default window");
		} else {
			double a = INFINITY, b = -INFINITY, m, lim = INFINITY;
			int ka = 0, kb = 0;
			if (F.n > 16) {	/* many: the ones nearest 0 */
				double d[MAXFEAT];
				for (k = 0; k < F.n; k++)
					d[k] = fabs(F.x[k]);
				qsort(d, F.n, sizeof d[0], dcmp);
				lim = d[7] * (1 + 1e-9);
			}
			for (k = 0; k < F.n; k++) {
				if (fabs(F.x[k]) > lim)
					continue;
				if (F.x[k] < a) {
					a = F.x[k];
					ka = F.k[k];
				}
				if (F.x[k] > b) {
					b = F.x[k];
					kb = F.k[k];
				}
			}
			w = b - a;
			m = w > 4 ? w * 0.25 : w > 0 ? w * 0.5 : 2;
			if (m < 1.5)
				m = 1.5;
			lo = a - m;
			hi = b + m;
			/* do not run past a domain edge into nothing */
			if (fabs(a) < 1e-9 * (1 + w))
				a = 0;
			if (fabs(b) < 1e-9 * (1 + w))
				b = 0;
			{
				int cl = ka == FT_EDGE && undef_on(e, base, s, lo, a);
				int cr = kb == FT_EDGE && undef_on(e, base, s, b, hi);
				if (cl && cr) {	/* defined between two edges only */
					lo = a - w * 0.05;
					hi = b + w * 0.05;
				} else if (cl) {
					lo = a;
					if (hi - lo < 2 * m)
						hi = lo + 2 * m + w;
				} else if (cr) {
					hi = b;
					if (hi - lo < 2 * m)
						lo = hi - 2 * m - w;
				}
			}
			{	/* round outwards to a twentieth of the span */
				double st = nice_step(hi - lo, 20);
				if (ka != FT_EDGE || lo != a)
					lo = floor(lo / st + 1e-9) * st;
				if (kb != FT_EDGE || hi != b)
					hi = ceil(hi / st - 1e-9) * st;
				if (fabs(lo) < st * 1e-6)
					lo = 0;
				if (fabs(hi) < st * 1e-6)
					hi = 0;
			}
			if (lim < INFINITY)
				sb_printf(why, "covering the features nearest 0 (%d found in "
				    "[%s, %s])", F.n, g4(-win[wi]), g4(win[wi]));
			else
				sb_printf(why, "covering the %d feature%s found in [%s, %s]",
				    F.n, F.n == 1 ? "" : "s", g4(-win[wi]), g4(win[wi]));
			if (verbose) {
				printf("  %s in [%s, %s]:\n", vn, g4(-win[wi]), g4(win[wi]));
				print_feats(&F, vn);
			}
		}
	}
	if (isint) {
		lo = floor(lo);
		hi = ceil(hi);
		if (hi - lo < 2)
			hi = lo + 2;
		if (pp && TY_SIZED(pp->ty)) {
			int64_t a, b;
			param_bounds(pp, &a, &b);
			if (lo < (double)a)
				lo = (double)a;
			if (hi > (double)b)
				hi = (double)b;
		}
	}
	*plo = lo;
	*phi = hi;
}

/* ---- 2-D axes ---- */

struct ax2 {
	double lo[2], hi[2];
	int lg[2];
	int L, T, R, B;
};

static double
ax_u(const struct ax2 *a, int k, double v)
{
	return a->lg[k] ? (v > 0 ? log10(v) : NAN) : v;
}

static double
PX(const struct ax2 *a, double v)
{
	double u0 = ax_u(a, 0, a->lo[0]), u1 = ax_u(a, 0, a->hi[0]);

	return a->L + (ax_u(a, 0, v) - u0) / (u1 - u0) * (a->R - a->L);
}

static double
PY(const struct ax2 *a, double v)
{
	double u0 = ax_u(a, 1, a->lo[1]), u1 = ax_u(a, 1, a->hi[1]);

	return a->B - (ax_u(a, 1, v) - u0) / (u1 - u0) * (a->B - a->T);
}

/* the data value at pixel column px / row py */
static double
UX(const struct ax2 *a, double px)
{
	double u0 = ax_u(a, 0, a->lo[0]), u1 = ax_u(a, 0, a->hi[0]);
	double u = u0 + (px - a->L) / (a->R - a->L) * (u1 - u0);

	return a->lg[0] ? pow(10, u) : u;
}

static double
UY(const struct ax2 *a, double py)
{
	double u0 = ax_u(a, 1, a->lo[1]), u1 = ax_u(a, 1, a->hi[1]);
	double u = u0 + (a->B - py) / (a->B - a->T) * (u1 - u0);

	return a->lg[1] ? pow(10, u) : u;
}

/* the plot rectangle for this canvas, leaving room for the y labels */
static void
layout2(struct ax2 *a, int x0, int y0, int w, int h, int rightpad, int small)
{
	struct ticks ty;
	int k, maxw = 0;

	mkticks(&ty, a->lo[1], a->hi[1], a->lg[1], small ? 4 : h / 46,
	    tick_int[1]);
	for (k = 0; k < ty.n; k++)
		if (textw(ty.lab[k]) > maxw)
			maxw = textw(ty.lab[k]);
	a->L = x0 + (small ? 4 : 10) + maxw + 5;
	a->R = x0 + w - (small ? 6 : 22) - rightpad;
	a->T = y0 + (small ? 14 : 34);
	a->B = y0 + h - (small ? 16 : 34);
}

static void
axes2(const struct ax2 *a, const char *xn, const char *yn, int small,
    int labels_x, int labels_y)
{
	struct ticks tx, ty;
	int k, x, y;

	mkticks(&tx, a->lo[0], a->hi[0], a->lg[0], small ? 4 :
	    (a->R - a->L) / 72, tick_int[0]);
	mkticks(&ty, a->lo[1], a->hi[1], a->lg[1], small ? 4 :
	    (a->B - a->T) / 46, tick_int[1]);
	for (k = 0; k < tx.n; k++) {
		x = (int)floor(PX(a, tx.v[k]) + 0.5);
		if (x > a->L && x < a->R)
			for (y = a->T; y <= a->B; y++)
				if (soft(get(x, y)))
					put(x, y, P_GRID);
	}
	for (k = 0; k < ty.n; k++) {
		y = (int)floor(PY(a, ty.v[k]) + 0.5);
		if (y > a->T && y < a->B)
			for (x = a->L; x <= a->R; x++)
				if (soft(get(x, y)))
					put(x, y, P_GRID);
	}
	if (!a->lg[0] && a->lo[0] < 0 && a->hi[0] > 0) {
		x = (int)floor(PX(a, 0) + 0.5);
		for (y = a->T; y <= a->B; y++)
			if (soft(get(x, y)))
				put(x, y, P_ZERO);
	}
	if (!a->lg[1] && a->lo[1] < 0 && a->hi[1] > 0) {
		y = (int)floor(PY(a, 0) + 0.5);
		for (x = a->L; x <= a->R; x++)
			if (soft(get(x, y)))
				put(x, y, P_ZERO);
	}
	rect(a->L, a->T, a->R - a->L + 1, a->B - a->T + 1, P_AXIS);
	for (k = 0; k < tx.n; k++) {
		x = (int)floor(PX(a, tx.v[k]) + 0.5);
		if (x < a->L - 1 || x > a->R + 1)
			continue;
		fill(x, a->B + 1, 1, 3, P_AXIS);
		if (small && (x - a->L < 10 || a->R - x < 10))
			continue;	/* the panels' end labels would collide */
		if (labels_x)
			text(x - textw(tx.lab[k]) / 2 + 1, a->B + 5, tx.lab[k], P_DIM);
	}
	for (k = 0; k < ty.n; k++) {
		y = (int)floor(PY(a, ty.v[k]) + 0.5);
		if (y < a->T - 1 || y > a->B + 1)
			continue;
		fill(a->L - 3, y, 3, 1, P_AXIS);
		if (labels_y)
			text(a->L - 5 - textw(ty.lab[k]), y - 5, ty.lab[k], P_DIM);
	}
	if (xn && xn[0] && labels_x)
		text(a->R - textw(xn), a->B + (small ? 5 : 18) + (small ? 0 : 0),
		    xn, P_TEXT);
	if (yn && yn[0] && labels_y && !small)
		text(a->L - 8 - textw(yn), a->T - 18, yn, P_TEXT);
}

static void
title_at(int cx, int y, const char *s, int maxw)
{
	char t[256];

	fit_text(t, sizeof t, s, maxw);
	text(cx - textw(t) / 2, y, t, P_TEXT);
}

/* ---- legend ---- */

struct legend {
	int n;
	char lab[MAXSER + 2][80];
	int col[MAXSER + 2];
	int style[MAXSER + 2];	/* 0 line, 1 dot, 2 swatch */
};

static int
busy(int x, int y, int w, int h)
{
	int i, j, n = 0;

	for (j = y; j < y + h; j++)
		for (i = x; i < x + w; i++)
			if (!soft(get(i, j)))
				n++;
	return n;
}

static void
legend_draw(const struct legend *lg, const struct ax2 *a)
{
	int k, w = 0, h, x, y, best = -1, bx = 0, by = 0, c;
	int cand[4][2];

	if (lg->n == 0)
		return;
	for (k = 0; k < lg->n; k++) {
		char t[96];
		fit_text(t, sizeof t, lg->lab[k], (a->R - a->L) / 2);
		if (textw(t) > w)
			w = textw(t);
	}
	w += 34;
	h = lg->n * 13 + 7;
	cand[0][0] = a->R - 8 - w; cand[0][1] = a->T + 8;
	cand[1][0] = a->L + 8; cand[1][1] = a->T + 8;
	cand[2][0] = a->R - 8 - w; cand[2][1] = a->B - 8 - h;
	cand[3][0] = a->L + 8; cand[3][1] = a->B - 8 - h;
	for (k = 0; k < 4; k++) {
		c = busy(cand[k][0], cand[k][1], w, h);
		if (best < 0 || c < best) {
			best = c;
			bx = cand[k][0];
			by = cand[k][1];
		}
	}
	fill(bx, by, w, h, P_LEGBG);
	rect(bx, by, w, h, P_LEGBORDER);
	for (k = 0; k < lg->n; k++) {
		char t[96];
		y = by + 5 + k * 13;
		x = bx + 6;
		if (lg->style[k] == 1) {
			cov_disc(x + 8, y + 5, 2.6);
		} else if (lg->style[k] == 2) {
			fill(x + 2, y + 1, 14, 9, P_SER + lg->col[k] * 4 + 2);
			rect(x + 2, y + 1, 14, 9, P_SER + lg->col[k] * 4);
		} else
			cov_seg(x, y + 5, x + 16, y + 5, lg->style[k] == 3 ? 1.6 : 2.2);
		if (lg->col[k] >= 0)
			cov_ser(lg->col[k]);
		else
			cov_solid(-lg->col[k]);
		fit_text(t, sizeof t, lg->lab[k], (a->R - a->L) / 2);
		text(x + 22, y, t, P_TEXT);
	}
}

/* ---- robust automatic value range ---- */


/*
 * The range of the values v[0..n-1]: all of them, or (poles, or a few
 * values far out) the bulk between the 10th and 90th percentile widened
 * by 1.5 times its width; then a 5% margin.  Returns 1 when clipped.
 */
static int
vrange(double *v, int n, int haspole, int lg, double *lo, double *hi,
    double *vmin, double *vmax)
{
	double *s, q10, q90, w, mn, mx;
	int i, m = 0, clipped = 0;

	s = xmalloc((n + 1) * sizeof *s);
	for (i = 0; i < n; i++)
		if (isfinite(v[i]) && (!lg || v[i] > 0))
			s[m++] = lg ? log10(v[i]) : v[i];
	if (m == 0) {
		free(s);
		*lo = lg ? 0.1 : -1;
		*hi = lg ? 10 : 1;
		*vmin = *vmax = NAN;
		return 0;
	}
	qsort(s, m, sizeof *s, dcmp);
	mn = s[0];
	mx = s[m - 1];
	q10 = s[(int)(0.1 * (m - 1))];
	q90 = s[(int)(0.9 * (m - 1))];
	w = q90 - q10;
	if (w > 0 && (haspole || mx > q90 + 20 * w || mn < q10 - 20 * w)) {
		double a = q10 - 1.5 * w, b = q90 + 1.5 * w;
		if (a > mn) {
			mn = a;
			clipped = 1;
		}
		if (b < mx) {
			mx = b;
			clipped = 1;
		}
	}
	*vmin = lg ? pow(10, s[0]) : s[0];
	*vmax = lg ? pow(10, s[m - 1]) : s[m - 1];
	if (mx - mn <= 1e-12 * (fabs(mx) + 1e-300)) {
		double d = fabs(mx) > 0 ? fabs(mx) * 0.1 : 1;
		mn -= d;
		mx += d;
	} else {
		double d = (mx - mn) * 0.05;
		mn -= d;
		mx += d;
	}
	*lo = lg ? pow(10, mn) : mn;
	*hi = lg ? pow(10, mx) : mx;
	free(s);
	return clipped;
}

/* ---- 2-D charts ---- */

static struct ev EV[MAXSER][3];
static struct sbuf SUM;		/* the text summary, printed once */
static int first_frame;

#define NSAMP 1201

static int
is_intvar(const struct ser *s)
{
	return s->sx >= 0 && !TY_FP(s->f[0].p[s->sx].ty);
}

/* undefined runs of a sampled curve, as text */
static void
undef_runs(struct ev *e, const double *base, int sx, const double *xs,
    const uint8_t *ok, int n, const char *vn)
{
	int i = 0, runs = 0;

	while (i < n) {
		int a, b;
		double l, r;
		if (ok[i] != 0) {
			i++;
			continue;
		}
		for (a = i; i < n && ok[i] == 0; i++)
			;
		b = i - 1;
		if (runs < 4) {
			l = a == 0 ? xs[0] : bisect1(e, base, sx, xs[a - 1], xs[a], 1);
			r = b == n - 1 ? xs[n - 1] : bisect1(e, base, sx, xs[b], xs[b + 1],
			    1);
			if (a == b && a > 0 && b < n - 1)
				sb_printf(&SUM, "%s %s ≈ %s", runs ? "," : "    undefined at",
				    vn, g4(l));
			else
				sb_printf(&SUM, "%s %s in [%s, %s]", runs ? "," :
				    "    undefined for", vn, g4(l), g4(r));
		}
		runs++;
	}
	if (runs > 4)
		sb_printf(&SUM, ", ... (%d stretches)", runs);
	if (runs)
		sb_printf(&SUM, "\n");
}

static void
list_x(const char *what, const double *v, int n, const char *vn)
{
	int k;

	if (n == 0)
		return;
	sb_printf(&SUM, "    %s at %s ≈ ", what, vn);
	for (k = 0; k < n && k < 8; k++)
		sb_printf(&SUM, "%s%s", k ? ", " : "", gpi(v[k]));
	if (n > 8)
		sb_printf(&SUM, ", ... (%d in all)", n);
	sb_printf(&SUM, "\n");
}

/* y = f(x): sampled, broken at undefined points, jumps and poles */
static void
draw_fx(struct ser *s, int si, const struct ax2 *A)
{
	struct ev *e = &EV[si][0];
	int isint = is_intvar(s), n, i, m = 0, pts = 0, step = 0;
	double *xs, *ys, *px, *py, ysp = A->hi[1] - A->lo[1];
	double poles[64], jumps[64];
	int npole = 0, njump = 0;
	uint8_t *ok;
	const char *vn = s->sx >= 0 ? sym_str(s->f[0].p[s->sx].name) : "x";

	if (isint) {
		double a = ceil(A->lo[0]), b = floor(A->hi[0]);
		double cnt = b - a + 1;
		if (cnt <= 0)
			return;
		n = cnt <= 4000 ? (int)cnt : 4000;
		pts = cnt <= 160;
		step = !pts;
		xs = xmalloc(n * sizeof *xs);
		for (i = 0; i < n; i++)
			xs[i] = n == 1 ? a : floor(a + (b - a) * i / (n - 1) + 0.5);
	} else {
		double u0 = ax_u(A, 0, A->lo[0]), u1 = ax_u(A, 0, A->hi[0]);
		n = O.npts > 1 ? O.npts : NSAMP;
		if (n > 100000)
			n = 100000;
		xs = xmalloc(n * sizeof *xs);
		for (i = 0; i < n; i++) {
			double u = u0 + (u1 - u0) * i / (n - 1);
			xs[i] = A->lg[0] ? pow(10, u) : u;
		}
	}
	ys = xmalloc(n * sizeof *ys);
	ok = xmalloc(n);
	px = xmalloc((3 * n + 2) * sizeof *px);
	py = xmalloc((3 * n + 2) * sizeof *py);
	f1n(e, s->fix[0], s->sx, n, xs, ys, ok);
	if (A->lg[1])
		for (i = 0; i < n; i++)
			if (ok[i] == 1 && ys[i] <= 0)
				ok[i] = 3;	/* not on a log axis */
	/* an undefined sample between huge values of opposite sign: a pole */
	for (i = 1; !isint && i + 1 < n; i++)
		if (ok[i] == 0 && ok[i - 1] == 1 && ok[i + 1] == 1) {
			double at;
			if (jump_kind(e, s->fix[0], s->sx, xs[i - 1], ys[i - 1], xs[i + 1],
			    ys[i + 1], &at) == 2) {
				int x = (int)floor(PX(A, at) + 0.5), y;
				ok[i] = 5;
				if (npole < 64)
					poles[npole++] = at;
				for (y = A->T + 1; y < A->B; y++)
					if ((y / 3) % 2 == 0 && soft(get(x, y)))
						put(x, y, P_ASYM);
			}
		}
	for (i = 0; i < n; i++) {
		int brk = 0;
		if (ok[i] != 1) {
			px[m] = py[m] = NAN;
			m++;
			continue;
		}
		if (pts) {
			cov_disc(PX(A, xs[i]), PY(A, ys[i]), 2.7);
			continue;
		}
		if (!isint && i > 0 && ok[i - 1] == 1 &&
		    fabs(ys[i] - ys[i - 1]) > ysp * 0.1) {
			double at;
			int jk = jump_kind(e, s->fix[0], s->sx, xs[i - 1], ys[i - 1],
			    xs[i], ys[i], &at);
			if (jk) {
				brk = 1;
				if (jk == 2 && npole < 64)
					poles[npole++] = at;
				else if (jk == 1 && njump < 64)
					jumps[njump++] = at;
				if (jk == 2) {	/* the asymptote, dashed */
					int x = (int)floor(PX(A, at) + 0.5), y;
					for (y = A->T + 1; y < A->B; y++)
						if ((y / 3) % 2 == 0 && soft(get(x, y)))
							put(x, y, P_ASYM);
				}
			}
		}
		if (brk) {
			px[m] = py[m] = NAN;
			m++;
		}
		if (step && i > 0 && ok[i - 1] == 1) {	/* riser, then tread */
			px[m] = PX(A, xs[i]);
			py[m] = PY(A, ys[i - 1]);
			m++;
		}
		px[m] = PX(A, xs[i]);
		py[m] = PY(A, ys[i]);
		m++;
		if (step && i == n - 1 && xs[i] + 1 <= A->hi[0] + 1e-9) {
			px[m] = PX(A, xs[i] + 1);
			py[m] = PY(A, ys[i]);
			m++;
		}
	}
	if (!pts)
		poly(px, py, m, s->kind == S_DERIV ? 1.7 : 2.3);
	cov_ser(s->col);
	if (first_frame) {
		double lo = INFINITY, hi = -INFINITY;
		int nd = 0;
		for (i = 0; i < n; i++)
			if (ok[i] == 1) {
				nd++;
				if (ys[i] < lo)
					lo = ys[i];
				if (ys[i] > hi)
					hi = ys[i];
			}
		sb_printf(&SUM, "  %s: %d %s", s->label, n, pts ? "points" : step ?
		    "integer inputs (steps)" : "samples");
		if (nd)
			sb_printf(&SUM, ", values in [%s, %s]", g4(lo), g4(hi));
		sb_printf(&SUM, "\n");
		undef_runs(e, s->fix[0], s->sx, xs, ok, n, vn);
		list_x("poles", poles, npole, vn);
		list_x("jumps", jumps, njump, vn);
		if (A->lg[1]) {
			int c = 0;
			for (i = 0; i < n; i++)
				c += ok[i] == 3;
			if (c)
				sb_printf(&SUM, "    %d samples <= 0 left out (log y axis)\n",
				    c);
		}
	}
	free(xs);
	free(ys);
	free(ok);
	free(px);
	free(py);
}

/* the input vector for the plane point (x, y) */
static void
plane_in(const struct ser *s, int c, double x, double y, double *in)
{
	int np = s->f[c].np, a = s->sx, b = s->sy;

	memcpy(in, s->fix[c], (size_t)np * sizeof *in);
	if (s->kind == S_VFIELD) {	/* each component has its own slots */
		a = slot_named(&s->f[c], sym_str(s->xname));
		b = slot_named(&s->f[c], sym_str(s->yname));
	}
	if (a >= 0)
		in[a] = x;
	if (b >= 0)
		in[b] = y;
}

/* marching squares over a grid of g pixels: the curve e = 0 */
static void
contour_grid(struct ev *e, const struct ser *s, const struct ax2 *A, int g,
    double level, double w, double *bb)
{
	int nx = (A->R - A->L) / g + 2, ny = (A->B - A->T) / g + 2, i, j, np;
	int n = nx * ny;
	double *in, *v;
	uint8_t *ok;

	np = e->f.np;
	in = xmalloc((size_t)n * (np ? np : 1) * sizeof *in);
	v = xmalloc(n * sizeof *v);
	ok = xmalloc(n);
	for (j = 0; j < ny; j++)
		for (i = 0; i < nx; i++)
			plane_in(s, 0, UX(A, A->L + i * g), UY(A, A->T + j * g),
			    &in[(size_t)(j * nx + i) * np]);
	ev_run(e, n, np ? in : NULL, v, ok);
	for (j = 0; j + 1 < ny; j++)
		for (i = 0; i + 1 < nx; i++) {
			int k00 = j * nx + i, k10 = k00 + 1, k01 = k00 + nx, k11 = k01 + 1;
			double a = v[k00] - level, b = v[k10] - level, c = v[k11] - level,
			    d = v[k01] - level, ex[4], ey[4], x0 = A->L + i * g,
			    y0 = A->T + j * g, cv, mx;
			int idx, ne = 0;
			if (ok[k00] != 1 || ok[k10] != 1 || ok[k01] != 1 || ok[k11] != 1)
				continue;
			idx = (a > 0) | (b > 0) << 1 | (c > 0) << 2 | (d > 0) << 3;
			if (idx == 0 || idx == 15)
				continue;
			/* crossings on the edges top (a-b), right (b-c), bottom (d-c),
			 * left (a-d) */
			if ((a > 0) != (b > 0)) {
				ex[ne] = x0 + g * a / (a - b);
				ey[ne++] = y0;
			}
			if ((b > 0) != (c > 0)) {
				ex[ne] = x0 + g;
				ey[ne++] = y0 + g * b / (b - c);
			}
			if ((d > 0) != (c > 0)) {
				ex[ne] = x0 + g * d / (d - c);
				ey[ne++] = y0 + g;
			}
			if ((a > 0) != (d > 0)) {
				ex[ne] = x0;
				ey[ne++] = y0 + g * a / (a - d);
			}
			/* a pole between the corners is not a crossing */
			{
				double cin[MAXP], cvv;
				plane_in(s, 0, UX(A, x0 + g * 0.5), UY(A, y0 + g * 0.5), cin);
				if (ev1(e, cin, &cvv) != 1)
					continue;
				cv = cvv - level;
				mx = fabs(a);
				if (fabs(b) > mx)
					mx = fabs(b);
				if (fabs(c) > mx)
					mx = fabs(c);
				if (fabs(d) > mx)
					mx = fabs(d);
				if (fabs(cv) > 3 * mx)
					continue;
			}
			if (ne == 2)
				cov_seg(ex[0], ey[0], ex[1], ey[1], w);
			else if (ne == 4) {	/* a saddle: the centre decides */
				if ((cv > 0) == (a > 0)) {
					cov_seg(ex[0], ey[0], ex[1], ey[1], w);
					cov_seg(ex[2], ey[2], ex[3], ey[3], w);
				} else {
					cov_seg(ex[0], ey[0], ex[3], ey[3], w);
					cov_seg(ex[1], ey[1], ex[2], ey[2], w);
				}
			}
			if (bb && ne) {
				int q;
				for (q = 0; q < ne; q++) {
					double X = UX(A, ex[q]), Y = UY(A, ey[q]);
					if (X < bb[0]) bb[0] = X;
					if (X > bb[1]) bb[1] = X;
					if (Y < bb[2]) bb[2] = Y;
					if (Y > bb[3]) bb[3] = Y;
				}
			}
		}
	free(in);
	free(v);
	free(ok);
}

static void
draw_impl(struct ser *s, int si, const struct ax2 *A)
{
	double bb[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY};

	contour_grid(&EV[si][0], s, A, 2, 0, 2.3, bb);
	cov_ser(s->col);
	if (first_frame) {
		if (bb[0] <= bb[1])
			sb_printf(&SUM, "  %s: the curve spans x in [%s, %s], y in "
			    "[%s, %s]\n", s->label, g4(bb[0]), g4(bb[1]), g4(bb[2]),
			    g4(bb[3]));
		else
			sb_printf(&SUM, "  %s: no points in view\n", s->label);
	}
}

/* a region: where the condition holds, shaded; its boundary drawn */
static void
draw_region(struct ser *s, int si, const struct ax2 *A)
{
	struct ev *e = &EV[si][0];
	int w = A->R - A->L - 1, h = A->B - A->T - 1, i, j, np = e->f.np, cnt = 0;
	int n = w * h;
	double *in = xmalloc((size_t)n * (np ? np : 1) * sizeof *in), *v;
	uint8_t *ok;

	v = xmalloc(n * sizeof *v);
	ok = xmalloc(n);
	for (j = 0; j < h; j++)
		for (i = 0; i < w; i++)
			plane_in(s, 0, UX(A, A->L + 1 + i), UY(A, A->T + 1 + j),
			    &in[(size_t)(j * w + i) * np]);
	ev_run(e, n, np ? in : NULL, v, ok);
	for (j = 0; j < h; j++)
		for (i = 0; i < w; i++)
			if (ok[j * w + i] == 1 && v[j * w + i] != 0) {
				cnt++;
				if (soft(get(A->L + 1 + i, A->T + 1 + j)))
					put(A->L + 1 + i, A->T + 1 + j, P_SER + s->col * 4 + 3);
			}
	free(in);
	free(v);
	free(ok);
	if (s->bound) {
		struct ev be;
		ev_open(&be, &s->f[0], s->bound);
		contour_grid(&be, s, A, 2, 0, 1.8, NULL);
		ev_close(&be);
		cov_ser(s->col);
	}
	if (first_frame)
		sb_printf(&SUM, "  %s: holds on %.1f%% of the view\n", s->label,
		    100.0 * cnt / (n ? n : 1));
}

/* parametric and polar curves: their points were sampled in setup */
static void
draw_curve(struct ser *s, const struct ax2 *A)
{
	double *px = xmalloc((s->cn + 1) * sizeof *px);
	double *py = xmalloc((s->cn + 1) * sizeof *py);
	double diag = hypot(A->R - A->L, A->B - A->T);
	int i, m = 0, und = 0;

	for (i = 0; i < s->cn; i++) {
		if (isnan(s->cx[i])) {
			und++;
			px[m] = py[m] = NAN;
			m++;
			continue;
		}
		px[m] = PX(A, s->cx[i]);
		py[m] = PY(A, s->cy[i]);
		if (m > 0 && !isnan(px[m - 1]) && hypot(px[m] - px[m - 1],
		    py[m] - py[m - 1]) > diag * 0.25) {
			px[m + 1] = px[m];
			py[m + 1] = py[m];
			px[m] = py[m] = NAN;
			m++;
		}
		m++;
	}
	poly(px, py, m, 2.3);
	cov_ser(s->col);
	if (first_frame) {
		const char *tn = s->sx >= 0 ? sym_str(s->f[0].p[s->sx].name) :
		    s->kind == S_POLAR ? "theta" : "t";
		sb_printf(&SUM, "  %s: %s in [%s, %s], %d samples", s->label, tn,
		    gpi(s->t0), gpi(s->t1), s->cn);
		if (und)
			sb_printf(&SUM, ", undefined at %d", und);
		sb_printf(&SUM, "\n");
	}
	free(px);
	free(py);
}

/* an arrow from (x0, y0) by (dx, dy) pixels, into the coverage */
static void
arrow(double x0, double y0, double dx, double dy, double w)
{
	double l = hypot(dx, dy), ux, uy, hl;

	if (l < 0.5)
		return;
	ux = dx / l;
	uy = dy / l;
	hl = l * 0.32 < 5 ? l * 0.32 : 5;
	cov_seg(x0, y0, x0 + dx, y0 + dy, w);
	cov_seg(x0 + dx, y0 + dy, x0 + dx - hl * (ux * 0.87 - uy * 0.5),
	    y0 + dy - hl * (uy * 0.87 + ux * 0.5), w);
	cov_seg(x0 + dx, y0 + dy, x0 + dx - hl * (ux * 0.87 + uy * 0.5),
	    y0 + dy - hl * (uy * 0.87 - ux * 0.5), w);
}

/* a vector field (P, Q): arrows coloured by length */
static void
draw_vfield(struct ser *s, int si, const struct ax2 *A)
{
	enum { NX = 24 };
	int ny = (int)(NX * (double)(A->B - A->T) / (A->R - A->L) + 0.5), i, j, l;
	double cw = (double)(A->R - A->L) / NX, ch, mx = 0;
	double P[NX * 32], Q[NX * 32];
	uint8_t okv[NX * 32];
	double minv = INFINITY;

	if (ny < 4)
		ny = 4;
	if (ny > 32)
		ny = 32;
	ch = (double)(A->B - A->T) / ny;
	for (j = 0; j < ny; j++)
		for (i = 0; i < NX; i++) {
			double in[MAXP], x = UX(A, A->L + (i + 0.5) * cw),
			    y = UY(A, A->T + (j + 0.5) * ch);
			int k = j * NX + i, o1, o2;
			plane_in(s, 0, x, y, in);
			o1 = ev1(&EV[si][0], in, &P[k]);
			plane_in(s, 1, x, y, in);
			o2 = ev1(&EV[si][1], in, &Q[k]);
			okv[k] = o1 == 1 && o2 == 1;
			if (okv[k]) {
				double m = hypot(P[k], Q[k]);
				if (m > mx)
					mx = m;
				if (m < minv)
					minv = m;
			}
		}
	for (l = 0; l < 8; l++) {	/* one coverage pass per colour */
		int t[4], any = 0;
		for (j = 0; j < ny; j++)
			for (i = 0; i < NX; i++) {
				int k = j * NX + i, lv;
				double m, sx, sy, len, dpx, dpy, dl;
				if (!okv[k] || mx <= 0)
					continue;
				m = hypot(P[k], Q[k]);
				lv = (int)(m / mx * 7.999);
				if (lv != l)
					continue;
				/* the direction in pixels (the axes' scales differ) */
				sx = (A->R - A->L) / (A->hi[0] - A->lo[0]);
				sy = (A->B - A->T) / (A->hi[1] - A->lo[1]);
				dpx = P[k] * sx;
				dpy = -Q[k] * sy;
				dl = hypot(dpx, dpy);
				if (dl == 0) {
					cov_disc(A->L + (i + 0.5) * cw, A->T + (j + 0.5) * ch, 1.2);
					any = 1;
					continue;
				}
				len = (cw < ch ? cw : ch) * (0.35 + 0.55 * sqrt(m / mx));
				dpx *= len / dl;
				dpy *= len / dl;
				arrow(A->L + (i + 0.5) * cw - dpx / 2, A->T + (j + 0.5) * ch -
				    dpy / 2, dpx, dpy, 1.3);
				any = 1;
			}
		if (!any)
			continue;
		t[0] = t[1] = cmap(0.08 + 0.8 * l / 7.0, 3);
		t[2] = t[3] = cmap(0.08 + 0.8 * l / 7.0, 5);
		cov_put(t);
	}
	if (first_frame)
		sb_printf(&SUM, "  %s: %dx%d arrows, length in [%s, %s] (longest "
		    "brightest)\n", s->label, NX, ny, g4(minv), g4(mx));
}

/* one RK4 solution of y' = f(t, y) from (t0, y0) towards t1 */
static int
rk4(struct ev *e, const struct ser *s, double t0, double y0, double t1,
    int n, double *ts, double *yv, double ylim0, double ylim1)
{
	double h = (t1 - t0) / n, t = t0, y = y0, in[MAXP];
	int i, m = 0;

#define F(tt, yy, out) (plane_in(s, 0, (tt), (yy), in), ev1(e, in, (out)) == 1)
	ts[m] = t;
	yv[m++] = y;
	for (i = 0; i < n; i++) {
		double k1, k2, k3, k4;
		if (!F(t, y, &k1) || !F(t + h / 2, y + h / 2 * k1, &k2) ||
		    !F(t + h / 2, y + h / 2 * k2, &k3) || !F(t + h, y + h * k3, &k4))
			break;
		y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
		t = t0 + h * (i + 1);
		if (!isfinite(y))
			break;
		ts[m] = t;
		yv[m++] = y;
		if (y < ylim0 || y > ylim1)
			break;
	}
#undef F
	return m;
}

/* a slope field of y' = f(t, y), with solution curves */
static void
draw_slope(struct ser *s, int si, const struct ax2 *A)
{
	enum { NX = 26, NR = 400 };
	struct ev *e = &EV[si][0];
	int ny = (int)(NX * (double)(A->B - A->T) / (A->R - A->L) + 0.5), i, j, k;
	double cw = (double)(A->R - A->L) / NX, ch, sx, sy, ysp;
	double ts[2 * NR + 2], yv[2 * NR + 2], px[2 * NR + 2], py[2 * NR + 2];

	if (ny < 4)
		ny = 4;
	ch = (double)(A->B - A->T) / ny;
	sx = (A->R - A->L) / (A->hi[0] - A->lo[0]);
	sy = (A->B - A->T) / (A->hi[1] - A->lo[1]);
	ysp = A->hi[1] - A->lo[1];
	for (j = 0; j < ny; j++)
		for (i = 0; i < NX; i++) {
			double in[MAXP], v, cxp = A->L + (i + 0.5) * cw,
			    cyp = A->T + (j + 0.5) * ch, dx, dy, l, len;
			plane_in(s, 0, UX(A, cxp), UY(A, cyp), in);
			if (ev1(e, in, &v) != 1)
				continue;
			dx = sx;
			dy = -v * sy;
			l = hypot(dx, dy);
			len = (cw < ch ? cw : ch) * 0.36;
			cov_seg(cxp - dx / l * len, cyp - dy / l * len, cxp + dx / l * len,
			    cyp + dy / l * len, 1.3);
		}
	cov_solid(P_AXIS);
	/* a family of solutions from the left edge */
	for (k = 0; k < 9; k++) {
		double y0 = A->lo[1] + ysp * (k + 0.5) / 9;
		int m = rk4(e, s, A->lo[0], y0, A->hi[0], NR, ts, yv,
		    A->lo[1] - ysp, A->hi[1] + ysp), q;
		for (q = 0; q < m; q++) {
			px[q] = PX(A, ts[q]);
			py[q] = PY(A, yv[q]);
		}
		poly(px, py, m, 1.4);
	}
	{	/* the family lighter than the solution through the initial value */
		int t[4] = {P_SER + 1, P_SER + 2, P_SER + 3, P_SER + 3};
		cov_put(t);
	}
	if (s->has_ic) {
		int m1, m2, q, m = 0;
		double ta[NR + 1], ya[NR + 1];
		m1 = rk4(e, s, s->ic_t, s->ic_y, A->hi[0], NR, ts, yv,
		    A->lo[1] - ysp, A->hi[1] + ysp);
		m2 = s->ic_t > A->lo[0] ? rk4(e, s, s->ic_t, s->ic_y, A->lo[0], NR, ta,
		    ya, A->lo[1] - ysp, A->hi[1] + ysp) : 0;
		for (q = m2 - 1; q > 0; q--) {
			px[m] = PX(A, ta[q]);
			py[m++] = PY(A, ya[q]);
		}
		for (q = 0; q < m1; q++) {
			px[m] = PX(A, ts[q]);
			py[m++] = PY(A, yv[q]);
		}
		poly(px, py, m, 2.6);
		cov_ser(1);
		cov_disc(PX(A, s->ic_t), PY(A, s->ic_y), 3.4);
		cov_solid(P_MARK);
		if (first_frame && m1 > 0)
			sb_printf(&SUM, "  solution through (%s, %s): at %s = %s it is "
			    "%s\n", g4(s->ic_t), g4(s->ic_y), sym_str(s->xname),
			    g4(ts[m1 - 1]), g4(yv[m1 - 1]));
	}
	if (first_frame)
		sb_printf(&SUM, "  %s: slopes on a %dx%d grid, 9 solutions (RK4, %d "
		    "steps)\n", s->label, NX, ny, NR);
}

static void
draw_tangent(struct ser *s, const struct ax2 *A)
{
	double x0 = A->lo[0], x1 = A->hi[0];

	cov_seg(PX(A, x0), PY(A, s->tm * x0 + s->tc), PX(A, x1),
	    PY(A, s->tm * x1 + s->tc), 1.6);
	cov_ser(s->col);
	cov_disc(PX(A, s->ta), PY(A, s->tm * s->ta + s->tc), 3.6);
	cov_solid(P_MARK);
	if (first_frame)
		sb_printf(&SUM, "  %s: slope %s, through (%s, %s)\n", s->label,
		    g4(s->tm), g4(s->ta), g4(s->tm * s->ta + s->tc));
}

/* the whole chart, for the series' current fixed values */
static void
chart2d(struct ax2 *A, int x0, int y0, int w, int h, const char *title,
    const char *xn, const char *yn)
{
	struct legend lg;
	int k;

	layout2(A, x0, y0, w, h, 0, 0);
	cv_clip(A->L + 1, A->T + 1, A->R - 1, A->B - 1);
	axes2(A, xn, yn, 0, 1, 1);
	for (k = 0; k < O.ns; k++)
		if (O.s[k].kind == S_REGION)
			draw_region(&O.s[k], k, A);
	axes2(A, xn, yn, 0, 1, 1);	/* the grid over the shading */
	for (k = 0; k < O.ns; k++)
		switch (O.s[k].kind) {
		case S_VFIELD:
			draw_vfield(&O.s[k], k, A);
			break;
		case S_SLOPE:
			draw_slope(&O.s[k], k, A);
			break;
		default:
			break;
		}
	for (k = 0; k < O.ns; k++)
		switch (O.s[k].kind) {
		case S_FX:
		case S_DERIV:
			draw_fx(&O.s[k], k, A);
			break;
		case S_IMPL:
			draw_impl(&O.s[k], k, A);
			break;
		case S_PARAM:
		case S_POLAR:
			draw_curve(&O.s[k], A);
			break;
		case S_TANGENT:
			draw_tangent(&O.s[k], A);
			break;
		default:
			break;
		}
	memset(&lg, 0, sizeof lg);
	for (k = 0; k < O.ns; k++) {
		struct ser *s = &O.s[k];
		if (s->kind == S_VFIELD || s->kind == S_SLOPE)
			continue;
		snprintf(lg.lab[lg.n], sizeof lg.lab[0], "%s", s->label);
		lg.col[lg.n] = s->col;
		lg.style[lg.n] = s->kind == S_REGION ? 2 : is_intvar(s) &&
		    s->kind == S_FX ? 1 : s->kind == S_TANGENT || s->kind == S_DERIV ?
		    3 : 0;
		lg.n++;
	}
	if (lg.n > 1) {
		cv_clip(A->L + 1, A->T + 1, A->R - 1, A->B - 1);
		legend_draw(&lg, A);
	}
	cv_clip(0, 0, C.W - 1, C.H - 1);
	title_at(x0 + w / 2, y0 + 8, title, w - 20);
}

/* ---- colour bar ---- */

static void
colorbar(int x, int y0, int y1, double zlo, double zhi, int sh,
    const char *name)
{
	struct ticks t;
	int y, k;

	for (y = y0; y <= y1; y++)
		fill(x, y, 12, 1, cmap((double)(y1 - y) / (y1 - y0 + 1), sh));
	rect(x - 1, y0 - 1, 14, y1 - y0 + 3, P_AXIS);
	mkticks(&t, zlo, zhi, 0, (y1 - y0) / 40, tick_int[2]);
	for (k = 0; k < t.n; k++) {
		y = (int)floor(y1 - (t.v[k] - zlo) / (zhi - zlo) * (y1 - y0) + 0.5);
		fill(x + 12, y, 3, 1, P_AXIS);
		text(x + 17, y - 5, t.lab[k], P_DIM);
	}
	if (name)
		text(x + 6 - textw(name) / 2, y0 - 15, name, P_TEXT);
}

/* ---- heat maps and contours of z = f(x, y) ---- */

static void
heat_into(struct ser *s, int si, const struct ax2 *A, double zlo, double zhi,
    int view, int blk, int *nund)
{
	struct ev *e = &EV[si][0];
	int w = (A->R - A->L - 1 + blk - 1) / blk, h = (A->B - A->T - 1 + blk - 1) /
	    blk, i, j, np = e->f.np, n = w * h, a, b, und = 0;
	double *in = xmalloc((size_t)n * (np ? np : 1) * sizeof *in), *v;
	uint8_t *ok;

	v = xmalloc(n * sizeof *v);
	ok = xmalloc(n);
	for (j = 0; j < h; j++)
		for (i = 0; i < w; i++)
			plane_in(s, 0, UX(A, A->L + 1 + i * blk + (blk - 1) * 0.5),
			    UY(A, A->T + 1 + j * blk + (blk - 1) * 0.5),
			    &in[(size_t)(j * w + i) * np]);
	ev_run(e, n, np ? in : NULL, v, ok);
	for (j = 0; j < h; j++)
		for (i = 0; i < w; i++) {
			int k = j * w + i, c;
			if (ok[k] != 1) {
				c = P_UNDEF;
				und++;
			} else
				c = cmap((v[k] - zlo) / (zhi - zlo), view == V_CONTOUR ? 5 : 4);
			for (b = 0; b < blk; b++)
				for (a = 0; a < blk; a++) {
					int x = A->L + 1 + i * blk + a, y = A->T + 1 + j * blk + b;
					if (x < A->R && y < A->B)
						put(x, y, c);
				}
		}
	if (nund)
		*nund = und;
	free(in);
	free(v);
	free(ok);
	if (view == V_CONTOUR) {
		struct ticks t;
		int k;
		mkticks(&t, zlo, zhi, 0, 10, 0);   /* levels between integers */
		for (k = 0; k < t.n; k++) {
			contour_grid(e, s, A, blk > 1 ? 4 : 2, t.v[k], 1.2, NULL);
			cov_solid(P_WIRE);
		}
	}
}

/* ---- 3-D: projection, panes, surfaces ---- */

struct v3 {
	double az, el, cx, cy, S, h;
};

static void
proj(const struct v3 *V, double X, double Y, double Z, double *sx, double *sy,
    double *dep)
{
	double ca = cos(V->az), sa = sin(V->az), ce = cos(V->el), se = sin(V->el);
	double xr = X * ca - Y * sa, yr = X * sa + Y * ca;
	double u = xr, v = Z * ce + yr * se, d = yr * ce - Z * se;
	double k = 1.0 / (1 + 0.09 * d);

	*sx = V->cx + V->S * u * k;
	*sy = V->cy - V->S * v * k;
	if (dep)
		*dep = d;
}

/* scale and centre so that the box fits x0..x1, y0..y1 */
static void
v3_fit(struct v3 *V, int x0, int y0, int x1, int y1)
{
	double a = INFINITY, b = -INFINITY, c = INFINITY, d = -INFINITY, sx, sy;
	double az;
	int i, k;

	V->S = 1;
	V->cx = V->cy = 0;
	/* the same scale at every angle: a rotation does not pump */
	for (k = 0; k < 16; k++) {
		az = V->az;
		V->az = k * M_PI / 8;
		for (i = 0; i < 8; i++) {
			proj(V, i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? V->h : -V->h, &sx,
			    &sy, NULL);
			if (sx < a) a = sx;
			if (sx > b) b = sx;
			if (sy < c) c = sy;
			if (sy > d) d = sy;
		}
		V->az = az;
	}
	V->S = fmin((x1 - x0) / (b - a), (y1 - y0) / (d - c));
	V->cx = (x0 + x1) / 2.0 - V->S * (a + b) / 2;
	V->cy = (y0 + y1) / 2.0 - V->S * (c + d) / 2;
}

/* a filled triangle; with zt, depth-tested against the z-buffer */
static void
tri(const double *x, const double *y, const double *z, int col, int zt)
{
	int i, j, ia, ib, ja, jb;
	double area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);

	if (fabs(area) < 1e-9)
		return;
	ia = (int)floor(fmin(x[0], fmin(x[1], x[2])));
	ib = (int)ceil(fmax(x[0], fmax(x[1], x[2])));
	ja = (int)floor(fmin(y[0], fmin(y[1], y[2])));
	jb = (int)ceil(fmax(y[0], fmax(y[1], y[2])));
	if (ia < 0) ia = 0;
	if (ja < 0) ja = 0;
	if (ib >= C.W) ib = C.W - 1;
	if (jb >= C.H) jb = C.H - 1;
	for (j = ja; j <= jb; j++)
		for (i = ia; i <= ib; i++) {
			double px = i + 0.5, py = j + 0.5, w0, w1, w2, d;
			w0 = ((x[1] - px) * (y[2] - py) - (x[2] - px) * (y[1] - py)) / area;
			w1 = ((x[2] - px) * (y[0] - py) - (x[0] - px) * (y[2] - py)) / area;
			w2 = 1 - w0 - w1;
			if (w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9)
				continue;
			if (zt) {
				float *zp = &C.zb[(size_t)j * C.W + i];
				d = w0 * z[0] + w1 * z[1] + w2 * z[2];
				if (d >= *zp)
					continue;
				*zp = (float)d;
			}
			if (col >= 0)
				C.px[(size_t)j * C.W + i] = (unsigned char)col;
		}
}

static void
quad_fill(const struct v3 *V, const double (*P)[3], int col)
{
	double x[4], y[4], z[4], tx[3], ty[3], tz[3];
	int k;

	for (k = 0; k < 4; k++)
		proj(V, P[k][0], P[k][1], P[k][2], &x[k], &y[k], &z[k]);
	tx[0] = x[0]; ty[0] = y[0]; tz[0] = z[0];
	tx[1] = x[1]; ty[1] = y[1]; tz[1] = z[1];
	tx[2] = x[2]; ty[2] = y[2]; tz[2] = z[2];
	tri(tx, ty, tz, col, 0);
	tx[1] = x[2]; ty[1] = y[2]; tz[1] = z[2];
	tx[2] = x[3]; ty[2] = y[3]; tz[2] = z[3];
	tri(tx, ty, tz, col, 0);
}

static void
line3(const struct v3 *V, double X0, double Y0, double Z0, double X1,
    double Y1, double Z1, int col)
{
	double a, b, c, d;

	proj(V, X0, Y0, Z0, &a, &b, NULL);
	proj(V, X1, Y1, Z1, &c, &d, NULL);
	line1((int)floor(a + 0.5), (int)floor(b + 0.5), (int)floor(c + 0.5),
	    (int)floor(d + 0.5), col);
}

/* the box's ranges for labels */
struct box3 {
	double lo[3], hi[3];
	const char *name[3];
};

static double
nrm(const struct box3 *B, int k, double v, double h)
{
	double r = (v - B->lo[k]) / (B->hi[k] - B->lo[k]) * 2 - 1;

	return k == 2 ? r * h : r;
}

/* back panes with grid lines, then the labels on the front edges */
static void
panes(const struct v3 *V, const struct box3 *B, int labels)
{
	double xb, yb, d1, d2, sx, sy, h = V->h;
	double P[4][3];
	struct ticks t[3];
	int k, i;

	proj(V, 1, 0, 0, &sx, &sy, &d1);
	proj(V, -1, 0, 0, &sx, &sy, &d2);
	xb = d1 > d2 ? 1 : -1;
	proj(V, 0, 1, 0, &sx, &sy, &d1);
	proj(V, 0, -1, 0, &sx, &sy, &d2);
	yb = d1 > d2 ? 1 : -1;
	for (k = 0; k < 3; k++)
		mkticks(&t[k], B->lo[k], B->hi[k], 0, 5, tick_int[k]);
	/* floor */
	P[0][0] = -1; P[0][1] = -1; P[0][2] = -h;
	P[1][0] = 1; P[1][1] = -1; P[1][2] = -h;
	P[2][0] = 1; P[2][1] = 1; P[2][2] = -h;
	P[3][0] = -1; P[3][1] = 1; P[3][2] = -h;
	quad_fill(V, (const double (*)[3])P, P_PANE);
	/* the wall at x = xb */
	P[0][0] = xb; P[0][1] = -1; P[0][2] = -h;
	P[1][0] = xb; P[1][1] = 1; P[1][2] = -h;
	P[2][0] = xb; P[2][1] = 1; P[2][2] = h;
	P[3][0] = xb; P[3][1] = -1; P[3][2] = h;
	quad_fill(V, (const double (*)[3])P, P_PANE);
	P[0][0] = -1; P[0][1] = yb; P[0][2] = -h;
	P[1][0] = 1; P[1][1] = yb; P[1][2] = -h;
	P[2][0] = 1; P[2][1] = yb; P[2][2] = h;
	P[3][0] = -1; P[3][1] = yb; P[3][2] = h;
	quad_fill(V, (const double (*)[3])P, P_PANE);
	for (i = 0; i < t[0].n; i++) {
		double X = nrm(B, 0, t[0].v[i], h);
		line3(V, X, -1, -h, X, 1, -h, P_PANEGRID);
		line3(V, X, yb, -h, X, yb, h, P_PANEGRID);
	}
	for (i = 0; i < t[1].n; i++) {
		double Y = nrm(B, 1, t[1].v[i], h);
		line3(V, -1, Y, -h, 1, Y, -h, P_PANEGRID);
		line3(V, xb, Y, -h, xb, Y, h, P_PANEGRID);
	}
	for (i = 0; i < t[2].n; i++) {
		double Z = nrm(B, 2, t[2].v[i], h);
		line3(V, xb, -1, Z, xb, 1, Z, P_PANEGRID);
		line3(V, -1, yb, Z, 1, yb, Z, P_PANEGRID);
	}
	/* the panes' outer edges */
	line3(V, -1, -1, -h, 1, -1, -h, P_LEGBORDER);
	line3(V, -1, 1, -h, 1, 1, -h, P_LEGBORDER);
	line3(V, -1, -1, -h, -1, 1, -h, P_LEGBORDER);
	line3(V, 1, -1, -h, 1, 1, -h, P_LEGBORDER);
	line3(V, xb, yb, -h, xb, yb, h, P_LEGBORDER);
	line3(V, xb, -yb, -h, xb, -yb, h, P_LEGBORDER);
	line3(V, -xb, yb, -h, -xb, yb, h, P_LEGBORDER);
	line3(V, xb, -1, h, xb, 1, h, P_LEGBORDER);
	line3(V, -1, yb, h, 1, yb, h, P_LEGBORDER);
	if (!labels)
		return;
	/* x labels along the front edge y = -yb, y labels along x = -xb */
	for (i = 0; i < t[0].n; i++) {
		double X = nrm(B, 0, t[0].v[i], h);
		proj(V, X, -yb * 1.2, -h, &sx, &sy, NULL);
		text((int)sx - textw(t[0].lab[i]) / 2, (int)sy - 4, t[0].lab[i], P_DIM);
	}
	for (i = 0; i < t[1].n; i++) {
		double Y = nrm(B, 1, t[1].v[i], h);
		proj(V, -xb * 1.2, Y, -h, &sx, &sy, NULL);
		text((int)sx - textw(t[1].lab[i]) / 2, (int)sy - 4, t[1].lab[i], P_DIM);
	}
	proj(V, 0, -yb * 1.48, -h, &sx, &sy, NULL);
	text((int)sx - textw(B->name[0]) / 2, (int)sy - 4, B->name[0], P_TEXT);
	proj(V, -xb * 1.48, 0, -h, &sx, &sy, NULL);
	text((int)sx - textw(B->name[1]) / 2, (int)sy - 4, B->name[1], P_TEXT);
	/* z labels on the vertical edge that is furthest left on screen */
	{
		double bx = 0, by = 0, best = INFINITY, ex, ey;
		int c;
		for (c = 0; c < 4; c++) {
			double X = c & 1 ? 1 : -1, Y = c & 2 ? 1 : -1;
			if (X == xb && Y == yb)
				continue;
			proj(V, X, Y, 0, &ex, &ey, NULL);
			if (ex < best) {
				best = ex;
				bx = X;
				by = Y;
			}
		}
		for (i = 0; i < t[2].n; i++) {
			double Z = nrm(B, 2, t[2].v[i], h);
			proj(V, bx, by, Z, &sx, &sy, NULL);
			fill((int)sx - 4, (int)sy, 4, 1, P_AXIS);
			text((int)sx - 7 - textw(t[2].lab[i]), (int)sy - 5, t[2].lab[i],
			    P_DIM);
		}
		proj(V, bx, by, h, &sx, &sy, NULL);
		text((int)sx - textw(B->name[2]) / 2, (int)sy - 18, B->name[2], P_TEXT);
	}
}

#define G3 64

/*
 * z = f(x, y) as a shaded, height-coloured surface over a grid of G3
 * cells, through the z-buffer, with every fourth grid line drawn over it
 * (only the lines when wire).  Vertices outside [zlo, zhi] or undefined
 * leave holes.
 */
static void
surface(struct ser *s, int si, const struct v3 *V, const struct box3 *B,
    int wire, int *nund, int *nclip)
{
	static double Z[(G3 + 1) * (G3 + 1)];
	static uint8_t okz[(G3 + 1) * (G3 + 1)];
	double in[MAXP * (G3 + 1)], vals[G3 + 1], h = V->h;
	double L[3] = {-0.45, -0.6, 0.66};
	int i, j, und = 0, clip = 0, np = s->f[0].np;
	uint8_t ok[G3 + 1];
	size_t q;

	for (q = 0; q < (size_t)C.W * C.H; q++)
		C.zb[q] = 1e30f;
	for (j = 0; j <= G3; j++) {
		for (i = 0; i <= G3; i++)
			plane_in(s, 0, B->lo[0] + (B->hi[0] - B->lo[0]) * i / G3,
			    B->lo[1] + (B->hi[1] - B->lo[1]) * j / G3, &in[i * np]);
		ev_run(&EV[si][0], G3 + 1, np ? in : NULL, vals, ok);
		for (i = 0; i <= G3; i++) {
			int k = j * (G3 + 1) + i;
			double zs = B->hi[2] - B->lo[2], v = vals[i];
			okz[k] = ok[i] == 1;
			if (!okz[k]) {
				und++;
				continue;
			}
			if (v < B->lo[2] - zs * 0.01 || v > B->hi[2] + zs * 0.01) {
				okz[k] = 0;
				clip++;
				continue;
			}
			Z[k] = nrm(B, 2, fmin(fmax(v, B->lo[2]), B->hi[2]), h);
		}
	}
	for (j = 0; j < G3; j++)
		for (i = 0; i < G3; i++) {
			int k00 = j * (G3 + 1) + i, k10 = k00 + 1, k01 = k00 + G3 + 1,
			    k11 = k01 + 1, c, sh;
			double X0 = -1 + 2.0 * i / G3, X1 = -1 + 2.0 * (i + 1) / G3;
			double Y0 = -1 + 2.0 * j / G3, Y1 = -1 + 2.0 * (j + 1) / G3;
			double ax, ay, az, bx, by, bz, nx, ny, nz, nl, lam, zm;
			double x[3], y[3], z[3], px[4], py[4], pz[4];
			if (!okz[k00] || !okz[k10] || !okz[k01] || !okz[k11])
				continue;
			ax = X1 - X0; ay = Y1 - Y0; az = Z[k11] - Z[k00];
			bx = X0 - X1; by = Y1 - Y0; bz = Z[k01] - Z[k10];
			nx = ay * bz - az * by;
			ny = az * bx - ax * bz;
			nz = ax * by - ay * bx;
			nl = sqrt(nx * nx + ny * ny + nz * nz);
			lam = nl > 0 ? fabs(nx * L[0] + ny * L[1] + nz * L[2]) / nl /
			    sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]) : 1;
			sh = 1 + (int)(lam * 3.999);
			zm = (Z[k00] + Z[k10] + Z[k01] + Z[k11]) / 4;
			c = wire ? -1 : cmap((zm + h) / (2 * h), sh);
			proj(V, X0, Y0, Z[k00], &px[0], &py[0], &pz[0]);
			proj(V, X1, Y0, Z[k10], &px[1], &py[1], &pz[1]);
			proj(V, X1, Y1, Z[k11], &px[2], &py[2], &pz[2]);
			proj(V, X0, Y1, Z[k01], &px[3], &py[3], &pz[3]);
			x[0] = px[0]; y[0] = py[0]; z[0] = pz[0];
			x[1] = px[1]; y[1] = py[1]; z[1] = pz[1];
			x[2] = px[2]; y[2] = py[2]; z[2] = pz[2];
			tri(x, y, z, c, 1);
			x[1] = px[2]; y[1] = py[2]; z[1] = pz[2];
			x[2] = px[3]; y[2] = py[3]; z[2] = pz[3];
			tri(x, y, z, c, 1);
		}
	/* grid lines, depth-tested */
	{
		int step = wire ? 2 : 4, dir;
		for (dir = 0; dir < 2; dir++)
			for (j = 0; j <= G3; j += step)
				for (i = 0; i < G3; i++) {
					int ka = dir ? i * (G3 + 1) + j : j * (G3 + 1) + i;
					int kb = dir ? (i + 1) * (G3 + 1) + j : ka + 1, n, t;
					double Xa, Ya, Xb, Yb, sa, ta, da, sb, tb, db;
					if (!okz[ka] || !okz[kb])
						continue;
					Xa = -1 + 2.0 * (dir ? j : i) / G3;
					Ya = -1 + 2.0 * (dir ? i : j) / G3;
					Xb = dir ? Xa : -1 + 2.0 * (i + 1) / G3;
					Yb = dir ? -1 + 2.0 * (i + 1) / G3 : Ya;
					proj(V, Xa, Ya, Z[ka], &sa, &ta, &da);
					proj(V, Xb, Yb, Z[kb], &sb, &tb, &db);
					n = (int)ceil(fmax(fabs(sb - sa), fabs(tb - ta))) + 1;
					for (t = 0; t <= n; t++) {
						double f = (double)t / n, X = sa + (sb - sa) * f,
						    Y = ta + (tb - ta) * f, D = da + (db - da) * f;
						int xi = (int)floor(X), yi = (int)floor(Y);
						double zz = Z[ka] + (Z[kb] - Z[ka]) * f;
						if (xi < 0 || yi < 0 || xi >= C.W || yi >= C.H)
							continue;
						if (D <= C.zb[(size_t)yi * C.W + xi] + 0.03)
							put(xi, yi, cmap((zz + h) / (2 * h), wire ? 2 : 0));
					}
				}
	}
	if (nund)
		*nund = und;
	if (nclip)
		*nclip = clip;
}

/* a space curve (x(t), y(t), z(t)) in the box, coloured along t */
static void
space_curve(struct ser *s, const struct v3 *V, const struct box3 *B)
{
	int i, l;

	for (l = 0; l < 8; l++) {
		int t[4], any = 0;
		for (i = 0; i + 1 < s->cn; i++) {
			double a, b, c, d;
			if (isnan(s->cx[i]) || isnan(s->cx[i + 1]))
				continue;
			if ((int)((double)i / s->cn * 8) != l)
				continue;
			proj(V, nrm(B, 0, s->cx[i], V->h), nrm(B, 1, s->cy[i], V->h),
			    nrm(B, 2, s->cy[s->cn + i], V->h), &a, &b, NULL);
			proj(V, nrm(B, 0, s->cx[i + 1], V->h), nrm(B, 1, s->cy[i + 1],
			    V->h), nrm(B, 2, s->cy[s->cn + i + 1], V->h), &c, &d, NULL);
			cov_seg(a, b, c, d, 2.2);
			any = 1;
		}
		if (!any)
			continue;
		t[0] = t[1] = cmap(0.05 + 0.85 * l / 7.0, 3);
		t[2] = t[3] = cmap(0.05 + 0.85 * l / 7.0, 5);
		cov_put(t);
	}
}

/* ---- building the series from the text ---- */

static int
named_ui(uint32_t name)
{
	const char *n = sym_str(name);

	return !strcmp(n, "it") || !strcmp(n, "that") || !strcmp(n, "lhs") ||
	    !strcmp(n, "rhs") || n[0] == '_';
}

static void
label_of(struct ser *s, const char *prefix)
{
	const struct func *f = &s->f[0];
	struct sbuf b = {0};
	int k;

	sb_puts(&b, prefix);
	if (!f->anon && !named_ui(f->name)) {
		sb_printf(&b, "%s(", sym_str(f->name));
		for (k = 0; k < f->np; k++)
			sb_printf(&b, "%s%s", k ? ", " : "", sym_str(f->p[k].name));
		sb_puts(&b, ") = ");
	}
	sb_puts(&b, node_str(f->body));
	ascii_of(s->label, sizeof s->label, b.p);
	free(b.p);
}

/* the slots of x and y: by name, else in order */
static void
pick_xy(const struct func *f, int *sx, int *sy)
{
	int k;

	*sx = slot_named(f, "x");
	*sy = slot_named(f, "y");
	if (f->np == 1 && *sx < 0 && *sy < 0) {
		*sx = 0;
		return;
	}
	for (k = 0; k < f->np && *sx < 0; k++)
		if (k != *sy)
			*sx = k;
	for (k = 0; k < f->np && *sy < 0; k++)
		if (k != *sx)
			*sy = k;
}

static double
const_text(const char *t)
{
	struct func tmp;
	nid e;

	memset(&tmp, 0, sizeof tmp);
	tmp.name = sym_cstr("_");
	e = calc_parse_text(t, &tmp, TY_REAL);
	if (tmp.np || !is_const(e))
		fail("plot: expected a number, found '%s'", t);
	return TY_FP(N.ty[e]) ? N.kr[e] : (double)N.ki[e];
}

/* y' = f(t, y) from its right-hand side; dep is the unknown */
static void
slope_from(struct ser *s, const char *rhs, uint32_t dep, int stored)
{
	const char *t0, *val;
	int k;

	s->kind = S_SLOPE;
	s->f[0] = *cmd_func_text(rhs);
	s->nf = 1;
	s->yname = dep;
	s->sy = -1;
	s->sx = -1;
	for (k = 0; k < s->f[0].np; k++)
		if (s->f[0].p[k].name == dep)
			s->sy = k;
	for (k = 0; k < s->f[0].np; k++)
		if (k != s->sy && s->sx < 0)
			s->sx = k;
	if (s->f[0].np > 2)
		fail("plot: a slope field needs y' = f(t, y) of two variables (%s "
		    "has %d)", rhs, s->f[0].np);
	s->xname = s->sx >= 0 ? s->f[0].p[s->sx].name : sym_cstr("t");
	if (O.has_ic && O.ic_name != dep)
		printf("  note: the initial value is for %s, the unknown is %s: not "
		    "used\n", sym_str(O.ic_name), sym_str(dep));
	if (O.has_ic && O.ic_name == dep) {
		s->has_ic = 1;
		s->ic_t = O.ic_t;
		s->ic_y = O.ic_y;
	} else 	if (stored && ode_ic_text(dep, 0, &t0, &val)) {
		s->has_ic = 1;
		s->ic_t = const_text(t0);
		s->ic_y = const_text(val);
	}
	snprintf(s->label, sizeof s->label, "%s' = ", sym_str(dep));
	ascii_of(s->label + strlen(s->label), sizeof s->label -
	    strlen(s->label), node_str(s->f[0].body));
}

static void
add_piece(int a, int b)
{
	char txt[1024], lhs[256], rhs[1024];
	int k, eq = -1;
	struct ser *s;
	struct func *f;

	if (a >= b)
		return;
	if (O.ns == MAXSER)
		fail("plot: at most %d functions at once", MAXSER);
	s = &O.s[O.ns];
	memset(s, 0, sizeof *s);
	s->sx = s->sy = -1;
	s->col = O.ns;
	tk_str(a, b, txt, sizeof txt);
	for (k = a; k < b; k++)
		if (T[k].d == 0 && T[k].k == '=') {
			eq = k;
			break;
		}
	if (eq < 0 && T[a].k == '(' && T[b - 1].k == ')' && T[b - 1].d == T[a].d) {
		/* a tuple: (x(t), y(t)[, z(t)]) or (P, Q) */
		int d1 = T[a].d + 1, st = a + 1, nc = 0, ok = 1;
		for (k = a + 1; k < b - 1 && ok; k++)
			if (T[k].d < d1)
				ok = 0;	/* (a)(b): not one group */
		if (ok)
			for (k = a + 1; k <= b - 1; k++)
				if (k == b - 1 || (T[k].d == d1 && T[k].k == ',')) {
					char ct[512];
					if (nc == 3)
						fail("plot: a tuple has at most three components");
					tk_str(st, k, ct, sizeof ct);
					s->f[nc++] = *cmd_func_text(ct);
					st = k + 1;
				}
		if (ok && nc >= 2) {
			uint32_t names[8];
			int nn = 0, c, j;
			s->nf = nc;
			for (c = 0; c < nc; c++)
				for (k = 0; k < s->f[c].np; k++) {
					for (j = 0; j < nn; j++)
						if (names[j] == s->f[c].p[k].name)
							break;
					if (j == nn && nn < 8)
						names[nn++] = s->f[c].p[k].name;
				}
			if (nn <= 1) {
				s->kind = nc == 3 ? S_SPACE : S_PARAM;
				s->xname = nn ? names[0] : sym_cstr("t");
			} else if (nn == 2 && nc == 2) {
				s->kind = S_VFIELD;
				s->xname = names[0] == sym_cstr("y") ? names[1] : names[0];
				s->yname = names[0] == sym_cstr("y") ? names[0] : names[1];
				if (names[1] == sym_cstr("x")) {
					s->xname = names[1];
					s->yname = names[0];
				}
			} else
				fail("plot: %s: a curve has one parameter, a vector field "
				    "(P, Q) two variables", txt);
			ascii_of(s->label, sizeof s->label, txt);
			O.ns++;
			return;
		}
	}
	if (eq < 0) {
		uint32_t nm = 0;
		int ord = 0;
		const char *ode;
		if (b == a + 1 && T[a].k == 'w') {
			nm = sym_cstr(txt);
			if ((ode = ode_rhs_text(nm, &ord)) != NULL && ord == 1 &&
			    (func_find(nm) == NULL || O.field)) {
				slope_from(s, ode, nm, 1);
				O.ns++;
				return;
			}
		}
		f = cmd_func_text(txt);
		s->f[0] = *f;
		s->nf = 1;
		if (O.field && f->np <= 2) {
			uint32_t dep = sym_cstr("y");
			char rt[1024];
			snprintf(rt, sizeof rt, "%s", txt);
			slope_from(s, rt, dep, 0);
			O.ns++;
			return;
		}
		if (O.polar && f->np <= 1) {
			s->kind = S_POLAR;
			s->sx = f->np ? 0 : -1;
			label_of(s, "r = ");
		} else if (f->np <= 1) {
			s->kind = S_FX;
			s->sx = f->np ? 0 : -1;
			label_of(s, "");
		} else if (f->np == 2) {
			s->kind = TYK(N.ty[f->body]) == K_BOOL ? S_REGION : S_SURF;
			pick_xy(f, &s->sx, &s->sy);
			label_of(s, "");
			if (s->kind == S_REGION) {
				int op = N.op[f->body];
				if (op == O_LT || op == O_LE || op == O_GT || op == O_GE ||
				    op == O_EQ)
					s->bound = mk_bin(O_SUB, N.a[f->body], N.b[f->body]);
			}
		} else {
			s->kind = S_ND;
			pick_xy(f, &s->sx, &s->sy);
			label_of(s, "");
		}
		O.ns++;
		return;
	}
	tk_str(a, eq, lhs, sizeof lhs);
	tk_str(eq + 1, b, rhs, sizeof rhs);
	if (!rhs[0] || !lhs[0])
		fail("plot: %s: an equation needs two sides", txt);
	if (lhs[strlen(lhs) - 1] == '\'' && eq == a + 1) {
		char dep[64];
		snprintf(dep, sizeof dep, "%s", lhs);
		while (dep[0] && dep[strlen(dep) - 1] == '\'')
			dep[strlen(dep) - 1] = 0;
		if (strlen(lhs) - strlen(dep) != 1)
			fail("plot: slope fields are for first-order equations");
		slope_from(s, rhs, sym_cstr(dep), 0);
		O.ns++;
		return;
	}
	if (eq == a + 1 || (eq > a + 2 && T[a].k == 'w' && T[a + 1].k == '(' &&
	    T[eq - 1].k == ')' && T[eq - 1].d == T[a + 1].d)) {
		/* y = ..., r = ..., z = ..., f(x) = ... */
		f = cmd_func_text(rhs);
		if (eq == a + 1 && slot_named(f, lhs) < 0 && strcmp(lhs, "x")) {
			s->f[0] = *f;
			s->nf = 1;
			if (!strcmp(lhs, "r") && f->np <= 1) {
				s->kind = S_POLAR;
				s->sx = f->np ? 0 : -1;
				label_of(s, "r = ");
				O.ns++;
				return;
			}
			if (f->np <= 1) {
				char pre[80];
				s->kind = S_FX;
				s->sx = f->np ? 0 : -1;
				snprintf(pre, sizeof pre, "%s = ", lhs);
				label_of(s, pre);
				s->yname = sym_cstr(lhs);
				O.ns++;
				return;
			}
			if (f->np == 2) {
				char pre[80];
				s->kind = S_SURF;
				pick_xy(f, &s->sx, &s->sy);
				snprintf(pre, sizeof pre, "%s = ", lhs);
				label_of(s, pre);
				s->yname = sym_cstr(lhs);
				O.ns++;
				return;
			}
		} else if (eq != a + 1 && f->np <= 1) {
			char pre[300];
			s->f[0] = *f;
			s->nf = 1;
			s->kind = S_FX;
			s->sx = f->np ? 0 : -1;
			snprintf(pre, sizeof pre, "%s = ", lhs);
			s->label[0] = 0;
			ascii_of(s->label, sizeof s->label, pre);
			ascii_of(s->label + strlen(s->label), sizeof s->label -
			    strlen(s->label), node_str(f->body));
			O.ns++;
			return;
		}
	}
	/* an implicit curve: lhs - rhs = 0 */
	{
		char it[1400];
		snprintf(it, sizeof it, "(%s) - (%s)", lhs, rhs);
		f = cmd_func_text(it);
		s->f[0] = *f;
		s->nf = 1;
		s->kind = S_IMPL;
		if (f->np > 2)
			fail("plot: %s: an implicit curve in more than two variables "
			    "(implicit surfaces are not drawn)", txt);
		pick_xy(f, &s->sx, &s->sy);
		if (f->np == 1 && slot_named(f, "y") == 0) {
			s->sx = -1;
			s->sy = 0;
		}
		ascii_of(s->label, sizeof s->label, txt);
		O.ns++;
	}
}

/* ---- ranges ---- */

static void
range_line(const char *vn, double lo, double hi, const char *why)
{
	printf("  %s in [%s, %s]%s%s\n", vn, gpi(lo), gpi(hi), why && why[0] ?
	    ": " : "", why ? why : "");
}

/* the default value of an unplotted slot: the middle of its range */
static double
slot_mid(const struct func *f, int k)
{
	const struct param *p = &f->p[k];
	struct rng *r = rng_find(p->name);

	if (r)
		return (r->lo + r->hi) / 2;
	if (p->ranged)
		return TY_FP(p->ty) ? (p->rlo + p->rhi) / 2 :
		    floor(((double)p->lo + (double)p->hi) / 2);
	if (TY_SIZED(p->ty)) {
		int64_t a, b;
		param_bounds(p, &a, &b);
		return floor(((double)a + (double)b) / 2);
	}
	return 0;
}

static void
base_of(const struct func *f, double *base, double generic)
{
	int k;

	for (k = 0; k < MAXP; k++)
		base[k] = 0;
	for (k = 0; k < f->np; k++) {
		const struct param *p = &f->p[k];
		/* a different value per slot: x - t is not 0 on the slice */
		base[k] = (rng_find(p->name) || p->ranged || TY_SIZED(p->ty)) ?
		    slot_mid(f, k) : TY_FP(p->ty) ? generic + (generic ? 0.37 * k : 0) :
		    1;
	}
}

static char last_why[300];

/*
 * The range of slot k of series si, component c: given, declared, or
 * chosen; printed with the reason.  at: index into O.at, or -1.
 */
static void
slot_auto(int si, int c, int k, int at, double *lo, double *hi,
    struct sbuf *whyp, int show)
{
	struct ser *s = &O.s[si];
	const struct func *f = &s->f[c];
	struct sbuf why = {0};
	double base[MAXP];

	base_of(f, base, 0.7);
	auto1(&EV[si][c], base, k, at >= 0 && O.nat > at, at >= 0 ? O.at[at] : 0,
	    lo, hi, &why, show);
	/*
	 * With other variables the slice at 0.7 can be lopsided (x^2 + 0.7x)
	 * or degenerate (sin(0.7 - t) cos(y) at t = 0.7): the slices at a few
	 * more values decide too; a period wins, features are united, and a
	 * range around 0 is made symmetric.
	 */
	if (f->np > 1 && why.p && !(at >= 0 && O.nat > at) &&
	    (strstr(why.p, "feature") || strstr(why.p, "default"))) {
		static const double bv[] = {0.3, -0.7, 0, 1.3};
		int g, feat = strstr(why.p, "feature") != NULL;
		for (g = 0; g < 4; g++) {
			struct sbuf w2 = {0};
			double l2, h2;
			base_of(f, base, bv[g]);
			auto1(&EV[si][c], base, k, 0, 0, &l2, &h2, &w2, 0);
			if (w2.p && strstr(w2.p, "periodic")) {
				*lo = l2;
				*hi = h2;
				sb_clear(&why);
				sb_puts(&why, w2.p);
				free(w2.p);
				feat = 0;
				break;
			}
			if (w2.p && strstr(w2.p, "feature")) {
				if (!feat) {
					*lo = l2;
					*hi = h2;
					sb_clear(&why);
					sb_puts(&why, w2.p);
					feat = 1;
				}
				if (l2 < *lo)
					*lo = l2;
				if (h2 > *hi)
					*hi = h2;
			}
			free(w2.p);
		}
		if (feat && *lo < 0 && *hi > 0) {
			double m = fmax(-*lo, *hi);
			*lo = -m;
			*hi = m;
		}
	}
	*whyp = why;
}

/* the automatic ranges of this plot's slots, computed once */
static struct {
	int si, c, k;
	double lo, hi;
	char why[300];
} RC[24];
static int nrc;

static const char *
slot_auto_cached(int si, int c, int k, double *lo, double *hi)
{
	struct sbuf w = {0};
	int i;

	for (i = 0; i < nrc; i++)
		if (RC[i].si == si && RC[i].c == c && RC[i].k == k) {
			*lo = RC[i].lo;
			*hi = RC[i].hi;
			return RC[i].why;
		}
	slot_auto(si, c, k, -1, lo, hi, &w, 0);
	if (nrc == 24)
		nrc = 0;
	RC[nrc].si = si;
	RC[nrc].c = c;
	RC[nrc].k = k;
	RC[nrc].lo = *lo;
	RC[nrc].hi = *hi;
	snprintf(RC[nrc].why, sizeof RC[0].why, "%s", w.p ? w.p : "");
	free(w.p);
	return RC[nrc++].why;
}

/* f(.., x, .., y, ..) = ±f(.., y, .., x, ..) at 32 points */
static int
swap_sym(int si, int c, int k, int j)
{
	const struct func *f = &O.s[si].f[c];
	double in[MAXP], v1, v2;
	uint32_t s = 2654435761u;
	int i, n = 0;

	for (i = 0; i < 32; i++) {
		double a, b;
		base_of(f, in, 0.7);
		s = s * 1103515245u + 12345u;
		a = -3 + 6 * ((s >> 8) & 0xffff) / 65536.0;
		s = s * 1103515245u + 12345u;
		b = -3 + 6 * ((s >> 8) & 0xffff) / 65536.0;
		in[k] = a;
		in[j] = b;
		if (!ev1(&EV[si][c], in, &v1)) {
			in[k] = b;
			in[j] = a;
			if (ev1(&EV[si][c], in, &v2))
				return 0;
			continue;
		}
		in[k] = b;
		in[j] = a;
		if (!ev1(&EV[si][c], in, &v2) ||
		    fabs(fabs(v1) - fabs(v2)) > 1e-9 * fmax(1, fabs(v1)))
			return 0;
		n++;
	}
	return n >= 16;
}

/*
 * The range of slot k of series si, component c: given, declared, or
 * chosen; printed with the reason.  at: index into O.at, or -1.  With
 * several variables, a range chosen from the features is made the same
 * as another's when f is symmetric in them (up to sign) or the two are
 * alike (around 0, spans within 1.5x): x^2 - y^2 gets a square.
 */
static void
slot_range(int si, int c, int k, int at, double *lo, double *hi, int quiet,
    int *periodic)
{
	const struct func *f = &O.s[si].f[c];
	const char *vn = sym_str(f->p[k].name);
	struct rng *r = rng_find(f->p[k].name);
	struct sbuf why = {0};

	if (periodic)
		*periodic = 0;
	if (r) {
		*lo = r->lo;
		*hi = r->hi;
		if (!quiet)
			range_line(vn, *lo, *hi, "given");
		snprintf(last_why, sizeof last_why, "given");
		return;
	}
	slot_auto(si, c, k, at, lo, hi, &why, O.mode == 2);
	if (f->np > 1 && !(at >= 0 && O.nat > at) && why.p &&
	    strstr(why.p, "feature") && TY_FP(f->p[k].ty) && !f->p[k].ranged) {
		double l0 = *lo, h0 = *hi;
		int j, by = -1, bysym = 0;
		for (j = 0; j < f->np; j++) {
			const char *wj;
			double lj, hj, sk, sj;
			int sy;
			if (j == k || !TY_FP(f->p[j].ty) || f->p[j].ranged ||
			    rng_find(f->p[j].name))
				continue;
			wj = slot_auto_cached(si, c, j, &lj, &hj);
			if (!strstr(wj, "feature"))
				continue;
			sk = h0 - l0;
			sj = hj - lj;
			sy = swap_sym(si, c, k, j);
			if (!sy && !(l0 == -h0 && lj == -hj && fmax(sk, sj) <=
			    1.5 * fmin(sk, sj)))
				continue;
			if (lj < *lo || hj > *hi) {
				by = j;
				bysym = sy;
			}
			if (lj < *lo)
				*lo = lj;
			if (hj > *hi)
				*hi = hj;
		}
		if (by >= 0 && bysym)
			sb_printf(&why, "; as for %s (f(%s, %s) = \xc2\xb1" "f(%s, %s))",
			    sym_str(f->p[by].name), vn, sym_str(f->p[by].name),
			    sym_str(f->p[by].name), vn);
		else if (by >= 0)
			sb_printf(&why, "; as for %s (alike: the features do not set "
			    "them apart)", sym_str(f->p[by].name));
	}
	if (periodic && why.p && strstr(why.p, "periodic"))
		*periodic = 1;
	if (!quiet)
		range_line(vn, *lo, *hi, why.p);
	snprintf(last_why, sizeof last_why, "%s", why.p ? why.p : "");
	free(why.p);
}

/* frames: delays in hundredths that add up to the right time */
static int
delay_cs(int k, int fps)
{
	return (int)floor(100.0 * (k + 1) / fps + 0.5) -
	    (int)floor(100.0 * k / fps + 0.5);
}

static void
set_fix(uint32_t var, double v)
{
	int k, c, j;

	for (k = 0; k < O.ns; k++)
		for (c = 0; c < O.s[k].nf; c++)
			for (j = 0; j < O.s[k].f[c].np; j++)
				if (O.s[k].f[c].p[j].name == var)
					O.s[k].fix[c][j] = v;
}

/* the animation over var: its range and frame values */
struct anim {
	uint32_t var;
	double lo, hi;
	int n, fps, loop;
};

static void
anim_setup(struct anim *an, uint32_t var, int dfr, int dfps)
{
	int k, c, j, per = 0;

	an->var = var;
	an->n = 1;
	an->fps = O.fps > 0 ? O.fps : dfps;
	if (an->fps > 50)
		an->fps = 50;
	if (!var)
		return;
	for (k = 0; k < O.ns; k++)
		for (c = 0; c < O.s[k].nf; c++)
			for (j = 0; j < O.s[k].f[c].np; j++)
				if (O.s[k].f[c].p[j].name == var) {
					slot_range(k, c, j, -1, &an->lo, &an->hi, 0, &per);
					goto found;
				}
	fail("plot: nothing depends on %s", sym_str(var));
found:
	if (per && an->lo == -an->hi) {	/* one period loops seamlessly */
		an->lo = 0;
		an->loop = 1;
		printf("  %s: one period [0, %s), looping\n", sym_str(var),
		    gpi(an->hi));
	}
	an->n = O.frames > 0 ? O.frames : O.seconds > 0 ?
	    (int)(O.seconds * an->fps + 0.5) : dfr;
	if (an->n < 2)
		an->n = 2;
	if (an->n > 240)
		an->n = 240;
}

static double
anim_val(const struct anim *an, int f)
{
	if (an->n <= 1)
		return an->lo;
	return an->lo + (an->hi - an->lo) * f / (an->loop ? an->n : an->n - 1);
}

static void
equal_aspect(struct ax2 *A, int W, int H)
{
	struct ax2 t = *A;
	double ux, uy;

	layout2(&t, 0, 0, W, H, 0, 0);
	ux = (A->hi[0] - A->lo[0]) / (t.R - t.L);
	uy = (A->hi[1] - A->lo[1]) / (t.B - t.T);
	if (ux > uy) {
		double m = (A->lo[1] + A->hi[1]) / 2, h = ux * (t.B - t.T) / 2;
		A->lo[1] = m - h;
		A->hi[1] = m + h;
	} else {
		double m = (A->lo[0] + A->hi[0]) / 2, w = uy * (t.R - t.L) / 2;
		A->lo[0] = m - w;
		A->hi[0] = m + w;
	}
}

/* where the implicit curve is, in [-10, 10]^2 */
static int
impl_bbox(int si, double *bb)
{
	enum { NG = 161 };
	struct ser *s = &O.s[si];
	struct ev *e = &EV[si][0];
	int np = e->f.np, i, j, found = 0;
	double *in = xmalloc((size_t)NG * NG * (np ? np : 1) * sizeof *in);
	double *v = xmalloc(NG * NG * sizeof *v);
	uint8_t *ok = xmalloc(NG * NG);

	for (j = 0; j < NG; j++)
		for (i = 0; i < NG; i++)
			plane_in(s, 0, -10 + 20.0 * i / (NG - 1), -10 + 20.0 * j / (NG - 1),
			    &in[(size_t)(j * NG + i) * np]);
	ev_run(e, NG * NG, np ? in : NULL, v, ok);
	bb[0] = bb[2] = INFINITY;
	bb[1] = bb[3] = -INFINITY;
	for (j = 0; j + 1 < NG; j++)
		for (i = 0; i + 1 < NG; i++) {
			int k = j * NG + i, q[4] = {k, k + 1, k + NG, k + NG + 1}, a, pos = 0,
			    neg = 0;
			for (a = 0; a < 4; a++)
				if (ok[q[a]] == 1) {
					pos += v[q[a]] > 0;
					neg += v[q[a]] <= 0;
				}
			if (pos && neg && pos + neg == 4) {
				double x = -10 + 20.0 * (i + 0.5) / (NG - 1), y = -10 + 20.0 *
				    (j + 0.5) / (NG - 1);
				found = 1;
				if (x < bb[0]) bb[0] = x;
				if (x > bb[1]) bb[1] = x;
				if (y < bb[2]) bb[2] = y;
				if (y > bb[3]) bb[3] = y;
			}
		}
	free(in);
	free(v);
	free(ok);
	return found;
}

/* a parametric or polar curve's points */
static void
sample_curve(int si)
{
	struct ser *s = &O.s[si];
	int n = O.npts > 1 ? O.npts : s->kind == S_POLAR ? 3000 : 2000, i, c;
	int nc = s->kind == S_SPACE ? 3 : 2;
	double *ts = xmalloc(n * sizeof *ts), *vv[3];
	uint8_t *ok[3];

	free(s->cx);
	free(s->cy);
	s->cx = xmalloc(n * sizeof *s->cx);
	s->cy = xmalloc(2 * n * sizeof *s->cy);
	s->cn = n;
	for (i = 0; i < n; i++)
		ts[i] = s->t0 + (s->t1 - s->t0) * i / (n - 1);
	for (c = 0; c < s->nf; c++) {
		int slot = slot_named(&s->f[c], sym_str(s->xname));
		vv[c] = xmalloc(n * sizeof *vv[c]);
		ok[c] = xmalloc(n);
		f1n(&EV[si][c], s->fix[c], slot, n, ts, vv[c], ok[c]);
	}
	for (i = 0; i < n; i++) {
		int good = 1;
		for (c = 0; c < s->nf; c++)
			good &= ok[c][i] == 1;
		if (!good) {
			s->cx[i] = s->cy[i] = NAN;
			if (nc == 3)
				s->cy[n + i] = NAN;
			continue;
		}
		if (s->kind == S_POLAR) {
			s->cx[i] = vv[0][i] * cos(ts[i]);
			s->cy[i] = vv[0][i] * sin(ts[i]);
		} else {
			s->cx[i] = vv[0][i];
			s->cy[i] = vv[1][i];
			if (nc == 3)
				s->cy[n + i] = vv[2][i];
		}
	}
	for (c = 0; c < s->nf; c++) {
		free(vv[c]);
		free(ok[c]);
	}
	free(ts);
}

/* a curve's parameter range: given, declared, else a default */
static void
curve_trange(int si)
{
	struct ser *s = &O.s[si];
	struct rng *r = rng_find(s->xname);
	const char *tn = sym_str(s->xname);
	int c, k;

	if (s->kind == S_POLAR && s->sx >= 0)
		s->xname = s->f[0].p[s->sx].name, tn = sym_str(s->xname),
		r = rng_find(s->xname);
	if (r) {
		s->t0 = r->lo;
		s->t1 = r->hi;
		range_line(tn, s->t0, s->t1, "given");
		return;
	}
	for (c = 0; c < s->nf; c++)
		if ((k = slot_named(&s->f[c], tn)) >= 0 && s->f[c].p[k].ranged) {
			s->t0 = s->f[c].p[k].rlo;
			s->t1 = s->f[c].p[k].rhi;
			range_line(tn, s->t0, s->t1, "declared");
			return;
		}
	s->t0 = 0;
	s->t1 = 2 * M_PI;
	if (s->kind == S_POLAR && s->sx >= 0) {
		double P, base[MAXP];
		base_of(&s->f[0], base, 0.7);
		if (guess_period(&EV[si][0], base, s->sx, &P) && P > 1e-6) {
			int m;
			/* the curve closes after m P, the first multiple of 2pi */
			for (m = 1; m <= 64; m++)
				if (fabs(m * P / (2 * M_PI) - floor(m * P / (2 * M_PI) + 0.5)) <
				    1e-6 && m * P > 2 * M_PI - 1e-6)
					break;
			if (m > 64 || m * P <= 2 * M_PI + 1e-6)
				goto dflt;
			s->t1 = m * P;
			if (s->t1 > 16 * M_PI)
				s->t1 = 16 * M_PI;
			range_line(tn, s->t0, s->t1, "until the curve closes");
			return;
		}
	}
dflt:
	range_line(tn, s->t0, s->t1, "default");
}

/* ---- the image ---- */

static void
finish_gif(struct gif *g, int W, int H, int nfr, int fps)
{
	gif_end(g);
	if (SUM.p)
		fputs(SUM.p, stdout);
	if (nfr <= 1)
		printf("  image: %dx%d, %zu bytes\n", W, H, g->out.n);
	else
		printf("  animation: %dx%d, %d frames at %d fps (%.1f s), %zu "
		    "bytes\n", W, H, nfr, fps, (double)nfr / fps, g->out.n);
	gif_deliver(g, O.name, O.nofile);
	free(g->out.p);
	free(g->prev);
}

static void
join_labels(char *t, size_t n)
{
	int k, c = 0;

	t[0] = 0;
	for (k = 0; k < O.ns; k++) {
		if (O.s[k].kind == S_DERIV || O.s[k].kind == S_TANGENT)
			continue;
		if (c++)
			strncat(t, ", ", n - strlen(t) - 1);
		strncat(t, O.s[k].label, n - strlen(t) - 1);
	}
}

/* ---- 2-D: curves, implicit curves, regions, fields ---- */

static void
plot_2d(uint32_t av)
{
	struct ax2 A;
	struct anim an;
	struct gif g;
	int k, has_fx = 0, has_plane = 0, has_curve = 0, has_slope = 0, f;
	int xset = 0, yset = 0, W = 640, H = 440, eq = 0;
	uint32_t xn = 0, yn = 0;
	char title[400], tt[480];
	struct rng *r;

	memset(&A, 0, sizeof A);
	A.lg[0] = O.logx;
	A.lg[1] = O.logy;
	for (k = 0; k < O.ns; k++) {
		struct ser *s = &O.s[k];
		base_of(&s->f[0], s->fix[0], 0);
		if (s->nf > 1)
			base_of(&s->f[1], s->fix[1], 0);
		if (s->nf > 2)
			base_of(&s->f[2], s->fix[2], 0);
		switch (s->kind) {
		case S_FX:
		case S_DERIV:
			has_fx = 1;
			if (!xn && s->sx >= 0)
				xn = s->f[0].p[s->sx].name;
			if (!yn && s->yname)
				yn = s->yname;
			break;
		case S_SLOPE:
			has_slope = 1;
			xn = s->xname;
			yn = s->yname;
			break;
		case S_VFIELD:
			has_plane = 1;
			if (!xn)
				xn = s->xname;
			if (!yn)
				yn = s->yname;
			break;
		case S_IMPL:
		case S_REGION:
			has_plane = 1;
			if (!xn)
				xn = s->sx >= 0 ? s->f[0].p[s->sx].name : sym_cstr("x");
			if (!yn)
				yn = s->sy >= 0 ? s->f[0].p[s->sy].name : sym_cstr("y");
			break;
		case S_PARAM:
		case S_POLAR:
			has_curve = 1;
			break;
		}
	}
	if (!xn)
		xn = sym_cstr("x");
	if (!yn)
		yn = sym_cstr("y");
	anim_setup(&an, av, 30, 12);
	if (av)
		set_fix(av, anim_val(&an, 0));
	/* the horizontal range */
	if ((r = rng_find(xn)) != NULL || (r = rng_find(sym_cstr("x"))) != NULL) {
		A.lo[0] = r->lo;
		A.hi[0] = r->hi;
		xset = 1;
		range_line(sym_str(xn), A.lo[0], A.hi[0], "given");
	} else if (has_fx) {
		int first = 1, nfx = 0;
		for (k = 0; k < O.ns; k++)
			if (O.s[k].kind == S_FX && O.s[k].sx >= 0)
				nfx++;
		double rl[MAXSER], rh[MAXSER];
		char rw[MAXSER][300];
		int ri[MAXSER], nr = 0, same = 1;
		for (k = 0; k < O.ns; k++) {
			struct ser *s = &O.s[k];
			if (s->kind != S_FX || s->sx < 0)
				continue;
			slot_range(k, 0, s->sx, 0, &rl[nr], &rh[nr], 1, NULL);
			snprintf(rw[nr], sizeof rw[0], "%s", last_why);
			ri[nr] = k;
			if (nr && (rl[nr] != rl[0] || rh[nr] != rh[0]))
				same = 0;
			if (first || rl[nr] < A.lo[0])
				A.lo[0] = rl[nr];
			if (first || rh[nr] > A.hi[0])
				A.hi[0] = rh[nr];
			first = 0;
			nr++;
		}
		(void)nfx;
		if (first) {
			A.lo[0] = -5;
			A.hi[0] = 5;
			range_line(sym_str(xn), A.lo[0], A.hi[0], "constant: a default");
		} else if (same)
			range_line(sym_str(xn), A.lo[0], A.hi[0], rw[0]);
		else {
			for (k = 0; k < nr; k++) {
				char t[400];
				snprintf(t, sizeof t, "%s for %s", rw[k], O.s[ri[k]].label);
				range_line(sym_str(xn), rl[k], rh[k], t);
			}
			range_line(sym_str(xn), A.lo[0], A.hi[0], "all of them");
		}
		xset = 1;
	} else if (has_slope) {
		struct ser *s = NULL;
		for (k = 0; k < O.ns; k++)
			if (O.s[k].kind == S_SLOPE)
				s = &O.s[k];
		A.lo[0] = s->has_ic ? s->ic_t : 0;
		A.hi[0] = A.lo[0] + 5;
		xset = 1;
		range_line(sym_str(xn), A.lo[0], A.hi[0], s->has_ic ?
		    "from the initial value, 5 units" : "default");
	}
	if ((r = rng_find(yn)) != NULL || (!(has_fx && yn == sym_cstr("x")) &&
	    (r = rng_find(sym_cstr("y"))) != NULL)) {
		A.lo[1] = r->lo;
		A.hi[1] = r->hi;
		yset = 1;
		range_line(sym_str(yn), A.lo[1], A.hi[1], "given");
	}
	/* curves: their parameter, points and bounding box */
	if (has_curve) {
		double bb[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY};
		for (k = 0; k < O.ns; k++) {
			struct ser *s = &O.s[k];
			int i;
			if (s->kind != S_PARAM && s->kind != S_POLAR)
				continue;
			curve_trange(k);
			sample_curve(k);
			for (i = 0; i < s->cn; i++)
				if (!isnan(s->cx[i])) {
					if (s->cx[i] < bb[0]) bb[0] = s->cx[i];
					if (s->cx[i] > bb[1]) bb[1] = s->cx[i];
					if (s->cy[i] < bb[2]) bb[2] = s->cy[i];
					if (s->cy[i] > bb[3]) bb[3] = s->cy[i];
				}
		}
		if (!xset && bb[0] <= bb[1]) {
			double m = (bb[1] - bb[0]) * 0.06 + 1e-9;
			A.lo[0] = bb[0] - m;
			A.hi[0] = bb[1] + m;
			xset = 2;
		}
		if (!yset && bb[2] <= bb[3]) {
			double m = (bb[3] - bb[2]) * 0.06 + 1e-9;
			A.lo[1] = bb[2] - m;
			A.hi[1] = bb[3] + m;
			yset = 2;
		}
	}
	/* implicit curves, regions and fields: the plane */
	if (has_plane && (!xset || !yset)) {
		double bb[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY}, b1[4];
		int any = 0;
		for (k = 0; k < O.ns; k++) {
			if (O.s[k].kind != S_IMPL && !(O.s[k].kind == S_REGION &&
			    O.s[k].bound))
				continue;
			if (O.s[k].kind == S_REGION) {
				struct ev save = EV[k][0];
				ev_open(&EV[k][0], &O.s[k].f[0], O.s[k].bound);
				any |= impl_bbox(k, b1);
				ev_close(&EV[k][0]);
				EV[k][0] = save;
				if (b1[0] > b1[1])
					continue;
			} else if (!impl_bbox(k, b1))
				continue;
			else
				any = 1;
			if (b1[0] < bb[0]) bb[0] = b1[0];
			if (b1[1] > bb[1]) bb[1] = b1[1];
			if (b1[2] < bb[2]) bb[2] = b1[2];
			if (b1[3] > bb[3]) bb[3] = b1[3];
		}
		if (any) {
			double mx = fmax((bb[1] - bb[0]) * 0.15, 0.5), my = fmax((bb[3] -
			    bb[2]) * 0.15, 0.5);
			if (bb[1] - bb[0] > 19) {	/* it runs off: the middle */
				bb[0] = -5;
				bb[1] = 5;
				mx = 0;
			}
			if (bb[3] - bb[2] > 19) {
				bb[2] = -5;
				bb[3] = 5;
				my = 0;
			}
			if (!xset) {
				A.lo[0] = bb[0] - mx;
				A.hi[0] = bb[1] + mx;
				range_line(sym_str(xn), A.lo[0], A.hi[0],
				    "around the curve found in [-10, 10]");
				xset = 1;
			}
			if (!yset) {
				A.lo[1] = bb[2] - my;
				A.hi[1] = bb[3] + my;
				range_line(sym_str(yn), A.lo[1], A.hi[1],
				    "around the curve found in [-10, 10]");
				yset = 1;
			}
			eq = 1;
		} else {
			if (!xset) {
				A.lo[0] = -3;
				A.hi[0] = 3;
				range_line(sym_str(xn), A.lo[0], A.hi[0], "default");
				xset = 1;
			}
			if (!yset) {
				A.lo[1] = -3;
				A.hi[1] = 3;
				range_line(sym_str(yn), A.lo[1], A.hi[1], "default");
				yset = 1;
			}
		}
	}
	if (!xset) {
		A.lo[0] = -5;
		A.hi[0] = 5;
	}
	if (A.lg[0] && A.lo[0] <= 0) {
		/*
		 * the positive part: two decades below the smallest positive
		 * zero or extremum (sampled on a log grid down to 1e-6 of the
		 * top), else three decades below the top decade
		 */
		double top = A.hi[0] > 0 ? pow(10, ceil(log10(A.hi[0]) - 1e-9)) :
		    1, pf = INFINITY;
		int i;
		if (A.hi[0] <= 0)
			A.hi[0] = 1;
		for (k = 0; k < O.ns; k++) {
			struct ser *s = &O.s[k];
			double pv = NAN, pd = NAN;
			if (s->kind != S_FX || s->sx < 0)
				continue;
			for (i = 0; i <= 400; i++) {
				double x = top * pow(10, -6 + 6.0 * i / 400), v, d;
				if (x > A.hi[0])
					break;
				if (!f1(&EV[k][0], s->fix[0], s->sx, x, &v) ||
				    !isfinite(v)) {
					pv = pd = NAN;
					continue;
				}
				d = v - pv;
				if (!isnan(pv) && ((v == 0) || (pv < 0) != (v < 0) ||
				    (!isnan(pd) && d != 0 && pd != 0 && (d < 0) !=
				    (pd < 0)))) {
					if (x < pf)
						pf = x;
					break;
				}
				if (!isnan(pv) && d != 0)
					pd = d;
				pv = v;
			}
		}
		if (isfinite(pf) && pf > 0) {
			A.lo[0] = pow(10, floor(log10(pf)) - 2);
			if (A.lo[0] > top / 100)
				A.lo[0] = top / 100;
			printf("  note: log x axis: from %s, two decades below the "
			    "first positive feature (near %s)\n", g4(A.lo[0]),
			    g4(pf));
		} else {
			A.lo[0] = top / 1000;
			printf("  note: log x axis: from %s, three decades below the "
			    "top\n", g4(A.lo[0]));
		}
		if (A.hi[0] <= A.lo[0])
			A.hi[0] = A.lo[0] * 1000;
	}
	/* tangent and derivative series need nothing more here */
	if (!yset && has_fx) {
		/* the values of the curves, over the animation too */
		int nfr = an.n < 8 ? an.n : 8, fr, cap = 0, n = 0, pole = 0;
		double *vals = NULL, vmin, vmax;
		for (fr = 0; fr < nfr; fr++) {
			if (av)
				set_fix(av, anim_val(&an, fr * (an.n - 1) / (nfr > 1 ?
				    nfr - 1 : 1)));
			for (k = 0; k < O.ns; k++) {
				struct ser *s = &O.s[k];
				double xs[NSAMP], ys[NSAMP], u0, u1;
				uint8_t ok[NSAMP];
				int i;
				if (s->kind != S_FX && s->kind != S_DERIV)
					continue;
				u0 = ax_u(&A, 0, A.lo[0]);
				u1 = ax_u(&A, 0, A.hi[0]);
				for (i = 0; i < NSAMP; i++) {
					double u = u0 + (u1 - u0) * i / (NSAMP - 1);
					xs[i] = A.lg[0] ? pow(10, u) : u;
					if (is_intvar(s))
						xs[i] = floor(xs[i] + 0.5);
				}
				f1n(&EV[k][0], s->fix[0], s->sx, NSAMP, xs, ys, ok);
				if (n + NSAMP > cap) {
					cap = (n + NSAMP) * 2;
					vals = xrealloc(vals, cap * sizeof *vals);
				}
				for (i = 0; i < NSAMP; i++) {
					if (ok[i] == 1)
						vals[n++] = ys[i];
					if (ok[i] == 2)
						pole = 1;
					if (!pole && i > 1 && ok[i] == 1 && ok[i - 1] == 0 &&
					    ok[i - 2] == 1 && !is_intvar(s)) {
						double at;
						if (jump_kind(&EV[k][0], s->fix[0], s->sx, xs[i - 2],
						    ys[i - 2], xs[i], ys[i], &at) == 2)
							pole = 1;
					}
					if (!pole && i > 0 && ok[i] == 1 && ok[i - 1] == 1 &&
					    ((ys[i] > 0 && ys[i - 1] < 0) ||
					    (ys[i] < 0 && ys[i - 1] > 0)) && !is_intvar(s)) {
						double at;
						if (jump_kind(&EV[k][0], s->fix[0], s->sx, xs[i - 1],
						    ys[i - 1], xs[i], ys[i], &at) == 2)
							pole = 1;
					}
				}
			}
		}
		for (k = 0; k < O.ns; k++)
			if (O.s[k].kind == S_TANGENT && n < cap)
				vals[n++] = O.s[k].tm * O.s[k].ta + O.s[k].tc;
		if (av)
			set_fix(av, anim_val(&an, 0));
		if (vrange(vals, n, pole, A.lg[1], &A.lo[1], &A.hi[1], &vmin,
		    &vmax))
			printf("  %s in [%s, %s]: the bulk of the values (they reach "
			    "[%s, %s])\n", sym_str(yn), g4(A.lo[1]), g4(A.hi[1]),
			    g4(vmin), g4(vmax));
		else
			printf("  %s in [%s, %s]: the values\n", sym_str(yn),
			    g4(A.lo[1]), g4(A.hi[1]));
		free(vals);
		yset = 1;
	}
	if (!yset) {
		if (has_slope) {
			struct ser *s = NULL;
			for (k = 0; k < O.ns; k++)
				if (O.s[k].kind == S_SLOPE)
					s = &O.s[k];
			if (s->has_ic) {
				double ts[401], yv[401], vmin, vmax;
				int m = rk4(&EV[s - O.s][0], s, A.lo[0], s->ic_y, A.hi[0], 400,
				    ts, yv, -1e6, 1e6);
				vrange(yv, m, 0, 0, &A.lo[1], &A.hi[1], &vmin, &vmax);
				if (A.hi[1] - A.lo[1] < 2) {
					double c = (A.lo[1] + A.hi[1]) / 2;
					A.lo[1] = c - 1;
					A.hi[1] = c + 1;
				} else {
					double w = A.hi[1] - A.lo[1];
					A.lo[1] -= w * 0.25;
					A.hi[1] += w * 0.25;
				}
				range_line(sym_str(yn), A.lo[1], A.hi[1],
				    "around the solution through the initial value");
			} else {
				A.lo[1] = -3;
				A.hi[1] = 3;
				range_line(sym_str(yn), A.lo[1], A.hi[1], "default");
			}
		} else {
			A.lo[1] = -5;
			A.hi[1] = 5;
		}
	}
	if (A.lg[1] && A.lo[1] <= 0) {
		A.lo[1] = A.hi[1] > 0 ? A.hi[1] * 1e-6 : 1e-3;
		if (A.hi[1] <= A.lo[1])
			A.hi[1] = A.lo[1] * 1000;
	}
	/* curves and implicit curves: one unit is as long on both axes */
	for (k = 0; k < O.ns; k++)
		if (O.s[k].kind == S_IMPL)
			eq = 1;
	if (!has_slope && (has_curve || eq) && !A.lg[0] && !A.lg[1] &&
	    !(rng_find(xn) && rng_find(yn))) {
		W = 600;
		H = 540;
		equal_aspect(&A, W, H);
		if (xset == 2 || yset == 2)
			printf("  %s in [%s, %s], %s in [%s, %s]: the curve, the same "
			    "scale on both axes\n", sym_str(xn), g4(A.lo[0]), g4(A.hi[0]),
			    sym_str(yn), g4(A.lo[1]), g4(A.hi[1]));
	}
	{
		/* integer ticks when every function has an integer input (output) */
		int nfx = 0, ix = 1, iy = 1;
		for (k = 0; k < O.ns; k++) {
			if (O.s[k].kind != S_FX || O.s[k].sx < 0) {
				ix = iy = 0;
				continue;
			}
			nfx++;
			ix &= is_intvar(&O.s[k]);
			iy &= !TY_FP(N.ty[O.s[k].f[0].body]);
		}
		tick_int[0] = nfx && ix && !A.lg[0];
		tick_int[1] = nfx && iy && !A.lg[1];
	}
	join_labels(title, sizeof title);
	if (O.mode == 2 && O.nat)
		snprintf(title + strlen(title), sizeof title - strlen(title),
		    " near %s = %s", sym_str(xn), g4(O.at[0]));
	cv_open(W, H);
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, NPAL);
	for (f = 0; f < an.n; f++) {
		memset(C.px, P_BG, (size_t)W * H);
		first_frame = f == 0;
		if (av) {
			set_fix(av, anim_val(&an, f));
			snprintf(tt, sizeof tt, "%s   (%s = %s)", title, sym_str(av),
			    g4(anim_val(&an, f)));
		} else
			snprintf(tt, sizeof tt, "%s", title);
		chart2d(&A, 0, 0, W, H, tt, sym_str(xn), sym_str(yn));
		gif_frame(&g, C.px, an.n > 1 ? delay_cs(f, an.fps) : 0);
	}
	cv_close();
	if (av)
		printf("  frames over %s in [%s, %s]\n", sym_str(av), g4(an.lo),
		    g4(an.hi));
	finish_gif(&g, W, H, an.n, an.fps);
}

/* ---- 3-D and heat-map views of z = f(x, y) ---- */

#define DEG (M_PI / 180)

static void
surf_frame(int si, const struct box3 *B, double az, int W, int H,
    const char *title, int wire, int *und, int *clip)
{
	struct v3 V;

	memset(C.px, P_BG, (size_t)W * H);
	V.az = az;
	V.el = 27 * DEG;
	V.h = 0.62;
	v3_fit(&V, 46, 36, W - 104, H - 34);
	cv_clip(0, 0, W - 1, H - 1);
	panes(&V, B, 1);
	surface(&O.s[si], si, &V, B, wire, und, clip);
	colorbar(W - 70, 64, H - 70, B->lo[2], B->hi[2], 4, B->name[2]);
	title_at(W / 2, 8, title, W - 20);
}

static void
heat_frame(int si, struct ax2 *A, double zlo, double zhi, int view, int W,
    int H, const char *title, const char *xn, const char *yn, const char *zn,
    int blk, int *und)
{
	memset(C.px, P_BG, (size_t)W * H);
	layout2(A, 0, 0, W, H, 76, 0);
	cv_clip(A->L + 1, A->T + 1, A->R - 1, A->B - 1);
	heat_into(&O.s[si], si, A, zlo, zhi, view, blk, und);
	cv_clip(0, 0, W - 1, H - 1);
	axes2(A, xn, yn, 0, 1, 1);
	colorbar(A->R + 20, A->T + 16, A->B, zlo, zhi, view == V_CONTOUR ? 5 : 4,
	    zn);
	title_at(W / 2, 8, title, W - 20);
}

/* the value range of slot-pair plane over frames/panels */
static void
zrange(int si, const struct ax2 *A, const int *vs, const double *vv, int nv,
    double *zlo, double *zhi, const char *zn)
{
	enum { NG = 49 };
	struct ser *s = &O.s[si];
	struct rng *r = rng_find(sym_cstr(zn));
	double *vals = xmalloc((size_t)NG * NG * (nv ? nv : 1) * sizeof *vals);
	double in[MAXP], vmin, vmax, save[MAXP];
	int n = 0, i, j, q, pole = 0;

	if (r) {
		*zlo = r->lo;
		*zhi = r->hi;
		range_line(zn, *zlo, *zhi, "given");
		free(vals);
		return;
	}
	memcpy(save, s->fix[0], sizeof save);
	for (q = 0; q < (nv ? nv : 1); q++) {
		if (nv)
			for (j = 0; j < 2; j++)
				if (vs[2 * q + j] >= 0)
					s->fix[0][vs[2 * q + j]] = vv[2 * q + j];
		for (j = 0; j < NG; j++)
			for (i = 0; i < NG; i++) {
				double v;
				int ok;
				plane_in(s, 0, A->lo[0] + (A->hi[0] - A->lo[0]) * i / (NG - 1),
				    A->lo[1] + (A->hi[1] - A->lo[1]) * j / (NG - 1), in);
				ok = ev1(&EV[si][0], in, &v);
				if (ok == 1)
					vals[n++] = v;
				else if (ok == 2)
					pole = 1;
			}
	}
	memcpy(s->fix[0], save, sizeof save);
	if (vrange(vals, n, pole, 0, zlo, zhi, &vmin, &vmax))
		printf("  %s in [%s, %s]: the bulk of the values (they reach [%s, "
		    "%s])\n", zn, g4(*zlo), g4(*zhi), g4(vmin), g4(vmax));
	else
		printf("  %s in [%s, %s]: the values\n", zn, g4(*zlo), g4(*zhi));
	free(vals);
}

static void
plot_surf(int si)
{
	struct ser *s = &O.s[si];
	struct ax2 A;
	struct box3 B;
	struct gif g;
	int W = 640, H = 500, view = O.view, f, nfr = 1, fps = 12, und = 0,
	    clip = 0;
	const char *xn = sym_str(s->f[0].p[s->sx].name);
	const char *yn = sym_str(s->f[0].p[s->sy].name);
	const char *zn = s->yname ? sym_str(s->yname) : "z";
	char title[300];

	if (view == V_AUTO)
		view = V_SURF;
	if (view == V_FLAT)
		view = V_HEAT;
	printf("%s: %s = f(%s, %s)\n", view == V_CONTOUR ? "contour plot" :
	    view == V_HEAT ? "heat map" : view == V_WIRE ? "wireframe surface" :
	    "surface", zn, xn, yn);
	memset(&A, 0, sizeof A);
	base_of(&s->f[0], s->fix[0], 0);
	slot_range(si, 0, s->sx, O.nat ? 0 : -1, &A.lo[0], &A.hi[0], 0, NULL);
	slot_range(si, 0, s->sy, O.nat > 1 ? 1 : -1, &A.lo[1], &A.hi[1], 0, NULL);
	zrange(si, &A, NULL, NULL, 0, &B.lo[2], &B.hi[2], zn);
	tick_int[0] = !TY_FP(s->f[0].p[s->sx].ty);
	tick_int[1] = !TY_FP(s->f[0].p[s->sy].ty);
	tick_int[2] = !TY_FP(N.ty[s->f[0].body]);
	B.lo[0] = A.lo[0];
	B.hi[0] = A.hi[0];
	B.lo[1] = A.lo[1];
	B.hi[1] = A.hi[1];
	B.name[0] = xn;
	B.name[1] = yn;
	B.name[2] = zn;
	snprintf(title, sizeof title, "%s%s%s", s->yname ? "" : "z = ", s->label,
	    O.mode == 2 && O.nat ? " (neighbourhood)" : "");
	if (O.rotate && (view == V_SURF || view == V_WIRE)) {
		nfr = O.frames > 0 ? O.frames : O.seconds > 0 ? (int)(O.seconds *
		    (O.fps > 0 ? O.fps : fps)) : 36;
		fps = O.fps > 0 ? O.fps : 12;
		if (nfr < 2)
			nfr = 2;
		if (nfr > 180)
			nfr = 180;
		W = 560;
		H = 440;
	}
	cv_open(W, H);
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, NPAL);
	for (f = 0; f < nfr; f++) {
		if (view == V_SURF || view == V_WIRE)
			surf_frame(si, &B, (-55 + 360.0 * f / nfr) * DEG, W, H, title,
			    view == V_WIRE, &und, &clip);
		else
			heat_frame(si, &A, B.lo[2], B.hi[2], view, W, H, title, xn, yn,
			    zn, 1, &und);
		gif_frame(&g, C.px, nfr > 1 ? delay_cs(f, fps) : 0);
		if (f == 0) {
			if (view == V_SURF || view == V_WIRE) {
				sb_printf(&SUM, "  %dx%d grid", G3 + 1, G3 + 1);
				if (und)
					sb_printf(&SUM, ", undefined at %d points", und);
				if (clip)
					sb_printf(&SUM, ", %d points outside the %s range "
					    "(holes)", clip, zn);
				sb_printf(&SUM, "\n");
			} else if (und)
				sb_printf(&SUM, "  undefined on %d pixels (grey)\n", und);
		}
	}
	cv_close();
	if (nfr > 1)
		printf("  rotating once around the vertical axis\n");
	finish_gif(&g, W, H, nfr, fps);
}

/* ---- functions of three or more variables ---- */

static void
plot_parallel(int si)
{
	enum { NPT = 400 };
	struct ser *s = &O.s[si];
	struct func *f = &s->f[0];
	int np = f->np, W = 120 + np * 120, H = 440, k, i, l, na = np + 1;
	double lo[MAXP + 1], hi[MAXP + 1], *in, *v, vmin, vmax;
	uint8_t *ok;
	uint64_t rng = 0x9e3779b97f4a7c15ull;
	struct gif g;
	int ax[MAXP + 1], top = 52, bot = H - 40, cnt = 0;
	char title[300];

	if (W > 900)
		W = 900;
	printf("parallel coordinates: %s\n", s->label);
	for (k = 0; k < np; k++)
		slot_range(si, 0, k, -1, &lo[k], &hi[k], 0, NULL);
	in = xmalloc((size_t)NPT * np * sizeof *in);
	v = xmalloc(NPT * sizeof *v);
	ok = xmalloc(NPT);
	for (i = 0; i < NPT; i++)
		for (k = 0; k < np; k++) {
			double u;
			rng ^= rng << 13;
			rng ^= rng >> 7;
			rng ^= rng << 17;
			u = (double)(rng >> 11) / 9007199254740992.0;
			in[(size_t)i * np + k] = lo[k] + (hi[k] - lo[k]) * u;
			if (!TY_FP(f->p[k].ty))
				in[(size_t)i * np + k] = floor(in[(size_t)i * np + k] + 0.5);
		}
	ev_run(&EV[si][0], NPT, in, v, ok);
	{
		double *vals = xmalloc(NPT * sizeof *vals);
		for (i = 0; i < NPT; i++)
			if (ok[i] == 1)
				vals[cnt++] = v[i];
		vrange(vals, cnt, 0, 0, &lo[np], &hi[np], &vmin, &vmax);
		free(vals);
	}
	cv_open(W, H);
	memset(C.px, P_BG, (size_t)W * H);
	for (k = 0; k < na; k++)
		ax[k] = 60 + (W - 120) * k / (na - 1);
	for (l = 0; l < 8; l++) {
		int t[4], any = 0;
		for (i = 0; i < NPT; i++) {
			double c;
			if (ok[i] != 1)
				continue;
			c = (v[i] - lo[np]) / (hi[np] - lo[np]);
			if ((int)(fmin(fmax(c, 0), 0.999) * 8) != l)
				continue;
			for (k = 0; k < na - 1; k++) {
				double a = k < np ? in[(size_t)i * np + k] : v[i];
				double b = k + 1 < np ? in[(size_t)i * np + k + 1] : v[i];
				double ya = bot - (a - lo[k]) / (hi[k] - lo[k]) * (bot - top);
				double yb = bot - (b - lo[k + 1]) / (hi[k + 1] - lo[k + 1]) *
				    (bot - top);
				cov_seg(ax[k], ya, ax[k + 1], yb, 1.0);
			}
			any = 1;
		}
		if (!any)
			continue;
		t[0] = t[1] = cmap(0.05 + 0.9 * l / 7.0, 3);
		t[2] = t[3] = cmap(0.05 + 0.9 * l / 7.0, 5);
		cov_put(t);
	}
	for (k = 0; k < na; k++) {
		const char *nm = k < np ? sym_str(f->p[k].name) : "f";
		fill(ax[k], top, 1, bot - top + 1, P_FRAME);
		text(ax[k] - textw(nm) / 2, top - 26, nm, P_TEXT);
		text_bg(ax[k] - textw(g4(hi[k])) / 2, top - 13, g4(hi[k]), P_DIM,
		    P_BG);
		text_bg(ax[k] - textw(g4(lo[k])) / 2, bot + 5, g4(lo[k]), P_DIM, P_BG);
	}
	snprintf(title, sizeof title, "%s", s->label);
	title_at(W / 2, 8, title, W - 20);
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, NPAL);
	gif_frame(&g, C.px, 0);
	cv_close();
	sb_printf(&SUM, "  %d random points, %d defined, f in [%s, %s] (colour)\n",
	    NPT, cnt, g4(vmin), g4(vmax));
	free(in);
	free(v);
	free(ok);
	finish_gif(&g, W, H, 1, 1);
}

static void
plot_nd(int si)
{
	struct ser *s = &O.s[si];
	struct func *f = &s->f[0];
	int np = f->np, rem[MAXP], nrem = 0, ps = -1, as = -1, k, fr, view = O.view;
	int W, H, und = 0;
	struct ax2 A;
	struct anim an;
	struct gif g;
	double plo = 0, phi = 0, zlo, zhi;
	const char *xn, *yn;
	char title[400], tt[480];

	if (O.parallel) {
		plot_parallel(si);
		return;
	}
	for (k = 0; k < np; k++)
		if (k != s->sx && k != s->sy)
			rem[nrem++] = k;
	if (O.animvar)
		as = slot_named(f, sym_str(O.animvar));
	if (np == 3 && O.slices && as < 0)
		ps = rem[0];
	else {
		if (np >= 4 || O.slices) {
			for (k = 0; k < nrem; k++)
				if (rem[k] != as && !strcmp(sym_str(f->p[rem[k]].name), "z"))
					ps = rem[k];
			for (k = 0; k < nrem && ps < 0; k++)
				if (rem[k] != as)
					ps = rem[k];
		}
		if (as < 0) {
			for (k = 0; k < nrem; k++)
				if (rem[k] != ps && (!strcmp(sym_str(f->p[rem[k]].name), "t") ||
				    !strcmp(sym_str(f->p[rem[k]].name), "w")))
					as = rem[k];
			for (k = 0; k < nrem && as < 0; k++)
				if (rem[k] != ps)
					as = rem[k];
		}
	}
	if (view == V_AUTO)
		view = O.mode == 1 ? V_SURF : V_HEAT;
	if (view == V_FLAT)
		view = V_HEAT;
	xn = sym_str(f->p[s->sx].name);
	yn = sym_str(f->p[s->sy].name);
	printf("%s of %d variables: %s over %s, %s", s->label, np,
	    view == V_SURF ? "a surface" : view == V_CONTOUR ? "contours" :
	    view == V_WIRE ? "a wireframe" : "a heat map", xn, yn);
	if (ps >= 0)
		printf("; small multiples over %s", sym_str(f->p[ps].name));
	if (as >= 0)
		printf("; frames over %s", sym_str(f->p[as].name));
	printf("\n");
	if (view == V_WIRE || (view == V_SURF && ps >= 0))
		view = view == V_WIRE && ps < 0 ? V_WIRE : ps >= 0 ? V_HEAT : view;
	memset(&A, 0, sizeof A);
	base_of(f, s->fix[0], 0);
	slot_range(si, 0, s->sx, O.nat ? 0 : -1, &A.lo[0], &A.hi[0], 0, NULL);
	slot_range(si, 0, s->sy, O.nat > 1 ? 1 : -1, &A.lo[1], &A.hi[1], 0, NULL);
	if (ps >= 0)
		slot_range(si, 0, ps, -1, &plo, &phi, 0, NULL);
	tick_int[0] = !TY_FP(f->p[s->sx].ty);
	tick_int[1] = !TY_FP(f->p[s->sy].ty);
	tick_int[2] = !TY_FP(N.ty[f->body]);
	for (k = 0; k < nrem; k++)
		if (rem[k] != ps && rem[k] != as)
			printf("  note: %s fixed at %s (the middle of its range; give "
			    "%s in a..b to change)\n", sym_str(f->p[rem[k]].name),
			    g4(s->fix[0][rem[k]]), sym_str(f->p[rem[k]].name));
	anim_setup(&an, as >= 0 ? f->p[as].name : 0, ps >= 0 ? 16 : 24, 8);
	{
		int vs[2 * 9 * 6], nv = 0, a, b;
		double vv[2 * 9 * 6];
		for (a = 0; a < (ps >= 0 ? 9 : 1); a++)
			for (b = 0; b < (as >= 0 ? 6 : 1); b++) {
				vs[2 * nv] = ps;
				vv[2 * nv] = plo + (phi - plo) * a / 8;
				vs[2 * nv + 1] = as;
				vv[2 * nv + 1] = as >= 0 ? anim_val(&an, b * (an.n - 1) / 5) : 0;
				nv++;
			}
		zrange(si, &A, vs, vv, nv, &zlo, &zhi, "f");
	}
	snprintf(title, sizeof title, "%s", s->label);
	if (ps >= 0) {
		W = 780;
		H = 640;
	} else if (view == V_SURF || view == V_WIRE) {
		W = 560;
		H = 440;
	} else {
		W = 600;
		H = 460;
	}
	cv_open(W, H);
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, NPAL);
	for (fr = 0; fr < an.n; fr++) {
		if (as >= 0)
			s->fix[0][as] = anim_val(&an, fr);
		if (as >= 0)
			snprintf(tt, sizeof tt, "%s   (%s = %s)", title,
			    sym_str(f->p[as].name), g4(s->fix[0][as]));
		else
			snprintf(tt, sizeof tt, "%s", title);
		if (ps < 0) {
			if (view == V_SURF || view == V_WIRE) {
				struct box3 B;
				B.lo[0] = A.lo[0];
				B.hi[0] = A.hi[0];
				B.lo[1] = A.lo[1];
				B.hi[1] = A.hi[1];
				B.lo[2] = zlo;
				B.hi[2] = zhi;
				B.name[0] = xn;
				B.name[1] = yn;
				B.name[2] = "f";
				surf_frame(si, &B, -55 * DEG, W, H, tt, view == V_WIRE, &und,
				    NULL);
			} else
				heat_frame(si, &A, zlo, zhi, view, W, H, tt, xn, yn, "f",
				    an.n > 1 ? 2 : 1, &und);
		} else {
			int pi, pw = (W - 90) / 3, ph = (H - 34) / 3;
			memset(C.px, P_BG, (size_t)W * H);
			for (pi = 0; pi < 9; pi++) {
				struct ax2 P = A;
				int px0 = 4 + (pi % 3) * pw, py0 = 30 + (pi / 3) * ph;
				char pt[64];
				s->fix[0][ps] = plo + (phi - plo) * pi / 8;
				layout2(&P, px0, py0, pw, ph, 0, 1);
				P.L = px0 + 46;
				cv_clip(P.L + 1, P.T + 1, P.R - 1, P.B - 1);
				heat_into(s, si, &P, zlo, zhi, view, 2, NULL);
				cv_clip(0, 0, W - 1, H - 1);
				axes2(&P, pi / 3 == 2 ? xn : NULL, NULL, 1, pi / 3 == 2,
				    pi % 3 == 0);
				{
					char pf[24];
					snprintf(pt, sizeof pt, "%s = %s", sym_str(f->p[ps].name),
					    pi_frac(s->fix[0][ps], pf, sizeof pf) ? pf :
					    g4(s->fix[0][ps]));
				}
				text((P.L + P.R) / 2 - textw(pt) / 2, P.T - 12, pt, P_TEXT);
				if (pi % 3 == 0)
					text(px0 + 2, (P.T + P.B) / 2 - 5, yn, P_TEXT);
			}
			colorbar(W - 70, 60, H - 60, zlo, zhi, view == V_CONTOUR ? 5 : 4,
			    "f");
			title_at(W / 2, 8, tt, W - 20);
		}
		gif_frame(&g, C.px, an.n > 1 ? delay_cs(fr, an.fps) : 0);
	}
	cv_close();
	if (as >= 0)
		printf("  frames over %s in [%s, %s]\n", sym_str(f->p[as].name),
		    g4(an.lo), g4(an.hi));
	finish_gif(&g, W, H, an.n, an.fps);
}

/* ---- space curves ---- */

static void
plot_space(void)
{
	struct box3 B;
	struct gif g;
	int k, i, c, W = 560, H = 460, nfr = 1, fps = 12, fr;
	char title[300];

	printf("space curve%s:\n", O.ns > 1 ? "s" : "");
	for (c = 0; c < 3; c++) {
		B.lo[c] = INFINITY;
		B.hi[c] = -INFINITY;
	}
	for (k = 0; k < O.ns; k++) {
		struct ser *s = &O.s[k];
		base_of(&s->f[0], s->fix[0], 0);
		base_of(&s->f[1], s->fix[1], 0);
		base_of(&s->f[2], s->fix[2], 0);
		curve_trange(k);
		sample_curve(k);
		for (i = 0; i < s->cn; i++) {
			double p[3];
			if (isnan(s->cx[i]))
				continue;
			p[0] = s->cx[i];
			p[1] = s->cy[i];
			p[2] = s->cy[s->cn + i];
			for (c = 0; c < 3; c++) {
				if (p[c] < B.lo[c]) B.lo[c] = p[c];
				if (p[c] > B.hi[c]) B.hi[c] = p[c];
			}
		}
		sb_printf(&SUM, "  %s: %s in [%s, %s], %d samples\n", s->label,
		    sym_str(s->xname), gpi(s->t0), gpi(s->t1), s->cn);
	}
	for (c = 0; c < 3; c++) {
		double m;
		if (!(B.lo[c] <= B.hi[c])) {
			B.lo[c] = -1;
			B.hi[c] = 1;
		}
		m = (B.hi[c] - B.lo[c]) * 0.05;
		if (m == 0)
			m = 1;
		B.lo[c] -= m;
		B.hi[c] += m;
	}
	printf("  box x in [%s, %s], y in [%s, %s], z in [%s, %s]\n", g4(B.lo[0]),
	    g4(B.hi[0]), g4(B.lo[1]), g4(B.hi[1]), g4(B.lo[2]), g4(B.hi[2]));
	B.name[0] = "x";
	B.name[1] = "y";
	B.name[2] = "z";
	join_labels(title, sizeof title);
	if (O.rotate) {
		fps = O.fps > 0 ? O.fps : 12;
		nfr = O.frames > 0 ? O.frames : 36;
	}
	cv_open(W, H);
	gif_begin(&g, W, H, (const unsigned char (*)[3])pal, NPAL);
	for (fr = 0; fr < nfr; fr++) {
		struct v3 V;
		memset(C.px, P_BG, (size_t)W * H);
		V.az = (-55 + 360.0 * fr / nfr) * DEG;
		V.el = 24 * DEG;
		V.h = 0.75;
		v3_fit(&V, 46, 36, W - 40, H - 34);
		cv_clip(0, 0, W - 1, H - 1);
		panes(&V, &B, 1);
		for (k = 0; k < O.ns; k++)
			space_curve(&O.s[k], &V, &B);
		title_at(W / 2, 8, title, W - 20);
		gif_frame(&g, C.px, nfr > 1 ? delay_cs(fr, fps) : 0);
	}
	cv_close();
	finish_gif(&g, W, H, nfr, fps);
}

/* ---- the command ---- */

static int ev_opened[MAXSER][3];

static void
plot_free(void)
{
	int k, c;

	for (k = 0; k < MAXSER; k++) {
		for (c = 0; c < 3; c++)
			if (ev_opened[k][c]) {
				ev_close(&EV[k][c]);
				ev_opened[k][c] = 0;
			}
		free(O.s[k].cx);
		free(O.s[k].cy);
		O.s[k].cx = O.s[k].cy = NULL;
	}
	if (C.px)
		cv_close();
	sb_clear(&SUM);
}

/* derivative and tangent series for the curves y = f(x) */
static void
add_extras(void)
{
	int n0 = O.ns, k, tangent_done = 0;

	for (k = 0; k < n0; k++) {
		struct ser *s = &O.s[k], *d;
		struct func g;
		if (s->kind != S_FX || s->f[0].np != 1)
			continue;
		if (O.deriv) {
			if (!deriv_func(&s->f[0], &g))
				fail("plot: no derivative of %s", s->label);
			if (O.ns == MAXSER)
				fail("plot: too many functions with their derivatives");
			d = &O.s[O.ns];
			memset(d, 0, sizeof *d);
			d->kind = S_DERIV;
			d->f[0] = g;
			d->nf = 1;
			d->sx = 0;
			d->sy = -1;
			d->col = O.ns;
			snprintf(d->label, sizeof d->label, "d/d%s: ",
			    sym_str(g.p[0].name));
			ascii_of(d->label + strlen(d->label), sizeof d->label -
			    strlen(d->label), node_str(g.body));
			O.ns++;
		}
		if (O.tangent && !tangent_done) {
			struct ev e;
			double a = O.tan_at, fa, m, in[MAXP];
			int ok;
			tangent_done = 1;
			ev_open(&e, &s->f[0], s->f[0].body);
			in[0] = a;
			ok = ev1(&e, in, &fa);
			ev_close(&e);
			if (ok != 1)
				fail("plot: %s is not defined at %s = %s", s->label,
				    sym_str(s->f[0].p[0].name), g4(a));
			if (deriv_func(&s->f[0], &g)) {
				ev_open(&e, &g, g.body);
				ok = ev1(&e, in, &m);
				ev_close(&e);
			} else
				ok = 0;
			if (ok != 1) {	/* a central difference */
				double h = 1e-6 * (1 + fabs(a)), l, r;
				ev_open(&e, &s->f[0], s->f[0].body);
				in[0] = a - h;
				ok = ev1(&e, in, &l) == 1;
				in[0] = a + h;
				ok &= ev1(&e, in, &r) == 1;
				ev_close(&e);
				if (!ok)
					fail("plot: %s has no tangent at %s", s->label, g4(a));
				m = (r - l) / (2 * h);
			}
			if (O.ns == MAXSER)
				fail("plot: too many functions with the tangent");
			d = &O.s[O.ns];
			memset(d, 0, sizeof *d);
			d->kind = S_TANGENT;
			d->f[0] = s->f[0];
			d->nf = 1;
			d->sx = 0;
			d->sy = -1;
			d->col = O.ns;
			d->ta = a;
			d->tm = m;
			d->tc = fa - m * a;
			snprintf(d->label, sizeof d->label, "tangent at %s = %s: y = %s*%s "
			    "%s %s", sym_str(s->f[0].p[0].name), g4(a), g4(m),
			    sym_str(s->f[0].p[0].name), d->tc < 0 ? "-" : "+",
			    g4(fabs(d->tc)));
			O.ns++;
		}
	}
	if (O.tangent && !tangent_done)
		fail("plot: a tangent needs a function of one variable");
}

void
cmd_plot(struct P *p, int mode)
{
	jmp_buf saved;
	int fa, fb, pe, i, k, c, nsurf = 0, nnd = 0, nspace = 0, n2 = 0, si = -1;
	uint32_t av = 0;

	plot_free();
	memset(&O, 0, sizeof O);
	O.mode = mode;
	memset(tick_int, 0, sizeof tick_int);
	nrc = 0;
	snprintf(O.name, sizeof O.name, "plot");
	if (!cmdp_rest(p, O.text, sizeof O.text, NULL))
		fail("%s what? e.g. %s", mode == 0 ? "plot" : mode == 1 ? "plot3d" :
		    "neighbourhood", mode == 2 ? "neighbourhood sin(x)/x" :
		    "plot sin x, cos x x in -pi..pi");
	tk_lex(O.text);
	fa = parse_opts(0, 1);
	for (fb = fa; fb < nt; fb++)
		if (fb > fa && T[fb].d == 0 && (optstart(fb) || (T[fb].k == 'n' &&
		    tw(fb + 1, "fps"))))
			break;
	pe = fb;
	/*
	 * "sin x in 0..1": the range's variable is also the argument of a
	 * function named just before it
	 */
	if (fb < nt && fb - 1 >= fa && T[fb - 1].k == 'w' && (tw(fb + 1, "in") ||
	    tw(fb + 1, "from")) && (fb - 1 == fa || T[fb - 2].k != 'w')) {
		static const char *const un[] = {"sin", "cos", "tan", "exp", "log",
			"ln", "sqrt", "abs", "floor", "ceil", "sign", "sec", "csc",
			"cot", "atan", "isqrt", "clz", "ctz", "popcount", "W0", "Wm1",
			"lambertw", "productlog", NULL};
		char w[64];
		int q;
		tk_str(fb - 1, fb, w, sizeof w);
		for (q = 0; un[q] && strcmp(un[q], w); q++)
			;
		if (un[q] || (func_find(sym_cstr(w)) && func_find(sym_cstr(w))->np))
			pe = fb + 1;
	}
	i = parse_opts(fb, 0);
	if (i < nt)
		fail("plot: I don't understand '%s'", O.text + T[i].a);
	for (k = fa, i = fa; k <= pe; k++)
		if (k == pe || (T[k].d == 0 && T[k].k == ',')) {
			add_piece(i, k);
			i = k + 1;
		}
	if (O.ns == 0)
		fail("plot what? (a function or an expression)");
	if (O.fps < 0 || O.frames < 0)
		fail("plot: fps and frames are positive");
	pal_init();
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp)) {
		plot_free();
		memcpy(fail_jmp, saved, sizeof saved);
		longjmp(fail_jmp, 1);
	}
	add_extras();
	for (k = 0; k < O.ns; k++)
		for (c = 0; c < O.s[k].nf; c++) {
			ev_open(&EV[k][c], &O.s[k].f[c], O.s[k].f[c].body);
			ev_opened[k][c] = 1;
		}
	/* an animation variable turns a surface into moving curves */
	if (O.anim) {
		for (k = 0; k < O.ns; k++) {
			struct ser *s = &O.s[k];
			int a = -1;
			if (s->kind != S_SURF && s->kind != S_FX)
				continue;
			if (O.animvar)
				a = slot_named(&s->f[0], sym_str(O.animvar));
			if (a < 0 && s->f[0].np == 2) {
				a = slot_named(&s->f[0], "t");
				if (a < 0)
					a = s->f[0].np - 1;
			}
			if (a < 0)
				continue;
			av = s->f[0].p[a].name;
			if (s->kind == S_SURF) {
				s->kind = S_FX;
				s->sx = a == 0 ? 1 : 0;
				s->sy = -1;
			}
		}
		if (!av && !O.animvar) {
			for (k = 0; k < O.ns; k++)
				if (O.s[k].kind == S_ND)
					break;
			if (k == O.ns)
				fail("plot: animate over what? (no function has a second "
				    "variable)");
		}
		if (O.animvar)
			av = O.animvar;
	}
	for (k = 0; k < O.ns; k++)
		switch (O.s[k].kind) {
		case S_SURF:
			nsurf++;
			si = k;
			break;
		case S_ND:
			nnd++;
			si = k;
			break;
		case S_SPACE:
			nspace++;
			break;
		default:
			n2++;
		}
	if (nsurf + nnd > 1 || ((nsurf || nnd) && (n2 || nspace)))
		fail("plot: one function of two or more variables at a time (curves "
		    "of one variable can share a plot)");
	if (nspace && n2)
		fail("plot: space curves and plane curves need separate plots");
	if (O.mode == 2)
		printf("neighbourhood: bounds from %s\n", O.nat ?
		    "the features near the point" : "declared ranges, else the "
		    "function's features");
	if (mode == 1 && !nsurf && !nnd && !nspace)
		printf("  note: plot3d of a function of one variable: a 2d plot\n");
	if (nnd) {
		plot_nd(si);
	} else if (nsurf) {
		plot_surf(si);
	} else if (nspace) {
		plot_space();
	} else {
		int nfx = 0;
		for (k = 0; k < O.ns; k++) {
			int kd = O.s[k].kind;
			nfx += kd == S_FX || kd == S_DERIV || kd == S_TANGENT;
		}
		if (O.ns == 1 && nfx)
			printf("plot of %s\n", O.s[0].label);
		else if (O.ns == 1)
			printf("%s: %s\n", O.s[0].kind == S_IMPL ? "implicit curve" :
			    O.s[0].kind == S_REGION ? "region" : O.s[0].kind == S_PARAM ?
			    "parametric curve" : O.s[0].kind == S_POLAR ? "polar curve" :
			    O.s[0].kind == S_VFIELD ? "vector field" : "slope field",
			    O.s[0].label);
		else
			printf("plot of %d functions\n", O.ns);
		plot_2d(av);
	}
	plot_free();
	memcpy(fail_jmp, saved, sizeof saved);
}

/* ---- the renderer, exported for viz.c (viz.h) ---- */

#include "viz.h"

void
pr_begin(int w, int h)
{
	pal_init();
	if (C.px)
		cv_close();
	cv_open(w, h);
	memset(C.px, P_BG, (size_t)w * h);
}

void
pr_end(void)
{
	if (C.px)
		cv_close();
}

void
pr_clear(void)
{
	size_t q;

	memset(C.px, P_BG, (size_t)C.W * C.H);
	for (q = 0; q < (size_t)C.W * C.H; q++)
		C.zb[q] = 1e30f;
	cv_clip(0, 0, C.W - 1, C.H - 1);
}

unsigned char *
pr_pixels(void)
{
	return C.px;
}

float *
pr_zbuf(void)
{
	return C.zb;
}

int
pr_palette(unsigned char out[256][3], int *ramp0, int *nramp)
{
	memcpy(out, pal, sizeof pal);
	*ramp0 = P_CM;		/* the colour map's entries are free to reuse */
	*nramp = NCM * NSH;
	return NPAL;
}

int
pr_ink(int role)
{
	static const int m[] = {P_BG, P_TEXT, P_GRID, P_AXIS, P_FRAME, P_DIM,
		P_ZERO, P_MARK, P_PANE, P_PANEGRID, P_LEGBG, P_LEGBORDER, P_UNDEF,
		P_ASYM, P_WIRE};

	return role >= 0 && role < (int)(sizeof m / sizeof m[0]) ? m[role] : P_TEXT;
}

int
pr_ser(int s, int tint)
{
	return P_SER + (s % NSER) * 4 + (tint & 3);
}

void pr_clip(int x0, int y0, int x1, int y1) { cv_clip(x0, y0, x1, y1); }
void pr_put(int x, int y, int c) { put(x, y, c); }
int pr_get(int x, int y) { return get(x, y); }
void pr_fill(int x, int y, int w, int h, int c) { fill(x, y, w, h, c); }
void pr_rect(int x, int y, int w, int h, int c) { rect(x, y, w, h, c); }
void pr_line1(int x0, int y0, int x1, int y1, int c) { line1(x0, y0, x1, y1, c); }
int pr_text(int x, int y, const char *s, int c) { return text(x, y, s, c); }
int pr_textw(const char *s) { return textw(s); }
void pr_text_bg(int x, int y, const char *s, int c, int bg) { text_bg(x, y, s, c, bg); }
void pr_ascii(char *dst, size_t n, const char *s) { ascii_of(dst, n, s); }
void pr_title(int cx, int y, const char *s, int maxw) { title_at(cx, y, s, maxw); }
void pr_seg(double x0, double y0, double x1, double y1, double w) { cov_seg(x0, y0, x1, y1, w); }
void pr_disc(double x, double y, double r) { cov_disc(x, y, r); }
void pr_ink_ser(int s) { cov_ser(s); }
void pr_ink_solid(int c) { cov_solid(c); }
void pr_ink_tints(const int *t) { cov_put(t); }
const char *pr_g4(double v) { return g4(v); }
const char *pr_gpi(double v) { return gpi(v); }

int
pr_ticks(double lo, double hi, int maxn, double *v, char (*lab)[20])
{
	struct ticks t;
	int k;

	mkticks(&t, lo, hi, 0, maxn, 0);
	for (k = 0; k < t.n; k++) {
		v[k] = t.v[k];
		memcpy(lab[k], t.lab[k], 20);
	}
	return t.n;
}

static void
ax_from(struct ax2 *a, const struct pr_ax *p)
{
	memset(a, 0, sizeof *a);
	a->lo[0] = p->lo[0];
	a->hi[0] = p->hi[0];
	a->lo[1] = p->lo[1];
	a->hi[1] = p->hi[1];
	a->L = p->L;
	a->T = p->T;
	a->R = p->R;
	a->B = p->B;
}

void
pr_layout(struct pr_ax *p, int x0, int y0, int w, int h, int small)
{
	struct ax2 a;

	ax_from(&a, p);
	layout2(&a, x0, y0, w, h, 0, small);
	p->L = a.L;
	p->T = a.T;
	p->R = a.R;
	p->B = a.B;
}

void
pr_axes(const struct pr_ax *p, const char *xn, const char *yn, int small)
{
	struct ax2 a;

	ax_from(&a, p);
	axes2(&a, xn, yn, small, 1, 1);
}

void
pr_legend(int n, const char (*lab)[80], const int *col, const int *style,
    const struct pr_ax *p)
{
	struct legend lg;
	struct ax2 a;
	int k;

	memset(&lg, 0, sizeof lg);
	for (k = 0; k < n && k < MAXSER + 2; k++) {
		snprintf(lg.lab[k], sizeof lg.lab[0], "%s", lab[k]);
		lg.col[k] = col[k];
		lg.style[k] = style[k];
	}
	lg.n = k;
	ax_from(&a, p);
	legend_draw(&lg, &a);
}

void
pr_proj(const struct pr_v3 *P, double X, double Y, double Z, double *sx,
    double *sy, double *dep)
{
	struct v3 V;

	V.az = P->az;
	V.el = P->el;
	V.cx = P->cx;
	V.cy = P->cy;
	V.S = P->S;
	V.h = 1;
	proj(&V, X, Y, Z, sx, sy, dep);
}

void
pr_tri(const double *x, const double *y, const double *z, int col, int zt)
{
	tri(x, y, z, col, zt);
}

/* the values of f (one variable) at xs: ok 0 undefined, 1 a value, 2 infinite */
void
pr_eval1(const struct func *f, nid body, int n, const double *xs, double *ys,
    uint8_t *ok)
{
	struct ev e;

	ev_open(&e, f, body);
	ev_run(&e, n, f->np ? xs : NULL, ys, ok);
	ev_close(&e);
}

/* plot's automatic bounds for f's slot 0 (auto1), with the reason */
void
pr_auto_range(const struct func *f, double *lo, double *hi, char *why,
    size_t n)
{
	struct ev e;
	struct sbuf w = {0};
	double base[MAXP];

	memset(&O, 0, sizeof O);
	base_of(f, base, 0.7);
	ev_open(&e, f, f->body);
	auto1(&e, base, 0, 0, 0, lo, hi, &w, 0);
	ev_close(&e);
	snprintf(why, n, "%s", w.p ? w.p : "");
	free(w.p);
}

int
pr_vrange(double *v, int n, double *lo, double *hi, double *vmin,
    double *vmax)
{
	return vrange(v, n, 0, 0, lo, hi, vmin, vmax);
}
