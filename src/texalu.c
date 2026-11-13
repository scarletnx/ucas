/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * The texture unit as an ALU (texalu.h has the operations' semantics).
 *
 * The first half is the semantics: the address modes on an integer texel
 * index, the linear filter with its weights on the sub-texel grid and
 * one rounding, the plain-operation definition of a fetch (for SAT), the
 * cost classes.
 *
 * The second half is the search.  Nothing in it knows a GPU: a
 * recogniser proposes a read through the unit for a table read whose
 * index is an address mode of some subterm, and for a weighted sum of
 * neighbouring table entries; the candidate is proved (the index
 * identity by enumeration or SAT; a filtered sum by a derivation of its
 * error against the exact sum, from the model's subtexel and tex_within),
 * priced under the active model, and kept only when it is cheaper there.
 * What the pipeline needs for it is spelled out, never applied.
 */
#include "calc.h"
#include "fuse.h"
#include "memk.h"
#include "report.h"
#include "texalu.h"
#include "texenc.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- semantics ---- */

static int64_t
floored_mod(int64_t i, int64_t n)
{
	int64_t r = i % n;

	return r < 0 ? r + n : r;
}

int64_t
tex_wrap(int op, int64_t i, int64_t n)
{
	int64_t r;

	switch (op) {
	case O_FETCH_REPEAT:
		return floored_mod(i, n);
	case O_FETCH_MIRROR:
		r = floored_mod(i, 2 * n);
		return r < n ? r : 2 * n - 1 - r;
	case O_FETCH_CLAMP:
		return i < 0 ? 0 : i >= n ? n - 1 : i;
	case O_FETCH_MCLAMP:
		r = i >= 0 ? i : -(1 + i);
		return r >= n ? n - 1 : r;
	default:	/* O_FETCH_BORDER */
		return i >= 0 && i < n ? i : -1;
	}
}

int64_t
tex_layer(double z, int64_t n)
{
	double f, d, r;

	if (isnan(z))
		return -1;
	if (!(z >= 0))
		return 0;
	if (!(z < (double)n))
		return n - 1;
	f = floor(z);
	d = z - f;		/* exact */
	r = d > 0.5 ? f + 1 : d < 0.5 ? f : fmod(f, 2) == 0 ? f : f + 1;
	return r > (double)(n - 1) ? n - 1 : (int64_t)r;
}

double
tex_weight(double x, int bits, int64_t *i0)
{
	double s, f, a, g = ldexp(1.0, bits);

	if (!isfinite(x)) {
		*i0 = 0;
		return NAN;
	}
	/* far outside any table: the clamp decides, the weight does not */
	if (x > 4e18)
		x = 4e18;
	if (x < -4e18)
		x = -4e18;
	s = x - 0.5;		/* exact: x is an f32 */
	f = floor(s);
	a = s - f;		/* exact, in [0, 1) */
	*i0 = (int64_t)f;
	return floor(a * g + 0.5) / g;
}

/*
 * sum w[k] t[k] rounded once to f32.  Each product is exact in a double
 * (29 + 24 bits); the sum is kept as non-overlapping partials (Shewchuk),
 * its leading part rounded to odd from the sign of what is left, and a
 * double rounded to odd rounds to the same f32 as the exact sum.
 */
double
tex_blend(const double *t, const double *w, int n)
{
	double p[16], x, y, hi, lo = 0, naive = 0;
	int m = 0, i, j, k, fin = 1;
	uint64_t u;

	for (k = 0; k < n; k++) {
		x = w[k] * t[k];
		naive += x;
		if (!isfinite(x))
			fin = 0;
		if (w[k] == 0 || !fin)
			continue;
		for (i = 0, j = 0; j < m; j++) {
			y = p[j];
			if (fabs(x) < fabs(y)) {
				double sw = x;
				x = y;
				y = sw;
			}
			hi = x + y;
			lo = y - (hi - x);
			if (lo != 0)
				p[i++] = lo;
			x = hi;
		}
		p[i] = x;
		m = i + 1;
	}
	if (!fin)
		return (double)(float)naive;
	if (m == 0)
		return 0;
	hi = p[--m];
	lo = 0;
	while (m > 0) {
		x = hi;
		y = p[--m];
		hi = x + y;
		lo = y - (hi - x);
		if (lo != 0)
			break;
	}
	if (lo != 0) {
		memcpy(&u, &hi, sizeof u);
		if (!(u & 1))
			hi = nextafter(hi, lo > 0 ? INFINITY : -INFINITY);
	}
	return (double)(float)hi;
}

/* the least or the greatest of the texels whose weight is not 0 (a NaN
 * texel makes the result a NaN) */
static double
tex_reduce(const double *t, const double *w, int n, int red)
{
	double r = NAN;
	int k, any = 0;

	for (k = 0; k < n; k++) {
		if (w[k] == 0)
			continue;
		if (isnan(t[k]))
			return t[k];
		if (!any || (red == TEX_REDMIN ? t[k] < r : t[k] > r))
			r = t[k];
		any = 1;
	}
	return r;
}

/*
 * The eight texels of a 3-D read: each weight is a product of three
 * on the 2^-bits grid, up to 42 bits, so it is split in two of at most
 * 21 (each product with an f32 texel exact in a double) and the sixteen
 * go through tex_blend's one rounding.  A texel that is not finite takes
 * tex_blend's own rule (the plain sum) on the eight.
 */
static double
tex_blend3(const double *t, const double *w, int bits)
{
	double t2[16], w2[16], g = ldexp(1.0, 3 * bits), h = ldexp(1.0, 21);
	int k;

	for (k = 0; k < 8; k++)
		if (!isfinite(t[k]))
			return tex_blend(t, w, 8);
	for (k = 0; k < 8; k++) {
		double wi = w[k] * g, hi = floor(wi / h) * h;
		t2[2 * k] = t2[2 * k + 1] = t[k];
		w2[2 * k] = hi / g;
		w2[2 * k + 1] = (wi - hi) / g;
	}
	return tex_blend(t2, w2, 16);
}

double
tex_filter(int op, int64_t ki, int64_t n, double x, double y, double z,
    double (*tv)(void *ctx, int64_t j), void *ctx)
{
	int bits = TEX_BITS(ki);
	int64_t W = op == O_TEX_LINEAR ? n : TEX_WIDTH(ki), H = op ==
	    O_TEX_TRILINEAR ? TEX_HEIGHT(ki) : n / W, D = n / (W * H);

	/* REPEAT and MIRRORED_REPEAT take normalised coordinates: the unit's
	 * x is u n (exact in a double: 24 and at most 24 bits); so does a 3-D
	 * read (Vulkan has no unnormalizedCoordinates for one) */
	if (ki & TEX_WRAPMASK)
		x *= (double)W;
	if (op == O_TEX_TRILINEAR) {
		x *= (double)W;
		y *= (double)H;
		z *= (double)D;
	}
	int64_t i0, j0 = 0, k0 = 0, c0, c1, r0, r1, s0, s1;
	double a = tex_weight(x, bits, &i0), b = 0, c = 0, t[8], w[8];
	int q;

	if (op != O_TEX_LINEAR)
		b = tex_weight(y, bits, &j0);
	if (op == O_TEX_TRILINEAR)
		c = tex_weight(z, bits, &k0);
	if (isnan(a) || isnan(b) || isnan(c))
		return NAN;
	c0 = tex_wrap(TEX_WRAP(ki), i0, W);
	c1 = tex_wrap(TEX_WRAP(ki), i0 >= INT64_MAX - 1 ? i0 : i0 + 1, W);
	if (op == O_TEX_LINEAR) {
		t[0] = tv(ctx, c0);
		t[1] = tv(ctx, c1);
		w[0] = 1 - a;
		w[1] = a;
		return TEX_RED(ki) ? tex_reduce(t, w, 2, TEX_RED(ki)) :
		    tex_blend(t, w, 2);
	}
	r0 = tex_wrap(O_FETCH_CLAMP, j0, H);
	r1 = tex_wrap(O_FETCH_CLAMP, j0 >= INT64_MAX - 1 ? j0 : j0 + 1, H);
	if (op == O_TEX_TRILINEAR) {
		/* texel (dx, dy, dz) at bit 0, 1, 2 of q */
		s0 = tex_wrap(O_FETCH_CLAMP, k0, D);
		s1 = tex_wrap(O_FETCH_CLAMP, k0 >= INT64_MAX - 1 ? k0 : k0 + 1,
		    D);
		for (q = 0; q < 8; q++) {
			t[q] = tv(ctx, ((q & 4 ? s1 : s0) * H + (q & 2 ? r1 :
			    r0)) * W + (q & 1 ? c1 : c0));
			w[q] = (q & 1 ? a : 1 - a) * (q & 2 ? b : 1 - b) *
			    (q & 4 ? c : 1 - c);
		}
		return tex_blend3(t, w, bits);
	}
	t[0] = tv(ctx, r0 * W + c0);
	t[1] = tv(ctx, r0 * W + c1);
	t[2] = tv(ctx, r1 * W + c0);
	t[3] = tv(ctx, r1 * W + c1);
	w[0] = (1 - a) * (1 - b);
	w[1] = a * (1 - b);
	w[2] = (1 - a) * b;
	w[3] = a * b;
	return TEX_RED(ki) ? tex_reduce(t, w, 4, TEX_RED(ki)) : tex_blend(t, w,
	    4);
}

/* a fetch_ read's index as plain operations on a 64-bit copy of i */
static nid
wrap_expr(int op, nid i, int64_t n)
{
	ty_t t = TY(K_SINT, 64);
	nid j = mk_cast(t, i), kn = mk_const_i(t, n), r;

	switch (op) {
	case O_FETCH_REPEAT:
		return mk_bin(O_MOD, mk_bin(O_ADD, mk_bin(O_MOD, j, kn), kn), kn);
	case O_FETCH_MIRROR:
		kn = mk_const_i(t, 2 * n);
		r = mk_bin(O_MOD, mk_bin(O_ADD, mk_bin(O_MOD, j, kn), kn), kn);
		return mk_bin(O_MIN, r, mk_bin(O_SUB, mk_const_i(t, 2 * n - 1),
		    r));
	case O_FETCH_CLAMP:
		return mk_bin(O_MIN, mk_bin(O_MAX, j, mk_const_i(t, 0)),
		    mk_const_i(t, n - 1));
	case O_FETCH_MCLAMP:
		return mk_bin(O_MIN, mk_bin(O_MAX, j, mk_bin(O_SUB,
		    mk_const_i(t, -1), j)), mk_const_i(t, n - 1));
	}
	return j;
}

nid
tex_expand(nid n)
{
	int op = N.op[n];
	nid tab = N.a[n], i = N.b[n], j;
	int64_t cnt = N.b[tab];
	ty_t t = TY(K_SINT, 64);

	if (IS_TEXEL(op))
		return mk_index(tab, i);
	if (!IS_FETCH(op))
		return n;
	if (TEX_OFF(N.ki[n]))
		i = mk_bin(O_ADD, mk_cast(t, i), mk_const_i(t,
		    TEX_OFF(N.ki[n])));
	j = wrap_expr(op, i, cnt);
	if (op != O_FETCH_BORDER)
		return mk_index(tab, j);
	return mk_sel(mk_bin(O_LAND, mk_bin(O_GE, j, mk_const_i(t, 0)),
	    mk_bin(O_LT, j, mk_const_i(t, cnt))), mk_index(tab, j), N.c[n]);
}

/* is v the f32 of k/255 for a whole k in 0..255: an 8-bit normalised texel */
static int
unorm8(double v, int *k)
{
	double q = floor(v * 255 + 0.5);

	if (!(q >= 0 && q <= 255) || (double)(float)(q / 255) != v)
		return 0;
	if (k)
		*k = (int)q;
	return 1;
}

/* the same for k/65535: a 16-bit normalised texel */
static int
unorm16(double v, int *k)
{
	double q = floor(v * 65535 + 0.5);

	if (!(q >= 0 && q <= 65535) || (double)(float)(q / 65535) != v)
		return 0;
	if (k)
		*k = (int)q;
	return 1;
}

/*
 * A filtered read's class from what its texture holds: of the formats
 * whose texels are the f32 values t exactly (bytes k/255, 16-bit units
 * k/65535, half floats, f32), the one m prices lowest; *price INFINITY
 * when m lists none (the class then the bytes' or f32's, as before the
 * 16-bit forms).
 */
static int
tex_fmt_of(const struct model *m, const double *t, uint32_t n, float *price)
{
	static const int cls[4] = {C_TEX_SAMPLE, C_TEX_SAMPLE_16,
	    C_TEX_SAMPLE_F16, C_TEX_SAMPLE_F32};
	int hold[4] = {1, 1, 1, 1}, i, best = -1;
	uint32_t j;

	for (j = 0; j < n; j++) {
		if (!unorm8(t[j], NULL))
			hold[0] = 0;
		if (!unorm16(t[j], NULL))
			hold[1] = 0;
		if (fround(16, t[j]) != t[j])
			hold[2] = 0;
	}
	for (i = 0; i < 4; i++)
		if (hold[i] && m && model_lists(m, cls[i]) && (best < 0 ||
		    m->c[cls[i]] < m->c[cls[best]]))
			best = i;
	*price = best < 0 ? INFINITY : m->c[cls[best]];
	return best >= 0 ? cls[best] : hold[0] ? C_TEX_SAMPLE :
	    C_TEX_SAMPLE_F32;
}

/* a filtered read's class under m (its table's texels) */
static int
tex_fmt(const struct model *m, nid tab, float *price)
{
	uint32_t j, n = N.b[tab];
	double *t = xmalloc((n ? n : 1) * sizeof *t);
	int k;

	for (j = 0; j < n; j++) {
		nid e = N.lst[N.a[tab] + j];
		t[j] = !is_const(e) ? NAN : TY_FP(N.ty[e]) ? N.kr[e] :
		    (double)N.ki[e];
	}
	k = tex_fmt_of(m, t, n, price);
	free(t);
	return k;
}

int
tex_class(nid n)
{
	float p;

	if (IS_FETCH(N.op[n]))
		return N.ki[n] == TEX_SRGB ? C_TEX_SRGB : C_TEX_NEAREST;
	if (N.op[n] == O_FETCH_LAYER)
		return C_TEX_LAYER;
	if (IS_TEXEL(N.op[n]))
		return N.op[n] == O_FETCH_TEXEL ? C_TEX_FETCH : C_TEX_BUFFER;
	if (TEX_RED(N.ki[n]))
		return C_TEX_MINMAX;
	return tex_fmt(model_active(), N.a[n], &p);
}

float
tex_cost(const struct model *m, nid n, int *cls)
{
	int k;
	float p;

	if (!IS_FETCH(N.op[n]) && N.op[n] != O_FETCH_LAYER &&
	    !IS_TEXEL(N.op[n])) {
		if (TEX_RED(N.ki[n])) {
			*cls = C_TEX_MINMAX;
			return model_lists(m, C_TEX_MINMAX) ? m->c[C_TEX_MINMAX] :
			    INFINITY;
		}
		*cls = tex_fmt(m, N.a[n], &p);
		/* a 3-D read: its format's 2-D read and what the third axis
		 * adds, each only where the model lists it */
		if (N.op[n] == O_TEX_TRILINEAR && p < INFINITY) {
			*cls = C_TEX_3D;
			return cclass_available(m, C_TEX_3D) ? p +
			    m->c[C_TEX_3D] : INFINITY;
		}
		return p;
	}
	k = tex_class(n);
	*cls = k;
	/* unlike a memory kind of memk.c, a read through the unit has no
	 * fallback: a model that does not list its class cannot do it */
	if (!model_lists(m, k))
		return INFINITY;
	/* a nearest read's coordinate is made from the index: (i + 1/2)
	 * times 1/n, a conversion and a multiply-add */
	if (k == C_TEX_NEAREST)
		return m->c[k] + m->c[C_CONV] + m->c[C_FMUL];
	/* texelFetch: the byte or the index is the coordinate; a layer:
	 * the float is */
	return m->c[k];
}

int
tex_srgb_table(nid tab)
{
	uint32_t k;

	if (N.op[tab] != O_TABLE || N.b[tab] != 256)
		return 0;
	for (k = 0; k < 256; k++) {
		nid e = N.lst[N.a[tab] + k];
		if (!is_const(e) || N.ty[e] != TY_F32 || N.kr[e] !=
		    texenc_srgb8((int)k))
			return 0;
	}
	return 1;
}

int
tex_lookup(const char *s, size_t n, int *bits, int64_t *wrapki)
{
	static const int fetch[] = {O_FETCH_REPEAT, O_FETCH_MIRROR,
	    O_FETCH_CLAMP, O_FETCH_MCLAMP, O_FETCH_BORDER, O_FETCH_LAYER,
	    O_FETCH_TEXEL, O_FETCH_BUFFER};
	size_t i, l;
	int op = 0, red = 0, wrap = 0;

	*bits = 0;
	*wrapki = 0;
	if (n == 10 && !memcmp(s, "fetch_srgb", 10)) {
		*bits = TEX_SRGB;
		return O_FETCH_CLAMP;
	}
	for (i = 0; i < sizeof fetch / sizeof fetch[0]; i++) {
		l = strlen(opinfo[fetch[i]].name);
		if (l == n && !memcmp(s, opinfo[fetch[i]].name, n))
			return fetch[i];
		/* name_offset: the read with a ConstOffset */
		if (IS_FETCH(fetch[i]) && l + 7 == n && !memcmp(s,
		    opinfo[fetch[i]].name, l) && !memcmp(s + l, "_offset", 7)) {
			*bits = TEX_OFFNAME;
			return fetch[i];
		}
	}
	if (n > 10 && !memcmp(s, "tex_linear", 10)) {
		op = O_TEX_LINEAR;
		i = 10;
	} else if (n > 12 && !memcmp(s, "tex_bilinear", 12)) {
		op = O_TEX_BILINEAR;
		i = 12;
	} else if (n > 13 && !memcmp(s, "tex_trilinear", 13)) {
		op = O_TEX_TRILINEAR;
		i = 13;
	} else if (n > 10 && (!memcmp(s, "tex_repeat", 10) || !memcmp(s,
	    "tex_mirror", 10))) {
		op = O_TEX_LINEAR;
		wrap = s[4] == 'r' ? O_FETCH_REPEAT : O_FETCH_MIRROR;
		i = 10;
	} else if (n > 7 && (!memcmp(s, "tex_min", 7) || !memcmp(s,
	    "tex_max", 7))) {
		op = O_TEX_LINEAR;
		red = s[5] == 'i' ? TEX_REDMIN : TEX_REDMAX;
		i = 7;
	} else if (n > 9 && (!memcmp(s, "tex_bimin", 9) || !memcmp(s,
	    "tex_bimax", 9))) {
		op = O_TEX_BILINEAR;
		red = s[7] == 'i' ? TEX_REDMIN : TEX_REDMAX;
		i = 9;
	} else
		return 0;
	for (; i < n; i++) {
		if (s[i] < '0' || s[i] > '9' || *bits > 99)
			return 0;
		*bits = *bits * 10 + (s[i] - '0');
	}
	/* the grid's bits stay below the reduction's */
	if (*bits > 31)
		return 0;
	*bits |= red;
	*wrapki = TEX_KWRAP(wrap);
	return op;
}

void
tex_print_name(struct sbuf *b, nid n)
{
	if (IS_TEXEL(N.op[n]))
		sb_puts(b, opinfo[N.op[n]].name);
	else if (IS_FETCH(N.op[n]) || N.op[n] == O_FETCH_LAYER)
		sb_printf(b, "%s%s", N.ki[n] == TEX_SRGB ? "fetch_srgb" :
		    opinfo[N.op[n]].name, IS_FETCH(N.op[n]) && TEX_OFF(N.ki[n]) ?
		    "_offset" : "");
	else if (N.ki[n] & TEX_WRAPMASK)
		sb_printf(b, "tex_%s%d", TEX_WRAP(N.ki[n]) == O_FETCH_REPEAT ?
		    "repeat" : "mirror", TEX_BITS(N.ki[n]));
	else if (TEX_RED(N.ki[n]))
		sb_printf(b, "tex_%s%s%d", N.op[n] == O_TEX_BILINEAR ? "bi" : "",
		    TEX_RED(N.ki[n]) == TEX_REDMIN ? "min" : "max",
		    TEX_BITS(N.ki[n]));
	else
		sb_printf(b, "%s%d", opinfo[N.op[n]].name, TEX_BITS(N.ki[n]));
}

/* ---- the search ---- */

#define MAXTERM 20	/* table entries in one sum */
#define MAXREST 12	/* its other terms */
#define MAXREPL 16	/* reads replaced in one function */
#define MAXATOM 6	/* variables of an index */

/*
 * A weight: the sum of c[m] times the atoms whose bit is set in m, over
 * at most three atoms s, t, r (bits 1, 2, 4; any float subterms that are
 * not sums or products), none squared: c[0] + c[1] s + c[2] t + c[3] s t
 * + c[4] r + ...  What the weights of a mix, a bilinear or trilinear
 * blend or a constant stencil are.
 */
struct mform {
	double c[8];
	nid s, t, r;
};

/* an index: c + sum coef[k] atom[k], atoms sorted */
struct aff {
	int n;
	nid atom[MAXATOM];
	int64_t coef[MAXATOM], c;
};

struct term {
	struct mform k;
	nid tab;
	struct aff ix;
	int dx, dy, dz, grp, used;
	int wrap;		/* O_FETCH_REPEAT, _MIRROR: it wraps */
	struct aff ix0;		/* a wrapping index as it is */
	int ok0;		/* ix0 is affine */
};

struct rest {
	struct mform k;
	nid e;
};

struct repl {
	nid from, to;
	float cost;		/* the body's with this one alone */
	double err, eorig;	/* filtered: the bounds, new and as written */
	int exact, filtered, ok;
	int plain;		/* the read as it is, through the unit */
	int approx;		/* FB_: texels that approximate a table */
	int curve;		/* against the formula it replaces (seed_srgb) */
	struct sbuf what, holds, proof, pipe, why;
};

/* the texels of a float table that only approximate it (table_info) */
enum { FB_LEV8 = 1, FB_LEV16 = 2, FB_F16 = 4 };

static struct {
	const struct func *f;
	struct func g;		/* f with its integer ranges narrowed */
	struct plan p;
	struct ival *iv;
	const struct model *m;
	int bits;		/* the model's subtexel, 0: not stated */
	double ulp;		/* its tex_within, 0: not stated */
	struct repl r[MAXREPL];
	int nr;
	int unavail[C_NCLASS];	/* a class a candidate needed */
	int noulp, nobits;
	double tol;		/* the error the request allows */
	int fmtban;		/* FB_: what seed_sum's retry drops */
	struct sbuf far;	/* an index identity out of f32's reach */
	struct sbuf nosum;	/* neighbours that are not one read, and why */
	struct sbuf nolayer;	/* a layer selection not taken, and why */
	struct sbuf nopacked;	/* a packed decode not taken, and why */
	int unavail_packed;	/* tex_packed was needed */
	struct sbuf nocube;	/* a cube map's face not taken, and why */
	int unavail_cube;	/* cube or tex_cube was needed */
	int unavail_3d;		/* tex_3d was needed */
} X;

static double
cval(nid n)
{
	return TY_FP(N.ty[n]) ? N.kr[n] : (double)N.ki[n];
}

/* the proved range of an integer subterm of the body; 0: none */
static int
irange(nid e, int64_t *lo, int64_t *hi)
{
	uint32_t k;

	if (is_const(e)) {
		*lo = *hi = N.ki[e];
		return 1;
	}
	k = plan_loc(&X.p, e);
	if (k == NOTOK || X.iv[k].top || X.iv[k].wrap)
		return 0;
	*lo = X.iv[k].lo;
	*hi = X.iv[k].hi;
	return 1;
}

/* a float atom's range: a ranged parameter's, through casts */
static int
frange(nid e, double *lo, double *hi)
{
	while (N.op[e] == O_CAST && TY_FP(N.ty[N.a[e]]))
		e = N.a[e];
	if (N.op[e] != O_VAR || N.ki[e] >= X.f->np ||
	    !X.f->p[N.ki[e]].ranged || !TY_FP(X.f->p[N.ki[e]].ty))
		return 0;
	*lo = X.f->p[N.ki[e]].rlo;
	*hi = X.f->p[N.ki[e]].rhi;
	return 1;
}

static int
has_read(nid e, int depth)
{
	uint32_t j;

	if (e == 0 || depth > 48)
		return 0;
	switch (N.op[e]) {
	case O_CONST: case O_VAR: case O_UNDEF:
		return 0;
	case O_INDEX:
		return 1;
	case O_TABLE:
		for (j = 0; j < N.b[e]; j++)
			if (has_read(N.lst[N.a[e] + j], depth + 1))
				return 1;
		return 0;
	}
	return has_read(N.a[e], depth + 1) || has_read(N.b[e], depth + 1) ||
	    has_read(N.c[e], depth + 1);
}

/* ---- weights ---- */

static struct mform
mf_const(double v)
{
	struct mform r = {{0, 0, 0, 0, 0, 0, 0, 0}, 0, 0, 0};

	r.c[0] = v;
	return r;
}

static nid
mf_atom(const struct mform *x, int i)
{
	return i == 0 ? x->s : i == 1 ? x->t : x->r;
}

/* x over the atoms u[0..2] (0: none), which include its own; 0 if they
 * do not */
static int
mf_over(struct mform *x, const nid *u)
{
	struct mform r = mf_const(x->c[0]);
	int m, i, j, nm;

	r.s = u[0];
	r.t = u[1];
	r.r = u[2];
	for (m = 1; m < 8; m++) {
		if (x->c[m] == 0)
			continue;
		for (nm = 0, i = 0; i < 3; i++) {
			if (!(m >> i & 1))
				continue;
			for (j = 0; j < 3 && u[j] != mf_atom(x, i); j++)
				;
			if (j == 3)
				return 0;
			nm |= 1 << j;
		}
		r.c[nm] = x->c[m];
	}
	*x = r;
	return 1;
}

/* the atoms it uses, first ones first */
static void
mf_tidy(struct mform *x)
{
	nid u[3] = {0, 0, 0};
	int m, i, n = 0, used[3] = {0, 0, 0};

	for (m = 1; m < 8; m++)
		for (i = 0; i < 3; i++)
			if ((m >> i & 1) && x->c[m] != 0)
				used[i] = 1;
	for (i = 0; i < 3; i++)
		if (used[i])
			u[n++] = mf_atom(x, i);
	if (!used[0] || !used[1] || !used[2]) {
		for (i = 0; i < 3; i++)
			if (!used[i])
				for (m = 1; m < 8; m++)
					if (m >> i & 1)
						x->c[m] = 0;
		(void)mf_over(x, u);	/* its own atoms: always */
	}
}

/* both over the same atoms; 0 if they have more than three between them */
static int
mf_join(struct mform *a, struct mform *b)
{
	nid at[6], u[3] = {0, 0, 0};
	int i, j, n = 0;

	mf_tidy(a);
	mf_tidy(b);
	for (i = 0; i < 3; i++) {
		at[i] = mf_atom(a, i);
		at[3 + i] = mf_atom(b, i);
	}
	for (i = 0; i < 6; i++) {
		if (!at[i])
			continue;
		for (j = 0; j < n; j++)
			if (u[j] == at[i])
				break;
		if (j < n)
			continue;
		if (n == 3)
			return 0;
		u[n++] = at[i];
	}
	return mf_over(a, u) && mf_over(b, u);
}

static int
mf_add(struct mform *r, struct mform a, struct mform b, double sb)
{
	int i;

	if (!mf_join(&a, &b))
		return 0;
	for (i = 0; i < 8; i++)
		a.c[i] += sb * b.c[i];
	mf_tidy(&a);
	*r = a;
	return 1;
}

static int
mf_has_r(const struct mform *a)
{
	return a->c[4] != 0 || a->c[5] != 0 || a->c[6] != 0 || a->c[7] != 0;
}

static int
mf_mul(struct mform *r, struct mform a, struct mform b)
{
	struct mform o;
	int i, j;

	if (!mf_join(&a, &b))
		return 0;
	/* no square of an atom */
	for (i = 1; i < 8; i++)
		for (j = 1; j < 8; j++)
			if ((i & j) && a.c[i] != 0 && b.c[j] != 0)
				return 0;
	o = a;
	if (!mf_has_r(&a) && !mf_has_r(&b)) {
		o.c[0] = a.c[0] * b.c[0];
		o.c[1] = a.c[0] * b.c[1] + a.c[1] * b.c[0];
		o.c[2] = a.c[0] * b.c[2] + a.c[2] * b.c[0];
		o.c[3] = a.c[0] * b.c[3] + a.c[3] * b.c[0] + a.c[1] * b.c[2] +
		    a.c[2] * b.c[1];
	} else {
		for (i = 0; i < 8; i++)
			o.c[i] = 0;
		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				if (!(i & j))
					o.c[i | j] += a.c[i] * b.c[j];
	}
	mf_tidy(&o);
	*r = o;
	return 1;
}

static int
mf_isconst(const struct mform *a)
{
	return a->c[1] == 0 && a->c[2] == 0 && a->c[3] == 0 && !mf_has_r(a);
}

/* its range over its atoms' ranges (multilinear: at the corners) */
static int
mf_range(const struct mform *a, double *lo, double *hi)
{
	double sl = 0, sh = 0, tl = 0, th = 0, rl = 0, rh = 0, v;
	int i;

	if (a->s && !frange(a->s, &sl, &sh))
		return 0;
	if (a->t && !frange(a->t, &tl, &th))
		return 0;
	if (a->r && !frange(a->r, &rl, &rh))
		return 0;
	for (i = 0; i < 8; i++) {
		double s = i & 1 ? sh : sl, t = i & 2 ? th : tl, r = i & 4 ? rh :
		    rl;
		v = a->c[0] + a->c[1] * s + a->c[2] * t + a->c[3] * s * t;
		if (mf_has_r(a))
			v += a->c[4] * r + a->c[5] * s * r + a->c[6] * t * r +
			    a->c[7] * s * t * r;
		if (!i || v < *lo)
			*lo = v;
		if (!i || v > *hi)
			*hi = v;
	}
	return 1;
}

/* the largest |a| over its atoms' ranges; infinity when one has none */
static double
mf_mag(const struct mform *a)
{
	double lo, hi;

	if (!mf_range(a, &lo, &hi))
		return INFINITY;
	return fabs(lo) > fabs(hi) ? fabs(lo) : fabs(hi);
}

static nid
kf(double v)
{
	return mk_cast(TY_F32, mk_const_r(v));
}

/* the weight as an f32 expression; *ops: the operations it takes */
static nid
mf_node(const struct mform *a, int *ops)
{
	nid r = 0, x;
	int i;

	for (i = 7; i >= 1; i--) {
		int k;
		if (a->c[i] == 0)
			continue;
		/* the monomial: its atoms' product, left to right */
		for (x = 0, k = 0; k < 3; k++) {
			if (!(i >> k & 1))
				continue;
			if (x)
				++*ops;
			x = x ? mk_bin(O_MUL, x, mf_atom(a, k)) : mf_atom(a, k);
		}
		x = mk_cast(TY_F32, x);
		if (a->c[i] != 1) {
			x = mk_bin(O_MUL, kf(a->c[i]), x);
			++*ops;
		}
		if (r)
			++*ops;
		r = r ? mk_bin(O_ADD, r, x) : x;
	}
	if (!r)
		return kf(a->c[0]);
	if (a->c[0] != 0) {
		r = mk_bin(O_ADD, r, kf(a->c[0]));
		++*ops;
	}
	return r;
}

/* a weight expression (no table read in it) as a form; 0: it is not one */
static int
mf_of(nid e, struct mform *r, int depth)
{
	struct mform a, b;

	if (depth > 24)
		return 0;
	if (is_const(e)) {
		*r = mf_const(cval(e));
		return isfinite(r->c[0]);
	}
	switch (N.op[e]) {
	case O_ADD: case O_SUB:
		return mf_of(N.a[e], &a, depth + 1) && mf_of(N.b[e], &b,
		    depth + 1) && mf_add(r, a, b, N.op[e] == O_ADD ? 1 : -1);
	case O_NEG:
		if (!mf_of(N.a[e], &a, depth + 1))
			return 0;
		return mf_add(r, mf_const(0), a, -1);
	case O_MUL:
		return mf_of(N.a[e], &a, depth + 1) && mf_of(N.b[e], &b,
		    depth + 1) && mf_mul(r, a, b);
	case O_DIV:
		if (!is_const(N.b[e]) || cval(N.b[e]) == 0 ||
		    !mf_of(N.a[e], &a, depth + 1))
			return 0;
		return mf_mul(r, a, mf_const(1 / cval(N.b[e])));
	}
	if (!TY_FP(N.ty[e]) || has_read(e, 0))
		return 0;
	*r = mf_const(0);
	r->c[1] = 1;
	r->s = e;
	return 1;
}

/* ---- indices ---- */

static int
aff_add(struct aff *r, const struct aff *b, int64_t k)
{
	int i, j;

	r->c += k * b->c;
	for (i = 0; i < b->n; i++) {
		for (j = 0; j < r->n; j++)
			if (r->atom[j] == b->atom[i])
				break;
		if (j == r->n) {
			if (r->n == MAXATOM)
				return 0;
			r->atom[r->n] = b->atom[i];
			r->coef[r->n++] = 0;
		}
		r->coef[j] += k * b->coef[i];
	}
	return 1;
}

/*
 * An integer expression as an affine form, where the machine's
 * arithmetic is the integers' (no node of it wraps on the domain).
 */
static int
aff_of(nid e, struct aff *r, int depth)
{
	struct aff a, b;
	int64_t lo, hi, k;
	int i, j;

	memset(r, 0, sizeof *r);
	if (depth > 24 || !TY_ISINT(N.ty[e]))
		return 0;
	if (is_const(e)) {
		r->c = N.ki[e];
		return 1;
	}
	switch (N.op[e]) {
	case O_ADD: case O_SUB:
		if (!irange(e, &lo, &hi) || !aff_of(N.a[e], &a, depth + 1) ||
		    !aff_of(N.b[e], &b, depth + 1))
			break;
		*r = a;
		if (!aff_add(r, &b, N.op[e] == O_ADD ? 1 : -1))
			return 0;
		goto tidy;
	case O_MUL: case O_SHL:
		if (!irange(e, &lo, &hi))
			break;
		if (N.op[e] == O_MUL && is_const(N.a[e])) {
			k = N.ki[N.a[e]];
			e = N.b[e];
		} else if (is_const(N.b[e])) {
			k = N.ki[N.b[e]];
			if (N.op[e] == O_SHL) {
				if (k < 0 || k > 30)
					break;
				k = (int64_t)1 << k;
			}
			e = N.a[e];
		} else
			break;
		if (k > 0xffffff || k < -0xffffff || !aff_of(e, &a, depth + 1))
			return 0;
		if (!aff_add(r, &a, k))
			return 0;
		goto tidy;
	}
	memset(r, 0, sizeof *r);
	r->n = 1;
	r->atom[0] = e;
	r->coef[0] = 1;
	return 1;
tidy:
	for (i = j = 0; i < r->n; i++)
		if (r->coef[i] != 0) {
			r->atom[j] = r->atom[i];
			r->coef[j++] = r->coef[i];
		}
	r->n = j;
	for (i = 1; i < r->n; i++)	/* by atom */
		for (j = i; j > 0 && r->atom[j - 1] > r->atom[j]; j--) {
			nid ta = r->atom[j];
			int64_t tc = r->coef[j];
			r->atom[j] = r->atom[j - 1];
			r->coef[j] = r->coef[j - 1];
			r->atom[j - 1] = ta;
			r->coef[j - 1] = tc;
		}
	return 1;
}

/* two tables with the same entries (a table written twice is two nodes) */
static int
same_table(nid a, nid b)
{
	uint32_t j;

	if (a == b)
		return 1;
	if (N.b[a] != N.b[b] || N.ty[a] != N.ty[b])
		return 0;
	for (j = 0; j < N.b[a]; j++)
		if (N.lst[N.a[a] + j] != N.lst[N.a[b] + j])
			return 0;
	return 1;
}

static int
aff_same_base(const struct aff *a, const struct aff *b)
{
	int i;

	if (a->n != b->n)
		return 0;
	for (i = 0; i < a->n; i++)
		if (a->atom[i] != b->atom[i] || a->coef[i] != b->coef[i])
			return 0;
	return 1;
}

/* sum of the atoms whose coefficient is k, each once: the expression
 * and its proved range; 0 when there is none or no range */
static nid
aff_part(const struct aff *a, int64_t k, int64_t *lo, int64_t *hi)
{
	nid r = 0;
	int i;
	int64_t l, h;

	*lo = *hi = 0;
	for (i = 0; i < a->n; i++) {
		if (a->coef[i] != k)
			continue;
		if (!irange(a->atom[i], &l, &h) || l < -0x3fffffff ||
		    h > 0x3fffffff)
			return 0;
		*lo += l;
		*hi += h;
		r = r ? mk_bin(O_ADD, r, a->atom[i]) : a->atom[i];
	}
	return r;
}

/* ---- replacing nodes ---- */

static nid
swap1(nid n, const nid *from, const nid *to, int nm, nid *memo, uint32_t lim)
{
	nid r, *e;
	uint32_t i;
	int k;

	if (n == 0)
		return 0;
	for (k = 0; k < nm; k++)
		if (from[k] == n)
			return to[k];
	if (n < lim && memo[n])
		return memo[n];
	switch (N.op[n]) {
	case O_CONST: case O_VAR: case O_UNDEF:
		return n;
	case O_TABLE:
		e = xmalloc(N.b[n] * sizeof *e + 1);
		for (i = 0; i < N.b[n]; i++)
			e[i] = swap1(N.lst[N.a[n] + i], from, to, nm, memo, lim);
		r = mk_table(e, N.b[n]);
		free(e);
		break;
	case O_SEL:
		r = mk_sel(swap1(N.a[n], from, to, nm, memo, lim),
		    swap1(N.b[n], from, to, nm, memo, lim),
		    swap1(N.c[n], from, to, nm, memo, lim));
		break;
	case O_CAST:
		r = mk_cast(N.ty[n], swap1(N.a[n], from, to, nm, memo, lim));
		break;
	case O_INDEX:
		r = mk_index(swap1(N.a[n], from, to, nm, memo, lim),
		    swap1(N.b[n], from, to, nm, memo, lim));
		break;
	default:
		r = mk_like(n, swap1(N.a[n], from, to, nm, memo, lim),
		    swap1(N.b[n], from, to, nm, memo, lim),
		    swap1(N.c[n], from, to, nm, memo, lim));
	}
	if (n < lim)
		memo[n] = r;
	return r;
}

/* root with each from[k] replaced by to[k] */
static nid
swap_nodes(nid root, const nid *from, const nid *to, int nm)
{
	uint32_t lim = N.n;
	nid *memo = xcalloc(lim, sizeof *memo), r;

	r = swap1(root, from, to, nm, memo, lim);
	free(memo);
	return r;
}

/* loops, barriers and atomics are not rebuilt here: such a body is left */
static int
plain_body(nid n, uint8_t *seen)
{
	uint32_t j;

	if (n == 0 || seen[n])
		return 1;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_CONST: case O_VAR: case O_UNDEF:
		return 1;
	case O_LOOP: case O_BARRIER: case O_ATOMIC:
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			if (!plain_body(N.lst[N.a[n] + j], seen))
				return 0;
		return 1;
	}
	return plain_body(N.a[n], seen) && plain_body(N.b[n], seen) &&
	    plain_body(N.c[n], seen);
}

static float
price(nid body)
{
	return fuse_expr_cost(X.m, body);
}

static struct repl *
new_repl(nid from)
{
	struct repl *r;
	int i;

	for (i = 0; i < X.nr; i++)
		if (X.r[i].from == from)
			return NULL;
	if (X.nr == MAXREPL)
		return NULL;
	r = &X.r[X.nr++];
	memset(r, 0, sizeof *r);
	r->from = from;
	return r;
}

static void
drop_repl(struct repl *r)
{
	free(r->what.p);
	free(r->holds.p);
	free(r->proof.p);
	free(r->pipe.p);
	free(r->why.p);
	X.nr--;
}

/* ---- tables as textures ---- */

struct tinfo {
	nid tab;		/* the source's table */
	int64_t n;
	int isint;		/* integers 0..q: stored as k/q */
	int levels;		/* floats lo + step k: stored as k/q */
	int q;			/* 255 or 65535: texels k/q, else 0 */
	int approx;		/* FB_: what the texels only approximate */
	int cls;
	double tmax, trep;	/* max |v|, max |v - its texel| */
	nid tex;		/* the f32 table the read takes */
	double scale, off;	/* an entry is off + scale times its texel */
	double rmax;		/* the largest scale times a texel */
};

/*
 * Evenly spaced levels: every entry lo + step m for a whole m of at most
 * q.  The texture holds k/q (m spread over 0..q when the largest m
 * divides q, so the filter works on the whole range; m itself
 * otherwise), the read is mapped back by one multiply-add, and what the
 * entries differ from that by (a few ulps at most, or it is not taken)
 * is in trep.  0: not levels.
 */
static int
table_levels(const double *v, uint32_t n, int q, double tmax, double *t,
    double *scale, double *off, double *rmax)
{
	double lo = v[0], hi = v[0], step, gap = INFINITY, worst, mmax, spread;
	uint32_t j, i2;

	for (j = 1; j < n; j++) {
		if (v[j] < lo)
			lo = v[j];
		if (v[j] > hi)
			hi = v[j];
	}
	for (j = 0; j < n; j++)		/* the smallest gap */
		for (i2 = 0; i2 < n; i2++) {
			double d = v[j] - v[i2];
			if (d > 0 && d < gap)
				gap = d;
		}
	/* first the whole range as q steps, then the smallest gap as the
	 * step */
	for (i2 = 0; i2 < 2 && hi > lo; i2++) {
		step = i2 == 0 ? (hi - lo) / q : gap;
		mmax = floor((hi - lo) / step + 0.5);
		if (mmax < 1 || mmax > q)
			continue;
		worst = 0;
		spread = fmod(q, mmax) == 0 ? q / mmax : 1;
		for (j = 0; j < n; j++) {
			double k = floor((v[j] - lo) / step + 0.5) * spread;
			double d = fabs(v[j] - (lo + step / spread * q *
			    (double)(float)(k / q)));
			if (d > worst)
				worst = d;
		}
		if (worst <= ldexp(tmax, -21)) {
			*scale = step / spread * q;
			*off = lo;
			*rmax = hi - lo;
			for (j = 0; j < n; j++)
				t[j] = (double)(float)(floor((v[j] - lo) /
				    *scale * q + 0.5) / q);
			return 1;
		}
	}
	return 0;
}

/*
 * The texture's format for a filtered read of a float table, chosen as
 * the cheapest under the model among those that hold it (#94): bytes
 * (R8_UNORM, tex_sample) for entries k/255, whole numbers 0..255 or 256
 * evenly spaced levels; 16-bit units (R16_UNORM, tex_sample_16) for
 * k/65535, 0..65535 or 65536 levels; half floats (R16_SFLOAT,
 * tex_sample_f16) for entries an f16 holds, or rounded to f16; f32
 * otherwise.  Levels and the f16 rounding only approximate the entries:
 * seed_sum drops them (X.fmtban) when the bound is over the budget.  A
 * model that lists none of them gets the bytes or f32, as it is told.
 */
/* candidate c's texels t of the entries v (table_info), 0 if it does
 * not apply: c 0 bytes of whole numbers 0..255, 1 bytes k/255, 2 256
 * evenly spaced levels, 3-5 the same in 16 bits, 6 half floats that hold
 * every entry, 7 the entries rounded to f16, 8 f32 */
static int
table_cand(int c, const double *v, uint32_t n, const int *fl, double tmax,
    double *t, struct tinfo *w)
{
	uint32_t j;

	w->scale = 1;
	w->off = 0;
	w->rmax = tmax;
	switch (c) {
	case 0: case 3:
		if (!fl[c == 0 ? 0 : 1])
			return 0;
		w->isint = 1;
		w->q = c == 0 ? 255 : 65535;
		w->scale = w->q;
		for (j = 0; j < n; j++)
			t[j] = (double)(float)(v[j] / w->scale);
		return 1;
	case 1: case 4:
		if (fl[1] || !fl[c == 1 ? 2 : 3])
			return 0;
		w->q = c == 1 ? 255 : 65535;
		for (j = 0; j < n; j++)
			t[j] = (double)(float)v[j];
		return 1;
	case 2: case 5:
		if ((c == 2 ? fl[0] || fl[2] : fl[1] || fl[3]) || (X.fmtban &
		    (c == 2 ? FB_LEV8 : FB_LEV16)) || !table_levels(v, n, c == 2 ?
		    255 : 65535, tmax, t, &w->scale, &w->off, &w->rmax))
			return 0;
		w->levels = 1;
		w->q = c == 2 ? 255 : 65535;
		w->approx = c == 2 ? FB_LEV8 : FB_LEV16;
		return 1;
	case 6: case 7:
		if (c == 6 ? !fl[4] : fl[4] || !fl[5] || (X.fmtban & FB_F16))
			return 0;
		for (j = 0; j < n; j++)
			t[j] = fround(16, v[j]);
		w->approx = c == 7 ? FB_F16 : 0;
		return 1;
	}
	for (j = 0; j < n; j++)
		t[j] = (double)(float)v[j];
	return 1;
}

static int
table_info(nid tab, struct tinfo *ti)
{
	uint32_t j, n = N.b[tab];
	double *v, *t, *bt = NULL;
	/* whole numbers 0..255, 0..65535; k/255, k/65535; f16 exact; f16
	 * range */
	static const int order0[9] = {0, 1, 3, 4, 6, 8, 2, 5, 7};
	static const int order1[4] = {0, 1, 2, 8};
	int fl[6] = {1, 1, 1, 1, 1, 1}, c, pass, i;
	float best = INFINITY, pc;
	double tmax = 0;
	nid *e;
	struct tinfo w;

	memset(ti, 0, sizeof *ti);
	v = xmalloc(n * sizeof *v);
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		if (!is_const(x)) {
			free(v);
			return 0;
		}
		v[j] = cval(x);
		if (!isfinite(v[j]) || fabs(v[j]) > 3e38) {
			free(v);
			return 0;
		}
		if (fabs(v[j]) > tmax)
			tmax = fabs(v[j]);
		if (!TY_ISINT(N.ty[x]) || v[j] < 0 || v[j] > 255)
			fl[0] = 0;
		if (!TY_ISINT(N.ty[x]) || v[j] < 0 || v[j] > 65535)
			fl[1] = 0;
		if (!unorm8((double)(float)v[j], NULL))
			fl[2] = 0;
		if (!unorm16((double)(float)v[j], NULL))
			fl[3] = 0;
		if (fround(16, v[j]) != v[j])
			fl[4] = 0;
		if (!(fabs(fround(16, v[j])) <= 65504))
			fl[5] = 0;
	}
	t = xmalloc(n * sizeof *t);
	bt = xmalloc(n * sizeof *bt);
	/*
	 * The cheapest the model lists, the exact forms first so that an
	 * approximation is taken only when it costs less; when the model
	 * lists none, the bytes or f32, the class the report says is missing
	 */
	for (pass = 0; pass < 2 && best == INFINITY; pass++)
		for (i = 0; i < (pass ? 4 : 9); i++) {
			c = pass ? order1[i] : order0[i];
			memset(&w, 0, sizeof w);
			if (!table_cand(c, v, n, fl, tmax, t, &w))
				continue;
			w.cls = tex_fmt_of(X.m, t, n, &pc);
			/* a form scaled for 8 or 16 bits or rounded for f16
			 * that only f32 prices is f32 with more error */
			if (!pass && c != 8 && w.cls == C_TEX_SAMPLE_F32)
				continue;
			if (pass) {
				w.cls = c == 8 ? C_TEX_SAMPLE_F32 :
				    C_TEX_SAMPLE;
				pc = 0;
			}
			if (!(pc < best))
				continue;
			best = pc;
			*ti = w;
			memcpy(bt, t, n * sizeof *bt);
			if (pass)
				break;
		}
	ti->tab = tab;
	ti->n = n;
	ti->tmax = tmax;
	for (j = 0; j < n; j++) {
		double d = fabs(v[j] - (ti->off + bt[j] * ti->scale));
		if (d > ti->trep)
			ti->trep = d;
	}
	e = xmalloc(n * sizeof *e);
	for (j = 0; j < n; j++)
		e[j] = kf(bt[j]);
	ti->tex = mk_table(e, n);
	free(e);
	free(v);
	free(t);
	free(bt);
	return 1;
}

/* the largest step between neighbours along x (in a row) and along y */
static void
table_steps(const struct tinfo *ti, int64_t W, double *dx, double *dy)
{
	int64_t j, n = ti->n;

	*dx = *dy = 0;
	for (j = 0; j < n; j++) {
		double v = cval(N.lst[N.a[ti->tab] + j]), d;
		if ((j + 1) % W) {
			d = fabs(cval(N.lst[N.a[ti->tab] + j + 1]) - v);
			if (d > *dx)
				*dx = d;
		}
		if (j + W < n) {
			d = fabs(cval(N.lst[N.a[ti->tab] + j + W]) - v);
			if (d > *dy)
				*dy = d;
		}
	}
}

static const char *
fmt_g(double v)
{
	static char buf[8][32];
	static int at;
	char *s = buf[at++ & 7];

	snprintf(s, 32, "%.3g", v);
	return s;
}

/* a filtered read's format, by its class */
static const char *
fmt_vk(int cls)
{
	return cls == C_TEX_SAMPLE ? "R8_UNORM" : cls == C_TEX_SAMPLE_16 ?
	    "R16_UNORM" : cls == C_TEX_SAMPLE_F16 ? "R16_SFLOAT" : "R32_SFLOAT";
}

/* what the texture holds of the table's entries */
static void
fmt_says(struct sbuf *b, const struct tinfo *ti)
{
	const char *unit = ti->cls == C_TEX_SAMPLE ? "bytes" : "16-bit units";
	int q = ti->cls == C_TEX_SAMPLE ? 255 : 65535;

	if (ti->cls == C_TEX_SAMPLE_F16) {
		sb_puts(b, ti->approx & FB_F16 ? " rounded to f16" : " as f16 "
		    "(exact)");
		return;
	}
	if (ti->cls != C_TEX_SAMPLE && ti->cls != C_TEX_SAMPLE_16) {
		sb_puts(b, " as f32");
		return;
	}
	/* texels made for the other width, which this one holds too */
	if (q != ti->q) {
		if (ti->isint)
			sb_printf(b, " as %s (the shader sees the entry / %g)",
			    unit, ti->scale);
		else if (ti->levels)
			sb_printf(b, "' levels as %s (evenly spaced: the "
			    "shader maps the texel back by %.9g + %.9g times "
			    "it)", unit, ti->off, ti->scale);
		else
			sb_printf(b, " as %s (each texel is the entry)", unit);
		return;
	}
	if (ti->isint)
		sb_printf(b, " as %s (the shader sees k/%d)", unit, q);
	else if (ti->levels)
		sb_printf(b, "' levels k as %s (evenly spaced: the shader maps "
		    "k/%d back)", unit, q);
	else
		sb_printf(b, " times %d as %s", q, unit);
}

/* what a texture class's read is, for the report */
static const char *
class_says(int k)
{
	switch (k) {
	case C_TEX_NEAREST: return "a nearest read with an address mode";
	case C_TEX_SRGB: return "an sRGB texture's decoded fetch";
	case C_TEX_LAYER: return "a read of an array texture's layer";
	case C_TEX_SAMPLE: return "a filtered read of an 8-bit normalised "
	    "texture";
	case C_TEX_SAMPLE_16: return "a filtered read of a 16-bit normalised "
	    "texture";
	case C_TEX_SAMPLE_F16: return "a filtered read of a half-float texture";
	case C_TEX_MINMAX: return "a read with a min or max reduction sampler";
	case C_TEX_FETCH: return "a texel fetch of a texture";
	case C_TEX_BUFFER: return "a texel fetch of a buffer texture";
	}
	return "a filtered read of a 32-bit float texture";
}

/* ---- address modes ---- */

static const char *
mode_vk(int op)
{
	switch (op) {
	case O_FETCH_REPEAT: return "REPEAT";
	case O_FETCH_MIRROR: return "MIRRORED_REPEAT";
	case O_FETCH_CLAMP: return "CLAMP_TO_EDGE";
	case O_FETCH_MCLAMP: return "MIRROR_CLAMP_TO_EDGE";
	}
	return "CLAMP_TO_BORDER";
}

/* the largest |i| whose coordinate (i + 1/2)/n, made in f32, still lands
 * in texel i whichever way the unit snaps it to its grid of bits */
static int64_t
index_limit(int bits)
{
	double eps = ldexp(1.0, -23) + ldexp(1.0, -47);

	return (int64_t)floor((0.5 - ldexp(1.0, -bits)) / eps - 0.5) - 1;
}

static void
fetch_format(struct sbuf *b, nid tab)
{
	uint32_t j;
	int64_t lo = 0, hi = 0;

	if (TY_FP(N.ty[tab])) {
		/* half the bytes when an f16 holds every entry: a half float
		 * widens to f32 exactly */
		for (j = 0; j < N.b[tab]; j++) {
			nid x = N.lst[N.a[tab] + j];
			if (!is_const(x) || fround(16, cval(x)) != cval(x))
				break;
		}
		sb_puts(b, j == N.b[tab] ? "R16_SFLOAT (each entry exact as an "
		    "f16)" : "R32_SFLOAT (each entry as its f32)");
		return;
	}
	for (j = 0; j < N.b[tab]; j++) {
		nid x = N.lst[N.a[tab] + j];
		if (!is_const(x)) {
			sb_puts(b, "R32_SINT");
			return;
		}
		if (N.ki[x] < lo)
			lo = N.ki[x];
		if (N.ki[x] > hi)
			hi = N.ki[x];
	}
	sb_puts(b, lo >= 0 && hi <= 255 ? "R8_UINT" : lo >= 0 && hi <= 65535 ?
	    "R16_UINT" : lo >= -128 && hi <= 127 ? "R8_SINT" : lo >= -32768 &&
	    hi <= 32767 ? "R16_SINT" : lo >= 0 ? "R32_UINT" : "R32_SINT");
}

/* e is x + c (or x - c) with c a constant in -8..7, a ConstOffset's
 * range: c (0 if not) */
static int
const_add(nid e, nid *x)
{
	int64_t c;

	if (N.op[e] == O_ADD && is_const(N.b[e])) {
		*x = N.a[e];
		c = N.ki[N.b[e]];
	} else if (N.op[e] == O_ADD && is_const(N.a[e])) {
		*x = N.b[e];
		c = N.ki[N.a[e]];
	} else if (N.op[e] == O_SUB && is_const(N.b[e])) {
		*x = N.a[e];
		c = -N.ki[N.b[e]];
	} else
		return 0;
	if (!TY_ISINT(N.ty[*x]) || !TY_ISINT(N.ty[e]) || c < -8 || c > 7)
		return 0;
	return (int)c;
}

static const char *
offtext(int c)
{
	static char b[16];

	snprintf(b, sizeof b, ", %d", c);
	return b;
}

/* idx is the REPEAT or MIRRORED_REPEAT of an index x over n texels,
 * written out (x % n; min(x % 2n, 2n - 1 - x % 2n)): the mode, x in *x;
 * 0 if not.  Where the read is defined the truncated remainder is the
 * floored one (mode_shape) */
static int mode_shape(nid idx, nid x, int mode, int64_t n);

static int
wrap_of(nid idx, int64_t n, nid *x)
{
	nid a;

	if (N.op[idx] == O_MOD && is_const(N.b[idx]) && N.ki[N.b[idx]] == n &&
	    TY_ISINT(N.ty[idx])) {
		*x = N.a[idx];
		return O_FETCH_REPEAT;
	}
	if (N.op[idx] != O_MIN)
		return 0;
	a = N.op[N.a[idx]] == O_MOD ? N.a[idx] : N.b[idx];
	if (N.op[a] != O_MOD || !mode_shape(idx, N.a[a], O_FETCH_MIRROR, n))
		return 0;
	*x = N.a[a];
	return O_FETCH_MIRROR;
}

/* subterms of an index expression, the expression first (bounded) */
static int
subterms(nid e, nid *out, int n, int max, int depth)
{
	int i;

	if (e == 0 || is_const(e) || !TY_ISINT(N.ty[e]) || depth > 5)
		return n;
	for (i = 0; i < n; i++)
		if (out[i] == e)
			return n;
	if (n < max)
		out[n++] = e;
	if (N.op[e] == O_VAR || N.op[e] == O_INDEX || N.op[e] == O_TABLE)
		return n;
	n = subterms(N.a[e], out, n, max, depth + 1);
	n = subterms(N.b[e], out, n, max, depth + 1);
	return subterms(N.c[e], out, n, max, depth + 1);
}

/*
 * A 32-bit index too far for its coordinate, i = h 2^16 + l with h = i >>
 * 16 (arithmetic for a signed i) and l = i & 0xffff: for a period p (n
 * for REPEAT, 2n for MIRROR) up to 2^16, h r + l with r = 2^16 mod p is
 * congruent to i mod p, so a REPEAT or MIRROR read takes the same texel
 * from it; it is the index when its range, from i's type, stays within
 * lim (r below about 64 at 8 bits of sub-texel).  0: no such index.
 */
/* x % k with k a constant (ucas's %: truncated), or 0 */
static int
mod_of(nid e, nid x, int64_t k)
{
	return N.op[e] == O_MOD && N.a[e] == x && is_const(N.b[e]) &&
	    N.ki[N.b[e]] == k;
}

/*
 * idx is the mode written out on x: x % n for REPEAT, min(x % 2n, 2n - 1 -
 * x % 2n) for MIRROR.  Where the read is defined (idx >= 0) the truncated
 * remainder is the floored one (it is negative only for a negative x not
 * a multiple), so idx = mode(x) there: the identity by its definition,
 * which SAT would have to find through two dividers.
 */
static int
mode_shape(nid idx, nid x, int mode, int64_t n)
{
	nid a, b;

	if (mode == O_FETCH_REPEAT)
		return mod_of(idx, x, n);
	if (mode != O_FETCH_MIRROR || N.op[idx] != O_MIN)
		return 0;
	a = N.a[idx];
	b = N.b[idx];
	if (!mod_of(a, x, 2 * n)) {
		nid t = a;
		a = b;
		b = t;
	}
	return mod_of(a, x, 2 * n) && N.op[b] == O_SUB && is_const(N.a[b]) &&
	    N.ki[N.a[b]] == 2 * n - 1 && N.b[b] == a;
}

static nid
reduce_index(nid i, int64_t per, int64_t lim, int64_t *r, int64_t *lo,
    int64_t *hi)
{
	ty_t t = N.ty[i];
	int sg = TYK(t) == K_SINT;

	if ((TYK(t) != K_UINT && !sg) || TYW(t) != 32 || per > 65536)
		return 0;
	*r = 65536 % per;
	*lo = sg ? -32768 * *r : 0;
	*hi = (sg ? 32767 : 65535) * *r + 65535;
	if (*lo < -lim || *hi > lim)
		return 0;
	if (*r == 0)
		return mk_bin(O_AND, i, mk_const_i(t, 65535));
	return mk_bin(O_ADD, mk_bin(O_MUL, mk_bin(O_SHR, i, mk_const_i(t, 16)),
	    mk_const_i(t, *r)), mk_bin(O_AND, i, mk_const_i(t, 65535)));
}

/*
 * A texel fetch of the table at an integer index: texelFetch of a texture
 * (fetch_texel, priced tex_fetch) or of a buffer texture (fetch_buffer,
 * priced tex_buffer).  No sampler, no address mode, no coordinate to
 * make: it is T[i], undefined outside the table as the read is.  The
 * cheaper of the two the model lists as `from`'s replacement in body, 0
 * when it lists neither.
 */
static nid
texel_best(nid body, nid from, nid tab, nid idx, ty_t t, float *cost)
{
	static const int ops[2] = {O_FETCH_TEXEL, O_FETCH_BUFFER};
	nid best = 0;
	int k;

	*cost = INFINITY;
	for (k = 0; k < 2; k++) {
		nid c;
		float p;
		if (!model_lists(X.m, ops[k] == O_FETCH_TEXEL ? C_TEX_FETCH :
		    C_TEX_BUFFER))
			continue;
		c = mk_tex(ops[k], 0, tab, idx, 0);
		if (N.ty[c] != t)
			c = mk_cast(t, c);
		p = price(swap_nodes(body, &from, &c, 1));
		if (p < *cost) {
			*cost = p;
			best = c;
		}
	}
	return best;
}

/* what the pipeline needs for texel_best's read of tab at index ix */
static void
texel_pipe(struct sbuf *b, nid to, nid tab, const char *what, const char *ix)
{
	long long n = (long long)N.b[tab];

	while (N.op[to] == O_CAST)
		to = N.a[to];
	if (N.op[to] == O_FETCH_BUFFER) {
		sb_puts(b, "a buffer view ");
		fetch_format(b, tab);
		sb_printf(b, " over a buffer of %lld texels, %s%s\none uniform "
		    "texel buffer (samplerBuffer) more in the layout, no sampler; "
		    "the shader reads texelFetch(b, int(%s)).r", n, what, n >
		    65536 ? " (more than the 65536 texels every Vulkan device "
		    "takes: it needs that maxTexelBufferElements)" : "", ix);
		return;
	}
	sb_puts(b, "a texture ");
	fetch_format(b, tab);
	sb_printf(b, ", %lld x 1, one level, its texels %s%s\nno sampler; the "
	    "shader reads texelFetch(t, ivec2(int(%s), 0), 0).r", n, what, n >
	    4096 ? " (wider than the 4096 texels every Vulkan device takes in a "
	    "row: it needs that maxImageDimension2D)" : "", ix);
}

/*
 * A table read T[idx], or a select between one and a constant, as a
 * nearest read with an address mode: for a subterm i of the index and a
 * mode, idx = mode(i) wherever the read is defined.  That identity is
 * between integers and is proved by enumeration or SAT.  A subterm too far
 * for the coordinate is read through reduce_index's smaller index, and
 * the identity proved is idx = mode of that index.
 */
static void
seed_fetch(nid body, nid n)
{
	nid tab, idx, sub[16], sel = 0, kb = 0;
	int ns, i, mode, bestmode = 0, avail;
	int64_t cnt, lo, hi, lim;
	struct repl *r;
	nid best = 0, besti = 0, bestfar = 0;
	int64_t bestr = 0, bestlo = 0, besthi = 0;
	float bc = INFINITY;
	struct eqproof bpr, pr;
	ty_t t64 = TY(K_SINT, 64);
	nid ox, bat, tb = 0;
	int off, boff = 0;
	float ctb = INFINITY;

	if (N.op[n] == O_SEL) {
		/* c ? T[i] : k, or c ? k : T[i] */
		sel = n;
		if (N.op[N.b[n]] == O_INDEX && is_const(N.c[n])) {
			n = N.b[sel];
			kb = N.c[sel];
		} else if (N.op[N.c[n]] == O_INDEX && is_const(N.b[n])) {
			n = N.c[sel];
			kb = N.b[sel];
		} else
			return;
	}
	tab = N.a[n];
	idx = N.b[n];
	cnt = N.b[tab];
	if (cnt < 1 || cnt > 0x3fffff)
		return;
	/* a model that cannot do it is told so only when there is an
	 * identity to tell about: the search runs without prices then */
	avail = model_lists(X.m, C_TEX_NEAREST) && X.bits;
	lim = X.bits ? index_limit(X.bits) : 0x3fffff;
	/* the read as it is, by a texel fetch, where that pays: one that
	 * does not is not a replacement, and what is inside the read is
	 * still looked at */
	if (!sel)
		tb = texel_best(body, n, tab, idx, N.ty[n], &ctb);
	if (tb && !(ctb < price(body)))
		tb = 0;
	ns = subterms(idx, sub, 0, 16, 0);
	memset(&bpr, 0, sizeof bpr);
	for (i = 0; i < ns; i++) {
		/* an index whose range is unknown or beyond what f32 carries
		 * to a texel: tried without a price, to say so */
		int far0 = !irange(sub[i], &lo, &hi) || lo < -lim || hi > lim;
		for (mode = O_FETCH_REPEAT; mode <= O_FETCH_BORDER; mode++) {
			nid cand, nb, a, b, at = sub[i];
			float c;
			int far = far0;
			int64_t rr = 0, rlo = 0, rhi = 0;
			if (far && sub[i] != idx && !sel && avail &&
			    mode_shape(idx, sub[i], mode, cnt)) {
				nid rd = reduce_index(sub[i], mode == O_FETCH_MIRROR ?
				    2 * cnt : cnt, lim, &rr, &rlo, &rhi);
				if (rd) {
					at = rd;
					far = 0;
				}
			}
			if (far && (X.far.n || sub[i] == idx))
				continue;
			if ((mode == O_FETCH_BORDER) != (sel != 0))
				continue;
			/* the plain read T[i] is every mode: the clamp stands */
			if (sub[i] == idx && !sel && (mode != O_FETCH_CLAMP ||
			    !avail))
				continue;
			if (sel && sub[i] != idx)
				continue;
			cand = mk_tex(mode, 0, tab, at, kb);
			nb = swap_nodes(body, sel ? &sel : &n, &cand, 1);
			c = avail && !far ? price(nb) : 0;
			if (avail && !far && !(c < bc))
				continue;
			if (sel) {
				/* the select against the read's definition */
				a = sel;
				b = tex_expand(cand);
				if (!TY_ISINT(N.ty[a])) {
					/* float texels: the condition is the range */
					a = mk_cast(t64, N.a[sel]);
					b = mk_bin(O_LAND, mk_bin(O_GE, mk_cast(t64,
					    idx), mk_const_i(t64, 0)), mk_bin(O_LT,
					    mk_cast(t64, idx), mk_const_i(t64, cnt)));
					if (kb == N.b[sel])
						b = mk_un(O_LNOT, b);
					b = mk_cast(t64, b);
				}
			} else {
				nid in = mk_bin(O_LAND, mk_bin(O_GE, mk_cast(t64,
				    idx), mk_const_i(t64, 0)), mk_bin(O_LT,
				    mk_cast(t64, idx), mk_const_i(t64, cnt)));
				a = mk_sel(in, mk_cast(t64, idx), mk_undef(t64));
				b = wrap_expr(mode, at, cnt);
			}
			if (!avail) {
				/* only to say the model cannot: compared on points */
				int smp;
				if (!sel)	/* undefined where the read is */
					b = mk_sel(N.a[a], b, mk_undef(t64));
				if (far || equiv_check(X.f, a, b, NULL, &smp))
					continue;
				if (!model_lists(X.m, C_TEX_NEAREST))
					X.unavail[C_TEX_NEAREST] = 1;
				else
					X.nobits = 1;
				if (!tb)
					return;
				i = ns;		/* no address mode: the fetch alone */
				break;
			}
			if (far) {
				/* only to say why not: compared on points, no proof */
				int smp;
				if (!sel)	/* undefined where the read is */
					b = mk_sel(N.a[a], b, mk_undef(t64));
				if (!X.far.n && !equiv_check(X.f, a, b, NULL, &smp))
					sb_printf(&X.far, "%s agrees with %s on the "
					    "points tried, but %s is not known to stay "
					    "within +-%lld, where its coordinate (i + "
					    "1/2)/n is still exact enough in f32: give "
					    "it a range", node_str(sel ? sel : n),
					    node_str(cand), node_str(sub[i]),
					    (long long)lim);
				continue;
			}
			if (at != sub[i]) {
				/* derived (mode_shape, then the congruence), and
				 * compared on points as a second opinion */
				int smp;
				if (!sel)
					b = mk_sel(N.a[a], b, mk_undef(t64));
				if (equiv_check(X.f, a, b, NULL, &smp))
					continue;
				memset(&pr, 0, sizeof pr);
				pr.how = -1;
			} else if (equiv_decide(X.f, a, b, NULL, &pr) ||
			    pr.how == EQ_SAMPLED)
				continue;
			best = cand;
			besti = at;
			bestfar = at != sub[i] ? sub[i] : 0;
			bestr = rr;
			bestlo = rlo;
			besthi = rhi;
			bestmode = mode;
			bc = c;
			bpr = pr;
		}
	}
	/*
	 * An index x + c with c in -8..7 is the read of x with a ConstOffset
	 * of c (#99): the unit adds c to the unnormalised coordinate before
	 * the floor and the address mode, so the add goes, where x + c does
	 * not wrap in its type.  With |x| + 8 below 2^21 the unit's add keeps
	 * the texel (an ulp of the sum at most 2^-3).
	 */
	bat = besti;
	if (best && !bestfar && avail && (off = const_add(besti, &ox)) != 0 &&
	    irange(besti, &lo, &hi) &&	/* x + c does not wrap */
	    irange(ox, &lo, &hi) && (hi > -lo ? hi : -lo) + 8 <= lim &&
	    (hi > -lo ? hi : -lo) + 8 <= 2097152) {
		nid c2 = mk_tex(bestmode, TEX_KOFF(off), tab, ox, kb);
		float p2 = price(swap_nodes(body, sel ? &sel : &n, &c2, 1));
		if (p2 < bc) {
			best = c2;
			besti = ox;
			bc = p2;
			boff = off;
		}
	}
	if (tb && (!best || ctb < bc)) {
		/* the texel fetch is the read itself: nothing to prove */
		if ((r = new_repl(n)) == NULL)
			return;
		r->to = tb;
		r->cost = ctb;
		r->exact = 1;
		r->ok = 1;
		r->plain = 1;
		sb_printf(&r->what, "%s is %s", node_str(n), node_str(tb));
		sb_printf(&r->holds, "holds wherever the read is defined: a "
		    "texel fetch takes the integer index as it is, with no "
		    "sampler and no address mode, and outside 0..%lld both are "
		    "undefined", (long long)(cnt - 1));
		sb_puts(&r->proof, "the texel fetch is the table read by "
		    "definition: no coordinate is made, nothing rounds");
		texel_pipe(&r->pipe, tb, tab, "the table's entries",
		    node_str(idx));
		return;
	}
	if (!best || (r = new_repl(sel ? sel : n)) == NULL)
		return;
	r->to = best;
	r->cost = bc;
	r->exact = 1;
	r->ok = 1;
	r->plain = bat == idx && !sel;
	if (bestfar) {
		lo = bestlo;
		hi = besthi;
	} else
		irange(besti, &lo, &hi);
	sb_printf(&r->what, "%s is %s", node_str(sel ? sel : n), node_str(best));
	sb_puts(&r->holds, "holds where the read is defined: there ");
	switch (bestmode) {
	case O_FETCH_REPEAT:
		sb_printf(&r->holds, "%s = %s%s%s mod %lld", node_str(idx),
		    bestfar ? "(" : "", node_str(bat), bestfar ? ")" : "",
		    (long long)cnt);
		break;
	case O_FETCH_MIRROR:
		sb_printf(&r->holds, "%s = min(r, %lld - r) with r = %s%s%s mod "
		    "%lld", node_str(idx), (long long)(2 * cnt - 1), bestfar ?
		    "(" : "", node_str(bat), bestfar ? ")" : "",
		    (long long)(2 * cnt));
		break;
	case O_FETCH_CLAMP:
		if (bat == idx)
			sb_printf(&r->holds, "%s is in 0..%lld, where every address "
			    "mode is the plain read", node_str(idx),
			    (long long)(cnt - 1));
		else
			sb_printf(&r->holds, "%s = min(max(%s, 0), %lld)",
			    node_str(idx), node_str(bat), (long long)(cnt - 1));
		break;
	case O_FETCH_MCLAMP:
		sb_printf(&r->holds, "%s = min(max(%s, -1 - (%s)), %lld)",
		    node_str(idx), node_str(bat), node_str(bat),
		    (long long)(cnt - 1));
		break;
	default:
		sb_printf(&r->holds, "the condition is 0 <= %s < %lld, the other "
		    "arm the border's value", node_str(bat), (long long)cnt);
	}
	if (boff)
		sb_printf(&r->holds, "\nthe read takes %s with a ConstOffset of "
		    "%d (Vulkan's minTexelOffset is at most -8 and its "
		    "maxTexelOffset at least 7): the unit adds it before the "
		    "address mode, the add goes", node_str(besti), boff);
	if (bestfar)
		sb_printf(&r->holds, "\n%s itself is not known to stay within "
		    "+-%lld, where its coordinate is exact enough in f32; 2^16 = "
		    "%lld mod %lld, so %s is congruent to it and in [%lld, %lld] "
		    "(from %s's type): the read takes that", node_str(bestfar),
		    (long long)lim, (long long)bestr, (long long)(bestmode ==
		    O_FETCH_MIRROR ? 2 * cnt : cnt), node_str(besti),
		    (long long)lo, (long long)hi, node_str(bestfar));
	if (bpr.how == -1)
		sb_printf(&r->proof, "that identity is derived, not by SAT (a "
		    "divider against a divider is beyond it): %s is the mode "
		    "written out on %s, and where the read is defined (the index "
		    "at least 0) the truncated remainder is the floored one; %s "
		    "= 2^16 h + l exactly (h = %s >> 16, floored; l = %s & "
		    "65535), and 2^16 = %lld mod %lld, so h %lld + l is congruent "
		    "to it, and the mode reads a residue alone; compared on the "
		    "points tried, the same", node_str(idx), node_str(bestfar),
		    node_str(bestfar), node_str(bestfar), node_str(bestfar),
		    (long long)bestr, (long long)(bestmode == O_FETCH_MIRROR ?
		    2 * cnt : cnt), (long long)bestr);
	else {
		sb_puts(&r->proof, "that identity between integers is proved: ");
		eqproof_text(&r->proof, &bpr);
	}
	sb_printf(&r->proof, "\nthe coordinate (i + 1/2)/n is made in f32: within "
	    "(|i| + 1/2) 2^-23 <= %s of a texel's centre for i in [%lld, %lld] "
	    "(its range), and the unit snaps by less than 2^-%d: %s + %s < "
	    "1/2, the texel is i's", fmt_g(((double)(hi > -lo ? hi : -lo) +
	    0.5) * ldexp(1.0, -23)), (long long)lo, (long long)hi, X.bits,
	    fmt_g(((double)(hi > -lo ? hi : -lo) + 0.5) * ldexp(1.0, -23)),
	    fmt_g(ldexp(1.0, -X.bits)));
	sb_printf(&r->pipe, "a texture ");
	fetch_format(&r->pipe, tab);
	sb_printf(&r->pipe, ", %lld x 1, one level, its texels the table's "
	    "entries\na sampler: NEAREST filters, %s%s, normalised coordinates, "
	    "lod 0\none combined image sampler more in the layout; the shader "
	    "reads textureLod%s(s, (float(%s) + 0.5) * (1.0 / %lld.0), 0.0%s).r"
	    "%s%s",
	    (long long)cnt, mode_vk(bestmode), bestmode == O_FETCH_BORDER ?
	    (const_is(kb, 0) ? " with a transparent black border" :
	    " with the border's value as a custom border colour "
	    "(VK_EXT_custom_border_color)") : bestmode == O_FETCH_MCLAMP ?
	    " (samplerMirrorClampToEdge)" : "", boff ? "Offset" : "", bestfar ?
	    "j" : boff ? node_str(besti) : "i", (long long)cnt, boff ?
	    offtext(boff) : "", bestfar ? ", j = " : "", bestfar ?
	    node_str(besti) : "");
}

/* ---- array layers ---- */

/*
 * The layer a read of an array texture selects (#90): clamp(RNE(z), 0,
 * L - 1), the specification's RNE either roundTiesToEven (preferred) or
 * floor(z + 1/2) (allowed).  A clamp to 0..L-1 of roundeven(z) or of
 * floor(z + 0.5), as a float or through a conversion to an integer, is
 * fetch_layer of L layers that hold 0..L-1 (the identity), and a table
 * read with that index is fetch_layer of the table's first L entries.
 * The two readings differ at z = k + 1/2 for an even k in 0..L-2 (one
 * takes k, the other k + 1; past the ends both clamp), and floor(z + 0.5)
 * made in f32 differs from both at z = 0.5 - 2^-25, where the add rounds
 * up to 1: a rewrite holds whichever reading the unit has only where z
 * takes none of those values.
 */

/* r rounds z as a layer is selected: 1 roundeven(z), 2 floor(z + 0.5) in
 * f32; 0 not */
static int
layer_round(nid r, nid *z)
{
	nid s;

	if (N.ty[r] != TY_F32)
		return 0;
	if (N.op[r] == O_ROUNDEVEN) {
		*z = N.a[r];
		return 1;
	}
	if (N.op[r] != O_FLOOR || N.op[s = N.a[r]] != O_ADD)
		return 0;
	if (is_const(N.b[s]) && cval(N.b[s]) == 0.5) {
		*z = N.a[s];
		return 2;
	}
	if (is_const(N.a[s]) && cval(N.a[s]) == 0.5) {
		*z = N.b[s];
		return 2;
	}
	return 0;
}

/* e is min(max(x, lo), hi) or max(min(x, hi), lo), constant ends */
static int
clamp_of(nid e, nid *x, double *lo, double *hi)
{
	int op = N.op[e], inner = op == O_MIN ? O_MAX : O_MIN;
	nid in, k1, k2;

	if (op != O_MIN && op != O_MAX)
		return 0;
	if (is_const(N.b[e])) {
		in = N.a[e];
		k1 = N.b[e];
	} else if (is_const(N.a[e])) {
		in = N.b[e];
		k1 = N.a[e];
	} else
		return 0;
	if (N.op[in] != inner)
		return 0;
	if (is_const(N.b[in])) {
		*x = N.a[in];
		k2 = N.b[in];
	} else if (is_const(N.a[in])) {
		*x = N.b[in];
		k2 = N.a[in];
	} else
		return 0;
	*lo = cval(op == O_MIN ? k2 : k1);
	*hi = cval(op == O_MIN ? k1 : k2);
	return *lo <= *hi;
}

/*
 * e selects a layer of z among *L: a clamp to 0..L-1 of roundeven(z) or
 * floor(z + 0.5), in float, converted to an integer after it, or of the
 * rounding converted first (*conv: that conversion, which can wrap).  The
 * rounding (1, 2); 0 if e is not one.
 */
static int
layer_form(nid e, nid *z, int64_t *L, nid *conv)
{
	nid x;
	double lo, hi;

	*conv = 0;
	if (N.op[e] == O_CAST && TY_ISINT(N.ty[e]) && N.ty[N.a[e]] == TY_F32)
		e = N.a[e];
	if (!clamp_of(e, &x, &lo, &hi) || lo != 0 || !(hi >= 1) ||
	    hi > 65535 || hi != floor(hi))
		return 0;
	if (N.op[x] == O_CAST && TY_ISINT(N.ty[x]) && N.ty[N.a[x]] == TY_F32) {
		*conv = x;
		x = N.a[x];
	}
	*L = (int64_t)hi + 1;
	return layer_round(x, z);
}

/*
 * Is z a whole number wherever it is defined: an integer converted, a
 * rounding, a whole constant, or a sum, difference, product, negation,
 * min, max or select of those (an f32 result rounded from a whole number
 * is whole: past 2^24 every f32 is).  *src: the integer converted when z
 * is just that, else 0.
 */
static int
whole1(nid z, int depth)
{
	if (depth > 12 || N.ty[z] != TY_F32)
		return 0;
	switch (N.op[z]) {
	case O_CONST:
		return N.kr[z] == floor(N.kr[z]);
	case O_CAST:
		return TY_ISINT(N.ty[N.a[z]]);
	case O_FLOOR: case O_CEIL: case O_ROUNDEVEN:
		return 1;
	case O_NEG: case O_ABS:
		return whole1(N.a[z], depth + 1);
	case O_ADD: case O_SUB: case O_MUL: case O_MIN: case O_MAX:
		return whole1(N.a[z], depth + 1) && whole1(N.b[z], depth + 1);
	case O_SEL:
		return whole1(N.b[z], depth + 1) && whole1(N.c[z], depth + 1);
	}
	return 0;
}

/* an f32 subterm's range by intervals: constants, ranged parameters,
 * integers converted (their proved range), +, -, *, neg, abs, min, max,
 * selects and roundings (a whole number at most 1 away); 0: not known */
static int
zrange(nid z, double *lo, double *hi, int depth)
{
	double al, ah, bl, bh, c[4];
	int64_t il, ih;
	int k;

	if (depth > 12)
		return 0;
	switch (N.op[z]) {
	case O_CONST:
		*lo = *hi = cval(z);
		return isfinite(*lo);
	case O_VAR:
		return frange(z, lo, hi);
	case O_CAST:
		if (TY_ISINT(N.ty[N.a[z]])) {
			if (!irange(N.a[z], &il, &ih))
				return 0;
			*lo = (double)il;
			*hi = (double)ih;
			return 1;
		}
		return TY_FP(N.ty[N.a[z]]) && zrange(N.a[z], lo, hi, depth + 1);
	case O_FLOOR: case O_CEIL: case O_ROUNDEVEN:
		if (!zrange(N.a[z], &al, &ah, depth + 1))
			return 0;
		*lo = floor(al);
		*hi = ceil(ah);
		return 1;
	case O_NEG:
		if (!zrange(N.a[z], &al, &ah, depth + 1))
			return 0;
		*lo = -ah;
		*hi = -al;
		return 1;
	case O_ABS:
		if (!zrange(N.a[z], &al, &ah, depth + 1))
			return 0;
		*lo = al > 0 ? al : ah < 0 ? -ah : 0;
		*hi = fabs(al) > fabs(ah) ? fabs(al) : fabs(ah);
		return 1;
	case O_SEL:
		if (!zrange(N.b[z], &al, &ah, depth + 1) || !zrange(N.c[z], &bl,
		    &bh, depth + 1))
			return 0;
		*lo = al < bl ? al : bl;
		*hi = ah > bh ? ah : bh;
		return 1;
	case O_ADD: case O_SUB: case O_MUL: case O_MIN: case O_MAX:
		if (!zrange(N.a[z], &al, &ah, depth + 1) || !zrange(N.b[z], &bl,
		    &bh, depth + 1))
			return 0;
		break;
	default:
		return 0;
	}
	switch (N.op[z]) {
	case O_ADD:
		*lo = al + bl;
		*hi = ah + bh;
		break;
	case O_SUB:
		*lo = al - bh;
		*hi = ah - bl;
		break;
	case O_MIN:
		*lo = al < bl ? al : bl;
		*hi = ah < bh ? ah : bh;
		break;
	case O_MAX:
		*lo = al > bl ? al : bl;
		*hi = ah > bh ? ah : bh;
		break;
	default:
		c[0] = al * bl;
		c[1] = al * bh;
		c[2] = ah * bl;
		c[3] = ah * bh;
		*lo = *hi = c[0];
		for (k = 1; k < 4; k++) {
			if (c[k] < *lo)
				*lo = c[k];
			if (c[k] > *hi)
				*hi = c[k];
		}
	}
	/* the f32 result is rounded: widen by an ulp of the larger end */
	if (*lo != floor(*lo) || *hi != floor(*hi) || fabs(*lo) > 16777216 ||
	    fabs(*hi) > 16777216) {
		double u = ldexp(fabs(*lo) > fabs(*hi) ? fabs(*lo) : fabs(*hi),
		    -23);
		*lo -= u;
		*hi += u;
	}
	return isfinite(*lo) && isfinite(*hi);
}

static int
whole_valued(nid z, nid *src)
{
	*src = N.op[z] == O_CAST && TY_ISINT(N.ty[N.a[z]]) ? N.a[z] : 0;
	return whole1(z, 0);
}

/*
 * Where the readings and the form part on z's values: 0 when z takes
 * none of them (*how says why), else 1 with *at the first value z may
 * take there.
 */
static int
layer_ties(nid z, int form, int64_t L, double *at, struct sbuf *how)
{
	nid src;
	double lo, hi, t;
	int64_t k;

	if (whole_valued(z, &src)) {
		sb_printf(how, "%s is a whole number (%s), none of them",
		    node_str(z), src ? "an integer converted" : "made of "
		    "integers converted, roundings and whole constants");
		return 0;
	}
	if (zrange(z, &lo, &hi, 0)) {
		if (form == 2 && lo <= 0.5 - ldexp(1.0, -25) &&
		    0.5 - ldexp(1.0, -25) <= hi) {
			*at = 0.5 - ldexp(1.0, -25);
			return 1;
		}
		for (k = 0; k + 1 < L; k += 2) {
			t = (double)k + 0.5;
			if (lo <= t && t <= hi) {
				*at = t;
				return 1;
			}
		}
		sb_printf(how, "%s is in [%.9g, %.9g] (its range), which holds "
		    "none of them", node_str(z), lo, hi);
		return 0;
	}
	*at = form == 2 ? 0.5 - ldexp(1.0, -25) : 0.5;
	return 1;
}

/* the conversion conv of a rounding of z does not wrap on z's values:
 * 0 if it may (or it is not known not to) */
static int
layer_conv_ok(nid conv, nid z, struct sbuf *how)
{
	ty_t t = N.ty[conv];
	double tlo, thi, lo, hi;

	if (!conv)
		return 1;
	if (TYK(t) == K_INT)
		return 1;
	tlo = TYK(t) == K_SINT ? -ldexp(1.0, TYW(t) - 1) : 0;
	thi = TYK(t) == K_SINT ? ldexp(1.0, TYW(t) - 1) - 1 : ldexp(1.0,
	    TYW(t)) - 1;
	if (!zrange(z, &lo, &hi, 0))
		return 0;
	/* the rounding moves a value by at most 1/2 */
	if (!(lo - 0.5 >= tlo && hi + 0.5 <= thi))
		return 0;
	sb_printf(how, "; %s does not wrap: %s is in [%.9g, %.9g]",
	    node_str(conv), node_str(z), lo, hi);
	return 1;
}

static int
seed_layer(nid body, nid n)
{
	nid tab = 0, idx = n, z, conv, *e, cand, nb;
	int64_t L, k, cnt = 0;
	int form, smp;
	double at;
	struct repl *r;
	struct sbuf how = {0}, wit = {0};

	if (N.op[n] == O_INDEX) {
		tab = N.a[n];
		idx = N.b[n];
		cnt = N.b[tab];
		if (!TY_ISINT(N.ty[idx]))
			return 0;
	}
	if (!(form = layer_form(idx, &z, &L, &conv)) || (tab && L > cnt) ||
	    is_const(z))
		return 0;
	e = xmalloc((size_t)L * sizeof *e);
	for (k = 0; k < L; k++)
		e[k] = tab ? N.lst[N.a[tab] + (uint32_t)k] : TY_FP(N.ty[n]) ?
		    kf((double)k) : mk_const_i(N.ty[n], k);
	cand = mk_tex(O_FETCH_LAYER, 0, mk_table(e, (uint32_t)L), z, 0);
	free(e);
	if (!model_lists(X.m, C_TEX_LAYER)) {
		X.unavail[C_TEX_LAYER] = 1;
		return 0;
	}
	/* not taken: said when nothing else is (a fetch_ read of the
	 * rounded index may still be) */
	if (layer_ties(z, form, L, &at, &how)) {
		sb_printf(&X.nolayer, "%s%s would be %s, but %s may be %.9g, "
		    "where %s", X.nolayer.n ? "\n" : "", node_str(n),
		    node_str(cand), node_str(z), at, at == 0.5 - ldexp(1.0,
		    -25) ? "floor(z + 0.5) in f32 is 1 (the add rounds up) and "
		    "the unit takes layer 0 either way" : "the unit takes layer "
		    "0 with roundTiesToEven (the specification's preferred "
		    "RNE) or 1 with floor(z + 1/2) (allowed): it needs a range "
		    "without those values");
		free(how.p);
		return 0;
	}
	if (!layer_conv_ok(conv, z, &how)) {
		sb_printf(&X.nolayer, "%s%s would be %s, but %s can wrap on "
		    "the values %s takes, where the clamp no longer gives the "
		    "unit's layer: it needs a range", X.nolayer.n ? "\n" : "",
		    node_str(n), node_str(cand), node_str(conv), node_str(z));
		free(how.p);
		return 0;
	}
	if (equiv_check(X.f, n, cand, &wit, &smp)) {
		sb_printf(&X.nolayer, "%s%s would be %s by the derivation, "
		    "but they differ on a point tried (%s): not taken",
		    X.nolayer.n ? "\n" : "", node_str(n), node_str(cand),
		    sb_str(&wit));
		free(how.p);
		free(wit.p);
		return 0;
	}
	free(wit.p);
	if ((r = new_repl(n)) == NULL) {
		free(how.p);
		return 0;
	}
	nb = swap_nodes(body, &n, &cand, 1);
	r->to = cand;
	r->exact = 1;
	r->ok = 1;
	r->cost = price(nb);
	sb_printf(&r->what, "%s is %s", node_str(n), node_str(cand));
	sb_printf(&r->holds, "holds wherever %s is defined: the layer the unit "
	    "selects is clamp(RNE(%s), 0, %lld)", node_str(n), node_str(z),
	    (long long)(L - 1));
	sb_printf(&r->proof, "the specification's RNE is roundTiesToEven "
	    "(preferred) or floor(z + 1/2) (allowed); %s is the %s, and the "
	    "two differ only at z = k + 1/2 for an even k in 0..%lld%s; %s; "
	    "the same layer either way%s", form == 1 ? "roundeven(z)" :
	    "floor(z + 0.5)", form == 1 ? "first" : "second", (long long)(L - 2),
	    form == 2 ? " (and floor(z + 0.5) made in f32 at z = 0.5 - 2^-25, "
	    "where the add rounds up to 1)" : "", sb_str(&how), tab ? "" :
	    "; layer l holds l");
	sb_puts(&r->proof, "\ncompared on the points tried as a second opinion: "
	    "the same");
	sb_puts(&r->pipe, "a texture ");
	fetch_format(&r->pipe, N.a[cand]);
	sb_printf(&r->pipe, ", 1 x 1, %lld layers (an array view, "
	    "VK_IMAGE_VIEW_TYPE_1D_ARRAY), layer l holding %s\na sampler: "
	    "NEAREST filters, CLAMP_TO_EDGE, lod 0\none combined image sampler "
	    "more in the layout; the shader reads textureLod(s, vec2(0.5, %s), "
	    "0.0).r (the unit rounds and clamps the layer)", (long long)L, tab ?
	    "the table's entry l" : "l", node_str(z));
	free(how.p);
	return 1;
}

/* ---- packed formats ---- */

/*
 * A packed texel format's channel decoded by hand (#92): a float subterm
 * whose one input is a 32-bit word parameter w, that reads no bit of w
 * outside one channel's fields of B10G11R11_UFLOAT, E5B9G9R9_UFLOAT or
 * A2B10G10R10_UNORM (the bits it reads, by what each operation needs of
 * its operands), and that is that channel's decode on every value of
 * those fields (16384 at most, enumerated), is unpack_FORMAT(w, c): what
 * a texel fetch of that format gives, priced by tex_packed.  The float
 * formats decode exactly; the UNORM one is within the unit's tex_within
 * ulps of k/1023.
 */

/* the bits of w (slot j) that e's value can depend on, for the bits d
 * of e that are needed (an integer e; all of a float's) */
static uint64_t
wbits(nid e, uint64_t d, int j, int depth)
{
	ty_t t = N.ty[e];
	int w = TY_ISINT(t) && TYW(t) > 0 && TYW(t) < 64 ? TYW(t) : 64;
	uint64_t all = w == 64 ? ~0ull : (1ull << w) - 1, r = 0;
	int op = N.op[e], k;
	nid a = N.a[e], b = N.b[e];

	if (depth > 40)
		return ~0ull;
	if (TY_FP(t) || t == TY_BOOL)
		d = all;
	d &= all;
	if (!d)
		return 0;
	switch (op) {
	case O_CONST: case O_UNDEF:
		return 0;
	case O_VAR:
		return N.ki[e] == j ? d : 0;
	case O_CAST:
		if (TY_ISINT(N.ty[a]) && TY_ISINT(t) && N.ty[a] != TY_BOOL) {
			int ws = TYW(N.ty[a]) > 0 && TYW(N.ty[a]) < 64 ?
			    TYW(N.ty[a]) : 64;
			uint64_t ms = ws == 64 ? ~0ull : (1ull << ws) - 1;
			/* a wider signed result's high bits copy the sign */
			if ((d & ~ms) && TYK(N.ty[a]) == K_SINT)
				d |= 1ull << (ws - 1);
			return wbits(a, d & ms, j, depth + 1);
		}
		break;
	case O_AND: case O_OR: case O_XOR:
		if (op == O_AND && is_const(b))
			return wbits(a, d & (uint64_t)N.ki[b], j, depth + 1);
		if (op == O_AND && is_const(a))
			return wbits(b, d & (uint64_t)N.ki[a], j, depth + 1);
		return wbits(a, d, j, depth + 1) | wbits(b, d, j, depth + 1);
	case O_SHR:
		if (is_const(b) && N.ki[b] >= 0 && N.ki[b] < w) {
			uint64_t da = d << N.ki[b];
			if (TYK(t) == K_SINT && (d >> (w - N.ki[b])))
				da |= 1ull << (w - 1);
			return wbits(a, da & all, j, depth + 1);
		}
		break;
	case O_SHL:
		if (is_const(b) && N.ki[b] >= 0 && N.ki[b] < w)
			return wbits(a, d >> N.ki[b], j, depth + 1);
		break;
	case O_ADD: case O_SUB: case O_MUL: case O_NEG:
		if (TY_ISINT(t)) {
			/* a carry moves up only: the bits up to d's highest */
			for (k = 63; k > 0 && !(d >> k); k--)
				;
			d = k == 63 ? ~0ull : (2ull << k) - 1;
			r = wbits(a, d, j, depth + 1);
			return op == O_NEG ? r : r | wbits(b, d, j, depth + 1);
		}
		break;
	case O_BFE:
		if (is_const(b) && is_const(N.c[e]) && N.ki[N.c[e]] < 64)
			return wbits(a, (d & ((1ull << N.ki[N.c[e]]) - 1)) <<
			    N.ki[b], j, depth + 1);
		break;
	case O_SEL:
		return wbits(a, ~0ull, j, depth + 1) | wbits(b, d, j, depth +
		    1) | wbits(N.c[e], d, j, depth + 1);
	case O_TABLE: case O_LOOP: case O_INDEX: case O_ATOMIC:
		return ~0ull;
	}
	/* anything else needs every bit of its operands */
	if (a)
		r |= wbits(a, ~0ull, j, depth + 1);
	if (b)
		r |= wbits(b, ~0ull, j, depth + 1);
	if (N.c[e])
		r |= wbits(N.c[e], ~0ull, j, depth + 1);
	return r;
}

/* the fields channel c of a packed format occupies in its word */
static uint32_t
pk_field(int op, int c)
{
	switch (op) {
	case O_UNPACK_B10G11R11:
		return c == 2 ? 0x3ffu << 22 : 0x7ffu << (11 * c);
	case O_UNPACK_E5B9G9R9:
		return 0x1ffu << (9 * c) | 0xf8000000u;
	}
	return c == 3 ? 0xc0000000u : 0x3ffu << (10 * c);
}

static const char *
pk_vk(int op)
{
	return op == O_UNPACK_B10G11R11 ? "B10G11R11_UFLOAT_PACK32" : op ==
	    O_UNPACK_E5B9G9R9 ? "E5B9G9R9_UFLOAT_PACK32" :
	    "A2B10G10R10_UNORM_PACK32";
}

/*
 * e against unpack(op, w, c) on every value of the channel's field bits
 * of slot j (the others 0): the largest difference (NaN against NaN is
 * none), infinity where one is a NaN or undefined and the other not;
 * *nbad the values that differ, *onlybig 1 when they all have the
 * exponent field all ones (B10G11R11's Inf and NaN encodings)
 */
static double
pk_diff(nid e, nid cand, int j, int op, int c, int *nbad, int *onlybig)
{
	uint32_t f = pk_field(op, c), w, sub;
	int nb = 0, k, n = 0, i;
	uint32_t vals[16384];
	struct plan p;
	nid roots[2] = {e, cand};
	int64_t *in_i;
	double *in_r, md = 0;
	uint32_t la, lb;

	for (k = 0; k < 32; k++)
		nb += f >> k & 1;
	if (nb > 14)
		return INFINITY;
	/* the subsets of f, in order */
	sub = 0;
	do {
		vals[n++] = sub;
		sub = (sub - f) & f;
	} while (sub);
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	plan_build(&p, roots, 2, 0);
	la = plan_loc(&p, e);
	lb = plan_loc(&p, cand);
	*nbad = 0;
	*onlybig = op == O_UNPACK_B10G11R11;
	for (i = 0; i < n; i += CH) {
		int m = n - i < CH ? n - i : CH;
		for (k = 0; k < m; k++)
			in_i[(size_t)j * CH + k] = (int64_t)vals[i + k];
		plan_run(&p, (uint32_t)m, in_i, in_r);
		for (k = 0; k < m; k++) {
			double x = COLR(&p, la)[k], y = COLR(&p, lb)[k], d;
			int ux = COLF(&p, la)[k] & F_UNDEF;
			int uy = COLF(&p, lb)[k] & F_UNDEF;
			if (ux != uy || (!ux && isnan(x) != isnan(y)))
				d = INFINITY;
			else if (ux || isnan(x) || x == y)
				continue;
			else
				d = fabs(x - y);
			(*nbad)++;
			w = vals[i + k];
			if (op == O_UNPACK_B10G11R11 && ((w >> (11 * c + (c ==
			    2 ? 5 : 6))) & 31) != 31)
				*onlybig = 0;
			if (!(d <= md))
				md = d;
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return md;
}

/* channel c of format op is n's decode of slot j's word: 1, with the
 * largest difference in *md (0 for the float formats); 0 if not */
static int
packed_match(nid n, nid cand, int j, int op, int c, double *md)
{
	int nbad, big;

	*md = pk_diff(n, cand, j, op, c, &nbad, &big);
	if (op == O_UNPACK_A2B10G10R10)
		/* a normalised channel within half a step of k/q is that
		 * decode; farther, it is some other function */
		return *md <= 0.5 / (c == 3 ? 3 : 1023);
	if (nbad && big && !X.nopacked.n)
		sb_printf(&X.nopacked, "%s is %s but where %s's exponent field "
		    "is all ones: the unit decodes those as Inf and NaN",
		    node_str(n), node_str(cand), node_str(mk_var((uint32_t)j,
		    X.f->p[j].name, X.f->p[j].ty)));
	return nbad == 0;
}

static int
seed_packed(nid body, nid n)
{
	static const int ops[3] = {O_UNPACK_B10G11R11, O_UNPACK_E5B9G9R9,
	    O_UNPACK_A2B10G10R10};
	static const char *const chn = "rgba";
	uint64_t sl = node_slots(n), m;
	int j = 0, f, c = 0, op = 0, exact;
	nid v, cand = 0;
	struct repl *r;
	double md = 0, err;

	if (N.ty[n] != TY_F32 || !sl || (sl & (sl - 1)) || (sl >> MAXP) ||
	    is_const(n) || has_read(n, 0))
		return 0;
	while (!(sl >> j & 1))
		j++;
	if (j >= X.f->np || !TY_ISINT(X.f->p[j].ty) || TYW(X.f->p[j].ty) != 32)
		return 0;
	m = wbits(n, ~0ull, j, 0);
	if (!m || (m >> 32))
		return 0;
	v = mk_var((uint32_t)j, X.f->p[j].name, X.f->p[j].ty);
	for (f = 0; f < 3 && !op; f++)
		for (c = 0; c < (f == 2 ? 4 : 3); c++) {
			if (m & ~(uint64_t)pk_field(ops[f], c))
				continue;
			cand = mk_inst(ops[f], v, mk_const_i(TY_LIT, c), 0);
			if (packed_match(n, cand, j, ops[f], c, &md)) {
				op = ops[f];
				break;
			}
		}
	if (!op)
		return 0;
	exact = op != O_UNPACK_A2B10G10R10;
	/* the UNORM decode: within the unit's tex_within */
	err = exact ? 0 : md + X.ulp * ldexp(1.0, -24);
	if (!cclass_available(X.m, C_TEX_PACKED)) {
		X.unavail_packed = 1;
		return 0;
	}
	if (!exact && X.ulp == 0) {
		X.noulp = 1;
		return 0;
	}
	if ((r = new_repl(n)) == NULL)
		return 0;
	r->to = cand;
	r->cost = price(swap_nodes(body, &n, &cand, 1));
	r->ok = 1;
	r->exact = exact;
	r->filtered = !exact;
	r->curve = !exact;
	r->err = err;
	sb_printf(&r->what, "%s is %s: the %c channel of %s", node_str(n),
	    node_str(cand), chn[c], pk_vk(op));
	sb_printf(&r->holds, "holds for every %s: %s reads no bit of it "
	    "outside the channel's fields (0x%08x)", node_str(v), node_str(n),
	    pk_field(op, c));
	if (exact)
		sb_printf(&r->proof, "on all %d values of those bits (the others "
		    "0), the two are the same (NaN for NaN): exhaustive; the "
		    "format's values are all f32, so the unit's decode is exact",
		    1 << (op == O_UNPACK_E5B9G9R9 ? 14 : c == 2 ? 10 : 11));
	else
		sb_printf(&r->proof, "on all %d values of those bits (the others "
		    "0), the two differ by at most %s; the unit's decode of k/%d "
		    "is within %g ulp of it (the model's tex_within): %s in all",
		    c == 3 ? 4 : 1024, fmt_g(md), c == 3 ? 3 : 1023, X.ulp,
		    fmt_g(err));
	sb_printf(&r->pipe, "%s read as a texel of %s %s %s: texelFetch(s, "
	    "index).%c is the channel decoded, in place of the load of %s and "
	    "of this decode (the load is in neither price)", node_str(v), op ==
	    O_UNPACK_B10G11R11 ? "a" : "an", pk_vk(op), op ==
	    O_UNPACK_E5B9G9R9 ? "image (a sampled image: Vulkan does not "
	    "require it of texel buffers)" : "uniform texel buffer view or "
	    "image", chn[c], node_str(v));
	return 1;
}

/* ---- min/max reduction ---- */

/*
 * A min or max of neighbouring table entries (#87): two along a row, or
 * a 2 x 2 block of a table read as rows of W, is one linear read at the
 * centre of the footprint with a MIN or MAX reduction sampler
 * (VK_SAMPLER_REDUCTION_MODE_MIN, _MAX), which takes the least or the
 * greatest of the texels with a non-zero weight.  At the centre every
 * weight is 1/2 (or 1/4), on any sub-texel grid (Vulkan asks at least 4
 * bits), so every texel of the footprint counts whichever way the unit
 * snaps: the read is the min or max exactly, as long as the texture holds
 * the entries exactly (half or single floats; whole numbers 0..255 or
 * 0..65535 as normalised units, scaled back and rounded, which needs the
 * unit's tex_within).
 */

/* the min or max tree under n: its leaves (at most 4), 0 if more */
static int
mm_leaves(nid e, int op, nid *leaf, int nl)
{
	if (N.op[e] == op) {
		nl = mm_leaves(N.a[e], op, leaf, nl);
		return nl ? mm_leaves(N.b[e], op, leaf, nl) : 0;
	}
	if (nl < 0 || nl == 4)
		return 0;
	leaf[nl] = e;
	return nl + 1;
}

/* why a min or max of neighbours is not one read: it can leave the
 * table (or a row), or its index has no proved range */
static void
mm_out(nid n, nid xa, nid ya, int64_t lo, int64_t hi, int64_t ylo,
    int64_t yhi, int ranged)
{
	if (X.nosum.n)
		return;
	if (!ranged) {
		sb_printf(&X.nosum, "%s takes neighbouring entries, but %s has "
		    "no proved range: a read of them could leave the table",
		    node_str(n), node_str(ya ? ya : xa));
		return;
	}
	sb_printf(&X.nosum, "%s takes neighbouring entries, but a read of them "
	    "could leave the table: %s in [%lld, %lld]", node_str(n),
	    node_str(xa), (long long)lo, (long long)hi);
	if (ya)
		sb_printf(&X.nosum, ", %s in [%lld, %lld]", node_str(ya),
		    (long long)ylo, (long long)yhi);
	sb_puts(&X.nosum, " (a filter clamps at the edge, the table's index "
	    "does not)");
}

static int
seed_minmax(nid body, nid n)
{
	int op = N.op[n], nl, k, i, two, bits = X.bits ? X.bits : 4, isint;
	int f16 = 1;
	nid leaf[4], tab, tex, x, y = 0, cand, nb, xa = 0, ya = 0, *e;
	struct aff ax[4];
	int64_t c0, W, cnt, lo, hi, ylo = 0, yhi = 0, cx = 0, cy = 0, q = 0;
	int64_t vmin = 0, vmax = 0;
	double err = 0;
	struct repl *r;
	struct sbuf wit = {0};
	uint32_t j;

	if (op != O_MIN && op != O_MAX)
		return 0;
	nl = mm_leaves(n, op, leaf, 0);
	if (nl != 2 && nl != 4)
		return 0;
	for (k = 0; k < nl; k++)
		if (N.op[leaf[k]] != O_INDEX || !same_table(N.a[leaf[k]],
		    N.a[leaf[0]]) || !aff_of(N.b[leaf[k]], &ax[k], 0) ||
		    !aff_same_base(&ax[0], &ax[k]))
			return 0;
	tab = N.a[leaf[0]];
	cnt = N.b[tab];
	/* the entries: whole numbers, or floats an f32 holds */
	isint = TY_ISINT(N.ty[tab]);
	for (j = 0; j < cnt; j++) {
		nid t = N.lst[N.a[tab] + j];
		double v;
		if (!is_const(t))
			return 0;
		v = cval(t);
		if (isint) {
			if (!j || N.ki[t] < vmin)
				vmin = N.ki[t];
			if (!j || N.ki[t] > vmax)
				vmax = N.ki[t];
		} else if ((double)(float)v != v)
			return 0;
		if (fround(16, v) != v)
			f16 = 0;
	}
	if (isint && (vmin < -16777216 || vmax > 16777216))
		return 0;
	/* the offsets: c0, c0 + 1 (and c0 + W, c0 + W + 1) */
	c0 = ax[0].c;
	for (k = 1; k < nl; k++)
		if (ax[k].c < c0)
			c0 = ax[k].c;
	if (nl == 2) {
		W = cnt;
		two = (ax[0].c == c0 && ax[1].c == c0 + 1) || (ax[1].c == c0 &&
		    ax[0].c == c0 + 1);
		if (!two)
			return 0;
		/* texel c0's index, and its range inside the table */
		for (k = 0; k < 2 && ax[k].c != c0; k++)
			;
		xa = N.b[leaf[k]];
		if (!irange(xa, &lo, &hi) || lo < 0 || hi + 1 > cnt - 1) {
			mm_out(n, xa, 0, lo, hi, 0, 0, irange(xa, &lo, &hi));
			return 0;
		}
	} else {
		/* rows of W: one atom of coefficient 1, one of W */
		if (ax[0].n != 2)
			return 0;
		W = ax[0].coef[0] == 1 ? ax[0].coef[1] : ax[0].coef[0];
		if (W < 2 || cnt % W || (ax[0].coef[0] != 1 &&
		    ax[0].coef[1] != 1))
			return 0;
		for (k = 0; k < 4; k++) {
			int64_t d = ax[k].c - c0;
			if (d != 0 && d != 1 && d != W && d != W + 1)
				return 0;
			for (i = 0; i < k; i++)
				if (ax[i].c == ax[k].c)
					return 0;
		}
		xa = ax[0].atom[ax[0].coef[0] == 1 ? 0 : 1];
		ya = ax[0].atom[ax[0].coef[0] == 1 ? 1 : 0];
		cx = ((c0 % W) + W) % W;
		cy = (c0 - cx) / W;
		if (!irange(xa, &lo, &hi) || !irange(ya, &ylo, &yhi) || lo + cx <
		    0 || hi + cx + 1 > W - 1 || ylo + cy < 0 || yhi + cy + 1 >
		    cnt / W - 1) {
			mm_out(n, xa, ya, lo, hi, ylo, yhi, irange(xa, &lo,
			    &hi) && irange(ya, &ylo, &yhi));
			return 0;
		}
	}
	if (hi + cx > 16777215 || yhi + cy > 16777215)
		return 0;
	if (!model_lists(X.m, C_TEX_MINMAX)) {
		X.unavail[C_TEX_MINMAX] = 1;
		return 0;
	}
	/* the texture: normalised units for whole numbers 0..65535 (scaled
	 * back, rounded), else the entries as f16 or f32 */
	if (isint && vmin >= 0 && vmax <= 65535) {
		q = vmax <= 255 ? 255 : 65535;
		if (X.ulp == 0) {
			X.noulp = 1;
			return 0;
		}
		/* the decode within tex_within ulps of k/q, the product by q
		 * and the add of 1/2 rounded */
		err = (double)q * (X.ulp + 2) * ldexp(1.0, -24) + ldexp(1.0,
		    -24) * (q + 1);
		if (!(err < 0.5))
			return 0;
	}
	e = xmalloc(cnt * sizeof *e);
	for (j = 0; j < cnt; j++)
		e[j] = kf(q ? (double)(float)((double)N.ki[N.lst[N.a[tab] + j]] /
		    (double)q) : cval(N.lst[N.a[tab] + j]));
	tex = mk_table(e, (uint32_t)cnt);
	free(e);
	if (nl == 2)
		x = mk_bin(O_ADD, mk_cast(TY_F32, xa), kf(1));
	else {
		x = mk_bin(O_ADD, mk_cast(TY_F32, xa), kf((double)cx + 1));
		y = mk_bin(O_ADD, mk_cast(TY_F32, ya), kf((double)cy + 1));
	}
	cand = mk_tex(nl == 2 ? O_TEX_LINEAR : O_TEX_BILINEAR, TEX_KI(bits, nl ==
	    2 ? 0 : W) | (op == O_MIN ? TEX_REDMIN : TEX_REDMAX), tex, x, y);
	if (q)
		cand = mk_cast(N.ty[n], mk_bin(O_ADD, mk_bin(O_MUL, cand,
		    kf((double)q)), kf(0.5)));
	else if (N.ty[cand] != N.ty[n])
		cand = mk_cast(N.ty[n], cand);
	if (equiv_check(X.f, n, cand, &wit, &i)) {
		sb_printf(&X.nosum, "%s%s would be %s, but they differ on a "
		    "point tried (%s): not taken", X.nosum.n ? "\n" : "",
		    node_str(n), node_str(cand), sb_str(&wit));
		free(wit.p);
		return 0;
	}
	free(wit.p);
	if ((r = new_repl(n)) == NULL)
		return 0;
	nb = swap_nodes(body, &n, &cand, 1);
	r->to = cand;
	r->exact = 1;
	r->ok = 1;
	r->cost = price(nb);
	sb_printf(&r->what, "%s is %s: the %s of %d neighbouring entries, one "
	    "read with a %s reduction", node_str(n), node_str(cand), op ==
	    O_MIN ? "least" : "greatest", nl, op == O_MIN ? "MIN" : "MAX");
	if (nl == 2)
		sb_printf(&r->holds, "holds where %s is in [%lld, %lld] (its "
		    "range): both texels are inside the table", node_str(xa),
		    (long long)lo, (long long)hi);
	else
		sb_printf(&r->holds, "holds where %s is in [%lld, %lld] and %s "
		    "in [%lld, %lld] (their ranges): the block is inside the "
		    "%lld x %lld table, no row wraps", node_str(xa), (long long)lo,
		    (long long)hi, node_str(ya), (long long)ylo, (long long)yhi,
		    (long long)W, (long long)(cnt / W));
	sb_printf(&r->proof, "the read is at the centre of the footprint, "
	    "fraction 1/2%s: on any sub-texel grid (Vulkan asks at least 4 "
	    "bits) every weight is %s, none 0, so the %s reduction takes every "
	    "texel of it, and gives one of them, no sum", nl == 4 ? " on both "
	    "axes" : "", nl == 4 ? "1/4" : "1/2", op == O_MIN ? "MIN" : "MAX");
	if (q)
		sb_printf(&r->proof, "\nthe entries are whole numbers 0..%lld, "
		    "stored as k/%lld: the decode within %g ulp (the model's "
		    "tex_within), the product by %lld and the add of 1/2 are "
		    "within %s of the entry, below 1/2, so the conversion gives "
		    "it", (long long)q, (long long)q, X.ulp, (long long)q,
		    fmt_g(err));
	else
		sb_printf(&r->proof, "\nthe texture holds the entries exactly "
		    "(%s): the read is the %s of them exactly", f16 ?
		    "half floats" :
		    "f32", op == O_MIN ? "least" : "greatest");
	sb_puts(&r->proof, "\ncompared on the points tried as a second opinion: "
	    "the same");
	sb_printf(&r->pipe, "a texture %s, %lld x %lld, one level: the table's "
	    "entries%s\na sampler: LINEAR mag and min filters, NEAREST mipmap "
	    "mode, CLAMP_TO_EDGE, unnormalizedCoordinates, lod 0, and a "
	    "VkSamplerReductionModeCreateInfo of %s (Vulkan 1.2 or "
	    "VK_EXT_sampler_filter_minmax; filterMinmaxSingleComponentFormats "
	    "covers R8_UNORM, R16_UNORM, R16_SFLOAT and R32_SFLOAT)\none "
	    "combined image sampler more in the layout; the shader reads "
	    "textureLod(s, %s, 0.0).r", q == 255 ? "R8_UNORM" : q ? "R16_UNORM" :
	    f16 ? "R16_SFLOAT" : "R32_SFLOAT", (long long)W, (long long)(cnt /
	    W), q ? " as normalised units (the shader sees k/q, times q back)" :
	    "", op == O_MIN ? "VK_SAMPLER_REDUCTION_MODE_MIN" :
	    "VK_SAMPLER_REDUCTION_MODE_MAX", nl == 2 ? "x" : "vec2(x, y)");
	return 1;
}

/* ---- cube maps ---- */

/*
 * A cube map's face selected by hand (#97): a subterm of three f32
 * parameters that is a selection among constants by comparisons of the
 * coordinates, their magnitudes (abs, negation, min, max of them) and
 * zero depends on its inputs only through their signs, which are zero
 * or NaN, and the order of their magnitudes.  Every such combination is
 * taken by a point with coordinates in {NaN, -3, -2, -1, -0, 0, 1, 2, 3}
 * (a rank of the magnitudes for each value), so equality on those 729
 * points is equality everywhere.  Equal to cubeid there, the subterm is
 * the face: the instruction where the model prices `cube`; the read of
 * an identity cube map (tex_cube) where it prices tex_cube and z's, y's
 * or x's magnitude is the largest alone on the ranges, since the
 * specification lets the unit break a tie of magnitude its own way.
 */

/* a magnitude term over the coordinates (slots in sl) */
static int
cube_mag(nid e, uint64_t sl, int depth)
{
	if (depth > 12)
		return 0;
	switch (N.op[e]) {
	case O_VAR:
		return N.ki[e] < 64 && (sl >> N.ki[e] & 1);
	case O_CONST:
		return cval(e) == 0;
	case O_ABS: case O_NEG:
		return cube_mag(N.a[e], sl, depth + 1);
	case O_MIN: case O_MAX:
		return cube_mag(N.a[e], sl, depth + 1) && cube_mag(N.b[e], sl,
		    depth + 1);
	case O_CAST:
		return TY_FP(N.ty[N.a[e]]) && TY_FP(N.ty[e]) &&
		    cube_mag(N.a[e], sl, depth + 1);
	}
	return 0;
}

/* a coordinate leaf: a coordinate, negated, or times 2 or 1/2 */
static int
cube_leaf(nid e, uint64_t sl, int depth)
{
	if (depth > 6)
		return 0;
	switch (N.op[e]) {
	case O_VAR:
		return N.ki[e] < 64 && (sl >> N.ki[e] & 1);
	case O_NEG:
		return cube_leaf(N.a[e], sl, depth + 1);
	case O_MUL:
		return (is_const(N.a[e]) && (fabs(cval(N.a[e])) == 2 ||
		    fabs(cval(N.a[e])) == 0.5) && cube_leaf(N.b[e], sl, depth +
		    1)) || (is_const(N.b[e]) && (fabs(cval(N.b[e])) == 2 ||
		    fabs(cval(N.b[e])) == 0.5) && cube_leaf(N.a[e], sl, depth +
		    1));
	case O_ADD:
		return N.a[e] == N.b[e] && cube_leaf(N.a[e], sl, depth + 1);
	}
	return 0;
}

/* a selection among constants or coordinate leaves by comparisons of
 * magnitude terms */
static int
cube_shape(nid e, uint64_t sl, int depth)
{
	if (depth > 24)
		return 0;
	switch (N.op[e]) {
	case O_CONST:
		return 1;
	case O_VAR: case O_NEG: case O_MUL: case O_ADD:
		return cube_leaf(e, sl, 0);
	case O_SEL:
		return cube_shape(N.a[e], sl, depth + 1) && cube_shape(N.b[e], sl,
		    depth + 1) && cube_shape(N.c[e], sl, depth + 1);
	case O_LAND: case O_LOR:
		return cube_shape(N.a[e], sl, depth + 1) && cube_shape(N.b[e], sl,
		    depth + 1);
	case O_LNOT: case O_CAST:
		return cube_shape(N.a[e], sl, depth + 1);
	case O_LT: case O_LE: case O_GT: case O_GE: case O_EQ: case O_NE:
		return cube_mag(N.a[e], sl, 0) && cube_mag(N.b[e], sl, 0);
	}
	return 0;
}

/*
 * e against the face cand on the 729 points (s the slots of x, y, z): 1
 * when equal at every one (tie 0) or at every one with one largest
 * magnitude, no NaN and not all zero (tie 1); *at a point where not
 */
static int
cube_same(nid e, nid cand, const int *s, int tie, double *at)
{
	/* three magnitudes that no leaf's factor (2, 1/2) maps on another */
	static const double v[9] = {NAN, -25, -5, -1, -0.0, 0, 1, 5, 25};
	struct plan p;
	nid roots[2] = {e, cand};
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t la, lb;
	int i, k, m = 0, ok = 1, idx[CH];

	plan_build(&p, roots, 2, 0);
	la = plan_loc(&p, e);
	lb = plan_loc(&p, cand);
	for (i = 0; i <= 729 && ok; i++) {
		if (i < 729) {
			double x = v[i / 81], y = v[i / 9 % 9], z = v[i % 9];
			double ax = fabs(x), ay = fabs(y), az = fabs(z), mx;
			mx = ax > ay ? ax : ay;
			mx = mx > az ? mx : az;
			if (tie && (isnan(x) || isnan(y) || isnan(z) || mx == 0 ||
			    (ax == mx) + (ay == mx) + (az == mx) > 1))
				continue;
			in_r[(size_t)s[0] * CH + m] = x;
			in_r[(size_t)s[1] * CH + m] = y;
			in_r[(size_t)s[2] * CH + m] = z;
			idx[m++] = i;
			if (m < CH)
				continue;
		}
		if (!m)
			break;
		plan_run(&p, (uint32_t)m, in_i, in_r);
		for (k = 0; k < m && ok; k++) {
			double a = COLR(&p, la)[k], b = COLR(&p, lb)[k];
			if (TY_ISINT(N.ty[e]))
				a = (double)COLI(&p, la)[k];
			if (TY_ISINT(N.ty[cand]))
				b = (double)COLI(&p, lb)[k];
			if (((COLF(&p, la)[k] | COLF(&p, lb)[k]) & F_UNDEF) ||
			    !(a == b || (isnan(a) && isnan(b)))) {
				ok = 0;
				at[0] = v[idx[k] / 81];
				at[1] = v[idx[k] / 9 % 9];
				at[2] = v[idx[k] % 9];
			}
		}
		m = 0;
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return ok;
}

/* the axis whose magnitude is the largest alone on the ranges (0 x, 1
 * y, 2 z), -1 when none is */
static int
cube_major(const nid *v)
{
	double lo[3], hi[3], mlo[3], mhi[3];
	int k, j;

	for (k = 0; k < 3; k++) {
		if (!frange(v[k], &lo[k], &hi[k]))
			return -1;
		mlo[k] = lo[k] > 0 ? lo[k] : hi[k] < 0 ? -hi[k] : 0;
		mhi[k] = fabs(lo[k]) > fabs(hi[k]) ? fabs(lo[k]) : fabs(hi[k]);
	}
	for (k = 0; k < 3; k++) {
		for (j = 0; j < 3; j++)
			if (j != k && !(mlo[k] > mhi[j]))
				break;
		if (j == 3)
			return k;
	}
	return -1;
}

static int
seed_cube(nid body, nid n)
{
	static const int perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2,
	    0}, {2, 0, 1}, {2, 1, 0}};
	static const char *const said[5] = {"the cube map face", "the face's "
	    "s before the division (sc)", "the face's t before the division "
	    "(tc)", "twice the major axis (2 ma)", "the major axis (ma)"};
	uint64_t sl = node_slots(n);
	int sv[3], s[3], k, j = 0, p, q, ninst, ntex, major = -1, kind = -1;
	nid v[3], cand = 0, c, ctx = 0, nb;
	double at[3];
	float cinst = INFINITY, ctex = INFINITY;
	struct repl *r;

	if (is_const(n) || (sl >> MAXP) || N.op[n] != O_SEL)
		return 0;
	for (k = 0; k < 64 && j < 4; k++)
		if (sl >> k & 1)
			sv[j++] = k;
	if (j != 3)
		return 0;
	for (k = 0; k < 3; k++)
		if (sv[k] >= X.f->np || X.f->p[sv[k]].ty != TY_F32)
			return 0;
	if (!cube_shape(n, sl, 0))
		return 0;
	ninst = cclass_available(X.m, C_CUBE);
	ntex = cclass_available(X.m, C_TEX_CUBE);
	for (p = 0; p < 6 && kind < 0; p++) {
		for (k = 0; k < 3; k++) {
			s[k] = sv[perm[p][k]];
			v[k] = mk_var((uint32_t)s[k], X.f->p[s[k]].name, TY_F32);
		}
		for (q = 0; q < 5 && kind < 0; q++) {
			static const int ops[5] = {O_CUBEID, O_CUBESC, O_CUBETC,
			    O_CUBEMA, O_CUBEMA};
			c = mk_inst(ops[q], v[0], v[1], v[2]);
			if (q == 4)
				c = mk_bin(O_MUL, c, kf(0.5));
			if (N.ty[n] != TY_F32)
				c = mk_cast(N.ty[n], c);
			if (cube_same(n, c, s, 0, at)) {
				cand = c;
				kind = q;
			} else if (q == 0 && cube_same(n, c, s, 1, at))
				kind = 0;	/* the face away from ties */
		}
	}
	if (kind < 0)
		return 0;
	if (cand && ninst)
		cinst = price(swap_nodes(body, &n, &cand, 1));
	if (kind == 0 && ntex) {
		ctx = mk_inst(O_TEX_CUBE, v[0], v[1], v[2]);
		if (N.ty[n] != TY_F32)
			ctx = mk_cast(N.ty[n], ctx);
		if ((major = cube_major(v)) >= 0)
			ctex = price(swap_nodes(body, &n, &ctx, 1));
	}
	if (!ninst && !(kind == 0 && ntex)) {
		X.unavail_cube = 1;
		return 0;
	}
	if (!(cinst < INFINITY) && !(ctex < INFINITY)) {
		if (X.nocube.n)
			return 0;
		sb_printf(&X.nocube, "%s is %s of (%s, %s, %s)", node_str(n),
		    said[kind], node_str(v[0]), node_str(v[1]), node_str(v[2]));
		if (!cand)
			sb_printf(&X.nocube, " where one magnitude is the "
			    "largest, but not cubeid at (%g, %g, %g)", at[0],
			    at[1], at[2]);
		if (ntex)
			sb_printf(&X.nocube, "%s where two magnitudes tie for "
			    "the largest the specification lets the unit choose "
			    "its own face: the ranges must keep one axis the "
			    "largest alone for a cube map's read", cand ? ", but" :
			    ";");
		return 0;
	}
	if ((r = new_repl(n)) == NULL)
		return 0;
	nb = cinst <= ctex ? cand : ctx;
	r->to = nb;
	r->cost = cinst <= ctex ? cinst : ctex;
	r->exact = 1;
	r->ok = 1;
	sb_printf(&r->what, "%s is %s: %s of (%s, %s, %s)", node_str(n),
	    node_str(nb), said[kind], node_str(v[0]), node_str(v[1]),
	    node_str(v[2]));
	if (cinst <= ctex)
		sb_printf(&r->holds, "holds for every input");
	else
		sb_printf(&r->holds, "holds where %s is in its range, the "
		    "largest magnitude alone: the unit's face, whatever its rule "
		    "on ties", node_str(v[major]));
	sb_printf(&r->proof, "%s depends on its inputs only through their "
	    "signs, which are zero or NaN, and the order of their magnitudes "
	    "(a selection among constants and the coordinates, negated or "
	    "times 2 or 1/2, by comparisons of the coordinates, their "
	    "magnitudes and zero); a point with coordinates in {NaN, -25, -5, "
	    "-1, -0, 0, 1, 5, 25} takes every such combination (no factor "
	    "maps one of those magnitudes on another), and on all 729 %s it "
	    "is the same: exhaustive", node_str(n), cinst <= ctex ? "" :
	    "with one largest magnitude, no NaN and not all zero");
	if (cinst <= ctex)
		sb_printf(&r->pipe, "nothing: one instruction of the model's "
		    "cube class (%s)", kind == 0 ? "v_cubeid_f32" : kind == 1 ?
		    "v_cubesc_f32" : kind == 2 ? "v_cubetc_f32" : "v_cubema_f32");
	else
		sb_printf(&r->pipe, "a cube texture, 1 x 1 a face, face k "
		    "holding k (R8_UINT, or R16_SFLOAT for a float)\na sampler: "
		    "NEAREST filters, lod 0\none combined image sampler more in "
		    "the layout; the shader reads textureLod(s, vec3(%s, %s, %s), "
		    "0.0).r", node_str(v[0]), node_str(v[1]), node_str(v[2]));
	return 1;
}

/* ---- weighted sums of neighbouring texels ---- */

struct sum {
	struct term t[MAXTERM];
	nid tidx[MAXTERM];	/* each term's index expression */
	int nt;
	struct rest r[MAXREST];
	int nr;
	int isint, bad;
};

static void
add_rest(struct sum *s, struct mform k, nid e)
{
	if (s->nr == MAXREST) {
		s->bad = 1;
		return;
	}
	s->r[s->nr].k = k;
	s->r[s->nr++].e = e;
}

/* a factor that is a weight: a form over float atoms, or (in an integer
 * sum) a constant */
static int
weight_of(const struct sum *s, nid e, struct mform *w)
{
	if (has_read(e, 0))
		return 0;
	if (s->isint) {
		if (!is_const(e))
			return 0;
		*w = mf_const((double)N.ki[e]);
		return 1;
	}
	return mf_of(e, w, 0);
}

static void
collect(struct sum *s, nid e, struct mform k, int depth)
{
	struct mform w, k2;
	int64_t lo, hi;
	int op = N.op[e];

	if (s->bad)
		return;
	if (depth > 40) {
		s->bad = 1;
		return;
	}
	/* an integer sum is the integers' only where it does not wrap */
	if (s->isint && (op == O_ADD || op == O_SUB || op == O_NEG ||
	    op == O_MUL || op == O_SHL) && !irange(e, &lo, &hi))
		goto opaque;
	switch (op) {
	case O_ADD: case O_SUB:
		collect(s, N.a[e], k, depth + 1);
		if (op == O_SUB && !mf_add(&k, mf_const(0), k, -1))
			s->bad = 1;
		collect(s, N.b[e], k, depth + 1);
		return;
	case O_NEG:
		if (!mf_add(&k, mf_const(0), k, -1))
			s->bad = 1;
		collect(s, N.a[e], k, depth + 1);
		return;
	case O_MUL:
		if (weight_of(s, N.a[e], &w) && has_read(N.b[e], 0) &&
		    mf_mul(&k2, k, w)) {
			collect(s, N.b[e], k2, depth + 1);
			return;
		}
		if (weight_of(s, N.b[e], &w) && has_read(N.a[e], 0) &&
		    mf_mul(&k2, k, w)) {
			collect(s, N.a[e], k2, depth + 1);
			return;
		}
		break;
	case O_SHL:
		if (s->isint && is_const(N.b[e]) && N.ki[N.b[e]] >= 0 &&
		    N.ki[N.b[e]] < 24 && mf_mul(&k2, k, mf_const(ldexp(1.0,
		    (int)N.ki[N.b[e]])))) {
			collect(s, N.a[e], k2, depth + 1);
			return;
		}
		break;
	case O_DIV:
		if (!s->isint && is_const(N.b[e]) && cval(N.b[e]) != 0 &&
		    mf_mul(&k2, k, mf_const(1 / cval(N.b[e])))) {
			collect(s, N.a[e], k2, depth + 1);
			return;
		}
		break;
	case O_FMA:
		if (weight_of(s, N.a[e], &w) && has_read(N.b[e], 0) &&
		    mf_mul(&k2, k, w))
			collect(s, N.b[e], k2, depth + 1);
		else if (weight_of(s, N.b[e], &w) && has_read(N.a[e], 0) &&
		    mf_mul(&k2, k, w))
			collect(s, N.a[e], k2, depth + 1);
		else
			add_rest(s, k, mk_bin(O_MUL, N.a[e], N.b[e]));
		collect(s, N.c[e], k, depth + 1);
		return;
	case O_CAST:
		/* between float types, and an integer entry read as a float:
		 * the same number */
		if (!s->isint && (TY_FP(N.ty[N.a[e]]) || N.op[N.a[e]] ==
		    O_INDEX)) {
			collect(s, N.a[e], k, depth + 1);
			return;
		}
		/* an entry of 0..255 in a type that holds it */
		if (s->isint && N.op[N.a[e]] == O_INDEX && (!TY_SIZED(N.ty[e]) ||
		    TYW(N.ty[e]) >= 16 || (TYK(N.ty[e]) == K_UINT &&
		    TYW(N.ty[e]) >= 8))) {
			collect(s, N.a[e], k, depth + 1);
			return;
		}
		break;
	case O_INDEX: {
		struct tinfo ti;
		struct term *t;
		nid ix = N.b[e], wx;
		int wrap;
		if (s->nt == MAXTERM || !table_info(N.a[e], &ti) ||
		    ti.tmax > 16777216.0 || (s->isint && !ti.isint))
			break;
		t = &s->t[s->nt];
		memset(t, 0, sizeof *t);
		/* T[x % n], T[mirror of x]: a read that wraps (#101); as it
		 * is too, for when no other entry wraps with it */
		if ((wrap = wrap_of(ix, N.b[N.a[e]], &wx)) != 0) {
			t->ok0 = aff_of(ix, &t->ix0, 0);
			ix = wx;
		}
		if (!aff_of(ix, &t->ix, 0)) {
			if (!t->ok0)
				break;
			t->ix = t->ix0;
			wrap = 0;
		}
		t->wrap = wrap;
		t->k = k;
		t->tab = N.a[e];
		s->tidx[s->nt++] = N.b[e];
		return;
	}
	}
opaque:
	add_rest(s, k, e);
}

/* a block of up to 2 x 2 (x 2) neighbouring texels read by one filtered
 * tap */
struct block {
	int m[8];		/* its terms at (0,0) (1,0) (0,1) (1,1), then the
				 * same at z 1 (bits x 1, y 2, z 4), -1 none */
	int ox, oy, oz;		/* the offset of its (0,0,0) */
	double S;		/* the weights' sum */
	struct mform fu, fv, fw;	/* the fractions along x, y and z */
	double resid;		/* weights not reproduced, times nothing yet */
	int split;		/* a weight shared with another read */
};

/* why a block of neighbours is not one read (block_derive, block_inside) */
enum { BW_OK, BW_SUM, BW_SIGN, BW_RANK, BW_OUT };

/* a block whose texels have the weights w, at (0,0) (1,0) (0,1) (1,1) */
static int
block_derive_w(const struct mform *w, struct block *b, int *why)
{
	struct mform tot = mf_const(0), su = mf_const(0), sv = mf_const(0), p,
	    wr[4], d, one = mf_const(1), t1, t2;
	double lo, hi, mag;
	int i;

	for (i = 0; i < 4; i++)
		if (!mf_add(&tot, tot, w[i], 1))
			return 0;
	*why = BW_SUM;
	if (!mf_isconst(&tot) || tot.c[0] == 0)
		return 0;
	b->S = tot.c[0];
	if (!mf_add(&su, w[1], w[3], 1) || !mf_add(&sv, w[2], w[3], 1) ||
	    !mf_mul(&b->fu, su, mf_const(1 / b->S)) ||
	    !mf_mul(&b->fv, sv, mf_const(1 / b->S)))
		return 0;
	mf_tidy(&b->fu);
	mf_tidy(&b->fv);
	*why = BW_SIGN;
	if (b->fu.t || b->fv.t)		/* a fraction is linear in one atom */
		return 0;
	if (!mf_range(&b->fu, &lo, &hi) || lo < 0 || hi > 1)
		return 0;
	if (!mf_range(&b->fv, &lo, &hi) || lo < 0 || hi > 1)
		return 0;
	/* the weights the fractions give back, against the source's */
	*why = BW_RANK;
	if (!mf_mul(&p, b->fu, b->fv) || !mf_add(&t1, b->fu, p, -1) ||
	    !mf_add(&t2, b->fv, p, -1) || !mf_add(&wr[0], one, b->fu, -1) ||
	    !mf_add(&wr[0], wr[0], t2, -1))
		return 0;
	wr[1] = t1;
	wr[2] = t2;
	wr[3] = p;
	b->resid = 0;
	for (i = 0; i < 4; i++) {
		if (!mf_mul(&wr[i], wr[i], mf_const(b->S)) || !mf_add(&d, w[i],
		    wr[i], -1))
			return 0;
		mag = mf_mag(&d);
		if (!(mag <= 1e-9 * fabs(b->S)))
			return 0;
		b->resid += mag;
	}
	*why = BW_OK;
	return 1;
}

static int
block_derive(const struct sum *s, struct block *b, int *why)
{
	struct mform w[4];
	int i;

	for (i = 0; i < 4; i++)
		w[i] = b->m[i] >= 0 ? s->t[b->m[i]].k : mf_const(0);
	b->split = 0;
	b->oz = 0;
	b->fw = mf_const(0);
	return block_derive_w(w, b, why);
}

/*
 * A 2 x 2 x 2 block (or any box of it) of a 3-D table: its eight weights
 * are S times the products of a fraction along each axis, each linear in
 * one atom and in [0, 1], within 1e-9 S.
 */
static int
block_derive3(const struct sum *s, struct block *b, int *why)
{
	struct mform w[8], tot = mf_const(0), f[3], one = mf_const(1), p, d;
	double lo, hi, mag;
	int q, k;

	b->split = 0;
	for (q = 0; q < 8; q++) {
		w[q] = b->m[q] >= 0 ? s->t[b->m[q]].k : mf_const(0);
		if (!mf_add(&tot, tot, w[q], 1))
			return 0;
	}
	*why = BW_SUM;
	if (!mf_isconst(&tot) || tot.c[0] == 0)
		return 0;
	b->S = tot.c[0];
	for (k = 0; k < 3; k++) {
		struct mform su = mf_const(0);
		for (q = 0; q < 8; q++)
			if ((q >> k & 1) && !mf_add(&su, su, w[q], 1))
				return 0;
		if (!mf_mul(&f[k], su, mf_const(1 / b->S)))
			return 0;
		mf_tidy(&f[k]);
		*why = BW_SIGN;
		if (f[k].t)	/* a fraction is linear in one atom */
			return 0;
		if (!mf_range(&f[k], &lo, &hi) || lo < 0 || hi > 1)
			return 0;
	}
	b->fu = f[0];
	b->fv = f[1];
	b->fw = f[2];
	/* the weights the fractions give back, against the source's */
	*why = BW_RANK;
	b->resid = 0;
	for (q = 0; q < 8; q++) {
		p = mf_const(b->S);
		for (k = 0; k < 3; k++) {
			struct mform g = f[k];
			if (!(q >> k & 1) && !mf_add(&g, one, f[k], -1))
				return 0;
			if (!mf_mul(&p, p, g))
				return 0;
		}
		if (!mf_add(&d, w[q], p, -1))
			return 0;
		mag = mf_mag(&d);
		if (!(mag <= 1e-9 * fabs(b->S)))
			return 0;
		b->resid += mag;
	}
	*why = BW_OK;
	return 1;
}

/* the terms of a group laid on the table's grid */
struct grid {
	int ix[MAXTERM], n;	/* the group's terms (indices into the sum) */
	int wrap;		/* O_FETCH_REPEAT, _MIRROR: it wraps (1D) */
	int64_t W, H, D;	/* D > 1: slices of H rows of W, a 3-D texture */
	nid xe, ye, ze;		/* the column, row and slice expressions, 0:
				 * none */
	int64_t xlo, xhi, ylo, yhi, zlo, zhi;
	struct tinfo ti;
	double dxmax, dymax, dzmax;
};

/* the group's member (its place in g->ix) at an offset, -1: none */
static int
grid_at(const struct sum *s, const struct grid *g, int dx, int dy, int dz)
{
	int i;

	for (i = 0; i < g->n; i++)
		if (s->t[g->ix[i]].dx == dx && s->t[g->ix[i]].dy == dy &&
		    s->t[g->ix[i]].dz == dz)
			return i;
	return -1;
}

/* is the block's footprint inside the table for every index */
static int
block_inside(const struct grid *g, const struct block *b)
{
	int wx = 0, wy = 0, wz = 0, q;

	for (q = 0; q < 8; q++)
		if (b->m[q] >= 0) {
			wx |= q & 1;
			wy |= q >> 1 & 1;
			wz |= q >> 2 & 1;
		}
	if (g->wrap)	/* the address mode wraps every texel in */
		return 1;
	return g->xlo + b->ox >= 0 && g->xhi + b->ox + wx <= g->W - 1 &&
	    g->ylo + b->oy >= 0 && g->yhi + b->oy + wy <= g->H - 1 &&
	    g->zlo + b->oz >= 0 && g->zhi + b->oz + wz <= g->D - 1;
}

/* the largest step between neighbours along x, y and z of slices of H
 * rows of W */
static void
table_steps3(const struct tinfo *ti, int64_t W, int64_t H, double *dx,
    double *dy, double *dz)
{
	int64_t j, n = ti->n;

	*dx = *dy = *dz = 0;
	for (j = 0; j < n; j++) {
		double v = cval(N.lst[N.a[ti->tab] + j]), d;
		if ((j + 1) % W) {
			d = fabs(cval(N.lst[N.a[ti->tab] + j + 1]) - v);
			if (d > *dx)
				*dx = d;
		}
		if ((j / W + 1) % H) {
			d = fabs(cval(N.lst[N.a[ti->tab] + j + W]) - v);
			if (d > *dy)
				*dy = d;
		}
		if (j + W * H < n) {
			d = fabs(cval(N.lst[N.a[ti->tab] + j + W * H]) - v);
			if (d > *dz)
				*dz = d;
		}
	}
}

/* the offsets c along an axis of W: c = d + W r with the d in the window
 * that keeps them together; the d into d[], the r back into c[] */
static void
axis_window(int64_t *c, int n, int64_t W, int *d)
{
	int64_t best = 0, bw = -1, lo = 0, hi = 0, e;
	int i, sh;

	for (sh = 0; sh < W && sh < 64; sh++) {
		for (i = 0; i < n; i++) {
			e = floored_mod(c[i] + sh, W) - sh;
			if (!i || e < lo)
				lo = e;
			if (!i || e > hi)
				hi = e;
		}
		if (bw < 0 || hi - lo < bw) {
			bw = hi - lo;
			best = sh;
		}
	}
	for (i = 0; i < n; i++) {
		e = floored_mod(c[i] + best, W) - best;
		d[i] = (int)e;
		c[i] = (c[i] - e) / W;
	}
}

/*
 * A 3-D table: slices of H rows of W, the index's coefficients 1, W and
 * V = W H (V divides the table); the offsets (dx, dy, dz) in the windows
 * that keep dx, then dy, together.
 */
static int
grid_make3(struct sum *s, struct grid *g, int64_t W, int64_t V)
{
	const struct aff *a = &s->t[g->ix[0]].ix;
	int64_t c[MAXTERM];
	int d[MAXTERM], i;

	g->W = W;
	g->H = V / W;
	g->D = g->ti.n / V;
	g->xe = aff_part(a, 1, &g->xlo, &g->xhi);
	g->ye = aff_part(a, W, &g->ylo, &g->yhi);
	g->ze = aff_part(a, V, &g->zlo, &g->zhi);
	for (i = 0; i < a->n; i++)	/* a variable without a range */
		if (!(a->coef[i] == 1 ? g->xe : a->coef[i] == W ? g->ye : g->ze))
			return 0;
	if (!g->xe)
		g->xlo = g->xhi = 0;
	for (i = 0; i < g->n; i++)
		c[i] = s->t[g->ix[i]].ix.c;
	axis_window(c, g->n, W, d);
	for (i = 0; i < g->n; i++)
		s->t[g->ix[i]].dx = d[i];
	axis_window(c, g->n, g->H, d);
	for (i = 0; i < g->n; i++) {
		s->t[g->ix[i]].dy = d[i];
		s->t[g->ix[i]].dz = (int)c[i];
	}
	table_steps3(&g->ti, W, g->H, &g->dxmax, &g->dymax, &g->dzmax);
	return 1;
}

/*
 * Lay the group on the grid: the width from the index's coefficients
 * (rows of W when every variable's coefficient is 1 or W and W divides
 * the table; slices of rows with a second coefficient, grid_make3), the
 * offsets as (dx, dy) in the window that keeps dx together.
 */
static int
grid_make(struct sum *s, struct grid *g)
{
	const struct aff *a = &s->t[g->ix[0]].ix;
	int64_t W = 0, V = 0, cmin, best = 0, bw = -1;
	int i, sh;

	if (!table_info(s->t[g->ix[0]].tab, &g->ti))
		return 0;
	g->wrap = s->t[g->ix[0]].wrap;
	g->D = 1;
	for (i = 0; i < a->n; i++) {
		int64_t c = a->coef[i];
		if (c == 1)
			continue;
		if (g->wrap)	/* a wrapping read is along one axis */
			return 0;
		if (c < 1 || g->ti.n % c)
			return 0;
		if (!W || W == c)
			W = c;
		else if (!V || V == c)
			V = c;
		else
			return 0;
	}
	if (V) {
		int64_t t = V < W ? V : W;
		V = V < W ? W : V;
		return V % t == 0 && grid_make3(s, g, t, V);
	}
	g->W = W ? W : g->ti.n;
	g->H = g->ti.n / g->W;
	g->xe = aff_part(a, 1, &g->xlo, &g->xhi);
	g->ye = W ? aff_part(a, W, &g->ylo, &g->yhi) : 0;
	for (i = 0; i < a->n; i++)	/* a variable without a range */
		if (!(a->coef[i] == 1 ? g->xe : g->ye))
			return 0;
	if (!g->xe)
		g->xlo = g->xhi = 0;
	if (!g->ye)
		g->ylo = g->yhi = 0;
	cmin = s->t[g->ix[0]].ix.c;
	for (i = 1; i < g->n; i++)
		if (s->t[g->ix[i]].ix.c < cmin)
			cmin = s->t[g->ix[i]].ix.c;
	if (g->H == 1) {
		for (i = 0; i < g->n; i++) {
			s->t[g->ix[i]].dx = (int)s->t[g->ix[i]].ix.c;
			s->t[g->ix[i]].dy = 0;
		}
	} else {
		for (sh = 0; sh < g->W && sh < 64; sh++) {
			int64_t lo = 0, hi = 0, d;
			for (i = 0; i < g->n; i++) {
				d = floored_mod(s->t[g->ix[i]].ix.c + sh, g->W) - sh;
				if (!i || d < lo)
					lo = d;
				if (!i || d > hi)
					hi = d;
			}
			if (bw < 0 || hi - lo < bw) {
				bw = hi - lo;
				best = sh;
			}
		}
		for (i = 0; i < g->n; i++) {
			int64_t c = s->t[g->ix[i]].ix.c, d = floored_mod(c + best,
			    g->W) - best;
			s->t[g->ix[i]].dx = (int)d;
			s->t[g->ix[i]].dy = (int)((c - d) / g->W);
		}
	}
	table_steps(&g->ti, g->W, &g->dxmax, &g->dymax);
	return 1;
}

/* 2^e above every coordinate of the group's reads along an axis */
static int
coord_exp(int64_t lo, int64_t hi, int dmin, int dmax)
{
	double m = fabs((double)lo + dmin), n = fabs((double)hi + dmax + 2);
	int e;

	(void)frexp((m > n ? m : n) + 1, &e);
	return e;
}

/* one axis' coordinate: f32(index) + (d + 1/2 + fraction); what the unit
 * may move the weight by, into *dw */
static nid
coord_node(nid ie, int d, const struct mform *fr, int e, double *dw, int *ops,
    int *ongrid, int *cexp)
{
	double cst = d + 0.5 + fr->c[0], q = ldexp(cst, 24 - e);
	int exact = fr->c[1] == 0 && q == floor(q) && e <= 24;
	nid x = ie ? mk_cast(TY_F32, ie) : 0;
	struct mform f = *fr;

	/*
	 * f32(index) is exact; the constant is exact, and the add with it,
	 * when it is a multiple of the coordinates' ulp; else the constant
	 * and the add each round by half an ulp of a value below 2^e; a
	 * fraction that varies adds its product's rounding and an add's
	 */
	*ongrid = exact && ldexp(fr->c[0], X.bits) == floor(ldexp(fr->c[0],
	    X.bits));
	*cexp = exact ? -1000 : fr->c[1] == 0 ? e - 24 : e - 22;
	*dw = (exact ? 0 : ldexp(1.0, *cexp)) + (*ongrid ? 0 : ldexp(1.0,
	    -X.bits));
	f.c[0] = cst;
	if (!x)
		return mf_node(&f, ops);
	++*ops;		/* the conversion */
	++*ops;		/* and the add of the constant */
	if (fr->c[1] == 0)
		return mk_bin(O_ADD, x, kf(cst));
	return mk_bin(O_ADD, mk_bin(O_ADD, x, kf(cst)), mf_node(&(struct mform){
	    {0, fr->c[1], 0, 0}, fr->s, 0}, ops));
}

/*
 * A coordinate x below 2^e, normalised: u = x f32(1/n), and the unit
 * reads u n, within |x| 2^-23 of x (two roundings of 2^-24), none when n
 * is a power of two; that into *dw, *ongrid and *cexp as coord_node's.
 */
static nid
norm_coord(nid x, int64_t n, int e, double *dw, int *ongrid, int *cexp)
{
	double rn = (double)(float)(1.0 / (double)n);

	x = mk_bin(O_MUL, x, kf(rn));
	if (n & (n - 1)) {
		*dw += ldexp(1.0, e - 23);
		*ongrid = 0;
		if (*cexp < e - 23)
			*cexp = e - 23;
	}
	return x;
}

/*
 * The cover: each member is read by itself or is in one block (2 x 2, a
 * pair along x, a pair along y; of a 3-D table any box of the 2 x 2 x 2
 * cube) that derives; the cheapest under the model's prices, by dynamic
 * programming over the members left (the lowest one decides what is
 * tried), greedily past 16 members.
 */
#define NKIND 7
/* a kind's extent: the axes it spans (x 1, y 2, z 4), the biggest first */
static const int kind_ext2[3] = {3, 1, 2}, kind_ext3[NKIND] = {7, 3, 5, 6, 1,
    2, 4};

struct cover {
	const struct sum *s;
	const struct grid *g;
	struct block b[MAXTERM][NKIND];	/* by member and kind */
	uint8_t ok[MAXTERM][NKIND];
	uint32_t mask[MAXTERM][NKIND];
	int nk;			/* the kinds: 3 (a 2-D table), 7 (3-D) */
	float ptap, pone;
	int why;		/* BW_: the first block of neighbours refused */
	float *memo;
	uint8_t *pick;
};

static void
cover_blocks(struct cover *c)
{
	const struct sum *s = c->s;
	const struct grid *g = c->g;
	const int *ext = g->D > 1 ? kind_ext3 : kind_ext2;
	int i, k, q, why = BW_OK;

	c->nk = g->D > 1 ? NKIND : 3;
	for (i = 0; i < g->n; i++) {
		const struct term *t = &s->t[g->ix[i]];
		for (k = 0; k < c->nk; k++) {
			struct block *b = &c->b[i][k];
			int all = 1;
			c->ok[i][k] = 0;
			c->mask[i][k] = 0;
			for (q = 0; q < 8; q++) {
				int mi = -1;
				if (!(q & ~ext[k])) {
					mi = grid_at(s, g, t->dx + (q & 1),
					    t->dy + (q >> 1 & 1),
					    t->dz + (q >> 2 & 1));
					if (mi < 0)
						all = 0;
				}
				b->m[q] = mi >= 0 ? g->ix[mi] : -1;
				if (mi >= 0)
					c->mask[i][k] |= 1u << mi;
			}
			if (!all)
				continue;
			b->ox = t->dx;
			b->oy = t->dy;
			b->oz = t->dz;
			c->ok[i][k] = g->D > 1 ? block_derive3(s, b, &why) :
			    block_derive(s, b, &why);
			if (c->ok[i][k] && !block_inside(g, b)) {
				c->ok[i][k] = 0;
				why = BW_OUT;
			}
			if (!c->ok[i][k] && !c->why)
				c->why = why;
		}
	}
}

static float
cover_dp(struct cover *c, uint32_t done)
{
	int n = c->g->n, i, k, best = NKIND;
	float v, bv;

	if (done == (1u << n) - 1)
		return 0;
	if (c->memo[done] >= 0)
		return c->memo[done];
	for (i = 0; done >> i & 1; i++)
		;
	bv = c->pone + cover_dp(c, done | 1u << i);
	for (k = 0; k < c->nk; k++) {
		if (!c->ok[i][k] || (c->mask[i][k] & done))
			continue;
		v = c->ptap + cover_dp(c, done | c->mask[i][k]);
		if (v < bv) {
			bv = v;
			best = k;
		}
	}
	c->memo[done] = bv;
	c->pick[done] = (uint8_t)best;
	return bv;
}

/* the members a mask holds */
static int
mask_count(uint32_t m)
{
	int n = 0;

	for (; m; m &= m - 1)
		n++;
	return n;
}

/* the blocks chosen, into out; their count */
static int
cover_pick(struct cover *c, struct block *out)
{
	int n = c->g->n, i, k, nb = 0;
	uint32_t done = 0;

	if (n <= 16) {
		c->memo = xmalloc(((size_t)1 << n) * sizeof *c->memo);
		c->pick = xcalloc((size_t)1 << n, 1);
		for (i = 0; i < 1 << n; i++)
			c->memo[i] = -1;
		cover_dp(c, 0);
		while (done != (1u << n) - 1) {
			for (i = 0; done >> i & 1; i++)
				;
			k = c->pick[done];
			if (k < NKIND) {
				out[nb++] = c->b[i][k];
				done |= c->mask[i][k];
			} else
				done |= 1u << i;
		}
		free(c->memo);
		free(c->pick);
		return nb;
	}
	for (i = 0; i < n; i++) {
		if (done >> i & 1)
			continue;
		for (k = 0; k < c->nk; k++)
			if (c->ok[i][k] && !(c->mask[i][k] & done) && c->ptap <
			    c->pone * mask_count(c->mask[i][k]))
				break;
		if (k < c->nk) {
			out[nb++] = c->b[i][k];
			done |= c->mask[i][k];
		} else
			done |= 1u << i;
	}
	return nb;
}

/*
 * A cover that shares weights: a separable kernel (constant weights over
 * a filled rectangle, a column of them times a row, all of one sign) is
 * cut along each axis into pairs, the last pair of an odd length sharing
 * its first entry with the pair before it, and a read is a pair along y
 * times a pair along x: the reads' weights add up to the kernel's, and a
 * 3 x 3 kernel is four reads where the cover above takes three and an
 * entry.  The shared entry is split so that the first pair's fraction is
 * on the 2^-bits grid and the second's as near it as any split puts it
 * (both on it for [1 2 1]).
 */
struct apair {
	int a, b;		/* the entries on the axis, b -1: a lone one */
	double wa, wb;		/* the weights this pair reads of them */
};

static int
axis_pairs(const double *v, int n, struct apair *p)
{
	double g = ldexp(1.0, X.bits), a, m, c, x, f, fb, e, best = -1, bx;
	int np = 0, i, k;

	for (i = 0; i + 1 < n; i += 2)
		p[np++] = (struct apair){i, i + 1, v[i], v[i + 1]};
	if (n % 2 == 0)
		return np;
	if (n == 1) {
		p[np++] = (struct apair){0, -1, v[0], 0};
		return np;
	}
	a = v[n - 3];
	m = v[n - 2];
	c = v[n - 1];
	bx = m / 2;
	for (k = 1; k < 1 << X.bits; k++) {
		f = k / g;
		x = a * f / (1 - f);
		if (!(x > 0 && x < m))
			continue;
		fb = c / (m - x + c);
		e = fabs(fb * g - nearbyint(fb * g));
		if (best < 0 || e < best - 1e-12 || (e <= best + 1e-12 &&
		    fabs(x - m / 2) < fabs(bx - m / 2))) {
			best = e;
			bx = x;
		}
	}
	p[np - 1].wb = bx;
	p[np++] = (struct apair){n - 2, n - 1, m - bx, c};
	return np;
}

static int
cover_split(const struct sum *s, const struct grid *g, struct block *out)
{
	double W[MAXTERM], cv[MAXTERM], rv[MAXTERM], sg, got[MAXTERM] = {0};
	struct apair pc[MAXTERM], pr[MAXTERM];
	int dmin = 0, dmax = 0, emin = 0, emax = 0, C, R, i, x, y, nc, nr, P, Q,
	    nb = 0, why;

	if (g->D > 1)	/* a 2-D kernel's: not cut along z */
		return 0;
	for (i = 0; i < g->n; i++) {
		const struct term *t = &s->t[g->ix[i]];
		if (!mf_isconst(&t->k))
			return 0;
		if (!i || t->dx < dmin)
			dmin = t->dx;
		if (!i || t->dx > dmax)
			dmax = t->dx;
		if (!i || t->dy < emin)
			emin = t->dy;
		if (!i || t->dy > emax)
			emax = t->dy;
	}
	C = dmax - dmin + 1;
	R = emax - emin + 1;
	/* a filled rectangle with an odd side: even sides need no share */
	if (C * R != g->n || (C % 2 == 0 && R % 2 == 0) || g->n > MAXTERM)
		return 0;
	for (i = 0; i < g->n; i++) {
		const struct term *t = &s->t[g->ix[i]];
		W[(t->dy - emin) * C + t->dx - dmin] = t->k.c[0];
	}
	sg = W[0] < 0 ? -1 : 1;
	for (i = 0; i < C * R; i++)
		if (!(W[i] * sg > 0))
			return 0;
	for (x = 0; x < C; x++)
		rv[x] = W[x] * sg;
	for (y = 0; y < R; y++)
		cv[y] = W[y * C] / W[0];
	for (y = 0; y < R; y++)
		for (x = 0; x < C; x++)
			if (!(fabs(cv[y] * rv[x] - W[y * C + x] * sg) <= 1e-12 *
			    fabs(W[y * C + x])))
				return 0;
	nc = axis_pairs(rv, C, pc);
	nr = axis_pairs(cv, R, pr);
	for (P = 0; P < nr; P++)
		for (Q = 0; Q < nc; Q++) {
			struct block *b = &out[nb++];
			struct mform w[4];
			int ax = pc[Q].a, bxi = pc[Q].b, ay = pr[P].a,
			    by = pr[P].b;
			int at[4][2] = {{ax, ay}, {bxi, ay}, {ax, by},
			    {bxi, by}};
			double ww[4] = {pr[P].wa * pc[Q].wa,
			    pr[P].wa * pc[Q].wb, pr[P].wb * pc[Q].wa,
			    pr[P].wb * pc[Q].wb};
			memset(b, 0, sizeof *b);
			for (i = 4; i < 8; i++)
				b->m[i] = -1;
			for (i = 0; i < 4; i++) {
				int mi = at[i][0] < 0 || at[i][1] < 0 ? -1 :
				    grid_at(s, g, dmin + at[i][0],
				    emin + at[i][1], 0);
				b->m[i] = mi >= 0 ? g->ix[mi] : -1;
				w[i] = mf_const(b->m[i] >= 0 ? ww[i] * sg : 0);
				if (mi >= 0)
					got[mi] += ww[i] * sg;
				if (b->m[i] >= 0 && fabs(ww[i] * sg -
				    s->t[b->m[i]].k.c[0]) > 1e-12 * fabs(ww[i]))
					b->split = 1;
			}
			b->ox = dmin + ax;
			b->oy = emin + ay;
			if (!block_derive_w(w, b, &why) || !block_inside(g, b))
				return 0;
		}
	/* the shares of each entry add up to its weight */
	for (i = 0; i < g->n; i++)
		if (!(fabs(got[i] - s->t[g->ix[i]].k.c[0]) <= 1e-12 *
		    fabs(s->t[g->ix[i]].k.c[0])))
			return 0;
	return nb;
}

/* every block's fractions constants on the 2^-bits grid: no snapping */
static int
on_grid(const struct block *b, int nb)
{
	double g = ldexp(1.0, X.bits);
	int k;

	for (k = 0; k < nb; k++)
		if (b[k].fu.c[1] != 0 || b[k].fv.c[1] != 0 ||
		    b[k].fw.c[1] != 0 ||
		    b[k].fu.c[0] * g != nearbyint(b[k].fu.c[0] * g) ||
		    b[k].fv.c[0] * g != nearbyint(b[k].fv.c[0] * g) ||
		    b[k].fw.c[0] * g != nearbyint(b[k].fw.c[0] * g))
			return 0;
	return 1;
}

static void
mf_text(struct sbuf *b, const struct mform *f)
{
	if (f->c[1] == 0) {
		sb_printf(b, "%.6g", f->c[0]);
		return;
	}
	if (f->c[0] != 0)
		sb_printf(b, "%.6g %c ", f->c[0], f->c[1] < 0 ? '-' : '+');
	else if (f->c[1] < 0)
		sb_putc(b, '-');
	if (fabs(f->c[1]) != 1)
		sb_printf(b, "%.6g ", fabs(f->c[1]));
	sb_puts(b, node_str(f->s));
}

/* the largest |a - b| of two roots over the domain's points, and how many */
static double
max_diff(nid a, nid b, uint64_t *pts, int *sampled)
{
	struct plan p;
	struct diter d;
	nid roots[3];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r), md = 0;
	uint8_t ok[CH];
	uint32_t np, i, la, lb, wl;

	roots[0] = a;
	roots[1] = b;
	roots[2] = X.f->where;
	plan_build(&p, roots, 3, 0);
	la = p.root[0];
	lb = p.root[1];
	wl = X.f->where ? plan_loc(&p, X.f->where) : NOTOK;
	diter_init(&d, X.f, 1u << 16, 65, 4096);
	*pts = 0;
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wl, np, ok);
		for (i = 0; i < np; i++) {
			double va, vb;
			if (!ok[i] || (COLF(&p, la)[i] & F_UNDEF))
				continue;
			va = TY_FP(N.ty[a]) ? COLR(&p, la)[i] : (double)COLI(&p,
			    la)[i];
			vb = TY_FP(N.ty[b]) ? COLR(&p, lb)[i] : (double)COLI(&p,
			    lb)[i];
			++*pts;
			if ((COLF(&p, lb)[i] & F_UNDEF) || !(fabs(va - vb) <= md))
				md = (COLF(&p, lb)[i] & F_UNDEF) ? INFINITY :
				    fabs(va - vb);
		}
	}
	*sampled = !d.exhaustive;
	plan_free(&p);
	free(in_i);
	free(in_r);
	return md;
}

static void
seed_sum1(nid body, nid root)
{
	struct sum *s = xcalloc(1, sizeof *s);
	struct block blk[MAXTERM], sblk[MAXTERM];
	struct repl *r = NULL;
	nid taps = 0, to, ity_root = 0;
	double enew = 0, mabs = 0, mtap = 0, u = ldexp(1.0, -24);
	int i, j, ng = 0, ntap = 0, nleft = 0, isint, nterm = 0, njoin = 0;
	int allpos = 1, approx = 0;
	struct sbuf what = {0}, holds = {0}, proof = {0}, pipe = {0};

	(void)ity_root;
	isint = s->isint = TY_ISINT(N.ty[root]);
	collect(s, root, mf_const(1), 0);
	if (s->bad || s->nt < 2)
		goto out;
	for (i = 0; i < s->nt; i++)
		s->t[i].grp = -1;
	/* a wrapping read with no neighbour that wraps with it is read as
	 * it is: its index an affine form like any other */
	for (i = 0; i < s->nt; i++) {
		if (!s->t[i].wrap || !s->t[i].ok0)
			continue;
		for (j = 0; j < s->nt; j++)
			if (j != i && s->t[j].wrap == s->t[i].wrap &&
			    same_table(s->t[j].tab, s->t[i].tab) &&
			    aff_same_base(&s->t[i].ix, &s->t[j].ix))
				break;
		if (j == s->nt) {
			s->t[i].ix = s->t[i].ix0;
			s->t[i].wrap = 0;
		}
	}
	for (i = 0; i < s->nt; i++) {
		if (s->t[i].grp >= 0)
			continue;
		s->t[i].grp = ng;
		for (j = i + 1; j < s->nt; j++)
			if (s->t[j].grp < 0 && same_table(s->t[j].tab, s->t[i].tab) &&
			    s->t[j].wrap == s->t[i].wrap &&
			    aff_same_base(&s->t[i].ix, &s->t[j].ix)) {
				s->t[j].grp = ng;
				/* the same entry twice: one weight */
				if (s->t[j].ix.c == s->t[i].ix.c) {
					if (!mf_add(&s->t[i].k, s->t[i].k, s->t[j].k, 1))
						goto out;
					s->t[j].used = 2;
				}
			}
		ng++;
	}
	for (i = 0; i < s->nt; i++) {
		struct tinfo ti;
		if (s->t[i].used == 2)
			continue;
		table_info(s->t[i].tab, &ti);
		mabs += mf_mag(&s->t[i].k) * ti.tmax;
		nterm++;
	}
	if (!isfinite(mabs))
		goto out;
	for (j = 0; j < ng; j++) {
		struct grid g;
		struct cover cv;
		int nb, ns, k, dmin = 0, dmax = 0, emin = 0, emax = 0, ex, ey;
		int fmin = 0, fmax = 0, ez = 0, tri = 0;
		double sx, sy;

		memset(&g, 0, sizeof g);
		for (i = 0; i < s->nt; i++)
			if (s->t[i].grp == j && s->t[i].used != 2)
				g.ix[g.n++] = i;
		if (g.n < 2 || !grid_make(s, &g))
			continue;
		if (!model_lists(X.m, g.ti.cls)) {
			X.unavail[g.ti.cls] = 1;
			continue;
		}
		tri = g.D > 1;
		if (tri && !cclass_available(X.m, C_TEX_3D)) {
			X.unavail_3d = 1;
			continue;
		}
		if (!X.bits) {
			X.nobits = 1;
			continue;
		}
		if (X.ulp == 0) {
			X.noulp = 1;
			continue;
		}
		/* by slice, row, then column: the lowest member is a block's
		 * corner */
		for (i = 1; i < g.n; i++)
			for (k = i; k > 0; k--) {
				const struct term *a = &s->t[g.ix[k - 1]], *b =
				    &s->t[g.ix[k]];
				int t;
				if (a->dz < b->dz || (a->dz == b->dz && (a->dy <
				    b->dy || (a->dy == b->dy && a->dx < b->dx))))
					break;
				t = g.ix[k];
				g.ix[k] = g.ix[k - 1];
				g.ix[k - 1] = t;
			}
		for (i = 0; i < g.n; i++) {
			const struct term *a = &s->t[g.ix[i]];
			if (!i || a->dx < dmin)
				dmin = a->dx;
			if (!i || a->dx > dmax)
				dmax = a->dx;
			if (!i || a->dy < emin)
				emin = a->dy;
			if (!i || a->dy > emax)
				emax = a->dy;
			if (!i || a->dz < fmin)
				fmin = a->dz;
			if (!i || a->dz > fmax)
				fmax = a->dz;
		}
		memset(&cv, 0, sizeof cv);
		cv.s = s;
		cv.g = &g;
		cv.ptap = X.m->c[g.ti.cls] + (tri ? 3 : g.H > 1 ? 2 : 1) *
		    X.m->c[C_FADD] + X.m->c[C_FMUL] + (g.wrap ? X.m->c[C_FMUL] :
		    0);
		/* a 3-D read: the third axis, and the coordinates normalised */
		if (tri)
			cv.ptap += X.m->c[C_TEX_3D] + 3 * X.m->c[C_FMUL];
		cv.pone = X.m->c[C_LOAD] + (isint ? X.m->c[C_IADD] :
		    X.m->c[C_FMUL]);
		cover_blocks(&cv);
		nb = cover_pick(&cv, blk);
		/*
		 * a cover that shares weights, when it is cheaper than this
		 * one, or as cheap and this one snaps a fraction to the grid
		 * and that one none; with no tolerance given (a snap is an
		 * error the sum as written does not have), also when it costs
		 * less than the entries read alone
		 */
		if ((ns = cover_split(s, &g, sblk)) > 0) {
			int nin = 0, q, exact;
			float cd, cs = ns * cv.ptap;
			for (k = 0; k < nb; k++)
				for (q = 0; q < 4; q++)
					nin += blk[k].m[q] >= 0;
			cd = nb * cv.ptap + (g.n - nin) * cv.pone;
			exact = nb > 0 && !on_grid(blk, nb) &&
			    on_grid(sblk, ns);
			if (cs < cd || (exact && cs <= cd) || (exact &&
			    X.tol == 0 && cs < g.n * cv.pone)) {
				memcpy(blk, sblk, (size_t)ns * sizeof *blk);
				nb = ns;
			}
		}
		if (nb == 0) {
			/* no tap is cheaper than the reads it replaces: the
			 * cover with the most taps, to say so with its price */
			cv.ptap = 0;
			nb = cover_pick(&cv, blk);
		}
		if (nb == 0) {
			/* neighbours, and no block of them is a read: say why */
			if (cv.why && !X.nosum.n) {
				sb_printf(&X.nosum, "entries of the %lld x %lld",
				    (long long)g.W, (long long)g.H);
				if (tri)
					sb_printf(&X.nosum, " x %lld", (long long)g.D);
				sb_printf(&X.nosum, " table are neighbours in %s, but ",
				    shared_str(root, 120));
				if (cv.why == BW_OUT) {
					sb_puts(&X.nosum, "a read of them could leave the "
					    "table:");
					if (g.xe)
						sb_printf(&X.nosum, " %s in [%lld, %lld]",
						    node_str(g.xe), (long long)g.xlo,
						    (long long)g.xhi);
					if (g.ye)
						sb_printf(&X.nosum, "%s %s in [%lld, %lld]",
						    g.xe ? "," : "", node_str(g.ye),
						    (long long)g.ylo, (long long)g.yhi);
					if (g.ze)
						sb_printf(&X.nosum, ", %s in [%lld, %lld]",
						    node_str(g.ze), (long long)g.zlo,
						    (long long)g.zhi);
					sb_puts(&X.nosum, " (a filter clamps at the edge, "
					    "the table's index does not)");
				} else
					sb_puts(&X.nosum, cv.why == BW_SUM ? "their "
					    "weights do not sum to a constant other "
					    "than 0" : cv.why == BW_SIGN ? "their "
					    "weights are not all of one sign for every "
					    "input (a fraction leaves [0, 1], or its "
					    "variable has no range)" : tri ? "their weights "
					    "are not the product of a fraction along each "
					    "of x, y and z" : "their weights "
					    "are not the product of a fraction along x "
					    "and one along y");
			}
			continue;
		}
		ex = coord_exp(g.xlo, g.xhi, dmin, dmax);
		ey = coord_exp(g.ylo, g.yhi, emin, emax);
		if (tri)
			ez = coord_exp(g.zlo, g.zhi, fmin, fmax);
		sx = sy = 0;
		for (k = 0; k < nb; k++) {
			struct block *b = &blk[k];
			double dwx, dwy = 0, dwz = 0, e, S = b->S;
			int ops = 0, gx, gy = 1, gz = 1, cnt = 0, q, cx, cy = -1000,
			    cz = -1000;
			nid x, y = 0, z = 0, tap;
			double lo, hi;

			x = coord_node(g.xe, b->ox, &b->fu, ex, &dwx, &ops, &gx,
			    &cx);
			if (tri) {
				/* normalised: a 3-D texture has no
				 * unnormalizedCoordinates */
				y = coord_node(g.ye, b->oy, &b->fv, ey, &dwy, &ops,
				    &gy, &cy);
				z = coord_node(g.ze, b->oz, &b->fw, ez, &dwz, &ops,
				    &gz, &cz);
				x = norm_coord(x, g.W, ex, &dwx, &gx, &cx);
				y = norm_coord(y, g.H, ey, &dwy, &gy, &cy);
				z = norm_coord(z, g.D, ez, &dwz, &gz, &cz);
				tap = mk_tex(O_TEX_TRILINEAR, TEX_KI(X.bits, g.W) |
				    TEX_KH(g.H), g.ti.tex, x, tex_args(y, z));
			} else if (g.H > 1) {
				y = coord_node(g.ye, b->oy, &b->fv, ey, &dwy, &ops,
				    &gy, &cy);
				tap = mk_tex(O_TEX_BILINEAR, TEX_KI(X.bits, g.W),
				    g.ti.tex, x, y);
			} else if (g.wrap) {
				/*
				 * normalised: u = x f32(1/n), and the unit reads
				 * u n: within |x| 2^-23 of x (two roundings of
				 * 2^-24), none when n is a power of two
				 */
				x = norm_coord(x, g.W, ex, &dwx, &gx, &cx);
				tap = mk_tex(O_TEX_LINEAR, TEX_KI(X.bits, 0) |
				    TEX_KWRAP(g.wrap), g.ti.tex, x, 0);
			} else
				tap = mk_tex(O_TEX_LINEAR, TEX_KI(X.bits, 0),
				    g.ti.tex, x, 0);
			if (S * g.ti.scale != 1)
				tap = mk_bin(O_MUL, kf(S * g.ti.scale), tap);
			approx |= g.ti.approx;
			if (g.ti.off != 0)	/* the levels' map back, with the product */
				tap = mk_bin(O_ADD, tap, kf(S * g.ti.off));
			if (S < 0)
				allpos = 0;
			if (taps)
				njoin++;
			taps = taps ? mk_bin(O_ADD, taps, tap) : tap;
			for (q = 0; q < 8; q++)
				if (b->m[q] >= 0) {
					s->t[b->m[q]].used = 1;
					cnt++;
				}
			/* the snapped weights, the filter, the texels' rounding,
			 * the weights not reproduced, and the map back to the
			 * entries: the scale's constant and product, the offset's
			 * constant and add */
			e = fabs(S) * (dwx * g.dxmax + dwy * g.dymax + dwz *
			    g.dzmax + X.ulp *
			    2 * u * g.ti.rmax + g.ti.trep) + b->resid * g.ti.tmax +
			    ((S * g.ti.scale != 1 ? 2 : 0) + (g.ti.off != 0 ? 2 : 0)) *
			    u * fabs(S) * (fabs(g.ti.off) + g.ti.rmax);
			enew += e;
			mtap += fabs(S) * g.ti.tmax;
			ntap++;
			if (tri)
				sb_printf(&what, "%s%d entr%s of the %lld x %lld x "
				    "%lld table from offset (%d, %d, %d), weights' sum "
				    "%.6g: one read at fraction ", what.n ? "\n" : "",
				    cnt, cnt == 1 ? "y" : "ies", (long long)g.W,
				    (long long)g.H, (long long)g.D, b->ox, b->oy,
				    b->oz, S);
			else
				sb_printf(&what, "%s%d entr%s of the %lld x %lld "
				    "table from offset (%d, %d), weights' sum %.6g%s: "
				    "one read at fraction ", what.n ? "\n" : "", cnt,
				    cnt == 1 ? "y" : "ies", (long long)g.W,
				    (long long)g.H, b->ox, b->oy, S, b->split ?
				    " (a share of a weight read twice)" : "");
			mf_text(&what, &b->fu);
			if (g.H > 1) {
				sb_puts(&what, " along x, ");
				mf_text(&what, &b->fv);
				sb_puts(&what, " along y");
			}
			if (tri) {
				sb_puts(&what, ", ");
				mf_text(&what, &b->fw);
				sb_puts(&what, " along z");
			}
			if (tri) {
				sb_printf(&proof, "%sa linear read of a 3-D texture at "
				    "fractions (u, v, w) of the 2 x 2 x 2 texels is the "
				    "sum of each texel times (1-u or u)(1-v or v)(1-w "
				    "or w), u along x, v along y, w along z: times %.6g "
				    "with u = ", proof.n ? "\n" : "", S * g.ti.scale);
				mf_text(&proof, &b->fu);
				sb_puts(&proof, ", v = ");
				mf_text(&proof, &b->fv);
				sb_puts(&proof, ", w = ");
				mf_text(&proof, &b->fw);
			} else if (g.H > 1) {
				sb_printf(&proof, "%sa linear read at fractions (u, v) "
				    "of texels a b / c d is (1-u)(1-v) a + u(1-v) b + "
				    "(1-u)v c + uv d: times %.6g with u = ", proof.n ?
				    "\n" : "", S * g.ti.scale);
				mf_text(&proof, &b->fu);
				sb_puts(&proof, ", v = ");
				mf_text(&proof, &b->fv);
			} else {
				sb_printf(&proof, "%sa linear read at fraction u of "
				    "texels a, b is (1-u) a + u b: times %.6g with u "
				    "= ", proof.n ? "\n" : "", S * g.ti.scale);
				mf_text(&proof, &b->fu);
			}
			for (q = 0; q < (tri ? 3 : 2); q++) {
				const struct mform *fr = q == 2 ? &b->fw : q ? &b->fv :
				    &b->fu;
				if (fr->c[1] == 0 || !frange(fr->s, &lo, &hi))
					continue;
				sb_printf(&holds, "%s%s in [%g, %g] (its range): the "
				    "fraction ", holds.n ? "; " : "holds where ",
				    node_str(fr->s), lo, hi);
				mf_text(&holds, fr);
				sb_puts(&holds, " is in [0, 1]");
			}
			sb_printf(&proof, " those are the entries' weights%s",
			    b->split ? ", an entry read by two reads with its "
			    "weight split between them (column times row: the "
			    "pairs' shares sum to the kernel's weights)" : "");
			if (!g.ti.levels && g.ti.scale != 1)
				sb_printf(&proof, " (the texels are the "
				    "entries / %g)", g.ti.scale);
			if (g.ti.levels)
				sb_printf(&proof, " (an entry is %.9g + %.9g "
				    "times its texel k/%d: the weights' sum times "
				    "%.9g is added back)", g.ti.off, g.ti.scale,
				    g.ti.q, g.ti.off);
			if (b->resid > 0)
				sb_printf(&proof, " to within %s", fmt_g(b->resid));
			if (gx && gy && gz)
				sb_printf(&proof, "\n  the fractions are on the 2^-%d "
				    "grid and the coordinates exact in f32: the "
				    "weights do not move", X.bits);
			else if (tri) {
				int cm = cx > cy ? cx : cy;
				sb_printf(&proof, "\n  a fraction not on the grid is "
				    "snapped by less than 2^-%d, and f32 moves a "
				    "coordinate that is not exact by at most 2^%d: "
				    "the read moves by at most %s (x), %s (y) and %s "
				    "(z) times the largest step between neighbours, "
				    "%s, %s and %s", X.bits, cm > cz ? cm : cz,
				    fmt_g(dwx), fmt_g(dwy), fmt_g(dwz), fmt_g(g.dxmax),
				    fmt_g(g.dymax), fmt_g(g.dzmax));
			} else
				sb_printf(&proof, "\n  a fraction not on the grid is "
				    "snapped by less than 2^-%d, and f32 moves a "
				    "coordinate that is not exact by at most 2^%d: "
				    "the read moves by at most %s (x) and %s (y) "
				    "times the largest step between neighbours, %s "
				    "and %s", X.bits, cx > cy ? cx : cy, fmt_g(dwx),
				    fmt_g(dwy), fmt_g(g.dxmax), fmt_g(g.dymax));
			sb_printf(&proof, "\n  the filter is within %g ulp of that "
			    "sum (the model's tex_within): %s; the texels as "
			    "%s: %s; in all %s", X.ulp, fmt_g(fabs(S) * X.ulp * 2 *
			    u * g.ti.rmax), g.ti.cls == C_TEX_SAMPLE_F16 ? "f16" :
			    "f32", fmt_g(fabs(S) * g.ti.trep), fmt_g(e));
		}
		sb_printf(&holds, "%s", holds.n ? "; " : "holds where ");
		if (g.xe)
			sb_printf(&holds, "%s in [%lld, %lld]", node_str(g.xe),
			    (long long)g.xlo, (long long)g.xhi);
		if (g.ye)
			sb_printf(&holds, "%s%s in [%lld, %lld]", g.xe ? ", " : "",
			    node_str(g.ye), (long long)g.ylo, (long long)g.yhi);
		if (g.ze)
			sb_printf(&holds, ", %s in [%lld, %lld]", node_str(g.ze),
			    (long long)g.zlo, (long long)g.zhi);
		if (g.wrap)
			sb_printf(&holds, "%sthe address mode (%s) wraps every "
			    "texel into the table", g.xe || g.ye ? " (proved "
			    "ranges): " : "", mode_vk(g.wrap));
		else
			sb_printf(&holds, "%severy texel read is inside the "
			    "table", g.xe || g.ye || g.ze ? " (proved ranges): " :
			    "");
		if (g.wrap)
			sb_printf(&proof, "\n  the coordinate is normalised, u = "
			    "x f32(1/%lld), and the unit reads u %lld: %s",
			    (long long)g.W, (long long)g.W, g.W & (g.W - 1) ?
			    "within |x| 2^-23 of x (two roundings of 2^-24), in "
			    "the bound above" : "exact: the size is a power of "
			    "two");
		if (tri)
			sb_printf(&proof, "\n  the coordinates are normalised, u = "
			    "x f32(1/%lld), v = y f32(1/%lld), w = z f32(1/%lld), "
			    "and the unit reads (u %lld, v %lld, w %lld): %s",
			    (long long)g.W, (long long)g.H, (long long)g.D,
			    (long long)g.W, (long long)g.H, (long long)g.D,
			    (g.W & (g.W - 1)) || (g.H & (g.H - 1)) || (g.D & (g.D -
			    1)) ? "along a size that is not a power of two within "
			    "|x| 2^-23 of x (two roundings of 2^-24), in the bound "
			    "above" : "exact: the sizes are powers of two");
		if (tri)
			sb_printf(&pipe, "%sa 3-D texture %s, %lld x %lld x %lld, "
			    "one level: the table's entries", pipe.n ? "\n" : "",
			    fmt_vk(g.ti.cls), (long long)g.W, (long long)g.H,
			    (long long)g.D);
		else
			sb_printf(&pipe, "%sa texture %s, %lld x %lld, one level: "
			    "the table's entries", pipe.n ? "\n" : "",
			    fmt_vk(g.ti.cls), (long long)g.W, (long long)g.H);
		fmt_says(&pipe, &g.ti);
		if (tri)
			sb_printf(&pipe, "\na sampler: LINEAR mag and min filters, "
			    "NEAREST mipmap mode, CLAMP_TO_EDGE on u, v and w, "
			    "normalised coordinates (a 3-D texture has no "
			    "unnormalizedCoordinates), lod 0\none combined image "
			    "sampler more in the layout; the shader reads "
			    "textureLod(s, vec3(u, v, w), 0.0).r");
		else
			sb_printf(&pipe, "\na sampler: LINEAR mag and min filters, "
			    "NEAREST mipmap mode, %s, lod 0\none combined image "
			    "sampler more in the layout; the shader reads "
			    "textureLod(s, %s, 0.0).r", g.wrap ? (g.wrap ==
			    O_FETCH_REPEAT ? "REPEAT, normalised coordinates" :
			    "MIRRORED_REPEAT, normalised coordinates") :
			    "CLAMP_TO_EDGE, unnormalizedCoordinates", g.H > 1 ?
			    "vec2(x, y)" : g.wrap ? "u" : "x");
		if (g.ti.cls == C_TEX_SAMPLE && g.H > 1 && !tri && !g.ti.levels) {
			/* the same texels in half the bytes, when BC4's palettes
			 * hold them exactly (texenc.c) */
			struct texenc te;
			if (texenc_bc4(g.ti.tab, (int)g.W, &te)) {
				if (te.err == 0 && te.bw * te.bh * 8 < g.W * g.H)
					sb_printf(&pipe, "\nthe same texels are exact as "
					    "BC4_UNORM blocks, %d bytes against %lld "
					    "(texenc; a BC4 read is not priced apart)",
					    te.bw * te.bh * 8, (long long)(g.W * g.H));
				texenc_free(&te);
			}
		}
		(void)sx;
		(void)sy;
	}
	if (!ntap)
		goto out;
	/* the new sum: the taps, the entries left, the other terms */
	if (isint) {
		enew += u * (mtap + 1) + njoin * u * mtap;
		/* a sum that is never negative: the conversion's truncation
		 * is the floor */
		to = mk_bin(O_ADD, taps, kf(0.5));
		to = mk_cast(N.ty[root], allpos ? to : mk_un(O_FLOOR, to));
		for (i = 0; i < s->nt; i++)
			if (!s->t[i].used) {
				nid e = mk_index(s->t[i].tab, s->tidx[i]);
				double k = s->t[i].k.c[0];
				if (k != 1 && k != -1)
					e = mk_bin(O_MUL, mk_const_i(N.ty[root],
					    (int64_t)fabs(k)), e);
				to = mk_bin(k < 0 ? O_SUB : O_ADD, to, e);
				nleft++;
			}
		for (i = 0; i < s->nr; i++) {
			nid e = s->r[i].e;
			double k = s->r[i].k.c[0];
			if (k != 1 && k != -1)
				e = mk_bin(O_MUL, mk_const_i(N.ty[root],
				    (int64_t)fabs(k)), e);
			to = mk_bin(k < 0 ? O_SUB : O_ADD, to, e);
		}
	} else {
		int ops = 0;
		to = taps;
		for (i = 0; i < s->nt; i++)
			if (!s->t[i].used) {
				nid e = mk_bin(O_MUL, mf_node(&s->t[i].k, &ops),
				    mk_cast(TY_F32, mk_index(s->t[i].tab,
				    s->tidx[i])));
				to = mk_bin(O_ADD, to, e);
				nleft++;
			}
		enew += (nleft * 2 + njoin) * u * mabs;
		for (i = 0; i < s->nr; i++) {
			nid e = s->r[i].e;
			if (!mf_isconst(&s->r[i].k) || s->r[i].k.c[0] != 1)
				e = mk_bin(O_MUL, mf_node(&s->r[i].k, &ops), e);
			to = mk_bin(O_ADD, to, e);
		}
		if (N.ty[to] != N.ty[root])
			to = mk_cast(N.ty[root], to);
	}
	if ((r = new_repl(root)) == NULL)
		goto out;
	r->to = to;
	r->filtered = 1;
	r->approx = approx;
	r->exact = isint;
	r->err = enew;
	r->eorig = isint ? 0 : (nterm * u / (1 - nterm * u)) * mabs;
	r->cost = price(swap_nodes(body, &root, &to, 1));
	r->what = what;
	r->holds = holds;
	r->proof = proof;
	r->pipe = pipe;
	memset(&what, 0, sizeof what);
	memset(&holds, 0, sizeof holds);
	memset(&proof, 0, sizeof proof);
	memset(&pipe, 0, sizeof pipe);
	sb_printf(&r->pipe, "\nsetup (the descriptor, the upload) is not in the "
	    "model: not measured");
	if (isint) {
		sb_printf(&r->proof, "\nthe entries' sum is a whole number and the "
		    "reads are within %s of it (with the f32 products and "
		    "adds): %s", fmt_g(enew), enew < 0.5 - 1e-6 ? "below 1/2, so "
		    "floor(v + 1/2) is the sum" : "not below 1/2: not exact");
		if (allpos && enew < 0.5 - 1e-6)
			sb_puts(&r->proof, " (never negative: the conversion to "
			    "an integer truncates, which is that floor)");
		r->ok = enew < 0.5 - 1e-6;
		if (!r->ok)
			sb_printf(&r->why, "the reads are within %s of the sum, "
			    "not below 1/2", fmt_g(enew));
	} else {
		sb_printf(&r->proof, "\nagainst the exact weighted sum of the "
		    "entries the reads are within %s; as written, the sum's own "
		    "f32 rounding is within %s (gamma_%d times the sum of |w "
		    "t|, %s)", fmt_g(enew), fmt_g(r->eorig), nterm, fmt_g(mabs));
		r->ok = 1;
	}
	if (r->ok) {
		uint64_t pts;
		int sampled;
		double md = max_diff(root, to, &pts, &sampled), lim = isint ? 0 :
		    (r->err + r->eorig) * (1 + 1e-9);
		if (!(md <= lim)) {
			r->ok = 0;
			sb_printf(&r->why, "evaluated on %llu points the two "
			    "differ by %s, more than the derived %s: not offered",
			    (unsigned long long)pts, fmt_g(md), fmt_g(lim));
		} else
			sb_printf(&r->proof, "\nchecked again by evaluation with "
			    "ucas's own filter (exact weights' sums, one "
			    "rounding) on %s%llu points: %s%s", sampled ?
			    "a sample of " : "all ", (unsigned long long)pts,
			    isint ? "equal" : "largest difference ", isint ? "" :
			    fmt_g(md));
	}
out:
	free(what.p);
	free(holds.p);
	free(proof.p);
	free(pipe.p);
	free(s);
}

/*
 * A float table whose texels only approximate it (evenly spaced levels,
 * entries rounded to f16) is read that way (a cheaper class) when its
 * bound is within what is allowed; when not, again without that
 * approximation, down to f32.
 */
static void
seed_sum(nid body, nid root)
{
	int n0 = X.nr, tries;
	struct repl *r;

	for (tries = 0; tries < 4; tries++) {
		seed_sum1(body, root);
		if (X.nr == n0)
			break;
		r = &X.r[X.nr - 1];
		if (!r->approx || r->exact || r->err <= (X.tol > r->eorig ?
		    X.tol : r->eorig))
			break;
		X.fmtban |= r->approx;
		drop_repl(r);
	}
	X.fmtban = 0;
}

/* ---- the driver ---- */

/*
 * The sRGB EOTF of a byte (#91): an f32 subterm of one integer parameter
 * in 0..255 that is, on every one of its 256 values, the specification's
 * curve of k/255 (the table texenc's sRGB formats use: the curve rounded
 * once) within what the request allows is a texel fetch of an R8_SRGB
 * texture whose texel k holds the byte k: the unit decodes it, within
 * tex_within ulps of the curve.  The inverse on store is not proposed.
 */
/* the largest |a - b| over parameter j's 256 values 0..255 (a and b read
 * no other parameter); infinity where either is undefined */
static double
byte_diff(nid a, nid b, int j)
{
	struct plan p;
	nid roots[2] = {a, b};
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r), md = 0;
	uint32_t la, lb;
	int k;

	for (k = 0; k < 256; k++)
		in_i[(size_t)j * CH + k] = k;
	plan_build(&p, roots, 2, 0);
	la = plan_loc(&p, a);
	lb = plan_loc(&p, b);
	plan_run(&p, 256, in_i, in_r);
	for (k = 0; k < 256; k++) {
		if ((COLF(&p, la)[k] & F_UNDEF) || (COLF(&p, lb)[k] & F_UNDEF)) {
			md = INFINITY;
			break;
		}
		if (!(fabs(COLR(&p, la)[k] - COLR(&p, lb)[k]) <= md))
			md = fabs(COLR(&p, la)[k] - COLR(&p, lb)[k]);
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return md;
}

static int
seed_srgb(nid body, nid n)
{
	uint64_t sl = node_slots(n);
	int j = 0, k;
	int64_t lo, hi;
	nid e[256], tab, v, cand, nb;
	struct repl *r;
	double md;
	float c;

	if (N.ty[n] != TY_F32 || !sl || (sl & (sl - 1)) || (sl >> MAXP) ||
	    is_const(n))
		return 0;
	while (!(sl >> j & 1))
		j++;
	if (j >= X.f->np || !TY_ISINT(X.f->p[j].ty) || X.f->p[j].ty == TY_BOOL)
		return 0;
	v = mk_var((uint32_t)j, X.f->p[j].name, X.f->p[j].ty);
	if (!irange(v, &lo, &hi) || lo < 0 || hi > 255)
		return 0;
	for (k = 0; k < 256; k++)
		e[k] = kf(texenc_srgb8(k));
	tab = mk_table(e, 256);
	cand = mk_tex(O_FETCH_CLAMP, TEX_SRGB, tab, v, 0);
	md = byte_diff(n, cand, j);
	/* not the curve: nothing to say (a tolerance this wide is the
	 * request's, and the check below says so) */
	if (!(md <= (X.tol > 0 ? X.tol : 1.0 / 512)))
		return 0;
	if (!model_lists(X.m, C_TEX_SRGB)) {
		X.unavail[C_TEX_SRGB] = 1;
		return 0;
	}
	if ((r = new_repl(n)) == NULL)
		return 0;
	nb = swap_nodes(body, &n, &cand, 1);
	c = price(nb);
	r->to = cand;
	r->cost = c;
	r->ok = 1;
	r->filtered = 1;
	r->curve = 1;
	/* against the formula: what it differs by, the table's rounding and
	 * the unit's decode (tex_within ulps of the largest value, 1) */
	r->err = md + ldexp(1.0, -24) + X.ulp * ldexp(1.0, -23);
	r->eorig = 0;
	sb_printf(&r->what, "%s is %s: the sRGB EOTF of %s/255", node_str(n),
	    node_str(cand), node_str(v));
	sb_printf(&r->holds, "holds where %s is in [%lld, %lld] (its range)",
	    node_str(v), (long long)lo, (long long)hi);
	sb_printf(&r->proof, "on all 256 values of %s (0..255; its range is "
	    "inside), %s is the sRGB EOTF of %s/255 (the specification's "
	    "curve, rounded once to f32) within %s; the unit's decode is within "
	    "%g ulp of the curve (the model's tex_within): %s in all, with the "
	    "table's own rounding", node_str(v), node_str(n), node_str(v),
	    fmt_g(md), X.ulp, fmt_g(r->err));
	sb_printf(&r->pipe, "a texture R8_SRGB, 256 x 1, one level: texel k "
	    "holds the byte k\nthe shader reads texelFetch(s, int(%s), 0).r, "
	    "which the unit decodes by the sRGB EOTF (the store's inverse is "
	    "not proposed)", node_str(v));
	return 1;
}

/* ---- the function itself as a texture ---- */

/* knots at every 2^j-th point, j up to FN_MAXJ; at most FN_MAXK of them,
 * the row every Vulkan device's image takes */
#define FN_MAXJ TEX_MAXBITS
#define FN_MAXK 4096

struct fcand {
	int ok;			/* derived; an integer result: exact */
	nid to;
	float cost;
	int j, quant;		/* the stride 2^j; knots on 256 levels */
	int j2;			/* a grid: the second parameter's stride */
	int64_t K, K2;		/* knots (a grid: along each parameter) */
	double D, e, err;	/* |F - line|, the read's error, both */
	double dmin, dmax, c;	/* F - line; an integer's added constant */
	nid coord, coord2;
	struct tinfo ti;
};

/*
 * The knots g (K of them) as a texture: with quant moved onto 256 evenly
 * spaced levels, which a byte holds; then the cheapest format the model
 * lists that holds them (table_info).  g comes back as what the texels
 * stand for, *gmax its largest magnitude.  0: no such table, or a class
 * the model does not list.
 */
static int
func_table(struct fcand *o, double *g, int64_t K, int quant, int isint,
    double *gmax)
{
	ty_t t64 = TY(K_SINT, 64);
	nid *e, tabk;
	int64_t k;

	*gmax = 0;
	if (quant) {
		double glo = g[0], ghi = g[0], step;
		for (k = 1; k < K; k++) {
			if (g[k] < glo)
				glo = g[k];
			if (g[k] > ghi)
				ghi = g[k];
		}
		step = (ghi - glo) / 255;
		if (!(step > 0))
			return 0;
		for (k = 0; k < K; k++)
			g[k] = glo + step * nearbyint((g[k] - glo) / step);
	}
	e = xmalloc((size_t)K * sizeof *e);
	for (k = 0; k < K; k++) {
		e[k] = isint ? mk_const_i(t64, (int64_t)g[k]) : kf(g[k]);
		if (!isint)
			g[k] = cval(e[k]);
		if (fabs(g[k]) > *gmax)
			*gmax = fabs(g[k]);
	}
	tabk = mk_table(e, (uint32_t)K);
	free(e);
	if (!table_info(tabk, &o->ti) || (quant && !o->ti.levels))
		return 0;
	if (!model_lists(X.m, o->ti.cls)) {
		X.unavail[o->ti.cls] = 1;
		return 0;
	}
	return 1;
}

/* a texel coordinate (x - lo)/2^j + 1/2 of parameter s: exact in f32 */
static nid
func_coord(int s, int64_t lo, int j)
{
	nid x = mk_cast(TY_F32, mk_var((uint32_t)s, X.f->p[s].name,
	    X.f->p[s].ty));

	if (j)
		x = mk_bin(O_MUL, x, kf(ldexp(1.0, -j)));
	return mk_bin(O_ADD, x, kf(0.5 - ldexp((double)lo, -j)));
}

/*
 * The read `tap` of the knots' texture against the function, whose value
 * less the knots' line or patch is in [o->dmin, o->dmax] over the points:
 * the read's own error, then for an integer result the constant c that
 * makes floor(read + c) the function (none: ok stays 0), for a float one
 * the bound.
 */
static void
func_finish(struct fcand *o, nid tap, double gmax, double vmax, double vmin)
{
	ty_t t = N.ty[X.f->body];
	double u = ldexp(1.0, -24);

	o->D = (o->dmax > -o->dmin ? o->dmax : -o->dmin) + 4 * DBL_EPSILON *
	    (gmax + vmax);
	if (o->ti.scale != 1)
		tap = mk_bin(O_MUL, kf(o->ti.scale), tap);
	if (o->ti.off != 0)
		tap = mk_bin(O_ADD, tap, kf(o->ti.off));
	/* the filter, the texels' rounding, the map back to the entries */
	o->e = X.ulp * 2 * u * o->ti.rmax + o->ti.trep + ((o->ti.scale != 1 ?
	    2 : 0) + (o->ti.off != 0 ? 2 : 0)) * u * (fabs(o->ti.off) +
	    o->ti.rmax);
	if (TY_ISINT(t)) {
		double lob, hib, c;
		/* floor(read + c) is F where F <= line + c -+ e < F + 1 */
		o->e += u * ((gmax > vmax ? gmax : vmax) + 2);
		lob = o->dmax + o->e;
		hib = o->dmin + 1 - o->e;
		o->err = o->dmax - o->dmin + 2 * o->e;
		if (!(o->err < 1 - 1e-6))
			return;
		c = nearbyint((lob + hib) / 2 * 4096) / 4096;
		if (!(c > lob && c < hib))
			c = (double)(float)((lob + hib) / 2);
		if (!(c > lob && c < hib))
			return;
		o->c = c;
		if (c != 0)
			tap = mk_bin(O_ADD, tap, kf(c));
		/* never negative: the conversion's truncation is the floor */
		o->to = mk_cast(t, vmin >= 0 ? tap : mk_un(O_FLOOR, tap));
	} else {
		o->err = o->D + o->e;
		o->to = N.ty[tap] == t ? tap : mk_cast(t, tap);
	}
	o->ok = 1;
	o->cost = price(o->to);
}

/*
 * The linear read over the knots at every 2^j-th point of the one
 * parameter (v: the function at its n points, in order).  The knots are
 * the function's values there; a last knot past the range is on the line
 * through the knot before it and the last point.
 */
static int
func_linear(const struct tabfn *tf, const double *v, int j, int quant,
    struct fcand *o)
{
	int isint = TY_ISINT(N.ty[X.f->body]), s = tf->rd[0];
	int64_t n = (int64_t)tf->total, st = (int64_t)1 << j, K, k, i, rem;
	double *g, vmax = 0, vmin = INFINITY, gmax;

	memset(o, 0, sizeof *o);
	o->j = j;
	o->quant = quant;
	K = (n - 1) / st + 1;
	rem = (n - 1) % st;
	if (rem)
		K++;
	if (K < 2 || K > FN_MAXK)
		return 0;
	g = xmalloc((size_t)K * sizeof *g);
	for (k = 0; k < K; k++)
		if (k * st <= n - 1)
			g[k] = v[k * st];
		else {
			g[k] = g[k - 1] + (v[n - 1] - g[k - 1]) * (double)st /
			    (double)rem;
			g[k] = isint ? nearbyint(g[k]) : (double)(float)g[k];
		}
	if (!func_table(o, g, K, quant, isint, &gmax)) {
		free(g);
		return 0;
	}
	o->K = K;
	/* the line between the knots against the function, at every point */
	o->dmin = INFINITY;
	o->dmax = -INFINITY;
	for (i = 0; i < n; i++) {
		int64_t r = i & (st - 1);
		double L, d;
		k = i >> j;
		L = g[k];
		if (r)
			L += (g[k + 1] - g[k]) * (double)r / (double)st;
		d = v[i] - L;
		if (d < o->dmin)
			o->dmin = d;
		if (d > o->dmax)
			o->dmax = d;
		if (fabs(v[i]) > vmax)
			vmax = fabs(v[i]);
		if (v[i] < vmin)
			vmin = v[i];
	}
	free(g);
	o->coord = func_coord(s, tf->blo[s], j);
	func_finish(o, mk_tex(O_TEX_LINEAR, TEX_KI(X.bits, 0), o->ti.tex,
	    o->coord, 0), gmax, vmax, vmin);
	return 1;
}

/* a row's knot a at stride st: its value there, or past the row's n
 * points the line through the knot before and the row's last point */
static double
knot_along(const double *row, int64_t n, int64_t st, int64_t a)
{
	double prev;

	if (a * st <= n - 1)
		return row[a * st];
	prev = row[(a - 1) * st];
	return prev + (row[n - 1] - prev) * (double)st / (double)((n - 1) % st);
}

/*
 * The bilinear read over a grid of knots, two parameters: the function at
 * every 2^jx-th value of the first and every 2^jy-th of the second (v:
 * its n0 x n1 values, the first parameter along a row).  A knot past the
 * range is extended along its row first, then along its column, as
 * func_linear's last knot is.  Both fractions are on the unit's grid and
 * both coordinates exact, so the read is the bilinear patch of four knots,
 * compared with the function at every point.
 */
static int
func_grid(const struct tabfn *tf, const double *v, int jx, int jy, int quant,
    struct fcand *o)
{
	int isint = TY_ISINT(N.ty[X.f->body]), s0 = tf->rd[0], s1 = tf->rd[1];
	int64_t n0 = (int64_t)tf->size[s0], n1 = (int64_t)tf->size[s1];
	int64_t sx = (int64_t)1 << jx, sy = (int64_t)1 << jy, Kx, Ky, a, b, ix,
	    iy;
	double *g, *col, vmax = 0, vmin = INFINITY, gmax;

	memset(o, 0, sizeof *o);
	o->j = jx;
	o->j2 = jy;
	o->quant = quant;
	Kx = (n0 - 1) / sx + 1 + ((n0 - 1) % sx != 0);
	Ky = (n1 - 1) / sy + 1 + ((n1 - 1) % sy != 0);
	if (Kx < 2 || Ky < 2 || Kx * Ky > FN_MAXK)
		return 0;
	g = xmalloc((size_t)(Kx * Ky) * sizeof *g);
	col = xmalloc((size_t)n1 * sizeof *col);
	for (a = 0; a < Kx; a++) {
		/* column a of the knots: every row's knot there, then along it */
		for (iy = 0; iy < n1; iy++)
			col[iy] = knot_along(v + iy * n0, n0, sx, a);
		for (b = 0; b < Ky; b++) {
			double k = knot_along(col, n1, sy, b);
			if (a * sx > n0 - 1 || b * sy > n1 - 1)
				k = isint ? nearbyint(k) : (double)(float)k;
			g[b * Kx + a] = k;
		}
	}
	free(col);
	if (!func_table(o, g, Kx * Ky, quant, isint, &gmax)) {
		free(g);
		return 0;
	}
	o->K = Kx;
	o->K2 = Ky;
	/* the patch of four knots against the function, at every point */
	o->dmin = INFINITY;
	o->dmax = -INFINITY;
	for (iy = 0; iy < n1; iy++)
		for (ix = 0; ix < n0; ix++) {
			int64_t rx = ix & (sx - 1), ry = iy & (sy - 1);
			double fu = (double)rx / (double)sx, fv = (double)ry /
			    (double)sy, L, L1, d, f = v[iy * n0 + ix];
			const double *q = g + (iy >> jy) * Kx + (ix >> jx);
			L = q[0];
			if (rx)
				L += (q[1] - q[0]) * fu;
			if (ry) {
				L1 = q[Kx];
				if (rx)
					L1 += (q[Kx + 1] - q[Kx]) * fu;
				L += (L1 - L) * fv;
			}
			d = f - L;
			if (d < o->dmin)
				o->dmin = d;
			if (d > o->dmax)
				o->dmax = d;
			if (fabs(f) > vmax)
				vmax = fabs(f);
			if (f < vmin)
				vmin = f;
		}
	free(g);
	/* the patch in doubles, not exact as the line is: its rounding is
	 * left out of what the function less the patch may be */
	o->dmin -= 8 * DBL_EPSILON * (gmax + vmax);
	o->dmax += 8 * DBL_EPSILON * (gmax + vmax);
	o->coord = func_coord(s0, tf->blo[s0], jx);
	o->coord2 = func_coord(s1, tf->blo[s1], jy);
	func_finish(o, mk_tex(O_TEX_BILINEAR, TEX_KI(X.bits, Kx), o->ti.tex,
	    o->coord, o->coord2), gmax, vmax, vmin);
	return 1;
}

/* "(x - lo)" as the report writes a parameter from its range's start */
static void
from_lo(char *buf, size_t n, const char *xn, int64_t lo)
{
	if (lo)
		snprintf(buf, n, "(%s %c %lld)", xn, lo < 0 ? '+' : '-',
		    (long long)(lo < 0 ? -lo : lo));
	else
		snprintf(buf, n, "%s", xn);
}

/* "every 8th value of x from 0", "every value of x" */
static void
every_say(struct sbuf *b, int j, const char *xn, int64_t lo)
{
	int64_t st = (int64_t)1 << j;

	if (j)
		sb_printf(b, "%lld%s value of %s from %lld", (long long)st, st ==
		    2 ? "nd" : "th", xn, (long long)lo);
	else
		sb_printf(b, "value of %s", xn);
}

/* seed_func's report of a read of knots: what, where, why, the pipeline */
static void
knots_say(struct repl *r, const struct tabfn *tf, const struct fcand *b,
    const char *fn)
{
	int isint = TY_ISINT(N.ty[X.f->body]), grid = tf->nrd == 2;
	int s = tf->rd[0], s1 = grid ? tf->rd[1] : s;
	const char *xn = sym_str(X.f->p[s].name), *yn = sym_str(X.f->p[s1].name);
	const char *ln = grid ? "patch" : "line";
	int64_t st = (int64_t)1 << b->j, st2 = (int64_t)1 << b->j2;
	int64_t n = (int64_t)tf->total, lo = tf->blo[s], lo2 = tf->blo[s1];
	int64_t n0 = (int64_t)tf->size[s], n1 = (int64_t)tf->size[s1];
	char xo[96], yo[96];

	from_lo(xo, sizeof xo, xn, lo);
	from_lo(yo, sizeof yo, yn, lo2);
	if (grid) {
		sb_printf(&r->what, "%s as a %lld x %lld grid of knots read with "
		    "the linear filter: the function at every ", fn,
		    (long long)b->K, (long long)b->K2);
		every_say(&r->what, b->j, xn, lo);
		sb_puts(&r->what, " and every ");
		every_say(&r->what, b->j2, yn, lo2);
	} else {
		sb_printf(&r->what, "%s as %lld knots read with the linear "
		    "filter: the function at every ", fn, (long long)b->K);
		every_say(&r->what, b->j, xn, lo);
	}
	if (b->quant)
		sb_puts(&r->what, ", moved onto 256 evenly spaced levels");
	sb_printf(&r->holds, "holds where %s is in [%lld, %lld]", xn,
	    (long long)lo, (long long)tf->bhi[s]);
	if (grid)
		sb_printf(&r->holds, " and %s in [%lld, %lld] (their ranges)", yn,
		    (long long)lo2, (long long)tf->bhi[s1]);
	else
		sb_puts(&r->holds, " (its range)");
	if (grid) {
		sb_printf(&r->proof, "a linear read at fractions (u, v) of texels "
		    "a b / c d is (1-u)(1-v) a + u(1-v) b + (1-u)v c + uv d; at "
		    "texel coordinates (%s/%lld + 1/2, %s/%lld + 1/2) they are "
		    "the knots (k, l), (k + 1, l), (k, l + 1), (k + 1, l + 1) "
		    "with k = %s >> %d, l = %s >> %d, u = (%s & %lld)/%lld and v "
		    "= (%s & %lld)/%lld", xo, (long long)st, yo, (long long)st2,
		    xo, b->j, yo, b->j2, xo, (long long)(st - 1), (long long)st,
		    yo, (long long)(st2 - 1), (long long)st2);
		if ((n0 - 1) % st || (n1 - 1) % st2)
			sb_printf(&r->proof, " (a knot past a range is on the line "
			    "through the knot before it and the range's last "
			    "point, along its row and then along its column%s)",
			    isint ? ", rounded to a whole number" : "");
		sb_printf(&r->proof, "\n  the fractions are multiples of 2^-%d "
		    "and 2^-%d, on the unit's 2^-%d grid, and the coordinates are "
		    "exact in f32: the weights do not move\n  that patch against "
		    "the function at every one of the %lld points: the function "
		    "less the patch is in [%s, %s]", b->j, b->j2, X.bits,
		    (long long)n, fmt_g(b->dmin), fmt_g(b->dmax));
	} else if (b->j) {
		sb_printf(&r->proof, "a linear read at fraction u of texels "
		    "a, b is (1-u) a + u b; at texel coordinate %s/%lld + "
		    "1/2 the texels are knots k and k + 1 with k = %s >> %d "
		    "and u = (%s & %lld)/%lld", xo, (long long)st, xo, b->j,
		    xo, (long long)(st - 1), (long long)st);
		if ((n - 1) % st)
			sb_printf(&r->proof, " (the last knot, past the range, "
			    "is on the line through the knot before it and "
			    "the last point%s)", isint ? ", rounded to a whole "
			    "number" : "");
		sb_printf(&r->proof, "\n  the fraction is a multiple of "
		    "2^-%d, on the unit's 2^-%d grid, and the coordinate "
		    "is exact in f32: the weights do not move\n  that line "
		    "against the function at every one of the %lld points: "
		    "the function less the line is in [%s, %s]", b->j,
		    X.bits, (long long)n, fmt_g(b->dmin), fmt_g(b->dmax));
	} else {
		sb_printf(&r->proof, "at texel coordinate %s + 1/2, a "
		    "texel's centre, a linear read is that texel (the next "
		    "one's weight is 0), and the coordinate is exact in "
		    "f32: the weight does not move\n  every point has its "
		    "own texel, the function's value there", xo);
		if (b->quant)
			sb_printf(&r->proof, " to within its level (the "
			    "function less its level is in [%s, %s])",
			    fmt_g(b->dmin), fmt_g(b->dmax));
	}
	sb_printf(&r->proof, "\n  the filter is within %g ulp of that sum "
	    "(the model's tex_within), the texels as %s, the map back "
	    "to the values%s: %s", X.ulp, b->ti.cls == C_TEX_SAMPLE_F16 ?
	    "f16" : b->ti.cls == C_TEX_SAMPLE_F32 ? "f32" : "k/q",
	    isint ? " and the add" : "", fmt_g(b->e));
	if (isint) {
		sb_printf(&r->proof, "\nthe function's value F is a whole "
		    "number, and floor(read + c) is F where F <= %s + c "
		    "- e and %s + c + e < F + 1: every c in (%s, %s), "
		    "%.9g taken", ln, ln, fmt_g(b->dmax + b->e), fmt_g(b->dmin +
		    1 - b->e), b->c);
		if (N.op[N.op[b->to] == O_CAST ? N.a[b->to] : b->to] !=
		    O_FLOOR)
			sb_puts(&r->proof, " (never negative: the conversion "
			    "to an integer truncates, which is that floor)");
	} else
		sb_printf(&r->proof, "\nagainst the function's f32 value "
		    "the read is within %s", fmt_g(b->err));
	sb_printf(&r->pipe, "a texture %s, %lld x %lld, one level: the knots",
	    fmt_vk(b->ti.cls), (long long)b->K, (long long)(grid ? b->K2 : 1));
	fmt_says(&r->pipe, &b->ti);
	sb_puts(&r->pipe, "\na sampler: LINEAR mag and min filters, "
	    "NEAREST mipmap mode, CLAMP_TO_EDGE, unnormalizedCoordinates, "
	    "lod 0\none combined image sampler more in the layout; the "
	    "shader reads ");
	if (grid) {
		sb_printf(&r->pipe, "textureLod(s, vec2(cx, cy), 0.0).r at the "
		    "texel coordinates cx = %s, ", node_str(b->coord));
		sb_printf(&r->pipe, "cy = %s", node_str(b->coord2));
	} else
		sb_printf(&r->pipe, "textureLod(s, c, 0.0).r at the texel "
		    "coordinate c = %s", node_str(b->coord));
	sb_puts(&r->pipe, "\nsetup (the descriptor, the upload) is not in "
	    "the model: not measured");
}

/*
 * seed_func's linear read: every stride (of each parameter, when there
 * are two) and both kinds of knots, the cheapest that holds with the
 * fewest knots; for a float result that
 * none holds within the tolerance, the nearest miss that would pay, to
 * say what tolerance takes it (ok 0, its to set).
 */
static void
func_knots(const struct tabfn *tf, float before, struct fcand *lin)
{
	int isint = TY_ISINT(N.ty[X.f->body]), un[C_NCLASS], j, j2, jm, q;
	int grid = tf->nrd == 2;
	int64_t n = (int64_t)tf->total;
	struct tinfo ti;
	struct fcand c;
	double *v;
	uint64_t k;

	if (!X.bits) {
		X.nobits = 1;
		return;
	}
	if (X.ulp == 0) {
		/* which class it would be, to say what is missing */
		if (table_info(tf->tab, &ti) && !model_lists(X.m, ti.cls))
			X.unavail[ti.cls] = 1;
		else
			X.noulp = 1;
		return;
	}
	memcpy(un, X.unavail, sizeof un);
	v = xmalloc((size_t)n * sizeof *v);
	for (k = 0; k < tf->total; k++)
		v[k] = cval(N.lst[N.a[tf->tab] + k]);
	jm = X.bits < FN_MAXJ ? X.bits : FN_MAXJ;
	for (q = 0; q < (isint ? 1 : 2); q++)
		for (j = jm; j >= 0; j--)
			for (j2 = grid ? jm : 0; j2 >= 0; j2--) {
				if (grid) {
					if ((j && ((int64_t)1 << j) > (int64_t)tf->size[
					    tf->rd[0]] - 1) || (j2 && ((int64_t)1 << j2) >
					    (int64_t)tf->size[tf->rd[1]] - 1))
						continue;
					if (!func_grid(tf, v, j, j2, q, &c))
						continue;
				} else if ((((int64_t)1 << j) > n - 1 && j) ||
				    !func_linear(tf, v, j, q, &c))
					continue;
				if (!c.to || !(c.cost < before))
					continue;
				if (isint ? !c.ok : c.err > X.tol) {
					if (!isint && !lin->ok && (!lin->to || c.err <
					    lin->err)) {
						*lin = c;
						lin->ok = 0;
					}
					continue;
				}
				/* the cheapest; at one price the fewest knots */
				if (!lin->ok || c.cost < lin->cost || (grid &&
				    c.cost == lin->cost && c.K * c.K2 < lin->K *
				    lin->K2))
					*lin = c;
			}
	free(v);
	if (lin->ok)	/* a class another stride or format missed */
		memcpy(X.unavail, un, sizeof un);
}

/*
 * A function that reads no table, of integer parameters with at most 2^16
 * points between them, is its table (tab.c's: the function at every
 * point, at an index packed from the parameters), and the table can be a
 * texture:
 *
 * - read by a nearest fetch at the packed index: the function exactly;
 * - for one parameter x, read by the linear filter over knots, the
 *   function at every 2^j-th x, at texel coordinate (x - lo)/2^j + 1/2
 *   (for two parameters, by the bilinear filter over a grid of knots, a
 *   stride for each: the line below is then the patch of four knots).
 *   The fraction is a multiple of 2^-j, on the unit's grid (j at most its
 *   subtexel bits), and the coordinate is exact in f32, so the weights do
 *   not move: the read is the line between two knots, within the filter's
 *   own error, and that line is compared with the function at every point.
 *   An integer result is floor(read + c): the function wherever F - line
 *   varies by less than 1 less twice the read's error (c in the middle:
 *   1/2 for a line through the values, less for a line the function is
 *   the floor of).  A float result is within the largest difference plus
 *   the read's error, which tol has to allow; under tol the knots are
 *   also tried on 256 evenly spaced levels, the bytes of an R8 texture.
 *   The fewest knots that hold at the lowest price are taken, at most
 *   FN_MAXK of them; j = 0 is a texel for each point.
 *
 * It is offered where it costs less than the function as written and
 * than the same table read by a load.
 */
static void
seed_func(nid body)
{
	const struct func *f = X.f;
	ty_t t = N.ty[body];
	int isint = TYK(t) == K_UINT || TYK(t) == K_SINT, q, sampled, s;
	struct tabfn tf;
	struct fcand near, lin, fx, *b = NULL;
	struct repl *r;
	struct sbuf ix = {0};
	float before = price(body), floorp = INFINITY, lc = INFINITY;
	double md, lim;
	uint64_t pts;
	int64_t n;

	if (!(isint || t == TY_F32) || (isint && TYW(t) > 32) ||
	    (N.fl[body] & NF_SG) || N.op[body] == O_VAR || is_const(body) ||
	    has_read(body, 0))
		return;
	/* the least a read could cost, before the work */
	if (model_lists(X.m, C_TEX_NEAREST))
		floorp = X.m->c[C_TEX_NEAREST];
	if (model_lists(X.m, C_TEX_FETCH) && X.m->c[C_TEX_FETCH] < floorp)
		floorp = X.m->c[C_TEX_FETCH];
	if (model_lists(X.m, C_TEX_BUFFER) && X.m->c[C_TEX_BUFFER] < floorp)
		floorp = X.m->c[C_TEX_BUFFER];
	for (q = 0; q < 4; q++) {
		static const int fc[4] = {C_TEX_SAMPLE, C_TEX_SAMPLE_16,
		    C_TEX_SAMPLE_F16, C_TEX_SAMPLE_F32};
		if (model_lists(X.m, fc[q]) && X.m->c[fc[q]] < floorp)
			floorp = X.m->c[fc[q]];
	}
	if (floorp < INFINITY && !(floorp < before))
		return;
	if (!tab_domain(f, &tf) || tf.total < 2 || tf.nrd == 0)
		goto out;
	tab_fill(f, &tf);
	if (tf.nonfinite || tf.undef == tf.total)
		goto out;
	n = (int64_t)tf.total;
	s = tf.rd[0];
	memset(&near, 0, sizeof near);
	memset(&lin, 0, sizeof lin);
	/* the nearest read of the whole table */
	if (!model_lists(X.m, C_TEX_NEAREST))
		X.unavail[C_TEX_NEAREST] = 1;
	else if (!X.bits)
		X.nobits = 1;
	else {
		near.to = mk_tex(O_FETCH_CLAMP, 0, tf.tab, tf.idx, 0);
		if (N.ty[near.to] != t)
			near.to = mk_cast(t, near.to);
		near.cost = price(near.to);
		near.ok = 1;
	}
	/* the linear filter over knots: one parameter, or two on a grid */
	if (tf.nrd <= 2 && !tf.undef && tf.blo[s] > -4194304 && tf.bhi[s] <
	    4194304 && (tf.nrd == 1 || (tf.blo[tf.rd[1]] > -4194304 &&
	    tf.bhi[tf.rd[1]] < 4194304 && tf.size[s] > 1 &&
	    tf.size[tf.rd[1]] > 1)))
		func_knots(&tf, before, &lin);
	/* a texel fetch of it: no sampler, no coordinate */
	memset(&fx, 0, sizeof fx);
	fx.to = texel_best(body, body, tf.tab, tf.idx, t, &fx.cost);
	fx.ok = fx.to != 0;
	/* the cheapest that holds and pays: a formula that costs no more
	 * stays a formula, with nothing to say */
	if (near.ok && near.cost < before)
		b = &near;
	if (fx.ok && fx.cost < before && (!b || fx.cost < b->cost))
		b = &fx;
	if (lin.ok && lin.cost < before && (!b || lin.cost < b->cost))
		b = &lin;
	if (!b && lin.to && !lin.ok)
		b = &lin;	/* a float result past the tolerance */
	if (!b)
		goto out;
	/* and a load stays a load */
	if (cclass_available(X.m, C_LOAD))
		lc = price(mk_cast(t, mk_index(tf.tab, tf.idx)));
	if (!(b->cost < lc) || (r = new_repl(body)) == NULL)
		goto out;
	r->to = b->to;
	r->cost = b->cost;
	r->plain = 1;
	sb_puts(&ix, node_str(tf.idx));
	if (b == &fx) {
		r->exact = 1;
		sb_printf(&r->what, "%s as its table through a texel fetch: "
		    "%lld entries, one for each point of %s; entry k is the "
		    "function at the point with index k = %s%s", shared_str(body,
		    120), (long long)n, sb_str(&tf.dom), sb_str(&ix), tf.undef ?
		    " (0 where the function is undefined or the where clause "
		    "fails)" : "");
		sb_printf(&r->holds, "holds on the function's domain: there "
		    "the index is in 0..%lld, inside the table", (long long)(n -
		    1));
		sb_printf(&r->proof, "the table is the function at every point, "
		    "all %lld evaluated\na texel fetch takes the integer index as "
		    "it is: no coordinate is made, nothing rounds", (long long)n);
		texel_pipe(&r->pipe, b->to, tf.tab, "the function's values",
		    sb_str(&ix));
	} else if (b == &near) {
		r->exact = 1;
		sb_printf(&r->what, "%s as its table through a nearest read: "
		    "%lld entries, one for each point of %s; entry k is the "
		    "function at the point with index k = %s%s", shared_str(body,
		    120),
		    (long long)n, sb_str(&tf.dom), sb_str(&ix), tf.undef ?
		    " (0 where the function is undefined or the where clause "
		    "fails)" : "");
		sb_printf(&r->holds, "holds on the function's domain: there "
		    "the index is in 0..%lld, where every address mode is the "
		    "plain read", (long long)(n - 1));
		sb_printf(&r->proof, "the table is the function at every point, "
		    "all %lld evaluated\nthe coordinate (i + 1/2)/n is made in "
		    "f32: within (|i| + 1/2) 2^-23 <= %s of a texel's centre for i "
		    "in [0, %lld], and the unit snaps by less than 2^-%d: %s + %s "
		    "< 1/2, the texel is i's", (long long)n, fmt_g(((double)(n -
		    1) + 0.5) * ldexp(1.0, -23)), (long long)(n - 1), X.bits,
		    fmt_g(((double)(n - 1) + 0.5) * ldexp(1.0, -23)),
		    fmt_g(ldexp(1.0, -X.bits)));
		sb_puts(&r->pipe, "a texture ");
		fetch_format(&r->pipe, tf.tab);
		sb_printf(&r->pipe, ", %lld x 1, one level, its texels the "
		    "function's values%s\na sampler: NEAREST filters, "
		    "CLAMP_TO_EDGE, normalised coordinates, lod 0\none combined "
		    "image sampler more in the layout; the shader reads "
		    "textureLod(s, (float(i) + 0.5) * (1.0 / %lld.0), 0.0).r with "
		    "i = %s", (long long)n, n > 4096 ? " (wider than the 4096 "
		    "texels every Vulkan device takes in a row: it needs that "
		    "maxImageDimension2D)" : "", (long long)n, sb_str(&ix));
	} else {
		r->filtered = 1;
		r->exact = isint;
		r->curve = 1;
		r->approx = b->ti.approx;
		r->err = isint ? 0 : b->err;
		knots_say(r, &tf, b, shared_str(body, 120));
	}
	r->ok = 1;
	if (b == &lin && !isint && !lin.ok)
		goto out;	/* past the tolerance: tex_search says by how much */
	/* a second opinion: both evaluated on the domain */
	md = max_diff(body, b->to, &pts, &sampled);
	lim = r->exact ? 0 : r->err * (1 + 1e-9);
	if (!(md <= lim)) {
		r->ok = 0;
		sb_printf(&r->why, "evaluated on %llu points the two differ by "
		    "%s, more than the derived %s: not offered",
		    (unsigned long long)pts, fmt_g(md), fmt_g(lim));
	} else
		sb_printf(&r->proof, "\nchecked again by evaluation%s on %s%llu "
		    "points: %s%s", b != &lin ? "" : " with ucas's own filter "
		    "(exact weights' sums, one rounding)", sampled ? "a sample "
		    "of " : "all ", (unsigned long long)pts, r->exact ? "equal" :
		    "largest difference ", r->exact ? "" : fmt_g(md));
out:
	free(tf.dom.p);
	free(ix.p);
}

static void
visit(nid body, nid n, uint8_t *seen, uint32_t lim)
{
	uint32_t j;
	int op, nr0 = X.nr;

	if (n == 0 || n >= lim || seen[n])
		return;
	seen[n] = 1;
	op = N.op[n];
	switch (op) {
	case O_CONST: case O_VAR: case O_UNDEF:
		return;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			visit(body, N.lst[N.a[n] + j], seen, lim);
		return;
	}
	/* a float of one byte: the sRGB decode, replaced whole */
	if (X.bits && X.ulp != 0 && seed_srgb(body, n))
		return;
	/* a packed format's channel decoded by hand, replaced whole */
	if (N.ty[n] == TY_F32 && seed_packed(body, n))
		return;
	/* a cube map's face selected by hand, replaced whole */
	if (op == O_SEL && seed_cube(body, n))
		return;
	/* a min or max of neighbouring entries, replaced whole */
	if ((op == O_MIN || op == O_MAX) && seed_minmax(body, n))
		return;
	/* a layer of an array texture, replaced whole */
	if ((op == O_INDEX || op == O_MIN || op == O_MAX || op == O_CAST) &&
	    seed_layer(body, n))
		return;
	switch (op) {
	case O_ADD: case O_SUB: case O_MUL: case O_FMA: case O_NEG: case O_DIV:
	case O_SHL:
		if (has_read(n, 0))
			seed_sum(body, n);
		break;
	case O_SEL: case O_INDEX:
		seed_fetch(body, n);
		break;
	}
	if (X.nr > nr0)		/* replaced whole: nothing inside it is */
		return;
	visit(body, N.a[n], seen, lim);
	visit(body, N.b[n], seen, lim);
	visit(body, N.c[n], seen, lim);
}

void
texres_free(struct texres *r)
{
	free(r->what.p);
	free(r->holds.p);
	free(r->proof.p);
	free(r->pipeline.p);
	free(r->why.p);
	memset(r, 0, sizeof *r);
}

static void
add_lines(struct sbuf *b, const struct sbuf *x)
{
	if (!x->n)
		return;
	if (b->n)
		sb_putc(b, '\n');
	sb_puts(b, x->p);
}

int
tex_search(const struct func *f, double tol, struct texres *res)
{
	uint8_t *seen;
	uint32_t lim = N.n;
	int64_t blo[MAXP], bhi[MAXP];
	nid roots[2], from[MAXREPL], to[MAXREPL];
	int j, na = 0, k;

	memset(res, 0, sizeof *res);
	free(X.far.p);
	free(X.nosum.p);
	free(X.nolayer.p);
	free(X.nopacked.p);
	free(X.nocube.p);
	memset(&X, 0, sizeof X);
	X.f = f;
	X.m = model_active();
	X.tol = tol;
	X.bits = (int)X.m->par[MP_SUBTEXEL];
	X.ulp = X.m->par[MP_TEX_WITHIN];
	res->tol = tol;
	res->before = res->after = price(f->body);
	seen = xcalloc(lim, 1);
	if (!plain_body(f->body, seen)) {
		free(seen);
		return res->status = TEXR_NONE;
	}
	memset(seen, 0, lim);
	/* proved ranges: the parameters' (the where clause narrowing them),
	 * then intervals */
	X.g = *f;
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &blo[j], &bhi[j]);
	where_bounds(f, blo, bhi, NULL);
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty) && !(TYK(f->p[j].ty) == K_UINT &&
		    TYW(f->p[j].ty) == 64)) {
			X.g.p[j].ranged = 1;
			X.g.p[j].lo = blo[j];
			X.g.p[j].hi = bhi[j];
		}
	roots[0] = f->body;
	roots[1] = f->where;
	plan_build(&X.p, roots, 2, 0);
	X.iv = xcalloc(X.p.n ? X.p.n : 1, sizeof *X.iv);
	plan_ival(&X.p, &X.g, X.iv);
	visit(f->body, f->body, seen, lim);
	free(seen);
	if (!X.nr)	/* nothing in it is a read: the function itself */
		seed_func(f->body);
	for (j = 0; j < X.nr; j++) {
		struct repl *r = &X.r[j];
		double allow = tol > r->eorig ? tol : r->eorig;
		if (!r->ok) {
			if (r->why.n) {
				sb_printf(&res->why, "%s%s: %s", res->why.n ? "\n" : "",
				    r->what.p ? r->what.p : "", sb_str(&r->why));
				if (res->status == TEXR_NONE)
					res->status = TEXR_OVERTOL;
			}
			continue;
		}
		if (r->filtered && !r->exact && r->err > allow) {
			sb_printf(&res->why, "%sderived, but within %s of %s, "
			    "more than the %s allowed (%s): tol %s takes it\n%s",
			    res->why.n ? "\n" : "", fmt_g(r->err), r->curve ?
			    "the formula" : "the exact sum", fmt_g(allow), tol >
			    r->eorig ? "tol" : r->curve ? "no tolerance given" :
			    "the sum's own rounding as written", fmt_g(r->err),
			    r->what.p);
			res->status = TEXR_OVERTOL;
			if (r->err > res->err)
				res->err = r->err;
			continue;
		}
		if (!(r->cost < res->before)) {
			if (r->plain)	/* nothing to say: a read stays a read */
				continue;
			sb_printf(&res->why, "%sderived, but not cheaper under %s "
			    "(cost %g, as written %g):\n%s", res->why.n ? "\n" :
			    "", sym_str(X.m->name), (double)r->cost,
			    (double)res->before, r->what.p);
			if (res->status == TEXR_NONE)
				res->status = TEXR_NOTCHEAPER;
			continue;
		}
		from[na] = r->from;
		to[na++] = r->to;
		add_lines(&res->what, &r->what);
		add_lines(&res->holds, &r->holds);
		add_lines(&res->proof, &r->proof);
		add_lines(&res->pipeline, &r->pipe);
		if (r->filtered && !r->exact)
			res->err += r->err;
	}
	if (na) {
		res->body = swap_nodes(f->body, from, to, na);
		res->after = price(res->body);
		res->exact = res->err == 0;
		res->status = TEXR_PROVED;
	} else if (res->status == TEXR_NONE) {
		for (k = C_MEM_FIRST; k <= C_MEM_LAST; k++)
			if (X.unavail[k]) {
				sb_printf(&res->why, "%s%s lists no %s: %s is not "
				    "available", res->why.n ? "\n" : "",
				    sym_str(X.m->name), cclass_name[k],
				    class_says(k));
				res->status = TEXR_UNAVAILABLE;
			}
		if (X.unavail_cube) {
			sb_printf(&res->why, "%s%s lists neither cube nor "
			    "tex_cube: the cube map face's instruction and read "
			    "are not available", res->why.n ? "\n" : "",
			    sym_str(X.m->name));
			res->status = TEXR_UNAVAILABLE;
		}
		if (X.unavail_packed) {
			sb_printf(&res->why, "%s%s lists no tex_packed: a "
			    "texel fetch of a packed format is not available",
			    res->why.n ? "\n" : "", sym_str(X.m->name));
			res->status = TEXR_UNAVAILABLE;
		}
		if (X.unavail_3d) {
			sb_printf(&res->why, "%s%s lists no tex_3d: a filtered "
			    "read of a 3-D texture is not available",
			    res->why.n ? "\n" : "", sym_str(X.m->name));
			res->status = TEXR_UNAVAILABLE;
		}
		if (X.nobits) {
			sb_printf(&res->why, "%s%s does not say what its texture "
			    "unit resolves (subtexel): no read through it is "
			    "proved", res->why.n ? "\n" : "", sym_str(X.m->name));
			res->status = TEXR_UNAVAILABLE;
		}
		if (X.noulp) {
			sb_printf(&res->why, "%s%s does not say how far its filter "
			    "is from the exact weighted sum (tex_within): no "
			    "filtered read is proved", res->why.n ? "\n" : "",
			    sym_str(X.m->name));
			res->status = TEXR_UNAVAILABLE;
		}
	}
	if (!na && X.nosum.n) {
		sb_printf(&res->why, "%s%s", res->why.n ? "\n" : "",
		    sb_str(&X.nosum));
		if (res->status == TEXR_NONE)
			res->status = TEXR_OVERTOL;
	}
	if (!na && X.nocube.n) {
		sb_printf(&res->why, "%s%s", res->why.n ? "\n" : "",
		    sb_str(&X.nocube));
		if (res->status == TEXR_NONE)
			res->status = TEXR_OVERTOL;
	}
	if (!na && X.nopacked.n) {
		sb_printf(&res->why, "%s%s", res->why.n ? "\n" : "",
		    sb_str(&X.nopacked));
		if (res->status == TEXR_NONE)
			res->status = TEXR_OVERTOL;
	}
	if (!na && X.nolayer.n) {
		sb_printf(&res->why, "%s%s", res->why.n ? "\n" : "",
		    sb_str(&X.nolayer));
		if (res->status == TEXR_NONE)
			res->status = TEXR_OVERTOL;
	}
	if (!na && X.far.n) {
		sb_printf(&res->why, "%s%s", res->why.n ? "\n" : "",
		    sb_str(&X.far));
		if (res->status == TEXR_NONE)
			res->status = TEXR_OVERTOL;
	}
	while (X.nr)
		drop_repl(&X.r[X.nr - 1]);
	plan_free(&X.p);
	free(X.iv);
	return res->status;
}

static void
print_lines(const char *ind, const char *s)
{
	const char *e;

	while (s && *s) {
		e = strchr(s, '\n');
		if (!e)
			e = s + strlen(s);
		printf("%s%.*s\n", ind, (int)(e - s), s);
		s = *e ? e + 1 : e;
	}
}

/* the report, for the command (all = 1) and for find's section (1 when
 * that has something to say) */
static int
tex_report(const struct func *f, double tol, int all)
{
	struct texres r;
	const char *mn = sym_str(model_active()->name);
	const char *fn = f->anon ? node_str(f->body) : sym_str(f->name);
	int st = tex_search(f, tol, &r);

	if (!all && st != TEXR_PROVED && st != TEXR_NOTCHEAPER &&
	    st != TEXR_OVERTOL) {
		texres_free(&r);
		return 0;
	}
	if (all)
		rep_begin("texalu");
	else
		rep_obj("texture_unit");
	rep_str("model", mn);
	rep_num("cost_before", r.before);
	rep_str("status", st == TEXR_PROVED ? "proved" : st == TEXR_NOTCHEAPER ?
	    "not_cheaper" : st == TEXR_OVERTOL ? "over_tolerance" :
	    st == TEXR_UNAVAILABLE ? "unavailable" : "none");
	if (st == TEXR_PROVED) {
		printf("%s%s%sreads through the texture unit under %s (cost %g "
		    "as written):\n", all ? "texalu " : "  ", all ? fn : "",
		    all ? ": " : "", mn, (double)r.before);
		printf("  cost %-3g %s\n", (double)r.after, shared_str(r.body,
		    600));
		print_lines("      ", sb_str(&r.what));
		print_lines("      ", sb_str(&r.holds));
		print_lines("      ", sb_str(&r.proof));
		printf("      the pipeline needs (a proposal: nothing is "
		    "applied):\n");
		print_lines("        ", sb_str(&r.pipeline));
		rep_num("cost", r.after);
		rep_expr("expr", r.body);
		rep_bool("exact", r.exact);
		if (!r.exact)
			rep_num("error_bound", r.err);
		rep_lines("reads", sb_str(&r.what));
		rep_str("holds_where", sb_str(&r.holds));
		rep_lines("derivation", sb_str(&r.proof));
		rep_lines("pipeline", sb_str(&r.pipeline));
	} else if (all && st == TEXR_NONE)
		printf("texalu %s: no table read or weighted sum of neighbouring "
		    "entries the texture unit could compute\n", fn);
	else if (all && !r.why.n)
		printf("texalu %s: nothing under %s\n", fn, mn);
	if (r.why.n) {
		if (st != TEXR_PROVED)
			printf("%s%s%sthe texture unit under %s (cost %g as "
			    "written): nothing taken\n", all ? "texalu " : "  ",
			    all ? fn : "", all ? ": " : "", mn, (double)r.before);
		print_lines("  ", sb_str(&r.why));
		rep_lines("not_taken", sb_str(&r.why));
	}
	if (!all)
		rep_end();
	texres_free(&r);
	return 1;
}

int
tex_find(const struct func *f)
{
	return tex_report(f, 0, 0);
}

int
texalu_command(struct P *p)
{
	static const char *const stop[] = {"tol", "model", "under", NULL};
	struct model *m = model_active(), *was = m;
	struct func *f;
	char w[48];
	uint32_t at = cmdp_pos(p);
	double tol = 0;

	if (!cmdp_word(p, w, sizeof w) || strcmp(w, "texalu") != 0 ||
	    func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	f = cmdp_func(p, stop);
	while (!cmdp_eos(p)) {
		if (cmdp_kw(p, "tol")) {
			tol = cmdp_number(p, stop);
			if (!(tol >= 0) || !isfinite(tol))
				fail("texalu: tol E, the error allowed on a "
				    "filtered sum, is 0 or more, not %g", tol);
		} else if (cmdp_kw(p, "model") || cmdp_kw(p, "under")) {
			if (!cmdp_word(p, w, sizeof w) ||
			    (m = model_find(sym_cstr(w))) == NULL)
				fail("texalu: unknown model '%s' (see 'models')", w);
		} else {
			cmdp_fail_here(p);
			fail("texalu F [tol E] [model M]");
		}
	}
	notes_flush();
	model_set_active(m);
	tex_report(f, tol, 1);
	model_set_active(was);
	return 1;
}

/* ---- English ---- */

/*
 * "what can the texture unit compute of f", "use the sampler for f within
 * 0.01 under cal", "f through the texture unit", "can texture reads do
 * f": texalu F [tol E] [under M].  The sentence must name the unit and a
 * defined function; NULL otherwise.
 */
const char *
texalu_nlp_raw(const char *sent)
{
	static const char *const unit[] = {"texture unit", "the sampler",
		"a sampler", "texture reads", "texture read", "sampled reads",
		"sampled read", "bilinear taps", "bilinear tap", "bilinear reads",
		"texture filter", "texture filtering", NULL};
	static char out[200];
	char w[48], prev[48] = "", fn[48] = "", model[48] = "", tol[32] = "";
	const char *s;
	char *low;
	size_t n, k;
	int i, hit = 0;

	n = strlen(sent);
	if (n > 400)
		return NULL;
	low = xmalloc(n + 1);
	for (k = 0; k < n; k++)
		low[k] = (sent[k] >= 'A' && sent[k] <= 'Z') ? (char)(sent[k] + 32) :
		    sent[k];
	low[n] = 0;
	for (i = 0; unit[i]; i++)
		if (strstr(low, unit[i]))
			hit = 1;
	free(low);
	if (!hit)
		return NULL;
	for (s = sent; *s;) {
		while (*s && !(*s == '_' || *s == '.' || *s == '-' ||
		    (*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'z') ||
		    (*s >= 'A' && *s <= 'Z')))
			s++;
		for (k = 0; *s && (*s == '_' || *s == '.' || *s == '-' ||
		    (*s >= '0' && *s <= '9') || (*s >= 'a' && *s <= 'z') ||
		    (*s >= 'A' && *s <= 'Z')); s++)
			if (k + 1 < sizeof w)
				w[k++] = *s;
		while (k > 0 && w[k - 1] == '.')	/* the sentence's stop */
			k--;
		w[k] = 0;
		if (!k)
			continue;
		if ((!strcmp(prev, "within") || !strcmp(prev, "tol") ||
		    !strcmp(prev, "tolerance")) && (w[0] == '.' || (w[0] >= '0' &&
		    w[0] <= '9')))
			snprintf(tol, sizeof tol, "%.31s", w);
		else if ((!strcmp(prev, "under") || !strcmp(prev, "model") ||
		    !strcmp(prev, "on")) && model_find(sym_cstr(w)))
			snprintf(model, sizeof model, "%s", w);
		else if (!fn[0] && func_find(sym_cstr(w)))
			snprintf(fn, sizeof fn, "%s", w);
		snprintf(prev, sizeof prev, "%s", w);
	}
	if (!fn[0])
		return NULL;
	snprintf(out, sizeof out, "texalu %s%s%s%s%s", fn, tol[0] ? " tol " : "",
	    tol, model[0] ? " under " : "", model);
	return out;
}
