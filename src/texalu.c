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
	double p[8], x, y, hi, lo = 0, naive = 0;
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

double
tex_filter(int op, int64_t ki, int64_t n, double x, double y,
    double (*tv)(void *ctx, int64_t j), void *ctx)
{
	int bits = TEX_BITS(ki);
	int64_t W = op == O_TEX_BILINEAR ? TEX_WIDTH(ki) : n, H = n / W;
	int64_t i0, j0 = 0, c0, c1, r0, r1;
	double a = tex_weight(x, bits, &i0), b = 0, t[4], w[4];

	if (op == O_TEX_BILINEAR)
		b = tex_weight(y, bits, &j0);
	if (isnan(a) || isnan(b))
		return NAN;
	c0 = tex_wrap(O_FETCH_CLAMP, i0, W);
	c1 = tex_wrap(O_FETCH_CLAMP, i0 >= INT64_MAX - 1 ? i0 : i0 + 1, W);
	if (op == O_TEX_LINEAR) {
		t[0] = tv(ctx, c0);
		t[1] = tv(ctx, c1);
		w[0] = 1 - a;
		w[1] = a;
		return tex_blend(t, w, 2);
	}
	r0 = tex_wrap(O_FETCH_CLAMP, j0, H);
	r1 = tex_wrap(O_FETCH_CLAMP, j0 >= INT64_MAX - 1 ? j0 : j0 + 1, H);
	t[0] = tv(ctx, r0 * W + c0);
	t[1] = tv(ctx, r0 * W + c1);
	t[2] = tv(ctx, r1 * W + c0);
	t[3] = tv(ctx, r1 * W + c1);
	w[0] = (1 - a) * (1 - b);
	w[1] = a * (1 - b);
	w[2] = (1 - a) * b;
	w[3] = a * b;
	return tex_blend(t, w, 4);
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

	if (!IS_FETCH(op))
		return n;
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

int
tex_class(nid n)
{
	nid tab = N.a[n];
	uint32_t j;

	if (IS_FETCH(N.op[n]))
		return C_TEX_NEAREST;
	for (j = 0; j < N.b[tab]; j++) {
		nid e = N.lst[N.a[tab] + j];
		if (!is_const(e) || !unorm8(N.kr[e], NULL))
			return C_TEX_SAMPLE_F32;
	}
	return C_TEX_SAMPLE;
}

float
tex_cost(const struct model *m, nid n, int *cls)
{
	int k = tex_class(n);

	*cls = k;
	/* unlike a memory kind of memk.c, a read through the unit has no
	 * fallback: a model that does not list its class cannot do it */
	if (!model_lists(m, k))
		return INFINITY;
	/* a nearest read's coordinate is made from the index: (i + 1/2)
	 * times 1/n, a conversion and a multiply-add */
	if (k == C_TEX_NEAREST)
		return m->c[k] + m->c[C_CONV] + m->c[C_FMUL];
	return m->c[k];
}

int
tex_lookup(const char *s, size_t n, int *bits)
{
	static const int fetch[] = {O_FETCH_REPEAT, O_FETCH_MIRROR,
	    O_FETCH_CLAMP, O_FETCH_MCLAMP, O_FETCH_BORDER};
	size_t i, l;
	int op = 0;

	*bits = 0;
	for (i = 0; i < sizeof fetch / sizeof fetch[0]; i++) {
		l = strlen(opinfo[fetch[i]].name);
		if (l == n && !memcmp(s, opinfo[fetch[i]].name, n))
			return fetch[i];
	}
	if (n > 10 && !memcmp(s, "tex_linear", 10)) {
		op = O_TEX_LINEAR;
		i = 10;
	} else if (n > 12 && !memcmp(s, "tex_bilinear", 12)) {
		op = O_TEX_BILINEAR;
		i = 12;
	} else
		return 0;
	for (; i < n; i++) {
		if (s[i] < '0' || s[i] > '9' || *bits > 99)
			return 0;
		*bits = *bits * 10 + (s[i] - '0');
	}
	return op;
}

void
tex_print_name(struct sbuf *b, nid n)
{
	if (IS_FETCH(N.op[n]))
		sb_puts(b, opinfo[N.op[n]].name);
	else
		sb_printf(b, "%s%d", opinfo[N.op[n]].name, TEX_BITS(N.ki[n]));
}

/* ---- the search ---- */

#define MAXTERM 20	/* table entries in one sum */
#define MAXREST 12	/* its other terms */
#define MAXREPL 16	/* reads replaced in one function */
#define MAXATOM 6	/* variables of an index */

/*
 * A weight: c[0] + c[1] s + c[2] t + c[3] s t over at most two atoms (any
 * float subterms that are not sums or products): what the weights of a
 * mix, a bilinear blend or a constant stencil are.
 */
struct mform {
	double c[4];
	nid s, t;
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
	int dx, dy, grp, used;
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
	int levels;		/* a float table read as 8-bit levels */
	struct sbuf what, holds, proof, pipe, why;
};

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
	int nolevels;		/* float tables as f32 only (seed_sum's retry) */
	struct sbuf far;	/* an index identity out of f32's reach */
	struct sbuf nosum;	/* neighbours that are not one read, and why */
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
	struct mform r = {{0, 0, 0, 0}, 0, 0};

	r.c[0] = v;
	return r;
}

/* x over the atoms (s, t), which include its own; 0 if they do not */
static int
mf_over(struct mform *x, nid s, nid t)
{
	struct mform r = {{0, 0, 0, 0}, 0, 0};

	r.s = s;
	r.t = t;
	r.c[0] = x->c[0];
	if (x->c[1] != 0 || x->c[3] != 0) {
		if (x->s == s)
			r.c[1] = x->c[1];
		else if (x->s == t)
			r.c[2] = x->c[1];
		else
			return 0;
	}
	if (x->c[2] != 0 || x->c[3] != 0) {
		if (x->t == t)
			r.c[2] += x->c[2];
		else if (x->t == s)
			r.c[1] += x->c[2];
		else
			return 0;
	}
	r.c[3] = x->c[3];
	*x = r;
	return 1;
}

static void
mf_tidy(struct mform *x)
{
	if (x->c[1] == 0 && x->c[3] == 0)
		x->s = 0;
	if (x->c[2] == 0 && x->c[3] == 0)
		x->t = 0;
	if (!x->s && x->t) {
		x->s = x->t;
		x->c[1] = x->c[2];
		x->t = 0;
		x->c[2] = 0;
	}
}

/* both over the same two atoms; 0 if they have three between them */
static int
mf_join(struct mform *a, struct mform *b)
{
	nid at[4], u[2] = {0, 0};
	int i, j, n = 0;

	mf_tidy(a);
	mf_tidy(b);
	at[0] = a->s;
	at[1] = a->t;
	at[2] = b->s;
	at[3] = b->t;
	for (i = 0; i < 4; i++) {
		if (!at[i])
			continue;
		for (j = 0; j < n; j++)
			if (u[j] == at[i])
				break;
		if (j < n)
			continue;
		if (n == 2)
			return 0;
		u[n++] = at[i];
	}
	return mf_over(a, u[0], u[1]) && mf_over(b, u[0], u[1]);
}

static int
mf_add(struct mform *r, struct mform a, struct mform b, double sb)
{
	int i;

	if (!mf_join(&a, &b))
		return 0;
	for (i = 0; i < 4; i++)
		a.c[i] += sb * b.c[i];
	mf_tidy(&a);
	*r = a;
	return 1;
}

static int
mf_mul(struct mform *r, struct mform a, struct mform b)
{
	struct mform o;

	if (!mf_join(&a, &b))
		return 0;
	/* no square of an atom */
	if ((a.c[1] != 0 || a.c[3] != 0) && (b.c[1] != 0 || b.c[3] != 0))
		return 0;
	if ((a.c[2] != 0 || a.c[3] != 0) && (b.c[2] != 0 || b.c[3] != 0))
		return 0;
	o = a;
	o.c[0] = a.c[0] * b.c[0];
	o.c[1] = a.c[0] * b.c[1] + a.c[1] * b.c[0];
	o.c[2] = a.c[0] * b.c[2] + a.c[2] * b.c[0];
	o.c[3] = a.c[0] * b.c[3] + a.c[3] * b.c[0] + a.c[1] * b.c[2] +
	    a.c[2] * b.c[1];
	mf_tidy(&o);
	*r = o;
	return 1;
}

static int
mf_isconst(const struct mform *a)
{
	return a->c[1] == 0 && a->c[2] == 0 && a->c[3] == 0;
}

/* its range over its atoms' ranges (multilinear: at the corners) */
static int
mf_range(const struct mform *a, double *lo, double *hi)
{
	double sl = 0, sh = 0, tl = 0, th = 0, v;
	int i;

	if (a->s && !frange(a->s, &sl, &sh))
		return 0;
	if (a->t && !frange(a->t, &tl, &th))
		return 0;
	for (i = 0; i < 4; i++) {
		double s = i & 1 ? sh : sl, t = i & 2 ? th : tl;
		v = a->c[0] + a->c[1] * s + a->c[2] * t + a->c[3] * s * t;
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

	for (i = 3; i >= 1; i--) {
		if (a->c[i] == 0)
			continue;
		x = i == 1 ? a->s : i == 2 ? a->t : mk_bin(O_MUL, a->s, a->t);
		if (i == 3)
			++*ops;
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
	int isint;		/* integers 0..255: stored as k/255 */
	int levels;		/* floats lo + (hi - lo) k/255: stored as k/255 */
	int cls;
	double tmax, trep;	/* max |v|, max |v - its texel| */
	nid tex;		/* the f32 table the read takes */
	double scale, off;	/* an entry is off + scale times its texel */
	double rmax;		/* the largest scale times a texel */
};

static int
table_info(nid tab, struct tinfo *ti)
{
	uint32_t j, n = N.b[tab];
	nid *e;
	int u8 = 1, un = 1;

	memset(ti, 0, sizeof *ti);
	ti->tab = tab;
	ti->n = n;
	ti->scale = 1;
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v;
		if (!is_const(x))
			return 0;
		v = cval(x);
		if (!isfinite(v) || fabs(v) > 3e38)
			return 0;
		if (fabs(v) > ti->tmax)
			ti->tmax = fabs(v);
		if (!TY_ISINT(N.ty[x]) || v < 0 || v > 255)
			u8 = 0;
		if (!unorm8((double)(float)v, NULL))
			un = 0;
	}
	ti->isint = u8;
	ti->cls = u8 || un ? C_TEX_SAMPLE : C_TEX_SAMPLE_F32;
	if (u8)
		ti->scale = 255;
	ti->rmax = ti->tmax;
	if (!u8 && !un && !X.nolevels) {
		/*
		 * Evenly spaced levels: every entry lo + step m for a whole m
		 * of at most 255.  The texture holds bytes k (m spread over
		 * 0..255 when the largest m divides 255, so the filter works
		 * on the whole range; m itself otherwise), the read is mapped
		 * back by one multiply-add, and what the entries differ from
		 * that by (a few ulps at most, or it is not taken) is in trep.
		 */
		double lo = cval(N.lst[N.a[tab]]), hi = lo, step = 0, gap = INFINITY;
		double worst = 0, mmax, spread;
		uint32_t i2;
		for (j = 1; j < n; j++) {
			double v = cval(N.lst[N.a[tab] + j]);
			if (v < lo)
				lo = v;
			if (v > hi)
				hi = v;
		}
		for (j = 0; j < n; j++)		/* the smallest gap */
			for (i2 = 0; i2 < n; i2++) {
				double d = cval(N.lst[N.a[tab] + j]) - cval(N.lst[N.a[tab] +
				    i2]);
				if (d > 0 && d < gap)
					gap = d;
			}
		/* first the whole range as 255 steps, then the smallest gap
		 * as the step */
		for (i2 = 0; i2 < 2 && !ti->levels && hi > lo; i2++) {
			step = i2 == 0 ? (hi - lo) / 255 : gap;
			mmax = floor((hi - lo) / step + 0.5);
			if (mmax < 1 || mmax > 255)
				continue;
			worst = 0;
			spread = fmod(255, mmax) == 0 ? 255 / mmax : 1;
			for (j = 0; j < n; j++) {
				double v = cval(N.lst[N.a[tab] + j]), k = floor((v - lo) /
				    step + 0.5) * spread, d = fabs(v - (lo + step / spread *
				    255 * (double)(float)(k / 255)));
				if (d > worst)
					worst = d;
			}
			if (worst <= ldexp(ti->tmax, -21)) {
				ti->levels = 1;
				ti->cls = C_TEX_SAMPLE;
				ti->scale = step / spread * 255;
				ti->off = lo;
				ti->rmax = hi - lo;
			}
		}
	}
	e = xmalloc(n * sizeof *e);
	for (j = 0; j < n; j++) {
		double v = cval(N.lst[N.a[tab] + j]), t, d;
		if (ti->levels)
			t = (double)(float)(floor((v - ti->off) / ti->scale * 255 +
			    0.5) / 255);
		else
			t = (double)(float)(v / ti->scale);
		d = fabs(v - (ti->off + t * ti->scale));
		if (d > ti->trep)
			ti->trep = d;
		e[j] = kf(t);
	}
	ti->tex = mk_table(e, n);
	free(e);
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
		sb_puts(b, "R32_SFLOAT (each entry as its f32)");
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
				return;
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
	if (!best || (r = new_repl(sel ? sel : n)) == NULL)
		return;
	r->to = best;
	r->cost = bc;
	r->exact = 1;
	r->ok = 1;
	r->plain = besti == idx && !sel;
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
		    bestfar ? "(" : "", node_str(besti), bestfar ? ")" : "",
		    (long long)cnt);
		break;
	case O_FETCH_MIRROR:
		sb_printf(&r->holds, "%s = min(r, %lld - r) with r = %s%s%s mod "
		    "%lld", node_str(idx), (long long)(2 * cnt - 1), bestfar ?
		    "(" : "", node_str(besti), bestfar ? ")" : "",
		    (long long)(2 * cnt));
		break;
	case O_FETCH_CLAMP:
		if (besti == idx)
			sb_printf(&r->holds, "%s is in 0..%lld, where every address "
			    "mode is the plain read", node_str(idx),
			    (long long)(cnt - 1));
		else
			sb_printf(&r->holds, "%s = min(max(%s, 0), %lld)",
			    node_str(idx), node_str(besti), (long long)(cnt - 1));
		break;
	case O_FETCH_MCLAMP:
		sb_printf(&r->holds, "%s = min(max(%s, -1 - (%s)), %lld)",
		    node_str(idx), node_str(besti), node_str(besti),
		    (long long)(cnt - 1));
		break;
	default:
		sb_printf(&r->holds, "the condition is 0 <= %s < %lld, the other "
		    "arm the border's value", node_str(besti), (long long)cnt);
	}
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
	    "reads textureLod(s, (float(%s) + 0.5) * (1.0 / %lld.0), 0.0).r%s%s",
	    (long long)cnt, mode_vk(bestmode), bestmode == O_FETCH_BORDER ?
	    (const_is(kb, 0) ? " with a transparent black border" :
	    " with the border's value as a custom border colour "
	    "(VK_EXT_custom_border_color)") : bestmode == O_FETCH_MCLAMP ?
	    " (samplerMirrorClampToEdge)" : "", bestfar ? "j" : "i",
	    (long long)cnt, bestfar ? ", j = " : "", bestfar ? node_str(besti) :
	    "");
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
		if (s->nt == MAXTERM || !table_info(N.a[e], &ti) ||
		    ti.tmax > 16777216.0 || (s->isint && !ti.isint))
			break;
		t = &s->t[s->nt];
		memset(t, 0, sizeof *t);
		if (!aff_of(N.b[e], &t->ix, 0))
			break;
		t->k = k;
		t->tab = N.a[e];
		s->tidx[s->nt++] = N.b[e];
		return;
	}
	}
opaque:
	add_rest(s, k, e);
}

/* a block of up to 2 x 2 neighbouring texels read by one filtered tap */
struct block {
	int m[4];		/* its terms at (0,0) (1,0) (0,1) (1,1), -1 none */
	int ox, oy;		/* the offset of its (0,0) */
	double S;		/* the weights' sum */
	struct mform fu, fv;	/* the fractions along x and y */
	double resid;		/* weights not reproduced, times nothing yet */
};

/* why a block of neighbours is not one read (block_derive, block_inside) */
enum { BW_OK, BW_SUM, BW_SIGN, BW_RANK, BW_OUT };

static int
block_derive(const struct sum *s, struct block *b, int *why)
{
	struct mform w[4], tot = mf_const(0), su = mf_const(0), sv =
	    mf_const(0), p, wr[4], d, one = mf_const(1), t1, t2;
	double lo, hi, mag;
	int i;

	for (i = 0; i < 4; i++) {
		w[i] = b->m[i] >= 0 ? s->t[b->m[i]].k : mf_const(0);
		if (!mf_add(&tot, tot, w[i], 1))
			return 0;
	}
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

/* the terms of a group laid on the table's grid */
struct grid {
	int ix[MAXTERM], n;	/* the group's terms (indices into the sum) */
	int64_t W, H;
	nid xe, ye;		/* the column and row expressions, 0: none */
	int64_t xlo, xhi, ylo, yhi;
	struct tinfo ti;
	double dxmax, dymax;
};

/* the group's member (its place in g->ix) at an offset, -1: none */
static int
grid_at(const struct sum *s, const struct grid *g, int dx, int dy)
{
	int i;

	for (i = 0; i < g->n; i++)
		if (s->t[g->ix[i]].dx == dx && s->t[g->ix[i]].dy == dy)
			return i;
	return -1;
}

/* is the block's footprint inside the table for every index */
static int
block_inside(const struct grid *g, const struct block *b)
{
	int wx = b->m[1] >= 0 || b->m[3] >= 0, wy = b->m[2] >= 0 || b->m[3] >= 0;

	return g->xlo + b->ox >= 0 && g->xhi + b->ox + wx <= g->W - 1 &&
	    g->ylo + b->oy >= 0 && g->yhi + b->oy + wy <= g->H - 1;
}

/*
 * Lay the group on the grid: the width from the index's coefficients
 * (rows of W when every variable's coefficient is 1 or W and W divides
 * the table), the offsets as (dx, dy) in the window that keeps dx
 * together.
 */
static int
grid_make(struct sum *s, struct grid *g)
{
	const struct aff *a = &s->t[g->ix[0]].ix;
	int64_t W = 0, cmin, best = 0, bw = -1;
	int i, sh;

	if (!table_info(s->t[g->ix[0]].tab, &g->ti))
		return 0;
	for (i = 0; i < a->n; i++) {
		int64_t c = a->coef[i];
		if (c == 1)
			continue;
		if (c < 1 || g->ti.n % c || (W && W != c))
			return 0;
		W = c;
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
 * The cover: each member is read by itself or is in one block (2 x 2, a
 * pair along x, a pair along y) that derives; the cheapest under the
 * model's prices, by dynamic programming over the members left (the
 * lowest one decides what is tried), greedily past 16 members.
 */
struct cover {
	const struct sum *s;
	const struct grid *g;
	struct block b[MAXTERM][3];	/* by member: 2 x 2, x pair, y pair */
	uint8_t ok[MAXTERM][3];
	uint32_t mask[MAXTERM][3];
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
	int i, k, j, why = BW_OK;

	for (i = 0; i < g->n; i++) {
		int dx = s->t[g->ix[i]].dx, dy = s->t[g->ix[i]].dy;
		int r = grid_at(s, g, dx + 1, dy), d = grid_at(s, g, dx, dy + 1),
		    rd = grid_at(s, g, dx + 1, dy + 1);
		for (k = 0; k < 3; k++) {
			struct block *b = &c->b[i][k];
			int mem[4];
			mem[0] = i;
			mem[1] = k != 2 ? r : -1;
			mem[2] = k != 1 ? d : -1;
			mem[3] = k == 0 ? rd : -1;
			c->ok[i][k] = 0;
			if ((k == 0 && (r < 0 || d < 0 || rd < 0)) || (k == 1 &&
			    r < 0) || (k == 2 && d < 0))
				continue;
			c->mask[i][k] = 0;
			for (j = 0; j < 4; j++) {
				b->m[j] = mem[j] >= 0 ? g->ix[mem[j]] : -1;
				if (mem[j] >= 0)
					c->mask[i][k] |= 1u << mem[j];
			}
			b->ox = dx;
			b->oy = dy;
			c->ok[i][k] = block_derive(s, b, &why);
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
	int n = c->g->n, i, k, best = 3;
	float v, bv;

	if (done == (1u << n) - 1)
		return 0;
	if (c->memo[done] >= 0)
		return c->memo[done];
	for (i = 0; done >> i & 1; i++)
		;
	bv = c->pone + cover_dp(c, done | 1u << i);
	for (k = 0; k < 3; k++) {
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
			if (k < 3) {
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
		for (k = 0; k < 3; k++)
			if (c->ok[i][k] && !(c->mask[i][k] & done) && c->ptap <
			    c->pone * (k == 0 ? 4 : 2))
				break;
		if (k < 3) {
			out[nb++] = c->b[i][k];
			done |= c->mask[i][k];
		} else
			done |= 1u << i;
	}
	return nb;
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
	struct block blk[MAXTERM];
	struct repl *r = NULL;
	nid taps = 0, to, ity_root = 0;
	double enew = 0, mabs = 0, mtap = 0, u = ldexp(1.0, -24);
	int i, j, ng = 0, ntap = 0, nleft = 0, isint, nterm = 0, njoin = 0;
	int allpos = 1, levels = 0;
	struct sbuf what = {0}, holds = {0}, proof = {0}, pipe = {0};

	(void)ity_root;
	isint = s->isint = TY_ISINT(N.ty[root]);
	collect(s, root, mf_const(1), 0);
	if (s->bad || s->nt < 2)
		goto out;
	for (i = 0; i < s->nt; i++)
		s->t[i].grp = -1;
	for (i = 0; i < s->nt; i++) {
		if (s->t[i].grp >= 0)
			continue;
		s->t[i].grp = ng;
		for (j = i + 1; j < s->nt; j++)
			if (s->t[j].grp < 0 && same_table(s->t[j].tab, s->t[i].tab) &&
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
		int nb, k, dmin = 0, dmax = 0, emin = 0, emax = 0, ex, ey;
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
		if (!X.bits) {
			X.nobits = 1;
			continue;
		}
		if (X.ulp == 0) {
			X.noulp = 1;
			continue;
		}
		/* by row, then column: the lowest member is a block's corner */
		for (i = 1; i < g.n; i++)
			for (k = i; k > 0; k--) {
				const struct term *a = &s->t[g.ix[k - 1]], *b =
				    &s->t[g.ix[k]];
				int t;
				if (a->dy < b->dy || (a->dy == b->dy && a->dx < b->dx))
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
		}
		memset(&cv, 0, sizeof cv);
		cv.s = s;
		cv.g = &g;
		cv.ptap = X.m->c[g.ti.cls] + (g.H > 1 ? 2 : 1) * X.m->c[C_FADD] +
		    X.m->c[C_FMUL];
		cv.pone = X.m->c[C_LOAD] + (isint ? X.m->c[C_IADD] :
		    X.m->c[C_FMUL]);
		cover_blocks(&cv);
		nb = cover_pick(&cv, blk);
		if (nb == 0) {
			/* no tap is cheaper than the reads it replaces: the
			 * cover with the most taps, to say so with its price */
			cv.ptap = 0;
			nb = cover_pick(&cv, blk);
		}
		if (nb == 0) {
			/* neighbours, and no block of them is a read: say why */
			if (cv.why && !X.nosum.n) {
				sb_printf(&X.nosum, "entries of the %lld x %lld table "
				    "are neighbours in %s, but ", (long long)g.W,
				    (long long)g.H, shared_str(root, 120));
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
					sb_puts(&X.nosum, " (a filter clamps at the edge, "
					    "the table's index does not)");
				} else
					sb_puts(&X.nosum, cv.why == BW_SUM ? "their "
					    "weights do not sum to a constant other "
					    "than 0" : cv.why == BW_SIGN ? "their "
					    "weights are not all of one sign for every "
					    "input (a fraction leaves [0, 1], or its "
					    "variable has no range)" : "their weights "
					    "are not the product of a fraction along x "
					    "and one along y");
			}
			continue;
		}
		ex = coord_exp(g.xlo, g.xhi, dmin, dmax);
		ey = coord_exp(g.ylo, g.yhi, emin, emax);
		sx = sy = 0;
		for (k = 0; k < nb; k++) {
			struct block *b = &blk[k];
			double dwx, dwy = 0, e, S = b->S;
			int ops = 0, gx, gy = 1, cnt = 0, q, cx, cy = -1000;
			nid x, y = 0, tap;
			double lo, hi;

			x = coord_node(g.xe, b->ox, &b->fu, ex, &dwx, &ops, &gx,
			    &cx);
			if (g.H > 1) {
				y = coord_node(g.ye, b->oy, &b->fv, ey, &dwy, &ops,
				    &gy, &cy);
				tap = mk_tex(O_TEX_BILINEAR, TEX_KI(X.bits, g.W),
				    g.ti.tex, x, y);
			} else
				tap = mk_tex(O_TEX_LINEAR, TEX_KI(X.bits, 0),
				    g.ti.tex, x, 0);
			if (S * g.ti.scale != 1)
				tap = mk_bin(O_MUL, kf(S * g.ti.scale), tap);
			levels |= g.ti.levels;
			if (g.ti.off != 0)	/* the levels' map back, with the product */
				tap = mk_bin(O_ADD, tap, kf(S * g.ti.off));
			if (S < 0)
				allpos = 0;
			if (taps)
				njoin++;
			taps = taps ? mk_bin(O_ADD, taps, tap) : tap;
			for (q = 0; q < 4; q++)
				if (b->m[q] >= 0) {
					s->t[b->m[q]].used = 1;
					cnt++;
				}
			/* the snapped weights, the filter, the texels' rounding,
			 * the weights not reproduced, and the map back to the
			 * entries: the scale's constant and product, the offset's
			 * constant and add */
			e = fabs(S) * (dwx * g.dxmax + dwy * g.dymax + X.ulp *
			    2 * u * g.ti.rmax + g.ti.trep) + b->resid * g.ti.tmax +
			    ((S * g.ti.scale != 1 ? 2 : 0) + (g.ti.off != 0 ? 2 : 0)) *
			    u * fabs(S) * (fabs(g.ti.off) + g.ti.rmax);
			enew += e;
			mtap += fabs(S) * g.ti.tmax;
			ntap++;
			sb_printf(&what, "%s%d entr%s of the %lld x %lld table from "
			    "offset (%d, %d), weights' sum %.6g: one read at "
			    "fraction ", what.n ? "\n" : "", cnt, cnt == 1 ? "y" :
			    "ies", (long long)g.W, (long long)g.H, b->ox, b->oy,
			    S);
			mf_text(&what, &b->fu);
			if (g.H > 1) {
				sb_puts(&what, " along x, ");
				mf_text(&what, &b->fv);
				sb_puts(&what, " along y");
			}
			if (g.H > 1) {
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
			for (q = 0; q < 2; q++) {
				const struct mform *fr = q ? &b->fv : &b->fu;
				if (fr->c[1] == 0 || !frange(fr->s, &lo, &hi))
					continue;
				sb_printf(&holds, "%s%s in [%g, %g] (its range): the "
				    "fraction ", holds.n ? "; " : "holds where ",
				    node_str(fr->s), lo, hi);
				mf_text(&holds, fr);
				sb_puts(&holds, " is in [0, 1]");
			}
			sb_printf(&proof, " those are the entries' weights%s",
			    g.ti.levels ? "" : g.ti.scale != 1 ?
			    " (the texels are the entries / 255)" : "");
			if (g.ti.levels)
				sb_printf(&proof, " (an entry is %.9g + %.9g times its "
				    "texel k/255: the weights' sum times %.9g is "
				    "added back)", g.ti.off, g.ti.scale, g.ti.off);
			if (b->resid > 0)
				sb_printf(&proof, " to within %s", fmt_g(b->resid));
			if (gx && gy)
				sb_printf(&proof, "\n  the fractions are on the 2^-%d "
				    "grid and the coordinates exact in f32: the "
				    "weights do not move", X.bits);
			else
				sb_printf(&proof, "\n  a fraction not on the grid is "
				    "snapped by less than 2^-%d, and f32 moves a "
				    "coordinate that is not exact by at most 2^%d: "
				    "the read moves by at most %s (x) and %s (y) "
				    "times the largest step between neighbours, %s "
				    "and %s", X.bits, cx > cy ? cx : cy, fmt_g(dwx),
				    fmt_g(dwy), fmt_g(g.dxmax), fmt_g(g.dymax));
			sb_printf(&proof, "\n  the filter is within %g ulp of that "
			    "sum (the model's tex_within): %s; the texels as f32: "
			    "%s; in all %s", X.ulp, fmt_g(fabs(S) * X.ulp * 2 * u *
			    g.ti.rmax), fmt_g(fabs(S) * g.ti.trep), fmt_g(e));
		}
		sb_printf(&holds, "%s", holds.n ? "; " : "holds where ");
		if (g.xe)
			sb_printf(&holds, "%s in [%lld, %lld]", node_str(g.xe),
			    (long long)g.xlo, (long long)g.xhi);
		if (g.ye)
			sb_printf(&holds, "%s%s in [%lld, %lld]", g.xe ? ", " : "",
			    node_str(g.ye), (long long)g.ylo, (long long)g.yhi);
		sb_printf(&holds, "%severy texel read is inside the table",
		    g.xe || g.ye ? " (proved ranges): " : "");
		sb_printf(&pipe, "%sa texture %s, %lld x %lld, one level: the "
		    "table's entries%s\na sampler: LINEAR mag and min filters, "
		    "NEAREST mipmap mode, CLAMP_TO_EDGE, unnormalizedCoordinates, "
		    "lod 0\none combined image sampler more in the layout; the "
		    "shader reads textureLod(s, %s, 0.0).r", pipe.n ? "\n" : "",
		    g.ti.cls == C_TEX_SAMPLE ? "R8_UNORM" : "R32_SFLOAT",
		    (long long)g.W, (long long)g.H, g.ti.isint ? " as bytes "
		    "(the shader sees k/255)" : g.ti.levels ? "' levels k as "
		    "bytes (evenly spaced: the shader maps k/255 back)" :
		    g.ti.cls == C_TEX_SAMPLE ? " times 255 as bytes" : " as f32",
		    g.H > 1 ? "vec2(x, y)" : "x");
		if (g.ti.cls == C_TEX_SAMPLE && g.H > 1 && !g.ti.levels) {
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
	r->levels = levels;
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
 * A float table of evenly spaced levels is read as bytes (the cheaper
 * class) when its bound is within what is allowed; when the map back
 * costs more precision than that, as f32.
 */
static void
seed_sum(nid body, nid root)
{
	int n0 = X.nr;
	struct repl *r;

	seed_sum1(body, root);
	if (X.nr == n0)
		return;
	r = &X.r[X.nr - 1];
	if (!r->levels || r->exact || r->err <= (X.tol > r->eorig ? X.tol :
	    r->eorig))
		return;
	drop_repl(r);
	X.nolevels = 1;
	seed_sum1(body, root);
	X.nolevels = 0;
}

/* ---- the driver ---- */

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
			sb_printf(&res->why, "%sderived, but within %s of the exact "
			    "sum, more than the %s allowed (%s): tol %s takes "
			    "it\n%s", res->why.n ? "\n" : "", fmt_g(r->err),
			    fmt_g(allow), tol > r->eorig ? "tol" :
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
				    sym_str(X.m->name), cclass_name[k], k ==
				    C_TEX_NEAREST ? "a nearest read with an address "
				    "mode" : k == C_TEX_SAMPLE ? "a filtered read of "
				    "an 8-bit normalised texture" : "a filtered read "
				    "of a 32-bit float texture");
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
