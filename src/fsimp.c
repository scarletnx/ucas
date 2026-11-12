/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * fsimp: float simplify's mathematics (see fsimp.h).
 *
 * Proof methods, all derivations rather than sampling:
 *   - IEEE binary32 interval arithmetic in the total order
 *     -inf < -max < ... < -0 < +0 < ... < +inf: a correctly rounded
 *     operation is monotone in each operand (for the operation's sign
 *     pattern), so an interval's image is spanned by its corners;
 *   - monotonicity rules (x + K, x * K with K > 0, pow(x, y > 0) on
 *     x >= -0, exp2, log2, sqrt ...), so a preimage {x : f(x) in V} of a
 *     monotone f is an interval, found by bisection on the order (32
 *     steps: a solution, not a search);
 *   - the standard model fl(a op b) = (a op b)(1 + d), |d| <= u = 2^-24,
 *     exact cases (Sterbenz, scaling by 2^k, exact reciprocals), and
 *     |v| / ulp(v) < 2^24 for normal v;
 *   - the Vulkan precision table (Vulkan 1.3, "Precision and Operation of
 *     SPIR-V Instructions") composed through the rewrite: a rewrite is
 *     within the original's precision when its derived error bound is at
 *     most the error the original was already allowed, at every input;
 *   - for approximations: minimax fits (remez.c) whose error is bounded on
 *     4096 geometric pieces by a second-order Taylor bound with explicit
 *     derivative bounds, plus the Horner/fma rounding bound gamma_2n.
 */
#include "fsimp.h"
#include "approx.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define U24 0x1p-24
#define FDEN 0x1p-149f		/* smallest subnormal */

/* ---- the total order ---- */

static int64_t
tkey(float f)
{
	uint32_t u;

	memcpy(&u, &f, 4);
	return (u >> 31) ? -(int64_t)(u & 0x7fffffffu) - 1 : (int64_t)u;
}

static float
funkey(int64_t k)
{
	uint32_t u = k < 0 ? (uint32_t)(-(k + 1)) | 0x80000000u : (uint32_t)k;
	float f;

	memcpy(&f, &u, 4);
	return f;
}

#define KEY_NINF (-(int64_t)0x7f800000 - 1)
#define KEY_PINF ((int64_t)0x7f800000)

static int
tlt(float a, float b)
{
	return tkey(a) < tkey(b);
}

static float
tmin(float a, float b)
{
	return tlt(b, a) ? b : a;
}

static float
tmax(float a, float b)
{
	return tlt(a, b) ? b : a;
}

static float
tnext(float f)
{
	return tkey(f) >= KEY_PINF ? f : funkey(tkey(f) + 1);
}

static float
tprev(float f)
{
	return tkey(f) <= KEY_NINF ? f : funkey(tkey(f) - 1);
}

/* ulps between two floats (in the order; -0 and +0 one apart) */
static long long
ulps_apart(float a, float b)
{
	int64_t d = tkey(a) - tkey(b);

	return d < 0 ? -d : d;
}

const char *
fx_str(float v, char *b, size_t n)
{
	int p;

	if (isnan(v))
		snprintf(b, n, "NaN");
	else if (isinf(v))
		snprintf(b, n, "%sinf", v < 0 ? "-" : "+");
	else if (v == 0)
		snprintf(b, n, "%s0", signbit(v) ? "-" : "+");
	else {
		/* the shortest decimal that reads back as v */
		for (p = 6; p <= 9; p++) {
			snprintf(b, n, "%.*g", p, (double)v);
			if (strtof(b, NULL) == v)
				break;
		}
	}
	return b;
}

/* ---- the reference ---- */

static float
ev1(int op, float a, float b, float c, float d, int *undef)
{
	switch (op) {
	case FX_ADD: return (float)((double)a + (double)b);
	case FX_SUB: return (float)((double)a - (double)b);
	case FX_MUL: return (float)((double)a * (double)b);
	case FX_DIV: return (float)((double)a / (double)b);
	case FX_FMA: return fmaf(a, b, c);
	case FX_NEG: return -a;
	case FX_POW:
		if (a < 0 || (a == 0 && b <= 0))
			*undef = 1;
		return (float)powl(a, b);
	case FX_EXP2: return (float)exp2l(a);
	case FX_LOG2:
		if (a <= 0)
			*undef = 1;
		return (float)log2l(a);
	case FX_SQRT:
		if (a < 0)
			*undef = 1;
		return (float)sqrt((double)a);
	case FX_ISQRT:
		if (a <= 0)
			*undef = 1;
		return (float)(1.0L / sqrtl(a));
	case FX_STEP: return b < a ? 0.0f : 1.0f;
	case FX_MIX: {
		float s = (float)(1.0 - (double)c);
		float p = (float)((double)a * s), q = (float)((double)b * c);
		return (float)((double)p + (double)q);
	}
	case FX_SELGE: return !(a < b) ? c : d;
	case FX_MIN:
		if (isnan(a) || isnan(b))
			*undef = 1;
		return b < a ? b : a;
	case FX_MAX:
		if (isnan(a) || isnan(b))
			*undef = 1;
		return a < b ? b : a;
	case FX_ABS: return fabsf(a);
	}
	return a;
}

float
fx_eval(const struct fxg *g, int node, const float *x, int *undef)
{
	float v[FX_MAXN];
	char u[FX_MAXN];
	int i;

	for (i = 0; i <= node; i++) {
		const struct fxn *n = &g->n[i];
		int uu = 0;
		float a = n->a >= 0 ? v[n->a] : 0, b = n->b >= 0 ? v[n->b] : 0,
		    c = n->c >= 0 ? v[n->c] : 0, d = n->d >= 0 ? v[n->d] : 0;
		if (n->op == FX_IN || n->op == FX_CVT || n->op == FX_OPQ)
			v[i] = x[n->in];
		else if (n->op == FX_K)
			v[i] = n->k;
		else
			v[i] = ev1(n->op, a, b, c, d, &uu);
		if (n->op == FX_SELGE)	/* only the chosen side counts */
			uu |= u[n->a] | u[n->b] | (!(a < b) ? u[n->c] : u[n->d]);
		else
			uu |= (n->a >= 0 && u[n->a]) | (n->b >= 0 && u[n->b]) |
			    (n->c >= 0 && u[n->c]) | (n->d >= 0 && u[n->d]);
		u[i] = (char)uu;
	}
	if (undef)
		*undef = u[node];
	return v[node];
}

double
fx_price(int op)
{
	const float *c = model_active()->c;

	switch (op) {
	case FX_ADD: case FX_SUB: case FX_NEG: return c[C_FADD];
	case FX_MUL: case FX_FMA: return c[C_FMUL];
	case FX_DIV: return c[C_FDIV];
	case FX_POW: return c[C_FPOW];
	case FX_EXP2: case FX_LOG2: case FX_ISQRT: return c[C_FTRANS];
	case FX_SQRT: return c[C_FSQRT];
	/* as lowered: step = compare + select, mix = subtract + fma */
	case FX_STEP: case FX_SELGE: return c[C_CMP] + c[C_SELECT];
	case FX_MIX: return c[C_FADD] + c[C_FMUL];
	case FX_MIN: case FX_MAX: case FX_ABS: return c[C_MINMAX];
	case FX_CVT: return c[C_CONV];
	}
	return 0;
}

/* ---- intervals ---- */

static struct fiv
fi_none(void)
{
	struct fiv r;

	memset(&r, 0, sizeof r);
	r.empty = 1;
	return r;
}

static void
fi_put(struct fiv *r, float v)
{
	if (isnan(v)) {
		r->nan = 1;
		return;
	}
	if (r->empty) {
		r->lo = r->hi = v;
		r->empty = 0;
		return;
	}
	r->lo = tmin(r->lo, v);
	r->hi = tmax(r->hi, v);
}

static struct fiv
fi_pt(float v)
{
	struct fiv r = fi_none();

	fi_put(&r, v);
	return r;
}

static struct fiv
fi_hull(struct fiv a, struct fiv b)
{
	struct fiv r = a;

	if (!b.empty) {
		fi_put(&r, b.lo);
		fi_put(&r, b.hi);
	}
	r.nan |= b.nan;
	r.undef |= b.undef;
	return r;
}

/* A restricted to [lo, hi] in the order */
static struct fiv
fi_clip(struct fiv A, float lo, float hi)
{
	if (A.empty || tlt(A.hi, lo) || tlt(hi, A.lo)) {
		A.empty = 1;
		return A;
	}
	A.lo = tmax(A.lo, lo);
	A.hi = tmin(A.hi, hi);
	return A;
}

static int
fi_has(struct fiv A, float v)
{
	return !A.empty && !tlt(v, A.lo) && !tlt(A.hi, v);
}

/* corners of a box: the op is monotone in each operand there */
static struct fiv
fi_corners(int op, const struct fiv *A, int na)
{
	struct fiv r = fi_none();
	int m, i;

	for (m = 0; m < (1 << na); m++) {
		float v[4] = {0, 0, 0, 0};
		int u = 0;
		for (i = 0; i < na; i++)
			v[i] = (m >> i & 1) ? A[i].hi : A[i].lo;
		fi_put(&r, ev1(op, v[0], v[1], v[2], v[3], &u));
	}
	return r;
}

static struct fiv
fi_op(int op, struct fiv a, struct fiv b, struct fiv c, struct fiv d)
{
	struct fiv r = fi_none(), A[3];
	int nan = a.nan | b.nan, undef = a.undef | b.undef;

	if (op == FX_FMA || op == FX_MIX) {
		nan |= c.nan;
		undef |= c.undef;
	}
	switch (op) {
	case FX_ADD: case FX_SUB: case FX_MUL: case FX_MIN: case FX_MAX:
		if (a.empty || b.empty)
			break;
		A[0] = a;
		A[1] = b;
		r = fi_corners(op, A, 2);
		if (r.nan && (op == FX_ADD || op == FX_SUB)) {
			r.lo = -INFINITY;	/* inf - inf somewhere */
			r.hi = INFINITY;
			r.empty = 0;
		}
		if (op == FX_MIN || op == FX_MAX)
			undef |= nan;
		break;
	case FX_DIV:
		if (a.empty || b.empty)
			break;
		if (fi_has(b, -0.0f) || fi_has(b, 0.0f)) {
			r.lo = -INFINITY;
			r.hi = INFINITY;
			r.empty = 0;
			r.nan = fi_has(a, 0.0f) || fi_has(a, -0.0f) ||
			    isinf(a.lo) || isinf(a.hi);
		} else {
			A[0] = a;
			A[1] = b;
			r = fi_corners(op, A, 2);
		}
		break;
	case FX_FMA:
		if (a.empty || b.empty || c.empty)
			break;
		A[0] = a;
		A[1] = b;
		A[2] = c;
		r = fi_corners(op, A, 3);
		if (r.nan) {
			r.lo = -INFINITY;
			r.hi = INFINITY;
			r.empty = 0;
		}
		break;
	case FX_NEG: case FX_EXP2:
		if (a.empty)
			break;
		r = fi_corners(op, &a, 1);
		break;
	case FX_ABS:
		if (a.empty)
			break;
		if (tlt(a.lo, 0.0f) && !tlt(a.hi, 0.0f)) {
			r = fi_pt(0.0f);
			fi_put(&r, fabsf(a.lo));
			fi_put(&r, fabsf(a.hi));
		} else
			r = fi_corners(op, &a, 1);
		break;
	case FX_SQRT: case FX_LOG2: case FX_ISQRT: {
		/* defined: sqrt on x >= -0, log2 and inversesqrt on x > 0 */
		float lo = op == FX_SQRT ? -0.0f : FDEN;
		if (a.empty)
			break;
		if (tlt(a.lo, lo))
			undef = 1;
		a = fi_clip(a, lo, INFINITY);
		if (a.empty)
			break;
		r = fi_corners(op, &a, 1);
		break;
	}
	case FX_POW: {
		/* defined: x >= -0 when y > 0, x > 0 when y can be <= 0 */
		float lo;
		if (a.empty || b.empty)
			break;
		lo = b.lo > 0 ? -0.0f : FDEN;
		if (tlt(a.lo, lo))
			undef = 1;
		a = fi_clip(a, lo, INFINITY);
		if (a.empty)
			break;
		A[0] = a;
		A[1] = b;
		r = fi_corners(op, A, 2);
		break;
	}
	case FX_STEP:	/* (edge, x) */
		if (a.empty || b.empty) {
			if (b.nan || a.nan)
				r = fi_pt(1);
			break;
		}
		if (b.lo < a.hi)
			fi_put(&r, 0);
		if (b.hi >= a.lo || b.nan || a.nan)
			fi_put(&r, 1);
		nan = 0;
		break;
	case FX_MIX: {
		struct fiv s = fi_op(FX_SUB, fi_pt(1), c, fi_none(), fi_none());
		r = fi_op(FX_ADD, fi_op(FX_MUL, a, s, fi_none(), fi_none()),
		    fi_op(FX_MUL, b, c, fi_none(), fi_none()), fi_none(),
		    fi_none());
		break;
	}
	case FX_SELGE: {	/* !(a < b) ? c : d */
		int th = a.nan || b.nan || (!a.empty && !b.empty &&
		    a.hi >= b.lo), el = !a.empty && !b.empty && a.lo < b.hi;
		r = fi_none();
		nan = 0;
		undef = a.undef | b.undef;
		if (th)
			r = fi_hull(r, c);
		if (el)
			r = fi_hull(r, d);
		break;
	}
	}
	r.nan |= nan;
	r.undef |= undef;
	return r;
}

/* every node's range, the inputs in dom[] */
static void
ranges(const struct fxg *g, int upto, const struct fiv *dom, struct fiv *r)
{
	int i;

	for (i = 0; i <= upto; i++) {
		const struct fxn *n = &g->n[i];
		struct fiv z = fi_none();
		if (n->op == FX_IN || n->op == FX_CVT || n->op == FX_OPQ)
			r[i] = dom[n->in];
		else if (n->op == FX_K)
			r[i] = fi_pt(n->k);
		else
			r[i] = fi_op(n->op, n->a >= 0 ? r[n->a] : z,
			    n->b >= 0 ? r[n->b] : z, n->c >= 0 ? r[n->c] : z,
			    n->d >= 0 ? r[n->d] : z);
	}
}

struct fiv
fx_range(const struct fxg *g, int node, const struct fiv *dom)
{
	struct fiv r[FX_MAXN];

	ranges(g, node, dom, r);
	return r[node];
}

static void
default_dom(const struct fxg *g, struct fiv *dom)
{
	int i;

	for (i = 0; i < g->nin; i++) {
		dom[i] = fi_none();
		dom[i].empty = 0;
		dom[i].lo = g->in[i].lo;
		dom[i].hi = g->in[i].hi;
		dom[i].nan = g->in[i].nan;
	}
}

/* ---- dependence and monotonicity ---- */

static unsigned
deps(const struct fxg *g, int i)
{
	const struct fxn *n = &g->n[i];
	unsigned m = 0;

	if (n->op == FX_IN || n->op == FX_CVT || n->op == FX_OPQ)
		return 1u << n->in;
	if (n->a >= 0)
		m |= deps(g, n->a);
	if (n->b >= 0)
		m |= deps(g, n->b);
	if (n->c >= 0)
		m |= deps(g, n->c);
	if (n->d >= 0)
		m |= deps(g, n->d);
	return m;
}

static int
is_k(const struct fxg *g, int i, float *k)
{
	if (i < 0 || g->n[i].op != FX_K)
		return 0;
	if (k)
		*k = g->n[i].k;
	return 1;
}

/*
 * +1 non-decreasing, -1 non-increasing in input `in` on the defined part
 * (2: a constant), 0 unknown.  Rules: x + K, K + x, x - K; K - x
 * reverses; x * K, x / K with K > 0 keep (K < 0 reverses); fma(x, K, C);
 * pow(x, y > 0), exp2, log2, sqrt keep; inversesqrt, pow(x, y < 0) and
 * negation reverse.  Rounding is monotone, so the rules hold for the
 * float operations, not only the real ones.
 */
static int
mono(const struct fxg *g, int i, int in)
{
	const struct fxn *n = &g->n[i];
	int ma, mb;
	float k;

	switch (n->op) {
	case FX_K: return 2;
	case FX_IN: return n->in == in ? 1 : 0;
	case FX_ADD: case FX_SUB: case FX_MIN: case FX_MAX:
		ma = mono(g, n->a, in);
		mb = mono(g, n->b, in);
		if (n->op == FX_SUB && (mb == 1 || mb == -1))
			mb = -mb;
		if (ma == 2)
			return mb;
		if (mb == 2)
			return ma;
		return ma == mb ? ma : 0;
	case FX_MUL: case FX_DIV:
		if (is_k(g, n->b, &k) && k != 0) {
			ma = mono(g, n->a, in);
			return ma == 2 ? 2 : k > 0 ? ma : -ma;
		}
		if (n->op == FX_MUL && is_k(g, n->a, &k) && k != 0) {
			mb = mono(g, n->b, in);
			return mb == 2 ? 2 : k > 0 ? mb : -mb;
		}
		return 0;
	case FX_FMA:
		if (is_k(g, n->b, &k) && k != 0 && mono(g, n->c, in) == 2) {
			ma = mono(g, n->a, in);
			return ma == 2 ? 2 : k > 0 ? ma : -ma;
		}
		return 0;
	case FX_NEG:
		ma = mono(g, n->a, in);
		return ma == 2 ? 2 : -ma;
	case FX_EXP2: case FX_LOG2: case FX_SQRT:
		return mono(g, n->a, in);
	case FX_ISQRT:
		ma = mono(g, n->a, in);
		return ma == 2 ? 2 : -ma;
	case FX_POW:
		if (is_k(g, n->b, &k) && k != 0) {
			ma = mono(g, n->a, in);
			return ma == 2 ? 2 : k > 0 ? ma : -ma;
		}
		return 0;
	}
	return 0;
}

/* inputs for an evaluation: x for input `in`, a value of each other's
 * domain (1 when it is in the domain, else its low end) */
static void
base_inputs(const struct fxg *g, const struct fiv *dom, float *x)
{
	int i;

	for (i = 0; i < g->nin; i++)
		x[i] = fi_has(dom[i], 1.0f) ? 1.0f : dom[i].empty ? 0 :
		    isinf(dom[i].lo) ? dom[i].hi : dom[i].lo;
}

static float
eval_at(const struct fxg *g, int node, const struct fiv *dom, int in,
    float v, int *undef)
{
	float x[FX_MAXIN];

	base_inputs(g, dom, x);
	x[in] = v;
	return fx_eval(g, node, x, undef);
}

/* the least key in [a, b] where P holds (P false then true), b+1 if none */
struct prd {
	const struct fxg *g;
	const struct fiv *dom;
	int node, in, below;	/* below: P = f(x) < v; else f(x) > v */
	float v;
	int dir;
};

static int
prd_at(const struct prd *p, int64_t k)
{
	int u = 0;
	float f = eval_at(p->g, p->node, p->dom, p->in, funkey(k), &u);
	int r;

	if (isnan(f))
		return 0;
	r = p->below ? tkey(f) < tkey(p->v) : tkey(f) > tkey(p->v);
	return r;
}

static int64_t
first_true(const struct prd *p, int64_t a, int64_t b, int want)
{
	/* prd_at == want is a suffix of [a, b] */
	int64_t lo = a, hi = b + 1;

	while (lo < hi) {
		int64_t m = lo + (hi - lo) / 2;
		if (prd_at(p, m) == want)
			hi = m;
		else
			lo = m + 1;
	}
	return lo;
}

/*
 * {x in [lo, hi] : f(x) in [vlo, vhi]} for f = node, monotone (dir) in
 * input `in` (the others at their base values): an interval, as f's
 * preimage of an interval under a monotone map.  1 with *rlo, *rhi.
 */
static int
preimage(const struct fxg *g, const struct fiv *dom, int node, int in,
    int dir, float lo, float hi, float vlo, float vhi, float *rlo,
    float *rhi)
{
	struct prd p;
	int64_t a = tkey(lo), b = tkey(hi), L, R;

	if (a > b)
		return 0;
	/* the defined part first: for a monotone chain the undefined inputs
	 * (sqrt, log2, pow of a negative operand) are a prefix or a suffix */
	{
		int u1 = 0, u2 = 0;
		float f1 = eval_at(g, node, dom, in, lo, &u1);
		float f2 = eval_at(g, node, dom, in, hi, &u2);
		u1 |= isnan(f1);
		u2 |= isnan(f2);
		if (u1 && u2)
			return 0;
		if (u1 || u2) {
			int64_t L = a, H = b + 1;
			while (L < H) {	/* the first key with the other state */
				int64_t m = L + (H - L) / 2, u = 0;
				int uu = 0;
				float f = eval_at(g, node, dom, in, funkey(m), &uu);
				u = uu || isnan(f);
				if (u == (u1 ? 0 : 1))
					H = m;
				else
					L = m + 1;
			}
			if (u1)
				a = L;
			else
				b = L - 1;
			if (a > b)
				return 0;
		}
	}
	p.g = g;
	p.dom = dom;
	p.node = node;
	p.in = in;
	p.dir = dir;
	if (dir > 0) {
		/* below vlo: a prefix; above vhi: a suffix */
		p.below = 1;
		p.v = vlo;
		L = first_true(&p, a, b, 0);	/* first not below */
		p.below = 0;
		p.v = vhi;
		R = first_true(&p, a, b, 1) - 1;	/* last not above */
	} else {
		p.below = 0;
		p.v = vhi;
		L = first_true(&p, a, b, 0);	/* first not above */
		p.below = 1;
		p.v = vlo;
		R = first_true(&p, a, b, 1) - 1;	/* last not below */
	}
	if (L > R)
		return 0;
	*rlo = funkey(L);
	*rhi = funkey(R);
	return 1;
}

/* ---- sets of intervals of one input ---- */

#define ISMAX 8
struct iset {
	int n;
	float lo[ISMAX], hi[ISMAX];
	int maybe;		/* depends on other inputs: "may" */
};

static void
is_add(struct iset *s, float lo, float hi)
{
	if (s->n < ISMAX && !tlt(hi, lo)) {
		s->lo[s->n] = lo;
		s->hi[s->n++] = hi;
	}
}

static struct iset
is_and(const struct iset *a, const struct iset *b)
{
	struct iset r;
	int i, j;

	memset(&r, 0, sizeof r);
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++)
			is_add(&r, tmax(a->lo[i], b->lo[j]), tmin(a->hi[i], b->hi[j]));
	r.maybe = a->maybe | b->maybe;
	return r;
}

/* a minus b */
static struct iset
is_minus(const struct iset *a, const struct iset *b)
{
	struct iset cur = *a, nx;
	int j, i;

	for (j = 0; j < b->n; j++) {
		memset(&nx, 0, sizeof nx);
		nx.maybe = cur.maybe;
		for (i = 0; i < cur.n; i++) {
			float lo = cur.lo[i], hi = cur.hi[i];
			if (tlt(b->hi[j], lo) || tlt(hi, b->lo[j])) {
				is_add(&nx, lo, hi);
				continue;
			}
			if (tlt(lo, b->lo[j]))
				is_add(&nx, lo, tprev(b->lo[j]));
			if (tlt(b->hi[j], hi))
				is_add(&nx, tnext(b->hi[j]), hi);
		}
		cur = nx;
	}
	return cur;
}

/*
 * {x in piece : node in [vlo, vhi]}, x input `in`: exact for a node
 * monotone in x alone (a preimage); for a node not depending on x, all
 * or nothing by its range (or "maybe": depends on another input).
 */
static struct iset
class_set(const struct fxg *g, const struct fiv *dom, int node, int in,
    float plo, float phi, float vlo, float vhi)
{
	struct iset s;
	unsigned d = deps(g, node);
	int m = mono(g, node, in);
	struct fiv dd[FX_MAXIN], r;
	float a, b;

	memset(&s, 0, sizeof s);
	if (d == (1u << in) && (m == 1 || m == -1)) {
		if (preimage(g, dom, node, in, m, plo, phi, vlo, vhi, &a, &b))
			is_add(&s, a, b);
		return s;
	}
	memcpy(dd, dom, sizeof dd);
	dd[in] = fi_clip(dd[in], plo, phi);
	dd[in].nan = 0;
	r = fx_range(g, node, dd);
	if (r.empty || tlt(r.hi, vlo) || tlt(vhi, r.lo))
		return s;
	is_add(&s, plo, phi);
	if (tlt(r.lo, vlo) || tlt(vhi, r.hi) || r.nan)
		s.maybe = 1;
	return s;
}

/* ---- text ---- */

static void
xtext(const struct fxg *g, int i, struct sbuf *b, int depth)
{
	const struct fxn *n = &g->n[i];
	char t[40];
	static const char *const fn[] = {
		[FX_POW] = "pow", [FX_EXP2] = "exp2", [FX_LOG2] = "log2",
		[FX_SQRT] = "sqrt", [FX_ISQRT] = "inversesqrt", [FX_STEP] = "step",
		[FX_MIX] = "mix", [FX_MIN] = "min", [FX_MAX] = "max",
		[FX_ABS] = "abs", [FX_FMA] = "fma"};
	const char *op = n->op == FX_ADD ? " + " : n->op == FX_SUB ? " - " :
	    n->op == FX_MUL ? " * " : n->op == FX_DIV ? " / " : NULL;

	if (depth > 5) {
		sb_puts(b, "...");
		return;
	}
	switch (n->op) {
	case FX_IN: case FX_OPQ:
		sb_puts(b, g->in[n->in].name);
		return;
	case FX_CVT:
		sb_printf(b, "float(%s)", g->in[n->in].name);
		return;
	case FX_K:
		sb_puts(b, fx_str(n->k, t, sizeof t));
		return;
	case FX_NEG:
		sb_puts(b, "-");
		xtext(g, n->a, b, depth + 1);
		return;
	case FX_SELGE:
		sb_puts(b, "(");
		xtext(g, n->a, b, depth + 1);
		sb_puts(b, " >= ");
		xtext(g, n->b, b, depth + 1);
		sb_puts(b, " ? ");
		xtext(g, n->c, b, depth + 1);
		sb_puts(b, " : ");
		xtext(g, n->d, b, depth + 1);
		sb_puts(b, ")");
		return;
	}
	if (op) {
		sb_puts(b, depth ? "(" : "");
		xtext(g, n->a, b, depth + 1);
		sb_puts(b, op);
		xtext(g, n->b, b, depth + 1);
		sb_puts(b, depth ? ")" : "");
		return;
	}
	sb_printf(b, "%s(", n->op < (int)(sizeof fn / sizeof fn[0]) &&
	    fn[n->op] ? fn[n->op] : "?");
	xtext(g, n->a, b, depth + 1);
	if (n->b >= 0) {
		sb_puts(b, ", ");
		xtext(g, n->b, b, depth + 1);
	}
	if (n->c >= 0) {
		sb_puts(b, ", ");
		xtext(g, n->c, b, depth + 1);
	}
	sb_puts(b, ")");
}

static const char *
xstr(const struct fxg *g, int i)
{
	static char ring[4][300];
	static int at;
	struct sbuf b = {0};
	char *r = ring[at++ & 3];

	xtext(g, i, &b, 0);
	snprintf(r, 300, "%s", b.p ? b.p : "");
	free(b.p);
	return r;
}

static const char *
fs(float v)
{
	static char ring[8][40];
	static int at;

	return fx_str(v, ring[at++ & 7], 40);
}

static void
is_text(char *b, size_t n, const struct fxg *g, int in, const struct iset *s)
{
	int i, at = 0;

	b[0] = 0;
	for (i = 0; i < s->n && (size_t)at < n; i++) {
		if (s->lo[i] == s->hi[i] && signbit(s->lo[i]) == signbit(s->hi[i]))
			at += snprintf(b + at, n - (size_t)at, "%s%s = %s",
			    i ? ", " : "", g->in[in].name, fs(s->lo[i]));
		else
			at += snprintf(b + at, n - (size_t)at, "%s%s in [%s, %s]",
			    i ? ", " : "", g->in[in].name, fs(s->lo[i]),
			    fs(s->hi[i]));
	}
}

/* ---- graph building for the rewritten forms ---- */

static int
add_node(struct fxg *g, int op, int a, int b, int c, float k)
{
	struct fxn *n;

	if (g->nn >= FX_MAXN)
		return -1;
	n = &g->n[g->nn];
	memset(n, 0, sizeof *n);
	n->op = op;
	n->a = a;
	n->b = b;
	n->c = c;
	n->d = -1;
	n->k = k;
	n->ins = n->sid = -1;
	return g->nn++;
}

static int
add_k(struct fxg *g, float k)
{
	return add_node(g, FX_K, -1, -1, -1, k);
}

/*
 * A witness that two nodes differ: candidate inputs chosen by the
 * analysis (cand[], for input `in`), each evaluated; the first with
 * different values (NaN payloads ignored, -0 and +0 distinct) is
 * written to wit.  1 when one was found.
 */
static int
witness(const struct fxg *g, const struct fiv *dom, int orig, int rw, int in,
    const float *cand, int nc, char *wit, size_t n, int skip_undef)
{
	int i;

	for (i = 0; i < nc; i++) {
		int u1 = 0, u2 = 0;
		float a, b;
		if (isnan(cand[i]) && !dom[in].nan)
			continue;
		if (!isnan(cand[i]) && !fi_has(dom[in], cand[i]))
			continue;
		a = eval_at(g, orig, dom, in, cand[i], &u1);
		b = eval_at(g, rw, dom, in, cand[i], &u2);
		if (u1 && skip_undef)
			continue;
		if ((isnan(a) && isnan(b)) || tkey(a) == tkey(b))
			continue;
		if (!isnan(a) && !isnan(b) && !isinf(a) && !isinf(b) &&
		    a != 0 && b != 0)
			snprintf(wit, n, "%s = %s: original %s, rewrite %s (%lld "
			    "ulp apart)", g->in[in].name, fs(cand[i]), fs(a), fs(b),
			    ulps_apart(a, b));
		else
			snprintf(wit, n, "%s = %s: original %s, rewrite %s",
			    g->in[in].name, fs(cand[i]), fs(a), fs(b));
		return 1;
	}
	return 0;
}

/* the single input a node depends on, or -1 */
static int
single_input(const struct fxg *g, int node)
{
	unsigned d = deps(g, node);
	int i;

	for (i = 0; i < g->nin; i++)
		if (d == (1u << i))
			return i;
	return -1;
}

/* candidates from the analysis: a domain's ends and their neighbours,
 * values whose significand has its last bit set (the sums round), and
 * extra points */
static int
cands(const struct fiv *d, float *c, int max, const float *extra, int ne)
{
	int n = 0, i;
	float lo, hi;

	for (i = 0; i < ne && n < max; i++)
		c[n++] = extra[i];
	if (d->empty)
		return n;
	lo = isinf(d->lo) ? -FLT_MAX : d->lo;
	hi = isinf(d->hi) ? FLT_MAX : d->hi;
	if (n < max) c[n++] = lo;
	if (n < max) c[n++] = tnext(lo);
	if (n < max) c[n++] = hi;
	if (n < max) c[n++] = tprev(hi);
	/* odd significands near the low end, then spread over the binades
	 * between the ends */
	if (hi > 0 && hi > lo) {
		float v = lo > 0 ? lo : hi / 64;
		for (i = 0; i < 24 && n < max; i++) {
			uint32_t u;
			memcpy(&u, &v, 4);
			u |= 1;
			memcpy(&v, &u, 4);
			if (v > hi)
				break;
			c[n++] = v;
			v = (float)(v * 1.1);
		}
	}
	if (n < max)
		c[n++] = NAN;
	return n;
}

/* ---- source divisors: "x / 12.92", "x / vec3(2.0, 4.0, 1.5)" ---- */

static int
src_divisors(const char *s, double *d, int max)
{
	int n = 0;

	for (; s && *s && n < max; s++) {
		const char *p;
		if (*s != '/' || s[1] == '/' || s[1] == '*')
			continue;
		p = s + 1;
		while (*p == ' ')
			p++;
		if (!strncmp(p, "vec", 3) || !strncmp(p, "float", 5) ||
		    *p == '(') {
			while (*p && *p != '(')
				p++;
			while (*p && *p != ')' && n < max) {
				char *e;
				double v;
				p++;
				while (*p == ' ')
					p++;
				v = strtod(p, &e);
				if (e == p)
					break;
				d[n++] = v;
				p = e;
				while (*p == ' ')
					p++;
				if (*p != ',' && *p != ')')
					break;
			}
		} else {
			char *e;
			double v = strtod(p, &e);
			if (e > p && (strchr(p, '.') == NULL || strchr(p, '.') < e ||
			    1))
				d[n++] = v;
		}
	}
	return n;
}

/* D when the constant K is fl(1/fl(D)) for a divisor D in the source */
static int
src_divisor_of(const struct fxg *g, float K, float *D)
{
	double d[16];
	int n = src_divisors(g->srcline, d, 16), i;

	for (i = 0; i < n; i++) {
		float fd = (float)d[i];
		if (fd != 0 && (float)(1.0 / (double)fd) == K && fd != K) {
			*D = fd;
			return 1;
		}
	}
	return 0;
}

/* ---- the Vulkan precision table (Vulkan 1.3, 'Precision and Operation
 * of SPIR-V Instructions'), as cited in the derivations ---- */

static const char *const vk_div = "OpFDiv: 2.5 ulp (for |y| in [2^-126, "
    "2^126])";
static const char *const vk_pow = "pow: inherited from exp2(y * log2(x)); "
    "exp2: 3 + 2|x| ulp; log2: 3 ulp outside [0.5, 2], absolute error < "
    "2^-21 inside";
static const char *const vk_fma = "fma: inherited from OpFMul followed by "
    "OpFAdd (each correctly rounded)";
static const char *const vk_sqrt = "sqrt: inherited from 1.0 / "
    "inversesqrt(x); inversesqrt: 2 ulp";
static const char *const vk_ctl = "default float controls (no "
    "SignedZeroInfNanPreserve): the sign of a zero, infinities and NaNs "
    "need not be preserved";

/*
 * The least relative deviation pow(q, y) is allowed on q in [l, r]
 * (q > 0): log2's error (absolute 2^-21 in [0.5, 2], else 3 ulp >=
 * 3 |L| 2^-24), times y, through exp2 (factor 2^(y dL)), and exp2's own
 * 3 + 2|t| ulp >= (3 + 2|t|) 2^-24 relative.  The smaller side,
 * 1 - 2^(-y dL)(1 - rho3), is the lower bound used.
 */
static double
pow_allow(double y, double l, double r)
{
	double L1 = log2(l), L2 = log2(r), Lmin, tmin, dL, rho3;

	Lmin = (L1 <= 0 && L2 >= 0) ? 0 : fmin(fabs(L1), fabs(L2));
	if (l >= 0.5 && r <= 2)
		dL = 0x1p-21;
	else if (r < 0.5 || l > 2)
		dL = 3 * Lmin * U24;
	else
		dL = fmin(0x1p-21, 3 * Lmin * U24);
	tmin = fabs(y) * Lmin;
	rho3 = (3 + 2 * tmin) * U24;
	return 1 - exp2(-fabs(y) * dL) * (1 - rho3);
}

/* ---- accept ---- */

int fx_accept(const struct fxrw *r, const struct fxopt *o);

int
fx_accept(const struct fxrw *r, const struct fxopt *o)
{
	if (r->kind == FXR_DIVNOTE || !r->refine || r->gain <= 0)
		return 0;
	if (r->sink)
		return 1;
	if (r->approx)
		return o->approx && o->mode == 2 && r->ulp >= 0 && r->ulp <= o->ulp;
	if (o->mode == 1)
		return r->exact;
	if (o->mode == 2)
		return r->exact || (r->ulp >= 0 && r->ulp <= o->ulp);
	return r->exact || r->vulkan;
}

static struct fxrw *
new_rw(struct fxrw *rw, int *nrw, int max, int kind, int node)
{
	struct fxrw *r;

	if (*nrw >= max)
		return NULL;
	r = &rw[(*nrw)++];
	memset(r, 0, sizeof *r);
	r->kind = kind;
	r->node = node;
	r->x = r->e = r->a = r->b = -1;
	r->ulp = -1;
	r->refine = 1;
	r->dep = -1;
	return r;
}

#define P(...) sb_printf(&r->proof, __VA_ARGS__)

/* ---- mix(a, b, step(e, x)) -> select ---- */

static void
an_select(struct fxg *g, const struct fiv *dom, struct fxrw *r)
{
	const struct fxn *n = &g->n[r->node], *st = &g->n[n->c];
	int a = n->a, b = n->b, x = st->b, e = st->a, xi = -1, sel, i;
	float ev = 0;
	struct iset S0, S1, U, E[6];
	const char *Ew[6];
	int nE = 0, nan_case = 0;
	char t[400];
	float nx;

	r->a = a;
	r->b = b;
	r->x = x;
	r->e = e;
	r->gain = fx_price(FX_MIX) + (st->uses <= 1 ? fx_price(FX_STEP) : 0) -
	    fx_price(FX_SELGE);
	r->cost = fx_price(FX_SELGE);
	snprintf(r->why, sizeof r->why, "mix by a step is a select");
	P("step(e, x) = x < e ? 0 : 1 (GLSL.std.450 Step: NaN fails x < e, "
	    "so it gives 1) and mix(a, b, t) = a*(1 - t) + b*t (FMix), so t is "
	    "0 or 1 and\n");
	P("  t = 0: a*1 + b*0 = a, unless b is +-inf or NaN (b*0 is NaN), or "
	    "a = -0 and b*0 = +0 (-0 + +0 = +0)\n");
	P("  t = 1: a*0 + b*1 = b, unless a is +-inf or NaN, or b = -0 and "
	    "a*0 = +0\n");
	P("  the select !(x < e) ? b : a (OpFUnordGreaterThanEqual: true for "
	    "NaN, as step) is a or b exactly\n");
	sel = add_node(g, FX_SELGE, x, e, b, 0);
	if (sel >= 0)
		g->n[sel].d = a;
	if (g->n[x].op == FX_IN && is_k(g, e, &ev))
		xi = g->n[x].in;
	if (xi < 0 || sel < 0) {
		/* no single input to split on: the conditions as they are */
		snprintf(r->exc, sizeof r->exc, "where b*0 or a*0 is not a zero "
		    "of the right sign (an infinity, a NaN, or a = -0 / b = -0)");
		r->exact = 0;
		r->vulkan = !g->preserve;
		P("  (x is not a plain input: the exceptions stay conditions)\n");
		if (r->vulkan)
			P("  Vulkan: %s, so it is within\n", vk_ctl);
		return;
	}
	{
		/* the two sides of the input x: x < e and !(x < e) */
		float e1 = ev == 0 ? -0.0f : ev;
		float lo = dom[xi].lo, hi = dom[xi].hi;
		struct iset s;
		memset(&S0, 0, sizeof S0);
		memset(&S1, 0, sizeof S1);
		if (!tlt(tprev(e1), lo))
			is_add(&S0, lo, tmin(hi, tprev(e1)));
		if (!tlt(hi, e1))
			is_add(&S1, tmax(lo, e1), hi);
		/* where the original is undefined: pow of a negative base, log2
		 * or inversesqrt of x <= 0, sqrt of x < 0 (GLSL.std.450) */
		memset(&U, 0, sizeof U);
		for (i = 0; i < g->nn && i <= r->node; i++) {
			const struct fxn *q = &g->n[i];
			float vlo = -INFINITY, vhi, k;
			int m, src = q->a;
			if (q->op == FX_POW && is_k(g, q->b, &k))
				vhi = k > 0 ? -FDEN : 0.0f;
			else if (q->op == FX_LOG2 || q->op == FX_ISQRT)
				vhi = 0.0f;
			else if (q->op == FX_SQRT)
				vhi = -FDEN;
			else
				continue;
			if (deps(g, src) != (1u << xi) ||
			    ((m = mono(g, src, xi)) != 1 && m != -1))
				continue;
			s = class_set(g, dom, src, xi, lo, hi, vlo, vhi);
			if (s.n) {
				char u[200];
				is_text(u, sizeof u, g, xi, &s);
				P("  the original is undefined for %s (%s of %s < 0: "
				    "%s is monotone in %s, so this is its preimage)\n",
				    u, q->op == FX_POW ? "pow" : q->op == FX_LOG2 ?
				    "log2" : q->op == FX_SQRT ? "sqrt" : "inversesqrt",
				    xstr(g, src), xstr(g, src), g->in[xi].name);
				for (m = 0; m < s.n; m++)
					is_add(&U, s.lo[m], s.hi[m]);
			}
		}
		/* t = 0 */
		if (S0.n) {
			struct iset bi, bn, az, bp;
			bi = class_set(g, dom, b, xi, S0.lo[0], S0.hi[0], INFINITY,
			    INFINITY);
			bn = class_set(g, dom, b, xi, S0.lo[0], S0.hi[0], -INFINITY,
			    -INFINITY);
			az = class_set(g, dom, a, xi, S0.lo[0], S0.hi[0], -0.0f, -0.0f);
			bp = class_set(g, dom, b, xi, S0.lo[0], S0.hi[0], 0.0f,
			    FLT_MAX);
			E[nE] = is_minus(&bi, &U);
			Ew[nE++] = "b = +inf, b*0 = NaN";
			E[nE] = is_minus(&bn, &U);
			Ew[nE++] = "b = -inf, b*0 = NaN";
			s = is_and(&az, &bp);
			E[nE] = is_minus(&s, &U);
			Ew[nE++] = "a = -0 and b*0 = +0";
		}
		if (S1.n) {
			struct iset ai, an, bz, ap;
			ai = class_set(g, dom, a, xi, S1.lo[0], S1.hi[0], INFINITY,
			    INFINITY);
			an = class_set(g, dom, a, xi, S1.lo[0], S1.hi[0], -INFINITY,
			    -INFINITY);
			bz = class_set(g, dom, b, xi, S1.lo[0], S1.hi[0], -0.0f, -0.0f);
			ap = class_set(g, dom, a, xi, S1.lo[0], S1.hi[0], 0.0f,
			    FLT_MAX);
			E[nE] = is_minus(&ai, &U);
			Ew[nE++] = "a = +inf, a*0 = NaN";
			E[nE] = is_minus(&an, &U);
			Ew[nE++] = "a = -inf, a*0 = NaN";
			s = is_and(&bz, &ap);
			E[nE] = is_minus(&s, &U);
			Ew[nE++] = "b = -0 and a*0 = +0";
		}
		{
			char u[200];
			if (S0.n) {
				is_text(u, sizeof u, g, xi, &S0);
				P("  t = 0 for %s: a = %s, b = %s\n", u, xstr(g, a),
				    xstr(g, b));
			}
			if (S1.n) {
				is_text(u, sizeof u, g, xi, &S1);
				P("  t = 1 for %s%s\n", u, dom[xi].nan ? " and NaN" : "");
			}
			P("  by monotonicity in %s (each a preimage of a value class):"
			    "\n", g->in[xi].name);
			if (g->nin > 1) {
				float bx[FX_MAXIN];
				int z, any = 0;
				base_inputs(g, dom, bx);
				P("    (values shown with ");
				for (z = 0; z < g->nin; z++)
					if (z != xi) {
						P("%s%s%s%s = %s", any ? ", " : "", g->in[z].cvt ?
						    "float(" : "", g->in[z].name, g->in[z].cvt ? ")" : "",
						    fs(bx[z]));
						any = 1;
					}
				P("; the sets hold for every value of %s)\n", any > 1 ?
				    "them" : "it");
			}
		}
	}
	/* the exceptions, each with both values at an end */
	r->exact = 1;
	t[0] = 0;
	for (i = 0; i < nE; i++) {
		char u[200];
		int u1 = 0;
		float v1, v2;
		if (E[i].n == 0) {
			P("    %s: never\n", Ew[i]);
			continue;
		}
		is_text(u, sizeof u, g, xi, &E[i]);
		if (E[i].maybe) {
			P("    %s: possibly for %s (depending on the other "
			    "inputs)\n", Ew[i], u);
			r->exact = 0;
			snprintf(t + strlen(t), sizeof t - strlen(t), "%s%s (%s, "
			    "possibly)", t[0] ? "; " : "", u, Ew[i]);
			continue;
		}
		v1 = eval_at(g, r->node, dom, xi, E[i].lo[0], &u1);
		v2 = eval_at(g, sel, dom, xi, E[i].lo[0], &u1);
		if ((isnan(v1) && isnan(v2)) || tkey(v1) == tkey(v2)) {
			P("    %s: for %s, with the same result\n", Ew[i], u);
			continue;
		}
		P("    %s: for %s (original %s, select %s)\n", Ew[i], u, fs(v1),
		    fs(v2));
		r->exact = 0;
		snprintf(t + strlen(t), sizeof t - strlen(t), "%s%s (original %s, "
		    "select %s)", t[0] ? "; " : "", u, fs(v1), fs(v2));
		if (!r->wit[0])
			snprintf(r->wit, sizeof r->wit, "%s = %s: original %s, select "
			    "%s", g->in[xi].name, fs(E[i].lo[0]), fs(v1), fs(v2));
	}
	/* x = NaN: t = 1, a*0 is NaN when a is */
	if (dom[xi].nan) {
		int u1 = 0;
		float v1 = eval_at(g, r->node, dom, xi, NAN, &u1);
		float v2 = eval_at(g, sel, dom, xi, NAN, &u1);
		nx = v2;
		if (!(isnan(v1) && isnan(v2)) && tkey(v1) != tkey(v2)) {
			nan_case = 1;
			r->exact = 0;
			P("    %s = NaN: a = %s, so a*0 is NaN (original %s, select "
			    "%s)\n", g->in[xi].name, fs(eval_at(g, a, dom, xi, NAN,
			    &u1)), fs(v1), fs(v2));
			snprintf(t + strlen(t), sizeof t - strlen(t), "%s%s = NaN "
			    "(original %s, select %s)", t[0] ? "; " : "",
			    g->in[xi].name, fs(v1), fs(v2));
		} else
			P("    %s = NaN: both NaN\n", g->in[xi].name);
		(void)nx;
	}
	(void)nan_case;
	if (U.n) {
		char u[200];
		is_text(u, sizeof u, g, xi, &U);
		P("  where the original is undefined (%s) any result refines it\n",
		    u);
	}
	snprintf(r->exc, sizeof r->exc, "%s", t);
	r->vulkan = !g->preserve;
	/* finite nonzero results are identical: 0 ulp, -0 and +0 being one
	 * value, an infinity against a NaN left to the float controls */
	r->ulp = r->exact || r->vulkan ? 0 : -1;
	if (r->exact)
		P("  so the select is bit-exact wherever the original is "
		    "defined\n");
	else if (r->vulkan)
		P("  every difference is in the sign of a zero or an infinity "
		    "against a NaN: within Vulkan's %s\n", vk_ctl);
	else
		P("  the module asks for SignedZeroInfNanPreserve: those "
		    "differences count\n");
}

/* ---- x / d -> x * fl(1/d), and spirv-opt's version of it ---- */

static void
an_recip(struct fxg *g, const struct fiv *dom, struct fxrw *r, int x,
    float D, float K)
{
	double delta = (double)K * (double)D - 1;	/* exact: 24 x 24 bits */
	double b24 = fabs(delta) * 0x1p24;
	int in = single_input(g, x);

	r->x = x;
	r->k[0] = K;
	r->nk = 1;
	P("k = fl(1/%s) = %s; k*d = 1 + delta with delta = %.3g (k*d is exact "
	    "in double)\n", fs(D), fs(K), delta);
	if (delta == 0) {
		r->exact = 1;
		r->vulkan = 1;
		r->ulp = 0;
		P("  delta = 0: x*k and x/d are the same real number, rounded once:"
		    " bit-exact for every x\n");
		return;
	}
	P("  x*k = (x/d)(1 + delta), so |fl(x*k) - x/d| <= ulp/2 + |x/d| "
	    "|delta| < (0.5 + 2^24 |delta|) ulp = %.3g ulp (|v| < 2^24 ulp(v); "
	    "subnormal results alike)\n", 0.5 + b24);
	P("  x/d correctly rounded is within 0.5 ulp, and both are floats: they"
	    " differ by at most %d ulp\n", (int)floor(1 + b24 - 1e-12));
	r->ulp = floor(1 + b24 - 1e-12);
	if (fabs(D) >= 0x1p-126 && fabs(D) <= 0x1p126) {
		r->vulkan = 0.5 + b24 <= 2.5;
		P("  Vulkan %s; the multiply is correctly rounded: %.3g <= 2.5, "
		    "so it is within the precision the division already had\n",
		    vk_div, 0.5 + b24);
	} else
		P("  |d| is outside [2^-126, 2^126]: Vulkan bounds the division "
		    "nowhere there; not claimed\n");
	/* a witness: x/d just off a rounding midpoint by less than |delta|:
	 * quotient (M + 1/2) 2^-23 for M from 2^23, x = fl(that * d) */
	if (in >= 0) {
		/* scaled by 2^s and signed into the operand's range */
		struct fiv rx = fx_range(g, x, dom);
		int M, s = 0, sg = 1, j, found = 0;
		for (j = 0; j < 160 && !found; j++) {
			s = (j / 2 % 2 ? -1 : 1) * (j / 4);
			sg = j % 2 ? -1 : 1;
			found = fi_has(rx, (float)(ldexp(1.5, s) * D * sg)) &&
			    fi_has(rx, (float)(ldexp(1.0, s) * D * sg));
		}
		if (!found)
			s = 0, sg = 1;
		for (M = 0; M < 8192; M++) {
			double q = ldexp(1 + (M + 0.5) * 0x1p-23, s) * sg;
			float xv = (float)(q * D);
			int u = 0;
			float a = ev1(FX_DIV, xv, D, 0, 0, &u);
			float b2 = ev1(FX_MUL, xv, K, 0, 0, &u);
			if (tkey(a) != tkey(b2)) {
				snprintf(r->wit, sizeof r->wit, "%s = %s: / %s gives %s, * "
				    "%s gives %s (the quotient within |delta| of a rounding "
				    "midpoint)", xstr(g, x), fs(xv), fs(D), fs(a), fs(K),
				    fs(b2));
				break;
			}
		}
	}
	r->exact = 0;
}

/* ---- (x + A) * K -> fma(x, K1, K2) ---- */

static void
an_fma(struct fxg *g, const struct fiv *dom, struct fxrw *r, int x, float A,
    float K, int add_uses)
{
	float D = 0, K1, K2;
	int haveD = src_divisor_of(g, K, &D), in = single_input(g, x);
	double e1, e2, me;
	struct fiv rx = fx_range(g, x, dom);
	int same_sign;
	char wb[300] = "";

	r->x = x;
	r->gain = fx_price(FX_MUL) + (add_uses <= 1 ? fx_price(FX_ADD) : 0) -
	    fx_price(FX_FMA);
	r->cost = fx_price(FX_FMA);
	if (haveD) {
		K1 = (float)(1.0L / (long double)D);
		K2 = (float)((long double)A / (long double)D);
		e1 = (double)K1 * (double)D - 1;
		e2 = (double)(((long double)K2 * D - A) / A);
		P("the source divides: T = (x + %s) / %s; K1 = fl(1/d) = %s, K2 = "
		    "fl(%s/d) = %s (re-optimised for T), with K1 = (1/d)(1 + e1), "
		    "K2 = (A/d)(1 + e2), e1 = %.3g, e2 = %.3g\n", fs(A), fs(D),
		    fs(K1), fs(A), fs(K2), e1, e2);
	} else {
		K1 = K;
		K2 = (float)((double)A * (double)K);
		e1 = 0;
		e2 = A != 0 ? ((double)K2 - (double)A * K) / ((double)A * K) : 0;
		P("T = (x + %s) * %s; K1 = K, K2 = fl(A*K) = %s = A*K(1 + e2), "
		    "e2 = %.3g (A*K is exact in double)\n", fs(A), fs(K), fs(K2), e2);
	}
	r->k[0] = K1;
	r->k[1] = K2;
	r->nk = 2;
	me = fmax(fabs(e1), fabs(e2));
	same_sign = !rx.empty && ((A >= 0 && !tlt(rx.lo, -0.0f)) ||
	    (A <= 0 && !tlt(0.0f, rx.hi)));
	{
		char u[120];
		snprintf(u, sizeof u, "%s in [%s, %s]%s", xstr(g, x), fs(rx.lo),
		    fs(rx.hi), rx.nan ? " or NaN" : "");
		P("  where it is used, %s\n", u);
	}
	/* an exact case: K a power of two (scaling commutes with rounding),
	 * A*K exact */
	if (!haveD || (haveD && e1 == 0)) {
		int k2 = 0;
		float fr = frexpf(K1, &k2);
		if (fabsf(fr) == 0.5f && e2 == 0 && fabsf(A) >= 0x1p-100f) {
			r->exact = 1;
			r->vulkan = 1;
			r->ulp = 0;
			P("  K1 = 2^%d and K2 = A*K1 exactly: fma(x, K1, K2) = fl((x + "
			    "A) K1) and fl(fl(x + A) K1) = fl(x + A) K1 (scaling by 2^k "
			    "commutes with rounding); where the scaled value would be "
			    "subnormal, |x + A| < 2^-126/K1 puts x within a factor 2 of "
			    "-A, so x + A is exact (Sterbenz): bit-exact for every x\n",
			    k2 - 1);
			return;
		}
	}
	if (!same_sign) {
		/* cancellation: relative error unbounded near x = -A */
		float c[4] = {-A, tnext(-A), tprev(-A)};
		int on = r->node, rn;
		rn = add_node(g, FX_FMA, x, add_k(g, K1), add_k(g, K2), 0);
		r->exact = 0;
		r->vulkan = 0;
		P("  x and A can have opposite signs: near x = -A the exact value "
		    "T is near 0 while K2's rounding leaves |A e2/d|: no relative "
		    "bound, and the original's error is relative\n");
		if (in >= 0 && rn >= 0 && witness(g, dom, on, rn, in, c, 3, wb,
		    sizeof wb, 1))
			snprintf(r->fail, sizeof r->fail, "fails at %s", wb);
		else
			snprintf(r->fail, sizeof r->fail, "no relative bound where x "
			    "+ A cancels");
		return;
	}
	P("  x and A have the same sign, so |x e1 + A e2| / |x + A| <= max(|e1|,"
	    " |e2|) = %.3g = %.3g u: x*K1 + K2 = T(1 + e), |e| <= %.3g\n", me,
	    me / U24, me);
	P("  fma rounds once: |r - T| <= (0.5 + 2^24 * %.3g) ulp = %.3g ulp; "
	    "%s: at most (1 + %.3g) ulp = %.3g ulp\n", me, 0.5 + me * 0x1p24,
	    vk_fma, me * 0x1p24, 1 + me * 0x1p24);
	if (haveD) {
		/* the original may return anything within 2.5 ulp of s/d, s =
		 * fl(x + A): |r - s/d| <= |r - T| + |x + A - s|/d, and
		 * |x + A - s| <= ulp(s)/2 <= 2^ceil(log2 d) ulp(T)/2 */
		double f = ldexp(1, (int)ceil(log2((double)D))) / (2 * D);
		double tot = 1 + me * 0x1p24 + f;
		r->vulkan = tot <= 2.5;
		P("  the original divides s = fl(x + A) by d: allowed anything "
		    "within 2.5 ulp of s/d (%s); |r - s/d| <= |r - T| + |x + A - s|/d"
		    " <= %.3g + ulp(s)/(2d) <= %.3g + 2^ceil(log2 d)/(2d) = %.3g ulp, "
		    "%s 2.5: %s\n", vk_div, 1 + me * 0x1p24, 1 + me * 0x1p24, tot,
		    r->vulkan ? "<=" : ">", r->vulkan ? "within" : "not within");
	} else {
		r->vulkan = 0;
		P("  the original is an add and a multiply, both correctly rounded "
		    "(Vulkan allows them no slack): only a bit-exact form is "
		    "within\n");
	}
	/* against the reference: the module's fl(fl(x + A) K) is within
	 * (2 + 2^24 |eK|) u of T, eK = K's error against 1/d */
	{
		double eK = haveD ? fabs((double)K * D - 1) : 0;
		double rr = (me + U24) + (2 * U24 + eK);
		r->ulp = floor(rr * 0x1p24 * (1 + rr));
		P("  against the reference fl(fl(x + A) * K) (itself within (2u + "
		    "%.3g) of T): |r - ref| < %.3g |T| < %.3g ulp\n", eK, rr,
		    rr * 0x1p24 * (1 + rr));
	}
	{
		float c[40];
		int nc = cands(&rx, c, 40, NULL, 0), rn, on = r->node;
		rn = add_node(g, FX_FMA, x, add_k(g, K1), add_k(g, K2), 0);
		if (in >= 0 && rn >= 0) {
			/* the candidates are values of x; x is an input here */
			if (g->n[x].op == FX_IN &&
			    witness(g, dom, on, rn, in, c, nc, r->wit, sizeof r->wit,
			    1))
				;
		}
	}
	r->exact = 0;
}

/* ---- x * K + B -> fma(x, K, B): a contraction ---- */

static void
an_fma2(struct fxg *g, const struct fiv *dom, struct fxrw *r, int x, float K,
    float B, int mul_uses)
{
	int in = single_input(g, x), k2 = 0;
	float fr = frexpf(K, &k2);
	struct fiv rx = fx_range(g, x, dom);

	r->x = x;
	r->k[0] = K;
	r->k[1] = B;
	r->nk = 2;
	r->gain = fx_price(FX_ADD) + (mul_uses <= 1 ? fx_price(FX_MUL) : 0) -
	    fx_price(FX_FMA);
	r->cost = fx_price(FX_FMA);
	P("x*K + B as one fma: the multiply is not decorated NoContraction, so "
	    "Vulkan lets an implementation fuse it already (SPIR-V NoContraction; "
	    "%s): within\n", vk_fma);
	r->vulkan = 1;
	if (fabsf(fr) == 0.5f) {
		P("  K = 2^%d: x*K is exact unless it underflows, so fl(x*K + B) is "
		    "the same rounding of the same number (bit-exact where x*K is "
		    "normal or zero)\n", k2 - 1);
		r->exact = 0;
		r->ulp = 1;
	} else {
		P("  the unfused form rounds x*K first: |fl(fl(x K) + B) - fl(xK + "
		    "B)| can reach ulp(x K)/2 + ulp/2; with cancellation (x K near "
		    "-B) that is a large relative difference\n");
		r->ulp = -1;
	}
	if (in >= 0 && g->n[x].op == FX_IN) {
		float c[40];
		int nc = cands(&rx, c, 40, NULL, 0);
		int rn = add_node(g, FX_FMA, x, add_k(g, K), add_k(g, B), 0);
		if (rn >= 0)
			witness(g, dom, r->node, rn, in, c, nc, r->wit, sizeof r->wit,
			    1);
	}
}

/* ---- sqrt(x) -> x * inversesqrt(x) ---- */

static void
an_sqrt(struct fxg *g, const struct fiv *dom, struct fxrw *r, int x)
{
	struct fiv rx = fx_range(g, x, dom);
	int in = single_input(g, x);

	r->x = x;
	r->gain = fx_price(FX_SQRT) - fx_price(FX_ISQRT) - fx_price(FX_MUL);
	r->cost = fx_price(FX_ISQRT) + fx_price(FX_MUL);
	snprintf(r->why, sizeof r->why, "sqrt as x * inversesqrt(x)");
	P("where it is used, %s in [%s, %s]%s\n", xstr(g, x), fs(rx.lo),
	    fs(rx.hi), rx.nan ? " or NaN" : "");
	if (rx.empty || !tlt(0.0f, rx.lo)) {
		r->refine = 0;
		P("  x can be 0 (or less): inversesqrt is undefined for x <= 0 "
		    "while sqrt(0) = 0 is defined: not a refinement\n");
		snprintf(r->fail, sizeof r->fail, "inversesqrt(0) is undefined, "
		    "sqrt(0) is not");
		return;
	}
	P("  x > 0, so inversesqrt(x) is defined; with z = inversesqrt(x) = "
	    "(1/sqrt x)(1 + z'), x*z = sqrt(x)(1 + z')\n");
	P("  Vulkan: %s; both forms carry the same inversesqrt error z' "
	    "(first order; the second-order term |z'|^2 < 2^-44 is far below the "
	    "slack), then the original divides (2.5 ulp) where the rewrite "
	    "multiplies (0.5 ulp): within\n", vk_sqrt);
	r->vulkan = 1;
	if (isinf(rx.hi))
		P("  x = +inf: sqrt gives +inf, inf * inversesqrt(inf) = inf * 0 "
		    "= NaN (an infinity against a NaN: %s)\n", vk_ctl);
	P("  against the reference (correctly rounded sqrt and inversesqrt): "
	    "|r - sqrt x| <= (2u + u^2)|sqrt x| + ... < 3u: under 3 ulp of a "
	    "result within 0.5 ulp of it\n");
	r->ulp = 2;
	if (in >= 0 && g->n[x].op == FX_IN) {
		float c[40], ex[6] = {2, 3, 5, 7, 10, 0.3f};
		int nc = cands(&rx, c, 40, ex, 6);
		int iz = add_node(g, FX_ISQRT, x, -1, -1, 0);
		int rn = iz >= 0 ? add_node(g, FX_MUL, x, iz, -1, 0) : -1;
		if (rn >= 0)
			witness(g, dom, r->node, rn, in, c, nc, r->wit, sizeof r->wit,
			    1);
	}
	r->exact = 0;
}

/* ---- pow(q, y) -> exp2(y * log2 q), approximations ---- */

struct powctx {
	ld y;
};

static ld
powfn(void *ctx, ld x)
{
	return powl(x, ((struct powctx *)ctx)->y);
}

/*
 * The certified relative error of P(q - c) (f32 coefficients a[0..n],
 * Horner with fma) against q^y on [qa, qb], per geometric piece, and
 * whether it stays within pow's allowance there.
 */
static void
poly_cert(ld y, float qa, float qb, float c, const float *a, int n,
    double *real, double *total, int *within, double *worst_q)
{
	const int NP = 4096;
	double ratio = pow((double)qb / qa, 1.0 / NP), l = qa;
	double gam = 2 * n * U24 / (1 - 2 * n * U24);
	int p, i;

	*real = *total = 0;
	*within = 1;
	*worst_q = qa;
	for (p = 0; p < NP; p++) {
		double r = p == NP - 1 ? qb : l * ratio, h = r - l;
		ld m = ((ld)l + r) / 2, t = m - c, pv = 0, pd = 0, E, E1;
		double T = fmax(fabs(l - c), fabs(r - c)), S = 0, S1 = 0, M2 = 0;
		double fmn, f2, eabs, rel, rt, tot, al;
		for (i = n; i >= 0; i--) {
			pd = pd * t + pv;
			pv = pv * t + a[i];
		}
		E = pv - powl(m, y);
		E1 = pd - y * powl(m, y - 1);
		for (i = 0; i <= n; i++) {
			S += fabs(a[i]) * pow(T, i);
			if (i >= 1)
				S1 += i * fabs(a[i]) * pow(T, i - 1);
			if (i >= 2)
				M2 += i * (i - 1) * fabs(a[i]) * pow(T, i - 2);
		}
		f2 = fabs((double)(y * (y - 1))) * fmax(pow(l, (double)y - 2),
		    pow(r, (double)y - 2));
		M2 += f2;
		eabs = fabsl(E) + fabsl(E1) * h / 2 + M2 * h * h / 8 +
		    1e-17 * (S + pow(r, (double)y));
		fmn = fmin(pow(l, (double)y), pow(r, (double)y));
		rel = eabs / fmn;
		/* t = fl(q - c): exact (Sterbenz) when c/2 <= q <= 2c */
		rt = (l >= c / 2 && r <= 2 * c) ? 0 : S1 * U24 * T / fmn;
		tot = (1 + rel) * (1 + gam * S / fmn) * (1 + rt) - 1;
		if (rel > *real)
			*real = rel;
		if (tot > *total) {
			*total = tot;
			*worst_q = m;
		}
		al = pow_allow((double)y, l, r);
		if (tot > al)
			*within = 0;
		l = r;
	}
}

static void
an_pow(struct fxg *g, const struct fiv *dom, struct fxrw *rw, int *nrw,
    int max, int node, const struct fxopt *o, int dep)
{
	struct fxrw *r = new_rw(rw, nrw, max, FXR_POWEXP, node);
	const struct fxn *n = &g->n[node];
	int q = n->a, in = single_input(g, q);
	float y = g->n[n->b].k;
	struct fiv rq = fx_range(g, q, dom);
	double tmaxv, rho, rr;

	if (r == NULL)
		return;
	r->dep = dep;
	r->x = q;
	r->k[0] = y;
	r->nk = 1;
	r->gain = fx_price(FX_POW) - 2 * fx_price(FX_EXP2) - fx_price(FX_MUL);
	r->cost = 2 * fx_price(FX_EXP2) + fx_price(FX_MUL);
	snprintf(r->why, sizeof r->why, "pow as exp2(y * log2 x)");
	P("where it is used, the base %s is in [%s, %s]%s%s\n", xstr(g, q),
	    fs(rq.lo), fs(rq.hi), rq.nan ? " or NaN" : "", dep >= 0 ?
	    " (only the select's side for it)" : "");
	if (rq.empty || !tlt(0.0f, rq.lo)) {
		r->refine = 0;
		P("  the base can be <= 0: log2 is undefined at 0 where pow(0, %s) "
		    "= 0 is defined: not a refinement\n", fs(y));
		snprintf(r->fail, sizeof r->fail, "the base can be 0 (log2(0) is "
		    "undefined, pow(0, y) is not)");
		return;
	}
	P("  Vulkan: %s; the rewrite is that expression, so its bound is the "
	    "original's by definition: within\n", vk_pow);
	r->vulkan = 1;
	/* against the reference: log2 and exp2 correctly rounded */
	tmaxv = fabs(y) * fmax(fabs(log2(rq.lo)), fabs(log2(isinf(rq.hi) ?
	    FLT_MAX : rq.hi)));
	if (tmaxv > 152)
		tmaxv = 152;	/* beyond that both overflow or underflow */
	rho = exp2(tmaxv * (2 * U24 + U24 * U24)) * (1 + U24) - 1;
	rr = rho + U24 * (1 + rho);
	r->ulp = floor(rr * 0x1p24 * (1 + rr));
	P("  against the correctly rounded pow: L = log2(q)(1 + d1), t = y L (1 +"
	    " d2), |t^ - t| <= |t|(2u + u^2) with |t| <= %.4g here, so exp2(t^)"
	    "(1 + d3) = q^y 2^(t^ - t)(1 + d3): relative error < %.3g, under "
	    "%.3g ulp of a result within 0.5 ulp\n", tmaxv, rho,
	    rr * 0x1p24 * (1 + rr));
	if (in >= 0 && g->n[q].op != FX_K) {
		float c[40];
		struct fiv d2[FX_MAXIN];
		int nc, l2, m2, e2, k;
		memcpy(d2, dom, sizeof d2);
		nc = cands(&dom[in], c, 40, NULL, 0);
		l2 = add_node(g, FX_LOG2, q, -1, -1, 0);
		k = add_k(g, y);
		m2 = l2 >= 0 && k >= 0 ? add_node(g, FX_MUL, l2, k, -1, 0) : -1;
		e2 = m2 >= 0 ? add_node(g, FX_EXP2, m2, -1, -1, 0) : -1;
		if (e2 >= 0)
			witness(g, d2, node, e2, in, c, nc, r->wit, sizeof r->wit, 1);
	}
	r->exact = 0;

	/* cheaper forms considered: their costs and bounds, then the
	 * polynomial approximations on the reached base */
	{
		const float *mc = model_active()->c;
		double sig = 0.0860713320559342;	/* max log2(1+m) - m */
		double bh = exp2((fabs(y) + 1) * sig / 2) - 1;
		double c_bh = 2 * mc[C_IADD] + 2 * mc[C_CONV] + mc[C_FMUL];
		double ylo = floor(y), fr = y - ylo, yq = ylo, c_sq;
		int nsq = 0, nmul = 0, j;
		for (j = 1; j <= 3; j++)
			if (fr >= ldexp(1, -j) - 1e-12) {
				fr -= ldexp(1, -j);
				yq += ldexp(1, -j);
				nsq = j;
				nmul++;
			}
		nmul += (int)ylo > 1 ? (int)ylo - 1 : 0;
		c_sq = nsq * mc[C_FSQRT] + nmul * mc[C_FMUL];
		P("  considered, none cheaper than exp2(y * log2 x) at %g:\n",
		    r->cost);
		P("    x^%g as x*x*pow(x, %g): a pow remains (%g + %g)\n", y, y - 2,
		    fx_price(FX_POW), 2 * mc[C_FMUL]);
		if (nsq > 0 && !isinf(rq.hi))
			P("    x^%g by %d sqrt and %d multiplies (exponent %g): %g, and "
			    "an approximation: q^(%g) - 1 reaches %.3g on the base's "
			    "range\n", y, nsq, nmul, yq, c_sq, y - yq,
			    fmax(fabs(pow(rq.lo, y - yq) - 1), fabs(pow(isinf(rq.hi) ?
			    1 : rq.hi, y - yq) - 1)));
		P("    the integer seed asfloat(int(%g * (asuint(q) - 0x3f800000)) "
		    "+ 0x3f800000 + bias) (Mitchell: log2(1 + m) - m is in [0, "
		    "%.4f]): %g, error up to 2^(%.3g) - 1 = %.3g with the best bias "
		    "(more with Newton steps)\n", y, sig, c_bh, (fabs(y) + 1) * sig /
		    2, bh);
	}
	if (isinf(rq.hi) || rq.hi > 1e6f) {
		P("  polynomial approximations need a bounded base: assume a range "
		    "(e.g. `c in [0, 1]`) to have them listed\n");
		return;
	}
	{
		/* minimax (relative) of q^y on the reached base, f32
		 * coefficients, certified per piece */
		struct powctx pc;
		struct fitprob pb;
		struct fitres fr;
		int deg, shown = 0;
		double cpow = fx_price(FX_POW), best_cost = r->cost;
		float c0;
		memset(&pb, 0, sizeof pb);
		pc.y = y;
		pb.f = powfn;
		pb.fctx = &pc;
		pb.xa = rq.lo;
		pb.xb = rq.hi;
		{
			double mid = ((double)rq.lo + rq.hi) / 2;
			int e = ilogb(mid);
			c0 = (float)ldexp(round(ldexp(mid, 6 - e)), e - 6);
		}
		pb.c = c0;
		pb.k0 = 0;
		pb.step = 1;
		pb.rel = 1;
		pb.nd = 0;
		pb.method = FM_REMEZ;
		for (deg = 2; deg <= 12; deg++) {
			double cost = deg * fx_price(FX_FMA) + fx_price(FX_SUB);
			float a[16];
			double real, tot, wq;
			int within, i;
			struct fxrw *pr;
			if (cost >= cpow)
				break;
			pb.nt = deg + 1;
			if (!fit_run(&pb, &fr))
				continue;
			for (i = 0; i <= deg; i++)
				a[i] = (float)fr.num[i];
			poly_cert(y, rq.lo, rq.hi, c0, a, deg, &real, &tot, &within,
			    &wq);
			pr = new_rw(rw, nrw, max, FXR_POLY, node);
			if (pr == NULL)
				return;
			pr->dep = dep;
			pr->x = q;
			pr->deg = deg;
			pr->approx = !within;
			pr->vulkan = within;
			pr->k[0] = c0;
			for (i = 0; i <= deg && i + 1 < FX_MAXK; i++)
				pr->k[1 + i] = a[deg - i];
			pr->nk = deg + 2;
			pr->cost = cost;
			pr->gain = cpow - cost;
			pr->ulp = floor((tot + U24) * 0x1p24 * (1 + tot));
			pr->rel = tot;
			snprintf(pr->why, sizeof pr->why, "pow ~ degree-%d minimax "
			    "polynomial", deg);
			sb_printf(&pr->proof, "relative minimax of q^%s on [%s, %s] "
			    "(remez.c), t = q - %s, coefficients rounded to f32, "
			    "Horner with fma:\n", fs(y), fs(rq.lo), fs(rq.hi), fs(c0));
			sb_printf(&pr->proof, "  real error certified on 4096 "
			    "geometric pieces (|E| <= |E(m)| + |E'(m)| h/2 + max|E''| "
			    "h^2/8, E'' bounded by sum i(i-1)|a_i| T^(i-2) + "
			    "|y(y-1)| q^(y-2)): %.3g relative\n", real);
			sb_printf(&pr->proof, "  plus rounding (gamma_%d = %d u/(1 - "
			    "%d u) times sum |a_i||t|^i; fl(q - c) exact by Sterbenz "
			    "for q in [c/2, 2c]): %.3g relative in all, the worst "
			    "near q = %.4g\n", 2 * deg, 2 * deg, 2 * deg, tot, wq);
			sb_printf(&pr->proof, "  against pow's allowance on each piece "
			    "(log2's error times y through exp2, plus exp2's 3 + 2|t| "
			    "ulp, the lower bounds): %s\n", within ? "within on every "
			    "piece" : "exceeds it (an approximation)");
			snprintf(pr->fail, sizeof pr->fail, "%s", within ? "" :
			    "exceeds the precision pow already had");
			(void)best_cost;
			shown++;
		}
		(void)shown;
	}
}

#undef P

/* ---- the search ---- */

/* ---- the sink: an error budget at the root, carried to the rewrites ---- */

/*
 * A value that ends in a format with fewer bits than f32 (an 8-bit
 * UNORM target, a 16-bit float) may move by one step of that format
 * (g->sink, absolute, or g->sinkrel, relative) without the stored
 * result moving by more than one.  That budget at the root is what the
 * rewrites under it may spend together: each rewrite's own bound is an
 * absolute deviation E at its node, and the deviations are carried up
 * to the root through every operation,
 *
 *   dev(n) = E(n) + sum over operands k of max|dn/dk| dev(k) + 4 ulp(n)
 *
 * (the 4 ulp: an operation whose operand moved rounds again), max|dn/dk|
 * bounded on the operands' ranges where the node is used.  The ranges a
 * derivative is bounded on are those of the unperturbed program; the
 * perturbed operands leave them by at most their deviation, so a second
 * pass bounds the derivatives on each range widened by twice the first
 * pass's deviation and must find every deviation inside that widening:
 * then no operand can have left the widened range (the deviations grow
 * continuously with the size of the rewrites' errors and would have to
 * cross it first), and the second pass's bound holds.  A compare or a
 * step whose operand moves has no bound, and a select takes the larger
 * of its sides.
 */
#define SINK_SLOP 4
#define LN2 0.693147180559945309417

/*
 * Magnitudes are taken over the numbers an interval holds: where the
 * reference's value is NaN or undefined nothing is claimed.
 */
static double
fi_mag(struct fiv a, double w)
{
	if (a.empty)
		return 0;
	return fmax(fabs((double)a.lo), fabs((double)a.hi)) + w;
}

/* the least |v| on the interval widened by w; 0 when that reaches 0 */
static double
fi_least(struct fiv a, double w)
{
	double lo = (double)a.lo - w, hi = (double)a.hi + w;

	if (a.empty || (lo <= 0 && hi >= 0))
		return 0;
	return fmin(fabs(lo), fabs(hi));
}

struct sinkctx {
	const struct fxg *g;
	int root;
	struct fiv R[FX_MAXN];		/* each node where it is used */
	struct fiv Rop[FX_MAXN][4];	/* its operands there */
	const struct fxrw *at[FX_MAXN];	/* the rewrite taken at a node */
	double E[FX_MAXN];		/* and its own deviation */
};

/* max |d node / d operand c| with the operands widened by w[] */
static double
sink_part(const struct sinkctx *s, int i, int c, const double *w)
{
	const struct fxn *n = &s->g->n[i];
	int ch[4] = {n->a, n->b, n->c, n->d};
	const struct fiv *o = s->Rop[i];
	double wa = ch[0] >= 0 ? w[ch[0]] : 0, wb = ch[1] >= 0 ? w[ch[1]] : 0;
	double wc = ch[2] >= 0 ? w[ch[2]] : 0, m, lo, hi;

	switch (n->op) {
	case FX_ADD: case FX_SUB: case FX_NEG: case FX_ABS: case FX_MIN:
	case FX_MAX:
		return 1;
	case FX_MUL:
		return c == 0 ? fi_mag(o[1], wb) : fi_mag(o[0], wa);
	case FX_FMA:
		return c == 0 ? fi_mag(o[1], wb) : c == 1 ? fi_mag(o[0], wa) : 1;
	case FX_DIV:
		m = fi_least(o[1], wb);
		if (m <= 0)
			return INFINITY;
		return c == 0 ? 1 / m : fi_mag(o[0], wa) / (m * m);
	case FX_POW: {
		double y;
		if (c != 0 || o[1].empty || o[1].lo != o[1].hi || wb > 0)
			return INFINITY;	/* a moving exponent: not bounded here */
		y = o[1].lo;
		lo = (double)o[0].lo - wa;
		hi = (double)o[0].hi + wa;
		if (o[0].empty || lo <= 0)
			return INFINITY;
		/* |y| q^(y-1) is monotone in q: the larger end */
		return fabs(y) * fmax(pow(lo, y - 1), pow(hi, y - 1));
	}
	case FX_EXP2:
		if (o[0].empty)
			return INFINITY;
		return LN2 * exp2((double)o[0].hi + wa);
	case FX_LOG2:
		lo = (double)o[0].lo - wa;
		if (o[0].empty || lo <= 0)
			return INFINITY;
		return 1 / (LN2 * lo);
	case FX_SQRT:
		lo = (double)o[0].lo - wa;
		if (o[0].empty || lo <= 0)
			return INFINITY;
		return 1 / (2 * sqrt(lo));
	case FX_ISQRT:
		lo = (double)o[0].lo - wa;
		if (o[0].empty || lo <= 0)
			return INFINITY;
		return 0.5 * pow(lo, -1.5);
	case FX_MIX:	/* a (1 - t) + b t */
		if (o[2].empty)
			return INFINITY;
		lo = (double)o[2].lo - wc;
		hi = (double)o[2].hi + wc;
		if (c == 0)
			return fmax(fabs(1 - lo), fabs(1 - hi));
		if (c == 1)
			return fmax(fabs(lo), fabs(hi));
		return fi_mag(o[0], wa) + fi_mag(o[1], wb);
	case FX_SELGE:
		return c >= 2 ? 1 : INFINITY;
	}
	return INFINITY;	/* a step, or an operation not known here */
}

/* max |P'| of a polynomial rewrite on its base widened by w */
static double
sink_polylip(const struct sinkctx *s, const struct fxrw *r, double w)
{
	struct fiv q = s->Rop[r->node][0];
	double c = r->k[0], T, lip = 0, tp = 1;
	int j;

	if (q.empty)
		return INFINITY;
	T = fmax(fabs((double)q.lo - w - c), fabs((double)q.hi + w - c));
	/* P(t) = sum a_j t^j, a_j = k[1 + deg - j] */
	for (j = 1; j <= r->deg; j++) {
		lip += j * fabs((double)r->k[1 + r->deg - j]) * tp;
		tp *= T;
	}
	return lip;
}

/* one pass: the deviations with the ranges widened by w[] (NULL: 0) */
static void
sink_carry(const struct sinkctx *s, const double *w, double *dev)
{
	static const double zero[FX_MAXN];
	int i, c;

	if (w == NULL)
		w = zero;
	for (i = 0; i <= s->root; i++) {
		const struct fxn *n = &s->g->n[i];
		int ch[4] = {n->a, n->b, n->c, n->d}, moved = 0;
		double d = 0;

		dev[i] = 0;
		if (n->op == FX_IN || n->op == FX_K || n->op == FX_CVT ||
		    n->op == FX_OPQ || s->R[i].empty)
			continue;
		for (c = 0; c < 4; c++) {
			double p;
			if (ch[c] < 0 || dev[ch[c]] == 0)
				continue;
			moved = 1;
			if (s->at[i] && s->at[i]->kind == FXR_POLY)
				p = c == 0 ? sink_polylip(s, s->at[i], w[ch[0]]) :
				    INFINITY;
			else
				p = sink_part(s, i, c, w);
			if (n->op == FX_SELGE && c >= 2)
				d = fmax(d, p * dev[ch[c]]);	/* one side runs */
			else
				d += p * dev[ch[c]];
		}
		if (moved)
			d += SINK_SLOP * (0x1p-23 * fi_mag(s->R[i], w[i]) + 0x1p-149);
		dev[i] = d + s->E[i];
	}
}

/* the root's deviation with the rewrites in s->at; INFINITY: no bound */
static double
sink_dev(const struct sinkctx *s)
{
	static double d1[FX_MAXN], d2[FX_MAXN], w[FX_MAXN];
	int i;

	sink_carry(s, NULL, d1);
	for (i = 0; i <= s->root; i++) {
		if (!isfinite(d1[i]))
			return INFINITY;
		w[i] = 2 * d1[i];
	}
	sink_carry(s, w, d2);
	for (i = 0; i <= s->root; i++)
		if (!isfinite(d2[i]) || (d1[i] > 0 && !(d2[i] < w[i])) ||
		    (d1[i] == 0 && d2[i] != 0))
			return INFINITY;
	return d2[s->root];
}

/* a rewrite's own deviation at its node (absolute), from its bound */
static double
sink_own(const struct sinkctx *s, const struct fxrw *r)
{
	double mag = fi_mag(s->R[r->node], 0);

	if (r->exact)
		return 0;
	if (r->kind == FXR_POLY && r->rel > 0)
		return (r->rel + 0x1p-24) * mag;
	if (r->ulp < 0 && r->kind == FXR_FMA2) {
		/* fl(fl(x K) + B) against fma(x, K, B): the product's rounding
		 * and the sum's, half an ulp each; x K is no larger than the
		 * sum of the add's operands' magnitudes */
		double ops = fi_mag(s->Rop[r->node][0], 0) +
		    fi_mag(s->Rop[r->node][1], 0);
		return 0x1p-24 * ops + 0x1p-24 * mag + 0x1p-148;
	}
	if (r->ulp < 0)
		return INFINITY;
	return r->ulp * (0x1p-23 * mag + 0x1p-149);
}

static int
sink_gain_cmp(const void *a, const void *b)
{
	const struct fxrw *x = *(const struct fxrw *const *)a;
	const struct fxrw *y = *(const struct fxrw *const *)b;

	if (x->gain != y->gain)
		return x->gain > y->gain ? -1 : 1;
	return x < y ? -1 : x > y;
}

/*
 * The rewrites the sink's budget allows: first those the other rules
 * accept anyway (they move the root too, and the budget pays for
 * them), then each node's cheapest remaining rewrite, the largest
 * savings first, while the root stays within the budget.
 */
static void
sink_pass(struct fxg *g, int root, const struct fxopt *o, struct fxrw *rw,
    int nrw, struct fiv (*reach)[FX_MAXIN])
{
	static struct sinkctx s;
	static struct fiv r[FX_MAXN];
	const struct fxrw *cand[64];
	struct fiv top;
	double budget, dev, base;
	int i, c, nc = 0, taken = 0;

	if (g->sink <= 0 && g->sinkrel <= 0)
		return;
	memset(&s, 0, sizeof s);
	s.g = g;
	s.root = root;
	for (i = 0; i <= root; i++) {
		const struct fxn *n = &g->n[i];
		int ch[4] = {n->a, n->b, n->c, n->d};
		ranges(g, i, reach[i], r);
		s.R[i] = r[i];
		for (c = 0; c < 4; c++)
			if (ch[c] >= 0)
				s.Rop[i][c] = r[ch[c]];
	}
	top = s.R[root];
	if (g->sink > 0)
		budget = g->sink;
	else {
		/* a relative step: of the least magnitude the root takes */
		double least = fi_least(top, 0);
		if (least <= 0 || least < g->sinkmin) {
			snprintf(g->sinknote, sizeof g->sinknote, "the value's "
			    "magnitude can go below %.3g, where a step of the format "
			    "is not that fraction of it: no budget (assume a range "
			    "that keeps it away from 0)", g->sinkmin);
			return;
		}
		budget = g->sinkrel * least;
	}
	g->sinkbudget = budget;
	/* what the other rules accept: one per node, the most gain */
	for (i = 0; i < nrw; i++) {
		const struct fxrw *x = &rw[i];
		if (!fx_accept(x, o) || x->node > root)
			continue;
		if (!s.at[x->node] || x->gain > s.at[x->node]->gain)
			s.at[x->node] = x;
	}
	for (i = 0; i <= root; i++)
		if (s.at[i])
			s.E[i] = sink_own(&s, s.at[i]);
	base = sink_dev(&s);
	if (!(base <= budget)) {
		snprintf(g->sinknote, sizeof g->sinknote, "the rewrites within "
		    "the original's precision already move the stored value by "
		    "up to %.3g of the %.3g a step allows: nothing more is taken",
		    base, budget);
		return;
	}
	g->sinkdev = base;
	for (i = 0; i < nrw && nc < 64; i++)
		if (!fx_accept(&rw[i], o) && rw[i].kind != FXR_DIVNOTE &&
		    rw[i].refine && rw[i].gain > 0 && rw[i].node <= root)
			cand[nc++] = &rw[i];
	qsort(cand, (size_t)nc, sizeof *cand, sink_gain_cmp);
	for (i = 0; i < nc; i++) {
		const struct fxrw *x = cand[i], *old = s.at[x->node];
		double olde = s.E[x->node];
		struct fxrw *xm = &rw[x - rw];

		if (old && old->gain >= x->gain)
			continue;
		/* whichever of the two is applied, the larger bound covers it */
		s.at[x->node] = x;
		s.E[x->node] = fmax(sink_own(&s, x), old ? olde : 0);
		dev = sink_dev(&s);
		if (!(dev <= budget)) {
			snprintf(xm->sinkfail, sizeof xm->sinkfail, "it would move "
			    "the stored value by up to %.3g, more than the %.3g a "
			    "step of the sink allows", dev, budget);
			s.at[x->node] = old;
			s.E[x->node] = olde;
			continue;
		}
		xm->sink = 1;
		xm->sinkdev = dev;
		g->sinkdev = dev;
		taken++;
		sb_printf(&xm->proof, "  the sink: its own bound moves this value "
		    "by at most %.3g; carried to the stored value through every "
		    "operation above it (max |derivative| on the ranges, widened "
		    "by twice the deviation and checked to stay inside; 4 ulp "
		    "where an operation rounds again), together with the other "
		    "rewrites taken: at most %.3g of the %.3g one step allows\n",
		    s.E[x->node], dev, budget);
	}
	(void)taken;
}

int
fx_rewrites(struct fxg *g, int root, const struct fxopt *o, struct fxrw *rw,
    int max)
{
	static struct fiv reach[FX_MAXN][FX_MAXIN];
	static char seen[FX_MAXN];
	static int rdep[FX_MAXN];
	int nrw = 0, i, j, nn0 = g->nn;
	int selrw[FX_MAXN];

	memset(seen, 0, sizeof seen);
	for (i = 0; i < FX_MAXN; i++) {
		selrw[i] = -1;
		rdep[i] = -1;
	}
	seen[root] = 1;
	for (i = root; i >= 0; i--) {
		const struct fxn *n = &g->n[i];
		if (!seen[i])
			continue;
		if (n->a >= 0) seen[n->a] = 1;
		if (n->b >= 0) seen[n->b] = 1;
		if (n->c >= 0) seen[n->c] = 1;
		if (n->d >= 0) seen[n->d] = 1;
	}
	/* 1: selects, on the full domain */
	{
		struct fiv dom[FX_MAXIN];
		default_dom(g, dom);
		for (i = root; i >= 0; i--) {
			struct fxrw *r;
			if (!seen[i] || i >= nn0 || g->n[i].op != FX_MIX ||
			    g->n[i].c < 0 || g->n[g->n[i].c].op != FX_STEP)
				continue;
			if ((r = new_rw(rw, &nrw, max, FXR_SELECT, i)) == NULL)
				break;
			an_select(g, dom, r);
			if (fx_accept(r, o))
				selrw[i] = nrw - 1;
		}
	}
	/* 2: the domain each node is used on: an accepted select uses b only
	 * where !(x < e), a only where x < e */
	for (i = 0; i <= root; i++)
		default_dom(g, reach[i]);
	{
		static char got[FX_MAXN];
		memset(got, 0, sizeof got);
		got[root] = 1;
		for (i = root; i >= 0; i--) {
			const struct fxn *n = &g->n[i];
			int ch[4] = {n->a, n->b, n->c, n->d}, c;
			if (!got[i] || i >= nn0)
				continue;
			for (c = 0; c < 4; c++) {
				struct fiv d[FX_MAXIN];
				int k = ch[c];
				if (k < 0)
					continue;
				memcpy(d, reach[i], sizeof d);
				if (selrw[i] >= 0 && (c == 0 || c == 1)) {
					const struct fxn *st = &g->n[n->c];
					float e;
					if (g->n[st->b].op == FX_IN && is_k(g, st->a, &e)) {
						int xi = g->n[st->b].in;
						float e1 = e == 0 ? -0.0f : e;
						if (c == 1)	/* b */
							d[xi] = fi_clip(d[xi], e1, INFINITY);
						else {
							d[xi] = fi_clip(d[xi], -INFINITY, tprev(e1));
							d[xi].nan = 0;
						}
					}
				}
				if (!got[k]) {
					memcpy(reach[k], d, sizeof d);
					got[k] = 1;
					rdep[k] = selrw[i] >= 0 && c <= 1 ? selrw[i] :
					    rdep[i];
				} else {
					for (j = 0; j < g->nin; j++)
						reach[k][j] = fi_hull(reach[k][j], d[j]);
					if (rdep[k] != rdep[i])
						rdep[k] = rdep[i] >= 0 ? rdep[i] : rdep[k];
				}
			}
		}
	}
	/* 3: the other rewrites, on the domain each node is used on */
	for (i = root; i >= 0; i--) {
		const struct fxn *n = &g->n[i];
		struct fxrw *r;
		float k, k2;
		if (!seen[i] || i >= nn0)
			continue;
		if (n->op == FX_DIV && is_k(g, n->b, &k) && k != 0 &&
		    !is_k(g, n->a, NULL)) {
			float K = (float)(1.0 / (double)k);
			if ((r = new_rw(rw, &nrw, max, FXR_RECIP, i)) == NULL)
				break;
			r->dep = rdep[i];
			snprintf(r->why, sizeof r->why, "divide by a constant as a "
			    "multiply");
			r->gain = fx_price(FX_DIV) - fx_price(FX_MUL);
			r->cost = fx_price(FX_MUL);
			an_recip(g, reach[i], r, n->a, k, K);
			continue;
		}
		if (n->op == FX_MUL && (is_k(g, n->b, &k) || is_k(g, n->a, &k))) {
			int x = is_k(g, n->b, NULL) ? n->a : n->b;
			float D;
			if (src_divisor_of(g, k, &D) && !is_k(g, x, NULL)) {
				if ((r = new_rw(rw, &nrw, max, FXR_DIVNOTE, i)) == NULL)
					break;
				r->dep = rdep[i];
				snprintf(r->why, sizeof r->why, "spirv-opt -O: x / %s as x "
				    "* %s", fs(D), fs(k));
				an_recip(g, reach[i], r, x, D, k);
			}
			/* (x + A) * K -> fma */
			if (g->n[x].op == FX_ADD && (is_k(g, g->n[x].b, &k2) ||
			    is_k(g, g->n[x].a, &k2))) {
				int xx = is_k(g, g->n[x].b, NULL) ? g->n[x].a :
				    g->n[x].b;
				if (!is_k(g, xx, NULL)) {
					if ((r = new_rw(rw, &nrw, max, FXR_FMA, i)) == NULL)
						break;
					r->dep = rdep[i];
					snprintf(r->why, sizeof r->why, "add and multiply as "
					    "one fma");
					an_fma(g, reach[i], r, xx, k2, k, g->n[x].uses);
				}
			}
			continue;
		}
		if (n->op == FX_ADD) {
			int m = -1, bb = -1;
			float K;
			if (g->n[n->a].op == FX_MUL && is_k(g, n->b, &k2))
				m = n->a, bb = n->b;
			else if (g->n[n->b].op == FX_MUL && is_k(g, n->a, &k2))
				m = n->b, bb = n->a;
			if (m >= 0 && (is_k(g, g->n[m].b, &K) || is_k(g, g->n[m].a,
			    &K))) {
				int xx = is_k(g, g->n[m].b, NULL) ? g->n[m].a : g->n[m].b;
				if (!is_k(g, xx, NULL)) {
					if ((r = new_rw(rw, &nrw, max, FXR_FMA2, i)) == NULL)
						break;
					r->dep = rdep[i];
					snprintf(r->why, sizeof r->why, "multiply-add as one "
					    "fma");
					an_fma2(g, reach[i], r, xx, K, k2, g->n[m].uses);
				}
			}
			(void)bb;
			continue;
		}
		if (n->op == FX_POW && is_k(g, n->b, &k) && !is_k(g, n->a, NULL)) {
			an_pow(g, reach[i], rw, &nrw, max, i, o, rdep[i]);
			continue;
		}
		if (n->op == FX_SQRT && !is_k(g, n->a, NULL)) {
			if ((r = new_rw(rw, &nrw, max, FXR_SQRT, i)) == NULL)
				break;
			r->dep = rdep[i];
			an_sqrt(g, reach[i], r, n->a);
			continue;
		}
	}
	sink_pass(g, root, o, rw, nrw, reach);
	g->nn = nn0;	/* the evaluation nodes go */
	return nrw;
}

void
fx_free(struct fxrw *rw, int n)
{
	int i;

	for (i = 0; i < n; i++)
		free(rw[i].proof.p);
}
