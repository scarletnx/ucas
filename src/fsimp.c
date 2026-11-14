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
#include "memk.h"
#include "numsearch.h"

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
	/* a table read is proposed, never written */
	if (r->kind == FXR_DIVNOTE || r->kind == FXR_TABLE || !r->refine ||
	    r->gain <= 0)
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

/* ---- the coefficient searches on the polynomials (numsearch.c) ---- */

#define FXS_NP 2000

/* the certificate's bound (poly_cert's relative one, pf_cert's absolute
 * one) at points of the range, P's coefficients a whole number of ulps
 * from base[] */
struct fxsctx {
	const float *base;
	float a[16];
	int n, rel, np;
	float c;
	float xs[FXS_NP];
	ld ys[FXS_NP];
};

static double
fxs_obj(void *ctx, const int *off)
{
	struct fxsctx *s = ctx;
	double worst = 0, gam = 2 * s->n * U24 / (1 - 2 * s->n * U24);
	int i, j;

	for (i = 0; i <= s->n; i++) {
		float v = s->base[i];
		for (j = 0; j < abs(off[i]); j++)
			v = nextafterf(v, off[i] > 0 ? INFINITY : -INFINITY);
		s->a[i] = v;
	}
	for (i = 0; i < s->np; i++) {
		ld t = (ld)s->xs[i] - s->c, pv = 0, fy = fabsl(s->ys[i]);
		double T = fabs((double)t), S = 0, S1 = 0, ex, rt, tot;
		int exact = s->c == 0 || (s->c > 0 && s->xs[i] >= s->c / 2 &&
		    s->xs[i] <= 2 * s->c);
		for (j = s->n; j >= 0; j--)
			pv = pv * t + s->a[j];
		for (j = 0; j <= s->n; j++) {
			S += fabs(s->a[j]) * pow(T, j);
			if (j >= 1)
				S1 += j * fabs(s->a[j]) * pow(T, j - 1);
		}
		ex = (double)fabsl(pv - s->ys[i]);
		rt = exact ? 0 : S1 * U24 * T;
		if (s->rel)
			tot = (1 + ex / (double)fy) * (1 + gam * S / (double)fy) *
			    (1 + rt / (double)fy) - 1;
		else
			tot = (ex + gam * S + rt) * (1 + gam);
		if (!(tot <= 1e300))
			return INFINITY;
		if (tot > worst)
			worst = tot;
	}
	return worst;
}

/*
 * The search the options name, over the f32 coefficients a[0..n] of
 * P(x - c) on [xa, xb], from where rounding the fit left them; the
 * objective the certificate's bound (relative or absolute) at 2000
 * points of the range (1000 Chebyshev-spaced, 1000 even in the floats'
 * order).  found[] gets what it finds; 1 when that is better there.
 */
static int
fx_search(const struct fxopt *o, ldfn f, void *fc, float xa, float xb,
    float c, int rel, const float *a, int n, float *found,
    struct nsres *res, int *npts)
{
	struct fxsctx *s = xcalloc(1, sizeof *s);
	struct nsopt no;
	int best[16], i, m = FXS_NP / 2, moved;
	int64_t ka = fkey(32, xa), kb = fkey(32, xb);

	s->base = a;
	s->n = n;
	s->rel = rel;
	s->c = c;
	for (i = 0; i < 2 * m; i++) {
		float x = i < m ? (float)(xa + (1 - cosl(3.14159265358979323846L *
		    i / (m - 1))) * ((ld)xb - xa) / 2) : (float)fromkey(32, ka +
		    (int64_t)(((ld)kb - ka) * (i - m) / (m - 1)));
		ld y;
		if (!(x >= xa && x <= xb))
			continue;
		y = f(fc, x);
		if (!isfinite(y) || (rel && y == 0))
			continue;
		s->xs[s->np] = x;
		s->ys[s->np++] = y;
	}
	ns_defaults(&no);
	no.kind = o->search;
	no.seed = o->seed;
	memset(best, 0, sizeof best);
	ns_run(n + 1, fxs_obj, s, &no, best, res);
	fxs_obj(s, best);
	memcpy(found, s->a, (size_t)(n + 1) * sizeof *found);
	moved = res->after < res->before;
	*npts = s->np;
	free(s);
	return moved;
}

/* the search's line in a polynomial's derivation */
static void
fx_search_line(struct sbuf *b, const struct fxopt *o, const struct nsres *sr,
    int np, int moved, double cert0, double cert1, const char *unit)
{
	sb_printf(b, "  search: %s over the f32 coefficients' ulp offsets from "
	    "the rounded fit's, seed %llu, %ld evaluations of the bound above "
	    "at %d points of the range: %.6g -> %.6g %s there",
	    ns_name(o->search),
	    (unsigned long long)o->seed, sr->evals, np, sr->before, sr->after,
	    unit);
	if (!moved)
		sb_printf(b, " (nothing better found)\n");
	else if (cert1 < cert0)
		sb_printf(b, "; certified as above: %.6g against the rounded fit's "
		    "%.6g: kept\n", cert1, cert0);
	else
		sb_printf(b, "; certified %.6g against the rounded fit's %.6g: not "
		    "kept\n", cert1, cert0);
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
			double real, tot, wq, scert0 = 0, scert1 = 0;
			int within, i, smoved = 0, snp = 0;
			struct fxrw *pr;
			struct nsres sr;
			if (cost >= cpow)
				break;
			pb.nt = deg + 1;
			if (!fit_run(&pb, &fr))
				continue;
			for (i = 0; i <= deg; i++)
				a[i] = (float)fr.num[i];
			poly_cert(y, rq.lo, rq.hi, c0, a, deg, &real, &tot, &within,
			    &wq);
			if (o->search) {
				float b[16];
				double real2, tot2, wq2;
				int within2;
				smoved = fx_search(o, powfn, &pc, rq.lo, rq.hi, c0, 1, a,
				    deg, b, &sr, &snp);
				scert0 = scert1 = tot;
				if (smoved) {
					poly_cert(y, rq.lo, rq.hi, c0, b, deg, &real2, &tot2,
					    &within2, &wq2);
					scert1 = tot2;
					if (tot2 < tot) {
						memcpy(a, b, (size_t)(deg + 1) * sizeof *a);
						real = real2;
						tot = tot2;
						within = within2;
						wq = wq2;
					}
				}
			}
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
			if (o->search)
				fx_search_line(&pr->proof, o, &sr, snp, smoved, scert0,
				    scert1, "relative");
			snprintf(pr->fail, sizeof pr->fail, "%s", within ? "" :
			    "exceeds the precision pow already had");
			(void)best_cost;
			shown++;
		}
		(void)shown;
	}
}

/* ---- exp2, log2, sqrt, inversesqrt as polynomials ---- */

#define PF_LN2 0.693147180559945309417L

/*
 * f on a bounded range, by its value, first derivative and the largest
 * |f''| on a piece (each monotone there, so the larger end's), and the
 * precision Vulkan gives it there (a lower bound, absolute: n ulp of a
 * value v is at least n u |v|).
 */
static ld
pf_val(int op, ld x)
{
	switch (op) {
	case FX_EXP2: return exp2l(x);
	case FX_LOG2: return log2l(x);
	case FX_SQRT: return sqrtl(x);
	}
	return 1 / sqrtl(x);
}

static ld
pf_d1(int op, ld x)
{
	switch (op) {
	case FX_EXP2: return exp2l(x) * (ld)PF_LN2;
	case FX_LOG2: return 1 / (x * (ld)PF_LN2);
	case FX_SQRT: return 0.5L / sqrtl(x);
	}
	return -0.5L / (x * sqrtl(x));
}

static double
pf_d2max(int op, double l, double r)
{
	switch (op) {
	case FX_EXP2: return exp2(r) * PF_LN2 * PF_LN2;
	case FX_LOG2: return 1 / (l * l * PF_LN2);
	case FX_SQRT: return 0.25 / (l * sqrt(l));
	}
	return 0.75 / (l * l * sqrt(l));
}

static double
pf_allow(int op, double l, double r)
{
	double fmn = fmin(fabs((double)pf_val(op, l)), fabs((double)pf_val(op,
	    r))), L;

	switch (op) {
	case FX_EXP2:	/* 3 + 2|x| ulp */
		return (3 + 2 * fmin(fabs(l), fabs(r)) * (l * r > 0)) * U24 * fmn;
	case FX_LOG2:	/* absolute 2^-21 inside [0.5, 2], 3 ulp outside */
		if (l >= 0.5 && r <= 2)
			return 0x1p-21;
		L = (l <= 1 && r >= 1) ? 0 : fmn;
		return l < 0.5 && r > 0.5 ? fmin(0x1p-21, 3 * U24 * L) :
		    3 * U24 * L;
	}
	return 2 * U24 * fmn;	/* inversesqrt 2 ulp; sqrt inherits it */
}

static const char *
pf_name(int op)
{
	return op == FX_EXP2 ? "exp2" : op == FX_LOG2 ? "log2" : op == FX_SQRT ?
	    "sqrt" : "inversesqrt";
}

struct pfctx {
	int op;
};

static ld
pffn(void *ctx, ld x)
{
	return pf_val(((struct pfctx *)ctx)->op, x);
}

/*
 * The certified absolute error of P(x - c) (f32 coefficients a[0..n],
 * Horner with fma) against f on [xa, xb], on 4096 pieces (geometric for
 * a range above 0, even otherwise), and whether it stays within f's
 * precision there.
 */
static void
pf_cert(int op, float xa, float xb, float c, const float *a, int n,
    double *real, double *total, int *within, double *worst_x)
{
	const int NP = 4096;
	int geo = xa > 0, p, i;
	double ratio = geo ? pow((double)xb / xa, 1.0 / NP) : 0;
	double step = ((double)xb - xa) / NP, l = xa;
	double gam = 2 * n * U24 / (1 - 2 * n * U24);

	*real = *total = 0;
	*within = 1;
	*worst_x = xa;
	for (p = 0; p < NP; p++) {
		double r = p == NP - 1 ? xb : geo ? l * ratio : xa + (p + 1) * step;
		double h = r - l, T = fmax(fabs(l - c), fabs(r - c));
		double S = 0, S1 = 0, M2 = 0, eabs, rt, tot;
		ld m = ((ld)l + r) / 2, t = m - c, pv = 0, pd = 0;
		for (i = n; i >= 0; i--) {
			pd = pd * t + pv;
			pv = pv * t + a[i];
		}
		for (i = 0; i <= n; i++) {
			S += fabs(a[i]) * pow(T, i);
			if (i >= 1)
				S1 += i * fabs(a[i]) * pow(T, i - 1);
			if (i >= 2)
				M2 += i * (i - 1) * fabs(a[i]) * pow(T, i - 2);
		}
		M2 += pf_d2max(op, l, r);
		eabs = fabsl(pv - pf_val(op, m)) + fabsl(pd - pf_d1(op, m)) * h /
		    2 + M2 * h * h / 8 + 1e-17 * (S + fabs((double)pf_val(op, r)) +
		    fabs((double)pf_val(op, l)));
		/* t = fl(x - c): exact when c is 0, and (Sterbenz) when c/2 <=
		 * x <= 2c */
		rt = (c == 0 || (c > 0 && l >= c / 2 && r <= 2 * c)) ? 0 :
		    S1 * U24 * T;
		tot = (eabs + gam * S + rt) * (1 + gam);
		if (eabs > *real)
			*real = eabs;
		if (tot > *total) {
			*total = tot;
			*worst_x = m;
		}
		if (tot > pf_allow(op, l, r))
			*within = 0;
		l = r;
	}
}

static void
an_poly(struct fxg *g, const struct fiv *dom, struct fxrw *rw, int *nrw,
    int max, int node, const struct fxopt *o, int dep)
{
	const struct fxn *n = &g->n[node];
	int op = n->op, x = n->a, deg;
	struct fiv rx = fx_range(g, x, dom);
	struct pfctx pc;
	struct fitprob pb;
	struct fitres fr;
	double cf = fx_price(op), fmn;
	float c0;

	/* a bounded range, inside the function's domain (above 0 for log2
	 * and inversesqrt; sqrt's derivative grows without bound at 0) */
	if (rx.empty || rx.nan || isinf(rx.lo) || isinf(rx.hi) || !(rx.lo <
	    rx.hi) || (op != FX_EXP2 && !(rx.lo > 0)) || (op == FX_EXP2 &&
	    (rx.lo < -126 || rx.hi > 127)))
		return;
	{
		double mid = ((double)rx.lo + rx.hi) / 2;
		int e = mid != 0 ? ilogb(mid) : 0;
		c0 = mid != 0 ? (float)ldexp(round(ldexp(mid, 6 - e)), e - 6) : 0;
	}
	fmn = fmin(fabs((double)pf_val(op, rx.lo)), fabs((double)pf_val(op,
	    rx.hi)));
	if (op == FX_LOG2 && rx.lo <= 1 && rx.hi >= 1)
		fmn = 0;	/* a zero inside: no relative bound */
	memset(&pb, 0, sizeof pb);
	pc.op = op;
	pb.f = pffn;
	pb.fctx = &pc;
	pb.xa = rx.lo;
	pb.xb = rx.hi;
	pb.c = c0;
	pb.k0 = 0;
	pb.step = 1;
	pb.rel = 0;
	pb.nd = 0;
	pb.method = FM_REMEZ;
	for (deg = 1; deg <= 12; deg++) {
		double cost = deg * fx_price(FX_FMA) + fx_price(FX_SUB);
		double real, tot, wx, scert0 = 0, scert1 = 0;
		float a[16];
		int within, i, smoved = 0, snp = 0;
		struct fxrw *pr;
		struct nsres sr;
		if (cost >= cf)
			break;
		pb.nt = deg + 1;
		if (!fit_run(&pb, &fr))
			continue;
		for (i = 0; i <= deg; i++)
			a[i] = (float)fr.num[i];
		pf_cert(op, rx.lo, rx.hi, c0, a, deg, &real, &tot, &within, &wx);
		if (o->search) {
			float b[16];
			double real2, tot2, wx2;
			int within2;
			smoved = fx_search(o, pffn, &pc, rx.lo, rx.hi, c0, 0, a, deg,
			    b, &sr, &snp);
			scert0 = scert1 = tot;
			if (smoved) {
				pf_cert(op, rx.lo, rx.hi, c0, b, deg, &real2, &tot2,
				    &within2, &wx2);
				scert1 = tot2;
				if (tot2 < tot) {
					memcpy(a, b, (size_t)(deg + 1) * sizeof *a);
					real = real2;
					tot = tot2;
					within = within2;
					wx = wx2;
				}
			}
		}
		pr = new_rw(rw, nrw, max, FXR_POLY, node);
		if (pr == NULL)
			return;
		pr->dep = dep;
		pr->x = x;
		pr->deg = deg;
		pr->approx = !within;
		pr->vulkan = within;
		pr->exact = 0;
		pr->k[0] = c0;
		for (i = 0; i <= deg && i + 1 < FX_MAXK; i++)
			pr->k[1 + i] = a[deg - i];
		pr->nk = deg + 2;
		pr->cost = cost;
		pr->gain = cf - cost;
		pr->abserr = tot;
		pr->rel = fmn > 0 ? tot / fmn : 0;
		pr->ulp = fmn > 0 ? floor((pr->rel + U24) * 0x1p24 * (1 + pr->rel))
		    : -1;
		snprintf(pr->why, sizeof pr->why, "%s ~ degree-%d minimax "
		    "polynomial", pf_name(op), deg);
		sb_printf(&pr->proof, "absolute minimax of %s(x) on [%s, %s] "
		    "(remez.c), t = x - %s, coefficients rounded to f32, Horner "
		    "with fma:\n", pf_name(op), fs(rx.lo), fs(rx.hi), fs(c0));
		sb_printf(&pr->proof, "  real error certified on 4096 %s pieces "
		    "(|E| <= |E(m)| + |E'(m)| h/2 + max|E''| h^2/8, E'' bounded by "
		    "sum i(i-1)|a_i| T^(i-2) + max|%s''| on the piece): %.3g "
		    "absolute\n", rx.lo > 0 ? "geometric" : "even", pf_name(op),
		    real);
		sb_printf(&pr->proof, "  plus rounding (gamma_%d = %d u/(1 - %d u) "
		    "times sum |a_i||t|^i; fl(x - c) exact for c = 0, and by "
		    "Sterbenz for x in [c/2, 2c]): %.3g absolute in all, the worst "
		    "near x = %.4g%s\n",
		    2 * deg, 2 * deg, 2 * deg, tot, wx, fmn > 0 ? "" : " (the "
		    "function is 0 inside the range: no relative bound)");
		sb_printf(&pr->proof, "  against %s's precision on each piece (%s, "
		    "the lower bounds): %s\n", pf_name(op), op == FX_EXP2 ?
		    "3 + 2|x| ulp" : op == FX_LOG2 ? "3 ulp outside [0.5, 2], "
		    "absolute 2^-21 inside" : "2 ulp", within ? "within on every "
		    "piece" : "exceeds it (an approximation)");
		if (o->search)
			fx_search_line(&pr->proof, o, &sr, snp, smoved, scert0, scert1,
			    "absolute");
		snprintf(pr->fail, sizeof pr->fail, "%s", within ? "" : op ==
		    FX_EXP2 ? "exceeds the precision exp2 already had" : op ==
		    FX_LOG2 ? "exceeds the precision log2 already had" : op ==
		    FX_SQRT ? "exceeds the precision sqrt already had" :
		    "exceeds the precision inversesqrt already had");
	}
}

#undef P

/* ---- a polynomial already in the graph, refitted to fewer terms ---- */

#define PL_MAXD 12

struct plinfo {
	int ok, x, deg;		/* a polynomial of degree deg in node x */
	int leaf;		/* the variable itself: x (an input's every
				 * load is one variable) */
	ld c[PL_MAXD + 1];	/* its exact coefficients, of x^0 first */
};

/* node i as a polynomial in one node that is not one (its x): adds,
 * subtracts, multiplies, fmas and negations of it and of constants */
static void
pl_of(const struct fxg *g, int i, struct plinfo *P)
{
	const struct fxn *n = &g->n[i];
	struct plinfo *A = n->a >= 0 ? &P[n->a] : NULL, *B = n->b >= 0 ?
	    &P[n->b] : NULL, *C = n->c >= 0 ? &P[n->c] : NULL;
	struct plinfo *r = &P[i];
	int j, k, x = -1;

	memset(r, 0, sizeof *r);
	r->x = -1;
	if (n->op == FX_K) {
		r->ok = 1;
		r->c[0] = n->k;
		return;
	}
	if (n->op == FX_ADD || n->op == FX_SUB || n->op == FX_MUL ||
	    n->op == FX_FMA || n->op == FX_NEG) {
		struct plinfo *o[3] = {A, B, C};
		int no = n->op == FX_NEG ? 1 : n->op == FX_FMA ? 3 : 2, good = 1;
		for (j = 0; j < no; j++) {
			if (o[j] == NULL || !o[j]->ok)
				good = 0;
			else if (o[j]->x >= 0) {
				if (x >= 0 && x != o[j]->x)
					good = 0;
				x = o[j]->x;
			}
		}
		if (good) {
			ld t[2 * PL_MAXD + 2] = {0};
			int dt = 0;
			if (n->op == FX_NEG) {
				for (j = 0; j <= A->deg; j++)
					t[j] = -A->c[j];
				dt = A->deg;
			} else if (n->op == FX_ADD || n->op == FX_SUB) {
				dt = A->deg > B->deg ? A->deg : B->deg;
				for (j = 0; j <= dt; j++)
					t[j] = (j <= A->deg ? A->c[j] : 0) + (n->op == FX_SUB ?
					    -1 : 1) * (j <= B->deg ? B->c[j] : 0);
			} else {
				dt = A->deg + B->deg;
				for (j = 0; j <= A->deg; j++)
					for (k = 0; k <= B->deg; k++)
						t[j + k] += A->c[j] * B->c[k];
				if (n->op == FX_FMA) {
					if (C->deg > dt)
						dt = C->deg;
					for (j = 0; j <= C->deg; j++)
						t[j] += C->c[j];
				}
			}
			if (dt <= PL_MAXD) {
				r->ok = 1;
				r->x = x;
				r->deg = dt;
				for (j = 0; j <= dt; j++)
					r->c[j] = t[j];
				return;
			}
		}
	}
	/* anything else is a variable for the nodes above it; the loads of
	 * one input are one variable */
	r->ok = r->leaf = 1;
	r->x = i;
	if (n->op == FX_IN || n->op == FX_CVT)
		for (j = 0; j < i; j++)
			if (g->n[j].op == n->op && g->n[j].in == n->in) {
				r->x = j;
				break;
			}
	r->deg = 1;
	r->c[1] = 1;
}

/* the polynomial's own operations under root (in x), marked in in[] */
static int
pl_mark(const struct fxg *g, const struct plinfo *P, int i, int x,
    uint8_t *in)
{
	const struct fxn *n = &g->n[i];
	int k = 0;

	if (i < 0 || P[i].leaf || in[i] || n->op == FX_K)
		return 0;
	in[i] = 1;
	k = 1;
	k += pl_mark(g, P, n->a, x, in);
	if (n->op != FX_NEG)
		k += pl_mark(g, P, n->b, x, in);
	if (n->op == FX_FMA)
		k += pl_mark(g, P, n->c, x, in);
	return k;
}

/* the largest magnitude the polynomial's operations reach with |x| <= X,
 * each operand by its own (the sum of products the rounding acts on) */
static double
pl_mag(const struct fxg *g, const struct plinfo *P, int i, double X)
{
	const struct fxn *n = &g->n[i];

	if (P[i].leaf)
		return X;
	switch (n->op) {
	case FX_K: return fabs(n->k);
	case FX_NEG: return pl_mag(g, P, n->a, X);
	case FX_ADD: case FX_SUB:
		return pl_mag(g, P, n->a, X) + pl_mag(g, P, n->b, X);
	case FX_MUL: return pl_mag(g, P, n->a, X) * pl_mag(g, P, n->b, X);
	case FX_FMA:
		return pl_mag(g, P, n->a, X) * pl_mag(g, P, n->b, X) +
		    pl_mag(g, P, n->c, X);
	}
	return X;
}

struct plctx {
	const ld *b;
	int n;
};

static ld
plfn(void *ctx, ld x)
{
	const struct plctx *p = ctx;
	ld v = 0;
	int i;

	for (i = p->n; i >= 0; i--)
		v = v * x + p->b[i];
	return v;
}

/*
 * The certified absolute error of Q(x - c) (f32 coefficients a[0..m],
 * Horner with fma) against the polynomial B (exact coefficients b[0..n]
 * in x) on [xa, xb], on 4096 even pieces, its rounding included.
 */
static void
pl_cert(const ld *b, int nb, float xa, float xb, float c, const float *a,
    int m, double *real, double *total, double *worst_x, double *minabs)
{
	const int NP = 4096;
	int p, i;
	double step = ((double)xb - xa) / NP, l = xa;
	double gam = 2 * m * U24 / (1 - 2 * m * U24);

	*real = *total = 0;
	*worst_x = xa;
	*minabs = INFINITY;
	for (p = 0; p < NP; p++) {
		double r = p == NP - 1 ? xb : xa + (p + 1) * step;
		double h = r - l, T = fmax(fabs(l - c), fabs(r - c));
		double X = fmax(fabs(l), fabs(r)), S = 0, S1 = 0, M2 = 0, e, rt, tot;
		double B2 = 0, lowb;
		ld mid = ((ld)l + r) / 2, t = mid - c, qv = 0, qd = 0, bv = 0, bd = 0;
		for (i = m; i >= 0; i--) {
			qd = qd * t + qv;
			qv = qv * t + a[i];
		}
		for (i = nb; i >= 0; i--) {
			bd = bd * mid + bv;
			bv = bv * mid + b[i];
		}
		for (i = 0; i <= m; i++) {
			S += fabs(a[i]) * pow(T, i);
			if (i >= 1)
				S1 += i * fabs(a[i]) * pow(T, i - 1);
			if (i >= 2)
				M2 += i * (i - 1) * fabs(a[i]) * pow(T, i - 2);
		}
		for (i = 2; i <= nb; i++)
			B2 += i * (i - 1) * fabs((double)b[i]) * pow(X, i - 2);
		M2 += B2;
		/* |B| on the piece, from below: its value at the middle less
		 * what its slope and curvature can take away */
		lowb = fabs((double)bv) - fabs((double)bd) * h / 2 - B2 * h * h / 8;
		if (lowb < *minabs)
			*minabs = lowb > 0 ? lowb : 0;
		e = fabsl(qv - bv) + fabsl(qd - bd) * h / 2 + M2 * h * h / 8 +
		    1e-17 * (S + fabs((double)bv));
		rt = (c == 0 || (c > 0 && l >= c / 2 && r <= 2 * c)) ? 0 :
		    S1 * U24 * T;
		tot = (e + gam * S + rt) * (1 + gam);
		if (e > *real)
			*real = e;
		if (tot > *total) {
			*total = tot;
			*worst_x = (double)mid;
		}
		l = r;
	}
}


/*
 * A polynomial the shader already evaluates (a fitted curve, a Taylor
 * series), refitted on the range its variable reaches to a lower degree:
 * an approximation of it, offered for a sink to take.  Its error is the
 * refit's, certified against the polynomial's exact value, plus the
 * rounding of the new evaluation and of the original one (gamma_k times
 * the magnitudes its k operations reach, Higham's bound for a sum of
 * products).
 */
static void
an_fewer(struct fxg *g, const struct fiv *dom, struct fxrw *rw, int *nrw,
    int max, int node, const struct plinfo *P, int dep)
{
	const struct plinfo *pl = &P[node];
	int x = pl->x, ops, i, deg;
	uint8_t in[FX_MAXN] = {0};
	struct fiv rx = fx_range(g, x, dom);
	double cf = 0, X, gk, r0;
	struct plctx pc;
	struct fitprob pb;
	struct fitres fr;
	float c0;

	if (rx.empty || rx.nan || isinf(rx.lo) || isinf(rx.hi) ||
	    !(rx.lo < rx.hi))
		return;
	ops = pl_mark(g, P, node, x, in);
	/* what goes with it: an operation also read outside it stays */
	for (i = 0; i < g->nn; i++) {
		int j, inner = 0;
		if (!in[i] || i == node)
			continue;
		for (j = 0; j < g->nn; j++)
			if (in[j] && (g->n[j].a == i || g->n[j].b == i ||
			    g->n[j].c == i))
				inner++;
		if (g->n[i].uses > inner)
			return;
	}
	for (i = 0; i < g->nn; i++)
		if (in[i])
			cf += fx_price(g->n[i].op);
	X = fmax(fabs(rx.lo), fabs(rx.hi));
	gk = ops * U24 / (1 - ops * U24);
	r0 = gk * pl_mag(g, P, node, X);
	/* centred at 0 when the range starts near it (no subtract) */
	c0 = rx.lo >= -(rx.hi - rx.lo) && rx.lo <= (rx.hi - rx.lo) ? 0 :
	    (float)(((double)rx.lo + rx.hi) / 2);
	memset(&pb, 0, sizeof pb);
	pc.b = pl->c;
	pc.n = pl->deg;
	pb.f = plfn;
	pb.fctx = &pc;
	pb.xa = rx.lo;
	pb.xb = rx.hi;
	pb.c = c0;
	pb.step = 1;
	pb.method = FM_REMEZ;
	for (deg = 1; deg < pl->deg; deg++) {
		double cost = deg * fx_price(FX_FMA) + (c0 != 0 ? fx_price(FX_SUB) :
		    0), real, tot, wx, fmn;
		float a[PL_MAXD + 1];
		struct fxrw *pr;
		if (cost >= cf)
			break;
		pb.nt = deg + 1;
		if (!fit_run(&pb, &fr))
			continue;
		for (i = 0; i <= deg; i++)
			a[i] = (float)fr.num[i];
		pl_cert(pl->c, pl->deg, rx.lo, rx.hi, c0, a, deg, &real, &tot, &wx,
		    &fmn);
		tot += r0;
		pr = new_rw(rw, nrw, max, FXR_POLY, node);
		if (pr == NULL)
			return;
		pr->dep = dep;
		pr->x = x;
		pr->deg = deg;
		pr->approx = 1;
		pr->exact = pr->vulkan = 0;
		pr->k[0] = c0;
		for (i = 0; i <= deg && i + 1 < FX_MAXK; i++)
			pr->k[1 + i] = a[deg - i];
		pr->nk = deg + 2;
		pr->cost = cost;
		pr->gain = cf - cost;
		pr->abserr = tot;
		pr->rel = fmn > 0 ? tot / fmn : 0;
		pr->ulp = fmn > 0 ? floor((pr->rel + U24) * 0x1p24 * (1 + pr->rel)) :
		    -1;
		snprintf(pr->why, sizeof pr->why, "polynomial of degree %d ~ "
		    "degree-%d minimax polynomial", pl->deg, deg);
		sb_printf(&pr->proof, "the degree-%d polynomial in %s (%d "
		    "operations) refitted: absolute minimax on [%s, %s] (remez.c), "
		    "t = x - %s, coefficients rounded to f32, Horner with fma:\n",
		    pl->deg, xstr(g, x), ops, fs(rx.lo), fs(rx.hi), fs(c0));
		sb_printf(&pr->proof, "  real error against its exact value, "
		    "certified on 4096 even pieces: %.3g absolute\n", real);
		sb_printf(&pr->proof, "  plus the rounding of the new evaluation "
		    "(gamma_%d times sum |a_i||t|^i) and of the original "
		    "(gamma_%d times the magnitudes its operations reach, %.3g): "
		    "%.3g absolute in all, the worst near x = %.4g\n", 2 * deg, ops,
		    r0, tot, wx);
		snprintf(pr->fail, sizeof pr->fail, "an approximation of the "
		    "polynomial: only a sink takes it");
	}
}

/* ---- products and powers in the log domain ---- */

struct ldfac {
	int a, y, e, neg;	/* base (or exp2's exponent when e), exponent node
				 * or -1, negated (a divisor) */
	float yk;		/* the constant exponent */
};

/* the factors of the product under i, and its own operations in in[] */
static int
ld_collect(const struct fxg *g, int i, int neg, struct ldfac *f, int nf,
    ld *logk, uint8_t *in, int *bad)
{
	const struct fxn *n = &g->n[i];
	float k;

	if (*bad)
		return nf;
	if ((n->op == FX_MUL || n->op == FX_DIV) && !in[i]) {
		in[i] = 1;
		nf = ld_collect(g, n->a, neg, f, nf, logk, in, bad);
		return ld_collect(g, n->b, n->op == FX_DIV ? !neg : neg, f, nf,
		    logk, in, bad);
	}
	if (is_k(g, i, &k)) {
		if (!(k > 0) || isinf(k)) {
			*bad = 1;
			return nf;
		}
		*logk += (neg ? -1 : 1) * log2l((ld)k);
		return nf;
	}
	if (nf == FX_MAXF) {
		*bad = 1;
		return nf;
	}
	memset(&f[nf], 0, sizeof f[nf]);
	f[nf].neg = neg;
	f[nf].y = -1;
	f[nf].yk = 1;
	if (n->op == FX_POW && !in[i]) {
		in[i] = 1;
		f[nf].a = n->a;
		if (is_k(g, n->b, &k))
			f[nf].yk = k;
		else if (neg)
			*bad = 1;	/* a variable power of a divisor: not read */
		else
			f[nf].y = n->b;
	} else if (n->op == FX_EXP2 && !in[i]) {
		in[i] = 1;
		f[nf].a = n->a;
		f[nf].e = 1;
	} else
		f[nf].a = i;
	return nf + 1;
}

/*
 * A product of powers (and of exp2s, plain factors, divisors and positive
 * constants) as exp2 of the sum of its exponents times log2 of its bases:
 * pow(a, x) * pow(b, y) is 16 + 1 under gt1030 against two log2, a
 * multiply, an fma and an exp2.  Against the reference (log2, exp2 and
 * pow correctly rounded, every operation rounded): each log2 L(1 + d1),
 * each term y L (1 + d1)(1 + d2), the sum rounding gamma_(k-1) of the
 * terms' magnitudes, so the exponent is off by at most D and the result
 * 2^D (1 + u) - 1 relatively, plus the reference's own rounding of its
 * operations; an approximation, for `approx ulp N` or a sink.
 */
static void
an_logdom(struct fxg *g, const struct fiv *dom, struct fxrw *rw, int *nrw,
    int max, int node, int dep)
{
	struct ldfac f[FX_MAXF];
	uint8_t in[FX_MAXN] = {0};
	int nf, bad = 0, i, j, nlog = 0, ops = 0, terms;
	ld logk = 0;
	double cf = 0, cost, tlog = 0, tall = 0, slo, shi, D, rel, relrw, gk, gt;
	struct fxrw *r;
	struct sbuf why = {0};

	nf = ld_collect(g, node, 0, f, 0, &logk, in, &bad);
	if (bad || nf < 2)
		return;
	/* what goes with it: an operation also read outside it stays */
	for (i = 0; i < g->nn; i++) {
		int inner = 0;
		if (!in[i] || i == node)
			continue;
		for (j = 0; j < g->nn; j++)
			if (in[j] && (g->n[j].a == i || g->n[j].b == i))
				inner++;
		if (g->n[i].uses > inner)
			return;
	}
	for (i = 0; i < g->nn; i++)
		if (in[i]) {
			cf += fx_price(g->n[i].op);
			ops++;
		}
	slo = shi = (double)logk;
	for (j = 0; j < nf; j++) {
		struct fiv ra = fx_range(g, f[j].a, dom), ry;
		double llo, lhi, ylo, yhi, p[4], mn, mx, m;
		int q;
		if (ra.empty || ra.nan || isinf(ra.lo) || isinf(ra.hi))
			return;
		if (f[j].e) {
			llo = ra.lo;
			lhi = ra.hi;
		} else {
			if (!tlt(0.0f, ra.lo))
				return;	/* log2 of 0 or less */
			llo = log2((double)ra.lo);
			lhi = log2((double)ra.hi);
			nlog++;
		}
		if (f[j].y >= 0) {
			ry = fx_range(g, f[j].y, dom);
			if (ry.empty || ry.nan || isinf(ry.lo) || isinf(ry.hi))
				return;
			ylo = ry.lo;
			yhi = ry.hi;
		} else
			ylo = yhi = f[j].neg ? -(double)f[j].yk : f[j].yk;
		p[0] = ylo * llo;
		p[1] = ylo * lhi;
		p[2] = yhi * llo;
		p[3] = yhi * lhi;
		mn = mx = p[0];
		for (q = 1; q < 4; q++) {
			mn = fmin(mn, p[q]);
			mx = fmax(mx, p[q]);
		}
		slo += mn;
		shi += mx;
		m = fmax(fabs(mn), fabs(mx));
		tall += m;
		if (!f[j].e)
			tlog += m;
	}
	if (slo < -126 || shi > 126)
		return;		/* a result outside the normal range */
	/* the new form: the logs, a term each (the first a multiply unless
	 * its power is 1; then an fma, or an add when it is 1), the
	 * constants' log in one add, one exp2 */
	terms = nf + (logk != 0);
	cost = nlog * fx_price(FX_LOG2) + fx_price(FX_EXP2);
	for (j = 0; j < nf; j++) {
		int unit = f[j].y < 0 && (f[j].yk == 1 || f[j].yk == -1);
		if (j == 0)
			cost += unit && !f[j].neg ? 0 : unit ? fx_price(FX_NEG) :
			    fx_price(FX_MUL);
		else
			cost += unit ? fx_price(FX_ADD) : fx_price(FX_FMA);
	}
	if (logk != 0)
		cost += fx_price(FX_ADD);
	if (cost >= cf)
		return;
	r = new_rw(rw, nrw, max, FXR_LOGDOM, node);
	if (r == NULL)
		return;
	r->dep = dep;
	r->nf = nf;
	for (j = 0; j < nf; j++) {
		r->fa[j] = f[j].a;
		r->fy[j] = f[j].y;
		r->fe[j] = (unsigned char)f[j].e;
		r->k[j] = f[j].y >= 0 ? 1 : f[j].neg ? -f[j].yk : f[j].yk;
	}
	r->k[FX_MAXF] = (float)logk;
	r->nk = FX_MAXF + 1;
	r->cost = cost;
	r->gain = cf - cost;
	r->approx = 1;
	r->exact = r->vulkan = 0;
	gt = (terms - 1) * U24 / (1 - (terms - 1) * U24);
	gk = ops * U24 / (1 - ops * U24);
	D = tlog * (2 * U24 + U24 * U24) + gt * (tall + fabs((double)logk)) *
	    (1 + 2 * U24) + fabs((double)logk) * U24;
	relrw = exp2(D) * (1 + U24) - 1;
	rel = relrw + gk * (1 + relrw);
	r->rel = rel;
	r->abserr = rel * exp2(shi);
	r->ulp = floor(rel * 0x1p24 * (1 + rel));
	for (j = 0; j < nf; j++)
		sb_printf(&why, "%s%s", j ? ", " : "", xstr(g, f[j].a));
	snprintf(r->why, sizeof r->why, "a product of powers as exp2 of a sum "
	    "of logs");
	sb_printf(&r->proof, "%s as exp2 of the sum of its %d terms (of %s; "
	    "%d log2), the sum in [%.4g, %.4g] where it is used\n", xstr(g,
	    node), terms, sb_str(&why), nlog, slo, shi);
	sb_printf(&r->proof, "  against the reference (log2, exp2, pow "
	    "correctly rounded): each log2 L(1 + d1), each term y L (1 + d1)(1 + "
	    "d2), the sum gamma_%d of its terms' magnitudes (%.4g): the "
	    "exponent is off by at most %.3g, the result by 2^that (1 + u) - 1 = "
	    "%.3g relatively\n", terms - 1, tall + fabs((double)logk), D, relrw);
	sb_printf(&r->proof, "  plus the reference's own rounding of its %d "
	    "operations (gamma_%d): %.3g relative, %.3g absolute at most (the "
	    "product reaches 2^%.4g), %g ulp\n", ops, ops, rel, r->abserr, shi,
	    r->ulp);
	snprintf(r->fail, sizeof r->fail, "an approximation: `approx ulp N` "
	    "or a sink takes it");
	free(why.p);
}

int
fx_inner(const struct fxg *g, const struct fxrw *r, unsigned char *in)
{
	static struct plinfo P[FX_MAXN];
	int i;

	memset(in, 0, FX_MAXN);
	if (r->kind == FXR_LOGDOM) {
		struct ldfac f[FX_MAXF];
		ld logk = 0;
		int bad = 0;
		ld_collect(g, r->node, 0, f, 0, &logk, in, &bad);
		for (i = 0; i < FX_MAXN && i < g->nn; i++)
			if (in[i])
				return 1;
		return 0;
	}
	if (r->kind != FXR_POLY || strncmp(r->why, "polynomial of degree", 20))
		return 0;
	for (i = 0; i <= r->node && i < g->nn; i++)
		pl_of(g, i, P);
	return P[r->node].leaf ? 0 : pl_mark(g, P, r->node, P[r->node].x, in);
}

/* i, a multiply or a divide, is a product's root: nothing that is read
 * multiplies or divides by it */
static int
ld_whole(const struct fxg *g, int i, int nn0, const char *seen)
{
	int j;

	for (j = i + 1; j < nn0; j++)
		if (seen[j] && (g->n[j].op == FX_MUL || g->n[j].op == FX_DIV) &&
		    (g->n[j].a == i || g->n[j].b == i))
			return 0;
	return 1;
}

/* ---- a table read: a function of one bounded value ---- */

/*
 * A function of one value x on [lo, hi] (an expression whose only input
 * is x) is a table of its values at N evenly spaced knots, read with the
 * texture unit's linear filter: texel j holds the value at x_j = lo + j h,
 * h = (hi - lo)/(N - 1), and its centre is at texel coordinate j + 1/2,
 * so x reads at X = (x - lo)/h + 1/2 = fma(x, A, B) the line between the
 * two texels around it.  The line's distance from the original's f32
 * value is bounded piece by piece: on [p, q] the original's f32 values
 * lie in the interval its own operations give on [p, q] (fx_range: the
 * same rounding, so no rounding analysis of its own), the line's within
 * its values at p, q and the knots between; a piece whose bound is above
 * the goal is halved, down to two adjacent floats, where both values are
 * evaluated.  To that: the coordinate's f32 error and the unit's snap of
 * a weight (by less than 2^-subtexel) times the largest step between
 * neighbouring texels, the filter's own (tex_within ulps of the exact
 * weighted sum), and an R8 read's fma back to the value.
 */

#define TAB_MAXK 11		/* N = 2^k + 1 texels, k = 1..11 */
#define TAB_MAXN ((1 << TAB_MAXK) + 1)
#define TAB_WORK 200000		/* intervals per certification */
#define TAB_TIGHT 50000		/* of them for the tighter bound first */
#define TAB_ALL 400000		/* for every table tried for one value */

static long tab_left;		/* what is left of TAB_ALL */

struct tabw {
	const struct fxg *g;
	int node, in, n;
	ld lo, hi, h;		/* the knots */
	const ld *v;		/* the texels' values */
	int deg;		/* a trend R taken out first: its degree, 0 none */
	ld c[3];		/* R's coefficients (f32 values), c0 + c1 x + c2 x^2 */
};

/* R(x), exactly but for long double's rounding */
static ld
tab_R(const struct tabw *t, ld x)
{
	return t->deg == 0 ? 0 : t->c[0] + x * (t->c[1] + x * t->c[2]);
}

/* R's interval on [p, q], widened by long double's rounding: its ends
 * and, of degree 2, its vertex inside */
static void
tab_Rrange(const struct tabw *t, ld p, ld q, ld *lo, ld *hi)
{
	ld a = tab_R(t, p), b = tab_R(t, q), w;

	*lo = a < b ? a : b;
	*hi = a < b ? b : a;
	if (t->deg == 2 && t->c[2] != 0) {
		ld v = -t->c[1] / (2 * t->c[2]);
		if (v > p && v < q) {
			ld m = tab_R(t, v);
			*lo = m < *lo ? m : *lo;
			*hi = m > *hi ? m : *hi;
		}
	}
	w = 0x1p-60L * (fabsl(t->c[0]) + fabsl(t->c[1]) * fmaxl(fabsl(p),
	    fabsl(q)) + fabsl(t->c[2]) * fmaxl(p * p, q * q));
	*lo -= w;
	*hi += w;
}

/* the original's f32 value at x; *bad when it is not a finite number */
static ld
tab_f(const struct tabw *t, float x, int *bad)
{
	float in[FX_MAXIN] = {0};
	int u = 0;
	float v;

	in[t->in] = x;
	v = fx_eval(t->g, t->node, in, &u);
	if (u || !isfinite(v))
		*bad = 1;
	return (ld)v - tab_R(t, x);
}

/* the line through the texels, at x in [lo, hi] */
static ld
tab_line(const struct tabw *t, ld x)
{
	ld s = (x - t->lo) / t->h;
	int j = (int)floorl(s);

	if (j < 0)
		j = 0;
	if (j > t->n - 2)
		j = t->n - 2;
	return t->v[j] + (t->v[j + 1] - t->v[j]) * (s - j);
}

/* the knots' values: the function at each, or (shift) moved by a
 * quarter of the neighbouring segments' errors at their midpoints, which
 * halves the error of a segment of even curvature; 0: not finite */
static int
tab_targets(const struct tabw *t, ld *v, int shift)
{
	static ld e[TAB_MAXN];
	int j, bad = 0;

	for (j = 0; j < t->n; j++)
		v[j] = tab_f(t, j == t->n - 1 ? (float)t->hi : (float)(t->lo +
		    j * t->h), &bad);
	if (!shift)
		return !bad;
	for (j = 0; j + 1 < t->n; j++)
		e[j] = tab_f(t, (float)(t->lo + (j + 0.5L) * t->h), &bad) -
		    (v[j] + v[j + 1]) / 2;
	for (j = 0; j < t->n; j++)
		v[j] += j == 0 ? e[0] / 2 : j == t->n - 1 ? e[j - 1] / 2 :
		    (e[j - 1] + e[j]) / 4;
	return !bad;
}

/* the largest |f - line| at eight points a segment */
static double
tab_est(const struct tabw *t)
{
	double m = 0;
	int j, s, bad = 0;

	for (j = 0; j + 1 < t->n; j++)
		for (s = 0; s <= 8; s++) {
			float x = (float)(t->lo + (j + s / 8.0L) * t->h);
			if (x < t->lo)
				x = (float)t->lo;
			if (x > t->hi)
				x = (float)t->hi;
			m = fmax(m, fabsl(tab_f(t, x, &bad) - tab_line(t, x)));
		}
	return bad ? INFINITY : m;
}

/* the most |f - line| over every float of [lo, hi], bounded piece by
 * piece within goal; -1 when a piece cannot be brought within it (two
 * adjacent floats, or the work runs out) */
static double
tab_cert(const struct tabw *t, double goal, long work_max, int *npieces,
    double *wx)
{
	static float st[512][2];
	struct fiv dom[FX_MAXIN];
	double worst = 0;
	int j, sp, work = 0, k;

	for (k = 0; k < FX_MAXIN; k++)
		dom[k] = fi_pt(0);
	*npieces = 0;
	for (j = 0; j + 1 < t->n; j++) {
		st[0][0] = j == 0 ? (float)t->lo : (float)(t->lo + j * t->h);
		st[0][1] = j + 2 == t->n ? (float)t->hi : (float)(t->lo + (j + 1) *
		    t->h);
		sp = 1;
		while (sp > 0) {
			float p = st[sp - 1][0], q = st[sp - 1][1], m;
			struct fiv F;
			ld a, b, lmin, lmax, Flo, Fhi;
			double e;
			int bad = 0;
			sp--;
			if (++work > work_max || --tab_left < 0)
				return -1;
			m = (float)(((double)p + q) / 2);
			if (!(m > p && m < q)) {
				/* nothing between them: the two floats themselves */
				e = fmax(fabsl(tab_f(t, p, &bad) - tab_line(t, p)),
				    fabsl(tab_f(t, q, &bad) - tab_line(t, q)));
				if (bad || !(e <= goal))
					return -1;
				if (e > worst)
					worst = e, *wx = p;
				++*npieces;
				continue;
			}
			dom[t->in] = fi_pt(p);
			dom[t->in].hi = q;
			F = fx_range(t->g, t->node, dom);
			if (F.empty || F.nan || F.undef || !isfinite(F.lo) ||
			    !isfinite(F.hi))
				return -1;
			/* the trend taken out: F less R's interval on [p, q] */
			Flo = F.lo;
			Fhi = F.hi;
			if (t->deg) {
				ld rlo, rhi;
				tab_Rrange(t, p, q, &rlo, &rhi);
				Flo -= rhi;
				Fhi -= rlo;
			}
			a = tab_line(t, p);
			b = tab_line(t, q);
			lmin = a < b ? a : b;
			lmax = a < b ? b : a;
			for (k = (int)ceill((p - t->lo) / t->h); k < t->n &&
			    t->lo + k * t->h < q; k++)
				if (k >= 0 && t->lo + k * t->h > p) {
					lmin = t->v[k] < lmin ? t->v[k] : lmin;
					lmax = t->v[k] > lmax ? t->v[k] : lmax;
				}
			e = fmax((double)(Fhi - lmin), (double)(lmax - Flo));
			if (e <= goal) {
				if (e > worst)
					worst = e, *wx = p;
				++*npieces;
				continue;
			}
			if (sp + 2 > 512)
				return -1;
			st[sp][0] = m;
			st[sp][1] = q;
			st[sp + 1][0] = p;
			st[sp + 1][1] = m;
			sp += 2;
		}
	}
	return worst;
}

struct tabres {
	int n, fmt, shift, pieces, uncert;
	double E, Ei, Ec, Ef, Ed, ER, est, dmax, cx, wx;
	float A, B, S, O;
	float tab[TAB_MAXN];
	int deg;		/* the trend's degree, 0 none */
	float c[3];		/* its coefficients, the R8 offset in c[0] */
	double Rmax;		/* its largest magnitude on the range */
};

/*
 * A trend of degree deg (1 or 2) through the knots' values v by least
 * squares, in t = (x - mid) / half for its conditioning, then in x with
 * f32 coefficients: whatever they round to, R is the polynomial they
 * make, and what is certified is f - R.  0: the fit is singular.
 */
static int
tab_fit(const struct tabw *t, const ld *v, int deg, ld *c)
{
	ld M[3][4] = {{0}}, mid = (t->lo + t->hi) / 2, hw = (t->hi - t->lo) / 2;
	ld b[3], f;
	int i, j, k, n = deg + 1;

	for (k = 0; k < t->n; k++) {
		ld x = (t->lo + k * t->h - mid) / hw, pw[5];
		pw[0] = 1;
		for (i = 1; i < 5; i++)
			pw[i] = pw[i - 1] * x;
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++)
				M[i][j] += pw[i + j];
			M[i][3] += pw[i] * v[k];
		}
	}
	for (i = 0; i < n; i++) {
		int piv = i;
		for (j = i + 1; j < n; j++)
			if (fabsl(M[j][i]) > fabsl(M[piv][i]))
				piv = j;
		if (M[piv][i] == 0)
			return 0;
		for (k = 0; k < 4; k++) {
			ld z = M[i][k];
			M[i][k] = M[piv][k];
			M[piv][k] = z;
		}
		for (j = 0; j < n; j++) {
			if (j == i)
				continue;
			f = M[j][i] / M[i][i];
			for (k = i; k < 4; k++)
				if (k < n || k == 3)
					M[j][k] -= f * M[i][k];
		}
	}
	for (i = 0; i < n; i++)
		b[i] = M[i][3] / M[i][i];
	for (i = n; i < 3; i++)
		b[i] = 0;
	/* sum b_i ((x - mid) / hw)^i in powers of x */
	c[2] = b[2] / (hw * hw);
	c[1] = b[1] / hw - 2 * mid * c[2];
	c[0] = b[0] - b[1] * mid / hw + b[2] * mid * mid / (hw * hw);
	for (i = 0; i < 3; i++)
		c[i] = (float)c[i];
	return 1;
}

/* r's function as a table of n texels in format fmt, its own error
 * within goal: 1 with *o filled (o->est and o->E also when 0: the
 * sampled distance and the rest's part, for the report).  deg > 0 (R8
 * only): a trend R of that degree fitted to the knots is taken out, the
 * bytes hold f - R from 0 (R's constant takes the offset) and the read
 * is fma(r, S, R(x)), R by Horner in f32 */
static int
tab_try(const struct fxg *g, const struct fxrw *r, int n, int fmt,
    int deg, double goal, struct tabres *o)
{
	static ld tv[TAB_MAXN], act[TAB_MAXN], best[TAB_MAXN];
	static long kq[TAB_MAXN];
	const struct model *m = model_active();
	struct tabw t;
	ld Aex, Bex, xmax;
	double snap = ldexp(1, -(int)m->par[MP_SUBTEXEL]);
	double W = m->par[MP_TEX_WITHIN], bestv = INFINITY;
	int j, sh;

	t.g = g;
	t.node = r->node;
	t.in = r->tin;
	t.n = n;
	t.lo = r->k[4];
	t.hi = r->k[5];
	t.h = (t.hi - t.lo) / (n - 1);
	t.deg = 0;
	t.c[0] = t.c[1] = t.c[2] = 0;
	Aex = (n - 1) / (t.hi - t.lo);
	Bex = 0.5L - t.lo * Aex;
	memset(o, 0, sizeof *o);
	o->n = n;
	o->fmt = fmt;
	o->est = INFINITY;
	if (deg) {
		ld lo = 0;
		if (fmt != FXT_R8 || !tab_targets(&t, tv, 0) ||
		    !tab_fit(&t, tv, deg, t.c))
			return 0;
		t.deg = deg;
		/* the bytes from 0: R's constant takes f - R's least value */
		if (!tab_targets(&t, tv, 0))
			return 0;
		for (j = 0; j < n; j++)
			lo = j == 0 || tv[j] < lo ? tv[j] : lo;
		t.c[0] = (float)(t.c[0] + lo);
		o->deg = deg;
		for (j = 0; j < 3; j++)
			o->c[j] = (float)t.c[j];
		/* Horner in f32: fma(c1, x, c0), or fma(fma(c2, x, c1), x, c0),
		 * each fma rounding once */
		{
			ld rlo, rhi, m1;
			tab_Rrange(&t, t.lo, t.hi, &rlo, &rhi);
			o->Rmax = (double)fmaxl(fabsl(rlo), fabsl(rhi));
			m1 = fmaxl(fabsl(t.c[2] * t.lo + t.c[1]), fabsl(t.c[2] *
			    t.hi + t.c[1]));
			o->ER = deg == 1 ? U24 * o->Rmax * (1 + U24) : U24 *
			    (double)(m1 * fmaxl(fabsl(t.lo), fabsl(t.hi)) + o->Rmax) *
			    (1 + 2 * U24);
		}
	}
	o->A = (float)Aex;
	o->B = (float)Bex;
	xmax = fmaxl(fabsl(t.lo), fabsl(t.hi));
	/* fl(fma(x, A, B)) against (x - lo)/h + 1/2: A and B rounded, then
	 * the fma's own half ulp */
	o->cx = (double)(xmax * fabsl(o->A - Aex) + fabsl(o->B - Bex) + U24 *
	    (xmax * fabsl((ld)o->A) + fabsl((ld)o->B))) + 0x1p-149;
	for (sh = 0; sh < 2; sh++) {
		ld lo = 0, hi = 0, dmax = 0, vmax = 0;
		double Ec, Ef, Ed, est;
		float S = 0, O = 0;
		t.v = tv;
		if (!tab_targets(&t, tv, sh))
			return 0;
		for (j = 0; j < n; j++) {
			lo = j == 0 || tv[j] < lo ? tv[j] : lo;
			hi = j == 0 || tv[j] > hi ? tv[j] : hi;
		}
		if (fmt == FXT_R8) {
			/* with a trend, from 0 (a value below it clamps, and the
			 * certification sees the clamped line) */
			O = deg ? 0 : (float)lo;
			S = (float)(hi - (deg ? 0 : lo));
			if (!(S > 0))
				return 0;
			for (j = 0; j < n; j++) {
				long q = lroundl((tv[j] - O) / S * 255);
				kq[j] = q < 0 ? 0 : q > 255 ? 255 : q;
				act[j] = O + S * (ld)kq[j] / 255;
			}
		} else
			for (j = 0; j < n; j++)
				act[j] = (float)tv[j];
		for (j = 0; j < n; j++) {
			if (j + 1 < n && fabsl(act[j + 1] - act[j]) > dmax)
				dmax = fabsl(act[j + 1] - act[j]);
			vmax = fabsl(act[j]) > vmax ? fabsl(act[j]) : vmax;
		}
		Ec = (double)dmax * (snap + o->cx);
		Ef = W * 0x1p-23 * (fmt == FXT_R8 ? fabs(S) : (double)vmax);
		Ed = fmt == FXT_R8 ? U24 * (fabs(O) + fabs(S) + (double)o->Rmax) +
		    0x1p-149 : 0;
		t.v = act;
		est = tab_est(&t);
		if (est + Ec + Ef + Ed < bestv) {
			bestv = est + Ec + Ef + Ed;
			o->shift = sh;
			o->est = est;
			o->Ec = Ec;
			o->Ef = Ef;
			o->Ed = Ed;
			o->dmax = (double)dmax;
			o->S = S;
			o->O = O;
			memcpy(best, act, (size_t)n * sizeof *act);
			for (j = 0; j < n; j++)
				o->tab[j] = fmt == FXT_R8 ? (float)kq[j] : (float)act[j];
		}
	}
	o->E = o->Ec + o->Ef + o->Ed + o->ER + 0x1p-149;
	if (!(o->est + o->E <= goal))
		return 0;
	t.v = best;
	/* close to what the samples show first, then the whole allowance */
	o->Ei = -1;
	if (o->est * 1.25 < goal - o->E)
		o->Ei = tab_cert(&t, o->est * 1.25, TAB_TIGHT, &o->pieces, &o->wx);
	if (o->Ei < 0)
		o->Ei = tab_cert(&t, goal - o->E, TAB_WORK, &o->pieces, &o->wx);
	if (o->Ei < 0) {
		o->uncert = 1;
		return 0;
	}
	o->E += o->Ei;
	return 1;
}

/* a table form: its format, the degree of the trend taken out, its
 * read's price */
struct tabform {
	int fmt, deg;
	double c;
};

static const char *
tab_formname(int fmt, int deg)
{
	return fmt == FXT_R32F ? "R32F" : deg == 1 ? "R8 less a line" : deg ==
	    2 ? "R8 less a parabola" : "R8";
}

/* the smallest table within r->tmax in the cheapest form that has one:
 * R8, R8 of f less a trend of degree 1 or 2 (Horner's fmas more), R32F,
 * by the model's prices; r's table, cost and derivation; else why not,
 * in r->sinkfail */
static int
tab_pick(const struct fxg *g, struct fxrw *r)
{
	static struct tabres o, near;
	const struct model *m = model_active();
	double c8, cf = r->gain + r->cost;
	struct tabform fm[4], z;
	int f, k, i, nf = 0, have = 0;

	c8 = model_lists(m, tex8_class(m)) ? m->c[tex8_class(m)] + 2 *
	    fx_price(FX_FMA) : INFINITY;
	for (k = 0; k < 3; k++) {
		fm[nf].fmt = FXT_R8;
		fm[nf].deg = k;
		fm[nf++].c = c8 + k * fx_price(FX_FMA);
	}
	fm[nf].fmt = FXT_R32F;
	fm[nf].deg = 0;
	fm[nf++].c = model_lists(m, C_TEX_SAMPLE_F32) ?
	    m->c[C_TEX_SAMPLE_F32] + fx_price(FX_FMA) : INFINITY;
	/* by price, the order above where they tie */
	for (f = 1; f < nf; f++)
		for (i = f; i > 0 && fm[i].c < fm[i - 1].c; i--) {
			z = fm[i];
			fm[i] = fm[i - 1];
			fm[i - 1] = z;
		}
	near.est = INFINITY;
	tab_left = TAB_ALL;
	for (f = 0; f < nf; f++) {
		if (!(fm[f].c < cf))
			continue;
		for (k = 1; k <= TAB_MAXK; k++) {
			if (tab_try(g, r, (1 << k) + 1, fm[f].fmt, fm[f].deg, r->tmax,
			    &o))
				goto found;
			if (o.est + o.E < near.est + near.E)
				near = o;
		}
		have = 1;
	}
	if (have && isfinite(near.est))
		snprintf(r->sinkfail, sizeof r->sinkfail, "no table of up to %d "
		    "texels is within the %.3g its value may move: %d texels %s "
		    "come to about %.3g (sampled)%s", TAB_MAXN, r->tmax, near.n,
		    tab_formname(near.fmt, near.deg), near.est + near.E,
		    near.uncert ? ", not proved within it on every float before "
		    "the pieces ran out" : "");
	else
		snprintf(r->sinkfail, sizeof r->sinkfail, "no table of up to %d "
		    "texels is within the %.3g its value may move", TAB_MAXN,
		    r->tmax);
	return 0;
found:
	free(r->tab);
	r->tab = xmalloc((size_t)o.n * sizeof *r->tab);
	memcpy(r->tab, o.tab, (size_t)o.n * sizeof *r->tab);
	r->ntab = o.n;
	r->tfmt = o.fmt;
	r->k[0] = o.A;
	r->k[1] = o.B;
	r->k[2] = o.S;
	r->k[3] = o.O;
	r->tdeg = o.deg;
	r->k[6] = o.c[0];
	r->k[7] = o.c[1];
	r->k[8] = o.c[2];
	r->nk = 9;
	r->cost = fm[f].c;
	r->gain = cf - fm[f].c;
	r->abserr = o.E;
	if (o.deg) {
		sb_printf(&r->proof, "f less the trend R(x) = %.9g + %.9g x",
		    (double)o.c[0], (double)o.c[1]);
		if (o.deg == 2)
			sb_printf(&r->proof, " + %.9g x^2", (double)o.c[2]);
		sb_puts(&r->proof, " (f32 coefficients, fitted by least squares to "
		    "f at the knots, its constant taking f - R's least value "
		    "there): ");
	}
	sb_printf(&r->proof, "a table of %d texels at x_j = %.9g + j h, h = %.9g, "
	    "each the value there%s, %s, read with linear filtering at texel "
	    "coordinate fma(x, %.9g, %.9g) (texel j's centre at j + 1/2):\n",
	    o.n, (double)(r->k[4] + 0.0f), (double)(((ld)r->k[5] - r->k[4]) /
	    (o.n - 1)), o.shift ?
	    " moved by a quarter of the two neighbouring segments' errors at "
	    "their midpoints" : "", o.fmt != FXT_R8 ? "stored as f32" : o.deg ?
	    "stored as bytes k for the R8 read k/255 that fma(r, S, R(x)) turns "
	    "back, R by Horner in f32" : "stored as bytes k for the R8 read "
	    "k/255 that fma(r, S, O) turns back", (double)o.A, (double)o.B);
	sb_printf(&r->proof, "  the line between texels against the original's "
	    "f32 value%s, on every float of the range: within %.3g (%d pieces, "
	    "each bounded by the interval of the original's own operations on "
	    "it%s against the line's values at its ends and knots; sampled at 9 "
	    "points a segment, %.3g)\n", o.deg ? " less R" : "", o.Ei, o.pieces,
	    o.deg ? ", less R's interval there (its ends, and its vertex inside)"
	    : "", o.est);
	sb_printf(&r->proof, "  the coordinate: f32 moves it by at most %.3g "
	    "texel and the unit snaps a weight by less than 2^-%d, times the "
	    "largest step between neighbouring texels (%.3g): %.3g\n", o.cx,
	    (int)m->par[MP_SUBTEXEL], o.dmax, o.Ec);
	sb_printf(&r->proof, "  the filter within %g ulp of the exact weighted "
	    "sum (the model's tex_within): %.3g%s", m->par[MP_TEX_WITHIN], o.Ef,
	    o.fmt == FXT_R8 ? "" : "\n");
	if (o.fmt == FXT_R8 && o.deg)
		sb_printf(&r->proof, "; R by Horner in f32, each fma rounding once "
		    "(|R| <= %.3g on the range): %.3g; fma(r, %.9g, R(x)) rounds "
		    "once: %.3g\n", o.Rmax, o.ER, (double)o.S, o.Ed);
	else if (o.fmt == FXT_R8)
		sb_printf(&r->proof, "; fma(r, %.9g, %.9g) rounds once: %.3g\n",
		    (double)o.S, (double)o.O, o.Ed);
	sb_printf(&r->proof, "  in all %.3g, against the %.3g its value may "
	    "move\n", o.E, r->tmax);
	return 1;
}

/* i's function of one value, the largest (no operation that reads i
 * has the same one input), as a table read where the model prices the
 * read below the operations that go with it; only under a sink, whose
 * budget sink_pass carries down to it */
static void
an_table(struct fxg *g, const struct fiv *dom, struct fxrw *rw, int *nrw,
    int max, int node, int in, int dep)
{
	const struct model *m = model_active();
	uint8_t sub[FX_MAXN] = {0}, dead[FX_MAXN] = {0};
	double cf = 0, c8, c32;
	struct fiv F;
	struct fxrw *r;
	int i, j, x = -1;

	if ((g->sink <= 0 && g->sinkrel <= 0) || dep >= 0 || dom[in].empty)
		return;
	/* the operations under it; those read only there go with it */
	sub[node] = 1;
	for (i = node; i >= 0; i--) {
		const struct fxn *n = &g->n[i];
		if (!sub[i])
			continue;
		if (n->op == FX_IN || n->op == FX_CVT) {
			x = i;
			continue;
		}
		if (n->a >= 0) sub[n->a] = 1;
		if (n->b >= 0) sub[n->b] = 1;
		if (n->c >= 0) sub[n->c] = 1;
		if (n->d >= 0) sub[n->d] = 1;
	}
	dead[node] = 1;
	for (i = node - 1; i >= 0; i--) {
		const struct fxn *n = &g->n[i];
		int refs = 0;
		if (!sub[i] || n->op == FX_IN || n->op == FX_K || n->op == FX_CVT ||
		    n->op == FX_OPQ)
			continue;
		for (j = i + 1; j <= node; j++)
			if (dead[j])
				refs += (g->n[j].a == i) + (g->n[j].b == i) +
				    (g->n[j].c == i) + (g->n[j].d == i);
		if (refs > 0 && n->uses <= refs)
			dead[i] = 1;
	}
	for (i = 0; i <= node; i++)
		if (dead[i])
			cf += fx_price(g->n[i].op);
	c8 = model_lists(m, tex8_class(m)) ? m->c[tex8_class(m)] +
	    2 * fx_price(FX_FMA) : INFINITY;
	c32 = model_lists(m, C_TEX_SAMPLE_F32) ? m->c[C_TEX_SAMPLE_F32] +
	    fx_price(FX_FMA) : INFINITY;
	if (x < 0 || !(fmin(c8, c32) < cf))
		return;
	F = fx_range(g, node, dom);
	if (isfinite(dom[in].lo) && isfinite(dom[in].hi) && !dom[in].nan &&
	    (!(dom[in].lo < dom[in].hi) || (F.lo == F.hi && !F.empty &&
	    !F.nan && !F.undef)))
		return;		/* one value on the range */
	if ((r = new_rw(rw, nrw, max, FXR_TABLE, node)) == NULL)
		return;
	r->x = x;
	r->tin = in;
	r->approx = 1;
	r->exact = r->vulkan = 0;
	r->k[4] = dom[in].lo;
	r->k[5] = dom[in].hi;
	r->nk = 6;
	r->cost = fmin(c8, c32);
	r->gain = cf - r->cost;
	snprintf(r->why, sizeof r->why, "a function of %s on [%.9g, %.9g] as "
	    "a table read", g->in[in].name, (double)(dom[in].lo + 0.0f),
	    (double)(dom[in].hi + 0.0f));
	if (!isfinite(dom[in].lo) || !isfinite(dom[in].hi) || dom[in].nan)
		snprintf(r->fail, sizeof r->fail, "a table needs %s's range to be "
		    "bounded and without NaN (assume one: %s in [a, b])",
		    g->in[in].name, g->in[in].name);
	else if (F.empty || F.nan || F.undef || !isfinite(F.lo) ||
	    !isfinite(F.hi))
		snprintf(r->fail, sizeof r->fail, "its value can be NaN, infinite "
		    "or undefined on the range: no table holds it");
	else if (!m->par[MP_SUBTEXEL])
		snprintf(r->fail, sizeof r->fail, "%s does not say what its "
		    "texture unit resolves (subtexel): no table read is offered",
		    sym_str(m->name));
	else if (!m->par[MP_TEX_WITHIN])
		snprintf(r->fail, sizeof r->fail, "%s does not say how far its "
		    "filter is from the exact weighted sum (tex_within): no table "
		    "read is offered", sym_str(m->name));
}

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
		    n->op == FX_OPQ || s->R[i].empty) {
			if (!s->R[i].empty)
				dev[i] = s->E[i];	/* an input's own: a loop's sum */
			continue;
		}
		if (s->at[i] && s->at[i]->kind == FXR_TABLE) {
			/* a read: what is under it no longer reaches it */
			dev[i] = s->E[i];
			continue;
		}
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
	static char live[FX_MAXN];
	int i, cut = 0;

	/* what reaches the root: all of it, but under a table read only
	 * what something else also reads */
	for (i = 0; i <= s->root; i++)
		cut |= s->at[i] && s->at[i]->kind == FXR_TABLE;
	memset(live, !cut, sizeof live);
	if (cut) {
		live[s->root] = 1;
		for (i = s->root; i >= 0; i--) {
			const struct fxn *n = &s->g->n[i];
			if (!live[i] || (s->at[i] && s->at[i]->kind == FXR_TABLE))
				continue;
			if (n->a >= 0) live[n->a] = 1;
			if (n->b >= 0) live[n->b] = 1;
			if (n->c >= 0) live[n->c] = 1;
			if (n->d >= 0) live[n->d] = 1;
		}
	}
	sink_carry(s, NULL, d1);
	for (i = 0; i <= s->root; i++) {
		if (live[i] && !isfinite(d1[i]))
			return INFINITY;
		w[i] = live[i] ? 2 * d1[i] : 0;
	}
	sink_carry(s, w, d2);
	for (i = 0; i <= s->root; i++)
		if (live[i] && (!isfinite(d2[i]) || (d1[i] > 0 && !(d2[i] <
		    w[i])) || (d1[i] == 0 && d2[i] != 0)))
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
	if (r->kind == FXR_POLY && r->abserr > 0)
		return r->abserr + 0x1p-24 * mag + 0x1p-149;
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

/* the table reads, when the sink gives nothing to spend */
static void
tab_nobudget(struct fxrw *rw, int nrw)
{
	int i;

	for (i = 0; i < nrw; i++)
		if (rw[i].kind == FXR_TABLE && !rw[i].fail[0])
			snprintf(rw[i].sinkfail, sizeof rw[i].sinkfail, "the sink "
			    "leaves nothing for it (its note above)");
}

/* the most deviation at node that keeps the stored value within target,
 * the rest as it is in s */
static double
sink_room(struct sinkctx *s, int node, double target)
{
	double lo = 0, hi;
	int it;

	for (hi = target; hi < 0x1p100; hi *= 2) {
		s->E[node] = hi;
		if (!(sink_dev(s) <= target))
			break;
		lo = hi;
	}
	for (it = 0; it < 60 && hi < 0x1p100; it++) {
		double mid = (lo + hi) / 2;
		s->E[node] = mid;
		if (sink_dev(s) <= target)
			lo = mid;
		else
			hi = mid;
	}
	s->E[node] = 0;
	return lo;
}

/*
 * The table reads, on what the sink took.  Together first: every one in
 * place (a read cuts off what is under it), each given the most its own
 * error may move it while the stored value stays within its share of
 * what the rest leaves, the others exact or at the bound already chosen;
 * the smallest table within that.  When those do not hold together (a
 * read that found no table leaves its operations in), each alone, with
 * the whole budget.
 */
static void
sink_tables(struct sinkctx *s, struct fxrw *rw, int nrw, int root,
    double budget)
{
	static const struct fxrw *old[16];
	static double olde[16], cost0[16], gain0[16];
	struct fxrw *t[16];
	int nt = 0, i, pass;
	double d0 = 0, d;

	for (i = 0; i < nrw && nt < 16; i++)
		if (rw[i].kind == FXR_TABLE && !rw[i].fail[0] && rw[i].node <= root)
			t[nt++] = &rw[i];
	for (i = 0; i < nt; i++) {
		old[i] = s->at[t[i]->node];
		olde[i] = s->E[t[i]->node];
		cost0[i] = t[i]->cost;
		gain0[i] = t[i]->gain;
	}
	for (pass = 0; pass < 2 && nt > 0; pass++) {
		int together = pass == 0, n = t[0]->node;
		for (i = 0; i < nt; i++) {
			struct fxrw *r = t[i];
			free(r->proof.p);
			memset(&r->proof, 0, sizeof r->proof);
			free(r->tab);
			r->tab = NULL;
			r->ntab = 0;
			r->sink = 0;
			r->sinkfail[0] = 0;
			r->cost = cost0[i];
			r->gain = gain0[i];
			s->at[r->node] = together ? r : old[i];
			s->E[r->node] = together ? 0 : olde[i];
		}
		if (together)
			d0 = sink_dev(s);
		for (i = 0; i < nt; i++) {
			struct fxrw *r = t[i];
			double target;
			int got = 0;
			n = r->node;
			if (!together) {
				s->at[n] = r;
				s->E[n] = 0;
				d0 = sink_dev(s);
			}
			if (!(d0 <= budget))
				snprintf(r->sinkfail, sizeof r->sinkfail, "the rewrites "
				    "taken already move the stored value by up to %.3g of "
				    "the %.3g a step allows", d0, budget);
			else {
				/* together: the reads before it at their bounds, up to
				 * its share more */
				target = together ? d0 + (budget - d0) * (i + 1) / nt :
				    budget;
				r->tmax = sink_room(s, n, target);
				if (!(r->tmax > 0))
					snprintf(r->sinkfail, sizeof r->sinkfail, "the stored "
					    "value cannot move at all where it is read");
				else
					got = tab_pick(s->g, r);
			}
			if (got)
				s->E[n] = r->abserr;
			if (!together && got) {
				r->sinkdev = sink_dev(s);
				r->sink = 1;
				sb_printf(&r->proof, "  the sink: the most it may move this "
				    "value is the largest bound that, carried to the stored "
				    "value with the other rewrites taken (as for them), stays "
				    "within the step; with this table's, at most %.3g of the "
				    "%.3g one step allows\n", r->sinkdev, budget);
			}
			if (!together || !got) {
				s->at[n] = old[i];
				s->E[n] = olde[i];
			}
		}
		if (!together)
			break;
		d = sink_dev(s);
		if (!(d0 <= budget))
			break;	/* alone, less is cut off */
		if (d <= budget) {
			for (i = 0; i < nt; i++)
				if (t[i]->tab) {
					t[i]->sink = 1;
					t[i]->sinkdev = d;
					if (nt == 1)
						sb_printf(&t[i]->proof, "  the sink: the most it "
						    "may move this value is the largest bound that, "
						    "carried to the stored value with the other "
						    "rewrites taken (as for them), stays within the "
						    "step; with this table's, at most %.3g of the %.3g "
						    "one step allows\n", d, budget);
					else
						sb_printf(&t[i]->proof, "  the sink: with the %d "
						    "table reads here in place, exact, and the other "
						    "rewrites taken, the stored value moves by %.3g; "
						    "the reads in turn, those before at their bounds, "
						    "each takes the largest bound that keeps it within "
						    "one more of %d equal shares of the rest of the "
						    "%.3g one step allows; with every read's, at most "
						    "%.3g\n", nt, d0, nt, budget, d);
				}
			break;
		}
	}
	for (i = 0; i < nt; i++) {
		s->at[t[i]->node] = old[i];
		s->E[t[i]->node] = olde[i];
	}
}

/*
 * The rewrites the sink's budget allows: first those the other rules
 * accept anyway (they move the root too, and the budget pays for
 * them), then each node's cheapest remaining rewrite, the largest
 * savings first, while the root stays within the budget.
 */
static void sink_offers(struct sinkctx *s, double budget);

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
			tab_nobudget(rw, nrw);
			return;
		}
		budget = g->sinkrel * least;
	}
	g->sinkbudget = budget;
	/* the original against the exact value: compensated forms */
	sink_offers(&s, budget);
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
		/* a table read cuts off what is under it: it may still fit */
		sink_tables(&s, rw, nrw, root, budget);
		return;
	}
	g->sinkdev = base;
	for (i = 0; i < nrw && nc < 64; i++)
		if (!fx_accept(&rw[i], o) && rw[i].kind != FXR_DIVNOTE &&
		    rw[i].kind != FXR_TABLE && rw[i].refine && rw[i].gain > 0 && rw[i].node <= root)
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
	/* the table reads, on what was taken */
	sink_tables(&s, rw, nrw, root, budget);
}

/*
 * Node i's program in absolute values with |x| at mx: subtractions as
 * sums, negations dropped, constants by their magnitudes; its rounding
 * operations (a multiply, an add, an fma: one each) counted once in *k.
 * Each rounds as fl(a op b) = (a op b)(1 + d), so the program's value is
 * within gamma_k of this of its exact value (first order is |d| <= u on
 * each path, and no path has more than k roundings).
 */
static double
sink_absval(const struct fxg *g, const struct plinfo *P, int i, int x,
    double mx, int *k, char *seen, int depth)
{
	const struct fxn *n = &g->n[i];
	double a, b, c;

	/* x, or another load of it */
	if (i == x || (P[i].leaf && P[i].x == x))
		return mx;
	if (n->op == FX_K)
		return fabs((double)n->k);
	if (depth > 64)
		return INFINITY;
	a = n->a >= 0 ? sink_absval(g, P, n->a, x, mx, k, seen, depth + 1) : 0;
	b = n->b >= 0 ? sink_absval(g, P, n->b, x, mx, k, seen, depth + 1) : 0;
	c = n->c >= 0 ? sink_absval(g, P, n->c, x, mx, k, seen, depth + 1) : 0;
	if (!seen[i] && n->op != FX_NEG) {
		seen[i] = 1;
		++*k;
	}
	switch (n->op) {
	case FX_ADD:
	case FX_SUB:
		return (a + b) * (1 + 0x1p-40);
	case FX_MUL:
		return a * b * (1 + 0x1p-40);
	case FX_FMA:
		return (a * b + c) * (1 + 0x1p-40);
	case FX_NEG:
		return a;
	}
	return INFINITY;
}

/*
 * The compensated forms the sink asks for (#47): a polynomial of one
 * value (degree 2 at least, the largest of its chain) whose original
 * evaluation errs, against its exact value, by up to gamma_k of its
 * program in absolute values, which carried to the stored value passes
 * the budget; compensated Horner of its exact coefficients errs by at
 * most u |p| + gamma_2n^2 p~ (Graillat, Langlois and Louvet), plus the
 * coefficients' own rounding to f32, and is offered where that carried
 * fits.  An offer, not a rewrite: it moves the stored value toward the
 * exact one, away from the original's.
 */
static void
sink_offers(struct sinkctx *s, double budget)
{
	static struct plinfo P[FX_MAXN];
	static char seen[FX_MAXN];
	const struct fxg *g = s->g;
	int i, j, k, n;

	for (i = 0; i <= s->root; i++)
		pl_of(g, i, P);
	for (i = 0; i <= s->root; i++) {
		const struct fxn *nd = &g->n[i];
		double mx, abs_, ep, ec, dp, dc, pt = 0, ce = 0, gk, g2, c_as = 0,
		    c_comp, u = 0x1p-24, mp, mc;
		int top = 1, x = P[i].x, kk = 0;
		if (!P[i].ok || P[i].leaf || P[i].deg < 2 || x < 0)
			continue;
		/* the largest: nothing above it is a polynomial in the same x */
		for (j = i + 1; j <= s->root && top; j++) {
			const struct fxn *m = &g->n[j];
			if ((m->a == i || m->b == i || m->c == i || m->d == i) &&
			    P[j].ok && !P[j].leaf && P[j].x == x)
				top = 0;
		}
		if (!top)
			continue;
		mx = fi_mag(s->R[x], 0);
		if (!isfinite(mx))
			continue;
		memset(seen, 0, sizeof seen);
		abs_ = sink_absval(g, P, i, x, mx, &kk, seen, 0);
		if (!isfinite(abs_) || kk < 1 || kk * u >= 0.5)
			continue;
		for (k = 0; k <= s->root; k++)
			if (seen[k])
				c_as += fx_price(g->n[k].op);
		gk = kk * u / (1 - kk * u);
		ep = gk * abs_;
		n = P[i].deg;
		for (k = n; k >= 0; k--) {
			double c = (double)P[i].c[k];
			pt = pt * mx + fabs((double)(float)c);
			ce = ce * mx + fabs(c - (double)(float)c);
		}
		g2 = 2 * n * u / (1 - 2 * n * u);
		ec = (u + g2 * g2) * pt * (1 + 0x1p-40) + ce;
		memset(s->E, 0, sizeof s->E);
		s->E[i] = ep;
		dp = sink_dev(s);
		s->E[i] = ec;
		dc = sink_dev(s);
		s->E[i] = 0;
		if (!(dp > budget) || !(dc <= budget))
			continue;
		c_comp = n * (fx_price(FX_MUL) + 2 * fx_price(FX_FMA) + 7 *
		    fx_price(FX_ADD)) + fx_price(FX_ADD);
		/* a check, sampled: 4097 points of x's range, the original's
		 * f32 value and the compensated Horner's against the exact
		 * polynomial (long double) */
		{
			float in[FX_MAXIN] = {0}, lo = s->R[x].lo, hi = s->R[x].hi;
			int q, un = 0, xin = g->n[x].op == FX_IN || g->n[x].op ==
			    FX_CVT ? g->n[x].in : -1;
			mp = mc = 0;
			for (q = 0; q <= 4096 && xin >= 0 && isfinite(lo) &&
			    isfinite(hi); q++) {
				float xv = (float)(lo + (hi - lo) * (q / 4096.0)), sh, cc = 0,
				    v;
				ld ex = 0;
				in[xin] = xv;
				v = fx_eval(g, i, in, &un);
				for (k = n; k >= 0; k--)
					ex = ex * xv + P[i].c[k];
				sh = (float)P[i].c[n];
				for (k = n - 1; k >= 0; k--) {
					volatile float pr = sh * xv, pi = fmaf(sh, xv, -pr),
					    ck = (float)P[i].c[k], t = pr + ck, bb = t - pr;
					volatile float sg = (pr - (t - bb)) + (ck - bb);
					sh = t;
					cc = fmaf(cc, xv, pi + sg);
				}
				if (!un && isfinite(v))
					mp = fmax(mp, (double)fabsl((ld)v - ex));
				mc = fmax(mc, (double)fabsl((ld)(sh + cc) - ex));
			}
		}
		snprintf(g->sinkoffer + strlen(g->sinkoffer), sizeof g->sinkoffer -
		    strlen(g->sinkoffer), "    offered (not applied: it moves the "
		    "stored value toward the exact one, away from the original's): "
		    "%s, a polynomial of degree %d in %s on |%s| <= %.4g, errs as "
		    "written by up to %.3g against its exact value (gamma_%d of its "
		    "operations in absolute values), %.3g at the stored value, more "
		    "than the %.3g a step allows; compensated Horner of its "
		    "coefficients, u |p| + gamma_%d^2 p~%s = %.3g, %.3g at the stored "
		    "value: within it, cost %g against %g; sampled (not a proof) on "
		    "4097 points: %.3g as written, %.3g compensated\n", xstr(g, i),
		    n, xstr(g, x), xstr(g, x), mx, ep, kk, dp, budget, 2 * n, ce > 0 ?
		    " and the coefficients' f32 rounding" : "", ec, dc, c_comp, c_as,
		    mp, mc);
		(void)nd;
	}
	/* a loop's f32 sum read here (spirv.c's lsum_of): its adds' rounding
	 * as written, and as Kahan's compensated sum */
	for (i = 0; i <= s->root; i++) {
		const struct fxn *nd = &g->n[i];
		const struct fxin *f;
		double dp, dk;
		char *off;
		if (nd->op != FX_IN || nd->in < 0 || (!g->in[nd->in].lsum_n &&
		    !g->in[nd->in].lsum_why[0]))
			continue;
		f = &g->in[nd->in];
		off = (char *)g->sinkoffer;	/* the graph's lines, written here */
		if (!f->lsum_n) {
			snprintf(off + strlen(off), sizeof g->sinkoffer - strlen(off),
			    "    %s, a loop's f32 sum s = s + t: no bound on its "
			    "rounding, so Kahan's sum is not weighed: %s\n%s",
			    f->lsum_s, f->lsum_why, f->lsum_src);
			continue;
		}
		memset(s->E, 0, sizeof s->E);
		s->E[i] = f->lsum_plain;
		dp = sink_dev(s);
		s->E[i] = f->lsum_kahan;
		dk = sink_dev(s);
		s->E[i] = 0;
		if (!(dp > budget) || !(dk <= budget)) {
			snprintf(off + strlen(off), sizeof g->sinkoffer - strlen(off),
			    "    %s, a loop's f32 sum s = s + t of %ld passes, errs as "
			    "written by up to %.3g against the exact sum of its f32 "
			    "terms, %.3g at the stored value: %s %.3g a step allows%s"
			    "\n%s", f->lsum_s, f->lsum_n, f->lsum_plain, dp, dp <=
			    budget ? "within the" : "more than the", budget, dp <=
			    budget ? ", so Kahan's sum is not offered" : isfinite(dk) ?
			    "; Kahan's compensated sum may pass it too, so it is not "
			    "offered" : "; Kahan's sum has no bound there", f->lsum_src);
			continue;
		}
		snprintf(off + strlen(off), sizeof g->sinkoffer - strlen(off),
		    "    offered (not applied: it moves the "
		    "stored value toward the exact sum, away from the original's): "
		    "%s, a loop's f32 sum s = s + t of %ld passes, t = %s in [%.4g, "
		    "%.4g], errs as written by up to %.3g against the exact sum of "
		    "its f32 terms (each add rounded, on the partial sums' "
		    "magnitudes), %.3g at the stored value, more than the %.3g a "
		    "step allows; Kahan's compensated sum, y = t - c, w = s + y, c "
		    "= (w - s) - y, s = w, each written precise (NoContraction: no "
		    "driver may fold c to 0), errs by at most %.3g, %.3g at the "
		    "stored value: within it, cost %g a pass against %g (%g more "
		    "over the loop)\n%s", f->lsum_s, f->lsum_n, f->lsum_t,
		    f->lsum_tlo, f->lsum_thi, f->lsum_plain, dp, budget,
		    f->lsum_kahan, dk, f->lsum_kcost, f->lsum_cost, (f->lsum_kcost -
		    f->lsum_cost) * (double)f->lsum_n, f->lsum_src);
	}
}

int
fx_rewrites(struct fxg *g, int root, const struct fxopt *o, struct fxrw *rw,
    int max)
{
	static struct fiv reach[FX_MAXN][FX_MAXIN];
	static char seen[FX_MAXN];
	static int rdep[FX_MAXN];
	static struct plinfo pl[FX_MAXN];
	static unsigned dm[FX_MAXN];
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
	/* each node as a polynomial in one variable, operands first; and the
	 * inputs it depends on */
	for (i = 0; i < nn0; i++) {
		const struct fxn *n = &g->n[i];
		pl_of(g, i, pl);
		dm[i] = n->op == FX_IN || n->op == FX_CVT || n->op == FX_OPQ ?
		    1u << n->in : 0;
		if (n->a >= 0) dm[i] |= dm[n->a];
		if (n->b >= 0) dm[i] |= dm[n->b];
		if (n->c >= 0) dm[i] |= dm[n->c];
		if (n->d >= 0) dm[i] |= dm[n->d];
	}
	/* 3: the other rewrites, on the domain each node is used on */
	for (i = root; i >= 0; i--) {
		const struct fxn *n = &g->n[i];
		struct fxrw *r;
		float k, k2;
		if (!seen[i] || i >= nn0)
			continue;
		/* a polynomial of degree 2 or more, the whole of it (no
		 * operation above it extends it in the same variable) */
		if ((n->op == FX_ADD || n->op == FX_SUB || n->op == FX_MUL ||
		    n->op == FX_FMA || n->op == FX_NEG) && pl[i].x >= 0 &&
		    !pl[i].leaf && pl[i].deg >= 2) {
			int whole = 1;
			for (j = i + 1; j < nn0 && whole; j++) {
				const struct fxn *m = &g->n[j];
				if (seen[j] && pl[j].x == pl[i].x && !pl[j].leaf &&
				    (m->a == i || m->b == i || m->c == i))
					whole = 0;
			}
			if (whole)
				an_fewer(g, reach[i], rw, &nrw, max, i, pl, rdep[i]);
		}
		/* a product whose factors include powers, the whole of it */
		if ((n->op == FX_MUL || n->op == FX_DIV) && ld_whole(g, i, nn0,
		    seen))
			an_logdom(g, reach[i], rw, &nrw, max, i, rdep[i]);
		/* a function of one value, the largest (nothing that reads it
		 * has the same one input) */
		if (dm[i] && !(dm[i] & (dm[i] - 1)) && n->op != FX_IN &&
		    n->op != FX_K && n->op != FX_CVT && n->op != FX_OPQ) {
			int whole = 1, in = 0;
			for (j = i + 1; j < nn0 && whole; j++) {
				const struct fxn *m = &g->n[j];
				if (seen[j] && dm[j] == dm[i] && (m->a == i || m->b == i ||
				    m->c == i || m->d == i))
					whole = 0;
			}
			while (!(dm[i] >> in & 1))
				in++;
			if (whole)
				an_table(g, reach[i], rw, &nrw, max, i, in, rdep[i]);
		}
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
		if ((n->op == FX_EXP2 || n->op == FX_LOG2 || n->op == FX_SQRT ||
		    n->op == FX_ISQRT) && !is_k(g, n->a, NULL))
			an_poly(g, reach[i], rw, &nrw, max, i, o, rdep[i]);
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

	for (i = 0; i < n; i++) {
		free(rw[i].proof.p);
		free(rw[i].tab);
	}
}
