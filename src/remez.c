/*
 * Numerics for approx: a long double reference evaluator with Taylor
 * jets, Chebyshev interpolation and economisation, Pade, and the Remez
 * exchange for weighted polynomial and rational minimax.
 *
 * A fit approximates f(x) by t^k0 * N(v) / D(v) with t = x - c and
 * v = t^step.  Odd functions use k0 = 1, step = 2 (x * P(x^2)), even ones
 * k0 = 0, step = 2.  Exact points are folded in by writing
 *
 *	N(v) = L(v) + Z(v) Q(v)
 *
 * where L interpolates f at the exact points and Z vanishes there, so the
 * free part Q solves an unconstrained weighted problem: approximate
 * G = (f - t^k0 L) / (t^k0 Z) with weight W = w |t^k0 Z|.  Q (and the
 * denominator) live in the Chebyshev basis of z, the affine image of v on
 * [-1, 1]; only the final result is converted to monomials.
 */
#include "approx.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- reference evaluator ---- */

static int
lsupported(int op)
{
	switch (op) {
	case O_CONST: case O_VAR: case O_NEG: case O_ABS: case O_SIGN:
	case O_SQRT: case O_EXP: case O_LOG: case O_SIN: case O_COS:
	case O_TAN: case O_FLOOR: case O_CEIL: case O_CAST: case O_ADD:
	case O_SUB: case O_MUL: case O_DIV: case O_POW: case O_MIN:
	case O_MAX: case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT:
	case O_GE: case O_LAND: case O_LOR: case O_LNOT: case O_SEL:
	case O_BARRIER: case O_SEC: case O_CSC: case O_COT: case O_ATAN:
	case O_COPYSIGN: case O_FMA: case O_LAMBERTW: case O_LAMBERTWM1:
		return 1;
	}
	return 0;
}

void
lref_init(struct lref *r, nid body, int slot)
{
	uint8_t *mark = xcalloc(N.n, 1);
	uint32_t *map = xcalloc(N.n, sizeof *map);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0, k;

	memset(r, 0, sizeof *r);
	r->slot = slot;
	r->full = 1;
	st[sp++] = body;
	mark[body] = 1;
	while (sp) {
		g = st[--sp];
		if (!lsupported(N.op[g]) ||
		    (N.op[g] == O_VAR && N.ki[g] != slot))
			r->full = 0;
		if (N.op[g] == O_CONST || N.op[g] == O_VAR ||
		    N.op[g] == O_UNDEF || N.op[g] == O_TABLE)
			continue;
		if (N.a[g] && !mark[N.a[g]])
			mark[N.a[g]] = 1, st[sp++] = N.a[g];
		if (N.b[g] && !mark[N.b[g]])
			mark[N.b[g]] = 1, st[sp++] = N.b[g];
		if (N.c[g] && !mark[N.c[g]])
			mark[N.c[g]] = 1, st[sp++] = N.c[g];
	}
	if (r->full) {
		for (g = 1; g < N.n; g++)
			if (mark[g])
				r->n++;
		r->op = xmalloc(r->n);
		r->a = xcalloc(r->n, sizeof *r->a);
		r->b = xcalloc(r->n, sizeof *r->b);
		r->c = xcalloc(r->n, sizeof *r->c);
		r->k = xcalloc(r->n, sizeof *r->k);
		r->v = xcalloc(r->n, sizeof *r->v);
		for (g = 1, k = 0; g < N.n; g++) {
			if (!mark[g])
				continue;
			map[g] = k;
			r->op[k] = N.op[g];
			r->k[k] = TY_FP(N.ty[g]) ? (ld)N.kr[g] : (ld)N.ki[g];
			if (N.op[g] != O_CONST && N.op[g] != O_VAR) {
				r->a[k] = N.a[g] ? map[N.a[g]] : 0;
				r->b[k] = N.b[g] ? map[N.b[g]] : 0;
				r->c[k] = N.c[g] ? map[N.c[g]] : 0;
			}
			k++;
		}
		r->root = map[body];
	} else {
		nid roots[1];
		roots[0] = body;
		plan_build(&r->dp, roots, 1, 1);
		r->root = r->dp.root[0];
		r->in_i = xcalloc((size_t)MAXP * CH, sizeof *r->in_i);
		r->in_r = xcalloc((size_t)MAXP * CH, sizeof *r->in_r);
	}
	free(mark);
	free(map);
	free(st);
}

void
lref_free(struct lref *r)
{
	if (r->full) {
		free(r->op);
		free(r->a);
		free(r->b);
		free(r->c);
		free(r->k);
		free(r->v);
	} else {
		plan_free(&r->dp);
		free(r->in_i);
		free(r->in_r);
	}
	memset(r, 0, sizeof *r);
}

ld
lref_eval(struct lref *r, ld x)
{
	uint32_t k;
	ld *v = r->v, a, b;

	if (!r->full) {
		r->in_r[(size_t)r->slot * CH] = (double)x;
		plan_run(&r->dp, 1, r->in_i, r->in_r);
		if (COLF(&r->dp, r->root)[0] & F_UNDEF)
			return NAN;
		return TY_FP(r->dp.ty[r->root]) ? COLR(&r->dp, r->root)[0] :
		    (ld)COLI(&r->dp, r->root)[0];
	}
	for (k = 0; k < r->n; k++) {
		a = v[r->a[k]];
		b = v[r->b[k]];
		switch (r->op[k]) {
		case O_CONST: v[k] = r->k[k]; break;
		case O_VAR: v[k] = x; break;
		case O_NEG: v[k] = -a; break;
		case O_ABS: v[k] = fabsl(a); break;
		case O_SIGN: v[k] = (a > 0) - (a < 0); break;
		case O_SQRT: v[k] = a < 0 ? NAN : sqrtl(a); break;
		case O_EXP: v[k] = expl(a); break;
		case O_LOG: v[k] = a <= 0 ? NAN : logl(a); break;
		case O_SIN: v[k] = sinl(a); break;
		case O_COS: v[k] = cosl(a); break;
		case O_TAN: v[k] = tanl(a); break;
		case O_SEC: v[k] = 1 / cosl(a); break;
		case O_CSC: v[k] = 1 / sinl(a); break;
		case O_COT: v[k] = cosl(a) / sinl(a); break;
		case O_ATAN: v[k] = atanl(a); break;
		case O_LAMBERTW: v[k] = lambert_w((double)a, 0); break;
		case O_LAMBERTWM1: v[k] = lambert_w((double)a, 1); break;
		case O_FLOOR: v[k] = floorl(a); break;
		case O_CEIL: v[k] = ceill(a); break;
		case O_CAST: v[k] = a; break;
		case O_BARRIER: v[k] = a; break;
		case O_ADD: v[k] = a + b; break;
		case O_SUB: v[k] = a - b; break;
		case O_MUL: v[k] = a * b; break;
		case O_DIV: v[k] = b == 0 ? NAN : a / b; break;
		case O_POW:
			if ((a < 0 && b != floorl(b)) || (a == 0 && b < 0))
				v[k] = NAN;
			else
				v[k] = powl(a, b);
			break;
		case O_MIN: v[k] = a < b ? a : b; break;
		case O_MAX: v[k] = a > b ? a : b; break;
		case O_EQ: v[k] = a == b; break;
		case O_NE: v[k] = a != b; break;
		case O_LT: v[k] = a < b; break;
		case O_LE: v[k] = a <= b; break;
		case O_GT: v[k] = a > b; break;
		case O_GE: v[k] = a >= b; break;
		case O_LAND: v[k] = a != 0 && b != 0; break;
		case O_LOR: v[k] = a != 0 || b != 0; break;
		case O_LNOT: v[k] = a == 0; break;
		case O_SEL: v[k] = a != 0 ? b : v[r->c[k]]; break;
		case O_COPYSIGN: v[k] = copysignl(a, b); break;
		case O_FMA: v[k] = a * b + v[r->c[k]]; break;
		default: v[k] = NAN;
		}
	}
	return v[r->root];
}

ld
lref_fn(void *ctx, ld x)
{
	return lref_eval(ctx, x);
}

/* ---- Taylor jets ---- */

static void
jmul(const ld *a, const ld *b, ld *c, int n)
{
	int k, i;

	for (k = 0; k < n; k++) {
		ld s = 0;
		for (i = 0; i <= k; i++)
			s += a[i] * b[k - i];
		c[k] = s;
	}
}

static int
jdiv(const ld *a, const ld *b, ld *c, int n)
{
	int k, i;

	if (b[0] == 0)
		return 0;
	for (k = 0; k < n; k++) {
		ld s = a[k];
		for (i = 1; i <= k; i++)
			s -= b[i] * c[k - i];
		c[k] = s / b[0];
	}
	return 1;
}

static void
jsincos(const ld *a, ld *s, ld *c, int n)
{
	int k, i;

	s[0] = sinl(a[0]);
	c[0] = cosl(a[0]);
	for (k = 1; k < n; k++) {
		ld ss = 0, cc = 0;
		for (i = 1; i <= k; i++) {
			ss += i * a[i] * c[k - i];
			cc += i * a[i] * s[k - i];
		}
		s[k] = ss / k;
		c[k] = -cc / k;
	}
}

/*
 * Taylor coefficients of the body at c, a[0..n-1].  Returns 0 when an
 * operation has no series rule there (bit operations, a kink at c).
 */
int
lref_taylor(struct lref *r, ld c, int n, ld *out)
{
	ld *v, *t1, *t2, *t3;
	uint32_t k;
	int i, j, ok = 1;

	if (!r->full || n > AMAXJ || n < 1)
		return 0;
	v = xcalloc((size_t)r->n * AMAXJ, sizeof *v);
	t1 = xcalloc(AMAXJ, sizeof *t1);
	t2 = xcalloc(AMAXJ, sizeof *t2);
	t3 = xcalloc(AMAXJ, sizeof *t3);
#define S(q) (v + (size_t)(q) * AMAXJ)
	for (k = 0; k < r->n && ok; k++) {
		ld *o = S(k), *a = S(r->a[k]), *b = S(r->b[k]);
		switch (r->op[k]) {
		case O_CONST:
			o[0] = r->k[k];
			break;
		case O_VAR:
			o[0] = c;
			if (n > 1)
				o[1] = 1;
			break;
		case O_CAST:
		case O_BARRIER:
			memcpy(o, a, n * sizeof *o);
			break;
		case O_NEG:
			for (i = 0; i < n; i++)
				o[i] = -a[i];
			break;
		case O_ADD:
		case O_SUB:
			for (i = 0; i < n; i++)
				o[i] = r->op[k] == O_ADD ? a[i] + b[i] : a[i] - b[i];
			break;
		case O_MUL:
			jmul(a, b, o, n);
			break;
		case O_FMA:
			jmul(a, b, o, n);
			for (i = 0; i < n; i++)
				o[i] += S(r->c[k])[i];
			break;
		case O_DIV:
			ok = jdiv(a, b, o, n);
			break;
		case O_EXP:
			o[0] = expl(a[0]);
			for (j = 1; j < n; j++) {
				ld s = 0;
				for (i = 1; i <= j; i++)
					s += i * a[i] * o[j - i];
				o[j] = s / j;
			}
			break;
		case O_LOG:
			if (a[0] <= 0) {
				ok = 0;
				break;
			}
			o[0] = logl(a[0]);
			for (j = 1; j < n; j++) {
				ld s = 0;
				for (i = 1; i < j; i++)
					s += i * o[i] * a[j - i];
				o[j] = (a[j] - s / j) / a[0];
			}
			break;
		case O_SQRT:
			if (a[0] <= 0) {
				ok = 0;
				break;
			}
			o[0] = sqrtl(a[0]);
			for (j = 1; j < n; j++) {
				ld s = a[j];
				for (i = 1; i < j; i++)
					s -= o[i] * o[j - i];
				o[j] = s / (2 * o[0]);
			}
			break;
		case O_SIN:
		case O_COS:
		case O_TAN:
		case O_SEC:
		case O_CSC:
		case O_COT:
			jsincos(a, t1, t2, n);
			if (r->op[k] == O_SIN)
				memcpy(o, t1, n * sizeof *o);
			else if (r->op[k] == O_COS)
				memcpy(o, t2, n * sizeof *o);
			else if (r->op[k] == O_TAN)
				ok = jdiv(t1, t2, o, n);
			else if (r->op[k] == O_COT)
				ok = jdiv(t2, t1, o, n);
			else {
				memset(t3, 0, sizeof(ld) * AMAXJ);
				t3[0] = 1;
				ok = jdiv(t3, r->op[k] == O_SEC ? t2 : t1, o, n);
			}
			break;
		case O_ATAN:
			/* atan' = a' / (1 + a^2) */
			jmul(a, a, t1, n);
			t1[0] += 1;
			for (i = 0; i + 1 < n; i++)
				t2[i] = (i + 1) * a[i + 1];
			t2[n - 1] = 0;
			ok = jdiv(t2, t1, t3, n);
			o[0] = atanl(a[0]);
			for (i = 1; i < n; i++)
				o[i] = t3[i - 1] / i;
			break;
		case O_POW:
			if (r->b[k] && r->op[r->b[k]] == O_CONST &&
			    b[0] == floorl(b[0]) && b[0] >= 0 && b[0] <= 64) {
				int e = (int)b[0];
				memset(o, 0, n * sizeof *o);
				o[0] = 1;
				for (j = 0; j < e; j++) {
					jmul(o, a, t1, n);
					memcpy(o, t1, n * sizeof *o);
				}
			} else if (r->b[k] && r->op[r->b[k]] == O_CONST &&
			    a[0] > 0) {
				ld p = b[0];
				o[0] = powl(a[0], p);
				for (j = 1; j < n; j++) {
					ld s = 0;
					for (i = 1; i <= j; i++)
						s += ((p + 1) * i - j) * a[i] *
						    o[j - i];
					o[j] = s / (j * a[0]);
				}
			} else
				ok = 0;
			break;
		case O_ABS:
		case O_COPYSIGN: {
			ld sg = r->op[k] == O_ABS ? 1 : (signbit(b[0]) ? -1 : 1);
			if (a[0] == 0) {
				ok = 0;
				break;
			}
			if (a[0] < 0)
				sg = -sg;
			if (r->op[k] == O_COPYSIGN)
				sg = (signbit(b[0]) ? -1 : 1) *
				    (a[0] < 0 ? -1 : 1);
			for (i = 0; i < n; i++)
				o[i] = sg * a[i];
			break;
		}
		case O_SIGN:
		case O_FLOOR:
		case O_CEIL:
			if (r->op[k] == O_SIGN)
				o[0] = a[0] == 0 ? (ok = 0) : (a[0] > 0 ? 1 : -1);
			else {
				if (a[0] == floorl(a[0]))
					ok = 0;
				o[0] = r->op[k] == O_FLOOR ? floorl(a[0]) :
				    ceill(a[0]);
			}
			break;
		case O_MIN:
		case O_MAX:
			if (a[0] == b[0]) {
				ok = 0;
				break;
			}
			memcpy(o, (a[0] < b[0]) == (r->op[k] == O_MIN) ? a : b,
			    n * sizeof *o);
			break;
		case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT:
		case O_GE: case O_LAND: case O_LOR: case O_LNOT:
			/* constant near c unless c sits on the boundary */
			{
				ld x = a[0], y = b[0];
				switch (r->op[k]) {
				case O_EQ: o[0] = x == y; ok = x != y; break;
				case O_NE: o[0] = x != y; ok = x != y; break;
				case O_LT: o[0] = x < y; ok = x != y; break;
				case O_LE: o[0] = x <= y; ok = x != y; break;
				case O_GT: o[0] = x > y; ok = x != y; break;
				case O_GE: o[0] = x >= y; ok = x != y; break;
				case O_LAND: o[0] = x != 0 && y != 0; break;
				case O_LOR: o[0] = x != 0 || y != 0; break;
				case O_LNOT: o[0] = x == 0; break;
				}
			}
			break;
		case O_SEL:
			memcpy(o, a[0] != 0 ? b : S(r->c[k]), n * sizeof *o);
			break;
		default:
			ok = 0;
		}
	}
	if (ok) {
		memcpy(out, S(r->root), n * sizeof *out);
		for (i = 0; i < n; i++)
			if (!isfinite(out[i]))
				ok = 0;
	}
#undef S
	free(v);
	free(t1);
	free(t2);
	free(t3);
	return ok;
}

/* ---- dense linear algebra ---- */

/* solve A x = b (n x n, row major) by partial pivoting; 0 if singular */
static int
lsolve(ld *A, ld *b, int n)
{
	int i, j, k, p;
	ld m, t;

	for (k = 0; k < n; k++) {
		p = k;
		for (i = k + 1; i < n; i++)
			if (fabsl(A[i * n + k]) > fabsl(A[p * n + k]))
				p = i;
		if (A[p * n + k] == 0 || !isfinite(A[p * n + k]))
			return 0;
		if (p != k) {
			for (j = 0; j < n; j++) {
				t = A[k * n + j];
				A[k * n + j] = A[p * n + j];
				A[p * n + j] = t;
			}
			t = b[k];
			b[k] = b[p];
			b[p] = t;
		}
		for (i = k + 1; i < n; i++) {
			m = A[i * n + k] / A[k * n + k];
			if (m == 0)
				continue;
			for (j = k; j < n; j++)
				A[i * n + j] -= m * A[k * n + j];
			b[i] -= m * b[k];
		}
	}
	for (k = n - 1; k >= 0; k--) {
		t = b[k];
		for (j = k + 1; j < n; j++)
			t -= A[k * n + j] * b[j];
		b[k] = t / A[k * n + k];
	}
	for (k = 0; k < n; k++)
		if (!isfinite(b[k]))
			return 0;
	return 1;
}

/* ---- polynomials (monomial, ascending) ---- */

static void
pmul(const ld *a, int na, const ld *b, int nb, ld *c)
{
	int i, j;

	for (i = 0; i < na + nb - 1; i++)
		c[i] = 0;
	for (i = 0; i < na; i++)
		for (j = 0; j < nb; j++)
			c[i + j] += a[i] * b[j];
}

static ld
peval(const ld *a, int n, ld x)
{
	ld s = 0;

	while (n-- > 0)
		s = s * x + a[n];
	return s;
}

/* Chebyshev series in z -> monomials in V, where z = al V + be */
static void
cheb2mono(const ld *ch, int n, ld al, ld be, ld *out)
{
	ld tp[AMAXC + 2], tc[AMAXC + 2], tn[AMAXC + 2];
	int j, i;

	memset(out, 0, n * sizeof *out);
	memset(tp, 0, sizeof tp);
	memset(tc, 0, sizeof tc);
	tp[0] = 1;			/* T0 */
	tc[0] = be;			/* T1 = al V + be */
	tc[1] = al;
	if (n > 0)
		out[0] += ch[0];
	if (n > 1)
		for (i = 0; i < 2; i++)
			out[i] += ch[1] * tc[i];
	for (j = 2; j < n; j++) {
		/* T_j = 2 z T_{j-1} - T_{j-2} */
		memset(tn, 0, sizeof tn);
		for (i = 0; i < j; i++) {
			tn[i] += 2 * be * tc[i];
			tn[i + 1] += 2 * al * tc[i];
		}
		for (i = 0; i < j; i++)
			tn[i] -= tp[i];
		for (i = 0; i <= j; i++)
			out[i] += ch[j] * tn[i];
		memcpy(tp, tc, sizeof tp);
		memcpy(tc, tn, sizeof tc);
	}
}

/* monomials in V -> Chebyshev series in z, V = (z - be) / al */
static void
mono2cheb(const ld *m, int n, ld al, ld be, ld *out)
{
	ld acc[AMAXJ + 2], tmp[AMAXJ + 2];
	ld A = 1 / al, B = -be / al;	/* V = A z + B */
	int i, j, deg = 0;

	memset(acc, 0, sizeof acc);
	/* Horner in the Chebyshev basis: acc = acc * V + m[i] */
	for (i = n - 1; i >= 0; i--) {
		memset(tmp, 0, sizeof tmp);
		for (j = 0; j <= deg; j++) {
			/* z T_j = (T_{j+1} + T_{|j-1|}) / 2, z T_0 = T_1 */
			if (j == 0)
				tmp[1] += A * acc[0];
			else {
				tmp[j + 1] += A * acc[j] / 2;
				tmp[j - 1] += A * acc[j] / 2;
			}
			tmp[j] += B * acc[j];
		}
		tmp[0] += m[i];
		memcpy(acc, tmp, sizeof acc);
		if (i < n - 1)
			deg++;
	}
	memcpy(out, acc, (size_t)n * sizeof *out);
}

static ld
clenshaw(const ld *c, int n, ld z)
{
	ld b1 = 0, b2 = 0, t;
	int j;

	if (n == 0)
		return 0;
	for (j = n - 1; j >= 1; j--) {
		t = 2 * z * b1 - b2 + c[j];
		b2 = b1;
		b1 = t;
	}
	return z * b1 - b2 + c[0];
}

/* ---- the transformed problem ---- */

struct fctx {
	const struct fitprob *pb;
	ld s, Va, Vb;
	int nk;
	ld vk[AMAXK];
	ld lm[AMAXK];		/* L in monomials of V */
	ld zm[AMAXK + 1];	/* Z in monomials of V */
	int nq, nd;		/* free numerator terms, denominator terms */
	ld q[AMAXC], d[AMAXC];	/* Chebyshev coefficients (d[0] = 1) */
	ld E;
};

static ld
powi(ld x, int k)
{
	ld r = 1;

	while (k-- > 0)
		r *= x;
	return r;
}

static ld
x_of_z(const struct fctx *c, ld z)
{
	ld V = c->Va + (z + 1) * (c->Vb - c->Va) / 2, u;

	if (z <= -1)
		V = c->Va;
	if (z >= 1)
		V = c->Vb;
	u = c->pb->step == 2 ? sqrtl(V > 0 ? V : 0) : V;
	return c->pb->c + c->s * u;
}

static ld
z_of_x(const struct fctx *c, ld x)
{
	ld u = (x - c->pb->c) / c->s;
	ld V = c->pb->step == 2 ? u * u : u;

	return 2 * (V - c->Va) / (c->Vb - c->Va) - 1;
}

/* G and W at x; W = 0 where the error vanishes by construction */
static void
transformed(const struct fctx *c, ld x, ld *G, ld *W)
{
	const struct fitprob *pb = c->pb;
	ld u = (x - pb->c) / c->s, V = pb->step == 2 ? u * u : u;
	ld f = pb->f(pb->fctx, x), D, w, uk = powi(u, pb->k0);

	D = uk * peval(c->zm, c->nk + 1, V);
	if (!isfinite(f) || fabsl(D) < 1e-30L) {
		*G = 0;
		*W = 0;
		return;
	}
	*G = (f - uk * peval(c->lm, c->nk, V)) / D;
	w = pb->rel ? 1 / (fabsl(f) > 1e-300L ? fabsl(f) : 1e-300L) : 1;
	if (pb->w)
		w *= pb->w(pb->wctx, x);
	*W = w * fabsl(D);
}

static ld
rq(const struct fctx *c, ld z)
{
	ld p = clenshaw(c->q, c->nq, z);

	if (c->nd > 0)
		p /= 1 + clenshaw(c->d, c->nd + 1, z) - c->d[0];
	return p;
}

/* signed weighted error at z */
static ld
werr(const struct fctx *c, ld z, ld *xo)
{
	ld x = x_of_z(c, z), G, W;

	transformed(c, x, &G, &W);
	if (xo)
		*xo = x;
	if (W == 0)
		return 0;
	return W * (G - rq(c, z));
}

static int
setup(struct fctx *c, const struct fitprob *pb)
{
	ld ua, ub, A[AMAXK * AMAXK], bb[AMAXK];
	int i, j;

	memset(c, 0, sizeof *c);
	c->pb = pb;
	c->s = fmaxl(fabsl(pb->xa - pb->c), fabsl(pb->xb - pb->c));
	if (!(c->s > 0))
		return 0;
	ua = (pb->xa - pb->c) / c->s;
	ub = (pb->xb - pb->c) / c->s;
	if (pb->step == 2 && ua < 0)
		return 0;
	c->Va = pb->step == 2 ? ua * ua : ua;
	c->Vb = pb->step == 2 ? ub * ub : ub;
	/* exact points, deduplicated in V */
	for (i = 0; i < pb->nkeep; i++) {
		ld u = (pb->keep[i] - pb->c) / c->s, V, f;
		V = pb->step == 2 ? u * u : u;
		f = pb->f(pb->fctx, pb->keep[i]);
		if (u == 0 && pb->k0 > 0)
			continue;	/* implied: t^k0 vanishes there */
		for (j = 0; j < c->nk; j++)
			if (fabsl(c->vk[j] - V) < 1e-30L)
				break;
		if (j < c->nk)
			continue;
		if (c->nk == AMAXK || !isfinite(f))
			return 0;
		c->vk[c->nk] = V;
		bb[c->nk] = f / powi(u, pb->k0);
		c->nk++;
	}
	/* L: Vandermonde solve; Z = prod (V - vk) */
	for (i = 0; i < c->nk; i++)
		for (j = 0; j < c->nk; j++)
			A[i * c->nk + j] = powi(c->vk[i], j);
	if (c->nk && !lsolve(A, bb, c->nk))
		return 0;
	memcpy(c->lm, bb, c->nk * sizeof *bb);
	memset(c->zm, 0, sizeof c->zm);
	c->zm[0] = 1;
	for (i = 0; i < c->nk; i++) {
		ld t[AMAXK + 1];
		memset(t, 0, sizeof t);
		for (j = 0; j <= i; j++) {
			t[j + 1] += c->zm[j];
			t[j] -= c->vk[i] * c->zm[j];
		}
		memcpy(c->zm, t, sizeof t);
	}
	c->nq = pb->nt - c->nk;
	c->nd = pb->nd;
	if (c->nq < 0)
		return 0;
	return 1;
}

/* the result in monomials of t */
static void
finish(struct fctx *c, struct fitres *r)
{
	const struct fitprob *pb = c->pb;
	ld al = 2 / (c->Vb - c->Va), be = -2 * c->Va / (c->Vb - c->Va) - 1;
	ld qm[AMAXC], dm[AMAXC], zq[2 * AMAXC], ld_[2 * AMAXC];
	int i, nn;

	memset(qm, 0, sizeof qm);
	memset(dm, 0, sizeof dm);
	cheb2mono(c->q, c->nq, al, be, qm);
	if (c->nd > 0)
		cheb2mono(c->d, c->nd + 1, al, be, dm);
	else
		dm[0] = 1;
	/* N = L D + Z Q */
	memset(zq, 0, sizeof zq);
	memset(ld_, 0, sizeof ld_);
	if (c->nq > 0)
		pmul(c->zm, c->nk + 1, qm, c->nq, zq);
	if (c->nk > 0)
		pmul(c->lm, c->nk, dm, c->nd + 1, ld_);
	nn = c->nk + c->nq;
	if (c->nk + c->nd > nn)
		nn = c->nk + c->nd;
	if (nn > AMAXC)
		nn = AMAXC;
	r->nn = nn;
	r->nd = c->nd + 1;
	for (i = 0; i < nn; i++)
		r->num[i] = zq[i] + ld_[i];
	for (i = 0; i <= c->nd; i++)
		r->den[i] = dm[i];
	/* normalise D(0) = 1, then unscale u = t / s */
	if (r->den[0] != 0 && r->den[0] != 1) {
		ld d0 = r->den[0];
		for (i = 0; i < r->nn; i++)
			r->num[i] /= d0;
		for (i = 0; i < r->nd; i++)
			r->den[i] /= d0;
	}
	for (i = 0; i < r->nn; i++)
		r->num[i] /= powi(c->s, pb->k0 + pb->step * i);
	for (i = 0; i < r->nd; i++)
		r->den[i] /= powi(c->s, pb->step * i);
}

ld
fit_eval(const struct fitprob *pb, const struct fitres *r, ld x)
{
	ld t = x - pb->c, v = pb->step == 2 ? t * t : t;
	ld n = peval(r->num, r->nn, v), d = peval(r->den, r->nd, v);

	return powi(t, pb->k0) * n / d;
}

void
fit_measure(const struct fitprob *pb, struct fitres *r)
{
	const int M = 6000;
	int g;

	r->err = r->eabs = r->erel = 0;
	r->xworst = pb->xa;
	for (g = 0; g < M; g++) {
		ld z = -cosl(3.14159265358979323846264338327950288L * g /
		    (M - 1)), x, f, p, ae, re, we;
		ld V, u;
		/* uniform in the native variable, Chebyshev-spaced */
		x = pb->xa + (z + 1) * (pb->xb - pb->xa) / 2;
		if (pb->step == 2 && pb->xa >= 0) {
			ld ua = pb->xa, ub = pb->xb;
			V = ua * ua + (z + 1) * (ub * ub - ua * ua) / 2;
			u = sqrtl(V > 0 ? V : 0);
			x = u;
		}
		f = pb->f(pb->fctx, x);
		if (!isfinite(f))
			continue;
		p = fit_eval(pb, r, x);
		ae = fabsl(f - p);
		re = f != 0 ? ae / fabsl(f) : (ae == 0 ? 0 : ae);
		we = pb->rel ? re : ae;
		if (pb->w)
			we *= pb->w(pb->wctx, x);
		if (!isfinite(p))
			ae = re = we = INFINITY;
		if (ae > r->eabs)
			r->eabs = ae;
		if (re > r->erel)
			r->erel = re;
		if (we > r->err) {
			r->err = we;
			r->xworst = x;
		}
	}
}

/* ---- Chebyshev interpolation and economisation ---- */

static int
fit_cheb(struct fctx *c)
{
	ld G[AMAXC], W;
	int k, j, n = c->nq;
	const ld PI = 3.14159265358979323846264338327950288L;

	for (k = 0; k < n; k++) {
		ld z = cosl(PI * (k + 0.5L) / n);
		transformed(c, x_of_z(c, z), &G[k], &W);
		if (W == 0)
			return 0;
	}
	for (j = 0; j < n; j++) {
		ld s = 0;
		for (k = 0; k < n; k++)
			s += G[k] * cosl(PI * j * (k + 0.5L) / n);
		c->q[j] = 2 * s / n * (j == 0 ? 0.5L : 1);
	}
	return 1;
}

/* the reduced series R(V) = sum r_j V^j (in scaled u) from f's Taylor */
static int
reduced_series(const struct fctx *c, ld *rs, int *nr)
{
	const struct fitprob *pb = c->pb;
	int j, n = 0;

	for (j = 0; pb->k0 + pb->step * j < pb->ntaylor && j < AMAXJ; j++)
		rs[n++] = pb->taylor[pb->k0 + pb->step * j] *
		    powi(c->s, pb->k0 + pb->step * j);
	*nr = n;
	return n > 0;
}

static int
fit_econ(struct fctx *c, const char **why)
{
	ld rs[AMAXJ], ch[AMAXJ];
	ld al = 2 / (c->Vb - c->Va), be = -2 * c->Va / (c->Vb - c->Va) - 1;
	int nr, j;

	if (c->nk > 0 || c->nd > 0 || !c->pb->taylor) {
		*why = "needs the Taylor series and no exact points";
		return 0;
	}
	if (!reduced_series(c, rs, &nr) || nr < c->nq + 4) {
		*why = "Taylor series too short";
		return 0;
	}
	/* the tail must shrink, or the series does not converge here */
	{
		ld lv = fmaxl(fabsl(c->Va), fabsl(c->Vb));
		ld head = 0, tail = 0;
		for (j = 0; j < nr; j++) {
			ld t = fabsl(rs[j]) * powl(lv, j);
			if (j < nr / 2)
				head = fmaxl(head, t);
			else
				tail = fmaxl(tail, t);
		}
		if (!(tail < 1e-3L * head)) {
			*why = "Taylor series does not converge on the interval";
			return 0;
		}
	}
	mono2cheb(rs, nr, al, be, ch);
	for (j = 0; j < c->nq; j++)
		c->q[j] = ch[j];
	return 1;
}

/* Pade [n-1/m] of the reduced series, straight to monomials */
static int
fit_pade(struct fctx *c, struct fitres *r, const char **why)
{
	ld rs[AMAXJ], A[AMAXC * AMAXC], b[AMAXC], q[AMAXC + 1];
	int nr, n = c->nq, m = c->nd, i, k;
	const struct fitprob *pb = c->pb;

	if (c->nk > 0 || !pb->taylor) {
		*why = "needs the Taylor series and no exact points";
		return 0;
	}
	if (!reduced_series(c, rs, &nr) || nr < n + m) {
		*why = "Taylor series too short";
		return 0;
	}
	/* sum_{k=0..m} q_k rs[n-1+i-k] = 0, i = 1..m, q_0 = 1 */
	for (i = 1; i <= m; i++) {
		for (k = 1; k <= m; k++) {
			int idx = n - 1 + i - k;
			A[(i - 1) * m + (k - 1)] = idx >= 0 ? rs[idx] : 0;
		}
		b[i - 1] = -rs[n - 1 + i];
	}
	if (m > 0 && !lsolve(A, b, m)) {
		*why = "singular Pade system";
		return 0;
	}
	q[0] = 1;
	for (k = 1; k <= m; k++)
		q[k] = b[k - 1];
	r->nn = n;
	r->nd = m + 1;
	for (i = 0; i < n; i++) {
		ld s = 0;
		for (k = 0; k <= m && k <= i; k++)
			s += q[k] * rs[i - k];
		r->num[i] = s / powi(c->s, pb->k0 + pb->step * i);
	}
	for (k = 0; k <= m; k++)
		r->den[k] = q[k] / powi(c->s, pb->step * k);
	return 1;
}

/* ---- Remez exchange ---- */

#define MAXREF (2 * AMAXC + 2)

static int
solve_ref(struct fctx *c, const ld *zr, int npt)
{
	ld A[MAXREF * MAXREF], b[MAXREF], G[MAXREF], W[MAXREF];
	ld Eold = 0, sgn;
	int n = c->nq, m = c->nd, N = n + m + 1, i, j, it;

	if (npt != N)
		return 0;
	for (i = 0; i < N; i++) {
		transformed(c, x_of_z(c, zr[i]), &G[i], &W[i]);
		if (W[i] == 0)
			return 0;
	}
	for (it = 0; it < (m ? 16 : 1); it++) {
		for (i = 0; i < N; i++) {
			ld T[MAXREF + 1];
			int nT = n > m + 1 ? n : m + 1;
			sgn = (i & 1) ? -1 : 1;
			T[0] = 1;
			T[1] = zr[i];
			for (j = 2; j < nT; j++)
				T[j] = 2 * zr[i] * T[j - 1] - T[j - 2];
			for (j = 0; j < n; j++)
				A[i * N + j] = T[j];
			for (j = 1; j <= m; j++)
				A[i * N + n + j - 1] = -T[j] *
				    (G[i] - sgn * Eold / W[i]);
			A[i * N + N - 1] = sgn / W[i];
			b[i] = G[i];
		}
		if (!lsolve(A, b, N))
			return 0;
		for (j = 0; j < n; j++)
			c->q[j] = b[j];
		c->d[0] = 1;
		for (j = 1; j <= m; j++)
			c->d[j] = b[n + j - 1];
		c->E = b[N - 1];
		if (m && fabsl(c->E - Eold) <= 1e-12L * fabsl(c->E))
			break;
		Eold = c->E;
	}
	return 1;
}

/* golden-section refinement of a local maximum of |e| near z */
static ld
refine_max(const struct fctx *c, ld lo, ld hi, ld *ez)
{
	const ld g = 0.6180339887498948482L;
	ld a = lo, b = hi, x1 = b - g * (b - a), x2 = a + g * (b - a);
	ld f1 = fabsl(werr(c, x1, NULL)), f2 = fabsl(werr(c, x2, NULL));
	int i;

	for (i = 0; i < 30; i++) {
		if (f1 > f2) {
			b = x2;
			x2 = x1;
			f2 = f1;
			x1 = b - g * (b - a);
			f1 = fabsl(werr(c, x1, NULL));
		} else {
			a = x1;
			x1 = x2;
			f1 = f2;
			x2 = a + g * (b - a);
			f2 = fabsl(werr(c, x2, NULL));
		}
	}
	*ez = werr(c, (a + b) / 2, NULL);
	return (a + b) / 2;
}

/* local extrema of the error with alternating signs, in order */
static int
extrema(const struct fctx *c, ld *ze, ld *ee, int max, ld *gmax)
{
	enum { M = 3000 };
	static ld zg[M], eg[M];
	const ld PI = 3.14159265358979323846264338327950288L;
	int g, ne = 0, best, start;

	*gmax = 0;
	for (g = 0; g < M; g++) {
		zg[g] = -cosl(PI * g / (M - 1));
		eg[g] = werr(c, zg[g], NULL);
		if (!isfinite(eg[g]))
			return -1;
		if (fabsl(eg[g]) > *gmax)
			*gmax = fabsl(eg[g]);
	}
	g = 0;
	while (g < M) {
		int sg;
		while (g < M && eg[g] == 0)
			g++;
		if (g == M)
			break;
		sg = eg[g] > 0;
		start = g;
		best = g;
		while (g < M && eg[g] != 0 && (eg[g] > 0) == sg) {
			if (fabsl(eg[g]) > fabsl(eg[best]))
				best = g;
			g++;
		}
		(void)start;
		{
			ld zz = zg[best], ez = eg[best];
			if (best > 0 && best < M - 1) {
				ld e2;
				ld z2 = refine_max(c, zg[best - 1], zg[best + 1], &e2);
				if (fabsl(e2) > fabsl(ez) && (e2 > 0) == sg)
					zz = z2, ez = e2;
			}
			if (fabsl(ez) > *gmax)
				*gmax = fabsl(ez);
			/* merge with a same-signed neighbour across a zero */
			if (ne > 0 && (ee[ne - 1] > 0) == sg) {
				if (fabsl(ez) > fabsl(ee[ne - 1])) {
					ze[ne - 1] = zz;
					ee[ne - 1] = ez;
				}
			} else if (ne < max) {
				ze[ne] = zz;
				ee[ne] = ez;
				ne++;
			}
		}
	}
	return ne;
}

static int
fit_remez(struct fctx *c, struct fitres *r, const char **why)
{
	int N = c->nq + c->nd + 1, i, it, ne, k, bestit = 0;
	ld zr[MAXREF], ze[4 * MAXREF], ee[4 * MAXREF], gmax, best = INFINITY;
	ld bq[AMAXC], bd[AMAXC];
	const ld PI = 3.14159265358979323846264338327950288L;

	if (N > MAXREF - 1) {
		*why = "too many coefficients";
		return 0;
	}
	/* Chebyshev extrema, nudged off points where the error vanishes */
	for (i = 0; i < N; i++) {
		ld G, W, z = -cosl(PI * i / (N - 1 > 0 ? N - 1 : 1));
		if (N == 1)
			z = 0;
		for (k = 0; k < 20; k++) {
			transformed(c, x_of_z(c, z), &G, &W);
			if (W != 0)
				break;
			z += (i == 0 ? 1 : -1) * 1e-3L / N;
		}
		zr[i] = z;
	}
	for (it = 0; it < 60; it++) {
		if (!solve_ref(c, zr, N)) {
			if (best < INFINITY)
				break;
			*why = "singular Remez system";
			return 0;
		}
		ne = extrema(c, ze, ee, 4 * MAXREF, &gmax);
		if (ne < 0) {
			*why = "error is not finite (a pole on the interval?)";
			if (best < INFINITY)
				break;
			return 0;
		}
		if (gmax < best) {
			if (gmax < best * (1 - 1e-4L))
				bestit = it;
			best = gmax;
			memcpy(bq, c->q, sizeof bq);
			memcpy(bd, c->d, sizeof bd);
		}
		r->iters = it + 1;
		if (gmax - fabsl(c->E) <= 1e-6L * gmax || it - bestit > 6)
			break;
		if (ne >= N) {
			/* keep N alternating extrema including the largest */
			int lo = 0, hi = ne - 1, gm = 0;
			for (i = 1; i < ne; i++)
				if (fabsl(ee[i]) > fabsl(ee[gm]))
					gm = i;
			while (hi - lo + 1 > N) {
				if (lo == gm)
					hi--;
				else if (hi == gm)
					lo++;
				else if (fabsl(ee[lo]) < fabsl(ee[hi]))
					lo++;
				else
					hi--;
			}
			for (i = 0; i < N; i++)
				zr[i] = ze[lo + i];
		} else {
			/* single exchange: swap in the largest error */
			ld zm = 0, em = 0, er[MAXREF];
			int gm = 0;
			for (i = 0; i < ne; i++)
				if (fabsl(ee[i]) > fabsl(em))
					em = ee[i], zm = ze[i], gm = i;
			(void)gm;
			for (i = 0; i < N; i++)
				er[i] = werr(c, zr[i], NULL);
			if (zm < zr[0]) {
				if ((em > 0) == (er[0] > 0))
					zr[0] = zm;
				else {
					memmove(zr + 1, zr, (N - 1) * sizeof *zr);
					zr[0] = zm;
				}
			} else if (zm > zr[N - 1]) {
				if ((em > 0) == (er[N - 1] > 0))
					zr[N - 1] = zm;
				else {
					memmove(zr, zr + 1, (N - 1) * sizeof *zr);
					zr[N - 1] = zm;
				}
			} else {
				for (i = 0; i + 1 < N; i++)
					if (zr[i] <= zm && zm <= zr[i + 1])
						break;
				if (i + 1 >= N)
					break;
				if ((em > 0) == (er[i] > 0))
					zr[i] = zm;
				else
					zr[i + 1] = zm;
			}
		}
		/* keep the reference strictly increasing */
		for (i = 1; i < N; i++)
			if (!(zr[i] > zr[i - 1]))
				break;
		if (i < N)
			break;
	}
	if (best == INFINITY) {
		*why = "no solution";
		return 0;
	}
	memcpy(c->q, bq, sizeof bq);
	memcpy(c->d, bd, sizeof bd);
	return 1;
}

int
fit_run(const struct fitprob *pb, struct fitres *r)
{
	static struct fctx c;
	const char *why = "";
	int ok = 0, pade = 0;

	memset(r, 0, sizeof *r);
	if (pb->nt < 1 || pb->nt + pb->nd > AMAXC - AMAXK || !(pb->xb > pb->xa)) {
		r->why = "bad problem";
		return 0;
	}
	if (!setup(&c, pb)) {
		r->why = "cannot place the exact points";
		return 0;
	}
	if (c.nq == 0 && c.nd == 0) {
		ok = 1;		/* everything fixed by the exact points */
	} else switch (pb->method) {
	case FM_CHEB:
		ok = c.nd == 0 && fit_cheb(&c);
		why = "interpolation failed";
		break;
	case FM_ECON:
		ok = fit_econ(&c, &why);
		break;
	case FM_PADE:
		ok = fit_pade(&c, r, &why);
		pade = 1;
		break;
	case FM_REMEZ:
		ok = c.nq > 0 && fit_remez(&c, r, &why);
		if (c.nq == 0)
			why = "no free coefficients";
		break;
	}
	if (!ok) {
		r->why = why;
		return 0;
	}
	if (!pade)
		finish(&c, r);
	fit_measure(pb, r);
	if (r->nd > 1) {
		/* the denominator must keep its sign on the interval */
		int g;
		for (g = 0; g <= 2000; g++) {
			ld x = pb->xa + (pb->xb - pb->xa) * g / 2000, t, v;
			t = x - pb->c;
			v = pb->step == 2 ? t * t : t;
			if (!(peval(r->den, r->nd, v) > 0)) {
				r->why = "denominator vanishes on the interval";
				return 0;
			}
		}
	}
	if (!isfinite(r->err)) {
		r->why = "not finite on the interval";
		return 0;
	}
	r->ok = 1;
	return 1;
}
