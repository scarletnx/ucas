/*
 * calc2: derivatives and their conditions, gradient/Hessian/Jacobian/
 * Laplacian/directional derivatives, Wronskians, limits, Taylor series
 * and implicit differentiation.
 *
 * The symbolic work is done with diff_node (analyze.c) and tidy
 * (closed.c).  Nothing is taken on trust: every derivative is compared
 * with a Richardson-extrapolated central difference of the previous one,
 * points where a derivative may fail (corners of abs/min/max, piece
 * boundaries, sqrt and fractional powers at 0, jumps of floor/sign) are
 * located and examined by one-sided difference quotients, limits are
 * computed symbolically and then approached numerically, a Taylor
 * polynomial is compared with the function near the point, and an
 * implicit derivative with the slope of the curve found numerically.
 */
#include "calc.h"
#include "calc2.h"
#include "approx.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- formatting ---- */

/* a number as a short exact-looking string: 3, -1/2, pi/4, else %.10g */
static const char *
fnum(double v)
{
	static char buf[8][48];
	static int k;
	char *b = buf[k++ & 7];
	struct q q;

	if (isnan(v))
		return "undefined";
	if (isinf(v))
		return v > 0 ? "inf" : "-inf";
	if (v == 0)
		return "0";
	if (fabs(v - nearbyint(v)) <= 1e-12 * fmax(1, fabs(v)) &&
	    fabs(v) < 1e15 && (nearbyint(v) != 0 || fabs(v) < 1e-300)) {
		snprintf(b, 48, "%.0f", nearbyint(v));
		return b;
	}
	if (q_from_double(v, &q) && q.d <= 1000 && llabs(q.n) < 100000) {
		snprintf(b, 48, "%lld/%lld", (long long)q.n, (long long)q.d);
		return b;
	}
	if (q_from_double(v / M_PI, &q) && q.d <= 12 && llabs(q.n) <= 48) {
		if (q.d == 1)
			snprintf(b, 48, "%s%s", q.n == 1 ? "" : q.n == -1 ? "-" :
			    fnum((double)q.n), "pi");
		else
			snprintf(b, 48, "%spi/%lld", q.n == 1 ? "" : q.n == -1 ? "-" :
			    fnum((double)q.n), (long long)q.d);
		if (q.n != 1 && q.n != -1 && q.d != 1)
			snprintf(b, 48, "%lld*pi/%lld", (long long)q.n,
			    (long long)q.d);
		return b;
	}
	if (fabs(v * M_E - nearbyint(v * M_E)) <= 1e-12 * fabs(v * M_E) &&
	    nearbyint(v * M_E) != 0 && fabs(nearbyint(v * M_E)) <= 12) {
		snprintf(b, 48, "%.0f/e", nearbyint(v * M_E));
		return b;
	}
	snprintf(b, 48, "%.10g", v);
	return b;
}

const char *
c2_fnum(double v)
{
	return fnum(v);
}

static const char *const supd[] = {"", "", "\xc2\xb2", "\xc2\xb3",
	"\xe2\x81\xb4", "\xe2\x81\xb5", "\xe2\x81\xb6", "\xe2\x81\xb7",
	"\xe2\x81\xb8", "\xe2\x81\xb9"};

static void
sup(struct sbuf *b, int n)
{
	if (n >= 2 && n <= 9)
		sb_puts(b, supd[n]);
	else if (n > 9)
		sb_printf(b, "^%d", n);
}

static const char *
pname(const struct func *f, int s)
{
	return sym_str(f->p[s].name);
}

static const char *
fname2(const struct func *f)
{
	return f->anon ? node_str(f->body) : sym_str(f->name);
}

/*
 * d/dx, d²/dx², ∂/∂x, ∂²/∂y∂x: slots[0] is applied first, so it is
 * written last (operator order).
 */
static const char *
dname(const struct func *f, const int *slots, int n)
{
	static struct sbuf b[4];
	static int k;
	struct sbuf *s = &b[k++ & 3];
	const char *d = f->np > 1 ? "\xe2\x88\x82" : "d";
	int i, j;

	sb_clear(s);
	sb_puts(s, d);
	sup(s, n);
	sb_puts(s, "/");
	for (i = n - 1; i >= 0; i = j) {
		for (j = i; j >= 0 && slots[j] == slots[i]; j--)
			;
		sb_printf(s, "%s%s", d, pname(f, slots[i]));
		sup(s, i - j);
	}
	return sb_str(s);
}

/* ---- evaluation ---- */

/* several variables, one point at a time (double, ideal integers) */
struct mev {
	struct plan p;
	int64_t *ii;
	double *ir;
	uint32_t loc;
};

static void
mev_open(struct mev *m, nid e)
{
	nid r = mk_cast(TY_REAL, e);

	plan_build(&m->p, &r, 1, 1);
	m->loc = m->p.root[0];
	m->ii = xcalloc((size_t)MAXP * CH, sizeof *m->ii);
	m->ir = xcalloc((size_t)MAXP * CH, sizeof *m->ir);
}

static void
mev_close(struct mev *m)
{
	plan_free(&m->p);
	free(m->ii);
	free(m->ir);
}

static double
mev_at(struct mev *m, const double *x, int np, int *ok)
{
	double v;
	int k;

	for (k = 0; k < np; k++) {
		m->ir[(size_t)k * CH] = x[k];
		m->ii[(size_t)k * CH] = (int64_t)x[k];
	}
	plan_run(&m->p, 1, m->ii, m->ir);
	v = COLR(&m->p, m->loc)[0];
	*ok = !(COLF(&m->p, m->loc)[0] & F_UNDEF) && isfinite(v);
	return v;
}

/* one variable in long double (lref, remez.c) */
static void
lr_open(struct lref *r, nid e, int slot)
{
	lref_init(r, mk_cast(TY_REAL, e), slot);
}

static ld
lr(struct lref *r, ld x)
{
	ld v = lref_eval(r, x);

	return isfinite(v) ? v : NAN;
}

/* where checks sample: the parameter's range, else [-10, 10] */
static void
check_interval(const struct func *f, int s, double *lo, double *hi)
{
	if (f->p[s].ranged && TY_FP(f->p[s].ty) && f->p[s].rlo < f->p[s].rhi) {
		*lo = f->p[s].rlo;
		*hi = f->p[s].rhi;
	} else {
		*lo = -10;
		*hi = 10;
	}
}

/* catch a fail() from diff_node and friends: 0 if it failed */
static nid simp(nid e, int agg);

static nid
try_diff(nid e, int slot, int quiet)
{
	jmp_buf saved;
	nid d = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		d = simp(diff_node(mk_cast(TY_REAL, e), (uint32_t)slot), 0);
	memcpy(fail_jmp, saved, sizeof saved);
	if (quiet)
		notes_drop();
	return d;
}

static nid
dif(nid e, int slot)
{
	return simp(diff_node(mk_cast(TY_REAL, e), (uint32_t)slot), 0);
}

/* ---- simplification of derivative and limit expressions ---- */

/*
 * A sum of terms c * prod(base^e).  In the default mode a factor and its
 * reciprocal are kept apart (x*y/y stays, so nothing defined only on
 * part of the line silently becomes total); in limit mode (agg) they
 * cancel, which is right for a limit: it does not see single points.
 * Like terms are collected, products of short sums distributed, numeric
 * coefficients folded, exp(a) exp(b) = exp(a + b), 1/exp(a) = exp(-a),
 * and (agg) sqrt(u)^2 = u.
 */
#define SMAXF 12
#define SMAXT 48
struct sterm {
	double c;
	int nf;
	nid b[SMAXF];
	double ep[SMAXF], en[SMAXF];   /* exponent in the numerator / denominator */
};
struct ssum {
	int n, bad;
	struct sterm t[SMAXT];
};

static nid simp_node(nid e, int agg);

static void
st_mulbase(struct sterm *t, nid b, double e, int agg, int *bad)
{
	int k;

	if (e == 0)
		return;
	for (k = 0; k < t->nf; k++)
		if (t->b[k] == b) {
			if (agg) {
				double net = t->ep[k] - t->en[k] + e;
				t->ep[k] = net > 0 ? net : 0;
				t->en[k] = net < 0 ? -net : 0;
			} else if (e > 0)
				t->ep[k] += e;
			else
				t->en[k] -= e;
			return;
		}
	if (t->nf == SMAXF) {
		*bad = 1;
		return;
	}
	t->b[t->nf] = b;
	t->ep[t->nf] = e > 0 ? e : 0;
	t->en[t->nf] = e < 0 ? -e : 0;
	t->nf++;
}

static void
ss_add_term(struct ssum *s, const struct sterm *t)
{
	if (s->n == SMAXT) {
		s->bad = 1;
		return;
	}
	s->t[s->n++] = *t;
}

static void
ss_atom(struct ssum *s, nid b)
{
	struct sterm t;

	memset(&t, 0, sizeof t);
	t.c = 1;
	s->n = 0;
	s->bad = 0;
	st_mulbase(&t, b, 1, 0, &s->bad);
	ss_add_term(s, &t);
}

static void
ss_mul(struct ssum *r, const struct ssum *a, const struct ssum *b, int agg)
{
	int i, j, k;

	r->n = 0;
	r->bad = a->bad || b->bad;
	if (a->n * b->n > SMAXT) {
		r->bad = 1;
		return;
	}
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			struct sterm t = a->t[i];
			t.c *= b->t[j].c;
			for (k = 0; k < b->t[j].nf; k++) {
				if (b->t[j].ep[k])
					st_mulbase(&t, b->t[j].b[k], b->t[j].ep[k], agg, &r->bad);
				if (b->t[j].en[k])
					st_mulbase(&t, b->t[j].b[k], -b->t[j].en[k], agg,
					    &r->bad);
			}
			ss_add_term(r, &t);
		}
}

/* the node of a sum, for use as an opaque base */
static nid ss_node(const struct ssum *s);

static void
ss_of(nid e, int agg, struct ssum *s)
{
	struct ssum *pa, *pb;
	double v;
	int k;

	s->n = 0;
	s->bad = 0;
	if (num_of(e, &v) && (TY_FP(N.ty[e]) || TYK(N.ty[e]) == K_LIT)) {
		struct sterm t;
		memset(&t, 0, sizeof t);
		t.c = v;
		if (v != 0)
			ss_add_term(s, &t);
		return;
	}
	if (!TY_FP(N.ty[e]) && N.op[e] != O_CAST) {
		ss_atom(s, e);
		return;
	}
	pa = xmalloc(sizeof *pa);
	pb = xmalloc(sizeof *pb);
	switch (N.op[e]) {
	case O_CAST:
		if (TY_FP(N.ty[N.a[e]])) {
			ss_of(N.a[e], agg, s);
			goto done;
		}
		ss_atom(s, e);
		goto done;
	case O_ADD:
	case O_SUB:
		ss_of(N.a[e], agg, s);
		ss_of(N.b[e], agg, pb);
		for (k = 0; k < pb->n; k++) {
			if (N.op[e] == O_SUB)
				pb->t[k].c = -pb->t[k].c;
			ss_add_term(s, &pb->t[k]);
		}
		s->bad |= pb->bad;
		goto done;
	case O_NEG:
		ss_of(N.a[e], agg, s);
		for (k = 0; k < s->n; k++)
			s->t[k].c = -s->t[k].c;
		goto done;
	case O_MUL:
		ss_of(N.a[e], agg, pa);
		ss_of(N.b[e], agg, pb);
		ss_mul(s, pa, pb, agg);
		if (s->bad && !pa->bad && !pb->bad) {
			/* too many terms: keep the second factor whole */
			nid bn = ss_node(pb);
			ss_atom(pb, bn);
			ss_mul(s, pa, pb, agg);
		}
		goto done;
	case O_DIV:
		ss_of(N.a[e], agg, pa);
		ss_of(N.b[e], agg, pb);
		if (pb->n == 1 && !pb->bad && pb->t[0].c != 0) {
			/* invert one term: c * prod b^e -> 1/c * prod b^-e */
			struct sterm t = pb->t[0];
			struct ssum *inv = pb;
			memset(inv, 0, sizeof *inv);
			inv->n = 1;
			inv->t[0].c = 1 / t.c;
			for (k = 0; k < t.nf; k++) {
				nid base = t.b[k];
				double ep = t.ep[k], en = t.en[k];
				/* 1/exp(u) = exp(-u): exp is never 0 */
				if (N.op[base] == O_EXP && ep && !en) {
					nid nb = simp_node(mk_un(O_NEG, MUL(KR(ep), N.a[base])),
					    agg);
					st_mulbase(&inv->t[0], mk_un(O_EXP, nb), 1, agg,
					    &inv->bad);
					continue;
				}
				if (en)
					st_mulbase(&inv->t[0], base, en, agg, &inv->bad);
				if (ep)
					st_mulbase(&inv->t[0], base, -ep, agg, &inv->bad);
			}
			ss_mul(s, pa, inv, agg);
			goto done;
		}
		if (!pb->bad && pb->n > 0) {
			/* a sum: keep the denominator as written (factored) */
			nid bn = N.op[N.b[e]] == O_MUL ? simp_node(N.b[e], -1 - agg) :
			    ss_node(pb);
			memset(pb, 0, sizeof *pb);
			pb->n = 1;
			pb->t[0].c = 1;
			st_mulbase(&pb->t[0], bn, -1, agg, &pb->bad);
			ss_mul(s, pa, pb, agg);
			goto done;
		}
		break;
	case O_POW:
		if (num_of(N.a[e], &v) && v == M_E) {
			/* e^u is exp(u) */
			ss_atom(s, mk_un(O_EXP, simp_node(N.b[e], agg)));
			goto done;
		}
		if (num_of(N.b[e], &v) && v == floor(v) && fabs(v) <= 16) {
			ss_of(N.a[e], agg, pa);
			if (pa->bad)
				break;
			if (pa->n == 1) {
				struct sterm t = pa->t[0];
				int neg = v < 0;
				if (neg && t.c == 0)
					break;
				t.c = pow(t.c, v);
				for (k = 0; k < t.nf; k++) {
					double ep = t.ep[k] * fabs(v), en = t.en[k] * fabs(v);
					t.ep[k] = neg ? en : ep;
					t.en[k] = neg ? ep : en;
					if (agg) {
						double net = t.ep[k] - t.en[k];
						t.ep[k] = net > 0 ? net : 0;
						t.en[k] = net < 0 ? -net : 0;
					}
				}
				s->n = 0;
				ss_add_term(s, &t);
				goto done;
			}
			if (v == 2 && pa->n <= 4) {
				ss_mul(s, pa, pa, agg);
				goto done;
			}
			if (pa->n > 0) {
				/* a sum to a power: one base */
				nid bn = ss_node(pa);
				struct sterm t;
				memset(&t, 0, sizeof t);
				t.c = 1;
				st_mulbase(&t, bn, v, agg, &s->bad);
				s->n = 0;
				ss_add_term(s, &t);
				goto done;
			}
		}
		break;
	case O_SQRT:
		if (agg) {
			/* sqrt(u) as u^(1/2) so that sqrt(u)^2 = u */
			nid u = simp_node(N.a[e], agg);
			struct sterm t;
			memset(&t, 0, sizeof t);
			t.c = 1;
			st_mulbase(&t, u, 0.5, agg, &s->bad);
			ss_add_term(s, &t);
			goto done;
		}
		break;
	}
	/* an opaque base: its operands simplified */
	ss_atom(s, simp_node(e, -1 - agg));
done:
	free(pa);
	free(pb);
}

static int
st_same(const struct sterm *a, const struct sterm *b)
{
	int i, j;

	if (a->nf != b->nf)
		return 0;
	for (i = 0; i < a->nf; i++) {
		for (j = 0; j < b->nf; j++)
			if (b->b[j] == a->b[i] && b->ep[j] == a->ep[i] &&
			    b->en[j] == a->en[i])
				break;
		if (j == b->nf)
			return 0;
	}
	return 1;
}

static nid
bpow(nid b, double e)
{
	if (e == 1)
		return b;
	if (e == 0.5)
		return mk_un(O_SQRT, b);
	if (e == floor(e))
		return POW(b, KI((int64_t)e));
	if (e * 2 == floor(e * 2))
		return POW(mk_un(O_SQRT, b), KI((int64_t)(e * 2)));
	return POW(b, KR(e));
}

static nid
ss_node(const struct ssum *s)
{
	struct sterm *t = xmalloc(sizeof(struct sterm) * SMAXT);
	int n = 0, i, j, k;
	nid r = 0;

	for (i = 0; i < s->n; i++) {
		struct sterm u = s->t[i];
		/* drop exponent-0 factors; exp(a) exp(b) = exp(a + b) */
		nid ex = 0;
		int nexp = 0;
		for (k = 0; k < u.nf; k++)
			if (N.op[u.b[k]] == O_EXP && u.en[k] == 0 && u.ep[k])
				nexp++;
		for (k = 0, j = 0; k < u.nf; k++) {
			if (u.ep[k] == 0 && u.en[k] == 0)
				continue;
			if (nexp > 1 && N.op[u.b[k]] == O_EXP && u.en[k] == 0) {
				nid arg = u.ep[k] == 1 ? N.a[u.b[k]] : MUL(KR(u.ep[k]),
				    N.a[u.b[k]]);
				ex = ex ? ADD(ex, arg) : arg;
				continue;
			}
			u.b[j] = u.b[k];
			u.ep[j] = u.ep[k];
			u.en[j] = u.en[k];
			j++;
		}
		u.nf = j;
		if (ex) {
			nid en = simp_node(ex, 0);
			if (!const_is(en, 0) && u.nf < SMAXF) {
				u.b[u.nf] = mk_un(O_EXP, en);
				u.ep[u.nf] = 1;
				u.en[u.nf] = 0;
				u.nf++;
			}
		}
		/* sort the factors by id: like terms compare equal */
		for (j = 1; j < u.nf; j++)
			for (k = j; k > 0 && u.b[k - 1] > u.b[k]; k--) {
				nid tb = u.b[k];
				double a = u.ep[k], b = u.en[k];
				u.b[k] = u.b[k - 1];
				u.ep[k] = u.ep[k - 1];
				u.en[k] = u.en[k - 1];
				u.b[k - 1] = tb;
				u.ep[k - 1] = a;
				u.en[k - 1] = b;
			}
		for (j = 0; j < n; j++)
			if (st_same(&t[j], &u)) {
				t[j].c += u.c;
				break;
			}
		if (j == n)
			t[n++] = u;
	}
	/* terms over the same denominator are added over it: a/d + b/d */
	{
		nid dens[SMAXT], nums[SMAXT];
		int nd = 0, g;
		for (i = 0; i < n; i++) {
			nid den = 0, num;
			struct sterm u = t[i];
			int any = 0;
			for (k = 0; k < u.nf; k++)
				if (u.en[k]) {
					den = den ? MUL(den, bpow(u.b[k], u.en[k])) :
					    bpow(u.b[k], u.en[k]);
					u.en[k] = 0;
					any = 1;
				}
			if (!any || fabs(u.c) < 1e-300)
				continue;
			for (g = 0; g < nd; g++)
				if (dens[g] == den)
					break;
			if (g == nd) {
				dens[nd] = den;
				nums[nd++] = 0;
			}
			{
				struct ssum one;
				one.n = 1;
				one.bad = 0;
				one.t[0] = u;
				num = ss_node(&one);
			}
			nums[g] = nums[g] ? ADD(nums[g], num) : num;
			t[i].c = 0;   /* taken */
			t[i].nf = -1 - g;
		}
		for (g = 0; g < nd; g++) {
			nid term = DIV(nums[g], dens[g]), nm = nums[g];
			double cv;
			/* a + -1/d is a - 1/d */
			if (r && N.op[nm] == O_NEG)
				term = SUB(r, DIV(N.a[nm], dens[g]));
			else if (r && num_of(nm, &cv) && cv < 0)
				term = SUB(r, DIV(KR(-cv), dens[g]));
			else if (r)
				term = ADD(r, term);
			r = term;
		}
	}
	for (i = 0; i < n; i++) {
		nid num = 0, den = 0, term;
		double c = t[i].c;
		if (fabs(c) < 1e-300 || t[i].nf < 0)
			continue;
		if (fabs(c - nearbyint(c)) < 1e-12 * fmax(1, fabs(c)))
			c = nearbyint(c);
		{
			/* a coefficient p/q: p in front, q in the denominator */
			struct q q;
			double cn = fabs(c), cd = 1;
			if (cn != floor(cn) && q_from_double(cn, &q) && q.d <= 1000) {
				cn = (double)q.n;
				cd = (double)q.d;
			}
			if (cn != 1)
				num = KR(cn);
			if (cd != 1)
				den = KR(cd);
		}
		for (k = 0; k < t[i].nf; k++) {
			if (t[i].ep[k])
				num = num ? MUL(num, bpow(t[i].b[k], t[i].ep[k])) :
				    bpow(t[i].b[k], t[i].ep[k]);
			if (t[i].en[k])
				den = den ? MUL(den, bpow(t[i].b[k], t[i].en[k])) :
				    bpow(t[i].b[k], t[i].en[k]);
		}
		if (num == 0)
			num = KR(1);
		term = den ? DIV(num, den) : num;
		if (r == 0)
			r = c < 0 ? mk_un(O_NEG, term) : term;
		else
			r = c < 0 ? SUB(r, term) : ADD(r, term);
	}
	free(t);
	return r ? r : KR(0);
}

/*
 * agg >= 0: the whole expression; agg < 0: only the operands of e (it
 * is an opaque base), simplified with mode -1 - agg.
 */
static nid
simp_node(nid e, int agg)
{
	struct ssum *s;
	int op = N.op[e];
	nid r;

	if (agg < 0) {
		int m = -1 - agg;
		switch (op) {
		case O_CONST: case O_VAR: case O_UNDEF: case O_TABLE: case O_LOOP:
		case O_INDEX: case O_ATOMIC: case O_BARRIER: case O_CAST:
			return e;
		case O_SEL:
			return mk_sel(N.a[e], simp_node(N.b[e], m),
			    simp_node(N.c[e], m));
		}
		if (opinfo[op].arity == 1 && N.a[e])
			return mk_un(op, simp_node(N.a[e], m));
		if (opinfo[op].arity == 2 && N.a[e] && N.b[e] &&
		    !(opinfo[op].fl & OF_CMP) && op != O_LAND && op != O_LOR)
			return mk_bin(op, simp_node(N.a[e], m), simp_node(N.b[e], m));
		return e;
	}
	if (!TY_FP(N.ty[e]))
		return e;
	s = xmalloc(sizeof *s);
	ss_of(e, agg, s);
	r = s->bad ? e : ss_node(s);
	free(s);
	return r;
}

static nid
simp_loop(nid e, int agg)
{
	nid r = tidy(e);
	int k;

	for (k = 0; k < 3; k++) {
		nid r2 = simp_node(tidy(r), agg);
		if (r2 == r)
			break;
		r = r2;
	}
	return r;
}

/* sin cos tan sec csc cot sinh cosh tanh anywhere in e */
static int
has_trig(nid e)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0;
	int found = 0;

	st[sp++] = e;
	mark[e] = 1;
	while (sp && !found) {
		g = st[--sp];
		switch (N.op[g]) {
		case O_SIN: case O_COS: case O_TAN: case O_SEC: case O_CSC:
		case O_COT: case O_SINH: case O_COSH: case O_TANH:
			found = 1;
			break;
		}
		if (N.a[g] && !mark[N.a[g]])
			mark[N.a[g]] = 1, st[sp++] = N.a[g];
		if (N.b[g] && !mark[N.b[g]])
			mark[N.b[g]] = 1, st[sp++] = N.b[g];
		if (N.c[g] && !mark[N.c[g]])
			mark[N.c[g]] = 1, st[sp++] = N.c[g];
	}
	free(mark);
	free(st);
	return found;
}

/* a and b agree (and are defined at the same points) at 32 sample points */
static int
same_values(nid a, nid b)
{
	struct mev ma, mb;
	double x[MAXP];
	int i, k, oa, ob, n = 0, bad = 0;
	uint32_t s = 12345;

	mev_open(&ma, a);
	mev_open(&mb, b);
	for (i = 0; i < 32 && !bad; i++) {
		double va, vb;
		for (k = 0; k < MAXP; k++) {
			s = s * 1103515245u + 12345u;
			x[k] = -2.9 + 5.8 * ((s >> 8) & 0xffff) / 65536.0;
		}
		va = mev_at(&ma, x, MAXP, &oa);
		vb = mev_at(&mb, x, MAXP, &ob);
		if (oa != ob)
			bad = 1;
		else if (oa) {
			n++;
			if (fabs(va - vb) > 1e-9 * fmax(1, fmax(fabs(va), fabs(vb))))
				bad = 1;
		}
	}
	mev_close(&ma);
	mev_close(&mb);
	return !bad && n >= 8;
}

/*
 * The identities forms.c knows (sin^2 + cos^2 = 1, cosh^2 - sinh^2 = 1):
 * taken when the result is shorter and agrees numerically
 */
/* is f a factor of the divisor of a quotient in e (so e needs f != 0)? */
static int
factor_of(nid d, nid f)
{
	while (N.op[d] == O_NEG || N.op[d] == O_CAST ||
	    (N.op[d] == O_POW && is_const(N.b[d])))
		d = N.a[d];
	if (d == f)
		return 1;
	return N.op[d] == O_MUL && (factor_of(N.a[d], f) ||
	    factor_of(N.b[d], f));
}

static int
in_divisor(nid e, nid f)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0;
	int found = 0;

	st[sp++] = e;
	mark[e] = 1;
	while (sp && !found) {
		g = st[--sp];
		if (N.op[g] == O_DIV && factor_of(N.b[g], f))
			found = 1;
		if (N.a[g] && !mark[N.a[g]])
			mark[N.a[g]] = 1, st[sp++] = N.a[g];
		if (N.b[g] && !mark[N.b[g]])
			mark[N.b[g]] = 1, st[sp++] = N.b[g];
	}
	free(mark);
	free(st);
	return found;
}

static nid
trig_simp(nid e)
{
	nid r, can[8];
	int nc, k;

	if (!TY_FP(N.ty[e]) || !has_trig(e))
		return e;
	r = forms_norm(e, can, 8, &nc);
	if (r == 0 || r == e)
		return e;
	/* x/x only where e already divides by x: no point is gained */
	for (k = 0; k < nc; k++)
		if (!in_divisor(e, can[k]))
			return e;
	r = simp_loop(r, 0);
	if (r == e || strlen(node_str(r)) >= strlen(node_str(e)) ||
	    !same_values(e, r))
		return e;
	return r;
}

static nid
simp(nid e, int agg)
{
	jmp_buf saved;
	nid r = e;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0) {
		r = simp_loop(e, agg);
		if (agg >= 0) {
			nid r0 = r;
			r = trig_simp(r);
			if (r != r0)
				r = simp_loop(r, agg);
		}
	}
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return r;
}

/* ---- conditions: where a function is defined ---- */

struct cond {
	nid u;
	int rel;           /* 0 u > c, 1 u >= c, 2 u != c, 3 u < c, 4 u <= c */
	double c;
};
#define MAXCOND 24
static const char *const relname[] = {">", ">=", "!=", "<", "<="};

static int
add_cond(struct cond *cs, int n, nid u, int rel, double c)
{
	int k;

	if (!(N.fl[u] & NF_VAR) || n >= MAXCOND)
		return n;
	for (k = 0; k < n; k++)
		if (cs[k].u == u && cs[k].c == c) {
			/* u > 0 is stronger than u >= 0 and u != 0 */
			if (cs[k].rel == 0 && (rel == 1 || rel == 2))
				return n;
			if (rel == 0 && (cs[k].rel == 1 || cs[k].rel == 2)) {
				cs[k].rel = 0;
				return n;
			}
			if (cs[k].rel == rel)
				return n;
		}
	cs[n].u = u;
	cs[n].rel = rel;
	cs[n].c = c;
	return n + 1;
}

/* the conditions the partial operations of e impose */
static int
domain_conds(nid e, struct cond *cs, int n)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g, a, b;
	uint32_t sp = 0;
	double k;

	st[sp++] = e;
	mark[e] = 1;
	while (sp) {
		g = st[--sp];
		if (N.op[g] == O_CONST || N.op[g] == O_VAR ||
		    N.op[g] == O_UNDEF || N.op[g] == O_TABLE ||
		    N.op[g] == O_LOOP)
			continue;
		a = N.a[g];
		b = N.b[g];
		switch (N.op[g]) {
		case O_LOG:
			n = add_cond(cs, n, a, 0, 0);
			break;
		case O_SQRT:
			n = add_cond(cs, n, a, 1, 0);
			break;
		case O_ASIN:	/* forms.c: |a| <= 1, a >= 1, |a| < 1 */
		case O_ACOS:
			n = add_cond(cs, n, SUB(KI(1), POW(a, KI(2))), 1, 0);
			break;
		case O_ACOSH:
			n = add_cond(cs, n, SUB(a, KI(1)), 1, 0);
			break;
		case O_ATANH:
			n = add_cond(cs, n, SUB(KI(1), POW(a, KI(2))), 0, 0);
			break;
		case O_DIV:
			n = add_cond(cs, n, b, 2, 0);
			break;
		case O_POW:
			if (num_of(b, &k)) {
				if (k != floor(k))
					n = add_cond(cs, n, a, k > 0 ? 1 : 0, 0);
				else if (k < 0)
					n = add_cond(cs, n, a, 2, 0);
			} else if (N.fl[a] & NF_VAR)
				n = add_cond(cs, n, a, 0, 0);
			else if (num_of(a, &k) && k <= 0)
				n = add_cond(cs, n, a, 0, 0);
			break;
		case O_TAN:
		case O_SEC:
			n = add_cond(cs, n, mk_un(O_COS, a), 2, 0);
			break;
		case O_CSC:
		case O_COT:
			n = add_cond(cs, n, mk_un(O_SIN, a), 2, 0);
			break;
		case O_LAMBERTW:
			n = add_cond(cs, n, a, 1, -exp(-1.0));
			break;
		case O_LAMBERTWM1:
			n = add_cond(cs, n, a, 1, -exp(-1.0));
			n = add_cond(cs, n, a, 3, 0);
			break;
		}
		if (a && !mark[a])
			mark[a] = 1, st[sp++] = a;
		if (b && !mark[b])
			mark[b] = 1, st[sp++] = b;
		if (N.c[g] && !mark[N.c[g]] && N.op[g] != O_LOOP)
			mark[N.c[g]] = 1, st[sp++] = N.c[g];
	}
	free(mark);
	free(st);
	return n;
}

/* u linear in slot s with numeric coefficients: u = a x + b */
static int
lin_of(nid u, int s, double *a, double *b)
{
	struct npoly p;

	if (!npoly_of(u, s, &p, 1) || p.deg != 1 || !num_of(p.c[1], a) ||
	    !num_of(p.c[0], b) || *a == 0)
		return 0;
	return 1;
}

/*
 * one condition as text; with one variable a linear u is solved for it.
 * *pt and *ptrel receive the solved form (x rel pt) when that worked.
 */
static const char *
cond_text(const struct func *f, const struct cond *c, double *pt, int *ptrel)
{
	static char buf[8][160];
	static int k;
	char *b = buf[k++ & 7];
	double a, bb;
	int rel = c->rel;

	if (ptrel)
		*ptrel = -1;
	if (f->np == 1 && lin_of(c->u, 0, &a, &bb)) {
		double x = (c->c - bb) / a;
		static const int flip[] = {3, 4, 2, 0, 1};
		if (a < 0)
			rel = flip[rel];
		snprintf(b, 160, "%s %s %s", pname(f, 0), relname[rel], fnum(x));
		if (pt)
			*pt = x;
		if (ptrel)
			*ptrel = rel;
		return b;
	}
	snprintf(b, 160, "%s %s %s", node_str(c->u), relname[rel], fnum(c->c));
	return b;
}

/* ---- where a derivative can fail ---- */

enum { KK_CORNER, KK_JUMP, KK_INT, KK_VERT, KK_PIECE };
struct cand {
	int kind;
	nid u;          /* the derivative can fail where u = 0 */
	nid src;        /* the operation */
};
#define MAXCAND 24

static int
add_cand(struct cand *c, int n, int kind, nid u, nid src)
{
	int k;

	if (!(N.fl[u] & NF_VAR) || n >= MAXCAND)
		return n;
	for (k = 0; k < n; k++)
		if (c[k].u == u)
			return n;
	c[n].kind = kind;
	c[n].u = u;
	c[n].src = src;
	return n + 1;
}

static int
kink_cands(nid e, struct cand *c, int n)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g, a, b;
	uint32_t sp = 0;
	double k;

	st[sp++] = e;
	mark[e] = 1;
	while (sp) {
		g = st[--sp];
		if (N.op[g] == O_CONST || N.op[g] == O_VAR ||
		    N.op[g] == O_UNDEF || N.op[g] == O_TABLE ||
		    N.op[g] == O_LOOP)
			continue;
		a = N.a[g];
		b = N.b[g];
		switch (N.op[g]) {
		case O_ABS:
			n = add_cand(c, n, KK_CORNER, a, g);
			break;
		case O_SIGN:
			n = add_cand(c, n, KK_JUMP, a, g);
			break;
		case O_COPYSIGN:
			n = add_cand(c, n, KK_JUMP, b, g);
			break;
		case O_FLOOR:
		case O_CEIL:
			n = add_cand(c, n, KK_INT, a, g);
			break;
		case O_SQRT:
			n = add_cand(c, n, KK_VERT, a, g);
			break;
		case O_ASIN:
		case O_ACOS:
			n = add_cand(c, n, KK_VERT, SUB(KI(1), POW(a, KI(2))), g);
			break;
		case O_ACOSH:
			n = add_cand(c, n, KK_VERT, SUB(a, KI(1)), g);
			break;
		case O_POW:
			if (num_of(b, &k) && k != floor(k) && k > 0)
				n = add_cand(c, n, KK_VERT, a, g);
			break;
		case O_MIN:
		case O_MAX:
			n = add_cand(c, n, KK_CORNER, SUB(a, b), g);
			break;
		case O_LT: case O_LE: case O_GT: case O_GE: case O_EQ: case O_NE:
			if (TY_FP(N.ty[a]) || TY_FP(N.ty[b]))
				n = add_cand(c, n, KK_PIECE, SUB(mk_cast(TY_REAL, a),
				    mk_cast(TY_REAL, b)), g);
			break;
		case O_LAMBERTW:
		case O_LAMBERTWM1:
			n = add_cand(c, n, KK_VERT, ADD(a, KR(exp(-1.0))), g);
			break;
		}
		if (a && !mark[a])
			mark[a] = 1, st[sp++] = a;
		if (b && !mark[b])
			mark[b] = 1, st[sp++] = b;
		if (N.c[g] && !mark[N.c[g]])
			mark[N.c[g]] = 1, st[sp++] = N.c[g];
	}
	free(mark);
	free(st);
	return n;
}

/* the zeros of u (one variable) in [lo, hi]; integer crossings for KK_INT */
static int
cand_points(const struct cand *c, int slot, double lo, double hi,
    double *pts, int max)
{
	struct lref r;
	double a, b;
	int n = 0, i, j;
	const int G = 4000;
	ld prev = NAN, cur, pprev = NAN;

	if (c->kind != KK_INT && lin_of(c->u, slot, &a, &b)) {
		double x = -b / a;
		if (x >= lo && x <= hi && max > 0)
			pts[n++] = x;
		return n;
	}
	lr_open(&r, c->u, slot);
	for (i = 0; i <= G && n < max; i++) {
		ld x = lo + (hi - lo) * (ld)i / G;
		cur = lr(&r, x);
		if (c->kind == KK_INT) {
			if (!isnan(cur) && !isnan(prev) && floorl(cur) != floorl(prev)) {
				/* bisect for the crossing of the integer between */
				ld xa = x - (hi - lo) / G, xb = x, t = floorl(cur) >
				    floorl(prev) ? floorl(cur) : floorl(prev);
				for (j = 0; j < 80; j++) {
					ld m = 0.5 * (xa + xb), v = lr(&r, m);
					if (isnan(v))
						break;
					if ((v >= t) == (prev >= t))
						xa = m;
					else
						xb = m;
				}
				pts[n++] = (double)(0.5 * (xa + xb));
			}
		} else if (!isnan(cur) && cur == 0) {
			pts[n++] = (double)x;
		} else if (!isnan(cur) && !isnan(prev) && prev != 0 &&
		    (cur > 0) != (prev > 0)) {
			ld xa = x - (hi - lo) / G, xb = x, fa = prev;
			for (j = 0; j < 100; j++) {
				ld m = 0.5 * (xa + xb), v = lr(&r, m);
				if (isnan(v) || v == 0) {
					xa = xb = m;
					break;
				}
				if ((v > 0) == (fa > 0))
					xa = m, fa = v;
				else
					xb = m;
			}
			pts[n++] = (double)(0.5 * (xa + xb));
		} else if (!isnan(cur) && !isnan(prev) && !isnan(pprev) &&
		    fabsl(prev) < fabsl(cur) && fabsl(prev) < fabsl(pprev) &&
		    fabsl(prev) < 1e-3) {
			/* a touching zero (x^2 at 0): refine the minimum of |u| */
			ld xa = x - 2 * (hi - lo) / G, xb = x;
			for (j = 0; j < 100; j++) {
				ld m1 = xa + (xb - xa) / 3, m2 = xb - (xb - xa) / 3;
				ld v1 = fabsl(lr(&r, m1)), v2 = fabsl(lr(&r, m2));
				if (isnan(v1) || isnan(v2))
					break;
				if (v1 < v2)
					xb = m2;
				else
					xa = m1;
			}
			if (fabsl(lr(&r, 0.5 * (xa + xb))) < 1e-14)
				pts[n++] = (double)(0.5 * (xa + xb));
		}
		pprev = prev;
		prev = cur;
	}
	lref_free(&r);
	/* round to a nearby simple number: 0.49999999 is 1/2 */
	for (i = 0; i < n; i++) {
		struct q q;
		if (fabs(pts[i]) < 1e-12)
			pts[i] = 0;
		else if (q_from_double(pts[i], &q) && q.d <= 1000)
			pts[i] = q_val(q);
	}
	return n;
}

enum { DX_OK, DX_CORNER, DX_VERT, DX_JUMP, DX_EDGE, DX_UNDEF, DX_OSC };
struct dx {
	int kind;
	double x, sl, sr;       /* one-sided slopes (NAN: none) */
	int lok, rok;           /* the side is defined */
	int linf, rinf;         /* the slope grows without bound */
};

/* is g differentiable at x0?  one-sided difference quotients */
static void
dx_at(struct lref *g, double x0, struct dx *d)
{
	ld g0 = lr(g, x0), sc = fmax(1, fabs(x0));
	static const double hs[3] = {1e-3, 1e-5, 1e-7};
	ld s[2][3], dd[2][3];
	int side, i, ok[2] = {1, 1}, conv[2], inf[2], cont[2];

	memset(d, 0, sizeof *d);
	d->x = x0;
	d->sl = d->sr = NAN;
	if (isnan(g0)) {
		d->kind = DX_UNDEF;
		return;
	}
	for (side = 0; side < 2; side++) {
		ld sg = side ? 1 : -1;
		for (i = 0; i < 3; i++) {
			ld h = hs[i] * sc, v = lr(g, x0 + sg * h);
			if (isnan(v)) {
				ok[side] = 0;
				break;
			}
			dd[side][i] = fabsl(v - g0);
			s[side][i] = (v - g0) / (sg * h);
		}
		if (!ok[side]) {
			conv[side] = inf[side] = 0;
			cont[side] = 1;
			continue;
		}
		cont[side] = !(dd[side][2] > 0.5 * dd[side][0] &&
		    dd[side][2] > 1e-9 * (1 + fabsl(g0)));
		conv[side] = fabsl(s[side][1] - s[side][2]) <=
		    1e-4 * (1 + fabsl(s[side][2]));
		inf[side] = !conv[side] && fabsl(s[side][2]) > 3 *
		    fabsl(s[side][1]) && fabsl(s[side][1]) > 3 * fabsl(s[side][0]);
	}
	d->lok = ok[0];
	d->rok = ok[1];
	d->linf = inf[0];
	d->rinf = inf[1];
	/* first-order quotients: extrapolate with the step 2h */
	for (side = 0; side < 2; side++)
		if (ok[side] && conv[side]) {
			ld sg = side ? 1 : -1, h = 2 * hs[2] * sc, v = lr(g, x0 + sg * h);
			ld s2 = isnan(v) ? s[side][2] : (v - g0) / (sg * h);
			ld e = 2 * s[side][2] - s2;
			double r = (double)e;
			if (fabs(r - nearbyint(r)) < 1e-6)
				r = nearbyint(r);
			if (side)
				d->sr = r;
			else
				d->sl = r;
		}
	if ((ok[0] && !cont[0]) || (ok[1] && !cont[1])) {
		d->kind = DX_JUMP;
		return;
	}
	if (!ok[0] || !ok[1]) {
		d->kind = (ok[0] || ok[1]) ? DX_EDGE : DX_UNDEF;
		if (!ok[0] && !ok[1])
			d->kind = DX_UNDEF;
		return;
	}
	if (inf[0] || inf[1]) {
		d->kind = DX_VERT;
		return;
	}
	if (!conv[0] || !conv[1]) {
		d->kind = DX_OSC;
		return;
	}
	d->kind = fabs(d->sl - d->sr) <= 1e-4 * (1 + fmax(fabs(d->sl),
	    fabs(d->sr))) ? DX_OK : DX_CORNER;
}

/* a short reason for a bad point */
static const char *
dx_reason(const struct func *f, const char *gname, const struct dx *d)
{
	static char buf[4][256];
	static int k;
	char *b = buf[k++ & 3];
	const char *x = pname(f, 0), *at = fnum(d->x);

	switch (d->kind) {
	case DX_CORNER:
		snprintf(b, 256, "at %s = %s %s has a corner (slope %s from the "
		    "left, %s from the right)", x, at, gname, fnum(d->sl),
		    fnum(d->sr));
		break;
	case DX_VERT:
		snprintf(b, 256, "at %s = %s the slope of %s is infinite "
		    "(vertical tangent)", x, at, gname);
		break;
	case DX_JUMP:
		snprintf(b, 256, "at %s = %s %s jumps (not continuous)", x, at,
		    gname);
		break;
	case DX_EDGE:
		if (d->linf || d->rinf)
			snprintf(b, 256, "at %s = %s (an end of the domain) %s has an "
			    "infinite one-sided slope", x, at, gname);
		else if (!isnan(d->lok ? d->sl : d->sr))
			snprintf(b, 256, "at %s = %s (an end of the domain) %s has "
			    "only a one-sided derivative, %s", x, at, gname,
			    fnum(d->lok ? d->sl : d->sr));
		else
			snprintf(b, 256, "at %s = %s (an end of the domain) %s has "
			    "no one-sided derivative", x, at, gname);
		break;
	case DX_OSC:
		snprintf(b, 256, "at %s = %s the difference quotients of %s do "
		    "not settle (no derivative)", x, at, gname);
		break;
	default:
		b[0] = 0;
	}
	return b;
}

/*
 * Check d = g' numerically: Richardson-extrapolated central differences
 * of g at points of [lo, hi] away from the points in avoid[].
 */
static int
check1(nid g, nid d, int slot, double lo, double hi, const double *avoid,
    int navoid, int *npts, double *maxrel)
{
	struct lref rg, rd;
	int i, j, n = 0, bad = 0;
	double worst = 0;

	lr_open(&rg, g, slot);
	lr_open(&rd, d, slot);
	for (i = 0; i <= 200; i++) {
		ld x = lo + (hi - lo) * (i + 0.37) / 201.37, h, D1, D2, R, dv, sc;
		int skip = 0;
		for (j = 0; j < navoid; j++)
			if (fabsl(x - avoid[j]) < 1e-2 * fmax(1, fabs(avoid[j])))
				skip = 1;
		if (skip)
			continue;
		h = (rg.full ? 1e-4 : 1e-3) * fmaxl(1, fabsl(x));
		{
			ld a1 = lr(&rg, x + h), b1 = lr(&rg, x - h);
			ld a2 = lr(&rg, x + h / 2), b2 = lr(&rg, x - h / 2);
			dv = lr(&rd, x);
			if (isnan(a1) || isnan(b1) || isnan(a2) || isnan(b2) ||
			    isnan(dv))
				continue;
			D1 = (a1 - b1) / (2 * h);
			D2 = (a2 - b2) / h;
			R = (4 * D2 - D1) / 3;
			sc = fmaxl(1, fmaxl(fabsl(dv), fabsl(lr(&rg, x))));
			/* not resolved at this step (a pole nearby): skip */
			if (fabsl(D2 - D1) > 0.01 * fmaxl(fabsl(D1), fabsl(D2)) &&
			    fabsl(D2 - D1) > 1e-9 * sc)
				continue;
		}
		n++;
		/* the extrapolation error is about |D2 - D1| / 15 */
		if (fabsl(R - dv) > 1e-6 * sc + fabsl(D2 - D1) / 5) {
			bad++;
		}
		if (fabsl(R - dv) / sc > worst)
			worst = (double)(fabsl(R - dv) / sc);
	}
	lref_free(&rg);
	lref_free(&rd);
	*npts = n;
	*maxrel = worst;
	return bad == 0;
}

/* several variables: sample points in [-2, 2]^n (or the ranges) */
static void
sample_point(const struct func *f, int i, double *x)
{
	uint64_t s = 0x9e3779b97f4a7c15ull * (uint64_t)(i + 1);
	int k;

	for (k = 0; k < f->np; k++) {
		double lo = -2, hi = 2, u;
		s ^= s >> 31;
		s *= 0xbf58476d1ce4e5b9ull;
		s ^= s >> 27;
		u = (double)(s >> 11) / 9007199254740992.0;
		if (f->p[k].ranged && TY_FP(f->p[k].ty)) {
			lo = f->p[k].rlo;
			hi = f->p[k].rhi;
		} else if (!TY_FP(f->p[k].ty) && f->p[k].ranged) {
			lo = (double)f->p[k].lo;
			hi = (double)f->p[k].hi;
		}
		x[k] = lo + (hi - lo) * u;
		if (!TY_FP(f->p[k].ty))
			x[k] = floor(x[k] + 0.5);
	}
}

static int
checkn(const struct func *f, nid g, nid d, int slot, int *npts,
    double *maxrel)
{
	struct mev mg, md;
	double x[MAXP], worst = 0;
	int i, n = 0, bad = 0, ok;

	mev_open(&mg, g);
	mev_open(&md, d);
	for (i = 0; i < 400 && n < 100; i++) {
		double h, a1, b1, a2, b2, dv, x0, D1, D2, R, sc;
		int o[5];
		sample_point(f, i, x);
		x0 = x[slot];
		h = 1e-3 * fmax(1, fabs(x0));
		x[slot] = x0 + h;
		a1 = mev_at(&mg, x, f->np, &o[0]);
		x[slot] = x0 - h;
		b1 = mev_at(&mg, x, f->np, &o[1]);
		x[slot] = x0 + h / 2;
		a2 = mev_at(&mg, x, f->np, &o[2]);
		x[slot] = x0 - h / 2;
		b2 = mev_at(&mg, x, f->np, &o[3]);
		x[slot] = x0;
		dv = mev_at(&md, x, f->np, &o[4]);
		if (!o[0] || !o[1] || !o[2] || !o[3] || !o[4])
			continue;
		D1 = (a1 - b1) / (2 * h);
		D2 = (a2 - b2) / h;
		R = (4 * D2 - D1) / 3;
		sc = fmax(1, fmax(fabs(dv), fabs(mev_at(&mg, x, f->np, &ok))));
		if (fabs(D2 - D1) > 0.01 * fmax(fabs(D1), fabs(D2)) &&
		    fabs(D2 - D1) > 1e-9 * sc)
			continue;   /* not resolved at this step */
		n++;
		if (fabs(R - dv) > 1e-6 * sc + fabs(D2 - D1) / 5)
			bad++;
		if (fabs(R - dv) / sc > worst)
			worst = fabs(R - dv) / sc;
	}
	mev_close(&mg);
	mev_close(&md);
	*npts = n;
	*maxrel = worst;
	return bad == 0;
}

/* ---- derivatives ---- */

/* the argument of the first floor/ceil of a linear argument in g[] */
static nid
kink_int_arg(const nid *g, int n)
{
	struct cand c[MAXCAND];
	double a, b;
	int k, i, nc;

	for (k = 0; k < n; k++) {
		nc = kink_cands(g[k], c, 0);
		for (i = 0; i < nc; i++)
			if (c[i].kind == KK_INT && lin_of(c[i].u, 0, &a, &b))
				return c[i].u;
	}
	return KR(0);
}

struct badpt {
	double x;
	const char *why;
};

/*
 * Where g_0 .. g_{n-1} (each differentiated once more) can fail to be
 * differentiable: one variable only.  Returns the bad points.
 */
static int
find_bad(const struct func *f, const nid *g, int n, double lo, double hi,
    struct badpt *bp, int max, double *cpts, int *ncpts)
{
	struct cand c[MAXCAND];
	int nb = 0, k, i, j, m, nc, np;
	double pts[16];

	*ncpts = 0;
	for (k = 0; k < n; k++) {
		struct lref r;
		nc = kink_cands(g[k], c, 0);
		lr_open(&r, g[k], 0);
		for (i = 0; i < nc; i++) {
			double la, lb;
			if (c[i].kind == KK_INT && lin_of(c[i].u, 0, &la, &lb) &&
			    nb < max) {
				/* floor(a x + b) jumps wherever a x + b is an integer */
				struct dx d;
				double x0 = (nearbyint(la * 0.5 * (lo + hi) + lb) - lb) / la;
				dx_at(&r, x0, &d);
				if (d.kind == DX_OK || d.kind == DX_UNDEF)
					continue;
				bp[nb].x = NAN;
				{
					char b[200];
					snprintf(b, sizeof b, "%s not an integer", node_str(
					    c[i].u));
					bp[nb].why = xmalloc(strlen(b) + 1);
					strcpy((char *)bp[nb].why, b);
				}
				nb++;
				if (*ncpts < 64)
					cpts[(*ncpts)++] = x0;
				continue;
			}
			np = cand_points(&c[i], 0, lo, hi, pts, 16);
			for (j = 0; j < np; j++) {
				struct dx d;
				if (*ncpts < 64)
					cpts[(*ncpts)++] = pts[j];
				for (m = 0; m < nb; m++)
					if (fabs(bp[m].x - pts[j]) <= 1e-9 *
					    fmax(1, fabs(pts[j])))
						break;
				if (m < nb)
					continue;
				dx_at(&r, pts[j], &d);
				if (d.kind == DX_OK || d.kind == DX_UNDEF || nb >= max)
					continue;
				{
					struct sbuf sb = {0};
					const char *gn;
					int sl[C2MAXO];
					for (m = 0; m < k; m++)
						sl[m] = 0;
					if (k == 0)
						gn = fname2(f);
					else {
						sb_printf(&sb, "%s %s", dname(f, sl, k), fname2(f));
						gn = sb_str(&sb);
					}
					bp[nb].x = pts[j];
					bp[nb].why = dx_reason(f, gn, &d);
					/* dx_reason's buffer rotates: keep a copy */
					{
						size_t L = strlen(bp[nb].why) + 1;
						char *cp = xmalloc(L);
						memcpy(cp, bp[nb].why, L);
						bp[nb].why = cp;
					}
					free(sb.p);
					nb++;
				}
			}
		}
		lref_free(&r);
	}
	return nb;
}

/* "holds for x > 0 and x != 1", from f's domain and the bad points */
static void
print_holds(const struct func *f, const struct cond *cs, int nc,
    const struct badpt *bp, int nb, const char *lead)
{
	struct sbuf sb = {0};
	double pt[MAXCOND];
	int rel[MAXCOND], i, j, nterm = 0;
	char *dropped = xcalloc((size_t)nb + 1, 1);

	for (i = 0; i < nc; i++)
		(void)cond_text(f, &cs[i], &pt[i], &rel[i]);
	/* a bad point at the edge of x >= c makes it x > c */
	for (j = 0; j < nb; j++)
		for (i = 0; i < nc; i++) {
			if (rel[i] < 0 || isnan(bp[j].x) || fabs(pt[i] - bp[j].x) > 1e-9 *
			    fmax(1, fabs(pt[i])))
				continue;
			if (rel[i] == 1)
				rel[i] = 0, dropped[j] = 1;
			else if (rel[i] == 4)
				rel[i] = 3, dropped[j] = 1;
			else if (rel[i] == 0 || rel[i] == 3 || rel[i] == 2)
				dropped[j] = 1;
		}
	for (i = 0; i < nc; i++) {
		const char *t;
		if (rel[i] >= 0) {
			static char b[96];
			snprintf(b, sizeof b, "%s %s %s", pname(f, 0), relname[rel[i]],
			    fnum(pt[i]));
			t = b;
		} else
			t = cond_text(f, &cs[i], NULL, NULL);
		if (strstr(sb.p ? sb.p : "", t))
			continue;
		sb_printf(&sb, "%s%s", nterm++ ? " and " : "", t);
	}
	for (j = 0; j < nb; j++) {
		if (dropped[j])
			continue;
		if (isnan(bp[j].x))
			sb_printf(&sb, "%s%s", nterm++ ? " and " : "", bp[j].why);
		else
			sb_printf(&sb, "%s%s != %s", nterm++ ? " and " : "",
			    pname(f, 0), fnum(bp[j].x));
	}
	if (nterm)
		printf("  %s for %s\n", lead, sb.p);
	else
		printf("  %s for every real %s\n", lead, f->np == 1 ? pname(f, 0) :
		    "input");
	free(sb.p);
	free(dropped);
}

void
c2_deriv(struct func *f, const int *slots, int n)
{
	nid g[C2MAXO + 1];
	struct cond cs[MAXCOND];
	int k, nc, i, ok = 1, npts, allpts = 0;
	double lo, hi, worst = 0, maxrel;

	if (f->np == 0) {
		printf("%s is a constant: its derivative is 0\n", fname2(f));
		return;
	}
	if (n == 0) {
		if (f->np > 1) {
			struct sbuf sb = {0};
			for (k = 0; k < f->np; k++)
				sb_printf(&sb, "%s%s", k ? ", " : "", pname(f, k));
			printf("  note: %s has %d variables (%s): all first partial "
			    "derivatives, the gradient (say `diff F %s` for one)\n",
			    fname2(f), f->np, sb.p, pname(f, 0));
			free(sb.p);
			c2_grad(f, 0, NULL, 0);
			return;
		}
		n = 1;
		slots = (const int[]){0};
	}
	if (n > C2MAXO)
		fail("derivatives up to order %d", C2MAXO);
	for (k = 0; k < n; k++)
		if (!TY_FP(f->p[slots[k]].ty)) {
			if (n == 1) {
				cmd_diff(f, slots[0]);
				return;
			}
			fail("%s is an integer parameter: only its forward difference "
			    "(diff %s %s) is defined", pname(f, slots[k]), fname2(f),
			    pname(f, slots[k]));
		}
	g[0] = mk_cast(TY_REAL, f->body);
	for (k = 0; k < n; k++)
		g[k + 1] = dif(g[k], slots[k]);
	/* the intermediate orders of one variable, then the result */
	for (k = 1; k <= n; k++) {
		if (k < n && (f->np > 1 || n > 4))
			continue;
		printf("%s %s = %s\n", dname(f, slots, k), fname2(f),
		    node_str(g[k]));
	}
	nc = domain_conds(f->body, cs, 0);
	if (f->np == 1) {
		struct badpt bp[16];
		double cpts[64];
		int nb, ncp;
		check_interval(f, 0, &lo, &hi);
		nb = find_bad(f, g, n, lo, hi, bp, 16, cpts, &ncp);
		print_holds(f, cs, nc, bp, nb, "holds");
		for (i = 0; i < nb; i++)
			if (isnan(bp[i].x))
				printf("    where %s is an integer it jumps (no derivative "
				    "there)\n", node_str(kink_int_arg(g, n)));
			else
				printf("    %s\n", bp[i].why);
		if (nb == 0 && ncp > 0) {
			struct sbuf sb = {0};
			for (i = 0; i < ncp && i < 6; i++)
				sb_printf(&sb, "%s%s", i ? ", " : "", fnum(cpts[i]));
			printf("    (smooth at the suspect point%s %s = %s%s: the "
			    "one-sided slopes agree)\n", ncp > 1 ? "s" : "",
			    pname(f, 0), sb.p, ncp > 6 ? ", ..." : "");
			free(sb.p);
		}
		for (i = 0; i < nb; i++)
			free((void *)bp[i].why);
		if (ncp > 0 || nb > 0)
			printf("    (points looked for in [%s, %s])\n", fnum(lo),
			    fnum(hi));
		for (k = 0; k < n; k++) {
			ok &= check1(g[k], g[k + 1], 0, lo, hi, cpts, ncp, &npts,
			    &maxrel);
			allpts += npts;
			worst = fmax(worst, maxrel);
		}
	} else {
		struct cand c[MAXCAND];
		int ncand = 0;
		struct sbuf sb = {0};
		for (k = 0; k < n; k++)
			ncand = kink_cands(g[k], c, ncand);
		for (i = 0; i < nc; i++)
			sb_printf(&sb, "%s%s", i ? " and " : "",
			    cond_text(f, &cs[i], NULL, NULL));
		printf("  holds %s%s\n", nc ? "for " : "wherever ", nc ? sb.p :
		    "the formula is defined");
		for (i = 0; i < ncand; i++)
			printf("    may fail where %s = 0 (%s)\n", node_str(c[i].u),
			    c[i].kind == KK_PIECE ? "a piece boundary" :
			    c[i].kind == KK_JUMP || c[i].kind == KK_INT ? "a jump" :
			    c[i].kind == KK_VERT ? "an infinite slope" : "a corner");
		free(sb.p);
		for (k = 0; k < n; k++) {
			ok &= checkn(f, g[k], g[k + 1], slots[k], &npts, &maxrel);
			allpts += npts;
			worst = fmax(worst, maxrel);
		}
	}
	if (allpts == 0)
		printf("  check: no sample point where it is defined\n");
	else if (ok)
		printf("  check: %s central differences at %d points "
		    "(max relative difference %.1g)\n", n > 1 ?
		    "each order matches" : "matches", allpts, worst);
	else
		printf("  check FAILED: differs from central differences by up "
		    "to %.3g (relative)\n", worst);
}

/* ---- gradient, Hessian, Laplacian, Jacobian, directional ---- */

/* the determinant of an n x n matrix of nodes, by cofactors */
static nid
det_node(const nid *m, int n, int stride)
{
	nid r = 0, minor[C2MAXF * C2MAXF], t;
	int c, i, j, k;

	if (n == 1)
		return m[0];
	if (n == 2)
		return SUB(MUL(m[0], m[stride + 1]), MUL(m[1], m[stride]));
	for (c = 0; c < n; c++) {
		if (const_is(m[c], 0))
			continue;
		for (i = 1, k = 0; i < n; i++)
			for (j = 0; j < n; j++)
				if (j != c)
					minor[k++] = m[i * stride + j];
		t = MUL(m[c], det_node(minor, n - 1, n - 1));
		r = r == 0 ? ((c & 1) ? mk_un(O_NEG, t) : t) :
		    (c & 1) ? SUB(r, t) : ADD(r, t);
	}
	return r ? r : KR(0);
}

/* numeric determinant (Gaussian elimination with pivoting) */
static double
det_num(const double *m0, int n)
{
	double m[C2MAXF * C2MAXF], d = 1, t;
	int i, j, k, p;

	memcpy(m, m0, sizeof(double) * (size_t)n * n);
	for (k = 0; k < n; k++) {
		p = k;
		for (i = k + 1; i < n; i++)
			if (fabs(m[i * n + k]) > fabs(m[p * n + k]))
				p = i;
		if (m[p * n + k] == 0)
			return 0;
		if (p != k) {
			for (j = 0; j < n; j++) {
				t = m[k * n + j];
				m[k * n + j] = m[p * n + j];
				m[p * n + j] = t;
			}
			d = -d;
		}
		d *= m[k * n + k];
		for (i = k + 1; i < n; i++) {
			t = m[i * n + k] / m[k * n + k];
			for (j = k; j < n; j++)
				m[i * n + j] -= t * m[k * n + j];
		}
	}
	return d;
}

/* print rows of node entries as an aligned matrix */
static void
print_matrix(const nid *m, int rows, int cols, const char *const *rowlab)
{
	size_t w[C2MAXF] = {0}, L;
	int i, j;
	char *s[C2MAXF * C2MAXF];

	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++) {
			const char *t = node_str(m[i * cols + j]);
			L = strlen(t);
			s[i * cols + j] = xmalloc(L + 1);
			memcpy(s[i * cols + j], t, L + 1);
			if (L > w[j])
				w[j] = L;
		}
	for (i = 0; i < rows; i++) {
		printf("  [ ");
		for (j = 0; j < cols; j++) {
			printf("%-*s%s", (int)w[j], s[i * cols + j],
			    j + 1 < cols ? "   " : " ]");
			free(s[i * cols + j]);
		}
		if (rowlab && rowlab[i])
			printf("   %s", rowlab[i]);
		putchar('\n');
	}
}

static void
print_point(const struct func *f, const double *at, int np)
{
	int k;

	putchar('(');
	for (k = 0; k < np; k++)
		printf("%s%s = %s", k ? ", " : "", pname(f, k), fnum(at[k]));
	putchar(')');
}

static double
val_at(nid e, const struct func *f, const double *at, int *ok)
{
	struct mev m;
	double v;

	mev_open(&m, e);
	v = mev_at(&m, at, f->np, ok);
	mev_close(&m);
	return v;
}

static void
check_report(int ok, int npts, double worst)
{
	if (npts == 0)
		printf("  check: no sample point where it is defined\n");
	else if (ok)
		printf("  check: every entry matches central differences at %d "
		    "points (max relative difference %.1g)\n", npts, worst);
	else
		printf("  check FAILED: an entry differs from central differences "
		    "by up to %.3g (relative)\n", worst);
}

/* the conditions of f's domain, in words */
static void
holds_line(const struct func *f, nid extra)
{
	struct cond cs[MAXCOND];
	struct cand c[MAXCAND];
	struct sbuf sb = {0};
	int nc = domain_conds(f->body, cs, 0), ncand, i;

	ncand = kink_cands(extra ? extra : f->body, c, 0);
	for (i = 0; i < nc; i++)
		sb_printf(&sb, "%s%s", i ? " and " : "", cond_text(f, &cs[i],
		    NULL, NULL));
	printf("  holds %s%s\n", nc ? "for " : "wherever ", nc ? sb.p :
	    "the formulas are defined");
	for (i = 0; i < ncand; i++)
		printf("    may fail where %s = 0 (%s)\n", node_str(c[i].u),
		    c[i].kind == KK_PIECE ? "a piece boundary" :
		    c[i].kind == KK_JUMP || c[i].kind == KK_INT ? "a jump" :
		    c[i].kind == KK_VERT ? "an infinite slope" : "a corner");
	free(sb.p);
}

static int
check_any(const struct func *f, nid g, nid d, int slot, int *npts,
    double *maxrel)
{
	double lo, hi;

	if (f->np == 1) {
		check_interval(f, 0, &lo, &hi);
		return check1(g, d, 0, lo, hi, NULL, 0, npts, maxrel);
	}
	return checkn(f, g, d, slot, npts, maxrel);
}

void
c2_grad(struct func *f, int what, const double *at, int hasat)
{
	nid g[MAXP], H[MAXP * MAXP], lap = 0, body = mk_cast(TY_REAL, f->body);
	int n = f->np, i, j, ok = 1, npts, all = 0, o;
	double worst = 0, mr;
	struct sbuf sb = {0};

	if (n == 0) {
		printf("%s is a constant: every derivative is 0\n", fname2(f));
		return;
	}
	for (i = 0; i < n; i++)
		if (!TY_FP(f->p[i].ty))
			fail("%s is an integer parameter: no partial derivative "
			    "(diff %s %s gives the forward difference)", pname(f, i),
			    fname2(f), pname(f, i));
	if (hasat)
		for (i = 0; i < n; i++)
			if (!isfinite(at[i]))
				fail("the point needs finite coordinates");
	for (i = 0; i < n; i++)
		g[i] = dif(body, i);
	if (what == 0) {
		for (i = 0; i < n; i++)
			sb_printf(&sb, "%s%s", i ? ", " : "", node_str(g[i]));
		printf("\xe2\x88\x87 %s = (%s)\n", fname2(f), sb.p);
		if (n > 1)
			for (i = 0; i < n; i++)
				printf("  %s %s = %s\n", dname(f, &i, 1), fname2(f),
				    node_str(g[i]));
		holds_line(f, 0);
		for (i = 0; i < n; i++) {
			ok &= check_any(f, body, g[i], i, &npts, &mr);
			all += npts;
			worst = fmax(worst, mr);
		}
		check_report(ok, all, worst);
		if (hasat) {
			double v, len = 0;
			printf("  at ");
			print_point(f, at, n);
			printf(": (");
			for (i = 0; i < n; i++) {
				v = val_at(g[i], f, at, &o);
				printf("%s%s", i ? ", " : "", o ? fnum(v) : "undefined");
				len += v * v;
			}
			printf("), |\xe2\x88\x87| = %s\n", fnum(sqrt(len)));
		}
		free(sb.p);
		return;
	}
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			H[i * n + j] = dif(g[i], j);
	if (what == 2) {
		for (i = 0; i < n; i++)
			lap = lap ? ADD(lap, H[i * n + i]) : H[i * n + i];
		lap = simp(lap, 0);
		printf("\xe2\x88\x87\xc2\xb2 %s = %s\n", fname2(f), node_str(lap));
		for (i = 0; i < n; i++) {
			int s2[2] = {i, i};
			printf("  %s %s = %s\n", dname(f, s2, 2), fname2(f),
			    node_str(H[i * n + i]));
		}
		holds_line(f, 0);
		for (i = 0; i < n; i++) {
			ok &= check_any(f, body, g[i], i, &npts, &mr);
			all += npts;
			worst = fmax(worst, mr);
			ok &= check_any(f, g[i], H[i * n + i], i, &npts, &mr);
			all += npts;
			worst = fmax(worst, mr);
		}
		check_report(ok, all, worst);
		{
			/* harmonic? sample the Laplacian */
			double x[MAXP], mx = 0;
			int cnt = 0;
			struct mev m;
			mev_open(&m, lap);
			for (i = 0; i < 200; i++) {
				double v;
				sample_point(f, i, x);
				v = mev_at(&m, x, n, &o);
				if (o) {
					cnt++;
					mx = fmax(mx, fabs(v));
				}
			}
			mev_close(&m);
			if (cnt && mx <= 1e-12)
				printf("  %s is harmonic: its Laplacian is 0 at all %d "
				    "sample points (and symbolically %s)\n", fname2(f), cnt,
				    const_is(lap, 0) ? "0" : "not visibly 0");
		}
		if (hasat) {
			double v = val_at(lap, f, at, &o);
			printf("  at ");
			print_point(f, at, n);
			printf(": %s\n", o ? fnum(v) : "undefined");
		}
		return;
	}
	/* the Hessian */
	{
		const char *lab[MAXP];
		char lb[MAXP][64];
		int nsym = 0, asym = 0;
		double x[MAXP];
		for (i = 0; i < n; i++) {
			snprintf(lb[i], sizeof lb[i], "\xe2\x88\x82/\xe2\x88\x82%s",
			    pname(f, i));
			lab[i] = lb[i];
		}
		printf("Hessian of %s (row i, column j: \xe2\x88\x82\xc2\xb2/"
		    "\xe2\x88\x82j\xe2\x88\x82i):\n", fname2(f));
		print_matrix(H, n, n, n > 1 ? lab : NULL);
		if (n > 1 && n <= 4) {
			nid d = simp(det_node(H, n, n), 0);
			printf("  det = %s\n", node_str(d));
		}
		holds_line(f, 0);
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++) {
				ok &= check_any(f, g[i], H[i * n + j], j, &npts, &mr);
				all += npts;
				worst = fmax(worst, mr);
			}
		check_report(ok, all, worst);
		/* Schwarz: the mixed partials agree */
		for (int s = 0; s < 100 && n > 1; s++) {
			sample_point(f, s, x);
			for (i = 0; i < n; i++)
				for (j = i + 1; j < n; j++) {
					int o1, o2;
					double a = val_at(H[i * n + j], f, x, &o1);
					double b = val_at(H[j * n + i], f, x, &o2);
					if (!o1 || !o2)
						continue;
					nsym++;
					if (fabs(a - b) > 1e-9 * fmax(1, fabs(a)))
						asym++;
				}
		}
		if (n > 1 && nsym)
			printf("  symmetric (the mixed partials agree) at %d of %d "
			    "sample pairs%s\n", nsym - asym, nsym, asym ?
			    ": NOT everywhere (the second partials are not "
			    "continuous)" : "");
		if (hasat) {
			double M[MAXP * MAXP], gv[MAXP], gl = 0, minor[MAXP];
			int bad = 0, k, pos = 1, neg = 1;
			for (i = 0; i < n; i++) {
				gv[i] = val_at(g[i], f, at, &o);
				bad |= !o;
				gl = fmax(gl, fabs(gv[i]));
				for (j = 0; j < n; j++) {
					M[i * n + j] = val_at(H[i * n + j], f, at, &o);
					bad |= !o;
				}
			}
			printf("  at ");
			print_point(f, at, n);
			if (bad) {
				printf(": undefined\n");
				return;
			}
			printf(":\n");
			for (i = 0; i < n; i++) {
				printf("  [ ");
				for (j = 0; j < n; j++)
					printf("%-10s%s", fnum(M[i * n + j]), j + 1 < n ? " " :
					    " ]\n");
			}
			printf("  det = %s\n", fnum(det_num(M, n)));
			for (k = 1; k <= n; k++) {
				double sub[MAXP * MAXP];
				int a2, b2;
				for (a2 = 0; a2 < k; a2++)
					for (b2 = 0; b2 < k; b2++)
						sub[a2 * k + b2] = M[a2 * n + b2];
				minor[k - 1] = det_num(sub, k);
				if (!(minor[k - 1] > 1e-12))
					pos = 0;
				if (!((k & 1) ? minor[k - 1] < -1e-12 :
				    minor[k - 1] > 1e-12))
					neg = 0;
			}
			if (gl > 1e-9)
				printf("  the gradient is not 0 there (largest component "
				    "%s): not a critical point\n", fnum(gl));
			else if (pos)
				printf("  the gradient is 0 and the Hessian is positive "
				    "definite (leading minors > 0): a local minimum\n");
			else if (neg)
				printf("  the gradient is 0 and the Hessian is negative "
				    "definite (leading minors alternate, starting < 0): a "
				    "local maximum\n");
			else if (fabs(det_num(M, n)) > 1e-12)
				printf("  the gradient is 0 and the Hessian is indefinite: "
				    "a saddle point\n");
			else
				printf("  the gradient is 0 but the Hessian is singular: "
				    "the second-derivative test is inconclusive\n");
		}
	}
}

void
c2_jacobian(struct func *f, const nid *fs, int nf, const int *vars, int nv,
    const double *at, int hasat)
{
	nid J[C2MAXF * C2MAXF];
	int i, j, ok = 1, npts, all = 0, o;
	double worst = 0, mr, M[C2MAXF * C2MAXF];
	const char *lab[C2MAXF];

	if (nf > C2MAXF || nv > C2MAXF)
		fail("at most %d functions and variables", C2MAXF);
	if (nv == 0)
		fail("the functions have no variables: the Jacobian is empty");
	for (j = 0; j < nv; j++)
		if (!TY_FP(f->p[vars[j]].ty))
			fail("%s is an integer parameter: no partial derivative",
			    pname(f, vars[j]));
	for (i = 0; i < nf; i++) {
		lab[i] = NULL;
		for (j = 0; j < nv; j++)
			J[i * nv + j] = dif(fs[i], vars[j]);
	}
	printf("Jacobian of (");
	for (i = 0; i < nf; i++)
		printf("%s%s", i ? ", " : "", node_str(fs[i]));
	printf(") wrt (");
	for (j = 0; j < nv; j++)
		printf("%s%s", j ? ", " : "", pname(f, vars[j]));
	printf("): row i = the gradient of function i\n");
	print_matrix(J, nf, nv, lab);
	if (nf == nv && nf > 1 && nf <= 4)
		printf("  det = %s\n", node_str(simp(det_node(J, nf, nv), 0)));
	{
		struct func tmp = *f;
		nid all_b = fs[0];
		for (i = 1; i < nf; i++)
			all_b = ADD(all_b, fs[i]);   /* only for the conditions */
		tmp.body = all_b;
		holds_line(&tmp, 0);
		for (i = 0; i < nf; i++)
			for (j = 0; j < nv; j++) {
				ok &= check_any(&tmp, fs[i], J[i * nv + j], vars[j], &npts,
				    &mr);
				all += npts;
				worst = fmax(worst, mr);
			}
	}
	check_report(ok, all, worst);
	if (hasat) {
		int bad = 0;
		printf("  at ");
		print_point(f, at, f->np);
		printf(":\n");
		for (i = 0; i < nf; i++) {
			printf("  [ ");
			for (j = 0; j < nv; j++) {
				M[i * nv + j] = val_at(J[i * nv + j], f, at, &o);
				bad |= !o;
				printf("%-10s%s", o ? fnum(M[i * nv + j]) : "undefined",
				    j + 1 < nv ? " " : " ]\n");
			}
		}
		if (nf == nv && !bad) {
			double d = det_num(M, nf);
			printf("  det = %s%s\n", fnum(d), fabs(d) > 1e-12 ?
			    " (nonzero: the map is locally invertible there)" :
			    " (singular there)");
		}
	}
}

void
c2_ddir(struct func *f, const double *v, int nv, const double *at,
    int hasat, int raw)
{
	nid body = mk_cast(TY_REAL, f->body), D = 0;
	double len = 0, u[MAXP], x[MAXP], worst = 0;
	int i, n = f->np, cnt = 0, bad = 0, o;
	struct mev mf, md;

	if (nv != n)
		fail("%s has %d variable%s; the direction has %d component%s",
		    fname2(f), n, n == 1 ? "" : "s", nv, nv == 1 ? "" : "s");
	for (i = 0; i < n; i++)
		len += v[i] * v[i];
	len = sqrt(len);
	if (len == 0)
		fail("the direction is the zero vector");
	for (i = 0; i < n; i++) {
		nid t;
		if (!TY_FP(f->p[i].ty))
			fail("%s is an integer parameter", pname(f, i));
		u[i] = raw ? v[i] : v[i] / len;
		if (u[i] == 0)
			continue;
		t = MUL(KR(u[i]), dif(body, i));
		D = D ? ADD(D, t) : t;
	}
	D = simp(D, 0);
	printf("D_u %s = %s\n", fname2(f), node_str(D));
	if (raw)
		printf("  (\xe2\x88\x87%s \xc2\xb7 v with v as given, not "
		    "normalized)\n", fname2(f));
	else {
		printf("  u = v/|v| = (");
		for (i = 0; i < n; i++)
			printf("%s%s", i ? ", " : "", fnum(u[i]));
		printf("), |v| = %s", fnum(len));
		printf("  (the rate of change per unit length; `raw` for "
		    "\xe2\x88\x87%s \xc2\xb7 v)\n", fname2(f));
	}
	holds_line(f, 0);
	mev_open(&mf, body);
	mev_open(&md, D);
	for (int s = 0; s < 300 && cnt < 100; s++) {
		double h = 1e-3, a1, b1, a2, b2, dv, R, D1, D2, sc;
		int ok1, ok2, ok3, ok4, ok5;
		sample_point(f, s, x);
		{
			double y[MAXP];
			for (i = 0; i < n; i++)
				y[i] = x[i] + h * u[i];
			a1 = mev_at(&mf, y, n, &ok1);
			for (i = 0; i < n; i++)
				y[i] = x[i] - h * u[i];
			b1 = mev_at(&mf, y, n, &ok2);
			for (i = 0; i < n; i++)
				y[i] = x[i] + h / 2 * u[i];
			a2 = mev_at(&mf, y, n, &ok3);
			for (i = 0; i < n; i++)
				y[i] = x[i] - h / 2 * u[i];
			b2 = mev_at(&mf, y, n, &ok4);
		}
		dv = mev_at(&md, x, n, &ok5);
		if (!ok1 || !ok2 || !ok3 || !ok4 || !ok5)
			continue;
		D1 = (a1 - b1) / (2 * h);
		D2 = (a2 - b2) / h;
		R = (4 * D2 - D1) / 3;
		sc = fmax(1, fabs(dv));
		cnt++;
		if (fabs(R - dv) > 1e-6 * sc + fabs(D2 - D1) / 5)
			bad++;
		worst = fmax(worst, fabs(R - dv) / sc);
	}
	mev_close(&mf);
	mev_close(&md);
	if (cnt == 0)
		printf("  check: no sample point where it is defined\n");
	else if (!bad)
		printf("  check: matches (f(p + h u) - f(p - h u))/2h at %d points "
		    "(max relative difference %.1g)\n", cnt, worst);
	else
		printf("  check FAILED: differs from the difference quotient by up "
		    "to %.3g\n", worst);
	if (hasat) {
		double dv = val_at(D, f, at, &o);
		printf("  at ");
		print_point(f, at, n);
		printf(": %s\n", o ? fnum(dv) : "undefined");
	}
}

/* ---- the Wronskian ---- */

/* built only from operations that are analytic where defined */
static int
analytic(nid e)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0;
	int ok = 1;

	st[sp++] = e;
	mark[e] = 1;
	while (sp && ok) {
		g = st[--sp];
		switch (N.op[g]) {
		case O_CONST: case O_VAR:
			continue;
		case O_NEG: case O_ADD: case O_SUB: case O_MUL: case O_DIV:
		case O_POW: case O_EXP: case O_LOG: case O_SIN: case O_COS:
		case O_TAN: case O_SQRT: case O_SEC: case O_CSC: case O_COT:
		case O_ATAN: case O_LAMBERTW: case O_LAMBERTWM1:
		case O_ASIN: case O_ACOS: case O_ATAN2: case O_SINH:
		case O_COSH: case O_TANH: case O_ASINH: case O_ACOSH:
		case O_ATANH:	/* forms.c */
			break;
		case O_CAST:
			if (!TY_FP(N.ty[g]))
				ok = 0;
			break;
		default:
			ok = 0;
		}
		if (N.a[g] && !mark[N.a[g]])
			mark[N.a[g]] = 1, st[sp++] = N.a[g];
		if (N.b[g] && !mark[N.b[g]])
			mark[N.b[g]] = 1, st[sp++] = N.b[g];
	}
	free(mark);
	free(st);
	return ok;
}

/* a rational with a small denominator close to v, else v */
static double
snap(double v)
{
	struct q q;

	if (fabs(v) < 1e-12)
		return 0;
	if (q_from_double(v, &q) && q.d <= 360 &&
	    fabs(q_val(q) - v) <= 1e-9 * fmax(1, fabs(v)))
		return q_val(q);
	return v;
}

/*
 * A simpler form of a one-variable function known only by values:
 * a constant, c exp(a x), or a polynomial with rational coefficients,
 * checked at every point.
 */
static nid
recognize(const double *xs, const double *ys, int m, nid xv)
{
	int i, d, ok;
	double c, a, sc = 0;

	if (m < 8)
		return 0;
	for (i = 0; i < m; i++)
		sc = fmax(sc, fabs(ys[i]));
	if (sc == 0)
		return KR(0);
	/* constant */
	c = snap(ys[0]);
	for (i = 0, ok = 1; i < m; i++)
		if (fabs(ys[i] - c) > 1e-9 * fmax(1, fabs(c)))
			ok = 0;
	if (ok)
		return KR(c);
	/* c exp(a x): log|y| linear, the sign fixed */
	for (i = 0, ok = 1; i < m; i++)
		if (ys[i] == 0 || (ys[i] > 0) != (ys[0] > 0))
			ok = 0;
	if (ok) {
		double l0 = log(fabs(ys[0])), l1 = log(fabs(ys[m - 1]));
		a = snap((l1 - l0) / (xs[m - 1] - xs[0]));
		c = snap(ys[0] / exp(a * xs[0]));
		for (i = 0; i < m; i++)
			if (fabs(ys[i] - c * exp(a * xs[i])) > 1e-8 * fabs(ys[i]))
				ok = 0;
		if (ok && a != 0)
			return MUL(KR(c), mk_un(O_EXP, MUL(KR(a), xv)));
	}
	/* a polynomial of degree <= 8 */
	for (d = 1; d <= 8; d++) {
		double A[64 * 9], b[64], co[9];
		int mm = m < 64 ? m : 64, j;
		for (i = 0; i < mm; i++) {
			double p = 1;
			for (j = 0; j <= d; j++) {
				A[i * (d + 1) + j] = p;
				p *= xs[i];
			}
			b[i] = ys[i];
		}
		if (!solve_lsq(A, b, mm, d + 1, co))
			continue;
		for (j = 0; j <= d; j++)
			co[j] = snap(co[j]);
		for (i = 0, ok = 1; i < m && ok; i++) {
			double p = 0;
			for (j = d; j >= 0; j--)
				p = p * xs[i] + co[j];
			if (fabs(p - ys[i]) > 1e-9 * fmax(1, sc))
				ok = 0;
		}
		if (ok) {
			nid r = 0;
			for (j = d; j >= 0; j--) {
				nid t;
				if (co[j] == 0)
					continue;
				t = j == 0 ? KR(co[j]) : MUL(KR(co[j]), j == 1 ? xv :
				    POW(xv, KI(j)));
				r = r ? ADD(r, t) : t;
			}
			return r ? tidy(r) : KR(0);
		}
	}
	return 0;
}

void
c2_wronskian(struct func *f, const nid *fs, int nf, int slot)
{
	nid M[C2MAXF * C2MAXF], W;
	const char *lab[C2MAXF];
	char lb[C2MAXF][32];
	int i, j, k, ok = 1, npts, all = 0, m = 0, o, nzero = 0, okv;
	int detbad = 0;
	double worst = 0, mr, lo, hi, xs[256], ws[256], x[MAXP], wmax = 0,
	    xw = 0, wv = 0;
	struct mev mw;

	if (nf < 2)
		fail("the Wronskian needs at least two functions");
	if (nf > C2MAXF)
		fail("at most %d functions", C2MAXF);
	if (f->np == 0)
		fail("the functions are constants: name the variable (wrt x)");
	if (!TY_FP(f->p[slot].ty))
		fail("%s is an integer parameter", pname(f, slot));
	for (j = 0; j < nf; j++)
		M[j] = mk_cast(TY_REAL, fs[j]);
	for (k = 1; k < nf; k++)
		for (j = 0; j < nf; j++)
			M[k * nf + j] = dif(M[(k - 1) * nf + j], slot);
	for (k = 0; k < nf; k++) {
		if (k == 0)
			snprintf(lb[k], sizeof lb[k], "the functions");
		else {
			struct sbuf sb = {0};
			int sl[C2MAXF];
			for (i = 0; i < k; i++)
				sl[i] = slot;
			{
				struct func t1 = *f;
				t1.np = 1;   /* d/dx notation */
				t1.p[0] = f->p[slot];
				sl[0] = 0;
				for (i = 0; i < k; i++)
					sl[i] = 0;
				sb_printf(&sb, "%s", dname(&t1, sl, k));
			}
			snprintf(lb[k], sizeof lb[k], "%s", sb.p);
			free(sb.p);
		}
		lab[k] = lb[k];
	}
	printf("Wronskian matrix of ");
	for (j = 0; j < nf; j++)
		printf("%s%s", j ? ", " : "", node_str(fs[j]));
	printf(" (row k: the k-th derivatives wrt %s)\n", pname(f, slot));
	print_matrix(M, nf, nf, lab);
	W = simp(det_node(M, nf, nf), 0);
	printf("  W = det = %s\n", node_str(W));
	for (k = 1; k < nf; k++)
		for (j = 0; j < nf; j++) {
			ok &= check_any(f, M[(k - 1) * nf + j], M[k * nf + j], slot,
			    &npts, &mr);
			all += npts;
			worst = fmax(worst, mr);
		}
	/* values of W, and the determinant of the numeric matrix */
	check_interval(f, slot, &lo, &hi);
	mev_open(&mw, W);
	{
		struct mev me[C2MAXF * C2MAXF];
		for (i = 0; i < nf * nf; i++)
			mev_open(&me[i], M[i]);
		for (i = 0; i < 200; i++) {
			double v, E[C2MAXF * C2MAXF], dn, sc = 1;
			sample_point(f, i, x);
			if (f->np == 1 || !f->p[slot].ranged)
				x[slot] = lo + (hi - lo) * (i + 0.5) / 200;
			v = mev_at(&mw, x, f->np, &o);
			if (!o)
				continue;
			for (k = 0, okv = 1; k < nf * nf; k++) {
				E[k] = mev_at(&me[k], x, f->np, &o);
				okv &= o;
			}
			if (!okv)
				continue;
			for (j = 0; j < nf; j++) {
				double cm = 0;
				for (k = 0; k < nf; k++)
					cm = fmax(cm, fabs(E[k * nf + j]));
				sc *= fmax(cm, 1e-300);
			}
			dn = det_num(E, nf);
			if (fabs(dn - v) > 1e-8 * fmax(sc, fabs(v)))
				detbad++;
			xs[m] = x[slot];
			ws[m] = v;
			m++;
			if (fabs(v) <= 1e-10 * sc)
				nzero++;
			else if (wmax == 0 || fabs(x[slot]) < fabs(xw)) {
				wmax = 1;
				xw = x[slot];
				wv = v;
			}
		}
		for (i = 0; i < nf * nf; i++)
			mev_close(&me[i]);
		if (detbad)
			printf("  check FAILED: the symbolic determinant differs from "
			    "the numeric one at %d points\n", detbad);
	}
	mev_close(&mw);
	if (f->np == 1 && m >= 8) {
		nid sw = recognize(xs, ws, m, mk_var(0, f->p[0].name, TY_REAL));
		if (sw && sw != W && strcmp(node_str(sw), node_str(W)))
			printf("  W = %s  (equal to the determinant at all %d sample "
			    "points of [%s, %s])\n", node_str(sw), m, fnum(lo),
			    fnum(hi));
		else if (sw && !detbad)
			printf("  (W equals the numeric determinant at all %d sample "
			    "points of [%s, %s])\n", m, fnum(lo), fnum(hi));
	}
	check_report(ok, all, worst);
	if (m == 0) {
		printf("  W is undefined at every sample point\n");
		return;
	}
	if (nzero < m) {
		printf("  linearly independent: W(%s) = %s != 0, so the functions "
		    "are independent on every interval containing %s = %s\n",
		    fnum(xw), fnum(wv), pname(f, slot), fnum(xw));
		if (nzero)
			printf("  (W vanishes at %d of the %d sample points)\n", nzero,
			    m);
		return;
	}
	/* W vanishes at every sample: look for a linear relation */
	{
		double A[200 * C2MAXF], b[200], c[C2MAXF], res = 0, sc = 0;
		int mm = 0, an = 1, r;
		struct mev mf[C2MAXF];
		for (j = 0; j < nf; j++) {
			mev_open(&mf[j], fs[j]);
			an &= analytic(fs[j]);
		}
		for (i = 0; i < 200; i++) {
			double v[C2MAXF];
			sample_point(f, i, x);
			if (f->np == 1 || !f->p[slot].ranged)
				x[slot] = lo + (hi - lo) * (i + 0.5) / 200;
			for (j = 0, okv = 1; j < nf; j++) {
				v[j] = mev_at(&mf[j], x, f->np, &o);
				okv &= o;
			}
			if (!okv)
				continue;
			/* c_0 f_0 + ... = 0 with c_{nf-1} = 1 tried first */
			for (j = 0; j < nf - 1; j++)
				A[mm * (nf - 1) + j] = v[j];
			b[mm] = -v[nf - 1];
			sc = fmax(sc, fabs(v[nf - 1]));
			mm++;
		}
		r = mm >= nf && solve_lsq(A, b, mm, nf - 1, c);
		if (r) {
			for (i = 0; i < mm; i++) {
				double s = -b[i];
				for (j = 0; j < nf - 1; j++)
					s += c[j] * A[i * (nf - 1) + j];
				res = fmax(res, fabs(s));
			}
		}
		for (j = 0; j < nf; j++)
			mev_close(&mf[j]);
		printf("  W = 0 at all %d sample points\n", m);
		if (r && res <= 1e-9 * fmax(1, sc)) {
			int first = 1;
			printf("  a linear relation holds at all %d points: ", mm);
			for (j = 0; j < nf - 1; j++) {
				double cj = snap(c[j]);
				if (cj == 0)
					continue;
				printf("%s", first ? (cj < 0 ? "-" : "") : cj < 0 ? " - " :
				    " + ");
				if (fabs(cj) != 1)
					printf("%s*", fnum(fabs(cj)));
				printf("(%s)", node_str(fs[j]));
				first = 0;
			}
			printf("%s(%s) = 0\n", first ? "" : " + ", node_str(fs[nf - 1]));
			printf("  %s\n", an ? "linearly dependent (the functions are "
			    "analytic, so W = 0 everywhere implies dependence; the "
			    "relation confirms it)" : "linearly dependent on the "
			    "sampled interval (the relation above)");
		} else if (an)
			printf("  the functions are analytic, so W = 0 on an interval "
			    "means they are linearly dependent (but no relation with "
			    "the last coefficient 1 was found: it involves the "
			    "others only)\n");
		else
			printf("  W = 0 does not prove dependence for functions that "
			    "are not analytic (x^2 and x*abs(x) have W = 0 and are "
			    "independent), and no linear relation fits their values: "
			    "independent on [%s, %s]\n", fnum(lo), fnum(hi));
	}
}

/* ---- Taylor series ---- */

void
c2_taylor(struct func *f, int slot, double a, int n)
{
	nid g[C2MAXO + 2], c[C2MAXO + 2], P = 0, xv, map[NSLOT];
	int k, i, allnum = 1;
	double fact = 1, cv[C2MAXO + 2];
	const char *x;

	if (f->np == 0)
		fail("%s is a constant: its series is itself", fname2(f));
	if (n < 0 || n > C2MAXO)
		fail("order 0 to %d", C2MAXO);
	if (!TY_FP(f->p[slot].ty))
		fail("%s is an integer parameter", pname(f, slot));
	if (!isfinite(a))
		fail("a Taylor series is about a finite point (for x -> inf, "
		    "expand f(1/t) at t = 0)");
	x = pname(f, slot);
	xv = mk_var((uint32_t)slot, f->p[slot].name, TY_REAL);
	g[0] = mk_cast(TY_REAL, f->body);
	for (k = 1; k <= n; k++)
		g[k] = dif(g[k - 1], slot);
	memset(map, 0, sizeof map);
	map[slot] = KR(a);
	/* the derivatives must exist at a: check the kinks there (1 var) */
	if (f->np == 1) {
		int an = analytic(g[0]);
		for (k = 0; k < n; k++) {
			struct lref r;
			struct dx d;
			lr_open(&r, g[k], 0);
			dx_at(&r, a, &d);
			lref_free(&r);
			/* analytic where defined: unsettled difference
			 * quotients of a high derivative are rounding noise */
			if (d.kind != DX_OK && !(an && (d.kind == DX_OSC ||
			    d.kind == DX_CORNER))) {
				printf("no Taylor series of %s at %s = %s: ", fname2(f), x,
				    fnum(a));
				if (d.kind == DX_UNDEF)
					printf("%s is undefined there\n", k ? "a derivative" :
					    "it");
				else
					printf("%s\n", dx_reason(f, k ? "a derivative" :
					    fname2(f), &d));
				return;
			}
		}
	}
	for (k = 0; k <= n; k++) {
		nid v = tidy(subst(g[k], map, NSLOT));
		if (k)
			fact *= k;
		c[k] = tidy(DIV(v, KR(fact)));
		if (N.op[c[k]] == O_UNDEF) {
			printf("no Taylor series of %s at %s = %s: the %d%s "
			    "derivative is undefined there\n", fname2(f), x, fnum(a),
			    k, k == 1 ? "st" : k == 2 ? "nd" : k == 3 ? "rd" : "th");
			return;
		}
		if (!num_of(c[k], &cv[k]))
			allnum = 0;
		else
			c[k] = KR(snap(cv[k]));
	}
	{
		/* the polynomial as text, ascending: x - x^3/6 + x^5/120 */
		struct sbuf sb = {0}, cl = {0};
		char T[96];
		const char *dx = a == 0 ? x : node_str(SUB(xv, KR(a)));
		double kf = 1;
		int first = 1;
		for (k = 0; k <= n; k++) {
			nid t, dxn;
			int64_t pn = 0, qd = 1;
			if (k)
				kf *= k;
			if (k == 0)
				T[0] = 0;
			else if (k == 1)
				snprintf(T, sizeof T, a == 0 ? "%s" : "(%s)", dx);
			else if (a == 0)
				snprintf(T, sizeof T, "%s^%d", x, k);
			else
				snprintf(T, sizeof T, "(%s)^%d", dx, k);
			if (allnum) {
				struct q qq;
				double dv = cv[k] * kf;
				int rat = q_from_double(dv, &qq) && qq.d <= 1000 &&
				    fabs(q_val(qq) - dv) <= 1e-12 * fmax(1, fabs(dv)) &&
				    kf < 1e15;
				if (rat) {
					int64_t g;
					pn = qq.n;
					qd = qq.d * (int64_t)kf;
					g = gcd64(pn < 0 ? -pn : pn, qd);
					if (g > 1)
						pn /= g, qd /= g;
					if (pn == 0)
						sb_printf(&cl, " 0");
					else if (qd == 1)
						sb_printf(&cl, " %lld", (long long)pn);
					else
						sb_printf(&cl, " %lld/%lld", (long long)pn,
						    (long long)qd);
				} else
					sb_printf(&cl, " %.10g", cv[k]);
				if (cv[k] == 0 || (rat && pn == 0))
					continue;
				if (rat) {
					int64_t ap = pn < 0 ? -pn : pn;
					sb_printf(&sb, "%s", first ? (pn < 0 ? "-" : "") :
					    pn < 0 ? " - " : " + ");
					if (!T[0])
						sb_printf(&sb, "%lld", (long long)ap);
					else if (ap != 1)
						sb_printf(&sb, "%lld%s%s", (long long)ap,
						    T[0] == '(' ? "" : "", T);
					else if (k == 1 && pn > 0 && qd == 1)
						sb_printf(&sb, "%s", dx);
					else
						sb_printf(&sb, "%s", T);
					if (qd != 1)
						sb_printf(&sb, "/%lld", (long long)qd);
				} else {
					sb_printf(&sb, "%s%.10g%s%s", first ? (cv[k] < 0 ? "-" :
					    "") : cv[k] < 0 ? " - " : " + ", fabs(cv[k]),
					    T[0] ? "*" : "", T);
				}
			} else {
				sb_printf(&cl, " %s", node_str(c[k]));
				if (const_is(c[k], 0))
					continue;
				sb_printf(&sb, "%s(%s)%s%s", first ? "" : " + ",
				    node_str(c[k]), T[0] ? "*" : "", T);
			}
			first = 0;
			dxn = a == 0 ? xv : SUB(xv, KR(a));
			t = k == 0 ? c[0] : MUL(c[k], k == 1 ? dxn : POW(dxn, KI(k)));
			P = P ? ADD(P, t) : t;
		}
		if (!P)
			P = KR(0);
		printf("%s series of %s at %s = %s, through order %d:\n",
		    a == 0 ? "Maclaurin (Taylor at 0)" : "Taylor", fname2(f), x,
		    fnum(a), n);
		printf("  %s = %s + O(%s%s%s^%d)\n", fname2(f), sb.p ? sb.p : "0",
		    a == 0 ? "" : "(", dx, a == 0 ? "" : ")", n + 1);
		printf("  as %s -> %s; coefficients c_k = f^(k)(%s)/k!:%s\n", x,
		    fnum(a), fnum(a), cl.p);
		free(sb.p);
		free(cl.p);
	}
	if (f->np != 1 || !allnum) {
		printf("  (several variables: the coefficients are symbolic; "
		    "checked through the derivatives only)\n");
		return;
	}
	/* Taylor-jet arithmetic in long double as an independent check */
	{
		struct lref r;
		ld jet[AMAXJ];
		lr_open(&r, f->body, 0);
		if (n + 1 <= AMAXJ && lref_taylor(&r, a, n + 1, jet)) {
			double md = 0;
			for (k = 0; k <= n; k++)
				md = fmax(md, fabs((double)jet[k] - cv[k]) /
				    fmax(1, fabs(cv[k])));
			if (md <= 1e-9)
				printf("  check: the coefficients agree with Taylor-jet "
				    "arithmetic (long double) to %.1g\n", md);
			else
				printf("  check FAILED: Taylor-jet arithmetic gives "
				    "coefficients differing by %.3g\n", md);
		}
		/* the error shrinks like h^(n+1) */
		{
			double e[3] = {0, 0, 0}, h0 = 0.1 * fmax(1, fabs(a));
			int cnt[3] = {0, 0, 0};
			struct lref rp;
			lr_open(&rp, P, 0);
			for (i = 0; i < 3; i++) {
				ld h = h0 / (1 << i), s;
				for (s = -1; s <= 1; s += 2) {
					ld fv = lr(&r, a + s * h), pv = lr(&rp, a + s * h);
					if (isnan(fv) || isnan(pv))
						continue;
					e[i] = fmax(e[i], (double)fabsl(fv - pv));
					cnt[i]++;
				}
			}
			lref_free(&rp);
			if (cnt[0] && cnt[1] && cnt[2]) {
				printf("  check: |f - P| = %.3g at %s = %s \xc2\xb1 %s, %.3g "
				    "at \xc2\xb1 %s, %.3g at \xc2\xb1 %s", e[0], x, fnum(a),
				    fnum(h0), e[1], fnum(h0 / 2), e[2], fnum(h0 / 4));
				if (e[2] > 1e-15 && e[1] > 1e-15)
					printf(" (shrinks like h^%.1f; expected >= %d)\n",
					    log2(e[1] / e[2]), n + 1);
				else
					printf(" (at rounding level)\n");
			}
		}
		lref_free(&r);
	}
}

/* ---- implicit differentiation ---- */

/* y with F(x, y) = 0 near y0 by Newton, F_y given */
static int
newton_y(struct mev *mF, struct mev *mFy, double *pt, int xs, int ys,
    double y0, double *y)
{
	int i, o1, o2;
	double v, d;

	*y = y0;
	for (i = 0; i < 60; i++) {
		pt[ys] = *y;
		v = mev_at(mF, pt, MAXP, &o1);
		d = mev_at(mFy, pt, MAXP, &o2);
		if (!o1 || !o2 || d == 0)
			return 0;
		*y -= v / d;
		if (fabs(v / d) <= 1e-15 * fmax(1, fabs(*y)))
			break;
	}
	pt[ys] = *y;
	v = mev_at(mF, pt, MAXP, &o1);
	(void)xs;
	return o1 && fabs(v) <= 1e-10 && fabs(*y - y0) < 0.5;
}

void
c2_implicit(struct func *f, nid F, int xs, int ys, int order)
{
	nid Fx, Fy, D1, D2 = 0, Fxx, Fxy, Fyy;
	const char *x = pname(f, xs), *y = pname(f, ys);
	struct mev mF, mFy, m1, m2;
	int found = 0, bad = 0, i, j, o;
	double worst = 0;

	if (!TY_FP(f->p[xs].ty) || !TY_FP(f->p[ys].ty))
		fail("implicit differentiation needs real variables");
	F = mk_cast(TY_REAL, F);
	Fx = dif(F, xs);
	Fy = dif(F, ys);
	D1 = simp(mk_un(O_NEG, DIV(Fx, Fy)), 0);
	printf("on the curve %s = 0, %s as a function of %s:\n", node_str(F), y,
	    x);
	printf("  d%s/d%s = -F_%s/F_%s = %s\n", y, x, x, y, node_str(D1));
	if (order >= 2) {
		Fxx = dif(Fx, xs);
		Fxy = dif(Fx, ys);
		Fyy = dif(Fy, ys);
		D2 = simp(mk_un(O_NEG, DIV(ADD(SUB(MUL(Fxx, POW(Fy, KI(2))),
		    MUL(MUL(KR(2), Fxy), MUL(Fx, Fy))), MUL(Fyy, POW(Fx, KI(2)))),
		    POW(Fy, KI(3)))), 0);
		printf("  d\xc2\xb2%s/d%s\xc2\xb2 = %s\n", y, x, node_str(D2));
	}
	printf("  holds where F_%s = %s != 0 (the implicit function theorem: "
	    "there %s is locally a differentiable function of %s)\n", y,
	    node_str(Fy), y, x);
	if (f->np != 2) {
		printf("  (more than two variables: the others are held fixed; "
		    "the numeric check needs exactly two)\n");
		return;
	}
	mev_open(&mF, F);
	mev_open(&mFy, Fy);
	mev_open(&m1, D1);
	if (D2)
		mev_open(&m2, D2);
	for (i = 0; i < 40 && found < 12; i++) {
		double pt[MAXP] = {0}, x0 = -3.7 + i * 0.19, prev = NAN, py = 0;
		pt[xs] = x0;
		for (j = 0; j <= 400 && found < 12; j++) {
			double yy = -10 + j * 0.05, v;
			pt[ys] = yy;
			v = mev_at(&mF, pt, MAXP, &o);
			if (!o) {
				prev = NAN;
				continue;
			}
			if (!isnan(prev) && (v > 0) != (prev > 0)) {
				double y0, ya, yb, s1, s2, d1v, h = 1e-4, R, num2;
				int k;
				/* bisect, then the slope from nearby curve points */
				double a = py, b = yy, fa = prev;
				for (k = 0; k < 80; k++) {
					double m = 0.5 * (a + b), fm;
					pt[ys] = m;
					fm = mev_at(&mF, pt, MAXP, &o);
					if (!o)
						break;
					if ((fm > 0) == (fa > 0))
						a = m, fa = fm;
					else
						b = m;
				}
				y0 = 0.5 * (a + b);
				pt[xs] = x0;
				pt[ys] = y0;
				if (fabs(mev_at(&mF, pt, MAXP, &o)) > 1e-9 || !o)
					goto next;
				d1v = mev_at(&m1, pt, MAXP, &o);
				if (!o || fabs(d1v) > 1e3)
					goto next;
				pt[xs] = x0 + h;
				if (!newton_y(&mF, &mFy, pt, xs, ys, y0 + h * d1v, &yb))
					goto next;
				pt[xs] = x0 - h;
				if (!newton_y(&mF, &mFy, pt, xs, ys, y0 - h * d1v, &ya))
					goto next;
				s1 = (yb - ya) / (2 * h);
				pt[xs] = x0 + h / 2;
				if (!newton_y(&mF, &mFy, pt, xs, ys, y0 + h / 2 * d1v, &yb))
					goto next;
				pt[xs] = x0 - h / 2;
				if (!newton_y(&mF, &mFy, pt, xs, ys, y0 - h / 2 * d1v, &ya))
					goto next;
				s2 = (yb - ya) / h;
				R = (4 * s2 - s1) / 3;
				found++;
				if (fabs(R - d1v) > 1e-6 * fmax(1, fabs(d1v)))
					bad++;
				worst = fmax(worst, fabs(R - d1v) / fmax(1, fabs(d1v)));
				if (D2) {
					double d2v;
					pt[xs] = x0;
					pt[ys] = y0;
					d2v = mev_at(&m2, pt, MAXP, &o);
					num2 = (yb - 2 * y0 + ya) / (h * h / 4);
					if (o && fabs(num2 - d2v) > 1e-3 * fmax(1, fabs(d2v)))
						bad++;
				}
			next:
				pt[xs] = x0;
			}
			prev = v;
			py = yy;
		}
	}
	mev_close(&mF);
	mev_close(&mFy);
	mev_close(&m1);
	if (D2)
		mev_close(&m2);
	if (!found)
		printf("  check: found no point of the curve with x in [-3.7, 3.7] "
		    "and y in [-10, 10]\n");
	else if (!bad)
		printf("  check: matches the slope of the curve (points solved by "
		    "bisection and Newton) at %d points (max relative difference "
		    "%.1g)\n", found, worst);
	else
		printf("  check FAILED at %d of %d curve points (max relative "
		    "difference %.3g)\n", bad, found, worst);
}

/* ---- limits ---- */

static const char *c2_xname = "x";   /* the limit variable, for the steps */
#define pname2(s) c2_xname

enum { LV_NUM, LV_PINF, LV_NINF, LV_OSC, LV_DNE, LV_UNDEF, LV_UNKNOWN };
struct lv {
	int k;
	double v;          /* LV_NUM */
	double lo, hi;     /* LV_OSC: the values keep sweeping [lo, hi] */
};
struct lctx {
	int slot;
	double a;
	int side;          /* x approaches a from below (-1) or above (+1) */
	int lh, depth;     /* L'Hopital steps, recursion depth */
	struct sbuf *log;
};

static struct lv
LN(double v)
{
	struct lv r = {LV_NUM, v, 0, 0};
	if (isinf(v))
		r.k = v > 0 ? LV_PINF : LV_NINF;
	return r;
}

static struct lv
LK(int k)
{
	struct lv r = {k, 0, 0, 0};
	return r;
}

static int
isinfl_(struct lv l)
{
	return l.k == LV_PINF || l.k == LV_NINF;
}

static int
iszero(struct lv l)
{
	return l.k == LV_NUM && l.v == 0;
}

/* the k-th probe point (k = 1..8): closer to a, or further out */
static ld
probe_x(const struct lctx *c, int k)
{
	if (isinf(c->a))
		return (c->a > 0 ? 1 : -1) * powl(10, k + 1);
	return c->a + c->side * fmax(1, fabs(c->a)) * powl(10, -k - 1);
}

static ld
probe(nid e, const struct lctx *c, int k)
{
	struct lref r;
	ld v;

	lr_open(&r, e, c->slot);
	v = lref_eval(&r, probe_x(c, k));
	lref_free(&r);
	return v;
}

/* the side from which e approaches L: +1 above, -1 below, 0 neither */
static int
side_rel(nid e, const struct lctx *c, double L)
{
	struct lref r;
	int k, up = 0, dn = 0, n = 0;

	lr_open(&r, e, c->slot);
	for (k = 3; k <= 7; k++) {
		ld v = lref_eval(&r, probe_x(c, k));
		if (isnan(v))
			continue;
		n++;
		if (v > L)
			up++;
		else if (v < L)
			dn++;
	}
	lref_free(&r);
	if (n == 0)
		return 0;
	return up == n ? 1 : dn == n ? -1 : 0;
}

/* defined at every probe point */
static int
defined_near(nid e, const struct lctx *c)
{
	struct lref r;
	int k, ok = 1;

	lr_open(&r, e, c->slot);
	for (k = 2; k <= 7; k++)
		if (isnan(lref_eval(&r, probe_x(c, k))))
			ok = 0;
	lref_free(&r);
	return ok;
}

static void
lstep(struct lctx *c, const char *fmt, ...)
{
	va_list ap;
	char b[1024];
	int lines = 0;
	size_t k;

	if (!c->log)
		return;
	va_start(ap, fmt);
	vsnprintf(b, sizeof b, fmt, ap);
	va_end(ap);
	if (strlen(b) > 200)
		strcpy(b + 196, " ...");
	for (k = 0; k < c->log->n; k++)
		lines += c->log->p[k] == '\n';
	if (c->log->p && strstr(c->log->p, b))
		return;   /* the same step again (both sides, retries) */
	if (lines >= 14) {
		if (!strstr(c->log->p, "  (more steps)"))
			sb_puts(c->log, "  (more steps)\n");
		return;
	}
	sb_printf(c->log, "  %s\n", b);
}

static struct lv lim(nid e, struct lctx *c);

/* N/D with both -> 0 or both -> inf: L'Hopital, bounded */
static struct lv
lhopital(nid Nn, nid Dn, struct lctx *c, const char *form)
{
	nid dn, dd;

	if (c->lh >= 12)
		return LK(LV_UNKNOWN);
	dn = try_diff(Nn, c->slot, 1);
	dd = try_diff(Dn, c->slot, 1);
	if (!dn || !dd)
		return LK(LV_UNKNOWN);
	c->lh++;
	{
		nid q = simp(DIV(dn, dd), 1);
		lstep(c, "%s: L'H\xc3\xb4pital: lim (%s)/(%s) = lim %s", form,
		    node_str(dn), node_str(dd), node_str(q));
		return lim(q, c);
	}
}

/*
 * e ~ c |x|^p as x -> +-inf, for algebraic e (sums, products, quotients,
 * constant powers, sqrt); 0 when a sum's leading terms cancel or an
 * operation is not algebraic.
 */
static int
lead(nid e, const struct lctx *c, double *co, double *p)
{
	double ca, pa, cb, pb, k;

	if (num_of(e, &k)) {
		*co = k;
		*p = 0;
		return 1;
	}
	switch (N.op[e]) {
	case O_VAR:
		if (N.ki[e] != c->slot)
			return 0;
		*co = c->a > 0 ? 1 : -1;
		*p = 1;
		return 1;
	case O_CAST:
		return TY_FP(N.ty[e]) && lead(N.a[e], c, co, p);
	case O_NEG:
		if (!lead(N.a[e], c, co, p))
			return 0;
		*co = -*co;
		return 1;
	case O_ADD:
	case O_SUB:
		if (!lead(N.a[e], c, &ca, &pa) || !lead(N.b[e], c, &cb, &pb))
			return 0;
		if (N.op[e] == O_SUB)
			cb = -cb;
		if (ca == 0 || (cb != 0 && pb > pa + 1e-12)) {
			*co = cb;
			*p = pb;
		} else if (cb == 0 || pa > pb + 1e-12) {
			*co = ca;
			*p = pa;
		} else {
			*co = ca + cb;
			*p = pa;
			if (fabs(*co) <= 1e-12 * fmax(fabs(ca), fabs(cb)))
				return 0;
		}
		return 1;
	case O_MUL:
		if (!lead(N.a[e], c, &ca, &pa) || !lead(N.b[e], c, &cb, &pb))
			return 0;
		*co = ca * cb;
		*p = pa + pb;
		return *co != 0;
	case O_DIV:
		if (!lead(N.a[e], c, &ca, &pa) || !lead(N.b[e], c, &cb, &pb) ||
		    cb == 0)
			return 0;
		*co = ca / cb;
		*p = pa - pb;
		return 1;
	case O_POW:
		if (!num_of(N.b[e], &k) || !lead(N.a[e], c, &ca, &pa) || ca == 0)
			return 0;
		if (ca < 0 && k != floor(k))
			return 0;
		*co = pow(ca, k);
		*p = pa * k;
		return 1;
	case O_SQRT:
		if (!lead(N.a[e], c, &ca, &pa) || ca <= 0)
			return 0;
		*co = sqrt(ca);
		*p = pa / 2;
		return 1;
	}
	return 0;
}

/* leading terms of a ratio of polynomials at +-inf */
static int
poly_ratio_inf(nid Nn, nid Dn, struct lctx *c, struct lv *out)
{
	struct npoly p, q;
	double ln, ld_;
	int dp;

	if (!isinf(c->a))
		return 0;
	{
		double cn, pn, cd, pd;
		if ((!npoly_of(Nn, c->slot, &p, 12) || !npoly_of(Dn, c->slot, &q,
		    12)) && lead(Nn, c, &cn, &pn) && lead(Dn, c, &cd, &pd) &&
		    cd != 0 && cn != 0 && (pn != 0 || pd != 0)) {
			double r = cn / cd, pw = pn - pd;
			if (pn != 0 && pd != 0)
				lstep(c, "at %sinf: %s ~ %s|%s|^%s, %s ~ %s|%s|^%s",
			    c->a < 0 ? "-" : "", node_str(Nn), fnum(cn), pname2(0),
			    fnum(pn), node_str(Dn), fnum(cd), pname2(0), fnum(pd));
			if (pw < -1e-12)
				*out = LN(0);
			else if (pw < 1e-12)
				*out = LN(r);
			else
				*out = LK(r > 0 ? LV_PINF : LV_NINF);
			return 1;
		}
	}
	if (!npoly_of(Nn, c->slot, &p, 12) ||
	    !npoly_of(Dn, c->slot, &q, 12) || p.deg + q.deg == 0)
		return 0;
	if (!num_of(p.c[p.deg], &ln) || !num_of(q.c[q.deg], &ld_) || ld_ == 0)
		return 0;
	dp = p.deg - q.deg;
	if (p.deg > 0 && q.deg > 0)
		lstep(c, "leading terms: %s*%s^%d / (%s*%s^%d)", fnum(ln),
	    pname2(c->slot), p.deg, fnum(ld_), pname2(c->slot), q.deg);
	if (dp < 0)
		*out = LN(0);
	else if (dp == 0)
		*out = LN(ln / ld_);
	else {
		double s = ln / ld_ * ((c->a < 0 && (dp & 1)) ? -1 : 1);
		*out = LK(s > 0 ? LV_PINF : LV_NINF);
	}
	return 1;
}

static struct lv
lim_body(nid e, struct lctx *c)
{
	nid A = N.a[e], B = N.b[e];
	struct lv la, lb, r;
	double v;
	int op = N.op[e], s;

	if (!(N.fl[e] & NF_VAR)) {
		int ok;
		v = eval_const_at(e, c->slot, 0, &ok);
		return ok ? LN(v) : LK(LV_UNDEF);
	}
	/* continuous at a finite a: substitute */
	if (isfinite(c->a)) {
		int ok, k;
		double fa = eval_const_at(e, c->slot, c->a, &ok);
		if (ok && op != O_VAR) {
			ld prev = INFINITY;
			int good = 1;
			for (k = 2; k <= 7; k++) {
				ld d = fabsl(probe(e, c, k) - fa);
				if (isnan(d) || d > prev * 0.9 + 1e-15 * (1 + fabs(fa)))
					good = 0;
				prev = d;
			}
			if (good && prev <= 1e-5 * (1 + fabs(fa))) {
				if (c->depth == 1)
					lstep(c, "continuous at %s = %s: substitute", pname2(0),
					    fnum(c->a));
				return LN(fa);
			}
		}
	}
	switch (op) {
	case O_VAR:
		return LN(c->a);
	case O_CAST:
	case O_BARRIER:
		return lim(A, c);
	case O_NEG:
		la = lim(A, c);
		if (la.k == LV_NUM)
			return LN(-la.v);
		if (la.k == LV_PINF)
			return LK(LV_NINF);
		if (la.k == LV_NINF)
			return LK(LV_PINF);
		if (la.k == LV_OSC) {
			r = la;
			r.lo = -la.hi;
			r.hi = -la.lo;
			return r;
		}
		return la;
	case O_ADD:
	case O_SUB:
		la = lim(A, c);
		lb = lim(B, c);
		if (op == O_SUB) {
			struct lv t = lb;
			if (lb.k == LV_NUM)
				t.v = -lb.v;
			else if (lb.k == LV_PINF)
				t.k = LV_NINF;
			else if (lb.k == LV_NINF)
				t.k = LV_PINF;
			else if (lb.k == LV_OSC)
				t.lo = -lb.hi, t.hi = -lb.lo;
			lb = t;
		}
		if (la.k == LV_UNDEF || lb.k == LV_UNDEF)
			return LK(LV_UNDEF);
		if (la.k == LV_NUM && lb.k == LV_NUM)
			return LN(la.v + lb.v);
		if (isinfl_(la) && isinfl_(lb) && la.k != lb.k) {
			/* inf - inf: A - B = (1/B - 1/A) / (1/(A B)) */
			nid Bn = op == O_SUB ? B : mk_un(O_NEG, B);
			if (N.op[A] == O_SQRT || N.op[Bn] == O_SQRT) {
				nid q = simp(DIV(SUB(POW(A, KI(2)), POW(Bn, KI(2))),
				    ADD(A, Bn)), 1);
				lstep(c, "inf - inf: times the conjugate: (%s)^2 - (%s)^2 "
				    "over their sum = %s", node_str(A), node_str(Bn),
				    node_str(q));
				return lim(q, c);
			}
			lstep(c, "inf - inf: %s - %s = (1/(%s) - 1/(%s))/(1/((%s)*(%s)))",
			    node_str(A), node_str(Bn), node_str(Bn), node_str(A),
			    node_str(A), node_str(Bn));
			return lhopital(tidy(SUB(DIV(KR(1), Bn), DIV(KR(1), A))),
			    tidy(DIV(KR(1), MUL(A, Bn))), c, "0/0");
		}
		if (isinfl_(la) && (lb.k == LV_NUM || lb.k == LV_OSC || la.k ==
		    lb.k))
			return la;
		if (isinfl_(lb) && (la.k == LV_NUM || la.k == LV_OSC))
			return lb;
		if (la.k == LV_OSC && lb.k == LV_NUM) {
			la.lo += lb.v;
			la.hi += lb.v;
			return la;
		}
		if (lb.k == LV_OSC && la.k == LV_NUM) {
			lb.lo += la.v;
			lb.hi += la.v;
			return lb;
		}
		return LK(LV_UNKNOWN);
	case O_MUL:
		la = lim(A, c);
		lb = lim(B, c);
		if (la.k == LV_UNDEF || lb.k == LV_UNDEF)
			return LK(LV_UNDEF);
		if (la.k == LV_NUM && lb.k == LV_NUM)
			return LN(la.v * lb.v);
		if (isinf(c->a) && ((iszero(la) && isinfl_(lb)) || (iszero(lb) &&
		    isinfl_(la)))) {
			/* exp(u) with u ~ c|x|^p, p > 0, beats any power of x */
			int w;
			for (w = 0; w < 2; w++) {
				nid E = w ? B : A, P = w ? A : B;
				double cu, pu, cp, pp;
				if (N.op[E] == O_EXP && lead(N.a[E], c, &cu, &pu) && pu > 0 &&
				    lead(P, c, &cp, &pp)) {
					lstep(c, "%s grows or decays faster than any power of "
					    "%s", node_str(E), pname2(0));
					if (cu < 0)
						return LN(0);
					return LK(cp > 0 ? LV_PINF : LV_NINF);
				}
			}
		}
		if ((iszero(la) && isinfl_(lb)) || (iszero(lb) && isinfl_(la))) {
			nid z = iszero(la) ? A : B, w = iszero(la) ? B : A, iw, iz;
			size_t mark = c->log ? c->log->n : 0;
			int lh = c->lh;
			/* both quotients; the logarithm-like factor on top first */
			iz = simp(DIV(KR(1), z), 1);
			iw = simp(DIV(KR(1), w), 1);
			/* exp(-x) x^5 = x^5/exp(x); x log x = log(x)/(1/x) */
			if (N.op[w] == O_LOG || N.op[w] == O_ATAN || N.op[z] == O_EXP ||
			    (N.op[z] == O_MUL && (N.op[N.a[z]] == O_EXP ||
			    N.op[N.b[z]] == O_EXP))) {
				lstep(c, "0*inf: %s*%s = (%s)/(%s)", node_str(A),
				    node_str(B), node_str(w), node_str(iz));
				r = lhopital(w, iz, c, "inf/inf");
				if (r.k != LV_UNKNOWN)
					return r;
				if (c->log) {
					c->log->n = mark;
					if (c->log->p)
						c->log->p[mark] = 0;
				}
				c->lh = lh;
			}
			lstep(c, "0*inf: %s*%s = (%s)/(%s)", node_str(A),
			    node_str(B), node_str(z), node_str(iw));
			r = lhopital(z, iw, c, "0/0");
			if (r.k != LV_UNKNOWN)
				return r;
			if (c->log) {
				c->log->n = mark;
				if (c->log->p)
					c->log->p[mark] = 0;
			}
			c->lh = lh;
			lstep(c, "0*inf: %s*%s = (%s)/(%s)", node_str(A),
			    node_str(B), node_str(w), node_str(iz));
			return lhopital(w, iz, c, "inf/inf");
		}
		if ((iszero(la) && lb.k == LV_OSC) || (iszero(lb) &&
		    la.k == LV_OSC)) {
			lstep(c, "squeeze: %s -> 0 times a bounded factor", node_str(
			    iszero(la) ? A : B));
			return LN(0);
		}
		if (isinfl_(la) || isinfl_(lb)) {
			struct lv o = isinfl_(la) ? lb : la, i = isinfl_(la) ? la : lb;
			int sg = i.k == LV_PINF ? 1 : -1;
			if (o.k == LV_NUM && o.v != 0)
				return LK(sg * (o.v > 0 ? 1 : -1) > 0 ? LV_PINF : LV_NINF);
			if (isinfl_(o))
				return LK(sg * (o.k == LV_PINF ? 1 : -1) > 0 ? LV_PINF :
				    LV_NINF);
			if (o.k == LV_OSC)
				return LK(o.lo > 0 ? (sg > 0 ? LV_PINF : LV_NINF) :
				    o.hi < 0 ? (sg > 0 ? LV_NINF : LV_PINF) : LV_DNE);
			return LK(LV_UNKNOWN);
		}
		if (la.k == LV_OSC && lb.k == LV_NUM) {
			double p = la.lo * lb.v, q = la.hi * lb.v;
			la.lo = fmin(p, q);
			la.hi = fmax(p, q);
			return la;
		}
		if (lb.k == LV_OSC && la.k == LV_NUM) {
			double p = lb.lo * la.v, q = lb.hi * la.v;
			lb.lo = fmin(p, q);
			lb.hi = fmax(p, q);
			return lb;
		}
		return LK(la.k == LV_OSC || lb.k == LV_OSC ? LV_DNE : LV_UNKNOWN);
	case O_DIV:
		if (poly_ratio_inf(A, B, c, &r))
			return r;
		la = lim(A, c);
		lb = lim(B, c);
		if (la.k == LV_UNDEF || lb.k == LV_UNDEF)
			return LK(LV_UNDEF);
		if (la.k == LV_NUM && lb.k == LV_NUM && lb.v != 0)
			return LN(la.v / lb.v);
		if (iszero(la) && iszero(lb))
			return lhopital(A, B, c, "0/0");
		if (isinfl_(la) && isinfl_(lb))
			return lhopital(A, B, c, "inf/inf");
		if (iszero(lb)) {
			s = side_rel(B, c, 0);
			if (la.k == LV_NUM || isinfl_(la)) {
				int sa = la.k == LV_NUM ? (la.v > 0 ? 1 : -1) :
				    la.k == LV_PINF ? 1 : -1;
				if (s == 0) {
					lstep(c, "%s -> 0 changing sign: %s/%s takes both "
					    "signs", node_str(B), node_str(A), node_str(B));
					return LK(LV_DNE);
				}
				lstep(c, "%s -> %s (the numerator -> %s, %s -> 0%s)",
				    node_str(e), sa * s > 0 ? "inf" : "-inf", la.k == LV_NUM ?
				    fnum(la.v) : la.k == LV_PINF ? "inf" : "-inf",
				    node_str(B), s > 0 ? "+" : "-");
				return LK(sa * s > 0 ? LV_PINF : LV_NINF);
			}
			if (la.k == LV_OSC)
				return LK(LV_DNE);
			return LK(LV_UNKNOWN);
		}
		if (isinfl_(lb) && (la.k == LV_NUM || la.k == LV_OSC))
			return LN(0);
		if (isinfl_(la) && lb.k == LV_NUM)
			return LK((la.k == LV_PINF) == (lb.v > 0) ? LV_PINF : LV_NINF);
		if (la.k == LV_OSC && lb.k == LV_NUM) {
			double p = la.lo / lb.v, q = la.hi / lb.v;
			la.lo = fmin(p, q);
			la.hi = fmax(p, q);
			return la;
		}
		return LK(LV_UNKNOWN);
	case O_POW: {
		double k;
		if (num_of(B, &k)) {
			la = lim(A, c);
			if (la.k == LV_NUM) {
				if (la.v == 0 && k < 0) {
					s = side_rel(A, c, 0);
					if (k == floor(k) && fmod(fabs(k), 2) == 1) {
						if (s == 0)
							return LK(LV_DNE);
						return LK(s > 0 ? LV_PINF : LV_NINF);
					}
					return LK(LV_PINF);
				}
				v = pow(la.v, k);
				return isnan(v) ? LK(LV_UNDEF) : LN(v);
			}
			if (la.k == LV_PINF)
				return k > 0 ? LK(LV_PINF) : k < 0 ? LN(0) : LN(1);
			if (la.k == LV_NINF && k == floor(k))
				return k < 0 ? LN(0) : k == 0 ? LN(1) :
				    fmod(k, 2) == 1 ? LK(LV_NINF) : LK(LV_PINF);
			if (la.k == LV_OSC && k > 0 && k == floor(k)) {
				double p = pow(la.lo, k), q = pow(la.hi, k);
				la.lo = fmin(p, q);
				la.hi = fmax(p, q);
				if (la.lo < 0 && la.hi > 0 && fmod(k, 2) == 0)
					la.lo = 0;
				return la;
			}
			return LK(LV_UNKNOWN);
		}
		/* A^B = exp(B log A) */
		if (side_rel(A, c, 0) <= 0 && !(!(N.fl[A] & NF_VAR) &&
		    num_of(A, &k) && k > 0))
			return LK(LV_UNKNOWN);
		lstep(c, "%s = exp((%s)*log(%s))", node_str(e), node_str(B),
		    node_str(A));
		la = lim(simp(MUL(B, mk_un(O_LOG, A)), 1), c);
		if (la.k == LV_NUM)
			return LN(exp(la.v));
		if (la.k == LV_PINF)
			return LK(LV_PINF);
		if (la.k == LV_NINF)
			return LN(0);
		return la.k == LV_OSC ? LK(LV_DNE) : la;
	}
	case O_EXP:
		la = lim(A, c);
		if (la.k == LV_NUM)
			return LN(exp(la.v));
		if (la.k == LV_PINF)
			return LK(LV_PINF);
		if (la.k == LV_NINF)
			return LN(0);
		if (la.k == LV_OSC) {
			la.lo = exp(la.lo);
			la.hi = exp(la.hi);
		}
		return la;
	case O_LOG:
		la = lim(A, c);
		if (la.k == LV_NUM && la.v > 0)
			return LN(log(la.v));
		if (la.k == LV_NUM && la.v == 0)
			return side_rel(A, c, 0) > 0 ? LK(LV_NINF) : LK(LV_UNDEF);
		if (la.k == LV_PINF)
			return LK(LV_PINF);
		return la.k == LV_UNKNOWN ? la : LK(LV_UNDEF);
	case O_SQRT:
		la = lim(A, c);
		if (la.k == LV_NUM && la.v >= 0)
			return (la.v > 0 || side_rel(A, c, 0) >= 0) ? LN(sqrt(la.v)) :
			    LK(LV_UNDEF);
		if (la.k == LV_PINF)
			return LK(LV_PINF);
		return la.k == LV_UNKNOWN ? la : LK(LV_UNDEF);
	case O_SIN:
	case O_COS:
		la = lim(A, c);
		if (la.k == LV_NUM)
			return LN(op == O_SIN ? sin(la.v) : cos(la.v));
		if (isinfl_(la)) {
			lstep(c, "%s -> %sinf: %s(%s) keeps oscillating between -1 and 1",
			    node_str(A), la.k == LV_NINF ? "-" : "", op == O_SIN ?
			    "sin" : "cos", node_str(A));
			r = LK(LV_OSC);
			r.lo = -1;
			r.hi = 1;
			return r;
		}
		if (la.k == LV_OSC) {
			r = LK(LV_OSC);
			r.lo = -1;
			r.hi = 1;
			return r;
		}
		return la;
	case O_TAN:
		la = lim(A, c);
		if (la.k == LV_NUM) {
			if (fabs(cos(la.v)) > 1e-12)
				return LN(tan(la.v));
			s = side_rel(A, c, la.v);
			if (s == 0)
				return LK(LV_DNE);
			/* tan -> +inf just below pi/2 */
			return LK(s < 0 ? LV_PINF : LV_NINF);
		}
		return isinfl_(la) ? LK(LV_DNE) : la;
	case O_ATAN:
		la = lim(A, c);
		if (la.k == LV_NUM)
			return LN(atan(la.v));
		if (la.k == LV_PINF)
			return LN(M_PI / 2);
		if (la.k == LV_NINF)
			return LN(-M_PI / 2);
		if (la.k == LV_OSC) {
			la.lo = atan(la.lo);
			la.hi = atan(la.hi);
		}
		return la;
	case O_ABS:
		la = lim(A, c);
		if (la.k == LV_NUM)
			return LN(fabs(la.v));
		if (isinfl_(la))
			return LK(LV_PINF);
		if (la.k == LV_OSC) {
			double lo = la.lo, hi = la.hi;
			la.lo = (lo <= 0 && hi >= 0) ? 0 : fmin(fabs(lo), fabs(hi));
			la.hi = fmax(fabs(lo), fabs(hi));
		}
		return la;
	case O_SIGN:
	case O_FLOOR:
	case O_CEIL:
		la = lim(A, c);
		if (la.k == LV_NUM) {
			double L = la.v;
			s = side_rel(A, c, L);
			if (op == O_SIGN)
				return L != 0 ? LN(L > 0 ? 1 : -1) : s ? LN(s) :
				    LK(LV_UNKNOWN);
			if (L != floor(L))
				return LN(op == O_FLOOR ? floor(L) : ceil(L));
			if (s == 0)
				return side_rel(A, c, L - 0.5) == 1 && side_rel(A, c,
				    L + 0.5) == -1 ? LK(LV_UNKNOWN) : LK(LV_UNKNOWN);
			lstep(c, "%s -> %s from %s", node_str(A), fnum(L), s > 0 ?
			    "above" : "below");
			return LN(op == O_FLOOR ? (s > 0 ? L : L - 1) :
			    (s > 0 ? L + 1 : L));
		}
		if (isinfl_(la))
			return op == O_SIGN ? LN(la.k == LV_PINF ? 1 : -1) : la;
		return LK(LV_UNKNOWN);
	case O_MIN:
	case O_MAX:
		la = lim(A, c);
		lb = lim(B, c);
		if (la.k == LV_NUM && lb.k == LV_NUM)
			return LN(op == O_MIN ? fmin(la.v, lb.v) : fmax(la.v, lb.v));
		if (la.k == LV_PINF)
			return op == O_MIN ? lb : la;
		if (lb.k == LV_PINF)
			return op == O_MIN ? la : lb;
		if (la.k == LV_NINF)
			return op == O_MIN ? la : lb;
		if (lb.k == LV_NINF)
			return op == O_MIN ? lb : la;
		return LK(LV_UNKNOWN);
	case O_SEL:
		/* the branch taken near a */
		s = side_rel(A, c, 0.5);
		if (s == 0)
			return LK(LV_UNKNOWN);
		lstep(c, "near the limit point %s holds: the %s branch",
		    s > 0 ? node_str(A) : node_str(mk_un(O_LNOT, A)), s > 0 ?
		    "first" : "second");
		return lim(s > 0 ? B : N.c[e], c);
	case O_LAMBERTW:
		la = lim(A, c);
		if (la.k == LV_NUM && la.v >= -exp(-1.0))
			return LN(lambert_w(la.v, 0));
		if (la.k == LV_PINF)
			return LK(LV_PINF);
		return LK(LV_UNKNOWN);
	case O_SEC:
	case O_CSC:
	case O_COT:
		/* through cos, sin */
		return lim(tidy(op == O_SEC ? DIV(KR(1), mk_un(O_COS, A)) :
		    op == O_CSC ? DIV(KR(1), mk_un(O_SIN, A)) :
		    DIV(mk_un(O_COS, A), mk_un(O_SIN, A))), c);
	}
	return LK(LV_UNKNOWN);
}

static struct lv
lim(nid e, struct lctx *c)
{
	struct lv r;

	if (c->depth > 60)
		return LK(LV_UNKNOWN);
	c->depth++;
	r = lim_body(e, c);
	c->depth--;
	if (r.k == LV_NUM && isnan(r.v))
		r = LK(LV_UNDEF);
	return r;
}

static const char *
lv_str(struct lv l)
{
	static char buf[4][96];
	static int k;
	char *b = buf[k++ & 3];

	switch (l.k) {
	case LV_NUM:
		snprintf(b, 96, "%s", fnum(snap(l.v)));
		if (strcmp(b, fnum(l.v)) && fabs(snap(l.v) - l.v) > 0)
			snprintf(b, 96, "%s", fnum(l.v));
		return b;
	case LV_PINF:
		return "inf";
	case LV_NINF:
		return "-inf";
	case LV_OSC:
		snprintf(b, 96, "none (it oscillates between %s and %s)",
		    fnum(l.lo), fnum(l.hi));
		return b;
	case LV_DNE:
		return "none (it oscillates without bound)";
	case LV_UNDEF:
		return "undefined (the function is not defined near the point)";
	}
	return "not found symbolically";
}

static int
lv_eq(struct lv a, struct lv b)
{
	if (a.k != b.k)
		return 0;
	if (a.k == LV_NUM)
		return fabs(a.v - b.v) <= 1e-12 * fmax(1, fabs(a.v));
	return a.k == LV_PINF || a.k == LV_NINF;
}

/*
 * The numeric approach: values at a + s h, h = h0 2^-k (or x = x0 2^k),
 * and Richardson extrapolation in h (or 1/x).  Returns the estimate's
 * kind: LV_NUM with *est and *err, LV_PINF/LV_NINF when the values grow
 * without bound, LV_OSC when they do not settle, LV_UNDEF if undefined.
 */
static int
numeric_limit(nid e, int slot, double a, int side, double *est, double *err,
    struct sbuf *show, double *last, double *earlier)
{
	struct lref r;
	ld v[32], R[32][8], best = NAN, besterr = INFINITY;
	int n = 0, k, j, m = 22;

	lr_open(&r, e, slot);
	for (k = 0; k < m; k++) {
		ld x = isinf(a) ? (a > 0 ? 1 : -1) * 10.0L * powl(2, k) :
		    a + side * 0.1L * fmax(1, fabs(a)) * powl(2, -k);
		v[k] = lr(&r, x);
		if (isnan(v[k]))
			break;
		n++;
	}
	lref_free(&r);
	if (show) {
		static const int pick[] = {0, 4, 8, 12, 16, 21};
		for (j = 0; j < 6; j++)
			if (pick[j] < n)
				sb_printf(show, "%s%.10Lg", j ? ", " : "", v[pick[j]]);
	}
	if (n < 8)
		return LV_UNDEF;
	*last = (double)v[n - 1];
	*earlier = (double)v[n - 6];
	/* growing without bound: |v| increases steadily and is large */
	{
		int grow = 1;
		for (k = n - 6; k < n; k++)
			if (!(fabsl(v[k]) > fabsl(v[k - 1]) * 1.05) ||
			    (v[k] > 0) != (v[n - 1] > 0))
				grow = 0;
		if (grow && fabsl(v[n - 1]) > 1e4) {
			*est = v[n - 1] > 0 ? INFINITY : -INFINITY;
			*err = 0;
			return v[n - 1] > 0 ? LV_PINF : LV_NINF;
		}
	}
	/* Richardson on the tail: errors in powers of h */
	for (k = 0; k < n; k++) {
		R[k][0] = v[k];
		for (j = 1; j < 8 && j <= k; j++) {
			ld p = powl(2, j);
			R[k][j] = (p * R[k][j - 1] - R[k - 1][j - 1]) / (p - 1);
		}
	}
	for (k = 6; k < n; k++)
		for (j = 1; j < 8 && j <= k - 1; j++) {
			ld d = fabsl(R[k][j] - R[k - 1][j]);
			if (d < besterr) {
				besterr = d;
				best = R[k][j];
			}
		}
	/* the raw values also settle (sqrt-like behaviour defeats h-powers) */
	{
		ld d1 = fabsl(v[n - 1] - v[n - 2]), d0 = fabsl(v[n - 5] - v[n - 6]);
		ld sc = fmaxl(1, fabsl(v[n - 1]));
		if (!(besterr <= 1e-6 * sc)) {
			if (d1 < d0 * 0.9 + 1e-15 * sc && d1 < 1e-3 * sc) {
				best = v[n - 1];
				besterr = d1 * 4;
			} else {
				ld lo = v[n - 8], hi = v[n - 8];
				for (k = n - 8; k < n; k++) {
					lo = fminl(lo, v[k]);
					hi = fmaxl(hi, v[k]);
				}
				*est = (double)lo;
				*err = (double)hi;
				return LV_OSC;
			}
		}
	}
	*est = (double)best;
	*err = (double)besterr;
	return LV_NUM;
}

struct side_res {
	struct lv l;
	struct sbuf log, show;
	int nk;
	double est, err, last, earlier;
};

static void
limit_side(nid e, int slot, double a, int side, struct side_res *s)
{
	struct lctx c;

	memset(s, 0, sizeof *s);
	memset(&c, 0, sizeof c);
	c.slot = slot;
	c.a = a;
	c.side = side;
	c.log = &s->log;
	s->l = lim(e, &c);
	s->nk = numeric_limit(e, slot, a, side, &s->est, &s->err, &s->show,
	    &s->last, &s->earlier);
	/* undefined symbolically yet defined numerically: not found */
	if (s->l.k == LV_UNDEF && s->nk != LV_UNDEF)
		s->l = LK(LV_UNKNOWN);
}

static void
limit_numeric_line(const char *x, double a, int side, struct side_res *s)
{
	const char *pts = isinf(a) ? (a > 0 ? "x = 10*2^k" : "x = -10*2^k") :
	    side > 0 ? "a + h" : "a - h";

	printf("  numeric%s (%s, %s): %s\n", side == 0 || isinf(a) ? "" :
	    side > 0 ? " from the right" : " from the left", pts, isinf(a) ?
	    "k = 0..21" : "h = 0.1*2^-k, k = 0..21", s->show.p ? s->show.p :
	    "undefined");
	switch (s->nk) {
	case LV_NUM:
		printf("    extrapolated (Richardson): %.12g (estimated error %.1g)",
		    s->est, s->err);
		break;
	case LV_PINF:
	case LV_NINF:
		printf("    the values grow without bound (%s)", s->nk == LV_PINF ?
		    "+" : "-");
		break;
	case LV_OSC:
		printf("    the values do not settle (the last ones lie in [%.6g, "
		    "%.6g])", s->est, s->err);
		break;
	default:
		printf("    undefined at the sample points");
	}
	switch (s->l.k) {
	case LV_NUM:
		if (s->nk == LV_NUM && fabs(s->est - s->l.v) <= fmax(10 * s->err,
		    1e-7 * (1 + fabs(s->l.v))))
			printf(": agrees\n");
		else if ((s->nk == LV_NUM || s->nk == LV_OSC) &&
		    fabs(s->last - s->l.v) <= 1e-5 * (1 + fabs(s->l.v)) &&
		    fabs(s->last - s->l.v) <= fabs(s->earlier - s->l.v))
			printf("\n    the values approach %s (|f - L| = %.1g at the "
			    "last point; not smooth enough to extrapolate): agrees\n",
			    lv_str(s->l), fabs(s->last - s->l.v));
		else
			printf(": DISAGREES with %s\n", lv_str(s->l));
		break;
	case LV_PINF:
	case LV_NINF:
		printf(s->nk == s->l.k ? ": agrees\n" : ": DISAGREES\n");
		break;
	case LV_OSC:
	case LV_DNE:
		printf(s->nk == LV_OSC || s->nk == LV_PINF || s->nk == LV_NINF ?
		    ": consistent\n" : ": the values seem to settle, which "
		    "contradicts the symbolic answer\n");
		break;
	case LV_UNKNOWN:
		printf("\n    (no symbolic answer: this numeric estimate is not a "
		    "proof)\n");
		break;
	default:
		putchar('\n');
	}
	(void)x;
}

void
c2_limit(struct func *f, int slot, double a, int side)
{
	nid e;
	const char *x;
	struct side_res L, R;
	int two;

	if (f->np == 0) {
		printf("%s is a constant: the limit is itself\n", fname2(f));
		return;
	}
	if (f->np > 1)
		fail("the limit of a function of several variables needs the "
		    "others fixed (give them values)");
	if (!TY_FP(f->p[slot].ty))
		fail("%s is an integer parameter: a limit needs a real one",
		    pname(f, slot));
	x = pname(f, slot);
	c2_xname = x;
	e = mk_cast(TY_REAL, f->body);
	if (isinf(a))
		side = a > 0 ? -1 : 1;
	two = side == 0;
	printf("limit of %s as %s -> %s%s:\n", fname2(f), x, fnum(a),
	    isinf(a) ? "" : side > 0 ? "+ (from the right)" : side < 0 ?
	    "- (from the left)" : "");
	if (!two) {
		limit_side(e, slot, a, side, &R);
		if (R.log.p)
			fputs(R.log.p, stdout);
		printf("  = %s\n", lv_str(R.l));
		if (R.l.k == LV_OSC || R.l.k == LV_DNE)
			printf("  the limit does not exist\n");
		limit_numeric_line(x, a, side, &R);
		free(R.log.p);
		free(R.show.p);
		return;
	}
	limit_side(e, slot, a, -1, &L);
	limit_side(e, slot, a, 1, &R);
	if (L.l.k == LV_UNDEF && R.l.k != LV_UNDEF) {
		printf("  %s is undefined just left of %s = %s: the limit is the "
		    "one from the right\n", fname2(f), x, fnum(a));
		if (R.log.p)
			fputs(R.log.p, stdout);
		printf("  = %s\n", lv_str(R.l));
		limit_numeric_line(x, a, 1, &R);
	} else if (R.l.k == LV_UNDEF && L.l.k != LV_UNDEF) {
		printf("  %s is undefined just right of %s = %s: the limit is the "
		    "one from the left\n", fname2(f), x, fnum(a));
		if (L.log.p)
			fputs(L.log.p, stdout);
		printf("  = %s\n", lv_str(L.l));
		limit_numeric_line(x, a, -1, &L);
	} else if (lv_eq(L.l, R.l) || (L.l.k == R.l.k && (L.l.k == LV_UNKNOWN ||
	    L.l.k == LV_UNDEF || L.l.k == LV_DNE))) {
		if (R.log.p)
			fputs(R.log.p, stdout);
		else if (L.log.p)
			fputs(L.log.p, stdout);
		printf("  = %s%s\n", lv_str(R.l), R.l.k == LV_NUM ||
		    isinfl_(R.l) ? " (from both sides)" : "");
		if (R.l.k == LV_DNE)
			printf("  the limit does not exist\n");
		limit_numeric_line(x, a, -1, &L);
		limit_numeric_line(x, a, 1, &R);
	} else if (L.l.k == LV_OSC && R.l.k == LV_OSC) {
		if (R.log.p)
			fputs(R.log.p, stdout);
		printf("  = %s\n", lv_str(R.l));
		printf("  the limit does not exist\n");
		limit_numeric_line(x, a, -1, &L);
		limit_numeric_line(x, a, 1, &R);
	} else {
		printf("  from the left:\n");
		if (L.log.p)
			fputs(L.log.p, stdout);
		printf("  = %s\n", lv_str(L.l));
		printf("  from the right:\n");
		if (R.log.p)
			fputs(R.log.p, stdout);
		printf("  = %s\n", lv_str(R.l));
		if (L.l.k == LV_UNKNOWN || R.l.k == LV_UNKNOWN)
			printf("  (one side was not found symbolically)\n");
		else
			printf("  the one-sided limits differ: the (two-sided) limit "
			    "does not exist\n");
		limit_numeric_line(x, a, -1, &L);
		limit_numeric_line(x, a, 1, &R);
	}
	free(L.log.p);
	free(L.show.p);
	free(R.log.p);
	free(R.show.p);
}

/* ---- the n-th derivative for a symbolic n ---- */

int
c2_nth(struct func *f, int slot, uint32_t nname)
{
	nid e = mk_cast(TY_REAL, f->body), u, nv, P = 0;
	double a, b, p, cdiv = 1;
	const char *x = pname(f, slot), *n = sym_str(nname);
	int k, op;

	nv = mk_var(MAXP - 1, nname, TY_REAL);
	while (N.op[e] == O_CAST)
		e = N.a[e];
	op = N.op[e];
	u = N.a[e];
	/* exp(u), sin(u), cos(u), log(u) for a linear u; u^p */
	if ((op == O_EXP || op == O_SIN || op == O_COS || op == O_LOG) &&
	    lin_of(u, slot, &a, &b)) {
		nid an = POW(KR(a), nv);
		if (op == O_EXP)
			P = MUL(an, e);
		else if (op == O_SIN)
			P = MUL(an, mk_un(O_SIN, ADD(u, MUL(nv, KR(M_PI / 2)))));
		else if (op == O_COS)
			P = MUL(an, mk_un(O_COS, ADD(u, MUL(nv, KR(M_PI / 2)))));
		else
			P = 0;
		if (op == O_LOG) {
			printf("d^%s/d%s^%s %s = (-1)^(%s - 1) (%s - 1)! %s^%s / (%s)^%s"
			    "  for %s >= 1\n", n, x, n, fname2(f), n, n, fnum(a), n,
			    node_str(u), n, n);
			/* check n = 1..6 */
			{
				nid g = e;
				double fact = 1;
				int ok = 1, cnt = 0, j;
				for (k = 1; k <= 6; k++) {
					struct lref r;
					g = dif(g, slot);
					if (k > 1)
						fact *= k - 1;
					lr_open(&r, g, slot);
					for (j = 0; j < 20; j++) {
						double xx = -b / a + (j + 1) * 0.37 / fabs(a), want;
						ld got = lr(&r, xx);
						want = ((k - 1) & 1 ? -1 : 1) * fact * pow(a, k) /
						    pow(a * xx + b, k);
						if (isnan(got))
							continue;
						cnt++;
						if (fabsl(got - want) > 1e-8 * fmax(1, fabs(want)))
							ok = 0;
					}
					lref_free(&r);
				}
				printf("  holds for %s %s %s; check: %s the derivatives of "
				    "orders 1..6 at %d points\n", x, a > 0 ? ">" : "<",
				    fnum(-b / a), ok ? "equals" : "DIFFERS FROM", cnt);
			}
			return 1;
		}
	} else if (((op == O_POW && num_of(N.b[e], &p)) || (op == O_SQRT &&
	    (p = 0.5, 1)) || (op == O_DIV && num_of(N.a[e], &cdiv) &&
	    (u = N.b[e], p = -1, 1))) && lin_of(u, slot, &a, &b)) {
		if (cdiv != 1)
			printf("  (%s = %s*(%s)^-1)\n", fname2(f), fnum(cdiv),
			    node_str(u));
		/* p (p-1) ... (p-n+1) a^n u^(p-n) */
		if (p == floor(p) && p >= 0)
			printf("  note: %s is a polynomial of degree %s: its "
			    "derivatives of order > %s are 0\n", fname2(f), fnum(p),
			    fnum(p));
		{
			struct sbuf t = {0};
			if (a != 1)
				sb_printf(&t, "%s%s%s^%s ", a < 0 ? "(" : "", fnum(a),
				    a < 0 ? ")" : "", n);
			printf("d^%s/d%s^%s %s = %s(%s - 1)...(%s - %s + 1) %s%s%s%s^(%s "
			    "- %s)\n", n, x, n, fname2(f), fnum(p), fnum(p), fnum(p), n,
			    t.p ? t.p : "", N.op[u] == O_VAR ? "" : "(", node_str(u),
			    N.op[u] == O_VAR ? "" : ")", fnum(p), n);
			free(t.p);
		}
		{
			nid g = e;
			double ff = 1;
			int ok = 1, cnt = 0, j;
			for (k = 1; k <= 6; k++) {
				struct lref r;
				g = dif(g, slot);
				ff *= p - k + 1;
				lr_open(&r, g, slot);
				for (j = 0; j < 20; j++) {
					double xx = -b / a + (j + 1) * 0.37 / fabs(a), want;
					ld got = lr(&r, xx);
					want = cdiv * ff * pow(a, k) * pow(a * xx + b, p - k);
					if (isnan(got))
						continue;
					cnt++;
					if (fabsl(got - want) > 1e-8 * fmax(1, fabs(want)))
						ok = 0;
				}
				lref_free(&r);
			}
			if (p == floor(p) && p >= 0)
				printf("  holds for every real %s and 0 <= n <= %s (0 for n "
				    "> %s); check: %s the derivatives of orders 1..6 at %d "
				    "points\n", x, fnum(p), fnum(p), ok ? "equals" :
				    "DIFFERS FROM", cnt);
			else
				printf("  holds where %s %s 0; check: %s the derivatives of "
				    "orders 1..6 at %d points\n", node_str(u), p == floor(p) ?
				    "!=" : ">", ok ? "equals" : "DIFFERS FROM", cnt);
		}
		return 1;
	} else
		return 0;
	/* exp, sin, cos: print and check orders 1..6 numerically */
	if (op == O_EXP)
		printf("d^%s/d%s^%s %s = %s\n", n, x, n, fname2(f),
		    node_str(simp(P, 0)));
	else
	{
		struct sbuf t = {0};
		if (a != 1)
			sb_printf(&t, "%s%s%s^%s*", a < 0 ? "(" : "", fnum(a),
			    a < 0 ? ")" : "", n);
		printf("d^%s/d%s^%s %s = %s%s(%s + %s*pi/2)\n", n, x, n, fname2(f),
		    t.p ? t.p : "", op == O_SIN ? "sin" : "cos", node_str(u), n);
		free(t.p);
	}
	{
		nid g = e, map[NSLOT];
		int ok = 1, cnt = 0, j;
		for (k = 1; k <= 6; k++) {
			struct lref r1, r2;
			nid want;
			g = dif(g, slot);
			memset(map, 0, sizeof map);
			map[MAXP - 1] = KI(k);
			want = subst(P, map, NSLOT);
			lr_open(&r1, g, slot);
			lr_open(&r2, want, slot);
			for (j = 0; j < 20; j++) {
				double xx = -3 + j * 0.31;
				ld g1 = lr(&r1, xx), g2 = lr(&r2, xx);
				if (isnan(g1) || isnan(g2))
					continue;
				cnt++;
				if (fabsl(g1 - g2) > 1e-9 * fmaxl(1, fabsl(g1)))
					ok = 0;
			}
			lref_free(&r1);
			lref_free(&r2);
		}
		printf("  holds for every real %s and n >= 0; check: %s the "
		    "derivatives of orders 1..6 at %d points\n", x, ok ? "equals" :
		    "DIFFERS FROM", cnt);
	}
	return 1;
}

/* ---- more antiderivative rules (called from integ.c) ---- */

/* p(x) with numeric coefficients, degree <= max: c[0..deg] */
static int
numpoly(nid e, int s, double *c, int max, int *deg)
{
	struct npoly p;
	int k;

	if (!npoly_of(e, s, &p, max))
		return 0;
	for (k = 0; k <= p.deg; k++)
		if (!num_of(p.c[k], &c[k]))
			return 0;
	*deg = p.deg;
	while (*deg > 0 && c[*deg] == 0)
		(*deg)--;
	return 1;
}

static nid
poly_node(const double *c, int deg, nid x)
{
	nid r = 0, t;
	int k;

	for (k = deg; k >= 0; k--) {
		if (c[k] == 0)
			continue;
		t = k == 0 ? KR(c[k]) : MUL(KR(c[k]), k == 1 ? x : POW(x, KI(k)));
		r = r ? ADD(r, t) : t;
	}
	return r ? r : KR(0);
}

/*
 * c/(A x^2 + B x + C) with no real roots -> atan; c/sqrt(quadratic) ->
 * atan (asin, written with atan) or log; P'/P -> log|P|; a polynomial
 * times log(a x + b) by parts.  Each result is checked by the caller.
 */
static nid c2_antideriv1(nid f, int s, nid x);

nid
c2_antideriv(nid f, int s, nid x)
{
	nid r = c2_antideriv1(f, s, x);

	return r ? simp(r, 0) : 0;
}

static int
has_epow(nid e)
{
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_TABLE ||
	    N.op[e] == O_LOOP || N.op[e] == O_UNDEF)
		return 0;
	if (N.op[e] == O_POW && const_is(N.a[e], M_E))
		return 1;
	return (N.a[e] && has_epow(N.a[e])) || (N.b[e] && has_epow(N.b[e])) ||
	    (N.c[e] && has_epow(N.c[e]));
}

static nid
c2_antideriv1(nid f, int s, nid x)
{
	double q[16], p[16], k;
	int dq, dp, i;

	if (has_epow(f)) {
		/* x e^x: written with exp, the rules apply */
		nid g = simp(f, 0);
		if (g != f && !has_epow(g))
			return antideriv(g, s, NULL);
	}

	if (N.op[f] == O_DIV && !has_slot(N.a[f], s)) {
		nid c = N.a[f], den = N.b[f];
		if (numpoly(den, s, q, 2, &dq) && dq == 2) {
			double A = q[2], B = q[1], C = q[0], D = 4 * A * C - B * B;
			if (D > 0) {
				double r = sqrt(D);
				return MUL(DIV(MUL(KR(2), c), KR(r)), mk_un(O_ATAN,
				    DIV(ADD(MUL(KR(2 * A), x), KR(B)), KR(r))));
			}
		}
		if (N.op[den] == O_SQRT && numpoly(N.a[den], s, q, 2, &dq) &&
		    dq == 2) {
			double A = q[2], B = q[1], C = q[0];
			if (A < 0) {
				double m = -B / (2 * A), K = C - B * B / (4 * A);
				if (K > 0)
					return MUL(DIV(c, KR(sqrt(-A))), mk_un(O_ASIN,
					    DIV(MUL(KR(sqrt(-A)), SUB(x, KR(m))),
					    KR(sqrt(K)))));
			} else if (A > 0) {
				return MUL(DIV(c, KR(sqrt(A))), mk_un(O_LOG, mk_un(O_ABS,
				    ADD(ADD(MUL(KR(2 * A), x), KR(B)), MUL(KR(2 * sqrt(A)),
				    den)))));
			}
		}
	}
	/* P'/P: k log|P| */
	if (N.op[f] == O_DIV && numpoly(N.a[f], s, p, 12, &dp) &&
	    numpoly(N.b[f], s, q, 12, &dq) && dq >= 1 && dp == dq - 1) {
		double kk = 0;
		int ok = 1;
		for (i = 0; i <= dp && ok; i++) {
			double d = (i + 1) * q[i + 1];
			if (d == 0) {
				ok = p[i] == 0;
				continue;
			}
			if (kk == 0)
				kk = p[i] / d;
			else if (fabs(p[i] - kk * d) > 1e-12 * fabs(p[i]))
				ok = 0;
		}
		if (ok && kk != 0)
			return MUL(KR(kk), mk_un(O_LOG, mk_un(O_ABS, N.b[f])));
	}
	/* P(x) log(a x + b): (Q - Q(r)) log(u) - integral of (Q - Q(r))/(x - r) */
	if (N.op[f] == O_MUL) {
		int w;
		for (w = 0; w < 2; w++) {
			nid P = w ? N.b[f] : N.a[f], L = w ? N.a[f] : N.b[f];
			double a, b, r, Qr = 0, S[16];
			if (N.op[L] != O_LOG || !numpoly(P, s, p, 12, &dp) ||
			    !lin_of(N.a[L], s, &a, &b))
				continue;
			r = -b / a;
			/* Q = integral of P, Q(0) = 0 */
			q[0] = 0;
			for (i = 0; i <= dp; i++)
				q[i + 1] = p[i] / (i + 1);
			dq = dp + 1;
			for (i = dq; i >= 0; i--)
				Qr = Qr * r + q[i];
			q[0] -= Qr;
			/* S = (Q - Q(r)) / (x - r), synthetic division */
			S[dq - 1] = q[dq];
			for (i = dq - 1; i >= 1; i--)
				S[i - 1] = q[i] + r * S[i];
			/* the integral of S */
			{
				double R[16];
				R[0] = 0;
				for (i = 0; i < dq; i++)
					R[i + 1] = S[i] / (i + 1);
				return SUB(MUL(poly_node(q, dq, x), L), poly_node(R, dq, x));
			}
		}
	}
	(void)k;
	return 0;
}

/* F^(ord)(x) exists?  NULL, or why not (F'(a) in calc2_syntax.inc) */
const char *
c2_point_check(struct func *f, int ord, double x)
{
	nid g = mk_cast(TY_REAL, f->body);
	int k;

	for (k = 0; k < ord; k++) {
		struct lref r;
		struct dx d;
		lr_open(&r, g, 0);
		dx_at(&r, x, &d);
		lref_free(&r);
		if (d.kind != DX_OK) {
			struct sbuf sb = {0};
			int sl[C2MAXO] = {0};
			const char *why;
			if (k)
				sb_printf(&sb, "%s %s", dname(f, sl, k), fname2(f));
			why = d.kind == DX_UNDEF ? "undefined there" :
			    dx_reason(f, k ? sb_str(&sb) : fname2(f), &d);
			free(sb.p);
			return why;
		}
		g = dif(g, 0);
	}
	return NULL;
}
