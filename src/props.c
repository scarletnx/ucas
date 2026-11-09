/*
 * Properties of functions and equations (see props.h for the commands).
 *
 * Each analysis fills a small result struct; the commands print it, and
 * identify prints one line per property from the same structs.  Every
 * result says how it was established:
 *
 *   proved, exhaustive   every point of a finite (integer) domain
 *   proved, symbolic     from the expression: polynomial roots, parity
 *                        rules, periods of trigonometric subterms with
 *                        linear arguments, interval arithmetic
 *   checked numerically  on a grid (20001 points) refined by bisection,
 *                        or at sample points; not a proof
 *
 * A real parameter without a range stands for the whole real line: the
 * symbolic arguments hold there, the numeric ones are made on [-10, 10]
 * plus probes at large |x|, and the output says so.
 */
#include "calc.h"
#include "props.h"

#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NG 20001
#define DLO (-10.0)
#define DHI 10.0
#define EXH (1ULL << 22)
#define NSAMP 65536

static const char *
fname(const struct func *f)
{
	return f->anon ? node_str(f->body) : sym_str(f->name);
}

static const char *
vname(const struct func *f, int s)
{
	return sym_str(f->p[s].name);
}

static char *
rbuf(void)
{
	static char b[32][128];
	static int k;

	return b[k++ & 31];
}

static const char *
num(double v)
{
	char *b = rbuf();

	if (v == 0)
		v = 0;   /* no -0 */
	if (isinf(v))
		snprintf(b, 128, "%s", v > 0 ? "inf" : "-inf");
	else
		snprintf(b, 128, "%.10g", v);
	return b;
}

static int64_t
gcdi(int64_t a, int64_t b)
{
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	while (b) {
		int64_t t = a % b;
		a = b;
		b = t;
	}
	return a;
}

/* the exact form v seems to have (pi/2, 3/4, sqrt(2)), "" if none */
static const char *
nice(double v)
{
	char *b = rbuf();
	long long n;
	int d;
	double k;

	b[0] = 0;
	if (!isfinite(v) || v == floor(v) || fabs(v) > 1e6)
		return b;
	for (d = 1; d <= 12; d++) {
		k = v * d / M_PI;
		n = (long long)floor(k + 0.5);
		if (n != 0 && fabs(k - n) < 1e-9 * (1 + fabs(k))) {
			int64_t g = gcdi(n, d);
			long long nn = n / g, dd = d / g;
			if (nn == 1)
				snprintf(b, 128, "pi");
			else if (nn == -1)
				snprintf(b, 128, "-pi");
			else
				snprintf(b, 128, "%lld*pi", nn);
			if (dd > 1)
				snprintf(b + strlen(b), 128 - strlen(b), "/%lld", dd);
			return b;
		}
	}
	for (d = 2; d <= 64; d++) {
		k = v * d;
		n = (long long)floor(k + 0.5);
		if (fabs(k - n) < 1e-9 * (1 + fabs(k)) && gcdi(n, d) == 1) {
			snprintf(b, 128, "%lld/%d", n, d);
			return b;
		}
	}
	k = v * v;
	n = (long long)floor(k + 0.5);
	if (n > 1 && n <= 1000 && fabs(k - n) < 1e-9 * k) {
		long long r = (long long)floor(sqrt((double)n) + 0.5);
		if (r * r != n)
			snprintf(b, 128, "%ssqrt(%lld)", v < 0 ? "-" : "", n);
	}
	return b;
}

/* a numeric result: "1.570796327 (~pi/2)" */
static const char *
numx(double v)
{
	const char *n = nice(v);
	char *b = rbuf();

	if (*n)
		snprintf(b, 128, "%s (~%s)", num(v), n);
	else
		snprintf(b, 128, "%s", num(v));
	return b;
}

static int
close_to(double a, double b, double rel)
{
	if (a == b)
		return 1;
	if (!isfinite(a) || !isfinite(b))
		return 0;
	return fabs(a - b) <= rel * (fabs(a) + fabs(b)) + 1e-300;
}

/* a derivative, or 0 when the body is not differentiable */
static nid
try_diff(nid e, int slot)
{
	jmp_buf saved;
	nid d = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		d = diff_node(e, (uint32_t)slot);
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return d;
}

/* ---- evaluating one parameter of f ---- */

struct ev {
	const struct func *f;
	int slot;
	struct plan p;
	uint32_t loc, wloc;
	int64_t *ii;
	double *ir;
	ty_t rt, xt;
};

static void
ev_open(struct ev *e, const struct func *f, nid root, int slot, int ideal)
{
	nid rs[2];
	uint32_t n = 0;

	memset(e, 0, sizeof *e);
	e->f = f;
	e->slot = slot;
	rs[n++] = root;
	if (f->where)
		rs[n++] = f->where;
	plan_build(&e->p, rs, n, ideal);
	e->loc = e->p.root[0];
	e->wloc = f->where ? e->p.root[1] : NOTOK;
	e->ii = xcalloc((size_t)MAXP * CH, sizeof *e->ii);
	e->ir = xcalloc((size_t)MAXP * CH, sizeof *e->ir);
	e->rt = N.ty[root];
	e->xt = f->p[slot].ty;
}

/* the value of another parameter */
static void
ev_fix(struct ev *e, int slot, double v)
{
	int k;

	for (k = 0; k < CH; k++) {
		e->ii[slot * CH + k] = (int64_t)v;
		e->ir[slot * CH + k] = v;
	}
}

static void
ev_close(struct ev *e)
{
	plan_free(&e->p);
	free(e->ii);
	free(e->ir);
}

static double
ival_d(ty_t t, int64_t v)
{
	if (TYK(t) == K_UINT && TYW(t) == 64)
		return (double)(uint64_t)v;
	return (double)v;
}

/*
 * y = f(x) at n points; xi (integer parameters) or x; yi gets integer
 * results exactly.  ok: defined (inside where, not undefined, not NaN).
 */
static void
ev_run(struct ev *e, const double *x, const int64_t *xi, uint32_t n,
    double *y, int64_t *yi, uint8_t *ok)
{
	uint32_t s, i, m;
	uint8_t vm[CH];
	int real = TY_FP(e->xt);

	for (s = 0; s < n; s += m) {
		m = n - s < CH ? n - s : CH;
		for (i = 0; i < m; i++) {
			if (real)
				e->ir[e->slot * CH + i] = x[s + i];
			else
				e->ii[e->slot * CH + i] = xi ? xi[s + i] :
				    (int64_t)x[s + i];
		}
		plan_run(&e->p, m, e->ii, e->ir);
		valid_mask(&e->p, e->wloc, m, vm);
		for (i = 0; i < m; i++) {
			uint8_t fl = COLF(&e->p, e->loc)[i];
			int64_t vi = COLI(&e->p, e->loc)[i];
			double v = TY_FP(e->rt) ? COLR(&e->p, e->loc)[i] :
			    ival_d(e->rt, vi);
			if (y)
				y[s + i] = v;
			if (yi && TY_FP(e->rt)) {
				/* a real result: its bits, so equal values compare equal */
				double w = v == 0 ? 0 : v;
				memcpy(&vi, &w, sizeof vi);
			}
			if (yi)
				yi[s + i] = vi;
			ok[s + i] = vm[i] && !(fl & F_UNDEF) && !isnan(v);
		}
	}
}

static int
ev1(struct ev *e, double x, double *y)
{
	uint8_t ok;

	ev_run(e, &x, NULL, 1, y, NULL, &ok);
	return ok;
}

static double urand(double a, double b);
static int evm(struct ev *e, const double *v, double *y);

/* ---- the domain of one parameter ---- */

struct dom {
	int isint, unranged, sampled;
	int64_t lo, hi;
	double rlo, rhi;
	uint64_t n;
};

static void
dom_of(const struct func *f, int s, struct dom *d)
{
	const struct param *p = &f->p[s];
	uint64_t span;

	memset(d, 0, sizeof *d);
	if (TY_FP(p->ty)) {
		d->unranged = !p->ranged;
		d->rlo = p->ranged ? p->rlo : DLO;
		d->rhi = p->ranged ? p->rhi : DHI;
		return;
	}
	d->isint = 1;
	if (TYK(p->ty) == K_INT && !p->ranged) {
		d->lo = -4096;
		d->hi = 4096;
		d->unranged = 1;
	} else if (TYK(p->ty) == K_UINT && TYW(p->ty) == 64 && !p->ranged) {
		d->lo = 0;
		d->hi = INT64_MAX;
	} else
		param_bounds(p, &d->lo, &d->hi);
	if (d->hi < d->lo) {
		d->n = 0;
		return;
	}
	span = (uint64_t)d->hi - (uint64_t)d->lo;
	d->n = span == UINT64_MAX ? UINT64_MAX : span + 1;
	d->sampled = d->n > EXH;
	d->rlo = (double)d->lo;
	d->rhi = (double)d->hi;
}

/* the points examined: all of an integer domain or a sample; a real grid */
static uint32_t
dom_points(const struct dom *d, uint32_t ng, double **xs, int64_t **xi)
{
	uint32_t n, i;

	if (d->isint) {
		n = d->sampled ? NSAMP : (uint32_t)d->n;
		*xi = xcalloc(n ? n : 1, sizeof **xi);
		*xs = xcalloc(n ? n : 1, sizeof **xs);
		for (i = 0; i < n; i++) {
			if (d->sampled)
				(*xi)[i] = d->lo + (int64_t)((double)i / (n - 1) *
				    ((double)d->hi - (double)d->lo));
			else
				(*xi)[i] = d->lo + (int64_t)i;
			(*xs)[i] = (double)(*xi)[i];
		}
		if (d->sampled)
			(*xi)[n - 1] = d->hi;
		return n;
	}
	n = ng;
	*xi = NULL;
	*xs = xcalloc(n, sizeof **xs);
	for (i = 0; i < n; i++)
		(*xs)[i] = d->rlo + (d->rhi - d->rlo) * i / (n - 1);
	return n;
}

static const char *
dom_text(const struct dom *d)
{
	char *b = rbuf();

	if (d->isint)
		snprintf(b, 128, "%lld..%lld", (long long)d->lo, (long long)d->hi);
	else if (d->unranged)
		snprintf(b, 128, "all real x");
	else
		snprintf(b, 128, "[%s, %s]", num(d->rlo), num(d->rhi));
	return b;
}

/* how a numeric scan covered the domain */
static const char *
scan_text(const struct dom *d, const char *x)
{
	char *b = rbuf();

	if (d->isint && !d->sampled)
		snprintf(b, 128, "proved, exhaustive over all %llu points",
		    (unsigned long long)d->n);
	else if (d->isint)
		snprintf(b, 128, "checked on %d of %llu points (sampled; not "
		    "proved)", NSAMP, (unsigned long long)d->n);
	else if (d->unranged)
		snprintf(b, 128, "checked numerically on [%g, %g] (%s has no "
		    "range; give it one)", d->rlo, d->rhi, x);
	else
		snprintf(b, 128, "checked numerically, grid of %d points", NG);
	return b;
}

/* real analyses need one real parameter (or the one asked for) */
static int
one_param(const struct func *f, const char *cmd)
{
	if (f->np == 0) {
		printf("%s %s: constant (no parameters)\n", cmd, fname(f));
		return 0;
	}
	if (f->np > 1) {
		printf("%s %s: %d parameters; give the others values, e.g. "
		    "%s %s(%s, 1)\n", cmd, fname(f), f->np, cmd,
		    f->anon ? "f" : sym_str(f->name), vname(f, 0));
		return 0;
	}
	return 1;
}

static const char *
mult_text(int k)
{
	char *b = rbuf();

	snprintf(b, 128, " (multiplicity %d)", k);
	return b;
}

/* ---- polynomials with numeric coefficients ---- */

struct rpoly {
	int deg;
	double c[13];
};

static int
rpoly_of(nid e, int s, struct rpoly *r)
{
	struct npoly p;
	int i;

	if (!npoly_of(e, s, &p, 12))
		return 0;
	r->deg = p.deg;
	for (i = 0; i <= p.deg; i++)
		if (!num_of(p.c[i], &r->c[i]) || !isfinite(r->c[i]))
			return 0;
	while (r->deg > 0 && r->c[r->deg] == 0)
		r->deg--;
	return 1;
}

static double
rpoly_at(const struct rpoly *r, double x)
{
	double v = 0;
	int i;

	for (i = r->deg; i >= 0; i--)
		v = v * x + r->c[i];
	return v;
}

/* the sign of D inside (a, b), where it has no sign change */
static int
rp_sign_in(const struct rpoly *D, double a, double b)
{
	static const double fr[] = {0.5, 0.25, 0.75, 0.125, 0.875, 0.3, 0.7};
	int k;

	for (k = 0; k < 7; k++) {
		double x = isinf(a) && isinf(b) ? (fr[k] - 0.5) * 10 + 0.123 :
		    isinf(a) ? b - 1 - fr[k] : isinf(b) ? a + 1 + fr[k] :
		    a + (b - a) * fr[k], v = rpoly_at(D, x);
		if (v != 0)
			return v > 0 ? 1 : -1;
	}
	return 0;
}

static void
rpoly_deriv(const struct rpoly *r, struct rpoly *d)
{
	int i;

	d->deg = r->deg > 0 ? r->deg - 1 : 0;
	d->c[0] = 0;
	for (i = 1; i <= r->deg; i++)
		d->c[i - 1] = r->c[i] * i;
}

/* roots; rational coefficients are scaled to integers for exact roots */
static int
rpoly_roots(const struct rpoly *P, struct root *r)
{
	double c[16];
	int64_t L = 1;
	struct q q;
	int i;

	if (P->deg < 1)
		return 0;
	for (i = 0; i <= P->deg; i++) {
		if (!q_from_double(P->c[i], &q) || q.d > 1000000) {
			L = 0;
			break;
		}
		L = L / gcdi(L, q.d) * q.d;
		if (L > 1000000000) {
			L = 0;
			break;
		}
	}
	for (i = 0; i <= P->deg; i++)
		c[i] = L ? floor(P->c[i] * L + 0.5) : P->c[i];
	return poly_roots(c, P->deg, r);
}

static const char *
root_str(const struct root *r)
{
	char *b = rbuf();

	if (r->exact) {
		if (r->qr.d == 1)
			snprintf(b, 128, "%lld", (long long)r->qr.n);
		else
			snprintf(b, 128, "%lld/%lld", (long long)r->qr.n,
			    (long long)r->qr.d);
	} else if (r->quad && r->im == 0) {
		/* (b ± k sqrt(m))/a with the square part of D taken out */
		int64_t bb = r->qb.n, D = r->qd.n, a = r->qa.n, k = 1, m = D, j, g;
		int sg = r->quad > 0 ? 1 : -1;
		char kt[32] = "";
		for (j = 2; j * j <= m; j++)
			while (m % (j * j) == 0) {
				m /= j * j;
				k *= j;
			}
		g = gcdi(gcdi(bb, k), a);
		if (g > 1)
			bb /= g, k /= g, a /= g;
		if (a < 0)
			bb = -bb, a = -a, sg = -sg;
		if (k != 1)
			snprintf(kt, sizeof kt, "%lld*", (long long)k);
		if (bb == 0)
			snprintf(b, 128, "%s%ssqrt(%lld)", sg < 0 ? "-" : "", kt,
			    (long long)m);
		else
			snprintf(b, 128, "%s%lld %c %ssqrt(%lld)%s", a != 1 ? "(" : "",
			    (long long)bb, sg > 0 ? '+' : '-', kt, (long long)m,
			    a != 1 ? ")" : "");
		if (a != 1)
			snprintf(b + strlen(b), 128 - strlen(b), "/%lld", (long long)a);
	} else if (r->im != 0)
		snprintf(b, 128, "%s%s%si", r->re == 0 ? "" : num(r->re),
		    r->re == 0 ? "±" : " ± ", r->im == 1 ? "" : num(r->im));
	else
		snprintf(b, 128, "%s", numx(r->re));
	return b;
}

static nid
rpoly_node(const struct rpoly *r, nid x)
{
	nid e = 0, t;
	int i;

	for (i = r->deg; i >= 0; i--) {
		double c = r->c[i];
		nid k;
		if (c == 0 && !(i == 0 && e == 0))
			continue;
		k = c == floor(c) && fabs(c) < 1e15 ? KI((int64_t)c) : KR(c);
		t = i == 0 ? k : i == 1 ? x : POW(x, KI(i));
		if (i > 0 && c != 1)
			t = c == -1 ? mk_un(O_NEG, t) : MUL(k, t);
		e = e ? ADD(e, t) : t;
	}
	return e;
}

/* ---- monotonicity ---- */

enum { MO_NONE = 0, MO_INC = 1, MO_SINC = 2, MO_CONST = 3, MO_DEC = -1,
	MO_SDEC = -2 };
#define MAXPC 48

struct mono {
	int verdict;
	int proved;              /* 2 exhaustive, 1 symbolic, 0 numeric */
	int npc;
	double pa[MAXPC], pb[MAXPC];
	int pdir[MAXPC], pstrict[MAXPC];
	int ntotal;
	int hasinc, hasdec;      /* witnesses x1 < x2 */
	double ia, ib, da, db;   /* f(ia) < f(ib); f(da) > f(db) */
	double iya, iyb, dya, dyb;
	int ideal_verdict;       /* integer types: without wrapping */
	char how[256];
	struct dom d;
};

/* report.h's proof kinds for proved = 2, 1, 0 */
static const char *
proofkind(int p)
{
	return p == 2 ? "exhaustive" : p == 1 ? "symbolic" : "numeric";
}

static const char *
mono_word(int v)
{
	switch (v) {
	case MO_SINC:
		return "strictly increasing";
	case MO_INC:
		return "non-decreasing (increasing, not strictly)";
	case MO_CONST:
		return "constant";
	case MO_DEC:
		return "non-increasing (decreasing, not strictly)";
	case MO_SDEC:
		return "strictly decreasing";
	}
	return "not monotonic";
}

static int
step_sign(double a, double b, int exact)
{
	if (exact)
		return b > a ? 1 : b < a ? -1 : 0;
	if (fabs(b - a) <= 1e-13 * (fabs(a) + fabs(b)))
		return 0;
	return b > a ? 1 : -1;
}

static void
mono_piece(struct mono *m, double a, double b, int dir, int strict)
{
	if (m->npc < MAXPC) {
		m->pa[m->npc] = a;
		m->pb[m->npc] = b;
		m->pdir[m->npc] = dir;
		m->pstrict[m->npc] = strict;
		m->npc++;
	}
	m->ntotal++;
}

/*
 * The sequence y over sorted x: the verdict, the maximal pieces (split
 * where the step changes sign or at a gap where f is undefined) and
 * witnesses.  Comparisons run across gaps: 1/x is not monotonic.
 */
static void
mono_seq(const double *x, const double *y, const uint8_t *ok, uint32_t n,
    int exact, struct mono *m)
{
	uint32_t i;
	int have = 0, gap = 0, cur = 0, strict = 1, s, flat = 0;
	double px = 0, py = 0, a = 0;

	m->npc = m->ntotal = 0;
	m->hasinc = m->hasdec = 0;
	for (i = 0; i < n; i++) {
		if (!ok[i]) {
			gap |= have;
			continue;
		}
		if (!have) {
			have = 1;
			a = px = x[i];
			py = y[i];
			continue;
		}
		s = step_sign(py, y[i], exact);
		if (s > 0 && !m->hasinc) {
			m->hasinc = 1;
			m->ia = px, m->ib = x[i], m->iya = py, m->iyb = y[i];
		}
		if (s < 0 && !m->hasdec) {
			m->hasdec = 1;
			m->da = px, m->db = x[i], m->dya = py, m->dyb = y[i];
		}
		if (s == 0)
			flat = 1;
		if (gap) {
			mono_piece(m, a, px, cur, strict);
			a = x[i];
			cur = 0;
			strict = 1;
			gap = 0;
		} else if (s != 0 && cur != 0 && s != cur) {
			mono_piece(m, a, px, cur, strict);
			a = px;
			cur = s;
			strict = 1;
		} else {
			if (s == 0)
				strict = 0;
			else if (cur == 0)
				cur = s;
		}
		px = x[i];
		py = y[i];
	}
	if (have)
		mono_piece(m, a, px, cur, strict);
	if (!have)
		m->verdict = MO_CONST;
	else if (!m->hasinc && !m->hasdec)
		m->verdict = MO_CONST;
	else if (!m->hasdec)
		m->verdict = flat ? MO_INC : MO_SINC;
	else if (!m->hasinc)
		m->verdict = flat ? MO_DEC : MO_SDEC;
	else
		m->verdict = MO_NONE;
}

/* a piece boundary at a grid point: the sign change of f' near it */
static double
refine_turn(struct ev *de, double a, double b)
{
	double fa, fb, fm, m;
	int k;

	if (!de || !ev1(de, a, &fa) || !ev1(de, b, &fb) || (fa > 0) == (fb > 0))
		return NAN;
	for (k = 0; k < 80; k++) {
		m = 0.5 * (a + b);
		if (!ev1(de, m, &fm))
			return NAN;
		if (fm == 0)
			return m;
		if ((fm > 0) == (fa > 0))
			a = m, fa = fm;
		else
			b = m;
	}
	return 0.5 * (a + b);
}

static void
mono_analyse(struct func *f, int s, struct mono *m)
{
	struct ev e;
	double *xs, *ys;
	int64_t *xi;
	uint8_t *ok;
	uint32_t n;
	struct rpoly P;

	memset(m, 0, sizeof *m);
	dom_of(f, s, &m->d);
	if (!has_slot(f->body, s) && !f->where) {
		m->verdict = MO_CONST;
		m->proved = 1;
		snprintf(m->how, sizeof m->how, "proved: %s does not depend on %s",
		    fname(f), vname(f, s));
		return;
	}
	/* a real polynomial: the real roots of f' split it */
	if (!m->d.isint && !f->where && rpoly_of(f->body, s, &P) &&
	    TYK(f->p[s].ty) == K_REAL) {
		struct rpoly D;
		struct root r[16];
		double cut[16], lo = m->d.rlo, hi = m->d.rhi;
		int nr, nc = 0, k, j, allexact = 1;
		struct sbuf sb = {0};
		if (P.deg == 0) {
			m->verdict = MO_CONST;
			m->proved = 1;
			snprintf(m->how, sizeof m->how, "proved: a constant");
			return;
		}
		rpoly_deriv(&P, &D);
		nr = D.deg >= 1 ? rpoly_roots(&D, r) : 0;
		for (k = 0; k < nr; k++) {
			if (r[k].im != 0)
				continue;
			if (!r[k].exact && !r[k].quad)
				allexact = 0;
			sb_printf(&sb, "%s%s%s", sb.n ? ", " : "", root_str(&r[k]),
			    r[k].mult > 1 ? mult_text(r[k].mult) : "");
			if (r[k].mult % 2 == 0)
				continue;
			if (!m->d.unranged && (r[k].re <= lo || r[k].re >= hi))
				continue;
			cut[nc++] = r[k].re;
		}
		/* sort the cuts */
		for (k = 1; k < nc; k++)
			for (j = k; j > 0 && cut[j] < cut[j - 1]; j--) {
				double t = cut[j];
				cut[j] = cut[j - 1];
				cut[j - 1] = t;
			}
		for (k = 0; k <= nc; k++) {
			double a = k == 0 ? (m->d.unranged ? -INFINITY : lo) :
			    cut[k - 1];
			double b = k == nc ? (m->d.unranged ? INFINITY : hi) : cut[k];
			mono_piece(m, a, b, rp_sign_in(&D, a, b) > 0 ? 1 : -1, 1);
		}
		for (k = 0; k < m->npc; k++) {
			double a = isinf(m->pa[k]) ? m->pb[k] - 1 : m->pa[k];
			double b = isinf(m->pb[k]) ? m->pa[k] + 1 : m->pb[k];
			if (m->pdir[k] > 0 && !m->hasinc) {
				m->hasinc = 1;
				m->ia = a, m->ib = b;
				m->iya = rpoly_at(&P, a), m->iyb = rpoly_at(&P, b);
			}
			if (m->pdir[k] < 0 && !m->hasdec) {
				m->hasdec = 1;
				m->da = a, m->db = b;
				m->dya = rpoly_at(&P, a), m->dyb = rpoly_at(&P, b);
			}
		}
		m->verdict = m->npc == 1 ? (m->pdir[0] > 0 ? MO_SINC : MO_SDEC) :
		    MO_NONE;
		m->proved = 1;
		snprintf(m->how, sizeof m->how, "proved symbolically: %s' = %s; "
		    "its real roots: %s%s; f' keeps its sign between roots of odd "
		    "multiplicity%s", fname(f), node_str(rpoly_node(&D,
		    mk_var(0, f->p[s].name, TY_REAL))), sb.n ? sb.p : "none",
		    sb.n ? (allexact ? " (exact)" : " (numeric)") : "",
		    nc == 0 && nr > 0 ? " (zeros of f' of even multiplicity are "
		    "isolated, so it stays strict)" : "");
		free(sb.p);
		return;
	}
	ev_open(&e, f, f->body, s, 0);
	n = dom_points(&m->d, NG, &xs, &xi);
	ys = xcalloc(n, sizeof *ys);
	ok = xcalloc(n, 1);
	ev_run(&e, xs, xi, n, ys, NULL, ok);
	mono_seq(xs, ys, ok, n, m->d.isint, m);
	m->proved = m->d.isint && !m->d.sampled ? 2 : 0;
	if (m->d.isint) {
		/* without wrapping: the same function on unbounded integers */
		struct ev ie;
		struct mono mi;
		ev_open(&ie, f, f->body, s, 1);
		ev_run(&ie, xs, xi, n, ys, NULL, ok);
		memset(&mi, 0, sizeof mi);
		mono_seq(xs, ys, ok, n, 1, &mi);
		m->ideal_verdict = mi.verdict;
		ev_close(&ie);
		snprintf(m->how, sizeof m->how, "%s (forward difference "
		    "f(%s+1) - f(%s))", scan_text(&m->d, vname(f, s)), vname(f, s),
		    vname(f, s));
	} else {
		/* refine the turning points on f' */
		nid d = try_diff(f->body, s);
		struct ev de, *dep = NULL;
		double h = (m->d.rhi - m->d.rlo) / (NG - 1);
		int k;
		if (d) {
			ev_open(&de, f, d, s, 0);
			dep = &de;
		}
		for (k = 1; k < m->npc && k < MAXPC; k++) {
			if (m->pa[k] == m->pb[k - 1]) {
				double c = refine_turn(dep, m->pa[k] - h, m->pa[k] + h);
				if (!isnan(c))
					m->pa[k] = m->pb[k - 1] = c;
			}
		}
		if (dep)
			ev_close(dep);
		/* witnesses from whole pieces read better than grid steps */
		{
			int gi = 0, gd = 0;
			for (k = 0; k < m->npc; k++) {
				double ya, yb;
				if (m->pa[k] >= m->pb[k] || !ev1(&e, m->pa[k], &ya) ||
				    !ev1(&e, m->pb[k], &yb))
					continue;
				if (m->pdir[k] > 0 && !gi && yb > ya) {
					gi = 1;
					m->ia = m->pa[k], m->ib = m->pb[k];
					m->iya = ya, m->iyb = yb;
				}
				if (m->pdir[k] < 0 && !gd && yb < ya) {
					gd = 1;
					m->da = m->pa[k], m->db = m->pb[k];
					m->dya = ya, m->dyb = yb;
				}
			}
		}
		snprintf(m->how, sizeof m->how, "%s%s",
		    scan_text(&m->d, vname(f, s)), d ? "; turning points refined "
		    "on f'" : "");
	}
	free(xs);
	free(xi);
	free(ys);
	free(ok);
	ev_close(&e);
}

int
fn_monotonic(struct func *f, double lo, double hi)
{
	struct mono m;
	struct func t;

	if (f->np != 1)
		return 0;
	t = *f;
	if (lo <= hi) {
		t.p[0].ranged = 1;
		t.p[0].rlo = lo;
		t.p[0].rhi = hi;
		t.p[0].lo = (int64_t)ceil(lo);
		t.p[0].hi = (int64_t)floor(hi);
	}
	mono_analyse(&t, 0, &m);
	return m.verdict;
}

static const char *
iv_text(double a, double b, int isint)
{
	char *r = rbuf();

	if (isint)
		snprintf(r, 128, "%lld..%lld", (long long)a, (long long)b);
	else
		snprintf(r, 128, "%s%s, %s%s", isinf(a) ? "(" : "[", isinf(a) ?
		    "-inf" : numx(a), isinf(b) ? "inf" : numx(b),
		    isinf(b) ? ")" : "]");
	return r;
}

static void
mono_print(struct func *f, int s, const struct mono *m)
{
	int k;
	const char *x = vname(f, s);

	rep_begin("monotonic");
	rep_str("var", x);
	rep_str("verdict", mono_word(m->verdict));
	rep_str("on", m->d.unranged && m->proved != 1 ? iv_text(m->d.rlo,
	    m->d.rhi, 0) : dom_text(&m->d));
	rep_list("intervals");
	for (k = 0; k < m->npc; k++) {
		rep_obj(NULL);
		rep_num("lo", m->pa[k]);
		rep_num("hi", m->pb[k]);
		rep_str("direction", m->pdir[k] > 0 ? "increasing" : m->pdir[k] <
		    0 ? "decreasing" : "constant");
		rep_bool("strict", m->pstrict[k]);
		rep_end();
	}
	rep_end();
	if (m->verdict == MO_NONE) {
		rep_list("witness_not_decreasing");
		rep_num(NULL, m->ia);
		rep_num(NULL, m->ib);
		rep_end();
		rep_list("witness_not_increasing");
		rep_num(NULL, m->da);
		rep_num(NULL, m->db);
		rep_end();
	}
	rep_proof(proofkind(m->proved), m->how);
	printf("  %s on %s\n", mono_word(m->verdict), m->d.unranged &&
	    m->proved != 1 ? iv_text(m->d.rlo, m->d.rhi, 0) : dom_text(&m->d));
	if (m->verdict == MO_NONE || (m->npc > 1 && m->proved == 1)) {
		printf("  maximal intervals of monotonicity:\n");
		for (k = 0; k < m->npc; k++)
			printf("    %s in %s: %s%s\n", x, iv_text(m->pa[k], m->pb[k],
			    m->d.isint), m->pdir[k] > 0 ? "increasing" :
			    m->pdir[k] < 0 ? "decreasing" : "constant",
			    m->pdir[k] && !m->pstrict[k] ? " (not strictly)" : "");
		if (m->ntotal > m->npc)
			printf("    ... %d intervals in all\n", m->ntotal);
	}
	if (m->verdict == MO_NONE) {
		printf("  witness: %s = %s < %s = %s but f(%s) = %s %s f(%s) = %s"
		    " (not decreasing)\n", x, num(m->ia), x, num(m->ib), num(m->ia),
		    num(m->iya), "<", num(m->ib), num(m->iyb));
		printf("  witness: %s = %s < %s = %s but f(%s) = %s %s f(%s) = %s"
		    " (not increasing)\n", x, num(m->da), x, num(m->db), num(m->da),
		    num(m->dya), ">", num(m->db), num(m->dyb));
	}
	if (m->d.isint && m->verdict == MO_NONE && m->ideal_verdict != MO_NONE)
		printf("  note: without wrapping (as int) it is %s; the machine "
		    "type wraps: f(%s) = %s, f(%s) = %s\n",
		    mono_word(m->ideal_verdict), num(m->da), num(m->dya),
		    num(m->db), num(m->dyb));
	printf("  %s\n", m->how);
}

/* ---- periods ---- */

#define MAXAT 16
struct patom {
	double p;
	int pi, exact;       /* p = q, times pi when pi */
	struct q q;
	nid at;
};

struct per {
	int status;          /* 1 periodic, 0 not, -1 constant, -2 unknown */
	double p;
	int exact, pi;
	struct q q;
	int proved;          /* 1 symbolic, 2 exhaustive, 0 numeric */
	int nat;
	struct patom at[MAXAT];
	nid bad;             /* where x appears outside a periodic subterm */
	int cyclic;          /* integer: the pattern also wraps around */
	int nonneg;          /* integer: periodic only for x >= 0 */
	char why[400], how[400];
	struct dom d;
};

static const char *
grow_text(double x, double y)
{
	char *b = rbuf();

	snprintf(b, 128, "; it grows without bound (f(%g) = %s), and a "
	    "continuous periodic function is bounded", x, num(y));
	return b;
}

static const char *
q_text(struct q q, int pi)
{
	char *b = rbuf();

	if (!pi)
		snprintf(b, 128, q.d == 1 ? "%lld" : "%lld/%lld",
		    (long long)q.n, (long long)q.d);
	else if (q.d == 1)
		snprintf(b, 128, q.n == 1 ? "pi" : "%lld*pi", (long long)q.n);
	else
		snprintf(b, 128, q.n == 1 ? "pi/%lld" : "%lld*pi/%lld",
		    q.n == 1 ? (long long)q.d : (long long)q.n, (long long)q.d);
	return b;
}

static const char *
per_text(const struct per *r)
{
	char *b = rbuf();

	if (r->exact && !(r->q.d == 1 && !r->pi))
		snprintf(b, 128, "%s ≈ %s", q_text(r->q, r->pi), num(r->p));
	else if (r->exact)
		snprintf(b, 128, "%s", q_text(r->q, 0));
	else
		snprintf(b, 128, "%s", numx(r->p));
	return b;
}

/* the slope of a linear argument, else 0 */
static double
lin_slope(nid a, int s)
{
	nid d = try_diff(a, s);
	double k;

	if (!d || node_slots(d) || !num_of(d, &k) || !isfinite(k))
		return 0;
	return k;
}

/* a subterm periodic in x with period base/|k| (times pi when pi) */
static int
per_add(struct per *r, nid e, int pi, double base, double k)
{
	struct patom *t;
	struct q kq;

	if (k == 0 || r->nat == MAXAT)
		return 0;
	t = &r->at[r->nat++];
	t->at = e;
	t->pi = pi;
	t->p = base * (pi ? M_PI : 1) / fabs(k);
	t->exact = 0;
	if (base == floor(base) && q_from_double(fabs(k), &kq) && kq.d <= 1000 &&
	    kq.n <= 1000000)
		t->exact = q_make((int64_t)base * kq.d, kq.n, &t->q);
	return 1;
}

static int
per_walk(nid e, int s, struct per *r)
{
	double k, m;

	if (!has_slot(e, s))
		return 1;
	switch (N.op[e]) {
	case O_SIN:
	case O_COS:
	case O_SEC:
	case O_CSC:
		if (per_add(r, e, 1, 2, lin_slope(N.a[e], s)))
			return 1;
		break;
	case O_TAN:
	case O_COT:
		if (per_add(r, e, 1, 1, lin_slope(N.a[e], s)))
			return 1;
		break;
	case O_MOD:
		if (!has_slot(N.b[e], s) && num_of(N.b[e], &m) && m != 0 &&
		    (k = lin_slope(N.a[e], s)) != 0 && per_add(r, e, 0, fabs(m), k))
			return 1;
		break;
	case O_AND: {
		nid c = has_slot(N.a[e], s) ? N.b[e] : N.a[e];
		nid v = c == N.b[e] ? N.a[e] : N.b[e];
		int64_t mk;
		if (!has_slot(c, s) && int_of(c, &mk) && mk >= 0 &&
		    lin_slope(v, s) != 0 &&
		    per_add(r, e, 0, (double)(1ULL << bits_for((uint64_t)mk)),
		    lin_slope(v, s)))
			return 1;
		break;
	}
	case O_VAR:
	case O_LOOP:
		break;
	default:
		if (!(per_walk(N.a[e], s, r) && per_walk(N.b[e], s, r) &&
		    per_walk(N.c[e], s, r))) {
			/* name the smallest compound term around a bare x */
			if (r->bad && N.op[r->bad] == O_VAR)
				r->bad = e;
			return 0;
		}
		return 1;
	}
	if (!r->bad)
		r->bad = e;
	return 0;
}

static int64_t
lcmi(int64_t a, int64_t b)
{
	int64_t g = gcdi(a, b);

	return g ? a / g * b : 0;
}

/* the least common multiple of the subterms' periods; 0: incommensurate */
static int
per_combine(struct per *r)
{
	int k;

	r->p = r->at[0].p;
	r->exact = r->at[0].exact;
	r->pi = r->at[0].pi;
	r->q = r->at[0].q;
	for (k = 1; k < r->nat; k++) {
		const struct patom *t = &r->at[k];
		struct q ratio;
		if (r->exact && t->exact && r->pi == t->pi) {
			int64_t n = lcmi(r->q.n, t->q.n), d = gcdi(r->q.d, t->q.d);
			if (n <= 0 || n > 1000000000)
				return 0;
			q_make(n, d, &r->q);
			r->p = q_val(r->q) * (r->pi ? M_PI : 1);
			continue;
		}
		if (r->exact && t->exact) {
			snprintf(r->why, sizeof r->why, "the subterms' periods %s and "
			    "%s are incommensurate (their ratio involves pi, which is "
			    "irrational)", q_text(r->q, r->pi), q_text(t->q, t->pi));
			return 0;
		}
		if (q_from_double(t->p / r->p, &ratio) && ratio.d <= 1000 &&
		    ratio.n <= 1000 && close_to(q_val(ratio), t->p / r->p, 1e-12)) {
			r->p *= (double)ratio.n;
			if (r->exact)
				q_make(r->q.n * ratio.n, r->q.d, &r->q);
			continue;
		}
		snprintf(r->why, sizeof r->why, "the subterms' periods %s and %s "
		    "are incommensurate (ratio %s is not a ratio of small "
		    "integers)", numx(r->p), numx(t->p), num(t->p / r->p));
		return 0;
	}
	return 1;
}

/* f(x + p) = f(x) at npts points of [lo, hi] (formula, not the range) */
static int
per_holds(struct ev *e, double lo, double hi, double p, int npts)
{
	double x[1024], y[1024];
	uint8_t ok[1024];
	int i, cmp = 0, n = npts > 512 ? 512 : npts;

	for (i = 0; i < n; i++) {
		x[i] = lo + (hi - lo) * (i + 0.3711) / n;
		x[n + i] = x[i] + p;
	}
	ev_run(e, x, NULL, (uint32_t)(2 * n), y, NULL, ok);
	for (i = 0; i < n; i++) {
		if (ok[i] != ok[n + i])
			return 0;
		if (!ok[i] || fabs(y[i]) > 1e8)
			continue;
		if (!close_to(y[i], y[n + i], 1e-7) && fabs(y[i] - y[n + i]) > 1e-9)
			return 0;
		cmp++;
	}
	return cmp >= n / 4;
}

static double
per_err(struct ev *e, double lo, double hi, double p)
{
	double x[64], y[64], m = 0;
	uint8_t ok[64];
	int i;

	for (i = 0; i < 32; i++) {
		x[i] = lo + (hi - lo) * (i + 0.2913) / 32;
		x[32 + i] = x[i] + p;
	}
	ev_run(e, x, NULL, 64, y, NULL, ok);
	for (i = 0; i < 32; i++) {
		if (ok[i] != ok[32 + i])
			return INFINITY;
		if (ok[i] && fabs(y[i] - y[32 + i]) > m)
			m = fabs(y[i] - y[32 + i]);
	}
	return m;
}

/* the smallest p/k (k <= 64) that is still a period */
static int
per_minimise(struct ev *e, struct per *r, double lo, double hi)
{
	int k;

	for (k = 64; k >= 2; k--)
		if (per_holds(e, lo, hi, r->p / k, 997)) {
			r->p /= k;
			if (r->exact)
				q_make(r->q.n, r->q.d * k, &r->q);
			return k;
		}
	return 1;
}

static void
per_int(struct func *f, int s, struct per *r)
{
	struct ev e;
	double *xs;
	int64_t *xi, *yi, n, p, i, start = 0, pass;
	uint8_t *ok;
	uint32_t np;

	ev_open(&e, f, f->body, s, 0);
	np = dom_points(&r->d, 0, &xs, &xi);
	if (r->d.sampled || np > 65536) {
		/* a window at the start of the domain */
		np = 65536;
		for (i = 0; i < np; i++)
			xi[i] = r->d.lo + i;
	}
	yi = xcalloc(np, sizeof *yi);
	ok = xcalloc(np, 1);
	ev_run(&e, xs, xi, np, NULL, yi, ok);
	for (pass = 0; pass < 2; pass++) {
		n = np - start;
		for (p = 1; p <= n / 2; p++) {
			for (i = start; i + p < (int64_t)np; i++)
				if (ok[i] != ok[i + p] || (ok[i] && yi[i] != yi[i + p]))
					break;
			if (i + p >= (int64_t)np)
				break;
		}
		if (p <= n / 2)
			break;
		/* C's % truncates: x % 7 repeats only on x >= 0 */
		if (pass == 0 && r->d.lo < 0 && r->d.hi > 0)
			start = -r->d.lo;
		else
			break;
	}
	if (p <= n / 2) {
		r->status = 1;
		r->p = (double)p;
		r->exact = 1;
		q_make(p, 1, &r->q);
		r->nonneg = start > 0;
		if (p == 1) {
			r->status = -1;
		}
		r->proved = r->d.sampled || r->d.unranged ? 0 : 2;
		r->cyclic = !r->d.sampled && !start && r->d.n % (uint64_t)p == 0;
		snprintf(r->how, sizeof r->how, "%s: f(%s + %lld) = f(%s) for "
		    "every %s%s with both in %lld..%lld, and no smaller shift "
		    "works", r->proved ? "proved, exhaustive" : "checked (not "
		    "proved)", vname(f, s), (long long)p, vname(f, s), vname(f, s),
		    start ? " >= 0" : "", (long long)xi[start], (long long)xi[np - 1]);
	} else {
		r->status = 0;
		r->proved = r->d.sampled || r->d.unranged ? 0 : 2;
		snprintf(r->why, sizeof r->why, "no shift p <= %lld has f(%s + p) "
		    "= f(%s) for every %s with both in %lld..%lld (%s)",
		    (long long)(np / 2), vname(f, s), vname(f, s), vname(f, s),
		    (long long)xi[0], (long long)xi[np - 1], r->proved ?
		    "exhaustive" : "a window of the domain");
	}
	free(xs);
	free(xi);
	free(yi);
	free(ok);
	ev_close(&e);
}

static void
per_analyse(struct func *f, int s, struct per *r)
{
	struct ev e;
	double lo, hi, ymin = INFINITY, ymax = -INFINITY;
	int k;

	memset(r, 0, sizeof *r);
	dom_of(f, s, &r->d);
	if (!has_slot(f->body, s)) {
		r->status = -1;
		r->proved = 1;
		return;
	}
	per_walk(f->body, s, r);
	if (r->d.isint) {
		per_int(f, s, r);
		return;
	}
	lo = r->d.unranged ? DLO : r->d.rlo;
	hi = r->d.unranged ? DHI : r->d.rhi;
	if (hi - lo < 1)
		hi = lo + 1;
	ev_open(&e, f, f->body, s, 0);
	/* the other parameters: a value that is not special */
	for (k = 0; k < f->np; k++)
		if (k != s)
			ev_fix(&e, k, 0.7310585786);
	{
		double x[2001], y[2001];
		uint8_t ok[2001];
		for (k = 0; k < 2001; k++)
			x[k] = lo + (hi - lo) * k / 2000.0;
		ev_run(&e, x, NULL, 2001, y, NULL, ok);
		for (k = 0; k < 2001; k++)
			if (ok[k] && fabs(y[k]) < 1e8) {
				if (y[k] < ymin)
					ymin = y[k];
				if (y[k] > ymax)
					ymax = y[k];
			}
		if (ymin <= ymax && close_to(ymin, ymax, 1e-12)) {
			r->status = -1;
			r->proved = 0;
			snprintf(r->how, sizeof r->how, "f = %s at all 2001 sample "
			    "points of [%g, %g] (checked numerically)", num(ymin), lo,
			    hi);
			ev_close(&e);
			return;
		}
	}
	if (!r->bad && r->nat > 0) {
		struct sbuf sb = {0};
		int kk;
		for (k = 0; k < r->nat; k++)
			sb_printf(&sb, "%s%s has period %s", k ? ", " : "",
			    node_str(r->at[k].at), r->at[k].exact ?
			    q_text(r->at[k].q, r->at[k].pi) : numx(r->at[k].p));
		if (!per_combine(r)) {
			r->status = 0;
			r->proved = 0;
			snprintf(r->how, sizeof r->how, "%s; a sum of such terms is "
			    "almost periodic, not periodic (unless they cancel)", sb.p);
			free(sb.p);
			ev_close(&e);
			return;
		}
		if (!per_holds(&e, lo, hi, r->p, 997)) {
			/* cannot happen unless evaluation disagrees */
			r->status = -2;
			snprintf(r->why, sizeof r->why, "the candidate %s failed the "
			    "numeric check", num(r->p));
			free(sb.p);
			ev_close(&e);
			return;
		}
		{
			const char *P0 = per_text(r);
			kk = per_minimise(&e, r, lo, hi);
			r->status = 1;
			r->proved = kk == 1;
			if (kk == 1)
				snprintf(r->how, sizeof r->how, "proved symbolically: "
				    "%s%s; %s occurs only inside them, so f(%s + p) = "
				    "f(%s).  Smallest: no p/k (2 <= k <= 64) is a period "
				    "(checked at 997 points)%s", sb.p, r->nat > 1 ? "; p is "
				    "their least common multiple" : "", vname(f, s),
				    vname(f, s), vname(f, s), f->np > 1 ?
				    "; other parameters at 0.731" : "");
			else
				snprintf(r->how, sizeof r->how, "%s; so %s is a period "
				    "(proved symbolically); its 1/%d also is, checked "
				    "numerically at 997 points (not proved), and no smaller "
				    "fraction is", sb.p, P0, kk);
		}
		free(sb.p);
		ev_close(&e);
		return;
	}
	{
		struct rpoly P;
		if (rpoly_of(f->body, s, &P) && P.deg >= 1) {
			r->status = 0;
			r->proved = 1;
			snprintf(r->why, sizeof r->why, "a polynomial of degree %d is "
			    "not periodic (proved: f(x) - f(0) is a non-zero "
			    "polynomial, so it has finitely many zeros)", P.deg);
			ev_close(&e);
			return;
		}
	}
	/* a numeric search over p in (0, 20] */
	{
		double spread = ymax - ymin, best = 0, pe = INFINITY, ce, ne;
		double p, step = 0.01;
		int left = 0;   /* past the shifts too small to change f */
		ce = per_err(&e, lo, hi, step);
		for (p = step; p <= 20 + 1e-9; p += step) {
			ne = per_err(&e, lo, hi, p + step);
			if (ce > 0.1 * spread)
				left = 1;
			if (left && ce <= pe && ce <= ne && ce < 0.01 * spread) {
				/* refine on [p - step, p + step] */
				double a = p - step, b = p + step;
				int it;
				for (it = 0; it < 100; it++) {
					double m1 = a + (b - a) * 0.382, m2 = a + (b - a) * 0.618;
					if (per_err(&e, lo, hi, m1) < per_err(&e, lo, hi, m2))
						b = m2;
					else
						a = m1;
				}
				if (per_holds(&e, lo, hi, 0.5 * (a + b), 997)) {
					best = 0.5 * (a + b);
					break;
				}
			}
			pe = ce;
			ce = ne;
		}
		if (best > 0) {
			r->p = best;
			r->status = 1;
			r->proved = 0;
			per_minimise(&e, r, lo, hi);
			snprintf(r->how, sizeof r->how, "found numerically (not "
			    "proved): f(x + p) = f(x) at 997 points of [%g, %g]", lo, hi);
			ev_close(&e);
			return;
		}
	}
	{
		double y1 = 0, y2 = 0;
		int o1 = ev1(&e, 1000, &y1), o2 = ev1(&e, -1000, &y2);
		double big = fmax(fabs(ymin), fabs(ymax));
		r->status = 0;
		r->proved = 0;
		snprintf(r->why, sizeof r->why, "%s appears outside the periodic "
		    "subterms (in %s), and no period p <= 20 was found numerically%s",
		    vname(f, s), node_str(r->bad ? r->bad : f->body),
		    (o1 && fabs(y1) > 1000 * big) || (o2 && fabs(y2) > 1000 * big) ?
		    grow_text(o1 && fabs(y1) > 1000 * big ? 1000.0 : -1000.0,
		    o1 && fabs(y1) > 1000 * big ? y1 : y2) : "");
	}
	ev_close(&e);
}

int
fn_period(struct func *f, int slot, double *p)
{
	struct per r;

	if (slot < 0 || slot >= f->np)
		return 0;
	per_analyse(f, slot, &r);
	if (r.status == 1) {
		*p = r.p;
		return 1;
	}
	return r.status == -1 ? -1 : 0;
}

static void
per_print(struct func *f, int s, const struct per *r, int yesno)
{
	rep_begin("period");
	rep_str("var", vname(f, s));
	rep_str("status", r->status == 1 ? "periodic" : r->status == 0 ?
	    "not periodic" : r->status == -1 ? "constant" : "unknown");
	if (r->status == 1) {
		rep_exact("period", r->p, r->exact ? q_text(r->q, r->pi) : NULL);
		if (r->nonneg)
			rep_cond("%s >= 0", vname(f, s));
	}
	if (r->why[0])
		rep_str("why", r->why);
	if (r->how[0])
		rep_proof(proofkind(r->proved), r->how);
	if (r->status == -1) {
		printf("  %sconstant in %s: every p > 0 is a period, there is no "
		    "fundamental period\n", yesno ? "yes, trivially: " : "",
		    vname(f, s));
		if (r->how[0])
			printf("  %s\n", r->how);
		return;
	}
	if (r->status == 1) {
		printf("  %sfundamental period %s%s\n", yesno ? "yes: " : "",
		    per_text(r), r->nonneg ? " for " "x >= 0 only (C's % truncates "
		    "towards zero, so the pattern breaks at 0; mod repeats "
		    "everywhere)" : "");
		printf("  %s\n", r->how);
		if (r->d.isint)
			printf("  %s\n", r->cyclic ? "the pattern also continues "
			    "across the wrap-around of the type (the period divides "
			    "the domain size)" : "the pattern does not continue across "
			    "the end of the domain");
		return;
	}
	if (r->status == 0)
		printf("  %snot periodic\n", yesno ? "no: " : "");
	else
		printf("  %sunknown\n", yesno ? "unknown: " : "");
	if (r->why[0])
		printf("  %s\n", r->why);
	if (r->how[0])
		printf("  %s\n", r->how);
}

/* ---- parity ---- */

enum { PA_NONE, PA_EVEN, PA_ODD, PA_CONST };

/* symbolic parity rules over the expression (real arithmetic) */
static int
parity_of(nid e, int s)
{
	int a, b, op = N.op[e];
	int64_t k;

	if (!has_slot(e, s))
		return PA_CONST;
	switch (op) {
	case O_VAR:
		return PA_ODD;
	case O_NEG:
		return parity_of(N.a[e], s);
	case O_ADD:
	case O_SUB:
		a = parity_of(N.a[e], s);
		b = parity_of(N.b[e], s);
		if (a == PA_CONST)
			a = PA_EVEN;
		if (b == PA_CONST)
			b = PA_EVEN;
		return a == b ? a : PA_NONE;
	case O_MUL:
	case O_DIV:
		a = parity_of(N.a[e], s);
		b = parity_of(N.b[e], s);
		if (a == PA_NONE || b == PA_NONE)
			return PA_NONE;
		if (a == PA_CONST)
			a = PA_EVEN;
		if (b == PA_CONST)
			b = PA_EVEN;
		return a == b ? PA_EVEN : PA_ODD;
	case O_POW:
		a = parity_of(N.a[e], s);
		b = parity_of(N.b[e], s);
		if (b == PA_CONST) {
			if (a == PA_EVEN)
				return PA_EVEN;
			if (a == PA_ODD && int_of(N.b[e], &k))
				return k % 2 ? PA_ODD : PA_EVEN;
			return PA_NONE;
		}
		if ((a == PA_CONST || a == PA_EVEN) && b == PA_EVEN)
			return PA_EVEN;
		return PA_NONE;
	case O_SIN:
	case O_TAN:
	case O_ATAN:
	case O_CSC:
	case O_COT:
	case O_SIGN:
	case O_ASIN: case O_SINH: case O_TANH: case O_ASINH: case O_ATANH:
		a = parity_of(N.a[e], s);
		return a == PA_CONST ? PA_EVEN : a;
	case O_COSH:
	case O_COS:
	case O_SEC:
	case O_ABS:
		a = parity_of(N.a[e], s);
		return a == PA_NONE ? PA_NONE : PA_EVEN;
	case O_CAST:
		if (TY_FP(N.ty[e]) && TY_FP(N.ty[N.a[e]]))
			return parity_of(N.a[e], s);
		/* FALLTHROUGH */
	default:
		/* g(even arguments) is even */
		if ((!N.a[e] || parity_of(N.a[e], s) >= PA_EVEN + 0) &&
		    (!N.a[e] || parity_of(N.a[e], s) != PA_ODD) &&
		    (!N.b[e] || parity_of(N.b[e], s) == PA_EVEN ||
		    parity_of(N.b[e], s) == PA_CONST) &&
		    (!N.c[e] || parity_of(N.c[e], s) == PA_EVEN ||
		    parity_of(N.c[e], s) == PA_CONST) && op != O_TABLE &&
		    op != O_LOOP && op != O_INDEX)
			return PA_EVEN;
		if (op == O_SEL) {
			int c = parity_of(N.a[e], s);
			a = parity_of(N.b[e], s);
			b = parity_of(N.c[e], s);
			if ((c == PA_EVEN || c == PA_CONST) && a == b && a != PA_NONE)
				return a;
		}
		return PA_NONE;
	}
}

struct par {
	int even, odd;      /* 1 holds, 0 fails */
	int na;             /* not applicable: the domain is not symmetric */
	int proved;
	double wx;          /* witness: f(-wx) is neither f(wx) nor -f(wx) */
	double wy, wny;
	int weven, wodd;    /* witnesses exist */
	double ex, ey, eny, ox, oy, ony;
	char wtxt[2][160];     /* machine integers: equiv's witnesses */
	char how[256];
	struct dom d;
};

static void
par_analyse(struct func *f, int s, struct par *r)
{
	struct ev e;
	double *xs, *ys, *ns;
	int64_t *xi = NULL, *ni = NULL;
	uint8_t *ok, *nok;
	uint32_t n, i, cmp = 0;
	int pr;

	memset(r, 0, sizeof *r);
	dom_of(f, s, &r->d);
	if (r->d.isint && TY_SIZED(f->p[s].ty) && f->np == 1 &&
	    (!f->p[s].ranged || TYK(f->p[s].ty) != K_UINT)) {
		/*
		 * Machine integers: -x is the two's complement negation, which
		 * maps the type onto itself; compare f(-x) with f(x) and -f(x)
		 * in the type's arithmetic, exhaustively.
		 */
		nid x = mk_var((uint32_t)s, f->p[s].name, f->p[s].ty), map[MAXP] =
		    {0}, fn;
		struct sbuf we = {0}, wo = {0};
		int sampled = 0, sa2 = 0;
		map[s] = mk_un(O_NEG, x);
		fn = subst(f->body, map, f->np);
		if (TYK(f->p[s].ty) == K_UINT || !f->p[s].ranged ||
		    f->p[s].lo == -f->p[s].hi) {
			r->even = equiv_check(f, fn, f->body, &we, &sampled) == 0;
			r->odd = equiv_check(f, fn, mk_un(O_NEG, f->body), &wo,
			    &sa2) == 0;
			snprintf(r->wtxt[0], sizeof r->wtxt[0], "f(-x) vs f(x) at %s",
			    we.n ? we.p : "");
			snprintf(r->wtxt[1], sizeof r->wtxt[1], "f(-x) vs -f(x) at %s",
			    wo.n ? wo.p : "");
			r->weven = !r->even;
			r->wodd = !r->odd;
			r->proved = sampled || sa2 ? 0 : 2;
			snprintf(r->how, sizeof r->how, "%s, with -x and -f(x) in %s "
			    "arithmetic (two's complement%s)", r->proved ? "proved, "
			    "exhaustive" : "checked on a sample", ty_str(f->p[s].ty),
			    TYK(f->p[s].ty) == K_UINT ? ": -x is 2^w - x" : ": -x of the "
			    "minimum wraps to itself");
			free(we.p);
			free(wo.p);
			return;
		}
	}
	if (r->d.isint && r->d.lo >= 0) {
		r->na = 1;
		snprintf(r->how, sizeof r->how, "the domain %s is not symmetric "
		    "about 0", dom_text(&r->d));
		return;
	}
	if (!r->d.isint && !r->d.unranged && !close_to(r->d.rlo, -r->d.rhi,
	    1e-12)) {
		r->na = 1;
		snprintf(r->how, sizeof r->how, "the domain %s is not symmetric "
		    "about 0", dom_text(&r->d));
		return;
	}
	pr = TYK(f->p[s].ty) == K_REAL && !f->where ? parity_of(f->body, s) :
	    PA_NONE;
	if (pr != PA_NONE) {
		r->proved = 1;
		r->even = pr == PA_EVEN || pr == PA_CONST;
		r->odd = pr == PA_ODD;
		snprintf(r->how, sizeof r->how, "proved symbolically (parity "
		    "rules: x is odd; sums keep a common parity, products multiply "
		    "it; sin/tan/atan preserve it, cos/abs make it even)");
		if (r->even && r->odd)
			return;
	}
	/* numerically, or exhaustively for integers */
	if (r->d.isint) {
		int64_t m = -r->d.lo < r->d.hi ? -r->d.lo : r->d.hi, k;
		n = m > (int64_t)EXH ? NSAMP : (uint32_t)m + 1;
		xi = xcalloc(n, sizeof *xi);
		ni = xcalloc(n, sizeof *ni);
		for (i = 0; i < n; i++) {
			k = n == (uint32_t)m + 1 ? (int64_t)i : (int64_t)((double)i /
			    (n - 1) * m);
			xi[i] = k;
			ni[i] = -k;
		}
		xs = xcalloc(n, sizeof *xs);
	} else {
		n = 2000;
		xs = xcalloc(n, sizeof *xs);
		for (i = 0; i < n; i++)
			xs[i] = r->d.rhi * (i + 1) / n;
	}
	ns = xcalloc(n, sizeof *ns);
	ys = xcalloc(n, sizeof *ys);
	ok = xcalloc(n, 1);
	nok = xcalloc(n, 1);
	ev_open(&e, f, f->body, s, 0);
	ev_run(&e, xs, xi, n, ys, NULL, ok);
	if (!r->d.isint)
		for (i = 0; i < n; i++)
			xs[i] = -xs[i];
	ev_run(&e, xs, ni, n, ns, NULL, nok);
	if (!r->proved) {
		r->even = r->odd = 1;
		for (i = 0; i < n; i++) {
			double x = r->d.isint ? (double)xi[i] : -xs[i];
			int ev, od;
			if (!ok[i] && !nok[i])
				continue;
			cmp++;
			ev = ok[i] && nok[i] && (r->d.isint ? ys[i] == ns[i] :
			    close_to(ys[i], ns[i], 1e-9) || fabs(ys[i] - ns[i]) < 1e-12);
			od = ok[i] && nok[i] && (r->d.isint ? ys[i] == -ns[i] :
			    close_to(ys[i], -ns[i], 1e-9) || fabs(ys[i] + ns[i]) < 1e-12);
			/* the witness nearest x = 1 reads best */
			if (!ev && (r->even || fabs(x - 1) < fabs(r->ex - 1))) {
				r->even = 0;
				r->weven = 1;
				r->ex = x, r->ey = ys[i], r->eny = ns[i];
			}
			if (!od && (r->odd || fabs(x - 1) < fabs(r->ox - 1))) {
				r->odd = 0;
				r->wodd = 1;
				r->ox = x, r->oy = ys[i], r->ony = ns[i];
			}
		}
		snprintf(r->how, sizeof r->how, "%s", r->d.isint ? (n == (uint32_t)
		    (-r->d.lo < r->d.hi ? -r->d.lo : r->d.hi) + 1 ?
		    "proved, exhaustive over every x with -x in the domain" :
		    "checked on a sample (not proved)") : r->d.unranged ?
		    "checked numerically at 2000 points of [-10, 10]" :
		    "checked numerically at 2000 points");
		if (r->d.isint && r->d.lo == -r->d.hi - 1)
			snprintf(r->how + strlen(r->how), sizeof r->how -
			    strlen(r->how), "; x = %lld is left out: its negation "
			    "wraps", (long long)r->d.lo);
		if (!r->odd && !r->even && r->weven && r->wodd)
			r->wx = r->ex;
		if (cmp == 0)
			r->na = 1;
	}
	free(xs);
	free(xi);
	free(ni);
	free(ys);
	free(ns);
	free(ok);
	free(nok);
	ev_close(&e);
}

static const char *
par_word(const struct par *r)
{
	if (r->na)
		return "neither (parity does not apply)";
	if (r->even && r->odd)
		return "both even and odd (identically zero)";
	if (r->even)
		return "even: f(-x) = f(x)";
	if (r->odd)
		return "odd: f(-x) = -f(x)";
	return "neither even nor odd";
}

static void
par_print(struct func *f, int s, const struct par *r)
{
	const char *x = vname(f, s);

	printf("  %s\n", par_word(r));
	if (r->wtxt[0][0]) {
		if (!r->even && !r->odd)
			printf("  not even: %s\n  not odd: %s\n", r->wtxt[0],
			    r->wtxt[1]);
		printf("  %s\n", r->how);
		return;
	}
	if (!r->na && !r->even && r->weven)
		printf("  not even: f(%s) = %s, f(-%s) = %s at %s = %s\n", x,
		    num(r->ey), x, num(r->eny), x, num(r->ex));
	if (!r->na && !r->odd && r->wodd && !r->even)
		printf("  not odd: f(%s) = %s, f(-%s) = %s at %s = %s\n", x,
		    num(r->oy), x, num(r->ony), x, num(r->ox));
	printf("  %s\n", r->how);
}

/* ---- interval arithmetic (bounded) ---- */

struct rng {
	double lo, hi;
};

static struct rng
rg(double a, double b)
{
	struct rng r;

	r.lo = a;
	r.hi = b;
	return r;
}

static double
mulx(double a, double b)
{
	if ((a == 0 && isinf(b)) || (b == 0 && isinf(a)))
		return 0;
	return a * b;
}

static struct rng
iv_of(nid e, int s, struct rng x, struct rng *memo, uint8_t *done)
{
	struct rng a, b, r = rg(-INFINITY, INFINITY);
	double v, t[4];
	int64_t k;
	int i;

	if (e < N.n && done[e])
		return memo[e];
	switch (N.op[e]) {
	case O_CONST:
		num_of(e, &v);
		r = rg(v, v);
		break;
	case O_VAR:
		if (N.ki[e] == s)
			r = x;
		break;
	case O_NEG:
		a = iv_of(N.a[e], s, x, memo, done);
		r = rg(-a.hi, -a.lo);
		break;
	case O_ADD:
	case O_SUB:
		a = iv_of(N.a[e], s, x, memo, done);
		b = iv_of(N.b[e], s, x, memo, done);
		if (N.op[e] == O_SUB)
			b = rg(-b.hi, -b.lo);
		r = rg(a.lo + b.lo, a.hi + b.hi);
		if (isnan(r.lo))
			r.lo = -INFINITY;
		if (isnan(r.hi))
			r.hi = INFINITY;
		break;
	case O_MUL:
		a = iv_of(N.a[e], s, x, memo, done);
		b = iv_of(N.b[e], s, x, memo, done);
		if (N.a[e] == N.b[e] && a.lo < 0 && a.hi > 0) {
			r = rg(0, fmax(mulx(a.lo, a.lo), mulx(a.hi, a.hi)));
			break;
		}
		t[0] = mulx(a.lo, b.lo), t[1] = mulx(a.lo, b.hi);
		t[2] = mulx(a.hi, b.lo), t[3] = mulx(a.hi, b.hi);
		r = rg(t[0], t[0]);
		for (i = 1; i < 4; i++)
			r = rg(fmin(r.lo, t[i]), fmax(r.hi, t[i]));
		break;
	case O_DIV:
		a = iv_of(N.a[e], s, x, memo, done);
		b = iv_of(N.b[e], s, x, memo, done);
		if (b.lo > 0 || b.hi < 0) {
			t[0] = a.lo / b.lo, t[1] = a.lo / b.hi;
			t[2] = a.hi / b.lo, t[3] = a.hi / b.hi;
			r = rg(t[0], t[0]);
			for (i = 0; i < 4; i++) {
				if (isnan(t[i]))
					t[i] = 0;   /* inf/inf: handled by the bounds */
				r = rg(fmin(r.lo, t[i]), fmax(r.hi, t[i]));
			}
			if (isinf(a.lo) || isinf(a.hi))
				r = rg(-INFINITY, INFINITY);
		}
		break;
	case O_POW:
		a = iv_of(N.a[e], s, x, memo, done);
		if (int_of(N.b[e], &k) && k >= 0 && k <= 64) {
			if (k % 2 == 0) {
				double m0 = a.lo <= 0 && a.hi >= 0 ? 0 :
				    fmin(fabs(a.lo), fabs(a.hi));
				r = rg(pow(m0, (double)k), pow(fmax(fabs(a.lo),
				    fabs(a.hi)), (double)k));
			} else
				r = rg(pow(a.lo, (double)k), pow(a.hi, (double)k));
		} else if (int_of(N.b[e], &k) && k < 0 && (a.lo > 0 || a.hi < 0)) {
			if (-k % 2 == 0 || a.lo > 0)
				r = rg(fmin(pow(a.lo, (double)k), pow(a.hi, (double)k)),
				    fmax(pow(a.lo, (double)k), pow(a.hi, (double)k)));
		}
		break;
	case O_SIN:
	case O_COS:
		r = rg(-1, 1);
		break;
	case O_ATAN:
		r = rg(-M_PI / 2, M_PI / 2);
		break;
	case O_ASIN:	/* forms.c */
		r = rg(-M_PI / 2, M_PI / 2);
		break;
	case O_ACOS:
		r = rg(0, M_PI);
		break;
	case O_ATAN2:
		r = rg(-M_PI, M_PI);
		break;
	case O_TANH:
		r = rg(-1, 1);
		break;
	case O_SIGN:
		r = rg(-1, 1);
		break;
	case O_EXP:
		a = iv_of(N.a[e], s, x, memo, done);
		r = rg(exp(a.lo), exp(a.hi));
		break;
	case O_LOG:
		a = iv_of(N.a[e], s, x, memo, done);
		r = rg(a.lo > 0 ? log(a.lo) : -INFINITY, log(fmax(a.hi, 0)));
		break;
	case O_SQRT:
		a = iv_of(N.a[e], s, x, memo, done);
		r = rg(sqrt(fmax(a.lo, 0)), sqrt(fmax(a.hi, 0)));
		break;
	case O_ABS:
		a = iv_of(N.a[e], s, x, memo, done);
		r = rg(a.lo <= 0 && a.hi >= 0 ? 0 : fmin(fabs(a.lo), fabs(a.hi)),
		    fmax(fabs(a.lo), fabs(a.hi)));
		break;
	case O_FLOOR:
	case O_CEIL:
		a = iv_of(N.a[e], s, x, memo, done);
		r = rg(floor(a.lo), ceil(a.hi));
		break;
	case O_MIN:
	case O_MAX:
		a = iv_of(N.a[e], s, x, memo, done);
		b = iv_of(N.b[e], s, x, memo, done);
		r = N.op[e] == O_MIN ? rg(fmin(a.lo, b.lo), fmin(a.hi, b.hi)) :
		    rg(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
		break;
	case O_SEL:
		a = iv_of(N.b[e], s, x, memo, done);
		b = iv_of(N.c[e], s, x, memo, done);
		r = rg(fmin(a.lo, b.lo), fmax(a.hi, b.hi));
		break;
	case O_CAST:
		if (TY_FP(N.ty[e]))
			r = iv_of(N.a[e], s, x, memo, done);
		break;
	default:
		break;
	}
	if (isnan(r.lo))
		r.lo = -INFINITY;
	if (isnan(r.hi))
		r.hi = INFINITY;
	if (e < N.n) {
		memo[e] = r;
		done[e] = 1;
	}
	return r;
}

static struct rng
iv_bound(const struct func *f, int s, struct rng x)
{
	struct rng *memo = xcalloc(N.n + 1, sizeof *memo), r;
	uint8_t *done = xcalloc(N.n + 1, 1);

	r = iv_of(f->body, s, x, memo, done);
	free(memo);
	free(done);
	return r;
}

/* ---- sampling a real function on its interval ---- */

struct samp {
	uint32_t n;
	double *x, *y;
	uint8_t *ok;
	double lo, hi;
	double ymin, ymax, xmin, xmax;
	int any;
};

static void
samp_make(struct ev *e, double lo, double hi, uint32_t n, struct samp *g)
{
	uint32_t i;

	memset(g, 0, sizeof *g);
	g->n = n;
	g->lo = lo;
	g->hi = hi;
	g->x = xcalloc(n, sizeof *g->x);
	g->y = xcalloc(n, sizeof *g->y);
	g->ok = xcalloc(n, 1);
	for (i = 0; i < n; i++)
		g->x[i] = lo + (hi - lo) * i / (n - 1);
	ev_run(e, g->x, NULL, n, g->y, NULL, g->ok);
	g->ymin = INFINITY;
	g->ymax = -INFINITY;
	for (i = 0; i < n; i++) {
		if (!g->ok[i] || isinf(g->y[i]))
			continue;
		g->any = 1;
		if (g->y[i] < g->ymin)
			g->ymin = g->y[i], g->xmin = g->x[i];
		if (g->y[i] > g->ymax)
			g->ymax = g->y[i], g->xmax = g->x[i];
	}
}

static void
samp_free(struct samp *g)
{
	free(g->x);
	free(g->y);
	free(g->ok);
}

/* ---- discontinuities ---- */

enum { DI_REMOVABLE = 1, DI_JUMP, DI_INFINITE, DI_OSC, DI_EDGE };
#define MAXDI 32

struct disc {
	int n, ntotal;
	double x[MAXDI], l[MAXDI], r[MAXDI], v[MAXDI];
	int kind[MAXDI], vdef[MAXDI], side[MAXDI];  /* side: EDGE, -1 left */
	int elementary;      /* built from operations continuous on their domain */
	int ndom;            /* the intervals where f is defined */
	double da[MAXDI], db[MAXDI];
	char how[256];
	struct dom d;
};

/* the limit of f at c from one side (dir -1 left, +1 right) */
static int
one_sided(struct ev *e, double c, int dir, double scale, double *L)
{
	double h[5] = {1e-3, 1e-5, 1e-7, 1e-9, 1e-11}, y[5];
	int k;

	for (k = 0; k < 5; k++) {
		double x = c + dir * h[k] * scale;
		if (x == c || !ev1(e, x, &y[k]))
			return 0;
	}
	{
		int grow = 1;
		for (k = 1; k < 5; k++)
			if (!(fabs(y[k]) > fabs(y[k - 1])) || (y[k] > 0) != (y[0] > 0))
				grow = 0;
		if (grow && fabs(y[4] - y[3]) >= 0.5 * fabs(y[1] - y[0]) &&
		    fabs(y[4] - y[0]) > 1) {
			*L = y[4] > 0 ? INFINITY : -INFINITY;
			return 1;
		}
	}
	if (fabs(y[4] - y[3]) <= 1e-6 * (1 + fabs(y[4])) &&
	    fabs(y[3] - y[2]) <= 1e-4 * (1 + fabs(y[4]))) {
		*L = y[4];
		return 1;
	}
	return 2;   /* no limit found: oscillates */
}

static int
discont_op(nid e)
{
	int op = N.op[e];

	if (op == O_FLOOR || op == O_CEIL || op == O_SIGN || op == O_SEL ||
	    op == O_MOD || op == O_INDEX || op == O_LOOP || op == O_CAST ||
	    (opinfo[op].fl & OF_CMP) || op == O_LAND || op == O_LOR ||
	    op == O_LNOT || (opinfo[op].fl & OF_INT))
		return op != O_CAST || !TY_FP(N.ty[e]) || !TY_FP(N.ty[N.a[e]]);
	return 0;
}

static int
has_discont_op(nid e)
{
	if (e == 0 || !(N.fl[e] & NF_VAR) || N.op[e] == O_VAR ||
	    N.op[e] == O_TABLE)
		return 0;
	if (discont_op(e))
		return 1;
	return has_discont_op(N.a[e]) || has_discont_op(N.b[e]) ||
	    has_discont_op(N.c[e]);
}

static void
disc_add(struct disc *D, double x, int kind, double l, double r, int vdef,
    double v, int side)
{
	int k;

	for (k = 0; k < D->n; k++)
		if (fabs(D->x[k] - x) < 1e-7 * (1 + fabs(x)))
			return;
	D->ntotal++;
	if (D->n == MAXDI)
		return;
	D->x[D->n] = x;
	D->kind[D->n] = kind;
	D->l[D->n] = l;
	D->r[D->n] = r;
	D->vdef[D->n] = vdef;
	D->v[D->n] = v;
	D->side[D->n] = side;
	D->n++;
}

/* a point c that looked bad: classify it from its one-sided limits */
static void
disc_classify(struct ev *e, struct disc *D, double c, double scale)
{
	double L = 0, R = 0, v = 0;
	int kl = one_sided(e, c, -1, scale, &L), kr = one_sided(e, c, 1, scale,
	    &R), vd = ev1(e, c, &v);
	double sn = nice(c)[0] ? c : c;

	if (kl == 0 && kr == 0)
		return;
	if (kl == 0 || kr == 0) {
		/* an end of the domain: only one side exists */
		double M = kl ? L : R;
		int k = kl ? kl : kr;
		if (k == 1 && isinf(M))
			disc_add(D, sn, DI_EDGE, kl ? L : NAN, kr ? R : NAN, vd, v,
			    kl ? -1 : 1);
		return;
	}
	if (kl == 2 || kr == 2) {
		disc_add(D, sn, DI_OSC, L, R, vd, v, 0);
		return;
	}
	if (isinf(L) || isinf(R)) {
		disc_add(D, sn, DI_INFINITE, L, R, vd, v, 0);
		return;
	}
	if (close_to(L, R, 1e-6) || fabs(L - R) < 1e-9) {
		if (vd && (close_to(v, L, 1e-6) || fabs(v - L) < 1e-9))
			return;   /* continuous after all */
		disc_add(D, sn, DI_REMOVABLE, L, R, vd, v, 0);
		return;
	}
	disc_add(D, sn, DI_JUMP, L, R, vd, v, 0);
}

/* the point where definedness changes between a (defined: da) and b */
static double
def_edge(struct ev *e, double a, double b)
{
	double y, m;
	int da = ev1(e, a, &y), k;

	for (k = 0; k < 100; k++) {
		m = 0.5 * (a + b);
		if (m == a || m == b)
			break;
		if (ev1(e, m, &y) == da)
			a = m;
		else
			b = m;
	}
	return 0.5 * (a + b);
}

/* snap a located point to a short exact form when f agrees */
static double
snap_x(double c, double h)
{
	const char *n = nice(c);
	double v;
	int d;

	if (fabs(c) < h * 1e-6)
		return 0;
	if (fabs(c - floor(c + 0.5)) < 1e-12 * (1 + fabs(c)))
		return floor(c + 0.5);
	if (*n) {
		for (d = 1; d <= 12; d++) {
			v = floor(c * d / M_PI + 0.5) * M_PI / d;
			if (fabs(v - c) < 1e-12 * (1 + fabs(c)))
				return v;
			v = floor(c * d + 0.5) / d;
			if (fabs(v - c) < 1e-12 * (1 + fabs(c)))
				return v;
		}
	}
	return c;
}

static void
disc_analyse(struct func *f, int s, struct disc *D)
{
	struct ev e;
	struct samp g;
	double lo, hi, h, scale;
	uint32_t i;
	int in = 0;

	memset(D, 0, sizeof *D);
	dom_of(f, s, &D->d);
	D->elementary = !has_discont_op(f->body) && !f->where;
	lo = D->d.rlo;
	hi = D->d.rhi;
	h = (hi - lo) / (NG - 1);
	scale = fmax(h, 1e-300);
	ev_open(&e, f, f->body, s, 0);
	samp_make(&e, lo, hi, NG, &g);
	/* the defined intervals, their ends refined */
	for (i = 0; i < g.n; i++) {
		if (g.ok[i] && !in) {
			double a = i == 0 ? lo : snap_x(def_edge(&e, g.x[i], g.x[i - 1]),
			    h);
			if (D->ndom < MAXDI)
				D->da[D->ndom] = a;
			in = 1;
		} else if (!g.ok[i] && in) {
			double b = snap_x(def_edge(&e, g.x[i - 1], g.x[i]), h);
			if (D->ndom < MAXDI)
				D->db[D->ndom] = b;
			D->ndom++;
			in = 0;
		}
	}
	if (in) {
		if (D->ndom < MAXDI)
			D->db[D->ndom] = hi;
		D->ndom++;
	}
	if (D->ndom > MAXDI)
		D->ndom = MAXDI;
	/* gaps between defined intervals and the domain's inner ends */
	for (i = 0; i < (uint32_t)D->ndom; i++) {
		if (D->da[i] > lo)
			disc_classify(&e, D, D->da[i], scale);
		if (D->db[i] < hi)
			disc_classify(&e, D, D->db[i], scale);
	}
	/* jumps between grid points */
	for (i = 1; i + 1 < g.n; i++) {
		double d0, d1, d2, a, b, fa, fb, m, fm;
		int k, bad = 0;
		if (!g.ok[i - 1] || !g.ok[i] || !g.ok[i + 1] ||
		    (i + 2 < g.n && !g.ok[i + 2]))
			continue;
		d0 = fabs(g.y[i] - g.y[i - 1]);
		d1 = fabs(g.y[i + 1] - g.y[i]);
		d2 = i + 2 < g.n ? fabs(g.y[i + 2] - g.y[i + 1]) : d0;
		if (!((d1 > 2 * fmax(d0, d2) || d1 > 20 * fmin(d0, d2)) &&
		    d1 > 1e-9 * (1 + fabs(g.y[i]))))
			continue;
		a = g.x[i];
		b = g.x[i + 1];
		fa = g.y[i];
		fb = g.y[i + 1];
		for (k = 0; k < 200; k++) {
			m = 0.5 * (a + b);
			if (m == a || m == b)
				break;
			if (!ev1(&e, m, &fm)) {
				bad = 1;
				a = b = m;
				break;
			}
			if (fabs(fm - fa) >= fabs(fb - fm))
				b = m, fb = fm;
			else
				a = m, fa = fm;
		}
		if (bad || (fabs(fb - fa) > 1e-6 * fabs(g.y[i + 1] - g.y[i]) &&
		    fabs(fb - fa) > 1e-12 * (1 + fabs(fa))))
			disc_classify(&e, D, snap_x(0.5 * (a + b), h), scale);
	}
	snprintf(D->how, sizeof D->how, "%s%s", D->elementary ?
	    "proved: built from operations that are continuous where defined, "
	    "so it is continuous on its domain; points outside it located "
	    "numerically" : "checked numerically: a grid of 20001 points, "
	    "jumps located by bisection, limits from both sides",
	    D->d.unranged ? "; x has no range: searched [-10, 10]" : "");
	samp_free(&g);
	ev_close(&e);
}

int
fn_discontinuities(struct func *f, double lo, double hi, double *x,
    int *kind, int max)
{
	struct disc D;
	struct func t = *f;
	int k, n = 0;

	if (f->np != 1 || !TY_FP(f->p[0].ty))
		return 0;
	t.p[0].ranged = 1;
	t.p[0].rlo = lo;
	t.p[0].rhi = hi;
	disc_analyse(&t, 0, &D);
	for (k = 0; k < D.n && n < max; k++)
		if (D.kind[k] != DI_EDGE) {
			x[n] = D.x[k];
			kind[n++] = D.kind[k];
		}
	return n;
}

static const char *
val_text(double v, int paren)
{
	char *b = rbuf();

	snprintf(b, 128, paren ? " (f = %s there)" : "has the value %s there",
	    num(v));
	return b;
}

static const char *
lim_text(double v)
{
	return isinf(v) ? (v > 0 ? "+inf" : "-inf") : numx(v);
}

static void
disc_print(struct func *f, int s, const struct disc *D)
{
	int k;
	const char *x = vname(f, s);

	if (D->ndom == 0) {
		printf("  undefined everywhere on %s\n", iv_text(D->d.rlo,
		    D->d.rhi, 0));
		return;
	}
	if (D->ndom > 1 || D->da[0] > D->d.rlo || D->db[0] < D->d.rhi) {
		printf("  undefined");
		if (D->da[0] > D->d.rlo)
			printf(" on [%s, %s)", num(D->d.rlo), numx(D->da[0]));
		for (k = 0; k + 1 < D->ndom; k++) {
			if (close_to(D->db[k], D->da[k + 1], 1e-12) ||
			    fabs(D->db[k] - D->da[k + 1]) < 1e-300)
				printf("%s at %s = %s", k || D->da[0] > D->d.rlo ? "," : "",
				    x, numx(D->db[k]));
			else
				printf("%s on (%s, %s)", k || D->da[0] > D->d.rlo ? "," :
				    "", numx(D->db[k]), numx(D->da[k + 1]));
		}
		if (D->db[D->ndom - 1] < D->d.rhi)
			printf("%s on (%s, %s]", D->ndom > 1 || D->da[0] > D->d.rlo ?
			    "," : "", numx(D->db[D->ndom - 1]), num(D->d.rhi));
		printf(" (the ends located numerically)\n");
	}
	for (k = 0; k < D->n; k++) {
		const char *at = numx(D->x[k]);
		switch (D->kind[k]) {
		case DI_REMOVABLE:
			printf("  %s = %s: removable: both limits are %s, f %s\n", x, at,
			    lim_text(D->l[k]), D->vdef[k] ? val_text(D->v[k], 0) :
			    "is undefined there");
			break;
		case DI_JUMP:
			printf("  %s = %s: jump from %s to %s (size %s)%s\n", x, at,
			    lim_text(D->l[k]), lim_text(D->r[k]),
			    num(D->r[k] - D->l[k]), D->vdef[k] ? val_text(D->v[k], 1) : "");
			break;
		case DI_INFINITE:
			printf("  %s = %s: infinite (a pole): %s from the left, %s "
			    "from the right\n", x, at, lim_text(D->l[k]),
			    lim_text(D->r[k]));
			break;
		case DI_OSC:
			printf("  %s = %s: essential (no limit: f oscillates)\n", x, at);
			break;
		case DI_EDGE:
			printf("  %s = %s: end of the domain, f -> %s from the %s\n", x,
			    at, lim_text(D->side[k] < 0 ? D->l[k] : D->r[k]),
			    D->side[k] < 0 ? "left" : "right");
			break;
		}
	}
	if (D->ntotal > D->n)
		printf("  ... %d points in all\n", D->ntotal);
	printf("  %s\n", D->how);
}

/* ---- convexity ---- */

static int
has_kink(nid e)
{
	int op;

	if (e == 0 || !(N.fl[e] & NF_VAR) || N.op[e] == O_VAR ||
	    N.op[e] == O_TABLE)
		return 0;
	op = N.op[e];
	if (op == O_ABS || op == O_MIN || op == O_MAX || discont_op(e))
		return 1;
	return has_kink(N.a[e]) || has_kink(N.b[e]) || has_kink(N.c[e]);
}

struct cvx {
	int verdict;          /* 1 convex, -1 concave, 2 both (affine), 0 neither */
	int strict;
	int proved;
	int npc, ntotal;
	double pa[MAXPC], pb[MAXPC];
	int pdir[MAXPC];
	int ninf;
	double inf[MAXPC];
	int gaps;
	double wa, wb, wm, wya, wyb, wym;   /* witness against the verdict */
	int hasw;
	char how[256];
	struct dom d;
};

static void
cvx_piece(struct cvx *c, double a, double b, int dir)
{
	if (c->npc < MAXPC) {
		c->pa[c->npc] = a;
		c->pb[c->npc] = b;
		c->pdir[c->npc] = dir;
		c->npc++;
	}
	c->ntotal++;
}

/* pieces of constant sign of v over sorted x (gaps break them) */
static void
sign_pieces(const double *x, const double *v, const uint8_t *ok,
    const double *tol, uint32_t n, struct cvx *c)
{
	uint32_t i;
	int have = 0, cur = 0, gap = 0, s;
	double a = 0, px = 0;

	for (i = 0; i < n; i++) {
		if (!ok[i]) {
			gap |= have;
			continue;
		}
		s = fabs(v[i]) <= tol[i] ? 0 : v[i] > 0 ? 1 : -1;
		if (!have) {
			have = 1;
			a = px = x[i];
			cur = s;
			continue;
		}
		if (gap) {
			cvx_piece(c, a, px, cur);
			c->gaps = 1;
			a = x[i];
			cur = s;
			gap = 0;
		} else if (s != 0 && cur != 0 && s != cur) {
			cvx_piece(c, a, px, cur);
			a = px;
			cur = s;
		} else if (cur == 0)
			cur = s;
		px = x[i];
	}
	if (have)
		cvx_piece(c, a, px, cur);
}

static void
cvx_analyse(struct func *f, int s, struct cvx *c)
{
	struct rpoly P;
	struct ev e;
	int k, any_pos = 0, any_neg = 0;
	double lo, hi;

	memset(c, 0, sizeof *c);
	dom_of(f, s, &c->d);
	lo = c->d.rlo;
	hi = c->d.rhi;
	if (!f->where && TYK(f->p[s].ty) == K_REAL && rpoly_of(f->body, s, &P)) {
		struct rpoly D1, D2;
		struct root r[16];
		double cut[16];
		int nr, nc = 0, j;
		rpoly_deriv(&P, &D1);
		rpoly_deriv(&D1, &D2);
		if (P.deg <= 1) {
			c->verdict = 2;
			c->proved = 1;
			cvx_piece(c, c->d.unranged ? -INFINITY : lo, c->d.unranged ?
			    INFINITY : hi, 0);
			snprintf(c->how, sizeof c->how, "proved: f'' = 0 (degree %d)",
			    P.deg);
			return;
		}
		nr = D2.deg >= 1 ? rpoly_roots(&D2, r) : 0;
		for (k = 0; k < nr; k++)
			if (r[k].im == 0 && r[k].mult % 2 == 1 && (c->d.unranged ||
			    (r[k].re > lo && r[k].re < hi)))
				cut[nc++] = r[k].re;
		for (k = 1; k < nc; k++)
			for (j = k; j > 0 && cut[j] < cut[j - 1]; j--) {
				double t = cut[j];
				cut[j] = cut[j - 1];
				cut[j - 1] = t;
			}
		for (k = 0; k <= nc; k++) {
			double a = k == 0 ? (c->d.unranged ? -INFINITY : lo) : cut[k - 1];
			double b = k == nc ? (c->d.unranged ? INFINITY : hi) : cut[k];
			cvx_piece(c, a, b, rp_sign_in(&D2, a, b) > 0 ? 1 : -1);
		}
		for (k = 0; k < nc; k++)
			c->inf[c->ninf++] = cut[k];
		c->proved = 1;
		c->strict = 1;
		snprintf(c->how, sizeof c->how, "proved symbolically: f'' = %s, "
		    "a polynomial; its sign changes only at real roots of odd "
		    "multiplicity", node_str(rpoly_node(&D2, mk_var(0,
		    f->p[s].name, TY_REAL))));
	} else {
		/* f'' misses kinks (abs, min, max): those use differences */
		nid d1 = has_kink(f->body) ? 0 : try_diff(f->body, s);
		nid d2 = d1 ? try_diff(d1, s) : 0;
		struct samp g;
		double *v, *tol;
		uint32_t i;
		ev_open(&e, f, f->body, s, 0);
		samp_make(&e, lo, hi, NG, &g);
		v = xcalloc(g.n, sizeof *v);
		tol = xcalloc(g.n, sizeof *tol);
		if (d2) {
			struct ev de;
			uint8_t *ok2 = xcalloc(g.n, 1);
			double mx = 0;
			ev_open(&de, f, d2, s, 0);
			ev_run(&de, g.x, NULL, g.n, v, NULL, ok2);
			for (i = 0; i < g.n; i++) {
				g.ok[i] &= ok2[i];
				if (g.ok[i] && fabs(v[i]) > mx && isfinite(v[i]))
					mx = fabs(v[i]);
			}
			for (i = 0; i < g.n; i++)
				tol[i] = 1e-12 * mx;
			free(ok2);
			ev_close(&de);
		} else {
			for (i = 1; i + 1 < g.n; i++) {
				v[i] = g.y[i - 1] - 2 * g.y[i] + g.y[i + 1];
				tol[i] = 1e-9 * (fabs(g.y[i - 1]) + 2 * fabs(g.y[i]) +
				    fabs(g.y[i + 1]));
				if (!g.ok[i - 1] || !g.ok[i + 1])
					g.ok[i] = 0;
			}
			g.ok[0] = g.ok[g.n - 1] = 0;
		}
		sign_pieces(g.x, v, g.ok, tol, g.n, c);
		/* inflection points: refine on f'' */
		for (k = 1; k < c->npc; k++) {
			double t = c->pa[k];
			if (c->pa[k] != c->pb[k - 1] || c->pdir[k] == c->pdir[k - 1])
				continue;
			if (d2) {
				struct ev de;
				double hh = (hi - lo) / (NG - 1), r;
				ev_open(&de, f, d2, s, 0);
				r = refine_turn(&de, t - hh, t + hh);
				if (!isnan(r))
					t = r;
				ev_close(&de);
			}
			c->pa[k] = c->pb[k - 1] = t;
			if (c->ninf < MAXPC)
				c->inf[c->ninf++] = t;
		}
		snprintf(c->how, sizeof c->how, "%s, sign of %s", scan_text(&c->d,
		    vname(f, s)), d2 ? "f''" : "the second difference");
		free(v);
		free(tol);
		samp_free(&g);
		ev_close(&e);
	}
	for (k = 0; k < c->npc; k++) {
		if (c->pdir[k] > 0)
			any_pos = 1;
		if (c->pdir[k] < 0)
			any_neg = 1;
	}
	c->verdict = c->gaps ? 0 : !any_pos && !any_neg ? 2 : !any_neg ? 1 :
	    !any_pos ? -1 : 0;
	/* a witness: the chord lies on the wrong side of f */
	ev_open(&e, f, f->body, s, 0);
	for (k = 0; k < c->npc && !c->hasw; k++) {
		double a = isinf(c->pa[k]) ? c->pb[k] - 2 : c->pa[k];
		double b = isinf(c->pb[k]) ? c->pa[k] + 2 : c->pb[k];
		double m = 0.5 * (a + b), ya, yb, ym;
		if (c->pdir[k] == 0 || c->verdict == 2 || c->gaps ||
		    (c->verdict == 1 && c->pdir[k] > 0) ||
		    (c->verdict == -1 && c->pdir[k] < 0))
			continue;
		if (ev1(&e, a, &ya) && ev1(&e, b, &yb) && ev1(&e, m, &ym) &&
		    (c->pdir[k] < 0 ? ym > 0.5 * (ya + yb) : ym < 0.5 * (ya + yb))) {
			c->hasw = 1;
			c->wa = a, c->wb = b, c->wm = m;
			c->wya = ya, c->wyb = yb, c->wym = ym;
		}
	}
	ev_close(&e);
}

static void
cvx_print(struct func *f, int s, const struct cvx *c)
{
	int k;
	const char *x = vname(f, s);

	printf("  %s on %s\n", c->verdict == 2 ? "both convex and concave "
	    "(affine)" : c->verdict == 1 ? "convex" : c->verdict == -1 ?
	    "concave" : c->gaps ? "neither (its domain is not an interval)" :
	    "neither convex nor concave", c->proved ? dom_text(&c->d) :
	    iv_text(c->d.rlo, c->d.rhi, 0));
	if (c->npc > 1 || c->gaps) {
		for (k = 0; k < c->npc; k++)
			printf("    %s in %s: %s\n", x, iv_text(c->pa[k], c->pb[k], 0),
			    c->pdir[k] > 0 ? "convex (f'' > 0)" : c->pdir[k] < 0 ?
			    "concave (f'' < 0)" : "affine");
		if (c->ntotal > c->npc)
			printf("    ... %d intervals in all\n", c->ntotal);
	}
	if (c->ninf) {
		printf("  inflection points:");
		for (k = 0; k < c->ninf; k++)
			printf("%s %s = %s", k ? "," : "", x, numx(c->inf[k]));
		printf("\n");
	}
	if (c->hasw)
		printf("  witness: f(%s) = %s %s (f(%s) + f(%s))/2 = %s\n",
		    num(c->wm), num(c->wym), c->wym > 0.5 * (c->wya + c->wyb) ?
		    ">" : "<", num(c->wa), num(c->wb), num(0.5 * (c->wya + c->wyb)));
	printf("  %s\n", c->how);
}

/* ---- zeros ---- */

#define MAXZ 64
struct zer {
	int n, ntotal;
	double x[MAXZ];
	int mult[MAXZ];
	char txt[MAXZ][64];
	int ncomplex;
	char ctxt[256];
	int proved;
	int isint;
	char how[256];
	struct dom d;
};

static void
zer_add(struct zer *z, double x, int mult, const char *txt)
{
	int k;

	for (k = 0; k < z->n; k++)
		if (fabs(z->x[k] - x) <= 1e-9 * (1 + fabs(x)))
			return;
	z->ntotal++;
	if (z->n == MAXZ)
		return;
	z->x[z->n] = x;
	z->mult[z->n] = mult;
	snprintf(z->txt[z->n], sizeof z->txt[0], "%s", txt ? txt : numx(x));
	z->n++;
}

static void
zer_sort(struct zer *z)
{
	int i, j;

	for (i = 1; i < z->n; i++)
		for (j = i; j > 0 && z->x[j] < z->x[j - 1]; j--) {
			double t = z->x[j];
			int m = z->mult[j];
			char b[64];
			z->x[j] = z->x[j - 1];
			z->x[j - 1] = t;
			z->mult[j] = z->mult[j - 1];
			z->mult[j - 1] = m;
			memcpy(b, z->txt[j], 64);
			memcpy(z->txt[j], z->txt[j - 1], 64);
			memcpy(z->txt[j - 1], b, 64);
		}
}

static const char *
on_text(const struct dom *d)
{
	char *b = rbuf();

	if (d->isint)
		snprintf(b, 128, " in %lld..%lld", (long long)d->lo,
		    (long long)d->hi);
	else
		snprintf(b, 128, " on [%s, %s]", num(d->rlo), num(d->rhi));
	return b;
}

static void
zer_analyse(struct func *f, int s, struct zer *z)
{
	struct rpoly P;
	struct ev e;
	double lo, hi;

	memset(z, 0, sizeof *z);
	dom_of(f, s, &z->d);
	lo = z->d.rlo;
	hi = z->d.rhi;
	if (z->d.isint) {
		double *xs;
		int64_t *xi, *yi;
		uint8_t *ok;
		uint32_t n = dom_points(&z->d, 0, &xs, &xi), i;
		z->isint = 1;
		yi = xcalloc(n, sizeof *yi);
		ok = xcalloc(n, 1);
		ev_open(&e, f, f->body, s, 0);
		ev_run(&e, xs, xi, n, NULL, yi, ok);
		for (i = 0; i < n; i++)
			if (ok[i] && yi[i] == 0) {
				z->ntotal++;
				if (z->n < MAXZ) {
					z->x[z->n] = (double)xi[i];
					snprintf(z->txt[z->n], 64, "%lld", (long long)xi[i]);
					z->mult[z->n++] = 1;
				}
			}
		z->proved = !z->d.sampled;
		snprintf(z->how, sizeof z->how, "%s", scan_text(&z->d, vname(f, s)));
		free(xs);
		free(xi);
		free(yi);
		free(ok);
		ev_close(&e);
		return;
	}
	if (!f->where && TYK(f->p[s].ty) == K_REAL && rpoly_of(f->body, s, &P)) {
		struct root r[16];
		int nr, k;
		struct sbuf cs = {0};
		if (P.deg == 0) {
			z->proved = 1;
			snprintf(z->how, sizeof z->how, P.c[0] == 0 ? "identically zero"
			    : "a non-zero constant: no zeros");
			if (P.c[0] == 0)
				z->ntotal = -1;
			return;
		}
		nr = rpoly_roots(&P, r);
		for (k = 0; k < nr; k++) {
			if (r[k].im != 0) {
				z->ncomplex += 2 * r[k].mult;
				sb_printf(&cs, "%s%s", cs.n ? ", " : "", root_str(&r[k]));
				continue;
			}
			if (!z->d.unranged && (r[k].re < lo || r[k].re > hi))
				continue;
			zer_add(z, r[k].re, r[k].mult, root_str(&r[k]));
		}
		snprintf(z->ctxt, sizeof z->ctxt, "%s", cs.n ? cs.p : "");
		zer_sort(z);
		free(cs.p);
		z->proved = 1;
		snprintf(z->how, sizeof z->how, "proved: a polynomial of degree %d "
		    "has %d roots counted with multiplicity%s", P.deg, P.deg,
		    nr && !r[0].exact && !r[0].quad ? " (found numerically, then "
		    "checked)" : "");
		return;
	}
	ev_open(&e, f, f->body, s, 0);
	{
		struct samp g;
		uint32_t i;
		double cs[64], scale;
		int nc, k;
		samp_make(&e, lo, hi, NG, &g);
		scale = g.any ? fmax(fabs(g.ymin), fabs(g.ymax)) : 1;
		for (i = 0; i < g.n; i++) {
			if (!g.ok[i])
				continue;
			if (g.y[i] == 0) {
				int touch = i > 0 && i + 1 < g.n && g.ok[i - 1] &&
				    g.ok[i + 1] && (g.y[i - 1] > 0) == (g.y[i + 1] > 0);
				zer_add(z, g.x[i], touch ? -2 : 1, NULL);
				continue;
			}
			if (i > 0 && g.ok[i - 1] && g.y[i - 1] != 0 &&
			    (g.y[i - 1] > 0) != (g.y[i] > 0)) {
				double a = g.x[i - 1], b = g.x[i], fa = g.y[i - 1], m, fm;
				int it, okm = 1;
				for (it = 0; it < 200; it++) {
					m = 0.5 * (a + b);
					if (m == a || m == b)
						break;
					if (!ev1(&e, m, &fm)) {
						okm = 0;
						break;
					}
					if (fm == 0) {
						a = b = m;
						break;
					}
					if ((fm > 0) == (fa > 0))
						a = m, fa = fm;
					else
						b = m;
				}
				m = 0.5 * (a + b);
				if (okm && ev1(&e, m, &fm) && fabs(fm) <= 1e-6 *
				    (fabs(g.y[i - 1]) + fabs(g.y[i])))
					zer_add(z, snap_x(m, 1e-9), 1, NULL);
			}
		}
		/* zeros that touch without a sign change: critical points */
		if (f->np == 1) {
			nc = crit_roots(f, lo, hi, cs, 64);
			for (k = 0; k < nc; k++) {
				double v;
				if (ev1(&e, cs[k], &v) && fabs(v) <= 1e-10 * (1 + scale))
					zer_add(z, snap_x(cs[k], 1e-9), -2, NULL);
				(void)i;
			}
		}
		samp_free(&g);
	}
	zer_sort(z);
	snprintf(z->how, sizeof z->how, "%s: sign changes refined by bisection, "
	    "touching zeros at critical points; zeros between grid points that "
	    "do not change sign can be missed", scan_text(&z->d, vname(f, s)));
	ev_close(&e);
}

int
fn_zeros(struct func *f, double lo, double hi, double *x, int max)
{
	struct zer z;
	struct func t = *f;
	int k;

	if (f->np != 1)
		return 0;
	t.p[0].ranged = 1;
	t.p[0].rlo = lo;
	t.p[0].rhi = hi;
	t.p[0].lo = (int64_t)ceil(lo);
	t.p[0].hi = (int64_t)floor(hi);
	zer_analyse(&t, 0, &z);
	for (k = 0; k < z.n && k < max; k++)
		x[k] = z.x[k];
	return k;
}

static void
zer_print(struct func *f, int s, const struct zer *z)
{
	int k;
	const char *x = vname(f, s);

	rep_begin("zeros");
	rep_str("var", x);
	rep_bool("identically_zero", z->ntotal < 0);
	rep_int("count", z->ntotal < 0 ? 0 : z->ntotal);
	rep_list("zeros");
	for (k = 0; k < z->n; k++) {
		rep_obj(NULL);
		rep_exact("value", z->x[k], z->proved ? z->txt[k] : NULL);
		rep_int("multiplicity", z->mult[k] < 0 ? 2 : z->mult[k]);
		if (z->mult[k] < 0)
			rep_bool("touches", 1);
		rep_end();
	}
	rep_end();
	rep_int("complex", z->ncomplex);
	if (z->ncomplex)
		rep_str("complex_zeros", z->ctxt);
	rep_proof(z->proved ? (z->isint ? "exhaustive" : "symbolic") :
	    "numeric", z->how);
	if (z->ntotal < 0)
		printf("  identically zero: every %s is a zero\n", x);
	else if (z->n == 0)
		printf("  no real zeros%s\n", z->proved && !z->isint &&
		    z->d.unranged ? "" : on_text(&z->d));
	else if (z->isint) {
		int prog = z->n > 3;
		for (k = 2; k < z->n && prog; k++)
			if (z->x[k] - z->x[k - 1] != z->x[1] - z->x[0])
				prog = 0;
		printf("  %d zero%s%s: %s =", z->ntotal, z->ntotal == 1 ? "" : "s",
		    on_text(&z->d), x);
		if (prog && z->n == z->ntotal)
			printf(" %s, %s, ..., %s (every %s)\n", z->txt[0], z->txt[1],
			    z->txt[z->n - 1], num(z->x[1] - z->x[0]));
		else {
			for (k = 0; k < z->n && k < 16; k++)
				printf("%s %s", k ? "," : "", z->txt[k]);
			printf("%s\n", z->ntotal > 16 ? ", ..." : "");
		}
	} else {
		printf("  %d zero%s%s:\n", z->ntotal, z->ntotal == 1 ? "" : "s",
		    z->proved && !z->isint && z->d.unranged ? "" : on_text(&z->d));
		for (k = 0; k < z->n; k++)
			printf("    %s = %s%s\n", x, z->txt[k], z->mult[k] > 1 ?
			    mult_text(z->mult[k]) : z->mult[k] < 0 ? " (touches zero "
			    "without crossing)" : "");
		if (z->ntotal > z->n)
			printf("    ... %d in all\n", z->ntotal);
	}
	if (z->ncomplex)
		printf("  and %d complex: %s (with conjugates)\n", z->ncomplex,
		    z->ctxt);
	printf("  %s\n", z->how);
}

/* ---- the range of a one-parameter function (min/max) ---- */

struct rngres {
	double lo, hi, xlo, xhi;
	int any;
	int exact;            /* integer: exhaustive */
	uint64_t distinct;
};

static int
i64cmp(const void *a, const void *b)
{
	int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;

	return x < y ? -1 : x > y;
}

static void
range_of(struct func *f, int s, const struct dom *d, struct rngres *R)
{
	struct ev e;

	memset(R, 0, sizeof *R);
	ev_open(&e, f, f->body, s, 0);
	R->lo = INFINITY;
	R->hi = -INFINITY;
	if (d->isint) {
		double *xs, *ys;
		int64_t *xi, *yi;
		uint8_t *ok;
		uint32_t n = dom_points(d, 0, &xs, &xi), i, m = 0;
		ys = xcalloc(n, sizeof *ys);
		yi = xcalloc(n, sizeof *yi);
		ok = xcalloc(n, 1);
		ev_run(&e, xs, xi, n, ys, yi, ok);
		for (i = 0; i < n; i++) {
			if (!ok[i])
				continue;
			R->any = 1;
			if (ys[i] < R->lo)
				R->lo = ys[i], R->xlo = xs[i];
			if (ys[i] > R->hi)
				R->hi = ys[i], R->xhi = xs[i];
			yi[m++] = yi[i];
		}
		qsort(yi, m, sizeof *yi, i64cmp);
		for (i = 0; i < m; i++)
			if (i == 0 || yi[i] != yi[i - 1])
				R->distinct++;
		R->exact = !d->sampled;
		free(xs);
		free(ys);
		free(xi);
		free(yi);
		free(ok);
	} else {
		struct samp g;
		double cs[64], v;
		int nc, k;
		samp_make(&e, d->rlo, d->rhi, NG, &g);
		R->any = g.any;
		R->lo = g.ymin, R->hi = g.ymax, R->xlo = g.xmin, R->xhi = g.xmax;
		nc = f->np == 1 ? crit_roots(f, d->rlo, d->rhi, cs, 64) : 0;
		for (k = 0; k < nc; k++)
			if (ev1(&e, cs[k], &v) && isfinite(v)) {
				if (v < R->lo)
					R->lo = v, R->xlo = cs[k];
				if (v > R->hi)
					R->hi = v, R->xhi = cs[k];
			}
		samp_free(&g);
	}
	ev_close(&e);
}

/* ---- bounded ---- */

enum { BD_UNB = 0, BD_PROVED, BD_NUM, BD_UNBNUM };

struct bnd {
	int up, dn;           /* BD_* for above and below */
	double sup, inf;      /* the values seen */
	double psup, pinf;    /* bounds, when proved */
	double wxu, wyu, wxd, wyd;   /* witnesses of unboundedness */
	char hu[200], hd[200];
	struct dom d;
};

static void
bnd_analyse(struct func *f, int s, struct bnd *b)
{
	struct rngres R;
	struct rng iv;
	struct rpoly P;
	struct ev e;

	memset(b, 0, sizeof *b);
	dom_of(f, s, &b->d);
	range_of(f, s, &b->d, &R);
	b->sup = b->psup = R.hi;
	b->inf = b->pinf = R.lo;
	if (b->d.isint) {
		b->up = b->dn = R.exact ? BD_PROVED : BD_NUM;
		snprintf(b->hu, sizeof b->hu, "the domain is finite: max %s at %s "
		    "= %s (%s)", num(R.hi), vname(f, s), num(R.xhi),
		    R.exact ? "exhaustive" : "sampled");
		snprintf(b->hd, sizeof b->hd, "min %s at %s = %s", num(R.lo),
		    vname(f, s), num(R.xlo));
		return;
	}
	iv = iv_bound(f, s, b->d.unranged ? rg(-INFINITY, INFINITY) :
	    rg(b->d.rlo, b->d.rhi));
	b->up = b->dn = BD_NUM;
	if (isfinite(iv.hi) && !f->where) {
		b->up = BD_PROVED;
		b->psup = iv.hi;
		snprintf(b->hu, sizeof b->hu, "proved by interval arithmetic: f "
		    "<= %s; the largest value seen is %s at %s = %s", num(iv.hi),
		    num(R.hi), vname(f, s), numx(R.xhi));
	}
	if (isfinite(iv.lo) && !f->where) {
		b->dn = BD_PROVED;
		b->pinf = iv.lo;
		snprintf(b->hd, sizeof b->hd, "proved by interval arithmetic: f "
		    ">= %s; the smallest value seen is %s at %s = %s", num(iv.lo),
		    num(R.lo), vname(f, s), numx(R.xlo));
	}
	if (!f->where && TYK(f->p[s].ty) == K_REAL && rpoly_of(f->body, s, &P) &&
	    P.deg >= 1 && b->d.unranged) {
		double lc = P.c[P.deg], big = 1000;
		int even = P.deg % 2 == 0;
		/* x -> +inf: sign lc; x -> -inf: sign lc * (-1)^deg */
		if (lc > 0 || !even) {
			b->up = BD_UNB;
			b->wxu = lc > 0 ? big : -big;
			b->wyu = rpoly_at(&P, b->wxu);
			snprintf(b->hu, sizeof b->hu, "proved: a polynomial of degree %d "
			    "with leading coefficient %s: f -> +inf as %s -> %s", P.deg,
			    num(lc), vname(f, s), lc > 0 ? "+inf" : "-inf");
		}
		if (lc < 0 || !even) {
			b->dn = BD_UNB;
			b->wxd = lc < 0 ? big : -big;
			b->wyd = rpoly_at(&P, b->wxd);
			snprintf(b->hd, sizeof b->hd, "proved: a polynomial of degree %d "
			    "with leading coefficient %s: f -> -inf as %s -> %s", P.deg,
			    num(lc), vname(f, s), lc > 0 ? "-inf" : "+inf");
		}
		if (even && lc > 0) {
			b->dn = BD_PROVED;
			snprintf(b->hd, sizeof b->hd, "proved: an even-degree polynomial "
			    "with a positive leading coefficient has a minimum, %s at %s "
			    "= %s (a root of f')", num(R.lo), vname(f, s), numx(R.xlo));
		}
		if (even && lc < 0) {
			b->up = BD_PROVED;
			snprintf(b->hu, sizeof b->hu, "proved: an even-degree polynomial "
			    "with a negative leading coefficient has a maximum, %s at %s "
			    "= %s (a root of f')", num(R.hi), vname(f, s), numx(R.xhi));
		}
		return;
	}
	if (b->up == BD_PROVED && b->dn == BD_PROVED)
		return;
	/* poles and ends of the domain where f is infinite */
	{
		struct disc D;
		int k;
		disc_analyse(f, s, &D);
		for (k = 0; k < D.n; k++) {
			double L = D.l[k], Rr = D.r[k];
			if (D.kind[k] != DI_INFINITE && D.kind[k] != DI_EDGE)
				continue;
			if (D.kind[k] == DI_EDGE)
				L = Rr = D.side[k] < 0 ? D.l[k] : D.r[k];
			if ((L == INFINITY || Rr == INFINITY) && b->up != BD_UNB &&
			    b->up != BD_UNBNUM) {
				b->up = BD_UNBNUM;
				b->wxu = D.x[k];
				snprintf(b->hu, sizeof b->hu, "f -> +inf at %s = %s "
				    "(checked numerically)", vname(f, s), numx(D.x[k]));
			}
			if ((L == -INFINITY || Rr == -INFINITY) && b->dn != BD_UNB &&
			    b->dn != BD_UNBNUM) {
				b->dn = BD_UNBNUM;
				b->wxd = D.x[k];
				snprintf(b->hd, sizeof b->hd, "f -> -inf at %s = %s "
				    "(checked numerically)", vname(f, s), numx(D.x[k]));
			}
		}
	}
	/* growth at large |x| */
	ev_open(&e, f, f->body, s, 0);
	if (b->d.unranged) {
		static const double px[] = {1e2, 1e4, 1e6, 1e8};
		double ref = fmax(1, fmax(fabs(R.lo), fabs(R.hi)));
		int k, sg;
		for (sg = -1; sg <= 1; sg += 2)
			for (k = 3; k >= 0; k--) {
				double x = sg * px[k], y;
				if (!ev1(&e, x, &y) || fabs(y) < 1e3 * ref)
					continue;
				if (y > 0 && b->up == BD_NUM) {
					b->up = BD_UNBNUM;
					b->wxu = x, b->wyu = y;
					snprintf(b->hu, sizeof b->hu, "f(%s) = %s, beyond any "
					    "value on [-10, 10] (checked numerically)", num(x),
					    num(y));
				}
				if (y < 0 && b->dn == BD_NUM) {
					b->dn = BD_UNBNUM;
					b->wxd = x, b->wyd = y;
					snprintf(b->hd, sizeof b->hd, "f(%s) = %s, beyond any "
					    "value on [-10, 10] (checked numerically)", num(x),
					    num(y));
				}
				break;
			}
	}
	if (b->up == BD_NUM)
		snprintf(b->hu, sizeof b->hu, "appears bounded: max %s at %s = %s "
		    "on %s%s (checked numerically, not proved)", num(R.hi),
		    vname(f, s), numx(R.xhi), iv_text(b->d.rlo, b->d.rhi, 0),
		    b->d.unranged ? " and at |x| up to 1e8" : "");
	if (b->dn == BD_NUM)
		snprintf(b->hd, sizeof b->hd, "appears bounded: min %s at %s = %s "
		    "on %s%s (checked numerically, not proved)", num(R.lo),
		    vname(f, s), numx(R.xlo), iv_text(b->d.rlo, b->d.rhi, 0),
		    b->d.unranged ? " and at |x| up to 1e8" : "");
	ev_close(&e);
}

static const char *
bnd_word(const struct bnd *b)
{
	int u = b->up == BD_PROVED || b->up == BD_NUM;
	int d = b->dn == BD_PROVED || b->dn == BD_NUM;

	return u && d ? "bounded" : u ? "bounded above, not below" : d ?
	    "bounded below, not above" : "unbounded above and below";
}

static void
bnd_print(struct func *f, int s, const struct bnd *b)
{
	printf("  %s%s\n", bnd_word(b), b->d.isint ? "" : b->d.unranged ?
	    " (on all real x)" : on_text(&b->d));
	printf("  above: %s\n  below: %s\n", b->hu, b->hd);
}

/* ---- injective, surjective ---- */

struct inj {
	int inj;              /* 1 yes, 0 no */
	int proved;
	double x1, x2, y;     /* witness f(x1) = f(x2) = y */
	char how[300];
};

/* x in [a, b] with f(x) = y, f monotonic there */
static double
solve_in(struct ev *e, double a, double b, double y)
{
	double fa, fm, m;
	int k;

	if (!ev1(e, a, &fa))
		return NAN;
	for (k = 0; k < 200; k++) {
		m = 0.5 * (a + b);
		if (m == a || m == b)
			break;
		if (!ev1(e, m, &fm))
			return NAN;
		if ((fm - y > 0) == (fa - y > 0))
			a = m, fa = fm;
		else
			b = m;
	}
	return 0.5 * (a + b);
}

static void
inj_analyse(struct func *f, int s, struct inj *r)
{
	struct dom d;
	struct mono m;
	struct ev e;
	int k, j;

	memset(r, 0, sizeof *r);
	dom_of(f, s, &d);
	if (d.isint) {
		double *xs;
		int64_t *xi, *yi;
		uint8_t *ok;
		uint32_t n = dom_points(&d, 0, &xs, &xi), i, mm = 0;
		struct pair { int64_t y, x; } *pp;
		ev_open(&e, f, f->body, s, 0);
		yi = xcalloc(n, sizeof *yi);
		ok = xcalloc(n, 1);
		ev_run(&e, xs, xi, n, NULL, yi, ok);
		pp = xcalloc(n ? n : 1, sizeof *pp);
		for (i = 0; i < n; i++)
			if (ok[i]) {
				pp[mm].y = yi[i];
				pp[mm++].x = xi[i];
			}
		qsort(pp, mm, sizeof *pp, i64cmp);   /* by y, then x */
		r->inj = 1;
		for (i = 1; i < mm; i++)
			if (pp[i].y == pp[i - 1].y) {
				r->inj = 0;
				r->x1 = (double)pp[i - 1].x;
				r->x2 = (double)pp[i].x;
				r->y = ival_d(N.ty[f->body], pp[i].y);
				if (r->x1 > r->x2) {
					double t = r->x1;
					r->x1 = r->x2;
					r->x2 = t;
				}
				break;
			}
		r->proved = !d.sampled || !r->inj;
		snprintf(r->how, sizeof r->how, "%s", r->inj ?
		    scan_text(&d, vname(f, s)) : "a witness pair (exhaustive search "
		    "over the domain)");
		free(xs);
		free(xi);
		free(yi);
		free(ok);
		free(pp);
		ev_close(&e);
		return;
	}
	mono_analyse(f, s, &m);
	if (m.verdict == MO_SINC || m.verdict == MO_SDEC) {
		r->inj = 1;
		r->proved = m.proved;
		snprintf(r->how, sizeof r->how, "%s is %s, so no two inputs share a "
		    "value (%s)", fname(f), mono_word(m.verdict), m.proved ?
		    "proved" : "checked numerically");
		return;
	}
	ev_open(&e, f, f->body, s, 0);
	if (m.verdict == MO_CONST) {
		r->inj = 0;
		r->x1 = d.rlo;
		r->x2 = d.rhi;
		ev1(&e, r->x1, &r->y);
		snprintf(r->how, sizeof r->how, "constant");
		ev_close(&e);
		return;
	}
	/* pieces whose value ranges overlap share a value */
	{
		double lo[MAXPC], hi[MAXPC];
		for (k = 0; k < m.npc; k++) {
			double a = isinf(m.pa[k]) ? fmin(-10, m.pb[k] - 10) : m.pa[k];
			double b = isinf(m.pb[k]) ? fmax(10, m.pa[k] + 10) : m.pb[k];
			double ya, yb;
			int i;
			lo[k] = INFINITY;
			hi[k] = -INFINITY;
			for (i = 0; i <= 64; i++) {
				double x = a + (b - a) * i / 64, y;
				if (ev1(&e, x, &y) && isfinite(y)) {
					lo[k] = fmin(lo[k], y);
					hi[k] = fmax(hi[k], y);
				}
			}
			(void)ya;
			(void)yb;
			if (m.pdir[k] == 0 || !m.pstrict[k]) {
				/* a flat stretch: two points on it */
				double y1, y2, x1 = a, x2 = a + (b - a) / 64;
				if (ev1(&e, x1, &y1) && ev1(&e, x2, &y2) && y1 == y2) {
					r->inj = 0;
					r->x1 = x1, r->x2 = x2, r->y = y1;
					snprintf(r->how, sizeof r->how, "f is constant on part "
					    "of %s (checked numerically)", iv_text(a, b, 0));
					ev_close(&e);
					return;
				}
			}
		}
		for (k = 0; k < m.npc; k++)
			for (j = k + 1; j < m.npc; j++) {
				double ol = fmax(lo[k], lo[j]), oh = fmin(hi[k], hi[j]), y;
				double a1, b1, a2, b2, dd;
				if (!(ol < oh))
					continue;
				y = 0.5 * (ol + oh);
				/* adjacent pieces: a short step past the turning point */
				if (j == k + 1 && m.pb[k] == m.pa[j])
					for (dd = 1; dd > 1e-3; dd /= 2) {
						double yy;
						if (m.pa[j] + dd < m.pb[j] &&
						    ev1(&e, m.pa[j] + dd, &yy) && yy > ol && yy < oh) {
							y = yy;
							break;
						}
					}
				a1 = isinf(m.pa[k]) ? fmin(-10, m.pb[k] - 10) : m.pa[k];
				b1 = isinf(m.pb[k]) ? fmax(10, m.pa[k] + 10) : m.pb[k];
				a2 = isinf(m.pa[j]) ? fmin(-10, m.pb[j] - 10) : m.pa[j];
				b2 = isinf(m.pb[j]) ? fmax(10, m.pa[j] + 10) : m.pb[j];
				r->x1 = solve_in(&e, a1, b1, y);
				r->x2 = solve_in(&e, a2, b2, y);
				if (isnan(r->x1) || isnan(r->x2))
					continue;
				r->x1 = snap_x(r->x1, 1e-9);
				r->x2 = snap_x(r->x2, 1e-9);
				ev1(&e, r->x1, &r->y);
				r->inj = 0;
				r->proved = m.proved;
				snprintf(r->how, sizeof r->how, "the value is taken on two "
				    "pieces of monotonicity (%s, witness solved by "
				    "bisection)", m.proved ? "pieces proved" :
				    "checked numerically");
				ev_close(&e);
				return;
			}
	}
	r->inj = 1;
	r->proved = 0;
	snprintf(r->how, sizeof r->how, "strictly monotonic on each piece of "
	    "its domain with disjoint value ranges (checked numerically)");
	ev_close(&e);
}

struct sur {
	int sur;              /* 1 yes, 0 no, -1 unknown */
	int proved;
	double miss;          /* a value never taken */
	double olo, ohi;      /* the codomain */
	char cod[96];
	char how[300];
};

static void
sur_analyse(struct func *f, int s, int onto, double olo, double ohi,
    struct sur *r)
{
	struct dom d;
	struct rngres R;
	ty_t rt = N.ty[f->body];

	memset(r, 0, sizeof *r);
	dom_of(f, s, &d);
	r->sur = -1;
	if (TY_SIZED(rt) && !onto) {
		struct ity it;
		ity_of(&it, rt, 0);
		olo = TYK(rt) == K_SINT ? (double)it.smin : 0;
		ohi = TYK(rt) == K_SINT ? (double)it.smax : (double)it.mask;
		onto = 1;
		snprintf(r->cod, sizeof r->cod, "its type %s (%s..%s)", ty_str(rt),
		    num(olo), num(ohi));
	} else if (onto)
		snprintf(r->cod, sizeof r->cod, "[%s, %s]", num(olo), num(ohi));
	else if (TY_FP(rt))
		snprintf(r->cod, sizeof r->cod, "the reals");
	else
		snprintf(r->cod, sizeof r->cod, "the integers");
	r->olo = onto ? olo : -INFINITY;
	r->ohi = onto ? ohi : INFINITY;
	range_of(f, s, &d, &R);
	if (d.isint) {
		double *xs;
		int64_t *xi, *yi, *v;
		uint8_t *ok;
		uint32_t n = dom_points(&d, 0, &xs, &xi), i, m = 0;
		struct ev e;
		if (!onto || ohi - olo > 1e8) {
			r->sur = 0;
			r->proved = 1;
			r->miss = R.hi + 1;
			snprintf(r->how, sizeof r->how, "a finite domain cannot cover "
			    "%s; %s is never taken", r->cod, num(r->miss));
			free(xs);
			free(xi);
			return;
		}
		ev_open(&e, f, f->body, s, 0);
		yi = xcalloc(n, sizeof *yi);
		ok = xcalloc(n, 1);
		ev_run(&e, xs, xi, n, NULL, yi, ok);
		v = xcalloc(n ? n : 1, sizeof *v);
		for (i = 0; i < n; i++)
			if (ok[i])
				v[m++] = TYK(rt) == K_UINT ? (int64_t)ival_d(rt, yi[i]) :
				    yi[i];
		qsort(v, m, sizeof *v, i64cmp);
		{
			int64_t want = (int64_t)olo;
			r->sur = 1;
			for (i = 0; i < m && want <= (int64_t)ohi; i++) {
				if (v[i] < want)
					continue;
				if (v[i] > want)
					break;
				want++;
			}
			if (want <= (int64_t)ohi) {
				r->sur = 0;
				r->miss = (double)want;
			}
		}
		r->proved = !d.sampled || !r->sur;
		snprintf(r->how, sizeof r->how, "%llu distinct values (%s)",
		    (unsigned long long)R.distinct, scan_text(&d, vname(f, s)));
		free(xs);
		free(xi);
		free(yi);
		free(ok);
		free(v);
		ev_close(&e);
		return;
	}
	{
		struct bnd b;
		struct disc D;
		struct rpoly P;
		bnd_analyse(f, s, &b);
		if (!f->where && TYK(f->p[s].ty) == K_REAL && d.unranged &&
		    rpoly_of(f->body, s, &P) && P.deg >= 1 && P.deg % 2 == 1 &&
		    !onto) {
			r->sur = 1;
			r->proved = 1;
			snprintf(r->how, sizeof r->how, "proved: a polynomial of odd "
			    "degree %d is continuous and tends to -inf and +inf, so it "
			    "takes every value (intermediate value theorem)", P.deg);
			return;
		}
		if ((b.up == BD_PROVED || b.up == BD_NUM) && b.sup < r->ohi) {
			r->sur = 0;
			r->miss = isinf(r->ohi) ? (b.up == BD_PROVED ? b.psup : b.sup) +
			    1 : r->ohi;
			r->proved = b.up == BD_PROVED;
			snprintf(r->how, sizeof r->how, "f is bounded above by %s: %s",
			    num(b.sup), b.hu);
			return;
		}
		if ((b.dn == BD_PROVED || b.dn == BD_NUM) && b.inf > r->olo) {
			r->sur = 0;
			r->miss = isinf(r->olo) ? (b.dn == BD_PROVED ? b.pinf : b.inf) -
			    1 : r->olo;
			r->proved = b.dn == BD_PROVED;
			snprintf(r->how, sizeof r->how, "f is bounded below by %s: %s",
			    num(b.inf), b.hd);
			return;
		}
		disc_analyse(f, s, &D);
		if (D.ndom == 1 && D.n == 0) {
			r->sur = 1;
			r->proved = 0;
			snprintf(r->how, sizeof r->how, "continuous on one interval, "
			    "with values from %s to %s, so every value between is taken "
			    "(intermediate value theorem; continuity proved only for "
			    "elementary bodies, the extremes checked numerically)",
			    b.dn == BD_UNB || b.dn == BD_UNBNUM ? "-inf" : num(b.inf),
			    b.up == BD_UNB || b.up == BD_UNBNUM ? "inf" : num(b.sup));
			return;
		}
		r->sur = -1;
		snprintf(r->how, sizeof r->how, "f is discontinuous or its domain "
		    "has gaps; whether every value is taken was not decided "
		    "(checked numerically: values from %s to %s)", num(R.lo),
		    num(R.hi));
	}
}

static void
inj_print(struct func *f, int s, const struct inj *r)
{
	const char *x = vname(f, s);

	printf("  injective: %s\n", r->inj ? "yes (one-to-one)" : "no");
	if (!r->inj)
		printf("  witness: f(%s) = f(%s) = %s with %s = %s and %s = %s\n",
		    num(r->x1), num(r->x2), num(fabs(r->y) < 1e-12 ? 0 : r->y), x,
		    numx(r->x1), x, numx(r->x2));
	printf("  %s\n", r->how);
}

static void
sur_print(const struct sur *r)
{
	printf("  surjective onto %s: %s\n", r->cod, r->sur > 0 ? "yes" :
	    r->sur == 0 ? "no" : "not decided");
	if (r->sur == 0)
		printf("  witness: %s is never taken\n", num(r->miss));
	printf("  %s\n", r->how);
}

/* ---- rational functions N/D ---- */

static int
rp_mul(const struct rpoly *a, const struct rpoly *b, struct rpoly *r)
{
	struct rpoly t;
	int i, j;

	if (a->deg + b->deg > 12)
		return 0;
	memset(&t, 0, sizeof t);
	t.deg = a->deg + b->deg;
	for (i = 0; i <= a->deg; i++)
		for (j = 0; j <= b->deg; j++)
			t.c[i + j] += a->c[i] * b->c[j];
	while (t.deg > 0 && t.c[t.deg] == 0)
		t.deg--;
	*r = t;
	return 1;
}

static void
rp_add(const struct rpoly *a, const struct rpoly *b, double sg,
    struct rpoly *r)
{
	struct rpoly t;
	int i;

	memset(&t, 0, sizeof t);
	t.deg = a->deg > b->deg ? a->deg : b->deg;
	for (i = 0; i <= t.deg; i++)
		t.c[i] = (i <= a->deg ? a->c[i] : 0) + sg * (i <= b->deg ? b->c[i] :
		    0);
	while (t.deg > 0 && t.c[t.deg] == 0)
		t.deg--;
	*r = t;
}

static void
rp_const(struct rpoly *r, double c)
{
	memset(r, 0, sizeof *r);
	r->c[0] = c;
}

static int
rat_of(nid e, int s, struct rpoly *n, struct rpoly *d)
{
	struct rpoly n1, d1, n2, d2, t;
	double v;
	int64_t k;
	int i;

	if (!has_slot(e, s)) {
		if (!num_of(e, &v))
			return 0;
		rp_const(n, v);
		rp_const(d, 1);
		return 1;
	}
	switch (N.op[e]) {
	case O_VAR:
		rp_const(n, 0);
		n->deg = 1;
		n->c[1] = 1;
		rp_const(d, 1);
		return 1;
	case O_NEG:
		if (!rat_of(N.a[e], s, n, d))
			return 0;
		for (i = 0; i <= n->deg; i++)
			n->c[i] = -n->c[i];
		return 1;
	case O_CAST:
		return TY_FP(N.ty[e]) && TY_FP(N.ty[N.a[e]]) &&
		    rat_of(N.a[e], s, n, d);
	case O_ADD:
	case O_SUB:
		if (!rat_of(N.a[e], s, &n1, &d1) || !rat_of(N.b[e], s, &n2, &d2) ||
		    !rp_mul(&n1, &d2, &n1) || !rp_mul(&n2, &d1, &n2) ||
		    !rp_mul(&d1, &d2, d))
			return 0;
		rp_add(&n1, &n2, N.op[e] == O_ADD ? 1 : -1, n);
		return 1;
	case O_MUL:
		return rat_of(N.a[e], s, &n1, &d1) && rat_of(N.b[e], s, &n2, &d2) &&
		    rp_mul(&n1, &n2, n) && rp_mul(&d1, &d2, d);
	case O_DIV:
		return rat_of(N.a[e], s, &n1, &d1) && rat_of(N.b[e], s, &n2, &d2) &&
		    rp_mul(&n1, &d2, n) && rp_mul(&d1, &n2, d);
	case O_POW:
		if (!int_of(N.b[e], &k) || k < -12 || k > 12 ||
		    !rat_of(N.a[e], s, &n1, &d1))
			return 0;
		rp_const(n, 1);
		rp_const(d, 1);
		for (i = 0; i < (k < 0 ? -k : k); i++)
			if (!rp_mul(n, &n1, n) || !rp_mul(d, &d1, d))
				return 0;
		if (k < 0) {
			t = *n;
			*n = *d;
			*d = t;
		}
		return 1;
	}
	return 0;
}

/* the multiplicity of r as a root of P (numerically: derivatives vanish) */
static int
root_mult(const struct rpoly *P, double r)
{
	struct rpoly D = *P, t;
	double sc = 0;
	int i, m = 0;

	for (i = 0; i <= P->deg; i++)
		sc += fabs(P->c[i]) * pow(1 + fabs(r), i);
	while (m <= P->deg && fabs(rpoly_at(&D, r)) <= 1e-9 * sc) {
		m++;
		rpoly_deriv(&D, &t);
		D = t;
		sc *= D.deg + 1;
	}
	return m;
}

/* ---- asymptotes ---- */

struct asy {
	int nv, nhole;
	double vx[MAXDI], vl[MAXDI], vr[MAXDI], hx[MAXDI], hy[MAXDI];
	int vside[MAXDI];
	int kind[2];          /* x -> -inf, +inf: 0 none, 1 horizontal, 2 oblique,
				 3 undefined there, 4 the domain is bounded */
	double m[2], b[2];
	int proved;
	int rational;
	char how[300];
	struct dom d;
};

static void
asy_analyse(struct func *f, int s, struct asy *A)
{
	struct rpoly Nn, Dd;
	struct ev e;
	int k, sd;

	memset(A, 0, sizeof *A);
	dom_of(f, s, &A->d);
	ev_open(&e, f, f->body, s, 0);
	if (!f->where && TYK(f->p[s].ty) == K_REAL &&
	    rat_of(f->body, s, &Nn, &Dd) && Dd.deg >= 1) {
		struct root r[16];
		int nr = rpoly_roots(&Dd, r);
		double lN = Nn.c[Nn.deg], lD = Dd.c[Dd.deg];
		A->rational = 1;
		A->proved = 1;
		for (k = 0; k < nr; k++) {
			int md = r[k].mult, mn;
			double x = r[k].re, yl = 0, yr = 0, h;
			if (r[k].im != 0)
				continue;
			if (!A->d.unranged && (x < A->d.rlo || x > A->d.rhi))
				continue;
			mn = root_mult(&Nn, x);
			if (mn >= md) {
				if (A->nhole < MAXDI) {
					A->hx[A->nhole] = x;
					h = 1e-7 * (1 + fabs(x));
					ev1(&e, x - h, &yl);
					ev1(&e, x + h, &yr);
					A->hy[A->nhole++] = 0.5 * (yl + yr);
				}
				continue;
			}
			h = 1e-6 * (1 + fabs(x));
			ev1(&e, x - h, &yl);
			ev1(&e, x + h, &yr);
			if (A->nv < MAXDI) {
				A->vx[A->nv] = x;
				A->vl[A->nv] = yl > 0 ? INFINITY : -INFINITY;
				A->vr[A->nv] = yr > 0 ? INFINITY : -INFINITY;
				A->vside[A->nv++] = 0;
			}
		}
		for (sd = 0; sd < 2; sd++) {
			if (!A->d.unranged) {
				A->kind[sd] = 4;
				continue;
			}
			if (Nn.deg < Dd.deg) {
				A->kind[sd] = 1;
				A->b[sd] = 0;
			} else if (Nn.deg == Dd.deg) {
				A->kind[sd] = 1;
				A->b[sd] = lN / lD;
			} else if (Nn.deg == Dd.deg + 1) {
				A->kind[sd] = 2;
				A->m[sd] = lN / lD;
				A->b[sd] = (Nn.c[Nn.deg - 1] - A->m[sd] * Dd.c[Dd.deg - 1]) /
				    lD;
			} else
				A->kind[sd] = 0;
		}
		snprintf(A->how, sizeof A->how, "proved symbolically: f = N/D with "
		    "deg N = %d, deg D = %d; poles are the real roots of D that "
		    "are roots of N of lower multiplicity (others are holes); at "
		    "infinity the quotient of N by D", Nn.deg, Dd.deg);
		ev_close(&e);
		return;
	}
	{
		struct disc D;
		disc_analyse(f, s, &D);
		for (k = 0; k < D.n && A->nv < MAXDI; k++) {
			if (D.kind[k] == DI_INFINITE || (D.kind[k] == DI_EDGE &&
			    isinf(D.side[k] < 0 ? D.l[k] : D.r[k]))) {
				A->vx[A->nv] = D.x[k];
				A->vl[A->nv] = D.l[k];
				A->vr[A->nv] = D.r[k];
				A->vside[A->nv++] = D.kind[k] == DI_EDGE ? D.side[k] : 0;
			}
			if (D.kind[k] == DI_REMOVABLE && A->nhole < MAXDI) {
				A->hx[A->nhole] = D.x[k];
				A->hy[A->nhole++] = D.l[k];
			}
		}
	}
	for (sd = 0; sd < 2; sd++) {
		static const double px[] = {1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
		double y[6], mm[6], bb[6];
		int okc = 1;
		if (!A->d.unranged) {
			A->kind[sd] = 4;
			continue;
		}
		for (k = 0; k < 6; k++) {
			double x = sd ? px[k] : -px[k];
			if (!ev1(&e, x, &y[k]) || !isfinite(y[k])) {
				okc = 0;
				break;
			}
			mm[k] = y[k] / x;
		}
		if (!okc) {
			A->kind[sd] = k == 0 ? 3 : 0;
			continue;
		}
		if (!(fabs(mm[5] - mm[4]) <= 1e-6 * (1 + fabs(mm[5])) &&
		    fabs(mm[4] - mm[3]) <= 1e-4 * (1 + fabs(mm[5])))) {
			A->kind[sd] = 0;
			continue;
		}
		if (fabs(mm[5]) < 1e-7)
			mm[5] = 0;
		for (k = 0; k < 6; k++)
			bb[k] = y[k] - mm[5] * (sd ? px[k] : -px[k]);
		if (!(fabs(bb[5] - bb[4]) <= 1e-5 * (1 + fabs(bb[5])) &&
		    fabs(bb[4] - bb[3]) <= 1e-3 * (1 + fabs(bb[5])))) {
			A->kind[sd] = 0;
			continue;
		}
		A->kind[sd] = mm[5] == 0 ? 1 : 2;
		A->m[sd] = snap_x(mm[5], 1e-9);
		/* f - m x ~ b + c/x: extrapolate from the last two */
		bb[5] = (px[5] * bb[5] - px[4] * bb[4]) / (px[5] - px[4]);
		A->b[sd] = fabs(bb[5]) < 1e-6 ? 0 : bb[5];
		if (*nice(A->b[sd]) || fabs(A->b[sd] - floor(A->b[sd] + 0.5)) < 1e-6)
			A->b[sd] = fabs(A->b[sd] - floor(A->b[sd] + 0.5)) < 1e-6 ?
			    floor(A->b[sd] + 0.5) : A->b[sd];
	}
	snprintf(A->how, sizeof A->how, "checked numerically: poles from a grid "
	    "scan%s; at infinity the limits of f(x)/x and f(x) - m x estimated at "
	    "|x| = 1e3..1e8", A->d.unranged ? " of [-10, 10]" : "");
	ev_close(&e);
}

static const char *
side_text(double v, const char *side)
{
	char *r = rbuf();

	snprintf(r, 128, "%s from the %s", lim_text(v), side);
	return r;
}

static const char *
line_text(double m, double b)
{
	char *r = rbuf();

	char mx[96];

	if (m == 1 || m == -1)
		snprintf(mx, sizeof mx, "%sx", m < 0 ? "-" : "");
	else
		snprintf(mx, sizeof mx, "%s*x", numx(m));
	if (m == 0)
		snprintf(r, 128, "y = %s", numx(b));
	else if (b == 0)
		snprintf(r, 128, "y = %s", mx);
	else
		snprintf(r, 128, "y = %s %c %s", mx, b < 0 ? '-' : '+',
		    numx(fabs(b)));
	return r;
}

static void
asy_print(struct func *f, int s, const struct asy *A)
{
	int k, any = 0;
	const char *x = vname(f, s);

	for (k = 0; k < A->nv; k++, any++)
		printf("  vertical: %s = %s (f -> %s%s%s)\n", x, numx(A->vx[k]),
		    A->vside[k] > 0 ? "" : side_text(A->vl[k], "left"),
		    A->vside[k] == 0 ? ", " : "",
		    A->vside[k] < 0 ? "" : side_text(A->vr[k], "right"));
	if (A->kind[0] == A->kind[1] && (A->kind[0] == 1 || A->kind[0] == 2) &&
	    A->m[0] == A->m[1] && close_to(A->b[0], A->b[1], 1e-9)) {
		printf("  %s: %s as %s -> ±inf\n", A->kind[0] == 1 ? "horizontal" :
		    "oblique", line_text(A->m[0], A->b[0]), x);
		any++;
	} else
		for (k = 0; k < 2; k++)
			if (A->kind[k] == 1 || A->kind[k] == 2) {
				printf("  %s: %s as %s -> %s\n", A->kind[k] == 1 ?
				    "horizontal" : "oblique", line_text(A->m[k], A->b[k]),
				    x, k ? "+inf" : "-inf");
				any++;
			}
	for (k = 0; k < A->nhole; k++)
		printf("  hole (removable, not an asymptote): %s = %s, f -> %s\n", x,
		    numx(A->hx[k]), numx(A->hy[k]));
	if (!any)
		printf("  none%s\n", A->kind[0] == 4 ? " (no poles in the interval; "
		    "it is bounded, so nothing at infinity)" : "");
	else if (A->kind[0] == 4)
		printf("  (%s is bounded to %s: nothing at infinity)\n", x,
		    iv_text(A->d.rlo, A->d.rhi, 0));
	for (k = 0; k < 2; k++)
		if (A->kind[k] == 0 && A->d.unranged && any)
			printf("  none as %s -> %s\n", x, k ? "+inf" : "-inf");
	printf("  %s\n", A->how);
}

/* ---- symmetry: parity, axes and centres ---- */

static int
axis_at(struct ev *e, double c, double H, int point, double *wh)
{
	double fc = 0;
	int k, cmp = 0;

	if (point && !ev1(e, c, &fc))
		return 0;
	for (k = 1; k <= 200; k++) {
		double h = H * k / 200.0, a, b;
		int oa = ev1(e, c + h, &a), ob = ev1(e, c - h, &b);
		if (oa != ob) {
			*wh = h;
			return 0;
		}
		if (!oa || fabs(a) > 1e8 || fabs(b) > 1e8)
			continue;
		cmp++;
		if (point ? !(close_to(a + b, 2 * fc, 1e-9) || fabs(a + b - 2 * fc)
		    < 1e-9 * (1 + fabs(a) + fabs(b))) : !(close_to(a, b, 1e-9) ||
		    fabs(a - b) < 1e-12)) {
			*wh = h;
			return 0;
		}
	}
	return cmp > 20;
}

static void
cmd_symmetry(struct func *f, int s)
{
	struct par r;

	printf("symmetry of %s:\n", fname(f));
	if (f->np == 2 && s == 0) {
		nid map[MAXP] = {0}, sw;
		struct sbuf w = {0};
		int sampled = 0, bad, abad;
		map[0] = mk_var(1, f->p[1].name, f->p[1].ty);
		map[1] = mk_var(0, f->p[0].name, f->p[0].ty);
		if (f->p[0].ty != f->p[1].ty) {
			printf("  %s and %s have different types: no swap symmetry\n",
			    vname(f, 0), vname(f, 1));
			return;
		}
		sw = subst(f->body, map, 2);
		if (TY_FP(f->p[0].ty)) {
			/* real arithmetic: equal up to rounding */
			struct ev e1, e2;
			double v[2], a1, a2;
			int i;
			ev_open(&e1, f, f->body, 0, 0);
			ev_open(&e2, f, sw, 0, 0);
			bad = abad = 0;
			for (i = 0; i < 400; i++) {
				v[0] = urand(-10, 10);
				v[1] = urand(-10, 10);
				if (evm(&e1, v, &a1) != evm(&e2, v, &a2)) {
					bad = abad = 1;
					break;
				}
				if (!close_to(a1, a2, 1e-9) && fabs(a1 - a2) > 1e-12 &&
				    !bad++)
					sb_printf(&w, "%s=%s, %s=%s: %s vs %s", vname(f, 0),
					    num(v[0]), vname(f, 1), num(v[1]), num(a1), num(a2));
				if (!close_to(a1, -a2, 1e-9) && fabs(a1 + a2) > 1e-12)
					abad = 1;
			}
			ev_close(&e1);
			ev_close(&e2);
			printf("  f(%s, %s) = f(%s, %s): %s%s\n", vname(f, 0),
			    vname(f, 1), vname(f, 1), vname(f, 0), bad ? "no, e.g. at " :
			    "yes (symmetric)", bad ? w.p : "");
			if (!abad)
				printf("  f(%s, %s) = -f(%s, %s): yes (antisymmetric)\n",
				    vname(f, 0), vname(f, 1), vname(f, 1), vname(f, 0));
			printf("  checked numerically at 400 random points of "
			    "[-10, 10]^2\n");
			free(w.p);
			return;
		}
		bad = equiv_check(f, f->body, sw, &w, &sampled);
		printf("  f(%s, %s) = f(%s, %s): %s%s%s\n", vname(f, 0), vname(f, 1),
		    vname(f, 1), vname(f, 0), bad ? "no, e.g. at " : "yes "
		    "(symmetric)", bad ? w.p : "", "");
		sb_clear(&w);
		abad = equiv_check(f, f->body, mk_un(O_NEG, sw), &w, &sampled);
		if (!abad)
			printf("  f(%s, %s) = -f(%s, %s): yes (antisymmetric)\n",
			    vname(f, 0), vname(f, 1), vname(f, 1), vname(f, 0));
		printf("  %s\n", sampled ? "checked on a sample (not proved)" :
		    TY_FP(f->p[0].ty) ? "checked numerically on a grid" :
		    "proved, exhaustive");
		free(w.p);
		return;
	}
	if (!one_param(f, "symmetry"))
		return;
	par_analyse(f, s, &r);
	par_print(f, s, &r);
	if (!TY_FP(f->p[s].ty) || r.even || r.odd)
		return;
	{
		struct ev e;
		struct dom d;
		double cs[64], H, wh = 0;
		int nc, k, found = 0;
		struct cvx c;
		dom_of(f, s, &d);
		ev_open(&e, f, f->body, s, 0);
		nc = crit_roots(f, d.rlo, d.rhi, cs, 60);
		for (k = 0; k < nc && !found; k++) {
			double c0 = snap_x(cs[k], 1e-9);
			H = d.unranged ? 10 : fmin(c0 - d.rlo, d.rhi - c0);
			if (H > 0 && axis_at(&e, c0, H, 0, &wh)) {
				printf("  symmetric about the line %s = %s: f(c + h) = "
				    "f(c - h) (checked numerically at 200 h up to %g)\n",
				    vname(f, s), numx(c0), H);
				found = 1;
			}
		}
		cvx_analyse(f, s, &c);
		for (k = 0; k < c.ninf && found < 2; k++) {
			double c0 = snap_x(c.inf[k], 1e-9), fc = 0;
			H = d.unranged ? 10 : fmin(c0 - d.rlo, d.rhi - c0);
			if (H > 0 && axis_at(&e, c0, H, 1, &wh)) {
				ev1(&e, c0, &fc);
				printf("  point symmetric about (%s, %s): f(c + h) + f(c - "
				    "h) = 2 f(c) (checked numerically at 200 h up to %g)\n",
				    numx(c0), numx(fc), H);
				found = 2;
			}
		}
		if (!found)
			printf("  no axis of symmetry at a critical point and no centre "
			    "at an inflection point (checked numerically)\n");
		ev_close(&e);
	}
	{
		struct per pr;
		per_analyse(f, s, &pr);
		if (pr.status == 1)
			printf("  periodic with period %s: the pattern repeats (a "
			    "translation symmetry)\n", per_text(&pr));
	}
}

/* ---- the commands ---- */

struct pargs {
	struct func *f;
	int slot;
	int onto;
	double olo, ohi;
	struct func tmp;
};

static void
parse_range(struct P *p, const char *cmd, double *lo, double *hi)
{
	static const char *const to[] = {"to", NULL};

	if (cmdp_tok(p, T_LB) || cmdp_tok(p, T_LP)) {
		*lo = cmdp_number(p, NULL);
		if (!cmdp_tok(p, T_COMMA))
			fail("%s: expected [a, b]", cmd);
		*hi = cmdp_number(p, NULL);
		if (!cmdp_tok(p, T_RB) && !cmdp_tok(p, T_RP))
			fail("%s: expected [a, b]", cmd);
		return;
	}
	*lo = cmdp_number(p, to);
	if (!cmdp_tok(p, T_DOTDOT) && !cmdp_kw(p, "to"))
		fail("%s: expected a..b", cmd);
	*hi = cmdp_number(p, NULL);
}

static void
parse_args(struct P *p, struct pargs *a, const char *cmd)
{
	static const char *const stop[] = {"in", "on", "over", "from", "onto",
		NULL};
	double lo = 0, hi = 0;
	int ranged = 0, k;
	char w[64];
	uint32_t pos;

	memset(a, 0, sizeof *a);
	a->f = cmdp_func(p, stop);
	pos = cmdp_pos(p);
	if (cmdp_word(p, w, sizeof w)) {
		uint32_t nm = sym_cstr(w);
		for (k = 0; k < a->f->np; k++)
			if (a->f->p[k].name == nm)
				break;
		if (k < a->f->np)
			a->slot = k;
		else
			cmdp_seek(p, pos);
	}
	for (;;) {
		if (cmdp_kw(p, "in") || cmdp_kw(p, "on") || cmdp_kw(p, "over")) {
			parse_range(p, cmd, &lo, &hi);
			ranged = 1;
		} else if (cmdp_kw(p, "from")) {
			static const char *const to[] = {"to", NULL};
			lo = cmdp_number(p, to);
			if (!cmdp_kw(p, "to"))
				fail("%s: expected from a to b", cmd);
			hi = cmdp_number(p, NULL);
			ranged = 1;
		} else if (cmdp_kw(p, "onto")) {
			parse_range(p, cmd, &a->olo, &a->ohi);
			a->onto = 1;
		} else
			break;
	}
	if (!cmdp_eos(p)) {
		cmdp_fail_here(p);
		fail("%s: unexpected text after the function", cmd);
	}
	if (ranged) {
		struct param *pp;
		if (a->f->np == 0)
			fail("%s: %s has no parameter to restrict", cmd, fname(a->f));
		if (lo > hi) {
			double t = lo;
			lo = hi;
			hi = t;
			note("read the interval as [%g, %g]", lo, hi);
		}
		a->tmp = *a->f;
		pp = &a->tmp.p[a->slot];
		pp->ranged = 1;
		pp->rlo = lo;
		pp->rhi = hi;
		pp->lo = (int64_t)ceil(lo);
		pp->hi = (int64_t)floor(hi);
		a->f = &a->tmp;
	}
	if (a->onto && a->olo > a->ohi) {
		double t = a->olo;
		a->olo = a->ohi;
		a->ohi = t;
	}
}

/*
 * An unbounded int parameter (English "x in 0..3" infers int) is read as
 * real for the analyses about real arguments, with a note.
 */
static struct func *
as_real(struct func *f, int s, struct func *tmp)
{
	nid map[MAXP] = {0};

	if (TYK(f->p[s].ty) != K_INT)
		return f;
	*tmp = *f;
	tmp->p[s].ty = TY_REAL;
	if (tmp->p[s].ranged) {
		tmp->p[s].rlo = (double)tmp->p[s].lo;
		tmp->p[s].rhi = (double)tmp->p[s].hi;
	}
	map[s] = mk_var((uint32_t)s, f->p[s].name, TY_REAL);
	tmp->body = subst(f->body, map, f->np);
	if (f->where)
		tmp->where = subst(f->where, map, f->np);
	printf("  note: %s read as real (it was int)\n", vname(f, s));
	return tmp;
}

static int
real_only(struct func *f, int s, const char *what)
{
	if (TY_FP(f->p[s].ty))
		return 1;
	printf("  %s is about real arguments; %s is %s (a discrete domain): "
	    "see monotonic, crit or table\n", what, vname(f, s),
	    ty_str(f->p[s].ty));
	return 0;
}

static void cmd_identify(struct P *p);

static const char *const pcmds[] = {"monotonic", "monotone", "period",
	"periodic", "identify", "classify", "parity", "symmetry", "bounded",
	"injective", "surjective", "bijective", "continuity",
	"discontinuities", "convexity", "convex", "concave", "asymptotes",
	"zeros", "roots", NULL};

int
props_command(struct P *p)
{
	struct pargs a;
	struct func *f, rt;
	char w[32];
	uint32_t at = cmdp_pos(p);
	int k, s, hdr = 0;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; pcmds[k]; k++)
		if (!strcmp(pcmds[k], w))
			break;
	if (!pcmds[k] || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	if (!strcmp(w, "identify") || !strcmp(w, "classify")) {
		cmd_identify(p);
		return 1;
	}
	parse_args(p, &a, w);
	notes_flush();
	f = a.f;
	s = a.slot;
	if (f->np == 1 && (!strcmp(w, "continuity") || !strcmp(w,
	    "discontinuities") || !strcmp(w, "convexity") || !strcmp(w,
	    "convex") || !strcmp(w, "concave") || !strcmp(w, "asymptotes"))) {
		printf("%s of %s:\n", !strncmp(w, "con", 3) && w[3] != 't' ?
		    "convexity" : w, fname(f));
		f = as_real(f, s, &rt);
		hdr = 1;
	}
	if (!strcmp(w, "monotonic") || !strcmp(w, "monotone")) {
		struct mono m;
		printf("monotonic %s:\n", fname(f));
		if (!one_param(f, "monotonic"))
			return 1;
		mono_analyse(f, s, &m);
		mono_print(f, s, &m);
		if (m.verdict == MO_NONE && m.d.unranged) {
			struct per pr;
			per_analyse(f, s, &pr);
			if (pr.status == 1)
				printf("  periodic (period %s): not monotonic on any "
				    "interval longer than one period\n", per_text(&pr));
		}
	} else if (!strcmp(w, "period") || !strcmp(w, "periodic")) {
		struct per r;
		printf("%s %s:\n", !strcmp(w, "period") ? "period of" : "periodic",
		    fname(f));
		if (f->np == 0) {
			printf("  constant (no parameters)\n");
			return 1;
		}
		per_analyse(f, s, &r);
		per_print(f, s, &r, !strcmp(w, "periodic"));
	} else if (!strcmp(w, "parity")) {
		struct par r;
		printf("parity of %s:\n", fname(f));
		if (!one_param(f, "parity"))
			return 1;
		par_analyse(f, s, &r);
		par_print(f, s, &r);
	} else if (!strcmp(w, "symmetry")) {
		cmd_symmetry(f, s);
	} else if (!strcmp(w, "bounded")) {
		struct bnd b;
		printf("bounded %s:\n", fname(f));
		if (!one_param(f, "bounded"))
			return 1;
		bnd_analyse(f, s, &b);
		bnd_print(f, s, &b);
	} else if (!strcmp(w, "injective") || !strcmp(w, "surjective") ||
	    !strcmp(w, "bijective")) {
		struct inj ij;
		struct sur su;
		printf("%s %s:\n", w, fname(f));
		if (!one_param(f, w))
			return 1;
		if (w[0] != 's') {
			inj_analyse(f, s, &ij);
			inj_print(f, s, &ij);
		}
		if (w[0] != 'i') {
			sur_analyse(f, s, a.onto, a.olo, a.ohi, &su);
			sur_print(&su);
		}
		if (w[0] == 'b')
			printf("  bijective: %s\n", ij.inj && su.sur > 0 ? "yes" :
			    !ij.inj || su.sur == 0 ? "no" : "not decided");
	} else if (!strcmp(w, "continuity") || !strcmp(w, "discontinuities")) {
		struct disc D;
		if (!hdr)
			printf("%s of %s:\n", w, fname(f));
		if (!one_param(f, w) || !real_only(f, s, "continuity"))
			return 1;
		disc_analyse(f, s, &D);
		if (D.n == 0)
			printf("  continuous wherever it is defined on %s%s\n",
			    D.d.unranged ? "[-10, 10]" : dom_text(&D.d), D.elementary &&
			    D.d.unranged ? " (and on all real x where defined)" : "");
		disc_print(f, s, &D);
	} else if (!strcmp(w, "convexity") || !strcmp(w, "convex") ||
	    !strcmp(w, "concave")) {
		struct cvx c;
		if (!hdr)
			printf("convexity of %s:\n", fname(f));
		if (!one_param(f, w) || !real_only(f, s, "convexity"))
			return 1;
		cvx_analyse(f, s, &c);
		cvx_print(f, s, &c);
	} else if (!strcmp(w, "asymptotes")) {
		struct asy A;
		if (!hdr)
			printf("asymptotes of %s:\n", fname(f));
		if (!one_param(f, w) || !real_only(f, s, "asymptotes"))
			return 1;
		asy_analyse(f, s, &A);
		asy_print(f, s, &A);
	} else if (!strcmp(w, "zeros") || !strcmp(w, "roots")) {
		struct zer z;
		printf("zeros of %s:\n", fname(f));
		if (!one_param(f, w))
			return 1;
		zer_analyse(f, s, &z);
		zer_print(f, s, &z);
	}
	return 1;
}

/* ---- identify ---- */

static const char *
rot_text(double ang)
{
	char *b = rbuf();

	snprintf(b, 128, ", rotated by %s rad", numx(ang));
	return b;
}

static const char *
rk_text(int k)
{
	char *b = rbuf();

	snprintf(b, 128, " r^%d", k);
	return b;
}

/* t^m exp(re t) cos(im t) (or sin) */
static const char *
basis_term(double re, double im, int cosine, int m, const char *t)
{
	char *b = rbuf();
	char p[64] = "", ex[64] = "", tr[64] = "";

	if (m == 1)
		snprintf(p, sizeof p, "%s", t);
	else if (m > 1)
		snprintf(p, sizeof p, "%s^%d", t, m);
	if (re == 1 || re == -1)
		snprintf(ex, sizeof ex, "exp(%s%s)", re < 0 ? "-" : "", t);
	else if (re != 0)
		snprintf(ex, sizeof ex, "exp(%s*%s)", num(re), t);
	if (im == 1)
		snprintf(tr, sizeof tr, "%s(%s)", cosine ? "cos" : "sin", t);
	else if (im != 0)
		snprintf(tr, sizeof tr, "%s(%s*%s)", cosine ? "cos" : "sin",
		    num(im), t);
	snprintf(b, 128, "%s%s%s%s%s", p, *p && (*ex || *tr) ? "*" : "", ex,
	    *ex && *tr ? "*" : "", tr);
	if (!*b)
		snprintf(b, 128, "1");
	return b;
}

static const char *
kinds_s(const char *t)
{
	char *b = rbuf();

	snprintf(b, 128, ": %s", t);
	return b;
}

static const char *
per_s(const struct per *r)
{
	char *b = rbuf();

	snprintf(b, 128, "yes, period %s", per_text(r));
	return b;
}

static const char *
wit_s(const struct inj *r)
{
	char *b = rbuf();

	snprintf(b, 128, "no, f(%s) = f(%s) = %s", num(r->x1), num(r->x2),
	    num(fabs(r->y) < 1e-12 ? 0 : r->y));
	return b;
}

static const char *
pieces_s(const struct mono *m)
{
	char *b = rbuf();

	snprintf(b, 128, " (%d intervals of monotonicity)", m->ntotal);
	return b;
}

static const char *
infl_s(const struct cvx *c)
{
	char *b = rbuf();
	int k;

	snprintf(b, 128, ", inflection at");
	for (k = 0; k < c->ninf && k < 3; k++)
		snprintf(b + strlen(b), 128 - strlen(b), "%s %s", k ? "," : "",
		    num(c->inf[k]));
	if (c->ninf > 3)
		snprintf(b + strlen(b), 128 - strlen(b), ", ...");
	return b;
}

struct kinds {
	int exp, log, trig, invtrig, alg, piece, bits, loop, lambert, rat;
	const char *piecewhy;
};

static void
kinds_of(nid e, int s, struct kinds *k)
{
	int op;

	if (e == 0 || !(N.fl[e] & NF_VAR) || N.op[e] == O_VAR ||
	    N.op[e] == O_TABLE)
		return;
	op = N.op[e];
	switch (op) {
	case O_EXP:
		k->exp = 1;
		break;
	case O_POW:
		if (node_slots(N.b[e]))
			k->exp = 1;
		else {
			int64_t n;
			if (!int_of(N.b[e], &n))
				k->alg = 1;
			else if (n < 0)
				k->rat = 1;
		}
		break;
	case O_LOG:
		k->log = 1;
		break;
	case O_SIN:
	case O_COS:
	case O_TAN:
	case O_SEC:
	case O_CSC:
	case O_COT:
		k->trig = 1;
		break;
	case O_ASIN: case O_ACOS: case O_ATAN2:	/* forms.c */
	case O_ATAN:
		k->invtrig = 1;
		break;
	case O_SINH: case O_COSH: case O_TANH:	/* exponentials */
		k->exp = 1;
		break;
	case O_ASINH: case O_ACOSH: case O_ATANH:	/* logarithms */
		k->log = 1;
		break;
	case O_SQRT:
		k->alg = 1;
		break;
	case O_DIV:
		if (node_slots(N.b[e]))
			k->rat = 1;
		break;
	case O_LAMBERTW:
	case O_LAMBERTWM1:
		k->lambert = 1;
		break;
	case O_LOOP:
		k->loop = 1;
		return;
	case O_ABS:
	case O_SIGN:
	case O_FLOOR:
	case O_CEIL:
	case O_MIN:
	case O_MAX:
	case O_SEL:
		k->piece = 1;
		if (!k->piecewhy)
			k->piecewhy = opinfo[op].name;
		break;
	default:
		if ((opinfo[op].fl & OF_INT) || op == O_INDEX || op == O_CLZ ||
		    op == O_CTZ || op == O_POPCNT || op == O_ISQRT)
			k->bits = 1;
	}
	kinds_of(N.a[e], s, k);
	kinds_of(N.b[e], s, k);
	kinds_of(N.c[e], s, k);
}

static const char *
kinds_text(const struct kinds *k)
{
	char *b = rbuf();
	int n = 0;

	b[0] = 0;
#define K(f, w) if (k->f) n += snprintf(b + strlen(b), 128 - strlen(b), \
	"%s%s", n ? ", " : "", w)
	K(exp, "exponential");
	K(log, "logarithmic");
	K(trig, "trigonometric");
	K(invtrig, "inverse trigonometric");
	K(lambert, "Lambert W");
#undef K
	return b;
}

static const char *
tag(int proved)
{
	return proved == 2 ? "[exhaustive]" : proved ? "[proved]" :
	    "[numeric]";
}

static const char *
poly_name(int d)
{
	static const char *const nm[] = {"constant", "linear", "quadratic",
		"cubic", "quartic", "quintic"};

	return d <= 5 ? nm[d] : "polynomial";
}

/* (x - r)^m factors over the reals, from the roots */
static void
factored(const struct rpoly *P, const char *x, struct sbuf *sb)
{
	struct root r[16];
	int nr = rpoly_roots(P, r), k, cnt = 0;
	double lc = P->c[P->deg];

	for (k = 0; k < nr; k++)
		cnt += r[k].mult * (r[k].im != 0 ? 2 : 1);
	if (cnt != P->deg)
		return;
	if (lc != 1)
		sb_printf(sb, "%s", lc == -1 ? "-" : num(lc));
	for (k = 0; k < nr; k++) {
		if (r[k].im != 0) {
			double a = r[k].re, q = r[k].re * r[k].re + r[k].im * r[k].im;
			sb_printf(sb, "(%s^2 %c %s%s + %s)", x, a > 0 ? '-' : '+',
			    num(fabs(2 * a)), x, num(q));
		} else if (r[k].re == 0)
			sb_printf(sb, "%s", x);
		else
		{
			struct root ng = r[k];
			ng.qr.n = -ng.qr.n;
			ng.qb.n = -ng.qb.n;
			ng.quad = -ng.quad;
			sb_printf(sb, "(%s %c %s)", x, r[k].re > 0 ? '-' : '+',
			    r[k].exact || r[k].quad ? root_str(r[k].re > 0 ? &r[k] :
			    &ng) : num(fabs(r[k].re)));
		}
		if (r[k].mult > 1)
			sb_printf(sb, "^%d", r[k].mult);
	}
}

static void
identify_func(struct func *f)
{
	struct kinds k;
	const char *x;

	memset(&k, 0, sizeof k);
	if (f->np == 0) {
		printf("  a constant\n");
		return;
	}
	kinds_of(f->body, 0, &k);
	if (f->np > 1) {
		int j, poly = 1;
		struct sbuf sb = {0};
		for (j = 0; j < f->np; j++) {
			struct npoly P;
			if (npoly_of(f->body, j, &P, 12))
				sb_printf(&sb, "%sdegree %d in %s", j ? ", " : "", P.deg,
				    vname(f, j));
			else
				poly = 0;
		}
		printf("  a function of %d variables%s%s%s\n", f->np, poly ?
		    ": polynomial, " : "", poly ? sb.p : "", !poly && *kinds_text(&k)
		    ? kinds_s(kinds_text(&k)) : "");
		free(sb.p);
		cmd_symmetry(f, 0);
		return;
	}
	x = vname(f, 0);
	if (!TY_FP(f->p[0].ty)) {
		struct dom d;
		struct rngres R;
		struct mono m;
		struct per pr;
		struct inj ij;
		struct sur su;
		struct par pa;
		struct zer z;
		dom_of(f, 0, &d);
		printf("  an integer function of %s: %s -> %s%s\n", x,
		    ty_str(f->p[0].ty), ty_str(N.ty[f->body]), k.bits ?
		    " (bit-level operations)" : "");
		range_of(f, 0, &d, &R);
		printf("  domain: %s (%llu points)\n", dom_text(&d),
		    (unsigned long long)d.n);
		printf("  range: %s..%s, %llu distinct values %s\n", num(R.lo),
		    num(R.hi), (unsigned long long)R.distinct, tag(R.exact ? 2 : 0));
		mono_analyse(f, 0, &m);
		printf("  monotonic: %s %s\n", mono_word(m.verdict), tag(m.proved));
		per_analyse(f, 0, &pr);
		printf("  periodic: %s %s\n", pr.status == 1 ? per_s(&pr) :
		    pr.status == -1 ? "constant" : "no", tag(pr.proved));
		par_analyse(f, 0, &pa);
		printf("  parity: %s%s%s\n", par_word(&pa), pa.na ? "" : " ",
		    pa.na ? "" : tag(pa.proved));
		inj_analyse(f, 0, &ij);
		sur_analyse(f, 0, 0, 0, 0, &su);
		printf("  injective: %s; surjective onto %s: %s %s\n", ij.inj ?
		    "yes" : wit_s(&ij), su.cod, su.sur > 0 ? "yes" : "no",
		    tag(ij.proved ? 2 : 0));
		zer_analyse(f, 0, &z);
		printf("  zeros: %d %s\n", z.ntotal, tag(z.proved ? 2 : 0));
		printf("  (each property has its own command: monotonic, period, "
		    "parity, injective, zeros ...)\n");
		return;
	}
	{
		struct rpoly P, Nn, Dd;
		int ispoly = !f->where && rpoly_of(f->body, 0, &P);
		int israt = !ispoly && !f->where && rat_of(f->body, 0, &Nn, &Dd) &&
		    Dd.deg >= 1;
		struct mono m;
		struct per pr;
		struct par pa;
		struct bnd b;
		struct disc D;
		struct inj ij;
		struct zer z;
		struct cvx c;
		struct asy A;
		struct rngres R;
		const char *xs = x;
		if (ispoly) {
			struct sbuf sb = {0};
			printf("  polynomial of degree %d (%s), leading coefficient %s\n",
			    P.deg, poly_name(P.deg), num(P.c[P.deg]));
			if (P.deg >= 2) {
				factored(&P, xs, &sb);
				if (sb.n)
					printf("  factored over the reals: %s\n", sb.p);
				free(sb.p);
			}
			if (P.deg == 1)
				printf("  a line: slope %s, intercept %s\n", numx(P.c[1]),
				    numx(P.c[0]));
			if (P.deg == 2) {
				double disc = P.c[1] * P.c[1] - 4 * P.c[2] * P.c[0];
				printf("  discriminant b^2 - 4ac = %s: %s; vertex at %s = "
				    "%s, a parabola opening %s\n", num(disc), disc > 0 ?
				    "two real roots" : disc == 0 ? "a double root" :
				    "no real roots", xs, numx(-P.c[1] / (2 * P.c[2])),
				    P.c[2] > 0 ? "upwards" : "downwards");
			}
		} else if (israt)
			printf("  rational function: numerator of degree %d, "
			    "denominator of degree %d%s\n", Nn.deg, Dd.deg,
			    Nn.deg < Dd.deg ? " (proper)" : "");
		else if (k.loop)
			printf("  defined by a loop (a sum, product or recurrence)\n");
		else {
			const char *t = kinds_text(&k);
			if (*t)
				printf("  transcendental: %s%s%s\n", t, k.alg ?
				    ", with radicals" : "", k.rat ? ", with quotients" : "");
			else if (k.alg)
				printf("  algebraic (radicals of rational functions)\n");
			else if (!k.piece)
				printf("  elementary\n");
		}
		if (k.piece)
			printf("  piecewise (%s)\n", k.piecewhy);
		disc_analyse(f, 0, &D);
		if (!(N.fl[f->body] & NF_PARTIAL) && !f->where)
			printf("  domain: all real x [proved: no partial operation]\n");
		else if (D.ndom == 1 && D.da[0] == D.d.rlo && D.db[0] == D.d.rhi)
			printf("  domain: %s, no undefined point found [numeric]\n",
			    D.d.unranged ? "all real x" : dom_text(&D.d));
		else {
			int j, pts = D.da[0] == D.d.rlo && D.db[D.ndom - 1] == D.d.rhi;
			for (j = 0; j + 1 < D.ndom; j++)
				if (!close_to(D.db[j], D.da[j + 1], 1e-12) &&
				    fabs(D.db[j] - D.da[j + 1]) > 1e-300)
					pts = 0;
			if (pts) {
				printf("  domain: %s except", D.d.unranged ? "all real x" :
				    dom_text(&D.d));
				for (j = 0; j + 1 < D.ndom; j++)
					printf("%s %s = %s", j ? "," : "", xs, numx(D.db[j]));
			} else {
				printf("  domain: defined on");
				for (j = 0; j < D.ndom && j < 4; j++)
					printf("%s %s", j ? "," : "", iv_text(D.da[j], D.db[j], 0));
				printf("%s (of %s)", D.ndom > 4 ? ", ..." : "",
				    iv_text(D.d.rlo, D.d.rhi, 0));
			}
			printf(" [numeric]\n");
		}
		{
			int j, nd = 0;
			for (j = 0; j < D.n; j++)
				if (D.kind[j] != DI_EDGE)
					nd++;
			if (nd == 0)
				printf("  continuous on its domain %s\n", D.elementary ?
				    "[proved: continuous operations]" : "[numeric]");
			else {
				printf("  discontinuous at");
				for (j = 0, nd = 0; j < D.n; j++)
					if (D.kind[j] != DI_EDGE)
						printf("%s %s = %s (%s)", nd++ ? "," : "", xs,
						    numx(D.x[j]), D.kind[j] == DI_REMOVABLE ?
						    "removable" : D.kind[j] == DI_JUMP ? "jump" :
						    D.kind[j] == DI_INFINITE ? "pole" : "essential");
				printf(" [numeric]\n");
			}
		}
		par_analyse(f, 0, &pa);
		printf("  parity: %s%s%s\n", par_word(&pa), pa.na ? "" : " ",
		    pa.na ? "" : tag(pa.proved));
		per_analyse(f, 0, &pr);
		printf("  periodic: %s %s\n", pr.status == 1 ? per_s(&pr) :
		    pr.status == -1 ? "constant" : pr.status == 0 ? "no" : "unknown",
		    tag(pr.proved));
		mono_analyse(f, 0, &m);
		printf("  monotonic: %s%s %s\n", mono_word(m.verdict),
		    m.verdict == MO_NONE ? pieces_s(&m) : "", tag(m.proved));
		cvx_analyse(f, 0, &c);
		printf("  %s%s %s\n", c.verdict == 2 ? "affine" : c.verdict == 1 ?
		    "convex" : c.verdict == -1 ? "concave" :
		    "neither convex nor concave", c.ninf ? infl_s(&c) : "",
		    tag(c.proved));
		bnd_analyse(f, 0, &b);
		printf("  %s %s\n", bnd_word(&b), tag((b.up == BD_PROVED ||
		    b.up == BD_UNB) && (b.dn == BD_PROVED || b.dn == BD_UNB)));
		range_of(f, 0, &D.d, &R);
		printf("  values on %s: from %s to %s [numeric]\n",
		    iv_text(D.d.rlo, D.d.rhi, 0), numx(R.lo), numx(R.hi));
		inj_analyse(f, 0, &ij);
		printf("  injective: %s %s\n", ij.inj ? "yes" : wit_s(&ij),
		    ij.inj ? tag(ij.proved) : "[witness]");
		zer_analyse(f, 0, &z);
		if (z.ntotal < 0)
			printf("  zeros: everywhere\n");
		else {
			int j;
			printf("  zeros: %d%s", z.ntotal, z.ntotal ? " (" : "");
			for (j = 0; j < z.n && j < 6; j++)
				printf("%s%s", j ? ", " : "", z.txt[j]);
			printf("%s%s %s\n", z.n > 6 ? ", ..." : "", z.ntotal ? ")" : "",
			    tag(z.proved));
		}
		asy_analyse(f, 0, &A);
		if (A.nv || A.kind[0] == 1 || A.kind[0] == 2 || A.kind[1] == 1 ||
		    A.kind[1] == 2) {
			int j;
			printf("  asymptotes:");
			for (j = 0; j < A.nv && j < 4; j++)
				printf("%s %s = %s", j ? "," : "", xs, numx(A.vx[j]));
			for (j = 0; j < 2; j++)
				if ((A.kind[j] == 1 || A.kind[j] == 2) && (j == 0 ||
				    A.kind[0] != A.kind[1] || A.m[0] != A.m[1] ||
				    !close_to(A.b[0], A.b[1], 1e-9)))
					printf("%s %s (%s)", A.nv || j ? "," : "",
					    line_text(A.m[j], A.b[j]), j == 0 && A.kind[0] ==
					    A.kind[1] && A.m[0] == A.m[1] && close_to(A.b[0],
					    A.b[1], 1e-9) ? "x -> ±inf" : j ? "x -> +inf" :
					    "x -> -inf");
			printf(" %s\n", tag(A.proved));
		}
		if (D.d.unranged)
			printf("  (numeric results are for [-10, 10]: %s has no range; "
			    "each property has its own command)\n", xs);
		else
			printf("  (each property has its own command: monotonic, "
			    "period, parity, bounded, zeros ...)\n");
	}
}

/* ---- equations ---- */

static uint64_t prng = 0x2545f4914f6cdd1dULL;

static double
urand(double a, double b)
{
	prng ^= prng << 13;
	prng ^= prng >> 7;
	prng ^= prng << 17;
	return a + (b - a) * (double)(prng >> 11) / 9007199254740992.0;
}

/* all parameters at v[] */
static int
evm(struct ev *e, const double *v, double *y)
{
	int j;

	for (j = 1; j < e->f->np; j++)
		ev_fix(e, j, v[j]);
	return ev1(e, v[0], y);
}

/* G(x, y) as A x^2 + B xy + C y^2 + D x + E y + F, checked at points */
static int
conic_coeffs(struct ev *e, double *c)
{
	double v[2], g[9];
	int i, j, k = 0, cmp = 0;

	for (i = -1; i <= 1; i++)
		for (j = -1; j <= 1; j++) {
			v[0] = i;
			v[1] = j;
			if (!evm(e, v, &g[k++]))
				return 0;
		}
	/* g index: (i+1)*3 + (j+1) */
#define G(i, j) g[((i) + 1) * 3 + (j) + 1]
	c[5] = G(0, 0);
	c[3] = (G(1, 0) - G(-1, 0)) / 2;
	c[0] = (G(1, 0) + G(-1, 0)) / 2 - c[5];
	c[4] = (G(0, 1) - G(0, -1)) / 2;
	c[2] = (G(0, 1) + G(0, -1)) / 2 - c[5];
	c[1] = (G(1, 1) - G(1, -1) - G(-1, 1) + G(-1, -1)) / 4;
#undef G
	for (k = 0; k < 30; k++) {
		double y, q;
		v[0] = urand(-3, 3);
		v[1] = urand(-3, 3);
		if (!evm(e, v, &y))
			continue;
		q = c[0] * v[0] * v[0] + c[1] * v[0] * v[1] + c[2] * v[1] * v[1] +
		    c[3] * v[0] + c[4] * v[1] + c[5];
		if (!close_to(y, q, 1e-9) && fabs(y - q) > 1e-9)
			return 0;
		cmp++;
	}
	for (k = 0; k < 6; k++)
		if (fabs(c[k]) < 1e-12)
			c[k] = 0;
	return cmp >= 10;
}

/* "y = a(x - h)^2 + k", tidied */
static const char *
vform(const char *lhs, double a, const char *x, double h, double k)
{
	char *b = rbuf();
	char sq[64];

	if (h == 0)
		snprintf(sq, sizeof sq, "%s^2", x);
	else
		snprintf(sq, sizeof sq, "(%s %c %s)^2", x, h > 0 ? '-' : '+',
		    num(fabs(h)));
	snprintf(b, 128, "%s = %s%s%s", lhs, a == 1 ? "" : a == -1 ? "-" :
	    num(a), a == 1 || a == -1 ? "" : "*", sq);
	if (k != 0)
		snprintf(b + strlen(b), 128 - strlen(b), " %c %s", k > 0 ? '+' : '-',
		    num(fabs(k)));
	return b;
}

static void
conic_print(const double *c, const char *xn, const char *yn)
{
	double A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5];
	double disc = B * B - 4 * A * C;
	double det = A * (C * F - E * E / 4) - B / 2 * (B / 2 * F - E * D / 4) +
	    D / 2 * (B / 2 * E / 2 - C * D / 2);
	double sc = fabs(A) + fabs(B) + fabs(C) + fabs(D) + fabs(E) + fabs(F);
	double x0 = 0, y0 = 0, Fc;
	int degen = fabs(det) <= 1e-12 * sc * sc * sc;

	printf("  A %s^2 + B %s%s + C %s^2 + D %s + E %s + F = 0 with A = %s, "
	    "B = %s, C = %s, D = %s, E = %s, F = %s\n", xn, xn, yn, yn, xn, yn,
	    num(A), num(B), num(C), num(D), num(E), num(F));
	if (A == 0 && B == 0 && C == 0) {
		if (D == 0 && E == 0)
			printf("  no variable left: %s\n", F == 0 ? "always true" :
			    "never true (no points)");
		else if (E == 0)
			printf("  a vertical line: %s = %s\n", xn, numx(-F / D));
		else
			printf("  a line: %s = %s*%s %c %s (slope %s)\n", yn,
			    numx(-D / E), xn, -F / E < 0 ? '-' : '+', numx(fabs(F / E)),
			    numx(-D / E));
		return;
	}
	if (fabs(disc) > 1e-12 * sc * sc) {
		/* a centre: the gradient vanishes */
		double dt = 4 * A * C - B * B;
		x0 = (B * E - 2 * C * D) / dt;
		y0 = (B * D - 2 * A * E) / dt;
	}
	Fc = F + (D * x0 + E * y0) / 2;
	printf("  discriminant B^2 - 4AC = %s (%s); determinant of the 3x3 "
	    "matrix %s (%s)\n", num(disc), disc < 0 ? "elliptic" : disc > 0 ?
	    "hyperbolic" : "parabolic", num(det), degen ? "degenerate" :
	    "non-degenerate");
	if (disc < -1e-12 * sc * sc) {
		double l1, l2, t = (A + C) / 2, r = sqrt((A - C) * (A - C) / 4 +
		    B * B / 4);
		l1 = t - r;
		l2 = t + r;
		if (degen)
			printf("  degenerate ellipse: the single point (%s, %s)\n",
			    numx(x0), numx(y0));
		else if ((A + C) * det > 0)
			printf("  an imaginary ellipse: no real points\n");
		else if (B == 0 && A == C)
			printf("  circle: centre (%s, %s), radius %s\n", numx(x0),
			    numx(y0), numx(sqrt(-Fc / A)));
		else {
			double ang = 0.5 * atan2(B, A - C);
			printf("  ellipse: centre (%s, %s), semi-axes %s and %s%s\n",
			    numx(x0), numx(y0), numx(sqrt(-Fc / l1)), numx(sqrt(-Fc /
			    l2)), B != 0 ? rot_text(ang) : "");
		}
	} else if (disc > 1e-12 * sc * sc) {
		if (degen)
			printf("  degenerate hyperbola: two lines crossing at (%s, %s)\n",
			    numx(x0), numx(y0));
		else {
			printf("  hyperbola: centre (%s, %s)", numx(x0), numx(y0));
			if (C != 0) {
				double s1 = (-B + sqrt(disc)) / (2 * C), s2 = (-B -
				    sqrt(disc)) / (2 * C);
				printf(", asymptotes %s and %s", line_text(s1, y0 - s1 * x0),
				    line_text(s2, y0 - s2 * x0));
			} else
				printf(", asymptotes %s = %s and %s", xn, numx(x0),
				    line_text(-A / B, y0 + A / B * x0));
			printf("\n");
		}
	} else {
		if (degen)
			printf("  degenerate parabola: two parallel lines, one line or "
			    "no points\n");
		else if (B == 0 && C == 0)
			printf("  parabola: %s, vertex (%s, %s), axis %s = %s\n",
			    vform(yn, -A / E, xn, -D / (2 * A), (D * D / (4 * A) - F) / E),
			    numx(-D / (2 * A)), numx((D * D / (4 * A) - F) / E), xn,
			    numx(-D / (2 * A)));
		else if (B == 0 && A == 0)
			printf("  parabola: %s, vertex (%s, %s), axis %s = %s\n",
			    vform(xn, -C / D, yn, -E / (2 * C), (E * E / (4 * C) - F) / D),
			    numx((E * E / (4 * C) - F) / D), numx(-E / (2 * C)), yn,
			    numx(-E / (2 * C)));
		else
			printf("  parabola (rotated: axis along %s)\n",
			    rot_text(0.5 * atan2(B, A - C)));
	}
}


static void
symmetries2(struct ev *e, const char *xn, const char *yn)
{
	static const char *const nm[] = {"the y-axis", "the x-axis",
		"the origin", "the line y = x"};
	int t, k, hold[4] = {1, 1, 1, 1}, cmp = 0;

	for (k = 0; k < 40; k++) {
		double v[2], w[2], a, b;
		v[0] = urand(-3, 3);
		v[1] = urand(-3, 3);
		if (!evm(e, v, &a))
			continue;
		cmp++;
		for (t = 0; t < 4; t++) {
			w[0] = t == 0 || t == 2 ? -v[0] : t == 3 ? v[1] : v[0];
			w[1] = t == 1 || t == 2 ? -v[1] : t == 3 ? v[0] : v[1];
			if (!evm(e, w, &b) || !(close_to(a, b, 1e-9) || close_to(a, -b,
			    1e-9) || fabs(fabs(a) - fabs(b)) < 1e-12))
				hold[t] = 0;
		}
	}
	if (cmp == 0)
		return;
	for (t = 0, k = 0; t < 4; t++)
		k += hold[t];
	if (k == 0)
		printf("  no symmetry about the axes, the origin or y = x");
	else
		printf("  symmetric about:");
	for (t = 0, k = 0; t < 4; t++)
		if (hold[t])
			printf("%s %s", k++ ? "," : "", nm[t]);
	printf(" [numeric, 40 points]\n");
	(void)xn;
	(void)yn;
}

static void
identify_eq(struct func *fn, nid F)
{
	struct kinds k;
	struct func t = *fn;

	memset(&k, 0, sizeof k);
	kinds_of(F, 0, &k);
	t.body = F;
	if (fn->np == 0) {
		double v = 0;
		num_of(F, &v);
		printf("  no variables: %s\n", is_const(F) ? (v == 0 ?
		    "always true" : "never true") : "constant");
		return;
	}
	if (fn->np == 1) {
		struct rpoly P;
		struct zer z;
		if (rpoly_of(F, 0, &P))
			printf("  a polynomial equation of degree %d (%s) in %s\n",
			    P.deg, poly_name(P.deg), vname(fn, 0));
		else
			printf("  an equation in %s%s%s\n", vname(fn, 0),
			    *kinds_text(&k) ? ": transcendental, " : "", kinds_text(&k));
		zer_analyse(&t, 0, &z);
		printf("  solutions (zeros of lhs - rhs):\n");
		zer_print(&t, 0, &z);
		return;
	}
	if (fn->np == 2) {
		struct ev e;
		double c[6];
		const char *xn, *yn;
		if (!strcmp(vname(fn, 0), "y") && !strcmp(vname(fn, 1), "x")) {
			/* y = x^2 + 1 named y first: x is the horizontal axis */
			nid map[MAXP] = {0};
			struct param pp = t.p[0];
			map[0] = mk_var(1, t.p[0].name, t.p[0].ty);
			map[1] = mk_var(0, t.p[1].name, t.p[1].ty);
			F = subst(F, map, 2);
			t.p[0] = t.p[1];
			t.p[1] = pp;
			t.body = F;
			*fn = t;
		}
		xn = vname(fn, 0);
		yn = vname(fn, 1);
		nid dy;
		ev_open(&e, &t, F, 0, 0);
		if (!k.exp && !k.log && !k.trig && !k.piece && !k.bits &&
		    conic_coeffs(&e, c)) {
			int lin = c[0] == 0 && c[1] == 0 && c[2] == 0;
			printf("  %s in %s and %s\n", lin ? "a linear equation" :
			    "a conic section (degree 2)", xn, yn);
			conic_print(c, xn, yn);
			printf("  [coefficients by evaluation at 9 points, the form "
			    "checked at 30 more: numeric]\n");
		} else {
			struct npoly P;
			int poly = npoly_of(F, 0, &P, 12) && npoly_of(F, 1, &P, 12);
			printf("  an implicit curve G(%s, %s) = 0%s%s\n", xn, yn,
			    poly ? ": polynomial of degree > 2" : *kinds_text(&k) ?
			    ": transcendental, " : "", poly ? "" : kinds_text(&k));
		}
		dy = try_diff(F, 1);
		if (dy && !has_slot(dy, 1) && !const_is(dy, 0)) {
			nid map[MAXP] = {0}, g;
			map[1] = KR(0);
			g = tidy(DIV(mk_un(O_NEG, subst(F, map, 2)), dy));
			printf("  the graph of a function: %s = %s\n", yn, node_str(g));
		}
		symmetries2(&e, xn, yn);
		ev_close(&e);
		return;
	}
	{
		int j;
		printf("  an implicit surface in %d variables%s%s\n", fn->np,
		    *kinds_text(&k) ? ": transcendental, " : "", kinds_text(&k));
		for (j = 0; j < fn->np; j++) {
			struct npoly P;
			if (npoly_of(F, j, &P, 12))
				printf("    degree %d in %s\n", P.deg, vname(fn, j));
		}
	}
}

/* ---- differential equations ---- */

static int
primes_of(const char *s, uint32_t *base)
{
	size_t n = strlen(s);
	int k = 0;

	while (n > 0 && s[n - 1] == '\'') {
		n--;
		k++;
	}
	*base = sym_intern(s, n);
	return k;
}

static void
identify_ode(struct func *fn, nid F)
{
	uint32_t base[MAXP], dep = 0;
	int ord[MAXP], ys[16], n = 0, j, k, ts = -1, lin = 1, cc = 1, nother = 0;
	uint64_t ymask = 0;
	struct ev e;
	nid coef[16], F0, map[MAXP] = {0};
	char nonlin[160] = "";
	const char *tn;

	for (j = 0; j < fn->np; j++) {
		ord[j] = primes_of(vname(fn, j), &base[j]);
		if (ord[j] > n) {
			n = ord[j];
			dep = base[j];
		}
	}
	for (k = 0; k < 16; k++)
		ys[k] = -1;
	for (j = 0; j < fn->np; j++) {
		if (base[j] == dep && ord[j] < 16) {
			ys[ord[j]] = j;
			ymask |= 1ULL << j;
		} else if (ord[j] > 0) {
			printf("  derivatives of more than one unknown (%s and %s): a "
			    "system; identify takes one equation in one unknown\n",
			    sym_str(dep), sym_str(base[j]));
			return;
		}
	}
	for (j = 0; j < fn->np; j++)
		if (!(ymask >> j & 1)) {
			if (ts < 0 || !strcmp(vname(fn, j), "t") ||
			    (!strcmp(vname(fn, j), "x") && strcmp(vname(fn, ts), "t")))
				ts = j;
		}
	for (j = 0; j < fn->np; j++)
		if (!(ymask >> j & 1) && j != ts)
			nother++;
	tn = ts >= 0 ? vname(fn, ts) : "t";
	printf("  an ordinary differential equation of order %d in %s(%s)\n", n,
	    sym_str(dep), tn);
	for (k = 0; k <= n; k++) {
		coef[k] = 0;
		if (ys[k] < 0) {
			coef[k] = KI(0);
			continue;
		}
		coef[k] = try_diff(F, ys[k]);
		if (!coef[k]) {
			lin = 0;
			snprintf(nonlin, sizeof nonlin, "not differentiable in %s",
			    vname(fn, ys[k]));
			break;
		}
		if (node_slots(coef[k]) & ymask) {
			lin = 0;
			if (!nonlin[0])
				snprintf(nonlin, sizeof nonlin, "d/d%s = %s depends on "
				    "%s", vname(fn, ys[k]), node_str(coef[k]),
				    sym_str(dep));
		}
		if (ts >= 0 && has_slot(coef[k], ts))
			cc = 0;
	}
	for (k = 0; k <= n; k++)
		if (ys[k] >= 0)
			map[ys[k]] = KR(0);
	F0 = subst(F, map, fn->np);
	if (lin) {
		double v;
		int hom = num_of(F0, &v) && v == 0;
		struct sbuf sb = {0};
		if (!hom && !is_const(F0)) {
			/* numerically zero? */
			struct func t = *fn;
			struct ev e2;
			double pt[MAXP] = {0}, y;
			int z = 1;
			t.body = F0;
			ev_open(&e2, &t, F0, 0, 0);
			for (j = 0; j < 20 && z; j++) {
				for (k = 0; k < fn->np; k++)
					pt[k] = urand(-2, 2);
				if (evm(&e2, pt, &y) && fabs(y) > 1e-12)
					z = 0;
			}
			ev_close(&e2);
			hom = z;
		}
		printf("  linear: every %s-term appears to the first power [proved: "
		    "the partial derivatives in %s, %s', ... do not involve %s]\n",
		    sym_str(dep), sym_str(dep), sym_str(dep), sym_str(dep));
		printf("  %s", hom ? "homogeneous" : "inhomogeneous");
		if (!hom)
			printf(": forcing term %s", node_str(mk_un(O_NEG, F0)));
		printf("\n");
		for (k = n; k >= 0; k--)
			sb_printf(&sb, "%s(%s)%s", sb.n ? " + " : "", node_str(coef[k]),
			    k == 0 ? "" : k == 1 ? " r" : rk_text(k));
		if (cc && nother == 0) {
			double cv[16];
			struct root r[16];
			int nr, ok = 1;
			struct sbuf bs = {0};
			for (k = 0; k <= n; k++)
				if (!num_of(coef[k], &cv[k]))
					ok = 0;
			printf("  constant coefficients%s\n", ts < 0 ?
			    " (and autonomous)" : "");
			if (ok && n <= 12) {
				struct rpoly P;
				P.deg = n;
				for (k = 0; k <= n; k++)
					P.c[k] = cv[k];
				nr = rpoly_roots(&P, r);
				printf("  characteristic equation: %s = 0, roots",
				    node_str(rpoly_node(&P, mk_var(0, sym_cstr("r"),
				    TY_REAL))));
				for (k = 0; k < nr; k++)
					printf("%s %s%s", k ? "," : "", root_str(&r[k]),
					    r[k].mult > 1 ? mult_text(r[k].mult) : "");
				printf("\n");
				for (k = 0; k < nr; k++) {
					int m;
					for (m = 0; m < r[k].mult; m++) {
						sb_printf(&bs, "%s%s", bs.n ? ", " : "",
						    basis_term(r[k].re, r[k].im, 1, m, tn));
						if (r[k].im != 0)
							sb_printf(&bs, ", %s", basis_term(r[k].re,
							    r[k].im, 0, m, tn));
					}
				}
				printf("  homogeneous solutions: %s(%s) = a combination of "
				    "%s\n", sym_str(dep), tn, bs.p);
				free(bs.p);
			}
		} else if (!cc)
			printf("  variable coefficients (they depend on %s)\n", tn);
		else
			printf("  constant coefficients with parameters\n");
		free(sb.p);
	} else
		printf("  nonlinear: %s\n", nonlin);
	printf("  %s\n", ts < 0 || !has_slot(F, ts) ? "autonomous: the "
	    "independent variable does not appear explicitly" : "not "
	    "autonomous: it depends on the independent variable explicitly");
	/* first order: y' = g(t, y) */
	if (n == 1 && coef[1] && !(node_slots(coef[1]) & (1ULL << ys[1]))) {
		nid m1[MAXP] = {0}, M, g;
		struct func t = *fn;
		double pt[MAXP] = {0};
		int sep = 1, ex = 1, hom0 = 1, cmp = 0;
		m1[ys[1]] = KR(0);
		M = subst(F, m1, fn->np);
		g = tidy(DIV(mk_un(O_NEG, M), coef[1]));
		t.body = g;
		ev_open(&e, &t, g, 0, 0);
		for (j = 0; j < 40 && ys[0] >= 0 && ts >= 0; j++) {
			double t1 = urand(0.2, 2), t2 = urand(0.2, 2), y1 = urand(0.2, 2),
			    y2 = urand(0.2, 2), a, b, c2, d2;
			memset(pt, 0, sizeof pt);
			pt[ts] = t1, pt[ys[0]] = y1;
			if (!evm(&e, pt, &a))
				continue;
			pt[ts] = t2, pt[ys[0]] = y2;
			if (!evm(&e, pt, &b))
				continue;
			pt[ts] = t1, pt[ys[0]] = y2;
			if (!evm(&e, pt, &c2))
				continue;
			pt[ts] = t2, pt[ys[0]] = y1;
			if (!evm(&e, pt, &d2))
				continue;
			cmp++;
			if (!close_to(a * b, c2 * d2, 1e-8) && fabs(a * b - c2 * d2) >
			    1e-12)
				sep = 0;
			pt[ts] = 2 * t1, pt[ys[0]] = 2 * y1;
			if (!evm(&e, pt, &b) || (!close_to(a, b, 1e-8) &&
			    fabs(a - b) > 1e-12))
				hom0 = 0;
		}
		ev_close(&e);
		printf("  first order: %s' = %s\n", sym_str(dep), node_str(g));
		if (ts < 0 || ys[0] < 0 || cmp == 0)
			printf("  separable: yes (%s)\n", ts < 0 ? "autonomous" :
			    "no y");
		else {
			printf("  separable: %s [numeric, 40 point pairs: g(t1, y1) "
			    "g(t2, y2) = g(t1, y2) g(t2, y1)]\n", sep ? "yes, g = p(t) "
			    "q(y)" : "no");
			if (hom0)
				printf("  homogeneous of degree 0: g(k t, k y) = g(t, y) "
				    "(substitute v = y/t) [numeric]\n");
			/* exact: M + N y' = 0 with dM/dy = dN/dt */
			{
				nid My = try_diff(M, ys[0]), Nt = try_diff(coef[1], ts);
				if (My && Nt) {
					struct func t2 = *fn;
					struct ev e2;
					nid dd = SUB(My, Nt);
					double y;
					t2.body = dd;
					ev_open(&e2, &t2, dd, 0, 0);
					for (j = 0; j < 30 && ex; j++) {
						for (k = 0; k < fn->np; k++)
							pt[k] = urand(0.2, 2);
						if (evm(&e2, pt, &y) && fabs(y) > 1e-9)
							ex = 0;
					}
					ev_close(&e2);
					printf("  exact (M + N %s' = 0 with M = %s, N = %s): %s "
					    "[numeric: dM/d%s - dN/d%s = %s]\n", sym_str(dep),
					    node_str(M), node_str(coef[1]), ex ? "yes" : "no",
					    sym_str(dep), tn, node_str(dd));
				}
			}
			{
				struct npoly P;
				if (!lin && npoly_of(g, ys[0], &P, 2) && P.deg == 2)
					printf("  Riccati: %s' is quadratic in %s\n",
					    sym_str(dep), sym_str(dep));
			}
		}
	}
	/* the form dsolve takes: y^(n) = rhs */
	if (n >= 1 && coef[n] && !(node_slots(coef[n]) & (1ULL << ys[n]))) {
		nid m1[MAXP] = {0}, rhs;
		struct sbuf sb = {0};
		m1[ys[n]] = KR(0);
		rhs = tidy(DIV(mk_un(O_NEG, subst(F, m1, fn->np)), coef[n]));
		sb_printf(&sb, "%s", sym_str(dep));
		for (k = 0; k < n; k++)
			sb_putc(&sb, '\'');
		printf("  solved for the highest derivative: %s = %s  (then "
		    "initial values and: dsolve %s)\n", sb.p, node_str(rhs),
		    sym_str(dep));
		free(sb.p);
	}
}

/* the first '=' that is not part of ==, <=, >=, != outside brackets */
static char *
top_assign(char *s)
{
	int depth = 0;
	char *q;

	for (q = s; *q; q++) {
		if (*q == '(' || *q == '[' || *q == '{')
			depth++;
		else if (*q == ')' || *q == ']' || *q == '}')
			depth--;
		else if (*q == '=' && depth == 0 && q[1] != '=' &&
		    (q == s || !strchr("<>!=", q[-1])))
			return q;
	}
	return NULL;
}

static void
trim(char *s)
{
	size_t n = strlen(s), i = 0;

	while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '?' ||
	    s[n - 1] == '.'))
		s[--n] = 0;
	while (s[i] == ' ' || s[i] == '\t')
		i++;
	memmove(s, s + i, n - i + 1);
}

static void
cmd_identify(struct P *p)
{
	char buf[1024], *eq, *rhs = NULL;
	struct func fn, *f;
	nid L, R = 0;
	int j, prim = 0;
	ty_t ty = TY_REAL;
	const char *q;

	if (!cmdp_rest(p, buf, sizeof buf, NULL))
		fail("identify: what? (an expression, an equation lhs = rhs, a "
		    "function's name or an ODE's unknown)");
	trim(buf);
	/* "identify equation x^2 + y^2 = 1": a noun before the expression */
	{
		static const char *const nouns[] = {"the equation ", "equation ",
			"the function ", "function ", "the expression ", "expression ",
			"the curve ", "curve ", "the ode ", "ode ", NULL};
		size_t ln;
		for (j = 0; nouns[j]; j++) {
			ln = strlen(nouns[j]);
			if (!strncmp(buf, nouns[j], ln)) {
				memmove(buf, buf + ln, strlen(buf + ln) + 1);
				trim(buf);
				break;
			}
		}
	}
	/* a defined function or an ODE's unknown, by name */
	for (q = buf; *q && (isalnum((unsigned char)*q) || *q == '_'); q++)
		;
	if (*q == 0) {
		uint32_t nm = sym_cstr(buf);
		int ord;
		const char *r;
		if ((f = func_find(nm)) != NULL) {
			notes_flush();
			printf("identify %s:\n  ", buf);
			cmd_show(f);
			identify_func(f);
			return;
		}
		if ((r = ode_rhs(nm, &ord)) != NULL) {
			size_t n = strlen(buf);
			for (j = 0; j < ord && n + 2 < sizeof buf; j++)
				buf[n++] = '\'';
			snprintf(buf + n, sizeof buf - n, " = %s", r);
		}
	}
	eq = top_assign(buf);
	if (eq) {
		*eq = 0;
		rhs = eq + 1;
		trim(buf);
		trim(rhs);
	}
	for (q = buf; *q; q++)
		if (*q == '&' || *q == '%' || *q == '~' || !strncmp(q, "<<", 2) ||
		    !strncmp(q, ">>", 2))
			ty = TY_INT;
	memset(&fn, 0, sizeof fn);
	fn.name = sym_cstr("_");
	fn.anon = 1;
	L = calc_parse_text(buf, &fn, ty);
	if (rhs)
		R = calc_parse_text(rhs, &fn, ty);
	notes_flush();
	for (j = 0; j < fn.np; j++)
		if (strchr(vname(&fn, j), '\''))
			prim = 1;
	printf("identify %s%s%s:\n", buf, rhs ? " = " : "", rhs ? rhs : "");
	rep_begin("identify");
	rep_strf("input", "%s%s%s", buf, rhs ? " = " : "", rhs ? rhs : "");
	rep_text_list("properties");
	if (prim) {
		for (j = 0; j < fn.np; j++)
			fn.p[j].ty = TY_REAL;
		identify_ode(&fn, rhs ? SUB(L, R) : L);
		return;
	}
	if (rhs) {
		identify_eq(&fn, SUB(L, R));
		return;
	}
	fn.body = L;
	f = func_new(fn.name);
	*f = fn;
	identify_func(f);
}

/* ---- English: equations and ODEs go straight to identify ---- */

static int
in_words(const char *w, const char *const *list)
{
	int k;

	for (k = 0; list[k]; k++)
		if (!strcmp(list[k], w))
			return 1;
	return 0;
}

const char *
props_nlp_raw(const char *sent)
{
	static const char *const verbs[] = {"identify", "classify",
		"characterize", "characterise", "categorize", "categorise",
		"describe", "analyze", "analyse", "recognize", "recognise", NULL};
	static const char *const lead[] = {"identify", "classify",
		"characterize", "characterise", "categorize", "categorise",
		"describe", "analyze", "analyse", "recognize", "recognise", "what",
		"which", "whats", "what's", "kind", "type", "sort", "class",
		"category", "of", "is", "are", "does", "do", "the", "a", "an",
		"this", "these", "equation", "equations", "curve", "conic", "ode",
		"differential", "function", "graph", "please", "can", "could",
		"you", "tell", "me", "us", "give", "list", "properties", "property",
		"characteristics", "find", "determine", "nature", "and", "an", NULL};
	static const char *const trail[] = {"have", "has", "represent",
		"represents", "describe", "describes", "a", "an", "circle",
		"ellipse", "parabola", "hyperbola", "line", "conic", "linear",
		"nonlinear", "separable", "exact", "homogeneous", "autonomous",
		"ode", "equation", "is", "it", "of", "kind", "type", "what",
		"possess", "possesses", "satisfy", "satisfies", NULL};
	static const char *const nouns[] = {"kind", "type", "sort", "class",
		"category", "properties", "property", "characteristics", "nature",
		"circle", "ellipse", "parabola", "hyperbola", "conic", "linear",
		"separable", "exact", "homogeneous", "autonomous", NULL};
	static char out[1100];
	char buf[1024], w[40], first[40] = "";
	const char *s = sent, *start, *q;
	size_t n, k;
	int trig = 0, math = 0, words = 0;

	while (*s == ' ' || *s == '\t' || *s == '?')
		s++;
	/* leading English words */
	for (;;) {
		const char *t = s;
		k = 0;
		while (*t && (isalpha((unsigned char)*t) || (*t == '\'' &&
		    isalpha((unsigned char)t[1]))) && k + 1 < sizeof w)
			w[k++] = (char)tolower((unsigned char)*t++);
		w[k] = 0;
		if (k == 0 || (*t && *t != ' ' && *t != ',' && *t != ':') ||
		    !in_words(w, lead))
			break;
		if (!words++)
			snprintf(first, sizeof first, "%s", w);
		if (in_words(w, verbs) || in_words(w, nouns))
			trig = 1;
		s = t;
		while (*s == ' ' || *s == ',' || *s == ':')
			s++;
	}
	if (!words)
		return NULL;
	start = s;
	snprintf(buf, sizeof buf, "%s", start);
	/* trailing English words */
	for (;;) {
		n = strlen(buf);
		while (n > 0 && (buf[n - 1] == ' ' || buf[n - 1] == '?' ||
		    buf[n - 1] == '.' || buf[n - 1] == ','))
			buf[--n] = 0;
		k = n;
		while (k > 0 && isalpha((unsigned char)buf[k - 1]))
			k--;
		if (k == n || (k > 0 && buf[k - 1] != ' '))
			break;
		snprintf(w, sizeof w, "%s", buf + k);
		for (q = w; *q; q++)
			;
		{
			char *c;
			for (c = w; *c; c++)
				*c = (char)tolower((unsigned char)*c);
		}
		if (!in_words(w, trail))
			break;
		if (in_words(w, nouns))
			trig = 1;
		buf[k] = 0;
	}
	if (!trig)
		return NULL;
	/* an equation or a derivative: y' (not a contraction) */
	if (top_assign(buf))
		math = 1;
	for (q = buf; *q; q++)
		if (*q == '\'' && q > buf && isalpha((unsigned char)q[-1]) &&
		    !isalpha((unsigned char)q[1]))
			math = 1;
	if (!math || !buf[0])
		return NULL;
	snprintf(out, sizeof out, "identify %s", buf);
	return out;
}
