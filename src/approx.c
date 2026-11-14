/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * approx: cheap polynomial and rational approximations of a function on
 * an interval, measured as they will run.
 *
 *	approx sin(x) in -pi/2..pi/2 [options]
 *
 * The pipeline for each candidate:
 *
 *   1. shape: odd/even symmetry and zeros are detected numerically, so
 *	sin is fitted as x*P(x^2) on [0, pi/2] and log on [1, 2] as
 *	(x-1)*P(x-1); exact points (keep) are folded into the fit;
 *   2. fit in real arithmetic (remez.c): Chebyshev interpolation and
 *	economisation as starting points, then the Remez exchange for
 *	weighted minimax, polynomial or rational;
 *   3. machine form: coefficients rounded to f32/f64, optionally
 *	re-optimised (round one coefficient, refit the rest, then an ulp
 *	walk on each coefficient), evaluated in Horner or Estrin form,
 *	with or without fma;
 *   4. the candidate is written as ucas source, parsed back, priced under
 *	the active cost model, and its error measured with ucas's own IEEE
 *	evaluator against a long double reference.
 *
 * Range reductions and identities (quadrants with Cody-Waite constants,
 * exp by 2^k, log by exponent split, sqrt by halving the exponent, atan
 * by reciprocal, sin/cos for tan, sec, csc, cot, half angles) are
 * offered only after the target is checked numerically to be that
 * function, and each one is measured end to end like any other
 * candidate.
 */
#include "approx.h"
#include "numsearch.h"
#include "stoch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI_L 3.14159265358979323846264338327950288L
#define LN2_L 0.693147180559945309417232121458176568L

/* ---- options and run state ---- */

struct aopt {
	int fmt;		/* 0 real, 32, 64 */
	int dmin, dmax;		/* degree in x, 0 = automatic */
	double tol_err, tol_ulp;
	int metric;		/* -1 auto, 0 absolute, 1 relative */
	int sym;		/* -1 auto, 0 none, 1 odd, 2 even */
	int nkeep, keep_ends;
	double keep[AMAXK];
	int hasc;
	double c;
	int fma, refine, list, emitc, npts;
	int rational, reduce, split;
	char name[24];
	int haswt;
	int method;		/* fit that seeds the machine forms: FM_* */
	int keep_crit;		/* exact at the critical points too */
	int method_given;	/* `method` named: report that method's fit */
	int compare;		/* ...but still search every other shape */
	int ends_free;		/* do not ask about special endpoints */
	struct nsopt search;	/* a stochastic search after the ulp walk */
};

struct cand {
	char kind[16];
	char shape[40];
	char form[16];
	int deg, qdeg;
	char *text;		/* body and where clause */
	double eabs, erel, eulp;
	double rerr;		/* fit error in real arithmetic, -1 n/a */
	double naive;		/* metric with naively rounded coefficients */
	double freeerr;		/* fit error without the exact points */
	double cost;
	double xw;		/* where the primary metric is worst */
	int keepok;		/* -1 n/a */
	int bad;		/* NaN/inf where the target is finite */
	int front, meets, useless;
	/* numsearch.c ran from this candidate's walked coefficients */
	int skind, skept, snsx;
	long sevals;
	double sbefore, safter;	/* on the search's own sample */
	double smw, sms;	/* the walk's and the search's, on the table's */
};

static struct {
	struct aopt o;
	struct func f, wf;
	struct lref ref, wref;
	char fname[128];
	ld lo, hi;
	int w, prec, emin;
	int rel, odd, even, fn;
	ld c;
	int k0, step;
	ld xa, xb;		/* fitting interval after folding */
	int nkeep;
	ld keep[AMAXK];
	uint32_t ns;
	double *xs;
	ld *ys;
	int exhaustive;
	uint64_t total;
	ld fmax;
	struct cand *cd;
	int nc, capc, shown, valid, stalled;
	int seq;
} A;

static const char *const optwords[] = {
	"in", "degree", "error", "ulp", "as", "relative", "absolute", "odd",
	"even", "nosym", "keep", "weight", "at", "fma", "nofma", "norefine",
	"list", "name", "points", "norational", "noreduce", "nosplit", "method",
	"compare", "free", "search", "budget", "temp", "pop", "sigma", "seed",
	NULL
};

/* ---- small helpers ---- */

static double
rnd(int w, ld v)
{
	if (w == 32)
		return (double)(float)v;
	return (double)v;
}

static ld
ulp_of(ld y)
{
	int e;

	if (A.w == 0)
		return 0;
	y = fabsl(y);
	if (y == 0)
		return ldexpl(1, A.emin - (A.prec - 1));
	e = ilogbl(y);
	if (e < A.emin)
		e = A.emin;
	return ldexpl(1, e - (A.prec - 1));
}

static void
kfmt(char *buf, size_t n, double v, int w)
{
	snprintf(buf, n, w == 32 ? "%.9g" : "%.17g", v);
	if (!strpbrk(buf, ".eni"))
		strncat(buf, ".0", n - strlen(buf) - 1);
}

static const char *
fmt_name(int w)
{
	return w == 32 ? "f32" : w == 64 ? "f64" : "real";
}

/* the primary metric of a candidate */
static double
cmetric(const struct cand *c)
{
	if (c->bad)
		return INFINITY;
	if (A.w && A.rel)
		return c->eulp;
	return A.rel ? c->erel : c->eabs;
}

static const char *
metric_name(void)
{
	if (A.w && A.rel)
		return "max ulp";
	return A.rel ? "max rel" : "max abs";
}

/* ---- a straight-line program for the polynomial cores ---- */

enum { I_X, I_K, I_ADD, I_SUB, I_MUL, I_DIV, I_FMA };
#define IRMAX 192

struct ir {
	int n, root, ncoef;
	uint8_t op[IRMAX];
	int16_t a[IRMAX], b[IRMAX], c[IRMAX], ki[IRMAX];
	double kv[IRMAX];
	char name[IRMAX][8];
	double coef[2 * AMAXC];
	int w;
};

static int
ir_add(struct ir *r, int op, int a, int b, int c)
{
	if (r->n == IRMAX)
		fail("approx: program too long");
	r->op[r->n] = (uint8_t)op;
	r->a[r->n] = (int16_t)a;
	r->b[r->n] = (int16_t)b;
	r->c[r->n] = (int16_t)c;
	r->ki[r->n] = -1;
	r->kv[r->n] = 0;
	r->name[r->n][0] = 0;
	return r->n++;
}

static int
ir_k(struct ir *r, double v)
{
	int i = ir_add(r, I_K, 0, 0, 0);

	r->kv[i] = v;
	return i;
}

static int
ir_slot(struct ir *r, int slot)
{
	int i = ir_add(r, I_K, 0, 0, 0);

	r->ki[i] = (int16_t)slot;
	return i;
}

static ld
ir_eval(const struct ir *r, ld x)
{
	ld v[IRMAX], a, b;
	int i, w = r->w;

	for (i = 0; i < r->n; i++) {
		a = v[r->a[i]];
		b = v[r->b[i]];
		switch (r->op[i]) {
		case I_X: v[i] = x; break;
		case I_K: v[i] = r->ki[i] >= 0 ? r->coef[r->ki[i]] : r->kv[i];
			break;
		case I_ADD: v[i] = w ? rnd(w, (double)a + (double)b) : a + b;
			break;
		case I_SUB: v[i] = w ? rnd(w, (double)a - (double)b) : a - b;
			break;
		case I_MUL: v[i] = w ? rnd(w, (double)a * (double)b) : a * b;
			break;
		case I_DIV: v[i] = w ? rnd(w, (double)a / (double)b) : a / b;
			break;
		case I_FMA:
			if (w == 32)	/* correctly rounded (eval.c) */
				v[i] = fma32((double)a, (double)b,
				    (double)v[r->c[i]]);
			else if (w == 64)
				v[i] = fma64((double)a, (double)b,
				    (double)v[r->c[i]]);
			else
				v[i] = a * b + v[r->c[i]];
			break;
		}
	}
	return v[r->root];
}

static double
ir_kval(const struct ir *r, int i)
{
	return r->ki[i] >= 0 ? r->coef[r->ki[i]] : r->kv[i];
}

static void
ir_pr(const struct ir *r, int i, struct sbuf *b, int def)
{
	char k[48], *s;
	struct sbuf t = {0};
	int op = r->op[i];

	if (!def && r->name[i][0]) {
		sb_puts(b, r->name[i]);
		return;
	}
	switch (op) {
	case I_X:
		sb_puts(b, r->name[i][0] ? r->name[i] : "x");
		break;
	case I_K:
		kfmt(k, sizeof k, ir_kval(r, i), r->w);
		sb_puts(b, k);
		break;
	case I_ADD:
		/* a + b, written as a subtraction when one side is negative */
		if (r->op[r->a[i]] == I_K && ir_kval(r, r->a[i]) < 0) {
			ir_pr(r, r->b[i], b, 0);
			kfmt(k, sizeof k, -ir_kval(r, r->a[i]), r->w);
			sb_printf(b, " - %s", k);
			break;
		}
		ir_pr(r, r->a[i], b, 0);
		ir_pr(r, r->b[i], &t, 0);
		s = t.p ? t.p : "";
		if (s[0] == '-')
			sb_printf(b, " - %s", s + 1);
		else
			sb_printf(b, " + %s", s);
		free(t.p);
		break;
	case I_SUB:
		ir_pr(r, r->a[i], b, 0);
		sb_puts(b, " - ");
		if (!r->name[r->b[i]][0] && (r->op[r->b[i]] == I_ADD ||
		    r->op[r->b[i]] == I_SUB)) {
			sb_putc(b, '(');
			ir_pr(r, r->b[i], b, 0);
			sb_putc(b, ')');
		} else
			ir_pr(r, r->b[i], b, 0);
		break;
	case I_MUL:
	case I_DIV: {
		int j, side[2] = {r->a[i], r->b[i]};
		for (j = 0; j < 2; j++) {
			int q = side[j], par;
			par = !r->name[q][0] && (r->op[q] == I_ADD ||
			    r->op[q] == I_SUB || (j == 1 && (r->op[q] == I_MUL ||
			    r->op[q] == I_DIV) && op == I_DIV) ||
			    (j == 1 && r->op[q] == I_K && ir_kval(r, q) < 0));
			if (j)
				sb_puts(b, op == I_MUL ? "*" : "/");
			if (par)
				sb_putc(b, '(');
			ir_pr(r, q, b, 0);
			if (par)
				sb_putc(b, ')');
		}
		break;
	}
	case I_FMA:
		sb_puts(b, "fma(");
		ir_pr(r, r->a[i], b, 0);
		sb_puts(b, ", ");
		ir_pr(r, r->b[i], b, 0);
		sb_puts(b, ", ");
		ir_pr(r, r->c[i], b, 0);
		sb_putc(b, ')');
		break;
	}
}

/* where-clause bindings, deduplicated */
struct blist {
	int n;
	char *name[32], *def[32];
};

static void
bl_add(struct blist *l, const char *name, const char *def)
{
	int i;

	for (i = 0; i < l->n; i++)
		if (!strcmp(l->name[i], name))
			return;
	if (l->n == 32)
		fail("approx: too many bindings");
	l->name[l->n] = strdup(name);
	l->def[l->n] = strdup(def);
	l->n++;
}

static void
bl_free(struct blist *l)
{
	int i;

	for (i = 0; i < l->n; i++) {
		free(l->name[i]);
		free(l->def[i]);
	}
	l->n = 0;
}

/* the body text of r; its named nodes go to the binding list */
static void
ir_text(const struct ir *r, struct sbuf *b, struct blist *l)
{
	int i;

	for (i = 0; i < r->n; i++)
		if (r->name[i][0] && r->op[i] != I_X) {
			struct sbuf d = {0};
			ir_pr(r, i, &d, 1);
			bl_add(l, r->name[i], sb_str(&d));
			free(d.p);
		}
	ir_pr(r, r->root, b, 0);
}

/* sum coef[first + j] v^j, j < n */
static int
ir_poly(struct ir *r, int v, int first, int n, int estrin, int fma_)
{
	int p[AMAXC], np, i, acc, pw, lvl = 0;

	if (n == 1)
		return ir_slot(r, first);
	if (!estrin || n <= 3) {
		acc = ir_slot(r, first + n - 1);
		for (i = n - 2; i >= 0; i--) {
			if (fma_)
				acc = ir_add(r, I_FMA, acc, v, ir_slot(r, first + i));
			else if (r->op[acc] == I_K)
				acc = ir_add(r, I_ADD, ir_slot(r, first + i),
				    ir_add(r, I_MUL, acc, v, 0), 0);
			else
				acc = ir_add(r, I_ADD, ir_slot(r, first + i),
				    ir_add(r, I_MUL, v, acc, 0), 0);
		}
		return acc;
	}
	np = 0;
	for (i = 0; i < n; i += 2) {
		if (i + 1 < n) {
			if (fma_)
				p[np++] = ir_add(r, I_FMA, ir_slot(r, first + i + 1),
				    v, ir_slot(r, first + i));
			else
				p[np++] = ir_add(r, I_ADD, ir_slot(r, first + i),
				    ir_add(r, I_MUL, ir_slot(r, first + i + 1), v,
				    0), 0);
		} else
			p[np++] = ir_slot(r, first + i);
	}
	pw = v;
	while (np > 1) {
		int q = 0;
		pw = ir_add(r, I_MUL, pw, pw, 0);
		snprintf(r->name[pw], sizeof r->name[pw], "%s%d",
		    r->name[v][0] ? r->name[v] : "v", 2 << lvl);
		lvl++;
		for (i = 0; i < np; i += 2) {
			if (i + 1 < np) {
				if (fma_)
					p[q++] = ir_add(r, I_FMA, p[i + 1], pw, p[i]);
				else
					p[q++] = ir_add(r, I_ADD, p[i],
					    ir_add(r, I_MUL, pw, p[i + 1], 0), 0);
			} else
				p[q++] = p[i];
		}
		np = q;
	}
	return p[0];
}

/*
 * A core: t^k0 N(v) / D(v) on the input variable var, t = var - c,
 * v = t^step.  Coefficient slots: num then den (den[0] = 1 is implicit).
 */
struct core {
	struct fitprob pb;
	struct fitres fr;
	int w, estrin, fma_;
	const char *var;
	char sname[8];
	struct ir ir;
	int lead1;
	double naive;		/* objective before refinement */
	double ncoef[2 * AMAXC];	/* the naively rounded coefficients */
	double obj;
	/* search sample */
	int nsx;
	double *sx;
	ld *sy;
	int objrel;		/* objective: ulp (1) or abs error (0) */
};

static void
core_build_ir(struct core *k)
{
	struct ir *r = &k->ir;
	const struct fitprob *pb = &k->pb;
	int x, t, v, nn = k->fr.nn, nd = k->fr.nd, N1, res, i;
	double c0;

	r->n = 0;
	r->w = k->w;
	x = ir_add(r, I_X, 0, 0, 0);
	snprintf(r->name[x], sizeof r->name[x], "%s", k->var);
	t = x;
	if (pb->c != 0) {
		t = ir_add(r, I_SUB, x, ir_k(r, rnd(k->w, pb->c)), 0);
		snprintf(r->name[t], sizeof r->name[t], "t");
	}
	v = t;
	if (pb->step == 2) {
		v = ir_add(r, I_MUL, t, t, 0);
		snprintf(r->name[v], sizeof r->name[v], "%s", k->sname);
	}
	c0 = r->coef[0];
	k->lead1 = pb->k0 == 1 && nd == 1 && c0 == 1 && nn > 1;
	if (k->lead1) {
		/* t + (t v) N1(v): the leading term is exact */
		int tv = ir_add(r, I_MUL, t, v, 0);
		N1 = ir_poly(r, v, 1, nn - 1, k->estrin, k->fma_);
		if (k->fma_)
			res = ir_add(r, I_FMA, tv, N1, t);
		else
			res = ir_add(r, I_ADD, t, ir_add(r, I_MUL, tv, N1, 0), 0);
	} else {
		res = ir_poly(r, v, 0, nn, k->estrin, k->fma_);
		if (nd > 1) {
			int D;
			/* the denominator's constant is 1: a fixed constant */
			r->coef[nn] = 1;
			D = ir_poly(r, v, nn, nd, 0, k->fma_);
			res = ir_add(r, I_DIV, res, D, 0);
		}
		for (i = 0; i < pb->k0; i++)
			res = ir_add(r, I_MUL, t, res, 0);
	}
	r->root = res;
	r->ncoef = nn + nd;
}

static ld
core_ref(const struct core *k, ld x)
{
	return k->pb.f(k->pb.fctx, x);
}

/* the search sample: Chebyshev-spaced values and evenly spaced keys */
static void
core_sample(struct core *k)
{
	int i, n = 0, m = 1500;
	ld a = k->pb.xa, b = k->pb.xb;

	free(k->sx);
	free(k->sy);
	k->sx = xmalloc(sizeof *k->sx * (2 * m + 8));
	k->sy = xmalloc(sizeof *k->sy * (2 * m + 8));
	for (i = 0; i < m; i++) {
		ld z = -cosl(PI_L * i / (m - 1));
		k->sx[n++] = rnd(k->w, a + (z + 1) * (b - a) / 2);
	}
	if (k->w) {
		int64_t ka = fkey(k->w, (double)a), kb = fkey(k->w, (double)b);
		for (i = 0; i < m; i++)
			k->sx[n++] = fromkey(k->w, ka + (int64_t)(((ld)kb - (ld)ka) *
			    i / (m - 1)));
	}
	for (i = 0; i < k->pb.nkeep; i++)
		k->sx[n++] = rnd(k->w, k->pb.keep[i]);
	k->nsx = 0;
	for (i = 0; i < n; i++) {
		ld y = core_ref(k, k->sx[i]);
		if (!isfinite(y) || k->sx[i] < a || k->sx[i] > b)
			continue;
		k->sx[k->nsx] = k->sx[i];
		k->sy[k->nsx++] = y;
	}
}

static double
core_obj(struct core *k)
{
	double worst = 0, e;
	int i, j;

	for (i = 0; i < k->nsx; i++) {
		ld p = ir_eval(&k->ir, k->sx[i]), y = k->sy[i];
		if (!isfinite(p))
			return INFINITY;
		if (k->objrel)
			e = (double)(fabsl(p - y) / (k->w ? ulp_of(y) :
			    fmaxl(fabsl(y), 1e-300L)));
		else
			e = (double)fabsl(p - y);
		if (e > worst)
			worst = e;
	}
	/* exact points must come out correctly rounded */
	for (j = 0; j < k->pb.nkeep && k->w; j++) {
		double xk = rnd(k->w, k->pb.keep[j]);
		ld y = core_ref(k, xk);
		if (isfinite(y) && ir_eval(&k->ir, xk) != rnd(k->w, y))
			worst += 1e6;
	}
	return worst;
}

static double
nextfl(int w, double v, int dir)
{
	if (w == 32)
		return nextafterf((float)v, dir > 0 ? INFINITY : -INFINITY);
	return nextafter(v, dir > 0 ? INFINITY : -INFINITY);
}

/* fixed lower coefficients for the sequential refit */
struct seqctx {
	const struct fitprob *pb;
	double fixed[AMAXC];
	int nfix;
};

static ld
seq_f(void *ctx, ld x)
{
	struct seqctx *s = ctx;
	const struct fitprob *pb = s->pb;
	ld t = x - pb->c, v = pb->step == 2 ? t * t : t, sum = 0, pv = 1;
	ld f = pb->f(pb->fctx, x);
	int i;

	for (i = 0; i < s->nfix; i++) {
		sum += s->fixed[i] * pv;
		pv *= v;
	}
	for (i = 0; i < pb->k0; i++)
		sum *= t;
	return f - sum;
}

static ld
seq_w(void *ctx, ld x)
{
	struct seqctx *s = ctx;
	const struct fitprob *pb = s->pb;
	ld w = 1;

	if (pb->rel) {
		ld f = fabsl(pb->f(pb->fctx, x));
		w = 1 / (f > 1e-300L ? f : 1e-300L);
	}
	if (pb->w)
		w *= pb->w(pb->wctx, x);
	return w;
}

/*
 * Round the fitted coefficients and (optionally) re-optimise them:
 * fix one coefficient at a time and refit the higher ones, then walk
 * each coefficient by a few ulps while the measured error drops.
 */
static void
core_round(struct core *k, int refine)
{
	struct fitres *fr = &k->fr;
	int nn = fr->nn, nd = fr->nd, i, j, pass, improved;
	double best;

	for (i = 0; i < nn; i++)
		k->ir.coef[i] = k->w ? rnd(k->w, fr->num[i]) : (double)fr->num[i];
	for (i = 1; i < nd; i++)
		k->ir.coef[nn + i] = k->w ? rnd(k->w, fr->den[i]) :
		    (double)fr->den[i];
	core_build_ir(k);
	if (!k->w) {
		k->naive = k->obj = core_obj(k);
		return;
	}
	memcpy(k->ncoef, k->ir.coef, sizeof k->ncoef);
	core_sample(k);
	k->naive = core_obj(k);
	k->obj = k->naive;
	if (!refine)
		return;
	{
		/* refining by ulps only matters near the rounding floor */
		ld fm = 0, u = ldexpl(1, k->w == 32 ? -24 : -53);
		for (i = 0; i < k->nsx; i++)
			fm = fmaxl(fm, fabsl(k->sy[i]));
		if (fr->err > 64 * u * (k->pb.rel ? 1 : fm))
			return;
	}
	if (nd == 1 && nn > 1) {
		/* sequential: fix c0..cj, refit the rest */
		struct seqctx s;
		struct fitprob sp;
		struct fitres sr;
		double save[AMAXC], o;
		memcpy(save, k->ir.coef, sizeof save);
		memset(&s, 0, sizeof s);
		s.pb = &k->pb;
		/* a leading coefficient near 1 is pinned to 1 (exact term) */
		if (fabs(k->ir.coef[0] - 1) < 1.0 / 256 && k->pb.nkeep == 0)
			k->ir.coef[0] = 1;
		for (j = 0; j + 1 < nn; j++) {
			s.fixed[j] = k->ir.coef[j];
			s.nfix = j + 1;
			sp = k->pb;
			sp.f = seq_f;
			sp.fctx = &s;
			sp.w = seq_w;
			sp.wctx = &s;
			sp.rel = 0;
			sp.k0 = k->pb.k0 + k->pb.step * (j + 1);
			sp.nt = nn - j - 1;
			sp.method = FM_REMEZ;
			sp.nd = 0;
			if (!fit_run(&sp, &sr))
				break;
			for (i = 0; i < sr.nn && j + 1 + i < nn; i++)
				k->ir.coef[j + 1 + i] = rnd(k->w, sr.num[i]);
		}
		core_build_ir(k);
		o = core_obj(k);
		if (o > k->obj) {
			memcpy(k->ir.coef, save, sizeof save);
			core_build_ir(k);
		} else
			k->obj = o;
	}
	/* ulp walk */
	best = k->obj;
	for (pass = 0; pass < 6; pass++) {
		improved = 0;
		for (i = 0; i < nn + nd; i++) {
			int d, step;
			if (i == nn)	/* den[0] = 1 stays */
				continue;
			if (k->lead1 && i == 0)
				continue;
			for (d = -1; d <= 1; d += 2)
				for (step = 1; step <= 4; step *= 2) {
					double old = k->ir.coef[i], nv = old, o;
					int s;
					for (s = 0; s < step; s++)
						nv = nextfl(k->w, nv, d);
					k->ir.coef[i] = nv;
					o = core_obj(k);
					if (o < best) {
						best = o;
						improved = 1;
					} else
						k->ir.coef[i] = old;
				}
		}
		if (!improved)
			break;
	}
	k->obj = best;
}

/* numsearch.c's objective: the coefficients a whole number of ulps from
 * the walk's, measured on the core's sample */
struct sctx {
	struct core *k;
	double base[2 * AMAXC];
	int map[2 * AMAXC], n;
};

static double
search_obj(void *ctx, const int *off)
{
	struct sctx *s = ctx;
	int i, j;

	for (i = 0; i < s->n; i++) {
		double v = s->base[s->map[i]];
		int d = off[i] > 0 ? 1 : -1, c = off[i] > 0 ? off[i] : -off[i];
		for (j = 0; j < c; j++)
			v = nextfl(s->k->w, v, d);
		s->k->ir.coef[s->map[i]] = v;
	}
	return core_obj(s->k);
}

/* the search the options name, from the core's coefficients; 1 when it
 * found better ones on the sample (then they are the core's) */
static int
core_search(struct core *k, const struct nsopt *o, struct nsres *res)
{
	struct sctx s;
	int best[2 * AMAXC], nn = k->fr.nn, nd = k->fr.nd, i;

	memset(&s, 0, sizeof s);
	s.k = k;
	memcpy(s.base, k->ir.coef, sizeof s.base);
	for (i = 0; i < nn + nd; i++) {
		if (i == nn || (k->lead1 && i == 0))
			continue;	/* den[0] = 1 and a pinned 1 stay */
		s.map[s.n++] = i;
	}
	memset(best, 0, sizeof best);
	ns_run(s.n, search_obj, &s, o, best, res);
	search_obj(&s, best);
	core_build_ir(k);
	return res->after < res->before;
}

/* several cores searched, each on its own sample: the evaluations
 * summed, the objective the largest of theirs before and after; 1 when
 * one of them moved */
static int
cores_search(struct core **ks, int nk, struct nsres *sr, int *nsx)
{
	int i, moved = 0;

	memset(sr, 0, sizeof *sr);
	*nsx = 0;
	for (i = 0; i < nk; i++) {
		struct nsres r;
		if (ks[i]->nsx <= 0)
			continue;
		moved |= core_search(ks[i], &A.o.search, &r);
		sr->evals += r.evals;
		sr->accepted += r.accepted;
		if (r.before > sr->before)
			sr->before = r.before;
		if (r.after > sr->after)
			sr->after = r.after;
		*nsx += ks[i]->nsx;
	}
	return moved;
}

static void
core_free(struct core *k)
{
	free(k->sx);
	free(k->sy);
	k->sx = NULL;
	k->sy = NULL;
}

/* ---- defining, pricing and measuring a candidate ---- */

/* parse "name(x: ty in lo..hi) = body" with the regular parser */
static struct func *
define(const char *name, const char *body)
{
	jmp_buf saved;
	struct toks *st = diag_toks;
	const char *sf = diag_file;
	uint32_t stok = diag_tok;
	struct sbuf b = {0};
	char lo[48], hi[48];
	struct func *f;
	uint32_t sym = sym_cstr(name);

	kfmt(lo, sizeof lo, (double)A.lo, 64);
	kfmt(hi, sizeof hi, (double)A.hi, 64);
	sb_printf(&b, "%s(x: %s in %s..%s) = %s\n", name, A.w ?
	    fmt_name(A.w) : "real", lo, hi, body);
	if ((f = func_find(sym)) != NULL)
		f->body = 0;
	memcpy(saved, fail_jmp, sizeof saved);
	run_source(b.p, "<approx>", 0);
	memcpy(fail_jmp, saved, sizeof saved);
	diag_toks = st;
	diag_file = sf;
	diag_tok = stok;
	free(b.p);
	f = func_find(sym);
	if (f == NULL || f->body == 0)
		return NULL;
	return f;
}

static double
price(nid body)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0, j;
	double sum = 0;
	struct model *m = model_active();
	int cls;

	st[sp++] = body;
	mark[body] = 1;
	while (sp) {
		g = st[--sp];
		sum += node_cost(m, g, &cls);
		if (N.op[g] == O_CONST || N.op[g] == O_VAR ||
		    N.op[g] == O_UNDEF)
			continue;
		if (N.op[g] == O_TABLE) {
			for (j = 0; j < N.b[g]; j++)
				if (!mark[N.lst[N.a[g] + j]]) {
					mark[N.lst[N.a[g] + j]] = 1;
					st[sp++] = N.lst[N.a[g] + j];
				}
			continue;
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
	return sum;
}

/* error of the defined function against the reference, on the sample */
static void
measure(struct func *f, struct cand *c)
{
	struct plan p;
	nid roots[1];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t base, i, np, loc;
	int j;

	roots[0] = f->body;
	plan_build(&p, roots, 1, 0);
	loc = p.root[0];
	c->eabs = c->erel = c->eulp = 0;
	c->bad = 0;
	for (base = 0; base < A.ns; base += CH) {
		np = A.ns - base < CH ? A.ns - base : CH;
		for (i = 0; i < np; i++)
			in_r[i] = A.xs[base + i];
		plan_run(&p, np, in_i, in_r);
		for (i = 0; i < np; i++) {
			ld y = A.ys[base + i], v, ae, re;
			if (!isfinite(y))
				continue;
			v = COLR(&p, loc)[i];
			if ((COLF(&p, loc)[i] & F_UNDEF) || !isfinite(v)) {
				c->bad++;
				continue;
			}
			ae = fabsl(v - y);
			re = y != 0 ? ae / fabsl(y) : (ae == 0 ? 0 : INFINITY);
			if (ae > c->eabs) {
				c->eabs = (double)ae;
				if (!A.rel)
					c->xw = A.xs[base + i];
			}
			if (re > c->erel) {
				c->erel = (double)re;
				if (A.rel && !A.w)
					c->xw = A.xs[base + i];
			}
			if (A.w) {
				ld u = ae / ulp_of(y);
				if (u > c->eulp) {
					c->eulp = (double)u;
					if (A.rel)
						c->xw = A.xs[base + i];
				}
			}
		}
	}
	/* exact points: the machine result equals the rounded value */
	c->keepok = -1;
	if (A.nkeep && A.w) {
		c->keepok = 1;
		for (j = 0; j < A.nkeep; j++) {
			ld y;
			in_r[0] = rnd(A.w, A.keep[j]);
			y = lref_eval(&A.ref, in_r[0]);
			plan_run(&p, 1, in_i, in_r);
			if (isfinite(y) && COLR(&p, loc)[0] != rnd(A.w, y))
				c->keepok = 0;
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
}

static struct cand *
cand_new(void)
{
	struct cand *c;

	if (A.nc == A.capc) {
		A.capc = A.capc ? A.capc * 2 : 64;
		A.cd = xrealloc(A.cd, (size_t)A.capc * sizeof *A.cd);
	}
	c = &A.cd[A.nc++];
	memset(c, 0, sizeof *c);
	c->rerr = c->naive = c->freeerr = -1;
	c->keepok = -1;
	return c;
}

/* define, price and measure body; keep it as a candidate */
static struct cand *
submit(const char *kind, const char *shape, const char *form, int deg,
    int qdeg, const char *body)
{
	struct func *f = define("_approx", body);
	struct cand *c;

	if (f == NULL) {
		printf("  internal: candidate did not parse: %s\n", body);
		return NULL;
	}
	c = cand_new();
	snprintf(c->kind, sizeof c->kind, "%s", kind);
	snprintf(c->shape, sizeof c->shape, "%s", shape);
	snprintf(c->form, sizeof c->form, "%s", form);
	c->deg = deg;
	c->qdeg = qdeg;
	c->text = strdup(body);
	c->cost = price(f->body);
	measure(f, c);
	return c;
}

static void
compose(struct sbuf *out, const char *body, struct blist *l)
{
	int i;

	sb_puts(out, body);
	for (i = 0; i < l->n; i++)
		sb_printf(out, "%s%s = %s", i ? ", " : " where ", l->name[i],
		    l->def[i]);
}

/* ---- measurement sample ---- */

static int
cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

static void
build_sample(void)
{
	uint32_t n = 0, cap, i, m = (uint32_t)A.o.npts;
	double *xs;

	cap = m + 64 + 2 * AMAXK;
	xs = xmalloc(cap * sizeof *xs);
	A.exhaustive = 0;
	if (A.w) {
		int64_t ka = fkey(A.w, (double)A.lo), kb = fkey(A.w, (double)A.hi);
		A.total = (uint64_t)kb - (uint64_t)ka + 1;
		if (A.total <= m) {
			A.exhaustive = 1;
			for (i = 0; i < A.total; i++)
				xs[n++] = fromkey(A.w, ka + (int64_t)i);
		} else {
			uint32_t h = m / 2;
			for (i = 0; i < h; i++)
				xs[n++] = fromkey(A.w, ka + (int64_t)(((ld)kb - (ld)ka) *
				    i / (h - 1)));
			for (i = 0; i < h; i++)
				xs[n++] = rnd(A.w, A.lo + (A.hi - A.lo) * i /
				    (h - 1));
		}
	} else {
		A.total = 0;
		for (i = 0; i < m / 8; i++)
			xs[n++] = (double)(A.lo + (A.hi - A.lo) * i / (m / 8 - 1));
	}
	for (i = 0; i < (uint32_t)A.nkeep; i++)
		xs[n++] = rnd(A.w, A.keep[i]);
	if (A.lo < 0 && A.hi > 0)
		xs[n++] = 0;
	if (A.w) {
		/* the floats around each zero of f: relative error peaks there */
		enum { G = 20000, NB = 16 };
		ld px = A.lo, py = lref_eval(&A.ref, A.lo);
		uint32_t nz = 0;
		for (i = 1; i <= G && nz < 1024; i++) {
			ld x = A.lo + (A.hi - A.lo) * i / G, y = lref_eval(&A.ref, x);
			if (isfinite(y) && isfinite(py) && py != 0 && (y == 0 ||
			    (y > 0) != (py > 0))) {
				int64_t ka = fkey(A.w, (double)px), kb = fkey(A.w,
				    (double)x), km, j;
				while (kb - ka > 1) {
					km = ka + (kb - ka) / 2;
					if ((lref_eval(&A.ref, fromkey(A.w, km)) > 0) ==
					    (py > 0))
						ka = km;
					else
						kb = km;
				}
				if (n + 2 * NB + 2 > cap) {
					cap = cap * 2 + 4 * NB;
					xs = xrealloc(xs, cap * sizeof *xs);
				}
				for (j = -NB; j <= NB; j++)
					xs[n++] = fromkey(A.w, ka + j);
				nz++;
			}
			px = x;
			py = y;
		}
	}
	qsort(xs, n, sizeof *xs, cmpd);
	A.ns = 0;
	for (i = 0; i < n; i++)
		if ((A.ns == 0 || xs[i] != xs[A.ns - 1]) && xs[i] >= A.lo &&
		    xs[i] <= A.hi)
			xs[A.ns++] = xs[i];
	A.xs = xs;
	A.ys = xmalloc(A.ns * sizeof *A.ys);
	A.fmax = 0;
	for (i = 0; i < A.ns; i++) {
		A.ys[i] = lref_eval(&A.ref, A.xs[i]);
		if (isfinite(A.ys[i]) && fabsl(A.ys[i]) > A.fmax)
			A.fmax = fabsl(A.ys[i]);
	}
}

/* ---- the target as an ldfn ---- */

static ld
target_f(void *ctx, ld x)
{
	(void)ctx;
	return lref_eval(&A.ref, x);
}

static ld
weight_f(void *ctx, ld x)
{
	ld w;

	(void)ctx;
	w = lref_eval(&A.wref, x);
	return isfinite(w) ? fabsl(w) : 0;
}

/* ---- direct polynomial / rational fits ---- */

static void
base_prob(struct fitprob *pb, int nt, int nd, int method)
{
	memset(pb, 0, sizeof *pb);
	pb->f = target_f;
	if (A.o.haswt)
		pb->w = weight_f;
	pb->xa = A.xa;
	pb->xb = A.xb;
	pb->c = A.c;
	pb->k0 = A.k0;
	pb->step = A.step;
	pb->rel = A.rel;
	pb->nkeep = A.nkeep;
	memcpy(pb->keep, A.keep, sizeof pb->keep);
	pb->nt = nt;
	pb->nd = nd;
	pb->method = method;
}

static const char *
shape_str(int k0, int step, ld c, int nd)
{
	static char buf[48];
	char cs[40], t[48];

	if (c == 0)
		snprintf(t, sizeof t, "x");
	else {
		kfmt(cs, sizeof cs, rnd(A.w, c), A.w ? A.w : 64);
		snprintf(t, sizeof t, "(x%s%s)", c < 0 ? "+" : "-",
		    cs[0] == '-' ? cs + 1 : cs);
	}
	if (step == 2)
		snprintf(buf, sizeof buf, "%s%sP(x^2)%s", k0 ? "x" : "",
		    k0 ? "*" : "", nd ? "/Q(x^2)" : "");
	else
		snprintf(buf, sizeof buf, "%s%sP%s%s", k0 ? t : "",
		    k0 ? "*" : "", c == 0 ? "(x)" : "(t)", nd ? "/Q" : "");
	return buf;
}

static double
metric_of_fit(const struct fitres *r)
{
	return (double)r->err;
}

/* round a fitted core in each machine form and submit the candidates */
static void
machine_forms(struct core *k, const char *kind, const char *shape, int deg,
    int qdeg, double rerr, double freeerr)
{
	int forms[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	int i, done = 0, first;
	double coef[2 * AMAXC], naive = -1;

	k->var = "x";
	snprintf(k->sname, sizeof k->sname, "s");
	k->objrel = A.rel;
	for (i = 0; i < 4; i++) {
		struct blist l = {0};
		struct sbuf body = {0}, full = {0};
		struct cand *c;
		char fname[16];
		int estrin = forms[i][1], fm = forms[i][0];
		if (fm && (!A.o.fma || !A.w))
			continue;
		if (!fm && A.o.fma && A.w)
			continue;
		if (estrin && k->fr.nn <= 3)
			continue;
		k->estrin = estrin;
		k->fma_ = fm;
		first = !done;
		if (!done) {
			core_round(k, A.o.refine);
			memcpy(coef, k->ir.coef, sizeof coef);
			done = 1;
			if (A.w && A.o.refine && memcmp(k->ncoef, coef,
			    sizeof(double) * (k->fr.nn + k->fr.nd))) {
				/* measure the naive rounding the same way */
				memcpy(k->ir.coef, k->ncoef, sizeof coef);
				core_build_ir(k);
				ir_text(&k->ir, &body, &l);
				compose(&full, sb_str(&body), &l);
				c = submit(kind, shape, "", deg, qdeg, sb_str(&full));
				if (c) {
					naive = cmetric(c);
					free(c->text);
					A.nc--;
				}
				bl_free(&l);
				sb_clear(&body);
				sb_clear(&full);
				memcpy(k->ir.coef, coef, sizeof coef);
				core_build_ir(k);
			}
		} else {
			memcpy(k->ir.coef, coef, sizeof coef);
			core_build_ir(k);
		}
		ir_text(&k->ir, &body, &l);
		compose(&full, sb_str(&body), &l);
		snprintf(fname, sizeof fname, "%s%s", estrin ? "estrin" :
		    "horner", fm ? "+fma" : "");
		c = submit(kind, shape, fname, deg, qdeg, sb_str(&full));
		if (c && first && naive >= 0 && naive < cmetric(c)) {
			/* the walk overfitted its sample: keep the naive one */
			free(c->text);
			A.nc--;
			memcpy(coef, k->ncoef, sizeof coef);
			memcpy(k->ir.coef, coef, sizeof coef);
			core_build_ir(k);
			sb_clear(&body);
			sb_clear(&full);
			bl_free(&l);
			ir_text(&k->ir, &body, &l);
			compose(&full, sb_str(&body), &l);
			c = submit(kind, shape, fname, deg, qdeg, sb_str(&full));
		}
		if (c && first && A.w && A.o.search.kind && k->nsx > 0) {
			/*
			 * the search starts where the walk stopped; what it finds
			 * is kept only if the table's measure agrees it is better
			 */
			struct nsres sr;
			double walked[2 * AMAXC], mw = cmetric(c), ms = -1;
			int kept = 0;
			memcpy(walked, coef, sizeof walked);
			memcpy(k->ir.coef, coef, sizeof coef);
			core_build_ir(k);
			if (core_search(k, &A.o.search, &sr)) {
				struct cand *c2;
				double found[2 * AMAXC];
				memcpy(found, k->ir.coef, sizeof found);
				sb_clear(&body);
				sb_clear(&full);
				bl_free(&l);
				ir_text(&k->ir, &body, &l);
				compose(&full, sb_str(&body), &l);
				c2 = submit(kind, shape, fname, deg, qdeg, sb_str(&full));
				if (c2) {
					ms = cmetric(c2);
					if (ms < mw) {
						struct cand t = A.cd[A.nc - 2];
						A.cd[A.nc - 2] = A.cd[A.nc - 1];
						A.cd[A.nc - 1] = t;
						memcpy(coef, found, sizeof coef);
						kept = 1;
					}
					free(A.cd[A.nc - 1].text);
					A.nc--;
				}
			}
			if (!kept) {
				memcpy(k->ir.coef, walked, sizeof walked);
				core_build_ir(k);
			}
			c = &A.cd[A.nc - 1];
			c->skind = A.o.search.kind;
			c->skept = kept;
			c->snsx = k->nsx;
			c->sevals = sr.evals;
			c->sbefore = sr.before;
			c->safter = sr.after;
			c->smw = mw;
			c->sms = ms;
		}
		if (c) {
			c->rerr = rerr;
			c->freeerr = freeerr;
			if (A.w && naive >= 0 && first)
				c->naive = naive;
		}
		bl_free(&l);
		free(body.p);
		free(full.p);
	}
}

static double
floor_err(void)
{
	ld u = A.w ? ldexpl(1, -A.prec) : 1e-18L;

	return (double)(A.rel ? u / 32 : u / 32 * A.fmax);
}

static int
degree_of(int nt)
{
	return A.k0 + A.step * (nt - 1);
}

static ld taylor[AMAXJ];
static int ntay;

/* the named method's real fit per degree, for focus_report */
static struct {
	struct fitprob pb;
	struct fitres fr;
	int deg;
} focus[AMAXC];
static int nfocus;

static const char *
S_pow(int pw)
{
	static char buf[4][16];
	static int k;
	char *b = buf[k++ & 3];

	snprintf(b, 16, "^%d", pw);
	return b;
}

static void
direct_polys(void)
{
	int nt, ntmax = AMAXC - AMAXK - 2, stop = 0, printed = 0;
	struct fitprob pb;
	struct fitres rc, re, rr, rf;
	struct core k;
	double hist[AMAXC + 1];

	if (A.o.dmax == 0)
		A.o.dmax = A.w == 64 ? 27 : 17;
	ntay = 0;
	if (A.c >= A.lo && A.c <= A.hi) {
		ntay = AMAXJ;
		if (!lref_taylor(&A.ref, A.c, ntay, taylor))
			ntay = 0;
	}
	memset(&k, 0, sizeof k);
	k.w = A.w;
	nfocus = 0;
	for (nt = 1; nt <= ntmax && !stop; nt++) {
		int deg = degree_of(nt);
		double fe = -1;
		char cs[4][16];
		if (deg > A.o.dmax)
			break;
		base_prob(&pb, nt, 0, FM_CHEB);
		fit_run(&pb, &rc);
		pb.method = FM_ECON;
		pb.taylor = ntay ? taylor : NULL;
		pb.ntaylor = ntay;
		fit_run(&pb, &re);
		pb.method = FM_REMEZ;
		fit_run(&pb, &rr);
		if (A.nkeep) {
			struct fitprob fp = pb;
			fp.nkeep = 0;
			if (fit_run(&fp, &rf))
				fe = metric_of_fit(&rf);
		}
		hist[nt] = rr.ok ? metric_of_fit(&rr) : INFINITY;
		if (deg < A.o.dmin)
			continue;
		if (!printed) {
			printf("  fits in real arithmetic (%s error of %s):\n",
			    A.rel ? "max relative" : "max absolute",
			    shape_str(A.k0, A.step, A.c, 0));
			printf("    deg  cheb interp  cheb econ    minimax%s\n",
			    A.nkeep ? "      no exact pts" : "");
			printed = 1;
		}
		if (rc.ok)
			snprintf(cs[0], 16, "%-11.3g", metric_of_fit(&rc));
		else
			snprintf(cs[0], 16, "%-11s", "-");
		if (re.ok)
			snprintf(cs[1], 16, "%-11.3g", metric_of_fit(&re));
		else
			snprintf(cs[1], 16, "%-11s", "-");
		if (rr.ok)
			snprintf(cs[2], 16, "%-11.3g", metric_of_fit(&rr));
		else
			snprintf(cs[2], 16, "%-11s", "-");
		if (fe >= 0)
			snprintf(cs[3], 16, "%.3g", fe);
		else
			cs[3][0] = 0;
		printf("    %-4d %s  %s  %s  %s\n", deg, cs[0], cs[1], cs[2],
		    A.nkeep ? cs[3] : "");
		if (!rr.ok)
			continue;
		if (nfocus < AMAXC) {
			struct fitres *ch = A.o.method == FM_CHEB ? &rc :
			    A.o.method == FM_ECON ? &re : &rr;
			if (ch->ok) {
				focus[nfocus].pb = pb;
				focus[nfocus].pb.method = A.o.method;
				focus[nfocus].fr = *ch;
				focus[nfocus].deg = deg;
				nfocus++;
			}
		}
		k.pb = pb;
		k.fr = rr;
		if (A.o.method == FM_CHEB && rc.ok) {
			k.pb.method = FM_CHEB;
			k.fr = rc;
		} else if (A.o.method == FM_ECON && re.ok) {
			k.pb.method = FM_ECON;
			k.fr = re;
		}
		machine_forms(&k, k.pb.method == FM_CHEB ? "cheb" :
		    k.pb.method == FM_ECON ? "econ" : "minimax",
		    shape_str(A.k0, A.step, A.c, 0), deg, 0, metric_of_fit(&k.fr),
		    fe);
		if (metric_of_fit(&rr) < floor_err())
			stop = 1;
		/* polynomials that barely converge are not worth more terms */
		if (deg >= 4 && nt >= 3 && (hist[nt] > 0.25 * (A.rel ? 1 :
		    (double)A.fmax) || hist[nt] > 0.7 * hist[nt - 2])) {
			printf("    (stopped: the error shrinks too slowly; a range "
			    "reduction or a split fits better)\n");
			A.stalled = 1;
			stop = 1;
		}
		if (A.o.tol_err > 0 && metric_of_fit(&rr) < A.o.tol_err / 4)
			stop = 1;
		if (A.o.tol_ulp > 0 && A.nc > 0 &&
		    A.cd[A.nc - 1].eulp <= A.o.tol_ulp && !A.cd[A.nc - 1].bad)
			stop = 1;
	}
	if (!printed)
		printf("  no polynomial fit in the degree range\n");
	core_free(&k);
}

/*
 * The report for a named method: the polynomial itself, as an expression
 * and in the Chebyshev basis on the interval, for one degree: the one
 * asked for, else the lowest that reaches the format's precision.
 */
static const char *
method_name(int m)
{
	return m == FM_CHEB ? "Chebyshev interpolation" : m == FM_ECON ?
	    "Chebyshev-economised Taylor" : "minimax";
}

static void
focus_report(void)
{
	double goal;
	int i, pick = -1, k, j, N;
	struct sbuf b = {0};
	ld mid = (A.hi + A.lo) / 2, half = (A.hi - A.lo) / 2, ck[AMAXC * 2 + 2];
	ld big = 0;

	if (nfocus == 0)
		return;
	goal = A.o.tol_err > 0 ? A.o.tol_err : A.w == 32 ? 6e-8 : A.w == 64 ?
	    1.2e-16 : 1e-12;
	if (!A.rel)
		goal *= (double)(A.fmax > 0 ? A.fmax : 1);
	for (i = 0; i < nfocus; i++)
		if (metric_of_fit(&focus[i].fr) <= goal) {
			pick = i;
			break;
		}
	if (A.o.dmin > 0 && A.o.dmin == A.o.dmax)
		pick = nfocus - 1;
	if (pick < 0)
		pick = nfocus - 1;
	printf("  %s polynomial of %s, degree %d (max %s error %.3g in real "
	    "arithmetic%s):\n", method_name(A.o.method), A.fname,
	    focus[pick].deg, A.rel ? "relative" : "absolute",
	    metric_of_fit(&focus[pick].fr), A.o.dmin > 0 &&
	    A.o.dmin == A.o.dmax ? "" : A.o.tol_err > 0 ?
	    ", the lowest degree within the error asked" :
	    ", the lowest degree that reaches the format's precision");
	/* as an expression in x (or x - c) */
	for (j = focus[pick].fr.nn; j-- > 0;) {
		ld cf = focus[pick].fr.num[j];
		int pw = A.k0 + A.step * j;
		char v[48];
		if (cf == 0)
			continue;
		if (A.c != 0)
			snprintf(v, sizeof v, "(x - %.9g)", (double)A.c);
		else
			snprintf(v, sizeof v, "x");
		if (b.n)
			sb_puts(&b, cf < 0 ? " - " : " + ");
		else if (cf < 0)
			sb_puts(&b, "-");
		if (pw == 0)
			sb_printf(&b, "%.10Lg", fabsl(cf));
		else
			sb_printf(&b, "%.10Lg*%s%s", fabsl(cf), v, pw == 1 ? "" :
			    S_pow(pw));
	}
	printf("    %s(x) = %s\n", A.fname, b.n ? b.p : "0");
	/* in the Chebyshev basis on the whole interval: exact for degree < N */
	N = focus[pick].deg + 1;
	for (k = 0; k < N; k++) {
		ld sum = 0;
		for (j = 0; j < N; j++) {
			ld th = PI_L * ((ld)j + 0.5L) / N;
			ld x = mid + half * cosl(th);
			sum += fit_eval(&focus[pick].pb, &focus[pick].fr, x) *
			    cosl((ld)k * th);
		}
		ck[k] = sum * 2 / N / (k == 0 ? 2 : 1);
		if (fabsl(ck[k]) > big)
			big = fabsl(ck[k]);
	}
	sb_clear(&b);
	for (k = 0; k < N; k++) {
		if (fabsl(ck[k]) <= big * 1e-13L)
			continue;
		if (b.n)
			sb_puts(&b, ck[k] < 0 ? " - " : " + ");
		else if (ck[k] < 0)
			sb_puts(&b, "-");
		sb_printf(&b, "%.10Lg*T%d(u)", fabsl(ck[k]), k);
	}
	printf("    in the Chebyshev basis: %s\n      where u = ", b.p);
	if (mid == 0 && half == 1)
		printf("x");
	else if (mid == 0)
		printf("x/%.10g", (double)half);
	else
		printf("(x - %.10g)/%.10g", (double)mid, (double)half);
	printf(" maps [%.9g, %.9g] to [-1, 1]\n", (double)A.lo, (double)A.hi);
	printf("    other degrees: add `degree n`; the other methods and "
	    "shapes: add `compare`\n");
	free(b.p);
}

static void
direct_rationals(void)
{
	int nt, nd, tot;
	struct fitprob pb;
	struct fitres rr, rp;
	struct core k;
	int any = 0;

	memset(&k, 0, sizeof k);
	k.w = A.w;
	for (tot = 2; tot <= 8; tot++)
		for (nd = 1; nd <= 3 && nd < tot; nd++) {
			int deg;
			char shape[40];
			nt = tot - nd;
			deg = degree_of(nt);
			if (deg > A.o.dmax || nt < 1)
				continue;
			base_prob(&pb, nt, nd, FM_PADE);
			pb.taylor = ntay ? taylor : NULL;
			pb.ntaylor = ntay;
			if (!fit_run(&pb, &rp))
				rp.ok = 0;
			pb.method = FM_REMEZ;
			if (!fit_run(&pb, &rr))
				continue;
			if (!any) {
				printf("  rational fits (num/den degree: minimax, "
				    "Pade; %s error):\n   ", A.rel ? "max rel" :
				    "max abs");
				any = 1;
			}
			if (++any % 6 == 0)
				printf("\n   ");
			if (rp.ok)
				printf(" %d/%d %.3g (%.3g)", deg, A.step * nd,
				    metric_of_fit(&rr), metric_of_fit(&rp));
			else
				printf(" %d/%d %.3g (-)", deg, A.step * nd,
				    metric_of_fit(&rr));
			snprintf(shape, sizeof shape, "%s",
			    shape_str(A.k0, A.step, A.c, 1));
			k.pb = pb;
			k.fr = rr;
			machine_forms(&k, "rational", shape, deg, A.step * nd,
			    metric_of_fit(&rr), -1);
		}
	if (any)
		printf("\n");
	core_free(&k);
}

/*
 * The walk's candidate is A.cd[A.nc - 2] (its measure mw) and the
 * search's c2 the last one (NULL: none was made): c2 is kept only when
 * the table's measure agrees it is better.  The one left carries the
 * search's report and is returned.
 */
static struct cand *
search_settle(struct cand *c2, double mw, const struct nsres *sr, int nsx)
{
	struct cand *c;
	double ms = -1;
	int kept = 0;

	if (c2) {
		ms = cmetric(c2);
		if (ms < mw) {
			struct cand t = A.cd[A.nc - 2];
			A.cd[A.nc - 2] = A.cd[A.nc - 1];
			A.cd[A.nc - 1] = t;
			kept = 1;
		}
		free(A.cd[A.nc - 1].text);
		A.nc--;
	}
	c = &A.cd[A.nc - 1];
	c->skind = A.o.search.kind;
	c->skept = kept;
	c->snsx = nsx;
	c->sevals = sr->evals;
	c->sbefore = sr->before;
	c->safter = sr->after;
	c->smw = mw;
	c->sms = ms;
	return c;
}

/* ---- piecewise: two pieces, coefficients chosen by a select ---- */

/* the two pieces as one candidate: a = |x| (folded) or x; t = a -
 * c(piece); Horner in t, each coefficient a select of the pieces' */
static void
piece_text(const struct core *k, int nt, double B, int fold,
    struct sbuf *full, char *shape, size_t ns)
{
	struct sbuf body = {0};
	struct blist l = {0};
	char kb[48], tmp[160];
	int j;

	bl_add(&l, "a", fold ? "abs(x)" : "x");
	kfmt(kb, sizeof kb, B, A.w ? A.w : 64);
	snprintf(tmp, sizeof tmp, "a < %s", kb);
	bl_add(&l, "lo", tmp);
	{
		char c0[48], c1[48];
		kfmt(c0, sizeof c0, rnd(A.w, k[0].pb.c), A.w ? A.w : 64);
		kfmt(c1, sizeof c1, rnd(A.w, k[1].pb.c), A.w ? A.w : 64);
		/* typed constants: a select of reals would convert */
		snprintf(tmp, sizeof tmp, "a - (lo ? %s(%s) : %s(%s))",
		    fmt_name(A.w), c0, fmt_name(A.w), c1);
		bl_add(&l, "t", tmp);
	}
	for (j = 0; j < nt; j++) {
		char c0[48], c1[48], nm[8];
		kfmt(c0, sizeof c0, k[0].ir.coef[j], A.w ? A.w : 64);
		kfmt(c1, sizeof c1, k[1].ir.coef[j], A.w ? A.w : 64);
		snprintf(nm, sizeof nm, "k%d", j);
		snprintf(tmp, sizeof tmp, "lo ? %s(%s) : %s(%s)",
		    fmt_name(A.w), c0, fmt_name(A.w), c1);
		bl_add(&l, nm, tmp);
	}
	{
		struct sbuf h = {0};
		sb_printf(&h, "k%d", nt - 1);
		for (j = nt - 2; j >= 0; j--) {
			struct sbuf h2 = {0};
			if (A.o.fma && A.w)
				sb_printf(&h2, "fma(%s, t, k%d)", sb_str(&h), j);
			else
				sb_printf(&h2, "k%d + t*%s%s%s", j,
				    j == nt - 2 ? "" : "(", sb_str(&h),
				    j == nt - 2 ? "" : ")");
			free(h.p);
			h = h2;
		}
		if (fold && A.odd)
			sb_printf(&body, "copysign(%s, x)", sb_str(&h));
		else
			sb_puts(&body, sb_str(&h));
		free(h.p);
	}
	compose(full, sb_str(&body), &l);
	snprintf(shape, ns, "2 pieces%s", fold ? (A.odd ?
	    ", copysign(.., x)" : ", of |x|") : "");
	bl_free(&l);
	free(body.p);
}

static void
piecewise(void)
{
	int nt, i, j, fold = A.step == 2;
	ld a = fold ? A.xa : A.lo, b = fold ? A.xb : A.hi;
	double B = rnd(A.w, (a + b) / 2);
	struct core k[2];
	ld pa[2], pb_[2];
	int any = 0;

	if (A.o.haswt)
		return;
	memset(k, 0, sizeof k);
	pa[0] = a;
	pb_[0] = B;
	pa[1] = B;
	pb_[1] = b;
	for (nt = 2; nt <= 8; nt++) {
		struct sbuf full = {0};
		char shape[40];
		struct cand *c;
		double worst = 0;
		int ok = 1;
		if (nt - 1 > A.o.dmax)
			break;
		for (i = 0; i < 2 && ok; i++) {
			struct fitprob *p = &k[i].pb;
			ld ctr = rnd(A.w, (pa[i] + pb_[i]) / 2), f0;
			memset(p, 0, sizeof *p);
			p->f = target_f;
			p->xa = pa[i];
			p->xb = pb_[i];
			p->step = 1;
			p->rel = A.rel;
			p->nt = nt;
			p->method = FM_REMEZ;
			f0 = pa[i] == 0 ? lref_eval(&A.ref, 0) : 1;
			if (pa[i] <= 0 && pb_[i] >= 0 && f0 == 0) {
				/* through the zero at 0: keep it exact */
				ctr = 0;
				p->keep[p->nkeep++] = 0;
			}
			p->c = ctr;
			for (j = 0; j < A.nkeep; j++) {
				ld x = fold ? fabsl(A.keep[j]) : A.keep[j];
				if (x >= pa[i] && x <= pb_[i] &&
				    p->nkeep < AMAXK)
					p->keep[p->nkeep++] = x;
			}
			if (p->nkeep >= nt || !fit_run(p, &k[i].fr)) {
				ok = 0;
				break;
			}
			if (k[i].fr.err > worst)
				worst = (double)k[i].fr.err;
			k[i].w = A.w;
			k[i].var = "a";
			snprintf(k[i].sname, sizeof k[i].sname, "s");
			k[i].objrel = A.rel;
			k[i].fma_ = A.o.fma && A.w;
			core_round(&k[i], A.o.refine);
		}
		if (!ok)
			continue;
		if (!any) {
			printf("  piecewise (2 pieces split at %.9g, %s error):",
			    B, A.rel ? "max rel" : "max abs");
			any = 1;
		}
		printf(" %d: %.3g", nt - 1, worst);
		piece_text(k, nt, B, fold, &full, shape, sizeof shape);
		c = submit("piecewise", shape, A.o.fma && A.w ? "horner+fma" :
		    "horner", nt - 1, 0, sb_str(&full));
		if (c && A.w && A.o.search.kind && k[0].nsx > 0 && k[1].nsx > 0) {
			/* both pieces' coefficients moved, each on its piece's
			 * sample; kept only if the table's measure agrees */
			struct core *ks[2] = {&k[0], &k[1]};
			struct nsres sr;
			double mw = cmetric(c);
			int nsx = 0;
			struct cand *c2 = NULL;
			if (cores_search(ks, 2, &sr, &nsx)) {
				sb_clear(&full);
				piece_text(k, nt, B, fold, &full, shape, sizeof shape);
				c2 = submit("piecewise", shape, A.o.fma && A.w ?
				    "horner+fma" : "horner", nt - 1, 0, sb_str(&full));
			}
			c = search_settle(c2, mw, &sr, nsx);
		}
		if (c)
			c->rerr = worst;
		free(full.p);
	}
	if (any)
		printf("\n");
	core_free(&k[0]);
	core_free(&k[1]);
}

/* ---- range reductions and identities ---- */

enum {
	F_SIN = 1, F_COS, F_TAN, F_SEC, F_CSC, F_COT, F_EXP, F_LOG, F_SQRT,
	F_RSQRT, F_ATAN, F_SINH, F_COSH, F_TANH
};
static const char *const fnames[] = {
	"", "sin", "cos", "tan", "sec", "csc", "cot", "exp", "log", "sqrt",
	"rsqrt", "atan", "sinh", "cosh", "tanh"
};

static ld l_sin(void *c, ld x) { (void)c; return sinl(x); }
static ld l_cos(void *c, ld x) { (void)c; return cosl(x); }
static ld l_tan(void *c, ld x) { (void)c; return tanl(x); }
static ld l_exp(void *c, ld x) { (void)c; return expl(x); }
static ld l_expm1(void *c, ld x) { (void)c; return expm1l(x); }
static ld l_log1p(void *c, ld x) { (void)c; return log1pl(x); }
static ld l_sqrt(void *c, ld x) { (void)c; return sqrtl(x); }
static ld l_rsqrt(void *c, ld x) { (void)c; return 1 / sqrtl(x); }
static ld l_atan(void *c, ld x) { (void)c; return atanl(x); }
static ld l_sinh(void *c, ld x) { (void)c; return sinhl(x); }
static ld l_tanh(void *c, ld x) { (void)c; return tanhl(x); }

static ld
libm_of(int fn, ld x)
{
	switch (fn) {
	case F_SIN: return sinl(x);
	case F_COS: return cosl(x);
	case F_TAN: return tanl(x);
	case F_SEC: return 1 / cosl(x);
	case F_CSC: return 1 / sinl(x);
	case F_COT: return cosl(x) / sinl(x);
	case F_EXP: return expl(x);
	case F_LOG: return logl(x);
	case F_SQRT: return sqrtl(x);
	case F_RSQRT: return 1 / sqrtl(x);
	case F_ATAN: return atanl(x);
	case F_SINH: return sinhl(x);
	case F_COSH: return coshl(x);
	case F_TANH: return tanhl(x);
	}
	return NAN;
}

static nid
strip(nid n)
{
	while (N.op[n] == O_CAST && TY_FP(N.ty[N.a[n]]))
		n = N.a[n];
	return n;
}

/* which function is this, structurally; then checked numerically */
static int
recognise(nid body)
{
	nid b = strip(body), a;
	int fn = 0, i, ok = 1;

	switch (N.op[b]) {
	case O_SIN: fn = F_SIN; break;
	case O_COS: fn = F_COS; break;
	case O_TAN: fn = F_TAN; break;
	case O_SEC: fn = F_SEC; break;
	case O_CSC: fn = F_CSC; break;
	case O_COT: fn = F_COT; break;
	case O_EXP: fn = F_EXP; break;
	case O_LOG: fn = F_LOG; break;
	case O_SQRT: fn = F_SQRT; break;
	case O_ATAN: fn = F_ATAN; break;
	case O_SINH: fn = F_SINH; break;
	case O_COSH: fn = F_COSH; break;
	case O_TANH: fn = F_TANH; break;
	case O_DIV:
		if (is_const(strip(N.a[b])) && const_is(strip(N.a[b]), 1) &&
		    N.op[strip(N.b[b])] == O_SQRT) {
			fn = F_RSQRT;
			b = strip(N.b[b]);
		}
		break;
	case O_POW:
		if (is_const(N.b[b]) && const_is(N.b[b], -0.5))
			fn = F_RSQRT;
		break;
	}
	if (!fn)
		return 0;
	a = strip(N.a[b]);
	if (N.op[a] != O_VAR)
		return 0;
	for (i = 0; i <= 32; i++) {
		ld x = A.lo + (A.hi - A.lo) * i / 32, y = lref_eval(&A.ref, x);
		ld z = libm_of(fn, x);
		if (!isfinite(y) && !isfinite(z))
			continue;
		if (!(fabsl(y - z) <= 1e-13L * fabsl(z) + 1e-300L))
			ok = 0;
	}
	if (!ok)
		return 0;
	printf("  checked: the target equals %s(x) at 33 points: range "
	    "reductions and identities for %s apply\n", fnames[fn],
	    fnames[fn]);
	return fn;
}

/*
 * Cody-Waite constants.  pi/2 and ln 2 are kept as three doubles (about
 * 160 bits), so the low parts are right even for f64.
 */
static const double pio2_3[3] = {
	1.5707963267948966, 6.123233995736766e-17, -1.4973849048591698e-33
};
static const double ln2_3[3] = {
	0.6931471805599453, 2.3190468138462996e-17, 5.707708438416212e-34
};

/* split v = v[0] + v[1] + v[2] into parts of `bits` bits; n parts */
static void
cw_parts(const double *v, int bits, int n, double *out)
{
	ld rest[3];
	int i, j;

	rest[0] = v[0];
	rest[1] = v[1];
	rest[2] = v[2];
	for (i = 0; i < n; i++) {
		ld c = rest[0] + rest[1] + rest[2], h;
		int e = ilogbl(c);
		if (i == n - 1) {
			out[i] = rnd(A.w, c);
			break;
		}
		h = ldexpl(roundl(ldexpl(c, bits - 1 - e)), e - (bits - 1));
		out[i] = (double)h;
		/* subtract exactly from the largest part first */
		rest[0] -= h;
		for (j = 0; j < 2; j++)
			if (fabsl(rest[j]) < fabsl(rest[j + 1])) {
				ld t = rest[j];
				rest[j] = rest[j + 1];
				rest[j + 1] = t;
			}
	}
}

/* r = x - k*c1 - k*c2, signs folded into the constants */
static void
cw_text(char *t, size_t n, double c1, double c2, int fm)
{
	char s1[48], s2[48];

	kfmt(s1, sizeof s1, -c1, A.w);
	kfmt(s2, sizeof s2, -c2, A.w);
	if (fm)
		snprintf(t, n, "fma(k, %s, fma(k, %s, x))", s2, s1);
	else {
		kfmt(s1, sizeof s1, c1, A.w);
		kfmt(s2, sizeof s2, fabs(c2), A.w);
		snprintf(t, n, "x - k*%s %s k*%s", s1, c2 < 0 ? "+" : "-", s2);
	}
}

/* a reduction built again with its cores searched (reductions) */
static struct {
	int on, moved, nsx;
	struct nsres sr;
} RS;

static int
core_make(struct core *k, ldfn f, ld a, ld b, int k0, int step, ld c,
    int nt, const char *var, const char *sname)
{
	memset(&k->pb, 0, sizeof k->pb);
	k->pb.f = f;
	k->pb.xa = a;
	k->pb.xb = b;
	k->pb.c = c;
	k->pb.k0 = k0;
	k->pb.step = step;
	k->pb.rel = 1;
	k->pb.nt = nt;
	k->pb.method = FM_REMEZ;
	k->w = A.w;
	k->var = var;
	snprintf(k->sname, sizeof k->sname, "%s", sname);
	k->estrin = 0;
	k->fma_ = A.o.fma && A.w;
	k->objrel = 1;
	if (!fit_run(&k->pb, &k->fr))
		return 0;
	core_round(k, A.o.refine);
	if (RS.on && k->w && k->nsx > 0) {
		struct core *ks[1] = {k};
		struct nsres r;
		int nsx;
		RS.moved |= cores_search(ks, 1, &r, &nsx);
		RS.sr.evals += r.evals;
		RS.sr.accepted += r.accepted;
		if (r.before > RS.sr.before)
			RS.sr.before = r.before;
		if (r.after > RS.sr.after)
			RS.sr.after = r.after;
		RS.nsx += nsx;
	}
	return 1;
}

enum {
	R_QUAD, R_COSSHIFT, R_COSHALF, R_SINCOS, R_TANHALF, R_EXP, R_LOG,
	R_SQRT, R_NEWTON1, R_NEWTON2, R_ATAN, R_SQRTN, R_HYP, R_EXPTAB, R_NRED
};

/* R_EXPTAB's table: 2^tab_t entries (reductions picks it) */
static int tab_t;

struct fmtc {
	const char *ut, *it, *ft;
	int mant, bias, signsh;
	const char *shifter, *onebits, *sqrthalf, *expmask, *magic;
};

static const struct fmtc fc32 = {
	"u32", "i32", "f32", 23, 127, 30, "12582912.0", "0x3f800000",
	"0x3f3504f3", "0xff800000", "0x5f375a86"
};
static const struct fmtc fc64 = {
	"u64", "i64", "f64", 52, 1023, 62, "6755399441055744.0",
	"0x3ff0000000000000", "0x3fe6a09e667f3bcd", "0xfff0000000000000",
	"0x5fe6eb50c7b537a9"
};

/* the text of a core: its bindings go to l, its body to out */
static void
core_text(const struct core *k, struct blist *l, struct sbuf *out)
{
	sb_clear(out);
	ir_text(&k->ir, out, l);
}

/* one reduced candidate at level L; returns the core degree or 0 */
static int
reduction(int r, int L, struct sbuf *full, char *shape, size_t ns)
{
	const struct fmtc *F = A.w == 32 ? &fc32 : &fc64;
	struct core k1, k2;
	struct blist l = {0};
	struct sbuf b1 = {0}, b2 = {0}, body = {0};
	char s1[48], s2[48], s3[48], s4[48], t[400];
	ld M = fmaxl(fabsl(A.lo), fabsl(A.hi)), R;
	int fm = A.o.fma && A.w, deg = 0, wb = A.w;
	double c1, c2, cw3 = 0;

	memset(&k1, 0, sizeof k1);
	memset(&k2, 0, sizeof k2);
	switch (r) {
	case R_QUAD:
		/* x = k pi/2 + r, |r| <= pi/4; quadrant from the shifter bits */
		R = PI_L / 4 * 1.001L + M * ldexpl(1, 2 - A.prec);
		if (!core_make(&k1, l_sin, 0, R, 1, 2, 0, L + 1, "r", "s") ||
		    !core_make(&k2, l_cos, 0, R, 0, 2, 0, L + 1, "r", "s"))
			goto fail;
		{
			double cc[3];
			if (M * 2 / PI_L > 4) {
				/* large k: a third part keeps r accurate near zeros */
				cw_parts(pio2_3, A.w == 64 ? 33 : 12, 3, cc);
				cw3 = cc[2];
			} else
				cw_parts(pio2_3, A.w == 64 ? 33 : 12, 2, cc);
			c1 = cc[0];
			c2 = cc[1];
		}
		kfmt(s1, sizeof s1, rnd(wb, 2 / PI_L), wb);
		snprintf(t, sizeof t, "x*%s + %s", s1, F->shifter);
		bl_add(&l, "j", t);
		snprintf(t, sizeof t, "asuint(j)%s", A.fn == F_COS ? " + 1" : "");
		bl_add(&l, "q", t);
		snprintf(t, sizeof t, "j - %s", F->shifter);
		bl_add(&l, "k", t);
		cw_text(t, sizeof t, c1, c2, fm);
		if (cw3 != 0) {
			char t2[400];
			kfmt(s3, sizeof s3, fm ? -cw3 : fabs(cw3), wb);
			if (fm)
				snprintf(t2, sizeof t2, "fma(k, %s, %s)", s3, t);
			else
				snprintf(t2, sizeof t2, "%s %s k*%s", t,
				    cw3 < 0 ? "+" : "-", s3);
			snprintf(t, sizeof t, "%s", t2);
		}
		bl_add(&l, "r", t);
		core_text(&k1, &l, &b1);
		core_text(&k2, &l, &b2);
		sb_printf(&body, "asfloat(asuint((q & 1) == 0 ? %s : %s) xor "
		    "((q & 2) << %d))", sb_str(&b1), sb_str(&b2), F->signsh);
		snprintf(shape, ns, "quadrant r=x-k*pi/2 (%d-part), sin+cos",
		    cw3 != 0 ? 3 : 2);
		deg = 2 * L + 1;
		break;
	case R_COSSHIFT:
		/* cos x = sin(pi/2 - x) */
		R = fmaxl(fabsl(PI_L / 2 - A.lo), fabsl(PI_L / 2 - A.hi)) * 1.0001L;
		if (!core_make(&k1, l_sin, 0, R, 1, 2, 0, L + 1, "r", "s"))
			goto fail;
		/* pi/2 in two parts, so r is accurate where cos x is tiny */
		c1 = rnd(wb, PI_L / 2);
		c2 = rnd(wb, PI_L / 2 - c1);
		kfmt(s1, sizeof s1, c1, wb);
		kfmt(s2, sizeof s2, fabs(c2), wb);
		snprintf(t, sizeof t, "%s - x %s %s", s1, c2 < 0 ? "-" : "+", s2);
		bl_add(&l, "r", t);
		core_text(&k1, &l, &b1);
		sb_puts(&body, sb_str(&b1));
		snprintf(shape, ns, "sin(pi/2 - x), pi/2 in 2 parts");
		deg = 2 * L + 1;
		break;
	case R_COSHALF:
		/* cos x = 1 - 2 sin^2(x/2) */
		if (!core_make(&k1, l_sin, 0, M / 2 * 1.0001L, 1, 2, 0, L + 1,
		    "h", "s"))
			goto fail;
		bl_add(&l, "h", "0.5*x");
		core_text(&k1, &l, &b1);
		bl_add(&l, "p", sb_str(&b1));
		sb_puts(&body, "1.0 - 2.0*p*p");
		snprintf(shape, ns, "1 - 2 sin(x/2)^2");
		deg = 2 * L + 1;
		break;
	case R_SINCOS:
		/* tan = sin/cos, sec = 1/cos, csc = 1/sin, cot = cos/sin */
		if (!core_make(&k1, l_sin, 0, M * 1.0001L, 1, 2, 0, L + 1, "x",
		    "s") || !core_make(&k2, l_cos, 0, M * 1.0001L, 0, 2, 0,
		    L + 1, "x", "s"))
			goto fail;
		core_text(&k1, &l, &b1);
		core_text(&k2, &l, &b2);
		switch (A.fn) {
		case F_TAN:
			sb_printf(&body, "(%s)/(%s)", sb_str(&b1), sb_str(&b2));
			snprintf(shape, ns, "sin/cos");
			break;
		case F_COT:
			sb_printf(&body, "(%s)/(%s)", sb_str(&b2), sb_str(&b1));
			snprintf(shape, ns, "cos/sin");
			break;
		case F_SEC:
			sb_printf(&body, "1.0/(%s)", sb_str(&b2));
			snprintf(shape, ns, "1/cos");
			break;
		default:
			sb_printf(&body, "1.0/(%s)", sb_str(&b1));
			snprintf(shape, ns, "1/sin");
		}
		deg = 2 * L + 1;
		break;
	case R_TANHALF:
		/* tan x = 2t/(1 - t^2), t = tan(x/2) */
		if (!core_make(&k1, l_tan, 0, M / 2 * 1.0001L, 1, 2, 0, L + 1,
		    "h", "s"))
			goto fail;
		bl_add(&l, "h", "0.5*x");
		core_text(&k1, &l, &b1);
		bl_add(&l, "p", sb_str(&b1));
		sb_puts(&body, "2.0*p/(1.0 - p*p)");
		snprintf(shape, ns, "2t/(1-t^2), t = tan(x/2)");
		deg = 2 * L + 1;
		break;
	case R_EXP:
		/* x = k ln2 + r; 2^k from the shifter bits into the exponent */
		R = LN2_L / 2 * 1.001L + M * ldexpl(1, 2 - A.prec);
		if (!core_make(&k1, l_exp, -R, R, 0, 1, 0, L + 2, "r", "s"))
			goto fail;
		{
			double cc[2];
			cw_parts(ln2_3, A.prec - 11, 2, cc);
			c1 = cc[0];
			c2 = cc[1];
		}
		kfmt(s1, sizeof s1, rnd(wb, 1 / LN2_L), wb);
		snprintf(t, sizeof t, "x*%s + %s", s1, F->shifter);
		bl_add(&l, "j", t);
		snprintf(t, sizeof t, "j - %s", F->shifter);
		bl_add(&l, "k", t);
		cw_text(t, sizeof t, c1, c2, fm);
		bl_add(&l, "r", t);
		core_text(&k1, &l, &b1);
		sb_printf(&body, "(%s)*asfloat((asuint(j) << %d) + %s)",
		    sb_str(&b1), F->mant, F->onebits);
		snprintf(shape, ns, "2^k * P(r), r = x - k*ln2");
		deg = L + 1;
		break;
	case R_EXPTAB: {
		/*
		 * Tang's table: x = k ln2/N + r, |r| <= ln2/2N, k = kh N + kl,
		 * exp x = 2^kh T[kl] (1 + q(r)), T[kl] = 2^(kl/N) as the format
		 * rounds it, q a fit of e^r - 1 (its error reaches the result
		 * times |r|); 2^kh is an add on T[kl]'s exponent field
		 */
		int N = 1 << tab_t, nb = 1, j;
		double cc[3];
		ld kmax;
		struct sbuf tb = {0};

		R = LN2_L / (2 * N) * 1.001L + M * ldexpl(1, 2 - A.prec);
		if (!core_make(&k1, l_expm1, -R, R, 1, 1, 0, L + 1, "r", "s"))
			goto fail;
		kfmt(s1, sizeof s1, rnd(wb, N / LN2_L), wb);
		snprintf(t, sizeof t, "x*%s + %s", s1, F->shifter);
		bl_add(&l, "j", t);
		snprintf(t, sizeof t, "j - %s", F->shifter);
		bl_add(&l, "k", t);
		/*
		 * r = x - k ln2/N.  With an fma k c1 is exact inside it, so c1
		 * is ln2/N to the format's precision; without, k c1 and k c2
		 * must be exact (Cody-Waite): c1 and c2 of prec - nb bits for k
		 * of nb bits, and a third part when two leave r short
		 */
		for (kmax = M * N / LN2_L + 1; ldexpl(1, nb) <= kmax; nb++)
			;
		if (fm) {
			cw_parts(ln2_3, A.prec, 2, cc);
			cc[2] = 0;
		} else if (2 * nb > A.prec - 5)
			cw_parts(ln2_3, A.prec - nb, 3, cc);
		else {
			cw_parts(ln2_3, A.prec - nb, 2, cc);
			cc[2] = 0;
		}
		for (j = 0; j < 3; j++)
			cc[j] /= N;
		cw_text(t, sizeof t, cc[0], cc[1], fm);
		if (cc[2] != 0) {
			char t2[400];
			kfmt(s3, sizeof s3, fabs(cc[2]), wb);
			snprintf(t2, sizeof t2, "%s %s k*%s", t,
			    cc[2] < 0 ? "+" : "-", s3);
			snprintf(t, sizeof t, "%s", t2);
		}
		bl_add(&l, "r", t);
		core_text(&k1, &l, &b1);
		for (j = 0; j < N; j++) {
			kfmt(s2, sizeof s2, rnd(wb, exp2l((ld)j / N)), wb);
			sb_printf(&tb, "%s%s(%s)", j ? ", " : "", F->ft, s2);
		}
		{
			struct sbuf sc = {0};
			sb_printf(&sc, "asfloat(asuint([%s][asuint(j) & %d]) + "
			    "((asuint(j) << %d) & %s))", sb_str(&tb), N - 1,
			    F->mant - tab_t, F->expmask);
			bl_add(&l, "sc", sb_str(&sc));
			free(sc.p);
		}
		free(tb.p);
		if (fm)
			sb_printf(&body, "fma(sc, %s, sc)", sb_str(&b1));
		else
			sb_printf(&body, "sc + sc*(%s)", sb_str(&b1));
		snprintf(shape, ns, "2^(k/%d)*(1 + q(r)), %d-entry table", N,
		    N);
		deg = L + 1;
		break;
	}
	case R_LOG:
		/* x = 2^e m, m in [sqrt(1/2), sqrt(2)); log = e ln2 + log1p(m-1) */
		if (!core_make(&k1, l_log1p, rnd(wb, sqrtl(0.5L)) - 1,
		    rnd(wb, sqrtl(2.0L)) - 1, 1, 1, 0, L + 1, "m", "s"))
			goto fail;
		{
			double cc[2];
			cw_parts(ln2_3, A.prec - 11, 2, cc);
			c1 = cc[0];
			c2 = cc[1];
		}
		snprintf(t, sizeof t, "asuint(x) - %s", F->sqrthalf);
		bl_add(&l, "u", t);
		snprintf(t, sizeof t, "%s(%s(u) >> %d)", F->ft, F->it, F->mant);
		bl_add(&l, "ex", t);
		snprintf(t, sizeof t, "asfloat(asuint(x) - (u & %s)) - 1.0",
		    F->expmask);
		bl_add(&l, "m", t);
		core_text(&k1, &l, &b1);
		kfmt(s1, sizeof s1, c1, wb);
		kfmt(s2, sizeof s2, c2, wb);
		if (fm)
			sb_printf(&body, "fma(ex, %s, fma(ex, %s, %s))", s1, s2,
			    sb_str(&b1));
		else
			sb_printf(&body, "ex*%s + (%s + ex*%s)", s1, sb_str(&b1), s2);
		snprintf(shape, ns, "e*ln2 + m*P(m), exponent split");
		deg = L + 1;
		break;
	case R_SQRT:
		/* x = 4^k m, m in [1/2, 2); sqrt x = 2^k sqrt m */
		if (!core_make(&k1, A.fn == F_SQRT ? l_sqrt : l_rsqrt, 0.5L, 2,
		    0, 1, 1, L + 1, "m", "s"))
			goto fail;
		bl_add(&l, "u", "asuint(x)");
		snprintf(t, sizeof t, "((u >> %d) + %d) >> 1", F->mant,
		    F->bias + 3);
		bl_add(&l, "kb", t);
		snprintf(t, sizeof t, "asfloat(u - ((kb - %d) << %d))",
		    F->bias + 1, F->mant + 1);
		bl_add(&l, "m", t);
		core_text(&k1, &l, &b1);
		sb_printf(&body, "asfloat(asuint(%s) %s ((kb - %d) << %d))",
		    sb_str(&b1), A.fn == F_SQRT ? "+" : "-", F->bias + 1,
		    F->mant);
		snprintf(shape, ns, "2^k * P(m), exponent halved");
		deg = L;
		break;
	case R_SQRTN:
		/* a short reduced seed for 1/sqrt, then one Newton step */
		if (L > (A.w == 64 ? 18 : 10) || !core_make(&k1, l_rsqrt, 0.5L, 2, 0, 1, 1, L + 1,
		    "m", "s"))
			goto fail;
		bl_add(&l, "u", "asuint(x)");
		snprintf(t, sizeof t, "((u >> %d) + %d) >> 1", F->mant,
		    F->bias + 3);
		bl_add(&l, "kb", t);
		snprintf(t, sizeof t, "asfloat(u - ((kb - %d) << %d))",
		    F->bias + 1, F->mant + 1);
		bl_add(&l, "m", t);
		core_text(&k1, &l, &b1);
		{
			struct sbuf y = {0};
			sb_printf(&y, "asfloat(asuint(%s) - ((kb - %d) << %d))",
			    sb_str(&b1), F->bias + 1, F->mant);
			bl_add(&l, "y0", sb_str(&y));
			free(y.p);
		}
		bl_add(&l, "h", "0.5*x");
		/* y0 + y0 (1/2 - h y0^2): the correction is small */
		if (fm)
			sb_printf(&body, "%sfma(y0, fma(-(h*y0), y0, 0.5), y0)%s",
			    A.fn == F_SQRT ? "x*" : "", "");
		else
			sb_printf(&body, "%s(y0 + y0*(0.5 - h*y0*y0))%s",
			    A.fn == F_SQRT ? "x*" : "", "");
		snprintf(shape, ns, "2^-k * P(m) seed + 1 Newton step");
		deg = L;
		break;
	case R_NEWTON1:
	case R_NEWTON2:
		/* the bit-hack seed and Newton steps for 1/sqrt */
		if (L > 1)
			goto fail;
		snprintf(t, sizeof t, "asfloat(%s - (asuint(x) >> 1))", F->magic);
		bl_add(&l, "y0", t);
		bl_add(&l, "h", "0.5*x");
		bl_add(&l, "y1", "y0*(1.5 - h*y0*y0)");
		if (r == R_NEWTON2)
			bl_add(&l, "y2", "y1*(1.5 - h*y1*y1)");
		sb_printf(&body, "%s%s", A.fn == F_SQRT ? "x*" : "",
		    r == R_NEWTON2 ? "y2" : "y1");
		snprintf(shape, ns, "bit hack + %d Newton step%s",
		    r == R_NEWTON2 ? 2 : 1, r == R_NEWTON2 ? "s" : "");
		deg = 0;
		goto out;
	case R_ATAN:
		/* |x| > 1: atan x = pi/2 - atan(1/x); sign by copysign */
		if (!core_make(&k1, l_atan, 0, 1.0001L, 1, 2, 0, L + 1, "r", "s"))
			goto fail;
		bl_add(&l, "a", "abs(x)");
		bl_add(&l, "r", "a > 1.0 ? 1.0/a : a");
		core_text(&k1, &l, &b1);
		bl_add(&l, "p", sb_str(&b1));
		kfmt(s1, sizeof s1, rnd(wb, PI_L / 2), wb);
		sb_printf(&body, "copysign(a > 1.0 ? %s - p : p, x)", s1);
		snprintf(shape, ns, "|x|>1: pi/2 - atan(1/|x|), copysign");
		deg = 2 * L + 1;
		break;
	case R_HYP:
		/*
		 * through e = exp(|x|): cosh = (e + 1/e)/2; sinh = (e - 1/e)/2
		 * and tanh = 1 - 2/(e^2 + 1) cancel near 0, so an odd
		 * polynomial below a cut (sign by copysign)
		 */
		bl_add(&l, "a", "abs(x)");
		if (A.fn == F_COSH) {
			if (L > 1)
				goto fail;
			bl_add(&l, "e", "exp(a)");
			sb_puts(&body, "0.5*(e + 1.0/e)");
			snprintf(shape, ns, "(e + 1/e)/2, e = exp(|x|)");
			deg = 0;
			goto out;
		}
		R = A.fn == F_TANH ? 0.55L : 0.5L;
		if (!core_make(&k1, A.fn == F_TANH ? l_tanh : l_sinh, 0,
		    R * 1.0001L, 1, 2, 0, L + 1, "a", "s"))
			goto fail;
		core_text(&k1, &l, &b1);
		bl_add(&l, "p", sb_str(&b1));
		kfmt(s1, sizeof s1, rnd(wb, R), wb);
		if (A.fn == F_TANH) {
			bl_add(&l, "e", "exp(2.0*a)");
			sb_printf(&body, "copysign(a < %s ? p : 1.0 - 2.0/(e + 1.0), "
			    "x)", s1);
			snprintf(shape, ns, "P | 1 - 2/(exp(2|x|) + 1)");
		} else {
			bl_add(&l, "e", "exp(a)");
			sb_printf(&body, "copysign(a < %s ? p : 0.5*(e - 1.0/e), x)",
			    s1);
			snprintf(shape, ns, "P | (e - 1/e)/2, e = exp(|x|)");
		}
		deg = 2 * L + 1;
		break;
	default:
		goto fail;
	}
out:
	(void)s3;
	(void)s4;
	compose(full, sb_str(&body), &l);
	bl_free(&l);
	free(b1.p);
	free(b2.p);
	free(body.p);
	core_free(&k1);
	core_free(&k2);
	return deg ? deg : -1;
fail:
	bl_free(&l);
	free(b1.p);
	free(b2.p);
	free(body.p);
	core_free(&k1);
	core_free(&k2);
	return 0;
}

/*
 * R_EXPTAB's table at core degree L + 1: the smallest of 2 to 256 entries
 * at which q's error, carried to the result, is under an eighth of a unit
 * roundoff, so a larger table would only cost memory; the largest when
 * none is.  The error is Taylor's bound, R^(L+2)/(L+2)! e^2R with e^2R <=
 * 1 + 4R, above what the minimax fit reaches: plain arithmetic, so the
 * choice does not move with a platform's libm or its fits.
 */
static int
exptab_bits(int L)
{
	ld M = fmaxl(fabsl(A.lo), fabsl(A.hi)), goal = ldexpl(1, -A.prec - 3);
	int t, i;

	for (t = 1; t < 8; t++) {
		ld R = LN2_L / (2 << t) * 1.001L + M * ldexpl(1, 2 - A.prec),
		    e = 1 + 4 * R;
		for (i = 1; i <= L + 2; i++)
			e *= R / i;
		if (e <= goal)
			return t;
	}
	return 8;
}

static void
reductions(void)
{
	int list[8], n = 0, i, L;
	ld M = fmaxl(fabsl(A.lo), fabsl(A.hi));

	if (!A.w || !A.fn || A.o.haswt)
		return;
	switch (A.fn) {
	case F_SIN:
		if (M > PI_L / 4)
			list[n++] = R_QUAD;
		break;
	case F_COS:
		if (M > PI_L / 4)
			list[n++] = R_QUAD;
		list[n++] = R_COSSHIFT;
		list[n++] = R_COSHALF;
		break;
	case F_TAN:
		if (M < PI_L / 2)
			list[n++] = R_SINCOS;
		if (M < PI_L)
			list[n++] = R_TANHALF;
		break;
	case F_SEC:
	case F_CSC:
	case F_COT:
		if (M < PI_L / 2)
			list[n++] = R_SINCOS;
		break;
	case F_EXP:
		list[n++] = R_EXP;
		list[n++] = R_EXPTAB;
		break;
	case F_LOG:
		if (A.lo > 0)
			list[n++] = R_LOG;
		break;
	case F_SQRT:
	case F_RSQRT:
		if (A.lo > 0) {
			list[n++] = R_SQRT;
			list[n++] = R_SQRTN;
			list[n++] = R_NEWTON1;
			list[n++] = R_NEWTON2;
		}
		break;
	case F_ATAN:
		if (M > 1)
			list[n++] = R_ATAN;
		break;
	case F_SINH:
	case F_TANH:
		if (M > 0.6L)
			list[n++] = R_HYP;
		break;
	case F_COSH:
		list[n++] = R_HYP;
		break;
	}
	for (i = 0; i < n; i++) {
		double prev = INFINITY;
		for (L = 1; L <= (A.w == 64 ? 21 : 12); L++) {
			struct sbuf full = {0};
			char shape[40];
			struct cand *c;
			int deg;
			if (list[i] == R_EXPTAB)
				tab_t = exptab_bits(L);
			deg = reduction(list[i], L, &full, shape, sizeof shape);
			double m;
			const char *kind, *form;
			if (deg == 0) {
				free(full.p);
				break;
			}
			kind = list[i] == R_HYP ? "identity" : deg < 0 ? "bithack" :
			    list[i] == R_SINCOS || list[i] == R_COSSHIFT ||
			    list[i] == R_COSHALF || list[i] == R_TANHALF ? "identity" :
			    list[i] == R_SQRTN ? "seed+nwt" : "reduced";
			form = deg < 0 ? "-" : A.o.fma ? "horner+fma" : "horner";
			c = submit(kind, shape, form, deg, 0, sb_str(&full));
			free(full.p);
			if (c && deg > 0 && A.w && A.o.search.kind) {
				/* the same reduction, its cores searched; kept
				 * only if the table's measure agrees */
				struct sbuf f2 = {0};
				char sh2[40];
				struct cand *c2 = NULL;
				double mw = cmetric(c);
				memset(&RS, 0, sizeof RS);
				RS.on = 1;
				reduction(list[i], L, &f2, sh2, sizeof sh2);
				RS.on = 0;
				if (RS.moved)
					c2 = submit(kind, sh2, form, deg, 0, sb_str(&f2));
				free(f2.p);
				if (RS.nsx > 0)
					c = search_settle(c2, mw, &RS.sr, RS.nsx);
			}
			if (c == NULL)
				break;
			m = cmetric(c);
			if (deg < 0)
				break;
			if (A.o.tol_ulp > 0 && !c->bad && c->eulp <= A.o.tol_ulp)
				break;
			if (A.o.tol_err > 0 && !c->bad && (A.rel ? c->erel :
			    c->eabs) <= A.o.tol_err)
				break;
			if (!(m < 0.8 * prev) && L > 2)
				break;
			if (A.rel && m < 0.5 + ldexpl(1, -4) && L > 1 &&
			    m >= 0.95 * prev)
				break;
			prev = m;
		}
	}
}

/* ---- C output ---- */

static const char *
cty(ty_t t)
{
	switch (TYK(t)) {
	case K_FLOAT: return TYW(t) == 32 ? "float" : "double";
	case K_REAL: return "double";
	case K_UINT: return TYW(t) <= 32 ? "uint32_t" : "uint64_t";
	case K_SINT: return TYW(t) <= 32 ? "int32_t" : "int64_t";
	case K_BOOL: return "int";
	}
	return "int64_t";
}

/* emit_c's tables: printed as NAME_t0, NAME_t1, ... */
static struct {
	const char *name;
	nid tab[8];
	int n;
} CT;

static void
c_expr(struct sbuf *b, nid n, const int32_t *tmp, int *helpers)
{
	ty_t t = N.ty[n];
	int f32 = TYK(t) == K_FLOAT && TYW(t) == 32, op = N.op[n], i;
	const char *sfx = f32 ? "f" : "";
	char k[48];

	if (tmp[n] >= 0) {
		sb_printf(b, "t%d", tmp[n]);
		return;
	}
	switch (op) {
	case O_INDEX:
		/* a table: a static array emit_c prints before the function */
		for (i = 0; i < CT.n && CT.tab[i] != N.a[n]; i++)
			;
		sb_printf(b, "%s_t%d[", CT.name, i);
		c_expr(b, N.b[n], tmp, helpers);
		sb_putc(b, ']');
		return;
	case O_CONST:
		if (TY_FP(t)) {
			kfmt(k, sizeof k, N.kr[n], f32 ? 32 : 64);
			sb_printf(b, "%s%s", k, sfx);
		} else if (TYK(t) == K_UINT)
			sb_printf(b, "0x%llx%s", (unsigned long long)N.ki[n] &
			    (TYW(t) >= 64 ? ~0ULL : (1ULL << TYW(t)) - 1),
			    TYW(t) > 32 ? "ull" : "u");
		else
			sb_printf(b, "%lld", (long long)N.ki[n]);
		return;
	case O_VAR:
		sb_puts(b, "x");
		return;
	case O_NEG:
		sb_puts(b, "-");
		c_expr(b, N.a[n], tmp, helpers);
		return;
	case O_CAST:
		sb_printf(b, "(%s)", cty(t));
		c_expr(b, N.a[n], tmp, helpers);
		return;
	case O_ASUINT:
	case O_ASINT:
		*helpers |= TYW(t) == 32 ? 1 : 2;
		sb_printf(b, "%sucas_asuint%d(", op == O_ASINT ? (TYW(t) == 32 ?
		    "(int32_t)" : "(int64_t)") : "", TYW(t));
		c_expr(b, N.a[n], tmp, helpers);
		sb_putc(b, ')');
		return;
	case O_ASFLOAT:
		*helpers |= TYW(t) == 32 ? 1 : 2;
		sb_printf(b, "ucas_asfloat%d(", TYW(t));
		c_expr(b, N.a[n], tmp, helpers);
		sb_putc(b, ')');
		return;
	case O_SEL:
		sb_putc(b, '(');
		c_expr(b, N.a[n], tmp, helpers);
		sb_puts(b, " ? ");
		c_expr(b, N.b[n], tmp, helpers);
		sb_puts(b, " : ");
		c_expr(b, N.c[n], tmp, helpers);
		sb_putc(b, ')');
		return;
	}
	if (opinfo[op].arity == 2 && !(op == O_MIN || op == O_MAX ||
	    op == O_COPYSIGN || op == O_POW)) {
		static const char *const sym[O_NOPS] = {
			[O_ADD] = "+", [O_SUB] = "-", [O_MUL] = "*",
			[O_DIV] = "/", [O_MOD] = "%", [O_AND] = "&",
			[O_OR] = "|", [O_XOR] = "^", [O_SHL] = "<<",
			[O_SHR] = ">>", [O_EQ] = "==", [O_NE] = "!=",
			[O_LT] = "<", [O_LE] = "<=", [O_GT] = ">",
			[O_GE] = ">=", [O_LAND] = "&&", [O_LOR] = "||",
		};
		sb_putc(b, '(');
		c_expr(b, N.a[n], tmp, helpers);
		sb_printf(b, " %s ", sym[op] ? sym[op] : "?");
		c_expr(b, N.b[n], tmp, helpers);
		sb_putc(b, ')');
		return;
	}
	switch (op) {
	case O_ABS: sb_printf(b, "fabs%s(", sfx); break;
	case O_MIN: sb_printf(b, "fmin%s(", sfx); break;
	case O_MAX: sb_printf(b, "fmax%s(", sfx); break;
	case O_LNOT: sb_puts(b, "!("); break;
	case O_NOT: sb_puts(b, "~("); break;
	case O_SEC: sb_printf(b, "1%s/cos%s(", f32 ? ".0f" : ".0", sfx); break;
	case O_CSC: sb_printf(b, "1%s/sin%s(", f32 ? ".0f" : ".0", sfx); break;
	case O_COT: sb_printf(b, "1%s/tan%s(", f32 ? ".0f" : ".0", sfx); break;
	default: sb_printf(b, "%s%s(", opinfo[op].name, sfx);
	}
	c_expr(b, N.a[n], tmp, helpers);
	if (opinfo[op].arity >= 2) {
		sb_puts(b, ", ");
		c_expr(b, N.b[n], tmp, helpers);
	}
	if (opinfo[op].arity == 3) {
		sb_puts(b, ", ");
		c_expr(b, N.c[n], tmp, helpers);
	}
	sb_putc(b, ')');
}

static void
emit_c(struct func *f, const char *name)
{
	uint32_t *uses = xcalloc(N.n, sizeof *uses);
	int32_t *tmp = xmalloc(N.n * sizeof *tmp);
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), g;
	uint32_t sp = 0;
	int nt = 0, helpers = 0, ti;
	struct sbuf b = {0}, e = {0};
	ty_t rt = N.ty[f->body];

	memset(tmp, 0xff, N.n * sizeof *tmp);
	CT.name = name;
	CT.n = 0;
	st[sp++] = f->body;
	mark[f->body] = 1;
	while (sp) {
		nid ch[3];
		int i;
		g = st[--sp];
		if (N.op[g] == O_CONST || N.op[g] == O_VAR)
			continue;
		if (N.op[g] == O_TABLE) {
			/* its entries are constants, written in the array */
			if (CT.n == (int)(sizeof CT.tab / sizeof CT.tab[0])) {
				printf("/* %s: more than %d tables, not written as "
				    "C */\n", name, CT.n);
				goto out;
			}
			CT.tab[CT.n++] = g;
			continue;
		}
		ch[0] = N.a[g];
		ch[1] = N.b[g];
		ch[2] = N.c[g];
		for (i = 0; i < 3; i++) {
			if (!ch[i])
				continue;
			uses[ch[i]]++;
			if (!mark[ch[i]]) {
				mark[ch[i]] = 1;
				st[sp++] = ch[i];
			}
		}
	}
	/* shared subterms become locals, in topological (id) order */
	for (g = 1; g < N.n; g++) {
		if (!mark[g] || uses[g] < 2 || N.op[g] == O_CONST ||
		    N.op[g] == O_VAR || N.op[g] == O_TABLE || g == f->body)
			continue;
		sb_printf(&b, "\tconst %s t%d = ", cty(N.ty[g]), nt);
		c_expr(&b, g, tmp, &helpers);
		sb_puts(&b, ";\n");
		tmp[g] = nt++;
	}
	c_expr(&e, f->body, tmp, &helpers);
	if (helpers & 1)
		printf("#ifndef UCAS_BITCAST32\n#define UCAS_BITCAST32\n"
		    "static inline uint32_t ucas_asuint32(float f) "
		    "{ uint32_t u; memcpy(&u, &f, 4); return u; }\n"
		    "static inline float ucas_asfloat32(uint32_t u) "
		    "{ float f; memcpy(&f, &u, 4); return f; }\n#endif\n");
	if (helpers & 2)
		printf("#ifndef UCAS_BITCAST64\n#define UCAS_BITCAST64\n"
		    "static inline uint64_t ucas_asuint64(double f) "
		    "{ uint64_t u; memcpy(&u, &f, 8); return u; }\n"
		    "static inline double ucas_asfloat64(uint64_t u) "
		    "{ double f; memcpy(&f, &u, 8); return f; }\n#endif\n");
	for (ti = 0; ti < CT.n; ti++) {
		nid tb = CT.tab[ti];
		ty_t et = N.ty[tb];
		uint32_t j;
		printf("static const %s %s_t%d[%u] = {", cty(et), name, ti,
		    (unsigned)N.b[tb]);
		for (j = 0; j < N.b[tb]; j++) {
			char k[48];
			nid v = N.lst[N.a[tb] + j];
			int w = TYW(et) == 32 ? 32 : 64;
			if (TY_FP(et))
				kfmt(k, sizeof k, N.kr[v], w);
			else
				snprintf(k, sizeof k, "%lld",
				    (long long)N.ki[v]);
			printf("%s%s%s%s", j ? "," : "", j % 4 ? " " :
			    "\n\t", k, TYK(et) == K_FLOAT && w == 32 ?
			    "f" : "");
		}
		printf("\n};\n");
	}
	printf("static inline %s\n%s(%s x)\n{\n%s\treturn %s;\n}\n", cty(rt),
	    name, cty(f->p[0].ty),
	    b.p ? b.p : "", sb_str(&e));
out:
	free(b.p);
	free(e.p);
	free(uses);
	free(tmp);
	free(mark);
	free(st);
}

/* ---- ranking and output ---- */

static int
cand_cmp(const void *pa, const void *pb)
{
	const struct cand *a = pa, *b = pb;
	double ma = cmetric(a), mb = cmetric(b);

	if (a->meets != b->meets)
		return b->meets - a->meets;
	if (a->useless != b->useless)
		return a->useless - b->useless;
	if (a->front != b->front)
		return b->front - a->front;
	if (a->cost != b->cost)
		return a->cost < b->cost ? -1 : 1;
	if (ma != mb)
		return ma < mb ? -1 : 1;
	return strcmp(a->text, b->text);
}

static void
rank(void)
{
	int i, j;

	/* an explicit degree range applies to every kind of candidate */
	if (A.o.dmin > 0 || A.o.dmax > 0) {
		for (i = j = 0; i < A.nc; i++) {
			int d = A.cd[i].deg;
			if (d >= 0 && ((A.o.dmin && d < A.o.dmin) ||
			    (A.o.dmax && d > A.o.dmax))) {
				free(A.cd[i].text);
				continue;
			}
			A.cd[j++] = A.cd[i];
		}
		A.nc = j;
	}

	for (i = 0; i < A.nc; i++) {
		struct cand *c = &A.cd[i];
		double m = cmetric(c);
		c->useless = c->bad || (A.rel ? c->erel >= 0.1 :
		    c->eabs >= 0.1 * (double)A.fmax);
		c->front = !c->useless;
		for (j = 0; j < A.nc && c->front; j++) {
			const struct cand *d = &A.cd[j];
			double md = cmetric(d);
			if (j == i || d->useless)
				continue;
			if (md <= m && d->cost <= c->cost && (md < m ||
			    d->cost < c->cost || j < i))
				c->front = 0;
		}
		c->meets = 0;
		if (A.o.tol_ulp > 0 && A.w)
			c->meets = !c->bad && c->eulp <= A.o.tol_ulp;
		if (A.o.tol_err > 0)
			c->meets = !c->bad && (A.rel ? c->erel : c->eabs) <=
			    A.o.tol_err;
		if (c->meets)
			c->front = 1;
	}
	qsort(A.cd, A.nc, sizeof *A.cd, cand_cmp);
}

/*
 * A candidate's text for the listing: a table of more than 8 entries (exp's
 * reduction holds up to 256) by its first two and its last; `show` prints
 * it whole, as do the JSON and `as c`.
 */
static void
elide_tables(const char *s, struct sbuf *out)
{
	sb_clear(out);
	while (*s) {
		const char *e = s, *c2 = NULL, *last = NULL;
		int d = 0, n = 1;
		if (*s != '[') {
			sb_putc(out, *s++);
			continue;
		}
		for (e = s + 1; *e && !(d == 0 && *e == ']'); e++) {
			if (*e == '(' || *e == '[')
				d++;
			else if (*e == ')' || *e == ']')
				d--;
			else if (d == 0 && *e == ',') {
				if (++n == 3)
					c2 = e;
				last = e;
			}
		}
		if (!*e || n <= 8) {
			sb_printf(out, "%.*s", (int)(e - s + (*e != 0)), s);
			s = e + (*e != 0);
			continue;
		}
		sb_printf(out, "%.*s, ... %d more ...%.*s", (int)(c2 - s), s,
		    n - 3, (int)(e - last + 1), last);
		s = e + 1;
	}
}

static void
print_rows(int from, int to)
{
	int i;
	char nm[48], deg[16], ulp[48];
	struct sbuf shown = {0};

	for (i = from; i < to && i < A.nc; i++) {
		struct cand *c = &A.cd[i];
		struct func *f;
		struct acert ac;
		if (c->qdeg)
			snprintf(deg, sizeof deg, "%d/%d", c->deg, c->qdeg);
		else if (c->deg >= 0)
			snprintf(deg, sizeof deg, "%d", c->deg);
		else
			snprintf(deg, sizeof deg, "-");
		if (c->bad)
			snprintf(ulp, sizeof ulp, "NaN/inf at %d points", c->bad);
		else if (A.w && c->naive >= 0 && A.rel)
			snprintf(ulp, sizeof ulp, "%.3g ulp (%.3g)", c->eulp,
			    c->naive);
		else if (A.w && A.rel)
			snprintf(ulp, sizeof ulp, "%.3g ulp", c->eulp);
		else if (A.w)
			snprintf(ulp, sizeof ulp, "%.3g abs", c->eabs);
		else
			ulp[0] = 0;
		printf("  %-3d %-9s %-5s %-11s %-24s %-13.3g %-13.3g %-4g at "
		    "x=%.9g", i + 1, c->kind, deg, c->form, ulp, c->erel,
		    c->eabs, c->cost, c->xw);
		if (c->keepok >= 0)
			printf("  exact pts %s", c->keepok ? "yes" : "NO");
		if (c->freeerr > 0 && c->rerr > 0)
			printf("  keep +%.0f%%", 100 * (c->rerr / c->freeerr - 1));
		if (c->meets)
			printf("  *");
		printf("\n");
		snprintf(nm, sizeof nm, "%s%d", A.o.name, i + 1);
		elide_tables(c->text, &shown);
		printf("      %s: %s\n", c->shape, sb_str(&shown));
		f = define(nm, c->text);
		memset(&ac, 0, sizeof ac);
		if (c->skind && c->skept && A.w) {
			/* what the search kept, certified on the whole interval */
			if (f)
				acert_run(A.f.body, f->body, A.lo, A.hi, A.w, A.rel,
				    A.k0, A.c, &ac);
			else
				snprintf(ac.why, sizeof ac.why, "the candidate did not "
				    "parse back");
		}
		if (c->skind) {
			const char *un = A.rel ? "ulp" : "abs";
			printf("      search: %s over the coefficients' ulp offsets "
			    "from the walk's, seed %llu, %ld evaluations on %d "
			    "sample points: %.4g -> %.4g %s there", ns_name(c->skind),
			    (unsigned long long)A.o.search.seed, c->sevals, c->snsx,
			    c->sbefore, c->safter, un);
			if (c->sms < 0)
				printf(" (nothing better found)\n");
			else if (c->skept && ac.ok)
				printf("; on the table's measure %.4g -> %.4g %s: "
				    "kept, certified below\n", c->smw, c->sms, un);
			else if (c->skept)
				printf("; on the table's measure %.4g -> %.4g %s: "
				    "kept (measured%s, not certified: %s)\n", c->smw,
				    c->sms, un, A.exhaustive ? " on every float" :
				    " on that sample", ac.why);
			else
				printf("; on the table's measure %.4g against the "
				    "walk's %.4g %s: not kept\n", c->sms, c->smw, un);
			if (ac.ok)
				printf("      certified: at most %.4g %s for every %s x in "
				    "[%.9g, %.9g], %ld pieces: the coefficients' error at "
				    "most %.4g (%.4g reached at a piece's middle), rounding "
				    "the rest (each operation's result (1 + d), |d| <= u, "
				    "carried through); Taylor of order %d on intervals, "
				    "libm's long double functions taken within 2^%d%s\n",
				    ac.bound, un, fmt_name(A.w), (double)A.lo,
				    (double)A.hi, ac.pieces, ac.approx, ac.reached,
				    ac.order, ac.slack, ac.mexp ? ", the candidate's exp "
				    "(libm's double exp) within 2^-52" : "");
		}
		rep_obj(NULL);
		rep_int("rank", i + 1);
		rep_str("name", nm);
		rep_str("kind", c->kind);
		rep_str("degree", deg);
		rep_str("form", c->form);
		rep_str("shape", c->shape);
		rep_str("expr", c->text);
		rep_num("max_rel_error", c->erel);
		rep_num("max_abs_error", c->eabs);
		if (A.w && A.rel && !c->bad)
			rep_num("max_ulp", c->eulp);
		if (A.w && A.rel && !c->bad && c->naive >= 0)
			rep_num("naive_ulp", c->naive);
		rep_num("worst_at", c->xw);
		rep_num("cost", c->cost);
		if (c->keepok >= 0)
			rep_bool("exact_points_kept", c->keepok);
		if (c->bad)
			rep_int("bad_points", c->bad);
		rep_bool("meets_target", c->meets);
		if (c->skind) {
			rep_obj("search");
			rep_str("method", c->skind == NS_ANNEAL ? "anneal" : "cmaes");
			rep_int("seed", (long long)A.o.search.seed);
			rep_int("evaluations", c->sevals);
			rep_int("sample_points", c->snsx);
			rep_num("sample_before", c->sbefore);
			rep_num("sample_after", c->safter);
			rep_num("measure_walk", c->smw);
			if (c->sms >= 0)
				rep_num("measure_search", c->sms);
			rep_bool("kept", c->skept);
			rep_str("label", ac.ok ? "certified" :
			    "measured, not certified");
			if (ac.ok) {
				rep_obj("certificate");
				rep_num("bound", ac.bound);
				rep_num("approximation", ac.approx);
				rep_num("reached", ac.reached);
				rep_int("pieces", ac.pieces);
				rep_int("taylor_order", ac.order);
				rep_end();
			} else if (c->skept && ac.why[0])
				rep_str("not_certified", ac.why);
			rep_end();
		}
		rep_end();
		if (f && A.o.emitc)
			emit_c(f, nm);
		if (f && A.w && i == 0 && from == 0) {
			/* how much of the best one's error can be rounding */
			struct rounderr re;
			rounderr_measure(f, 0, &re);
			rounderr_line(&re, "      ", nm);
		}
	}
	if (to > A.nc)
		to = A.nc;
	A.shown = to;
	free(shown.p);
}

static void
print_list(void)
{
	int n = A.o.list, fr = 0, i;

	for (i = 0; i < A.nc; i++)
		if (A.cd[i].front)
			fr++;
	if (n <= 0) {
		int mt = 0;
		for (i = 0; i < A.nc; i++)
			mt += A.cd[i].meets;
		n = fr < 8 ? fr : 8;
		if (mt)
			n = mt + 2 < 6 ? mt + 2 : 6;
		if (n == 0)
			n = 3;
		if (nlp_defer && n > 3)
			n = 3;   /* chat: the best three; 'approx more' for the rest */
	}
	printf("  candidates ranked by %s and cost under %s (%s%s); defined "
	    "as %s1..:\n", metric_name(), sym_str(model_active()->name),
	    A.w ? (A.exhaustive ? "every float" : "sampled floats") :
	    "real arithmetic", A.w ? "" : ", no rounding", A.o.name);
	if (A.w && A.rel)
		printf("  measured on %u %s in [%.9g, %.9g]%s; ulp against the "
		    "exact value (naive rounding in parentheses)\n", A.ns,
		    fmt_name(A.w), (double)A.lo, (double)A.hi, A.exhaustive ?
		    " (all of them)" : "");
	else if (A.w)
		printf("  measured on %u %s in [%.9g, %.9g]%s; absolute error: "
		    "the target reaches 0 here, where relative error and ulp "
		    "are unbounded\n", A.ns, fmt_name(A.w), (double)A.lo,
		    (double)A.hi, A.exhaustive ? " (all of them)" : "");
	printf("  #   kind      deg   form        %-24s %-13s %-13s cost\n",
	    A.w ? "error (naive)" : "", "max rel", "max abs");
	rep_begin("approx");
	rep_str("function", A.fname);
	rep_num("lo", (double)A.lo);
	rep_num("hi", (double)A.hi);
	rep_str("metric", metric_name());
	rep_str("model", sym_str(model_active()->name));
	rep_str("format", A.w ? fmt_name(A.w) : "real");
	rep_int("sample_points", A.ns);
	rep_bool("exhaustive", A.exhaustive);
	rep_list("candidates");
	print_rows(0, n);
	rep_end();
	rep_int("more", A.nc - A.shown);
	if (A.shown < A.nc)
		printf("  %d more: 'approx more [n]'%s\n", A.nc - A.shown,
		    fr > A.shown ? "" : " (dominated or worse)");
}

static void
approx_more(int n)
{
	if (!A.valid)
		fail("no approx results yet");
	if (A.shown >= A.nc) {
		printf("no more candidates\n");
		return;
	}
	rep_begin("approx");
	rep_list("candidates");
	print_rows(A.shown, A.shown + n);
	rep_end();
	rep_int("more", A.nc - A.shown);
	if (A.shown < A.nc)
		printf("  %d more\n", A.nc - A.shown);
}

/* ---- setup ---- */

static void
reset(void)
{
	int i;

	for (i = 0; i < A.nc; i++)
		free(A.cd[i].text);
	A.nc = 0;
	A.shown = 0;
	A.valid = 0;
	A.stalled = 0;
	if (A.ref.n || A.ref.dp.n)
		lref_free(&A.ref);
	if (A.wref.n || A.wref.dp.n)
		lref_free(&A.wref);
	free(A.xs);
	free(A.ys);
	A.xs = NULL;
	A.ys = NULL;
}

static int
check_sym(int odd)
{
	ld M = fmaxl(fabsl(A.lo), fabsl(A.hi));
	int i, nz = 0;

	for (i = 0; i < 64; i++) {
		ld x = M * (i + 0.5L) / 64, a = lref_eval(&A.ref, x);
		ld b = lref_eval(&A.ref, -x);
		ld d = odd ? a + b : a - b;
		if (!isfinite(a) || !isfinite(b))
			return 0;
		if (!(fabsl(d) <= 1e-13L * (fabsl(a) + fabsl(b)) + 1e-300L))
			return 0;
		if (a != 0)
			nz = 1;
	}
	return nz;
}

static void
setup_shape(void)
{
	ld f0;
	int i, hardzero = 0;

	A.odd = A.even = 0;
	if (A.lo <= 0 && A.hi >= 0 && A.o.sym != 0) {
		if (A.o.sym == -1 || A.o.sym == 1)
			A.odd = check_sym(1);
		if (!A.odd && (A.o.sym == -1 || A.o.sym == 2))
			A.even = check_sym(0);
		if (A.o.sym == 1 && !A.odd)
			printf("  warning: %s is not odd on the interval "
			    "(f(-x) != -f(x)); fitting without symmetry\n",
			    A.fname);
		if (A.o.sym == 2 && !A.even)
			printf("  warning: %s is not even on the interval "
			    "(f(-x) != f(x)); fitting without symmetry\n",
			    A.fname);
	} else if (A.o.sym > 0)
		printf("  warning: odd/even needs 0 in the interval\n");
	if (A.odd || A.even) {
		A.c = 0;
		A.step = 2;
		A.k0 = A.odd;
		A.xa = 0;
		A.xb = fmaxl(fabsl(A.lo), fabsl(A.hi));
		printf("  checked: %s is %s (f(-x) = %sf(x) at 64 points): "
		    "fitting %s on [0, %.9g]\n", A.fname, A.odd ? "odd" : "even",
		    A.odd ? "-" : "", shape_str(A.k0, 2, 0, 0), (double)A.xb);
	} else {
		A.step = 1;
		A.xa = A.lo;
		A.xb = A.hi;
		if (A.o.hasc)
			A.c = A.o.c;
		else if (A.lo <= 0 && A.hi >= 0)
			A.c = 0;
		else if (fabsl(lref_eval(&A.ref, A.lo)) < 1e-300L)
			A.c = A.lo;
		else if (fabsl(lref_eval(&A.ref, A.hi)) < 1e-300L)
			A.c = A.hi;
		else {
			ld mid = (A.lo + A.hi) / 2;
			int e = ilogbl(mid);
			A.c = ldexpl(roundl(ldexpl(mid, 6 - e)), e - 6);
		}
		A.c = rnd(A.w, A.c);
		f0 = lref_eval(&A.ref, A.c);
		A.k0 = 0;
		if (f0 == 0) {
			/* a simple zero: f(c + h)/h settles as h shrinks */
			ld h = (A.hi - A.lo) * 1e-3L, r1, r2;
			if (A.c + h > A.hi)
				h = -h;
			r1 = lref_eval(&A.ref, A.c + h) / h;
			r2 = lref_eval(&A.ref, A.c + h / 64) / (h / 64);
			if (isfinite(r1) && isfinite(r2) && r1 != 0 &&
			    fabsl(r2 / r1 - 1) < 0.25L) {
				A.k0 = 1;
				printf("  checked: simple zero at x = %.9g: fitting "
				    "%s, exact zero\n", (double)A.c,
				    shape_str(1, 1, A.c, 0));
			} else {
				hardzero = 1;
				printf("  checked: zero at x = %.9g is not simple "
				    "(f(x)/(x - c) is unbounded): no factor\n",
				    (double)A.c);
			}
		}
	}
	/* metric: relative unless there is a zero away from c */
	A.rel = A.o.metric;
	if (A.rel < 0 && A.o.tol_ulp > 0)
		A.rel = 1;	/* an ulp target is a relative one */
	if (A.rel < 0) {
		ld prev = 0, px = 0;
		int bad = 0;
		for (i = 0; i <= 2000; i++) {
			ld x = A.lo + (A.hi - A.lo) * i / 2000;
			ld y = lref_eval(&A.ref, x);
			if (!isfinite(y))
				continue;
			if (fabsl(x - A.c) > (A.hi - A.lo) * 1e-9L) {
				if (y == 0)
					bad = 1;
				if (i && prev != 0 && (y > 0) != (prev > 0) &&
				    !(px <= A.c && A.c <= x))
					bad = 1;
			}
			prev = y;
			px = x;
		}
		A.rel = !bad && !hardzero;
		printf("  metric: %s error%s\n", A.rel ? "relative" : "absolute",
		    A.rel ? "" : hardzero ? " (a zero that a polynomial factor "
		    "cannot match)" : " (the function crosses zero away from "
		    "the expansion point)");
	} else
		printf("  metric: %s error\n", A.rel ? "relative" : "absolute");
}

static void
run(void)
{
	int i;

	printf("approx %s on [%.9g, %.9g] as %s%s\n", A.fname, (double)A.lo,
	    (double)A.hi, fmt_name(A.w), A.o.fma && A.w ? ", fma" : "");
	setup_shape();
	A.fn = recognise(A.f.body);
	if (A.nkeep) {
		printf("  exact at:");
		for (i = 0; i < A.nkeep; i++) {
			ld y = lref_eval(&A.ref, A.keep[i]);
			printf("%s x = %.9g -> %.9g", i ? "," : "",
			    (double)A.keep[i], (double)y);
		}
		printf("\n");
	}
	build_sample();
	direct_polys();
	if (A.o.method_given && !A.o.compare) {
		focus_report();
	} else {
		if (A.o.rational && !A.stalled)
			direct_rationals();
		if (A.o.split && !A.stalled)
			piecewise();
		if (A.o.reduce)
			reductions();
	}
	rank();
	A.valid = 1;
	print_list();
}

/*
 * Zeros or extrema of F at the interval's ends.  A zero makes relative
 * error unbounded there; an extremum is a bound callers may rely on
 * (|sin| <= 1).  Keeping them exact costs a little error elsewhere.  The
 * user decides: asked on a terminal or in the bot (via nlp.c's
 * questions), otherwise a note says how to choose.
 */
static void
ask_endpoints(double lo, double hi)
{
	static const char *const yn[] = {"yes, keep them exact (keep endpoints)",
	    "no, fit freely"};
	ld e[2], v[2], d[2], fm = 0, h = (hi - lo) * 1e-6;
	int i, zero[2], ext[2], any = 0, z = 0, dflt;
	char what[320];
	struct sbuf b = {0};

	e[0] = lo;
	e[1] = hi;
	for (i = 0; i <= 32; i++) {
		ld y = lref_eval(&A.ref, lo + (hi - lo) * i / 32);
		if (isfinite(y) && fabsl(y) > fm)
			fm = fabsl(y);
	}
	if (fm == 0)
		return;
	for (i = 0; i < 2; i++) {
		v[i] = lref_eval(&A.ref, e[i]);
		d[i] = (lref_eval(&A.ref, e[i] + h) - lref_eval(&A.ref, e[i] - h)) /
		    (2 * h);
		zero[i] = isfinite(v[i]) && fabsl(v[i]) <= fm * 1e-12L;
		ext[i] = !zero[i] && isfinite(d[i]) &&
		    fabsl(d[i]) * (hi - lo) <= fm * 1e-9L;
		if (zero[i] || ext[i]) {
			sb_printf(&b, "%sat x = %.9g %s", any ? ", " : "",
			    (double)e[i], zero[i] ? "it is 0" : v[i] > 0 ?
			    "a maximum" : "a minimum");
			any = 1;
			z |= zero[i];
		}
	}
	if (!any) {
		free(b.p);
		return;
	}
	/* fitting freely is cheaper; keeping them is the user's call */
	dflt = 1;
	snprintf(what, sizeof what, "%s has special points at the interval's "
	    "ends (%s)%s; keep them exact (costs some error elsewhere)",
	    A.fname, b.p, z ? ": near a zero, relative error is unbounded "
	    "unless the zero is exact" : ": bounds a caller may rely on");
	free(b.p);
	if (nlp_can_ask()) {
		if (nlp_ask_choice(what, yn, 2, dflt) == 0)
			A.o.keep_ends = 1;
		return;
	}
	if (z)
		printf("  note: %s? not by default: say `keep endpoints` to keep "
		    "them\n", what);
}

void
cmd_approx(struct P *p)
{
	struct func *f;
	struct aopt *o = &A.o;
	int haverange = 0, i;
	double lo = 0, hi = 0;

	if (cmdp_kw(p, "more")) {
		int n = cmdp_at(p, T_NUM) ? (int)cmdp_number(p, NULL) : 5;
		if (!cmdp_eos(p))
			fail("expected: approx more [n]");
		notes_flush();
		approx_more(n);
		return;
	}
	reset();
	memset(o, 0, sizeof *o);
	o->metric = -1;
	o->sym = -1;
	o->fma = -1;
	o->refine = 1;
	ns_defaults(&o->search);
	o->search.kind = NS_NONE;
	o->search.seed = stoch_seed();
	o->list = 0;
	o->npts = 1 << 17;
	o->rational = 1;
	o->method = FM_REMEZ;
	o->keep_crit = 0;
	o->reduce = 1;
	o->split = 1;
	o->fmt = -1;
	snprintf(o->name, sizeof o->name, "ap");
	f = cmdp_func(p, optwords);
	A.f = *f;
	if (A.f.np != 1)
		fail("approx needs a function of one variable (%s has %d)",
		    f->anon ? "the expression" : sym_str(f->name), A.f.np);
	snprintf(A.fname, sizeof A.fname, "%s", A.f.anon ?
	    node_str(A.f.body) : sym_str(A.f.name));
	if (A.f.anon)
		snprintf(A.fname, sizeof A.fname, "%s", node_str(A.f.body));
	if (cmdp_kw(p, "in")) {
		lo = cmdp_number(p, optwords);
		if (!cmdp_tok(p, T_DOTDOT))
			fail("expected '..' in the interval");
		hi = cmdp_number(p, optwords);
		haverange = 1;
	}
	while (!cmdp_eos(p)) {
		cmdp_fail_here(p);
		if (cmdp_kw(p, "degree")) {
			o->dmin = o->dmax = (int)cmdp_number(p, optwords);
			if (cmdp_tok(p, T_DOTDOT))
				o->dmax = (int)cmdp_number(p, optwords);
		} else if (cmdp_kw(p, "error"))
			o->tol_err = cmdp_number(p, optwords);
		else if (cmdp_kw(p, "ulp"))
			o->tol_ulp = cmdp_number(p, optwords);
		else if (cmdp_kw(p, "as")) {
			int any = 0;
			for (;;) {
				if (cmdp_kw(p, "f32") || cmdp_kw(p, "float"))
					o->fmt = 32;
				else if (cmdp_kw(p, "f64") || cmdp_kw(p, "double"))
					o->fmt = 64;
				else if (cmdp_kw(p, "real"))
					o->fmt = 0;
				else if (cmdp_kw(p, "c"))
					o->emitc = 1;
				else
					break;
				any = 1;
			}
			if (!any)
				fail("as f32, f64, real or c");
		} else if (cmdp_kw(p, "relative"))
			o->metric = 1;
		else if (cmdp_kw(p, "absolute"))
			o->metric = 0;
		else if (cmdp_kw(p, "odd"))
			o->sym = 1;
		else if (cmdp_kw(p, "even"))
			o->sym = 2;
		else if (cmdp_kw(p, "nosym"))
			o->sym = 0;
		else if (cmdp_kw(p, "keep")) {
			do {
				if (cmdp_kw(p, "endpoints") || cmdp_kw(p, "ends"))
					o->keep_ends = 1;
				else if (cmdp_kw(p, "crit") || cmdp_kw(p, "critical"))
					o->keep_crit = 1;
				else if (cmdp_kw(p, "none"))
					;
				else if (o->nkeep < AMAXK - 2)
					o->keep[o->nkeep++] = cmdp_number(p,
					    optwords);
				else
					fail("too many exact points");
			} while (cmdp_tok(p, T_COMMA));
		} else if (cmdp_kw(p, "weight")) {
			A.wf = *cmdp_func(p, optwords);
			if (A.wf.np > 1)
				fail("the weight is a function of x");
			o->haswt = 1;
		} else if (cmdp_kw(p, "at")) {
			o->c = cmdp_number(p, optwords);
			o->hasc = 1;
		} else if (cmdp_kw(p, "fma"))
			o->fma = 1;
		else if (cmdp_kw(p, "nofma"))
			o->fma = 0;
		else if (cmdp_kw(p, "norefine"))
			o->refine = 0;
		else if (cmdp_kw(p, "list"))
			o->list = (int)cmdp_number(p, optwords);
		else if (cmdp_kw(p, "points")) {
			o->npts = (int)cmdp_number(p, optwords);
			if (o->npts < 64 || o->npts > (1 << 24))
				fail("points: 64 to 16777216");
		} else if (cmdp_kw(p, "name")) {
			if (!cmdp_word(p, o->name, sizeof o->name))
				fail("name needs an identifier");
		} else if (cmdp_kw(p, "method")) {
			o->method_given = 1;
			if (cmdp_kw(p, "cheb") || cmdp_kw(p, "chebyshev"))
				o->method = FM_CHEB;
			else if (cmdp_kw(p, "econ"))
				o->method = FM_ECON;
			else if (cmdp_kw(p, "minimax") || cmdp_kw(p, "remez"))
				o->method = FM_REMEZ;
			else
				fail("method is cheb, econ or minimax");
		} else if (cmdp_kw(p, "compare") || cmdp_kw(p, "all"))
			o->compare = 1;
		else if (cmdp_kw(p, "free"))
			o->ends_free = 1;
		else if (cmdp_kw(p, "search")) {
			if (cmdp_kw(p, "anneal") || cmdp_kw(p, "annealing"))
				o->search.kind = NS_ANNEAL;
			else if (cmdp_kw(p, "cmaes") || cmdp_kw(p, "cma"))
				o->search.kind = NS_CMAES;
			else if (cmdp_kw(p, "walk"))
				o->search.kind = NS_NONE;
			else
				fail("search is walk (the ulp walk alone, the default), "
				    "anneal or cmaes");
		} else if (cmdp_kw(p, "budget")) {
			double v = cmdp_number(p, optwords);
			if (!(v >= 10) || v > 1e7)
				fail("budget: evaluations from 10 to 10000000, not %g", v);
			o->search.budget = (long)v;
		} else if (cmdp_kw(p, "temp")) {
			o->search.t0 = cmdp_number(p, optwords);
			o->search.t1 = o->search.t0 / 100;
			if (cmdp_tok(p, T_DOTDOT))
				o->search.t1 = cmdp_number(p, optwords);
			if (!(o->search.t0 > 0) || !(o->search.t1 > 0) ||
			    o->search.t1 > o->search.t0)
				fail("temp: T0..T1 with T0 >= T1 > 0 (fractions of the "
				    "starting error), not %g..%g", o->search.t0,
				    o->search.t1);
		} else if (cmdp_kw(p, "pop")) {
			double v = cmdp_number(p, optwords);
			if (!(v >= 4) || v > 1000)
				fail("pop: a population from 4 to 1000, not %g", v);
			o->search.pop = (int)v;
		} else if (cmdp_kw(p, "sigma")) {
			o->search.sigma = cmdp_number(p, optwords);
			if (!(o->search.sigma > 0) || o->search.sigma > 64)
				fail("sigma: a first step from above 0 to 64 ulps, not "
				    "%g", o->search.sigma);
		} else if (cmdp_kw(p, "seed")) {
			double v = cmdp_number(p, optwords);
			if (!(v >= 0) || v > 9e15)
				fail("seed: a whole number, not %g", v);
			o->search.seed = (uint64_t)v;
		} else if (cmdp_kw(p, "norational"))
			o->rational = 0;
		else if (cmdp_kw(p, "noreduce"))
			o->reduce = 0;
		else if (cmdp_kw(p, "nosplit"))
			o->split = 0;
		else
			fail("unknown approx option (see help)");
	}
	notes_flush();
	if (!haverange) {
		if (!A.f.p[0].ranged)
			fail("give an interval: approx F in a..b");
		lo = A.f.p[0].rlo;
		hi = A.f.p[0].rhi;
	}
	if (!(lo < hi) || !isfinite(lo) || !isfinite(hi))
		fail("the interval must be finite and non-empty");
	if (TYK(A.f.p[0].ty) == K_FLOAT && TYW(A.f.p[0].ty) == 16)
		fail("approx fits f32, f64 or real formats: %s is f16, which it "
		    "has no coefficients, error bounds or C for",
		    sym_str(A.f.p[0].name));
	if (o->fmt < 0)
		o->fmt = TYK(A.f.p[0].ty) == K_FLOAT ? TYW(A.f.p[0].ty) : 32;
	if (o->fma < 0)
		o->fma = 1;
	A.w = o->fmt;
	A.prec = A.w == 32 ? 24 : 53;
	A.emin = A.w == 32 ? -126 : -1022;
	A.lo = A.w ? rnd(A.w, lo) : lo;
	A.hi = A.w ? rnd(A.w, hi) : hi;
	lref_init(&A.ref, A.f.body, 0);
	if (o->haswt)
		lref_init(&A.wref, A.wf.body, 0);
	if (o->keep_crit) {
		/* extrema stay exact: the critical points inside the interval */
		double cr[8];
		int nc = crit_roots(&A.f, lo, hi, cr, 8);
		printf("  keep crit: ");
		if (nc == 0)
			printf("no critical points inside [%.9g, %.9g]", lo, hi);
		for (i = 0; i < nc; i++) {
			printf("%sx = %.9g", i ? ", " : "", cr[i]);
			if (o->nkeep < AMAXK - 2)
				o->keep[o->nkeep++] = cr[i];
		}
		putchar('\n');
	}
	if (!o->keep_ends && !o->ends_free && !o->haswt)
		ask_endpoints(lo, hi);
	if (nlp_asked)
		return;   /* the answer comes back with the message (nlp.c) */
	A.nkeep = 0;
	if (o->keep_ends) {
		A.keep[A.nkeep++] = A.lo;
		A.keep[A.nkeep++] = A.hi;
	}
	for (i = 0; i < o->nkeep; i++) {
		ld x = A.w ? rnd(A.w, o->keep[i]) : o->keep[i];
		if (x < A.lo || x > A.hi)
			fail("exact point %.9g is outside the interval",
			    o->keep[i]);
		A.keep[A.nkeep++] = x;
	}
	run();
}
