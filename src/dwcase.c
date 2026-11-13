/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * dwcase.c: the bounds of dword.c's double-word additions whatever
 * cancels, by a case analysis this program runs instead of one written
 * by hand.
 *
 * The error of DWPlusFP or AccurateDWPlusDW relative to the exact sum is
 * the same for the operands scaled by a power of two (away from overflow
 * and underflow) and for both negated, so one operand's high word is
 * taken in [1, 2), the other's in a binade 2^d of either sign, d from 0
 * down to DW_DLO and a last case for the binades below.  A pair is
 * normalized, xh = RN(xh + xl), so |xl| <= ulp(xh) / 2.  Each case is cut
 * further by the binade of the high words' sum s (in DW_SLICE slices) and,
 * for the pairs' sum, by the binade and the sign of the low words' sum t.
 * In a case every value of the algorithm is an interval with two facts:
 * every value in it is a multiple of 2^g (a float in a binade is one of
 * its ulp; a sum of multiples is one of the smaller), and whether it is a
 * float.  An addition rounds exactly when its interval's values are
 * floats, or multiples of 2^g below 2^(g + p); otherwise it errs by at
 * most half an ulp of the largest binade its interval reaches.  Fast2Sum
 * must be exact in every case: a's exponent at least b's (a nonzero
 * multiple of 2^g has one of g or more), a = 0, or b the error of the
 * rounding that gave a.  The case's bound is the sum of its
 * roundings' errors over the least |x + y| of the case; a case where
 * x + y may be 0 needs those errors to be 0.  The constant is the largest
 * over the cases, each endpoint rounded outward.
 */
#include "dword.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DW_DLO (-40)		/* the binades enumerated: 0 .. DW_DLO */
#define DW_SLICE 8		/* slices of a binade of s */
#define DW_TLO 40		/* binades of t enumerated below its largest */
#define GNONE INT_MIN

struct av {
	long double lo, hi;	/* the exact value's interval */
	int g;			/* every value a multiple of 2^g; GNONE */
	int fl;			/* every value a binary32 */
	int id;			/* a rounding's result: the same value, same id */
	int errof;		/* the error of the rounding whose result is id */
};

#define DW_PIECES 2048

static struct {
	struct {
		long double worst;	/* the largest bound so far */
		char where[160];	/* its case */
		long ncase, nfail;
		char fail[160];		/* the first case that fails */
		int nid;		/* ids given */
	} run;
	int p;			/* the format's precision: binary32's 24 */
	int s;			/* the small formats' scale: a value is v 2^-s */
	struct av sp[DW_PIECES], tp[DW_PIECES];	/* the pieces of s and t */
	long double done[2][64];	/* each bound once, by kind and p */
	long cnt[2][64];
	char said[2][64][400];
} g_dwcase;

#define P (g_dwcase.p)

/* v rounded to nearest, ties to even, at P bits (binary32's conversion
 * for 24) */
static long double
rnp(long double v)
{
	int e;
	long double m;

	if (v == 0)
		return 0;
	m = frexpl(v, &e);
	return ldexpl(nearbyintl(ldexpl(m, P)), e - P);
}

/* a + b rounded down or up: the long double sum and its exact error */
static long double
add_dir(long double a, long double b, int up)
{
	long double s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);

	if (up && e > 0)
		return nextafterl(s, (long double)INFINITY);
	if (!up && e < 0)
		return nextafterl(s, -(long double)INFINITY);
	return s;
}

static long double
av_mag(const struct av *a)
{
	return fabsl(a->lo) > fabsl(a->hi) ? fabsl(a->lo) : fabsl(a->hi);
}

static long double
av_mig(const struct av *a)
{
	return a->lo > 0 ? a->lo : a->hi < 0 ? -a->hi : 0;
}

static int
is_zero(const struct av *a)
{
	return a->lo == 0 && a->hi == 0;
}

/* the binade of v > 0: 2^k <= v < 2^(k + 1) */
static int
binade(long double v)
{
	int e;

	(void)frexpl(v, &e);
	return e - 1;
}

/* the lowest set bit of a dyadic v != 0 */
static int
lowbit(long double v)
{
	int e, k = 0;
	long double m = frexpl(fabsl(v), &e);

	while (m != floorl(m) && k < 80) {
		m *= 2;
		k++;
	}
	return e - k;
}

static struct av
av_make(long double lo, long double hi, int g, int fl)
{
	struct av a = {lo, hi, g, fl, 0, 0};

	/* a point is a multiple of its lowest bit */
	if (lo == hi && lo != 0 && (g == GNONE || lowbit(lo) > g))
		a.g = lowbit(lo);
	return a;
}

static struct av
av_add(const struct av *a, const struct av *b)
{
	if (is_zero(a))
		return *b;
	if (is_zero(b))
		return *a;
	return av_make(add_dir(a->lo, b->lo, 0), add_dir(a->hi, b->hi, 1),
	    a->g == GNONE || b->g == GNONE ? GNONE : a->g < b->g ? a->g : b->g,
	    0);
}

/*
 * r = RN(a) and e = a - r, a a sum of two floats (so e is a float).
 * Exact when a's values are floats or multiples of 2^g below 2^(g + p);
 * else |e| <= 2^(k - p), k the largest binade a value below M = max |a|
 * reaches (M itself, a power of two, is a float).  r lies between a's
 * ends rounded to nearest (binary32's conversion).  r is a multiple of
 * 2^g (inexact, |a| >= 2^(g + p) and so is |r|, whose ulp is 2^(g + 1)
 * or more), and of r's least binade's ulp.  Returns 1 when exact.
 */
static int
av_rn(const struct av *a, struct av *r, struct av *e)
{
	long double M = av_mag(a), E, m;
	int k;

	if (a->fl || is_zero(a) || (a->g != GNONE && M < ldexpl(1, a->g +
	    P))) {
		*r = *a;
		r->fl = 1;
		*e = av_make(0, 0, GNONE, 1);
		return 1;
	}
	k = binade(M);
	if (ldexpl(1, k) == M)
		k--;
	E = ldexpl(1, k - P);
	/* RN is monotone: r between the ends rounded (a float, kept) */
	*r = av_make(rnp(a->lo), rnp(a->hi), a->g, 1);
	m = av_mig(a);
	if (m > 0 && (r->g == GNONE || binade(m) - (P - 1) > r->g))
		r->g = binade(m) - (P - 1);
	r->id = ++g_dwcase.run.nid;
	*e = av_make(-E, E, a->g, 1);
	e->errof = r->id;
	return 0;
}

/* Fast2Sum(a, b) is exact when b = 0, a = 0 (s = b, t = 0), a's exponent
 * is b's or more (Dekker: |a| >= |b| is a case of it; a nonzero multiple
 * of 2^g has an exponent g or more), or b is the error of the rounding
 * that gave a (|b| <= ulp(a) / 2) */
static int
f2s_ok(const struct av *a, const struct av *b)
{
	long double mb = av_mag(b), ma = av_mig(a);
	int ea;

	if (mb == 0 || is_zero(a) || (a->id && b->errof == a->id))
		return 1;
	if (ma > 0)
		ea = binade(ma);
	else if (a->g != GNONE)
		ea = a->g;
	else
		return 0;
	return ea >= binade(mb);
}

/* err / |x + y| over a case, rounded up; -1 when x + y may be 0 and err
 * is not */
static long double
ratio(long double err, const struct av *xy)
{
	long double m = av_mig(xy), r;

	if (err == 0)
		return 0;
	if (m <= 0)
		return -1;
	r = err / m;
	return nextafterl(r, (long double)INFINITY);
}

static void
case_note(long double r, const char *what)
{
	g_dwcase.run.ncase++;
	if (r < 0) {
		if (getenv("DWCASE_DEBUG"))
			fprintf(stderr, "fail: %s\n", what);
		if (g_dwcase.run.nfail++ == 0)
			snprintf(g_dwcase.run.fail, sizeof g_dwcase.run.fail, "%s", what);
		return;
	}
	if (r > g_dwcase.run.worst) {
		g_dwcase.run.worst = r;
		snprintf(g_dwcase.run.where, sizeof g_dwcase.run.where, "%s", what);
	}
}

/*
 * The pieces of an interval by sign and binade: |v| in [2^k, 2^(k+1))
 * cut in DW_SLICE (slices > 1) or whole; below 2^lo one piece holding 0,
 * or with g known the point 0 (a nonzero multiple of 2^g is 2^g or
 * more).  At most cap pieces.
 */
static int
pieces(const struct av *a, int slices, int lo, struct av *out, int cap)
{
	int n = 0, sg, k, j, kmax;
	long double M = av_mag(a);

	if (M == 0) {
		out[0] = *a;
		return 1;
	}
	kmax = binade(M);
	if (a->g != GNONE && a->g > lo)
		lo = a->g;
	/* below 2^lo: one piece (or 0 itself) */
	if (a->lo <= 0 && a->hi >= 0) {
		if (a->g != GNONE && lo == a->g)
			out[n++] = av_make(0, 0, a->g, 1);
		else {
			long double b = ldexpl(1, lo);
			out[n++] = av_make(a->lo > -b ? a->lo : -b, a->hi < b ? a->hi :
			    b, a->g, 0);
		}
	}
	for (sg = -1; sg <= 1; sg += 2)
		for (k = lo; k <= kmax && n < cap; k++)
			for (j = 0; j < slices && n < cap; j++) {
				long double p = ldexpl(1, k) * (1 + (long double)j /
				    slices), q = ldexpl(1, k) * (1 + (long double)(j +
				    1) / slices), plo = sg < 0 ? -q : p, phi = sg < 0 ?
				    -p : q;
				if (plo < a->lo)
					plo = a->lo;
				if (phi > a->hi)
					phi = a->hi;
				if (plo > phi)
					continue;
				out[n++] = av_make(plo, phi, a->g, a->fl);
			}
	return n;
}

/* DWPlusFP(xh + xl, y): (sh, sl) = 2Sum(xh, y), v = RN(xl + sl),
 * Fast2Sum(sh, v) */
static void
case_fp(const struct av *xh, const struct av *xl, const struct av *y,
    const char *name)
{
	struct av *sp = g_dwcase.sp, s = av_add(xh, y), sh, sl, vv, v, ev, xy;
	int n, i;
	char what[160];

	n = pieces(&s, DW_SLICE, binade(av_mag(&s)) - 60, sp, DW_PIECES);
	for (i = 0; i < n; i++) {
		long double r;
		(void)av_rn(&sp[i], &sh, &sl);
		vv = av_add(xl, &sl);
		(void)av_rn(&vv, &v, &ev);
		xy = av_add(&sp[i], xl);
		snprintf(what, sizeof what, "%s, s in [%.6Lg, %.6Lg]", name,
		    sp[i].lo, sp[i].hi);
		r = f2s_ok(&sh, &v) ? ratio(av_mag(&ev), &xy) : -1;
		case_note(r, what);
	}
}

/* AccurateDWPlusDW(x~, y~): (sh, sl) = 2Sum(xh, yh), (th, tl) =
 * 2Sum(xl, yl), c = RN(sl + th), (vh, vl) = Fast2Sum(sh, c),
 * w = RN(tl + vl), Fast2Sum(vh, w) */
static long double
acc_one(const struct av *s, const struct av *t, int *ok)
{
	struct av sh, sl, th, tl, cc, c, ec, vv, vh, vl, ww, w, ew, xy;

	(void)av_rn(s, &sh, &sl);
	(void)av_rn(t, &th, &tl);
	cc = av_add(&sl, &th);
	(void)av_rn(&cc, &c, &ec);
	*ok = f2s_ok(&sh, &c);
	vv = av_add(&sh, &c);
	(void)av_rn(&vv, &vh, &vl);
	ww = av_add(&tl, &vl);
	(void)av_rn(&ww, &w, &ew);
	*ok = *ok && f2s_ok(&vh, &w);
	xy = av_add(s, t);
	return ratio(add_dir(av_mag(&ec), av_mag(&ew), 1), &xy);
}

static void
case_acc(const struct av *xh, const struct av *xl, const struct av *yh,
    const struct av *yl, const char *name)
{
	struct av *sp = g_dwcase.sp, *tp = g_dwcase.tp, s = av_add(xh, yh),
	    t = av_add(xl, yl);
	int ns, nt, i, j, ok;
	long double r;
	char what[160];

	ns = pieces(&s, DW_SLICE, binade(av_mag(&s)) - 60, sp, DW_PIECES);
	for (i = 0; i < ns; i++) {
		/* t whole first; cut by binade and sign where that is not
		 * enough */
		r = acc_one(&sp[i], &t, &ok);
		snprintf(what, sizeof what, "%s, s in [%.6Lg, %.6Lg]", name,
		    sp[i].lo, sp[i].hi);
		if (ok && r >= 0 && r <= g_dwcase.run.worst) {
			case_note(r, what);
			continue;
		}
		nt = pieces(&t, 1, binade(av_mag(&t)) - DW_TLO, tp, DW_PIECES);
		for (j = 0; j < nt; j++) {
			r = acc_one(&sp[i], &tp[j], &ok);
			snprintf(what, sizeof what, "%s, s in [%.6Lg, %.6Lg], t in "
			    "[%.4Lg, %.4Lg]", name, sp[i].lo, sp[i].hi, tp[j].lo,
			    tp[j].hi);
			case_note(ok ? r : -1, what);
		}
	}
}

/* a high word in the binade 2^d of sign sg, its low word within half its
 * ulp; d < DW_DLO: every binade below 2^(DW_DLO + 1), 0 included */
static void
word(int d, int sg, struct av *h, struct av *l)
{
	long double a = d < DW_DLO ? 0 : ldexpl(1, d), b = ldexpl(1, (d <
	    DW_DLO ? DW_DLO : d) + 1), hu = ldexpl(1, (d < DW_DLO ? DW_DLO :
	    d) - P);

	*h = av_make(sg < 0 ? -b : a, sg < 0 ? -a : b, d < DW_DLO ? GNONE :
	    d - (P - 1), 1);
	*l = av_make(-hu, hu, GNONE, 1);
}

/*
 * The constant of DWPlusFP (kind 0) or AccurateDWPlusDW (1), whatever
 * cancels: the largest bound over its cases, LINF-like (infinity) when a
 * case fails.  The cases run once; what they found is said in why.
 */
long double
dwcase_add(int kind, long *ncase, char *why, size_t n)
{
	return dwcase_add_p(kind, 24, ncase, why, n);
}

/* the same at a precision of p bits (u = 2^-p), so a small format's
 * every pair can check it */
long double
dwcase_add_p(int kind, int p, long *ncase, char *why, size_t n)
{
	struct av xh, xl, yh, yl, one;
	int d, sg;
	char nm[64];

	if (kind < 0 || kind > 1 || p < 3 || p > 48)
		return (long double)INFINITY;
	if (g_dwcase.done[kind][p] == 0) {
		memset(&g_dwcase.run, 0, sizeof g_dwcase.run);
		g_dwcase.p = p;
		/* one high word in [1, 2), a multiple of its ulp */
		one = av_make(1, 2, 1 - P, 1);
		xh = one;
		xl = av_make(-ldexpl(1, -P), ldexpl(1, -P), GNONE, 1);
		for (d = 0; d >= DW_DLO - 1; d--)
			for (sg = -1; sg <= 1; sg += 2) {
				word(d, sg, &yh, &yl);
				snprintf(nm, sizeof nm, "%s in %s2^%s%d", kind ? "yh" :
				    "y", sg < 0 ? "-" : "", d < DW_DLO ? "<" : "", d <
				    DW_DLO ? DW_DLO + 1 : d);
				if (kind)
					case_acc(&xh, &xl, &yh, &yl, nm);
				else {
					/* y the larger: y in [1, 2), xh in the binade */
					case_fp(&xh, &xl, &yh, nm);
					if (d < 0) {
						snprintf(nm, sizeof nm, "y in [1, 2), xh in "
						    "%s2^%s%d", sg < 0 ? "-" : "", d < DW_DLO ?
						    "<" : "", d < DW_DLO ? DW_DLO + 1 : d);
						case_fp(&yh, &yl, &xh, nm);
					}
				}
			}
		g_dwcase.cnt[kind][p] = g_dwcase.run.ncase;
		if (g_dwcase.run.nfail) {
			g_dwcase.done[kind][p] = (long double)INFINITY;
			snprintf(g_dwcase.said[kind][p], sizeof g_dwcase.said[kind][p], "%ld of %ld cases not "
			    "closed (the first: %s)", g_dwcase.run.nfail, g_dwcase.run.ncase,
			    g_dwcase.run.fail);
		} else {
			g_dwcase.done[kind][p] = g_dwcase.run.worst;
			snprintf(g_dwcase.said[kind][p], sizeof g_dwcase.said[kind][p], "%s",
			    g_dwcase.run.where);
		}
	}
	if (ncase)
		*ncase = g_dwcase.cnt[kind][p];
	if (why)
		snprintf(why, n, "%s", g_dwcase.said[kind][p]);
	return g_dwcase.done[kind][p];
}

/* ---- every operand of a small format, against the bounds ---- */

/*
 * At p bits (p <= 5), every operand of the format in a window: xh in
 * [1, 2) (the bounds are the same for any scale and sign), y or yh in the
 * binades -(2p + 4)..2 (AccurateDWPlusDW: |yh| <= xh, the algorithm being
 * symmetric) of either sign and 0, the low words 0 and every normalized
 * one (h = RN(h + l)) in the 4 binades below their high word's ulp.
 * Values are integers scaled by 2^S, rounded to nearest even; the largest
 * relative error of each algorithm, and the count of a Fast2Sum found
 * inexact or of an error where x + y = 0 (both must be 0).
 */
#define EV_LO 4

typedef long long fx;

static int
ev_top(unsigned long long a)
{
	int b = 63;

	while (!(a >> b))
		b--;
	return b;
}

static fx
ev_rn(fx x)
{
	unsigned long long a = x < 0 ? -(unsigned long long)x :
	    (unsigned long long)x, q, r, h;
	int sh;

	if (a == 0)
		return 0;
	sh = ev_top(a) - (g_dwcase.p - 1);
	if (sh <= 0)
		return x;
	q = a >> sh;
	r = a & ((1ULL << sh) - 1);
	h = 1ULL << (sh - 1);
	if (r > h || (r == h && (q & 1)))
		q++;
	a = q << sh;
	return x < 0 ? -(fx)a : (fx)a;
}

/* every float of the binades elo..ehi, both signs; their count */
static int
ev_floats(int elo, int ehi, fx *out)
{
	int e, m, sg, n = 0;

	for (e = elo; e <= ehi; e++)
		for (m = 1 << (g_dwcase.p - 1); m < 1 << g_dwcase.p; m++)
			for (sg = -1; sg <= 1; sg += 2)
				out[n++] = (fx)sg * ((fx)m << (g_dwcase.s + e - (g_dwcase.p - 1)));
	return n;
}

/* 0 and the normalized low words of h */
static int
ev_lows(fx h, fx *out, fx *tmp)
{
	int e, n, k, m = 0;

	out[m++] = 0;
	if (h == 0)
		return m;
	e = ev_top((unsigned long long)(h < 0 ? -h : h)) - g_dwcase.s;
	n = ev_floats(e - g_dwcase.p - EV_LO, e - g_dwcase.p, tmp);
	for (k = 0; k < n; k++)
		if (ev_rn(h + tmp[k]) == h)
			out[m++] = tmp[k];
	return m;
}

static void
ev_2sum(fx a, fx b, fx *s, fx *t)
{
	fx ap, bp;

	*s = ev_rn(a + b);
	ap = ev_rn(*s - b);
	bp = ev_rn(*s - ap);
	*t = ev_rn(ev_rn(a - ap) + ev_rn(b - bp));
}

static int
ev_f2s(fx a, fx b, fx *s, fx *t)
{
	*s = ev_rn(a + b);
	*t = ev_rn(b - ev_rn(*s - a));
	return *s + *t == a + b;
}

static void
ev_err(fx z, fx T, double *worst, long *bad)
{
	fx e = z - T;

	if (T == 0) {
		*bad += e != 0;
		return;
	}
	if (e < 0)
		e = -e;
	if ((double)e / fabs((double)T) > *worst)
		*worst = (double)e / fabs((double)T);
}

int
dwcase_every(int p, struct dwcase_ev *r)
{
	enum { NF = 2 * 16 * 24 };
	fx ys[NF], xl[NF], yl[NF], tmp[NF];
	int ry = 2 * p + 4, m, i, j, a, b, nx, ny, nyl;
	double u2 = ldexp(1, -2 * p);

	memset(r, 0, sizeof *r);
	if (p < 3 || p > 5)
		return 0;
	g_dwcase.p = p;
	g_dwcase.s = ry + 2 * p + EV_LO;
	for (m = 1 << (p - 1); m < 1 << p; m++) {
		fx xh = (fx)m << (g_dwcase.s - (p - 1));
		nx = ev_lows(xh, xl, tmp);
		ny = ev_floats(-ry, 2, ys);
		ys[ny++] = 0;
		for (i = 0; i < nx; i++)
			for (j = 0; j < ny; j++) {
				fx sh, sl, v, zh, zl;
				ev_2sum(xh, ys[j], &sh, &sl);
				v = ev_rn(xl[i] + sl);
				r->bad += !ev_f2s(sh, v, &zh, &zl);
				ev_err(zh + zl, xh + xl[i] + ys[j], &r->fp, &r->bad);
				r->nfp++;
			}
		ny = ev_floats(-ry, 0, ys);
		ys[ny++] = 0;
		for (j = 0; j < ny; j++) {
			fx yh = ys[j];
			if ((yh < 0 ? -yh : yh) > xh)
				continue;
			nyl = ev_lows(yh, yl, tmp);
			for (a = 0; a < nx; a++)
				for (b = 0; b < nyl; b++) {
					fx sh, sl, th, tl, c, vh, vl, w, zh, zl;
					ev_2sum(xh, yh, &sh, &sl);
					ev_2sum(xl[a], yl[b], &th, &tl);
					c = ev_rn(sl + th);
					r->bad += !ev_f2s(sh, c, &vh, &vl);
					w = ev_rn(tl + vl);
					r->bad += !ev_f2s(vh, w, &zh, &zl);
					ev_err(zh + zl, xh + xl[a] + yh + yl[b], &r->acc,
					    &r->bad);
					r->nacc++;
				}
		}
	}
	r->fp /= u2;
	r->acc /= u2;
	r->ry = ry;
	return 1;
}
