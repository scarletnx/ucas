/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * rint: integration.  Rational functions completely (polynomial part,
 * Hermite reduction for the rational part, partial fractions over the
 * irreducible factors for the logarithmic part: log|x - r|, log and atan
 * for quadratics, c log|f| when the numerator is c f', quartics that
 * split into quadratics over Q(sqrt d), and otherwise numeric roots with
 * a note), and then the other methods, each result checked by
 * differentiating it back before it is used: linearity, R(e^x) through
 * u = e^x, rational functions of sin and cos (Bioche's rules: t = cos,
 * sin, tan, else the Weierstrass t = tan(x/2)), R(x, sqrt(a x + b)) and
 * R(x, sqrt(quadratic)) (reduced to P/sqrt(q), solved by Q sqrt(q) +
 * lambda * integral of 1/sqrt(q): asin, asinh or log), integration by
 * parts (log, atan, asin ... times an integrable factor; exp times sin
 * or cos), and substitution u = g(x) when g' divides the integrand.
 */
#include "alg.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int alg_check(nid a, nid b, double *worst, int *onlyone);
const char *alg_int_note;
static char notebuf[256];

/* ---- term lists ---- */

struct tl {
	nid t[96];
	int n;
};

static void
tl_add(struct tl *l, nid t)
{
	if (t && !const_is(t, 0) && l->n < 96)
		l->t[l->n++] = t;
}

static nid
tl_sum(const struct tl *l)
{
	nid r = 0;
	int i;

	for (i = 0; i < l->n; i++)
		r = r ? ADD(r, l->t[i]) : l->t[i];
	return r ? r : KR(0);
}

static nid
logabs(nid u)
{
	return mk_un(O_LOG, mk_un(O_ABS, u));
}

/* (p/q) / sqrt(m) * T, written T/sqrt(m) or p T/(q sqrt(m)) */
static nid
over_sqrt(aq r, int64_t m, nid T)
{
	int neg = r.n < 0;
	nid num, den;

	if (m == 1)
		return qmul_node(r, T);
	if (neg)
		r.n = -r.n;
	num = r.n == 1 ? T : MUL(KR((double)r.n), T);
	den = r.d == 1 ? sqrt_cnode(m) : MUL(KR((double)r.d), sqrt_cnode(m));
	num = DIV(num, den);
	return neg ? mk_un(O_NEG, num) : num;
}

/* (u x + v)/sqrt(m) with integer-cleared coefficients */
static nid
lin_over_sqrt(aq u, aq v, int64_t m, nid x)
{
	struct up l;
	int64_t L = lcm64(u.d, v.d);
	nid p;

	l.d = 1;
	l.c[1] = qmul(u, qi(L));
	l.c[0] = qmul(v, qi(L));
	up_trim(&l);
	p = up_node(&l, x);
	if (m == 1 && L == 1)
		return p;
	if (m == 1)
		return DIV(p, KR((double)L));
	return DIV(p, L == 1 ? sqrt_cnode(m) : MUL(KR((double)L), sqrt_cnode(m)));
}

static int
no_real_roots(const struct up *f)
{
	double c[UD + 1];
	struct root r[UD + 1];
	int i, n;

	if (f->d % 2)
		return 0;
	if (f->d == 2)
		return qsgn(qsub(qmul(qi(4), qmul(f->c[2], f->c[0])),
		    qmul(f->c[1], f->c[1]))) > 0;
	if (f->d > 14)
		return 0;
	for (i = 0; i <= f->d; i++)
		c[i] = qd(f->c[i]);
	n = poly_roots(c, f->d, r);
	for (i = 0; i < n; i++)
		if (r[i].im == 0)
			return 0;
	return n > 0;
}

/* ---- partial fractions ---- */

struct pfterm {
	struct up a, f;
	int k;
};

int
up_apart(const struct up *n0, const struct up *d0, struct up *poly,
    void *vterms, int max, int *partial)
{
	struct pfterm *t = vterms;
	struct up n, d, r, f[UD], F, G, s, tt, g, C, tmp;
	int m[UD], nf, j, i, nt = 0;
	aq lc;

	alg_ovf = 0;
	lc = d0->c[d0->d];
	up_scale(&n, n0, qdiv(qi(1), lc));
	up_scale(&d, d0, qdiv(qi(1), lc));
	up_divmod(&n, &d, poly, &r);
	nf = up_factor(&d, f, m, UD, partial);
	for (j = 0; j < nf && !alg_ovf; j++) {
		F.d = 0;
		F.c[0] = qi(1);
		for (i = 0; i < m[j]; i++)
			up_mul(&F, &F, &f[j]);
		up_divmod(&d, &F, &G, NULL);
		if (G.d == 0 && qeq(G.c[0], qi(1))) {
			up_set(&C, &r);
		} else {
			if (!up_gcdex(&G, &F, &s, &tt, &g))
				return -1;
			up_mul(&tmp, &r, &s);
			up_divmod(&tmp, &F, NULL, &C);
		}
		for (i = 0; i < m[j] && !up_iszero(&C); i++) {
			struct up ci;
			up_divmod(&C, &f[j], &C, &ci);
			if (!up_iszero(&ci) && nt < max) {
				up_set(&t[nt].a, &ci);
				up_set(&t[nt].f, &f[j]);
				t[nt].k = m[j] - i;
				nt++;
			}
		}
	}
	/* terms by factor, the highest power last */
	return alg_ovf ? -1 : nt;
}

/* ---- the logarithmic part ---- */

static int
qnear(double v, aq *r)
{
	double x = fabs(v), h0 = 0, h1 = 1, k0 = 1, k1 = 0, a;
	int i;

	if (!isfinite(v) || fabs(v) > 1e9)
		return 0;
	for (i = 0; i < 30; i++) {
		double h, k;
		a = floor(x);
		h = a * h1 + h0;
		k = a * k1 + k0;
		if (k > 10000)
			return 0;
		h0 = h1, h1 = h, k0 = k1, k1 = k;
		if (fabs(h1 / k1 - fabs(v)) <= 1e-9 * fmax(1, fabs(v))) {
			*r = qmk((int64_t)(v < 0 ? -h1 : h1), (int64_t)k1);
			return 1;
		}
		if (x - a < 1e-12)
			return 0;
		x = 1 / (x - a);
	}
	return 0;
}

static struct qs
qsq(aq a)
{
	return (struct qs){a, qi(0)};
}

static int
qs_zero(struct qs x)
{
	return x.a.n == 0 && x.b.n == 0;
}

/* c * T for c in Q(sqrt d) */
static nid
qs_times(struct qs c, nid T)
{
	if (c.b.n == 0)
		return qmul_node(c.a, T);
	if (c.a.n == 0)
		return qmul_node(c.b, const_is(T, 1) ? sqrt_cnode(qs_d) :
		    MUL(sqrt_cnode(qs_d), T));
	return const_is(T, 1) ? qs_node(c) : MUL(qs_node(c), T);
}

static nid
qs_quad_node(struct qs p, struct qs q, nid x)
{
	nid r = POW(x, KI(2));

	if (!qs_zero(p))
		r = ADD(r, qs_times(p, x));
	if (!qs_zero(q))
		r = ADD(r, qs_times(q, KR(1)));
	return r;
}

/* T / sq with the sign outside */
static nid
div_neg(nid T, nid sq)
{
	if (N.op[T] == O_NEG)
		return mk_un(O_NEG, DIV(N.a[T], sq));
	return DIV(T, sq);
}

/* c T / sq: a rational c as p T/(q sq) */
static nid
qs_over(struct qs c, nid T, nid sq)
{
	aq r = c.b.n == 0 ? c.a : c.b;
	int neg = r.n < 0;
	nid num, den;

	if (c.a.n && c.b.n)
		return div_neg(qs_times(c, T), sq);
	if (neg)
		r.n = -r.n;
	num = r.n == 1 ? T : MUL(KR((double)r.n), T);
	if (c.b.n)
		num = MUL(sqrt_cnode(qs_d), num);
	den = r.d == 1 ? sq : MUL(KR((double)r.d), sq);
	num = DIV(num, den);
	return neg ? mk_un(O_NEG, num) : num;
}

/* sqrt(D) = k sqrt(D0), the rational content's square taken out */
static nid
qs_sqrt_node(struct qs D, aq *k)
{
	aq c = qi(1), kk;
	int64_t m;
	struct qs D0;

	if (D.a.n && D.b.n) {
		int64_t g = gcd64(D.a.n, D.b.n), l = lcm64(D.a.d, D.b.d);
		c = qmk(g, l);
	} else
		c = D.a.n ? D.a : D.b;
	if (qsgn(c) < 0)
		c = qneg(c);
	sqrt_split(c, &kk, &m);
	/* D = kk^2 m D0' : D0 = D / kk^2 */
	D0 = qs_div(D, qsq(qmul(kk, kk)));
	*k = kk;
	return mk_raw(O_SQRT, TY_REAL, qs_node(D0), 0, 0, 0, 0);
}

/* the integral of (al x + be)/(x^2 + p x + q) over Q(sqrt d), into l */
static void
qs_quad_int(struct qs al, struct qs be, struct qs p, struct qs q, nid x,
    struct tl *l)
{
	struct qs two = qsq(qi(2)), half = qsq(qmk(1, 2));
	struct qs D = qs_sub(qs_mul(qsq(qi(4)), q), qs_mul(p, p));
	struct qs bp = qs_sub(be, qs_mul(al, qs_mul(p, half)));
	nid f = qs_quad_node(p, q, x), lin = ADD(MUL(KR(2), x), qs_times(p,
	    KR(1)));
	struct qs S;

	if (qs_zero(p))
		lin = MUL(KR(2), x);
	if (qs_val(D) > 0) {
		if (!qs_zero(al))
			tl_add(l, qs_times(qs_mul(al, half), mk_un(O_LOG, f)));
		if (qs_zero(bp))
			return;
		if (qs_sqrt(D, &S)) {
			struct qs u = qs_div(two, S), v = qs_div(p, S);
			nid arg = qs_zero(v) ? qs_times(u, x) : ADD(qs_times(u, x),
			    qs_times(v, KR(1)));
			tl_add(l, qs_times(qs_div(qs_mul(two, bp), S), mk_un(O_ATAN,
			    arg)));
		} else {
			aq k;
			nid sq = qs_sqrt_node(D, &k);
			struct qs u = qsq(qdiv(qi(2), k)), v = qs_div(p, qsq(k));
			nid ln = qs_zero(v) ? qs_times(u, x) : ADD(qs_times(u, x),
			    qs_times(v, KR(1)));
			tl_add(l, qs_over(qs_div(qs_mul(two, bp), qsq(k)),
			    mk_un(O_ATAN, DIV(ln, sq)), sq));
		}
	} else {
		struct qs dl = qs_sub(qs_mul(p, p), qs_mul(qsq(qi(4)), q));
		nid sq;
		if (!qs_zero(al))
			tl_add(l, qs_times(qs_mul(al, half), logabs(f)));
		if (qs_zero(bp))
			return;
		if (qs_sqrt(dl, &S)) {
			sq = qs_node(S);
			tl_add(l, div_neg(qs_times(bp, logabs(DIV(SUB(lin, sq), ADD(lin,
			    sq)))), sq));
		} else {
			aq k;
			nid ks;
			sq = qs_sqrt_node(dl, &k);
			ks = qmul_node(k, sq);
			tl_add(l, qs_over(qs_div(bp, qsq(k)), logabs(DIV(SUB(lin,
			    ks), ADD(lin, ks))), sq));
		}
	}
}

/* f monic irreducible quartic = (x^2 + p1 x + q1)(x^2 + p2 x + q2) over Q(sqrt d) */
int
quartic_split(const struct up *f, struct qs *p, struct qs *q)
{
	static const int pr[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
	double c[5];
	struct root rr[8];
	double complex z[4];
	int nr, i, k, nz = 0;
	aq c3 = f->c[3];

	for (i = 0; i <= 4; i++)
		c[i] = qd(f->c[i]);
	nr = poly_roots(c, 4, rr);
	for (i = 0; i < nr; i++)
		for (k = 0; k < rr[i].mult && nz < 4; k++) {
			z[nz++] = rr[i].re + I * rr[i].im;
			if (rr[i].im != 0 && nz < 4)
				z[nz++] = rr[i].re - I * rr[i].im;
		}
	if (nz != 4)
		return 0;
	for (k = 0; k < 3; k++) {
		double complex s1 = z[pr[k][0]] + z[pr[k][1]], m1 = z[pr[k][0]] *
		    z[pr[k][1]], s2 = z[pr[k][2]] + z[pr[k][3]], m2 = z[pr[k][2]] *
		    z[pr[k][3]];
		double P1, P2, Q1, Q2;
		aq Pp, Qs, Qp, Dp, Dq, kp, kq;
		int64_t mp1, mq1, d;
		struct qs a1, a2, b1, b2, t;
		if (fabs(cimag(s1)) + fabs(cimag(m1)) + fabs(cimag(s2)) +
		    fabs(cimag(m2)) > 1e-7)
			continue;
		P1 = -creal(s1);
		P2 = -creal(s2);
		Q1 = creal(m1);
		Q2 = creal(m2);
		if (!qnear(P1 * P2, &Pp) || !qnear(Q1 + Q2, &Qs) ||
		    !qnear(Q1 * Q2, &Qp))
			continue;
		alg_ovf = 0;
		Dp = qsub(qmul(c3, c3), qmul(qi(4), Pp));
		Dq = qsub(qmul(Qs, Qs), qmul(qi(4), Qp));
		if (qsgn(Dp) < 0 || qsgn(Dq) < 0 || alg_ovf)
			continue;
		sqrt_split(Dp, &kp, &mp1);
		sqrt_split(Dq, &kq, &mq1);
		if (Dp.n && Dq.n && mp1 != mq1)
			continue;
		d = Dp.n ? mp1 : mq1;
		if (d == 1)
			continue;
		qs_d = d;
		a1 = (struct qs){qdiv(c3, qi(2)), qdiv(kp, qi(2))};
		a2 = (struct qs){qdiv(c3, qi(2)), qneg(qdiv(kp, qi(2)))};
		if (fabs(qs_val(a2) - P1) < fabs(qs_val(a1) - P1))
			t = a1, a1 = a2, a2 = t;
		b1 = (struct qs){qdiv(Qs, qi(2)), qdiv(kq, qi(2))};
		b2 = (struct qs){qdiv(Qs, qi(2)), qneg(qdiv(kq, qi(2)))};
		if (fabs(qs_val(b2) - Q1) < fabs(qs_val(b1) - Q1))
			t = b1, b1 = b2, b2 = t;
		/* exact check of the product */
		{
			struct qs e3 = qs_add(a1, a2);
			struct qs e2 = qs_add(qs_add(b1, b2), qs_mul(a1, a2));
			struct qs e1 = qs_add(qs_mul(a1, b2), qs_mul(a2, b1));
			struct qs e0 = qs_mul(b1, b2);
			if (alg_ovf || e3.b.n || e2.b.n || e1.b.n || e0.b.n ||
			    !qeq(e3.a, f->c[3]) || !qeq(e2.a, f->c[2]) ||
			    !qeq(e1.a, f->c[1]) || !qeq(e0.a, f->c[0]))
				continue;
		}
		p[0] = a1;
		p[1] = a2;
		q[0] = b1;
		q[1] = b2;
		return 1;
	}
	return 0;
}

/* solve the n x n system over Q(sqrt d) (n <= 4) */
static int
qs_solve(struct qs *A, struct qs *b, int n)
{
	int i, j, k, piv;

	for (k = 0; k < n; k++) {
		for (piv = k; piv < n && qs_zero(A[piv * n + k]); piv++)
			;
		if (piv == n)
			return 0;
		if (piv != k) {
			for (j = 0; j < n; j++) {
				struct qs t = A[k * n + j];
				A[k * n + j] = A[piv * n + j];
				A[piv * n + j] = t;
			}
			{
				struct qs t = b[k];
				b[k] = b[piv];
				b[piv] = t;
			}
		}
		for (i = 0; i < n; i++) {
			struct qs fct;
			if (i == k || qs_zero(A[i * n + k]))
				continue;
			fct = qs_div(A[i * n + k], A[k * n + k]);
			for (j = k; j < n; j++)
				A[i * n + j] = qs_sub(A[i * n + j], qs_mul(fct, A[k * n + j]));
			b[i] = qs_sub(b[i], qs_mul(fct, b[k]));
		}
	}
	for (i = 0; i < n; i++)
		b[i] = qs_div(b[i], A[i * n + i]);
	return !alg_ovf;
}

static int numeric_used, hermite_used;

/* c * T with a decimal c, the sign outside */
static nid
numterm(double c, nid T)
{
	nid r = fabs(c) == 1 ? T : MUL(KR(fabs(c)), T);

	return c < 0 ? mk_un(O_NEG, r) : r;
}

static nid
xminus(nid x, double s)
{
	return s < 0 ? ADD(x, KR(-s)) : SUB(x, KR(s));
}

static int
has_op(nid e, int op, int depth)
{
	if (e == 0 || depth > 40)
		return 0;
	if (N.op[e] == op)
		return 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return 0;
	return has_op(N.a[e], op, depth + 1) || has_op(N.b[e], op, depth + 1);
}

/* the integral of a/f, f monic irreducible, deg a < deg f */
static void
log_part(const struct up *a, const struct up *f, nid x, struct tl *l)
{
	struct up fd;
	aq c;
	int i;

	if (up_iszero(a))
		return;
	up_diff(&fd, f);
	/* a = c f' */
	if (a->d == fd.d) {
		struct up t;
		c = qdiv(a->c[a->d], fd.c[fd.d]);
		up_scale(&t, &fd, c);
		up_sub(&t, &t, a);
		if (up_iszero(&t)) {
			nid pf = up_pnode(f, x);
			tl_add(l, qmul_node(c, no_real_roots(f) ? mk_un(O_LOG, pf) :
			    logabs(pf)));
			return;
		}
	}
	if (f->d == 1) {
		/* c / (x - r) */
		tl_add(l, qmul_node(a->c[0], logabs(up_pnode(f, x))));
		return;
	}
	if (f->d == 2) {
		aq p = f->c[1], q = f->c[0], al = a->d == 1 ? a->c[1] : qi(0),
		    be = a->c[0], bp, D, k;
		int64_t m;
		bp = qsub(be, qmul(al, qdiv(p, qi(2))));
		D = qsub(qmul(qi(4), q), qmul(p, p));
		if (al.n)
			tl_add(l, qmul_node(qdiv(al, qi(2)), qsgn(D) > 0 ?
			    mk_un(O_LOG, up_pnode(f, x)) : logabs(up_pnode(f, x))));
		if (bp.n == 0)
			return;
		if (qsgn(D) > 0) {
			/* 2 bp / sqrt(D) atan((2x + p)/sqrt(D)), sqrt(D) = k sqrt(m) */
			sqrt_split(D, &k, &m);
			tl_add(l, over_sqrt(qdiv(qmul(qi(2), bp), k), m, mk_un(O_ATAN,
			    lin_over_sqrt(qdiv(qi(2), k), qdiv(p, k), m, x))));
		} else {
			/* bp / sqrt(dl) log|(2x + p - sqrt(dl))/(2x + p + sqrt(dl))| */
			aq dl = qneg(D), u, v;
			int64_t L;
			struct up lp;
			nid lin, sq;
			sqrt_split(dl, &k, &m);
			/* scale 2x + p and k sqrt(m) together to integers */
			L = lcm64(p.d, k.d);
			lp.d = 1;
			lp.c[1] = qi(2 * L);
			lp.c[0] = qmul(p, qi(L));
			u = qmul(k, qi(L));
			{
				int64_t gg = gcd64(gcd64(2 * L, lp.c[0].n), u.n);
				if (gg > 1) {
					lp.c[1] = qdiv(lp.c[1], qi(gg));
					lp.c[0] = qdiv(lp.c[0], qi(gg));
					u = qdiv(u, qi(gg));
				}
			}
			up_trim(&lp);
			lin = up_node(&lp, x);
			sq = qmul_node(u, sqrt_cnode(m));
			v = qdiv(bp, k);
			tl_add(l, over_sqrt(v, m, logabs(DIV(SUB(lin, sq), ADD(lin,
			    sq)))));
		}
		return;
	}
	if (f->d == 4) {
		struct qs p[2], q[2], A[16], b[4];
		if (quartic_split(f, p, q)) {
			/* a = (al1 x + be1) f2 + (al2 x + be2) f1 */
			struct qs z = qsq(qi(0)), one = qsq(qi(1));
			for (i = 0; i < 16; i++)
				A[i] = z;
			/* unknowns al1, be1, al2, be2; rows x^3, x^2, x, 1 */
			A[0] = one;
			A[2] = one;
			A[4] = p[1];
			A[5] = one;
			A[6] = p[0];
			A[7] = one;
			A[8] = q[1];
			A[9] = p[1];
			A[10] = q[0];
			A[11] = p[0];
			A[13] = q[1];
			A[15] = q[0];
			for (i = 0; i < 4; i++)
				b[i] = qsq(3 - i <= a->d ? a->c[3 - i] : qi(0));
			if (qs_solve(A, b, 4)) {
				qs_quad_int(b[0], b[1], p[0], q[0], x, l);
				qs_quad_int(b[2], b[3], p[1], q[1], x, l);
				return;
			}
		}
	}
	/* numeric roots: sum over the roots r of a(r)/f'(r) log(x - r) */
	{
		double cc[UD + 1];
		struct root rr[UD + 1];
		int nr;
		numeric_used = 1;
		for (i = 0; i <= f->d; i++)
			cc[i] = qd(f->c[i]);
		nr = f->d <= 14 ? poly_roots(cc, f->d, rr) : 0;
		for (i = 0; i < nr; i++) {
			double complex r = rr[i].re + I * rr[i].im, av = 0, dv = 0, res;
			int j;
			for (j = a->d; j >= 0; j--)
				av = av * r + qd(a->c[j]);
			for (j = fd.d; j >= 0; j--)
				dv = dv * r + qd(fd.c[j]);
			res = av / dv;
			if (rr[i].im == 0)
				tl_add(l, numterm(creal(res), logabs(xminus(x, rr[i].re))));
			else {
				double s = rr[i].re, t = rr[i].im, P = creal(res),
				    Q = cimag(res);
				nid sx = xminus(x, s);
				tl_add(l, numterm(P, mk_un(O_LOG, ADD(POW(sx, KI(2)),
				    KR(t * t)))));
				tl_add(l, numterm(-2 * Q, mk_un(O_ATAN, DIV(sx, KR(t)))));
			}
		}
	}
}

/* a/d (univariate, as nodes): numerator expanded, denominator factored */
static nid
upq_node(const struct up *n, const struct up *d, nid x)
{
	int sn = nat;
	nid sa[AV], r;
	struct rf q;

	memcpy(sa, at_node, sizeof sa);
	atoms_reset();
	atom_of(x);
	q.n = mp_of_up(n, 0);
	q.d = mp_of_up(d, 0);
	q = rf_norm(q);
	r = rf_node(q, 2);
	nat = sn;
	memcpy(at_node, sa, sizeof sa);
	return r;
}

/* the integral of n/d in x */
static nid
rat_int(const struct up *n0, const struct up *d0, nid x)
{
	struct up n, d, g, q, r, s[UD], A, D, Gn, Gd;
	struct tl l;
	int ns, i, j, nf, m[UD], partial = 0;
	struct up f[UD];
	aq lc;

	alg_ovf = 0;
	l.n = 0;
	up_gcd(&g, n0, d0);
	up_divmod(n0, &g, &n, NULL);
	up_divmod(d0, &g, &d, NULL);
	lc = d.c[d.d];
	up_scale(&n, &n, qdiv(qi(1), lc));
	up_scale(&d, &d, qdiv(qi(1), lc));
	up_divmod(&n, &d, &q, &r);
	/* polynomial part */
	if (!up_iszero(&q)) {
		struct up P;
		P.d = q.d + 1;
		P.c[0] = qi(0);
		for (i = 0; i <= q.d; i++)
			P.c[i + 1] = qdiv(q.c[i], qi(i + 1));
		tl_add(&l, up_node(&P, x));
	}
	/* Hermite reduction */
	up_set(&A, &r);
	up_set(&D, &d);
	Gn.d = -1;
	Gd.d = 0;
	Gd.c[0] = qi(1);
	ns = up_sqfree(&d, s, UD);
	for (i = 1; i < ns && !alg_ovf; i++) {
		struct up V, U, Vp, UVp, Vi, sg, tg, gg;
		if (s[i].d <= 0)
			continue;
		up_set(&V, &s[i]);
		Vi.d = 0;
		Vi.c[0] = qi(1);
		for (j = 0; j <= i; j++)
			up_mul(&Vi, &Vi, &V);
		up_divmod(&D, &Vi, &U, NULL);
		up_diff(&Vp, &V);
		up_mul(&UVp, &U, &Vp);
		if (!up_gcdex(&UVp, &V, &sg, &tg, &gg))
			return 0;
		for (j = i; j >= 1; j--) {
			struct up c, B, C, t, Bp, Vj;
			int k;
			up_scale(&c, &A, qdiv(qi(-1), qi(j)));
			up_mul(&t, &c, &sg);
			up_divmod(&t, &V, NULL, &B);
			up_mul(&t, &B, &UVp);
			up_sub(&t, &c, &t);
			up_divmod(&t, &V, &C, NULL);
			/* G += B / V^j */
			Vj.d = 0;
			Vj.c[0] = qi(1);
			for (k = 0; k < j; k++)
				up_mul(&Vj, &Vj, &V);
			{
				struct up t1, t2, gcd;
				up_mul(&t1, &Gn, &Vj);
				up_mul(&t2, &B, &Gd);
				up_add(&Gn, &t1, &t2);
				up_mul(&Gd, &Gd, &Vj);
				if (!up_iszero(&Gn)) {
					up_gcd(&gcd, &Gn, &Gd);
					up_divmod(&Gn, &gcd, &Gn, NULL);
					up_divmod(&Gd, &gcd, &Gd, NULL);
				}
			}
			/* A = -j C - U B' */
			up_diff(&Bp, &B);
			up_mul(&t, &U, &Bp);
			up_scale(&C, &C, qi(-j));
			up_sub(&A, &C, &t);
		}
		up_mul(&D, &U, &V);
	}
	if (alg_ovf)
		return 0;
	if (!up_iszero(&Gn)) {
		tl_add(&l, upq_node(&Gn, &Gd, x));
		hermite_used = 1;
	}
	/* logarithmic part: A/D with D squarefree */
	if (!up_iszero(&A)) {
		struct up Dm, rem;
		up_monic(&Dm, &D);
		up_scale(&A, &A, qdiv(qi(1), D.c[D.d]));
		up_divmod(&A, &Dm, NULL, &rem);
		up_set(&A, &rem);
		nf = up_factor(&Dm, f, m, UD, &partial);
		for (j = 0; j < nf && !alg_ovf; j++) {
			struct up G, sg, tg, gg, t, Aj;
			up_divmod(&Dm, &f[j], &G, NULL);
			if (G.d == 0) {
				up_scale(&Aj, &A, qdiv(qi(1), G.c[0]));
			} else {
				if (!up_gcdex(&G, &f[j], &sg, &tg, &gg))
					return 0;
				up_mul(&t, &A, &sg);
				up_divmod(&t, &f[j], NULL, &Aj);
			}
			log_part(&Aj, &f[j], x, &l);
		}
	}
	if (alg_ovf)
		return 0;
	/* the atan terms after the logarithms */
	{
		struct tl o;
		int pass;
		o.n = 0;
		for (pass = 0; pass < 2; pass++)
			for (i = 0; i < l.n; i++)
				if (has_op(l.t[i], O_ATAN, 0) == pass)
					tl_add(&o, l.t[i]);
		return tl_sum(&o);
	}
}

/* ---- the driver ---- */

static int I_slot, I_depth;

static nid
try_diff(nid F)
{
	jmp_buf saved;
	nid d = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		d = diff_node(mk_cast(TY_REAL, F), (uint32_t)I_slot);
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return d;
}

/* F' = f at sample points? */
static int
verified(nid F, nid f)
{
	nid d;
	double w;
	int n;

	if (F == 0 || (d = try_diff(F)) == 0)
		return 0;
	n = alg_check(d, f, &w, NULL);
	return n >= 6 && w <= 1e-6;
}

/* f as n/d in x alone: 1 */
static int
as_upq(nid f, nid x, struct up *n, struct up *d)
{
	struct rf r;
	size_t mk = ar_mark();
	int ok;

	atoms_reset();
	atom_of(x);
	ok = rf_of(f, &r, NULL) && (mp_vars(r.n) | mp_vars(r.d)) <= 1 &&
	    up_of_mp(r.n, 0, n) && up_of_mp(r.d, 0, d);
	ar_release(mk);
	return ok && !alg_ovf;
}

/* from -> to, keeping constant subterms as they are (sqrt(3) stays) */
static nid
rsub(nid e, nid from, nid to, int depth)
{
	int op = N.op[e];

	if (e == from)
		return to;
	if (!(N.fl[e] & NF_VAR) || depth > 200)
		return e;
	switch (op) {
	case O_VAR:
	case O_CONST:
	case O_TABLE:
	case O_LOOP:
	case O_INDEX:
		return e;
	case O_CAST:
		return mk_cast(N.ty[e], rsub(N.a[e], from, to, depth + 1));
	case O_SEL:
		return mk_sel(rsub(N.a[e], from, to, depth + 1), rsub(N.b[e], from,
		    to, depth + 1), rsub(N.c[e], from, to, depth + 1));
	}
	if (opinfo[op].arity == 1)
		return mk_un(op, rsub(N.a[e], from, to, depth + 1));
	if (opinfo[op].arity == 2)
		return mk_bin(op, rsub(N.a[e], from, to, depth + 1), rsub(N.b[e],
		    from, to, depth + 1));
	return e;
}

static nid I_x;

static nid
xsubst(nid e, nid val)
{
	return rsub(e, I_x, val, 0);
}

/* a + b or a - b with b's terms added one by one: no parentheses */
static nid
flat(nid a, nid b, int sub)
{
	if (N.op[b] == O_ADD)
		return flat(flat(a, N.a[b], sub), N.b[b], sub);
	if (N.op[b] == O_SUB)
		return flat(flat(a, N.a[b], sub), N.b[b], !sub);
	if (N.op[b] == O_NEG)
		return flat(a, N.a[b], !sub);
	return sub ? SUB(a, b) : ADD(a, b);
}

/* P(x) * s with P's content in front: (3x^2 - 2)*s/15 */
static nid
poly_times(const struct up *P, nid x, nid s)
{
	int64_t l = 1, g = 0;
	int j;
	struct up p;
	aq c;

	for (j = 0; j <= P->d; j++)
		l = lcm64(l, P->c[j].d);
	for (j = 0; j <= P->d; j++)
		g = gcd64(g, qmul(P->c[j], qi(l)).n);
	if (P->d >= 0 && qsgn(P->c[P->d]) < 0)
		g = -g;
	c = qmk(g, l);
	up_scale(&p, P, qdiv(qi(1), c));
	if (p.d == 0)
		return qmul_node(c, s);
	return qmul_node(c, MUL(up_node(&p, x), s));
}

static nid sub_int(nid h);

/* u = a x + b: a, b numbers */
static int
lin_in(nid u, double *a, double *b)
{
	struct npoly p;

	if (!npoly_of(u, I_slot, &p, 1) || p.deg != 1 || !num_of(p.c[1], a) ||
	    !num_of(p.c[0], b) || *a == 0)
		return 0;
	return 1;
}

static int
has_x(nid e)
{
	return has_slot(e, I_slot);
}

/* every maximal subterm that is op(arg) with op in ops: collect args */
static void
walk(nid e, nid *c, int *nc, int max, int depth)
{
	int op, i;

	if (e == 0 || depth > 60 || !has_x(e) || *nc >= max)
		return;
	op = N.op[e];
	if (op == O_VAR || op == O_CONST)
		return;
	if (opinfo[op].arity == 1 && op != O_NEG && op != O_CAST) {
		for (i = 0; i < *nc; i++)
			if (c[i] == e)
				break;
		if (i == *nc && *nc < max)
			c[(*nc)++] = e;
	}
	walk(N.a[e], c, nc, max, depth + 1);
	walk(N.b[e], c, nc, max, depth + 1);
}

static nid multi_angle(nid f, int *mixed);

/* R(sin u, cos u) with u = a x + b: Bioche */
static nid
trig_int(nid f, nid x)
{
	nid fn[24], u = 0, S, C, g;
	int nf = 0, i, sv, cv;
	double a, b;
	struct rf r, t, tm;
	size_t mk;
	nid T = x, res = 0;
	const char *how = NULL;

	{
		int mixed;
		f = multi_angle(f, &mixed);
	}
	walk(f, fn, &nf, 24, 0);
	for (i = 0; i < nf; i++) {
		int op = N.op[fn[i]];
		if (op == O_SIN || op == O_COS || op == O_TAN || op == O_SEC ||
		    op == O_CSC || op == O_COT) {
			if (u == 0)
				u = N.a[fn[i]];
			else if (N.a[fn[i]] != u)
				return 0;
		}
	}
	if (u == 0 || !lin_in(u, &a, &b))
		return 0;
	S = mk_un(O_SIN, u);
	C = mk_un(O_COS, u);
	g = xreplace(f, mk_un(O_TAN, u), DIV(S, C));
	g = xreplace(g, mk_un(O_SEC, u), DIV(KR(1), C));
	g = xreplace(g, mk_un(O_CSC, u), DIV(KR(1), S));
	g = xreplace(g, mk_un(O_COT, u), DIV(C, S));
	mk = ar_mark();
	atoms_reset();
	sv = atom_of(S);
	cv = atom_of(C);
	if (!rf_of(g, &r, NULL) || ((mp_vars(r.n) | mp_vars(r.d)) & ~3u)) {
		ar_release(mk);
		return 0;
	}
	/* sin^2m cos^2n: power reduction to cos(2u), no tangent poles */
	if (mp_isconst(r.d, NULL) && I_depth < 4) {
		int ok = 1, ii;
		for (ii = 0; ii < r.n.n; ii++)
			if ((r.n.t[ii].e[sv] | r.n.t[ii].e[cv]) & 1)
				ok = 0;
		if (ok) {
			nid c2 = mk_un(O_COS, MUL(KR(2), u)), body, F;
			int w = atom_of(c2);
			aq dc;
			struct mp acc = mp_zero(), hm, hp;
			mp_isconst(r.d, &dc);
			hm = mp_sub(mp_const(qmk(1, 2)), mp_scale(mp_atom(w), qmk(1, 2)));
			hp = mp_add(mp_const(qmk(1, 2)), mp_scale(mp_atom(w), qmk(1, 2)));
			for (ii = 0; ii < r.n.n; ii++)
				acc = mp_add(acc, mp_mul(mp_const(qdiv(r.n.t[ii].c, dc)),
				    mp_mul(mp_pow(hm, r.n.t[ii].e[sv] / 2), mp_pow(hp,
				    r.n.t[ii].e[cv] / 2))));
			body = mp_node(acc);
			ar_release(mk);
			F = antideriv(body, I_slot, NULL);
			if (F) {
				snprintf(notebuf, sizeof notebuf, "power reduction: sin^2 = "
				    "(1 - cos 2u)/2, cos^2 = (1 + cos 2u)/2");
				alg_int_note = notebuf;
			}
			return F;
		}
	}
	{
		/* Bioche's invariants */
		struct mp mS = mp_neg(mp_atom(sv)), mC = mp_neg(mp_atom(cv));
		struct rf r1, r2, r3;
		r1.n = mp_subst(r.n, sv, mS);
		r1.d = mp_subst(r.d, sv, mS);
		r2.n = mp_subst(r.n, cv, mC);
		r2.d = mp_subst(r.d, cv, mC);
		r3.n = mp_subst(r1.n, cv, mC);
		r3.d = mp_subst(r1.d, cv, mC);
#define SAME(p, sg) mp_eq(mp_mul((p).n, r.d), mp_scale(mp_mul(r.n, (p).d), \
	qi(sg)))
		if (SAME(r1, -1)) {
			/* t = cos u: du = -dt/sin u */
			t = rf_div(r, rf_poly(mp_neg(mp_atom(sv))));
			t.n = mp_pyth(t.n, sv, cv, 0);
			t.d = mp_pyth(t.d, sv, cv, 0);
			t = rf_norm(t);
			how = "Bioche: t = ";
			T = C;
		} else if (SAME(r2, -1)) {
			t = rf_div(r, rf_poly(mp_atom(cv)));
			t.n = mp_pyth(t.n, cv, sv, 0);
			t.d = mp_pyth(t.d, cv, sv, 0);
			t = rf_norm(t);
			how = "Bioche: t = ";
			T = S;
		} else {
			/* tan or Weierstrass: substitute rational values */
			int tv = atom_of(x);
			struct rf sval, cval, one = rf_poly(mp_const(qi(1))), tt =
			    rf_poly(mp_atom(tv)), opt = rf_add(one, rf_mul(tt, tt));
			if (SAME(r3, 1)) {
				/* S = t C, C^2 = 1/(1 + t^2): only even powers of C stay */
				struct rf w;
				w.n = mp_subst(r.n, sv, mp_mul(mp_atom(tv), mp_atom(cv)));
				w.d = mp_subst(r.d, sv, mp_mul(mp_atom(tv), mp_atom(cv)));
				w = rf_norm(w);
				/* C^2 -> 1/(1 + t^2): numerator and denominator */
				{
					struct rf parts[2];
					int z, ok = 1;
					for (z = 0; z < 2; z++) {
						struct mp p = z ? w.d : w.n;
						int dg = mp_deg(p, cv), k;
						struct rf acc = rf_poly(mp_zero());
						for (k = 0; k <= dg; k++) {
							struct mp ck = mp_coef(p, cv, k);
							if (ck.n == 0)
								continue;
							if (k & 1) {
								ok = 0;
								break;
							}
							acc = rf_add(acc, rf_mul(rf_poly(ck), rf_pow(opt,
							    -k / 2)));
						}
						parts[z] = acc;
					}
					if (!ok) {
						ar_release(mk);
						return 0;
					}
					t = rf_div(parts[0], parts[1]);
					t = rf_div(t, opt);
				}
				how = "Bioche: t = ";
				T = mk_un(O_TAN, u);
			} else {
				struct rf w, tn, td;
				sval = rf_div(rf_mul(rf_poly(mp_const(qi(2))), tt), opt);
				cval = rf_div(rf_sub(one, rf_mul(tt, tt)), opt);
				/* n(S, C) with S, C rational in t */
				{
					struct rf parts[2];
					int z;
					for (z = 0; z < 2; z++) {
						struct mp p = z ? r.d : r.n;
						struct rf acc = rf_poly(mp_zero());
						int ii;
						for (ii = 0; ii < p.n; ii++) {
							struct rf mo = rf_poly(mp_const(p.t[ii].c));
							mo = rf_mul(mo, rf_pow(sval, p.t[ii].e[sv]));
							mo = rf_mul(mo, rf_pow(cval, p.t[ii].e[cv]));
							acc = rf_add(acc, mo);
						}
						parts[z] = acc;
					}
					tn = parts[0];
					td = parts[1];
				}
				w = rf_div(tn, td);
				t = rf_mul(w, rf_div(rf_poly(mp_const(qi(2))), opt));
				how = "Weierstrass: t = ";
				T = mk_un(O_TAN, DIV(u, KR(2)));
			}
			/* rename atom tv to index of a single variable */
			{
				struct up un, ud;
				if (alg_ovf || (mp_vars(t.n) | mp_vars(t.d)) & ~(1u << tv) ||
				    !up_of_mp(t.n, tv, &un) || !up_of_mp(t.d, tv, &ud)) {
					ar_release(mk);
					return 0;
				}
				res = rat_int(&un, &ud, x);
				ar_release(mk);
				if (res == 0)
					return 0;
				res = xsubst(res, T);
				res = DIV(res, KR(a));
				goto out;
			}
		}
#undef SAME
	}
	/* t = cos or sin: t is the atom T */
	{
		int v = T == C ? cv : sv;
		struct up un, ud;
		tm = t;
		if (alg_ovf || (mp_vars(tm.n) | mp_vars(tm.d)) & ~(1u << v) ||
		    !up_of_mp(tm.n, v, &un) || !up_of_mp(tm.d, v, &ud)) {
			ar_release(mk);
			return 0;
		}
		res = rat_int(&un, &ud, x);
		ar_release(mk);
		if (res == 0)
			return 0;
		res = DIV(xsubst(res, T), KR(a));
	}
out:
	res = rsub(res, mk_un(O_ATAN, mk_un(O_TAN, u)), u, 0);
	if (how) {
		snprintf(notebuf, sizeof notebuf, "%s%s%s", how, node_str(T),
		    !strncmp(how, "Weier", 5) ? "; the antiderivative holds on "
		    "each interval between the poles of the tangent" : "");
		alg_int_note = notebuf;
	}
	return res;
}

/*
 * sin(k u), cos(k u) ... (k integer) written in sin u, cos u with u = g x,
 * g the gcd of the slopes: sin x cos 2x -> sin x (cos^2 x - sin^2 x).
 * Returns f unchanged when every argument is the same.
 */
static nid
multi_angle(nid f, int *mixed)
{
	nid fn[24], g = f;
	int nf = 0, i, nt = 0;
	aq gq = qi(0), sl[24];
	double a0 = 0;

	*mixed = 0;
	walk(f, fn, &nf, 24, 0);
	for (i = 0; i < nf; i++) {
		int op = N.op[fn[i]];
		double a, b;
		if (op != O_SIN && op != O_COS && op != O_TAN && op != O_SEC &&
		    op != O_CSC && op != O_COT)
			continue;
		if (!lin_in(N.a[fn[i]], &a, &b) || b != 0 || !qexact(a, &sl[i]))
			return f;
		if (nt++ && a != a0)
			*mixed = 1;
		a0 = a;
		if (gq.n == 0)
			gq = sl[i];
		else {
			int64_t L = lcm64(gq.d, sl[i].d);
			gq = qmk(gcd64(gq.n * (L / gq.d), sl[i].n * (L / sl[i].d)), L);
		}
	}
	if (!*mixed)
		return f;
	{
		nid u = qmul_node(gq, I_x), S = mk_un(O_SIN, u), C = mk_un(O_COS, u);
		int sn = nat;
		nid sa[AV];
		size_t mk = ar_mark();
		memcpy(sa, at_node, sizeof sa);
		atoms_reset();
		atom_of(S);
		atom_of(C);
		for (i = 0; i < nf; i++) {
			int op = N.op[fn[i]], k, j;
			struct mp re, im, t;
			nid sn2, cn2, rep;
			aq kq;
			if (op != O_SIN && op != O_COS && op != O_TAN && op != O_SEC &&
			    op != O_CSC && op != O_COT)
				continue;
			kq = qdiv(sl[i], gq);
			k = (int)kq.n;
			if (kq.d != 1 || llabs(kq.n) > 12)
				continue;
			re = mp_const(qi(1));
			im = mp_zero();
			for (j = 0; j < (k < 0 ? -k : k); j++) {
				t = mp_sub(mp_mul(re, mp_atom(1)), mp_mul(im, mp_atom(0)));
				im = mp_add(mp_mul(re, mp_atom(0)), mp_mul(im, mp_atom(1)));
				re = t;
			}
			if (k < 0)
				im = mp_neg(im);
			sn2 = mp_node(im);
			cn2 = mp_node(re);
			rep = op == O_SIN ? sn2 : op == O_COS ? cn2 : op == O_TAN ?
			    DIV(sn2, cn2) : op == O_SEC ? DIV(KR(1), cn2) : op == O_CSC ?
			    DIV(KR(1), sn2) : DIV(cn2, sn2);
			g = rsub(g, fn[i], rep, 0);
		}
		ar_release(mk);
		nat = sn;
		memcpy(at_node, sa, sizeof sa);
	}
	return g;
}

/* P(x) times a polynomial in sin u, cos u: product to sum, then by parts */
#define TK 24
struct tpoly {
	aq c[TK + 1], s[TK + 1];      /* sum c_k cos(k u) + s_k sin(k u) */
};

static void
tp_mul1(struct tpoly *t, int sin)
{
	struct tpoly r;
	int k;

	for (k = 0; k <= TK; k++)
		r.c[k] = r.s[k] = qi(0);
	for (k = 0; k < TK; k++) {
		aq h = qmk(1, 2), cc = qmul(t->c[k], h), ss = qmul(t->s[k], h);
		/* k - 1 with cos(-a) = cos a, sin(-a) = -sin a */
		int km = k - 1 < 0 ? 1 - k : k - 1, sg = k - 1 < 0 ? -1 : 1;
		if (sin) {
			/* S cos(ku) = (sin((k+1)u) - sin((k-1)u))/2 */
			r.s[k + 1] = qadd(r.s[k + 1], cc);
			r.s[km] = qsub(r.s[km], sg > 0 ? cc : qneg(cc));
			/* S sin(ku) = (cos((k-1)u) - cos((k+1)u))/2 */
			r.c[km] = qadd(r.c[km], ss);
			r.c[k + 1] = qsub(r.c[k + 1], ss);
		} else {
			r.c[k + 1] = qadd(r.c[k + 1], cc);
			r.c[km] = qadd(r.c[km], cc);
			r.s[k + 1] = qadd(r.s[k + 1], ss);
			r.s[km] = qadd(r.s[km], sg > 0 ? ss : qneg(ss));
		}
	}
	r.s[0] = qi(0);
	*t = r;
}

static nid
triglin_int(nid f, nid x)
{
	nid fn[24], u = 0, S, C, sum = 0;
	int nf = 0, i, sv, cv, xv, ii, k, mixed;
	double a, b;
	struct rf r;
	size_t mk;
	aq dc;

	f = multi_angle(f, &mixed);
	walk(f, fn, &nf, 24, 0);
	for (i = 0; i < nf; i++) {
		int op = N.op[fn[i]];
		if (op == O_SIN || op == O_COS) {
			if (u == 0)
				u = N.a[fn[i]];
			else if (N.a[fn[i]] != u)
				return 0;
		} else if (op != O_NEG)
			return 0;
	}
	if (u == 0 || !lin_in(u, &a, &b))
		return 0;
	S = mk_un(O_SIN, u);
	C = mk_un(O_COS, u);
	mk = ar_mark();
	atoms_reset();
	xv = atom_of(x);
	sv = atom_of(S);
	cv = atom_of(C);
	if (!rf_of(f, &r, NULL) || !mp_isconst(r.d, &dc) ||
	    (!(mp_vars(r.n) & (1u << xv)) && !mixed) || (mp_vars(r.n) & ~7u)) {
		ar_release(mk);
		return 0;
	}
	{
		static struct tpoly acc[16];
		int e0max = -1, e0;
		for (ii = 0; ii < r.n.n; ii++) {
			struct tpoly t;
			e0 = r.n.t[ii].e[xv];
			if (r.n.t[ii].e[sv] + r.n.t[ii].e[cv] >= TK || e0 >= 16) {
				ar_release(mk);
				return 0;
			}
			for (; e0max < e0; e0max++)
				for (k = 0; k <= TK; k++)
					acc[e0max + 1].c[k] = acc[e0max + 1].s[k] = qi(0);
			for (k = 0; k <= TK; k++)
				t.c[k] = t.s[k] = qi(0);
			t.c[0] = qdiv(r.n.t[ii].c, dc);
			for (k = 0; k < r.n.t[ii].e[sv]; k++)
				tp_mul1(&t, 1);
			for (k = 0; k < r.n.t[ii].e[cv]; k++)
				tp_mul1(&t, 0);
			for (k = 0; k <= TK; k++) {
				acc[e0].c[k] = qadd(acc[e0].c[k], t.c[k]);
				acc[e0].s[k] = qadd(acc[e0].s[k], t.s[k]);
			}
		}
		for (e0 = e0max; e0 >= 0; e0--)
			for (k = 0; k <= TK; k++) {
				nid xp = e0 == 0 ? KR(1) : e0 == 1 ? x : POW(x, KI(e0));
				nid arg = k == 1 ? u : MUL(KR(k), u), t;
				if (acc[e0].c[k].n) {
					t = qmul_node(acc[e0].c[k], k ? (e0 ? MUL(xp,
					    mk_un(O_COS, arg)) : mk_un(O_COS, arg)) : xp);
					sum = sum ? ADD(sum, t) : t;
				}
				if (k && acc[e0].s[k].n) {
					t = qmul_node(acc[e0].s[k], e0 ? MUL(xp, mk_un(O_SIN,
					    arg)) : mk_un(O_SIN, arg));
					sum = sum ? ADD(sum, t) : t;
				}
			}
	}
	ar_release(mk);
	if (sum == 0 || alg_ovf || I_depth > 3)
		return 0;
	snprintf(notebuf, sizeof notebuf, "product to sum: %s%s",
	    node_str(sum), has_x(sum) && (mp_vars(r.n) & (1u << xv)) ?
	    ", then by parts" : "");
	{
		nid F = antideriv(sum, I_slot, NULL);
		if (F)
			alg_int_note = notebuf;
		return F;
	}
}

/* sinh, cosh, tanh of a linear argument */
static nid
hyp_int(nid f, nid x)
{
	double a, b;
	int op = N.op[f];

	if ((op != O_SINH && op != O_COSH && op != O_TANH) || !lin_in(N.a[f], &a,
	    &b))
		return 0;
	if (op == O_SINH)
		return DIV(mk_un(O_COSH, N.a[f]), KR(a));
	if (op == O_COSH)
		return DIV(mk_un(O_SINH, N.a[f]), KR(a));
	return DIV(mk_un(O_LOG, mk_un(O_COSH, N.a[f])), KR(a));
}

/* R(e^(a x)): u = e^(g x) */
static nid
exp_int(nid f, nid x)
{
	nid fn[24], g = f, T, res;
	int nf = 0, i;
	double sl[24];
	int nsl = 0;
	aq gq = qi(0);
	struct up n, d;

	walk(f, fn, &nf, 24, 0);
	for (i = 0; i < nf; i++) {
		int op = N.op[fn[i]];
		double a, b;
		aq q;
		if (op == O_SINH || op == O_COSH || op == O_TANH) {
			nid e1 = mk_un(O_EXP, N.a[fn[i]]), e2 = mk_un(O_EXP,
			    mk_un(O_NEG, N.a[fn[i]]));
			nid rep = op == O_SINH ? DIV(SUB(e1, e2), KR(2)) : op == O_COSH ?
			    DIV(ADD(e1, e2), KR(2)) : DIV(SUB(e1, e2), ADD(e1, e2));
			g = xreplace(g, fn[i], rep);
			continue;
		}
		if (op != O_EXP)
			continue;
		if (!lin_in(N.a[fn[i]], &a, &b) || !qexact(a, &q))
			return 0;
		(void)sl;
		(void)nsl;
		if (gq.n == 0)
			gq = q;
		else {
			/* gcd of rationals */
			int64_t gn = gcd64(gq.n * (lcm64(gq.d, q.d) / gq.d), q.n *
			    (lcm64(gq.d, q.d) / q.d));
			gq = qmk(gn, lcm64(gq.d, q.d));
		}
	}
	if (g != f)
		return exp_int(g, x);
	if (gq.n == 0)
		return 0;
	/* replace each exp(a x + b) by e^b T^(a/g); T stands in x's slot */
	{
		nf = 0;
		walk(g, fn, &nf, 24, 0);
		for (i = 0; i < nf; i++) {
			double a, b;
			aq q, k;
			if (N.op[fn[i]] != O_EXP)
				continue;
			lin_in(N.a[fn[i]], &a, &b);
			qexact(a, &q);
			k = qdiv(q, gq);
			g = xreplace(g, fn[i], MUL(b == 0 ? KR(1) : mk_un(O_EXP, KR(b)),
			    POW(x, KI(k.n))));
		}
	}
	if (has_slot(g, I_slot) && !as_upq(g, x, &n, &d))
		return 0;
	/* du = g u dx: the integrand in u is g(u)/(g u) */
	g = DIV(g, MUL(q_cnode(gq), x));
	if (!as_upq(g, x, &n, &d))
		return 0;
	res = rat_int(&n, &d, x);
	if (res == 0)
		return 0;
	T = mk_un(O_EXP, qmul_node(gq, x));
	res = xsubst(res, T);
	snprintf(notebuf, sizeof notebuf, "substitution u = %s", node_str(T));
	alg_int_note = notebuf;
	return res;
}

/* the integral of dx/sqrt(a x^2 + b x + c), a != 0 */
static nid
inv_sqrt_quad(aq a, aq b, aq c, nid x, nid s)
{
	aq k, D = qsub(qmul(qi(4), qmul(a, c)), qmul(b, b)), kk;
	int64_t m, mm;
	struct up lin;

	lin.d = 1;
	lin.c[1] = qmul(qi(2), a);
	lin.c[0] = b;
	if (qsgn(a) > 0) {
		sqrt_split(a, &k, &m);
		if (qsgn(D) > 0) {
			/* asinh((2a x + b)/sqrt(4ac - b^2)) / sqrt(a) */
			aq kd;
			int64_t md;
			sqrt_split(D, &kd, &md);
			return over_sqrt(qdiv(qi(1), k), m, mk_un(O_ASINH,
			    lin_over_sqrt(qdiv(lin.c[1], kd), qdiv(b, kd), md, x)));
		}
		/* log|2a x + b + 2 sqrt(a) sqrt(q)| / sqrt(a), common factors out */
		{
			aq tk = qmul(qi(2), k);
			int64_t L = lcm64(lcm64(lin.c[1].d, lin.c[0].d), tk.d), g;
			lin.c[1] = qmul(lin.c[1], qi(L));
			lin.c[0] = qmul(lin.c[0], qi(L));
			tk = qmul(tk, qi(L));
			g = gcd64(gcd64(lin.c[1].n, lin.c[0].n), tk.n);
			if (g > 1) {
				lin.c[1] = qdiv(lin.c[1], qi(g));
				lin.c[0] = qdiv(lin.c[0], qi(g));
				tk = qdiv(tk, qi(g));
			}
			up_trim(&lin);
			return over_sqrt(qdiv(qi(1), k), m, logabs(ADD(up_node(&lin, x),
			    qmul_node(tk, m == 1 ? s : MUL(sqrt_cnode(m), s)))));
		}
	}
	if (qsgn(D) >= 0)
		return 0;   /* q < 0 everywhere */
	/* asin((-2a x - b)/sqrt(b^2 - 4ac)) / sqrt(-a) */
	sqrt_split(qneg(a), &k, &m);
	sqrt_split(qneg(D), &kk, &mm);
	return over_sqrt(qdiv(qi(1), k), m, mk_un(O_ASIN,
	    lin_over_sqrt(qneg(qdiv(lin.c[1], kk)), qneg(qdiv(b, kk)), mm, x)));
}

/* integral of P(x)/sqrt(q): Q sqrt(q) + lambda * integral dx/sqrt(q) */
static nid
poly_over_sqrt(const struct up *P, const struct up *q, nid x, nid s)
{
	int n = P->d, i, j;
	struct q A[(UD + 2) * (UD + 2)], B[UD + 2];
	struct up Q, t1, t2, qd1;
	nid base, r = 0;

	if (n < 0)
		return KR(0);
	if (n > 12)
		return 0;
	up_diff(&qd1, q);
	/* unknowns c_0..c_{n-1} (Q), lambda; equations x^0..x^n */
	memset(A, 0, sizeof A);
	for (i = 0; i <= n; i++) {
		B[i] = i <= P->d ? P->c[i] : qi(0);
		for (j = 0; j <= n; j++)
			A[i * (n + 1) + j] = qi(0);
	}
	for (j = 0; j < n; j++) {
		/* the column of x^j: (x^j)' q + x^j q'/2 */
		struct up e, ed, c;
		e.d = j;
		for (i = 0; i <= j; i++)
			e.c[i] = qi(i == j);
		up_diff(&ed, &e);
		up_mul(&t1, &ed, q);
		up_mul(&t2, &e, &qd1);
		up_scale(&t2, &t2, qmk(1, 2));
		up_add(&c, &t1, &t2);
		for (i = 0; i <= c.d && i <= n; i++)
			A[i * (n + 1) + j] = c.c[i];
		if (c.d > n)
			return 0;
	}
	A[0 * (n + 1) + n] = qi(1);
	if (!solve_q(A, B, n + 1))
		return 0;
	Q.d = n - 1;
	for (j = 0; j < n; j++)
		Q.c[j] = B[j];
	up_trim(&Q);
	if (!up_iszero(&Q))
		r = poly_times(&Q, x, s);
	if (B[n].n) {
		base = q->d == 2 ? inv_sqrt_quad(q->c[2], q->c[1], q->c[0], x, s) : 0;
		if (q->d == 1)
			base = qmul_node(qdiv(qi(2), q->c[1]), s);
		if (base == 0)
			return 0;
		if (N.op[base] == O_NEG) {
			B[n] = qneg(B[n]);
			base = N.a[base];
		}
		base = qmul_node(B[n], base);
		r = r ? ADD(r, base) : base;
	}
	return r ? r : KR(0);
}

/* R(x, sqrt(q)), q of degree 1 or 2 */
static nid
sqrt_int(nid f, nid x)
{
	nid fn[24], s = 0, res = 0;
	int nf = 0, i, xv, sv;
	struct rf r;
	struct up q;
	size_t mk;

	walk(f, fn, &nf, 24, 0);
	for (i = 0; i < nf; i++)
		if (N.op[fn[i]] == O_SQRT) {
			if (s && fn[i] != s)
				return 0;
			s = fn[i];
		}
	if (s == 0)
		return 0;
	{
		struct up qq, dd;
		if (!as_upq(N.a[s], x, &qq, &dd) || dd.d != 0)
			return 0;
		up_scale(&q, &qq, qdiv(qi(1), dd.c[0]));
	}
	if (q.d < 1 || q.d > 2)
		return 0;
	mk = ar_mark();
	atoms_reset();
	xv = atom_of(x);
	sv = atom_of(s);
	/* POW(q, k/2) shows up as sqrt(q) atoms through rf_of */
	if (!rf_of(f, &r, NULL) || ((mp_vars(r.n) | mp_vars(r.d)) & ~3u)) {
		ar_release(mk);
		return 0;
	}
	if (q.d == 1) {
		/* t = sqrt(a x + b): x = (t^2 - b)/a, dx = 2t/a dt */
		aq a = q.c[1], b = q.c[0];
		struct mp tx = mp_scale(mp_sub(mp_mul(mp_atom(sv), mp_atom(sv)),
		    mp_const(b)), qdiv(qi(1), a));
		struct rf w;
		struct up un, ud;
		w.n = mp_subst(r.n, xv, tx);
		w.d = mp_subst(r.d, xv, tx);
		w = rf_norm(w);
		w = rf_mul(w, rf_poly(mp_scale(mp_atom(sv), qdiv(qi(2), a))));
		if (alg_ovf || !up_of_mp(w.n, sv, &un) || !up_of_mp(w.d, sv, &ud)) {
			ar_release(mk);
			return 0;
		}
		res = rat_int(&un, &ud, x);
		ar_release(mk);
		if (res)
			res = xsubst(res, s);
		snprintf(notebuf, sizeof notebuf, "substitution t = %s",
		    node_str(s));
		alg_int_note = notebuf;
		return res;
	}
	{
		/* reduce s^2 = q: num = N0 + N1 s, den = D0 + D1 s */
		struct mp qm = mp_of_up(&q, xv), parts[2][2];
		struct mp E, R1n, R2n;
		struct up a1, b1, a2, b2;
		int z, k;
		for (z = 0; z < 2; z++) {
			struct mp p = z ? r.d : r.n;
			parts[z][0] = mp_zero();
			parts[z][1] = mp_zero();
			for (k = 0; k <= mp_deg(p, sv); k++) {
				struct mp ck = mp_coef(p, sv, k);
				if (ck.n == 0)
					continue;
				parts[z][k & 1] = mp_add(parts[z][k & 1], mp_mul(ck,
				    mp_pow(qm, k / 2)));
			}
		}
		/* times (D0 - D1 s) */
		E = mp_sub(mp_mul(parts[1][0], parts[1][0]), mp_mul(mp_mul(parts[1][1],
		    parts[1][1]), qm));
		R1n = mp_sub(mp_mul(parts[0][0], parts[1][0]),
		    mp_mul(mp_mul(parts[0][1], parts[1][1]), qm));
		R2n = mp_sub(mp_mul(parts[0][1], parts[1][0]),
		    mp_mul(parts[0][0], parts[1][1]));
		/* f = R1n/E + (R2n q / E) / s */
		R2n = mp_mul(R2n, qm);
		if (alg_ovf || !up_of_mp(R1n, xv, &a1) || !up_of_mp(E, xv, &b1) ||
		    !up_of_mp(R2n, xv, &a2) || !up_of_mp(E, xv, &b2)) {
			ar_release(mk);
			return 0;
		}
		ar_release(mk);
		{
			struct tl l;
			struct up pol, g, nn, dd;
			struct pfterm *t = xmalloc(sizeof *t * 64);
			int nt, partial;
			l.n = 0;
			if (!up_iszero(&a1)) {
				nid r1 = rat_int(&a1, &b1, x);
				if (r1 == 0) {
					free(t);
					return 0;
				}
				tl_add(&l, r1);
			}
			if (!up_iszero(&a2)) {
				nid pr;
				up_gcd(&g, &a2, &b2);
				up_divmod(&a2, &g, &nn, NULL);
				up_divmod(&b2, &g, &dd, NULL);
				nt = up_apart(&nn, &dd, &pol, t, 64, &partial);
				if (nt < 0) {
					free(t);
					return 0;
				}
				pr = poly_over_sqrt(&pol, &q, x, s);
				if (pr == 0) {
					free(t);
					return 0;
				}
				tl_add(&l, pr);
				for (i = 0; i < nt; i++) {
					/* c/(x - al)^k / s: t = 1/(x - al) */
					aq al, c, qa, A2, B2, C2;
					struct up qt, tk;
					nid G, sgn, ts;
					if (t[i].f.d != 1 || t[i].a.d > 0) {
						free(t);
						return 0;
					}
					al = qneg(t[i].f.c[0]);
					c = t[i].a.c[0];
					qa = up_evalq(&q, al);
					/* q(al + 1/t) t^2 = qa t^2 + (2 a al + b) t + a */
					A2 = qa;
					B2 = qadd(qmul(qmul(qi(2), q.c[2]), al), q.c[1]);
					C2 = q.c[2];
					qt.d = 2;
					qt.c[2] = A2;
					qt.c[1] = B2;
					qt.c[0] = C2;
					up_trim(&qt);
					if (qt.d < 1) {
						free(t);
						return 0;
					}
					tk.d = t[i].k - 1;
					for (k = 0; k < t[i].k - 1; k++)
						tk.c[k] = qi(0);
					tk.c[t[i].k - 1] = c;
					/* the integral in t of t^(k-1)/sqrt(qt(t)), x stands for t */
					{
						nid st = mk_un(O_SQRT, up_node(&qt, x));
						G = poly_over_sqrt(&tk, &qt, x, st);
					}
					if (G == 0) {
						free(t);
						return 0;
					}
					ts = DIV(KR(1), SUB(x, q_cnode(al)));
					if (al.n == 0)
						ts = DIV(KR(1), x);
					G = xsubst(G, ts);
					sgn = mk_un(O_SIGN, al.n ? SUB(x, q_cnode(al)) : x);
					tl_add(&l, mk_un(O_NEG, MUL(sgn, G)));
				}
			}
			free(t);
			res = tl_sum(&l);
		}
	}
	snprintf(notebuf, sizeof notebuf, "sqrt of a quadratic: "
	    "Q(x) sqrt(q) + c * integral of "
	    "dx/sqrt(q) (Ostrogradsky)");
	alg_int_note = notebuf;
	return res;
}

static int
is_logish(int op)
{
	return op == O_LOG || op == O_ATAN || op == O_ASIN || op == O_ACOS ||
	    op == O_ASINH || op == O_ACOSH || op == O_ATANH;
}

static int
has_logish(nid e, int depth)
{
	if (e == 0 || depth > 60 || !has_x(e))
		return 0;
	if (is_logish(N.op[e]))
		return 1;
	return has_logish(N.a[e], depth + 1) || has_logish(N.b[e], depth + 1);
}

/* by parts: A * B, B with log/atan/asin...: (int A) B - int (int A) B' */
static nid
parts_int(nid f, nid x)
{
	nid A = 0, B = 0, F, G, d;
	int w;

	if (N.op[f] == O_MUL) {
		for (w = 0; w < 2 && !B; w++) {
			nid p = w ? N.b[f] : N.a[f], q = w ? N.a[f] : N.b[f];
			if (has_logish(q, 0) && !has_logish(p, 0)) {
				A = p;
				B = q;
			}
		}
		/* exp(a x + b) sin|cos(c x + d) */
		for (w = 0; w < 2; w++) {
			nid e = w ? N.b[f] : N.a[f], t = w ? N.a[f] : N.b[f];
			double a, b, c, dd;
			aq qa, qc, den;
			if (N.op[e] == O_EXP && (N.op[t] == O_SIN || N.op[t] == O_COS) &&
			    lin_in(N.a[e], &a, &b) && lin_in(N.a[t], &c, &dd) &&
			    qexact(a, &qa) && qexact(c, &qc)) {
				nid sn = mk_un(O_SIN, N.a[t]), cs = mk_un(O_COS, N.a[t]), in;
				den = qadd(qmul(qa, qa), qmul(qc, qc));
				if (N.op[t] == O_SIN)
					in = SUB(qmul_node(qdiv(qa, den), sn), qmul_node(qdiv(qc,
					    den), cs));
				else
					in = ADD(qmul_node(qdiv(qa, den), cs), qmul_node(qdiv(qc,
					    den), sn));
				snprintf(notebuf, sizeof notebuf, "by parts twice "
				    "(the integral returns)");
				alg_int_note = notebuf;
				return MUL(e, in);
			}
		}
	} else if (has_logish(f, 0) && (is_logish(N.op[f]) || N.op[f] == O_POW)) {
		A = KR(1);
		B = f;
	}
	if (!B || I_depth > 3)
		return 0;
	F = has_x(A) ? antideriv(A, I_slot, NULL) : MUL(A, I_x);
	if (F == 0)
		return 0;
	d = try_diff(B);
	if (d == 0)
		return 0;
	G = antideriv(alg_simplify(MUL(F, d), NULL), I_slot, NULL);
	if (G == 0)
		return 0;
	snprintf(notebuf, sizeof notebuf, "by parts (u = %s)", node_str(B));
	alg_int_note = notebuf;
	return flat(MUL(F, B), G, 1);
}

/* substitution: f = h(u) u' */
static nid
subst_int(nid f, nid x)
{
	nid fn[24], P;
	int nf = 0, i;
	const nid T = mk_var(60, sym_cstr("u"), TY_REAL);

	walk(f, fn, &nf, 24, 0);
	/* the arguments, and the subterms themselves */
	for (i = 0, P = 0; i < nf && nf < 24; i++) {
		nid a = N.a[fn[i]];
		int j;
		if (N.op[a] == O_VAR || N.op[a] == O_CAST)
			continue;
		for (j = 0; j < nf; j++)
			if (fn[j] == a)
				break;
		if (j == nf && nf < 24)
			fn[nf++] = a;
	}
	(void)P;
	for (i = 0; i < nf; i++) {
		nid u = fn[i], du, q, h, H;
		double a, b;
		if (lin_in(u, &a, &b) && N.op[u] != O_EXP)
			continue;
		du = try_diff(u);
		if (du == 0 || const_is(du, 0))
			continue;
		q = alg_simplify(DIV(f, du), NULL);
		h = xreplace(q, u, T);
		if (has_x(h)) {
			/* u = a x^k + b: x^k = (T - b)/a */
			struct npoly p;
			int k;
			if (!npoly_of(u, I_slot, &p, 8))
				continue;
			for (k = 1; k < p.deg && const_is(p.c[k], 0); k++)
				;
			if (k != p.deg || k < 2 || !num_of(p.c[k], &a) ||
			    !num_of(p.c[0], &b))
				continue;
			{
				nid xk = POW(x, KI(k)), rep = DIV(SUB(T, KR(b)), KR(a));
				h = xreplace(h, xk, rep);
				if (has_x(h)) {
					/* through the polynomial form */
					struct rf r;
					size_t mk = ar_mark();
					int xv, tv, ok = 1, z;
					atoms_reset();
					xv = atom_of(x);
					tv = atom_of(T);
					if (!rf_of(h, &r, NULL) || ((mp_vars(r.n) | mp_vars(r.d)) &
					    ~((1u << xv) | (1u << tv)))) {
						ar_release(mk);
						continue;
					}
					for (z = 0; z < 2 && ok; z++) {
						struct mp m = z ? r.d : r.n, out = mp_zero();
						int t;
						struct mp rp;
						aq qb, qa2;
						if (!qexact(b, &qb) || !qexact(a, &qa2)) {
							ok = 0;
							break;
						}
						rp = mp_scale(mp_sub(mp_atom(tv), mp_const(qb)),
						    qdiv(qi(1), qa2));
						for (t = 0; t < m.n; t++) {
							struct mp mo;
							int e = m.t[t].e[xv];
							if (e % k) {
								ok = 0;
								break;
							}
							mo.n = 1;
							mo.t = &m.t[t];
							mo = mp_copy(mo);
							mo.t[0].e[xv] = 0;
							out = mp_add(out, mp_mul(mo, mp_pow(rp, e / k)));
						}
						if (z)
							r.d = out;
						else
							r.n = out;
					}
					if (ok) {
						r = rf_norm(r);
						h = rf_node(r, 0);
					}
					ar_release(mk);
					if (!ok || has_x(h))
						continue;
				}
			}
		}
		if (has_x(h) || I_depth > 3)
			continue;
		{
			nid map[NSLOT];
			memset(map, 0, sizeof map);
			map[60] = x;
			h = subst(h, map, NSLOT);
		}
		H = antideriv(h, I_slot, NULL);
		if (H == 0)
			continue;
		H = xsubst(H, u);
		snprintf(notebuf, sizeof notebuf, "substitution u = %s",
		    node_str(u));
		alg_int_note = notebuf;
		return H;
	}
	return 0;
}

static nid
sub_int(nid h)
{
	return antideriv(h, I_slot, NULL);
}

/*
 * first: only the rational case (before integ.c's rules); otherwise
 * every method, after them.
 */
nid
alg_antideriv(nid f, int s, nid x, int first, const char **how)
{
	struct up n, d;
	nid r = 0;
	int saved_slot = I_slot;
	size_t mk = ar_mark();
	const char *saved_note = alg_int_note;

	nid saved_x = I_x;

	(void)sub_int;
	I_slot = s;
	I_x = x;
	if (first) {
		alg_int_note = NULL;
		if (as_upq(f, x, &n, &d) && d.d >= 1) {
			numeric_used = 0;
			hermite_used = 0;
			r = rat_int(&n, &d, x);
			if (r && numeric_used) {
				snprintf(notebuf, sizeof notebuf, "partial fractions; "
				    "the denominator has irreducible factors of degree >= 3:"
				    " their roots are numeric, so the coefficients are "
				    "decimals");
				alg_int_note = notebuf;
			} else if (r)
				alg_int_note = hermite_used ? "partial fractions, Hermite "
				    "reduction for the repeated factors" : "partial fractions";
		} else if (I_depth == 0) {
			/* R(x, sqrt(quadratic)) before the older sqrt rules */
			nid c;
			I_depth++;
			c = sqrt_int(f, x);
			I_depth--;
			if (c && verified(c, f))
				r = alg_light(c);
			else
				alg_int_note = NULL;
		}
		ar_release(mk);
		I_slot = saved_slot;
		I_x = saved_x;
		/* plain partial fractions need no method line */
		if (how && r && alg_int_note && strcmp(alg_int_note,
		    "partial fractions"))
			*how = alg_int_note;
		return r;
	}
	if (I_depth > 4) {
		I_slot = saved_slot;
		I_x = saved_x;
		return 0;
	}
	I_depth++;
	alg_int_note = NULL;
	/* linearity */
	if (N.op[f] == O_ADD || N.op[f] == O_SUB) {
		nid A = antideriv(N.a[f], s, NULL), B = A ? antideriv(N.b[f], s,
		    NULL) : 0;
		if (A && B)
			r = flat(A, B, N.op[f] == O_SUB);
		alg_int_note = NULL;
	} else if (N.op[f] == O_NEG) {
		nid A = antideriv(N.a[f], s, NULL);
		if (A)
			r = mk_un(O_NEG, A);
	} else if (N.op[f] == O_MUL && !has_slot(N.a[f], s)) {
		nid A = antideriv(N.b[f], s, NULL);
		if (A)
			r = MUL(N.a[f], A);
	} else if (N.op[f] == O_DIV && !has_slot(N.b[f], s)) {
		nid A = antideriv(N.a[f], s, NULL);
		if (A)
			r = DIV(A, N.b[f]);
	}
	if (r && !verified(r, f))
		r = 0;
	{
		nid (*m[])(nid, nid) = {hyp_int, exp_int, triglin_int, trig_int,
		    sqrt_int, parts_int, subst_int};
		const char *saved = alg_int_note;
		int k;
		for (k = 0; k < 7 && r == 0; k++) {
			nid c;
			alg_int_note = NULL;
			c = m[k](f, x);
			if (c && verified(c, f))
				r = c;
			else
				alg_int_note = saved;
		}
	}
	I_depth--;
	if (r == 0)
		alg_int_note = saved_note;
	if (r) {
		/* the normal form when it is shorter (power reduction's nests) */
		nid t = alg_simplify(r, NULL), l = alg_light(r);
		r = strlen(node_str(t)) < strlen(node_str(l)) && verified(t, f) ? t : l;
	}
	ar_release(mk);
	I_slot = saved_slot;
	I_x = saved_x;
	if (how && r)
		*how = alg_int_note ? alg_int_note : "";
	return r;
}
