/*
 * solve: equations and systems.  Every solution is substituted back and
 * the residual printed.
 *
 *   polynomial (after clearing denominators): factored over Q, roots
 *     exact for linear and quadratic factors, cubics by the trigonometric
 *     or hyperbolic form, quartics that split over Q(sqrt d); the rest
 *     numeric, with a note.  Points where a denominator vanishes are
 *     excluded (and said so).  Symbolic coefficients: degree 1 and 2 by
 *     formula, with the condition that the leading coefficient is not 0.
 *   one function of x set to a value (sin u = c, exp u = c, ...), or a
 *     polynomial in it: inverted, with the general solution in k.
 *   linear systems: exact elimination (a unique solution, a family with
 *     free unknowns, or none); two polynomial equations in two unknowns
 *     by the resultant.
 *   otherwise: sign changes on a grid, refined by bisection (numeric).
 */
#include "alg.h"
#include "calc2.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- text of exact numbers ---- */

static char *
fq(aq c)
{
	static char b[16][64];
	static int k;
	char *s = b[k++ & 15];

	if (c.d == 1)
		snprintf(s, 64, "%lld", (long long)c.n);
	else
		snprintf(s, 64, "%lld/%lld", (long long)c.n, (long long)c.d);
	return s;
}

/* c sqrt(m) as text */
static char *
fsurd(aq c, int64_t m)
{
	static char b[16][96];
	static int k;
	char *s = b[k++ & 15];
	int neg = c.n < 0;

	if (m == 1)
		return fq(c);
	if (neg)
		c.n = -c.n;
	if (c.n == 1 && c.d == 1)
		snprintf(s, 96, "%ssqrt(%lld)", neg ? "-" : "", (long long)m);
	else if (c.d == 1)
		snprintf(s, 96, "%s%lld*sqrt(%lld)", neg ? "-" : "", (long long)c.n,
		    (long long)m);
	else if (c.n == 1)
		snprintf(s, 96, "%ssqrt(%lld)/%lld", neg ? "-" : "", (long long)m,
		    (long long)c.d);
	else
		snprintf(s, 96, "%s%lld*sqrt(%lld)/%lld", neg ? "-" : "",
		    (long long)c.n, (long long)m, (long long)c.d);
	return s;
}

static char *
fqs(struct qs x)
{
	static char b[16][160];
	static int k;
	char *s = b[k++ & 15];

	if (x.b.n == 0)
		return fq(x.a);
	if (x.a.n == 0)
		return fsurd(x.b, qs_d);
	if (x.b.n < 0)
		snprintf(s, 160, "%s - %s", fq(x.a), fsurd(qneg(x.b), qs_d));
	else
		snprintf(s, 160, "%s + %s", fq(x.a), fsurd(x.b, qs_d));
	return s;
}

static int
qs_zero2(struct qs x)
{
	return x.a.n == 0 && x.b.n == 0;
}

/* ---- solutions ---- */

#define MAXSOL 64
struct sol {
	char text[256];
	double v;
	int mult;
	int general;      /* contains k */
	double per;       /* the period in k */
};
static struct sol S[MAXSOL];
static int nS;
static int complex_n;
static char excl[512];
static int symbolic_done;

static void
addsol(const char *text, double v, int mult)
{
	int i;

	for (i = 0; i < nS; i++)
		if (!S[i].general && fabs(S[i].v - v) <= 1e-12 * fmax(1, fabs(v))) {
			S[i].mult += mult;
			return;
		}
	if (nS < MAXSOL) {
		snprintf(S[nS].text, sizeof S[nS].text, "%s", text);
		S[nS].v = v;
		S[nS].mult = mult;
		S[nS].general = 0;
		nS++;
	}
}

/* the real roots of an irreducible f (monic, over Q) */
static void
roots_irred(const struct up *f, int mult)
{
	char t[256];

	if (f->d == 1) {
		aq r = qneg(f->c[0]);
		addsol(fq(r), qd(r), mult);
		return;
	}
	if (f->d == 2) {
		aq b = f->c[1], c = f->c[0], D = qsub(qmul(b, b), qmul(qi(4), c)), k,
		    h = qneg(qdiv(b, qi(2)));
		int64_t m;
		if (qsgn(D) < 0) {
			complex_n += 2 * mult;
			return;
		}
		sqrt_split(D, &k, &m);
		k = qdiv(k, qi(2));
		/* -b/2 +- k sqrt(m) */
		{
			int s;
			for (s = -1; s <= 1; s += 2) {
				if (h.n == 0)
					snprintf(t, sizeof t, "%s", fsurd(s < 0 ? qneg(k) : k, m));
				else
					snprintf(t, sizeof t, "%s %s %s", fq(h), s < 0 ? "-" : "+",
					    fsurd(k, m));
				addsol(t, qd(h) + s * qd(k) * sqrt((double)m), mult);
			}
		}
		return;
	}
	{
		/* x^n = c */
		int j, bin = f->d >= 3;
		for (j = 1; j < f->d && bin; j++)
			if (f->c[j].n)
				bin = 0;
		if (bin) {
			aq c = qneg(f->c[0]);
			int n = f->d;
			double v;
			if (qsgn(c) < 0 && n % 2 == 0) {
				complex_n += n * mult;
				return;
			}
			v = pow(fabs(qd(c)), 1.0 / n);
			if (c.d == 1)
				snprintf(t, sizeof t, "%s%lld^(1/%d)", qsgn(c) < 0 ? "-" : "",
				    (long long)llabs(c.n), n);
			else
				snprintf(t, sizeof t, "%s(%s)^(1/%d)", qsgn(c) < 0 ? "-" : "",
				    fq(qsgn(c) < 0 ? qneg(c) : c), n);
			addsol(t, qsgn(c) < 0 ? -v : v, mult);
			if (n % 2 == 0) {
				char t2[256];
				snprintf(t2, sizeof t2, "-%s", t);
				addsol(t2, -v, mult);
				complex_n += (n - 2) * mult;
			} else
				complex_n += (n - 1) * mult;
			return;
		}
	}
	if (f->d == 3) {
		/* x = t - a2/3: t^3 + p t + q */
		aq a2 = f->c[2], a1 = f->c[1], a0 = f->c[0];
		aq sh = qdiv(a2, qi(3));
		aq p = qsub(a1, qdiv(qmul(a2, a2), qi(3)));
		aq q = qadd(qsub(qdiv(qmul(qi(2), qmul(a2, qmul(a2, a2))), qi(27)),
		    qdiv(qmul(a2, a1), qi(3))), a0);
		double pd = qd(p), qdv = qd(q), disc = -(4 * pd * pd * pd + 27 * qdv *
		    qdv);
		char shs[64];
		if (sh.n == 0)
			shs[0] = 0;
		else
			snprintf(shs, sizeof shs, " %s %s", qsgn(sh) > 0 ? "-" : "+",
			    fq(qsgn(sh) > 0 ? sh : qneg(sh)));
		if (disc > 0 && pd < 0) {
			/* 2 sqrt(-p/3) cos(acos(3q/(2p) sqrt(-3/p))/3 - 2 pi k/3) */
			aq A, Ak, Qc, Qk;
			int64_t Am, Qm, kk;
			sqrt_split(qdiv(qneg(p), qi(3)), &Ak, &Am);
			A = qmul(qi(2), Ak);
			sqrt_split(qdiv(qi(-3), p), &Qk, &Qm);
			Qc = qmul(qdiv(qmul(qi(3), q), qmul(qi(2), p)), Qk);
			for (kk = 0; kk < 3; kk++) {
				double v = qd(A) * sqrt((double)Am) * cos(acos(qd(Qc) *
				    sqrt((double)Qm)) / 3 - 2 * M_PI * kk / 3) - qd(sh);
				snprintf(t, sizeof t, "%s*cos(acos(%s)/3%s)%s", fsurd(A, Am),
				    fsurd(Qc, Qm), kk == 0 ? "" : kk == 1 ? " - 2pi/3" :
				    " - 4pi/3", shs);
				addsol(t, v, mult);
			}
			return;
		}
		if (pd < 0) {
			/* one real root: -2 sign(q) sqrt(-p/3) cosh(acosh(-3|q|/(2p) sqrt(-3/p))/3) */
			aq Ak, Qk, Qc;
			int64_t Am, Qm;
			double v;
			sqrt_split(qdiv(qneg(p), qi(3)), &Ak, &Am);
			sqrt_split(qdiv(qi(-3), p), &Qk, &Qm);
			Qc = qmul(qdiv(qmul(qi(-3), qsgn(q) < 0 ? qneg(q) : q),
			    qmul(qi(2), p)), Qk);
			Ak = qmul(qi(qsgn(q) > 0 ? -2 : 2), Ak);
			v = qd(Ak) * sqrt((double)Am) * cosh(acosh(qd(Qc) *
			    sqrt((double)Qm)) / 3) - qd(sh);
			snprintf(t, sizeof t, "%s*cosh(acosh(%s)/3)%s", fsurd(Ak, Am),
			    fsurd(Qc, Qm), shs);
			addsol(t, v, mult);
			complex_n += 2 * mult;
			return;
		}
		if (pd > 0) {
			/* -2 sqrt(p/3) sinh(asinh(3q/(2p) sqrt(3/p))/3) */
			aq Ak, Qk, Qc;
			int64_t Am, Qm;
			double v;
			sqrt_split(qdiv(p, qi(3)), &Ak, &Am);
			sqrt_split(qdiv(qi(3), p), &Qk, &Qm);
			Qc = qmul(qdiv(qmul(qi(3), q), qmul(qi(2), p)), Qk);
			Ak = qmul(qi(-2), Ak);
			v = qd(Ak) * sqrt((double)Am) * sinh(asinh(qd(Qc) *
			    sqrt((double)Qm)) / 3) - qd(sh);
			snprintf(t, sizeof t, "%s*sinh(asinh(%s)/3)%s", fsurd(Ak, Am),
			    fsurd(Qc, Qm), shs);
			addsol(t, v, mult);
			complex_n += 2 * mult;
			return;
		}
	}
	if (f->d == 4) {
		struct qs p[2], q[2];
		if (quartic_split(f, p, q)) {
			int j, s;
			for (j = 0; j < 2; j++) {
				/* x^2 + p x + q: (-p +- sqrt(p^2 - 4q))/2 */
				struct qs D = qs_sub(qs_mul(p[j], p[j]), qs_mul((struct qs){
				    qi(4), qi(0)}, q[j])), R, h = qs_div(p[j], (struct qs){
				    qi(-2), qi(0)});
				double dv = qs_val(D);
				if (dv < 0) {
					complex_n += 2 * mult;
					continue;
				}
				for (s = -1; s <= 1; s += 2) {
					double v = qs_val(h) + s * sqrt(dv) / 2;
					if (D.b.n == 0) {
						aq kk;
						int64_t mm;
						sqrt_split(D.a, &kk, &mm);
						kk = qdiv(kk, qi(2));
						if (qs_zero2(h))
							snprintf(t, sizeof t, "%s", fsurd(s < 0 ? qneg(kk) :
							    kk, mm));
						else
							snprintf(t, sizeof t, "%s %s %s", fqs(h), s < 0 ?
							    "-" : "+", fsurd(kk, mm));
					} else if (qs_sqrt(D, &R)) {
						struct qs half = {qmk(1, 2), qi(0)}, r = qs_add(h,
						    qs_mul(qs_mul(R, half), (struct qs){qi(s), qi(0)}));
						snprintf(t, sizeof t, "%s", fqs(r));
					} else if (h.a.n == 0 && h.b.n == 0)
						snprintf(t, sizeof t, "%ssqrt(%s)/2", s < 0 ? "-" : "",
						    fqs(D));
					else
						snprintf(t, sizeof t, "%s %s sqrt(%s)/2", fqs(h),
						    s < 0 ? "-" : "+", fqs(D));
					addsol(t, v, mult);
				}
			}
			return;
		}
	}
	/* numeric */
	{
		double c[UD + 1];
		struct root r[UD + 1];
		int n, i;
		if (f->d > 14)
			return;
		for (i = 0; i <= f->d; i++)
			c[i] = qd(f->c[i]);
		n = poly_roots(c, f->d, r);
		for (i = 0; i < n; i++) {
			if (r[i].im != 0) {
				complex_n += 2 * mult;
				continue;
			}
			snprintf(t, sizeof t, "%.15g", r[i].re);
			addsol(t, r[i].re, mult);
			if (nS)
				S[nS - 1].general = -1;   /* numeric */
		}
	}
}

/* the real roots of u (over Q) */
static void
roots_up(const struct up *u)
{
	struct up f[UD];
	int m[UD], n, i, partial;

	if (u->d < 1)
		return;
	n = up_factor(u, f, m, UD, &partial);
	for (i = 0; i < n; i++)
		roots_irred(&f[i], m[i]);
}

/* ---- inverting a function ---- */

static double
evalx(nid e, int slot, double x, int *ok)
{
	return eval_const_at(e, slot, x, ok);
}

/* v as exact-looking text if c2_fnum finds a form, else fn(arg) */
static void
ntext(char *out, size_t n, double v, const char *fn, const char *arg)
{
	const char *t = c2_fnum(v);

	if (strchr(t, '.') == NULL && strchr(t, 'e') == NULL)
		snprintf(out, n, "%s", t);
	else
		snprintf(out, n, "%s(%s)", fn, arg);
}

static int
simple_text(const char *t)
{
	return strchr(t, ' ') == NULL && (t[0] != '-' || strchr(t + 1, '-') ==
	    NULL);
}

/*
 * solutions of A(u) = c with u = a x + b: x = (base - b)/a [+ per/|a| k];
 * pb is the base of a power c^u (op O_POW)
 */
static int
invert(int op, double c, double a, double b, double pb, int general_ok)
{
	char bt[2][128], cs[64], t[256];
	double base[2], per = 0;
	int nb = 1, i;

	snprintf(cs, sizeof cs, "%s", c2_fnum(c));
	switch (op) {
	case O_SIN:
	case O_COS:
		if (fabs(c) > 1)
			return 0;
		base[0] = op == O_SIN ? asin(c) : acos(c);
		ntext(bt[0], sizeof bt[0], base[0], op == O_SIN ? "asin" : "acos", cs);
		if (fabs(fabs(c) - 1) > 1e-15 && !(op == O_COS && c == -1)) {
			nb = 2;
			base[1] = op == O_SIN ? M_PI - base[0] : -base[0];
			if (strchr(bt[0], '(') == NULL)
				snprintf(bt[1], sizeof bt[1], "%s", c2_fnum(base[1]));
			else
				snprintf(bt[1], sizeof bt[1], "%s%s", op == O_SIN ? "pi - " :
				    "-", bt[0]);
		}
		per = 2 * M_PI;
		break;
	case O_TAN:
		base[0] = atan(c);
		ntext(bt[0], sizeof bt[0], base[0], "atan", cs);
		per = M_PI;
		break;
	case O_EXP:
		if (c <= 0)
			return 0;
		base[0] = log(c);
		ntext(bt[0], sizeof bt[0], base[0], "log", cs);
		break;
	case O_POW:
		if (c <= 0 || pb <= 0 || pb == 1)
			return 0;
		base[0] = log(c) / log(pb);
		{
			const char *q = c2_fnum(base[0]);
			if (strchr(q, '.') == NULL)
				snprintf(bt[0], sizeof bt[0], "%s", q);
			else
				snprintf(bt[0], sizeof bt[0], "log(%s)/log(%s)", cs,
				    c2_fnum(pb));
		}
		break;
	case O_LOG:
		base[0] = exp(c);
		ntext(bt[0], sizeof bt[0], base[0], "exp", cs);
		break;
	case O_SQRT:
		if (c < 0)
			return 0;
		base[0] = c * c;
		snprintf(bt[0], sizeof bt[0], "%s", c2_fnum(c * c));
		break;
	case O_ABS:
	case O_COSH:
		if (c < (op == O_ABS ? 0 : 1))
			return 0;
		base[0] = op == O_ABS ? c : acosh(c);
		if (op == O_ABS)
			snprintf(bt[0], sizeof bt[0], "%s", cs);
		else
			ntext(bt[0], sizeof bt[0], base[0], "acosh", cs);
		if (base[0] != 0) {
			nb = 2;
			base[1] = -base[0];
			snprintf(bt[1], sizeof bt[1], "-%s", bt[0]);
		}
		break;
	case O_SINH:
		base[0] = asinh(c);
		ntext(bt[0], sizeof bt[0], base[0], "asinh", cs);
		break;
	case O_TANH:
		if (fabs(c) >= 1)
			return 0;
		base[0] = atanh(c);
		ntext(bt[0], sizeof bt[0], base[0], "atanh", cs);
		break;
	case O_ATAN:
		if (fabs(c) >= M_PI / 2)
			return 0;
		base[0] = tan(c);
		ntext(bt[0], sizeof bt[0], base[0], "tan", cs);
		break;
	case O_ASIN:
	case O_ACOS:
		if (op == O_ASIN ? fabs(c) > M_PI / 2 : (c < 0 || c > M_PI))
			return 0;
		base[0] = op == O_ASIN ? sin(c) : cos(c);
		ntext(bt[0], sizeof bt[0], base[0], op == O_ASIN ? "sin" : "cos",
		    cs);
		break;
	default:
		return 0;
	}
	for (i = 0; i < nb; i++) {
		double x0 = (base[i] - b) / a, xp = per / fabs(a);
		char num[160], x0t[200];
		const char *q = c2_fnum(x0);
		if (strchr(q, '.') == NULL && strchr(q, 'e') == NULL)
			snprintf(x0t, sizeof x0t, "%s", q);
		else {
			if (b == 0)
				snprintf(num, sizeof num, "%s", bt[i]);
			else
				snprintf(num, sizeof num, "%s %s %s", bt[i], b > 0 ? "-" : "+",
				    c2_fnum(fabs(b)));
			if (a == 1)
				snprintf(x0t, sizeof x0t, "%s", num);
			else if (a == -1)
				snprintf(x0t, sizeof x0t, simple_text(num) ? "-%s" : "-(%s)",
				    num);
			else
				snprintf(x0t, sizeof x0t, simple_text(num) ? "%s/%s" :
				    "(%s)/%s", num, c2_fnum(a));
		}
		if (per && general_ok) {
			const char *pt = c2_fnum(xp);
			if (!strcmp(x0t, "0"))
				snprintf(t, sizeof t, "%s*k", pt);
			else
				snprintf(t, sizeof t, "%s + %s*k", x0t, pt);
			if (nS < MAXSOL) {
				snprintf(S[nS].text, sizeof S[nS].text, "%s", t);
				S[nS].v = x0;
				S[nS].per = xp;
				S[nS].mult = 1;
				S[nS].general = 1;
				nS++;
			}
		} else
			addsol(x0t, x0, 1);
	}
	return 1;
}

/* ---- the single-equation solver ---- */

static int
lin_coef(nid u, int slot, double *a, double *b)
{
	struct npoly p;

	if (!npoly_of(u, slot, &p, 1) || p.deg != 1 || !num_of(p.c[1], a) ||
	    !num_of(p.c[0], b) || *a == 0)
		return 0;
	return 1;
}

/* sign changes on a grid, bisected */
static void
numeric_scan(nid g, int slot, double lo, double hi)
{
	enum { M = 4000 };
	double h = (hi - lo) / M, xp = lo, gp = 0;
	int i, ok, okp = 0;

	for (i = 0; i <= M; i++) {
		double x = lo + h * i, v = evalx(g, slot, x, &ok);
		if (ok && v == 0) {
			addsol(c2_fnum(x), x, 1);
			S[nS - 1].general = -1;
		} else if (ok && okp && (v < 0) != (gp < 0) && gp != 0) {
			double l = xp, r = x, gl = gp, m = l, gm;
			int k, o2;
			for (k = 0; k < 100; k++) {
				m = 0.5 * (l + r);
				gm = evalx(g, slot, m, &o2);
				if (!o2)
					break;
				if ((gm < 0) == (gl < 0))
					l = m, gl = gm;
				else
					r = m;
			}
			/* a sign change through a pole is not a root */
			gm = evalx(g, slot, m, &o2);
			if (o2 && fabs(gm) < 1e-6 * (1 + fabs(gp) + fabs(v))) {
				char t[64];
				snprintf(t, sizeof t, "%.15g", m);
				addsol(t, m, 1);
				S[nS - 1].general = -1;
			}
		}
		if (ok) {
			gp = v;
			xp = x;
		}
		okp = ok;
	}
}

static void
solve1(nid g, nid x, int slot, const char *xn, const char *eqt)
{
	struct rf r;
	struct mp dall;
	size_t mk = ar_mark();
	uint32_t vs;
	int xv, i;

	atoms_reset();
	xv = atom_of(x);
	if (!rf_of(g, &r, &dall)) {
		ar_release(mk);
		goto numeric;
	}
	vs = mp_vars(r.n) | mp_vars(r.d) | mp_vars(dall);
	if (vs == (1u << xv)) {
		/* rational in x */
		struct up n, dd;
		if (!up_of_mp(r.n, xv, &n) || !up_of_mp(dall, xv, &dd)) {
			ar_release(mk);
			goto numeric;
		}
		if (n.d < 0) {
			printf("  every %s is a solution (where the equation is "
			    "defined)\n", xn);
			ar_release(mk);
			return;
		}
		if (n.d == 0) {
			printf("  no solution: the equation reduces to %s = 0\n",
			    fq(n.c[0]));
			ar_release(mk);
			return;
		}
		roots_up(&n);
		/* the poles: excluded */
		if (dd.d >= 1) {
			int j;
			for (j = 0; j < nS; j++) {
				double dv = up_eval(&dd, S[j].v);
				if (fabs(dv) <= 1e-9 * fmax(1, fabs(S[j].v))) {
					snprintf(excl + strlen(excl), sizeof excl - strlen(excl),
					    "%s%s = %s", excl[0] ? ", " : "", xn, S[j].text);
					S[j].mult = 0;
				}
			}
		}
		ar_release(mk);
		return;
	}
	{
		int any = 0;
		for (i = 0; i < nat; i++)
			if ((vs & (1u << i)) && has_slot(at_node[i], slot))
				any = 1;
		if (!any) {
			ar_release(mk);
			printf("  %s does not occur in the equation\n", xn);
			return;
		}
	}
	/* constants that are not rational (log 3): numeric coefficients */
	{
		int allc = 1;
		double av[AV] = {0};
		for (i = 0; i < nat; i++)
			if (i != xv && (vs & (1u << i))) {
				if (!num_of(at_node[i], &av[i]))
					allc = 0;
			}
		if (allc && mp_deg(r.n, xv) >= 1 && mp_deg(r.n, xv) <= 14) {
			double c[16];
			struct root rr[16];
			int dg = mp_deg(r.n, xv), nr, k;
			for (k = 0; k <= dg; k++)
				c[k] = mp_eval(mp_coef(r.n, xv, k), av);
			ar_release(mk);
			nr = poly_roots(c, dg, rr);
			for (k = 0; k < nr; k++) {
				char t[64];
				int ok;
				if (rr[k].im != 0) {
					complex_n += 2 * rr[k].mult;
					continue;
				}
				eval_const_at(g, slot, rr[k].re, &ok);
				if (!ok) {
					snprintf(excl + strlen(excl), sizeof excl - strlen(excl),
					    "%s%s = %.10g", excl[0] ? ", " : "", xn, rr[k].re);
					continue;
				}
				snprintf(t, sizeof t, "%.15g", rr[k].re);
				addsol(t, rr[k].re, rr[k].mult);
			}
			return;
		}
	}
	/* symbolic coefficients, polynomial of degree 1 or 2 in x */
	if (!((mp_vars(r.d) | mp_vars(dall)) & (1u << xv)) &&
	    mp_deg(r.n, xv) <= 2) {
		struct mp A = mp_coef(r.n, xv, 2), B = mp_coef(r.n, xv, 1), C =
		    mp_coef(r.n, xv, 0);
		int dg = mp_deg(r.n, xv), ok = 1;
		for (i = 0; i < nat; i++)
			if (i != xv && has_slot(at_node[i], slot))
				ok = 0;
		if (ok) {
			nid sols[2], mapx[NSLOT];
			int ns = 0, s;
			if (dg == 1) {
				sols[ns++] = rf_node(rf_norm((struct rf){mp_neg(C), B}), 0);
				printf("  %s = %s\n", xn, node_str(sols[0]));
				if (!mp_isconst(B, NULL))
					printf("  where %s != 0\n", mp_str(B));
			} else {
				struct mp D = mp_sub(mp_mul(B, B), mp_scale(mp_mul(A, C),
				    qi(4)));
				nid Dn = mp_node(D), sq = mk_un(O_SQRT, Dn);
				nid nb = mp_node(mp_neg(B)), den = mp_node(mp_scale(A, qi(2)));
				for (s = 0; s < 2; s++)
					sols[ns++] = DIV(s ? SUB(nb, sq) : ADD(nb, sq), den);
				for (s = 0; s < 2; s++)
					printf("  %s = (%s %s %s)/%s%s%s\n", xn, node_str(nb),
					    s ? "-" : "+", node_str(sq), N.op[den] == O_MUL ? "(" :
					    "", node_str(den), N.op[den] == O_MUL ? ")" : "");
				if (!mp_isconst(A, NULL))
					printf("  where %s != 0 (else the equation is linear)\n",
					    mp_str(A));
				printf("  real when the discriminant %s >= 0\n", mp_str(D));
			}
			ar_release(mk);
			for (s = 0; s < ns; s++) {
				double w;
				int n;
				memset(mapx, 0, sizeof mapx);
				mapx[slot] = sols[s];
				n = alg_check(subst(g, mapx, NSLOT), KR(0), &w, NULL);
				printf("  check: %s = %s substituted, the equation holds at %d "
				    "sample points of the other variables (residual %.2g)\n",
				    xn, ns == 1 ? "it" : s ? "the second" : "the first", n, w);
			}
			symbolic_done = 1;
			return;
		}
	}
	/* a polynomial in one function of x: invert */
	{
		int fv = -1, cnt = 0;
		double pbv = 0;
		for (i = 0; i < nat; i++)
			if ((vs & (1u << i)) && has_slot(at_node[i], slot)) {
				fv = i;
				cnt++;
			}
		if (cnt == 1 && fv != xv && !(vs & ~(1u << fv)) &&
		    (opinfo[N.op[at_node[fv]]].arity == 1 ||
		    (N.op[at_node[fv]] == O_POW && num_of(N.a[at_node[fv]], &pbv) &&
		    pbv > 0))) {
			nid A = at_node[fv], u = N.op[A] == O_POW ? N.b[A] : N.a[A];
			int op = N.op[A];
			double pb = op == O_POW ? pbv : 0;
			double a, b;
			struct up n;
			struct sol vals[MAXSOL];
			int nv, j, gen;
			if (!up_of_mp(r.n, fv, &n)) {
				ar_release(mk);
				goto numeric;
			}
			ar_release(mk);
			nS = 0;
			roots_up(&n);
			nv = nS;
			memcpy(vals, S, sizeof vals);
			nS = 0;
			if (lin_coef(u, slot, &a, &b)) {
				for (j = 0; j < nv; j++) {
					gen = 1;
					invert(op, vals[j].v, a, b, pb, gen);
				}
				if (nS == 0) {
					printf("  no real solution: %s never takes the value%s "
					    "%s\n", node_str(A), nv > 1 ? "s" : "", nv ?
					    vals[0].text : "");
					symbolic_done = 1;
				}
				return;
			}
			/* u nonlinear: solve u = value for each value (no k) */
			if (op != O_SIN && op != O_COS && op != O_TAN) {
				struct sol keep[MAXSOL];
				int nk = 0, t;
				for (j = 0; j < nv; j++) {
					nS = 0;
					if (invert(op, vals[j].v, 1, 0, pb, 0))
						for (t = 0; t < nS && nk < MAXSOL; t++)
							keep[nk++] = S[t];
				}
				nS = 0;
				for (j = 0; j < nk; j++) {
					nid gg = SUB(u, KR(keep[j].v));
					solve1(gg, x, slot, xn, eqt);
				}
				return;
			}
			nS = 0;
			goto numeric;
		}
	}
	/* one square root of a polynomial: isolate, square, drop what squaring added */
	{
		int sv = -1, cnt = 0;
		for (i = 0; i < nat; i++)
			if ((vs & (1u << i)) && i != xv && has_slot(at_node[i], slot)) {
				sv = i;
				cnt++;
			}
		if (cnt == 1 && N.op[at_node[sv]] == O_SQRT &&
		    !((mp_vars(r.d) | mp_vars(dall)) & (1u << sv))) {
			struct mp qm, P = mp_zero(), A, B, E;
			struct up e;
			struct rf qr;
			atom_of(x);
			if (rf_of(N.a[at_node[sv]], &qr, NULL) && mp_isconst(qr.d, NULL)) {
				aq dq;
				int t;
				mp_isconst(qr.d, &dq);
				qm = mp_scale(qr.n, qdiv(qi(1), dq));
				for (t = 0; t < r.n.n; t++) {
					struct mp mo;
					int ex = r.n.t[t].e[sv];
					mo.n = 1;
					mo.t = &r.n.t[t];
					mo = mp_copy(mo);
					mo.t[0].e[sv] = (uint8_t)(ex & 1);
					P = mp_add(P, mp_mul(mo, mp_pow(qm, ex / 2)));
				}
				A = mp_coef(P, sv, 0);
				B = mp_coef(P, sv, 1);
				E = mp_sub(mp_mul(A, A), mp_mul(mp_mul(B, B), qm));
				if (!alg_ovf && up_of_mp(E, xv, &e) && e.d >= 1) {
					int j;
					ar_release(mk);
					printf("  isolating %s and squaring: %s = 0\n",
					    node_str(at_node[sv]), up_str(&e, x));
					roots_up(&e);
					for (j = 0; j < nS; j++) {
						int ok;
						double v = eval_const_at(g, slot, S[j].v, &ok);
						if (!ok || fabs(v) > 1e-9 * (1 + fabs(S[j].v))) {
							snprintf(excl + strlen(excl), sizeof excl -
							    strlen(excl), "%s%s = %s", excl[0] ? ", " : "",
							    xn, S[j].text);
							S[j].mult = 0;
						}
					}
					if (excl[0]) {
						printf("  extraneous (from squaring, the original "
						    "equation fails): %s\n", excl);
						excl[0] = 0;
					}
					return;
				}
			}
		}
	}
	ar_release(mk);
numeric:
	printf("  note: no exact method applies; sign changes in [-100, 100], "
	    "refined by bisection (numeric: solutions may be missed)\n");
	numeric_scan(g, slot, -100, 100);
}

static void
print_sols(nid g, int slot, const char *xn)
{
	int i, any = 0, k;

	for (i = 0; i < nS; i++) {
		if (S[i].mult == 0)
			continue;
		any = 1;
		printf("  %s = %s", xn, S[i].text);
		if (S[i].general == 1)
			printf("   (k any integer)");
		if (S[i].mult == 2)
			printf("   (a double root)");
		else if (S[i].mult > 2)
			printf("   (a root of multiplicity %d)", S[i].mult);
		if (S[i].general == -1)
			printf("   (numeric)");
		else if (S[i].general == 0 && strpbrk(S[i].text, "sqrtcoh(") &&
		    strcmp(c2_fnum(S[i].v), S[i].text))
			printf("   (= %.10g)", S[i].v);
		printf("\n");
	}
	if (excl[0])
		printf("  excluded: %s (a denominator vanishes there)\n", excl);
	if (!any)
		printf("  no real solution\n");
	if (complex_n)
		printf("  (and %d complex solution%s)\n", complex_n, complex_n > 1 ?
		    "s" : "");
	if (!any)
		return;
	/* check by substitution */
	{
		double worst = 0;
		int n = 0, ok;
		for (i = 0; i < nS; i++) {
			if (S[i].mult == 0)
				continue;
			for (k = S[i].general == 1 ? -2 : 0; k <= (S[i].general == 1 ? 2 :
			    0); k++) {
				double x = S[i].v + k * S[i].per, v = eval_const_at(g, slot, x,
				    &ok);
				if (!ok) {
					worst = INFINITY;
					continue;
				}
				n++;
				if (fabs(v) > worst)
					worst = fabs(v);
			}
		}
		printf("  check: %s substituted (%d value%s%s): residual %.2g%s\n",
		    "each solution", n, n == 1 ? "" : "s", nS && S[0].general == 1 ?
		    ", k = -2..2" : "", worst, worst <= 1e-8 ? "" :
		    "  (LARGE: check it)");
	}
}

/* ---- linear systems ---- */

static void
solve_linear(const nid *eq, int neq, const nid *vars, int nv)
{
	struct q A[64], B[8], M[8][9];
	int i, j, r, c, rank = 0, piv[8];
	size_t mk = ar_mark();
	int vi[8];

	atoms_reset();
	for (j = 0; j < nv; j++)
		vi[j] = atom_of(vars[j]);
	for (i = 0; i < neq; i++) {
		struct mp p;
		aq k;
		if (!mp_of(eq[i], &p) || mp_tdeg(p) > 1 || (mp_vars(p) >> nv)) {
			ar_release(mk);
			printf("  not a linear system in %d unknowns with numeric "
			    "coefficients\n", nv);
			return;
		}
		for (j = 0; j < nv; j++) {
			struct mp cj = mp_coef(p, vi[j], 1);
			mp_isconst(cj, &k);
			if (!mp_isconst(cj, &k)) {
				ar_release(mk);
				printf("  not linear\n");
				return;
			}
			M[i][j] = k;
		}
		{
			struct mp c0 = p;
			for (j = 0; j < nv; j++)
				c0 = mp_coef(c0, vi[j], 0);
			mp_isconst(c0, &k);
			M[i][nv] = qneg(k);
		}
	}
	(void)A;
	(void)B;
	ar_release(mk);
	alg_ovf = 0;
	/* Gauss-Jordan over Q */
	for (c = 0, r = 0; c < nv && r < neq; c++) {
		int p = -1;
		for (i = r; i < neq; i++)
			if (M[i][c].n) {
				p = i;
				break;
			}
		if (p < 0)
			continue;
		for (j = 0; j <= nv; j++) {
			aq t = M[r][j];
			M[r][j] = M[p][j];
			M[p][j] = t;
		}
		{
			aq iv = qdiv(qi(1), M[r][c]);
			for (j = 0; j <= nv; j++)
				M[r][j] = qmul(M[r][j], iv);
		}
		for (i = 0; i < neq; i++)
			if (i != r && M[i][c].n) {
				aq f = M[i][c];
				for (j = 0; j <= nv; j++)
					M[i][j] = qsub(M[i][j], qmul(f, M[r][j]));
			}
		piv[r++] = c;
	}
	rank = r;
	if (alg_ovf) {
		printf("  the coefficients grow past 64 bits\n");
		return;
	}
	for (i = rank; i < neq; i++)
		if (M[i][nv].n) {
			printf("  no solution: the equations are inconsistent (rank %d, "
			    "the augmented system has rank %d)\n", rank, rank + 1);
			return;
		}
	{
		int isfree[8] = {0};
		double vals[8];
		for (j = 0; j < nv; j++)
			isfree[j] = 1;
		for (i = 0; i < rank; i++)
			isfree[piv[i]] = 0;
		for (i = 0; i < rank; i++) {
			struct sbuf b = {0};
			int any = 0;
			sb_printf(&b, "%s", M[i][nv].n || rank == nv ? fq(M[i][nv]) : "");
			any = M[i][nv].n != 0 || rank == nv;
			for (j = 0; j < nv; j++)
				if (isfree[j] && M[i][j].n) {
					aq c2 = qneg(M[i][j]);
					const char *cs = qeq(c2, qi(1)) ? "" : qeq(c2, qi(-1)) ?
					    "-" : fq(c2);
					if (any)
						sb_printf(&b, " %s %s%s%s", qsgn(c2) < 0 ? "-" : "+",
						    qsgn(c2) < 0 ? (qeq(c2, qi(-1)) ? "" :
						    fq(qneg(c2))) : cs, (qsgn(c2) < 0 ?
						    !qeq(c2, qi(-1)) : !qeq(c2, qi(1))) ? "*" : "",
						    node_str(vars[j]));
					else
						sb_printf(&b, "%s%s%s", cs, *cs && strcmp(cs, "-") ?
						    "*" : "", node_str(vars[j]));
					any = 1;
				}
			if (!any)
				sb_puts(&b, "0");
			printf("  %s = %s\n", node_str(vars[piv[i]]), b.p);
			free(b.p);
		}
		for (j = 0; j < nv; j++)
			if (isfree[j])
				printf("  %s is free (any value)\n", node_str(vars[j]));
		if (rank < nv)
			printf("  (a family of solutions: %d free unknown%s)\n", nv - rank,
			    nv - rank > 1 ? "s" : "");
		/* check with free unknowns at 0 and 1 */
		{
			double worst = 0;
			int t, e;
			for (t = 0; t < 2; t++) {
				nid map[NSLOT];
				memset(map, 0, sizeof map);
				for (j = 0; j < nv; j++)
					vals[j] = isfree[j] ? t + 0.5 * j : 0;
				for (i = 0; i < rank; i++) {
					double v = qd(M[i][nv]);
					for (j = 0; j < nv; j++)
						if (isfree[j])
							v -= qd(M[i][j]) * vals[j];
					vals[piv[i]] = v;
				}
				for (j = 0; j < nv; j++)
					map[N.ki[vars[j]]] = KR(vals[j]);
				for (e = 0; e < neq; e++) {
					double v;
					if (num_of(subst(eq[e], map, NSLOT), &v) &&
					    fabs(v) > worst)
						worst = fabs(v);
				}
			}
			printf("  check: substituted into every equation%s: residual %.2g\n",
			    rank < nv ? " (free unknowns at two values)" : "", worst);
		}
	}
}

/* two polynomial equations in two unknowns: resultant in the second */
static void
solve_poly2(const nid *eq, const nid *vars)
{
	struct mp P, Q, R;
	struct up u;
	size_t mk = ar_mark();
	int xv, yv, i, j, n = 0;
	double worst = 0;
	const char *xn = node_str(vars[0]), *yn = node_str(vars[1]);

	atoms_reset();
	xv = atom_of(vars[0]);
	yv = atom_of(vars[1]);
	if (!mp_of(eq[0], &P) || !mp_of(eq[1], &Q) || ((mp_vars(P) | mp_vars(Q)) &
	    ~3u)) {
		ar_release(mk);
		printf("  not a polynomial system in %s, %s\n", xn, yn);
		return;
	}
	R = mp_resultant(P, Q, yv);
	if (alg_ovf || !up_of_mp(R, xv, &u)) {
		ar_release(mk);
		printf("  the resultant could not be formed (coefficients too "
		    "large)\n");
		return;
	}
	printf("  eliminating %s: the resultant in %s is %s\n", yn, xn,
	    node_str(mp_fnode(R)));
	nS = 0;
	complex_n = 0;
	roots_up(&u);
	{
		struct sol xs[MAXSOL];
		int nx = nS;
		memcpy(xs, S, sizeof xs);
		for (i = 0; i < nx; i++) {
			/* P(x0, y) = 0 and Q(x0, y) = 0: the y roots of P that Q shares */
			double xv0 = xs[i].v, c[UD + 1];
			struct root rr[UD + 1];
			int dg = mp_deg(P, yv), k, nr;
			double xvals[AV] = {0};
			if (dg < 1) {
				dg = mp_deg(Q, yv);
			}
			xvals[xv] = xv0;
			for (k = 0; k <= dg && k <= UD; k++) {
				struct mp ck = mp_coef(mp_deg(P, yv) >= 1 ? P : Q, yv, k);
				c[k] = mp_eval(ck, xvals);
			}
			nr = dg >= 1 ? poly_roots(c, dg, rr) : 0;
			for (j = 0; j < nr; j++) {
				double pv, qv;
				if (rr[j].im != 0)
					continue;
				xvals[yv] = rr[j].re;
				pv = mp_eval(P, xvals);
				qv = mp_eval(Q, xvals);
				if (fabs(pv) + fabs(qv) > 1e-6 * (1 + fabs(xv0) +
				    fabs(rr[j].re)))
					continue;
				printf("  %s = %s, %s = %s\n", xn, xs[i].text, yn,
				    c2_fnum(rr[j].re));
				worst = fmax(worst, fabs(pv) + fabs(qv));
				n++;
			}
		}
	}
	ar_release(mk);
	if (n == 0)
		printf("  no real solution\n");
	else
		printf("  check: %d solution%s substituted into both equations: "
		    "residual %.2g\n", n, n > 1 ? "s" : "", worst);
}

void
alg_solve(const nid *eq, const char *const *eqtext, int neq,
    const nid *vars, int nvars)
{
	int i;

	nS = 0;
	complex_n = 0;
	excl[0] = 0;
	printf("solve");
	for (i = 0; i < neq; i++)
		if (eqtext && eqtext[i])
			printf("%s %s", i ? "," : "", eqtext[i]);
		else
			printf("%s %s = 0", i ? "," : "", node_str(eq[i]));
	if (nvars)
		printf(" for");
	for (i = 0; i < nvars; i++)
		printf("%s %s", i ? "," : "", node_str(vars[i]));
	printf("\n");
	(void)eqtext;
	if (nvars == 0) {
		printf("  nothing to solve for: no variable\n");
		return;
	}
	if (neq == 1 && nvars == 1) {
		int slot = (int)N.ki[vars[0]];
		char xn[64];
		snprintf(xn, sizeof xn, "%s", node_str(vars[0]));
		symbolic_done = 0;
		solve1(eq[0], vars[0], slot, xn, NULL);
		if (!symbolic_done)
			print_sols(eq[0], slot, xn);
		return;
	}
	{
		/* linear? */
		size_t mk = ar_mark();
		int lin = 1;
		atoms_reset();
		for (i = 0; i < nvars; i++)
			atom_of(vars[i]);
		for (i = 0; i < neq; i++) {
			struct mp p;
			if (!mp_of(eq[i], &p) || mp_tdeg(p) > 1)
				lin = 0;
		}
		ar_release(mk);
		if (lin) {
			solve_linear(eq, neq, vars, nvars);
			return;
		}
	}
	if (neq == 2 && nvars == 2) {
		solve_poly2(eq, vars);
		return;
	}
	printf("  only linear systems and two polynomial equations in two "
	    "unknowns are solved\n");
}
