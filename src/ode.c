/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Differential equations and the continuous view of iteration.
 *
 *   y' = f(t, y), y'' = ..., systems x' = ..., y' = ...; y(0) = c
 *   dsolve y      linear systems with constant coefficients symbolically
 *                 (eigenvalues of the companion matrix, repeated and
 *                 complex roots; forcing by polynomials, exponentials and
 *                 sines through undetermined coefficients fitted at
 *                 collocation points), always checked: the residual of the
 *                 equation on a grid, the initial values, and an
 *                 independent numeric solution (Dormand-Prince 5(4))
 *   continuous F  a loop that steps x += h*g(x) is Euler's method for
 *                 x' = g(x): the ODE, its solution and the measured
 *                 difference; a sum is an integral plus Euler-Maclaurin
 *                 corrections, with the measured error
 *   pde name {}   1-D u_t = D u_xx + V u_x + R u + S, u_tt = c^2 u_xx + ...
 *                 by finite differences (Crank-Nicolson, Lax-Wendroff,
 *                 leapfrog), with a grid-refinement error estimate and a
 *                 separation-of-variables series where it applies
 */
#include "calc.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct pdeent;
struct pdeent *pde_find(uint32_t name);

/* ---- the equation store ---- */

#define MAXEQ 16
struct odeq {
	uint32_t var;
	int order;
	char *rhs;
};
struct odeic {
	uint32_t var;
	int order;
	char *t0, *val;
};
static struct odeq eqs[MAXEQ];
static int neqs;
static struct odeic ics[4 * MAXEQ];
static int nics;

static char *
xstrdup(const char *s)
{
	size_t n = strlen(s) + 1;
	char *r = xmalloc(n);

	memcpy(r, s, n);
	return r;
}

void
ode_equation(uint32_t var, int order, const char *rhs, uint32_t tok)
{
	int i;

	(void)tok;
	for (i = 0; i < neqs; i++)
		if (eqs[i].var == var)
			break;
	if (i == neqs) {
		if (neqs == MAXEQ)
			fail("too many differential equations");
		neqs++;
	} else
		free(eqs[i].rhs);
	eqs[i].var = var;
	eqs[i].order = order;
	eqs[i].rhs = xstrdup(rhs);
}

void
ode_initial(uint32_t var, int order, const char *t0, const char *val)
{
	int i;

	for (i = 0; i < nics; i++)
		if (ics[i].var == var && ics[i].order == order)
			break;
	if (i == nics) {
		if (nics == 4 * MAXEQ)
			fail("too many initial values");
		nics++;
	} else {
		free(ics[i].t0);
		free(ics[i].val);
	}
	ics[i].var = var;
	ics[i].order = order;
	ics[i].t0 = xstrdup(t0);
	ics[i].val = xstrdup(val);
}

static struct odeq *
eq_find(uint32_t var)
{
	int i;

	for (i = 0; i < neqs; i++)
		if (eqs[i].var == var)
			return &eqs[i];
	return NULL;
}

/* props.c (identify y): var^(order) = the returned text */
const char *
ode_rhs(uint32_t var, int *order)
{
	struct odeq *q = eq_find(var);

	if (q == NULL)
		return NULL;
	*order = q->order;
	return q->rhs;
}

/* ---- first-order systems z' = F(t, z): slot 0 is t, 1 + i is z_i ---- */

#define MAXS 15
struct osys {
	int n;
	nid F[MAXS];
	double z0[MAXS];
	double t0;
	uint32_t name[MAXS], tname;
};

static nid
tvar(const struct osys *S)
{
	return var_of(0, S->tname, TY_REAL);
}

static nid
zvar(const struct osys *S, int i)
{
	return var_of(1 + i, S->name[i], TY_REAL);
}

static double
const_val(nid e, int *ok)
{
	double v = 0;

	*ok = num_of(e, &v);
	return v;
}

/* evaluate the roots of p at one point */
struct ev {
	struct plan p;
	int64_t *in_i;
	double *in_r;
};

static void
ev_open(struct ev *E, const nid *roots, int n)
{
	plan_build(&E->p, roots, (uint32_t)n, 0);
	E->in_i = xcalloc((size_t)NSLOT * CH, sizeof *E->in_i);
	E->in_r = xcalloc((size_t)NSLOT * CH, sizeof *E->in_r);
}

static void
ev_close(struct ev *E)
{
	plan_free(&E->p);
	free(E->in_i);
	free(E->in_r);
}

static int
ev_run(struct ev *E, double t, const double *z, int n, double *out, int nout)
{
	int i, ok = 1;

	E->in_r[0] = t;
	for (i = 0; i < n; i++)
		E->in_r[(size_t)(1 + i) * CH] = z[i];
	plan_run(&E->p, 1, E->in_i, E->in_r);
	for (i = 0; i < nout; i++) {
		uint32_t l = E->p.root[i];
		if (l == NOTOK) {
			out[i] = 0;
			continue;
		}
		out[i] = TY_FP(E->p.ty[l]) ? COLR(&E->p, l)[0] :
		    (double)COLI(&E->p, l)[0];
		if ((COLF(&E->p, l)[0] & F_UNDEF) || !isfinite(out[i]))
			ok = 0;
	}
	return ok;
}

/* Dormand-Prince 5(4) from t0 to t1 */
static int
rk45(struct ev *E, int n, double t0, double t1, double *z, double rtol,
    int *steps, double *hint)
{
	static const double c2 = 1. / 5, c3 = 3. / 10, c4 = 4. / 5, c5 = 8. / 9;
	static const double a21 = 1. / 5, a31 = 3. / 40, a32 = 9. / 40,
	    a41 = 44. / 45, a42 = -56. / 15, a43 = 32. / 9,
	    a51 = 19372. / 6561, a52 = -25360. / 2187, a53 = 64448. / 6561,
	    a54 = -212. / 729, a61 = 9017. / 3168, a62 = -355. / 33,
	    a63 = 46732. / 5247, a64 = 49. / 176, a65 = -5103. / 18656,
	    b1 = 35. / 384, b3 = 500. / 1113, b4 = 125. / 192,
	    b5 = -2187. / 6784, b6 = 11. / 84, e1 = 71. / 57600,
	    e3 = -71. / 16695, e4 = 71. / 1920, e5 = -17253. / 339200,
	    e6 = 22. / 525, e7 = -1. / 40;
	double k1[MAXS], k2[MAXS], k3[MAXS], k4[MAXS], k5[MAXS], k6[MAXS],
	    k7[MAXS], y[MAXS], yn[MAXS], t = t0, h = *hint, dir = t1 > t0 ? 1 : -1;
	int i, it = 0;

	if (t1 == t0)
		return 1;
	if (h <= 0 || h > fabs(t1 - t0))
		h = fabs(t1 - t0) / 100;
	if (!ev_run(E, t, z, n, k1, n))
		return 0;
	while (dir * (t1 - t) > 0) {
		double err = 0, hh;
		if (++it > 2000000)
			return 0;
		if (h > fabs(t1 - t))
			h = fabs(t1 - t);
		hh = dir * h;
#define STAGE(K, c, expr) do { for (i = 0; i < n; i++) y[i] = z[i] + hh * (expr); \
	if (!ev_run(E, t + (c) * hh, y, n, K, n)) return 0; } while (0)
		STAGE(k2, c2, a21 * k1[i]);
		STAGE(k3, c3, a31 * k1[i] + a32 * k2[i]);
		STAGE(k4, c4, a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
		STAGE(k5, c5, a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
		STAGE(k6, 1, a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] +
		    a65 * k5[i]);
#undef STAGE
		for (i = 0; i < n; i++)
			yn[i] = z[i] + hh * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] +
			    b5 * k5[i] + b6 * k6[i]);
		if (!ev_run(E, t + hh, yn, n, k7, n))
			return 0;
		for (i = 0; i < n; i++) {
			double ei = hh * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] +
			    e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
			double sc = rtol * 1e-2 + rtol * fmax(fabs(z[i]), fabs(yn[i]));
			err = fmax(err, fabs(ei) / sc);
		}
		if (err <= 1) {
			t += hh;
			memcpy(z, yn, (size_t)n * sizeof *z);
			memcpy(k1, k7, (size_t)n * sizeof *k1);
			(*steps)++;
		}
		h *= err == 0 ? 5 : fmin(5, fmax(0.2, 0.9 * pow(err, -0.2)));
		if (h < 1e-14 * fmax(1, fabs(t)))
			return 0;
	}
	*hint = h;
	return 1;
}

/* ---- the symbolic path: z' = A z + g(t) ---- */

struct mode {
	double a, b;
	int deg;
};

static int
add_mode(struct mode *m, int *nm, int max, double a, double b, int deg)
{
	int k;

	for (k = 0; k < *nm; k++)
		if (m[k].a == a && m[k].b == b) {
			if (deg > m[k].deg)
				m[k].deg = deg;
			return 1;
		}
	if (*nm == max)
		return 0;
	m[(*nm)++] = (struct mode){a, b, deg};
	return 1;
}

/* exp, sin or cos of a linear argument: its (a, b) */
static int
mode_factor(nid X, double *a, double *b)
{
	struct npoly p;
	double v;

	if (N.op[X] != O_EXP && N.op[X] != O_SIN && N.op[X] != O_COS)
		return 0;
	if (!npoly_of(N.a[X], 0, &p, 1) || p.deg != 1 || !num_of(p.c[1], &v))
		return 0;
	*a = N.op[X] == O_EXP ? v : 0;
	*b = N.op[X] == O_EXP ? 0 : fabs(v);
	return 1;
}

/* the forcing g(t) as a sum of t^k e^(a t) cos/sin(b t) terms */
static int
modes_of(nid g, struct mode *m, int *nm, int max)
{
	struct npoly p;
	double a, b;

	if (!has_slot(g, 0))
		return add_mode(m, nm, max, 0, 0, 0);
	if (npoly_of(g, 0, &p, 8))
		return add_mode(m, nm, max, 0, 0, p.deg);
	if (mode_factor(g, &a, &b))
		return add_mode(m, nm, max, a, b, 0);
	switch (N.op[g]) {
	case O_ADD:
	case O_SUB:
		return modes_of(N.a[g], m, nm, max) && modes_of(N.b[g], m, nm, max);
	case O_NEG:
	case O_CAST:
		return modes_of(N.a[g], m, nm, max);
	case O_DIV:
		if (has_slot(N.b[g], 0))
			return 0;
		return modes_of(N.a[g], m, nm, max);
	case O_MUL:
		if (npoly_of(N.a[g], 0, &p, 8) && mode_factor(N.b[g], &a, &b))
			return add_mode(m, nm, max, a, b, p.deg);
		if (npoly_of(N.b[g], 0, &p, 8) && mode_factor(N.a[g], &a, &b))
			return add_mode(m, nm, max, a, b, p.deg);
		return 0;
	}
	return 0;
}

/* t^j e^(a t) cos(b t) (kind 0) or sin(b t) (kind 1) */
struct tfun {
	double a, b;
	int j, kind;
};

static void
tf_eval(const struct tfun *f, double t, double *v, double *d)
{
	double e = exp(f->a * t), tj = f->j ? pow(t, f->j) : 1;
	double tj1 = f->j ? f->j * (f->j > 1 ? pow(t, f->j - 1) : 1) : 0;
	double tr = f->kind ? sin(f->b * t) : cos(f->b * t);
	double trd = f->kind ? f->b * cos(f->b * t) : -f->b * sin(f->b * t);

	*v = tj * e * tr;
	*d = tj1 * e * tr + f->a * *v + tj * e * trd;
}

/* the k-th derivative at 0: Re or Im of C(k, j) j! lambda^(k - j) */
static double
tf_deriv0(const struct tfun *f, int k)
{
	double complex lam = f->a + I * f->b, r;
	double c = 1;
	int i;

	if (k < f->j)
		return 0;
	for (i = 0; i < f->j; i++)
		c *= (double)(k - i);
	r = c * cpow(lam, k - f->j);
	if (k == f->j)
		r = c;
	return f->kind ? cimag(r) : creal(r);
}

static nid
tf_node(const struct tfun *f, nid tau)
{
	nid r = 0, t;

	if (f->j)
		r = f->j == 1 ? tau : POW(tau, KR(f->j));
	if (f->a != 0) {
		t = mk_un(O_EXP, MUL(KR(f->a), tau));
		r = r ? MUL(r, t) : t;
	}
	if (f->b != 0) {
		t = mk_un(f->kind ? O_SIN : O_COS, MUL(KR(f->b), tau));
		r = r ? MUL(r, t) : t;
	}
	return r ? r : KR(1);
}

static nid
coef_node(double c)
{
	struct q r;

	if (q_from_double(c, &r) && r.d <= 1000)
		return mk_cast(TY_REAL, q_node(r));
	return KR(c);
}

static void
print_eig(struct sbuf *b, const struct root *r, int nr)
{
	int j;

	for (j = 0; j < nr; j++) {
		sb_printf(b, "%s%.10g", j ? ", " : "", r[j].re + 0.0);
		if (r[j].im != 0)
			sb_printf(b, " ± %.10gi", r[j].im);
		if (r[j].mult > 1)
			sb_printf(b, " (x%d)", r[j].mult);
	}
}

/*
 * The symbolic solution of a linear system with constant coefficients.
 * Z gets one node per state; returns 1 when the result satisfies the
 * equation and the initial values on [t0, t1] (residual in *res).
 */
static int
osys_symbolic(const struct osys *S, double t1, nid *Z, double *res,
    struct sbuf *info)
{
	double A[MAXS * MAXS], cp[MAXS + 1], w[MAXS], L;
	int n = S->n, i, j, k, l, nr, nb = 0, nt = 0, slot[MAXS];
	struct root rt[MAXS];
	struct tfun hb[MAXS], tr[32];
	nid g[MAXS], tau;
	double *cz = NULL;
	int gzero = 1;

	for (i = 0; i < n; i++)
		slot[i] = 1 + i;
	for (i = 0; i < n; i++) {
		double row[MAXST];
		if (n > MAXST)
			return 0;
		if (!linear_form(S->F[i], slot, n, row, &g[i]))
			return 0;
		for (l = 0; l < n; l++)
			A[i * n + l] = row[l];
		g[i] = tidy(g[i]);
		if (!const_is(g[i], 0))
			gzero = 0;
	}
	charpoly(A, n, cp);
	nr = poly_roots(cp, n, rt);
	for (j = 0; j < nr; j++) {
		int p;
		for (p = 0; p < rt[j].mult; p++) {
			if (nb + (rt[j].im != 0 ? 2 : 1) > MAXS)
				return 0;
			hb[nb++] = (struct tfun){rt[j].re, rt[j].im, p, 0};
			if (rt[j].im != 0)
				hb[nb++] = (struct tfun){rt[j].re, rt[j].im, p, 1};
		}
	}
	if (nb != n)
		return 0;
	sb_puts(info, "eigenvalues ");
	print_eig(info, rt, nr);
	L = t1 > S->t0 ? t1 - S->t0 : 1;
	/* a particular solution in the span of the forcing's modes */
	if (!gzero) {
		struct mode md[8];
		int nm = 0, q, K, ok = 1;
		double *M, *rhs, gv[MAXS], gscale = 0, worst = 0;
		struct ev E;
		for (i = 0; i < n; i++)
			if (!modes_of(g[i], md, &nm, 8))
				return 0;
		for (q = 0; q < nm; q++) {
			int mult = 0, jj;
			for (j = 0; j < nr; j++)
				if (fabs(rt[j].re - md[q].a) < 1e-9 &&
				    fabs(rt[j].im - md[q].b) < 1e-9)
					mult = rt[j].mult;
			for (jj = 0; jj <= md[q].deg + mult; jj++) {
				if (nt + 2 > 32)
					return 0;
				tr[nt++] = (struct tfun){md[q].a, md[q].b, jj, 0};
				if (md[q].b != 0)
					tr[nt++] = (struct tfun){md[q].a, md[q].b, jj, 1};
			}
		}
		K = 2 * nt + 6;
		M = xcalloc((size_t)K * n * n * nt, sizeof *M);
		rhs = xcalloc((size_t)K * n, sizeof *rhs);
		cz = xcalloc((size_t)n * nt, sizeof *cz);
		ev_open(&E, g, n);
		for (k = 0; k < K; k++) {
			double t = (k + 0.5) / K * L, fv[32], fd[32];
			ev_run(&E, S->t0 + t, w, 0, gv, n);
			for (q = 0; q < nt; q++)
				tf_eval(&tr[q], t, &fv[q], &fd[q]);
			for (i = 0; i < n; i++) {
				double *row = &M[((size_t)k * n + i) * n * nt];
				for (q = 0; q < nt; q++) {
					row[i * nt + q] += fd[q];
					for (l = 0; l < n; l++)
						row[l * nt + q] -= A[i * n + l] * fv[q];
				}
				rhs[k * n + i] = gv[i];
				gscale = fmax(gscale, fabs(gv[i]));
			}
		}
		if (!solve_lsq(M, rhs, K * n, n * nt, cz))
			ok = 0;
		/* the residual between the collocation points */
		for (k = 0; ok && k < K; k++) {
			double t = (k + 0.13) / K * L, fv, fd;
			ev_run(&E, S->t0 + t, w, 0, gv, n);
			for (i = 0; i < n; i++) {
				double r = -gv[i];
				for (q = 0; q < nt; q++) {
					tf_eval(&tr[q], t, &fv, &fd);
					r += cz[i * nt + q] * fd;
					for (l = 0; l < n; l++)
						r -= A[i * n + l] * cz[l * nt + q] * fv;
				}
				worst = fmax(worst, fabs(r));
			}
		}
		ev_close(&E);
		free(M);
		free(rhs);
		if (!ok || worst > 1e-8 * (1 + gscale)) {
			free(cz);
			return 0;
		}
		sb_puts(info, "; forcing by undetermined coefficients");
	}
	/* the homogeneous part from the initial values */
	for (i = 0; i < n; i++) {
		w[i] = S->z0[i];
		if (cz) {
			int q;
			for (q = 0; q < nt; q++) {
				double fv, fd;
				tf_eval(&tr[q], 0, &fv, &fd);
				w[i] -= cz[i * nt + q] * fv;
			}
		}
	}
	tau = S->t0 == 0 ? tvar(S) : SUB(tvar(S), KR(S->t0));
	for (i = 0; i < n; i++) {
		double W[MAXS * MAXS], b[MAXS], v[MAXS], vn[MAXS], scale = 0;
		nid z = 0;
		memcpy(v, w, sizeof v);
		for (k = 0; k < n; k++) {
			for (j = 0; j < n; j++)
				W[k * n + j] = tf_deriv0(&hb[j], k);
			b[k] = v[i];
			for (l = 0; l < n; l++) {
				double s = 0;
				for (j = 0; j < n; j++)
					s += A[l * n + j] * v[j];
				vn[l] = s;
			}
			memcpy(v, vn, sizeof v);
		}
		if (!solve_lin(W, b, n)) {
			free(cz);
			return 0;
		}
		for (j = 0; j < n; j++)
			scale = fmax(scale, fabs(b[j]));
		for (j = 0; j < n; j++) {
			nid term;
			if (fabs(b[j]) <= 1e-13 * scale || b[j] == 0)
				continue;
			term = tf_node(&hb[j], tau);
			term = const_is(coef_node(b[j]), 1) ? term :
			    MUL(coef_node(b[j]), term);
			z = z ? ADD(z, term) : term;
		}
		if (cz) {
			int q;
			for (q = 0; q < nt; q++) {
				nid term;
				double c = cz[i * nt + q];
				if (fabs(c) < 1e-12)
					continue;
				term = tf_node(&tr[q], tau);
				term = const_is(coef_node(c), 1) ? term :
				    MUL(coef_node(c), term);
				z = z ? ADD(z, term) : term;
			}
		}
		Z[i] = tidy(z ? z : KR(0));
	}
	free(cz);
	/* check: Z' = F(t, Z) on a grid, and Z(t0) = z0 */
	{
		nid map[NSLOT], R[MAXS], roots[2 * MAXS];
		struct ev E;
		double out[2 * MAXS], fs = 0, rmax = 0;
		jmp_buf saved;
		memset(map, 0, sizeof map);
		for (i = 0; i < n; i++)
			map[1 + i] = Z[i];
		memcpy(saved, fail_jmp, sizeof saved);
		if (setjmp(fail_jmp)) {
			memcpy(fail_jmp, saved, sizeof saved);
			notes_drop();
			return 0;
		}
		for (i = 0; i < n; i++) {
			R[i] = SUB(diff_node(Z[i], 0), subst(S->F[i], map, NSLOT));
			roots[i] = R[i];
			roots[n + i] = subst(S->F[i], map, NSLOT);
		}
		memcpy(fail_jmp, saved, sizeof saved);
		ev_open(&E, roots, 2 * n);
		for (k = 0; k <= 200; k++) {
			double t = S->t0 + L * k / 200.0;
			if (!ev_run(&E, t, w, 0, out, 2 * n))
				continue;
			for (i = 0; i < n; i++) {
				rmax = fmax(rmax, fabs(out[i]));
				fs = fmax(fs, fabs(out[n + i]));
			}
		}
		ev_close(&E);
		for (i = 0; i < n; i++) {
			int ok;
			double z0 = eval_const_at(Z[i], 0, S->t0, &ok);
			if (!ok || fabs(z0 - S->z0[i]) > 1e-9 * (1 + fabs(S->z0[i])))
				return 0;
		}
		*res = rmax / (1 + fs);
		return *res <= 1e-8;
	}
}

/* the numeric solution at times ts[0..nt-1] (increasing from t0) */
static int
osys_numeric(const struct osys *S, const double *ts, int nt, double *out,
    double rtol, int *steps)
{
	struct ev E;
	double z[MAXS], t = S->t0, h = 0;
	int i, k, ok = 1;

	ev_open(&E, S->F, S->n);
	memcpy(z, S->z0, sizeof z);
	*steps = 0;
	for (k = 0; k < nt; k++) {
		if (ok && !rk45(&E, S->n, t, ts[k], z, rtol, steps, &h))
			ok = 0;
		t = ts[k];
		for (i = 0; i < S->n; i++)
			out[k * S->n + i] = ok ? z[i] : NAN;
	}
	ev_close(&E);
	return ok;
}

/* ---- dsolve ---- */

struct stname {
	uint32_t base;
	int prime;
};

static uint32_t
primed(uint32_t base, int k)
{
	char buf[80];
	const char *s = sym_str(base);
	int i;

	snprintf(buf, sizeof buf - 8, "%s", s);
	for (i = 0; i < k; i++)
		strcat(buf, "'");
	return sym_cstr(buf);
}

static void
split_prime(uint32_t name, uint32_t *base, int *k)
{
	const char *s = sym_str(name);
	size_t n = strlen(s);

	*k = 0;
	while (n > 0 && s[n - 1] == '\'')
		n--, (*k)++;
	*base = sym_intern(s, n);
}

static double
parse_const(const char *src)
{
	struct func fn;
	nid v;
	double d;

	memset(&fn, 0, sizeof fn);
	fn.name = sym_cstr("_");
	v = calc_parse_text(src, &fn, TY_REAL);
	if (fn.np || !num_of(v, &d))
		fail("'%s' is not a constant", src);
	return d;
}

static void
print_table(const struct osys *S, int which, const double *ts, int nt,
    const double *num, nid sym)
{
	int k, ok;

	printf("      %-12s %s(%s)\n", sym_str(S->tname), sym_str(S->name[which]),
	    sym_str(S->tname));
	rep_str("function", sym_str(S->name[which]));
	rep_list("table");
	for (k = 0; k < nt; k++) {
		double v = num[k * S->n + which];
		if (sym)
			v = eval_const_at(sym, 0, ts[k], &ok);
		printf("      %-12.6g %.10g\n", ts[k], v);
		rep_list(NULL);
		rep_num(NULL, ts[k]);
		rep_num(NULL, v);
		rep_end();
	}
	rep_end();
}

static void
solve_and_report(const struct osys *S, int which, double t1, int steps)
{
	nid Z[MAXS];
	double res = 0, ts[257], *num, maxd = 0;
	struct sbuf info = {0};
	int k, i, nsteps, ok, sym;

	if (steps <= 0)
		steps = 10;
	if (steps > 256)
		steps = 256;
	sym = osys_symbolic(S, t1, Z, &res, &info);
	rep_begin("dsolve");
	rep_str("var", sym_str(S->tname));
	rep_num("from", S->t0);
	rep_num("to", t1);
	rep_bool("symbolic", sym);
	if (sym) {
		printf("  linear with constant coefficients: %s\n", sb_str(&info));
		rep_str("method", sb_str(&info));
		rep_obj("solution");
		for (i = 0; i < S->n; i++)
			rep_expr(sym_str(S->name[i]), Z[i]);
		rep_end();
		rep_check("residual", 201, res, 1, "satisfies the equations and "
		    "the initial values");
		for (i = 0; i < S->n; i++)
			if (i == which || S->n <= 3)
				printf("  %s(%s) = %s\n", sym_str(S->name[i]),
				    sym_str(S->tname), node_str(Z[i]));
		printf("  check: satisfies the equations at 201 points of [%g, %g] "
		    "(max relative residual %.2g) and the initial values\n", S->t0,
		    t1, res);
	} else if (info.n)
		printf("  linear with constant coefficients (%s), but no closed form "
		    "was found for the forcing: numeric only\n", sb_str(&info));
	else
		printf("  not linear with constant coefficients: numeric only\n");
	free(info.p);
	for (k = 0; k <= steps; k++)
		ts[k] = S->t0 + (t1 - S->t0) * k / steps;
	num = xcalloc((size_t)(steps + 1) * S->n, sizeof *num);
	ok = osys_numeric(S, ts, steps + 1, num, 1e-10, &nsteps);
	if (!ok) {
		printf("  numeric: the solution stops before %g (blow-up, or the "
		    "right side is undefined)\n", t1);
		for (k = 0; k <= steps && !isnan(num[k * S->n]); k++)
			;
		steps = k - 1;
	} else {
		double *loose = xcalloc((size_t)(steps + 1) * S->n, sizeof *loose);
		int ls;
		double ed = 0;
		osys_numeric(S, ts, steps + 1, loose, 1e-7, &ls);
		for (k = 0; k <= steps; k++)
			ed = fmax(ed, fabs(loose[k * S->n + which] - num[k * S->n +
			    which]));
		if (sym) {
			for (k = 0; k <= steps; k++) {
				double v = eval_const_at(Z[which], 0, ts[k], &i);
				maxd = fmax(maxd, fabs(v - num[k * S->n + which]));
			}
			printf("  numeric (Dormand-Prince 5(4), adaptive, %d steps) "
			    "agrees with it to %.2g\n", nsteps, maxd);
			rep_check("numeric_solution", steps + 1, maxd, maxd <= 1e-6,
			    "Dormand-Prince 5(4) against the symbolic solution");
		} else
			printf("  numeric (Dormand-Prince 5(4), adaptive, %d steps, "
			    "tolerance 1e-10); a 1000x looser solve differs by %.2g\n",
			    nsteps, ed);
		if (!sym)
			rep_num("numeric_error_estimate", ed);
		free(loose);
	}
	if (steps >= 0)
		print_table(S, which, ts, steps + 1, num, sym ? Z[which] : 0);
	free(num);
}

void
cmd_dsolve(uint32_t var, nid an, nid bn, int steps)
{
	struct osys S;
	uint32_t vars[MAXS];
	int nv = 0, i, j, k, which = -1, tset = 0;
	int base_st[MAXS];
	struct func fns[MAXS];
	nid rhs[MAXS];
	double t1;

	if (pde_find(var)) {
		cmd_pde(var, bn, steps);
		return;
	}
	if (!eq_find(var))
		fail("no equation for %s (write %s' = ...)", sym_str(var),
		    sym_str(var));
	memset(&S, 0, sizeof S);
	S.tname = 0;
	vars[nv++] = var;
	/* the variables the equations reach */
	for (i = 0; i < nv; i++) {
		struct odeq *q = eq_find(vars[i]);
		memset(&fns[i], 0, sizeof fns[i]);
		fns[i].name = sym_cstr("_");
		rhs[i] = calc_parse_text(q->rhs, &fns[i], TY_REAL);
		for (j = 0; j < fns[i].np; j++) {
			uint32_t b;
			int pk;
			split_prime(fns[i].p[j].name, &b, &pk);
			if (eq_find(b)) {
				for (k = 0; k < nv; k++)
					if (vars[k] == b)
						break;
				if (k == nv) {
					if (nv == MAXS)
						fail("system too large");
					vars[nv++] = b;
				}
			}
		}
	}
	/* states: v, v', ... v^(m-1) per variable */
	for (i = 0; i < nv; i++) {
		struct odeq *q = eq_find(vars[i]);
		base_st[i] = S.n;
		for (k = 0; k < q->order; k++) {
			if (S.n == MAXS)
				fail("system too large (%d states)", MAXS);
			S.name[S.n++] = primed(vars[i], k);
		}
	}
	/* the independent variable */
	for (i = 0; i < nv; i++)
		for (j = 0; j < fns[i].np; j++) {
			uint32_t b;
			int pk;
			split_prime(fns[i].p[j].name, &b, &pk);
			for (k = 0; k < nv; k++)
				if (vars[k] == b)
					break;
			if (k < nv) {
				if (pk >= eq_find(b)->order)
					fail("%s appears in the equations of order %d",
					    sym_str(fns[i].p[j].name), eq_find(b)->order);
				continue;
			}
			if (S.tname && S.tname != fns[i].p[j].name)
				fail("%s and %s are both free: define constants first "
				    "(k = 2)", sym_str(S.tname),
				    sym_str(fns[i].p[j].name));
			S.tname = fns[i].p[j].name;
		}
	if (!S.tname)
		S.tname = sym_cstr("t");
	/* right sides in the common slots */
	for (i = 0; i < nv; i++) {
		struct odeq *q = eq_find(vars[i]);
		nid map[MAXP];
		for (j = 0; j < fns[i].np; j++) {
			uint32_t b;
			int pk;
			split_prime(fns[i].p[j].name, &b, &pk);
			map[j] = tvar(&S);
			for (k = 0; k < nv; k++)
				if (vars[k] == b)
					map[j] = zvar(&S, base_st[k] + pk);
		}
		for (k = 0; k + 1 < q->order; k++)
			S.F[base_st[i] + k] = zvar(&S, base_st[i] + k + 1);
		S.F[base_st[i] + q->order - 1] = mk_cast(TY_REAL,
		    subst(rhs[i], map, fns[i].np));
	}
	/* initial values */
	for (i = 0; i < nv; i++) {
		struct odeq *q = eq_find(vars[i]);
		for (k = 0; k < q->order; k++) {
			int z;
			for (z = 0; z < nics; z++)
				if (ics[z].var == vars[i] && ics[z].order == k)
					break;
			if (z == nics)
				fail("missing initial value %s(t0) = ...",
				    sym_str(primed(vars[i], k)));
			{
				double t0 = parse_const(ics[z].t0);
				if (tset && t0 != S.t0)
					fail("initial values at different times %g and %g",
					    S.t0, t0);
				S.t0 = t0;
				tset = 1;
			}
			S.z0[base_st[i] + k] = parse_const(ics[z].val);
		}
	}
	which = base_st[0];
	if (an) {
		double a;
		num_of(an, &a);
		if (a != S.t0)
			fail("the initial values are at %s = %g; solve from there",
			    sym_str(S.tname), S.t0);
	}
	if (bn)
		num_of(bn, &t1);
	else
		t1 = S.t0 + 10;
	for (i = 0; i < nv; i++) {
		struct odeq *q = eq_find(vars[i]);
		printf("%s = %s", sym_str(primed(vars[i], q->order)), q->rhs);
		for (k = 0; k < q->order; k++)
			printf(", %s(%g) = %.10g", sym_str(primed(vars[i], k)), S.t0,
			    S.z0[base_st[i] + k]);
		printf("\n");
	}
	if (!bn)
		printf("  (on [%g, %g]; 'dsolve %s to T' for another end)\n", S.t0,
		    t1, sym_str(var));
	solve_and_report(&S, which, t1, steps);
}

/* ---- sample points of a function's domain ---- */

/* up to max points spread over the domain, as lanes of in_i/in_r */
static int
pick_points(const struct func *f, int64_t *in_i, double *in_r, int max)
{
	struct diter d;
	int64_t *ai = xcalloc((size_t)MAXP * CH, sizeof *ai);
	double *ar = xcalloc((size_t)MAXP * CH, sizeof *ar);
	int64_t *all_i = xmalloc((size_t)4096 * MAXP * sizeof *all_i);
	double *all_r = xmalloc((size_t)4096 * MAXP * sizeof *all_r);
	uint32_t np, i;
	int n = 0, k, j;

	diter_init(&d, f, 4096, 101, 4096);
	while ((np = diter_next(&d, ai, ar)) != 0)
		for (i = 0; i < np && n < 4096; i++, n++)
			for (j = 0; j < MAXP; j++) {
				all_i[(size_t)n * MAXP + j] = ai[(size_t)j * CH + i];
				all_r[(size_t)n * MAXP + j] = ar[(size_t)j * CH + i];
			}
	if (n > max) {
		for (k = 0; k < max; k++) {
			int src = (int)((double)k * (n - 1) / (max - 1));
			for (j = 0; j < MAXP; j++) {
				in_i[(size_t)j * CH + k] = all_i[(size_t)src * MAXP + j];
				in_r[(size_t)j * CH + k] = all_r[(size_t)src * MAXP + j];
			}
		}
		n = max;
	} else
		for (k = 0; k < n; k++)
			for (j = 0; j < MAXP; j++) {
				in_i[(size_t)j * CH + k] = all_i[(size_t)k * MAXP + j];
				in_r[(size_t)j * CH + k] = all_r[(size_t)k * MAXP + j];
			}
	free(ai);
	free(ar);
	free(all_i);
	free(all_r);
	return n;
}

static double
colval(struct plan *p, uint32_t loc, int lane, int *ok)
{
	*ok = !(COLF(p, loc)[lane] & F_UNDEF);
	return TY_FP(p->ty[loc]) ? COLR(p, loc)[lane] :
	    (double)COLI(p, loc)[lane];
}

/* ---- the continuous view ---- */

static int
guarded_diff(nid e, int slot, nid *r)
{
	jmp_buf saved;
	int ok = 1;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		*r = diff_node(e, (uint32_t)slot);
	else
		ok = 0;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return ok;
}

static int
guarded_subst(nid e, int slot, nid v, nid *r)
{
	jmp_buf saved;
	nid map[NSLOT];
	int ok = 1;

	memset(map, 0, sizeof map);
	map[slot] = v;
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		*r = subst(e, map, NSLOT);
	else
		ok = 0;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return ok;
}

/* a sum (or product, through its logarithm) by Euler-Maclaurin */
static int
cont_sum(const struct func *f, nid n, const struct loopdef *d)
{
	int s = d->islot, isprod = d->orig == LO_PROD, k, np, ok;
	nid X = var_of(s, d->name, TY_REAL), g, F, d1, d3, Lr, Hr, Iv, E[3];
	nid lo = N.a[n], hi = N.b[n], roots[5], gl, gh;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	struct plan p;

	if (!guarded_subst(mk_cast(TY_REAL, d->aux), s, X, &g))
		return 0;
	if (isprod)
		g = mk_un(O_LOG, g);
	F = antideriv(g, s, NULL);
	if (!F || !guarded_diff(g, s, &d1) || !guarded_diff(d1, s, &d3) ||
	    !guarded_diff(d3, s, &d3)) {
		printf("  no elementary antiderivative of %s: no continuous form\n",
		    node_str(g));
		free(in_i);
		free(in_r);
		return 1;
	}
	Lr = mk_cast(TY_REAL, lo);
	Hr = mk_cast(TY_REAL, hi);
	{
		nid a, b;
		guarded_subst(F, s, Hr, &a);
		guarded_subst(F, s, Lr, &b);
		Iv = tidy(SUB(a, b));
		guarded_subst(g, s, Lr, &gl);
		guarded_subst(g, s, Hr, &gh);
		E[0] = Iv;
		E[1] = ADD(Iv, DIV(ADD(gl, gh), KR(2)));
		guarded_subst(d1, s, Hr, &a);
		guarded_subst(d1, s, Lr, &b);
		E[2] = ADD(E[1], DIV(SUB(a, b), KR(12)));
		guarded_subst(d3, s, Hr, &a);
		guarded_subst(d3, s, Lr, &b);
		E[2] = tidy(SUB(E[2], DIV(SUB(a, b), KR(720))));
		E[1] = tidy(E[1]);
	}
	printf("  %s ≈ ∫ %s d%s from %s to %s, plus Euler-Maclaurin corrections\n",
	    isprod ? "log of the product" : "the sum", node_str(g),
	    sym_str(d->name), node_str(lo), node_str(hi));
	printf("  ∫ %s d%s = %s + C\n", node_str(g), sym_str(d->name),
	    node_str(F));
	printf("  integral: %s%s%s\n", isprod ? "exp(" : "", node_str(E[0]),
	    isprod ? ")" : "");
	printf("  + (g(lo) + g(hi))/2: %s%s%s\n", isprod ? "exp(" : "",
	    node_str(E[1]), isprod ? ")" : "");
	printf("  + (g'(hi) - g'(lo))/12 - (g'''(hi) - g'''(lo))/720 (evaluated "
	    "below)\n");
	for (k = 0; k < 3; k++)
		if (isprod)
			E[k] = mk_un(O_EXP, E[k]);
	roots[0] = n;
	roots[1] = E[0];
	roots[2] = E[1];
	roots[3] = E[2];
	roots[4] = f->where;
	plan_build(&p, roots, 5, 0);
	np = pick_points(f, in_i, in_r, 6);
	plan_run(&p, (uint32_t)np, in_i, in_r);
	printf("  measured against the %s (relative error):\n",
	    isprod ? "product" : "sum");
	printf("      %-24s %-14s %-11s %-11s %-11s\n", "at", "exact",
	    "integral", "+ endpoint", "+ g', g'''");
	for (k = 0; k < np; k++) {
		struct sbuf b = {0};
		double v = colval(&p, p.root[0], k, &ok), e;
		int z;
		if (!ok)
			continue;
		fmt_point(&b, f, in_i, in_r, (uint32_t)k);
		printf("      %-24s %-14.8g", sb_str(&b), v);
		for (z = 1; z <= 3; z++) {
			int o2;
			e = colval(&p, p.root[z], k, &o2);
			if (o2 && v != 0)
				printf(" %-11.2g", fabs(e - v) / fabs(v));
			else
				printf(" %-11s", "-");
		}
		printf("\n");
		free(b.p);
	}
	if (np >= 3) {
		int o1, o2, o3, o4;
		double va = colval(&p, p.root[0], np - 2, &o1);
		double vb = colval(&p, p.root[0], np - 1, &o2);
		double ea = colval(&p, p.root[3], np - 2, &o3) - va;
		double eb = colval(&p, p.root[3], np - 1, &o4) - vb;
		if (o1 && o2 && o3 && o4 && va != 0 && vb != 0 &&
		    fabs(eb / vb) > 0.8 * fabs(ea / va) && fabs(eb / vb) > 1e-6)
			printf("  the remaining error does not shrink: the corrections "
			    "at the lower end are not small (it is a constant%s, "
			    "like %s)\n", isprod ? " factor" : "", isprod ?
			    "Stirling's sqrt(2 pi)/e" : "Euler's gamma for 1/i");
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return 1;
}

/* a loop whose states step by h * g: Euler's method for z' = g */
static int
cont_euler(const struct func *f, nid n, const struct loopdef *d)
{
	struct osys S;
	nid hnode = 0, D[MAXST], eta, Z[MAXS], T, map[NSLOT];
	double h = 0, lo, res;
	int j, k, np, ok, sym, es = KSLOT - 1;
	struct sbuf info = {0};
	int64_t *in_i;
	double *in_r;
	struct plan p;

	if (d->kind != LK_FOR || d->nst > MAXS)
		return 0;
	for (j = 0; j < d->nst; j++) {
		nid sv = var_of(d->sslot[j], d->sname[j], d->sty[j]);
		if (!TY_FP(d->sty[j]))
			return 0;
		D[j] = tidy(SUB(d->next[j], sv));
		if (!hnode && N.op[D[j]] == O_MUL && num_of(N.a[D[j]], &h) && h > 0)
			hnode = N.a[D[j]];
		else if (!hnode && N.op[D[j]] == O_NEG && N.op[N.a[D[j]]] == O_MUL &&
		    num_of(N.a[N.a[D[j]]], &h) && h > 0)
			hnode = N.a[N.a[D[j]]];
	}
	if (!hnode || !num_of(N.a[n], &lo))
		return 0;
	num_of(hnode, &h);
	eta = var_of(es, sym_cstr("h"), TY_REAL);
	memset(&S, 0, sizeof S);
	S.n = d->nst;
	S.tname = sym_cstr("t");
	for (j = 0; j < d->nst; j++) {
		nid E, e0, g;
		double v;
		if (!num_of(d->init[j], &S.z0[j])) {
			printf("  the initial value of %s depends on parameters: no "
			    "continuous form\n", sym_str(d->sname[j]));
			return 1;
		}
		S.name[j] = d->sname[j];
		E = xreplace(D[j], hnode, eta);
		if (!guarded_subst(E, es, KR(0), &e0) || !num_of(e0, &v) || v != 0)
			return 0;
		if (!guarded_diff(E, es, &g) || !guarded_subst(g, es, KR(0), &g))
			return 0;
		if (node_slots(g) & ((1ULL << MAXP) - 1)) {
			printf("  the step reads parameters: no continuous form\n");
			return 1;
		}
		D[j] = g;
	}
	/* states -> z, the index i -> lo + t/h */
	memset(map, 0, sizeof map);
	map[d->islot] = ADD(KR(lo), DIV(tvar(&S), KR(h)));
	for (j = 0; j < d->nst; j++)
		map[d->sslot[j]] = zvar(&S, j);
	for (j = 0; j < d->nst; j++)
		S.F[j] = tidy(mk_cast(TY_REAL, subst(D[j], map, NSLOT)));
	printf("  each iteration steps by h = %g times a rate: Euler's method "
	    "for\n", h);
	for (j = 0; j < S.n; j++)
		printf("    %s' = %s,  %s(0) = %.10g\n", sym_str(S.name[j]),
		    node_str(S.F[j]), sym_str(S.name[j]), S.z0[j]);
	printf("  with t = %g * (iterations)\n", h);
	/* where to compare: the domain's points, t = h * trip count */
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	T = ADD(SUB(N.b[n], N.a[n]), KI(1));
	{
		nid roots[3] = {n, T, f->where};
		double ts[8], tmax = 0, *num;
		int order[8], m = 0, steps, which = loop_res(n);
		plan_build(&p, roots, 3, 0);
		np = pick_points(f, in_i, in_r, 6);
		plan_run(&p, (uint32_t)np, in_i, in_r);
		for (k = 0; k < np; k++) {
			double tv = colval(&p, p.root[1], k, &ok) * h;
			if (ok && tv >= 0) {
				ts[m] = tv;
				order[m++] = k;
				tmax = fmax(tmax, tv);
			}
		}
		sym = osys_symbolic(&S, tmax, Z, &res, &info);
		if (sym) {
			printf("  solution (%s):\n", sb_str(&info));
			printf("    %s(t) = %s\n", sym_str(S.name[which]),
			    node_str(Z[which]));
			printf("    checked: residual %.2g on [0, %g]\n", res, tmax);
		} else
			printf("  solved numerically (Dormand-Prince 5(4))\n");
		/* the numeric solution needs increasing times */
		for (j = 0; j < m; j++)
			for (k = j + 1; k < m; k++)
				if (ts[k] < ts[j]) {
					double tt = ts[j];
					int oo = order[j];
					ts[j] = ts[k];
					ts[k] = tt;
					order[j] = order[k];
					order[k] = oo;
				}
		num = xcalloc((size_t)(m + 1) * MAXS, sizeof *num);
		osys_numeric(&S, ts, m, num, 1e-11, &steps);
		printf("      %-24s %-10s %-16s %-16s %s\n", "at", "t", "loop",
		    "ODE", "difference");
		for (j = 0; j < m; j++) {
			struct sbuf b = {0};
			double lv = colval(&p, p.root[0], order[j], &ok), ov;
			int o2 = 1;
			fmt_point(&b, f, in_i, in_r, (uint32_t)order[j]);
			ov = sym ? eval_const_at(Z[which], 0, ts[j], &o2) :
			    num[j * S.n + which];
			if (ok && o2)
				printf("      %-24s %-10g %-16.10g %-16.10g %.3g\n",
				    sb_str(&b), ts[j], lv, ov, fabs(lv - ov));
			free(b.p);
		}
		printf("  Euler's method has global error O(h): the difference "
		    "shrinks about in proportion to h\n");
		free(num);
		plan_free(&p);
	}
	free(info.p);
	free(in_i);
	free(in_r);
	return 1;
}

static void
find_loops(nid e, nid *out, int *n, int max, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || *n >= max)
		return;
	seen[e] = 1;
	if (N.op[e] == O_LOOP) {
		out[(*n)++] = e;
		find_loops(N.a[e], out, n, max, seen);
		find_loops(N.b[e], out, n, max, seen);
		return;
	}
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			find_loops(N.lst[N.a[e] + j], out, n, max, seen);
		return;
	}
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return;
	find_loops(N.a[e], out, n, max, seen);
	find_loops(N.b[e], out, n, max, seen);
	find_loops(N.c[e], out, n, max, seen);
}

void
cmd_continuous(struct func *f)
{
	nid loops[8];
	int nl = 0, i;
	uint8_t *seen = xcalloc(N.n, 1);

	find_loops(f->body, loops, &nl, 8, seen);
	free(seen);
	if (nl == 0) {
		printf("%s has no loops\n", f->anon ? node_str(f->body) :
		    sym_str(f->name));
		return;
	}
	for (i = 0; i < nl; i++) {
		struct loopdef d;
		nid n = loops[i];
		loop_get(loop_ent(n), &d);
		printf("%s\n", node_str(n));
		if ((d.orig == LO_SUM || d.orig == LO_PROD) && cont_sum(f, n, &d))
			continue;
		if (cont_euler(f, n, &d))
			continue;
		printf("  no continuous form recognised (a sum, a product, or "
		    "states stepping by h * rate)\n");
	}
}

/* ---- PDEs ---- */

struct pdeent {
	uint32_t name;
	struct pde pd;
};
static struct pdeent pdes[8];
static int npdes;

struct pdeent *
pde_find(uint32_t name)
{
	int i;

	for (i = 0; i < npdes; i++)
		if (pdes[i].name == name)
			return &pdes[i];
	return NULL;
}

void
pde_define(uint32_t name, struct pde *pd)
{
	struct pdeent *e = pde_find(name);

	if (!e) {
		if (npdes == 8)
			fail("too many pdes");
		e = &pdes[npdes++];
	}
	e->name = name;
	e->pd = *pd;
}

struct pdec {
	double D, V, R, DT;
	nid S;                 /* source: slot 0 x, slot 1 t */
	int hasS;
};

/* values of a node of x (slot 0) and t (slot 1) at many x */
static void
eval_xt(nid e, const double *x, int n, double t, double *out)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int i, k;

	plan_build(&p, &e, 1, 0);
	for (i = 0; i < n; i += CH) {
		int m = n - i < CH ? n - i : CH;
		for (k = 0; k < m; k++) {
			in_r[k] = x[i + k];
			in_r[CH + k] = t;
		}
		plan_run(&p, (uint32_t)m, in_i, in_r);
		for (k = 0; k < m; k++)
			out[i + k] = TY_FP(N.ty[e]) ? COLR(&p, p.root[0])[k] :
			    (double)COLI(&p, p.root[0])[k];
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
}

static double
eval_t(nid e, double t)
{
	int ok;

	if (!e)
		return 0;
	return eval_const_at(e, 0, t, &ok);
}

/* tridiagonal a x_{j-1} + b x_j + c x_{j+1} = d, in place in d */
static void
thomas(const double *a, const double *b, const double *c, double *d, int n)
{
	double *cp = xmalloc((size_t)n * sizeof *cp);
	int i;

	cp[0] = c[0] / b[0];
	d[0] = d[0] / b[0];
	for (i = 1; i < n; i++) {
		double m = b[i] - a[i] * cp[i - 1];
		cp[i] = c[i] / m;
		d[i] = (d[i] - a[i] * d[i - 1]) / m;
	}
	for (i = n - 2; i >= 0; i--)
		d[i] -= cp[i] * d[i + 1];
	free(cp);
}

/* cyclic tridiagonal by Sherman-Morrison */
static void
cyclic(const double *a, const double *b, const double *c, double *d, int n)
{
	double *bb = xmalloc((size_t)n * sizeof *bb), *u = xcalloc((size_t)n,
	    sizeof *u), g = -b[0], fac;
	int i;

	memcpy(bb, b, (size_t)n * sizeof *bb);
	bb[0] -= g;
	bb[n - 1] -= a[0] * c[n - 1] / g;
	thomas(a, bb, c, d, n);
	u[0] = g;
	u[n - 1] = c[n - 1];
	thomas(a, bb, c, u, n);
	fac = (d[0] + a[0] * d[n - 1] / g) / (1 + u[0] + a[0] * u[n - 1] / g);
	for (i = 0; i < n; i++)
		d[i] -= fac * u[i];
	free(bb);
	free(u);
}

/*
 * One grid solution with nx intervals at time tb; u gets nx + 1 values.
 * Returns the number of time steps.
 */
static int
pde_grid(const struct pde *pd, const struct pdec *C, int nx, double *u,
    double *dtp, const char **method)
{
	int n = nx + 1, j, step, M, per = pd->periodic, m = per ? nx : n;
	double L = pd->xb - pd->xa, dx = L / nx, T = pd->tb - pd->ta, dt, t;
	double *x = xmalloc((size_t)n * sizeof *x), *s0 = xcalloc((size_t)n,
	    sizeof *s0), *s1 = xcalloc((size_t)n, sizeof *s1);
	double *a = xmalloc((size_t)n * sizeof *a), *b = xmalloc((size_t)n *
	    sizeof *b), *c = xmalloc((size_t)n * sizeof *c);
	double *rhs = xmalloc((size_t)n * sizeof *rhs), *up = xmalloc((size_t)n *
	    sizeof *up), *un = xmalloc((size_t)n * sizeof *un);

	for (j = 0; j < n; j++)
		x[j] = pd->xa + j * dx;
	eval_xt(pd->u0, x, n, pd->ta, u);
#define UL(t) (pd->bl ? eval_t(pd->bl, t) : 0)
#define UR(t) (pd->br ? eval_t(pd->br, t) : 0)
	if (pd->order_t == 1 && (C->D > 0 || C->V == 0)) {
		/* Crank-Nicolson */
		double al = C->D / (dx * dx) - C->V / (2 * dx);
		double be = -2 * C->D / (dx * dx) + C->R;
		double ga = C->D / (dx * dx) + C->V / (2 * dx);
		M = (int)ceil(T / dx);
		if (M < 20)
			M = 20;
		dt = T / M;
		*method = "Crank-Nicolson";
		if (C->hasS)
			eval_xt(C->S, x, n, pd->ta, s0);
		for (step = 0; step < M; step++) {
			t = pd->ta + step * dt;
			if (C->hasS)
				eval_xt(C->S, x, n, t + dt, s1);
			for (j = 0; j < m; j++) {
				int jm = j - 1, jp = j + 1;
				if (per) {
					jm = (j + m - 1) % m;
					jp = (j + 1) % m;
				} else if (j == 0 || j == nx)
					continue;
				rhs[j] = u[j] + 0.5 * dt * (al * u[jm] + be * u[j] +
				    ga * u[jp]) + 0.5 * dt * (s0[j] + s1[j]);
				a[j] = -0.5 * dt * al;
				b[j] = 1 - 0.5 * dt * be;
				c[j] = -0.5 * dt * ga;
			}
			if (per) {
				cyclic(a, b, c, rhs, m);
				memcpy(u, rhs, (size_t)m * sizeof *u);
				u[nx] = u[0];
			} else {
				double l1 = UL(t + dt), r1 = UR(t + dt);
				rhs[1] -= a[1] * l1;
				rhs[nx - 1] -= c[nx - 1] * r1;
				a[1] = 0;
				c[nx - 1] = 0;
				thomas(a + 1, b + 1, c + 1, rhs + 1, nx - 1);
				memcpy(u + 1, rhs + 1, (size_t)(nx - 1) * sizeof *u);
				u[0] = l1;
				u[nx] = r1;
			}
			memcpy(s0, s1, (size_t)n * sizeof *s0);
		}
	} else if (pd->order_t == 1) {
		/* Lax-Wendroff for u_t = V u_x + R u + S */
		double nu;
		dt = 0.8 * dx / fabs(C->V);
		M = (int)ceil(T / dt);
		dt = T / M;
		nu = C->V * dt / dx;
		*method = "Lax-Wendroff";
		for (step = 0; step < M; step++) {
			t = pd->ta + step * dt;
			if (C->hasS)
				eval_xt(C->S, x, n, t, s0);
			for (j = 0; j < m; j++) {
				int jm = j - 1, jp = j + 1;
				if (per) {
					jm = (j + m - 1) % m;
					jp = (j + 1) % m;
				} else if (j == 0 || j == nx)
					continue;
				un[j] = u[j] + 0.5 * nu * (u[jp] - u[jm]) + 0.5 * nu * nu *
				    (u[jp] - 2 * u[j] + u[jm]) + dt * (C->R * u[j] + s0[j]);
			}
			if (per) {
				memcpy(u, un, (size_t)m * sizeof *u);
				u[nx] = u[0];
			} else {
				memcpy(u + 1, un + 1, (size_t)(nx - 1) * sizeof *u);
				u[0] = pd->bl ? UL(t + dt) : 2 * u[1] - u[2];
				u[nx] = pd->br ? UR(t + dt) : 2 * u[nx - 1] - u[nx - 2];
			}
		}
	} else {
		/* leapfrog for u_tt = D u_xx + V u_x + R u + DT u_t + S */
		double *v0 = xcalloc((size_t)n, sizeof *v0);
		dt = 0.8 * dx / sqrt(C->D);
		M = (int)ceil(T / dt);
		dt = T / M;
		*method = "leapfrog (central differences in x and t)";
		if (pd->v0)
			eval_xt(pd->v0, x, n, pd->ta, v0);
		memcpy(up, u, (size_t)n * sizeof *up);
		for (step = 0; step < M; step++) {
			t = pd->ta + step * dt;
			if (C->hasS)
				eval_xt(C->S, x, n, t, s0);
			for (j = 0; j < m; j++) {
				int jm = j - 1, jp = j + 1;
				double Lu;
				if (per) {
					jm = (j + m - 1) % m;
					jp = (j + 1) % m;
				} else if (j == 0 || j == nx)
					continue;
				Lu = C->D * (u[jp] - 2 * u[j] + u[jm]) / (dx * dx) +
				    C->V * (u[jp] - u[jm]) / (2 * dx) + C->R * u[j] + s0[j];
				if (step == 0)
					un[j] = u[j] + dt * v0[j] + 0.5 * dt * dt * (Lu +
					    C->DT * v0[j]);
				else
					/* (1 - k) u+ = 2u - (1 + k) u- + dt^2 Lu, k = DT dt/2 */
					un[j] = (2 * u[j] - (1 + 0.5 * C->DT * dt) * up[j] +
					    dt * dt * Lu) / (1 - 0.5 * C->DT * dt);
			}
			memcpy(up, u, (size_t)n * sizeof *up);
			if (per) {
				memcpy(u, un, (size_t)m * sizeof *u);
				u[nx] = u[0];
			} else {
				memcpy(u + 1, un + 1, (size_t)(nx - 1) * sizeof *u);
				u[0] = UL(t + dt);
				u[nx] = UR(t + dt);
			}
		}
		free(v0);
	}
#undef UL
#undef UR
	*dtp = dt;
	free(x);
	free(s0);
	free(s1);
	free(a);
	free(b);
	free(c);
	free(rhs);
	free(up);
	free(un);
	return M;
}

/* separation of variables (or the exact shift for advection) at x */
static int
pde_series(const struct pde *pd, const struct pdec *C, const double *x, int n,
    double *out, const char **how)
{
	double L = pd->xb - pd->xa, T = pd->tb - pd->ta;
	int j, k, Q = 4000, modes = 100;
	double *xs, *f0, *g0;

	if (C->hasS || C->V != 0 || C->DT != 0) {
		if (pd->periodic && pd->order_t == 1 && C->D == 0 && !C->hasS) {
			/* u(x, t) = u0(x + V t) e^(R t), periodic */
			double *xx = xmalloc((size_t)n * sizeof *xx);
			for (j = 0; j < n; j++) {
				double y = x[j] + C->V * T - pd->xa;
				y = fmod(y, L);
				if (y < 0)
					y += L;
				xx[j] = pd->xa + y;
			}
			eval_xt(pd->u0, xx, n, pd->ta, out);
			for (j = 0; j < n; j++)
				out[j] *= exp(C->R * T);
			free(xx);
			*how = "the exact solution u0(x + V t) e^(R t)";
			return 1;
		}
		return 0;
	}
	if (!pd->periodic && ((pd->bl && eval_t(pd->bl, pd->ta) != 0) ||
	    (pd->br && eval_t(pd->br, pd->ta) != 0)))
		return 0;
	if (pd->order_t == 1 && C->D <= 0)
		return 0;
	xs = xmalloc((size_t)(Q + 1) * sizeof *xs);
	f0 = xmalloc((size_t)(Q + 1) * sizeof *f0);
	g0 = xcalloc((size_t)(Q + 1), sizeof *g0);
	for (k = 0; k <= Q; k++)
		xs[k] = pd->xa + L * k / Q;
	eval_xt(pd->u0, xs, Q + 1, pd->ta, f0);
	if (pd->v0)
		eval_xt(pd->v0, xs, Q + 1, pd->ta, g0);
	for (j = 0; j < n; j++)
		out[j] = 0;
	for (k = pd->periodic ? 0 : 1; k <= modes; k++) {
		int ph, nph = pd->periodic && k > 0 ? 2 : 1;
		for (ph = 0; ph < nph; ph++) {
			/* basis: sin(k pi y / L) (Dirichlet), cos/sin(2 k pi y / L) */
			double w = pd->periodic ? 2 * M_PI * k / L : M_PI * k / L;
			double bf = 0, bg = 0, norm, lam, amp;
			int q;
			for (q = 0; q <= Q; q++) {
				double y = xs[q] - pd->xa, phi, sw;
				phi = pd->periodic ? (ph ? sin(w * y) : cos(w * y)) :
				    sin(w * y);
				sw = (q == 0 || q == Q) ? 1 : (q & 1) ? 4 : 2;
				bf += sw * f0[q] * phi;
				bg += sw * g0[q] * phi;
			}
			norm = pd->periodic ? (k == 0 ? L : L / 2) : L / 2;
			bf *= L / (3.0 * Q) / norm;
			bg *= L / (3.0 * Q) / norm;
			lam = -C->D * w * w + C->R;
			if (pd->order_t == 1)
				amp = bf * exp(lam * T);
			else {
				double om2 = -lam;
				if (om2 > 0) {
					double om = sqrt(om2);
					amp = bf * cos(om * T) + bg / om * sin(om * T);
				} else if (om2 < 0) {
					double ka = sqrt(-om2);
					amp = bf * cosh(ka * T) + bg / ka * sinh(ka * T);
				} else
					amp = bf + bg * T;
			}
			for (j = 0; j < n; j++) {
				double y = x[j] - pd->xa;
				out[j] += amp * (pd->periodic ? (ph ? sin(w * y) :
				    cos(w * y)) : sin(w * y));
			}
		}
	}
	free(xs);
	free(f0);
	free(g0);
	*how = pd->periodic ? "Fourier series (100 modes)" :
	    "separation of variables, sine series (100 modes)";
	return 1;
}

void
cmd_pde(uint32_t name, nid tend, int rows)
{
	struct pdeent *E = pde_find(name);
	struct pde pd;
	struct pdec C;
	nid map[MAXP], r2, d;
	int j, nx, M, k;
	double *u1, *u2, dt, dt2, maxd = 0, *xo, *so;
	const char *method, *how;

	if (!E)
		fail("no pde %s", sym_str(name));
	pd = E->pd;
	if (tend)
		num_of(tend, &pd.tb);
	/* name the roles: x 0, t 1, u 2, u_x 3, u_xx 4, u_t 5 */
	for (j = 0; j < pd.rf.np; j++) {
		uint32_t pn = pd.rf.p[j].name;
		const char *s = sym_str(pn), *us = sym_str(pd.u);
		size_t ul = strlen(us);
		int slot = -1;
		if (pn == pd.x)
			slot = 0;
		else if (pn == pd.t)
			slot = 1;
		else if (pn == pd.u)
			slot = 2;
		else if (!strncmp(s, us, ul) && s[ul] == '_') {
			if (!strcmp(s + ul, "_x"))
				slot = 3;
			else if (!strcmp(s + ul, "_xx"))
				slot = 4;
			else if (!strcmp(s + ul, "_t") && pd.order_t == 2)
				slot = 5;
		}
		if (slot < 0)
			fail("%s in the equation is not %s, %s_x, %s_xx, %s or %s "
			    "(define constants first: k = 2)", s, us, us, us,
			    sym_str(pd.x), sym_str(pd.t));
		map[j] = var_of(slot, pn, TY_REAL);
	}
	r2 = mk_cast(TY_REAL, subst(pd.rhs, map, pd.rf.np));
	memset(&C, 0, sizeof C);
	{
		double *cf[4] = {&C.R, &C.V, &C.D, &C.DT};
		for (k = 2; k <= 5; k++) {
			if (!guarded_diff(r2, k, &d))
				fail("the equation is not differentiable in its unknowns");
			d = tidy(d);
			if (!has_slot(r2, k))
				continue;
			if (!num_of(d, cf[k - 2]))
				fail("only linear equations with constant coefficients "
				    "(the coefficient of %s is %s)", k == 2 ? "u" :
				    k == 3 ? "u_x" : k == 4 ? "u_xx" : "u_t", node_str(d));
		}
	}
	{
		nid z[NSLOT];
		memset(z, 0, sizeof z);
		for (k = 2; k <= 5; k++)
			z[k] = KR(0);
		C.S = tidy(subst(r2, z, NSLOT));
		C.hasS = !const_is(C.S, 0);
	}
	printf("pde %s: %s_%s = %s on %s in [%g, %g], %s in [%g, %g]\n",
	    sym_str(name), sym_str(pd.u), pd.order_t == 1 ? "t" : "tt",
	    node_str(pd.rhs), sym_str(pd.x), pd.xa, pd.xb, sym_str(pd.t), pd.ta,
	    pd.tb);
	printf("  coefficients: u_xx %g, u_x %g, u %g", C.D, C.V, C.R);
	if (pd.order_t == 2)
		printf(", u_t %g", C.DT);
	printf("%s\n", C.hasS ? "; plus a source term" : "");
	if (pd.order_t == 1 && C.D < 0)
		fail("u_t = D u_xx with D < 0 is ill-posed (backward heat equation)");
	if (pd.order_t == 2 && C.D <= 0)
		fail("u_tt = c^2 u_xx needs c^2 > 0");
	printf("  boundary: %s\n", pd.periodic ? "periodic" :
	    pd.bl || pd.br ? "Dirichlet (given values)" :
	    pd.order_t == 1 && C.D == 0 ? "extrapolated (outflow)" :
	    "Dirichlet, u = 0 at both ends");
	nx = pd.nx;
	u1 = xmalloc((size_t)(2 * nx + 1) * sizeof *u1);
	u2 = xmalloc((size_t)(4 * nx + 1) * sizeof *u2);
	M = pde_grid(&pd, &C, nx, u1, &dt, &method);
	pde_grid(&pd, &C, 2 * nx, u2, &dt2, &method);
	for (j = 0; j <= nx; j++)
		maxd = fmax(maxd, fabs(u1[j] - u2[2 * j]));
	printf("  method: %s, %d intervals, dt = %.4g (%d steps)\n", method, nx,
	    dt, M);
	printf("  grid check: %d and %d intervals differ by at most %.2g\n", nx,
	    2 * nx, maxd);
	xo = xmalloc((size_t)(nx + 1) * sizeof *xo);
	so = xmalloc((size_t)(nx + 1) * sizeof *so);
	for (j = 0; j <= nx; j++)
		xo[j] = pd.xa + (pd.xb - pd.xa) * j / nx;
	if (pde_series(&pd, &C, xo, nx + 1, so, &how)) {
		double md = 0;
		for (j = 0; j <= nx; j++)
			md = fmax(md, fabs(so[j] - u2[2 * j]));
		printf("  %s agrees with the finer grid to %.2g\n", how, md);
	} else
		printf("  (no series solution for this case: the grid check is "
		    "the error estimate)\n");
	if (rows <= 0)
		rows = 10;
	if (rows > nx)
		rows = nx;
	printf("      %-10s %s(%s, %g)\n", sym_str(pd.x), sym_str(pd.u),
	    sym_str(pd.x), pd.tb);
	for (k = 0; k <= rows; k++) {
		int jj = (int)((double)k * nx / rows + 0.5);
		printf("      %-10.4g %.8g\n", xo[jj], u2[2 * jj]);
	}
	free(u1);
	free(u2);
	free(xo);
	free(so);
}

/* plot.c: a stored equation's right-hand side (NULL if none) */
const char *
ode_rhs_text(uint32_t var, int *order)
{
	struct odeq *q = eq_find(var);

	if (q == NULL)
		return NULL;
	*order = q->order;
	return q->rhs;
}

/* plot.c: a stored initial value var^(order)(t0) = val */
int
ode_ic_text(uint32_t var, int order, const char **t0, const char **val)
{
	int i;

	for (i = 0; i < nics; i++)
		if (ics[i].var == var && ics[i].order == order) {
			*t0 = ics[i].t0;
			*val = ics[i].val;
			return 1;
		}
	return 0;
}
