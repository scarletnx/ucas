/*
 * Integrals.  Symbolic antiderivatives for the elementary cases that can
 * be done reliably (polynomials, powers, exp/sin/cos/tan/log of a linear
 * argument, 1/u, polynomial times exp/sin/cos by parts), each checked by
 * differentiating it back; definite integrals numerically by adaptive
 * Gauss-Kronrod (7-15) with an error estimate, evaluated column-wise: all
 * pending intervals' nodes go through the plan together.
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static nid
xvar(nid f, int s)
{
	uint32_t j;
	nid r;

	if (f == 0 || !has_slot(f, s))
		return 0;
	if (N.op[f] == O_VAR)
		return f;
	if (N.op[f] == O_TABLE) {
		for (j = 0; j < N.b[f]; j++)
			if ((r = xvar(N.lst[N.a[f] + j], s)) != 0)
				return r;
		return 0;
	}
	if ((r = xvar(N.a[f], s)) != 0)
		return r;
	if ((r = xvar(N.b[f], s)) != 0)
		return r;
	return xvar(N.c[f], s);
}

/* u = a x + b with a a nonzero number: *a */
static int
linear_arg(nid u, int s, double *a)
{
	struct npoly p;

	if (!npoly_of(u, s, &p, 1) || p.deg != 1 || !num_of(p.c[1], a) || *a == 0)
		return 0;
	return 1;
}

static nid ad(nid f, int s, nid x);

/* integral of P(x) * g(u), g in exp sin cos, by parts (tabular) */
static nid
by_parts(nid P, nid g, int s, nid x)
{
	double a;
	nid u = N.a[g], acc = 0, d = P, term;
	int k, op = N.op[g];

	if (!linear_arg(u, s, &a))
		return 0;
	for (k = 0; k < 14 && !const_is(d, 0); k++) {
		/* the k-th derivative of P, divided by a^(k+1), alternating */
		nid c = DIV(d, KR(pow(a, k + 1)));
		if (op == O_EXP)
			term = MUL(c, g);
		else if (op == O_SIN)
			term = MUL(c, (k & 1) ? mk_un(O_SIN, u) :
			    mk_un(O_NEG, mk_un(O_COS, u)));
		else
			term = MUL(c, (k & 1) ? mk_un(O_COS, u) : mk_un(O_SIN, u));
		/* signs: exp alternates; sin/cos alternate in pairs */
		if (op == O_EXP ? (k & 1) : ((k >> 1) & 1))
			term = mk_un(O_NEG, term);
		acc = acc ? ADD(acc, term) : term;
		d = diff_node(d, (uint32_t)s);
	}
	if (!const_is(d, 0))
		return 0;
	(void)x;
	return acc;
}

static nid
ad(nid f, int s, nid x)
{
	struct npoly p;
	double a, k;
	nid u, r;
	int j;

	if (!has_slot(f, s))
		return MUL(f, x);
	if (npoly_of(f, s, &p, 12)) {
		r = 0;
		for (j = 0; j <= p.deg; j++) {
			nid t;
			if (const_is(p.c[j], 0))
				continue;
			t = DIV(MUL(p.c[j], POW(x, KR(j + 1))), KR(j + 1));
			if (j == 0)
				t = MUL(p.c[0], x);
			r = r ? ADD(r, t) : t;
		}
		return r ? r : KR(0);
	}
	switch (N.op[f]) {
	case O_ADD:
	case O_SUB: {
		nid A = ad(N.a[f], s, x), B = A ? ad(N.b[f], s, x) : 0;
		return A && B ? mk_bin(N.op[f], A, B) : 0;
	}
	case O_NEG:
		r = ad(N.a[f], s, x);
		return r ? mk_un(O_NEG, r) : 0;
	case O_CAST:
		if (!TY_FP(N.ty[f]))
			return 0;
		return ad(N.a[f], s, x);
	case O_MUL:
		if (!has_slot(N.a[f], s)) {
			r = ad(N.b[f], s, x);
			return r ? MUL(N.a[f], r) : 0;
		}
		if (!has_slot(N.b[f], s)) {
			r = ad(N.a[f], s, x);
			return r ? MUL(r, N.b[f]) : 0;
		}
		/* polynomial times exp/sin/cos of a linear argument */
		{
			nid A = N.a[f], B = N.b[f];
			int w;
			for (w = 0; w < 2; w++) {
				nid P = w ? B : A, g = w ? A : B;
				if ((N.op[g] == O_EXP || N.op[g] == O_SIN ||
				    N.op[g] == O_COS) && npoly_of(P, s, &p, 12) &&
				    (r = by_parts(P, g, s, x)) != 0)
					return r;
			}
		}
		return 0;
	case O_DIV:
		if (!has_slot(N.b[f], s)) {
			r = ad(N.a[f], s, x);
			return r ? DIV(r, N.b[f]) : 0;
		}
		/* c / u and c / u^n, u linear */
		if (!has_slot(N.a[f], s)) {
			u = N.b[f];
			if (linear_arg(u, s, &a))
				return DIV(MUL(N.a[f], mk_un(O_LOG, mk_un(O_ABS, u))),
				    KR(a));
			if (N.op[u] == O_POW && num_of(N.b[u], &k) &&
			    linear_arg(N.a[u], s, &a)) {
				if (k == 1)
					return DIV(MUL(N.a[f], mk_un(O_LOG,
					    mk_un(O_ABS, N.a[u]))), KR(a));
				return DIV(MUL(N.a[f], POW(N.a[u], KR(1 - k))),
				    KR(a * (1 - k)));
			}
		}
		return 0;
	case O_POW:
		u = N.a[f];
		if (has_slot(N.b[f], s)) {
			/* c^(a x + b) */
			if (has_slot(u, s) || !linear_arg(N.b[f], s, &a))
				return 0;
			return DIV(f, MUL(KR(a), mk_un(O_LOG, u)));
		}
		if (num_of(N.b[f], &k) && k == 2 && (N.op[u] == O_SIN ||
		    N.op[u] == O_COS) && linear_arg(N.a[u], s, &a)) {
			/* sin^2 u = (1 - cos 2u)/2, cos^2 u = (1 + cos 2u)/2 */
			nid v = N.a[u], h = DIV(mk_un(O_SIN, MUL(KR(2), v)), KR(4));
			return DIV(N.op[u] == O_SIN ? SUB(DIV(v, KR(2)), h) :
			    ADD(DIV(v, KR(2)), h), KR(a));
		}
		if (!num_of(N.b[f], &k) || !linear_arg(u, s, &a))
			return 0;
		if (k == -1)
			return DIV(mk_un(O_LOG, mk_un(O_ABS, u)), KR(a));
		return DIV(POW(u, KR(k + 1)), KR(a * (k + 1)));
	case O_SQRT:
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(MUL(KR(2), POW(u, KR(1.5))), KR(3 * a));
	case O_EXP:
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(f, KR(a));
	case O_SIN:
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(mk_un(O_NEG, mk_un(O_COS, u)), KR(a));
	case O_COS:
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(mk_un(O_SIN, u), KR(a));
	case O_TAN:
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(mk_un(O_NEG, mk_un(O_LOG, mk_un(O_ABS,
		    mk_un(O_COS, u)))), KR(a));
	case O_LOG:
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(SUB(MUL(u, f), u), KR(a));
	case O_LAMBERTW:
	case O_LAMBERTWM1:
		/* u = W e^W, so u/W = e^W: u (W - 1) + e^W, defined at u = 0 */
		u = N.a[f];
		if (!linear_arg(u, s, &a))
			return 0;
		return DIV(mk_bin(O_ADD, MUL(u, SUB(f, KR(1))), mk_un(O_EXP, f)),
		    KR(a));
	}
	return 0;
}

/* with the variable given: a constant c integrates to c x */
nid
antideriv_wrt(nid f, int s, nid xv, const char **how)
{
	if (xvar(f, s) == 0 && xv != 0) {
		if (how)
			*how = "";
		return mk_bin(O_MUL, mk_cast(TY_REAL, f), TY_FP(N.ty[xv]) ? xv :
		    mk_cast(TY_REAL, xv));
	}
	return antideriv(f, s, how);
}

nid
antideriv(nid f, int s, const char **how)
{
	nid x = xvar(f, s), r;
	jmp_buf saved;

	if (how)
		*how = "";
	if (x == 0)
		return 0;
	if (!TY_FP(N.ty[x]))
		x = mk_cast(TY_REAL, x);
	f = mk_cast(TY_REAL, f);
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp)) {
		memcpy(fail_jmp, saved, sizeof saved);
		notes_drop();
		return 0;
	}
	r = ad(f, s, x);
	memcpy(fail_jmp, saved, sizeof saved);
	return r ? tidy(r) : 0;
}

/* ---- evaluation helpers ---- */

double
eval_const_at(nid e, int slot, double x, int *ok)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r), v;
	uint32_t l;

	plan_build(&p, &e, 1, 0);
	in_r[(size_t)slot * CH] = x;
	in_i[(size_t)slot * CH] = (int64_t)x;
	plan_run(&p, 1, in_i, in_r);
	l = p.root[0];
	*ok = !(COLF(&p, l)[0] & F_UNDEF);
	v = TY_FP(N.ty[e]) ? COLR(&p, l)[0] : (double)COLI(&p, l)[0];
	if (isnan(v) || isinf(v))
		*ok = 0;
	plan_free(&p);
	free(in_i);
	free(in_r);
	return v;
}

/* ---- adaptive Gauss-Kronrod 7-15 ---- */

static const double xgk[8] = {
	0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
	0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
	0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
	0.207784955007898467600689403773245, 0.000000000000000000000000000000000,
};
static const double wgk[8] = {
	0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
	0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
	0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
	0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
};
static const double wg[4] = {
	0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
	0.381830050505118944950369775488975, 0.417959183673469387755102040816327,
};

struct gkint {
	double a, b, v, e;
};

/*
 * x(t) for a mapped interval: mode 0 none, 1 [a, inf): x = a + t/(1-t),
 * 2 (-inf, b]: x = b - (1-t)/t, 3 (-inf, inf): x = t/(1-t^2).
 */
static double
xmap(int mode, double a, double t, double *jac)
{
	switch (mode) {
	case 1:
		*jac = 1 / ((1 - t) * (1 - t));
		return a + t / (1 - t);
	case 2:
		*jac = 1 / (t * t);
		return a - (1 - t) / t;
	case 3:
		*jac = (1 + t * t) / ((1 - t * t) * (1 - t * t));
		return t / (1 - t * t);
	}
	*jac = 1;
	return t;
}

int
gk_integrate(const struct func *f, nid e, int slot, double a, double b,
    double *val, double *err, int *nev)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	struct gkint *iv = xmalloc(4096 * sizeof *iv);
	int niv = 0, mode = 0, i, j, k, bad = 0, iter;
	double ta = a, tb = b, base = 0, sgn = 1, tot, toterr;
	uint32_t l;

	(void)f;
	if (a > b) {
		double t = a;
		a = b;
		b = t;
		sgn = -1;
	}
	if (isinf(a) && isinf(b))
		mode = 3, ta = -1, tb = 1;
	else if (isinf(b))
		mode = 1, base = a, ta = 0, tb = 1;
	else if (isinf(a))
		mode = 2, base = b, ta = 0, tb = 1;
	else
		ta = a, tb = b;
	e = mk_cast(TY_REAL, e);
	plan_build(&p, &e, 1, 0);
	l = p.root[0];
	*nev = 0;
	iv[niv++] = (struct gkint){ta, tb, 0, -1};
	for (iter = 0; iter < 200; iter++) {
		/* evaluate every interval without a value: 15 points each */
		int pend[17], np = 0, w;
		for (i = 0; i < niv && np < 17; i++)
			if (iv[i].e < 0)
				pend[np++] = i;
		if (np == 0) {
			/* split the worst interval */
			int worst = 0;
			tot = 0;
			toterr = 0;
			for (i = 0; i < niv; i++) {
				tot += iv[i].v;
				toterr += iv[i].e;
				if (iv[i].e > iv[worst].e)
					worst = i;
			}
			if (toterr <= fmax(1e-13 * fabs(tot), 1e-15) || niv >= 4000)
				break;
			{
				struct gkint w0 = iv[worst];
				double m = 0.5 * (w0.a + w0.b);
				iv[worst] = (struct gkint){w0.a, m, 0, -1};
				iv[niv++] = (struct gkint){m, w0.b, 0, -1};
			}
			continue;
		}
		for (w = 0; w < np; w++) {
			struct gkint *g = &iv[pend[w]];
			double c = 0.5 * (g->a + g->b), h = 0.5 * (g->b - g->a), jac;
			for (k = 0; k < 15; k++) {
				double t = k < 7 ? c - h * xgk[k] : k == 7 ? c :
				    c + h * xgk[14 - k];
				in_r[(size_t)slot * CH + w * 15 + k] = xmap(mode, base, t,
				    &jac);
			}
		}
		plan_run(&p, (uint32_t)np * 15, in_i, in_r);
		*nev += np * 15;
		for (w = 0; w < np; w++) {
			struct gkint *g = &iv[pend[w]];
			double c = 0.5 * (g->a + g->b), h = 0.5 * (g->b - g->a), jac;
			double fv[15], rk = 0, rg = 0;
			for (k = 0; k < 15; k++) {
				double t = k < 7 ? c - h * xgk[k] : k == 7 ? c :
				    c + h * xgk[14 - k];
				xmap(mode, base, t, &jac);
				if (COLF(&p, l)[w * 15 + k] & F_UNDEF ||
				    !isfinite(COLR(&p, l)[w * 15 + k])) {
					bad = 1;
					fv[k] = 0;
				} else
					fv[k] = COLR(&p, l)[w * 15 + k] * jac;
			}
			for (j = 0; j < 7; j++) {
				rk += wgk[j] * (fv[j] + fv[14 - j]);
				if (j & 1)
					rg += wg[j / 2] * (fv[j] + fv[14 - j]);
			}
			rk += wgk[7] * fv[7];
			rg += wg[3] * fv[7];
			g->v = rk * h;
			g->e = fabs((rk - rg) * h);
			/* the usual QUADPACK rescaling of the estimate */
			if (g->e > 0)
				g->e = fmax(g->e * fmin(1, pow(200 * g->e /
				    fmax(fabs(g->v), 1e-300), 1.5)), 50 * 2.2e-16 *
				    fabs(g->v));
		}
		if (bad)
			break;
	}
	tot = 0;
	toterr = 0;
	for (i = 0; i < niv; i++) {
		tot += iv[i].v;
		toterr += iv[i].e < 0 ? 0 : iv[i].e;
	}
	*val = sgn * tot;
	*err = toterr;
	plan_free(&p);
	free(in_i);
	free(in_r);
	free(iv);
	return !bad;
}

/* ---- the command ---- */

static nid
at_const(nid e, int s, double x)
{
	nid map[NSLOT];

	memset(map, 0, sizeof map);
	map[s] = KR(x);
	return subst(mk_cast(TY_REAL, e), map, NSLOT);
}

void
cmd_integrate(struct func *f, int slot, nid an, nid bn)
{
	nid F, body = f->body, ibody;
	int removable = 0;
	double a = 0, b = 0, lo, hi, maxres = 0, scale = 0;
	int defd = an && bn, i, ok, checked = 0;
	const char *xn;

	if (slot >= f->np)
		fail("%s has no variable to integrate over", f->anon ?
		    node_str(body) : sym_str(f->name));
	xn = sym_str(f->p[slot].name);
	if (defd) {
		num_of(an, &a);
		num_of(bn, &b);
	}
	printf("∫ %s d%s\n", node_str(body), xn);
	{
		/* x/x is 1 where x != 0: integrate the simpler form, say so */
		struct sbuf conds = {0};
		nid xv = mk_var((uint32_t)slot, f->p[slot].name, f->p[slot].ty);
		nid simpler = cancel_quotients(mk_cast(TY_REAL, body), slot,
		    TY_FP(N.ty[xv]) ? xv : mk_cast(TY_REAL, xv), &conds);
		if (conds.n)
			printf("  note: %s = %s where %s; the result holds there\n",
			    node_str(body), node_str(simpler), conds.p);
		/*
		 * the excluded points are removable: a definite integral does
		 * not see them, so it uses the simpler form
		 */
		ibody = simpler;
		removable = conds.n > 0;
		F = antideriv_wrt(simpler, slot, xv, NULL);
		free(conds.p);
	}
	/* where to check: the interval, the parameter's range, or -10..10 */
	if (defd && isfinite(a) && isfinite(b))
		lo = fmin(a, b), hi = fmax(a, b);
	else if (defd && isfinite(a))
		lo = a, hi = a + 10;
	else if (defd && isfinite(b))
		lo = b - 10, hi = b;
	else if (f->p[slot].ranged)
		lo = f->p[slot].rlo, hi = f->p[slot].rhi;
	else
		lo = -10, hi = 10;
	if (F) {
		nid d = 0;
		jmp_buf saved;
		memcpy(saved, fail_jmp, sizeof saved);
		if (setjmp(fail_jmp) == 0)
			d = diff_node(F, (uint32_t)slot);
		memcpy(fail_jmp, saved, sizeof saved);
		notes_drop();
		printf("  = %s + C\n", node_str(F));
		if (f->np == 1 && d) {
			for (i = 0; i <= 400; i++) {
				double x = lo + (hi - lo) * i / 400.0, fv, dv;
				int o1, o2;
				fv = eval_const_at(mk_cast(TY_REAL, body), slot, x, &o1);
				dv = eval_const_at(d, slot, x, &o2);
				if (!o1 || !o2)
					continue;
				checked++;
				scale = fmax(scale, fabs(fv));
				maxres = fmax(maxres, fabs(fv - dv));
			}
			if (checked && maxres <= 1e-9 * fmax(1, scale))
				printf("  check: d/d%s of it equals the integrand at %d "
				    "points of [%g, %g] (max difference %.2g)\n", xn,
				    checked, lo, hi, maxres);
			else if (checked)
				printf("  check FAILED: d/d%s differs from the integrand by "
				    "%.3g on [%g, %g]\n", xn, maxres, lo, hi);
		} else if (d)
			printf("  (several parameters: the derivative check needs "
			    "values; checked symbolically only)\n");
	} else
		printf("  no antiderivative found (the rules here cover polynomials, powers and exp/sin/cos/tan/log of a linear argument, times a polynomial)\n");
	if (!defd)
		return;
	if (removable)
		printf("  (the excluded points are removable singularities: the "
		    "definite integral does not depend on them)\n");
	if (f->np != 1) {
		printf("  a definite integral needs the other parameters fixed\n");
		return;
	}
	{
		double v, err, sv = 0;
		int nev, good = gk_integrate(f, ibody, slot, a, b, &v, &err, &nev);
		int symok = 0;
		if (F && isfinite(a) && isfinite(b)) {
			nid fa = at_const(F, slot, a), fb = at_const(F, slot, b);
			double va, vb;
			/* F(b) - F(a) holds only if the integrand is defined between */
			int cont = 1;
			for (i = 1; i < 400; i++) {
				eval_const_at(mk_cast(TY_REAL, ibody), slot,
				    lo + (hi - lo) * i / 400.0, &ok);
				if (!ok)
					cont = 0;
			}
			if (cont && num_of(fa, &va) && num_of(fb, &vb)) {
				sv = vb - va;
				symok = 1;
				printf("  from %g to %g: F(%g) - F(%g) = %.15g\n", a, b, b,
				    a, sv);
			}
		}
		if (good) {
			printf("  numeric (adaptive Gauss-Kronrod 7-15, %d evaluations): "
			    "%.15g, estimated error %.2g\n", nev, v, err);
			if (symok)
				printf("  the two agree to %.2g%s\n", fabs(v - sv),
				    fabs(v - sv) <= 10 * err + 1e-12 * fabs(sv) ? "" :
				    "  (MORE than the error estimate: check the "
				    "antiderivative's continuity)");
		} else
			printf("  numeric: the integrand is undefined or infinite "
			    "inside the interval (singular or divergent)\n");
	}
}
