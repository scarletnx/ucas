/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Cheap evaluation schemes, priced with the active cost model:
 *
 *   polynomials    naive (pow), powers by multiplication, Horner, Estrin
 *                  (cost and critical path); forward differences for
 *                  tabulating at consecutive integers (d additions a point)
 *   powers         x^n by squaring: the addition chain against pow()
 *   loops          iteration against the closed form (and the trip count
 *                  where the closed form wins), the matrix power of a
 *                  linear recurrence (O(log n) by squaring), and strength
 *                  reduction of a polynomial loop body by differences
 *
 * Every rewritten form is evaluated against the original on the domain.
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- pricing a DAG: each node once; depth is the critical path ---- */

static float
dc1(const struct model *m, nid n, uint8_t *seen, float *dep, int *nops)
{
	float c = 0, d = 0, t;
	int cls;
	uint32_t j;

	if (n == 0)
		return 0;
	if (seen[n])
		return 0;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
		dep[n] = 0;
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			c += dc1(m, N.lst[N.a[n] + j], seen, dep, nops);
		dep[n] = 0;
		return c;
	case O_LOOP:
		c = loop_iter_cost(m, n);
		break;
	default:
		c = node_cost(m, n, &cls);
	}
	t = c;
	c += dc1(m, N.a[n], seen, dep, nops) + dc1(m, N.b[n], seen, dep, nops) +
	    dc1(m, N.c[n], seen, dep, nops);
	if (N.a[n] && dep[N.a[n]] > d)
		d = dep[N.a[n]];
	if (N.b[n] && dep[N.b[n]] > d)
		d = dep[N.b[n]];
	if (N.c[n] && N.op[n] != O_LOOP && dep[N.c[n]] > d)
		d = dep[N.c[n]];
	dep[n] = d + t;
	if (t > 0)
		(*nops)++;
	return c;
}

float
dag_cost(const struct model *m, nid root, int *nops, float *depth)
{
	uint8_t *seen = xcalloc(N.n, 1);
	float *dep = xcalloc(N.n, sizeof *dep), c;

	*nops = 0;
	c = dc1(m, root, seen, dep, nops);
	*depth = dep[root];
	free(seen);
	free(dep);
	return c;
}

/* ---- comparing two forms on the domain ---- */

struct cmpres {
	uint64_t n, bad;
	double maxrel;
	int exact;
};

static void
compare(const struct func *f, nid a, nid b, struct cmpres *r)
{
	struct vres v;

	memset(r, 0, sizeof *r);
	if (!TY_FP(N.ty[a])) {
		verify_nodes(f, a, b, &v);
		r->n = v.n;
		r->bad = v.bad + v.covf;
		r->exact = 1;
		vres_free(&v);
		return;
	}
	{
		struct plan p;
		struct diter d;
		nid roots[3] = {a, mk_cast(N.ty[a], b), f->where};
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		uint8_t ok[CH];
		uint32_t np, i;
		plan_build(&p, roots, 3, 0);
		diter_init(&d, f, 1u << 16, 201, 4096);
		while ((np = diter_next(&d, in_i, in_r)) != 0) {
			plan_run(&p, np, in_i, in_r);
			valid_mask(&p, f->where ? p.root[2] : NOTOK, np, ok);
			for (i = 0; i < np; i++) {
				double x = COLR(&p, p.root[0])[i], y = COLR(&p, p.root[1])[i];
				if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF) ||
				    !isfinite(x))
					continue;
				r->n++;
				if (!isfinite(y)) {
					r->bad++;
					continue;
				}
				if (x != y)
					r->maxrel = fmax(r->maxrel, fabs(x - y) /
					    fmax(fabs(x), 1e-300));
			}
		}
		plan_free(&p);
		free(in_i);
		free(in_r);
	}
}

static void
print_cmp(const struct cmpres *r)
{
	if (r->exact)
		printf(r->bad ? "differs at %llu of %llu points" :
		    "equal at all %llu points checked", r->bad ? (unsigned long long)
		    r->bad : (unsigned long long)r->n, (unsigned long long)r->n);
	else if (r->bad)
		printf("NOT FINITE at %llu of %llu points",
		    (unsigned long long)r->bad, (unsigned long long)r->n);
	else if (r->maxrel == 0)
		printf("bit-identical at %llu points", (unsigned long long)r->n);
	else
		printf("max relative difference %.2g (rounding) at %llu points",
		    r->maxrel, (unsigned long long)r->n);
}

static void
line(const struct model *m, const char *name, nid e, const struct func *f,
    nid ref)
{
	int nops;
	float depth, c = dag_cost(m, e, &nops, &depth);
	struct cmpres r;

	printf("  %-10s cost %-6g depth %-5g %2d ops  %s\n", name, (double)c,
	    (double)depth, nops, node_str(e));
	if (ref && ref != e) {
		compare(f, ref, e, &r);
		printf("  %-10s ", "");
		print_cmp(&r);
		printf("\n");
	}
}

/* ---- polynomials ---- */

static nid
mkc(ty_t t, double v)
{
	if (!TY_FP(t))
		return mk_const_i(t, (int64_t)v);
	if (TYK(t) == K_FLOAT)
		return mk_cast(t, KR(v));
	return KR(v);
}

static int
poly_scheme(const struct model *m, struct func *f)
{
	struct npoly P;
	double c[13];
	int s, d, k;
	ty_t t = N.ty[f->body];
	nid x, naive = 0, mulp = 0, horner, estrin, pw[13];

	for (s = 0; s < f->np; s++)
		if (npoly_of(f->body, s, &P, 12) && P.deg >= 2)
			break;
	if (s == f->np)
		return 0;
	for (k = 0; k <= P.deg; k++)
		if (!num_of(P.c[k], &c[k]))
			return 0;
	d = P.deg;
	x = var_of(s, f->p[s].name, f->p[s].ty);
	if (N.ty[x] != t)
		x = mk_cast(t, x);
	printf("polynomial of degree %d in %s:\n", d, sym_str(f->p[s].name));
	/* naive with pow, and with a chain of multiplications */
	pw[1] = x;
	for (k = 2; k <= d; k++)
		pw[k] = MUL(pw[k - 1], x);
	for (k = d; k >= 0; k--) {
		nid a, b;
		double ac = fabs(c[k]);
		if (c[k] == 0)
			continue;
		a = k == 0 ? mkc(t, ac) : ac == 1 ? POW(x, KI(k)) :
		    MUL(mkc(t, ac), POW(x, KI(k)));
		b = k == 0 ? mkc(t, ac) : ac == 1 ? pw[k] :
		    MUL(mkc(t, ac), pw[k]);
		if (k == 1)
			a = ac == 1 ? x : MUL(mkc(t, ac), x);
		if (c[k] < 0) {
			naive = naive ? SUB(naive, a) : mk_un(O_NEG, a);
			mulp = mulp ? SUB(mulp, b) : mk_un(O_NEG, b);
		} else {
			naive = naive ? ADD(naive, a) : a;
			mulp = mulp ? ADD(mulp, b) : b;
		}
	}
	horner = mkc(t, c[d]);
	for (k = d - 1; k >= 0; k--) {
		horner = MUL(horner, x);
		if (c[k] > 0)
			horner = ADD(horner, mkc(t, c[k]));
		else if (c[k] < 0)
			horner = SUB(horner, mkc(t, -c[k]));
	}
	/* Estrin: pairs a + b x, then combine with x^2, x^4, ... */
	{
		nid lev[13], xp = x;
		int n = d + 1, j;
		for (j = 0; j < n; j++)
			lev[j] = mkc(t, c[j]);
		while (n > 1) {
			int nn = 0;
			for (j = 0; j < n; j += 2) {
				double hv;
				nid hi;
				if (j + 1 >= n) {
					lev[nn++] = lev[j];
					continue;
				}
				/* lev[j] + lev[j+1] * xp, written without -1* and + -c */
				if (num_of(lev[j + 1], &hv) && fabs(hv) == 1)
					hi = xp;
				else if (num_of(lev[j + 1], &hv) && hv < 0)
					hi = MUL(mkc(t, -hv), xp);
				else
					hi = MUL(lev[j + 1], xp), hv = 1;
				if (const_is(lev[j], 0))
					lev[nn++] = hv < 0 ? mk_un(O_NEG, hi) : hi;
				else if (hv > 0 && num_of(lev[j], &hv) && hv < 0)
					lev[nn++] = SUB(hi, mkc(t, -hv));
				else
					lev[nn++] = hv < 0 ? SUB(lev[j], hi) : ADD(lev[j], hi);
			}
			n = nn;
			xp = MUL(xp, xp);
		}
		estrin = lev[0];
	}
	line(m, "as written", f->body, f, 0);
	line(m, "pow()", naive, f, f->body);
	line(m, "products", mulp, f, f->body);
	line(m, "Horner", horner, f, f->body);
	line(m, "Estrin", estrin, f, f->body);
	printf("  (depth is the critical path: Estrin trades operations for "
	    "parallelism)\n");
	/* forward differences for consecutive integers */
	if (!TY_FP(f->p[s].ty) && f->np == 1) {
		int64_t lo, hi, D[13], v, j, n, bad = 0;
		int64_t vals[14];
		param_bounds(&f->p[s], &lo, &hi);
		if (hi - lo > 100000)
			hi = lo + 100000;
		for (j = 0; j <= d; j++) {
			double acc = 0, xv = (double)(lo + j);
			for (k = d; k >= 0; k--)
				acc = acc * xv + c[k];
			vals[j] = (int64_t)acc;
		}
		for (j = 0; j <= d; j++)
			D[j] = vals[j];
		for (k = 1; k <= d; k++)
			for (j = d; j >= k; j--)
				D[j] -= D[j - 1];
		printf("  differences, for %s = %lld, %lld, ...: start", sym_str(
		    f->p[s].name), (long long)lo, (long long)lo + 1);
		for (j = 0; j <= d; j++)
			printf(" d%lld = %lld", (long long)j, (long long)D[j]);
		printf("\n    per point: ");
		for (j = 0; j < d; j++)
			printf("d%lld += d%lld; ", (long long)j, (long long)j + 1);
		printf("(%d additions, cost %g a point after the start)\n", d,
		    (double)(d * m->c[TY_FP(t) ? C_FADD : C_IADD]));
		/* check: run the difference engine against the polynomial */
		n = 0;
		for (v = lo; v <= hi; v++, n++) {
			double acc = 0;
			for (k = d; k >= 0; k--)
				acc = acc * (double)v + c[k];
			if (fabs(acc) < 9e15 && (int64_t)acc != D[0])
				bad++;
			for (j = 0; j < d; j++)
				D[j] += D[j + 1];
		}
		if (bad)
			printf("    check: differs at %lld of %lld points\n",
			    (long long)bad, (long long)n);
		else
			printf("    check: equals the polynomial at all %lld points "
			    "%lld..%lld\n", (long long)n, (long long)lo, (long long)hi);
	}
	return 1;
}

/* ---- powers by squaring ---- */

static nid
pow_chain(nid x, int64_t e)
{
	nid r = 0, sq = x;

	while (e) {
		if (e & 1)
			r = r ? MUL(r, sq) : sq;
		e >>= 1;
		if (e)
			sq = MUL(sq, sq);
	}
	return r;
}

static void
find_pows(nid e, nid *out, int *n, int max, uint8_t *seen)
{
	if (e == 0 || seen[e] || *n >= max)
		return;
	seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_TABLE ||
	    N.op[e] == O_LOOP)
		return;
	if (N.op[e] == O_POW)
		out[(*n)++] = e;
	find_pows(N.a[e], out, n, max, seen);
	find_pows(N.b[e], out, n, max, seen);
	find_pows(N.c[e], out, n, max, seen);
}

static void
pow_scheme(const struct model *m, struct func *f)
{
	nid pows[8];
	int n = 0, i;
	uint8_t *seen = xcalloc(N.n, 1);

	find_pows(f->body, pows, &n, 8, seen);
	free(seen);
	for (i = 0; i < n; i++) {
		nid p = pows[i], ch;
		int64_t e;
		int cls, nops;
		float cp, cc, dep;
		if (!int_of(N.b[p], &e)) {
			nid x = N.b[p];
			while (N.op[x] == O_CAST)
				x = N.a[x];
			if (!TY_FP(N.ty[x]))
				printf("%s: an integer power by squaring takes at most "
				    "2 log2(%s) multiplications\n", node_str(p),
				    node_str(x));
			continue;
		}
		if (e < 3 || e > 1000000)
			continue;
		ch = pow_chain(N.a[p], e);
		cp = node_cost(m, p, &cls);
		cc = dag_cost(m, ch, &nops, &dep);
		printf("%s by squaring: %d multiplications (%d squarings), cost %g "
		    "against %g for the power\n", node_str(p), nops,
		    ilog2u((uint64_t)e), (double)cc, (double)cp);
		printf("  %s\n", node_str(ch));
		{
			struct func g = *f;
			struct cmpres r;
			g.body = p;
			compare(&g, p, ch, &r);
			printf("  ");
			print_cmp(&r);
			printf("\n");
		}
	}
}

/* ---- loops ---- */

static void
find_loops2(nid e, nid *out, int *n, int max, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || *n >= max)
		return;
	seen[e] = 1;
	if (N.op[e] == O_LOOP) {
		if ((node_slots(e) >> MAXP) == 0)
			out[(*n)++] = e;
		return;
	}
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			find_loops2(N.lst[N.a[e] + j], out, n, max, seen);
		return;
	}
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return;
	find_loops2(N.a[e], out, n, max, seen);
	find_loops2(N.b[e], out, n, max, seen);
	find_loops2(N.c[e], out, n, max, seen);
}

/* the largest trip count over the domain */
static double
max_trips(const struct func *f, nid n)
{
	struct plan p;
	struct diter d;
	nid T = ADD(SUB(N.b[n], N.a[n]), KI(1)), roots[2] = {T, f->where};
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r), mx = 0;
	uint32_t np, i;
	uint8_t ok[CH];

	if (loops.kind[loop_ent(n)] != LK_FOR)
		return -1;
	plan_build(&p, roots, 2, 0);
	diter_init(&d, f, 1u << 16, 101, 4096);
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[1] : NOTOK, np, ok);
		for (i = 0; i < np; i++)
			if (ok[i] && !(COLF(&p, p.root[0])[i] & F_UNDEF))
				mx = fmax(mx, (double)COLI(&p, p.root[0])[i]);
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return mx;
}

/* integer matrix products in int64; 0 on overflow */
static int
imatmul(const int64_t *a, const int64_t *b, int64_t *c, int n)
{
	int i, j, l;
	int64_t s, t;

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {
			s = 0;
			for (l = 0; l < n; l++)
				if (__builtin_mul_overflow(a[i * n + l], b[l * n + j], &t) ||
				    __builtin_add_overflow(s, t, &s))
					return 0;
			c[i * n + j] = s;
		}
	return 1;
}

/* M^k v by squaring; exactly in int64 when everything is an integer */
static int
matpow(const double *M, const double *v, int n, int64_t k, int res,
    double *out, int exact, int64_t *iout)
{
	double A[256], R[256], T[256], x[16];
	int i, j, l;

	if (exact) {
		int64_t IA[256], IR[256], IT[256], s, t;
		for (i = 0; i < n * n; i++) {
			IA[i] = (int64_t)M[i];
			IR[i] = (i / n == i % n);
		}
		while (k > 0) {
			if (k & 1) {
				if (!imatmul(IR, IA, IT, n))
					return 0;
				memcpy(IR, IT, (size_t)n * n * sizeof *IR);
			}
			k >>= 1;
			if (k) {
				if (!imatmul(IA, IA, IT, n))
					return 0;
				memcpy(IA, IT, (size_t)n * n * sizeof *IA);
			}
		}
		s = 0;
		for (l = 0; l < n; l++)
			if (__builtin_mul_overflow(IR[res * n + l], (int64_t)v[l], &t) ||
			    __builtin_add_overflow(s, t, &s))
				return 0;
		*iout = s;
		return 1;
	}

	for (i = 0; i < n * n; i++) {
		A[i] = M[i];
		R[i] = (i / n == i % n);
	}
	while (k > 0) {
		if (k & 1) {
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++) {
					double s = 0;
					for (l = 0; l < n; l++)
						s += R[i * n + l] * A[l * n + j];
					T[i * n + j] = s;
				}
			memcpy(R, T, sizeof T);
		}
		k >>= 1;
		if (k) {
			for (i = 0; i < n; i++)
				for (j = 0; j < n; j++) {
					double s = 0;
					for (l = 0; l < n; l++)
						s += A[i * n + l] * A[l * n + j];
					T[i * n + j] = s;
				}
			memcpy(A, T, sizeof T);
		}
	}
	for (i = 0; i < n; i++) {
		double s = 0;
		for (l = 0; l < n; l++)
			s += R[i * n + l] * v[l];
		x[i] = s;
	}
	*out = x[res];
	return 1;
}

static void
loop_scheme(const struct model *m, struct func *f, nid n)
{
	float per = loop_iter_cost(m, n), cc;
	double T = max_trips(f, n), Mx[256], v0[16];
	nid cf;
	int nops, dim, res, j;
	float dep;

	printf("%s\n", node_str(n));
	if (T >= 0)
		printf("  iterating: %g per iteration, up to %g iterations: %g\n",
		    (double)per, T, (double)per * T);
	else
		printf("  iterating: %g per iteration (a while loop)\n",
		    (double)per);
	cf = loop_closed(n, NULL, NULL);
	if (cf) {
		struct vres v;
		cf = tidy(cf);
		cc = dag_cost(m, cf, &nops, &dep);
		verify_nodes(f, n, cf, &v);
		printf("  closed form: %g (%d ops): %s\n", (double)cc, nops,
		    node_str(cf));
		if (v.bad || !v.n)
			printf("    but it differs from the loop at %llu of %llu points "
			    "(e.g. %s): not used\n", (unsigned long long)v.bad,
			    (unsigned long long)v.n, sb_str(&v.wit));
		else {
			printf("    matches the loop at %llu points; ",
			    (unsigned long long)v.n);
			if (per > 0)
				printf("cheaper than iterating from %.0f iterations on\n",
				    ceil(cc / per));
			else
				printf("\n");
		}
		vres_free(&v);
	}
	if (lin_rec_matrix(n, Mx, v0, &dim, &res) && dim >= 2) {
		int exact = 1, bad = 0, chk = 0, skipped = 0;
		float mc = dim * dim * dim * m->c[C_IMUL] + dim * dim * (dim - 1) *
		    m->c[C_IADD];
		double steps = T > 1 ? ceil(log2(T)) : 1;
		ty_t st = N.ty[n];
		if (TY_FP(st))
			mc = dim * dim * dim * m->c[C_FMUL] + dim * dim * (dim - 1) *
			    m->c[C_FADD];
		for (j = 0; j < dim * dim; j++)
			if (Mx[j] != floor(Mx[j]) || fabs(Mx[j]) > 1e15)
				exact = 0;
		for (j = 0; j < dim; j++)
			if (v0[j] != floor(v0[j]) || fabs(v0[j]) > 1e15)
				exact = 0;
		if (TY_FP(st))
			exact = 0;
		printf("  matrix form: state' = M state, M = [");
		for (j = 0; j < dim * dim; j++)
			printf("%s%g", j == 0 ? "" : j % dim ? " " : "; ", Mx[j]);
		printf("]; the answer is (M^k v0)[%d]\n", res);
		printf("    M^k by squaring: at most %g matrix products of %g "
		    "each: %g (O(log k))\n", 2 * steps, (double)mc,
		    2 * steps * mc);
		/* check against the loop at sample points */
		{
			struct plan p;
			struct diter d;
			nid Tn = ADD(SUB(N.b[n], N.a[n]), KI(1));
			nid roots[3] = {n, Tn, f->where};
			int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
			double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
			uint8_t ok[CH];
			uint32_t np, i;
			plan_build(&p, roots, 3, 0);
			diter_init(&d, f, 1u << 12, 51, 512);
			while ((np = diter_next(&d, in_i, in_r)) != 0) {
				plan_run(&p, np, in_i, in_r);
				valid_mask(&p, f->where ? p.root[2] : NOTOK, np, ok);
				for (i = 0; i < np; i++) {
					double want, got = 0;
					int64_t k = COLI(&p, p.root[1])[i], ig = 0;
					if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF) ||
					    k < 0)
						continue;
					want = TY_FP(st) ? COLR(&p, p.root[0])[i] :
					    (double)COLI(&p, p.root[0])[i];
					if (!matpow(Mx, v0, dim, k, res, &got, exact, &ig)) {
						skipped++;
						continue;
					}
					chk++;
					if (exact ? ig != COLI(&p, p.root[0])[i] :
					    fabs(got - want) > 1e-9 * fmax(1, fabs(want)))
						bad++;
				}
			}
			plan_free(&p);
			free(in_i);
			free(in_r);
		}
		if (bad)
			printf("    check: differs from the loop at %d of %d points\n",
			    bad, chk);
		else
			printf("    check: equals the loop at %d points%s\n", chk,
			    exact ? " (in exact int64 arithmetic: no rounding, "
			    "unlike the closed form's real powers)" : "");
		if (skipped)
			printf("    (%d points not checked: the matrix power overflows "
			    "int64)\n", skipped);
		if (per > 0)
			printf("    cheaper than iterating once k > about %.0f\n",
			    ceil(2 * steps * mc / per));
	}
	/* strength reduction of a polynomial sum body */
	{
		struct loopdef d;
		struct npoly P;
		loop_get(loop_ent(n), &d);
		if (d.orig == LO_SUM && npoly_of(d.aux, d.islot, &P, 12) &&
		    P.deg >= 1) {
			ty_t t = N.ty[d.aux];
			int dg = P.deg;
			float hc = dg * (m->c[TY_FP(t) ? C_FMUL : C_IMUL] +
			    m->c[TY_FP(t) ? C_FADD : C_IADD]);
			float dc = dg * m->c[TY_FP(t) ? C_FADD : C_IADD];
			printf("  strength reduction: the body is a degree-%d "
			    "polynomial in %s; keep its %d differences and update "
			    "them by additions\n", dg, sym_str(d.name), dg);
			printf("    per iteration: Horner %g, differences %g\n",
			    (double)hc, (double)dc);
		}
	}
}

void
cmd_scheme(struct func *f, struct model *mm)
{
	const struct model *m = mm ? mm : model_active();
	nid loops[8];
	int nl = 0, i, any;
	uint8_t *seen;

	printf("evaluation schemes for %s under %s\n", f->anon ?
	    node_str(f->body) : sym_str(f->name), sym_str(m->name));
	any = poly_scheme(m, f);
	if (!any)
		pow_scheme(m, f);
	seen = xcalloc(N.n, 1);
	find_loops2(f->body, loops, &nl, 8, seen);
	free(seen);
	for (i = 0; i < nl; i++)
		loop_scheme(m, f, loops[i]);
	if (!any && nl == 0) {
		int nops;
		float dep, c = dag_cost(m, f->body, &nops, &dep);
		printf("  as written: cost %g, depth %g, %d operations (no "
		    "polynomial, power or loop to rewrite)\n", (double)c,
		    (double)dep, nops);
	}
	notes_drop();
}
