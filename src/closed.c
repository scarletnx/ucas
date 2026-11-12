/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Closed forms of loops.
 *
 *   sums     polynomial bodies (Faulhaber), r^i times a polynomial
 *            (geometric, arithmetico-geometric), telescoping g(i) - g(i+k),
 *            rational bodies with integer poles (partial fractions), and
 *            bodies that branch on i % m or on i < c (split sums)
 *   products constants, powers r^e(i), exp(e(i)), telescoping ratios,
 *            ratios of linear factors with integer offsets
 *   loops    state machines whose step is affine with constant
 *            coefficients (linear recurrences: characteristic roots,
 *            repeated and complex roots, matrix form); one state at a
 *            time otherwise: accumulators (sums, products), x' = a x + g(k)
 *            (particular solutions), power maps, Mobius maps, quadratic
 *            maps conjugate to y' = y^2
 *   while    trip counts of v += d, v *= c, v /= c, v >>= k against a
 *            bound, and v*v <= n
 *
 * Nothing is claimed unchecked: cmd_closed evaluates every closed form
 * against direct iteration on the function's domain.
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
	M_FREE, M_FAUL, M_GEOM, M_AGEO, M_TELE, M_PFRAC, M_SPLITMOD, M_SPLITLT,
	M_LINREC, M_AFFINE, M_ACC, M_POWMAP, M_MOBIUS, M_QUAD, M_TRIP,
	M_PRODPOW, M_PRODTELE, M_PRODRAT, M_LAST, M_N
};
static const char *const mname[M_N] = {
	"constant body", "Faulhaber (polynomial in the index)", "geometric",
	"arithmetico-geometric", "telescoping", "partial fractions",
	"split by residue", "split at a threshold",
	"linear recurrence (characteristic roots)",
	"affine recurrence (particular solution)", "accumulator",
	"power map", "Mobius map (2x2 matrix power)",
	"quadratic map conjugate to y' = y^2", "trip count",
	"product of powers", "telescoping product",
	"ratio of shifted linear factors", "last value",
};
static unsigned cl_meth;
static struct sbuf cl_extra;
static int cl_depth;

#define METH(m) (cl_meth |= 1u << (m))

/* ---- verification against the evaluator ---- */

int
verify_nodes(const struct func *f, nid ref, nid cand, struct vres *v)
{
	struct plan p;
	struct diter d;
	nid roots[3];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint8_t ok[CH];
	uint32_t np, i, lr, lc, wl;
	ty_t rt = N.ty[ref], ct = N.ty[cand];

	memset(v, 0, sizeof *v);
	roots[0] = ref;
	roots[1] = cand;
	roots[2] = f->where;
	plan_build(&p, roots, 3, 0);
	lr = p.root[0];
	lc = p.root[1];
	wl = f->where ? p.root[2] : NOTOK;
	diter_init(&d, f, 1u << 16, 201, 4096);
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		uint64_t sk = loop_skipped;
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wl, np, ok);
		for (i = 0; i < np; i++) {
			int ur, uc, same;
			double a, b, rel;
			if (!ok[i])
				continue;
			ur = COLF(&p, lr)[i] & F_UNDEF;
			uc = COLF(&p, lc)[i] & F_UNDEF;
			if (ur && uc)
				continue;
			if (ur && loop_skipped != sk) {
				v->skipped++;
				continue;
			}
			if (ur || uc) {
				/* one side left the 64-bit range or its domain */
				struct sbuf *w = ur ? &v->wrovf : &v->wcovf;
				if (!(ur ? v->rovf++ : v->covf++))
					fmt_point(w, f, in_i, in_r, i);
				continue;
			}
			a = TY_FP(rt) ? COLR(&p, lr)[i] : (double)COLI(&p, lr)[i];
			b = TY_FP(ct) ? COLR(&p, lc)[i] : (double)COLI(&p, lc)[i];
			if (ur != uc)
				same = 0;
			else if (!TY_FP(rt) && !TY_FP(ct))
				same = COLI(&p, lr)[i] == COLI(&p, lc)[i];
			else if (!TY_FP(rt))
				same = fabs(b) < 9e15 && (int64_t)floor(b + 0.5) ==
				    COLI(&p, lr)[i];
			else {
				double sc = fmax(1, fmax(fabs(a), fabs(b)));
				rel = fabs(a - b) / sc;
				same = (isnan(a) && isnan(b)) || a == b || rel <= 1e-9;
				if (same && rel > v->maxrel)
					v->maxrel = rel;
			}
			v->n++;
			if (!same && !v->bad++) {
				fmt_point(&v->wit, f, in_i, in_r, i);
				sb_puts(&v->wit, ": ");
				if (ur)
					sb_puts(&v->wit, "undefined");
				else
					fmt_val(&v->wit, rt, COLI(&p, lr)[i],
					    COLR(&p, lr)[i]);
				sb_puts(&v->wit, " by iteration vs ");
				if (uc)
					sb_puts(&v->wit, "undefined");
				else
					fmt_val(&v->wit, ct, COLI(&p, lc)[i],
					    COLR(&p, lc)[i]);
			}
		}
	}
	v->exhaustive = d.exhaustive;
	plan_free(&p);
	free(in_i);
	free(in_r);
	return v->bad == 0 && v->covf == 0 && v->n > 0;
}

void
vres_free(struct vres *v)
{
	free(v->wit.p);
	free(v->wcovf.p);
	free(v->wrovf.p);
	memset(v, 0, sizeof *v);
}

void
print_verify(const char *what, const struct vres *v)
{
	if (v->n)
		rep_check("iteration", (long)v->n, v->maxrel, !v->bad,
		    v->exhaustive ? "against direct iteration on every point of "
		    "the domain" : "against direct iteration (sampled)");
	if (v->n && !v->bad)
		rep_proof(v->exhaustive ? "exhaustive" : "numeric", v->exhaustive ?
		    "equals direct iteration on every point of the domain" :
		    "equals direct iteration at sampled points");
	if (v->bad) {
		printf("  check: %s differs from direct iteration at %llu of %llu "
		    "points, e.g. %s\n", what, (unsigned long long)v->bad,
		    (unsigned long long)v->n, sb_str((struct sbuf *)&v->wit));
		return;
	}
	if (v->n == 0) {
		printf("  check: no point of the domain could be checked\n");
		return;
	}
	printf("  check: equals direct iteration at %s%llu points%s",
	    v->exhaustive ? "all " : "", (unsigned long long)v->n,
	    v->exhaustive ? " of the domain" : " (sampled, not exhaustive)");
	if (v->maxrel > 0)
		printf(", max relative difference %.2g", v->maxrel);
	printf("\n");
	if (v->covf)
		printf("  but at %llu points the closed form is undefined (an "
		    "intermediate leaves the 64-bit range) where iteration is not, "
		    "e.g. %s\n", (unsigned long long)v->covf,
		    sb_str((struct sbuf *)&v->wcovf));
	if (v->rovf)
		printf("  (at %llu points the iteration itself is undefined or "
		    "overflows, e.g. %s; not compared)\n",
		    (unsigned long long)v->rovf, sb_str((struct sbuf *)&v->wrovf));
	if (v->skipped)
		printf("  (%llu points not checked: more than %u iterations)\n",
		    (unsigned long long)v->skipped, MAXTRIP);
}

/* ---- tidying a closed form for printing: a + c1 - c2 -> a + c ---- */

static int
tidy_ty(ty_t t)
{
	return TYK(t) == K_INT || TYK(t) == K_LIT || TYK(t) == K_REAL;
}

struct tterm {
	nid e[64];
	int sg[64], n;
	int64_t ci;
	double cr;
};

static nid tidy1(nid e, nid *memo, uint32_t lim);

static void
tcollect(struct tterm *T, nid e, int sg, ty_t t, nid *memo, uint32_t lim)
{
	double v;
	int j;

	if (N.ty[e] == t || (TYK(N.ty[e]) == K_LIT && TYK(t) == K_INT)) {
		switch (N.op[e]) {
		case O_ADD:
			tcollect(T, N.a[e], sg, t, memo, lim);
			tcollect(T, N.b[e], sg, t, memo, lim);
			return;
		case O_SUB:
			tcollect(T, N.a[e], sg, t, memo, lim);
			tcollect(T, N.b[e], -sg, t, memo, lim);
			return;
		case O_NEG:
			tcollect(T, N.a[e], -sg, t, memo, lim);
			return;
		case O_CONST:
			num_of(e, &v);
			if (TY_FP(t))
				T->cr += sg * v;
			else
				T->ci += sg * N.ki[e];
			return;
		case O_MUL:
		case O_DIV:
			if (num_of(N.a[e], &v) && v < 0) {
				nid a = TY_FP(N.ty[N.a[e]]) ? KR(-v) :
				    mk_const_i(N.ty[N.a[e]], -N.ki[N.a[e]]);
				nid b = tidy1(N.b[e], memo, lim);
				e = mk_bin(N.op[e], a, b);
				sg = -sg;
				if (N.op[e] == O_CONST) {
					tcollect(T, e, sg, t, memo, lim);
					return;
				}
				goto term;
			}
			break;
		}
	}
	e = tidy1(e, memo, lim);
term:
	/* x - x */
	for (j = 0; j < T->n; j++)
		if (T->e[j] == e && T->sg[j] == -sg && !(N.fl[e] & NF_PARTIAL)) {
			T->e[j] = T->e[--T->n];
			T->sg[j] = T->sg[T->n];
			return;
		}
	if (T->n < 64) {
		T->e[T->n] = e;
		T->sg[T->n++] = sg;
	}
}

static nid
tidy1(nid e, nid *memo, uint32_t lim)
{
	nid r;

	if (e == 0)
		return 0;
	if (e < lim && memo[e])
		return memo[e];
	switch (N.op[e]) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
	case O_TABLE:
	case O_LOOP:
	case O_BARRIER:
	case O_ATOMIC:
	case O_INDEX:
		r = e;
		break;
	case O_SEL:
		r = mk_sel(tidy1(N.a[e], memo, lim), tidy1(N.b[e], memo, lim),
		    tidy1(N.c[e], memo, lim));
		break;
	case O_CAST:
		r = mk_cast(N.ty[e], tidy1(N.a[e], memo, lim));
		break;
	case O_NEG: {
		nid a = tidy1(N.a[e], memo, lim);
		if (N.op[a] == O_SUB && tidy_ty(N.ty[e]))
			r = SUB(N.b[a], N.a[a]);
		else
			r = mk_un(O_NEG, a);
		break;
	}
	case O_DIV: {
		/* a / -c -> -(a / c), (x - y) / -c -> (y - x) / c */
		double v;
		nid a = tidy1(N.a[e], memo, lim);
		if (TY_FP(N.ty[e]) && TYK(N.ty[e]) == K_REAL &&
		    num_of(N.b[e], &v) && v < 0) {
			if (N.op[a] == O_SUB)
				r = DIV(SUB(N.b[a], N.a[a]), KR(-v));
			else
				r = mk_un(O_NEG, DIV(a, KR(-v)));
		} else
			r = mk_bin(O_DIV, a, tidy1(N.b[e], memo, lim));
		break;
	}
	case O_ADD:
	case O_SUB:
		if (tidy_ty(N.ty[e])) {
			struct tterm T;
			int j, first = -1;
			ty_t t = N.ty[e];
			memset(&T, 0, sizeof T);
			tcollect(&T, e, 1, t, memo, lim);
			r = 0;
			for (j = 0; j < T.n; j++)
				if (T.sg[j] > 0) {
					first = j;
					break;
				}
			if (first >= 0)
				r = T.e[first];
			for (j = 0; j < T.n; j++) {
				if (j == first || T.sg[j] < 0)
					continue;
				r = ADD(r, T.e[j]);
			}
			if (!r && (TY_FP(t) ? T.cr != 0 : T.ci != 0)) {
				r = TY_FP(t) ? KR(T.cr) : mk_const_i(t, T.ci);
				T.cr = 0;
				T.ci = 0;
			}
			for (j = 0; j < T.n; j++) {
				if (T.sg[j] > 0)
					continue;
				r = r ? SUB(r, T.e[j]) : mk_un(O_NEG, T.e[j]);
			}
			if (TY_FP(t) ? T.cr > 0 : T.ci > 0)
				r = r ? ADD(r, TY_FP(t) ? KR(T.cr) : KI(T.ci)) :
				    (TY_FP(t) ? KR(T.cr) : KI(T.ci));
			else if (TY_FP(t) ? T.cr < 0 : T.ci < 0)
				r = r ? SUB(r, TY_FP(t) ? KR(-T.cr) : KI(-T.ci)) :
				    (TY_FP(t) ? KR(T.cr) : KI(T.ci));
			if (!r)
				r = mk_const_i(t, 0);
			r = mk_cast(t, r);
			break;
		}
		/* FALLTHROUGH */
	default:
		if (opinfo[N.op[e]].arity == 1)
			r = mk_un(N.op[e], tidy1(N.a[e], memo, lim));
		else if (IS_TEX(N.op[e]))
			r = mk_tex(N.op[e], N.ki[e], N.a[e],
			    tidy1(N.b[e], memo, lim),
			    N.c[e] ? tidy1(N.c[e], memo, lim) : 0);
		else if (IS_DOT(N.op[e]) || IS_SG(N.op[e]))
			r = mk_op(N.op[e], tidy1(N.a[e], memo, lim),
			    tidy1(N.b[e], memo, lim),
			    N.c[e] ? tidy1(N.c[e], memo, lim) : 0);
		else
			r = mk_bin(N.op[e], tidy1(N.a[e], memo, lim),
			    tidy1(N.b[e], memo, lim));
	}
	if (e < lim)
		memo[e] = r;
	return r;
}

nid
tidy(nid e)
{
	uint32_t lim = N.n;
	nid *memo = xcalloc(lim, sizeof *memo), r;

	r = tidy1(e, memo, lim);
	free(memo);
	return r;
}

/* ---- polynomials in one variable with node coefficients ---- */

static nid
zero_of(nid e)
{
	return mk_const_i(TY_FP(N.ty[e]) ? N.ty[e] : TY_LIT, 0);
}

static int
np_mul(const struct npoly *a, const struct npoly *b, struct npoly *r,
    int maxdeg)
{
	int i, j;
	struct npoly t;

	if (a->deg + b->deg > maxdeg || a->deg + b->deg > 12)
		return 0;
	t.deg = a->deg + b->deg;
	for (i = 0; i <= t.deg; i++)
		t.c[i] = KI(0);
	for (i = 0; i <= a->deg; i++)
		for (j = 0; j <= b->deg; j++)
			if (!const_is(a->c[i], 0) && !const_is(b->c[j], 0))
				t.c[i + j] = ADD(t.c[i + j], MUL(a->c[i], b->c[j]));
	*r = t;
	return 1;
}

static void
np_trim(struct npoly *p)
{
	while (p->deg > 0 && const_is(p->c[p->deg], 0))
		p->deg--;
}

int
npoly_of(nid e, int s, struct npoly *p, int maxdeg)
{
	struct npoly a, b;
	int i;
	int64_t k;

	if (!has_slot(e, s)) {
		p->deg = 0;
		p->c[0] = e;
		return 1;
	}
	switch (N.op[e]) {
	case O_VAR:
		if (N.ki[e] != s)
			return 0;
		p->deg = 1;
		p->c[0] = KI(0);
		p->c[1] = KI(1);
		return 1;
	case O_CAST:
		if (!npoly_of(N.a[e], s, p, maxdeg))
			return 0;
		if (!TY_FP(N.ty[e]) && TYK(N.ty[e]) != K_INT)
			return 0;
		for (i = 0; i <= p->deg; i++)
			p->c[i] = mk_cast(N.ty[e], p->c[i]);
		return 1;
	case O_NEG:
		if (!npoly_of(N.a[e], s, p, maxdeg))
			return 0;
		for (i = 0; i <= p->deg; i++)
			p->c[i] = mk_un(O_NEG, p->c[i]);
		return 1;
	case O_ADD:
	case O_SUB:
		if (!npoly_of(N.a[e], s, &a, maxdeg) ||
		    !npoly_of(N.b[e], s, &b, maxdeg))
			return 0;
		p->deg = a.deg > b.deg ? a.deg : b.deg;
		for (i = 0; i <= p->deg; i++) {
			nid x = i <= a.deg ? a.c[i] : KI(0);
			nid y = i <= b.deg ? b.c[i] : KI(0);
			p->c[i] = mk_bin(N.op[e], x, y);
		}
		np_trim(p);
		return 1;
	case O_MUL:
		if (!npoly_of(N.a[e], s, &a, maxdeg) ||
		    !npoly_of(N.b[e], s, &b, maxdeg))
			return 0;
		if (!np_mul(&a, &b, p, maxdeg))
			return 0;
		np_trim(p);
		return 1;
	case O_POW:
		if (!int_of(N.b[e], &k) || k < 0 || k > maxdeg ||
		    has_slot(N.b[e], s))
			return 0;
		if (!npoly_of(N.a[e], s, &a, maxdeg))
			return 0;
		p->deg = 0;
		p->c[0] = KI(1);
		for (i = 0; i < k; i++)
			if (!np_mul(p, &a, p, maxdeg))
				return 0;
		np_trim(p);
		return 1;
	case O_DIV:
		if (has_slot(N.b[e], s))
			return 0;
		if (!npoly_of(N.a[e], s, p, maxdeg))
			return 0;
		if (!TY_FP(N.ty[e])) {
			/* exact when P(i) is a multiple of c at every integer */
			int64_t c, ci[13], x, v;
			if (!int_of(N.b[e], &c) || c <= 0 || c > 1000)
				return 0;
			for (i = 0; i <= p->deg; i++)
				if (!is_const(p->c[i]) || TY_FP(N.ty[p->c[i]]) ||
				    !int_of(p->c[i], &ci[i]))
					return 0;
			for (x = 0; x < c; x++) {
				v = 0;
				for (i = p->deg; i >= 0; i--)
					v = (v * x + ci[i] % c) % c;
				if (v)
					return 0;
			}
			for (i = 0; i <= p->deg; i++)
				p->c[i] = KR((double)ci[i] / (double)c);
			return 1;
		}
		for (i = 0; i <= p->deg; i++)
			p->c[i] = DIV(p->c[i], N.b[e]);
		return 1;
	}
	return 0;
}

/* ---- rational functions P/Q in one variable ---- */

static int
np_addsub(const struct npoly *a, const struct npoly *b, int op,
    struct npoly *r)
{
	int i;
	struct npoly t;

	t.deg = a->deg > b->deg ? a->deg : b->deg;
	for (i = 0; i <= t.deg; i++)
		t.c[i] = mk_bin(op, i <= a->deg ? a->c[i] : KI(0),
		    i <= b->deg ? b->c[i] : KI(0));
	np_trim(&t);
	*r = t;
	return 1;
}

static int
ratfun_of(nid e, int s, struct npoly *P, struct npoly *Q)
{
	struct npoly a, b, c, d, x, y;
	int i;

	if (npoly_of(e, s, P, 10)) {
		Q->deg = 0;
		Q->c[0] = KI(1);
		return 1;
	}
	switch (N.op[e]) {
	case O_CAST:
		if (!TY_FP(N.ty[e]))
			return 0;
		return ratfun_of(N.a[e], s, P, Q);
	case O_NEG:
		if (!ratfun_of(N.a[e], s, P, Q))
			return 0;
		for (i = 0; i <= P->deg; i++)
			P->c[i] = mk_un(O_NEG, P->c[i]);
		return 1;
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_DIV:
		if (!TY_FP(N.ty[e]) || !ratfun_of(N.a[e], s, &a, &b) ||
		    !ratfun_of(N.b[e], s, &c, &d))
			return 0;
		if (N.op[e] == O_MUL)
			return np_mul(&a, &c, P, 10) && np_mul(&b, &d, Q, 10);
		if (N.op[e] == O_DIV)
			return np_mul(&a, &d, P, 10) && np_mul(&b, &c, Q, 10);
		if (!np_mul(&a, &d, &x, 10) || !np_mul(&c, &b, &y, 10) ||
		    !np_mul(&b, &d, Q, 10))
			return 0;
		return np_addsub(&x, &y, N.op[e], P);
	}
	return 0;
}

/* all coefficients rational numbers? */
static int
np_rational(const struct npoly *p, struct q *c)
{
	int i;
	double v;

	for (i = 0; i <= p->deg; i++)
		if (!num_of(p->c[i], &v) || !q_from_double(v, &c[i]))
			return 0;
	return 1;
}

/* x = a v + b with v a variable and a, b rational: then *v, *a, *b */
static int
lin_in_var(nid x, nid *v, struct q *a, struct q *b)
{
	struct npoly p;
	uint64_t m = node_slots(x);
	int s;

	if (m == 0 || (m & (m - 1)))
		return 0;
	for (s = 0; !((m >> s) & 1); s++)
		;
	if (!npoly_of(x, s, &p, 1) || p.deg != 1)
		return 0;
	{
		struct q c[2];
		if (!np_rational(&p, c))
			return 0;
		*a = c[1];
		*b = c[0];
	}
	{
		/* the variable node itself */
		nid y = x;
		while (N.op[y] != O_VAR) {
			if (N.op[y] == O_CAST || N.op[y] == O_NEG)
				y = N.a[y];
			else if (has_slot(N.a[y], s))
				y = N.a[y];
			else
				y = N.b[y];
		}
		*v = y;
	}
	return 1;
}

/* sum_{i=lo}^{hi} P(i) for a rational polynomial P, as a node */
static nid
qsum_node(const struct q *pc, int pdeg, nid lo, nid hi, ty_t t)
{
	struct q G[14], F[14], g0, x;
	int p, j, gdeg = pdeg + 1;
	nid v, X;
	struct q a, b;
	double lv;

	for (j = 0; j <= gdeg; j++)
		G[j] = (struct q){0, 1};
	for (p = 0; p <= pdeg; p++) {
		if (pc[p].n == 0)
			continue;
		if (!faulhaber(p, F))
			return 0;
		for (j = 0; j <= p + 1; j++) {
			struct q u;
			if (!q_mul(pc[p], F[j], &u) || !q_add(G[j], u, &G[j]))
				return 0;
		}
	}
	X = TY_FP(t) ? mk_cast(t, hi) : hi;
	if (num_of(lo, &lv) && lv == floor(lv)) {
		/* G(hi) - G(lo - 1): the second part is a number */
		g0 = (struct q){0, 1};
		x = (struct q){(int64_t)lv - 1, 1};
		for (j = gdeg; j >= 0; j--)
			if (!q_mul(g0, x, &g0) || !q_add(g0, G[j], &g0))
				return 0;
		if (!q_sub(G[0], g0, &G[0]))
			return 0;
		if (lin_in_var(hi, &v, &a, &b)) {
			struct q H[14];
			if (qpoly_compose_lin(G, gdeg, a, b, H)) {
				/* compute in the bound's type, never a narrower one */
				v = mk_cast(TY_FP(t) ? t : N.ty[hi], v);
				return qpoly_node(H, gdeg, v);
			}
		}
		return qpoly_node(G, gdeg, X);
	}
	{
		nid A = qpoly_node(G, gdeg, X), B;
		nid L = SUB(TY_FP(t) ? mk_cast(t, lo) : lo, KI(1));
		B = qpoly_node(G, gdeg, L);
		if (!A || !B)
			return 0;
		return SUB(A, B);
	}
}

/* ---- sums ---- */

static nid
ivar_in(nid e, int s)
{
	uint32_t j;
	nid r;

	if (e == 0 || !has_slot(e, s))
		return 0;
	if (N.op[e] == O_VAR)
		return e;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((r = ivar_in(N.lst[N.a[e] + j], s)) != 0)
				return r;
		return 0;
	}
	if (N.op[e] == O_LOOP && !has_slot(N.a[e], s) && !has_slot(N.b[e], s) &&
	    !has_slot(N.c[e], s))
		return 0;
	if ((r = ivar_in(N.a[e], s)) != 0)
		return r;
	if ((r = ivar_in(N.b[e], s)) != 0)
		return r;
	return ivar_in(N.c[e], s);
}

static nid
at(nid e, int s, nid x)
{
	nid map[NSLOT];

	memset(map, 0, sizeof map);
	map[s] = x;
	return subst(e, map, NSLOT);
}

/*
 * floor(x / m) for an integer node x and m > 0.  The simple forms are
 * right only for x >= 0 (mode 0) or x >= -m (mode 1); cmd_closed tries
 * them first and keeps one only if the check passes.
 */
static int cl_fdmode = 2, cl_fdused;

static nid
floordiv(nid x, int64_t m)
{
	nid r;

	cl_fdused = 1;
	if (cl_fdmode == 0)
		return DIV(x, KI(m));
	if (cl_fdmode == 1)
		return SUB(DIV(ADD(x, KI(m)), KI(m)), KI(1));
	r = mk_bin(O_MOD, x, KI(m));
	r = mk_bin(O_MOD, ADD(r, KI(m)), KI(m));
	return DIV(SUB(x, r), KI(m));
}

/* the geometric part of a product: body = K * r^(i) * P(i) */
struct gsplit {
	nid K, r, P;
	int haver;
};

static int
gsplit(nid e, int s, struct gsplit *g)
{
	struct npoly ex;
	int i;

	if (!has_slot(e, s)) {
		g->K = MUL(g->K, e);
		return 1;
	}
	switch (N.op[e]) {
	case O_MUL:
		return gsplit(N.a[e], s, g) && gsplit(N.b[e], s, g);
	case O_NEG:
		g->K = mk_un(O_NEG, g->K);
		return gsplit(N.a[e], s, g);
	case O_CAST:
		if (TY_SIZED(N.ty[e]))
			break;
		return gsplit(N.a[e], s, g);
	case O_DIV:
		if (!TY_FP(N.ty[e]))
			break;
		if (has_slot(N.b[e], s)) {
			/* dividing by a geometric factor multiplies by 1/r */
			struct gsplit d = {KI(1), KI(1), KI(1), 0};
			if (!gsplit(N.b[e], s, &d) || !d.haver || !const_is(d.P, 1))
				break;
			g->K = DIV(g->K, mk_cast(TY_REAL, d.K));
			g->r = DIV(g->r, mk_cast(TY_REAL, d.r));
			g->haver = 1;
			return gsplit(N.a[e], s, g);
		}
		g->K = DIV(g->K, N.b[e]);
		return gsplit(N.a[e], s, g);
	case O_POW:
		if (has_slot(N.a[e], s) && !has_slot(N.b[e], s))
			break;
		if (has_slot(N.a[e], s))
			return 0;
		if (!npoly_of(N.b[e], s, &ex, 1) || ex.deg != 1)
			return 0;
		/* r^(a i + b) = r^b (r^a)^i */
		g->K = MUL(g->K, POW(N.a[e], ex.c[0]));
		g->r = MUL(g->r, POW(N.a[e], ex.c[1]));
		g->haver = 1;
		return 1;
	case O_EXP:
		if (!npoly_of(N.a[e], s, &ex, 1) || ex.deg != 1)
			return 0;
		g->K = MUL(g->K, mk_un(O_EXP, ex.c[0]));
		g->r = MUL(g->r, mk_un(O_EXP, ex.c[1]));
		g->haver = 1;
		return 1;
	}
	g->P = MUL(g->P, e);
	(void)i;
	return 1;
}

nid sum_closed(nid body, int s, nid lo, nid hi, const char **how);

static nid
sum_geometric(nid body, int s, nid lo, nid hi)
{
	struct gsplit g;
	struct npoly P;
	struct q pc[14], rq, qq[14], t1;
	double rv;
	int j, l, d;
	ty_t t = N.ty[body];
	nid X, Lm1, r;

	g.K = KI(1);
	g.r = KI(1);
	g.P = KI(1);
	g.haver = 0;
	if (!gsplit(body, s, &g) || !g.haver)
		return 0;
	if (!npoly_of(g.P, s, &P, 8))
		return 0;
	r = g.r;
	X = TY_FP(t) ? mk_cast(t, hi) : hi;
	Lm1 = SUB(TY_FP(t) ? mk_cast(t, lo) : lo, KI(1));
	if (num_of(r, &rv) && rv == 1)
		return MUL(g.K, sum_closed(g.P, s, lo, hi, NULL));
	d = P.deg;
	if (num_of(r, &rv) && q_from_double(rv, &rq) && np_rational(&P, pc)) {
		/* r Q(i) - Q(i-1) = P(i), T(i) = r^(i+1) Q(i) */
		struct q rm1;
		int64_t D = 1;
		struct q Qi[14];
		if (!q_sub(rq, (struct q){1, 1}, &rm1))
			return 0;
		for (j = d; j >= 0; j--) {
			struct q acc = pc[j];
			for (l = j + 1; l <= d; l++) {
				int64_t bc = 1;
				int z;
				for (z = 1; z <= j; z++)
					bc = bc * (l - j + z) / z;
				if (((l - j) & 1))
					bc = -bc;
				if (!q_mul(qq[l], (struct q){bc, 1}, &t1) ||
				    !q_add(acc, t1, &acc))
					return 0;
			}
			if (!q_div(acc, rm1, &qq[j]))
				return 0;
		}
		for (j = 0; j <= d; j++) {
			int64_t g0 = gcd64(D, qq[j].d);
			if (__builtin_mul_overflow(D / g0, qq[j].d, &D))
				return 0;
		}
		for (j = 0; j <= d; j++)
			if (!q_mul(qq[j], (struct q){D, 1}, &Qi[j]))
				return 0;
		METH(d ? M_AGEO : M_GEOM);
		{
			nid rn = TY_FP(t) && !TY_FP(N.ty[r]) ? mk_cast(t, r) : r;
			nid a = MUL(POW(rn, ADD(X, KI(1))), qpoly_node(Qi, d, X));
			nid b = MUL(POW(rn, ADD(Lm1, KI(1))), qpoly_node(Qi, d, Lm1));
			nid v = SUB(a, b);
			if (D != 1)
				v = DIV(v, KI(D));
			return MUL(g.K, v);
		}
	}
	if (d > 0 && TY_FP(t)) {
		/* symbolic ratio: the same recurrence for Q with nodes */
		nid qn[14], Q1, Q0, rm1 = SUB(r, KI(1)), v, ps;
		for (j = d; j >= 0; j--) {
			nid acc = P.c[j];
			for (l = j + 1; l <= d; l++) {
				int64_t bc = 1;
				int z;
				for (z = 1; z <= j; z++)
					bc = bc * (l - j + z) / z;
				if ((l - j) & 1)
					bc = -bc;
				acc = ADD(acc, MUL(KI(bc), qn[l]));
			}
			qn[j] = DIV(acc, rm1);
		}
		Q1 = qn[d];
		Q0 = qn[d];
		for (j = d - 1; j >= 0; j--) {
			Q1 = ADD(MUL(Q1, X), qn[j]);
			Q0 = ADD(MUL(Q0, Lm1), qn[j]);
		}
		if (const_is(Lm1, 0))
			Q0 = qn[0];
		v = SUB(MUL(POW(r, ADD(X, KI(1))), Q1), MUL(POW(r, ADD(Lm1, KI(1))),
		    Q0));
		ps = sum_closed(g.P, s, lo, hi, NULL);
		if (!ps)
			return 0;
		METH(M_AGEO);
		return MUL(g.K, mk_sel(mk_bin(O_EQ, r, KI(1)), ps, v));
	}
	if (d == 0) {
		/* symbolic ratio: guard r == 1 */
		nid c = P.c[0], v;
		METH(M_GEOM);
		v = DIV(SUB(POW(r, ADD(X, KI(1))), POW(r, ADD(Lm1, KI(1)))),
		    SUB(r, KI(1)));
		return MUL(g.K, MUL(c, mk_sel(mk_bin(O_EQ, r, KI(1)),
		    ADD(SUB(X, Lm1), KI(0)), v)));
	}
	return 0;
}

/* g(i) - g(i + k) */
static nid
sum_telescoping(nid body, int s, nid lo, nid hi)
{
	nid iv = ivar_in(body, s), A, B, r = 0;
	int k, sign;

	if (N.op[body] != O_SUB || iv == 0)
		return 0;
	A = N.a[body];
	B = N.b[body];
	for (sign = 1; sign >= -1; sign -= 2) {
		for (k = 1; k <= 4; k++) {
			nid sh = at(sign > 0 ? A : B, s, ADD(iv, KI(k))), acc = 0;
			int j;
			if (sh != (sign > 0 ? B : A))
				continue;
			/* sum (g(i) - g(i+k)) = sum_j g(lo+j) - g(hi+1+j) */
			for (j = 0; j < k; j++) {
				nid g = sign > 0 ? A : B;
				nid term = SUB(at(g, s, ADD(lo, KI(j))),
				    at(g, s, ADD(hi, KI(j + 1))));
				acc = acc ? ADD(acc, term) : term;
			}
			r = sign > 0 ? acc : mk_un(O_NEG, acc);
			METH(M_TELE);
			return r;
		}
	}
	return 0;
}

/* P(i)/Q(i) with Q = lead * prod (i + a_j), distinct integers a_j */
static nid
sum_pfrac(nid body, int s, nid lo, nid hi)
{
	struct npoly Pn, Qn;
	double qc[14], A[14];
	struct root rt[14];
	int64_t off[14], a0;
	int nr, j, t, m;
	double sumA = 0, scale = 0;
	nid acc, X, Lo;

	if (!TY_FP(N.ty[body]) || !ratfun_of(body, s, &Pn, &Qn))
		return 0;
	if (Qn.deg < 1 || Pn.deg >= Qn.deg)
		return 0;
	for (j = 0; j <= Qn.deg; j++)
		if (!num_of(Qn.c[j], &qc[j]))
			return 0;
	for (j = 0; j <= Pn.deg; j++)
		if (!num_of(Pn.c[j], &A[j]))
			return 0;
	nr = poly_roots(qc, Qn.deg, rt);
	m = 0;
	for (j = 0; j < nr; j++) {
		if (!rt[j].exact || rt[j].im != 0 || rt[j].mult != 1 ||
		    rt[j].qr.d != 1)
			return 0;
		off[m++] = -rt[j].qr.n;
	}
	if (m != Qn.deg)
		return 0;
	/* A_j = P(-a_j) / Q'(-a_j) */
	{
		double res[14];
		for (j = 0; j < m; j++) {
			double x = (double)-off[j], pv = 0, dq = 0;
			for (t = Pn.deg; t >= 0; t--)
				pv = pv * x + A[t];
			for (t = Qn.deg; t >= 1; t--)
				dq = dq * x + t * qc[t];
			res[j] = pv / dq;
			sumA += res[j];
			scale += fabs(res[j]);
		}
		if (fabs(sumA) > 1e-12 * scale) {
			sb_printf(&cl_extra, "  %s has a harmonic part (residues do not "
			    "cancel): no elementary closed form; see 'continuous'\n",
			    node_str(body));
			return 0;
		}
		a0 = off[0];
		for (j = 1; j < m; j++)
			if (off[j] < a0)
				a0 = off[j];
		X = mk_cast(N.ty[body], hi);
		Lo = mk_cast(N.ty[body], lo);
		acc = 0;
		for (j = 0; j < m; j++) {
			nid c;
			struct q cq;
			if (res[j] == 0)
				continue;
			c = q_from_double(res[j], &cq) ? mk_cast(N.ty[body],
			    q_node(cq)) : KR(res[j]);
			for (t = 1; t <= off[j] - a0; t++) {
				nid term = DIV(c, ADD(X, KI(a0 + t)));
				acc = acc ? ADD(acc, term) : term;
			}
			for (t = 0; t < off[j] - a0; t++) {
				nid term = DIV(c, ADD(Lo, KI(a0 + t)));
				acc = acc ? SUB(acc, term) : mk_un(O_NEG, term);
			}
		}
	}
	METH(M_PFRAC);
	METH(M_TELE);
	return acc ? acc : KI(0);
}

/* a modulus i % m (m constant) inside a condition */
static nid
find_mod(nid e, int s)
{
	nid r;
	int64_t m;

	if (e == 0 || !has_slot(e, s) || N.op[e] == O_LOOP)
		return 0;
	if (N.op[e] == O_MOD && int_of(N.b[e], &m) && m >= 2 && m <= 16 &&
	    ivar_in(N.a[e], s) && npoly_of(N.a[e], s, &(struct npoly){0}, 1))
		return e;
	if ((r = find_mod(N.a[e], s)) != 0)
		return r;
	if ((r = find_mod(N.b[e], s)) != 0)
		return r;
	return find_mod(N.c[e], s);
}

static nid
sum_split_mod(nid body, int s, nid lo, nid hi)
{
	nid md = find_mod(body, s), iv, acc = 0;
	int64_t m, r;
	struct npoly ip;

	if (md == 0)
		return 0;
	int_of(N.b[md], &m);
	/* only i % m with i the bare index */
	if (!npoly_of(N.a[md], s, &ip, 1) || ip.deg != 1 ||
	    !const_is(ip.c[1], 1) || !const_is(ip.c[0], 0))
		return 0;
	iv = ivar_in(N.a[md], s);
	if (TY_FP(N.ty[iv]))
		return 0;
	for (r = 0; r < m; r++) {
		nid br = xreplace(body, md, mk_const_i(N.ty[md], r)), sub;
		nid j0, j1;
		br = at(br, s, ADD(MUL(KI(m), iv), KI(r)));
		if (has_slot(br, s) && find_mod(br, s))
			return 0;
		j0 = floordiv(SUB(ADD(lo, KI(m - 1)), KI(r)), m);
		j1 = floordiv(SUB(hi, KI(r)), m);
		sub = sum_closed(br, s, j0, j1, NULL);
		if (!sub)
			return 0;
		acc = acc ? ADD(acc, sub) : sub;
	}
	METH(M_SPLITMOD);
	return acc;
}

static nid
sum_split_lt(nid body, int s, nid lo, nid hi)
{
	nid c, x, y, A, B, t, h1, s1, s2;
	int op;
	int64_t k;

	if (N.op[body] != O_SEL)
		return 0;
	c = N.a[body];
	A = N.b[body];
	B = N.c[body];
	op = N.op[c];
	if (!(opinfo[op].fl & OF_CMP) || op == O_EQ || op == O_NE)
		return 0;
	x = N.a[c];
	y = N.b[c];
	if (has_slot(y, s) && !has_slot(x, s)) {
		nid z = x;
		x = y;
		y = z;
		op = op == O_LT ? O_GT : op == O_LE ? O_GE : op == O_GT ? O_LT :
		    O_LE;
	}
	if (has_slot(y, s) || N.op[x] != O_VAR || N.ki[x] != s ||
	    TY_FP(N.ty[x]) || TY_FP(N.ty[y]))
		return 0;
	/* the A part holds for i <= t (LT/LE) or the B part does (GT/GE) */
	k = op == O_LT || op == O_GE ? 1 : 0;
	t = k ? SUB(y, KI(1)) : y;
	if (op == O_GT || op == O_GE) {
		nid z = A;
		A = B;
		B = z;
	}
	h1 = mk_bin(O_MAX, SUB(lo, KI(1)), mk_bin(O_MIN, hi, mk_cast(TY_INT, t)));
	s1 = sum_closed(A, s, lo, h1, NULL);
	s2 = s1 ? sum_closed(B, s, ADD(h1, KI(1)), hi, NULL) : 0;
	if (!s1 || !s2)
		return 0;
	METH(M_SPLITLT);
	return ADD(s1, s2);
}

nid
sum_closed(nid body, int s, nid lo, nid hi, const char **how)
{
	struct npoly P;
	struct q pc[14];
	nid r = 0;
	ty_t t = N.ty[body];

	if (cl_depth > 12)
		return 0;
	cl_depth++;
	if (!has_slot(body, s)) {
		METH(M_FREE);
		r = MUL(ADD(SUB(hi, lo), KI(1)), body);
		goto out;
	}
	if ((r = sum_telescoping(body, s, lo, hi)) != 0)
		goto out;
	if (npoly_of(body, s, &P, 11)) {
		int j;
		if (np_rational(&P, pc)) {
			r = qsum_node(pc, P.deg, lo, hi, t);
		} else {
			/* symbolic coefficients: one Faulhaber sum per power */
			for (j = 0; j <= P.deg; j++) {
				struct q e[14];
				int z;
				nid term;
				if (const_is(P.c[j], 0))
					continue;
				for (z = 0; z <= j; z++)
					e[z] = (struct q){z == j, 1};
				term = qsum_node(e, j, lo, hi, t);
				if (!term) {
					r = 0;
					break;
				}
				term = MUL(P.c[j], term);
				r = r ? ADD(r, term) : term;
			}
		}
		if (r) {
			METH(M_FAUL);
			goto out;
		}
	}
	if ((r = sum_geometric(body, s, lo, hi)) != 0)
		goto out;
	if ((r = sum_pfrac(body, s, lo, hi)) != 0)
		goto out;
	switch (N.op[body]) {
	case O_ADD:
	case O_SUB: {
		nid a = sum_closed(N.a[body], s, lo, hi, NULL), b;
		b = a ? sum_closed(N.b[body], s, lo, hi, NULL) : 0;
		if (a && b)
			r = mk_bin(N.op[body], a, b);
		break;
	}
	case O_NEG: {
		nid a = sum_closed(N.a[body], s, lo, hi, NULL);
		if (a)
			r = mk_un(O_NEG, a);
		break;
	}
	case O_MUL:
		if (!has_slot(N.a[body], s)) {
			nid a = sum_closed(N.b[body], s, lo, hi, NULL);
			if (a)
				r = MUL(N.a[body], a);
		} else if (!has_slot(N.b[body], s)) {
			nid a = sum_closed(N.a[body], s, lo, hi, NULL);
			if (a)
				r = MUL(a, N.b[body]);
		}
		break;
	case O_DIV:
		if (!has_slot(N.b[body], s) && TY_FP(t)) {
			nid a = sum_closed(N.a[body], s, lo, hi, NULL);
			if (a)
				r = DIV(a, N.b[body]);
		}
		break;
	case O_CAST:
		if (TY_FP(t) || TYK(t) == K_INT) {
			nid a = sum_closed(N.a[body], s, lo, hi, NULL);
			if (a)
				r = mk_cast(t, a);
		}
		break;
	}
	if (r)
		goto out;
	if ((r = sum_split_lt(body, s, lo, hi)) != 0)
		goto out;
	r = sum_split_mod(body, s, lo, hi);
out:
	cl_depth--;
	if (r && how)
		*how = "";
	if (r && N.ty[r] != t && !(TY_FP(N.ty[r]) && !TY_FP(t)))
		r = mk_cast(t, r);
	return r;
}

/* ---- products ---- */

static nid
prod_rational(nid body, int s, nid lo, nid hi)
{
	struct npoly Pn, Qn;
	double c[2][14];
	int64_t off[2][14];
	int cnt[2], w, j, nr;
	struct root rt[14];
	nid acc = KI(1), X, Lo, lead;
	ty_t t = N.ty[body];

	if (!TY_FP(t) || !ratfun_of(body, s, &Pn, &Qn))
		return 0;
	if (Pn.deg != Qn.deg || Pn.deg < 1)
		return 0;
	for (w = 0; w < 2; w++) {
		struct npoly *pp = w ? &Qn : &Pn;
		for (j = 0; j <= pp->deg; j++)
			if (!num_of(pp->c[j], &c[w][j]))
				return 0;
		nr = poly_roots(c[w], pp->deg, rt);
		cnt[w] = 0;
		for (j = 0; j < nr; j++) {
			int z;
			if (!rt[j].exact || rt[j].im != 0 || rt[j].qr.d != 1)
				return 0;
			for (z = 0; z < rt[j].mult; z++)
				off[w][cnt[w]++] = -rt[j].qr.n;
		}
		if (cnt[w] != pp->deg)
			return 0;
	}
	/* sort both offset lists, pair them in order */
	for (w = 0; w < 2; w++) {
		int a, b;
		for (a = 0; a < cnt[w]; a++)
			for (b = a + 1; b < cnt[w]; b++)
				if (off[w][b] < off[w][a]) {
					int64_t z = off[w][a];
					off[w][a] = off[w][b];
					off[w][b] = z;
				}
	}
	X = mk_cast(t, hi);
	Lo = mk_cast(t, lo);
	for (j = 0; j < cnt[0]; j++) {
		int64_t a = off[0][j], b = off[1][j], m;
		if (a > b) {
			for (m = b + 1; m <= a; m++)
				acc = MUL(acc, ADD(X, KI(m)));
			for (m = b; m <= a - 1; m++)
				acc = DIV(acc, ADD(Lo, KI(m)));
		} else if (a < b) {
			for (m = a + 1; m <= b; m++)
				acc = DIV(acc, ADD(X, KI(m)));
			for (m = a; m <= b - 1; m++)
				acc = MUL(acc, ADD(Lo, KI(m)));
		}
	}
	lead = DIV(KR(c[0][Pn.deg]), KR(c[1][Qn.deg]));
	if (!const_is(lead, 1))
		acc = MUL(POW(lead, mk_cast(t, ADD(SUB(hi, lo), KI(1)))), acc);
	METH(M_PRODRAT);
	return acc;
}

nid
prod_closed(nid body, int s, nid lo, nid hi, const char **how)
{
	nid r = 0, cnt;
	ty_t t = N.ty[body];

	if (cl_depth > 12)
		return 0;
	cl_depth++;
	cnt = ADD(SUB(hi, lo), KI(1));
	if (!has_slot(body, s)) {
		METH(M_FREE);
		r = POW(body, TY_FP(N.ty[body]) ? mk_cast(N.ty[body], cnt) : cnt);
		goto out;
	}
	switch (N.op[body]) {
	case O_MUL: {
		nid a = prod_closed(N.a[body], s, lo, hi, NULL), b;
		b = a ? prod_closed(N.b[body], s, lo, hi, NULL) : 0;
		if (a && b)
			r = MUL(a, b);
		break;
	}
	case O_NEG: {
		nid a = prod_closed(N.a[body], s, lo, hi, NULL);
		if (a)
			r = MUL(POW(KI(-1), cnt), a);
		break;
	}
	case O_POW:
		if (!has_slot(N.a[body], s)) {
			nid e = sum_closed(N.b[body], s, lo, hi, NULL);
			if (e) {
				r = POW(N.a[body], e);
				METH(M_PRODPOW);
			}
		}
		break;
	case O_EXP: {
		nid e = sum_closed(N.a[body], s, lo, hi, NULL);
		if (e) {
			r = mk_un(O_EXP, e);
			METH(M_PRODPOW);
		}
		break;
	}
	case O_DIV: {
		/* g(i+k)/g(i) */
		nid A = N.a[body], B = N.b[body], iv = ivar_in(body, s);
		int k, j;
		for (k = 1; k <= 4 && iv && TY_FP(t); k++) {
			if (at(B, s, ADD(iv, KI(k))) != A)
				continue;
			r = KI(1);
			for (j = 0; j < k; j++)
				r = MUL(r, DIV(at(B, s, ADD(hi, KI(j + 1))),
				    at(B, s, ADD(lo, KI(j)))));
			METH(M_PRODTELE);
			break;
		}
		if (!r)
			r = prod_rational(body, s, lo, hi);
		if (!r && !has_slot(B, s)) {
			nid a = prod_closed(A, s, lo, hi, NULL);
			if (a)
				r = DIV(a, POW(B, mk_cast(t, cnt)));
		}
		break;
	}
	}
	if (!r)
		r = prod_rational(body, s, lo, hi);
	if (!r) {
		struct npoly P;
		if (npoly_of(body, s, &P, 8) && P.deg >= 1)
			sb_printf(&cl_extra, "  prod of %s: a factorial-like product "
			    "(Gamma function); no elementary closed form; see "
			    "'continuous' for Stirling's approximation\n",
			    node_str(body));
	}
out:
	cl_depth--;
	if (r && how)
		*how = "";
	return r;
}

/* ---- loops: linear systems with constant coefficients ---- */

/* e = sum_l M[l] * var(slot l) + rest, M numeric; rest reads no state */
int
linear_form(nid e, const int *slot, int m, double *M, nid *rest)
{
	double ma[MAXST], mb[MAXST], k;
	nid ra, rb;
	int l;

	for (l = 0; l < m; l++)
		M[l] = 0;
	*rest = KI(0);
	{
		uint64_t sm = 0;
		for (l = 0; l < m; l++)
			sm |= 1ULL << slot[l];
		if (!(node_slots(e) & sm)) {
			*rest = e;
			return 1;
		}
	}
	switch (N.op[e]) {
	case O_VAR:
		for (l = 0; l < m; l++)
			if (N.ki[e] == slot[l]) {
				M[l] = 1;
				return 1;
			}
		return 0;
	case O_CAST:
		if (!TY_FP(N.ty[e]) && TYK(N.ty[e]) != K_INT &&
		    TYK(N.ty[N.a[e]]) != TYK(N.ty[e]))
			return 0;
		if (!linear_form(N.a[e], slot, m, M, rest))
			return 0;
		*rest = mk_cast(N.ty[e], *rest);
		return 1;
	case O_NEG:
		if (!linear_form(N.a[e], slot, m, M, rest))
			return 0;
		for (l = 0; l < m; l++)
			M[l] = -M[l];
		*rest = mk_un(O_NEG, *rest);
		return 1;
	case O_ADD:
	case O_SUB:
		if (!linear_form(N.a[e], slot, m, ma, &ra) ||
		    !linear_form(N.b[e], slot, m, mb, &rb))
			return 0;
		for (l = 0; l < m; l++)
			M[l] = N.op[e] == O_ADD ? ma[l] + mb[l] : ma[l] - mb[l];
		*rest = mk_bin(N.op[e], ra, rb);
		return 1;
	case O_MUL:
		if (num_of(N.a[e], &k)) {
			if (!linear_form(N.b[e], slot, m, M, rest))
				return 0;
		} else if (num_of(N.b[e], &k)) {
			if (!linear_form(N.a[e], slot, m, M, rest))
				return 0;
		} else
			return 0;
		for (l = 0; l < m; l++)
			M[l] *= k;
		*rest = MUL(KR(k), *rest);
		if (k == floor(k) && fabs(k) < 1e15)
			*rest = MUL(KI((int64_t)k), N.op[*rest] == O_MUL ?
			    N.b[*rest] : *rest);
		return 1;
	case O_DIV:
		if (!TY_FP(N.ty[e]) || !num_of(N.b[e], &k) || k == 0)
			return 0;
		if (!linear_form(N.a[e], slot, m, M, rest))
			return 0;
		for (l = 0; l < m; l++)
			M[l] /= k;
		*rest = DIV(*rest, N.b[e]);
		return 1;
	}
	return 0;
}

void
charpoly(const double *M, int n, double *c)
{
	double A[256], B[256];
	int i, j, l, k;

	/* Faddeev-LeVerrier: c[n] = 1 */
	memset(c, 0, (size_t)(n + 1) * sizeof *c);
	c[n] = 1;
	memcpy(A, M, (size_t)n * n * sizeof *A);
	for (k = 1; k <= n; k++) {
		double tr = 0;
		for (i = 0; i < n; i++)
			tr += A[i * n + i];
		c[n - k] = -tr / k;
		if (k == n)
			break;
		for (i = 0; i < n; i++)
			A[i * n + i] += c[n - k];
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++) {
				double s = 0;
				for (l = 0; l < n; l++)
					s += M[i * n + l] * A[l * n + j];
				B[i * n + j] = s;
			}
		memcpy(A, B, (size_t)n * n * sizeof *A);
	}
	for (i = 0; i <= n; i++)
		if (fabs(c[i] - floor(c[i] + 0.5)) < 1e-9 * (1 + fabs(c[i])))
			c[i] = floor(c[i] + 0.5);
}

struct rbasis {
	int nb;
	double re[16], im[16];
	int pw[16], kind[16];      /* kind: 0 real, 1 cos, 2 sin, 3 zero root */
	struct root *rt[16];
};

static double
basis_at(const struct rbasis *B, int b, int k)
{
	double kp = pow((double)k, B->pw[b]);

	if (B->pw[b] == 0)
		kp = 1;
	switch (B->kind[b]) {
	case 3:
		return k == B->pw[b] ? 1 : 0;
	case 0:
		return kp * pow(B->re[b], k);
	default: {
		double rho = hypot(B->re[b], B->im[b]), th = atan2(B->im[b],
		    B->re[b]);
		return kp * pow(rho, k) * (B->kind[b] == 1 ? cos(k * th) :
		    sin(k * th));
	}
	}
}

static int
basis_q(const struct rbasis *B, int b, int k, struct q *r)
{
	struct q x = {1, 1};
	int i;

	if (B->kind[b] == 3) {
		*r = (struct q){k == B->pw[b], 1};
		return 1;
	}
	if (B->kind[b] != 0 || !B->rt[b]->exact)
		return 0;
	for (i = 0; i < k; i++)
		if (!q_mul(x, B->rt[b]->qr, &x))
			return 0;
	for (i = 0; i < B->pw[b]; i++)
		if (!q_mul(x, (struct q){k, 1}, &x))
			return 0;
	*r = x;
	return 1;
}

static nid
root_node(const struct root *r)
{
	if (r->exact)
		return q_node(r->qr);
	if (r->quad && r->im == 0) {
		nid s = mk_un(O_SQRT, mk_cast(TY_REAL, q_node(r->qd)));
		return DIV(r->quad > 0 ? ADD(q_node(r->qb), s) : SUB(q_node(r->qb),
		    s), mk_cast(TY_REAL, q_node(r->qa)));
	}
	return KR(r->re);
}

static nid
basis_node(const struct rbasis *B, int b, nid K, int exact, ty_t st)
{
	nid kp = B->pw[b] ? (B->pw[b] == 1 ? K : POW(K, KI(B->pw[b]))) : 0, v;

	switch (B->kind[b]) {
	case 3:
		return mk_sel(mk_bin(O_EQ, K, KI(B->pw[b])), KI(1), KI(0));
	case 0: {
		nid r = root_node(B->rt[b]);
		if (!exact)
			r = mk_cast(TY_REAL, r);
		else if (!TY_FP(st) && B->rt[b]->qr.d == 1)
			r = mk_cast(TY_INT, r);
		if (const_is(r, 1))
			v = 0;
		else
			v = POW(r, exact ? K : mk_cast(TY_REAL, K));
		break;
	}
	default: {
		double rho = hypot(B->re[b], B->im[b]), th = atan2(B->im[b],
		    B->re[b]);
		struct q rq;
		nid rn = q_from_double(rho, &rq) ? mk_cast(TY_REAL, q_node(rq)) :
		    KR(rho);
		nid arg, Kr = mk_cast(TY_REAL, K);
		struct q tq;
		/* angles that are simple fractions of pi print as such */
		if (q_from_double(th / M_PI, &tq) && tq.d <= 12 &&
		    fabs(q_val(tq) * M_PI - th) < 1e-12) {
			arg = MUL(KR(M_PI), Kr);
			if (tq.n != 1)
				arg = MUL(KI(tq.n), arg);
			if (tq.d != 1)
				arg = DIV(arg, KI(tq.d));
		} else
			arg = MUL(KR(th), Kr);
		v = mk_un(B->kind[b] == 1 ? O_COS : O_SIN, arg);
		if (!const_is(rn, 1))
			v = MUL(POW(rn, mk_cast(TY_REAL, K)), v);
		break;
	}
	}
	if (kp && v)
		return MUL(kp, v);
	return kp ? kp : v ? v : KI(1);
}

static void
print_poly(struct sbuf *b, const double *c, int n)
{
	int i, first = 1;

	for (i = n; i >= 0; i--) {
		double a = c[i];
		if (a == 0)
			continue;
		if (!first)
			sb_puts(b, a < 0 ? " - " : " + ");
		else if (a < 0)
			sb_puts(b, "-");
		a = fabs(a);
		if (a != 1 || i == 0)
			sb_printf(b, "%.10g", a);
		if (i > 0)
			sb_printf(b, "%s", "x");
		if (i > 1)
			sb_printf(b, "^%d", i);
		first = 0;
	}
}

static void
print_root(struct sbuf *b, const struct root *r)
{
	if (r->exact) {
		if (r->qr.d == 1)
			sb_printf(b, "%lld", (long long)r->qr.n);
		else
			sb_printf(b, "%lld/%lld", (long long)r->qr.n,
			    (long long)r->qr.d);
	} else if (r->quad && r->im == 0)
		sb_printf(b, "(%lld %c √%lld)/%lld = %.10g", (long long)r->qb.n,
		    r->quad > 0 ? '+' : '-', (long long)r->qd.n,
		    (long long)r->qa.n, r->re);
	else if (r->im != 0)
		sb_printf(b, "%.10g ± %.10gi", r->re + 0.0, r->im);
	else
		sb_printf(b, "%.10g", r->re);
	if (r->mult == 2)
		sb_puts(b, " (double)");
	else if (r->mult > 2)
		sb_printf(b, " (x%d)", r->mult);
}

/*
 * The loop is s' = M s + b with constant M; extra states carry symbolic
 * constants and 1.  Each state is sum_l init_l F_l(k), F_l a fixed
 * combination of k^p r^k terms for the roots r of the characteristic
 * polynomial.
 */
struct linsys {
	int n, m;              /* n: all states, m: the loop's own */
	double M[256];
	nid init[16];
	int res;
};

static int
linsys_of(const struct loopdef *d, struct linsys *S)
{
	nid rest[MAXST], extra[8];
	int j, l, n, ne = 0, unit = -1;
	double row[MAXST];

	memset(S, 0, sizeof *S);
	S->m = d->nst;
	for (j = 0; j < d->nst; j++) {
		if (!linear_form(d->next[j], d->sslot, d->nst, row, &rest[j]))
			return 0;
		if (d->kind == LK_FOR && has_slot(rest[j], d->islot))
			return 0;
		for (l = 0; l < d->nst; l++)
			S->M[j * 16 + l] = row[l];
	}
	/* constants: numbers through a unit state, others as their own */
	n = d->nst;
	for (j = 0; j < d->nst; j++) {
		double v;
		if (const_is(rest[j], 0))
			continue;
		if (num_of(rest[j], &v)) {
			if (unit < 0) {
				if (n == 16)
					return 0;
				unit = n++;
				S->init[unit] = KI(1);
				S->M[unit * 16 + unit] = 1;
			}
			S->M[j * 16 + unit] = v;
		} else {
			int z;
			for (z = 0; z < ne; z++)
				if (extra[z] == rest[j])
					break;
			if (z == ne) {
				if (n == 16 || ne == 8)
					return 0;
				extra[ne++] = rest[j];
				S->init[n] = rest[j];
				S->M[n * 16 + n] = 1;
				n++;
			}
			for (l = d->nst; l < n; l++)
				if (S->init[l] == rest[j])
					S->M[j * 16 + l] = 1;
		}
	}
	for (j = 0; j < d->nst; j++)
		S->init[j] = d->init[j];
	S->n = n;
	return 1;
}

static nid
linsys_closed(const struct loopdef *d, int res, nid K, int *isexact)
{
	struct linsys S;
	double M[256], cp[17], y[16][16], W[256], rhs[16];
	struct root rt[16];
	struct rbasis B;
	int n, nr, j, l, k, b, exact;
	nid acc = 0;
	ty_t st = d->sty[res];
	double coef[16][16];
	struct q qcoef[16][16];

	if (!linsys_of(d, &S))
		return 0;
	n = S.n;
	if (n > 12)
		return 0;
	for (j = 0; j < n; j++)
		for (l = 0; l < n; l++)
			M[j * n + l] = S.M[j * 16 + l];
	charpoly(M, n, cp);
	nr = poly_roots(cp, n, rt);
	memset(&B, 0, sizeof B);
	exact = 1;
	for (j = 0; j < nr; j++) {
		int p;
		for (p = 0; p < rt[j].mult; p++) {
			if (B.nb + (rt[j].im != 0 ? 2 : 1) > 16)
				return 0;
			if (rt[j].re == 0 && rt[j].im == 0) {
				B.kind[B.nb] = 3;
				B.pw[B.nb] = p;
				B.rt[B.nb++] = &rt[j];
				continue;
			}
			B.re[B.nb] = rt[j].re;
			B.im[B.nb] = rt[j].im;
			B.pw[B.nb] = p;
			B.rt[B.nb] = &rt[j];
			B.kind[B.nb++] = rt[j].im != 0 ? 1 : 0;
			if (rt[j].im != 0) {
				B.re[B.nb] = rt[j].re;
				B.im[B.nb] = rt[j].im;
				B.pw[B.nb] = p;
				B.rt[B.nb] = &rt[j];
				B.kind[B.nb++] = 2;
			}
			if (!rt[j].exact)
				exact = 0;
		}
	}
	if (B.nb != n)
		return 0;
	for (j = 0; j < n; j++)
		for (l = 0; l < n; l++)
			if (M[j * n + l] != floor(M[j * n + l]))
				exact = 0;
	/* y[l][k] = (M^k e_l)[res] for k < n */
	for (l = 0; l < n; l++) {
		double v[16], w[16];
		for (j = 0; j < n; j++)
			v[j] = j == l;
		for (k = 0; k < n; k++) {
			y[l][k] = v[res];
			for (j = 0; j < n; j++) {
				double s = 0;
				int z;
				for (z = 0; z < n; z++)
					s += M[j * n + z] * v[z];
				w[j] = s;
			}
			memcpy(v, w, sizeof v);
		}
	}
	for (l = 0; l < n; l++) {
		int nz = 0;
		for (k = 0; k < n; k++)
			nz |= y[l][k] != 0;
		if (!nz) {
			for (b = 0; b < n; b++) {
				coef[l][b] = 0;
				qcoef[l][b] = (struct q){0, 1};
			}
			continue;
		}
		if (exact) {
			struct q Wq[256], bq[16];
			for (k = 0; k < n; k++) {
				for (b = 0; b < n; b++)
					if (!basis_q(&B, b, k, &Wq[k * n + b]))
						exact = 0;
				bq[k] = (struct q){(int64_t)y[l][k], 1};
			}
			if (exact && solve_q(Wq, bq, n)) {
				for (b = 0; b < n; b++) {
					qcoef[l][b] = bq[b];
					coef[l][b] = q_val(bq[b]);
				}
				continue;
			}
			exact = 0;
		}
		for (k = 0; k < n; k++) {
			for (b = 0; b < n; b++)
				W[k * n + b] = basis_at(&B, b, k);
			rhs[k] = y[l][k];
		}
		if (!solve_lin(W, rhs, n))
			return 0;
		for (b = 0; b < n; b++)
			coef[l][b] = rhs[b];
	}
	/* report */
	sb_puts(&cl_extra, "  characteristic polynomial ");
	print_poly(&cl_extra, cp, n);
	sb_puts(&cl_extra, ", roots ");
	for (j = 0; j < nr; j++) {
		if (j)
			sb_puts(&cl_extra, ", ");
		print_root(&cl_extra, &rt[j]);
	}
	sb_puts(&cl_extra, "\n");
	/* combine: numbers fold together, symbols stay as factors */
	{
		double num[16] = {0};
		struct q nq[16];
		int allnum = 1, qok = exact;
		for (b = 0; b < n; b++)
			nq[b] = (struct q){0, 1};
		for (l = 0; l < n; l++) {
			double v;
			int64_t iv;
			if (!num_of(S.init[l], &v)) {
				allnum = 0;
				continue;
			}
			if (!int_of(S.init[l], &iv))
				qok = 0;
			for (b = 0; b < n; b++) {
				num[b] += v * coef[l][b];
				if (qok) {
					struct q t;
					if (!q_mul(qcoef[l][b], (struct q){iv, 1}, &t) ||
					    !q_add(nq[b], t, &nq[b]))
						qok = 0;
				}
			}
		}
		/* the numeric part */
		{
			double scale = 0;
			int64_t D = 1;
			nid part = 0;
			for (b = 0; b < n; b++)
				scale = fmax(scale, fabs(num[b]));
			if (qok)
				for (b = 0; b < n; b++) {
					int64_t g = gcd64(D, nq[b].d);
					if (__builtin_mul_overflow(D / g, nq[b].d, &D))
						qok = 0;
				}
			for (b = 0; b < n; b++) {
				nid term, c;
				if (qok) {
					struct q t;
					if (nq[b].n == 0)
						continue;
					q_mul(nq[b], (struct q){D, 1}, &t);
					c = KI(t.n);
				} else {
					struct q t;
					if (fabs(num[b]) <= 1e-13 * scale)
						continue;
					c = q_from_double(num[b], &t) && t.d <= 1000 ?
					    mk_cast(TY_REAL, q_node(t)) : KR(num[b]);
				}
				term = basis_node(&B, b, K, qok, st);
				term = const_is(c, 1) ? term : MUL(c, term);
				part = part ? ADD(part, term) : term;
			}
			if (part && qok && D != 1)
				part = DIV(part, KI(D));
			acc = part;
		}
		if (!allnum) {
			for (l = 0; l < n; l++) {
				double v;
				nid F = 0;
				if (num_of(S.init[l], &v))
					continue;
				for (b = 0; b < n; b++) {
					nid term, c;
					struct q t;
					if (fabs(coef[l][b]) < 1e-13)
						continue;
					if (exact)
						c = q_node(qcoef[l][b]);
					else
						c = q_from_double(coef[l][b], &t) &&
						    t.d <= 1000 ? q_node(t) : KR(coef[l][b]);
					term = basis_node(&B, b, K, exact, st);
					term = const_is(c, 1) ? term : MUL(c, term);
					F = F ? ADD(F, term) : term;
				}
				if (F) {
					F = MUL(S.init[l], F);
					acc = acc ? ADD(acc, F) : F;
				}
			}
			qok = 0;
		}
		if (!acc)
			acc = KI(0);
		*isexact = qok || (exact && !allnum);
	}
	if (!TY_FP(st) && TY_FP(N.ty[acc]))
		acc = mk_cast(st, mk_un(O_FLOOR, ADD(acc, KR(0.5))));
	else
		acc = mk_cast(st, acc);
	METH(M_LINREC);
	return acc;
}

int
lin_rec_matrix(nid n, double *m, double *v0, int *dim, int *res)
{
	struct loopdef d;
	struct linsys S;
	int j, l;
	double v;

	if (N.op[n] != O_LOOP)
		return 0;
	loop_get(loop_ent(n), &d);
	if (d.kind != LK_FOR || !linsys_of(&d, &S))
		return 0;
	for (j = 0; j < S.n; j++) {
		if (!num_of(S.init[j], &v))
			return 0;
		v0[j] = v;
		for (l = 0; l < S.n; l++)
			m[j * S.n + l] = S.M[j * 16 + l];
	}
	*dim = S.n;
	*res = loop_res(n);
	return 1;
}

/* ---- loops: one state at a time ---- */

struct ls {
	struct loopdef d;
	nid lo, K, iv;
	nid S[MAXST];
	int busy[MAXST];
};

static nid state_closed(struct ls *L0, int j);

/* expression e at iteration k (index lo + k, states S(k)) */
static nid
at_iter(struct ls *L0, nid e, nid k)
{
	nid map[NSLOT];
	int l;

	memset(map, 0, sizeof map);
	if (L0->d.kind == LK_FOR)
		map[L0->d.islot] = ADD(L0->lo, k);
	for (l = 0; l < L0->d.nst; l++) {
		if (!has_slot(e, L0->d.sslot[l]))
			continue;
		if (!state_closed(L0, l))
			return 0;
		map[L0->d.sslot[l]] = at(L0->S[l], KSLOT, k);
	}
	return subst(e, map, NSLOT);
}

/* particular solution of P(t+1) = a P(t) + g(t), g polynomial in t */
static nid
affine_poly(struct ls *L0, int j, double a, nid g, int s)
{
	struct npoly G;
	struct q gc[14], pc[14], aq, one = {1, 1}, oma;
	int d, jj, l;
	int64_t D = 1;
	ty_t t = L0->d.sty[j];
	nid K = L0->K, init = L0->d.init[j], P0, PK;

	if (!npoly_of(g, s, &G, 8) || !np_rational(&G, gc) ||
	    !q_from_double(a, &aq) || !q_sub(one, aq, &oma) || oma.n == 0)
		return 0;
	d = G.deg;
	/* (1 - a) p_j + sum_{l>j} p_l C(l, j) = g_j */
	for (jj = d; jj >= 0; jj--) {
		struct q acc = gc[jj], tq;
		for (l = jj + 1; l <= d; l++) {
			int64_t bc = 1;
			int z;
			for (z = 1; z <= jj; z++)
				bc = bc * (l - jj + z) / z;
			if (!q_mul(pc[l], (struct q){bc, 1}, &tq) ||
			    !q_sub(acc, tq, &acc))
				return 0;
		}
		if (!q_div(acc, oma, &pc[jj]))
			return 0;
	}
	for (jj = 0; jj <= d; jj++) {
		int64_t g0 = gcd64(D, pc[jj].d);
		if (__builtin_mul_overflow(D / g0, pc[jj].d, &D))
			return 0;
	}
	for (jj = 0; jj <= d; jj++)
		q_mul(pc[jj], (struct q){D, 1}, &pc[jj]);
	if (aq.d != 1 && !TY_FP(t))
		return 0;
	P0 = KI(pc[0].n);
	PK = qpoly_node(pc, d, TY_FP(t) ? mk_cast(t, K) : K);
	{
		nid an = aq.d == 1 ? KI(aq.n) : q_node(aq);
		nid r = ADD(MUL(POW(an, TY_FP(t) ? mk_cast(t, K) : K),
		    SUB(MUL(KI(D), init), P0)), PK);
		if (D != 1)
			r = DIV(r, KI(D));
		METH(M_AFFINE);
		return mk_cast(t, r);
	}
}

/* nx = sv + g with g free of sv: g, else 0 */
static nid
acc_part(nid nx, nid sv, uint64_t own)
{
	nid x, y;

	if (nx == sv)
		return KI(0);
	switch (N.op[nx]) {
	case O_ADD:
		if (N.a[nx] == sv && !(node_slots(N.b[nx]) & own))
			return N.b[nx];
		if (N.b[nx] == sv && !(node_slots(N.a[nx]) & own))
			return N.a[nx];
		break;
	case O_SUB:
		if (N.a[nx] == sv && !(node_slots(N.b[nx]) & own))
			return mk_un(O_NEG, N.b[nx]);
		break;
	case O_SEL:
		if (node_slots(N.a[nx]) & own)
			break;
		if ((x = acc_part(N.b[nx], sv, own)) == 0 ||
		    (y = acc_part(N.c[nx], sv, own)) == 0)
			break;
		return mk_sel(N.a[nx], x, y);
	}
	return 0;
}

static nid
state_closed(struct ls *L0, int j)
{
	struct loopdef *d = &L0->d;
	nid nx = d->next[j], sv = var_of(d->sslot[j], d->sname[j], d->sty[j]);
	nid K = L0->K, g, r = 0;
	int s = d->islot;
	ty_t t = d->sty[j];
	uint64_t own = 1ULL << d->sslot[j];

	if (L0->S[j])
		return L0->S[j];
	if (L0->busy[j])
		return 0;
	L0->busy[j] = 1;
	/* (a) overwritten every iteration */
	if (!(node_slots(nx) & own)) {
		nid last = at_iter(L0, nx, SUB(K, KI(1)));
		if (last) {
			r = mk_sel(mk_bin(O_EQ, K, KI(0)), d->init[j], last);
			METH(M_LAST);
		}
		goto out;
	}
	/* (b) s + g, also through branches: c ? s + a : s - b */
	if ((g = acc_part(nx, sv, own)) != 0) {
		nid iv = var_of(s, d->name, d->ity);
		nid body = at_iter(L0, g, SUB(iv, L0->lo));
		if (!body)
			goto out;
		g = sum_closed(body, s, L0->lo, SUB(ADD(L0->lo, K), KI(1)), NULL);
		if (g) {
			r = mk_cast(t, ADD(d->init[j], g));
			METH(M_ACC);
		}
		goto out;
	}
	if (N.op[nx] == O_MUL && d->kind == LK_FOR &&
	    ((N.a[nx] == sv && !(node_slots(N.b[nx]) & own)) ||
	    (N.b[nx] == sv && !(node_slots(N.a[nx]) & own)))) {
		nid iv = var_of(s, d->name, d->ity), body;
		body = at_iter(L0, N.a[nx] == sv ? N.b[nx] : N.a[nx],
		    SUB(iv, L0->lo));
		g = body ? prod_closed(body, s, L0->lo, SUB(ADD(L0->lo, K),
		    KI(1)), NULL) : 0;
		if (g) {
			r = mk_cast(t, MUL(d->init[j], g));
			METH(M_ACC);
		}
		goto out;
	}
	/* (d) a s + g(k): a constant */
	{
		int sl = d->sslot[j];
		double M1;
		nid rest;
		if (linear_form(nx, &sl, 1, &M1, &rest) && M1 != 1 && M1 != 0) {
			/* g as a function of the iteration t, in the index slot */
			nid gk = at_iter(L0, rest, var_of(s, d->name, TY_INT));
			if (gk)
				r = affine_poly(L0, j, M1, gk, s);
			if (r)
				goto out;
		}
	}
	/* symbolic a with a constant g: a s + c */
	if (N.op[nx] == O_ADD || N.op[nx] == O_MUL) {
		nid a = 0, c = 0;
		uint64_t sm = 0;
		int l;
		for (l = 0; l < d->nst; l++)
			sm |= 1ULL << d->sslot[l];
		if (d->kind == LK_FOR)
			sm |= 1ULL << s;
		if (N.op[nx] == O_MUL && N.b[nx] == sv && !(node_slots(N.a[nx]) & sm))
			a = N.a[nx], c = KI(0);
		else if (N.op[nx] == O_ADD && N.op[N.a[nx]] == O_MUL &&
		    N.b[N.a[nx]] == sv && !(node_slots(N.a[N.a[nx]]) & sm) &&
		    !(node_slots(N.b[nx]) & sm))
			a = N.a[N.a[nx]], c = N.b[nx];
		if (a) {
			nid Kt = TY_FP(t) ? mk_cast(t, K) : K, aK = POW(a, Kt);
			nid gen = ADD(MUL(aK, d->init[j]), DIV(MUL(c, SUB(aK, KI(1))),
			    SUB(a, KI(1))));
			r = mk_cast(t, mk_sel(mk_bin(O_EQ, a, KI(1)),
			    ADD(d->init[j], MUL(Kt, c)), gen));
			METH(M_AFFINE);
			goto out;
		}
	}
	/* (e) nonlinear maps of one state */
	{
		struct npoly P;
		uint64_t sm = 0;
		int l, sl = d->sslot[j];
		for (l = 0; l < d->nst; l++)
			sm |= 1ULL << d->sslot[l];
		if (d->kind == LK_FOR)
			sm |= 1ULL << s;
		if (node_slots(nx) & sm & ~own)
			goto mobius;
		/* c s^p */
		if (N.op[nx] == O_POW && N.a[nx] == sv) {
			int64_t pw;
			if (int_of(N.b[nx], &pw) && pw >= 2) {
				nid Kt = TY_FP(t) ? mk_cast(t, K) : K;
				r = mk_cast(t, POW(d->init[j], POW(KI(pw), Kt)));
				METH(M_POWMAP);
				goto out;
			}
		}
		if (npoly_of(nx, sl, &P, 2) && P.deg == 2 && TY_FP(t)) {
			double al, be, ga, cc;
			if (num_of(P.c[2], &al) && num_of(P.c[1], &be) &&
			    num_of(P.c[0], &ga)) {
				cc = al * ga + be / 2 - be * be / 4;
				if (fabs(cc) < 1e-12 && al != 0) {
					/* y = al s + be/2, y' = y^2 */
					nid y0 = ADD(MUL(KR(al), d->init[j]), KR(be / 2));
					nid yk = POW(y0, POW(KR(2), mk_cast(t, K)));
					r = DIV(SUB(yk, KR(be / 2)), KR(al));
					METH(M_QUAD);
					goto out;
				}
			}
		}
	mobius:
		/* (a s + b) / (c s + e) */
		if (N.op[nx] == O_DIV && TY_FP(t)) {
			struct npoly A, C;
			double a, b, c, e;
			if (npoly_of(N.a[nx], sl, &A, 1) && npoly_of(N.b[nx], sl, &C, 1) &&
			    A.deg <= 1 && C.deg == 1 && num_of(A.c[0], &b) &&
			    (A.deg == 0 ? (a = 0, 1) : num_of(A.c[1], &a)) &&
			    num_of(C.c[0], &e) && num_of(C.c[1], &c) && c != 0) {
				struct loopdef md;
				nid e00, e01, e10, e11;
				int dummy;
				memset(&md, 0, sizeof md);
				md.kind = LK_FOR;
				md.nst = 2;
				md.sslot[0] = 1;
				md.sslot[1] = 2;
				md.sty[0] = md.sty[1] = TY_REAL;
				/* (u, v)' = (a u + b v, c u + e v), M^k columns */
				md.next[0] = ADD(MUL(KR(a), var_of(1, 1, TY_REAL)),
				    MUL(KR(b), var_of(2, 1, TY_REAL)));
				md.next[1] = ADD(MUL(KR(c), var_of(1, 1, TY_REAL)),
				    MUL(KR(e), var_of(2, 1, TY_REAL)));
				size_t keep = cl_extra.n;
				md.init[0] = KR(1);
				md.init[1] = KR(0);
				e00 = linsys_closed(&md, 0, K, &dummy);
				e10 = linsys_closed(&md, 1, K, &dummy);
				md.init[0] = KR(0);
				md.init[1] = KR(1);
				e01 = linsys_closed(&md, 0, K, &dummy);
				e11 = linsys_closed(&md, 1, K, &dummy);
				cl_meth &= ~(1u << M_LINREC);
				/* keep one copy of the characteristic polynomial */
				if (cl_extra.n > keep) {
					char *nl = strchr(cl_extra.p + keep, '\n');
					if (nl) {
						cl_extra.n = (size_t)(nl - cl_extra.p) + 1;
						cl_extra.p[cl_extra.n] = 0;
					}
				}
				if (e00 && e01 && e10 && e11) {
					nid x0 = d->init[j];
					r = DIV(ADD(MUL(e00, x0), e01), ADD(MUL(e10, x0),
					    e11));
					METH(M_MOBIUS);
					goto out;
				}
			}
		}
	}
out:
	L0->busy[j] = 0;
	L0->S[j] = r;
	return r;
}

/* ---- while loops: the trip count ---- */

static int
is_state(const struct loopdef *d, nid e, int *j)
{
	int l;

	while (N.op[e] == O_CAST && !TY_FP(N.ty[e]))
		e = N.a[e];
	if (N.op[e] != O_VAR)
		return 0;
	for (l = 0; l < d->nst; l++)
		if (N.ki[e] == d->sslot[l]) {
			*j = l;
			return 1;
		}
	return 0;
}

static nid
bitlen(nid v)
{
	ty_t t = N.ty[v];
	int w = TYK(t) == K_UINT || TYK(t) == K_SINT ? TYW(t) : 64;

	if (TYK(t) == K_INT || TYK(t) == K_LIT)
		v = mk_cast(TY(K_UINT, 64), v);
	return SUB(KI(w), mk_cast(TY_INT, mk_un(O_CLZ, v)));
}

static nid
trip_count(const struct loopdef *d)
{
	nid c = d->cond, x, y, nx, v0, sv, T = 0;
	int op, j, l;
	uint64_t sm = 0;
	double dv;

	for (l = 0; l < d->nst; l++)
		sm |= 1ULL << d->sslot[l];
	if (!(opinfo[N.op[c]].fl & OF_CMP))
		return 0;
	op = N.op[c];
	x = N.a[c];
	y = N.b[c];
	if (node_slots(y) & sm) {
		nid z = x;
		x = y;
		y = z;
		op = op == O_LT ? O_GT : op == O_LE ? O_GE : op == O_GT ? O_LT :
		    op == O_GE ? O_LE : op;
	}
	if (node_slots(y) & sm)
		return 0;
	/* x is v or v*v */
	{
		int sq = 0;
		nid base = x;
		if ((N.op[x] == O_MUL && N.a[x] == N.b[x]) ||
		    (N.op[x] == O_POW && const_is(N.b[x], 2))) {
			base = N.a[x];
			sq = 1;
		}
		if (!is_state(d, base, &j))
			return 0;
		nx = d->next[j];
		sv = var_of(d->sslot[j], d->sname[j], d->sty[j]);
		if (node_slots(nx) & sm & ~(1ULL << d->sslot[j]))
			return 0;
		v0 = d->init[j];
		y = mk_cast(d->sty[j], y);
		if (sq) {
			/* v += 1 while v*v <= n (or < n) */
			if (!(N.op[nx] == O_ADD && N.a[nx] == sv && const_is(N.b[nx], 1)))
				return 0;
			if (TY_FP(d->sty[j]))
				return 0;
			if (op == O_LE)
				T = SUB(ADD(mk_un(O_ISQRT, y), KI(1)), v0);
			else if (op == O_LT)
				T = SUB(ADD(mk_un(O_ISQRT, SUB(y, KI(1))), KI(1)), v0);
			else
				return 0;
			T = mk_bin(O_MAX, KI(0), T);
			METH(M_TRIP);
			return T;
		}
	}
	/* v + d */
	if ((N.op[nx] == O_ADD || N.op[nx] == O_SUB) && N.a[nx] == sv &&
	    num_of(N.b[nx], &dv) && dv != 0) {
		int64_t di;
		if (N.op[nx] == O_SUB)
			dv = -dv;
		if (TY_FP(d->sty[j])) {
			nid diff = dv > 0 ? SUB(y, v0) : SUB(v0, y);
			double ad = fabs(dv);
			if ((dv > 0 && op == O_LT) || (dv < 0 && op == O_GT))
				T = mk_un(O_CEIL, DIV(diff, KR(ad)));
			else if ((dv > 0 && op == O_LE) || (dv < 0 && op == O_GE))
				T = ADD(mk_un(O_FLOOR, DIV(diff, KR(ad))), KR(1));
			else
				return 0;
			T = mk_cast(TY_INT, mk_bin(O_MAX, KR(0), T));
		} else {
			di = (int64_t)fabs(dv);
			if ((dv > 0 && op == O_LT) || (dv < 0 && op == O_GT))
				T = DIV(ADD(dv > 0 ? SUB(y, v0) : SUB(v0, y), KI(di - 1)),
				    KI(di));
			else if ((dv > 0 && op == O_LE) || (dv < 0 && op == O_GE))
				T = ADD(DIV(dv > 0 ? SUB(y, v0) : SUB(v0, y), KI(di)),
				    KI(1));
			else if (op == O_NE && di == 1)
				T = dv > 0 ? SUB(y, v0) : SUB(v0, y);
			else
				return 0;
			T = mk_sel(mk_bin(dv > 0 ? (op == O_LE ? O_LE : O_LT) :
			    (op == O_GE ? O_GE : O_GT), v0, y), T, KI(0));
			T = mk_cast(TY_INT, T);
		}
		METH(M_TRIP);
		return T;
	}
	/* v * c, c > 1, v0 > 0 */
	if (N.op[nx] == O_MUL && (N.b[nx] == sv || N.a[nx] == sv) &&
	    num_of(N.a[nx] == sv ? N.b[nx] : N.a[nx], &dv) && dv > 1 &&
	    (op == O_LT || op == O_LE)) {
		nid ratio = DIV(mk_cast(TY_REAL, y), mk_cast(TY_REAL, v0));
		nid lg = DIV(mk_un(O_LOG, ratio), KR(log(dv)));
		if (op == O_LT)
			T = mk_un(O_CEIL, SUB(lg, KR(1e-12)));
		else
			T = ADD(mk_un(O_FLOOR, ADD(lg, KR(1e-12))), KR(1));
		T = mk_sel(mk_bin(op, v0, y), mk_cast(TY_INT, T), KI(0));
		METH(M_TRIP);
		return T;
	}
	/* v / c or v >> k (integers, v >= 0): while v > N */
	if ((N.op[nx] == O_DIV || N.op[nx] == O_SHR) && N.a[nx] == sv &&
	    num_of(N.b[nx], &dv) && !TY_FP(d->sty[j])) {
		int64_t cv = N.op[nx] == O_SHR ? (int64_t)1 << (int64_t)dv :
		    (int64_t)dv, Nv;
		nid Nn;
		if (cv < 2)
			return 0;
		/* normalise the condition to v > N */
		if (op == O_GT)
			Nn = y;
		else if (op == O_GE)
			Nn = SUB(y, KI(1));
		else if (op == O_NE && const_is(y, 0))
			Nn = KI(0);
		else
			return 0;
		if ((cv & (cv - 1)) == 0 && int_of(Nn, &Nv) && Nv >= 0 &&
		    ((Nv + 1) & Nv) == 0) {
			/* c = 2^k, N + 1 = 2^m: (bitlen(v0) - m) / k rounded up */
			int k = ilog2u((uint64_t)cv), m = ilog2u((uint64_t)Nv + 1);
			nid bl = SUB(bitlen(v0), KI(m));
			T = k == 1 ? bl : DIV(ADD(bl, KI(k - 1)), KI(k));
		} else {
			nid lg = DIV(mk_un(O_LOG, DIV(mk_cast(TY_REAL, v0),
			    mk_cast(TY_REAL, ADD(Nn, KI(1))))), KR(log((double)cv)));
			T = ADD(mk_cast(TY_INT, mk_un(O_FLOOR, ADD(lg, KR(1e-12)))),
			    KI(1));
		}
		T = mk_sel(mk_bin(O_GT, v0, Nn), T, KI(0));
		METH(M_TRIP);
		return T;
	}
	return 0;
}

/* the loop with an extra counter state: how many iterations ran */
static nid
counted(nid n)
{
	struct loopdef d;
	int top = MAXP, j;

	loop_get(loop_ent(n), &d);
	if (d.nst == MAXST)
		return 0;
	for (j = 0; j < d.nst; j++)
		if (d.sslot[j] > top)
			top = d.sslot[j];
	if (top + 1 >= KSLOT - 1)
		return 0;
	d.sslot[d.nst] = top + 1;
	d.sname[d.nst] = sym_cstr("iterations");
	d.sty[d.nst] = TY_INT;
	d.init[d.nst] = KI(0);
	d.init[d.nst] = mk_const_i(TY_INT, 0);
	d.next[d.nst] = ADD(var_of(top + 1, d.sname[d.nst], TY_INT), KI(1));
	d.nst++;
	d.orig = LO_BLOCK;
	return mk_loop(loop_intern(&d), d.nst - 1, N.a[n], N.b[n], 0);
}

/* ---- one loop node ---- */

static const struct func *cl_func;
static int cl_trip_shown;

nid
loop_closed(nid n, const char **how, struct sbuf *extra)
{
	struct ls L0;
	nid T, r, K;
	int res = loop_res(n), ex;

	memset(&L0, 0, sizeof L0);
	loop_get(loop_ent(n), &L0.d);
	(void)how;
	(void)extra;
	if (L0.d.orig == LO_SUM && L0.d.nst == 1)
		return sum_closed(L0.d.aux, L0.d.islot, N.a[n], N.b[n], NULL);
	if (L0.d.orig == LO_PROD && L0.d.nst == 1)
		return prod_closed(L0.d.aux, L0.d.islot, N.a[n], N.b[n], NULL);
	K = var_of(KSLOT, sym_cstr("k"), TY_INT);
	L0.K = K;
	if (L0.d.kind == LK_FOR) {
		L0.lo = N.a[n];
		T = ADD(SUB(N.b[n], N.a[n]), KI(1));
	} else {
		/* a while loop runs T times; its states read no index */
		L0.lo = KI(0);
		T = trip_count(&L0.d);
		if (!T) {
			sb_puts(&cl_extra, "  no closed form for the trip count of "
			    "the while loop\n");
			return 0;
		}
		/* check the trip count on its own when it can be */
		if (cl_func && (node_slots(n) >> MAXP) == 0) {
			nid cn = counted(n);
			struct vres v;
			if (cn && !cl_trip_shown) {
				T = tidy(T);
				verify_nodes(cl_func, cn, T, &v);
				if (v.bad)
					sb_printf(&cl_extra, "  trip count %s is wrong: %s\n",
					    node_str(T), sb_str(&v.wit));
				else
					sb_printf(&cl_extra, "  trip count %s (%s against the "
					    "loop at %llu points)\n", node_str(T),
					    v.n ? "checked" : "unchecked",
					    (unsigned long long)v.n);
				vres_free(&v);
				if (v.bad)
					return 0;
			}
		}
	}
	if (L0.d.kind == LK_WHILE) {
		L0.d.kind = LK_FOR;
		L0.d.islot = KSLOT - 1;
		L0.d.name = sym_cstr("t");
		L0.d.ity = TY_INT;
	}
	/* linear system first: it handles mutual dependence */
	r = linsys_closed(&L0.d, res, K, &ex);
	if (!r) {
		r = state_closed(&L0, res);
		if (!r)
			return 0;
	}
	return at(r, KSLOT, T);
}

/* ---- the command ---- */

struct clrep {
	nid orig, cf;
	unsigned meth;
	struct sbuf extra;
	int top, checked;
	struct vres v;
};
static struct clrep reps[32];
static int nreps;

static nid cl_walk(nid n, nid *memo, uint32_t lim);

/* the value of a counted loop that runs no iteration */
static nid
zero_init(nid n)
{
	uint32_t e = loop_ent(n);

	return loops.init[loops.st[e] + loop_res(n)];
}

static uint32_t
cl_ent(uint32_t e, nid *memo, uint32_t lim)
{
	struct loopdef d;
	int k, ch = 0;
	nid t;

	loop_get(e, &d);
	for (k = 0; k < d.nst; k++) {
		t = cl_walk(d.init[k], memo, lim);
		ch |= t != d.init[k];
		d.init[k] = t;
		t = mk_cast(d.sty[k], cl_walk(d.next[k], memo, lim));
		ch |= t != d.next[k];
		d.next[k] = t;
	}
	if (d.cond) {
		t = cl_walk(d.cond, memo, lim);
		ch |= t != d.cond;
		d.cond = t;
	}
	if (d.aux) {
		t = cl_walk(d.aux, memo, lim);
		ch |= t != d.aux;
		d.aux = t;
	}
	return ch ? loop_intern(&d) : e;
}

static nid
cl_walk(nid n, nid *memo, uint32_t lim)
{
	nid r;

	if (n == 0)
		return 0;
	if (n < lim && memo[n])
		return memo[n];
	switch (N.op[n]) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
	case O_TABLE:
	case O_BARRIER:
	case O_ATOMIC:
		r = n;
		break;
	case O_SEL:
		r = mk_sel(cl_walk(N.a[n], memo, lim), cl_walk(N.b[n], memo, lim),
		    cl_walk(N.c[n], memo, lim));
		break;
	case O_CAST:
		r = mk_cast(N.ty[n], cl_walk(N.a[n], memo, lim));
		break;
	case O_INDEX:
		r = mk_index(N.a[n], cl_walk(N.b[n], memo, lim));
		break;
	case O_LOOP: {
		nid ln = mk_loop(cl_ent(loop_ent(n), memo, lim), loop_res(n),
		    cl_walk(N.a[n], memo, lim), cl_walk(N.b[n], memo, lim),
		    cl_walk(N.c[n], memo, lim)), cf;
		struct clrep *rp = nreps < 32 ? &reps[nreps] : NULL;
		int isfor = loops.kind[loop_ent(ln)] == LK_FOR;
		r = ln;
		if (N.op[ln] != O_LOOP)
			break;
		{
			int top = (node_slots(ln) >> MAXP) == 0 && cl_func && rp;
			int mode;
			for (mode = top ? 0 : 2; mode <= 2; mode++) {
				struct vres v;
				cl_meth = 0;
				sb_clear(&cl_extra);
				cl_trip_shown = 0;
				cl_fdmode = mode;
				cl_fdused = 0;
				cf = loop_closed(ln, NULL, NULL);
				cl_fdmode = 2;
				if (cf)
					cf = tidy(cf);
				if (!cf || !cl_fdused || mode == 2)
					break;
				if (verify_nodes(cl_func, n, cf, &v)) {
					vres_free(&v);
					break;
				}
				vres_free(&v);
			}
		}
		if (rp) {
			memset(rp, 0, sizeof *rp);
			rp->orig = n;
			rp->cf = cf;
			rp->meth = cl_meth;
			rp->top = (node_slots(ln) >> MAXP) == 0;
			if (cl_extra.n)
				sb_puts(&rp->extra, sb_str(&cl_extra));
			nreps++;
		}
		if (cf && rp && rp->top && cl_func) {
			/* the unguarded form, or one guarded for empty ranges */
			int ok = verify_nodes(cl_func, n, cf, &rp->v);
			if (!ok && isfor) {
				struct vres v2;
				nid T = ADD(SUB(N.b[ln], N.a[ln]), KI(1));
				nid alt = (loops.flags[loop_ent(ln)] & LF_NEGUNDEF) ?
				    mk_undef(N.ty[cf]) : zero_init(ln);
				nid g = tidy(mk_sel(mk_bin(O_GE, T, KI(0)), cf, alt));
				if (verify_nodes(cl_func, n, g, &v2)) {
					rp->cf = cf = g;
					vres_free(&rp->v);
					rp->v = v2;
				} else
					vres_free(&v2);
			}
			rp->checked = 1;
			/* a form that disagrees with the loop is not used */
			if (rp->v.bad || rp->v.n == 0)
				cf = 0;
		}
		if (cf)
			r = cf;
		break;
	}
	default:
		if (opinfo[N.op[n]].arity == 1)
			r = mk_un(N.op[n], cl_walk(N.a[n], memo, lim));
		else if (IS_TEX(N.op[n]))
			r = mk_tex(N.op[n], N.ki[n], N.a[n],
			    cl_walk(N.b[n], memo, lim),
			    N.c[n] ? cl_walk(N.c[n], memo, lim) : 0);
		else if (IS_DOT(N.op[n]) || IS_SG(N.op[n]))
			r = mk_op(N.op[n], cl_walk(N.a[n], memo, lim),
			    cl_walk(N.b[n], memo, lim),
			    N.c[n] ? cl_walk(N.c[n], memo, lim) : 0);
		else
			r = mk_bin(N.op[n], cl_walk(N.a[n], memo, lim),
			    cl_walk(N.b[n], memo, lim));
	}
	if (n < lim)
		memo[n] = r;
	return r;
}

static const char *
fname(const struct func *f)
{
	return f->anon ? node_str(f->body) : sym_str(f->name);
}

static void
print_meth(unsigned m)
{
	int k, first = 1;

	if (!m)
		return;
	printf("  method: ");
	rep_list("method");
	for (k = 0; k < M_N; k++)
		if ((m >> k) & 1) {
			printf("%s%s", first ? "" : ", ", mname[k]);
			rep_str(NULL, mname[k]);
			first = 0;
		}
	rep_end();
	printf("\n");
}

void
cmd_closed(struct func *f)
{
	uint32_t lim = N.n;
	nid *memo = xcalloc(lim, sizeof *memo), body;
	int i, solved = 0, alltop = 1;

	cl_func = f;
	for (i = 0; i < nreps; i++) {
		free(reps[i].extra.p);
		vres_free(&reps[i].v);
	}
	nreps = 0;
	body = cl_walk(f->body, memo, lim);
	free(memo);
	cl_func = NULL;
	rep_begin("closed");
	if (nreps == 0) {
		printf("%s has no loops\n", fname(f));
		rep_list("loops");
		rep_end();
		return;
	}
	rep_list("loops");
	for (i = 0; i < nreps; i++) {
		struct clrep *rp = &reps[i];
		printf("%s\n", node_str(rp->orig));
		rep_obj(NULL);
		rep_expr("loop", rp->orig);
		if (rp->cf)
			rep_expr("closed_form", rp->cf);
		else
			rep_null("closed_form");
		rep_bool("accepted", rp->cf && !(rp->checked && (rp->v.bad ||
		    rp->v.n == 0)));
		rep_bool("checked", rp->cf && rp->checked);
		if (rp->cf && rp->checked && (rp->v.bad || rp->v.n == 0)) {
			printf("  candidate %s (rejected)\n", node_str(rp->cf));
			alltop = 0;
		} else if (rp->cf) {
			printf("  = %s\n", node_str(rp->cf));
			solved++;
		} else {
			printf("  no closed form found\n");
			alltop = 0;
		}
		print_meth(rp->meth);
		if (rp->extra.n)
			fputs(sb_str(&rp->extra), stdout);
		if (rp->cf && rp->checked)
			print_verify("the closed form", &rp->v);
		else if (rp->cf && !rp->top)
			printf("  (inside another loop: checked as part of it)\n");
		if (!rp->top)
			alltop = 0;
		rep_end();
	}
	rep_end();
	if (body != f->body && !(nreps == 1 && alltop && reps[0].orig == f->body)) {
		struct vres v;
		printf("%s = %s\n", fname(f), node_str(body));
		rep_expr("result", body);
		if (!(node_slots(body) >> MAXP)) {
			verify_nodes(f, f->body, body, &v);
			print_verify("it", &v);
			vres_free(&v);
		}
	}
	(void)solved;
}
