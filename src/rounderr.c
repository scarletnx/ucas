/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * rounderr: the rounding error of a float expression, to first order.
 *
 * Every rounded operation i returns its exact result times (1 + d_i),
 * |d_i| <= u.  To first order the computed value is the exact one plus
 * sum c_i d_i, with c_i = z_i dG/dz_i (the operation's value times the
 * sensitivity of the result to it), found by one reverse sweep of the
 * expression at each input.  Three numbers follow from the same c_i:
 *
 *   worst case   u sum |c_i|: every error at its bound and aligned.  For
 *                n operations of equal weight this is n u, the classical
 *                gamma_n = n u / (1 - n u).
 *   probable     lambda u sqrt(sum c_i^2), which holds with probability
 *                at least 1 - 2 exp(-lambda^2 / 2) when the d_i are
 *                independent with mean zero (Hoeffding's inequality for
 *                bounded independent terms; for equal weights it is
 *                Higham and Mary's gamma~_n = lambda sqrt(n) u, "A new
 *                approach to probabilistic rounding error analysis",
 *                SIAM J. Sci. Comput. 41(5), 2019).
 *   expected     u sqrt(sum c_i^2 / 3): the root mean square when the
 *                d_i are independent and uniform in [-u, u].
 *
 * The independence is a model, not a theorem (rounding errors are
 * deterministic), and the probability is over that model at one input,
 * not over inputs; the report says so.  The values are taken in long
 * double as the exact ones, and each operation as correctly rounded
 * (which libm functions and a GPU's are not always: the report names
 * the assumption).
 */
#include "numsearch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef long double ld;

struct rnode {
	nid id;
	int a, b, c;		/* operand indices, -1 */
	int rounds;		/* a rounded float operation */
	ld u;
	ld v, adj;
};

static struct {
	struct rnode *n;
	int nn, cap;
	int *ix;		/* nid -> index + 1 */
	uint32_t nix;
	char why[120];
} R;

static ld
unit(ty_t t)
{
	return TYW(t) == 16 ? 0x1p-11L : TYW(t) == 32 ? 0x1p-24L : 0x1p-53L;
}

static int
collect(nid e)
{
	struct rnode *x;
	int a = -1, b = -1, c = -1, i;
	ty_t t = N.ty[e];

	if (e < R.nix && R.ix[e])
		return R.ix[e] - 1;
	switch (N.op[e]) {
	case O_CONST: case O_VAR:
		break;
	case O_CAST: case O_NEG: case O_ABS: case O_SQRT: case O_EXP:
	case O_LOG: case O_SIN: case O_COS: case O_TAN:
		a = collect(N.a[e]);
		break;
	case O_ADD: case O_SUB: case O_MUL: case O_DIV: case O_POW:
	case O_LT: case O_LE: case O_GT: case O_GE: case O_EQ: case O_NE:
	case O_MIN: case O_MAX:
		a = collect(N.a[e]);
		b = collect(N.b[e]);
		break;
	case O_FMA: case O_SEL:
		a = collect(N.a[e]);
		b = collect(N.b[e]);
		c = collect(N.c[e]);
		break;
	default:
		if (!R.why[0])
			snprintf(R.why, sizeof R.why, "%s", opinfo[N.op[e]].name);
		return -1;
	}
	if (N.op[e] == O_POW && !is_const(N.b[e]) && !R.why[0])
		snprintf(R.why, sizeof R.why, "a power with a variable exponent");
	if (N.op[e] == O_CAST && !TY_FP(t) && !R.why[0])
		snprintf(R.why, sizeof R.why, "a conversion to %s", ty_str(t));
	if (R.why[0])
		return -1;
	if (R.nn == R.cap) {
		R.cap = R.cap ? 2 * R.cap : 64;
		R.n = xrealloc(R.n, (size_t)R.cap * sizeof *R.n);
	}
	i = R.nn++;
	x = &R.n[i];
	memset(x, 0, sizeof *x);
	x->id = e;
	x->a = a;
	x->b = b;
	x->c = c;
	if (TYK(t) == K_FLOAT) {
		x->u = unit(t);
		switch (N.op[e]) {
		case O_ADD: case O_SUB: case O_MUL: case O_DIV: case O_SQRT:
		case O_FMA: case O_POW: case O_EXP: case O_LOG: case O_SIN:
		case O_COS: case O_TAN:
			x->rounds = 1;
			break;
		case O_CAST: {
			/* an integer or a wider float does not always fit */
			ty_t ta = N.ty[N.a[e]];
			x->rounds = TYK(ta) == K_REAL || (TYK(ta) == K_FLOAT &&
			    TYW(ta) > TYW(t)) || (TY_ISINT(ta) && (!TY_SIZED(ta) ||
			    TYW(ta) > (TYW(t) == 16 ? 11 : TYW(t) == 32 ? 24 : 53)));
			break;
		}
		}
	}
	if (e < R.nix)
		R.ix[e] = i + 1;
	return i;
}

static ld
param_val(const struct func *g, nid e, const int64_t *in_i,
    const double *in_r, uint32_t k)
{
	int s = (int)N.ki[e];

	(void)g;
	return TY_FP(N.ty[e]) ? (ld)in_r[(size_t)s * CH + k] :
	    (ld)in_i[(size_t)s * CH + k];
}

/* values forward, adjoints backward; 0 when a value is not finite */
static int
sweep(const struct func *g, const int64_t *in_i, const double *in_r,
    uint32_t k)
{
	int i;

	for (i = 0; i < R.nn; i++) {
		struct rnode *x = &R.n[i];
		ld a = x->a >= 0 ? R.n[x->a].v : 0, b = x->b >= 0 ? R.n[x->b].v : 0;
		ld c = x->c >= 0 ? R.n[x->c].v : 0;
		switch (N.op[x->id]) {
		case O_CONST:
			x->v = TY_FP(N.ty[x->id]) ? (ld)N.kr[x->id] : (ld)N.ki[x->id];
			break;
		case O_VAR: x->v = param_val(g, x->id, in_i, in_r, k); break;
		case O_CAST: x->v = a; break;
		case O_NEG: x->v = -a; break;
		case O_ABS: x->v = fabsl(a); break;
		case O_SQRT: x->v = sqrtl(a); break;
		case O_EXP: x->v = expl(a); break;
		case O_LOG: x->v = logl(a); break;
		case O_SIN: x->v = sinl(a); break;
		case O_COS: x->v = cosl(a); break;
		case O_TAN: x->v = tanl(a); break;
		case O_ADD: x->v = a + b; break;
		case O_SUB: x->v = a - b; break;
		case O_MUL: x->v = a * b; break;
		case O_DIV: x->v = a / b; break;
		case O_POW: x->v = powl(a, b); break;
		case O_MIN: x->v = a < b ? a : b; break;
		case O_MAX: x->v = a > b ? a : b; break;
		case O_LT: x->v = a < b; break;
		case O_LE: x->v = a <= b; break;
		case O_GT: x->v = a > b; break;
		case O_GE: x->v = a >= b; break;
		case O_EQ: x->v = a == b; break;
		case O_NE: x->v = a != b; break;
		case O_FMA: x->v = a * b + c; break;
		case O_SEL: x->v = a != 0 ? b : c; break;
		}
		if (!isfinite((double)x->v))
			return 0;
		x->adj = 0;
	}
	R.n[R.nn - 1].adj = 1;
	for (i = R.nn - 1; i >= 0; i--) {
		struct rnode *x = &R.n[i];
		ld a = x->a >= 0 ? R.n[x->a].v : 0, b = x->b >= 0 ? R.n[x->b].v : 0;
		ld w = x->adj;
#define TO(j, d) do { if ((j) >= 0) R.n[j].adj += w * (d); } while (0)
		if (w == 0)
			continue;
		switch (N.op[x->id]) {
		case O_CAST: TO(x->a, 1); break;
		case O_NEG: TO(x->a, -1); break;
		case O_ABS: TO(x->a, a < 0 ? -1 : 1); break;
		case O_SQRT: TO(x->a, 1 / (2 * x->v)); break;
		case O_EXP: TO(x->a, x->v); break;
		case O_LOG: TO(x->a, 1 / a); break;
		case O_SIN: TO(x->a, cosl(a)); break;
		case O_COS: TO(x->a, -sinl(a)); break;
		case O_TAN: TO(x->a, 1 + x->v * x->v); break;
		case O_ADD: TO(x->a, 1); TO(x->b, 1); break;
		case O_SUB: TO(x->a, 1); TO(x->b, -1); break;
		case O_MUL: TO(x->a, b); TO(x->b, a); break;
		case O_DIV: TO(x->a, 1 / b); TO(x->b, -a / (b * b)); break;
		case O_POW: TO(x->a, b * powl(a, b - 1)); break;
		case O_MIN: TO(a < b ? x->a : x->b, 1); break;
		case O_MAX: TO(a > b ? x->a : x->b, 1); break;
		case O_FMA: TO(x->a, b); TO(x->b, a); TO(x->c, 1); break;
		case O_SEL: TO(a != 0 ? x->b : x->c, 1); break;
		}
#undef TO
		if (!isfinite((double)x->adj))
			return 0;
	}
	return 1;
}

#define RE_SAMPLE 4096

void
rounderr_measure(const struct func *g, double lambda, struct rounderr *r)
{
	struct diter d;
	int64_t *in_i;
	double *in_r;
	struct plan p;
	nid roots[2];
	uint8_t ok[CH];
	uint32_t np, k, wloc;
	int i, root;

	memset(r, 0, sizeof *r);
	r->lambda = lambda > 0 ? lambda : 4;
	r->conf = 1 - 2 * ns_exp(-r->lambda * r->lambda / 2);
	if (TYK(N.ty[g->body]) != K_FLOAT) {
		snprintf(r->why, sizeof r->why, "the result is %s, not a float",
		    ty_str(N.ty[g->body]));
		return;
	}
	r->w = TYW(N.ty[g->body]);
	memset(&R, 0, sizeof R);
	R.nix = N.n;
	R.ix = xcalloc(R.nix, sizeof *R.ix);
	root = collect(g->body);
	if (root < 0 || R.why[0]) {
		snprintf(r->why, sizeof r->why, "%.90s is not analysed",
		    R.why[0] ? R.why : "an operation");
		free(R.ix);
		free(R.n);
		return;
	}
	for (i = 0; i < R.nn; i++)
		r->nops += R.n[i].rounds;
	/* the where clause decides which inputs count */
	roots[0] = g->where ? g->where : g->body;
	roots[1] = 0;
	plan_build(&p, roots, 1, 0);
	wloc = g->where ? plan_loc(&p, g->where) : NOTOK;
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	diter_init(&d, g, RE_SAMPLE, 257, RE_SAMPLE);
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wloc, np, ok);
		for (k = 0; k < np; k++) {
			ld s1 = 0, s2 = 0, gv;
			if (!ok[k] || !sweep(g, in_i, in_r, k))
				continue;
			gv = fabsl(R.n[root].v);
			if (gv == 0)
				continue;
			for (i = 0; i < R.nn; i++)
				if (R.n[i].rounds) {
					ld c = fabsl(R.n[i].adj * R.n[i].v) * R.n[i].u;
					s1 += c;
					s2 += c * c;
				}
			if (!isfinite((double)s1))
				continue;
			r->points++;
			s1 /= gv;
			s2 = sqrtl(s2) / gv;
			if ((double)s1 > r->worst)
				r->worst = (double)s1;
			if ((double)(r->lambda * s2) > r->prob)
				r->prob = (double)(r->lambda * s2);
			if ((double)(s2 / sqrtl(3)) > r->rms)
				r->rms = (double)(s2 / sqrtl(3));
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	free(R.ix);
	free(R.n);
	r->ok = r->points > 0;
	if (!r->ok)
		snprintf(r->why, sizeof r->why, "no input of the sample gives a "
		    "finite, non-zero value");
}

void
rounderr_print(const struct rounderr *r, const char *ind, int rep)
{
	double u = r->w == 16 ? 0x1p-11 : r->w == 32 ? 0x1p-24 : 0x1p-53;
	int e = r->w == 16 ? 11 : r->w == 32 ? 24 : 53;

	if (!r->ok) {
		printf("%srounding: not analysed (%s)\n", ind, r->why);
		return;
	}
	printf("%srounding, first order in u = 2^-%d: %d rounded operation%s, "
	    "each taken as correctly rounded; relative to the value, the "
	    "largest over %llu inputs\n", ind, e, r->nops, r->nops == 1 ? "" :
	    "s", (unsigned long long)r->points);
	printf("%s  worst case  %.3g (%.3g u): every error at its bound and "
	    "aligned, u sum |c_i|\n", ind, r->worst, r->worst / u);
	if (r->prob < r->worst)
		printf("%s  probable    %.3g (%.3g u): with probability >= %.5f at "
		    "one input, %g u sqrt(sum c_i^2), if the errors are "
		    "independent with mean 0 (a model; Hoeffding, Higham-Mary)\n",
		    ind, r->prob, r->prob / u, r->conf, r->lambda);
	else
		printf("%s  probable    no better than the worst case: %g u "
		    "sqrt(sum c_i^2) is %.3g u here (it gains from about %g "
		    "operations of similar weight on)\n", ind, r->lambda,
		    r->prob / u, r->lambda * r->lambda);
	printf("%s  expected    %.3g (%.3g u): the root mean square if they "
	    "are independent and uniform in [-u, u]\n", ind, r->rms,
	    r->rms / u);
	if (r->nops > 0 && r->nops * u < 1)
		printf("%s  for %d operations of equal weight: gamma_%d = %d u / "
		    "(1 - %d u) = %.3g, gamma~_%d(%g) = %g sqrt(%d) u = %.3g\n", ind,
		    r->nops, r->nops, r->nops, r->nops, r->nops * u / (1 - r->nops *
		    u), r->nops, r->lambda, r->lambda, r->nops, r->lambda *
		    sqrt((double)r->nops) * u);
	if (!rep)
		return;
	rep_obj("rounding");
	rep_int("operations", r->nops);
	rep_int("points", (long long)r->points);
	rep_num("worst_case", r->worst);
	rep_num("probable", r->prob < r->worst ? r->prob : r->worst);
	rep_num("confidence", r->conf);
	rep_num("lambda", r->lambda);
	rep_num("expected_rms", r->rms);
	rep_str("assumptions", "first order; operations correctly rounded; "
	    "probable and expected: rounding errors independent, mean zero "
	    "(uniform for expected), at one input");
	rep_end();
}

/* the same in one line, for a table (approx): in units of u */
void
rounderr_line(const struct rounderr *r, const char *ind, const char *name)
{
	double u = r->w == 16 ? 0x1p-11 : r->w == 32 ? 0x1p-24 : 0x1p-53;

	if (!r->ok) {
		printf("%srounding of %s: not analysed (%s)\n", ind, name, r->why);
		return;
	}
	printf("%srounding of %s, first order, %d operation%s, relative, in u = "
	    "2^-%d: worst case %.3g", ind, name, r->nops, r->nops == 1 ? "" : "s",
	    r->w == 16 ? 11 : r->w == 32 ? 24 : 53, r->worst / u);
	if (r->prob < r->worst)
		printf(", probable %.3g (p >= %.5f if the errors are independent)",
		    r->prob / u, r->conf);
	else
		printf(", probable no better (under %g operations)", r->lambda *
		    r->lambda);
	printf(", expected %.3g; `error %s ~ F` says what each means\n",
	    r->rms / u, name);
}
