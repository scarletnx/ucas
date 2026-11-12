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
 *
 * Error-free transformations break the independence on purpose: 2Sum
 * (Knuth), Fast2Sum (Dekker) and 2Prod (an fma) compute the rounding
 * error of a sum or a product exactly, s + t = a + b and p + e = a b.
 * Each one found is taken as exact as a whole: its operations add no
 * error, and since the sweep differentiates through them, the error of
 * the s or p it corrects cancels to first order wherever the result adds
 * the error term back (dG/ds = 1 - 1).  They are matched as written, up
 * to negations (RN(-x) = -RN(x)), and only when nothing outside uses
 * their inner values; Fast2Sum's and 2Prod's preconditions are checked
 * at each input, and where one fails its operations round there.
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
	int eft;		/* inside error-free transformation eft - 1 */
	ld u;
	ld v, adj;
};

enum { EF_2SUM, EF_FAST2SUM, EF_2PROD };

struct reft {
	int kind;
	int s;			/* the sum or product it corrects */
	int x, y;		/* Fast2Sum: the operand taken first and the
				 * other; 2Prod: the factors */
	int in[5], nin;		/* its operations */
	int holds;		/* its precondition, at this input */
};

static struct {
	struct rnode *n;
	int nn, cap;
	int *ix;		/* nid -> index + 1 */
	uint32_t nix;
	char why[120];
	struct reft *e;
	int ne, ecap;
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

/* a rounded sum or difference: its operands without their negations and
 * the sign each enters with */
static int
sum_ops(int i, int o[2], int sg[2])
{
	int k, op;

	if (i < 0 || !R.n[i].rounds)
		return 0;
	op = N.op[R.n[i].id];
	if (op != O_ADD && op != O_SUB)
		return 0;
	o[0] = R.n[i].a;
	o[1] = R.n[i].b;
	sg[0] = 1;
	sg[1] = op == O_SUB ? -1 : 1;
	for (k = 0; k < 2; k++)
		while (o[k] >= 0 && N.op[R.n[o[k]].id] == O_NEG) {
			sg[k] = -sg[k];
			o[k] = R.n[o[k]].a;
		}
	return o[0] >= 0 && o[1] >= 0;
}

static int
strip(int i, int *sg)
{
	while (i >= 0 && N.op[R.n[i].id] == O_NEG) {
		*sg = -*sg;
		i = R.n[i].a;
	}
	return i;
}

static int
pair_is(const int o[2], int p, int q)
{
	return (o[0] == p && o[1] == q) || (o[0] == q && o[1] == p);
}

/* i as a sum of the atoms at[] (coefficients added to f, times sg),
 * through sums, differences and negations; 0 when it is not one */
static int
lform(int i, const int at[3], int f[3], int sg, int depth)
{
	int k, op;

	for (k = 0; k < 3; k++)
		if (i == at[k]) {
			f[k] += sg;
			return 1;
		}
	if (i < 0 || depth > 8)
		return 0;
	op = N.op[R.n[i].id];
	if (op == O_NEG)
		return lform(R.n[i].a, at, f, -sg, depth + 1);
	if (op == O_ADD || op == O_SUB)
		return lform(R.n[i].a, at, f, sg, depth + 1) &&
		    lform(R.n[i].b, at, f, op == O_SUB ? -sg : sg, depth + 1);
	return 0;
}

/*
 * The value of i with s = cA a + cB b put in: 0, +-a or +-b (want 0, 1,
 * 2), and in raw the coefficients before that; -1 when it is none.
 */
static int
lvalue(int i, const int at[3], int ca, int cb, int raw[3])
{
	int f[3] = {0, 0, 0}, va, vb;

	if (!lform(i, at, f, 1, 0))
		return -1;
	memcpy(raw, f, sizeof f);
	va = f[0] + f[2] * ca;
	vb = f[1] + f[2] * cb;
	if (va == 0 && vb == 0)
		return 0;
	if ((va == 1 || va == -1) && vb == 0)
		return 1;
	if (va == 0 && (vb == 1 || vb == -1))
		return 2;
	return -1;
}

/* t = +-(a + b - s): the error term itself, not 0 */
static int
is_error(const int raw[3], int ca, int cb)
{
	return (raw[2] == -1 && raw[0] == ca && raw[1] == cb) ||
	    (raw[2] == 1 && raw[0] == -ca && raw[1] == -cb);
}

/* only the EFT's own operations (and the error term's users) read them */
static int
inner_only(const int *in, int nin)
{
	int i, k, j;

	for (i = 0; i < R.nn; i++) {
		int op[3] = {R.n[i].a, R.n[i].b, R.n[i].c}, mine = 0;
		for (j = 0; j < nin; j++)
			mine |= in[j] == i;
		if (mine)
			continue;
		for (k = 0; k < 3; k++)
			for (j = 0; j + 1 < nin; j++)
				if (op[k] == in[j])
					return 0;
	}
	for (j = 0; j < nin; j++)
		if (R.n[in[j]].eft || N.ty[R.n[in[j]].id] != N.ty[R.n[in[nin - 1]].id])
			return 0;
	return 1;
}

static void
eft_add(int kind, int s, int x, int y, const int *in, int nin)
{
	struct reft *e;
	int j;

	if (R.ne == R.ecap) {
		R.ecap = R.ecap ? 2 * R.ecap : 8;
		R.e = xrealloc(R.e, (size_t)R.ecap * sizeof *R.e);
	}
	e = &R.e[R.ne++];
	memset(e, 0, sizeof *e);
	e->kind = kind;
	e->s = s;
	e->x = x;
	e->y = y;
	e->nin = nin;
	for (j = 0; j < nin; j++) {
		e->in[j] = in[j];
		R.n[in[j]].eft = R.ne;
	}
}

/*
 * 2Sum with t its error term: s = a + b, a' = s - b, b' = s - a',
 * da = a - a', db = b - b', t = da + db.
 */
static int
match_2sum(int t)
{
	int to[2], ts[2], k, j, m;

	if (!sum_ops(t, to, ts))
		return 0;
	for (k = 0; k < 2; k++) {
		int da = to[k], db = to[!k], dao[2], das[2], dbo[2], dbs[2];
		if (!sum_ops(da, dao, das) || !sum_ops(db, dbo, dbs))
			continue;
		for (j = 0; j < 2; j++) {
			int x = dao[j], ap = dao[!j], apo[2], aps[2];
			if (!sum_ops(ap, apo, aps))
				continue;
			for (m = 0; m < 2; m++) {
				int s = apo[m], y = apo[!m], so[2], ss[2], bp, bpo[2],
				    bps[2], at[3], raw[3], in[5];
				if (x == y || !sum_ops(s, so, ss) || !pair_is(so, x, y))
					continue;
				bp = dbo[0] == y ? dbo[1] : dbo[1] == y ? dbo[0] : -1;
				if (!sum_ops(bp, bpo, bps) || !pair_is(bpo, s, ap))
					continue;
				at[0] = so[0];
				at[1] = so[1];
				at[2] = s;
				if (lvalue(ap, at, ss[0], ss[1], raw) != (x == so[0] ? 1 : 2) ||
				    lvalue(bp, at, ss[0], ss[1], raw) != (y == so[0] ? 1 : 2) ||
				    lvalue(da, at, ss[0], ss[1], raw) != 0 ||
				    lvalue(db, at, ss[0], ss[1], raw) != 0 ||
				    lvalue(t, at, ss[0], ss[1], raw) != 0 ||
				    !is_error(raw, ss[0], ss[1]))
					continue;
				in[0] = ap;
				in[1] = bp;
				in[2] = da;
				in[3] = db;
				in[4] = t;
				if (!inner_only(in, 5))
					continue;
				eft_add(EF_2SUM, s, x, y, in, 5);
				return 1;
			}
		}
	}
	return 0;
}

/* Fast2Sum: s = x + y, z = s - x, t = y - z (exact when e(x) >= e(y)) */
static int
match_fast2sum(int t)
{
	int to[2], ts[2], k, m;

	if (!sum_ops(t, to, ts))
		return 0;
	for (k = 0; k < 2; k++) {
		int z = to[k], y = to[!k], zo[2], zs[2];
		if (!sum_ops(z, zo, zs))
			continue;
		for (m = 0; m < 2; m++) {
			int s = zo[m], x = zo[!m], so[2], ss[2], at[3], raw[3], in[2];
			if (x == y || !sum_ops(s, so, ss) || !pair_is(so, x, y))
				continue;
			at[0] = so[0];
			at[1] = so[1];
			at[2] = s;
			if (lvalue(z, at, ss[0], ss[1], raw) != (y == so[0] ? 1 : 2) ||
			    lvalue(t, at, ss[0], ss[1], raw) != 0 ||
			    !is_error(raw, ss[0], ss[1]))
				continue;
			in[0] = z;
			in[1] = t;
			if (!inner_only(in, 2))
				continue;
			eft_add(EF_FAST2SUM, s, x, y, in, 2);
			return 1;
		}
	}
	return 0;
}

/* 2Prod: p = x y, e = fma(x, y, -p) (exact while x y does not underflow) */
static int
match_2prod(int e)
{
	int sx = 1, sy = 1, sc = 1, s1 = 1, s2 = 1, x, y, p, m1, m2;

	if (e < 0 || !R.n[e].rounds || N.op[R.n[e].id] != O_FMA)
		return 0;
	x = strip(R.n[e].a, &sx);
	y = strip(R.n[e].b, &sy);
	p = strip(R.n[e].c, &sc);
	if (p < 0 || !R.n[p].rounds || N.op[R.n[p].id] != O_MUL)
		return 0;
	m1 = strip(R.n[p].a, &s1);
	m2 = strip(R.n[p].b, &s2);
	if (!((m1 == x && m2 == y) || (m1 == y && m2 == x)) ||
	    sc * s1 * s2 != -(sx * sy) || !inner_only(&e, 1))
		return 0;
	eft_add(EF_2PROD, p, x, y, &e, 1);
	return 1;
}

static void
eft_find(void)
{
	int i;

	/* 2Sum first: its da alone has Fast2Sum's shape */
	for (i = 0; i < R.nn; i++)
		if (!R.n[i].eft)
			match_2sum(i);
	for (i = 0; i < R.nn; i++)
		if (!R.n[i].eft && !match_fast2sum(i))
			match_2prod(i);
}

/* Fast2Sum: e(x) >= e(y); 2Prod: x y = 0 or |x y| >= 2^(emin + p) */
static int
eft_holds(const struct reft *e)
{
	ld x = R.n[e->x].v, y = R.n[e->y].v, p;
	int w = TYW(N.ty[R.n[e->s].id]);

	switch (e->kind) {
	case EF_FAST2SUM:
		return x == 0 || y == 0 || ilogbl(x) >= ilogbl(y);
	case EF_2PROD:
		p = fabsl(x * y);
		return p == 0 || p >= (w == 16 ? 0x1p-3L : w == 32 ? 0x1p-102L :
		    0x1p-969L);
	}
	return 1;
}

static const char *const eft_name[] = {"2Sum", "Fast2Sum", "2Prod"};

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

/* the first input where a precondition fails: which, on what, where */
static void
eft_fail(const struct func *g, const struct reft *e, const int64_t *in_i,
    const double *in_r, uint32_t k, struct rounderr *r)
{
	struct sbuf b = {0}, at = {0}, why = {0};
	int w = TYW(N.ty[R.n[e->s].id]);

	print_node(&b, R.n[e->s].id, NULL);
	fmt_point(&at, g, in_i, in_r, k);
	if (e->kind == EF_FAST2SUM)
		sb_puts(&why, "the operand taken first has the smaller exponent");
	else
		sb_printf(&why, "the product is below 2^%d in magnitude, where its "
		    "error need not be representable", w == 16 ? -3 : w == 32 ?
		    -102 : -969);
	snprintf(r->eftwhy, sizeof r->eftwhy, "%s on %.40s%s: %s, at %.60s",
	    eft_name[e->kind], sb_str(&b), strlen(sb_str(&b)) > 40 ? "..." :
	    "", sb_str(&why), sb_str(&at));
	free(why.p);
	free(b.p);
	free(at.p);
}

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
	int i, root, bad;

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
	eft_find();
	for (i = 0; i < R.ne; i++)
		r->eft[R.e[i].kind]++;
	for (i = 0; i < R.nn; i++) {
		r->nops += R.n[i].rounds && !R.n[i].eft;
		r->nexact += R.n[i].rounds && R.n[i].eft;
	}
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
			bad = 0;
			for (i = 0; i < R.ne; i++) {
				R.e[i].holds = eft_holds(&R.e[i]);
				if (!R.e[i].holds && !bad++ && !r->eftfail)
					eft_fail(g, &R.e[i], in_i, in_r, k, r);
			}
			r->eftfail += bad > 0;
			for (i = 0; i < R.nn; i++)
				if (R.n[i].rounds && !(R.n[i].eft &&
				    R.e[R.n[i].eft - 1].holds)) {
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
	free(R.e);
	r->ok = r->points > 0;
	if (!r->ok)
		snprintf(r->why, sizeof r->why, "no input of the sample gives a "
		    "finite, non-zero value");
}

/* which error-free transformations were taken as exact, and where not */
static void
eft_print(const struct rounderr *r, const char *ind, int e)
{
	static const char *const what[] = {
		"s + t = a + b, Knuth",
		"s + t = a + b while a, taken first, has an exponent at least b's, Dekker",
		"p + e = a b by an fma while |a b| >= 2^%d",
	};
	int k, n = 0;

	if (!r->eft[0] && !r->eft[1] && !r->eft[2])
		return;
	printf("%s  error-free transformations:", ind);
	for (k = 0; k < 3; k++) {
		if (!r->eft[k])
			continue;
		printf("%s %d %s (", n++ ? "," : "", r->eft[k], eft_name[k]);
		printf(what[k], e == 11 ? -3 : e == 24 ? -102 : -969);
		printf(")");
	}
	printf("; taken as exact, so the errors of the sums and products they "
	    "correct cancel to first order where the result adds the error "
	    "terms back; what is left is of order u^2, not in these lines "
	    "(`dword` derives it)\n");
	if (r->eftfail)
		printf("%s  a precondition fails at %llu of the %llu inputs, where "
		    "those operations count as rounded; the first: %s\n", ind,
		    (unsigned long long)r->eftfail, (unsigned long long)r->points,
		    r->eftwhy);
	else if (r->eft[1] || r->eft[2])
		printf("%s  the preconditions hold at every input of the sample "
		    "(checked, not proved)\n", ind);
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
	    "each taken as correctly rounded", ind, e, r->nops, r->nops == 1 ?
	    "" : "s");
	if (r->nexact)
		printf(", and %d in error-free transformations", r->nexact);
	printf("; relative to the value, the largest over %llu inputs\n",
	    (unsigned long long)r->points);
	eft_print(r, ind, e);
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
	if (r->eft[0] + r->eft[1] + r->eft[2]) {
		rep_int("error_free_transformations", r->eft[0] + r->eft[1] +
		    r->eft[2]);
		rep_int("exact_operations", r->nexact);
		rep_int("precondition_failures", (long long)r->eftfail);
	}
	rep_str("assumptions", r->eft[0] + r->eft[1] + r->eft[2] ? "first "
	    "order; operations correctly rounded; error-free transformations "
	    "(2Sum, Fast2Sum, 2Prod) exact as a whole where their "
	    "preconditions hold at the input; probable and expected: the "
	    "other rounding errors independent, mean zero (uniform for "
	    "expected), at one input" : "first order; operations correctly "
	    "rounded; probable and expected: rounding errors independent, "
	    "mean zero (uniform for expected), at one input");
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
	printf("%srounding of %s, first order, %d operation%s", ind, name,
	    r->nops, r->nops == 1 ? "" : "s");
	if (r->nexact)
		printf(" (and %d in error-free transformations, exact)",
		    r->nexact);
	printf(", relative, in u = 2^-%d: worst case %.3g", r->w == 16 ? 11 :
	    r->w == 32 ? 24 : 53, r->worst / u);
	if (r->prob < r->worst)
		printf(", probable %.3g (p >= %.5f if the errors are independent)",
		    r->prob / u, r->conf);
	else
		printf(", probable no better (under %g operations)", r->lambda *
		    r->lambda);
	printf(", expected %.3g; `error %s ~ F` says what each means\n",
	    r->rms / u, name);
}
