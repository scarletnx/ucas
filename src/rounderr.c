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
 *
 * Choices are where a small error can make a large one: floor, ceil and
 * a conversion to an integer jump by 1 at the integers, copysign by twice
 * its magnitude where its sign operand crosses 0, and min, max and ?:
 * take one side or the other.  The sweep follows what the exact values
 * do; at each input, where an operand is within its own first-order
 * error (a sweep rooted at it) of the point where the choice changes, the
 * worst case adds what the other side would bring, and the report counts
 * those inputs and names the first.  Exact operations stay exact: floor,
 * ceil, copysign, a float's bits read back as the float.
 *
 * Integers are exact too, and so are a float's bits as an integer when
 * the float carries no rounding error (an input, a constant, their
 * negation): and, or, xor, not and the shifts are worked out on the bits,
 * asfloat gives the float those bits are (the fast inverse square root's
 * first guess is a value like an input's).  What flows back through them
 * is the change a step of 1 in the integer makes, the larger of the two
 * sides, which is what a conversion that can go the other way moves it
 * by.
 *
 * A float plus 1.5 2^(p-1) (p the precision: 12582912 for an f32) is not
 * an error of u: while the float is within 2^(p-2) the sum is the
 * constant plus the float rounded to the nearest integer, ties to even,
 * exactly.  It is read as that: an exact value, and a choice at the half
 * integers.  A step choice (this one, floor, ceil, roundeven, a
 * conversion) that can go the other way adds what the other side gives:
 * the function run again from the choice on with the neighbouring
 * integer, not the jump times a slope, which a step of 1 is too large
 * for (2^k doubles while r = x - k ln 2 takes it back).
 *
 * The bits of a float that does carry rounding error are linear in it
 * inside a binade, one unit for each ulp.  They are followed to first
 * order where they only meet an add or a subtract of an exact integer (a
 * step of the exponent: the float times a power of two) or the sign bit
 * alone (and, or, xor with it or with nothing), and go back to a float;
 * anywhere else (a shift, a mask of other bits) the error moves them by
 * more than first order says, and the function is not analysed.
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
	int inexact;		/* it or something it reads rounds */
	int magic;		/* x + 1.5 2^(p-1): the operand's index + 1 */
	int mhold;		/* at this input it is the rounding to an
				 * integer (the operand within 2^(p-2)) */
	int unmagic;		/* that sum less the constant again, the
				 * integer itself: the sum's index + 1 */
	int frac;		/* the bits of a float with error, as a real:
				 * between two integers */
	ld slope;		/* asuint: bits for a unit of the float */
	int eft;		/* inside error-free transformation eft - 1 */
	int choice;		/* a choice (CH_*), or 0 */
	ld u;
	ld v;
};

/* what can go the other way (rounderr.h's struct rounderr counts these) */
enum { CH_NONE, CH_STEP, CH_CONV, CH_SIGN, CH_MINMAX, CH_SEL, CH_MAGIC };

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
	ld *adj, *adj2;		/* d result / d node; d node r / d node */
	const struct func *g;	/* the input the values are at */
	const int64_t *in_i;
	const double *in_r;
	uint32_t k;
} R;

static ld
unit(ty_t t)
{
	return TYW(t) == 16 ? 0x1p-11L : TYW(t) == 32 ? 0x1p-24L : 0x1p-53L;
}

static int frac_op(int op, ty_t t, ld a, ld b, ld *r);

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
	case O_ASFLOAT:
		/* a float's bits read back: the float itself */
		if ((N.op[N.a[e]] == O_ASUINT || N.op[N.a[e]] == O_ASINT) &&
		    N.ty[N.a[N.a[e]]] == t) {
			i = collect(N.a[N.a[e]]);
			if (i >= 0 && e < R.nix)
				R.ix[e] = i + 1;
			return i;
		}
		if (TYW(t) == 16 && !R.why[0])
			snprintf(R.why, sizeof R.why, "asfloat to an f16");
		a = collect(N.a[e]);
		break;
	case O_ASUINT: case O_ASINT:
		a = collect(N.a[e]);
		if (a >= 0 && !R.why[0] && TYW(N.ty[N.a[e]]) == 16)
			snprintf(R.why, sizeof R.why, "%s of an f16",
			    opinfo[N.op[e]].name);
		break;
	case O_NOT:
		a = collect(N.a[e]);
		break;
	case O_AND: case O_OR: case O_XOR: case O_SHL: case O_SHR:
		a = collect(N.a[e]);
		b = collect(N.b[e]);
		break;
	case O_CAST: case O_NEG: case O_ABS: case O_SQRT: case O_EXP:
	case O_LOG: case O_SIN: case O_COS: case O_TAN: case O_FLOOR:
	case O_CEIL: case O_ROUNDEVEN:
		a = collect(N.a[e]);
		break;
	case O_ADD: case O_SUB: case O_MUL: case O_DIV: case O_POW:
	case O_LT: case O_LE: case O_GT: case O_GE: case O_EQ: case O_NE:
	case O_MIN: case O_MAX: case O_COPYSIGN:
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
	switch (N.op[e]) {
	case O_FLOOR: case O_CEIL: case O_ROUNDEVEN: x->choice = CH_STEP; break;
	case O_CAST:
		if (!TY_FP(t) && TY_FP(N.ty[N.a[e]]))
			x->choice = CH_CONV;
		break;
	case O_COPYSIGN: x->choice = CH_SIGN; break;
	case O_MIN: case O_MAX: x->choice = CH_MINMAX; break;
	case O_SEL: x->choice = CH_SEL; break;
	}
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
	x->inexact = x->rounds || (a >= 0 && R.n[a].inexact) || (b >= 0 &&
	    R.n[b].inexact) || (c >= 0 && R.n[c].inexact);
	/* x + 1.5 2^(p-1): x rounded to an integer, an exact value */
	if (N.op[e] == O_ADD && TYK(t) == K_FLOAT && TYW(t) != 16) {
		ld M = TYW(t) == 32 ? 0x1.8p23L : 0x1.8p52L;
		int k;
		for (k = 0; k < 2; k++) {
			nid cst = k ? N.b[e] : N.a[e];
			if (N.op[cst] == O_CONST && TY_FP(N.ty[cst]) &&
			    (ld)N.kr[cst] == M) {
				x->magic = (k ? a : b) + 1;
				x->choice = CH_MAGIC;
				x->inexact = 0;
			}
		}
	}
	/* ... and that sum less the constant: the integer, exactly */
	if ((N.op[e] == O_SUB || N.op[e] == O_ADD) && TYK(t) == K_FLOAT &&
	    TYW(t) != 16 && a >= 0 && b >= 0) {
		ld M = TYW(t) == 32 ? 0x1.8p23L : 0x1.8p52L;
		int k;
		for (k = 0; k < 2; k++) {
			nid cst = k ? N.b[e] : N.a[e];
			int o = k ? a : b;
			/* j - M, or j + -M either way round */
			if (R.n[o].magic && N.op[cst] == O_CONST && TY_FP(N.ty[cst]) &&
			    (ld)N.kr[cst] == (N.op[e] == O_SUB ? M : -M) &&
			    (N.op[e] == O_ADD || k == 1)) {
				x->unmagic = o + 1;
				x->inexact = 0;
			}
		}
	}
	/* the bits of a float with error: only through an add or subtract
	 * of an exact integer or a bitwise operation with one (the sign
	 * bit's, checked at each input), and back to a float */
	if ((N.op[e] == O_ASUINT || N.op[e] == O_ASINT) && R.n[a].inexact)
		x->frac = 1;
	else if ((a >= 0 && R.n[a].frac) || (b >= 0 && R.n[b].frac) ||
	    (c >= 0 && R.n[c].frac)) {
		int both = a >= 0 && b >= 0 && R.n[a].frac && R.n[b].frac;
		switch (N.op[e]) {
		case O_AND: case O_OR: case O_XOR: {
			/* a constant has to leave all but the sign bit alone */
			nid cst = a >= 0 && R.n[a].frac ? N.b[e] : N.a[e];
			ld dummy;
			if (!both && N.op[cst] == O_CONST && !frac_op(N.op[e], t, 0,
			    (ld)N.ki[cst], &dummy)) {
				if (!R.why[0])
					snprintf(R.why, sizeof R.why, "the bits of a value "
					    "that carries rounding error, under a mask of "
					    "more than the sign bit");
				return -1;
			}
		}
			/* FALLTHROUGH */
		case O_ADD: case O_SUB:
			if (!both && TY_ISINT(t)) {
				x->frac = 1;
				break;
			}
			/* FALLTHROUGH */
		default:
			if (!R.why[0])
				snprintf(R.why, sizeof R.why, "the bits of a value that "
				    "carries rounding error, used as an integer by %s",
				    both ? "two of them at once" : opinfo[N.op[e]].name);
			return -1;
		case O_ASFLOAT:
			break;
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

	if (i < 0 || !R.n[i].rounds || R.n[i].magic)
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

/* an integer of type t from v: wrapped to its width as the machine does */
static ld
int_wrap(ty_t t, ld v)
{
	ld m, h;

	if (!TY_SIZED(t) || TYW(t) >= 64)
		return v;
	m = ldexpl(1, TYW(t));
	h = fmodl(v, m);
	if (h < 0)
		h += m;
	if (TYK(t) == K_SINT && h >= m / 2)
		h -= m;
	return h;
}

/* the 64 bits the machine holds for the integer v */
static uint64_t
int_bits(ld v)
{
	return v < 0 ? (uint64_t)(int64_t)v : (uint64_t)v;
}

/* the integer of type t those bits are */
static ld
bits_int(ty_t t, uint64_t v)
{
	int w = TYK(t) == K_UINT || TYK(t) == K_SINT ? TYW(t) : 64;

	if (w < 64)
		v &= (1ULL << w) - 1;
	if (TYK(t) == K_UINT)
		return (ld)v;
	if (w < 64 && (v >> (w - 1)))
		v |= ~0ULL << w;
	return (ld)(int64_t)v;
}

/* not, and, or, xor or a shift of integers, in type t; 0 when a shift's
 * count is outside the type */
static int
bit_op(int op, ty_t t, ld a, ld b, ld *r)
{
	uint64_t x = int_bits(a), y = int_bits(b), v;
	int w = TYK(t) == K_UINT || TYK(t) == K_SINT ? TYW(t) : 64;

	switch (op) {
	case O_NOT: v = ~x; break;
	case O_AND: v = x & y; break;
	case O_OR: v = x | y; break;
	case O_XOR: v = x ^ y; break;
	default:
		if (b < 0 || b >= w)
			return 0;
		if (op == O_SHL)
			v = x << y;
		else if (TYK(t) == K_UINT)
			v = (w < 64 ? x & ((1ULL << w) - 1) : x) >> y;
		else
			v = (uint64_t)((int64_t)x >> y);
	}
	*r = bits_int(t, v);
	return 1;
}

/* the bits of the float a of type tf, as an integer of type ti; 0 when a
 * is not a float of that type */
static int
float_bits(ty_t tf, ty_t ti, ld a, ld *r)
{
	if (TYW(tf) == 32) {
		float f = (float)a;
		uint32_t u;
		if ((ld)f != a)
			return 0;
		memcpy(&u, &f, sizeof u);
		*r = bits_int(ti, u);
	} else {
		double d = (double)a;
		uint64_t u;
		if ((ld)d != a)
			return 0;
		memcpy(&u, &d, sizeof u);
		*r = bits_int(ti, u);
	}
	return 1;
}

/* the float of type tf whose bits the integer a holds */
static ld
bits_float(ty_t tf, ld a)
{
	uint64_t v = int_bits(a);

	if (TYW(tf) == 32) {
		uint32_t u = (uint32_t)v;
		float f;
		memcpy(&f, &u, sizeof f);
		return f;
	} else {
		double d;
		memcpy(&d, &v, sizeof d);
		return d;
	}
}

/* of two changes, the larger in magnitude that is finite */
static ld
larger(ld p, ld q)
{
	if (!isfinite((double)p))
		p = 0;
	if (!isfinite((double)q))
		q = 0;
	return fabsl(p) >= fabsl(q) ? p : q;
}

/* what a step of 1 in operand a (or in b, when inb) of the bit operation
 * moves its value by: the larger of the step up and the step down */
static ld
bit_step(int op, ty_t t, ld a, ld b, int inb)
{
	ld v, up, dn;

	if (!bit_op(op, t, a, b, &v))
		return 0;
	if (!bit_op(op, t, inb ? a : a + 1, inb ? b + 1 : b, &up))
		up = v;
	if (!bit_op(op, t, inb ? a : a - 1, inb ? b - 1 : b, &dn))
		dn = v;
	return larger(up - v, v - dn);
}

/* d root / d node for every node up to root, into adj; 0 when one is not
 * finite */
static int
backprop(int root, ld *adj)
{
	int i;

	for (i = 0; i <= root; i++)
		adj[i] = 0;
	adj[root] = 1;
	for (i = root; i >= 0; i--) {
		struct rnode *x = &R.n[i];
		ld a = x->a >= 0 ? R.n[x->a].v : 0, b = x->b >= 0 ? R.n[x->b].v : 0;
		ld w = adj[i];
#define TO(j, d) do { if ((j) >= 0) adj[j] += w * (d); } while (0)
		if (w == 0)
			continue;
		switch (N.op[x->id]) {
		case O_CAST:
			/* a float to an integer: a step, nothing flows back */
			if (TY_FP(N.ty[x->id]) || !TY_FP(N.ty[N.a[x->id]]))
				TO(x->a, 1);
			break;
		case O_NEG: TO(x->a, -1); break;
		case O_ABS: TO(x->a, a < 0 ? -1 : 1); break;
		case O_SQRT: TO(x->a, 1 / (2 * x->v)); break;
		case O_EXP: TO(x->a, x->v); break;
		case O_LOG: TO(x->a, 1 / a); break;
		case O_SIN: TO(x->a, cosl(a)); break;
		case O_COS: TO(x->a, -sinl(a)); break;
		case O_TAN: TO(x->a, 1 + x->v * x->v); break;
		case O_ADD:
			/* a rounding to an integer: a step, nothing flows back */
			if (x->magic && x->mhold)
				break;
			TO(x->a, 1);
			TO(x->b, 1);
			break;
		case O_SUB: TO(x->a, 1); TO(x->b, -1); break;
		case O_MUL: TO(x->a, b); TO(x->b, a); break;
		case O_DIV: TO(x->a, 1 / b); TO(x->b, -a / (b * b)); break;
		case O_POW: TO(x->a, b * powl(a, b - 1)); break;
		case O_MIN: TO(a < b ? x->a : x->b, 1); break;
		case O_MAX: TO(a > b ? x->a : x->b, 1); break;
		case O_FMA: TO(x->a, b); TO(x->b, a); TO(x->c, 1); break;
		case O_SEL: TO(a != 0 ? x->b : x->c, 1); break;
		case O_COPYSIGN:
			TO(x->a, (signbit(a) ? -1 : 1) * (signbit(b) ? -1 : 1));
			break;
		case O_NOT: case O_AND: case O_OR: case O_XOR: case O_SHL:
		case O_SHR:
			/* the bits of a float with error pass as they are (only
			 * the sign bit is touched); an integer: what a step of 1
			 * moves the value by */
			if (x->frac) {
				int fa = x->a >= 0 && R.n[x->a].frac;
				TO(fa ? x->a : x->b, 1);
				TO(fa ? x->b : x->a, bit_step(N.op[x->id], N.ty[x->id],
				    floorl(fa ? a : b), fa ? b : a, 1));
				break;
			}
			TO(x->a, bit_step(N.op[x->id], N.ty[x->id], a, b, 0));
			TO(x->b, bit_step(N.op[x->id], N.ty[x->id], a, b, 1));
			break;
		case O_ASUINT: case O_ASINT:
			if (x->frac)
				TO(x->a, x->slope);
			break;
		case O_ASFLOAT: {
			ld fl = floorl(a);
			if (fl != a)	/* between two floats: their distance */
				TO(x->a, bits_float(N.ty[x->id], fl + 1) -
				    bits_float(N.ty[x->id], fl));
			else
				TO(x->a, larger(bits_float(N.ty[x->id], a + 1) - x->v,
				    x->v - bits_float(N.ty[x->id], a - 1)));
			break;
		}
		}
#undef TO
		if (!isfinite((double)adj[i]))
			return 0;
	}
	return 1;
}

/* the bits of the float a of type tf, which carries rounding error, as a
 * real: those of the nearest float and, beyond it, one unit for each ulp
 * (down, for a negative float); 0 at 0 and where it is not finite */
static int
frac_bits(ty_t tf, ty_t ti, ld a, ld *v, ld *slope)
{
	ld f, ulp, base;

	if (TYW(tf) == 32) {
		float x = (float)a, nb;
		uint32_t u;
		if (!isfinite(x) || x == 0)
			return 0;
		nb = nextafterf(x, a > x ? INFINITY : -INFINITY);
		memcpy(&u, &x, sizeof u);
		base = bits_int(ti, u);
		f = x;
		ulp = fabsl((ld)nb - f);
	} else {
		double x = (double)a, nb;
		uint64_t u;
		if (!isfinite(x) || x == 0)
			return 0;
		nb = nextafter(x, a > x ? INFINITY : -INFINITY);
		memcpy(&u, &x, sizeof u);
		base = bits_int(ti, u);
		f = x;
		ulp = fabsl((ld)nb - f);
	}
	if (!(ulp > 0) || !isfinite((double)ulp))
		return 0;
	*slope = (f < 0 ? -1 : 1) / ulp;
	*v = base + (a - f) * *slope;
	return 1;
}

/* and, or or xor of such bits (a, between two integers) with the exact
 * integer b: only when b leaves every bit but the sign as it is */
static int
frac_op(int op, ty_t t, ld a, ld b, ld *r)
{
	int w = TYK(t) == K_UINT || TYK(t) == K_SINT ? TYW(t) : 64;
	uint64_t all = w < 64 ? (1ULL << w) - 1 : ~0ULL, sign = 1ULL << (w - 1);
	uint64_t y = int_bits(b) & all;
	ld fl = floorl(a);

	if (op == O_AND ? (y != all && y != (all ^ sign)) : (y != 0 && y != sign))
		return 0;
	if (!bit_op(op, t, fl, b, r))
		return 0;
	*r += a - fl;
	return 1;
}

/* the value of node i from its operands' at the input R is at; 0 when it
 * is not finite */
static int
eval_node(int i)
{
	struct rnode *x = &R.n[i];
	ld a = x->a >= 0 ? R.n[x->a].v : 0, b = x->b >= 0 ? R.n[x->b].v : 0;
	ld c = x->c >= 0 ? R.n[x->c].v : 0;
	ty_t t = N.ty[x->id];

	switch (N.op[x->id]) {
	case O_CONST:
		x->v = TY_FP(t) ? (ld)N.kr[x->id] : (ld)N.ki[x->id];
		break;
	case O_VAR: x->v = param_val(R.g, x->id, R.in_i, R.in_r, R.k); break;
	case O_CAST:
		x->v = TY_FP(t) ? a : int_wrap(t, TY_FP(N.ty[N.a[x->id]]) ?
		    truncl(a) : a);
		break;
	case O_NEG: x->v = -a; break;
	case O_NOT: case O_AND: case O_OR: case O_XOR: case O_SHL:
	case O_SHR:
		if (x->frac) {
			int fa = x->a >= 0 && R.n[x->a].frac;
			if (N.op[x->id] == O_NOT || N.op[x->id] == O_SHL ||
			    N.op[x->id] == O_SHR || !frac_op(N.op[x->id], t, fa ? a :
			    b, fa ? b : a, &x->v))
				return 0;
		} else if (!bit_op(N.op[x->id], t, a, b, &x->v))
			return 0;
		break;
	case O_ASUINT: case O_ASINT:
		x->slope = 0;
		if (float_bits(N.ty[N.a[x->id]], t, a, &x->v))
			break;
		if (!x->frac || !frac_bits(N.ty[N.a[x->id]], t, a, &x->v,
		    &x->slope))
			return 0;
		break;
	case O_ASFLOAT: {
		ld fl = floorl(a);
		x->v = bits_float(t, fl);
		if (fl != a)	/* between two floats */
			x->v += (a - fl) * (bits_float(t, fl + 1) - x->v);
		break;
	}
	case O_ABS: x->v = fabsl(a); break;
	case O_SQRT: x->v = sqrtl(a); break;
	case O_EXP: x->v = expl(a); break;
	case O_LOG: x->v = logl(a); break;
	case O_SIN: x->v = sinl(a); break;
	case O_COS: x->v = cosl(a); break;
	case O_TAN: x->v = tanl(a); break;
	case O_FLOOR: x->v = floorl(a); break;
	case O_CEIL: x->v = ceill(a); break;
	case O_ROUNDEVEN: x->v = nearbyintl(a); break;
	case O_ADD:
		x->v = a + b;
		if (x->magic) {
			/* the operand within 2^(p-2): the constant plus it
			 * rounded to the nearest integer, exactly */
			ld o = R.n[x->magic - 1].v;
			x->mhold = fabsl(o) <= (TYW(t) == 32 ? 0x1p22L : 0x1p51L);
			if (x->mhold)
				x->v = x->v - o + nearbyintl(o);
		}
		break;
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
	case O_COPYSIGN: x->v = copysignl(a, b); break;
	}
	return isfinite((double)x->v);
}

/* values forward, adjoints backward; 0 when a value is not finite */
static int
sweep(const struct func *g, const int64_t *in_i, const double *in_r,
    uint32_t k)
{
	int i;

	R.g = g;
	R.in_i = in_i;
	R.in_r = in_r;
	R.k = k;
	for (i = 0; i < R.nn; i++)
		if (!eval_node(i))
			return 0;
	return backprop(R.nn - 1, R.adj);
}

/* the result with node d's value put at v and everything after it run
 * again; 0 when that is not finite.  The values are put back. */
static int
other_side(int d, ld v, ld *res)
{
	struct rnode *save = xmalloc((size_t)(R.nn - d) * sizeof *save);
	int i, ok = 1;

	memcpy(save, R.n + d, (size_t)(R.nn - d) * sizeof *save);
	R.n[d].v = v;
	for (i = d + 1; i < R.nn && ok; i++)
		ok = eval_node(i);
	if (ok)
		*res = R.n[R.nn - 1].v;
	memcpy(R.n + d, save, (size_t)(R.nn - d) * sizeof *save);
	free(save);
	return ok;
}

/* node i rounds at this input */
static int
rounds_here(int i)
{
	return R.n[i].rounds && !(R.n[i].eft && R.e[R.n[i].eft - 1].holds) &&
	    !(R.n[i].magic && R.n[i].mhold) && !(R.n[i].unmagic &&
	    R.n[R.n[i].unmagic - 1].mhold);
}

/* the first-order bound of node r's absolute error at this input: u sum
 * |z_i dr/dz_i| over the operations that round here */
static ld
node_err(int r)
{
	ld s = 0;
	int i;

	if (r < 0 || !backprop(r, R.adj2))
		return 0;
	for (i = 0; i <= r; i++)
		if (rounds_here(i))
			s += fabsl(R.adj2[i] * R.n[i].v) * R.n[i].u;
	return s;
}

static const char *const choice_name[] = {
	"", "floor, ceil or roundeven", "conversion to an integer", "copysign",
	"min or max", "select (?:)", "rounding written as an add",
};

/*
 * At this input, what choice d adds to the worst case when it can go
 * the other way (0 when it cannot); why says how, for the first.
 */
static ld
choice_extra(int d, char *why, size_t wn)
{
	struct rnode *x = &R.n[d];
	ld w = fabsl(R.adj[d]), va = x->a >= 0 ? R.n[x->a].v : 0;
	ld vb = x->b >= 0 ? R.n[x->b].v : 0, e, ea, eb, k;
	int p, q, cop;

	switch (x->choice) {
	case CH_STEP: case CH_CONV: case CH_MAGIC: {
		/* a jump of 1 at each integer (a conversion: not at 0;
		 * roundeven and the rounding add: at each half integer) */
		int o = x->magic ? x->magic - 1 : x->a;
		int half = x->magic || N.op[x->id] == O_ROUNDEVEN;
		ld cur = x->v, off = 0, other, res;
		if (x->magic && !x->mhold)
			return 0;
		va = R.n[o].v;
		if (x->magic) {
			off = cur - nearbyintl(va);
			cur -= off;
		}
		k = half ? floorl(va) + 0.5L : nearbyintl(va);
		if (x->choice == CH_CONV && k == 0)
			return 0;
		e = node_err(o);
		if (e <= 0 || fabsl(va - k) > e)
			return 0;
		snprintf(why, wn, "the operand of %s, %.6Lg, is within its error "
		    "%.3Lg of %.6Lg", x->magic ? "a rounding written as an add" :
		    x->choice == CH_STEP ? opinfo[N.op[x->id]].name :
		    "a conversion to an integer", va, e, k);
		/* what the other side gives: the neighbouring integer, and
		 * everything after it run again */
		if (half)
			other = cur > k ? cur - 1 : cur + 1;
		else if (N.op[x->id] == O_CEIL || (x->choice == CH_CONV && va < 0))
			other = cur > k ? k : k + 1;
		else
			other = cur >= k ? k - 1 : k;
		if (other_side(d, other + off, &res))
			return fabsl(res - R.n[R.nn - 1].v);
		return w;
	}
	case CH_SIGN:
		e = node_err(x->b);
		if (e <= 0 || fabsl(vb) > e || va == 0)
			return 0;
		snprintf(why, wn, "copysign's sign operand %.3Lg is within its "
		    "error %.3Lg of 0", vb, e);
		return w * 2 * fabsl(va);
	case CH_MINMAX:
		ea = node_err(x->a);
		eb = node_err(x->b);
		if (ea + eb <= 0 || fabsl(va - vb) > ea + eb)
			return 0;
		snprintf(why, wn, "%s's sides %.6Lg and %.6Lg are closer than "
		    "their errors %.3Lg + %.3Lg", opinfo[N.op[x->id]].name, va, vb,
		    ea, eb);
		/* the side the exact values do not take, and its own error */
		return w * (fabsl(va - vb) + ((N.op[x->id] == O_MIN) == (va < vb) ?
		    eb : ea));
	case CH_SEL:
		cop = N.op[R.n[x->a].id];
		if (cop != O_LT && cop != O_LE && cop != O_GT && cop != O_GE &&
		    cop != O_EQ && cop != O_NE)
			return 0;
		p = R.n[x->a].a;
		q = R.n[x->a].b;
		if (p < 0 || q < 0 || !TY_FP(N.ty[R.n[p].id]))
			return 0;
		ea = node_err(p);
		eb = node_err(q);
		if (ea + eb <= 0 || fabsl(R.n[p].v - R.n[q].v) > ea + eb)
			return 0;
		snprintf(why, wn, "the condition of ?: compares %.6Lg and %.6Lg, "
		    "closer than their errors %.3Lg + %.3Lg", R.n[p].v, R.n[q].v,
		    ea, eb);
		vb = R.n[x->b].v;
		k = R.n[x->c].v;
		return w * (fabsl(vb - k) + node_err(va != 0 ? x->c : x->b));
	}
	return 0;
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
	int i, root, bad, flip;

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
		int op = N.op[R.n[i].id];
		r->nops += R.n[i].rounds && !R.n[i].eft && !R.n[i].magic &&
		    !R.n[i].unmagic;
		r->nexact += R.n[i].rounds && R.n[i].eft;
		r->choices[R.n[i].choice]++;
		r->nbits += op == O_ASUINT || op == O_ASINT || op == O_ASFLOAT;
	}
	R.adj = xcalloc((size_t)R.nn, sizeof *R.adj);
	R.adj2 = xcalloc((size_t)R.nn, sizeof *R.adj2);
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
				if (rounds_here(i)) {
					ld c = fabsl(R.adj[i] * R.n[i].v) * R.n[i].u;
					s1 += c;
					s2 += c * c;
				}
			/* a choice that can go the other way: in the worst case */
			flip = 0;
			for (i = 0; i < R.nn; i++)
				if (R.n[i].choice) {
					char why[200];
					ld x = choice_extra(i, why, sizeof why);
					if (x > 0 && !flip++ && !r->flips) {
						struct sbuf at = {0};
						fmt_point(&at, g, in_i, in_r, k);
						snprintf(r->flipwhy, sizeof r->flipwhy, "%s, at "
						    "%.60s", why, sb_str(&at));
						free(at.p);
					}
					s1 += x;
				}
			if (!isfinite((double)s1))
				continue;
			r->flips += flip > 0;
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
	free(R.adj);
	free(R.adj2);
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

/* the choices, and whether one can go the other way */
static void
choice_print(const struct rounderr *r, const char *ind)
{
	int k, n = 0;

	for (k = 1; k < 7; k++)
		if (r->choices[k])
			printf("%s %d %s", n++ ? "," : "", r->choices[k],
			    choice_name[k]);
	if (!n)
		return;
	if (r->flips)
		printf("; at %llu of the %llu inputs one can go the other way, "
		    "and the worst case adds what the other side brings there "
		    "(the probable and expected lines do not); the first: %s\n",
		    (unsigned long long)r->flips, (unsigned long long)r->points,
		    r->flipwhy);
	else
		printf("; at no input of the sample is one within its operands' "
		    "error of going the other way\n");
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
	if (r->choices[1] + r->choices[2] + r->choices[3] + r->choices[4] +
	    r->choices[5] + r->choices[6]) {
		printf("%s  choices:", ind);
		choice_print(r, ind);
	}
	if (r->nbits)
		printf("%s  bit casts: %d, exact: integers are, and a float "
		    "whose bits are read carries no rounding error\n", ind,
		    r->nbits);
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
	if (r->nops > 1 && r->nops * u < 1)
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
	if (r->choices[1] + r->choices[2] + r->choices[3] + r->choices[4] +
	    r->choices[5] + r->choices[6]) {
		rep_int("choices", r->choices[1] + r->choices[2] + r->choices[3] +
		    r->choices[4] + r->choices[5] + r->choices[6]);
		rep_int("choice_flips", (long long)r->flips);
	}
	if (r->nbits)
		rep_int("bit_casts", r->nbits);
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
	printf(", expected %.3g", r->rms / u);
	if (r->flips)
		printf(", a choice can go the other way at %llu inputs (in the "
		    "worst case)", (unsigned long long)r->flips);
	printf("; `error %s ~ F` says what each means\n", name);
}
