/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * fpfold: conversion round trips a driver folds.
 *
 * A cost model can say which float round trips T -> M -> T its driver
 * drops (`folds f32 -> f16 -> f32` in the model's block): the narrowing
 * conversion and the one back are then priced as the driver runs them,
 * at nothing, and the rounding to M does not happen.  A program whose
 * values are not all M values relies on that rounding (quantisation), so
 * `cost` says where the folded program differs, with an input, and
 * `foldproof F` gives the rounding as integer arithmetic on the mantissa
 * bits, which no float fold reaches.  It is proved equal to the IEEE
 * round trip (to nearest, ties to even, subnormal results, overflow to
 * infinity): the grid argument in words, the add-and-mask rounding
 * against its definition by SAT on every bit pattern.
 *
 * Nothing names a driver here: a model that declares no fold prices the
 * conversions as written and foldproof has nothing to change under it.
 */
#include "dword.h"
#include "fuse.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int
model_folds(const struct model *m, ty_t from, ty_t mid)
{
	int i;

	for (i = 0; i < m->nfold; i++)
		if (m->fold[i][0] == from && m->fold[i][1] == mid)
			return 1;
	return 0;
}

/* n is T(M(x)) with x of float type T and M a narrower float */
static int
round_trip(nid n, nid *x)
{
	nid c;

	if (N.op[n] != O_CAST || TYK(N.ty[n]) != K_FLOAT)
		return 0;
	c = N.a[n];
	if (N.op[c] != O_CAST || TYK(N.ty[c]) != K_FLOAT ||
	    N.ty[N.a[c]] != N.ty[n] || TYW(N.ty[c]) >= TYW(N.ty[n]))
		return 0;
	if (x)
		*x = N.a[c];
	return 1;
}

/* the nodes under root, in[] set; tables through their lists */
static void
reach_set(nid root, uint8_t *in, uint32_t lim)
{
	nid *st = xmalloc((size_t)lim * sizeof *st), n;
	uint32_t sp = 0, j;

	if (root == 0 || root >= lim) {
		free(st);
		return;
	}
	st[sp++] = root;
	in[root] = 1;
	while (sp) {
		n = st[--sp];
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF)
			continue;
		if (N.op[n] == O_TABLE) {
			for (j = 0; j < N.b[n]; j++) {
				nid e = N.lst[N.a[n] + j];
				if (e < lim && !in[e])
					in[e] = 1, st[sp++] = e;
			}
			continue;
		}
#define PUSH(c) do { \
	if ((c) && (c) < lim && !in[c]) \
		in[c] = 1, st[sp++] = (c); \
} while (0)
		PUSH(N.a[n]);
		PUSH(N.b[n]);
		PUSH(N.c[n]);
#undef PUSH
	}
	free(st);
}

int
fold_marks(const struct model *m, nid root, uint8_t *out, uint32_t lim)
{
	uint8_t *in;
	uint32_t *tot, *fol, g;
	int count = 0;

	if (m->nfold == 0)
		return 0;
	in = xcalloc(lim, 1);
	tot = xcalloc(lim, sizeof *tot);
	fol = xcalloc(lim, sizeof *fol);
	reach_set(root, in, lim);
	for (g = 1; g < lim; g++) {
		int rt;
		if (!in[g] || N.op[g] == O_CONST || N.op[g] == O_VAR ||
		    N.op[g] == O_UNDEF || N.op[g] == O_TABLE)
			continue;
		rt = round_trip(g, NULL) && model_folds(m, N.ty[g],
		    N.ty[N.a[g]]);
		if (N.a[g] && N.a[g] < lim) {
			tot[N.a[g]]++;
			fol[N.a[g]] += (uint32_t)rt;
		}
		if (N.b[g] && N.b[g] < lim)
			tot[N.b[g]]++;
		if (N.c[g] && N.c[g] < lim)
			tot[N.c[g]]++;
	}
	for (g = 1; g < lim; g++)
		if (in[g] && round_trip(g, NULL) && model_folds(m, N.ty[g],
		    N.ty[N.a[g]])) {
			out[g] = 1;
			count++;
			/* the way there too, when nothing else reads it */
			if (tot[N.a[g]] == fol[N.a[g]])
				out[N.a[g]] = 1;
		}
	free(in);
	free(tot);
	free(fol);
	return count;
}

/* root with every round trip m's driver folds replaced by its operand */
static nid *fa_memo;
static uint32_t fa_n;
static const struct model *fa_m;

static nid
fa_rebuild(nid n)
{
	nid r, x, *e;
	uint32_t i;

	if (n == 0 || n >= fa_n)
		return n;
	if (fa_memo[n])
		return fa_memo[n];
	if (round_trip(n, &x) && model_folds(fa_m, N.ty[n], N.ty[N.a[n]]))
		return fa_memo[n] = fa_rebuild(x);
	switch (N.op[n]) {
	case O_CONST: case O_UNDEF: case O_VAR: case O_LOOP:
		r = n;
		break;
	case O_TABLE:
		e = xmalloc(N.b[n] * sizeof *e + 1);
		for (i = 0; i < N.b[n]; i++)
			e[i] = fa_rebuild(N.lst[N.a[n] + i]);
		r = mk_table(e, N.b[n]);
		free(e);
		break;
	default: {
		nid a = fa_rebuild(N.a[n]), b = fa_rebuild(N.b[n]);
		nid c = fa_rebuild(N.c[n]);
		/* untouched below: the node itself (atomics and barriers are
		 * never rebuilt: each is its own effect) */
		r = a == N.a[n] && b == N.b[n] && c == N.c[n] ? n :
		    N.op[n] == O_ATOMIC || N.op[n] == O_BARRIER ? n :
		    N.op[n] == O_CAST ? mk_cast(N.ty[n], a) :
		    N.op[n] == O_INDEX ? mk_index(a, b) : mk_op(N.op[n], a, b, c);
	}
	}
	return fa_memo[n] = r;
}

nid
fold_apply(const struct model *m, nid root)
{
	nid r;

	if (m->nfold == 0)
		return root;
	fa_n = N.n;
	fa_memo = xcalloc(fa_n, sizeof *fa_memo);
	fa_m = m;
	r = fa_rebuild(root);
	free(fa_memo);
	fa_memo = NULL;
	return r;
}

/* ---- the formats ---- */

struct ffmt {
	int w, mant, bias, emin, emax;
};

static struct ffmt
fmt_of(ty_t t)
{
	struct ffmt f;

	f.w = TYW(t);
	f.mant = f.w == 16 ? 10 : f.w == 32 ? 23 : 52;
	f.bias = f.w == 16 ? 15 : f.w == 32 ? 127 : 1023;
	f.emin = 1 - f.bias;
	f.emax = f.bias;
	return f;
}

/* every value of [lo, hi] is 0 or a normal M value's neighbourhood: the
 * rounding to M neither overflows nor lands on a subnormal */
static int
normal_range(const struct ffmt *M, double lo, double hi)
{
	double mn = ldexp(1.0, M->emin);
	double ovf = ldexp(2.0 - ldexp(1.0, -(M->mant + 1)), M->emax);
	double a = fabs(lo) > fabs(hi) ? fabs(lo) : fabs(hi);

	if (!(a < ovf))
		return 0;
	if (lo <= 0 && hi >= 0)
		return lo == 0 && hi == 0;
	return (lo > 0 ? lo : -hi) >= mn;
}

static nid
ku(ty_t t, uint64_t v)
{
	return mk_const_i(t, (int64_t)v);
}

/* round-to-nearest-even of the bits b to a multiple of 2^D: add and mask */
static nid
rne_mask(nid b, int D)
{
	ty_t t = N.ty[b];
	uint64_t half1 = ((uint64_t)1 << (D - 1)) - 1;
	uint64_t mask = ~(((uint64_t)1 << D) - 1);
	nid odd = mk_bin(O_AND, mk_bin(O_SHR, b, ku(t, (uint64_t)D)), ku(t, 1));
	nid s = mk_bin(O_ADD, mk_bin(O_ADD, b, ku(t, half1)), odd);

	return mk_bin(O_AND, s, ku(t, mask & (TYW(t) == 64 ? ~(uint64_t)0 :
	    (((uint64_t)1 << TYW(t)) - 1))));
}

/* its definition: the quotient, up when the remainder is past a half or
 * at it with an odd quotient */
static nid
rne_spec(nid b, int D)
{
	ty_t t = N.ty[b];
	nid q = mk_bin(O_SHR, b, ku(t, (uint64_t)D));
	nid rem = mk_bin(O_AND, b, ku(t, ((uint64_t)1 << D) - 1));
	nid half = ku(t, (uint64_t)1 << (D - 1));
	nid up = mk_bin(O_LOR, mk_bin(O_GT, rem, half), mk_bin(O_LAND,
	    mk_bin(O_EQ, rem, half), mk_bin(O_EQ, mk_bin(O_AND, q, ku(t, 1)),
	    ku(t, 1))));

	return mk_bin(O_SHL, mk_bin(O_ADD, q, mk_cast(t, up)),
	    ku(t, (uint64_t)D));
}

/* the same with a variable count k, giving the quotient (subnormals) */
static nid
rneq_form(nid sig, nid k)
{
	ty_t t = N.ty[sig];
	nid half1 = mk_bin(O_SUB, mk_bin(O_SHL, ku(t, 1), mk_bin(O_SUB, k,
	    ku(t, 1))), ku(t, 1));
	nid odd = mk_bin(O_AND, mk_bin(O_SHR, sig, k), ku(t, 1));

	return mk_bin(O_SHR, mk_bin(O_ADD, mk_bin(O_ADD, sig, half1), odd), k);
}

static nid
rneq_spec(nid sig, nid k)
{
	ty_t t = N.ty[sig];
	nid q = mk_bin(O_SHR, sig, k);
	nid one = mk_bin(O_SHL, ku(t, 1), k);
	nid rem = mk_bin(O_AND, sig, mk_bin(O_SUB, one, ku(t, 1)));
	nid half = mk_bin(O_SHL, ku(t, 1), mk_bin(O_SUB, k, ku(t, 1)));
	nid up = mk_bin(O_LOR, mk_bin(O_GT, rem, half), mk_bin(O_LAND,
	    mk_bin(O_EQ, rem, half), mk_bin(O_EQ, mk_bin(O_AND, q, ku(t, 1)),
	    ku(t, 1))));

	return mk_bin(O_ADD, q, mk_cast(t, up));
}

/*
 * T(M(x)) on x's bits.  Normal results: the M values of a binade are the
 * T values whose D = mant(T) - mant(M) low mantissa bits are zero, and the
 * bit pattern is monotone in |x|, so rounding the pattern to a multiple
 * of 2^D rounds the value (a carry out of the mantissa is the next
 * binade's first value).  Full: results at or past 2^(emax + 1) are
 * infinity, a NaN stays, and below 2^emin the grid is fixed at 2^(emin -
 * mant(M)): the significand loses k = (emin exponent field + D) - e bits
 * and the quotient q is the value q 2^(emin - mant(M)).
 */
static nid
form_of(nid x, ty_t T, ty_t Mt, int full)
{
	struct ffmt F = fmt_of(T), M = fmt_of(Mt);
	ty_t u = TY(K_UINT, F.w);
	int D = F.mant - M.mant;
	uint64_t signbit = (uint64_t)1 << (F.w - 1);
	uint64_t inf = (uint64_t)(2 * F.bias + 1) << F.mant;
	uint64_t ovf = (uint64_t)(F.bias + M.emax + 1) << F.mant;
	uint64_t en = (uint64_t)(F.bias + M.emin);	/* first normal M */
	nid b = mk_un(O_ASUINT, x), m, e, r, k, sig, q, sub, mag;

	if (!full)
		return mk_un(O_ASFLOAT, rne_mask(b, D));
	m = mk_bin(O_AND, b, ku(u, signbit - 1));
	e = mk_bin(O_SHR, m, ku(u, (uint64_t)F.mant));
	r = rne_mask(m, D);
	r = mk_sel(mk_bin(O_GE, r, ku(u, ovf)), ku(u, inf), r);
	r = mk_sel(mk_bin(O_GT, m, ku(u, inf)), m, r);
	/* k = min(en + D - e, mant(T) + 2): more bits than there are give 0 */
	k = mk_bin(O_MIN, mk_bin(O_SUB, ku(u, en + (uint64_t)D), e),
	    ku(u, (uint64_t)F.mant + 2));
	sig = mk_bin(O_OR, mk_bin(O_AND, m, ku(u, ((uint64_t)1 << F.mant) - 1)),
	    ku(u, (uint64_t)1 << F.mant));
	q = rneq_form(sig, k);
	sub = mk_un(O_ASUINT, mk_bin(O_MUL, mk_cast(T, q), mk_cast(T,
	    mk_const_r(ldexp(1.0, M.emin - M.mant)))));
	mag = mk_sel(mk_bin(O_LT, e, ku(u, en)), sub, r);
	return mk_un(O_ASFLOAT, mk_bin(O_OR, mag, mk_bin(O_AND, b,
	    ku(u, signbit))));
}

nid
fold_proof_form(nid x, int known, double lo, double hi, int *full)
{
	ty_t T = N.ty[x];
	struct ffmt M = fmt_of(TY_F16);

	*full = !(known && normal_range(&M, lo, hi));
	return form_of(x, T, TY_F16, *full);
}

/* ---- does the program rely on the rounding ---- */

enum { Q_SAME_PROVED, Q_SAME_ALL, Q_SAME_SAMPLED, Q_DIFFERS };

/*
 * x's values against T(M(x)): proved the same (an integer of at most
 * mant(M) + 1 bits, a value that came from M), the same on every point of
 * a small domain, the same on a sample, or different (with the point).
 */
static int
quantises(const struct func *f, nid n, nid x, struct sbuf *wit,
    uint64_t *points, const char **why)
{
	struct ffmt M = fmt_of(N.ty[N.a[n]]);
	struct plan p;
	struct diter d;
	nid roots[3];
	int64_t *in_i;
	double *in_r, lo, hi;
	uint8_t ok[CH];
	uint32_t np, k, lx, ln, wloc;
	int res;

	*points = 0;
	*why = NULL;
	if (N.op[x] == O_CAST && N.ty[N.a[x]] == N.ty[N.a[n]]) {
		*why = "its operand was converted from that type";
		return Q_SAME_PROVED;
	}
	if (N.op[x] == O_CAST && TY_ISINT(N.ty[N.a[x]]) &&
	    dword_interval(f, N.a[x], &lo, &hi) &&
	    fabs(lo) <= ldexp(1.0, M.mant + 1) &&
	    fabs(hi) <= ldexp(1.0, M.mant + 1)) {
		*why = "its operand is an integer that fits the significand";
		return Q_SAME_PROVED;
	}
	roots[0] = x;
	roots[1] = n;
	roots[2] = f->where;
	plan_build(&p, roots, 3, 0);
	wloc = f->where ? plan_loc(&p, f->where) : NOTOK;
	lx = plan_loc(&p, x);
	ln = plan_loc(&p, n);
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	diter_init(&d, f, 1 << 16, 257, 4096);
	res = d.exhaustive ? Q_SAME_ALL : Q_SAME_SAMPLED;
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wloc, np, ok);
		for (k = 0; k < np; k++) {
			double a = COLR(&p, lx)[k], b = COLR(&p, ln)[k];
			if (!ok[k] || isnan(a))
				continue;
			(*points)++;
			if (a != b && res != Q_DIFFERS) {
				res = Q_DIFFERS;
				fmt_point(wit, f, in_i, in_r, k);
				sb_printf(wit, ": the program expects %.9g, the "
				    "folded one gives %.9g", b, a);
			}
		}
		if (res == Q_DIFFERS)
			break;
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return res;
}

static const char *
trip_name(nid n)
{
	static char b[64];

	snprintf(b, sizeof b, "%s -> ", ty_str(N.ty[n]));
	snprintf(b + strlen(b), sizeof b - strlen(b), "%s -> ",
	    ty_str(N.ty[N.a[n]]));
	snprintf(b + strlen(b), sizeof b - strlen(b), "%s", ty_str(N.ty[n]));
	return b;
}

void
fold_cost_notes(const struct func *f, const struct model *m)
{
	uint32_t lim = N.n, g;
	uint8_t *fo;
	int any = 0;

	if (m->nfold == 0)
		return;
	fo = xcalloc(lim, 1);
	if (fold_marks(m, f->body, fo, lim) == 0) {
		free(fo);
		return;
	}
	rep_list("driver_folds");
	for (g = 1; g < lim; g++) {
		struct sbuf wit = {0};
		uint64_t pts;
		const char *why;
		nid x;
		int q;

		if (!fo[g] || !round_trip(g, &x) || !model_folds(m, N.ty[g],
		    N.ty[N.a[g]]))
			continue;
		any = 1;
		printf("  the driver folds %s (model %s): %s is priced 0 and "
		    "computes %s itself\n", trip_name(g), sym_str(m->name),
		    shared_str(g, 80), shared_str(x, 60));
		q = quantises(f, g, x, &wit, &pts, &why);
		rep_obj(NULL);
		rep_str("round_trip", trip_name(g));
		rep_expr("expression", g);
		rep_bool("relies_on_rounding", q == Q_DIFFERS);
		switch (q) {
		case Q_SAME_PROVED:
			printf("      nothing is lost: %s, so the round trip never "
			    "rounds\n", why);
			break;
		case Q_SAME_ALL:
			printf("      nothing is lost: on all %llu points of the "
			    "domain the round trip returns its operand\n",
			    (unsigned long long)pts);
			break;
		case Q_SAME_SAMPLED:
			printf("      no difference on %llu sampled points (not a "
			    "proof that none exists)\n", (unsigned long long)pts);
			break;
		default:
			printf("      warning: %s relies on that rounding "
			    "(quantisation): at %s\n      `foldproof %s` gives "
			    "the rounding in a form no float fold reaches\n",
			    f->anon ? "the expression" : sym_str(f->name),
			    sb_str(&wit), f->anon ? "F" : sym_str(f->name));
			rep_str("witness", sb_str(&wit));
		}
		rep_end();
		free(wit.p);
	}
	rep_end();
	(void)any;
	free(fo);
}

/* ---- foldproof ---- */

static nid *rmemo, rfrom[16], rto[16];
static uint32_t rmemon;
static int nrep;

static nid
replace(nid n)
{
	nid r, *e;
	uint32_t i;
	int j;

	if (n == 0 || n >= rmemon)
		return n;
	if (rmemo[n])
		return rmemo[n];
	for (j = 0; j < nrep; j++)
		if (rfrom[j] == n)
			return rmemo[n] = rto[j];
	switch (N.op[n]) {
	case O_CONST: case O_UNDEF: case O_VAR: case O_LOOP:
		r = n;
		break;
	case O_TABLE:
		e = xmalloc(N.b[n] * sizeof *e + 1);
		for (i = 0; i < N.b[n]; i++)
			e[i] = replace(N.lst[N.a[n] + i]);
		r = mk_table(e, N.b[n]);
		free(e);
		break;
	case O_SEL:
		r = mk_sel(replace(N.a[n]), replace(N.b[n]), replace(N.c[n]));
		break;
	case O_CAST:
		r = mk_cast(N.ty[n], replace(N.a[n]));
		break;
	case O_INDEX:
		r = mk_index(replace(N.a[n]), replace(N.b[n]));
		break;
	case O_BARRIER:
		r = mk_barrier(replace(N.a[n]));
		break;
	case O_ATOMIC:
		r = mk_atomic((int)(N.ki[n] & 0xff), replace(N.a[n]),
		    replace(N.b[n]));
		break;
	case O_FMA:
		r = mk_fma(replace(N.a[n]), replace(N.b[n]), replace(N.c[n]));
		break;
	default:
		if (opinfo[N.op[n]].arity == 1)
			r = mk_un(N.op[n], replace(N.a[n]));
		else
			r = mk_bin(N.op[n], replace(N.a[n]), replace(N.b[n]));
	}
	return rmemo[n] = r;
}

/* the add-and-mask rounding against its definition, on every pattern */
static int
prove_rne(ty_t u, int D, int *how)
{
	struct func t;
	struct sbuf w = {0};
	struct eqproof pr;
	nid b;
	int bad;

	memset(&t, 0, sizeof t);
	t.name = sym_cstr("_rne");
	t.anon = 1;
	t.np = 1;
	t.p[0].name = sym_cstr("b");
	t.p[0].ty = u;
	b = mk_var(0, t.p[0].name, u);
	t.body = rne_spec(b, D);
	bad = equiv_decide(&t, t.body, rne_mask(b, D), &w, &pr);
	*how = pr.how;
	free(w.p);
	return !bad && pr.how != EQ_SAMPLED;
}

static int
prove_rneq(ty_t u, int mant, int klo, int khi, int *how)
{
	struct func t;
	struct sbuf w = {0};
	struct eqproof pr;
	nid sig, k;
	int bad;

	memset(&t, 0, sizeof t);
	t.name = sym_cstr("_rneq");
	t.anon = 1;
	t.np = 2;
	t.p[0].name = sym_cstr("sig");
	t.p[0].ty = u;
	t.p[0].ranged = 1;
	t.p[0].lo = 0;
	t.p[0].hi = ((int64_t)1 << (mant + 1)) - 1;
	t.p[1].name = sym_cstr("k");
	t.p[1].ty = u;
	t.p[1].ranged = 1;
	t.p[1].lo = klo;
	t.p[1].hi = khi;
	sig = mk_var(0, t.p[0].name, u);
	k = mk_var(1, t.p[1].name, u);
	t.body = rneq_spec(sig, k);
	bad = equiv_decide(&t, t.body, rneq_form(sig, k), &w, &pr);
	*how = pr.how;
	free(w.p);
	return !bad && pr.how != EQ_SAMPLED;
}

/*
 * A secondary check, not the proof: the form against the evaluator's
 * round trip on bit patterns of every binade from elo to ehi, both
 * signs: ties and their neighbours at every mantissa bit, and 64 others;
 * inside [lo, hi] only when the form is for that range.
 */
static long
pattern_check(nid x, nid trip, nid form, int slot, int elo, int ehi,
    int ranged, double lo, double hi, long *bad)
{
	struct ffmt F = fmt_of(N.ty[x]);
	struct plan p;
	nid roots[2];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint64_t rng = 0x9e3779b97f4a7c15ULL, mm = ((uint64_t)1 << F.mant) - 1;
	uint64_t pat[5 * 52 + 66];
	uint32_t lt, lf, np = 0, k;
	long n = 0;
	int e, s, t, i, npat = 0, last;

	for (t = 0; t < F.mant; t++) {
		uint64_t one = (uint64_t)1 << t;
		pat[npat++] = one - 1;
		pat[npat++] = one;
		pat[npat++] = one + 1;
		pat[npat++] = (one | (one << 1)) & mm;
		pat[npat++] = (((one << 1) - 1) | (one << 2)) & mm;
	}
	for (i = 0; i < 64; i++) {
		rng ^= rng << 13;
		rng ^= rng >> 7;
		rng ^= rng << 17;
		pat[npat++] = rng & mm;
	}
	pat[npat++] = 0;
	pat[npat++] = mm;
	roots[0] = trip;
	roots[1] = form;
	plan_build(&p, roots, 2, 0);
	lt = plan_loc(&p, trip);
	lf = plan_loc(&p, form);
	*bad = 0;
	for (s = 0; s < 2; s++)
		for (e = elo; e <= ehi; e++)
			for (i = 0; i < npat; i++) {
				double v = frombits(F.w, ((uint64_t)s << (F.w - 1)) |
				    ((uint64_t)e << F.mant) | pat[i]);
				last = s == 1 && e == ehi && i == npat - 1;
				if (!ranged || (v >= lo && v <= hi))
					in_r[(size_t)slot * CH + np++] = v;
				if (np < CH && !(last && np))
					continue;
				plan_run(&p, np, in_i, in_r);
				for (k = 0; k < np; k++) {
					double a = COLR(&p, lt)[k];
					double b = COLR(&p, lf)[k];
					n++;
					if (isnan(a) ? !isnan(b) : fbits(F.w, a) !=
					    fbits(F.w, b))
						(*bad)++;
				}
				np = 0;
			}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return n;
}

static double
cost_under(const struct model *m, nid root)
{
	return (double)fuse_expr_cost(m, root);
}

/* as the driver runs it: the folded round trips gone */
static double
cost_folded(const struct model *m, nid root)
{
	return (double)fuse_expr_cost(m, fold_apply(m, root));
}

static const char *
how_text(int how)
{
	return how == EQ_SAT ? "proved by SAT (both sides bit-blasted)" :
	    how == EQ_EXHAUSTIVE ? "every case enumerated" :
	    "sampled only: NOT proved";
}

/*
 * One folded round trip g of f, reported: whether the program relies on
 * its rounding (with an input), the fold-proof form, its cost and the
 * proof.  Returns the form, the operand itself when the round trip is
 * the identity on the domain (*identity), or 0 when a step is not proved.
 */
nid
fold_one(const struct func *f, nid g, const struct model *m, int *identity)
{
	struct sbuf wit = {0};
	struct ffmt F, M;
	uint64_t pts;
	const char *why;
	double lo = 0, hi = 0, cform, ctrip;
	nid x, form;
	int q, known, full, D, how1 = EQ_SAMPLED, how2 = EQ_SAMPLED;
	int ok1, ok2 = 1, elo, ehi, slot = -1;
	long nchk = 0, bad = 0;

	*identity = 0;
	if (!round_trip(g, &x))
		return 0;
	F = fmt_of(N.ty[g]);
	M = fmt_of(N.ty[N.a[g]]);
	D = F.mant - M.mant;
	known = dword_interval(f, x, &lo, &hi);
	q = quantises(f, g, x, &wit, &pts, &why);
	rep_obj(NULL);
	rep_expr("round_trip", g);
	rep_bool("relies_on_rounding", q == Q_DIFFERS);
	if (q == Q_SAME_PROVED || q == Q_SAME_ALL) {
		if (q == Q_SAME_PROVED)
			printf("    nothing is lost: %s; the round trip is the "
			    "identity here and can go\n", why);
		else
			printf("    nothing is lost: on all %llu points of the "
			    "domain it returns its operand; it can go\n",
			    (unsigned long long)pts);
		rep_str("form", "identity");
		rep_end();
		free(wit.p);
		*identity = 1;
		return x;
	}
	if (q == Q_DIFFERS)
		printf("    the program relies on the rounding: at %s\n",
		    sb_str(&wit));
	else
		printf("    no difference on %llu sampled points (not a proof); "
		    "the fold-proof form keeps the rounding anyway\n",
		    (unsigned long long)pts);
	free(wit.p);
	full = !(known && normal_range(&M, lo, hi));
	form = form_of(x, N.ty[g], N.ty[N.a[g]], full);
	cform = cost_under(m, form) - cost_under(m, x);
	{
		struct model plain = *m;
		plain.nfold = 0;
		ctrip = cost_under(&plain, g) - cost_under(&plain, x);
	}
	printf("    fold-proof: %s\n", shared_str(form, 600));
	printf("      cost %g under %s (the round trip as written would be "
	    "%g; folded it is 0 and does not round)\n", cform,
	    sym_str(m->name), ctrip);
	ok1 = prove_rne(TY(K_UINT, F.w), D, &how1);
	printf("    proof:\n");
	printf("      1. between 2^%d and 2^%d the %s values of a binade are "
	    "the %s values whose %d low mantissa\n         bits are 0; the "
	    "bit pattern grows with |x| and a carry out of the mantissa is "
	    "the next\n         binade's first value, so rounding the value "
	    "to %s (nearest, ties to even) is rounding\n         the pattern "
	    "to a multiple of 2^%d (nearest, ties to the even quotient)\n",
	    M.emin, M.emax + 1, ty_str(N.ty[N.a[g]]), ty_str(N.ty[g]), D,
	    ty_str(N.ty[N.a[g]]), D);
	printf("      2. (b + %llu + (b >> %d & 1)) & ~%llu is that rounding "
	    "of b, for every %d-bit b: against\n         q = b >> %d, r = b "
	    "mod 2^%d, (q + (r > 2^%d or r = 2^%d and q odd)) << %d: %s\n",
	    (unsigned long long)(((uint64_t)1 << (D - 1)) - 1), D,
	    (unsigned long long)(((uint64_t)1 << D) - 1), F.w, D, D, D - 1,
	    D - 1, D, how_text(how1));
	if (!full) {
		printf("      3. the range: every operand is 0 or has 2^%d <= "
		    "|x| < %.9g, so no result is subnormal or\n         "
		    "infinite and the carry never reaches the sign bit\n",
		    M.emin, ldexp(2.0 - ldexp(1.0, -(M.mant + 1)), M.emax));
		elo = F.bias + M.emin;
		ehi = F.bias + M.emax;
		if (lo > 0 || hi < 0) {
			int e1, e2;
			frexp(lo > 0 ? lo : -hi, &e1);
			frexp(lo > 0 ? hi : -lo, &e2);
			elo = F.bias + e1 - 1;
			ehi = F.bias + e2 - 1;
		}
	} else {
		ok2 = prove_rneq(TY(K_UINT, F.w), F.mant, D + 1, F.mant + 2, &how2);
		printf("      3. a pattern rounded to 2^%d or more is infinity "
		    "(|x| >= %.9g rounds past the largest %s);\n         a NaN "
		    "stays a NaN; the sign is put back\n", M.emax + 1,
		    ldexp(2.0 - ldexp(1.0, -(M.mant + 1)), M.emax),
		    ty_str(N.ty[N.a[g]]));
		printf("      4. below 2^%d the %s grid is fixed at 2^%d: with "
		    "exponent field e the significand\n         (1.m, %d bits) "
		    "loses k = %d - e bits (at most %d: more leave 0); the "
		    "quotient q is\n         (sig + 2^(k-1) - 1 + (sig >> k & "
		    "1)) >> k, for every sig and k in %d..%d against its\n"
		    "         definition: %s; q <= 2^%d, so %s(q) * 2^%d is "
		    "exact\n", M.emin, ty_str(N.ty[N.a[g]]), M.emin - M.mant,
		    F.mant + 1, F.bias + M.emin + D, F.mant + 2, D + 1,
		    F.mant + 2, how_text(how2), M.mant, ty_str(N.ty[g]),
		    M.emin - M.mant);
		elo = 0;
		ehi = 2 * F.bias + 1;
	}
	/* the check sets x: a parameter */
	if (N.op[x] == O_VAR && N.ki[x] >= 0 && N.ki[x] < f->np)
		slot = (int)N.ki[x];
	if (slot >= 0) {
		nchk = pattern_check(x, g, form, slot, elo, ehi, !full, lo, hi,
		    &bad);
		printf("    check (not the proof): %ld bit patterns, both signs, "
		    "exponent fields %d..%d, ties and their\n      neighbours at "
		    "every mantissa bit: %s\n", nchk, elo, ehi, bad ?
		    "DIFFERENCES FOUND" : "all equal to the evaluator's round "
		    "trip");
		rep_int("check_patterns", nchk);
		rep_int("check_differences", bad);
	}
	rep_expr("form", form);
	rep_bool("full", full);
	rep_num("cost", cform);
	rep_bool("proved", ok1 && ok2 && !bad);
	rep_end();
	if (ok1 && ok2 && !bad)
		return form;
	printf("    not applied: a step above is not proved\n");
	return 0;
}

static void
cmd_foldproof(struct func *f, const struct model *m)
{
	uint32_t lim = N.n, g;
	uint8_t *fo = xcalloc(lim, 1), *in = xcalloc(lim, 1);
	int nfold, ntrip = 0;
	const char *fn = f->anon ? shared_str(f->body, 100) : sym_str(f->name);
	nid body;

	reach_set(f->body, in, lim);
	for (g = 1; g < lim; g++)
		if (in[g] && round_trip(g, NULL))
			ntrip++;
	nfold = fold_marks(m, f->body, fo, lim);
	rep_begin("foldproof");
	rep_str("model", sym_str(m->name));
	rep_int("round_trips", ntrip);
	rep_int("folded", nfold);
	if (nfold == 0) {
		if (ntrip == 0)
			printf("foldproof %s: no float conversion round trip (T "
			    "-> a narrower M -> T) in it\n", fn);
		else
			printf("foldproof %s under %s: %d round trip%s, none "
			    "folded: the model declares no fold for %s, so the "
			    "conversions run as written and nothing changes\n", fn,
			    sym_str(m->name), ntrip, ntrip == 1 ? "" : "s",
			    ntrip == 1 ? "it" : "them");
		rep_bool("rewritten", 0);
		free(fo);
		free(in);
		return;
	}
	printf("foldproof %s under %s: %d conversion round trip%s the driver "
	    "folds\n", fn, sym_str(m->name), nfold, nfold == 1 ? "" : "s");
	nrep = 0;
	rep_list("round_trip_forms");
	for (g = 1; g < lim && nrep < 16; g++) {
		double lo, hi;
		nid x, form;
		int ident;

		if (!fo[g] || !round_trip(g, &x) || !model_folds(m, N.ty[g],
		    N.ty[N.a[g]]))
			continue;
		printf("  %s", shared_str(g, 100));
		if (dword_interval(f, x, &lo, &hi))
			printf(", its operand in [%.9g, %.9g]", lo, hi);
		printf(": %s folds %s, so the card computes the operand itself\n",
		    sym_str(m->name), trip_name(g));
		form = fold_one(f, g, m, &ident);
		if (form != 0) {
			rfrom[nrep] = g;
			rto[nrep++] = form;
		}
	}
	rep_end();
	if (nrep) {
		struct func copy = *f, *h;
		double cfolded = cost_folded(m, f->body);
		char nm[96];

		rmemon = lim;
		rmemo = xcalloc(lim, sizeof *rmemo);
		body = replace(f->body);
		free(rmemo);
		rmemo = NULL;
		snprintf(nm, sizeof nm, "%s_fp", f->anon ? "" : sym_str(f->name));
		copy.name = sym_cstr(nm);
		copy.anon = 0;
		copy.body = body;
		h = func_new(copy.name);
		*h = copy;
		printf("  defined %s = %s\n", nm, shared_str(body, 400));
		printf("  cost under %s: %g (as written, folded: %g)\n",
		    sym_str(m->name), cost_under(m, body), cfolded);
		rep_bool("rewritten", 1);
		rep_str("defined", nm);
		rep_expr("body", body);
	} else
		rep_bool("rewritten", 0);
	free(fo);
	free(in);
}

int
fold_command(struct P *p)
{
	static const char *const stop[] = {"model", NULL};
	struct func *f;
	struct model *m = model_active();
	char w[48];
	uint32_t at = cmdp_pos(p);

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (strcmp(w, "foldproof") != 0 || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	f = cmdp_func(p, stop);
	while (!cmdp_eos(p)) {
		if (cmdp_kw(p, "model")) {
			if (!cmdp_word(p, w, sizeof w) ||
			    (m = model_find(sym_cstr(w))) == NULL)
				fail("foldproof: unknown model (see 'models')");
		} else {
			cmdp_fail_here(p);
			fail("foldproof F [model M]");
		}
	}
	notes_flush();
	cmd_foldproof(f, m);
	return 1;
}
