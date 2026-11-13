/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * stoch: stochastic search engines and probabilistic answers.
 *
 * stoke.  Markov-chain Monte Carlo over straight-line programs, after
 * STOKE (Schkufza, Sharma, Aiken, "Stochastic superoptimization", 2013):
 * a program is up to `len` instructions over the integer operations the
 * model prices (add sub mul and or xor shl shr not neg, and what the
 * function has: compares, a select, an arithmetic shift, zext and sext
 * between its widths, loads from its tables, the dot forms where a value
 * is 32 bits wide) on inputs, earlier results and constants, each
 * instruction of one width; a proposal changes an opcode, a width, an
 * operand, a constant, a whole instruction, or swaps two, and is accepted
 * by Metropolis-Hastings on cost = Hamming distance to the target on a
 * set of test inputs + the program's price under the model, as the
 * report prices it: the instructions the result reads, covered by the
 * model's fusions (a multiply-add, a shift-add).  Chains
 * start from the function itself when it is straight-line (the
 * optimisation phase) and from random programs (synthesis).  A program
 * at distance 0 is only a candidate: it is verified on the whole domain
 * (enumerated, by SAT, or for polynomials by identity testing, labelled
 * as probabilistic), and a counterexample joins the tests.  Seeded:
 * the same seed gives the same run.
 *
 * pit.  Schwartz-Zippel polynomial identity testing: two polynomials of
 * total degree at most d that differ agree at a point drawn uniformly
 * from S^n with probability at most d/|S|.  S is the field of 2^61 - 1
 * elements.  Equality over the integers implies equality in a wrapping
 * type (reduction mod 2^w is a ring homomorphism); the converse fails,
 * so a difference over the integers is not a verdict on machine
 * functions.  equiv uses it when enumeration and SAT cannot decide.
 *
 * Expected divergence (diverge_stats, for cost): how often a warp splits
 * on a condition, from the probability that it holds over the domain
 * (independent lanes), or counted over the warps of consecutive values
 * of the one varying index.
 *
 * phash.  The keys of a switch hashed injectively into a table: masks
 * and shifts, (x K) >> s, ((x xor x >> a) K) >> s with K drawn at
 * random, and a two-level hash with a displacement table; injectivity
 * is checked on the key set exactly, a key check (or the parameter's
 * range) handles everything else, the lookup is proved equal to the
 * switch, and it replaces it when the model prices it lower.
 */
#include "stoch.h"
#include "nodesat.h"
#include "fuse.h"
#include "texenc.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* ---- random numbers: xorshift64*, seeded ---- */

static uint64_t rng_s = 1;

static void
rseed(uint64_t s)
{
	rng_s = s * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
	if (rng_s == 0)
		rng_s = 1;
}

static uint64_t
rnd(void)
{
	rng_s ^= rng_s >> 12;
	rng_s ^= rng_s << 25;
	rng_s ^= rng_s >> 27;
	return rng_s * 0x2545F4914F6CDD1DULL;
}

/*
 * x^n and exp(-d) from + * / alone, so that a run is the same on every
 * libm: an acceptance test must not turn on the last bit of exp().
 */
static double
ipow(double x, int n)
{
	double r = 1;

	while (n-- > 0)
		r *= x;
	return r;
}

static double
exp_neg(double d)
{
	double x, t = 1, s = 1;
	int k;

	if (d <= 0)
		return 1;
	if (d > 60)
		return 0;
	x = -d / 1024;
	for (k = 1; k <= 10; k++) {
		t *= x / k;
		s += t;
	}
	for (k = 0; k < 10; k++)
		s *= s;
	return s;
}

static double
runif(void)
{
	return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0);
}

uint64_t
stoch_seed(void)
{
	const char *e = getenv("UCAS_SEED");

	if (e != NULL && *e)
		return strtoull(e, NULL, 10);
	return 1;
}

/* the price under the model, by its cover with its fusions */
static double
dag_cost(const struct model *m, nid root)
{
	return (double)fuse_expr_cost(m, root);
}

/* the operations under root, each shared one once */
static int
dag_ops(nid root)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc((size_t)N.n * sizeof *st), n;
	uint32_t sp = 0;
	int cnt = 0;

	st[sp++] = root;
	mark[root] = 1;
	while (sp) {
		n = st[--sp];
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF
		    || N.op[n] == O_TABLE)
			continue;
		cnt += N.op[n] != O_CAST;
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
	}
	free(mark);
	free(st);
	return cnt;
}

static const char *
fnm(const struct func *f)
{
	return f->anon ? shared_str(f->body, 120) : sym_str(f->name);
}

/* g's body over f's parameters, by position */
static nid
rebind(const struct func *f, const struct func *g)
{
	nid map[MAXP];
	int j;

	if (g->np > f->np)
		fail("%s has %d parameters, %s has %d", fnm(g), g->np, fnm(f),
		    f->np);
	for (j = 0; j < g->np; j++)
		map[j] = mk_cast(g->p[j].ty, mk_var((uint32_t)j, f->p[j].name,
		    f->p[j].ty));
	return subst(g->body, map, g->np);
}

/* ---- Schwartz-Zippel ---- */

#define PIT_P 2305843009213693951ULL	/* 2^61 - 1 */

static uint64_t
mulp(uint64_t a, uint64_t b)
{
	unsigned __int128 t = (unsigned __int128)a * b;
	uint64_t s = ((uint64_t)t & PIT_P) + (uint64_t)(t >> 61);

	s = (s & PIT_P) + (s >> 61);
	return s >= PIT_P ? s - PIT_P : s;
}

static uint64_t
redp(int64_t v)
{
	if (v >= 0)
		return (uint64_t)v % PIT_P;
	return (PIT_P - ((uint64_t)(-(v + 1)) + 1) % PIT_P) % PIT_P;
}

/* e is a polynomial over the integers; its total degree bound */
static int
poly_deg(nid e, int *deg, int depth)
{
	int a, b;
	ty_t t = N.ty[e];

	if (depth > 200)
		return 0;
	switch (N.op[e]) {
	case O_CONST:
		if (TY_FP(t) && N.kr[e] != floor(N.kr[e]))
			return 0;
		if (TY_FP(t) && fabs(N.kr[e]) > 9e15)
			return 0;
		*deg = 0;
		return 1;
	case O_VAR:
		if (!TY_ISINT(t) || t == TY_BOOL)
			return 0;
		*deg = 1;
		return 1;
	case O_NEG:
		return TY_ISINT(t) && poly_deg(N.a[e], deg, depth + 1);
	case O_ADD: case O_SUB:
		if (!TY_ISINT(t) || !poly_deg(N.a[e], &a, depth + 1) ||
		    !poly_deg(N.b[e], &b, depth + 1))
			return 0;
		*deg = a > b ? a : b;
		return 1;
	case O_MUL:
		if (!TY_ISINT(t) || !poly_deg(N.a[e], &a, depth + 1) ||
		    !poly_deg(N.b[e], &b, depth + 1))
			return 0;
		*deg = a + b;
		return *deg <= 4096;
	case O_POW:
		if (!TY_ISINT(t) || N.op[N.b[e]] != O_CONST || N.ki[N.b[e]] < 0 ||
		    N.ki[N.b[e]] > 64 || !poly_deg(N.a[e], &a, depth + 1))
			return 0;
		*deg = a * (int)N.ki[N.b[e]];
		return *deg <= 4096;
	case O_SHL:
		if (!TY_ISINT(t) || N.op[N.b[e]] != O_CONST || N.ki[N.b[e]] < 0 ||
		    N.ki[N.b[e]] > 60)
			return 0;
		return poly_deg(N.a[e], deg, depth + 1);
	case O_CAST: {
		ty_t ta = N.ty[N.a[e]];
		/* a narrowing (or the same width) reduces mod 2^w: a ring
		 * homomorphism; a widening is not one */
		if (!TY_ISINT(t) || !TY_ISINT(ta) || t == TY_BOOL || ta == TY_BOOL)
			return 0;
		if (TY_SIZED(ta) && (!TY_SIZED(t) || TYW(t) > TYW(ta)))
			return 0;
		return poly_deg(N.a[e], deg, depth + 1);
	}
	}
	return 0;
}

static uint64_t *pv;
static uint32_t *pstamp, pcur, plim;

static uint64_t
poly_eval(nid e, const uint64_t *x)
{
	uint64_t r, a;
	int k;

	if (e < plim && pstamp[e] == pcur)
		return pv[e];
	switch (N.op[e]) {
	case O_CONST:
		r = TY_FP(N.ty[e]) ? redp((int64_t)N.kr[e]) : redp(N.ki[e]);
		break;
	case O_VAR:
		r = x[N.ki[e]];
		break;
	case O_NEG:
		r = (PIT_P - poly_eval(N.a[e], x)) % PIT_P;
		break;
	case O_ADD:
		r = (poly_eval(N.a[e], x) + poly_eval(N.b[e], x)) % PIT_P;
		break;
	case O_SUB:
		r = (poly_eval(N.a[e], x) + PIT_P - poly_eval(N.b[e], x)) % PIT_P;
		break;
	case O_MUL:
		r = mulp(poly_eval(N.a[e], x), poly_eval(N.b[e], x));
		break;
	case O_POW:
		a = poly_eval(N.a[e], x);
		r = 1;
		for (k = 0; k < (int)N.ki[N.b[e]]; k++)
			r = mulp(r, a);
		break;
	case O_SHL:
		r = mulp(poly_eval(N.a[e], x), redp((int64_t)1 << N.ki[N.b[e]]));
		break;
	default:	/* a cast */
		r = poly_eval(N.a[e], x);
	}
	if (e < plim) {
		pv[e] = r;
		pstamp[e] = pcur;
	}
	return r;
}

int
pit_check(const struct func *f, nid a, nid b, int trials, uint64_t seed,
    double *bound, struct sbuf *text)
{
	uint64_t x[MAXP], va = 0, vb = 0;
	int da, db, d, t, j, differ = 0;

	if (!poly_deg(a, &da, 0) || !poly_deg(b, &db, 0))
		return -1;
	for (j = 0; j < f->np; j++)
		if (!TY_ISINT(f->p[j].ty) || f->p[j].ty == TY_BOOL)
			return -1;
	d = da > db ? da : db;
	plim = N.n;
	pv = xcalloc(plim, sizeof *pv);
	pstamp = xcalloc(plim, sizeof *pstamp);
	pcur = 0;
	rseed(seed);
	for (t = 0; t < trials && !differ; t++) {
		for (j = 0; j < f->np; j++)
			x[j] = rnd() % PIT_P;
		pcur++;
		va = poly_eval(a, x);
		vb = poly_eval(b, x);
		if (va != vb)
			differ = 1;
	}
	free(pv);
	free(pstamp);
	pv = NULL;
	pstamp = NULL;
	if (differ) {
		sb_printf(text, "they differ as polynomials over the integers: "
		    "at ");
		for (j = 0; j < f->np; j++)
			sb_printf(text, "%s%s = %llu", j ? ", " : "",
			    sym_str(f->p[j].name), (unsigned long long)x[j]);
		sb_printf(text, " (mod 2^61 - 1) the values are %llu and %llu",
		    (unsigned long long)va, (unsigned long long)vb);
		*bound = 0;
		return 0;
	}
	*bound = d == 0 ? 0 : ipow((double)d / (double)PIT_P, trials);
	sb_printf(text, "polynomials of total degree at most %d, equal at %d "
	    "random point%s mod 2^61 - 1: two that differ agree at one with "
	    "probability at most %d/(2^61 - 1) = %.3g (Schwartz-Zippel), at "
	    "all %d with at most %.3g", d, trials, trials == 1 ? "" : "s", d,
	    (double)d / (double)PIT_P, trials, *bound);
	return 1;
}

/* ---- polynomial identities mod 2^w, by expansion ---- */

/*
 * Two expressions of a sized integer type of w bits are expanded as
 * polynomials mod 2^w in the parameters and in what is not a ring
 * operation of a type at least w bits wide (y / 8, x & 255, a narrower
 * value widened), each such subterm taken whole as one more unknown, so
 * that what holds for every value of the unknowns holds for the values
 * they take.  The difference is read in falling factorials, x^(j) =
 * x (x - 1) ... (x - j + 1) = j! C(x, j): a term c x1^(j1) ... xn^(jn)
 * is c j1! ... jn! times integers, so it is 0 at every point when 2^w
 * divides c j1! ... jn! (x^2 + x is even: 2^1 divides 2!).  A
 * difference that is 0 as a polynomial is the case where every c is.
 * The unknowns must be defined at every input (a division by a nonzero
 * constant, a shift by a constant below the width), so that the second
 * is defined wherever the first is.
 */
#define PI_MAXV 16		/* unknowns */
#define PI_MAXT 4096		/* terms */
#define PI_MAXE 63		/* one unknown's degree */

struct pmono {
	uint64_t c;
	uint8_t e[PI_MAXV];
};

struct ppoly {
	struct pmono *m;
	int n;
};

static struct {
	nid atom[PI_MAXV];
	int na, w, fail;
	uint64_t mask;
} PI;

static int
pmono_cmp(const void *a, const void *b)
{
	return memcmp(((const struct pmono *)a)->e, ((const struct pmono *)b)->e,
	    PI_MAXV);
}

/* sorted, equal monomials merged, zero coefficients dropped */
static void
ppoly_norm(struct ppoly *p)
{
	int i, n = 0;

	if (p->n > 1)
		qsort(p->m, (size_t)p->n, sizeof *p->m, pmono_cmp);
	for (i = 0; i < p->n; i++)
		if (n && !memcmp(p->m[n - 1].e, p->m[i].e, PI_MAXV))
			p->m[n - 1].c = (p->m[n - 1].c + p->m[i].c) & PI.mask;
		else
			p->m[n++] = p->m[i];
	p->n = 0;
	for (i = 0; i < n; i++)
		if (p->m[i].c & PI.mask)
			p->m[p->n++] = p->m[i];
	if (p->n > PI_MAXT)
		PI.fail = 1;
}

static void
ppoly_const(struct ppoly *p, uint64_t c)
{
	p->m = xcalloc(1, sizeof *p->m);
	p->n = (c & PI.mask) != 0;
	p->m[0].c = c & PI.mask;
}

/* p += s q */
static void
ppoly_addto(struct ppoly *p, const struct ppoly *q, uint64_t s)
{
	int i;

	p->m = xrealloc(p->m, (size_t)(p->n + q->n + 1) * sizeof *p->m);
	for (i = 0; i < q->n; i++) {
		p->m[p->n] = q->m[i];
		p->m[p->n++].c = q->m[i].c * s & PI.mask;
	}
	ppoly_norm(p);
}

static void
ppoly_mul(struct ppoly *r, const struct ppoly *a, const struct ppoly *b)
{
	int i, j, k;

	r->n = 0;
	if ((int64_t)a->n * b->n > (1 << 20)) {
		PI.fail = 1;
		ppoly_const(r, 0);
		return;
	}
	r->m = xcalloc((size_t)a->n * (size_t)b->n + 1, sizeof *r->m);
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			struct pmono *t = &r->m[r->n++];
			t->c = a->m[i].c * b->m[j].c & PI.mask;
			for (k = 0; k < PI_MAXV; k++) {
				int e = a->m[i].e[k] + b->m[j].e[k];
				if (e > PI_MAXE)
					PI.fail = 1;
				t->e[k] = (uint8_t)(e > PI_MAXE ? PI_MAXE : e);
			}
		}
	ppoly_norm(r);
}

/* e is defined at every input of a sized integer type */
static int
pi_total(nid e, int depth)
{
	ty_t t = N.ty[e];
	int64_t k;

	if (depth > 200 || !TY_SIZED(t))
		return 0;
	switch (N.op[e]) {
	case O_CONST: case O_VAR:
		return 1;
	case O_NEG: case O_NOT: case O_LNOT: case O_CAST:
		return pi_total(N.a[e], depth + 1);
	case O_ADD: case O_SUB: case O_MUL: case O_AND: case O_OR: case O_XOR:
	case O_MIN: case O_MAX: case O_EQ: case O_NE: case O_LT: case O_LE:
	case O_GT: case O_GE: case O_LAND: case O_LOR:
		return pi_total(N.a[e], depth + 1) && pi_total(N.b[e], depth + 1);
	case O_DIV: case O_MOD:
		if (!is_const(N.b[e]))
			return 0;
		k = N.ki[N.b[e]];
		return (TYW(t) >= 64 ? k : k & (int64_t)((1ULL << TYW(t)) - 1)) !=
		    0 && pi_total(N.a[e], depth + 1);
	case O_SHL: case O_SHR:
		return is_const(N.b[e]) && N.ki[N.b[e]] >= 0 &&
		    N.ki[N.b[e]] < TYW(t) && pi_total(N.a[e], depth + 1);
	case O_SEL:
		return pi_total(N.a[e], depth + 1) &&
		    pi_total(N.b[e], depth + 1) && pi_total(N.c[e], depth + 1);
	}
	return 0;
}

static void
pi_expand(nid e, struct ppoly *p, int depth)
{
	ty_t t = N.ty[e], ta;
	struct ppoly a, b;
	int i;

	if (PI.fail || depth > 400) {
		PI.fail = 1;
		ppoly_const(p, 0);
		return;
	}
	if (TY_SIZED(t) && t != TY_BOOL && TYW(t) >= PI.w)
		switch (N.op[e]) {
		case O_CONST:
			ppoly_const(p, (uint64_t)N.ki[e]);
			return;
		case O_NEG:
			pi_expand(N.a[e], &a, depth + 1);
			ppoly_const(p, 0);
			ppoly_addto(p, &a, PI.mask);
			free(a.m);
			return;
		case O_ADD: case O_SUB:
			pi_expand(N.a[e], p, depth + 1);
			pi_expand(N.b[e], &b, depth + 1);
			ppoly_addto(p, &b, N.op[e] == O_ADD ? 1 : PI.mask);
			free(b.m);
			return;
		case O_MUL:
			pi_expand(N.a[e], &a, depth + 1);
			pi_expand(N.b[e], &b, depth + 1);
			ppoly_mul(p, &a, &b);
			free(a.m);
			free(b.m);
			return;
		case O_SHL:
			if (!is_const(N.b[e]) || N.ki[N.b[e]] < 0 ||
			    N.ki[N.b[e]] >= TYW(t))
				break;
			pi_expand(N.a[e], &a, depth + 1);
			ppoly_const(&b, N.ki[N.b[e]] >= 64 ? 0 :
			    (uint64_t)1 << N.ki[N.b[e]]);
			ppoly_mul(p, &a, &b);
			free(a.m);
			free(b.m);
			return;
		case O_CAST:
			ta = N.ty[N.a[e]];
			if (!TY_SIZED(ta) || ta == TY_BOOL || TYW(ta) < PI.w)
				break;
			pi_expand(N.a[e], p, depth + 1);
			return;
		}
	/* an unknown */
	if (!pi_total(e, 0)) {
		PI.fail = 1;
		ppoly_const(p, 0);
		return;
	}
	for (i = 0; i < PI.na && PI.atom[i] != e; i++)
		;
	if (i == PI.na) {
		if (PI.na == PI_MAXV) {
			PI.fail = 1;
			ppoly_const(p, 0);
			return;
		}
		PI.atom[PI.na++] = e;
	}
	ppoly_const(p, 1);
	p->m[0].e[i] = 1;
}

static int
ppoly_deg(const struct ppoly *p)
{
	int i, k, d = 0;

	for (i = 0; i < p->n; i++) {
		int s = 0;
		for (k = 0; k < PI_MAXV; k++)
			s += p->m[i].e[k];
		if (s > d)
			d = s;
	}
	return d;
}

/* the difference d, in falling factorials, is 0 at every point mod 2^w */
static int
ff_zero(const struct ppoly *d)
{
	static uint64_t S[PI_MAXE + 1][PI_MAXE + 1], fact[PI_MAXE + 1];
	struct ppoly ff = {0};
	int i, k, j, ok = 1;

	if (!fact[0]) {
		/* Stirling numbers of the second kind, x^k = sum S(k, j)
		 * x^(j); everything mod 2^64, which 2^w divides */
		S[0][0] = 1;
		for (k = 1; k <= PI_MAXE; k++)
			for (j = 1; j <= k; j++)
				S[k][j] = (uint64_t)j * S[k - 1][j] + S[k - 1][j - 1];
		fact[0] = 1;
		for (j = 1; j <= PI_MAXE; j++)
			fact[j] = fact[j - 1] * (uint64_t)j;
	}
	for (i = 0; i < d->n && !PI.fail; i++) {
		const struct pmono *m = &d->m[i];
		int jv[PI_MAXV];
		for (k = 0; k < PI_MAXV; k++)
			jv[k] = m->e[k] ? 1 : 0;
		for (;;) {
			struct pmono t;
			t.c = m->c;
			for (k = 0; k < PI_MAXV; k++) {
				t.e[k] = (uint8_t)jv[k];
				t.c *= S[m->e[k]][jv[k]];
			}
			t.c &= PI.mask;
			if (t.c) {
				if (ff.n % 256 == 0)
					ff.m = xrealloc(ff.m, (size_t)(ff.n + 256) *
					    sizeof *ff.m);
				ff.m[ff.n++] = t;
				if (ff.n > 64 * PI_MAXT)
					PI.fail = 1;
			}
			/* the next j vector, 1 <= j_k <= e_k */
			for (k = 0; k < PI_MAXV; k++) {
				if (!m->e[k])
					continue;
				if (jv[k] < m->e[k]) {
					jv[k]++;
					break;
				}
				jv[k] = 1;
			}
			if (k == PI_MAXV || PI.fail)
				break;
		}
	}
	if (!PI.fail)
		ppoly_norm(&ff);
	for (i = 0; i < ff.n && ok && !PI.fail; i++) {
		uint64_t c = ff.m[i].c;
		for (k = 0; k < PI_MAXV; k++)
			c *= fact[ff.m[i].e[k]];
		ok = (c & PI.mask) == 0;
	}
	free(ff.m);
	return ok && !PI.fail;
}

int
poly_identity(nid a, nid b, int *deg, struct sbuf *how)
{
	ty_t t = N.ty[a];
	struct ppoly pa, pb;
	int i, k, d, r = 0, nat = 0;

	*deg = 0;
	if (!TY_SIZED(t) || t == TY_BOOL)
		return 0;
	memset(&PI, 0, sizeof PI);
	PI.w = TYW(t);
	PI.mask = PI.w >= 64 ? ~0ULL : (1ULL << PI.w) - 1;
	pi_expand(a, &pa, 0);
	pi_expand(N.ty[b] == t ? b : mk_cast(t, b), &pb, 0);
	if (PI.fail) {
		free(pa.m);
		free(pb.m);
		return 0;
	}
	*deg = ppoly_deg(&pa);
	if ((d = ppoly_deg(&pb)) > *deg)
		*deg = d;
	ppoly_addto(&pa, &pb, PI.mask);
	if (!PI.fail && (pa.n == 0 || ff_zero(&pa))) {
		struct sbuf v = {0};
		for (i = 0; i < PI.na; i++)
			if (N.op[PI.atom[i]] != O_VAR)
				nat++;
		for (i = k = 0; i < PI.na; i++)
			if (N.op[PI.atom[i]] == O_VAR)
				sb_printf(&v, "%s%s", k++ ? ", " : "",
				    node_str(PI.atom[i]));
		for (i = 0; i < PI.na; i++)
			if (N.op[PI.atom[i]] != O_VAR)
				sb_printf(&v, "%s%s", k++ ? ", " : "",
				    shared_str(PI.atom[i], 60));
		if (pa.n == 0)
			sb_printf(how, "proved by algebra: both sides expand to one "
			    "polynomial in %s mod 2^%d", k ? sb_str(&v) : "no "
			    "unknown", PI.w);
		else
			sb_printf(how, "proved by algebra: the two sides differ by a "
			    "polynomial in %s that is 0 at every point mod 2^%d, "
			    "each coefficient in falling factorials times its "
			    "factorials being a multiple of 2^%d", sb_str(&v), PI.w,
			    PI.w);
		if (nat)
			sb_printf(how, "; %s taken whole as unknown%s", nat == 1 ?
			    "one subterm" : "the subterms", nat == 1 ? "" : "s");
		free(v.p);
		r = 1;
	}
	free(pa.m);
	free(pb.m);
	return r;
}

static void
cmd_pit(struct func *f, struct func *g, int trials, uint64_t seed)
{
	struct sbuf t = {0}, w = {0};
	nid gb = rebind(f, g);
	double bound = 0;
	int r, da = 0, db = 0, sampled = 0, sized = TY_SIZED(N.ty[f->body]);

	if (g->np != f->np)
		fail("%s and %s take different numbers of parameters", fnm(f),
		    fnm(g));
	printf("pit %s %s: polynomial identity testing (Schwartz-Zippel), seed "
	    "%llu\n", fnm(f), fnm(g), (unsigned long long)seed);
	rep_begin("pit");
	rep_int("seed", (long long)seed);
	rep_int("trials", trials);
	r = pit_check(f, f->body, gb, trials, seed, &bound, &t);
	if (r < 0) {
		printf("  not polynomials over the integers in integer "
		    "parameters (only + - *, constant powers and shifts, integer "
		    "constants, conversions that do not widen): nothing to test; "
		    "`equiv` compares them as functions\n");
		rep_str("verdict", "not_polynomials");
		free(t.p);
		return;
	}
	poly_deg(f->body, &da, 0);
	poly_deg(gb, &db, 0);
	printf("  %s has total degree at most %d, %s at most %d\n", fnm(f), da,
	    fnm(g), db);
	rep_int("degree", da > db ? da : db);
	if (r == 1) {
		printf("  %s\n", sb_str(&t));
		printf("  -> equal as polynomials over the integers, except with "
		    "probability at most %.3g (not a proof: a probabilistic "
		    "answer)\n", bound);
		if (sized)
			printf("     in %s they are then equal for every input: "
			    "reduction mod 2^%d is a ring homomorphism\n",
			    ty_str(N.ty[f->body]), TYW(N.ty[f->body]));
		rep_str("verdict", "equal_probably");
		rep_num("error_probability", bound);
		rep_proof("probabilistic", sb_str(&t));
	} else {
		int m;
		printf("  %s (certain)\n", sb_str(&t));
		rep_str("verdict", "differ_over_integers");
		if (sized) {
			/* the machine functions are another question */
			m = equiv_check(f, f->body, gb, &w, &sampled);
			if (m)
				printf("  -> and as %s functions they differ too, e.g. "
				    "%s\n", ty_str(N.ty[f->body]), sb_str(&w));
			else
				printf("  -> as %s functions they agree on %s: "
				    "polynomials that differ over the integers can agree "
				    "mod 2^%d\n     for every input (2^%d x (x + 1) is 0), "
				    "so this test cannot tell; `equiv` decides the "
				    "machine functions\n", ty_str(N.ty[f->body]),
				    sampled ? "a sample of the domain" : "the whole "
				    "domain", TYW(N.ty[f->body]), TYW(N.ty[f->body]) - 1);
			rep_bool("machine_functions_differ", m != 0);
		} else
			printf("  -> different functions\n");
	}
	free(t.p);
	free(w.p);
}

/* ---- expected divergence ---- */

#define DV_MAXCASES (1u << 16)

static int dv_cons = -1;

void
diverge_consecutive(int slot)
{
	dv_cons = slot;
}

/* the parameters cond reads */
static void
used_params(nid e, uint8_t *used, uint8_t *mark)
{
	uint32_t j;

	if (e == 0 || mark[e])
		return;
	mark[e] = 1;
	if (N.op[e] == O_VAR) {
		if (N.ki[e] >= 0 && N.ki[e] < MAXP)
			used[N.ki[e]] = 1;
		return;
	}
	if (N.op[e] == O_CONST || N.op[e] == O_UNDEF)
		return;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			used_params(N.lst[N.a[e] + j], used, mark);
		return;
	}
	used_params(N.a[e], used, mark);
	used_params(N.b[e], used, mark);
	used_params(N.c[e], used, mark);
}

/* what f reads besides its lane index is held at one value; 0 when a float
 * is read beside the index (the lanes are then independent) */
static int
hold_unused(struct func *g, const uint8_t *used, int lane)
{
	int j;

	for (j = 0; j < g->np; j++) {
		if (used[j] || j == lane)
			continue;
		if (TY_ISINT(g->p[j].ty)) {
			int64_t lo, hi;
			param_bounds(&g->p[j], &lo, &hi);
			g->p[j].ranged = 1;
			g->p[j].lo = g->p[j].hi = lo;
			g->p[j].rlo = g->p[j].rhi = (double)lo;
		} else {
			double v = g->p[j].ranged ? g->p[j].rlo : 0;
			g->p[j].ranged = 1;
			g->p[j].rlo = g->p[j].rhi = v;
		}
	}
	if (lane >= 0)
		for (j = 0; j < g->np; j++)
			if (j != lane && used[j] && !TY_ISINT(g->p[j].ty))
				return 0;
	return 1;
}

/*
 * The warps: the aligned runs of `warp` values of the parameter in slot
 * lane, the other parameters at each of their values (every case when
 * there are at most DV_MAXCASES, *exact; 4096 drawn otherwise).  fn gets
 * each warp's lanes after pl ran on them, ok[k] when lane k meets the
 * where clause (at wloc).
 */
static void
warp_walk(const struct func *f, struct plan *pl, uint32_t wloc, int lane,
    int warp, void (*fn)(void *, const struct plan *, const uint8_t *,
    uint32_t), void *ctx, int *exact)
{
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int64_t lo[MAXP], hi[MAXP], cur[MAXP];
	uint64_t total = 1, wins, c;
	uint64_t rs = 0x9e3779b97f4a7c15ULL;
	uint8_t ok[CH];
	uint32_t k, np;
	int j;

	for (j = 0; j < f->np; j++) {
		lo[j] = hi[j] = 0;
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &lo[j], &hi[j]);
		else
			in_r[(size_t)j * CH] = f->p[j].rlo;
	}
	for (j = 0; j < f->np; j++)
		if (!TY_ISINT(f->p[j].ty))
			for (k = 1; k < CH; k++)
				in_r[(size_t)j * CH + k] = in_r[(size_t)j * CH];
	wins = (uint64_t)(hi[lane] / warp - lo[lane] / warp) + 1;
	for (j = 0; j < f->np; j++)
		if (j != lane) {
			uint64_t sz = (uint64_t)(hi[j] - lo[j]) + 1;
			total = sz == 0 || sz > DV_MAXCASES || total * sz >
			    DV_MAXCASES ? DV_MAXCASES + 1 : total * sz;
		}
	total = wins == 0 || wins > DV_MAXCASES || total * wins >
	    DV_MAXCASES ? DV_MAXCASES + 1 : total * wins;
	*exact = total <= DV_MAXCASES;
	for (j = 0; j < f->np; j++)
		cur[j] = j == lane ? lo[j] / warp : lo[j];
	for (c = 0; c < (*exact ? total : 4096); c++) {
		int64_t base;
		uint32_t w;
		if (!*exact)
			for (j = 0; j < f->np; j++) {
				uint64_t span = j == lane ? wins :
				    (uint64_t)(hi[j] - lo[j]) + 1;
				rs ^= rs << 13;
				rs ^= rs >> 7;
				rs ^= rs << 17;
				cur[j] = (j == lane ? lo[j] / warp : lo[j]) +
				    (int64_t)(span ? rs % span : rs >> 1);
			}
		base = cur[lane] * warp;
		np = 0;
		for (w = 0; w < (uint32_t)warp; w++) {
			int64_t l = base + w;
			if (l < lo[lane] || l > hi[lane])
				continue;
			for (j = 0; j < f->np; j++)
				in_i[(size_t)j * CH + np] = j == lane ? l :
				    cur[j];
			np++;
		}
		if (np) {
			plan_run(pl, np, in_i, in_r);
			valid_mask(pl, wloc, np, ok);
			fn(ctx, pl, ok, np);
		}
		if (*exact)
			for (j = 0; j < f->np; j++) {
				int64_t top = j == lane ? hi[j] / warp : hi[j];
				if (cur[j] < top) {
					cur[j]++;
					break;
				}
				cur[j] = j == lane ? lo[j] / warp : lo[j];
			}
	}
	free(in_i);
	free(in_r);
}

struct dvw {
	uint32_t loc, rloc;	/* the condition; the lanes that reach it */
	uint64_t nt, n, nat, naf, nw, nwv;
};

/* a warp of a ?:'s condition: how many of the lanes that reach it hold
 * it, all or none; nwv counts the warps, nw those it is reached in */
static void
dv_warp(void *ctx, const struct plan *pl, const uint8_t *ok, uint32_t np)
{
	struct dvw *s = ctx;
	uint32_t k, t = 0, w = 0, a = 0;

	for (k = 0; k < np; k++)
		if (ok[k]) {
			w++;
			if (s->loc == NOTOK || (s->rloc != NOTOK &&
			    !COLI(pl, s->rloc)[k]))
				continue;
			a++;
			t += COLI(pl, s->loc)[k] != 0;
		}
	s->nwv += w > 0;
	if (a) {
		s->nw++;
		s->n += a;
		s->nt += t;
		s->nat += t == a;
		s->naf += t == 0;
	}
}

/*
 * The lanes are independent draws from the domain (1: split = 1 - p^warp
 * - (1 - p)^warp) unless a parameter was named consecutive: the warps are
 * then the aligned runs of `warp` values of it, counted (2).  Only the
 * parameters the condition reads are enumerated; *exact: every case of
 * them was, not a sample.
 *
 * reach (0: every lane): the lanes that get to the ?: at all, those that
 * take its arm of the enclosing branches.  A warp none of whose lanes
 * does skips it; *reached is the fraction of warps that do not, and
 * *all_t, *all_f are fractions of those, over the lanes that reach it
 * (independent lanes, a lane reaching with probability r: a warp with
 * 1 - (1 - r)^warp, all true with (1 - r + r p)^warp - (1 - r)^warp).
 *
 * loop (LOOP_NONE: none): cond is in that loop's body; it is counted at
 * every iteration, over the lanes that run it (loop.c's probe), the
 * whole function evaluated, every parameter drawn.
 */
int
diverge_stats(const struct func *f, nid cond, nid reach, uint32_t loop,
    int warp, double *p, double *all_t, double *all_f, double *reached,
    int *exact, uint64_t *points)
{
	struct func g = *f;
	struct plan pl;
	nid roots[3];
	int64_t *in_i;
	double *in_r;
	uint8_t ok[CH], used[MAXP] = {0}, *mark;
	uint32_t loc, rloc, wloc, np, k;
	int j, lane = dv_cons, nr = 0;
	uint64_t nt = 0, n = 0, nv = 0;

	*exact = 0;
	*points = 0;
	*reached = 1;
	if (warp < 2 || warp > CH)
		return 0;
	mark = xcalloc(N.n, 1);
	used_params(cond, used, mark);
	used_params(reach, used, mark);
	used_params(f->where, used, mark);
	free(mark);
	if (loop != LOOP_NONE)
		for (j = 0; j < g.np; j++)
			used[j] = 1;
	if (!hold_unused(&g, used, lane))
		lane = -1;
	f = &g;
	roots[nr++] = loop != LOOP_NONE ? f->body : cond;
	if (reach)
		roots[nr++] = reach;
	roots[nr++] = f->where;
	plan_build(&pl, roots, (uint32_t)nr, 0);
	loc = loop != LOOP_NONE ? NOTOK : plan_loc(&pl, cond);
	rloc = reach ? plan_loc(&pl, reach) : NOTOK;
	wloc = f->where ? plan_loc(&pl, f->where) : NOTOK;
	if (loop != LOOP_NONE)
		loop_probe(loop, cond, warp);
	if (lane >= 0) {
		struct dvw s = {0};
		s.loc = loc;
		s.rloc = rloc;
		warp_walk(f, &pl, wloc, lane, warp, dv_warp, &s, exact);
		plan_free(&pl);
		if (loop != LOOP_NONE)
			loop_probe_counts(&s.nw, &s.nat, &s.naf, &s.n, &s.nt);
		else
			*reached = s.nwv ? (double)s.nw / (double)s.nwv : 0;
		if (s.nw == 0)
			return 0;
		*p = (double)s.nt / (double)s.n;
		*all_t = (double)s.nat / (double)s.nw;
		*all_f = (double)s.naf / (double)s.nw;
		*points = s.nw;
		return 2;
	}
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	{
		/* independent lanes: the condition's frequency on the domain */
		struct diter d;
		diter_init(&d, f, DV_MAXCASES, 257, 4096);
		*exact = d.exhaustive;
		while ((np = diter_next(&d, in_i, in_r)) != 0) {
			plan_run(&pl, np, in_i, in_r);
			valid_mask(&pl, wloc, np, ok);
			for (k = 0; k < np; k++)
				if (ok[k] && loc != NOTOK && !(COLF(&pl, loc)[k] &
				    F_UNDEF)) {
					nv++;
					if (rloc != NOTOK && !COLI(&pl, rloc)[k])
						continue;
					n++;
					nt += COLI(&pl, loc)[k] != 0;
				}
		}
	}
	plan_free(&pl);
	free(in_i);
	free(in_r);
	if (loop != LOOP_NONE) {
		uint64_t nw, nat, naf;
		loop_probe_counts(&nw, &nat, &naf, &n, &nt);
		nv = n;
	}
	if (nv == 0)
		return 0;
	if (n == 0) {
		/* no lane of the sample reaches it */
		*reached = 0;
		*p = *all_t = *all_f = 0;
		*points = nv;
		return 1;
	}
	*p = (double)nt / (double)n;
	/* a sample that never (or always) saw it bounds it: the rule of
	 * three, 3/n at 95% confidence */
	if (!*exact && (nt == 0 || nt == n)) {
		*p = nt == 0 ? 3.0 / (double)n : 1 - 3.0 / (double)n;
		*exact = -1;
	}
	{
		double r = (double)n / (double)nv, none = ipow(1 - r, warp);
		*reached = 1 - none;
		*all_t = (ipow(1 - r + r * *p, warp) - none) / *reached;
		*all_f = (ipow(1 - r * *p, warp) - none) / *reached;
	}
	*points = loop != LOOP_NONE ? n : nv;
	return 1;
}

struct wtrip {
	uint32_t loc[2];
	int counted;            /* loc[0] is the count, else lo and hi */
	double *t;              /* independent lanes: every count measured */
	uint64_t n, cap, nw, nsplit;
	double summax;
};

/* the trip count in lane k; 0 when it is undefined there */
static int
lane_trips(const struct wtrip *s, const struct plan *pl, uint32_t k,
    double *t)
{
	if (s->counted) {
		if (COLF(pl, s->loc[0])[k] & F_UNDEF)
			return 0;
		*t = (double)COLI(pl, s->loc[0])[k];
	} else {
		if ((COLF(pl, s->loc[0])[k] | COLF(pl, s->loc[1])[k]) & F_UNDEF)
			return 0;
		*t = (double)COLI(pl, s->loc[1])[k] -
		    (double)COLI(pl, s->loc[0])[k] + 1;
	}
	if (*t < 0)
		*t = 0;
	return 1;
}

/* a warp of a loop: the most trips of its lanes, and whether they differ */
static void
trip_warp(void *ctx, const struct plan *pl, const uint8_t *ok, uint32_t np)
{
	struct wtrip *s = ctx;
	double t, mx = 0, first = -1;
	uint32_t k;
	int split = 0;

	for (k = 0; k < np; k++)
		if (ok[k] && lane_trips(s, pl, k, &t)) {
			if (first < 0)
				first = t;
			split |= t != first;
			if (t > mx)
				mx = t;
		}
	if (first < 0)
		return;
	s->nw++;
	s->nsplit += split;
	s->summax += mx;
}

static int
cmp_dbl(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;

	return x < y ? -1 : x > y;
}

int
warp_trips(const struct func *f, nid trips, nid lo, nid hi, int warp,
    double *emax, double *split, int *exact, uint64_t *points)
{
	struct func g = *f;
	struct plan pl;
	struct wtrip s = {0};
	nid roots[3];
	uint8_t used[MAXP] = {0}, *mark;
	int nr = 0, lane = dv_cons;
	uint32_t wloc;

	*exact = 0;
	*points = 0;
	if (warp < 2 || warp > CH)
		return 0;
	s.counted = trips != 0;
	if (s.counted)
		roots[nr++] = trips;
	else {
		roots[nr++] = lo;
		roots[nr++] = hi;
	}
	roots[nr++] = f->where;
	plan_build(&pl, roots, nr, 0);
	s.loc[0] = plan_loc(&pl, roots[0]);
	s.loc[1] = s.counted ? s.loc[0] : plan_loc(&pl, roots[1]);
	wloc = f->where ? plan_loc(&pl, f->where) : NOTOK;
	if (lane >= 0) {
		mark = xcalloc(N.n, 1);
		used_params(roots[0], used, mark);
		if (!s.counted)
			used_params(roots[1], used, mark);
		used_params(f->where, used, mark);
		free(mark);
		if (!hold_unused(&g, used, lane))
			lane = -1;
	}
	if (lane >= 0) {
		warp_walk(&g, &pl, wloc, lane, warp, trip_warp, &s, exact);
		plan_free(&pl);
		if (s.nw == 0)
			return 0;
		*emax = s.summax / (double)s.nw;
		*split = (double)s.nsplit / (double)s.nw;
		*points = s.nw;
		return 2;
	}
	{
		/* independent lanes: every count over the domain, sorted; a warp
		 * takes the most of `warp` draws, so its count is at most t with
		 * probability F(t)^warp, and its lanes agree with probability
		 * sum_t P(t)^warp */
		struct diter d;
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		uint8_t ok[CH];
		uint32_t np, k;
		uint64_t i, j;
		double t, fb = 0, agree = 0;

		diter_init(&d, f, 1u << 16, 2, 4096);
		*exact = d.exhaustive;
		while ((np = diter_next(&d, in_i, in_r)) != 0) {
			plan_run(&pl, np, in_i, in_r);
			valid_mask(&pl, wloc, np, ok);
			for (k = 0; k < np; k++)
				if (ok[k] && lane_trips(&s, &pl, k, &t)) {
					if (s.n == s.cap) {
						s.cap = s.cap ? 2 * s.cap : 1024;
						s.t = xrealloc(s.t, s.cap *
						    sizeof *s.t);
					}
					s.t[s.n++] = t;
				}
		}
		plan_free(&pl);
		free(in_i);
		free(in_r);
		if (s.n == 0) {
			free(s.t);
			return 0;
		}
		qsort(s.t, s.n, sizeof *s.t, cmp_dbl);
		*emax = 0;
		for (i = 0; i < s.n; i = j) {
			double pt, fa;
			for (j = i; j < s.n && s.t[j] == s.t[i]; j++)
				;
			pt = (double)(j - i) / (double)s.n;
			fa = (double)j / (double)s.n;
			*emax += s.t[i] * (ipow(fa, warp) - ipow(fb, warp));
			agree += ipow(pt, warp);
			fb = fa;
		}
		*split = 1 - agree;
		*points = s.n;
		free(s.t);
		return 1;
	}
}

/* ---- phash ---- */

#define PH_MAXKEYS 256

struct phk {
	int64_t k, v;
};

/* the constants a condition compares x with: x == c, their ||, a small
 * range lo <= x && x <= hi; 0 when it is anything else */
static int
cond_keys(nid c, nid x, int64_t *keys, int *nk, int depth)
{
	nid a, b;

	if (depth > 64)
		return 0;
	switch (N.op[c]) {
	case O_EQ:
		a = N.a[c];
		b = N.b[c];
		if (N.op[a] == O_CAST)
			a = N.a[a];
		if (a != x || N.op[b] != O_CONST || *nk >= PH_MAXKEYS)
			return 0;
		keys[(*nk)++] = N.ki[b];
		return 1;
	case O_LOR:
		return cond_keys(N.a[c], x, keys, nk, depth + 1) &&
		    cond_keys(N.b[c], x, keys, nk, depth + 1);
	case O_LAND: {
		nid g = N.a[c], l = N.b[c];
		int64_t v;
		if (N.op[g] != O_GE || N.op[l] != O_LE || N.op[N.b[g]] != O_CONST
		    || N.op[N.b[l]] != O_CONST)
			return 0;
		a = N.a[g];
		b = N.a[l];
		if (N.op[a] == O_CAST)
			a = N.a[a];
		if (N.op[b] == O_CAST)
			b = N.a[b];
		if (a != x || b != x || N.ki[N.b[l]] - N.ki[N.b[g]] > 64)
			return 0;
		for (v = N.ki[N.b[g]]; v <= N.ki[N.b[l]]; v++) {
			if (*nk >= PH_MAXKEYS)
				return 0;
			keys[(*nk)++] = v;
		}
		return 1;
	}
	}
	return 0;
}

enum { PH_MASK, PH_MUL, PH_XSM, PH_DISP };

struct phf {
	int kind, b;		/* table of 2^b slots */
	int s, a, c;		/* shifts; c: the displacement table's bits */
	uint64_t M, M2;
	int disp[64];
};

/* the displacement table under a hash's index */
static nid
ph_table_in(nid n)
{
	nid t;

	if (N.op[n] == O_TABLE)
		return n;
	if (n == 0 || N.op[n] == O_CONST || N.op[n] == O_VAR)
		return 0;
	if (N.a[n] && (t = ph_table_in(N.a[n])))
		return t;
	if (N.b[n] && (t = ph_table_in(N.b[n])))
		return t;
	return 0;
}

static uint64_t
wmask64(int w)
{
	return w >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << w) - 1);
}

static uint64_t
ph_eval(const struct phf *h, uint64_t x, int W)
{
	uint64_t m = wmask64(W), t;

	x &= m;
	switch (h->kind) {
	case PH_MASK:
		return (x >> h->s) & wmask64(h->b);
	case PH_MUL:
		return ((x * h->M) & m) >> (W - h->b);
	case PH_XSM:
		t = x ^ (x >> h->a);
		return ((t * h->M) & m) >> (W - h->b);
	}
	t = ((x * h->M2) & m) >> (W - h->c);
	return ((((x * h->M) & m) >> (W - h->b)) + (uint64_t)h->disp[t]) &
	    wmask64(h->b);
}

static nid
ph_node(const struct phf *h, nid x)
{
	ty_t u = N.ty[x];
	int W = TYW(u);
	nid t, d[64];
	int i;

	switch (h->kind) {
	case PH_MASK:
		t = h->s ? mk_bin(O_SHR, x, mk_const_i(u, h->s)) : x;
		return mk_bin(O_AND, t, mk_const_i(u, (int64_t)wmask64(h->b)));
	case PH_MUL:
		return mk_bin(O_SHR, mk_bin(O_MUL, x, mk_const_i(u,
		    (int64_t)h->M)), mk_const_i(u, W - h->b));
	case PH_XSM:
		t = mk_bin(O_XOR, x, mk_bin(O_SHR, x, mk_const_i(u, h->a)));
		return mk_bin(O_SHR, mk_bin(O_MUL, t, mk_const_i(u,
		    (int64_t)h->M)), mk_const_i(u, W - h->b));
	}
	for (i = 0; i < (1 << h->c); i++)
		d[i] = mk_const_i(u, h->disp[i]);
	t = mk_index(mk_table(d, (uint32_t)1 << h->c), mk_bin(O_SHR,
	    mk_bin(O_MUL, x, mk_const_i(u, (int64_t)h->M2)), mk_const_i(u,
	    W - h->c)));
	return mk_bin(O_AND, mk_bin(O_ADD, mk_bin(O_SHR, mk_bin(O_MUL, x,
	    mk_const_i(u, (int64_t)h->M)), mk_const_i(u, W - h->b)), t),
	    mk_const_i(u, (int64_t)wmask64(h->b)));
}

static int
ph_injective(const struct phf *h, const struct phk *k, int n, int W)
{
	uint8_t used[1 << 12];
	int i;

	if (h->b > 12)
		return 0;
	memset(used, 0, (size_t)1 << h->b);
	for (i = 0; i < n; i++) {
		uint64_t s = ph_eval(h, (uint64_t)k[i].k, W);
		if (used[s])
			return 0;
		used[s] = 1;
	}
	return 1;
}

/* hash and displace: buckets by a second hash, the largest first, each
 * given the smallest displacement that lands its keys on free slots */
static int
ph_displace(struct phf *h, const struct phk *k, int n, int W)
{
	uint8_t used[1 << 12];
	int nb = 1 << h->c, size = 1 << h->b, cnt[64], ord[64], i, j, d;

	if (h->c > 6 || h->b > 12)
		return 0;
	memset(cnt, 0, sizeof cnt);
	memset(used, 0, (size_t)size);
	for (i = 0; i < n; i++)
		cnt[(((uint64_t)k[i].k * h->M2) & wmask64(W)) >> (W - h->c)]++;
	for (i = 0; i < nb; i++)
		ord[i] = i;
	for (i = 1; i < nb; i++) {
		int o = ord[i];
		for (j = i; j > 0 && cnt[ord[j - 1]] < cnt[o]; j--)
			ord[j] = ord[j - 1];
		ord[j] = o;
	}
	for (i = 0; i < nb; i++) {
		int bk = ord[i];
		for (d = 0; d < size; d++) {
			uint8_t mine[1 << 12];
			int okd = 1;
			memset(mine, 0, (size_t)size);
			for (j = 0; j < n && okd; j++) {
				uint64_t x = (uint64_t)k[j].k & wmask64(W), s;
				if ((int)(((x * h->M2) & wmask64(W)) >> (W - h->c)) != bk)
					continue;
				s = ((((x * h->M) & wmask64(W)) >> (W - h->b)) +
				    (uint64_t)d) & (uint64_t)(size - 1);
				if (used[s] || mine[s])
					okd = 0;
				mine[s] = 1;
			}
			if (okd) {
				h->disp[bk] = d;
				for (j = 0; j < size; j++)
					used[j] |= mine[j];
				break;
			}
		}
		if (d == size)
			return 0;
	}
	return 1;
}

static void
ph_text(const struct phf *h, int W, char *b, size_t n)
{
	switch (h->kind) {
	case PH_MASK:
		if (h->s)
			snprintf(b, n, "(x >> %d) & %llu", h->s,
			    (unsigned long long)wmask64(h->b));
		else
			snprintf(b, n, "x & %llu", (unsigned long long)wmask64(h->b));
		break;
	case PH_MUL:
		snprintf(b, n, "(x * %llu) >> %d", (unsigned long long)h->M,
		    W - h->b);
		break;
	case PH_XSM:
		snprintf(b, n, "((x xor x >> %d) * %llu) >> %d", h->a,
		    (unsigned long long)h->M, W - h->b);
		break;
	default:
		snprintf(b, n, "((x * %llu) >> %d) + D[(x * %llu) >> %d] & %llu, "
		    "D of %d", (unsigned long long)h->M, W - h->b,
		    (unsigned long long)h->M2, W - h->c,
		    (unsigned long long)wmask64(h->b), 1 << h->c);
	}
}

#define PH_TRIES 4000

static void
cmd_phash(struct func *f, uint64_t seed, const struct model *m)
{
	struct phk key[PH_MAXKEYS];
	int64_t raw[PH_MAXKEYS], lo, hi;
	int nraw = 0, nk = 0, i, j, W, b0, b, covered, found = 0, tried = 0;
	nid x, n, dflt, best_form = 0, xu, best_v = 0, best_k = 0, best_i = 0;
	ty_t u;
	struct phf best, h;
	double cchain, cbest = 0;
	char tb[200];
	struct plan p;
	nid roots[1];

	rep_begin("phash");
	rep_str("model", sym_str(m->name));
	rep_int("seed", (long long)seed);
	if (f->np != 1 || !TY_SIZED(f->p[0].ty) || f->p[0].ty == TY_BOOL ||
	    !TY_ISINT(N.ty[f->body]))
		fail("phash: a function of one machine integer with integer "
		    "values (a match or an if-chain on its keys)");
	x = mk_var(0, f->p[0].name, f->p[0].ty);
	W = TYW(f->p[0].ty);
	for (n = f->body; N.op[n] == O_SEL; n = N.c[n])
		if (!cond_keys(N.a[n], x, raw, &nraw, 0))
			break;
	dflt = n;
	if (N.op[dflt] == O_SEL || nraw < 3) {
		printf("phash %s: not a chain of comparisons of %s with at least "
		    "3 constant keys (x == k, their ||, small ranges): nothing "
		    "to hash\n", fnm(f), sym_str(f->p[0].name));
		rep_bool("rewritten", 0);
		return;
	}
	/* the keys once each, with the value the function gives there */
	param_bounds(&f->p[0], &lo, &hi);
	roots[0] = f->body;
	plan_build(&p, roots, 1, 0);
	{
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		for (i = 0; i < nraw; i++) {
			for (j = 0; j < nk; j++)
				if (key[j].k == raw[i])
					break;
			if (j < nk || raw[i] < lo || raw[i] > hi)
				continue;	/* a repeat, or a key no input takes */
			in_i[0] = raw[i];
			plan_run(&p, 1, in_i, in_r);
			key[nk].k = raw[i];
			key[nk++].v = COLI(&p, plan_loc(&p, f->body))[0];
		}
		free(in_i);
		free(in_r);
	}
	plan_free(&p);
	covered = (uint64_t)(hi - lo) + 1 == (uint64_t)nk;
	cchain = dag_cost(m, f->body);
	printf("phash %s under %s: %d keys, the compare chain costs %g; "
	    "seed %llu\n", fnm(f), sym_str(m->name), nk, cchain,
	    (unsigned long long)seed);
	rep_int("keys", nk);
	rep_num("cost_chain", cchain);
	if (nk < 3) {
		printf("  -> not rewritten: fewer than 3 keys in the parameter's "
		    "range\n");
		rep_bool("rewritten", 0);
		return;
	}
	for (b0 = 0; (1 << b0) < nk; b0++)
		;
	u = TY(K_UINT, W);
	xu = mk_cast(u, x);
	rseed(seed);
	memset(&best, 0, sizeof best);
	for (b = b0; b <= b0 + 3 && b <= 12 && b < W; b++) {
		int kind, t;
		for (kind = PH_MASK; kind <= PH_DISP; kind++) {
			int tries = kind == PH_MASK ? W - b + 1 : kind == PH_DISP ?
			    PH_TRIES / 8 : PH_TRIES;
			/* a displacement table only for the smallest table */
			if (kind == PH_DISP && b != b0)
				continue;
			for (t = 0; t < tries; t++) {
				nid form, idx, vt[1 << 12], kt[1 << 12], vtab, ktab;
				double c;
				int size = 1 << b, slot;
				memset(&h, 0, sizeof h);
				h.kind = kind;
				h.b = b;
				if (kind == PH_MASK)
					h.s = t;
				else {
					h.M = (rnd() | 1) & wmask64(W);
					h.a = W / 2 + (int)(rnd() % 5) - 2;
					if (h.a < 1)
						h.a = 1;
					h.M2 = (rnd() | 1) & wmask64(W);
					h.c = b > 2 ? b - 2 : 1;
					if (h.c > 6)
						h.c = 6;
				}
				tried++;
				if (kind == PH_DISP) {
					if (!ph_displace(&h, key, nk, W))
						continue;
				}
				if (!ph_injective(&h, key, nk, W))
					continue;
				/* the tables: a slot no key takes holds a key that
				 * hashes elsewhere, so its check fails */
				for (slot = 0; slot < size; slot++) {
					int owner = -1, other = -1;
					for (j = 0; j < nk; j++) {
						if ((int)ph_eval(&h, (uint64_t)key[j].k, W) ==
						    slot)
							owner = j;
						else if (other < 0)
							other = j;
					}
					if (owner >= 0) {
						kt[slot] = mk_const_i(f->p[0].ty, key[owner].k);
						vt[slot] = mk_const_i(N.ty[f->body],
						    key[owner].v);
					} else {
						kt[slot] = mk_const_i(f->p[0].ty, key[other].k);
						vt[slot] = mk_const_i(N.ty[f->body], 0);
					}
				}
				idx = ph_node(&h, xu);
				vtab = mk_table(vt, (uint32_t)size);
				ktab = mk_table(kt, (uint32_t)size);
				form = mk_index(vtab, idx);
				if (!covered)
					form = mk_sel(mk_bin(O_EQ, mk_index(ktab, idx), x),
					    form, dflt);
				c = dag_cost(m, form);
				if (!found || c < cbest - 1e-9) {
					found = 1;
					cbest = c;
					best = h;
					best_form = form;
					best_v = vtab;
					best_k = ktab;
					best_i = idx;
				}
				/* the first of a kind is its cheapest shape */
				break;
			}
		}
		if (found && b >= b0 + 1)
			break;	/* a larger table costs the same instructions */
	}
	printf("  %d hash functions tried: masks and shifts, (x K) >> s, "
	    "((x xor x >> a) K) >> s, and a displacement table, K drawn at "
	    "random\n", tried);
	if (!found) {
		printf("  -> not rewritten: none is injective on the keys into "
		    "a table of up to %d slots\n", 1 << (b0 + 3 > 12 ? 12 :
		    b0 + 3));
		rep_bool("rewritten", 0);
		return;
	}
	ph_text(&best, W, tb, sizeof tb);
	printf("  injective on the %d keys (each slot checked): h(x) = %s, "
	    "a table of %d slots\n", nk, tb, 1 << best.b);
	if (covered)
		printf("  every value of %s in %lld..%lld is a key: no key check "
		    "is needed\n", sym_str(f->p[0].name), (long long)lo,
		    (long long)hi);
	else
		printf("  other inputs: the slot's key is compared with x, and "
		    "the default taken when they differ\n");
	printf("  the lookup costs %g\n", cbest);
	rep_str("hash", tb);
	rep_int("slots", 1 << best.b);
	rep_num("cost_lookup", cbest);
	if (cbest >= cchain) {
		printf("  -> not rewritten: the lookup costs %g, the compare "
		    "chain %g under %s\n", cbest, cchain, sym_str(m->name));
		rep_bool("rewritten", 0);
		return;
	}
	{
		struct sbuf w = {0}, how = {0};
		struct eqproof pr;
		int bad = equiv_decide(f, f->body, best_form, &w, &pr);
		if (bad || pr.how == EQ_SAMPLED) {
			printf("  -> not rewritten: the lookup is not proved equal "
			    "to the switch%s%s\n", bad ? ": they differ at " : "",
			    bad ? sb_str(&w) : " (sampled only)");
			rep_bool("rewritten", 0);
			free(w.p);
			return;
		}
		sb_puts(&how, pr.how == EQ_SAT ? "proved by SAT" :
		    "every input enumerated");
		printf("  -> %s\n     cost %g against %g (%+.1f%%); equal to "
		    "the switch for every input: %s\n", shared_str(best_form,
		    400), cbest, cchain, 100 * (cbest - cchain) / cchain,
		    sb_str(&how));
		rep_proof(pr.how == EQ_SAT ? "sat" : "exhaustive", sb_str(&how));
		free(w.p);
		free(how.p);
	}
	/* the tables as BC4, where that holds them exactly */
	texenc_note(best_v, 0, "values", "     ");
	if (!covered)
		texenc_note(best_k, 0, "keys", "     ");
	if (best.kind == PH_DISP)
		texenc_note(ph_table_in(best_i), 0, "displacements", "     ");
	{
		struct func copy = *f, *g;
		char nm[96];
		snprintf(nm, sizeof nm, "%s_ph", f->anon ? "" : sym_str(f->name));
		copy.name = sym_cstr(nm);
		copy.anon = 0;
		copy.body = best_form;
		g = func_new(copy.name);
		*g = copy;
		printf("  defined %s\n", nm);
		rep_bool("rewritten", 1);
		rep_expr("body", best_form);
		rep_str("defined", nm);
	}
}

/* ---- stoke ---- */

enum { SO_UNUSED, SO_ADD, SO_SUB, SO_MUL, SO_AND, SO_OR, SO_XOR, SO_SHL,
	SO_SHR, SO_NOT, SO_NEG, SO_SAR, SO_LTU, SO_LTS, SO_EQ, SO_SEL, SO_ZX,
	SO_SX, SO_LOAD, SO_DOT, SO_N };
static const char *const so_name[SO_N] = {"-", "add", "sub", "mul", "and",
	"or", "xor", "shl", "shr", "not", "neg", "sar", "ltu", "lts", "eq",
	"sel", "zext", "sext", "load", "dot"};

#define ST_MAXLEN 12
#define ST_MAXT 256
#define ST_MAXTAB 4
#define ST_MAXW 6

struct sopnd {
	uint8_t kind;		/* 0 input, 1 an earlier result, 2 constant */
	uint8_t ix;
	uint64_t k;
};

/*
 * An instruction w bits wide: its operands are read at w (zero-extended
 * or cut), its value is kept at w.  zext and sext read their operand at
 * its own width, a load its index; c is a select's else value and a
 * saturating dot's accumulator; aux the table or the dot form.
 */
struct sins1 {
	uint8_t op, w, aux;
	struct sopnd a, b, c;
};

struct sprog {
	struct sins1 in[ST_MAXLEN];
};

static struct {
	const struct func *f;
	const struct model *m;
	int W, len, np, nt;
	uint64_t mask;
	uint64_t tin[ST_MAXT][MAXP], tout[ST_MAXT];
	int pw[MAXP];		/* each parameter's width */
	int nw, wid[ST_MAXW];	/* the widths an instruction takes */
	int nops, ops[SO_N];	/* the operations proposed */
	int ntab;		/* f's tables, for loads */
	nid tab[ST_MAXTAB];
	int64_t *tv[ST_MAXTAB];
	uint32_t tn[ST_MAXTAB];
	int ndot, dot[O_DOT_LAST - O_DOT_FIRST + 1];	/* priced dot forms */
	int kw, npool;		/* constants: the widest width, the pool's size */
	uint64_t kmask;
	/* statistics */
	uint64_t proposals, accepted, zeros, verified, cex;
} ST;

/* operands read: 1, 2, or 3 (a select, a saturating dot) */
static int
so_arity(const struct sins1 *I)
{
	switch (I->op) {
	case SO_NOT: case SO_NEG: case SO_ZX: case SO_SX: case SO_LOAD:
		return 1;
	case SO_SEL:
		return 3;
	case SO_DOT:
		return dot_form(ST.dot[I->aux])->sat ? 3 : 2;
	}
	return 2;
}

/* the operand is read at its own width (not cut or extended to w) */
static int
so_own(int op)
{
	return op == SO_ZX || op == SO_SX || op == SO_LOAD;
}

static uint64_t
sopnd_val(const struct sopnd *o, const uint64_t *in, const uint64_t *tmp)
{
	return o->kind == 0 ? in[o->ix] : o->kind == 1 ? tmp[o->ix] : o->k;
}

/* the width an operand of an instruction w bits wide has */
static int
sopnd_w(const struct sprog *p, const struct sopnd *o, int w)
{
	return o->kind == 0 ? ST.pw[o->ix] : o->kind == 1 && p->in[o->ix].op !=
	    SO_UNUSED ? p->in[o->ix].w : w;
}

static int64_t
sx64(uint64_t v, int w)
{
	return w >= 64 ? (int64_t)v : (int64_t)(v << (64 - w)) >> (64 - w);
}

/* the program's value on one input; *bad: a shift by the width or more,
 * a load past its table, a signed dot that overflows */
static uint64_t
sp_run(const struct sprog *p, const uint64_t *in, int *bad)
{
	uint64_t tmp[ST_MAXLEN], out = 0, a, b, c, m;
	int i, w;

	for (i = 0; i < ST.len; i++) {
		const struct sins1 *I = &p->in[i];
		if (I->op == SO_UNUSED) {
			tmp[i] = 0;
			continue;
		}
		w = I->w;
		m = wmask64(w);
		a = sopnd_val(&I->a, in, tmp);
		b = sopnd_val(&I->b, in, tmp) & m;
		c = sopnd_val(&I->c, in, tmp) & m;
		if (!so_own(I->op) || I->a.kind == 2)
			a &= m;
		switch (I->op) {
		case SO_ADD: a += b; break;
		case SO_SUB: a -= b; break;
		case SO_MUL: a *= b; break;
		case SO_AND: a &= b; break;
		case SO_OR: a |= b; break;
		case SO_XOR: a ^= b; break;
		case SO_SHL: case SO_SHR: case SO_SAR:
			if (b >= (uint64_t)w)
				*bad = 1, a = 0;
			else
				a = I->op == SO_SHL ? a << b : I->op == SO_SHR ? a >> b :
				    (uint64_t)(sx64(a, w) >> b);
			break;
		case SO_NOT: a = ~a; break;
		case SO_NEG: a = (uint64_t)0 - a; break;
		case SO_LTU: a = a < b; break;
		case SO_LTS: a = sx64(a, w) < sx64(b, w); break;
		case SO_EQ: a = a == b; break;
		case SO_SEL: a = a ? b : c; break;
		case SO_ZX: break;
		case SO_SX: a = (uint64_t)sx64(a, sopnd_w(p, &I->a, w)); break;
		case SO_LOAD:
			if (a >= ST.tn[I->aux])
				*bad = 1, a = 0;
			else
				a = (uint64_t)ST.tv[I->aux][a];
			break;
		case SO_DOT: {
			const struct dotform *df = dot_form(ST.dot[I->aux]);
			uint8_t fl = 0;
			a = (uint64_t)dot_scalar(ST.dot[I->aux], 0, (int64_t)a,
			    (int64_t)b, df->sa || df->sb ? sx64(c, 32) : (int64_t)c,
			    &fl);
			if (fl & F_SW)
				*bad = 1;
			break;
		}
		}
		tmp[i] = out = a & m;
	}
	return out & ST.mask;
}

static int
popc(uint64_t v)
{
	int n = 0;

	for (; v; v &= v - 1)
		n++;
	return n;
}

/*
 * The program's price: the report's own, the cover of its expression by
 * the model's fusions (a * b + c one imad, a shift and an add one
 * instruction), so what the result does not read costs nothing.  A chain
 * meets the same programs again and again: the prices are kept by the
 * program's live instructions, renumbered, and the expression is built
 * only for a program not met lately.
 */
#define ST_CACHE 4096
#define ST_KEY (5 * ST_MAXLEN)

static struct {
	uint64_t (*key)[ST_KEY];
	uint64_t *h;
	int *len;		/* 0: an empty slot */
	double *cost;
} SC;

static void
sc_free(void)
{
	free(SC.key);
	free(SC.h);
	free(SC.len);
	free(SC.cost);
	memset(&SC, 0, sizeof SC);
}

/* the live instructions, renumbered (dead code and unused slots gone, a
 * read of an unused slot the constant 0 it reads as); their count */
static int
sp_live(const struct sprog *p, struct sins1 *out)
{
	uint8_t live[ST_MAXLEN] = {0};
	int map[ST_MAXLEN], i, q, last = -1, n = 0;

	for (i = 0; i < ST.len; i++)
		if (p->in[i].op != SO_UNUSED)
			last = i;
	if (last < 0)
		return 0;
	live[last] = 1;
	for (i = last; i >= 0; i--) {
		const struct sins1 *I = &p->in[i];
		if (!live[i])
			continue;
		for (q = 0; q < so_arity(I); q++) {
			const struct sopnd *o = q == 0 ? &I->a : q == 1 ? &I->b : &I->c;
			if (o->kind == 1 && p->in[o->ix].op != SO_UNUSED)
				live[o->ix] = 1;
		}
	}
	for (i = 0; i <= last; i++) {
		struct sins1 I;
		if (!live[i])
			continue;
		memset(&I, 0, sizeof I);
		I.op = p->in[i].op;
		I.w = p->in[i].w;
		I.aux = p->in[i].aux;
		for (q = 0; q < so_arity(&I); q++) {
			const struct sopnd *s = q == 0 ? &p->in[i].a : q == 1 ?
			    &p->in[i].b : &p->in[i].c;
			struct sopnd *o = q == 0 ? &I.a : q == 1 ? &I.b : &I.c;
			o->kind = s->kind;
			if (s->kind == 0)
				o->ix = s->ix;
			else if (s->kind == 1 && p->in[s->ix].op != SO_UNUSED)
				o->ix = (uint8_t)map[s->ix];
			else {
				/* an unused slot reads as 0 at its reader's width */
				o->kind = 2;
				o->k = s->kind == 2 ? s->k & wmask64(I.w) : 0;
			}
		}
		map[i] = n;
		out[n++] = I;
	}
	return n;
}

/* the live instructions as words: five an instruction */
static int
sp_key(const struct sins1 *in, int n, uint64_t *key)
{
	int i, at = 0;

	for (i = 0; i < n; i++) {
		key[at++] = (uint64_t)in[i].op | (uint64_t)in[i].w << 8 |
		    (uint64_t)in[i].aux << 16 | (uint64_t)in[i].a.kind << 24 |
		    (uint64_t)in[i].a.ix << 32 | (uint64_t)in[i].b.kind << 40 |
		    (uint64_t)in[i].b.ix << 48;
		key[at++] = (uint64_t)in[i].c.kind | (uint64_t)in[i].c.ix << 8;
		key[at++] = in[i].a.k;
		key[at++] = in[i].b.k;
		key[at++] = in[i].c.k;
	}
	return at;
}

static nid sp_node(const struct sprog *p);

static double
sp_perf(const struct sprog *p)
{
	struct sins1 in[ST_MAXLEN];
	uint64_t key[ST_KEY], h = 0x9e3779b97f4a7c15ULL;
	int n = sp_live(p, in), kn, i;
	size_t s;

	if (n == 0)
		return 0;
	kn = sp_key(in, n, key);
	for (i = 0; i < kn; i++) {
		h = (h ^ key[i]) * 0xff51afd7ed558ccdULL;
		h ^= h >> 32;
	}
	if (SC.key == NULL) {
		SC.key = xcalloc(ST_CACHE, sizeof *SC.key);
		SC.h = xcalloc(ST_CACHE, sizeof *SC.h);
		SC.len = xcalloc(ST_CACHE, sizeof *SC.len);
		SC.cost = xcalloc(ST_CACHE, sizeof *SC.cost);
	}
	s = (size_t)(h % ST_CACHE);
	if (SC.len[s] == kn && SC.h[s] == h && !memcmp(SC.key[s], key,
	    (size_t)kn * sizeof *key))
		return SC.cost[s];
	SC.h[s] = h;
	SC.len[s] = kn;
	memcpy(SC.key[s], key, (size_t)kn * sizeof *key);
	return SC.cost[s] = dag_cost(ST.m, sp_node(p));
}

static double
sp_cost(const struct sprog *p, int *dist)
{
	int d = 0, t, bad;

	for (t = 0; t < ST.nt; t++) {
		bad = 0;
		d += popc(sp_run(p, ST.tin[t], &bad) ^ ST.tout[t]);
		if (bad)
			d += ST.W;
	}
	*dist = d;
	return (double)d + sp_perf(p);
}

/* the constants a proposal draws; the last, a byte of ones in each lane,
 * only when dot forms are in the set */
static const uint64_t so_pool[] = {0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32,
	63, 64, 127, 255, 0x5555555555555555ULL, 0x3333333333333333ULL,
	0x0f0f0f0f0f0f0f0fULL, 0xffff, ~(uint64_t)0, 0x0101010101010101ULL};

static void
rand_opnd(struct sopnd *o, int at)
{
	double r = runif();

	if (r < 0.4 || (at == 0 && r < 0.75)) {
		o->kind = 0;
		o->ix = (uint8_t)(rnd() % (uint64_t)ST.np);
	} else if (r < 0.75) {
		o->kind = 1;
		o->ix = (uint8_t)(rnd() % (uint64_t)at);
	} else {
		o->kind = 2;
		o->k = so_pool[rnd() % (uint64_t)ST.npool] & ST.kmask;
	}
}

/* an operation, and what it takes: its width (a dot is 32 bits wide), its
 * table or dot form, its third operand.  With one width and none of the
 * others nothing more is drawn: the add ... neg searches run as before */
static void
rand_op(struct sins1 *I, int at)
{
	I->op = (uint8_t)ST.ops[rnd() % (uint64_t)ST.nops];
	if (I->op == SO_DOT)
		I->w = 32;
	else
		I->w = (uint8_t)ST.wid[ST.nw > 1 ? rnd() % (uint64_t)ST.nw : 0];
	if (I->op == SO_LOAD)
		I->aux = (uint8_t)(rnd() % (uint64_t)ST.ntab);
	if (I->op == SO_DOT)
		I->aux = (uint8_t)(rnd() % (uint64_t)ST.ndot);
	if (so_arity(I) == 3)
		rand_opnd(&I->c, at);
}

static void
rand_ins(struct sins1 *I, int at)
{
	rand_op(I, at);
	rand_opnd(&I->a, at);
	rand_opnd(&I->b, at);
}

static void
sp_propose(struct sprog *p)
{
	int i = (int)(rnd() % (uint64_t)ST.len), j, ar;
	struct sins1 *I = &p->in[i], t;
	double r = runif();

	if (r < 0.2) {			/* the opcode, or the width alone */
		if (I->op != SO_UNUSED && I->op != SO_DOT && ST.nw > 1 &&
		    runif() < 0.5)
			I->w = (uint8_t)ST.wid[rnd() % (uint64_t)ST.nw];
		else if (I->op != SO_UNUSED)
			rand_op(I, i);
	} else if (r < 0.5) {		/* an operand */
		ar = I->op == SO_UNUSED ? 2 : so_arity(I);
		rand_opnd(ar == 3 ? (j = (int)(rnd() % 3)) == 0 ? &I->a : j == 1 ?
		    &I->b : &I->c : rnd() & 1 ? &I->a : &I->b, i);
	} else if (r < 0.6) {		/* two instructions swapped */
		j = (int)(rnd() % (uint64_t)ST.len);
		t = p->in[i];
		p->in[i] = p->in[j];
		p->in[j] = t;
		/* an operand may not point forward */
		for (j = 0; j < ST.len; j++) {
			if (p->in[j].a.kind == 1 && p->in[j].a.ix >= j)
				rand_opnd(&p->in[j].a, j);
			if (p->in[j].b.kind == 1 && p->in[j].b.ix >= j)
				rand_opnd(&p->in[j].b, j);
			if (p->in[j].c.kind == 1 && p->in[j].c.ix >= j)
				rand_opnd(&p->in[j].c, j);
		}
	} else if (r < 0.8) {		/* a whole instruction, or none */
		if (rnd() & 1)
			I->op = SO_UNUSED;
		else
			rand_ins(I, i);
	} else {			/* a constant */
		struct sopnd *o = I->b.kind == 2 ? &I->b : I->a.kind == 2 ? &I->a :
		    NULL;
		if (o == NULL)
			return;
		switch (rnd() % 5) {
		case 0: o->k += 1; break;
		case 1: o->k -= 1; break;
		case 2: o->k ^= (uint64_t)1 << (rnd() % (uint64_t)ST.kw); break;
		case 3: o->k <<= 1; break;
		default: o->k >>= 1;
		}
		o->k &= ST.kmask;
	}
}

/* an operand as an expression: at the instruction's width, or at its own
 * (own) for zext, sext and a load's index */
static nid
sp_opnd(const struct sins1 *I, const struct sopnd *o, const nid *tmp,
    int own)
{
	ty_t u = TY(K_UINT, I->w);
	nid v;

	if (o->kind == 2)
		return mk_const_i(u, (int64_t)(o->k & wmask64(I->w)));
	if (o->kind == 0)
		v = mk_cast(TY(K_UINT, ST.pw[o->ix]), mk_var(o->ix,
		    ST.f->p[o->ix].name, ST.f->p[o->ix].ty));
	else
		v = tmp[o->ix] ? tmp[o->ix] : mk_const_i(u, 0);
	return own ? v : mk_cast(u, v);
}

/* the program as an expression over f's parameters */
static nid
sp_node(const struct sprog *p)
{
	nid tmp[ST_MAXLEN], out = mk_const_i(TY(K_UINT, ST.W), 0), a, b, c;
	int i;

	for (i = 0; i < ST.len; i++) {
		const struct sins1 *I = &p->in[i];
		ty_t u = TY(K_UINT, I->w), s = TY(K_SINT, I->w);
		tmp[i] = 0;
		if (I->op == SO_UNUSED)
			continue;
		a = sp_opnd(I, &I->a, tmp, so_own(I->op));
		b = so_arity(I) > 1 ? sp_opnd(I, &I->b, tmp, 0) : 0;
		c = so_arity(I) > 2 ? sp_opnd(I, &I->c, tmp, 0) : 0;
		switch (I->op) {
		case SO_ADD: a = mk_bin(O_ADD, a, b); break;
		case SO_SUB: a = mk_bin(O_SUB, a, b); break;
		case SO_MUL: a = mk_bin(O_MUL, a, b); break;
		case SO_AND: a = mk_bin(O_AND, a, b); break;
		case SO_OR: a = mk_bin(O_OR, a, b); break;
		case SO_XOR: a = mk_bin(O_XOR, a, b); break;
		case SO_SHL: a = mk_bin(O_SHL, a, b); break;
		case SO_SHR: a = mk_bin(O_SHR, a, b); break;
		case SO_NOT: a = mk_un(O_NOT, a); break;
		case SO_NEG: a = mk_un(O_NEG, a); break;
		case SO_SAR: a = mk_cast(u, mk_bin(O_SHR, mk_cast(s, a), b)); break;
		case SO_LTU: a = mk_cast(u, mk_bin(O_LT, a, b)); break;
		case SO_LTS:
			a = mk_cast(u, mk_bin(O_LT, mk_cast(s, a), mk_cast(s, b)));
			break;
		case SO_EQ: a = mk_cast(u, mk_bin(O_EQ, a, b)); break;
		case SO_SEL:
			a = mk_sel(mk_bin(O_NE, a, mk_const_i(u, 0)), b, c);
			break;
		case SO_ZX: a = mk_cast(u, a); break;
		case SO_SX:
			a = mk_cast(u, mk_cast(s, mk_cast(TY(K_SINT, sopnd_w(p, &I->a,
			    I->w)), a)));
			break;
		case SO_LOAD: a = mk_cast(u, mk_index(ST.tab[I->aux], a)); break;
		default:	/* SO_DOT */
			a = mk_cast(u, mk_dot(ST.dot[I->aux], a, b, c));
		}
		tmp[i] = out = a;
	}
	return mk_cast(N.ty[ST.f->body], out);
}

/* f's own body as a program, when it is straight-line in these
 * operations; 0 when it is not */
static int lower_at;
static struct sprog *lower_p;
static nid lower_n[ST_MAXLEN];

static int sp_lower(nid e, struct sopnd *o);
static int sp_has_op(int op);

/* a new instruction for e (I filled but for its operands' lowering) */
static int
sp_lower_add(nid e, struct sins1 *I, struct sopnd *o)
{
	if (lower_at >= ST.len)
		return 0;
	lower_n[lower_at] = e;
	lower_p->in[lower_at] = *I;
	o->kind = 1;
	o->ix = (uint8_t)lower_at++;
	return 1;
}

/* the width e is lowered at: its own, or for a compare its operands' */
static int
sp_lower_w(nid e)
{
	ty_t t = N.ty[e];

	if (t == TY_BOOL && N.a[e] && TY_SIZED(N.ty[N.a[e]]))
		return TYW(N.ty[N.a[e]]);
	return TY_SIZED(t) ? TYW(t) : 0;
}

static int
sp_has_w(int w)
{
	int i;

	for (i = 0; i < ST.nw; i++)
		if (ST.wid[i] == w)
			return 1;
	return 0;
}

/* x = tab[i] read at width w, as the instruction for e */
static int
sp_lower_load(nid x, int w, nid e, struct sopnd *o)
{
	struct sins1 I;
	int i;

	for (i = 0; i < ST.ntab && ST.tab[i] != N.a[x]; i++)
		;
	if (i == ST.ntab || !sp_has_op(SO_LOAD) || !sp_has_w(w))
		return 0;
	memset(&I, 0, sizeof I);
	I.op = SO_LOAD;
	I.w = (uint8_t)w;
	I.aux = (uint8_t)i;
	if (!sp_lower(N.b[x], &I.a))
		return 0;
	return sp_lower_add(e, &I, o);
}

static int
sp_lower(nid e, struct sopnd *o)
{
	static const struct { int op, so; } map[] = {{O_ADD, SO_ADD},
		{O_SUB, SO_SUB}, {O_MUL, SO_MUL}, {O_AND, SO_AND}, {O_OR, SO_OR},
		{O_XOR, SO_XOR}, {O_SHL, SO_SHL}, {O_SHR, SO_SHR}, {O_NOT, SO_NOT},
		{O_NEG, SO_NEG}, {0, 0}};
	ty_t t = N.ty[e];
	struct sins1 I;
	int i, so = 0, w = sp_lower_w(e), op = N.op[e];

	if (op == O_CONST && TY_ISINT(t)) {
		o->kind = 2;
		o->k = (uint64_t)N.ki[e] & ST.kmask;
		return 1;
	}
	if (op == O_VAR) {
		if (!TY_SIZED(t) || N.ki[e] < 0 || N.ki[e] >= ST.np)
			return 0;
		o->kind = 0;
		o->ix = (uint8_t)N.ki[e];
		return 1;
	}
	for (i = 0; i < lower_at; i++)
		if (lower_n[i] == e) {
			o->kind = 1;
			o->ix = (uint8_t)i;
			return 1;
		}
	if (w == 0 || !sp_has_w(w))
		return 0;
	memset(&I, 0, sizeof I);
	I.w = (uint8_t)w;
	switch (op) {
	case O_CAST: {
		ty_t ta = N.ty[N.a[e]];
		int wa = sp_lower_w(N.a[e]);
		/* a table of literals read as a machine integer: the load at
		 * that width */
		if (TY_ISINT(t) && !TY_SIZED(ta) && N.op[N.a[e]] == O_INDEX)
			return sp_lower_load(N.a[e], w, e, o);
		if (!TY_ISINT(t) || !(TY_SIZED(ta) || ta == TY_BOOL) || wa == 0)
			return 0;
		if (wa == w)	/* a sign read again, or 0/1 at its width */
			return sp_lower(N.a[e], o);
		I.op = wa < w && TYK(ta) == K_SINT ? SO_SX : SO_ZX;
		if (!sp_has_op(I.op) || !sp_lower(N.a[e], &I.a))
			return 0;
		return sp_lower_add(e, &I, o);
	}
	case O_LT: case O_GT: case O_LE: case O_GE: case O_EQ: case O_NE: {
		/* a < b; b < a; !(b < a); !(a < b); a == b; !(a == b) */
		int sw = op == O_GT || op == O_LE, neg = op == O_LE || op == O_GE ||
		    op == O_NE;
		ty_t ta = N.ty[N.a[e]];
		struct sopnd r;
		if (!TY_SIZED(ta) || !TY_ISINT(ta))
			return 0;
		I.op = op == O_EQ || op == O_NE ? SO_EQ : TYK(ta) == K_SINT ? SO_LTS :
		    SO_LTU;
		if (!sp_has_op(I.op) || !sp_lower(N.a[e], sw ? &I.b : &I.a) ||
		    !sp_lower(N.b[e], sw ? &I.a : &I.b))
			return 0;
		if (!neg)
			return sp_lower_add(e, &I, o);
		/* the instruction for the compare, then xor 1 for e */
		if (lower_at >= ST.len)
			return 0;
		lower_n[lower_at] = 0;
		lower_p->in[lower_at] = I;
		r.kind = 1;
		r.ix = (uint8_t)lower_at++;
		r.k = 0;
		memset(&I, 0, sizeof I);
		I.op = SO_XOR;
		I.w = (uint8_t)w;
		I.a = r;
		I.b.kind = 2;
		I.b.k = 1;
		return sp_lower_add(e, &I, o);
	}
	case O_SEL:
		if (!TY_SIZED(t) || !sp_has_op(SO_SEL))
			return 0;
		I.op = SO_SEL;
		if (!sp_lower(N.a[e], &I.a) || !sp_lower(N.b[e], &I.b) ||
		    !sp_lower(N.c[e], &I.c))
			return 0;
		return sp_lower_add(e, &I, o);
	case O_INDEX:
		return sp_lower_load(e, w, e, o);
	}
	if (IS_DOT(op)) {
		for (i = 0; i < ST.ndot && ST.dot[i] != op; i++)
			;
		if (i == ST.ndot || w != 32)
			return 0;
		I.op = SO_DOT;
		I.aux = (uint8_t)i;
		if (!sp_lower(N.a[e], &I.a) || !sp_lower(N.b[e], &I.b) ||
		    (dot_form(op)->sat && !sp_lower(N.c[e], &I.c)))
			return 0;
		return sp_lower_add(e, &I, o);
	}
	if (!TY_SIZED(t) || !TY_ISINT(t))
		return 0;
	for (i = 0; map[i].so; i++)
		if (map[i].op == op)
			so = map[i].so;
	if (so == SO_SHR && TYK(t) == K_SINT)
		so = SO_SAR;
	if (!so || !sp_has_op(so))
		return 0;
	I.op = (uint8_t)so;
	if (!sp_lower(N.a[e], &I.a))
		return 0;
	if (so == SO_NOT || so == SO_NEG)
		I.b = I.a;
	else if (!sp_lower(N.b[e], &I.b))
		return 0;
	return sp_lower_add(e, &I, o);
}

static int
sp_has_op(int op)
{
	int i;

	for (i = 0; i < ST.nops; i++)
		if (ST.ops[i] == op)
			return 1;
	return 0;
}

static void
sp_add_w(int w)
{
	if (w > 0 && w <= 64 && !sp_has_w(w) && ST.nw < ST_MAXW)
		ST.wid[ST.nw++] = w;
}

/*
 * What a search draws from: add ... neg always; the compares, the
 * select and the arithmetic shift where f has a compare, a select or a
 * signed shift; zext and sext where f's values have more than one width
 * (each of them a width an instruction takes, f's own first); a load
 * for each of f's tables of integer constants; the dot forms the model
 * prices where one of f's widths is 32.  With none of them it is the
 * search it always was, draw for draw.
 */
static void
sp_setup(const struct func *f, const struct model *m)
{
	nid seen[1024], st[1024], e;
	int nseen = 0, sp = 0, i, j, cmpsel = 0;

	ST.nw = 0;
	sp_add_w(ST.W);
	for (j = 0; j < ST.np; j++) {
		ST.pw[j] = TYW(f->p[j].ty);
		sp_add_w(ST.pw[j]);
	}
	st[sp++] = f->body;
	while (sp > 0) {
		e = st[--sp];
		for (i = 0; i < nseen && seen[i] != e; i++)
			;
		if (i < nseen || nseen == 1024)
			continue;
		seen[nseen++] = e;
		if (TY_SIZED(N.ty[e]) && TY_ISINT(N.ty[e]))
			sp_add_w(TYW(N.ty[e]));
		if ((opinfo[N.op[e]].fl & OF_CMP) || N.op[e] == O_SEL ||
		    (N.op[e] == O_SHR && TYK(N.ty[e]) == K_SINT))
			cmpsel = 1;
		if (N.op[e] == O_INDEX && N.op[N.a[e]] == O_TABLE &&
		    ST.ntab < ST_MAXTAB) {
			nid tb = N.a[e];
			uint32_t k;
			for (i = 0; i < ST.ntab && ST.tab[i] != tb; i++)
				;
			for (k = 0; i == ST.ntab && k < N.b[tb]; k++) {
				nid x = N.lst[N.a[tb] + k];
				if (N.op[x] != O_CONST || (!TY_ISINT(N.ty[x]) &&
				    TYK(N.ty[x]) != K_LIT))
					break;
			}
			if (i == ST.ntab && k == N.b[tb]) {
				ST.tab[ST.ntab] = tb;
				ST.tn[ST.ntab] = N.b[tb];
				ST.tv[ST.ntab] = xmalloc(((size_t)N.b[tb] + 1) *
				    sizeof *ST.tv[ST.ntab]);
				for (k = 0; k < N.b[tb]; k++)
					ST.tv[ST.ntab][k] = N.ki[N.lst[N.a[tb] + k]];
				ST.ntab++;
			}
		}
		if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_TABLE ||
		    N.op[e] == O_UNDEF)
			continue;
		if (N.a[e] && sp < 1024)
			st[sp++] = N.a[e];
		if (N.b[e] && sp < 1024)
			st[sp++] = N.b[e];
		if (N.c[e] && sp < 1024)
			st[sp++] = N.c[e];
	}
	for (i = O_DOT_FIRST; i <= O_DOT_LAST && sp_has_w(32); i++)
		if (cclass_available(m, DOT_CLASS(i)))
			ST.dot[ST.ndot++] = i;
	ST.nops = 0;
	for (i = SO_ADD; i <= SO_NEG; i++)
		ST.ops[ST.nops++] = i;
	if (cmpsel)
		for (i = SO_SAR; i <= SO_SEL; i++)
			ST.ops[ST.nops++] = i;
	if (ST.nw > 1) {
		ST.ops[ST.nops++] = SO_ZX;
		ST.ops[ST.nops++] = SO_SX;
	}
	if (ST.ntab)
		ST.ops[ST.nops++] = SO_LOAD;
	if (ST.ndot)
		ST.ops[ST.nops++] = SO_DOT;
	ST.kw = ST.W;
	for (i = 0; i < ST.nw; i++)
		if (ST.wid[i] > ST.kw)
			ST.kw = ST.wid[i];
	ST.kmask = wmask64(ST.kw);
	ST.npool = (int)(sizeof so_pool / sizeof so_pool[0]) - (ST.ndot ? 0 : 1);
}

static void
sp_release(void)
{
	int i;

	for (i = 0; i < ST.ntab; i++)
		free(ST.tv[i]);
	ST.ntab = 0;
}

/* "add sub ... neg", then what f adds: the compares, the widths, the
 * tables, the dot forms by name */
static void
sp_set_text(struct sbuf *b)
{
	int i;

	for (i = 0; i < ST.nops; i++) {
		if (ST.ops[i] == SO_DOT) {
			int d;
			for (d = 0; d < ST.ndot; d++)
				sb_printf(b, " %s", dot_form(ST.dot[d])->name);
		} else
			sb_printf(b, "%s%s", i ? " " : "", so_name[ST.ops[i]]);
	}
	if (ST.nw > 1) {
		sb_printf(b, "; widths");
		for (i = 0; i < ST.nw; i++)
			sb_printf(b, " %d", ST.wid[i]);
	}
	if (ST.ntab)
		sb_printf(b, "; %d table%s", ST.ntab, ST.ntab == 1 ? "" : "s");
}

/* a test input: f's value there, when it is defined */
static int
st_add_test(const int64_t *in)
{
	static struct plan p;
	static const struct func *pf;
	static int64_t *in_i;
	static double *in_r;
	nid roots[2];
	uint8_t ok[CH];
	uint32_t loc, wloc;
	int j, t;

	if (in == NULL) {		/* done: release */
		if (pf != NULL)
			plan_free(&p);
		pf = NULL;
		return 0;
	}
	if (pf != ST.f) {
		if (pf != NULL)
			plan_free(&p);
		roots[0] = ST.f->body;
		roots[1] = ST.f->where;
		plan_build(&p, roots, 2, 0);
		pf = ST.f;
		if (in_i == NULL) {
			in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
			in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		}
	}
	if (ST.nt >= ST_MAXT)
		return 0;
	for (t = 0; t < ST.nt; t++) {
		for (j = 0; j < ST.np; j++)
			if (ST.tin[t][j] != ((uint64_t)in[j] & wmask64(ST.pw[j])))
				break;
		if (j == ST.np)
			return 0;	/* already a test */
	}
	for (j = 0; j < ST.np; j++)
		in_i[(size_t)j * CH] = in[j];
	loc = plan_loc(&p, ST.f->body);
	wloc = ST.f->where ? plan_loc(&p, ST.f->where) : NOTOK;
	plan_run(&p, 1, in_i, in_r);
	valid_mask(&p, wloc, 1, ok);
	if (!ok[0] || (COLF(&p, loc)[0] & F_UNDEF))
		return 0;
	for (j = 0; j < ST.np; j++)
		ST.tin[ST.nt][j] = (uint64_t)in[j] & wmask64(ST.pw[j]);
	ST.tout[ST.nt++] = (uint64_t)COLI(&p, loc)[0] & ST.mask;
	return 1;
}

#define ST_MAXFOUND 8
struct sfound {
	nid e;
	double cost;
	int ops;		/* at one price the fewer operations first */
	char how[160];
	int proved;		/* 2 proved, 1 probabilistic */
};

/*
 * A program at distance 0 on the tests: equal to f on the whole domain?
 * Enumerated when the domain is small, else SAT, else (polynomials)
 * identity testing.  A counterexample becomes a test.
 */
static int
st_verify(nid e, char *how, size_t n, uint64_t seed)
{
	struct sbuf w = {0}, t = {0};
	struct eqproof pr;
	double bound;
	int bad, r;

	equiv_algebra = 1;
	bad = equiv_decide(ST.f, ST.f->body, e, &w, &pr);
	equiv_algebra = 0;
	if (bad) {
		/* the witness names the point: "x=3, y=5" */
		int64_t in[MAXP] = {0};
		const char *q = w.p ? w.p : "";
		int j;
		for (j = 0; j < ST.np; j++) {
			const char *eq = strchr(q, '=');
			if (eq == NULL)
				break;
			in[j] = strtoll(eq + 1, NULL, 0);
			q = eq + 1;
		}
		if (pr.sat_status == 1)
			memcpy(in, pr.sat.in, sizeof in);
		if (j == ST.np || pr.sat_status == 1)
			ST.cex += (uint64_t)st_add_test(in);
		free(w.p);
		return 0;
	}
	free(w.p);
	if (pr.how == EQ_EXHAUSTIVE) {
		snprintf(how, n, "every input enumerated");
		return 2;
	}
	if (pr.how == EQ_SAT) {
		snprintf(how, n, "proved by SAT");
		return 2;
	}
	if (pr.how == EQ_ALGEBRA) {
		snprintf(how, n, "proved by algebra");
		return 2;
	}
	r = pit_check(ST.f, ST.f->body, e, 4, seed, &bound, &t);
	if (r == 1) {
		snprintf(how, n, "probably: equal as polynomials except with "
		    "probability at most %.3g (identity testing), SAT gave up",
		    bound);
		free(t.p);
		return 1;
	}
	free(t.p);
	return 0;
}

static void
cmd_stoke(struct func *f, long steps, int restarts, int len, uint64_t seed,
    const struct model *m)
{
	struct sprog cur, best, prop, start;
	struct sfound found[ST_MAXFOUND];
	nid tried[512];
	int nfound = 0, ntried = 0, r, i, j, dist, d2, have_start = 0;
	int64_t lo[MAXP], hi[MAXP], in[MAXP];
	double ccur, cprop, cf, beta = 1.0;
	long s;
	struct sopnd o;

	if (!TY_SIZED(N.ty[f->body]) || N.ty[f->body] == TY_BOOL || f->np < 1)
		fail("stoke: an integer function on machine types (uN, iN)");
	for (j = 0; j < f->np; j++)
		if (!TY_SIZED(f->p[j].ty))
			fail("stoke: parameter %s is %s; machine integers only",
			    sym_str(f->p[j].name), ty_str(f->p[j].ty));
	memset(&ST, 0, sizeof ST);
	ST.f = f;
	ST.m = m;
	ST.W = TYW(N.ty[f->body]);
	ST.len = len;
	ST.np = f->np;
	ST.mask = wmask64(ST.W);
	sc_free();
	sp_setup(f, m);
	rseed(seed);
	/* the tests: the ends of the ranges, then random points */
	for (j = 0; j < f->np; j++)
		param_bounds(&f->p[j], &lo[j], &hi[j]);
	for (i = 0; i < 4; i++) {
		for (j = 0; j < f->np; j++)
			in[j] = i == 0 ? lo[j] : i == 1 ? hi[j] : i == 2 ?
			    (lo[j] < hi[j] ? lo[j] + 1 : lo[j]) : (hi[j] > lo[j] ?
			    hi[j] - 1 : hi[j]);
		st_add_test(in);
	}
	for (i = 0; i < 400 && ST.nt < 40; i++) {
		for (j = 0; j < f->np; j++) {
			uint64_t span = (uint64_t)(hi[j] - lo[j]) + 1;
			in[j] = lo[j] + (int64_t)(span ? rnd() % span : rnd());
		}
		st_add_test(in);
	}
	if (ST.nt == 0) {
		sp_release();
		fail("stoke: %s is defined nowhere on the points tried", fnm(f));
	}
	cf = dag_cost(m, f->body);
	/* the function itself as a program: where the optimisation starts */
	memset(&start, 0, sizeof start);
	lower_at = 0;
	lower_p = &start;
	if (sp_lower(f->body, &o) && o.kind == 1 && lower_at > 0)
		have_start = 1;
	{
		struct sbuf set = {0};
		sp_set_text(&set);
		printf("stoke %s under %s: Metropolis-Hastings over programs of up "
		    "to %d instructions (%s),\n  %d chain%s of %ld proposals, seed "
		    "%llu; %s as written costs %g\n", fnm(f), sym_str(m->name), len,
		    set.p, restarts, restarts == 1 ? "" : "s", steps,
		    (unsigned long long)seed, fnm(f), cf);
		free(set.p);
	}
	rep_begin("stoke");
	rep_str("model", sym_str(m->name));
	rep_int("seed", (long long)seed);
	rep_int("steps", steps);
	rep_int("restarts", restarts);
	rep_num("cost_as_written", cf);
	for (r = 0; r < restarts; r++) {
		if (r == 0 && have_start)
			cur = start;
		else {
			memset(&cur, 0, sizeof cur);
			for (i = 0; i < len; i++)
				if (rnd() & 1)
					rand_ins(&cur.in[i], i);
		}
		ccur = sp_cost(&cur, &dist);
		best = cur;
		for (s = 0; s < steps; s++) {
			prop = cur;
			sp_propose(&prop);
			ST.proposals++;
			cprop = sp_cost(&prop, &d2);
			if (cprop > ccur && runif() >= exp_neg(beta * (cprop - ccur)))
				continue;
			ST.accepted++;
			cur = prop;
			ccur = cprop;
			dist = d2;
			if (dist != 0)
				continue;
			ST.zeros++;
			{
				/* a candidate: worth verifying when it is new and
				 * cheaper than what is already proved */
				nid e = sp_node(&cur);
				double c = dag_cost(m, e);
				char how[160];
				int v, k, ops = dag_ops(e);
#define WORSE(f) ((f).cost < c || ((f).cost == c && (f).ops <= ops))
				for (k = 0; k < ntried; k++)
					if (tried[k] == e)
						break;
				if (k < ntried || (nfound == ST_MAXFOUND &&
				    WORSE(found[nfound - 1])))
					continue;
				if (ntried < 512)
					tried[ntried++] = e;
				if (nfound >= 3 && WORSE(found[0]))
					continue;
				v = st_verify(e, how, sizeof how, seed);
				if (!v) {
					/* a new test: the chain's cost has changed */
					ccur = sp_cost(&cur, &dist);
					continue;
				}
				ST.verified++;
				for (k = nfound < ST_MAXFOUND ? nfound : ST_MAXFOUND - 1;
				    k > 0 && !WORSE(found[k - 1]); k--)
					found[k] = found[k - 1];
#undef WORSE
				found[k].e = e;
				found[k].cost = c;
				found[k].ops = ops;
				found[k].proved = v;
				snprintf(found[k].how, sizeof found[k].how, "%s", how);
				if (nfound < ST_MAXFOUND)
					nfound++;
			}
		}
		(void)best;
	}
	st_add_test(NULL);
	sc_free();
	sp_release();
	printf("  %llu proposals, %llu accepted, %llu at distance 0 on the "
	    "%d tests (%llu counterexamples joined them), %llu verified\n",
	    (unsigned long long)ST.proposals, (unsigned long long)ST.accepted,
	    (unsigned long long)ST.zeros, ST.nt, (unsigned long long)ST.cex,
	    (unsigned long long)ST.verified);
	rep_int("proposals", (long long)ST.proposals);
	rep_int("verified", (long long)ST.verified);
	rep_list("programs");
	j = 0;
	for (i = 0; i < nfound && j < 3; i++) {
		if (found[i].e == f->body || found[i].cost >= cf)
			continue;
		printf("  cost %-6g %s\n      %s\n", found[i].cost,
		    shared_str(found[i].e, 300), found[i].how);
		rep_obj(NULL);
		rep_expr("expression", found[i].e);
		rep_num("cost", found[i].cost);
		rep_str("verified", found[i].how);
		rep_bool("proved", found[i].proved == 2);
		rep_end();
		j++;
	}
	rep_end();
	if (j)
		printf("  -> %g against %g (%+.1f%%)\n", found[0].cost, cf,
		    100 * (found[0].cost - cf) / cf);
	else
		printf("  -> nothing cheaper than %g was found and verified (the "
		    "search is stochastic: more `steps`, another `seed`)\n", cf);
	(void)so_name;
}

void
stoke_find(struct func *f)
{
	cmd_stoke(f, 200000, 4, 8, stoch_seed(), model_active());
}

/* ---- commands ---- */

int
stoch_command(struct P *p)
{
	static const char *const stop[] = {"steps", "restarts", "len", "seed",
		"model", "trials", NULL};
	struct model *m = model_active();
	char w[48], cmd[48];
	uint32_t at = cmdp_pos(p);
	uint64_t seed = stoch_seed();
	int64_t v;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if ((strcmp(w, "stoke") != 0 && strcmp(w, "pit") != 0 &&
	    strcmp(w, "phash") != 0) || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	snprintf(cmd, sizeof cmd, "%s", w);
	if (!strcmp(cmd, "pit")) {
		struct func *f = cmdp_func(p, stop), *g;
		uint32_t fn = f->name;
		int trials = 4;
		(void)cmdp_tok(p, T_COMMA);
		g = cmdp_func(p, stop);
		f = func_find(fn);
		while (!cmdp_eos(p)) {
			if (cmdp_kw(p, "trials")) {
				if (!cmdp_int(p, &v) || v < 1 || v > 64)
					fail("pit: trials N, 1 to 64");
				trials = (int)v;
			} else if (cmdp_kw(p, "seed")) {
				if (!cmdp_int(p, &v))
					fail("pit: seed K");
				seed = (uint64_t)v;
			} else {
				cmdp_fail_here(p);
				fail("pit F G [trials N] [seed K]");
			}
		}
		notes_flush();
		cmd_pit(f, g, trials, seed);
		return 1;
	}
	{
		struct func *f = cmdp_func(p, stop);
		long steps = 200000;
		int restarts = 4, len = 8;

		while (!cmdp_eos(p)) {
			if (cmdp_kw(p, "seed")) {
				if (!cmdp_int(p, &v))
					fail("%s: seed K", cmd);
				seed = (uint64_t)v;
			} else if (cmdp_kw(p, "model")) {
				if (!cmdp_word(p, w, sizeof w) ||
				    (m = model_find(sym_cstr(w))) == NULL)
					fail("%s: unknown model (see 'models')", cmd);
			} else if (!strcmp(cmd, "stoke") && cmdp_kw(p, "steps")) {
				if (!cmdp_int(p, &v) || v < 1 || v > 100000000)
					fail("stoke: steps N, 1 to 100000000");
				steps = (long)v;
			} else if (!strcmp(cmd, "stoke") && cmdp_kw(p, "restarts")) {
				if (!cmdp_int(p, &v) || v < 1 || v > 1000)
					fail("stoke: restarts R, 1 to 1000");
				restarts = (int)v;
			} else if (!strcmp(cmd, "stoke") && cmdp_kw(p, "len")) {
				if (!cmdp_int(p, &v) || v < 1 || v > ST_MAXLEN)
					fail("stoke: len L, 1 to %d", ST_MAXLEN);
				len = (int)v;
			} else {
				cmdp_fail_here(p);
				fail(!strcmp(cmd, "stoke") ? "stoke F [steps N] [restarts "
				    "R] [len L] [seed K] [model M]" : "phash F [seed K] "
				    "[model M]");
			}
		}
		notes_flush();
		if (!strcmp(cmd, "stoke"))
			cmd_stoke(f, steps, restarts, len, seed, m);
		else
			cmd_phash(f, seed, m);
	}
	return 1;
}

/* ---- English ---- */

/*
 * superoptimize F, stochastic search for F, mcmc search for F, stoke
 * search F; are F and G the same polynomial, polynomial identity test F
 * and G, schwartz-zippel F and G; hash the switch F, perfect hash for F,
 * turn the switch F into a lookup; each with [under | on MODEL] [with
 * seed K].
 */
const char *
stoch_nlp_raw(const char *sent)
{
	static char out[1200];
	char buf[1000], tail[120] = "", *s, *q, *e;
	size_t n;
	const char *cmd = NULL;
	static const struct { const char *pre, *cmd; } pre[] = {
		{"superoptimize ", "stoke"}, {"superoptimise ", "stoke"},
		{"stochastic search for ", "stoke"}, {"mcmc search for ", "stoke"},
		{"stochastically search ", "stoke"},
		{"polynomial identity test ", "pit"},
		{"polynomial identity test of ", "pit"},
		{"schwartz-zippel ", "pit"}, {"schwartz zippel ", "pit"},
		{"hash the switch ", "phash"}, {"perfect hash for ", "phash"},
		{"a perfect hash for ", "phash"}, {"hash the keys of ", "phash"},
		{NULL, NULL}};
	int i;

	while (*sent == ' ' || *sent == '\t')
		sent++;
	if (strlen(sent) >= sizeof buf)
		return NULL;
	strcpy(buf, sent);
	n = strlen(buf);
	while (n && (buf[n - 1] == ' ' || buf[n - 1] == '?' || buf[n - 1] == '.'
	    || buf[n - 1] == '\n'))
		buf[--n] = 0;
	s = buf;
	if (!strncasecmp(s, "please ", 7))
		s += 7;
	if ((q = strstr(s, " with seed ")) != NULL && isdigit((unsigned char)
	    q[11])) {
		snprintf(tail, sizeof tail, " seed %ld", atol(q + 11));
		*q = 0;
	}
	q = NULL;
	for (e = s; (e = strstr(e, " under ")) != NULL; e++)
		q = e;
	if (q == NULL)
		for (e = s; (e = strstr(e, " on ")) != NULL; e++)
			q = e;
	if (q != NULL) {
		const char *w = strrchr(q, ' ') + 1;
		size_t wn = strlen(w);
		if (wn && wn < 40 && strchr(q + 1, ' ') == strrchr(q, ' ') &&
		    model_find(sym_intern(w, wn)) != NULL) {
			snprintf(tail + strlen(tail), sizeof tail - strlen(tail),
			    " model %s", w);
			*q = 0;
		}
	}
	for (i = 0; pre[i].pre && !cmd; i++)
		if (!strncasecmp(s, pre[i].pre, strlen(pre[i].pre))) {
			s += strlen(pre[i].pre);
			cmd = pre[i].cmd;
		}
	if (!cmd && !strncasecmp(s, "are ", 4)) {
		static const char *const post[] = {" the same polynomial",
			" the same polynomials", " identical polynomials",
			" equal as polynomials", NULL};
		n = strlen(s);
		for (i = 0; post[i]; i++) {
			size_t k = strlen(post[i]);
			if (n > k + 4 && !strcasecmp(s + n - k, post[i])) {
				s[n - k] = 0;
				s += 4;
				cmd = "pit";
				break;
			}
		}
	}
	if (!cmd && !strncasecmp(s, "turn the switch ", 16)) {
		n = strlen(s);
		if (n > 31 && !strcasecmp(s + n - 14, " into a lookup")) {
			s[n - 14] = 0;
			s += 16;
			cmd = "phash";
		}
	}
	if (!cmd || *s == 0 || !strcmp(s, "it"))
		return NULL;
	if (!strcmp(cmd, "pit")) {
		char *a = strstr(s, " and ");
		if (a == NULL)
			return NULL;
		if (strstr(tail, "model"))
			return NULL;	/* pit prices nothing */
		memmove(a + 1, a + 5, strlen(a + 5) + 1);
	}
	snprintf(out, sizeof out, "%s %s%s", cmd, s, tail);
	return out;
}
