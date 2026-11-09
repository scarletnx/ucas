/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * find: other ways to compute the same function, cheapest first, lazily.
 *
 * Integer targets.  Terms are built bottom-up by cost (the active cost
 * model prices the operations).  Each term is identified with its column of
 * values on the domain points, so two terms with the same column are one
 * term (observational equivalence); the bank is struct-of-arrays and the
 * columns live in one flat array.  Every new column is also tried as the
 * core of a template with a solved constant: a*t + b, t xor K, t & K,
 * t | K.  Closed forms are tried up front: a packed constant table
 * (K >> k*x) & mask, a low-degree polynomial, and the plain lookup table.
 * `more` resumes the enumeration where it stopped.
 *
 * Float targets.  Bit-level approximation families asfloat(K +/- (asuint(x)
 * >> s)) and friends get their magic constant fitted (see `fit`).
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FPTS 1024
#define MAXOPS 10

enum { M_EXACT, M_AFFINE, M_XOR, M_AND, M_OR, M_FORM };

struct match {
	int kind;
	int cost;
	nid e;
	uint64_t sig;
	int shown;
	const char *why;
};

static struct {
	int active;
	struct func f;
	ty_t W;
	struct ity it;
	uint32_t P;
	int64_t *y;
	/* term bank */
	uint8_t *op;
	uint32_t *a, *b;
	uint8_t *cost;
	nid *node;
	int64_t *col;
	uint32_t n, cap, maxn;
	uint32_t *ht;
	uint32_t htcap;
	uint32_t lvl[64];
	int done;           /* last finished cost level */
	int nconst;         /* leaves that are constants start here */
	uint32_t firstk;
	/* operations */
	int nops;
	int ops[MAXOPS];
	int opc[MAXOPS];
	/* results */
	struct match *m;
	int nm, capm;
	int sampled;
	int full;
} F;

static void
finder_free(void)
{
	free(F.y);
	free(F.op);
	free(F.a);
	free(F.b);
	free(F.cost);
	free(F.node);
	free(F.col);
	free(F.ht);
	free(F.m);
	memset(&F, 0, sizeof F);
}

static uint32_t
colhash(const int64_t *c)
{
	uint64_t h = 1469598103934665603ULL;
	uint32_t i;

	for (i = 0; i < F.P; i++)
		h = (h ^ (uint64_t)c[i]) * 1099511628211ULL;
	return (uint32_t)(h ^ (h >> 32));
}

static int64_t *
COL(uint32_t t)
{
	return &F.col[(size_t)t * F.P];
}

/* returns the existing term with this column, or NOTOK */
static uint32_t
lookup(const int64_t *c, uint32_t *slot)
{
	uint32_t j = colhash(c) & (F.htcap - 1), t;

	while ((t = F.ht[j]) != 0) {
		if (memcmp(COL(t - 1), c, F.P * sizeof *c) == 0) {
			*slot = j;
			return t - 1;
		}
		j = (j + 1) & (F.htcap - 1);
	}
	*slot = j;
	return NOTOK;
}

static int
insert(int op, uint32_t a, uint32_t b, int cost, const int64_t *c,
    uint32_t *out)
{
	uint32_t slot, t;

	if (lookup(c, &slot) != NOTOK)
		return 0;
	if (F.n >= F.maxn) {
		F.full = 1;
		return 0;
	}
	if (F.n == F.cap) {
		F.cap = newcap(F.cap, F.n + 1);
		F.op = xrealloc(F.op, F.cap);
		F.a = xrealloc(F.a, F.cap * sizeof *F.a);
		F.b = xrealloc(F.b, F.cap * sizeof *F.b);
		F.cost = xrealloc(F.cost, F.cap);
		F.node = xrealloc(F.node, F.cap * sizeof *F.node);
		F.col = xrealloc(F.col, (size_t)F.cap * F.P * sizeof *F.col);
	}
	t = F.n++;
	F.op[t] = (uint8_t)op;
	F.a[t] = a;
	F.b[t] = b;
	F.cost[t] = (uint8_t)cost;
	F.node[t] = 0;
	memcpy(COL(t), c, F.P * sizeof *c);
	F.ht[slot] = t + 1;
	*out = t;
	return 1;
}

static nid
term_node(uint32_t t)
{
	nid r;

	if (F.node[t])
		return F.node[t];
	if (opinfo[F.op[t]].arity == 1)
		r = mk_un(F.op[t], term_node(F.a[t]));
	else
		r = mk_bin(F.op[t], term_node(F.a[t]), term_node(F.b[t]));
	F.node[t] = r;
	return r;
}

static int
mcost(int cls)
{
	float c = model_active()->c[cls];
	int r = (int)(c + 0.5f);

	return r < 1 ? 1 : r > 60 ? 60 : r;
}

static int
op_cost(int op)
{
	switch (op) {
	case O_ADD: case O_SUB: case O_NEG: return mcost(C_IADD);
	case O_MUL: return mcost(C_IMUL);
	case O_SHL: case O_SHR: return mcost(C_SHIFT);
	default: return mcost(C_LOGIC);
	}
}

/*
 * Two matches are the same approach when they use the same multiset of
 * operations (x + x + 3 and x + (x + 3)); only the cheapest is kept.
 */
static uint64_t
signature(nid e)
{
	static const int slot[O_NOPS] = {
		[O_ADD] = 1, [O_SUB] = 2, [O_MUL] = 3, [O_AND] = 4, [O_OR] = 5,
		[O_XOR] = 6, [O_SHL] = 7, [O_SHR] = 8, [O_NEG] = 9, [O_NOT] = 10,
		[O_INDEX] = 11, [O_POW] = 12, [O_DIV] = 13, [O_SEL] = 14,
	};
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), n;
	uint32_t sp = 0;
	uint64_t sig = 0, cnt;
	int s;

	st[sp++] = e;
	while (sp) {
		n = st[--sp];
		if (mark[n])
			continue;
		mark[n] = 1;
		s = slot[N.op[n]];
		if (s) {
			cnt = (sig >> (4 * s)) & 15;
			if (cnt < 15)
				sig += 1ULL << (4 * s);
		}
		if (N.op[n] == O_CONST || N.op[n] == O_VAR ||
		    N.op[n] == O_TABLE || N.op[n] == O_UNDEF)
			continue;
		if (N.a[n])
			st[sp++] = N.a[n];
		if (N.b[n])
			st[sp++] = N.b[n];
		if (N.c[n])
			st[sp++] = N.c[n];
	}
	free(mark);
	free(st);
	return sig;
}

static void
add_match(int kind, int cost, nid e, const char *why)
{
	int i;
	uint64_t sig = signature(e);

	for (i = 0; i < F.nm; i++)
		if (F.m[i].e == e || (F.m[i].sig == sig && kind != M_FORM &&
		    F.m[i].kind != M_FORM)) {
			if (cost < F.m[i].cost && !F.m[i].shown) {
				F.m[i].cost = cost;
				F.m[i].e = e;
				F.m[i].why = why;
			}
			return;
		}
	if (F.nm == F.capm) {
		F.capm = F.capm ? F.capm * 2 : 64;
		F.m = xrealloc(F.m, (size_t)F.capm * sizeof *F.m);
	}
	F.m[F.nm].kind = kind;
	F.m[F.nm].cost = cost;
	F.m[F.nm].e = e;
	F.m[F.nm].sig = sig;
	F.m[F.nm].shown = 0;
	F.m[F.nm].why = why;
	F.nm++;
}

static int64_t
wrapmul(int64_t a, int64_t b)
{
	return int_norm(F.W, (int64_t)((uint64_t)a * (uint64_t)b));
}

static int
ispow2(int64_t v)
{
	return v > 0 && !(v & (v - 1));
}

static nid
kconst(int64_t v)
{
	return mk_const_i(F.W, v);
}

static int
is_kleaf(uint32_t t)
{
	return t >= F.firstk && t < F.lvl[1];
}

/* t is (u op K) for a constant leaf K */
static int
has_const_operand(uint32_t t, int op)
{
	if (F.op[t] != op || opinfo[op].arity != 2 || t < F.lvl[1])
		return 0;
	return is_kleaf(F.a[t]) || is_kleaf(F.b[t]);
}

/* t + b on an unsigned type reads better as t - (2^w - b) */
static nid
add_k(nid e, int64_t b)
{
	if (F.it.k == K_UINT && (uint64_t)b > F.it.mask / 2)
		return mk_bin(O_SUB, e, kconst(int_norm(F.W, -b)));
	if (F.it.k != K_UINT && b < 0)
		return mk_bin(O_SUB, e, kconst(-b));
	return mk_bin(O_ADD, e, kconst(b));
}

/* templates with one solved constant around a term */
static void
holes(uint32_t t)
{
	const int64_t *c = COL(t), *y = F.y;
	uint32_t i, p = NOTOK;
	int64_t a, b, k;
	int base = F.cost[t], cost;
	uint64_t acc;

	for (i = 1; i < F.P; i++)
		if (c[i] != c[0]) {
			p = i;
			break;
		}
	if (p == NOTOK)
		return;
	/* a constant at the top of t would only be absorbed by the hole */
	if (t >= F.lvl[1] && opinfo[F.op[t]].arity == 2 &&
	    (is_kleaf(F.a[t]) || is_kleaf(F.b[t])) && F.op[t] != O_SHL &&
	    F.op[t] != O_SHR && F.op[t] != O_MUL)
		return;
	/* y = a*t + b */
	if ((y[p] - y[0]) % (c[p] - c[0]) == 0) {
		a = (y[p] - y[0]) / (c[p] - c[0]);
		b = int_norm(F.W, y[0] - (int64_t)((uint64_t)a * (uint64_t)c[0]));
		for (i = 0; i < F.P; i++)
			if (int_norm(F.W, (int64_t)((uint64_t)wrapmul(a, c[i]) +
			    (uint64_t)b)) != y[i])
				break;
		if (i == F.P && a != 0 && !(a == 1 && b == 0)) {
			nid e = term_node(t);
			cost = base;
			if (a != 1 && a != -1)
				cost += ispow2(a < 0 ? -a : a) ?
				    op_cost(O_SHL) : op_cost(O_MUL);
			if (b != 0 || a < 0)
				cost += op_cost(O_ADD);
			if (a == -1)
				e = mk_bin(O_SUB, kconst(b), e);
			else {
				e = mk_bin(O_MUL, kconst(a), e);
				if (b)
					e = add_k(e, b);
			}
			add_match(M_AFFINE, cost, e, "affine fit");
		}
	}
	/* y = t xor K */
	k = c[0] ^ y[0];
	for (i = 0; i < F.P && (c[i] ^ y[i]) == k; i++)
		;
	if (i == F.P && k && !has_const_operand(t, O_XOR))
		add_match(M_XOR, base + op_cost(O_XOR),
		    mk_bin(O_XOR, term_node(t), kconst(k)), "xor constant");
	/* y = t & K */
	acc = 0;
	for (i = 0; i < F.P; i++)
		acc |= (uint64_t)c[i] & ~(uint64_t)y[i];
	k = int_norm(F.W, (int64_t)~acc);
	for (i = 0; i < F.P && (c[i] & k) == y[i]; i++)
		;
	if (i == F.P && k != int_norm(F.W, -1) && !has_const_operand(t, O_AND))
		add_match(M_AND, base + op_cost(O_AND),
		    mk_bin(O_AND, term_node(t), kconst(k)), "mask");
	/* y = t | K */
	acc = ~0ULL;
	for (i = 0; i < F.P; i++)
		acc &= (uint64_t)y[i];
	k = int_norm(F.W, (int64_t)acc);
	for (i = 0; i < F.P && (c[i] | k) == y[i]; i++)
		;
	if (i == F.P && k && !has_const_operand(t, O_OR))
		add_match(M_OR, base + op_cost(O_OR),
		    mk_bin(O_OR, term_node(t), kconst(k)), "or constant");
}

static int
is_target(const int64_t *c)
{
	return memcmp(c, F.y, F.P * sizeof *c) == 0;
}

static void
build(int op, uint32_t a, uint32_t b, int cost, int64_t *scratch)
{
	const int64_t *ca = COL(a), *cb = opinfo[op].arity > 1 ? COL(b) : ca;
	uint32_t i, t;
	uint8_t fl = 0;
	int64_t first;

	for (i = 0; i < F.P; i++) {
		scratch[i] = scalar_int(op, &F.it, &F.it, ca[i], cb[i], 0, &fl);
		if (fl & (F_UNDEF | F_UB))
			return;
	}
	first = scratch[0];
	for (i = 1; i < F.P && scratch[i] == first; i++)
		;
	if (i == F.P)
		return;   /* constants come from the leaves and the holes */
	if (is_target(scratch)) {
		nid e = opinfo[op].arity == 1 ? mk_un(op, term_node(a)) :
		    mk_bin(op, term_node(a), term_node(b));
		add_match(M_EXACT, cost, e, "exact");
	}
	if (insert(op, a, b, cost, scratch, &t))
		holes(t);
}

static void
enumerate_level(int c)
{
	int64_t *scratch = xmalloc(F.P * sizeof *scratch);
	int k, ca;
	uint32_t i, j;

	F.lvl[c] = F.n;
	for (k = 0; k < F.nops && !F.full; k++) {
		int op = F.ops[k], oc = F.opc[k];
		if (oc > c)
			continue;
		if (opinfo[op].arity == 1) {
			for (i = F.lvl[c - oc]; i < F.lvl[c - oc + 1] && !F.full; i++)
				if (i < F.firstk || c - oc > 0)
					build(op, i, 0, c, scratch);
			continue;
		}
		for (ca = 0; ca <= c - oc; ca++) {
			int cb = c - oc - ca;
			int comm = (opinfo[op].fl & OF_COMM) != 0;
			if (comm && ca > cb)
				continue;
			for (i = F.lvl[ca]; i < F.lvl[ca + 1] && !F.full; i++) {
				for (j = F.lvl[cb]; j < F.lvl[cb + 1] && !F.full; j++) {
					if (comm && ca == cb && j < i)
						continue;
					/* two constant leaves fold to a constant */
					if (i >= F.firstk && i < F.lvl[1] &&
					    j >= F.firstk && j < F.lvl[1])
						continue;
					/* shift amounts are constants or parameters */
					if ((op == O_SHL || op == O_SHR) && cb > 0)
						continue;
					/*
					 * Constants go to the top of a chain: no sum
					 * whose operand already adds a constant, the
					 * same for xor, and, or, mul.  The solved holes put
					 * the final constant back.
					 */
					if ((op == O_ADD || op == O_SUB) &&
					    (has_const_operand(i, O_ADD) ||
					    has_const_operand(i, O_SUB) ||
					    has_const_operand(j, O_ADD) ||
					    has_const_operand(j, O_SUB)))
						continue;
					if ((op == O_XOR || op == O_AND || op == O_OR ||
					    op == O_MUL) && (has_const_operand(i, op) ||
					    has_const_operand(j, op)))
						continue;
					/* identities: x & ~0, x | 0, x * 1 */
					if ((op == O_AND || op == O_OR || op == O_XOR) &&
					    ((is_kleaf(i) && (COL(i)[0] == 0 ||
					    (op == O_AND && COL(i)[0] == int_norm(F.W, -1)))) ||
					    (is_kleaf(j) && (COL(j)[0] == 0 ||
					    (op == O_AND && COL(j)[0] == int_norm(F.W, -1))))))
						continue;
					if (op == O_SUB && F.op[j] == O_NEG)
						continue;
					build(op, i, j, c, scratch);
				}
			}
		}
	}
	F.lvl[c + 1] = F.n;
	F.done = c;
	free(scratch);
}

static int
cmp_match(const void *x, const void *y)
{
	const struct match *a = x, *b = y;

	if (a->cost != b->cost)
		return a->cost - b->cost;
	return (int)a->e - (int)b->e;
}

static int
show_new(int want)
{
	int i, shown = 0;
	struct sbuf w = {0};

	qsort(F.m, (size_t)F.nm, sizeof *F.m, cmp_match);
	for (i = 0; i < F.nm && shown < want; i++) {
		int sampled, bad;
		struct eqproof pr;
		if (F.m[i].shown)
			continue;
		F.m[i].shown = 1;
		sb_clear(&w);
		bad = equiv_decide(&F.f, F.f.body, F.m[i].e, &w, &pr);
		if (bad) {
			F.m[i].shown = 2;
			continue;
		}
		sampled = pr.how == EQ_SAMPLED;
		printf("  cost %-3d %-40s %s%s\n", F.m[i].cost,
		    shared_str(F.m[i].e, 300),
		    F.m[i].why, sampled ? " (sampled)" : pr.how == EQ_SAT ?
		    " (proved by SAT)" : "");
		rep_obj(NULL);
		rep_num("cost", F.m[i].cost);
		rep_expr("expr", F.m[i].e);
		rep_str("why", F.m[i].why);
		rep_bool("verified", 1);
		rep_bool("exhaustive", pr.how == EQ_EXHAUSTIVE);
		rep_str("proof", pr.how == EQ_EXHAUSTIVE ? "exhaustive" :
		    pr.how == EQ_SAT ? "sat" : "sampled");
		rep_end();
		shown++;
	}
	free(w.p);
	return shown;
}

static void
run_levels(int want, int maxcost)
{
	int got = 0, c;

	for (c = F.done + 1; c <= maxcost && c < 62 && !F.full; c++) {
		enumerate_level(c);
		{
			int i, ready = 0;
			for (i = 0; i < F.nm; i++)
				if (!F.m[i].shown && F.m[i].cost <= c)
					ready++;
			if (ready >= want - got)
				break;
		}
	}
	got = show_new(want);
	if (got == 0)
		printf("  nothing new up to cost %d%s\n", F.done,
		    F.full ? " (term bank full)" : "");
	printf("  [searched every term up to cost %d: %u distinct columns%s; "
	    "'more' continues]\n", F.done, F.n, F.full ? ", bank full" : "");
}

/* closed forms tried before enumeration */
static void
forms(const int64_t *xs)
{
	uint32_t i, D = F.P;
	int64_t lo, maxy = 0, miny = 0;
	int k, contiguous = 1;
	const struct param *pp = &F.f.p[0];
	nid x;

	if (F.f.np != 1 || F.sampled)
		return;
	lo = xs[0];
	for (i = 0; i < D; i++) {
		if (xs[i] != lo + (int64_t)i)
			contiguous = 0;
		if (!i || F.y[i] > maxy)
			maxy = F.y[i];
		if (!i || F.y[i] < miny)
			miny = F.y[i];
	}
	if (!contiguous)
		return;
	x = mk_cast(F.W, mk_var(0, pp->name, pp->ty));
	/* lookup table: the baseline */
	{
		nid *e = xmalloc(D * sizeof *e), idx;
		for (i = 0; i < D; i++)
			e[i] = kconst(F.y[i]);
		idx = lo ? mk_bin(O_SUB, x, kconst(lo)) : x;
		if (D <= 256)
			add_match(M_FORM, mcost(C_LOAD) + (lo ? op_cost(O_SUB) : 0),
			    mk_index(mk_table(e, D), idx), "lookup table (baseline)");
		free(e);
	}
	/*
	 * packed constant: off + ((K >> x*k) & mask), entries stored minus
	 * their minimum; the index is not rebased when the slots below lo
	 * still fit in the constant.
	 */
	if (miny >= 0 || F.it.k != K_UINT) {
		int64_t off = miny > 0 || F.it.k != K_UINT ? miny : 0;
		int bits = bits_for((uint64_t)(maxy - off));
		int rebase = lo != 0 && (uint64_t)bits * (uint64_t)(lo + D) > 64;
		uint32_t slots = rebase ? D : (uint32_t)(lo + D);
		if (lo >= 0 && (uint32_t)bits * slots <= 64 && D > 1) {
			ty_t T = (uint32_t)bits * slots <= 32 ? TY(K_UINT, 32) :
			    TY(K_UINT, 64);
			uint64_t K = 0;
			nid xi, sh, e;
			int cost = op_cost(O_SHR) + op_cost(O_AND) +
			    (off ? op_cost(O_ADD) : 0);
			for (i = 0; i < D; i++)
				K |= (uint64_t)(F.y[i] - off) << ((i + (rebase ? 0 :
				    (uint32_t)lo)) * (uint32_t)bits);
			xi = mk_cast(T, mk_var(0, pp->name, pp->ty));
			if (rebase) {
				xi = mk_bin(O_SUB, xi, mk_const_i(T, lo));
				cost += op_cost(O_SUB);
			}
			if (bits == 1)
				sh = xi;
			else if (ispow2(bits)) {
				sh = mk_bin(O_SHL, xi, mk_const_i(T, ilog2u((uint64_t)bits)));
				cost += op_cost(O_SHL);
			} else {
				sh = mk_bin(O_MUL, mk_const_i(T, bits), xi);
				cost += op_cost(O_MUL);
			}
			e = mk_bin(O_SHR, mk_const_i(T, (int64_t)K), sh);
			e = mk_bin(O_AND, e, mk_const_i(T,
			    (int64_t)((1ULL << bits) - 1)));
			e = mk_cast(F.W, e);
			if (off)
				e = add_k(e, off);
			add_match(M_FORM, cost, e, "packed table in one constant");
		}
	}
	/* polynomial of degree 2 or 3 with integer coefficients */
	for (k = 2; k <= 3 && D > (uint32_t)k + 1; k++) {
		double A[4][5], c[4];
		int r, s, q, ok = 1;
		for (r = 0; r <= k; r++) {
			double xv = (double)xs[r];
			for (s = 0; s <= k; s++)
				A[r][s] = pow(xv, s);
			A[r][k + 1] = (double)F.y[r];
		}
		for (r = 0; r <= k; r++) {
			int piv = r;
			for (q = r + 1; q <= k; q++)
				if (fabs(A[q][r]) > fabs(A[piv][r]))
					piv = q;
			for (s = 0; s <= k + 1; s++) {
				double tmp = A[r][s];
				A[r][s] = A[piv][s];
				A[piv][s] = tmp;
			}
			if (A[r][r] == 0) {
				ok = 0;
				break;
			}
			for (q = 0; q <= k; q++) {
				double fct;
				if (q == r)
					continue;
				fct = A[q][r] / A[r][r];
				for (s = r; s <= k + 1; s++)
					A[q][s] -= fct * A[r][s];
			}
		}
		if (!ok)
			continue;
		for (r = 0; r <= k; r++) {
			c[r] = A[r][k + 1] / A[r][r];
			if (fabs(c[r] - floor(c[r] + 0.5)) > 1e-9)
				ok = 0;
			c[r] = floor(c[r] + 0.5);
		}
		if (!ok || c[k] == 0)
			continue;
		for (i = 0; i < D; i++) {
			int64_t v = 0;
			for (r = k; r >= 0; r--)
				v = (int64_t)((uint64_t)wrapmul(v, xs[i]) +
				    (uint64_t)(int64_t)c[r]);
			if (int_norm(F.W, v) != F.y[i])
				break;
		}
		if (i == D) {
			nid e = 0;
			for (r = k; r >= 0; r--) {
				nid term;
				if (c[r] == 0)
					continue;
				term = r == 0 ? kconst((int64_t)c[r]) :
				    mk_bin(O_MUL, kconst((int64_t)c[r]),
				    r == 1 ? x : mk_bin(O_POW, x, mk_const_i(TY_LIT, r)));
				e = e ? mk_bin(O_ADD, e, term) : term;
			}
			add_match(M_FORM, k * (op_cost(O_MUL) + op_cost(O_ADD)), e,
			    "polynomial (Horner)");
			break;
		}
	}
}

/* ---- float targets ---- */

static float
expr_cost(nid e)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), n;
	uint32_t sp = 0, j;
	float c = 0;
	int cls;

	st[sp++] = e;
	mark[e] = 1;
	while (sp) {
		n = st[--sp];
		c += node_cost(model_active(), n, &cls);
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF)
			continue;
		if (N.op[n] == O_TABLE) {
			for (j = 0; j < N.b[n]; j++)
				if (!mark[N.lst[N.a[n] + j]])
					mark[N.lst[N.a[n] + j]] = 1,
					st[sp++] = N.lst[N.a[n] + j];
			continue;
		}
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
	}
	free(mark);
	free(st);
	return c;
}

static void
find_float(struct func *f, int want)
{
	struct {
		const char *label;
		int sign, shift, div;
		double expo;
	} fam[] = {
		{"x^-1/2", -1, 1, 1, -0.5}, {"x^1/2", 1, 1, 1, 0.5},
		{"x^-1", -1, 0, 1, -1}, {"x^1/3", 1, 0, 3, 1.0 / 3},
		{"x^-1/3", -1, 0, 3, -1.0 / 3}, {"x^1/4", 1, 2, 1, 0.25},
		{"x^-1/4", -1, 2, 1, -0.25}, {"x^2", 2, 0, 1, 2},
	};
	int nf = (int)(sizeof fam / sizeof fam[0]), i, j, order[16];
	double obj[16];
	nid body[16];
	ty_t xt, ut;
	struct func g;

	if (f->np != 1 || TYK(f->p[0].ty) != K_FLOAT)
		fail("find on a float result needs one f32/f64 parameter");
	xt = f->p[0].ty;
	ut = TY(K_UINT, TYW(xt));
	printf("find %s: IEEE bit-level families, magic constant K fitted "
	    "for max relative error\n", f->anon ? node_str(f->body) :
	    sym_str(f->name));
	for (i = 0; i < nf; i++) {
		nid xb = mk_un(O_ASUINT, mk_var(0, f->p[0].name, xt));
		nid K = mk_var(1, sym_cstr("K"), ut), t, e;
		if (fam[i].sign == 2)
			e = mk_bin(O_SUB, mk_bin(O_SHL, xb, mk_const_i(ut, 1)), K);
		else {
			t = fam[i].shift ? mk_bin(O_SHR, xb, mk_const_i(ut,
			    fam[i].shift)) : xb;
			if (fam[i].div > 1)
				t = mk_bin(O_DIV, t, mk_const_i(ut, fam[i].div));
			e = mk_bin(fam[i].sign < 0 ? O_SUB : O_ADD, K, t);
		}
		e = mk_un(O_ASFLOAT, e);
		memset(&g, 0, sizeof g);
		g.name = sym_cstr("_family");
		g.anon = 1;
		g.np = 2;
		g.p[0] = f->p[0];
		g.p[1].name = sym_cstr("K");
		g.p[1].ty = ut;
		g.body = e;
		fit_quiet = 1;
		cmd_fit(&g, f, 0);
		fit_quiet = 0;
		obj[i] = fit_last_obj;
		{
			nid map[2];
			map[0] = 0;
			map[1] = mk_const_i(ut, (int64_t)fit_last_k[0]);
			body[i] = subst(e, map, 2);
		}
		order[i] = i;
	}
	for (i = 0; i < nf; i++)
		for (j = i + 1; j < nf; j++)
			if (obj[order[j]] < obj[order[i]]) {
				int tmp = order[i];
				order[i] = order[j];
				order[j] = tmp;
			}
	for (i = 0; i < nf && i < want; i++) {
		int q = order[i];
		struct sbuf b = {0};
		if (isinf(obj[q]) || obj[q] > 0.5)
			break;
		print_node(&b, body[q], NULL);
		printf("  max rel %-10.4g cost %-5g %-44s  (%s family)\n", obj[q],
		    (double)expr_cost(body[q]), sb_str(&b), fam[q].label);
		free(b.p);
	}
	printf("  [exact %s costs %g under %s; refine a family with a Newton "
	    "step and check it with 'error']\n", node_str(f->body),
	    (double)expr_cost(f->body), sym_str(model_active()->name));
}

/* ---- peephole: operations of the function itself that never bite ---- */

#define PEEP_ROUNDS 16
#define PEEP_PROOFS 40      /* equivalence proofs per find */
#define PEEP_SECONDS 8.0    /* and CPU time for them */
#define PEEP_SHOWN 6        /* single drops listed */

struct pcand {
	nid e;
	int cost;
	char what[120];
};

static int
icost(nid e)
{
	float c = expr_cost(e);

	return (int)(c + 0.5f);
}

/* e refines f on the points find works with (a quick filter) */
static int
quick_refines(const struct func *f, nid e)
{
	struct diter d;
	struct plan p;
	nid roots[3];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t np, i;
	int ok = 1;

	roots[0] = f->body;
	roots[1] = mk_cast(N.ty[f->body], e);
	roots[2] = f->where;
	plan_build(&p, roots, 3, 0);
	diter_init(&d, f, FPTS, 2, FPTS);
	while (ok && (np = diter_next(&d, in_i, in_r)) != 0) {
		uint8_t v[CH];
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[2] : NOTOK, np, v);
		for (i = 0; i < np; i++) {
			if (!v[i] || (COLF(&p, p.root[0])[i] & F_UNDEF))
				continue;
			if ((COLF(&p, p.root[1])[i] & F_UNDEF) ||
			    COLI(&p, p.root[0])[i] != COLI(&p, p.root[1])[i]) {
				ok = 0;
				break;
			}
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return ok;
}

static void
short_str(char *out, size_t n, nid e)
{
	const char *s = shared_str(e, 0);
	size_t l = strlen(s);

	if (l + 1 <= n && l <= 40)
		snprintf(out, n, "%s", s);
	else
		snprintf(out, n, "%.36s...", s);
}

/* replace `from` by `to` in root; 0 when the rebuilt term is ill-typed */
static nid
try_replace(nid root, nid from, nid to)
{
	jmp_buf saved;
	nid r = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = xreplace(root, from, to);
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	return r;
}

static void
cand_add(struct pcand **c, int *nc, int *cap, nid root, nid e,
    const char *what)
{
	int i;

	if (e == 0 || e == root || !TY_ISINT(N.ty[e]))
		return;
	for (i = 0; i < *nc; i++)
		if ((*c)[i].e == e)
			return;
	if (*nc == *cap) {
		*cap = *cap ? *cap * 2 : 32;
		*c = xrealloc(*c, (size_t)*cap * sizeof **c);
	}
	(*c)[*nc].e = e;
	(*c)[*nc].cost = icost(e);
	snprintf((*c)[*nc].what, sizeof (*c)[*nc].what, "%s", what);
	(*nc)++;
}

static int
cmp_pcand(const void *x, const void *y)
{
	const struct pcand *a = x, *b = y;

	if (a->cost != b->cost)
		return a->cost - b->cost;
	return (int)a->e - (int)b->e;
}

/*
 * Every way to drop one operation of root: an and, or, min, max, ?: or
 * conversion replaced by one of its operands, an operation with a
 * constant or a shift replaced by its first operand, a unary operation
 * by its operand.  Ill-typed results are skipped.
 */
static int
peep_cands(nid root, struct pcand **out)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), n;
	uint32_t sp = 0, lim = N.n;
	struct pcand *c = NULL;
	int nc = 0, cap = 0;
	char what[120], k1[48], k2[48];

	st[sp++] = root;
	while (sp) {
		int op;
		n = st[--sp];
		if (n == 0 || n >= lim || mark[n])
			continue;
		mark[n] = 1;
		op = N.op[n];
		if (op == O_CONST || op == O_VAR || op == O_UNDEF ||
		    op == O_TABLE || op == O_LOOP || op == O_BARRIER ||
		    op == O_ATOMIC)
			continue;
		if (N.a[n])
			st[sp++] = N.a[n];
		if (N.b[n])
			st[sp++] = N.b[n];
		if (N.c[n])
			st[sp++] = N.c[n];
		if (!TY_ISINT(N.ty[n]) || op == O_INDEX)
			continue;
		switch (op) {
		case O_SEL:
			short_str(k1, sizeof k1, N.a[n]);
			snprintf(what, sizeof what, "the ?: on %s (it always "
			    "takes the first arm)", k1);
			cand_add(&c, &nc, &cap, root, try_replace(root, n, N.b[n]),
			    what);
			snprintf(what, sizeof what, "the ?: on %s (it always "
			    "takes the second arm)", k1);
			cand_add(&c, &nc, &cap, root, try_replace(root, n, N.c[n]),
			    what);
			break;
		case O_CAST:
			if (!TY_ISINT(N.ty[N.a[n]]))
				break;
			snprintf(what, sizeof what, "the conversion %s(...)",
			    ty_str(N.ty[n]));
			cand_add(&c, &nc, &cap, root, try_replace(root, n, N.a[n]),
			    what);
			break;
		default:
			if (opinfo[op].arity == 1) {
				snprintf(what, sizeof what, "%s(...)",
				    opinfo[op].name);
				cand_add(&c, &nc, &cap, root,
				    try_replace(root, n, N.a[n]), what);
				break;
			}
			if (opinfo[op].arity != 2 || (opinfo[op].fl & OF_CMP))
				break;
			short_str(k2, sizeof k2, N.b[n]);
			snprintf(what, sizeof what, "%s %s", op == O_MIN ? "min with" :
			    op == O_MAX ? "max with" : opinfo[op].name, k2);
			if (op == O_AND || op == O_OR || op == O_MIN ||
			    op == O_MAX || op == O_LAND || op == O_LOR ||
			    is_const(N.b[n]) || op == O_SHL || op == O_SHR ||
			    op == O_ADD || op == O_SUB || op == O_XOR)
				cand_add(&c, &nc, &cap, root,
				    try_replace(root, n, N.a[n]), what);
			if (opinfo[op].fl & OF_COMM) {
				short_str(k1, sizeof k1, N.a[n]);
				snprintf(what, sizeof what, "%s %s", op == O_MIN ?
				    "min with" : op == O_MAX ? "max with" :
				    opinfo[op].name, k1);
				cand_add(&c, &nc, &cap, root,
				    try_replace(root, n, N.b[n]), what);
			}
		}
	}
	free(mark);
	free(st);
	notes_drop();
	qsort(c, (size_t)nc, sizeof *c, cmp_pcand);
	*out = c;
	return nc;
}

static const char *
proof_word(const struct eqproof *pr)
{
	return pr->how == EQ_EXHAUSTIVE ? "exhaustive" : pr->how == EQ_SAT ?
	    "proved by SAT" : "sampled, not proved";
}

/*
 * Peephole candidates from the function's own DAG, before the enumerative
 * bank: each single drop that holds (cheapest first), then the drops
 * applied repeatedly while a proved one remains (a fixpoint).
 */
static void
peephole(const struct func *f)
{
	struct pcand *c;
	int nc, i, shown = 0, proofs = 0, round, base = icost(f->body);
	nid cur = f->body;
	struct sbuf steps = {0}, w = {0};
	int nsteps = 0, more = 0;
	const char *last = "exhaustive";

	clock_t t0 = clock();

	if (!TY_ISINT(N.ty[f->body]))
		return;
	notes_flush();
	nc = peep_cands(f->body, &c);
	rep_list("peephole");
	for (i = 0; i < nc && proofs < PEEP_PROOFS; i++) {
		struct eqproof pr;
		if (c[i].cost >= base || !quick_refines(f, c[i].e))
			continue;
		if ((double)(clock() - t0) / CLOCKS_PER_SEC > PEEP_SECONDS)
			break;
		proofs++;
		sb_clear(&w);
		if (equiv_decide(f, f->body, c[i].e, &w, &pr))
			continue;
		if (!shown++)
			printf("  dropping operations of %s (cost %d as written; "
			    "each checked on the whole domain):\n", f->anon ?
			    "the expression" : sym_str(f->name), base);
		if (shown > PEEP_SHOWN) {
			more++;
			continue;
		}
		printf("  cost %-3d %-40s drops %s (%s)\n", c[i].cost,
		    shared_str(c[i].e, 300), c[i].what, proof_word(&pr));
		rep_obj(NULL);
		rep_num("cost", c[i].cost);
		rep_expr("expr", c[i].e);
		rep_str("drops", c[i].what);
		rep_str("proof", pr.how == EQ_EXHAUSTIVE ? "exhaustive" :
		    pr.how == EQ_SAT ? "sat" : "sampled");
		rep_end();
	}
	free(c);
	if (more)
		printf("  ... %d more single drops hold (not listed)\n", more);
	/* the fixpoint: take the cheapest proved drop until none is left */
	for (round = 0; round < PEEP_ROUNDS && proofs < PEEP_PROOFS; round++) {
		int took = 0, cc = icost(cur);
		nc = peep_cands(cur, &c);
		for (i = 0; i < nc && proofs < PEEP_PROOFS; i++) {
			struct eqproof pr;
			if (c[i].cost >= cc || !quick_refines(f, c[i].e))
				continue;
			if ((double)(clock() - t0) / CLOCKS_PER_SEC > PEEP_SECONDS)
				break;
			proofs++;
			if (equiv_decide(f, f->body, c[i].e, NULL, &pr) ||
			    pr.how == EQ_SAMPLED)
				continue;
			if (pr.how == EQ_SAT)
				last = "proved by SAT";
			sb_printf(&steps, "%s%s", nsteps ? ", then " : "",
			    c[i].what);
			nsteps++;
			cur = c[i].e;
			took = 1;
			break;
		}
		free(c);
		if (!took)
			break;
	}
	rep_end();
	if (nsteps >= 2) {
		printf("  cost %-3d %-40s all of them: drops %s (%s)\n",
		    icost(cur), shared_str(cur, 300), sb_str(&steps), last);
		rep_obj("peephole_fixpoint");
		rep_num("cost", icost(cur));
		rep_expr("expr", cur);
		rep_str("drops", sb_str(&steps));
		rep_str("proof", strcmp(last, "exhaustive") ? "sat" : "exhaustive");
		rep_end();
	}
	free(steps.p);
	free(w.p);
}

/* ---- parametric forms: split on a parameter with few values ---- */

#define PCASES 8
#define PXMAX (1 << 16)

struct pcase {
	int64_t c;              /* the case value */
	int64_t *x, *y;         /* the points */
	uint32_t n;
	int64_t klo, khi;       /* feasible multipliers for the chosen S, C */
};

static int64_t
fdiv(int64_t a, int64_t b)      /* floor(a / b), b > 0 */
{
	int64_t q = a / b;

	return (a % b != 0 && (a < 0)) ? q - 1 : q;
}

static int64_t
cdiv(int64_t a, int64_t b)      /* ceil(a / b), b > 0 */
{
	return -fdiv(-a, b);
}

/* K with floor((K x + C) / 2^S) = y at every point: [*lo, *hi] or 0 */
static int
k_range(const struct pcase *pc, int S, int64_t C, int64_t *lo, int64_t *hi)
{
	int64_t P = (int64_t)1 << S, l = INT64_MIN / 4, h = INT64_MAX / 4;
	uint32_t i;

	for (i = 0; i < pc->n; i++) {
		int64_t x = pc->x[i], y = pc->y[i];
		int64_t a = y * P - C, b = (y + 1) * P - 1 - C;
		if (x == 0) {
			if (a > 0 || b < 0)
				return 0;
			continue;
		}
		if (cdiv(a, x) > l)
			l = cdiv(a, x);
		if (fdiv(b, x) < h)
			h = fdiv(b, x);
		if (l > h)
			return 0;
	}
	if (l < 0)
		l = 0;
	*lo = l;
	*hi = h;
	return l <= h;
}

/* the smallest "nice" value in [lo, hi]: 2^k - 1, 2^k, or lo */
static int64_t
nice_in(int64_t lo, int64_t hi)
{
	int k;

	for (k = 1; k < 62; k++) {
		int64_t v = ((int64_t)1 << k) - 1;
		if (v >= lo && v <= hi)
			return v;
		if (v + 1 >= lo && v + 1 <= hi)
			return v + 1;
	}
	return lo;
}

static void
parametric(const struct func *f)
{
	int s, xs, k, S, ncase = 0, found = 0;
	int64_t lo[MAXP], hi[MAXP], C = 0, cv;
	struct pcase pc[PCASES];
	struct plan p;
	nid roots[2];
	int64_t *in_i;
	double *in_r;
	struct ity it;

	if (f->np != 2 || !TY_ISINT(N.ty[f->body]) ||
	    (F.it.k != K_UINT && F.it.k != K_SINT))
		return;
	for (k = 0; k < 2; k++) {
		if (!TY_ISINT(f->p[k].ty))
			return;
		param_bounds(&f->p[k], &lo[k], &hi[k]);
	}
	where_bounds(f, lo, hi, NULL);
	/* the case parameter: the one with few values */
	if (hi[0] - lo[0] < PCASES && hi[0] >= lo[0] &&
	    !(hi[1] - lo[1] < PCASES))
		s = 0;
	else if (hi[1] - lo[1] < PCASES && hi[1] >= lo[1] &&
	    !(hi[0] - lo[0] < PCASES))
		s = 1;
	else
		return;
	xs = 1 - s;
	if (lo[xs] < 0 || hi[xs] - lo[xs] >= PXMAX || hi[s] - lo[s] < 1)
		return;
	ity_of(&it, F.W, 0);
	roots[0] = f->body;
	roots[1] = f->where;
	plan_build(&p, roots, 2, 0);
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	memset(pc, 0, sizeof pc);
	for (cv = lo[s]; cv <= hi[s]; cv++) {
		struct pcase *q = &pc[ncase++];
		int64_t x = lo[xs];
		q->c = cv;
		q->x = xmalloc((size_t)(hi[xs] - lo[xs] + 1) * sizeof *q->x);
		q->y = xmalloc((size_t)(hi[xs] - lo[xs] + 1) * sizeof *q->y);
		while (x <= hi[xs]) {
			uint32_t np = 0, i;
			uint8_t ok[CH];
			for (; np < CH && x <= hi[xs]; np++, x++) {
				in_i[(size_t)s * CH + np] = cv;
				in_i[(size_t)xs * CH + np] = x;
			}
			plan_run(&p, np, in_i, in_r);
			valid_mask(&p, f->where ? p.root[1] : NOTOK, np, ok);
			for (i = 0; i < np; i++) {
				int64_t y = COLI(&p, p.root[0])[i];
				if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF))
					continue;
				if (y < 0 || y > ((int64_t)1 << 32))
					goto done;
				q->x[q->n] = in_i[(size_t)xs * CH + i];
				q->y[q->n++] = y;
			}
		}
	}
	/* the smallest shift S with one C for every case */
	for (S = 0; S <= 24 && !found; S++) {
		int64_t cands[4];
		int nc = 0, ci;
		if (S > 0)
			cands[nc++] = (int64_t)1 << (S - 1);
		cands[nc++] = 0;
		if (S > 0)
			cands[nc++] = ((int64_t)1 << (S - 1)) - 1;
		for (ci = 0; ci < nc && !found; ci++) {
			int all = 1;
			for (k = 0; k < ncase && all; k++)
				if (pc[k].n && !k_range(&pc[k], S, cands[ci],
				    &pc[k].klo, &pc[k].khi))
					all = 0;
			if (all) {
				C = cands[ci];
				found = 1;
			}
		}
	}
	S--;
	if (!found)
		goto done;
	/* no overflow in the computing type at the largest x */
	for (k = 0; k < ncase; k++) {
		double top = (double)pc[k].khi * (double)hi[xs] + (double)C;
		double tmax = it.k == K_SINT ? (double)it.smax : (double)it.mask;
		if (pc[k].n && top > tmax) {
			pc[k].khi = (int64_t)((tmax - (double)C) /
			    (double)(hi[xs] ? hi[xs] : 1));
			if (pc[k].khi < pc[k].klo)
				goto done;
		}
	}
	{
		nid xv = mk_cast(F.W, mk_var((uint32_t)xs, f->p[xs].name,
		    f->p[xs].ty));
		nid cvar = mk_cast(F.W, mk_var((uint32_t)s, f->p[s].name,
		    f->p[s].ty));
		nid kform[4];
		const char *kwhy[4];
		int nf = 0, i;
		int64_t Alo, Ahi, A;
		char why[512];
		/* K = A / (2^c - 1), A >> c, A / c, A*c + B */
		static const char *const fam[3] = {
			"a constant over 2^case - 1",
			"a constant shifted by the case",
			"a constant over the case" };
		for (i = 0; i < 3; i++) {
			int ok = 1;
			Alo = 0, Ahi = INT64_MAX / 4;
			for (k = 0; k < ncase && ok; k++) {
				int64_t c = pc[k].c, m, l, h;
				if (!pc[k].n)
					continue;
				if (i == 0) {
					if (c < 1 || c > 30) { ok = 0; break; }
					m = ((int64_t)1 << c) - 1;
				} else if (i == 1) {
					if (c < 0 || c > 30) { ok = 0; break; }
					m = (int64_t)1 << c;
				} else {
					if (c < 1) { ok = 0; break; }
					m = c;
				}
				l = pc[k].klo * m;
				h = (pc[k].khi + 1) * m - 1;
				if (l > Alo)
					Alo = l;
				if (h < Ahi)
					Ahi = h;
				if (Alo > Ahi)
					ok = 0;
			}
			if (!ok || Ahi > (it.k == K_SINT ? it.smax : (int64_t)it.mask))
				continue;
			A = nice_in(Alo, Ahi);
			if (i == 0)
				kform[nf] = mk_bin(O_DIV, kconst(A), mk_bin(O_SUB,
				    mk_bin(O_SHL, kconst(1), cvar), kconst(1)));
			else if (i == 1)
				kform[nf] = mk_bin(O_SHR, kconst(A), cvar);
			else
				kform[nf] = mk_bin(O_DIV, kconst(A), cvar);
			kwhy[nf++] = fam[i];
		}
		/* the select chain: one K per case */
		{
			nid chain = 0;
			for (k = ncase - 1; k >= 0; k--) {
				nid kk;
				if (!pc[k].n)
					continue;
				kk = kconst(nice_in(pc[k].klo, pc[k].khi));
				chain = chain ? mk_sel(mk_bin(O_EQ, cvar,
				    kconst(pc[k].c)), kk, chain) : kk;
			}
			kform[nf] = chain;
			kwhy[nf++] = "one constant per case";
		}
		printf("  split on %s (%d values), fitted (%s*K + C) >> S per "
		    "case: S = %d, C = %lld\n", sym_str(f->p[s].name), ncase,
		    sym_str(f->p[xs].name), S, (long long)C);
		rep_list("parametric");
		for (i = 0; i < nf; i++) {
			struct eqproof pr;
			nid e = mk_bin(O_MUL, xv, kform[i]);
			if (C)
				e = mk_bin(O_ADD, e, kconst(C));
			if (S)
				e = mk_bin(O_SHR, e, kconst(S));
			if (equiv_decide(f, f->body, e, NULL, &pr))
				continue;
			snprintf(why, sizeof why, "K = %s (%s) depends only on "
			    "%s: compute it once where %s is set", shared_str(kform[i],
			    100), kwhy[i], sym_str(f->p[s].name),
			    sym_str(f->p[s].name));
			printf("  cost %-3d %s\n           %s; %d per use once K "
			    "is known (%s)\n", icost(e), shared_str(e, 300), why,
			    icost(e) - icost(kform[i]), proof_word(&pr));
			rep_obj(NULL);
			rep_num("cost", icost(e));
			rep_num("cost_per_use", icost(e) - icost(kform[i]));
			rep_expr("expr", e);
			rep_expr("K", kform[i]);
			rep_str("why", why);
			rep_str("proof", pr.how == EQ_EXHAUSTIVE ? "exhaustive" :
			    pr.how == EQ_SAT ? "sat" : "sampled");
			rep_end();
		}
		rep_end();
	}
done:
	for (k = 0; k < ncase; k++) {
		free(pc[k].x);
		free(pc[k].y);
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
}

/* ---- entry points ---- */

void
cmd_find(struct func *f, int want, const uint8_t *opsel, int maxcost)
{
	struct diter d;
	struct plan p;
	nid roots[2 + MAXP];
	int64_t in_i[MAXP * CH], *xs;
	double in_r[MAXP * CH];
	uint32_t np, i, t, slot;
	int j, nk = 0;
	int64_t ks[32];
	ty_t rt = N.ty[f->body];

	(void)opsel;
	if (TY_FP(rt)) {
		find_float(f, want);
		return;
	}
	for (j = 0; j < f->np; j++)
		if (!TY_ISINT(f->p[j].ty))
			fail("find needs integer parameters (or a float result)");
	finder_free();
	F.f = *f;
	F.W = rt;
	if (!TY_SIZED(rt) || rt == TY_BOOL) {
		F.W = 0;
		for (j = 0; j < f->np; j++)
			if (TY_SIZED(f->p[j].ty) && f->p[j].ty != TY_BOOL &&
			    (!F.W || TYW(f->p[j].ty) > TYW(F.W)))
				F.W = f->p[j].ty;
		if (!F.W)
			F.W = TY_INT;
		if (rt != F.W)
			note("the result is %s: searching in %s", ty_str(rt),
			    ty_str(F.W));
		notes_flush();
	}
	ity_of(&F.it, F.W, 0);

	/* the points */
	diter_init(&d, f, FPTS, 2, FPTS);
	F.sampled = !d.exhaustive;
	roots[0] = f->body;
	roots[1] = f->where;
	for (j = 0; j < f->np; j++)
		roots[2 + j] = mk_cast(F.W, mk_var((uint32_t)j, f->p[j].name,
		    f->p[j].ty));
	plan_build(&p, roots, 2 + (uint32_t)f->np, 0);
	F.y = xmalloc(FPTS * sizeof *F.y);
	xs = xmalloc(FPTS * sizeof *xs);
	F.P = FPTS;    /* capacity during collection */
	{
		int64_t *pcols = xmalloc((size_t)FPTS * MAXP * sizeof *pcols);
		uint32_t n = 0;
		while ((np = diter_next(&d, in_i, in_r)) != 0) {
			uint8_t ok[CH];
			plan_run(&p, np, in_i, in_r);
			valid_mask(&p, f->where ? p.root[1] : NOTOK, np, ok);
			for (i = 0; i < np; i++) {
				if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF))
					continue;
				F.y[n] = int_norm(F.W, COLI(&p, p.root[0])[i]);
				xs[n] = in_i[i];
				for (j = 0; j < f->np; j++)
					pcols[(size_t)j * FPTS + n] =
					    COLI(&p, p.root[2 + j])[i];
				n++;
			}
		}
		F.P = n;
		if (n == 0) {
			plan_free(&p);
			free(pcols);
			free(xs);
			fail("the function is undefined on its whole domain");
		}
		F.maxn = (uint32_t)((128u << 20) / (F.P * sizeof(int64_t)));
		if (F.maxn > (4u << 20))
			F.maxn = 4u << 20;
		F.htcap = 1;
		while (F.htcap < F.maxn * 2)
			F.htcap *= 2;
		F.ht = xcalloc(F.htcap, sizeof *F.ht);
		/* leaves: parameters */
		for (j = 0; j < f->np; j++) {
			int64_t *c = &pcols[(size_t)j * FPTS];
			if (insert(O_VAR, 0, 0, 0, c, &t)) {
				F.node[t] = roots[2 + j];
				if (is_target(c))
					add_match(M_EXACT, 0, F.node[t], "exact");
				holes(t);
			}
		}
		free(pcols);
	}
	plan_free(&p);
	F.firstk = F.n;
	/* leaves: constants */
	ks[nk++] = 1;
	ks[nk++] = 2;
	ks[nk++] = 3;
	ks[nk++] = 4;
	ks[nk++] = 5;
	ks[nk++] = 8;
	ks[nk++] = F.it.w > 1 ? F.it.w - 1 : 1;
	ks[nk++] = -1;
	{
		int64_t *c = xmalloc(F.P * sizeof *c);
		for (j = 0; j < nk; j++) {
			int64_t v = int_norm(F.W, ks[j]);
			for (i = 0; i < F.P; i++)
				c[i] = v;
			if (lookup(c, &slot) == NOTOK && insert(O_CONST, 0, 0, 0, c, &t))
				F.node[t] = kconst(v);
		}
		free(c);
	}
	F.lvl[0] = 0;
	F.lvl[1] = F.n;
	F.done = 0;
	F.nops = 0;
	{
		static const int defops[] = {O_ADD, O_SUB, O_MUL, O_AND, O_OR,
		    O_XOR, O_SHL, O_SHR, O_NEG, O_NOT};
		for (j = 0; j < (int)(sizeof defops / sizeof defops[0]); j++) {
			F.ops[F.nops] = defops[j];
			F.opc[F.nops++] = op_cost(defops[j]);
		}
	}
	/* constant target */
	for (i = 1; i < F.P && F.y[i] == F.y[0]; i++)
		;
	if (i == F.P)
		add_match(M_EXACT, 0, kconst(F.y[0]), "constant");
	forms(xs);
	free(xs);
	F.active = 1;
	printf("find %s: %u point%s%s, computing in %s, costs from %s\n",
	    f->anon ? node_str(f->body) : sym_str(f->name), F.P,
	    F.P == 1 ? "" : "s", F.sampled ? " (sampled; matches verified on "
	    "the whole domain)" : "", ty_str(F.W),
	    sym_str(model_active()->name));
	rep_begin("find");
	rep_int("points", F.P);
	rep_bool("sampled", F.sampled);
	rep_str("type", ty_str(F.W));
	rep_str("model", sym_str(model_active()->name));
	peephole(&F.f);
	parametric(&F.f);
	rep_list("implementations");
	run_levels(want, maxcost > 0 ? maxcost : 12);
	rep_end();
}

void
cmd_more(int want)
{
	if (!F.active)
		fail("nothing to continue: run 'find' first");
	printf("more for %s:\n", F.f.anon ? node_str(F.f.body) :
	    sym_str(F.f.name));
	rep_begin("find");
	rep_list("implementations");
	run_levels(want, F.done + 8);
	rep_end();
}
