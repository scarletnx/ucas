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
#include "x86.h"
#include "fuse.h"
#include "irw.h"
#include "dword.h"
#include "texalu.h"
#include "stoch.h"

#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FPTS 1024
#define MAXOPS 14

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

static int fquiet;	/* find_cands: no output, no proofs */

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

/*
 * x86 BMI1 instructions as one operation each (under an x86 model with
 * BMI1): andn(a, b) = ~a & b, blsr(a) = a & (a - 1), blsi(a) = a & -a,
 * blsmsk(a) = a xor (a - 1).  Their terms are ordinary nodes; the cost is
 * one instruction's.  pdep/pext with a constant mask (BMI2) are solved
 * as templates over a parameter (holes).
 */
enum { XO_ANDN = O_NOPS, XO_BLSR, XO_BLSI, XO_BLSMSK };

static int
oarity(int op)
{
	return op >= O_NOPS ? (op == XO_ANDN ? 2 : 1) : opinfo[op].arity;
}

static nid
xo_node(int op, nid a, nid b)
{
	nid m1;

	switch (op) {
	case XO_ANDN: return mk_bin(O_AND, mk_un(O_NOT, a), b);
	case XO_BLSI: return mk_bin(O_AND, a, mk_un(O_NEG, a));
	}
	m1 = mk_bin(O_SUB, a, mk_const_i(N.ty[a], 1));
	return mk_bin(op == XO_BLSR ? O_AND : O_XOR, a, m1);
}

static int64_t
xo_eval(int op, int64_t a, int64_t b)
{
	uint64_t u = (uint64_t)a;

	switch (op) {
	case XO_ANDN: return int_norm(F.W, (int64_t)(~u & (uint64_t)b));
	case XO_BLSR: return int_norm(F.W, (int64_t)(u & (u - 1)));
	case XO_BLSI: return int_norm(F.W, (int64_t)(u & (0 - u)));
	}
	return int_norm(F.W, (int64_t)(u ^ (u - 1)));
}

/* the active model's x86 µarch with feature f, or -1 */
static int
x86_with(int f)
{
	int ua = uarch_of_model(model_active());

	return ua >= 0 && (uarchs[ua].feat & XB(f)) ? ua : -1;
}

static nid
term_node(uint32_t t)
{
	nid r;

	if (F.node[t])
		return F.node[t];
	if (F.op[t] >= O_NOPS)
		r = xo_node(F.op[t], term_node(F.a[t]), oarity(F.op[t]) > 1 ?
		    term_node(F.b[t]) : 0);
	else if (opinfo[F.op[t]].arity == 1)
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

	/* the bank adds its operations' prices; the model's fusions may
	 * make the form cheaper than that */
	if (fuse_count(model_active()) && e) {
		int fc = (int)(fuse_expr_cost(model_active(), e) + 0.5f);
		if (fc < cost)
			cost = fc;
	}
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
	if (F.op[t] != op || oarity(op) != 2 || t < F.lvl[1])
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

/*
 * y = pext(t, K) or pdep(t, K) for a constant mask K, solved exactly.
 * pext: bit j of y is bit p_j of t with p_0 < p_1 < ...; taking each p_j
 * as the first position after p_(j-1) whose column equals y's bit j is
 * complete (a subsequence match: if any choice works the earliest one
 * does).  pdep: positions p in order, each either in K (it receives the
 * next bit of t) or not (y's bit p must be 0 there); reach[p][j] is a
 * dynamic program over (position, bits used).  The result is spelled
 * with shifts and masks (one node per run of consecutive positions) and,
 * like every match, checked against the function by enumeration or SAT.
 */
static int
colbit_eq(const int64_t *a, int ia, const int64_t *b, int ib)
{
	uint32_t i;

	for (i = 0; i < F.P; i++)
		if ((((uint64_t)a[i] >> ia) ^ ((uint64_t)b[i] >> ib)) & 1)
			return 0;
	return 1;
}

static int
colbit_zero(const int64_t *a, int ia)
{
	uint32_t i;

	for (i = 0; i < F.P; i++)
		if (((uint64_t)a[i] >> ia) & 1)
			return 0;
	return 1;
}

/* the node for y = OR over j of bit src_j of x, put at bit dst_j */
static nid
bitmove_node(nid x, const int *src, const int *dst, int n)
{
	nid r = 0;
	int j = 0;

	while (j < n) {
		int k = j + 1, sh;
		uint64_t m;
		nid e;
		while (k < n && src[k] == src[k - 1] + 1 && dst[k] == dst[k - 1] + 1)
			k++;
		m = (k - j >= 64 ? ~0ULL : ((1ULL << (k - j)) - 1)) << dst[j];
		sh = dst[j] - src[j];
		e = sh > 0 ? mk_bin(O_SHL, x, kconst(sh)) : sh < 0 ?
		    mk_bin(O_SHR, x, kconst(-sh)) : x;
		e = mk_bin(O_AND, e, kconst(int_norm(F.W, (int64_t)m)));
		r = r ? mk_bin(O_OR, r, e) : e;
		j = k;
	}
	return r;
}

static void
pdep_pext(uint32_t t)
{
	const int64_t *c = COL(t), *y = F.y;
	int w = F.it.w, top = -1, j, p, src[64], dst[64], n;
	const struct xcost *xc = xcost_get(x86_with(XF_BMI2), XC_PDEP, VW_X);
	int pc = xc ? (int)(xc->lat + 0.5f) : 3;
	uint32_t i;
	uint64_t K;
	char why[64];

	if (F.it.k != K_UINT || w > 64 || F.it.w < 8)
		return;
	for (i = 0; i < F.P; i++)
		for (j = w; j-- > top + 1;)
			if (((uint64_t)y[i] >> j) & 1) {
				top = j;
				break;
			}
	if (top < 0)
		return;
	/* pext */
	for (j = 0, p = 0, K = 0; j <= top; j++, p++) {
		while (p < w && !colbit_eq(c, p, y, j))
			p++;
		if (p == w)
			break;
		K |= 1ULL << p;
		src[j] = p;
		dst[j] = j;
	}
	if (j > top && __builtin_popcountll(K) > 1) {
		n = j;
		snprintf(why, sizeof why, "pext(%s, 0x%llx)", node_str(term_node(t)),
		    (unsigned long long)K);
		add_match(M_FORM, pc, bitmove_node(term_node(t), src, dst, n),
		    strdup(why));
	}
	/* pdep: reach[p][j], positions 0..p-1 decided with j bits of t used */
	{
		static uint8_t reach[65][65], how[65][65];
		int q;
		memset(reach, 0, sizeof reach);
		reach[0][0] = 1;
		for (p = 0; p < w; p++)
			for (q = 0; q <= p; q++) {
				if (!reach[p][q])
					continue;
				/* p not in K: y's bit p is 0 */
				if (colbit_zero(y, p) && !reach[p + 1][q]) {
					reach[p + 1][q] = 1;
					how[p + 1][q] = 0;
				}
				/* p in K: y's bit p is t's bit q */
				if (q < w && colbit_eq(c, q, y, p) && !reach[p + 1][q + 1]) {
					reach[p + 1][q + 1] = 1;
					how[p + 1][q + 1] = 1;
				}
			}
		/* the fewest bits of t: a smaller mask */
		for (q = 0; q <= w && !reach[w][q]; q++)
			;
		if (q > 1 && q <= w) {
			int pp = w, qq = q;
			K = 0;
			n = 0;
			while (pp > 0) {
				if (how[pp][qq]) {
					K |= 1ULL << (pp - 1);
					qq--;
				}
				pp--;
			}
			for (p = 0, j = 0; p < w; p++)
				if ((K >> p) & 1) {
					src[n] = j++;
					dst[n++] = p;
				}
			/* a plain shift or mask is found by the search itself */
			snprintf(why, sizeof why, "pdep(%s, 0x%llx)",
			    node_str(term_node(t)), (unsigned long long)K);
			add_match(M_FORM, pc, bitmove_node(term_node(t), src, dst, n),
			    strdup(why));
		}
	}
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
	if (t >= F.lvl[1] && oarity(F.op[t]) == 2 &&
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
	if (F.op[t] == O_VAR && x86_with(XF_BMI2) >= 0)
		pdep_pext(t);
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
	const int64_t *ca = COL(a), *cb = oarity(op) > 1 ? COL(b) : ca;
	uint32_t i, t;
	uint8_t fl = 0;
	int64_t first;

	for (i = 0; i < F.P; i++) {
		scratch[i] = op >= O_NOPS ? xo_eval(op, ca[i], cb[i]) :
		    scalar_int(op, &F.it, &F.it, ca[i], cb[i], 0, &fl);
		if (fl & (F_UNDEF | F_UB))
			return;
	}
	first = scratch[0];
	for (i = 1; i < F.P && scratch[i] == first; i++)
		;
	if (i == F.P)
		return;   /* constants come from the leaves and the holes */
	if (is_target(scratch)) {
		static const char *const xname[] = {"andn", "blsr", "blsi",
		    "blsmsk"};
		nid e = op >= O_NOPS ? xo_node(op, term_node(a), oarity(op) > 1 ?
		    term_node(b) : 0) : opinfo[op].arity == 1 ? mk_un(op,
		    term_node(a)) : mk_bin(op, term_node(a), term_node(b));
		add_match(M_EXACT, cost, e, op >= O_NOPS ? xname[op - O_NOPS] :
		    "exact");
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
		if (oarity(op) == 1) {
			for (i = F.lvl[c - oc]; i < F.lvl[c - oc + 1] && !F.full; i++)
				if (i < F.firstk || c - oc > 0)
					build(op, i, 0, c, scratch);
			continue;
		}
		for (ca = 0; ca <= c - oc; ca++) {
			int cb = c - oc - ca;
			int comm = op < O_NOPS && (opinfo[op].fl & OF_COMM) != 0;
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

/*
 * A map on scattered points (a where clause that leaves only some keys):
 * the values with no table, ((x M + A) >> s) & mask plus the least value,
 * M searched and A solved for it (ms_solve, stoch.c).  On a run of
 * consecutive points the values of that form step by one of two adjacent
 * amounts, a floor of a line that idrw.c solves exactly, so only
 * scattered points come here.
 */
#define MS_TRIES (1L << 20)

static void
sparse_mulshift(const int64_t *xs)
{
	uint32_t *k, *t, M, A, i;
	int64_t vmin = F.y[0], vmax = F.y[0];
	int b, s, cost;
	long tries;
	int exh;
	ty_t u = TY(K_UINT, 32);
	const struct param *pp = &F.f.p[0];
	nid e;

	if (F.P < 3 || F.P > 256 || !TY_SIZED(pp->ty) || TYW(pp->ty) > 32)
		return;
	for (i = 1; i < F.P; i++) {
		vmin = F.y[i] < vmin ? F.y[i] : vmin;
		vmax = F.y[i] > vmax ? F.y[i] : vmax;
	}
	if (vmax == vmin || (uint64_t)(vmax - vmin) >= (1u << 16))
		return;
	for (b = 1; ((int64_t)1 << b) <= vmax - vmin; b++)
		;
	k = xmalloc(F.P * sizeof *k);
	t = xmalloc(F.P * sizeof *t);
	for (i = 0; i < F.P; i++) {
		k[i] = (uint32_t)(uint64_t)xs[i];
		t[i] = (uint32_t)(F.y[i] - vmin);
	}
	if (!ms_solve(k, t, (int)F.P, b, stoch_seed(), MS_TRIES, &M, &A, &s,
	    &tries, &exh)) {
		free(k);
		free(t);
		return;
	}
	free(k);
	free(t);
	e = mk_bin(O_MUL, mk_cast(u, mk_var(0, pp->name, pp->ty)),
	    mk_const_i(u, M));
	cost = op_cost(O_MUL);
	if (A) {
		e = mk_bin(O_ADD, e, mk_const_i(u, A));
		cost += op_cost(O_ADD);
	}
	if (s) {
		e = mk_bin(O_SHR, e, mk_const_i(u, s));
		cost += op_cost(O_SHR);
	}
	if (s + b < 32) {
		e = mk_bin(O_AND, e, mk_const_i(u, (int64_t)((1u << b) - 1)));
		cost += op_cost(O_AND);
	}
	e = mk_cast(F.W, e);
	if (vmin) {
		e = add_k(e, vmin);
		cost += op_cost(O_ADD);
	}
	add_match(M_FORM, cost, e, "the values on the scattered points by a "
	    "multiply-add and a shift (M searched, A solved)");
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
	if (!contiguous) {
		sparse_mulshift(xs);
		return;
	}
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

/* e's price under the active model: the cheapest cover by its fusions */
static float
expr_cost(nid e)
{
	return fuse_expr_cost(model_active(), e);
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
		fail("find on a float result needs one f16, f32 or f64 parameter");
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
	char what[200];
	int derived;		/* irw.c: an identity, proved by derivation */
	int loose;		/* irw.c: it relies on a GPU's looser floats */
	int nosat;		/* irw.c: checked on the domain's points only */
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
	(*c)[*nc].derived = 0;
	(*c)[*nc].loose = 0;
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
		/* a subgroup operation's operand is a value per lane, not
		 * something the operation could be dropped for */
		if (!TY_ISINT(N.ty[n]) || op == O_INDEX || IS_SG(op))
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
				/* the bits of a float (asint, asuint) are not
				 * the float: nothing to drop */
				if (!TY_ISINT(N.ty[N.a[n]]))
					break;
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
 * The second opinion on a derived rewrite: equiv_decide, or for one
 * SAT does not finish on (a multiplier against a divider: its derivation
 * is the proof) the domain's points alone, all of them or a sample.
 */
static int
derived_check(const struct func *f, nid e, int nosat, struct sbuf *w,
    struct eqproof *pr)
{
	int sampled = 0, bad;

	if (!nosat)
		return equiv_decide(f, f->body, e, w, pr);
	memset(pr, 0, sizeof *pr);
	bad = equiv_check(f, f->body, e, w, &sampled);
	pr->how = sampled ? EQ_SAMPLED : EQ_EXHAUSTIVE;
	return bad;
}

/* the rewrites irw.c derives for root, added to the candidates */
static int
with_rewrites(const struct func *f, nid root, struct pcand **c, int nc)
{
	struct irw *r;
	int nr = irw_cands(f, root, &r), i, cap = nc;

	for (i = 0; i < nr; i++) {
		if (r[i].nearmiss)
			continue;
		*c = xrealloc(*c, (size_t)(cap + 1) * sizeof **c);
		cap++;
		(*c)[nc].e = r[i].e;
		(*c)[nc].cost = icost(r[i].e);
		snprintf((*c)[nc].what, sizeof (*c)[nc].what, "%s", r[i].what);
		(*c)[nc].derived = 1;
		(*c)[nc].loose = r[i].loose;
		(*c)[nc].nosat = r[i].nosat;
		nc++;
	}
	irw_free(r, nr);
	qsort(*c, (size_t)nc, sizeof **c, cmp_pcand);
	return nc;
}

/*
 * The rewrites by identities (irw.c) that the model prices lower than
 * the function as written: each with the conditions it holds under and
 * its derivation, then checked again on the domain (exhaustively, by
 * SAT, or sampled when floats are involved) as a second opinion.  One
 * that would be cheaper but is not proved says why.
 */
static void
rewrites(const struct func *f, int base)
{
	struct irw *r;
	int nr = irw_cands(f, f->body, &r), i, shown = 0;
	struct sbuf w = {0};

	for (i = 0; i < nr; i++) {
		struct eqproof pr;
		int cost = icost(r[i].e), bad;
		if (cost >= base) {
			if (!r[i].nearmiss && !r[i].quiet)
				printf("  derived, but not cheaper under %s (cost %d, as "
				    "written %d): %s\n", sym_str(model_active()->name),
				    cost, base, r[i].what);
			continue;
		}
		if (r[i].nearmiss) {
			printf("  not applied (it would cost %d): %s\n", cost,
			    r[i].what);
			irw_print(&r[i], "      ");
			rep_obj(NULL);
			rep_num("cost", cost);
			rep_str("rewrite", r[i].what);
			rep_str("not_proved", r[i].proof);
			rep_end();
			continue;
		}
		sb_clear(&w);
		bad = derived_check(f, r[i].e, r[i].nosat, &w, &pr);
		if (!shown++)
			printf("  rewrites the model prices lower (each derived, with "
			    "what it holds under):\n");
		printf("  cost %-3d %-40s %s (derived; %s%s%s)\n", cost,
		    shared_str(r[i].e, 300), r[i].what, bad ? "but contradicted "
		    "by the check: " : pr.how == EQ_EXHAUSTIVE ? "checked on every "
		    "point" : pr.how == EQ_SAT ? "checked by SAT" : "checked on a "
		    "sample", bad ? sb_str(&w) : "", r[i].loose ? ", with ucas's "
		    "correctly rounded division; the derivation covers 2.5 ulp" :
		    "");
		irw_print(&r[i], "      ");
		rep_obj(NULL);
		rep_num("cost", cost);
		rep_expr("expr", r[i].e);
		rep_str("rewrite", r[i].what);
		rep_str("holds_where", r[i].cond);
		rep_str("derivation", r[i].proof);
		rep_str("check", bad ? "counterexample" : pr.how == EQ_EXHAUSTIVE ?
		    "exhaustive" : pr.how == EQ_SAT ? "sat" : "sampled");
		rep_end();
	}
	irw_free(r, nr);
	free(w.p);
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
	struct sbuf steps = {0}, steps2 = {0}, w = {0};
	int nsteps = 0, more = 0, nderived = 0, loose = 0;
	const char *last = "exhaustive";

	clock_t t0 = clock();

	if (!TY_ISINT(N.ty[f->body]))
		return;
	if (fquiet) {
		/* for programs: every cheaper drop, unproved (the caller
		 * proves what it takes) */
		nc = peep_cands(f->body, &c);
		for (i = 0; i < nc; i++)
			if (c[i].cost < base && quick_refines(f, c[i].e)) {
				char w[120];
				snprintf(w, sizeof w, "drops %s", c[i].what);
				add_match(M_FORM, c[i].cost, c[i].e, strdup(w));
			}
		free(c);
		return;
	}
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
	rewrites(f, base);
	/* the fixpoint: take the cheapest proved drop until none is left */
	for (round = 0; round < PEEP_ROUNDS && proofs < PEEP_PROOFS; round++) {
		int took = 0, cc = icost(cur);
		nc = peep_cands(cur, &c);
		nc = with_rewrites(f, cur, &c, nc);
		for (i = 0; i < nc && proofs < PEEP_PROOFS; i++) {
			struct eqproof pr;
			if (c[i].cost >= cc || !quick_refines(f, c[i].e))
				continue;
			/*
			 * after a rewrite that relies on a GPU's float precision,
			 * a drop checked with ucas's exact floats may remove what
			 * that precision needs (a correction step): only
			 * derivations from here on
			 */
			if (loose && !c[i].derived)
				continue;
			if ((double)(clock() - t0) / CLOCKS_PER_SEC > PEEP_SECONDS)
				break;
			proofs++;
			if (derived_check(f, c[i].e, c[i].derived && c[i].nosat,
			    NULL, &pr) || (pr.how == EQ_SAMPLED && !c[i].derived))
				continue;
			if (pr.how == EQ_SAT)
				last = "proved by SAT";
			if (c[i].derived)
				nderived++;
			loose |= c[i].loose;
			sb_printf(&steps, "%s%s", nsteps ? ", then " : "",
			    c[i].what);
			sb_printf(&steps2, "%s%s%s", nsteps ? ", then " : "",
			    c[i].derived ? "" : "drops ", c[i].what);
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
		printf("  cost %-3d %-40s all of them: %s%s (%s%s)\n",
		    icost(cur), shared_str(cur, 300), nderived ? "" : "drops ",
		    sb_str(nderived ? &steps2 : &steps), !nderived ? "" :
		    nderived == nsteps ? "derived" : "derived; the drops ",
		    nderived == nsteps ? "" : last);
		rep_obj("peephole_fixpoint");
		rep_num("cost", icost(cur));
		rep_expr("expr", cur);
		rep_str("drops", sb_str(nderived ? &steps2 : &steps));
		rep_str("proof", strcmp(last, "exhaustive") ? "sat" : "exhaustive");
		rep_end();
	}
	free(steps.p);
	free(steps2.p);
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
div_floor(int64_t a, int64_t b)      /* floor(a / b), b > 0 */
{
	int64_t q = a / b;

	return (a % b != 0 && (a < 0)) ? q - 1 : q;
}

static int64_t
cdiv(int64_t a, int64_t b)      /* ceil(a / b), b > 0 */
{
	return -div_floor(-a, b);
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
		if (div_floor(b, x) < h)
			h = div_floor(b, x);
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
		if (!fquiet) {
			printf("  split on %s (%d values), fitted (%s*K + C) >> S "
			    "per case: S = %d, C = %lld\n", sym_str(f->p[s].name),
			    ncase, sym_str(f->p[xs].name), S, (long long)C);
			rep_list("parametric");
		}
		for (i = 0; i < nf; i++) {
			struct eqproof pr;
			nid e = mk_bin(O_MUL, xv, kform[i]);
			if (C)
				e = mk_bin(O_ADD, e, kconst(C));
			if (S)
				e = mk_bin(O_SHR, e, kconst(S));
			if (fquiet) {
				snprintf(why, sizeof why, "split on %s, (%s*K + %lld) >> "
				    "%d per case, K = %s (%s)", sym_str(f->p[s].name),
				    sym_str(f->p[xs].name), (long long)C, S,
				    shared_str(kform[i], 60), kwhy[i]);
				add_match(M_FORM, icost(e), e, strdup(why));
				continue;
			}
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
		if (!fquiet)
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

/* the points, the leaves and the closed forms: find's state for f */
static void
find_setup(struct func *f)
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
		if (x86_with(XF_BMI1) >= 0) {
			static const int xops[] = {XO_ANDN, XO_BLSR, XO_BLSI,
			    XO_BLSMSK};
			for (j = 0; j < 4; j++) {
				F.ops[F.nops] = xops[j];
				F.opc[F.nops++] = op_cost(O_AND);
			}
		}
	}
	/* constant target */
	for (i = 1; i < F.P && F.y[i] == F.y[0]; i++)
		;
	if (i == F.P)
		add_match(M_EXACT, 0, kconst(F.y[0]), "constant");
	forms(xs);
	free(xs);
}

/* ---- library calls as the special-function unit's instructions ---- */

/*
 * The hyperbolic functions and their inverses have no instruction (tanh
 * an approximate one from Turing on): the driver runs a library sequence
 * (the models price one as 4 ftrans; tanh measured 17.7 on the GT 1030).  Each is written with exp,
 * log and sqrt, which are: form 0 is the formula Vulkan inherits its
 * precision from (so within that precision by definition), form 1 a
 * cheaper one with one exp, where there is one.
 */
static const struct {
	int op;
	const char *name, *vulkan, *one;
} sf_fn[] = {
	{O_TANH, "tanh", "sinh(x) / cosh(x), the halves cancelling exactly",
	    "1 - 2/(exp(2x) + 1), one exp, finite at both ends"},
	{O_SINH, "sinh", "(exp(x) - exp(-x)) * 0.5", "one exp and its reciprocal"},
	{O_COSH, "cosh", "(exp(x) + exp(-x)) * 0.5", "one exp and its reciprocal"},
	{O_ASINH, "asinh", "log(x + sqrt(x * x + 1))", NULL},
	{O_ACOSH, "acosh", "log(x + sqrt(x * x - 1))", NULL},
	{O_ATANH, "atanh", "log((1 + x) / (1 - x)) * 0.5", NULL},
};

static nid
sf_k(ty_t t, double v)
{
	return mk_cast(t, mk_const_r(v));
}

/* form k of the call op(x) of type t (sf_fn's comments), 0 none */
static nid
sf_form(int op, int k, nid x, ty_t t)
{
	nid e, em, one = sf_k(t, 1), half = sf_k(t, 0.5);

	switch (op) {
	case O_TANH:
		if (k)
			return mk_bin(O_SUB, one, mk_bin(O_DIV, sf_k(t, 2),
			    mk_bin(O_ADD, mk_un(O_EXP, mk_bin(O_MUL, sf_k(t, 2), x)),
			    one)));
		/* FALLTHROUGH */
	case O_SINH: case O_COSH:
		e = mk_un(O_EXP, x);
		em = k ? mk_bin(O_DIV, one, e) : mk_un(O_EXP, mk_un(O_NEG, x));
		if (op == O_TANH)
			return mk_bin(O_DIV, mk_bin(O_SUB, e, em), mk_bin(O_ADD, e,
			    em));
		return mk_bin(O_MUL, mk_bin(op == O_SINH ? O_SUB : O_ADD, e, em),
		    half);
	case O_ASINH: case O_ACOSH:
		return k ? 0 : mk_un(O_LOG, mk_bin(O_ADD, x, mk_un(O_SQRT,
		    mk_bin(op == O_ASINH ? O_ADD : O_SUB, mk_bin(O_MUL, x, x),
		    one))));
	case O_ATANH:
		return k ? 0 : mk_bin(O_MUL, mk_un(O_LOG, mk_bin(O_DIV,
		    mk_bin(O_ADD, one, x), mk_bin(O_SUB, one, x))), half);
	}
	return 0;
}

static int
sf_has_in(nid e, int op, uint8_t *seen)
{
	int ar = opinfo[N.op[e]].arity, r = 0;

	if (seen[e])
		return 0;
	seen[e] = 1;
	if (N.op[e] == op)
		return 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR)
		return 0;
	if (ar >= 1 && N.a[e])
		r |= sf_has_in(N.a[e], op, seen);
	if (ar >= 2 && N.b[e])
		r |= sf_has_in(N.b[e], op, seen);
	if (ar >= 3 && N.c[e])
		r |= sf_has_in(N.c[e], op, seen);
	return r;
}

/* e has a call of op */
static int
find_op(nid e, int op)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int r = sf_has_in(e, op, seen);

	free(seen);
	return r;
}

/* the operations sf_subst rebuilds from their operands */
static int
sf_plain(int op)
{
	return (op >= O_NEG && op < O_TABLE && op != O_CAST) || (op >= O_SEC &&
	    op <= O_FMA) || (op >= O_ASIN && op <= O_ATANH);
}

/* e with every op(x) as its form k; *ok 0 when one sits where e cannot be
 * rebuilt */
static nid
sf_subst(nid e, int op, int k, int *ok)
{
	nid a, b, c;
	int o = N.op[e];

	if (o == O_CONST || o == O_VAR)
		return e;
	a = N.a[e] ? sf_subst(N.a[e], op, k, ok) : 0;
	if (o == O_CAST)
		return a == N.a[e] ? e : mk_cast(N.ty[e], a);
	if (!sf_plain(o)) {
		if (find_op(e, op))
			*ok = 0;
		return e;
	}
	b = N.b[e] ? sf_subst(N.b[e], op, k, ok) : 0;
	c = N.c[e] ? sf_subst(N.c[e], op, k, ok) : 0;
	if (o == op) {
		nid r = sf_form(op, k, a, N.ty[e]);
		if (r == 0)
			*ok = 0;
		return r ? r : e;
	}
	if (a == N.a[e] && b == N.b[e] && c == N.c[e])
		return e;
	return mk_op(o, a, b, c);
}

/*
 * find: f's hyperbolic calls as their forms, each priced and measured
 * against f on its domain.  The count of forms listed.
 */
static int
sfu_find(struct func *f)
{
	int i, k, said = 0;
	double c0 = (double)expr_cost(f->body);

	for (i = 0; i < (int)(sizeof sf_fn / sizeof sf_fn[0]); i++) {
		if (!find_op(f->body, sf_fn[i].op))
			continue;
		for (k = 0; k < 2; k++) {
			double abs_, rel, ulp, c;
			uint64_t n, bad;
			int ok = 1;
			struct sbuf b = {0};
			nid e = sf_subst(f->body, sf_fn[i].op, k, &ok);
			if (!ok || e == f->body)
				continue;
			if (!said++)
				printf("find %s: library calls as exp, log and sqrt, "
				    "the special-function unit's instructions (%s "
				    "costs %g under %s);\n  measured with every "
				    "operation correctly rounded, which a GPU's exp, log, "
				    "sqrt and division need not be (Vulkan: exp 3 + 2|x| "
				    "ulp, log 3 ulp, division 2.5 ulp)\n", f->anon ?
				    node_str(f->body) : sym_str(f->name), f->anon ? "it" :
				    sym_str(f->name), c0, sym_str(model_active()->name));
			c = (double)expr_cost(e);
			error_measured(f, e, &abs_, &rel, &ulp, &n, &bad);
			print_node(&b, e, NULL);
			printf("  cost %-6g %s\n    %s as %s%s;\n    against %s on %llu "
			    "points: max abs %.3g, max rel %.3g (%.0f ulp)", c,
			    sb_str(&b), sf_fn[i].name, k ? sf_fn[i].one :
			    sf_fn[i].vulkan, k ? "" : ", the formula Vulkan inherits "
			    "its precision from: within it by definition", f->anon ?
			    "it" : sym_str(f->name), (unsigned long long)n,
			    abs_, rel, ulp);
			if (bad)
				printf(", NaN or infinite at %llu where %s is finite "
				    "(an exp out of range)", (unsigned long long)bad,
				    f->anon ? "it" : sym_str(f->name));
			printf("%s\n", c < c0 ? "" : "; not cheaper");
			free(b.p);
		}
	}
	return said;
}

/* ---- values packed into one f32: a flag in the sign bit, a value in a
 * quiet NaN's payload ---- */

#define PK_U32 TY(K_UINT, 32)

static nid
pk_u(uint64_t v)
{
	return mk_const_i(PK_U32, (int64_t)v);
}

/* a flag: a bool, or an integer whose range is 0..1 */
static int
pk_flag(const struct param *p)
{
	return p->ty == TY_BOOL || (TY_SIZED(p->ty) && p->ranged && p->lo == 0 &&
	    p->hi == 1);
}

/* the flag of type t in s's sign bit: a bool as the bits' signed
 * compare with 0 (one instruction, true at -0 too), 0/1 as the bit */
static nid
pk_sign_flag(nid s, ty_t t)
{
	if (t == TY_BOOL)
		return mk_bin(O_LT, mk_un(O_ASINT, s), mk_const_i(TY(K_SINT, 32),
		    0));
	return mk_cast(t, mk_bin(O_SHR, mk_un(O_ASUINT, s), pk_u(31)));
}

/* the flag p's value as a bool */
static nid
pk_true(const struct func *f, int k)
{
	nid v = mk_var((uint32_t)k, f->p[k].name, f->p[k].ty);

	return f->p[k].ty == TY_BOOL ? v : mk_bin(O_NE, v, mk_const_i(f->p[k].ty,
	    0));
}

/* what pk_guarded walks: anything else is not proved */
static int
pk_walks(int op)
{
	return sf_plain(op) || op == O_CAST || op == O_ASUINT || op == O_ASINT ||
	    op == O_ASFLOAT;
}

/*
 * Every use of parameter v in e sits in the arm of a select on the bool
 * parameter node c that is taken when c is want: the other arms and the
 * conditions are walked, those arms are not.  Shared nodes are walked
 * once (seen); only unguarded visits mark them.
 */
static int
pk_guarded(nid e, int v, nid c, int want, uint8_t *seen)
{
	int op = N.op[e], ar;

	if (seen[e])
		return 1;
	seen[e] = 1;
	if (op == O_VAR)
		return (int)N.ki[e] != v;
	if (op == O_CONST)
		return 1;
	if (!pk_walks(op))
		return 0;
	if (op == O_SEL) {
		nid cond = N.a[e];
		int pol = cond == c ? 1 : N.op[cond] == O_LNOT && N.a[cond] == c ?
		    0 : -1;
		if (pol >= 0)
			return pk_guarded(cond, v, c, want, seen) && pk_guarded(pol ==
			    want ? N.c[e] : N.b[e], v, c, want, seen);
	}
	ar = op == O_CAST ? 1 : opinfo[op].arity;
	return (ar < 1 || !N.a[e] || pk_guarded(N.a[e], v, c, want, seen)) &&
	    (ar < 2 || !N.b[e] || pk_guarded(N.b[e], v, c, want, seen)) &&
	    (ar < 3 || !N.c[e] || pk_guarded(N.c[e], v, c, want, seen));
}

static int
pk_only_under(const struct func *f, int v, int c, int want)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int r = pk_guarded(f->body, v, mk_var((uint32_t)c, f->p[c].name,
	    f->p[c].ty), want, seen);

	free(seen);
	return r;
}

/* f's body with the parameters in map[] replaced */
static nid
pk_subst(const struct func *f, const int *slot, const nid *to, int n)
{
	nid map[MAXP];
	int k, j;

	for (k = 0; k < f->np; k++) {
		map[k] = mk_var((uint32_t)k, f->p[k].name, f->p[k].ty);
		for (j = 0; j < n; j++)
			if (slot[j] == k)
				map[k] = to[j];
	}
	return subst(f->body, map, f->np);
}

/*
 * One packing, priced and checked: g is f over the packed word s (in
 * parameter x's slot), enc the word written from f's own parameters.
 */
static void
pk_report(const struct func *f, nid g, nid enc, int x, int words,
    const char *how)
{
	const struct model *m = model_active();
	double c0 = (double)expr_cost(f->body), c1 = (double)expr_cost(g);
	double abs_, rel, ulp;
	uint64_t n, bad;
	int sl[1] = {x};
	nid to[1], back;
	struct sbuf b = {0}, w = {0};
	struct func sf = *f;

	/* g at the word the writer stores: f again, which the check measures */
	sf.body = g;
	to[0] = enc;
	back = pk_subst(&sf, sl, to, 1);
	error_measured(f, back, &abs_, &rel, &ulp, &n, &bad);
	print_node(&b, g, NULL);
	print_node(&w, enc, NULL);
	printf("    %s\n    the writer stores s = %s; read back: %s\n", how,
	    sb_str(&w), sb_str(&b));
	printf("    cost %g against %g under %s (%+g to read them back), and %d "
	    "word%s less to store and to load (store %g, load %g each)\n", c1, c0,
	    sym_str(m->name), c1 - c0, words, words == 1 ? "" : "s",
	    (double)m->c[C_STORE], (double)m->c[C_LOAD]);
	if (abs_ == 0 && bad == 0)
		printf("    check (sampled, not the proof): equal to %s on %llu "
		    "points of its domain\n", f->anon ? "it" : sym_str(f->name),
		    (unsigned long long)n);
	else
		printf("    check: differs from %s, by up to %.3g (%llu points NaN "
		    "or infinite where it is not) on %llu points\n", f->anon ? "it" :
		    sym_str(f->name), abs_, (unsigned long long)bad,
		    (unsigned long long)n);
	free(b.p);
	free(w.p);
}

/*
 * find: f's parameters that one f32 can carry together, proposed for the
 * code that stores them (an interface change: f only reads them back).
 * A flag in the sign bit of a value its range keeps at +0 or above; a
 * value and a payload that are never needed at once, told apart by a
 * flag, as the value or a quiet NaN carrying the payload, where the model
 * says the target keeps that payload.  The count of packings said.
 */
static int
pack_find(struct func *f)
{
	const struct model *m = model_active();
	int x, b, p, said = 0;

	for (x = 0; x < f->np; x++) {
		const struct param *px = &f->p[x];
		if (px->ty != TY_F32 || !px->ranged)
			continue;
		/* a flag in x's sign bit */
		for (b = 0; px->rlo >= 0 && b < f->np; b++) {
			nid s, g, enc;
			int sl[2];
			nid to[2];
			char how[400];
			if (b == x || !pk_flag(&f->p[b]))
				continue;
			if (!said++)
				printf("find %s: parameters one f32 can carry together "
				    "(an interface change, proposed for the code that "
				    "stores them)\n", f->anon ? node_str(f->body) :
				    sym_str(f->name));
			s = mk_var((uint32_t)x, sym_cstr("s"), TY_F32);
			sl[0] = x;
			sl[1] = b;
			to[0] = mk_un(O_ABS, s);
			to[1] = pk_sign_flag(s, f->p[b].ty);
			g = pk_subst(f, sl, to, 2);
			/* by the bits: the default float controls may ignore a
			 * zero's sign, so a float negate need not give the -0 */
			enc = mk_un(O_ASFLOAT, mk_bin(O_OR, mk_bin(O_AND,
			    mk_un(O_ASUINT, mk_var((uint32_t)x, px->name, px->ty)),
			    pk_u(0x7fffffff)), mk_bin(O_SHL, mk_cast(PK_U32,
			    mk_var((uint32_t)b, f->p[b].name, f->p[b].ty)),
			    pk_u(31))));
			snprintf(how, sizeof how, "%s and %s as one f32, %s in the sign "
			    "bit: %s is in [%g, %g], so abs(s) is %s and the sign bit "
			    "%s, read as a bit (-0 when %s is 0, written by the bits: "
			    "a float negate need not keep a zero's sign)",
			    sym_str(px->name),
			    sym_str(f->p[b].name), sym_str(f->p[b].name),
			    sym_str(px->name), px->rlo, px->rhi, sym_str(px->name),
			    sym_str(f->p[b].name), sym_str(px->name));
			pk_report(f, g, enc, x, 1, how);
		}
		/* x or a payload p in a quiet NaN, flag b telling which */
		for (b = 0; b < f->np; b++) {
			if (b == x || f->p[b].ty != TY_BOOL ||
			    !pk_only_under(f, x, b, 1))
				continue;
			for (p = 0; p < f->np; p++) {
				const struct param *pp = &f->p[p];
				nid s, bits, g, enc, xv;
				int sl[3];
				nid to[3];
				char how[400];
				if (p == x || p == b || !TY_SIZED(pp->ty) || pp->ty ==
				    TY_BOOL || !pp->ranged || pp->lo < 0 || pp->hi >=
				    (1 << 22) || !pk_only_under(f, p, b, 0) ||
				    isinf(px->rlo) || isinf(px->rhi))
					continue;
				if (!said++)
					printf("find %s: parameters one f32 can carry "
					    "together (an interface change, proposed for the "
					    "code that stores them)\n", f->anon ?
					    node_str(f->body) : sym_str(f->name));
				snprintf(how, sizeof how, "%s, %s and %s as one f32: %s "
				    "is used only where %s holds and %s only where it "
				    "does not, %s is finite and %s in [0, 2^22), so s is "
				    "%s or a quiet NaN carrying %s, and the NaN never "
				    "reaches arithmetic", sym_str(px->name),
				    sym_str(pp->name), sym_str(f->p[b].name),
				    sym_str(px->name), sym_str(f->p[b].name),
				    sym_str(pp->name), sym_str(px->name),
				    sym_str(pp->name), sym_str(px->name),
				    sym_str(pp->name));
				if (m->par[MP_NAN_PAYLOAD] == 0) {
					printf("    %s;\n    not offered: %s does not say that "
					    "its target keeps a quiet NaN's payload through "
					    "loads, stores and moves (`nan_payload: 1` in a "
					    "model says so)\n", how, sym_str(m->name));
					continue;
				}
				s = mk_var((uint32_t)x, sym_cstr("s"), TY_F32);
				bits = mk_un(O_ASUINT, s);
				sl[0] = x;
				sl[1] = p;
				sl[2] = b;
				to[0] = s;
				to[1] = mk_cast(pp->ty, mk_bin(O_AND, bits,
				    pk_u(0x3fffff)));
				to[2] = mk_bin(O_NE, mk_bin(O_AND, bits, pk_u(0x7f800000)),
				    pk_u(0x7f800000));
				g = pk_subst(f, sl, to, 3);
				xv = mk_var((uint32_t)x, px->name, px->ty);
				/* selected as bits: by default a float instruction may
				 * assume that no operand is a NaN */
				enc = mk_un(O_ASFLOAT, mk_sel(pk_true(f, b), mk_un(O_ASUINT,
				    xv), mk_bin(O_OR, pk_u(0x7fc00000), mk_cast(PK_U32,
				    mk_var((uint32_t)p, pp->name, pp->ty)))));
				pk_report(f, g, enc, x, 2, how);
			}
		}
	}
	return said;
}

/*
 * A float function's rewrites by identities (irw.c: compare and select
 * as min/max, clamps), those the model prices lower, each with what it
 * holds under (NaN, the sign of a zero) and its derivation; then applied
 * while one is cheaper.  The count of those listed.
 */
static int
float_rewrites(struct func *f)
{
	nid cur = f->body;
	int round, shown = 0, base = icost(f->body);
	struct sbuf steps = {0};

	for (round = 0; round < 8; round++) {
		struct irw *r;
		int nr = irw_cands(f, cur, &r), i, best = -1, bc = icost(cur);
		for (i = 0; i < nr; i++) {
			int c = icost(r[i].e);
			if (!r[i].nearmiss && !r[i].quiet && c >= bc && round == 0) {
				printf("find %s: derived, but not cheaper under %s (cost %d, "
				    "as written %d): %s\n", f->anon ? node_str(f->body) :
				    sym_str(f->name), sym_str(model_active()->name), c, bc,
				    r[i].what);
				shown = shown ? shown : -1;
			}
			if (r[i].nearmiss || c >= bc)
				continue;
			if (round == 0) {
				if (shown < 0)
					shown = 0;
				if (!shown++) {
					printf("find %s: rewrites by identities the model "
					    "prices lower (cost %d as written, under %s)\n",
					    f->anon ? node_str(f->body) : sym_str(f->name),
					    base, sym_str(model_active()->name));
					rep_begin("find");
					rep_str("model", sym_str(model_active()->name));
					rep_list("rewrites");
				}
				printf("  cost %-3d %-40s %s (derived)\n", c,
				    shared_str(r[i].e, 300), r[i].what);
				irw_print(&r[i], "      ");
				rep_obj(NULL);
				rep_num("cost", c);
				rep_expr("expr", r[i].e);
				rep_str("rewrite", r[i].what);
				rep_str("holds_where", r[i].cond);
				rep_str("derivation", r[i].proof);
				rep_end();
			}
			if (best < 0 || c < icost(r[best].e))
				best = i;
		}
		if (best < 0) {
			irw_free(r, nr);
			break;
		}
		sb_printf(&steps, "%s%s", round ? ", then " : "", r[best].what);
		cur = r[best].e;
		irw_free(r, nr);
	}
	if (shown > 0) {
		rep_end();
		if (round >= 2) {
			printf("  cost %-3d %-40s all of them: %s (derived)\n",
			    icost(cur), shared_str(cur, 300), sb_str(&steps));
			rep_obj("fixpoint");
			rep_num("cost", icost(cur));
			rep_expr("expr", cur);
			rep_str("steps", sb_str(&steps));
			rep_end();
		}
	}
	free(steps.p);
	return shown;
}

void
cmd_find(struct func *f, int want, const uint8_t *opsel, int maxcost)
{
	int j, fpar = 0;

	(void)opsel;
	fuse_domain(f);		/* prices that depend on proved ranges */
	if (TY_FP(N.ty[f->body])) {
		/* its rewrites by identities, the texture unit's section (in
		 * the same report), and its f64 arithmetic over f32 (dword.c);
		 * the bit-level families are for one float parameter */
		int said = float_rewrites(f);
		if (!said)
			rep_begin("find");
		said += tex_find(f);
		said += dword_find(f);
		said += sfu_find(f);
		said += pack_find(f);
		if (N.fl[f->body] & NF_SG) {
			/* a float value per lane: its subgroup forms (#204) */
			printf("find %s: a float value per lane: the subgroup forms "
			    "of floats (derivatives), costs from %s\n", f->anon ?
			    node_str(f->body) : sym_str(f->name),
			    sym_str(model_active()->name));
			sg_find(f);
			fuse_domain(NULL);
			return;
		}
		if (said && (f->np != 1 || TYK(f->p[0].ty) != K_FLOAT)) {
			fuse_domain(NULL);
			return;
		}
		find_float(f, want);
		fuse_domain(NULL);
		return;
	}
	/* an integer result over float parameters: the search enumerates
	 * integer domains, but the rewrites by identities are derived */
	for (j = 0; j < f->np; j++)
		fpar |= TY_FP(f->p[j].ty) != 0;
	if (fpar && float_rewrites(f)) {
		fuse_domain(NULL);
		return;
	}
	if (N.fl[f->body] & NF_LANES) {
		/* a word in each lane: no points to search, and what is
		 * derived is proved by SAT over the lanes' words */
		printf("find %s: a value in each lane (lanes), so there are no "
		    "points to search; forms derived and proved by SAT, costs "
		    "from %s\n", f->anon ? node_str(f->body) : sym_str(f->name),
		    sym_str(model_active()->name));
		rep_begin("find");
		rep_str("model", sym_str(model_active()->name));
		sg_find(f);
		fuse_domain(NULL);
		return;
	}
	find_setup(f);
	fuse_domain(&F.f);
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
	dot_find(&F.f);
	sg_find(&F.f);
	inst_find(&F.f);
	swar_find(&F.f);
	tab_find(&F.f);
	tex_find(&F.f);
	rep_list("implementations");
	run_levels(want, maxcost > 0 ? maxcost : 12);
	rep_end();
	fuse_domain(NULL);
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
	fuse_domain(&F.f);
	run_levels(want, F.done + 8);
	rep_end();
	fuse_domain(NULL);
}

/*
 * For programs (cfront's bridge): the forms find would list for f, not
 * proved (the caller proves what it takes): the closed forms, the
 * peephole drops, the per-case fits and the enumerated terms up to
 * maxcost or the seconds.  Nothing is printed; `more` does not continue
 * it.  Returns the count; fail()s like find on bad input.
 */
int
find_cands(struct func *f, int maxcost, double seconds, nid *out,
    char (*why)[96], int max)
{
	clock_t t0 = clock();
	int c, n = 0, i;

	jmp_buf saved;

	if (TY_FP(N.ty[f->body]))
		return 0;
	fquiet = 1;
	note_mute++;
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) != 0) {
		memcpy(fail_jmp, saved, sizeof saved);
		note_mute--;
		fquiet = 0;
		notes_drop();
		F.active = 0;
		return 0;
	}
	find_setup(f);
	peephole(&F.f);
	parametric(&F.f);
	for (c = F.done + 1; c <= maxcost && c < 62 && !F.full; c++) {
		if ((double)(clock() - t0) / CLOCKS_PER_SEC > seconds)
			break;
		enumerate_level(c);
	}
	memcpy(fail_jmp, saved, sizeof saved);
	note_mute--;
	fquiet = 0;
	qsort(F.m, (size_t)F.nm, sizeof *F.m, cmp_match);
	for (i = 0; i < F.nm && n < max; i++) {
		out[n] = F.m[i].e;
		snprintf(why[n], sizeof why[n], "%s", F.m[i].why ? F.m[i].why : "");
		n++;
	}
	F.active = 0;
	return n;
}
