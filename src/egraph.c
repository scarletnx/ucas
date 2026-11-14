/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Equality saturation over a function's integer arithmetic (`find F
 * saturate`).
 *
 * An e-graph holds the function's body as classes of equal terms: an
 * e-node is an operation on classes (add, sub, mul, and, or, xor, shl,
 * neg, not), or a leaf kept whole (a constant, a parameter, anything
 * else).  Every rule is applied to every match in every class, both ways,
 * the other side added and its class merged with the matched one; after
 * each round the classes are closed under congruence (two nodes with the
 * same operation on the same classes are one class).  It stops when a
 * round adds nothing (saturated), or at the node limit, or after the
 * rounds.  The rules: a written set (commutativity, associativity,
 * distributivity of * over + and -, a shift by a constant as a multiply,
 * negation, subtraction and not, de Morgan, x + x, x - x, x xor x, x & x,
 * x | x) and the rules `rules infer` kept, at their width.  Constants
 * fold where both operands of a node are constants (the classes'
 * constant analysis).  The cheapest term of the root's class is then
 * extracted under the model's prices (a tree a class, so a subterm used
 * twice is priced twice there), priced again by the cover with its
 * fusions, and offered when that is below the function as written,
 * checked on the domain like every match.  What pays only through a
 * dearer form on the way is found this way: the dearer form is a member
 * of a class, not a step that had to pay.
 */
#include "calc.h"

#include <stdlib.h>
#include <string.h>

#define EG_NODES 10000
#define EG_ROUNDS 10
#define EG_MATCHES 2048
#define EG_INF 1e30f

struct en {
	int op;			/* O_* of an operation; O_NONE a leaf */
	ty_t ty;
	uint32_t a, b;		/* classes; NOTOK: none */
	nid leaf;
};

static struct {
	struct en *n;
	uint32_t nn, cap;
	uint32_t *cls;		/* node -> its class when added */
	uint32_t *par;		/* classes: union-find */
	uint32_t nc;
	uint32_t *ht, htcap;
	uint32_t *head, *next;	/* each class's nodes, rebuilt each round */
	int full;
	uint32_t merges;
} G;

static void
eg_free(void)
{
	free(G.n);
	free(G.cls);
	free(G.par);
	free(G.ht);
	free(G.head);
	free(G.next);
	memset(&G, 0, sizeof G);
}

static uint32_t
eg_find(uint32_t c)
{
	while (G.par[c] != c) {
		G.par[c] = G.par[G.par[c]];
		c = G.par[c];
	}
	return c;
}

static int
eg_union(uint32_t a, uint32_t b)
{
	a = eg_find(a);
	b = eg_find(b);
	if (a == b)
		return 0;
	if (a < b)
		G.par[b] = a;
	else
		G.par[a] = b;
	G.merges++;
	return 1;
}

static uint64_t
eg_key(int op, ty_t ty, uint32_t a, uint32_t b, nid leaf)
{
	uint64_t h = 1469598103934665603ULL;

	h = (h ^ (uint64_t)op) * 1099511628211ULL;
	h = (h ^ (uint64_t)ty) * 1099511628211ULL;
	h = (h ^ (uint64_t)a) * 1099511628211ULL;
	h = (h ^ (uint64_t)b) * 1099511628211ULL;
	h = (h ^ (uint64_t)leaf) * 1099511628211ULL;
	return h;
}

static uint32_t
eg_canon_b(uint32_t b)
{
	return b == NOTOK ? NOTOK : eg_find(b);
}

/* the node with this operation on these classes, NOTOK when none */
static uint32_t
eg_lookup(int op, ty_t ty, uint32_t a, uint32_t b, nid leaf, uint32_t *slot)
{
	uint32_t j = (uint32_t)eg_key(op, ty, a, b, leaf) & (G.htcap - 1), i;

	while ((i = G.ht[j]) != 0) {
		const struct en *e = &G.n[i - 1];
		if (e->op == op && e->ty == ty && e->leaf == leaf &&
		    eg_canon_b(e->a) == a && eg_canon_b(e->b) == b) {
			*slot = j;
			return i - 1;
		}
		j = (j + 1) & (G.htcap - 1);
	}
	*slot = j;
	return NOTOK;
}

/* a constant's node in class c: its value */
static int
eg_const(uint32_t c, int64_t *v)
{
	uint32_t i;

	c = eg_find(c);
	for (i = G.head[c]; i != NOTOK; i = G.next[i])
		if (G.n[i].op == O_NONE && N.op[G.n[i].leaf] == O_CONST) {
			*v = N.ki[G.n[i].leaf];
			return 1;
		}
	return 0;
}

static uint32_t eg_add(int op, ty_t ty, uint32_t a, uint32_t b, nid leaf);

static uint32_t
eg_leaf(nid t)
{
	return eg_add(O_NONE, N.ty[t], NOTOK, NOTOK, t);
}

/* the class of a node, added when new; NOTOK when the graph is full */
static uint32_t
eg_add(int op, ty_t ty, uint32_t a, uint32_t b, nid leaf)
{
	uint32_t slot, i, c;

	if (a != NOTOK)
		a = eg_find(a);
	if (b != NOTOK)
		b = eg_find(b);
	if ((i = eg_lookup(op, ty, a, b, leaf, &slot)) != NOTOK)
		return eg_find(G.cls[i]);
	if (G.nn >= EG_NODES) {
		G.full = 1;
		return NOTOK;
	}
	if (G.nn == G.cap) {
		G.cap = G.cap ? G.cap * 2 : 1024;
		G.n = xrealloc(G.n, G.cap * sizeof *G.n);
		G.cls = xrealloc(G.cls, G.cap * sizeof *G.cls);
		G.par = xrealloc(G.par, G.cap * sizeof *G.par);
		G.next = xrealloc(G.next, G.cap * sizeof *G.next);
		G.head = xrealloc(G.head, G.cap * sizeof *G.head);
	}
	i = G.nn++;
	G.n[i].op = op;
	G.n[i].ty = ty;
	G.n[i].a = a;
	G.n[i].b = b;
	G.n[i].leaf = leaf;
	c = G.nc++;
	G.par[c] = c;
	G.cls[i] = c;
	G.head[c] = i;
	G.next[i] = NOTOK;
	G.ht[slot] = i + 1;
	/* constants fold */
	if (op != O_NONE && TY_ISINT(ty) && TY_SIZED(ty)) {
		int64_t x, y = 0, v;
		uint8_t fl = 0;
		struct ity it;
		if (eg_const(a, &x) && (b == NOTOK || eg_const(b, &y))) {
			ity_of(&it, ty, 0);
			v = scalar_int(op, &it, &it, x, b == NOTOK ? x : y, 0, &fl);
			if (!(fl & (F_UNDEF | F_UB))) {
				uint32_t k = eg_leaf(mk_const_i(ty, int_norm(ty, v)));
				if (k != NOTOK)
					eg_union(c, k);
			}
		}
	}
	return eg_find(c);
}

static int
eg_op(int op)
{
	return op == O_ADD || op == O_SUB || op == O_MUL || op == O_AND ||
	    op == O_OR || op == O_XOR || op == O_SHL || op == O_NEG ||
	    op == O_NOT;
}

static uint32_t
eg_term(nid t)
{
	uint32_t a, b = NOTOK;

	if (!eg_op(N.op[t]) || !TY_ISINT(N.ty[t]) || !TY_SIZED(N.ty[t]) ||
	    N.ty[t] == TY_BOOL)
		return eg_leaf(t);
	if ((a = eg_term(N.a[t])) == NOTOK)
		return NOTOK;
	if (opinfo[N.op[t]].arity > 1 && (b = eg_term(N.b[t])) == NOTOK)
		return NOTOK;
	return eg_add(N.op[t], N.ty[t], a, b, 0);
}

/* congruence: classes whose nodes are one node merge, until none do;
 * then each class's list of nodes */
static void
eg_rebuild(void)
{
	uint32_t i, slot, j;
	int changed = 1;

	while (changed) {
		changed = 0;
		memset(G.ht, 0, G.htcap * sizeof *G.ht);
		for (i = 0; i < G.nn; i++) {
			struct en *e = &G.n[i];
			uint32_t a = eg_canon_b(e->a), b = eg_canon_b(e->b);
			if ((j = eg_lookup(e->op, e->ty, a, b, e->leaf, &slot)) !=
			    NOTOK) {
				changed |= eg_union(G.cls[i], G.cls[j]);
				continue;
			}
			G.ht[slot] = i + 1;
		}
	}
	for (i = 0; i < G.nc; i++)
		G.head[i] = NOTOK;
	for (i = G.nn; i-- > 0;) {
		uint32_t c = eg_find(G.cls[i]);
		G.next[i] = G.head[c];
		G.head[c] = i;
	}
}

/* ---- rules ---- */

struct erule {
	nid l, r;		/* patterns over the variables pv */
	ty_t ty;		/* 0: any integer type; else only that one */
	nid pv[3];
};

#define EG_MAXRULES (64 + 512)
static struct erule rules[EG_MAXRULES];
static int nrules;
static nid pvar[3];
static const nid *curpv;	/* the variables of the rule being applied */

/* a binding: the classes of the pattern variables, and the type of the
 * matched root (what a written rule's constants and operations take) */
struct ebind {
	uint32_t v[3];
	ty_t ty;
};

static int
pv_index(nid p)
{
	int k;

	for (k = 0; k < 3; k++)
		if (curpv[k] && p == curpv[k])
			return k;
	return -1;
}

/* pattern p reads variable v */
static int
pv_used(nid p, int v)
{
	int k = pv_index(p);

	if (k >= 0)
		return k == v;
	if (N.op[p] == O_CONST || N.op[p] == O_VAR)
		return 0;
	return (N.a[p] && pv_used(N.a[p], v)) || (N.b[p] && pv_used(N.b[p], v));
}

/* every way pattern p matches class c under b, appended to out */
static int
ematch(nid p, uint32_t c, const struct ebind *b, int anyty,
    struct ebind *out, int n, int max)
{
	int k = pv_index(p);
	uint32_t i;

	c = eg_find(c);
	if (k >= 0) {
		if (b->v[k] != NOTOK && eg_find(b->v[k]) != c)
			return n;
		if (n < max) {
			out[n] = *b;
			out[n].v[k] = c;
			n++;
		}
		return n;
	}
	if (N.op[p] == O_CONST) {
		int64_t v;
		if (eg_const(c, &v) && v == int_norm(b->ty ? b->ty : N.ty[p],
		    N.ki[p]) && n < max)
			out[n++] = *b;
		return n;
	}
	for (i = G.head[c]; i != NOTOK && n < max; i = G.next[i]) {
		const struct en *e = &G.n[i];
		struct ebind mid[EG_MATCHES / 4];
		int nm, q;
		if (e->op != N.op[p] || (!anyty && e->ty != N.ty[p]))
			continue;
		if (b->ty && e->ty != b->ty)
			continue;
		{
			struct ebind b0 = *b;
			b0.ty = e->ty;
			nm = ematch(N.a[p], e->a, &b0, anyty, mid, 0,
			    EG_MATCHES / 4);
		}
		for (q = 0; q < nm && n < max; q++) {
			if (opinfo[N.op[p]].arity > 1)
				n = ematch(N.b[p], e->b, &mid[q], anyty, out, n, max);
			else
				out[n++] = mid[q];
		}
	}
	return n;
}

/* pattern p added under b: its class, NOTOK when full */
static uint32_t
einst(nid p, const struct ebind *b)
{
	int k = pv_index(p);
	uint32_t a, c = NOTOK;

	if (k >= 0)
		return b->v[k];
	if (N.op[p] == O_CONST)
		return eg_leaf(mk_const_i(b->ty, int_norm(b->ty, N.ki[p])));
	if ((a = einst(N.a[p], b)) == NOTOK)
		return NOTOK;
	if (opinfo[N.op[p]].arity > 1 && (c = einst(N.b[p], b)) == NOTOK)
		return NOTOK;
	return eg_add(N.op[p], b->ty, a, c, 0);
}

static void
rule_add(nid l, nid r, ty_t ty, const nid *pv)
{
	if (nrules < EG_MAXRULES) {
		rules[nrules].l = l;
		rules[nrules].r = r;
		rules[nrules].ty = ty;
		memcpy(rules[nrules].pv, pv, sizeof rules[nrules].pv);
		nrules++;
	}
}

/* a pattern node as written: mk_bin would fold a - a, a & a, a + 0 ... */
static nid
P2(int op, nid a, nid b)
{
	return mk_raw(op, TY(K_UINT, 32), a, b, 0, 0, 0);
}

static nid
P1(int op, nid a)
{
	return mk_raw(op, TY(K_UINT, 32), a, 0, 0, 0, 0);
}

/* the written rules, over u32 variables a, b, c (any integer type) */
static void
written_rules(void)
{
	ty_t u = TY(K_UINT, 32);
	nid a = pvar[0], b = pvar[1], c = pvar[2];
	nid k0 = mk_const_i(u, 0), k1 = mk_const_i(u, 1), k2 = mk_const_i(u, 2),
	    km1 = mk_const_i(u, -1);

	rule_add(P2(O_ADD, a, b), P2(O_ADD, b, a), 0, pvar);
	rule_add(P2(O_MUL, a, b), P2(O_MUL, b, a), 0, pvar);
	rule_add(P2(O_AND, a, b), P2(O_AND, b, a), 0, pvar);
	rule_add(P2(O_OR, a, b), P2(O_OR, b, a), 0, pvar);
	rule_add(P2(O_XOR, a, b), P2(O_XOR, b, a), 0, pvar);
	rule_add(P2(O_ADD, P2(O_ADD, a, b), c), P2(O_ADD, a, P2(O_ADD, b, c)),
	    0, pvar);
	rule_add(P2(O_MUL, P2(O_MUL, a, b), c), P2(O_MUL, a, P2(O_MUL, b, c)),
	    0, pvar);
	rule_add(P2(O_AND, P2(O_AND, a, b), c), P2(O_AND, a, P2(O_AND, b, c)),
	    0, pvar);
	rule_add(P2(O_OR, P2(O_OR, a, b), c), P2(O_OR, a, P2(O_OR, b, c)), 0, pvar);
	rule_add(P2(O_XOR, P2(O_XOR, a, b), c), P2(O_XOR, a, P2(O_XOR, b, c)),
	    0, pvar);
	rule_add(P2(O_MUL, a, P2(O_ADD, b, c)), P2(O_ADD, P2(O_MUL, a, b),
	    P2(O_MUL, a, c)), 0, pvar);
	rule_add(P2(O_MUL, a, P2(O_SUB, b, c)), P2(O_SUB, P2(O_MUL, a, b),
	    P2(O_MUL, a, c)), 0, pvar);
	rule_add(P2(O_AND, a, P2(O_OR, b, c)), P2(O_OR, P2(O_AND, a, b),
	    P2(O_AND, a, c)), 0, pvar);
	rule_add(P2(O_SUB, a, b), P2(O_ADD, a, P1(O_NEG, b)), 0, pvar);
	rule_add(P1(O_NEG, a), P2(O_SUB, k0, a), 0, pvar);
	rule_add(P1(O_NOT, a), P2(O_SUB, km1, a), 0, pvar);
	rule_add(P1(O_NOT, P2(O_AND, a, b)), P2(O_OR, P1(O_NOT, a), P1(O_NOT,
	    b)), 0, pvar);
	rule_add(P1(O_NOT, P2(O_OR, a, b)), P2(O_AND, P1(O_NOT, a), P1(O_NOT,
	    b)), 0, pvar);
	rule_add(P2(O_ADD, a, a), P2(O_MUL, a, k2), 0, pvar);
	rule_add(P2(O_SHL, a, k1), P2(O_MUL, a, k2), 0, pvar);
	rule_add(P2(O_MUL, a, k1), a, 0, pvar);
	rule_add(P2(O_ADD, a, k0), a, 0, pvar);
	rule_add(P2(O_SUB, a, a), k0, 0, pvar);
	rule_add(P2(O_XOR, a, a), k0, 0, pvar);
	rule_add(P2(O_AND, a, a), a, 0, pvar);
	rule_add(P2(O_OR, a, a), a, 0, pvar);
}

/* a shift by a constant k and a multiply by 2^k are one class */
static void
shift_mul(void)
{
	uint32_t i, n = G.nn;

	for (i = 0; i < n && !G.full; i++) {
		const struct en *e = &G.n[i];
		int64_t k;
		uint32_t to;
		if (e->op != O_SHL || !eg_const(e->b, &k) || k < 1 ||
		    k >= TYW(e->ty))
			continue;
		to = eg_add(O_MUL, e->ty, e->a, eg_leaf(mk_const_i(e->ty,
		    int_norm(e->ty, (int64_t)1 << k))), 0);
		if (to != NOTOK)
			eg_union(G.cls[i], to);
	}
}

/* ---- extraction ---- */

static float
eg_opcost(const struct model *m, int op)
{
	switch (op) {
	case O_ADD: case O_SUB: case O_NEG: return m->c[C_IADD];
	case O_MUL: return m->c[C_IMUL];
	case O_SHL: return m->c[C_SHIFT];
	}
	return m->c[C_LOGIC];
}

static nid
eg_build(uint32_t c, const uint32_t *choice, int depth)
{
	const struct en *e;

	c = eg_find(c);
	if (choice[c] == NOTOK || depth > 64)
		return 0;
	e = &G.n[choice[c]];
	if (e->op == O_NONE)
		return e->leaf;
	{
		nid a = eg_build(e->a, choice, depth + 1), b = 0;
		if (!a)
			return 0;
		if (opinfo[e->op].arity > 1 && (b = eg_build(e->b, choice,
		    depth + 1)) == 0)
			return 0;
		if (N.ty[a] != e->ty)
			a = mk_cast(e->ty, a);
		if (b && N.ty[b] != e->ty)
			b = mk_cast(e->ty, b);
		return opinfo[e->op].arity > 1 ? mk_bin(e->op, a, b) :
		    mk_un(e->op, a);
	}
}

/*
 * The cheapest member of root's class after saturation, 0 when the
 * graph has none (or the body has no arithmetic to saturate).  *stat:
 * what was done, a line.
 */
nid
egraph_best(nid root, const struct rwrule *rr, int nr, char *stat,
    size_t nstat)
{
	const struct model *m = model_active();
	uint32_t rc, i, *choice;
	float *cost;
	int round, k, saturated = 0, changed;
	nid best;

	eg_free();
	G.htcap = 1 << 15;
	G.ht = xcalloc(G.htcap, sizeof *G.ht);
	nrules = 0;
	pvar[0] = mk_var(0, sym_cstr("a"), TY(K_UINT, 32));
	pvar[1] = mk_var(1, sym_cstr("b"), TY(K_UINT, 32));
	pvar[2] = mk_var(2, sym_cstr("c"), TY(K_UINT, 32));
	written_rules();
	for (k = 0; k < nr; k++) {
		nid pv[3] = {0, 0, 0};
		int j;
		for (j = 0; j < rr[k].nin && j < 3; j++)
			pv[j] = rr[k].pv[j];
		rule_add(rr[k].l, rr[k].r, rr[k].ty, pv);
	}
	if ((rc = eg_term(root)) == NOTOK) {
		snprintf(stat, nstat, "the body alone fills the graph");
		eg_free();
		return 0;
	}
	eg_rebuild();
	for (round = 0; round < EG_ROUNDS && !G.full; round++) {
		uint32_t before = G.nn, merges = G.merges, c, nc = G.nc;
		for (k = 0; k < nrules && !G.full; k++) {
			int d;
			curpv = rules[k].pv;
			for (d = 0; d < 2 && !G.full; d++) {
				nid from = d ? rules[k].r : rules[k].l;
				nid to = d ? rules[k].l : rules[k].r;
				for (c = 0; c < nc && !G.full; c++) {
					struct ebind b0, res[64];
					int nm, q;
					if (eg_find(c) != c)
						continue;
					/* a side that is a variable matches everything */
					if (pv_index(from) >= 0 || N.op[from] == O_CONST)
						continue;
					b0.v[0] = b0.v[1] = b0.v[2] = NOTOK;
					b0.ty = rules[k].ty;
					nm = ematch(from, c, &b0, rules[k].ty == 0, res, 0,
					    64);
					for (q = 0; q < nm && !G.full; q++) {
						uint32_t t;
						int v, ok = 1;
						/* every variable of the other side bound */
						for (v = 0; v < 3; v++)
							if (res[q].v[v] == NOTOK &&
							    pv_used(to, v))
								ok = 0;
						if (!ok)
							continue;
						if ((t = einst(to, &res[q])) != NOTOK)
							eg_union(c, t);
					}
				}
			}
		}
		shift_mul();
		eg_rebuild();
		if (G.nn == before && G.merges == merges) {
			saturated = 1;
			break;
		}
	}
	/* the cheapest tree of each class */
	cost = xmalloc(G.nc * sizeof *cost);
	choice = xmalloc(G.nc * sizeof *choice);
	for (i = 0; i < G.nc; i++) {
		cost[i] = EG_INF;
		choice[i] = NOTOK;
	}
	changed = 1;
	for (k = 0; k < 200 && changed; k++) {
		changed = 0;
		for (i = 0; i < G.nn; i++) {
			const struct en *e = &G.n[i];
			uint32_t c = eg_find(G.cls[i]);
			float v;
			if (e->op == O_NONE)
				v = 0;
			else {
				float ca = cost[eg_find(e->a)], cb = e->b == NOTOK ? 0 :
				    cost[eg_find(e->b)];
				if (ca >= EG_INF || cb >= EG_INF)
					continue;
				v = eg_opcost(m, e->op) + ca + cb;
			}
			if (v < cost[c] - 1e-6f) {
				cost[c] = v;
				choice[c] = i;
				changed = 1;
			}
		}
	}
	best = eg_build(rc, choice, 0);
	snprintf(stat, nstat, "%u classes, %u nodes, %s after %d round%s, %d "
	    "rules (%d kept for find)", (unsigned)(G.nc - G.merges), G.nn, saturated ?
	    "saturated" : G.full ? "stopped at the node limit" :
	    "stopped at the round limit", round + (saturated ? 1 : 0),
	    round + (saturated ? 1 : 0) == 1 ? "" : "s", nrules, nr);
	free(cost);
	free(choice);
	eg_free();
	return best;
}
