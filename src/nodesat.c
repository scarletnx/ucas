/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Bit-precise proofs about expression DAGs: a node DAG on machine
 * integers is bit-blasted into the and-inverter graph of sat.c, with the
 * evaluator's exact semantics (eval.c's scalar_int), and a miter is
 * handed to the CDCL solver.
 *
 * Every node becomes a word (its value in its type's width) and a literal
 * "defined".  Undefined is what the evaluator marks F_UNDEF: a shift by a
 * negative amount or by the width or more, a division or remainder by
 * zero, a table index out of range, `int` (64-bit ideal) arithmetic that
 * overflows, an undefined operand.  ?:, && and || are lazy, as in the
 * evaluator: only the taken arm matters.  Machine integers wrap; INT_MIN /
 * -1 in iN wraps to INT_MIN (remainder 0), as the evaluator computes it.
 *
 * Loops (O_LOOP) with constant bounds are unrolled: each iteration's body
 * is blasted in a context binding the index and the states; subterms that
 * do not read the loop's variables come from the enclosing context.
 *
 * nsat_equiv decides "b refines a on f's domain": for every input within
 * the parameters' ranges where the where clause holds and a is defined, b
 * is defined and equal.  UNSAT is a proof; SAT gives a counterexample,
 * which the caller re-checks with the evaluator.
 */
#include "calc.h"
#include "sat.h"
#include "nodesat.h"

#include <stdlib.h>
#include <string.h>

#define UNROLL 256          /* most iterations of one loop unrolled */
#define MAXAIG (4 << 20)    /* AIG nodes before giving up */

struct bv {
	struct word v;
	lit d;
};

struct nctx {
	struct nctx *up;
	uint64_t bound;           /* slots bound in this context */
	struct word env[NSLOT];   /* 64-bit values of the bound slots */
	nid *hk;                  /* memo: node -> value index */
	uint32_t *hv;
	uint32_t hcap, hn;
};

static struct {
	struct aig *g;
	struct bv *v;
	uint32_t n, cap;
	const char *why;          /* unsupported: what */
	struct word par[MAXP];    /* parameters, 64-bit */
	int npar;
} B;

static void
ctx_init(struct nctx *c, struct nctx *up, uint64_t bound)
{
	memset(c, 0, sizeof *c);
	c->up = up;
	c->bound = bound;
	c->hcap = 256;
	c->hk = xcalloc(c->hcap, sizeof *c->hk);
	c->hv = xcalloc(c->hcap, sizeof *c->hv);
}

static void
ctx_free(struct nctx *c)
{
	free(c->hk);
	free(c->hv);
}

static uint32_t *
ctx_slot(struct nctx *c, nid n)
{
	uint32_t j;

	if ((c->hn + 1) * 2 > c->hcap) {
		nid *ok = c->hk;
		uint32_t *ov = c->hv, oc = c->hcap, i;
		c->hcap *= 2;
		c->hk = xcalloc(c->hcap, sizeof *c->hk);
		c->hv = xcalloc(c->hcap, sizeof *c->hv);
		for (i = 0; i < oc; i++)
			if (ok[i]) {
				j = (ok[i] * 2654435761u) & (c->hcap - 1);
				while (c->hk[j])
					j = (j + 1) & (c->hcap - 1);
				c->hk[j] = ok[i];
				c->hv[j] = ov[i];
			}
		free(ok);
		free(ov);
	}
	j = (n * 2654435761u) & (c->hcap - 1);
	while (c->hk[j] && c->hk[j] != n)
		j = (j + 1) & (c->hcap - 1);
	if (!c->hk[j]) {
		c->hk[j] = n;
		c->hv[j] = NOTOK;
		c->hn++;
	}
	return &c->hv[j];
}

static uint32_t
newval(void)
{
	if (B.n == B.cap) {
		B.cap = newcap(B.cap, B.n + 1);
		B.v = xrealloc(B.v, B.cap * sizeof *B.v);
	}
	memset(&B.v[B.n], 0, sizeof B.v[B.n]);
	return B.n++;
}

static int
unsup(const char *why)
{
	if (!B.why)
		B.why = why;
	return 0;
}

/* the width of a type in bits, 0 if it is not a machine integer */
static int
bw(ty_t t)
{
	switch (TYK(t)) {
	case K_BOOL: return 1;
	case K_UINT:
	case K_SINT: return TYW(t);
	case K_INT:
	case K_LIT: return 64;
	}
	return 0;
}

static int
sgn(ty_t t)
{
	return TYK(t) == K_SINT || TYK(t) == K_INT || TYK(t) == K_LIT;
}

static int
ideal(ty_t t)
{
	return TYK(t) == K_INT || TYK(t) == K_LIT;
}

/* the 64-bit value of a word of type t (sign- or zero-extended) */
static void
ext64(struct word *r, const struct word *a, ty_t t)
{
	w_resize(r, a, 64, sgn(t) && a->w < 64);
}

static lit
nz(const struct word *a)
{
	lit r = L_FALSE;
	int i;

	for (i = 0; i < a->w; i++)
		r = aig_or(B.g, r, a->b[i]);
	return r;
}

static lit
msb(const struct word *a)
{
	return a->b[a->w - 1];
}

/* the shift or index amount a (of type t) is in [0, lim) */
static lit
below(const struct word *a, ty_t t, uint64_t lim)
{
	struct word x, k;
	lit r;

	ext64(&x, a, t);
	w_const(&k, 64, lim);
	r = w_ult(B.g, &x, &k);
	if (sgn(t))
		r = aig_and(B.g, r, L_NOT(msb(&x)));
	return r;
}

static uint32_t blast(struct nctx *c, nid n);

static uint32_t blast_loop(struct nctx *c, nid n);

/*
 * A packed dot product (dot.c): lane k of a 32-bit word is bits
 * [k*bits, (k+1)*bits), zero- or sign-extended.  Without an accumulator
 * the result is the sum's low 32 bits, so each product and the sum are
 * built in 32 bits (the same circuit as a 32-bit a*b + c, which lets
 * the AIG's hashing share it).  With one, the sum is exact in
 * 2*bits + 4 bits, then acc is added in 64 bits and the total clamped
 * to the result's range.
 */
static int
nconst(const struct word *x)
{
	int i, n = 0;

	for (i = 0; i < x->w; i++)
		n += x->b[i] == L_FALSE || x->b[i] == L_TRUE;
	return n;
}

static void
blast_dot(int op, struct word *r, const struct word *a,
    const struct word *b, const struct word *acc)
{
	const struct dotform *df = dot_form(op);
	struct aig *g = B.g;
	int E = acc ? 2 * df->bits + 4 : 32, k, i, sres = df->sa || df->sb;
	struct word la, lb, sum, p, x, y;

	w_const(&sum, E, 0);
	for (k = 0; k < df->lanes; k++) {
		la.w = lb.w = df->bits;
		for (i = 0; i < df->bits; i++) {
			la.b[i] = a->b[k * df->bits + i];
			lb.b[i] = b->b[k * df->bits + i];
		}
		w_resize(&x, &la, E, df->sa);
		w_resize(&y, &lb, E, df->sb);
		/* the more constant operand first, as the node constructors
		 * put a constant factor first: the same circuit as K * x */
		if (nconst(&y) > nconst(&x))
			w_mul(g, &p, &y, &x);
		else
			w_mul(g, &p, &x, &y);
		w_add(g, &sum, &sum, &p);
	}
	if (!acc) {
		*r = sum;
		return;
	}
	{
		struct word s64, a64, t, hi, lo;
		lit over, under;
		w_resize(&s64, &sum, 64, 1);
		w_resize(&a64, acc, 64, sres);
		w_add(g, &t, &s64, &a64);
		w_const(&hi, 64, sres ? 0x7fffffffULL : 0xffffffffULL);
		w_const(&lo, 64, sres ? (uint64_t)-0x80000000LL : 0);
		over = w_slt(g, &hi, &t);
		under = w_slt(g, &t, &lo);
		w_mux(g, &t, over, &hi, &t);
		w_mux(g, &t, under, &lo, &t);
		w_resize(r, &t, 32, 0);
	}
}

#define V(i) (&B.v[i].v)
#define D(i) (B.v[i].d)

static uint32_t
blast1(struct nctx *c, nid n)
{
	int op = N.op[n], w = bw(N.ty[n]);
	ty_t t = N.ty[n], ta = N.a[n] ? N.ty[N.a[n]] : 0;
	struct word r, x, y, z;
	lit d = L_TRUE;
	uint32_t a = 0, b = 0, k, res;
	struct aig *g = B.g;

	if (op != O_TABLE && w == 0) {
		unsup("a float or real value");
		w = 1;
	}
	w_const(&r, w ? w : 1, 0);
	switch (op) {
	case O_CONST:
		w_const(&r, w, (uint64_t)N.ki[n]);
		break;
	case O_UNDEF:
		d = L_FALSE;
		break;
	case O_VAR: {
		int s = (int)N.ki[n];
		struct nctx *q = c;
		while (q && !((q->bound >> s) & 1))
			q = q->up;
		if (q)
			x = q->env[s];
		else if (s < B.npar)
			x = B.par[s];
		else {
			unsup("a free variable");
			w_const(&x, 64, 0);
		}
		w_resize(&r, &x, w, 0);
		break;
	}
	case O_LOOP:
		return blast_loop(c, n);
	case O_TABLE:
		unsup("a table outside an index");
		break;
	case O_SEL:
		a = blast(c, N.a[n]);
		b = blast(c, N.b[n]);
		k = blast(c, N.c[n]);
		{
			lit s = nz(V(a));
			w_mux(g, &r, s, V(b), V(k));
			d = aig_and(g, D(a), aig_mux(g, s, D(b), D(k)));
		}
		break;
	case O_LAND:
	case O_LOR:
		a = blast(c, N.a[n]);
		b = blast(c, N.b[n]);
		{
			lit av = nz(V(a)), bv = nz(V(b));
			if (op == O_LAND) {
				r.b[0] = aig_and(g, av, bv);
				d = aig_and(g, D(a), aig_or(g, L_NOT(av), D(b)));
			} else {
				r.b[0] = aig_or(g, av, bv);
				d = aig_and(g, D(a), aig_or(g, av, D(b)));
			}
		}
		break;
	case O_INDEX: {
		nid tab = N.a[n];
		uint32_t cnt = N.b[tab], j;
		b = blast(c, N.b[n]);
		ext64(&x, V(b), N.ty[N.b[n]]);
		d = aig_and(g, D(b), below(V(b), N.ty[N.b[n]], cnt));
		{
			lit de = L_FALSE;
			for (j = cnt; j-- > 0;) {
				uint32_t e = blast(c, N.lst[N.a[tab] + j]);
				lit hit;
				w_const(&y, 64, j);
				hit = w_eq(g, &x, &y);
				w_mux(g, &r, hit, V(e), &r);
				de = aig_mux(g, hit, D(e), de);
			}
			d = aig_and(g, d, de);
		}
		break;
	}
	default:
		if (N.a[n])
			a = blast(c, N.a[n]);
		if (N.b[n])
			b = blast(c, N.b[n]);
		if (opinfo[op].arity >= 3 && N.c[n] && !IS_DOT(op))
			unsup(opinfo[op].name);
		if (IS_DOT(op) && N.c[n]) {
			k = blast(c, N.c[n]);
			d = aig_and(g, aig_and(g, D(a), D(b)), D(k));
			blast_dot(op, &r, V(a), V(b), V(k));
			break;
		}
		if (IS_DOT(op)) {
			d = aig_and(g, D(a), D(b));
			blast_dot(op, &r, V(a), V(b), NULL);
			break;
		}
		d = a ? D(a) : L_TRUE;
		if (opinfo[op].arity >= 2 && b)
			d = aig_and(g, d, D(b));
		switch (op) {
		case O_ADD:
		case O_SUB:
			if (op == O_ADD)
				w_add(g, &r, V(a), V(b));
			else
				w_sub(g, &r, V(a), V(b));
			if (ideal(t)) {
				lit sa = msb(V(a)), sb = msb(V(b)), sr = msb(&r);
				lit same = op == O_ADD ? L_NOT(aig_xor(g, sa, sb)) :
				    aig_xor(g, sa, sb);
				d = aig_and(g, d, L_NOT(aig_and(g, same,
				    aig_xor(g, sr, sa))));
			}
			break;
		case O_NEG:
			w_neg(g, &r, V(a));
			if (ideal(t)) {
				w_const(&z, 64, 1ULL << 63);
				d = aig_and(g, d, L_NOT(w_eq(g, V(a), &z)));
			}
			break;
		case O_MUL:
			if (ideal(t))
				unsup("int multiplication (overflow)");
			w_mul(g, &r, V(a), V(b));
			break;
		case O_POW: {
			int64_t e;
			if (ideal(t) || !is_const(N.b[n]) || !TY_ISINT(N.ty[N.b[n]])) {
				unsup("a power");
				break;
			}
			e = N.ki[N.b[n]];
			if (e < 0) {
				d = L_FALSE;
				break;
			}
			w_const(&r, w, 1);
			x = *V(a);
			while (e) {
				if (e & 1)
					w_mul(g, &r, &r, &x);
				e >>= 1;
				if (e)
					w_mul(g, &x, &x, &x);
			}
			break;
		}
		case O_AND:
			w_and(g, &r, V(a), V(b));
			break;
		case O_OR:
			w_or(g, &r, V(a), V(b));
			break;
		case O_XOR:
			w_xor(g, &r, V(a), V(b));
			break;
		case O_NOT:
			w_not(g, &r, V(a));
			break;
		case O_SHL:
		case O_SHR:
			d = aig_and(g, d, below(V(b), N.ty[N.b[n]], (uint64_t)w));
			w_resize(&y, V(b), V(b)->w < 64 ? V(b)->w : 64, 0);
			if (op == O_SHL) {
				w_shl(g, &r, V(a), &y);
				if (ideal(t)) {
					w_ashr(g, &z, &r, &y);
					d = aig_and(g, d, w_eq(g, &z, V(a)));
				}
			} else if (sgn(t))
				w_ashr(g, &r, V(a), &y);
			else
				w_lshr(g, &r, V(a), &y);
			break;
		case O_DIV:
		case O_MOD:
			w_const(&z, w, 0);
			d = aig_and(g, d, L_NOT(w_eq(g, V(b), &z)));
			if (!sgn(t)) {
				w_udivrem(g, &x, &y, V(a), V(b));
				r = op == O_DIV ? x : y;
			} else {
				struct word ua, ub, na, nb, q, m;
				lit sa = msb(V(a)), sb = msb(V(b));
				w_neg(g, &na, V(a));
				w_neg(g, &nb, V(b));
				w_mux(g, &ua, sa, &na, V(a));
				w_mux(g, &ub, sb, &nb, V(b));
				w_udivrem(g, &q, &m, &ua, &ub);
				if (op == O_DIV) {
					w_neg(g, &x, &q);
					w_mux(g, &r, aig_xor(g, sa, sb), &x, &q);
				} else {
					w_neg(g, &x, &m);
					w_mux(g, &r, sa, &x, &m);
				}
				if (ideal(t)) {
					/* INT64_MIN / -1 leaves int64 */
					w_const(&x, 64, 1ULL << 63);
					w_const(&y, 64, ~0ULL);
					d = aig_and(g, d, L_NOT(aig_and(g,
					    w_eq(g, V(a), &x), w_eq(g, V(b), &y))));
				}
			}
			break;
		case O_MIN:
		case O_MAX: {
			lit lt = op == O_MIN ? (sgn(t) ? w_slt(g, V(b), V(a)) :
			    w_ult(g, V(b), V(a))) : (sgn(t) ? w_slt(g, V(a), V(b)) :
			    w_ult(g, V(a), V(b)));
			w_mux(g, &r, lt, V(b), V(a));
			break;
		}
		case O_ABS:
			if (!sgn(t)) {
				r = *V(a);
				break;
			}
			w_neg(g, &x, V(a));
			w_mux(g, &r, msb(V(a)), &x, V(a));
			if (ideal(t)) {
				w_const(&z, 64, 1ULL << 63);
				d = aig_and(g, d, L_NOT(w_eq(g, V(a), &z)));
			}
			break;
		case O_SIGN: {
			lit nzv = nz(V(a));
			w_const(&x, w, 1);
			w_const(&y, w, ~0ULL);
			w_const(&z, w, 0);
			w_mux(g, &r, nzv, &x, &z);
			if (sgn(t))
				w_mux(g, &r, msb(V(a)), &y, &r);
			break;
		}
		case O_POPCNT:
			w_popcount(g, &r, V(a));
			break;
		case O_CLZ:
		case O_CTZ: {
			int i;
			w_const(&r, w, (uint64_t)w);
			for (i = 0; i < w; i++) {
				/* clz: the highest set bit wins, so go up;
				 * ctz: the lowest wins, so go down */
				int bit = op == O_CLZ ? i : w - 1 - i;
				w_const(&x, w, (uint64_t)(op == O_CLZ ? w - 1 - bit :
				    bit));
				w_mux(g, &r, V(a)->b[bit], &x, &r);
			}
			break;
		}
		case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE: {
			lit v;
			int s = sgn(ta);
			if (bw(ta) == 0) {
				unsup("a float comparison");
				break;
			}
			switch (op) {
			case O_EQ: v = w_eq(g, V(a), V(b)); break;
			case O_NE: v = w_ne(g, V(a), V(b)); break;
			case O_LT: v = s ? w_slt(g, V(a), V(b)) : w_ult(g, V(a), V(b)); break;
			case O_GT: v = s ? w_slt(g, V(b), V(a)) : w_ult(g, V(b), V(a)); break;
			case O_LE: v = L_NOT(s ? w_slt(g, V(b), V(a)) :
			    w_ult(g, V(b), V(a))); break;
			default: v = L_NOT(s ? w_slt(g, V(a), V(b)) :
			    w_ult(g, V(a), V(b))); break;
			}
			r.b[0] = v;
			break;
		}
		case O_LNOT:
			r.b[0] = L_NOT(nz(V(a)));
			break;
		case O_CAST:
			if (bw(ta) == 0) {
				unsup("a float conversion");
				break;
			}
			ext64(&x, V(a), ta);
			if (ideal(t) && TYK(ta) == K_UINT && TYW(ta) == 64)
				d = aig_and(g, d, L_NOT(msb(V(a))));
			w_resize(&r, &x, w, 0);
			break;
		case O_BARRIER:
			r = *V(a);
			break;
		case O_ATOMIC:
			r = *V(a);
			break;
		default:
			unsup(opinfo[op].name);
		}
	}
	res = newval();
	B.v[res].v = r;
	B.v[res].d = d;
	return res;
}

static uint32_t
blast(struct nctx *c, nid n)
{
	uint32_t *slot, r;

	while (c->up && !(node_slots(n) & c->bound))
		c = c->up;
	slot = ctx_slot(c, n);
	if (*slot != NOTOK)
		return *slot;
	if (B.why || aig_nodes(B.g) > MAXAIG) {
		unsup("too large a circuit");
		r = newval();
		w_const(&B.v[r].v, 1, 0);
		B.v[r].d = L_TRUE;
	} else
		r = blast1(c, n);
	*ctx_slot(c, n) = r;
	return r;
}

static uint32_t
blast_loop(struct nctx *c, nid n)
{
	uint32_t e = loop_ent(n), s = loops.st[e], res, k, m = loops.nst[e];
	int64_t lo, hi, trip, it;
	struct word st[MAXST];
	lit d = L_TRUE;
	struct nctx cc;
	int w = bw(N.ty[n]);

	res = newval();
	w_const(&B.v[res].v, w ? w : 1, 0);
	B.v[res].d = L_TRUE;
	if (loops.kind[e] != LK_FOR) {
		unsup("a while loop");
		return res;
	}
	if (!int_of(N.a[n], &lo) || !int_of(N.b[n], &hi)) {
		unsup("a loop with variable bounds");
		return res;
	}
	trip = hi - lo + 1;
	if (trip < 0) {
		if (loops.flags[e] & LF_NEGUNDEF)
			d = L_FALSE;
		trip = 0;
	}
	if (trip > UNROLL) {
		unsup("a loop of more than 256 iterations");
		return res;
	}
	for (k = 0; k < m; k++)
		w_const(&st[k], 64, 0);
	/* the initial values see the index at lo and the states at 0 */
	ctx_init(&cc, c, loops.bound[e]);
	w_const(&cc.env[loops.islot[e]], 64, (uint64_t)lo);
	for (k = 0; k < m; k++)
		cc.env[loops.sslot[s + k]] = st[k];
	for (k = 0; k < m; k++) {
		uint32_t v = blast(&cc, loops.init[s + k]);
		d = aig_and(B.g, d, D(v));
		ext64(&st[k], V(v), N.ty[loops.init[s + k]]);
	}
	ctx_free(&cc);
	for (it = 0; it < trip && !B.why; it++) {
		struct word nx[MAXST];
		ctx_init(&cc, c, loops.bound[e]);
		w_const(&cc.env[loops.islot[e]], 64, (uint64_t)(lo + it));
		for (k = 0; k < m; k++)
			cc.env[loops.sslot[s + k]] = st[k];
		for (k = 0; k < m; k++) {
			uint32_t v = blast(&cc, loops.next[s + k]);
			d = aig_and(B.g, d, D(v));
			ext64(&nx[k], V(v), N.ty[loops.next[s + k]]);
		}
		for (k = 0; k < m; k++)
			st[k] = nx[k];
		ctx_free(&cc);
	}
	w_resize(&B.v[res].v, &st[loop_res(n)], w ? w : 1, 0);
	B.v[res].d = d;
	return res;
}

/* lo <= x <= hi for a parameter's 64-bit value */
static lit
in_bounds(const struct param *p, const struct word *x)
{
	int64_t lo, hi;
	struct word k;
	lit r = L_TRUE;

	if (TYK(p->ty) == K_UINT && TYW(p->ty) == 64) {
		if (!p->ranged || p->lo < 0 || p->hi < 0)
			return L_TRUE;
		w_const(&k, 64, (uint64_t)p->lo);
		r = L_NOT(w_ult(B.g, x, &k));
		w_const(&k, 64, (uint64_t)p->hi);
		return aig_and(B.g, r, L_NOT(w_ult(B.g, &k, x)));
	}
	param_bounds(p, &lo, &hi);
	w_const(&k, 64, (uint64_t)lo);
	r = L_NOT(w_slt(B.g, x, &k));
	w_const(&k, 64, (uint64_t)hi);
	return aig_and(B.g, r, L_NOT(w_slt(B.g, &k, x)));
}

/*
 * nsat_narrow: a parameter whose range fits in fewer bits than its type
 * gets only those as inputs; the bits above are 0 (lo >= 0) or copies of
 * the sign bit.  That is what the range assumption says anyway (it is
 * still asserted), but the circuit shrinks: a u32 multiply of operands
 * in 0..255 becomes an 8 x 8 one.
 */
int nsat_narrow;

static void
narrow_input(struct word *x, const struct param *p)
{
	int64_t lo, hi;
	int k, i;

	if (TYK(p->ty) == K_UINT && TYW(p->ty) == 64 && (!p->ranged ||
	    p->lo < 0 || p->hi < 0))
		return;
	param_bounds(p, &lo, &hi);
	if (lo >= 0) {
		k = bits_for((uint64_t)hi);
		for (i = k; i < x->w; i++)
			x->b[i] = L_FALSE;
		return;
	}
	for (k = 1; k < 64; k++)
		if (lo >= -((int64_t)1 << (k - 1)) && hi < ((int64_t)1 << (k - 1)))
			break;
	for (i = k; i < x->w; i++)
		x->b[i] = x->b[k - 1];
}

int
nsat_supported(const struct func *f)
{
	int j;

	for (j = 0; j < f->np; j++)
		if (!bw(f->p[j].ty))
			return 0;
	return 1;
}

int
nsat_equiv(const struct func *f, nid a, nid b, long budget,
    struct nsat_res *out)
{
	struct nctx root;
	uint32_t va, vb, vw = 0;
	lit assume = L_TRUE, goal;
	int j, r;

	memset(out, 0, sizeof *out);
	memset(&B, 0, sizeof B);
	if (!nsat_supported(f) || !bw(N.ty[a])) {
		out->status = -2;
		out->why = "a float or real parameter or result";
		return -2;
	}
	B.g = aig_new();
	B.npar = f->np;
	for (j = 0; j < f->np; j++) {
		struct word x;
		w_input(B.g, &x, bw(f->p[j].ty));
		if (nsat_narrow)
			narrow_input(&x, &f->p[j]);
		ext64(&B.par[j], &x, f->p[j].ty);
		assume = aig_and(B.g, assume, in_bounds(&f->p[j], &B.par[j]));
	}
	b = mk_cast(N.ty[a], b);
	ctx_init(&root, NULL, 0);
	if (f->where)
		vw = blast(&root, f->where);
	va = blast(&root, a);
	vb = blast(&root, b);
	if (f->where)
		assume = aig_and(B.g, assume, aig_and(B.g, D(vw), nz(V(vw))));
	if (B.why) {
		out->status = -2;
		out->why = B.why;
		out->nodes = aig_nodes(B.g);
		goto done;
	}
	goal = aig_and(B.g, assume, aig_and(B.g, D(va), aig_or(B.g,
	    L_NOT(D(vb)), w_ne(B.g, V(va), V(vb)))));
	out->nodes = aig_nodes(B.g);
	aig_seconds = NSAT_SECONDS;
	r = aig_sat(B.g, goal, budget);
	aig_seconds = 0;
	out->conflicts = aig_conflicts;
	out->status = r == 0 ? 0 : r == 1 ? 1 : -1;
	if (r == 1)
		for (j = 0; j < f->np; j++)
			out->in[j] = (int64_t)aig_word_value(B.g, &B.par[j]);
done:
	ctx_free(&root);
	aig_free(B.g);
	free(B.v);
	memset(&B, 0, sizeof B);
	return out->status;
}
