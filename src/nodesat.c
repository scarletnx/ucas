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
 * Floats (K_FLOAT) are fpsat.c's circuits: a float node keeps its
 * unpacked value (an fpv in B.f) beside the word, which holds its bits
 * with a NaN's payload read as the quiet NaN, so that comparisons and the
 * miter see equal values as equal literals; the payload is packed only
 * where a bit cast reads it.  A NaN that arithmetic makes has a free
 * payload, as the host's does; a binary32 parameter reads quiet, as the
 * evaluator holds it in a double; its domain is the evaluator's: the
 * finite values within its range, -0 read as +0.
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
#include "texalu.h"

#include <stdlib.h>
#include <string.h>

#define UNROLL 256          /* most iterations of one loop unrolled */
#define MAXAIG (4 << 20)    /* AIG nodes before giving up */

struct bv {
	struct word v;
	lit d;
	int32_t f;                /* a float's unpacked value in B.f, or -1 */
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
	struct fpv *fl;           /* floats' unpacked values */
	uint32_t nfl, flcap;
	struct word par[MAXP];    /* parameters, 64-bit */
	int npar;
	const struct func *f;
	struct word *lw;          /* the lanes' words of `lanes` parameters:
				   * inputs made when first read */
	uint8_t lwset[MAXP][SG_MAXW];
	lit lassume;              /* their ranges */
} B;

int64_t nsat_lanes[MAXP][SG_MAXW];
uint8_t nsat_lanes_set[MAXP][SG_MAXW];

static lit in_bounds(const struct param *p, const struct word *x);

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
	B.v[B.n].f = -1;
	return B.n++;
}

static int
unsup(const char *why)
{
	if (!B.why)
		B.why = why;
	return 0;
}

/* the width of a type in bits, 0 if it is neither a machine integer nor
 * an IEEE float */
static int
bw(ty_t t)
{
	switch (TYK(t)) {
	case K_BOOL: return 1;
	case K_UINT:
	case K_SINT:
	case K_FLOAT: return TYW(t);
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

static uint32_t blast_flt(struct nctx *c, nid n);

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

	/* a float, or an integer read from one (N.a is a node only for
	 * some operations) */
	if (TYK(t) == K_FLOAT || (TYK(ta) == K_FLOAT && ((opinfo[op].fl &
	    OF_CMP) || op == O_CAST || op == O_ASUINT || op == O_ASINT ||
	    op == O_LNOT || op == O_LAND || op == O_LOR)))
		return blast_flt(c, n);
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
		while (q && (s >= 64 || !((q->bound >> s) & 1)))
			q = q->up;
		if (q)
			x = q->env[s];
		else if (s < B.npar)
			x = B.par[s];
		else if (s >= SG_FREE_SLOT && s < SG_FREE_SLOT + MAXP * SG_MAXW &&
		    (s - SG_FREE_SLOT) / SG_MAXW < B.npar && B.lw) {
			/* a `lanes` parameter in one lane: a free word */
			int j = (s - SG_FREE_SLOT) / SG_MAXW;
			int l = (s - SG_FREE_SLOT) % SG_MAXW;
			struct word *lw = &B.lw[j * SG_MAXW + l];
			if (!B.lwset[j][l]) {
				struct word in;
				w_input(g, &in, bw(B.f->p[j].ty));
				ext64(lw, &in, B.f->p[j].ty);
				B.lassume = aig_and(g, B.lassume, in_bounds(&B.f->p[j],
				    lw));
				B.lwset[j][l] = 1;
			}
			x = *lw;
		} else {
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
		if (TYK(ta) == K_FLOAT) {
			unsup("a float condition");
			break;
		}
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
	case O_FETCH_REPEAT: case O_FETCH_MIRROR: case O_FETCH_CLAMP:
	case O_FETCH_MCLAMP: case O_FETCH_BORDER: case O_FETCH_TEXEL:
	case O_FETCH_BUFFER:
		/* a read with an address mode, a texel fetch: its definition
		 * on plain operations (texalu.c) */
		return blast(c, tex_expand(n));
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
		/* a value's index can be 0: the node's operands say whether
		 * there is one */
		d = N.a[n] ? D(a) : L_TRUE;
		if (opinfo[op].arity >= 2 && N.b[n])
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
					lit amin = w_eq(g, V(a), &x);
					d = aig_and(g, d, L_NOT(aig_and(g, amin,
					    w_eq(g, V(b), &y))));
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


#define FV(i) (&B.fl[B.v[i].f])

/* a new value for a float: its unpacked form, and its bits with any NaN
 * read as the quiet one */
static uint32_t
fval(const struct fpv *x, ty_t t, lit d)
{
	uint32_t r;

	if (B.nfl == B.flcap) {
		B.flcap = newcap(B.flcap, B.nfl + 1);
		B.fl = xrealloc(B.fl, B.flcap * sizeof *B.fl);
	}
	B.fl[B.nfl] = *x;
	r = newval();
	B.v[r].f = (int32_t)B.nfl++;
	fp_pack(B.g, &B.v[r].v, x, fp_fmt(TYW(t)), 0);
	B.v[r].d = d;
	return r;
}

/* the sign bit, a NaN's own included (copysign reads it) */
static lit
fsign(const struct fpv *x)
{
	return aig_mux(B.g, x->nan, x->nb.b[x->nb.w - 1], x->s);
}

/*
 * nsat_ftz: a subnormal x (bits its packed word) as the zero of its sign,
 * in both forms
 */
int nsat_ftz;

/*
 * nsat_floats: floats are blasted (equiv's own question); elsewhere (find,
 * simplify, the C frontend) a candidate on floats stays sampled, as each
 * proof may take the solver's whole budget
 */
int nsat_floats;

/*
 * nsat_nanbits (equiv's `nanbits`): two NaN results are equal only with
 * the same sign and payload; the others compare as before
 */
int nsat_nanbits;

static void
flush(struct fpv *x, struct word *bits, const struct fpfmt *F)
{
	int fb = F->p - 1, i;
	lit ez = L_TRUE, fz = L_TRUE, sub;
	struct fpv z;

	/* m >= 1, so 2^elo or more: nothing below 2^emin, nothing to flush
	 * (an f32 widened to f64 stays one circuit) */
	if (x->elo >= 2 - (1 << (F->eb - 1)))
		return;
	for (i = 0; i < fb; i++)
		fz = aig_and(B.g, fz, L_NOT(bits->b[i]));
	for (i = 0; i < F->eb; i++)
		ez = aig_and(B.g, ez, L_NOT(bits->b[fb + i]));
	sub = aig_and(B.g, ez, L_NOT(fz));
	if (sub == L_FALSE)
		return;
	fp_const(B.g, &z, 0, F);
	z.s = x->s;
	fp_mux(B.g, x, sub, &z, x);
	for (i = 0; i < bits->w - 1; i++)
		bits->b[i] = aig_and(B.g, bits->b[i], L_NOT(sub));
}

/* an operation's float result, flushed under ftz */
static uint32_t
fres(struct fpv *r, ty_t t, lit d)
{
	if (nsat_ftz) {
		struct word w;
		fp_pack(B.g, &w, r, fp_fmt(TYW(t)), 0);
		flush(r, &w, fp_fmt(TYW(t)));
	}
	return fval(r, t, d);
}

/* what a float operation that has no circuit is called */
static const char *
flt_unsup(int op)
{
	switch (op) {
	case O_DIV: return "a float division";
	case O_MOD: return "a float remainder";
	case O_POW: return "a float power";
	case O_SQRT: return "a float square root";
	case O_LAND: case O_LOR: case O_LNOT: return "a float truth value";
	case O_ATOMIC: return "a float atomic";
	}
	return "a float function";
}

/*
 * A node of float type, or one that reads a float (a comparison, a bit
 * cast): fpsat.c's circuits, rounded to nearest even as the evaluator
 * rounds.
 */
static uint32_t
blast_flt(struct nctx *c, nid n)
{
	int op = N.op[n];
	ty_t t = N.ty[n], ta = N.a[n] ? N.ty[N.a[n]] : 0;
	const struct fpfmt *F = TYK(t) == K_FLOAT ? fp_fmt(TYW(t)) : NULL;
	const struct fpfmt *FA = TYK(ta) == K_FLOAT ? fp_fmt(TYW(ta)) : NULL;
	struct aig *g = B.g;
	struct fpv r, xa, xb;
	struct word w, x, wa, wb;
	uint32_t a = 0, b = 0, k, res;
	lit d = L_TRUE;
	int has_a, has_b;

	if (!nsat_floats) {
		unsup("a float or real value");
		op = O_NONE;
	} else if (IS_TEX(op) && !IS_FETCH(op) && !IS_TEXEL(op) &&
	    op != O_FETCH_LAYER) {
		/* the unit's filter: weights on its grid, one rounding */
		unsup("a filtered texture read");
		op = O_NONE;
	}
	switch (op) {
	case O_CONST:
		fp_const(g, &r, fbits(TYW(t), N.kr[n]), F);
		return fval(&r, t, L_TRUE);
	case O_VAR: {
		int s = (int)N.ki[n];
		struct nctx *q = c;
		while (q && (s >= 64 || !((q->bound >> s) & 1)))
			q = q->up;
		if (q)
			x = q->env[s];
		else if (s < B.npar)
			x = B.par[s];
		else {
			unsup(s >= SG_FREE_SLOT ? "a float `lanes` parameter" :
			    "a free variable");
			w_const(&x, 64, 0);
		}
		w_resize(&w, &x, TYW(t), 0);
		fp_unpack(g, &r, &w, F);
		return fval(&r, t, L_TRUE);
	}
	case O_LOOP:
		k = blast_loop(c, n);
		fp_unpack(g, &r, V(k), F);
		return fval(&r, t, D(k));
	case O_SEL:
		if (TYK(ta) == K_FLOAT) {
			unsup("a float condition");
			break;
		}
		a = blast(c, N.a[n]);
		b = blast(c, N.b[n]);
		k = blast(c, N.c[n]);
		{
			lit s = nz(V(a));
			fp_mux(g, &r, s, FV(b), FV(k));
			d = aig_and(g, D(a), aig_mux(g, s, D(b), D(k)));
		}
		return fval(&r, t, d);
	case O_INDEX: {
		nid tab = N.a[n];
		uint32_t cnt = N.b[tab], j;
		struct fpv e;
		lit de = L_FALSE;
		b = blast(c, N.b[n]);
		ext64(&x, V(b), N.ty[N.b[n]]);
		d = aig_and(g, D(b), below(V(b), N.ty[N.b[n]], cnt));
		fp_const(g, &r, 0, F);
		for (j = cnt; j-- > 0;) {
			uint32_t v = blast(c, N.lst[N.a[tab] + j]);
			struct word y;
			lit hit;
			w_const(&y, 64, j);
			hit = w_eq(g, &x, &y);
			e = *FV(v);
			fp_mux(g, &r, hit, &e, &r);
			de = aig_mux(g, hit, D(v), de);
		}
		return fval(&r, t, aig_and(g, d, de));
	}
	case O_FETCH_REPEAT: case O_FETCH_MIRROR: case O_FETCH_CLAMP:
	case O_FETCH_MCLAMP: case O_FETCH_BORDER:
		return blast(c, tex_expand(n));
	}
	/* a value's index can be 0: has_a and has_b say whether there is one */
	has_a = N.a[n] != 0;
	has_b = opinfo[op].arity >= 2 && N.b[n] != 0;
	if (has_a)
		a = blast(c, N.a[n]);
	if (has_b)
		b = blast(c, N.b[n]);
	if (has_a)
		d = D(a);
	if (has_b)
		d = aig_and(g, d, D(b));
	if ((op == O_ADD || op == O_SUB || op == O_MUL || op == O_MIN ||
	    op == O_MAX || op == O_COPYSIGN || op == O_FMA) &&
	    (TYW(N.ty[N.b[n]]) != TYW(t) ||
	    TYW(ta) != TYW(t))) {
		unsup("floats of two widths in one operation");
		op = O_NONE;
	}
	/* the float operands, flushed under ftz but for a bit cast */
	if (FA) {
		xa = *FV(a);
		wa = *V(a);
		if (nsat_ftz && op != O_ASUINT && op != O_ASINT && op != O_BARRIER)
			flush(&xa, &wa, FA);
	}
	if (has_b && TYK(N.ty[N.b[n]]) == K_FLOAT) {
		xb = *FV(b);
		wb = *V(b);
		if (nsat_ftz)
			flush(&xb, &wb, fp_fmt(TYW(N.ty[N.b[n]])));
	}
	switch (op) {
	case O_NEG:
	case O_ABS:
		r = xa;
		r.s = op == O_NEG ? L_NOT(r.s) : L_FALSE;
		r.nb.b[r.nb.w - 1] = op == O_NEG ? L_NOT(r.nb.b[r.nb.w - 1]) :
		    L_FALSE;
		return fres(&r, t, d);
	case O_COPYSIGN: {
		lit s = fsign(&xb);
		r = xa;
		r.s = s;
		r.nb.b[r.nb.w - 1] = s;
		return fres(&r, t, d);
	}
	case O_ADD:
	case O_SUB:
		if (op == O_SUB)
			xb.s = L_NOT(xb.s);
		fp_add(g, &r, &xa, &xb, F, FR_RNE);
		return fres(&r, t, d);
	case O_MUL:
		fp_mul(g, &r, &xa, &xb, F, FR_RNE);
		return fres(&r, t, d);
	case O_FMA: {
		struct fpv xc;
		struct word wc;
		if (TYW(N.ty[N.c[n]]) != TYW(t)) {
			unsup("floats of two widths in one operation");
			break;
		}
		k = blast(c, N.c[n]);
		d = aig_and(g, d, D(k));
		xc = *FV(k);
		wc = *V(k);
		if (nsat_ftz)
			flush(&xc, &wc, F);
		fp_fma(g, &r, &xa, &xb, &xc, F, FR_RNE);
		return fres(&r, t, d);
	}
	case O_MIN:
	case O_MAX:
		fp_minmax(g, &r, &xa, &wa, &xb, &wb, op == O_MAX, F);
		return fres(&r, t, d);
	case O_SIGN: {
		/* NaN stays, else 1, -1 or +0 */
		struct fpv one, mone, zero;
		struct word z;
		lit gt, lt;
		fp_const(g, &one, fbits(TYW(t), 1.0), F);
		fp_const(g, &mone, fbits(TYW(t), -1.0), F);
		fp_const(g, &zero, 0, F);
		w_const(&z, TYW(t), 0);
		gt = fp_lt(g, &z, &wa, F);
		lt = fp_lt(g, &wa, &z, F);
		fp_mux(g, &r, lt, &mone, &zero);
		fp_mux(g, &r, gt, &one, &r);
		fp_mux(g, &r, xa.nan, &xa, &r);
		return fres(&r, t, d);
	}
	case O_CAST:
		if (FA && F) {
			if (TYW(ta) == TYW(t))
				r = xa;
			else
				fp_round(g, &r, &xa, F, FR_RNE);
			return fres(&r, t, d);
		}
		if (FA) {
			/* toward zero, defined within [-2^63, 2^63) (eval.c,
			 * real_to_int), then wrapped to the type */
			lit ok = fp_to_int(g, &w, &xa);
			res = newval();
			w_resize(&B.v[res].v, &w, bw(t) ? bw(t) : 1, 0);
			B.v[res].d = aig_and(g, d, ok);
			return res;
		}
		fp_from_int(g, &r, V(a), sgn(ta), F, FR_RNE);
		return fres(&r, t, d);
	case O_FLOOR:
	case O_CEIL:
	case O_ROUNDEVEN:
		fp_rint(g, &r, &xa, F, op == O_FLOOR ? FR_RDN : op == O_CEIL ?
		    FR_RUP : FR_RNE);
		return fres(&r, t, d);
	case O_ASFLOAT:
		w_resize(&w, V(a), TYW(t), 0);
		if (TYW(t) == 32)
			fp_quiet(g, &w, F);
		fp_unpack(g, &r, &w, F);
		return fval(&r, t, d);
	case O_BARRIER:
		r = xa;
		return fval(&r, t, d);
	case O_ASUINT:
	case O_ASINT:
		fp_pack(g, &w, &xa, FA, 1);
		res = newval();
		w_resize(&B.v[res].v, &w, bw(t), 0);
		B.v[res].d = d;
		return res;
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE: {
		lit v;
		switch (op) {
		case O_EQ: v = fp_eq(g, &wa, &wb, FA); break;
		case O_NE: v = L_NOT(fp_eq(g, &wa, &wb, FA)); break;
		case O_LT: v = fp_lt(g, &wa, &wb, FA); break;
		case O_GT: v = fp_lt(g, &wb, &wa, FA); break;
		case O_LE:
			v = fp_lt(g, &wa, &wb, FA);
			v = aig_or(g, v, fp_eq(g, &wa, &wb, FA));
			break;
		default:
			v = fp_lt(g, &wb, &wa, FA);
			v = aig_or(g, v, fp_eq(g, &wa, &wb, FA));
			break;
		}
		res = newval();
		w_const(&B.v[res].v, bw(t) ? bw(t) : 1, 0);
		B.v[res].v.b[0] = v;
		B.v[res].d = d;
		return res;
	}
	case O_NONE:
		break;
	default:
		unsup(flt_unsup(op));
	}
	/* unsupported: a placeholder of the node's kind */
	if (F) {
		fp_const(g, &r, 0, F);
		return fval(&r, t, L_TRUE);
	}
	res = newval();
	w_const(&B.v[res].v, bw(t) ? bw(t) : 1, 0);
	B.v[res].d = L_TRUE;
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
		if (TYK(N.ty[n]) == K_FLOAT) {
			struct fpv z;
			fp_const(B.g, &z, 0, fp_fmt(TYW(N.ty[n])));
			r = fval(&z, N.ty[n], L_TRUE);
		} else {
			r = newval();
			w_const(&B.v[r].v, 1, 0);
			B.v[r].d = L_TRUE;
		}
	} else
		r = blast1(c, n);
	*ctx_slot(c, n) = r;
	return r;
}

/* a value's bits as a loop keeps them: a float's NaN with its payload */
static void
state_bits(struct word *r, uint32_t v, ty_t t)
{
	struct word w;

	if (TYK(t) == K_FLOAT) {
		fp_pack(B.g, &w, FV(v), fp_fmt(TYW(t)), 1);
		ext64(r, &w, t);
	} else
		ext64(r, V(v), t);
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
		uint32_t v = blast(&cc, sg_lower(loops.init[s + k]));
		d = aig_and(B.g, d, D(v));
		state_bits(&st[k], v, N.ty[loops.init[s + k]]);
	}
	ctx_free(&cc);
	for (it = 0; it < trip && !B.why; it++) {
		struct word nx[MAXST];
		ctx_init(&cc, c, loops.bound[e]);
		w_const(&cc.env[loops.islot[e]], 64, (uint64_t)(lo + it));
		for (k = 0; k < m; k++)
			cc.env[loops.sslot[s + k]] = st[k];
		for (k = 0; k < m; k++) {
			uint32_t v = blast(&cc, sg_lower(loops.next[s + k]));
			d = aig_and(B.g, d, D(v));
			state_bits(&nx[k], v, N.ty[loops.next[s + k]]);
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

	if (TYK(p->ty) == K_FLOAT)
		return;
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

/*
 * A float parameter's domain, as the evaluator enumerates it (eval.c,
 * diter_init): the finite values whose ordered keys lie within the
 * range's (the whole finite line without one), -0 being +0's key.
 */
static lit
flt_domain(const struct param *p, const struct word *x)
{
	int w = TYW(p->ty), i;
	double mx = w == 32 ? 3.4028234663852886e38 : w == 16 ? 65504.0 :
	    1.7976931348623157e308;
	double rlo = p->ranged ? p->rlo : -mx, rhi = p->ranged ? p->rhi : mx;
	int64_t lo = fkey(w, rlo), hi = fkey(w, rhi);
	struct word mag, key, neg, k;
	lit mz = L_TRUE, r;

	if (fromkey(w, lo) < rlo)
		lo++;
	if (fromkey(w, hi) > rhi)
		hi--;
	if (hi < lo)
		return L_FALSE;
	w_const(&mag, 64, 0);
	for (i = 0; i < w - 1; i++) {
		mag.b[i] = x->b[i];
		mz = aig_and(B.g, mz, L_NOT(x->b[i]));
	}
	w_neg(B.g, &neg, &mag);
	w_mux(B.g, &key, x->b[w - 1], &neg, &mag);
	w_const(&k, 64, (uint64_t)lo);
	r = L_NOT(w_slt(B.g, &key, &k));
	w_const(&k, 64, (uint64_t)hi);
	r = aig_and(B.g, r, L_NOT(w_slt(B.g, &k, &key)));
	return aig_and(B.g, r, L_NOT(aig_and(B.g, x->b[w - 1], mz)));
}

static int
has_flt(nid e, uint8_t *seen)
{
	int op;

	if (!e || seen[e])
		return 0;
	seen[e] = 1;
	op = N.op[e];
	if (TYK(N.ty[e]) == K_FLOAT)
		return 1;
	if (op == O_CONST || op == O_VAR)
		return 0;
	if (op == O_INDEX) {
		nid tab = N.a[e];
		uint32_t j;
		for (j = 0; j < N.b[tab]; j++)
			if (has_flt(N.lst[N.a[tab] + j], seen))
				return 1;
		return has_flt(N.b[e], seen);
	}
	if (op == O_LOOP) {
		uint32_t le = loop_ent(e), s = loops.st[le], k;
		for (k = 0; k < loops.nst[le]; k++)
			if (has_flt(loops.init[s + k], seen) ||
			    has_flt(loops.next[s + k], seen))
				return 1;
	}
	if (op == O_TABLE)
		return 0;
	return has_flt(N.a[e], seen) || has_flt(N.b[e], seen) ||
	    (opinfo[op].arity >= 3 && has_flt(N.c[e], seen));
}

/* a float in f's parameters, its where clause or e */
int
nsat_has_float(const struct func *f, nid e)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int j, r = 0;

	for (j = 0; j < f->np; j++)
		r |= TYK(f->p[j].ty) == K_FLOAT;
	r = r || has_flt(e, seen) || has_flt(f->where, seen);
	free(seen);
	return r;
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
	lit assume = L_TRUE, goal, ne;
	int j, r;
	nid where = sg_lower(f->where);

	/* subgroup operations as the circuits of their lanes (sg.c) */
	a = sg_lower(a);
	b = sg_lower(b);
	memset(out, 0, sizeof *out);
	memset(&B, 0, sizeof B);
	if (!nsat_supported(f) || !bw(N.ty[a]) || (!nsat_floats &&
	    (!TY_ISINT(N.ty[a]) || nsat_has_float(f, 0)))) {
		out->status = -2;
		out->why = nsat_floats ? "a real parameter or result" :
		    "a float or real parameter or result";
		return -2;
	}
	B.g = aig_new();
	B.npar = f->np;
	B.f = f;
	B.lassume = L_TRUE;
	for (j = 0; j < f->np; j++)
		if (f->p[j].lanes && !B.lw)
			B.lw = xcalloc((size_t)MAXP * SG_MAXW, sizeof *B.lw);
	fp_err = NULL;
	for (j = 0; j < f->np; j++) {
		struct word x;
		ty_t pt = f->p[j].ty;
		w_input(B.g, &x, bw(pt));
		if (TYK(pt) == K_FLOAT) {
			if (TYW(pt) == 32)
				fp_quiet(B.g, &x, fp_fmt(32));
			assume = aig_and(B.g, assume, flt_domain(&f->p[j], &x));
			ext64(&B.par[j], &x, pt);
			continue;
		}
		if (nsat_narrow)
			narrow_input(&x, &f->p[j]);
		ext64(&B.par[j], &x, pt);
		assume = aig_and(B.g, assume, in_bounds(&f->p[j], &B.par[j]));
	}
	b = mk_cast(N.ty[a], b);
	ctx_init(&root, NULL, 0);
	if (where)
		vw = blast(&root, where);
	va = blast(&root, a);
	vb = blast(&root, b);
	if (where)
		assume = aig_and(B.g, assume, aig_and(B.g, D(vw), nz(V(vw))));
	if (fp_err)
		unsup(fp_err);
	if (B.why) {
		out->status = -2;
		out->why = B.why;
		out->nodes = aig_nodes(B.g);
		goto done;
	}
	assume = aig_and(B.g, assume, B.lassume);
	/* floats differ as the evaluator's equiv sees them: not both NaN,
	 * and not == (so -0 and +0 are one value); under nanbits two NaNs
	 * are the same only bit for bit (their words with the payload) */
	if (TYK(N.ty[a]) == K_FLOAT) {
		const struct fpfmt *F = fp_fmt(TYW(N.ty[a]));
		lit same = fp_same(B.g, V(va), V(vb), F);
		if (nsat_nanbits) {
			struct word pa, pb;
			fp_pack(B.g, &pa, FV(va), F, 1);
			fp_pack(B.g, &pb, FV(vb), F, 1);
			same = aig_or(B.g, fp_eq(B.g, V(va), V(vb), F),
			    aig_and(B.g, aig_and(B.g, FV(va)->nan, FV(vb)->nan),
			    w_eq(B.g, &pa, &pb)));
		}
		ne = L_NOT(same);
	} else
		ne = w_ne(B.g, V(va), V(vb));
	goal = aig_and(B.g, assume, aig_and(B.g, D(va), aig_or(B.g,
	    L_NOT(D(vb)), ne)));
	out->nodes = aig_nodes(B.g);
	aig_seconds = NSAT_SECONDS;
	r = aig_sat(B.g, goal, budget);
	aig_seconds = 0;
	out->conflicts = aig_conflicts;
	out->status = r == 0 ? 0 : r == 1 ? 1 : -1;
	if (r == 1) {
		int l;
		for (j = 0; j < f->np; j++)
			out->in[j] = (int64_t)aig_word_value(B.g, &B.par[j]);
		memcpy(nsat_lanes_set, B.lwset, sizeof nsat_lanes_set);
		for (j = 0; j < f->np; j++)
			for (l = 0; l < SG_MAXW; l++)
				nsat_lanes[j][l] = B.lwset[j][l] ? (int64_t)
				    aig_word_value(B.g, &B.lw[j * SG_MAXW + l]) : 0;
	}
done:
	ctx_free(&root);
	aig_free(B.g);
	free(B.v);
	free(B.lw);
	free(B.fl);
	memset(&B, 0, sizeof B);
	fp_err = NULL;
	return out->status;
}
