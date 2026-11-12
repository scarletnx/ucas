/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * swar.c: packed lanes in a u32, worked on all at once (SIMD within a
 * register).
 *
 * A value that treats a word as 8 x 4, 4 x 8 or 2 x 16 bits, each lane of
 * the result a function of the same lane of one or two words, is one of
 * a few word-wide forms.  With H the lanes' top bits, L = ~H the rest and
 * ONE their low bits:
 *
 *   add      ((a & L) + (b & L)) ^ ((a ^ b) & H)     the top bits are
 *   sub      ((a | H) - (b & L)) ^ (~(a ^ b) & H)    added apart, so no
 *                                                    carry crosses a lane
 *   avg      (a & b) + (((a ^ b) & ~ONE) >> 1)       floor; the ceiling
 *            (a | b) - (((a ^ b) & ~ONE) >> 1)       is the second
 *   a >= b   G = ((a & ~b) | (~(a ^ b) & ((a | H) - (b & L)))) & H: the
 *            top bit of each lane; a mask of whole lanes is
 *            M(G) = (G - (G >> (w - 1))) | G
 *   max, min b ^ ((a ^ b) & M), a ^ ((a ^ b) & M), M = M(a >= b)
 *   |a - b|  max - min in one subtraction: no lane borrows
 *   a +sat b S | M(((a & b) | ((a | b) & ~S)) & H), S the lanes' sum
 *   a -sat b D & M(a >= b), D the lanes' difference
 *   a == b   M(~(((x & L) + L) | x) & H), x = a ^ b; != without the ~
 *
 * The lanes are unsigned.  Which form a value is, if any, is read from
 * its values: it is run on words whose lanes are random, equal, zero and
 * all ones, and each lane of the result is compared with the lane
 * operation's (swar_of).  With one word, the other is a constant looked
 * for lane by lane.  What that finds is a guess: find proves the
 * rewritten function equal on the whole domain before it takes it.
 */
#include "calc.h"
#include "fuse.h"
#include "sg.h"

#include <stdlib.h>
#include <string.h>

#define U32 TY(K_UINT, 32)
#define SW_SAMPLES 256
#define SW_MAXNODE 512
#define SW_PROOFS 24	/* equivalence proofs per find */

enum {
	L_ADD, L_SUB, L_AVGF, L_AVGC, L_MAX, L_MIN, L_ABSD, L_SATADD, L_SATSUB,
	L_EQ, L_NE, L_GE, L_GT, L_N
};

#define TOPS "the lanes' top bits go apart, so no carry crosses a lane"
#define MASK "whole lanes from the top bit a word-wide compare leaves"

static const struct {
	const char *name;	/* the lane operation, over its two operands */
	int comm;
	const char *how;
} lops[L_N] = {
	{"%s + %s", 1, TOPS}, {"%s - %s", 0, TOPS},
	{"(%s + %s) >> 1", 1, "what both have plus half of what one has"},
	{"(%s + %s + 1) >> 1", 1, "what either has less half of what one has"},
	{"max(%s, %s)", 1, "chosen by " MASK},
	{"min(%s, %s)", 1, "chosen by " MASK},
	{"|%s - %s|", 1, "max - min, which borrows from no lane"},
	{"min(%s + %s, the lane's largest)", 1,
	    "the sum, all ones where a lane carried"},
	{"max(%s - %s, 0)", 0, "the difference, 0 where a lane would borrow"},
	{"%s == %s ? all ones : 0", 1, MASK},
	{"%s != %s ? all ones : 0", 1, MASK},
	{"%s >= %s ? all ones : 0", 0, MASK},
	{"%s > %s ? all ones : 0", 0, MASK},
};

/* the operation on one lane of w bits */
static uint32_t
lane_op(int op, uint32_t a, uint32_t b, int w)
{
	uint32_t m = (1u << w) - 1;

	switch (op) {
	case L_ADD: return (a + b) & m;
	case L_SUB: return (a - b) & m;
	case L_AVGF: return (a + b) >> 1;
	case L_AVGC: return (a + b + 1) >> 1;
	case L_MAX: return a > b ? a : b;
	case L_MIN: return a < b ? a : b;
	case L_ABSD: return a > b ? a - b : b - a;
	case L_SATADD: return a + b > m ? m : a + b;
	case L_SATSUB: return a > b ? a - b : 0;
	case L_EQ: return a == b ? m : 0;
	case L_NE: return a != b ? m : 0;
	case L_GE: return a >= b ? m : 0;
	default: return a > b ? m : 0;
	}
}

static nid
k32(uint32_t v)
{
	return mk_const_i(U32, (int64_t)v);
}

#define AND(a, b) mk_bin(O_AND, (a), (b))
#define OR(a, b) mk_bin(O_OR, (a), (b))
#define XOR(a, b) mk_bin(O_XOR, (a), (b))
#define NOT(a) mk_un(O_NOT, (a))

/* whole lanes from their top bits */
static nid
lane_mask(nid g, int w)
{
	return OR(SUB(g, mk_bin(O_SHR, g, k32((uint32_t)w - 1))), g);
}

/* the word-wide form of op on lanes of w bits */
static nid
swar_form(int op, nid a, nid b, int w)
{
	uint32_t one = 0, h;
	nid H, L, x, t, g, m, s;
	int k;

	for (k = 0; k < 32; k += w)
		one |= 1u << k;
	h = one << (w - 1);
	H = k32(h);
	L = k32(~h);
	x = XOR(a, b);
	switch (op) {
	case L_ADD:
		return XOR(ADD(AND(a, L), AND(b, L)), AND(x, H));
	case L_SUB:
		return XOR(SUB(OR(a, H), AND(b, L)), AND(NOT(x), H));
	case L_AVGF:
		return ADD(AND(a, b), mk_bin(O_SHR, AND(x, k32(~one)), k32(1)));
	case L_AVGC:
		return SUB(OR(a, b), mk_bin(O_SHR, AND(x, k32(~one)), k32(1)));
	case L_EQ:
	case L_NE:
		t = OR(ADD(AND(x, L), L), x);
		return lane_mask(AND(op == L_EQ ? NOT(t) : t, H), w);
	case L_SATADD:
		s = swar_form(L_ADD, a, b, w);
		return OR(s, lane_mask(AND(OR(AND(a, b), AND(OR(a, b), NOT(s))),
		    H), w));
	}
	/* the rest goes through a >= b */
	t = SUB(OR(a, H), AND(b, L));
	g = AND(OR(AND(a, NOT(b)), AND(NOT(x), t)), H);
	if (op == L_GT) {
		/* a > b is not (b >= a) */
		t = SUB(OR(b, H), AND(a, L));
		g = AND(NOT(OR(AND(b, NOT(a)), AND(NOT(x), t))), H);
	}
	m = lane_mask(g, w);
	switch (op) {
	case L_GE:
	case L_GT:
		return m;
	case L_MAX:
		return XOR(b, AND(x, m));
	case L_MIN:
		return XOR(a, AND(x, m));
	case L_ABSD:
		return SUB(XOR(b, AND(x, m)), XOR(a, AND(x, m)));
	default:	/* L_SATSUB */
		return AND(swar_form(L_SUB, a, b, w), m);
	}
}

/*
 * Lane k (its low bit) of every sampled result is op of the same lane of
 * the first word and of the second, or of the constant c when vb is
 * NULL; sw: the operands the other way round.
 */
static int
lane_fits(const uint32_t *va, const uint32_t *vb, const uint32_t *vr, int op,
    int sw, int w, int k, uint32_t c)
{
	uint32_t lm = (1u << w) - 1, p, q;
	int smp;

	for (smp = 0; smp < SW_SAMPLES; smp++) {
		p = (va[smp] >> k) & lm;
		q = vb ? (vb[smp] >> k) & lm : c;
		if (((vr[smp] >> k) & lm) != (sw ? lane_op(op, q, p, w) :
		    lane_op(op, p, q, w)))
			return 0;
	}
	return 1;
}

struct swhit {
	int op, w, swap;	/* swap: the lane operation is op(b, a) */
	nid a, b;		/* the words; b a constant when one is read */
};

/*
 * The lane operations x's values agree with, for a value x of one or two
 * u32 parameters that fits a u32: up to max of them into out.
 */
static int
swar_of(const struct func *f, nid x, struct swhit *out, int max)
{
	static const int widths[3] = {8, 16, 4};
	uint64_t sl = node_slots(x), rs = 0x9e3779b97f4a7c15ULL;
	int slot[2] = {-1, -1}, ns = 0, j, smp, n = 0, wi, op, sw, k;
	struct plan pl;
	int64_t *in_i;
	double *in_r;
	uint32_t *va, *vb, *vr, loc;
	nid roots[1];

	if ((N.ty[x] != U32 && TYK(N.ty[x]) != K_INT &&
	    TYK(N.ty[x]) != K_LIT) || N.op[x] == O_VAR ||
	    N.op[x] == O_CONST || (sl >> MAXP) ||
	    (N.fl[x] & (NF_SG | NF_LANE | NF_LANEREAD)))
		return 0;
	for (j = 0; j < f->np; j++)
		if ((sl >> j) & 1) {
			/* whole words only: the samples are any u32 */
			int64_t lo, hi;
			if (ns == 2 || f->p[j].ty != U32)
				return 0;
			param_bounds(&f->p[j], &lo, &hi);
			if (lo != 0 || hi != 0xffffffffLL)
				return 0;
			slot[ns++] = j;
		}
	if (ns == 0)
		return 0;
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	va = xcalloc(3 * SW_SAMPLES, sizeof *va);
	vb = va + SW_SAMPLES;
	vr = vb + SW_SAMPLES;
	for (smp = 0; smp < SW_SAMPLES; smp++) {
		/* the corners first, then random words; in every other one,
		 * bytes (or halves) of the second made equal to the first's,
		 * or either's 0 or all ones, so compares and saturation show */
		static const uint32_t corner[4] = {
			0, 0xffffffff, 1, 0x80808080
		};
		uint32_t w0, w1, pick;
		int lw = (smp & 3) == 3 ? 16 : 8;
		rs ^= rs << 13;
		rs ^= rs >> 7;
		rs ^= rs << 17;
		w0 = (uint32_t)rs;
		w1 = (uint32_t)(rs >> 32);
		rs ^= rs << 13;
		rs ^= rs >> 7;
		rs ^= rs << 17;
		pick = (uint32_t)rs;
		for (k = 0; k < 32 && (smp & 1); k += lw, pick >>= 4) {
			uint32_t bm = (lw == 16 ? 65535u : 255u) << k;
			switch (pick & 7) {
			case 0: w1 = (w1 & ~bm) | (w0 & bm); break;
			case 1: w1 &= ~bm; break;
			case 2: w1 |= bm; break;
			case 3: w0 &= ~bm; break;
			case 4: w0 |= bm; break;
			}
		}
		if (smp < 16) {
			w0 = corner[smp & 3];
			w1 = corner[smp >> 2];
		}
		va[smp] = w0;
		vb[smp] = w1;
		in_i[(size_t)slot[0] * CH + smp] = (int64_t)w0;
		if (ns == 2)
			in_i[(size_t)slot[1] * CH + smp] = (int64_t)w1;
	}
	roots[0] = x;
	plan_build(&pl, roots, 1, 0);
	loc = plan_loc(&pl, x);
	plan_run(&pl, SW_SAMPLES, in_i, in_r);
	for (smp = 0; smp < SW_SAMPLES; smp++) {
		/* an int that holds a word (a ?: over literals) is one too */
		if ((COLF(&pl, loc)[smp] & F_UNDEF) ||
		    ((uint64_t)COLI(&pl, loc)[smp] >> 32))
			break;
		vr[smp] = (uint32_t)COLI(&pl, loc)[smp];
	}
	plan_free(&pl);
	free(in_i);
	free(in_r);
	if (smp < SW_SAMPLES) {
		free(va);
		return 0;
	}
	for (wi = 0; wi < 3 && n < max; wi++) {
		int w = widths[wi];

		/* a constant is looked for among 256 values at most */
		if (ns == 1 && w > 8)
			continue;
		for (op = 0; op < 2 * L_N && n < max; op++) {
			uint32_t kw = 0, c = 0;

			sw = op >= L_N;
			if (sw && lops[op - L_N].comm)
				continue;
			for (k = 0; k < 32; k += w) {
				/* the second word's lane, or the one constant
				 * that fits every sample */
				uint32_t nc = ns == 2 ? 1u : 1u << w;
				for (c = 0; c < nc; c++)
					if (lane_fits(va, ns == 2 ? vb : NULL,
					    vr, op % L_N, sw, w, k, c))
						break;
				if (c == nc)
					break;
				kw |= c << k;
			}
			if (k < 32)
				continue;
			out[n].op = op % L_N;
			out[n].w = w;
			out[n].swap = sw;
			out[n].a = var_of(slot[0], f->p[slot[0]].name, U32);
			out[n].b = ns == 2 ? var_of(slot[1], f->p[slot[1]].name,
			    U32) : k32(kw);
			n++;
		}
	}
	free(va);
	return n;
}

static int
all_nodes(nid e, nid *out, int n, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || n == SW_MAXNODE)
		return n;
	seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_UNDEF)
		return n;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			n = all_nodes(N.lst[N.a[e] + j], out, n, seen);
		return n;
	}
	out[n++] = e;
	n = all_nodes(N.a[e], out, n, seen);
	n = all_nodes(N.b[e], out, n, seen);
	return all_nodes(N.c[e], out, n, seen);
}

/* a word as the text of a lane operation's operand: a constant in hex */
static void
operand(char *out, size_t n, nid e)
{
	int64_t v;

	if (int_of(e, &v))
		snprintf(out, n, "0x%08x", (unsigned)v);
	else
		snprintf(out, n, "%s", node_str(e));
}

/*
 * find's section: the values of f that are lane operations, each
 * replaced by its word-wide form where the model prices the function
 * lower and the function so rewritten is proved equal to f on its
 * domain.  A guess the proof refuses is dropped without a word, like one
 * that is not cheaper: nothing is said when no value is left.
 */
void
swar_find(const struct func *f)
{
	const struct model *m = model_active();
	const char *mn = sym_str(m->name), *s, *e, *chk;
	struct sbuf why = {0}, best_why = {0}, w = {0};
	struct eqproof pr;
	struct swhit hit[16];
	nid cur = f->body, *nd;
	float before, cost;
	char pa[40], pb[40];
	int iter, tries = 0, how = EQ_SAMPLED;

	if (!TY_ISINT(N.ty[cur]) || f->where ||
	    (N.fl[cur] & (NF_SG | NF_LANE | NF_LANEREAD)))
		return;
	nd = xmalloc(SW_MAXNODE * sizeof *nd);
	before = cost = fuse_expr_cost(m, cur);
	for (iter = 0; iter < 8; iter++) {
		uint8_t *seen = xcalloc(N.n, 1);
		int n = all_nodes(cur, nd, 0, seen), i, j, nh;
		nid best = 0;
		float bc = cost;

		free(seen);
		for (i = 0; i < n; i++) {
			nh = swar_of(f, nd[i], hit, 16);
			for (j = 0; j < nh; j++) {
				const struct swhit *h = &hit[j];
				nid to = h->swap ? swar_form(h->op, h->b, h->a,
				    h->w) : swar_form(h->op, h->a, h->b, h->w);
				nid cand = sg_replaced(cur, nd[i], mk_cast(
				    N.ty[nd[i]], to));
				float c;
				if (!cand)
					continue;
				c = fuse_expr_cost(m, cand);
				if (c >= bc)
					continue;
				/* the samples guessed; the domain decides */
				sb_clear(&w);
				if (tries++ >= SW_PROOFS ||
				    equiv_decide(f, f->body, cand, &w, &pr))
					continue;
				how = pr.how;
				bc = c;
				best = cand;
				operand(pa, sizeof pa, h->swap ? h->b : h->a);
				operand(pb, sizeof pb, h->swap ? h->a : h->b);
				sb_clear(&best_why);
				sb_printf(&best_why, "%d x %d-bit lanes, ",
				    32 / h->w, h->w);
				sb_printf(&best_why, lops[h->op].name, pa, pb);
				sb_printf(&best_why, " in each: %s",
				    lops[h->op].how);
			}
		}
		if (!best)
			break;
		sb_printf(&why, "%s\n", sb_str(&best_why));
		cur = best;
		cost = bc;
	}
	free(nd);
	if (cur == f->body)
		goto out;
	chk = how == EQ_EXHAUSTIVE ? "checked on every point of the domain" :
	    how == EQ_SAT ? "checked by SAT" :
	    "checked on a sample of the domain";
	printf("  packed lanes under %s (every lane in one word-wide "
	    "operation; cost %g as written):\n", mn, (double)before);
	printf("  cost %-3g %s\n", (double)cost, shared_str(cur, 400));
	for (s = sb_str(&why); *s; s = *e ? e + 1 : e) {
		e = strchr(s, '\n');
		if (!e)
			e = s + strlen(s);
		printf("           %.*s\n", (int)(e - s), s);
	}
	printf("           %s\n", chk);
	rep_obj("packed_lanes");
	rep_str("model", mn);
	rep_num("cost_before", before);
	rep_num("cost", cost);
	rep_expr("expr", cur);
	rep_str("derivation", sb_str(&why));
	rep_str("check", how == EQ_EXHAUSTIVE ? "exhaustive" : how == EQ_SAT ?
	    "sat" : "sampled");
	rep_end();
out:
	free(why.p);
	free(best_why.p);
	free(w.p);
}
