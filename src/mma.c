/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * mma.c: a warp's matrix product, D = A B + C on fragments spread over
 * the 32 lanes as PTX's mma.sync.aligned.m16n8k16 (and m16n8k8) .row.col
 * lays them out, with f16 A and B and f32 or f16 C and D, for a GPU with
 * subgroups and no matrix unit.
 *
 *   mma [m16n8k16 | m16n8k8] [f32 | f16] [model M] [all] [validate]
 *       [nofile] [seed N] [seconds N]
 *
 * Both searches run to completion by default (fully deterministic result).
 * Pass `seconds N` to cap a search by wall clock; when a search is cut short
 * it says so, rather than reporting the best plan as if the search had
 * finished.
 *
 * Nothing here names a strategy.  A dataflow is a choice of where each
 * product is formed and how its operands get there:
 *
 *   - output-stationary: a lane forms the products of its own elements
 *     of D, fetching the words of A's rows and B's columns it needs; the
 *     order of the k words in each lane is a search variable (a xor of
 *     the lane's bits), which lets two lanes that need the same words
 *     fetch half each and swap the rest;
 *   - A- or B-stationary: a lane forms every product its own words of A
 *     (or B) take part in, fetching the other operand's words, and the
 *     partial sums meet at D's owners by a butterfly over the lane bits
 *     that tell the partial sums apart; which partial sum sits in which
 *     register is a search variable (a xor of the lane's bits), as a
 *     butterfly needs the same registers sent from every lane;
 *
 * and for each operand fetched: by subgroup exchanges, the words packed
 * or unpacked to f32 first, or through a slot of shared memory (two
 * layouts).  Every fetch looks for a word already at hand in another
 * lane that a xor shuffle reaches, and each exchange is classified by
 * the lane map it is (a xor of a constant, a broadcast within the quad,
 * anything else as a shuffle by an index computed from the lane's bits).
 * The model's prices decide: sg_shuffle, sg_shuffle_xor and sg_quad
 * (both a shuffle where the model does not list them), load_shared,
 * store_shared and bank_conflict over its banks, sg_barrier (barrier
 * where it is absent), conv for an unpack or a pack of f16, fmul for a
 * product or a multiply-add, fdot2 (two f16 products added into an f32,
 * absent unless listed) and fadd.
 *
 * The chosen dataflow is run on symbols over all 32 lanes: each element
 * of D must come out as the sum of exactly its K products and its C,
 * each once; that, the exactness of a product of two f16 in f32, and the
 * longest chain of roundings an element's terms go through (h) give the
 * bound |D - (A B + C)| <= gamma_h (sum_k |A_ik B_kj| + |C_ij|).  It is
 * then run on random fragments in f32 against the exact sums, and the
 * bound checked (a secondary check, sampled).
 */
#include <math.h>
#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#include "ucas.h"
#include "memk.h"
#include "mma.h"
#include "report.h"

void deliver_file(const char *name, const char *type, const char *data,
    size_t n, int nofile);	/* anim.c */

/* The search is a bounded hunt, not an unbounded proof: a full run is tens of
 * seconds, far past a normal client's timeout.  A default wall-clock deadline
 * cuts it short honestly (see MMA_BUDGET_S and mma_run) instead of leaving the
 * caller waiting, or reporting "no dataflow" as if the search had finished. */
/* The default is an unbounded, fully deterministic search: a wall-clock cut
 * makes the result depend on machine speed (sometimes full, sometimes cut),
 * which breaks reproducibility and can hide a cheaper dataflow.  Clients that
 * need a deadline pass `seconds N` (or the protocol-level `timeout_s`). */
#define MMA_BUDGET_S	0.0
#define MMA_BUDGET_MAX	120.0

static jmp_buf mma_deadline_jmp;
static struct timespec mma_deadline_end;
static int mma_deadline_set, mma_stopped;
static double mma_budget_s = MMA_BUDGET_S;

/* Abort the current search when its wall-clock budget is up.  Called both
 * per plan (keep) and from the blocked search's inner loops, which can do a
 * lot of expensive reduction work before a plan reaches keep(). */
static void
mma_check_deadline(void)
{
	struct timespec now;

	if (!mma_deadline_set)
		return;
	clock_gettime(CLOCK_MONOTONIC, &now);
	if (now.tv_sec > mma_deadline_end.tv_sec ||
	    (now.tv_sec == mma_deadline_end.tv_sec &&
	    now.tv_nsec >= mma_deadline_end.tv_nsec)) {
		mma_stopped = 1;
		longjmp(mma_deadline_jmp, 1);
	}
}

#define ML 32		/* the lanes of a warp */
#define MAXI 2048	/* a plan's instructions */
#define MAXT 512	/* its lane maps and address maps */
#define MAXF 1024	/* the values it knows the lanes' contents of */
#define MAXS 128	/* partial sums a lane, in the stationary flows */
#define MAXTILE 4	/* HMMAs in a tile */
#define MAXRA 16	/* registers of A in a tile, of B, of C */
#define MAXRB 8
#define MAXRC 16
#define MAXOUT 16	/* D's elements a lane */
/* products and C's elements: M K N <= 16 16 8 MAXTILE, M N <= 16 8 4 */
#define NTERM (16 * 16 * 8 * MAXTILE + 16 * 8 * MAXTILE)
#define NWDMAX ((NTERM + 63) / 64)

/* ---- the fragments ---- */

struct pos {
	int8_t lane, reg, half;
};

struct shape {
	char name[24];
	int M, N, K, f32, k16;
	int tm, tn, tk;			/* the tile's HMMAs along M, N, K */
	int na, nb, nc, nout;		/* registers of A, B, C (and D); D's
					 * elements a lane */
	int nwd;			/* words of a set of terms */
	int8_t ar[ML][MAXRA][2], ak[ML][MAXRA][2];	/* A: row, column (k) */
	int8_t bk[ML][MAXRB][2], bc[ML][MAXRB][2];	/* B: row (k), column */
	int8_t cr[ML][MAXRC][2], cc[ML][MAXRC][2];	/* C, D: row, column */
	struct pos aown[64][64], bown[64][32], cown[64][32];
};

/*
 * PTX's layouts with g = lane / 4, t = lane % 4: A's register j holds row
 * g + 8 (j & 1), columns 2t + 8 (j >> 1) and the next; B's register j
 * column g, rows 2t + 8j and the next; an f32 C's register j row
 * g + 8 (j >> 1), column 2t + (j & 1); an f16 C's register j row g + 8j,
 * columns 2t and 2t + 1.  A tile of tm x tn x tk HMMAs is one product of
 * 16 tm x 8 tn x K tk: the HMMA (mi, ni, ki) reads A's registers from
 * (mi tk + ki) na0, B's from (ki tn + ni) nb0, and accumulates into C's
 * from (mi tn + ni) nc0, the k tiles chained (D of one the next's C).
 */
static void
shape_build(struct shape *s, int k16, int f32, int tm, int tn, int tk)
{
	int l, r, h, na0 = k16 ? 4 : 2, nb0 = k16 ? 2 : 1, nc0 = f32 ? 4 : 2;
	int K0 = k16 ? 16 : 8, mi, ni, ki;

	memset(s, 0, sizeof *s);
	s->tm = tm;
	s->tn = tn;
	s->tk = tk;
	s->k16 = k16;
	s->M = 16 * tm;
	s->N = 8 * tn;
	s->K = K0 * tk;
	s->f32 = f32;
	s->na = na0 * tm * tk;
	s->nb = nb0 * tk * tn;
	s->nc = nc0 * tm * tn;
	s->nout = 4 * tm * tn;
	s->nwd = (s->M * s->K * s->N + s->M * s->N + 63) / 64;
	if (tm * tn * tk == 1)
		snprintf(s->name, sizeof s->name, "m16n8k%d", K0);
	else
		snprintf(s->name, sizeof s->name, "m%dn%dk%d", s->M, s->N, s->K);
	for (l = 0; l < ML; l++) {
		int g = l >> 2, t = l & 3;
		for (mi = 0; mi < tm; mi++)
			for (ki = 0; ki < tk; ki++)
				for (r = 0; r < na0; r++)
					for (h = 0; h < 2; h++) {
						int R = (mi * tk + ki) * na0 + r;
						struct pos p = {(int8_t)l, (int8_t)R, (int8_t)h};
						s->ar[l][R][h] = (int8_t)(16 * mi + g + 8 * (r & 1));
						s->ak[l][R][h] = (int8_t)(K0 * ki + 2 * t + h +
						    8 * (r >> 1));
						s->aown[s->ar[l][R][h]][s->ak[l][R][h]] = p;
					}
		for (ki = 0; ki < tk; ki++)
			for (ni = 0; ni < tn; ni++)
				for (r = 0; r < nb0; r++)
					for (h = 0; h < 2; h++) {
						int R = (ki * tn + ni) * nb0 + r;
						struct pos p = {(int8_t)l, (int8_t)R, (int8_t)h};
						s->bk[l][R][h] = (int8_t)(K0 * ki + 2 * t + h + 8 * r);
						s->bc[l][R][h] = (int8_t)(8 * ni + g);
						s->bown[s->bk[l][R][h]][s->bc[l][R][h]] = p;
					}
		for (mi = 0; mi < tm; mi++)
			for (ni = 0; ni < tn; ni++)
				for (r = 0; r < nc0; r++)
					for (h = 0; h < (f32 ? 1 : 2); h++) {
						int R = (mi * tn + ni) * nc0 + r;
						struct pos p = {(int8_t)l, (int8_t)R, (int8_t)h};
						s->cr[l][R][h] = (int8_t)(16 * mi + (f32 ? g + 8 *
						    (r >> 1) : g + 8 * r));
						s->cc[l][R][h] = (int8_t)(8 * ni + (f32 ? 2 * t +
						    (r & 1) : 2 * t + h));
						s->cown[s->cr[l][R][h]][s->cc[l][R][h]] = p;
					}
	}
}

/* ---- plans ---- */

enum { I_IN, I_XCH, I_STS, I_BAR, I_LDS, I_UNPK, I_MUL, I_FMA, I_DOT2,
	I_FADD, I_OUT, I_SEL };
enum { VW, VH, VF, VNONE };	/* a word of two f16, one as f32, an f32 */
enum { XK_XOR, XK_QUAD, XK_SHUF };
enum { OA, OB, OC };		/* operands */
enum { WA, WB, WP };		/* what an exchange carries: A, B, a partial sum */
enum { T_SHUF, T_SHUFU, T_STAGE0, T_STAGE1, T_N };
enum { F_OUT, F_ASTAT, F_BSTAT, F_BLOCK, F_N };

struct ins {
	uint8_t op, vt, opnd, reg, half, xk, what;
	int16_t a, b, c, tab, xm;
	/* a shared access widened: the leader's width (2, 4; 0 none), a
	 * member's leader (+1; 0 none) and component */
	uint8_t vw, vc;
	int16_t vl;
};

struct mplan {
	int n, nt;
	/* what it is */
	int fam, trans[2], perm[5], order[5], nord, q[5];
	int lb[5], sk[5];	/* F_BLOCK: the lane bits' iteration bits, skew */
	int bad;
	double cost;
	/* the tally */
	int nx[3][3], nlds, nsts, nbar, nunpk, nfma, ndot2, nfadd, npack, nsel;
	int nidx, idxops, maxdeg, nwide;
	/* from the proof */
	int depth, proved;
	/* (last: not cleared for each plan tried) */
	struct ins in[MAXI];
	int16_t tab[MAXT][ML];
};

struct bld {
	struct mplan *p;
	const struct shape *s;
	const struct model *m;
	int fdot2;
	int own[3][MAXRA];
	int nf, ne;
	int staged[2];
	int noreuse;	/* the reference: each word fetched from its owner */
	/* (last: not cleared for each plan tried) */
	int16_t fv[MAXF];
	int32_t fid[MAXF][ML];
	/* the values at hand by contents: (value k, lane) holding ident x,
	 * a list from ihead[x] (-1 ends it; igen[x] != gen: empty) */
	int32_t ihead[8192], enext[MAXF * ML];
	uint32_t igen[8192], gen;
	int16_t ek[MAXF * ML];
	int8_t el[MAXF * ML];
	int16_t vk[MAXI];	/* a value's k, or -1 */
};

static int unp[MAXI][2];	/* a word's unpacked halves, -1 */

static double
price(const struct model *m, int k)
{
	return (double)model_cost(m, k);
}

static int
avail(const struct model *m, int k)
{
	return cclass_available(m, k);
}

/* an exchange's price: a xor within the quad is a quad swap too */
static double
xch_price(const struct model *m, int xk, int xm)
{
	double best = price(m, C_SG_SHUF);

	if (xk == XK_XOR && avail(m, C_SG_SHUFXOR) &&
	    price(m, C_SG_SHUFXOR) < best)
		best = price(m, C_SG_SHUFXOR);
	if ((xk == XK_QUAD || (xk == XK_XOR && xm < 4)) && avail(m, C_SG_QUAD)
	    && price(m, C_SG_QUAD) < best)
		best = price(m, C_SG_QUAD);
	return best;
}

static int
barrier_class(const struct model *m)
{
	return avail(m, C_SG_BARRIER) ? C_SG_BARRIER : C_BARRIER;
}

/* t[l] = t[0] ^ (the xor of t[1 << i] ^ t[0] over l's bits i) */
static int
affine(const int16_t *t)
{
	int l, i, v;

	for (l = 0; l < ML; l++) {
		v = t[0];
		for (i = 0; i < 5; i++)
			if (l >> i & 1)
				v ^= t[0] ^ t[1 << i];
		if (v != t[l])
			return 0;
	}
	return 1;
}

/* a lane map's kind: -1 the lane itself, -2 not affine */
static int
classify(const int16_t *t, int *xm)
{
	int l, m = t[0], q = t[0] & 3, isx = 1, isq = 1;

	for (l = 0; l < ML; l++) {
		if ((t[l] ^ l) != m)
			isx = 0;
		if ((t[l] & ~3) != (l & ~3) || (t[l] & 3) != q)
			isq = 0;
	}
	if (isx) {
		*xm = m;
		return m ? XK_XOR : -1;
	}
	if (isq) {
		*xm = q;
		return XK_QUAD;
	}
	*xm = 0;
	return affine(t) ? XK_SHUF : -2;
}

/* the terms of an affine map as an expression of the lane: for each
 * shift d, the lane bits that land d places up; *ops: its operations */
struct aff {
	int c, nt, mask[24], sh[24];
};

static void
aff_terms(const int16_t *t, struct aff *a)
{
	int i, j, d, mask[32];

	memset(mask, 0, sizeof mask);
	a->c = t[0];
	a->nt = 0;
	for (i = 0; i < 5; i++) {
		int col = t[1 << i] ^ t[0];
		for (j = 0; j < 16; j++)
			if (col >> j & 1)
				mask[j - i + 8] |= 1 << i;
	}
	for (d = 0; d < 32; d++)
		if (mask[d]) {
			a->mask[a->nt] = mask[d];
			a->sh[a->nt++] = d - 8;
		}
}

static void
aff_text(struct sbuf *b, const int16_t *t)
{
	struct aff a;
	int k, l;

	aff_terms(t, &a);
	/* the expression as written gives the map, in every lane */
	for (l = 0; l < ML; l++) {
		int v = a.c;
		for (k = 0; k < a.nt; k++)
			v ^= a.sh[k] >= 0 ? (l & a.mask[k]) << a.sh[k] :
			    (l & a.mask[k]) >> -a.sh[k];
		if (v != t[l])
			fail("mma: internal error, lane %d's index is %d, not %d", l,
			    v, t[l]);
	}
	for (k = 0; k < a.nt; k++) {
		const char *x = k ? " ^ " : "";
		int paren = a.nt > 1 || a.c;
		if (a.mask[k] == 31 && a.sh[k] == 0)
			sb_printf(b, "%slane", x);
		else if (a.sh[k] == 0)
			sb_printf(b, "%s(lane & %du)", x, a.mask[k]);
		else if (a.mask[k] == 31 && !paren)
			sb_printf(b, "%slane %s %du", x, a.sh[k] > 0 ? "<<" : ">>",
			    abs(a.sh[k]));
		else
			sb_printf(b, "%s((lane & %du) %s %du)", x, a.mask[k],
			    a.sh[k] > 0 ? "<<" : ">>", abs(a.sh[k]));
	}
	if (a.nt == 0)
		sb_printf(b, "%du", a.c);
	else if (a.c)
		sb_printf(b, " ^ %du", a.c);
}

/* its price: the ands, shifts and xors */
static double
aff_price(const struct model *m, const int16_t *t, int *ops)
{
	struct aff a;
	int k, nl = 0, ns = 0;

	aff_terms(t, &a);
	for (k = 0; k < a.nt; k++) {
		nl += a.mask[k] != 31;
		ns += a.sh[k] != 0;
	}
	nl += a.nt > 1 ? a.nt - 1 : 0;
	nl += a.nt && a.c;
	*ops = nl + ns;
	return nl * price(m, C_LOGIC) + ns * price(m, C_SHIFT);
}

static int
tab_add(struct mplan *p, const int16_t *t)
{
	int k;

	for (k = 0; k < p->nt; k++)
		if (!memcmp(p->tab[k], t, sizeof p->tab[k]))
			return k;
	if (p->nt == MAXT) {
		p->bad = 1;
		return 0;
	}
	memcpy(p->tab[p->nt], t, sizeof p->tab[0]);
	return p->nt++;
}

static int
emit(struct bld *b, int op, int vt, int a, int bb, int c)
{
	struct mplan *p = b->p;
	struct ins *x;

	if (p->n == MAXI) {
		p->bad = 1;
		return 0;
	}
	x = &p->in[p->n];
	memset(x, 0, sizeof *x);
	x->op = (uint8_t)op;
	x->vt = (uint8_t)vt;
	x->a = (int16_t)a;
	x->b = (int16_t)bb;
	x->c = (int16_t)c;
	x->tab = -1;
	unp[p->n][0] = unp[p->n][1] = -1;
	return p->n++;
}

/* a value's contents in every lane: operand, lane, register, half + 1 */
static int32_t
mkid(int opnd, int lane, int reg, int hs)
{
	return opnd << 11 | lane << 6 | reg << 2 | hs;
}

static void
know(struct bld *b, int v, const int32_t *id)
{
	if (b->nf == MAXF) {
		b->p->bad = 1;
		return;
	}
	int l;

	b->fv[b->nf] = (int16_t)v;
	b->vk[v] = (int16_t)b->nf;
	memcpy(b->fid[b->nf], id, sizeof b->fid[0]);
	for (l = 0; l < ML; l++) {
		int x = id[l];
		if (b->igen[x] != b->gen) {
			b->igen[x] = b->gen;
			b->ihead[x] = -1;
		}
		b->ek[b->ne] = (int16_t)b->nf;
		b->el[b->ne] = (int8_t)l;
		b->enext[b->ne] = b->ihead[x];
		b->ihead[x] = b->ne++;
	}
	b->nf++;
}

/* the first entry of ident x's list, -1 */
static int
first(const struct bld *b, int32_t x)
{
	return b->igen[x] == b->gen ? b->ihead[x] : -1;
}

static const int32_t *
known(const struct bld *b, int v)
{
	return v < MAXI && b->vk[v] >= 0 ? b->fid[b->vk[v]] : NULL;
}

static int
own(struct bld *b, int opnd, int reg)
{
	int v, l;
	int32_t id[ML];

	if (b->own[opnd][reg] >= 0)
		return b->own[opnd][reg];
	v = emit(b, I_IN, opnd == OC && b->s->f32 ? VF : VW, -1, -1, -1);
	b->p->in[v].opnd = (uint8_t)opnd;
	b->p->in[v].reg = (uint8_t)reg;
	b->own[opnd][reg] = v;
	if (opnd != OC) {
		for (l = 0; l < ML; l++)
			id[l] = mkid(opnd, l, reg, 0);
		know(b, v, id);
	}
	return v;
}

static int
unpack(struct bld *b, int w, int h)
{
	const int32_t *wid;
	int32_t id[ML];
	int v, l;

	if (unp[w][h] >= 0)
		return unp[w][h];
	/* an f16 C's half is a sum (of one term) */
	v = emit(b, I_UNPK, b->p->in[w].op == I_IN && b->p->in[w].opnd == OC ?
	    VF : VH, w, -1, -1);
	b->p->in[v].half = (uint8_t)h;
	b->p->nunpk++;
	unp[w][h] = v;
	if ((wid = known(b, w)) != NULL) {
		for (l = 0; l < ML; l++)
			id[l] = wid[l] | (h + 1);
		know(b, v, id);
	}
	return v;
}

/* the address of an operand's register of a lane in a warp's slot */
static int
stage_addr(const struct bld *b, int opnd, int lane, int reg)
{
	const struct shape *s = b->s;
	int base = opnd == OA ? 0 : ML * s->na, nr = opnd == OA ? s->na : s->nb;

	return b->p->trans[opnd] == T_STAGE0 ? base + reg * ML + lane :
	    base + lane * nr + reg;
}

static int
stage_words(const struct shape *s)
{
	return ML * (s->na + s->nb);
}

/* the most distinct words the lanes put on one bank */
static int
degree(const struct model *m, const int16_t *a)
{
	int banks = (int)m->par[MP_BANKS], bb = (int)m->par[MP_BANK_BYTES];
	int l, j, d, best = 1;

	if (banks <= 0 || bb <= 0)
		return 1;
	for (l = 0; l < ML; l++) {
		int bank = a[l] * 4 / bb % banks;
		d = 0;
		for (j = 0; j < ML; j++) {
			int k;
			if (a[j] * 4 / bb % banks != bank)
				continue;
			for (k = 0; k < j; k++)
				if (a[k] == a[j])
					break;
			if (k == j)
				d++;
		}
		if (d > best)
			best = d;
	}
	return best;
}

static double
lds_price(const struct model *m, int deg)
{
	return price(m, C_LOAD_SHARED) + (deg - 1) * price(m, C_BANK_CONFLICT);
}

/* the stores of the staged operands, between two barriers */
static void
stage_setup(struct bld *b)
{
	const struct shape *s = b->s;
	int o, r, l, v;

	if (!b->staged[OA] && !b->staged[OB])
		return;
	emit(b, I_BAR, VNONE, -1, -1, -1);
	for (o = OA; o <= OB; o++) {
		if (!b->staged[o])
			continue;
		for (r = 0; r < (o == OA ? s->na : s->nb); r++) {
			int16_t t[ML];
			int w = own(b, o, r);
			for (l = 0; l < ML; l++)
				t[l] = (int16_t)stage_addr(b, o, l, r);
			v = emit(b, I_STS, VNONE, w, -1, -1);
			b->p->in[v].tab = (int16_t)tab_add(b->p, t);
			b->p->in[v].opnd = (uint8_t)o;
		}
	}
	emit(b, I_BAR, VNONE, -1, -1, -1);
}

/* a way to get the wanted words into the lanes of a mask */
struct piece {
	double cost;
	int kind;	/* -1 at hand in the lane, XK_*, 99 a load */
	int xm, from, deg;	/* from: a value at hand (else the register) */
};

/*
 * The cheapest piece for the lanes of mask: the register itself (read
 * from lane sl[l] by a xor or a quad's broadcast, or over the whole warp
 * by a shuffle or a load), or a value at hand read from lane l ^ q.
 */
static void
piece_best(struct bld *b, int opnd, const int8_t *sl, const int8_t *rg,
    int hs, const int32_t *id, uint32_t mask, struct piece *pc)
{
	int l, k, q, e, l0 = __builtin_ctz(mask), uni = 1, isx = 1, isq = 1;
	int xm = sl[l0] ^ l0, qi = sl[l0] & 3, full = mask == 0xffffffffu;
	double pr;

	pc->cost = INFINITY;
	pc->from = -1;
	for (l = 0; l < ML; l++) {
		if (!(mask >> l & 1))
			continue;
		uni &= rg[l] == rg[l0];
		isx &= (sl[l] ^ l) == xm;
		isq &= (sl[l] & ~3) == (l & ~3) && (sl[l] & 3) == qi;
	}
	if (uni && isx && xm == 0) {
		pc->cost = 0;
		pc->kind = -1;
		return;
	}
	if (uni && b->staged[opnd] && full) {
		int16_t t[ML];
		for (l = 0; l < ML; l++)
			t[l] = (int16_t)stage_addr(b, opnd, sl[l], rg[0]);
		pc->deg = degree(b->m, t);
		pc->cost = lds_price(b->m, pc->deg) + (hs ? price(b->m, C_CONV) : 0);
		pc->kind = 99;
	} else if (uni && !b->staged[opnd]) {
		int16_t t[ML];
		for (l = 0; l < ML; l++)
			t[l] = sl[l];
		if (isx) {
			pc->cost = xch_price(b->m, XK_XOR, xm);
			pc->kind = XK_XOR;
			pc->xm = xm;
		} else if (isq) {
			pc->cost = xch_price(b->m, XK_QUAD, qi);
			pc->kind = XK_QUAD;
			pc->xm = qi;
		} else if (full && affine(t)) {
			pc->cost = xch_price(b->m, XK_SHUF, 0);
			pc->kind = XK_SHUF;
			pc->xm = 0;
		}
	}
	if (b->noreuse)
		return;
	for (e = first(b, id[l0]); e >= 0; e = b->enext[e]) {
		const int32_t *f;
		k = b->ek[e];
		f = b->fid[k];
		q = b->el[e] ^ l0;	/* lane l reads lane l ^ q */
		{
			for (l = l0 + 1; l < ML; l++)
				if ((mask >> l & 1) && f[l ^ q] != id[l])
					break;
			if (l < ML)
				continue;
			pr = q ? xch_price(b->m, XK_XOR, q) : 0;
			if (pr < pc->cost) {
				pc->cost = pr;
				pc->kind = q ? XK_XOR : -1;
				pc->xm = q;
				pc->from = b->fv[k];
			}
		}
	}
}

/* the piece's value (its lanes outside the mask hold anything) */
static int
piece_make(struct bld *b, int opnd, const int8_t *sl, const int8_t *rg,
    int hs, const struct piece *pc)
{
	struct mplan *p = b->p;
	int16_t t[ML];
	int l, v, from = pc->from;

	if (pc->kind == 99) {
		int32_t wid[ML];
		for (l = 0; l < ML; l++) {
			t[l] = (int16_t)stage_addr(b, opnd, sl[l], rg[0]);
			wid[l] = mkid(opnd, sl[l], rg[0], 0);
		}
		v = emit(b, I_LDS, VW, -1, -1, -1);
		p->in[v].tab = (int16_t)tab_add(p, t);
		p->in[v].opnd = (uint8_t)opnd;
		p->nlds++;
		if (pc->deg > p->maxdeg)
			p->maxdeg = pc->deg;
		know(b, v, wid);
		return hs ? unpack(b, v, hs - 1) : v;
	}
	if (from < 0) {
		from = own(b, opnd, rg[0]);	/* the same in every lane here */
		if (hs)
			from = unpack(b, from, hs - 1);
		if (pc->kind == -1)
			return from;
		for (l = 0; l < ML; l++)
			t[l] = (int16_t)(pc->kind == XK_XOR ? l ^ pc->xm : pc->kind ==
			    XK_QUAD ? (l & ~3) | pc->xm : sl[l]);
	} else {
		if (pc->kind == -1)
			return from;
		for (l = 0; l < ML; l++)
			t[l] = (int16_t)(l ^ pc->xm);
	}
	v = emit(b, I_XCH, p->in[from].vt, from, -1, -1);
	p->in[v].xk = (uint8_t)pc->kind;
	p->in[v].xm = (int16_t)pc->xm;
	p->in[v].tab = (int16_t)tab_add(p, t);
	p->in[v].what = (uint8_t)(opnd == OA ? WA : WB);
	p->nx[pc->kind][opnd == OA ? WA : WB]++;
	return v;
}

/*
 * The value whose lane l holds operand opnd's register rg[l] of lane
 * sl[l] (half hs - 1, or the word for hs 0): one at hand, the cheapest
 * piece for the whole warp, or two pieces for the lanes either side of a
 * lane bit and a select between them.
 */
static int
fetch(struct bld *b, int opnd, const int8_t *sl, const int8_t *rg, int hs)
{
	struct mplan *p = b->p;
	struct piece whole, half[2], bh[2];
	int32_t id[ML];
	int l, k, bit, bb = -1, v;
	double best, sel = price(b->m, C_SELECT);

	for (l = 0; l < ML; l++)
		id[l] = mkid(opnd, sl[l], rg[l], hs);
	for (k = first(b, id[0]); k >= 0; k = b->enext[k])
		if (b->el[k] == 0 && !memcmp(b->fid[b->ek[k]], id, sizeof id))
			return b->fv[b->ek[k]];
	piece_best(b, opnd, sl, rg, hs, id, 0xffffffffu, &whole);
	best = whole.cost;
	for (bit = 0; bit < 5 && !b->noreuse; bit++) {
		uint32_t m1 = 0;
		for (l = 0; l < ML; l++)
			if (l >> bit & 1)
				m1 |= 1u << l;
		piece_best(b, opnd, sl, rg, hs, id, ~m1, &half[0]);
		piece_best(b, opnd, sl, rg, hs, id, m1, &half[1]);
		if (half[0].cost + half[1].cost + sel < best) {
			best = half[0].cost + half[1].cost + sel;
			bb = bit;
			bh[0] = half[0];
			bh[1] = half[1];
		}
	}
	if (!isfinite(best)) {
		p->bad = 1;
		return 0;
	}
	if (bb < 0)
		v = piece_make(b, opnd, sl, rg, hs, &whole);
	else {
		int8_t s2[2][ML], r2[2][ML];
		int v0, v1, h;
		/* each piece built from the lanes it covers */
		for (h = 0; h < 2; h++)
			for (l = 0; l < ML; l++) {
				int src = (l >> bb & 1) == h ? l : (l ^ 1 << bb);
				s2[h][l] = sl[src];
				r2[h][l] = rg[src];
			}
		v0 = piece_make(b, opnd, s2[0], r2[0], hs, &bh[0]);
		v1 = piece_make(b, opnd, s2[1], r2[1], hs, &bh[1]);
		v = emit(b, I_SEL, p->in[v0].vt, v0, v1, -1);
		p->in[v].xm = (int16_t)bb;
		p->nsel++;
	}
	if (p->bad)
		return 0;
	know(b, v, id);
	return v;
}

/* acc + x y, x and y one f16 each (acc < 0: none) */
static int
fma1(struct bld *b, int acc, int x, int y)
{
	b->p->nfma++;
	return acc < 0 ? emit(b, I_MUL, VF, x, y, -1) :
	    emit(b, I_FMA, VF, x, y, acc);
}

/* acc + the dot product of words x and y */
static int
dotw(struct bld *b, int acc, int x, int y)
{
	int h;

	if (b->fdot2 && b->p->in[x].vt == VW && b->p->in[y].vt == VW) {
		b->p->ndot2++;
		return emit(b, I_DOT2, VF, x, y, acc);
	}
	for (h = 0; h < 2; h++)
		acc = fma1(b, acc, b->p->in[x].vt == VW ? unpack(b, x, h) : x,
		    b->p->in[y].vt == VW ? unpack(b, y, h) : y);
	return acc;
}

/* half h of an operand's word: x (the lane's own), else fetched, as a
 * word (u 0) or as the half itself (u 1) */
static int
term(struct bld *b, int opnd, int x, const int8_t *sl, const int8_t *rg,
    int u, int h)
{
	if (x >= 0)
		return unpack(b, x, h);
	return u ? fetch(b, opnd, sl, rg, h + 1) :
	    unpack(b, fetch(b, opnd, sl, rg, 0), h);
}

/* acc + the dot product of a word of A and a word of B, each the lane's
 * own (xa, xb) or fetched */
static int
dot_fetch(struct bld *b, int acc, int xa, const int8_t *sla,
    const int8_t *rga, int ua, int xb, const int8_t *slb, const int8_t *rgb,
    int ub)
{
	int h;

	if (!ua && !ub) {
		int wa = xa >= 0 ? xa : fetch(b, OA, sla, rga, 0);
		int wb = xb >= 0 ? xb : fetch(b, OB, slb, rgb, 0);
		if (b->p->bad)
			return 0;
		return dotw(b, acc, wa, wb);
	}
	for (h = 0; h < 2 && !b->p->bad; h++) {
		int ha = term(b, OA, xa, sla, rga, ua, h);
		int hb = term(b, OB, xb, slb, rgb, ub, h);
		acc = fma1(b, acc, ha, hb);
	}
	return acc;
}

/* the lane's C as f32, for output element o */
static int
c_value(struct bld *b, int o)
{
	if (b->s->f32)
		return own(b, OC, o);
	return unpack(b, own(b, OC, o >> 1), o & 1);
}

/* D's register (and half) of output element o */
static void
d_out(struct bld *b, const int *val)
{
	int o, v;

	if (b->s->f32)
		for (o = 0; o < b->s->nc; o++) {
			v = emit(b, I_OUT, VNONE, val[o], -1, -1);
			b->p->in[v].reg = (uint8_t)o;
		}
	else
		for (o = 0; o < b->s->nc; o++) {
			v = emit(b, I_OUT, VNONE, val[2 * o], val[2 * o + 1], -1);
			b->p->in[v].reg = (uint8_t)o;
			b->p->npack++;
		}
}

static void
out_elem(const struct shape *s, int o, int l, int *r, int *c)
{
	int reg = s->f32 ? o : o >> 1, h = s->f32 ? 0 : o & 1;

	*r = s->cr[l][reg][h];
	*c = s->cc[l][reg][h];
}

static int
xperm(const int *P, int l)
{
	int i, v = 0;

	for (i = 0; i < 5; i++)
		if (l >> i & 1)
			v ^= P[i];
	return v;
}

static void
bld_init(struct bld *b, struct mplan *p, const struct shape *s,
    const struct model *m)
{
	uint32_t gen = b->gen + 1;

	memset(b, 0, offsetof(struct bld, fv));
	memset(p, 0, offsetof(struct mplan, in));
	b->gen = gen;
	if (gen == 1)
		memset(b->igen, 0, sizeof b->igen);
	memset(b->vk, -1, sizeof b->vk);
	b->p = p;
	b->s = s;
	b->m = m;
	b->fdot2 = avail(m, C_FDOT2);
	memset(b->own, -1, sizeof b->own);
}

/*
 * Output-stationary: lane l sums the products of its own elements of D,
 * its k words taken in the order kw ^ perm(l).
 */
static void
gen_out(struct bld *b)
{
	const struct shape *s = b->s;
	struct mplan *p = b->p;
	int o, l, kw, nk = s->K / 2, acc[MAXOUT];
	int ua = p->trans[OA] == T_SHUFU, ub = p->trans[OB] == T_SHUFU;

	b->staged[OA] = p->trans[OA] >= T_STAGE0;
	b->staged[OB] = p->trans[OB] >= T_STAGE0;
	stage_setup(b);
	for (o = 0; o < s->nout; o++)
		acc[o] = c_value(b, o);
	for (kw = 0; kw < nk; kw++)
		for (o = 0; o < s->nout; o++) {
			int8_t sla[ML], rga[ML], slb[ML], rgb[ML];
			for (l = 0; l < ML; l++) {
				int r, c, k = 2 * (kw ^ xperm(p->perm, l));
				struct pos pa, pb;
				if (k >= s->K) {
					p->bad = 1;
					return;
				}
				out_elem(s, o, l, &r, &c);
				pa = s->aown[r][k];
				pb = s->bown[k][c];
				sla[l] = pa.lane;
				rga[l] = pa.reg;
				slb[l] = pb.lane;
				rgb[l] = pb.reg;
			}
			acc[o] = dot_fetch(b, acc[o], -1, sla, rga, ua, -1, slb, rgb, ub);
			if (p->bad)
				return;
		}
	d_out(b, acc);
}

/* ---- the stationary flows' partial sums and their butterfly ---- */

#define GS 32	/* registers in a group of the stationary operand */
#define NG 16	/* groups */

struct red {
	int ns, nst, bit[5], nsend[5];
	int16_t from[5][MAXS], to[5][MAXS];
	int16_t oslot[MAXOUT];
	int16_t ohold[MAXOUT][ML];
	int16_t key[MAXS][ML];	/* r * N + c of each partial sum */
};

/*
 * The partial sums' keys under the stationary operand X's words, the
 * other index of slot j being j ^ perm(L, l): slot (group, j).
 */
static int
stat_keys(const struct shape *s, int X, const int *L, struct red *R,
    int *grp, int *ng, int *gof)
{
	int nx = X == OA ? s->na : s->nb, ny = X == OA ? s->N : s->M;
	int r, g, l, j;

	*ng = 0;
	for (r = 0; r < nx; r++) {
		for (g = 0; g < *ng; g++) {
			int r0 = grp[g * GS];
			for (l = 0; l < ML; l++)
				if ((X == OA ? s->ar[l][r][0] != s->ar[l][r0][0] :
				    s->bc[l][r][0] != s->bc[l][r0][0]))
					break;
			if (l == ML)
				break;
		}
		if (g == *ng) {
			if (g == NG)
				return 0;
			gof[g] = 0;
			(*ng)++;
		}
		grp[g * GS + gof[g]++] = r;
	}
	R->ns = *ng * ny;
	if (R->ns > MAXS)
		return 0;
	for (g = 0; g < *ng; g++)
		for (j = 0; j < ny; j++)
			for (l = 0; l < ML; l++) {
				int y = j ^ xperm(L, l), x, rr, cc;
				if (y >= ny)
					return 0;
				x = X == OA ? s->ar[l][grp[g * GS]][0] :
				    s->bc[l][grp[g * GS]][0];
				rr = X == OA ? x : y;
				cc = X == OA ? y : x;
				R->key[g * ny + j][l] = (int16_t)(rr * s->N + cc);
			}
	return 1;
}

static int
popc(unsigned v)
{
	int n = 0;

	for (; v; v &= v - 1)
		n++;
	return n;
}

/*
 * The butterfly: the lanes that hold a key differ in the bits of bm; a
 * key is summed at red(key), the holder whose bit b of those is D's
 * owner's (Q[b] < 0) or the key's bit Q[b] (r * N + c), the bits taken
 * in the order given; each step must send the same slots from every lane
 * into the same slots.  Then each of D's elements is read from its
 * holder's slot, the same slot in every lane (an exchange when the
 * holder is another lane).
 */
static int
red_plan(const struct shape *s, struct red *R, const int *order, int nord,
    const int *Q, int *bmp)
{
	static uint32_t hold[16 * 8 * MAXTILE];
	static int16_t redl[16 * 8 * MAXTILE];
	int nkey = s->M * s->N, k, l, sl, st, i, live[MAXS];
	int bm = -1;

	/* Every expensive search path goes through here; without this check
	 * a long run of reductions can overrun the deadline by seconds. */
	mma_check_deadline();
	memset(hold, 0, sizeof hold);
	for (sl = 0; sl < R->ns; sl++)
		for (l = 0; l < ML; l++) {
			k = R->key[sl][l];
			if (hold[k] >> l & 1)
				return 0;
			hold[k] |= 1u << l;
		}
	for (k = 0; k < nkey; k++) {
		int h0 = __builtin_ctz(hold[k] | 0x80000000u), m = 0;
		struct pos d;
		if (!hold[k])
			return 0;
		for (l = 0; l < ML; l++)
			if (hold[k] >> l & 1)
				m |= l ^ h0;
		if (popc(hold[k]) != 1 << popc((unsigned)m))
			return 0;
		for (l = 0; l < ML; l++)
			if ((hold[k] >> l & 1) && ((l ^ h0) & ~m))
				return 0;
		if (bm < 0)
			bm = m;
		else if (bm != m)
			return 0;
		d = s->cown[k / s->N][k % s->N];
		{
			int b, q = 0;
			for (b = 0; b < 5; b++)
				if (m >> b & 1)
					q |= (Q == NULL || Q[b] < 0 ? d.lane >> b : k >> Q[b]) & 1
					    ? 1 << b : 0;
			redl[k] = (int16_t)((h0 & ~m) | q);
		}
	}
	*bmp = bm;
	if (nord != popc((unsigned)bm))
		return -1;
	for (i = 0; i < nord; i++)
		if (!(bm >> order[i] & 1))
			return -1;
	for (sl = 0; sl < R->ns; sl++)
		live[sl] = 1;
	R->nst = nord;
	for (st = 0; st < nord; st++) {
		int b = order[st], send[MAXS];
		R->bit[st] = b;
		R->nsend[st] = 0;
		for (sl = 0; sl < R->ns; sl++) {
			send[sl] = 0;
			if (!live[sl])
				continue;
			send[sl] = redl[R->key[sl][0]] >> b & 1;
			for (l = 1; l < ML; l++)
				if (((redl[R->key[sl][l]] ^ l) >> b & 1) != send[sl])
					return 0;
		}
		for (sl = 0; sl < R->ns; sl++) {
			int to = -1, s2;
			if (!live[sl] || !send[sl])
				continue;
			for (l = 0; l < ML; l++) {
				int kk = R->key[sl][l ^ 1 << b];
				for (s2 = 0; s2 < R->ns; s2++)
					if (live[s2] && !send[s2] && R->key[s2][l] == kk)
						break;
				if (s2 == R->ns || (to >= 0 && s2 != to))
					return 0;
				to = s2;
			}
			R->from[st][R->nsend[st]] = (int16_t)sl;
			R->to[st][R->nsend[st]++] = (int16_t)to;
		}
		for (sl = 0; sl < R->ns; sl++)
			if (send[sl])
				live[sl] = 0;
	}
	/* D's elements: the slot of each, the same in every lane */
	for (i = 0; i < s->nout; i++) {
		int slot = -1;
		for (l = 0; l < ML; l++) {
			int r, c, h;
			out_elem(s, i, l, &r, &c);
			k = r * s->N + c;
			h = redl[k];
			for (sl = 0; sl < R->ns; sl++)
				if (live[sl] && R->key[sl][h] == k)
					break;
			if (sl == R->ns || (slot >= 0 && sl != slot))
				return 0;
			slot = sl;
			R->ohold[i][l] = (int16_t)h;
		}
		R->oslot[i] = (int16_t)slot;
	}
	return 1;
}

/* the butterfly of the partial sums in val (slots), then D: each element
 * read from its holder, plus C */
static void
red_finish(struct bld *b, const struct red *R, int *val)
{
	const struct shape *s = b->s;
	struct mplan *p = b->p;
	int st, i, l;

	for (st = 0; st < R->nst; st++)
		for (i = 0; i < R->nsend[st]; i++) {
			int16_t t[ML];
			int x;
			for (l = 0; l < ML; l++)
				t[l] = (int16_t)(l ^ 1 << R->bit[st]);
			x = emit(b, I_XCH, VF, val[R->from[st][i]], -1, -1);
			p->in[x].xk = XK_XOR;
			p->in[x].xm = (int16_t)(1 << R->bit[st]);
			p->in[x].tab = (int16_t)tab_add(p, t);
			p->in[x].what = WP;
			p->nx[XK_XOR][WP]++;
			val[R->to[st][i]] = emit(b, I_FADD, VF, val[R->to[st][i]], x, -1);
			p->nfadd++;
		}
	{
		int out[MAXOUT];
		for (i = 0; i < s->nout; i++) {
			int v = val[R->oslot[i]], xm, k;
			k = classify(R->ohold[i], &xm);
			if (k == -2) {
				p->bad = 1;
				return;
			}
			if (k >= 0) {
				int x = emit(b, I_XCH, VF, v, -1, -1);
				p->in[x].xk = (uint8_t)k;
				p->in[x].xm = (int16_t)xm;
				p->in[x].tab = (int16_t)tab_add(p, R->ohold[i]);
				p->in[x].what = WP;
				p->nx[k][WP]++;
				v = x;
			}
			out[i] = emit(b, I_FADD, VF, v, c_value(b, i), -1);
			p->nfadd++;
		}
		d_out(b, out);
	}
}


/*
 * X-stationary: lane l forms the products of its own words of X with
 * the other operand's, the partial sums of one group of X's registers
 * (one row of A, one column of B) in slots j, then the butterfly.
 */
static void
gen_stat(struct bld *b, const struct red *R, int X)
{
	const struct shape *s = b->s;
	struct mplan *p = b->p;
	int Y = 1 - X, grp[NG * GS], ng, gof[NG], g, j, l, q, val[MAXS];
	int ny = X == OA ? s->N : s->M, uy = p->trans[Y] == T_SHUFU;
	struct red R2;

	b->staged[Y] = p->trans[Y] >= T_STAGE0;
	stat_keys(s, X, p->perm, &R2, grp, &ng, gof);
	stage_setup(b);
	for (g = 0; g < ng; g++)
		for (j = 0; j < ny; j++) {
			int acc = -1;
			for (q = 0; q < gof[g]; q++) {
				int rho = grp[g * GS + q], xw = own(b, X, rho);
				int8_t sy[ML], ry[ML];
				for (l = 0; l < ML; l++) {
					int y = j ^ xperm(p->perm, l), k;
					struct pos py;
					k = X == OA ? s->ak[l][rho][0] : s->bk[l][rho][0];
					py = X == OA ? s->bown[k][y] : s->aown[y][k];
					sy[l] = py.lane;
					ry[l] = py.reg;
				}
				if (X == OA)
					acc = dot_fetch(b, acc, xw, NULL, NULL, 0, -1, sy, ry,
					    uy);
				else
					acc = dot_fetch(b, acc, -1, sy, ry, uy, xw, NULL, NULL,
					    0);
				if (p->bad)
					return;
			}
			val[g * ny + j] = acc;
		}
	red_finish(b, R, val);
}

/* ---- prices ---- */

enum { FM_XOR, FM_QSWAP, FM_QBCAST, FM_SHUF };

/* how an exchange is written: the cheapest form its lane map allows */
static int
xch_form(const struct model *m, int xk, int xm, double *pr)
{
	int f = FM_SHUF;
	double best = price(m, C_SG_SHUF);

	if (xk == XK_XOR) {
		f = FM_XOR;	/* a shuffle's price where the model lists none */
		if (avail(m, C_SG_SHUFXOR) && price(m, C_SG_SHUFXOR) < best)
			best = price(m, C_SG_SHUFXOR);
		if (xm < 4 && avail(m, C_SG_QUAD) && price(m, C_SG_QUAD) < best) {
			best = price(m, C_SG_QUAD);
			f = FM_QSWAP;
		}
	} else if (xk == XK_QUAD && avail(m, C_SG_QUAD) &&
	    price(m, C_SG_QUAD) <= best) {
		best = price(m, C_SG_QUAD);
		f = FM_QBCAST;
	}
	*pr = best;
	return f;
}

/* a shuffle index already written that this one is a xor of a constant
 * from (one operation), or -1; seen[t] 1 for the index tables written */
static int
idx_parent(const struct mplan *p, int tab, const uint8_t *seen)
{
	int t, l;

	for (t = 0; t < p->nt; t++) {
		int c;
		if (seen[t] != 1 || t == tab)
			continue;
		c = p->tab[t][0] ^ p->tab[tab][0];
		for (l = 1; l < ML && (p->tab[t][l] ^ p->tab[tab][l]) == c; l++)
			;
		if (l == ML)
			return t;
	}
	return -1;
}

/* What-if: the widened accesses a model does not list, priced as one
 * 32-bit access (else they are not used). */
static int whatif_vec;

static double
vec_price(const struct model *m, int store, int w)
{
	int k = store ? (w == 4 ? C_STS_X4 : C_STS_X2) : (w == 4 ? C_LDS_X4 :
	    C_LDS_X2);

	if (avail(m, k))
		return price(m, k);
	if (whatif_vec)
		return price(m, store ? C_STORE_SHARED : C_LOAD_SHARED);
	return INFINITY;
}

/*
 * Shared accesses of consecutive words, aligned, made one access of 64 or
 * 128 bits where the model prices that below the separate ones: a load's
 * group led by its first, a store's by its last (its words are all
 * computed there).  The proof and the run see the separate accesses.
 */
static void
widen(struct mplan *p, const struct model *m)
{
	int i, j, w, l, o, d;

	for (w = 4; w >= 2; w /= 2)
		for (i = 0; i < p->n; i++) {
			struct ins *x = &p->in[i];
			int mem[4], ok = 1, st = x->op == I_STS;
			double one, sep;
			if ((x->op != I_LDS && x->op != I_STS) || x->vw || x->vl)
				continue;
			for (l = 0; l < ML && ok; l++)
				ok = p->tab[x->tab][l] % w == 0 && (p->tab[x->tab][l] & 3) ==
				    (p->tab[x->tab][0] & 3);
			if (!ok)
				continue;
			mem[0] = i;
			for (o = 1; o < w && ok; o++) {
				mem[o] = -1;
				for (j = 0; j < p->n && mem[o] < 0; j++) {
					const struct ins *y = &p->in[j];
					if (y->op != x->op || y->vw || y->vl || j == i)
						continue;
					for (l = 0; l < ML && p->tab[y->tab][l] ==
					    p->tab[x->tab][l] + o; l++)
						;
					if (l == ML)
						mem[o] = j;
				}
				ok = mem[o] >= 0;
			}
			if (!ok)
				continue;
			one = vec_price(m, st, w);
			sep = w * price(m, st ? C_STORE_SHARED : C_LOAD_SHARED);
			if (!(one < sep))
				continue;
			/* the leader: a load's first, a store's last */
			d = mem[0];
			for (o = 1; o < w; o++)
				if (st ? mem[o] > d : mem[o] < d)
					d = mem[o];
			for (o = 0; o < w; o++) {
				p->in[mem[o]].vc = (uint8_t)o;
				if (mem[o] == d)
					p->in[mem[o]].vw = (uint8_t)w;
				else
					p->in[mem[o]].vl = (int16_t)(d + 1);
			}
			/* the leader's own component: its word's place */
			(void)l;
		}
}

/* an address already computed that this one is a constant past (the
 * access's immediate offset: no operation), or -1 */
static int
addr_parent(const struct mplan *p, int tab, const uint8_t *seen)
{
	int t, l;

	for (t = 0; t < p->nt; t++) {
		int c;
		if (seen[t] != 2 || t == tab)
			continue;
		c = p->tab[tab][0] - p->tab[t][0];
		for (l = 1; l < ML && p->tab[tab][l] - p->tab[t][l] == c; l++)
			;
		if (l == ML)
			return t;
	}
	return -1;
}

static void
plan_price(struct mplan *p, const struct shape *s, const struct model *m)
{
	static uint8_t seen[MAXT];

	widen(p, m);
	{
	double c = 0, pr;
	int i, ops, lane = 0, selbits = 0;

	memset(seen, 0, sizeof seen);
	p->nidx = p->idxops = p->nsts = p->nbar = p->nwide = 0;
	for (i = 0; i < p->n; i++) {
		const struct ins *x = &p->in[i];
		switch (x->op) {
		case I_XCH:
			if (xch_form(m, x->xk, x->xm, &pr) == FM_SHUF) {
				lane = 1;
				if (!seen[x->tab]) {
					if (idx_parent(p, x->tab, seen) >= 0) {
						c += price(m, C_LOGIC);
						p->idxops++;
					} else {
						c += aff_price(m, p->tab[x->tab], &ops) +
						    price(m, C_LOGIC);
						p->idxops += ops + 1;
					}
					seen[x->tab] = 1;
					p->nidx++;
				}
			}
			c += pr;
			break;
		case I_STS:
		case I_LDS:
			lane = 1;
			if (!seen[x->tab]) {
				if (addr_parent(p, x->tab, seen) < 0) {
					c += aff_price(m, p->tab[x->tab], &ops) + price(m, C_IADD);
					p->idxops += ops + 1;
				}
				seen[x->tab] = 2;	/* an address: no shuffle index */
			}
			if (x->vl)
				break;	/* in its leader's access */
			if (x->vw) {
				c += vec_price(m, x->op == I_STS, x->vw) + (degree(m,
				    p->tab[x->tab]) - 1) * price(m, C_BANK_CONFLICT);
				p->nwide++;
				if (x->op == I_STS)
					p->nsts++;
			} else if (x->op == I_STS) {
				c += price(m, C_STORE_SHARED) + (degree(m, p->tab[x->tab]) -
				    1) * price(m, C_BANK_CONFLICT);
				p->nsts++;
			} else
				c += lds_price(m, degree(m, p->tab[x->tab]));
			break;
		case I_BAR:
			c += price(m, barrier_class(m));
			p->nbar++;
			break;
		case I_SEL:
			c += price(m, C_SELECT);
			if (!(selbits >> x->xm & 1))	/* (lane & 1 << b) != 0 */
				c += price(m, C_LOGIC) + price(m, C_CMP);
			selbits |= 1 << x->xm;
			lane = 1;
			break;
		case I_UNPK:
			c += price(m, C_CONV);
			break;
		case I_MUL:
		case I_FMA:
			c += price(m, C_FMUL);
			break;
		case I_DOT2:
			c += price(m, C_FDOT2);
			break;
		case I_FADD:
			c += price(m, C_FADD);
			break;
		case I_OUT:
			if (!s->f32)
				c += 2 * price(m, C_CONV);
			break;
		}
	}
	if (lane)
		c += price(m, C_LOGIC);		/* the lane in its 32 */
	if (p->nsts)
		c += price(m, C_SHIFT) + price(m, C_IMUL);	/* the warp's slot */
	p->cost = c;
	}
}

/* ---- the proof: the plan run on symbols over the 32 lanes ---- */

struct sv {
	uint16_t t[2];
	int16_t d;
};
static struct sv SV[MAXI][ML];
/* the sets of terms of the f32 values, nwd words a lane: grown as needed */
static uint64_t *SBP;
static size_t SBN;
static int vfx[MAXI], nwd;
#define BITS(i, l) (SBP + ((size_t)vfx[i] * ML + (size_t)(l)) * (size_t)nwd)

static uint16_t
tag(int opnd, int r, int c)
{
	return (uint16_t)(opnd << 12 | r << 6 | c);
}

static int
prod_term(const struct shape *s, int ta, int tb)
{
	if (ta >> 12 == OB) {
		int t = ta;
		ta = tb;
		tb = t;
	}
	if (ta >> 12 != OA || tb >> 12 != OB || (ta & 63) != (tb >> 6 & 63))
		return -1;
	return ((ta >> 6 & 63) * s->K + (ta & 63)) * s->N + (tb & 63);
}

static int
c_term(const struct shape *s, int r, int c)
{
	return s->M * s->K * s->N + r * s->N + c;
}

static int
bit_add(uint64_t *b, int k)
{
	if (k < 0 || b[k >> 6] >> (k & 63) & 1)
		return 0;
	b[k >> 6] |= 1ULL << (k & 63);
	return 1;
}

/*
 * 1 when every element of D comes out as exactly its K products and its
 * C, each once; *depth the most roundings any term goes through.  why:
 * the first thing wrong.
 */
static int
prove(struct mplan *p, const struct shape *s, struct sbuf *why)
{
	static struct sv mem[1024];
	static int8_t memw[1024];
	size_t need;
	int nvf = 0;
	int i, l, k, j, depth = 0;

	nwd = s->nwd;
	for (i = 0; i < p->n; i++)
		vfx[i] = p->in[i].vt == VF ? nvf++ : -1;
	need = (size_t)nvf * ML * (size_t)nwd;
	if (need > SBN) {
		uint64_t *q = realloc(SBP, need * sizeof *SBP);
		if (q == NULL)
			fail("mma: out of memory");
		SBP = q;
		SBN = need;
	}
	memset(memw, 0, sizeof memw);
	for (i = 0; i < p->n; i++) {
		const struct ins *x = &p->in[i];
		for (l = 0; l < ML; l++) {
			struct sv *v = &SV[i][l];
			uint64_t *b = x->vt == VF ? BITS(i, l) : NULL;
			if (x->vt == VF)
				memset(b, 0, (size_t)nwd * 8);
			switch (x->op) {
			case I_IN:
				for (j = 0; j < 2; j++)
					v->t[j] = x->opnd == OA ? tag(OA, s->ar[l][x->reg][j],
					    s->ak[l][x->reg][j]) : x->opnd == OB ? tag(OB,
					    s->bk[l][x->reg][j], s->bc[l][x->reg][j]) : tag(OC,
					    s->cr[l][x->reg][j], s->cc[l][x->reg][j]);
				if (x->vt == VF) {
					bit_add(b, c_term(s, s->cr[l][x->reg][0],
					    s->cc[l][x->reg][0]));
					v->d = 0;
				}
				break;
			case I_XCH:
				*v = SV[x->a][p->tab[x->tab][l]];
				if (x->vt == VF)
					memcpy(b, BITS(x->a, p->tab[x->tab][l]), (size_t)nwd * 8);
				break;
			case I_STS:
				k = p->tab[x->tab][l];
				if (memw[k]) {
					sb_printf(why, "two lanes store to word %d of the slot", k);
					return 0;
				}
				memw[k] = 1;
				mem[k] = SV[x->a][l];
				break;
			case I_LDS:
				k = p->tab[x->tab][l];
				if (!memw[k]) {
					sb_printf(why, "lane %d loads word %d, never stored", l,
					    k);
					return 0;
				}
				*v = mem[k];
				break;
			case I_UNPK:
				v->t[0] = SV[x->a][l].t[x->half];
				if (x->vt == VF) {
					bit_add(b, c_term(s, v->t[0] >> 6 & 63, v->t[0] & 63));
					v->d = 0;
				}
				break;
			case I_SEL:
				k = l >> x->xm & 1 ? x->b : x->a;
				*v = SV[k][l];
				if (x->vt == VF)
					memcpy(b, BITS(k, l), (size_t)nwd * 8);
				break;
			case I_MUL:
			case I_FMA:
			case I_DOT2:
				if (x->c >= 0)
					memcpy(b, BITS(x->c, l), (size_t)nwd * 8);
				if (x->op == I_DOT2) {
					for (j = 0; j < 2; j++)
						if (!bit_add(b, prod_term(s, SV[x->a][l].t[j],
						    SV[x->b][l].t[j])))
							goto bad;
					v->d = (int16_t)(x->c >= 0 ? SV[x->c][l].d + 2 : 1);
				} else {
					if (!bit_add(b, prod_term(s, SV[x->a][l].t[0],
					    SV[x->b][l].t[0])))
						goto bad;
					v->d = (int16_t)(x->c >= 0 ? SV[x->c][l].d + 1 : 0);
				}
				break;
			case I_FADD:
				for (k = 0; k < nwd; k++) {
					if (BITS(x->a, l)[k] & BITS(x->b, l)[k])
						goto bad;
					b[k] = BITS(x->a, l)[k] | BITS(x->b, l)[k];
				}
				v->d = (int16_t)((SV[x->a][l].d > SV[x->b][l].d ?
				    SV[x->a][l].d : SV[x->b][l].d) + 1);
				break;
			case I_OUT:
				for (j = 0; j < (s->f32 ? 1 : 2); j++) {
					uint64_t want[NWDMAX];
					int r = s->cr[l][x->reg][j], c = s->cc[l][x->reg][j];
					int src = j ? x->b : x->a, kk;
					memset(want, 0, sizeof want);
					for (kk = 0; kk < s->K; kk++)
						bit_add(want, (r * s->K + kk) * s->N + c);
					bit_add(want, c_term(s, r, c));
					if (memcmp(want, BITS(src, l), (size_t)nwd * 8)) {
						sb_printf(why, "lane %d's D[%d][%d] is not the sum "
						    "of its %d products and C", l, r, c, s->K);
						return 0;
					}
					if (SV[src][l].d > depth)
						depth = SV[src][l].d;
				}
				break;
			}
			continue;
bad:
			sb_printf(why, "instruction %d in lane %d: a product of "
			    "elements that do not meet, or a term counted twice", i, l);
			return 0;
		}
	}
	p->depth = depth;
	p->proved = 1;
	return 1;
}

/* ---- the plan run on numbers ---- */

static float
h2f(uint32_t h)
{
	int e = (int)(h >> 10 & 31), mt = (int)(h & 1023);
	float v;

	if (e == 0)
		v = ldexpf((float)mt, -24);
	else if (e == 31)
		v = mt ? NAN : INFINITY;
	else
		v = ldexpf((float)(mt + 1024), e - 25);
	return h >> 15 & 1 ? -v : v;
}

/* f32 to f16, to nearest even */
static uint32_t
f2h(float f)
{
	uint32_t sg = signbit(f) ? 0x8000u : 0;
	float a = fabsf(f), q;
	int e;

	if (isnan(f))
		return sg | 0x7e00u;
	if (a < 0x1p-14f)
		return sg | (uint32_t)rintf(a * 0x1p24f);
	frexpf(a, &e);
	q = rintf(ldexpf(a, 11 - e));
	if (q == 2048.0f) {
		q = 1024.0f;
		e++;
	}
	if (e + 14 > 30)
		return sg | 0x7c00u;
	return sg | (uint32_t)(e + 14) << 10 | ((uint32_t)q - 1024u);
}

struct nv {
	uint32_t w;
	float f;
};
static struct nv NV[MAXI][ML];

static void
run(const struct mplan *p, const struct shape *s, uint32_t A[ML][MAXRA],
    uint32_t B[ML][MAXRB], uint32_t C[ML][MAXRC], uint32_t D[ML][MAXRC])
{
	static uint32_t mem[1024];
	int i, l;

	for (i = 0; i < p->n; i++) {
		const struct ins *x = &p->in[i];
		for (l = 0; l < ML; l++) {
			struct nv *v = &NV[i][l];
			const struct nv *a = x->a >= 0 ? &NV[x->a][l] : NULL;
			const struct nv *b = x->b >= 0 ? &NV[x->b][l] : NULL;
			float acc = x->c >= 0 ? NV[x->c][l].f : 0.0f;
			switch (x->op) {
			case I_IN:
				v->w = x->opnd == OA ? A[l][x->reg] : x->opnd == OB ?
				    B[l][x->reg] : C[l][x->reg];
				memcpy(&v->f, &v->w, 4);
				break;
			case I_XCH:
				*v = NV[x->a][p->tab[x->tab][l]];
				break;
			case I_STS:
				mem[p->tab[x->tab][l]] = a->w;
				break;
			case I_LDS:
				v->w = mem[p->tab[x->tab][l]];
				break;
			case I_UNPK:
				v->f = h2f(a->w >> (16 * x->half) & 0xffff);
				break;
			case I_SEL:
				*v = l >> x->xm & 1 ? *b : *a;
				break;
			case I_MUL:
				v->f = a->f * b->f;
				break;
			case I_FMA:
				v->f = fmaf(a->f, b->f, acc);
				break;
			case I_DOT2:
				v->f = fmaf(h2f(a->w >> 16), h2f(b->w >> 16),
				    fmaf(h2f(a->w & 0xffff), h2f(b->w & 0xffff), acc));
				break;
			case I_FADD:
				v->f = a->f + b->f;
				break;
			case I_OUT:
				if (s->f32)
					memcpy(&D[l][x->reg], &a->f, 4);
				else
					D[l][x->reg] = f2h(a->f) | f2h(b->f) << 16;
				break;
			}
		}
	}
}

static uint64_t
rnd(uint64_t *st)
{
	*st ^= *st << 13;
	*st ^= *st >> 7;
	*st ^= *st << 17;
	return *st;
}

/* a random f16, normal, its exponent in -e..e */
static uint32_t
rnd_half(uint64_t *st, int e)
{
	uint64_t r = rnd(st);

	return (uint32_t)((r & 1) << 15 | (15 - e + r % (2 * e + 1)) << 10 |
	    (r >> 8 & 1023));
}

static double
gamma_n(int h)
{
	double u = 0x1p-24;

	return h * u / (1 - h * u);
}

/*
 * The plan on n random fragments against the exact sums: the largest
 * error as a fraction of the bound; *bad, the count over it.
 */
static double
check(const struct mplan *p, const struct shape *s, int n, uint64_t seed,
    int *bad)
{
	static uint32_t A[ML][MAXRA], B[ML][MAXRB], C[ML][MAXRC], D[ML][MAXRC];
	uint64_t st = seed * 0x9e3779b97f4a7c15ULL + 1;
	double worst = 0;
	int it, l, r, c, k, e = s->f32 ? 8 : 3;

	*bad = 0;
	for (it = 0; it < n; it++) {
		for (l = 0; l < ML; l++) {
			for (r = 0; r < s->na; r++)
				A[l][r] = rnd_half(&st, e) | rnd_half(&st, e) << 16;
			for (r = 0; r < s->nb; r++)
				B[l][r] = rnd_half(&st, e) | rnd_half(&st, e) << 16;
			for (r = 0; r < s->nc; r++)
				if (s->f32) {
					float f = h2f(rnd_half(&st, e)) * 4.0f;
					memcpy(&C[l][r], &f, 4);
				} else
					C[l][r] = rnd_half(&st, e) | rnd_half(&st, e) << 16;
		}
		run(p, s, A, B, C, D);
		for (r = 0; r < s->M; r++)
			for (c = 0; c < s->N; c++) {
				long double sum = 0, mag = 0, e;
				struct pos pc = s->cown[r][c];
				float got, cv;
				double bound;
				for (k = 0; k < s->K; k++) {
					struct pos pa = s->aown[r][k], pb = s->bown[k][c];
					long double t = (long double)h2f(A[pa.lane][pa.reg] >>
					    (16 * pa.half) & 0xffff) *
					    h2f(B[pb.lane][pb.reg] >> (16 * pb.half) & 0xffff);
					sum += t;
					mag += fabsl(t);
				}
				if (s->f32) {
					memcpy(&cv, &C[pc.lane][pc.reg], 4);
					memcpy(&got, &D[pc.lane][pc.reg], 4);
				} else {
					cv = h2f(C[pc.lane][pc.reg] >> (16 * pc.half) & 0xffff);
					got = h2f(D[pc.lane][pc.reg] >> (16 * pc.half) & 0xffff);
				}
				sum += cv;
				mag += fabsl((long double)cv);
				bound = gamma_n(p->depth) * (double)mag;
				if (!s->f32)	/* the f16 rounding of the f32 sum */
					bound = 0x1p-11 * fabs((double)sum) + (1 + 0x1p-11) *
					    bound + 0x1p-25;
				e = fabsl((long double)got - sum);
				if ((double)e > bound)
					(*bad)++;
				if (bound > 0 && (double)e / bound > worst)
					worst = (double)e / bound;
			}
	}
	return worst;
}

/* ---- the search ---- */

/* the best plan of each family and fetch: [family][trans A][trans B] */
static struct mplan *best[F_N][T_N][T_N];
static struct mplan cur, ref;
static long tried;
static int feasible[F_N], shaped[F_N];	/* priced; a butterfly that fits */

static int
better(const struct mplan *a, const struct mplan *b)
{
	if (b == NULL)
		return 1;
	if (a->cost != b->cost)
		return a->cost < b->cost;
	return a->n < b->n;
}

static void
keep(const struct mplan *p)
{
	struct mplan **slot;

	tried++;
	mma_check_deadline();
	if (p->bad || !isfinite(p->cost))
		return;
	{
		/* an operand staged and never loaded: the plan without it */
		int i, o, ld[2] = {0, 0};
		for (i = 0; i < p->n; i++)
			if (p->in[i].op == I_LDS)
				ld[p->in[i].opnd] = 1;
		for (o = 0; o < 2; o++)
			if (p->trans[o] >= T_STAGE0 && !ld[o])
				return;
	}
	feasible[p->fam] = 1;
	slot = &best[p->fam][p->trans[0]][p->trans[1]];
	if (!better(p, *slot))
		return;
	if (*slot == NULL && (*slot = malloc(sizeof **slot)) == NULL)
		fail("mma: out of memory");
	**slot = *p;
}

/* the masks of a xor of lane bits into n values, at most k bits nonzero */
static int
next_perm(int *P, int n, int k)
{
	int i, nz;

	for (;;) {
		for (i = 0; i < 5; i++) {
			if (++P[i] < n)
				break;
			P[i] = 0;
		}
		if (i == 5)
			return 0;
		for (nz = 0, i = 0; i < 5; i++)
			nz += P[i] != 0;
		if (nz <= k)
			return 1;
	}
}

/* the next choice of each reduce bit's source: -1 (D's owner), 0..nk-1 */
static int
next_q(int *Q, int bm, int nk)
{
	int b;

	for (b = 0; b < 5; b++) {
		if (!(bm >> b & 1))
			continue;
		if (++Q[b] < nk)
			return 1;
		Q[b] = -1;
	}
	return 0;
}

static int
next_order(int *ord, int n)
{
	/* the next permutation, lexicographically */
	int i = n - 2, j, t;

	while (i >= 0 && ord[i] >= ord[i + 1])
		i--;
	if (i < 0)
		return 0;
	for (j = n - 1; ord[j] <= ord[i]; j--)
		;
	t = ord[i];
	ord[i] = ord[j];
	ord[j] = t;
	for (i++, j = n - 1; i < j; i++, j--) {
		t = ord[i];
		ord[i] = ord[j];
		ord[j] = t;
	}
	return 1;
}

/* ---- blocked: a lane's share of the iteration space ---- */

/*
 * The iteration space of a tile is its points (r, c, k word): ar bits of
 * the row, then ac of the column, then ak of the k word (each point two
 * products, the halves of an A word and a B word).  Lane bit i is the
 * iteration bit lb[i]; the other bits are the lane's slots, in order;
 * and a lane bit set xors sk[i] into the point's row and column bits (a
 * skew, so that a butterfly sends the same registers from every lane).
 * A lane sums its slots that differ in k word bits; the lanes that hold
 * one (r, c) differ in the lane bits that are k word bits, and those
 * partial sums meet by red_plan's butterfly.  Output-stationary and
 * A-stationary are two of these maps; the search sees all of them.
 */
struct bmap {
	int lb[5], sk[5];
};

struct ispace {
	int ar, ac, ak, nb, smask, rcmask, nslot;
};

static int
ilog2(int v)
{
	int n = 0;

	while (1 << n < v)
		n++;
	return n;
}

static void
isp_init(const struct shape *s, const struct bmap *B, struct ispace *I)
{
	int i, used = 0;

	I->ar = ilog2(s->M);
	I->ac = ilog2(s->N);
	I->ak = ilog2(s->K / 2);
	I->nb = I->ar + I->ac + I->ak;
	for (i = 0; i < 5; i++)
		used |= 1 << B->lb[i];
	I->smask = ((1 << I->nb) - 1) & ~used;
	I->rcmask = (1 << (I->ar + I->ac)) - 1;
	I->nslot = 1 << (I->nb - 5);
}

/* slot j's point in lane l */
static int
blk_point(const struct ispace *I, const struct bmap *B, int l, int j)
{
	int p = 0, i, q = 0;

	for (i = 0; i < I->nb; i++)
		if (I->smask >> i & 1)
			p |= (j >> q++ & 1) << i;
	for (i = 0; i < 5; i++)
		if (l >> i & 1)
			p ^= 1 << B->lb[i] ^ B->sk[i];
	return p;
}

/* the partial sum slot j adds into: its row and column bits, packed */
static int
blk_part(const struct ispace *I, int j)
{
	int i, q = 0, k = 0, ps = 0;

	for (i = 0; i < I->nb; i++)
		if (I->smask >> i & 1) {
			if (I->rcmask >> i & 1)
				ps |= (j >> q & 1) << k++;
			q++;
		}
	return ps;
}

static void
blk_rck(const struct ispace *I, int p, int *r, int *c, int *kw)
{
	*r = p & ((1 << I->ar) - 1);
	*c = p >> I->ar & ((1 << I->ac) - 1);
	*kw = p >> (I->ar + I->ac);
}

static int
blk_keys(const struct shape *s, const struct bmap *B, struct red *R)
{
	struct ispace I;
	int j, l;

	isp_init(s, B, &I);
	R->ns = 1 << popc((unsigned)(I.smask & I.rcmask));
	if (R->ns > MAXS)
		return 0;
	for (j = 0; j < I.nslot; j++)
		for (l = 0; l < ML; l++) {
			int r, c, kw;
			blk_rck(&I, blk_point(&I, B, l, j), &r, &c, &kw);
			R->key[blk_part(&I, j)][l] = (int16_t)(r * s->N + c);
		}
	return 1;
}

static void
gen_block(struct bld *b, const struct bmap *B, const struct red *R)
{
	const struct shape *s = b->s;
	struct mplan *p = b->p;
	struct ispace I;
	int j, l, val[MAXS];
	int ua = p->trans[OA] == T_SHUFU, ub = p->trans[OB] == T_SHUFU;

	isp_init(s, B, &I);
	for (j = 0; j < R->ns; j++)
		val[j] = -1;
	b->staged[OA] = p->trans[OA] >= T_STAGE0;
	b->staged[OB] = p->trans[OB] >= T_STAGE0;
	stage_setup(b);
	for (j = 0; j < I.nslot && !p->bad; j++) {
		int8_t sla[ML], rga[ML], slb[ML], rgb[ML], ps = (int8_t)blk_part(&I, j);
		for (l = 0; l < ML; l++) {
			int r, c, kw;
			struct pos pa, pb;
			blk_rck(&I, blk_point(&I, B, l, j), &r, &c, &kw);
			pa = s->aown[r][2 * kw];
			pb = s->bown[2 * kw][c];
			sla[l] = pa.lane;
			rga[l] = pa.reg;
			slb[l] = pb.lane;
			rgb[l] = pb.reg;
		}
		val[ps] = dot_fetch(b, val[ps], -1, sla, rga, ua, -1, slb, rgb, ub);
	}
	if (!p->bad)
		red_finish(b, R, val);
}

static uint64_t
tab_hash(const int8_t *a, const int8_t *b2)
{
	uint64_t h = 1469598103934665603ULL;
	int l;

	for (l = 0; l < ML; l++)
		h = (h ^ (uint64_t)(uint8_t)a[l] << 8 ^ (uint8_t)b2[l]) *
		    1099511628211ULL;
	return h;
}

/* a quick price of a map: its distinct words fetched as shuffles, its
 * butterfly, without the skew */
static double
blk_est(const struct shape *s, const struct model *m, const struct bmap *B)
{
	static uint64_t seen[2][1024];
	struct ispace I;
	int j, l, o, n[2] = {0, 0}, nk = 0, i, ns;
	double fc = 0, conv2 = avail(m, C_FDOT2) ? 0 : 2 * price(m, C_CONV);

	isp_init(s, B, &I);
	for (j = 0; j < I.nslot; j++)
		for (o = 0; o < 2; o++) {
			int8_t sl[ML], rg[ML];
			int self = 1, k;
			uint64_t h;
			for (l = 0; l < ML; l++) {
				int r, c, kw;
				struct pos q;
				blk_rck(&I, blk_point(&I, B, l, j), &r, &c, &kw);
				q = o == 0 ? s->aown[r][2 * kw] : s->bown[2 * kw][c];
				sl[l] = q.lane;
				rg[l] = q.reg;
				self &= q.lane == l && q.reg == rg[0];
			}
			if (self)
				continue;
			h = tab_hash(sl, rg) | 1;
			if (j == 0 && o == 0)
				memset(seen, 0, sizeof seen);
			for (k = (int)(h >> 20 & 1023); seen[o][k] && seen[o][k] != h;
			    k = (k + 1) & 1023)
				;
			if (!seen[o][k]) {
				seen[o][k] = h;
				n[o]++;
				int16_t t[ML];
				int xk, xm, uni = 1;
				double x;
				for (l = 0; l < ML; l++) {
					t[l] = sl[l];
					uni &= rg[l] == rg[0];
				}
				xk = classify(t, &xm);
				x = xk >= 0 ? xch_price(m, xk, xm) : price(m, C_SG_SHUF);
				if (!uni)	/* by a select of two, at least */
					x = 2 * x + price(m, C_SELECT);
				fc += x + conv2 < 2 * x ? x + conv2 : 2 * x;
			}
		}
	for (i = 0; i < 5; i++)
		nk += B->lb[i] >= I.ar + I.ac;
	ns = 1 << popc((unsigned)(I.smask & I.rcmask));
	return fc + ns * (1.0 - 1.0 / (1 << nk)) * (xch_price(m, XK_XOR, 1) +
	    price(m, C_FADD));
}

#define NTOP 48
#define NFIN 4	/* maps tried with every fetch */

static void
search_block(const struct shape *s, const struct model *m, struct bld *b)
{
	static struct bmap top[NTOP];
	static struct red R;
	double tope[NTOP];
	struct bmap B;
	struct ispace I;
	int ntop = 0, nb, i, t, a0, a1, a2, a3, a4, within;
	struct ivec { int v[5], n; };
	static struct bmap fin[NFIN];
	static struct ivec fin_o[NFIN], fin_q[NFIN];
	double fin_c[NFIN];


	for (i = 0; i < NFIN; i++)
		fin_c[i] = INFINITY;

	memset(&B, 0, sizeof B);
	isp_init(s, &B, &I);
	nb = I.nb;
	/* the lanes share out one HMMA's bits (16 rows, 8 columns, its k
	 * words); a tile's own bits are slots */
	within = 15 | 7 << I.ar | ((s->k16 ? 7 : 3) << (I.ar + I.ac));
	for (a0 = 0; a0 < nb; a0++)
	for (a1 = 0; a1 < nb; a1++)
	for (a2 = 0; a2 < nb; a2++)
	for (a3 = 0; a3 < nb; a3++)
	for (a4 = 0; a4 < nb; a4++) {
		double e;
		int u = 1 << a0 | 1 << a1 | 1 << a2 | 1 << a3 | 1 << a4;
		if (popc((unsigned)u) != 5 || (u & ~within))
			continue;
		B.lb[0] = a0;
		B.lb[1] = a1;
		B.lb[2] = a2;
		B.lb[3] = a3;
		B.lb[4] = a4;
		if (1 << popc((unsigned)(I.rcmask & ~u)) > MAXS)
			continue;
		e = blk_est(s, m, &B);
		if (ntop == NTOP && e >= tope[NTOP - 1])
			continue;
		/* one of each price among ties of the same split of the lane
		 * bits over rows, columns and k */
		{
			int dup = 0, sig = 0, q;
			for (i = 0; i < 5; i++)
				sig += B.lb[i] < I.ar ? 1 : B.lb[i] < I.ar + I.ac ? 8 : 64;
			for (q = 0; q < ntop && !dup; q++) {
				int sig2 = 0;
				for (i = 0; i < 5; i++)
					sig2 += top[q].lb[i] < I.ar ? 1 : top[q].lb[i] < I.ar +
					    I.ac ? 8 : 64;
				dup = sig2 == sig && tope[q] == e;
			}
			if (dup)
				continue;
		}
		for (i = ntop < NTOP ? ntop++ : NTOP - 1; i > 0 && tope[i - 1] > e;
		    i--) {
			tope[i] = tope[i - 1];
			top[i] = top[i - 1];
		}
		tope[i] = e;
		top[i] = B;
	}
	for (t = 0; t < ntop; t++) {
		int kl[5], nkl = 0, rcb[16], nrc = 0, ch[5], bm, ord[5], nord, Q[5];
		int nkb = 0, found, tries, nz;
		mma_check_deadline();
		B = top[t];
		isp_init(s, &B, &I);
		while (1 << nkb < s->M * s->N)
			nkb++;
		for (i = 0; i < 5; i++)
			if (B.lb[i] >= I.ar + I.ac)
				kl[nkl++] = i;
		for (i = 0; i < I.ar + I.ac; i++)
			if (I.smask >> i & 1)
				rcb[nrc++] = i;
		/* the skew: each k lane bit xors one row or column slot bit in */
		memset(ch, 0, sizeof ch);
		for (;;) {
			mma_check_deadline();
			memset(B.sk, 0, sizeof B.sk);
			nz = 0;
			for (i = 0; i < nkl; i++) {
				B.sk[kl[i]] = ch[i] ? 1 << rcb[ch[i] - 1] : 0;
				nz += ch[i] != 0;
			}
			found = 0;
			if (nz <= 2 && blk_keys(s, &B, &R) && red_plan(s, &R, ord, -1, NULL, &bm)
			    != 0) {
				nord = 0;
				for (i = 0; i < 5; i++)
					if (bm >> i & 1)
						ord[nord++] = i;
				memset(Q, -1, sizeof Q);
				tries = 0;
				do {
					do {
						tries++;
						if (red_plan(s, &R, ord, nord, Q, &bm) == 1) {
							found = 1;
							break;
						}
					} while (next_order(ord, nord));
				} while (!found && tries < 24 && next_q(Q, bm, nkb));
			}
			if (found) {
				/* priced with shuffles of packed words alone, here */
				shaped[F_BLOCK] = 1;
				bld_init(b, &cur, s, m);
				cur.fam = F_BLOCK;
				gen_block(b, &B, &R);
				if (!cur.bad)
					plan_price(&cur, s, m);
				if (!cur.bad && cur.cost < fin_c[NFIN - 1]) {
					int q2;
					for (q2 = NFIN - 1; q2 > 0 && fin_c[q2 - 1] > cur.cost;
					    q2--) {
						fin_c[q2] = fin_c[q2 - 1];
						fin[q2] = fin[q2 - 1];
						fin_o[q2] = fin_o[q2 - 1];
						fin_q[q2] = fin_q[q2 - 1];
					}
					fin_c[q2] = cur.cost;
					fin[q2] = B;
					memcpy(fin_o[q2].v, ord, sizeof ord);
					fin_o[q2].n = nord;
					memcpy(fin_q[q2].v, Q, sizeof Q);
				}
			}
			for (i = 0; i < nkl; i++) {
				if (++ch[i] <= nrc)
					break;
				ch[i] = 0;
			}
			if (i == nkl)
				break;
		}
	}
	/* the best maps, with every way to fetch */
	for (t = 0; t < NFIN && isfinite(fin_c[t]); t++) {
		int ta, tb, bm;
		B = fin[t];
		blk_keys(s, &B, &R);
		if (red_plan(s, &R, fin_o[t].v, fin_o[t].n, fin_q[t].v, &bm) != 1)
			continue;
		for (ta = 0; ta < T_N; ta++)
			for (tb = 0; tb < T_N; tb++) {
				bld_init(b, &cur, s, m);
				cur.fam = F_BLOCK;
				cur.trans[OA] = ta;
				cur.trans[OB] = tb;
				memcpy(cur.lb, B.lb, sizeof B.lb);
				memcpy(cur.sk, B.sk, sizeof B.sk);
				memcpy(cur.order, fin_o[t].v, sizeof cur.order);
				cur.nord = fin_o[t].n;
				gen_block(b, &B, &R);
				if (!cur.bad)
					plan_price(&cur, s, m);
				keep(&cur);
			}
	}
}

static void
search(const struct shape *s, const struct model *m)
{
	static struct bld b;
	int P[5], ta, tb, X, i;
	struct red R;

	tried = 0;
	memset(feasible, 0, sizeof feasible);
	memset(shaped, 0, sizeof shaped);
	for (X = 0; X < F_N; X++)
		for (ta = 0; ta < T_N; ta++)
			for (tb = 0; tb < T_N; tb++)
				if (best[X][ta][tb]) {
					free(best[X][ta][tb]);
					best[X][ta][tb] = NULL;
				}
	/* the reference: output-stationary, each word from its owner, packed */
	memset(P, 0, sizeof P);
	bld_init(&b, &ref, s, m);
	b.noreuse = 1;
	ref.fam = F_OUT;
	gen_out(&b);
	if (!ref.bad)
		plan_price(&ref, s, m);
	/* output-stationary, the k words in the order kw ^ perm(l) */
	do
		for (ta = 0; ta < T_N; ta++)
			for (tb = 0; tb < T_N; tb++) {
				bld_init(&b, &cur, s, m);
				cur.fam = F_OUT;
				cur.trans[OA] = ta;
				cur.trans[OB] = tb;
				memcpy(cur.perm, P, sizeof P);
				gen_out(&b);
				if (!cur.bad)
					plan_price(&cur, s, m);
				keep(&cur);
			}
	while (next_perm(P, s->K / 2 < 8 ? s->K / 2 : 8, 2));
	/* X-stationary, slot j's other index j ^ perm(l) */
	for (X = OA; X <= OB; X++) {
		int ny = X == OA ? s->N : s->M, grp[NG * GS], ng, gof[NG], Q[5];
		int nkb = 0;	/* the bits of r * N + c */
		while (1 << nkb < s->M * s->N)
			nkb++;
		memset(P, 0, sizeof P);
		do {
			int bm, ord[5], nord = 0, r;
			if (!stat_keys(s, X, P, &R, grp, &ng, gof))
				continue;
			r = red_plan(s, &R, ord, -1, NULL, &bm);
			if (r == 0)
				continue;
			for (i = 0; i < 5; i++)
				if (bm >> i & 1)
					ord[nord++] = i;
			/* the reduce target's bits: D's owner's, or a key bit each */
			memset(Q, -1, sizeof Q);
			do {
			do {
				if (red_plan(s, &R, ord, nord, Q, &bm) != 1)
					continue;
				shaped[X == OA ? F_ASTAT : F_BSTAT] = 1;
				for (ta = 0; ta < T_N; ta++) {
					bld_init(&b, &cur, s, m);
					cur.fam = X == OA ? F_ASTAT : F_BSTAT;
					cur.trans[1 - X] = ta;
					cur.trans[X] = 0;
					memcpy(cur.perm, P, sizeof P);
					memcpy(cur.order, ord, sizeof ord);
					cur.nord = nord;
					gen_stat(&b, &R, X);
					if (!cur.bad)
						plan_price(&cur, s, m);
					keep(&cur);
				}
			} while (next_order(ord, nord));
			} while (next_q(Q, bm, nkb));
		} while (next_perm(P, ny < 16 ? ny : 16, ny > 8 ? 2 : 3));
	}
	/* blocked: any share of the iteration space */
	search_block(s, m, &b);
}

/* ---- words ---- */

static const struct shape *txt_shape;	/* plan_text's */

static const char *const famname[F_N] = {"output-stationary",
	"A-stationary", "B-stationary", "blocked"};

static void
plan_text(struct sbuf *b, const struct mplan *p, const struct model *m)
{
	static const char *const what[3] = {"A", "B", "partial sums"};
	int w, k, i, nl[2] = {0, 0}, any;

	for (i = 0; i < p->n; i++)
		if (p->in[i].op == I_LDS)
			nl[p->in[i].opnd]++;
	sb_printf(b, "%s", famname[p->fam]);
	if (p->fam == F_BLOCK && txt_shape != NULL) {
		struct ispace I;
		struct bmap B;
		int i, nr = 0, nc = 0, nk = 0;
		memcpy(B.lb, p->lb, sizeof B.lb);
		isp_init(txt_shape, &B, &I);
		for (i = 0; i < 5; i++)
			if (p->lb[i] < I.ar)
				nr++;
			else if (p->lb[i] < I.ar + I.ac)
				nc++;
			else
				nk++;
		sb_printf(b, " (a lane: %d rows x %d columns x %d of k)",
		    txt_shape->M >> nr, txt_shape->N >> nc, txt_shape->K >> nk);
	}
	for (w = 0; w < 3; w++) {
		int n[4] = {0, 0, 0, 0};
		double pr;
		for (i = 0; i < p->n; i++)
			if (p->in[i].op == I_XCH && p->in[i].what == w)
				n[xch_form(m, p->in[i].xk, p->in[i].xm, &pr)]++;
		if (w < 2 && !nl[w] && !n[0] && !n[1] && !n[2] && !n[3])
			continue;
		if (w == 2 && !n[0] && !n[1] && !n[2] && !n[3])
			continue;
		sb_printf(b, "; %s:", what[w]);
		any = 0;
		if (w < 2 && nl[w]) {
			int acc = 0;
			for (i = 0; i < p->n; i++)
				if (p->in[i].op == I_LDS && p->in[i].opnd == w &&
				    !p->in[i].vl)
					acc++;
			sb_printf(b, " %d loads from shared memory (%s", nl[w],
			    p->trans[w] == T_STAGE0 ? "register-major" : "lane-major");
			if (acc < nl[w])
				sb_printf(b, ", in %d wide accesses", acc);
			sb_printf(b, ")");
			any = 1;
		}
		{
			static const char *const fn[4] = {"xor shuffles", "quad swaps",
				"quad broadcasts", "shuffles"};
			for (k = 0; k < 4; k++)
				if (n[k]) {
					sb_printf(b, "%s %d %s", any ? "," : "", n[k], fn[k]);
					any = 1;
				}
		}
		if (w < 2 && p->trans[w] == T_SHUFU)
			sb_printf(b, " (unpacked)");
	}
	if (p->nsel)
		sb_printf(b, "; %d selects", p->nsel);
	if (p->ndot2)
		sb_printf(b, "; %d fdot2", p->ndot2);
	if (p->fam == F_OUT && xperm(p->perm, 31))
		sb_printf(b, "; k order by lane");
}

/* ---- GLSL ---- */

static const char *
vtype(int vt)
{
	return vt == VW ? "uint" : "float";
}

static void
glsl(struct sbuf *o, const struct mplan *p, const struct shape *s,
    const struct model *m, const char *fname, const char *head)
{
	static uint8_t idx[MAXT];
	int i, quad = 0, shuf = 0, dot2 = 0, stage = 0, sw = stage_words(s);
	int sgbar, vec = 0;
	double pr;

	for (i = 0; i < p->n; i++) {
		const struct ins *x = &p->in[i];
		if (x->op == I_XCH) {
			int f = xch_form(m, x->xk, x->xm, &pr);
			quad |= f == FM_QSWAP || f == FM_QBCAST;
			shuf = 1;
		}
		dot2 |= x->op == I_DOT2;
		stage |= x->op == I_STS;
		vec |= x->vw != 0;
	}
	/* a subgroup barrier where the warp is within one subgroup (it
	 * exchanges, or the model has one), else the workgroup's */
	sgbar = shuf || avail(m, C_SG_BARRIER);
	if (head)
		sb_puts(o, head);
	if (shuf || sgbar)
		sb_printf(o, "#extension GL_KHR_shader_subgroup_basic : require\n");
	if (shuf)
		sb_printf(o, "#extension GL_KHR_shader_subgroup_shuffle : require\n");
	if (quad)
		sb_printf(o, "#extension GL_KHR_shader_subgroup_quad : require\n");
	if (dot2)
		sb_printf(o, "#extension GL_EXT_spirv_intrinsics : require\n"
		    "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : "
		    "require\n"
		    "// OpFDot2MixAcc32VALVE: a.x * b.x + a.y * b.y + c in f32\n"
		    "spirv_instruction(extensions = "
		    "[\"SPV_VALVE_mixed_float_dot_product\"], capabilities = [6912], "
		    "id = 6916)\n"
		    "float mma_fdot2(f16vec2 a, f16vec2 b, float c);\n");
	if (stage)
		sb_printf(o, "#ifndef MMA_WARPS\n#define MMA_WARPS 1  "
		    "// the workgroup's warps of 32\n#endif\n"
		    "%s[MMA_WARPS * %du];\n", vec ? "shared uvec4 mma_slot4" :
		    "shared uint mma_slot", vec ? sw / 4 : sw);
	if (s->f32)
		sb_printf(o, "void %s(uint a[%d], uint b[%d], float c[%d], "
		    "out float d[%d])\n{\n", fname, s->na, s->nb, s->nc, s->nc);
	else
		sb_printf(o, "void %s(uint a[%d], uint b[%d], uint c[%d], "
		    "out uint d[%d])\n{\n", fname, s->na, s->nb, s->nc, s->nc);
	sb_printf(o, "\tuint lane = %s & 31u;\n", sgbar ?
	    "gl_SubgroupInvocationID" : "gl_LocalInvocationIndex");
	if (shuf)
		sb_printf(o, "\tuint base = gl_SubgroupInvocationID & ~31u;\n");
	if (stage)
		sb_printf(o, "\tuint slot = (gl_LocalInvocationIndex >> 5) * %du;\n",
		    sw);
	memset(idx, 0, sizeof idx);
	for (i = 0; i < p->n; i++) {
		const struct ins *x = &p->in[i];
		int f;
		switch (x->op) {
		case I_IN:
			sb_printf(o, "\t%s v%d = %c[%d];\n", vtype(x->vt), i,
			    "abc"[x->opnd], x->reg);
			break;
		case I_XCH:
			f = xch_form(m, x->xk, x->xm, &pr);
			if (f == FM_SHUF && !idx[x->tab]) {
				int par = idx_parent(p, x->tab, idx);
				if (par >= 0)
					sb_printf(o, "\tuint i%d = i%d ^ %du;\n", x->tab, par,
					    p->tab[par][0] ^ p->tab[x->tab][0]);
				else {
					sb_printf(o, "\tuint i%d = base | (", x->tab);
					aff_text(o, p->tab[x->tab]);
					sb_printf(o, ");\n");
				}
				idx[x->tab] = 1;
			}
			sb_printf(o, "\t%s v%d = ", vtype(x->vt), i);
			if (f == FM_XOR)
				sb_printf(o, "subgroupShuffleXor(v%d, %du);\n", x->a, x->xm);
			else if (f == FM_QSWAP)
				sb_printf(o, "subgroupQuadSwap%s(v%d);\n", x->xm == 1 ?
				    "Horizontal" : x->xm == 2 ? "Vertical" : "Diagonal",
				    x->a);
			else if (f == FM_QBCAST)
				sb_printf(o, "subgroupQuadBroadcast(v%d, %du);\n", x->a,
				    x->xm);
			else
				sb_printf(o, "subgroupShuffle(v%d, i%d);\n", x->a, x->tab);
			break;
		case I_STS:
		case I_LDS:
			if (!idx[x->tab]) {
				int par = addr_parent(p, x->tab, idx);
				if (par < 0) {
					sb_printf(o, "\tuint a%d = slot + (", x->tab);
					aff_text(o, p->tab[x->tab]);
					sb_printf(o, ");\n");
				}
				idx[x->tab] = 2;
			}
			{
				int par = addr_parent(p, x->tab, idx), off = 0;
				char at[48];
				if (par >= 0 && par != x->tab)
					off = p->tab[x->tab][0] - p->tab[par][0];
				else
					par = x->tab;
				if (off)
					snprintf(at, sizeof at, "a%d + %du", par, off);
				else
					snprintf(at, sizeof at, "a%d", par);
				if (!vec) {
					if (x->op == I_STS)
						sb_printf(o, "\tmma_slot[%s] = v%d;\n", at, x->a);
					else
						sb_printf(o, "\tuint v%d = mma_slot[%s];\n", i, at);
				} else if (x->vl) {
					if (x->op == I_LDS)
						sb_printf(o, "\tuint v%d = q%d[%d];\n", i, x->vl - 1,
						    x->vc);
				} else if (x->vw && x->op == I_LDS) {
					sb_printf(o, "\tuvec%d q%d = mma_slot4[(%s) >> 2u]%s;\n",
					    x->vw, i, at, x->vw == 4 ? "" :
					    p->tab[x->tab][0] & 2 ? ".zw" : ".xy");
					sb_printf(o, "\tuint v%d = q%d[%d];\n", i, i, x->vc);
				} else if (x->vw) {
					int comp[4], j2;
					comp[x->vc] = x->a;
					for (j2 = 0; j2 < p->n; j2++)
						if (p->in[j2].vl == i + 1)
							comp[p->in[j2].vc] = p->in[j2].a;
					if (x->vw == 4)
						sb_printf(o, "\tmma_slot4[(%s) >> 2u] = uvec4(v%d, v%d, "
						    "v%d, v%d);\n", at, comp[0], comp[1], comp[2],
						    comp[3]);
					else
						sb_printf(o, "\tmma_slot4[(%s) >> 2u].%s = uvec2(v%d, "
						    "v%d);\n", at, p->tab[x->tab][0] & 2 ? "zw" :
						    "xy", comp[0], comp[1]);
				} else if (x->op == I_STS)
					sb_printf(o, "\tmma_slot4[(%s) >> 2u][(%s) & 3u] = v%d;\n",
					    at, at, x->a);
				else
					sb_printf(o, "\tuint v%d = mma_slot4[(%s) >> 2u][(%s) & "
					    "3u];\n", i, at, at);
			}
			break;
		case I_BAR:
			sb_printf(o, "\t%s();\n", sgbar ? "subgroupBarrier" : "barrier");
			break;
		case I_UNPK:
			sb_printf(o, "\tfloat v%d = unpackHalf2x16(v%d).%c;\n", i, x->a,
			    "xy"[x->half]);
			break;
		case I_SEL:
			sb_printf(o, "\t%s v%d = (lane & %du) != 0u ? v%d : v%d;\n",
			    vtype(x->vt), i, 1 << x->xm, x->b, x->a);
			break;
		case I_MUL:
			sb_printf(o, "\tfloat v%d = v%d * v%d;\n", i, x->a, x->b);
			break;
		case I_FMA:
			sb_printf(o, "\tfloat v%d = fma(v%d, v%d, v%d);\n", i, x->a, x->b,
			    x->c);
			break;
		case I_DOT2:
			sb_printf(o, "\tfloat v%d = mma_fdot2(unpackFloat2x16(v%d), "
			    "unpackFloat2x16(v%d), ", i, x->a, x->b);
			if (x->c >= 0)
				sb_printf(o, "v%d);\n", x->c);
			else
				sb_printf(o, "0.0);\n");
			break;
		case I_FADD:
			sb_printf(o, "\tfloat v%d = v%d + v%d;\n", i, x->a, x->b);
			break;
		case I_OUT:
			if (s->f32)
				sb_printf(o, "\td[%d] = v%d;\n", x->reg, x->a);
			else
				sb_printf(o, "\td[%d] = packHalf2x16(vec2(v%d, v%d));\n",
				    x->reg, x->a, x->b);
			break;
		}
	}
	sb_printf(o, "}\n");
}

/* glslangValidator and spirv-val on a compute shader that calls it: 1
 * accepted, 0 refused (why), -1 no tools */
static int
validate(const char *fn, const char *src, const struct shape *s,
    struct sbuf *why)
{
	char dir[] = "/tmp/ucas-mma.XXXXXX", path[256], cmd[1024];
	struct sbuf w = {0};
	FILE *fp;
	int rc;

	if (system("command -v glslangValidator >/dev/null 2>&1 && "
	    "command -v spirv-val >/dev/null 2>&1") != 0)
		return -1;
	if (mkdtemp(dir) == NULL)
		return -1;
	sb_printf(&w, "#version 450\n%s"
	    "layout(local_size_x = 32) in;\n"
	    "layout(std430, binding = 0) buffer IO { uint ab[]; } io;\n"
	    "void main()\n{\n\tuint i = gl_LocalInvocationIndex;\n"
	    "\tuint a[%d], b[%d];\n", src, s->na, s->nb);
	sb_printf(&w, "\tfor (int k = 0; k < %d; k++) a[k] = io.ab[i * 64u + "
	    "uint(k)];\n", s->na);
	sb_printf(&w, "\tfor (int k = 0; k < %d; k++) b[k] = io.ab[i * 64u + 16u "
	    "+ uint(k)];\n", s->nb);
	if (s->f32)
		sb_printf(&w, "\tfloat c[%d], d[%d];\n\tfor (int k = 0; k < %d; "
		    "k++) c[k] = uintBitsToFloat(io.ab[i * 64u + 24u + uint(k)]);\n"
		    "\t%s(a, b, c, d);\n\tfor (int k = 0; k < %d; k++) "
		    "io.ab[i * 64u + 40u + uint(k)] = floatBitsToUint(d[k]);\n}\n",
		    s->nc, s->nc, s->nc, fn, s->nc);
	else
		sb_printf(&w, "\tuint c[%d], d[%d];\n\tfor (int k = 0; k < %d; "
		    "k++) c[k] = io.ab[i * 64u + 24u + uint(k)];\n"
		    "\t%s(a, b, c, d);\n\tfor (int k = 0; k < %d; k++) "
		    "io.ab[i * 64u + 40u + uint(k)] = d[k];\n}\n", s->nc, s->nc,
		    s->nc, fn, s->nc);
	/* the extensions go first */
	{
		struct sbuf v = {0};
		const char *t = sb_str(&w), *e;
		sb_printf(&v, "#version 450\n");
		t = strchr(t, '\n') + 1;
		while (!strncmp(t, "#extension", 10)) {
			e = strchr(t, '\n') + 1;
			sb_printf(&v, "%.*s", (int)(e - t), t);
			t = e;
		}
		sb_printf(&v, "%s", t);
		snprintf(path, sizeof path, "%s/mma.comp", dir);
		if ((fp = fopen(path, "w")) == NULL)
			return -1;
		fputs(sb_str(&v), fp);
		fclose(fp);
		free(v.p);
	}
	free(w.p);
	snprintf(cmd, sizeof cmd, "glslangValidator -V --target-env vulkan1.1 "
	    "-o %s/mma.spv %s > %s/log 2>&1 && spirv-val --target-env "
	    "vulkan1.1 %s/mma.spv >> %s/log 2>&1", dir, path, dir, dir, dir);
	rc = system(cmd);
	if (rc != 0) {
		char line[300];
		snprintf(path, sizeof path, "%s/log", dir);
		if ((fp = fopen(path, "r")) != NULL) {
			while (fgets(line, sizeof line, fp))
				if (strstr(line, "ERROR") || strstr(line, "error")) {
					line[strcspn(line, "\n")] = 0;
					sb_printf(why, "%s", line);
					break;
				}
			fclose(fp);
		}
	}
	snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
	if (system(cmd) != 0)
		;
	return rc == 0;
}

/* ---- the command ---- */

static int
cmp_plan(const void *a, const void *b)
{
	const struct mplan *p = *(const struct mplan *const *)a;
	const struct mplan *q = *(const struct mplan *const *)b;

	return better(p, q) ? -1 : better(q, p) ? 1 : 0;
}

static void
mma_run(int k16, int f32, int tm, int tn, int tk, const struct model *m,
    int all, int val, int nofile, uint64_t seed)
{
	static struct shape S;
	struct mplan *list[F_N * T_N * T_N], *p;
	struct sbuf why = {0}, t = {0}, g = {0};
	int n = 0, i, ta, tb, f, shown, bad, nterm;
	char fname[48], file[64], base[16];
	double worst, per;

	volatile double single = 0;
	volatile int single_done = 0;

	snprintf(base, sizeof base, "m16n8k%d", k16 ? 16 : 8);
	/* The search is cut by a wall-clock deadline rather than left to run
	 * for tens of seconds.  seconds 0 asks for the full, unbounded hunt. */
	mma_stopped = 0;
	mma_deadline_set = mma_budget_s > 0;
	if (mma_deadline_set) {
		clock_gettime(CLOCK_MONOTONIC, &mma_deadline_end);
		mma_deadline_end.tv_sec += (time_t)mma_budget_s;
		mma_deadline_end.tv_nsec +=
		    (long)((mma_budget_s - (double)(time_t)mma_budget_s) * 1e9);
		if (mma_deadline_end.tv_nsec >= 1000000000L) {
			mma_deadline_end.tv_sec++;
			mma_deadline_end.tv_nsec -= 1000000000L;
		}
	}
	if (setjmp(mma_deadline_jmp) == 0) {
		if (tm * tn * tk > 1) {
			/* the HMMAs one at a time, each the best alone */
			shape_build(&S, k16, f32, 1, 1, 1);
			search(&S, m);
			single_done = 1;
			single = INFINITY;
			for (f = 0; f < F_N; f++)
				for (ta = 0; ta < T_N; ta++)
					for (tb = 0; tb < T_N; tb++)
						if (best[f][ta][tb] &&
						    best[f][ta][tb]->cost < single)
							single = best[f][ta][tb]->cost;
		}
		shape_build(&S, k16, f32, tm, tn, tk);
		txt_shape = &S;
		if (!avail(m, C_SG_SHUF) && !(isfinite(price(m, C_LOAD_SHARED)) &&
		    isfinite(price(m, C_STORE_SHARED))))
			fail("mma: %s has neither subgroup shuffles (sg_shuffle) nor "
			    "shared memory to move the fragments with", sym_str(m->name));
		search(&S, m);
	}
	mma_deadline_set = 0;
	/* Cut during the first (single-HMMA) search of a tile: report the
	 * single HMMA we actually searched, not a tile we never got to. */
	if (mma_stopped && tm * tn * tk > 1 && !single_done)
		tm = tn = tk = 1;
	for (f = 0; f < F_N; f++)
		for (ta = 0; ta < T_N; ta++)
			for (tb = 0; tb < T_N; tb++)
				if (best[f][ta][tb])
					list[n++] = best[f][ta][tb];
	if (n == 0)
		fail("mma: no dataflow is possible under %s (its subgroup "
		    "operations or its shared memory are missing)",
		    sym_str(m->name));
	xsort(list, (size_t)n, sizeof list[0], cmp_plan);
	/* the proof of the cheapest that holds (each shown is proved) */
	for (i = 0; i < n; i++) {
		sb_clear(&why);
		if (!prove(list[i], &S, &why))
			fail("mma: internal error, a plan failed its proof: %s",
			    sb_str(&why));
	}
	p = list[0];
	nterm = S.K;
	snprintf(fname, sizeof fname, "mma_%s_%s", S.name, f32 ? "f32" : "f16");
	rep_begin("mma");
	per = tm * tn * tk;
	printf("mma %s: D = A B + C on a warp's fragments as PTX's "
	    "mma.sync.aligned.%s.row.col lays them out, f16 A and B, %s C and "
	    "D, 32 lanes, under %s (", S.name, base, f32 ? "f32" : "f16",
	    sym_str(m->name));
	if (mma_stopped)
		printf("search budget exhausted after %g s", mma_budget_s);
	else
		printf("%ld dataflows tried", tried);
	printf(")\n");
	if (mma_stopped)
		printf("  the search was cut by its %g s budget: showing the best "
		    "dataflow found so far, not a proved absence of a cheaper one"
		    "\n", mma_budget_s);
	if (per > 1)
		printf("  a tile of %d x %d x %d HMMAs (%s): %s; the costs are the "
		    "tile's, each HMMA's a %gth of them\n", tm, tn, tk, base,
		    tk > 1 && tm * tn == 1 ? "a chain, each one's D the next one's C"
		    : tk > 1 ? "chains along k, sharing fragments along m and n" :
		    tm > 1 && tn > 1 ? "sharing their A and B fragments" : tm > 1 ?
		    "one B fragment times several A ones" :
		    "one A fragment times several B ones", per);
	rep_str("shape", S.name);
	rep_str("accumulator", f32 ? "f32" : "f16");
	rep_str("model", sym_str(m->name));
	if (mma_stopped)
		rep_str("search", "budget exhausted");
	else
		rep_int("tried", tried);
	printf("  each of D's %d elements is a sum of %d products of f16 and its "
	    "C\n", S.M * S.N, nterm);
	rep_list("dataflows");
	shown = all ? n : n < 6 ? n : 6;
	for (i = 0; i < shown; i++) {
		sb_clear(&t);
		plan_text(&t, list[i], m);
		if (per > 1)
			printf("  %2d. %-9g (%g an HMMA) %s, depth %d\n", i + 1,
			    list[i]->cost, list[i]->cost / per, sb_str(&t),
			    list[i]->depth);
		else
			printf("  %2d. %-9g %s, depth %d\n", i + 1, list[i]->cost,
			    sb_str(&t), list[i]->depth);
		rep_obj(NULL);
		rep_num("cost", list[i]->cost);
		rep_str("dataflow", sb_str(&t));
		rep_int("depth", list[i]->depth);
		rep_end();
	}
	rep_end();
	if (shown < n)
		printf("  (%d more: mma ... all)\n", n - shown);
	for (f = 0; f < F_N; f++)
		if (!feasible[f])
			printf("  %s: none possible: %s\n", famname[f], f == F_OUT ||
			    shaped[f] ? "its exchanges need subgroup shuffles, which "
			    "the model does not list (sg_shuffle)" : "no assignment of "
			    "the partial sums to registers (a xor of the lane's bits) "
			    "lets a butterfly send and read them from the same "
			    "registers in every lane");
	if (per > 1 && isfinite(single)) {
		printf("  the %g HMMAs one at a time, each the best alone: %g (%g "
		    "each); the tile is %.2fx cheaper\n", per, single * per, single,
		    single * per / list[0]->cost);
		rep_num("separate_cost", single * per);
	}
	if (!ref.bad) {
		sb_clear(&why);
		if (!prove(&ref, &S, &why))
			fail("mma: internal error, the reference failed its proof: %s",
			    sb_str(&why));
		sb_clear(&t);
		plan_text(&t, &ref, m);
		printf("  reference, each word fetched from the lane that holds it: "
		    "%g (%s, depth %d)", ref.cost, sb_str(&t), ref.depth);
		if (ref.cost > list[0]->cost * 1.005)
			printf("; the chosen is %.2fx cheaper\n", ref.cost /
			    list[0]->cost);
		else
			printf("; nothing cheaper under %s\n", sym_str(m->name));
		rep_num("reference_cost", ref.cost);
	}
	sb_clear(&t);
	plan_text(&t, p, m);
	printf("  chosen: %s, cost %g\n", sb_str(&t), p->cost);
	rep_str("chosen", sb_str(&t));
	rep_num("cost", p->cost);
	{
		int nx = 0, w, k;
		for (w = 0; w < 3; w++)
			for (k = 0; k < 3; k++)
				nx += p->nx[k][w];
		printf("    %d exchanges, %d loads and %d stores of shared memory "
		    "(bank conflicts at most %d-way), %d barriers, %d unpacks, "
		    "%d %s, %d adds%s\n", nx, p->nlds, p->nsts, p->maxdeg ?
		    p->maxdeg : 1, p->nbar, p->nunpk, p->ndot2 ? p->ndot2 : p->nfma,
		    p->ndot2 ? "fdot2" : "multiply-adds", p->nfadd, f32 ? "" :
		    ", 2 packs");
		if (p->nidx)
			printf("    %d shuffle indices, %d integer operations\n",
			    p->nidx, p->idxops);
	}
	printf("  proof: the plan run on symbols in all 32 lanes gives every "
	    "element of D as exactly its %d products and its C, each once "
	    "(exhaustive over the lanes)\n", nterm);
	rep_proof("exhaustive", "the plan run on symbols in all 32 lanes: "
	    "every element of D is exactly its K products and its C, each once");
	printf("  bound: |D - (A B + C)| <= gamma_%d (sum_k |A_ik B_kj| + "
	    "|C_ij|), gamma_%d = %d u / (1 - %d u) = %.4g, u = 2^-24%s\n",
	    p->depth, p->depth, p->depth, p->depth, gamma_n(p->depth),
	    f32 ? "" : "; then D's f16 rounding, 2^-11 |D| (2^-25 below "
	    "the normals)");
	rep_strf("bound", "gamma_%d (sum_k |A_ik B_kj| + |C_ij|)%s", p->depth,
	    f32 ? "" : " + f16 rounding");
	rep_int("depth", p->depth);
	printf("    a product of two f16 is exact in f32 (22 significant bits, "
	    "exponents within f32's normals), so only the sums round: each "
	    "term through at most %d roundings%s\n", p->depth, p->ndot2 ?
	    " (an fdot2 counted as two)" : "");
	{
		int nx = 0, w, k;
		const char *c1, *c3 = NULL;
		for (w = 0; w < 3; w++)
			for (k = 0; k < 3; k++)
				nx += p->nx[k][w];
		c1 = nx ? "every lane of the warp active, the warp a subgroup of 32 "
		    "or 32 aligned lanes of a larger one" : avail(m, C_SG_BARRIER) ?
		    "the warp within one subgroup, every lane active" :
		    "every invocation of the workgroup calls it (a workgroup "
		    "barrier), each warp 32 consecutive invocations";
		if (p->ndot2)
			c3 = "an fdot2 rounds at most twice (SPV_VALVE_mixed_float_"
			    "dot_product leaves its rounding to the device)";
		printf("  holds when: %s; the f32 sums round to nearest%s%s\n", c1,
		    c3 ? "; " : "", c3 ? c3 : "");
		rep_cond("%s", c1);
		rep_cond("the f32 sums round to nearest");
		if (c3)
			rep_cond("%s", c3);
	}
	worst = check(p, &S, 64, seed, &bad);
	printf("  check: 64 random sets of fragments in f32 against the exact "
	    "sums: %s, the largest error %.3g of the bound (sampled, not a "
	    "proof)\n", bad ? "OVER THE BOUND" : "within the bound", worst);
	rep_check("random", 64L * S.M * S.N, worst, !bad, "random fragments "
	    "against the exact sums, as a fraction of the bound");
	sb_clear(&t);
	plan_text(&t, p, m);
	sb_clear(&why);
	sb_printf(&why, "// D = A B + C on a warp's fragments as PTX's "
	    "mma.sync.aligned.%s.row.col\n// lays them out (f16 A and B, %s C "
	    "and D), for a GPU without a matrix unit.\n// ucas mma %s %s under "
	    "%s: %s;\n// cost %g (each word fetched from its lane: %g).\n// "
	    "Proved over the 32 lanes: each element of D is its %d products "
	    "and C, each once;\n// |D - (A B + C)| <= gamma_%d (sum_k |A_ik "
	    "B_kj| + |C_ij|)%s, u = 2^-24.\n", S.name, f32 ? "f32" : "f16",
	    S.name, f32 ? "f32" : "f16", sym_str(m->name), sb_str(&t), p->cost,
	    ref.cost, S.K, p->depth, f32 ? "" : ", then D's f16 rounding");
	glsl(&g, p, &S, m, fname, sb_str(&why));
	if (val) {
		sb_clear(&why);
		i = validate(fname, sb_str(&g), &S, &why);
		if (i < 0)
			printf("  validate: glslangValidator or spirv-val is not on "
			    "PATH\n");
		else
			printf("  validate: a compute shader of 32 lanes calling it "
			    "%s glslangValidator and spirv-val%s%s\n", i ? "passes" :
			    "FAILS", i ? "" : ": ", i ? "" : sb_str(&why));
		if (i >= 0)
			rep_check("glslang", 1, 0, i, "glslangValidator and spirv-val");
	}
	snprintf(file, sizeof file, "%s.glsl", fname);
	printf("  GLSL (%s):\n", file);
	{
		const char *q = sb_str(&g), *e;
		for (; *q; q = e) {
			e = strchr(q, '\n');
			e = e ? e + 1 : q + strlen(q);
			printf("    %.*s", (int)(e - q), q);
		}
	}
	rep_str("glsl", sb_str(&g));
	deliver_file(file, "text/plain", sb_str(&g), g.n, nofile);
	/* what measuring the wide shared accesses could change */
	if (!mma_stopped &&
	    !(avail(m, C_LDS_X2) && avail(m, C_LDS_X4) && avail(m, C_STS_X2) &&
	    avail(m, C_STS_X4)) && isfinite(price(m, C_LOAD_SHARED))) {
		double chosen = p->cost, wb = INFINITY;
		struct mplan *w = NULL;
		/* This is a second, full search.  Run it under the same absolute
		 * deadline as the first; if that budget runs out, skip the
		 * speculation rather than hold the caller for another 6 s. */
		mma_deadline_set = mma_budget_s > 0;
		if (setjmp(mma_deadline_jmp) == 0) {
			whatif_vec = 1;
			search(&S, m);
		}
		whatif_vec = 0;
		mma_deadline_set = 0;
		if (mma_stopped) {
			printf("  the optional what-if comparison was cut by the "
			    "search budget; the chosen dataflow above is unaffected\n");
			goto done;
		}
		for (f = 0; f < F_N; f++)
			for (ta = 0; ta < T_N; ta++)
				for (tb = 0; tb < T_N; tb++)
					if (best[f][ta][tb] && best[f][ta][tb]->cost < wb) {
						wb = best[f][ta][tb]->cost;
						w = best[f][ta][tb];
					}
		if (w != NULL && wb < chosen * 0.995) {
			sb_clear(&t);
			plan_text(&t, w, m);
			printf("  what if: were a 64- or 128-bit shared access priced as "
			    "one 32-bit access (load_shared_x2, load_shared_x4, "
			    "store_shared_x2, store_shared_x4, which %s does not list): "
			    "%g, %.2fx cheaper (%s)\n", sym_str(m->name), wb, chosen / wb,
			    sb_str(&t));
			rep_num("whatif_wide_cost", wb);
		}
	}
done:
	free(why.p);
	free(t.p);
	free(g.p);
}

int
mma_command(struct P *p)
{
	char w[48];
	uint32_t at = cmdp_pos(p);
	struct model *m = model_active();
	int k16 = 1, f32 = 1, all = 0, val = 0, nofile = 0, tm = 1, tn = 1, tk = 1;
	char which;
	int64_t v;
	uint64_t seed = 1;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if ((strcmp(w, "mma") != 0 && strcmp(w, "hmma") != 0) ||
	    func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	mma_budget_s = MMA_BUDGET_S;	/* per-command default */
	while (!cmdp_eos(p)) {
		if (cmdp_kw(p, "m16n8k16"))
			k16 = 1;
		else if (cmdp_kw(p, "m16n8k8"))
			k16 = 0;
		else if (cmdp_kw(p, "f32"))
			f32 = 1;
		else if (cmdp_kw(p, "f16"))
			f32 = 0;
		else if (cmdp_kw(p, "all"))
			all = 1;
		else if (cmdp_kw(p, "validate"))
			val = 1;
		else if (cmdp_kw(p, "nofile"))
			nofile = 1;
		else if (cmdp_kw(p, "seconds") || cmdp_kw(p, "budget")) {
			double s = cmdp_number(p, NULL);
			if (!isfinite(s) ||
			    !(s == 0 || (s >= 0.5 && s <= MMA_BUDGET_MAX)))
				fail("mma: seconds N, 0 or %g to %g", 0.5,
				    MMA_BUDGET_MAX);
			mma_budget_s = s;
		}
		else if (cmdp_kw(p, "chain")) {
			if (!cmdp_int(p, &v) || v < 1 || v > MAXTILE)
				fail("mma: chain N, 1 to %d HMMAs each one's D the next "
				    "one's C", MAXTILE);
			tk = (int)v;
		} else if (cmdp_kw(p, "tile")) {
			int64_t a, b2, c2;
			if (!cmdp_int(p, &a) || !cmdp_int(p, &b2) || !cmdp_int(p, &c2) ||
			    a < 1 || b2 < 1 || c2 < 1 || a * b2 * c2 > MAXTILE)
				fail("mma: tile TM TN TK, HMMAs along m, n and k, at most "
				    "%d in all", MAXTILE);
			tm = (int)a;
			tn = (int)b2;
			tk = (int)c2;
		} else if ((which = cmdp_kw(p, "tm") ? 'm' : cmdp_kw(p, "tn") ? 'n' :
		    cmdp_kw(p, "tk") ? 'k' : 0) != 0) {
			/* one of a tile's counts (the JSON frontend's form) */
			if (!cmdp_int(p, &v) || v < 1 || v > MAXTILE)
				fail("mma: tm, tn, tk N: 1 to %d HMMAs along m, n, k",
				    MAXTILE);
			if (which == 'm')
				tm = (int)v;
			else if (which == 'n')
				tn = (int)v;
			else
				tk = (int)v;
		} else if (cmdp_kw(p, "seed")) {
			if (!cmdp_int(p, &v) || v < 0)
				fail("mma: seed N, a whole number");
			seed = (uint64_t)v;
		} else if (cmdp_kw(p, "model") || cmdp_kw(p, "under") ||
		    cmdp_kw(p, "on")) {
			if (!cmdp_word(p, w, sizeof w) ||
			    (m = model_find(sym_cstr(w))) == NULL)
				fail("mma: unknown model (see 'models')");
		} else {
			cmdp_fail_here(p);
			fail("mma [m16n8k16 | m16n8k8] [f32 | f16] [chain N | tile TM TN "
			    "TK] [model M] [all] [validate] [nofile] [seed N] "
			    "[seconds N]");
		}
	}
	notes_flush();
	if (tm * tn * tk > MAXTILE)
		fail("mma: a tile of at most %d HMMAs", MAXTILE);
	mma_run(k16, f32, tm, tn, tk, m, all, val, nofile, seed);
	return 1;
}

/* ---- English ---- */

/*
 * "emulate hmma on rdna2", "a warp matrix multiply m16n8k8 with f16
 * accumulators under gt1030", "tensor core mma without cooperative
 * matrices", "mma.sync m16n8k16 f32"
 */
const char *
mma_nlp_raw(const char *sent)
{
	static char out[200];
	static const char *const keys[] = {"hmma", "mma", "tensor core",
		"tensor-core", "matrix multiply-accumulate", "matrix multiply "
		"accumulate", "warp matrix", "mma.sync", NULL};
	char buf[600], *s, *q, model[48] = "";
	int i, k16 = 1, f32 = 1, hit = 0;
	size_t n;

	n = strlen(sent);
	if (n >= sizeof buf)
		return NULL;
	for (i = 0; (size_t)i <= n; i++)
		buf[i] = (char)(sent[i] >= 'A' && sent[i] <= 'Z' ? sent[i] + 32 :
		    sent[i]);
	s = buf;
	for (i = 0; keys[i]; i++) {
		const char *h = strstr(s, keys[i]);
		size_t kn = strlen(keys[i]);
		if (h && (h == s || !(h[-1] >= 'a' && h[-1] <= 'z')) &&
		    !(h[kn] >= 'a' && h[kn] <= 'z'))
			hit = 1;
	}
	if (!hit)
		return NULL;
	/* not a definition or another command's statement */
	if (strchr(s, '=') || strchr(s, '('))
		return NULL;
	if (strstr(s, "k8") && !strstr(s, "k16"))
		k16 = 0;
	if (strstr(s, "f16 acc") || strstr(s, "half acc") ||
	    strstr(s, "f16 c") || strstr(s, "f16 output") ||
	    strstr(s, "f16 result") || (strstr(s, "f16") && !strstr(s, "f32") &&
	    strstr(s, "accum")))
		f32 = 0;
	for (q = s; (q = strpbrk(q, " ")) != NULL; q++) {
		const char *w = q + 1;
		size_t wn = strcspn(w, " ?.,");
		if (wn && wn < sizeof model && (q - s >= 3) &&
		    (!strncmp(q - 3, " on", 3) || !strncmp(q - 3, "der", 3) ||
		    !strncmp(q - 3, "for", 3) || !strncmp(q - 3, "del", 3)) &&
		    model_find(sym_intern(w, wn)) != NULL) {
			memcpy(model, w, wn);
			model[wn] = 0;
		}
	}
	snprintf(out, sizeof out, "mma %s %s%s%s%s%s%s", k16 ? "m16n8k16" :
	    "m16n8k8", f32 ? "f32" : "f16", model[0] ? " model " : "", model,
	    strstr(s, "validat") ? " validate" : "", strstr(s, "nofile") ?
	    " nofile" : "", strstr(s, "every dataflow") || strstr(s,
	    "all dataflows") ? " all" : "");
	return out;
}
