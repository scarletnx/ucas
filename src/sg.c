/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * sg.c: operations on the lanes of a subgroup.
 *
 * A subgroup is W invocations that run in lockstep (a warp, a wave), lane
 * 0 to lane W - 1.  A value is per lane or shared by the lanes (a
 * `uniform` parameter, a constant).  The operations here read a value in
 * every lane at once, and are defined exactly, for a full subgroup (all W
 * lanes active):
 *
 *   sg_ballot(p)       the word whose bit l is p in lane l (u32; u64 when
 *                      W > 32): OpGroupNonUniformBallot
 *   sg_any(p), sg_all(p)
 *                      p in some lane, p in every lane (bool)
 *   sg_add(v)          the sum of v over the lanes, in v's type (wrapping):
 *                      OpGroupNonUniformIAdd, Reduce
 *   sg_shuffle(v, i)   v in lane i; undefined when i is not in 0..W-1:
 *                      OpGroupNonUniformShuffle
 *   sg_scan(v)         in lane l, the sum of v over lanes 0..l (wrapping):
 *                      OpGroupNonUniformIAdd, InclusiveScan
 *
 * and, written with them: sg_exclusive(v) = sg_scan(v) - v,
 * sg_shuffle_xor(v, k) = sg_shuffle(v, lane xor k), sg_broadcast(v, i) =
 * sg_shuffle(v, i), sg_broadcast_first(v) = sg_shuffle(v, 0),
 * sg_rotate(v, d) = sg_shuffle(v, (lane + d) mod W), and sg_sort(v), in
 * lane l the l-th smallest of v over the lanes: the bitonic network
 * (sg_sorted).
 *
 * In an operand that is a value per lane, the lane's index is the bound
 * variable `lane` (slot SG_LANE_SLOT); everything else in it is shared by
 * the lanes, so a per-lane input is written as a function of the lane: a
 * bit or a field of a shared word, an entry of a table.  That is what
 * makes the operations exact here: spelled out lane by lane (sg_expand)
 * they are ordinary expressions, which is how they are evaluated, bounded
 * and bit-blasted (sg_lower, called by plan_build and nodesat's blast).
 * The node keeps the operation, for printing and for the price: one
 * instruction of the model's class, its operand computed once in each
 * lane.
 *
 * Node layout: a the value per lane; sg_shuffle b the index, sg_scan b
 * the lane whose prefix it is; the last operand (b, or c for those two)
 * the constant W.
 */
#include "calc.h"
#include "fuse.h"
#include "irw.h"
#include "sg.h"

#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

nid
sg_lane(void)
{
	nid n = mk_var(SG_LANE_SLOT, sym_cstr("lane"), TY(K_UINT, 32));

	N.fl[n] |= NF_LANE;
	return n;
}

int
sg_is_lane(nid n)
{
	nid x = n;

	if (N.op[x] == O_CAST)
		x = N.a[x];
	return N.op[x] == O_VAR && N.a[x] == sym_cstr("lane");
}

int
sg_class(int op)
{
	switch (op) {
	case O_SGBALLOT:
		return C_SG_BALLOT;
	case O_SGANY: case O_SGALL:
		return C_SG_VOTE;
	case O_SGADD:
		return C_SG_ADD;
	case O_SGSHUF:
		return C_SG_SHUF;
	}
	return C_SG_SCAN;
}

int
sg_width(nid n)
{
	nid w = opinfo[N.op[n]].arity == 3 ? N.c[n] : N.b[n];

	return (int)N.ki[w];
}

/* e in lane l */
static nid
in_lane(nid e, int l)
{
	nid map[SG_LANE_SLOT + 1];

	if (!(N.fl[e] & NF_LANE))
		return e;
	memset(map, 0, sizeof map);
	map[SG_LANE_SLOT] = mk_const_i(TY(K_UINT, 32), l);
	return subst(e, map, SG_LANE_SLOT + 1);
}

static ty_t
sg_type(int op, nid a, int W)
{
	switch (op) {
	case O_SGBALLOT:
		return TY(K_UINT, W > 32 ? 64 : 32);
	case O_SGANY: case O_SGALL:
		return TY_BOOL;
	}
	return N.ty[a];
}

/* a parameter's copy in e (sg_lane_copy), or 0 */
static nid
lane_copy_in(nid e, int depth)
{
	nid r;
	uint32_t j;

	if (e == 0 || !(N.fl[e] & NF_LANEREAD) || depth > 200)
		return 0;
	if (N.op[e] == O_VAR)
		return e;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((r = lane_copy_in(N.lst[N.a[e] + j],
			    depth + 1)) != 0)
				return r;
		return 0;
	}
	if ((r = lane_copy_in(N.a[e], depth + 1)) != 0 ||
	    (r = lane_copy_in(N.b[e], depth + 1)) != 0)
		return r;
	return lane_copy_in(N.c[e], depth + 1);
}

nid
sg_expand(int op, nid a, nid b, nid c)
{
	int W = (int)N.ki[opinfo[op].arity == 3 ? c : b], l;
	ty_t t = sg_type(op, a, W);
	nid r = 0, e[SG_MAXW], v;

	/* the other lanes' values of a parameter are not known here */
	if ((v = lane_copy_in(a, 0)) != 0 && N.kr[v] != 0) {
		/* a loop's state that follows one */
		const char *from = sym_str((uint32_t)N.kr[v]);
		fail("%s reads %s in every lane, and %s is a loop's value that "
		    "follows %s, a value per lane: each lane has its own %s, and "
		    "ucas does not run the loop in every lane at once to know the "
		    "others', so the value depends on the other lanes and has "
		    "none here (cost prices it at the worst%s%s%s)",
		    opinfo[op].name, sym_str(N.a[v]), sym_str(N.a[v]), from,
		    sym_str(N.a[v]), strcmp(from, "lane") ? "; declare " : "",
		    strcmp(from, "lane") ? from : "", strcmp(from, "lane") ?
		    " uniform for an exact value" : "");
	}
	if (v != 0)
		fail("%s reads %s in every lane, and %s has a value in each "
		    "lane that ucas does not know for the others: the value "
		    "depends on the other lanes, so it has none here (cost "
		    "prices it at the worst; declare %s uniform for an exact "
		    "value)", opinfo[op].name, sym_str(N.a[v]), sym_str(N.a[v]),
		    sym_str(N.a[v]));

	switch (op) {
	case O_SGBALLOT:
		r = mk_const_i(t, 0);
		for (l = 0; l < W; l++)
			r = mk_bin(O_OR, r, mk_sel(in_lane(a, l),
			    mk_const_i(t, (int64_t)(1ULL << l)), mk_const_i(t, 0)));
		break;
	case O_SGANY:
		r = mk_const_i(TY_BOOL, 0);
		for (l = 0; l < W; l++)
			r = mk_bin(O_LOR, r, in_lane(a, l));
		break;
	case O_SGALL:
		r = mk_const_i(TY_BOOL, 1);
		for (l = 0; l < W; l++)
			r = mk_bin(O_LAND, r, in_lane(a, l));
		break;
	case O_SGADD:
		r = in_lane(a, 0);
		for (l = 1; l < W; l++)
			r = mk_bin(O_ADD, r, in_lane(a, l));
		break;
	case O_SGSHUF:
		for (l = 0; l < W; l++)
			e[l] = in_lane(a, l);
		r = mk_index(mk_table(e, (uint32_t)W), b);
		break;
	case O_SGSCAN:
		r = mk_const_i(t, 0);
		for (l = 0; l < W; l++)
			r = mk_bin(O_ADD, r, mk_sel(mk_bin(O_LE,
			    mk_const_i(TY_LIT, l), b), in_lane(a, l),
			    mk_const_i(t, 0)));
		break;
	}
	return mk_cast(t, r);
}

nid
mk_sg(int op, nid a, nid b, nid c)
{
	int ar = opinfo[op].arity, W, fl = NF_SG;
	nid w = ar == 3 ? c : b, r;
	const char *nm = opinfo[op].name;
	ty_t t;

	if (!w || !is_const(w) || !TY_ISINT(N.ty[w]) || N.ki[w] < 2 ||
	    N.ki[w] > SG_MAXW)
		fail("%s: a subgroup has 2 to %d lanes", nm, SG_MAXW);
	W = (int)N.ki[w];
	w = mk_const_i(TY_LIT, W);
	switch (op) {
	case O_SGBALLOT: case O_SGANY: case O_SGALL:
		a = mk_bool(a);
		break;
	case O_SGADD: case O_SGSCAN:
		if (TY_FP(N.ty[a]))
			fail("%s of floats: the order of the additions is the "
			    "implementation's, so the sum has no exact value; "
			    "integers only", nm);
		if (N.ty[a] == TY_BOOL || TYK(N.ty[a]) == K_LIT)
			a = mk_cast(TY(K_UINT, 32), a);
		break;
	}
	t = sg_type(op, a, W);
	if (ar == 3 && !TY_ISINT(N.ty[b]))
		fail("%s: a lane's index is an integer", nm);
	if (N.op[a] == O_UNDEF || (ar == 3 && N.op[b] == O_UNDEF))
		return mk_undef(t);
	/* nothing but the lane in it: the same value whatever the inputs */
	if (!node_slots(a) && !(N.fl[a] & NF_LANEREAD) && (ar == 2 ||
	    (!node_slots(b) && !(N.fl[b] & (NF_LANE | NF_LANEREAD)))))
		return sg_expand(op, (N.fl[a] & NF_SG) ? sg_lower_once(a) : a,
		    ar == 3 && (N.fl[b] & NF_SG) ? sg_lower_once(b) : b, w);
	fl |= (N.fl[a] | (ar == 3 ? N.fl[b] : 0)) & NF_PARTIAL;
	if (op == O_SGSHUF)
		fl |= NF_PARTIAL;
	r = ar == 3 ? mk_raw(op, t, a, b, w, 0, fl) :
	    mk_raw(op, t, a, w, 0, 0, fl);
	/* the operation binds the lane of its value per lane */
	N.fl[r] &= ~NF_LANE;
	if (ar == 3 && (N.fl[b] & NF_LANE))
		N.fl[r] |= NF_LANE;
	return r;
}

/*
 * e with its subgroup operations spelled out.  Nodes are never freed, so
 * the results are kept by node.
 */
static nid *lowered;
static uint32_t nlowered;

nid
sg_lower(nid e)
{
	nid r;

	if (e == 0 || !(N.fl[e] & NF_SG))
		return e;
	if (e < nlowered && lowered[e])
		return lowered[e];
	r = sg_lower_once(e);
	if (e >= nlowered) {
		uint32_t n = N.n + 1024;
		lowered = xrealloc(lowered, n * sizeof *lowered);
		memset(lowered + nlowered, 0, (n - nlowered) * sizeof *lowered);
		nlowered = n;
	}
	lowered[e] = r;
	return r;
}

/*
 * A parameter of mask (bit = slot) that some subgroup operation of e reads
 * in its value per lane: its slot + 1, or 0.  Such an operand may read
 * the lane and what the lanes share, nothing else.
 */
static int
sg_reads1(nid e, uint64_t mask, uint8_t *seen)
{
	uint32_t j;
	int r;

	if (e == 0 || !(N.fl[e] & NF_SG) || seen[e])
		return 0;
	seen[e] = 1;
	if (IS_SG(N.op[e])) {
		uint64_t m = node_slots(N.a[e]) & mask;
		for (r = 0; r < 64; r++)
			if ((m >> r) & 1)
				return r + 1;
	}
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((r = sg_reads1(N.lst[N.a[e] + j], mask, seen)) != 0)
				return r;
		return 0;
	}
	if ((r = sg_reads1(N.a[e], mask, seen)) != 0 ||
	    (r = sg_reads1(N.b[e], mask, seen)) != 0)
		return r;
	return sg_reads1(N.c[e], mask, seen);
}

int
sg_reads(nid e, uint64_t mask)
{
	uint8_t *seen;
	int r;

	if (e == 0 || !(N.fl[e] & NF_SG) || !mask)
		return 0;
	seen = xcalloc(N.n, 1);
	r = sg_reads1(e, mask, seen);
	free(seen);
	return r;
}

int
sg_has_lane(nid e)
{
	return e && (N.fl[e] & NF_LANE) != 0;
}

/* ---- sorting the lanes ---- */

/*
 * Batcher's bitonic network over W = 2^m lanes: for k = 2, 4 ... W, then
 * j = k/2 ... 1, each lane and lane xor j compare and exchange, the lane
 * keeping the smaller when (lane & j) == 0 and (lane & k) == 0 agree: the
 * aligned blocks of k lanes come out ascending where lane & k is 0,
 * descending elsewhere, and the last, k = W, ascending.  m(m + 1)/2 steps
 * of a shuffle, a min and a max.
 */
static int
keeps_min(int l, int j, int k)
{
	return ((l & j) == 0) == ((l & k) == 0);
}

/* keeps_min at lane l; in the last stage (k = W) lane & k is 0 */
static nid
keeps_min_at(nid l, int j, int k, int W)
{
	ty_t u = TY(K_UINT, 32);
	nid z = mk_const_i(u, 0);
	nid lo = mk_bin(O_EQ, mk_bin(O_AND, l, mk_const_i(u, j)), z);

	return k == W ? lo : mk_bin(O_EQ, lo, mk_bin(O_EQ, mk_bin(O_AND, l,
	    mk_const_i(u, k)), z));
}

/* one compare and exchange: at lane l, s and p the values of l and of its
 * partner */
static nid
exchange(nid s, nid p, nid l, int j, int k, int W)
{
	return mk_sel(keeps_min_at(l, j, k, W), mk_bin(O_MIN, s, p),
	    mk_bin(O_MAX, s, p));
}

nid
sg_sorted(nid v, nid here, nid at, int W)
{
	nid w = mk_const_i(TY_LIT, W), L = sg_lane(), sb = v, sl = 0;
	nid map[SG_LANE_SLOT + 1];
	ty_t u = TY(K_UINT, 32);
	int j, k;

	if (W < 2 || (W & (W - 1)))
		return 0;
	if (!at) {
		memset(map, 0, sizeof map);
		map[SG_LANE_SLOT] = here = mk_cast(u, here);
		sl = subst(v, map, SG_LANE_SLOT + 1);
	}
	/* sb is the value in every lane (the shuffles read it), sl in here */
	for (k = 2; k <= W; k *= 2)
		for (j = k / 2; j >= 1; j /= 2) {
			nid kj = mk_const_i(u, j), pb, pl;
			pb = mk_sg(O_SGSHUF, sb, mk_bin(O_XOR, L, kj), w);
			if (!at) {
				pl = mk_sg(O_SGSHUF, sb, mk_bin(O_XOR, here, kj),
				    w);
				sl = exchange(sl, pl, here, j, k, W);
			}
			sb = exchange(sb, pb, L, j, k, W);
		}
	return at ? mk_sg(O_SGSHUF, sb, mk_cast(u, at), w) : sl;
}

/*
 * The network sorts every input: by the 0-1 principle it is enough that
 * it sorts every input of 0s and 1s, and that follows from each stage.
 * After the stage for k/2 the two halves of a block of k lanes are sorted
 * in opposite directions, the lower ascending (keeps_min), so a block of
 * 0s and 1s is 0^a 1^b 0^c; the merge of the stage for k (j = k/2 ... 1)
 * is run on each of those, at a block of each direction, and must sort
 * it.  Checked once per W.
 */
/* the merge of the stage for k, at the block of k lanes from base, sorts
 * 0^a 1^b 0^(k - a - b) in the block's direction */
static int
merge_sorts(int base, int k, int a, int b)
{
	int x[SG_MAXW], i, j, up = (base & k) == 0;

	for (i = 0; i < k; i++)
		x[i] = i >= a && i < a + b;
	for (j = k / 2; j >= 1; j /= 2)
		for (i = 0; i < k; i++) {
			int pi = i ^ j, lo, hi;
			if (pi < i)
				continue;
			lo = x[i] < x[pi] ? x[i] : x[pi];
			hi = x[i] ^ x[pi] ^ lo;
			x[i] = keeps_min(base + i, j, k) ? lo : hi;
			x[pi] = keeps_min(base + pi, j, k) ? lo : hi;
		}
	for (i = 0; i + 1 < k; i++)
		if (up ? x[i] > x[i + 1] : x[i] < x[i + 1])
			return 0;
	return 1;
}

static int
sort_proved(int W, int *cases)
{
	static int8_t known[SG_MAXW + 1];
	static int ncase[SG_MAXW + 1];
	int k, base, a, b, ok = 1, n = 0;

	if (known[W]) {
		*cases = ncase[W];
		return known[W] > 0;
	}
	for (k = 2; k <= W && ok; k *= 2)
		for (base = 0; base < (k < W ? 2 * k : k) && ok; base += k)
			for (a = 0; a <= k && ok; a++)
				for (b = 0; a + b <= k && ok; b++, n++)
					ok = merge_sorts(base, k, a, b);
	known[W] = ok ? 1 : -1;
	ncase[W] = n;
	*cases = n;
	return ok;
}

/* ---- the active lanes (sg_active) ---- */

static nid *actbits;	/* the tests, then each one's mask and lane */
static int nactbits;

nid
sg_active_bit(nid mask, nid l)
{
	ty_t t = N.ty[mask];
	nid r = mk_bin(O_NE, mk_bin(O_AND, mk_bin(O_SHR, mask, mk_cast(t, l)),
	    mk_const_i(t, 1)), mk_const_i(t, 0));
	int k;

	for (k = 0; k < nactbits && actbits[k] != r; k += 3)
		;
	if (k == nactbits) {
		actbits = xrealloc(actbits, (size_t)(nactbits + 3) *
		    sizeof *actbits);
		actbits[nactbits++] = r;
		actbits[nactbits++] = mask;
		actbits[nactbits++] = l;
	}
	return r;
}

nid
sg_active_of(nid bit, nid *lane)
{
	int k;

	for (k = 0; k < nactbits; k += 3)
		if (actbits[k] == bit) {
			if (lane)
				*lane = actbits[k + 2];
			return actbits[k + 1];
		}
	return 0;
}

/* e under sg_active: the mask its operations take, or 0 */
static nid
active_mask_in(nid e, uint8_t *seen)
{
	uint32_t j;
	int k;
	nid r;

	if (e == 0 || seen[e] || N.op[e] == O_CONST || N.op[e] == O_VAR)
		return 0;
	seen[e] = 1;
	for (k = 0; k < nactbits; k += 3)
		if (actbits[k] == e)
			return actbits[k + 1];
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((r = active_mask_in(N.lst[N.a[e] + j], seen)) != 0)
				return r;
		return 0;
	}
	if ((r = active_mask_in(N.a[e], seen)) != 0 ||
	    (r = active_mask_in(N.b[e], seen)) != 0)
		return r;
	return active_mask_in(N.c[e], seen);
}

/* ---- find: forms the model's subgroup operations make cheaper ---- */

/*
 * Each form is built by a recogniser, priced under the active model as
 * one lane runs it (sg_priced), kept only when the whole function gets
 * cheaper, and comes with its derivation and what it holds under; the
 * whole is then checked against the function on its domain.  Nothing
 * here knows a target: a model without the operation (or without a
 * subgroup size) prices none of these.
 *
 *   a scan or a sum of a value that is 0 or 1 in every lane (its range,
 *     by intervals): a count of ballot bits;
 *   a scan of anything: a ladder of ceil(log2 W) shuffles and adds;
 *   a constant table of at most W entries, held one entry per lane and
 *     read by a shuffle;
 *   a GF(2)-linear function of a word the lanes share: each lane works
 *     out one bit of the result and the ballot puts the bits together;
 *   an order statistic of a value per lane, however written: the
 *     bitonic network, read where it is needed.
 */
struct sgf {
	const struct func *f;
	const struct model *m;
	nid lanep;		/* f's lane parameter as a u32, or 0 */
	struct sbuf why;
	struct sbuf notes;	/* derived, but not cheaper */
};

static nid
lane_param(const struct func *f)
{
	uint32_t nm = sym_cstr("lane");
	int j;

	for (j = 0; j < f->np; j++)
		if (f->p[j].name == nm && TY_ISINT(f->p[j].ty))
			return mk_cast(TY(K_UINT, 32), mk_var((uint32_t)j, nm,
			    f->p[j].ty));
	return 0;
}

nid
sg_priced(const struct func *f, nid e)
{
	nid lp = lane_param(f);

	return lp ? sg_flat(e, lp) : e;
}

static float
price(const struct sgf *g, nid e)
{
	return fuse_expr_cost(g->m, sg_priced(g->f, e));
}

/* root with from replaced by to; 0 when the result cannot be built */
nid
sg_replaced(nid root, nid from, nid to)
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

/*
 * The proved range of e, a value in a lane (its lane running over
 * 0..W-1): the parameters' ranges narrowed by the where clause, then
 * intervals.
 */
int
sg_range_of(const struct func *f, nid e, int W, int64_t *lo, int64_t *hi)
{
	struct func g = *f;
	int64_t blo[MAXP], bhi[MAXP];
	nid roots[2];
	struct plan p;
	struct ival *iv;
	uint32_t l;
	int j, ok;

	if (is_const(e) && TY_ISINT(N.ty[e])) {
		*lo = *hi = N.ki[e];
		return 1;
	}
	if (!TY_ISINT(N.ty[e]))
		return 0;
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &blo[j], &bhi[j]);
	where_bounds(f, blo, bhi, NULL);
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty) && !(TYK(f->p[j].ty) == K_UINT &&
		    TYW(f->p[j].ty) == 64)) {
			g.p[j].ranged = 1;
			g.p[j].lo = blo[j];
			g.p[j].hi = bhi[j];
		}
	if (sg_has_lane(e)) {
		/* the bound lane as one more parameter, 0..W-1 */
		nid map[SG_LANE_SLOT + 1];
		struct param *pp;
		if (g.np == MAXP)
			return 0;
		pp = &g.p[g.np];
		memset(pp, 0, sizeof *pp);
		pp->name = sym_cstr("lane");
		pp->ty = TY(K_UINT, 32);
		pp->ranged = 1;
		pp->hi = W - 1;
		memset(map, 0, sizeof map);
		map[SG_LANE_SLOT] = mk_var((uint32_t)g.np, pp->name, pp->ty);
		g.np++;
		e = subst(e, map, SG_LANE_SLOT + 1);
	}
	roots[0] = e;
	roots[1] = f->where;
	plan_build(&p, roots, 2, 0);
	iv = xcalloc(p.n ? p.n : 1, sizeof *iv);
	plan_ival(&p, &g, iv);
	l = plan_loc(&p, e);
	ok = l != NOTOK && !iv[l].top && !iv[l].wrap;
	if (ok) {
		*lo = iv[l].lo;
		*hi = iv[l].hi;
	}
	free(iv);
	plan_free(&p);
	return ok;
}

/* the ballot of v != 0, v a value per lane */
static nid
ballot_of(nid v, int W)
{
	return mk_sg(O_SGBALLOT, mk_bin(O_NE, v, mk_const_i(TY_LIT, 0)),
	    mk_const_i(TY_LIT, W), 0);
}

/*
 * The scan of v by a ladder (Hillis and Steele): s = v; then for d = 1,
 * 2, 4 ... below W, s = s + (lane >= d ? s in lane (lane - d) : 0).
 * After the step for d a lane holds the sum of its 2d lanes ending at
 * itself.  v is the value per lane, l the lane the result is for.
 */
static nid
ladder(nid v, nid l, int W)
{
	nid w = mk_const_i(TY_LIT, W), L = sg_lane(), zero, sb = v, sl;
	nid map[SG_LANE_SLOT + 1];
	int d;

	memset(map, 0, sizeof map);
	map[SG_LANE_SLOT] = l = mk_cast(TY(K_UINT, 32), l);
	sl = subst(v, map, SG_LANE_SLOT + 1);
	zero = mk_const_i(N.ty[v], 0);
	for (d = 1; d < W; d *= 2) {
		nid k = mk_const_i(TY(K_UINT, 32), d);
		nid nl = mk_bin(O_ADD, sl, mk_sel(mk_bin(O_GE, l, k),
		    mk_sg(O_SGSHUF, sb, mk_bin(O_SUB, l, k), w), zero));
		sb = mk_bin(O_ADD, sb, mk_sel(mk_bin(O_GE, L, k),
		    mk_sg(O_SGSHUF, sb, mk_bin(O_SUB, L, k), w), zero));
		sl = nl;
	}
	return sl;
}

/*
 * The ladder is the scan whatever v is: with a free word v_j in each
 * lane, lane l's result must be v_0 + ... + v_l regrouped (+ associates
 * and commutes mod 2^N), every lane once.  Checked once per W.
 */
#define LEAF0 200	/* slots of the free words: no parameter, no loop */

static int
sum_leaves(nid e, int *cnt, int W)
{
	if (N.op[e] == O_ADD)
		return sum_leaves(N.a[e], cnt, W) && sum_leaves(N.b[e], cnt, W);
	if (N.op[e] == O_VAR && N.ki[e] >= LEAF0 && N.ki[e] < LEAF0 + W) {
		cnt[N.ki[e] - LEAF0]++;
		return 1;
	}
	return is_const(e) && N.ki[e] == 0;
}

static int
ladder_proved(int W)
{
	static int8_t known[SG_MAXW + 1];
	nid e[SG_MAXW], v;
	int l, j, ok = 1;

	if (known[W])
		return known[W] > 0;
	for (l = 0; l < W; l++)
		e[l] = mk_var((uint32_t)(LEAF0 + l), sym_cstr("v"), TY(K_UINT, 32));
	v = mk_index(mk_table(e, (uint32_t)W), sg_lane());
	for (l = 0; l < W && ok; l++) {
		int cnt[SG_MAXW] = {0};
		nid r = sg_lower_once(ladder(v, mk_const_i(TY(K_UINT, 32), l), W));
		ok = sum_leaves(r, cnt, W);
		for (j = 0; j < W && ok; j++)
			ok = cnt[j] == (j <= l);
	}
	known[W] = ok ? 1 : -1;
	return ok;
}

/* the subgroup operations of e (kind 0: sums and scans; 2: shuffles) or
 * its lookups in constant tables (kind 1), in out (at most max) */
static int
gather(nid e, int kind, nid *out, int n, int max, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || n == max)
		return n;
	seen[e] = 1;
	if (kind != 1 && !(N.fl[e] & NF_SG))
		return n;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_UNDEF)
		return n;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			n = gather(N.lst[N.a[e] + j], kind, out, n, max, seen);
		return n;
	}
	if (kind == 0 && (N.op[e] == O_SGADD || N.op[e] == O_SGSCAN))
		out[n++] = e;
	if (kind == 1 && N.op[e] == O_INDEX && !(N.fl[N.a[e]] & NF_VAR))
		out[n++] = e;
	if (kind == 2 && N.op[e] == O_SGSHUF)
		out[n++] = e;
	if (kind == 2 && N.op[e] == O_LOOP) {
		/* the shuffles of a loop over the lanes are in its body */
		uint32_t le = loop_ent(e), st = loops.st[le], k;
		for (k = 0; k < loops.nst[le]; k++) {
			n = gather(loops.init[st + k], kind, out, n, max, seen);
			n = gather(loops.next[st + k], kind, out, n, max, seen);
		}
		n = gather(loops.cond[le], kind, out, n, max, seen);
	}
	n = gather(N.a[e], kind, out, n, max, seen);
	n = gather(N.b[e], kind, out, n, max, seen);
	return gather(N.c[e], kind, out, n, max, seen);
}

#define MAXSGN 64

static int
gather_all(nid e, int kind, nid *out)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int n = gather(e, kind, out, 0, MAXSGN, seen);

	free(seen);
	return n;
}

static void
note_dear(struct sgf *g, const char *what, float c, float base)
{
	if (strstr(sb_str(&g->notes), what))
		return;
	sb_printf(&g->notes, "  subgroup forms: derived, but not cheaper under "
	    "%s (cost %g, as written %g): %s\n", sym_str(g->m->name), (double)c,
	    (double)base, what);
}


/* ---- order statistics of the lanes ---- */

#define SORT_SAMPLES 1024

struct lanev {
	int64_t v;
	ty_t t;
};

static int
cmp_lanev(const void *a, const void *b)
{
	const struct lanev *x = a, *y = b;

	if (TYK(x->t) == K_UINT && TYW(x->t) == 64)
		return (uint64_t)x->v < (uint64_t)y->v ? -1 :
		    (uint64_t)x->v > (uint64_t)y->v;
	return x->v < y->v ? -1 : x->v > y->v;
}

/*
 * Which order statistic of v over the W lanes e is at the domain's points
 * (every one when there are at most SORT_SAMPLES, a sample otherwise):
 * *k = 0..W-1 the k-th smallest wherever it is read, -1 v sorted
 * ascending along the lanes (lane l the l-th smallest), -2 descending.
 * 0 when it is none of them at some point.
 */
static int
order_stat(const struct func *f, nid e, nid v, nid lanep, int W, int *k)
{
	struct diter d;
	struct plan p;
	nid roots[SG_MAXW + 3];
	uint32_t loc[SG_MAXW + 3], np, i, wloc;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	/* the k still possible */
	uint64_t ks = ((uint64_t)1 << (W - 1) << 1) - 1;
	int l, nr = 0, asc = lanep != 0, desc = lanep != 0, any = 0;
	uint8_t ok[CH];

	roots[nr++] = e;
	roots[nr++] = f->where;
	roots[nr++] = lanep;
	for (l = 0; l < W; l++)
		roots[nr++] = in_lane(v, l);
	plan_build(&p, roots, nr, 0);
	for (l = 0; l < nr; l++)
		loc[l] = roots[l] ? plan_loc(&p, roots[l]) : NOTOK;
	wloc = f->where ? loc[1] : NOTOK;
	diter_init(&d, f, SORT_SAMPLES, 2, SORT_SAMPLES);
	while ((ks || asc || desc) && (np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, wloc, np, ok);
		for (i = 0; i < np; i++) {
			struct lanev s[SG_MAXW];
			int64_t r, at = 0;
			int und = (COLF(&p, loc[0])[i] & F_UNDEF) != 0;
			if (!ok[i])
				continue;
			for (l = 0; l < W; l++) {
				und |= (COLF(&p, loc[3 + l])[i] & F_UNDEF) != 0;
				s[l].v = COLI(&p, loc[3 + l])[i];
				s[l].t = N.ty[v];
			}
			if (und)
				continue;
			qsort(s, (size_t)W, sizeof *s, cmp_lanev);
			r = COLI(&p, loc[0])[i];
			any = 1;
			for (l = 0; l < W; l++)
				if (s[l].v != r)
					ks &= ~((uint64_t)1 << l);
			if (lanep) {
				at = COLI(&p, loc[2])[i];
				if (at < 0 || at >= W)
					asc = desc = 0;
				else {
					asc &= s[at].v == r;
					desc &= s[W - 1 - at].v == r;
				}
			}
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	if (!any)
		return 0;
	for (l = 0; l < W; l++)
		if ((ks >> l) & 1) {
			*k = l;
			return 1;
		}
	if (asc || desc) {
		*k = asc ? -1 : -2;
		return 1;
	}
	return 0;
}

/*
 * A function whose value is an order statistic of a value per lane v (the
 * least, the median, the greatest, the k-th smallest; or v sorted along
 * the lanes, the first k lanes then holding the top k), however it is
 * written: a loop over the lanes that counts with ballots, a chain of
 * selections.  It is told by evaluating the function beside the lanes'
 * values sorted, at the domain's points, and written as the bitonic
 * network read at the lane it needs (the whole is then checked).
 */
static nid
sorts(struct sgf *g, nid cur, float *cost)
{
	nid shuf[MAXSGN], cand[MAXSGN], lp = lane_param(g->f);
	int ns, nc = 0, i, j, W, k, cases, m;

	ns = gather_all(cur, 2, shuf);
	for (i = 0; i < ns; i++) {
		nid v = N.a[shuf[i]];
		for (j = 0; j < nc && cand[j] != v; j++)
			;
		if (j == nc && TY_ISINT(N.ty[v]) && N.ty[v] != TY_BOOL &&
		    TYK(N.ty[v]) != K_LIT && (N.fl[v] & NF_LANE) &&
		    !(N.fl[v] & NF_SG))
			cand[nc++] = v;
	}
	for (i = 0; i < nc; i++) {
		nid v = cand[i], at, r;
		float c;
		struct sbuf what = {0};
		W = sg_width(shuf[0]);
		if ((W & (W - 1)) || !order_stat(g->f, cur, v, lp, W, &k) ||
		    !sort_proved(W, &cases))
			continue;
		at = k >= 0 ? mk_const_i(TY(K_UINT, 32), k) : k == -1 ? 0 :
		    mk_bin(O_SUB, mk_const_i(TY(K_UINT, 32), W - 1), lp);
		r = mk_cast(N.ty[cur], sg_sorted(v, lp, at, W));
		c = price(g, r);
		if (k == 0)
			sb_puts(&what, "the least");
		else if (k == W - 1)
			sb_puts(&what, "the greatest");
		else if (k >= 0)
			sb_printf(&what, "the order statistic k = %d (0 the "
			    "least, %d the greatest%s)", k, W - 1, k == W / 2 ||
			    k == W / 2 - 1 ? "; a median" : "");
		else if (k == -1)
			sb_puts(&what, "sorted ascending along the lanes");
		else
			sb_puts(&what, "sorted descending along the lanes (the "
			    "top k in the first k)");
		sb_printf(&what, " of %s over the %d lanes", shared_str(v, 80),
		    W);
		if (c >= *cost) {
			note_dear(g, sb_str(&what), c, *cost);
			free(what.p);
			continue;
		}
		for (m = 0; (1 << m) < W; m++)
			;
		sb_printf(&g->why, "%s, by Batcher's bitonic network: %d steps "
		    "of a shuffle by lane xor j, a min and a max%s\n",
		    sb_str(&what), m * (m + 1) / 2, k == -1 ? "" : ", then the "
		    "lane it needs read by a shuffle");
		sb_printf(&g->why, "the network sorts every input: by the 0-1 "
		    "principle, as each stage's merge sorts every block 0^a 1^b "
		    "0^c its halves make (%d blocks run)\n", cases);
		*cost = c;
		free(what.p);
		return r;
	}
	return cur;
}

/*
 * A segmented scan as sg_segscan writes it, sg_scan(v, l) -
 * sg_shuffle(sg_scan(v) - v, start) with start the last head at or below
 * l: the sum of v over the lanes start..l.  When v is 0 or 1 in every
 * lane that is the number of those lanes where it is not 0, the set bits
 * of the ballot from bit start to bit l, with no scan and no shuffle.
 */
static int
segscan_parts(nid x, nid *v, nid *l, nid *start)
{
	nid a, sh, q, qs, qv;

	if (N.op[x] != O_SUB)
		return 0;
	a = N.a[x];
	sh = N.b[x];
	if (N.op[a] != O_SGSCAN || N.op[sh] != O_SGSHUF)
		return 0;
	q = N.a[sh];
	if (N.op[q] != O_SUB)
		return 0;
	qs = N.a[q];
	qv = N.b[q];
	if (N.op[qv] == O_CAST)
		qv = N.a[qv];
	if (N.op[qs] != O_SGSCAN || N.a[qs] != N.a[a] || qv != N.a[a] ||
	    !sg_is_lane(N.b[qs]) || sg_width(qs) != sg_width(a))
		return 0;
	*v = N.a[a];
	*l = N.b[a];
	*start = N.b[sh];
	return 1;
}

static nid
segscans(struct sgf *g, nid cur, float *cost)
{
	int round;

	for (round = 0; round < 8; round++) {
		nid st[MAXSGN], *stk, best = 0, bx = 0, x;
		uint8_t *seen;
		uint32_t sp = 0, j;
		int n = 0, i;
		float bc = *cost;

		seen = xcalloc(N.n, 1);
		stk = xmalloc(N.n * sizeof *stk);
		stk[sp++] = cur;
		seen[cur] = 1;
		while (sp && n < MAXSGN) {
			nid e = stk[--sp];
			if (!(N.fl[e] & NF_SG) || N.op[e] == O_CONST ||
			    N.op[e] == O_VAR || N.op[e] == O_UNDEF)
				continue;
			if (N.op[e] == O_SUB && N.op[N.a[e]] == O_SGSCAN)
				st[n++] = e;
			for (j = 0; j < (N.op[e] == O_TABLE ? N.b[e] : 3); j++) {
				nid c = N.op[e] == O_TABLE ? N.lst[N.a[e] + j] : j == 0 ?
				    N.a[e] : j == 1 ? N.b[e] : N.c[e];
				if (c && c < N.n && !seen[c]) {
					seen[c] = 1;
					stk[sp++] = c;
				}
			}
		}
		free(seen);
		free(stk);
		for (i = 0; i < n; i++) {
			nid v, l, start, to, cand;
			int W;
			int64_t lo, hi;
			ty_t bt;
			float c;
			x = st[i];
			if (!segscan_parts(x, &v, &l, &start))
				continue;
			W = sg_width(N.a[x]);
			if (!sg_range_of(g->f, v, W, &lo, &hi) || lo < 0 || hi > 1)
				continue;
			bt = TY(K_UINT, W > 32 ? 64 : 32);
			to = mk_bin(O_AND, mk_bin(O_AND, ballot_of(v, W),
			    mk_bin(O_SUB, mk_bin(O_SHL, mk_const_i(bt, 2),
			    mk_cast(bt, l)), mk_const_i(bt, 1))), mk_bin(O_SHL,
			    mk_const_i(bt, -1), mk_cast(bt, start)));
			to = mk_cast(N.ty[x], mk_un(O_POPCNT, to));
			cand = sg_replaced(cur, x, to);
			if (!cand)
				continue;
			c = price(g, cand);
			if (c < bc) {
				bc = c;
				best = cand;
				bx = x;
			} else if (c >= *cost && isfinite(c))
				note_dear(g, "a segmented scan of a value that is 0 or 1 "
				    "in every lane as the ballot's bits from the last "
				    "head to the lane", c, *cost);
		}
		if (!best)
			break;
		sb_printf(&g->why, "%s: a segmented scan of %s, which is 0 or 1 in "
		    "every lane (its range, by intervals): its sum over the lanes "
		    "from the last head at or below the lane to the lane is the "
		    "number of those lanes where it is not 0, the set bits of the "
		    "ballot from that head's bit to the lane's\n",
		    node_str(bx), node_str(N.a[N.a[bx]]));
		cur = best;
		*cost = bc;
	}
	return cur;
}

/*
 * Sums and scans, one at a time, the best first: bits through the
 * ballot, a scan as a ladder.
 */
static nid
sums_and_scans(struct sgf *g, nid cur, float *cost)
{
	int iter;

	for (iter = 0; iter < 16; iter++) {
		nid nd[MAXSGN], best = 0, bn = 0;
		int n = gather_all(cur, 0, nd), i, o, bo = 0;
		float bc = *cost;

		for (i = 0; i < n; i++) {
			nid x = nd[i], v = N.a[x], to, cand;
			int scan = N.op[x] == O_SGSCAN, W = sg_width(x);
			int64_t lo, hi, llo = 0, lhi = 0;
			int bits = sg_range_of(g->f, v, W, &lo, &hi) && lo >= 0 &&
			    hi <= 1;
			int lok = !scan || (sg_range_of(g->f, N.b[x], W, &llo, &lhi) &&
			    llo >= 0 && lhi <= W - 1);
			ty_t bt = TY(K_UINT, W > 32 ? 64 : 32);
			for (o = 0; o < 2; o++) {
				float c;
				if (o == 0 && !(bits && lok))
					continue;
				if (o == 1 && !(scan && lok && ladder_proved(W)))
					continue;
				if (o == 0) {
					to = ballot_of(v, W);
					if (scan)
						to = mk_bin(O_AND, to, mk_bin(O_SUB,
						    mk_bin(O_SHL, mk_const_i(bt, 2),
						    mk_cast(bt, N.b[x])),
						    mk_const_i(bt, 1)));
					to = mk_cast(N.ty[x], mk_un(O_POPCNT, to));
				} else
					to = ladder(v, N.b[x], W);
				cand = sg_replaced(cur, x, to);
				if (!cand)
					continue;
				c = price(g, cand);
				if (c < bc) {
					bc = c;
					best = cand;
					bn = x;
					bo = o;
				} else if (c >= *cost && isfinite(c))
					note_dear(g, o == 0 ? "the bits of a ballot "
					    "counted for a sum or a scan of a value "
					    "that is 0 or 1 in every lane" :
					    "a scan as a ladder of shuffles and adds",
					    c, *cost);
			}
		}
		if (!best)
			break;
		if (bo == 0 && N.op[bn] == O_SGSCAN)
			sb_printf(&g->why, "%s: %s is 0 or 1 in every lane (its "
			    "range, by intervals), so its sum over lanes 0..lane "
			    "is the number of those lanes where it is not 0: the "
			    "set bits of the ballot at positions 0..lane, which "
			    "(2 << lane) - 1 keeps\n", node_str(bn),
			    node_str(N.a[bn]));
		else if (bo == 0)
			sb_printf(&g->why, "%s: %s is 0 or 1 in every lane (its "
			    "range, by intervals), so its sum over the %d lanes "
			    "is the number of lanes where it is not 0: the set "
			    "bits of the ballot\n", node_str(bn),
			    node_str(N.a[bn]), sg_width(bn));
		else
			sb_printf(&g->why, "%s: a ladder of %d shuffles and adds: "
			    "after the step for d a lane holds the sum of the 2d "
			    "lanes ending at it; proved for any value per lane "
			    "(with a free word in each lane, every lane's result "
			    "is the words of lanes 0..lane, each once: + "
			    "associates and commutes mod 2^N)\n", node_str(bn),
			    sg_width(bn) <= 2 ? 1 : sg_width(bn) <= 4 ? 2 :
			    sg_width(bn) <= 8 ? 3 : sg_width(bn) <= 16 ? 4 :
			    sg_width(bn) <= 32 ? 5 : 6);
		cur = best;
		*cost = bc;
	}
	return cur;
}

/*
 * cand reads `read`, a constant table read at the lane's index, whose
 * entries at 0..n-1 are the ones anything uses (a lane past n is never
 * read, or its bit is cut away): cand with the read as a closed form of
 * the lane where that prices lower, since the lane's index costs no load.
 * The forms: c + k lane when the entries are that (each checked), and
 * find's maps of one small-range value (idrw.c: a table in one
 * constant, a multiply and a shift), each derived and read back at every
 * lane 0..n-1.  A map is evaluated in every lane of the subgroup, so one
 * that is not defined in all of them (a shift past the width beyond the
 * table) reads lane & (2^b - 1) instead, 2^b >= n, which is the lane
 * itself at 0..n-1.  *cost is cand's price, lowered when a form is
 * taken; how says which.
 */
static nid
lane_forms(struct sgf *g, nid cand, nid read, int n, int W, float *cost,
    struct sbuf *how, nid *formp)
{
	struct func f;
	struct irw *r;
	nid best = cand, tab = N.a[read], map[SG_LANE_SLOT + 1], e[SG_MAXW];
	int nr, i, l;

	*formp = 0;
	if (n < 2 || n > W || N.op[tab] != O_TABLE || (int)N.b[tab] < n)
		return cand;
	for (i = 0; i < n; i++)
		e[i] = N.lst[N.a[tab] + i];
	memset(&f, 0, sizeof f);
	f.anon = 1;
	f.np = 1;
	f.p[0].name = sym_cstr("lane");
	f.p[0].ty = TY(K_UINT, 32);
	f.p[0].ranged = 1;
	f.p[0].lo = 0;
	f.p[0].hi = n - 1;
	f.body = mk_index(mk_table(e, (uint32_t)n), mk_var(0, f.p[0].name,
	    f.p[0].ty));
	/* c + k lane: every entry checked */
	if (is_const(e[0]) && is_const(e[1])) {
		int64_t c0 = N.ki[e[0]], k = N.ki[e[1]] - c0;
		for (i = 2; i < n && is_const(e[i]) && N.ki[e[i]] == c0 + k * i; i++)
			;
		if (i == n && k != 0) {
			ty_t t = N.ty[read];
			nid L = mk_cast(t, sg_lane()), form, c2;
			float c;
			form = k == 1 ? L : mk_bin(O_MUL, L, mk_const_i(t, k));
			if (c0)
				form = mk_bin(O_ADD, form, mk_const_i(t, c0));
			c2 = sg_replaced(cand, read, form);
			if (c2 && (c = price(g, c2)) < *cost) {
				*cost = c;
				best = c2;
				*formp = form;
				sb_clear(how);
				sb_printf(how, "entry i is ");
				if (c0)
					sb_printf(how, "%lld + ", (long long)c0);
				sb_printf(how, "%lld i at i = 0..%d, each checked",
				    (long long)k, n - 1);
			}
		}
	}
	nr = irw_cands(&f, f.body, &r);
	memset(map, 0, sizeof map);
	for (i = 0; i < nr; i++) {
		nid form, c2, at, L = sg_lane();
		float c;
		int ok = 1, masked = 0;
		if (r[i].nearmiss || !r[i].derived)
			continue;
		for (l = 0; l < W && ok; l++) {
			map[0] = mk_const_i(TY(K_UINT, 32), l);
			at = subst(r[i].e, map, 1);
			ok = N.op[at] != O_UNDEF && !(N.fl[at] & NF_PARTIAL);
		}
		if (!ok) {
			/* every lane reads a lane of the table's own range */
			int64_t m = 1;
			while (m < n)
				m <<= 1;
			for (l = 0, ok = 1; l < W && ok; l++) {
				map[0] = mk_const_i(TY(K_UINT, 32), l & (m - 1));
				at = subst(r[i].e, map, 1);
				ok = N.op[at] != O_UNDEF && !(N.fl[at] & NF_PARTIAL);
			}
			if (!ok)
				continue;
			L = mk_bin(O_AND, L, mk_const_i(TY(K_UINT, 32), m - 1));
			masked = 1;
		}
		map[0] = L;
		form = mk_cast(N.ty[read], subst(r[i].e, map, 1));
		c2 = sg_replaced(cand, read, form);
		if (!c2)
			continue;
		c = price(g, c2);
		if (c < *cost) {
			*cost = c;
			best = c2;
			*formp = form;
			sb_clear(how);
			sb_printf(how, "%s, holds where %s%s", r[i].what, r[i].cond,
			    masked ? "; read at lane & mask, the lane itself there and "
			    "inside the table's range past it" : "");
		}
	}
	irw_free(r, nr);
	return best;
}

/*
 * A constant table of at most W entries as a register of the subgroup:
 * lane i holds entry i (the table read once in each lane, by the lane's
 * index), and a lookup is a shuffle.  Every lookup of the table, or none.
 */
static nid
tables(struct sgf *g, nid cur, float *cost)
{
	int W = (int)g->m->par[MP_WARP], round, ntried = 0;
	nid tried[8];

	if (W < 2 || W > SG_MAXW || !cclass_available(g->m, C_SG_SHUF))
		return cur;
	for (round = 0; round < 8; round++) {
		nid nd[MAXSGN], tab = 0, reg, cand = cur, e[SG_MAXW];
		int n = gather_all(cur, 1, nd), i, k, done = 0;
		float c;

		for (i = 0; i < n && !tab; i++) {
			nid t = N.a[nd[i]];
			if ((int)N.b[t] > W || sg_is_lane(N.b[nd[i]]) ||
			    !TY_ISINT(N.ty[t]))
				continue;
			for (k = 0; k < ntried && tried[k] != t; k++)
				;
			if (k == ntried)
				tab = t;
		}
		if (!tab)
			break;
		tried[ntried++] = tab;
		for (k = 0; k < W; k++)
			e[k] = k < (int)N.b[tab] ? N.lst[N.a[tab] + k] :
			    mk_const_i(N.ty[tab], 0);
		reg = mk_index(mk_table(e, (uint32_t)W), sg_lane());
		for (k = 0; k < MAXSGN && cand; k++) {
			nid x = 0;
			n = gather_all(cand, 1, nd);
			for (i = 0; i < n && !x; i++)
				if (N.a[nd[i]] == tab && !sg_is_lane(N.b[nd[i]]))
					x = nd[i];
			if (!x)
				break;
			cand = sg_replaced(cand, x, mk_sg(O_SGSHUF, reg, N.b[x],
			    mk_const_i(TY_LIT, W)));
			done++;
		}
		if (!cand || !done)
			continue;
		c = price(g, cand);
		{
			struct sbuf how = {0};
			float cl = c;
			nid form, c2 = lane_forms(g, cand, reg, (int)N.b[tab], W, &cl,
			    &how, &form);
			int closed = 0;
			if (form) {
				/* a form a lane computes for itself, each lookup can
				 * compute at its own index without the shuffle: the
				 * register is worth it only below that */
				nid dir = cur, map[SG_LANE_SLOT + 1], y;
				float cd = INFINITY;
				memset(map, 0, sizeof map);
				for (k = 0; k < MAXSGN && dir; k++) {
					n = gather_all(dir, 1, nd);
					for (i = 0, y = 0; i < n && !y; i++)
						if (N.a[nd[i]] == tab)
							y = nd[i];
					if (!y)
						break;
					map[SG_LANE_SLOT] = mk_cast(TY(K_UINT, 32), N.b[y]);
					dir = sg_replaced(dir, y, subst(form, map,
					    SG_LANE_SLOT + 1));
				}
				if (dir)
					cd = price(g, dir);
				if (cd <= cl) {
					if (!strstr(sb_str(&g->notes), "computed at each"))
						sb_printf(&g->notes, "  subgroup forms: the table "
						    "%s is a closed form of the index (%s): "
						    "computed at each lookup's own index, without a "
						    "register or a shuffle, the function costs %g "
						    "(as written %g)\n", shared_str(tab, 80),
						    sb_str(&how), (double)cd, (double)*cost);
				} else {
					closed = 1;
					cand = c2;
					c = cl;
				}
			}
			if (!(c < *cost)) {
				free(how.p);
				continue;
			}
			if (closed)
				sb_printf(&g->why, "the table %s, read %d time%s: lane i "
				    "computes entry i from its own index, without a load "
				    "(%s), and sg_shuffle(v, i) is v in lane i, so each "
				    "lookup is one shuffle; an index outside the table "
				    "stays outside (undefined before, the form's value "
				    "there or undefined now)\n", shared_str(tab, 80), done,
				    done == 1 ? "" : "s", sb_str(&how));
			else
				sb_printf(&g->why, "the table %s, read %d time%s: lane i "
				    "holds entry i (the table read once in each lane, at "
				    "the lane's index%s), and sg_shuffle(v, i) is v in "
				    "lane i, so each lookup is one shuffle; an index "
				    "outside the table stays outside (undefined before, "
				    "%s now)\n", shared_str(tab, 80), done, done == 1 ?
				    "" : "s", (int)N.b[tab] < W ? ", the lanes past its "
				    "end holding 0" : "", (int)N.b[tab] < W ?
				    "0 or undefined" : "undefined");
			free(how.p);
		}
		cur = cand;
		*cost = c;
	}
	return cur;
}


/*
 * One function of a small-range value read at several keys (h(i) +
 * h(j)*h(k), h(x) = (x*x + 3) % 11, i, j, k in 0..7) as a constant table
 * of its values read at each key: tables() can then hold it one entry
 * per lane and read it by shuffles.  tab.c tabulates a function read
 * once, where a register costs more than the load; here the repeats pay
 * for it.  A subterm is such a read when it reads one parameter, and two
 * are the same function when putting one variable for their parameters
 * makes them the same node; the table is that function at every point of
 * the keys' range, each evaluated (a point where it is undefined, and
 * none is taken).  The largest such function by its price is tried, and
 * kept only when tables() makes the whole cheaper.
 */
#define FT_MAX 512	/* subterms looked at */

static nid
funcs_as_tables(struct sgf *g, nid cur, float *cost)
{
	const struct func *f = g->f;
	int W = (int)g->m->par[MP_WARP], nn = 0, i, j, best = -1, nocc, K;
	nid *nd, *canon, x, T, cand, e[SG_MAXW], map[MAXP];
	int *slot;
	uint8_t *seen;
	nid *st;
	uint32_t sp = 0;
	float bp = 0, c2;
	int64_t lo = INT64_MAX, hi = INT64_MIN, vals[SG_MAXW];
	size_t wn = g->why.n, nnn = g->notes.n;

	if (W < 2 || W > SG_MAXW || !cclass_available(g->m, C_SG_SHUF) ||
	    (N.fl[cur] & NF_SG))
		return cur;
	nd = xmalloc(FT_MAX * sizeof *nd);
	canon = xmalloc(FT_MAX * sizeof *canon);
	slot = xmalloc(FT_MAX * sizeof *slot);
	seen = xcalloc(N.n, 1);
	st = xmalloc(N.n * sizeof *st);
	st[sp++] = cur;
	seen[cur] = 1;
	while (sp && nn < FT_MAX) {
		nid n = st[--sp];
		uint64_t sl = node_slots(n);
		uint32_t k;
		int op = N.op[n];
		if (op == O_CONST || op == O_VAR || op == O_UNDEF || op == O_TABLE ||
		    op == O_LOOP)
			continue;
		if (TY_ISINT(N.ty[n]) && N.ty[n] != TY_BOOL && sl &&
		    !(sl & (sl - 1)) && !(N.fl[n] & NF_LANE) && op != O_CAST) {
			for (j = 0; !((sl >> j) & 1); j++)
				;
			if (j < f->np && TY_ISINT(f->p[j].ty)) {
				memset(map, 0, sizeof map);
				map[j] = mk_var(0, sym_cstr("x"), f->p[j].ty);
				nd[nn] = n;
				slot[nn] = j;
				canon[nn++] = subst(n, map, MAXP);
			}
		}
		for (k = 0; k < 3; k++) {
			nid a = k == 0 ? N.a[n] : k == 1 ? N.b[n] : N.c[n];
			if (a && a < N.n && !seen[a]) {
				seen[a] = 1;
				st[sp++] = a;
			}
		}
	}
	free(seen);
	free(st);
	/* the dearest function read at two or more different parameters */
	for (i = 0; i < nn; i++) {
		int other = 0;
		float pc;
		for (j = 0; j < nn && !other; j++)
			other = canon[j] == canon[i] && slot[j] != slot[i];
		if (!other || (pc = fuse_expr_cost(g->m, canon[i])) <= bp)
			continue;
		bp = pc;
		best = i;
	}
	if (best < 0)
		goto none;
	/* the keys' range */
	for (i = 0, nocc = 0; i < nn; i++)
		if (canon[i] == canon[best]) {
			int64_t l, h;
			param_bounds(&f->p[slot[i]], &l, &h);
			if (l < lo)
				lo = l;
			if (h > hi)
				hi = h;
			nocc++;
		}
	if (hi < lo || hi - lo + 1 > W || hi - lo < 1)
		goto none;
	K = (int)(hi - lo + 1);
	/* the function at every point */
	{
		struct func tf;
		struct plan p;
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		nid roots[1];
		int bad = 0;
		memset(&tf, 0, sizeof tf);
		tf.anon = 1;
		tf.np = 1;
		tf.p[0].name = sym_cstr("x");
		tf.p[0].ty = f->p[slot[best]].ty;
		tf.body = canon[best];
		roots[0] = canon[best];
		plan_build(&p, roots, 1, 0);
		for (i = 0; i < K; i++)
			in_i[i] = lo + i;
		plan_run(&p, (uint32_t)K, in_i, in_r);
		for (i = 0; i < K && !bad; i++) {
			if (COLF(&p, p.root[0])[i] & F_UNDEF)
				bad = 1;
			vals[i] = int_norm(N.ty[canon[best]], COLI(&p, p.root[0])[i]);
		}
		plan_free(&p);
		free(in_i);
		free(in_r);
		(void)tf;
		if (bad)
			goto none;
	}
	for (i = 0; i < K; i++)
		e[i] = mk_const_i(N.ty[canon[best]], vals[i]);
	T = mk_table(e, (uint32_t)K);
	cand = cur;
	for (i = 0; i < nn && cand; i++)
		if (canon[i] == canon[best]) {
			nid key = mk_var((uint32_t)slot[i], f->p[slot[i]].name,
			    f->p[slot[i]].ty);
			if (lo)
				key = mk_bin(O_SUB, key, mk_const_i(f->p[slot[i]].ty, lo));
			cand = sg_replaced(cand, nd[i], mk_cast(N.ty[nd[i]],
			    mk_index(T, key)));
		}
	if (!cand)
		goto none;
	c2 = *cost;
	x = tables(g, cand, &c2);
	if (x == cand || !(c2 < *cost)) {
		/* nothing of it: take back what tables() said */
		g->why.n = wn;
		if (g->why.p)
			g->why.p[wn] = 0;
		g->notes.n = nnn;
		if (g->notes.p)
			g->notes.p[nnn] = 0;
		goto none;
	}
	{
		/* this line before the register's */
		struct sbuf w = {0};
		sb_printf(&w, "%s is read at %d keys (", shared_str(canon[best],
		    80), nocc);
		{
			int sl, said = 0;
			for (sl = 0; sl < f->np; sl++)
				for (i = 0; i < nn; i++)
					if (canon[i] == canon[best] && slot[i] == sl) {
						sb_printf(&w, "%s%s", said++ ? ", " : "",
						    sym_str(f->p[sl].name));
						break;
					}
		}
		sb_printf(&w, "); at x = %lld..%lld, every point evaluated, it is "
		    "the table %s\n", (long long)lo, (long long)hi,
		    shared_str(T, 80));
		sb_puts(&w, g->why.p + wn);
		g->why.n = wn;
		if (g->why.p)
			g->why.p[wn] = 0;
		sb_puts(&g->why, sb_str(&w));
		free(w.p);
	}
	*cost = c2;
	free(nd);
	free(canon);
	free(slot);
	return x;
none:
	free(nd);
	free(canon);
	free(slot);
	return cur;
}

/*
 * A GF(2)-affine map of one word: bit i of the result is the parity of
 * x & row[i], xor bit i of k.  Derived from the structure of the
 * expression: xor, not, shifts and masks by constants, conversions, and
 * an or or a sum of fields that cannot both be 1 at any bit.
 */

static uint64_t
wmask(int w)
{
	return w >= 64 ? ~0ULL : (1ULL << w) - 1;
}

static int
int_width(ty_t t)
{
	return TYK(t) == K_UINT || TYK(t) == K_SINT ? TYW(t) : 0;
}

/* bit i of a is always 0 / the constant 1 */
#define BIT0(a, i) ((a)->row[i] == 0 && !(((a)->k >> (i)) & 1))
#define BIT1(a, i) ((a)->row[i] == 0 && (((a)->k >> (i)) & 1))

int
gf2_affine(nid e, int slot, struct aff *r, int depth)
{
	struct aff a, b;
	int w = int_width(N.ty[e]), i, op = N.op[e];

	memset(r, 0, sizeof *r);
	r->w = w;
	if (is_const(e) && TY_ISINT(N.ty[e])) {
		r->w = w ? w : 64;
		r->k = (uint64_t)N.ki[e] & wmask(r->w);
		return 1;
	}
	if (!w || depth > 200)
		return 0;
	switch (op) {
	case O_VAR:
		if (N.ki[e] != slot)
			return 0;
		for (i = 0; i < w; i++)
			r->row[i] = 1ULL << i;
		return 1;
	case O_NOT:
		if (!gf2_affine(N.a[e], slot, r, depth + 1))
			return 0;
		r->k ^= wmask(w);
		r->w = w;
		return 1;
	case O_CAST: {
		ty_t ta = N.ty[N.a[e]];
		int wa = int_width(ta), sgn = TYK(ta) == K_SINT;
		if (!gf2_affine(N.a[e], slot, &a, depth + 1))
			return 0;
		if (!wa)
			wa = a.w;
		for (i = 0; i < w; i++) {
			int j = i < wa ? i : sgn ? wa - 1 : -1;
			if (j < 0)
				continue;
			r->row[i] = a.row[j];
			r->k |= ((a.k >> j) & 1) << i;
		}
		return 1;
	}
	case O_SHL: case O_SHR: {
		int64_t s;
		int sgn = TYK(N.ty[e]) == K_SINT;
		if (!is_const(N.b[e]) || !gf2_affine(N.a[e], slot, &a, depth + 1))
			return 0;
		s = N.ki[N.b[e]];
		if (s < 0 || s >= w)
			return 0;
		for (i = 0; i < w; i++) {
			int j = op == O_SHL ? i - (int)s : i + (int)s;
			if (op == O_SHR && j >= w && sgn)
				j = w - 1;
			if (j < 0 || j >= w)
				continue;
			r->row[i] = a.row[j];
			r->k |= ((a.k >> j) & 1) << i;
		}
		return 1;
	}
	case O_XOR: case O_AND: case O_OR: case O_ADD:
		if (!gf2_affine(N.a[e], slot, &a, depth + 1) ||
		    !gf2_affine(N.b[e], slot, &b, depth + 1))
			return 0;
		for (i = 0; i < w; i++) {
			const struct aff *keep = NULL;
			int one = 0;
			if (op == O_XOR) {
				r->row[i] = a.row[i] ^ b.row[i];
				r->k |= (((a.k ^ b.k) >> i) & 1) << i;
				continue;
			}
			if (op == O_AND) {
				if (BIT0(&a, i) || BIT0(&b, i))
					continue;
				if (BIT1(&a, i))
					keep = &b;
				else if (BIT1(&b, i))
					keep = &a;
				else
					return 0;
			} else {
				/* or, +: no bit where both can be 1 (so no carry) */
				if (BIT0(&a, i))
					keep = &b;
				else if (BIT0(&b, i))
					keep = &a;
				else if (op == O_OR && (BIT1(&a, i) || BIT1(&b, i)))
					one = 1;
				else
					return 0;
			}
			if (one)
				r->k |= 1ULL << i;
			else {
				r->row[i] = keep->row[i];
				r->k |= ((keep->k >> i) & 1) << i;
			}
		}
		return 1;
	}
	return 0;
}

static int
popc64(uint64_t v)
{
	int n = 0;

	for (; v; v &= v - 1)
		n++;
	return n;
}

static int
ctz64(uint64_t v)
{
	int n = 0;

	while (!(v & 1)) {
		v >>= 1;
		n++;
	}
	return n;
}

/*
 * The bit a lane works out: x's bit s[lane] when every row is one bit
 * (the shift amount lane + b, b - lane or lane xor b when the rows follow
 * one of those, else a table), the parity of x & row[lane] otherwise.
 * kind: 0 parity, 1 a table of shifts, 2.. a closed shift.
 */
static nid
lane_bit(const struct aff *m, nid x, int W, int kind, nid *read)
{
	int n = int_width(N.ty[x]), i;
	nid L = sg_lane(), e[SG_MAXW], s = 0, one = mk_const_i(N.ty[x], 1);
	int64_t sh[SG_MAXW];

	*read = 0;
	if (kind == 0) {
		for (i = 0; i < W; i++)
			e[i] = mk_const_i(N.ty[x], (int64_t)(i < m->w ? m->row[i] :
			    0));
		*read = mk_index(mk_table(e, (uint32_t)W), L);
		return mk_bin(O_NE, mk_bin(O_AND, mk_un(O_POPCNT, mk_bin(O_AND, x,
		    *read)), mk_const_i(TY_LIT, 1)), mk_const_i(TY_LIT, 0));
	}
	if (m->w != W)
		return 0;
	for (i = 0; i < W; i++) {
		if (popc64(m->row[i]) != 1)
			return 0;
		sh[i] = ctz64(m->row[i]);
	}
	switch (kind) {
	case 1:
		for (i = 0; i < W; i++)
			e[i] = mk_const_i(TY(K_UINT, 32), sh[i]);
		s = *read = mk_index(mk_table(e, (uint32_t)W), L);
		break;
	case 2:		/* (lane + b) mod n, n a power of two */
	case 3:		/* (b - lane) mod n */
		if (n & (n - 1))
			return 0;
		for (i = 0; i < W; i++)
			if (sh[i] != ((kind == 2 ? sh[0] + i : sh[0] - i) & (n - 1)))
				return 0;
		s = kind == 2 ? mk_bin(O_ADD, L, mk_const_i(TY(K_UINT, 32), sh[0])) :
		    mk_bin(O_SUB, mk_const_i(TY(K_UINT, 32), sh[0]), L);
		if (kind == 2 ? sh[0] + W - 1 >= n : sh[0] - (W - 1) < 0)
			s = mk_bin(O_AND, s, mk_const_i(TY(K_UINT, 32), n - 1));
		if (kind == 2 && sh[0] == 0)
			s = L;
		break;
	case 4:		/* lane xor b */
		for (i = 0; i < W; i++)
			if (sh[i] != (i ^ sh[0]))
				return 0;
		s = mk_bin(O_XOR, L, mk_const_i(TY(K_UINT, 32), sh[0]));
		break;
	default:
		return 0;
	}
	return mk_bin(O_NE, mk_bin(O_AND, mk_bin(O_SHR, x, s), one),
	    mk_const_i(TY_LIT, 0));
}

static nid
bitslice(struct sgf *g, nid cur, float *cost)
{
	static const char *const how[] = {"that parity, the rows in a table "
	    "read at the lane's index", "every row has one bit: that bit of "
	    "the word, its position in a table read at the lane's index",
	    "every row has one bit: that bit of the word, its position the "
	    "lane's index plus a constant", "every row has one bit: that bit "
	    "of the word, its position a constant less the lane's index",
	    "every row has one bit: that bit of the word, its position the "
	    "lane's index xor a constant"};
	uint64_t kk;
	const struct func *f = g->f;
	int W = (int)g->m->par[MP_WARP], slot = -1, j, kind, bk = -1, closed = 0;
	struct sbuf how2 = {0};
	uint64_t slots = node_slots(cur);
	struct aff *m;
	nid x, best = 0;
	float bc = *cost;
	ty_t t = N.ty[cur], bt = TY(K_UINT, W > 32 ? 64 : 32);

	if (W < 2 || W > SG_MAXW || !cclass_available(g->m, C_SG_BALLOT) ||
	    (N.fl[cur] & NF_SG) ||
	    !int_width(t) || int_width(t) > W)
		return cur;
	for (j = 0; j < f->np; j++)
		if ((slots >> j) & 1) {
			if (slot >= 0)
				return cur;
			slot = j;
		}
	if (slot < 0 || (slots >> MAXP) || !f->p[slot].uniform ||
	    !int_width(f->p[slot].ty))
		return cur;
	m = xmalloc(sizeof *m);
	if (!gf2_affine(cur, slot, m, 0)) {
		free(m);
		return cur;
	}
	x = mk_var((uint32_t)slot, f->p[slot].name, f->p[slot].ty);
	for (kind = 0; kind < 5; kind++) {
		nid rd, b = lane_bit(m, x, W, kind, &rd), cand;
		struct sbuf h = {0};
		int took = 0;
		float c;
		if (!b)
			continue;
		cand = mk_sg(O_SGBALLOT, b, mk_const_i(TY_LIT, W), 0);
		if (m->k)
			cand = mk_bin(O_XOR, cand, mk_const_i(bt, (int64_t)m->k));
		cand = mk_cast(t, cand);
		c = price(g, cand);
		if (rd) {
			/* the rows (or shifts) a lane reads, as a closed form of
			 * the lane; a row past the result's width is cut away */
			nid fm, c2 = lane_forms(g, cand, rd, kind == 0 ? m->w : W, W,
			    &c, &h, &fm);
			took = c2 != cand;
			cand = c2;
		}
		if (c < bc) {
			bc = c;
			best = cand;
			bk = kind;
			closed = took;
			sb_clear(&how2);
			if (took)
				sb_puts(&how2, sb_str(&h));
		}
		free(h.p);
	}
	kk = m->k;
	free(m);
	if (!best) {
		free(how2.p);
		return cur;
	}
	sb_printf(&g->why, "every operation of the function is linear over "
	    "GF(2) in the bits of %s (xor, not, shifts and masks by constants, "
	    "conversions, an or or a sum of fields that share no bit), so bit i "
	    "of the result is the parity of %s & row[i]%s; lane i works that "
	    "bit out (%s%s%s) and the ballot puts it at bit i\n",
	    sym_str(f->p[slot].name), sym_str(f->p[slot].name), kk ?
	    ", xor bit i of a constant" : "", how[bk], closed ? "; the table "
	    "read without a load, from the lane's index: " : "", closed ?
	    sb_str(&how2) : "");
	free(how2.p);
	*cost = bc;
	return best;
}

/* the size of the subgroup e's operations run in */
static int
width_of(nid e, uint8_t *seen)
{
	uint32_t j;
	int w;

	if (e == 0 || !(N.fl[e] & NF_SG) || seen[e])
		return 0;
	seen[e] = 1;
	if (IS_SG(N.op[e]))
		return sg_width(e);
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((w = width_of(N.lst[N.a[e] + j], seen)) != 0)
				return w;
		return 0;
	}
	if ((w = width_of(N.a[e], seen)) != 0 ||
	    (w = width_of(N.b[e], seen)) != 0)
		return w;
	return width_of(N.c[e], seen);
}

static int
width_in(nid e)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int w = width_of(e, seen);

	free(seen);
	return w;
}

/* what the shader needs for the operations of e */
static void
needs(struct sbuf *b, nid e)
{
	static const struct {
		int op;
		const char *what;
	} ext[] = {
		{O_SGBALLOT, "ballot (GL_KHR_shader_subgroup_ballot)"},
		{O_SGANY, "vote (GL_KHR_shader_subgroup_vote)"},
		{O_SGADD, "arithmetic (GL_KHR_shader_subgroup_arithmetic)"},
		{O_SGSHUF, "shuffle (GL_KHR_shader_subgroup_shuffle)"},
	};
	uint8_t *seen = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), x;
	uint32_t sp = 0, j;
	int has[O_NOPS] = {0}, i, n = 0;

	st[sp++] = e;
	seen[e] = 1;
	while (sp) {
		x = st[--sp];
		has[N.op[x]] = 1;
		for (j = 0; j < (N.op[x] == O_TABLE ? N.b[x] : 3); j++) {
			nid a = N.op[x] == O_TABLE ? N.lst[N.a[x] + j] : j == 0 ?
			    N.a[x] : j == 1 ? N.b[x] : N.c[x];
			if (N.op[x] == O_CONST || N.op[x] == O_VAR ||
			    N.op[x] == O_UNDEF)
				break;
			if (a && a < N.n && !seen[a] && (N.fl[a] & NF_SG)) {
				seen[a] = 1;
				st[sp++] = a;
			}
		}
	}
	has[O_SGANY] |= has[O_SGALL];
	has[O_SGADD] |= has[O_SGSCAN];
	for (i = 0; i < 4; i++)
		if (has[ext[i].op])
			sb_printf(b, "%s%s", n++ ? ", " : "", ext[i].what);
	free(seen);
	free(st);
}

void
sg_find(const struct func *f)
{
	struct sgf g;
	struct eqproof pr;
	struct sbuf w = {0}, nd = {0};
	const char *mn, *s, *e, *chk;
	nid cur = f->body, am;
	float before, cost;
	int W, bad;

	memset(&g, 0, sizeof g);
	g.f = f;
	g.m = model_active();
	mn = sym_str(g.m->name);
	if (!TY_ISINT(N.ty[cur]))
		return;
	before = cost = price(&g, cur);
	if (N.fl[cur] & NF_SG)
		cur = sorts(&g, cur, &cost);
	if (cur == f->body && (N.fl[cur] & NF_SG))
		cur = segscans(&g, cur, &cost);
	if (N.fl[cur] & NF_SG)
		cur = sums_and_scans(&g, cur, &cost);
	cur = tables(&g, cur, &cost);
	if (cur == f->body)
		cur = funcs_as_tables(&g, cur, &cost);
	if (cur == f->body)
		cur = bitslice(&g, cur, &cost);
	if (g.notes.n)
		fputs(sb_str(&g.notes), stdout);
	if (cur == f->body)
		goto out;
	W = width_in(cur);
	bad = equiv_decide(f, f->body, cur, &w, &pr);
	chk = bad ? "contradicted by the check" : pr.how == EQ_EXHAUSTIVE ?
	    "checked on every point of the domain" : pr.how == EQ_SAT ?
	    "checked by SAT" : "checked on a sample of the domain";
	needs(&nd, cur);
	if (bad) {
		printf("  subgroup forms under %s: %s (cost %g) does not hold: "
		    "%s\n", mn, shared_str(cur, 300), (double)cost, sb_str(&w));
		rep_obj("subgroup_forms");
		rep_str("model", mn);
		rep_num("cost_before", before);
		rep_expr("candidate", cur);
		rep_str("not_proved", sb_str(&w));
		rep_end();
		goto out;
	}
	printf("  subgroup forms under %s (each derived, with what it holds "
	    "under; cost %g as written):\n", mn, (double)before);
	printf("  cost %-3g %s\n", (double)cost, shared_str(cur, 400));
	for (s = sb_str(&g.why); *s; s = *e ? e + 1 : e) {
		e = strchr(s, '\n');
		if (!e)
			e = s + strlen(s);
		printf("           %.*s\n", (int)(e - s), s);
	}
	{
		uint8_t *seen = xcalloc(N.n, 1);
		am = active_mask_in(f->body, seen);
		free(seen);
	}
	if (am)
		printf("           holds with the lanes active in %s (sg_active), "
		    "checked for every value of it; needs %s\n", node_str(am),
		    sb_str(&nd));
	else
		printf("           holds in a full subgroup (all %d lanes "
		    "active); needs %s\n", W, sb_str(&nd));
	printf("           %s\n", chk);
	rep_obj("subgroup_forms");
	rep_str("model", mn);
	rep_int("lanes", W);
	if (am)
		rep_str("active", node_str(am));
	rep_num("cost_before", before);
	rep_num("cost", cost);
	rep_expr("expr", cur);
	rep_str("derivation", sb_str(&g.why));
	rep_str("needs", sb_str(&nd));
	rep_str("check", pr.how == EQ_EXHAUSTIVE ? "exhaustive" : pr.how ==
	    EQ_SAT ? "sat" : "sampled");
	rep_end();
out:
	free(g.why.p);
	free(g.notes.p);
	free(w.p);
	free(nd.p);
}
