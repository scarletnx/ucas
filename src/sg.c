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
 *   sg_dfdx(v), sg_dfdy(v), sg_fwidth(v)
 *                      of a float: the lanes in fours are quads, lane
 *                      bit 0 the column and bit 1 the row (Vulkan's
 *                      layout); in lane l, v in lane l | 1 less v in
 *                      lane l & ~1, the same with bit 1, and |dfdx| +
 *                      |dfdy|: dFdxFine, dFdyFine, fwidthFine
 *                      (NV_compute_shader_derivatives in compute)
 *
 * and, written with them: sg_exclusive(v) = sg_scan(v) - v,
 * sg_shuffle_xor(v, k) = sg_shuffle(v, lane xor k), sg_broadcast(v, i) =
 * sg_shuffle(v, i), sg_broadcast_first(v) = sg_shuffle(v, 0),
 * sg_rotate(v, d) = sg_shuffle(v, (lane + d) mod W), the quad's swaps
 * sg_quad_swap_h(v), _v(v), _d(v) = sg_shuffle(v, lane xor 1, 2, 3) and
 * sg_quad_broadcast(v, i) = sg_shuffle(v, (lane & ~3) | i) (i in 0..3),
 * and sg_sort(v), in lane l the l-th smallest of v over the lanes: the
 * bitonic network (sg_sorted).
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
	case O_SGDDX: case O_SGDDY: case O_SGFWIDTH:
		return C_SG_DERIV;
	}
	return C_SG_SCAN;
}

int
sg_width(nid n)
{
	nid w = opinfo[N.op[n]].arity == 3 ? N.c[n] : N.b[n];

	return (int)N.ki[w];
}

/* a `lanes` parameter's copy in e (sg_lanes_copy) or one of its words,
 * the first found; with map, every copy's word in lane l put into it */
static nid
lanes_in(nid e, uint8_t *seen, nid *map, int l)
{
	nid r = 0, x;
	uint32_t j;

	if (e == 0 || !(N.fl[e] & NF_LANES) || seen[e])
		return 0;
	seen[e] = 1;
	if (N.op[e] == O_VAR) {
		if (map && N.ki[e] >= SG_LCOPY_SLOT && N.ki[e] < SG_LCOPY_SLOT +
		    MAXP)
			map[N.ki[e]] = sg_lanes_word((int)N.ki[e] - SG_LCOPY_SLOT,
			    l, N.a[e], N.ty[e]);
		return e;
	}
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((x = lanes_in(N.lst[N.a[e] + j], seen, map, l)) != 0)
				r = x;
		return r;
	}
	if (N.op[e] == O_LOOP) {
		/* what the loop's body reads */
		struct loopdef d;
		int k;
		loop_get(loop_ent(e), &d);
		for (k = 0; k < d.nst; k++) {
			if ((x = lanes_in(d.init[k], seen, map, l)) != 0)
				r = x;
			if ((x = lanes_in(d.next[k], seen, map, l)) != 0)
				r = x;
		}
		if (d.cond && (x = lanes_in(d.cond, seen, map, l)) != 0)
			r = x;
	}
	if ((x = lanes_in(N.a[e], seen, map, l)) != 0)
		r = x;
	if ((x = lanes_in(N.b[e], seen, map, l)) != 0)
		r = x;
	if ((x = lanes_in(N.c[e], seen, map, l)) != 0)
		r = x;
	return r;
}

const char *
sg_lanes_name(nid e)
{
	uint8_t *seen = xcalloc(N.n, 1);
	nid v = lanes_in(e, seen, NULL, 0);

	free(seen);
	return v ? sym_str(N.a[v]) : NULL;
}

/* e in lane l: the lane's index, and a `lanes` parameter's word there */
static nid
in_lane(nid e, int l)
{
	nid map[SG_LCOPY_SLOT + MAXP];

	if (!(N.fl[e] & (NF_LANE | NF_LANES)))
		return e;
	memset(map, 0, sizeof map);
	map[SG_LANE_SLOT] = mk_const_i(TY(K_UINT, 32), l);
	if (N.fl[e] & NF_LANES) {
		uint8_t *seen = xcalloc(N.n, 1);
		lanes_in(e, seen, map, l);
		free(seen);
	}
	return subst(e, map, SG_LCOPY_SLOT + MAXP);
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
	case O_SGDDX: case O_SGDDY: case O_SGFWIDTH: {
		ty_t u = TY(K_UINT, 32);
		nid tab, i = mk_cast(u, b), d[2];
		int k;
		for (l = 0; l < W; l++)
			e[l] = in_lane(a, l);
		tab = mk_table(e, (uint32_t)W);
		/* the pair's other lane less its first: bit 0, then bit 1 */
		for (k = 0; k < 2; k++)
			d[k] = mk_bin(O_SUB, mk_index(tab, mk_bin(O_OR, i,
			    mk_const_i(u, 1 << k))), mk_index(tab, mk_bin(O_AND, i,
			    mk_const_i(u, ~(1u << k)))));
		r = op == O_SGDDX ? d[0] : op == O_SGDDY ? d[1] :
		    mk_bin(O_ADD, mk_un(O_ABS, d[0]), mk_un(O_ABS, d[1]));
		break;
	}
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
	case O_SGDDX: case O_SGDDY: case O_SGFWIDTH:
		if (TYK(N.ty[a]) != K_FLOAT)
			fail("%s takes a float (f16, f32, f64) per lane, not %s",
			    nm, ty_str(N.ty[a]));
		if (W % 4)
			fail("%s reads quads of 4 lanes: a subgroup of %d lanes "
			    "has none", nm, W);
		break;
	}
	t = sg_type(op, a, W);
	if (ar == 3 && !TY_ISINT(N.ty[b]))
		fail("%s: a lane's index is an integer", nm);
	if (N.op[a] == O_UNDEF || (ar == 3 && N.op[b] == O_UNDEF))
		return mk_undef(t);
	/* nothing but the lane in it: the same value whatever the inputs */
	if (!node_slots(a) && !(N.fl[a] & (NF_LANEREAD | NF_LANES)) && (ar == 2 ||
	    (!node_slots(b) && !(N.fl[b] & (NF_LANE | NF_LANEREAD | NF_LANES)))))
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
 *   a GF(2)-linear function of the words the lanes share: each lane
 *     works out one bit of the result and the ballot puts the bits
 *     together;
 *   a map applied field by field to a word the lanes share: each lane
 *     works out one bit of its field's value, the ballot puts them
 *     together;
 *   an order statistic of a value per lane, however written: the
 *     bitonic network, read where it is needed.
 */
struct sgf {
	const struct func *f;
	const struct model *m;
	nid lanep;		/* f's lane parameter as a u32, or 0 */
	struct sbuf why;
	struct sbuf notes;	/* derived, but not cheaper */
	nid proved;		/* a form already proved equal to f, in pr */
	struct eqproof pr;
	int bycons;		/* proved by its derivation, the steps in why */
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

int
sg_lanes_width(const struct func *f)
{
	uint32_t nm = sym_cstr("lane");
	int64_t lo, hi;
	int j, W;

	for (j = 0; j < f->np; j++)
		if (f->p[j].name == nm && TY_ISINT(f->p[j].ty)) {
			param_bounds(&f->p[j], &lo, &hi);
			if (lo == 0 && hi >= 1 && hi < SG_MAXW)
				return (int)hi + 1;
		}
	W = (int)model_active()->par[MP_WARP];
	return W >= 2 && W <= SG_MAXW ? W : 0;
}

/*
 * e for SAT: a `lanes` parameter read in the lane itself is the word of
 * f's lane, one of the W free words the subgroup operations read (their
 * operands' copies become those words where sg_lower spells them out).
 */
nid
sg_lanes_lower(const struct func *f, nid e)
{
	nid map[MAXP] = {0}, ent[SG_MAXW], lp = lane_param(f);
	int W = sg_lanes_width(f), j, l, any = 0;
	uint64_t slots = node_slots(e);

	for (j = 0; j < f->np; j++) {
		if (!f->p[j].lanes || !((slots >> j) & 1))
			continue;
		if (!lp || !W)
			fail("%s has its own value in each lane (lanes) and is read "
			    "in the lane itself: the function needs its lane, a "
			    "parameter lane: u32 in 0..N-1", sym_str(f->p[j].name));
		for (l = 0; l < W; l++)
			ent[l] = sg_lanes_word(j, l, f->p[j].name, f->p[j].ty);
		map[j] = mk_index(mk_table(ent, (uint32_t)W), lp);
		any = 1;
	}
	return any ? subst(e, map, f->np) : e;
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
 * A GF(2)-affine map of the words the lanes share: bit i of the result
 * is the parity of the words' bits that row[i] names, xor bit i of k
 * (the words side by side in a row, each at its offset).  Derived from
 * the structure of the expression: xor, not, shifts and masks by
 * constants, conversions, an or or a sum of fields that cannot both be 1
 * at any bit; a bit negated or multiplied by a constant; a ?: on one bit
 * whose sides differ by a constant; a constant table that is affine in
 * its index; and a for loop of a constant count, run through with its
 * variables as maps.
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

/* bit i of a is always 0 / the constant 1 / not affine */
#define BAD(a, i) ((int)(((a)->bad >> (i)) & 1))
#define BIT0(a, i) ((a)->row[i] == 0 && !(((a)->k >> (i)) & 1) && !BAD(a, i))
#define BIT1(a, i) ((a)->row[i] == 0 && (((a)->k >> (i)) & 1) && !BAD(a, i))

enum { AF_BIT = 1, AF_SEL = 2, AF_TAB = 4, AF_LOOP = 8 };

struct affcx {
	int off[MAXP], pw[MAXP];	/* a word's bits in a row; pw 0: not one */
	const struct aff *env[NSLOT];	/* a loop's variables as they stand */
	int budget;			/* nodes left to visit */
	int used;			/* AF_*: what the map went through */
};

static int aff_of(nid e, struct affcx *cx, struct aff *r, int depth);

/* a is a constant */
static int
aff_const(const struct aff *a)
{
	int i;

	for (i = 0; i < 64; i++)
		if (a->row[i])
			return 0;
	return !a->bad;
}

/* a is 0 above its bit 0 */
static int
aff_one_bit(const struct aff *a)
{
	int i;

	for (i = 1; i < 64; i++)
		if (!BIT0(a, i))
			return 0;
	return 1;
}

/* a as a value of type t: cut to its width (an untyped one keeps a's) */
static void
aff_fit(struct aff *a, ty_t t)
{
	int w = int_width(t), i;

	if (!w)
		return;
	for (i = w; i < 64; i++)
		a->row[i] = 0;
	a->k &= wmask(w);
	a->bad &= wmask(w);
	a->w = w;
}

/* bit i of r is bit j of a */
static void
aff_take(struct aff *r, int i, const struct aff *a, int j)
{
	r->row[i] = a->row[j];
	r->k |= ((a->k >> j) & 1) << i;
	r->bad |= (uint64_t)BAD(a, j) << i;
}

/*
 * A condition that is one affine bit: a comparison with == or != whose
 * sides differ at one bit position only.  *row and *k are that bit.
 */
static int
aff_bit(nid c, struct affcx *cx, uint64_t *row, int *k, int depth)
{
	struct aff a, b;
	int op = N.op[c], i, at = -1;

	if (depth > 200)
		return 0;
	if (op == O_LNOT) {
		if (!aff_bit(N.a[c], cx, row, k, depth + 1))
			return 0;
		*k ^= 1;
		return 1;
	}
	/* unsigned sides only: the bits compared are then the values */
	if ((op != O_NE && op != O_EQ) || TYK(N.ty[N.a[c]]) == K_SINT ||
	    TYK(N.ty[N.b[c]]) == K_SINT || !aff_of(N.a[c], cx, &a, depth + 1) ||
	    !aff_of(N.b[c], cx, &b, depth + 1))
		return 0;
	if (a.bad || b.bad)
		return 0;
	*row = 0;
	*k = 0;
	for (i = 0; i < 64; i++) {
		uint64_t rw = a.row[i] ^ b.row[i];
		int kd = (int)(((a.k ^ b.k) >> i) & 1);
		if (!rw && !kd)
			continue;
		if (at >= 0)
			return 0;
		at = i;
		*row = rw;
		*k = kd;
	}
	if (op == O_EQ)
		*k ^= 1;
	return 1;
}

/* the value of a constant map of type t */
static int64_t
aff_value(const struct aff *a, ty_t t)
{
	if (TYK(t) == K_SINT && a->w < 64 && ((a->k >> (a->w - 1)) & 1))
		return (int64_t)(a->k | ~wmask(a->w));
	return (int64_t)a->k;
}

/* a for loop of a constant count: its states' maps after every trip */
static int
aff_loop(nid e, struct affcx *cx, struct aff *r, int depth)
{
	struct loopdef d;
	struct aff *cur, *nxt, ix, a, b;
	const struct aff *saved[MAXST + 1];
	int64_t lo, hi, it;
	int k, ok = 1;

	loop_get(loop_ent(e), &d);
	if (d.kind != LK_FOR || !aff_of(N.a[e], cx, &a, depth + 1) ||
	    !aff_const(&a) || !aff_of(N.b[e], cx, &b, depth + 1) ||
	    !aff_const(&b))
		return 0;
	lo = aff_value(&a, N.ty[N.a[e]]);
	hi = aff_value(&b, N.ty[N.b[e]]);
	if ((hi >= lo && hi - lo >= 4096) || hi < lo - 1)
		return 0;
	cur = xmalloc(2 * MAXST * sizeof *cur);
	nxt = cur + MAXST;
	for (k = 0; k < d.nst && ok; k++) {
		ok = aff_of(d.init[k], cx, &cur[k], depth + 1);
		aff_fit(&cur[k], d.sty[k]);
	}
	saved[MAXST] = cx->env[d.islot];
	for (k = 0; k < d.nst; k++)
		saved[k] = cx->env[d.sslot[k]];
	for (it = lo; it <= hi && ok; it++) {
		memset(&ix, 0, sizeof ix);
		ix.w = int_width(d.ity) ? int_width(d.ity) : 64;
		ix.k = (uint64_t)it & wmask(ix.w);
		cx->env[d.islot] = &ix;
		for (k = 0; k < d.nst; k++)
			cx->env[d.sslot[k]] = &cur[k];
		for (k = 0; k < d.nst && ok; k++) {
			ok = aff_of(d.next[k], cx, &nxt[k], depth + 1);
			aff_fit(&nxt[k], d.sty[k]);
		}
		memcpy(cur, nxt, (size_t)d.nst * sizeof *cur);
	}
	cx->env[d.islot] = saved[MAXST];
	for (k = 0; k < d.nst; k++)
		cx->env[d.sslot[k]] = saved[k];
	if (ok) {
		*r = cur[loop_res(e)];
		aff_fit(r, N.ty[e]);
		cx->used |= AF_LOOP;
	}
	free(cur);
	return ok;
}

/* a constant table read at an affine index: affine when its entries are */
static int
aff_table(nid e, struct affcx *cx, struct aff *r, int depth)
{
	struct aff ix;
	nid tab = N.a[e];
	int w = int_width(N.ty[e]) ? int_width(N.ty[e]) : 64, nb = 0, i, j;
	uint64_t t0, D[8], v;

	if (N.op[tab] != O_TABLE || !aff_of(N.b[e], cx, &ix, depth + 1) ||
	    ix.bad)
		return 0;
	for (i = 0; i < 64; i++)
		if (!BIT0(&ix, i))
			nb = i + 1;
	if (nb > 8 || (1u << nb) > N.b[tab])
		return 0;
	for (v = 0; v < 1u << nb; v++) {
		nid c = N.lst[N.a[tab] + v];
		if (!is_const(c) || !TY_ISINT(N.ty[c]) || (w == 64 && N.ki[c] < 0))
			return 0;
	}
#define ENT(v) ((uint64_t)N.ki[N.lst[N.a[tab] + (v)]] & wmask(w))
	t0 = ENT(0);
	for (j = 0; j < nb; j++)
		D[j] = ENT(1u << j) ^ t0;
	for (v = 0; v < 1u << nb; v++) {
		uint64_t x = t0;
		for (j = 0; j < nb; j++)
			if ((v >> j) & 1)
				x ^= D[j];
		if (x != ENT(v))
			return 0;
	}
#undef ENT
	memset(r, 0, sizeof *r);
	r->w = w;
	r->k = t0;
	for (i = 0; i < w; i++)
		for (j = 0; j < nb; j++)
			if ((D[j] >> i) & 1) {
				r->row[i] ^= ix.row[j];
				r->k ^= ((ix.k >> j) & 1) << i;
			}
	cx->used |= AF_TAB;
	return 1;
}

static int
aff_of(nid e, struct affcx *cx, struct aff *r, int depth)
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
	if (depth > 200 || --cx->budget < 0)
		return 0;
	if (op == O_VAR && N.ki[e] >= MAXP && N.ki[e] < NSLOT &&
	    cx->env[N.ki[e]]) {
		*r = *cx->env[N.ki[e]];
		return 1;
	}
	if (op == O_INDEX)
		return aff_table(e, cx, r, depth);
	if (op == O_ADD || op == O_SUB || op == O_MUL || op == O_DIV ||
	    op == O_MOD || (!w && (op == O_AND || op == O_OR || op == O_XOR ||
	    op == O_SHL || op == O_SHR))) {
		/* constants (a loop's index among them) are worked out; an
		 * untyped integer is one, or nothing */
		int64_t x, y;
		if (!aff_of(N.a[e], cx, &a, depth + 1) ||
		    !aff_of(N.b[e], cx, &b, depth + 1))
			return 0;
		x = aff_value(&a, N.ty[N.a[e]]);
		y = aff_value(&b, N.ty[N.b[e]]);
		if (aff_const(&a) && aff_const(&b) && ((op != O_DIV &&
		    op != O_MOD && op != O_SHL && op != O_SHR) || (x >= 0 &&
		    y > 0 && (op == O_DIV || op == O_MOD || y < 63)))) {
			r->w = w ? w : 64;
			r->k = (uint64_t)(op == O_ADD ? a.k + b.k : op == O_SUB ?
			    a.k - b.k : op == O_MUL ? a.k * b.k : op == O_DIV ?
			    (uint64_t)(x / y) : op == O_MOD ? (uint64_t)(x % y) :
			    op == O_AND ? a.k & b.k : op == O_OR ? a.k | b.k :
			    op == O_XOR ? a.k ^ b.k : op == O_SHL ?
			    (uint64_t)x << y : (uint64_t)(x >> y)) & wmask(r->w);
			return 1;
		}
		if (op == O_DIV || op == O_MOD)
			return 0;
	}
	if (!w)
		return 0;
	switch (op) {
	case O_VAR:
		if (N.ki[e] < 0 || N.ki[e] >= MAXP || !cx->pw[N.ki[e]])
			return 0;
		for (i = 0; i < w && i < cx->pw[N.ki[e]]; i++)
			r->row[i] = 1ULL << (cx->off[N.ki[e]] + i);
		return 1;
	case O_LOOP:
		return aff_loop(e, cx, r, depth);
	case O_NOT:
		if (!aff_of(N.a[e], cx, r, depth + 1))
			return 0;
		r->k ^= wmask(w);
		r->w = w;
		return 1;
	case O_NEG:
	case O_SUB:
		/* a bit negated: that bit at every position */
		if (op == O_NEG ? !aff_of(N.a[e], cx, &b, depth + 1) :
		    (!aff_const(&a) || a.k))
			return 0;
		if (!aff_one_bit(&b))
			return 0;
		for (i = 0; i < w; i++)
			aff_take(r, i, &b, 0);
		cx->used |= AF_BIT;
		return 1;
	case O_MUL: {
		const struct aff *v = &a;
		uint64_t K;
		if (aff_const(&a))
			v = &b;
		else if (!aff_const(&b))
			return 0;
		K = (v == &a ? b.k : a.k) & wmask(w);
		/* a bit times a constant: the bit wherever the constant has one */
		if (!aff_one_bit(v))
			return 0;
		for (i = 0; i < w; i++)
			if ((K >> i) & 1)
				aff_take(r, i, v, 0);
		cx->used |= AF_BIT;
		return 1;
	}
	case O_SEL: {
		uint64_t crow;
		int ck;
		if (!aff_bit(N.a[e], cx, &crow, &ck, depth + 1) ||
		    !aff_of(N.b[e], cx, &a, depth + 1) ||
		    !aff_of(N.c[e], cx, &b, depth + 1))
			return 0;
		/* c ? a : b is b xor (c and (a xor b)): affine where a xor b
		 * is a constant */
		for (i = 0; i < w; i++) {
			uint64_t dk = ((a.k ^ b.k) >> i) & 1;
			if (a.row[i] != b.row[i] || BAD(&a, i) || BAD(&b, i)) {
				r->bad |= 1ULL << i;
				continue;
			}
			r->row[i] = b.row[i] ^ (dk ? crow : 0);
			r->k |= (((b.k >> i) & 1) ^ (dk & (uint64_t)ck)) << i;
		}
		cx->used |= AF_SEL;
		return 1;
	}
	case O_CAST: {
		ty_t ta = N.ty[N.a[e]];
		int wa = int_width(ta), sgn = TYK(ta) == K_SINT;
		if (!aff_of(N.a[e], cx, &a, depth + 1))
			return 0;
		if (!wa)
			wa = a.w;
		for (i = 0; i < w; i++) {
			int j = i < wa ? i : sgn ? wa - 1 : -1;
			if (j < 0)
				continue;
			aff_take(r, i, &a, j);
		}
		return 1;
	}
	case O_SHL: case O_SHR: {
		int64_t s;
		int sgn = TYK(N.ty[e]) == K_SINT;
		if (!aff_of(N.b[e], cx, &b, depth + 1) || !aff_const(&b) ||
		    !aff_of(N.a[e], cx, &a, depth + 1))
			return 0;
		s = aff_value(&b, N.ty[N.b[e]]);
		if (s < 0 || s >= w)
			return 0;
		for (i = 0; i < w; i++) {
			int j = op == O_SHL ? i - (int)s : i + (int)s;
			if (op == O_SHR && j >= w && sgn)
				j = w - 1;
			if (j < 0 || j >= w)
				continue;
			aff_take(r, i, &a, j);
		}
		return 1;
	}
	case O_XOR: case O_AND: case O_OR: case O_ADD:
		if (op != O_ADD && (!aff_of(N.a[e], cx, &a, depth + 1) ||
		    !aff_of(N.b[e], cx, &b, depth + 1)))
			return 0;
		for (i = 0; i < w; i++) {
			const struct aff *keep = NULL;
			int one = 0;
			if (op == O_XOR) {
				r->row[i] = a.row[i] ^ b.row[i];
				r->k |= (((a.k ^ b.k) >> i) & 1) << i;
				r->bad |= (uint64_t)(BAD(&a, i) | BAD(&b, i)) << i;
				continue;
			}
			if (op == O_AND) {
				if (BIT0(&a, i) || BIT0(&b, i))
					continue;
				if (BIT1(&a, i))
					keep = &b;
				else if (BIT1(&b, i))
					keep = &a;
			} else {
				/* or, +: no bit where both can be 1 (so no carry) */
				if (BIT0(&a, i))
					keep = &b;
				else if (BIT0(&b, i))
					keep = &a;
				else if (op == O_OR && (BIT1(&a, i) || BIT1(&b, i)))
					one = 1;
			}
			if (one)
				r->k |= 1ULL << i;
			else if (keep)
				aff_take(r, i, keep, i);
			else if (op == O_ADD) {
				/* a carry leaves here: nothing above is affine */
				r->bad |= ~0ULL << i;
				break;
			} else
				r->bad |= 1ULL << i;
		}
		r->bad &= wmask(w);
		return 1;
	}
	return 0;
}

int
gf2_affine(nid e, int slot, struct aff *r, int depth)
{
	struct affcx cx;

	(void)depth;
	memset(&cx, 0, sizeof cx);
	cx.pw[slot] = 64;
	cx.budget = 1 << 20;
	return aff_of(e, &cx, r, 0) && !r->bad;
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

/* a word the lanes share, and where its bits sit in a row */
struct sgword {
	nid x;
	int off, w;
};

/* the word a one-bit row reads, or -1 */
static int
row_word(uint64_t row, const struct sgword *wd, int nw)
{
	int p;

	if (popc64(row) != 1)
		return -1;
	for (p = 0; p < nw; p++)
		if ((row >> wd[p].off) & wmask(wd[p].w))
			return p;
	return -1;
}

/*
 * The bit a lane works out: one word's bit s[lane] when every row is one
 * bit (the shift amount lane + b, b - lane or lane xor b when the rows
 * follow one of those, else a table; with the rows in two words, the
 * word a table names), the parity of the words under row[lane]
 * otherwise.  kind: 0 parity, 1 a table of shifts, 2.. a closed shift, 5
 * two words, 6 the shift c + k lane.  A lane whose row is empty has no
 * bit to work out: the tables and the shift c + k lane leave it any, and
 * *live, the lanes that have one, masks the ballot.  The tables read at
 * the lane's index go to rd.
 */
static nid
lane_bit(const struct aff *m, const struct sgword *wd, int nw, int W, int kind,
    nid *rd, int *nrd, uint64_t *live)
{
	int n, i, p = -1, q = -1, l0 = -1, l1 = -1;
	nid L = sg_lane(), e[SG_MAXW], s = 0, x, one;
	int64_t sh[SG_MAXW], c = 0, k = 0;

	*nrd = 0;
	*live = 0;
	for (i = 0; i < W && i < m->w; i++)
		if (m->row[i])
			*live |= 1ULL << i;
	if (kind == 0) {
		nid acc = 0;
		int used = 0, maxw = 0;
		for (p = 0; p < nw; p++) {
			uint64_t any = 0;
			for (i = 0; i < W && i < m->w; i++)
				any |= (m->row[i] >> wd[p].off) & wmask(wd[p].w);
			if (any) {
				used++;
				if (wd[p].w > maxw)
					maxw = wd[p].w;
			}
		}
		for (p = 0; p < nw; p++) {
			uint64_t any = 0;
			nid t;
			for (i = 0; i < W; i++) {
				uint64_t rw = i < m->w ? (m->row[i] >> wd[p].off) &
				    wmask(wd[p].w) : 0;
				any |= rw;
				e[i] = mk_const_i(N.ty[wd[p].x], (int64_t)rw);
			}
			if (!any)
				continue;
			rd[*nrd] = mk_index(mk_table(e, (uint32_t)W), L);
			t = mk_bin(O_AND, wd[p].x, rd[(*nrd)++]);
			if (used > 1)
				t = mk_cast(TY(K_UINT, maxw), t);
			acc = acc ? mk_bin(O_XOR, acc, t) : t;
		}
		return mk_bin(O_NE, mk_bin(O_AND, mk_un(O_POPCNT, acc),
		    mk_const_i(TY_LIT, 1)), mk_const_i(TY_LIT, 0));
	}
	if (*live != wmask(W) && kind != 1 && kind != 5 && kind != 6)
		return 0;
	for (i = 0; i < W; i++) {
		int pi;
		sh[i] = 0;
		if (!((*live >> i) & 1))
			continue;
		pi = row_word(m->row[i], wd, nw);
		if (pi < 0)
			return 0;
		if (p < 0)
			p = pi;
		if (pi != p) {
			if (q >= 0 && pi != q)
				return 0;
			q = pi;
		}
		sh[i] = ctz64(m->row[i]) - wd[pi].off;
		if (l0 < 0)
			l0 = i;
		else if (l1 < 0)
			l1 = i;
	}
	if (p < 0 || (kind == 5) != (q >= 0))
		return 0;
	x = wd[p].x;
	n = wd[p].w;
	one = mk_const_i(N.ty[x], 1);
	switch (kind) {
	case 1:
		for (i = 0; i < W; i++)
			e[i] = mk_const_i(TY(K_UINT, 32), sh[i]);
		s = rd[(*nrd)++] = mk_index(mk_table(e, (uint32_t)W), L);
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
	case 5: {	/* two words: which one, and the shift, by tables */
		ty_t t = TY(K_UINT, wd[p].w > wd[q].w ? wd[p].w : wd[q].w);
		nid which = 0;
		int b;
		for (i = 0; i < W; i++)
			e[i] = mk_const_i(TY(K_UINT, 32), sh[i]);
		s = rd[(*nrd)++] = mk_index(mk_table(e, (uint32_t)W), L);
		/* the word a bit of the lane's index names, else a table */
		for (b = 0; b < 6 && !which; b++) {
			for (i = 0; i < W; i++)
				if (((*live >> i) & 1) && (row_word(m->row[i], wd,
				    nw) == q) != ((i >> b) & 1))
					break;
			if (i == W)
				which = mk_bin(O_AND, b ? mk_bin(O_SHR, L,
				    mk_const_i(TY(K_UINT, 32), b)) : L,
				    mk_const_i(TY(K_UINT, 32), 1));
		}
		if (!which) {
			for (i = 0; i < W; i++)
				e[i] = mk_const_i(TY(K_UINT, 32), ((*live >> i) &
				    1) && row_word(m->row[i], wd, nw) == q);
			which = rd[(*nrd)++] = mk_index(mk_table(e, (uint32_t)W),
			    L);
		}
		x = mk_sel(mk_bin(O_NE, which, mk_const_i(TY_LIT, 0)),
		    mk_cast(t, wd[q].x), mk_cast(t, wd[p].x));
		one = mk_const_i(t, 1);
		break;
	}
	case 6:		/* c + k lane, in the lanes that have a bit */
		if (l1 < 0 || (sh[l1] - sh[l0]) % (l1 - l0))
			return 0;
		k = (sh[l1] - sh[l0]) / (l1 - l0);
		c = sh[l0] - k * l0;
		if (k == 0 || ((k == 1 || k == -1) && *live == wmask(W)))
			return 0;
		for (i = 0; i < W; i++)
			if (((*live >> i) & 1) && sh[i] != c + k * i)
				return 0;
		s = k > 0 ? mk_bin(O_MUL, L, mk_const_i(TY(K_UINT, 32), k)) :
		    mk_bin(O_MUL, L, mk_const_i(TY(K_UINT, 32), -k));
		if (k == 1 || k == -1)
			s = L;
		s = k > 0 ? (c ? mk_bin(O_ADD, s, mk_const_i(TY(K_UINT, 32), c)) :
		    s) : mk_bin(O_SUB, mk_const_i(TY(K_UINT, 32), c), s);
		/* the lanes without a bit shift inside the word too */
		if (c < 0 || c >= n || c + k * (W - 1) < 0 || c + k * (W - 1) >= n) {
			if (n & (n - 1))
				return 0;
			s = mk_bin(O_AND, s, mk_const_i(TY(K_UINT, 32), n - 1));
		}
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
	    "lane's index xor a constant", "every row has one bit: that bit "
	    "of one of the two words, which word and its position by the "
	    "lane's index", "every row has one bit, or none: that bit of the "
	    "word, its position a multiple of the lane's index plus a "
	    "constant"};
	uint64_t kk;
	const struct func *f = g->f;
	int W = (int)g->m->par[MP_WARP], nw = 0, tot = 0, j, kind, bk = -1;
	int closed = 0, used, rows = 0;
	struct sbuf how2 = {0}, names = {0};
	uint64_t slots = node_slots(cur);
	struct affcx cx;
	struct sgword wd[MAXP];
	struct aff *m;
	nid best = 0;
	float bc = *cost;
	ty_t t = N.ty[cur], bt = TY(K_UINT, W > 32 ? 64 : 32);

	if (W < 2 || W > SG_MAXW || !cclass_available(g->m, C_SG_BALLOT) ||
	    (N.fl[cur] & NF_SG) ||
	    !int_width(t) || int_width(t) > W)
		return cur;
	/* the words the function reads: each shared by the lanes, side by
	 * side in a row */
	memset(&cx, 0, sizeof cx);
	cx.budget = 1 << 20;
	for (j = 0; j < f->np; j++)
		if ((slots >> j) & 1) {
			int pw = int_width(f->p[j].ty);
			if (!f->p[j].uniform || !pw || tot + pw > 64)
				return cur;
			cx.off[j] = tot;
			cx.pw[j] = pw;
			wd[nw].x = mk_var((uint32_t)j, f->p[j].name, f->p[j].ty);
			wd[nw].off = tot;
			wd[nw++].w = pw;
			tot += pw;
		}
	if (!nw || (slots >> MAXP))
		return cur;
	/* several words go side by side as unsigned ones */
	for (j = 0; j < nw && nw > 1; j++)
		if (TYK(N.ty[wd[j].x]) != K_UINT)
			return cur;
	m = xmalloc(sizeof *m);
	if (!aff_of(cur, &cx, m, 0) || m->bad) {
		free(m);
		return cur;
	}
	used = cx.used;
	for (j = 0; j < m->w && j < W; j++)
		rows += m->row[j] != 0;
	for (kind = 0; kind < 7 && rows; kind++) {
		nid rd[MAXP + 1], b, cand;
		struct sbuf h = {0};
		uint64_t live;
		int took = 0, nrd, q;
		float c;
		b = lane_bit(m, wd, nw, W, kind, rd, &nrd, &live);
		if (!b)
			continue;
		cand = mk_sg(O_SGBALLOT, b, mk_const_i(TY_LIT, W), 0);
		if (kind && live != wmask(W))
			cand = mk_bin(O_AND, cand, mk_const_i(bt, (int64_t)live));
		if (m->k)
			cand = mk_bin(O_XOR, cand, mk_const_i(bt, (int64_t)m->k));
		cand = mk_cast(t, cand);
		c = price(g, cand);
		for (q = 0; q < nrd; q++) {
			/* the rows (or shifts) a lane reads, as a closed form of
			 * the lane; a row past the result's width is cut away */
			struct sbuf h1 = {0};
			nid fm, c2 = lane_forms(g, cand, rd[q], kind == 0 ? m->w : W,
			    W, &c, &h1, &fm);
			if (c2 != cand) {
				sb_printf(&h, "%s%s", took ? "; " : "", sb_str(&h1));
				took = 1;
			}
			cand = c2;
			free(h1.p);
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
	for (j = 0; j < nw; j++)
		sb_printf(&names, "%s%s", j == 0 ? "" : j == nw - 1 ? " and " :
		    ", ", node_str(wd[j].x));
	sb_printf(&g->why, "every operation of the function is linear over "
	    "GF(2) in the bits of %s (xor, not, shifts and masks by constants, "
	    "conversions, an or or a sum of fields that share no bit%s%s%s%s), "
	    "so bit i of the result is the parity of ", sb_str(&names),
	    (used & AF_BIT) ? ", a bit negated or times a constant" : "",
	    (used & AF_SEL) ? ", a ?: on one bit whose sides differ by a "
	    "constant" : "", (used & AF_TAB) ? ", a table that is linear in "
	    "its index, every entry checked" : "", (used & AF_LOOP) ? ", a "
	    "loop of a constant count run through" : "");
	if (nw == 1)
		sb_printf(&g->why, "%s & row[i]", sb_str(&names));
	else
		sb_printf(&g->why, "the bits of %s that row[i] names",
		    sb_str(&names));
	sb_printf(&g->why, "%s; lane i works that "
	    "bit out (%s%s%s) and the ballot puts it at bit i\n", kk ?
	    ", xor bit i of a constant" : "", how[bk], closed ? "; the table "
	    "read without a load, from the lane's index: " : "", closed ?
	    sb_str(&how2) : "");
	free(how2.p);
	free(names.p);
	*cost = bc;
	return best;
}

/* ---- a map applied field by field ---- */

#define FM_BITS 8		/* a field's bits, at most */
#define FM_POINTS 48		/* the values each bit is flipped at */
#define FM_TABLE 1024		/* the entries of the fields' functions */

static int
msb64(uint64_t v)
{
	int n = 0;

	while (v >>= 1)
		n++;
	return n;
}

/* f's body at n values of its word; 0 when one of them is undefined */
static int
fm_eval(struct plan *p, const struct func *f, int slot, const uint64_t *x,
    uint64_t *y, int n, int64_t *in_i, double *in_r)
{
	int i;

	for (i = 0; i < n; i++)
		in_i[(size_t)slot * CH + i] = int_norm(f->p[slot].ty,
		    (int64_t)x[i]);
	plan_run(p, (uint32_t)n, in_i, in_r);
	for (i = 0; i < n; i++) {
		if (COLF(p, p->root[0])[i] & F_UNDEF)
			return 0;
		y[i] = (uint64_t)COLI(p, p->root[0])[i];
	}
	return 1;
}

/* v[i] in the lanes of live as a form of the lane's index: lane & c, a
 * field of it times m, or c0 + m (lane >> a), inside 0..lim-1 in every
 * lane; 0 when it is none */
static nid
fm_closed(const int *v, uint64_t live, int W, int lim)
{
	nid L = sg_lane();
	ty_t t = TY(K_UINT, 32);
	int c = 0, i, a;

	for (i = 0; i < W; i++)
		if ((live >> i) & 1)
			c |= v[i];
	for (i = 0; i < W; i++)
		if (((live >> i) & 1) && v[i] != (i & c))
			break;
	if (i == W && ((W - 1) & c) < lim)
		return c ? mk_bin(O_AND, L, mk_const_i(t, c)) : mk_const_i(t, 0);
	for (a = 0; a < 6; a++) {
		int b, m = 0;
		for (b = 1; a + b <= 6; b++) {
			int fm = (1 << b) - 1;
			for (i = 0, m = 0; i < W && !m; i++)
				if (((live >> i) & 1) && ((i >> a) & fm) == 1)
					m = v[i];
			if (m <= 0 || m * fm >= lim)
				continue;
			for (i = 0; i < W; i++)
				if (((live >> i) & 1) && v[i] != ((i >> a) & fm) * m)
					break;
			if (i == W && !(m & (m - 1))) {
				/* m a power of two: the field moved to its place */
				int sh = ctz64((uint64_t)m) - a;
				nid s = mk_bin(O_AND, L, mk_const_i(t, fm << a));
				return sh > 0 ? mk_bin(O_SHL, s, mk_const_i(t, sh)) :
				    sh < 0 ? mk_bin(O_SHR, s, mk_const_i(t, -sh)) : s;
			}
			if (i == W) {
				nid s = mk_bin(O_AND, a ? mk_bin(O_SHR, L,
				    mk_const_i(t, a)) : L, mk_const_i(t, fm));
				return mk_bin(O_MUL, s, mk_const_i(t, m));
			}
		}
	}
	for (a = 1; a < 6; a++) {
		int i0 = -1, i1 = -1, m, c0;
		for (i = 0; i < W; i++) {
			if (!((live >> i) & 1))
				continue;
			if (i0 < 0)
				i0 = i;
			else if (i1 < 0 && (i >> a) != (i0 >> a))
				i1 = i;
		}
		if (i1 < 0 || (v[i1] - v[i0]) % ((i1 >> a) - (i0 >> a)))
			continue;
		m = (v[i1] - v[i0]) / ((i1 >> a) - (i0 >> a));
		c0 = v[i0] - m * (i0 >> a);
		if (m <= 0 || c0 < 0 || c0 + m * ((W - 1) >> a) >= lim)
			continue;
		for (i = 0; i < W; i++)
			if (((live >> i) & 1) && v[i] != c0 + m * (i >> a))
				break;
		if (i == W) {
			nid s = mk_bin(O_SHR, L, mk_const_i(t, a));
			if (m > 1)
				s = mk_bin(O_MUL, s, mk_const_i(t, m));
			return c0 ? mk_bin(O_ADD, s, mk_const_i(t, c0)) : s;
		}
	}
	return 0;
}

/*
 * A map applied field by field to a word the lanes share: every bit of
 * the result reads one field of the word, at most FM_BITS wide (an S-box
 * on each nibble).  Lane i works out its bit, its field's function of
 * the field at bit i's place, and the ballot puts the bits together; the
 * functions the fields go through, FM_TABLE entries of them at most, sit
 * one after the other in one table.  Which bits of the word a bit
 * of the result reads is a guess, from the function with each bit
 * flipped at FM_POINTS values; the form built on it is kept only when it
 * is proved equal to the function (every point, or SAT), in g->pr.
 */
static nid
fieldmap(struct sgf *g, nid cur, float *cost, struct sbuf *why)
{
	const struct func *f = g->f;
	struct fld {
		int lo, hi;
		uint64_t out;
	} fl[64], tf;
	struct plan p;
	struct sbuf wit = {0};
	uint64_t slots = node_slots(cur), dep[64], x[CH], y[CH], y0[FM_POINTS];
	uint64_t *H, hq[1 << FM_BITS], live = 0, kc;
	uint64_t all = 0, seed = 0x9e3779b97f4a7c15ULL;
	int W = (int)g->m->par[MP_WARP], wr = int_width(N.ty[cur]), slot = -1;
	int n, i, j, q, nf = 0, k, nout, sv[SG_MAXW], bv[SG_MAXW], ok = 1, bad;
	int untyped = 0, nfun = 0, fv[SG_MAXW];
	int64_t *in_i;
	double *in_r;
	nid roots[1], xv, sexp, bexp, v, h, cand, *e, rd[3];
	ty_t bt = TY(K_UINT, W > 32 ? 64 : 32), ht;
	float c;
	int nrd = 0;

	if (W < 2 || W > SG_MAXW || !cclass_available(g->m, C_SG_BALLOT) ||
	    (N.fl[cur] & NF_SG) || f->where || (slots >> MAXP))
		return cur;
	for (j = 0; j < f->np; j++)
		if ((slots >> j) & 1) {
			if (slot >= 0)
				return cur;
			slot = j;
		}
	if (slot < 0 || !f->p[slot].uniform || f->p[slot].ranged ||
	    TYK(f->p[slot].ty) != K_UINT)
		return cur;
	if (TYK(N.ty[cur]) == K_LIT) {
		/* an untyped result: the bits its proved range needs */
		int64_t lo, hi;
		if (!sg_range_of(f, cur, W, &lo, &hi) || lo < 0 ||
		    (uint64_t)hi > wmask(W))
			return cur;
		wr = msb64((uint64_t)hi) + 1;
		untyped = 1;
	}
	if (!wr || wr > W)
		return cur;
	n = TYW(f->p[slot].ty);
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	roots[0] = cur;
	plan_build(&p, roots, 1, 0);
	/* the bits of the word each bit of the result changes with */
	for (i = 0; i < FM_POINTS; i++) {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		x[i] = i == 0 ? 0 : i == 1 ? wmask(n) : seed & wmask(n);
	}
	ok = fm_eval(&p, f, slot, x, y0, FM_POINTS, in_i, in_r);
	memset(dep, 0, sizeof dep);
	for (j = 0; j < n && ok; j++) {
		for (i = 0; i < FM_POINTS; i++)
			x[i] ^= 1ULL << j;
		ok = fm_eval(&p, f, slot, x, y, FM_POINTS, in_i, in_r);
		for (i = 0; i < FM_POINTS; i++) {
			uint64_t d = (y[i] ^ y0[i]) & wmask(wr);
			x[i] ^= 1ULL << j;
			for (; d; d &= d - 1)
				dep[ctz64(d)] |= 1ULL << j;
		}
	}
	/* the fields: the spans of those bits, joined where they overlap */
	for (i = 0; i < wr && ok; i++) {
		if (!dep[i])
			continue;
		tf.lo = ctz64(dep[i]);
		tf.hi = msb64(dep[i]);
		tf.out = 1ULL << i;
		for (q = 0; q < nf;)
			if (fl[q].lo <= tf.hi && tf.lo <= fl[q].hi) {
				if (fl[q].lo < tf.lo)
					tf.lo = fl[q].lo;
				if (fl[q].hi > tf.hi)
					tf.hi = fl[q].hi;
				tf.out |= fl[q].out;
				fl[q] = fl[--nf];
				q = 0;
			} else
				q++;
		fl[nf++] = tf;
	}
	/* in the order of their bits of the result */
	for (q = 1; q < nf; q++)
		for (j = q; j > 0 && ctz64(fl[j].out) < ctz64(fl[j - 1].out); j--) {
			tf = fl[j];
			fl[j] = fl[j - 1];
			fl[j - 1] = tf;
		}
	k = nf ? fl[0].hi - fl[0].lo + 1 : 0;
	nout = 0;
	ok = ok && nf >= 2 && k <= FM_BITS;
	/* each field's function, tabulated with the other fields 0; the
	 * different ones one after the other */
	H = xmalloc(FM_TABLE * sizeof *H);
	for (q = 0; q < nf && ok; q++) {
		int d = ctz64(fl[q].out), fn;
		if (fl[q].hi - fl[q].lo + 1 != k) {
			ok = 0;
			break;
		}
		if (msb64(fl[q].out >> d) + 1 > nout)
			nout = msb64(fl[q].out >> d) + 1;
		for (i = 0; i < 1 << k; i++)
			x[i] = (uint64_t)i << fl[q].lo;
		ok = fm_eval(&p, f, slot, x, y, 1 << k, in_i, in_r);
		for (i = 0; i < 1 << k && ok; i++)
			hq[i] = (y[i] & fl[q].out) >> d;
		for (fn = 0; fn < nfun; fn++)
			if (!memcmp(H + ((size_t)fn << k), hq, ((size_t)1 << k) *
			    sizeof *hq))
				break;
		if (fn == nfun && ok) {
			if ((nfun + 1) << k > FM_TABLE)
				ok = 0;
			else
				memcpy(H + ((size_t)nfun++ << k), hq, ((size_t)1 << k) *
				    sizeof *hq);
		}
		all |= fl[q].out;
		for (i = 0; i < wr; i++)
			if ((fl[q].out >> i) & 1) {
				live |= 1ULL << i;
				sv[i] = fl[q].lo;
				bv[i] = i - d;
				fv[i] = fn << k;
			}
	}
	kc = y0[0] & ~all & wmask(wr);
	plan_free(&p);
	free(in_i);
	free(in_r);
	if (!ok) {
		free(H);
		return cur;
	}
	for (i = 0; i < W; i++)
		if (!((live >> i) & 1))
			sv[i] = bv[i] = fv[i] = 0;
	/* where a lane's field starts and which bit of the function is its
	 * own: forms of the lane's index, else tables read at it */
	e = xmalloc(FM_TABLE * sizeof *e);
	sexp = fm_closed(sv, live, W, n);
	if (!sexp) {
		for (i = 0; i < W; i++)
			e[i] = mk_const_i(TY(K_UINT, 32), sv[i]);
		sexp = rd[nrd++] = mk_index(mk_table(e, (uint32_t)W), sg_lane());
	}
	bexp = fm_closed(bv, live, W, nout);
	if (!bexp) {
		for (i = 0; i < W; i++)
			e[i] = mk_const_i(TY(K_UINT, 32), bv[i]);
		bexp = rd[nrd++] = mk_index(mk_table(e, (uint32_t)W), sg_lane());
	}
	xv = mk_var((uint32_t)slot, f->p[slot].name, f->p[slot].ty);
	v = mk_bin(O_AND, mk_bin(O_SHR, xv, sexp), mk_const_i(f->p[slot].ty,
	    (1 << k) - 1));
	if (nfun > 1) {
		/* the lane's function starts at its number times 2^k */
		nid fexp = fm_closed(fv, live, W, nfun << k);
		if (!fexp) {
			for (i = 0; i < W; i++)
				e[i] = mk_const_i(TY(K_UINT, 32), fv[i]);
			fexp = rd[nrd++] = mk_index(mk_table(e, (uint32_t)W),
			    sg_lane());
		}
		v = mk_bin(O_ADD, mk_cast(TY(K_UINT, 32), v), fexp);
	}
	ht = TY(K_UINT, nout > 32 ? 64 : 32);
	for (i = 0; i < nfun << k; i++)
		e[i] = mk_const_i(ht, (int64_t)H[i]);
	h = mk_index(mk_table(e, (uint32_t)nfun << k), v);
	free(e);
	free(H);
	cand = mk_sg(O_SGBALLOT, mk_bin(O_NE, mk_bin(O_AND, mk_bin(O_SHR, h,
	    bexp), mk_const_i(ht, 1)), mk_const_i(TY_LIT, 0)), mk_const_i(TY_LIT,
	    W), 0);
	if (live != wmask(W))
		cand = mk_bin(O_AND, cand, mk_const_i(bt, (int64_t)live));
	if (kc)
		cand = mk_bin(O_XOR, cand, mk_const_i(bt, (int64_t)kc));
	if (!untyped)
		cand = mk_cast(N.ty[cur], cand);
	c = price(g, cand);
	for (q = 0; q < nrd; q++) {
		struct sbuf h1 = {0};
		nid fm;
		cand = lane_forms(g, cand, rd[q], W, W, &c, &h1, &fm);
		free(h1.p);
	}
	if (c >= *cost)
		return cur;
	/* the guess stands only when the whole is proved */
	bad = equiv_decide(f, f->body, cand, &wit, &g->pr);
	free(wit.p);
	if (bad || (g->pr.how != EQ_EXHAUSTIVE && g->pr.how != EQ_SAT))
		return cur;
	g->proved = cand;
	sb_printf(why, "every bit of the result reads one field of %s and "
	    "nothing else, %d fields of %d bits (told by flipping each bit of "
	    "%s at %d values of it), and ", sym_str(f->p[slot].name), nf, k,
	    sym_str(f->p[slot].name), FM_POINTS);
	if (nfun == 1)
		sb_printf(why, "each field goes through the same function, "
		    "tabulated at its %d values with the other fields 0; lane i "
		    "works out bit i, the function of the field it belongs to at "
		    "the bit's place in it", 1 << k);
	else
		sb_printf(why, "the fields go through %d functions, each "
		    "tabulated at its %d values with the other fields 0, one "
		    "after the other in a table; lane i works out bit i, its "
		    "field's function of the field at the bit's place in it",
		    nfun, 1 << k);
	sb_printf(why, ", and the ballot puts it at bit i; a guess until the "
	    "check below, which proves the form equal to the function\n");
	*cost = c;
	return cand;
}

/* ---- a wide sum, a limb in each lane ---- */

/* the terms of a sum, without the conversions between them */
static int
sum_terms(nid e, nid *out, int n, int max, int depth)
{
	if (N.op[e] == O_ADD && depth < 16) {
		n = sum_terms(N.a[e], out, n, max, depth + 1);
		return n < 0 ? n : sum_terms(N.b[e], out, n, max, depth + 1);
	}
	if (n >= max)
		return -1;
	out[n] = e;
	return n + 1;
}

/*
 * The sum of two numbers that have a limb in each lane (`lanes`
 * parameters, or values worked out from them in the lane): a + b plus
 * the carry the lanes below send in, however the function works that
 * out (a loop over the lanes below, reading them by shuffles).  With g
 * the lanes whose a + b carries out and p those where it is all ones,
 * which pass a carry on, the carries into the lanes are ((g << 1) + p)
 * xor p: one add of two ballots, a Kogge-Stone prefix done by the
 * adder.  Only the shape a + b + (something that reads the other lanes)
 * is read from the function; that the something is the carry is proved
 * by SAT over the lanes' words (equiv_decide), in g->pr.
 */
static nid
wideadd(struct sgf *g, nid cur, float *cost)
{
	const struct func *f = g->f;
	nid t[8], lim[2], body = cur, s, gen, prop, G, P, ks, cin, cand, L;
	nid map[MAXP] = {0};
	ty_t ty = N.ty[cur], bt;
	int W = sg_lanes_width(f), w = int_width(ty), n, i, j, nl = 0, nc = 0;
	uint64_t mask = 0;
	struct sbuf wit = {0};
	float c;

	L = lane_param(f);
	if (!L || W < 2 || TYK(ty) != K_UINT || !cclass_available(g->m,
	    C_SG_BALLOT))
		return cur;
	/* limbs narrower than their type: the sum under a mask */
	if (N.op[body] == O_AND && is_const(N.b[body]) && N.ki[N.b[body]] > 0 &&
	    !((uint64_t)(N.ki[N.b[body]] + 1) & (uint64_t)N.ki[N.b[body]]) &&
	    (uint64_t)N.ki[N.b[body]] < wmask(w) >> 1) {
		mask = (uint64_t)N.ki[N.b[body]];
		body = N.a[body];
	}
	n = sum_terms(body, t, 0, 8, 0);
	if (n < 3)
		return cur;
	for (i = 0; i < n; i++) {
		if (N.fl[t[i]] & NF_SG)
			nc++;
		else if (nl < 2)
			lim[nl++] = t[i];
		else
			return cur;
	}
	if (nl != 2 || !nc || N.ty[lim[0]] != ty || N.ty[lim[1]] != ty)
		return cur;
	/* the limbs as an operation reads them in every lane */
	for (j = 0; j < f->np; j++) {
		uint32_t lane = sym_cstr("lane");
		if (!((node_slots(lim[0]) | node_slots(lim[1])) >> j & 1))
			continue;
		if (f->p[j].name == lane)
			map[j] = mk_cast(f->p[j].ty, sg_lane());
		else if (f->p[j].lanes)
			map[j] = sg_lanes_copy(j, f->p[j].name, f->p[j].ty, W);
		else if (!f->p[j].uniform)
			return cur;
	}
	bt = TY(K_UINT, W > 32 ? 64 : 32);
	s = mk_bin(O_ADD, lim[0], lim[1]);
	if (mask) {
		gen = mk_bin(O_GT, s, mk_const_i(ty, (int64_t)mask));
		prop = mk_bin(O_EQ, s, mk_const_i(ty, (int64_t)mask));
	} else {
		gen = mk_bin(O_LT, s, lim[0]);
		prop = mk_bin(O_EQ, s, mk_const_i(ty, (int64_t)wmask(w)));
	}
	G = mk_sg(O_SGBALLOT, subst(gen, map, f->np), mk_const_i(TY_LIT, W), 0);
	P = mk_sg(O_SGBALLOT, subst(prop, map, f->np), mk_const_i(TY_LIT, W), 0);
	ks = mk_bin(O_XOR, mk_bin(O_ADD, mk_bin(O_SHL, G, mk_const_i(bt, 1)), P),
	    P);
	cin = mk_cast(ty, mk_bin(O_AND, mk_bin(O_SHR, ks, mk_cast(bt, L)),
	    mk_const_i(bt, 1)));
	cand = mk_bin(O_ADD, s, cin);
	if (mask)
		cand = mk_bin(O_AND, cand, mk_const_i(ty, (int64_t)mask));
	/* a guess from the shape: it stands when SAT proves it */
	if (equiv_decide(f, f->body, cand, &wit, &g->pr) || g->pr.how != EQ_SAT) {
		free(wit.p);
		return cur;
	}
	free(wit.p);
	c = price(g, cand);
	if (c >= *cost) {
		note_dear(g, "a wide sum's carries as one add of two ballots", c,
		    *cost);
		return cur;
	}
	g->proved = cand;
	sb_printf(&g->why, "the function is %s + %s plus what the lanes below "
	    "send in, however it works that out; taken as the carry of a sum "
	    "with a limb in each lane: with g the lanes where the limbs' sum "
	    "carries out and p those where it is all ones, which pass a carry "
	    "on, the carries into the lanes are ((g << 1) + p) xor p, one add "
	    "of two ballots, and each lane adds its bit; a guess from the "
	    "shape until the check below, which proves the form equal to the "
	    "function over the %d words of the lanes\n", node_str(lim[0]),
	    node_str(lim[1]), 2 * W);
	*cost = c;
	return cand;
}

/* ---- a carry chain over columns the lanes can compute ---- */

/*
 * A product of two numbers with a limb in each lane, written limb by
 * limb (the schoolbook): lane l takes the columns k = 0..l in turn, each
 * the sum of the products a_i b_(k-i), adds the carry of the one below
 * and keeps what is above its limb as the next carry.  Every lane sums
 * every column below its own.  A column is a function of k alone, the
 * same in every lane, so lane k computes it once and the others read it
 * by a shuffle (the column form).  The carry chain over the columns is
 * then carry-save steps, lane l keeping its value's low w bits and
 * adding the bits above of lane l - 1's, until each lane's value is
 * below 2^(w+1), whose carries are a wide sum's: one add of two ballots.
 *
 * A column is a loop that adds to one state (src/loop.c's entry):
 * s = s + t or s = q ? s + t : s, t and q not reading s, so its value
 * from any start is the start plus its value from 0 (mod 2^N); it reads
 * an enclosing loop's index k (its column) and values the lanes share,
 * k in no operand of a subgroup operation (there the lane would be the
 * operand's own).
 */
struct col {
	nid L;			/* the column's loop */
	nid inc;		/* what an iteration adds (0 where q fails) */
	int kslot;		/* the column's index: an enclosing loop's */
	uint32_t kname;
	ty_t kty;
	int64_t ilo, ihi;	/* the column loop's own index */
	int islot;
};

#define MAXCOL 4
#define MAXENC 8

struct cwalk {
	const struct func *f;
	int W, n, depth;
	int masked;		/* an sg_active test met, in a loop or not */
	uint8_t *seen;
	struct col c[MAXCOL];
	/* the enclosing counted loops with constant bounds */
	int eslot[MAXENC];
	uint32_t ename[MAXENC];
	ty_t ety[MAXENC];
	int64_t elo[MAXENC], ehi[MAXENC];
};

/* the integers' widths a column's casts may go through: as wide as s */
static int
iwidth(ty_t t)
{
	if (TYK(t) == K_INT || TYK(t) == K_LIT)
		return 64;
	return int_width(t);
}

/* x is the state (slot s) under casts to integers at least w wide */
static int
is_state(nid x, int s, int w)
{
	while (N.op[x] == O_CAST && TY_ISINT(N.ty[x]) && iwidth(N.ty[x]) >= w)
		x = N.a[x];
	return N.op[x] == O_VAR && (int)N.ki[x] == s;
}

/* what next adds to the state s each iteration, or 0 */
static nid
addend(nid nx, int s, int w)
{
	nid x = nx, t;

	while (N.op[x] == O_CAST && TY_ISINT(N.ty[x]) && iwidth(N.ty[x]) >= w)
		x = N.a[x];
	if (N.op[x] == O_SEL) {
		if (has_slot(N.a[x], s))
			return 0;
		if (is_state(N.c[x], s, w) && (t = addend(N.b[x], s, w)) != 0)
			return mk_sel(N.a[x], t, mk_const_i(N.ty[t], 0));
		if (is_state(N.b[x], s, w) && (t = addend(N.c[x], s, w)) != 0)
			return mk_sel(N.a[x], mk_const_i(N.ty[t], 0), t);
		return 0;
	}
	if (N.op[x] != O_ADD)
		return 0;
	if (is_state(N.a[x], s, w) && !has_slot(N.b[x], s))
		return N.b[x];
	if (is_state(N.b[x], s, w) && !has_slot(N.a[x], s))
		return N.a[x];
	return 0;
}

/* slot k is read inside an operand of a subgroup operation in e */
static int
k_in_operand(nid e, int k, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || !has_slot(e, k))
		return 0;
	seen[e] = 1;
	if (IS_SG(N.op[e]))
		return has_slot(N.a[e], k) || k_in_operand(N.b[e], k, seen);
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if (k_in_operand(N.lst[N.a[e] + j], k, seen))
				return 1;
		return 0;
	}
	if (N.op[e] == O_LOOP) {
		uint32_t le = loop_ent(e), st = loops.st[le], q;
		for (q = 0; q < loops.nst[le]; q++)
			if (k_in_operand(loops.init[st + q], k, seen) ||
			    k_in_operand(loops.next[st + q], k, seen))
				return 1;
		if (k_in_operand(loops.cond[le], k, seen))
			return 1;
	}
	return k_in_operand(N.a[e], k, seen) || k_in_operand(N.b[e], k, seen) ||
	    k_in_operand(N.c[e], k, seen);
}

/* n a column of f (above), recorded in w */
static void
try_col(struct cwalk *w, nid n)
{
	const struct func *f = w->f;
	struct loopdef d;
	struct col c;
	int64_t lo, hi;
	uint64_t fr;
	uint8_t *seen;
	int j, s, bad;

	if (w->n == MAXCOL)
		return;
	loop_get(loop_ent(n), &d);
	if (d.kind != LK_FOR || d.nst != 1 || loop_res(n) != 0 ||
	    TYK(d.sty[0]) != K_UINT || !int_of(N.a[n], &lo) ||
	    !int_of(N.b[n], &hi) || lo > hi ||
	    (N.fl[d.next[0]] & (NF_LANEREAD | NF_LANE)))
		return;
	s = d.sslot[0];
	memset(&c, 0, sizeof c);
	c.inc = addend(d.next[0], s, int_width(d.sty[0]));
	if (!c.inc)
		return;
	/* what it reads besides its own: one enclosing index, shared values */
	fr = node_slots(d.next[0]) & ~(1ULL << s) & ~(1ULL << d.islot);
	c.kslot = -1;
	for (j = 0; j < NSLOT; j++) {
		int e;
		if (!((fr >> j) & 1))
			continue;
		if (j < f->np) {
			if (!f->p[j].uniform)
				return;
			continue;
		}
		for (e = w->depth - 1; e >= 0 && w->eslot[e] != j; e--)
			;
		if (e < 0 || c.kslot >= 0)
			return;
		if (w->elo[e] < 0 || w->ehi[e] > w->W - 1)
			return;
		c.kslot = j;
		c.kname = w->ename[e];
		c.kty = w->ety[e];
	}
	if (c.kslot < 0)
		return;
	seen = xcalloc(N.n, 1);
	bad = k_in_operand(d.next[0], c.kslot, seen);
	free(seen);
	if (bad)
		return;
	c.L = n;
	c.islot = d.islot;
	c.ilo = lo;
	c.ihi = hi;
	w->c[w->n++] = c;
}

static void
walk_cols(struct cwalk *w, nid e)
{
	uint32_t j;

	if (e == 0 || w->seen[e])
		return;
	w->seen[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_UNDEF)
		return;
	if (sg_active_of(e, NULL))
		w->masked = 1;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			walk_cols(w, N.lst[N.a[e] + j]);
		return;
	}
	if (N.op[e] == O_LOOP) {
		uint32_t le = loop_ent(e), st = loops.st[le], q;
		int64_t lo, hi;
		int pushed = 0;
		try_col(w, e);
		if (loops.kind[le] == LK_FOR && int_of(N.a[e], &lo) &&
		    int_of(N.b[e], &hi) && w->depth < MAXENC) {
			w->eslot[w->depth] = loops.islot[le];
			w->ename[w->depth] = loops.name[le];
			w->ety[w->depth] = loops.ity[le];
			w->elo[w->depth] = lo;
			w->ehi[w->depth] = hi;
			w->depth++;
			pushed = 1;
		}
		for (q = 0; q < loops.nst[le]; q++) {
			walk_cols(w, loops.init[st + q]);
			walk_cols(w, loops.next[st + q]);
		}
		walk_cols(w, loops.cond[le]);
		w->depth -= pushed;
	}
	walk_cols(w, N.a[e]);
	walk_cols(w, N.b[e]);
	walk_cols(w, N.c[e]);
}

/*
 * A sound interval of what a column adds, its index and k within their
 * ranges, a word of a `lanes` parameter or a shared one within its
 * range: 0 when it meets an operation it does not follow or a value
 * that may be negative.
 */
struct civ {
	const struct func *f;
	int islot, kslot, W;
	int64_t ilo, ihi;
};

#define CIV_TOP ((int64_t)1 << 62)

/* [lo, hi] as a value of type t: a wrap is the type's whole range */
static int
civ_fit(ty_t t, int64_t *lo, int64_t *hi)
{
	int w = iwidth(t);

	if (TYK(t) == K_BOOL)
		return *lo >= 0 && *hi <= 1;
	if (TYK(t) == K_UINT && w < 63) {
		if (*hi >= ((int64_t)1 << w)) {
			*lo = 0;
			*hi = ((int64_t)1 << w) - 1;
		}
		return 1;
	}
	/* signed, or 63 and 64 bits: no wrap below CIV_TOP */
	return *hi < CIV_TOP && (w >= 63 || *hi < ((int64_t)1 << (w - 1)));
}

static int
civ_of(const struct civ *c, nid e, int64_t *lo, int64_t *hi, int depth)
{
	int64_t a0, a1, b0, b1;
	int s, op = N.op[e];

	if (depth > 64)
		return 0;
	switch (op) {
	case O_CONST:
		if (!TY_ISINT(N.ty[e]) || N.ki[e] < 0)
			return 0;
		*lo = *hi = N.ki[e];
		return 1;
	case O_VAR:
		s = (int)N.ki[e];
		if (s == c->islot) {
			*lo = c->ilo;
			*hi = c->ihi;
		} else if (s == c->kslot || s == SG_LANE_SLOT) {
			*lo = 0;
			*hi = c->W - 1;
		} else {
			if (s >= SG_LCOPY_SLOT && s < SG_LCOPY_SLOT + c->f->np)
				s -= SG_LCOPY_SLOT;
			else if (s >= c->f->np || !c->f->p[s].uniform)
				return 0;
			if (!TY_ISINT(c->f->p[s].ty))
				return 0;
			param_bounds(&c->f->p[s], lo, hi);
		}
		return *lo >= 0;
	case O_CAST:
		if (!TY_ISINT(N.ty[e]) || !civ_of(c, N.a[e], lo, hi, depth + 1))
			return 0;
		return civ_fit(N.ty[e], lo, hi);
	case O_SEL:
		if (!civ_of(c, N.b[e], &a0, &a1, depth + 1) ||
		    !civ_of(c, N.c[e], &b0, &b1, depth + 1))
			return 0;
		*lo = a0 < b0 ? a0 : b0;
		*hi = a1 > b1 ? a1 : b1;
		return 1;
	case O_SGSHUF:
		return civ_of(c, N.a[e], lo, hi, depth + 1);
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE:
	case O_LAND: case O_LOR: case O_LNOT:
		*lo = 0;
		*hi = 1;
		return 1;
	case O_ADD: case O_MUL: case O_AND: case O_SHR: case O_SHL:
		break;
	default:
		return 0;
	}
	if (!civ_of(c, N.a[e], &a0, &a1, depth + 1) ||
	    !civ_of(c, N.b[e], &b0, &b1, depth + 1))
		return 0;
	switch (op) {
	case O_ADD:
		if (a1 >= CIV_TOP - b1)
			return 0;
		*lo = a0 + b0;
		*hi = a1 + b1;
		break;
	case O_MUL:
		if (a1 && b1 >= CIV_TOP / a1)
			return 0;
		*lo = a0 * b0;
		*hi = a1 * b1;
		break;
	case O_AND:
		*lo = 0;
		*hi = a1 < b1 ? a1 : b1;
		break;
	case O_SHR:
		if (b0 != b1 || b0 > 63)
			return 0;
		*lo = a0 >> b0;
		*hi = a1 >> b0;
		break;
	default:	/* O_SHL */
		if (b0 != b1 || b0 > 61 || a1 >= (CIV_TOP >> b0))
			return 0;
		*lo = a0 << b0;
		*hi = a1 << b0;
	}
	return civ_fit(N.ty[e], lo, hi);
}

/* the column of c with k the lane it is computed in, from 0 */
static nid
col_in_lane(const struct col *c)
{
	struct loopdef d;
	nid map[NSLOT], n;

	loop_get(loop_ent(c->L), &d);
	d.init[0] = mk_const_i(d.sty[0], 0);
	n = mk_loop(loop_intern(&d), 0, N.a[c->L], N.b[c->L], N.c[c->L]);
	memset(map, 0, sizeof map);
	map[c->kslot] = mk_cast(c->kty, sg_lane());
	return subst(n, map, c->kslot + 1);
}

/* the operand-form carry chain over v (a value per lane, its lane the
 * bound one): s = c + v in lane k, c = s >> w, the result lane's s, as
 * the function's limbs (cast to ty, or masked to w bits) */
static nid
ripple(nid v, nid L, int W, int w, ty_t ty, int masked)
{
	ty_t T = N.ty[v];
	nid c = mk_const_i(T, 0), r = mk_const_i(T, 0), s;
	int k;

	for (k = 0; k < W; k++) {
		s = mk_bin(O_ADD, c, mk_sg(O_SGSHUF, v, mk_const_i(TY(K_UINT, 32),
		    k), mk_const_i(TY_LIT, W)));
		r = mk_sel(mk_bin(O_EQ, mk_const_i(TY(K_UINT, 32), k), L), s, r);
		c = mk_bin(O_SHR, s, mk_const_i(T, w));
	}
	if (masked)
		r = mk_bin(O_AND, r, mk_const_i(T, (int64_t)wmask(w)));
	return mk_cast(ty, r);
}

/* lane l's v (operand form) keeping its low w bits and adding the bits
 * above of lane l - 1's */
static nid
csa_step(nid v, int W, int w)
{
	ty_t T = N.ty[v];
	nid L = sg_lane(), one = mk_const_i(TY(K_UINT, 32), 1);

	return mk_bin(O_ADD, mk_bin(O_AND, v, mk_const_i(T, (int64_t)wmask(w))),
	    mk_sel(mk_bin(O_GE, L, one), mk_sg(O_SGSHUF, mk_bin(O_SHR, v,
	    mk_const_i(T, w)), mk_bin(O_SUB, L, one), mk_const_i(TY_LIT, W)),
	    mk_const_i(T, 0)));
}

/* the limbs of u (operand form, below 2^(w+1) - 1 in every lane; own
 * its value in the lane itself) with the carries of the wide sum: one
 * add of two ballots (wideadd) */
static nid
ballot_limbs(nid u, nid own, nid L, int W, int w, ty_t ty, int masked)
{
	ty_t T = N.ty[u], bt = TY(K_UINT, W > 32 ? 64 : 32);
	nid m = mk_const_i(T, (int64_t)wmask(w));
	nid G, P, ks, cin, r;

	G = mk_sg(O_SGBALLOT, mk_bin(O_GT, u, m), mk_const_i(TY_LIT, W), 0);
	P = mk_sg(O_SGBALLOT, mk_bin(O_EQ, u, m), mk_const_i(TY_LIT, W), 0);
	ks = mk_bin(O_XOR, mk_bin(O_ADD, mk_bin(O_SHL, G, mk_const_i(bt, 1)), P),
	    P);
	cin = mk_cast(T, mk_bin(O_AND, mk_bin(O_SHR, ks, mk_cast(bt, L)),
	    mk_const_i(bt, 1)));
	r = mk_bin(O_ADD, own, cin);
	if (masked)
		r = mk_bin(O_AND, r, m);
	return mk_cast(ty, r);
}

/* a and b proved equal on f by SAT (or every point), else why not in
 * why (a counterexample's words are not given: SAT's choice of them
 * changes with the compiler) */
static int
proved_eq(const struct func *f, nid a, nid b, struct sbuf *why)
{
	struct eqproof pr;
	int bad = equiv_decide(f, a, b, NULL, &pr);

	if (!bad && (pr.how == EQ_SAT || pr.how == EQ_EXHAUSTIVE))
		return 1;
	if (bad)
		sb_puts(why, "SAT finds words where they differ, run again by the "
		    "evaluator");
	else
		eqproof_text(why, &pr);
	return 0;
}

/*
 * One carry-save step, x'_k = (x_k & m) + (x_(k-1) >> w) with x below
 * xb and the carry of the chain over x' at most cb: with c = c' +
 * (x_(k-1) >> w), the chain over x and the one over x' keep the same
 * low w bits, (c + x_k) & m == (c' + x'_k) & m, and their next carries
 * keep that difference, (c + x_k) >> w == ((c' + x'_k) >> w) + (x_k >> w).
 * By induction over the lanes from c = c' = 0 (lane 0 adds nothing from
 * below), the two chains give the same limbs.  Checked over the three
 * free words in their ranges.
 */
static int
csa_lemma(const struct func *f, ty_t T, int w, int64_t xb, int64_t cb,
    struct sbuf *why)
{
	struct func g;
	nid cp, xp, x, c, xn, m = mk_const_i(T, (int64_t)wmask(w));
	nid kw = mk_const_i(T, w);
	static const char *nm[3] = {"c", "p", "x"};
	int j;

	memset(&g, 0, sizeof g);
	g.name = f->name;
	g.np = 3;
	for (j = 0; j < 3; j++) {
		g.p[j].name = sym_cstr(nm[j]);
		g.p[j].ty = T;
		g.p[j].ranged = 1;
		g.p[j].hi = j ? xb : cb;
	}
	cp = mk_var(0, g.p[0].name, T);
	xp = mk_var(1, g.p[1].name, T);
	x = mk_var(2, g.p[2].name, T);
	c = mk_bin(O_ADD, cp, mk_bin(O_SHR, xp, kw));
	xn = mk_bin(O_ADD, mk_bin(O_AND, x, m), mk_bin(O_SHR, xp, kw));
	return proved_eq(&g, mk_bin(O_AND, mk_bin(O_ADD, c, x), m),
	    mk_bin(O_AND, mk_bin(O_ADD, cp, xn), m), why) &&
	    proved_eq(&g, mk_bin(O_SHR, mk_bin(O_ADD, c, x), kw),
	    mk_bin(O_ADD, mk_bin(O_SHR, mk_bin(O_ADD, cp, xn), kw),
	    mk_bin(O_SHR, x, kw)), why);
}

#define MAXCSA 6

/* column c's carry-save form, what its proof needs */
struct csaf {
	nid cand, vb;
	ty_t T;
	int w, masked, ns;
	int64_t vmax, xb[MAXCSA + 1];
};

/* the form for column c (its value per lane vb): 0 when the function's
 * limbs or the column's bound are not known */
static int
csa_form(const struct sgf *g, nid cur, const struct col *c, nid vb,
    struct csaf *k)
{
	const struct func *f = g->f;
	struct civ cv;
	nid body = cur, x, map[SG_LANE_SLOT + 1];
	ty_t ty = N.ty[cur];
	int64_t tlo, thi;
	int i, W = sg_lanes_width(f), Tw;

	memset(k, 0, sizeof *k);
	k->vb = vb;
	k->T = N.ty[vb];
	Tw = int_width(k->T);
	if (TYK(ty) != K_UINT || f->np == MAXP || !cclass_available(g->m,
	    C_SG_BALLOT))
		return 0;
	/* the limbs: the result's width, or a mask's under it */
	if (N.op[body] == O_AND && is_const(N.b[body]) && N.ki[N.b[body]] > 0 &&
	    !((uint64_t)(N.ki[N.b[body]] + 1) & (uint64_t)N.ki[N.b[body]])) {
		for (k->w = 1; k->w < 64 && (uint64_t)N.ki[N.b[body]] >> k->w;
		    k->w++)
			;
		k->masked = 1;
	} else
		k->w = int_width(ty);
	memset(&cv, 0, sizeof cv);
	cv.f = f;
	cv.W = W;
	cv.islot = c->islot;
	cv.kslot = c->kslot;
	cv.ilo = c->ilo;
	cv.ihi = c->ihi;
	if (k->w < 1 || k->w >= Tw || !civ_of(&cv, c->inc, &tlo, &thi, 0) ||
	    (thi && c->ihi - c->ilo + 1 > CIV_TOP / thi))
		return 0;
	k->vmax = (c->ihi - c->ilo + 1) * thi;
	if (Tw < 63 && k->vmax >= ((int64_t)1 << Tw))
		return 0;
	/* carry-save steps until a lane's value is at most 2m */
	k->xb[0] = k->vmax;
	while (k->xb[k->ns] > 2 * (int64_t)wmask(k->w) && k->ns < MAXCSA) {
		k->xb[k->ns + 1] = (int64_t)wmask(k->w) + (k->xb[k->ns] >> k->w);
		k->ns++;
	}
	if (k->xb[k->ns] > 2 * (int64_t)wmask(k->w))
		return 0;
	x = vb;
	for (i = 0; i < k->ns; i++)
		x = csa_step(x, W, k->w);
	memset(map, 0, sizeof map);
	map[SG_LANE_SLOT] = lane_param(f);
	k->cand = ballot_limbs(x, subst(x, map, SG_LANE_SLOT + 1),
	    lane_param(f), W, k->w, ty, k->masked);
	return 1;
}

/*
 * The proof, with each lane's column a free word below vmax + 1 (an
 * extra `lanes` parameter): f1, the column form, is the carry chain over
 * them; the carry-save steps keep the chain's limbs; and once each lane
 * is at most 2m the ballots are the chain's carries.
 */
static int
csa_proved(const struct sgf *g, nid f1, const struct csaf *k,
    struct sbuf *why)
{
	const struct func *f = g->f;
	struct func f2 = *f;
	struct param *pz = &f2.p[f2.np];
	nid L = lane_param(f), Zc, f1z;
	ty_t ty = N.ty[f->body];
	int W = sg_lanes_width(f), i;
	int64_t cb, prev;

	memset(pz, 0, sizeof *pz);
	pz->name = sym_cstr("column");
	pz->ty = k->T;
	pz->lanes = 1;
	pz->ranged = 1;
	pz->hi = k->vmax;
	Zc = sg_lanes_copy(f2.np, pz->name, k->T, W);
	f2.np++;
	f1z = sg_replaced(f1, k->vb, Zc);
	if (!f1z || !proved_eq(&f2, f1z, ripple(Zc, L, W, k->w, ty, k->masked),
	    why)) {
		sb_puts(why, " (the function against the carry chain over its "
		    "columns)");
		return 0;
	}
	for (i = 0; i < k->ns; i++) {
		/* the chain over the next values carries at most cb */
		cb = 0;
		do {
			prev = cb;
			cb = (cb + k->xb[i + 1]) >> k->w;
		} while (cb != prev);
		if (!csa_lemma(f, k->T, k->w, k->xb[i], cb, why)) {
			sb_printf(why, " (a carry-save step below %lld)",
			    (long long)k->xb[i] + 1);
			return 0;
		}
	}
	pz->hi = k->xb[k->ns];
	if (!proved_eq(&f2, ripple(Zc, L, W, k->w, ty, k->masked),
	    ballot_limbs(Zc, mk_var((uint32_t)f->np, pz->name, k->T), L, W,
	    k->w, ty, k->masked), why)) {
		sb_puts(why, " (the add of two ballots against the carry chain)");
		return 0;
	}
	return 1;
}

static nid
lanecols(struct sgf *g, nid cur, float *cost)
{
	const struct func *f = g->f;
	struct cwalk cw;
	struct csaf kbest = {0};
	nid L = lane_param(f), best = cur;
	int W = sg_lanes_width(f), i, kind = 0;
	const struct col *cbest = NULL;
	float bc = *cost;

	if (!L || W < 2 || !TY_ISINT(N.ty[cur]) || !cclass_available(g->m,
	    C_SG_SHUF))
		return cur;
	memset(&cw, 0, sizeof cw);
	cw.f = f;
	cw.W = W;
	cw.seen = xcalloc(N.n, 1);
	walk_cols(&cw, cur);
	free(cw.seen);
	/* the derivation is for a full subgroup: a shuffle of an inactive
	 * lane is undefined */
	if (cw.masked)
		return cur;
	for (i = 0; i < cw.n; i++) {
		const struct col *c = &cw.c[i];
		struct loopdef d;
		struct csaf k;
		nid vb, f1;
		float p1, p2 = INFINITY;
		int chosen = 0;
		loop_get(loop_ent(c->L), &d);
		vb = col_in_lane(c);
		f1 = sg_replaced(cur, c->L, mk_bin(O_ADD, mk_cast(d.sty[0],
		    d.init[0]), mk_sg(O_SGSHUF, vb, mk_cast(TY(K_UINT, 32),
		    var_of(c->kslot, c->kname, c->kty)), mk_const_i(TY_LIT, W))));
		if (!f1)
			continue;
		p1 = price(g, f1);
		if (csa_form(g, cur, c, vb, &k)) {
			p2 = price(g, k.cand);
			if (p2 < bc && p2 < p1) {
				struct sbuf why = {0};
				if (csa_proved(g, f1, &k, &why)) {
					bc = p2;
					best = k.cand;
					kbest = k;
					cbest = c;
					kind = 2;
					chosen = 1;
				} else
					sb_printf(&g->notes, "  subgroup forms: the "
					    "carry chain over the columns as carry-save "
					    "steps and one add of two ballots (cost %g) "
					    "is not proved: %s\n", (double)p2,
					    sb_str(&why));
				free(why.p);
			} else if (p2 >= *cost)
				note_dear(g, "the carry chain over the columns as "
				    "carry-save steps and one add of two ballots", p2,
				    *cost);
		}
		if (p1 < bc) {
			bc = p1;
			best = f1;
			cbest = c;
			kind = 1;
		} else if (p1 >= *cost && !chosen)
			note_dear(g, "each column computed once, in its lane, and "
			    "read by shuffles", p1, *cost);
	}
	if (!kind)
		return cur;
	{
		struct loopdef d;
		loop_get(loop_ent(cbest->L), &d);
		sb_printf(&g->why, "%s, a loop that adds to %s (so it is its "
		    "start plus its value from 0), is a function of %s alone, "
		    "the same in every lane: lane j computes it once, from 0, "
		    "with %s = j, and the lanes read it by a shuffle at %s\n",
		    shared_str(cbest->L, 160), sym_str(d.sname[0]),
		    sym_str(cbest->kname), sym_str(cbest->kname),
		    sym_str(cbest->kname));
	}
	if (kind == 2)
		sb_printf(&g->why, "the function, with each lane's column a free "
		    "word below %lld, is the carry chain over the columns (s = c + "
		    "column k, c = s >> %d, the lane's limb the s of its own "
		    "column; checked by SAT), whose limbs %d carry-save step%s "
		    "keep, lane l keeping the low %d bits of its value and adding "
		    "the bits above of lane l - 1's (each step's limb and carry "
		    "checked by SAT over three free words, then induction over "
		    "the lanes); each lane's value is then below %lld, and the "
		    "carries are a wide sum's: with g the lanes whose value passes "
		    "2^%d - 1 and p those where it is 2^%d - 1, the carries into "
		    "the lanes are ((g << 1) + p) xor p, one add of two ballots "
		    "(checked by SAT against the carry chain over free words)\n",
		    (long long)kbest.vmax + 1, kbest.w, kbest.ns, kbest.ns == 1 ?
		    "" : "s", kbest.w, (long long)kbest.xb[kbest.ns] + 1, kbest.w,
		    kbest.w);
	g->proved = best;
	g->bycons = 1;
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
	if (N.op[e] == O_LOOP) {
		/* the operations of a loop's body (a column form's) */
		uint32_t le = loop_ent(e), st = loops.st[le], k;
		for (k = 0; k < loops.nst[le]; k++)
			if ((w = width_of(loops.init[st + k], seen)) != 0 ||
			    (w = width_of(loops.next[st + k], seen)) != 0)
				return w;
		if ((w = width_of(loops.cond[le], seen)) != 0)
			return w;
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
		{O_SGDDX, "derivatives (a fragment shader's quads; "
		    "GL_NV_compute_shader_derivatives in compute)"},
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
		if (N.op[x] == O_LOOP) {
			/* the operations of a loop's body (a column form's) */
			uint32_t le = loop_ent(x), s0 = loops.st[le], k;
			nid in[2 * MAXST + 1];
			int ni = 0;
			for (k = 0; k < loops.nst[le]; k++) {
				in[ni++] = loops.init[s0 + k];
				in[ni++] = loops.next[s0 + k];
			}
			in[ni++] = loops.cond[le];
			while (ni--)
				if (in[ni] && !seen[in[ni]] && (N.fl[in[ni]] &
				    NF_SG)) {
					seen[in[ni]] = 1;
					st[sp++] = in[ni];
				}
		}
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
	has[O_SGDDX] |= has[O_SGDDY] | has[O_SGFWIDTH];
	for (i = 0; i < (int)(sizeof ext / sizeof ext[0]); i++)
		if (has[ext[i].op])
			sb_printf(b, "%s%s", n++ ? ", " : "", ext[i].what);
	free(seen);
	free(st);
}

/* k (1 or 2) when i is the lane with bit k set (or) or cleared (and,
 * clear 1), else 0; *ln the lane */
static int
quad_bit(nid i, int clear, nid *ln)
{
	int64_t m;
	int k;

	while (N.op[i] == O_CAST)
		i = N.a[i];
	if (N.op[i] != (clear ? O_AND : O_OR))
		return 0;
	for (k = 0; k < 2; k++) {
		nid l = k ? N.b[i] : N.a[i], c = k ? N.a[i] : N.b[i];
		if (!sg_is_lane(l) || !is_const(c))
			continue;
		m = clear ? (int64_t)(~(uint32_t)N.ki[c] & 0xffffffffu) :
		    N.ki[c];
		if (m == 1 || m == 2) {
			*ln = l;
			return (int)m;
		}
	}
	return 0;
}

/*
 * A float difference of neighbouring lanes' values, v in lane | k less v
 * in lane & ~k, is a fine derivative (k 1: along x, 2: along y), where
 * the model prices sg_deriv lower (#204)
 */
static nid
derivs(struct sgf *g, nid cur, float *cost)
{
	nid *nd = xmalloc(N.n * sizeof *nd), x, a, b, ln = 0, ln2 = 0, cand;
	uint8_t *seen = xcalloc(N.n, 1);
	uint32_t sp = 0, j;
	float c;
	int k;

	nd[sp++] = cur;
	seen[cur] = 1;
	while (sp) {
		x = nd[--sp];
		for (j = 0; j < 3; j++) {
			nid o = j == 0 ? N.a[x] : j == 1 ? N.b[x] : N.c[x];
			if (N.op[x] == O_CONST || N.op[x] == O_VAR ||
			    N.op[x] == O_TABLE)
				break;
			if (o && o < N.n && !seen[o]) {
				seen[o] = 1;
				nd[sp++] = o;
			}
		}
		if (N.op[x] != O_SUB || TYK(N.ty[x]) != K_FLOAT)
			continue;
		a = N.a[x];
		b = N.b[x];
		if (N.op[a] != O_SGSHUF || N.op[b] != O_SGSHUF ||
		    N.a[a] != N.a[b] || N.c[a] != N.c[b] || N.ki[N.c[a]] % 4)
			continue;
		k = quad_bit(N.b[a], 0, &ln);
		if (!k || quad_bit(N.b[b], 1, &ln2) != k || ln != ln2)
			continue;
		cand = sg_replaced(cur, x, mk_sg(k == 1 ? O_SGDDX : O_SGDDY,
		    N.a[a], ln, N.c[a]));
		if (!cand || (c = price(g, cand)) >= *cost)
			continue;
		sb_printf(&g->why, "%s: %s in the lane with bit %d set less in the "
		    "lane with it clear, the quad's fine derivative along %s: "
		    "sg_%s (sg_deriv)\n", shared_str(x, 80), node_str(N.a[a]),
		    k - 1, k == 1 ? "x" : "y", k == 1 ? "dfdx" : "dfdy");
		cur = cand;
		*cost = c;
	}
	free(nd);
	free(seen);
	return cur;
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
	if (TYK(N.ty[cur]) == K_FLOAT && (N.fl[cur] & NF_SG)) {
		/* of floats only the derivatives: the rest reads integers */
		before = cost = price(&g, cur);
		cur = derivs(&g, cur, &cost);
		if (cur == f->body)
			printf("  no fine derivative here is cheaper under %s (cost "
			    "%g as written)\n", mn, (double)before);
		goto derived;
	}
	if (!TY_ISINT(N.ty[cur]))
		return;
	before = cost = price(&g, cur);
	if (N.fl[cur] & NF_LANES) {
		/* lanes' words: no values to tell a form by, a shape and SAT */
		cur = wideadd(&g, cur, &cost);
		if (cur == f->body)
			cur = lanecols(&g, cur, &cost);
		goto derived;
	}
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
	/* a map by fields stands when it is under the others */
	if (!(N.fl[f->body] & NF_SG)) {
		struct sbuf fw = {0};
		nid fm = fieldmap(&g, f->body, &cost, &fw);
		if (fm != f->body) {
			cur = fm;
			sb_clear(&g.why);
			sb_puts(&g.why, sb_str(&fw));
		}
		free(fw.p);
	}
derived:
	if (g.notes.n)
		fputs(sb_str(&g.notes), stdout);
	if (cur == f->body)
		goto out;
	W = width_in(cur);
	if (cur == g.proved) {
		bad = 0;
		pr = g.pr;
	} else
		bad = equiv_decide(f, f->body, cur, &w, &pr);
	chk = bad ? "contradicted by the check" : g.bycons && cur == g.proved ?
	    "proved by its derivation, each step above" : pr.how ==
	    EQ_EXHAUSTIVE ? "checked on every point of the domain" : pr.how ==
	    EQ_SAT ? "checked by SAT" : "checked on a sample of the domain";
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
	rep_str("check", g.bycons && cur == g.proved ? "derivation" : pr.how ==
	    EQ_EXHAUSTIVE ? "exhaustive" : pr.how == EQ_SAT ? "sat" : "sampled");
	rep_end();
out:
	free(g.why.p);
	free(g.notes.p);
	free(w.p);
	free(nd.p);
}
