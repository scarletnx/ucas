/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * astc.c: steps of the ASTC decoder (Khronos Data Format Specification,
 * "ASTC Compressed Texture Image Formats"), 2D blocks of the LDR
 * profile, as operations of the language.
 *
 * Each one is defined by an expression over the base integer operations
 * (astc_expand), like the catalogue's instructions (inst.c): that
 * expression is its exact semantics, and it is how the operation is
 * evaluated, bounded and bit-blasted, so equiv proves the step a decoder
 * writes against it.  A constant of the operation rides in the node's ki
 * (a node has three operands).  No target has one of them as an
 * instruction (the texture unit decodes a whole block), so an operation
 * is priced as its definition: the operations it spells out over its
 * operands, constants folded.
 *
 *   astc_partition(seed, x, y, n)
 *       the partition, 0..n-1, of texel (x, y) in a block of n
 *       partitions (n 1..4, n = 1 is 0; a count that is not a constant
 *       is a select of the four, undefined for any other) whose
 *       partition index is seed (its low 10 bits, the block's field): the
 *       specification's hash of the seed, its twelve squared nibbles
 *       shifted by the seed's bits, four scores (s1 x + s2 y + p >> 14)
 *       & 63 ... and the highest of the first n, the lowest partition
 *       on a tie.  z is 0 (2D); a block of fewer than 31 texels takes
 *       the hash at (2x, 2y).  x and y are any integers (u32 arithmetic,
 *       of which the scores keep the low 6 bits).
 */
#include "calc.h"
#include "fuse.h"

#include <stdint.h>
#include <stdlib.h>

#define U32 TY(K_UINT, 32)

static nid
ku(uint32_t v)
{
	return mk_const_i(U32, (int64_t)v);
}

static nid
shr(nid x, int k)
{
	return mk_bin(O_SHR, x, ku((uint32_t)k));
}

static nid
shl(nid x, int k)
{
	return mk_bin(O_SHL, x, ku((uint32_t)k));
}

static nid
band(nid x, uint32_t m)
{
	return mk_bin(O_AND, x, ku(m));
}

static nid
bit_set(nid x, uint32_t m)
{
	return mk_bin(O_NE, band(x, m), ku(0));
}

/* the partition of (x, y) among n (2..4) for the 10-bit seed */
static nid
partition(int n, nid seed, nid x, nid y)
{
	nid p, sd[13], b0, b1, b4, v[4], ge01, ge02, ge12;
	int i, k3 = n == 3 ? 6 : 5;

	/* the hash of the seed and the count: seed + 1024 (n - 1) */
	p = mk_bin(O_ADD, band(seed, 1023), ku(1024u * (uint32_t)(n - 1)));
	p = mk_bin(O_XOR, p, shr(p, 15));
	p = mk_bin(O_SUB, p, shl(p, 17));
	p = mk_bin(O_ADD, p, shl(p, 7));
	p = mk_bin(O_ADD, p, shl(p, 4));
	p = mk_bin(O_XOR, p, shr(p, 5));
	p = mk_bin(O_ADD, p, shl(p, 16));
	p = mk_bin(O_XOR, p, shr(p, 7));
	p = mk_bin(O_XOR, p, shr(p, 3));
	p = mk_bin(O_XOR, p, shl(p, 6));
	p = mk_bin(O_XOR, p, shr(p, 17));
	for (i = 1; i <= 8; i++)
		sd[i] = band(shr(p, 4 * (i - 1)), 15);
	sd[9] = band(shr(p, 18), 15);
	sd[10] = band(shr(p, 22), 15);
	sd[11] = band(shr(p, 26), 15);
	sd[12] = band(mk_bin(O_OR, shr(p, 30), shl(p, 2)), 15);
	/* bits 0, 1 and 4 of the seed choose the shifts, which adding a
	 * multiple of 1024 leaves as they are */
	b0 = bit_set(seed, 1);
	b1 = bit_set(seed, 2);
	b4 = bit_set(seed, 16);
	for (i = 1; i <= 12; i++) {
		nid q = mk_bin(O_MUL, sd[i], sd[i]);
		nid b45 = mk_sel(b1, shr(q, 4), shr(q, 5));
		/* sh1: bit 0 ? (bit 1 ? 4 : 5) : (n == 3 ? 6 : 5); sh2 the
		 * other way round; the last four by sh1 when bit 4 is set: a
		 * select of constant shifts, each in range */
		nid s1 = mk_sel(b0, b45, shr(q, k3));
		nid s2 = mk_sel(b0, shr(q, k3), b45);
		sd[i] = i <= 8 ? (i & 1 ? s1 : s2) : mk_sel(b4, s1, s2);
	}
	for (i = 0; i < 4; i++) {
		nid sx = mk_bin(O_MUL, sd[2 * i + 1], x);
		nid sy = mk_bin(O_MUL, sd[2 * i + 2], y);
		v[i] = band(mk_bin(O_ADD, mk_bin(O_ADD, sx, sy), shr(p,
		    14 - 4 * i)), 63);
	}
	ge01 = mk_bin(O_GE, v[0], v[1]);
	if (n == 2)
		return mk_sel(ge01, ku(0), ku(1));
	ge02 = mk_bin(O_GE, v[0], v[2]);
	ge12 = mk_bin(O_GE, v[1], v[2]);
	if (n == 3)
		return mk_sel(mk_bin(O_LAND, ge01, ge02), ku(0), mk_sel(ge12,
		    ku(1), ku(2)));
	return mk_sel(mk_bin(O_LAND, mk_bin(O_LAND, ge01, ge02), mk_bin(O_GE,
	    v[0], v[3])), ku(0), mk_sel(mk_bin(O_LAND, ge12, mk_bin(O_GE, v[1],
	    v[3])), ku(1), mk_sel(mk_bin(O_GE, v[2], v[3]), ku(2), ku(3))));
}

nid
astc_expand(int op, int64_t ki, nid a, nid b, nid c)
{
	switch (op) {
	case O_ASTC_PART:
		return partition((int)ki, a, b, c);
	}
	fail("%s: not an ASTC step", opinfo[op].name);
	return 0;
}

nid
mk_astc(int op, int64_t ki, nid a, nid b, nid c)
{
	const char *nm = opinfo[op].name;
	int fl = NF_SG;
	nid r;

	if (!TY_ISINT(N.ty[a]) || !TY_ISINT(N.ty[b]) || !TY_ISINT(N.ty[c]))
		fail("%s takes integer operands", nm);
	switch (op) {
	case O_ASTC_PART:
		if (ki < 1 || ki > 4)
			fail("astc_partition: a block has 1 to 4 partitions, "
			    "not %lld", (long long)ki);
		break;
	}
	a = mk_cast(U32, a);
	b = mk_cast(U32, b);
	c = mk_cast(U32, c);
	if (N.op[a] == O_UNDEF || N.op[b] == O_UNDEF || N.op[c] == O_UNDEF)
		return mk_undef(U32);
	if (op == O_ASTC_PART && ki == 1)
		return ku(0);
	if (is_const(a) && is_const(b) && is_const(c)) {
		r = astc_expand(op, ki, a, b, c);
		if (is_const(r))
			return r;
	}
	fl |= (N.fl[a] | N.fl[b] | N.fl[c]) & NF_PARTIAL;
	return mk_raw(op, U32, a, b, c, ki, fl);
}

/*
 * astc_partition's call: four operands, the count in the node's ki.  A
 * count that is not a constant (a decoder reads it from the block) is a
 * select of the four counts' nodes, undefined for any other count.
 */
nid
astc_call(int op, const nid *args)
{
	nid n = args[3], r;
	int k;

	if (!TY_ISINT(N.ty[n]))
		fail("astc_partition takes an integer partition count 1..4, not "
		    "%s", node_str(n));
	if (is_const(n))
		return mk_astc(op, N.ki[n], args[0], args[1], args[2]);
	r = mk_undef(U32);
	for (k = 4; k >= 1; k--)
		r = mk_sel(mk_bin(O_EQ, n, mk_const_i(N.ty[n], k)), mk_astc(op, k,
		    args[0], args[1], args[2]), r);
	return r;
}

/*
 * The node's price under m: its definition's operations, over its
 * operands as they are (a constant seed folds the hash), each operand
 * that is not a constant a free value.
 */
float
astc_cost(const struct model *m, nid n)
{
	static const char *const nm[3] = {"_a", "_b", "_c"};
	nid op[3] = {N.a[n], N.b[n], N.c[n]};
	int j;

	for (j = 0; j < 3; j++)
		if (!is_const(op[j]))
			op[j] = mk_var((uint32_t)j, sym_cstr(nm[j]),
			    N.ty[op[j]]);
	return fuse_expr_cost(m, astc_expand(N.op[n], N.ki[n], op[0], op[1],
	    op[2]));
}
