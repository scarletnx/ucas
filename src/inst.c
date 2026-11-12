/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * inst.c: a catalogue of single instructions that targets have and the
 * base operations do not name.
 *
 * Each one is an operation of the language whose value is defined by an
 * expression over the base operations (inst_expand): that expression is
 * its exact semantics, and it is how the operation is evaluated, bounded
 * and bit-blasted (sg_lower spells it out, as it does the subgroup
 * operations).  The node keeps the operation for printing and for the
 * price: one instruction of its cost class.  Like the dot forms, a model
 * that does not list the class cannot use the operation, and no built-in
 * model lists one yet: none is measured.
 *
 *   bfe(x, p, n)       bits p..p+n-1 of x, zero-extended (u32); n = 0 is
 *                      0; undefined when p + n > 32: OpBitFieldUExtract
 *   sbfe(x, p, n)      the same field sign-extended (i32):
 *                      OpBitFieldSExtract
 *   brev(x)            bit i is bit 31 - i of x: OpBitReverse
 *   mulhi(a, b)        the high 32 bits of the 64-bit product of two
 *                      u32; smulhi(a, b) of two i32: Op[US]MulExtended
 *   funl(hi, lo, k)    the high word of (hi:lo) << k, k in 0..31:
 *                      (hi << k) | (lo >> (32 - k)); funl(x, x, k) is x
 *                      rotated left by k
 *   funr(hi, lo, k)    the low word of (hi:lo) >> k, k in 0..31
 *   med3(a, b, c)      the median of three integers: max(min(a, b),
 *                      min(max(a, b), c)); with lo <= hi, med3(x, lo, hi)
 *                      is x clamped to lo..hi
 *   usad4x8(a, b, c)   c plus the sum of |a.byte[i] - b.byte[i]| over the
 *                      four bytes of two u32 (u32, wrapping)
 *   byteperm(a, b, s)  each byte of the result is one of the eight bytes
 *                      of b:a (byte 0 the low byte of a, byte 4 the low
 *                      byte of b): result byte i is chosen by nibble i of
 *                      s, whose low three bits are the byte's number and
 *                      whose top bit, when set, replaces the byte by
 *                      eight copies of its sign bit
 *   bfi(base, ins, p, n)
 *                      base with bits p..p+n-1 replaced by the low n bits
 *                      of ins (u32), p and n constant (a node has three
 *                      operands, so they ride in one, bfi_field):
 *                      OpBitFieldInsert
 *   mulwide(a, b)      the 64-bit product of two u32 (u64)
 *   madwide(a, b, c)   that product plus the u64 c (wrapping): IMAD.WIDE,
 *                      v_mad_u64_u32
 *   bytesel(hi, lo, s) each byte of the result chosen by a byte of s from
 *                      the eight bytes of hi:lo (byte 0 the low byte of
 *                      lo, byte 4 the low byte of hi): selector 0..7 that
 *                      byte, 8..11 eight copies of the sign bit of byte
 *                      1, 3, 5 or 7, 12 the byte 0, 13 and up 255:
 *                      v_perm_b32 (hi its S0, lo its S1)
 *
 * find (inst_find) offers them for the patterns they are, each kept only
 * when the active model prices the whole function lower, and checked on
 * the function's domain.
 */
#include "calc.h"
#include "fuse.h"
#include "sg.h"

#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#define U32 TY(K_UINT, 32)
#define U64 TY(K_UINT, 64)
#define I32 TY(K_SINT, 32)
#define I64 TY(K_SINT, 64)

int
inst_class(int op)
{
	switch (op) {
	case O_BFE: case O_SBFE:
		return C_BFE;
	case O_BREV:
		return C_BREV;
	case O_MULHI: case O_SMULHI:
		return C_MULHI;
	case O_FUNL: case O_FUNR:
		return C_FUNNEL;
	case O_MED3:
		return C_MED3;
	case O_USAD4X8:
		return C_SAD4X8;
	case O_BFI:
		return C_BFI;
	case O_MULWIDE: case O_MADWIDE:
		return C_MULWIDE;
	case O_BYTESEL:
		return C_BYTESEL;
	}
	return C_BYTEPERM;
}

static nid
ku(ty_t t, uint64_t v)
{
	return mk_const_i(t, (int64_t)v);
}

nid
bfi_field(int p, int n)
{
	return mk_const_i(TY_LIT, (int64_t)p | (int64_t)n << 6);
}

/* the mask of bits p..p+n-1 */
static uint32_t
field_mask(int p, int n)
{
	return n == 0 ? 0 : (uint32_t)(0xffffffffu >> (32 - n)) << p;
}

static nid
sum_type(nid a, nid b, nid c)
{
	nid s = mk_bin(O_ADD, mk_bin(O_ADD, a, b), c);

	return s;
}

static ty_t
inst_type(int op, nid a, nid b, nid c)
{
	switch (op) {
	case O_SBFE: case O_SMULHI:
		return I32;
	case O_MED3:
		/* the operands' common type, as their sum would have */
		return N.ty[sum_type(a, b, c)];
	case O_MULWIDE: case O_MADWIDE:
		return U64;
	}
	return U32;
}

/* byte k of w (u32 or u64), as a u32 */
static nid
byte_of(nid w, nid k8)
{
	return mk_cast(U32, mk_bin(O_AND, mk_bin(O_SHR, w, k8), ku(N.ty[w],
	    255)));
}

nid
inst_expand(int op, nid a, nid b, nid c)
{
	ty_t t = inst_type(op, a, b, c);
	nid r = 0, x, w, ok;
	int i;

	switch (op) {
	case O_BFE:
		/* in 64 bits, so a field of all 32 bits has its mask */
		x = mk_cast(U64, a);
		r = mk_bin(O_AND, mk_bin(O_SHR, x, mk_cast(U64, b)), mk_bin(O_SUB,
		    mk_bin(O_SHL, ku(U64, 1), mk_cast(U64, c)), ku(U64, 1)));
		ok = mk_bin(O_LE, mk_bin(O_ADD, mk_cast(U64, b), mk_cast(U64, c)),
		    ku(U64, 32));
		r = mk_sel(ok, mk_cast(t, r), mk_undef(t));
		break;
	case O_SBFE:
		/* the field, then copies of its top bit above it: all in
		 * unsigned 64 bits, the low 32 read as i32 */
		x = mk_cast(U64, a);
		w = mk_bin(O_SUB, mk_bin(O_SHL, ku(U64, 1), mk_cast(U64, c)),
		    ku(U64, 1));
		r = mk_bin(O_AND, mk_bin(O_SHR, x, mk_cast(U64, b)), w);
		r = mk_bin(O_OR, r, mk_bin(O_SHL, mk_bin(O_SUB, ku(U64, 0),
		    mk_bin(O_AND, mk_bin(O_SHR, r, mk_bin(O_SUB, mk_cast(U64, c),
		    ku(U64, 1))), ku(U64, 1))), mk_cast(U64, c)));
		ok = mk_bin(O_LE, mk_bin(O_ADD, mk_cast(U64, b), mk_cast(U64, c)),
		    ku(U64, 32));
		r = mk_sel(mk_bin(O_EQ, mk_cast(U64, c), ku(U64, 0)), ku(t, 0),
		    mk_sel(ok, mk_cast(t, mk_cast(U32, r)), mk_undef(t)));
		break;
	case O_BREV: {
		static const uint32_t m[5] = {0x55555555u, 0x33333333u,
		    0x0f0f0f0fu, 0x00ff00ffu, 0x0000ffffu};
		r = a;
		for (i = 0; i < 5; i++)
			r = mk_bin(O_OR, mk_bin(O_AND, mk_bin(O_SHR, r, ku(U32,
			    1u << i)), ku(U32, m[i])), mk_bin(O_SHL, mk_bin(O_AND,
			    r, ku(U32, m[i])), ku(U32, 1u << i)));
		break;
	}
	case O_MULHI:
		r = mk_bin(O_SHR, mk_bin(O_MUL, mk_cast(U64, a), mk_cast(U64, b)),
		    ku(U64, 32));
		break;
	case O_SMULHI:
		r = mk_bin(O_SHR, mk_bin(O_MUL, mk_cast(I64, a), mk_cast(I64, b)),
		    ku(I64, 32));
		break;
	case O_FUNL: case O_FUNR:
		w = mk_bin(O_OR, mk_bin(O_SHL, mk_cast(U64, a), ku(U64, 32)),
		    mk_cast(U64, b));
		r = op == O_FUNL ? mk_bin(O_SHR, mk_bin(O_SHL, w, mk_cast(U64, c)),
		    ku(U64, 32)) : mk_bin(O_SHR, w, mk_cast(U64, c));
		r = mk_sel(mk_bin(O_LT, mk_cast(U64, c), ku(U64, 32)),
		    mk_cast(t, r), mk_undef(t));
		break;
	case O_MED3:
		r = mk_bin(O_MAX, mk_bin(O_MIN, a, b), mk_bin(O_MIN,
		    mk_bin(O_MAX, a, b), c));
		break;
	case O_USAD4X8:
		r = c;
		for (i = 0; i < 4; i++) {
			nid p = byte_of(a, ku(U32, 8 * i));
			nid q = byte_of(b, ku(U32, 8 * i));
			r = mk_bin(O_ADD, r, mk_bin(O_SUB, mk_bin(O_MAX, p, q),
			    mk_bin(O_MIN, p, q)));
		}
		break;
	case O_MULWIDE: case O_MADWIDE:
		r = mk_bin(O_MUL, mk_cast(U64, a), mk_cast(U64, b));
		if (op == O_MADWIDE)
			r = mk_bin(O_ADD, r, c);
		break;
	case O_BFI: {
		uint32_t m = field_mask(BFI_P(N.ki[c]), BFI_N(N.ki[c]));
		r = mk_bin(O_OR, mk_bin(O_AND, a, ku(U32, ~m)), mk_bin(O_AND,
		    mk_bin(O_SHL, b, ku(U32, (uint64_t)BFI_P(N.ki[c]) & 31)),
		    ku(U32, m)));
		break;
	}
	case O_BYTEPERM:
		w = mk_bin(O_OR, mk_bin(O_SHL, mk_cast(U64, b), ku(U64, 32)),
		    mk_cast(U64, a));
		r = ku(U32, 0);
		for (i = 0; i < 4; i++) {
			nid s = mk_bin(O_AND, mk_bin(O_SHR, c, ku(U32, 4 * i)),
			    ku(U32, 15));
			nid by = byte_of(w, mk_cast(U64, mk_bin(O_SHL, mk_bin(O_AND,
			    s, ku(U32, 7)), ku(U32, 3))));
			by = mk_sel(mk_bin(O_NE, mk_bin(O_AND, s, ku(U32, 8)),
			    ku(U32, 0)), mk_sel(mk_bin(O_NE, mk_bin(O_AND, by,
			    ku(U32, 128)), ku(U32, 0)), ku(U32, 255), ku(U32, 0)),
			    by);
			r = mk_bin(O_OR, r, mk_bin(O_SHL, by, ku(U32, 8 * i)));
		}
		break;
	case O_BYTESEL:
		w = mk_bin(O_OR, mk_bin(O_SHL, mk_cast(U64, a), ku(U64, 32)),
		    mk_cast(U64, b));
		r = ku(U32, 0);
		for (i = 0; i < 4; i++) {
			nid s = mk_bin(O_AND, mk_bin(O_SHR, c, ku(U32, 8 * i)),
			    ku(U32, 255));
			/* the byte's number masked to 0..7, the sign's (byte 2k + 1)
			 * to 0..3: every shift in range whichever is taken */
			nid by = byte_of(w, mk_cast(U64, mk_bin(O_SHL, mk_bin(O_AND,
			    s, ku(U32, 7)), ku(U32, 3))));
			nid sg = mk_bin(O_AND, mk_bin(O_SHR, w, mk_cast(U64,
			    mk_bin(O_ADD, mk_bin(O_SHL, mk_bin(O_AND, s, ku(U32, 3)),
			    ku(U32, 4)), ku(U32, 15)))), ku(U64, 1));
			sg = mk_sel(mk_bin(O_NE, sg, ku(U64, 0)), ku(U32, 255),
			    ku(U32, 0));
			by = mk_sel(mk_bin(O_GE, s, ku(U32, 13)), ku(U32, 255),
			    mk_sel(mk_bin(O_EQ, s, ku(U32, 12)), ku(U32, 0),
			    mk_sel(mk_bin(O_GE, s, ku(U32, 8)), sg, by)));
			r = mk_bin(O_OR, r, mk_bin(O_SHL, by, ku(U32, 8 * i)));
		}
		break;
	}
	return mk_cast(t, r);
}

nid
mk_inst(int op, nid a, nid b, nid c)
{
	int ar = opinfo[op].arity, fl = NF_SG;
	const char *nm = opinfo[op].name;
	ty_t t;
	nid r;

	if (!TY_ISINT(N.ty[a]) || (ar >= 2 && !TY_ISINT(N.ty[b])) ||
	    (ar == 3 && !TY_ISINT(N.ty[c])))
		fail("%s takes integer operands", nm);
	switch (op) {
	case O_BFE: case O_SBFE: case O_BREV:
		a = mk_cast(U32, a);
		break;
	case O_MULHI: case O_USAD4X8: case O_BYTEPERM: case O_BYTESEL:
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		if (ar == 3)
			c = mk_cast(U32, c);
		break;
	case O_MULWIDE: case O_MADWIDE:
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		if (op == O_MADWIDE)
			c = mk_cast(U64, c);
		break;
	case O_BFI:
		if (!is_const(c) || BFI_P(N.ki[c]) + BFI_N(N.ki[c]) > 32 ||
		    N.ki[c] < 0)
			fail("bfi: its field is a constant offset and count of a "
			    "32-bit word, not %s", node_str(c));
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		break;
	case O_SMULHI:
		a = mk_cast(I32, a);
		b = mk_cast(I32, b);
		break;
	case O_FUNL: case O_FUNR:
		a = mk_cast(U32, a);
		b = mk_cast(U32, b);
		break;
	case O_MED3:
		t = inst_type(op, a, b, c);
		if (t == TY_BOOL)
			fail("med3 takes numbers");
		a = mk_cast(t, a);
		b = mk_cast(t, b);
		c = mk_cast(t, c);
		break;
	}
	t = inst_type(op, a, b, c);
	if ((op == O_BFE || op == O_SBFE) && is_const(b) && is_const(c) &&
	    (N.ki[b] < 0 || N.ki[c] < 0 || N.ki[b] + N.ki[c] > 32))
		fail("%s: bits %lld..%lld are not a field of a 32-bit word", nm,
		    (long long)N.ki[b], (long long)(N.ki[b] + N.ki[c] - 1));
	if ((op == O_FUNL || op == O_FUNR) && is_const(c) && (N.ki[c] < 0 ||
	    N.ki[c] > 31))
		fail("%s: a shift of %lld is not in 0..31", nm, (long long)N.ki[c]);
	if (N.op[a] == O_UNDEF || (ar >= 2 && N.op[b] == O_UNDEF) ||
	    (ar == 3 && N.op[c] == O_UNDEF))
		return mk_undef(t);
	if (is_const(a) && (ar < 2 || is_const(b)) && (ar < 3 || is_const(c))) {
		r = inst_expand(op, a, b, c);
		if (is_const(r) || N.op[r] == O_UNDEF)
			return r;
	}
	fl |= (N.fl[a] | (ar >= 2 ? N.fl[b] : 0) | (ar == 3 ? N.fl[c] : 0)) &
	    NF_PARTIAL;
	if ((op == O_BFE || op == O_SBFE) && !(is_const(b) && is_const(c)))
		fl |= NF_PARTIAL;
	if ((op == O_FUNL || op == O_FUNR) && !is_const(c))
		fl |= NF_PARTIAL;
	return mk_raw(op, t, a, ar >= 2 ? b : 0, ar == 3 ? c : 0, 0, fl);
}

/* ---- find: the instructions for the patterns they are ---- */

/*
 * Each pattern below is what one instruction computes, written with the
 * base operations.  Where the active model prices the instruction, the
 * pattern is offered as that instruction, kept when the whole function
 * gets cheaper, and the result is checked against the function on its
 * domain (the instruction's definition is the pattern, so the check is
 * usually the same circuit twice).  Nothing here knows a target.
 */

static int
klog2(int64_t v)	/* v + 1 a power of two: its exponent, else -1 */
{
	int n = 0;

	if (v <= 0 || (v & (v + 1)))
		return -1;
	while (v) {
		v >>= 1;
		n++;
	}
	return n;
}

static nid
kint(int64_t v)
{
	return mk_const_i(TY_LIT, v);
}

static int
cshift(nid n, int op, ty_t t, nid *x, int *k)
{
	if (N.op[n] != op || N.ty[n] != t || !is_const(N.b[n]) ||
	    N.ki[N.b[n]] < 0 || N.ki[N.b[n]] > 31)
		return 0;
	*x = N.a[n];
	*k = (int)N.ki[N.b[n]];
	return 1;
}

/* the constant operand of an and, and the other one */
static int
and_const(nid n, nid *other, uint32_t *k)
{
	if (N.op[n] != O_AND || N.ty[n] != U32)
		return 0;
	if (is_const(N.b[n])) {
		*k = (uint32_t)N.ki[N.b[n]];
		*other = N.a[n];
		return 1;
	}
	if (is_const(N.a[n])) {
		*k = (uint32_t)N.ki[N.a[n]];
		*other = N.b[n];
		return 1;
	}
	return 0;
}

/*
 * r as the field part of a bit field insert at bits p..p+n-1 (mask m):
 * (ins << p) & m, (ins & (2^n - 1)) << p, or ins << p (ins itself when
 * p is 0) where ins's range keeps it in n bits; the inserted value, or 0
 */
static nid
bfi_ins(const struct func *f, nid r, int p, int n, uint32_t m,
    struct sbuf *how)
{
	nid t, ins;
	uint32_t k;
	int s;
	int64_t lo, hi;

	if (and_const(r, &t, &k) && k == m && (p == 0 || (cshift(t, O_SHL, U32,
	    &ins, &s) && s == p)))
		return p == 0 ? t : ins;
	if (p > 0 && cshift(r, O_SHL, U32, &t, &s) && s == p) {
		if (and_const(t, &ins, &k) && k == (m >> p))
			return ins;
		r = t;
	} else if (p > 0)
		return 0;
	if (sg_range_of(f, r, 32, &lo, &hi) && lo >= 0 && hi <= (int64_t)(m >>
	    p)) {
		sb_printf(how, " (%s <= %lld, by intervals)", node_str(r),
		    (long long)hi);
		return r;
	}
	return 0;
}

/* q, a u64, as the u32 it widens: a word or narrower unsigned widened, a
 * constant below 2^32, or a value whose range stays below it; 0 if not */
static nid
narrow32(const struct func *f, nid q, struct sbuf *how)
{
	int64_t lo, hi;

	if (N.op[q] == O_CAST && TYK(N.ty[N.a[q]]) == K_UINT &&
	    TYW(N.ty[N.a[q]]) <= 32)
		return mk_cast(U32, N.a[q]);
	if (is_const(q))
		return (uint64_t)N.ki[q] <= 0xffffffffULL ? ku(U32,
		    (uint64_t)N.ki[q]) : 0;
	if (sg_range_of(f, q, 32, &lo, &hi) && lo >= 0 && hi <= 0xffffffffLL) {
		sb_printf(how, " (%s <= %lld, by intervals)", node_str(q),
		    (long long)hi);
		return mk_cast(U32, q);
	}
	return 0;
}

/* x, a u64 product of two words: *a and *b those words */
static int
wide_product(const struct func *f, nid x, nid *a, nid *b, struct sbuf *how)
{
	if (N.op[x] == O_MULWIDE) {
		*a = N.a[x];
		*b = N.b[x];
		return 1;
	}
	if (N.op[x] != O_MUL || N.ty[x] != U64)
		return 0;
	return (*a = narrow32(f, N.a[x], how)) != 0 &&
	    (*b = narrow32(f, N.b[x], how)) != 0;
}

/* x, a word's field replaced, as bfi: (base & ~m) | field, with or, xor
 * or +, m the field's bits */
static nid
bfi_of(const struct func *f, nid x, struct sbuf *why)
{
	int j, n, q, op = N.op[x];

	if ((op != O_OR && op != O_XOR && op != O_ADD) || N.ty[x] != U32)
		return 0;
	for (j = 0; j < 2; j++) {
		nid l = j ? N.b[x] : N.a[x], r = j ? N.a[x] : N.b[x];
		nid base, ins;
		uint32_t kk, m;
		struct sbuf how = {0};
		if (!and_const(l, &base, &kk) || (m = ~kk) == 0 ||
		    m == 0xffffffffu)
			continue;
		for (q = 0; !((m >> q) & 1); q++)
			;
		for (n = 0; q + n < 32 && ((m >> (q + n)) & 1); n++)
			;
		if (m != field_mask(q, n) ||
		    (ins = bfi_ins(f, r, q, n, m, &how)) == 0) {
			free(how.p);
			continue;
		}
		sb_printf(why, "%s: bits %d..%d of %s replaced by the low %d "
		    "bits of %s%s", shared_str(x, 80), q, q + n - 1,
		    node_str(base), n, node_str(ins), sb_str(&how));
		free(how.p);
		return mk_inst(O_BFI, base, ins, bfi_field(q, n));
	}
	return 0;
}

/* x, a wide product plus a u64, as madwide */
static nid
madwide_of(const struct func *f, nid x, struct sbuf *why)
{
	nid a, b;
	int j;

	if (N.op[x] != O_ADD || N.ty[x] != U64)
		return 0;
	for (j = 0; j < 2; j++) {
		nid pr = j ? N.b[x] : N.a[x], c = j ? N.a[x] : N.b[x];
		struct sbuf how = {0};
		if (wide_product(f, pr, &a, &b, &how)) {
			sb_printf(why, "%s: the 64-bit product of two words "
			    "plus %s%s", shared_str(x, 80), node_str(c),
			    sb_str(&how));
			free(how.p);
			return mk_inst(O_MADWIDE, a, b, c);
		}
		free(how.p);
	}
	return 0;
}

/* n without its conversions */
static nid
bare(nid n)
{
	while (N.op[n] == O_CAST && TY_ISINT(N.ty[n]))
		n = N.a[n];
	return n;
}

/* p as byte i of a u32 word w: (w >> 8i) & 255, w & 255, w >> 24 or
 * bfe(w, 8i, 8) */
static int
byte_field(nid p, nid *w, int *i)
{
	nid t;
	uint32_t k;
	int s;

	p = bare(p);
	if (N.op[p] == O_BFE && is_const(N.b[p]) && is_const(N.c[p]) &&
	    N.ki[N.c[p]] == 8 && N.ki[N.b[p]] % 8 == 0) {
		*w = N.a[p];
		*i = (int)N.ki[N.b[p]] / 8;
		return 1;
	}
	if (cshift(p, O_SHR, U32, &t, &s) && s == 24) {
		*w = t;
		*i = 3;
		return 1;
	}
	if (!and_const(p, &t, &k) || k != 255)
		return 0;
	if (cshift(t, O_SHR, U32, w, &s) && s % 8 == 0 && s <= 24) {
		*i = s / 8;
		return 1;
	}
	if (N.ty[t] != U32)
		return 0;
	*w = t;
	*i = 0;
	return 1;
}

/* t as |a.byte[i] - b.byte[i]|: max(p, q) - min(p, q) or abs(p - q) */
static int
byte_absdiff(nid t, nid *a, nid *b, int *i)
{
	nid p, q, x, y;
	int i2;

	t = bare(t);
	if (N.op[t] == O_SUB && N.op[N.a[t]] == O_MAX && N.op[N.b[t]] ==
	    O_MIN) {
		x = N.a[t];
		y = N.b[t];
		if (!((N.a[x] == N.a[y] && N.b[x] == N.b[y]) ||
		    (N.a[x] == N.b[y] && N.b[x] == N.a[y])))
			return 0;
		p = N.a[x];
		q = N.b[x];
	} else if (N.op[t] == O_ABS && N.op[bare(N.a[t])] == O_SUB) {
		p = N.a[bare(N.a[t])];
		q = N.b[bare(N.a[t])];
	} else
		return 0;
	return byte_field(p, a, i) && byte_field(q, b, &i2) && i2 == *i;
}

#define MAXTERM 16

static int
sum_terms(nid x, ty_t t, nid *out, int n)
{
	if (N.op[x] == O_ADD && N.ty[x] == t) {
		n = sum_terms(N.a[x], t, out, n);
		return n < 0 ? n : sum_terms(N.b[x], t, out, n);
	}
	if (n == MAXTERM)
		return -1;
	out[n++] = x;
	return n;
}

/* x, a sum with the four bytes' differences of two words in it: usad4x8,
 * the rest of the sum its accumulator */
static nid
usad_of(nid x, struct sbuf *why)
{
	nid tm[MAXTERM], a, b, a0 = 0, b0 = 0, c = 0;
	int n, k, i, seen = 0;
	uint8_t used[MAXTERM] = {0};

	if (N.op[x] != O_ADD || !TY_ISINT(N.ty[x]) || N.ty[x] == TY_BOOL ||
	    (n = sum_terms(x, N.ty[x], tm, 0)) < 4)
		return 0;
	for (k = 0; k < n; k++) {
		if (!byte_absdiff(tm[k], &a, &b, &i))
			continue;
		if (!a0) {
			a0 = a;
			b0 = b;
		}
		if (!((a == a0 && b == b0) || (a == b0 && b == a0)) ||
		    ((seen >> i) & 1))
			continue;
		seen |= 1 << i;
		used[k] = 1;
	}
	if (seen != 15)
		return 0;
	for (k = 0; k < n; k++)
		if (!used[k])
			c = c ? mk_bin(O_ADD, c, tm[k]) : tm[k];
	sb_printf(why, "%s: the sum of the four bytes' absolute differences "
	    "of %s and %s%s%s", shared_str(x, 80), node_str(a0), node_str(b0),
	    c ? ", plus " : "", c ? node_str(c) : "");
	return mk_cast(N.ty[x], mk_inst(O_USAD4X8, a0, b0, c ? c : ku(U32, 0)));
}

/*
 * x, a u32 of one or two u32 words and constants, told byte by byte at
 * random points of the words, then checked with the function: what each
 * result byte can be among byteperm's choices (bp: bit j for byte j of
 * b:a, j + 8 for that byte's sign; with one word b is 0) and bytesel's
 * (bs: byte j of hi:lo, 8 + k the sign of byte 2k + 1, 12 the byte 0, 13
 * the byte 255; with one word hi is the word again).  For what the GF(2)
 * matrix of one word does not say: two words, a byte's sign, a constant
 * byte.  0 when x is not of that kind.
 */
#define BP_SAMPLES 48

static int
byte_choices(const struct func *f, nid x, int *slot, int *ns, uint16_t *bp,
    uint16_t *bs)
{
	uint64_t sl = node_slots(x), rs = 0x9e3779b97f4a7c15ULL;
	int j, i, smp;
	struct plan pl;
	int64_t *in_i;
	double *in_r;
	nid roots[1] = {x};
	uint32_t loc;

	*ns = 0;
	if (N.ty[x] != U32 || N.op[x] == O_VAR || IS_INST(N.op[x]) ||
	    (sl >> MAXP) || (N.fl[x] & NF_SG))
		return 0;
	for (j = 0; j < f->np; j++)
		if ((sl >> j) & 1) {
			if (*ns == 2 || f->p[j].ty != U32)
				return 0;
			slot[(*ns)++] = j;
		}
	if (*ns == 0)
		return 0;
	for (i = 0; i < 4; i++)
		bp[i] = bs[i] = 0xffff;
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	for (smp = 0; smp < BP_SAMPLES; smp++)
		for (j = 0; j < *ns; j++) {
			rs ^= rs << 13;
			rs ^= rs >> 7;
			rs ^= rs << 17;
			in_i[(size_t)slot[j] * CH + smp] = (int64_t)(uint32_t)rs;
		}
	plan_build(&pl, roots, 1, 0);
	loc = plan_loc(&pl, x);
	plan_run(&pl, BP_SAMPLES, in_i, in_r);
	for (smp = 0; smp < BP_SAMPLES; smp++) {
		uint32_t r = (uint32_t)COLI(&pl, loc)[smp];
		uint32_t lo = (uint32_t)in_i[(size_t)slot[0] * CH + smp];
		uint64_t ab = lo | (*ns == 2 ? (uint64_t)(uint32_t)in_i[
		    (size_t)slot[1] * CH + smp] << 32 : 0);
		uint64_t hl = *ns == 2 ? ab : lo | (uint64_t)lo << 32;
		if (COLF(&pl, loc)[smp] & F_UNDEF) {
			bp[0] = bs[0] = 0;
			break;
		}
		for (i = 0; i < 4; i++) {
			uint32_t rb = (r >> (8 * i)) & 255;
			for (j = 0; j < 16; j++) {
				uint32_t by = (uint32_t)(ab >> (8 * (j & 7))) & 255;
				if (j & 8)
					by = (by & 128) ? 255 : 0;
				if (rb != by)
					bp[i] &= (uint16_t)~(1u << j);
			}
			for (j = 0; j < 14; j++) {
				uint32_t by = j < 8 ? (uint32_t)(hl >> (8 * j)) & 255 :
				    j < 12 ? ((hl >> (16 * (j - 8) + 15)) & 1) * 255 :
				    j == 12 ? 0 : 255;
				if (rb != by)
					bs[i] &= (uint16_t)~(1u << j);
			}
		}
	}
	plan_free(&pl);
	free(in_i);
	free(in_r);
	for (i = 0; i < 4; i++)
		bs[i] &= 0x3fff;
	return 1;
}

/* x as byteperm(a, b, s) */
static nid
byteperm_of(const struct func *f, nid x, struct sbuf *why)
{
	int slot[2] = {-1, -1}, ns, j, i, nib[4];
	uint16_t cand[4], bs[4];
	nid w[2];
	uint32_t sel = 0;

	if (!byte_choices(f, x, slot, &ns, cand, bs))
		return 0;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 16 && !((cand[i] >> j) & 1); j++)
			;
		if (j == 16)
			return 0;
		nib[i] = j;
		sel |= (uint32_t)j << (4 * i);
	}
	/* one word moving its whole bytes is the GF(2) pattern's */
	if (ns == 1 && nib[0] < 4 && nib[1] < 4 && nib[2] < 4 && nib[3] < 4)
		return 0;
	w[0] = mk_var((uint32_t)slot[0], f->p[slot[0]].name, U32);
	w[1] = ns == 2 ? mk_var((uint32_t)slot[1], f->p[slot[1]].name, U32) :
	    ku(U32, 0);
	sb_printf(why, "%s: bytes %x, %x, %x, %x of %s:%s (8 and up: a byte's "
	    "sign), told at %d random points", shared_str(x, 80), nib[0],
	    nib[1], nib[2], nib[3], node_str(w[1]), node_str(w[0]),
	    BP_SAMPLES);
	return mk_inst(O_BYTEPERM, w[0], w[1], ku(U32, sel));
}

/* x as bytesel(hi, lo, s): whole bytes of one word moved too, which only
 * byteperm's GF(2) pattern offers otherwise */
static nid
bytesel_of(const struct func *f, nid x, struct sbuf *why)
{
	int slot[2] = {-1, -1}, ns, j, i, by[4];
	uint16_t bp[4], cand[4];
	nid lo, hi;
	uint32_t sel = 0;

	if (!byte_choices(f, x, slot, &ns, bp, cand))
		return 0;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 14 && !((cand[i] >> j) & 1); j++)
			;
		if (j == 14)
			return 0;
		by[i] = j;
		sel |= (uint32_t)j << (8 * i);
	}
	if (ns == 1 && sel == 0x03020100u)
		return 0;	/* the word itself */
	lo = mk_var((uint32_t)slot[0], f->p[slot[0]].name, U32);
	hi = ns == 2 ? mk_var((uint32_t)slot[1], f->p[slot[1]].name, U32) : lo;
	sb_printf(why, "%s: selectors %d, %d, %d, %d over %s:%s (8 to 11: the "
	    "sign of byte 1, 3, 5, 7; 12: 0; 13: 255), told at %d random "
	    "points", shared_str(x, 80), by[0], by[1], by[2], by[3],
	    node_str(hi), node_str(lo), BP_SAMPLES);
	return mk_inst(O_BYTESEL, hi, lo, ku(U32, sel));
}

/* x, a word of a 64-bit shift of hi:lo: u32((hi:lo) >> k) is funr,
 * u32(((hi:lo) << k) >> 32) funl, k in 1..31 */
static nid
funnel64_of(nid x, struct sbuf *why)
{
	nid s, w, hi, lo, sh;
	int k, j, left = 0;

	if (N.op[x] != O_CAST || N.ty[x] != U32)
		return 0;
	s = N.a[x];
	if (N.op[s] != O_SHR || N.ty[s] != U64 || !is_const(N.b[s]))
		return 0;
	k = (int)N.ki[N.b[s]];
	w = N.a[s];
	if (k == 32 && N.op[w] == O_SHL && N.ty[w] == U64 &&
	    is_const(N.b[w]) && N.ki[N.b[w]] >= 1 && N.ki[N.b[w]] <= 31) {
		left = 1;
		k = (int)N.ki[N.b[w]];
		w = N.a[w];
	} else if (k < 1 || k > 31)
		return 0;
	if (N.op[w] != O_OR && N.op[w] != O_ADD && N.op[w] != O_XOR)
		return 0;
	for (j = 0; j < 2; j++) {
		sh = j ? N.b[w] : N.a[w];
		lo = j ? N.a[w] : N.b[w];
		if (N.op[sh] != O_SHL || !is_const(N.b[sh]) ||
		    N.ki[N.b[sh]] != 32)
			continue;
		hi = N.a[sh];
		if (N.op[hi] != O_CAST || N.op[lo] != O_CAST ||
		    TYK(N.ty[N.a[hi]]) != K_UINT || TYW(N.ty[N.a[hi]]) > 32 ||
		    TYK(N.ty[N.a[lo]]) != K_UINT || TYW(N.ty[N.a[lo]]) > 32)
			continue;
		hi = mk_cast(U32, N.a[hi]);
		lo = mk_cast(U32, N.a[lo]);
		sb_printf(why, "%s: the %s word of the 64-bit (%s:%s) %s %d",
		    shared_str(x, 80), left ? "high" : "low", node_str(hi),
		    node_str(lo), left ? "<<" : ">>", k);
		return mk_inst(left ? O_FUNL : O_FUNR, hi, lo, kint(k));
	}
	return 0;
}

/* x as the instruction of pattern p; 0 when it is not that pattern */
static nid
pattern(const struct func *f, int p, nid x, struct sbuf *why)
{
	nid a, b, s, m;
	int k, j, op = N.op[x];
	int64_t lo, hi, lo2, hi2;

	switch (p) {
	case 0:		/* (w >> k) & (2^n - 1); (w << j) >> k */
		if (op == O_AND && N.ty[x] == U32) {
			m = is_const(N.b[x]) ? N.b[x] : N.a[x];
			s = m == N.b[x] ? N.a[x] : N.b[x];
			if (is_const(m) && (j = klog2(N.ki[m])) > 0 &&
			    cshift(s, O_SHR, U32, &a, &k) && k + j <= 32 && k) {
				sb_printf(why, "%s: bits %d..%d of %s", shared_str(x, 80), k,
				    k + j - 1, shared_str(a, 40));
				return mk_inst(O_BFE, a, kint(k), kint(j));
			}
		}
		if (cshift(x, O_SHR, U32, &s, &k) && cshift(s, O_SHL, U32, &a,
		    &j) && k >= j && j) {
			sb_printf(why, "%s: bits %d..%d of %s", shared_str(x, 80), k - j,
			    31 - j, node_str(a));
			return mk_inst(O_BFE, a, kint(k - j), kint(32 - k));
		}
		break;
	case 1:		/* i32: (w << j) >> k keeps the field's sign */
		if (cshift(x, O_SHR, I32, &s, &k) && cshift(s, O_SHL, I32, &a,
		    &j) && k >= j && j) {
			sb_printf(why, "%s: bits %d..%d of %s with their sign",
			    shared_str(x, 80), k - j, 31 - j, node_str(a));
			return mk_inst(O_SBFE, a, kint(k - j), kint(32 - k));
		}
		break;
	case 2:		/* (a << k) | (b >> (32 - k)); or, xor or + alike */
		if ((op == O_OR || op == O_XOR || op == O_ADD) && N.ty[x] == U32)
			for (j = 0; j < 2; j++) {
				nid l = j ? N.b[x] : N.a[x], r = j ? N.a[x] : N.b[x];
				int kl, kr;
				if (cshift(l, O_SHL, U32, &a, &kl) && cshift(r, O_SHR,
				    U32, &b, &kr) && kl + kr == 32 && kl) {
					sb_printf(why, "%s: the high word of (%s:%s) "
					    "<< %d (the two fields share no bit)",
					    shared_str(x, 80), node_str(a), node_str(b), kl);
					return mk_inst(O_FUNL, a, b, kint(kl));
				}
			}
		break;
	case 3:		/* u32((u64(a) * u64(b)) >> 32), and the signed one */
		if (op == O_CAST && (N.ty[x] == U32 || N.ty[x] == I32)) {
			ty_t wide = N.ty[x] == U32 ? U64 : I64;
			s = N.a[x];
			if (N.op[s] != O_SHR || N.ty[s] != wide || !is_const(N.b[s]) ||
			    N.ki[N.b[s]] < 32 || N.ki[N.b[s]] > 63 ||
			    N.op[N.a[s]] != O_MUL)
				break;
			k = (int)N.ki[N.b[s]] - 32;
			a = N.a[N.a[s]];
			b = N.b[N.a[s]];
			for (j = 0; j < 2; j++) {
				nid *q = j ? &b : &a;
				if (is_const(*q) && (wide == U64 ? (uint64_t)N.ki[*q] <=
				    0xffffffffULL : N.ki[*q] >= INT32_MIN &&
				    N.ki[*q] <= INT32_MAX))
					*q = mk_const_i(N.ty[x], N.ki[*q]);
				else if (N.op[*q] == O_CAST && N.ty[N.a[*q]] == N.ty[x])
					*q = N.a[*q];
				else
					return 0;
			}
			sb_printf(why, "%s: the high half of the 64-bit product%s",
			    shared_str(x, 80), k ? ", shifted by what is left" : "");
			a = mk_inst(wide == U64 ? O_MULHI : O_SMULHI, a, b, 0);
			return k ? mk_bin(O_SHR, a, mk_const_i(N.ty[x], k)) : a;
		}
		break;
	case 4:		/* max(lo, min(v, hi)), min(hi, max(v, lo)), lo <= hi */
		if ((op == O_MAX || op == O_MIN) && TY_ISINT(N.ty[x]) &&
		    N.ty[x] != TY_BOOL)
			for (j = 0; j < 2; j++) {
				nid in = j ? N.b[x] : N.a[x], out = j ? N.a[x] : N.b[x];
				int in_op = op == O_MAX ? O_MIN : O_MAX, i;
				if (N.op[in] != in_op)
					continue;
				for (i = 0; i < 2; i++) {
					nid v = i ? N.b[in] : N.a[in];
					nid o2 = i ? N.a[in] : N.b[in];
					nid l = op == O_MAX ? out : o2;
					nid h = op == O_MAX ? o2 : out;
					if (!sg_range_of(f, l, 32, &lo, &hi) ||
					    !sg_range_of(f, h, 32, &lo2, &hi2) ||
					    hi > lo2)
						continue;
					sb_printf(why, "%s: %s clamped to %s..%s, which "
					    "is their median since %s <= %s (%lld <= "
					    "%lld, by intervals)", shared_str(x, 80),
					    node_str(v), node_str(l), node_str(h),
					    node_str(l), node_str(h), (long long)hi,
					    (long long)lo2);
					return mk_inst(O_MED3, v, l, h);
				}
			}
		break;
	case 5: {	/* a permutation of one word's bits */
		uint64_t slots = node_slots(x);
		struct aff *m2;
		int slot = -1, i, rev = 1, rot, perm[4], byp = 1;
		nid r = 0, w;

		if (N.ty[x] != U32 || op == O_VAR || IS_INST(op) ||
		    (slots >> MAXP))
			break;
		for (j = 0; j < f->np; j++)
			if ((slots >> j) & 1) {
				if (slot >= 0)
					return 0;
				slot = j;
			}
		if (slot < 0 || f->p[slot].ty != U32)
			break;
		m2 = xmalloc(sizeof *m2);
		if (!gf2_affine(x, slot, m2, 0) || m2->k || m2->w != 32) {
			free(m2);
			break;
		}
		w = mk_var((uint32_t)slot, f->p[slot].name, U32);
		for (i = 0; i < 32; i++)
			rev &= m2->row[i] == 1ULL << (31 - i);
		for (rot = 1; rot < 32; rot++) {
			for (i = 0; i < 32 && m2->row[i] == 1ULL << ((i - rot) & 31);
			    i++)
				;
			if (i == 32)
				break;
		}
		for (i = 0; i < 32 && byp; i++) {
			uint64_t rw = m2->row[i];
			int src = 0;
			if (!rw || (rw & (rw - 1))) {
				byp = 0;
				break;
			}
			while (!(rw & 1)) {
				rw >>= 1;
				src++;
			}
			if ((src & 7) != (i & 7) || ((i & 7) && perm[i >> 3] !=
			    src >> 3))
				byp = 0;
			perm[i >> 3] = src >> 3;
		}
		if (byp && perm[0] == 0 && perm[1] == 1 && perm[2] == 2 &&
		    perm[3] == 3)
			byp = 0;
		if (rev) {
			sb_printf(why, "%s: bit i is bit 31 - i of %s (linear over "
			    "GF(2) by its structure, every row one bit)",
			    shared_str(x, 80), node_str(w));
			r = mk_inst(O_BREV, w, 0, 0);
		} else if (rot < 32) {
			sb_printf(why, "%s: %s rotated left by %d (linear over "
			    "GF(2) by its structure, every row one bit): the "
			    "high word of (%s:%s) << %d", shared_str(x, 80),
			    node_str(w), rot, node_str(w), node_str(w), rot);
			r = mk_inst(O_FUNL, w, w, kint(rot));
		} else if (byp) {
			sb_printf(why, "%s: the bytes %d, %d, %d, %d of %s (linear "
			    "over GF(2) by its structure, every row one bit, "
			    "whole bytes moved)", shared_str(x, 80), perm[0], perm[1],
			    perm[2], perm[3], node_str(w));
			r = mk_inst(O_BYTEPERM, w, w, kint(perm[0] | perm[1] << 4 |
			    perm[2] << 8 | perm[3] << 12));
		}
		free(m2);
		return r;
	}
	case 6:		/* (base & ~m) | field, m the field's bits p..p+n-1 */
		return bfi_of(f, x, why);
	case 7:		/* u64(a) * u64(b), a and b words */
		if (op == O_MUL && N.ty[x] == U64) {
			struct sbuf how = {0};
			if (wide_product(f, x, &a, &b, &how)) {
				sb_printf(why, "%s: the 64-bit product of two "
				    "words%s", shared_str(x, 80), sb_str(&how));
				free(how.p);
				return mk_inst(O_MULWIDE, a, b, 0);
			}
			free(how.p);
		}
		break;
	case 8:		/* that product plus a u64 */
		return madwide_of(f, x, why);
	case 9:		/* the four bytes' |a - b| summed */
		return usad_of(x, why);
	case 10:	/* bytes of two words, signs of bytes */
		return byteperm_of(f, x, why);
	case 11:	/* a word of a 64-bit shift of two words */
		return funnel64_of(x, why);
	case 12:	/* bytes, signs of bytes 1, 3, 5, 7, 0 and 255 */
		return bytesel_of(f, x, why);
	}
	return 0;
}

#define NPAT 13

/*
 * The low word of a product, a * b in u32, where the function also reads
 * its high word (mulhi(a, b), or the pattern it is): both from one wide
 * product.  The candidate, or 0.
 */
static nid
wide_pair(const struct func *f, nid cur, const nid *nd, int n, nid x,
    struct sbuf *why)
{
	nid w, hi, cand;
	int i;
	struct sbuf scratch = {0};

	if (N.op[x] != O_MUL || N.ty[x] != U32)
		return 0;
	for (i = 0; i < n; i++) {
		hi = nd[i];
		if (N.op[hi] != O_MULHI) {
			nid h = pattern(f, 3, hi, &scratch);
			sb_clear(&scratch);
			if (!h || N.op[h] != O_MULHI)
				continue;
			if (!((N.a[h] == N.a[x] && N.b[h] == N.b[x]) ||
			    (N.a[h] == N.b[x] && N.b[h] == N.a[x])))
				continue;
		} else if (!((N.a[hi] == N.a[x] && N.b[hi] == N.b[x]) ||
		    (N.a[hi] == N.b[x] && N.b[hi] == N.a[x])))
			continue;
		w = mk_inst(O_MULWIDE, N.a[x], N.b[x], 0);
		cand = sg_replaced(cur, x, mk_cast(U32, w));
		if (cand)
			cand = sg_replaced(cand, hi, mk_cast(N.ty[hi],
			    mk_bin(O_SHR, w, ku(U64, 32))));
		if (!cand)
			continue;
		sb_printf(why, "%s and %s: the low and the high word of one "
		    "64-bit product", shared_str(x, 60), shared_str(hi, 60));
		free(scratch.p);
		return cand;
	}
	free(scratch.p);
	return 0;
}
#define MAXNODE 512

static int
all_nodes(nid e, nid *out, int n, uint8_t *seen)
{
	uint32_t j;

	if (e == 0 || seen[e] || n == MAXNODE)
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

void
inst_find(const struct func *f)
{
	const struct model *m = model_active();
	const char *mn = sym_str(m->name), *s, *e, *chk;
	struct sbuf why = {0}, one = {0}, best_why = {0}, w = {0};
	struct eqproof pr;
	nid cur = f->body, *nd;
	float before, cost;
	int k, iter, bad;

	for (k = C_BFE; k <= C_BYTESEL && !cclass_available(m, k); k++)
		;
	if (k > C_BYTESEL || !TY_ISINT(N.ty[cur]))
		return;
	nd = xmalloc(MAXNODE * sizeof *nd);
	before = cost = fuse_expr_cost(m, sg_priced(f, cur));
	for (iter = 0; iter < 32; iter++) {
		uint8_t *seen = xcalloc(N.n, 1);
		int n = all_nodes(cur, nd, 0, seen), i, p;
		nid best = 0;
		float bc = cost;

		free(seen);
		for (i = 0; i < n; i++) {
			nid cand;
			float c;
			sb_clear(&one);
			cand = wide_pair(f, cur, nd, n, nd[i], &one);
			if (cand && (c = fuse_expr_cost(m, sg_priced(f, cand))) <
			    bc) {
				bc = c;
				best = cand;
				sb_clear(&best_why);
				sb_puts(&best_why, sb_str(&one));
			}
			for (p = 0; p < NPAT; p++) {
				nid to, cand;
				float c;
				sb_clear(&one);
				to = pattern(f, p, nd[i], &one);
				if (!to || to == nd[i])
					continue;
				cand = sg_replaced(cur, nd[i], mk_cast(N.ty[nd[i]], to));
				if (!cand)
					continue;
				c = fuse_expr_cost(m, sg_priced(f, cand));
				if (c < bc) {
					bc = c;
					best = cand;
					sb_clear(&best_why);
					sb_puts(&best_why, sb_str(&one));
				}
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
	bad = equiv_decide(f, f->body, cur, &w, &pr);
	chk = pr.how == EQ_EXHAUSTIVE ? "checked on every point of the domain" :
	    pr.how == EQ_SAT ? "checked by SAT" :
	    "checked on a sample of the domain";
	if (bad) {
		printf("  instructions under %s: %s (cost %g) does not hold: %s\n",
		    mn, shared_str(cur, 300), (double)cost, sb_str(&w));
		rep_obj("instructions");
		rep_str("model", mn);
		rep_num("cost_before", before);
		rep_expr("candidate", cur);
		rep_str("not_proved", sb_str(&w));
		rep_end();
		goto out;
	}
	printf("  instructions under %s (each for the pattern it is; cost %g as "
	    "written):\n", mn, (double)before);
	printf("  cost %-3g %s\n", (double)cost, shared_str(cur, 400));
	for (s = sb_str(&why); *s; s = *e ? e + 1 : e) {
		e = strchr(s, '\n');
		if (!e)
			e = s + strlen(s);
		printf("           %.*s\n", (int)(e - s), s);
	}
	printf("           %s\n", chk);
	rep_obj("instructions");
	rep_str("model", mn);
	rep_num("cost_before", before);
	rep_num("cost", cost);
	rep_expr("expr", cur);
	rep_str("derivation", sb_str(&why));
	rep_str("check", pr.how == EQ_EXHAUSTIVE ? "exhaustive" : pr.how ==
	    EQ_SAT ? "sat" : "sampled");
	rep_end();
out:
	free(why.p);
	free(one.p);
	free(best_why.p);
	free(w.p);
}
