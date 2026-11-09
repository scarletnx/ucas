/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * The x86 intrinsics catalogue: every intrinsic with its exact semantics
 * per result lane as ucas nodes (so rewrites into it can be proved by
 * the evaluator and by SAT), its feature flags, its instruction class
 * (whose costs per µarch are in x86data.c) and its signature.
 *
 * Data, generated: a family row ("add_%s" over epi8..epi64 at 128, 256
 * and 512 bits, with AVX-512 mask/maskz variants) expands into many
 * intrinsics; irregular ones are explicit rows.  Semantics are written
 * in a small lane language, compiled to nodes on demand:
 *
 *   a b c d      the operands: the current lane i of a vector operand,
 *                the value of a scalar, mask or immediate one
 *   a[e]         lane e of a (a constant e is a plain lane; a computed
 *                one becomes a table lookup over a's lanes)
 *   i n na L w   the lane, the lanes of the result (of the first vector
 *                operand when the result is scalar), of a, per 128 bits,
 *                the result's element width in bits
 *   imm          the immediate
 *   + - * / % & | ^ ~ ! << >> == != < <= > >= && || ?:   as in C, on the
 *                operands' types; a constant ?: condition is lazy (the
 *                other arm is never built)
 *   u8(x) .. i64(x) f32(x) f64(x) int(x)   conversions (int: ideal 64-bit)
 *   u(x) s(x)    the unsigned / signed type of x's width
 *   sat(x)       x clamped to the result element's range
 *   m(c)         all ones (c true) or zero, as the result element
 *   asu(x) asf(x)  bits of a float, float of bits
 *   min max abs clz ctz popcnt sqrt fma copysign floor ceil
 *   bit(x, k)    (x >> k) & 1
 *   or(j, lo, hi, e) xor(...) add(...)   folds over a constant range
 *   @def F(p) = e;  @def X = e;           macros used by the expression
 *   builtins: rne(x) cvti(x) cvtti(x) rnd(x, mode) fcmp(x, y, pred)
 *     gfmul(x, y) ginv(x) sbox(x) isbox(x) xtime(x) aes_sr(k) aes_isr(k)
 *     crc32c(crc, v, bits) umulhi64(x, y)
 *
 * Floating point is IEEE with round-to-nearest-even (MXCSR's default);
 * a NaN result is any NaN (x86 returns the first operand's quieted NaN
 * or the default NaN; the payload is not modelled).  rcp/rsqrt are
 * approximations: their semantics is the exact value, and they are
 * checked against it with their documented relative error.
 */
#include "x86.h"
#include "calc.h"

#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ---- element types ---- */
enum { E_I8, E_U8, E_I16, E_U16, E_I32, E_U32, E_I64, E_U64, E_F32, E_F64,
	E_N };
static const char *const ename[E_N] = {"i8", "u8", "i16", "u16", "i32",
	"u32", "i64", "u64", "f32", "f64"};
static const int ebits[E_N] = {8, 8, 16, 16, 32, 32, 64, 64, 32, 64};

static ty_t
ety(int e)
{
	switch (e) {
	case E_F32: return TY_F32;
	case E_F64: return TY_F64;
	}
	return TY(e & 1 ? K_UINT : K_SINT, ebits[e]);
}

static int
efloat(int e)
{
	return e == E_F32 || e == E_F64;
}

static int
ename_find(const char *s, size_t n)
{
	int e;

	for (e = 0; e < E_N; e++)
		if (strlen(ename[e]) == n && !memcmp(ename[e], s, n))
			return e;
	return -1;
}

/* ---- the catalogue (SoA) ---- */
enum { OK_NONE, OK_V, OK_S, OK_I, OK_K, OK_P };
#define MAXOP 5
#define FA_APPROX12 1	/* relative error <= 1.5 * 2^-12 */
#define FA_APPROX14 2	/* <= 2^-14 */
#define FA_MEM 4	/* reads or writes memory: semantics in words only */
#define FA_PTROUT 8	/* a second result through the last (pointer) operand */
#define FA_NOEXEC 16	/* no lane semantics (memory) */
/*
 * Memory (FA_MEM with semantics): a pointer operand is the 64-byte
 * window at the pointer, its lanes the elements of the pointer's type
 * (a[3]: the fourth element; a alone: lane i).  A store's result is the
 * window after the store (FA_STORE); a gather's index operand stays
 * inside the window in the tests (FA_GATHER), and its semantics needs a
 * scale that is a multiple of the element size.
 */
#define FA_STORE 32
#define FA_GATHER 64
#define MEMBYTES 64
/* AVX-512 masking: r = k bit i ? base : src (or 0) */
enum { MM_NONE, MM_MASK, MM_MASKZ, MM_MASKA, MM_MASK3, MM_KMASK };

static struct {
	uint32_t n, cap;
	char **name;
	xfeat *feat;
	uint8_t *rk, *re;	/* result kind and element */
	uint16_t *rbits;
	uint8_t *nop;
	uint8_t (*ok)[MAXOP], (*oe)[MAXOP];
	uint16_t (*obits)[MAXOP];
	const char **sem, **sem2;
	char **semown;		/* generated semantics (mask variants) */
	uint8_t *cls, *fl, *imax;
	uint32_t *base;		/* mask variants: the unmasked intrinsic */
	uint8_t *mmode;		/* MM_* */
	uint32_t *ht;
	uint32_t htcap;
} I;

static uint32_t
hstr(const char *s)
{
	uint32_t h = 2166136261u;

	for (; *s; s++)
		h = (h ^ (uint8_t)*s) * 16777619u;
	return h;
}

static int
xi_find_raw(const char *name)
{
	uint32_t h;

	if (!I.htcap)
		return -1;
	h = hstr(name) & (I.htcap - 1);
	while (I.ht[h]) {
		if (!strcmp(I.name[I.ht[h] - 1], name))
			return (int)I.ht[h] - 1;
		h = (h + 1) & (I.htcap - 1);
	}
	return -1;
}

int
xi_find(const char *name)
{
	uint32_t h;

	x86_intr_init();
	h = hstr(name) & (I.htcap - 1);
	while (I.ht[h]) {
		if (!strcmp(I.name[I.ht[h] - 1], name))
			return (int)I.ht[h] - 1;
		h = (h + 1) & (I.htcap - 1);
	}
	return -1;
}

struct ispec {
	uint8_t rk, re, nop;
	uint16_t rbits;
	uint8_t ok[MAXOP], oe[MAXOP];
	uint16_t obits[MAXOP];
};

static void
xi_add(const char *name, xfeat feat, const struct ispec *sp,
    const char *sem, const char *sem2, int cls, int fl, int imax,
    char *own)
{
	uint32_t k = I.n, h;

	if (I.n == I.cap) {
		I.cap = newcap(I.cap, I.n + 1);
#define GROW(f) I.f = xrealloc(I.f, I.cap * sizeof *I.f)
		GROW(name); GROW(feat); GROW(rk); GROW(re); GROW(rbits);
		GROW(nop); GROW(ok); GROW(oe); GROW(obits); GROW(sem);
		GROW(sem2); GROW(semown); GROW(cls); GROW(fl); GROW(imax);
		GROW(base); GROW(mmode);
#undef GROW
	}
	I.name[k] = strdup(name);
	I.feat[k] = feat;
	I.rk[k] = sp->rk;
	I.re[k] = sp->re;
	I.rbits[k] = sp->rbits;
	I.nop[k] = sp->nop;
	memcpy(I.ok[k], sp->ok, MAXOP);
	memcpy(I.oe[k], sp->oe, MAXOP);
	memcpy(I.obits[k], sp->obits, sizeof I.obits[k]);
	I.sem[k] = sem;
	I.sem2[k] = sem2;
	I.semown[k] = own;
	I.cls[k] = (uint8_t)cls;
	I.fl[k] = (uint8_t)fl;
	I.imax[k] = (uint8_t)imax;
	I.base[k] = k;
	I.mmode[k] = 0;
	I.n++;
	/* the hash table (rebuilt when it fills) */
	if ((I.n + 1) * 2 > I.htcap) {
		uint32_t j;
		I.htcap = I.htcap ? I.htcap * 2 : 4096;
		free(I.ht);
		I.ht = xcalloc(I.htcap, sizeof *I.ht);
		for (j = 0; j < I.n; j++) {
			h = hstr(I.name[j]) & (I.htcap - 1);
			while (I.ht[h])
				h = (h + 1) & (I.htcap - 1);
			I.ht[h] = j + 1;
		}
		return;
	}
	h = hstr(name) & (I.htcap - 1);
	while (I.ht[h])
		h = (h + 1) & (I.htcap - 1);
	I.ht[h] = k + 1;
}

/* ---- the lane language ---- */

struct tp {
	const char *s;
	size_t pos;
	int id, lane, dry;
	int64_t imm;
	xi_lane_fn fn;
	void *ctx;
	/* bound names: fold variables and macro parameters */
	int nb;
	char bname[8][12];
	nid bval[8];
	/* macros */
	int nm;
	char mname[6][12], mparm[6][12];
	size_t mbody[6];
};

static nid tx(struct tp *t);

static void
terr(struct tp *t, const char *what)
{
	fail("intrinsic %s: semantics: %s at '%.20s'", I.name[t->id], what,
	    t->s + t->pos);
}

static void
ws(struct tp *t)
{
	while (isspace((unsigned char)t->s[t->pos]))
		t->pos++;
}

static int
eat(struct tp *t, const char *tok)
{
	size_t n = strlen(tok);

	ws(t);
	if (strncmp(t->s + t->pos, tok, n))
		return 0;
	/* "<" must not eat "<<" or "<=", "&" not "&&" ... */
	if (n == 1 && strchr("<>&|=!", tok[0]) && (t->s[t->pos + 1] == tok[0] ||
	    (t->s[t->pos + 1] == '=' && tok[0] != '=' && tok[0] != '!')))
		return 0;
	if (n == 1 && tok[0] == '!' && t->s[t->pos + 1] == '=')
		return 0;
	t->pos += n;
	return 1;
}

static void
need(struct tp *t, const char *tok)
{
	if (!eat(t, tok))
		terr(t, tok);
}

static int
ident(struct tp *t, char *buf, size_t n)
{
	size_t k = 0;

	ws(t);
	if (!isalpha((unsigned char)t->s[t->pos]) && t->s[t->pos] != '_')
		return 0;
	while (isalnum((unsigned char)t->s[t->pos]) || t->s[t->pos] == '_') {
		if (k + 1 < n)
			buf[k++] = t->s[t->pos];
		t->pos++;
	}
	buf[k] = 0;
	return 1;
}

static nid
K(int64_t v)
{
	return mk_const_i(TY_LIT, v);
}

static int
isk(nid n, int64_t *v)
{
	if (!is_const(n) || TY_FP(N.ty[n]))
		return 0;
	*v = N.ki[n];
	return 1;
}

static int
lit(nid n)
{
	return is_const(n) && (TYK(N.ty[n]) == K_LIT || TYK(N.ty[n]) == K_REAL);
}

/* operands brought to one type, so no "mixing" notes */
static void
unify(nid *x, nid *y)
{
	ty_t a = N.ty[*x], b = N.ty[*y], t;

	if (a == b)
		return;
	if (lit(*x) && !lit(*y)) {
		*x = mk_cast(b, *x);
		return;
	}
	if (lit(*y) && !lit(*x)) {
		*y = mk_cast(a, *y);
		return;
	}
	if (TY_FP(a) || TY_FP(b)) {
		if (TY_FP(a) && TY_FP(b))
			t = TYW(a) >= TYW(b) ? a : b;
		else
			t = TY_FP(a) ? a : b;
	} else if (a == TY_BOOL)
		t = b == TY_LIT ? TY_INT : b;
	else if (b == TY_BOOL)
		t = a == TY_LIT ? TY_INT : a;
	else if (TYK(a) == K_INT || TYK(b) == K_INT || TYK(a) == K_LIT ||
	    TYK(b) == K_LIT)
		t = TY_INT;
	else
		t = TYW(a) >= TYW(b) ? a : b;
	*x = mk_cast(t, *x);
	*y = mk_cast(t, *y);
}

static nid
bin(int op, nid x, nid y)
{
	int64_t p, q;

	/* integer arithmetic on lane indices stays integer (1/2 is not 0.5) */
	if ((op == O_DIV || op == O_MOD) && isk(x, &p) && isk(y, &q) && q &&
	    TYK(N.ty[x]) == K_LIT && TYK(N.ty[y]) == K_LIT)
		return K(op == O_DIV ? p / q : p % q);
	if (op != O_SHL && op != O_SHR)
		unify(&x, &y);
	else if (TYK(N.ty[x]) == K_LIT || N.ty[x] == TY_BOOL)
		x = mk_cast(TY_INT, x);
	return mk_bin(op, x, y);
}

static int
lanes_of(int bits, int e)
{
	return bits ? bits / ebits[e] : 1;
}

static int
rlanes(int id)
{
	return I.rk[id] == OK_V ? lanes_of(I.rbits[id], I.re[id]) : 1;
}

/* lanes of the first vector operand (n when the result is scalar) */
static int
vlanes(int id)
{
	int k;

	if (I.rk[id] == OK_V)
		return rlanes(id);
	for (k = 0; k < I.nop[id]; k++)
		if (I.ok[id][k] == OK_V)
			return lanes_of(I.obits[id][k], I.oe[id][k]);
	return 1;
}

static ty_t
rty(int id)
{
	if (I.rk[id] == OK_K)
		return TY(K_UINT, I.rbits[id]);
	return ety(I.re[id]);
}

static nid
opnd(struct tp *t, int k, int lane)
{
	int id = t->id;

	if (t->dry)
		return K(0);
	if (k >= I.nop[id])
		terr(t, "no such operand");
	switch (I.ok[id][k]) {
	case OK_I:
		return K(t->imm);
	case OK_P:
		if (!(I.fl[id] & FA_MEM))
			return t->fn(t->ctx, k, 0);
		if (lane < 0 || lane >= MEMBYTES * 8 / ebits[I.oe[id][k]])
			terr(t, "memory lane outside the 64-byte window");
		return t->fn(t->ctx, k, lane);
	case OK_V:
		if (lane < 0 || lane >= lanes_of(I.obits[id][k], I.oe[id][k]))
			terr(t, "lane out of range");
		return t->fn(t->ctx, k, lane);
	default:
		return t->fn(t->ctx, k, 0);
	}
}

static int
opletter(const char *w)
{
	if (w[0] >= 'a' && w[0] <= 'e' && w[1] == 0)
		return w[0] - 'a';
	return -1;
}

static nid
sat_to(ty_t t, nid x)
{
	int64_t lo, hi;
	int w = TYW(t);

	if (TYK(t) == K_UINT) {
		lo = 0;
		hi = w >= 63 ? INT64_MAX : (int64_t)((1ULL << w) - 1);
	} else {
		lo = -(int64_t)(1ULL << (w - 1));
		hi = (int64_t)((1ULL << (w - 1)) - 1);
	}
	x = mk_cast(TY_INT, x);
	x = mk_bin(O_MAX, x, mk_const_i(TY_INT, lo));
	x = mk_bin(O_MIN, x, mk_const_i(TY_INT, hi));
	return mk_cast(t, x);
}

static ty_t
same_width(ty_t t, int kind)
{
	if (TY_FP(t))
		return TY(kind, TYW(t));
	if (!TY_SIZED(t))
		return TY(kind, 64);
	return TY(kind, TYW(t));
}

/* round to nearest even, sign kept (-0.5 -> -0) */
static nid
rne_node(nid x)
{
	ty_t t = N.ty[x];
	nid f = mk_un(O_FLOOR, x), d, half = mk_cast(t, mk_const_r(0.5));
	nid one = mk_cast(t, mk_const_r(1)), two = mk_cast(t, mk_const_r(2));
	nid fp1 = mk_bin(O_ADD, f, one), even, r;

	d = mk_bin(O_SUB, x, f);
	even = mk_bin(O_EQ, mk_bin(O_SUB, f, mk_bin(O_MUL, two,
	    mk_un(O_FLOOR, mk_bin(O_MUL, f, half)))), mk_cast(t, mk_const_r(0)));
	r = mk_sel(mk_bin(O_GT, d, half), fp1, mk_sel(mk_bin(O_LT, d, half), f,
	    mk_sel(even, f, fp1)));
	return mk_bin(O_COPYSIGN, r, x);
}

static nid
trunc_node(nid x)
{
	nid z = mk_cast(N.ty[x], mk_const_r(0));

	return mk_bin(O_COPYSIGN, mk_sel(mk_bin(O_LT, x, z), mk_un(O_CEIL, x),
	    mk_un(O_FLOOR, x)), x);
}

/* float -> iN as cvt(t)ps2dq: out of range or NaN gives INT_MIN */
static nid
cvt_node(nid x, int w, int trunc)
{
	ty_t it = TY(K_SINT, w), ft = N.ty[x];
	double lim = ldexp(1, w - 1);
	nid r = trunc ? trunc_node(x) : rne_node(x);
	nid bad = mk_bin(O_LOR, mk_bin(O_NE, x, x), mk_bin(O_LOR,
	    mk_bin(O_GE, r, mk_cast(ft, mk_const_r(lim))),
	    mk_bin(O_LT, r, mk_cast(ft, mk_const_r(-lim)))));

	return mk_sel(bad, mk_const_i(it, -(int64_t)(1ULL << (w - 1))),
	    mk_cast(it, mk_sel(bad, mk_cast(ft, mk_const_r(0)), r)));
}

static nid
fcmp_node(nid a, nid b, int p)
{
	nid un = mk_bin(O_LOR, mk_bin(O_NE, a, a), mk_bin(O_NE, b, b));

	switch (p & 15) {
	case 0: return mk_bin(O_EQ, a, b);
	case 1: return mk_bin(O_LT, a, b);
	case 2: return mk_bin(O_LE, a, b);
	case 3: return un;
	case 4: return mk_bin(O_NE, a, b);
	case 5: return mk_un(O_LNOT, mk_bin(O_LT, a, b));
	case 6: return mk_un(O_LNOT, mk_bin(O_LE, a, b));
	case 7: return mk_un(O_LNOT, un);
	case 8: return mk_bin(O_LOR, un, mk_bin(O_EQ, a, b));
	case 9: return mk_un(O_LNOT, mk_bin(O_GE, a, b));
	case 10: return mk_un(O_LNOT, mk_bin(O_GT, a, b));
	case 11: return mk_const_i(TY_BOOL, 0);
	case 12: return mk_bin(O_LAND, mk_un(O_LNOT, un), mk_bin(O_NE, a, b));
	case 13: return mk_bin(O_GE, a, b);
	case 14: return mk_bin(O_GT, a, b);
	}
	return mk_const_i(TY_BOOL, 1);
}

/* GF(2^8) with x^8 + x^4 + x^3 + x + 1 (AES, GFNI) */
static int
gf_mul(int a, int b)
{
	int r = 0;

	while (b) {
		if (b & 1)
			r ^= a;
		a = (a << 1) ^ ((a & 0x80) ? 0x11b : 0);
		b >>= 1;
	}
	return r & 0xff;
}

static int gf_inv_t[256], sbox_t[256], isbox_t[256], tables_ready;

static void
gf_tables(void)
{
	int x, y;

	if (tables_ready)
		return;
	tables_ready = 1;
	for (x = 1; x < 256; x++)
		for (y = 1; y < 256; y++)
			if (gf_mul(x, y) == 1) {
				gf_inv_t[x] = y;
				break;
			}
	for (x = 0; x < 256; x++) {
		int b = gf_inv_t[x], s = b, k;
		for (k = 1; k <= 4; k++)
			s ^= ((b << k) | (b >> (8 - k))) & 0xff;
		sbox_t[x] = s ^ 0x63;
		isbox_t[sbox_t[x]] = x;
	}
}

static nid
table_lookup(const int *tab, nid x)
{
	nid e[256];
	int k;

	for (k = 0; k < 256; k++)
		e[k] = mk_const_i(TY(K_UINT, 8), tab[k]);
	return mk_index(mk_table(e, 256), mk_cast(TY(K_UINT, 8), x));
}

static nid
xtime_node(nid x)
{
	ty_t u8 = TY(K_UINT, 8);

	x = mk_cast(u8, x);
	return mk_bin(O_XOR, mk_bin(O_SHL, x, K(1)), mk_bin(O_MUL,
	    mk_bin(O_SHR, x, K(7)), mk_const_i(u8, 0x1b)));
}

static nid
gfmul_node(nid x, nid y)
{
	ty_t u8 = TY(K_UINT, 8);
	nid r = mk_const_i(u8, 0), a = mk_cast(u8, x), b = mk_cast(u8, y);
	int k;
	int64_t cv;

	if (isk(b, &cv)) {
		for (k = 0; k < 8; k++, a = xtime_node(a))
			if ((cv >> k) & 1)
				r = mk_bin(O_XOR, r, a);
		return r;
	}
	for (k = 0; k < 8; k++, a = xtime_node(a))
		r = mk_bin(O_XOR, r, mk_bin(O_AND, a, mk_bin(O_SUB,
		    mk_const_i(u8, 0), mk_bin(O_AND, mk_bin(O_SHR, b, K(k)),
		    mk_const_i(u8, 1)))));
	return r;
}

static nid
crc_node(nid crc, nid v, int bits)
{
	ty_t u64 = TY(K_UINT, 64);
	nid c = mk_bin(O_XOR, mk_cast(u64, mk_cast(TY(K_UINT, 32), crc)),
	    mk_cast(u64, v)), p = mk_const_i(u64, 0x82f63b78);
	int k;

	for (k = 0; k < bits; k++)
		c = mk_bin(O_XOR, mk_bin(O_SHR, c, K(1)), mk_bin(O_AND, p,
		    mk_bin(O_SUB, mk_const_i(u64, 0), mk_bin(O_AND, c,
		    mk_const_i(u64, 1)))));
	return mk_cast(TY(K_UINT, 32), c);
}

static nid
umulhi_node(nid x, nid y)
{
	ty_t u = TY(K_UINT, 64);
	nid m = mk_const_i(u, 0xffffffff), s = K(32);
	nid al, ah, bl, bh, ll, lh, hl, hh, mid;

	x = mk_cast(u, x);
	y = mk_cast(u, y);
	al = mk_bin(O_AND, x, m);
	ah = mk_bin(O_SHR, x, s);
	bl = mk_bin(O_AND, y, m);
	bh = mk_bin(O_SHR, y, s);
	ll = mk_bin(O_MUL, al, bl);
	lh = mk_bin(O_MUL, al, bh);
	hl = mk_bin(O_MUL, ah, bl);
	hh = mk_bin(O_MUL, ah, bh);
	mid = mk_bin(O_ADD, mk_bin(O_ADD, mk_bin(O_SHR, ll, s), mk_bin(O_AND,
	    lh, m)), mk_bin(O_AND, hl, m));
	return mk_bin(O_ADD, mk_bin(O_ADD, mk_bin(O_ADD, hh, mk_bin(O_SHR, lh,
	    s)), mk_bin(O_SHR, hl, s)), mk_bin(O_SHR, mid, s));
}

/* argument list after '(' */
static int
args(struct tp *t, nid *v, int max)
{
	int n = 0;

	if (eat(t, ")"))
		return 0;
	do {
		if (n == max)
			terr(t, "too many arguments");
		v[n++] = tx(t);
	} while (eat(t, ","));
	need(t, ")");
	return n;
}

static int64_t
kval(struct tp *t, nid n)
{
	int64_t v;

	if (t->dry)
		return 0;
	if (!isk(n, &v))
		terr(t, "a constant is needed");
	return v;
}

/* fold: or(j, lo, hi, expr) */
static nid
fold(struct tp *t, int op)
{
	char var[12];
	int64_t lo, hi, j;
	size_t body, end = 0;
	nid acc = 0, v;

	if (!ident(t, var, sizeof var))
		terr(t, "a fold variable");
	need(t, ",");
	lo = kval(t, tx(t));
	need(t, ",");
	hi = kval(t, tx(t));
	need(t, ",");
	body = t->pos;
	if (t->nb == 8)
		terr(t, "too deep");
	snprintf(t->bname[t->nb], sizeof t->bname[0], "%s", var);
	t->nb++;
	if (t->dry || hi < lo) {
		int dry = t->dry;
		t->dry = 1;
		t->bval[t->nb - 1] = K(0);
		tx(t);
		t->dry = dry;
		t->nb--;
		need(t, ")");
		return op == O_ADD || op == O_OR || op == O_XOR ? K(0) : K(0);
	}
	for (j = lo; j <= hi; j++) {
		t->pos = body;
		t->bval[t->nb - 1] = K(j);
		v = tx(t);
		end = t->pos;
		acc = acc ? bin(op, acc, v) : v;
	}
	t->pos = end;
	t->nb--;
	need(t, ")");
	return acc;
}

static nid
macro_call(struct tp *t, int m)
{
	size_t back;
	nid v = 0, r;
	int dry;

	if (t->mparm[m][0]) {
		need(t, "(");
		v = tx(t);
		need(t, ")");
	}
	back = t->pos;
	if (t->nb == 8)
		terr(t, "too deep");
	if (t->mparm[m][0]) {
		snprintf(t->bname[t->nb], sizeof t->bname[0], "%s", t->mparm[m]);
		t->bval[t->nb++] = v;
	}
	t->pos = t->mbody[m];
	dry = t->dry;
	r = tx(t);
	t->dry = dry;
	if (t->mparm[m][0])
		t->nb--;
	t->pos = back;
	return r;
}

static nid
primary(struct tp *t)
{
	char w[24];
	int k;
	nid v[4];

	ws(t);
	if (eat(t, "(")) {
		nid r = tx(t);
		need(t, ")");
		return r;
	}
	if (isdigit((unsigned char)t->s[t->pos])) {
		char *e;
		const char *s = t->s + t->pos;
		if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
			uint64_t u = strtoull(s, &e, 16);
			t->pos += (size_t)(e - s);
			return K((int64_t)u);
		}
		{
			double d = strtod(s, &e);
			size_t len = (size_t)(e - s);
			if (memchr(s, '.', len) || memchr(s, 'e', len)) {
				t->pos += len;
				return mk_const_r(d);
			}
			t->pos += (size_t)(e - s);
			return K((int64_t)strtoll(s, NULL, 10));
		}
	}
	if (!ident(t, w, sizeof w))
		terr(t, "an operand");
	for (k = t->nb; k-- > 0;)
		if (!strcmp(t->bname[k], w))
			return t->bval[k];
	for (k = 0; k < t->nm; k++)
		if (!strcmp(t->mname[k], w))
			return macro_call(t, k);
	if (!strcmp(w, "i"))
		return K(t->lane);
	if (!strcmp(w, "n"))
		return K(vlanes(t->id));
	if (!strcmp(w, "na"))
		return K(lanes_of(I.obits[t->id][0], I.oe[t->id][0]));
	if (!strcmp(w, "L"))
		return K(I.rk[t->id] == OK_V ? 128 / ebits[I.re[t->id]] :
		    vlanes(t->id));
	if (!strcmp(w, "w"))
		return K(I.rk[t->id] == OK_K ? I.rbits[t->id] : ebits[I.re[t->id]]);
	if (!strcmp(w, "imm"))
		return K(t->imm);
	if ((k = opletter(w)) >= 0) {
		if (eat(t, "[")) {
			nid ix = tx(t);
			int64_t lv;
			need(t, "]");
			if (t->dry)
				return K(0);
			if (isk(ix, &lv))
				return opnd(t, k, (int)lv);
			if (I.ok[t->id][k] != OK_V && (I.ok[t->id][k] != OK_P ||
			    !(I.fl[t->id] & FA_MEM)))
				terr(t, "indexing a scalar");
			{
				int nl = I.ok[t->id][k] == OK_P ? MEMBYTES * 8 /
				    ebits[I.oe[t->id][k]] : lanes_of(I.obits[t->id][k],
				    I.oe[t->id][k]), j;
				nid *e = xcalloc((size_t)nl, sizeof *e), r;
				for (j = 0; j < nl; j++)
					e[j] = opnd(t, k, j);
				r = mk_index(mk_table(e, (uint32_t)nl), ix);
				free(e);
				return r;
			}
		}
		return opnd(t, k, I.ok[t->id][k] == OK_V || (I.ok[t->id][k] ==
		    OK_P && (I.fl[t->id] & FA_MEM)) ? t->lane : 0);
	}
	/* functions */
	if (!strcmp(w, "or") || !strcmp(w, "xor") || !strcmp(w, "add")) {
		need(t, "(");
		return fold(t, w[0] == 'o' ? O_OR : w[0] == 'x' ? O_XOR : O_ADD);
	}
	need(t, "(");
	k = args(t, v, 4);
	if (t->dry)
		return K(0);
	{
		int e = ename_find(w, strlen(w));
		if (e >= 0 && k == 1)
			return mk_cast(ety(e), v[0]);
	}
#define NA(c) if (k != (c)) terr(t, "argument count")
	if (!strcmp(w, "int")) {
		NA(1);
		return mk_cast(TY_INT, v[0]);
	}
	if (!strcmp(w, "u") || !strcmp(w, "s")) {
		NA(1);
		return mk_cast(same_width(N.ty[v[0]], w[0] == 'u' ? K_UINT :
		    K_SINT), v[0]);
	}
	if (!strcmp(w, "sat")) {
		NA(1);
		return sat_to(ety(I.re[t->id]), v[0]);
	}
	if (!strcmp(w, "m")) {
		ty_t et = ety(I.re[t->id]), ut = same_width(et, K_UINT);
		nid r;
		NA(1);
		r = mk_sel(mk_bool(v[0]), mk_const_i(ut, -1), mk_const_i(ut, 0));
		return TY_FP(et) ? mk_un(O_ASFLOAT, r) : mk_cast(et, r);
	}
	if (!strcmp(w, "asu")) {
		NA(1);
		return mk_un(O_ASUINT, v[0]);
	}
	if (!strcmp(w, "asf")) {
		NA(1);
		return mk_un(O_ASFLOAT, mk_cast(same_width(N.ty[v[0]], K_UINT),
		    v[0]));
	}
	if (!strcmp(w, "min") || !strcmp(w, "max")) {
		NA(2);
		return bin(w[1] == 'i' ? O_MIN : O_MAX, v[0], v[1]);
	}
	if (!strcmp(w, "copysign")) {
		NA(2);
		return bin(O_COPYSIGN, v[0], v[1]);
	}
	if (!strcmp(w, "bit")) {
		NA(2);
		return mk_bin(O_AND, bin(O_SHR, v[0], v[1]), mk_const_i(N.ty[v[0]] ==
		    TY_LIT ? TY_INT : N.ty[v[0]], 1));
	}
	if (!strcmp(w, "fma")) {
		NA(3);
		unify(&v[0], &v[1]);
		unify(&v[1], &v[2]);
		unify(&v[0], &v[2]);
		return mk_fma(v[0], v[1], v[2]);
	}
	{
		static const struct { const char *n; int op; } un[] = {
			{"abs", O_ABS}, {"clz", O_CLZ}, {"ctz", O_CTZ},
			{"popcnt", O_POPCNT}, {"sqrt", O_SQRT}, {"floor", O_FLOOR},
			{"ceil", O_CEIL}, {NULL, 0}};
		int j;
		for (j = 0; un[j].n; j++)
			if (!strcmp(w, un[j].n)) {
				NA(1);
				return mk_un(un[j].op, v[0]);
			}
	}
	if (!strcmp(w, "rne")) {
		NA(1);
		return rne_node(v[0]);
	}
	if (!strcmp(w, "cvti") || !strcmp(w, "cvtti")) {
		NA(2);
		return cvt_node(v[0], (int)kval(t, v[1]), w[3] == 't');
	}
	if (!strcmp(w, "rnd")) {
		int64_t md;
		NA(2);
		md = kval(t, v[1]);
		if (md & 4)
			md = 0;	/* MXCSR: round to nearest even */
		switch (md & 3) {
		case 0: return rne_node(v[0]);
		case 1: return mk_un(O_FLOOR, v[0]);
		case 2: return mk_un(O_CEIL, v[0]);
		}
		return trunc_node(v[0]);
	}
	if (!strcmp(w, "fcmp")) {
		NA(3);
		unify(&v[0], &v[1]);
		return fcmp_node(v[0], v[1], (int)kval(t, v[2]));
	}
	gf_tables();
	if (!strcmp(w, "gfmul")) {
		NA(2);
		return gfmul_node(v[0], v[1]);
	}
	if (!strcmp(w, "ginv") || !strcmp(w, "sbox") || !strcmp(w, "isbox")) {
		NA(1);
		return table_lookup(w[0] == 'g' ? gf_inv_t : w[0] == 's' ? sbox_t :
		    isbox_t, v[0]);
	}
	if (!strcmp(w, "xtime")) {
		NA(1);
		return xtime_node(v[0]);
	}
	if (!strcmp(w, "aes_sr") || !strcmp(w, "aes_isr")) {
		int64_t q;
		NA(1);
		q = kval(t, v[0]);
		/* byte r + 4c of the state; ShiftRows takes column c + r */
		return K((q & 3) + 4 * (((q >> 2) + (w[4] == 'i' ? 4 - (q & 3) :
		    (q & 3))) & 3));
	}
	if (!strcmp(w, "ilog")) {
		int64_t q = 0, x;
		NA(1);
		x = kval(t, v[0]);
		while ((1LL << (q + 1)) <= x)
			q++;
		return K(q);
	}
	if (!strcmp(w, "crc32c")) {
		NA(3);
		return crc_node(v[0], v[1], (int)kval(t, v[2]));
	}
	if (!strcmp(w, "umulhi64")) {
		NA(2);
		return umulhi_node(v[0], v[1]);
	}
#undef NA
	terr(t, "unknown function");
	return 0;
}

static nid
postfix(struct tp *t)
{
	return primary(t);
}

static nid
unary(struct tp *t)
{
	nid x;

	if (eat(t, "-")) {
		x = unary(t);
		if (t->dry)
			return x;
		if (lit(x))
			return mk_bin(O_SUB, K(0), x);
		return mk_un(O_NEG, x);
	}
	if (eat(t, "~")) {
		x = unary(t);
		return t->dry ? x : mk_un(O_NOT, TYK(N.ty[x]) == K_LIT ?
		    mk_cast(TY_INT, x) : x);
	}
	if (eat(t, "!")) {
		x = unary(t);
		return t->dry ? x : mk_un(O_LNOT, x);
	}
	return postfix(t);
}

/* binary levels, loosest first */
static const struct { const char *tok; int op, lvl; } binops[] = {
	{"||", O_LOR, 0}, {"&&", O_LAND, 1}, {"|", O_OR, 2}, {"^", O_XOR, 3},
	{"&", O_AND, 4}, {"==", O_EQ, 5}, {"!=", O_NE, 5}, {"<=", O_LE, 6},
	{">=", O_GE, 6}, {"<<", O_SHL, 7}, {">>", O_SHR, 7}, {"<", O_LT, 6},
	{">", O_GT, 6}, {"+", O_ADD, 8}, {"-", O_SUB, 8}, {"*", O_MUL, 9},
	{"/", O_DIV, 9}, {"%", O_MOD, 9}, {NULL, 0, 0}};

static nid
binary(struct tp *t, int lvl)
{
	nid x;
	int k, again = 1;

	if (lvl > 9)
		return unary(t);
	x = binary(t, lvl + 1);
	while (again) {
		again = 0;
		for (k = 0; binops[k].tok; k++) {
			nid y;
			if (binops[k].lvl != lvl || !eat(t, binops[k].tok))
				continue;
			if (binops[k].op == O_LAND || binops[k].op == O_LOR) {
				int64_t cv;
				/* constant left side: lazy */
				if (!t->dry && isk(x, &cv) &&
				    (binops[k].op == O_LAND ? cv == 0 : cv != 0)) {
					t->dry++;
					binary(t, lvl + 1);
					t->dry--;
					x = mk_const_i(TY_BOOL, cv != 0);
					again = 1;
					break;
				}
			}
			y = binary(t, lvl + 1);
			if (!t->dry)
				x = binops[k].op == O_LAND || binops[k].op == O_LOR ?
				    mk_bin(binops[k].op, x, y) : bin(binops[k].op, x, y);
			again = 1;
			break;
		}
	}
	return x;
}

static nid
tx(struct tp *t)
{
	nid c = binary(t, 0), x, y;
	int64_t cv;

	if (!eat(t, "?"))
		return c;
	if (!t->dry && isk(c, &cv)) {
		if (cv) {
			x = tx(t);
			need(t, ":");
			t->dry++;
			tx(t);
			t->dry--;
			return x;
		}
		t->dry++;
		tx(t);
		t->dry--;
		need(t, ":");
		return tx(t);
	}
	x = tx(t);
	need(t, ":");
	y = tx(t);
	if (t->dry)
		return x;
	unify(&x, &y);
	return mk_sel(mk_bool(c), x, y);
}

/* @def NAME(p) = body; ... then the expression */
static void
macros(struct tp *t)
{
	for (;;) {
		char nm[12], pm[12] = "";
		ws(t);
		if (strncmp(t->s + t->pos, "@def", 4))
			return;
		t->pos += 4;
		if (!ident(t, nm, sizeof nm))
			terr(t, "a macro name");
		if (eat(t, "(")) {
			if (!ident(t, pm, sizeof pm))
				terr(t, "a parameter");
			need(t, ")");
		}
		need(t, "=");
		if (t->nm == 6)
			terr(t, "too many macros");
		snprintf(t->mname[t->nm], 12, "%s", nm);
		snprintf(t->mparm[t->nm], 12, "%s", pm);
		t->mbody[t->nm] = t->pos;
		t->nm++;
		/* skip the body: parse it dry */
		t->dry++;
		if (pm[0]) {
			snprintf(t->bname[t->nb], 12, "%s", pm);
			t->bval[t->nb++] = K(0);
		}
		tx(t);
		if (pm[0])
			t->nb--;
		t->dry--;
		need(t, ";");
	}
}

static nid
lane_sem(int id, const char *sem, int lane, int64_t imm, xi_lane_fn fn,
    void *ctx)
{
	struct tp t;
	nid r;

	memset(&t, 0, sizeof t);
	t.s = sem;
	t.id = id;
	t.lane = lane;
	t.imm = imm;
	t.fn = fn;
	t.ctx = ctx;
	macros(&t);
	r = tx(&t);
	ws(&t);
	if (t.s[t.pos])
		terr(&t, "trailing text");
	return mk_cast(rty(id), r);
}

/* a mask variant's operands seen as its base's */
struct remap {
	xi_lane_fn fn;
	void *ctx;
	int map[MAXOP];
};

static nid
remap_fn(void *c, int k, int lane)
{
	struct remap *r = c;

	return r->fn(r->ctx, r->map[k], lane);
}

nid
xi_lane(int id, int lane, int64_t imm, xi_lane_fn fn, void *ctx)
{
	if (I.mmode[id]) {
		int b = (int)I.base[id], k, kop, src = -1;
		struct remap rm;
		nid base, kb, other;
		rm.fn = fn;
		rm.ctx = ctx;
		switch (I.mmode[id]) {
		case MM_MASK:		/* (src, k, ops...) */
			src = 0, kop = 1;
			for (k = 0; k < I.nop[b]; k++)
				rm.map[k] = k + 2;
			break;
		case MM_MASKA:		/* (a, k, ops...): src is a */
			src = 0, kop = 1;
			rm.map[0] = 0;
			for (k = 1; k < I.nop[b]; k++)
				rm.map[k] = k + 1;
			break;
		case MM_MASK3:		/* (a, b, c, k): src is c */
			kop = I.nop[b];
			src = I.nop[b] - 1;
			for (k = 0; k < I.nop[b]; k++)
				rm.map[k] = k;
			break;
		default:		/* maskz, kmask: (k, ops...) */
			kop = 0;
			for (k = 0; k < I.nop[b]; k++)
				rm.map[k] = k + 1;
		}
		base = xi_lane(b, lane, imm, remap_fn, &rm);
		kb = fn(ctx, kop, 0);
		if (I.mmode[id] == MM_KMASK)
			return mk_bin(O_AND, base, mk_cast(N.ty[base], kb));
		kb = mk_bin(O_NE, mk_bin(O_AND, mk_bin(O_SHR, kb, K(lane)),
		    mk_const_i(N.ty[kb], 1)), mk_const_i(N.ty[kb], 0));
		other = src >= 0 ? fn(ctx, src, lane) : mk_cast(N.ty[base],
		    K(0));
		return mk_sel(kb, base, mk_cast(N.ty[base], other));
	}
	if (!I.sem[id])
		fail("%s has no lane semantics (it reads or writes memory)",
		    I.name[id]);
	return lane_sem(id, I.sem[id], lane, imm, fn, ctx);
}

static nid
xi_lane2(int id, int lane, int64_t imm, xi_lane_fn fn, void *ctx)
{
	struct ispec save;
	nid r;

	/* the pointer result's type is the pointer operand's element */
	save.re = I.re[id];
	I.re[id] = I.oe[id][I.nop[id] - 1];
	r = lane_sem(id, I.sem2[id], lane, imm, fn, ctx);
	I.re[id] = save.re;
	return r;
}

int xi_lanes(int id) { return rlanes(id); }
ty_t xi_rtype(int id) { return rty(id); }
ty_t xi_otype(int id, int k) { return ety(I.oe[id][k]); }
int xi_nops(int id) { return I.nop[id]; }
int xi_cls(int id) { return I.cls[id]; }
xfeat xi_feat(int id) { return I.feat[id]; }
const char *xi_name(int id) { return I.name[id]; }
/* operand k's kind: 'v' vector, 's' scalar, 'i' immediate, 'k' mask, 'p'
 * pointer */
int xi_okind(int id, int k) { return "?vsikp"[I.ok[id][k]]; }
uint32_t xi_count(void) { x86_intr_init(); return I.n; }

int
xi_vw(int id)
{
	int bits = I.rbits[id], k;

	for (k = 0; k < I.nop[id]; k++)
		if (I.ok[id][k] == OK_V && I.obits[id][k] > bits)
			bits = I.obits[id][k];
	if (I.rk[id] != OK_V && I.rk[id] != OK_K) {
		bits = 0;
		for (k = 0; k < I.nop[id]; k++)
			if (I.ok[id][k] == OK_V && I.obits[id][k] > bits)
				bits = I.obits[id][k];
	}
	return bits > 256 ? VW_Z : bits > 128 ? VW_Y : VW_X;
}

#include "x86intr_tab.inc"
