/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Expression nodes.  The store is a hash-consed DAG kept as parallel
 * arrays.  Constructors do type inference, implicit conversions, local
 * simplification and constant folding before interning, so equal
 * expressions get equal ids.
 */
#include "calc.h"
#include "texalu.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const struct opinfo opinfo[O_NOPS] = {
	[O_NONE] = {"none", 0, 0},
	[O_CONST] = {"const", 0, 0},
	[O_VAR] = {"var", 0, 0},
	[O_UNDEF] = {"undefined", 0, OF_PARTIAL},
	[O_NEG] = {"-", 1, 0},
	[O_NOT] = {"~", 1, OF_INT},
	[O_LNOT] = {"!", 1, 0},
	[O_ABS] = {"abs", 1, 0},
	[O_SIGN] = {"sign", 1, 0},
	[O_SQRT] = {"sqrt", 1, OF_REALFN | OF_PARTIAL},
	[O_ISQRT] = {"isqrt", 1, OF_INT},
	[O_EXP] = {"exp", 1, OF_REALFN},
	[O_LOG] = {"log", 1, OF_REALFN | OF_PARTIAL},
	[O_SIN] = {"sin", 1, OF_REALFN},
	[O_COS] = {"cos", 1, OF_REALFN},
	[O_TAN] = {"tan", 1, OF_REALFN | OF_PARTIAL},
	[O_FLOOR] = {"floor", 1, 0},
	[O_CEIL] = {"ceil", 1, 0},
	[O_CLZ] = {"clz", 1, OF_INT},
	[O_CTZ] = {"ctz", 1, OF_INT},
	[O_POPCNT] = {"popcount", 1, OF_INT},
	[O_CAST] = {"cast", 1, 0},
	[O_ADD] = {"+", 2, OF_COMM},
	[O_SUB] = {"-", 2, 0},
	[O_MUL] = {"*", 2, OF_COMM},
	[O_DIV] = {"/", 2, OF_PARTIAL},
	[O_MOD] = {"%", 2, OF_PARTIAL},
	[O_POW] = {"^", 2, 0},
	[O_AND] = {"&", 2, OF_INT | OF_COMM},
	[O_OR] = {"|", 2, OF_INT | OF_COMM},
	[O_XOR] = {"xor", 2, OF_INT | OF_COMM},
	[O_SHL] = {"<<", 2, OF_INT},
	[O_SHR] = {">>", 2, OF_INT},
	[O_MIN] = {"min", 2, OF_COMM},
	[O_MAX] = {"max", 2, OF_COMM},
	[O_EQ] = {"==", 2, OF_CMP | OF_COMM},
	[O_NE] = {"!=", 2, OF_CMP | OF_COMM},
	[O_LT] = {"<", 2, OF_CMP},
	[O_LE] = {"<=", 2, OF_CMP},
	[O_GT] = {">", 2, OF_CMP},
	[O_GE] = {">=", 2, OF_CMP},
	[O_LAND] = {"&&", 2, OF_COMM},
	[O_LOR] = {"||", 2, OF_COMM},
	[O_SEL] = {"?:", 3, 0},
	[O_TABLE] = {"table", 0, 0},
	[O_INDEX] = {"index", 2, OF_PARTIAL},
	[O_ASUINT] = {"asuint", 1, 0},
	[O_ASINT] = {"asint", 1, 0},
	[O_ASFLOAT] = {"asfloat", 1, 0},
	[O_BARRIER] = {"barrier", 1, 0},
	[O_ATOMIC] = {"atomic", 2, 0},
	[O_SEC] = {"sec", 1, OF_REALFN | OF_PARTIAL},
	[O_CSC] = {"csc", 1, OF_REALFN | OF_PARTIAL},
	[O_COT] = {"cot", 1, OF_REALFN | OF_PARTIAL},
	[O_ATAN] = {"atan", 1, OF_REALFN},
	[O_COPYSIGN] = {"copysign", 2, 0},
	[O_FMA] = {"fma", 3, 0},
	[O_LAMBERTW] = {"W", 1, OF_REALFN | OF_PARTIAL},
	[O_LAMBERTWM1] = {"Wm1", 1, OF_REALFN | OF_PARTIAL},
	[O_ASIN] = {"asin", 1, OF_REALFN | OF_PARTIAL},
	[O_ACOS] = {"acos", 1, OF_REALFN | OF_PARTIAL},
	[O_ATAN2] = {"atan2", 2, OF_PARTIAL},
	[O_SINH] = {"sinh", 1, OF_REALFN},
	[O_COSH] = {"cosh", 1, OF_REALFN},
	[O_TANH] = {"tanh", 1, OF_REALFN},
	[O_ASINH] = {"asinh", 1, OF_REALFN},
	[O_ACOSH] = {"acosh", 1, OF_REALFN | OF_PARTIAL},
	[O_ATANH] = {"atanh", 1, OF_REALFN | OF_PARTIAL},
	[O_LOOP] = {"loop", 2, 0},
	[O_CONJ] = {"conj", 1, OF_REALFN},
	[O_RE] = {"re", 1, OF_REALFN},
	[O_IM] = {"im", 1, OF_REALFN},
	[O_ARG] = {"arg", 1, OF_REALFN | OF_PARTIAL},
	[O_UDOT4X8] = {"udot4x8", 2, OF_INT},
	[O_SDOT4X8] = {"sdot4x8", 2, OF_INT},
	[O_SUDOT4X8] = {"sudot4x8", 2, OF_INT},
	[O_UDOT2X16] = {"udot2x16", 2, OF_INT},
	[O_SDOT2X16] = {"sdot2x16", 2, OF_INT},
	[O_SUDOT2X16] = {"sudot2x16", 2, OF_INT},
	[O_UDOT8X4] = {"udot8x4", 2, OF_INT},
	[O_SDOT8X4] = {"sdot8x4", 2, OF_INT},
	[O_SUDOT8X4] = {"sudot8x4", 2, OF_INT},
	[O_UDOT4X8S] = {"udot4x8sat", 3, OF_INT},
	[O_SDOT4X8S] = {"sdot4x8sat", 3, OF_INT},
	[O_SUDOT4X8S] = {"sudot4x8sat", 3, OF_INT},
	[O_UDOT2X16S] = {"udot2x16sat", 3, OF_INT},
	[O_SDOT2X16S] = {"sdot2x16sat", 3, OF_INT},
	[O_SUDOT2X16S] = {"sudot2x16sat", 3, OF_INT},
	[O_UDOT8X4S] = {"udot8x4sat", 3, OF_INT},
	[O_SDOT8X4S] = {"sdot8x4sat", 3, OF_INT},
	[O_SUDOT8X4S] = {"sudot8x4sat", 3, OF_INT},
	/* sg.c: the last operand is the subgroup's size */
	[O_SGBALLOT] = {"sg_ballot", 2, 0},
	[O_SGANY] = {"sg_any", 2, 0},
	[O_SGALL] = {"sg_all", 2, 0},
	[O_SGADD] = {"sg_add", 2, OF_INT},
	[O_SGSHUF] = {"sg_shuffle", 3, 0},
	[O_SGSCAN] = {"sg_scan", 3, OF_INT},
	[O_BFE] = {"bfe", 3, OF_INT},
	[O_SBFE] = {"sbfe", 3, OF_INT},
	[O_BREV] = {"brev", 1, OF_INT},
	[O_MULHI] = {"mulhi", 2, OF_INT},
	[O_SMULHI] = {"smulhi", 2, OF_INT},
	[O_FUNL] = {"funl", 3, OF_INT},
	[O_FUNR] = {"funr", 3, OF_INT},
	[O_MED3] = {"med3", 3, OF_INT},
	[O_USAD4X8] = {"usad4x8", 3, OF_INT},
	[O_BYTEPERM] = {"byteperm", 3, OF_INT},
	[O_BFI] = {"bfi", 3, OF_INT},
	[O_MULWIDE] = {"mulwide", 2, OF_INT},
	[O_MADWIDE] = {"madwide", 3, OF_INT},
	[O_BYTESEL] = {"bytesel", 3, OF_INT},
	[O_FETCH_REPEAT] = {"fetch_repeat", 2, 0},
	[O_FETCH_MIRROR] = {"fetch_mirror", 2, 0},
	[O_FETCH_CLAMP] = {"fetch_clamp", 2, 0},
	[O_FETCH_MCLAMP] = {"fetch_mirror_clamp", 2, 0},
	[O_FETCH_BORDER] = {"fetch_border", 3, 0},
	[O_TEX_LINEAR] = {"tex_linear", 2, 0},
	[O_TEX_BILINEAR] = {"tex_bilinear", 3, 0},
};

static const char *const atomic_names[] = {
	"add", "sub", "min", "max", "and", "or", "xor", "xchg",
};
static int64_t effect_serial;

struct nodes N;

/* ---- types ---- */

const char *
ty_str(ty_t t)
{
	static char buf[8][16];
	static int k;
	char *b = buf[k++ & 7];

	switch (TYK(t)) {
	case K_LIT:
	case K_INT: return "int";
	case K_REAL: return "real";
	case K_FLOAT: return TYW(t) == 32 ? "f32" : TYW(t) == 16 ? "f16" : "f64";
	case K_BOOL: return "bool";
	case K_UINT: snprintf(b, 16, "u%d", TYW(t)); return b;
	case K_SINT: snprintf(b, 16, "i%d", TYW(t)); return b;
	}
	return "?";
}

int
ty_parse(const char *s, size_t n, ty_t *out)
{
	char w[32];
	unsigned bits = 0;
	int k = 0;
	size_t i = 0;

	if (n == 0 || n >= sizeof w)
		return 0;
	memcpy(w, s, n);
	w[n] = 0;
	if (!strcmp(w, "real")) {
		*out = TY_REAL;
		return 1;
	}
	if (!strcmp(w, "f16")) {
		*out = TY_F16;
		return 1;
	}
	if (!strcmp(w, "float") || !strcmp(w, "f32")) {
		*out = TY_F32;
		return 1;
	}
	if (!strcmp(w, "double") || !strcmp(w, "f64")) {
		*out = TY_F64;
		return 1;
	}
	if (!strcmp(w, "int") || !strcmp(w, "integer")) {
		*out = TY_INT;
		return 1;
	}
	if (!strcmp(w, "bool")) {
		*out = TY_BOOL;
		return 1;
	}
	if (!strncmp(w, "uint", 4))
		k = K_UINT, i = 4;
	else if (!strncmp(w, "int", 3))
		k = K_SINT, i = 3;
	else if (w[0] == 'u')
		k = K_UINT, i = 1;
	else if (w[0] == 'i')
		k = K_SINT, i = 1;
	else
		return 0;
	if (w[i] < '1' || w[i] > '9')
		return 0;
	while (w[i] >= '0' && w[i] <= '9')
		bits = bits * 10 + (unsigned)(w[i++] - '0');
	if (i > 1 && w[0] != 'u' && w[0] != 'i')
		return 0;
	if (strcmp(w + i, "") != 0 && strcmp(w + i, "_t") != 0)
		return 0;
	if (bits < 1 || bits > 64)
		return 0;
	*out = TY(k, bits);
	return 1;
}

int64_t
int_norm(ty_t t, int64_t v)
{
	int w = TYW(t);
	uint64_t x, m;

	switch (TYK(t)) {
	case K_BOOL:
		return v & 1;
	case K_UINT:
		if (w >= 64)
			return v;
		return (int64_t)((uint64_t)v & ((1ULL << w) - 1));
	case K_SINT:
		if (w >= 64)
			return v;
		m = 1ULL << (w - 1);
		x = (uint64_t)v & ((1ULL << w) - 1);
		return (int64_t)((x ^ m) - m);
	}
	return v;
}

/* ---- the store ---- */

static uint32_t
nhash(int op, ty_t t, nid a, nid b, nid c, int64_t ki, double kr)
{
	uint64_t h = 1469598103934665603ULL, r;

	memcpy(&r, &kr, sizeof r);
	h = (h ^ (uint64_t)op) * 1099511628211ULL;
	h = (h ^ t) * 1099511628211ULL;
	h = (h ^ a) * 1099511628211ULL;
	h = (h ^ b) * 1099511628211ULL;
	h = (h ^ c) * 1099511628211ULL;
	h = (h ^ (uint64_t)ki) * 1099511628211ULL;
	h = (h ^ r) * 1099511628211ULL;
	return (uint32_t)(h ^ (h >> 32));
}

static void
ht_rehash(void)
{
	uint32_t i, j;

	free(N.ht);
	N.htcap = N.htcap ? N.htcap * 2 : 1024;
	N.ht = xcalloc(N.htcap, sizeof *N.ht);
	for (i = 1; i < N.n; i++) {
		j = nhash(N.op[i], N.ty[i], N.a[i], N.b[i], N.c[i], N.ki[i],
		    N.kr[i]) & (N.htcap - 1);
		while (N.ht[j])
			j = (j + 1) & (N.htcap - 1);
		N.ht[j] = i;
	}
}

static nid
intern(int op, ty_t t, nid a, nid b, nid c, int64_t ki, double kr, int fl)
{
	uint32_t j;
	nid n;

	if ((N.n + 1) * 2 > N.htcap)
		ht_rehash();
	j = nhash(op, t, a, b, c, ki, kr) & (N.htcap - 1);
	while ((n = N.ht[j]) != 0) {
		if (N.op[n] == op && N.ty[n] == t && N.a[n] == a &&
		    N.b[n] == b && N.c[n] == c && N.ki[n] == ki &&
		    memcmp(&N.kr[n], &kr, sizeof kr) == 0)
			return n;
		j = (j + 1) & (N.htcap - 1);
	}
	if (N.n == N.cap) {
		N.cap = newcap(N.cap, N.n + 1);
		N.op = xrealloc(N.op, N.cap);
		N.fl = xrealloc(N.fl, N.cap);
		N.ty = xrealloc(N.ty, N.cap * sizeof *N.ty);
		N.a = xrealloc(N.a, N.cap * sizeof *N.a);
		N.b = xrealloc(N.b, N.cap * sizeof *N.b);
		N.c = xrealloc(N.c, N.cap * sizeof *N.c);
		N.ki = xrealloc(N.ki, N.cap * sizeof *N.ki);
		N.kr = xrealloc(N.kr, N.cap * sizeof *N.kr);
	}
	n = N.n++;
	N.op[n] = (uint8_t)op;
	N.ty[n] = t;
	N.a[n] = a;
	N.b[n] = b;
	N.c[n] = c;
	N.ki[n] = ki;
	N.kr[n] = kr;
	N.fl[n] = (uint8_t)fl;
	N.ht[j] = n;
	return n;
}

void
nodes_init(void)
{
	if (N.n)
		return;
	N.cap = 1024;
	N.op = xcalloc(N.cap, 1);
	N.fl = xcalloc(N.cap, 1);
	N.ty = xcalloc(N.cap, sizeof *N.ty);
	N.a = xcalloc(N.cap, sizeof *N.a);
	N.b = xcalloc(N.cap, sizeof *N.b);
	N.c = xcalloc(N.cap, sizeof *N.c);
	N.ki = xcalloc(N.cap, sizeof *N.ki);
	N.kr = xcalloc(N.cap, sizeof *N.kr);
	N.n = 1; /* id 0 is "none" */
	ht_rehash();
}

int
is_const(nid n)
{
	return n && N.op[n] == O_CONST;
}

static double
const_val(nid n)
{
	if (TY_FP(N.ty[n]))
		return N.kr[n];
	if (TYK(N.ty[n]) == K_UINT && TYW(N.ty[n]) == 64)
		return (double)(uint64_t)N.ki[n];
	return (double)N.ki[n];
}

int
const_is(nid n, double v)
{
	return is_const(n) && const_val(n) == v;
}

static int
child_fl(nid a)
{
	return a ? N.fl[a] : 0;
}

static nid
mk_const_fp(ty_t t, double v)
{
	if (TYK(t) == K_FLOAT)
		return intern(O_CONST, t, 0, 0, 0, 0, fround(TYW(t), v), 0);
	return mk_const_r(v);
}

nid
mk_const_i(ty_t t, int64_t v)
{
	if (TY_FP(t))
		return mk_const_fp(t, (double)v);
	return intern(O_CONST, t, 0, 0, 0, int_norm(t, v), 0.0, 0);
}

nid
mk_const_r(double v)
{
	return intern(O_CONST, TY_REAL, 0, 0, 0, 0, v, 0);
}

nid
mk_var(uint32_t slot, uint32_t name, ty_t t)
{
	return intern(O_VAR, t, name, 0, 0, slot, 0.0, NF_VAR);
}

nid
sg_lane_copy(int slot, uint32_t name, ty_t t, int W)
{
	/* b: the W lanes it is read in */
	return intern(O_VAR, t, name, (nid)W, 0, SG_COPY_SLOT + slot, 0.0,
	    NF_VAR | NF_LANEREAD);
}

nid
mk_undef(ty_t t)
{
	return intern(O_UNDEF, t, 0, 0, 0, 0, 0.0, NF_PARTIAL);
}

/* calc: a node of an op the constructors here do not know (O_LOOP) */
nid
mk_raw(int op, ty_t t, nid a, nid b, nid c, int64_t ki, int fl)
{
	fl |= (child_fl(a) | child_fl(b) | child_fl(c)) & NF_INH;
	return intern(op, t, a, b, c, ki, 0.0, fl);
}

/* evaluation of constant operands lives with the evaluator */
int fold_const(int op, ty_t t, nid a, nid b, nid c, int64_t *ri, double *rr);

static nid
folded(int op, ty_t t, nid a, nid b, nid c)
{
	int64_t ri;
	double rr;

	if (!fold_const(op, t, a, b, c, &ri, &rr))
		return 0;
	return TY_FP(t) ? mk_const_fp(t, rr) : mk_const_i(t, ri);
}

static int
node_partial(int op, ty_t t, nid a, nid b, nid c)
{
	int fl = (child_fl(a) | child_fl(b) | child_fl(c)) & NF_PARTIAL;

	if (fl)
		return fl;
	if (TYK(t) == K_FLOAT || op == O_ASUINT || op == O_ASINT ||
	    op == O_ASFLOAT)
		return 0;   /* IEEE operations are total */
	if ((opinfo[op].fl & OF_CMP) && a && TYK(N.ty[a]) == K_FLOAT)
		return 0;
	switch (op) {
	case O_DIV:
	case O_MOD:
		if (is_const(b) && const_val(b) != 0 &&
		    !(TYK(t) == K_SINT && const_val(b) == -1))
			return 0;
		return NF_PARTIAL;
	case O_SHL:
	case O_SHR:
		if (is_const(b) && N.ki[b] >= 0 &&
		    (TYK(t) == K_INT || TYK(t) == K_LIT || N.ki[b] < TYW(t)))
			return 0;
		return NF_PARTIAL;
	case O_POW:
		if (TY_ISINT(t) && !(is_const(b) && N.ki[b] >= 0))
			return NF_PARTIAL;
		if (TYK(t) == K_REAL && !(is_const(b) &&
		    const_val(b) == floor(const_val(b))))
			return NF_PARTIAL;
		return 0;
	case O_ISQRT:
		return TYK(N.ty[a]) == K_SINT || TYK(N.ty[a]) == K_INT ||
		    TYK(N.ty[a]) == K_LIT ? NF_PARTIAL : 0;
	case O_CAST:
		return TY_FP(N.ty[a]) && TY_ISINT(t) ? NF_PARTIAL : 0;
	}
	return opinfo[op].fl & OF_PARTIAL ? NF_PARTIAL : 0;
}

static nid
node(int op, ty_t t, nid a, nid b, nid c)
{
	int fl = node_partial(op, t, a, b, c);

	fl |= (child_fl(a) | child_fl(b) | child_fl(c)) & NF_INH;
	return intern(op, t, a, b, c, 0, 0.0, fl);
}

/* ---- implicit conversions ---- */

static ty_t
common(ty_t x, ty_t y, int *mix)
{
	int kx = TYK(x), ky = TYK(y);

	if (x == y)
		return x;
	/* a machine float wins over everything; the wider float wins */
	if (kx == K_FLOAT && ky == K_FLOAT)
		return TYW(x) > TYW(y) ? x : y;
	if (kx == K_FLOAT)
		return x;
	if (ky == K_FLOAT)
		return y;
	if (kx == K_REAL || ky == K_REAL)
		return TY_REAL;
	if (kx == K_LIT)
		return y;
	if (ky == K_LIT)
		return x;
	if (kx == K_INT || ky == K_INT)
		return TY_INT;
	if (kx == K_BOOL)
		return y;
	if (ky == K_BOOL)
		return x;
	if (TYW(x) != TYW(y))
		return TYW(x) > TYW(y) ? x : y;
	*mix = 1;
	return TY(K_UINT, TYW(x));
}

static nid
conv(nid x, ty_t t)
{
	ty_t s = N.ty[x];
	int64_t v, nv;

	if (s == t)
		return x;
	if (N.op[x] == O_UNDEF)
		return mk_undef(t);
	if (N.op[x] == O_CONST) {
		if (TY_FP(t))
			return mk_const_fp(t, const_val(x));
		if (TY_FP(s)) {
			double r = N.kr[x];
			if (r == floor(r) && fabs(r) < 9.0e18)
				return mk_const_i(t, (int64_t)r);
			return mk_cast(t, x);
		}
		v = N.ki[x];
		nv = int_norm(t, v);
		if (nv != v && TYK(s) == K_LIT)
			note("literal %lld does not fit %s; it becomes %lld",
			    (long long)v, ty_str(t), (long long)nv);
		return mk_const_i(t, v);
	}
	return mk_cast(t, x);
}

nid
mk_cast(ty_t t, nid a)
{
	nid r;

	if (N.ty[a] == t)
		return a;
	if (N.op[a] == O_UNDEF)
		return mk_undef(t);
	if (TYK(t) == K_LIT)
		t = TY_INT;
	if (is_const(a) && (r = folded(O_CAST, t, a, 0, 0)) != 0)
		return r;
	return node(O_CAST, t, a, 0, 0);
}

nid
mk_bool(nid a)
{
	ty_t t = N.ty[a];

	if (t == TY_BOOL)
		return a;
	if (TY_FP(t))
		return mk_bin(O_NE, a, mk_const_fp(t, 0));
	return mk_bin(O_NE, a, mk_const_i(t, 0));
}

static int
negate_cmp(int op)
{
	switch (op) {
	case O_EQ: return O_NE;
	case O_NE: return O_EQ;
	case O_LT: return O_GE;
	case O_LE: return O_GT;
	case O_GT: return O_LE;
	case O_GE: return O_LT;
	}
	return 0;
}

nid
mk_un(int op, nid a)
{
	ty_t t = N.ty[a];
	nid r;

	if (IS_INST(op))
		return mk_inst(op, a, 0, 0);
	if (N.op[a] == O_UNDEF && op != O_LNOT)
		return a;
	switch (op) {
	case O_NEG:
		if (t == TY_BOOL)
			a = conv(a, TY_LIT), t = TY_LIT;
		if (N.op[a] == O_NEG)
			return N.a[a];
		if (TYK(t) == K_REAL && N.op[a] == O_SUB)
			return mk_bin(O_SUB, N.b[a], N.a[a]);
		break;
	case O_NOT:
		if (TY_FP(t))
			fail("~ needs an integer operand (use asuint(x) for "
			    "the bits of a float)");
		if (N.op[a] == O_NOT)
			return N.a[a];
		break;
	case O_LNOT:
		a = mk_bool(a);
		t = TY_BOOL;
		if (N.op[a] == O_LNOT)
			return N.a[a];
		if ((opinfo[N.op[a]].fl & OF_CMP) &&
		    !TY_FP(N.ty[N.a[a]]))
			return mk_bin(negate_cmp(N.op[a]), N.a[a], N.b[a]);
		break;
	case O_ABS:
	case O_SIGN:
		if (t == TY_BOOL)
			return op == O_ABS ? a : a;
		break;
	case O_FLOOR:
	case O_CEIL:
		if (TY_ISINT(t))
			return a;
		break;
	case O_ISQRT:
	case O_CLZ:
	case O_CTZ:
	case O_POPCNT:
		if (TY_FP(t))
			fail("%s needs an integer operand", opinfo[op].name);
		break;
	case O_ASUINT:
	case O_ASINT:
		if (TYK(t) == K_REAL || TYK(t) == K_LIT) {
			if (!is_const(a))
				fail("%s needs an f16, f32 or f64 operand (real is "
				    "the mathematical line and has no bits)",
				    opinfo[op].name);
			note("constant read as f32 for %s", opinfo[op].name);
			a = conv(a, TY_F32);
			t = TY_F32;
		}
		if (TYK(t) != K_FLOAT)
			fail("%s needs an f16, f32 or f64 operand",
			    opinfo[op].name);
		t = TY(op == O_ASUINT ? K_UINT : K_SINT, TYW(t));
		if (is_const(a) && (r = folded(op, t, a, 0, 0)) != 0)
			return r;
		return node(op, t, a, 0, 0);
	case O_ASFLOAT:
		if (TY_FP(t))
			fail("asfloat needs integer bits");
		if (TYK(t) == K_LIT || TYK(t) == K_INT) {
			int w = is_const(a) && N.ki[a] >= 0 &&
			    N.ki[a] <= 0xffffffffLL ? 32 : 64;
			note("asfloat of an untyped integer: read as u%d -> f%d",
			    w, w);
			a = conv(a, TY(K_UINT, w));
			t = N.ty[a];
		}
		if (TYW(t) != 16 && TYW(t) != 32 && TYW(t) != 64)
			fail("asfloat needs 16, 32 or 64 bits, not %s",
			    ty_str(t));
		t = TY(K_FLOAT, TYW(t));
		if (is_const(a) && (r = folded(op, t, a, 0, 0)) != 0)
			return r;
		return node(op, t, a, 0, 0);
	default:
		if ((opinfo[op].fl & OF_REALFN) && TYK(t) == K_FLOAT)
			break;
		if (opinfo[op].fl & OF_REALFN) {
			a = conv(a, TY_REAL);
			t = TY_REAL;
		}
	}
	if (is_const(a) && (r = folded(op, t, a, 0, 0)) != 0)
		return r;
	return node(op, t, a, 0, 0);
}

nid
mk_bin(int op, nid a, nid b)
{
	ty_t ta = N.ty[a], tb = N.ty[b], t, rt;
	int mix = 0;
	nid r, x;

	if (IS_DOT(op))
		return mk_dot(op, a, b, 0);
	if (IS_SG(op))
		return mk_sg(op, a, b, 0);
	if (IS_INST(op))
		return mk_inst(op, a, b, 0);
	if (IS_TEX(op)) {
		/* a rebuilder that knows neither the node's grid nor its
		 * third operand: mk_like() keeps both */
		if (op == O_FETCH_BORDER || !IS_FETCH(op))
			fail("%s cannot be rebuilt from two operands",
			    opinfo[op].name);
		return mk_tex(op, 0, a, b, 0);
	}
	switch (op) {
	case O_LAND:
	case O_LOR:
		a = mk_bool(a);
		b = mk_bool(b);
		t = rt = TY_BOOL;
		if (is_const(a)) {
			if ((op == O_LAND) == (N.ki[a] != 0))
				return b;
			return a;
		}
		if (is_const(b) && !(N.fl[a] & NF_PARTIAL)) {
			if ((op == O_LAND) == (N.ki[b] != 0))
				return a;
			return b;
		}
		break;
	case O_SHL:
	case O_SHR:
		if (!TY_ISINT(ta) || !TY_ISINT(tb))
			fail("shift needs integer operands");
		if (TYK(ta) == K_LIT && TY_SIZED(tb) && TYK(tb) != K_BOOL)
			a = conv(a, tb), ta = tb;
		if (ta == TY_BOOL)
			a = conv(a, TY_LIT), ta = TY_LIT;
		t = rt = ta;
		break;
	case O_POW:
		if (TY_FP(ta) || TY_FP(tb) ||
		    (TYK(ta) == K_LIT && is_const(b) && N.ki[b] < 0)) {
			t = common(ta, tb, &mix);
			if (!TY_FP(t))
				t = TY_REAL;
			a = conv(a, t);
			b = conv(b, t);
			rt = t;
		} else {
			if (TYK(ta) == K_LIT && TY_SIZED(tb) && tb != TY_BOOL)
				a = conv(a, tb), ta = tb;
			if (ta == TY_BOOL)
				a = conv(a, TY_LIT), ta = TY_LIT;
			t = rt = ta;
		}
		break;
	default:
		if (op == O_DIV && is_const(a) && is_const(b) &&
		    TYK(ta) == K_LIT && TYK(tb) == K_LIT && N.ki[b] != 0 &&
		    N.ki[a] % N.ki[b] != 0) {
			ta = tb = TY_REAL;
			a = conv(a, TY_REAL);
			b = conv(b, TY_REAL);
		}
		t = common(ta, tb, &mix);
		if ((op == O_COPYSIGN || op == O_ATAN2) && !TY_FP(t))
			t = TY_REAL;
		if (t == TY_BOOL && !(opinfo[op].fl & OF_CMP) &&
		    op != O_AND && op != O_OR && op != O_XOR &&
		    op != O_MIN && op != O_MAX)
			t = TY_LIT;
		if (mix)
			note("mixing %s and %s: computed as %s", ty_str(ta),
			    ty_str(tb), ty_str(t));
		if ((opinfo[op].fl & OF_INT) && TY_FP(t))
			fail("'%s' needs integer operands (declare the "
			    "parameter as int or uN)", opinfo[op].name);
		a = conv(a, t);
		b = conv(b, t);
		rt = (opinfo[op].fl & OF_CMP) ? TY_BOOL : t;
	}
	if (N.op[a] == O_UNDEF && op != O_LAND && op != O_LOR)
		return mk_undef(rt);
	if (N.op[b] == O_UNDEF && op != O_LAND && op != O_LOR)
		return mk_undef(rt);

	/* constants on a canonical side: c*x, x+c, x&c */
	if ((opinfo[op].fl & OF_COMM) && op != O_MUL && is_const(a) &&
	    !is_const(b))
		x = a, a = b, b = x;
	if (op == O_MUL && is_const(b) && !is_const(a))
		x = a, a = b, b = x;

	if (is_const(a) && is_const(b) &&
	    (r = folded(op, rt, a, b, 0)) != 0)
		return r;

#define TOTAL(n) (!(N.fl[n] & NF_PARTIAL))
	/*
	 * IEEE floats: x+0, 0-x, x-x, x*0 are not identities (-0, NaN, inf),
	 * and nothing reassociates.  Only exact identities are applied.
	 */
	if (TYK(t) == K_FLOAT) {
		if ((op == O_MUL && const_is(a, 1)) ||
		    (op == O_DIV && const_is(b, 1)) ||
		    (op == O_SUB && const_is(b, 0) && !signbit(N.kr[b])))
			return op == O_MUL ? b : a;
		return node(op, rt, a, b, 0);
	}
	switch (op) {
	case O_ADD:
		if (const_is(b, 0))
			return a;
		if (N.op[b] == O_NEG)
			return mk_bin(O_SUB, a, N.a[b]);
		if (N.op[a] == O_NEG)
			return mk_bin(O_SUB, b, N.a[a]);
		break;
	case O_SUB:
		if (const_is(b, 0))
			return a;
		if (const_is(a, 0))
			return mk_un(O_NEG, b);
		if (a == b && TOTAL(a))
			return mk_const_i(t, 0);
		if (N.op[b] == O_NEG)
			return mk_bin(O_ADD, a, N.a[b]);
		break;
	case O_MUL:
		if (const_is(a, 1))
			return b;
		if (const_is(a, 0) && TOTAL(b))
			return a;
		if (const_is(a, -1) && TYK(t) != K_UINT)
			return mk_un(O_NEG, b);
		if (TYK(t) == K_REAL && is_const(a)) {
			if (N.op[b] == O_MUL && is_const(N.a[b]))
				return mk_bin(O_MUL, mk_const_r(N.kr[a] *
				    N.kr[N.a[b]]), N.b[b]);
			if (N.op[b] == O_NEG)
				return mk_bin(O_MUL, mk_const_r(-N.kr[a]),
				    N.a[b]);
		}
		if (TYK(t) == K_REAL && N.op[a] == O_NEG)
			return mk_un(O_NEG, mk_bin(O_MUL, N.a[a], b));
		if (TYK(t) == K_REAL && N.op[b] == O_NEG)
			return mk_un(O_NEG, mk_bin(O_MUL, a, N.a[b]));
		break;
	case O_DIV:
		if (const_is(b, 1))
			return a;
		break;
	case O_POW:
		if (const_is(b, 1))
			return a;
		if (const_is(b, 0) && TOTAL(a))
			return mk_const_i(t, 1);
		break;
	case O_AND:
	case O_OR:
	case O_MIN:
	case O_MAX:
		if (a == b)
			return a;
		if (op == O_AND && const_is(b, 0) && TOTAL(a))
			return b;
		if (op == O_OR && const_is(b, 0))
			return a;
		break;
	case O_XOR:
		if (const_is(b, 0))
			return a;
		if (a == b && TOTAL(a))
			return mk_const_i(t, 0);
		break;
	case O_SHL:
	case O_SHR:
		if (const_is(b, 0))
			return a;
		break;
	}
#undef TOTAL
	return node(op, rt, a, b, 0);
}

/* fma(a, b, c) = a*b + c with one rounding */
nid
mk_fma(nid a, nid b, nid c)
{
	int mix = 0;
	ty_t t = common(common(N.ty[a], N.ty[b], &mix), N.ty[c], &mix);
	nid r;

	if (!TY_FP(t))
		t = TY_REAL;
	a = conv(a, t);
	b = conv(b, t);
	c = conv(c, t);
	if (N.op[a] == O_UNDEF || N.op[b] == O_UNDEF || N.op[c] == O_UNDEF)
		return mk_undef(t);
	if (is_const(a) && is_const(b) && is_const(c) &&
	    (r = folded(O_FMA, t, a, b, c)) != 0)
		return r;
	return node(O_FMA, t, a, b, c);
}

/*
 * A packed dot product (dot.c): a and b are u32 words of lanes, the
 * result is u32 (unsigned lanes) or i32; the _SAT forms add acc, of the
 * result's type, saturating.  Constant operands fold.
 */
nid
mk_dot(int op, nid a, nid b, nid acc)
{
	const struct dotform *df = dot_form(op);
	ty_t t = df->sa || df->sb ? TY(K_SINT, 32) : TY(K_UINT, 32);
	nid r;

	if (!TY_ISINT(N.ty[a]) || !TY_ISINT(N.ty[b]) ||
	    (df->sat && !TY_ISINT(N.ty[acc])))
		fail("%s takes integer operands (packed lanes in a u32)",
		    df->name);
	a = mk_cast(TY(K_UINT, 32), a);
	b = mk_cast(TY(K_UINT, 32), b);
	acc = df->sat ? mk_cast(t, acc) : 0;
	if (N.op[a] == O_UNDEF || N.op[b] == O_UNDEF ||
	    (acc && N.op[acc] == O_UNDEF))
		return mk_undef(t);
	if (is_const(a) && is_const(b) && (!acc || is_const(acc)) &&
	    (r = folded(op, t, a, b, acc)) != 0)
		return r;
	return node(op, t, a, b, acc);
}

/*
 * A read through the texture unit (texalu.h has the semantics): a is the
 * table; a fetch_ read takes an integer index b (and the border's value
 * c), a filter the coordinates b (and c), its texels and coordinates
 * f32.  ki is TEX_KI(bits, width).  Constant operands fold.
 */
static double
tex_const_texel(void *ctx, int64_t j)
{
	nid a = *(nid *)ctx;

	return N.kr[N.lst[N.a[a] + (uint32_t)j]];
}

nid
mk_tex(int op, int64_t ki, nid a, nid b, nid c)
{
	uint32_t n, j;
	int fl, bits = TEX_BITS(ki), allk = 1;
	int64_t w = TEX_WIDTH(ki);

	if (N.op[a] != O_TABLE || N.b[a] == 0)
		fail("%s takes a table [t0, t1, ...] first", opinfo[op].name);
	n = N.b[a];
	if (IS_FETCH(op)) {
		if (!TY_ISINT(N.ty[b]))
			fail("%s takes an integer texel index, not %s",
			    opinfo[op].name, ty_str(N.ty[b]));
		c = op == O_FETCH_BORDER ? conv(c, N.ty[a]) : 0;
		if (N.op[b] == O_UNDEF)
			return mk_undef(N.ty[a]);
		if (is_const(b)) {
			int64_t t = tex_wrap(op, N.ki[b], n);
			return t < 0 ? c : N.lst[N.a[a] + (uint32_t)t];
		}
		fl = (N.fl[a] | N.fl[b] | child_fl(c)) & (NF_PARTIAL | NF_VAR);
		return intern(op, N.ty[a], a, b, c, 0, 0.0, fl);
	}
	if (bits < 1 || bits > TEX_MAXBITS)
		fail("%s: the sub-texel grid has 1 to %d bits, not %d",
		    opinfo[op].name, TEX_MAXBITS, bits);
	if (op == O_TEX_BILINEAR) {
		if (w < 1 || n % w)
			fail("tex_bilinear: a table of %u texels is not rows of "
			    "%lld", n, (long long)w);
	} else {
		w = 0;
		c = 0;
	}
	if (N.ty[a] != TY_F32) {
		nid *e = xmalloc(n * sizeof *e);
		for (j = 0; j < n; j++)
			e[j] = mk_cast(TY_F32, N.lst[N.a[a] + j]);
		a = mk_table(e, n);
		free(e);
		if (N.ty[a] != TY_F32)
			fail("%s takes numbers as texels", opinfo[op].name);
	}
	b = mk_cast(TY_F32, b);
	if (c)
		c = mk_cast(TY_F32, c);
	if (N.op[b] == O_UNDEF || (c && N.op[c] == O_UNDEF))
		return mk_undef(TY_F32);
	for (j = 0; j < n; j++)
		allk &= is_const(N.lst[N.a[a] + j]);
	if (allk && is_const(b) && (!c || is_const(c)))
		return mk_const_fp(TY_F32, tex_filter(op, TEX_KI(bits, w), n,
		    N.kr[b], c ? N.kr[c] : 0, tex_const_texel, &a));
	fl = (N.fl[a] | N.fl[b] | child_fl(c)) & (NF_PARTIAL | NF_VAR);
	return intern(op, TY_F32, a, b, c, TEX_KI(bits, w), 0.0, fl);
}

nid
mk_like(nid n, nid a, nid b, nid c)
{
	if (IS_TEX(N.op[n]))
		return mk_tex(N.op[n], N.ki[n], a, b, c);
	return mk_op(N.op[n], a, b, c);
}

/* any operation from its operands (the generic rebuilders) */
nid
mk_op(int op, nid a, nid b, nid c)
{
	if (IS_DOT(op))
		return mk_dot(op, a, b, c);
	if (IS_SG(op))
		return mk_sg(op, a, b, c);
	if (IS_INST(op))
		return mk_inst(op, a, b, c);
	if (op == O_SEL)
		return mk_sel(a, b, c);
	if (op == O_FMA)
		return mk_fma(a, b, c);
	if (opinfo[op].arity == 1)
		return mk_un(op, a);
	return mk_bin(op, a, b);
}

nid
mk_sel(nid c, nid a, nid b)
{
	int mix = 0;
	ty_t t;

	c = mk_bool(c);
	if (N.op[a] == O_UNDEF && N.op[b] != O_UNDEF)
		a = mk_undef(N.ty[b]);
	if (N.op[b] == O_UNDEF && N.op[a] != O_UNDEF)
		b = mk_undef(N.ty[a]);
	t = common(N.ty[a], N.ty[b], &mix);
	if (t == TY_BOOL && (N.ty[a] != TY_BOOL || N.ty[b] != TY_BOOL))
		t = TY_LIT;
	a = conv(a, t);
	b = conv(b, t);
	if (is_const(c))
		return N.ki[c] ? a : b;
	if (a == b && !(N.fl[c] & NF_PARTIAL))
		return a;
	return node(O_SEL, t, c, a, b);
}

/*
 * Effects.  The value semantics are those of one invocation: barrier(e)
 * is e, an atomic returns the old value.  Each one is a distinct node so
 * hash-consing never merges two of them.
 */
nid
mk_barrier(nid a)
{
	return intern(O_BARRIER, N.ty[a], a, 0, 0, ++effect_serial, 0.0,
	    N.fl[a] & (NF_PARTIAL | NF_INH));
}

nid
mk_atomic(int kind, nid a, nid v)
{
	int mix = 0;
	ty_t t = common(N.ty[a], N.ty[v], &mix);
	int k = kind & 15;

	if (TY_FP(t) && k != AT_ADD && k != AT_XCHG && k != AT_MIN &&
	    k != AT_MAX)
		fail("atomic_%s needs integer operands", atomic_names[k]);
	a = conv(a, t);
	v = conv(v, t);
	return intern(O_ATOMIC, t, a, v, 0,
	    (int64_t)kind | (++effect_serial << 8), 0.0,
	    (N.fl[a] | N.fl[v]) & (NF_PARTIAL | NF_INH));
}

int
atomic_lookup(const char *s, size_t n, int *kind)
{
	size_t i, l;

	if (n < 8 || memcmp(s, "atomic_", 7) != 0)
		return 0;
	s += 7;
	n -= 7;
	*kind = 0;
	if (n > 7 && memcmp(s + n - 7, "_shared", 7) == 0) {
		*kind = AT_SHARED;
		n -= 7;
	}
	for (i = 0; i < sizeof atomic_names / sizeof atomic_names[0]; i++) {
		l = strlen(atomic_names[i]);
		if (l == n && memcmp(s, atomic_names[i], n) == 0) {
			*kind |= (int)i;
			return 1;
		}
	}
	return 0;
}

/* tables by their entries: open addressing over the table nodes */
static uint32_t *tabht, tabcap, tabn;
static uint32_t *tabhash, tabhcap;	/* by node */

nid
mk_table(const nid *e, uint32_t n)
{
	ty_t t = TY_LIT;
	uint32_t i, st, j, h = 2166136261u;
	int mix = 0, fl = 0;
	nid r;

	for (i = 0; i < n; i++)
		t = i ? common(t, N.ty[e[i]], &mix) : N.ty[e[i]];
	if (t == TY_BOOL)
		t = TY_LIT;
	st = N.nlst;
	if (N.nlst + n > N.lstcap) {
		N.lstcap = newcap(N.lstcap, N.nlst + n);
		N.lst = xrealloc(N.lst, N.lstcap * sizeof *N.lst);
	}
	for (i = 0; i < n; i++) {
		N.lst[st + i] = conv(e[i], t);
		fl |= N.fl[N.lst[st + i]];
		h = (h ^ N.lst[st + i]) * 16777619u;
	}
	/*
	 * A table with these entries already: the same node, as for any
	 * other expression (a rebuilt copy of a table is that table, and
	 * two lookups of it at one index are one load).
	 */
	h = (h ^ t ^ (n << 16)) * 16777619u;
	if ((tabn + 1) * 2 > tabcap) {
		uint32_t oc = tabcap, *o = tabht, q;
		tabcap = tabcap ? tabcap * 2 : 256;
		tabht = xcalloc(tabcap, sizeof *tabht);
		for (q = 0; q < oc; q++)
			if (o[q]) {
				j = tabhash[o[q]] & (tabcap - 1);
				while (tabht[j])
					j = (j + 1) & (tabcap - 1);
				tabht[j] = o[q];
			}
		free(o);
	}
	for (j = h & (tabcap - 1); (r = tabht[j]) != 0; j = (j + 1) & (tabcap - 1))
		if (tabhash[r] == h && N.ty[r] == t && N.b[r] == n &&
		    !memcmp(&N.lst[N.a[r]], &N.lst[st], n * sizeof *N.lst))
			return r;
	N.nlst += n;
	r = intern(O_TABLE, t, st, n, 0, 0, 0.0,
	    fl & (NF_PARTIAL | NF_INH));
	if (r >= tabhcap) {
		uint32_t c = newcap(tabhcap, r + 1);
		tabhash = xrealloc(tabhash, c * sizeof *tabhash);
		memset(tabhash + tabhcap, 0, (c - tabhcap) * sizeof *tabhash);
		tabhcap = c;
	}
	tabhash[r] = h;
	tabht[j] = r;
	tabn++;
	return r;
}

nid
mk_index(nid tab, nid i)
{
	int fl;

	if (N.op[tab] != O_TABLE)
		fail("only a table [a, b, ...] can be indexed");
	if (!TY_ISINT(N.ty[i]))
		fail("table index must be an integer");
	if (is_const(i)) {
		if (N.ki[i] >= 0 && (uint64_t)N.ki[i] < N.b[tab])
			return N.lst[N.a[tab] + (uint32_t)N.ki[i]];
		return mk_undef(N.ty[tab]);
	}
	fl = NF_PARTIAL | ((N.fl[tab] | N.fl[i]) & NF_INH);
	return intern(O_INDEX, N.ty[tab], tab, i, 0, 0, 0.0, fl);
}

/* ---- substitution ---- */

static nid *smemo;
static uint32_t smemon;
/* sg_lower: subgroup operations are spelled out over their lanes, and
 * what has none is left as it is */
static int sg_lowering;
/* sg_flat: the bound lane is replaced inside the operations too */
static int sg_capture;

static nid
rebuild(nid n, const nid *map, int nmap)
{
	nid r = 0, *e;
	uint32_t i;

	if (n == 0 || n >= smemon)
		return n;
	if (sg_lowering && !(N.fl[n] & NF_SG))
		return n;
	if (smemo[n])
		return smemo[n];
	switch (N.op[n]) {
	case O_CONST:
	case O_UNDEF:
		r = n;
		break;
	case O_VAR:
		r = n;
		if (N.ki[n] < nmap && map[N.ki[n]])
			r = map[N.ki[n]];
		break;
	case O_TABLE:
		e = xmalloc(N.b[n] * sizeof *e + 1);
		for (i = 0; i < N.b[n]; i++)
			e[i] = rebuild(N.lst[N.a[n] + i], map, nmap);
		r = mk_table(e, N.b[n]);
		free(e);
		break;
	case O_SEL:
		r = mk_sel(rebuild(N.a[n], map, nmap),
		    rebuild(N.b[n], map, nmap), rebuild(N.c[n], map, nmap));
		break;
	case O_CAST:
		r = mk_cast(N.ty[n], rebuild(N.a[n], map, nmap));
		break;
	case O_INDEX:
		r = mk_index(rebuild(N.a[n], map, nmap),
		    rebuild(N.b[n], map, nmap));
		break;
	case O_BARRIER:
		r = mk_barrier(rebuild(N.a[n], map, nmap));
		break;
	case O_ATOMIC:
		r = mk_atomic((int)(N.ki[n] & 0xff), rebuild(N.a[n], map, nmap),
		    rebuild(N.b[n], map, nmap));
		break;
	case O_FMA:
		r = mk_fma(rebuild(N.a[n], map, nmap),
		    rebuild(N.b[n], map, nmap), rebuild(N.c[n], map, nmap));
		break;
	case O_LOOP:
		/* spelling out stops at a loop: its body is spelled out
		 * where the loop is run (plan_build takes sg_lower again) */
		if (sg_lowering) {
			r = n;
			break;
		}
		r = loop_remake(n, rebuild(N.a[n], map, nmap),
		    rebuild(N.b[n], map, nmap), rebuild(N.c[n], map, nmap), map,
		    nmap);
		break;
	default:
		if (IS_TEX(N.op[n]))
			r = mk_tex(N.op[n], N.ki[n], rebuild(N.a[n], map, nmap),
			    rebuild(N.b[n], map, nmap), rebuild(N.c[n], map, nmap));
		else if (IS_SG(N.op[n])) {
			/* the operands first: an operation inside one is
			 * spelled out before the one around it */
			nid a, b, c;
			int low = sg_lowering;
			if (nmap > SG_LANE_SLOT && map[SG_LANE_SLOT] &&
			    !sg_capture) {
				/* the lane of its value per lane is its own */
				nid m2[SG_LANE_SLOT + 1];
				memcpy(m2, map, sizeof m2);
				m2[SG_LANE_SLOT] = 0;
				a = subst(N.a[n], m2, SG_LANE_SLOT + 1);
			} else
				a = rebuild(N.a[n], map, nmap);
			b = rebuild(N.b[n], map, nmap);
			c = rebuild(N.c[n], map, nmap);
			sg_lowering = 0;
			r = low ? sg_expand(N.op[n], a, b, c) :
			    mk_sg(N.op[n], a, b, c);
			sg_lowering = low;
		} else if (IS_INST(N.op[n])) {
			nid a = rebuild(N.a[n], map, nmap);
			nid b = rebuild(N.b[n], map, nmap);
			nid c = rebuild(N.c[n], map, nmap);
			r = sg_lowering ? inst_expand(N.op[n], a, b, c) :
			    mk_inst(N.op[n], a, b, c);
		} else if (IS_DOT(N.op[n]))
			r = mk_dot(N.op[n], rebuild(N.a[n], map, nmap),
			    rebuild(N.b[n], map, nmap), rebuild(N.c[n], map, nmap));
		else if (opinfo[N.op[n]].arity == 1)
			r = mk_un(N.op[n], rebuild(N.a[n], map, nmap));
		else
			r = mk_bin(N.op[n], rebuild(N.a[n], map, nmap),
			    rebuild(N.b[n], map, nmap));
	}
	smemo[n] = r;
	return r;
}

nid
subst(nid root, const nid *map, int nmap)
{
	nid *saved = smemo, r;
	uint32_t savedn = smemon;

	smemon = N.n;
	smemo = xcalloc(smemon, sizeof *smemo);
	r = rebuild(root, map, nmap);
	free(smemo);
	smemo = saved;
	smemon = savedn;
	return r;
}

/* every subgroup operation of root spelled out (sg.c keeps the results) */
nid
sg_lower_once(nid root)
{
	int low = sg_lowering;
	nid r;

	sg_lowering = 1;
	r = subst(root, NULL, 0);
	sg_lowering = low;
	return r;
}

/*
 * e as one lane runs it, for pricing: the bound lane of every operation
 * is the lane itself (to), so a value computed in each lane is one node
 * whether an operation reads it across the lanes or the lane uses it.
 */
nid
sg_flat(nid e, nid to)
{
	nid map[SG_LANE_SLOT + 1], r;
	int cap = sg_capture;

	if (e == 0 || !(N.fl[e] & NF_SG))
		return e;
	memset(map, 0, sizeof map);
	map[SG_LANE_SLOT] = to;
	sg_capture = 1;
	r = subst(e, map, SG_LANE_SLOT + 1);
	sg_capture = cap;
	return r;
}

/* ---- printing ---- */

static const char *sh_name(const struct pctx *pc, nid n);

static int
is_jux(nid n, const struct pctx *pc)
{
	nid a, b;

	if (pc->mode != PM_UCAS || N.op[n] != O_MUL)
		return 0;
	a = N.a[n];
	b = N.b[n];
	if (sh_name(pc, b))
		return 0;
	if (!is_const(a) || const_val(a) < 0 || is_const(b))
		return 0;
	if (TYK(N.ty[a]) == K_REAL && (N.kr[a] == M_PI || N.kr[a] == M_E))
		return 0;
	if (N.op[b] == O_NEG || (N.op[b] == O_MUL && is_const(N.a[b])))
		return 0;
	while (N.op[b] == O_CAST)
		b = N.a[b];
	if (N.op[b] == O_POW && is_const(N.a[b]))   /* 2*3^x is not 23^x */
		return 0;
	return 1;
}

static int
prec(nid n, const struct pctx *pc)
{
	if (sh_name(pc, n))
		return 15;
	switch (N.op[n]) {
	case O_SEL: return 1;
	case O_LOR: return 2;
	case O_LAND: return 3;
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE:
		return 4;
	case O_OR: return 5;
	case O_XOR: return 6;
	case O_AND: return 7;
	case O_SHL: case O_SHR: return 8;
	case O_ADD: case O_SUB: return 9;
	case O_MUL: return is_jux(n, pc) ? 12 : 10;
	case O_DIV: case O_MOD: return 10;
	case O_NEG: case O_NOT: case O_LNOT: return 11;
	case O_POW: return 13;
	case O_CONST:
		return const_val(n) < 0 ? 11 : 15;
	}
	return 15;
}

static void pr(struct sbuf *b, nid n, int minp, const struct pctx *pc,
    int hex);
static int pr_bitwise;   /* the expression being printed uses & | << ... */

static void
pr_const(struct sbuf *b, nid n, int hex)
{
	ty_t t = N.ty[n];
	double v;

	if (TYK(t) == K_FLOAT) {
		sb_printf(b, "%.*g", fdigits(TYW(t)), N.kr[n]);
		return;
	}
	if (TYK(t) == K_REAL) {
		v = N.kr[n];
		if (v == M_PI)
			sb_puts(b, "pi");
		else if (v == M_E)
			sb_puts(b, "e");
		else if (v == floor(v) && fabs(v) >= 1e10 && fabs(v) < 9007199254740992.0)
			sb_printf(b, "%.0f", v);   /* integers stay exact */
		else
			sb_printf(b, "%.10g", v);
		return;
	}
	if (TYK(t) == K_UINT && TYW(t) == 64) {
		if (hex && (uint64_t)N.ki[n] >= 16)
			sb_printf(b, "0x%llx", (unsigned long long)N.ki[n]);
		else
			sb_printf(b, "%llu", (unsigned long long)N.ki[n]);
		return;
	}
	if (hex && N.ki[n] >= 16)
		sb_printf(b, "0x%llx", (unsigned long long)N.ki[n]);
	else
		sb_printf(b, "%lld", (long long)N.ki[n]);
}

static const char *
c_type(ty_t t)
{
	static char buf[4][16];
	static int k;
	char *s = buf[k++ & 3];
	int w = TYW(t);

	switch (TYK(t)) {
	case K_UINT:
	case K_SINT:
		w = w <= 8 ? 8 : w <= 16 ? 16 : w <= 32 ? 32 : 64;
		snprintf(s, 16, "%sint%d_t", TYK(t) == K_UINT ? "u" : "", w);
		return s;
	case K_REAL: return "double";
	case K_FLOAT:
		return TYW(t) == 32 ? "float" : TYW(t) == 16 ? "_Float16" :
		    "double";
	case K_BOOL: return "_Bool";
	}
	return "int64_t";
}

static void
pr_bin(struct sbuf *b, nid n, const char *opstr, int p, int lp, int rp,
    const struct pctx *pc, int hex)
{
	pr(b, N.a[n], lp, pc, hex);
	sb_printf(b, "%s", opstr);
	pr(b, N.b[n], rp, pc, hex);
	(void)p;
}

/* v is exactly p/q (as a double) with 2 <= q <= 12 */
static int
small_frac(double v, long *p, long *q)
{
	long k;

	if (v == floor(v) || fabs(v) > 1e6)
		return 0;
	for (k = 2; k <= 12; k++) {
		double t = nearbyint(v * k);
		if (t / k == v) {
			*p = (long)t;
			*q = k;
			return 1;
		}
	}
	return 0;
}

static void
pr(struct sbuf *b, nid n, int minp, const struct pctx *pc, int hex)
{
	long fp, fq;
	int p = prec(n, pc), op = N.op[n], c = pc->mode != PM_UCAS;
	int bits = pc->mode == PM_CBITS;
	uint32_t i;
	const char *s;

	if ((s = sh_name(pc, n)) != NULL) {
		sb_puts(b, s);
		return;
	}
	if (p < minp)
		sb_putc(b, '(');
	switch (op) {
	case O_CONST:
		pr_const(b, n, hex);
		break;
	case O_VAR:
		if (pc->vn && pc->vn[N.ki[n]])
			sb_puts(b, pc->vn[N.ki[n]]);
		else
			sb_puts(b, sym_str(N.a[n]));
		break;
	case O_UNDEF:
		sb_puts(b, "undefined");
		break;
	case O_NEG:
		/* -(a*b) and -(a/b) read the same as (-a)*b: -x*sin(y), -1/x */
		sb_putc(b, '-');
		pr(b, N.a[n], TY_FP(N.ty[n]) && (N.op[N.a[n]] == O_MUL ||
		    N.op[N.a[n]] == O_DIV) ? 10 : 11, pc, 0);
		break;
	case O_NOT:
		sb_putc(b, '~');
		pr(b, N.a[n], 11, pc, 0);
		break;
	case O_LNOT:
		sb_putc(b, bits ? '~' : '!');
		pr(b, N.a[n], 11, pc, 0);
		break;
	case O_CAST:
		if (!c && TYK(N.ty[N.a[n]]) == K_LIT && TYK(N.ty[n]) == K_INT) {
			/* an untyped integer to int: nothing changes */
			pr(b, N.a[n], minp, pc, hex);
			break;
		}
		if (c)
			sb_printf(b, "(%s)", c_type(N.ty[n]));
		else
			sb_puts(b, ty_str(N.ty[n]));
		sb_putc(b, '(');
		pr(b, N.a[n], 0, pc, 0);
		sb_putc(b, ')');
		break;
	case O_ADD:
		if (is_const(N.b[n]) && const_val(N.b[n]) < 0 &&
		    TYK(N.ty[n]) != K_UINT && TYK(N.ty[n]) != K_FLOAT) {
			pr(b, N.a[n], 9, pc, 0);
			if (TYK(N.ty[n]) == K_REAL)
				sb_printf(b, " - %.10g", -N.kr[N.b[n]]);
			else
				sb_printf(b, " - %lld", -(long long)N.ki[N.b[n]]);
			break;
		}
		pr_bin(b, n, " + ", 9, 9, 10, pc, 0);
		break;
	case O_SUB: pr_bin(b, n, " - ", 9, 9, 10, pc, 0); break;
	case O_MUL:
		if (is_jux(n, pc)) {
			pr(b, N.a[n], 15, pc, 0);
			pr(b, N.b[n], 13, pc, 0);
		} else
			pr_bin(b, n, c ? " * " : "*", 10, 10, 11, pc, 0);
		break;
	case O_DIV:
		pr(b, N.a[n], 10, pc, 0);
		sb_putc(b, '/');
		pr(b, N.b[n], is_jux(N.b[n], pc) ? 13 : 11, pc, 0);
		break;
	case O_MOD: pr_bin(b, n, " % ", 10, 10, 11, pc, 0); break;
	case O_POW:
		if (c) {
			sb_puts(b, "pow(");
			pr(b, N.a[n], 0, pc, 0);
			sb_puts(b, ", ");
			pr(b, N.b[n], 0, pc, 0);
			sb_putc(b, ')');
		} else if (TYK(N.ty[n]) == K_REAL && is_const(N.b[n]) &&
		    small_frac(N.kr[N.b[n]], &fp, &fq)) {
			/* x^(1/3), x^(-2/3): the exponent as the fraction it is */
			pr(b, N.a[n], 14, pc, 0);
			sb_printf(b, "^(%ld/%ld)", fp, fq);
		} else   /* next to bitwise operators ^ would read as xor */
			pr_bin(b, n, pr_bitwise && TY_ISINT(N.ty[n]) ? "**" : "^", 13,
			    14, 13, pc, 0);
		break;
	case O_AND: pr_bin(b, n, " & ", 7, 7, 8, pc, 1); break;
	case O_OR: pr_bin(b, n, " | ", 5, 5, 6, pc, 1); break;
	case O_XOR: pr_bin(b, n, c ? " ^ " : " xor ", 6, 6, 7, pc, 1); break;
	case O_SHL: pr_bin(b, n, " << ", 8, 8, 9, pc, 0); break;
	case O_SHR: pr_bin(b, n, " >> ", 8, 8, 9, pc, 1); break;
	case O_EQ:
		if (bits) {
			sb_puts(b, "~(");
			pr_bin(b, n, " ^ ", 6, 7, 7, pc, 0);
			sb_putc(b, ')');
		} else
			pr_bin(b, n, " == ", 4, 5, 5, pc, 0);
		break;
	case O_NE: pr_bin(b, n, bits ? " ^ " : " != ", 4, 5, 5, pc, 0); break;
	case O_LT: pr_bin(b, n, " < ", 4, 5, 5, pc, 0); break;
	case O_LE: pr_bin(b, n, " <= ", 4, 5, 5, pc, 0); break;
	case O_GT: pr_bin(b, n, " > ", 4, 5, 5, pc, 0); break;
	case O_GE: pr_bin(b, n, " >= ", 4, 5, 5, pc, 0); break;
	case O_LAND: pr_bin(b, n, bits ? " & " : " && ", 3, 3, 4, pc, 0); break;
	case O_LOR: pr_bin(b, n, bits ? " | " : " || ", 2, 2, 3, pc, 0); break;
	case O_SEL:
		if (bits) {
			sb_puts(b, "((");
			pr(b, N.a[n], 7, pc, 0);
			sb_puts(b, " & ");
			pr(b, N.b[n], 7, pc, 0);
			sb_puts(b, ") | (~");
			pr(b, N.a[n], 11, pc, 0);
			sb_puts(b, " & ");
			pr(b, N.c[n], 7, pc, 0);
			sb_puts(b, "))");
			break;
		}
		pr(b, N.a[n], 2, pc, 0);
		sb_puts(b, " ? ");
		pr(b, N.b[n], 0, pc, 0);
		sb_puts(b, " : ");
		pr(b, N.c[n], 1, pc, 0);
		break;
	case O_TABLE:
		sb_putc(b, c ? '{' : '[');
		for (i = 0; i < N.b[n]; i++) {
			if (i)
				sb_puts(b, ", ");
			pr(b, N.lst[N.a[n] + i], 0, pc, 0);
		}
		sb_putc(b, c ? '}' : ']');
		break;
	case O_INDEX:
		if (c) {
			sb_printf(b, "((const %s[])", c_type(N.ty[n]));
			pr(b, N.a[n], 15, pc, 0);
			sb_putc(b, ')');
		} else
			pr(b, N.a[n], 15, pc, 0);
		sb_putc(b, '[');
		pr(b, N.b[n], 0, pc, 0);
		sb_putc(b, ']');
		break;
	case O_ATOMIC:
		sb_printf(b, "atomic_%s%s(", atomic_names[N.ki[n] & 15],
		    (N.ki[n] & AT_SHARED) ? "_shared" : "");
		pr(b, N.a[n], 0, pc, 0);
		sb_puts(b, ", ");
		pr(b, N.b[n], 0, pc, 0);
		sb_putc(b, ')');
		break;
	case O_LOOP:
		loop_print(b, n, pc);
		break;
	case O_FETCH_REPEAT: case O_FETCH_MIRROR: case O_FETCH_CLAMP:
	case O_FETCH_MCLAMP: case O_FETCH_BORDER: case O_TEX_LINEAR:
	case O_TEX_BILINEAR:
		tex_print_name(b, n);
		sb_putc(b, '(');
		pr(b, N.a[n], 0, pc, 0);
		if (op == O_TEX_BILINEAR)
			sb_printf(b, ", %lld", (long long)TEX_WIDTH(N.ki[n]));
		sb_puts(b, ", ");
		pr(b, N.b[n], 0, pc, 0);
		if (N.c[n]) {
			sb_puts(b, ", ");
			pr(b, N.c[n], 0, pc, 0);
		}
		sb_putc(b, ')');
		break;
	case O_UDOT4X8: case O_SDOT4X8: case O_SUDOT4X8:
	case O_UDOT2X16: case O_SDOT2X16: case O_SUDOT2X16:
	case O_UDOT8X4: case O_SDOT8X4: case O_SUDOT8X4:
	case O_UDOT4X8S: case O_SDOT4X8S: case O_SUDOT4X8S:
	case O_UDOT2X16S: case O_SDOT2X16S: case O_SUDOT2X16S:
	case O_UDOT8X4S: case O_SDOT8X4S: case O_SUDOT8X4S:
		/* the words in hex, without the conversion to u32 that
		 * reading them back inserts again */
		sb_printf(b, "%s(", opinfo[op].name);
		for (i = 0; i < (uint32_t)opinfo[op].arity; i++) {
			nid x = i == 0 ? N.a[n] : i == 1 ? N.b[n] : N.c[n];
			if (i)
				sb_puts(b, ", ");
			if (i < 2 && N.op[x] == O_CAST && TY_ISINT(N.ty[N.a[x]]) &&
			    N.ty[N.a[x]] != TY_BOOL && !sh_name(pc, x))
				x = N.a[x];
			pr(b, x, 0, pc, i < 2);
		}
		sb_putc(b, ')');
		break;
	case O_SGBALLOT: case O_SGANY: case O_SGALL: case O_SGADD:
	case O_SGSHUF: case O_SGSCAN:
		/* without the subgroup's size, which reading them back takes
		 * from the lane parameter or the model again; a scan's lane
		 * only when it is not the lane itself */
		sb_printf(b, "%s(", opinfo[op].name);
		pr(b, N.a[n], 0, pc, 0);
		if (op == O_SGSHUF || (op == O_SGSCAN && !sg_is_lane(N.b[n]))) {
			sb_puts(b, ", ");
			pr(b, N.b[n], 0, pc, 0);
		}
		sb_putc(b, ')');
		break;
	default:
		s = opinfo[op].name;
		if (c && op == O_POPCNT)
			s = "__builtin_popcountll";
		sb_printf(b, "%s(", s);
		pr(b, N.a[n], 0, pc, 0);
		if (opinfo[op].arity >= 2) {
			sb_puts(b, ", ");
			pr(b, N.b[n], 0, pc, 0);
		}
		if (op == O_BFI)
			/* the field's offset and count, which ride in one */
			sb_printf(b, ", %d, %d", BFI_P(N.ki[N.c[n]]),
			    BFI_N(N.ki[N.c[n]]));
		else if (opinfo[op].arity == 3) {
			sb_puts(b, ", ");
			/* a byte permute's selector reads by its nibbles, a
			 * byte select's by its bytes */
			pr(b, N.c[n], 0, pc, op == O_BYTEPERM || op == O_BYTESEL);
		}
		sb_putc(b, ')');
	}
	if (p < minp)
		sb_putc(b, ')');
}

/* does the expression use a bitwise operator?  (bounded walk) */
static int
has_bitwise(nid n, int depth)
{
	int op;

	if (n == 0 || depth > 64)
		return 0;
	op = N.op[n];
	if (op == O_AND || op == O_OR || op == O_XOR || op == O_SHL ||
	    op == O_SHR || op == O_NOT)
		return 1;
	if (op == O_CONST || op == O_VAR || op == O_UNDEF || op == O_TABLE)
		return 0;
	return has_bitwise(N.a[n], depth + 1) || has_bitwise(N.b[n], depth + 1) ||
	    has_bitwise(N.c[n], depth + 1);
}

void
print_node(struct sbuf *b, nid n, const struct pctx *pc)
{
	static const struct pctx def = {PM_UCAS, NULL};
	int saved = pr_bitwise;

	pr_bitwise = has_bitwise(n, 0);
	pr(b, n, 0, pc ? pc : &def, 0);
	pr_bitwise = saved;
}

const char *
node_str(nid n)
{
	static struct sbuf buf[8];
	static int k;
	struct sbuf *b = &buf[k++ & 7];

	sb_clear(b);
	print_node(b, n, NULL);
	return sb_str(b);
}

void
print_paren(struct sbuf *b, nid n)
{
	static const struct pctx def = {PM_UCAS, NULL};
	int op = N.op[n];
	uint32_t i;

	switch (op) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
		pr(b, n, 0, &def, 0);
		return;
	case O_TABLE:
		sb_putc(b, '[');
		for (i = 0; i < N.b[n]; i++) {
			if (i)
				sb_puts(b, ", ");
			print_paren(b, N.lst[N.a[n] + i]);
		}
		sb_putc(b, ']');
		return;
	case O_INDEX:
		print_paren(b, N.a[n]);
		sb_putc(b, '[');
		print_paren(b, N.b[n]);
		sb_putc(b, ']');
		return;
	case O_SEL:
		sb_putc(b, '(');
		print_paren(b, N.a[n]);
		sb_puts(b, " ? ");
		print_paren(b, N.b[n]);
		sb_puts(b, " : ");
		print_paren(b, N.c[n]);
		sb_putc(b, ')');
		return;
	case O_CAST:
		sb_printf(b, "%s(", ty_str(N.ty[n]));
		print_paren(b, N.a[n]);
		sb_putc(b, ')');
		return;
	case O_LOOP:
		loop_print(b, n, NULL);
		return;
	}
	if (opinfo[op].arity == 1) {
		if (op == O_NEG || op == O_NOT || op == O_LNOT) {
			sb_printf(b, "(%s", opinfo[op].name);
			print_paren(b, N.a[n]);
			sb_putc(b, ')');
		} else {
			sb_printf(b, "%s(", opinfo[op].name);
			print_paren(b, N.a[n]);
			sb_putc(b, ')');
		}
		return;
	}
	if (IS_SG(op)) {
		sb_printf(b, "%s(", opinfo[op].name);
		print_paren(b, N.a[n]);
		if (op == O_SGSHUF || (op == O_SGSCAN && !sg_is_lane(N.b[n]))) {
			sb_puts(b, ", ");
			print_paren(b, N.b[n]);
		}
		sb_putc(b, ')');
		return;
	}
	if (IS_TEX(op)) {
		tex_print_name(b, n);
		sb_putc(b, '(');
		print_paren(b, N.a[n]);
		if (op == O_TEX_BILINEAR)
			sb_printf(b, ", %lld", (long long)TEX_WIDTH(N.ki[n]));
		sb_puts(b, ", ");
		print_paren(b, N.b[n]);
		if (N.c[n]) {
			sb_puts(b, ", ");
			print_paren(b, N.c[n]);
		}
		sb_putc(b, ')');
		return;
	}
	if (op == O_MIN || op == O_MAX || op == O_COPYSIGN || op == O_FMA ||
	    op == O_ATAN2 || IS_DOT(op) || (IS_INST(op) &&
	    opinfo[op].arity >= 2)) {
		sb_printf(b, "%s(", opinfo[op].name);
		print_paren(b, N.a[n]);
		sb_puts(b, ", ");
		print_paren(b, N.b[n]);
		if (op == O_BFI)
			sb_printf(b, ", %d, %d", BFI_P(N.ki[N.c[n]]),
			    BFI_N(N.ki[N.c[n]]));
		else if (opinfo[op].arity == 3) {
			sb_puts(b, ", ");
			print_paren(b, N.c[n]);
		}
		sb_putc(b, ')');
		return;
	}
	sb_putc(b, '(');
	print_paren(b, N.a[n]);
	sb_printf(b, " %s ", opinfo[op].name);
	print_paren(b, N.b[n]);
	sb_putc(b, ')');
}

/* ---- printing with shared subterms named ---- */

struct shmap {
	nid *key;        /* open addressing over the reachable nodes */
	uint32_t *idx;
	uint32_t cap;
	nid *node;       /* reachable nodes */
	uint32_t *refs, *size;
	char **name;     /* NULL: printed inline */
	uint32_t n, ncap;
	nid *order, root;
	int nnamed;
};

static uint32_t
sh_find(const struct shmap *m, nid n)
{
	uint32_t j;

	if (m->cap == 0)
		return NOTOK;
	j = (n * 2654435761u) & (m->cap - 1);
	while (m->key[j]) {
		if (m->key[j] == n)
			return m->idx[j];
		j = (j + 1) & (m->cap - 1);
	}
	return NOTOK;
}

static const char *
sh_name(const struct pctx *pc, nid n)
{
	uint32_t i;

	if (pc->sh == NULL || n == pc->self || n == 0)
		return NULL;
	i = sh_find(pc->sh, n);
	return i == NOTOK ? NULL : pc->sh->name[i];
}

static void sh_visit(struct shmap *m, nid n);

static void
sh_grow(struct shmap *m)
{
	nid *ok = m->key;
	uint32_t *oi = m->idx, oc = m->cap, i, j;

	m->cap = m->cap ? m->cap * 2 : 256;
	m->key = xcalloc(m->cap, sizeof *m->key);
	m->idx = xcalloc(m->cap, sizeof *m->idx);
	for (i = 0; i < oc; i++)
		if (ok[i]) {
			j = (ok[i] * 2654435761u) & (m->cap - 1);
			while (m->key[j])
				j = (j + 1) & (m->cap - 1);
			m->key[j] = ok[i];
			m->idx[j] = oi[i];
		}
	free(ok);
	free(oi);
}

/* count one more reference to n; visit it the first time */
static void
sh_ref(struct shmap *m, nid n)
{
	uint32_t i, j;

	if (n == 0)
		return;
	if ((i = sh_find(m, n)) != NOTOK) {
		m->refs[i]++;
		return;
	}
	if ((m->n + 1) * 2 > m->cap)
		sh_grow(m);
	if (m->n == m->ncap) {
		m->ncap = m->ncap ? m->ncap * 2 : 64;
		m->node = xrealloc(m->node, m->ncap * sizeof *m->node);
		m->refs = xrealloc(m->refs, m->ncap * sizeof *m->refs);
		m->size = xrealloc(m->size, m->ncap * sizeof *m->size);
		m->name = xrealloc(m->name, m->ncap * sizeof *m->name);
	}
	i = m->n++;
	m->node[i] = n;
	m->refs[i] = 1;
	m->size[i] = 0;
	m->name[i] = NULL;
	j = (n * 2654435761u) & (m->cap - 1);
	while (m->key[j])
		j = (j + 1) & (m->cap - 1);
	m->key[j] = n;
	m->idx[j] = i;
	sh_visit(m, n);
}

static void
sh_visit(struct shmap *m, nid n)
{
	uint32_t j;

	switch (N.op[n]) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
		return;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			sh_ref(m, N.lst[N.a[n] + j]);
		return;
	case O_LOOP: {
		uint32_t e = loop_ent(n), s = loops.st[e], k;
		sh_ref(m, N.a[n]);
		sh_ref(m, N.b[n]);
		for (k = 0; k < loops.nst[e]; k++) {
			sh_ref(m, loops.init[s + k]);
			sh_ref(m, loops.next[s + k]);
		}
		sh_ref(m, loops.cond[e]);
		return;
	}
	}
	sh_ref(m, N.a[n]);
	sh_ref(m, N.b[n]);
	sh_ref(m, N.c[n]);
}

/* name hints: what the source called a value */
static struct {
	nid *key;
	uint32_t *sym;
	uint32_t cap, n;
} H;

void
name_hint(nid n, uint32_t sym)
{
	uint32_t j, i;

	if (n == 0 || N.op[n] == O_CONST || N.op[n] == O_VAR)
		return;
	if ((H.n + 1) * 2 > H.cap) {
		nid *ok = H.key;
		uint32_t *os = H.sym, oc = H.cap;
		H.cap = H.cap ? H.cap * 2 : 256;
		H.key = xcalloc(H.cap, sizeof *H.key);
		H.sym = xcalloc(H.cap, sizeof *H.sym);
		for (i = 0; i < oc; i++)
			if (ok[i]) {
				j = (ok[i] * 2654435761u) & (H.cap - 1);
				while (H.key[j])
					j = (j + 1) & (H.cap - 1);
				H.key[j] = ok[i];
				H.sym[j] = os[i];
			}
		free(ok);
		free(os);
	}
	j = (n * 2654435761u) & (H.cap - 1);
	while (H.key[j]) {
		if (H.key[j] == n)
			return;    /* the first name stays */
		j = (j + 1) & (H.cap - 1);
	}
	H.key[j] = n;
	H.sym[j] = sym;
	H.n++;
}

static uint32_t
hint_of(nid n)
{
	uint32_t j;

	if (H.cap == 0)
		return 0;
	j = (n * 2654435761u) & (H.cap - 1);
	while (H.key[j]) {
		if (H.key[j] == n)
			return H.sym[j];
		j = (j + 1) & (H.cap - 1);
	}
	return 0;
}

static int
cmp_nid(const void *x, const void *y)
{
	nid a = *(const nid *)x, b = *(const nid *)y;

	return a < b ? -1 : a > b;
}

/* "… (N more characters)" past cap bytes of b counted from `from` */
void
cap_text(struct sbuf *b, size_t from, size_t cap)
{
	size_t len = b->n - from, cut;
	char tail[64];

	if (len <= cap)
		return;
	cut = from + cap;
	while (cut > from && ((unsigned char)b->p[cut] & 0xC0) == 0x80)
		cut--;   /* not inside a UTF-8 sequence */
	snprintf(tail, sizeof tail, "… (%zu more characters)", b->n - cut);
	b->n = cut;
	b->p[b->n] = 0;
	sb_puts(b, tail);
}

static int
name_used(const struct shmap *m, const char *s)
{
	uint32_t i;

	for (i = 0; i < m->n; i++)
		if (m->name[i] && !strcmp(m->name[i], s))
			return 1;
	return 0;
}

static int
var_named(const struct shmap *m, const char *s)
{
	uint32_t i;

	for (i = 0; i < m->n; i++)
		if (N.op[m->node[i]] == O_VAR && !strcmp(sym_str(N.a[m->node[i]]), s))
			return 1;
	return 0;
}

/* the shared subterms of root, named; free with shmap_free */
struct shmap *
shmap_new(nid root)
{
	struct shmap *mp = xcalloc(1, sizeof *mp), m;
	nid *order;
	uint32_t i, k;
	int tn = 0;

	memset(&m, 0, sizeof m);
	sh_ref(&m, root);
	/* sizes bottom-up (operands have smaller ids), names where shared */
	order = xmalloc((m.n + 1) * sizeof *order);
	for (i = 0; i < m.n; i++)
		order[i] = m.node[i];
	qsort(order, m.n, sizeof *order, cmp_nid);
	for (k = 0; k < m.n; k++) {
		nid n = order[k];
		uint32_t s = 1, j, ch[3];
		int nch = 0;
		i = sh_find(&m, n);
		if (N.op[n] == O_TABLE) {
			for (j = 0; j < N.b[n]; j++) {
				uint32_t c = sh_find(&m, N.lst[N.a[n] + j]);
				s += c == NOTOK ? 1 : m.name[c] ? 1 : m.size[c];
			}
		} else if (N.op[n] != O_CONST && N.op[n] != O_VAR &&
		    N.op[n] != O_UNDEF) {
			ch[nch++] = N.a[n];
			ch[nch++] = N.b[n];
			ch[nch++] = N.c[n];
			for (j = 0; j < (uint32_t)nch; j++) {
				uint32_t c = ch[j] ? sh_find(&m, ch[j]) : NOTOK;
				if (c != NOTOK)
					s += m.name[c] ? 1 : m.size[c];
			}
			if (N.op[n] == O_LOOP)
				s += 8;
		}
		m.size[i] = s;
		if (n != root && m.refs[i] >= 2 && s >= 4 &&
		    N.op[n] != O_TABLE) {
			uint32_t h = hint_of(n);
			char nm[64];
			if (h && !name_used(&m, sym_str(h)) &&
			    !var_named(&m, sym_str(h)))
				snprintf(nm, sizeof nm, "%s", sym_str(h));
			else {
				do
					snprintf(nm, sizeof nm, "t%d", ++tn);
				while (name_used(&m, nm) || var_named(&m, nm));
			}
			m.name[i] = xmalloc(strlen(nm) + 1);
			strcpy(m.name[i], nm);
			m.size[i] = 1;
			m.nnamed++;
		}
	}
	m.order = order;
	m.root = root;
	*mp = m;
	return mp;
}

void
shmap_free(struct shmap *m)
{
	uint32_t i;

	if (m == NULL)
		return;
	for (i = 0; i < m->n; i++)
		free(m->name[i]);
	free(m->order);
	free(m->key);
	free(m->idx);
	free(m->node);
	free(m->refs);
	free(m->size);
	free(m->name);
	free(m);
}

int
shmap_count(const struct shmap *m)
{
	return m ? m->nnamed : 0;
}

/* e with m's names (e itself in full) */
void
print_with(struct sbuf *b, nid e, const struct shmap *m, size_t cap)
{
	struct pctx pc = {PM_UCAS, NULL, NULL, 0};
	size_t from = b->n;

	pc.sh = m;
	pc.self = e;
	print_node(b, e, &pc);
	if (cap)
		cap_text(b, from, cap);
}

/* "t1 = ..., t2 = ..." in dependency order, each binding capped */
void
shmap_bindings(struct sbuf *b, const struct shmap *m, const char *sep,
    size_t cap)
{
	uint32_t k, i;
	int first = 1;

	for (k = 0; k < m->n; k++) {
		i = sh_find(m, m->order[k]);
		if (!m->name[i])
			continue;
		sb_printf(b, "%s%s = ", first ? "" : sep, m->name[i]);
		print_with(b, m->order[k], m, cap);
		first = 0;
	}
}

void
print_shared(struct sbuf *b, nid root, size_t cap)
{
	struct shmap *m = shmap_new(root);
	size_t from = b->n;

	print_with(b, root, m, 0);
	if (m->nnamed) {
		sb_puts(b, " where ");
		shmap_bindings(b, m, ", ", 0);
	}
	if (cap)
		cap_text(b, from, cap);
	shmap_free(m);
}

const char *
shared_str(nid root, size_t cap)
{
	static struct sbuf buf[8];
	static int k;
	struct sbuf *b = &buf[k++ & 7];

	sb_clear(b);
	print_shared(b, root, cap);
	return sb_str(b);
}
