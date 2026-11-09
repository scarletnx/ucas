/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: lowering C to the node DAG.
 *
 * A predicated symbolic execution.  Every local scalar has a current
 * value (a node over the leaves); `alive` is the condition under which
 * control reaches the statement being executed.  An assignment under
 * alive is x = alive ? new : old, so both arms of an if run one after
 * the other with disjoint conditions and no merge is needed; a return
 * sets ret = alive ? v : ret and clears alive; break and continue
 * collect their conditions per loop or switch.  Loops are unrolled with
 * the same predication (up to CF_UNROLL iterations); a loop whose trip
 * count is not a constant leaves an obligation, "still running after K
 * iterations", which must be proved false (exhaustively or by SAT)
 * before the lowering is trusted.
 *
 * C99 on LP64 x86-64, as nodes:
 *   - integer promotions and the usual arithmetic conversions become
 *     explicit casts; conversion to a signed type that cannot hold the
 *     value is modulo 2^N (implementation-defined; GCC and clang do
 *     this), conversion to _Bool is x != 0;
 *   - unsigned arithmetic wraps; signed +, -, * wrap in the nodes and,
 *     unless `wrapv` is asked for, the overflow (UB, 6.5p5) becomes a
 *     domain condition, guarded by the path that evaluates it;
 *   - x / 0, x % 0, shifts by a negative amount or the width or more,
 *     out-of-range table indices and reads of uninitialised locals are
 *     undefined nodes (C UB); INT_MIN / -1 is UB in C (a condition);
 *   - >> of a negative value is arithmetic (implementation-defined;
 *     GCC and clang do this); char is signed.
 *
 * Not lowered (the function or expression stays as written): floats,
 * pointers other than reads of static const tables, stores to memory,
 * calls other than to functions of the same unit that lower themselves
 * (inlined), volatile, goto, case labels inside nested statements.
 */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfront.h"

#define CF_UNROLL 66
#define CF_MAXNODES 400000
#define CF_INLINE 4

struct lctx {
	nid brk, cont;		/* conditions collected (bool) */
	int isswitch;
};

static struct {
	struct cf_unit *u;
	struct cf_value *v;
	int mode;		/* 0 whole function, 1 one expression */
	int ub;
	nid alive, where, obl, ret;
	nid *env;		/* per symbol */
	uint8_t *set;
	struct lctx lc[32];
	int nlc;
	int depth;		/* inlining */
	uint32_t nbase;		/* N.n at the start */
	jmp_buf jb;
	char why[200];
	uint32_t leafkey[CF_MAXLEAF];	/* symbol + 1, or ~AST node */
	nid leafnode[CF_MAXLEAF];
	uint32_t fnsym;
} L;

#define AU (L.u->A)
#define SU (L.u->S)

static void
lfail(uint32_t at, const char *fmt, const char *a)
{
	char m[160];

	snprintf(m, sizeof m, fmt, a);
	if (at && at < AU.n)
		snprintf(L.why, sizeof L.why, "line %u: %s",
		    L.u->tk.line[AU.t0[at]], m);
	else
		snprintf(L.why, sizeof L.why, "%s", m);
	longjmp(L.jb, 1);
}

static nid T1, F0;
static nid bnot(nid a);

static nid
band(nid a, nid b)
{
	if (a == F0 || b == F0)
		return F0;
	if (a == T1)
		return b;
	if (b == T1)
		return a;
	if (a == b)
		return a;
	if (bnot(a) == b)
		return F0;
	return mk_bin(O_LAND, a, b);
}

static nid
bor(nid a, nid b)
{
	if (a == T1 || b == T1)
		return T1;
	if (a == F0)
		return b;
	if (b == F0)
		return a;
	if (a == b)
		return a;
	if (bnot(a) == b)
		return T1;
	return mk_bin(O_LOR, a, b);
}

static nid
bnot(nid a)
{
	if (a == T1)
		return F0;
	if (a == F0)
		return T1;
	return mk_un(O_LNOT, a);
}

static nid
sel(nid c, nid a, nid b)
{
	if (c == T1)
		return a;
	if (c == F0)
		return b;
	if (a == b)
		return a;
	return mk_sel(c, a, b);
}

static ty_t
uty(uint32_t cty, uint32_t at)
{
	ty_t t = ct_to_ucas(cty);

	if (t == 0) {
		struct sbuf b = {0};
		ct_str(&b, cty, "");
		snprintf(L.why, sizeof L.why, "line %u: a value of type %s (only "
		    "integer values are lowered)", at && at < AU.n ?
		    L.u->tk.line[AU.t0[at]] : 0, b.p ? b.p : "?");
		free(b.p);
		longjmp(L.jb, 1);
	}
	return t;
}

static nid
conv(nid x, uint32_t to, uint32_t at)
{
	ty_t t;

	if (CT.kind[ct_unqual(to)] == CT_BOOL)
		return N.ty[x] == TY_BOOL ? x : mk_bool(x);
	t = uty(to, at);
	return mk_cast(t, x);
}

static nid
kint(uint32_t cty, int64_t v, uint32_t at)
{
	ty_t t = uty(cty, at);

	return mk_const_i(t, int_norm(t, v));
}

static void
budget(void)
{
	if (N.n - L.nbase > CF_MAXNODES)
		lfail(0, "the lowered program is too large%s", "");
}

static void
add_cond(nid g, nid ok)
{
	nid c;

	if (ok == T1 || g == F0)
		return;
	c = bor(bnot(g), ok);
	L.where = L.where ? band(L.where, c) : c;
}

static int
is_local_scalar(uint32_t s)
{
	return (SU.kind[s] == CS_LOCAL || SU.kind[s] == CS_PARAM) &&
	    !SU.addr_taken[s] && ct_is_int(SU.ty[s]) &&
	    !(CT.qual[SU.ty[s]] & CQ_VOLATILE);
}

/* the source text of an AST node, or 0 */
static int
node_text(uint32_t n, char *buf, size_t sz)
{
	uint32_t a, b;

	if (!cf_span(L.u, n, &a, &b) || b - a + 1 > sz)
		return 0;
	snprintf(buf, sz, "%.*s", (int)(b - a), L.u->ftext[0] + a);
	return 1;
}

static nid
leaf(uint32_t key, uint32_t cty, const char *text, uint32_t at)
{
	struct cf_value *v = L.v;
	ty_t t = uty(cty, at);
	int k;

	for (k = 0; k < v->nleaf; k++)
		if (L.leafkey[k] == key)
			return L.leafnode[k];
	if (v->nleaf == CF_MAXLEAF)
		lfail(at, "more than 8 inputs%s", "");
	k = v->nleaf++;
	L.leafkey[k] = key;
	snprintf(v->leaf[k].text, sizeof v->leaf[k].text, "%s", text);
	v->leaf[k].cty = cty;
	v->leaf[k].ranged = 0;
	L.leafnode[k] = mk_var((uint32_t)k, sym_intern(text, strlen(text)), t);
	return L.leafnode[k];
}

/* a static const table: its elements, or 0 */
static nid
const_table(uint32_t s)
{
	uint32_t ty = SU.ty[s], d = SU.decl[s], init, k, cnt;
	nid el[4096];

	if (CT.kind[ty] != CT_ARRAY || !ct_is_int(CT.base[ty]) ||
	    !(CT.qual[CT.base[ty]] & CQ_CONST) || SU.kind[s] != CS_GLOBAL ||
	    d == 0 || AU.kind[d] != CA_DECL || (init = AU.a[d]) == 0 ||
	    AU.kind[init] != CA_INIT)
		return 0;
	cnt = AU.b[init];
	if (CT.count[ty] < 0 || CT.count[ty] > 4096 || cnt > (uint32_t)CT.count[ty])
		return 0;
	for (k = 0; k < (uint32_t)CT.count[ty]; k++) {
		int64_t v = 0;
		if (k < cnt) {
			uint32_t e = AU.lst[AU.a[init] + k];
			/* a designated or nested initializer: give up */
			if (AU.d[e] || AU.kind[e] == CA_INIT)
				return 0;
			while (AU.kind[e] == CA_CAST && AU.iv[e] == 1)
				e = AU.a[e];
			if (AU.kind[e] == CA_NUM || AU.kind[e] == CA_CHARLIT)
				v = AU.iv[e];
			else if (AU.kind[e] == CA_NEG && (AU.kind[AU.a[e]] ==
			    CA_NUM))
				v = -AU.iv[AU.a[e]];
			else
				return 0;
		}
		el[k] = kint(CT.base[ty], v, d);
	}
	return mk_table(el, (uint32_t)CT.count[ty]);
}

static nid ex(uint32_t n, nid g);
static void st(uint32_t n);

static int
sym_assigned_in(uint32_t n)
{
	/* does the subtree read a variable assigned in this lowering? */
	uint32_t k;

	if (n == 0)
		return 0;
	if (AU.kind[n] == CA_ID)
		return L.set[AU.iv[n]];
	if (AU.kind[n] == CA_CALL || AU.kind[n] == CA_ASSIGN ||
	    AU.kind[n] == CA_POSTINC || AU.kind[n] == CA_POSTDEC ||
	    AU.kind[n] == CA_PREINC || AU.kind[n] == CA_PREDEC)
		return 1;
	if (AU.kind[n] == CA_INIT)
		return 1;
	if (sym_assigned_in(AU.a[n]))
		return 1;
	if (AU.kind[n] == CA_MEMBER || AU.kind[n] == CA_ARROW)
		return 0;
	if (AU.kind[n] == CA_BIN || AU.kind[n] == CA_INDEX ||
	    AU.kind[n] == CA_LAND || AU.kind[n] == CA_LOR ||
	    AU.kind[n] == CA_COMMA)
		if (sym_assigned_in(AU.b[n]))
			return 1;
	if (AU.kind[n] == CA_COND)
		if (sym_assigned_in(AU.b[n]) || sym_assigned_in(AU.c[n]))
			return 1;
	(void)k;
	return 0;
}

/* a load: a table in the whole-function mode, a leaf in expression mode */
static nid
load(uint32_t n, nid g)
{
	char text[96];
	uint32_t ty = AU.ty[n];

	if (CT.qual[ty] & CQ_VOLATILE)
		lfail(n, "a volatile access%s", "");
	if (!ct_is_int(ty))
		(void)uty(ty, n);
	if (AU.kind[n] == CA_INDEX) {
		uint32_t b = AU.a[n];
		while (AU.kind[b] == CA_CAST && AU.iv[b] == 1)
			b = AU.a[b];
		if (AU.kind[b] == CA_ID) {
			nid tab = const_table((uint32_t)AU.iv[b]);
			if (tab) {
				nid i = ex(AU.b[n], g);
				if (TYK(N.ty[i]) == K_UINT || TYK(N.ty[i]) == K_BOOL)
					i = mk_cast(TY(K_UINT, 64), i);
				else
					i = mk_cast(TY(K_SINT, 64), i);
				return mk_index(tab, i);
			}
		}
	}
	if (L.mode == 0)
		lfail(n, "a memory access%s", "");
	if (g != T1)
		lfail(n, "a load under a condition%s", "");
	if (sym_assigned_in(n))
		lfail(n, "a load whose address changes%s", "");
	if (!node_text(n, text, sizeof text))
		lfail(n, "a load from a macro%s", "");
	return leaf(~n, ty, text, n);
}

static nid
read_var(uint32_t n)
{
	uint32_t s = (uint32_t)AU.iv[n], ty = SU.ty[s];
	char text[96];

	if (L.set[s])
		return L.env[s];
	if (CT.qual[ty] & CQ_VOLATILE)
		lfail(n, "a volatile variable%s", "");
	if (SU.kind[s] == CS_GLOBAL && (CT.qual[ty] & CQ_CONST) &&
	    ct_is_int(ty) && SU.decl[s] && AU.a[SU.decl[s]]) {
		uint32_t e = AU.a[SU.decl[s]];
		while (AU.kind[e] == CA_CAST && AU.iv[e] == 1)
			e = AU.a[e];
		if (AU.kind[e] == CA_NUM || AU.kind[e] == CA_CHARLIT)
			return kint(ty, AU.iv[e], n);
	}
	if (!ct_is_int(ty))
		(void)uty(ty, n);
	if (L.mode == 0 && SU.kind[s] != CS_PARAM && SU.kind[s] != CS_GLOBAL)
		return mk_undef(uty(ty, n));	/* uninitialised */
	if (L.mode == 0 && SU.kind[s] == CS_GLOBAL)
		lfail(n, "a global variable%s", "");
	snprintf(text, sizeof text, "%s", sym_str(SU.name[s]));
	if (cf_is_macro(L.u, SU.name[s]))
		lfail(n, "a variable whose name is a macro%s", "");
	return leaf(s + 1, ty, text, n);
}

static void
write_var(uint32_t lv, nid v)
{
	uint32_t s;

	if (AU.kind[lv] != CA_ID)
		lfail(lv, "a store to memory%s", "");
	s = (uint32_t)AU.iv[lv];
	if (!is_local_scalar(s))
		lfail(lv, "a store to %s", sym_str(SU.name[s]));
	if (L.mode == 1)
		lfail(lv, "an assignment inside the expression%s", "");
	if (!L.set[s]) {
		/* a parameter keeps its leaf as the old value */
		nid old = SU.kind[s] == CS_PARAM ? read_var(lv) :
		    mk_undef(uty(SU.ty[s], lv));
		L.env[s] = old;
		L.set[s] = 1;
	}
	L.env[s] = sel(L.alive, v, L.env[s]);
	budget();
}

/* the overflow condition of a signed op, in the C type ct */
static nid
no_overflow(int op, nid a, nid b, uint32_t ct)
{
	ty_t t = N.ty[a];
	int w = TYW(t);

	if (TYK(t) != K_SINT)
		return T1;
	if (w <= 32) {
		ty_t w64 = TY(K_SINT, 64);
		nid A = mk_cast(w64, a), B = mk_cast(w64, b), r;
		r = mk_bin(op, A, B);
		return band(mk_bin(O_GE, r, mk_const_i(w64,
		    -((int64_t)1 << (w - 1)))), mk_bin(O_LE, r, mk_const_i(w64,
		    ((int64_t)1 << (w - 1)) - 1)));
	}
	if (op == O_ADD || op == O_SUB) {
		nid r = mk_bin(op, a, b), x;
		if (op == O_ADD)
			x = mk_bin(O_AND, mk_bin(O_XOR, a, r), mk_bin(O_XOR, b, r));
		else
			x = mk_bin(O_AND, mk_bin(O_XOR, a, b), mk_bin(O_XOR, a, r));
		return mk_bin(O_GE, x, mk_const_i(t, 0));
	}
	(void)ct;
	return T1;	/* 64-bit multiply: not exploited */
}

static int
c2op(int op)
{
	switch (op) {
	case '+': return O_ADD;
	case '-': return O_SUB;
	case '*': return O_MUL;
	case '/': return O_DIV;
	case '%': return O_MOD;
	case '&': return O_AND;
	case '|': return O_OR;
	case '^': return O_XOR;
	case CO_SHL: return O_SHL;
	case CO_SHR: return O_SHR;
	case '<': return O_LT;
	case '>': return O_GT;
	case CO_LE: return O_LE;
	case CO_GE: return O_GE;
	case CO_EQ: return O_EQ;
	case CO_NE: return O_NE;
	}
	return 0;
}

/* a op b in C, a and b already lowered with C types ta and tb */
static nid
arith(int cop, nid a, uint32_t ta, nid b, uint32_t tb, uint32_t at, nid g)
{
	int op = c2op(cop);
	uint32_t rt;

	if (CT.kind[ct_unqual(ta)] == CT_PTR || CT.kind[ct_unqual(tb)] == CT_PTR)
		lfail(at, "pointer arithmetic%s", "");
	if (op == O_SHL || op == O_SHR) {
		rt = ct_promote(ta);
		a = conv(a, rt, at);
		b = conv(b, ct_promote(tb), at);
		return mk_bin(op, a, b);
	}
	if (opinfo[op].fl & OF_CMP) {
		rt = ct_usual(ta, tb);
		a = conv(a, rt, at);
		b = conv(b, rt, at);
		return mk_cast(TY(K_SINT, 32), mk_bin(op, a, b));
	}
	rt = ct_usual(ta, tb);
	a = conv(a, rt, at);
	b = conv(b, rt, at);
	if (L.ub && (op == O_ADD || op == O_SUB || op == O_MUL))
		add_cond(g, no_overflow(op, a, b, rt));
	if (L.ub && (op == O_DIV || op == O_MOD) && TYK(N.ty[a]) == K_SINT) {
		ty_t t = N.ty[a];
		int w = TYW(t);
		nid mn = mk_const_i(t, w >= 64 ? INT64_MIN : -((int64_t)1 <<
		    (w - 1)));
		add_cond(g, bnot(band(mk_bin(O_EQ, a, mn), mk_bin(O_EQ, b,
		    mk_const_i(t, -1)))));
	}
	return mk_bin(op, a, b);
}

static nid
truth(uint32_t n, nid g)
{
	nid v = ex(n, g);

	return N.ty[v] == TY_BOOL ? v : mk_bool(v);
}

static nid inline_call(uint32_t n, nid g);

static nid
ex(uint32_t n, nid g)
{
	nid a, b, v;
	uint32_t ty = AU.ty[n];

	budget();
	switch (AU.kind[n]) {
	case CA_NUM:
	case CA_CHARLIT:
		return kint(ty, AU.iv[n], n);
	case CA_ID:
		if (CT.kind[ct_unqual(ty)] == CT_ARRAY)
			lfail(n, "an array as a value%s", "");
		return read_var(n);
	case CA_INDEX:
	case CA_DEREF:
	case CA_MEMBER:
	case CA_ARROW:
		return load(n, g);
	case CA_CAST:
		if (!ct_is_int(ty))
			(void)uty(ty, n);
		if (!ct_is_int(AU.ty[AU.a[n]]))
			(void)uty(AU.ty[AU.a[n]], n);
		return conv(ex(AU.a[n], g), ty, n);
	case CA_PLUS:
		return conv(ex(AU.a[n], g), ty, n);
	case CA_NEG:
		a = conv(ex(AU.a[n], g), ty, n);
		if (L.ub)
			add_cond(g, no_overflow(O_SUB, mk_const_i(N.ty[a], 0), a,
			    ty));
		return mk_un(O_NEG, a);
	case CA_BNOT:
		return mk_un(O_NOT, conv(ex(AU.a[n], g), ty, n));
	case CA_LNOT:
		return mk_cast(TY(K_SINT, 32), bnot(truth(AU.a[n], g)));
	case CA_BIN:
		a = ex(AU.a[n], g);
		b = ex(AU.b[n], g);
		return arith((int)AU.iv[n], a, AU.ty[AU.a[n]], b, AU.ty[AU.b[n]],
		    n, g);
	case CA_LAND:
	case CA_LOR:
		a = truth(AU.a[n], g);
		b = truth(AU.b[n], AU.kind[n] == CA_LAND ? band(g, a) :
		    band(g, bnot(a)));
		return mk_cast(TY(K_SINT, 32), AU.kind[n] == CA_LAND ?
		    mk_bin(O_LAND, a, b) : mk_bin(O_LOR, a, b));
	case CA_COND: {
		nid c = truth(AU.a[n], g);
		a = conv(ex(AU.b[n], band(g, c)), ty, n);
		b = conv(ex(AU.c[n], band(g, bnot(c))), ty, n);
		return sel(c, a, b);
	}
	case CA_COMMA:
		(void)ex(AU.a[n], g);
		return ex(AU.b[n], g);
	case CA_ASSIGN: {
		uint32_t l = AU.a[n], lt = AU.ty[l];
		int op = (int)AU.iv[n];
		if (g != L.alive)
			lfail(n, "an assignment under a condition inside an "
			    "expression%s", "");
		b = ex(AU.b[n], g);
		if (op != '=') {
			a = ex(l, g);
			b = arith(op, a, lt, b, AU.ty[AU.b[n]], n, g);
		}
		v = conv(b, lt, n);
		write_var(l, v);
		return v;
	}
	case CA_PREINC:
	case CA_PREDEC:
	case CA_POSTINC:
	case CA_POSTDEC: {
		uint32_t l = AU.a[n], lt = AU.ty[l];
		nid one, nv;
		int inc = AU.kind[n] == CA_PREINC || AU.kind[n] == CA_POSTINC;
		if (g != L.alive)
			lfail(n, "++/-- under a condition inside an "
			    "expression%s", "");
		a = ex(l, g);
		one = kint(ct_basic(CT_INT), 1, n);
		nv = conv(arith(inc ? '+' : '-', a, lt, one, ct_basic(CT_INT), n,
		    g), lt, n);
		write_var(l, nv);
		return AU.kind[n] == CA_PREINC || AU.kind[n] == CA_PREDEC ? nv : a;
	}
	case CA_CALL:
		return inline_call(n, g);
	case CA_FNUM:
		lfail(n, "a floating-point value%s", "");
	}
	lfail(n, "an expression of this kind (%s)", AU.kind[n] == CA_ADDR ?
	    "&" : AU.kind[n] == CA_STR ? "a string" : AU.kind[n] == CA_SIZEOF_E ?
	    "sizeof" : AU.kind[n] == CA_COMPOUND ? "a compound literal" : "?");
	return 0;
}

/* a call of a function of this unit: inlined, its parameters bound */
static nid
inline_call(uint32_t n, nid g)
{
	uint32_t callee = AU.a[n], s, fn, k;
	nid args[64], saved_ret, saved_alive, *saved_env;
	uint8_t *saved_set;
	struct lctx saved_lc[32];
	int saved_nlc;
	uint32_t saved_fn;

	while (AU.kind[callee] == CA_CAST && AU.iv[callee] == 1)
		callee = AU.a[callee];
	if (AU.kind[callee] != CA_ID)
		lfail(n, "a call through a pointer%s", "");
	s = (uint32_t)AU.iv[callee];
	fn = SU.decl[s];
	if (SU.kind[s] != CS_FUNC || fn == 0 || AU.kind[fn] != CA_FUNC)
		lfail(n, "a call of %s (not defined in this file)",
		    sym_str(SU.name[s]));
	if (L.depth >= CF_INLINE)
		lfail(n, "calls nested too deeply (recursion?)%s", "");
	if (AU.c[n] != AU.d[fn])
		lfail(n, "a call with the wrong number of arguments%s", "");
	if (g != L.alive)
		lfail(n, "a call under a condition inside an expression%s", "");
	for (k = 0; k < AU.c[n] && k < 64; k++) {
		uint32_t ps = AU.lst[AU.c[fn] + k];
		args[k] = conv(ex(AU.lst[AU.b[n] + k], g), SU.ty[ps], n);
	}
	/* a fresh frame: the callee's symbols are its own */
	saved_env = xmalloc(SU.n * sizeof *saved_env);
	saved_set = xmalloc(SU.n);
	memcpy(saved_env, L.env, SU.n * sizeof *saved_env);
	memcpy(saved_set, L.set, SU.n);
	saved_ret = L.ret;
	saved_alive = L.alive;
	saved_nlc = L.nlc;
	memcpy(saved_lc, L.lc, sizeof saved_lc);
	saved_fn = L.fnsym;
	for (k = 0; k < AU.c[n]; k++) {
		uint32_t ps = AU.lst[AU.c[fn] + k];
		L.env[ps] = args[k];
		L.set[ps] = 1;
	}
	L.ret = mk_undef(ct_to_ucas(CT.base[SU.ty[s]]) ?
	    ct_to_ucas(CT.base[SU.ty[s]]) : TY(K_SINT, 32));
	L.nlc = 0;
	L.fnsym = s;
	L.depth++;
	st(AU.a[fn]);
	L.depth--;
	{
		nid r = L.ret;
		uint32_t rt = CT.base[SU.ty[s]];
		/* the callee's parameters and locals are dead now */
		memcpy(L.env, saved_env, SU.n * sizeof *saved_env);
		memcpy(L.set, saved_set, SU.n);
		free(saved_env);
		free(saved_set);
		L.ret = saved_ret;
		L.alive = saved_alive;
		L.nlc = saved_nlc;
		memcpy(L.lc, saved_lc, sizeof saved_lc);
		L.fnsym = saved_fn;
		if (CT.kind[rt] == CT_VOID)
			lfail(n, "a call of a void function%s", "");
		return r;
	}
}

/* ---- statements ---- */

static void
decl(uint32_t n)
{
	uint32_t s = (uint32_t)AU.iv[n], ty = SU.ty[s];

	if (SU.kind[s] == CS_FUNC)
		return;
	if (SU.kind[s] == CS_GLOBAL) {
		/* a static local: only a const table or constant is fine */
		if ((CT.qual[ty] & CQ_CONST) || (CT.kind[ty] == CT_ARRAY &&
		    (CT.qual[CT.base[ty]] & CQ_CONST)))
			return;
		lfail(n, "a static local variable%s", "");
	}
	if (!is_local_scalar(s))
		lfail(n, "a local of a non-integer type or whose address is "
		    "taken (%s)", sym_str(SU.name[s]));
	if (L.mode == 1)
		lfail(n, "a declaration%s", "");
	/* a new object each time the declaration is reached */
	if (!L.set[s]) {
		L.env[s] = mk_undef(uty(ty, n));
		L.set[s] = 1;
	}
	if (AU.a[n]) {
		nid v;
		if (AU.kind[AU.a[n]] == CA_INIT)
			lfail(n, "a braced initializer%s", "");
		v = conv(ex(AU.a[n], L.alive), ty, n);
		L.env[s] = sel(L.alive, v, L.env[s]);
	} else
		L.env[s] = sel(L.alive, mk_undef(uty(ty, n)), L.env[s]);
}

static void
ret(uint32_t n)
{
	uint32_t rt = CT.base[SU.ty[L.fnsym]];
	nid v;

	if (AU.a[n] == 0) {
		if (CT.kind[rt] != CT_VOID)
			lfail(n, "return without a value%s", "");
		L.alive = F0;
		return;
	}
	v = conv(ex(AU.a[n], L.alive), rt, n);
	L.ret = sel(L.alive, v, L.ret);
	L.alive = F0;
}

static void
loop(uint32_t cond, uint32_t body, uint32_t step, int dofirst, uint32_t at)
{
	nid exited = F0;
	int it;
	struct lctx *lc;

	if (L.nlc == 32)
		lfail(at, "loops nested too deeply%s", "");
	for (it = 0; it <= CF_UNROLL; it++) {
		nid c;
		if (L.alive == F0)
			break;
		if (!(dofirst && it == 0) && cond) {
			c = truth(cond, L.alive);
			exited = bor(exited, band(L.alive, bnot(c)));
			L.alive = band(L.alive, c);
			if (L.alive == F0)
				break;
		}
		if (it == CF_UNROLL) {
			/* still running: an obligation */
			L.obl = L.obl ? bor(L.obl, L.alive) : L.alive;
			L.alive = F0;
			break;
		}
		lc = &L.lc[L.nlc++];
		lc->brk = F0;
		lc->cont = F0;
		lc->isswitch = 0;
		st(body);
		L.nlc--;
		lc = &L.lc[L.nlc];
		exited = bor(exited, lc->brk);
		L.alive = bor(L.alive, lc->cont);
		if (step && L.alive != F0)
			(void)ex(step, L.alive);
		budget();
	}
	L.alive = exited;
}

/* the statements of a switch body, at its top level */
static void
switch_body(uint32_t body, nid v, uint32_t vty, uint32_t at)
{
	uint32_t k, cnt, *items, one[1];
	nid entry = L.alive, none = T1;
	int has_default = 0;
	struct lctx *lc;

	if (AU.kind[body] == CA_BLOCK) {
		items = &AU.lst[AU.a[body]];
		cnt = AU.b[body];
	} else {
		one[0] = body;
		items = one;
		cnt = 1;
	}
	/* which case values exist: default = none of them */
	for (k = 0; k < cnt; k++) {
		uint32_t s = items[k];
		while (AU.kind[s] == CA_CASE || AU.kind[s] == CA_DEFAULT) {
			if (AU.kind[s] == CA_CASE)
				none = band(none, mk_bin(O_NE, v, kint(vty,
				    AU.iv[s], s)));
			else
				has_default = 1;
			s = AU.a[s];
		}
	}
	lc = &L.lc[L.nlc++];
	lc->brk = F0;
	lc->cont = F0;
	lc->isswitch = 1;
	L.alive = F0;
	for (k = 0; k < cnt; k++) {
		uint32_t s = items[k];
		while (AU.kind[s] == CA_CASE || AU.kind[s] == CA_DEFAULT) {
			nid hit = AU.kind[s] == CA_CASE ? mk_bin(O_EQ, v, kint(vty,
			    AU.iv[s], s)) : none;
			L.alive = bor(L.alive, band(entry, hit));
			s = AU.a[s];
		}
		st(s);
	}
	L.nlc--;
	lc = &L.lc[L.nlc];
	L.alive = bor(L.alive, lc->brk);
	/* no label matched and no default: past the switch */
	if (!has_default)
		L.alive = bor(L.alive, band(entry, none));
	/* continue inside a switch belongs to the loop around it */
	if (lc->cont != F0) {
		int j;
		for (j = L.nlc - 1; j >= 0 && L.lc[j].isswitch; j--)
			;
		if (j < 0)
			lfail(at, "continue outside a loop%s", "");
		L.lc[j].cont = bor(L.lc[j].cont, lc->cont);
	}
}

static int
has_label(uint32_t n)
{
	uint32_t k;

	if (n == 0)
		return 0;
	switch (AU.kind[n]) {
	case CA_CASE:
	case CA_DEFAULT:
	case CA_LABEL:
		return 1;
	case CA_BLOCK:
		for (k = 0; k < AU.b[n]; k++)
			if (has_label(AU.lst[AU.a[n] + k]))
				return 1;
		return 0;
	case CA_IF:
		return has_label(AU.b[n]) || has_label(AU.c[n]);
	case CA_WHILE:
		return has_label(AU.b[n]);
	case CA_DO:
		return has_label(AU.a[n]);
	case CA_FOR:
		return has_label(AU.d[n]);
	}
	return 0;
}

static void
st(uint32_t n)
{
	uint32_t k;

	if (n == 0 || L.alive == F0)
		return;
	budget();
	switch (AU.kind[n]) {
	case CA_EMPTY:
	case CA_TYPEDEF:
		return;
	case CA_BLOCK:
		for (k = 0; k < AU.b[n]; k++)
			st(AU.lst[AU.a[n] + k]);
		return;
	case CA_DECL:
		decl(n);
		return;
	case CA_EXPR:
		(void)ex(AU.a[n], L.alive);
		return;
	case CA_IF: {
		nid c = truth(AU.a[n], L.alive), a0 = L.alive, at, ae;
		if (has_label(AU.b[n]) || has_label(AU.c[n]))
			lfail(n, "a case label inside an if%s", "");
		L.alive = band(a0, c);
		st(AU.b[n]);
		at = L.alive;
		L.alive = band(a0, bnot(c));
		st(AU.c[n]);
		ae = L.alive;
		L.alive = bor(at, ae);
		return;
	}
	case CA_WHILE:
		if (has_label(AU.b[n]))
			lfail(n, "a case label inside a loop%s", "");
		loop(AU.a[n], AU.b[n], 0, 0, n);
		return;
	case CA_DO:
		if (has_label(AU.a[n]))
			lfail(n, "a case label inside a loop%s", "");
		loop(AU.b[n], AU.a[n], 0, 1, n);
		return;
	case CA_FOR:
		if (has_label(AU.d[n]))
			lfail(n, "a case label inside a loop%s", "");
		st(AU.a[n]);
		loop(AU.b[n], AU.d[n], AU.c[n], 0, n);
		return;
	case CA_RETURN:
		ret(n);
		return;
	case CA_BREAK:
	case CA_CONTINUE:
		if (L.nlc == 0)
			lfail(n, "break or continue outside a loop%s", "");
		if (AU.kind[n] == CA_BREAK)
			L.lc[L.nlc - 1].brk = bor(L.lc[L.nlc - 1].brk, L.alive);
		else
			L.lc[L.nlc - 1].cont = bor(L.lc[L.nlc - 1].cont, L.alive);
		L.alive = F0;
		return;
	case CA_SWITCH: {
		uint32_t vt = ct_promote(AU.ty[AU.a[n]]);
		nid v = conv(ex(AU.a[n], L.alive), vt, n);
		switch_body(AU.b[n], v, vt, n);
		return;
	}
	case CA_CASE:
	case CA_DEFAULT:
		lfail(n, "a case label outside the top level of its "
		    "switch%s", "");
		return;
	case CA_GOTO:
	case CA_LABEL:
		lfail(n, "goto%s", "");
		return;
	}
	lfail(n, "a statement of this kind%s", "");
}

static void
lower_begin(struct cf_unit *u, struct cf_value *v, int mode, int ub)
{
	memset(&L, 0, sizeof L);
	memset(v, 0, sizeof *v);
	L.u = u;
	L.v = v;
	L.mode = mode;
	L.ub = ub;
	L.nbase = N.n;
	T1 = mk_const_i(TY_BOOL, 1);
	F0 = mk_const_i(TY_BOOL, 0);
	L.alive = T1;
	L.env = xcalloc(u->S.n + 1, sizeof *L.env);
	L.set = xcalloc(u->S.n + 1, 1);
}

static void
lower_end(void)
{
	free(L.env);
	free(L.set);
	L.env = NULL;
	L.set = NULL;
}

void
cf_value_func(const struct cf_value *v, struct func *f)
{
	int k;

	memset(f, 0, sizeof *f);
	f->name = sym_cstr("cfront");
	f->anon = 1;
	f->np = v->nleaf;
	for (k = 0; k < v->nleaf; k++) {
		f->p[k].name = sym_intern(v->leaf[k].text, strlen(v->leaf[k].text));
		f->p[k].ty = ct_to_ucas(v->leaf[k].cty);
		if (v->leaf[k].ranged) {
			f->p[k].ranged = 1;
			f->p[k].lo = v->leaf[k].lo;
			f->p[k].hi = v->leaf[k].hi;
		}
	}
	f->body = v->e;
	f->where = v->where;
}

/*
 * The whole function as one value of its parameters.  ub: signed
 * overflow and INT_MIN / -1 are outside the domain (C UB).
 */
int
cf_lower_func(struct cf_unit *u, uint32_t fn, struct cf_value *v, int ub)
{
	uint32_t s = (uint32_t)u->A.iv[fn], ft = u->S.ty[s], k;
	nid r;

	lower_begin(u, v, 0, ub);
	if (setjmp(L.jb)) {
		v->e = 0;
		v->why = strdup(L.why);
		lower_end();
		return 0;
	}
	L.fnsym = s;
	if (u->A.d[fn] > CF_MAXLEAF)
		lfail(fn, "more than 8 parameters%s", "");
	if (!ct_is_int(CT.base[ft]))
		lfail(fn, "the function does not return an integer%s", "");
	/* parameters first, in order */
	for (k = 0; k < u->A.d[fn]; k++) {
		uint32_t ps = u->A.lst[u->A.c[fn] + k], ty = u->S.ty[ps];
		if (!ct_is_int(ty))
			lfail(fn, "a parameter that is not an integer (%s)",
			    sym_str(u->S.name[ps]));
		if (u->S.addr_taken[ps])
			lfail(fn, "the address of a parameter is taken%s", "");
		(void)leaf(ps + 1, ty, sym_str(u->S.name[ps]), fn);
	}
	L.ret = mk_undef(uty(CT.base[ft], fn));
	st(u->A.a[fn]);
	r = L.ret;
	v->e = r;
	v->where = L.where;
	if (L.obl) {
		/* every loop must have finished within the unrolled count */
		struct func f;
		struct eqproof pr;
		nid ob = mk_cast(TY(K_UINT, 8), L.obl);
		cf_value_func(v, &f);
		if (equiv_decide(&f, mk_const_i(TY(K_UINT, 8), 0), ob, NULL, &pr) ||
		    pr.how == EQ_SAMPLED) {
			snprintf(L.why, sizeof L.why, "a loop that may run more "
			    "than %d times", CF_UNROLL);
			v->e = 0;
			v->why = strdup(L.why);
			lower_end();
			return 0;
		}
	}
	lower_end();
	return 1;
}

/* one expression; every variable and load in it is a leaf */
int
cf_lower_expr(struct cf_unit *u, uint32_t fn, uint32_t e, struct cf_value *v)
{
	lower_begin(u, v, 1, 0);
	if (setjmp(L.jb)) {
		v->e = 0;
		v->why = strdup(L.why);
		lower_end();
		return 0;
	}
	L.fnsym = (uint32_t)u->A.iv[fn];
	if (!ct_is_int(u->A.ty[e]))
		lfail(e, "not an integer expression%s", "");
	v->e = ex(e, T1);
	v->where = 0;
	lower_end();
	return 1;
}
