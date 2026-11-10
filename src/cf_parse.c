/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: the C99 parser and type checker.
 *
 * Recursive descent over the preprocessed tokens into the SoA AST of
 * cfront.h; types are computed as the tree is built (integer promotions,
 * the usual arithmetic conversions, array and function decay, pointer
 * arithmetic, lvalues).  Errors carry file:line:column.
 *
 * LP64 x86-64 (the System V ABI): char is signed, 8/16/32/64-bit short,
 * int, long, long long; pointers 64-bit; enum types are unsigned int
 * when no constant is negative, else int (as GCC and clang choose);
 * long double is 16 bytes (x87 extended, opaque here).
 *
 * Accepted and ignored: __attribute__((...)), __extension__, inline,
 * __inline, _Noreturn, register, auto, __restrict.  Not accepted: K&R
 * definitions, bit-fields' layout (parsed; the struct is opaque to the
 * lowering), _Complex, VLAs (parsed as arrays of unknown size), asm.
 */
#include <ctype.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfront.h"

struct cf_types CT;
static uint32_t *ct_uq;	/* the unqualified version of each type */

/* ---- types ---- */

static uint32_t
ct_new(int kind)
{
	struct cf_types *c = &CT;
	uint32_t i;

	if (c->n == c->cap) {
		c->cap = newcap(c->cap, c->n + 1);
		c->kind = xrealloc(c->kind, c->cap);
		c->qual = xrealloc(c->qual, c->cap);
		c->variadic = xrealloc(c->variadic, c->cap);
		c->complete = xrealloc(c->complete, c->cap);
		c->base = xrealloc(c->base, c->cap * sizeof *c->base);
		c->count = xrealloc(c->count, c->cap * sizeof *c->count);
		c->size = xrealloc(c->size, c->cap * sizeof *c->size);
		c->align = xrealloc(c->align, c->cap * sizeof *c->align);
		c->fst = xrealloc(c->fst, c->cap * sizeof *c->fst);
		c->nf = xrealloc(c->nf, c->cap * sizeof *c->nf);
		c->tag = xrealloc(c->tag, c->cap * sizeof *c->tag);
		ct_uq = xrealloc(ct_uq, c->cap * sizeof *ct_uq);
	}
	i = c->n++;
	c->kind[i] = (uint8_t)kind;
	c->qual[i] = 0;
	c->variadic[i] = 0;
	c->complete[i] = 1;
	c->base[i] = NOTOK;
	c->count[i] = 0;
	c->size[i] = 0;
	c->align[i] = 1;
	c->fst[i] = 0;
	c->nf[i] = 0;
	c->tag[i] = 0;
	ct_uq[i] = i;
	return i;
}

static void
ct_init(void)
{
	static const uint8_t sz[] = {1, 1, 1, 1, 1, 2, 2, 4, 4, 8, 8, 8, 8, 4, 8,
	    16};
	int k;

	if (CT.n)
		return;
	for (k = 0; k <= CT_LDOUBLE; k++) {
		uint32_t t = ct_new(k);
		CT.size[t] = sz[k];
		CT.align[t] = sz[k];
		if (k == CT_VOID)
			CT.complete[t] = 0;
	}
}

uint32_t
ct_basic(int kind)
{
	ct_init();
	return (uint32_t)kind;
}

uint32_t
ct_unqual(uint32_t t)
{
	return ct_uq[t];
}

uint32_t
ct_qual(uint32_t t, int q)
{
	uint32_t u = ct_uq[t], i;

	q |= CT.qual[t];
	if (q == 0)
		return u;
	for (i = 0; i < CT.n; i++)
		if (ct_uq[i] == u && CT.qual[i] == q)
			return i;
	i = ct_new(CT.kind[u]);
	CT.qual[i] = (uint8_t)q;
	CT.base[i] = CT.base[u];
	CT.count[i] = CT.count[u];
	CT.size[i] = CT.size[u];
	CT.align[i] = CT.align[u];
	CT.fst[i] = CT.fst[u];
	CT.nf[i] = CT.nf[u];
	CT.tag[i] = CT.tag[u];
	CT.complete[i] = CT.complete[u];
	CT.variadic[i] = CT.variadic[u];
	ct_uq[i] = u;
	return i;
}

uint32_t
ct_ptr(uint32_t base)
{
	uint32_t i;

	for (i = 0; i < CT.n; i++)
		if (CT.kind[i] == CT_PTR && CT.base[i] == base && CT.qual[i] == 0)
			return i;
	i = ct_new(CT_PTR);
	CT.base[i] = base;
	CT.size[i] = CT.align[i] = 8;
	return i;
}

uint32_t
ct_array(uint32_t base, int64_t n)
{
	uint32_t i;

	for (i = 0; i < CT.n; i++)
		if (CT.kind[i] == CT_ARRAY && CT.base[i] == base &&
		    CT.count[i] == n && CT.qual[i] == 0)
			return i;
	i = ct_new(CT_ARRAY);
	CT.base[i] = base;
	CT.count[i] = n;
	CT.size[i] = n >= 0 ? (uint32_t)(n * CT.size[base]) : 0;
	CT.align[i] = CT.align[base];
	CT.complete[i] = n >= 0;
	return i;
}

static uint32_t
ct_field(uint32_t name, uint32_t ty, uint32_t off)
{
	struct cf_types *c = &CT;

	if (c->nfl == c->fcap) {
		c->fcap = newcap(c->fcap, c->nfl + 1);
		c->fname = xrealloc(c->fname, c->fcap * sizeof *c->fname);
		c->fty = xrealloc(c->fty, c->fcap * sizeof *c->fty);
		c->foff = xrealloc(c->foff, c->fcap * sizeof *c->foff);
	}
	c->fname[c->nfl] = name;
	c->fty[c->nfl] = ty;
	c->foff[c->nfl] = off;
	return c->nfl++;
}

int
ct_is_int(uint32_t t)
{
	int k = CT.kind[t];

	return (k >= CT_BOOL && k <= CT_ULLONG) || k == CT_ENUM;
}

int
ct_is_lowered(uint32_t t)
{
	int k = CT.kind[t];

	return ct_is_int(t) || k == CT_FLOAT || k == CT_DOUBLE;
}

int
ct_is_arith(uint32_t t)
{
	return ct_is_int(t) || (CT.kind[t] >= CT_FLOAT &&
	    CT.kind[t] <= CT_LDOUBLE);
}

int
ct_is_scalar(uint32_t t)
{
	return ct_is_arith(t) || CT.kind[t] == CT_PTR;
}

int
ct_is_signed(uint32_t t)
{
	switch (CT.kind[t]) {
	case CT_CHAR: case CT_SCHAR: case CT_SHORT: case CT_INT: case CT_LONG:
	case CT_LLONG: case CT_FLOAT: case CT_DOUBLE: case CT_LDOUBLE:
		return 1;
	case CT_ENUM:
		return CT.count[ct_uq[t]] < 0;	/* count: -1 some negative */
	}
	return 0;
}

int
ct_rank(uint32_t t)
{
	switch (CT.kind[t]) {
	case CT_BOOL: return 0;
	case CT_CHAR: case CT_SCHAR: case CT_UCHAR: return 1;
	case CT_SHORT: case CT_USHORT: return 2;
	case CT_INT: case CT_UINT: case CT_ENUM: return 3;
	case CT_LONG: case CT_ULONG: return 4;
	case CT_LLONG: case CT_ULLONG: return 5;
	}
	return 6;
}

uint32_t
ct_promote(uint32_t t)
{
	t = ct_uq[t];
	if (!ct_is_int(t))
		return t;
	if (ct_rank(t) < 3)
		return ct_basic(CT_INT);
	if (CT.kind[t] == CT_ENUM)
		return ct_basic(ct_is_signed(t) ? CT_INT : CT_UINT);
	return t;
}

uint32_t
ct_usual(uint32_t a, uint32_t b)
{
	int ka, kb, ra, rb, sa, sb;

	a = ct_uq[a];
	b = ct_uq[b];
	ka = CT.kind[a];
	kb = CT.kind[b];
	if (ka == CT_LDOUBLE || kb == CT_LDOUBLE)
		return ct_basic(CT_LDOUBLE);
	if (ka == CT_DOUBLE || kb == CT_DOUBLE)
		return ct_basic(CT_DOUBLE);
	if (ka == CT_FLOAT || kb == CT_FLOAT)
		return ct_basic(CT_FLOAT);
	a = ct_promote(a);
	b = ct_promote(b);
	if (a == b)
		return a;
	ra = ct_rank(a);
	rb = ct_rank(b);
	sa = ct_is_signed(a);
	sb = ct_is_signed(b);
	if (sa == sb)
		return ra >= rb ? a : b;
	if (!sa && ra >= rb)
		return a;
	if (!sb && rb >= ra)
		return b;
	/* the signed type is wider: it holds every value of the other
	 * (LP64: int < long) */
	if (sa && CT.size[a] > CT.size[b])
		return a;
	if (sb && CT.size[b] > CT.size[a])
		return b;
	/* long vs unsigned long long ...: the unsigned of the signed one */
	return ct_basic(sa ? (CT.kind[a] == CT_LONG ? CT_ULONG : CT_ULLONG) :
	    (CT.kind[b] == CT_LONG ? CT_ULONG : CT_ULLONG));
}

ty_t
ct_to_ucas(uint32_t t)
{
	t = ct_uq[t];
	switch (CT.kind[t]) {
	case CT_BOOL: return TY_BOOL;
	case CT_CHAR: case CT_SCHAR: return TY(K_SINT, 8);
	case CT_UCHAR: return TY(K_UINT, 8);
	case CT_SHORT: return TY(K_SINT, 16);
	case CT_USHORT: return TY(K_UINT, 16);
	case CT_INT: return TY(K_SINT, 32);
	case CT_UINT: return TY(K_UINT, 32);
	case CT_LONG: case CT_LLONG: return TY(K_SINT, 64);
	case CT_ULONG: case CT_ULLONG: return TY(K_UINT, 64);
	case CT_ENUM: return ct_is_signed(t) ? TY(K_SINT, 32) : TY(K_UINT, 32);
	/* IEEE binary32/64 (x86-64 SSE: FLT_EVAL_METHOD 0) */
	case CT_FLOAT: return TY_F32;
	case CT_DOUBLE: return TY_F64;
	}
	return 0;
}

static const char *const basic_names[] = {
	"void", "_Bool", "char", "signed char", "unsigned char", "short",
	"unsigned short", "int", "unsigned int", "long", "unsigned long",
	"long long", "unsigned long long", "float", "double", "long double"
};

static void
quals(struct sbuf *b, int q)
{
	if (q & CQ_CONST)
		sb_puts(b, "const ");
	if (q & CQ_VOLATILE)
		sb_puts(b, "volatile ");
	if (q & CQ_RESTRICT)
		sb_puts(b, "restrict ");
}

void
ct_str(struct sbuf *b, uint32_t t, const char *name)
{
	struct sbuf in = {0}, tmp = {0};
	uint32_t k;

	sb_puts(&in, name ? name : "");
	for (;;) {
		int kind = CT.kind[t];
		if (kind == CT_PTR) {
			sb_clear(&tmp);
			sb_puts(&tmp, "*");
			if (CT.qual[t]) {
				quals(&tmp, CT.qual[t]);
			}
			sb_puts(&tmp, in.p);
			sb_clear(&in);
			sb_puts(&in, tmp.p);
			t = CT.base[t];
			if (CT.kind[t] == CT_ARRAY || CT.kind[t] == CT_FUNC) {
				sb_clear(&tmp);
				sb_printf(&tmp, "(%s)", in.p);
				sb_clear(&in);
				sb_puts(&in, tmp.p);
			}
			continue;
		}
		if (kind == CT_ARRAY) {
			if (CT.count[t] >= 0)
				sb_printf(&in, "[%lld]", (long long)CT.count[t]);
			else
				sb_puts(&in, "[]");
			t = CT.base[t];
			continue;
		}
		if (kind == CT_FUNC) {
			sb_putc(&in, '(');
			for (k = 0; k < CT.nf[t]; k++) {
				struct sbuf p = {0};
				ct_str(&p, CT.fty[CT.fst[t] + k], "");
				sb_printf(&in, "%s%s", k ? ", " : "", p.p);
				free(p.p);
			}
			if (CT.variadic[t])
				sb_puts(&in, CT.nf[t] ? ", ..." : "...");
			else if (CT.nf[t] == 0)
				sb_puts(&in, "void");
			sb_putc(&in, ')');
			t = CT.base[t];
			continue;
		}
		break;
	}
	quals(b, CT.qual[t] & ~CQ_RESTRICT);
	if (CT.kind[t] <= CT_LDOUBLE)
		sb_puts(b, basic_names[CT.kind[t]]);
	else if (CT.kind[t] == CT_STRUCT || CT.kind[t] == CT_UNION ||
	    CT.kind[t] == CT_ENUM || CT.kind[t] == CT_VEC)
		sb_printf(b, "%s %s", CT.kind[t] == CT_STRUCT || CT.kind[t] ==
		    CT_VEC ? "struct" : CT.kind[t] == CT_UNION ? "union" : "enum",
		    CT.tag[t] ? sym_str(CT.tag[t]) : "<anonymous>");
	if (in.n)
		sb_printf(b, " %s", in.p);
	free(in.p);
	free(tmp.p);
}

/* ---- the AST ---- */

uint32_t
ca_new(struct cf_ast *A, int kind)
{
	uint32_t i;

	if (A->n == A->cap) {
		A->cap = newcap(A->cap, A->n + 1);
		A->kind = xrealloc(A->kind, A->cap);
		A->ty = xrealloc(A->ty, A->cap * sizeof *A->ty);
		A->a = xrealloc(A->a, A->cap * sizeof *A->a);
		A->b = xrealloc(A->b, A->cap * sizeof *A->b);
		A->c = xrealloc(A->c, A->cap * sizeof *A->c);
		A->d = xrealloc(A->d, A->cap * sizeof *A->d);
		A->t0 = xrealloc(A->t0, A->cap * sizeof *A->t0);
		A->t1 = xrealloc(A->t1, A->cap * sizeof *A->t1);
		A->iv = xrealloc(A->iv, A->cap * sizeof *A->iv);
		A->fv = xrealloc(A->fv, A->cap * sizeof *A->fv);
	}
	i = A->n++;
	A->kind[i] = (uint8_t)kind;
	A->ty[i] = ct_basic(CT_VOID);
	A->a[i] = A->b[i] = A->c[i] = A->d[i] = 0;
	A->t0[i] = A->t1[i] = 0;
	A->iv[i] = 0;
	A->fv[i] = 0;
	return i;
}

/* ---- the parser ---- */

static struct {
	struct cf_unit *u;
	uint32_t i;
	jmp_buf jb;
	char err[400];
	/* the scope stack: ordinary identifiers and tags */
	uint32_t *sc_name, *sc_sym;
	uint8_t *sc_tag;
	uint32_t nsc, sccap;
	uint32_t depth;
	uint32_t fn_sym;	/* the function being parsed */
	uint32_t *lbuf;		/* list building */
	uint32_t nlbuf, lbcap;
} PS;

#define U (PS.u)
#define TK (U->tk)
#define A_ (U->A)

static void perr(const char *fmt, ...);

static void
perr(const char *fmt, ...)
{
	va_list ap;
	char m[300];
	uint32_t i = PS.i < TK.n ? PS.i : TK.n - 1;

	va_start(ap, fmt);
	vsnprintf(m, sizeof m, fmt, ap);
	va_end(ap);
	snprintf(PS.err, sizeof PS.err, "%s:%u:%u: %s", U->fname[TK.file[i]] ?
	    U->fname[TK.file[i]] : "?", TK.line[i], TK.col[i], m);
	longjmp(PS.jb, 1);
}

static const char *
tx(uint32_t i)
{
	return sym_str(TK.sym[i]);
}

static int
is(const char *s)
{
	return !strcmp(tx(PS.i), s) && TK.kind[PS.i] != CK_STR &&
	    TK.kind[PS.i] != CK_CHAR;
}

static int
is_at(uint32_t i, const char *s)
{
	return i < TK.n && !strcmp(tx(i), s) && TK.kind[i] != CK_STR &&
	    TK.kind[i] != CK_CHAR;
}

static int
accept(const char *s)
{
	if (is(s)) {
		PS.i++;
		return 1;
	}
	return 0;
}

static void
expect(const char *s)
{
	if (!accept(s))
		perr("expected '%s' before '%s'", s, TK.kind[PS.i] == CK_EOF ?
		    "end of input" : tx(PS.i));
}

static uint32_t
node(int kind, uint32_t t0)
{
	uint32_t n = ca_new(&A_, kind);

	A_.t0[n] = t0;
	A_.t1[n] = PS.i ? PS.i - 1 : 0;
	return n;
}

/* lists: build in lbuf from a mark, then copy to A.lst */
static uint32_t
lmark(void)
{
	return PS.nlbuf;
}

static void
lpush(uint32_t v)
{
	if (PS.nlbuf == PS.lbcap) {
		PS.lbcap = newcap(PS.lbcap, PS.nlbuf + 1);
		PS.lbuf = xrealloc(PS.lbuf, PS.lbcap * sizeof *PS.lbuf);
	}
	PS.lbuf[PS.nlbuf++] = v;
}

static uint32_t
lfinish(uint32_t mark, uint32_t *count)
{
	uint32_t n = PS.nlbuf - mark, st = A_.nlst, k;

	if (A_.nlst + n > A_.lcap) {
		A_.lcap = newcap(A_.lcap, A_.nlst + n);
		A_.lst = xrealloc(A_.lst, A_.lcap * sizeof *A_.lst);
	}
	for (k = 0; k < n; k++)
		A_.lst[st + k] = PS.lbuf[mark + k];
	A_.nlst += n;
	PS.nlbuf = mark;
	*count = n;
	return st;
}

/* ---- symbols and scopes ---- */

static uint32_t
sym_new(uint32_t name, uint32_t ty, int kind)
{
	struct cf_syms *S = &U->S;
	uint32_t i;

	if (S->n == S->cap) {
		S->cap = newcap(S->cap, S->n + 1);
		S->name = xrealloc(S->name, S->cap * sizeof *S->name);
		S->ty = xrealloc(S->ty, S->cap * sizeof *S->ty);
		S->scope = xrealloc(S->scope, S->cap * sizeof *S->scope);
		S->kind = xrealloc(S->kind, S->cap);
		S->stclass = xrealloc(S->stclass, S->cap);
		S->addr_taken = xrealloc(S->addr_taken, S->cap);
		S->assigned = xrealloc(S->assigned, S->cap);
		S->val = xrealloc(S->val, S->cap * sizeof *S->val);
		S->decl = xrealloc(S->decl, S->cap * sizeof *S->decl);
	}
	i = S->n++;
	S->name[i] = name;
	S->ty[i] = ty;
	S->scope[i] = PS.depth;
	S->kind[i] = (uint8_t)kind;
	S->stclass[i] = 0;
	S->addr_taken[i] = 0;
	S->assigned[i] = 0;
	S->val[i] = 0;
	S->decl[i] = 0;
	return i;
}

static void
sc_bind(uint32_t name, uint32_t sym, int tag)
{
	if (PS.nsc == PS.sccap) {
		PS.sccap = newcap(PS.sccap, PS.nsc + 1);
		PS.sc_name = xrealloc(PS.sc_name, PS.sccap * sizeof *PS.sc_name);
		PS.sc_sym = xrealloc(PS.sc_sym, PS.sccap * sizeof *PS.sc_sym);
		PS.sc_tag = xrealloc(PS.sc_tag, PS.sccap);
	}
	PS.sc_name[PS.nsc] = name;
	PS.sc_sym[PS.nsc] = sym;
	PS.sc_tag[PS.nsc] = (uint8_t)tag;
	PS.nsc++;
}

static uint32_t
sc_find(uint32_t name, int tag)
{
	uint32_t k = PS.nsc;

	while (k-- > 0)
		if (PS.sc_name[k] == name && PS.sc_tag[k] == tag)
			return PS.sc_sym[k];
	return NOTOK;
}

static uint32_t
sc_find_here(uint32_t name, int tag, uint32_t from)
{
	uint32_t k = PS.nsc;

	while (k-- > from)
		if (PS.sc_name[k] == name && PS.sc_tag[k] == tag)
			return PS.sc_sym[k];
	return NOTOK;
}

static int
is_typedef_name(uint32_t i)
{
	uint32_t s;

	if (TK.kind[i] != CK_ID)
		return 0;
	s = sc_find(TK.sym[i], 0);
	return s != NOTOK && U->S.kind[s] == CS_TYPEDEF;
}

static const char *const tyspec_words[] = {
	"void", "char", "short", "int", "long", "float", "double", "signed",
	"unsigned", "_Bool", "struct", "union", "enum", "const", "volatile",
	"restrict", "__restrict", "__restrict__", "typedef", "extern",
	"static", "auto", "register", "inline", "__inline", "__inline__",
	"_Noreturn", "__attribute__", "__extension__", "_Complex", NULL
};

static int
starts_type(uint32_t i)
{
	int k;

	if (TK.kind[i] != CK_ID)
		return 0;
	for (k = 0; tyspec_words[k]; k++)
		if (!strcmp(tx(i), tyspec_words[k]))
			return 1;
	return is_typedef_name(i);
}

static void
skip_attribute(void)
{
	int depth = 0;

	while (accept("__attribute__") || accept("__attribute")) {
		expect("(");
		depth = 1;
		while (depth && TK.kind[PS.i] != CK_EOF) {
			if (is("("))
				depth++;
			else if (is(")"))
				depth--;
			PS.i++;
		}
	}
}

/* ---- constant expressions ---- */

static int
const_int(uint32_t n, int64_t *v)
{
	int64_t a, b, c;
	uint32_t t = A_.ty[n];
	int uns = ct_is_int(t) && !ct_is_signed(t);
	int w = ct_is_int(t) ? (int)CT.size[ct_uq[t]] * 8 : 64;

	switch (A_.kind[n]) {
	case CA_NUM:
	case CA_CHARLIT:
		*v = A_.iv[n];
		return 1;
	case CA_CAST:
		if (!ct_is_int(t) || !const_int(A_.a[n], &a))
			return 0;
		*v = int_norm(ct_to_ucas(t), a);
		return 1;
	case CA_PLUS:
		return const_int(A_.a[n], v);
	case CA_NEG:
		if (!const_int(A_.a[n], &a))
			return 0;
		*v = int_norm(ct_to_ucas(t), (int64_t)(0 - (uint64_t)a));
		return 1;
	case CA_BNOT:
		if (!const_int(A_.a[n], &a))
			return 0;
		*v = int_norm(ct_to_ucas(t), ~a);
		return 1;
	case CA_LNOT:
		if (!const_int(A_.a[n], &a))
			return 0;
		*v = !a;
		return 1;
	case CA_COND:
		if (!const_int(A_.a[n], &a) || !const_int(A_.b[n], &b) ||
		    !const_int(A_.c[n], &c))
			return 0;
		*v = a ? b : c;
		return 1;
	case CA_LAND:
	case CA_LOR:
		if (!const_int(A_.a[n], &a) || !const_int(A_.b[n], &b))
			return 0;
		*v = A_.kind[n] == CA_LAND ? (a && b) : (a || b);
		return 1;
	case CA_BIN: {
		int op = (int)A_.iv[n];
		uint32_t ot = A_.ty[A_.a[n]];
		int ouns = ct_is_int(ot) && !ct_is_signed(ct_usual(ot,
		    A_.ty[A_.b[n]]));
		uint64_t ua, ub, m = w >= 64 ? ~0ULL : (1ULL << w) - 1;
		if (!const_int(A_.a[n], &a) || !const_int(A_.b[n], &b))
			return 0;
		ua = (uint64_t)a & m;
		ub = (uint64_t)b & m;
		switch (op) {
		case '+': *v = (int64_t)(ua + ub); break;
		case '-': *v = (int64_t)(ua - ub); break;
		case '*': *v = (int64_t)(ua * ub); break;
		case '/':
		case '%':
			if (b == 0)
				return 0;
			if (uns)
				*v = (int64_t)(op == '/' ? ua / ub : ua % ub);
			else if (b == -1)
				*v = op == '/' ? -a : 0;
			else
				*v = op == '/' ? a / b : a % b;
			break;
		case '&': *v = a & b; break;
		case '|': *v = a | b; break;
		case '^': *v = a ^ b; break;
		case CO_SHL: *v = (int64_t)((uint64_t)a << (b & 63)); break;
		case CO_SHR:
			*v = uns ? (int64_t)(ua >> (b & 63)) : a >> (b & 63);
			break;
		case '<': *v = ouns ? (uint64_t)a < (uint64_t)b : a < b; return 1;
		case '>': *v = ouns ? (uint64_t)a > (uint64_t)b : a > b; return 1;
		case CO_LE: *v = ouns ? (uint64_t)a <= (uint64_t)b : a <= b; return 1;
		case CO_GE: *v = ouns ? (uint64_t)a >= (uint64_t)b : a >= b; return 1;
		case CO_EQ: *v = a == b; return 1;
		case CO_NE: *v = a != b; return 1;
		default:
			return 0;
		}
		*v = int_norm(ct_to_ucas(t), *v);
		return 1;
	}
	}
	return 0;
}

/* ---- expressions ---- */

static uint32_t expr(void);
static uint32_t assign_expr(void);
static uint32_t cast_expr(void);
static uint32_t type_name(void);
static uint32_t initializer(uint32_t ty);

static int
is_lvalue(uint32_t n)
{
	int k = A_.kind[n];

	return k == CA_ID || k == CA_INDEX || k == CA_DEREF || k == CA_MEMBER ||
	    k == CA_ARROW || k == CA_STR || k == CA_COMPOUND;
}

/* the type of n as an rvalue: arrays and functions decay */
static uint32_t
rty(uint32_t n)
{
	uint32_t t = A_.ty[n];

	if (CT.kind[t] == CT_ARRAY)
		return ct_ptr(CT.base[t]);
	if (CT.kind[t] == CT_FUNC)
		return ct_ptr(t);
	return ct_unqual(t);
}

static uint32_t
num_literal(uint32_t t0)
{
	size_t len;
	const char *s = cf_tok_text(U, t0, &len);
	char buf[96];
	char *end;
	uint32_t n;
	int isf = 0, k;

	if (len >= sizeof buf)
		perr("number too long");
	memcpy(buf, s, len);
	buf[len] = 0;
	if (!(buf[0] == '0' && (buf[1] == 'x' || buf[1] == 'X'))) {
		for (k = 0; buf[k]; k++)
			if (buf[k] == '.' || buf[k] == 'e' || buf[k] == 'E')
				isf = 1;
	} else
		for (k = 0; buf[k]; k++)
			if (buf[k] == '.' || buf[k] == 'p' || buf[k] == 'P')
				isf = 1;
	if (isf) {
		double v = strtod(buf, &end);
		n = node(CA_FNUM, t0);
		A_.fv[n] = v;
		A_.ty[n] = ct_basic(CT_DOUBLE);
		if (*end == 'f' || *end == 'F') {
			/* rounded once, from the decimal (not through double) */
			A_.ty[n] = ct_basic(CT_FLOAT);
			A_.fv[n] = strtof(buf, NULL);
		} else if (*end == 'l' || *end == 'L')
			A_.ty[n] = ct_basic(CT_LDOUBLE);
		else if (*end)
			perr("bad number '%s'", buf);
		return n;
	}
	{
		uint64_t v;
		int u = 0, l = 0, dec = !(buf[0] == '0' && buf[1]);
		const int *cand;
		static const int dec_c[][4] = {
			{CT_INT, CT_LONG, CT_LLONG, -1},	/* no suffix */
			{CT_UINT, CT_ULONG, CT_ULLONG, -1},	/* u */
			{CT_LONG, CT_LLONG, -1, -1},		/* l */
			{CT_ULONG, CT_ULLONG, -1, -1},		/* ul */
			{CT_LLONG, -1, -1, -1},			/* ll */
			{CT_ULLONG, -1, -1, -1},		/* ull */
		};
		static const int oct_c[][6] = {
			{CT_INT, CT_UINT, CT_LONG, CT_ULONG, CT_LLONG, CT_ULLONG},
			{CT_UINT, CT_ULONG, CT_ULLONG, -1, -1, -1},
			{CT_LONG, CT_ULONG, CT_LLONG, CT_ULLONG, -1, -1},
			{CT_ULONG, CT_ULLONG, -1, -1, -1, -1},
			{CT_LLONG, CT_ULLONG, -1, -1, -1, -1},
			{CT_ULLONG, -1, -1, -1, -1, -1},
		};
		int row, j;
		v = strtoull(buf, &end, 0);
		for (; *end; end++) {
			if (*end == 'u' || *end == 'U')
				u = 1;
			else if (*end == 'l' || *end == 'L')
				l++;
			else
				perr("bad number '%s'", buf);
		}
		row = l == 0 ? (u ? 1 : 0) : l == 1 ? (u ? 3 : 2) : (u ? 5 : 4);
		n = node(CA_NUM, t0);
		A_.iv[n] = (int64_t)v;
		cand = dec ? dec_c[row] : oct_c[row];
		for (j = 0; j < (dec ? 4 : 6) && cand[j] >= 0; j++) {
			int kk = cand[j];
			int sg = kk == CT_INT || kk == CT_LONG || kk == CT_LLONG;
			int w = kk == CT_INT || kk == CT_UINT ? 32 : 64;
			uint64_t max = sg ? (w == 32 ? 0x7fffffffULL :
			    0x7fffffffffffffffULL) : (w == 32 ? 0xffffffffULL :
			    ~0ULL);
			if (v <= max) {
				A_.ty[n] = ct_basic(kk);
				return n;
			}
		}
		A_.ty[n] = ct_basic(u || !dec ? CT_ULLONG : CT_LLONG);
		return n;
	}
}

static int64_t
char_value(uint32_t t0)
{
	size_t len;
	const char *s = cf_tok_text(U, t0, &len), *e = s + len;

	while (s < e && *s != '\'')
		s++;
	s++;
	if (*s == '\\') {
		s++;
		switch (*s) {
		case 'n': return 10;
		case 't': return 9;
		case 'r': return 13;
		case '0': case '1': case '2': case '3': case '4': case '5':
		case '6': case '7': {
			int v = 0, k = 0;
			while (k < 3 && *s >= '0' && *s <= '7')
				v = v * 8 + (*s++ - '0'), k++;
			return (int64_t)(signed char)v;
		}
		case 'x': {
			int v = 0;
			s++;
			while (isxdigit((unsigned char)*s)) {
				v = v * 16 + (isdigit((unsigned char)*s) ? *s - '0' :
				    (tolower((unsigned char)*s) - 'a' + 10));
				s++;
			}
			return (int64_t)(signed char)v;
		}
		case 'a': return 7;
		case 'b': return 8;
		case 'f': return 12;
		case 'v': return 11;
		default: return (unsigned char)*s;
		}
	}
	return (int64_t)(signed char)*s;
}

static uint32_t
mk_cast_node(uint32_t e, uint32_t ty, uint32_t t0)
{
	uint32_t n = node(CA_CAST, t0);

	A_.a[n] = e;
	A_.ty[n] = ty;
	A_.t0[n] = t0;
	A_.t1[n] = A_.t1[e];
	A_.iv[n] = 1;	/* implicit */
	return n;
}

static uint32_t
primary(void)
{
	uint32_t t0 = PS.i, n, s;

	if (TK.kind[t0] == CK_NUM) {
		PS.i++;
		return num_literal(t0);
	}
	if (TK.kind[t0] == CK_CHAR) {
		PS.i++;
		n = node(CA_CHARLIT, t0);
		A_.iv[n] = char_value(t0);
		A_.ty[n] = ct_basic(CT_INT);
		return n;
	}
	if (TK.kind[t0] == CK_STR) {
		int64_t len = 0;
		while (TK.kind[PS.i] == CK_STR) {
			len += TK.len[PS.i] - 2;	/* escapes: an upper bound */
			PS.i++;
		}
		n = node(CA_STR, t0);
		A_.ty[n] = ct_array(ct_basic(CT_CHAR), len + 1);
		return n;
	}
	if (accept("(")) {
		n = expr();
		expect(")");
		A_.t0[n] = t0;	/* keep the parentheses in the span */
		A_.t1[n] = PS.i - 1;
		return n;
	}
	if (TK.kind[t0] == CK_ID) {
		PS.i++;
		s = sc_find(TK.sym[t0], 0);
		if (s == NOTOK) {
			/* an undeclared function call: implicit int f() */
			if (is("("))
				perr("'%s' is not declared (C99 has no implicit "
				    "declarations; #include its header)", tx(t0));
			perr("'%s' is not declared", tx(t0));
		}
		if (U->S.kind[s] == CS_TYPEDEF)
			perr("unexpected type name '%s'", tx(t0));
		if (U->S.kind[s] == CS_ENUMC) {
			n = node(CA_NUM, t0);
			A_.iv[n] = U->S.val[s];
			A_.ty[n] = ct_basic(CT_INT);
			return n;
		}
		n = node(CA_ID, t0);
		A_.iv[n] = s;
		A_.ty[n] = U->S.ty[s];
		return n;
	}
	perr("expected an expression before '%s'", TK.kind[t0] == CK_EOF ?
	    "end of input" : tx(t0));
	return 0;
}

static uint32_t
find_field(uint32_t st, uint32_t name, uint32_t *ty)
{
	uint32_t k, s = ct_unqual(st);

	if (CT.kind[s] != CT_STRUCT && CT.kind[s] != CT_UNION)
		perr("member access on a non-struct");
	if (!CT.complete[s])
		perr("member access on an incomplete type");
	for (k = 0; k < CT.nf[s]; k++)
		if (CT.fname[CT.fst[s] + k] == name) {
			*ty = CT.fty[CT.fst[s] + k];
			if (CT.qual[st])
				*ty = ct_qual(*ty, CT.qual[st] & ~CQ_RESTRICT);
			return CT.fst[s] + k;
		}
	/* anonymous members are not looked into */
	perr("no member named '%s'", sym_str(name));
	return 0;
}

static uint32_t
postfix(void)
{
	uint32_t t0 = PS.i, n = primary(), m;

	for (;;) {
		if (accept("[")) {
			uint32_t i = expr(), bt, it;
			expect("]");
			m = node(CA_INDEX, t0);
			bt = rty(n);
			it = rty(i);
			if (CT.kind[bt] != CT_PTR && CT.kind[it] == CT_PTR) {
				uint32_t x = n;
				n = i;
				i = x;
				bt = it;
			}
			if (CT.kind[bt] != CT_PTR)
				perr("subscript of a non-pointer");
			A_.a[m] = n;
			A_.b[m] = i;
			A_.ty[m] = CT.base[bt];
			n = m;
		} else if (accept("(")) {
			uint32_t mk = lmark(), cnt, ft = rty(n), k = 0;
			if (CT.kind[ft] == CT_PTR)
				ft = CT.base[ft];
			if (CT.kind[ft] != CT_FUNC)
				perr("call of a non-function");
			if (!is(")"))
				do {
					uint32_t a = assign_expr();
					if (k < CT.nf[ft]) {
						uint32_t pt = CT.fty[CT.fst[ft] + k];
						if (ct_is_arith(pt) && ct_unqual(pt) !=
						    rty(a))
							a = mk_cast_node(a, ct_unqual(pt),
							    A_.t0[a]);
					} else if (ct_is_arith(rty(a))) {
						/* default argument promotions */
						uint32_t pt = CT.kind[rty(a)] == CT_FLOAT ?
						    ct_basic(CT_DOUBLE) : ct_promote(rty(a));
						if (pt != rty(a))
							a = mk_cast_node(a, pt, A_.t0[a]);
					}
					lpush(a);
					k++;
				} while (accept(","));
			expect(")");
			if (k < CT.nf[ft] || (k > CT.nf[ft] && !CT.variadic[ft] &&
			    CT.nf[ft] > 0))
				perr("wrong number of arguments (%u, expected %u)",
				    k, CT.nf[ft]);
			m = node(CA_CALL, t0);
			A_.a[m] = n;
			A_.b[m] = lfinish(mk, &cnt);
			A_.c[m] = cnt;
			A_.ty[m] = CT.base[ft];
			n = m;
		} else if (is(".") || is("->")) {
			int arrow = is("->");
			uint32_t st, fty, f;
			PS.i++;
			if (TK.kind[PS.i] != CK_ID)
				perr("expected a member name");
			st = arrow ? rty(n) : A_.ty[n];
			if (arrow) {
				if (CT.kind[st] != CT_PTR)
					perr("-> on a non-pointer");
				st = CT.base[st];
			}
			f = find_field(st, TK.sym[PS.i], &fty);
			PS.i++;
			m = node(arrow ? CA_ARROW : CA_MEMBER, t0);
			A_.a[m] = n;
			A_.iv[m] = f;
			A_.ty[m] = fty;
			n = m;
		} else if (is("++") || is("--")) {
			int inc = is("++");
			PS.i++;
			if (!is_lvalue(n))
				perr("%s needs an lvalue", inc ? "++" : "--");
			m = node(inc ? CA_POSTINC : CA_POSTDEC, t0);
			A_.a[m] = n;
			A_.ty[m] = rty(n);
			n = m;
		} else
			return n;
	}
}

static uint32_t
unary(void)
{
	uint32_t t0 = PS.i, n, a;

	if (is("++") || is("--")) {
		int inc = is("++");
		PS.i++;
		a = unary();
		if (!is_lvalue(a))
			perr("%s needs an lvalue", inc ? "++" : "--");
		n = node(inc ? CA_PREINC : CA_PREDEC, t0);
		A_.a[n] = a;
		A_.ty[n] = rty(a);
		return n;
	}
	if (accept("&")) {
		a = cast_expr();
		if (!is_lvalue(a) && CT.kind[A_.ty[a]] != CT_FUNC)
			perr("& needs an lvalue");
		if (A_.kind[a] == CA_ID)
			U->S.addr_taken[A_.iv[a]] = 1;
		n = node(CA_ADDR, t0);
		A_.a[n] = a;
		A_.ty[n] = ct_ptr(A_.ty[a]);
		return n;
	}
	if (accept("*")) {
		uint32_t pt;
		a = cast_expr();
		pt = rty(a);
		if (CT.kind[pt] != CT_PTR)
			perr("* on a non-pointer");
		n = node(CA_DEREF, t0);
		A_.a[n] = a;
		A_.ty[n] = CT.base[pt];
		return n;
	}
	if (is("+") || is("-") || is("~") || is("!")) {
		char op = tx(PS.i)[0];
		PS.i++;
		a = cast_expr();
		n = node(op == '+' ? CA_PLUS : op == '-' ? CA_NEG : op == '~' ?
		    CA_BNOT : CA_LNOT, t0);
		A_.a[n] = a;
		if (op == '!')
			A_.ty[n] = ct_basic(CT_INT);
		else {
			if (!ct_is_arith(rty(a)) || (op == '~' &&
			    !ct_is_int(rty(a))))
				perr("bad operand of unary %c", op);
			A_.ty[n] = ct_promote(rty(a));
		}
		return n;
	}
	if (accept("sizeof")) {
		uint32_t ty;
		int64_t sz;
		if (is("(") && starts_type(PS.i + 1)) {
			PS.i++;
			ty = type_name();
			expect(")");
		} else {
			a = unary();
			ty = A_.ty[a];
		}
		if (!CT.complete[ty] && CT.kind[ty] != CT_ARRAY)
			perr("sizeof of an incomplete type");
		sz = CT.size[ty];
		n = node(CA_NUM, t0);
		A_.iv[n] = sz;
		A_.ty[n] = ct_basic(CT_ULONG);
		return n;
	}
	if (accept("_Alignof") || accept("__alignof__")) {
		uint32_t ty;
		expect("(");
		ty = type_name();
		expect(")");
		n = node(CA_NUM, t0);
		A_.iv[n] = CT.align[ty];
		A_.ty[n] = ct_basic(CT_ULONG);
		return n;
	}
	return postfix();
}

static uint32_t
cast_expr(void)
{
	uint32_t t0 = PS.i, ty, n, a;

	if (is("(") && starts_type(PS.i + 1)) {
		PS.i++;
		ty = type_name();
		expect(")");
		if (is("{")) {
			/* a compound literal */
			n = node(CA_COMPOUND, t0);
			a = initializer(ty);
			A_.a[n] = a;
			A_.ty[n] = ty;
			A_.t1[n] = PS.i - 1;
			return n;
		}
		a = cast_expr();
		n = node(CA_CAST, t0);
		A_.a[n] = a;
		A_.ty[n] = ct_unqual(ty);
		return n;
	}
	return unary();
}

static int
binop_prec(uint32_t i, int *op)
{
	const char *s = tx(i);

	if (TK.kind[i] != CK_PUNCT)
		return -1;
	if (!strcmp(s, "||")) return *op = 1, 1;
	if (!strcmp(s, "&&")) return *op = 2, 2;
	if (!strcmp(s, "|")) return *op = '|', 3;
	if (!strcmp(s, "^")) return *op = '^', 4;
	if (!strcmp(s, "&")) return *op = '&', 5;
	if (!strcmp(s, "==")) return *op = CO_EQ, 6;
	if (!strcmp(s, "!=")) return *op = CO_NE, 6;
	if (!strcmp(s, "<")) return *op = '<', 7;
	if (!strcmp(s, ">")) return *op = '>', 7;
	if (!strcmp(s, "<=")) return *op = CO_LE, 7;
	if (!strcmp(s, ">=")) return *op = CO_GE, 7;
	if (!strcmp(s, "<<")) return *op = CO_SHL, 8;
	if (!strcmp(s, ">>")) return *op = CO_SHR, 8;
	if (!strcmp(s, "+")) return *op = '+', 9;
	if (!strcmp(s, "-")) return *op = '-', 9;
	if (!strcmp(s, "*")) return *op = '*', 10;
	if (!strcmp(s, "/")) return *op = '/', 10;
	if (!strcmp(s, "%")) return *op = '%', 10;
	return -1;
}

/* the type of a op b; also used for compound assignment */
static uint32_t
bin_type(int op, uint32_t a, uint32_t b)
{
	uint32_t ta = rty(a), tb = rty(b);
	int pa = CT.kind[ta] == CT_PTR, pb = CT.kind[tb] == CT_PTR;

	switch (op) {
	case '+':
		if (pa && ct_is_int(tb))
			return ta;
		if (pb && ct_is_int(ta))
			return tb;
		break;
	case '-':
		if (pa && pb)
			return ct_basic(CT_LONG);
		if (pa && ct_is_int(tb))
			return ta;
		break;
	case '<': case '>': case CO_LE: case CO_GE: case CO_EQ: case CO_NE:
		if (pa || pb)
			return ct_basic(CT_INT);
		if (!ct_is_arith(ta) || !ct_is_arith(tb))
			perr("bad operands of a comparison");
		return ct_basic(CT_INT);
	case CO_SHL:
	case CO_SHR:
		if (!ct_is_int(ta) || !ct_is_int(tb))
			perr("shift needs integer operands");
		return ct_promote(ta);
	case '%': case '&': case '|': case '^':
		if (!ct_is_int(ta) || !ct_is_int(tb))
			perr("'%c' needs integer operands", op);
		return ct_usual(ta, tb);
	}
	if (!ct_is_arith(ta) || !ct_is_arith(tb))
		perr("bad operands of a binary operator");
	return ct_usual(ta, tb);
}

static uint32_t
binary(int minp)
{
	uint32_t t0 = PS.i, l = cast_expr(), r, n;
	int op, p;

	while ((p = binop_prec(PS.i, &op)) >= minp) {
		PS.i++;
		r = binary(p + 1);
		if (op == 1 || op == 2) {
			n = node(op == 1 ? CA_LOR : CA_LAND, t0);
			A_.ty[n] = ct_basic(CT_INT);
		} else {
			n = node(CA_BIN, t0);
			A_.iv[n] = op;
			A_.ty[n] = bin_type(op, l, r);
		}
		A_.a[n] = l;
		A_.b[n] = r;
		l = n;
	}
	return l;
}

static uint32_t
cond_expr(void)
{
	uint32_t t0 = PS.i, c = binary(1), a, b, n, ta, tb;

	if (!accept("?"))
		return c;
	a = expr();
	expect(":");
	b = cond_expr();
	n = node(CA_COND, t0);
	A_.a[n] = c;
	A_.b[n] = a;
	A_.c[n] = b;
	ta = rty(a);
	tb = rty(b);
	if (ct_is_arith(ta) && ct_is_arith(tb))
		A_.ty[n] = ct_usual(ta, tb);
	else if (CT.kind[ta] == CT_PTR)
		A_.ty[n] = ta;
	else
		A_.ty[n] = tb;
	return n;
}

static int
assign_op(uint32_t i, int *op)
{
	static const char *const ops[] = {"=", "+=", "-=", "*=", "/=", "%=",
	    "&=", "|=", "^=", "<<=", ">>=", NULL};
	static const int codes[] = {'=', '+', '-', '*', '/', '%', '&', '|',
	    '^', CO_SHL, CO_SHR};
	int k;

	if (TK.kind[i] != CK_PUNCT)
		return 0;
	for (k = 0; ops[k]; k++)
		if (!strcmp(tx(i), ops[k])) {
			*op = codes[k];
			return 1;
		}
	return 0;
}

static uint32_t
assign_expr(void)
{
	uint32_t t0 = PS.i, l = cond_expr(), r, n;
	int op;

	if (!assign_op(PS.i, &op))
		return l;
	PS.i++;
	if (!is_lvalue(l))
		perr("assignment to a non-lvalue");
	if (CT.qual[A_.ty[l]] & CQ_CONST)
		perr("assignment to a const object");
	r = assign_expr();
	if (op != '=')
		(void)bin_type(op, l, r);
	n = node(CA_ASSIGN, t0);
	A_.iv[n] = op;
	A_.a[n] = l;
	A_.b[n] = r;
	A_.ty[n] = ct_unqual(A_.ty[l]);
	if (A_.kind[l] == CA_ID)
		U->S.assigned[A_.iv[l]] = 1;
	return n;
}

static uint32_t
expr(void)
{
	uint32_t t0 = PS.i, l = assign_expr(), r, n;

	while (accept(",")) {
		r = assign_expr();
		n = node(CA_COMMA, t0);
		A_.a[n] = l;
		A_.b[n] = r;
		A_.ty[n] = rty(r);
		l = n;
	}
	return l;
}

/* ---- declarations ---- */

struct dspec {
	uint32_t ty;
	int stc;	/* 't' typedef, 's' static, 'e' extern */
	int inl;
};

static uint32_t struct_spec(int kind);
static uint32_t enum_spec(void);

static void
decl_spec(struct dspec *d)
{
	int nlong = 0, nsigned = 0, nunsigned = 0, base = -1, q = 0, any = 0;
	uint32_t ty = NOTOK;

	memset(d, 0, sizeof *d);
	for (;;) {
		skip_attribute();
		if (accept("__extension__"))
			continue;
		if (accept("typedef")) { d->stc = 't'; any = 1; continue; }
		if (accept("extern")) { d->stc = 'e'; any = 1; continue; }
		if (accept("static")) { d->stc = 's'; any = 1; continue; }
		if (accept("auto") || accept("register")) { any = 1; continue; }
		if (accept("inline") || accept("__inline") ||
		    accept("__inline__") || accept("_Noreturn")) {
			d->inl = 1;
			any = 1;
			continue;
		}
		if (accept("const")) { q |= CQ_CONST; any = 1; continue; }
		if (accept("volatile")) { q |= CQ_VOLATILE; any = 1; continue; }
		if (accept("restrict") || accept("__restrict") ||
		    accept("__restrict__")) {
			q |= CQ_RESTRICT;
			any = 1;
			continue;
		}
		if (accept("long")) { nlong++; any = 1; continue; }
		if (accept("signed") || accept("__signed__")) {
			nsigned = 1;
			any = 1;
			continue;
		}
		if (accept("unsigned")) { nunsigned = 1; any = 1; continue; }
		if (accept("_Complex"))
			perr("_Complex is not supported");
		if (base < 0 && ty == NOTOK) {
			static const char *const bw[] = {"void", "char", "short",
			    "int", "float", "double", "_Bool", NULL};
			int k, hit = 0;
			for (k = 0; bw[k]; k++)
				if (accept(bw[k])) {
					base = k;
					hit = 1;
					break;
				}
			if (hit) {
				any = 1;
				continue;
			}
			if (accept("struct")) {
				ty = struct_spec(CT_STRUCT);
				any = 1;
				continue;
			}
			if (accept("union")) {
				ty = struct_spec(CT_UNION);
				any = 1;
				continue;
			}
			if (accept("enum")) {
				ty = enum_spec();
				any = 1;
				continue;
			}
			if (nlong == 0 && !nsigned && !nunsigned &&
			    is_typedef_name(PS.i)) {
				uint32_t s = sc_find(TK.sym[PS.i], 0);
				ty = U->S.ty[s];
				PS.i++;
				any = 1;
				continue;
			}
		} else if (base == 1 /* char */ || base == 2 || base == 3) {
			/* "short int", "long int" handled by the flags */
			if (accept("int"))
				continue;
		}
		break;
	}
	if (!any)
		perr("expected a declaration");
	if (ty == NOTOK) {
		int k;
		switch (base) {
		case 0: k = CT_VOID; break;
		case 1: k = nunsigned ? CT_UCHAR : nsigned ? CT_SCHAR : CT_CHAR;
			break;
		case 2: k = nunsigned ? CT_USHORT : CT_SHORT; break;
		case 4: k = CT_FLOAT; break;
		case 5: k = nlong ? CT_LDOUBLE : CT_DOUBLE; break;
		case 6: k = CT_BOOL; break;
		default:
			if (nlong >= 2)
				k = nunsigned ? CT_ULLONG : CT_LLONG;
			else if (nlong == 1)
				k = nunsigned ? CT_ULONG : CT_LONG;
			else
				k = nunsigned ? CT_UINT : CT_INT;
		}
		ty = ct_basic(k);
	}
	d->ty = q ? ct_qual(ty, q) : ty;
}

static uint32_t declarator(uint32_t base, uint32_t *name, uint32_t *ntok,
    uint32_t *params, uint32_t *nparams);

static uint32_t
struct_spec(int kind)
{
	uint32_t tag = 0, t = NOTOK, s;
	uint32_t off = 0, align = 1, mk;

	skip_attribute();
	if (TK.kind[PS.i] == CK_ID) {
		tag = TK.sym[PS.i];
		PS.i++;
		s = sc_find(tag, 1);
		if (s != NOTOK && CT.kind[U->S.ty[s]] == kind)
			t = U->S.ty[s];
	}
	if (!is("{")) {
		if (t == NOTOK) {
			t = ct_new(kind);
			CT.tag[t] = tag;
			CT.complete[t] = 0;
			s = sym_new(tag, t, CS_TYPEDEF);
			sc_bind(tag, s, 1);
		}
		return t;
	}
	PS.i++;
	/* a definition: a new type unless an incomplete one of this scope */
	if (t == NOTOK || CT.complete[t] || (tag && sc_find_here(tag, 1,
	    0) == NOTOK)) {
		t = ct_new(kind);
		CT.tag[t] = tag;
		if (tag) {
			s = sym_new(tag, t, CS_TYPEDEF);
			sc_bind(tag, s, 1);
		}
	}
	mk = CT.nfl;
	{
		uint32_t fst = CT.nfl, nf = 0, maxsz = 0;
		/* fields are appended contiguously: collect then copy */
		uint32_t fn[256], ft[256], k;
		while (!accept("}")) {
			struct dspec d;
			decl_spec(&d);
			if (accept(";"))
				continue;	/* an anonymous struct member */
			do {
				uint32_t name = 0, nt, fty;
				if (is(":"))
					fty = d.ty;
				else
					fty = declarator(d.ty, &name, &nt, NULL, NULL);
				if (accept(":")) {
					(void)cond_expr();	/* a bit-field */
					CT.complete[t] = 1;
					fty = ct_basic(CT_INT) == fty ? fty : fty;
					CT.variadic[t] = 2;	/* has bit-fields */
				}
				if (nf == 256)
					perr("too many fields");
				fn[nf] = name;
				ft[nf] = fty;
				nf++;
				skip_attribute();
			} while (accept(","));
			expect(";");
		}
		fst = CT.nfl;
		for (k = 0; k < nf; k++) {
			uint32_t sz = CT.size[ft[k]], al = CT.align[ft[k]];
			if (al == 0)
				al = 1;
			if (kind == CT_STRUCT) {
				off = (off + al - 1) / al * al;
				ct_field(fn[k], ft[k], off);
				off += sz;
			} else {
				ct_field(fn[k], ft[k], 0);
				if (sz > maxsz)
					maxsz = sz;
			}
			if (al > align)
				align = al;
		}
		if (kind == CT_UNION)
			off = maxsz;
		off = (off + align - 1) / align * align;
		CT.fst[t] = fst;
		CT.nf[t] = nf;
		CT.size[t] = off;
		CT.align[t] = align;
		CT.complete[t] = 1;
		(void)mk;
	}
	skip_attribute();
	return t;
}

static uint32_t
enum_spec(void)
{
	uint32_t tag = 0, t, s;
	int64_t v = 0;
	int neg = 0;

	if (TK.kind[PS.i] == CK_ID) {
		tag = TK.sym[PS.i];
		PS.i++;
	}
	if (!accept("{")) {
		s = tag ? sc_find(tag, 1) : NOTOK;
		if (s != NOTOK)
			return U->S.ty[s];
		t = ct_new(CT_ENUM);
		CT.size[t] = CT.align[t] = 4;
		CT.tag[t] = tag;
		return t;
	}
	t = ct_new(CT_ENUM);
	CT.size[t] = CT.align[t] = 4;
	CT.tag[t] = tag;
	if (tag) {
		s = sym_new(tag, t, CS_TYPEDEF);
		sc_bind(tag, s, 1);
	}
	while (!accept("}")) {
		uint32_t name;
		if (TK.kind[PS.i] != CK_ID)
			perr("expected an enumerator");
		name = TK.sym[PS.i];
		PS.i++;
		if (accept("=")) {
			uint32_t e = cond_expr();
			if (!const_int(e, &v))
				perr("an enumerator needs a constant");
		}
		if (v < 0)
			neg = 1;
		s = sym_new(name, ct_basic(CT_INT), CS_ENUMC);
		U->S.val[s] = v;
		sc_bind(name, s, 0);
		v++;
		if (!accept(","))
			expect("}"), PS.i--, (void)0;
		if (is("}")) {
			PS.i++;
			break;
		}
	}
	CT.count[t] = neg ? -1 : 0;
	return t;
}

/* parameter list after '(' ; returns the function type */
static uint32_t
param_list(uint32_t ret, uint32_t *params, uint32_t *nparams)
{
	uint32_t pt[64], pn[64], np = 0, k, t;
	int variadic = 0;

	if (is(")")) {
		PS.i++;
		t = ct_new(CT_FUNC);
		CT.base[t] = ret;
		CT.variadic[t] = 1;	/* f(): unspecified parameters */
		CT.fst[t] = CT.nfl;
		if (nparams)
			*nparams = 0;
		return t;
	}
	if (is("void") && is_at(PS.i + 1, ")")) {
		PS.i += 2;
		t = ct_new(CT_FUNC);
		CT.base[t] = ret;
		CT.fst[t] = CT.nfl;
		if (nparams)
			*nparams = 0;
		return t;
	}
	for (;;) {
		struct dspec d;
		uint32_t name = 0, nt = 0, ty;
		if (accept("...")) {
			variadic = 1;
			break;
		}
		if (!starts_type(PS.i))
			perr("expected a parameter type (K&R definitions are not "
			    "supported)");
		decl_spec(&d);
		ty = declarator(d.ty, &name, &nt, NULL, NULL);
		/* adjust: arrays and functions become pointers */
		if (CT.kind[ty] == CT_ARRAY)
			ty = ct_qual(ct_ptr(CT.base[ty]), CT.qual[ty]);
		else if (CT.kind[ty] == CT_FUNC)
			ty = ct_ptr(ty);
		if (np == 64)
			perr("too many parameters");
		pt[np] = ty;
		pn[np] = name;
		if (params)
			params[np] = nt;
		np++;
		if (!accept(","))
			break;
	}
	expect(")");
	t = ct_new(CT_FUNC);
	CT.base[t] = ret;
	CT.variadic[t] = (uint8_t)variadic;
	CT.fst[t] = CT.nfl;
	for (k = 0; k < np; k++)
		ct_field(pn[k], pt[k], 0);
	CT.nf[t] = np;
	if (nparams)
		*nparams = np;
	return t;
}

/* the suffixes of a direct declarator: [n] and (params) */
static uint32_t
suffixes(uint32_t base, uint32_t *params, uint32_t *nparams)
{
	if (accept("[")) {
		int64_t n = -1;
		uint32_t t;
		while (accept("static") || accept("const") || accept("restrict") ||
		    accept("volatile") || accept("__restrict"))
			;
		if (!is("]")) {
			uint32_t e;
			if (accept("*"))
				e = 0;
			else {
				e = assign_expr();
				if (!const_int(e, &n))
					n = -1;	/* a VLA: unknown size */
			}
		}
		expect("]");
		t = suffixes(base, NULL, NULL);
		return ct_array(t, n);
	}
	if (accept("(")) {
		uint32_t t = param_list(NOTOK, params, nparams), r;
		r = suffixes(base, NULL, NULL);
		CT.base[t] = r;
		return t;
	}
	return base;
}

static uint32_t
declarator(uint32_t base, uint32_t *name, uint32_t *ntok, uint32_t *params,
    uint32_t *nparams)
{
	skip_attribute();
	while (accept("*")) {
		int q = 0;
		base = ct_ptr(base);
		for (;;) {
			if (accept("const"))
				q |= CQ_CONST;
			else if (accept("volatile"))
				q |= CQ_VOLATILE;
			else if (accept("restrict") || accept("__restrict") ||
			    accept("__restrict__"))
				q |= CQ_RESTRICT;
			else if (is("__attribute__"))
				skip_attribute();
			else
				break;
		}
		if (q)
			base = ct_qual(base, q);
	}
	if (is("(") && !starts_type(PS.i + 1) && !is_at(PS.i + 1, ")")) {
		/* a nested declarator: parse the suffixes after it first */
		uint32_t save = PS.i + 1, end, t;
		int depth = 1;
		PS.i++;
		while (depth && TK.kind[PS.i] != CK_EOF) {
			if (is("("))
				depth++;
			else if (is(")"))
				depth--;
			PS.i++;
		}
		base = suffixes(base, NULL, NULL);
		end = PS.i;
		PS.i = save;
		t = declarator(base, name, ntok, params, nparams);
		expect(")");
		PS.i = end;
		return t;
	}
	if (TK.kind[PS.i] == CK_ID && !starts_type(PS.i)) {
		if (name)
			*name = TK.sym[PS.i];
		if (ntok)
			*ntok = PS.i;
		PS.i++;
	} else if (TK.kind[PS.i] == CK_ID && is_typedef_name(PS.i) && name &&
	    !is_at(PS.i + 1, ",") && !is_at(PS.i + 1, ")")) {
		/* a typedef name redeclared as an object */
		*name = TK.sym[PS.i];
		if (ntok)
			*ntok = PS.i;
		PS.i++;
	}
	return suffixes(base, params, nparams);
}

static uint32_t
type_name(void)
{
	struct dspec d;

	decl_spec(&d);
	return declarator(d.ty, NULL, NULL, NULL, NULL);
}

/* initializers: an expression or a braced list (designators kept) */
static uint32_t
initializer(uint32_t ty)
{
	uint32_t t0 = PS.i, n, mk, cnt;

	if (!accept("{")) {
		uint32_t e = assign_expr();
		if (ct_is_arith(ty) && rty(e) != ct_unqual(ty) &&
		    ct_is_arith(rty(e)))
			e = mk_cast_node(e, ct_unqual(ty), A_.t0[e]);
		return e;
	}
	mk = lmark();
	while (!accept("}")) {
		uint32_t et = ty;
		int desig = 0;
		while (is("[") || is(".")) {
			desig = 1;
			if (accept("[")) {
				(void)cond_expr();
				expect("]");
			} else {
				PS.i++;
				PS.i++;
			}
		}
		if (desig)
			expect("=");
		if (CT.kind[ty] == CT_ARRAY)
			et = CT.base[ty];
		else if ((CT.kind[ty] == CT_STRUCT || CT.kind[ty] == CT_UNION) &&
		    CT.nf[ct_unqual(ty)] > 0)
			et = CT.fty[CT.fst[ct_unqual(ty)]];
		lpush(initializer(desig ? ct_basic(CT_VOID) : et));
		if (desig)
			A_.iv[PS.nlbuf ? PS.lbuf[PS.nlbuf - 1] : 0] |= 0;
		if (desig)
			U->A.d[PS.lbuf[PS.nlbuf - 1]] = 1;
		if (!accept(",")) {
			expect("}");
			break;
		}
	}
	n = node(CA_INIT, t0);
	A_.a[n] = lfinish(mk, &cnt);
	A_.b[n] = cnt;
	A_.ty[n] = ty;
	return n;
}

static uint32_t stmt(void);

static uint32_t
block_items(uint32_t t0)
{
	uint32_t mk = lmark(), n, cnt, sc = PS.nsc;

	PS.depth++;
	while (!accept("}")) {
		if (TK.kind[PS.i] == CK_EOF)
			perr("expected '}'");
		lpush(stmt());
	}
	PS.depth--;
	PS.nsc = sc;
	n = node(CA_BLOCK, t0);
	A_.a[n] = lfinish(mk, &cnt);
	A_.b[n] = cnt;
	return n;
}

/*
 * A declaration in a block (or at file scope with top = 1): one CA_DECL
 * per declarator, several wrapped in a transparent block (iv = 1).
 */
static uint32_t
declaration(int top)
{
	uint32_t t0 = PS.i, mk = lmark(), cnt, n;
	struct dspec d;

	decl_spec(&d);
	if (accept(";")) {
		n = node(CA_EMPTY, t0);
		return n;
	}
	do {
		uint32_t name = 0, nt = 0, ty, s, dn, pt[64], np = 0, prev;
		uint32_t dt0 = PS.i;
		ty = declarator(d.ty, &name, &nt, pt, &np);
		skip_attribute();
		if (name == 0)
			perr("a declaration without a name");
		if (d.stc == 't') {
			s = sym_new(name, ty, CS_TYPEDEF);
			sc_bind(name, s, 0);
			dn = node(CA_TYPEDEF, dt0);
			A_.iv[dn] = s;
			lpush(dn);
			continue;
		}
		if (CT.kind[ty] == CT_FUNC) {
			prev = sc_find(name, 0);
			if (prev != NOTOK && U->S.kind[prev] == CS_FUNC)
				s = prev;
			else {
				s = sym_new(name, ty, CS_FUNC);
				sc_bind(name, s, 0);
			}
			U->S.stclass[s] = (uint8_t)d.stc;
			if (top && is("{")) {
				/* a function definition */
				uint32_t fn = node(CA_FUNC, t0), body, k, sc = PS.nsc,
				    mkp = lmark(), c2;
				U->S.ty[s] = ty;
				U->S.decl[s] = fn;
				PS.depth++;
				for (k = 0; k < CT.nf[ty]; k++) {
					uint32_t ps = sym_new(CT.fname[CT.fst[ty] + k],
					    CT.fty[CT.fst[ty] + k], CS_PARAM);
					if (CT.fname[CT.fst[ty] + k])
						sc_bind(CT.fname[CT.fst[ty] + k], ps, 0);
					lpush(ps);
				}
				PS.fn_sym = s;
				{
					uint32_t bt0 = PS.i;
					PS.i++;
					body = block_items(bt0);
				}
				PS.depth--;
				PS.nsc = sc;
				A_.iv[fn] = s;
				A_.a[fn] = body;
				A_.c[fn] = lfinish(mkp, &c2);
				A_.d[fn] = c2;
				A_.t1[fn] = PS.i - 1;
				A_.ty[fn] = ty;
				return fn;
			}
			dn = node(CA_DECL, dt0);
			A_.iv[dn] = s;
			lpush(dn);
			continue;
		}
		if (CT.kind[ty] == CT_VOID)
			perr("a variable of type void");
		prev = top ? sc_find(name, 0) : NOTOK;
		if (prev != NOTOK && U->S.kind[prev] == CS_GLOBAL)
			s = prev;
		else {
			s = sym_new(name, ty, top ? CS_GLOBAL : CS_LOCAL);
			if (!top && (d.stc == 's' || d.stc == 'e'))
				U->S.kind[s] = CS_GLOBAL;	/* static storage */
			sc_bind(name, s, 0);
		}
		U->S.stclass[s] = (uint8_t)d.stc;
		dn = node(CA_DECL, dt0);
		A_.iv[dn] = s;
		U->S.decl[s] = dn;
		if (accept("=")) {
			uint32_t in = initializer(ty);
			A_.a[dn] = in;
			if (CT.kind[ty] == CT_ARRAY && CT.count[ty] < 0 &&
			    A_.kind[A_.a[dn]] == CA_INIT) {
				ty = ct_array(CT.base[ty], A_.b[A_.a[dn]]);
				U->S.ty[s] = ty;
			} else if (CT.kind[ty] == CT_ARRAY && CT.count[ty] < 0 &&
			    A_.kind[A_.a[dn]] == CA_STR) {
				ty = ct_array(CT.base[ty],
				    CT.count[A_.ty[A_.a[dn]]]);
				U->S.ty[s] = ty;
			}
		}
		A_.t1[dn] = PS.i - 1;
		lpush(dn);
	} while (accept(","));
	expect(";");
	if (PS.nlbuf - mk == 1) {
		n = PS.lbuf[mk];
		PS.nlbuf = mk;
		A_.t0[n] = t0;
		A_.t1[n] = PS.i - 1;
		return n;
	}
	n = node(CA_BLOCK, t0);
	A_.a[n] = lfinish(mk, &cnt);
	A_.b[n] = cnt;
	A_.iv[n] = 1;	/* transparent: no scope */
	return n;
}

static uint32_t
stmt(void)
{
	uint32_t t0 = PS.i, n, a;

	if (TK.kind[PS.i] == CK_ID && is_at(PS.i + 1, ":") &&
	    !is("default") && !starts_type(PS.i)) {
		uint32_t lab = TK.sym[PS.i];
		PS.i += 2;
		n = node(CA_LABEL, t0);
		A_.iv[n] = lab;
		a = is("}") ? node(CA_EMPTY, PS.i) : stmt();
		A_.a[n] = a;
		A_.t1[n] = PS.i - 1;
		return n;
	}
	if (accept("{"))
		return block_items(t0);
	if (accept(";"))
		return node(CA_EMPTY, t0);
	if (accept("if")) {
		uint32_t c, a, b = 0;
		expect("(");
		c = expr();
		expect(")");
		a = stmt();
		if (accept("else"))
			b = stmt();
		n = node(CA_IF, t0);
		A_.a[n] = c;
		A_.b[n] = a;
		A_.c[n] = b;
		return n;
	}
	if (accept("while")) {
		uint32_t c, b;
		expect("(");
		c = expr();
		expect(")");
		b = stmt();
		n = node(CA_WHILE, t0);
		A_.a[n] = c;
		A_.b[n] = b;
		return n;
	}
	if (accept("do")) {
		uint32_t c, b;
		b = stmt();
		expect("while");
		expect("(");
		c = expr();
		expect(")");
		expect(";");
		n = node(CA_DO, t0);
		A_.a[n] = b;
		A_.b[n] = c;
		return n;
	}
	if (accept("for")) {
		uint32_t i = 0, c = 0, s = 0, b, sc = PS.nsc;
		expect("(");
		PS.depth++;
		if (starts_type(PS.i))
			i = declaration(0);
		else if (!accept(";")) {
			uint32_t e0 = PS.i;
			i = node(CA_EXPR, e0);
			a = expr();
			A_.a[i] = a;
			expect(";");
			A_.t1[i] = PS.i - 1;
		}
		if (!is(";"))
			c = expr();
		expect(";");
		if (!is(")"))
			s = expr();
		expect(")");
		b = stmt();
		PS.depth--;
		PS.nsc = sc;
		n = node(CA_FOR, t0);
		A_.a[n] = i;
		A_.b[n] = c;
		A_.c[n] = s;
		A_.d[n] = b;
		return n;
	}
	if (accept("return")) {
		uint32_t e = 0, rt = CT.base[U->S.ty[PS.fn_sym]];
		if (!is(";")) {
			e = expr();
			if (ct_is_arith(rt) && ct_is_arith(rty(e)) &&
			    rty(e) != ct_unqual(rt))
				e = mk_cast_node(e, ct_unqual(rt), A_.t0[e]);
		}
		expect(";");
		n = node(CA_RETURN, t0);
		A_.a[n] = e;
		return n;
	}
	if (accept("break")) {
		expect(";");
		return node(CA_BREAK, t0);
	}
	if (accept("continue")) {
		expect(";");
		return node(CA_CONTINUE, t0);
	}
	if (accept("goto")) {
		n = node(CA_GOTO, t0);
		A_.iv[n] = TK.sym[PS.i];
		PS.i++;
		expect(";");
		A_.t1[n] = PS.i - 1;
		return n;
	}
	if (accept("switch")) {
		uint32_t c, b;
		expect("(");
		c = expr();
		expect(")");
		b = stmt();
		n = node(CA_SWITCH, t0);
		A_.a[n] = c;
		A_.b[n] = b;
		return n;
	}
	if (accept("case")) {
		uint32_t e = cond_expr();
		int64_t v;
		if (!const_int(e, &v))
			perr("a case label needs an integer constant");
		expect(":");
		n = node(CA_CASE, t0);
		A_.iv[n] = v;
		a = is("}") ? node(CA_EMPTY, PS.i) : stmt();
		A_.a[n] = a;
		A_.t1[n] = PS.i - 1;
		return n;
	}
	if (accept("default")) {
		expect(":");
		n = node(CA_DEFAULT, t0);
		a = is("}") ? node(CA_EMPTY, PS.i) : stmt();
		A_.a[n] = a;
		A_.t1[n] = PS.i - 1;
		return n;
	}
	if (is("asm") || is("__asm__") || is("__asm"))
		perr("inline assembly is not supported");
	if (starts_type(PS.i) && !(is_typedef_name(PS.i) &&
	    (is_at(PS.i + 1, "=") || is_at(PS.i + 1, "("))))
		return declaration(0);
	n = node(CA_EXPR, t0);
	a = expr();
	A_.a[n] = a;
	expect(";");
	A_.t1[n] = PS.i - 1;
	return n;
}

int
cf_parse(struct cf_unit *u)
{
	uint32_t mk, cnt, nfn = 0;
	int k;

	ct_init();
	memset(&PS, 0, sizeof PS);
	PS.u = u;
	ca_new(&u->A, CA_EMPTY);	/* node 0: "none" */
	if (setjmp(PS.jb)) {
		sb_puts(&u->err, PS.err);
		free(PS.sc_name);
		free(PS.sc_sym);
		free(PS.sc_tag);
		free(PS.lbuf);
		return 0;
	}
	/* builtins the lowering knows: __builtin_popcount ... are not
	 * declared (they are GNU); the prelude covers the headers */
	mk = lmark();
	while (TK.kind[PS.i] != CK_EOF) {
		if (accept(";"))
			continue;
		lpush(declaration(1));
	}
	u->top = lfinish(mk, &cnt);
	u->A.d[0] = cnt;	/* node 0's d: the top-level count */
	for (k = 0; k < (int)cnt; k++)
		if (u->A.kind[u->A.lst[u->top + k]] == CA_FUNC)
			nfn++;
	u->fns = xmalloc((nfn + 1) * sizeof *u->fns);
	u->nfn = 0;
	for (k = 0; k < (int)cnt; k++)
		if (u->A.kind[u->A.lst[u->top + k]] == CA_FUNC)
			u->fns[u->nfn++] = u->A.lst[u->top + k];
	free(PS.sc_name);
	free(PS.sc_sym);
	free(PS.sc_tag);
	free(PS.lbuf);
	return 1;
}

int
cf_span(const struct cf_unit *u, uint32_t node, uint32_t *from,
    uint32_t *to)
{
	return cf_tspan(u, u->A.t0[node], u->A.t1[node], from, to);
}

void
cf_unit_free(struct cf_unit *u)
{
	int k;
	struct cf_toks *t = &u->tk;
	struct cf_ast *A = &u->A;
	struct cf_syms *S = &u->S;

	for (k = 0; k < u->nfiles; k++) {
		free(u->ftext[k]);
		free(u->fname[k]);
	}
	free(u->ftext);
	free(u->fname);
	free(t->kind); free(t->fl); free(t->file); free(t->off); free(t->len);
	free(t->line); free(t->col); free(t->sym);
	free(u->tk_o0); free(u->tk_o1); free(u->tk_grp);
	free(A->kind); free(A->ty); free(A->a); free(A->b); free(A->c);
	free(A->d); free(A->t0); free(A->t1); free(A->iv); free(A->fv);
	free(A->lst);
	free(S->name); free(S->ty); free(S->scope); free(S->kind);
	free(S->stclass); free(S->addr_taken); free(S->assigned); free(S->val);
	free(S->decl);
	free(u->fns);
	free(u->includes.p);
	free(u->err.p);
	free(u->macros);
	memset(u, 0, sizeof *u);
}
