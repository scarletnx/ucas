/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: nodes to C99.
 *
 * The node semantics are exact machine arithmetic (uN/iN wrap), so the
 * C must not introduce UB the nodes do not have: +, -, *, <<, ~ and
 * unary - are computed in uint32_t or uint64_t and converted back (to a
 * signed type: modulo 2^N, implementation-defined, as GCC and clang
 * define it); >> of a signed value goes through int32_t/int64_t
 * (arithmetic, implementation-defined likewise); a signed / or % is
 * emitted only when the divisor is a constant other than -1 (or the
 * operands are narrower than int), since INT_MIN / -1 wraps in the
 * nodes and is UB in C.  ?:, && and || keep their laziness, so an
 * operation that is undefined on the untaken side stays untaken.
 *
 * Shared subterms of a whole-function body become `const` temporaries
 * when they are total (no operation in them can be undefined), so
 * hoisting never evaluates a division, a variable shift or a table
 * index that the original did not.
 *
 * popcount, clz, ctz, min, max, abs go through small static inline
 * helpers: the target's instruction under its feature macro
 * (_mm_popcnt_u32 under __POPCNT__, _lzcnt_u32 under __LZCNT__,
 * _tzcnt_u32 under __BMI__), else the GNU builtin, else portable C.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfront.h"

static const char *
cty(ty_t t)
{
	switch (TYK(t)) {
	case K_BOOL:
		return "int";
	case K_UINT:
		return TYW(t) <= 8 ? "uint8_t" : TYW(t) <= 16 ? "uint16_t" :
		    TYW(t) <= 32 ? "uint32_t" : "uint64_t";
	case K_SINT:
		return TYW(t) <= 8 ? "int8_t" : TYW(t) <= 16 ? "int16_t" :
		    TYW(t) <= 32 ? "int32_t" : "int64_t";
	}
	return NULL;
}

static int
std_width(ty_t t)
{
	int w = TYW(t);

	return TYK(t) == K_BOOL || w == 8 || w == 16 || w == 32 || w == 64;
}

static const char *
wide_u(ty_t t)
{
	return TYW(t) <= 32 ? "uint32_t" : "uint64_t";
}

static const char *
wide_s(ty_t t)
{
	return TYW(t) <= 32 ? "int32_t" : "int64_t";
}

struct em {
	struct cf_emit *em;
	const struct cf_value *v;
	const char *const *names;	/* temporaries: per node, or NULL */
	nid *nmap;
	uint32_t nnames;
	int bad;
	const char *why;
};

static void pe(struct em *E, struct sbuf *b, nid n);

static const char *
tmp_name(struct em *E, nid n)
{
	uint32_t k;

	for (k = 0; k < E->nnames; k++)
		if (E->nmap[k] == n)
			return E->names[k];
	return NULL;
}

static void
konst(struct sbuf *b, ty_t t, int64_t v)
{
	v = int_norm(t, v);
	switch (TYK(t)) {
	case K_BOOL:
		sb_printf(b, "%d", (int)(v & 1));
		return;
	case K_UINT:
		if (TYW(t) <= 32)
			sb_printf(b, "((%s)%lluu)", cty(t), (unsigned long long)v);
		else
			sb_printf(b, "((uint64_t)%lluull)", (unsigned long long)v);
		return;
	case K_SINT:
		if (TYW(t) <= 32) {
			if (v == INT32_MIN)
				sb_printf(b, "((%s)(-2147483647-1))", cty(t));
			else
				sb_printf(b, "((%s)%lld)", cty(t), (long long)v);
		} else {
			if (v == INT64_MIN)
				sb_puts(b, "((int64_t)(-9223372036854775807ll-1))");
			else
				sb_printf(b, "((int64_t)%lldll)", (long long)v);
		}
		return;
	}
	sb_printf(b, "%lld", (long long)v);
}

/* helpers, emitted once each */
enum {
	H_POP32, H_POP64, H_CLZ32, H_CLZ64, H_CTZ32, H_CTZ64, H_NH
};

static void
need(struct em *E, int h)
{
	struct sbuf *o = &E->em->helpers;

	if (E->em->used & (1ULL << h))
		return;
	E->em->used |= 1ULL << h;
	switch (h) {
	case H_POP32:
		sb_puts(o, "static inline uint32_t cf_popcount32(uint32_t x)\n{\n"
		    "#if defined(__POPCNT__)\n\treturn (uint32_t)_mm_popcnt_u32(x);\n"
		    "#elif defined(__GNUC__)\n\treturn (uint32_t)__builtin_popcount(x);\n"
		    "#else\n\tx = x - ((x >> 1) & 0x55555555u);\n"
		    "\tx = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);\n"
		    "\treturn (((x + (x >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;\n"
		    "#endif\n}\n");
		break;
	case H_POP64:
		sb_puts(o, "static inline uint64_t cf_popcount64(uint64_t x)\n{\n"
		    "#if defined(__POPCNT__)\n\treturn (uint64_t)_mm_popcnt_u64(x);\n"
		    "#elif defined(__GNUC__)\n\treturn (uint64_t)__builtin_popcountll(x);\n"
		    "#else\n\tx = x - ((x >> 1) & 0x5555555555555555ull);\n"
		    "\tx = (x & 0x3333333333333333ull) + ((x >> 2) & "
		    "0x3333333333333333ull);\n"
		    "\treturn (((x + (x >> 4)) & 0x0f0f0f0f0f0f0f0full) * "
		    "0x0101010101010101ull) >> 56;\n#endif\n}\n");
		break;
	case H_CLZ32:
		sb_puts(o, "static inline uint32_t cf_clz32(uint32_t x)\n{\n"
		    "#if defined(__LZCNT__)\n\treturn (uint32_t)_lzcnt_u32(x);\n"
		    "#elif defined(__GNUC__)\n\treturn x ? (uint32_t)__builtin_clz(x) : 32u;\n"
		    "#else\n\tuint32_t n = 32;\n\twhile (x) { x >>= 1; n--; }\n"
		    "\treturn n;\n#endif\n}\n");
		break;
	case H_CLZ64:
		sb_puts(o, "static inline uint64_t cf_clz64(uint64_t x)\n{\n"
		    "#if defined(__LZCNT__)\n\treturn (uint64_t)_lzcnt_u64(x);\n"
		    "#elif defined(__GNUC__)\n\treturn x ? (uint64_t)__builtin_clzll(x) : 64u;\n"
		    "#else\n\tuint64_t n = 64;\n\twhile (x) { x >>= 1; n--; }\n"
		    "\treturn n;\n#endif\n}\n");
		break;
	case H_CTZ32:
		sb_puts(o, "static inline uint32_t cf_ctz32(uint32_t x)\n{\n"
		    "#if defined(__BMI__)\n\treturn (uint32_t)_tzcnt_u32(x);\n"
		    "#elif defined(__GNUC__)\n\treturn x ? (uint32_t)__builtin_ctz(x) : 32u;\n"
		    "#else\n\tuint32_t n = 0;\n\tif (!x) return 32;\n"
		    "\twhile (!(x & 1)) { x >>= 1; n++; }\n\treturn n;\n#endif\n}\n");
		break;
	case H_CTZ64:
		sb_puts(o, "static inline uint64_t cf_ctz64(uint64_t x)\n{\n"
		    "#if defined(__BMI__)\n\treturn (uint64_t)_tzcnt_u64(x);\n"
		    "#elif defined(__GNUC__)\n\treturn x ? (uint64_t)__builtin_ctzll(x) : 64u;\n"
		    "#else\n\tuint64_t n = 0;\n\tif (!x) return 64;\n"
		    "\twhile (!(x & 1)) { x >>= 1; n++; }\n\treturn n;\n#endif\n}\n");
		break;
	}
}

static void
fail_e(struct em *E, const char *why)
{
	if (!E->bad) {
		E->bad = 1;
		E->why = why;
	}
}

/* (U)(a) zero- or sign-extended, masked to the type's width */
static void
masked_u(struct em *E, struct sbuf *b, nid a)
{
	ty_t t = N.ty[a];
	int w = TYW(t);

	if (TYK(t) == K_SINT && w < 32) {
		sb_printf(b, "((uint32_t)(");
		pe(E, b, a);
		sb_printf(b, ") & 0x%llxu)", (unsigned long long)((1ULL << w) - 1));
	} else {
		sb_printf(b, "(%s)(", wide_u(t));
		pe(E, b, a);
		sb_puts(b, ")");
	}
}

static void
pe(struct em *E, struct sbuf *b, nid n)
{
	int op = N.op[n];
	ty_t t = N.ty[n];
	const char *nm = tmp_name(E, n);
	const char *T = cty(t), *Uw = wide_u(t);
	nid a = N.a[n], c = N.b[n];

	if (nm) {
		sb_puts(b, nm);
		return;
	}
	if (E->bad)
		return;
	if (T == NULL || !std_width(t)) {
		fail_e(E, "a value of a non-standard width");
		return;
	}
	if (b->n > 60000) {
		fail_e(E, "the expression is too long");
		return;
	}
	switch (op) {
	case O_CONST:
		konst(b, t, N.ki[n]);
		return;
	case O_VAR: {
		int s = (int)N.ki[n];
		if (s >= E->v->nleaf) {
			fail_e(E, "a free variable");
			return;
		}
		sb_printf(b, "(%s)", E->v->leaf[s].text);
		return;
	}
	case O_UNDEF:
		sb_printf(b, "((%s)0)", T);
		return;
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_AND:
	case O_OR:
	case O_XOR: {
		const char *o = op == O_ADD ? "+" : op == O_SUB ? "-" : op == O_MUL ?
		    "*" : op == O_AND ? "&" : op == O_OR ? "|" : "^";
		if (t == TY_BOOL) {
			sb_puts(b, "(");
			pe(E, b, a);
			sb_printf(b, " %s ", o);
			pe(E, b, c);
			sb_puts(b, ")");
			return;
		}
		sb_printf(b, "((%s)((%s)(", T, Uw);
		pe(E, b, a);
		sb_printf(b, ") %s (%s)(", o, Uw);
		pe(E, b, c);
		sb_puts(b, ")))");
		return;
	}
	case O_NEG:
		sb_printf(b, "((%s)(0u - (%s)(", T, Uw);
		pe(E, b, a);
		sb_puts(b, ")))");
		return;
	case O_NOT:
		if (t == TY_BOOL) {
			sb_puts(b, "(!(");
			pe(E, b, a);
			sb_puts(b, "))");
			return;
		}
		sb_printf(b, "((%s)~(%s)(", T, Uw);
		pe(E, b, a);
		sb_puts(b, "))");
		return;
	case O_SHL:
		sb_printf(b, "((%s)((%s)(", T, Uw);
		pe(E, b, a);
		sb_puts(b, ") << (");
		pe(E, b, c);
		sb_puts(b, ")))");
		return;
	case O_SHR:
		sb_printf(b, "((%s)((%s)(", T, TYK(t) == K_SINT ? wide_s(t) : Uw);
		pe(E, b, a);
		sb_puts(b, ") >> (");
		pe(E, b, c);
		sb_puts(b, ")))");
		return;
	case O_DIV:
	case O_MOD:
		if (TYK(t) == K_SINT && TYW(t) >= 32 && !(is_const(c) &&
		    N.ki[c] != -1)) {
			fail_e(E, "a signed division by a variable (INT_MIN / -1 "
			    "is UB in C)");
			return;
		}
		sb_printf(b, "((%s)((%s)(", T, TYK(t) == K_SINT ? wide_s(t) : Uw);
		pe(E, b, a);
		sb_printf(b, ") %s (%s)(", op == O_DIV ? "/" : "%",
		    TYK(t) == K_SINT ? wide_s(t) : Uw);
		pe(E, b, c);
		sb_puts(b, ")))");
		return;
	case O_MIN:
	case O_MAX:
		/* both operands are values without side effects */
		sb_printf(b, "((%s)(", T);
		pe(E, b, a);
		sb_printf(b, ") %s (%s)(", op == O_MIN ? "<" : ">", T);
		pe(E, b, c);
		sb_printf(b, ") ? (%s)(", T);
		pe(E, b, a);
		sb_printf(b, ") : (%s)(", T);
		pe(E, b, c);
		sb_puts(b, "))");
		return;
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE: {
		static const char *const cs[] = {"==", "!=", "<", "<=", ">", ">="};
		ty_t ta = N.ty[a];
		const char *Ta = cty(ta);
		if (Ta == NULL) {
			fail_e(E, "a comparison of non-integers");
			return;
		}
		sb_printf(b, "((%s)(", Ta);
		pe(E, b, a);
		sb_printf(b, ") %s (%s)(", cs[op - O_EQ], Ta);
		pe(E, b, c);
		sb_puts(b, "))");
		return;
	}
	case O_LAND:
	case O_LOR:
	case O_LNOT:
		if (op == O_LNOT) {
			sb_puts(b, "(!(");
			pe(E, b, a);
			sb_puts(b, "))");
			return;
		}
		sb_puts(b, "((");
		pe(E, b, a);
		sb_printf(b, ") %s (", op == O_LAND ? "&&" : "||");
		pe(E, b, c);
		sb_puts(b, "))");
		return;
	case O_SEL:
		sb_puts(b, "((");
		pe(E, b, a);
		sb_printf(b, ") ? (%s)(", T);
		pe(E, b, c);
		sb_printf(b, ") : (%s)(", T);
		pe(E, b, N.c[n]);
		sb_puts(b, "))");
		return;
	case O_CAST:
		if (t == TY_BOOL) {
			sb_puts(b, "((int)((");
			pe(E, b, a);
			sb_puts(b, ") & 1))");
			return;
		}
		if (!TY_ISINT(N.ty[a])) {
			fail_e(E, "a conversion from a float");
			return;
		}
		sb_printf(b, "((%s)(", T);
		pe(E, b, a);
		sb_puts(b, "))");
		return;
	case O_ABS:
		if (TYK(t) != K_SINT) {
			pe(E, b, a);
			return;
		}
		sb_printf(b, "((%s)((%s)(", T, wide_s(t));
		pe(E, b, a);
		sb_printf(b, ") < 0 ? 0u - (%s)(", Uw);
		pe(E, b, a);
		sb_printf(b, ") : (%s)(", Uw);
		pe(E, b, a);
		sb_puts(b, ")))");
		return;
	case O_SIGN:
		sb_printf(b, "((%s)(((", T);
		pe(E, b, a);
		sb_puts(b, ") > 0) - ((");
		pe(E, b, a);
		sb_puts(b, ") < 0)))");
		return;
	case O_POPCNT:
	case O_CLZ:
	case O_CTZ: {
		int w = TYW(t), big = w > 32;
		int h = op == O_POPCNT ? (big ? H_POP64 : H_POP32) : op == O_CLZ ?
		    (big ? H_CLZ64 : H_CLZ32) : (big ? H_CTZ64 : H_CTZ32);
		need(E, h);
		sb_printf(b, "((%s)(", T);
		if (op == O_CTZ && w < 32) {
			/* ctz(0) is the width */
			sb_puts(b, "(");
			masked_u(E, b, a);
			sb_printf(b, ") ? cf_ctz32(");
			masked_u(E, b, a);
			sb_printf(b, ") : %du))", w);
			return;
		}
		sb_printf(b, "%s(", op == O_POPCNT ? (big ? "cf_popcount64" :
		    "cf_popcount32") : op == O_CLZ ? (big ? "cf_clz64" :
		    "cf_clz32") : (big ? "cf_ctz64" : "cf_ctz32"));
		masked_u(E, b, a);
		sb_puts(b, ")");
		if (op == O_CLZ && w < 32)
			sb_printf(b, " - %du", 32 - w);
		sb_puts(b, "))");
		return;
	}
	case O_INDEX: {
		nid tab = a;
		uint32_t cnt = N.b[tab], k;
		ty_t et = t;
		int id = E->em->ntab++;
		sb_printf(&E->em->helpers, "static const %s cf_tab%d[%u] = {",
		    cty(et), id, cnt);
		for (k = 0; k < cnt; k++) {
			nid e = N.lst[N.a[tab] + k];
			if (!is_const(e)) {
				fail_e(E, "a table of non-constants");
				return;
			}
			if (k % 12 == 0)
				sb_puts(&E->em->helpers, "\n\t");
			if (TYK(et) == K_UINT && TYW(et) == 64)
				sb_printf(&E->em->helpers, "%lluull, ",
				    (unsigned long long)N.ki[e]);
			else if (TYK(et) == K_UINT)
				sb_printf(&E->em->helpers, "%lluu, ",
				    (unsigned long long)int_norm(et, N.ki[e]));
			else if (TYW(et) == 64 && N.ki[e] == INT64_MIN)
				sb_puts(&E->em->helpers, "(-9223372036854775807ll-1), ");
			else
				sb_printf(&E->em->helpers, "%lld, ",
				    (long long)int_norm(et, N.ki[e]));
		}
		sb_puts(&E->em->helpers, "\n};\n");
		sb_printf(b, "cf_tab%d[", id);
		pe(E, b, c);
		sb_puts(b, "]");
		return;
	}
	}
	fail_e(E, opinfo[op].name);
}

int
cf_emit_expr(struct cf_emit *em, const struct cf_value *v, nid e,
    uint32_t ct, struct sbuf *out)
{
	struct em E;
	struct sbuf b = {0};
	struct sbuf decl = {0};

	memset(&E, 0, sizeof E);
	E.em = em;
	E.v = v;
	pe(&E, &b, e);
	if (E.bad) {
		free(b.p);
		return 0;
	}
	ct_str(&decl, ct_unqual(ct), "");
	sb_printf(out, "(%s)%s", decl.p, b.p);
	free(b.p);
	free(decl.p);
	return 1;
}

/* use counts from the root, for temporaries */
static void
count_uses(nid n, uint32_t *cnt, uint8_t *seen)
{
	if (n == 0)
		return;
	cnt[n]++;
	if (seen[n])
		return;
	seen[n] = 1;
	if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_TABLE)
		return;
	if (N.op[n] == O_INDEX) {
		count_uses(N.b[n], cnt, seen);
		return;
	}
	count_uses(N.a[n], cnt, seen);
	count_uses(N.b[n], cnt, seen);
	if (N.op[n] == O_SEL || N.op[n] == O_FMA)
		count_uses(N.c[n], cnt, seen);
}

int
cf_emit_body(struct cf_emit *em, const struct cf_value *v, nid e,
    uint32_t ct, struct sbuf *out)
{
	uint32_t *cnt = xcalloc(N.n, sizeof *cnt);
	uint8_t *seen = xcalloc(N.n, 1);
	struct em E;
	struct sbuf b = {0}, decl = {0};
	nid k;
	uint32_t nt = 0, cap = 0;
	char **names = NULL;
	int ok = 1;

	memset(&E, 0, sizeof E);
	E.em = em;
	E.v = v;
	count_uses(e, cnt, seen);
	sb_puts(out, "{\n");
	for (k = 1; k < N.n && ok; k++) {
		int op = N.op[k];
		if (!seen[k] || cnt[k] < 2 || k == e || op == O_CONST ||
		    op == O_VAR || op == O_TABLE || (N.fl[k] & NF_PARTIAL) ||
		    !cty(N.ty[k]) || !std_width(N.ty[k]))
			continue;
		sb_clear(&b);
		E.names = (const char *const *)names;
		E.nnames = nt;
		pe(&E, &b, k);
		if (E.bad) {
			ok = 0;
			break;
		}
		if (nt == cap) {
			cap = cap ? cap * 2 : 16;
			names = xrealloc(names, cap * sizeof *names);
			E.nmap = xrealloc(E.nmap, cap * sizeof *E.nmap);
		}
		names[nt] = xmalloc(16);
		snprintf(names[nt], 16, "cf_t%u", nt);
		E.nmap[nt] = k;
		sb_printf(out, "\tconst %s %s = %s;\n", cty(N.ty[k]), names[nt],
		    b.p);
		nt++;
	}
	if (ok) {
		sb_clear(&b);
		E.names = (const char *const *)names;
		E.nnames = nt;
		pe(&E, &b, e);
		if (E.bad)
			ok = 0;
	}
	if (ok) {
		ct_str(&decl, ct_unqual(ct), "");
		sb_printf(out, "\treturn (%s)%s;\n}", decl.p, b.p);
	}
	while (nt > 0)
		free(names[--nt]);
	free(names);
	free(E.nmap);
	free(b.p);
	free(decl.p);
	free(cnt);
	free(seen);
	return ok;
}
