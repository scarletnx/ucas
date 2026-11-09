/*
 * Parser and statement driver.
 *
 * Expressions use a Pratt parser over the token arrays.  Where the math
 * reading and the programmer reading of a construct disagree, the parser
 * picks one by a fixed rule and says so with a note:
 *
 *   2x, 3(x+1), 2 sin x      juxtaposition multiplies, binding tighter than
 *                            * and /, so 1/2x is 1/(2x)
 *   sin x^2, sin 2x          a function name without parentheses takes the
 *                            following juxtaposition product as argument
 *   x(y + 1)                 multiplication when x is not a function
 *   xy                       x*y when xy is unknown and x, y are known
 *   |x|                      absolute value in operand position; | is
 *                            bitwise or in operator position
 *   a < b < c                (a < b) && (b < c)
 *   x = 0 inside an expr     comparison ==
 *   sqrt(n) on an integer    floor square root
 *   ^                        power; xor is "xor" or ⊕
 *   1/2                      exact: two integer literals that do not divide
 *                            give a real
 */
#include "ucas.h"
#include "props.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct func *funcs;
int nfuncs;
static int capfuncs;

struct func *
func_find(uint32_t name)
{
	int i;

	for (i = 0; i < nfuncs; i++)
		if (funcs[i].name == name)
			return &funcs[i];
	return NULL;
}

struct func *
func_new(uint32_t name)
{
	struct func *f = func_find(name);

	if (f == NULL) {
		if (nfuncs == capfuncs) {
			capfuncs = capfuncs ? capfuncs * 2 : 32;
			funcs = xrealloc(funcs, (size_t)capfuncs * sizeof *funcs);
		}
		f = &funcs[nfuncs++];
	}
	memset(f, 0, sizeof *f);
	f->name = name;
	return f;
}

/* ---- parser state ---- */

#define MAXBIND 32
struct P {
	struct toks *t;
	uint32_t i;
	struct func *fn;
	int autop;
	ty_t autoty;
	int abs_depth;
	int fnarg;
	int top_jux;
	int top_paren;    /* the last operand was parenthesized */
	int cmdargs;      /* stop at command option words */
	const char *const *xstop;  /* approx: more option words; 'in' stops */
	ty_t force_ty;    /* animate exec of 32-bit ...: free variables' type */
	int nb;
	uint32_t bname[MAXBIND];
	nid bval[MAXBIND];
};

enum {
	BP_TERN = 1, BP_OR, BP_AND, BP_CMP, BP_BOR, BP_XOR, BP_BAND,
	BP_SHIFT, BP_ADD, BP_MUL, BP_UNARY, BP_JUX, BP_POW, BP_POST
};

static nid parse_expr(struct P *p, int minbp);

/* calc: loops, sums, recurrences, ODEs (calc_syntax.inc, included below) */
static nid calc_ident(struct P *p);
static int cs_is_block(struct P *p);
static nid calc_block(struct P *p);
static int calc_statement(struct P *p);
static int calc_command(struct P *p);
static int calc2_command(struct P *p);	/* calc2_syntax.inc */

static int
pk(struct P *p)
{
	return p->t->kind[p->i];
}

static int
pk2(struct P *p)
{
	return p->i + 1 < p->t->n ? p->t->kind[p->i + 1] : T_EOF;
}

static void
adv(struct P *p)
{
	if (pk(p) != T_EOF)
		p->i++;
	diag_tok = p->i;
}

static int
accept(struct P *p, int k)
{
	if (pk(p) == k) {
		adv(p);
		return 1;
	}
	return 0;
}

static int
is_kw(struct P *p, const char *w)
{
	return tok_is(p->t, p->i, w);
}

static int
accept_kw(struct P *p, const char *w)
{
	if (is_kw(p, w)) {
		adv(p);
		return 1;
	}
	return 0;
}

static void
expect(struct P *p, int k)
{
	diag_tok = p->i;
	if (!accept(p, k))
		fail("expected '%s', found '%.*s'", tok_name(k),
		    pk(p) == T_EOF || pk(p) == T_NL ? 3 : (int)p->t->len[p->i],
		    pk(p) == T_EOF ? "end" : pk(p) == T_NL ? "eol" :
		    p->t->src + p->t->off[p->i]);
}

static void
skip_nl(struct P *p)
{
	while (pk(p) == T_NL)
		adv(p);
}

/* inside an expression: continue onto the next line only if it is indented */
static void
skip_cont(struct P *p)
{
	while (pk(p) == T_NL && p->i + 1 < p->t->n &&
	    p->t->kind[p->i + 1] != T_EOF) {
		uint32_t off = p->t->off[p->i + 1];
		if (off == 0 || p->t->src[off - 1] == '\n' ||
		    p->t->kind[p->i] != T_NL || p->t->src[p->t->off[p->i]] == ';')
			break;
		adv(p);
	}
}

static const char *
ttext(struct P *p, uint32_t i, size_t *n)
{
	*n = p->t->len[i];
	return p->t->src + p->t->off[i];
}

static uint32_t
tsym(struct P *p, uint32_t i)
{
	size_t n;
	const char *s = ttext(p, i, &n);

	if (n == 2 && (unsigned char)s[0] == 0xcf && (unsigned char)s[1] == 0x80)
		return sym_cstr("pi");
	return sym_intern(s, n);
}

static const char *const keywords[] = {
	"and", "or", "not", "xor", "mod", "in", "if", "else", "otherwise",
	"where", "then", "match", "switch", "true", "false", "undefined",
	"uniform", NULL
};
static const char *const stopwords[] = {
	"basis", "max", "ops", "model", "as", "using", "to", "metric",
	"nodes", "from", "wrt", "steps", "nocrit", NULL
};

static int
word_in(struct P *p, uint32_t i, const char *const *list)
{
	int k;

	for (k = 0; list[k]; k++)
		if (tok_is(p->t, i, list[k]))
			return 1;
	return 0;
}

/* ---- numbers ---- */

static nid
number(struct P *p)
{
	size_t n, i, j = 0;
	const char *s = ttext(p, p->i, &n);
	char buf[80];
	int isreal = 0, f32 = 0;
	uint64_t v;
	nid r;

	if (n >= sizeof buf)
		fail("number too long");
	for (i = 0; i < n; i++)
		if (s[i] != '_')
			buf[j++] = s[i];
	buf[j] = 0;
	if (j > 1 && (buf[j - 1] == 'f' || buf[j - 1] == 'F') &&
	    !(buf[0] == '0' && (buf[1] == 'x' || buf[1] == 'X'))) {
		f32 = 1;
		buf[--j] = 0;
	}
	if (buf[0] == '0' && (buf[1] == 'x' || buf[1] == 'X'))
		v = strtoull(buf + 2, NULL, 16);
	else if (buf[0] == '0' && (buf[1] == 'b' || buf[1] == 'B'))
		v = strtoull(buf + 2, NULL, 2);
	else {
		for (i = 0; i < j; i++)
			if (buf[i] == '.' || buf[i] == 'e' || buf[i] == 'E')
				isreal = 1;
		if (isreal) {
			double d = strtod(buf, NULL);
			adv(p);
			if (f32)
				return mk_cast(TY_F32, mk_const_r(d));
			return mk_const_r(d);
		}
		v = strtoull(buf, NULL, 10);
	}
	adv(p);
	r = mk_const_i(v > (uint64_t)INT64_MAX ? TY(K_UINT, 64) : TY_LIT,
	    (int64_t)v);
	return r;
}

/* ---- names ---- */

struct bfn {
	const char *name;
	int op;
	int arity;
};
static const struct bfn bfns[] = {
	{"sin", O_SIN, 1}, {"cos", O_COS, 1}, {"tan", O_TAN, 1},
	{"exp", O_EXP, 1}, {"log", O_LOG, 1}, {"ln", O_LOG, 1},
	{"sqrt", O_SQRT, 1}, {"isqrt", O_ISQRT, 1}, {"abs", O_ABS, 1},
	{"floor", O_FLOOR, 1}, {"ceil", O_CEIL, 1}, {"sign", O_SIGN, 1},
	{"sgn", O_SIGN, 1}, {"clz", O_CLZ, 1}, {"ctz", O_CTZ, 1},
	{"popcount", O_POPCNT, 1}, {"popcnt", O_POPCNT, 1},
	{"min", O_MIN, 2}, {"max", O_MAX, 2},
	{"asuint", O_ASUINT, 1}, {"bits", O_ASUINT, 1},
	{"asint", O_ASINT, 1}, {"asfloat", O_ASFLOAT, 1},
	{"barrier", O_BARRIER, 1},
	{"sec", O_SEC, 1}, {"csc", O_CSC, 1}, {"cot", O_COT, 1},
	{"atan", O_ATAN, 1}, {"copysign", O_COPYSIGN, 2}, {"fma", O_FMA, 3},
	/* Lambert W; a bare W stays a variable unless called: W(x) */
	{"W", O_LAMBERTW, 1}, {"W0", O_LAMBERTW, 1}, {"lambertw", O_LAMBERTW, 1},
	{"productlog", O_LAMBERTW, 1}, {"Wm1", O_LAMBERTWM1, 1},
	{"lambertwm1", O_LAMBERTWM1, 1},
	/* forms.c */
	{"asin", O_ASIN, 1}, {"arcsin", O_ASIN, 1}, {"acos", O_ACOS, 1},
	{"arccos", O_ACOS, 1}, {"arctan", O_ATAN, 1}, {"atan2", O_ATAN2, 2},
	{"sinh", O_SINH, 1}, {"cosh", O_COSH, 1}, {"tanh", O_TANH, 1},
	{"asinh", O_ASINH, 1}, {"arsinh", O_ASINH, 1}, {"acosh", O_ACOSH, 1},
	{"arcosh", O_ACOSH, 1}, {"atanh", O_ATANH, 1}, {"artanh", O_ATANH, 1},
	{NULL, 0, 0}
};

static const struct bfn *
builtin(struct P *p, uint32_t i)
{
	int k;

	for (k = 0; bfns[k].name; k++)
		if (tok_is(p->t, i, bfns[k].name)) {
			/* one letter (W): only called, and a user's W wins */
			if (bfns[k].name[1] == 0 && (i + 1 >= p->t->n ||
			    p->t->kind[i + 1] != T_LP || func_find(tsym(p, i))))
				return NULL;
			return &bfns[k];
		}
	return NULL;
}

static int
param_slot(struct P *p, uint32_t name)
{
	int j;

	if (p->fn == NULL)
		return -1;
	for (j = 0; j < p->fn->np; j++)
		if (p->fn->p[j].name == name)
			return j;
	return -1;
}

static nid
param_var(struct P *p, int j)
{
	return mk_var((uint32_t)j, p->fn->p[j].name, p->fn->p[j].ty);
}

static int
binding(struct P *p, uint32_t name)
{
	int j;

	for (j = p->nb; j-- > 0;)
		if (p->bname[j] == name)
			return j;
	return -1;
}

/* a name that is a value: binding, parameter or constant */
static nid
value_name(struct P *p, const char *s, size_t n)
{
	uint32_t name = sym_intern(s, n);
	int j;

	if ((j = binding(p, name)) >= 0)
		return p->bval[j];
	if ((j = param_slot(p, name)) >= 0)
		return param_var(p, j);
	if ((n == 2 && !memcmp(s, "pi", 2)) ||
	    (n == 2 && (unsigned char)s[0] == 0xcf))
		return mk_const_r(M_PI);
	if (n == 1 && s[0] == 'e')
		return mk_const_r(M_E);
	if (n == 2 && (unsigned char)s[0] == 0xc2)   /* 45° */
		return mk_const_r(M_PI / 180);
	if (n == 3 && !memcmp(s, "inf", 3))
		return mk_const_r(INFINITY);
	return 0;
}

static nid
new_auto_param(struct P *p, uint32_t name)
{
	struct param *pp;
	ty_t t;

	if (p->fn->np >= MAXP)
		fail("too many parameters");
	pp = &p->fn->p[p->fn->np];
	memset(pp, 0, sizeof *pp);
	pp->name = name;
	pp->ty = p->autoty;
	/* inline annotation: x: u4 */
	if (pk(p) == T_COLON && pk2(p) == T_ID) {
		size_t n;
		const char *s = ttext(p, p->i + 1, &n);
		if (ty_parse(s, n, &t)) {
			pp->ty = t;
			adv(p);
			adv(p);
		}
	}
	p->fn->np++;
	return param_var(p, p->fn->np - 1);
}

static nid call_user(struct P *p, struct func *g, nid *args, int nargs);

static int
next_is_fn(struct P *p)
{
	uint32_t i = p->i;

	if (p->t->kind[i] != T_ID)
		return 0;
	if (builtin(p, i))
		return 1;
	{
		struct func *g = func_find(tsym(p, i));
		return g && g->np >= 1 && param_slot(p, tsym(p, i)) < 0;
	}
}

static int
starts_operand(struct P *p)
{
	switch (pk(p)) {
	case T_NUM:
	case T_LP:
	case T_SQRT:
		return 1;
	case T_ID:
		if (word_in(p, p->i, keywords) && !is_kw(p, "match") &&
		    !is_kw(p, "switch") && !is_kw(p, "not") &&
		    !is_kw(p, "true") && !is_kw(p, "false"))
			return 0;
		if (p->cmdargs && word_in(p, p->i, stopwords))
			return 0;
		if (p->xstop && word_in(p, p->i, p->xstop))
			return 0;
		return 1;
	}
	return 0;
}

static int
parse_args(struct P *p, nid *args, int max)
{
	int n = 0, saved = p->abs_depth, savedf = p->fnarg;

	p->abs_depth = 0;
	p->fnarg = 0;
	expect(p, T_LP);
	if (!accept(p, T_RP)) {
		do {
			if (n == max)
				fail("too many arguments");
			args[n++] = parse_expr(p, 0);
		} while (accept(p, T_COMMA));
		expect(p, T_RP);
	}
	p->abs_depth = saved;
	p->fnarg = savedf;
	return n;
}

static nid
apply_builtin(struct P *p, const struct bfn *b, nid *args, int n)
{
	nid a;

	if (n != b->arity)
		fail("%s takes %d argument%s", b->name, b->arity,
		    b->arity == 1 ? "" : "s");
	a = args[0];
	if (b->arity == 3)
		return mk_fma(args[0], args[1], args[2]);
	if (b->arity == 2)
		return mk_bin(b->op, args[0], args[1]);
	if (b->op == O_BARRIER)
		return mk_barrier(a);
	if (b->op == O_SQRT && TY_ISINT(N.ty[a]) &&
	    !(is_const(a) && TYK(N.ty[a]) == K_LIT)) {
		note("sqrt of the integer %s read as floor square root "
		    "isqrt(); write sqrt(real(%s)) for the real one",
		    node_str(a), node_str(a));
		return mk_un(O_ISQRT, a);
	}
	(void)p;
	return mk_un(b->op, a);
}

/* identifier in operand position */
static nid
ident(struct P *p)
{
	uint32_t at = p->i, name = tsym(p, at);
	size_t n;
	const char *s = ttext(p, at, &n);
	const struct bfn *b;
	struct func *g;
	nid args[MAXP], r;
	ty_t ct;
	int na, kind;

	if ((r = value_name(p, s, n)) != 0) {
		adv(p);
		return r;
	}
	if ((r = calc_ident(p)) != 0)
		return r;
	/* type conversion: u8(x), f32(x), real(n) */
	if (pk2(p) == T_LP && ty_parse(s, n, &ct)) {
		adv(p);
		na = parse_args(p, args, 1);
		if (na != 1)
			fail("a conversion takes one argument");
		return mk_cast(ct, args[0]);
	}
	if (atomic_lookup(s, n, &kind)) {
		adv(p);
		if (parse_args(p, args, 2) != 2)
			fail("atomics take (value, operand)");
		return mk_atomic(kind, args[0], args[1]);
	}
	if ((b = builtin(p, at)) != NULL) {
		adv(p);
		if (pk(p) == T_LP) {
			na = parse_args(p, args, 3);
			return apply_builtin(p, b, args, na);
		}
		if (b->arity != 1)
			fail("%s needs parentheses: %s(a, b)", b->name, b->name);
		p->fnarg++;
		args[0] = parse_expr(p, BP_JUX);
		p->fnarg--;
		return apply_builtin(p, b, args, 1);
	}
	if ((g = func_find(name)) != NULL && (p->fn == NULL || g != p->fn)) {
		adv(p);
		if (g->np == 0)
			return g->body;
		if (pk(p) == T_LP) {
			na = parse_args(p, args, MAXP);
			return call_user(p, g, args, na);
		}
		if (g->np != 1)
			fail("%s takes %d arguments", sym_str(name), g->np);
		p->fnarg++;
		args[0] = parse_expr(p, BP_JUX);
		p->fnarg--;
		return call_user(p, g, args, 1);
	}
	/* xy -> x*y when every piece is a known value */
	if (n > 1) {
		size_t i = 0, l;
		nid prod = 0, v = 0;
		int parts = 0;
		while (i < n) {
			for (l = n - i; l > 0; l--)
				if ((v = value_name(p, s + i, l)) != 0)
					break;
			if (l == 0)
				break;
			prod = prod ? mk_bin(O_MUL, prod, v) : v;
			parts++;
			i += l;
		}
		if (i == n && parts > 1) {
			note("'%.*s' is not defined: read as a product %s",
			    (int)n, s, node_str(prod));
			adv(p);
			return prod;
		}
		/* anonymous expressions: short letter runs are products */
		if (p->autop && n == 2 && isalpha((unsigned char)s[0]) &&
		    isalpha((unsigned char)s[1]) && pk2(p) != T_COLON) {
			nid x, y;
			adv(p);
			x = value_name(p, s, 1);
			if (!x)
				x = new_auto_param(p, sym_intern(s, 1));
			y = value_name(p, s + 1, 1);
			if (!y)
				y = new_auto_param(p, sym_intern(s + 1, 1));
			note("'%.*s' read as %c*%c (single-letter variables)",
			    (int)n, s, s[0], s[1]);
			return mk_bin(O_MUL, x, y);
		}
	}
	if (p->autop && p->fn) {
		adv(p);
		return new_auto_param(p, name);
	}
	fail("'%.*s' is not defined", (int)n, s);
	return 0;
}

static nid
call_user(struct P *p, struct func *g, nid *args, int na)
{
	nid map[MAXP];
	int j;

	if (na != g->np)
		fail("%s takes %d argument%s, got %d", sym_str(g->name), g->np,
		    g->np == 1 ? "" : "s", na);
	for (j = 0; j < na; j++) {
		ty_t want = g->p[j].ty, have = N.ty[args[j]];
		if (have != want && TYK(have) != K_LIT && !is_const(args[j]))
			note("argument %d of %s converted from %s to %s", j + 1,
			    sym_str(g->name), ty_str(have), ty_str(want));
		map[j] = mk_cast(want, args[j]);
	}
	(void)p;
	return subst(g->body, map, na);
}

/* ---- match / piecewise ---- */

static int64_t
pat_num(struct P *p)
{
	int neg = accept(p, T_MINUS);
	nid v;

	if (pk(p) != T_NUM)
		fail("expected a number in the pattern");
	v = number(p);
	if (!is_const(v) || !TY_ISINT(N.ty[v]))
		fail("patterns are integers");
	return neg ? -N.ki[v] : N.ki[v];
}

static nid
parse_match(struct P *p)
{
	nid scrut, conds[256], vals[256], dflt = 0, c, acc;
	int n = 0, k;

	adv(p);
	scrut = parse_expr(p, BP_CMP + 1);
	skip_nl(p);
	expect(p, T_LC);
	for (;;) {
		while (accept(p, T_NL) || accept(p, T_COMMA))
			;
		if (accept(p, T_RC))
			break;
		if (n == 256)
			fail("too many match arms");
		c = 0;
		if (accept_kw(p, "_") || accept_kw(p, "default") ||
		    accept_kw(p, "else") || accept_kw(p, "otherwise")) {
			c = 0;
		} else {
			do {
				int64_t lo = pat_num(p), hi = lo;
				nid one;
				if (accept(p, T_DOTDOT))
					hi = pat_num(p);
				if (lo == hi)
					one = mk_bin(O_EQ, scrut, mk_const_i(TY_LIT, lo));
				else
					one = mk_bin(O_LAND,
					    mk_bin(O_GE, scrut, mk_const_i(TY_LIT, lo)),
					    mk_bin(O_LE, scrut, mk_const_i(TY_LIT, hi)));
				c = c ? mk_bin(O_LOR, c, one) : one;
			} while (accept(p, T_BAR) || accept(p, T_COMMA));
		}
		if (!accept(p, T_ARROW) && !accept(p, T_COLON))
			fail("expected '=>' or ':' after the pattern");
		if (c == 0)
			dflt = parse_expr(p, 0);
		else {
			conds[n] = c;
			vals[n++] = parse_expr(p, 0);
		}
	}
	acc = dflt ? dflt : mk_undef(TY_LIT);
	if (!dflt)
		note("match has no default arm: other values are outside the "
		    "domain");
	for (k = n; k-- > 0;)
		acc = mk_sel(conds[k], vals[k], acc);
	return acc;
}

static nid
parse_piecewise(struct P *p)
{
	nid conds[256], vals[256], dflt = 0, acc, e;
	int n = 0, k;

	adv(p);
	for (;;) {
		while (accept(p, T_NL) || accept(p, T_COMMA))
			;
		if (accept(p, T_RC))
			break;
		if (n == 256)
			fail("too many pieces");
		e = parse_expr(p, 0);
		if (accept_kw(p, "if") || accept(p, T_COLON)) {
			conds[n] = parse_expr(p, 0);
			vals[n++] = e;
		} else {
			accept_kw(p, "otherwise");
			accept_kw(p, "else");
			dflt = e;
		}
	}
	acc = dflt ? dflt : mk_undef(TY_LIT);
	for (k = n; k-- > 0;)
		acc = mk_sel(conds[k], vals[k], acc);
	return acc;
}

/* ---- expressions ---- */

static nid
prefix(struct P *p)
{
	nid e, items[1024];
	int n, saved;

	diag_tok = p->i;
	p->top_jux = 0;
	p->top_paren = 0;
	switch (pk(p)) {
	case T_NUM:
		return number(p);
	case T_ID:
		if (is_kw(p, "match") || is_kw(p, "switch"))
			return parse_match(p);
		if (accept_kw(p, "not"))
			return mk_un(O_LNOT, parse_expr(p, BP_UNARY));
		if (accept_kw(p, "true"))
			return mk_const_i(TY_BOOL, 1);
		if (accept_kw(p, "false"))
			return mk_const_i(TY_BOOL, 0);
		if (accept_kw(p, "undefined"))
			return mk_undef(TY_LIT);
		return ident(p);
	case T_LP:
		saved = p->abs_depth;
		p->abs_depth = 0;
		n = p->fnarg;
		p->fnarg = 0;
		adv(p);
		skip_cont(p);
		e = parse_expr(p, 0);
		skip_cont(p);
		expect(p, T_RP);
		p->abs_depth = saved;
		p->fnarg = n;
		p->top_jux = 0;
		p->top_paren = 1;
		return e;
	case T_MINUS:
		adv(p);
		return mk_un(O_NEG, parse_expr(p, BP_UNARY));
	case T_PLUS:
		adv(p);
		return parse_expr(p, BP_UNARY);
	case T_TILDE:
		adv(p);
		return mk_un(O_NOT, parse_expr(p, BP_UNARY));
	case T_BANG:
		adv(p);
		return mk_un(O_LNOT, parse_expr(p, BP_UNARY));
	case T_SQRT:
		adv(p);
		return apply_builtin(p, &bfns[6], (e = parse_expr(p, BP_POW),
		    &e), 1);
	case T_BAR:
		adv(p);
		p->abs_depth++;
		n = p->fnarg;
		p->fnarg = 0;
		e = parse_expr(p, 0);
		p->fnarg = n;
		p->abs_depth--;
		if (pk(p) != T_BAR)
			fail("missing closing '|' of an absolute value");
		adv(p);
		p->top_jux = 0;
		return mk_un(O_ABS, e);
	case T_OROR:
		fail("'||' cannot start an operand; write '| |x| |' for nested "
		    "absolute values");
		break;
	case T_LB:
		saved = p->abs_depth;
		p->abs_depth = 0;
		adv(p);
		n = 0;
		skip_nl(p);
		if (pk(p) != T_RB) {
			do {
				skip_nl(p);
				if (n == 1024)
					fail("table too long");
				items[n++] = parse_expr(p, 0);
				skip_nl(p);
			} while (accept(p, T_COMMA));
		}
		expect(p, T_RB);
		p->abs_depth = saved;
		if (n == 0)
			fail("empty table");
		return mk_table(items, (uint32_t)n);
	case T_LC:
		if (cs_is_block(p))
			return calc_block(p);
		return parse_piecewise(p);
	}
	if (pk(p) == T_EOF || pk(p) == T_NL)
		fail("expected an expression");
	fail("unexpected '%s'", tok_name(pk(p)));
	return 0;
}

static int
cmp_op(struct P *p, int *op)
{
	switch (pk(p)) {
	case T_EQ: *op = O_EQ; return 1;
	case T_ASSIGN: *op = O_EQ; return 2;
	case T_NE: *op = O_NE; return 1;
	case T_LT: *op = O_LT; return 1;
	case T_LE: *op = O_LE; return 1;
	case T_GT: *op = O_GT; return 1;
	case T_GE: *op = O_GE; return 1;
	}
	return 0;
}

static int
infix(struct P *p, int *op)
{
	int c;

	if ((c = cmp_op(p, op)) != 0)
		return BP_CMP;
	switch (pk(p)) {
	case T_QUES: return BP_TERN;
	case T_OROR: *op = O_LOR; return BP_OR;
	case T_ANDAND: *op = O_LAND; return BP_AND;
	case T_BAR:
		if (p->abs_depth > 0)
			return 0;
		*op = O_OR;
		return BP_BOR;
	case T_XOR: *op = O_XOR; return BP_XOR;
	case T_AMP: *op = O_AND; return BP_BAND;
	case T_SHL: *op = O_SHL; return BP_SHIFT;
	case T_SHR: *op = O_SHR; return BP_SHIFT;
	case T_PLUS: *op = O_ADD; return BP_ADD;
	case T_MINUS: *op = O_SUB; return BP_ADD;
	case T_STAR: *op = O_MUL; return BP_MUL;
	case T_SLASH: *op = O_DIV; return BP_MUL;
	case T_PCT: *op = O_MOD; return BP_MUL;
	case T_CARET: *op = O_POW; return BP_POW;
	case T_ID:
		if (is_kw(p, "or")) { *op = O_LOR; return BP_OR; }
		if (is_kw(p, "and")) { *op = O_LAND; return BP_AND; }
		if (is_kw(p, "xor")) { *op = O_XOR; return BP_XOR; }
		if (is_kw(p, "mod")) { *op = O_MOD; return BP_MUL; }
		if (is_kw(p, "in") && !p->xstop) { *op = O_NONE; return BP_CMP; }
		return 0;
	}
	return 0;
}

/* & | << >> ~ xor in the statement around token i */
static int
bitwise_context(struct P *p, uint32_t i)
{
	uint32_t a = i, b = i;

	while (a > 0 && p->t->kind[a - 1] != T_NL)
		a--;
	while (b < p->t->n && p->t->kind[b] != T_NL && p->t->kind[b] != T_EOF)
		b++;
	for (; a < b; a++) {
		int k = p->t->kind[a];
		if (k == T_AMP || k == T_SHL || k == T_SHR || k == T_TILDE ||
		    k == T_XOR || tok_is(p->t, a, "xor") ||
		    (k == T_BAR && a > 0 && p->t->kind[a - 1] != T_LP))
			return 1;
	}
	return 0;
}

static nid
parse_expr(struct P *p, int minbp)
{
	nid lhs = prefix(p), rhs, c, x;
	int lastjux = p->top_jux, op, bp, chained = 0, prevcmp = 0;
	int lparen = p->top_paren, rparen;
	nid prevr = 0;

	for (;;) {
		diag_tok = p->i;
		if (pk(p) == T_LB && BP_POST >= minbp) {
			int saved = p->abs_depth;
			p->abs_depth = 0;
			adv(p);
			rhs = parse_expr(p, 0);
			expect(p, T_RB);
			p->abs_depth = saved;
			lhs = mk_index(lhs, rhs);
			lastjux = 0;
			lparen = 0;
			continue;
		}
		if (pk(p) == T_SUP && BP_POW >= minbp) {
			size_t n;
			const char *s = ttext(p, p->i, &n);
			int e = (unsigned char)s[1] == 0xb2 ? 2 : 3;
			adv(p);
			lhs = mk_bin(O_POW, lhs, mk_const_i(TY_LIT, e));
			lastjux = 0;
			lparen = 0;
			continue;
		}
		bp = infix(p, &op);
		if (bp == 0) {
			/* juxtaposition: implicit multiplication */
			if (starts_operand(p) && BP_JUX >= minbp &&
			    !(p->fnarg && next_is_fn(p))) {
				int paren = pk(p) == T_LP &&
				    !(p->t->fl[p->i] & TF_SPACE) &&
				    p->t->kind[p->i - 1] == T_ID;
				uint32_t at = p->i;
				rhs = parse_expr(p, BP_JUX + 1);
				if (paren && N.op[lhs] == O_VAR) {
					diag_tok = at;
					note("%s(...) read as multiplication: %s is "
					    "not a function", node_str(lhs),
					    node_str(lhs));
				}
				lhs = mk_bin(O_MUL, lhs, rhs);
				lastjux = 1;
				lparen = 0;
				continue;
			}
			break;
		}
		if (bp < minbp)
			break;
		if (bp == BP_CMP && prevcmp && minbp <= BP_CMP) {
			/* a < b < c */
			chained = 1;
		}
		adv(p);
		if (!(bp == BP_CMP && op != O_NONE))
			lparen = 0;
		if (bp == BP_TERN) {
			nid a, b;
			skip_cont(p);
			a = parse_expr(p, 0);
			skip_cont(p);
			expect(p, T_COLON);
			skip_cont(p);
			b = parse_expr(p, BP_TERN);
			lhs = mk_sel(lhs, a, b);
			lastjux = 0;
			continue;
		}
		skip_cont(p);
		if (bp == BP_CMP) {
			if (op == O_NONE) {   /* x in a..b */
				nid lo = parse_expr(p, BP_CMP + 1), hi;
				expect(p, T_DOTDOT);
				hi = parse_expr(p, BP_CMP + 1);
				c = mk_bin(O_LAND, mk_bin(O_GE, lhs, lo),
				    mk_bin(O_LE, lhs, hi));
				lhs = c;
				prevcmp = 0;
				lastjux = 0;
				continue;
			}
			if (p->t->kind[p->i - 1] == T_ASSIGN)
				note("'=' inside an expression read as '=='");
			rhs = parse_expr(p, BP_CMP + 1);
			rparen = p->top_paren;
			if ((!lparen && (N.op[lhs] == O_AND || N.op[lhs] == O_OR ||
			    N.op[lhs] == O_XOR)) || (!rparen && (N.op[rhs] == O_AND ||
			    N.op[rhs] == O_OR || N.op[rhs] == O_XOR)))
			{
				struct sbuf pb = {0};
				print_paren(&pb, mk_bin(op, lhs, rhs));
				note("bitwise operators bind tighter than comparisons "
				    "here (unlike C): read as %s", sb_str(&pb));
				free(pb.p);
			}
			c = mk_bin(op, chained ? prevr : lhs, rhs);
			if (chained) {
				note("chained comparison read as a conjunction");
				lhs = mk_bin(O_LAND, lhs, c);
			} else
				lhs = c;
			prevr = rhs;
			prevcmp = 1;
			lparen = rparen;
			chained = 0;
			lastjux = 0;
			continue;
		}
		prevcmp = 0;
		{
			uint32_t opat = p->i - 1;
			rhs = parse_expr(p, bp == BP_POW ? BP_POW : bp + 1);
			if (op == O_POW && p->t->len[opat] == 1 &&
			    TY_ISINT(N.ty[lhs]) && TY_ISINT(N.ty[rhs]) &&
			    strcmp(diag_file, "<english>") != 0 &&
			    bitwise_context(p, opat)) {
				static const char *const xo[] = {
				    "xor (bitwise exclusive or)", "power"};
				int k = nlp_ask_pref("caret", "\"^\" between integers, "
				    "next to bitwise operators", xo, 2, 0,
				    "write xor or ** to say which");
				if (nlp_asked)
					fail_quiet();
				if (k == 0)
					op = O_XOR;
			}
		}
		if (op == O_DIV && p->top_jux) {
			note("%s/%s read as %s/(%s); write (%s/...)%s for the "
			    "other reading", node_str(lhs), node_str(rhs),
			    node_str(lhs), node_str(rhs), node_str(lhs),
			    "...");
		}
		if (op == O_POW && TY_SIZED(N.ty[lhs]) && TYK(N.ty[lhs]) !=
		    K_BOOL && !is_const(rhs) && p->t->len[p->i - 1] == 1)
			note("'^' is power; use 'xor' or '⊕' for exclusive or");
		x = mk_bin(op, lhs, rhs);
		lhs = x;
		lastjux = 0;
		lparen = 0;
	}
	p->top_jux = lastjux;
	p->top_paren = lparen;
	return lhs;
}

/* ---- statements ---- */

static int
at_eos(struct P *p)
{
	return pk(p) == T_NL || pk(p) == T_EOF;
}

static void
expect_eos(struct P *p)
{
	diag_tok = p->i;
	if (!at_eos(p))
		fail("unexpected '%.*s' after the statement",
		    (int)p->t->len[p->i], p->t->src + p->t->off[p->i]);
}

/* default type of untyped parameters from what the body uses */
static ty_t
infer_param_type(struct P *p, uint32_t from, int *why)
{
	uint32_t i;
	int depth = 0;
	static const char *const intwords[] = {
		"xor", "mod", "popcount", "popcnt", "clz", "ctz", "isqrt",
		"match", "switch", NULL
	};

	*why = 0;
	for (i = from; i < p->t->n; i++) {
		int k = p->t->kind[i];
		if (k == T_LP || k == T_LB)
			depth++;
		if (k == T_RP || k == T_RB)
			depth--;
		if ((k == T_NL && depth <= 0) || k == T_EOF)
			break;
		/* in a command, '~' and option words end the expression */
		if (p->cmdargs && depth <= 0 && (k == T_TILDE ||
		    word_in(p, i, stopwords)))
			break;
		if (k == T_AMP || k == T_XOR || k == T_SHL || k == T_SHR ||
		    k == T_PCT || k == T_TILDE || k == T_LB) {
			*why = 1;
			return TY_INT;
		}
		if (k == T_ID && word_in(p, i, intwords)) {
			*why = 1;
			return TY_INT;
		}
	}
	return TY_REAL;
}

static void
parse_bound(struct P *p, int64_t *iv, double *rv, int *isreal)
{
	/* any constant expression: -pi/2, 1e-3, 2^16 - 1 */
	struct func *saved = p->fn;
	nid v;

	p->fn = NULL;
	v = parse_expr(p, BP_CMP + 1);
	p->fn = saved;
	if (!is_const(v))
		fail("a range bound must be a constant expression");
	if (TY_FP(N.ty[v])) {
		*isreal = 1;
		*rv = N.kr[v];
		*iv = (int64_t)*rv;
	} else {
		*iv = N.ki[v];
		*rv = (double)*iv;
	}
}

static void
parse_range(struct P *p, struct param *pp, int *realrange)
{
	int r1 = 0, r2 = 0;

	parse_bound(p, &pp->lo, &pp->rlo, &r1);
	expect(p, T_DOTDOT);
	parse_bound(p, &pp->hi, &pp->rhi, &r2);
	pp->ranged = 1;
	*realrange = r1 || r2;
	if (pp->rlo > pp->rhi)
		fail("empty range");
}

static void
parse_params(struct P *p, struct func *f, int *typed)
{
	struct param *pp;
	size_t n;
	const char *s;
	ty_t t;
	int rr;

	expect(p, T_LP);
	if (accept(p, T_RP))
		return;
	do {
		if (f->np == MAXP)
			fail("at most %d parameters", MAXP);
		if (pk(p) != T_ID)
			fail("expected a parameter name");
		pp = &f->p[f->np];
		memset(pp, 0, sizeof *pp);
		pp->name = tsym(p, p->i);
		pp->ty = 0;
		adv(p);
		rr = 0;
		if (accept(p, T_COLON)) {
			if (pk(p) == T_NUM || pk(p) == T_MINUS) {
				parse_range(p, pp, &rr);
			} else {
				s = ttext(p, p->i, &n);
				if (pk(p) != T_ID || !ty_parse(s, n, &t))
					fail("unknown type '%.*s' (real, f32, f64, "
					    "int, bool, uN, iN)", (int)n, s);
				pp->ty = t;
				adv(p);
			}
		}
		if (accept_kw(p, "uniform"))
			pp->uniform = 1;
		if (accept_kw(p, "in"))
			parse_range(p, pp, &rr);
		if (accept_kw(p, "uniform"))
			pp->uniform = 1;
		if (pp->ty == 0 && pp->ranged)
			pp->ty = rr ? TY_REAL : TY_INT;
		if (pp->ty && TY_ISINT(pp->ty) && rr)
			fail("integer parameter with a real range");
		if (pp->ty)
			typed[f->np] = 1;
		f->np++;
	} while (accept(p, T_COMMA));
	expect(p, T_RP);
}

/* where y = e, cond, ... : bindings first, then the body, then filters */
static uint32_t
find_where(struct P *p, uint32_t from)
{
	uint32_t i;
	int depth = 0;

	for (i = from; i < p->t->n; i++) {
		int k = p->t->kind[i];
		if (k == T_LP || k == T_LB || k == T_LC)
			depth++;
		if (k == T_RP || k == T_RB || k == T_RC)
			depth--;
		if (k == T_EOF)
			break;
		if (k == T_NL && depth <= 0) {
			if (i + 1 < p->t->n && tok_is(p->t, i + 1, "where"))
				continue;
			break;
		}
		if (depth <= 0 && tok_is(p->t, i, "where"))
			return i;
	}
	return NOTOK;
}

static uint32_t
parse_where(struct P *p, uint32_t w, nid *cond)
{
	*cond = 0;
	p->i = w;
	while (accept_kw(p, "where") || accept(p, T_COMMA)) {
		skip_nl(p);
		if (pk(p) == T_ID && pk2(p) == T_ASSIGN &&
		    param_slot(p, tsym(p, p->i)) < 0) {
			if (p->nb == MAXBIND)
				fail("too many where bindings");
			p->bname[p->nb] = tsym(p, p->i);
			adv(p);
			adv(p);
			p->bval[p->nb++] = parse_expr(p, 0);
		} else {
			nid c = mk_bool(parse_expr(p, 0));
			*cond = *cond ? mk_bin(O_LAND, *cond, c) : c;
		}
		while (pk(p) == T_NL && p->i + 1 < p->t->n &&
		    tok_is(p->t, p->i + 1, "where"))
			adv(p);
	}
	expect_eos(p);
	return p->i;
}

static void
parse_def(struct P *p)
{
	uint32_t name = tsym(p, p->i), w, end;
	struct func tmp, *f;
	int typed[MAXP] = {0}, why, j;
	ty_t dt;
	nid cond = 0;

	memset(&tmp, 0, sizeof tmp);
	tmp.name = name;
	adv(p);
	if (pk(p) == T_LP)
		parse_params(p, &tmp, typed);
	else
		p->autop = 1;
	expect(p, T_ASSIGN);
	dt = infer_param_type(p, p->i, &why);
	for (j = 0; j < tmp.np; j++)
		if (!typed[j] && tmp.p[j].ty == 0) {
			tmp.p[j].ty = dt;
			if (why)
				note("%s has no type: inferred %s from the "
				    "integer operations in the body",
				    sym_str(tmp.p[j].name), ty_str(dt));
		}
	p->fn = &tmp;
	p->autoty = dt;
	p->nb = 0;
	w = find_where(p, p->i);
	end = 0;
	if (w != NOTOK) {
		uint32_t bodyat = p->i;
		end = parse_where(p, w, &cond);
		p->i = bodyat;
	}
	tmp.body = parse_expr(p, 0);
	while (pk(p) == T_NL && w != NOTOK && p->i < w)
		adv(p);
	if (w != NOTOK) {
		diag_tok = p->i;
		if (p->i != w)
			fail("unexpected '%.*s'", (int)p->t->len[p->i],
			    p->t->src + p->t->off[p->i]);
		p->i = end;
	}
	expect_eos(p);
	if (p->autop && tmp.np > 0) {
		struct sbuf b = {0};
		for (j = 0; j < tmp.np; j++)
			sb_printf(&b, "%s%s", j ? ", " : "",
			    sym_str(tmp.p[j].name));
		note("%s has free variables: defined as %s(%s)",
		    sym_str(name), sym_str(name), sb_str(&b));
		free(b.p);
	}
	tmp.where = cond;
	f = func_new(name);
	*f = tmp;
	p->fn = NULL;
	p->autop = 0;
	p->nb = 0;
}

/* a function argument to a command: a name, or an anonymous expression */
static struct func *
cmd_func(struct P *p)
{
	static uint32_t anon;
	static unsigned anon_turn;
	struct func *f, tmp;
	uint32_t at = p->i;
	int why;

	if (pk(p) == T_ID && (f = func_find(tsym(p, at))) != NULL &&
	    (at + 1 >= p->t->n || p->t->kind[at + 1] == T_NL ||
	    p->t->kind[at + 1] == T_EOF || p->t->kind[at + 1] == T_TILDE ||
	    word_in(p, at + 1, stopwords) || p->t->kind[at + 1] == T_ID ||
	    p->t->kind[at + 1] == T_NUM)) {
		adv(p);
		return f;
	}
	/* a command takes at most two anonymous functions: alternate slots */
	anon = sym_cstr(anon_turn++ & 1 ? "_2" : "_");
	/*
	 * A unary function's name alone, "integrate W0", "range exp": the
	 * function of x.  (A bare one-letter W stays a variable.)
	 */
	if (pk(p) == T_ID && builtin(p, at) != NULL &&
	    builtin(p, at)->arity == 1 && builtin(p, at)->op != O_BARRIER &&
	    (at + 1 >= p->t->n || p->t->kind[at + 1] == T_NL ||
	    p->t->kind[at + 1] == T_EOF || p->t->kind[at + 1] == T_TILDE ||
	    word_in(p, at + 1, stopwords) ||
	    (p->xstop && word_in(p, at + 1, p->xstop)))) {
		const struct bfn *bf = builtin(p, at);
		nid xv;
		memset(&tmp, 0, sizeof tmp);
		tmp.name = anon;
		tmp.anon = 1;
		tmp.np = 1;
		tmp.p[0].name = sym_cstr("x");
		tmp.p[0].ty = TY_REAL;
		xv = mk_var(0, tmp.p[0].name, TY_REAL);
		adv(p);
		tmp.body = apply_builtin(p, bf, &xv, 1);
		note("%s alone read as %s", bf->name, node_str(tmp.body));
		f = func_new(anon);
		*f = tmp;
		return f;
	}
	if (tok_is(p->t, at, "W") && func_find(tsym(p, at)) == NULL &&
	    (at + 1 >= p->t->n || p->t->kind[at + 1] == T_NL ||
	    p->t->kind[at + 1] == T_EOF || word_in(p, at + 1, stopwords)))
		note("W alone is a variable here; W(x) or W0 is the Lambert W "
		    "function");
	memset(&tmp, 0, sizeof tmp);
	tmp.name = anon;
	tmp.anon = 1;
	p->fn = &tmp;
	p->autop = 1;
	p->cmdargs = 1;
	p->autoty = infer_param_type(p, p->i, &why);
	if (p->force_ty)
		p->autoty = p->force_ty;
	tmp.body = parse_expr(p, 0);
	p->cmdargs = 0;
	p->autop = 0;
	p->fn = NULL;
	f = func_new(anon);
	*f = tmp;
	return f;
}

/* ---- prelude: bases and the cost model ---- */

static const char prelude[] =
    "basis c { and(x, y) = x & y; or(x, y) = x | y; xor(x, y) = x xor y;"
    " not(x) = ~x }\n"
    "basis spirv : c {}\n"
    "basis a64 : c { bic(x, y) = x & ~y; orn(x, y) = x | ~y;"
    " eon(x, y) = x xor ~y }\n"
    "basis a64sha : a64 { bsl(s, x, y) = (s & x) | (~s & y);"
    " eor3(x, y, z) = x xor y xor z; bcax(x, y, z) = x xor (y & ~z) }\n"
    "basis x86bmi : c { andn(x, y) = ~x & y }\n"
    "basis nand { nand(x, y) = ~(x & y) }\n"
    /*
     * NVIDIA GT 1030 (GP108, sm_61) through SPIR-V.  Relative issue cost
     * per warp instruction, from the sm_61 per-SM throughput table:
     * fp32 add/mul/fma 128/clk -> 1; int add/logic 128 -> 1; shifts and
     * compares 64 -> 2; 32-bit IMUL is an XMAD sequence -> 3; integer
     * division is emulated -> 24, by a constant it is mul+shift -> 5;
     * SFU (rsqrt, rcp, sin, cos, ex2, lg2) 32 -> 4; conversions 32 -> 4;
     * fp64 is 1/32 rate -> 32; IEEE fdiv/sqrt are SFU + Newton fixups.
     * A divergent branch runs both arms plus reconvergence ("keep-up").
     */
    "model gt1030 {\n"
    "  iadd: 1, logic: 1, shift: 2, imul: 3, idiv: 24, idivk: 5, cmp: 2,\n"
    "  select: 1, minmax: 2, bitcnt: 4, conv: 4, bitcast: 0,\n"
    "  fadd: 1, fmul: 1, fdiv: 10, fsqrt: 8, ftrans: 4, fpow: 10, f64: 32,\n"
    "  load: 8, branch: 2, diverge: 6, barrier: 20,\n"
    "  atomic_shared: 8, atomic_global: 64, store: 8\n"
    "}\n";

static void
parse_basis(struct P *p)
{
	struct basis *b, *parent;
	struct func tmp;
	struct gate *g;
	uint32_t name;
	int j;

	adv(p);
	if (pk(p) != T_ID)
		fail("expected a basis name");
	name = tsym(p, p->i);
	adv(p);
	b = basis_new(name);
	if (accept(p, T_COLON)) {
		if (pk(p) != T_ID || (parent = basis_find(tsym(p, p->i))) == NULL)
			fail("unknown basis");
		adv(p);
		b->ng = parent->ng;
		memcpy(b->g, parent->g, sizeof b->g);
	}
	skip_nl(p);
	expect(p, T_LC);
	for (;;) {
		skip_nl(p);
		if (accept(p, T_RC))
			break;
		if (pk(p) != T_ID)
			fail("expected a gate: name(x, y) = expr");
		if (b->ng == MAXGATE)
			fail("too many gates");
		g = &b->g[b->ng];
		memset(g, 0, sizeof *g);
		memset(&tmp, 0, sizeof tmp);
		g->name = tsym(p, p->i);
		adv(p);
		expect(p, T_LP);
		do {
			if (pk(p) != T_ID || tmp.np == 3)
				fail("gates take 1 to 3 named inputs");
			tmp.p[tmp.np].name = tsym(p, p->i);
			tmp.p[tmp.np].ty = TY(K_UINT, 1);
			g->pname[tmp.np] = tmp.p[tmp.np].name;
			tmp.np++;
			adv(p);
		} while (accept(p, T_COMMA));
		expect(p, T_RP);
		expect(p, T_ASSIGN);
		p->fn = &tmp;
		g->body = mk_cast(TY(K_UINT, 1), parse_expr(p, 0));
		p->fn = NULL;
		g->arity = (uint8_t)tmp.np;
		g->cost = 1;
		if (accept_kw(p, "cost")) {
			nid c = number(p);
			g->cost = (uint8_t)N.ki[c];
		}
		/* a gate of the same name replaces the inherited one */
		for (j = 0; j < b->ng; j++)
			if (b->g[j].name == g->name) {
				b->g[j] = *g;
				break;
			}
		if (j == b->ng) {
			basis_finish_gate(b, g);
			b->ng++;
		} else
			basis_finish_gate(b, &b->g[j]);
	}
	expect_eos(p);
}

static void
parse_model(struct P *p)
{
	struct model *m, *parent;
	size_t n;
	const char *s;
	int k, inherited = 0, store_given = 0;
	nid v;

	adv(p);
	if (pk(p) != T_ID)
		fail("expected a model name");
	m = model_new(tsym(p, p->i));
	adv(p);
	if (accept(p, T_COLON)) {
		if (pk(p) != T_ID || (parent = model_find(tsym(p, p->i))) == NULL)
			fail("unknown model");
		adv(p);
		memcpy(m->c, parent->c, sizeof m->c);
		inherited = 1;
	}
	skip_nl(p);
	expect(p, T_LC);
	for (;;) {
		while (accept(p, T_NL) || accept(p, T_COMMA))
			;
		if (accept(p, T_RC))
			break;
		s = ttext(p, p->i, &n);
		if (pk(p) != T_ID)
			fail("expected a cost class");
		k = (n == 3 && !memcmp(s, "all", 3)) ? -1 : cclass_find(s, n);
		if (k == -2)
			fail("unknown cost class '%.*s' (see 'models')", (int)n, s);
		adv(p);
		if (!accept(p, T_COLON))
			expect(p, T_ASSIGN);
		v = number(p);
		if (k == -1) {
			for (k = 0; k < C_NCLASS; k++)
				m->c[k] = TY_FP(N.ty[v]) ? (float)N.kr[v] :
				    (float)N.ki[v];
		} else {
			m->c[k] = TY_FP(N.ty[v]) ? (float)N.kr[v] :
			    (float)N.ki[v];
			if (k == C_STORE)
				store_given = 1;
		}
	}
	/* a model that does not price stores: as dear as its loads */
	if (!inherited && !store_given)
		m->c[C_STORE] = m->c[C_LOAD];
	expect_eos(p);
}

/* ---- commands ---- */

static int
opt_int(struct P *p, const char *kw, int dflt)
{
	if (accept_kw(p, kw)) {
		nid v = number(p);
		return (int)N.ki[v];
	}
	return dflt;
}

static void
help(void)
{
	puts(
"definitions\n"
"  f(x: u8, y: i16 in -5..5) = expr [where y = expr, cond]\n"
"  g = x^2 + 1               free variables become parameters\n"
"  basis name [: parent] { gate(x, y) = expr [cost n]; ... }\n"
"  model name [: parent] { class: cost, ... }\n"
"types: real (math), f32 f64 (IEEE), int (math), uN iN (wrapping), bool\n"
"commands (F is a function name or an expression like isqrt(x: u4))\n"
"  show F | parse EXPR | table F [max n] | funcs | models | help\n"
"  range F [nocrit] | domain F | wrap F | crit F | diff F [x]\n"
"  equiv F G | error G ~ F | fit G ~ F [metric maxrel|maxabs|rms]\n"
"  gates F [basis b] [max k] [nodes n] [as c]\n"
"  find F [n] [max cost] | more [n]\n"
"  cost F [model m] | use m\n"
"  approx F [in a..b] [degree n|a..b] [error e] [ulp u] [as f32|f64|real]\n"
"    [relative|absolute] [odd|even|nosym] [keep endpoints|x1, x2..]\n"
"    [weight W] [at c] [fma|nofma] [norefine] [list n] [name p] [c]\n"
"  approx more [n]\n"
"  animate [hillis|blelloch|sequential] scan [n N] [fps F] [seconds S]\n"
"    [random [seed K] | values a, b, ...] [f32] [MODEL] [to NAME]\n"
"loops and calculus\n"
"  sum(i, lo, hi, e) | prod(...) | x[0] = c; x[k+1] = ... | { for/while/if }\n"
"  closed F | scheme F [model m] | continuous F\n"
"  integrate F [wrt x] [from a to b] | antiderivative F [wrt x]\n"
"  diff F [x y ...] [n] | grad|hessian|laplacian F [at a, b] | F'' | F'(a)\n"
"  jacobian F, G [wrt x, y] [at ...] | ddir F along v [at p] [raw]\n"
"  limit F x -> a[+|-] [from left|right] | lim x->a F | wronskian F, G, ...\n"
"  taylor F [x] [at a] [order n] | implicit L = R [for y] [wrt x] [order 2]\n"
"  y' = f(t, y); y(0) = c | pde name { ... } | dsolve y [from a to b] "
"[steps n]\n"
"complex numbers (i is the unit) and trig forms\n"
"  complex F | re F | im F | conj F | modulus F | arg F | polar F\n"
"  nthroots N F | unity N | trig F [expand|reduce|product|sum|sincos|exp|"
"harmonic|half|inverse]\n"
"English works too: 'what is the range of x squared for a signed byte x?'\n"
"  ('help english' for more)");
}

/* ---- approx: parsing hooks (the command itself is in approx.c) ---- */

int
cmdp_kw(struct P *p, const char *w)
{
	return accept_kw(p, w);
}

int
cmdp_tok(struct P *p, int kind)
{
	return accept(p, kind);
}

int
cmdp_at(struct P *p, int kind)
{
	return pk(p) == kind;
}

int
cmdp_eos(struct P *p)
{
	return at_eos(p);
}

void
cmdp_fail_here(struct P *p)
{
	diag_tok = p->i;
}

/* an identifier, copied to buf */
int
cmdp_word(struct P *p, char *buf, size_t n)
{
	size_t len;
	const char *s;

	if (pk(p) != T_ID)
		return 0;
	s = ttext(p, p->i, &len);
	snprintf(buf, n, "%.*s", (int)len, s);
	adv(p);
	return 1;
}

/* a function argument that also stops at 'in' and the words in stop[] */
struct func *
cmdp_func(struct P *p, const char *const *stop)
{
	struct func *f;

	p->xstop = stop;
	f = cmd_func(p);
	p->xstop = NULL;
	return f;
}

/* token position, to look ahead and back off ("32-bit" or "32*x") */
uint32_t
cmdp_pos(struct P *p)
{
	return p->i;
}

void
cmdp_seek(struct P *p, uint32_t pos)
{
	p->i = pos;
	diag_tok = pos;
}

/*
 * the rest of the statement as written, up to the end of its line or
 * the first identifier in stop[] (may be NULL)
 */
int
cmdp_rest(struct P *p, char *buf, size_t n, const char *const *stop)
{
	uint32_t a = p->i, b = p->i;
	size_t len;
	int from = 0;

	if (at_eos(p))
		return 0;

	while (b < p->t->n && p->t->kind[b] != T_NL && p->t->kind[b] != T_EOF) {
		if (stop != NULL && b > a && p->t->kind[b] == T_ID) {
			size_t l, k;
			const char *w = ttext(p, b, &l);
			for (k = 0; stop[k]; k++)
				if (strlen(stop[k]) == l && !strncmp(stop[k], w, l))
					break;
			if (l == 4 && !strncmp(w, "from", 4))
				from = 1;
			/*
			 * "to NAME" (a file) ends it only as one last word with
			 * no "from" before it: "from 0 to pi" is a range
			 */
			if (stop[k] && l == 2 && !strncmp(w, "to", 2)) {
				uint8_t k2 = b + 2 < p->t->n ? p->t->kind[b + 2] : T_EOF;
				if (!from && b + 1 < p->t->n && p->t->kind[b + 1] == T_ID &&
				    (k2 == T_NL || k2 == T_EOF))
					break;
				b++;
				continue;
			}
			if (stop[k])
				break;
		}
		b++;
	}
	len = p->t->off[b - 1] + p->t->len[b - 1] - p->t->off[a];
	if (len >= n)
		len = n - 1;
	memcpy(buf, p->t->src + p->t->off[a], len);
	buf[len] = 0;
	p->i = b;
	return 1;
}

/* a plain integer literal token, else 0 */
int
cmdp_int(struct P *p, int64_t *v)
{
	nid n;

	if (pk(p) != T_NUM)
		return 0;
	n = number(p);
	if (!is_const(n) || TY_FP(N.ty[n]))
		return 0;
	*v = N.ki[n];
	return 1;
}

/* an anonymous function whose free variables all have type ty */
struct func *
cmdp_func_typed(struct P *p, const char *const *stop, ty_t ty)
{
	struct func *f;

	p->xstop = stop;
	p->force_ty = ty;
	f = cmd_func(p);
	p->force_ty = 0;
	p->xstop = NULL;
	return f;
}

/* a constant expression such as -pi/2; stops at '..' and ',' */
double
cmdp_number(struct P *p, const char *const *stop)
{
	nid e;

	diag_tok = p->i;
	p->xstop = stop;
	p->cmdargs = 1;
	e = parse_expr(p, BP_CMP + 1);
	p->cmdargs = 0;
	p->xstop = NULL;
	if (!is_const(e))
		fail("expected a constant, found %s", node_str(e));
	return TY_FP(N.ty[e]) ? N.kr[e] : (double)N.ki[e];
}

static void
statement(struct P *p)
{
	struct func *f, *g;
	struct basis *b;
	struct model *m;
	nid e;
	int n, k;

	skip_nl(p);
	if (pk(p) == T_EOF)
		return;
	diag_tok = p->i;
	if (pk(p) == T_ID && (pk2(p) == T_ID || pk2(p) == T_NUM ||
	    pk2(p) == T_NL || pk2(p) == T_EOF || pk2(p) == T_LP ||
	    pk2(p) == T_BAR || pk2(p) == T_MINUS || pk2(p) == T_SQRT ||
	    pk2(p) == T_LB)) {
		/* commands */
		if (is_kw(p, "basis") && pk2(p) == T_ID) {
			parse_basis(p);
			return;
		}
		if (is_kw(p, "model") && pk2(p) == T_ID) {
			parse_model(p);
			return;
		}
#define CMD(w) (is_kw(p, w) && (adv(p), 1))
		if (calc2_command(p))
			return;
		/* simplify a SPIR-V block (spirv.c), from "SAT simplify" ... */
		if (is_kw(p, "spvsimplify")) {
			double budget = 6;
			int nofile = 0;
			adv(p);
			while (!at_eos(p)) {
				if (accept_kw(p, "budget") && pk(p) == T_NUM)
					budget = cmdp_number(p, NULL);
				else if (accept_kw(p, "nofile"))
					nofile = 1;
				else if (accept_kw(p, "assume") && pk(p) == T_ID) {
					char nm[48];
					int64_t lo, hi;
					int neg;
					cmdp_word(p, nm, sizeof nm);
					neg = accept(p, T_MINUS);
					if (!cmdp_int(p, &lo))
						fail("simplify: assume NAME LO HI");
					lo = neg ? -lo : lo;
					neg = accept(p, T_MINUS);
					if (!cmdp_int(p, &hi))
						fail("simplify: assume NAME LO HI");
					hi = neg ? -hi : hi;
					spirv_assume(nm, lo, hi);
				} else
					adv(p);
			}
			if (code_block == NULL)
				fail("simplify: attach the SPIR-V (a ```spv block or a "
				    "file)");
			if (budget < 0.5)
				budget = 0.5;
			if (budget > 60)
				budget = 60;
			spirv_simplify(code_block, budget, nofile);
			return;
		}
		/* animate (anim.c) */
		if (is_kw(p, "animate") && !func_find(tsym(p, p->i))) {
			adv(p);
			cmd_animate(p);
			return;
		}
		/* monotonic, period, identify, zeros ... (props.c) */
		if (props_command(p))
			return;
		/* approx (approx.c) */
		if (is_kw(p, "approx") && !func_find(tsym(p, p->i))) {
			adv(p);
			cmd_approx(p);
			return;
		}
		if (CMD("help")) {
			help();
			return;
		}
		if (CMD("funcs")) {
			for (k = 0; k < nfuncs; k++)
				if (!funcs[k].anon)
					cmd_show(&funcs[k]);
			return;
		}
		if (CMD("models")) {
			cmd_models();
			return;
		}
		if (CMD("use")) {
			if (pk(p) != T_ID || (m = model_find(tsym(p, p->i))) == NULL)
				fail("unknown model");
			adv(p);
			expect_eos(p);
			model_set_active(m);
			printf("cost model: %s\n", sym_str(m->name));
			return;
		}
		if (CMD("more")) {
			n = pk(p) == T_NUM ? (int)N.ki[number(p)] : 5;
			expect_eos(p);
			cmd_more(n);
			return;
		}
		if (CMD("parse")) {
			struct func tmp;
			struct sbuf sb = {0};
			memset(&tmp, 0, sizeof tmp);
			tmp.name = sym_cstr("_");
			p->fn = &tmp;
			p->autop = 1;
			p->autoty = infer_param_type(p, p->i, &k);
			e = parse_expr(p, 0);
			p->fn = NULL;
			p->autop = 0;
			expect_eos(p);
			print_paren(&sb, e);
			printf("%s\n", sb_str(&sb));
			free(sb.p);
			return;
		}
		if (pk(p) == T_ID && (is_kw(p, "show") || is_kw(p, "table") ||
		    is_kw(p, "range") || is_kw(p, "domain") ||
		    is_kw(p, "wrap") || is_kw(p, "crit") || is_kw(p, "diff") ||
		    is_kw(p, "equiv") || is_kw(p, "gates") || is_kw(p, "find") ||
		    is_kw(p, "fit") || is_kw(p, "error") || is_kw(p, "cost")) &&
		    !func_find(tsym(p, p->i))) {
			char cmd[16];
			size_t cn;
			const char *cs = ttext(p, p->i, &cn);
			snprintf(cmd, sizeof cmd, "%.*s", (int)cn, cs);
			adv(p);
			f = cmd_func(p);
			if (!strcmp(cmd, "show")) {
				expect_eos(p);
				notes_flush();
				cmd_show(f);
			} else if (!strcmp(cmd, "table")) {
				n = opt_int(p, "max", 64);
				expect_eos(p);
				notes_flush();
				cmd_table(f, n);
			} else if (!strcmp(cmd, "range")) {
				int nocrit = accept_kw(p, "nocrit");
				expect_eos(p);
				notes_flush();
				range_nocrit = nocrit;
				cmd_range(f);
				range_nocrit = 0;
			} else if (!strcmp(cmd, "domain")) {
				expect_eos(p);
				notes_flush();
				cmd_domain(f);
			} else if (!strcmp(cmd, "wrap")) {
				expect_eos(p);
				notes_flush();
				cmd_wrap(f);
			} else if (!strcmp(cmd, "crit")) {
				expect_eos(p);
				notes_flush();
				cmd_crit(f);
			} else if (!strcmp(cmd, "diff")) {
				k = 0;
				if (pk(p) == T_ID) {
					uint32_t nm = tsym(p, p->i);
					for (k = 0; k < f->np; k++)
						if (f->p[k].name == nm)
							break;
					if (k == f->np)
						fail("%s has no parameter %s",
						    sym_str(f->name), sym_str(nm));
					adv(p);
				}
				expect_eos(p);
				notes_flush();
				cmd_diff(f, k);
			} else if (!strcmp(cmd, "equiv") || !strcmp(cmd, "fit") ||
			    !strcmp(cmd, "error")) {
				int metric = 0;
				accept(p, T_TILDE);
				accept_kw(p, "to");
				g = cmd_func(p);
				if (accept_kw(p, "metric")) {
					if (accept_kw(p, "maxrel"))
						metric = 0;
					else if (accept_kw(p, "maxabs"))
						metric = 1;
					else if (accept_kw(p, "rms"))
						metric = 2;
					else
						fail("metric is maxrel, maxabs or rms");
				}
				expect_eos(p);
				notes_flush();
				if (!strcmp(cmd, "equiv"))
					cmd_equiv(f, g);
				else if (!strcmp(cmd, "fit"))
					cmd_fit(f, g, metric);
				else
					cmd_error(f, g);
			} else if (!strcmp(cmd, "gates")) {
				int emit = 0, maxc = 12;
				b = basis_find(sym_cstr("c"));
				for (;;) {
					if (accept_kw(p, "basis")) {
						if (pk(p) != T_ID ||
						    (b = basis_find(tsym(p, p->i))) == NULL)
							fail("unknown basis");
						adv(p);
					} else if (is_kw(p, "max"))
						maxc = opt_int(p, "max", 12);
					else if (accept_kw(p, "nodes")) {
						nid v = number(p);
						gates_budget = TY_FP(N.ty[v]) ?
						    (uint64_t)N.kr[v] : (uint64_t)N.ki[v];
					}
					else if (accept_kw(p, "as")) {
						if (!accept_kw(p, "c"))
							fail("only 'as c' is supported");
						emit = 1;
					} else
						break;
				}
				expect_eos(p);
				notes_flush();
				cmd_gates(f, b, maxc, emit);
			} else if (!strcmp(cmd, "find")) {
				int want = 5, maxc = 0;
				if (pk(p) == T_NUM)
					want = (int)N.ki[number(p)];
				maxc = opt_int(p, "max", 0);
				expect_eos(p);
				notes_flush();
				cmd_find(f, want, NULL, maxc);
			} else if (!strcmp(cmd, "cost")) {
				m = model_active();
				if (accept_kw(p, "model")) {
					if (pk(p) != T_ID ||
					    (m = model_find(tsym(p, p->i))) == NULL)
						fail("unknown model");
					adv(p);
				}
				expect_eos(p);
				notes_flush();
				cmd_cost(f, m);
			}
			return;
		}
		if (calc_command(p))
			return;
		if (forms_command(p))   /* complex, polar, trig ... (forms.c) */
			return;
#undef CMD
	}
	if (calc_statement(p))
		return;
	/* definition? */
	if (pk(p) == T_ID) {
		uint32_t i = p->i + 1;
		int depth = 0;
		if (p->t->kind[i] == T_LP) {
			for (; i < p->t->n; i++) {
				if (p->t->kind[i] == T_LP)
					depth++;
				else if (p->t->kind[i] == T_RP && --depth == 0)
					break;
				else if (p->t->kind[i] == T_EOF ||
				    p->t->kind[i] == T_NL)
					break;
			}
			i++;
		}
		if (i < p->t->n && p->t->kind[i] == T_ASSIGN) {
			parse_def(p);
			return;
		}
	}
	/* bare expression: evaluate it */
	{
		struct func tmp;
		memset(&tmp, 0, sizeof tmp);
		tmp.name = sym_cstr("_");
		tmp.anon = 1;
		p->fn = &tmp;
		p->autop = 1;
		p->autoty = infer_param_type(p, p->i, &k);
		e = parse_expr(p, 0);
		p->fn = NULL;
		p->autop = 0;
		expect_eos(p);
		notes_flush();
		if (tmp.np == 0 && is_const(e)) {
			struct sbuf sb = {0};
			fmt_val(&sb, N.ty[e], N.ki[e], N.kr[e]);
			printf("%s\n", sb_str(&sb));
			free(sb.p);
		} else if (tmp.np == 0 && N.op[e] == O_UNDEF) {
			printf("undefined\n");
		} else {
			tmp.body = e;
			f = func_new(tmp.name);
			*f = tmp;
			cmd_show(f);
		}
	}
}

static void
echo_stmt(struct P *p)
{
	uint32_t i = p->i, s, e;
	const char *src = p->t->src;

	while (i < p->t->n && p->t->kind[i] == T_NL)
		i++;
	if (i >= p->t->n || p->t->kind[i] == T_EOF)
		return;
	s = p->t->off[i];
	e = s;
	/* through the end of the statement's last line */
	{
		uint32_t j = i;
		int depth = 0;
		for (; j < p->t->n; j++) {
			int k = p->t->kind[j];
			if (k == T_LC)
				depth++;
			if (k == T_RC)
				depth--;
			if (k == T_EOF)
				break;
			if (k == T_NL && depth <= 0) {
				if (j + 1 < p->t->n && tok_is(p->t, j + 1, "where"))
					continue;
				break;
			}
			e = p->t->off[j] + p->t->len[j];
		}
	}
	printf("> ");
	for (; s < e; s++) {
		putchar(src[s]);
		if (src[s] == '\n')
			printf("  ");
	}
	putchar('\n');
}

static void
skip_stmt(struct P *p)
{
	int depth = 0;

	while (pk(p) != T_EOF) {
		if (pk(p) == T_LC)
			depth++;
		if (pk(p) == T_RC)
			depth--;
		if (pk(p) == T_NL && depth <= 0) {
			adv(p);
			if (!is_kw(p, "where"))
				break;
			continue;
		}
		adv(p);
	}
}

static void run_source1(const char *src, const char *file, int echo);

/*
 * Runs may nest (animate exec runs an approximation request): the
 * caller's fail_jmp is put back, or a later fail() jumps into a frame
 * that has returned.
 */
void
run_source(const char *src, const char *file, int echo)
{
	jmp_buf saved;
	struct toks *dt = diag_toks;
	const char *df = diag_file;
	uint32_t di = diag_tok;

	memcpy(saved, fail_jmp, sizeof saved);
	run_source1(src, file, echo);
	memcpy(fail_jmp, saved, sizeof saved);
	diag_toks = dt;	/* errors after it point at the caller's text */
	diag_file = df;
	diag_tok = di;
}

static void
run_source1(const char *src, const char *file, int echo)
{
	static int prelude_done;
	/* both live across setjmp, so they are reached through pointers */
	struct toks *const t = xcalloc(1, sizeof *t);
	struct P *const p = xcalloc(1, sizeof *p);

	nodes_init();
	if (!prelude_done) {
		prelude_done = 1;
		run_source(prelude, "<prelude>", 0);
		model_set_active(model_find(sym_cstr("gt1030")));
	}
	diag_file = file;
	diag_toks = t;
	p->t = t;
	if (setjmp(fail_jmp)) {   /* lexer error */
		toks_free(t);
		free(t);
		free(p);
		return;
	}
	lex(t, src);
	for (;;) {
		if (setjmp(fail_jmp)) {
			/* resume after the failed statement */
			p->fn = NULL;
			p->autop = 0;
			p->cmdargs = 0;
			p->xstop = NULL;
			p->force_ty = 0;
			p->abs_depth = 0;
			p->fnarg = 0;
			p->nb = 0;
			fflush(stdout);
			if (nlp_asked)
				break;   /* a question is out (fail_quiet) */
			skip_stmt(p);
			continue;
		}
		skip_nl(p);
		if (pk(p) == T_EOF)
			break;
		if (echo)
			echo_stmt(p);
		statement(p);
		notes_flush();
		fflush(stdout);
		if (nlp_asked)
			break;   /* a question is out: the rest waits for the answer */
		skip_nl(p);
	}
	toks_free(t);
	free(t);
	free(p);
	diag_toks = NULL;
}

/* calc: loops, sums, recurrences, ODE/PDE statements and commands */
#include "calc_syntax.inc"
#include "calc2_syntax.inc"
