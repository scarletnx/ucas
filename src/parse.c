/* SPDX-License-Identifier: GPL-3.0-or-later */
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
#include "cfront.h"
#include "prove.h"
#include "texalu.h"
#include "texenc.h"
#include "props.h"
#include "dword.h"
#include "memk.h"
#include "stoch.h"
#include "numsearch.h"
#include "ring.h"
#include "absalg.h"
#include "viz.h"
#include "xform.h"
#include "x86.h"
#include "fuse.h"
int homalg_command(struct P *p);	/* homalg.h */
int ec_command(struct P *p);	/* ec.h (absalg.h names a parameter I) */

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
	int sg_depth;     /* inside a subgroup operation's arguments */
	nid sgmask;       /* sg_active(m): the lanes that run, or 0 (all) */
	int sgops;        /* subgroup operations built so far */
	/*
	 * Loop states (bit = slot) that follow a value per lane (cs_loop):
	 * a subgroup operation reads each one's copy in the other lanes, as a
	 * parameter's.  sgread: the loop slots subgroup operations have read
	 * across the lanes.
	 */
	uint64_t lanest, sgread;
	uint32_t lane_name[64], lane_from[64];	/* the state's, what it follows */
	ty_t lane_ty[64];
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
static int alg_command(struct P *p);	/* alg_syntax.inc */
int ecff_command(struct P *p);		/* ecff_cmd.c: elliptic curves */

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
	"nodes", "from", "wrt", "steps", "nocrit", "numerically", NULL
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
	{"roundeven", O_ROUNDEVEN, 1},
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
	/* hyper.c: only when called, conj(z); a bare re or arg is a name */
	{"conj", O_CONJ, 1}, {"re", O_RE, 1}, {"im", O_IM, 1}, {"arg", O_ARG, 1},
	/* dot.c: packed integer dot products */
	{"udot4x8", O_UDOT4X8, 2}, {"sdot4x8", O_SDOT4X8, 2},
	{"sudot4x8", O_SUDOT4X8, 2}, {"udot2x16", O_UDOT2X16, 2},
	{"sdot2x16", O_SDOT2X16, 2}, {"sudot2x16", O_SUDOT2X16, 2},
	{"udot8x4", O_UDOT8X4, 2}, {"sdot8x4", O_SDOT8X4, 2},
	{"sudot8x4", O_SUDOT8X4, 2},
	{"udot4x8sat", O_UDOT4X8S, 3}, {"sdot4x8sat", O_SDOT4X8S, 3},
	{"sudot4x8sat", O_SUDOT4X8S, 3}, {"udot2x16sat", O_UDOT2X16S, 3},
	{"sdot2x16sat", O_SDOT2X16S, 3}, {"sudot2x16sat", O_SUDOT2X16S, 3},
	{"udot8x4sat", O_UDOT8X4S, 3}, {"sdot8x4sat", O_SDOT8X4S, 3},
	{"sudot8x4sat", O_SUDOT8X4S, 3},
	/* inst.c: the catalogue's instructions */
	{"bfe", O_BFE, 3}, {"sbfe", O_SBFE, 3}, {"brev", O_BREV, 1},
	{"mulhi", O_MULHI, 2}, {"smulhi", O_SMULHI, 2}, {"funl", O_FUNL, 3},
	{"funr", O_FUNR, 3}, {"med3", O_MED3, 3}, {"usad4x8", O_USAD4X8, 3},
	{"byteperm", O_BYTEPERM, 3}, {"bfi", O_BFI, 4},
	{"mulwide", O_MULWIDE, 2}, {"madwide", O_MADWIDE, 3},
	{"bytesel", O_BYTESEL, 3},
	{"add4x8", O_ADD4X8, 2}, {"sub4x8", O_SUB4X8, 2},
	{"absdiff4x8", O_ABSDIFF4X8, 2}, {"min4x8", O_MIN4X8, 2},
	{"max4x8", O_MAX4X8, 2}, {"msad4x8", O_MSAD4X8, 3},
	{"add2x16", O_ADD2X16, 2}, {"sub2x16", O_SUB2X16, 2},
	{"mul2x16", O_MUL2X16, 2}, {"min2x16", O_MIN2X16, 2},
	{"max2x16", O_MAX2X16, 2}, {"smin2x16", O_SMIN2X16, 2},
	{"smax2x16", O_SMAX2X16, 2},
	/* astc.c: the ASTC decoder's steps */
	{"astc_partition", O_ASTC_PART, 4}, {"astc_unquant", O_ASTC_UNQ, 2},
	{"astc_unquantw", O_ASTC_UNQW, 2}, {"astc_endpoint", O_ASTC_ENDPOINT, 3},
	{"astc_bilerp", O_ASTC_BILERP, 4}, {"astc_lerp", O_ASTC_LERP, 3},
	{NULL, 0, 0}
};

static const struct bfn *
builtin(struct P *p, uint32_t i)
{
	int k;

	for (k = 0; bfns[k].name; k++)
		if (tok_is(p->t, i, bfns[k].name)) {
			/* one letter (W), conj re im arg: only called, and a
			 * user's function wins */
			if ((bfns[k].name[1] == 0 || bfns[k].op == O_CONJ ||
			    bfns[k].op == O_RE || bfns[k].op == O_IM ||
			    bfns[k].op == O_ARG) && (i + 1 >= p->t->n ||
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

/*
 * Subgroup operations (sg.c), by ucas's names and GLSL's.  The subgroup's
 * size is the range of the function's `lane` parameter (lane: u32 in
 * 0..31), else the active model's `warp`.  In an operand that is a
 * value per lane the parameter becomes the operation's bound lane; an
 * operation that is itself a value per lane (a scan, a shuffle by the
 * lane) reads the lane it runs in.
 */
enum { SGF_OP, SGF_EXCL, SGF_SHUFXOR, SGF_BCAST, SGF_BCAST1, SGF_ROT,
	SGF_SEG, SGF_SORT, SGF_ACTIVE };
static const struct sgfn {
	const char *name, *glsl;
	int kind, op, arity;
} sgfns[] = {
	{"sg_ballot", "subgroupBallot", SGF_OP, O_SGBALLOT, 1},
	{"sg_any", "subgroupAny", SGF_OP, O_SGANY, 1},
	{"sg_all", "subgroupAll", SGF_OP, O_SGALL, 1},
	{"sg_add", "subgroupAdd", SGF_OP, O_SGADD, 1},
	{"sg_shuffle", "subgroupShuffle", SGF_OP, O_SGSHUF, 2},
	{"sg_scan", "subgroupInclusiveAdd", SGF_OP, O_SGSCAN, 1},
	{"sg_exclusive", "subgroupExclusiveAdd", SGF_EXCL, O_SGSCAN, 1},
	{"sg_shuffle_xor", "subgroupShuffleXor", SGF_SHUFXOR, O_SGSHUF, 2},
	{"sg_broadcast", "subgroupBroadcast", SGF_BCAST, O_SGSHUF, 2},
	{"sg_broadcast_first", "subgroupBroadcastFirst", SGF_BCAST1, O_SGSHUF, 1},
	{"sg_rotate", "subgroupRotate", SGF_ROT, O_SGSHUF, 2},
	{"sg_segscan", "sg_segscan", SGF_SEG, O_SGSCAN, 2},
	{"sg_sort", "sg_sort", SGF_SORT, O_SGSHUF, 1},
	{"sg_active", "sg_active", SGF_ACTIVE, O_SGBALLOT, 1},
	{NULL, NULL, 0, 0, 0}
};

static const struct sgfn *
sg_builtin(struct P *p, uint32_t i)
{
	int k;

	if (i + 1 >= p->t->n || p->t->kind[i + 1] != T_LP)
		return NULL;
	for (k = 0; sgfns[k].name; k++)
		if (tok_is(p->t, i, sgfns[k].name) || tok_is(p->t, i, sgfns[k].glsl))
			return &sgfns[k];
	return NULL;
}

static int
sg_lane_param(struct P *p)
{
	return param_slot(p, sym_cstr("lane"));
}

/*
 * A value per lane is a function of the lane and of what the lanes share:
 * a parameter that is not `uniform` has one value in each lane, which an
 * operand evaluated in every lane cannot name.
 */
static void
sg_shared_only(struct P *p, nid e, const char *what)
{
	uint64_t mask = 0;
	int j, s;

	if (p->fn == NULL)
		return;
	for (j = 0; j < p->fn->np; j++)
		if (!p->fn->p[j].uniform)
			mask |= 1ULL << j;
	/* a name the expression brought in itself (no declaration to say
	 * either way): the reading that makes sense, said */
	while (p->autop && (s = sg_reads(e, mask)) != 0) {
		p->fn->p[s - 1].uniform = 1;
		mask &= ~(1ULL << (s - 1));
		note("%s is read across the lanes by %s: taken as shared by "
		    "the lanes (uniform)", sym_str(p->fn->p[s - 1].name), what);
	}
	if ((s = sg_reads(e, mask)) != 0)
		fail("%s reads %s in every lane, and %s is a value per lane (not "
		    "`uniform`): declare it %s: %s uniform, or write the lanes' "
		    "values as a function of lane (a bit or a field of a "
		    "shared word, a table's entry)", what,
		    sym_str(p->fn->p[s - 1].name), sym_str(p->fn->p[s - 1].name),
		    sym_str(p->fn->p[s - 1].name), p->fn->p[s - 1].ty ?
		    ty_str(p->fn->p[s - 1].ty) : "u32");
}

/*
 * sg_active(m): m, a uniform parameter, is the lanes that run (bit l for
 * lane l) for the function's subgroup operations, all of them after it;
 * its value is m's low W bits, the ballot of true.
 */
static nid
sg_active_decl(struct P *p, nid m, int W)
{
	ty_t bt = TY(K_UINT, W > 32 ? 64 : 32);
	nid x = m;

	while (N.op[x] == O_CAST)
		x = N.a[x];
	if (!p->fn || N.op[x] != O_VAR || N.ki[x] < 0 || N.ki[x] >= p->fn->np ||
	    !p->fn->p[N.ki[x]].uniform || !TY_ISINT(N.ty[x]))
		fail("sg_active takes a uniform parameter of the function, the "
		    "mask of the lanes that run: %s is not one", node_str(m));
	if (p->sgops)
		fail("sg_active(%s) comes before the subgroup operations it "
		    "masks", node_str(m));
	x = mk_cast(bt, x);
	if (p->sgmask && p->sgmask != x)
		fail("sg_active: one mask a function (%s, then %s)",
		    node_str(p->sgmask), node_str(x));
	p->sgmask = x;
	return W == 64 ? x : mk_bin(O_AND, x, mk_const_i(bt,
	    (int64_t)((1ULL << W) - 1)));
}

/* a shuffle under sg_active: undefined when the lane it reads is not
 * active */
static nid
sg_masked_shuffle(struct P *p, const struct sgfn *g, nid v, nid *args,
    nid here, int W)
{
	nid w = mk_const_i(TY_LIT, W), i;
	ty_t u = TY(K_UINT, 32);

	switch (g->kind) {
	case SGF_SHUFXOR:
		i = mk_bin(O_XOR, here, args[1]);
		break;
	case SGF_BCAST1:
		i = mk_const_i(u, 0);
		break;
	case SGF_ROT:
		i = mk_bin(O_MOD, mk_bin(O_ADD, mk_cast(u, here), mk_cast(u,
		    args[1])), mk_const_i(u, W));
		break;
	default:
		i = args[1];
	}
	v = mk_sg(O_SGSHUF, v, i, w);
	return mk_sel(sg_active_bit(p->sgmask, mk_cast(u, i)), v,
	    mk_undef(N.ty[v]));
}

static nid
sg_apply(struct P *p, const struct sgfn *g, nid *args, int n)
{
	int j = sg_lane_param(p), W = 0;
	int64_t lo, hi;
	nid w, v, here = 0;

	if (n != g->arity && !(g->op == O_SGSCAN && g->kind == SGF_OP && n == 2))
		fail("%s takes %d argument%s", g->name, g->arity,
		    g->arity == 1 ? "" : "s");
	if (j >= 0 && TY_ISINT(p->fn->p[j].ty)) {
		param_bounds(&p->fn->p[j], &lo, &hi);
		if (lo == 0 && hi >= 1 && hi < SG_MAXW)
			W = (int)hi + 1;
	}
	if (!W)
		W = (int)model_active()->par[MP_WARP];
	if (W < 2 || W > SG_MAXW)
		fail("%s needs the subgroup's size: give the function a "
		    "parameter lane: u32 in 0..N-1, or use a model that states "
		    "`warp: N` (2 to %d)", g->name, SG_MAXW);
	w = mk_const_i(TY_LIT, W);
	/* the value per lane: the parameter is the operation's own lane */
	v = args[0];
	if (j >= 0) {
		nid map[MAXP] = {0};
		map[j] = mk_cast(p->fn->p[j].ty, sg_lane());
		v = subst(v, map, j + 1);
		if (g->kind == SGF_SEG)
			args[1] = subst(args[1], map, j + 1);
		here = param_var(p, j);
	} else if (p->sg_depth)
		here = sg_lane();
	/*
	 * A declared parameter that is not uniform has a value in each lane;
	 * read across the lanes it is that lane's value, which has no exact
	 * value here (sg_lane_copy): cost prices it at the worst, and
	 * evaluating it is refused.  A name the expression brought in itself
	 * is taken as shared (sg_shared_only).
	 */
	if (p->fn && !p->autop) {
		nid map[MAXP] = {0};
		int k, any = 0;
		for (k = 0; k < p->fn->np; k++)
			if (k != j && !p->fn->p[k].uniform) {
				map[k] = sg_lane_copy(k, p->fn->p[k].name,
				    p->fn->p[k].ty, W);
				any = 1;
			}
		if (any) {
			v = subst(v, map, p->fn->np);
			if (g->kind == SGF_SEG)
				args[1] = subst(args[1], map, p->fn->np);
		}
	}
	/* a loop's state that follows a value per lane: the same */
	if (p->fn && !p->autop) {
		uint64_t st = (node_slots(v) | (g->kind == SGF_SEG ?
		    node_slots(args[1]) : 0)) & ~((1ULL << MAXP) - 1);
		p->sgread |= st;
		if (st & p->lanest) {
			nid smap[64] = {0};
			int s;
			for (s = MAXP; s < 64; s++)
				if ((st & p->lanest) >> s & 1)
					smap[s] = sg_state_copy(s, p->lane_name[s],
					    p->lane_ty[s], W, p->lane_from[s]);
			v = subst(v, smap, 64);
			if (g->kind == SGF_SEG)
				args[1] = subst(args[1], smap, 64);
		}
	}
	if (g->kind == SGF_ACTIVE)
		return sg_active_decl(p, args[0], W);
	if (!here && (g->op == O_SGSCAN || g->kind == SGF_SHUFXOR ||
	    g->kind == SGF_ROT || g->kind == SGF_SORT))
		fail("%s is a value per lane: give the function a parameter "
		    "lane: u32 in 0..%d", g->name, W - 1);
	p->sgops++;
	if (p->sgmask) {
		/* the inactive lanes: no bit in a ballot, nothing to a vote, a
		 * sum or a scan, and undefined to a shuffle that reads one */
		nid act = sg_active_bit(p->sgmask, sg_lane());
		if (g->kind == SGF_SEG || g->kind == SGF_SORT)
			fail("%s under sg_active(%s): not defined for inactive "
			    "lanes yet", g->name, node_str(p->sgmask));
		if (g->op == O_SGBALLOT || g->op == O_SGANY)
			v = mk_bin(O_LAND, v, act);
		else if (g->op == O_SGALL)
			v = mk_bin(O_LOR, v, mk_un(O_LNOT, act));
		else if (g->op == O_SGADD || g->op == O_SGSCAN)
			v = mk_sel(act, v, mk_const_i(N.ty[v], 0));
		else
			return sg_masked_shuffle(p, g, v, args, here, W);
	}
	switch (g->kind) {
	case SGF_EXCL:
		v = mk_sg(O_SGSCAN, v, here, w);
		return mk_bin(O_SUB, v, mk_cast(N.ty[v], args[0]));
	case SGF_SEG: {
		/*
		 * the sum of v over the lanes from the last head at or below
		 * this one (lane 0 when none is) to this one: the scan here
		 * less the exclusive scan in that lane, read by a shuffle;
		 * that lane is the highest set bit of the heads' ballot, lane
		 * 0's bit added, at or below this lane
		 */
		ty_t bt = TY(K_UINT, W > 32 ? 64 : 32), u = TY(K_UINT, 32);
		nid hb = mk_sg(O_SGBALLOT, args[1], w, 0), m, start, q;
		m = mk_bin(O_SUB, mk_bin(O_SHL, mk_const_i(bt, 2), mk_cast(bt,
		    here)), mk_const_i(bt, 1));
		start = mk_bin(O_SUB, mk_const_i(u, W > 32 ? 63 : 31),
		    mk_cast(u, mk_un(O_CLZ, mk_bin(O_AND, mk_bin(O_OR, hb,
		    mk_const_i(bt, 1)), m))));
		q = mk_sg(O_SGSCAN, v, sg_lane(), w);
		q = mk_bin(O_SUB, q, mk_cast(N.ty[q], v));
		v = mk_sg(O_SGSCAN, v, here, w);
		return mk_bin(O_SUB, v, mk_sg(O_SGSHUF, q, start, w));
	}
	case SGF_SORT:
		/* in lane l the l-th smallest of v over the lanes */
		if (!TY_ISINT(N.ty[v]) || N.ty[v] == TY_BOOL)
			fail("sg_sort sorts integers: v is %s", ty_str(N.ty[v]));
		if (TYK(N.ty[v]) == K_LIT)
			v = mk_cast(TY(K_INT, 32), v);
		if ((v = sg_sorted(v, here, 0, W)) == 0)
			fail("sg_sort: the bitonic network needs a power-of-two "
			    "subgroup, and this one has %d lanes", W);
		return v;
	case SGF_SHUFXOR:
		return mk_sg(O_SGSHUF, v, mk_bin(O_XOR, here, args[1]), w);
	case SGF_BCAST:
		return mk_sg(O_SGSHUF, v, args[1], w);
	case SGF_BCAST1:
		return mk_sg(O_SGSHUF, v, mk_const_i(TY_LIT, 0), w);
	case SGF_ROT:
		return mk_sg(O_SGSHUF, v, mk_bin(O_MOD, mk_bin(O_ADD,
		    mk_cast(TY(K_UINT, 32), here), mk_cast(TY(K_UINT, 32),
		    args[1])), mk_const_i(TY(K_UINT, 32), W)), w);
	}
	if (g->op == O_SGSHUF)
		return mk_sg(O_SGSHUF, v, args[1], w);
	if (g->op == O_SGSCAN)
		return mk_sg(O_SGSCAN, v, n == 2 ? args[1] : here, w);
	return mk_sg(g->op, v, w, 0);
}

static nid
sg_call(struct P *p, const struct sgfn *g, nid *args, int n)
{
	nid r = sg_apply(p, g, args, n);

	sg_shared_only(p, r, g->name);
	if (!p->sg_depth && sg_has_lane(r))
		fail("%s: lane is a value per lane; outside the operand it has "
		    "to be a parameter, f(..., lane: u32 in 0..N-1)", g->name);
	return r;
}

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
	if (IS_DOT(b->op))
		return mk_dot(b->op, args[0], args[1], b->arity == 3 ? args[2] : 0);
	if (b->op == O_BFI) {
		/* four operands: the field's place and width ride in one */
		if (!is_const(args[2]) || !is_const(args[3]) ||
		    !TY_ISINT(N.ty[args[2]]) || !TY_ISINT(N.ty[args[3]]))
			fail("bfi takes a constant offset and count: %s and %s",
			    node_str(args[2]), node_str(args[3]));
		if (N.ki[args[2]] < 0 || N.ki[args[3]] < 0 ||
		    N.ki[args[2]] + N.ki[args[3]] > 32)
			fail("bfi: bits %lld..%lld are not a field of a 32-bit "
			    "word", (long long)N.ki[args[2]],
			    (long long)(N.ki[args[2]] + N.ki[args[3]] - 1));
		return mk_inst(O_BFI, args[0], args[1], bfi_field((int)
		    N.ki[args[2]], (int)N.ki[args[3]]));
	}
	if (IS_ASTC(b->op))
		return astc_call(b->op, args);
	if (IS_INST(b->op))
		return mk_inst(b->op, args[0], b->arity >= 2 ? args[1] : 0,
		    b->arity == 3 ? args[2] : 0);
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
	const struct sgfn *sf;
	struct func *g;
	nid args[MAXP], r;
	ty_t ct;
	int na, kind;

	if ((r = value_name(p, s, n)) != 0) {
		adv(p);
		return r;
	}
	/* the lane of a subgroup operation's operand, when the function
	 * has no parameter of that name */
	if (p->sg_depth && n == 4 && !memcmp(s, "lane", 4)) {
		adv(p);
		return sg_lane();
	}
	if ((sf = sg_builtin(p, at)) != NULL && !func_find(name)) {
		adv(p);
		p->sg_depth++;
		na = parse_args(p, args, 3);
		p->sg_depth--;
		return sg_call(p, sf, args, na);
	}
	if ((r = calc_ident(p)) != 0)
		return r;
	/* type conversion: u8(x), f32(x), f16(x), real(n) */
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
	/* the texture unit's reads (texalu.h) */
	if (pk2(p) == T_LP && (kind = tex_lookup(s, n, &na)) != 0) {
		int bits = na, want = kind == O_FETCH_BORDER ? 3 :
		    kind == O_TEX_BILINEAR ? 4 : 2;
		adv(p);
		if (parse_args(p, args, 4) != want)
			fail("%s takes %s", opinfo[kind].name, kind ==
			    O_TEX_BILINEAR ? "(table, width, x, y)" : kind ==
			    O_FETCH_BORDER ? "(table, index, border value)" :
			    IS_FETCH(kind) ? "(table, index)" : "(table, x)");
		if (kind == O_TEX_BILINEAR) {
			if (!is_const(args[1]) || !TY_ISINT(N.ty[args[1]]) ||
			    N.ki[args[1]] < 1 || N.ki[args[1]] > 0xffffff)
				fail("tex_bilinear's width is a constant from 1 to "
				    "16777215, not %s", node_str(args[1]));
			return mk_tex(kind, TEX_KI(bits, N.ki[args[1]]), args[0],
			    args[2], args[3]);
		}
		return mk_tex(kind, TEX_KI(bits, 0), args[0], args[1],
		    kind == O_FETCH_BORDER ? args[2] : 0);
	}
	if ((b = builtin(p, at)) != NULL) {
		adv(p);
		if (pk(p) == T_LP) {
			na = parse_args(p, args, b->arity > 3 ? b->arity : 3);
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

/* the W of slot's lane copies in e (sg_lane_copy), or 0 */
static int
copy_width(nid e, int slot, uint8_t *seen)
{
	uint32_t j;
	int w;

	if (e == 0 || seen[e] || !(N.fl[e] & NF_LANEREAD))
		return 0;
	seen[e] = 1;
	if (N.op[e] == O_VAR)
		return N.ki[e] == SG_COPY_SLOT + slot ? (int)N.b[e] : 0;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			if ((w = copy_width(N.lst[N.a[e] + j], slot, seen)) != 0)
				return w;
		return 0;
	}
	if ((w = copy_width(N.a[e], slot, seen)) != 0 ||
	    (w = copy_width(N.b[e], slot, seen)) != 0)
		return w;
	return copy_width(N.c[e], slot, seen);
}

/*
 * An argument as the other lanes pass it: the caller's parameters that
 * are not uniform become their own lane copies; what the lanes share
 * (constants, uniform parameters) is the same in every lane.
 */
static nid
arg_in_lanes(struct P *p, nid a, int W)
{
	nid map[MAXP] = {0};
	int k, any = 0;

	if (!p->fn || p->autop)
		return a;
	for (k = 0; k < p->fn->np; k++)
		if (!p->fn->p[k].uniform && !sg_is_lane(param_var(p, k))) {
			map[k] = sg_lane_copy(k, p->fn->p[k].name,
			    p->fn->p[k].ty, W);
			any = 1;
		}
	return any ? subst(a, map, p->fn->np) : a;
}

static nid
call_user(struct P *p, struct func *g, nid *args, int na)
{
	nid map[SG_COPY_SLOT + MAXP] = {0};
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
	/* g's parameters read in the other lanes: the arguments there */
	if (N.fl[g->body] & NF_LANEREAD)
		for (j = 0; j < na; j++) {
			uint8_t *seen = xcalloc(N.n, 1);
			int w = copy_width(g->body, j, seen);
			free(seen);
			if (w)
				map[SG_COPY_SLOT + j] = arg_in_lanes(p, map[j],
				    w);
		}
	{
		nid r = subst(g->body, map, (N.fl[g->body] & NF_LANEREAD) ?
		    SG_COPY_SLOT + MAXP : na);
		sg_shared_only(p, r, sym_str(g->name));
		return r;
	}
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
					fail("unknown type '%.*s' (real, f16, f32, "
					    "f64, int, bool, uN, iN)", (int)n, s);
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
			p->bval[p->nb] = parse_expr(p, 0);
			name_hint(p->bval[p->nb], p->bname[p->nb]);
			p->nb++;
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
	p->sgmask = 0;
	p->sgops = 0;
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
	p->sgmask = 0;
	p->sgops = 0;
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
    "basis avx512 : x86bmi { ternlog(x, y, z) = lut }\n"
    "basis nand { nand(x, y) = ~(x & y) }\n"
    /*
     * NVIDIA GT 1030 (GP108, sm_61) through SPIR-V, measured on the card
     * (driver 580.178.04) by tests/gt1030: issue cost per SPIR-V
     * operation relative to fadd, from throughput with the SMs full
     * (tests/gt1030/results/2026-11-11.md).  Integer division is
     * emulated (by a constant: multiply-high and shifts); fdiv, sqrt and
     * the SFU functions are 1/4 rate; fp64 is 1/32.  load, store and
     * atomic_global are cache-resident accesses (DRAM-bound streams cost
     * 90-140); branch is a taken uniform branch with its reconvergence,
     * diverge what a split warp adds ("keep-up").  The costs are on a 1/8
     * grid (exact binary fractions, within 1% of the measurements).  f16
     * is not measured: GP108 has no f16 arithmetic.
     *
     * Dot products: DP4A (OpUDot/OpSDot on 4 x 8-bit packed lanes) is
     * one instruction, measured as udot4x8 + iadd = 2.14 and sdot4x8 +
     * iadd = 2.00 (x_dp4a alone 1.14): dot4x8 is 1.  sudot4x8 is the
     * same instruction (IDP4A.S8.U8) and was not measured separately.
     * udot4x8sat measured 6.47: DP4A has no saturating form, the driver
     * emulates it.  No 2 x 16 or 8 x 4 form: those are not listed, so
     * they are unavailable here (sm_61's DP2A multiplies 16-bit lanes by
     * 8-bit ones, which SPIR-V cannot say).
     */
    "model gt1030 {\n"
    "  iadd: 1, logic: 1, shift: 2, imul: 3, idiv: 28.5, idivk: 8.75,\n"
    "  cmp: 2.375, select: 1.625, minmax: 1, bitcnt: 3, conv: 3, bitcast: 0,\n"
    "  fadd: 1, fmul: 1, fdiv: 3, fsqrt: 4, ftrans: 4, fpow: 8, f64: 32.375,\n"
    "  load: 6.875, branch: 6.25, diverge: 14.375, barrier: 9.125,\n"
    "  atomic_shared: 6.625, atomic_global: 50, store: 15,\n"
    "  dot4x8: 1, dot4x8sat: 6.5\n"
    /*
     * Subgroups (a warp: 32 lanes), from the same run.  A shuffle is
     * 4.00 (sg_shuffle; by xor, broadcast, rotate and the quad swap 4.00
     * to 4.07 with the add their step has), the sum over the lanes 32.63
     * and the inclusive scan 60.24 (each less the step's add).  The
     * ballot and the votes fit below zero: ballot((a & 1) != 0) + b is
     * 3.65 and any((a & 1) != 0) ? b : c 4.51 where the and, the compare
     * and the add or select alone are 4.375 and 5 (the compare's bool is
     * never made), so they are priced 0 and their compare as a compare.
     */
    "  sg_ballot: 0, sg_vote: 0, sg_add: 32.625, sg_shuffle: 4,\n"
    "  sg_scan: 60.25\n"
    /*
     * Signed division and the remainders (udiv, umod, sdiv and by 7, the
     * mean of the two runs, less the add each step has); a signed
     * remainder is not measured: sdiv plus what a remainder adds.  The
     * fusions the per-class prices miss, from the tests that measure
     * them: lop3_select and lop3_xor3 (one LOP3: 2.00 with the add),
     * imad and imul_const (the add folds into the XMAD chain: 3.0),
     * shl_const_add, shr_const_add, imul_pow2 and udiv_pow2 (2.0: a
     * shift by a constant and an add cost one shift), fmad_contract
     * (FFMA: 1.0; fmad_precise, NoContraction, 2.0), fneg_add and
     * fabs_add (source modifiers: 1.0).
     */
    "  sdiv: 36, sdivk: 21, irem: 32.25, iremk: 13.5\n"
    "  fuse lop3 = lut 3 : logic\n"
    "  fuse imad(a: int, b: int, c: int) = a * b + c : imul\n"
    "  fuse shl_add(a: int, k: const, b: int) = (a << k) + b : shift\n"
    "  fuse shr_add(a: int, k: const, b: int) = (a >> k) + b : shift\n"
    "  fuse ffma(a: float, b: float, c: float) = a * b + c : fmul contract\n"
    "  fuse ffms(a: float, b: float, c: float) = a * b - c : fmul contract\n"
    "  fuse ffmr(a: float, b: float, c: float) = c - a * b : fmul contract\n"
    "  fuse fneg(a: float) = -a : 0 in fadd fmul\n"
    "  fuse fabs(a: float) = abs(a) : 0 in fadd fmul\n"
    /*
     * The memory kinds, from the same run: workgroup memory is 32 banks
     * of 4 bytes, each address after the first on a bank costs
     * bank_conflict (32 on one bank: 127); a block read is the constant
     * cache (uniform address) or serialised (a different address in each
     * lane); streams are DRAM-bound loads and stores (16 MiB); a shared
     * float add is a compare-and-swap loop; a sampled read costs less
     * than a fetch.
     */
    "  load_shared: 3, store_shared: 3, bank_conflict: 4,\n"
    "  banks: 32, bank_bytes: 4, warp: 32,\n"
    "  load_ubo: 4.125, load_ubo_div: 148.125,\n"
    "  load_stream: 80.625, store_stream: 111,\n"
    "  atomic_shared_same: 63.75, atomic_shared_f: 24.25,\n"
    "  atomic_global_f: 138.75, atomic_global_64: 264.625,\n"
    "  atomic_global_cas: 143.5, atomic_global_stream: 137.5,\n"
    "  tex_fetch: 43.125, tex_sample: 21.625, tex_sample_f32: 54.75,\n"
    "  image_load: 31.875\n"
    /*
     * The texture unit (texalu.c): the two filtered reads above are an
     * explicit-LOD bilinear read of an rgba8 texture (x_tex_bilinear,
     * 21.65 and 21.31) and of an rgba32f one (54.80).  A nearest sampled
     * read was not measured, so tex_nearest is not listed, and neither
     * is tex_within (how far the filter's result is from the exact
     * weighted sum): nothing filtered is offered until a test measures
     * it.  subtexel is the driver's subTexelPrecisionBits.
     */
    "  subtexel: 8\n"
    /* measured: the f32 -> f16 -> f32 round trip costs nothing and does
     * not round (tests/gt1030/results/2026-11-11.md, f16 round trip) */
    "  folds f32 -> f16 -> f32\n"
    "}\n"
    /*
     * Later NVIDIA parts: ESTIMATES, not measurements.  The classes the
     * CUDA C++ Programming Guide's table of arithmetic throughput
     * (results per clock per SM) covers are its ratios to fp32 add; the
     * rest (division emulation, select, memory, branches, barriers,
     * atomics) are gt1030's measurements, carried over unchanged.
     *
     * sm75 (Turing): fp32 64/clk; int add, logic, shift, compare and
     * IMAD 64 (so imul 1: a 32-bit multiply is one IMAD from Volta on);
     * popc/clz and conversions 16; SFU 16; fp64 2.  IDP4A issues on the
     * integer pipe at the IMAD rate (64): dot4x8 1.  No saturating
     * IDP4A: dot4x8sat as on sm_61 (the same emulation, at the same
     * int:fp32 rate of 1:1).  cmp: the table's ratio plus the 0.375
     * gt1030 measured above its own (the comparison's bool result).
     */
    /* (`folds none` in these three: the f16 round trip is a measurement
     * of GP108's driver, where f16 has no arithmetic, and is not carried
     * over to parts that were not measured) */
    "model sm75 : gt1030 {\n"
    "  iadd: 1, logic: 1, shift: 1, imul: 1, cmp: 1.375, minmax: 1,\n"
    "  bitcnt: 4, conv: 4, fdiv: 3, fsqrt: 4, ftrans: 4, f64: 32,\n"
    "  dot4x8: 1, dot4x8sat: 6.5\n"
    "  folds none\n"
    "}\n"
    /*
     * sm86 (Ampere GA10x): fp32 128/clk (two FP32 paths), the integer
     * pipe 64: every integer class costs twice an fadd; SFU 16; fp64 2.
     * dot4x8 2 (IDP4A at the integer rate), dot4x8sat 13 (the sm_61
     * emulation at half the fp32 rate).  Estimates, as above.
     */
    "model sm86 : gt1030 {\n"
    "  iadd: 2, logic: 2, shift: 2, imul: 2, cmp: 2.75, minmax: 2,\n"
    "  bitcnt: 8, conv: 8, fdiv: 6, fsqrt: 8, ftrans: 8, f64: 64,\n"
    "  dot4x8: 2, dot4x8sat: 13\n"
    "  folds none\n"
    "}\n"
    /*
     * rdna2 (AMD gfx10.3): ESTIMATES from AMD's RDNA2 ISA guide and
     * published rates, not measurements; memory, branches, barriers and
     * atomics are gt1030's, carried over.  32-bit integer multiply
     * (v_mul_lo_u32) and the transcendental unit are quarter rate, fp64
     * 1/16.  The dot instructions are full rate and have a clamp bit, so
     * the saturating forms cost the same: v_dot4_u32_u8, v_dot4_i32_i8
     * (4 x 8 bits), v_dot8_u32_u4, v_dot8_i32_i4 (8 x 4 bits).  No mixed
     * signedness (RDNA3's v_dot4_i32_iu8) and no 2 x 16-bit integer dot.
     */
    "model rdna2 : gt1030 {\n"
    "  iadd: 1, logic: 1, shift: 1, imul: 4, cmp: 1.375, minmax: 1,\n"
    "  bitcnt: 1, conv: 1, fdiv: 3, fsqrt: 4, ftrans: 4, f64: 16,\n"
    "  dot4x8: none, dot4x8sat: none,\n"
    "  udot4x8: 1, sdot4x8: 1, udot4x8sat: 1, sdot4x8sat: 1,\n"
    "  udot8x4: 1, sdot8x4: 1, udot8x4sat: 1, sdot8x4sat: 1\n"
    "  folds none\n"
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
		g->arity = (uint8_t)tmp.np;
		if (is_kw(p, "lut") && pk2(p) != T_LP) {
			/* any function of the inputs: a truth-table instruction */
			adv(p);
			g->lut = 1;
			g->body = mk_const_i(TY(K_UINT, 1), 0);
		} else {
			p->fn = &tmp;
			p->sgmask = 0;
			p->sgops = 0;
			g->body = mk_cast(TY(K_UINT, 1), parse_expr(p, 0));
			p->fn = NULL;
		}
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
		if (g->lut)
			b->ng += j == b->ng;
		else if (j == b->ng) {
			basis_finish_gate(b, g);
			b->ng++;
		} else
			basis_finish_gate(b, &b->g[j]);
	}
	expect_eos(p);
}

/*
 * In a model: `fuse NAME(a: int, ...) = PATTERN : COST [contract] [in
 * CLASS ...] [on CLASS ...]`, `fuse NAME = lut K : COST`, `nofuse NAME`
 * (src/fuse.h).
 * COST is a sum of numbers, classes and number*class.
 */
static void
parse_fuse(struct P *p, struct model *m)
{
	static const char *const kinds[] = {"int", "float", "const", "any"};
	struct fusion f;
	struct func tmp;
	int off = is_kw(p, "nofuse"), k;
	size_t n;
	const char *s;
	nid v;

	adv(p);
	if (pk(p) != T_ID)
		fail("expected a fusion name");
	memset(&f, 0, sizeof f);
	memset(&tmp, 0, sizeof tmp);
	f.name = tsym(p, p->i);
	adv(p);
	if (off) {
		fuse_remove(m, f.name);
		return;
	}
	if (accept(p, T_LP)) {
		do {
			if (pk(p) != T_ID || tmp.np == FUSE_MAXP)
				fail("a fusion takes 1 to %d named inputs", FUSE_MAXP);
			tmp.p[tmp.np].name = f.pname[tmp.np] = tsym(p, p->i);
			adv(p);
			expect(p, T_COLON);
			for (k = 0; k < 4 && !is_kw(p, kinds[k]); k++)
				;
			if (k == 4 && pk(p) == T_ID) {
				/* uN, sN: an integer of a proved width */
				int bits = 0;
				size_t q;
				s = ttext(p, p->i, &n);
				for (q = 1; q < n && s[q] >= '0' && s[q] <= '9' &&
				    bits <= FUSE_MAXBITS; q++)
					bits = bits * 10 + (s[q] - '0');
				if ((s[0] == 'u' || s[0] == 's') && n > 1 && q == n &&
				    bits >= 1 && bits <= FUSE_MAXBITS) {
					f.pbits[tmp.np] = (uint8_t)bits;
					f.psgn[tmp.np] = s[0] == 's';
					k = FP_INT;
				}
			}
			if (k == 4)
				fail("a fusion's input is int, float, const, any, or an "
				    "integer of a proved width (u1 to u%d, s1 to s%d)",
				    FUSE_MAXBITS, FUSE_MAXBITS);
			adv(p);
			f.pkind[tmp.np] = (uint8_t)k;
			tmp.p[tmp.np].ty = k == FP_FLOAT ? TY_F32 : TY_INT;
			tmp.np++;
		} while (accept(p, T_COMMA));
		expect(p, T_RP);
		f.np = tmp.np;
	}
	expect(p, T_ASSIGN);
	if (is_kw(p, "lut") && pk2(p) == T_NUM) {
		adv(p);
		v = number(p);
		f.lut = (int)N.ki[v];
		if (f.lut < 2 || f.lut > 6 || f.np)
			fail("lut takes 2 to 6 inputs and no named ones");
	} else {
		if (!f.np)
			fail("expected `lut K` or named inputs: fuse %s(a: int, ...)",
			    sym_str(f.name));
		p->fn = &tmp;
		p->sgmask = 0;
		p->sgops = 0;
		f.pat = parse_expr(p, 0);
		p->fn = NULL;
		if (N.op[f.pat] == O_VAR || N.op[f.pat] == O_CONST)
			fail("a fusion's pattern is an operation");
	}
	expect(p, T_COLON);
	do {
		double mul = 1;
		if (pk(p) == T_NUM) {
			v = number(p);
			mul = TY_FP(N.ty[v]) ? N.kr[v] : (double)N.ki[v];
			if (!accept(p, T_STAR)) {
				f.k += (float)mul;
				continue;
			}
		}
		s = ttext(p, p->i, &n);
		if (pk(p) != T_ID || (k = cclass_find(s, n)) < 0)
			fail("unknown cost class '%.*s' (see 'models')", (int)n, s);
		if (f.nt == FUSE_MAXT)
			fail("a fusion's cost has at most %d classes", FUSE_MAXT);
		adv(p);
		f.tcls[f.nt] = (int8_t)k;
		f.tmul[f.nt++] = (float)mul;
	} while (accept(p, T_PLUS));
	for (;;) {
		if (accept_kw(p, "contract"))
			f.contract = 1;
		else if (accept_kw(p, "in")) {
			while (pk(p) == T_ID && (s = ttext(p, p->i, &n)) != NULL &&
			    (k = cclass_find(s, n)) >= 0) {
				if (k >= 64)
					fail("`in %s`: a modifier's users are among "
					    "the first 64 classes", cclass_name[k]);
				f.users |= 1ULL << k;
				adv(p);
			}
			if (!f.users)
				fail("`in` lists the cost classes that absorb it");
			if (opinfo[N.op[f.pat]].arity < 1 || f.lut)
				fail("a modifier is one operation");
		} else if (accept_kw(p, "on")) {
			while (pk(p) == T_ID && (s = ttext(p, p->i, &n)) != NULL &&
			    (k = cclass_find(s, n)) >= 0) {
				if (k >= C_NBASE)
					fail("`on` takes the classes of ordinary operations "
					    "(fadd, fmul, iadd ...), not %s", cclass_name[k]);
				f.makers |= 1ULL << k;
				adv(p);
			}
			if (!f.makers)
				fail("`on` lists the cost classes of the operations it "
				    "folds into");
			if (f.lut)
				fail("a result modifier is a pattern over its input");
		} else
			break;
	}
	if (f.users && f.makers)
		fail("a fusion is a source modifier (in) or a result modifier "
		    "(on), not both");
	fuse_add(m, &f);
}

static void
parse_model(struct P *p)
{
	struct model *m, *parent;
	size_t n;
	const char *s;
	int k, inherited = 0, store_given = 0;
	uint64_t given = 0;
	nid v;

	adv(p);
	if (pk(p) != T_ID)
		fail("expected a model name");
	m = model_new(tsym(p, p->i));
	fuse_reset(m);
	adv(p);
	if (accept(p, T_COLON)) {
		if (pk(p) != T_ID || (parent = model_find(tsym(p, p->i))) == NULL)
			fail("unknown model");
		adv(p);
		memcpy(m->c, parent->c, sizeof m->c);
		memcpy(m->par, parent->par, sizeof m->par);
		fuse_inherit(m, parent);
		memcpy(m->fold, parent->fold, sizeof m->fold);
		m->nfold = parent->nfold;
		memcpy(m->par, parent->par, sizeof m->par);
		inherited = 1;
	}
	skip_nl(p);
	expect(p, T_LC);
	for (;;) {
		while (accept(p, T_NL) || accept(p, T_COMMA))
			;
		if (accept(p, T_RC))
			break;
		if (is_kw(p, "fuse") || is_kw(p, "nofuse")) {
			parse_fuse(p, m);
			continue;
		}
		s = ttext(p, p->i, &n);
		if (pk(p) != T_ID)
			fail("expected a cost class");
		if (n == 5 && !memcmp(s, "folds", 5)) {
			/* folds T -> M -> T: a round trip the driver drops */
			ty_t ty[3];
			int j;
			adv(p);
			accept(p, T_COLON);
			if (accept_kw(p, "none")) {	/* not the parent's either */
				m->nfold = 0;
				continue;
			}
			for (j = 0; j < 3; j++) {
				size_t tn;
				const char *ts = ttext(p, p->i, &tn);
				if (pk(p) != T_ID || !ty_parse(ts, tn, &ty[j]) ||
				    TYK(ty[j]) != K_FLOAT)
					fail("folds: expected a float type (folds f32 "
					    "-> f16 -> f32)");
				adv(p);
				if (j < 2 && !accept(p, T_ARROW))
					fail("folds: expected '->' (folds f32 -> f16 "
					    "-> f32)");
			}
			if (ty[0] != ty[2] || TYW(ty[1]) >= TYW(ty[0]))
				fail("folds: a round trip T -> M -> T through a "
				    "narrower M, not %s -> %s -> %s", ty_str(ty[0]),
				    ty_str(ty[1]), ty_str(ty[2]));
			for (j = 0; j < m->nfold; j++)
				if (m->fold[j][0] == ty[0] && m->fold[j][1] == ty[1])
					break;
			if (j == m->nfold) {
				if (m->nfold == MAXFOLD)
					fail("folds: at most %d per model", MAXFOLD);
				m->fold[j][0] = ty[0];
				m->fold[j][1] = ty[1];
				m->nfold++;
			}
			continue;
		}
		k = (n == 3 && !memcmp(s, "all", 3)) ? -1 : cclass_find(s, n);
		if (k == -2) {
			/* parameters, not prices: banks, bank_bytes, warp (memk.h);
			 * subtexel, tex_within (texalu.h); `none` takes one back */
			int q;
			for (q = 0; q < MP_NPAR; q++)
				if (strlen(mparam_name[q]) == n && !memcmp(s,
				    mparam_name[q], n))
					break;
			if (q < MP_NPAR) {
				adv(p);
				if (!accept(p, T_COLON))
					expect(p, T_ASSIGN);
				if (is_kw(p, "none")) {
					adv(p);
					m->par[q] = 0;
					continue;
				}
				v = number(p);
				m->par[q] = TY_FP(N.ty[v]) ? (float)N.kr[v] :
				    (float)N.ki[v];
				if (q == MP_SUBTEXEL) {
					if (m->par[q] < 1 || m->par[q] > TEX_MAXBITS ||
					    m->par[q] != (int)m->par[q])
						fail("subtexel: a whole number of bits from 1 "
						    "to %d, not %g", TEX_MAXBITS,
						    (double)m->par[q]);
				} else if (q == MP_TEX_WITHIN) {
					if (!(m->par[q] > 0 && m->par[q] <= 1e6f))
						fail("tex_within: more than 0 and at most 1e6 "
						    "ulps, not %g (none: not stated)",
						    (double)m->par[q]);
				} else if (!(m->par[q] >= 1) || m->par[q] > 4096)
					fail("%s: a count from 1 to 4096, not %g",
					    mparam_name[q], (double)m->par[q]);
				continue;
			}
		}
		if (k == -2)
			fail("unknown cost class '%.*s' (see 'models')", (int)n, s);
		adv(p);
		if (!accept(p, T_COLON))
			expect(p, T_ASSIGN);
		if ((k <= -100 || k >= C_DOT_FIRST) && is_kw(p, "none")) {
			/* a dot form (or a subgroup operation) taken away from
			 * an inherited model */
			int j, j0 = k <= -100 ? -k - 100 : k;
			adv(p);
			/* (a memory kind: back to its fallback's price) */
			for (j = j0; j < j0 + (k <= -100 ? 3 : 1); j++)
				m->c[j] = C_ABSENT(j) ? INFINITY : NAN;
			continue;
		}
		v = number(p);
		if (k == -1) {
			/* every class the model has anyway: dot forms are
			 * listed one by one */
			for (k = 0; k < C_NBASE; k++)
				m->c[k] = TY_FP(N.ty[v]) ? (float)N.kr[v] :
				    (float)N.ki[v];
		} else if (k <= -100) {
			int j;
			for (j = -k - 100; j < -k - 100 + 3; j++)
				m->c[j] = TY_FP(N.ty[v]) ? (float)N.kr[v] :
				    (float)N.ki[v];
		} else {
			m->c[k] = TY_FP(N.ty[v]) ? (float)N.kr[v] :
			    (float)N.ki[v];
			if (k == C_STORE)
				store_given = 1;
			if (k < C_NBASE)
				if (k < 64)	/* the classes it asks about are */
					given |= 1ULL << k;
		}
	}
	/* a model that does not price stores: as dear as its loads */
	if (!inherited && !store_given)
		m->c[C_STORE] = m->c[C_LOAD];
	/*
	 * nor signed division or remainders: as its divisions; a signed
	 * remainder not given is the signed division plus what a remainder
	 * adds to the unsigned one
	 */
	if (!inherited) {
		for (k = 0; k < 2; k++) {
			if (!(given >> (C_SDIV + k) & 1))
				m->c[C_SDIV + k] = m->c[C_IDIV + k];
			if (!(given >> (C_IREM + k) & 1))
				m->c[C_IREM + k] = m->c[C_IDIV + k];
			if (!(given >> (C_SREM + k) & 1))
				m->c[C_SREM + k] = m->c[C_SDIV + k] +
				    m->c[C_IREM + k] - m->c[C_IDIV + k];
		}
	}
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
"  model name [: parent] { class: cost, ..., fuse f(a: int, ...) = expr : cost }\n"
"types: real (math), f16 f32 f64 (IEEE), int (math), uN iN (wrapping), bool\n"
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
"  plot F, G ... [x in a..b] [y in c..d] [deriv] [tangent a] [logx|logy]\n"
"    [contour|heatmap|wireframe|3d] [rotate] [animate t] [slices|parallel]\n"
"    [polar] [field] [fps F] [frames N] [to NAME]; plot3d F; neighbourhood F\n"
"    [at a]\n"
"loops and calculus\n"
"  sum(i, lo, hi, e) | prod(...) | x[0] = c; x[k+1] = ... | { for/while/if }\n"
"  closed F | scheme F [more] [model m] | continuous F\n"
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
"algebra\n"
"  expand|factor|simplify|together|cancel F | collect|apart F [in x]\n"
"  gcd|lcm P, Q | resultant P, Q [in x] | discriminant P [in x]\n"
"  quo|rem P, Q [in x] | divide P by Q | substitute x = a in F\n"
"  solve L = R [, L2 = R2] [for x, y]\n"
"properties\n"
"  monotonic|period|periodic|parity|symmetry|bounded|injective|surjective|\n"
"    bijective|continuity|discontinuities|convexity|asymptotes|zeros F\n"
"    [x] [in a..b] | identify EXPR | identify L = R\n"
"complex, quaternion, octonion (z, w complex; i j k; e1..e31)\n"
"  holomorphic F | residue F at a | poles F | integrate F around |z| = r\n"
"  table quaternion|octonion|sedenion | algebra NAME [property]\n"
"  zerodivisors | associator a, b, c | commutator a, b\n"
"  rotate V by Q | rotate V by ANGLE about AXIS | axisangle | rotmatrix |\n"
"    frommatrix | slerp\n"
"rings and finite fields (Z/n, mod n, GF(p^n [, aes|0x11b|f]))\n"
"  ring R [EXPR|units|zerodivisors|ideals|field|table|minpoly A|log A]\n"
"  inverse A mod N | order A mod N | phi N | carmichael N | factorint N\n"
"  crt | solve ... mod m | legendre | jacobi | sqrtmod | primroot N |\n"
"    dlog A base G mod N | magic D [uN|iN] | gf2matrix EXPR [uN]\n"
"  factor|irreducible|primitive|roots F over GF(q) | det|rank|nullspace|\n"
"    inverse M mod p\n"
"integrals and solids\n"
"  area F [and G] [from a to b] [riemann [N] [left|right|mid|trapezoid|\n"
"    simpson]] [accumulate] | revolve F [and G] [from a to b]\n"
"    [about x|y|y = c|x = c] [disk|washer|shell] [still] [rotate]\n"
"  volume | surfacearea | arclength F from a to b | centroid F [and G]\n"
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

/*
 * The source text from the next token to the next blank, the tokens
 * inside it skipped: a path ("work/a.png") as it was written
 */
int
cmdp_raw(struct P *p, char *buf, size_t n)
{
	const char *src = p->t->src;
	uint32_t o, e;

	if (at_eos(p))
		return 0;
	o = p->t->off[p->i];
	for (e = o; src[e] && !isspace((unsigned char)src[e]); e++)
		;
	snprintf(buf, n, "%.*s", (int)(e - o), src + o);
	while (!at_eos(p) && p->t->off[p->i] < e)
		adv(p);
	return 1;
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

/*
 * plot.c: a command's function argument given as text (a piece of a
 * plot's list), read as cmd_func reads one; errors unwind to the caller
 */
struct func *
cmd_func_text(const char *src)
{
	struct toks *const t = xcalloc(1, sizeof *t);
	struct P *const q = xcalloc(1, sizeof *q);
	struct toks *dt = diag_toks;
	uint32_t di = diag_tok;
	jmp_buf saved;
	struct func *f;

	memcpy(saved, fail_jmp, sizeof saved);
	q->t = t;
	diag_toks = t;
	if (setjmp(fail_jmp)) {
		memcpy(fail_jmp, saved, sizeof saved);
		diag_toks = dt;
		diag_tok = di;
		toks_free(t);
		free(t);
		free(q);
		longjmp(fail_jmp, 1);
	}
	lex(t, src);
	f = cmd_func(q);
	if (!at_eos(q))
		fail("unexpected '%.*s' in %s", (int)t->len[q->i],
		    t->src + t->off[q->i], src);
	memcpy(fail_jmp, saved, sizeof saved);
	diag_toks = dt;
	diag_tok = di;
	toks_free(t);
	free(t);
	free(q);
	return f;
}

/*
 * hyper.c: a statement with complex, quaternion or octonion numbers (i,
 * z, e1, q: quat ...) runs there, as text; 1 when it did.
 */
static int
hyper_hook(struct P *p)
{
	static char buf[8192];
	uint32_t a = p->i, b = a;
	int depth = 0;
	size_t len;

	while (b < p->t->n && p->t->kind[b] != T_EOF) {
		if (p->t->kind[b] == T_LC)
			depth++;
		if (p->t->kind[b] == T_RC)
			depth--;
		if (p->t->kind[b] == T_NL && depth <= 0) {
			if (b + 1 < p->t->n && tok_is(p->t, b + 1, "where"))
				return 0;
			break;
		}
		b++;
	}
	if (b == a)
		return 0;
	len = p->t->off[b - 1] + p->t->len[b - 1] - p->t->off[a];
	if (len >= sizeof buf)
		return 0;
	memcpy(buf, p->t->src + p->t->off[a], len);
	buf[len] = 0;
	if (!hyper_statement(buf))
		return 0;
	p->i = b;
	return 1;
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
	/* a line naming a ring (GF(2^8, aes) 0x57 * 0x83) is ring.c's, even
	 * when hyper.c would read it as a value */
	/* texalu F: reads through the texture unit (texalu.h) */
	if (pk(p) == T_ID && texalu_command(p))
		return;
	/* texenc T: a table as BC4 blocks (texenc.h) */
	if (pk(p) == T_ID && texenc_command(p))
		return;
	if (pk(p) == T_ID && ring_command(p))
		return;
	if (pk(p) == T_ID && prove_command(p))	/* theorem, prove ... (prove.h) */
		return;
	if (pk(p) == T_ID && ecff_command(p))	/* elliptic curves over F_q */
		return;
	/* elliptic curves, Diophantine equations (ec_cmd.c): before
	 * hyper.c tries the line ("ec y^2 = x^3 - x" as e*c ...) */
	if (pk(p) == T_ID && ec_command(p))
		return;
	if (pk(p) == T_ID && absalg_command(p))	/* snf, abgroup ... */
		return;
	if (pk(p) == T_ID && homalg_command(p))	/* homology ... (homalg) */
		return;
	/* model zen4, mca, intrinsic ... (x86.c): before hyper.c reads
	 * "mca f" as a product */
	if (x86_command(p))
		return;
	if (hyper_hook(p))
		return;
	if (pk(p) == T_ID && (pk2(p) == T_ID || pk2(p) == T_NUM ||
	    pk2(p) == T_NL || pk2(p) == T_EOF || pk2(p) == T_LP ||
	    pk2(p) == T_BAR || pk2(p) == T_MINUS || pk2(p) == T_SQRT ||
	    pk2(p) == T_LB || pk2(p) == T_LC)) {
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
		if (ring_command(p))    /* Z/n, GF(q), ... (ring.c) */
			return;
		/* transforms, series and sums (xform.c, series.c) */
		if (xform_command(p))
			return;
		if (alg_command(p))	/* expand, factor, solve ... (alg.c) */
			return;
		if (calc2_command(p))
			return;
		if (rop_command(p))	/* fixed function as compute (rop.c) */
			return;
		/* simplify a SPIR-V block (spirv.c), from "SAT simplify" ... */
		if (is_kw(p, "spvsimplify")) {
			double budget = 6, ulp = 0;
			int nofile = 0, fmode = 0, fapprox = 0, fpercomp = 0, fonly = 0;
			int fsearch = 0;
			uint64_t fseed = stoch_seed();
			adv(p);
			spirv_fn_opts(0);
			while (!at_eos(p)) {
				if (accept_kw(p, "budget") && pk(p) == T_NUM) {
					/* one number: "budget 60 nofile" is not 60 * nofile */
					nid nn = number(p);
					budget = TY_FP(N.ty[nn]) ? N.kr[nn] : (double)N.ki[nn];
				}
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
				} else if (accept_kw(p, "fassume") && pk(p) == T_ID) {
					/* fassume NAME LO HI LOOPEN HIOPEN (float simplify) */
					char nm[48];
					double v[4];
					int i, neg;
					cmdp_word(p, nm, sizeof nm);
					for (i = 0; i < 4; i++) {
						neg = accept(p, T_MINUS);
						if (pk(p) != T_NUM)
							fail("simplify: fassume NAME LO HI LOOPEN HIOPEN");
						{
							nid nn = number(p);
							v[i] = TY_FP(N.ty[nn]) ? N.kr[nn] :
							    (double)N.ki[nn];
						}
						v[i] = neg ? -v[i] : v[i];
					}
					spirv_assume_float(nm, v[0], v[1], v[2] != 0, v[3] != 0);
				} else if (accept_kw(p, "functions"))
					spirv_fn_opts(1);	/* the function level alone */
				else if (accept_kw(p, "nofunctions"))
					spirv_fn_opts(2);
				else if (accept_kw(p, "float"))
					fonly = 1;
				else if (accept_kw(p, "exact"))
					fmode = 1;
				else if (accept_kw(p, "ulp") && pk(p) == T_NUM) {
					nid nn = number(p);
					ulp = TY_FP(N.ty[nn]) ? N.kr[nn] : (double)N.ki[nn];
					fmode = 2;
				} else if (accept_kw(p, "approx"))
					fapprox = 1;
				else if (accept_kw(p, "search")) {
					/* the polynomials' coefficient search (fsimp.c) */
					char sw[24];
					if (!cmdp_word(p, sw, sizeof sw))
						sw[0] = 0;
					if (!strcmp(sw, "anneal") || !strcmp(sw, "annealing"))
						fsearch = NS_ANNEAL;
					else if (!strcmp(sw, "cmaes") || !strcmp(sw, "cma"))
						fsearch = NS_CMAES;
					else
						fail("simplify: search is anneal or cmaes (the float "
						    "polynomials' coefficient search), not '%s'", sw);
				} else if (accept_kw(p, "seed")) {
					nid nn;
					double v;
					if (pk(p) != T_NUM)
						fail("simplify: seed K, a whole number");
					nn = number(p);
					v = TY_FP(N.ty[nn]) ? N.kr[nn] : (double)N.ki[nn];
					if (!(v >= 0) || v > 9e15 || v != floor(v))
						fail("simplify: seed is a whole number from 0, not %g",
						    v);
					fseed = (uint64_t)v;
				} else if (accept_kw(p, "percomp"))
					fpercomp = 1;
				else if (accept_kw(p, "foldproof"))
					spirv_dword_opts(1);	/* folded round trips kept */
				else if (accept_kw(p, "advice"))
					spirv_advice_opt(1);
				else if (accept_kw(p, "sink")) {
					char fm[48];
					if (!cmdp_word(p, fm, sizeof fm) ||
					    !spirv_sink_opt(fm))
						fail("simplify: sink FORMAT is unormN or snormN (2 "
						    "to 23 bits), f16, or a format's name with fewer "
						    "bits than f32 (rgba8, rgba8_snorm, rgb10a2, "
						    "rgba16f, r8g8b8a8_unorm, b8g8r8a8_srgb): '%s' is "
						    "none of them", fm);
				}
				else
					adv(p);
			}
			spirv_float_opts(fmode, ulp, fapprox, fpercomp, fonly, fsearch,
			    fseed);
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
		/* plot (plot.c) */
		if ((is_kw(p, "plot") || is_kw(p, "plot3d") ||
		    is_kw(p, "neighbourhood") || is_kw(p, "neighborhood")) &&
		    !func_find(tsym(p, p->i))) {
			int mode = is_kw(p, "plot") ? 0 : is_kw(p, "plot3d") ? 1 : 2;
			adv(p);
			cmd_plot(p, mode);
			return;
		}
		/* area, revolve, volume, arclength ... (viz.c) */
		if (viz_command(p))
			return;
		/* animate (anim.c) */
		if (is_kw(p, "animate") && !func_find(tsym(p, p->i))) {
			adv(p);
			cmd_animate(p);
			return;
		}
		/* monotonic, period, identify, zeros ... (props.c) */
		if (props_command(p))
			return;
		/* dword, foldproof: fp64 as f32 pairs, driver folds (dword.h) */
		if (dword_command(p) || fold_command(p))
			return;
		/* banks, intatomic: the memory side of a model (memk.c) */
		if (memk_command(p))
			return;
		/* stoke, pit, phash: stochastic search (stoch.c) */
		if (stoch_command(p))
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
			if (pk(p) != T_ID || (m = model_find(tsym(p, p->i))) == NULL) {
				char nm[640];
				size_t n = 0;
				const char *s = pk(p) == T_ID ? ttext(p, p->i, &n) : "";
				model_names(nm, sizeof nm);
				fail("unknown model %s%.*s%s (the models: %s)", n ? "'" :
				    "", (int)n, s, n ? "'" : "", nm);
			}
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
			p->sgmask = 0;
			p->sgops = 0;
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
				int want = 5, maxc = 0, stoke;
				uint32_t fnm2 = f->name;
				if (pk(p) == T_NUM)
					want = (int)N.ki[number(p)];
				maxc = opt_int(p, "max", 0);
				stoke = accept_kw(p, "stoke");	/* then the MCMC search */
				expect_eos(p);
				notes_flush();
				cmd_find(f, want, NULL, maxc);
				if (stoke)
					stoke_find(func_find(fnm2));
			} else if (!strcmp(cmd, "cost")) {
				int cons = -1;
				m = model_active();
				for (;;) {
					if (accept_kw(p, "model")) {
						if (pk(p) != T_ID ||
						    (m = model_find(tsym(p, p->i))) == NULL)
							fail("unknown model");
						adv(p);
					} else if (accept_kw(p, "consecutive")) {
						/* an index that runs along the lanes of a warp */
						if (pk(p) != T_ID)
							fail("cost: consecutive NAME, a parameter");
						for (cons = 0; cons < f->np; cons++)
							if (f->p[cons].name == tsym(p, p->i))
								break;
						if (cons == f->np || !TY_ISINT(f->p[cons].ty))
							fail("cost: consecutive takes an integer "
							    "parameter of the function");
						adv(p);
					} else
						break;
				}
				expect_eos(p);
				notes_flush();
				diverge_consecutive(cons);
				cmd_cost(f, m);
				diverge_consecutive(-1);
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
		p->sgmask = 0;
		p->sgops = 0;
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
/*
 * cfront's commands take file paths and C, which the ucas lexer does not
 * read: a line starting "c optimize", "c lower" or "c targets" runs on
 * its text (cf_cmd.c), the rest around it as usual.
 */
static int
c_line(const char *p, size_t n)
{
	return n > 2 && p[0] == 'c' && p[1] == ' ' && (!strncmp(p + 2, "optimi",
	    6) || !strncmp(p + 2, "lower", 5) || !strncmp(p + 2, "targets", 7));
}

static void run_source0(const char *src, const char *file, int echo);

void
run_source(const char *src, const char *file, int echo)
{
	const char *p = src, *seg = src;

	while (*p) {
		const char *e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		if (c_line(p, (size_t)(e - p))) {
			jmp_buf saved;
			char *line;
			if (p > seg) {
				char *pre = xmalloc((size_t)(p - seg) + 1);
				memcpy(pre, seg, (size_t)(p - seg));
				pre[p - seg] = 0;
				run_source0(pre, file, echo);
				free(pre);
			}
			line = xmalloc((size_t)(e - p) + 1);
			memcpy(line, p, (size_t)(e - p));
			line[e - p] = 0;
			if (echo)
				printf("> %s\n", line);
			memcpy(saved, fail_jmp, sizeof saved);
			if (setjmp(fail_jmp) == 0)
				cfront_line(line + 2);
			else
				rep_failed();
			memcpy(fail_jmp, saved, sizeof saved);
			notes_flush();
			fflush(stdout);
			free(line);
			seg = *e ? e + 1 : e;
		}
		p = *e ? e + 1 : e;
	}
	if (seg == src)
		run_source0(src, file, echo);
	else if (*seg)
		run_source0(seg, file, echo);
}

/* the prelude (definitions, gt1030, the x86 models), once; a C line
 * that comes first (cfront_line) needs the models too */
void
ucas_prelude(void)
{
	static int prelude_done;

	nodes_init();
	if (!prelude_done) {
		prelude_done = 1;
		run_source(prelude, "<prelude>", 0);
		model_set_active(model_find(sym_cstr("gt1030")));
		x86_prelude();	/* the x86 µarchs as cost models (x86data.c) */
	}
}

static void
run_source0(const char *src, const char *file, int echo)
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
	/* both live across setjmp, so they are reached through pointers */
	struct toks *const t = xcalloc(1, sizeof *t);
	struct P *const p = xcalloc(1, sizeof *p);

	ucas_prelude();
	diag_file = file;
	diag_toks = t;
	p->t = t;
	if (setjmp(fail_jmp)) {   /* lexer error */
		rep_failed();
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
			rep_failed();
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
#include "alg_syntax.inc"
