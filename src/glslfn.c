/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * glslfn.c - simplify's function-level pass over a shader's GLSL source.
 *
 * The instruction-level rewrites in spirv.c see one SPIR-V instruction and
 * what feeds it.  This pass reads the GLSL text itself (OpSource, which
 * glslangValidator -g keeps) and works on whole functions:
 *
 *   1. parse: src/snxglslc (ScarletNX's GLSL ES compiler) reads the
 *      source, brought to GLSL ES 3.10 where Vulkan GLSL only spells
 *      things otherwise (es_text below); its typed syntax tree becomes
 *      the small tree here (cv_x, cv_s), with each node's source line;
 *      integer scalars, arrays and constant tables are understood, the
 *      rest (floats, vectors, structs) is opaque;
 *   2. infer each function's domain from its call sites: an abstract
 *      interpretation from main() (or every function nobody calls) whose
 *      state is a set of environments, one per value of the "key"
 *      variables (globals only ever assigned constants or selects of
 *      constants, such as sh_ib = mode == 6u ? 4u : 2u, and bool or
 *      constant-select locals), each variable an interval with the bits
 *      that may be one and an optional symbolic upper bound (t <= dd after
 *      clamp(x, 0, dd)).  A function with a loop is run once per value
 *      of the keys it reads and of its single-valued arguments (its
 *      summary), one without on each call's own values, so v < 2^n stays
 *      exact in Widen(v, n); each summary's entry is the domain reported
 *      and proved over.  An expression whose leaves repeat (a + (b - a) / 16) is
 *      evaluated over every value of its leaves when they are few, and one
 *      monotone in a leaf with a symbolic bound is evaluated at the bound;
 *   3. search: masks, min/max/clamp, selects, compares and branches that
 *      never bite; constant-bound loops whose first or last iterations
 *      never change the result; an interpolation ((2^S - w) a + w b + C)
 *      >> S with one multiply; a function of one argument split on a
 *      low-cardinality global or parameter and fitted (x K + C) >> S per
 *      case (K as a function of the case when one fits all); calls in a
 *      loop whose arguments and read set the loop never changes;
 *   4. prove: each candidate becomes a ucas function (the translation
 *      below: a code block, reads of globals as parameters, the inferred
 *      domain as ranges and a where clause) and equiv_check() compares it
 *      with the original exhaustively; nothing is reported on sampling.
 *
 * The suggestions are GLSL text for the user; the module is not changed.
 */
#include "texalu.h"

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ucas.h"
#include "calc.h"
#include "fuse.h"
#include "irw.h"
#include "snxglslc/internal.h"

/* ---- the syntax tree ---- */

enum { TV, TB, TI, TU, TF, TX };	/* void bool int uint float other */
enum { X_NUM, X_VAR, X_IDX, X_MEM, X_CALL, X_UN, X_BIN, X_SEL, X_ASG,
    X_INC, X_COMMA };
enum { OP_NONE, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_SHL, OP_SHR,
    OP_AND, OP_OR, OP_XOR, OP_LT, OP_LE, OP_GT, OP_GE, OP_EQ, OP_NE,
    OP_LAND, OP_LOR, OP_LXOR, OP_NEG, OP_NOT, OP_LNOT, OP_PLUS };
/* builtins: X_CALL with fn < 0 */
enum { B_MIN = 1, B_MAX, B_CLAMP, B_ABS, B_INT, B_UINT, B_BOOL, B_BFE, B_UNK };

struct gx {
	uint8_t k, op, ty, post;
	int a, b, c;		/* operands; X_CALL: args at xarg[a], b of them */
	int64_t v;		/* X_NUM */
	int sym;		/* X_VAR: the symbol; X_CALL: function (>= 0) or -B_* */
	int line;
};
enum { S_BLOCK, S_DECL, S_EXPR, S_IF, S_FOR, S_WHILE, S_DO, S_RET, S_BRK,
    S_CONT, S_EMPTY, S_UNK };
struct gs {
	uint8_t k, scoped;
	int a, b, c, d;		/* see stmt() */
	int sym;		/* S_DECL */
	int first, n;		/* S_BLOCK: children at slist[first..] */
	int line;
};
enum { Y_GLOBAL, Y_CONST, Y_BUF, Y_PARAM, Y_LOCAL, Y_RET };
struct gsym {
	char name[40];
	uint8_t kind, ty, cnst, key, out, flt;
	int arr;		/* elements, 0 for a scalar */
	int fn;			/* owner function (params, locals), -1 */
	int init;		/* initializer expression, -1 */
	int64_t *tab;		/* a const array's values */
	int nasg;		/* assignments outside the declaration */
	int line;
};
struct gfn {
	char name[40];
	uint8_t ty, unsup, pure, analysed, onstack;
	int np, p[MAXP + 8];
	int body, line, endline;
	int ret;		/* the return value's symbol */
	int ncaller;
	uint8_t *rd, *wr;	/* globals read and written, transitively */
};

static struct {
	struct gx *x;
	int nx, capx;
	struct gs *s;
	int ns, caps;
	int *xarg, nxarg, capxarg;
	int *slist, nslist, capslist;
	struct gsym *y;
	int ny, capy;
	struct gfn *f;
	int nf, capf;
	int cur;		/* the function being converted */
	int bad;		/* the parse gave up */
	char why[1200];
} G;


static int
nx_new(int k, int ty, int line)
{
	if (G.nx == G.capx) {
		G.capx = G.capx ? G.capx * 2 : 4096;
		G.x = xrealloc(G.x, (size_t)G.capx * sizeof *G.x);
	}
	memset(&G.x[G.nx], 0, sizeof G.x[0]);
	G.x[G.nx].k = (uint8_t)k;
	G.x[G.nx].ty = (uint8_t)ty;
	G.x[G.nx].a = G.x[G.nx].b = G.x[G.nx].c = -1;
	G.x[G.nx].sym = -1;
	G.x[G.nx].line = line;
	return G.nx++;
}

static int
ns_new(int k, int line)
{
	if (G.ns == G.caps) {
		G.caps = G.caps ? G.caps * 2 : 2048;
		G.s = xrealloc(G.s, (size_t)G.caps * sizeof *G.s);
	}
	memset(&G.s[G.ns], 0, sizeof G.s[0]);
	G.s[G.ns].k = (uint8_t)k;
	G.s[G.ns].a = G.s[G.ns].b = G.s[G.ns].c = G.s[G.ns].d = -1;
	G.s[G.ns].sym = -1;
	G.s[G.ns].line = line;
	return G.ns++;
}

static int
sym_new(const char *name, int kind, int ty, int line)
{
	if (G.ny == G.capy) {
		G.capy = G.capy ? G.capy * 2 : 512;
		G.y = xrealloc(G.y, (size_t)G.capy * sizeof *G.y);
	}
	memset(&G.y[G.ny], 0, sizeof G.y[0]);
	snprintf(G.y[G.ny].name, sizeof G.y[0].name, "%s", name);
	G.y[G.ny].kind = (uint8_t)kind;
	G.y[G.ny].ty = (uint8_t)ty;
	G.y[G.ny].fn = kind == Y_PARAM || kind == Y_LOCAL || kind == Y_RET ?
	    G.cur : -1;
	G.y[G.ny].init = -1;
	G.y[G.ny].line = line;
	return G.ny++;
}

static int
fn_find(const char *name)
{
	int i;
	for (i = 0; i < G.nf; i++)
		if (!strcmp(G.f[i].name, name))
			return i;
	return -1;
}

/* ---- helpers of the tree ---- */

static int
bin_ty(int op, int ta, int tb)
{
	if (op >= OP_LT && op <= OP_LXOR)
		return TB;
	if (op == OP_SHL || op == OP_SHR)
		return ta;
	if (ta == TX || tb == TX)
		return TX;
	if (ta == TF || tb == TF)
		return TF;
	if (ta == TU || tb == TU)
		return TU;	/* GLSL's implicit int -> uint */
	return ta;
}

static const struct {
	const char *s;
	int op, prec;
} binops[] = {
	{"||", OP_LOR, 1}, {"^^", OP_LXOR, 2}, {"&&", OP_LAND, 3},
	{"|", OP_OR, 4}, {"^", OP_XOR, 5}, {"&", OP_AND, 6},
	{"==", OP_EQ, 7}, {"!=", OP_NE, 7}, {"<", OP_LT, 8}, {">", OP_GT, 8},
	{"<=", OP_LE, 8}, {">=", OP_GE, 8}, {"<<", OP_SHL, 9},
	{">>", OP_SHR, 9}, {"+", OP_ADD, 10}, {"-", OP_SUB, 10},
	{"*", OP_MUL, 11}, {"/", OP_DIV, 11}, {"%", OP_MOD, 11}, {NULL, 0, 0}
};

static int
op_prec(int op)
{
	int i;
	for (i = 0; binops[i].s; i++)
		if (binops[i].op == op)
			return binops[i].prec;
	return 0;
}

static const char *
op_str(int op)
{
	int i;
	for (i = 0; binops[i].s; i++)
		if (binops[i].op == op)
			return binops[i].s;
	return op == OP_NEG ? "-" : op == OP_NOT ? "~" : op == OP_LNOT ? "!" :
	    "?";
}


static int
block_of(int *kids, int n, int line, int scoped)
{
	int s = ns_new(S_BLOCK, line), i;
	G.s[s].first = G.nslist;
	G.s[s].n = n;
	G.s[s].scoped = (uint8_t)scoped;
	for (i = 0; i < n; i++) {
		if (G.nslist == G.capslist) {
			G.capslist = G.capslist ? G.capslist * 2 : 2048;
			G.slist = xrealloc(G.slist, (size_t)G.capslist *
			    sizeof *G.slist);
		}
		G.slist[G.nslist++] = kids[i];
	}
	return s;
}


/* ---- the source through ScarletNX's GLSL ES parser (src/snxglslc) ---- */

/*
 * snxglslc takes GLSL ES 1.00 to 3.20; a Vulkan shader is GLSL 4.50.  The
 * text is brought to GLSL ES 3.10 where the two differ only in spelling,
 * keeping every line where it was: the #version line; layout(push_constant)
 * (a uniform block then); "T a[N] = { ... }" (T[N](...)); #extension lines;
 * and const on a local, which GLSL ES allows only for constant expressions
 * (the word is blanked, and remembered: ES_CONST below).  What it still
 * refuses is dropped one top-level declaration or function at a time
 * (blanked), and the report names what was dropped and why.
 */
static char *es_orig;		/* the text before local consts were blanked */
static char dropped[1024];

static void
blank(char *s, size_t a, size_t b)
{
	size_t i;
	for (i = a; i < b; i++)
		if (s[i] != '\n')
			s[i] = ' ';
}

static int
word_at(const char *s, size_t i, const char *w)
{
	size_t n = strlen(w);
	return !strncmp(s + i, w, n) && (i == 0 || !(isalnum((unsigned char)
	    s[i - 1]) || s[i - 1] == '_')) && !(isalnum((unsigned char)s[i + n]) ||
	    s[i + n] == '_');
}

/* comments and the inside of quotes skipped: the next code position */
static size_t
skip_comment(const char *s, size_t i)
{
	if (s[i] == '/' && s[i + 1] == '/') {
		while (s[i] && s[i] != '\n')
			i++;
		return i;
	}
	if (s[i] == '/' && s[i + 1] == '*') {
		i += 2;
		while (s[i] && !(s[i] == '*' && s[i + 1] == '/'))
			i++;
		return s[i] ? i + 2 : i;
	}
	return i;
}

/*
 * The subgroup built-ins (GL_KHR_shader_subgroup_*), which snxglslc does
 * not have: each one the text calls is declared, ahead of the code, as a
 * function of its overloads with a stub body (sg_stub_from on), and a
 * call to one is a value the function level does not know, as any
 * built-in's but min, max, clamp and abs.  T is every type of the family:
 * a arithmetic (float, int, uint), b bitwise (int, uint, bool), g all.
 */
static const struct sgbi {
	const char *name, *sig;	/* sig: result, then the parameters */
	char fam;
} sg_builtins[] = {
	{"subgroupBarrier", "void", 0}, {"subgroupMemoryBarrier", "void", 0},
	{"subgroupMemoryBarrierBuffer", "void", 0},
	{"subgroupMemoryBarrierShared", "void", 0},
	{"subgroupMemoryBarrierImage", "void", 0},
	{"subgroupElect", "bool", 0},
	{"subgroupAll", "bool bool", 0}, {"subgroupAny", "bool bool", 0},
	{"subgroupAllEqual", "bool T", 'g'},
	{"subgroupBallot", "uvec4 bool", 0},
	{"subgroupInverseBallot", "bool uvec4", 0},
	{"subgroupBallotBitExtract", "bool uvec4 uint", 0},
	{"subgroupBallotBitCount", "uint uvec4", 0},
	{"subgroupBallotInclusiveBitCount", "uint uvec4", 0},
	{"subgroupBallotExclusiveBitCount", "uint uvec4", 0},
	{"subgroupBallotFindLSB", "uint uvec4", 0},
	{"subgroupBallotFindMSB", "uint uvec4", 0},
	{"subgroupAdd", "T T", 'a'}, {"subgroupMul", "T T", 'a'},
	{"subgroupMin", "T T", 'a'}, {"subgroupMax", "T T", 'a'},
	{"subgroupAnd", "T T", 'b'}, {"subgroupOr", "T T", 'b'},
	{"subgroupXor", "T T", 'b'},
	{"subgroupInclusiveAdd", "T T", 'a'}, {"subgroupInclusiveMul", "T T", 'a'},
	{"subgroupInclusiveMin", "T T", 'a'}, {"subgroupInclusiveMax", "T T", 'a'},
	{"subgroupInclusiveAnd", "T T", 'b'}, {"subgroupInclusiveOr", "T T", 'b'},
	{"subgroupInclusiveXor", "T T", 'b'},
	{"subgroupExclusiveAdd", "T T", 'a'}, {"subgroupExclusiveMul", "T T", 'a'},
	{"subgroupExclusiveMin", "T T", 'a'}, {"subgroupExclusiveMax", "T T", 'a'},
	{"subgroupExclusiveAnd", "T T", 'b'}, {"subgroupExclusiveOr", "T T", 'b'},
	{"subgroupExclusiveXor", "T T", 'b'},
	{"subgroupClusteredAdd", "T T uint", 'a'},
	{"subgroupClusteredMul", "T T uint", 'a'},
	{"subgroupClusteredMin", "T T uint", 'a'},
	{"subgroupClusteredMax", "T T uint", 'a'},
	{"subgroupClusteredAnd", "T T uint", 'b'},
	{"subgroupClusteredOr", "T T uint", 'b'},
	{"subgroupClusteredXor", "T T uint", 'b'},
	{"subgroupBroadcast", "T T uint", 'g'},
	{"subgroupBroadcastFirst", "T T", 'g'},
	{"subgroupShuffle", "T T uint", 'g'}, {"subgroupShuffleXor", "T T uint", 'g'},
	{"subgroupShuffleUp", "T T uint", 'g'},
	{"subgroupShuffleDown", "T T uint", 'g'},
	{"subgroupRotate", "T T uint", 'g'},
	{"subgroupClusteredRotate", "T T uint uint", 'g'},
	{"subgroupQuadBroadcast", "T T uint", 'g'},
	{"subgroupQuadSwapHorizontal", "T T", 'g'},
	{"subgroupQuadSwapVertical", "T T", 'g'},
	{"subgroupQuadSwapDiagonal", "T T", 'g'},
};

/* its built-in variables, and Vulkan's that GLSL ES 3.10 lacks, read as
 * values the function level does not know: renamed snx... (the same
 * length) and declared uniform */
static const char *const sg_vars[] = {"gl_SubgroupSize",
    "gl_SubgroupInvocationID", "gl_NumSubgroups", "gl_SubgroupID",
    "gl_SubgroupEqMask", "gl_SubgroupGeMask", "gl_SubgroupGtMask",
    "gl_SubgroupLeMask", "gl_SubgroupLtMask", "gl_VertexIndex",
    "gl_InstanceIndex", "gl_BaseVertex", "gl_BaseInstance", "gl_DrawID",
    "gl_SampleID", "gl_ViewIndex", "gl_Layer", "gl_ViewportIndex"};

static const char *
sg_var_type(const char *name)
{
	return strstr(name, "Mask") ? "uvec4" : strstr(name, "Subgroup") ?
	    "uint" : "int";
}

/*
 * Types of GL_EXT_shader_8bit_storage and 16bit_storage, which snxglslc
 * does not have.  As a type, each is the 32-bit one it widens to (a
 * buffer's member read is a value the level does not know either way);
 * as a constructor, which truncates, a function of the stubs below, so
 * its value is one the level does not know.
 */
static const struct nty {
	const char *name, *as, *ctor;
} narrow_types[] = {
	{"uint8_t", "uint", "ucas_u8"}, {"int8_t", "int", "ucas_i8"},
	{"uint16_t", "uint", "ucas_u16"}, {"int16_t", "int", "ucas_i16"},
	{"float16_t", "float", "ucas_f16"},
	{"u8vec2", "uvec2", "ucas_u8v2"}, {"u8vec3", "uvec3", "ucas_u8v3"},
	{"u8vec4", "uvec4", "ucas_u8v4"}, {"i8vec2", "ivec2", "ucas_i8v2"},
	{"i8vec3", "ivec3", "ucas_i8v3"}, {"i8vec4", "ivec4", "ucas_i8v4"},
	{"u16vec2", "uvec2", "ucas_u16v2"}, {"u16vec3", "uvec3", "ucas_u16v3"},
	{"u16vec4", "uvec4", "ucas_u16v4"}, {"i16vec2", "ivec2", "ucas_i16v2"},
	{"i16vec3", "ivec3", "ucas_i16v3"}, {"i16vec4", "ivec4", "ucas_i16v4"},
	{"f16vec2", "vec2", "ucas_f16v2"}, {"f16vec3", "vec3", "ucas_f16v3"},
	{"f16vec4", "vec4", "ucas_f16v4"},
};
#define NNARROW (sizeof narrow_types / sizeof narrow_types[0])

/* layout qualifiers of Vulkan's GLSL that GLSL ES 3.10 has not, and the
 * function level has no use for: taken out of their layout(...) */
static const char *const vk_layout[] = {"offset", "constant_id", "align",
    "input_attachment_index", "component", "index", "xfb_buffer",
    "xfb_offset", "xfb_stride", "depth_any", "depth_greater", "depth_less",
    "depth_unchanged", "origin_upper_left", "pixel_center_integer", NULL};

static size_t sg_stub_from, sg_stub_to;	/* the declarations' bytes */

/* one overload of b's function, T being t */
static void
sg_overload(struct sbuf *o, const struct sgbi *b, const char *t)
{
	char w[8][16];
	int n = 0, k;
	const char *s = b->sig;

	while (*s && n < 8) {
		size_t l = strcspn(s, " ");
		snprintf(w[n], sizeof w[n], "%.*s", (int)l, s);
		if (!strcmp(w[n], "T"))
			snprintf(w[n], sizeof w[n], "%s", t);
		n++;
		s += l + (s[l] == ' ');
	}
	sb_printf(o, "%s %s(", w[0], b->name);
	for (k = 1; k < n; k++)
		sb_printf(o, "%shighp %s p%d", k > 1 ? ", " : "", w[k], k);
	if (!strcmp(w[0], "void"))
		sb_puts(o, ") { } ");
	else if (!strcmp(w[0], "bool") || w[0][0] == 'b')
		sb_printf(o, ") { return %s(false); } ", w[0]);
	else
		sb_printf(o, ") { return %s(0); } ", w[0]);
}

/* the declarations the text needs ahead of its code: the default
 * precisions GLSL ES lacks (a fragment shader's float, an image's or a
 * 3-D sampler's anywhere; Vulkan's GLSL has highp), the subgroup
 * built-ins it calls and the variables it reads */
static void
es_prelude(struct sbuf *o, const char *src, int stage)
{
	static const char *const fam_a[] = {"float", "vec2", "vec3", "vec4",
	    "int", "ivec2", "ivec3", "ivec4", "uint", "uvec2", "uvec3", "uvec4",
	    NULL};
	static const char *const fam_b[] = {"int", "ivec2", "ivec3", "ivec4",
	    "uint", "uvec2", "uvec3", "uvec4", "bool", "bvec2", "bvec3", "bvec4",
	    NULL};
	static const char *const fam_g[] = {"float", "vec2", "vec3", "vec4",
	    "int", "ivec2", "ivec3", "ivec4", "uint", "uvec2", "uvec3", "uvec4",
	    "bool", "bvec2", "bvec3", "bvec4", NULL};
	static const char *const opaque[] = {"sampler2D", "sampler3D",
	    "samplerCube", "sampler2DShadow", "samplerCubeShadow",
	    "sampler2DArray", "sampler2DArrayShadow", "isampler2D", "isampler3D",
	    "isamplerCube", "isampler2DArray", "usampler2D", "usampler3D",
	    "usamplerCube", "usampler2DArray", "sampler2DMS", "isampler2DMS",
	    "usampler2DMS", "image2D", "iimage2D", "uimage2D", "image3D",
	    "iimage3D", "uimage3D", "imageCube", "iimageCube", "uimageCube",
	    "image2DArray", "iimage2DArray", "uimage2DArray", NULL};
	size_t i, k;

	if (stage != SNX_GLSLC_COMPUTE)
		sb_puts(o, "precision highp float; precision highp int; ");
	for (k = 0; opaque[k]; k++)
		for (i = 0; src[i]; i++)
			if (word_at(src, i, opaque[k])) {
				sb_printf(o, "precision highp %s; ", opaque[k]);
				break;
			}
	for (k = 0; k < sizeof sg_vars / sizeof sg_vars[0]; k++)
		for (i = 0; src[i]; i++)
			if (word_at(src, i, sg_vars[k])) {
				/* a vertex shader writes gl_Layer and gl_ViewportIndex */
				sb_printf(o, "%shighp %s snx%s; ", stage ==
				    SNX_GLSLC_VERTEX && (!strcmp(sg_vars[k], "gl_Layer") ||
				    !strcmp(sg_vars[k], "gl_ViewportIndex")) ? "" : "uniform ",
				    sg_var_type(sg_vars[k]), sg_vars[k] + 3);
				break;
			}
	/* the narrow types' constructors: one argument of any number type */
	for (k = 0; k < NNARROW; k++) {
		static const char *const from[] = {"int", "uint", "float", "ivec2",
		    "ivec3", "ivec4", "uvec2", "uvec3", "uvec4", "vec2", "vec3",
		    "vec4", NULL};
		const struct nty *t = &narrow_types[k];
		const char *const *f;
		size_t nl = strlen(t->name);
		char last = t->as[strlen(t->as) - 1];
		for (i = 0; src[i]; i++)
			if (word_at(src, i, t->name)) {
				size_t e = i + nl;
				while (src[e] == ' ' || src[e] == '\t')
					e++;
				if (src[e] == '(')
					break;
			}
		if (!src[i])
			continue;
		for (f = from; *f; f++) {
			char fl = (*f)[strlen(*f) - 1];
			/* a scalar from a scalar, a vector from one as wide */
			if ((isdigit((unsigned char)last) || isdigit((unsigned char)fl)) &&
			    last != fl)
				continue;
			sb_printf(o, "%s %s(highp %s p1) { return %s(0); } ", t->as,
			    t->ctor, *f, t->as);
		}
	}
	/* GL_EXT_integer_dot_product: every pairing of signed and unsigned */
	{
		static const char *const dn[] = {"dotPacked4x8EXT",
		    "dotPacked4x8AccSatEXT", "dotEXT", "dotAccSatEXT"};
		static const char *const sg[] = {"int", "uint"};
		for (k = 0; k < 4; k++) {
			int a, b2, w;
			for (i = 0; src[i] && !word_at(src, i, dn[k]); i++)
				;
			if (!src[i])
				continue;
			for (w = k < 2 ? 1 : 2; w <= (k < 2 ? 1 : 4); w++)
				for (a = 0; a < 2; a++)
					for (b2 = 0; b2 < 2; b2++) {
						const char *r = sg[a & b2];
						char ta[8], tb[8];
						if (w == 1) {
							snprintf(ta, sizeof ta, "%s", sg[a]);
							snprintf(tb, sizeof tb, "%s", sg[b2]);
						} else {
							snprintf(ta, sizeof ta, "%cvec%d", sg[a][0], w);
							snprintf(tb, sizeof tb, "%cvec%d", sg[b2][0], w);
						}
						sb_printf(o, "%s %s(highp %s p1, highp %s p2%s%s%s) "
						    "{ return %s(0); } ", r, dn[k], ta, tb,
						    k & 1 ? ", highp " : "", k & 1 ? r : "",
						    k & 1 ? " p3" : "", r);
					}
		}
	}
	for (k = 0; k < sizeof sg_builtins / sizeof sg_builtins[0]; k++) {
		const struct sgbi *b = &sg_builtins[k];
		const char *const *f;
		for (i = 0; src[i] && !word_at(src, i, b->name); i++)
			;
		if (!src[i])
			continue;
		if (!b->fam) {
			sg_overload(o, b, "");
			continue;
		}
		f = b->fam == 'a' ? fam_a : b->fam == 'b' ? fam_b : fam_g;
		for (; *f; f++)
			sg_overload(o, b, *f);
	}
}

static char *
es_text(const char *src, int stage)
{
	struct sbuf b = {0};
	size_t i, n = strlen(src);
	int pre = 1;	/* the prelude: on the first line of code */
	int ifs = 0;	/* #if depth: the prelude goes outside any */
	int spec = 0;	/* after layout(constant_id): 1 to its const, 2 to its = */

	sg_stub_from = sg_stub_to = 0;
	sb_puts(&b, "");
	for (i = 0; i < n;) {
		size_t j = skip_comment(src, i);
		if (j != i) {
			sb_printf(&b, "%.*s", (int)(j - i), src + i);
			i = j;
			continue;
		}
		if ((i == 0 || src[i - 1] == '\n') && !strncmp(src + i, "#version",
		    8)) {
			size_t e = strcspn(src + i, "\n");
			if (memmem(src + i, e, " es", 3) == NULL)
				sb_puts(&b, "#version 320 es");
			else
				sb_printf(&b, "%.*s", (int)e, src + i);
			i += e;
			continue;
		}
		if ((i == 0 || src[i - 1] == '\n') && !strncmp(src + i, "#line", 5)) {
			/* the file's name goes (GL_GOOGLE_cpp_style_line_directive) */
			int q = 0;
			while (i < n && src[i] != '\n') {
				q |= src[i] == '"';
				sb_putc(&b, q ? ' ' : src[i]);
				i++;
			}
			continue;
		}
		if ((i == 0 || src[i - 1] == '\n') && (!strncmp(src + i,
		    "#extension", 10) || !strncmp(src + i, "#pragma", 7))) {
			while (i < n && src[i] != '\n') {
				sb_putc(&b, ' ');
				i++;
			}
			continue;
		}
		if (pre && (i == 0 || src[i - 1] == '\n') && src[i] == '#') {
			size_t k = i + 1;
			while (src[k] == ' ' || src[k] == '\t')
				k++;
			if (!strncmp(src + k, "if", 2))	/* #if, #ifdef, #ifndef */
				ifs++;
			else if (!strncmp(src + k, "endif", 5) && ifs > 0)
				ifs--;
		} else if (pre && (i == 0 || src[i - 1] == '\n') && !ifs &&
		    !(i >= 2 && src[i - 2] == '\\')) {
			sg_stub_from = b.n;
			es_prelude(&b, src, stage);
			sg_stub_to = b.n;
			pre = 0;
		}
		if (src[i] == 'g' && !strncmp(src + i, "gl_", 3)) {
			size_t k;
			for (k = 0; k < sizeof sg_vars / sizeof sg_vars[0]; k++)
				if (word_at(src, i, sg_vars[k])) {
					sb_printf(&b, "snx%s", sg_vars[k] + 3);
					i += strlen(sg_vars[k]);
					break;
				}
			if (k < sizeof sg_vars / sizeof sg_vars[0])
				continue;
		}
		/* a specialization constant: a uniform, its default left out (the
		 * value is the pipeline's, not the one written) */
		if (spec == 1 && word_at(src, i, "const")) {
			sb_puts(&b, "uniform");
			i += 5;
			spec = 2;
			continue;
		}
		if (spec && src[i] == ';')
			spec = 0;
		if (spec == 2 && src[i] == '=') {
			while (i < n && src[i] != ';') {
				sb_putc(&b, src[i] == '\n' ? '\n' : ' ');
				i++;
			}
			continue;
		}
		/* GL_EXT_spirv_intrinsics' declarations of the module's modes */
		if (src[i] == 's' && (word_at(src, i, "spirv_execution_mode") ||
		    word_at(src, i, "spirv_execution_mode_id"))) {
			while (i < n && src[i] != ';') {
				sb_putc(&b, src[i] == '\n' ? '\n' : ' ');
				i++;
			}
			if (i < n) {
				sb_putc(&b, ' ');
				i++;
			}
			continue;
		}
		if (isalpha((unsigned char)src[i]) && (i == 0 || !(isalnum(
		    (unsigned char)src[i - 1]) || src[i - 1] == '_'))) {
			size_t k;
			for (k = 0; k < NNARROW; k++)
				if (word_at(src, i, narrow_types[k].name))
					break;
			if (k < NNARROW) {
				size_t e = i + strlen(narrow_types[k].name);
				while (src[e] == ' ' || src[e] == '\t')
					e++;
				sb_puts(&b, src[e] == '(' ? narrow_types[k].ctor :
				    narrow_types[k].as);
				i += strlen(narrow_types[k].name);
				continue;
			}
		}
		if (word_at(src, i, "layout")) {
			/* layout(...push_constant...): blanked */
			size_t e = i + 6, depth = 0;
			while (e < n && src[e] != '(' && isspace((unsigned char)src[e]))
				e++;
			if (src[e] == '(') {
				size_t k = e;
				do {
					if (src[k] == '(')
						depth++;
					else if (src[k] == ')')
						depth--;
					k++;
				} while (k < n && depth > 0);
				if (memmem(src + i, k - i, "push_constant", 13) != NULL) {
					for (; i < k; i++)
						sb_putc(&b, src[i] == '\n' ? '\n' : ' ');
					continue;
				}
				/* Vulkan's qualifiers out of the list; the lines kept */
				{
					size_t a = e + 1, kept = 0, nls = 0, q;
					struct sbuf l = {0};
					sb_puts(&l, "");
					while (a < k - 1) {
						size_t z = a, w0, w1;
						int d2 = 0, v;
						while (z < k - 1 && (src[z] != ',' || d2 > 0)) {
							if (src[z] == '(' || src[z] == '[')
								d2++;
							else if (src[z] == ')' || src[z] == ']')
								d2--;
							z++;
						}
						w0 = a;
						while (w0 < z && isspace((unsigned char)src[w0]))
							w0++;
						w1 = w0;
						while (w1 < z && (isalnum((unsigned char)src[w1]) ||
						    src[w1] == '_'))
							w1++;
						for (v = 0; vk_layout[v]; v++)
							if (strlen(vk_layout[v]) == w1 - w0 &&
							    !strncmp(src + w0, vk_layout[v], w1 - w0))
								break;
						if (vk_layout[v] && !strcmp(vk_layout[v],
						    "constant_id"))
							spec = 1;
						if (!vk_layout[v] && w1 > w0) {
							size_t z1 = z;
							while (z1 > w0 && isspace((unsigned char)
							    src[z1 - 1]))
								z1--;
							sb_printf(&l, "%s%.*s", kept ? ", " : "",
							    (int)(z1 - w0), src + w0);
							kept++;
						}
						a = z + 1;
					}
					for (q = i; q < k; q++)
						nls += src[q] == '\n';
					if (kept)
						sb_printf(&b, "layout(%s)", l.p);
					else
						sb_putc(&b, ' ');
					while (nls--)
						sb_putc(&b, '\n');
					free(l.p);
					i = k;
					continue;
				}
			}
		}
		if (src[i] == '=') {
			/* "T name[N] = {" -> "T name[N] = T[N](" */
			size_t k = i + 1, rb, lb, ne, ns, te, ts;
			while (k < n && isspace((unsigned char)src[k]))
				k++;
			rb = i;
			while (rb > 0 && isspace((unsigned char)src[rb - 1]))
				rb--;
			if (src[k] == '{' && rb > 0 && src[rb - 1] == ']') {
				lb = rb - 1;
				while (lb > 0 && src[lb] != '[')
					lb--;
				ne = lb;
				while (ne > 0 && isspace((unsigned char)src[ne - 1]))
					ne--;
				ns = ne;
				while (ns > 0 && (isalnum((unsigned char)src[ns - 1]) ||
				    src[ns - 1] == '_'))
					ns--;
				te = ns;
				while (te > 0 && isspace((unsigned char)src[te - 1]))
					te--;
				ts = te;
				while (ts > 0 && (isalnum((unsigned char)src[ts - 1]) ||
				    src[ts - 1] == '_'))
					ts--;
				if (ts < te && ns < ne) {
					size_t e = k + 1, depth = 1, last = 0;
					sb_printf(&b, "%.*s%.*s%.*s(", (int)(k - i), src + i,
					    (int)(te - ts), src + ts, (int)(rb - lb), src + lb);
					while (e < n && depth > 0) {
						if (src[e] == '{')
							depth++;
						else if (src[e] == '}' && --depth == 0)
							break;
						e++;
					}
					/* a trailing comma goes */
					for (last = e; last > k + 1 && isspace((unsigned char)
					    src[last - 1]); last--)
						;
					{
						size_t q;
						for (q = k + 1; q < e; q++)
							sb_putc(&b, q == last - 1 && src[q] == ',' ? ' ' :
							    src[q]);
					}
					sb_putc(&b, ')');
					i = e + 1;
					continue;
				}
			}
		}
		sb_putc(&b, src[i]);
		i++;
	}
	return b.p ? b.p : xcalloc(1, 1);
}

/* a local's const (blanked for the parser) at the declaration of name_pos */
static int
es_const(size_t pos)
{
	size_t i = pos;
	if (es_orig == NULL)
		return 0;
	while (i > 0) {
		char c = es_orig[i - 1];
		if (c == ';' || c == '{' || c == '}' || c == '(')
			return 0;
		i--;
		if (word_at(es_orig, i, "const"))
			return 1;
	}
	return 0;
}

/* the top-level declaration or function around offset at: its extent */
static void
top_range(const char *s, size_t at, size_t *a, size_t *b)
{
	size_t i = 0, start = 0, n = strlen(s);
	int depth = 0, body = 0;	/* body: the brace opened a function's */
	while (i < n) {
		size_t j = skip_comment(s, i);
		if (j != i) {
			i = j;
			continue;
		}
		if (s[i] == '#' && depth == 0) {	/* a directive is its own */
			size_t e = i + strcspn(s + i, "\n");
			if (at >= i && at < e) {
				*a = i;
				*b = e;
				return;
			}
			i = e;
			start = i;
			continue;
		}
		if (s[i] == '{' && depth == 0) {
			/* after a parameter list: a function's body */
			size_t k = i;
			while (k > start && isspace((unsigned char)s[k - 1]))
				k--;
			body = k > start && s[k - 1] == ')';
		}
		if (s[i] == '{' || s[i] == '(')
			depth++;
		else if (s[i] == ')')
			depth--;
		else if (s[i] == '}') {
			depth--;
			if (depth == 0) {
				/* a function's body ends here; a block's or a struct's
				 * goes on to its ';' (and an instance's name) */
				size_t k = i + 1;
				while (k < n && isspace((unsigned char)s[k]))
					k++;
				if (body || (s[k] != ';' && !isalpha((unsigned char)s[k]))) {
					if (at >= start && at <= i) {
						*a = start;
						*b = i + 1;
						return;
					}
					start = i + 1;
				}
			}
		} else if (s[i] == ';' && depth == 0) {
			if (at >= start && at <= i) {
				*a = start;
				*b = i + 1;
				return;
			}
			start = i + 1;
		}
		i++;
	}
	*a = start;
	*b = n;
}

static Glslc *SG;		/* the parse */
static void *sg_mem;
static char *sg_text;
static int *ymap;		/* snxglslc symbol -> ours */
static int *fmap;		/* snxglslc function symbol -> ours */

static int
snx_ty(uint32_t t, int *arr)
{
	const GlslcType *y = &SG->types[t];
	if (arr)
		*arr = 0;
	if (y->kind == GLSLC_T_ARRAY) {
		if (arr)
			*arr = y->len ? y->len : 1 << 30;
		return snx_ty(y->elem, NULL);
	}
	if (y->kind == GLSLC_T_VOID)
		return TV;
	if (y->kind != GLSLC_T_SCALAR)
		return TX;
	return y->base == SNX_GLSLC_BASE_BOOL ? TB : y->base == SNX_GLSLC_BASE_INT ?
	    TI : y->base == SNX_GLSLC_BASE_UINT ? TU : TF;
}

static int
snx_line(uint32_t tok)
{
	uint32_t line, col;
	glslc_locate(SG, SG->toks[tok].site, &line, &col);
	return (int)line;
}

static void
snx_name(uint32_t s, char *buf, size_t n)
{
	GlslcView v = SG->syms[s].name;
	snprintf(buf, n, "%.*s", (int)v.len, glslc_view_text(SG, v));
}

static int
snx_sym(uint32_t s)
{
	const GlslcSymbol *y;
	char nm[64];
	int k, ty, arr;
	if (ymap[s] >= 0)
		return ymap[s];
	y = &SG->syms[s];
	snx_name(s, nm, sizeof nm);
	ty = snx_ty(y->type, &arr);
	if (y->kind == GLSLC_SK_PARAM)
		k = Y_PARAM;
	else if (y->kind == GLSLC_SK_VAR && y->storage == GLSLC_ST_LOCAL)
		k = Y_LOCAL;
	else if (y->kind == GLSLC_SK_VAR && y->storage == GLSLC_ST_GLOBAL)
		k = Y_GLOBAL;
	else if (y->kind == GLSLC_SK_VAR && y->storage == GLSLC_ST_CONST)
		k = y->node && SG->nodes[y->node].kind == GLSLC_N_DECL ? Y_LOCAL :
		    Y_CONST;
	else
		k = Y_BUF;
	ymap[s] = sym_new(nm, k, ty, y->node ? snx_line(SG->nodes[y->node].tok) :
	    0);
	G.y[ymap[s]].arr = arr;
	if (y->kind == GLSLC_SK_PARAM && (y->storage == GLSLC_ST_PARAM_OUT ||
	    y->storage == GLSLC_ST_PARAM_INOUT))
		G.y[ymap[s]].out = 1;
	if (k == Y_LOCAL && (y->storage == GLSLC_ST_CONST ||
	    (y->node && es_const(SG->toks[SG->nodes[y->node].tok].site))))
		G.y[ymap[s]].cnst = 1;
	if (ty == TF || ty == TX)
		G.y[ymap[s]].flt = 1;
	return ymap[s];
}

static int cv_x(uint32_t n);

static int
cv_args(uint32_t first, int *out, int max)
{
	int na = 0;
	uint32_t a;
	for (a = first; a; a = SG->nodes[a].next)
		if (na < max)
			out[na++] = cv_x(a);
	return na;
}

static int
call_node(int fn, int ty, const int *args, int na, int line)
{
	int x = nx_new(X_CALL, ty, line), i;
	G.x[x].a = G.nxarg;
	G.x[x].b = na;
	G.x[x].sym = fn;
	for (i = 0; i < na; i++) {
		if (G.nxarg == G.capxarg) {
			G.capxarg = G.capxarg ? G.capxarg * 2 : 4096;
			G.xarg = xrealloc(G.xarg, (size_t)G.capxarg * sizeof *G.xarg);
		}
		G.xarg[G.nxarg++] = args[i];
	}
	return x;
}

static int
op_of(int o)
{
	switch (o) {
	case GLSLC_OP_ADD: return OP_ADD;
	case GLSLC_OP_SUB: return OP_SUB;
	case GLSLC_OP_MUL: return OP_MUL;
	case GLSLC_OP_DIV: return OP_DIV;
	case GLSLC_OP_MOD: return OP_MOD;
	case GLSLC_OP_SHL: return OP_SHL;
	case GLSLC_OP_SHR: return OP_SHR;
	case GLSLC_OP_AND: return OP_AND;
	case GLSLC_OP_OR: return OP_OR;
	case GLSLC_OP_XOR: return OP_XOR;
	case GLSLC_OP_LT: return OP_LT;
	case GLSLC_OP_GT: return OP_GT;
	case GLSLC_OP_LE: return OP_LE;
	case GLSLC_OP_GE: return OP_GE;
	case GLSLC_OP_EQ: return OP_EQ;
	case GLSLC_OP_NE: return OP_NE;
	case GLSLC_OP_LOGIC_AND: return OP_LAND;
	case GLSLC_OP_LOGIC_OR: return OP_LOR;
	case GLSLC_OP_LOGIC_XOR: return OP_LXOR;
	}
	return OP_NONE;
}

/* an expression node as ours */
static int
cv_x(uint32_t n)
{
	const GlslcNode *e = &SG->nodes[n];
	int ty = snx_ty(e->type, NULL), line = snx_line(e->tok), x, a, b;
	int args[64], na;

	switch (e->kind) {
	case GLSLC_N_CONST: {
		uint32_t bits = (uint32_t)e->a | ((uint32_t)e->b << 16);
		x = nx_new(X_NUM, ty, line);
		G.x[x].v = ty == TU ? (int64_t)bits : ty == TI ? (int64_t)(int32_t)bits :
		    ty == TB ? bits != 0 : 0;
		return x;
	}
	case GLSLC_N_VAR: {
		int y = snx_sym(e->a);
		x = nx_new(X_VAR, G.y[y].arr ? TX : G.y[y].ty, line);
		G.x[x].sym = y;
		return x;
	}
	case GLSLC_N_UNARY:
		a = cv_x(e->a);
		if (e->op == GLSLC_OP_PLUS)
			return a;
		if (e->op >= GLSLC_OP_PRE_INC) {
			x = nx_new(X_INC, ty, line);
			G.x[x].op = e->op == GLSLC_OP_PRE_INC || e->op ==
			    GLSLC_OP_POST_INC ? OP_ADD : OP_SUB;
			G.x[x].post = e->op >= GLSLC_OP_POST_INC;
			G.x[x].a = a;
			return x;
		}
		if (e->op == GLSLC_OP_NEG && G.x[a].k == X_NUM && ty == TI) {
			G.x[a].v = -G.x[a].v;
			return a;
		}
		x = nx_new(X_UN, ty, line);
		G.x[x].op = e->op == GLSLC_OP_NEG ? OP_NEG : e->op == GLSLC_OP_NOT ?
		    OP_LNOT : OP_NOT;
		G.x[x].a = a;
		return x;
	case GLSLC_N_BINARY:
		a = cv_x(e->a);
		b = cv_x(e->b);
		x = nx_new(X_BIN, ty, line);
		G.x[x].op = (uint8_t)op_of(e->op);
		G.x[x].a = a;
		G.x[x].b = b;
		if (G.x[x].op == OP_NONE)
			G.x[x].ty = TX;
		return x;
	case GLSLC_N_ASSIGN:
		a = cv_x(e->a);
		b = cv_x(e->b);
		x = nx_new(X_ASG, ty, line);
		G.x[x].op = e->op == GLSLC_OP_ASSIGN ? OP_NONE : (uint8_t)op_of(e->op);
		G.x[x].a = a;
		G.x[x].b = b;
		return x;
	case GLSLC_N_TERNARY:
		x = nx_new(X_SEL, ty, line);
		G.x[x].a = cv_x(e->a);
		G.x[x].b = cv_x(e->b);
		G.x[x].c = cv_x(e->c);
		return x;
	case GLSLC_N_INDEX:
		a = cv_x(e->a);
		b = cv_x(e->b);
		x = nx_new(X_IDX, ty, line);
		G.x[x].a = a;
		G.x[x].b = b;
		return x;
	case GLSLC_N_SWIZZLE:
	case GLSLC_N_FIELD:
	case GLSLC_N_LENGTH:
		x = nx_new(X_MEM, ty == TI || ty == TU ? ty : TX, line);
		G.x[x].a = cv_x(e->a);
		return x;
	case GLSLC_N_CALL:
		na = cv_args(e->a, args, 64);
		return call_node(fmap[e->b] >= 0 ? fmap[e->b] : -B_UNK, ty, args, na,
		    line);
	case GLSLC_N_BUILTIN: {
		const char *nm = glslc_builtin_fns[e->b].name;
		int f = !strcmp(nm, "min") ? -B_MIN : !strcmp(nm, "max") ? -B_MAX :
		    !strcmp(nm, "clamp") ? -B_CLAMP : !strcmp(nm, "abs") ? -B_ABS :
		    !strcmp(nm, "bitfieldExtract") ? -B_BFE : -B_UNK;
		na = cv_args(e->a, args, 64);
		if (ty != TI && ty != TU)
			f = -B_UNK;
		if (f == -B_BFE && (na != 3 || G.x[args[0]].ty != ty))
			f = -B_UNK;
		return call_node(f, ty, args, na, line);
	}
	case GLSLC_N_CONSTRUCT:
		na = cv_args(e->a, args, 64);
		return call_node(na == 1 && (ty == TI || ty == TU || ty == TB) &&
		    (G.x[args[0]].ty == TI || G.x[args[0]].ty == TU ||
		    G.x[args[0]].ty == TB) ? (ty == TI ? -B_INT : ty == TU ? -B_UINT :
		    -B_BOOL) : -B_UNK, ty, args, na, line);
	case GLSLC_N_COMMA:
		x = nx_new(X_COMMA, ty, line);
		G.x[x].a = cv_x(e->a);
		G.x[x].b = cv_x(e->b);
		return x;
	}
	x = nx_new(X_CALL, TX, line);
	G.x[x].sym = -B_UNK;
	G.x[x].a = G.nxarg;
	G.x[x].b = 0;
	return x;
}

static int cv_s(uint32_t n);
static int cv_switch(uint32_t n, int line);
static int cv_tail(int s);
static int exits(int s, int inloop);

/* a statement list (a block's, linked by next) */
static int
cv_list(uint32_t first, int line, int scoped)
{
	int kids[2048], nk = 0;
	uint32_t s;
	for (s = first; s; s = SG->nodes[s].next)
		if (nk < 2048)
			kids[nk++] = cv_s(s);
	return block_of(kids, nk, line, scoped);
}

static int
cv_s(uint32_t n)
{
	const GlslcNode *e = &SG->nodes[n];
	int line = snx_line(e->tok), s;

	switch (e->kind) {
	case GLSLC_N_BLOCK:
		return cv_list(e->a, line, 1);
	case GLSLC_N_EXPR:
		s = ns_new(S_EXPR, line);
		G.s[s].a = cv_x(e->a);
		return s;
	case GLSLC_N_DECL:
		s = ns_new(S_DECL, line);
		G.s[s].sym = snx_sym(e->a);
		G.y[G.s[s].sym].fn = G.cur;
		if (e->b) {
			/* a local array's initializer: opaque */
			G.s[s].a = G.y[G.s[s].sym].arr ? -1 : cv_x(e->b);
			G.y[G.s[s].sym].init = G.s[s].a;
		}
		return s;
	case GLSLC_N_IF:
		s = ns_new(S_IF, line);
		G.s[s].a = cv_x(e->a);
		G.s[s].b = e->b ? cv_s(e->b) : ns_new(S_EMPTY, line);
		G.s[s].c = e->c ? cv_s(e->c) : -1;
		return s;
	case GLSLC_N_FOR: {
		const GlslcNode *r = &SG->nodes[e->c];
		s = ns_new(S_FOR, line);
		/* the init: one statement, or declarations linked by next */
		if (e->a)
			G.s[s].a = SG->nodes[e->a].next ? cv_list(e->a, line, 0) :
			    cv_s(e->a);
		G.s[s].b = e->b ? cv_x(e->b) : -1;
		G.s[s].c = r->a ? cv_x(r->a) : -1;
		G.s[s].d = r->b ? cv_s(r->b) : ns_new(S_EMPTY, line);
		return s;
	}
	case GLSLC_N_WHILE:
		s = ns_new(S_WHILE, line);
		G.s[s].b = cv_x(e->a);
		G.s[s].d = e->b ? cv_s(e->b) : ns_new(S_EMPTY, line);
		return s;
	case GLSLC_N_DO:
		s = ns_new(S_DO, line);
		G.s[s].d = e->a ? cv_s(e->a) : ns_new(S_EMPTY, line);
		G.s[s].b = cv_x(e->b);
		return s;
	case GLSLC_N_BREAK:
		return ns_new(S_BRK, line);
	case GLSLC_N_CONTINUE:
		return ns_new(S_CONT, line);
	case GLSLC_N_RETURN:
		s = ns_new(S_RET, line);
		G.s[s].a = e->a ? cv_x(e->a) : -1;
		G.s[s].sym = G.cur >= 0 ? G.f[G.cur].ret : -1;
		return s;
	case GLSLC_N_SWITCH:
		if ((s = cv_switch(n, line)) >= 0)
			return s;
		/* FALLTHROUGH */
	case GLSLC_N_DISCARD:
	default:
		if (G.cur >= 0)
			G.f[G.cur].unsup = 1;
		return ns_new(S_UNK, line);
	}
}

/* x reads and calls nothing that could write */
static int
cv_plain(int x)
{
	const struct gx *g = &G.x[x];
	int i;
	switch (g->k) {
	case X_NUM:
	case X_VAR:
		return 1;
	case X_ASG:
	case X_INC:
		return 0;
	case X_CALL:
		if (g->sym >= 0)
			return 0;
		for (i = 0; i < g->b; i++)
			if (!cv_plain(G.xarg[g->a + i]))
				return 0;
		return 1;
	case X_UN:
	case X_MEM:
		return cv_plain(g->a);
	case X_SEL:
		return cv_plain(g->a) && cv_plain(g->b) && cv_plain(g->c);
	default:
		return cv_plain(g->a) && cv_plain(g->b);
	}
}

/*
 * How a case's last statement leaves: 1 every path returns, 2 it ends in
 * a break (one at the end of a block is taken out: falling off the end
 * of the case is leaving the switch), 0 it may fall into the next case.
 */
static int
cv_tail(int s)
{
	struct gs *t = &G.s[s];
	int a, b;
	switch (t->k) {
	case S_RET:
		return 1;
	case S_BRK:
		return 2;
	case S_BLOCK:
		if (t->n == 0)
			return 0;
		a = cv_tail(G.slist[t->first + t->n - 1]);
		if (a == 2 && G.s[G.slist[t->first + t->n - 1]].k == S_BRK)
			t->n--;
		return a;
	case S_IF:
		if (t->c < 0)
			return 0;
		a = cv_tail(t->b);
		b = cv_tail(t->c);
		if (G.s[t->b].k == S_BRK)
			G.s[t->b].k = S_EMPTY;
		if (G.s[t->c].k == S_BRK)
			G.s[t->c].k = S_EMPTY;
		return a && b ? (a == 1 && b == 1 ? 1 : 2) : 0;
	}
	return 0;
}

/*
 * A switch as a chain of ifs on its selector, when no case falls into the
 * next: each group of labels' statements ends in a break, which goes, or
 * a return, and no other break leaves the switch (one inside a loop of
 * the case is that loop's).  default is the last else wherever it sits.
 * The selector is read in every compare, so it has to be plain.  -1 when
 * the switch is not one of those (it stays a statement the level does
 * not read).
 */
static int
cv_switch(uint32_t n, int line)
{
	const GlslcNode *e = &SG->nodes[n];
	int sel = cv_x(e->a), ng = 0, g, i, dflt = -1, s = -1;
	struct cgrp {
		int nl, lab[64], nk, kid[512], line;
	} *grp;
	uint32_t t;
	if (!cv_plain(sel))
		return -1;
	/* a case may hold a switch of its own: not static */
	grp = xcalloc(128, sizeof *grp);
	for (t = e->b; t; t = SG->nodes[t].next) {
		const GlslcNode *c = &SG->nodes[t];
		if (c->kind == GLSLC_N_CASE) {
			if (ng == 0 || grp[ng - 1].nk) {
				if (ng == 128)
					goto no;
				memset(&grp[ng], 0, sizeof grp[ng]);
				grp[ng].line = snx_line(c->tok);
				ng++;
			}
			if (grp[ng - 1].nl == 64)
				goto no;
			grp[ng - 1].lab[grp[ng - 1].nl++] = c->a ? cv_x(c->a) : -1;
			continue;
		}
		if (ng == 0 || grp[ng - 1].nk == 512)
			goto no;
		grp[ng - 1].kid[grp[ng - 1].nk++] = cv_s(t);
	}
	for (g = 0; g < ng; g++) {
		int nk = grp[g].nk, last = nk ? grp[g].kid[nk - 1] : -1;
		int tail = last >= 0 ? cv_tail(last) : 0;
		if (last >= 0 && G.s[last].k == S_BRK)
			grp[g].nk = --nk;	/* the break that ends the case */
		else if (tail == 0 && g + 1 < ng)
			goto no;		/* it falls into the next case */
		for (i = 0; i < grp[g].nk; i++)
			if (exits(grp[g].kid[i], 0) & 2)
				goto no;	/* a break out of the middle */
		for (i = 0; i < grp[g].nl; i++)
			if (grp[g].lab[i] < 0)
				dflt = g;
	}
	if (dflt >= 0)
		s = block_of(grp[dflt].kid, grp[dflt].nk, grp[dflt].line, 1);
	for (g = ng - 1; g >= 0; g--) {
		int cond = -1, b, k;
		if (g == dflt)
			continue;
		for (i = 0; i < grp[g].nl; i++) {
			int eq = nx_new(X_BIN, TB, grp[g].line);
			G.x[eq].op = OP_EQ;
			G.x[eq].a = sel;
			G.x[eq].b = grp[g].lab[i];
			if (cond < 0)
				cond = eq;
			else {
				int o = nx_new(X_BIN, TB, grp[g].line);
				G.x[o].op = OP_LOR;
				G.x[o].a = cond;
				G.x[o].b = eq;
				cond = o;
			}
		}
		b = block_of(grp[g].kid, grp[g].nk, grp[g].line, 1);
		k = ns_new(S_IF, grp[g].line);
		G.s[k].a = cond;
		G.s[k].b = b;
		G.s[k].c = s;
		s = k;
	}
	free(grp);
	return s >= 0 ? s : ns_new(S_EMPTY, line);
no:
	free(grp);
	return -1;
}

/* parse src (GLSL 4.50 or ES) with snxglslc and convert the tree: 0 ok */
static int
parse_source(const char *src, int stage)
{
	size_t ws = snx_glslc_workspace_size();
	GlslcResult *res = xcalloc(1, sizeof *res);
	char *t1 = es_text(src, stage), *t2;
	int tries, ok = 0, vulkan = 1;
	uint32_t s, n;

	{
		/* not GLSL ES: Vulkan's GLSL, with its implicit conversions */
		const char *v = strstr(src, "#version");
		if (v != NULL && memmem(v, strcspn(v, "\n"), " es", 3) != NULL)
			vulkan = 0;
	}
	dropped[0] = 0;
	if (strlen(t1) > SNX_GLSLC_MAX_SOURCE) {
		snprintf(G.why, sizeof G.why, "the source is %zu bytes, more than "
		    "the %u the GLSL parser takes", strlen(t1), SNX_GLSLC_MAX_SOURCE);
		free(t1);
		free(res);
		return -1;
	}
	/* local consts blanked (remembered in es_orig) */
	t2 = xmalloc(strlen(t1) + 1);
	memcpy(t2, t1, strlen(t1) + 1);
	{
		size_t i, len = strlen(t2);
		int depth = 0;
		for (i = 0; i < len; i++) {
			size_t j = skip_comment(t2, i);
			if (j != i) {
				i = j - 1;
				continue;
			}
			if (t2[i] == '{')
				depth++;
			else if (t2[i] == '}')
				depth--;
			else if (depth > 0 && word_at(t2, i, "const"))
				blank(t2, i, i + 5);
		}
	}
	free(es_orig);
	es_orig = t1;
	if (sg_mem == NULL)
		sg_mem = xmalloc(ws + 16);
	for (tries = 0; tries < 48; tries++) {
		void *m = (void *)(((uintptr_t)sg_mem + 15) & ~(uintptr_t)15);
		memset(res, 0, sizeof *res);
		SG = glslc_open(m, ws, (GlslcStage)stage, t2, strlen(t2), res);
		if (SG == NULL)
			break;
		SG->implicit = vulkan;
		if (glslc_run(SG, GLSLC_PASS_PARSE, NULL, 0) == 0) {
			ok = 1;
			break;
		}
		if (SG->ntoks == 0)
			break;	/* nothing was read: no piece to drop */
		if (strstr(res->error, "no main function"))
			break;	/* main was dropped: nothing else to drop brings it back */
		{
			/* drop what it refused: the top-level piece around it */
			size_t at = SG->toks[SG->pos < SG->ntoks ? SG->pos :
			    SG->ntoks - 1].site, a, b, l = strlen(dropped);
			char nm[48] = "";
			top_range(t2, at, &a, &b);
			if (b <= a)
				break;
			{
				/* its name: the word before the first '(' or ';' */
				size_t k = a, last = a, lastlen = 0;
				while (k < b && t2[k] != '(' && t2[k] != ';' &&
				    t2[k] != '=' && t2[k] != '{') {
					if (isalpha((unsigned char)t2[k]) || t2[k] == '_') {
						size_t e = k;
						while (isalnum((unsigned char)t2[e]) || t2[e] == '_')
							e++;
						last = k;
						lastlen = e - k;
						k = e;
						continue;
					}
					k++;
				}
				snprintf(nm, sizeof nm, "%.*s", (int)(lastlen < 40 ? lastlen :
				    40), t2 + last);
			}
			if (l + 120 < sizeof dropped)
				snprintf(dropped + l, sizeof dropped - l, "%s%s (line %u: %s)",
				    l ? "; " : "", nm[0] ? nm : "a declaration", res->line,
				    res->error);
			blank(t2, a, b);
			blank(t1, a, b);
		}
	}
	if (!ok) {
		if (dropped[0] && strstr(res->error, "no main function"))
			/* what it refused, main last: why main is not there */
			snprintf(G.why, sizeof G.why, "the GLSL parser left main out: "
			    "%s", dropped);
		else
			snprintf(G.why, sizeof G.why, "the GLSL parser refused it: line "
			    "%u: %s", res->line, res->error);
		free(t2);
		free(res);
		return -1;
	}
	free(sg_text);
	sg_text = t2;
	/* the tree, converted: functions first (calls refer to them) */
	n = SG->nsyms;
	ymap = xrealloc(ymap, (size_t)n * sizeof *ymap);
	fmap = xrealloc(fmap, (size_t)n * sizeof *fmap);
	for (s = 0; s < n; s++)
		ymap[s] = fmap[s] = -1;
	G.cur = -1;
	for (s = 1; s < n; s++) {
		const GlslcSymbol *y = &SG->syms[s];
		if (y->kind != GLSLC_SK_FUNC || !y->node)
			continue;
		{
			/* a subgroup built-in's stub: its calls are unknown values */
			size_t at = SG->toks[SG->nodes[y->node].tok].site;
			if (at >= sg_stub_from && at < sg_stub_to)
				continue;
		}
		if (G.nf == G.capf) {
			G.capf = G.capf ? G.capf * 2 : 64;
			G.f = xrealloc(G.f, (size_t)G.capf * sizeof *G.f);
		}
		memset(&G.f[G.nf], 0, sizeof G.f[0]);
		snx_name(s, G.f[G.nf].name, sizeof G.f[0].name);
		G.f[G.nf].body = -1;
		G.f[G.nf].ret = -1;
		fmap[s] = G.nf++;
	}
	for (s = 1; s < n; s++) {
		const GlslcSymbol *y = &SG->syms[s];
		const GlslcNode *d;
		int f = fmap[s], i, rt;
		if (f < 0)
			continue;
		d = &SG->nodes[y->node];
		G.cur = f;
		rt = snx_ty(y->type, NULL);	/* the return type */
		G.f[f].ty = (uint8_t)rt;
		G.f[f].line = snx_line(d->tok);
		for (i = 0; i < y->aux2; i++) {
			const GlslcParam *p = &SG->params[y->aux + i];
			int ps = p->sym ? snx_sym(p->sym) : sym_new("?", Y_PARAM,
			    snx_ty(p->type, NULL), G.f[f].line);
			G.y[ps].fn = f;
			if (G.y[ps].out || G.y[ps].ty == TX || G.y[ps].ty == TF ||
			    G.y[ps].arr)
				G.f[f].unsup = 1;
			if (G.f[f].np < MAXP + 8)
				G.f[f].p[G.f[f].np++] = ps;
			else
				G.f[f].unsup = 1;
		}
		G.f[f].ret = sym_new("<ret>", Y_RET, rt, G.f[f].line);
		G.y[G.f[f].ret].fn = f;
		G.f[f].body = d->b ? cv_s(d->b) : -1;
		G.f[f].endline = G.f[f].line;
		{
			/* the last line: the closing brace's */
			uint32_t k;
			for (k = d->tok; k < SG->ntoks; k++)
				if (SG->toks[k].kind == TK_LBRACE)
					break;
			{
				int depth = 0;
				for (; k < SG->ntoks; k++) {
					if (SG->toks[k].kind == TK_LBRACE)
						depth++;
					else if (SG->toks[k].kind == TK_RBRACE && --depth == 0)
						break;
				}
				if (k < SG->ntoks)
					G.f[f].endline = snx_line(k);
			}
		}
		G.cur = -1;
	}
	/* globals: their initializers, constant tables */
	for (s = 1; s < n; s++) {
		const GlslcSymbol *y = &SG->syms[s];
		int ys;
		if (y->kind != GLSLC_SK_VAR || !y->node ||
		    SG->nodes[y->node].kind != GLSLC_N_GLOBAL)
			continue;
		ys = snx_sym(s);
		if (SG->nodes[y->node].b) {
			uint32_t in = SG->nodes[y->node].b;
			if (G.y[ys].arr && SG->nodes[in].kind == GLSLC_N_CONSTRUCT) {
				int64_t vals[1024];
				int nv = 0, bad = 0;
				uint32_t a;
				for (a = SG->nodes[in].a; a; a = SG->nodes[a].next) {
					const GlslcNode *c = &SG->nodes[a];
					uint32_t bits = (uint32_t)c->a | ((uint32_t)c->b << 16);
					if (c->kind != GLSLC_N_CONST)
						bad = 1;
					if (nv < 1024)
						vals[nv++] = G.y[ys].ty == TI ? (int64_t)(int32_t)bits :
						    (int64_t)bits;
				}
				if (!bad && nv > 0) {
					G.y[ys].tab = xmalloc((size_t)nv * sizeof (int64_t));
					memcpy(G.y[ys].tab, vals, (size_t)nv * sizeof (int64_t));
					G.y[ys].arr = nv;
				} else
					G.y[ys].flt = 1;
			} else if (!G.y[ys].arr)
				G.y[ys].init = cv_x(in);
		}
	}
	free(res);
	return 0;
}


/* ---- GLSL text of the tree (suggestions) ---- */

static int xprec(int x);

static void
px_text(struct sbuf *b, int x, int prec);

/* patches: a candidate's changes, consulted when printing and running */
static int *xrep;	/* expression -> its replacement, -1 */
static int *srep;	/* statement -> its replacement, -1; -2 removed */

static int
XR(int x)
{
	while (x >= 0 && xrep && xrep[x] >= 0)
		x = xrep[x];
	return x;
}
static int
SR(int s)
{
	while (s >= 0 && srep && srep[s] != -1)
		s = srep[s];	/* -2: removed */
	return s;
}

static int
xprec(int x)
{
	const struct gx *e = &G.x[x];
	switch (e->k) {
	case X_BIN:
		return op_prec(e->op);
	case X_SEL:
		return 0;
	case X_ASG:
	case X_COMMA:
		return -1;
	case X_UN:
		return 12;
	default:
		return 13;
	}
}

static void
num_text(struct sbuf *b, int64_t v, int ty)
{
	if (ty == TB)
		sb_puts(b, v ? "true" : "false");
	else if (ty == TU)
		sb_printf(b, v >= 4096 && ((v + 1) & v) == 0 ? "0x%llXu" :
		    "%lluu", (unsigned long long)v);
	else
		sb_printf(b, "%lld", (long long)v);
}

static void
px_text(struct sbuf *b, int x, int prec)
{
	const struct gx *e;
	int p, i;
	x = XR(x);
	e = &G.x[x];
	p = xprec(x);
	if (p < prec)
		sb_putc(b, '(');
	switch (e->k) {
	case X_NUM:
		if (e->ty == TI && e->v < 0 && prec > 11)
			sb_printf(b, "(%lld)", (long long)e->v);
		else
			num_text(b, e->v, e->ty);
		break;
	case X_VAR:
		sb_puts(b, G.y[e->sym].name);
		break;
	case X_IDX:
		px_text(b, e->a, 13);
		sb_putc(b, '[');
		px_text(b, e->b, 0);
		sb_putc(b, ']');
		break;
	case X_MEM:
		px_text(b, e->a, 13);
		sb_puts(b, ".?");
		break;
	case X_CALL: {
		const char *nm = e->sym >= 0 ? G.f[e->sym].name :
		    e->sym == -B_MIN ? "min" : e->sym == -B_MAX ? "max" :
		    e->sym == -B_CLAMP ? "clamp" : e->sym == -B_ABS ? "abs" :
		    e->sym == -B_INT ? "int" : e->sym == -B_UINT ? "uint" :
		    e->sym == -B_BFE ? "bitfieldExtract" :
		    e->sym == -B_BOOL ? "bool" : "f";
		sb_printf(b, "%s(", nm);
		for (i = 0; i < e->b; i++) {
			if (i)
				sb_puts(b, ", ");
			px_text(b, G.xarg[e->a + i], 0);
		}
		sb_putc(b, ')');
		break;
	}
	case X_UN:
		sb_puts(b, op_str(e->op));
		px_text(b, e->a, 12);
		break;
	case X_BIN: {
		/* GLSL's precedence, but parentheses wherever a reader would want
		 * them: a shift's or a bitwise operation's operands, arithmetic
		 * inside a compare excepted */
		int i2;
		for (i2 = 0; i2 < 2; i2++) {
			int c = XR(i2 ? e->b : e->a), cp;
			if (i2)
				sb_printf(b, " %s ", op_str(e->op));
			cp = xprec(c);
			if (G.x[c].k == X_BIN && cp != p && !(p <= 3 && cp >= 7) &&
			    !(p >= 7 && p <= 8 && cp >= 9) && !(p == 10 && cp == 11))
				px_text(b, c, 99);
			else
				px_text(b, c, i2 ? p + 1 : p);
		}
		break;
	}
	case X_SEL:
		px_text(b, e->a, 1);
		sb_puts(b, " ? ");
		px_text(b, e->b, 0);
		sb_puts(b, " : ");
		px_text(b, e->c, 0);
		break;
	case X_ASG:
		px_text(b, e->a, 13);
		sb_printf(b, " %s= ", e->op ? op_str(e->op) : "");
		px_text(b, e->b, 0);
		break;
	case X_INC:
		if (!e->post)
			sb_puts(b, e->op == OP_ADD ? "++" : "--");
		px_text(b, e->a, 13);
		if (e->post)
			sb_puts(b, e->op == OP_ADD ? "++" : "--");
		break;
	case X_COMMA:
		px_text(b, e->a, 0);
		sb_puts(b, ", ");
		px_text(b, e->b, 0);
		break;
	}
	if (p < prec)
		sb_putc(b, ')');
}

static const char *
ty_name(int ty)
{
	return ty == TB ? "bool" : ty == TI ? "int" : ty == TU ? "uint" :
	    ty == TV ? "void" : ty == TF ? "float" : "?";
}

static void
indent(struct sbuf *b, int d)
{
	int i;
	for (i = 0; i < d; i++)
		sb_puts(b, "    ");
}

static void ps_text(struct sbuf *b, int s, int d);

/* a statement as the body of if/for: a block's braces kept */
static void
ps_body(struct sbuf *b, int s, int d)
{
	s = SR(s);
	sb_puts(b, " {\n");
	if (s >= 0 && G.s[s].k == S_BLOCK && G.s[s].scoped) {
		int i;
		for (i = 0; i < G.s[s].n; i++)
			ps_text(b, G.slist[G.s[s].first + i], d + 1);
	} else
		ps_text(b, s, d + 1);
	indent(b, d);
	sb_puts(b, "}");
}

static void
decl_text(struct sbuf *b, int s)
{
	int y = G.s[s].sym;
	sb_printf(b, "%s%s %s", G.y[y].cnst ? "const " : "", ty_name(G.y[y].ty),
	    G.y[y].name);
	if (G.y[y].arr)
		sb_printf(b, "[%d]", G.y[y].arr);
	if (G.s[s].a >= 0) {
		sb_puts(b, " = ");
		px_text(b, G.s[s].a, 0);
	}
}

static void
ps_text(struct sbuf *b, int s, int d)
{
	const struct gs *t;
	int i;
	s = SR(s);
	if (s == -2 || s < 0)
		return;
	t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		if (t->scoped) {
			indent(b, d);
			sb_puts(b, "{\n");
		}
		for (i = 0; i < t->n; i++)
			ps_text(b, G.slist[t->first + i], t->scoped ? d + 1 : d);
		if (t->scoped) {
			indent(b, d);
			sb_puts(b, "}\n");
		}
		return;
	case S_DECL:
		indent(b, d);
		decl_text(b, s);
		sb_puts(b, ";\n");
		return;
	case S_EXPR:
		indent(b, d);
		px_text(b, t->a, -1);
		sb_puts(b, ";\n");
		return;
	case S_IF:
		indent(b, d);
		sb_puts(b, "if (");
		px_text(b, t->a, 0);
		sb_puts(b, ")");
		ps_body(b, t->b, d);
		if (t->c >= 0 && SR(t->c) != -2) {
			sb_puts(b, " else");
			if (G.s[SR(t->c)].k == S_IF) {
				struct sbuf e = {0};
				ps_text(&e, t->c, d);
				sb_printf(b, " %s", e.p + 4 * d);
				free(e.p);
				return;
			}
			ps_body(b, t->c, d);
		}
		sb_puts(b, "\n");
		return;
	case S_FOR:
		indent(b, d);
		sb_puts(b, "for (");
		if (t->a >= 0) {
			int a = SR(t->a);
			if (G.s[a].k == S_DECL)
				decl_text(b, a);
			else if (G.s[a].k == S_EXPR)
				px_text(b, G.s[a].a, 0);
		}
		sb_puts(b, "; ");
		if (t->b >= 0)
			px_text(b, t->b, 0);
		sb_puts(b, "; ");
		if (t->c >= 0)
			px_text(b, t->c, -1);
		sb_puts(b, ")");
		ps_body(b, t->d, d);
		sb_puts(b, "\n");
		return;
	case S_WHILE:
		indent(b, d);
		sb_puts(b, "while (");
		px_text(b, t->b, 0);
		sb_puts(b, ")");
		ps_body(b, t->d, d);
		sb_puts(b, "\n");
		return;
	case S_DO:
		indent(b, d);
		sb_puts(b, "do");
		ps_body(b, t->d, d);
		sb_puts(b, " while (");
		px_text(b, t->b, 0);
		sb_puts(b, ");\n");
		return;
	case S_RET:
		indent(b, d);
		sb_puts(b, "return");
		if (t->a >= 0) {
			sb_putc(b, ' ');
			px_text(b, t->a, 0);
		}
		sb_puts(b, ";\n");
		return;
	case S_BRK:
		indent(b, d);
		sb_puts(b, "break;\n");
		return;
	case S_CONT:
		indent(b, d);
		sb_puts(b, "continue;\n");
		return;
	case S_EMPTY:
		return;
	default:
		indent(b, d);
		sb_puts(b, "/* ... */\n");
		return;
	}
}

static void
fn_text(struct sbuf *b, int f)
{
	int i;
	sb_printf(b, "%s %s(", ty_name(G.f[f].ty), G.f[f].name);
	for (i = 0; i < G.f[f].np; i++)
		sb_printf(b, "%s%s %s", i ? ", " : "", ty_name(G.y[G.f[f].p[i]].ty),
		    G.y[G.f[f].p[i]].name);
	sb_puts(b, ")");
	ps_body(b, G.f[f].body, 0);
	sb_puts(b, "\n");
}

/* ---- concrete semantics: 32-bit GLSL (SPIR-V) arithmetic ---- */

static int64_t
norm(int ty, int64_t v)
{
	if (ty == TU)
		return (int64_t)(uint32_t)v;
	if (ty == TI)
		return (int64_t)(int32_t)(uint32_t)v;
	if (ty == TB)
		return v != 0;
	return v;
}

/* the operands' common type for an arithmetic or compare operation */
static int
opnd_ty(const struct gx *e)
{
	if (e->op == OP_SHL || e->op == OP_SHR)
		return G.x[e->a].ty;
	if (e->op == OP_LAND || e->op == OP_LOR || e->op == OP_LXOR)
		return TB;
	return bin_ty(OP_ADD, G.x[e->a].ty, G.x[e->b].ty);
}

/* one operation on operands of type ty; *ub: undefined (shift >= 32,
 * division by zero, INT_MIN / -1) */
static int64_t
cop(int op, int ty, int64_t a, int64_t b, int *ub)
{
	uint32_t ua = (uint32_t)a, ub2 = (uint32_t)b;
	switch (op) {
	case OP_ADD: return norm(ty, a + b);
	case OP_SUB: return norm(ty, a - b);
	case OP_MUL: return norm(ty, (int64_t)(ua * ub2));
	case OP_DIV:
	case OP_MOD:
		if (b == 0 || (ty == TI && a == INT32_MIN && b == -1)) {
			*ub = 1;
			return 0;
		}
		if (ty == TU)
			return op == OP_DIV ? (int64_t)(ua / ub2) : (int64_t)(ua % ub2);
		return op == OP_DIV ? norm(ty, a / b) : norm(ty, a % b);
	case OP_SHL:
		if (b < 0 || b > 31) {
			*ub = 1;
			return 0;
		}
		return norm(ty, (int64_t)(ua << b));
	case OP_SHR:
		if (b < 0 || b > 31) {
			*ub = 1;
			return 0;
		}
		return ty == TI ? norm(ty, a >> b) : (int64_t)(ua >> b);
	case OP_AND: return norm(ty, (int64_t)(ua & ub2));
	case OP_OR: return norm(ty, (int64_t)(ua | ub2));
	case OP_XOR: return norm(ty, (int64_t)(ua ^ ub2));
	case OP_LT: return a < b;
	case OP_LE: return a <= b;
	case OP_GT: return a > b;
	case OP_GE: return a >= b;
	case OP_EQ: return a == b;
	case OP_NE: return a != b;
	case OP_LAND: return a && b;
	case OP_LOR: return a || b;
	case OP_LXOR: return (a != 0) != (b != 0);
	}
	*ub = 1;
	return 0;
}

static int64_t
conv(int64_t v, int from, int to)
{
	if (to == TB)
		return v != 0;
	(void)from;
	return norm(to, v);
}

/* ---- the concrete interpreter (checks before a proof, fits) ---- */

static struct {
	int64_t *v;		/* per symbol */
	int64_t **arr;		/* local arrays */
	int ub;			/* undefined behaviour met */
	long steps;		/* a run's budget */
	int *bind;		/* expression -> slot of bv, -1 */
	int64_t bv[16];
} C;

static int64_t ce(int x);
static uint8_t *expd;

static int64_t
ce_call(const struct gx *e)
{
	int64_t a[MAXP + 8], r = 0;
	int i, f = e->sym;
	int fl;
	for (i = 0; i < e->b && i < MAXP + 8; i++)
		a[i] = ce(G.xarg[e->a + i]);
	if (f < 0) {
		int ty = e->ty;
		switch (-f) {
		case B_MIN:
			return e->b >= 2 ? (a[0] < a[1] ? a[0] : a[1]) : 0;
		case B_MAX:
			return e->b >= 2 ? (a[0] > a[1] ? a[0] : a[1]) : 0;
		case B_CLAMP:
			if (e->b < 3)
				return 0;
			r = a[0] > a[1] ? a[0] : a[1];
			return r < a[2] ? r : a[2];
		case B_ABS:
			return norm(ty, a[0] < 0 ? -a[0] : a[0]);
		case B_INT:
		case B_UINT:
		case B_BOOL:
			return conv(a[0], G.x[G.xarg[e->a]].ty, ty);
		case B_BFE: {
			/* GLSL's: bits [off, off + n) of v, sign-extended for an
			 * int; undefined past the word */
			int64_t off = a[1], n = a[2];
			uint32_t v = (uint32_t)a[0], m;
			if (off < 0 || n < 0 || off + n > 32) {
				C.ub = 1;
				return 0;
			}
			if (n == 0)
				return 0;
			m = n == 32 ? 0xFFFFFFFFu : (1u << n) - 1u;
			v = (v >> off) & m;
			if (ty == TI && n < 32 && ((v >> (n - 1)) & 1u))
				v |= ~m;
			return norm(ty, (int64_t)v);
		}
		}
		C.ub = 1;
		return 0;
	}
	if (G.f[f].body < 0) {
		C.ub = 1;
		return 0;
	}
	for (i = 0; i < G.f[f].np && i < e->b; i++)
		C.v[G.f[f].p[i]] = conv(a[i], G.x[G.xarg[e->a + i]].ty,
		    G.y[G.f[f].p[i]].ty);
	{
		extern int cx_stmt(int s);
		fl = cx_stmt(G.f[f].body);
	}
	(void)fl;
	return G.f[f].ret >= 0 ? C.v[G.f[f].ret] : 0;
}

static void
c_store(int lhs, int64_t v)
{
	const struct gx *l = &G.x[XR(lhs)];
	if (l->k == X_VAR) {
		C.v[l->sym] = conv(v, TI, G.y[l->sym].ty);
		return;
	}
	if (l->k == X_IDX && G.x[l->a].k == X_VAR) {
		int y = G.x[l->a].sym;
		int64_t i = ce(l->b);
		if (C.arr && C.arr[y] && i >= 0 && i < G.y[y].arr)
			C.arr[y][i] = conv(v, TI, G.y[y].ty);
		else
			C.ub = 1;
		return;
	}
	C.ub = 1;
}

static int64_t
ce(int x)
{
	const struct gx *e;
	int64_t a, b;
	x = XR(x);
	if (C.bind && C.bind[x] >= 0)
		return C.bv[C.bind[x]];
	e = &G.x[x];
	switch (e->k) {
	case X_NUM:
		return e->v;
	case X_VAR:
		if ((G.y[e->sym].kind == Y_CONST || (expd && expd[e->sym])) &&
		    G.y[e->sym].init >= 0 && !G.y[e->sym].arr)
			return conv(ce(G.y[e->sym].init), G.x[G.y[e->sym].init].ty,
			    G.y[e->sym].ty);
		return C.v[e->sym];
	case X_IDX: {
		int y = G.x[e->a].k == X_VAR ? G.x[e->a].sym : -1;
		int64_t i = ce(e->b);
		if (y >= 0 && G.y[y].tab && i >= 0 && i < G.y[y].arr)
			return norm(G.y[y].ty, G.y[y].tab[i]);
		if (y >= 0 && C.arr && C.arr[y] && i >= 0 && i < G.y[y].arr)
			return C.arr[y][i];
		C.ub = 1;
		return 0;
	}
	case X_CALL:
		return ce_call(e);
	case X_UN:
		a = ce(e->a);
		if (e->op == OP_NEG)
			return norm(e->ty, -a);
		if (e->op == OP_NOT)
			return norm(e->ty, ~a);
		return !a;
	case X_BIN: {
		int t = opnd_ty(e);
		if (e->op == OP_LAND) {
			if (!ce(e->a))
				return 0;
			return ce(e->b) != 0;
		}
		if (e->op == OP_LOR) {
			if (ce(e->a))
				return 1;
			return ce(e->b) != 0;
		}
		a = conv(ce(e->a), G.x[e->a].ty, t);
		b = ce(e->b);
		if (e->op != OP_SHL && e->op != OP_SHR)
			b = conv(b, G.x[e->b].ty, t);
		return cop(e->op, t, a, b, &C.ub);
	}
	case X_SEL:
		return ce(e->a) ? conv(ce(e->b), G.x[e->b].ty, e->ty) :
		    conv(ce(e->c), G.x[e->c].ty, e->ty);
	case X_ASG: {
		int64_t v = ce(e->b);
		if (e->op) {
			int t = e->op == OP_SHL || e->op == OP_SHR ? G.x[e->a].ty :
			    bin_ty(OP_ADD, G.x[e->a].ty, G.x[e->b].ty);
			int64_t cur = ce(e->a);
			v = cop(e->op, t, conv(cur, G.x[e->a].ty, t), e->op == OP_SHL ||
			    e->op == OP_SHR ? v : conv(v, G.x[e->b].ty, t), &C.ub);
		}
		v = conv(v, G.x[e->b].ty, G.x[e->a].ty);
		c_store(e->a, v);
		return v;
	}
	case X_INC: {
		int64_t cur = ce(e->a), nv = norm(G.x[e->a].ty,
		    e->op == OP_ADD ? cur + 1 : cur - 1);
		c_store(e->a, nv);
		return e->post ? cur : nv;
	}
	case X_COMMA:
		(void)ce(e->a);
		return ce(e->b);
	}
	C.ub = 1;
	return 0;
}

/* a statement: 0 on, 1 break, 2 continue, 3 return */
int cx_stmt(int s);
int
cx_stmt(int s)
{
	const struct gs *t;
	int i, r;
	s = SR(s);
	if (s < 0 || C.ub)
		return 0;
	t = &G.s[s];
	if (++C.steps > 2000000) {
		C.ub = 1;
		return 3;
	}
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			if ((r = cx_stmt(G.slist[t->first + i])) != 0)
				return r;
		return 0;
	case S_DECL:
		if (G.y[t->sym].arr) {
			if (C.arr) {
				free(C.arr[t->sym]);
				C.arr[t->sym] = xcalloc((size_t)G.y[t->sym].arr,
				    sizeof (int64_t));
			}
			return 0;
		}
		C.v[t->sym] = t->a >= 0 ? conv(ce(t->a), G.x[XR(t->a)].ty,
		    G.y[t->sym].ty) : 0;
		return 0;
	case S_EXPR:
		(void)ce(t->a);
		return 0;
	case S_IF:
		if (ce(t->a))
			return cx_stmt(t->b);
		return t->c >= 0 ? cx_stmt(t->c) : 0;
	case S_FOR:
	case S_WHILE:
	case S_DO:
		if (t->k == S_FOR && t->a >= 0)
			(void)cx_stmt(t->a);
		for (;;) {
			if (t->k != S_DO && t->b >= 0 && !ce(t->b))
				return 0;
			if (C.ub)
				return 3;
			r = cx_stmt(t->d);
			if (r == 1)
				return 0;
			if (r == 3)
				return 3;
			if (t->k == S_FOR && t->c >= 0)
				(void)ce(t->c);
			if (t->k == S_DO && !ce(t->b))
				return 0;
			if (++C.steps > 2000000) {
				C.ub = 1;
				return 3;
			}
		}
	case S_RET:
		if (t->a >= 0 && t->sym >= 0)
			C.v[t->sym] = conv(ce(t->a), G.x[XR(t->a)].ty, G.y[t->sym].ty);
		return 3;
	case S_BRK:
		return 1;
	case S_CONT:
		return 2;
	case S_EMPTY:
		return 0;
	}
	C.ub = 1;
	return 3;
}


/* ---- abstract values: an interval, the bits that may be one, a bound ---- */

#define BIGLO (INT64_MIN / 4)
#define BIGHI (INT64_MAX / 4)

struct av {
	int64_t lo, hi;
	uint32_t may;		/* bits of the 32-bit pattern that may be one */
	int32_t sh;		/* a symbol whose value bounds this from above */
	uint8_t bot, wr;	/* no value; it wrapped */
};

static void
trange(int ty, int64_t *lo, int64_t *hi)
{
	switch (ty) {
	case TU: *lo = 0; *hi = 0xFFFFFFFFLL; return;
	case TI: *lo = INT32_MIN; *hi = INT32_MAX; return;
	case TB: *lo = 0; *hi = 1; return;
	default: *lo = BIGLO; *hi = BIGHI; return;
	}
}

static uint32_t
mask_upto(int64_t hi)
{
	uint32_t m;
	if (hi <= 0)
		return 0;
	if (hi > 0xFFFFFFFFLL)
		return 0xFFFFFFFFu;
	m = (uint32_t)hi;
	m |= m >> 1;
	m |= m >> 2;
	m |= m >> 4;
	m |= m >> 8;
	m |= m >> 16;
	return m;
}

static struct av
av_top(int ty)
{
	struct av r;
	memset(&r, 0, sizeof r);
	trange(ty, &r.lo, &r.hi);
	r.may = ty == TB ? 1 : 0xFFFFFFFFu;
	r.sh = -1;
	return r;
}

static struct av
av_bot(void)
{
	struct av r;
	memset(&r, 0, sizeof r);
	r.bot = 1;
	r.sh = -1;
	return r;
}

static struct av
av_k(int ty, int64_t v)
{
	struct av r;
	if (ty == TX || ty == TF)
		return av_top(ty);
	memset(&r, 0, sizeof r);
	r.lo = r.hi = v;
	r.may = (uint32_t)v;
	r.sh = -1;
	return r;
}

/* clipped to the type (out of it: it wrapped, any value), bits applied */
static struct av
av_fix(int ty, struct av r)
{
	int64_t tl, th;
	if (r.bot)
		return r;
	if (ty == TX || ty == TF)
		return av_top(ty);
	if (r.lo > r.hi)
		return av_bot();
	trange(ty, &tl, &th);
	if (r.lo < tl || r.hi > th) {
		struct av t = av_top(ty);
		t.wr = 1;
		return t;
	}
	if (r.lo >= 0) {
		r.may &= mask_upto(r.hi);
		if ((int64_t)r.may < r.hi)
			r.hi = (int64_t)r.may;
		if (r.lo > r.hi)
			return av_bot();
	} else
		r.may = 0xFFFFFFFFu;
	return r;
}

static struct av
av_iv(int ty, int64_t lo, int64_t hi)
{
	struct av r;
	memset(&r, 0, sizeof r);
	r.lo = lo;
	r.hi = hi;
	r.may = 0xFFFFFFFFu;
	r.sh = -1;
	return av_fix(ty, r);
}

static struct av
av_join(struct av a, struct av b)
{
	if (a.bot)
		return b;
	if (b.bot)
		return a;
	a.lo = a.lo < b.lo ? a.lo : b.lo;
	a.hi = a.hi > b.hi ? a.hi : b.hi;
	a.may |= b.may;
	a.wr |= b.wr;
	if (a.sh != b.sh)
		a.sh = -1;
	return a;
}

static struct av
av_meet(int ty, struct av a, struct av b)
{
	if (a.bot || b.bot)
		return av_bot();
	a.lo = a.lo > b.lo ? a.lo : b.lo;
	a.hi = a.hi < b.hi ? a.hi : b.hi;
	a.may &= b.may;
	if (a.sh < 0)
		a.sh = b.sh;
	if (a.lo > a.hi)
		return av_bot();
	return av_fix(ty, a);
}

static int
av_same(const struct av *a, const struct av *b)
{
	if (a->bot || b->bot)
		return a->bot == b->bot;
	return a->lo == b->lo && a->hi == b->hi && a->may == b->may &&
	    a->sh == b->sh;
}

static int64_t
mul_sat(int64_t a, int64_t b)
{
	double p = (double)a * (double)b;
	if (p > 4e18)
		return BIGHI;
	if (p < -4e18)
		return BIGLO;
	return a * b;
}

static struct av
av_conv(struct av a, int from, int to)
{
	if (a.bot)
		return a;
	if (to == TX || to == TF || from == TX || from == TF)
		return av_top(to);
	if (to == TB) {
		if (a.lo == 0 && a.hi == 0)
			return av_k(TB, 0);
		if (a.lo > 0 || a.hi < 0)
			return av_k(TB, 1);
		return av_top(TB);
	}
	if (to == TU && a.hi < 0) {
		a.lo += 0x100000000LL;
		a.hi += 0x100000000LL;
		a.may = 0xFFFFFFFFu;
		a.sh = -1;
		return av_fix(TU, a);
	}
	if (to == TI && a.lo > INT32_MAX) {
		a.lo -= 0x100000000LL;
		a.hi -= 0x100000000LL;
		a.may = 0xFFFFFFFFu;
		a.sh = -1;
		return av_fix(TI, a);
	}
	{
		struct av r = av_fix(to, a);
		if (r.wr)
			r.wr = 0;	/* a conversion wraps by definition */
		if (from != to && (a.lo < 0 || a.hi > INT32_MAX))
			r.sh = -1;
		return r;
	}
}

static struct av
av_bin(int op, int ot, int rt, struct av a, struct av b)
{
	struct av r;
	int64_t c[4], lo, hi;
	int i;
	if (a.bot || b.bot)
		return av_bot();
	if (ot == TX || ot == TF || rt == TX || rt == TF)
		return av_top(rt);
	switch (op) {
	case OP_ADD:
		return av_iv(rt, a.lo + b.lo, a.hi + b.hi);
	case OP_SUB:
		return av_iv(rt, a.lo - b.hi, a.hi - b.lo);
	case OP_MUL:
		c[0] = mul_sat(a.lo, b.lo);
		c[1] = mul_sat(a.lo, b.hi);
		c[2] = mul_sat(a.hi, b.lo);
		c[3] = mul_sat(a.hi, b.hi);
		lo = hi = c[0];
		for (i = 1; i < 4; i++) {
			lo = c[i] < lo ? c[i] : lo;
			hi = c[i] > hi ? c[i] : hi;
		}
		return av_iv(rt, lo, hi);
	case OP_DIV:
	case OP_MOD:
		if (b.lo == 0 && b.hi == 0)
			return av_bot();	/* undefined everywhere */
		if (b.lo == 0)
			b.lo = 1;
		if (b.hi == 0)
			b.hi = -1;
		if (op == OP_MOD) {
			int64_t m = b.hi > -b.lo ? b.hi : -b.lo;
			if (a.lo >= 0 && b.lo > 0) {
				if (a.hi < b.lo)
					return av_iv(rt, a.lo, a.hi);
				return av_iv(rt, 0, a.hi < m - 1 ? a.hi : m - 1);
			}
			lo = a.lo >= 0 ? 0 : -(m - 1);
			hi = a.hi <= 0 ? 0 : m - 1;
			return av_iv(rt, lo, hi);
		}
		if (b.lo < 0 && b.hi > 0) {
			int64_t m = a.hi > -a.lo ? a.hi : -a.lo;
			return av_iv(rt, -m, m);
		}
		c[0] = a.lo / b.lo;
		c[1] = a.lo / b.hi;
		c[2] = a.hi / b.lo;
		c[3] = a.hi / b.hi;
		lo = hi = c[0];
		for (i = 1; i < 4; i++) {
			lo = c[i] < lo ? c[i] : lo;
			hi = c[i] > hi ? c[i] : hi;
		}
		return av_iv(rt, lo, hi);
	case OP_SHL:
	case OP_SHR: {
		int64_t s0 = b.lo < 0 ? 0 : b.lo, s1 = b.hi > 31 ? 31 : b.hi, s;
		uint32_t m = 0;
		if (s0 > s1)
			return av_bot();	/* every shift amount undefined */
		for (s = s0; s <= s1; s++)
			m |= op == OP_SHL ? a.may << s : a.may >> s;
		if (op == OP_SHL) {
			if (a.lo >= 0 && (a.hi << s1) <= (rt == TI ? INT32_MAX :
			    0xFFFFFFFFLL)) {
				r = av_iv(rt, a.lo << s0, a.hi << s1);
				r.may &= m;
				return av_fix(rt, r);
			}
			if (rt == TU) {
				r = av_iv(rt, 0, 0xFFFFFFFFLL);
				r.may = m;
				return av_fix(rt, r);
			}
			return av_top(rt);
		}
		if (a.lo >= 0) {
			r = av_iv(rt, a.lo >> s1, a.hi >> s0);
			r.may &= m;
			return av_fix(rt, r);
		}
		/* an arithmetic shift of a negative value */
		lo = a.lo >> s0;
		hi = a.hi >= 0 ? a.hi >> s0 : a.hi >> s1;
		return av_iv(rt, lo, hi);
	}
	case OP_AND:
	case OP_OR:
	case OP_XOR: {
		uint32_t m = op == OP_AND ? a.may & b.may : a.may | b.may;
		if (op == OP_AND && (a.lo >= 0 || b.lo >= 0)) {
			hi = a.lo >= 0 && b.lo >= 0 ? (a.hi < b.hi ? a.hi : b.hi) :
			    a.lo >= 0 ? a.hi : b.hi;
			r = av_iv(rt, 0, hi);
			r.may &= m;
			return av_fix(rt, r);
		}
		if (a.lo >= 0 && b.lo >= 0) {
			r = av_iv(rt, op == OP_OR ? (a.lo > b.lo ? a.lo : b.lo) : 0,
			    (int64_t)mask_upto(a.hi > b.hi ? a.hi : b.hi));
			r.may &= m;
			return av_fix(rt, r);
		}
		if (rt == TU) {
			r = av_top(rt);
			r.may = m;
			return av_fix(rt, r);
		}
		return av_top(rt);
	}
	case OP_LT: case OP_LE: case OP_GT: case OP_GE:
	case OP_EQ: case OP_NE: {
		int t = 2;	/* 2: either */
		switch (op) {
		case OP_LT: t = a.hi < b.lo ? 1 : a.lo >= b.hi ? 0 : 2; break;
		case OP_LE: t = a.hi <= b.lo ? 1 : a.lo > b.hi ? 0 : 2; break;
		case OP_GT: t = a.lo > b.hi ? 1 : a.hi <= b.lo ? 0 : 2; break;
		case OP_GE: t = a.lo >= b.hi ? 1 : a.hi < b.lo ? 0 : 2; break;
		case OP_EQ:
			t = a.lo == a.hi && b.lo == b.hi && a.lo == b.lo ? 1 :
			    a.hi < b.lo || b.hi < a.lo ? 0 : 2;
			break;
		case OP_NE:
			t = a.lo == a.hi && b.lo == b.hi && a.lo == b.lo ? 0 :
			    a.hi < b.lo || b.hi < a.lo ? 1 : 2;
			break;
		}
		/* bits: a value with a bit b cannot have is not b */
		if (t == 2 && (op == OP_EQ || op == OP_NE) && b.lo == b.hi &&
		    ((uint32_t)b.lo & ~a.may) != 0)
			t = op == OP_NE;
		if (t == 2 && (op == OP_EQ || op == OP_NE) && a.lo == a.hi &&
		    ((uint32_t)a.lo & ~b.may) != 0)
			t = op == OP_NE;
		return t == 2 ? av_top(TB) : av_k(TB, t);
	}
	case OP_LAND:
		if ((a.lo == 0 && a.hi == 0) || (b.lo == 0 && b.hi == 0))
			return av_k(TB, 0);
		if (a.lo == 1 && b.lo == 1)
			return av_k(TB, 1);
		return av_top(TB);
	case OP_LOR:
		if (a.lo == 1 || b.lo == 1)
			return av_k(TB, 1);
		if (a.hi == 0 && b.hi == 0)
			return av_k(TB, 0);
		return av_top(TB);
	case OP_LXOR:
		if (a.lo == a.hi && b.lo == b.hi)
			return av_k(TB, a.lo != b.lo);
		return av_top(TB);
	}
	return av_top(rt);
}

/* ---- environments and their sets ---- */

typedef struct av *env_t;
struct eset {
	int n, cap;
	env_t *e;
};

static struct {
	int ns;			/* symbols */
	int *keys, nkeys;
	uint8_t *iskey;
	int *nasg;		/* assignments outside the declaration */
	uint8_t **rd, **wr;	/* per function: globals read, written */
	uint8_t *impure;	/* writes a buffer, an out parameter */
	int **rk, *nrk;		/* per function: the keys among its inputs */
	int **in, *nin;		/* per function: parameters and globals read */
	int cur;		/* the function being run */
	int **loc, *nloc;	/* per function: its locals and parameters */
	int **wrl, *nwrl;	/* per function: the globals it writes */
	uint8_t *inl;		/* per function: loop-free, run at each call */
	int changed;		/* a summary's entry grew this round */
	int recording;
	long work;
	double deadline;
	int out_of_time;
	int site;		/* the call whose own values a loop-free callee
				 * runs on (its expression), -1 none */
	int probe_round;	/* the round that runs loops' passes apart */
	int probe_grew;		/* a summary of the analysis grew in it */
	uint8_t *reach;		/* a probed pass: the statements it runs */
} A;

/*
 * A loop's first two passes, run apart from its fixed point (probe_loop),
 * over every state the loop starts from: whether the second pass ever
 * comes back to the test, where each pass leaves, what each one runs.
 */
static struct lprobe {
	uint8_t seen;		/* probed */
	uint8_t ran1, at2;	/* a first pass ran; one came back to the test */
	uint8_t ran2, more;	/* a second pass ran; one came back to the test */
	uint8_t brk1;		/* a first pass left by a break */
	uint8_t fail1, fail2;	/* the test failed as a pass started */
	struct av v1, v2;	/* the loop's variable as each pass starts */
	uint8_t *r1, *r2;	/* per statement: run in the first, second pass */
	struct av *entry;	/* per symbol: as the loop starts, joined */
} *LP;			/* per statement; NULL when no probe round ran */
static int lp_note;	/* the probe round: 0 ran, 1 out of time, 2 the
			 * analysis did not settle, 3 a summary grew in it */
static int *lp_n0;	/* per function: its summaries before the round */

static void
lp_free(void)
{
	int s;
	if (LP)
		for (s = 0; s <= G.ns; s++) {
			free(LP[s].r1);
			free(LP[s].r2);
			free(LP[s].entry);
		}
	free(LP);
	LP = NULL;
}

static env_t
env_dup(const struct av *e)
{
	env_t d = xmalloc((size_t)A.ns * sizeof *d);
	memcpy(d, e, (size_t)A.ns * sizeof *d);
	return d;
}

static int
env_samekey(const struct av *a, const struct av *b)
{
	int i;
	for (i = 0; i < A.nkeys; i++) {
		const struct av *x = &a[A.keys[i]], *y = &b[A.keys[i]];
		if (x->bot != y->bot || (!x->bot && (x->lo != y->lo || x->hi != y->hi)))
			return 0;
	}
	return 1;
}

/* d joined with s; widen: a bound that moves goes to the type's end */
static int
env_join_into(env_t d, const struct av *s, int widen)
{
	int i, ch = 0;
	for (i = 0; i < A.ns; i++) {
		struct av j = av_join(d[i], s[i]);
		if (av_same(&j, &d[i]))
			continue;
		if (widen && !d[i].bot && !A.iskey[i]) {
			int64_t tl, th;
			int ty = G.y[i].ty;
			trange(ty, &tl, &th);
			if (j.lo < d[i].lo)
				j.lo = tl;
			if (j.hi > d[i].hi)
				j.hi = th;
			j = av_fix(ty, j);
			j.wr = 0;
		}
		d[i] = j;
		ch = 1;
	}
	return ch;
}

static void
es_free(struct eset *S)
{
	int i;
	for (i = 0; i < S->n; i++)
		free(S->e[i]);
	free(S->e);
	memset(S, 0, sizeof *S);
}

/* e joins the set (the set owns it); 1 when the set changed */
static int
es_add(struct eset *S, env_t e, int widen)
{
	int i, ch;
	for (i = 0; i < S->n; i++)
		if (env_samekey(S->e[i], e)) {
			ch = env_join_into(S->e[i], e, widen);
			free(e);
			return ch;
		}
	if (S->n == S->cap) {
		S->cap = S->cap ? S->cap * 2 : 8;
		S->e = xrealloc(S->e, (size_t)S->cap * sizeof *S->e);
	}
	S->e[S->n++] = e;
	if (S->n > 1024) {
		/* too many: one environment, keys widened */
		env_t one = S->e[0];
		int k;
		for (k = 1; k < S->n; k++) {
			for (i = 0; i < A.ns; i++)
				one[i] = av_join(one[i], S->e[k][i]);
			free(S->e[k]);
		}
		S->n = 1;
	}
	return 1;
}

static void
es_move(struct eset *d, struct eset *s)
{
	int i;
	for (i = 0; i < s->n; i++)
		es_add(d, s->e[i], 0);
	free(s->e);
	memset(s, 0, sizeof *s);
}

/* ---- per-function summaries ---- */

struct summ {
	int64_t *kv;		/* the keys' bounds, two each */
	env_t entry;
	struct eset exit;
	uint8_t valid;
};
static struct fsum {
	struct summ *s;
	int n, cap;
} *FS;

/* ---- records: what an expression's leaves were, for the proofs ---- */

/* the calls a row was met at (the call expressions), up to 256, more is
 * any; a row met outside a call run on its own values is any call */
#define RSITES 256
struct rsite {
	int n, cap;		/* n -1: any call */
	int *s;
};
struct rec {
	int x;			/* the expression */
	int nl, lf[MAXP];	/* its leaves (expressions) */
	int nex, ex[MAXP];	/* symbols recorded too */
	int nbox, cap;
	struct av *box;		/* nbox rows of nl + nex */
	struct rsite *site;	/* per row */
};
static struct rec *RC;
static int nrc, caprc, *recn;	/* expression -> record, -1 */

static struct av ae(env_t e, int x);

/* row b of R was met at the call A.site */
static void
rsite_add(struct rsite *t, int site)
{
	int i;
	if (t->n < 0)
		return;
	if (site < 0) {
		t->n = -1;
		return;
	}
	for (i = 0; i < t->n; i++)
		if (t->s[i] == site)
			return;
	if (t->n == RSITES) {
		t->n = -1;
		return;
	}
	if (t->n == t->cap) {
		t->cap = t->cap ? t->cap * 2 : 2;
		t->s = xrealloc(t->s, (size_t)t->cap * sizeof *t->s);
	}
	t->s[t->n++] = site;
}

/* row b of R may have been met at call site c */
static int
rsite_has(const struct rec *R, int b, int c)
{
	const struct rsite *t = &R->site[b];
	int i;
	if (t->n < 0)
		return 1;
	for (i = 0; i < t->n; i++)
		if (t->s[i] == c)
			return 1;
	return 0;
}

static void
rec_take(env_t e, int r)
{
	struct rec *R = &RC[r];
	struct av row[2 * MAXP];
	int w = R->nl + R->nex, i, b;
	for (i = 0; i < R->nl; i++)
		row[i] = ae(e, R->lf[i]);
	for (i = 0; i < R->nex; i++)
		row[R->nl + i] = e[R->ex[i]];
	for (i = 0; i < w; i++)
		row[i].sh = -1;
	for (b = 0; b < R->nbox; b++) {
		int same = 1;
		for (i = 0; i < w && same; i++)
			same = av_same(&R->box[b * w + i], &row[i]);
		if (same) {
			rsite_add(&R->site[b], A.site);
			return;
		}
	}
	if (R->nbox >= 256) {
		for (i = 0; i < w; i++)
			R->box[(R->nbox - 1) * w + i] = av_join(R->box[(R->nbox - 1) *
			    w + i], row[i]);
		rsite_add(&R->site[R->nbox - 1], A.site);
		return;
	}
	if (R->nbox == R->cap) {
		int c0 = R->cap;
		R->cap = R->cap ? R->cap * 2 : 8;
		R->box = xrealloc(R->box, (size_t)R->cap * w * sizeof *R->box);
		R->site = xrealloc(R->site, (size_t)R->cap * sizeof *R->site);
		memset(R->site + c0, 0, (size_t)(R->cap - c0) * sizeof *R->site);
	}
	memcpy(&R->box[R->nbox * w], row, (size_t)w * sizeof *row);
	R->site[R->nbox].n = 0;
	rsite_add(&R->site[R->nbox], A.site);
	R->nbox++;
}

/* ---- the abstract evaluation of an expression in one environment ---- */

static void call_fn(env_t e, int f, struct av *args, int na, struct av *ret);
static struct av ae_fine(env_t e, int x);

static int
same_expr(int a, int b)
{
	const struct gx *p = &G.x[a], *q = &G.x[b];
	if (p->k != q->k || p->op != q->op || p->sym != q->sym)
		return 0;
	if (p->k == X_NUM)
		return p->v == q->v;
	if (p->k == X_VAR)
		return 1;
	if (p->k == X_BIN)
		return same_expr(p->a, q->a) && same_expr(p->b, q->b);
	if (p->k == X_UN)
		return same_expr(p->a, q->a);
	return 0;
}

static void
store(env_t e, int lhs, struct av v)
{
	const struct gx *l = &G.x[lhs];
	int y, i;
	if (l->k == X_VAR) {
		y = l->sym;
		if (G.y[y].kind == Y_BUF || G.y[y].kind == Y_CONST)
			return;
		if (v.sh == y || G.y[y].fn != A.cur || G.y[y].kind != Y_LOCAL)
			v.sh = -1;
		if (G.y[y].arr)
			e[y] = av_join(e[y], v);
		else
			e[y] = v;
		if (A.cur >= 0 && G.y[y].fn == A.cur)
			for (i = 0; i < A.nloc[A.cur]; i++)
				if (e[A.loc[A.cur][i]].sh == y)
					e[A.loc[A.cur][i]].sh = -1;
		return;
	}
	if (l->k == X_IDX && G.x[l->a].k == X_VAR) {
		y = G.x[l->a].sym;
		if (G.y[y].kind == Y_BUF || G.y[y].kind == Y_CONST)
			return;
		v.sh = -1;
		if (G.y[y].arr)
			e[y] = e[y].bot ? v : av_join(e[y], v);
		else
			e[y] = av_top(G.y[y].ty);	/* a vector's component */
	}
}

struct undo {
	int y;
	struct av v;
};

static int refine(env_t e, int x, int t, struct undo *u, int *nu, int max);

static void
undo_all(env_t e, struct undo *u, int nu)
{
	while (nu-- > 0)
		e[u[nu].y] = u[nu].v;
}

static struct av
ae(env_t e, int x)
{
	const struct gx *g = &G.x[x];
	struct av a, b, r;
	int i;

	A.work++;
	if (A.recording && recn[x] >= 0)
		rec_take(e, recn[x]);
	switch (g->k) {
	case X_NUM:
		return av_k(g->ty, g->v);
	case X_VAR: {
		const struct gsym *y = &G.y[g->sym];
		if (y->kind == Y_CONST && y->init >= 0 && !y->arr)
			return av_conv(ae(e, y->init), G.x[y->init].ty, y->ty);
		if (y->kind == Y_BUF || y->kind == Y_CONST)
			return av_top(y->arr ? TX : y->ty);
		if (y->arr)
			return av_top(TX);
		r = e[g->sym];
		if (r.bot)	/* not yet assigned on this path */
			return av_top(y->ty);
		return r;
	}
	case X_IDX: {
		int y = G.x[g->a].k == X_VAR ? G.x[g->a].sym : -1;
		struct av ix = ae(e, g->b);
		if (y < 0 || g->ty == TX || g->ty == TF)
			return av_top(g->ty);
		if (G.y[y].tab) {
			int64_t lo = ix.lo < 0 ? 0 : ix.lo, hi = ix.hi >=
			    G.y[y].arr ? G.y[y].arr - 1 : ix.hi, k;
			if (ix.bot || lo > hi || hi - lo > 4096)
				return av_top(g->ty);
			r = av_k(g->ty, norm(g->ty, G.y[y].tab[lo]));
			for (k = lo + 1; k <= hi; k++)
				r = av_join(r, av_k(g->ty, norm(g->ty, G.y[y].tab[k])));
			return r;
		}
		if (G.y[y].kind == Y_BUF || !G.y[y].arr)
			return av_top(g->ty);
		r = e[y];
		return r.bot ? av_top(g->ty) : av_fix(g->ty, r);
	}
	case X_MEM:
		return av_top(g->ty);
	case X_CALL: {
		struct av args[MAXP + 8];
		int na = g->b < MAXP + 8 ? g->b : MAXP + 8;
		for (i = 0; i < na; i++)
			args[i] = ae(e, G.xarg[g->a + i]);
		if (g->sym >= 0) {
			int f = g->sym;
			int sv = A.site;
			for (i = 0; i < na && i < G.f[f].np; i++)
				args[i] = av_conv(args[i], G.x[G.xarg[g->a + i]].ty,
				    G.y[G.f[f].p[i]].ty);
			A.site = x;
			call_fn(e, f, args, na, &r);
			A.site = sv;
			return r;
		}
		for (i = 0; i < na; i++)
			args[i] = av_conv(args[i], G.x[G.xarg[g->a + i]].ty, g->ty);
		switch (-g->sym) {
		case B_MIN:
		case B_MAX:
		case B_CLAMP:
			if (na < 2 || args[0].bot || args[1].bot)
				return av_top(g->ty);
			r = args[0];
			if (-g->sym != B_MIN) {	/* max(x, lo) */
				r.lo = r.lo > args[1].lo ? r.lo : args[1].lo;
				r.hi = r.hi > args[1].hi ? r.hi : args[1].hi;
				r.may |= args[1].may;
			} else {
				r.lo = r.lo < args[1].lo ? r.lo : args[1].lo;
				r.hi = r.hi < args[1].hi ? r.hi : args[1].hi;
				r.may |= args[1].may;
			}
			if (-g->sym == B_CLAMP && na >= 3 && !args[2].bot) {
				r.lo = r.lo < args[2].lo ? r.lo : args[2].lo;
				r.hi = r.hi < args[2].hi ? r.hi : args[2].hi;
				r.may |= args[2].may;
			}
			r.sh = -1;
			{
				/* min(x, y), clamp(x, lo, y): at most y */
				int hx = G.xarg[g->a + (-g->sym == B_CLAMP ? 2 : 1)];
				if (-g->sym != B_MAX && G.x[hx].k == X_VAR &&
				    G.y[G.x[hx].sym].kind != Y_BUF && !G.y[G.x[hx].sym].arr)
					r.sh = G.x[hx].sym;
			}
			r.may = 0xFFFFFFFFu;
			return av_fix(g->ty, r);
		case B_ABS:
			if (na < 1 || args[0].bot)
				return av_top(g->ty);
			a = args[0];
			if (a.lo >= 0)
				return a;
			if (a.hi <= 0)
				return av_iv(g->ty, -a.hi, -a.lo);
			return av_iv(g->ty, 0, -a.lo > a.hi ? -a.lo : a.hi);
		case B_INT:
		case B_UINT:
		case B_BOOL:
			if (na < 1)
				return av_top(g->ty);
			return av_conv(ae(e, G.xarg[g->a]), G.x[G.xarg[g->a]].ty, g->ty);
		case B_BFE: {
			/* n bits: 0..2^n - 1, or -2^(n-1)..2^(n-1) - 1 for an int;
			 * an unsigned one of a value below 2^(off + n) is that value
			 * shifted */
			int64_t n = args[2].hi;
			if (na != 3 || args[2].bot || args[2].lo < 0 || n > 32)
				return av_top(g->ty);
			if (n == 0)
				return av_k(g->ty, 0);
			if (n == 32)
				return args[0];
			if (g->ty == TI)
				return av_iv(g->ty, -(1LL << (n - 1)), (1LL << (n - 1)) - 1);
			if (!args[1].bot && args[1].lo == args[1].hi && !args[0].bot &&
			    args[0].lo >= 0 && args[0].hi < (1LL << (args[1].lo + n)))
				return av_bin(OP_SHR, TU, TU, args[0], args[1]);
			return av_iv(g->ty, 0, (1LL << n) - 1);
		}
		}
		return av_top(g->ty);
	}
	case X_UN:
		a = ae(e, g->a);
		if (a.bot)
			return a;
		if (g->op == OP_LNOT)
			return a.lo == a.hi ? av_k(TB, !a.lo) : av_top(TB);
		if (g->op == OP_NEG)
			return av_iv(g->ty, -a.hi, -a.lo);
		if (g->ty == TU)
			return av_iv(TU, 0xFFFFFFFFLL - a.hi, 0xFFFFFFFFLL - a.lo);
		return av_iv(g->ty, -a.hi - 1, -a.lo - 1);
	case X_BIN: {
		int ot = opnd_ty(g);
		if (g->op == OP_LAND || g->op == OP_LOR) {
			a = ae(e, g->a);
			if (g->op == OP_LAND && a.lo == 0 && a.hi == 0)
				return av_k(TB, 0);
			if (g->op == OP_LOR && a.lo == 1)
				return av_k(TB, 1);
			b = ae(e, g->b);
			return av_bin(g->op, TB, TB, a, b);
		}
		a = av_conv(ae(e, g->a), G.x[g->a].ty, ot);
		b = ae(e, g->b);
		if (g->op != OP_SHL && g->op != OP_SHR)
			b = av_conv(b, G.x[g->b].ty, ot);
		if (g->op == OP_MUL && same_expr(g->a, g->b) && !a.bot) {
			/* a square */
			int64_t m = a.hi > -a.lo ? a.hi : -a.lo;
			int64_t l = a.lo > 0 ? a.lo : a.hi < 0 ? -a.hi : 0;
			return av_iv(g->ty, mul_sat(l, l), mul_sat(m, m));
		}
		return av_bin(g->op, ot, g->ty, a, b);
	}
	case X_SEL: {
		struct undo u[32];
		int nu = 0;
		struct av c = ae(e, g->a), va = av_bot(), vb = av_bot();
		if (c.bot)
			return c;
		if (c.lo == c.hi)
			return av_conv(ae(e, c.lo ? g->b : g->c), G.x[c.lo ? g->b :
			    g->c].ty, g->ty);
		if (refine(e, g->a, 1, u, &nu, 32))
			va = av_conv(ae(e, g->b), G.x[g->b].ty, g->ty);
		undo_all(e, u, nu);
		nu = 0;
		if (refine(e, g->a, 0, u, &nu, 32))
			vb = av_conv(ae(e, g->c), G.x[g->c].ty, g->ty);
		undo_all(e, u, nu);
		return av_join(va, vb);
	}
	case X_ASG: {
		struct av v = ae_fine(e, g->b);
		if (g->op) {
			int ot = g->op == OP_SHL || g->op == OP_SHR ? G.x[g->a].ty :
			    bin_ty(OP_ADD, G.x[g->a].ty, G.x[g->b].ty);
			struct av cur = av_conv(ae(e, g->a), G.x[g->a].ty, ot);
			if (g->op != OP_SHL && g->op != OP_SHR)
				v = av_conv(v, G.x[g->b].ty, ot);
			v = av_bin(g->op, ot, ot, cur, v);
			v = av_conv(v, ot, G.x[g->a].ty);
		} else
			v = av_conv(v, G.x[g->b].ty, G.x[g->a].ty);
		store(e, g->a, v);
		return v;
	}
	case X_INC: {
		struct av cur = ae(e, g->a), nv;
		nv = av_bin(g->op, g->ty, g->ty, cur, av_k(g->ty, 1));
		store(e, g->a, nv);
		return g->post ? cur : nv;
	}
	case X_COMMA:
		(void)ae(e, g->a);
		return ae(e, g->b);
	}
	return av_top(g->ty);
}

/* narrow e where condition x is t; 0 when it cannot be */
static int
narrow(env_t e, int x, struct av to, struct undo *u, int *nu, int max)
{
	const struct gx *g = &G.x[x];
	if (g->k == X_CALL && (g->sym == -B_INT || g->sym == -B_UINT) &&
	    g->b == 1) {
		/* int(v), uint(v): v itself where the conversion keeps it */
		int a = G.xarg[g->a];
		if (to.lo >= 0 && to.hi <= INT32_MAX && G.x[a].ty != TB)
			return narrow(e, a, to, u, nu, max);
		return 1;
	}
	if (g->k != X_VAR)
		return 1;
	{
		int y = g->sym;
		struct av m;
		if (G.y[y].kind == Y_BUF || G.y[y].kind == Y_CONST || G.y[y].arr)
			return 1;
		if (e[y].bot)
			return 1;
		to.may = 0xFFFFFFFFu;
		to.sh = -1;
		m = av_meet(G.y[y].ty, e[y], to);
		if (m.bot)
			return 0;
		if (*nu < max) {
			u[*nu].y = y;
			u[*nu].v = e[y];
			(*nu)++;
			e[y] = m;
		}
		return 1;
	}
}

static int
refine(env_t e, int x, int t, struct undo *u, int *nu, int max)
{
	const struct gx *g = &G.x[x];
	struct av c = ae(e, x);
	if (c.bot)
		return 0;
	if (c.lo == c.hi)
		return c.lo == t;
	if (g->k == X_UN && g->op == OP_LNOT)
		return refine(e, g->a, !t, u, nu, max);
	if (g->k == X_VAR && G.y[g->sym].ty == TB)
		return narrow(e, x, av_k(TB, t), u, nu, max);
	if (g->k == X_BIN && ((g->op == OP_LAND && t) || (g->op == OP_LOR &&
	    !t)))
		return refine(e, g->a, t, u, nu, max) && refine(e, g->b, t, u, nu,
		    max);
	if (g->k == X_BIN && g->op >= OP_LT && g->op <= OP_NE) {
		int op = g->op, ot = opnd_ty(g);
		struct av a = av_conv(ae(e, g->a), G.x[g->a].ty, ot);
		struct av b = av_conv(ae(e, g->b), G.x[g->b].ty, ot);
		struct av na = a, nb = b;
		int64_t tl, th;
		if (a.bot || b.bot)
			return 0;
		if (!t)
			op = op == OP_LT ? OP_GE : op == OP_GE ? OP_LT : op == OP_LE ?
			    OP_GT : op == OP_GT ? OP_LE : op == OP_EQ ? OP_NE : OP_EQ;
		trange(ot, &tl, &th);
		switch (op) {
		case OP_LT:
			na.hi = a.hi < b.hi - 1 ? a.hi : b.hi - 1;
			nb.lo = b.lo > a.lo + 1 ? b.lo : a.lo + 1;
			break;
		case OP_LE:
			na.hi = a.hi < b.hi ? a.hi : b.hi;
			nb.lo = b.lo > a.lo ? b.lo : a.lo;
			break;
		case OP_GT:
			na.lo = a.lo > b.lo + 1 ? a.lo : b.lo + 1;
			nb.hi = b.hi < a.hi - 1 ? b.hi : a.hi - 1;
			break;
		case OP_GE:
			na.lo = a.lo > b.lo ? a.lo : b.lo;
			nb.hi = b.hi < a.hi ? b.hi : a.hi;
			break;
		case OP_EQ:
			na.lo = nb.lo = a.lo > b.lo ? a.lo : b.lo;
			na.hi = nb.hi = a.hi < b.hi ? a.hi : b.hi;
			break;
		case OP_NE:
			if (b.lo == b.hi) {
				if (na.lo == b.lo)
					na.lo++;
				if (na.hi == b.lo)
					na.hi--;
			}
			if (a.lo == a.hi) {
				if (nb.lo == a.lo)
					nb.lo++;
				if (nb.hi == a.lo)
					nb.hi--;
			}
			break;
		}
		if (na.lo > na.hi || nb.lo > nb.hi)
			return 0;
		/* only operands of the comparison's own type are narrowed */
		if (G.x[g->a].ty == ot && !narrow(e, g->a, na, u, nu, max))
			return 0;
		if (G.x[g->b].ty == ot && !narrow(e, g->b, nb, u, nu, max))
			return 0;
		(void)tl;
		(void)th;
		return 1;
	}
	return 1;
}

/* ---- an expression over every value of its leaves ---- */

/*
 * An expression whose leaves repeat ((b - a) / 16 + a, (t * 64 + dd / 2) /
 * dd) loses precision as intervals.  Its leaves are its variables, with a
 * single-assignment local that shares a leaf with the rest expanded into
 * its definition; when they take few values together, the expression is
 * run on every combination (the result is exact for independent leaves,
 * an over-approximation otherwise, so sound); when one leaf t has a bound
 * t <= u (clamp(x, 0, u)) and the expression grows with t, its largest
 * value is at t = u, run on every value of the others.
 */
struct fine {
	int nl, lf[6];		/* leaf symbols */
	uint8_t ok;
};
static struct fine *FN;		/* per expression, computed on demand */
static uint8_t *fn_done;

static int
simple_expr(int x)
{
	const struct gx *g = &G.x[x];
	int i;
	switch (g->k) {
	case X_NUM:
		return 1;
	case X_VAR:
		return !G.y[g->sym].arr && G.y[g->sym].kind != Y_BUF &&
		    G.y[g->sym].ty != TX && G.y[g->sym].ty != TF;
	case X_IDX:
		return G.x[g->a].k == X_VAR && G.y[G.x[g->a].sym].tab &&
		    simple_expr(g->b);
	case X_UN:
		return simple_expr(g->a);
	case X_BIN:
		return simple_expr(g->a) && simple_expr(g->b);
	case X_SEL:
		return simple_expr(g->a) && simple_expr(g->b) && simple_expr(g->c);
	case X_CALL:
		if (g->sym >= 0 || g->sym == -B_UNK)
			return 0;
		for (i = 0; i < g->b; i++)
			if (!simple_expr(G.xarg[g->a + i]))
				return 0;
		return 1;
	}
	return 0;
}

/* every variable under x a parameter or local never assigned again */
static int frz_fn = -1;
static int
frozen(int x, int depth)
{
	const struct gx *g = &G.x[x];
	int i;
	if (depth > 12)
		return 0;
	switch (g->k) {
	case X_NUM:
		return 1;
	case X_VAR:
		if (G.y[g->sym].kind == Y_CONST)
			return 1;
		/* a global the function (and what it calls) never writes */
		if (G.y[g->sym].kind == Y_GLOBAL && !G.y[g->sym].arr &&
		    frz_fn >= 0 && !A.wr[frz_fn][g->sym])
			return 1;
		return (G.y[g->sym].kind == Y_LOCAL || G.y[g->sym].kind ==
		    Y_PARAM) && !A.nasg[g->sym];
	case X_IDX:
		return frozen(g->b, depth);
	case X_UN:
		return frozen(g->a, depth);
	case X_BIN:
		return frozen(g->a, depth) && frozen(g->b, depth);
	case X_SEL:
		return frozen(g->a, depth) && frozen(g->b, depth) &&
		    frozen(g->c, depth);
	case X_CALL:
		for (i = 0; i < g->b; i++)
			if (!frozen(G.xarg[g->a + i], depth))
				return 0;
		return g->sym < 0;
	}
	return 0;
}

/* a single-assignment local whose definition can stand for it */
static int
expandable(int y)
{
	const struct gsym *s = &G.y[y];
	frz_fn = s->fn;
	return (s->kind == Y_LOCAL) && !s->arr && A.nasg[y] == 0 &&
	    s->init >= 0 && simple_expr(s->init) && frozen(s->init, 0);
}

/* the variables under x (expanded ones followed), counted */
static void
occ(int x, int *ys, int *cnt, int *n, int depth)
{
	const struct gx *g = &G.x[x];
	int i;
	if (depth > 12)
		return;
	switch (g->k) {
	case X_VAR:
		if (G.y[g->sym].kind == Y_CONST)
			return;
		if (expd[g->sym]) {
			occ(G.y[g->sym].init, ys, cnt, n, depth + 1);
			return;
		}
		for (i = 0; i < *n; i++)
			if (ys[i] == g->sym) {
				cnt[i]++;
				return;
			}
		if (*n < 16) {
			ys[*n] = g->sym;
			cnt[(*n)++] = 1;
		}
		return;
	case X_IDX:
		occ(g->b, ys, cnt, n, depth);
		return;
	case X_UN:
		occ(g->a, ys, cnt, n, depth);
		return;
	case X_BIN:
		occ(g->a, ys, cnt, n, depth);
		occ(g->b, ys, cnt, n, depth);
		return;
	case X_SEL:
		occ(g->a, ys, cnt, n, depth);
		occ(g->b, ys, cnt, n, depth);
		occ(g->c, ys, cnt, n, depth);
		return;
	case X_CALL:
		for (i = 0; i < g->b; i++)
			occ(G.xarg[g->a + i], ys, cnt, n, depth);
		return;
	}
}

static void
fine_info(int x)
{
	struct fine *F = &FN[x];
	int ys[16], cnt[16], n, i, j, rep, round;
	int exl[8], nexl = 0;
	fn_done[x] = 1;
	F->ok = 0;
	if (!simple_expr(x))
		return;
	for (round = 0; round < 4; round++) {
		int grew = 0;
		n = 0;
		occ(x, ys, cnt, &n, 0);
		for (i = 0; i < n && !grew; i++) {
			int ys2[16], cnt2[16], n2 = 0;
			if (!expandable(ys[i]))
				continue;
			/* its definition shares a leaf with the rest? */
			occ(G.y[ys[i]].init, ys2, cnt2, &n2, 0);
			for (j = 0; j < n2 && !grew; j++) {
				int k;
				for (k = 0; k < n; k++)
					if (k != i && ys[k] == ys2[j]) {
						expd[ys[i]] = 1;
						exl[nexl++] = ys[i];
						grew = 1;
						break;
					}
			}
		}
		if (!grew)
			break;
	}
	n = 0;
	occ(x, ys, cnt, &n, 0);
	rep = 0;
	for (i = 0; i < n; i++)
		if (cnt[i] > 1)
			rep = 1;
	if (rep && n <= 6) {
		F->ok = 1;
		F->nl = n;
		for (i = 0; i < n; i++)
			F->lf[i] = ys[i];
	}
	/* the expansions stay marked: ce() follows them (see ce_exp) */
	(void)exl;
	(void)nexl;
}

/* +1 grows with symbol y, -1 shrinks, 0 does not depend, 2 unknown */
static int
mono(env_t e, int x, int y)
{
	const struct gx *g = &G.x[x];
	int ma, mb;
	struct av v;
#define COMB(p, q) ((p) == 0 ? (q) : (q) == 0 ? (p) : (p) == (q) ? (p) : 2)
	switch (g->k) {
	case X_NUM:
		return 0;
	case X_VAR:
		if (g->sym == y)
			return 1;
		if (expd[g->sym])
			return mono(e, G.y[g->sym].init, y);
		return 0;
	case X_CALL:
		if ((g->sym == -B_INT || g->sym == -B_UINT) && g->b == 1) {
			v = ae(e, G.xarg[g->a]);
			if (v.bot || v.lo < 0 || v.hi > INT32_MAX)
				return 2;
			return mono(e, G.xarg[g->a], y);
		}
		return 2;
	case X_BIN:
		v = ae(e, x);
		if (v.wr)
			return 2;
		ma = mono(e, g->a, y);
		mb = mono(e, g->b, y);
		switch (g->op) {
		case OP_ADD:
			return COMB(ma, mb);
		case OP_SUB:
			return COMB(ma, mb == 2 ? 2 : -mb);
		case OP_MUL:
			if (mb == 0) {
				struct av w = ae(e, g->b);
				return w.lo >= 0 ? ma : w.hi <= 0 ? (ma == 2 ? 2 : -ma) : 2;
			}
			if (ma == 0) {
				struct av w = ae(e, g->a);
				return w.lo >= 0 ? mb : w.hi <= 0 ? (mb == 2 ? 2 : -mb) : 2;
			}
			return 2;
		case OP_DIV:
			if (mb == 0 && ae(e, g->b).lo > 0)
				return ma;
			return 2;
		case OP_SHR:
			return mb == 0 ? ma : 2;
		default:
			return ma == 0 && mb == 0 ? 0 : 2;
		}
	}
	return 2;
#undef COMB
}

static struct av
enum_leaves(env_t e, int x, const int *lf, const struct av *dom, int nl,
    int sub_from, int sub_to)
{
	int64_t cur[6];
	struct av r = av_bot();
	int i, ty = G.x[x].ty;
	long guard = 0;
	for (i = 0; i < nl; i++)
		cur[i] = dom[i].lo;
	for (;;) {
		int64_t v;
		for (i = 0; i < nl; i++)
			C.v[lf[i]] = cur[i];
		if (sub_from >= 0)
			C.v[sub_from] = C.v[sub_to];
		C.ub = 0;
		v = ce(x);
		if (!C.ub)
			r = av_join(r, av_k(ty, v));
		if (++guard > (1L << 22))
			return av_top(ty);
		for (i = 0; i < nl; i++) {
			if (cur[i] < dom[i].hi) {
				cur[i]++;
				break;
			}
			cur[i] = dom[i].lo;
		}
		if (i == nl)
			break;
	}
	(void)e;
	return r;
}

static int64_t *cvsave;

/* ae_fine's results, by expression and its leaves' bounds */
#define FMEMO 8192
static struct fmemo {
	int x, mode;
	int64_t b[12];
	struct av r;
	uint8_t used;
} *fmemo;

static struct fmemo *
fmemo_at(int x, const struct av *dom, int nl, int mode, int *hit)
{
	uint64_t h = (uint64_t)x * 0x9E3779B97F4A7C15ULL + (uint64_t)mode;
	int i, slot;
	struct fmemo *m;
	if (fmemo == NULL)
		fmemo = xcalloc(FMEMO, sizeof *fmemo);
	for (i = 0; i < nl; i++)
		h = (h ^ (uint64_t)dom[i].lo) * 0x100000001B3ULL ^ (uint64_t)dom[i].hi *
		    0x9E3779B97F4A7C15ULL;
	slot = (int)(h % FMEMO);
	m = &fmemo[slot];
	*hit = m->used && m->x == x && m->mode == mode;
	for (i = 0; i < nl && *hit; i++)
		if (m->b[2 * i] != dom[i].lo || m->b[2 * i + 1] != dom[i].hi)
			*hit = 0;
	if (!*hit) {
		m->used = 1;
		m->x = x;
		m->mode = mode;
		for (i = 0; i < nl; i++) {
			m->b[2 * i] = dom[i].lo;
			m->b[2 * i + 1] = dom[i].hi;
		}
	}
	return m;
}

static struct av
ae_fine(env_t e, int x)
{
	struct av v = ae(e, x), dom[6], f;
	struct fine *F;
	double prod = 1;
	int i;
	if (v.bot || v.lo == v.hi || G.x[x].ty == TX || G.x[x].ty == TF)
		return v;
	if (!fn_done[x])
		fine_info(x);
	F = &FN[x];
	if (!F->ok)
		return v;
	for (i = 0; i < F->nl; i++) {
		dom[i] = e[F->lf[i]];
		if (dom[i].bot || G.y[F->lf[i]].ty == TX)
			return v;
		prod *= (double)(dom[i].hi - dom[i].lo + 1);
	}
	memcpy(cvsave, C.v, (size_t)A.ns * sizeof *C.v);
	if (prod <= 65536) {
		int hit;
		struct fmemo *m = fmemo_at(x, dom, F->nl, -1, &hit);
		if (hit)
			f = m->r;
		else {
			f = enum_leaves(e, x, F->lf, dom, F->nl, -1, -1);
			m->r = f;
		}
		memcpy(C.v, cvsave, (size_t)A.ns * sizeof *C.v);
		return f.bot ? v : av_meet(G.x[x].ty, v, f);
	}
	/* a leaf bounded by another: its largest value at the bound */
	for (i = 0; i < F->nl; i++) {
		int s = dom[i].sh, j, k = -1, m;
		double p2 = 1;
		if (s < 0)
			continue;
		for (j = 0; j < F->nl; j++)
			if (F->lf[j] == s)
				k = j;
		if (k < 0)
			continue;
		for (j = 0; j < F->nl; j++)
			if (j != i)
				p2 *= (double)(dom[j].hi - dom[j].lo + 1);
		if (p2 > (1 << 20))
			continue;
		m = mono(e, x, F->lf[i]);
		if (m != 1)
			continue;
		{
			int hit;
			struct fmemo *mm = fmemo_at(x, dom, F->nl, i, &hit);
			if (hit) {
				memcpy(C.v, cvsave, (size_t)A.ns * sizeof *C.v);
				return mm->r.bot ? v : av_meet(G.x[x].ty, v, mm->r);
			}
			mm->r = av_bot();
		}
		{
			struct av d2[6], lo, hi;
			for (j = 0; j < F->nl; j++)
				d2[j] = dom[j];
			d2[i].hi = d2[i].lo;	/* t = its least: the least value */
			lo = enum_leaves(e, x, F->lf, d2, F->nl, -1, -1);
			hi = enum_leaves(e, x, F->lf, d2, F->nl, F->lf[i], s);
			memcpy(C.v, cvsave, (size_t)A.ns * sizeof *C.v);
			if (lo.bot || hi.bot)
				return v;
			f = av_iv(G.x[x].ty, lo.lo, hi.hi);
			{
				int hit;
				fmemo_at(x, dom, F->nl, i, &hit)->r = f;
			}
			return av_meet(G.x[x].ty, v, f);
		}
	}
	memcpy(C.v, cvsave, (size_t)A.ns * sizeof *C.v);
	return v;
}

/* ---- statements over sets of environments ---- */

struct ctx {
	struct eset brk, cont, ret;
};

static void ex(int s, struct eset *E, struct ctx *c);

/* a key variable that holds a few values: one environment per value */
static void
split_key(struct eset *E, int y)
{
	struct eset N = {0};
	int i;
	if (!A.iskey[y])
		return;
	for (i = 0; i < E->n; i++) {
		env_t e = E->e[i];
		struct av v = e[y];
		int64_t k;
		if (v.bot || v.lo == v.hi || v.hi - v.lo > 15) {
			es_add(&N, e, 0);
			continue;
		}
		for (k = v.lo; k <= v.hi; k++) {
			env_t d;
			if (v.lo >= 0 && ((uint32_t)k & ~v.may))
				continue;
			d = env_dup(e);
			d[y] = av_k(G.y[y].ty, k);
			es_add(&N, d, 0);
		}
		free(e);
	}
	free(E->e);
	*E = N;
}

/* the variable an assignment expression writes, or -1 */
static int
asg_target(int x)
{
	const struct gx *g = &G.x[x];
	if ((g->k == X_ASG || g->k == X_INC) && G.x[g->a].k == X_VAR)
		return G.x[g->a].sym;
	return -1;
}

static void
havoc_fn(env_t e, int f)
{
	int i;
	for (i = 0; i < A.ns; i++)
		if ((G.y[i].kind == Y_GLOBAL) || (G.y[i].fn == f &&
		    (G.y[i].kind == Y_LOCAL || G.y[i].kind == Y_PARAM)))
			e[i] = av_top(G.y[i].arr ? G.y[i].ty : G.y[i].ty);
}

/* e (the sets own it) to T where the test x holds, to F where it fails;
 * x < 0 always holds */
static void
split_test(int x, env_t e, struct eset *T, struct eset *F)
{
	struct av cv = x >= 0 ? ae(e, x) : av_k(TB, 1);
	struct undo u[64];
	int nu = 0;
	env_t d;
	if (cv.bot) {
		free(e);
		return;
	}
	if (cv.lo == cv.hi) {
		es_add(cv.lo ? T : F, e, 0);
		return;
	}
	d = env_dup(e);
	if (refine(d, x, 1, u, &nu, 64))
		es_add(T, d, 0);
	else
		free(d);
	nu = 0;
	if (refine(e, x, 0, u, &nu, 64))
		es_add(F, e, 0);
	else
		free(e);
}

/*
 * The first two passes of loop s apart, from P (the states it starts
 * from, freed): what each runs, whether the second comes back to the
 * test.  The loop's fixed point has run on these states already, so
 * every call here finds its summary; nothing is recorded, and loops
 * inside are not probed again (they are where they run from the fixed
 * point).
 */
static void
probe_loop(int s, struct eset *P)
{
	const struct gs *t = &G.s[s];
	struct lprobe *L = &LP[s];
	int var = t->k == S_FOR && t->c >= 0 ? asg_target(t->c) : -1, pass, i;
	if (L->r1 == NULL) {
		L->r1 = xcalloc((size_t)G.ns + 1, 1);
		L->r2 = xcalloc((size_t)G.ns + 1, 1);
		L->v1 = L->v2 = av_bot();
		L->entry = xmalloc(((size_t)A.ns + 1) * sizeof *L->entry);
		for (i = 0; i < A.ns; i++)
			L->entry[i] = av_bot();
	}
	L->seen = 1;
	for (pass = 0; pass < P->n; pass++)
		for (i = 0; i < A.ns; i++)
			L->entry[i] = av_join(L->entry[i], P->e[pass][i]);
	for (pass = 0; pass < 2 && P->n && !A.out_of_time; pass++) {
		struct eset T = {0}, F = {0};
		struct ctx lc;
		memset(&lc, 0, sizeof lc);
		for (i = 0; i < P->n; i++) {
			if (var >= 0) {
				struct av *v = pass ? &L->v2 : &L->v1;
				*v = av_join(*v, P->e[i][var]);
			}
			split_test(t->b, P->e[i], &T, &F);
		}
		free(P->e);
		memset(P, 0, sizeof *P);
		if (T.n)
			*(pass ? &L->ran2 : &L->ran1) = 1;
		if (F.n)
			*(pass ? &L->fail2 : &L->fail1) = 1;
		A.reach = pass ? L->r2 : L->r1;
		ex(t->d, &T, &lc);
		A.reach = NULL;
		if (pass == 0 && lc.brk.n)
			L->brk1 = 1;
		es_move(&T, &lc.cont);
		if (t->k == S_FOR && t->c >= 0)
			for (i = 0; i < T.n; i++)
				(void)ae(T.e[i], t->c);
		if (T.n)
			*(pass ? &L->more : &L->at2) = 1;
		es_free(&lc.brk);
		es_free(&lc.ret);
		es_free(&F);
		*P = T;
	}
	es_free(P);
}

static void
ex(int s, struct eset *E, struct ctx *c)
{
	const struct gs *t;
	int i;
	if (s < 0 || E->n == 0)
		return;
	t = &G.s[s];
	if (A.out_of_time)
		return;
	if (A.reach)
		A.reach[s] = 1;
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n && E->n; i++)
			ex(G.slist[t->first + i], E, c);
		return;
	case S_DECL: {
		int y = t->sym;
		for (i = 0; i < E->n; i++) {
			env_t e = E->e[i];
			struct av v;
			int z;
			if (G.y[y].arr) {
				e[y] = av_top(G.y[y].ty);
				continue;
			}
			v = t->a >= 0 ? av_conv(ae_fine(e, t->a), G.x[t->a].ty,
			    G.y[y].ty) : av_top(G.y[y].ty);
			if (v.sh == y)
				v.sh = -1;
			e[y] = v;
			if (A.cur >= 0)
				for (z = 0; z < A.nloc[A.cur]; z++)
					if (e[A.loc[A.cur][z]].sh == y && A.loc[A.cur][z] != y)
						e[A.loc[A.cur][z]].sh = -1;
		}
		split_key(E, y);
		return;
	}
	case S_EXPR: {
		int y = asg_target(t->a);
		struct eset N = {0};
		for (i = 0; i < E->n; i++)
			(void)ae(E->e[i], t->a);
		if (y >= 0 && A.iskey[y]) {
			/* an assignment can make two environments one */
			for (i = 0; i < E->n; i++)
				es_add(&N, E->e[i], 0);
			free(E->e);
			*E = N;
			split_key(E, y);
		}
		return;
	}
	case S_IF: {
		struct eset T = {0}, F = {0};
		for (i = 0; i < E->n; i++)
			split_test(t->a, E->e[i], &T, &F);
		free(E->e);
		memset(E, 0, sizeof *E);
		ex(t->b, &T, c);
		if (t->c >= 0)
			ex(t->c, &F, c);
		es_move(E, &T);
		es_move(E, &F);
		return;
	}
	case S_FOR:
	case S_WHILE:
	case S_DO: {
		struct eset head = {0}, X = {0}, P = {0};
		int it;
		if (t->k == S_FOR && t->a >= 0)
			ex(t->a, E, c);
		if (t->k == S_DO) {
			/* the body once before the first test */
			struct ctx lc;
			memset(&lc, 0, sizeof lc);
			ex(t->d, E, &lc);
			es_move(E, &lc.cont);
			es_move(&c->ret, &lc.ret);
			es_move(&X, &lc.brk);
		} else if (A.probe_round && A.reach == NULL)
			for (i = 0; i < E->n; i++)
				es_add(&P, env_dup(E->e[i]), 0);
		es_move(&head, E);
		for (it = 0;; it++) {
			struct eset T = {0}, Xi = {0};
			struct ctx lc;
			int ch = 0;
			memset(&lc, 0, sizeof lc);
			for (i = 0; i < head.n; i++)
				split_test(t->b, env_dup(head.e[i]), &T, &Xi);
			ex(t->d, &T, &lc);
			es_move(&T, &lc.cont);
			if (t->k == S_FOR && t->c >= 0)
				for (i = 0; i < T.n; i++)
					(void)ae(T.e[i], t->c);
			es_move(&c->ret, &lc.ret);
			for (i = 0; i < T.n; i++) {
				ch |= es_add(&head, T.e[i], it >= 3);
			}
			free(T.e);
			if (!ch || it > 60 || A.out_of_time) {
				es_move(&X, &Xi);
				es_move(&X, &lc.brk);
				break;
			}
			es_free(&Xi);
			es_free(&lc.brk);
		}
		es_free(&head);
		if (P.n)
			probe_loop(s, &P);
		es_move(E, &X);
		return;
	}
	case S_RET:
		for (i = 0; i < E->n; i++) {
			env_t e = E->e[i];
			if (t->a >= 0 && t->sym >= 0)
				e[t->sym] = av_conv(ae_fine(e, t->a), G.x[t->a].ty,
				    G.y[t->sym].ty);
			es_add(&c->ret, e, 0);
		}
		free(E->e);
		memset(E, 0, sizeof *E);
		return;
	case S_BRK:
	case S_CONT:
		es_move(t->k == S_BRK ? &c->brk : &c->cont, E);
		return;
	case S_UNK:
		for (i = 0; i < E->n; i++)
			havoc_fn(E->e[i], A.cur);
		return;
	}
}

/* ---- calls: a summary per value of the keys the callee reads ---- */

static void
analyse_fn(int f, struct summ *S)
{
	struct eset E = {0};
	struct ctx c;
	int save = A.cur, i;
	env_t e = env_dup(S->entry);
	memset(&c, 0, sizeof c);
	for (i = 0; i < A.nloc[f]; i++)
		if (G.y[A.loc[f][i]].kind != Y_PARAM)
			e[A.loc[f][i]] = av_top(G.y[A.loc[f][i]].ty);
	es_add(&E, e, 0);
	A.cur = f;
	G.f[f].onstack = 1;
	{
		/* a summary serves every call with its keys: no call's own */
		int sv = A.site;
		A.site = -1;
		ex(G.f[f].body, &E, &c);
		A.site = sv;
	}
	G.f[f].onstack = 0;
	A.cur = save;
	es_move(&c.ret, &E);
	es_free(&c.brk);
	es_free(&c.cont);
	es_free(&S->exit);
	S->exit = c.ret;
	G.f[f].analysed = 1;
}

/* a new summary keyed kv (nk pairs), its entry empty */
static void
summ_new(struct fsum *F, const int64_t *kv, int nk)
{
	struct summ *S;
	int i;
	if (F->n == F->cap) {
		F->cap = F->cap ? F->cap * 2 : 4;
		F->s = xrealloc(F->s, (size_t)F->cap * sizeof *F->s);
	}
	S = &F->s[F->n++];
	memset(S, 0, sizeof *S);
	S->kv = xmalloc((size_t)(2 * nk + 1) * sizeof *S->kv);
	memcpy(S->kv, kv, (size_t)(2 * nk) * sizeof *kv);
	S->entry = xmalloc((size_t)A.ns * sizeof *S->entry);
	for (i = 0; i < A.ns; i++)
		S->entry[i] = av_bot();
}

/* does entry hold this call of f (its arguments, the globals it reads)? */
static int
entry_covers(const struct av *entry, int f, const struct av *args, int na,
    const struct av *e)
{
	int i, k;
	for (i = 0; i < na && i < G.f[f].np; i++) {
		struct av v = args[i], j;
		v.sh = -1;
		j = av_join(entry[G.f[f].p[i]], v);
		if (!av_same(&j, &entry[G.f[f].p[i]]))
			return 0;
	}
	for (k = 0; k < A.nin[f]; k++) {
		int y = A.in[f][k];
		struct av v = e[y], j;
		if (G.y[y].kind == Y_PARAM)
			continue;
		v.sh = -1;
		j = av_join(entry[y], v);
		if (!av_same(&j, &entry[y]))
			return 0;
	}
	return 1;
}

static void
call_fn(env_t e, int f, struct av *args, int na, struct av *ret)
{
	struct fsum *F = &FS[f];
	struct summ *S = NULL;
	int i, k, ch = 0;
	int64_t kv[64 + 2 * (MAXP + 8)];
	int nk = A.nrk[f] < 32 ? A.nrk[f] : 32, nk0;

	if (G.f[f].body < 0 || G.f[f].onstack || A.out_of_time) {
		for (i = 0; i < A.nwrl[f]; i++)
			e[A.wrl[f][i]] = av_top(G.y[A.wrl[f][i]].ty);
		*ret = av_top(G.f[f].ty);
		return;
	}
	for (k = 0; k < nk; k++) {
		struct av v = e[A.rk[f][k]];
		kv[2 * k] = v.bot ? 1 : v.lo;
		kv[2 * k + 1] = v.bot ? 0 : v.hi;
	}
	/* an argument that is one value here keys the summary too (n in
	 * Widen(v, n)), so what depends on it stays apart */
	nk0 = nk;
	for (i = 0; i < na && i < G.f[f].np; i++, nk++) {
		int one = !args[i].bot && args[i].lo == args[i].hi &&
		    G.y[G.f[f].p[i]].ty != TX;
		kv[2 * nk] = one ? args[i].lo : 1;
		kv[2 * nk + 1] = one ? args[i].lo : 0;
	}
	(void)nk0;
	/* a probed pass (probe_loop) takes the probe round's own summaries
	 * first (made after the analysis's, below) */
	for (k = 0; k < F->n; k++) {
		i = A.reach ? F->n - 1 - k : k;
		if (!memcmp(F->s[i].kv, kv, (size_t)(2 * nk) * sizeof *kv)) {
			S = &F->s[i];
			break;
		}
	}
	if (S != NULL && A.reach && S - F->s < lp_n0[f] && (!S->valid ||
	    !entry_covers(S->entry, f, args, na, e))) {
		/* it reads the analysis's summary and never grows or runs it
		 * (its loops are probed where the analysis runs them): a copy
		 * of its own, dropped after the round */
		int at = (int)(S - F->s);
		S = NULL;
		summ_new(F, kv, nk);
		S = &F->s[F->n - 1];
		memcpy(S->entry, F->s[at].entry, (size_t)A.ns * sizeof *S->entry);
	}
	if (S == NULL) {
		summ_new(F, kv, nk);
		S = &F->s[F->n - 1];
	}
	/* the entry: parameters and the globals it reads, joined */
	for (i = 0; i < na && i < G.f[f].np; i++) {
		int p = G.f[f].p[i];
		struct av v = args[i], j;
		v.sh = -1;
		j = av_join(S->entry[p], v);
		if (!av_same(&j, &S->entry[p])) {
			S->entry[p] = j;
			ch = 1;
		}
	}
	for (k = 0; k < A.nin[f]; k++) {
		int y = A.in[f][k];
		struct av v = e[y], j;
		if (G.y[y].kind == Y_PARAM)
			continue;
		v.sh = -1;
		j = av_join(S->entry[y], v);
		if (!av_same(&j, &S->entry[y])) {
			S->entry[y] = j;
			ch = 1;
		}
	}
	if (ch)
		A.changed = 1;
	/* the probe round makes summaries of its own (a probed pass's
	 * arguments are narrower, so keyed apart), dropped after it; one the
	 * analysis made must not grow */
	if (ch && A.probe_round && S - F->s < lp_n0[f])
		A.probe_grew = 1;
	if (A.inl[f]) {
		/* loop-free: run its body on this call's own values (the entry
		 * above is kept for the domains) */
		struct eset E = {0};
		struct ctx c;
		int save = A.cur;
		env_t d = env_dup(e);
		memset(&c, 0, sizeof c);
		for (i = 0; i < A.nloc[f]; i++)
			if (G.y[A.loc[f][i]].kind != Y_PARAM)
				d[A.loc[f][i]] = av_top(G.y[A.loc[f][i]].ty);
		for (i = 0; i < na && i < G.f[f].np; i++) {
			d[G.f[f].p[i]] = args[i];
			d[G.f[f].p[i]].sh = -1;
		}
		es_add(&E, d, 0);
		A.cur = f;
		G.f[f].onstack = 1;
		ex(G.f[f].body, &E, &c);
		G.f[f].onstack = 0;
		A.cur = save;
		es_move(&c.ret, &E);
		es_free(&c.brk);
		es_free(&c.cont);
		*ret = av_bot();
		for (i = 0; i < A.nwrl[f]; i++) {
			int y = A.wrl[f][i];
			struct av j = av_bot();
			for (k = 0; k < c.ret.n; k++)
				j = av_join(j, c.ret.e[k][y]);
			e[y] = j.bot ? av_top(G.y[y].ty) : j;
		}
		for (k = 0; k < c.ret.n; k++)
			*ret = av_join(*ret, c.ret.e[k][G.f[f].ret]);
		es_free(&c.ret);
		if (ret->bot)
			*ret = av_top(G.f[f].ty);
		ret->sh = -1;
		S->valid = 1;
		G.f[f].analysed = 1;
		return;
	}
	if (ch || !S->valid) {
		S->valid = 1;
		analyse_fn(f, S);
		S = NULL;	/* FS[f].s may have moved: find it again */
		for (i = 0; i < FS[f].n; i++)
			if (!memcmp(FS[f].s[i].kv, kv, (size_t)(2 * nk) * sizeof *kv))
				S = &FS[f].s[i];
	}
	*ret = av_bot();
	for (i = 0; i < A.nwrl[f]; i++) {
		int y = A.wrl[f][i];
		struct av j = av_bot();
		for (k = 0; k < S->exit.n; k++)
			j = av_join(j, S->exit.e[k][y]);
		e[y] = j.bot ? av_top(G.y[y].ty) : j;
	}
	for (k = 0; k < S->exit.n; k++)
		*ret = av_join(*ret, S->exit.e[k][G.f[f].ret]);
	if (ret->bot)
		*ret = av_top(G.f[f].ty);
	ret->sh = -1;
	if (++A.work % 4096 == 0 && (double)clock() / CLOCKS_PER_SEC >
	    A.deadline)
		A.out_of_time = 1;
}

/* ---- what each function reads and writes; the keys ---- */

static uint8_t *nonconst;	/* per symbol: assigned something not a constant */
static int *callee_edge;	/* calls: (caller, callee) pairs */
static int ncall_edge, capcall_edge;

static int
constsel(int x)
{
	const struct gx *g = &G.x[x];
	if (g->k == X_NUM)
		return 1;
	if (g->k == X_SEL)
		return constsel(g->b) && constsel(g->c);
	if (g->k == X_VAR && G.y[g->sym].kind == Y_CONST && !G.y[g->sym].arr)
		return 1;
	if (g->k == X_CALL && (g->sym == -B_UINT || g->sym == -B_INT) && g->b == 1)
		return constsel(G.xarg[g->a]);
	return 0;
}

static void
rw_x(int f, int x, int lhs)
{
	const struct gx *g;
	int i;
	if (x < 0)
		return;
	g = &G.x[x];
	switch (g->k) {
	case X_VAR: {
		int y = g->sym;
		if (lhs) {
			if (G.y[y].kind == Y_GLOBAL)
				A.wr[f][y] = 1;
			if (G.y[y].kind == Y_BUF || (G.y[y].kind == Y_PARAM &&
			    G.y[y].out))
				A.impure[f] = 1;
		} else if (G.y[y].kind == Y_GLOBAL)
			A.rd[f][y] = 1;
		return;
	}
	case X_IDX:
		rw_x(f, g->a, lhs);
		if (lhs)
			rw_x(f, g->a, 0);	/* the rest of the array stays */
		rw_x(f, g->b, 0);
		return;
	case X_MEM:
		rw_x(f, g->a, lhs);
		return;
	case X_ASG:
	case X_INC: {
		int y = G.x[g->a].k == X_VAR ? G.x[g->a].sym : G.x[g->a].k == X_IDX &&
		    G.x[G.x[g->a].a].k == X_VAR ? G.x[G.x[g->a].a].sym : -1;
		rw_x(f, g->a, 1);
		if (g->k == X_INC || g->op)
			rw_x(f, g->a, 0);
		if (g->k == X_ASG)
			rw_x(f, g->b, 0);
		if (y >= 0) {
			A.nasg[y]++;
			if (g->k == X_INC || g->op || !constsel(g->b) ||
			    G.x[g->a].k != X_VAR)
				nonconst[y] = 1;
		}
		return;
	}
	case X_CALL:
		for (i = 0; i < g->b; i++) {
			int a = G.xarg[g->a + i];
			rw_x(f, a, 0);
			if (g->sym >= 0 && i < G.f[g->sym].np &&
			    G.y[G.f[g->sym].p[i]].out) {
				rw_x(f, a, 1);
				if (G.x[a].k == X_VAR) {
					A.nasg[G.x[a].sym]++;
					nonconst[G.x[a].sym] = 1;
				}
			}
		}
		if (g->sym >= 0) {
			if (ncall_edge == capcall_edge) {
				capcall_edge = capcall_edge ? capcall_edge * 2 : 256;
				callee_edge = xrealloc(callee_edge, (size_t)capcall_edge * 2 *
				    sizeof *callee_edge);
			}
			callee_edge[2 * ncall_edge] = f;
			callee_edge[2 * ncall_edge + 1] = g->sym;
			ncall_edge++;
		}
		return;
	case X_UN:
		rw_x(f, g->a, 0);
		return;
	case X_BIN:
	case X_COMMA:
		rw_x(f, g->a, 0);
		rw_x(f, g->b, 0);
		return;
	case X_SEL:
		rw_x(f, g->a, 0);
		rw_x(f, g->b, 0);
		rw_x(f, g->c, 0);
		return;
	}
}

static void
rw_s(int f, int s)
{
	const struct gs *t;
	int i;
	if (s < 0)
		return;
	t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			rw_s(f, G.slist[t->first + i]);
		return;
	case S_DECL:
		if (t->a >= 0) {
			rw_x(f, t->a, 0);
			if (!constsel(t->a))
				nonconst[t->sym] = 1;
		}
		return;
	case S_EXPR:
		rw_x(f, t->a, 0);
		return;
	case S_IF:
		rw_x(f, t->a, 0);
		rw_s(f, t->b);
		rw_s(f, t->c);
		return;
	case S_FOR:
	case S_WHILE:
	case S_DO:
		rw_s(f, t->a);
		rw_x(f, t->b, 0);
		rw_x(f, t->c, 0);
		rw_s(f, t->d);
		return;
	case S_RET:
		rw_x(f, t->a, 0);
		return;
	case S_UNK:
		A.impure[f] = 1;
		return;
	}
}

static int
has_loop(int s)
{
	const struct gs *t;
	int i;
	if (s < 0)
		return 0;
	t = &G.s[s];
	switch (t->k) {
	case S_FOR:
	case S_WHILE:
	case S_DO:
	case S_UNK:
		return 1;
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			if (has_loop(G.slist[t->first + i]))
				return 1;
		return 0;
	case S_IF:
		return has_loop(t->b) || has_loop(t->c);
	}
	return 0;
}

static int is_root(int f);

static void
prepare(void)
{
	int f, i, ch;
	A.ns = G.ny;
	A.nasg = xcalloc((size_t)A.ns, sizeof *A.nasg);
	nonconst = xcalloc((size_t)A.ns, 1);
	A.rd = xcalloc((size_t)G.nf, sizeof *A.rd);
	A.wr = xcalloc((size_t)G.nf, sizeof *A.wr);
	A.impure = xcalloc((size_t)G.nf, 1);
	for (f = 0; f < G.nf; f++) {
		A.rd[f] = xcalloc((size_t)A.ns, 1);
		A.wr[f] = xcalloc((size_t)A.ns, 1);
	}
	ncall_edge = 0;
	for (f = 0; f < G.nf; f++)
		rw_s(f, G.f[f].body);
	for (i = 0; i < ncall_edge; i++)
		G.f[callee_edge[2 * i + 1]].ncaller++;
	/* transitively */
	do {
		ch = 0;
		for (i = 0; i < ncall_edge; i++) {
			int a = callee_edge[2 * i], b = callee_edge[2 * i + 1], y;
			for (y = 0; y < A.ns; y++) {
				if (A.rd[b][y] && !A.rd[a][y])
					A.rd[a][y] = 1, ch = 1;
				if (A.wr[b][y] && !A.wr[a][y])
					A.wr[a][y] = 1, ch = 1;
			}
			if (A.impure[b] && !A.impure[a])
				A.impure[a] = 1, ch = 1;
		}
	} while (ch);
	/* keys */
	A.iskey = xcalloc((size_t)A.ns, 1);
	A.keys = xmalloc((size_t)A.ns * sizeof *A.keys);
	A.nkeys = 0;
	for (i = 0; i < A.ns; i++) {
		const struct gsym *y = &G.y[i];
		int k = 0;
		if (y->arr || (y->ty != TU && y->ty != TI && y->ty != TB))
			continue;
		if (y->kind == Y_GLOBAL)
			k = A.nasg[i] > 0 && !nonconst[i];
		else if (y->kind == Y_LOCAL)
			k = y->ty == TB || !nonconst[i];
		if (k) {
			A.iskey[i] = 1;
			A.keys[A.nkeys++] = i;
		}
	}
	A.rk = xcalloc((size_t)G.nf, sizeof *A.rk);
	A.nrk = xcalloc((size_t)G.nf, sizeof *A.nrk);
	A.in = xcalloc((size_t)G.nf, sizeof *A.in);
	A.nin = xcalloc((size_t)G.nf, sizeof *A.nin);
	for (f = 0; f < G.nf; f++) {
		A.rk[f] = xmalloc((size_t)(A.nkeys + 1) * sizeof **A.rk);
		A.in[f] = xmalloc((size_t)(A.ns + 1) * sizeof **A.in);
		for (i = 0; i < A.nkeys; i++)
			if (A.rd[f][A.keys[i]] && G.y[A.keys[i]].kind == Y_GLOBAL)
				A.rk[f][A.nrk[f]++] = A.keys[i];
		for (i = 0; i < G.f[f].np; i++)
			A.in[f][A.nin[f]++] = G.f[f].p[i];
		for (i = 0; i < A.ns; i++)
			if (A.rd[f][i])
				A.in[f][A.nin[f]++] = i;
	}
	A.loc = xcalloc((size_t)G.nf, sizeof *A.loc);
	A.nloc = xcalloc((size_t)G.nf, sizeof *A.nloc);
	A.wrl = xcalloc((size_t)G.nf, sizeof *A.wrl);
	A.nwrl = xcalloc((size_t)G.nf, sizeof *A.nwrl);
	for (f = 0; f < G.nf; f++) {
		A.loc[f] = xmalloc((size_t)(A.ns + 1) * sizeof **A.loc);
		A.wrl[f] = xmalloc((size_t)(A.ns + 1) * sizeof **A.wrl);
		for (i = 0; i < A.ns; i++) {
			if (G.y[i].fn == f && (G.y[i].kind == Y_LOCAL || G.y[i].kind ==
			    Y_RET || G.y[i].kind == Y_PARAM))
				A.loc[f][A.nloc[f]++] = i;
			if (A.wr[f][i])
				A.wrl[f][A.nwrl[f]++] = i;
		}
	}
	A.inl = xcalloc((size_t)G.nf, 1);
	for (f = 0; f < G.nf; f++)
		A.inl[f] = G.f[f].body >= 0 && !has_loop(G.f[f].body) &&
		    !is_root(f);
	FS = xcalloc((size_t)G.nf, sizeof *FS);
	C.v = xcalloc((size_t)A.ns + 1, sizeof *C.v);
	cvsave = xcalloc((size_t)A.ns + 1, sizeof *cvsave);
	expd = xcalloc((size_t)A.ns + 1, 1);
	FN = xcalloc((size_t)G.nx + 1, sizeof *FN);
	fn_done = xcalloc((size_t)G.nx + 1, 1);
	if (recn == NULL) {
		recn = xmalloc(((size_t)G.nx + 1) * sizeof *recn);
		for (i = 0; i <= G.nx; i++)
			recn[i] = -1;
	}
}

static int
is_root(int f)
{
	int m = -1, i;
	for (i = 0; i < G.nf; i++)
		if (!strcmp(G.f[i].name, "main") && G.f[i].body >= 0)
			m = i;
	return m >= 0 ? f == m : G.f[f].ncaller == 0 && G.f[f].body >= 0;
}

/* the analysis from the roots, until no summary's entry grows */
static int
analyse(double deadline)
{
	env_t init = xmalloc((size_t)A.ns * sizeof *init);
	int i, f, round, k;
	A.deadline = deadline;
	for (i = 0; i < A.ns; i++)
		init[i] = av_top(G.y[i].ty);
	for (i = 0; i < A.ns; i++)
		if (G.y[i].kind == Y_GLOBAL && G.y[i].init >= 0 && !G.y[i].arr)
			init[i] = av_conv(ae(init, G.y[i].init), G.x[G.y[i].init].ty,
			    G.y[i].ty);
	for (round = 0; round < 8; round++) {
		A.changed = 0;
		A.recording = 1;
		for (k = 0; k < nrc; k++)
			RC[k].nbox = 0;
		for (f = 0; f < G.nf; f++)
			for (i = 0; i < FS[f].n; i++)
				FS[f].s[i].valid = 0;
		for (f = 0; f < G.nf; f++)
			if (is_root(f)) {
				env_t e = env_dup(init);
				struct av r;
				call_fn(e, f, NULL, 0, &r);
				free(e);
			}
		if (A.out_of_time)
			break;
		if (!A.changed && round > 0)
			break;
	}
	free(init);
	return round;
}

/*
 * One more round from the roots once the analysis has settled, with each
 * loop's first two passes run apart as well (probe_loop).  It records
 * nothing and grows no summary (the states are the settled ones), so what
 * the searches read is the analysis's; when it cannot finish, LP goes and
 * lp_note says why.
 */
static void
probe_round(int settled)
{
	env_t init;
	int i, f;
	lp_note = !settled ? 2 : 0;
	if (A.out_of_time || !settled)
		return;
	LP = xcalloc((size_t)G.ns + 1, sizeof *LP);
	init = xmalloc((size_t)A.ns * sizeof *init);
	for (i = 0; i < A.ns; i++)
		init[i] = av_top(G.y[i].ty);
	for (i = 0; i < A.ns; i++)
		if (G.y[i].kind == Y_GLOBAL && G.y[i].init >= 0 && !G.y[i].arr)
			init[i] = av_conv(ae(init, G.y[i].init), G.x[G.y[i].init].ty,
			    G.y[i].ty);
	A.recording = 0;
	A.probe_round = 1;
	A.probe_grew = 0;
	lp_n0 = xmalloc(((size_t)G.nf + 1) * sizeof *lp_n0);
	for (f = 0; f < G.nf; f++) {
		lp_n0[f] = FS[f].n;
		for (i = 0; i < FS[f].n; i++)
			FS[f].s[i].valid = 0;
	}
	for (f = 0; f < G.nf && !A.out_of_time; f++)
		if (is_root(f)) {
			env_t e = env_dup(init);
			struct av r;
			call_fn(e, f, NULL, 0, &r);
			free(e);
		}
	A.probe_round = 0;
	free(init);
	for (f = 0; f < G.nf; f++) {
		for (i = lp_n0[f]; i < FS[f].n; i++) {
			free(FS[f].s[i].kv);
			free(FS[f].s[i].entry);
			es_free(&FS[f].s[i].exit);
		}
		FS[f].n = lp_n0[f];
	}
	free(lp_n0);
	lp_n0 = NULL;
	if (A.out_of_time || A.probe_grew) {
		lp_note = A.out_of_time ? 1 : 3;
		A.out_of_time = 0;	/* the analysis itself had finished */
		lp_free();
	}
}

/* ---- domains, as text ---- */

static void
av_text(struct sbuf *b, struct av v, int ty)
{
	if (v.bot) {
		sb_puts(b, "(never)");
		return;
	}
	if (ty == TB && v.lo == 0 && v.hi == 1) {
		sb_puts(b, "false..true");
		return;
	}
	if (v.lo == v.hi)
		sb_printf(b, "%lld", (long long)v.lo);
	else
		sb_printf(b, "%lld..%lld", (long long)v.lo, (long long)v.hi);
}

static int
is_full(struct av v, int ty)
{
	int64_t lo, hi;
	trange(ty, &lo, &hi);
	return !v.bot && v.lo <= lo && v.hi >= hi;
}

/*
 * vars[nv] over boxes rows (each row nv values): "n in {5, 6, 7, 8}; v in
 * 0..2^n - 1", a variable's range given per value of another when it
 * differs between rows
 */
static void
dom_text(struct sbuf *b, const int *tys, int nv, const struct av *rows,
    int nrow, const char *const *names)
{
	int i, r, first = 1;
	for (i = 0; i < nv; i++) {
		int ty = tys[i], same = 1, singles = 1, k;
		struct av u = av_bot();
		const char *nm = names[i];
		for (r = 0; r < nrow; r++) {
			struct av v = rows[r * nv + i];
			if (!av_same(&v, &rows[i]) && !(v.lo == rows[i].lo &&
			    v.hi == rows[i].hi))
				same = 0;
			if (v.lo != v.hi)
				singles = 0;
			u = av_join(u, v);
		}
		if (u.bot || is_full(u, ty))
			continue;
		sb_puts(b, first ? "" : "; ");
		first = 0;
		if (same || nrow == 1) {
			sb_printf(b, "%s in ", nm);
			av_text(b, u, ty);
			continue;
		}
		if (singles) {
			/* the values it takes */
			int64_t vals[32];
			int nvl = 0, j;
			for (r = 0; r < nrow && nvl < 32; r++) {
				int64_t x = rows[r * nv + i].lo;
				for (j = 0; j < nvl && vals[j] != x; j++)
					;
				if (j == nvl)
					vals[nvl++] = x;
			}
			/* sorted */
			for (j = 1; j < nvl; j++) {
				int64_t x = vals[j];
				int q = j;
				while (q > 0 && vals[q - 1] > x) {
					vals[q] = vals[q - 1];
					q--;
				}
				vals[q] = x;
			}
			if (nvl < 32) {
				sb_printf(b, "%s in {", nm);
				for (j = 0; j < nvl; j++)
					sb_printf(b, "%s%lld", j ? ", " : "", (long long)vals[j]);
				sb_puts(b, "}");
				continue;
			}
		}
		/* a function of another, single-valued one? */
		for (k = 0; k < nv; k++) {
			int ok = 1, pow2 = 1, r2;
			if (k == i)
				continue;
			for (r = 0; r < nrow && ok; r++) {
				struct av kv = rows[r * nv + k], v = rows[r * nv + i];
				if (kv.lo != kv.hi)
					ok = 0;
				for (r2 = 0; r2 < r && ok; r2++)
					if (rows[r2 * nv + k].lo == kv.lo &&
					    (rows[r2 * nv + i].lo != v.lo ||
					    rows[r2 * nv + i].hi != v.hi))
						ok = 0;
				if (v.lo != 0 || kv.lo < 0 || kv.lo > 31 || v.hi !=
				    ((int64_t)1 << kv.lo) - 1)
					pow2 = 0;
			}
			if (!ok)
				continue;
			if (pow2) {
				sb_printf(b, "%s in 0..2^%s - 1", nm, names[k]);
				break;
			}
			sb_printf(b, "%s in ", nm);
			{
				int64_t seen[64];
				int ns = 0, j;
				for (r = 0; r < nrow; r++) {
					int64_t kv = rows[r * nv + k].lo;
					for (j = 0; j < ns && seen[j] != kv; j++)
						;
					if (j < ns || ns >= 64)
						continue;
					seen[ns++] = kv;
					sb_puts(b, ns > 1 ? ", " : "");
					av_text(b, rows[r * nv + i], ty);
					sb_printf(b, " when %s = %lld", names[k], (long long)kv);
				}
			}
			break;
		}
		if (k == nv) {
			sb_printf(b, "%s in ", nm);
			av_text(b, u, ty);
		}
	}
	if (first)
		sb_puts(b, "any value");
}

static void
vars_text(struct sbuf *b, const int *vars, int nv, const struct av *rows,
    int nrow)
{
	int tys[MAXP + 8], i;
	const char *names[MAXP + 8];
	for (i = 0; i < nv && i < MAXP + 8; i++) {
		tys[i] = G.y[vars[i]].ty;
		names[i] = G.y[vars[i]].name;
	}
	dom_text(b, tys, nv, rows, nrow, names);
}

/* a function's inputs (parameters, scalar globals read) and its rows */
static int
fn_rows(int f, int *vars, struct av **rows)
{
	int nv = 0, i, r, nr = 0;
	for (i = 0; i < A.nin[f]; i++) {
		int y = A.in[f][i];
		if (G.y[y].arr || (G.y[y].ty != TU && G.y[y].ty != TI &&
		    G.y[y].ty != TB))
			continue;
		if (nv < MAXP + 8)
			vars[nv++] = y;
	}
	*rows = xmalloc(((size_t)FS[f].n + 1) * (size_t)(nv + 1) * sizeof **rows);
	for (r = 0; r < FS[f].n; r++) {
		int bad = 0;
		for (i = 0; i < nv; i++) {
			(*rows)[nr * nv + i] = FS[f].s[r].entry[vars[i]];
			if ((*rows)[nr * nv + i].bot)
				bad = 1;
		}
		if (!bad)
			nr++;
	}
	{
		/* a row inside another adds nothing: dropped */
		int a, b, k, out = 0;
		uint8_t *gone = xcalloc((size_t)nr + 1, 1);
		for (a = 0; a < nr; a++)
			for (b = 0; b < nr && !gone[a]; b++) {
				int in = 1;
				if (a == b || gone[b])
					continue;
				for (k = 0; k < nv && in; k++)
					in = (*rows)[a * nv + k].lo >= (*rows)[b * nv + k].lo &&
					    (*rows)[a * nv + k].hi <= (*rows)[b * nv + k].hi;
				if (in)
					gone[a] = 1;
			}
		for (a = 0; a < nr; a++)
			if (!gone[a]) {
				if (out != a)
					memmove(&(*rows)[out * nv], &(*rows)[a * nv],
					    (size_t)nv * sizeof **rows);
				out++;
			}
		free(gone);
		nr = out;
	}
	return nv * 1000 + nr;
}


/* ---- ucas text: a function or an expression as a ucas code block ---- */

/*
 * A GLSL function becomes a ucas code block (README, "Loops and
 * calculus"): its parameters and the globals it reads are the ucas
 * function's parameters; locals are typed variables (u32, i32, bool),
 * every literal and conversion explicit; a call is the callee's body
 * inlined as a nested block; a loop is a while loop; return, break and
 * continue set flags that guard what follows them.  Anything else (an
 * array other than a constant table, a vector, a write to a global) and
 * the translation gives up.
 */
struct ut {
	struct sbuf b;
	int ok;
	int inst;		/* the next inlined copy's number */
	int nloop;
	int gl[64], ngl;	/* the globals read: parameters */
	int *leaf;		/* expression -> leaf parameter, -1 */
	int depth;
};

struct uframe {
	int inst, f;
	int loop;		/* the innermost loop's number, -1 */
	int uses_ret;
};

static const char *
u_ty(int ty)
{
	return ty == TU ? "u32" : ty == TI ? "i32" : "bool";
}

static void
u_name(struct ut *U, struct sbuf *b, int y, const struct uframe *F)
{
	const struct gsym *s = &G.y[y];
	if (s->kind == Y_GLOBAL) {
		int i;
		for (i = 0; i < U->ngl && U->gl[i] != y; i++)
			;
		if (i == U->ngl && U->ngl < 64)
			U->gl[U->ngl++] = y;
		sb_printf(b, "g_%s", s->name);
		return;
	}
	sb_printf(b, "v_%s_%d", s->name, F ? F->inst : 0);
}

static void u_x(struct ut *U, struct sbuf *b, int x, const struct uframe *F);

static void
u_conv(struct ut *U, struct sbuf *b, int x, int to, const struct uframe *F)
{
	int from = G.x[XR(x)].ty;
	if (from == to || (from != TB && to != TB && from == TX)) {
		u_x(U, b, x, F);
		return;
	}
	if (to == TB) {
		sb_puts(b, "(");
		u_x(U, b, x, F);
		sb_printf(b, " != %s(0))", u_ty(from));
		return;
	}
	if (from == TB) {
		sb_puts(b, "(");
		u_x(U, b, x, F);
		sb_printf(b, " ? %s(1) : %s(0))", u_ty(to), u_ty(to));
		return;
	}
	sb_printf(b, "%s(", u_ty(to));
	u_x(U, b, x, F);
	sb_puts(b, ")");
}

static void u_fn_block(struct ut *U, struct sbuf *b, int f, const int *args,
    int na, const struct uframe *caller);

static const char *
u_op(int op)
{
	switch (op) {
	case OP_ADD: return "+";
	case OP_SUB: return "-";
	case OP_MUL: return "*";
	case OP_DIV: return "/";
	case OP_MOD: return "%";
	case OP_SHL: return "<<";
	case OP_SHR: return ">>";
	case OP_AND: return "&";
	case OP_OR: return "|";
	case OP_XOR: return "xor";
	case OP_LT: return "<";
	case OP_LE: return "<=";
	case OP_GT: return ">";
	case OP_GE: return ">=";
	case OP_EQ: return "==";
	case OP_NE: return "!=";
	case OP_LAND: return "&&";
	case OP_LOR: return "||";
	case OP_LXOR: return "!=";
	}
	return "?";
}

static void
u_x(struct ut *U, struct sbuf *b, int x, const struct uframe *F)
{
	const struct gx *g;
	int i;
	x = XR(x);
	if (U->leaf && U->leaf[x] >= 0) {
		sb_printf(b, "l%d", U->leaf[x]);
		return;
	}
	g = &G.x[x];
	if (++U->depth > 200) {
		U->ok = 0;
		return;
	}
	switch (g->k) {
	case X_NUM:
		if (g->ty == TB)
			sb_puts(b, g->v ? "true" : "false");
		else if (g->ty == TU)
			sb_printf(b, "u32(%llu)", (unsigned long long)g->v);
		else if (g->ty == TI)
			sb_printf(b, "i32(%lld)", (long long)g->v);
		else
			U->ok = 0;
		break;
	case X_VAR: {
		const struct gsym *s = &G.y[g->sym];
		if (s->arr || s->ty == TX || s->ty == TF || s->kind == Y_BUF) {
			U->ok = 0;
			break;
		}
		if (s->kind == Y_CONST) {
			if (s->init < 0) {
				U->ok = 0;
				break;
			}
			u_conv(U, b, s->init, s->ty, F);
			break;
		}
		u_name(U, b, g->sym, F);
		break;
	}
	case X_IDX: {
		int y = G.x[g->a].k == X_VAR ? G.x[g->a].sym : -1;
		if (y < 0 || !G.y[y].tab || G.y[y].arr > 256) {
			U->ok = 0;
			break;
		}
		sb_printf(b, "%s([", u_ty(G.y[y].ty));
		for (i = 0; i < G.y[y].arr; i++)
			sb_printf(b, "%s%lld", i ? ", " : "", (long long)norm(G.y[y].ty,
			    G.y[y].tab[i]));
		sb_puts(b, "][");
		u_x(U, b, g->b, F);
		sb_puts(b, "])");
		break;
	}
	case X_CALL: {
		int args[MAXP + 8];
		for (i = 0; i < g->b && i < MAXP + 8; i++)
			args[i] = G.xarg[g->a + i];
		if (g->sym >= 0) {
			u_fn_block(U, b, g->sym, args, g->b, F);
			break;
		}
		switch (-g->sym) {
		case B_MIN:
		case B_MAX:
			if (g->b != 2) {
				U->ok = 0;
				break;
			}
			sb_printf(b, "%s(", g->sym == -B_MIN ? "min" : "max");
			u_conv(U, b, args[0], g->ty, F);
			sb_puts(b, ", ");
			u_conv(U, b, args[1], g->ty, F);
			sb_puts(b, ")");
			break;
		case B_CLAMP:
			if (g->b != 3) {
				U->ok = 0;
				break;
			}
			sb_puts(b, "min(max(");
			u_conv(U, b, args[0], g->ty, F);
			sb_puts(b, ", ");
			u_conv(U, b, args[1], g->ty, F);
			sb_puts(b, "), ");
			u_conv(U, b, args[2], g->ty, F);
			sb_puts(b, ")");
			break;
		case B_ABS:
			sb_puts(b, "abs(");
			u_conv(U, b, args[0], g->ty, F);
			sb_puts(b, ")");
			break;
		case B_INT:
		case B_UINT:
		case B_BOOL:
			u_conv(U, b, args[0], g->ty, F);
			break;
		default:
			U->ok = 0;
		}
		break;
	}
	case X_UN:
		if (g->op == OP_LNOT) {
			sb_puts(b, "(!");
			u_conv(U, b, g->a, TB, F);
			sb_puts(b, ")");
		} else {
			sb_puts(b, g->op == OP_NEG ? "(-" : "(~");
			u_conv(U, b, g->a, g->ty, F);
			sb_puts(b, ")");
		}
		break;
	case X_BIN: {
		int ot = opnd_ty(g);
		sb_puts(b, "(");
		if (g->op == OP_SHL || g->op == OP_SHR) {
			u_conv(U, b, g->a, ot, F);
			sb_printf(b, " %s ", u_op(g->op));
			u_x(U, b, g->b, F);
		} else {
			u_conv(U, b, g->a, ot, F);
			sb_printf(b, " %s ", u_op(g->op));
			u_conv(U, b, g->b, ot, F);
		}
		sb_puts(b, ")");
		break;
	}
	case X_SEL:
		sb_puts(b, "(");
		u_conv(U, b, g->a, TB, F);
		sb_puts(b, " ? ");
		u_conv(U, b, g->b, g->ty, F);
		sb_puts(b, " : ");
		u_conv(U, b, g->c, g->ty, F);
		sb_puts(b, ")");
		break;
	default:
		U->ok = 0;
	}
	U->depth--;
}

/* what a statement may leave by: 1 return, 2 break, 4 continue (of the
 * loop it is in, not of loops inside it) */
static int
exits(int s, int inloop)
{
	const struct gs *t;
	int i, r = 0;
	s = SR(s);
	if (s < 0)
		return 0;
	t = &G.s[s];
	switch (t->k) {
	case S_RET:
		return 1;
	case S_BRK:
		return inloop ? 0 : 2;
	case S_CONT:
		return inloop ? 0 : 4;
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			r |= exits(G.slist[t->first + i], inloop);
		return r;
	case S_IF:
		return exits(t->b, inloop) | exits(t->c, inloop);
	case S_FOR:
	case S_WHILE:
	case S_DO:
		return exits(t->d, 1) & 1;
	}
	return 0;
}

static void u_s(struct ut *U, struct sbuf *b, int s, struct uframe *F);

static void
u_guard(struct sbuf *b, int ex, const struct uframe *F)
{
	int first = 1;
	sb_puts(b, "if !(");
	if (ex & 1) {
		sb_printf(b, "d_%d", F->inst);
		first = 0;
	}
	if (ex & 2) {
		sb_printf(b, "%sb_%d", first ? "" : " || ", F->loop);
		first = 0;
	}
	if (ex & 4)
		sb_printf(b, "%sc_%d", first ? "" : " || ", F->loop);
	sb_puts(b, ") { ");
}

/* statements kids[0..n): each after one that can leave is guarded */
static void
u_seq(struct ut *U, struct sbuf *b, const int *kids, int n, struct uframe *F)
{
	int i, closes = 0, any = 0;
	for (i = 0; i < n; i++) {
		int s = SR(kids[i]), ex;
		if (s < 0)
			continue;
		if (any)
			sb_puts(b, "; ");
		u_s(U, b, s, F);
		any = 1;
		ex = exits(s, 0);
		if (ex && i + 1 < n) {
			sb_puts(b, "; ");
			u_guard(b, ex, F);
			closes++;
			any = 0;
		}
	}
	if (!any)
		sb_printf(b, "d_%d = d_%d", F->inst, F->inst);
	while (closes-- > 0)
		sb_puts(b, " }");
}

static void
u_body(struct ut *U, struct sbuf *b, int s, struct uframe *F)
{
	s = SR(s);
	if (s >= 0 && G.s[s].k == S_BLOCK)
		u_seq(U, b, &G.slist[G.s[s].first], G.s[s].n, F);
	else if (s >= 0)
		u_seq(U, b, &s, 1, F);
	else
		sb_printf(b, "d_%d = d_%d", F->inst, F->inst);
}

static void
u_assign(struct ut *U, struct sbuf *b, int lhs, int op, int rhs, int inc,
    struct uframe *F)
{
	const struct gx *l = &G.x[lhs];
	int y, ty;
	if (l->k != X_VAR || G.y[l->sym].kind == Y_GLOBAL ||
	    G.y[l->sym].kind == Y_BUF || G.y[l->sym].arr) {
		U->ok = 0;
		return;
	}
	y = l->sym;
	ty = G.y[y].ty;
	u_name(U, b, y, F);
	sb_puts(b, " = ");
	if (inc) {
		sb_puts(b, "(");
		u_name(U, b, y, F);
		sb_printf(b, " %s %s(1))", inc > 0 ? "+" : "-", u_ty(ty));
		return;
	}
	if (op == OP_NONE) {
		u_conv(U, b, rhs, ty, F);
		return;
	}
	{
		int ot = op == OP_SHL || op == OP_SHR ? ty : bin_ty(OP_ADD, ty,
		    G.x[rhs].ty);
		if (ot != ty)
			sb_printf(b, "%s(", u_ty(ty));
		sb_puts(b, "(");
		if (ot != ty)
			sb_printf(b, "%s(", u_ty(ot));
		u_name(U, b, y, F);
		if (ot != ty)
			sb_puts(b, ")");
		sb_printf(b, " %s ", u_op(op));
		if (op == OP_SHL || op == OP_SHR)
			u_x(U, b, rhs, F);
		else
			u_conv(U, b, rhs, ot, F);
		sb_puts(b, ")");
		if (ot != ty)
			sb_puts(b, ")");
	}
}

static void
u_expr_stmt(struct ut *U, struct sbuf *b, int x, struct uframe *F)
{
	const struct gx *g = &G.x[XR(x)];
	if (g->k == X_ASG)
		u_assign(U, b, g->a, g->op, g->b, 0, F);
	else if (g->k == X_INC)
		u_assign(U, b, g->a, 0, -1, g->op == OP_ADD ? 1 : -1, F);
	else if (g->k == X_COMMA) {
		u_expr_stmt(U, b, g->a, F);
		sb_puts(b, "; ");
		u_expr_stmt(U, b, g->b, F);
	} else if (g->k == X_CALL && g->sym >= 0 && !A.impure[g->sym] &&
	    !A.nwrl[g->sym])
		sb_printf(b, "d_%d = d_%d", F->inst, F->inst);	/* no effect */
	else
		U->ok = 0;
}

static void
u_s(struct ut *U, struct sbuf *b, int s, struct uframe *F)
{
	const struct gs *t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		u_seq(U, b, &G.slist[t->first], t->n, F);
		return;
	case S_DECL: {
		int y = t->sym;
		if (G.y[y].arr || (G.y[y].ty != TU && G.y[y].ty != TI &&
		    G.y[y].ty != TB)) {
			U->ok = 0;
			return;
		}
		u_name(U, b, y, F);
		sb_printf(b, ": %s = ", u_ty(G.y[y].ty));
		if (t->a >= 0)
			u_conv(U, b, t->a, G.y[y].ty, F);
		else
			sb_puts(b, G.y[y].ty == TB ? "false" : G.y[y].ty == TU ?
			    "u32(0)" : "i32(0)");
		return;
	}
	case S_EXPR:
		u_expr_stmt(U, b, t->a, F);
		return;
	case S_IF:
		sb_puts(b, "if ");
		u_conv(U, b, t->a, TB, F);
		sb_puts(b, " { ");
		u_body(U, b, t->b, F);
		sb_puts(b, " }");
		if (t->c >= 0 && SR(t->c) != -2) {
			sb_puts(b, " else { ");
			u_body(U, b, t->c, F);
			sb_puts(b, " }");
		}
		return;
	case S_FOR:
	case S_WHILE: {
		int L = U->nloop++, save = F->loop, ex = exits(t->d, 0);
		if (t->k == S_FOR && SR(t->a) >= 0) {
			u_s(U, b, SR(t->a), F);
			sb_puts(b, "; ");
		}
		sb_printf(b, "b_%d: bool = false; c_%d: bool = false; while ", L, L);
		if (t->b >= 0)
			u_conv(U, b, t->b, TB, F);
		else
			sb_puts(b, "true");
		sb_printf(b, " && !b_%d && !d_%d { c_%d = false; ", L, F->inst, L);
		F->loop = L;
		u_body(U, b, t->d, F);
		if (t->k == S_FOR && t->c >= 0) {
			/* the step: after a continue too, not after a break or a
			 * return */
			sb_printf(b, "; if !(b_%d || d_%d) { ", L, F->inst);
			u_expr_stmt(U, b, t->c, F);
			sb_puts(b, " }");
		}
		sb_puts(b, " }");
		F->loop = save;
		(void)ex;
		return;
	}
	case S_RET:
		if (t->a >= 0) {
			sb_printf(b, "r_%d = ", F->inst);
			u_conv(U, b, t->a, G.f[F->f].ty, F);
			sb_puts(b, "; ");
		}
		sb_printf(b, "d_%d = true", F->inst);
		return;
	case S_BRK:
		sb_printf(b, "b_%d = true", F->loop);
		return;
	case S_CONT:
		sb_printf(b, "c_%d = true", F->loop);
		return;
	case S_EMPTY:
		sb_printf(b, "d_%d = d_%d", F->inst, F->inst);
		return;
	default:
		U->ok = 0;
	}
}

/* a function as a block: its parameters bound to args (NULL: the ucas
 * function's own parameters) */
static void
u_fn_block(struct ut *U, struct sbuf *b, int f, const int *args, int na,
    const struct uframe *caller)
{
	struct uframe F;
	int i, rt = G.f[f].ty;
	if (G.f[f].body < 0 || (rt != TU && rt != TI && rt != TB) ||
	    A.impure[f] || A.nwrl[f] || U->inst > 64) {
		U->ok = 0;
		return;
	}
	F.inst = U->inst++;
	F.f = f;
	F.loop = -1;
	F.uses_ret = 1;
	sb_puts(b, "{ ");
	if (args)
		for (i = 0; i < G.f[f].np && i < na; i++) {
			int p = G.f[f].p[i];
			u_name(U, b, p, &F);
			sb_printf(b, ": %s = ", u_ty(G.y[p].ty));
			u_conv(U, b, args[i], G.y[p].ty, caller);
			sb_puts(b, "; ");
		}
	sb_printf(b, "d_%d: bool = false; r_%d: %s = %s; ", F.inst, F.inst,
	    u_ty(rt), rt == TB ? "false" : rt == TU ? "u32(0)" : "i32(0)");
	u_body(U, b, G.f[f].body, &F);
	sb_printf(b, "; r_%d }", F.inst);
}

/* ---- ucas functions and their proofs ---- */

static int
u_tyof(int ty)
{
	return ty == TU ? TY(K_UINT, 32) : ty == TI ? TY(K_SINT, 32) : TY_BOOL;
}

/*
 * A ucas function: parameters names[]/tys[] ranged lo..hi, body and the
 * domain's where (NULL: none), parsed with output and errors kept quiet.
 * 0 on success.
 */
static int
u_parse(struct func *fn, int np, char names[][40], const int *tys,
    const int64_t *lo, const int64_t *hi, const char *body, const char *where,
    nid *bodyn)
{
	jmp_buf saved;
	int saved_rep = rep_on, fd, devnull, ok = 0, i;
	memset(fn, 0, sizeof *fn);
	fn->name = sym_cstr("_glslfn");
	fn->anon = 1;
	fn->np = np;
	for (i = 0; i < np; i++) {
		fn->p[i].name = sym_cstr(names[i]);
		fn->p[i].ty = (ty_t)u_tyof(tys[i]);
		if (tys[i] != TB) {
			fn->p[i].ranged = 1;
			fn->p[i].lo = lo[i];
			fn->p[i].hi = hi[i];
			fn->p[i].rlo = (double)lo[i];
			fn->p[i].rhi = (double)hi[i];
		}
	}
	fflush(stdout);
	fd = dup(1);
	devnull = open("/dev/null", O_WRONLY);
	if (devnull >= 0) {
		dup2(devnull, 1);
		close(devnull);
	}
	rep_on = 0;
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0) {
		*bodyn = calc_parse_text(body, fn, TY_INT);
		fn->body = *bodyn;
		if (where)
			fn->where = calc_parse_text(where, fn, TY_INT);
		ok = fn->np == np;
	}
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	rep_on = saved_rep;
	fflush(stdout);
	if (fd >= 0) {
		dup2(fd, 1);
		close(fd);
	}
	return ok ? 0 : -1;
}

/*
 * equal on the domain?  1 proved (exhaustive), 0 a difference (wit), -1
 * no proof (too many points, or the translation failed)
 */
static int
u_equiv(struct func *fn, nid a, nid b, char *wit, size_t nw, long *points)
{
	struct sbuf w = {0};
	int sampled = 0, m, i, fd, devnull;
	double pts = 1;
	for (i = 0; i < fn->np; i++)
		pts *= fn->p[i].ranged ? (double)(fn->p[i].hi - fn->p[i].lo + 1) : 2;
	if (pts > (double)(1 << 22))
		return -1;
	fflush(stdout);
	fd = dup(1);
	devnull = open("/dev/null", O_WRONLY);
	if (devnull >= 0) {
		dup2(devnull, 1);
		close(devnull);
	}
	m = equiv_check(fn, a, b, &w, &sampled);
	fflush(stdout);
	if (fd >= 0) {
		dup2(fd, 1);
		close(fd);
	}
	if (wit)
		snprintf(wit, nw, "%s", w.p ? w.p : "");
	free(w.p);
	*points += (long)pts;
	if (sampled)
		return -1;
	return m == 0;
}

/* ---- costs under the active model, loops by their trip counts ---- */

/* for (T i = K0; i < K1 | i <= K1; i++ | ++i | i += 1): i, K0, last */
static int
canon_loop(int s, int *var, int64_t *lo, int64_t *last)
{
	const struct gs *t = &G.s[s];
	const struct gx *c, *st;
	int a = SR(t->a), y;
	int64_t k;
	if (t->k != S_FOR || a < 0 || G.s[a].k != S_DECL || G.s[a].a < 0 ||
	    t->b < 0 || t->c < 0)
		return 0;
	y = G.s[a].sym;
	if (G.x[XR(G.s[a].a)].k != X_NUM)
		return 0;
	*lo = G.x[XR(G.s[a].a)].v;
	c = &G.x[XR(t->b)];
	if (c->k != X_BIN || G.x[XR(c->a)].k != X_VAR || G.x[XR(c->a)].sym != y ||
	    G.x[XR(c->b)].k != X_NUM)
		return 0;
	k = G.x[XR(c->b)].v;
	if (c->op == OP_LT)
		*last = k - 1;
	else if (c->op == OP_LE)
		*last = k;
	else
		return 0;
	st = &G.x[XR(t->c)];
	if (!((st->k == X_INC && st->op == OP_ADD) || (st->k == X_ASG &&
	    st->op == OP_ADD && G.x[XR(st->b)].k == X_NUM &&
	    G.x[XR(st->b)].v == 1)) || G.x[XR(st->a)].k != X_VAR ||
	    G.x[XR(st->a)].sym != y)
		return 0;
	if (A.nasg[y] > 1)
		return 0;	/* the body assigns it too */
	*var = y;
	return 1;
}

static int
is_pow2k(int x, int64_t *v)
{
	const struct gx *g = &G.x[XR(x)];
	if (g->k != X_NUM || g->v <= 0 || (g->v & (g->v - 1)))
		return 0;
	*v = g->v;
	return 1;
}

static double scost(int s);

static int *free_leaf;	/* expressions priced 0 (a candidate's leaves) */

static double
xcost(int x)
{
	const struct model *m = model_active();
	const struct gx *g;
	double c = 0;
	int64_t k;
	int i;
	x = XR(x);
	if (x < 0)
		return 0;
	if (free_leaf && free_leaf[x] >= 0)
		return 0;
	g = &G.x[x];
	switch (g->k) {
	case X_NUM:
	case X_VAR:
		return 0;
	case X_IDX:
		return xcost(g->b) + (G.x[XR(g->b)].k == X_NUM ? 0 : m->c[C_LOAD]);
	case X_MEM:
		return xcost(g->a);
	case X_CALL:
		for (i = 0; i < g->b; i++)
			c += xcost(G.xarg[g->a + i]);
		if (g->sym >= 0)
			return c + (G.f[g->sym].body >= 0 ? scost(G.f[g->sym].body) : 0);
		switch (-g->sym) {
		case B_MIN: case B_MAX: case B_ABS: return c + m->c[C_MINMAX];
		case B_CLAMP: return c + 2 * m->c[C_MINMAX];
		case B_BFE: return c + (cclass_available(m, C_BFE) ? m->c[C_BFE] :
		    m->c[C_SHIFT] + m->c[C_LOGIC]);
		case B_BOOL: return c + m->c[C_CMP];
		}
		return c;
	case X_UN:
		return xcost(g->a) + (g->op == OP_NEG ? m->c[C_IADD] : m->c[C_LOGIC]);
	case X_BIN:
		c = xcost(g->a) + xcost(g->b);
		switch (g->op) {
		case OP_ADD: case OP_SUB: return c + m->c[C_IADD];
		case OP_MUL:
			return c + (is_pow2k(g->a, &k) || is_pow2k(g->b, &k) ?
			    m->c[C_SHIFT] : m->c[C_IMUL]);
		case OP_DIV: case OP_MOD:
			if (is_pow2k(g->b, &k) && opnd_ty(g) == TU)
				return c + m->c[g->op == OP_DIV ? C_SHIFT : C_LOGIC];
			{
				/* signed or unsigned, by a constant or not */
				int u = opnd_ty(g) == TU, cls = g->op == OP_DIV ?
				    (u ? C_IDIV : C_SDIV) : (u ? C_IREM : C_SREM);
				if (is_pow2k(g->b, &k))	/* signed by 2^k: shifts */
					return c + (g->op == OP_DIV ? 3 * m->c[C_SHIFT] +
					    m->c[C_IADD] : 2 * m->c[C_SHIFT] +
					    2 * m->c[C_IADD] + m->c[C_LOGIC]);
				return c + m->c[G.x[XR(g->b)].k == X_NUM ?
				    C_DIVK(cls) : cls];
			}
		case OP_SHL: case OP_SHR: return c + m->c[C_SHIFT];
		case OP_LT: case OP_LE: case OP_GT: case OP_GE: case OP_EQ:
		case OP_NE: return c + m->c[C_CMP];
		}
		return c + m->c[C_LOGIC];
	case X_SEL:
		return xcost(g->a) + xcost(g->b) + xcost(g->c) + m->c[C_SELECT];
	case X_ASG:
		c = xcost(g->b) + xcost(g->a);
		if (g->op) {
			if (g->op == OP_MUL)
				c += m->c[C_IMUL];
			else if (g->op == OP_DIV || g->op == OP_MOD)
				c += m->c[C_IDIV];
			else if (g->op == OP_SHL || g->op == OP_SHR)
				c += m->c[C_SHIFT];
			else
				c += m->c[C_IADD];
		}
		if (G.x[XR(g->a)].k == X_IDX)
			c += m->c[C_STORE] - m->c[C_LOAD];
		return c;
	case X_INC:
		return xcost(g->a) + m->c[C_IADD];
	case X_COMMA:
		return xcost(g->a) + xcost(g->b);
	}
	return 0;
}

/* a statement's cost: a branch takes its dearer side, a loop of known trip
 * count is priced per iteration times the trips (else one iteration) */
static double
scost(int s)
{
	const struct model *m = model_active();
	const struct gs *t;
	double c = 0, a, b;
	int i, var;
	int64_t lo, last;
	s = SR(s);
	if (s < 0)
		return 0;
	t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			c += scost(G.slist[t->first + i]);
		return c;
	case S_DECL:
		return t->a >= 0 ? xcost(t->a) : 0;
	case S_EXPR:
	case S_RET:
		return t->a >= 0 ? xcost(t->a) : 0;
	case S_IF:
		a = scost(t->b);
		b = scost(t->c);
		return xcost(t->a) + m->c[C_BRANCH] + (a > b ? a : b);
	case S_FOR:
	case S_WHILE:
	case S_DO: {
		double trips = 1;
		if (t->k == S_FOR && canon_loop(s, &var, &lo, &last) && last >= lo)
			trips = (double)(last - lo + 1);
		c = t->k == S_FOR ? scost(t->a) : 0;
		return c + trips * (scost(t->d) + (t->b >= 0 ? xcost(t->b) : 0) +
		    (t->c >= 0 ? xcost(t->c) : 0) + m->c[C_BRANCH]);
	}
	}
	return 0;
}

/* ---- proposals ---- */

enum { P_FUNC, P_EXPR, P_FIT, P_INV, P_LOOP, P_TAB, P_CALLS };
struct prop {
	int f, kind, line;
	char what[1200];
	struct sbuf text;	/* GLSL */
	char proof[200];
	long points;
	double c0, c1;		/* per call (P_EXPR, P_LOOP: per execution) */
	char domain[1200];
	char relies[1200];
	int passes, nexit, exitl[16];	/* P_LOOP: the most passes, exits */
	int callee;		/* P_INV: the function called */
	uint8_t acts;		/* P_LOOP: the second pass acts before it leaves */
};
static struct prop *PR;
static int npr, cappr;

static struct prop *
prop_new(int f, int kind, int line)
{
	struct prop *p;
	if (npr == cappr) {
		cappr = cappr ? cappr * 2 : 16;
		PR = xrealloc(PR, (size_t)cappr * sizeof *PR);
	}
	p = &PR[npr++];
	memset(p, 0, sizeof *p);
	p->f = f;
	p->kind = kind;
	p->line = line;
	return p;
}

static void
xtext(char *buf, size_t n, int x)
{
	struct sbuf b = {0};
	px_text(&b, x, 0);
	snprintf(buf, n, "%s", b.p ? b.p : "");
	free(b.p);
}

/* ---- a whole function over its domain ---- */

struct dom {
	int nv, vars[MAXP + 8];
	int nr;
	struct av *rows;
	double points;
};

static int
fn_dom(int f, struct dom *D)
{
	int r, i, x;
	x = fn_rows(f, D->vars, &D->rows);
	D->nv = x / 1000;
	D->nr = x % 1000;
	D->points = 0;
	for (r = 0; r < D->nr; r++) {
		double p = 1;
		for (i = 0; i < D->nv; i++)
			p *= (double)(D->rows[r * D->nv + i].hi - D->rows[r * D->nv +
			    i].lo + 1);
		D->points += p;
	}
	return D->nr > 0;
}

static int64_t
run_fn(int f, int *ub)
{
	C.ub = 0;
	C.steps = 0;
	C.v[G.f[f].ret] = 0;
	(void)cx_stmt(G.f[f].body);
	*ub = C.ub;
	return C.v[G.f[f].ret];
}

/* every point of the domain: the function's values (ub: undefined) */
static int64_t *
fn_values(int f, const struct dom *D, uint8_t **ubs, long *np)
{
	long n = 0, cap = (long)D->points + 1;
	int64_t *v = xmalloc((size_t)cap * sizeof *v);
	int r, i;
	*ubs = xmalloc((size_t)cap);
	for (r = 0; r < D->nr; r++) {
		int64_t cur[MAXP + 8];
		const struct av *row = &D->rows[r * D->nv];
		for (i = 0; i < D->nv; i++)
			cur[i] = row[i].lo;
		for (;;) {
			int ub;
			for (i = 0; i < D->nv; i++)
				C.v[D->vars[i]] = cur[i];
			v[n] = run_fn(f, &ub);
			(*ubs)[n] = (uint8_t)ub;
			n++;
			for (i = 0; i < D->nv; i++) {
				if (cur[i] < row[i].hi) {
					cur[i]++;
					break;
				}
				cur[i] = row[i].lo;
			}
			if (i == D->nv || n >= cap)
				break;
		}
	}
	*np = n;
	return v;
}

/* the patched function against the reference values: 1 equal where the
 * original is defined */
static int
fn_same(int f, const struct dom *D, const int64_t *ref, const uint8_t *rub,
    char *wit, size_t nw)
{
	long n = 0;
	int r, i;
	for (r = 0; r < D->nr; r++) {
		int64_t cur[MAXP + 8];
		const struct av *row = &D->rows[r * D->nv];
		for (i = 0; i < D->nv; i++)
			cur[i] = row[i].lo;
		for (;;) {
			int ub;
			int64_t v;
			for (i = 0; i < D->nv; i++)
				C.v[D->vars[i]] = cur[i];
			v = run_fn(f, &ub);
			if (!rub[n] && (ub || v != ref[n])) {
				if (wit) {
					size_t at = 0;
					for (i = 0; i < D->nv && at < nw; i++)
						at += (size_t)snprintf(wit + at, nw - at, "%s%s = %lld",
						    i ? ", " : "", G.y[D->vars[i]].name,
						    (long long)cur[i]);
				}
				return 0;
			}
			n++;
			for (i = 0; i < D->nv; i++) {
				if (cur[i] < row[i].hi) {
					cur[i]++;
					break;
				}
				cur[i] = row[i].lo;
			}
			if (i == D->nv)
				break;
		}
	}
	return 1;
}

static void
where_text(struct sbuf *b, char names[][40], const int *tys, int nv,
    const struct av *rows, int nr, const int64_t *blo, const int64_t *bhi)
{
	int r, i;
	for (r = 0; r < nr; r++) {
		int any = 0;
		sb_puts(b, r ? " || (" : "(");
		for (i = 0; i < nv; i++) {
			const struct av *v = &rows[r * nv + i];
			if (tys[i] == TB) {
				if (v->lo == v->hi) {
					sb_printf(b, "%s%s%s", any ? " && " : "", v->lo ? "" : "!",
					    names[i]);
					any = 1;
				}
				continue;
			}
			if (v->lo > blo[i]) {
				sb_printf(b, "%s%s >= %lld", any ? " && " : "", names[i],
				    (long long)v->lo);
				any = 1;
			}
			if (v->hi < bhi[i]) {
				sb_printf(b, "%s%s <= %lld", any ? " && " : "", names[i],
				    (long long)v->hi);
				any = 1;
			}
		}
		if (!any)
			sb_puts(b, "true");
		sb_puts(b, ")");
	}
}

/*
 * the ucas proof that body texts a and b agree on rows (types tys, names):
 * 1 proved, 0 refuted, -1 no proof.  Rows are proved together under a
 * where clause when their bounding box is small enough, else one by one.
 */
static int
prove_rows(char names[][40], const int *tys, int nv, const struct av *rows,
    int nr, const char *a, const char *b, long *points, char *wit, size_t nw)
{
	int64_t blo[MAXP + 8], bhi[MAXP + 8];
	double box = 1;
	int i, r, res = 1;
	struct func fn;
	nid na, nb;
	if (nv > MAXP)
		return -1;
	for (i = 0; i < nv; i++) {
		blo[i] = rows[i].lo;
		bhi[i] = rows[i].hi;
		for (r = 1; r < nr; r++) {
			blo[i] = rows[r * nv + i].lo < blo[i] ? rows[r * nv + i].lo : blo[i];
			bhi[i] = rows[r * nv + i].hi > bhi[i] ? rows[r * nv + i].hi : bhi[i];
		}
		box *= (double)(bhi[i] - blo[i] + 1);
	}
	if (box <= (double)(1 << 22)) {
		struct sbuf w = {0};
		int rc;
		if (nr > 1)
			where_text(&w, names, tys, nv, rows, nr, blo, bhi);
		if (u_parse(&fn, nv, names, tys, blo, bhi, a, nr > 1 ? w.p : NULL,
		    &na) != 0) {
			free(w.p);
			return -1;
		}
		{
			struct func fb;
			if (u_parse(&fb, nv, names, tys, blo, bhi, b, NULL, &nb) != 0) {
				free(w.p);
				return -1;
			}
		}
		free(w.p);
		/* points counted are the rows' own, not the box's */
		{
			long dummy = 0;
			rc = u_equiv(&fn, na, nb, wit, nw, &dummy);
		}
		if (rc == 1)
			for (r = 0; r < nr; r++) {
				double p = 1;
				for (i = 0; i < nv; i++)
					p *= (double)(rows[r * nv + i].hi - rows[r * nv + i].lo + 1);
				*points += (long)p;
			}
		return rc;
	}
	/* one row at a time; a row too big is split on its widest variable */
	for (r = 0; r < nr && res == 1; r++) {
		struct av row[MAXP + 8];
		double p = 1;
		int wv = 0;
		for (i = 0; i < nv; i++) {
			row[i] = rows[r * nv + i];
			p *= (double)(row[i].hi - row[i].lo + 1);
			if (row[i].hi - row[i].lo > row[wv].hi - row[wv].lo)
				wv = i;
		}
		if (p > (double)(1 << 26))
			return -1;
		if (p <= (double)(1 << 22)) {
			res = prove_rows(names, tys, nv, row, 1, a, b, points, wit, nw);
			continue;
		}
		{
			int64_t lo = row[wv].lo, hi = row[wv].hi, step;
			step = (int64_t)((double)(hi - lo + 1) / (p / (double)(1 << 22)));
			if (step < 1)
				step = 1;
			for (; lo <= hi && res == 1; lo += step) {
				row[wv].lo = lo;
				row[wv].hi = lo + step - 1 < hi ? lo + step - 1 : hi;
				res = prove_rows(names, tys, nv, row, 1, a, b, points, wit,
				    nw);
			}
		}
	}
	return res;
}

/* the function's ucas text with the current patches: NULL if it cannot */
static char *
fn_ucas(int f, const struct dom *D, char names[][40], int *tys)
{
	struct ut U;
	int i, j;
	memset(&U, 0, sizeof U);
	U.ok = 1;
	u_fn_block(&U, &U.b, f, NULL, 0, NULL);
	for (i = 0; i < U.ngl; i++) {
		for (j = 0; j < D->nv && D->vars[j] != U.gl[i]; j++)
			;
		if (j == D->nv)
			U.ok = 0;	/* a global outside the domain's inputs */
	}
	if (!U.ok) {
		free(U.b.p);
		return NULL;
	}
	for (i = 0; i < D->nv; i++) {
		struct sbuf nb = {0};
		struct uframe F0 = {0, f, -1, 0};
		u_name(&U, &nb, D->vars[i], &F0);
		snprintf(names[i], 40, "%s", nb.p);
		free(nb.p);
		tys[i] = G.y[D->vars[i]].ty;
	}
	return U.b.p;
}

/* ---- patches ---- */

static int capxrep, capsrep;

static void
patch_room(void)
{
	int i;
	if (G.nx + 8 > capxrep) {
		int n = (G.nx + 8) * 2;
		xrep = xrealloc(xrep, (size_t)n * sizeof *xrep);
		for (i = capxrep; i < n; i++)
			xrep[i] = -1;
		capxrep = n;
	}
	if (G.ns + 8 > capsrep) {
		int n = (G.ns + 8) * 2;
		srep = xrealloc(srep, (size_t)n * sizeof *srep);
		for (i = capsrep; i < n; i++)
			srep[i] = -1;
		capsrep = n;
	}
	if (recn && G.nx + 8 > capxrep / 2) {
		/* recn and the bind table follow the expressions */
	}
}

static int
new_num(int ty, int64_t v, int line)
{
	int x = nx_new(X_NUM, ty, line);
	G.x[x].v = v;
	patch_room();
	return x;
}

static int
new_call(int b, int ty, const int *args, int na, int line)
{
	int x = call_node(-b, ty, args, na, line);
	patch_room();
	return x;
}

static int
new_bin(int op, int ty, int a, int b, int line)
{
	int x = nx_new(X_BIN, ty, line);
	G.x[x].op = (uint8_t)op;
	G.x[x].a = a;
	G.x[x].b = b;
	patch_room();
	return x;
}

/* the expressions reachable from statement s (patches followed) */
static void
reach_x(int x, int *out, int *n, int max)
{
	const struct gx *g;
	int i;
	x = XR(x);
	if (x < 0 || *n >= max)
		return;
	out[(*n)++] = x;
	g = &G.x[x];
	switch (g->k) {
	case X_IDX:
	case X_BIN:
	case X_ASG:
	case X_COMMA:
		reach_x(g->a, out, n, max);
		reach_x(g->b, out, n, max);
		return;
	case X_UN:
	case X_INC:
	case X_MEM:
		reach_x(g->a, out, n, max);
		return;
	case X_SEL:
		reach_x(g->a, out, n, max);
		reach_x(g->b, out, n, max);
		reach_x(g->c, out, n, max);
		return;
	case X_CALL:
		for (i = 0; i < g->b; i++)
			reach_x(G.xarg[g->a + i], out, n, max);
		return;
	}
}

static void
reach_s(int s, int *xs, int *nx, int *ss, int *nss, int max)
{
	const struct gs *t;
	int i;
	s = SR(s);
	if (s < 0)
		return;
	t = &G.s[s];
	if (ss && *nss < max)
		ss[(*nss)++] = s;
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			reach_s(G.slist[t->first + i], xs, nx, ss, nss, max);
		return;
	case S_DECL:
	case S_EXPR:
	case S_RET:
		if (t->a >= 0)
			reach_x(t->a, xs, nx, max);
		return;
	case S_IF:
		reach_x(t->a, xs, nx, max);
		reach_s(t->b, xs, nx, ss, nss, max);
		reach_s(t->c, xs, nx, ss, nss, max);
		return;
	case S_FOR:
	case S_WHILE:
	case S_DO:
		reach_s(t->a, xs, nx, ss, nss, max);
		if (t->b >= 0)
			reach_x(t->b, xs, nx, max);
		if (t->c >= 0)
			reach_x(t->c, xs, nx, max);
		reach_s(t->d, xs, nx, ss, nss, max);
		return;
	}
}

/* ---- loops that leave early: their first two passes (probe_loop) ---- */

static const char *fsrc;	/* the source the lines are of */

/* line n of the source, trimmed, cut at 60 characters */
static void
src_quote(char *buf, size_t n, int line)
{
	const char *p = fsrc, *e, *m;
	size_t len;
	int l = 1;
	*buf = 0;
	/* glslc.c's "#line N" before the source: the line after it is N (as
	 * spirv.c's src_line reads it) */
	if (p && (m = strstr(p, "#line ")) != NULL && atoi(m + 6) > 0 &&
	    atoi(m + 6) < 10000) {
		l = atoi(m + 6);
		p = strchr(m, '\n') ? strchr(m, '\n') + 1 : m + strlen(m);
	}
	while (p && *p && l < line) {
		if (*p++ == '\n')
			l++;
	}
	if (!p || !*p)
		return;
	while (*p == ' ' || *p == '\t')
		p++;
	e = strchr(p, '\n');
	len = e ? (size_t)(e - p) : strlen(p);
	while (len && (p[len - 1] == ' ' || p[len - 1] == '\t' ||
	    p[len - 1] == '\r'))
		len--;
	if (len > 60)
		snprintf(buf, n, "%.57s...", p);
	else
		snprintf(buf, n, "%.*s", (int)len, p);
}

/* the loop's own exits a pass ran: its breaks (not those of loops inside
 * it) and every return */
static void
loop_exits(int s, int inloop, const uint8_t *r, int *lines, int *kinds,
    int *n, int max)
{
	const struct gs *t;
	int i;
	if (s < 0 || !r[s])
		return;
	t = &G.s[s];
	switch (t->k) {
	case S_RET:
	case S_BRK:
		if (t->k == S_BRK && inloop)
			return;
		for (i = 0; i < *n && lines[i] != t->line; i++)
			;
		if (i == *n && *n < max) {
			kinds[*n] = t->k;
			lines[(*n)++] = t->line;
		}
		return;
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			loop_exits(G.slist[t->first + i], inloop, r, lines, kinds, n,
			    max);
		return;
	case S_IF:
		loop_exits(t->b, inloop, r, lines, kinds, n, max);
		if (t->c >= 0)
			loop_exits(t->c, inloop, r, lines, kinds, n, max);
		return;
	case S_FOR:
	case S_WHILE:
	case S_DO:
		loop_exits(t->d, 1, r, lines, kinds, n, max);
		return;
	}
}

/* x writes nothing but the variables in loc, and calls nothing that
 * writes or that the level does not know */
static int
quiet_x(int x, const uint8_t *loc)
{
	const struct gx *g;
	int i;
	if (x < 0)
		return 1;
	g = &G.x[x];
	switch (g->k) {
	case X_NUM:
	case X_VAR:
		return 1;
	case X_ASG:
	case X_INC:
		if (G.x[g->a].k != X_VAR || !loc[G.x[g->a].sym])
			return 0;
		return g->k == X_INC || quiet_x(g->b, loc);
	case X_CALL:
		if (g->sym == -B_UNK || (g->sym >= 0 && (A.impure[g->sym] ||
		    A.nwrl[g->sym])))
			return 0;
		if (g->sym >= 0)
			for (i = 0; i < g->b && i < G.f[g->sym].np; i++)
				if (G.y[G.f[g->sym].p[i]].out)
					return 0;
		for (i = 0; i < g->b; i++)
			if (!quiet_x(G.xarg[g->a + i], loc))
				return 0;
		return 1;
	case X_UN:
	case X_MEM:
		return quiet_x(g->a, loc);
	case X_SEL:
		return quiet_x(g->a, loc) && quiet_x(g->b, loc) &&
		    quiet_x(g->c, loc);
	default:
		return quiet_x(g->a, loc) && quiet_x(g->b, loc);
	}
}

/* what a pass ran (r) does that outlives the pass: a write outside the
 * loop's own variables (loc), a call that writes, a return, a loop */
static int
acts_s(int s, const uint8_t *r, const uint8_t *loc)
{
	const struct gs *t;
	int i;
	if (s < 0 || !r[s])
		return 0;
	t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			if (acts_s(G.slist[t->first + i], r, loc))
				return 1;
		return 0;
	case S_IF:
		return !quiet_x(t->a, loc) || acts_s(t->b, r, loc) ||
		    acts_s(t->c, r, loc);
	case S_DECL:
	case S_EXPR:
		return !quiet_x(t->a, loc);
	case S_BRK:
	case S_CONT:
	case S_EMPTY:
		return 0;
	}
	return 1;
}

/* a pass's cost: what it ran (r), priced as scost prices a statement */
static double
pcost(int s, const uint8_t *r)
{
	const struct model *m = model_active();
	const struct gs *t;
	double c = 0, a, b;
	int i;
	if (s < 0 || !r[s])
		return 0;
	t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++)
			c += pcost(G.slist[t->first + i], r);
		return c;
	case S_IF:
		a = pcost(t->b, r);
		b = t->c >= 0 ? pcost(t->c, r) : 0;
		/* one side never ran: its test is a constant there */
		if (!r[t->b] || (t->c >= 0 && !r[t->c]))
			return a + b;
		return xcost(t->a) + m->c[C_BRANCH] + (a > b ? a : b);
	}
	return scost(s);
}

/* what a pass reached (the statement) but never ran inside it: a branch
 * never taken, the rest of a block after it always left; loops inside
 * are not looked into */
static void
never_ran(int s, const uint8_t *r, struct sbuf *b, int *n)
{
	const struct gs *t;
	char q[80];
	int i;
	if (s < 0 || !r[s])
		return;
	t = &G.s[s];
	switch (t->k) {
	case S_BLOCK:
		for (i = 0; i < t->n; i++) {
			int k = G.slist[t->first + i];
			if (!r[k]) {
				if (G.s[k].k == S_EMPTY)
					continue;
				if (*n < 6) {
					src_quote(q, sizeof q, G.s[k].line);
					sb_printf(b, "%sfrom line %d on (`%s`)", *n ? "; " : "",
					    G.s[k].line, q);
				}
				(*n)++;
				return;
			}
			never_ran(k, r, b, n);
		}
		return;
	case S_IF:
		if (!r[t->b] && G.s[t->b].k != S_EMPTY) {
			if (*n < 6) {
				src_quote(q, sizeof q, t->line);
				sb_printf(b, "%sthe if at line %d (`%s`)", *n ? "; " : "",
				    t->line, q);
			}
			(*n)++;
		} else
			never_ran(t->b, r, b, n);
		if (t->c >= 0) {
			if (!r[t->c] && r[t->b]) {
				if (*n < 6)
					sb_printf(b, "%sthe else of the if at line %d", *n ?
					    "; " : "", t->line);
				(*n)++;
			} else
				never_ran(t->c, r, b, n);
		}
		return;
	}
}

static void callers_text(struct sbuf *b, int f);

/* "by a break at line 12 or 14, or a return at line 15" */
static void
exits_text(struct sbuf *b, const int *lines, const int *kinds, int n)
{
	int kind, i, said = 0;
	for (kind = 0; kind < 2; kind++) {
		int want = kind ? S_RET : S_BRK, cnt = 0, m = 0;
		for (i = 0; i < n; i++)
			cnt += kinds[i] == want;
		if (!cnt)
			continue;
		sb_printf(b, "%s a %s at line ", said ? ", or" : "by", kind ?
		    "return" : "break");
		for (i = 0; i < n; i++)
			if (kinds[i] == want) {
				sb_printf(b, "%s%d", m == 0 ? "" : m + 1 == cnt ? " or " :
				    ", ", lines[i]);
				m++;
			}
		said = 1;
	}
}

/* " (j = 0u)" when v is one value of variable var, else "" */
static void
var_is(struct sbuf *b, int var, struct av v)
{
	if (var < 0 || v.bot || v.lo != v.hi)
		return;
	sb_printf(b, " (%s = ", G.y[var].name);
	num_text(b, v.lo, G.y[var].ty);
	sb_puts(b, ")");
}

/*
 * Loops whose first two passes (run apart in the probe round) show they
 * never run a third, or a second: the first pass peeled out, or the loop
 * an if.  A dataflow fact over the inferred domains; nothing to prove
 * beyond them.  Loops whose bounds say as much (a constant count of one
 * or two) are left out.
 */
static void
search_loops(int f)
{
	int *xs, *ss, nx = 0, nss = 0, i, cap = G.nx + G.ns + 64;
	if (LP == NULL || G.f[f].body < 0)
		return;
	xs = xmalloc((size_t)cap * sizeof *xs);
	ss = xmalloc((size_t)cap * sizeof *ss);
	reach_s(G.f[f].body, xs, &nx, ss, &nss, cap);
	for (i = 0; i < nss; i++) {
		int s = ss[i], var, lines[16], kinds[16], ne = 0, l1[16], k1[16],
		    n1 = 0, k, *ls, nls = 0, *lx, nlx = 0, nnever = 0, once, acts = 0,
		    lo = INT_MAX, hi = 0;
		const struct gs *t = &G.s[s];
		const struct lprobe *L = &LP[s];
		uint8_t *loc;
		int64_t k0, klast;
		struct prop *p;
		struct sbuf w = {0}, v1 = {0}, cl = {0}, vt = {0}, nb = {0};
		char q[80];
		double c1;
		if ((t->k != S_FOR && t->k != S_WHILE) || !L->seen || !L->ran1 ||
		    L->more)
			continue;
		if (canon_loop(s, &var, &k0, &klast) && klast - k0 + 1 <= 2)
			continue;
		once = !L->ran2;	/* at most one pass */
		loop_exits(t->d, 0, L->r1, l1, k1, &n1, 16);
		if (!once)
			loop_exits(t->d, 0, L->r2, lines, kinds, &ne, 16);
		if (once && !L->at2 && n1 == 0)
			continue;	/* the first pass never ends: no fact here */
		/* the loop's own variables: declared in it */
		loc = xcalloc((size_t)A.ns + 1, 1);
		ls = xmalloc((size_t)cap * sizeof *ls);
		lx = xmalloc((size_t)cap * sizeof *lx);
		reach_s(s, lx, &nlx, ls, &nls, cap);
		for (k = 0; k < nls; k++) {
			if (G.s[ls[k]].k == S_DECL)
				loc[G.s[ls[k]].sym] = 1;
			if (G.s[ls[k]].line < lo)
				lo = G.s[ls[k]].line;
			if (G.s[ls[k]].line > hi)
				hi = G.s[ls[k]].line;
		}
		var = t->k == S_FOR && t->c >= 0 ? asg_target(t->c) : -1;
		if (!once)
			acts = acts_s(t->d, L->r2, loc) || !quiet_x(t->b, loc) ||
			    (t->c >= 0 && !quiet_x(t->c, loc));
		var_is(&v1, var, L->v1);
		if (!once) {
			sb_puts(&w, "the loop leaves on its second pass");
			var_is(&w, var, L->v2);
			sb_puts(&w, " ");
			exits_text(&w, lines, kinds, ne);
			if (L->fail2)
				sb_printf(&w, "%s its test", ne ? ", or by" : "by");
			if (acts)
				sb_printf(&w, ": peel its first pass%s out of the loop; "
				    "what is left is the second pass once, not a loop",
				    v1.p ? v1.p : "");
			else
				sb_printf(&w, ", doing nothing on the way: peel its first "
				    "pass%s out and the loop goes", v1.p ? v1.p : "");
			if (L->brk1 && n1) {
				sb_puts(&w, "; the first pass may leave too, ");
				exits_text(&w, l1, k1, n1);
			}
		} else {
			if (L->at2) {
				sb_printf(&w, "the loop's test fails on its second pass: its "
				    "body runs at most once%s, an if and not a loop", v1.p ?
				    v1.p : "");
				if (n1) {
					sb_puts(&w, "; that pass may also leave ");
					exits_text(&w, l1, k1, n1);
				}
			} else {
				sb_printf(&w, "the loop never comes back to its test: its body "
				    "runs at most once%s, leaving ", v1.p ? v1.p : "");
				exits_text(&w, l1, k1, n1);
				sb_puts(&w, ": an if, not a loop");
			}
		}
		p = prop_new(f, P_LOOP, t->line);
		snprintf(p->what, sizeof p->what, "%s", w.p);
		p->passes = once ? 1 : 2;
		if (once) {
			p->nexit = n1;
			memcpy(p->exitl, l1, (size_t)n1 * sizeof *l1);
		} else {
			p->nexit = ne;
			memcpy(p->exitl, lines, (size_t)ne * sizeof *lines);
		}
		p->acts = (uint8_t)acts;
		src_quote(q, sizeof q, t->line);
		sb_printf(&p->text, "%s\n  ->  its first pass alone%s", q,
		    v1.p ? v1.p : "");
		if (!once && acts)
			sb_puts(&p->text, ", then its second once");
		never_ran(t->d, L->r1, &nb, &nnever);
		if (nnever)
			sb_printf(&p->text, "; in the first pass these never run: %s%s",
			    nb.p, nnever > 6 ? "; and more" : "");
		snprintf(p->proof, sizeof p->proof, "dataflow: its first two passes "
		    "run apart in the abstract interpretation, from every state it "
		    "starts from; %s", once && L->at2 ? "the test fails on every "
		    "state of the second" : once ? "no state of the first comes "
		    "back to the test" : "no state of the second comes back to the "
		    "test");
		/* before: the loop as scost prices it; after: the passes kept */
		p->c0 = scost(s);
		c1 = t->k == S_FOR && t->a >= 0 ? scost(t->a) : 0;
		c1 += pcost(t->d, L->r1);
		if (L->fail1)
			c1 += xcost(t->b) + model_active()->c[C_BRANCH];
		if (!once && acts) {
			c1 += pcost(t->d, L->r2) + (t->c >= 0 ? xcost(t->c) : 0);
			if (L->fail2)
				c1 += xcost(t->b) + model_active()->c[C_BRANCH];
		}
		p->c1 = c1;
		if (var >= 0 && !L->v1.bot) {
			sb_printf(&vt, "%s %s ", G.y[var].name, L->v1.lo == L->v1.hi ?
			    "=" : "in");
			av_text(&vt, L->v1, G.y[var].ty);
			sb_puts(&vt, " as the first pass starts");
			if (!once && !L->v2.bot) {
				sb_printf(&vt, ", %s %s ", G.y[var].name, L->v2.lo ==
				    L->v2.hi ? "=" : "in");
				av_text(&vt, L->v2, G.y[var].ty);
				sb_puts(&vt, " as the second does");
			}
		}
		snprintf(p->domain, sizeof p->domain, "%s", vt.p ? vt.p : "");
		if (is_root(f))
			snprintf(p->relies, sizeof p->relies, "the states the loop "
			    "(lines %d to %d) starts from, inferred from %s's start", lo,
			    hi, G.f[f].name);
		else {
			callers_text(&cl, f);
			snprintf(p->relies, sizeof p->relies, "the states the loop "
			    "(lines %d to %d) starts from, inferred from %s's calls: %s",
			    lo, hi, G.f[f].name, cl.p ? cl.p : "");
		}
		free(w.p);
		free(v1.p);
		free(cl.p);
		free(vt.p);
		free(nb.p);
		free(loc);
		free(ls);
		free(lx);
	}
	free(xs);
	free(ss);
}

/* ---- loops of a constant count that compute a map of one small value ---- */

static void reads_x(int x, uint8_t *rd, int *pure);

static int glsl_u32(struct sbuf *b, nid e, const char *xn, int xty);

/* a uint constant as GLSL writes it */
static void
glsl_u(struct sbuf *b, int64_t v)
{
	if (v >= 4096 || (v >= 15 && ((v + 1) & v) == 0))
		sb_printf(b, "0x%llXu", (unsigned long long)(uint32_t)v);
	else
		sb_printf(b, "%lluu", (unsigned long long)(uint32_t)v);
}

/* e (one of find's maps of the variable xn, in u32; at the top a
 * conversion to the map's own integer type, which keeps the value) as
 * GLSL; 0 when it has an operation this does not write */
static int
glsl_nid(struct sbuf *b, nid e, const char *xn, int xty)
{
	while (N.op[e] == O_CAST && TY_SIZED(N.ty[N.a[e]]) && (TYK(N.ty[e]) ==
	    K_INT || TYK(N.ty[e]) == K_LIT || TYK(N.ty[e]) == K_SINT))
		e = N.a[e];
	return glsl_u32(b, e, xn, xty);
}

static int
glsl_u32(struct sbuf *b, nid e, const char *xn, int xty)
{
	static const struct {
		int op;
		const char *s;
	} ops[] = {{O_ADD, "+"}, {O_SUB, "-"}, {O_MUL, "*"}, {O_AND, "&"},
	    {O_OR, "|"}, {O_XOR, "^"}, {O_SHL, "<<"}, {O_SHR, ">>"}};
	int op = N.op[e], i;
	if (op == O_CONST) {
		if (N.ki[e] < 0 || N.ki[e] > 0xFFFFFFFFLL)
			return 0;
		glsl_u(b, N.ki[e]);
		return 1;
	}
	if (op == O_VAR) {
		sb_printf(b, xty == TU ? "%s" : "uint(%s)", xn);
		return 1;
	}
	if (op == O_CAST)
		return N.ty[e] == TY(K_UINT, 32) && glsl_u32(b, N.a[e], xn, xty);
	if (N.ty[e] != TY(K_UINT, 32))
		return 0;
	for (i = 0; i < (int)(sizeof ops / sizeof ops[0]); i++)
		if (ops[i].op == op) {
			/* a constant operand of + * & | ^ written second */
			int sw = N.op[N.a[e]] == O_CONST && N.op[N.b[e]] != O_CONST &&
			    op != O_SUB && op != O_SHL && op != O_SHR;
			sb_putc(b, '(');
			if (!glsl_u32(b, sw ? N.b[e] : N.a[e], xn, xty))
				return 0;
			sb_printf(b, " %s ", ops[i].s);
			if (!glsl_u32(b, sw ? N.a[e] : N.b[e], xn, xty))
				return 0;
			sb_putc(b, ')');
			return 1;
		}
	return 0;
}

/* "(a) * b" -> "a * b" at the top */
static void
glsl_strip(struct sbuf *b)
{
	size_t n = b->p ? strlen(b->p) : 0, i;
	int d = 0;
	if (n < 2 || b->p[0] != '(' || b->p[n - 1] != ')')
		return;
	for (i = 0; i < n; i++) {
		d += b->p[i] == '(' ? 1 : b->p[i] == ')' ? -1 : 0;
		if (d == 0 && i + 1 < n)
			return;	/* the first ( closes before the end */
	}
	memmove(b->p, b->p + 1, n - 2);
	b->p[n - 2] = 0;
}

/* what loop s does is only to write variables: no buffer, array or out
 * argument, no call that writes or the level does not know, no return */
static int
loop_quiet(int s, int *xs, int *ss, int cap)
{
	int nx = 0, nss = 0, i;
	reach_s(s, xs, &nx, ss, &nss, cap);
	if (nx >= cap || nss >= cap)
		return 0;
	for (i = 0; i < nss; i++)
		if (G.s[ss[i]].k == S_RET || G.s[ss[i]].k == S_UNK)
			return 0;
	for (i = 0; i < nx; i++) {
		const struct gx *g = &G.x[xs[i]];
		if ((g->k == X_ASG || g->k == X_INC) && (G.x[g->a].k != X_VAR ||
		    G.y[G.x[g->a].sym].arr))
			return 0;
		if (g->k == X_CALL && (g->sym == -B_UNK || (g->sym >= 0 &&
		    (A.impure[g->sym] || A.nwrl[g->sym]))))
			return 0;
		if (g->k == X_CALL && g->sym >= 0) {
			int k;
			for (k = 0; k < g->b && k < G.f[g->sym].np; k++)
				if (G.y[G.f[g->sym].p[k]].out)
					return 0;
		}
	}
	return 1;
}

/*
 * A loop of a constant count whose one result is a function of one value
 * of at most 256 (every other value it reads one constant where it
 * starts): run at each value with the concrete interpreter, the table
 * handed to find's maps of one value (idrw.c: a table in one constant, a
 * multiply and a shift, a field for each bit) and to the table itself,
 * the cheapest under the model proposed in place of the loop when it
 * costs less, proved equal to the table at every point.
 */
static void
search_tab(int f)
{
	int *xs, *ss, nx = 0, nss = 0, i, cap = G.nx + G.ns + 64;
	if (LP == NULL || G.f[f].body < 0)
		return;
	xs = xmalloc((size_t)cap * sizeof *xs);
	ss = xmalloc((size_t)cap * sizeof *ss);
	reach_s(G.f[f].body, xs, &nx, ss, &nss, cap);
	for (i = 0; i < nss; i++) {
		int s = ss[i], var, k, *lx, *ls, nlx = 0, nls = 0, y = -1, x = -1,
		    n, K, ok = 1, pure = 1, best = -1, nr = 0, q;
		const struct gs *t = &G.s[s];
		const struct av *E = LP[s].entry;
		int64_t k0, klast, *tab = NULL, y0 = 0;
		uint8_t *rd, *decl;
		struct sbuf body = {0}, txt = {0}, form = {0};
		struct func fn;
		nid bn;
		struct irw *r = NULL;
		char names[1][40];
		int tys[1];
		int64_t blo[1], bhi[1];
		double c0, c1, cbest = 0, ctab = 0;
		long pts = 0;
		int usef;
		struct prop *p;
		if (t->k != S_FOR || !LP[s].seen || E == NULL ||
		    !canon_loop(s, &var, &k0, &klast) || klast < k0 ||
		    klast - k0 + 1 > 64)
			continue;
		lx = xmalloc((size_t)cap * sizeof *lx);
		ls = xmalloc((size_t)cap * sizeof *ls);
		rd = xcalloc((size_t)A.ns + 1, 1);
		decl = xcalloc((size_t)A.ns + 1, 1);
		if (!loop_quiet(s, lx, ls, cap))
			ok = 0;
		reach_s(s, lx, &nlx, ls, &nls, cap);
		for (k = 0; k < nls; k++)
			if (G.s[ls[k]].k == S_DECL)
				decl[G.s[ls[k]].sym] = 1;
		/* its one result: the variable it assigns that outlives it */
		for (k = 0; k < nlx && ok; k++) {
			const struct gx *g = &G.x[lx[k]];
			int z;
			if (g->k != X_ASG && g->k != X_INC)
				continue;
			z = G.x[g->a].sym;
			if (decl[z] || z == var)
				continue;
			if (y >= 0 && y != z)
				ok = 0;
			y = z;
		}
		for (k = 0; k < nlx && ok; k++)
			reads_x(lx[k], rd, &pure);
		/* what it reads: one value of a few, the rest constants */
		for (k = 0; k < A.ns && ok && y >= 0; k++) {
			const struct av *v = &E[k];
			if (!rd[k] || decl[k] || k == var || G.y[k].tab ||
			    (G.y[k].kind == Y_CONST && G.y[k].init >= 0 && !G.y[k].arr))
				continue;
			if (G.y[k].arr || G.y[k].kind == Y_BUF || (G.y[k].ty != TU &&
			    G.y[k].ty != TI && G.y[k].ty != TB) || v->bot) {
				ok = 0;
				break;
			}
			if (v->lo == v->hi)
				continue;
			if (x >= 0 || k == y || v->hi - v->lo + 1 > 256)
				ok = 0;
			x = k;
		}
		if (!ok || y < 0 || x < 0 || G.y[y].ty == TB || G.y[y].arr ||
		    (rd[y] && E[y].lo != E[y].hi)) {
			free(lx);
			free(ls);
			free(rd);
			free(decl);
			continue;
		}
		y0 = rd[y] ? E[y].lo : 0;
		K = (int)(E[x].hi - E[x].lo + 1);
		tab = xmalloc((size_t)K * sizeof *tab);
		for (n = 0; n < K && ok; n++) {
			for (k = 0; k < A.ns; k++)
				if (rd[k] && !E[k].bot && E[k].lo == E[k].hi)
					C.v[k] = norm(G.y[k].ty, E[k].lo);
			C.v[x] = norm(G.y[x].ty, E[x].lo + n);
			C.v[y] = norm(G.y[y].ty, y0);
			C.ub = 0;
			C.steps = 0;
			(void)cx_stmt(s);
			if (C.ub)
				ok = 0;
			tab[n] = norm(G.y[y].ty, C.v[y]);
		}
		C.ub = 0;
		/* one value at every point: the loop is that constant */
		for (n = 1; ok && n < K && tab[n] == tab[0]; n++)
			;
		if (ok && n == K) {
			struct sbuf v0 = {0}, vk = {0};
			char q0[80];
			c0 = scost(s);
			c1 = model_active()->c[C_IADD];	/* its store */
			if (c1 < c0) {
				p = prop_new(f, P_TAB, t->line);
				num_text(&v0, y0, G.y[y].ty);
				if (G.y[y].ty == TU)
					glsl_u(&vk, tab[0]);
				else
					sb_printf(&vk, "%lld", (long long)tab[0]);
				snprintf(p->what, sizeof p->what, "the loop leaves %s = %s "
				    "whatever %s is (%s = %s as it starts)", G.y[y].name, vk.p,
				    G.y[x].name, G.y[y].name, v0.p);
				src_quote(q0, sizeof q0, t->line);
				sb_printf(&p->text, "%s\n  ->  %s = %s;", q0, G.y[y].name, vk.p);
				snprintf(p->proof, sizeof p->proof, "the loop run at each of the "
				    "%d values of %s: every one leaves %s", K, G.y[x].name, vk.p);
				p->points = K;
				p->c0 = c0;
				p->c1 = c1;
				snprintf(p->domain, sizeof p->domain, "%s in %lld..%lld",
				    G.y[x].name, (long long)E[x].lo, (long long)E[x].hi);
				snprintf(p->relies, sizeof p->relies, "%s where the loop starts "
				    "(the analysis's domain from %s)", p->domain, is_root(f) ?
				    "the start of main" : "the calls");
			}
			free(v0.p);
			free(vk.p);
			free(tab);
			free(lx);
			free(ls);
			free(rd);
			free(decl);
			continue;
		}
		/* the table, read by find's maps */
		snprintf(names[0], sizeof names[0], "%s", G.y[x].name);
		tys[0] = G.y[x].ty == TB ? TU : G.y[x].ty;
		blo[0] = E[x].lo;
		bhi[0] = E[x].hi;
		sb_puts(&body, "[");
		for (n = 0; n < K; n++)
			sb_printf(&body, "%s%lld", n ? ", " : "", (long long)tab[n]);
		if (E[x].lo)
			sb_printf(&body, "][%s - %lld]", names[0], (long long)E[x].lo);
		else
			sb_printf(&body, "][%s]", names[0]);
		if (ok && u_parse(&fn, 1, names, tys, blo, bhi, body.p, NULL, &bn) ==
		    0) {
			nr = irw_cands(&fn, bn, &r);
			ctab = fuse_expr_cost(model_active(), bn);
			for (q = 0; q < nr; q++) {
				double c;
				struct sbuf g = {0};
				long pp = 0;
				if (r[q].nearmiss)
					continue;
				c = fuse_expr_cost(model_active(), r[q].e);
				if ((best >= 0 && c >= cbest) || !glsl_nid(&g, r[q].e, G.y[x].name,
				    G.y[x].ty) || u_equiv(&fn, bn, r[q].e, NULL, 0, &pp) !=
				    1) {
					free(g.p);
					continue;
				}
				best = q;
				cbest = c;
				pts = pp;
				free(form.p);
				form = g;
			}
		} else
			ok = 0;
		/* the form when it is cheaper than the table, else the table; the
		 * other one is named after it */
		usef = best >= 0 && cbest < ctab;
		c0 = scost(s);
		c1 = (usef ? cbest : ctab) + model_active()->c[C_IADD];	/* + store */
		if (ok && c1 < c0) {
			char q0[80], ys[48];
			struct sbuf v0 = {0};
			p = prop_new(f, P_TAB, t->line);
			num_text(&v0, y0, G.y[y].ty);
			snprintf(ys, sizeof ys, "%s", G.y[y].ty == TI ? "int(" : "");
			if (usef) {
				/* find's words for the form, without the map's text */
				char key[60];
				const char *as;
				snprintf(key, sizeof key, " of %s as ", G.y[x].name);
				as = strstr(r[best].what, key);
				glsl_strip(&form);
				snprintf(p->what, sizeof p->what, "the loop leaves %s as a "
				    "function of %s alone (%s = %s as it starts): %s%s",
				    G.y[y].name, G.y[x].name, G.y[y].name, v0.p, as ? "find "
				    "writes it " : "", as ? as + strlen(key) - 3 : r[best].what);
				sb_printf(&txt, "%s = %s%s%s;", G.y[y].name, ys, form.p,
				    *ys ? ")" : "");
			} else {
				/* the table read itself is the cheapest */
				snprintf(p->what, sizeof p->what, "the loop leaves %s as a "
				    "function of %s alone (%s = %s as it starts): a table of "
				    "its %d values", G.y[y].name, G.y[x].name, G.y[y].name,
				    v0.p, K);
				sb_printf(&txt, "const %s %s_of_%s[%d] = %s[](", ty_name(G.y[y].ty),
				    G.y[y].name, G.y[x].name, K, ty_name(G.y[y].ty));
				for (n = 0; n < K; n++) {
					if (n)
						sb_puts(&txt, ", ");
					if (G.y[y].ty == TU)
						glsl_u(&txt, tab[n]);
					else
						sb_printf(&txt, "%lld", (long long)tab[n]);
				}
				sb_printf(&txt, ");\n    %s = %s_of_%s[%s", G.y[y].name,
				    G.y[y].name, G.y[x].name, G.y[x].name);
				if (E[x].lo) {
					sb_puts(&txt, " - ");
					if (G.y[x].ty == TU)
						glsl_u(&txt, E[x].lo);
					else
						sb_printf(&txt, "%lld", (long long)E[x].lo);
				}
				sb_puts(&txt, "];");
				pts = K;
			}
			src_quote(q0, sizeof q0, t->line);
			sb_printf(&p->text, "%s\n  ->  %s", q0, txt.p);
			if (usef) {
				sb_printf(&p->text, "\n  or  a table of its %d values read at "
				    "%s", K, G.y[x].name);
				if (E[x].lo) {
					sb_puts(&p->text, " - ");
					if (G.y[x].ty == TU)
						glsl_u(&p->text, E[x].lo);
					else
						sb_printf(&p->text, "%lld", (long long)E[x].lo);
				}
				sb_printf(&p->text, " (cost %g)", ctab +
				    model_active()->c[C_IADD]);
			}
			else if (best >= 0) {
				glsl_strip(&form);
				sb_printf(&p->text, "\n  or  %s = %s%s%s; (cost %g)",
				    G.y[y].name, ys, form.p, *ys ? ")" : "", cbest +
				    model_active()->c[C_IADD]);
			}
			snprintf(p->proof, sizeof p->proof, "the loop run at each of the "
			    "%d values of %s, %s", K, G.y[x].name, usef ? "and the "
			    "form equal to that table at every one (ucas equiv, "
			    "exhaustive)" : "the table its values");
			p->points = pts;
			p->c0 = c0;
			p->c1 = c1;
			{
				struct sbuf d = {0};
				sb_printf(&d, "%s in ", G.y[x].name);
				av_text(&d, E[x], G.y[x].ty);
				if (rd[y])
					sb_printf(&d, "; %s = %s", G.y[y].name, v0.p);
				for (k = 0; k < A.ns; k++)
					if (rd[k] && k != x && k != y && !decl[k] && k != var &&
					    !G.y[k].tab && G.y[k].kind != Y_CONST && !E[k].bot &&
					    E[k].lo == E[k].hi) {
						sb_printf(&d, "; %s = ", G.y[k].name);
						num_text(&d, E[k].lo, G.y[k].ty);
					}
				snprintf(p->domain, sizeof p->domain, "%s", d.p);
				snprintf(p->relies, sizeof p->relies, "%s where the loop starts "
				    "(the analysis's domain from %s)", d.p, is_root(f) ?
				    "the start of main" : "the calls");
				free(d.p);
			}
			free(v0.p);
		}
		if (r)
			irw_free(r, nr);
		free(body.p);
		free(txt.p);
		free(form.p);
		free(tab);
		free(lx);
		free(ls);
		free(rd);
		free(decl);
	}
	free(xs);
	free(ss);
}

/* operand replacements for a dead operation: up to 3 */
/*
 * k % d == r (or !=), alone or && with k != c: the values of k where it
 * holds, when k's recorded range has at most 4 of them, as equalities
 * (k % 27 == 0 && k != 0 for k in 0..107 is k == 27 || k == 54 || k ==
 * 81); -1 when it is not that or has more.
 */
static int same_text(int a, int b);

static int
divis_parts(int x, int *k, int64_t *d, int64_t *r, int *eq)
{
	const struct gx *g = &G.x[x];
	int m, c;

	if (g->k != X_BIN || (g->op != OP_EQ && g->op != OP_NE))
		return 0;
	m = g->a;
	c = g->b;
	if (G.x[XR(m)].k == X_NUM) {
		m = g->b;
		c = g->a;
	}
	if (G.x[XR(c)].k != X_NUM || G.x[m].k != X_BIN || G.x[m].op != OP_MOD ||
	    G.x[XR(G.x[m].b)].k != X_NUM || (G.x[m].ty != TU && G.x[m].ty != TI))
		return 0;
	*k = G.x[m].a;
	*d = G.x[XR(G.x[m].b)].v;
	*r = G.x[XR(c)].v;
	*eq = g->op == OP_EQ;
	return *d > 0;
}

static int
few_multiples(int x)
{
	const struct gx *g = &G.x[x];
	const struct rec *R;
	int k = -1, eq, j, b, ty, ln = g->line, kc = -1, n = 0, o = -1;
	int64_t d, r, lo = INT64_MAX, hi = INT64_MIN, v, ex = 0, hit[4];
	int has_ex = 0;

	if (g->ty != TB || recn[x] < 0)
		return -1;
	if (g->k == X_BIN && g->op == OP_LAND) {
		/* the test and a k != c beside it */
		int t = g->a, u = g->b, q;
		for (q = 0; q < 2; q++) {
			const struct gx *h = &G.x[u];
			if (divis_parts(t, &k, &d, &r, &eq) && eq && h->k == X_BIN &&
			    h->op == OP_NE && (same_text(h->a, k) || same_text(h->b,
			    k))) {
				kc = same_text(h->a, k) ? h->b : h->a;
				if (G.x[XR(kc)].k == X_NUM) {
					ex = G.x[XR(kc)].v;
					has_ex = 1;
					break;
				}
			}
			t = g->b;
			u = g->a;
		}
		if (!has_ex)
			return -1;
	} else if (!divis_parts(x, &k, &d, &r, &eq) || !eq)
		return -1;
	ty = G.x[k].ty;
	R = &RC[recn[x]];
	for (j = 0; j < R->nl && !same_text(R->lf[j], k); j++)
		;
	if (j == R->nl || R->nbox == 0)
		return -1;
	for (b = 0; b < R->nbox; b++) {
		const struct av *a = &R->box[b * (R->nl + R->nex) + j];
		if (a->bot)
			return -1;
		if (a->lo < lo)
			lo = a->lo;
		if (a->hi > hi)
			hi = a->hi;
	}
	if (hi - lo > 4096)
		return -1;
	for (v = lo; v <= hi; v++) {
		int64_t m = ty == TU ? (int64_t)((uint64_t)v % (uint64_t)d) : v % d;
		if (m != r || (has_ex && v == ex))
			continue;
		if (n == 4)
			return -1;
		hit[n++] = v;
	}
	for (j = n - 1; j >= 0; j--) {
		int e = new_bin(OP_EQ, TB, k, new_num(ty, hit[j], ln), ln);
		o = o < 0 ? e : new_bin(OP_LOR, TB, e, o, ln);
	}
	return o < 0 ? new_num(TB, 0, ln) : o;
}

static int
x_options(int x, int *opt, const char **why, int max)
{
	const struct gx *g = &G.x[x];
	int n = 0, i;
	int64_t k;
	if (g->ty != TU && g->ty != TI && g->ty != TB)
		return 0;
	if (g->k == X_CALL && (g->sym == -B_MIN || g->sym == -B_MAX) &&
	    g->b == 2) {
		for (i = 0; i < 2 && n < max; i++)
			if (G.x[G.xarg[g->a + i]].ty == g->ty) {
				why[n] = g->sym == -B_MIN ? "the min never bites" :
				    "the max never bites";
				opt[n++] = G.xarg[g->a + i];
			}
	}
	if (g->k == X_CALL && g->sym == -B_CLAMP && g->b == 3 && n + 3 <= max) {
		int a0 = G.xarg[g->a], a1 = G.xarg[g->a + 1], a2 = G.xarg[g->a + 2];
		int ar[2];
		if (G.x[a0].ty == g->ty) {
			why[n] = "the clamp never bites";
			opt[n++] = a0;
		}
		ar[0] = a0;
		ar[1] = a1;
		why[n] = "the clamp's upper bound never bites";
		opt[n++] = new_call(B_MAX, g->ty, ar, 2, g->line);
		ar[1] = a2;
		why[n] = "the clamp's lower bound never bites";
		opt[n++] = new_call(B_MIN, g->ty, ar, 2, g->line);
	}
	if (g->k == X_BIN && (g->op == OP_AND || g->op == OP_MOD) && n < max) {
		if (g->op == OP_AND && G.x[XR(g->a)].k == X_NUM &&
		    G.x[g->b].ty == g->ty) {
			why[n] = "the mask never bites";
			opt[n++] = g->b;
		} else if (G.x[XR(g->b)].k == X_NUM && G.x[g->a].ty == g->ty) {
			why[n] = g->op == OP_AND ? "the mask never bites" :
			    "the remainder never bites";
			opt[n++] = g->a;
		}
		(void)k;
	}
	if (g->k == X_SEL && n + 2 <= max) {
		if (G.x[g->b].ty == g->ty) {
			why[n] = "the select always takes its first value";
			opt[n++] = g->b;
		}
		if (G.x[g->c].ty == g->ty) {
			why[n] = "the select always takes its second value";
			opt[n++] = g->c;
		}
	}
	/*
	 * a signed division or remainder whose operands are never negative
	 * here: the unsigned one (trunc(a/b) = floor(a/b) for a >= 0, b > 0,
	 * and the bits read as uint are the same numbers); kept when the
	 * model prices it lower
	 */
	if (g->k == X_BIN && (g->op == OP_DIV || g->op == OP_MOD) &&
	    g->ty == TI && n < max) {
		int aa = g->a, bb = g->b, op = g->op, ln = g->line, ua, ub, d;
		ua = new_call(B_UINT, TU, &aa, 1, ln);
		ub = G.x[XR(bb)].k == X_NUM ? new_num(TU, G.x[XR(bb)].v, ln) :
		    new_call(B_UINT, TU, &bb, 1, ln);
		d = new_bin(op, TU, ua, ub, ln);
		g = &G.x[x];
		why[n] = g->op == OP_DIV ? "the signed division of operands "
		    "never negative here, unsigned (trunc = floor for a >= 0, b > 0)" :
		    "the signed remainder of operands never negative here, "
		    "unsigned (a - b trunc(a/b) = a mod b for a >= 0, b > 0)";
		opt[n++] = new_call(B_INT, TI, &d, 1, ln);
		g = &G.x[x];
	}
	if (g->k == X_BIN && g->op >= OP_LT && g->op <= OP_NE && n + 2 <= max) {
		why[n] = "the compare is always true";
		opt[n++] = new_num(TB, 1, g->line);
		why[n] = "the compare is always false";
		opt[n++] = new_num(TB, 0, g->line);
		g = &G.x[x];
	}
	if (n < max) {
		int e = few_multiples(x);
		if (e >= 0) {
			why[n] = "a divisibility test of a value with few multiples "
			    "where it runs, as equalities";
			opt[n++] = e;
		}
	}
	return n;
}

static int
x_candidate(int x)
{
	const struct gx *g = &G.x[x];
	if (g->ty != TU && g->ty != TI && g->ty != TB)
		return 0;
	if (g->k == X_CALL && (g->sym == -B_MIN || g->sym == -B_MAX ||
	    g->sym == -B_CLAMP))
		return 1;
	if (g->k == X_BIN && g->op == OP_AND && (G.x[g->a].k == X_NUM ||
	    G.x[g->b].k == X_NUM))
		return 1;
	if (g->k == X_BIN && g->op == OP_MOD && G.x[g->b].k == X_NUM)
		return 1;
	if (g->k == X_BIN && (g->op == OP_DIV || g->op == OP_MOD) && g->ty == TI)
		return 1;
	if (g->k == X_SEL)
		return 1;
	if (g->k == X_BIN && g->op >= OP_LT && g->op <= OP_NE)
		return 1;
	if (g->k == X_BIN && g->op == OP_LAND)
		return 1;	/* a divisibility test and a k != c */
	return 0;
}

/* ((2^S - w) a + w b + C) >> S, as int(...) or not: its parts */
static int
flat_add(int x, int *t, int n, int max)
{
	const struct gx *g = &G.x[x];
	if (g->k == X_BIN && g->op == OP_ADD) {
		n = flat_add(g->a, t, n, max);
		return flat_add(g->b, t, n, max);
	}
	if (n < max)
		t[n++] = x;
	return n;
}

static int
same_text(int a, int b)
{
	char x[200], y[200];
	xtext(x, sizeof x, a);
	xtext(y, sizeof y, b);
	return !strcmp(x, y);
}

static int
match_lerp(int x, int *w, int *a, int *b, int64_t *cc, int64_t *s)
{
	const struct gx *g = &G.x[x];
	int t[8], n, i, j, m1 = -1, m2 = -1;
	int64_t c = 0, sh;
	if (g->k == X_CALL && g->sym == -B_INT && g->b == 1)
		g = &G.x[G.xarg[g->a]];
	if (g->k != X_BIN || g->op != OP_SHR || G.x[g->b].k != X_NUM ||
	    g->ty != TU)
		return 0;
	sh = G.x[g->b].v;
	if (sh < 1 || sh > 16)
		return 0;
	n = flat_add(g->a, t, 0, 8);
	if (n != 3)
		return 0;
	for (i = 0; i < n; i++)
		if (G.x[t[i]].k == X_NUM)
			c = G.x[t[i]].v;
	for (i = 0; i < n; i++) {
		const struct gx *p = &G.x[t[i]];
		if (p->k != X_BIN || p->op != OP_MUL)
			continue;
		for (j = 0; j < 2; j++) {
			int l = j ? p->b : p->a, r = j ? p->a : p->b;
			const struct gx *q = &G.x[l];
			if (q->k == X_BIN && q->op == OP_SUB && G.x[q->a].k == X_NUM &&
			    G.x[q->a].v == ((int64_t)1 << sh)) {
				m1 = t[i];
				*w = q->b;
				*a = r;
			}
		}
	}
	if (m1 < 0)
		return 0;
	for (i = 0; i < n; i++) {
		const struct gx *p = &G.x[t[i]];
		if (t[i] == m1 || p->k != X_BIN || p->op != OP_MUL)
			continue;
		if (same_text(p->a, *w)) {
			m2 = t[i];
			*b = p->b;
		} else if (same_text(p->b, *w)) {
			m2 = t[i];
			*b = p->a;
		}
	}
	if (m2 < 0 || G.x[*a].ty != TU || G.x[*b].ty != TU || G.x[*w].ty != TU)
		return 0;
	*cc = c;
	*s = sh;
	return 1;
}

static int
lerp_repl(int x, int w, int a, int b, int64_t c, int64_t s)
{
	int line = G.x[x].line, ia, iw, ib, d, m, p, q, r;
	ia = new_call(B_INT, TI, &a, 1, line);
	iw = new_call(B_INT, TI, &w, 1, line);
	ib = new_call(B_INT, TI, &b, 1, line);
	d = new_bin(OP_SUB, TI, ib, ia, line);
	m = new_bin(OP_MUL, TI, iw, d, line);
	p = new_bin(OP_ADD, TI, m, new_num(TI, c, line), line);
	q = new_bin(OP_SHR, TI, p, new_num(TI, s, line), line);
	r = new_bin(OP_ADD, TI, ia, q, line);
	if (G.x[x].k == X_CALL)
		return r;
	return new_call(B_UINT, TU, &r, 1, line);
}

/* ---- records: candidates and calls, marked before the analysis ---- */

static int
add_leaf(struct rec *R, int x)
{
	int i;
	for (i = 0; i < R->nl; i++)
		if (same_text(R->lf[i], x))
			return 1;
	if (R->nl >= MAXP)
		return 0;
	if (G.x[x].ty != TU && G.x[x].ty != TI && G.x[x].ty != TB)
		return 0;
	R->lf[R->nl++] = x;
	return 1;
}

static int leaf_tables;	/* marking a candidate of search_tex */

static int
leaves_of(struct rec *R, int x)
{
	const struct gx *g = &G.x[x];
	int i;
	switch (g->k) {
	case X_NUM:
		return 1;
	case X_VAR:
		if (G.y[g->sym].kind == Y_CONST && G.y[g->sym].init >= 0 &&
		    !G.y[g->sym].arr)
			return 1;
		if (G.y[g->sym].arr)
			return 0;
		return add_leaf(R, x);
	case X_IDX:
		/* for search_tex a constant table's read is not a leaf: its
		 * index's leaves are */
		if (leaf_tables && G.x[g->a].k == X_VAR &&
		    G.y[G.x[g->a].sym].tab)
			return leaves_of(R, g->b);
		return add_leaf(R, x);
	case X_MEM:
		return add_leaf(R, x);
	case X_CALL:
		if (g->sym >= 0) {
			if (A.impure[g->sym] || A.nwrl[g->sym])
				return 0;
			return add_leaf(R, x);
		}
		if (g->sym == -B_UNK)
			return 0;
		for (i = 0; i < g->b; i++)
			if (!leaves_of(R, G.xarg[g->a + i]))
				return 0;
		return 1;
	case X_UN:
		return leaves_of(R, g->a);
	case X_BIN:
		return leaves_of(R, g->a) && leaves_of(R, g->b);
	case X_SEL:
		return leaves_of(R, g->a) && leaves_of(R, g->b) &&
		    leaves_of(R, g->c);
	}
	return 0;
}

static int
mark(int x, int calls)
{
	struct rec R;
	memset(&R, 0, sizeof R);
	R.x = x;
	if (calls) {
		const struct gx *g = &G.x[x];
		int i, f = g->sym;
		for (i = 0; i < g->b && i < MAXP; i++)
			R.lf[R.nl++] = G.xarg[g->a + i];
		for (i = 0; i < A.nrk[f] && R.nex < MAXP; i++)
			R.ex[R.nex++] = A.rk[f][i];
	} else if (!leaves_of(&R, x) || R.nl == 0)
		return -1;
	if (nrc == caprc) {
		caprc = caprc ? caprc * 2 : 256;
		RC = xrealloc(RC, (size_t)caprc * sizeof *RC);
	}
	RC[nrc] = R;
	recn[x] = nrc;
	return nrc++;
}

/* an integer sum with a product in it, or a product: for search_dot */
static int
has_prod(int x, int depth)
{
	const struct gx *g = &G.x[XR(x)];

	if (depth > 12 || g->k != X_BIN)
		return 0;
	if (g->op == OP_MUL)
		return 1;
	if (g->op != OP_ADD && g->op != OP_SUB && g->op != OP_SHL)
		return 0;
	return has_prod(g->a, depth + 1) || has_prod(g->b, depth + 1);
}

static int
dot_candidate(int x)
{
	const struct gx *g = &G.x[x];

	return (g->ty == TU || g->ty == TI) && g->k == X_BIN &&
	    (g->op == OP_ADD || g->op == OP_SUB || g->op == OP_MUL) &&
	    has_prod(x, 0);
}

/* a read of a constant table: for search_tex */
static int
table_read(int x)
{
	const struct gx *g = &G.x[XR(x)];

	return g->k == X_IDX && G.x[g->a].k == X_VAR && G.y[G.x[g->a].sym].tab &&
	    G.y[G.x[g->a].sym].arr <= 256;
}

static int
table_reads(int x, int depth)
{
	const struct gx *g = &G.x[XR(x)];

	if (depth > 12)
		return 0;
	if (table_read(x))
		return 1;
	if (g->k != X_BIN || (g->op != OP_ADD && g->op != OP_SUB &&
	    g->op != OP_MUL && g->op != OP_SHL))
		return 0;
	return table_reads(g->a, depth + 1) + table_reads(g->b, depth + 1);
}

/*
 * An integer table read, a select of one against a constant, or a sum
 * with two or more table reads in it: what the texture unit could
 * compute (texalu.c).  Only under a model that prices such reads: the
 * values of its leaves are recorded for nothing otherwise.
 */
static int
tex_candidate(int x)
{
	const struct gx *g = &G.x[x];
	const struct model *m = model_active();

	if (g->ty != TU && g->ty != TI)
		return 0;
	if (!m->par[MP_SUBTEXEL] || !(cclass_available(m, C_TEX_NEAREST) ||
	    (cclass_available(m, C_TEX_SAMPLE) && m->par[MP_TEX_WITHIN])))
		return 0;
	if (g->k == X_IDX)
		return table_read(x);
	if (g->k == X_SEL)
		return (table_read(g->b) && G.x[XR(g->c)].k == X_NUM) ||
		    (table_read(g->c) && G.x[XR(g->b)].k == X_NUM);
	return g->k == X_BIN && (g->op == OP_ADD || g->op == OP_SUB) &&
	    table_reads(x, 0) >= 2;
}

static void
mark_all(void)
{
	int f, i, *xs = xmalloc((size_t)(G.nx + 1) * sizeof *xs), nx;
	for (f = 0; f < G.nf; f++) {
		nx = 0;
		reach_s(G.f[f].body, xs, &nx, NULL, NULL, G.nx);
		for (i = 0; i < nx; i++) {
			int x = xs[i], w, a, b;
			int64_t c, s;
			if (recn[x] >= 0)
				continue;
			if (G.x[x].k == X_CALL && G.x[x].sym >= 0)
				mark(x, 1);
			else if (x_candidate(x) || match_lerp(x, &w, &a, &b, &c, &s) ||
			    dot_candidate(x))
				mark(x, 0);
			else if (tex_candidate(x)) {
				leaf_tables = 1;
				mark(x, 0);
				leaf_tables = 0;
			}
		}
	}
	free(xs);
}

/* ---- the searches ---- */

static int *xfn;	/* expression -> its function, -1 */
static int xfn_n;

static void
stmt_brief(char *buf, size_t n, int s)
{
	struct sbuf b = {0};
	char *p, *q;
	int sp = 0;
	ps_text(&b, s, 0);
	/* one line: runs of spaces and newlines to one space */
	for (p = q = b.p ? b.p : (char *)""; *p; p++) {
		if (*p == '\n' || *p == ' ') {
			if (!sp && q != b.p)
				*q++ = ' ';
			sp = 1;
			continue;
		}
		sp = 0;
		*q++ = *p;
	}
	*q = 0;
	while (q > b.p && q[-1] == ' ')
		*--q = 0;
	if (b.p && strlen(b.p) > 70)
		snprintf(buf, n, "%.67s...", b.p);
	else
		snprintf(buf, n, "%s", b.p ? b.p : "");
	free(b.p);
}

/* where f is called from: "Decoded (lines 207, 208), Quant (line 222)" */
static void
callers_text(struct sbuf *b, int f)
{
	int i, j, g, first = 1;
	for (g = 0; g < G.nf; g++) {
		int lines[32], nl = 0;
		for (i = 0; i < nrc; i++) {
			const struct gx *x = &G.x[RC[i].x];
			if (x->k != X_CALL || x->sym != f || RC[i].x >= xfn_n ||
			    xfn[RC[i].x] != g)
				continue;
			for (j = 0; j < nl && lines[j] != x->line; j++)
				;
			if (j == nl && nl < 32)
				lines[nl++] = x->line;
		}
		if (!nl)
			continue;
		sb_printf(b, "%s%s (line%s ", first ? "" : ", ", G.f[g].name,
		    nl > 1 ? "s" : "");
		for (j = 0; j < nl; j++)
			sb_printf(b, "%s%d", j ? ", " : "", lines[j]);
		sb_puts(b, ")");
		first = 0;
	}
	if (first)
		sb_puts(b, "no call");
}

static int
whole_ok(int f)
{
	int i;
	if (G.f[f].body < 0 || G.f[f].unsup || A.impure[f] || A.nwrl[f] ||
	    is_root(f) || FS[f].n == 0)
		return 0;
	if (G.f[f].ty != TU && G.f[f].ty != TI && G.f[f].ty != TB)
		return 0;
	for (i = 0; i < A.ns; i++)
		if (A.rd[f][i] && G.y[i].arr)
			return 0;	/* reads an array other than a table */
	return 1;
}

static void search_fit(int f, struct dom *D);

static void
search_whole(int f)
{
	struct dom D;
	int64_t *ref;
	uint8_t *rub;
	long np, pts = 0;
	char names[MAXP + 8][40], wit[200];
	int tys[MAXP + 8], kept = 0, i, again;
	char *orig, *pat;
	struct sbuf what = {0};
	double c0;
	uint8_t *tried_s, *tried_x;
	int tried_n;
	int *xs, *ss, nx, nss, cap;

	if (!whole_ok(f) || !fn_dom(f, &D))
		return;

	if (D.points > (double)(1 << 20) || D.nv > MAXP) {
		free(D.rows);
		return;
	}
	orig = fn_ucas(f, &D, names, tys);
	if (orig == NULL) {
		free(D.rows);
		return;
	}
	ref = fn_values(f, &D, &rub, &np);
	c0 = scost(G.f[f].body);
	cap = G.nx + G.ns + 64;
	tried_s = xcalloc((size_t)G.ns + 64, 1);
	tried_n = G.nx + 4096;
	tried_x = xcalloc((size_t)tried_n, 1);
	xs = xmalloc((size_t)cap * sizeof *xs);
	ss = xmalloc((size_t)cap * sizeof *ss);
	/* statements first: branches, then the ends of constant-bound loops */
	do {
		again = 0;
		nx = nss = 0;
		reach_s(G.f[f].body, xs, &nx, ss, &nss, cap);
		for (i = 0; i < nss && !again; i++) {
			int s = ss[i], v;
			const struct gs *t = &G.s[s];
			int64_t lo, last, k;
			char br[100];
			if (tried_s[s])
				continue;
			tried_s[s] = 1;
			if (t->k == S_IF) {
				stmt_brief(br, sizeof br, s);
				if (t->c < 0 || SR(t->c) == -2) {
					srep[s] = -2;
					if (fn_same(f, &D, ref, rub, NULL, 0)) {
						sb_printf(&what, "%sthe branch `%s` (line %d) never "
						    "changes the result", kept ? "; " : "", br, t->line);
						kept++;
						again = 1;
					} else
						srep[s] = -1;
				} else {
					int side;
					for (side = 0; side < 2 && !again; side++) {
						srep[s] = side ? t->c : t->b;
						if (fn_same(f, &D, ref, rub, NULL, 0)) {
							sb_printf(&what, "%sthe %s side of `%s` (line %d) is "
							    "never needed", kept ? "; " : "", side ? "then" :
							    "else", br, t->line);
							kept++;
							again = 1;
						} else
							srep[s] = -1;
					}
				}
			}
			if (t->k == S_FOR && canon_loop(s, &v, &lo, &last) && last > lo) {
				/* trim the first iterations, then the last */
				int a = SR(t->a), init = G.s[a].a, cmp = XR(t->b);
				int bnode = G.x[cmp].b, ty = G.y[v].ty, op = G.x[cmp].op;
				int64_t nlo = lo, nlast = last;
				for (k = lo + 1; k <= last; k++) {
					xrep[init] = new_num(ty, k, t->line);
					if (!fn_same(f, &D, ref, rub, NULL, 0))
						break;
					nlo = k;
				}
				xrep[init] = nlo != lo ? new_num(ty, nlo, t->line) : -1;
				for (k = last - 1; k >= nlo; k--) {
					xrep[bnode] = new_num(ty, op == OP_LT ? k + 1 : k, t->line);
					if (!fn_same(f, &D, ref, rub, NULL, 0))
						break;
					nlast = k;
				}
				xrep[bnode] = nlast != last ? new_num(ty, op == OP_LT ?
				    nlast + 1 : nlast, t->line) : -1;
				if (nlo != lo || nlast != last) {
					sb_printf(&what, "%sthe loop at line %d: %s = ", kept ? "; " :
					    "", t->line, G.y[v].name);
					if (nlo != lo)
						sb_printf(&what, "%lld%s", (long long)lo, nlo - lo > 1 ?
						    ".." : "");
					if (nlo - lo > 1)
						sb_printf(&what, "%lld", (long long)(nlo - 1));
					if (nlo != lo && nlast != last)
						sb_puts(&what, " and ");
					if (nlast != last) {
						sb_printf(&what, "%lld", (long long)(nlast + 1));
						if (last - nlast > 1)
							sb_printf(&what, "..%lld", (long long)last);
					}
					sb_printf(&what, " never change the result: %lld..%lld is "
					    "enough", (long long)nlo, (long long)nlast);
					kept++;
				}
			}
		}
	} while (again);
	/* then the operations */
	do {
		again = 0;
		nx = nss = 0;
		reach_s(G.f[f].body, xs, &nx, NULL, NULL, cap);
		for (i = 0; i < nx && !again; i++) {
			int x = xs[i], opt[4], n, j;
			const char *why[4];
			char t0[160];
			double c0;
			if (x >= tried_n || tried_x[x] || !x_candidate(x))
				continue;
			tried_x[x] = 1;
			xtext(t0, sizeof t0, x);
			n = x_options(x, opt, why, 4);
			c0 = xcost(x);
			for (j = 0; j < n && !again; j++) {
				/* only what the model prices lower */
				if (xcost(opt[j]) >= c0)
					continue;
				xrep[x] = opt[j];
				if (fn_same(f, &D, ref, rub, NULL, 0)) {
					char t1[160];
					xtext(t1, sizeof t1, opt[j]);
					sb_printf(&what, "%sline %d: `%s` can be `%s` (%s)", kept ?
					    "; " : "", G.x[x].line, t0, t1, strncmp(why[j],
					    "the signed", 10) ? "the result is the same" : why[j]);
					kept++;
					again = 1;
				} else
					xrep[x] = -1;
			}
		}
	} while (again);
	if (kept) {
		/* a || false, a && true, ... folded */
		int q;
		nx = 0;
		reach_s(G.f[f].body, xs, &nx, NULL, NULL, cap);
		for (q = nx - 1; q >= 0; q--) {
			const struct gx *g = &G.x[xs[q]];
			int a, b;
			if (g->k != X_BIN || (g->op != OP_LOR && g->op != OP_LAND))
				continue;
			a = XR(g->a);
			b = XR(g->b);
			if (G.x[b].k == X_NUM && G.x[b].v == (g->op == OP_LAND))
				xrep[xs[q]] = a;
			else if (G.x[a].k == X_NUM && G.x[a].v == (g->op == OP_LAND))
				xrep[xs[q]] = b;
		}
	}
	if (kept) {
		int tys2[MAXP + 8];
		char names2[MAXP + 8][40];
		pat = fn_ucas(f, &D, names2, tys2);
		if (pat) {
			int rc = prove_rows(names, tys, D.nv, D.rows, D.nr, orig, pat,
			    &pts, wit, sizeof wit);
			if (rc == 1) {
				struct prop *p = prop_new(f, P_FUNC, G.f[f].line);
				struct sbuf b = {0};
				snprintf(p->what, sizeof p->what, "%s", what.p);
				fn_text(&p->text, f);
				snprintf(p->proof, sizeof p->proof, "exhaustive: ucas equiv "
				    "of the two as ucas functions on every point of the "
				    "inferred domain (%ld points)", pts);
				p->points = pts;
				p->c0 = c0;
				p->c1 = scost(G.f[f].body);
				vars_text(&b, D.vars, D.nv, D.rows, D.nr);
				snprintf(p->domain, sizeof p->domain, "%s", b.p ? b.p : "");
				sb_clear(&b);
				callers_text(&b, f);
				snprintf(p->relies, sizeof p->relies, "%s at every call: %s",
				    p->domain, b.p);
				free(b.p);
			} else if (rc == 0)
				printf("  note: %s: the interpreter and ucas equiv disagree "
				    "(%s); nothing proposed\n", G.f[f].name, wit);
			free(pat);
		}
	}
	/* back to the source */
	for (i = 0; i < capxrep; i++)
		xrep[i] = -1;
	for (i = 0; i < capsrep; i++)
		srep[i] = -1;
	search_fit(f, &D);
	free(what.p);
	free(orig);
	free(ref);
	free(rub);
	free(tried_s);
	free(tried_x);
	free(xs);
	free(ss);
	free(D.rows);
}

/* ---- a dead operation inside any function, on its recorded leaves ---- */


/* check_rec's rows: those met at this call only (the call expression),
 * or all of them (-1) */
static int rec_site = -1;

static int
bind_leaves(const struct rec *R, int x, int r, int *map)
{
	int nodes[512], n = 0, i, j;
	reach_x(x, nodes, &n, 256);
	reach_x(r, nodes, &n, 512);
	for (i = 0; i < n; i++)
		for (j = 0; j < R->nl; j++)
			if (same_text(nodes[i], R->lf[j])) {
				map[nodes[i]] = j;
				break;
			}
	return n;
}

/* x and its replacement r agree on every recorded leaf value: by the
 * interpreter first, then by ucas equiv; 1 proved */
static int
check_rec(int x, int r, long *pts, char *wit, size_t nw)
{
	const struct rec *R = &RC[recn[x]];
	int w = R->nl + R->nex, i, b, *map, rc, nl = R->nl, nb = 0;
	struct av *rows;
	double total = 0;
	char names[MAXP + 8][40];
	int tys[MAXP + 8];
	struct ut U;
	struct uframe F0 = {0, -1, -1, 0};
	struct sbuf ta = {0}, tb = {0};
	if (R->nbox == 0)
		return -1;
	rows = xmalloc((size_t)R->nbox * (size_t)nl * sizeof *rows);
	for (b = 0; b < R->nbox; b++) {
		double p = 1;
		if (rec_site >= 0 && !rsite_has(R, b, rec_site))
			continue;
		for (i = 0; i < nl; i++) {
			rows[nb * nl + i] = R->box[b * w + i];
			if (rows[nb * nl + i].bot) {
				free(rows);
				return -1;
			}
			p *= (double)(rows[nb * nl + i].hi - rows[nb * nl + i].lo + 1);
		}
		total += p;
		nb++;
	}
	if (nb == 0) {
		free(rows);
		return -1;
	}
	if (total > (double)(1 << 24)) {
		free(rows);
		return -1;
	}
	map = xmalloc(((size_t)G.nx + 1) * sizeof *map);
	for (i = 0; i <= G.nx; i++)
		map[i] = -1;
	bind_leaves(R, x, r, map);
	/* the interpreter, every point */
	C.bind = map;
	for (b = 0; b < nb; b++) {
		int64_t cur[MAXP];
		const struct av *row = &rows[b * nl];
		for (i = 0; i < nl; i++)
			cur[i] = row[i].lo;
		for (;;) {
			int64_t vx, vr;
			int ubx;
			for (i = 0; i < nl; i++)
				C.bv[i] = cur[i];
			C.ub = 0;
			vx = conv(ce(x), G.x[XR(x)].ty, G.x[x].ty);
			ubx = C.ub;
			C.ub = 0;
			vr = conv(ce(r), G.x[XR(r)].ty, G.x[x].ty);
			if (!ubx && (C.ub || vx != vr)) {
				C.bind = NULL;
				free(map);
				free(rows);
				return 0;
			}
			for (i = 0; i < nl; i++) {
				if (cur[i] < row[i].hi) {
					cur[i]++;
					break;
				}
				cur[i] = row[i].lo;
			}
			if (i == nl)
				break;
		}
	}
	C.bind = NULL;
	/* the proof */
	memset(&U, 0, sizeof U);
	U.ok = 1;
	U.leaf = map;
	U.inst = 1;
	u_x(&U, &ta, x, &F0);
	u_conv(&U, &tb, r, G.x[x].ty, &F0);
	for (i = 0; i < nl; i++) {
		snprintf(names[i], 40, "l%d", i);
		tys[i] = G.x[R->lf[i]].ty;
	}
	rc = U.ok ? prove_rows(names, tys, nl, rows, nb, ta.p, tb.p, pts, wit,
	    nw) : -1;
	free(ta.p);
	free(tb.p);
	free(map);
	free(rows);
	return rc;
}

static void
rec_domain(struct sbuf *b, int x)
{
	const struct rec *R = &RC[recn[x]];
	int nl = R->nl, w = R->nl + R->nex, i, bx, tys[MAXP];
	struct av *rows = xmalloc(((size_t)R->nbox + 1) * (size_t)nl * sizeof *rows);
	char nm[MAXP][80];
	const char *names[MAXP];
	for (bx = 0; bx < R->nbox; bx++)
		for (i = 0; i < nl; i++)
			rows[bx * nl + i] = R->box[bx * w + i];
	for (i = 0; i < nl; i++) {
		xtext(nm[i], sizeof nm[i], R->lf[i]);
		names[i] = nm[i];
		tys[i] = G.x[R->lf[i]].ty;
	}
	dom_text(b, tys, nl, rows, R->nbox, names);
	free(rows);
}

static void
search_expr(int f)
{
	int *xs, nx = 0, i, cap = G.nx + 64;
	uint8_t *covered = xcalloc((size_t)G.nx + 1, 1);
	xs = xmalloc((size_t)cap * sizeof *xs);
	reach_s(G.f[f].body, xs, &nx, NULL, NULL, cap);
	for (i = 0; i < nx; i++) {
		int x = xs[i], opt[4], n, j, w, a, b;
		const char *why[4];
		int64_t c, s;
		if (x >= xfn_n || recn[x] < 0 || covered[x] || (G.x[x].k == X_CALL &&
		    G.x[x].sym >= 0))
			continue;
		if (RC[recn[x]].nbox == 0)
			continue;
		n = 0;
		if (match_lerp(x, &w, &a, &b, &c, &s)) {
			opt[n] = lerp_repl(x, w, a, b, c, s);
			why[n++] = "an interpolation with one multiply";
		} else
			n = x_options(x, opt, why, 4);
		for (j = 0; j < n; j++) {
			long pts = 0;
			char wit[200];
			double c0, c1;
			{
				/* the leaves cost the same either way: priced 0 */
				int q, *mp = xmalloc(((size_t)G.nx + 1) * sizeof *mp);
				for (q = 0; q <= G.nx; q++)
					mp[q] = -1;
				bind_leaves(&RC[recn[x]], x, opt[j], mp);
				free_leaf = mp;
				c0 = xcost(x);
				c1 = xcost(opt[j]);
				free_leaf = NULL;
				free(mp);
			}
			if (c1 >= c0)
				continue;
			if (check_rec(x, opt[j], &pts, wit, sizeof wit) == 1) {
				struct prop *p = prop_new(f, P_EXPR, G.x[x].line);
				char t0[300], t1[300];
				struct sbuf d = {0};
				{
					/* what is inside it is not proposed again */
					int sub[512], ns2 = 0, q;
					reach_x(x, sub, &ns2, 512);
					for (q = 0; q < ns2; q++)
						if (sub[q] <= G.nx)
							covered[sub[q]] = 1;
				}
				xtext(t0, sizeof t0, x);
				xtext(t1, sizeof t1, opt[j]);
				snprintf(p->what, sizeof p->what, "%s", why[j]);
				sb_printf(&p->text, "%s\n  ->  %s", t0, t1);
				snprintf(p->proof, sizeof p->proof, "exhaustive: ucas equiv "
				    "on every recorded value of its leaves (%ld points)", pts);
				p->points = pts;
				p->c0 = c0;
				p->c1 = c1;
				rec_domain(&d, x);
				snprintf(p->domain, sizeof p->domain, "%s", d.p ? d.p : "");
				snprintf(p->relies, sizeof p->relies, "%s where line %d runs",
				    d.p ? d.p : "", G.x[x].line);
				free(d.p);
				if (match_lerp(x, &w, &a, &b, &c, &s)) {
					/* b - a already somewhere else? */
					int k, sub = G.x[G.x[opt[j]].k == X_CALL ?
					    G.xarg[G.x[opt[j]].a] : opt[j]].b;
					/* sub is (w (b - a) + C) >> S: its b - a */
					sub = G.x[G.x[G.x[sub].a].a].b;
					for (k = 0; k < xfn_n; k++)
						if (xfn[k] >= 0 && xfn[k] != f && G.x[k].k == X_BIN &&
						    G.x[k].op == OP_SUB && same_text(k, sub)) {
							char tk[200];
							xtext(tk, sizeof tk, k);
							snprintf(p->what + strlen(p->what), sizeof p->what -
							    strlen(p->what), "; %s is already computed in "
							    "%s (line %d)", tk, G.f[xfn[k]].name, G.x[k].line);
							break;
						}
				}
				break;
			}
		}
	}
	free(xs);
	free(covered);
}

/* ---- calls that need less of a helper than its whole domain ---- */

/* a statement on one line, an if or loop by its head (patches followed) */
static void
stmt_head(struct sbuf *b, int s)
{
	const struct gs *t;
	s = SR(s);
	if (s < 0)
		return;
	t = &G.s[s];
	switch (t->k) {
	case S_DECL:
		decl_text(b, s);
		sb_putc(b, ';');
		return;
	case S_EXPR:
		px_text(b, t->a, -1);
		sb_putc(b, ';');
		return;
	case S_RET:
		sb_puts(b, "return");
		if (t->a >= 0) {
			sb_putc(b, ' ');
			px_text(b, t->a, 0);
		}
		sb_putc(b, ';');
		return;
	case S_IF:
		sb_puts(b, "if (");
		px_text(b, t->a, 0);
		sb_putc(b, ')');
		return;
	case S_WHILE:
	case S_DO:
		sb_puts(b, "while (");
		px_text(b, t->b, 0);
		sb_putc(b, ')');
		return;
	}
}

/* the value x takes at the first point of its rows met at rec_site */
static int
rec_first(int x, int64_t *v)
{
	const struct rec *R = &RC[recn[x]];
	int w = R->nl + R->nex, b, i, *map;
	for (b = 0; b < R->nbox; b++)
		if (rec_site < 0 || rsite_has(R, b, rec_site))
			break;
	if (b == R->nbox)
		return 0;
	map = xmalloc(((size_t)G.nx + 1) * sizeof *map);
	for (i = 0; i <= G.nx; i++)
		map[i] = -1;
	bind_leaves(R, x, x, map);
	for (i = 0; i < R->nl; i++) {
		if (R->box[b * w + i].bot) {
			free(map);
			return 0;
		}
		C.bv[i] = R->box[b * w + i].lo;
	}
	C.bind = map;
	C.ub = 0;
	*v = conv(ce(x), G.x[XR(x)].ty, G.x[x].ty);
	C.bind = NULL;
	free(map);
	i = !C.ub;
	C.ub = 0;
	return i;
}

/*
 * Is the change of x to opt at the call c one any compiler makes once the
 * helper is inlined there: what decides it (the condition of a select,
 * the operand dropped, else the whole of x) reads only parameters whose
 * argument at c is written as a constant?
 */
static int
trivial_at(int x, int opt, int c)
{
	const struct gx *g = &G.x[x];
	struct rec T;
	int d, i, k, f = G.x[c].sym;
	int idop = g->k == X_BIN && g->op != OP_AND && g->op != OP_DIV &&
	    g->op != OP_MOD && g->op >= OP_ADD && g->op <= OP_XOR;
	if (G.x[XR(opt)].k == X_NUM && g->k != X_SEL)
		d = x;
	else if (g->k == X_SEL)
		d = g->a;
	else if (idop && opt == g->a)
		d = g->b;	/* the other operand is 0, or 1 to a multiply */
	else if (idop && opt == g->b)
		d = g->a;
	else
		d = x;		/* a mask, clamp, min, max: by x's own range */
	memset(&T, 0, sizeof T);
	if (!leaves_of(&T, d))
		return 0;
	for (i = 0; i < T.nl; i++) {
		const struct gx *l = &G.x[T.lf[i]];
		if (l->k != X_VAR || G.y[l->sym].kind != Y_PARAM)
			return 0;
		for (k = 0; k < G.f[f].np && G.f[f].p[k] != l->sym; k++)
			;
		if (k == G.f[f].np || k >= G.x[c].b ||
		    G.x[XR(G.xarg[G.x[c].a + k])].k != X_NUM)
			return 0;
	}
	return 1;
}

static int
is_clampish(int x)
{
	const struct gx *g = &G.x[x];
	return g->k == X_CALL && (g->sym == -B_CLAMP || g->sym == -B_MIN ||
	    g->sym == -B_MAX);
}

struct cvar {
	char *text;		/* the helper with the call's changes */
	struct sbuf what;	/* the changes */
	struct sbuf diff;	/* the statements it changes, before -> after */
	double c1;
	long pts;
	int n, at[64];		/* the calls it serves */
	int nclamp;		/* clamps, mins and maxes it drops */
	int beyond;		/* a change search_expr does not try */
};

/*
 * A loop-free helper's operations checked at each of its calls on the
 * values their leaves take there (the rows met at that call): a clamp,
 * min, max or mask that never bites, a select or compare that is one
 * value, an operand that adds or multiplies nothing, an operation that
 * is one constant.  The helper with what holds at a call is that call's
 * variant; calls with the same variant are named together, and a
 * variant that drops every clamp is said to be one.  Each change is
 * proved on its own leaves at those calls (check_rec), so the variant
 * computes the same values there.
 */
static void
search_calls(int f)
{
	int sites[64], ns = 0, M = 0, i, j, nx = 0, cap = G.nx + 64, nv = 0,
	    nclamp = 0, *xs, *st = NULL, nst = 0;
	char **head0 = NULL;
	struct cvar V[64];
	struct sbuf orig = {0};
	double c0;
	if (G.f[f].body < 0 || !A.inl[f] || A.impure[f] || A.nwrl[f] ||
	    G.f[f].unsup || is_root(f) || (G.f[f].ty != TU && G.f[f].ty != TI &&
	    G.f[f].ty != TB))
		return;
	for (i = 0; i < nrc; i++) {
		int x = RC[i].x;
		if (G.x[x].k != X_CALL || G.x[x].sym != f || RC[i].nbox == 0 ||
		    x >= xfn_n || xfn[x] < 0)
			continue;
		M++;
		if (ns < 64)
			sites[ns++] = x;
	}
	if (M == 0 || ns != M)
		return;
	xs = xmalloc((size_t)cap * sizeof *xs);
	reach_s(G.f[f].body, xs, &nx, NULL, NULL, cap);
	for (i = 0; i < nx; i++)
		if (xs[i] < xfn_n && is_clampish(xs[i]))
			nclamp++;
	fn_text(&orig, f);
	c0 = scost(G.f[f].body);
	{
		int *ss2 = xmalloc((size_t)(G.ns + 64) * sizeof *ss2), nss2 = 0, nx2 = 0;
		int *xs2 = xmalloc((size_t)cap * sizeof *xs2);
		reach_s(G.f[f].body, xs2, &nx2, ss2, &nss2, G.ns + 64);
		nst = nss2;
		st = ss2;
		head0 = xcalloc((size_t)nst + 1, sizeof *head0);
		for (i = 0; i < nst; i++) {
			struct sbuf h = {0};
			stmt_head(&h, st[i]);
			head0[i] = h.p;
		}
		free(xs2);
	}
	for (j = 0; j < ns; j++) {
		int c = sites[j], k, nclamp_c = 0, beyond = 0, real = 0;
		int ncov = G.nx + 64;
		uint8_t *covered = xcalloc((size_t)ncov, 1);
		struct sbuf what = {0}, text = {0};
		long pts = 0;
		rec_site = c;
		for (i = 0; i < nx; i++) {
			int x = xs[i], opt[8], n = 0, q, n0;
			const char *why[8];
			int64_t v;
			if (x >= xfn_n || covered[x] || recn[x] < 0 ||
			    RC[recn[x]].nbox == 0 || G.x[x].k == X_NUM ||
			    G.x[x].k == X_VAR || (G.x[x].ty != TU && G.x[x].ty != TI &&
			    G.x[x].ty != TB) || (G.x[x].k == X_CALL && G.x[x].sym >= 0))
				continue;
			/* what search_expr would say, an operand that is all of it,
			 * then one value: a variant then keeps what it can of the
			 * helper (a clamp that never bites is its operand, even
			 * where that operand is one value) */
			n = x_options(x, opt, why, 8);
			n0 = n;
			if (G.x[x].k == X_BIN && G.x[x].op >= OP_ADD &&
			    G.x[x].op <= OP_XOR && G.x[x].op != OP_DIV &&
			    G.x[x].op != OP_MOD && G.x[x].op != OP_AND) {
				int op = G.x[x].op;
				if (G.x[G.x[x].a].ty == G.x[x].ty) {
					why[n] = op == OP_MUL ? "the other factor is 1 at this "
					    "call" : "the other operand changes nothing at this "
					    "call";
					opt[n++] = G.x[x].a;
				}
				if (op != OP_SUB && op != OP_SHL && op != OP_SHR &&
				    G.x[G.x[x].b].ty == G.x[x].ty) {
					why[n] = op == OP_MUL ? "the other factor is 1 at this "
					    "call" : "the other operand changes nothing at this "
					    "call";
					opt[n++] = G.x[x].b;
				}
			}
			if (n < 8 && G.x[x].ty != TB && rec_first(x, &v)) {
				why[n] = "one value at this call";
				opt[n++] = new_num(G.x[x].ty, v, G.x[x].line);
			}
			for (q = 0; q < n; q++) {
				char wit[200], t0[160], t1[160];
				long pp = 0;
				int sub[512], nsub = 0, keep[512], nkeep = 0, a, b2;
				if (xcost(opt[q]) >= xcost(x))
					continue;
				if (check_rec(x, opt[q], &pp, wit, sizeof wit) != 1)
					continue;
				/* what a constant argument decides is the compiler's
				 * own fold after inlining: not this search's to say,
				 * but for a clamp, which the variant still drops (the
				 * call needs none of its clamps either way) */
				if (trivial_at(x, opt[q], c)) {
					if (!is_clampish(x))
						break;
				} else
					real = 1;
				xtext(t0, sizeof t0, x);
				xtext(t1, sizeof t1, opt[q]);
				sb_printf(&what, "%sline %d: `%s` is `%s` (%s)", what.p ?
				    "; " : "", G.x[x].line, t0, t1, why[q]);
				if (q >= n0)
					beyond = 1;
				if (is_clampish(x))
					nclamp_c++;
				pts += pp;
				/* what it drops is not looked at again; what it keeps is */
				reach_x(x, sub, &nsub, 512);
				reach_x(opt[q], keep, &nkeep, 512);
				xrep[x] = opt[q];
				for (a = 0; a < nsub; a++) {
					for (b2 = 0; b2 < nkeep && keep[b2] != sub[a]; b2++)
						;
					if (b2 == nkeep && sub[a] < ncov)
						covered[sub[a]] = 1;
				}
				break;
			}
		}
		rec_site = -1;
		/* what the changes leave to fold, innermost first: constants,
		 * x + 0, x * 1, x * 0 and the like (exact in the word) */
		for (i = nx - 1; i >= 0; i--) {
			const struct gx *g = &G.x[xs[i]];
			int a, b, z, ub = 0;
			int64_t va, vb;
			if (g->k != X_BIN || xs[i] >= capxrep || xrep[xs[i]] >= 0 ||
			    (g->ty != TU && g->ty != TI) || g->op > OP_XOR)
				continue;
			a = XR(g->a);
			b = XR(g->b);
			if (G.x[a].k == X_NUM && G.x[b].k == X_NUM) {
				int64_t r = cop(g->op, opnd_ty(g), conv(G.x[a].v, G.x[a].ty,
				    opnd_ty(g)), g->op == OP_SHL || g->op == OP_SHR ?
				    G.x[b].v : conv(G.x[b].v, G.x[b].ty, opnd_ty(g)), &ub);
				if (!ub)
					xrep[xs[i]] = new_num(g->ty, norm(g->ty, r), g->line);
				continue;
			}
			va = G.x[a].k == X_NUM ? G.x[a].v : -1;
			vb = G.x[b].k == X_NUM ? G.x[b].v : -1;
			z = -1;
			switch (g->op) {
			case OP_ADD: case OP_OR: case OP_XOR:
				z = vb == 0 ? a : va == 0 ? b : -1;
				break;
			case OP_SUB: case OP_SHL: case OP_SHR:
				z = vb == 0 ? a : -1;
				break;
			case OP_MUL:
				z = vb == 1 ? a : va == 1 ? b : -1;
				if (va == 0 || vb == 0)
					z = new_num(g->ty, 0, g->line);
				break;
			case OP_AND:
				if (va == 0 || vb == 0)
					z = new_num(g->ty, 0, g->line);
				break;
			}
			if (z >= 0 && G.x[z].ty == g->ty)
				xrep[xs[i]] = z;
		}
		/* a select whose condition became one value is its arm */
		for (i = 0; i < nx; i++) {
			const struct gx *g = &G.x[xs[i]];
			int cd;
			if (g->k != X_SEL || xs[i] >= capxrep || xrep[xs[i]] >= 0)
				continue;
			cd = XR(g->a);
			if (G.x[cd].k == X_NUM && G.x[g->b].ty == g->ty &&
			    G.x[g->c].ty == g->ty)
				xrep[xs[i]] = G.x[cd].v ? g->b : g->c;
		}
		fn_text(&text, f);
		if (what.p && real && strcmp(text.p, orig.p)) {
			for (k = 0; k < nv && strcmp(V[k].text, text.p); k++)
				;
			if (k == nv) {
				memset(&V[nv], 0, sizeof V[nv]);
				V[nv].text = text.p;
				text.p = NULL;
				V[nv].what = what;
				what.p = NULL;
				V[nv].c1 = scost(G.f[f].body);
				V[nv].nclamp = nclamp_c;
				for (i = 0; i < nst; i++) {
					struct sbuf h = {0};
					stmt_head(&h, st[i]);
					if (h.p && head0[i] && strcmp(h.p, head0[i]))
						sb_printf(&V[nv].diff, "%s%s\n  ->  %s", V[nv].diff.p ?
						    "\n" : "", head0[i], h.p);
					free(h.p);
				}
				nv++;
			}
			V[k].at[V[k].n++] = c;
			V[k].pts += pts;
			V[k].beyond |= beyond;
		}
		for (i = 0; i < capxrep; i++)
			xrep[i] = -1;
		free(text.p);
		free(what.p);
		free(covered);
	}
	for (j = 0; j < nv; j++) {
		struct cvar *cv = &V[j];
		struct prop *p;
		struct sbuf at = {0}, rel = {0};
		int k;
		/* every call, with nothing search_expr would not say: said */
		if (cv->n == M && !cv->beyond)
			continue;
		{
			/* the lines, each once, with how many calls when more */
			int nl2 = 0, ln[64], cn[64], m;
			for (k = 0; k < cv->n; k++) {
				for (m = 0; m < nl2 && ln[m] != G.x[cv->at[k]].line; m++)
					;
				if (m == nl2) {
					ln[nl2] = G.x[cv->at[k]].line;
					cn[nl2++] = 0;
				}
				cn[m]++;
			}
			for (m = 0; m < nl2; m++) {
				sb_printf(&at, "%s%d", m == 0 ? "" : m + 1 == nl2 ? " and " :
				    ", ", ln[m]);
				if (cn[m] > 1)
					sb_printf(&at, " (%d calls)", cn[m]);
			}
		}
		p = prop_new(f, P_CALLS, G.x[cv->at[0]].line);
		for (k = 0; k < cv->n && p->nexit < 16; k++)
			p->exitl[p->nexit++] = G.x[cv->at[k]].line;
		if (M == 1)
			snprintf(p->what, sizeof p->what, "its only call (line %s)%s",
			    at.p, nclamp && cv->nclamp == nclamp ? ": none of its clamps "
			    "bites, an unclamped variant serves it" : "");
		else
			snprintf(p->what, sizeof p->what, "%d of its %d calls (line%s %s)%s",
			    cv->n, M, strchr(at.p, ',') || strstr(at.p, " and ") ? "s" : "",
			    at.p, nclamp && cv->nclamp == nclamp ? cv->n > 1 ? ": none of "
			    "its clamps bites, an unclamped variant serves them" : ": none "
			    "of its clamps bites, an unclamped variant serves it" : "");
		sb_puts(&p->text, cv->diff.p ? cv->diff.p : cv->text);
		snprintf(p->proof, sizeof p->proof, "exhaustive: ucas equiv of each "
		    "change on the values its leaves take at %s (%ld points)",
		    cv->n > 1 ? "those calls" : "that call", cv->pts);
		p->points = cv->pts;
		p->c0 = c0;
		p->c1 = cv->c1;
		for (k = 0; k < cv->n && k < 3; k++) {
			char ct[160];
			xtext(ct, sizeof ct, cv->at[k]);
			sb_printf(&rel, "%s%s at line %d (in %s): ", k ? "; " : "", ct,
			    G.x[cv->at[k]].line, G.f[xfn[cv->at[k]]].name);
			rec_domain(&rel, cv->at[k]);
		}
		if (cv->n > 3)
			sb_printf(&rel, "; and %d more", cv->n - 3);
		snprintf(p->domain, sizeof p->domain, "%s", rel.p ? rel.p : "");
		snprintf(p->relies, sizeof p->relies, "what the arguments are at "
		    "%s: %s", cv->n > 1 ? "those calls" : "that call", rel.p ? rel.p :
		    "");
		free(at.p);
		free(rel.p);
	}
	for (j = 0; j < nv; j++) {
		free(V[j].text);
		free(V[j].what.p);
		free(V[j].diff.p);
	}
	for (i = 0; i < nst; i++)
		free(head0[i]);
	free(head0);
	free(st);
	free(orig.p);
	free(xs);
}

/*
 * Sums of products as dot products (dot.c): an integer sum with a
 * product in it, read into ucas over its leaves (each ranged by the
 * values recorded where it runs, the rows as the where clause), searched
 * under the active model's 4 x 8-bit forms and proved there.  Proposed
 * only when the module declares SPV_KHR_integer_dot_product (in GLSL,
 * GL_EXT_integer_dot_product: its dotPacked4x8EXT is the call); in one
 * that does not, the dots found are counted and said in one line.
 */
static int dot_hinted;	/* the GLSL call was shown once */
static struct {
	int ok;		/* the module declares the extension */
	int n;		/* found where it does not: not proposed */
	double gain;
} DOTD;

static void
search_dot(int f)
{
	int *xs, nx = 0, i, cap = G.nx + 64;
	uint8_t *covered = xcalloc((size_t)G.nx + 1, 1);
	xs = xmalloc((size_t)cap * sizeof *xs);
	reach_s(G.f[f].body, xs, &nx, NULL, NULL, cap);
	for (i = 0; i < nx; i++) {
		int x = xs[i], nl, w, b, j, r, *map, ok;
		const struct rec *R;
		int64_t blo[MAXP], bhi[MAXP];
		char names[MAXP + 8][40];
		int tys[MAXP + 8];
		struct ut U;
		struct uframe F0 = {0, -1, -1, 0};
		struct sbuf ta = {0}, wh = {0};
		struct av *rows;
		struct func fn;
		struct dotres dr;
		nid na;
		if (x >= xfn_n || x > G.nx || recn[x] < 0 || covered[x] ||
		    !dot_candidate(x))
			continue;
		R = &RC[recn[x]];
		nl = R->nl;
		w = R->nl + R->nex;
		if (R->nbox == 0 || nl == 0 || nl > MAXP)
			continue;
		rows = xmalloc((size_t)R->nbox * (size_t)nl * sizeof *rows);
		ok = 1;
		for (b = 0; b < R->nbox; b++)
			for (j = 0; j < nl; j++) {
				rows[b * nl + j] = R->box[b * w + j];
				if (rows[b * nl + j].bot)
					ok = 0;
			}
		for (j = 0; j < nl && ok; j++) {
			blo[j] = rows[j].lo;
			bhi[j] = rows[j].hi;
			for (r = 1; r < R->nbox; r++) {
				if (rows[r * nl + j].lo < blo[j])
					blo[j] = rows[r * nl + j].lo;
				if (rows[r * nl + j].hi > bhi[j])
					bhi[j] = rows[r * nl + j].hi;
			}
			tys[j] = G.x[R->lf[j]].ty;
			/* the leaf's own name when it is one */
			xtext(names[j], sizeof names[j], R->lf[j]);
			{
				const char *q = names[j];
				int idn = isalpha((unsigned char)*q) || *q == '_';
				for (; *q && idn; q++)
					idn = isalnum((unsigned char)*q) || *q == '_';
				if (!idn)
					snprintf(names[j], sizeof names[j], "l%d", j);
			}
		}
		if (!ok) {
			free(rows);
			continue;
		}
		map = xmalloc(((size_t)G.nx + 1) * sizeof *map);
		for (j = 0; j <= G.nx; j++)
			map[j] = -1;
		bind_leaves(R, x, x, map);
		memset(&U, 0, sizeof U);
		U.ok = 1;
		U.leaf = map;
		U.inst = 1;
		u_x(&U, &ta, x, &F0);
		/* u_x names the leaves l0, l1, ...: the source's own names
		 * where they are names */
		for (j = 0; j < nl && ta.p; j++) {
			char lj[16];
			struct sbuf nb = {0};
			const char *q = ta.p, *m;
			size_t ll;
			snprintf(lj, sizeof lj, "l%d", j);
			ll = strlen(lj);
			while ((m = strstr(q, lj)) != NULL) {
				int word = (m == ta.p || !(isalnum((unsigned char)m[-1]) ||
				    m[-1] == '_')) && !(isalnum((unsigned char)m[ll]) ||
				    m[ll] == '_');
				sb_printf(&nb, "%.*s%s", (int)(m - q), q, word ? names[j] :
				    lj);
				q = m + ll;
			}
			sb_puts(&nb, q);
			free(ta.p);
			ta = nb;
		}
		if (R->nbox > 1)
			where_text(&wh, names, tys, nl, rows, R->nbox, blo, bhi);
		ok = U.ok && u_parse(&fn, nl, names, tys, blo, bhi, ta.p,
		    R->nbox > 1 ? wh.p : NULL, &na) == 0;
		free(ta.p);
		free(wh.p);
		free(map);
		free(rows);
		if (!ok)
			continue;
		dot_search(&fn, 1, &dr);
		if (dr.status == DOT_PROVED && !DOTD.ok) {
			int sub[512], ns2 = 0, q;
			reach_x(x, sub, &ns2, 512);
			for (q = 0; q < ns2; q++)
				if (sub[q] <= G.nx)
					covered[sub[q]] = 1;
			DOTD.n++;
			DOTD.gain += dr.before - dr.after;
		} else if (dr.status == DOT_PROVED) {
			struct prop *p = prop_new(f, P_EXPR, G.x[x].line);
			char t0[300];
			struct sbuf d = {0};
			int sub[512], ns2 = 0, q;
			const char *ds;
			reach_x(x, sub, &ns2, 512);
			for (q = 0; q < ns2; q++)
				if (sub[q] <= G.nx)
					covered[sub[q]] = 1;
			xtext(t0, sizeof t0, x);
			snprintf(p->what, sizeof p->what, "a sum of products as one "
			    "dot product");
			sb_printf(&p->text, "%s\n  ->  %s", t0, shared_str(dr.body,
			    280));
			/* the leaves that are not names */
			{
				int any = 0;
				for (j = 0; j < nl; j++) {
					char lt[200], ln[16];
					snprintf(ln, sizeof ln, "l%d", j);
					if (strcmp(names[j], ln))
						continue;
					xtext(lt, sizeof lt, R->lf[j]);
					sb_printf(&p->text, "%s%s = %s", any ? ", " :
					    "\n  where ", ln, lt);
					any = 1;
				}
			}
			if (!dot_hinted++)
				sb_puts(&p->text, "\n  (in GLSL, GL_EXT_integer_dot_product's "
				    "dotPacked4x8EXT(a, b): udot4x8 with uint operands, "
				    "sdot4x8 with int ones, sudot4x8 with an int a and a "
				    "uint b)");
			if (!strcmp(dr.kind, "lanes"))
				snprintf(p->proof, sizeof p->proof, "by lanes: each lane "
				    "of each word shown to be its factor (SAT), the dot "
				    "is the exact sum of its lane products, the other "
				    "terms regrouped mod 2^32");
			else
				snprintf(p->proof, sizeof p->proof, "%s, over the values "
				    "its leaves take where it runs", sb_str(&dr.proof));
			p->c0 = dr.before;
			p->c1 = dr.after;
			rec_domain(&d, x);
			snprintf(p->domain, sizeof p->domain, "%s", d.p ? d.p : "");
			{
				struct sbuf rl = {0};
				sb_printf(&rl, "%s where line %d runs; ", d.p ? d.p : "",
				    G.x[x].line);
				for (ds = sb_str(&dr.desc); *ds; ds++)
					if (*ds == '\n')
						sb_puts(&rl, "; ");
					else
						sb_putc(&rl, *ds);
				snprintf(p->relies, sizeof p->relies, "%s", sb_str(&rl));
				free(rl.p);
			}
			free(d.p);
		}
		dotres_free(&dr);
	}
	free(xs);
	free(covered);
}

/*
 * Table reads through the texture unit (texalu.c): a read whose index is
 * an address mode of a subterm, a select against a border, a sum of
 * neighbouring entries as one filtered read.  The expression is read
 * into ucas over its leaves, as for the dot products, and searched under
 * the active model; what is proved and cheaper there is proposed with
 * what the pipeline needs for it, since GLSL alone cannot say it.
 */
static void
search_tex(int f)
{
	int *xs, nx = 0, i, cap = G.nx + 64;
	uint8_t *covered = xcalloc((size_t)G.nx + 1, 1);
	xs = xmalloc((size_t)cap * sizeof *xs);
	reach_s(G.f[f].body, xs, &nx, NULL, NULL, cap);
	for (i = 0; i < nx; i++) {
		int x = xs[i], nl, w, b, j, r, *map, ok;
		const struct rec *R;
		int64_t blo[MAXP], bhi[MAXP];
		char names[MAXP + 8][40];
		int tys[MAXP + 8];
		struct ut U;
		struct uframe F0 = {0, -1, -1, 0};
		struct sbuf ta = {0}, wh = {0};
		struct av *rows;
		struct func fn;
		struct texres tr;
		nid na;
		if (x >= xfn_n || x > G.nx || recn[x] < 0 || covered[x] ||
		    !tex_candidate(x))
			continue;
		R = &RC[recn[x]];
		nl = R->nl;
		w = R->nl + R->nex;
		if (R->nbox == 0 || nl == 0 || nl > MAXP)
			continue;
		rows = xmalloc((size_t)R->nbox * (size_t)nl * sizeof *rows);
		ok = 1;
		for (b = 0; b < R->nbox; b++)
			for (j = 0; j < nl; j++) {
				rows[b * nl + j] = R->box[b * w + j];
				if (rows[b * nl + j].bot)
					ok = 0;
			}
		for (j = 0; j < nl && ok; j++) {
			blo[j] = rows[j].lo;
			bhi[j] = rows[j].hi;
			for (r = 1; r < R->nbox; r++) {
				if (rows[r * nl + j].lo < blo[j])
					blo[j] = rows[r * nl + j].lo;
				if (rows[r * nl + j].hi > bhi[j])
					bhi[j] = rows[r * nl + j].hi;
			}
			tys[j] = G.x[R->lf[j]].ty;
			xtext(names[j], sizeof names[j], R->lf[j]);
			{
				const char *q = names[j];
				int idn = isalpha((unsigned char)*q) || *q == '_';
				for (; *q && idn; q++)
					idn = isalnum((unsigned char)*q) || *q == '_';
				if (!idn)
					snprintf(names[j], sizeof names[j], "l%d", j);
			}
		}
		if (!ok) {
			free(rows);
			continue;
		}
		map = xmalloc(((size_t)G.nx + 1) * sizeof *map);
		for (j = 0; j <= G.nx; j++)
			map[j] = -1;
		bind_leaves(R, x, x, map);
		memset(&U, 0, sizeof U);
		U.ok = 1;
		U.leaf = map;
		U.inst = 1;
		u_x(&U, &ta, x, &F0);
		for (j = 0; j < nl && ta.p; j++) {
			char lj[16];
			struct sbuf nb = {0};
			const char *q = ta.p, *m;
			size_t ll;
			snprintf(lj, sizeof lj, "l%d", j);
			ll = strlen(lj);
			while ((m = strstr(q, lj)) != NULL) {
				int word = (m == ta.p || !(isalnum((unsigned char)m[-1]) ||
				    m[-1] == '_')) && !(isalnum((unsigned char)m[ll]) ||
				    m[ll] == '_');
				sb_printf(&nb, "%.*s%s", (int)(m - q), q, word ? names[j] :
				    lj);
				q = m + ll;
			}
			sb_puts(&nb, q);
			free(ta.p);
			ta = nb;
		}
		if (R->nbox > 1)
			where_text(&wh, names, tys, nl, rows, R->nbox, blo, bhi);
		ok = U.ok && u_parse(&fn, nl, names, tys, blo, bhi, ta.p,
		    R->nbox > 1 ? wh.p : NULL, &na) == 0;
		free(ta.p);
		free(wh.p);
		free(map);
		free(rows);
		if (!ok)
			continue;
		if (tex_search(&fn, 0, &tr) == TEXR_PROVED) {
			struct prop *p = prop_new(f, P_EXPR, G.x[x].line);
			char t0[300];
			struct sbuf d = {0}, rl = {0};
			int sub[512], ns2 = 0, q;
			const char *ds;
			reach_x(x, sub, &ns2, 512);
			for (q = 0; q < ns2; q++)
				if (sub[q] <= G.nx)
					covered[sub[q]] = 1;
			xtext(t0, sizeof t0, x);
			snprintf(p->what, sizeof p->what, "table reads through the "
			    "texture unit (a proposal: the pipeline changes)");
			sb_printf(&p->text, "%s\n  ->  %s", t0, shared_str(tr.body,
			    280));
			{
				int any = 0;
				for (j = 0; j < nl; j++) {
					char lt[200], ln[16];
					snprintf(ln, sizeof ln, "l%d", j);
					if (strcmp(names[j], ln))
						continue;
					xtext(lt, sizeof lt, R->lf[j]);
					sb_printf(&p->text, "%s%s = %s", any ? ", " :
					    "\n  where ", ln, lt);
					any = 1;
				}
			}
			sb_puts(&p->text, "\n  the pipeline needs:");
			for (ds = sb_str(&tr.pipeline); *ds;) {
				const char *e = strchr(ds, '\n');
				if (!e)
					e = ds + strlen(ds);
				sb_printf(&p->text, "\n    %.*s", (int)(e - ds), ds);
				ds = *e ? e + 1 : e;
			}
			snprintf(p->proof, sizeof p->proof, "%s, over the values its "
			    "leaves take where it runs", tr.exact ? "equal, by the "
			    "derivation it relies on (texalu)" : "within a derived "
			    "error bound (texalu)");
			p->c0 = tr.before;
			p->c1 = tr.after;
			rec_domain(&d, x);
			snprintf(p->domain, sizeof p->domain, "%s", d.p ? d.p : "");
			sb_printf(&rl, "%s where line %d runs; ", d.p ? d.p : "",
			    G.x[x].line);
			for (ds = sb_str(&tr.holds); *ds; ds++)
				if (*ds == '\n')
					sb_puts(&rl, "; ");
				else
					sb_putc(&rl, *ds);
			sb_puts(&rl, "; ");
			for (ds = sb_str(&tr.proof); *ds; ds++)
				if (*ds == '\n')
					sb_puts(&rl, "; ");
				else
					sb_putc(&rl, *ds);
			snprintf(p->relies, sizeof p->relies, "%s", sb_str(&rl));
			free(rl.p);
			free(d.p);
		}
		texres_free(&tr);
	}
	free(xs);
	free(covered);
}

/* ---- (x K + C) >> S per case of a low-cardinality input ---- */

struct fitc {
	int64_t key[MAXP + 8];	/* the case: the other inputs' values */
	int64_t xhi;		/* x's largest value in the case */
	int64_t y[4097];
	uint8_t ok[4097];
	int64_t L, klo, khi;	/* the fit holds for x in 0..L, K in klo..khi */
};

/* for one case and (C, S): the longest prefix 0..L with a K; 0 if none */
static int64_t
fit_prefix(struct fitc *c, int64_t C0, int S, int64_t *klo, int64_t *khi)
{
	int64_t lo = 0, hi = (int64_t)1 << 40, x, sc = (int64_t)1 << S;
	if (!c->ok[0] || c->y[0] * sc > C0 || C0 > c->y[0] * sc + sc - 1)
		return -1;
	for (x = 1; x <= c->xhi; x++) {
		int64_t a, b;
		if (!c->ok[x])
			continue;
		a = c->y[x] * sc - C0;
		b = c->y[x] * sc + sc - 1 - C0;
		/* ceil(a / x), floor(b / x) */
		a = a >= 0 ? (a + x - 1) / x : -((-a) / x);
		b = b >= 0 ? b / x : -((-b + x - 1) / x);
		if (a < 0)
			a = 0;
		if ((a > lo ? a : lo) > (b < hi ? b : hi))
			break;
		lo = a > lo ? a : lo;
		hi = b < hi ? b : hi;
	}
	*klo = lo;
	*khi = hi;
	return x - 1;
}

/* K as one formula of the case variable cv: A / ((1 << g) - 1), A / (1 << g)
 * or A / g, each case's K in its interval; the form's number, or -1 */
static int
find_form(struct fitc *cs, int nc, int cv, int64_t *A)
{
	int fm, j;
	for (fm = 0; fm < 3; fm++) {
		int64_t g0 = cs[0].key[cv], d0, k;
		d0 = fm == 0 ? ((int64_t)1 << g0) - 1 : fm == 1 ? (int64_t)1 << g0 :
		    g0;
		if (g0 < 0 || g0 > 30 || d0 <= 0)
			continue;
		for (k = cs[0].klo; k <= cs[0].khi && k - cs[0].klo < 100000; k++) {
			int64_t A0 = k * d0;
			for (j = 1; j < nc; j++) {
				int64_t g = cs[j].key[cv], d;
				if (g < 0 || g > 30)
					break;
				d = fm == 0 ? ((int64_t)1 << g) - 1 : fm == 1 ?
				    (int64_t)1 << g : g;
				if (d <= 0 || A0 % d || A0 / d < cs[j].klo || A0 / d >
				    cs[j].khi)
					break;
			}
			if (j == nc) {
				*A = A0;
				return fm;
			}
		}
	}
	return -1;
}

/*
 * (x >> S) & M per case: a bit field of x whose place is the case's
 * (et_sub's subblock is bit 3 or bit 1 of i by flip), one shift by a
 * selected count and a mask where the function compares or selects per
 * case.  Every case's every value must fit, M the same in all; proved
 * case by case on the domain, as the multiply form is.  1: proposed.
 */
static int
fit_shiftmask(int f, struct dom *D, struct fitc *cs, int nc, int wide,
    double c0)
{
	int Sj[64], j, i, b, S, rc;
	int64_t M = 0, x;
	char names[MAXP + 8][40], wit[200];
	int tys[MAXP + 8];
	double c1 = model_active()->c[C_SHIFT] + model_active()->c[C_LOGIC] +
	    model_active()->c[C_SELECT] * (nc - 1) + model_active()->c[C_CMP] *
	    (nc - 1);
	struct sbuf form_t = {0}, sel = {0}, kt = {0}, b2 = {0};
	struct av *rows;
	char *orig;
	long pts = 0;
	const char *wn = G.y[D->vars[wide]].name;

	if (nc > 4 || c1 >= c0)
		return 0;
	/* the mask: every value's bits */
	for (j = 0; j < nc; j++)
		for (x = 0; x <= cs[j].xhi; x++)
			if (cs[j].ok[x]) {
				if (cs[j].y[x] < 0)
					return 0;
				M |= cs[j].y[x];
			}
	for (b = 0; b < 32 && (M >> b); b++)
		;
	if (b == 0)
		return 0;
	M = ((int64_t)1 << b) - 1;
	for (j = 0; j < nc; j++) {
		for (S = 0; S < 32; S++) {
			for (x = 0; x <= cs[j].xhi; x++)
				if (cs[j].ok[x] && ((x >> S) & M) != cs[j].y[x])
					break;
			if (x > cs[j].xhi)
				break;
		}
		if (S == 32)
			return 0;
		Sj[j] = S;
	}
	for (j = 1; j < nc && Sj[j] == Sj[0]; j++)
		;
	if (j == nc)
		return 0;	/* one shift for all: not per case */
	orig = fn_ucas(f, D, names, tys);
	if (orig == NULL)
		return 0;
	rows = xmalloc((size_t)nc * (size_t)D->nv * sizeof *rows);
	for (j = 0; j < nc; j++) {
		for (i = 0; i < D->nv; i++)
			rows[j * D->nv + i] = av_k(tys[i], cs[j].key[i]);
		rows[j * D->nv + wide] = av_iv(tys[wide], 0, cs[j].xhi);
	}
	/* the shift: by a select on the case */
	for (j = 0; j < nc - 1; j++) {
		int first = 1;
		sb_puts(&sel, "(");
		for (i = 0; i < D->nv; i++)
			if (i != wide) {
				sb_printf(&sel, "%s%s == %s(%lld)", first ? "" : " && ",
				    names[i], u_ty(tys[i]), (long long)cs[j].key[i]);
				first = 0;
			}
		sb_printf(&sel, " ? %s(%d) : ", u_ty(tys[wide]), Sj[j]);
	}
	sb_printf(&sel, "%s(%d)", u_ty(tys[wide]), Sj[nc - 1]);
	for (j = 0; j < nc - 1; j++)
		sb_puts(&sel, ")");
	sb_printf(&form_t, "(%s >> %s) & %s(%lld)", names[wide], sel.p,
	    u_ty(tys[wide]), (long long)M);
	rc = prove_rows(names, tys, D->nv, rows, nc, orig, form_t.p, &pts, wit,
	    sizeof wit);
	if (rc == 1) {
		struct prop *p = prop_new(f, P_FIT, G.f[f].line);
		struct sbuf gs = {0};
		for (j = 0; j < nc; j++) {
			int first = 1;
			sb_printf(&kt, "%s%d for ", j ? "; " : "", Sj[j]);
			for (i = 0; i < D->nv; i++)
				if (i != wide) {
					sb_printf(&kt, "%s%s = %lld", first ? "" : ", ",
					    G.y[D->vars[i]].name, (long long)cs[j].key[i]);
					first = 0;
				}
		}
		/* the GLSL: the select of the counts, its compares on the case
		 * inputs as the source names them */
		for (j = 0; j < nc - 1; j++) {
			int first = 1;
			sb_puts(&gs, "(");
			for (i = 0; i < D->nv; i++)
				if (i != wide) {
					sb_printf(&gs, "%s%s == %lld%s", first ? "" : " && ",
					    G.y[D->vars[i]].name, (long long)cs[j].key[i],
					    tys[i] == TU ? "u" : "");
					first = 0;
				}
			sb_printf(&gs, " ? %du : ", Sj[j]);
		}
		sb_printf(&gs, "%du", Sj[nc - 1]);
		for (j = 0; j < nc - 1; j++)
			sb_puts(&gs, ")");
		snprintf(p->what, sizeof p->what, "per case a shift and a mask: "
		    "%s(%s) = (%s >> S) & %lldu with S = %s, one shift by a "
		    "selected count", G.f[f].name, wn, wn, (long long)M, kt.p);
		sb_printf(&p->text, "%s %s(", ty_name(G.f[f].ty), G.f[f].name);
		for (i = 0; i < G.f[f].np; i++)
			sb_printf(&p->text, "%s%s %s", i ? ", " : "",
			    ty_name(G.y[G.f[f].p[i]].ty), G.y[G.f[f].p[i]].name);
		sb_printf(&p->text, ") {\n    return (%s >> %s) & %lldu;\n}\n", wn,
		    gs.p, (long long)M);
		snprintf(p->proof, sizeof p->proof, "exhaustive: ucas equiv of the "
		    "function and the form, case by case (%ld points)", pts);
		p->points = pts;
		p->c0 = c0;
		p->c1 = c1;
		vars_text(&b2, D->vars, D->nv, rows, nc);
		snprintf(p->domain, sizeof p->domain, "%s", b2.p ? b2.p : "");
		snprintf(p->relies, sizeof p->relies, "the inferred domain (%s): the "
		    "form holds on all of it", b2.p ? b2.p : "");
		free(gs.p);
	} else if (rc == 0)
		printf("  note: %s: the interpreter and ucas equiv disagree (%s); "
		    "nothing proposed\n", G.f[f].name, wit);
	free(rows);
	free(form_t.p);
	free(sel.p);
	free(kt.p);
	free(b2.p);
	free(orig);
	return rc == 1;
}

static void
search_fit(int f, struct dom *D)
{
	int wide = -1, i, r, j, nc = 0, S, best_S = -1, ncase_var = 0, cv = -1;
	int have_sm = 0;
	struct fitc *cs;
	int64_t best_C = 0, best_cov = -1, total = 0;
	(void)total;
	double c0;
	char names[MAXP + 8][40];
	int tys[MAXP + 8];
	char *orig;

	if (D->nv < 2)
		return;
	/* the one wide input from 0; the others at most 8 values a row */
	for (i = 0; i < D->nv; i++) {
		int64_t span = 0;
		for (r = 0; r < D->nr; r++) {
			const struct av *v = &D->rows[r * D->nv + i];
			if (v->hi - v->lo > span)
				span = v->hi - v->lo;
		}
		if (span >= 8) {
			if (wide >= 0)
				return;
			wide = i;
		}
	}
	if (wide < 0 || G.y[D->vars[wide]].ty == TB)
		return;
	c0 = scost(G.f[f].body);
	if (c0 <= model_active()->c[C_IMUL] + model_active()->c[C_IADD] +
	    model_active()->c[C_SHIFT])
		return;		/* already as cheap as the form */
	cs = xcalloc(64, sizeof *cs);
	for (r = 0; r < D->nr; r++) {
		int64_t cur[MAXP + 8];
		const struct av *row = &D->rows[r * D->nv];
		if (row[wide].lo != 0 || row[wide].hi > 4096)
			goto out;
		for (i = 0; i < D->nv; i++)
			cur[i] = row[i].lo;
		for (;;) {
			int k;
			for (k = 0; k < nc; k++) {
				for (i = 0; i < D->nv; i++)
					if (i != wide && cs[k].key[i] != cur[i])
						break;
				if (i == D->nv)
					break;
			}
			if (k == nc) {
				if (nc == 64)
					goto out;
				for (i = 0; i < D->nv; i++)
					cs[nc].key[i] = cur[i];
				nc++;
			}
			if (row[wide].hi > cs[k].xhi)
				cs[k].xhi = row[wide].hi;
			/* next: every input but the wide one */
			for (i = 0; i < D->nv; i++) {
				if (i == wide)
					continue;
				if (cur[i] < row[i].hi) {
					cur[i]++;
					break;
				}
				cur[i] = row[i].lo;
			}
			if (i == D->nv)
				break;
		}
	}
	if (nc < 2)
		goto out;
	/* the values, case by case */
	for (j = 0; j < nc; j++) {
		int64_t x;
		for (i = 0; i < D->nv; i++)
			C.v[D->vars[i]] = cs[j].key[i];
		for (x = 0; x <= cs[j].xhi; x++) {
			int ub;
			C.v[D->vars[wide]] = x;
			cs[j].y[x] = run_fn(f, &ub);
			cs[j].ok[x] = !ub;
		}
		total += cs[j].xhi + 1;
	}
	/* the case inputs that vary */
	for (i = 0; i < D->nv; i++) {
		if (i == wide)
			continue;
		for (j = 1; j < nc; j++)
			if (cs[j].key[i] != cs[0].key[i])
				break;
		if (j < nc) {
			ncase_var++;
			cv = i;
		}
	}
	/* sorted by the case, for the report */
	for (j = 1; j < nc; j++) {
		struct fitc tmp = cs[j];
		int q = j;
		while (q > 0 && memcmp(cs[q - 1].key, tmp.key, sizeof tmp.key) > 0 &&
		    (cv < 0 || cs[q - 1].key[cv] > tmp.key[cv])) {
			cs[q] = cs[q - 1];
			q--;
		}
		cs[q] = tmp;
	}
	/* a bit field of x at a place per case: holds on the whole domain,
	 * so the multiply below is said only where it does too */
	have_sm = fit_shiftmask(f, D, cs, nc, wide, c0);
	/* the most the form can cover; then the least S that covers it with
	 * K one formula of the case (else the least S), C = 2^(S-1) first */
	{
		int pass, have_form = 0;
		for (pass = 0; pass < 2; pass++)
			for (S = 0; S <= 12 && !have_form; S++) {
				int64_t sc = (int64_t)1 << S, clo = INT64_MIN, chi = INT64_MAX;
				int64_t C0, cand[4097];
				int ncand = 0, q;
				for (j = 0; j < nc; j++) {
					if (!cs[j].ok[0])
						goto out;
					clo = cs[j].y[0] * sc > clo ? cs[j].y[0] * sc : clo;
					chi = cs[j].y[0] * sc + sc - 1 < chi ? cs[j].y[0] * sc +
					    sc - 1 : chi;
				}
				if (sc / 2 >= clo && sc / 2 <= chi)
					cand[ncand++] = sc / 2;
				for (C0 = clo; C0 <= chi && ncand < 4097; C0++)
					if (C0 != sc / 2)
						cand[ncand++] = C0;
				for (q = 0; q < ncand && !have_form; q++) {
					int64_t cov = 0, kl, kh;
					C0 = cand[q];
					for (j = 0; j < nc; j++) {
						int64_t L = fit_prefix(&cs[j], C0, S, &kl, &kh);
						if (L < 3) {
							cov = -1;
							break;
						}
						cov += L + 1;
					}
					if (pass == 0) {
						if (cov > best_cov) {
							best_cov = cov;
							best_S = S;
							best_C = C0;
						}
						continue;
					}
					if (cov != best_cov)
						continue;
					if (ncase_var == 1) {
						int64_t A0;
						for (j = 0; j < nc; j++)
							cs[j].L = fit_prefix(&cs[j], C0, S, &cs[j].klo,
							    &cs[j].khi);
						if (find_form(cs, nc, cv, &A0) >= 0) {
							best_S = S;
							best_C = C0;
							have_form = 1;
						}
					}
				}
			}
	}
	if (best_S < 0 || best_cov <= 0)
		goto out;
	for (j = 0; j < nc; j++)
		cs[j].L = fit_prefix(&cs[j], best_C, best_S, &cs[j].klo, &cs[j].khi);
	{
		/* K as one formula of the case: A / ((1 << g) - 1), A / (1 << g),
		 * A / g */
		int form = -1;
		int64_t A_ = 0, K[64];
		static const char *const fn_txt[] = {"((1u << %s) - 1u)", "(1u << %s)",
		    "%s"};
		if (ncase_var == 1)
			form = find_form(cs, nc, cv, &A_);
		for (j = 0; j < nc; j++) {
			if (form >= 0) {
				int64_t g = cs[j].key[cv];
				K[j] = A_ / (form == 0 ? ((int64_t)1 << g) - 1 : form == 1 ?
				    (int64_t)1 << g : g);
			} else {
				/* the slope's rounding where it is in range */
				int64_t ideal = cs[j].L > 0 ? (cs[j].y[cs[j].L] <<
				    best_S) / cs[j].L : cs[j].klo;
				K[j] = ideal < cs[j].klo ? cs[j].klo : ideal > cs[j].khi ?
				    cs[j].khi : ideal;
			}
		}
		/* the proof: per case, x in 0..L, original against the form */
		orig = fn_ucas(f, D, names, tys);
		if (orig == NULL)
			goto out;
		{
			struct sbuf form_t = {0}, rel = {0}, kt = {0}, b = {0};
			struct av *rows = xmalloc((size_t)nc * (size_t)D->nv *
			    sizeof *rows);
			long pts = 0;
			char wit[200];
			int rc, full = 1, pw2 = ncase_var == 1;
			const char *wn = G.y[D->vars[wide]].name;
			char kname[48];
			for (j = 0; j < nc; j++) {
				for (i = 0; i < D->nv; i++)
					rows[j * D->nv + i] = av_k(tys[i], cs[j].key[i]);
				rows[j * D->nv + wide] = av_iv(tys[wide], 0, cs[j].L);
				if (cs[j].L != cs[j].xhi)
					full = 0;
				if (ncase_var == 1 && (cs[j].key[cv] < 0 || cs[j].key[cv] >
				    30 || cs[j].L + 1 != (int64_t)1 << cs[j].key[cv]))
					pw2 = 0;
			}
			/* the form: K by a select on the case */
			sb_printf(&form_t, "((%s * ", names[wide]);
			for (j = 0; j < nc - 1; j++) {
				sb_puts(&form_t, "(");
				for (i = 0; i < D->nv; i++)
					if (i != wide)
						sb_printf(&form_t, "%s%s == %s(%lld)", i == (wide == 0 ?
						    1 : 0) ? "" : " && ", names[i], u_ty(tys[i]),
						    (long long)cs[j].key[i]);
				sb_printf(&form_t, " ? %s(%lld) : ", u_ty(tys[wide]),
				    (long long)K[j]);
			}
			sb_printf(&form_t, "%s(%lld)", u_ty(tys[wide]),
			    (long long)K[nc - 1]);
			for (j = 0; j < nc - 1; j++)
				sb_puts(&form_t, ")");
			sb_printf(&form_t, ") + %s(%lld)) >> %s(%d)", u_ty(tys[wide]),
			    (long long)best_C, u_ty(tys[wide]), best_S);
			rc = prove_rows(names, tys, D->nv, rows, nc, orig, form_t.p, &pts,
			    wit, sizeof wit);
			if (rc == 1 && have_sm && !full)
				rc = -1;	/* the shift form holds everywhere */
			if (rc == 1) {
				struct prop *p = prop_new(f, P_FIT, G.f[f].line);
				snprintf(kname, sizeof kname, "%s_K", G.f[f].name);
				for (j = 0; j < nc; j++) {
					sb_printf(&kt, "%s%lld", j ? ", " : "", (long long)K[j]);
				}
				sb_puts(&kt, " for ");
				for (j = 0; j < nc; j++) {
					int first = 1;
					sb_puts(&kt, j ? "; " : "");
					for (i = 0; i < D->nv; i++)
						if (i != wide && (ncase_var != 1 || i == cv)) {
							sb_printf(&kt, "%s%s = %lld", first ? "" : ", ",
							    G.y[D->vars[i]].name, (long long)cs[j].key[i]);
							first = 0;
						}
				}
				if (form >= 0) {
					char ft[64];
					snprintf(ft, sizeof ft, fn_txt[form],
					    G.y[D->vars[cv]].name);
					snprintf(p->what, sizeof p->what, "per case a multiply-add "
					    "and a shift: %s(%s) = (%s * K + %lldu) >> %du with K "
					    "= %lld / %s (%s), a constant per case (set it where "
					    "%s is set)", G.f[f].name, wn, wn, (long long)best_C,
					    best_S, (long long)A_, ft, kt.p,
					    G.y[D->vars[cv]].name);
				} else
				{
					char cpart[32] = "";
					if (best_C)
						snprintf(cpart, sizeof cpart, " + %lldu",
						    (long long)best_C);
					snprintf(p->what, sizeof p->what, "per case a multiply%s "
					    "and a shift: %s(%s) = (%s * K%s) >> %du with K = %s, "
					    "a constant per case", best_C ? "-add" : "",
					    G.f[f].name, wn, wn, cpart, best_S, kt.p);
				}
				sb_printf(&p->text, "/* %s: K for the case, set once per "
				    "case (%s) */\n%s %s(", kname, kt.p, ty_name(G.f[f].ty),
				    G.f[f].name);
				for (i = 0; i < G.f[f].np; i++)
					sb_printf(&p->text, "%s%s %s", i ? ", " : "",
					    ty_name(G.y[G.f[f].p[i]].ty), G.y[G.f[f].p[i]].name);
				if (best_C)
					sb_printf(&p->text, ") {\n    return (%s * %s + %lldu) >> "
					    "%du;\n}\n", wn, kname, (long long)best_C, best_S);
				else
					sb_printf(&p->text, ") {\n    return (%s * %s) >> %du;\n}"
					    "\n", wn, kname, best_S);
				snprintf(p->proof, sizeof p->proof, "exhaustive: ucas equiv "
				    "of the function and the form, case by case (%ld "
				    "points)", pts);
				p->points = pts;
				p->c0 = c0;
				p->c1 = model_active()->c[C_IMUL] + model_active()->c[C_IADD] +
				    model_active()->c[C_SHIFT];
				vars_text(&b, D->vars, D->nv, rows, nc);
				snprintf(p->domain, sizeof p->domain, "%s", b.p ? b.p : "");
				/* what must hold at the calls */
				if (full)
					sb_printf(&rel, "the inferred domain (%s): the form holds "
					    "on all of it", b.p);
				else {
					int k;
					if (pw2)
						sb_printf(&rel, "%s <= 2^%s - 1", wn,
						    G.y[D->vars[cv]].name);
					else {
						sb_printf(&rel, "%s <= ", wn);
						for (j = 0; j < nc; j++)
							sb_printf(&rel, "%s%lld", j ? ", " : "",
							    (long long)cs[j].L);
						sb_puts(&rel, " in the cases above");
					}
					sb_puts(&rel, " (beyond it the function is not this form); ");
					/* each call: does the analysis show it? */
					{
						struct sbuf yes = {0}, no = {0}, never = {0};
						for (k = 0; k < nrc; k++) {
							const struct rec *R = &RC[k];
							const struct gx *cx = &G.x[R->x];
							int wpos = -1, cpos = -1, bx, holds = 1;
							int64_t worst = 0;
							if (cx->k != X_CALL || cx->sym != f)
								continue;
							for (i = 0; i < G.f[f].np; i++)
								if (G.f[f].p[i] == D->vars[wide])
									wpos = i;
							for (i = 0; i < R->nex; i++)
								if (R->ex[i] == D->vars[cv])
									cpos = R->nl + i;
							for (i = 0; i < G.f[f].np; i++)
								if (G.f[f].p[i] == D->vars[cv])
									cpos = i;
							if (R->nbox == 0) {
								sb_printf(&never, "%s%d", never.n ? ", " : "",
								    cx->line);
								continue;
							}
							if (wpos < 0 || cpos < 0)
								holds = 0;
							for (bx = 0; bx < R->nbox && holds; bx++) {
								const struct av *row = &R->box[bx * (R->nl +
								    R->nex)];
								int64_t g = row[cpos].lo, L = -1;
								if (row[cpos].lo != row[cpos].hi) {
									holds = 0;
									break;
								}
								for (j = 0; j < nc; j++)
									if (cs[j].key[cv] == g)
										L = cs[j].L;
								if (row[wpos].hi > L) {
									holds = 0;
									worst = row[wpos].hi;
								}
							}
							if (holds)
								sb_printf(&yes, "%s%d", yes.n ? ", " : "", cx->line);
							else if (!strstr(no.p ? no.p : "", "") || 1) {
								char at[120];
								xtext(at, sizeof at, G.xarg[cx->a + (wpos >= 0 ?
								    wpos : 0)]);
								sb_printf(&no, "%s%d (%s up to %lld there)",
								    no.n ? ", " : "", cx->line, at,
								    (long long)worst);
							}
						}
						if (yes.n)
							sb_printf(&rel, "the analysis shows it at line%s %s",
							    strchr(yes.p, ',') ? "s" : "", yes.p);
						if (no.n)
							sb_printf(&rel, "%snot at line%s %s: true there only "
							    "if those values keep to it", yes.n ? "; " : "",
							    strchr(no.p, ',') ? "s" : "", no.p);
						if (never.n)
							sb_printf(&rel, "; line%s %s never run%s (the "
							    "analysis reaches no call there)", strchr(never.p,
							    ',') ? "s" : "", never.p, strchr(never.p, ',') ?
							    "" : "s");
						free(yes.p);
						free(no.p);
						free(never.p);
					}
				}
				snprintf(p->relies, sizeof p->relies, "%s", rel.p ? rel.p : "");
			}
			free(rows);
			free(form_t.p);
			free(rel.p);
			free(kt.p);
			free(b.p);
			free(orig);
		}
	}
out:
	free(cs);
}

/* ---- calls a loop never changes ---- */

/* the variables an expression reads (locals, parameters, globals) */
static void
reads_x(int x, uint8_t *rd, int *pure)
{
	const struct gx *g;
	int i;
	if (x < 0)
		return;
	g = &G.x[x];
	switch (g->k) {
	case X_VAR:
		rd[g->sym] = 1;
		return;
	case X_CALL:
		for (i = 0; i < g->b; i++)
			reads_x(G.xarg[g->a + i], rd, pure);
		if (g->sym >= 0) {
			int y;
			if (A.impure[g->sym] || A.nwrl[g->sym])
				*pure = 0;
			for (y = 0; y < A.ns; y++)
				if (A.rd[g->sym][y])
					rd[y] = 1;
		}
		return;
	case X_ASG:
	case X_INC:
		*pure = 0;
		reads_x(g->a, rd, pure);
		reads_x(g->b, rd, pure);
		return;
	default:
		reads_x(g->a, rd, pure);
		if (g->k != X_UN && g->k != X_MEM)
			reads_x(g->b, rd, pure);
		if (g->k == X_SEL)
			reads_x(g->c, rd, pure);
	}
}

/* what a loop writes: its locals and globals, and what its calls write */
static void
writes_s(int s, uint8_t *w)
{
	static int xs[8192], ss[8192];
	int nx = 0, i, nss = 0;
	reach_s(s, xs, &nx, ss, &nss, 8192);
	for (i = 0; i < nss; i++)
		if (G.s[ss[i]].k == S_DECL)
			w[G.s[ss[i]].sym] = 1;
	for (i = 0; i < nx; i++) {
		const struct gx *g = &G.x[xs[i]];
		if (g->k == X_ASG || g->k == X_INC) {
			int l = g->a;
			while (G.x[l].k == X_IDX || G.x[l].k == X_MEM)
				l = G.x[l].a;
			if (G.x[l].k == X_VAR)
				w[G.x[l].sym] = 1;
		}
		if (g->k == X_CALL && g->sym >= 0) {
			int y, f = g->sym, k;
			for (y = 0; y < A.ns; y++)
				if (A.wr[f][y])
					w[y] = 1;
			for (k = 0; k < g->b && k < G.f[f].np; k++)
				if (G.y[G.f[f].p[k]].out && G.x[G.xarg[g->a + k]].k == X_VAR)
					w[G.x[G.xarg[g->a + k]].sym] = 1;
		}
	}
}

/* is call x (in a function, outside its loops) the same on every pass of
 * a loop writing w?  args: the caller's parameters bound to invariant
 * values (inv[p] = 1) */
static int
call_invariant(int x, const uint8_t *w, const uint8_t *inv)
{
	uint8_t *rd = xcalloc((size_t)A.ns, 1);
	int pure = 1, y, ok = 1;
	const struct gx *g = &G.x[x];
	reads_x(x, rd, &pure);
	if (!pure || A.impure[g->sym] || A.nwrl[g->sym])
		ok = 0;
	for (y = 0; y < A.ns && ok; y++) {
		if (!rd[y])
			continue;
		if (w[y])
			ok = 0;
		else if (G.y[y].kind == Y_PARAM || G.y[y].kind == Y_LOCAL)
			ok = inv == NULL || inv[y];
	}
	free(rd);
	return ok;
}

static int
in_loop_stmt(int s, int x)
{
	int xs[4096], nx = 0, i;
	reach_s(s, xs, &nx, NULL, NULL, 4096);
	for (i = 0; i < nx; i++)
		if (xs[i] == x)
			return 1;
	return 0;
}

static void
search_invariant(void)
{
	int f, i, j, k;
	uint8_t *seen = xcalloc((size_t)G.nx + 1, 1);
	for (f = 0; f < G.nf; f++) {
		static int *ss, *xs;
		int nss = 0, nx = 0;
		if (G.f[f].body < 0)
			continue;
		ss = xrealloc(ss, ((size_t)G.nx + G.ns + 16) * sizeof *ss);
		xs = xrealloc(xs, ((size_t)G.nx + G.ns + 16) * sizeof *xs);
		reach_s(G.f[f].body, xs, &nx, ss, &nss, G.nx + G.ns);
		for (i = 0; i < nss; i++) {
			int L = ss[i], cx[4096], ncx = 0;
			uint8_t *w;
			const struct gs *t = &G.s[L];
			int var;
			int64_t lo, last;
			double trips = 0;
			if (t->k != S_FOR && t->k != S_WHILE && t->k != S_DO)
				continue;
			if (t->k == S_FOR && canon_loop(L, &var, &lo, &last))
				trips = (double)(last - lo + 1);
			w = xcalloc((size_t)A.ns, 1);
			writes_s(L, w);
			reach_s(t->d, cx, &ncx, NULL, NULL, 4096);
			if (t->b >= 0)
				reach_x(t->b, cx, &ncx, 4096);
			for (j = 0; j < ncx; j++) {
				const struct gx *g = &G.x[cx[j]];
				int f1;
				if (g->k != X_CALL || g->sym < 0)
					continue;
				f1 = g->sym;
				if (!seen[cx[j]] && call_invariant(cx[j], w, NULL)) {
					/* the call itself */
					struct prop *p;
					double c = G.f[f1].body >= 0 ? scost(G.f[f1].body) : 0;
					char ct[160];
					seen[cx[j]] = 1;
					xtext(ct, sizeof ct, cx[j]);
					p = prop_new(f, P_INV, g->line);
					p->callee = f1;
					snprintf(p->what, sizeof p->what, "%s at line %d (in %s) is "
					    "the same on every pass of the loop at line %d: hoist it "
					    "out of the loop at line %d", ct, g->line, G.f[f].name,
					    t->line, t->line);
					p->c0 = c;
					p->c1 = trips;
					continue;
				}
				/* inside the callee, outside its own loops: calls whose
				 * arguments come from parameters bound to invariant values */
				if (G.f[f1].body < 0)
					continue;
				{
					uint8_t *inv = xcalloc((size_t)A.ns, 1);
					int ys[2048], nys = 0, xs2[4096], nx2 = 0, m, fl[4096], nfl = 0;
					for (m = 0; m < g->b && m < G.f[f1].np; m++) {
						int pure = 1, y, ok = 1;
						uint8_t *rd = xcalloc((size_t)A.ns, 1);
						reads_x(G.xarg[g->a + m], rd, &pure);
						for (y = 0; y < A.ns; y++)
							if (rd[y] && w[y])
								ok = 0;
						if (ok && pure && A.nasg[G.f[f1].p[m]] == 0)
							inv[G.f[f1].p[m]] = 1;
						free(rd);
					}
					reach_s(G.f[f1].body, xs2, &nx2, ys, &nys, 2048);
					/* the calls inside f1's loops are not candidates */
					for (m = 0; m < nys; m++)
						if (G.s[ys[m]].k == S_FOR || G.s[ys[m]].k == S_WHILE ||
						    G.s[ys[m]].k == S_DO) {
							int n0 = nfl;
							reach_s(G.s[ys[m]].d, fl, &nfl, NULL, NULL, 4096);
							if (G.s[ys[m]].b >= 0)
								reach_x(G.s[ys[m]].b, fl, &nfl, 4096);
							(void)n0;
						}
					for (m = 0; m < nx2; m++) {
						const struct gx *h = &G.x[xs2[m]];
						int q, inl = 0;
						if (h->k != X_CALL || h->sym < 0 || seen[xs2[m]])
							continue;
						for (q = 0; q < nfl; q++)
							if (fl[q] == xs2[m])
								inl = 1;
						if (inl || !call_invariant(xs2[m], w, inv))
							continue;
						{
							struct prop *p;
							double c = G.f[h->sym].body >= 0 ?
							    scost(G.f[h->sym].body) : 0;
							char ct[160], ca[160];
							seen[xs2[m]] = 1;
							xtext(ct, sizeof ct, xs2[m]);
							xtext(ca, sizeof ca, cx[j]);
							p = prop_new(f1, P_INV, h->line);
							p->callee = h->sym;
							snprintf(p->what, sizeof p->what, "%s at line %d (in "
							    "%s) is the same on every pass of %s's loop at "
							    "line %d, which calls %s at line %d: hoist it out "
							    "of the loop at line %d (compute it before the "
							    "loop and pass it to %s)", ct, h->line,
							    G.f[f1].name, G.f[f].name, t->line, G.f[f1].name,
							    g->line, t->line, G.f[f1].name);
							p->c0 = c;
							p->c1 = trips;
						}
					}
					free(inv);
					(void)k;
				}
			}
			free(w);
		}
	}
	free(seen);
}

/* ---- the report ---- */

static void
pr_lines(const char *text, const char *ind)
{
	const char *p = text;
	while (p && *p) {
		const char *e = strchr(p, '\n');
		size_t n = e ? (size_t)(e - p) : strlen(p);
		if (n)
			printf("%s%.*s\n", ind, (int)n, p);
		p = e ? e + 1 : NULL;
	}
}

static const char *
kind_name(int k)
{
	return k == P_FUNC ? "function" : k == P_EXPR ? "expression" :
	    k == P_FIT ? "case_fit" : k == P_LOOP ? "loop" : k == P_TAB ?
	    "loop_map" : k == P_CALLS ? "call_variant" : "invariant_call";
}

static int loops_only;	/* no function calls another: the loops alone */

/*
 * Would a driver's own loop-invariant code motion move a call of f out of
 * a loop?  Once inlined, a call that runs no loop, reads no memory (no
 * buffer, no array but a constant table) and calls only such functions is
 * plain arithmetic on its arguments: every back end hoists that.
 */
static int
lic_ok(int f, int depth)
{
	int *xs, *ss, nx = 0, nss = 0, i, ok = 1, cap = G.nx + G.ns + 64;
	if (G.f[f].body < 0 || depth > 16 || has_loop(G.f[f].body))
		return 0;
	for (i = 0; i < A.ns; i++)
		if (A.rd[f][i] && (G.y[i].kind == Y_BUF || (G.y[i].arr &&
		    !G.y[i].tab)))
			return 0;
	xs = xmalloc((size_t)cap * sizeof *xs);
	ss = xmalloc((size_t)cap * sizeof *ss);
	reach_s(G.f[f].body, xs, &nx, ss, &nss, cap);
	for (i = 0; i < nx && ok; i++) {
		const struct gx *g = &G.x[xs[i]];
		if (g->k == X_CALL && g->sym == -B_UNK)
			ok = 0;
		else if (g->k == X_CALL && g->sym >= 0 && g->sym != f)
			ok = lic_ok(g->sym, depth + 1);
		else if (g->k == X_VAR && (G.y[g->sym].kind == Y_BUF ||
		    (G.y[g->sym].arr && !G.y[g->sym].tab)))
			ok = 0;	/* a load: what the loop writes may alias it */
	}
	free(xs);
	free(ss);
	return ok;
}

static int
has_loops(void)
{
	int s;
	for (s = 0; s < G.ns; s++)
		if (G.s[s].k == S_FOR || G.s[s].k == S_WHILE)
			return 1;
	return 0;
}

static void
report(int rounds, double secs)
{
	int i, f, nshown = 0, ncheap = 0, nprop = 0, moved[64], nmv[64], nmoved = 0,
	    said_inv = 0;
	const char *model = sym_str(model_active()->name);
	struct sbuf cheap = {0};

	for (i = 0; i < npr; i++)
		if (PR[i].kind != P_INV)
			nprop++;
	if (loops_only)
		printf("function level (the GLSL source read by ScarletNX's GLSL ES "
		    "parser): its loops only, as no function of the source calls "
		    "another (abstract interpretation from main, %d rounds", rounds +
		    1);
	else
		printf("function level (the GLSL source read by ScarletNX's GLSL ES "
		    "parser): %d functions; each one's domain inferred from every "
		    "call (abstract interpretation from main, %d rounds", G.nf,
		    rounds + 1);
	if (secs >= 0.5)	/* shorter times are noise, not news */
		printf(", %.1f s", secs);
	printf(")\n");
	if (dropped[0])
		printf("  not read (the parser refused them, so they are left out): "
		    "%s\n", dropped);
	if (A.out_of_time)
		printf("  note: the analysis stopped at the time budget: domains may be "
		    "wider than they are, nothing below depends on more\n");
	if (lp_note && has_loops())
		printf("  note: each loop's first two passes were not run apart (%s), "
		    "so no loop is said to leave early\n", lp_note == 1 ? "the time "
		    "budget ran out first: `budget N` for more" : lp_note == 2 ?
		    "the analysis did not settle in 8 rounds" : "doing it grew a "
		    "summary of the analysis");
	rep_list("function_level");
	for (f = 0; f < G.nf; f++) {
		int head = 0, said_calls = 0;
		for (i = 0; i < npr; i++) {
			struct prop *p = &PR[i];
			if (p->f != f || p->kind == P_INV)
				continue;
			if (!head) {
				struct sbuf cl = {0};
				callers_text(&cl, f);
				printf("  %s (line %d), called from %s\n", G.f[f].name,
				    G.f[f].line, cl.p);
				free(cl.p);
				head = 1;
			}
			if (p->kind == P_CALLS) {
				if (!said_calls)
					printf("    what its calls need (each change proved on the "
					    "values its leaves take at those calls):\n");
				said_calls = 1;
				printf("      %s: %g -> %g a call under %s (%ld points)\n",
				    p->what, p->c0, p->c1, model, p->points);
				pr_lines(p->text.p, "        ");
				printf("        where: %s\n", p->domain);
			} else if (p->kind == P_EXPR || p->kind == P_LOOP || p->kind == P_TAB) {
				printf("    line %d: %s\n", p->line, p->what);
				pr_lines(p->text.p, "      ");
				printf("      proof: %s\n", p->proof);
				printf("      cost under %s: %g -> %g each time %s runs\n",
				    model, p->c0, p->c1, p->kind == P_LOOP || p->kind == P_TAB ?
				    "the loop" : "it");
				printf("      relies on: %s\n", p->relies);
			} else {
				printf("    %s\n", p->what);
				pr_lines(p->text.p, "      ");
				printf("    proof: %s\n", p->proof);
				printf("    cost under %s (a loop priced per iteration times its "
				    "trips): %g -> %g per call\n", model, p->c0, p->c1);
				printf("    relies on: %s\n", p->relies);
			}
			rep_obj(NULL);
			rep_str("kind", kind_name(p->kind));
			rep_str("function", G.f[f].name);
			rep_int("line", p->line);
			rep_str("what", p->what);
			rep_str("replacement", p->text.p ? p->text.p : "");
			rep_str("proof", p->proof);
			rep_int("points", p->points);
			rep_num("cost_before", p->c0);
			rep_num("cost_after", p->c1);
			rep_str("domain", p->domain);
			rep_str("relies_on", p->relies);
			if (p->kind == P_LOOP) {
				int q;
				rep_int("passes", p->passes);
				rep_bool("second_pass_acts", p->acts);
				rep_list("exit_lines");
				for (q = 0; q < p->nexit; q++)
					rep_int(NULL, p->exitl[q]);
				rep_end();
			}
			if (p->kind == P_CALLS) {
				int q;
				rep_list("call_lines");	/* the first 16 */
				for (q = 0; q < p->nexit; q++)
					rep_int(NULL, p->exitl[q]);
				rep_end();
			}
			rep_end();
			nshown++;
		}
	}
	/* calls a loop never changes */
	for (i = 0; i < npr; i++) {
		struct prop *p = &PR[i];
		int k;
		if (p->kind != P_INV)
			continue;
		if (p->callee >= 0 && lic_ok(p->callee, 0)) {
			/* a driver moves it itself: counted, not advised */
			for (k = 0; k < nmoved && moved[k] != p->callee; k++)
				;
			if (k == nmoved && nmoved < 64) {
				moved[nmoved] = p->callee;
				nmv[nmoved++] = 0;
			}
			if (k < 64)
				nmv[k]++;
			rep_obj(NULL);
			rep_str("kind", "invariant_call");
			rep_str("function", G.f[p->f].name);
			rep_int("line", p->line);
			rep_str("what", p->what);
			rep_str("proof", "dataflow: the read and write sets");
			rep_num("cost_per_call", p->c0);
			rep_num("passes", p->c1);
			rep_bool("driver_moves_it", 1);
			rep_end();
			continue;
		}
		if (p->c0 < 8) {
			char nm[160];
			const char *at = strstr(p->what, " at line");
			snprintf(nm, sizeof nm, "%.*s", at ? (int)(at - p->what) : 40,
			    p->what);
			sb_printf(&cheap, "%s%s (line %d)", ncheap ? ", " : "", nm, p->line);
			ncheap++;
			continue;
		}
		if (!said_inv)
			printf("  invariant calls (a dataflow fact: neither the "
			    "arguments nor anything the callee reads is written in the "
			    "loop, and the callee writes nothing):\n");
		said_inv = 1;
		printf("    %s; it costs %g a call under %s", p->what, p->c0, model);
		if (p->c1 > 1)
			printf(", %g passes: %g saved a run of the loop\n", p->c1,
			    p->c0 * (p->c1 - 1));
		else
			printf("\n");
		rep_obj(NULL);
		rep_str("kind", "invariant_call");
		rep_str("function", G.f[p->f].name);
		rep_int("line", p->line);
		rep_str("what", p->what);
		rep_str("proof", "dataflow: the read and write sets");
		rep_num("cost_per_call", p->c0);
		rep_num("passes", p->c1);
		rep_end();
		nshown++;
	}
	if (ncheap)
		printf("  also invariant in their loops, but cheap (cost < 8 a call): "
		    "%s\n", cheap.p);
	if (nmoved) {
		int k, tot = 0;
		for (k = 0; k < nmoved; k++)
			tot += nmv[k];
		printf("  also the same on every pass of their loops, but %s no loop "
		    "and read%s no memory, so a driver's own loop-invariant code "
		    "motion moves %s out: ", tot > 1 ? "they run" : "it runs",
		    tot > 1 ? "" : "s", tot > 1 ? "them" : "it");
		for (k = 0; k < nmoved; k++)
			printf("%s%s (%d call%s)", k ? ", " : "", G.f[moved[k]].name,
			    nmv[k], nmv[k] > 1 ? "s" : "");
		printf("\n");
		nshown++;
	}
	rep_end();
	if (DOTD.n) {
		printf("  dot products: %d sum%s of products would be %s under %s "
		    "(cost -%g, each once), not proposed: the module does not "
		    "declare SPV_KHR_integer_dot_product (in GLSL, #extension "
		    "GL_EXT_integer_dot_product)\n", DOTD.n, DOTD.n == 1 ? "" : "s",
		    DOTD.n == 1 ? "a dot product" : "dot products", model, DOTD.gain);
		rep_obj("dot_products_not_declared");
		rep_int("sums", DOTD.n);
		rep_num("gain", DOTD.gain);
		rep_str("needs", "SPV_KHR_integer_dot_product");
		rep_end();
	} else if (!nshown)
		printf("  nothing found at the function level\n");
	free(cheap.p);
	(void)nprop;
}

static void
reset_all(void)
{
	int i, f;
	for (i = 0; i < npr; i++)
		free(PR[i].text.p);
	free(PR);
	PR = NULL;
	npr = cappr = 0;
	for (i = 0; i < nrc; i++) {
		int b;
		for (b = 0; b < RC[i].cap && RC[i].site; b++)
			free(RC[i].site[b].s);
		free(RC[i].box);
		free(RC[i].site);
	}
	free(RC);
	RC = NULL;
	nrc = caprc = 0;
	free(recn);
	recn = NULL;
	if (FS)
		for (f = 0; f < G.nf; f++) {
			for (i = 0; i < FS[f].n; i++) {
				free(FS[f].s[i].kv);
				free(FS[f].s[i].entry);
				es_free(&FS[f].s[i].exit);
			}
			free(FS[f].s);
		}
	free(FS);
	FS = NULL;
	free(xrep);
	free(srep);
	xrep = srep = NULL;
	capxrep = capsrep = 0;
	free(fmemo);
	fmemo = NULL;
	free(xfn);
	xfn = NULL;
	lp_free();
	lp_note = 0;
	loops_only = 0;
}

/* simplify's function-level pass over src (OpSource's GLSL); only: the
 * report alone (the caller ran nothing else); dotok: the module declares
 * SPV_KHR_integer_dot_product */
void glslfn_pass(const char *src, double budget, int only, const char *model,
    int dotok);
void
glslfn_pass(const char *src, double budget, int only, const char *model,
    int dotok)
{
	clock_t t0 = clock();
	int rounds, f, i;
	double secs;
	reset_all();
	memset(&G, 0, sizeof G);
	memset(&A, 0, sizeof A);
	A.site = -1;
	G.cur = -1;
	dot_hinted = 0;
	fsrc = src;
	memset(&DOTD, 0, sizeof DOTD);
	DOTD.ok = dotok;
	if (parse_source(src, model && !strcmp(model, "Fragment") ?
	    SNX_GLSLC_FRAGMENT : model && !strcmp(model, "Vertex") ?
	    SNX_GLSLC_VERTEX : SNX_GLSLC_COMPUTE) != 0) {
		printf("function level: not run (%s)\n", G.why);
		return;
	}
	prepare();
	{
		/* a snippet's functions called only from the completed main
		 * (glslc.c, after #line 10000) have no call sites of their own */
		int k, user = 0;
		for (k = 0; k < ncall_edge; k++)
			if (G.f[callee_edge[2 * k]].line < 10000 &&
			    G.f[callee_edge[2 * k + 1]].line < 10000)
				user = 1;
		/* no domain from a call, but a loop's passes and what it
		 * computes are read all the same */
		if (!user && !has_loops()) {
			if (only)
				printf("function level: no function of the source calls "
				    "another, so no call gives a domain: nothing to do\n");
			return;
		}
		loops_only = !user;
	}
	patch_room();
	mark_all();
	xfn_n = G.nx;
	xfn = xmalloc(((size_t)G.nx + 1) * sizeof *xfn);
	for (i = 0; i <= G.nx; i++)
		xfn[i] = -1;
	for (f = 0; f < G.nf; f++) {
		int *xs = xmalloc(((size_t)G.nx + 1) * sizeof *xs), nx = 0;
		reach_s(G.f[f].body, xs, &nx, NULL, NULL, G.nx);
		for (i = 0; i < nx; i++)
			if (xs[i] < xfn_n)
				xfn[xs[i]] = f;
		free(xs);
	}
	rounds = analyse((double)t0 / CLOCKS_PER_SEC + budget);
	secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
	if (A.out_of_time)
		lp_note = 1;
	else
		probe_round(rounds < 8);
	for (f = 0; f < G.nf && loops_only; f++) {
		search_loops(f);
		search_tab(f);
	}
	if (loops_only) {
		if (npr)
			report(rounds, secs);
		else if (lp_note)
			printf("function level: no function of the source calls "
			    "another, and its loops were not looked at (%s)\n", lp_note ==
			    1 ? "the time budget ran out first: `budget N` for more" :
			    lp_note == 2 ? "the analysis did not settle in 8 rounds" :
			    "running their passes grew a summary of the analysis");
		else if (only)
			printf("function level: no function of the source calls "
			    "another, so no call gives a domain; none of its loops leaves "
			    "early or computes a map of one small value\n");
		return;
	}
	for (f = 0; f < G.nf; f++) {
		struct dom D;
		int whole = 0;
		search_loops(f);
		search_tab(f);
		if (whole_ok(f) && fn_dom(f, &D)) {
			whole = D.points <= (double)(1 << 20);
			free(D.rows);
		}
		if (whole)
			search_whole(f);
		else
			search_expr(f);
		search_calls(f);
		search_dot(f);
		search_tex(f);
	}
	search_invariant();
	report(rounds, secs);
	(void)only;
}
