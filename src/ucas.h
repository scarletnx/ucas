/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ucas - a programmer's computer algebra system.
 *
 * Data layout: every hot table is a structure of arrays.
 *   tokens   kind[] fl[] off[] len[]
 *   nodes    op[] fl[] ty[] a[] b[] c[] ki[] kr[]  (hash-consed DAG; an id is
 *            always larger than the ids of its operands, so id order is a
 *            topological order and evaluation is one forward sweep)
 *   plans    a compacted copy of the nodes one function needs, plus value
 *            columns: CH points per node, evaluated column by column.
 */
#ifndef UCAS_H
#define UCAS_H

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MAXP 8           /* parameters per function */
#define CH 256           /* points per evaluation chunk */
#define NOTOK UINT32_MAX

/* ---- util.c ---- */
void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t sz);
void *xrealloc(void *p, size_t n);
uint32_t newcap(uint32_t cap, uint32_t need);

struct sbuf {
	char *p;
	size_t n, cap;
};
void sb_putc(struct sbuf *b, int c);
void sb_puts(struct sbuf *b, const char *s);
void sb_printf(struct sbuf *b, const char *fmt, ...);
void sb_clear(struct sbuf *b);
const char *sb_str(struct sbuf *b);

uint32_t sym_intern(const char *s, size_t n);
uint32_t sym_cstr(const char *s);
const char *sym_str(uint32_t id);

struct toks;
extern jmp_buf fail_jmp;
extern uint32_t diag_tok;
extern struct toks *diag_toks;
extern const char *diag_file;
void fail(const char *fmt, ...);
void fail_quiet(void);    /* end the statement: a question was asked */
void note(const char *fmt, ...);
extern int note_mute;      /* > 0: note() does nothing */
void notes_flush(void);
void notes_drop(void);
int ilog2u(uint64_t v);       /* floor(log2(v)), -1 for 0 */
int bits_for(uint64_t v);     /* bits needed to hold v (0 -> 1) */

/* ---- lex.c ---- */
enum {
	T_EOF, T_NL, T_NUM, T_ID, T_LP, T_RP, T_LB, T_RB, T_LC, T_RC,
	T_COMMA, T_COLON, T_QUES, T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
	T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PCT, T_CARET, T_AMP, T_BAR,
	T_TILDE, T_BANG, T_SHL, T_SHR, T_ANDAND, T_OROR, T_ARROW, T_DOTDOT,
	T_XOR, T_SQRT, T_SUP, T_NTOK
};
#define TF_SPACE 1 /* whitespace before the token */

struct toks {
	uint8_t *kind;
	uint8_t *fl;
	uint32_t *off;
	uint32_t *len;
	uint32_t n, cap;
	const char *src;
};
void lex(struct toks *t, const char *src);
void toks_free(struct toks *t);
const char *tok_name(int kind);
int tok_is(const struct toks *t, uint32_t i, const char *word);

/* ---- types ---- */
typedef uint32_t nid;
typedef uint16_t ty_t;
/*
 * K_REAL is the mathematical real line (evaluated in double, domain errors
 * are undefined).  K_FLOAT is an IEEE binary16/32/64 machine type: every result
 * is rounded, x/0 is inf, sqrt(-1) is NaN, and those are values, not errors.
 */
enum { K_LIT = 1, K_BOOL, K_UINT, K_SINT, K_INT, K_REAL, K_FLOAT };
#define TY(k, w) ((ty_t)(((k) << 8) | (w)))
#define TYK(t) ((t) >> 8)
#define TYW(t) ((t)&0xff)
#define TY_LIT TY(K_LIT, 64)
#define TY_BOOL TY(K_BOOL, 1)
#define TY_INT TY(K_INT, 64)
#define TY_REAL TY(K_REAL, 64)
#define TY_F16 TY(K_FLOAT, 16)
#define TY_F32 TY(K_FLOAT, 32)
#define TY_F64 TY(K_FLOAT, 64)
#define TY_FP(t) (TYK(t) == K_REAL || TYK(t) == K_FLOAT)
#define TY_ISINT(t) (!TY_FP(t))
#define TY_SIZED(t) (TYK(t) == K_UINT || TYK(t) == K_SINT || TYK(t) == K_BOOL)
const char *ty_str(ty_t t);   /* static buffer, rotated */
int ty_parse(const char *s, size_t n, ty_t *out);

/* ---- ops ---- */
enum {
	O_NONE, O_CONST, O_VAR, O_UNDEF,
	O_NEG, O_NOT, O_LNOT, O_ABS, O_SIGN, O_SQRT, O_ISQRT, O_EXP, O_LOG,
	O_SIN, O_COS, O_TAN, O_FLOOR, O_CEIL, O_CLZ, O_CTZ, O_POPCNT, O_CAST,
	O_ADD, O_SUB, O_MUL, O_DIV, O_MOD, O_POW, O_AND, O_OR, O_XOR,
	O_SHL, O_SHR, O_MIN, O_MAX,
	O_EQ, O_NE, O_LT, O_LE, O_GT, O_GE, O_LAND, O_LOR,
	O_SEL, O_TABLE, O_INDEX,
	O_ASUINT, O_ASINT, O_ASFLOAT,  /* bit casts float <-> int */
	O_BARRIER,   /* barrier(e): e, after a workgroup barrier */
	O_ATOMIC,    /* atomic_op(x, v): old value x; ki = AT_* | scope */
	/* approx: more functions, sign transfer and fused multiply-add */
	O_SEC, O_CSC, O_COT, O_ATAN, O_COPYSIGN, O_FMA,
	O_LOOP,      /* calc: loop over an entry (calc.h); a = lo, b = hi */
	O_LAMBERTW, O_LAMBERTWM1,  /* Lambert W: principal and -1 branches */
	/* forms.c: inverse trig, atan2(y, x), hyperbolic and their inverses */
	O_ASIN, O_ACOS, O_ATAN2, O_SINH, O_COSH, O_TANH, O_ASINH, O_ACOSH,
	O_ATANH,
	/* hyper.c: conj(z), re(z), im(z), arg(z); on reals x, x, 0, 0 or pi */
	O_CONJ, O_RE, O_IM, O_ARG,
	/*
	 * dot.c: packed integer dot products (SPV_KHR_integer_dot_product):
	 * 4 lanes of 8 bits, 2 of 16, 8 of 4, packed in a u32; unsigned,
	 * signed, and signed x unsigned lanes; the *SAT forms take an
	 * accumulator and saturate (see dot.c for the exact semantics)
	 */
	O_UDOT4X8, O_SDOT4X8, O_SUDOT4X8, O_UDOT2X16, O_SDOT2X16, O_SUDOT2X16,
	O_UDOT8X4, O_SDOT8X4, O_SUDOT8X4,
	O_UDOT4X8S, O_SDOT4X8S, O_SUDOT4X8S, O_UDOT2X16S, O_SDOT2X16S,
	O_SUDOT2X16S, O_UDOT8X4S, O_SDOT8X4S, O_SUDOT8X4S,
	/*
	 * sg.c: operations on the lanes of a subgroup of W invocations.  The
	 * operand is a value per lane, a function of the lane's index (the
	 * bound variable `lane`) and of values the lanes share; the last
	 * operand is W.  ballot(p): bit l is p in lane l; any, all; add: the
	 * sum over the lanes; shuffle(v, i): v in lane i; scan(v, l): the
	 * sum of v over lanes 0..l (see sg.c for the exact semantics)
	 */
	O_SGBALLOT, O_SGANY, O_SGALL, O_SGADD, O_SGSHUF, O_SGSCAN,
	/*
	 * inst.c: single instructions that targets have, each defined by an
	 * expression over the operations above (bit fields, bit reversal,
	 * the high half of a product, funnel shifts, the median of three,
	 * a sum of byte differences, a byte permute, a bit field insert,
	 * the 64-bit product of two words and its multiply-add)
	 */
	O_BFE, O_SBFE, O_BREV, O_MULHI, O_SMULHI, O_FUNL, O_FUNR, O_MED3,
	O_USAD4X8, O_BYTEPERM, O_BFI, O_MULWIDE, O_MADWIDE,
	/*
	 * texalu.c: the texture unit's fixed function as operations.  A
	 * nearest read with an address mode on an integer texel index
	 * (fetch_repeat(T, i) ...), and a linear filter on texel coordinates
	 * with its weights on the sub-texel grid (tex_linear8(T, x),
	 * tex_bilinear8(T, W, x, y)); ki holds the grid's bits and the
	 * table's width (see texalu.h for the exact semantics)
	 */
	O_FETCH_REPEAT, O_FETCH_MIRROR, O_FETCH_CLAMP, O_FETCH_MCLAMP,
	O_FETCH_BORDER, O_TEX_LINEAR, O_TEX_BILINEAR,
	O_NOPS
};
#define O_TEX_FIRST O_FETCH_REPEAT
#define O_TEX_LAST O_TEX_BILINEAR
#define IS_TEX(op) ((op) >= O_TEX_FIRST && (op) <= O_TEX_LAST)
#define IS_FETCH(op) ((op) >= O_FETCH_REPEAT && (op) <= O_FETCH_BORDER)
#define O_DOT_FIRST O_UDOT4X8
#define O_DOT_LAST O_SUDOT8X4S
#define IS_DOT(op) ((op) >= O_DOT_FIRST && (op) <= O_DOT_LAST)
#define O_SG_FIRST O_SGBALLOT
#define O_SG_LAST O_SGSCAN
#define IS_SG(op) ((op) >= O_SG_FIRST && (op) <= O_SG_LAST)
#define IS_INST(op) ((op) >= O_BFE && (op) <= O_MADWIDE)
/* built from all three operands by mk_op (the generic rebuilders) */
#define IS_OP3(op) (IS_DOT(op) || IS_SG(op) || IS_INST(op))
#define OF_COMM 1
#define OF_INT 2      /* integer operands only */
#define OF_REALFN 4   /* operand promoted to real */
#define OF_CMP 8
#define OF_PARTIAL 16 /* undefined on part of its domain */
enum { AT_ADD, AT_SUB, AT_MIN, AT_MAX, AT_AND, AT_OR, AT_XOR, AT_XCHG };
#define AT_SHARED 16   /* workgroup scope; otherwise device (global) */
#define AT_SERIAL 32   /* unique id bits start here: atomics never merge */
struct opinfo {
	const char *name;
	uint8_t arity;
	uint8_t fl;
};
extern const struct opinfo opinfo[O_NOPS];

/* ---- node.c ---- */
#define NF_PARTIAL 1 /* this subterm can be undefined */
#define NF_VAR 2     /* depends on a parameter */
#define NF_SG 4      /* has an operation that sg_lower spells out: a
		      * subgroup operation (sg.c) or an instruction (inst.c) */
#define NF_LANE 8    /* the lane of a subgroup operation's operand is free
		      * in it: part of a value per lane */
#define NF_LANEREAD 16 /* reads a parameter's value in another lane, which
			* has no exact value here (sg_lane_copy) */
#define NF_INH (NF_VAR | NF_SG | NF_LANE | NF_LANEREAD)   /* an operand's,
							  * its user's */
struct nodes {
	uint8_t *op;
	uint8_t *fl;
	ty_t *ty;
	nid *a, *b, *c;  /* operands; O_TABLE: a = list start, b = count */
	int64_t *ki;     /* O_CONST int value, O_VAR slot */
	double *kr;      /* O_CONST real value */
	uint32_t n, cap;
	uint32_t *ht;
	uint32_t htcap, htn;
	nid *lst;
	uint32_t nlst, lstcap;
};
extern struct nodes N;
void nodes_init(void);
int64_t int_norm(ty_t t, int64_t v);
nid mk_const_i(ty_t t, int64_t v);
nid mk_const_r(double v);
nid mk_var(uint32_t slot, uint32_t name, ty_t t);
nid mk_undef(ty_t t);
nid mk_un(int op, nid a);
nid mk_bin(int op, nid a, nid b);
nid mk_sel(nid c, nid a, nid b);
nid mk_cast(ty_t t, nid a);
nid mk_bool(nid a);
nid mk_table(const nid *e, uint32_t n);
nid mk_index(nid tab, nid i);
nid mk_barrier(nid a);
nid mk_atomic(int kind, nid a, nid v);
nid mk_fma(nid a, nid b, nid c);
nid mk_dot(int op, nid a, nid b, nid acc);  /* dot.c; acc 0 unless _SAT */
nid mk_op(int op, nid a, nid b, nid c);     /* any op from its operands */
nid mk_sg(int op, nid a, nid b, nid c);     /* sg.c; the last operand is W */
nid mk_inst(int op, nid a, nid b, nid c);   /* inst.c */
/* bfi's field (offset p, count n) as its one constant operand */
nid bfi_field(int p, int n);
#define BFI_P(k) ((int)((k) & 63))
#define BFI_N(k) ((int)((k) >> 6))
/* texalu.c's operations; ki as the node keeps it (TEX_KI) */
nid mk_tex(int op, int64_t ki, nid a, nid b, nid c);
/* a node like n on other operands (the generic rebuilders) */
nid mk_like(nid n, nid a, nid b, nid c);
double fma32(double a, double b, double c);	/* f32 fma, correctly rounded */
double fma16(double a, double b, double c);	/* f16 fma, correctly rounded */
double fma64(double a, double b, double c);	/* f64 fma, correctly rounded */
int atomic_lookup(const char *s, size_t n, int *kind);
nid subst(nid root, const nid *map, int nmap);
int is_const(nid n);
int const_is(nid n, double v);

enum { PM_UCAS, PM_C, PM_CBITS };
struct shmap;
struct pctx {
	int mode;
	const char *const *vn;  /* optional names per param slot */
	const struct shmap *sh; /* shared subterms printed by name */
	nid self;               /* the binding being defined (not by name) */
};
void print_node(struct sbuf *b, nid n, const struct pctx *pc);
const char *node_str(nid n);   /* static buffer, rotated; ucas syntax */
void print_paren(struct sbuf *b, nid n);
/*
 * Shared subterms named: a subterm used more than once and larger than a
 * few nodes is printed once as a binding, `e where t1 = ..., t2 = ...`
 * (a name from the source's let or where when there is one), and the
 * whole is capped at cap bytes with "… (N more characters)".
 */
#define EXPR_CAP 2000
#define NOTE_CAP 600
void print_shared(struct sbuf *b, nid root, size_t cap);
const char *shared_str(nid root, size_t cap);   /* static, rotated */
void name_hint(nid n, uint32_t sym);           /* the source named n */
struct shmap *shmap_new(nid root);   /* names for root's shared subterms */
void shmap_free(struct shmap *m);
int shmap_count(const struct shmap *m);
void print_with(struct sbuf *b, nid e, const struct shmap *m, size_t cap);
void shmap_bindings(struct sbuf *b, const struct shmap *m, const char *sep,
    size_t cap);
void cap_text(struct sbuf *b, size_t from, size_t cap);

/* ---- functions ---- */
struct param {
	uint32_t name;
	ty_t ty;
	int ranged;
	int uniform;    /* same value across a warp: branches on it do not diverge */
	int64_t lo, hi;
	double rlo, rhi;
};
struct func {
	uint32_t name;
	int np;
	struct param p[MAXP];
	nid body, where;
	int anon;
	/*
	 * hyper.c: a complex/quaternion/octonion function.  halg is log2 of
	 * the algebra's dimension (0: an ordinary real function), hk[j] that
	 * of parameter j; hf32: c64, every operation rounded to f32 parts.
	 */
	uint8_t halg, hf32, hk[MAXP];
};
struct func *func_find(uint32_t name);
struct func *func_new(uint32_t name);
extern struct func *funcs;
extern int nfuncs;

/* ---- basis (gates.c) ---- */
#define MAXGATE 24
struct gate {
	uint32_t name;
	uint8_t arity, tt, cost, symm; /* symm bit (i*3+j): args i,j interchangeable */
	uint8_t lut;    /* "= lut": any function of its inputs, the table per use
			 * (vpternlog, LOP3) */
	nid body;
	uint32_t pname[3];
};
struct basis {
	uint32_t name;
	int ng;
	struct gate g[MAXGATE];
};
struct basis *basis_find(uint32_t name);
struct basis *basis_new(uint32_t name);
struct basis *basis_nth(int i);
void basis_finish_gate(struct basis *b, struct gate *g);

/* ---- model.c: cost models ---- */
enum {
	C_IADD, C_LOGIC, C_SHIFT, C_IMUL, C_IDIV, C_IDIVK, C_CMP, C_SELECT,
	C_MINMAX, C_BITCNT, C_CONV, C_BITCAST,
	C_FADD, C_FMUL, C_FDIV, C_FSQRT, C_FTRANS, C_FPOW, C_F64,
	C_LOAD, C_BRANCH, C_DIVERGE, C_BARRIER, C_ATOM_SH, C_ATOM_GL,
	C_STORE,
	/* signed division and the remainders, each by a variable and by a
	 * constant (the K class is always the one after); a model that does
	 * not give them prices them as idiv/idivk */
	C_SDIV, C_SDIVK, C_IREM, C_IREMK, C_SREM, C_SREMK,
	C_NBASE,
	/*
	 * Dot products, one class per form (u, s, su; 4x8, 2x16, 8x4; plain
	 * and accumulate-saturate).  Unlike the classes above, a model that
	 * does not list one cannot use it: it costs infinity (unavailable).
	 */
	C_DOT_FIRST = C_NBASE,
	C_DOT_LAST = C_DOT_FIRST + 17,
	/*
	 * Memory kinds (memk.c): unset in a model that does not list them
	 * (NaN), and priced through model_cost() as their fallback then.
	 */
	C_MEM_FIRST,
	C_LOAD_SHARED = C_MEM_FIRST, C_STORE_SHARED, C_LOAD_UBO, C_LOAD_UBO_DIV,
	C_LOAD_STREAM, C_STORE_STREAM, C_BANK_CONFLICT,
	C_ATOM_SH_SAME, C_ATOM_SH_F, C_ATOM_GL_F, C_ATOM_GL_64, C_ATOM_GL_CAS,
	C_ATOM_GL_STREAM, C_TEX_FETCH, C_TEX_SAMPLE, C_TEX_SAMPLE_F32,
	C_IMAGE_LOAD,
	/* a nearest sampled read with an address mode (texalu.c) */
	C_TEX_NEAREST,
	C_MEM_LAST = C_TEX_NEAREST,
	/*
	 * Subgroup operations (sg.c), unavailable unless listed, as the dot
	 * forms: the ballot, the votes (any, all), the sum over the lanes,
	 * the shuffle (and broadcast, shuffle by xor, rotate) and the
	 * inclusive scan.
	 */
	C_SG_BALLOT, C_SG_VOTE, C_SG_ADD, C_SG_SHUF, C_SG_SCAN,
	/*
	 * A multiply of two 16-bit operands into 32 bits (XMAD on the
	 * NVIDIA parts whose 32-bit multiply is a chain of them): the price
	 * a model's fusions over inputs of a proved width refer to.  As the
	 * dot forms, unavailable unless the model lists it.
	 */
	C_MUL16,
	/*
	 * rop.c: graphics fixed function doing a scatter's update, a
	 * fragment each: the blend equation (and min, max), the advanced
	 * blend operations, logic ops, stencil ops.  Unavailable unless
	 * the model lists them (nothing is priced before it is measured).
	 */
	C_ROP_BLEND, C_ROP_ADV, C_ROP_LOGIC, C_ROP_STENCIL,
	/* the catalogue's instructions (inst.c), absent unless listed */
	C_BFE, C_BREV, C_MULHI, C_FUNNEL, C_MED3, C_SAD4X8, C_BYTEPERM, C_BFI,
	C_MULWIDE,
	C_NCLASS
};
#define C_SG_FIRST C_SG_BALLOT
#define C_SG_LAST C_SG_SCAN
/* an instruction the target may not have: absent unless the model lists it */
#define C_ABSENT(k) (((k) >= C_DOT_FIRST && (k) <= C_DOT_LAST) || \
    (k) >= C_SG_FIRST)	/* the subgroups, and what comes after them */
/* the class of a dot op; IS_DOT(op) */
#define DOT_CLASS(op) (C_DOT_FIRST + ((op) - O_DOT_FIRST))
#define C_DIVK(k) ((k) + 1)	/* the division class by a constant */
int div_class(nid n);		/* model.c: an integer / or %'s class */
#define MAXFOLD 4
/*
 * A model's parameters that are not prices (0: the model does not say).
 * memk.c: banks, bank_bytes, warp.  texalu.c: what the texture unit does
 * to a coordinate and a filtered result,
 *   subtexel    bits of a texel the address unit resolves (Vulkan's
 *               subTexelPrecisionBits): coordinates and filter weights
 *               snap to a grid of 2^-subtexel
 *   tex_within  how far, in ulps of the result, a filtered read may be
 *               from the exact weighted sum of its texels
 */
enum { MP_BANKS, MP_BANK_BYTES, MP_WARP, MP_SUBTEXEL, MP_TEX_WITHIN,
	MP_NPAR };
struct model {
	uint32_t name;
	float c[C_NCLASS];
	char desc[160];
	/* float round trips T -> M -> T the driver drops (fpfold.c) */
	ty_t fold[MAXFOLD][2];
	int nfold;
	float par[MP_NPAR];	/* 0: the model does not say */
};
extern const char *const cclass_name[C_NCLASS];
struct model *model_find(uint32_t name);
int cclass_available(const struct model *m, int k);
void model_classes_text(struct sbuf *b, const struct model *m);
struct model *model_new(uint32_t name);
struct model *model_active(void);
struct model *model_nth(int i);
void model_set_active(struct model *m);
int cclass_find(const char *s, size_t n);
extern const char *const mparam_name[MP_NPAR];
float node_cost(const struct model *m, nid n, int *cls);
void cmd_cost(struct func *f, struct model *m);
void cmd_models(void);

/* ---- rop.c: graphics fixed function as compute ---- */
struct P;
int rop_command(struct P *p);
/* simplify's question: the cheapest state that computes an atomic's update
 * F(s, d) and that the model prices below the atomic (rop.c) */
struct rop_offer {
	int found, cls;
	float cost, against;
	char family[24], state[320], holds[400], proof[700], pipeline[700];
	char againstwhy[160];
	char near[400];		/* none found: what would compute it, or why not */
	int bound;		/* near: exact only while the values are bounded */
};
int rop_offer(struct func *f, struct model *m, struct rop_offer *o);

/* ---- dot.c: packed integer dot products ---- */
struct dotform {
	const char *name;    /* "udot4x8", "sdot2x16sat", ... */
	uint8_t lanes, bits; /* 4 x 8, 2 x 16, 8 x 4 */
	uint8_t sa, sb;      /* lanes of a, b signed */
	uint8_t sat;         /* accumulates, saturating */
};
const struct dotform *dot_form(int op);
int dot_op(int lanes, int bits, int sa, int sb, int sat);  /* 0: none */
int64_t dot_scalar(int op, int ideal, int64_t a, int64_t b, int64_t acc,
    uint8_t *fl);
/* bounds of a dot from its operands' (a, b as u32 values; acc) */
int dot_ival(int op, int64_t alo, int64_t ahi, int64_t blo, int64_t bhi,
    int64_t clo, int64_t chi, int64_t *lo, int64_t *hi);
/* find: sums of products rewritten into the active model's dot forms */
void dot_find(const struct func *f);
enum { DOT_NONE, DOT_PROVED, DOT_UNPROVED, DOT_WRONG, DOT_UNAVAILABLE };
struct dotres {
	int status;          /* DOT_* */
	nid body;            /* the rewritten function body */
	float before, after; /* its cost under the active model */
	struct sbuf desc;    /* the lanes and what they rely on */
	struct sbuf proof;   /* the proof (or the counterexample) */
	const char *kind;    /* "exhaustive", "sat", "parts" */
};
/* fmask: 1 the 4 x 8 forms, 2 the 2 x 16, 4 the 8 x 4 */
int dot_search(const struct func *f, int fmask, struct dotres *r);
void dotres_free(struct dotres *r);

/* ---- sg.c: operations on the lanes of a subgroup ---- */
/* the bound variable `lane`: above the 64 slots of parameters, loops and
 * closed forms (calc.h), so no slot mask or loop renumbering sees it;
 * NF_LANE says where it is free */
#define SG_LANE_SLOT 100
#define SG_MAXW 64
nid sg_lane(void);		/* the lane's index inside an operand */
/* parameter slot's value in the lane an operand is read in, for a
 * parameter that is not uniform: no exact value, priced at the worst */
#define SG_COPY_SLOT 300
nid sg_lane_copy(int slot, uint32_t name, ty_t t, int W);	/* node.c */
/* a loop state's (slot) value in the lane it is read in, for a state that
 * follows a value per lane (from: its name): priced as a parameter's */
nid sg_state_copy(int slot, uint32_t name, ty_t t, int W, uint32_t from);
int sg_is_lane(nid n);		/* a variable named lane: the lane itself */
int sg_has_lane(nid e);		/* the bound lane is free in e */
nid sg_lower_once(nid root);	/* node.c: sg_lower without its memory */
nid sg_flat(nid e, nid to);	/* node.c: the bound lane replaced by to */
/* f's body as one lane runs it: what cost and find price */
nid sg_priced(const struct func *f, nid e);
int sg_class(int op);		/* its cost class */
int sg_width(nid n);		/* W of a subgroup node */
/* the operation spelled out over its W lanes, operands already lowered */
nid sg_expand(int op, nid a, nid b, nid c);
/* lane l (u32) is active in mask: ((mask >> l) & 1) != 0, kept so that
 * find can tell a function under sg_active */
nid sg_active_bit(nid mask, nid l);
nid sg_active_of(nid bit, nid *lane);	/* such a test's mask (lane), or 0 */
/* a parameter of mask that a subgroup operation of e reads per lane */
int sg_reads(nid e, uint64_t mask);
/* v sorted along W = 2^m lanes by the bitonic network, in the lane `here`
 * runs in, or read at lane `at` (0: here's own); 0 when W is not a power
 * of two */
nid sg_sorted(nid v, nid here, nid at, int W);
/* e with every subgroup operation spelled out (e itself when it has none) */
nid sg_lower(nid e);
/* find: forms the active model's subgroup operations make cheaper */
void sg_find(const struct func *f);

/* ---- inst.c: the catalogue of single instructions ---- */
int inst_class(int op);
/* the instruction as the expression that defines it */
nid inst_expand(int op, nid a, nid b, nid c);
/* find: the instructions the active model has, for the patterns they are */
void inst_find(const struct func *f);

/* ---- tab.c ---- */
/* find: a function of at most 16 input bits as one load of a table */
void tab_find(const struct func *f);

/* ---- eval.c ---- */
#define F_UNDEF 1  /* outside the operation's domain */
#define F_UW 2     /* the result wrapped as an unsigned value */
#define F_SW 4     /* the result wrapped as a signed value */
#define F_UB 8     /* undefined behaviour in C */
#define F_TRUNC 16 /* conversion changed the value */
#define F_OVF 32   /* left the 64-bit range of the ideal evaluator */
#define F_NAN 64   /* IEEE: produced a NaN from non-NaN operands */
#define F_INF 128  /* IEEE: produced an infinity from finite operands */
struct ity {
	uint8_t k, w;
	uint64_t mask;
	int64_t smin, smax;
};
void ity_of(struct ity *it, ty_t t, int ideal);
int64_t scalar_int(int op, const struct ity *t, const struct ity *ta,
    int64_t a, int64_t b, int64_t c, uint8_t *fl);
double scalar_real(int op, double a, double b, uint8_t *fl);
double scalar_float(int op, int w, double a, double b, uint8_t *fl);
double lambert_w(double x, int branch);   /* NaN outside the domain */
double fround(int w, double v);
int fdigits(int w);                 /* digits that print a float exactly */
int64_t fkey(int w, double v);      /* IEEE value -> ordered integer key */
double fromkey(int w, int64_t k);
uint64_t fbits(int w, double v);
double frombits(int w, uint64_t b);

struct plan {
	uint32_t n, nroot;
	int ideal;
	uint8_t *op;
	ty_t *ty;
	struct ity *it;
	uint32_t *a, *b, *c;
	int64_t *ki;
	double *kr;
	nid *gid;
	uint32_t *lst;
	uint32_t *root;
	int64_t *vi;
	double *vr;
	uint8_t *vf;
};
void plan_build(struct plan *p, const nid *roots, uint32_t nroots, int ideal);
void plan_free(struct plan *p);
void plan_run(struct plan *p, uint32_t np, const int64_t *in_i, const double *in_r);
uint32_t plan_loc(const struct plan *p, nid g);
#define COLI(p, k) (&(p)->vi[(size_t)(k)*CH])
#define COLR(p, k) (&(p)->vr[(size_t)(k)*CH])
#define COLF(p, k) (&(p)->vf[(size_t)(k)*CH])

/* domain iteration */
struct diter {
	const struct func *f;
	int np;
	int64_t lo[MAXP], hi[MAXP];  /* integer bounds, or IEEE keys */
	double rlo[MAXP], rhi[MAXP];
	uint8_t real[MAXP];          /* 1 real grid; keys: 2 f32, 3 f64, 4 f16 */
	uint8_t narrowed[MAXP];      /* bounds tightened by the where clause */
	int exhaustive;
	uint64_t total, pos, limit;
	uint32_t rsteps;
	uint64_t rng;
};
void diter_init(struct diter *d, const struct func *f, uint64_t maxexh,
    uint32_t rsteps, uint64_t nsample);
uint32_t diter_next(struct diter *d, int64_t *in_i, double *in_r);
void param_bounds(const struct param *p, int64_t *lo, int64_t *hi);
int where_bounds(const struct func *f, int64_t *lo, int64_t *hi,
    uint8_t *narrowed);
void valid_mask(const struct plan *p, uint32_t wloc, uint32_t np, uint8_t *ok);
void fmt_point(struct sbuf *b, const struct func *f, const int64_t *in_i,
    const double *in_r, uint32_t i);
void fmt_val(struct sbuf *b, ty_t t, int64_t vi, double vr);

/* interval evaluation over a plan (sound bounds, ideal integers) */
struct ival {
	int64_t lo, hi;
	double rlo, rhi;
	uint8_t top;   /* no useful bound */
	uint8_t wrap;  /* the ideal interval leaves the machine type */
};
void plan_live(const struct plan *p, uint32_t np, uint8_t *lv);
void plan_ival(const struct plan *p, const struct func *f, struct ival *out);

/* ---- analyses ---- */
void cmd_show(struct func *f);
void cmd_table(struct func *f, int max);
void cmd_range(struct func *f);
extern int range_nocrit;     /* range F nocrit: skip critical points */
int crit_roots(struct func *f, double lo, double hi, double *out, int max);
void cmd_domain(struct func *f);
void cmd_wrap(struct func *f);
void cmd_crit(struct func *f);
void cmd_diff(struct func *f, int slot);
void cmd_equiv(struct func *f, struct func *g);
void cmd_gates(struct func *f, struct basis *b, int maxcost, int emit_c);
/* gates.c for programs: a minimal network for one output (cfront) */
struct gprog {
	int nstep;
	uint8_t g[64], a[64][4];	/* gate, arguments, a[3]: lut table */
	int out;			/* the value that is the output */
	int cost;
	uint64_t nodes;
};
int gates_synth(const struct basis *bs, int nin, uint64_t tt, uint64_t care,
    int maxcost, uint64_t budget, struct gprog *out);
int gates_lut_imm(int tt, int n);
/* find.c for programs: candidate forms of f's body (not yet proved),
 * searched up to maxcost or seconds; why[k] says what each is */
int find_cands(struct func *f, int maxcost, double seconds, nid *out,
    char (*why)[96], int max);
extern uint64_t gates_budget;
void cmd_find(struct func *f, int want, const uint8_t *ops, int maxcost);
void cmd_more(int want);
void cmd_fit(struct func *g, struct func *f, int metric);
extern int fit_quiet;
extern double fit_last_k[MAXP], fit_last_obj;
void cmd_error(struct func *g, struct func *f);
nid diff_node(nid e, uint32_t slot);
int equiv_check(const struct func *f, nid a, nid b, struct sbuf *witness,
    int *sampled);
/* equiv_decide: b refines a on f's domain, by enumeration or SAT */
enum { EQ_EXHAUSTIVE, EQ_SAT, EQ_SAMPLED };
struct nsat_res {          /* nodesat.c */
	int status;         /* 0 proved, 1 counterexample, -1 gave up, -2 unsupported */
	const char *why;    /* -2: what could not be bit-blasted */
	int nodes;          /* AIG nodes */
	long conflicts;
	int64_t in[MAXP];   /* 1: the counterexample's parameters */
};
struct eqproof {
	int how;              /* EQ_* */
	uint64_t points;      /* compared points */
	int sat_status;       /* nsat_equiv's status when it ran; -3 not reproduced */
	const char *sat_why;
	struct nsat_res sat;
};
int equiv_decide(const struct func *f, nid a, nid b, struct sbuf *witness,
    struct eqproof *pr);
void eqproof_text(struct sbuf *b, const struct eqproof *pr);

/* ---- parse.c ---- */
void run_source(const char *src, const char *file, int echo);
void ucas_prelude(void);	/* once: the prelude and the cost models */

/* ---- approx.c: function approximation; parse.c hooks it uses ---- */
struct P;
void cmd_approx(struct P *p);
void cmd_animate(struct P *p);   /* anim.c: animated GIF of a scan */
void cmd_plot(struct P *p, int mode);  /* plot.c: 0 plot, 1 plot3d, 2 neighbourhood */
struct func *cmd_func_text(const char *src);  /* parse.c, for plot.c */
/* anim.c: the exec animation's back end, for spirv.c */
enum { I_OP, I_BRANCH, I_RECONV, I_BARRIER, I_ATOMIC, I_MERGE };
void xr_begin(int lanes, const char *title);
int xr_add(int kind, const char *txt, int cls, double cost, uint32_t mask,
    const char *dst);
void xr_val(int k, int lane, const char *v);
void xr_lanein(int lane, const char *v);
int xr_count(void);
void xr_done(int fps, double seconds, const char *name, int nofile);
/* an image beside the trace (8-bit RGBA; cover: NULL, or which pixels
 * were written, the rest discarded); outputs fill in as the trace runs */
void xr_panel(const char *label, int w, int h, const unsigned char *rgba,
    const unsigned char *cover, int output);
/* the traced lanes' pixels (x, y, w, h) in the image run's (iw, ih) */
void xr_quad(int x, int y, int w, int h, int iw, int ih);
/* spirv.c: run SPIR-V assembly on the lanes */
struct spvset {			/* "animate exec %19 = 2.5": an input's value */
	char name[40];		/* without the % */
	int mode;		/* 1 lane index, 4 the value */
	double v;
};
extern struct spvset spv_set[16];
extern int nspv_set;
void spirv_simplify(const char *src, double budget, int nofile);
void spirv_assume(const char *name, int64_t lo, int64_t hi);
void spirv_assume_float(const char *name, double lo, double hi, int lo_open,
    int hi_open);
void spirv_float_opts(int mode, double ulp, int approx, int percomp,
    int only);
int spirv_sink_opt(const char *fmt);	/* NULL: none; 0: not a known sink */
int spirv_sink_known(const char *fmt);
void spirv_advice_opt(int on);	/* report what could leave the shader */
void spirv_fn_opts(int mode);	/* function level: 0 too, 1 alone, 2 not */
/* glslc.c: GLSL (a snippet is completed) to SPIR-V assembly */
int glsl_to_spvasm(const char *src, struct sbuf *out, struct sbuf *notes);
int glsl_like(const char *code, const char *lang);
/* codelang.c: a code block's language, from its tag and its content;
 * *unclear: 1 untagged and could be either, 2 a ```c block of GLSL */
enum { CL_NONE, CL_SPV, CL_GLSL, CL_C };
int code_lang(const char *code, const char *lang, int *unclear);
/* spvcross.c: spirv-cross's GLSL of a module before and after simplify */
const char *spvcross_tool(int *silent);
int spvcross_view(const char *orig, const char *simp, struct sbuf *report,
    struct sbuf *oglsl, struct sbuf *sglsl);
void spvcross_after_simplify(const char *orig, const char *simp, int nofile);
/* the bot's JSON path: the view goes to the bot as for its text path */
extern int spvcross_defer;
void text_udiff(const char *a, const char *b, const char *na,
    const char *nb, int maxlines, struct sbuf *out);
int spv_disassemble(const char *data, size_t n, struct sbuf *out);
int tool_run(char *const argv[], struct sbuf *out);
int tool_tmp(char *tmpl, const char *data, size_t n);
void deliver_file(const char *name, const char *type, const char *data,
    size_t n, int nofile);
void spirv_exec(const char *src, int lanes, int fps, double seconds,
    int random, uint64_t seed, const char *name, int nofile);
int cmdp_kw(struct P *p, const char *w);	/* accept a word */
int cmdp_tok(struct P *p, int kind);		/* accept a token */
int cmdp_at(struct P *p, int kind);		/* peek */
int cmdp_eos(struct P *p);
struct func *cmdp_func(struct P *p, const char *const *stop);
struct func *cmdp_func_typed(struct P *p, const char *const *stop, ty_t ty);
uint32_t cmdp_pos(struct P *p);
void cmdp_seek(struct P *p, uint32_t pos);
int cmdp_int(struct P *p, int64_t *v);
int cmdp_rest(struct P *p, char *buf, size_t n,
    const char *const *stop);
double cmdp_number(struct P *p, const char *const *stop);
void cmdp_fail_here(struct P *p);
int cmdp_word(struct P *p, char *buf, size_t n);
int cmdp_raw(struct P *p, char *buf, size_t n);	/* "a/b.png" */
/* ---- calc: loops and calculus (loop.c; the rest in calc.h) ---- */
nid mk_raw(int op, ty_t t, nid a, nid b, nid c, int64_t ki, int fl);
void loop_run(struct plan *p, uint32_t k, uint32_t np, const int64_t *in_i,
    const double *in_r);
void loop_print(struct sbuf *b, nid n, const struct pctx *pc);
nid loop_remake(nid n, nid a, nid b, nid c, const nid *map, int nmap);
float loop_cost(const struct model *m, nid n, int *cls);
nid loop_diff(nid n, uint32_t slot);

/* ---- forms.c: complex, polar and trig forms; the ops it adds ---- */
int forms_scalar(int op, double a, double b, double *r);  /* 0: undefined */
nid forms_diff(nid e, uint32_t slot);
/* sin^2 + cos^2 = 1 ...; the factors it cancelled (x/x) */
nid forms_norm(nid e, nid *cancelled, int max, int *ncancelled);
const char *forms_cond(nid n);    /* an op's domain condition, or NULL */
int forms_command(struct P *p);
int forms_num(double v, char *buf, size_t n);   /* for hyper.c */
void forms_ztext(double re, double im, struct sbuf *out);
int forms_reim(const char *text, struct sbuf *re, struct sbuf *im);

/* ---- hyper.c: complex, quaternion, octonion, Cayley-Dickson ---- */
int hyper_statement(const char *stmt);  /* 1: it was hypercomplex, and ran */
int hyper_scalar(int op, double a, double *r);
int hyper_text_lg(const char *text);   /* nlp.c: the algebra text needs, 0 real */  /* conj re im arg on reals */
nid hyper_diff(nid e, uint32_t slot);

/* ---- nlp.c: English in, ucas statements out ---- */
extern int nlp_interactive;
extern int nlp_defer;        /* questions go to the front end, not a tty */
extern int nlp_asked;        /* a deferred question is waiting */
int nlp_can_ask(void);
int nlp_ask_choice(const char *what, const char *const *opt, int nopt,
    int dflt);
int nlp_ask_pref(const char *key, const char *what, const char *const *opt,
    int nopt, int dflt, const char *hint);  /* -1: ask only when stdin and stdout are ttys */
int nlp_is_english(const char *line, size_t n);
int nlp_translate(const char *req, struct sbuf *dst);
void nlp_run(const char *line, size_t n, const char *file, int echo);
void run_text(const char *src, const char *file, int echo);
extern const char *code_block;     /* a fenced block after the statement */
extern char code_block_lang[24];
extern int code_block_used;

#include "report.h"	/* structured results for the JSON frontend */

#endif
