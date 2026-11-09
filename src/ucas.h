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
 * are undefined).  K_FLOAT is an IEEE binary32/64 machine type: every result
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
	O_NOPS
};
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
int atomic_lookup(const char *s, size_t n, int *kind);
nid subst(nid root, const nid *map, int nmap);
int is_const(nid n);
int const_is(nid n, double v);

enum { PM_UCAS, PM_C, PM_CBITS };
struct pctx {
	int mode;
	const char *const *vn;  /* optional names per param slot */
};
void print_node(struct sbuf *b, nid n, const struct pctx *pc);
const char *node_str(nid n);   /* static buffer, rotated; ucas syntax */
void print_paren(struct sbuf *b, nid n);

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
void basis_finish_gate(struct basis *b, struct gate *g);

/* ---- model.c: cost models ---- */
enum {
	C_IADD, C_LOGIC, C_SHIFT, C_IMUL, C_IDIV, C_IDIVK, C_CMP, C_SELECT,
	C_MINMAX, C_BITCNT, C_CONV, C_BITCAST,
	C_FADD, C_FMUL, C_FDIV, C_FSQRT, C_FTRANS, C_FPOW, C_F64,
	C_LOAD, C_BRANCH, C_DIVERGE, C_BARRIER, C_ATOM_SH, C_ATOM_GL,
	C_STORE,
	C_NCLASS
};
struct model {
	uint32_t name;
	float c[C_NCLASS];
	char desc[160];
};
extern const char *const cclass_name[C_NCLASS];
struct model *model_find(uint32_t name);
struct model *model_new(uint32_t name);
struct model *model_active(void);
void model_set_active(struct model *m);
int cclass_find(const char *s, size_t n);
float node_cost(const struct model *m, nid n, int *cls);
void cmd_cost(struct func *f, struct model *m);
void cmd_models(void);

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
	uint8_t real[MAXP];          /* 1 real grid, 2 f32 keys, 3 f64 keys */
	int exhaustive;
	uint64_t total, pos, limit;
	uint32_t rsteps;
	uint64_t rng;
};
void diter_init(struct diter *d, const struct func *f, uint64_t maxexh,
    uint32_t rsteps, uint64_t nsample);
uint32_t diter_next(struct diter *d, int64_t *in_i, double *in_r);
void param_bounds(const struct param *p, int64_t *lo, int64_t *hi);
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

/* ---- parse.c ---- */
void run_source(const char *src, const char *file, int echo);

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
/* glslc.c: GLSL (a snippet is completed) to SPIR-V assembly */
int glsl_to_spvasm(const char *src, struct sbuf *out, struct sbuf *notes);
int glsl_like(const char *code, const char *lang);
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

#endif
