/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: C99 in, proved and x86-64-optimised C99 out.
 *
 * The pipeline (cfront.md, "F1"):
 *
 *   source --cf_pp--> tokens --cf_parse--> AST (SoA) + C types
 *          --cf_lower--> per function: the node DAG (src/node.c) of each
 *            value, over "leaves" (parameters, loads, opaque values),
 *            with the C99 LP64 semantics made explicit (promotions,
 *            conversions, wrap, UB as domain conditions)
 *          --cf_opt--> candidate rewrites, each proved (exhaustive, SAT,
 *            algebra mod 2^w, induction) against the original value, or
 *            reported as unproved and not applied
 *          --cf_emit--> the original text with the proved spans replaced
 *            (everything else stays byte for byte), plus helpers
 *            (feature-macro guarded intrinsics with scalar fallbacks)
 *          --cf_check--> optional: both versions compiled with the host
 *            cc, differential test on random+edge inputs, rdtsc bench.
 *
 * Stability: the structures and hooks in this header are what the x86
 * side (cost models, intrinsics catalogue, vectoriser: src/cfx86.c on
 * branch cfront-x86) targets.  Changes here are additive.
 *
 * Data layout as everywhere in ucas: structures of arrays, handles
 * (uint32_t indices), arenas that grow.
 */
#ifndef CFRONT_H
#define CFRONT_H

#include "ucas.h"
#include "calc.h"

/* ================================================================
 * Targets and hooks (the x86 side plugs in here)
 * ================================================================ */

/* instruction-set features of a target */
enum {
	CF_SSE2, CF_SSE3, CF_SSSE3, CF_SSE41, CF_SSE42, CF_POPCNT, CF_LZCNT,
	CF_BMI1, CF_BMI2, CF_AVX, CF_AVX2, CF_FMA, CF_AVX512F, CF_AVX512BW,
	CF_AVX512DQ, CF_AVX512VL, CF_AVX512VBMI, CF_AVX512VPOPCNTDQ, CF_GFNI,
	CF_PCLMUL, CF_AES, CF_MOVBE, CF_ADX,
	CF_NFEAT
};
#define CF_F(x) (1ULL << (x))

struct cf_target {
	const char *name;	/* "zen4", "x86-64-v3" ... */
	const char *march;	/* for the host cc: -march=... */
	uint64_t feat;		/* CF_F(CF_...) bits */
	const char *desc;
	/* scalar costs in cycles (reciprocal throughput, with latency mixed
	 * in by the default model); the x86 side's model replaces them */
	float c_alu, c_mul, c_div32, c_div64, c_bitcnt_hw, c_bitcnt_sw,
	    c_load, c_branch_miss;
};
const struct cf_target *cf_target_find(const char *name);
const struct cf_target *cf_target_nth(int i);
const struct cf_target *cf_target_default(void);
/* the x86 side registers more (or replaces one of the same name) */
void cf_target_register(const struct cf_target *t);
/* the C macro that guards a feature: "__AVX2__" ... */
const char *cf_feat_macro(int feat);
const char *cf_feat_name(int feat);

/*
 * Cost hook: the estimated cost in cycles of evaluating root (a node DAG,
 * each shared subterm once) on t.  The default (cf_cost_default) is a
 * simple sum over op classes with the target's numbers; the x86 side's
 * scheduler estimate replaces it with cf_set_cost_hook.
 */
typedef double (*cf_cost_fn)(const struct cf_target *t, nid root);
void cf_set_cost_hook(cf_cost_fn f);
double cf_cost(const struct cf_target *t, nid root);
double cf_cost_default(const struct cf_target *t, nid root);

/*
 * Candidate hook: more implementations of a value.  Given the value e
 * (over leaves: O_VAR slots 0..nleaf-1, their C text in leaf[]), a
 * generator appends candidates.  Each has
 *   sem   its exact semantics as a node over the same leaves (what is
 *         proved equal to e: SAT, exhaustive or algebra),
 *   text  C text with $0..$7 standing for the leaves (parenthesised
 *         when substituted); NULL: the emitter prints sem itself,
 *   fallback  C text used when the target lacks feat (NULL: sem printed),
 *   feat  CF_F bits text needs (guarded by cf_feat_macro in the output),
 *   cost  on the target (cycles), what: a short description.
 */
struct cf_cand {
	nid sem;
	const char *text, *fallback;
	uint64_t feat;
	double cost;
	char what[96];
	/* a local rewrite justified by a lemma under proved facts (sound
	 * intervals of its operands): when SAT gives up, the lemma is the
	 * proof (CF_PROOF_ALGEBRA), provided sampling found no difference */
	char lemma[120];
	int derived;	/* added: lemma is a derivation (a proof of its own,
			 * e.g. IEEE float identities), not a hint */
};
struct cf_cands {
	struct cf_cand *c;
	int n, cap;
};
void cf_cand_add(struct cf_cands *cs, nid sem, const char *text,
    const char *fallback, uint64_t feat, double cost, const char *what);
typedef void (*cf_cand_fn)(const struct cf_target *t, nid e, int nleaf,
    const char *const *leaf, struct cf_cands *out);
void cf_add_cand_hook(cf_cand_fn f);	/* up to 8 generators */

/* ================================================================
 * Loops for the vectoriser
 * ================================================================ */

/*
 * A counted loop in canonical form, as the lowering sees it:
 *   for (i = lo; i < hi; i += step) { loads; values; stores; reductions }
 * Values are nodes over slots: the index is islot, each load a leaf slot,
 * loop-invariant scalars leaf slots too (their C text in leaf[]).
 * Memory facts: a base is a pointer parameter or array; noalias[b] says
 * base b may not overlap any other base written in the loop (restrict,
 * distinct arrays, or proved disjoint).  A loop that does not fit this
 * shape (calls, volatile, unknown pointers, early exits other than the
 * condition) is not offered to the vectoriser.
 */
#define CF_MAXMEM 16
struct cf_mem {
	int base;		/* index into cf_loop.base[] */
	nid index;		/* element index, over islot and leaves */
	ty_t ety;		/* element type (ucas) */
	int esize;		/* bytes */
	int slot;		/* loads: the leaf slot holding the value */
	nid value;		/* stores: the value stored */
	int epoch;		/* stores before this access in the body:
				 * the order of loads and stores */
};
struct cf_loop {
	const char *func;	/* the enclosing function */
	uint32_t stmt;		/* the loop's AST statement */
	int islot;		/* the index's slot */
	ty_t ity;		/* its type */
	const char *iname;	/* its C name */
	nid lo, hi;		/* i runs lo, lo+step, ... while i < hi */
	int64_t step;
	const char *lo_text, *hi_text;	/* C text, as written */
	int nbase;
	const char *base[CF_MAXMEM];	/* C text of each base pointer */
	uint8_t noalias[CF_MAXMEM];
	uint8_t readonly[CF_MAXMEM];
	int nload, nstore;
	struct cf_mem load[CF_MAXMEM], store[CF_MAXMEM];
	int nred;		/* reductions: acc = next(acc, ...); rslot is
				 * acc's leaf (-1 when the loop never reads
				 * it), rinit 0: its value before the loop */
	int rslot[MAXST];
	const char *rname[MAXST];
	nid rinit[MAXST], rnext[MAXST];
	int nleaf;		/* slots below nleaf are leaves (C text) */
	const char *leaf[NSLOT];
	/* added: */
	int idecl;		/* the for statement declares the index */
	const char *init_text;	/* the for statement's first clause */
	ty_t leafty[NSLOT];	/* the leaves' types */
};

/*
 * Vectoriser hook: replace the loop.  On success it fills out->text with
 * C statements (intrinsics guarded by feature macros, with a scalar
 * tail/fallback) that replace the whole for statement, a description and
 * the proof kind it can claim (CF_PROOF_*), and returns 1.  The front end
 * then differential-tests the result when a host cc is available.
 */
struct cf_loopout {
	struct sbuf text;
	char what[160];
	int proof;
	double cost_before, cost_after;	/* per iteration, cycles */
};
typedef int (*cf_vec_fn)(const struct cf_loop *l, const struct cf_target *t,
    struct cf_loopout *out);
void cf_set_vectoriser(cf_vec_fn f);
/* the IR as text (c lower prints it) */
void cf_loop_print(const struct cf_loop *l, struct sbuf *b);
/* a stub vectoriser for tests (option vecstub): rebuilds the loop from
 * its IR, scalar, so a differential check validates the extraction */
int cf_vec_stub(const struct cf_loop *l, const struct cf_target *t,
    struct cf_loopout *out);

/*
 * The x86 side's entry point (src/cfx86.c, when it exists: the Makefile
 * then defines CF_HAVE_X86): it registers its targets, cost model,
 * candidate generators and vectoriser.  Called once before the first
 * optimisation.
 */
void cfx86_init(void);

/* ================================================================
 * Proofs and the rewrite report
 * ================================================================ */

enum {
	CF_PROOF_NONE,		/* not proved: not applied without opt-in */
	CF_PROOF_EXHAUSTIVE,	/* every input checked */
	CF_PROOF_SAT,		/* bit-blasted, the miter unsatisfiable */
	CF_PROOF_ALGEBRA,	/* ring identities mod 2^w */
	CF_PROOF_INDUCTION,	/* closed form: base and step proved (SAT) */
	CF_PROOF_STRUCT,	/* by construction (a semantics-preserving
				 * rewrite of the program's structure) */
	CF_PROOF_LANES,		/* intrinsics: lane semantics, per lane */
	CF_PROOF_TESTED		/* only differential testing (opt-in) */
};
const char *cf_proof_name(int k);

struct cf_rewrite {
	char func[64];
	int line;
	char kind[48];		/* "switch to table", "drop clamp" ... */
	char *before, *after;	/* C text */
	int proof;
	char proof_text[400];
	double cost_before, cost_after;
	int applied;
	int info;		/* not a rewrite: why nothing happened */
	char *steps;		/* every step's description */
	int trivial;		/* clang -O2 already produces the same code */
	char *domain;		/* input ranges assumed or from call sites */
	char *cost_note;	/* what the costs measure, when not per call */
	char *code;		/* info: code proposed but not applied */
};

/* ================================================================
 * Internals shared by the cf_*.c files
 * ================================================================ */

/* ---- tokens (after preprocessing) ---- */
enum {
	CK_EOF, CK_ID, CK_NUM, CK_CHAR, CK_STR, CK_PUNCT, CK_KW
};
#define CTF_MACRO 1	/* came out of a macro expansion */
#define CTF_BOL 2	/* first on its line */
#define CTF_SPACE 4	/* whitespace before */
#define CTF_BUILTIN 8	/* from the built-in prelude (system headers) */
struct cf_toks {
	uint8_t *kind, *fl;
	uint16_t *file;		/* 0: the main file */
	uint32_t *off, *len;	/* in the file's text */
	uint32_t *line, *col;
	uint32_t *sym;		/* interned text (sym_intern) */
	uint32_t n, cap;
};

/* ---- C types ---- */
enum {
	CT_VOID, CT_BOOL, CT_CHAR, CT_SCHAR, CT_UCHAR, CT_SHORT, CT_USHORT,
	CT_INT, CT_UINT, CT_LONG, CT_ULONG, CT_LLONG, CT_ULLONG,
	CT_FLOAT, CT_DOUBLE, CT_LDOUBLE,
	CT_PTR, CT_ARRAY, CT_FUNC, CT_STRUCT, CT_UNION, CT_ENUM,
	CT_VEC,			/* __m128i ... (opaque, by size) */
	CT_NKIND
};
#define CQ_CONST 1
#define CQ_VOLATILE 2
#define CQ_RESTRICT 4
struct cf_types {
	uint8_t *kind, *qual;
	uint32_t *base;		/* ptr/array: element; func: return */
	int64_t *count;		/* array: elements (-1 unknown) */
	uint32_t *size, *align;
	uint32_t *fst, *nf;	/* struct/union/func: fields/params in flist */
	uint32_t *tag;		/* struct/union/enum tag sym */
	uint8_t *variadic, *complete;
	uint32_t n, cap;
	/* fields and parameters */
	uint32_t *fname, *fty, *foff;
	uint32_t nfl, fcap;
};
extern struct cf_types CT;
uint32_t ct_basic(int kind);
uint32_t ct_qual(uint32_t t, int q);
uint32_t ct_unqual(uint32_t t);
uint32_t ct_ptr(uint32_t base);
uint32_t ct_array(uint32_t base, int64_t n);
int ct_is_int(uint32_t t);	/* integer types incl. bool, char, enum */
int ct_is_arith(uint32_t t);
int ct_is_scalar(uint32_t t);
int ct_is_signed(uint32_t t);
int ct_rank(uint32_t t);
uint32_t ct_promote(uint32_t t);
uint32_t ct_usual(uint32_t a, uint32_t b);
ty_t ct_to_ucas(uint32_t t);	/* integer C type -> uN/iN/bool, float and
				 * double -> f32/f64; 0 if none */
int ct_is_lowered(uint32_t t);	/* an integer, float or double */
void ct_str(struct sbuf *b, uint32_t t, const char *name);  /* a declaration */

/* ---- the AST ---- */
enum {
	/* expressions */
	CA_NUM, CA_FNUM, CA_CHARLIT, CA_STR, CA_ID, CA_CALL, CA_INDEX,
	CA_MEMBER, CA_ARROW, CA_POSTINC, CA_POSTDEC, CA_PREINC, CA_PREDEC,
	CA_ADDR, CA_DEREF, CA_PLUS, CA_NEG, CA_BNOT, CA_LNOT, CA_SIZEOF_E,
	CA_SIZEOF_T, CA_CAST, CA_BIN, CA_LAND, CA_LOR, CA_COND, CA_ASSIGN,
	CA_COMMA, CA_COMPOUND, CA_INIT,	/* (T){...}, {...} */
	/* statements */
	CA_BLOCK, CA_EXPR, CA_IF, CA_WHILE, CA_DO, CA_FOR, CA_RETURN,
	CA_BREAK, CA_CONTINUE, CA_GOTO, CA_LABEL, CA_SWITCH, CA_CASE,
	CA_DEFAULT, CA_DECL, CA_EMPTY,
	/* top level */
	CA_FUNC, CA_GDECL, CA_TYPEDEF,
	CA_NKIND
};
/* CA_BIN, CA_ASSIGN: op in iv (C op: '+', '-', ... or CO_*) */
enum {
	CO_SHL = 256, CO_SHR, CO_LE, CO_GE, CO_EQ, CO_NE
};
struct cf_ast {
	uint8_t *kind;
	uint32_t *ty;		/* C type of an expression (after sema) */
	uint32_t *a, *b, *c, *d;
	uint32_t *t0, *t1;	/* first and last token */
	int64_t *iv;		/* literal, op, sym, flags */
	double *fv;
	uint32_t n, cap;
	uint32_t *lst;		/* child lists: a = start, b = count */
	uint32_t nlst, lcap;
};

/* a declared object or function */
#define CS_PARAM 1
#define CS_LOCAL 2
#define CS_GLOBAL 3
#define CS_FUNC 4
#define CS_ENUMC 5
#define CS_TYPEDEF 6
struct cf_syms {
	uint32_t *name, *ty, *scope;
	uint8_t *kind, *stclass;	/* stclass: 's' static, 'e' extern */
	uint8_t *addr_taken, *assigned;
	int64_t *val;			/* enum constants */
	uint32_t *decl;			/* AST node */
	uint32_t n, cap;
};

/* one translation unit */
struct cf_unit {
	const char *path;
	char *text;			/* the main file */
	int nfiles;
	char **ftext;			/* every file's text, [0] = main */
	char **fname;
	struct cf_toks tk;
	struct cf_ast A;
	struct cf_syms S;
	uint32_t top;			/* the top-level list */
	uint32_t *fns;			/* CA_FUNC nodes */
	int nfn;
	struct sbuf includes;		/* system #include lines, in order */
	struct sbuf err;
	/* macro names defined anywhere (an emitted identifier must not
	 * collide with one) */
	uint32_t *macros;
	int nmacro;
	/* per token: the main-file extent of its origin (itself, or the
	 * outermost macro invocation it came out of) and that invocation's
	 * group (0 none, ~0 an invocation that cannot be mapped back) */
	uint32_t *tk_o0, *tk_o1, *tk_grp;
	int synfile;
};
int cf_tspan(const struct cf_unit *u, uint32_t t0, uint32_t t1,
    uint32_t *from, uint32_t *to);

/* cf_pp.c: the preprocessor and lexer */
int cf_preprocess(struct cf_unit *u, const char *path, const char *text,
    const struct cf_target *t);
const char *cf_tok_text(const struct cf_unit *u, uint32_t i, size_t *n);
int cf_is_macro(const struct cf_unit *u, uint32_t sym);

/* cf_parse.c: the parser and the type checker */
int cf_parse(struct cf_unit *u);
void cf_unit_free(struct cf_unit *u);
uint32_t ca_new(struct cf_ast *A, int kind);
#define CA_LIST(u, n, k) ((u)->A.lst[(u)->A.a[n] + (k)])

/* the span of an AST node in the main file: 0 if it is not one
 * contiguous piece of the main file without macros or directives */
int cf_span(const struct cf_unit *u, uint32_t node, uint32_t *from,
    uint32_t *to);

/* cf_lower.c: functions to node DAGs */
#define CF_MAXLEAF MAXP
struct cf_leaf {
	char text[96];		/* C text (a parameter, a load ...) */
	uint32_t cty;		/* its C type */
	int64_t lo, hi;		/* known range (ranged = 1) */
	int ranged;
	int from_calls;		/* the range is the call sites' (static) */
};
struct cf_value {
	nid e;			/* the value */
	nid where;		/* where it is defined (UB excluded); 0 all */
	int nleaf;
	struct cf_leaf leaf[CF_MAXLEAF];
	const char *why;	/* when e == 0: why it could not be lowered */
	/* added: loops summed in closed form (cf_lower_func), each a
	 * derivation (see closed_for in cf_lower.c), and the cost of one
	 * iteration of those loops */
	char closed[240];
	double iter_cost;
};
/* the whole function as one value (scalar parameters, no memory) */
int cf_lower_func(struct cf_unit *u, uint32_t fn, struct cf_value *v,
    int ub);
/* one expression, its operands' atoms as leaves */
int cf_lower_expr(struct cf_unit *u, uint32_t fn, uint32_t ex,
    struct cf_value *v);
/* a struct func for the proof engines over v's leaves */
void cf_value_func(const struct cf_value *v, struct func *f);

/*
 * One iteration of a loop, lowered on its own (the vectoriser's IR, body
 * rewrites, closed forms): a variable read before it is written is a
 * leaf (its value at the start of the iteration), so is each load (keyed
 * by its text and the number of stores before it: two reads of the same
 * address with no store between are one value).  Stores are recorded in
 * order; a load or store under a condition, a pointer that moves, a
 * break, a return, a nested loop or a call that touches memory is not
 * lowered.  The condition is lowered before the body (it must have no
 * side effects); the body as if the condition held.
 */
struct cf_memref {
	uint32_t node;		/* the AST node (CA_INDEX, CA_DEREF) */
	uint32_t base;		/* the base symbol */
	nid index;		/* the element index, over the leaves */
	uint32_t cty;		/* the element's C type */
	int epoch;		/* stores before it in the iteration */
	int slot;		/* loads: the leaf */
	nid value;		/* stores: the value stored */
};
#define CF_MAXSET 16
#define CF_MAXX 4
struct cf_iter {
	struct cf_value v;	/* v.e unused; v.where: the iteration's UB */
	uint32_t leafsym[CF_MAXLEAF];	/* symbol + 1, or 0 (a load) */
	int leafload[CF_MAXLEAF];	/* the load of a leaf, or -1 */
	nid cond;		/* the condition at the start, 0 none */
	int nset;		/* variables declared outside the body that
				 * the body (and step) write: their values
				 * at the end of the iteration */
	uint32_t setsym[CF_MAXSET];
	nid setval[CF_MAXSET];
	int nload, nstore;
	struct cf_memref load[CF_MAXMEM], store[CF_MAXMEM];
	int nx;			/* extra expressions lowered in the same
				 * leaves (before the body): x[k] -> xval[k] */
	uint32_t x[CF_MAXX];
	nid xval[CF_MAXX];
};
int cf_lower_iter(struct cf_unit *u, uint32_t fn, uint32_t loop,
    int with_step, struct cf_iter *it);
/* the loop in the vectoriser's form (cf_loop), from a for statement;
 * 0 and why when it does not fit */
int cf_extract_loop(struct cf_unit *u, uint32_t fn, uint32_t loop,
    struct cf_loop *l, struct cf_iter *it, const char **why);

/* cf_emit.c: nodes to C */
struct cf_emit {
	struct sbuf helpers;	/* static inline helpers, tables */
	uint64_t used;		/* helpers already emitted */
	int ntab;
	const struct cf_target *t;
	int need_string;	/* the output calls strlen: <string.h> */
	int floats;		/* float arithmetic emitted: FP_CONTRACT OFF */
};
/* the C text of e, with leaves printed from v; 0 if it cannot be
 * printed without introducing UB */
int cf_emit_expr(struct cf_emit *em, const struct cf_value *v, nid e,
    uint32_t cty, struct sbuf *out);
int cf_emit_body(struct cf_emit *em, const struct cf_value *v, nid e,
    uint32_t cty, struct sbuf *out);	/* { temps; return e; } */

/* cf_opt.c: the rewrite pipeline */
struct cf_opts {
	const struct cf_target *t;
	double budget;		/* seconds for proofs */
	int wrapv;		/* signed overflow wraps (no UB exploited) */
	int approx;		/* apply unproved (only tested) rewrites */
	int check, bench, nofile, show_lower;
	int nassume;
	struct {
		char name[48];
		int64_t lo, hi;
	} assume[16];
	/* added after the first version (cfront.md: changes are additive) */
	int trace;		/* print every candidate and why it was not
				 * taken (to stdout, "trace:" lines) */
	int keep;		/* check/bench: keep the build directory */
	int timed;		/* print the seconds spent */
	int vecstub;		/* register the stub vectoriser (tests) */
	/* the x86 vectoriser's loops are proposed, not applied, unless
	 * vectorize (the option), or bench measured them faster than the
	 * original as clang compiles it: then every function but those in
	 * vveto (slower or not clearly faster; their cycles per element,
	 * original and vectorised) */
	int vectorize;
	int nvveto;
	char vveto[16][64];
	double vveto_c[16][2];
	/* loop-body rewrites that block vectorisation (a table indexed by
	 * data, a helper call): by default proposed, not applied, on a loop
	 * the vectoriser can prove (clang vectorises the original); bench
	 * measures them: bodyall applies them all (its first measurement),
	 * novec keeps the vectoriser out, bapply lists the functions whose
	 * body rewrites won, bjudged says bench decided, bveto_c the cycles
	 * per element of those that lost */
	int bodyall, novec, bjudged;
	int nbapply, nbveto;
	char bapply[16][64], bveto[16][64];
	double bveto_c[16][2];
};
struct cf_result {
	struct sbuf out;		/* the emitted C */
	struct cf_rewrite *rw;
	int nrw, rwcap;
	double cost_before, cost_after;
	int nproofs;
	double seconds;
	uint8_t *calls_only;	/* per function: entered only from this file's
				 * calls, rewritten over their domain */
};
int cf_optimize(struct cf_unit *u, const struct cf_opts *o,
    struct cf_result *r);
void cf_result_free(struct cf_result *r);

/* cf_check.c: the host cc, differential tests and benchmarks */
int cf_have_cc(void);
int cf_difftest(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep);
int cf_bench(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep);
/* the last bench's medians for a function, and in how many of its runs
 * the output beat the original by more than 5%; the elements per call
 * (1 without buffers), 0 if it was not timed */
int cf_bench_get(const char *fn, double *c0, double *c1, int *wins,
    int *runs);

/* cf_cmd.c: commands */
struct P;
int cfront_line(const char *rest);	/* parse.c: "c " + rest of the line */
int cfront_claims(const char *stmt, const char *lang);	/* nlp.c */
int cfront_english(const char *stmt, struct sbuf *out);	/* nlp.c */
void cfront_run(const char *src, const char *path, const struct cf_opts *o);

#endif
