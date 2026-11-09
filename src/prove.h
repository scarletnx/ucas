/*
 * prove.h - an LCF-style proof kernel for exact abstract algebra.
 *
 * Terms and formulas share one hash-consed DAG in struct-of-arrays
 * columns (KT).  Bound variables are de Bruijn indices, so alpha-
 * equivalent formulas are the same id.  A theorem (kthm) is a sequent
 * Gamma |- phi stored in kernel.c's own SoA table; the only way to make
 * one is a kernel rule (kernel.c), and every theorem records the rule,
 * its premises and its term arguments, so the proof is a DAG that
 * kcheck.c re-checks with an independent implementation.
 *
 * Layers:
 *   kterm.c   sorts, signature, terms, substitution, parser, printer
 *   knorm.c   normal forms: commutative-ring polynomials over Z (int),
 *             Z[...]-module linear combinations (el), homomorphisms
 *   kernel.c  the rules and the axiom table
 *   kcheck.c  the independent re-checker
 *   kauto.c   decision procedures that return kernel theorems: linear
 *             arithmetic (Fourier-Motzkin with certificates, case splits,
 *             products), abelian-group equalities, subgroup membership,
 *             backward chaining over lemmas (diagram chasing)
 *   kscript.c theories: declarations, definitions, and structured proofs
 *   prove_cmd.c  theorem/prove/proof/check/axioms ... (parse.c hook)
 */
#ifndef PROVE_H
#define PROVE_H

#include <stdint.h>

struct sbuf;
struct P;

typedef uint32_t kt;	/* a term or formula */
typedef uint32_t kthm;	/* a theorem */
#define KNONE 0

/* ---- sorts ---- */
enum { KS_PROP, KS_INT, KS_EL, KS_USER };
#define KS_MAX 32
int ks_lookup(const char *name);	/* -1: none */
int ks_declare(const char *name);
const char *ks_name(int s);

/* ---- ops ---- */
enum {
	KO_NONE,
	KO_BVAR,	/* aux = de Bruijn index */
	KO_VAR,		/* aux = name (sym id); a free variable or constant */
	KO_META,	/* aux = metavariable index (kauto.c only) */
	KO_NUM,		/* num = value (int) */
	KO_ZERO,	/* 0 of el */
	KO_APP,		/* aux = function symbol; args */
	KO_ADD, KO_NEG, KO_MUL, KO_POW,	/* MUL: int*int or int*el */
	KO_TRUE, KO_FALSE,
	KO_PRED,	/* aux = predicate symbol; args */
	KO_EQ, KO_LE, KO_LT, KO_IN,
	KO_NOT, KO_AND, KO_OR, KO_IMP, KO_IFF,
	KO_ALL, KO_EX,	/* aux = bound sort; arg 0 = body; hint = name */
	KO_NOPS
};

struct kterms {
	uint8_t *op, *sort, *n;
	uint16_t *lb;		/* loose bound variables: max index + 1 */
	uint32_t *aux, *a0, *hint, *fv;	/* fv: bloom mask of VAR/META */
	int64_t *num;
	uint32_t *args;
	uint32_t len, cap, nargs, capargs;
};
extern struct kterms KT;
#define KOP(t) (KT.op[t])
#define KSORT(t) (KT.sort[t])
#define KN(t) (KT.n[t])
#define KA(t, i) (KT.args[KT.a0[t] + (i)])

/* ---- signature ---- */
enum { KSY_FUNC, KSY_PRED };
#define KSF_HOM 1	/* additive in its last (el) argument */
struct ksym {
	uint32_t name;
	uint8_t kind, arity, res, flags;
	uint8_t arg[8];
	uint32_t homthm;	/* the theorem that made it a hom (0: declared) */
};
extern struct ksym *ksyms;
extern uint32_t nksyms;
int ksym_lookup(const char *name);
int ksym_declare(const char *name, int kind, int arity, const uint8_t *args,
    int res, int flags);

/* ---- constructors ---- */
kt kt_bvar(uint32_t i, int sort);
kt kt_var(uint32_t name, int sort);
kt kt_varc(const char *name, int sort);
kt kt_meta(uint32_t i, int sort);
kt kt_num(int64_t v);
kt kt_zero(void);
kt kt_app(int sym, const kt *args, int n);
kt kt_add(kt a, kt b);
kt kt_sub(kt a, kt b);
kt kt_neg(kt a);
kt kt_mul(kt a, kt b);
kt kt_pow(kt a, kt b);
kt kf_true(void);
kt kf_false(void);
kt kf_pred(int sym, const kt *args, int n);
kt kf_eq(kt a, kt b);
kt kf_le(kt a, kt b);
kt kf_lt(kt a, kt b);
kt kf_in(kt x, kt s);
kt kf_not(kt a);
kt kf_and(kt a, kt b);
kt kf_or(kt a, kt b);
kt kf_imp(kt a, kt b);
kt kf_iff(kt a, kt b);
kt kf_bind(int op, int sort, uint32_t hint, kt body);
kt kt_mk(int op, int sort, uint32_t aux, int64_t num, const kt *args, int n);

/* ---- operations ---- */
kt kt_inst(kt body, kt s);		/* body[bvar 0 := s] (s closed) */
kt kt_abstract(kt t, kt var);		/* var -> bvar 0, for a binder */
int kt_occurs(kt v, kt t);		/* VAR or META v occurs in t */
int kt_hasmeta(kt t);
kt kt_metasub(kt t, const kt *val, uint32_t nval);  /* metas -> values */
int kt_isformula(kt t);
void kt_print(struct sbuf *b, kt t);
const char *kt_str(kt t);		/* rotating static buffers */
kt kt_fresh(const char *base, int sort);	/* a new free variable */

/* the parser: text in ucas syntax; scope gives free variables */
struct kscope {
	uint32_t name[64];
	kt var[64];
	int n;
};
kt kt_parse(const char *text, const struct kscope *sc, char *err,
    int errn);
kt kt_parse_term(const char *text, const struct kscope *sc, int sort,
    char *err, int errn);

/* ---- knorm.c ---- */
kt kn_norm(kt t);
int kn_equiv(kt a, kt b);
void kn_reset(void);			/* after a symbol becomes a hom */
int kn_poly_const(kt t, int64_t *v);	/* normal form is a numeral */
int kn_split(kt t, kt *atom, kt *coef, int max);

/* ---- kernel.c: rules (KNONE and kerr() on failure) ---- */
enum {
	KR_ASSUME, KR_AXIOM, KR_IMPI, KR_IMPE, KR_ANDI, KR_ANDE1, KR_ANDE2,
	KR_ORI1, KR_ORI2, KR_ORE, KR_FALSEE, KR_NOTI, KR_NOTE, KR_RAA,
	KR_IFFI, KR_IFFE1, KR_IFFE2, KR_ALLI, KR_ALLE, KR_EXI, KR_EXE,
	KR_REFL, KR_SUBST, KR_CONV, KR_LIN, KR_INDUCT, KR_HOMDECL,
	KR_NRULES
};
extern const char *const kr_name[KR_NRULES];

kthm kr_assume(kt phi);
kthm kr_axiom(const char *name);
kthm kr_impi(kthm th, kt phi);
kthm kr_impe(kthm imp, kthm a);
kthm kr_andi(kthm a, kthm b);
kthm kr_ande1(kthm th);
kthm kr_ande2(kthm th);
kthm kr_ori1(kthm th, kt psi);
kthm kr_ori2(kt phi, kthm th);
kthm kr_ore(kthm or, kthm c1, kthm c2);
kthm kr_falsee(kthm th, kt phi);
kthm kr_noti(kthm th, kt phi);
kthm kr_note(kthm notphi, kthm phi);
kthm kr_raa(kthm th, kt phi);
kthm kr_iffi(kthm ab, kthm ba);
kthm kr_iffe1(kthm th);
kthm kr_iffe2(kthm th);
kthm kr_alli(kthm th, kt var);
kthm kr_alle(kthm th, kt t);
kthm kr_exi(kt exphi, kt t, kthm th);
kthm kr_exe(kthm ex, kt var, kthm th);
kthm kr_refl(kt t);
kthm kr_subst(kthm eq, kt pat, kthm th);  /* pat: body with bvar 0 */
kthm kr_conv(kthm th, kt phi);
kthm kr_lin(const kthm *ths, const kt *coef, int n);
kthm kr_induct(kt base, kt pat, kthm b, kthm step);
kthm kr_homdecl(int sym, kthm add, kthm smul);

const char *kerr(void);
kt kthm_concl(kthm th);
uint32_t kthm_nhyps(kthm th);
kt kthm_hyp(kthm th, uint32_t i);
int kthm_rule(kthm th);
uint32_t kthm_nprem(kthm th);
kthm kthm_prem(kthm th, uint32_t i);
uint32_t kthm_nargs(kthm th);
kt kthm_arg(kthm th, uint32_t i);
uint32_t kthm_count(void);
uint32_t kthm_aux(kthm th);	/* KR_AXIOM: axiom index; KR_HOMDECL: sym */

/* axioms: only kscript.c registers them (definitions, arithmetic) */
struct kaxiom {
	uint32_t name;
	kt phi;
	const char *kind;	/* "definition", "arithmetic", ... */
	const char *text;	/* as written */
};
extern struct kaxiom *kaxioms;
extern uint32_t nkaxioms;
int kax_add(const char *name, kt phi, const char *kind, const char *text);
int kax_find(const char *name);

/* ---- kcheck.c ---- */
/* re-check th and everything it depends on; 1 ok, else msg says why */
int kcheck(kthm th, char *msg, int n, uint32_t *nsteps);

/* ---- kauto.c ---- */
struct kctx {
	kthm fact[512];
	uint32_t name[512];
	int n;
	kthm ref[64];		/* lemmas offered to the chainer */
	int nref;
	kt with[8];		/* witnesses for leading existentials */
	int nwith;
	int depth;
};
kthm ka_prove(struct kctx *c, kt goal);
kthm ka_close(struct kctx *c, kt goal);	/* a fact, up to conv */
void ka_add(struct kctx *c, kthm th, uint32_t name);
extern char ka_why[512];		/* why it failed */
extern uint64_t ka_budget;

/* ---- kscript.c ---- */
struct klemma {
	uint32_t name;
	kt phi;
	kthm th;		/* 0: stated, not proved */
	const char *src;	/* the statement as written */
	int lib;		/* from the built-in theory */
	const char *route;	/* "a", "b": which proof */
	uint32_t steps;		/* script steps */
	uint32_t kfirst, klast;	/* the theorem ids its proof made */
	int l0, l1;		/* its lines in the library */
};
extern struct klemma *klemmas;
extern uint32_t nklemmas;
int kl_find(const char *name);
void ks_load_library(void);
/* run a theory text; 0 on success, msg holds the error */
int ks_run(const char *text, const char *file, char *msg, int n);
int ks_add_user(const char *name, kt phi, const char *src);
void ks_print_script(int lemma);	/* its proof, as written */

/* ---- prove_cmd.c ---- */
int prove_command(struct P *p);
int prove_is_cmd(const char *s, unsigned long n);	/* nlp.c: ours, not English */

#endif
