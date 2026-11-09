/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * alg: symbolic algebra.  Exact rational arithmetic (int64 with 128-bit
 * intermediates; an overflow sets alg_ovf and the caller gives up),
 * sparse multivariate polynomials over Q in "atoms" (variables and
 * opaque subterms such as sin(x)), rational functions, dense univariate
 * polynomials (division, gcd, square-free and full factorisation over
 * Q, resultants), conversion between nodes and polynomials (poly.c);
 * the user commands expand, factor, collect, simplify, together, apart,
 * cancel, gcd, lcm, resultant, discriminant, quo, rem, substitute
 * (alg.c); integration of rational functions and the other methods
 * (rint.c); equation solving (solve.c).
 *
 * Memory: every polynomial lives in an arena.  ar_mark() before a piece
 * of work, ar_release() after it; nodes (the DAG) outlive the arena.
 */
#ifndef ALG_H
#define ALG_H

#include "calc.h"

/* ---- arena ---- */
size_t ar_mark(void);
void ar_release(size_t m);
void *ar_alloc(size_t n);

/* ---- rationals (struct q from calc.h) ---- */
typedef struct q aq;
extern int alg_ovf;          /* a coefficient left the int64 range */
aq qi(int64_t n);
aq qmk(int64_t n, int64_t d);
aq qadd(aq a, aq b);
aq qsub(aq a, aq b);
aq qmul(aq a, aq b);
aq qdiv(aq a, aq b);
aq qneg(aq a);
int qeq(aq a, aq b);
int qsgn(aq a);
int qexact(double v, aq *r);   /* v is exactly a fraction with a small denominator */
double qd(aq a);
int64_t lcm64(int64_t a, int64_t b);

/* ---- atoms ---- */
#define AV 10                /* atoms per polynomial */
extern nid at_node[AV];
extern int nat;
void atoms_reset(void);
int atom_of(nid e);          /* index, adding it; -1 when full */
int atom_find(nid e);

/* ---- sparse multivariate polynomials ---- */
struct mono {
	aq c;
	uint8_t e[AV];
};
struct mp {
	int n;
	struct mono *t;      /* sorted, lex order on e[0], e[1], ... descending */
};
struct mp mp_zero(void);
struct mp mp_const(aq c);
struct mp mp_atom(int v);
struct mp mp_add(struct mp a, struct mp b);
struct mp mp_sub(struct mp a, struct mp b);
struct mp mp_neg(struct mp a);
struct mp mp_mul(struct mp a, struct mp b);
struct mp mp_scale(struct mp a, aq c);
struct mp mp_pow(struct mp a, int n);
struct mp mp_copy(struct mp a);
int mp_iszero(struct mp a);
int mp_isconst(struct mp a, aq *c);
int mp_eq(struct mp a, struct mp b);
int mp_deg(struct mp a, int v);
int mp_tdeg(struct mp a);                    /* total degree */
uint32_t mp_vars(struct mp a);               /* bit mask of atoms used */
struct mp mp_coef(struct mp a, int v, int k);  /* coefficient of v^k */
struct mp mp_lcv(struct mp a, int v);         /* leading coefficient in v */
int mp_divexact(struct mp a, struct mp b, struct mp *q);
struct mp mp_prem(struct mp a, struct mp b, int v);
struct mp mp_gcd(struct mp a, struct mp b);
struct mp mp_content(struct mp a, int v);
struct mp mp_diff(struct mp a, int v);
struct mp mp_subst(struct mp a, int v, struct mp val);
aq mp_qcontent(struct mp a);                  /* rational content, sign of lt */
struct mp mp_prim(struct mp a);               /* a / content: integer, lt > 0 */
double mp_eval(struct mp a, const double *x);
struct mp mp_resultant(struct mp a, struct mp b, int v);
struct mp mp_discriminant(struct mp a, int v);
/* factor: c * prod f[i]^m[i] (primitive, lt > 0); returns the count */
int mp_factor(struct mp a, aq *c, struct mp *f, int *m, int max, int *partial);

/* ---- rational functions ---- */
struct rf {
	struct mp n, d;
};
struct rf rf_add(struct rf a, struct rf b);
struct rf rf_sub(struct rf a, struct rf b);
struct rf rf_mul(struct rf a, struct rf b);
struct rf rf_div(struct rf a, struct rf b);
struct rf rf_pow(struct rf a, int n);
struct rf rf_norm(struct rf a);       /* gcd cancelled, d primitive with lt > 0 */
struct rf rf_poly(struct mp a);

/*
 * node -> polynomial / rational function.  Sums, products, integer
 * powers and quotients are taken apart; anything else is an atom.
 * Returns 0 if the expression does not fit (too many atoms, a float
 * constant that is not exact, an overflow).  rf_of collects in *dall
 * the product of every denominator met (before cancelling), for the
 * conditions.
 */
int mp_of(nid e, struct mp *r);
int rf_of(nid e, struct rf *r, struct mp *dall);
nid mp_node(struct mp a);                    /* expanded */
nid mp_node_in(struct mp a, int v);          /* collected in atom v */
nid mp_fnode(struct mp a);                   /* factored when possible */
nid rf_node(struct rf a, int factored);
nid q_cnode(aq c);                           /* a rational that prints exactly */
nid qmul_node(aq c, nid x);                  /* c * x, written p x / q */
nid sqrt_cnode(int64_t m);                   /* sqrt(m), not folded */
const char *mp_str(struct mp a);
/* the condition text "f != 0" for factors of dall that rf's d lost */
void rf_conds(struct mp dall, struct mp d, struct sbuf *out);

/* ---- dense univariate polynomials over Q ---- */
#define UD 40
struct up {
	int d;               /* degree; -1 for 0 */
	aq c[UD + 1];
};
int up_of_mp(struct mp a, int v, struct up *r);   /* 0 if other atoms */
struct mp mp_of_up(const struct up *a, int v);
void up_zero(struct up *r);
void up_set(struct up *r, const struct up *a);
void up_trim(struct up *r);
void up_add(struct up *r, const struct up *a, const struct up *b);
void up_sub(struct up *r, const struct up *a, const struct up *b);
void up_mul(struct up *r, const struct up *a, const struct up *b);
void up_scale(struct up *r, const struct up *a, aq c);
int up_divmod(const struct up *a, const struct up *b, struct up *q,
    struct up *rem);
void up_gcd(struct up *g, const struct up *a, const struct up *b);
int up_gcdex(const struct up *a, const struct up *b, struct up *s,
    struct up *t, struct up *g);       /* s a + t b = g (monic) */
void up_diff(struct up *r, const struct up *a);
void up_monic(struct up *r, const struct up *a);
double up_eval(const struct up *a, double x);
aq up_evalq(const struct up *a, aq x);
int up_iszero(const struct up *a);
/* square-free parts: a = c * prod s[i]^(i+1); returns the top index + 1 */
int up_sqfree(const struct up *a, struct up *s, int max);
/* irreducible factors over Q (monic) with multiplicities */
int up_factor(const struct up *a, struct up *f, int *m, int max, int *partial);
nid up_node(const struct up *a, nid x);             /* integer-scaled? no: exact */
nid up_pnode(const struct up *a, nid x);            /* primitive integer form */
const char *up_str(const struct up *a, nid x);

/* ---- Q(sqrt d) ---- */
struct qs {
	aq a, b;                 /* a + b sqrt(d) */
};
extern int64_t qs_d;
struct qs qs_add(struct qs x, struct qs y);
struct qs qs_sub(struct qs x, struct qs y);
struct qs qs_mul(struct qs x, struct qs y);
struct qs qs_div(struct qs x, struct qs y);
double qs_val(struct qs x);
nid qs_node(struct qs x);
int qs_sqrt(struct qs x, struct qs *r);     /* exact square root in Q(sqrt d) */
void sqrt_split(aq v, aq *k, int64_t *m);   /* sqrt(v) = k sqrt(m), m squarefree */

/* ---- integration (rint.c) ---- */
int quartic_split(const struct up *f, struct qs *p, struct qs *q);
int up_apart(const struct up *n, const struct up *d, struct up *poly,
    void *terms, int max, int *partial);
int alg_check(nid a, nid b, double *worst, int *onlyone);
void alg_vars_reset(void);
nid alg_canon(struct func *f);
nid alg_var_named(uint32_t name);
void alg_cmd1(const char *cmd, nid e, nid var);
void alg_cmd2(const char *cmd, nid a, nid b, nid var);
void alg_subst(nid e, const nid *vars, const nid *vals, int n);
void alg_solve(const nid *eq, const char *const *eqtext, int neq,
    const nid *vars, int nvars);
nid alg_antideriv(nid f, int s, nid x, int first, const char **how);
extern const char *alg_int_note;          /* a note for the caller to print */

/* ---- printing helper (alg.c) ---- */
struct mp mp_pyth(struct mp a, int s, int c, int hyper);
nid alg_tidy(nid e);
nid alg_light(nid e);           /* function rules only */         /* exact rational coefficients, collected */
nid alg_simplify(nid e, struct sbuf *conds);

#endif
