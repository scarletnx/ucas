/*
 * calc: loops and iteration, closed forms, calculus, evaluation schemes.
 *
 * A loop is one node, O_LOOP, over an interned loop entry.  The entry is
 * a small state machine: up to MAXST state variables with initial values
 * and next-state expressions, an index variable for counted loops, or a
 * condition for while loops.  sum, prod, recurrences x[k+1] = ... and
 * code blocks { for ... } all become entries.
 *
 * Variable slots: 0..MAXP-1 are the function's parameters; bound
 * variables of loops (index, states) get slots from MAXP up, inner loops
 * above outer ones, so a loop body can read every enclosing variable.
 * KSLOT is the iteration count k of a closed form S(k).
 *
 * An O_LOOP node: a = lo, b = hi (counted loops: the index runs lo..hi
 * inclusive), c = the index written by the user (recurrences, printing
 * only), ki = entry | result state << 32.
 */
#ifndef CALC_H
#define CALC_H

#include "ucas.h"

#define NSLOT 64
#define MAXST 8
#define KSLOT 63
#define MAXTRIP (1u << 20)   /* iterations per point before giving up */

enum { LK_FOR, LK_WHILE };
enum { LO_SUM, LO_PROD, LO_REC, LO_BLOCK };
#define LF_NEGUNDEF 1        /* hi < lo - 1 is outside the domain */

/* one entry, unpacked (construction and analysis use this view) */
struct loopdef {
	int kind, orig, flags;
	uint32_t name;            /* recurrence name; else the index name */
	int islot;                /* index slot (LK_FOR) */
	ty_t ity;                 /* index type */
	int nst;
	int sslot[MAXST];
	uint32_t sname[MAXST];
	ty_t sty[MAXST];
	nid init[MAXST], next[MAXST];
	nid cond;                 /* LK_WHILE: run while true */
	nid aux;                  /* sum/prod: the body, for printing */
};

/* the entry table, struct of arrays; states are flat at st[e].. */
struct ltab {
	uint8_t *kind, *orig, *flags, *nst, *islot;
	ty_t *ity;
	uint32_t *name, *st;
	nid *cond, *aux;
	uint64_t *bound, *free;   /* slot masks */
	uint32_t n, cap;
	uint8_t *sslot;
	uint32_t *sname;
	ty_t *sty;
	nid *init, *next;
	uint32_t ns, scap;
	uint32_t *ht, htcap;
};
extern struct ltab loops;

/* loop.c */
uint32_t loop_intern(const struct loopdef *d);
void loop_get(uint32_t e, struct loopdef *d);
nid mk_loop(uint32_t e, int res, nid lo, nid hi, nid q);
uint32_t loop_ent(nid n);
int loop_res(nid n);
uint64_t node_slots(nid n);
int has_slot(nid n, int slot);
nid var_of(int slot, uint32_t name, ty_t t);
nid xreplace(nid root, nid from, nid to);
extern uint64_t loop_skipped;
int loop_top_slot(nid n);

/* small helpers shared by the calc files */
#define KI(v) mk_const_i(TY_LIT, (int64_t)(v))
#define KR(v) mk_const_r(v)
#define ADD(a, b) mk_bin(O_ADD, (a), (b))
#define SUB(a, b) mk_bin(O_SUB, (a), (b))
#define MUL(a, b) mk_bin(O_MUL, (a), (b))
#define DIV(a, b) mk_bin(O_DIV, (a), (b))
#define POW(a, b) mk_bin(O_POW, (a), (b))
int num_of(nid n, double *v);      /* numeric constant? */
int int_of(nid n, int64_t *v);     /* integer-valued numeric constant? */

/* cnum.c: rationals, linear algebra, polynomial roots */
struct q {
	int64_t n, d;
};
int q_make(int64_t n, int64_t d, struct q *r);
int q_add(struct q a, struct q b, struct q *r);
int q_sub(struct q a, struct q b, struct q *r);
int q_mul(struct q a, struct q b, struct q *r);
int q_div(struct q a, struct q b, struct q *r);
int q_from_double(double v, struct q *r);
double q_val(struct q a);
nid q_node(struct q a);
int64_t gcd64(int64_t a, int64_t b);
int solve_lin(double *a, double *b, int n);        /* a is n x n, row major */
int solve_lsq(const double *a, const double *b, int m, int n, double *x);
int solve_q(struct q *a, struct q *b, int n);
struct root {
	double re, im;
	int mult;
	int exact;            /* rational: re = qr */
	struct q qr;
	int quad;             /* from a quadratic: (qb ± sqrt(qd)) / qa */
	struct q qb, qd, qa;
};
int poly_roots(const double *c, int deg, struct root *r);
int faulhaber(int p, struct q *c);   /* sum_{i=1}^{n} i^p, coefficients of n^0.. */
nid qpoly_node(const struct q *c, int deg, nid x);
int qpoly_compose_lin(const struct q *c, int deg, struct q a, struct q b,
    struct q *out);

/* closed.c */
struct vres {
	uint64_t n, bad, skipped;
	uint64_t covf, rovf;      /* only the closed form / only the loop overflows */
	int exhaustive;
	double maxrel;
	struct sbuf wit, wcovf, wrovf;
};
void vres_free(struct vres *v);
nid tidy(nid e);
int verify_nodes(const struct func *f, nid ref, nid cand, struct vres *v);
void print_verify(const char *what, const struct vres *v);
nid sum_closed(nid body, int islot, nid lo, nid hi, const char **how);
nid prod_closed(nid body, int islot, nid lo, nid hi, const char **how);
nid loop_closed(nid n, const char **how, struct sbuf *extra);
void cmd_closed(struct func *f);
struct npoly {
	int deg;
	nid c[13];
};
int npoly_of(nid e, int slot, struct npoly *p, int maxdeg);
/* cancel.c: x/x -> 1 etc., appending "f != 0" conditions to conds */
nid cancel_quotients(nid e, int slot, nid x, struct sbuf *conds);
nid antideriv_wrt(nid f, int s, nid xv, const char **how);
int linear_form(nid e, const int *slot, int m, double *M, nid *rest);
void charpoly(const double *M, int n, double *c);
float loop_iter_cost(const struct model *m, nid n);
int lin_rec_matrix(nid n, double *m, double *v0, int *dim, int *res);

/* integ.c */
nid antideriv(nid f, int slot, const char **how);
int gk_integrate(const struct func *f, nid e, int slot, double a, double b,
    double *val, double *err, int *nev);
void cmd_integrate(struct func *f, int slot, nid a, nid b);
double eval_const_at(nid e, int slot, double x, int *ok);

/* ode.c */
void ode_equation(uint32_t var, int order, const char *rhs, uint32_t tok);
void ode_initial(uint32_t var, int order, const char *t0, const char *val);
void cmd_dsolve(uint32_t var, nid t0, nid t1, int steps);
void cmd_continuous(struct func *f);
const char *ode_rhs_text(uint32_t var, int *order);   /* plot.c */
int ode_ic_text(uint32_t var, int order, const char **t0, const char **val);
struct pde {
	uint32_t u, x, t;         /* names */
	int order_t;              /* 1: u_t = ..., 2: u_tt = ... */
	nid rhs;                  /* in rf's parameters, by name */
	struct func rf;
	nid u0, v0;               /* u(x, ta), u_t(x, ta): slot 0 is x */
	nid bl, br;               /* u(xa, t), u(xb, t): slot 0 is t */
	double blx, brx;
	int nbset, periodic, nx;
	double xa, xb, ta, tb;
};
void pde_define(uint32_t name, struct pde *pd);
void cmd_pde(uint32_t name, nid tend, int steps);

/* scheme.c */
void cmd_scheme(struct func *f, struct model *m);
float dag_cost(const struct model *m, nid root, int *nops, float *depth);

/* parsing a source fragment as an expression (calc_syntax.inc) */
nid calc_parse_text(const char *src, struct func *fn, ty_t autoty);

#endif
