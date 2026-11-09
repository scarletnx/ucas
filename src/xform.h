/*
 * Transforms, series and sums.
 *
 *   xfrat.c   numbers as text, a printer for transform results (u(t - a),
 *             delta(t), n!), polynomials and rational functions with
 *             double coefficients, partial fractions (xf_apart), the
 *             input rewriting (u[n], delta(t), n!, binom) and evaluation
 *   xform.c   Laplace, Z and Fourier transforms and their inverses, ODEs
 *             and recurrences solved by them, transfer functions, the
 *             DFT, the DTFT, generating functions; the command parser
 *   series.c  infinite sums and products: closed forms, convergence
 *             tests, radius of convergence, power series of a function,
 *             Euler-Maclaurin asymptotics, accelerated numeric values
 *
 * Seams: (1) the complex type being added in ucas-hyper: results here
 * are pairs of real nodes (Re, Im) or text "a + i b", and s, z, omega
 * are real symbolic variables; a transform's value at a complex point is
 * computed in C99 double complex.  (2) alg.c's partial fractions: every
 * inverse transform goes through xf_apart(), which can be replaced by
 * alg.c's apart once it is merged.
 */
#ifndef XFORM_H
#define XFORM_H

#include "ucas.h"
#include "calc.h"

#include <complex.h>

struct P;
int xform_command(struct P *p);    /* parse.c: 1 when it was ours */

/* ---- numbers and printing (xfrat.c) ---- */
const char *xf_num(double v);      /* 3, -1/2, pi^2/6, sqrt(2)/2, 0.1234 */
const char *xf_cnum(double complex z);   /* 1 - 2i */
const char *xf_str(nid n);         /* rotating buffers */
void xf_print(struct sbuf *b, nid n);
extern int xf_discrete;            /* print u[n], delta[n] */

/* nodes, kept real-typed */
nid xk(double v);
nid xadd(nid a, nid b);
nid xsub(nid a, nid b);
nid xmul(nid a, nid b);
nid xdiv(nid a, nid b);
nid xpow(nid a, nid b);
nid xneg(nid a);
nid xexp(nid u);
nid xheav(nid u);                  /* u(x): 1 for x >= 0 */
nid xkron(nid u);                  /* delta[x]: 1 at x = 0 */
nid xdirac(nid u);                 /* delta(x), a marker: 0 except at 0 */
nid xfact(nid m);                  /* m! as a product loop */
int xis_heav(nid n, nid *arg);
int xis_kron(nid n, nid *arg);
int xis_dirac(nid n, nid *arg);
int xis_fact(nid n, nid *arg);
int xis_exp(nid n, nid *u);        /* exp(u), e^u, c^u (u log c) */
nid xsubst1(nid e, int slot, nid v);
nid xv(int slot, const char *name);
int xnum(nid e, double *v);        /* a numeric constant (folded) */

/* evaluation: e with slots[] set to x[]; ok 0 where undefined */
double xf_eval(nid e, const int *slots, const double *x, int nx, int *ok);
double xf_eval1(nid e, int slot, double x, int *ok);
/* many points of one slot, the others fixed */
void xf_evalv(nid e, int slot, const double *xs, int n, const int *fs,
    const double *fx, int nf, double *out, uint8_t *ok);

/* ---- polynomials, rational functions ---- */
#define XD 40
struct xpoly {
	int deg;                       /* -1 for 0 */
	double c[XD + 1];
};
struct xrat {
	struct xpoly n, d;
};
void xp_zero(struct xpoly *p);
void xp_const(struct xpoly *p, double c);
void xp_trim(struct xpoly *p);
int xp_add(const struct xpoly *a, const struct xpoly *b, double sb,
    struct xpoly *r);
int xp_mul(const struct xpoly *a, const struct xpoly *b, struct xpoly *r);
void xp_scale(struct xpoly *p, double k);
void xp_deriv(const struct xpoly *a, struct xpoly *r);
double xp_eval(const struct xpoly *p, double x);
double complex xp_ceval(const struct xpoly *p, double complex x);
int xp_divmod(const struct xpoly *a, const struct xpoly *b, struct xpoly *q,
    struct xpoly *r);
int xp_of(nid e, int slot, struct xpoly *p);   /* polynomial in slot */
nid xp_node(const struct xpoly *p, nid x);
int xq_of(nid e, int slot, struct xrat *r);    /* rational in slot */
int xq_add(const struct xrat *a, const struct xrat *b, double sb,
    struct xrat *r);
int xq_mul(const struct xrat *a, const struct xrat *b, struct xrat *r);
int xq_div(const struct xrat *a, const struct xrat *b, struct xrat *r);
int xq_deriv(const struct xrat *a, struct xrat *r);
void xq_reduce(struct xrat *r);    /* cancel common roots, monic den */
double xq_eval(const struct xrat *r, double x);
double complex xq_ceval(const struct xrat *r, double complex x);
nid xq_node(const struct xrat *r, nid x);      /* factored denominator */
/* the denominator's factors as text: "(s + 1)^3 (s^2 + 1)" */

/*
 * Partial fractions: r = poly + sum over poles p (one per conjugate
 * pair, im >= 0) of sum_k a[k]/(x - p)^k, k = 1..m.
 * THE SEAM: replace by alg.c's apart once it is merged.
 */
struct xpole {
	double complex p;
	int m;
	double complex a[XD + 2];      /* a[1..m] */
};
struct xpf {
	struct xpoly poly;
	int np;
	struct xpole pole[XD];
};
int xf_apart(const struct xrat *r, struct xpf *pf);
int xf_roots(const struct xpoly *p, double complex *z, int *mult);

/* ---- input ---- */
/* u(t-a), H, step, theta -> heaviside; delta(t), dirac -> dirac;
 * u[n], delta[n] -> heaviside(n), kron(n); n! -> fact(n); binom */
void xf_prep(const char *in, struct sbuf *out);
void xf_helpers(void);             /* define them (before any parsing) */
nid xf_fix(nid e);                 /* fact placeholders to loops */
nid xf_nodirac(nid e);             /* delta(...) -> 0, for numerics */
struct func *xf_parse(const char *text);   /* prep, parse, fix */
int xf_slot(const struct func *f, const char *name);

/* tgamma for printing: Gamma(p) as text (sqrt(pi)/2 ...) */
double xf_gamma(double x);
double complex xf_cgamma(double complex z);   /* Lanczos */
double complex xf_cdigamma(double complex z);
double xf_zeta(double s);          /* s > 1 */
double xf_hurwitz(double s, double a);

/* ---- series.c ---- */
int series_command(struct P *p, const char *w);
/* recognise a constant: "pi^2/6", NULL if nothing short */
const char *xf_ident(double v);

/* ---- xform.c shared ---- */
void xf_head(const char *fmt, ...);

#endif
