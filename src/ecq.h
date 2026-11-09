/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * ecq.h - elliptic curves over Q and Diophantine equations (E1's
 * internal header; the shared core is ec.h).
 *
 *   ec.c       fields, curves, points, isomorphisms, models (ec.h), and
 *              the integer helpers below
 *   ecq.c      curves over Q: integral and minimal models, Tate's
 *              algorithm (Kodaira symbols, c_p, the conductor), torsion,
 *              heights, point search, 2-isogeny descent, generators
 *   ecan.c     the analytic side (floating point, error bounds stated):
 *              periods by AGM, the lattice and Weierstrass p, a_p, a_n,
 *              L(E, 1), L'(E, 1), the root number, the BSD quotient
 *   ecdio.c    plane cubics and quartics to Weierstrass form (maps kept
 *              with their inverses), rational equations in several
 *              variables, the solve loop, the classic families
 *   ec_cmd.c   the commands (ucas syntax), ec_command
 */
#ifndef ECQ_H
#define ECQ_H

#include "ec.h"

/* ---- integer helpers (ec.c) ---- */
zint zx_isqrt(zint a);                    /* floor(sqrt(a)), a >= 0 */
int zx_is_square(zint a, zint *root);     /* a >= 0 a perfect square */
zint zx_iroot(zint a, int k, int *exact); /* floor of the real k-th root
					   * (a < 0 allowed for odd k) */
int zq_is_square(struct zq a, struct zq *root);
zint zx_squarefree_part(zint a, zint *sq);  /* a = part * sq^2, sign in part */
/* integer roots of a monic integer cubic x^3 + a x^2 + b x + c, exact
 * (critical points by isqrt, then bisection); returns the count (with
 * multiplicity collapsed), sorted */
int zx_cubic_int_roots(zint a, zint b, zint c, zint *roots);
/* rational roots of a x^3 + b x^2 + c x + d (a != 0) */
int zq_cubic_roots(struct zq a, struct zq b, struct zq c, struct zq d,
    struct zq *roots);
double zx_to_double(zint a);
double zq_to_double(struct zq a);
double zx_log(zint a);                    /* log |a|, a != 0 */

/* ---- ternary cubics (ec.c): a plane cubic sum c[i] X^e0 Y^e1 Z^e2 ----
 * monomials in the order X^3, X^2Y, X^2Z, XY^2, XYZ, XZ^2, Y^3, Y^2Z, YZ^2,
 * Z^3 (ec_cub_mono gives the exponents) */
extern const unsigned char ec_cub_mono[10][3];
void ec_cub_eval(const struct ecf *F, const fe *c, const fe *P, fe *r);
void ec_cub_grad(const struct ecf *F, const fe *c, const fe *P, fe *g);
/* c'(U) = c(Minv U): the cubic in new coordinates U = M X (Minv given) */
void ec_cub_transform(const struct ecf *F, const fe *c, const fe *Minv,
    fe *out);
/* a cubic with a flex P (and its tangent there) to Weierstrass form:
 * the map is (X:Y:Z) -> U = M (X, Y, Z), x = U0 / (W lam), y = U1 / (W mu)
 * with W = U2; P -> O.  Returns 0 when P is not a smooth flex of c or the
 * cubic is singular. */
struct ec_flexmap {
	fe M[9], Minv[9];
	fe lam, mu;
	fe P[3];
};
int ec_cub_flex_to_w(const struct ecf *F, const fe *c, const fe *P,
    struct ec *E, struct ec_flexmap *m);
int ec_flexmap_fwd(const struct ec *E, const struct ec_flexmap *m,
    const fe *X, struct ecpt *R);
void ec_flexmap_back(const struct ec *E, const struct ec_flexmap *m,
    const struct ecpt *R, fe *X);
/* is P a flex of the cubic (smooth, the tangent meets it three times) */
int ec_cub_is_flex(const struct ecf *F, const fe *c, const fe *P);

/* ================================================================== */
/* curves over Q (ecq.c)                                               */
/* ================================================================== */

/* Kodaira symbols */
enum { KOD_I0 = 1, KOD_IN, KOD_II, KOD_III, KOD_IV, KOD_I0S, KOD_INS,
	KOD_IVS, KOD_IIIS, KOD_IIS };

struct ecq_local {
	zint p;
	int vdisc;      /* v_p of the minimal discriminant */
	int kod, n;     /* Kodaira symbol (n for I_n, I_n*) */
	int f;          /* the conductor exponent */
	int cp;         /* the Tamagawa number */
	int ap;         /* 1 split multiplicative, -1 non-split, 0 additive */
};

/* the integer value of a Q element (fail()s if it is not an integer) */
zint fe_qz(const fe *a);
fe fe_from_z(zint a);
int ecq_is_integral(const struct ec *E);
/* E -> an integral model (u = 1/k) */
void ecq_integral(const struct ec *E, struct ec *Ei, struct eciso *m);
/* E -> the reduced global minimal model (Laska-Kraus-Connell) */
void ecq_minimal(const struct ec *E, struct ec *Em, struct eciso *m);
/* c4, c6 -> a reduced integral model with them (0 if none) */
int ecq_from_c4c6(zint c4, zint c6, struct ec *E);
/* Tate's algorithm at p on an integral model (minimal or not: it scales
 * down when it finds the model non-minimal at p) */
void ecq_tate(const struct ec *E, zint p, struct ecq_local *L);
const char *ecq_kodaira_str(const struct ecq_local *L);
/* the bad primes, local data and conductor of a minimal model */
int ecq_local_all(const struct ec *Em, struct ecq_local *L, int max,
    zint *conductor);
/* #E(F_p) and a_p = p + 1 - #E(F_p) for a good prime p < 2^31 (minimal
 * or p-integral model) */
int64_t ecq_ap(const struct ec *E, int64_t p);
/* torsion: the points (O first) of E(Q)_tors; returns the order; the
 * structure Z/n1 x Z/n2 (n2 = 1 cyclic); gens[0..ngens-1] generate */
struct ecq_tors {
	int n, n1, n2;
	struct ecpt pts[16];
	struct ecpt gen[2];
	int ngen;
	int64_t bound;        /* gcd of #E(F_p) used as the bound */
};
void ecq_torsion(const struct ec *E, struct ecq_tors *T);
/* naive height log max(|num x|, |den x|) and the canonical height
 * (normalised as h(x(nP)) / n^2 -> h^(P), the BSD normalisation), with
 * an error bound for the archimedean series */
double ecq_naive_height(const struct ec *E, const struct ecpt *P);
double ecq_height(const struct ec *E, const struct ecpt *P, double *err);
/* the pairing <P, Q> = (h(P + Q) - h(P) - h(Q)) / 2 and the regulator */
double ecq_height_pairing(const struct ec *E, const struct ecpt *P,
    const struct ecpt *Q);
double ecq_regulator(const struct ec *E, const struct ecpt *P, int n);
/* points of naive height <= log H (x = a / d^2, |a| <= H, d^2 <= H) on
 * an integral model; appends to T (both P and -P); returns the count */
int ecq_search(const struct ec *E, int64_t H, struct ecpts *T, int max);

/* 2-isogeny descent (E with a rational 2-torsion point) */
struct ecq_descent {
	int ok;               /* E has a rational 2-torsion point */
	zint a, b;            /* y^2 = x^3 + a x^2 + b x, the model used */
	struct eciso to;      /* E -> that model */
	int nsel, nsel2;      /* |S^(phi)|, |S^(phi')| as powers of 2 */
	int nimg, nimg2;      /* images of E(Q), E'(Q) found (powers of 2) */
	zint sel[64], sel2[64];   /* Selmer elements (squarefree d) */
	int selfound[64], sel2found[64];
	int rank_lo, rank_hi;
	struct ecpts pts;     /* points on E found by the descent */
	int sha_hint;         /* elements left: Sha[phi] or missed points */
	int positive_only;    /* every element of S^(phi) is > 0 */
};
void ecq_descent2(const struct ec *E, struct ecq_descent *D, int64_t search);
void ecq_descent_free(struct ecq_descent *D);
/* is y^2 = c4 x^4 + c3 x^3 + c2 x^2 + c1 x + c0 soluble over Q_p (and
 * p = 0: over R)? c[0..4] = c0..c4 */
int ecq_quartic_locsol(const zint *c, zint p);

/* the Mordell-Weil group as far as it is determined */
struct ecq_mw {
	struct ec E;          /* the minimal model */
	struct eciso tomin;   /* given -> minimal */
	struct ecq_tors T;
	int rank_lo, rank_hi; /* bounds (rank_hi < 0: unknown) */
	int rank_how;         /* 0 none, 1 2-isogeny descent, 2 analytic */
	struct ecpts gens;    /* independent points of infinite order */
	double reg;
	double h[8];
	int saturated_to;     /* primes <= this checked for saturation */
	struct ecq_descent D;
	int have_descent;
};
void ecq_mw_compute(const struct ec *E, struct ecq_mw *M, int64_t search);
void ecq_mw_free(struct ecq_mw *M);
/* add a point to an independent set if it is independent (heights) */
int ecq_add_indep(const struct ec *E, struct ecpts *G, const struct ecpt *P);
/* enlarge G by Q when Q lies outside its lattice (modulo torsion) */
int ecq_enlarge(const struct ec *E, struct ecpts *G, const struct ecpt *Q);
/* p-saturation of the generators for p <= pmax by reduction; returns 1 if
 * saturated at every p checked (else replaces gens by a larger lattice) */
int ecq_saturate(const struct ec *E, struct ecpts *G, int pmax);

/* ================================================================== */
/* Diophantine equations (ecdio.c)                                     */
/* ================================================================== */

/* polynomials over Q in up to QP_MAXV variables: terms SoA (exponent
 * rows, rational coefficients), combined and without zero terms */
#define QP_MAXV 6
struct qpoly {
	int n, cap;
	uint8_t *e;          /* n * QP_MAXV exponents */
	struct zq *c;
};
struct qrat {
	struct qpoly n, d;
};
/* the variables of the current parse (names) */
struct qvars {
	int nv;
	char name[QP_MAXV][16];
};
void qp_init(struct qpoly *f);
void qp_free(struct qpoly *f);
void qp_copy(struct qpoly *d, const struct qpoly *s);
void qp_const(struct qpoly *f, struct zq c);
void qp_var(struct qpoly *f, int v);
void qp_add(struct qpoly *r, const struct qpoly *a, const struct qpoly *b);
void qp_sub(struct qpoly *r, const struct qpoly *a, const struct qpoly *b);
void qp_mul(struct qpoly *r, const struct qpoly *a, const struct qpoly *b);
void qp_scale(struct qpoly *r, const struct qpoly *a, struct zq c);
int qp_is_zero(const struct qpoly *f);
int qp_deg(const struct qpoly *f);            /* total; -1 for 0 */
int qp_deg_in(const struct qpoly *f, int v);
int qp_homogeneous(const struct qpoly *f);
struct zq qp_eval(const struct qpoly *f, const struct zq *x);
/* exact division: 1 and q = a / b, or 0 */
int qp_divexact(struct qpoly *q, const struct qpoly *a, const struct qpoly *b);
void qp_print(struct sbuf *s, const struct qpoly *f, const struct qvars *V);
/* "a/(b + c) + b/(a + c) + c/(a + b) = 4": lhs - rhs as a rational
 * function, the denominators met (each must not vanish at a solution) */
struct dio_eq {
	struct qvars V;
	struct qrat lhs, rhs;
	struct qpoly F;           /* the polynomial equation F = 0 (integral,
				   * spurious denominator factors removed) */
	struct qpoly den[16];
	int nden;
};
int dio_parse(const char *s, struct dio_eq *D, const char **why);
void dio_free(struct dio_eq *D);

int ecq_real_roots(long double b2, long double b4, long double b6,
    long double *r);
int ecq_egg_empty(const struct ec *E, char *why, size_t n);

/* the solve loop's options and result */
struct dio_opts {
	int positive;             /* every variable > 0 */
	int integral;             /* integer values (affine problems) */
	int nonzero;              /* every variable != 0 */
	int want;                 /* solutions wanted (default 1) */
	int maxmult;              /* coefficients |n_i| <= maxmult */
	int64_t search;           /* point search bound */
	int quiet;
};
/* run the pipeline on an equation and print the derivation; returns the
 * number of solutions found */
int dio_solve(const char *eq, const struct dio_opts *o);
/* the same for a plane cubic given directly (for the families) */
int dio_solve_eq(struct dio_eq *D, const struct dio_opts *o);

#endif
