/*
 * ec.h - elliptic curves: the shared core (elip.md).
 *
 * This header is the contract between the curves-over-Q side (E1: ec.c,
 * ecq*.c, branch elip) and the finite-field side (E2: branch elip-ff).
 * Changes are additive only.
 *
 *   ecf     a field (or Z/n) behind a small table of operations, with
 *           elements as fixed-size values (fe): Q (exact, zint
 *           rationals), F_p for any p (zint residues), Z/n (inversion can
 *           fail and then yields a factor: ECM), GF(p^k) through ring.h's
 *           u64 codes, and room for other implementations (ECF_USER: E2's
 *           fixed-width Montgomery fields keep their limbs in fe.w)
 *   ec      a curve in general Weierstrass form
 *               y^2 + a1 xy + a3 y = x^3 + a2 x^2 + a4 x + a6
 *           with b2 b4 b6 b8 c4 c6, the discriminant and j
 *   ecpt    an affine point or O; the group law (chord and tangent),
 *           negation, multiples; ecpts a struct-of-arrays point table
 *   eciso   the standard isomorphisms [u, r, s, t]:
 *               x = u^2 x' + r,  y = u^3 y' + s u^2 x' + t
 *   ecmodel other models (short Weierstrass, Montgomery, Edwards,
 *           twisted Edwards, Legendre, twisted Hessian) with the maps to
 *           and from a Weierstrass curve, both directions explicit
 *
 * Conventions:
 *
 * - Every operation takes the field first; results may alias operands.
 * - Elements of fields built on zint (Q, F_p, Z/n) hold zint handles, so
 *   they live in absalg's arena: z_mark() before temporaries and
 *   fe_keep(F, mark, elts, n) (z_release_keep underneath) to drop
 *   everything made since the mark except the given elements.  For
 *   other fields fe_keep does nothing.
 * - Errors in the input (a singular curve where a nonsingular one is
 *   needed, a denominator that is not invertible) fail() (ucas.h); a
 *   mathematical outcome (no square root, an inversion that failed in
 *   Z/n) is a return value.
 * - Exact everywhere.  Nothing here uses floating point.
 *
 * Commands: parse.c calls ec_command (curves over Q, Diophantine
 * equations: ec_cmd.c).  It declines statements whose curve is given
 * over a finite field or mod n ("... mod p", "... over GF(q)") so that a
 * later hook (E2's) can take them.
 */
#ifndef EC_H
#define EC_H

#include <stddef.h>
#include <stdint.h>

#include "absalg.h"

struct sbuf;
struct P;

/* ================================================================== */
/* fields                                                              */
/* ================================================================== */

/* a field element: 9 words hold a 521-bit fixed-width residue; Q uses
 * w[0] = numerator, w[1] = denominator (zint handles, den > 0, reduced),
 * F_p and Z/n w[0] = the residue in [0, p) (a zint), GF(p^k) w[0] = the
 * ring.h code.  Unused words are 0, so memcmp is NOT equality for Q
 * after arena moves: use fe_eq. */
#define EC_FW 9
typedef struct {
	uint64_t w[EC_FW];
} fe;

enum {
	ECF_Q = 1,      /* the rationals */
	ECF_FP,         /* Z/p, p prime, any size (zint) */
	ECF_ZN,         /* Z/n, n composite allowed (ECM) */
	ECF_FQ,         /* GF(p^k) (or Z/p, p < 2^64) through ring.h */
	ECF_USER        /* an implementation outside ec.c */
};

struct ecf;
struct ecf_ops {
	const char *name;
	void (*from_i64)(const struct ecf *F, fe *r, int64_t v);
	/* the image of a rational; 0 when the denominator is not invertible
	 * (in Z/n with a factor found, F->factor is set) */
	int (*from_zq)(const struct ecf *F, fe *r, struct zq a);
	void (*add)(const struct ecf *F, fe *r, const fe *a, const fe *b);
	void (*sub)(const struct ecf *F, fe *r, const fe *a, const fe *b);
	void (*neg)(const struct ecf *F, fe *r, const fe *a);
	void (*mul)(const struct ecf *F, fe *r, const fe *a, const fe *b);
	/* 1/a; 0 when a is not a unit (0, or a zero divisor in Z/n: then
	 * F->factor holds gcd(a, n) when it is a proper factor) */
	int (*inv)(const struct ecf *F, fe *r, const fe *a);
	int (*is_zero)(const struct ecf *F, const fe *a);
	int (*eq)(const struct ecf *F, const fe *a, const fe *b);
	/* a square root; 1 and *r, 0 when a is not a square; may be NULL
	 * (unsupported) */
	int (*sqrt)(const struct ecf *F, fe *r, const fe *a);
	/* the element as a rational: Q its value, F_p/Z/n the residue in
	 * [0, p), GF(p^k) the code when it lies in the prime field (else 0) */
	int (*to_zq)(const struct ecf *F, struct zq *r, const fe *a);
	void (*print)(const struct ecf *F, struct sbuf *b, const fe *a);
	/* text -> element ("3", "-3/4", GF: "0x57"); 0 on a syntax error */
	int (*parse)(const struct ecf *F, fe *r, const char *s,
	    const char **end);
	/* drop arena values made after m except those in a[0..n-1]; NULL
	 * when the elements are not arena values */
	void (*keep)(const struct ecf *F, zmark m, fe *a, size_t n);
};

struct ecf {
	int kind;               /* ECF_* */
	zint p;                 /* the characteristic (Q: 0; Z/n: n) */
	int deg;                /* GF(p^deg); 1 otherwise */
	struct ring *R;         /* ECF_FQ: ring.h's ring (u64 codes) */
	void *ctx;              /* ECF_USER: the implementation's state */
	zint factor;            /* ECF_ZN: a proper factor of n, found by a
	                         * failed inversion (0 until then) */
	const struct ecf_ops *op;
};

void ecf_q(struct ecf *F);
void ecf_fp(struct ecf *F, zint p);      /* p prime (checked: z_isprime) */
void ecf_zn(struct ecf *F, zint n);      /* n >= 2 */
void ecf_fq(struct ecf *F, struct ring *R);  /* R a field (R->field) */
const char *ecf_name(const struct ecf *F);   /* "Q", "F_101", "Z/91", "GF(2^8)" */

/* element helpers (thin wrappers over F->op) */
void fe_zero(const struct ecf *F, fe *r);
void fe_one(const struct ecf *F, fe *r);
void fe_set_i64(const struct ecf *F, fe *r, int64_t v);
int fe_set_zq(const struct ecf *F, fe *r, struct zq a);
int fe_set_z(const struct ecf *F, fe *r, zint a);
void fe_add(const struct ecf *F, fe *r, const fe *a, const fe *b);
void fe_sub(const struct ecf *F, fe *r, const fe *a, const fe *b);
void fe_neg(const struct ecf *F, fe *r, const fe *a);
void fe_mul(const struct ecf *F, fe *r, const fe *a, const fe *b);
void fe_sqr(const struct ecf *F, fe *r, const fe *a);
void fe_mul_i64(const struct ecf *F, fe *r, const fe *a, int64_t k);
int fe_inv(const struct ecf *F, fe *r, const fe *a);
int fe_div(const struct ecf *F, fe *r, const fe *a, const fe *b);  /* 0: b not a unit */
void fe_pow(const struct ecf *F, fe *r, const fe *a, uint64_t e);
int fe_is_zero(const struct ecf *F, const fe *a);
int fe_is_one(const struct ecf *F, const fe *a);
int fe_eq(const struct ecf *F, const fe *a, const fe *b);
int fe_sqrt(const struct ecf *F, fe *r, const fe *a);   /* 0: none/unsupported */
int fe_to_zq(const struct ecf *F, struct zq *r, const fe *a);
void fe_print(const struct ecf *F, struct sbuf *b, const fe *a);
const char *fe_str(const struct ecf *F, const fe *a);   /* rotating static */
int fe_parse(const struct ecf *F, fe *r, const char *s, const char **end);
void fe_keep(const struct ecf *F, zmark m, fe *a, size_t n);
/* Q only: the rational in an element, and back */
struct zq fe_q(const fe *a);
fe fe_from_q(struct zq a);

/* ================================================================== */
/* curves                                                              */
/* ================================================================== */

struct ec {
	const struct ecf *F;
	fe a1, a2, a3, a4, a6;
	fe b2, b4, b6, b8, c4, c6;
	fe disc;                /* the discriminant */
	fe j;                   /* c4^3 / disc when disc is a unit */
	int singular;           /* disc == 0 */
	int jok;                /* j was computed (disc invertible) */
};

/* a[] = a1, a2, a3, a4, a6.  The invariants are computed; a singular
 * curve is allowed (E->singular) so that reductions can be examined */
void ec_init(struct ec *E, const struct ecf *F, const fe a[5]);
void ec_init_i64(struct ec *E, const struct ecf *F, const int64_t a[5]);
void ec_init_short(struct ec *E, const struct ecf *F, const fe *a4,
    const fe *a6);          /* y^2 = x^3 + a4 x + a6 */
/* reduce a curve over Q (integral coefficients) to F: 0 when a
 * coefficient's denominator is not invertible there */
int ec_reduce(struct ec *r, const struct ec *Eq, const struct ecf *F);
void ec_coeffs(const struct ec *E, fe a[5]);
/* the equation's value at (x, y): lhs - rhs */
void ec_eval(const struct ec *E, fe *r, const fe *x, const fe *y);
/* "y^2 + x*y + y = x^3 - x^2 - 10*x - 20" */
void ec_print(struct sbuf *b, const struct ec *E);
/* "[a1, a2, a3, a4, a6]" */
void ec_print_ainv(struct sbuf *b, const struct ec *E);
/* the same curve keeping its field; dst may not alias src */
void ec_copy(struct ec *dst, const struct ec *src);
/* keep the curve's arena values (zint fields) past a release */
void ec_keep(struct ec *E, zmark m);

/* ================================================================== */
/* points and the group law                                            */
/* ================================================================== */

struct ecpt {
	fe x, y;
	int inf;                /* 1: the point at infinity O */
};

void ec_O(struct ecpt *P);
void ec_pt(const struct ec *E, struct ecpt *P, const fe *x, const fe *y);
int ec_on(const struct ec *E, const struct ecpt *P);   /* on the curve */
int ec_pt_eq(const struct ec *E, const struct ecpt *P, const struct ecpt *Q);
void ec_neg(const struct ec *E, struct ecpt *R, const struct ecpt *P);
/* R = P + Q, R = 2P, R = P - Q.  They return 1, or 0 when an inversion
 * failed (only in Z/n: E->F->factor then holds a factor, R is O) */
int ec_add(const struct ec *E, struct ecpt *R, const struct ecpt *P,
    const struct ecpt *Q);
int ec_dbl(const struct ec *E, struct ecpt *R, const struct ecpt *P);
int ec_sub(const struct ec *E, struct ecpt *R, const struct ecpt *P,
    const struct ecpt *Q);
/* R = k P for any integer k (negative allowed), double and add */
int ec_mul(const struct ec *E, struct ecpt *R, zint k, const struct ecpt *P);
int ec_mul_i64(const struct ec *E, struct ecpt *R, int64_t k,
    const struct ecpt *P);
/* the order of P when it is at most max (1 for O), else 0 */
int64_t ec_order_upto(const struct ec *E, const struct ecpt *P, int64_t max);
/* the points with given x (0, 1 or 2 of them); needs F->op->sqrt for
 * the general case (a1 x + a3 = 0 or char 2 aside) */
int ec_lift_x(const struct ec *E, const fe *x, struct ecpt *out);
/* "(x, y)", "O" */
void ec_pt_print(struct sbuf *b, const struct ec *E, const struct ecpt *P);
const char *ec_pt_str(const struct ec *E, const struct ecpt *P);
/* "(1, 2)", "(1/4, -3/8)", "O", "[1, 2]" in the curve's field */
int ec_pt_parse(const struct ec *E, struct ecpt *P, const char *s,
    const char **end);
void ec_pt_keep(const struct ec *E, zmark m, struct ecpt *P, size_t n);

/* a struct-of-arrays table of points on one curve */
struct ecpts {
	uint32_t n, cap;
	fe *x, *y;
	uint8_t *inf;
};
void ecpts_init(struct ecpts *T);
void ecpts_free(struct ecpts *T);
uint32_t ecpts_push(struct ecpts *T, const struct ecpt *P);
void ecpts_get(const struct ecpts *T, uint32_t i, struct ecpt *P);
void ecpts_set(struct ecpts *T, uint32_t i, const struct ecpt *P);
/* index of P in T, or NOTOK-like UINT32_MAX */
uint32_t ecpts_find(const struct ec *E, const struct ecpts *T,
    const struct ecpt *P);
void ecpts_keep(const struct ec *E, zmark m, struct ecpts *T);

/* ================================================================== */
/* isomorphisms [u, r, s, t]                                           */
/* ================================================================== */

/* x = u^2 x' + r, y = u^3 y' + s u^2 x' + t: E (x, y) -> E' (x', y') */
struct eciso {
	fe u, r, s, t;
};
void eciso_identity(const struct ecf *F, struct eciso *m);
/* E' = the image curve (a-invariants transformed); u must be a unit */
void eciso_apply(const struct ec *E, const struct eciso *m, struct ec *Ep);
void eciso_pt(const struct ec *E, const struct eciso *m, struct ecpt *R,
    const struct ecpt *P);      /* E -> E' */
void eciso_pt_inv(const struct ec *E, const struct eciso *m,
    struct ecpt *R, const struct ecpt *Pp);   /* E' -> E */
/* m = m2 o m1 (first m1: E -> E1, then m2: E1 -> E2) */
void eciso_compose(const struct ecf *F, struct eciso *m,
    const struct eciso *m1, const struct eciso *m2);
void eciso_inverse(const struct ecf *F, struct eciso *r,
    const struct eciso *m);
/* to the short model y^2 = x^3 - 27 c4 x - 54 c6 (char not 2 or 3;
 * u = 1/6, r = -b2/12, s = -a1/2, t = -a3/2 - r a1 / 2 ... composed),
 * integral when E is */
void eciso_to_short(const struct ec *E, struct eciso *m, struct ec *Es);
/* an isomorphism E1 -> E2 when j agree and one exists over F (Q: j not
 * 0 or 1728 or the twist matches); 0 when not isomorphic over F */
int eciso_find(const struct ec *E1, const struct ec *E2, struct eciso *m);
void eciso_print(struct sbuf *b, const struct ecf *F, const struct eciso *m);

/* ================================================================== */
/* other models                                                        */
/* ================================================================== */

enum {
	ECM_WEIERSTRASS = 1,   /* the general form, c[0..4] = a1 a2 a3 a4 a6 */
	ECM_SHORT,             /* y^2 = x^3 + A x + B          c = A, B */
	ECM_MONTGOMERY,        /* B y^2 = x^3 + A x^2 + x      c = A, B */
	ECM_EDWARDS,           /* x^2 + y^2 = 1 + d x^2 y^2    c = d */
	ECM_TWISTED_EDWARDS,   /* a x^2 + y^2 = 1 + d x^2 y^2  c = a, d */
	ECM_LEGENDRE,          /* y^2 = x (x - 1) (x - l)      c = l */
	ECM_HESSIAN            /* a x^3 + y^3 + 1 = d x y      c = a, d
	                        * (a = 1: the Hessian x^3+y^3+1 = 3k xy, d=3k) */
};

struct ecmodel {
	int kind;
	const struct ecf *F;
	fe c[5];
};

/* a point of a model: affine (x, y); inf marks the model's points that
 * have no affine chart (Montgomery's O; Hessian's points at infinity
 * x : y : 0 with x the given value, see ec.c) */
struct ecmpt {
	fe x, y;
	int inf;
};

const char *ecm_kind_name(int kind);      /* "montgomery", ... */
int ecm_kind_parse(const char *s);        /* 0: unknown */
void ecm_print(struct sbuf *b, const struct ecmodel *M);  /* the equation */
int ecm_on(const struct ecmodel *M, const struct ecmpt *P);
/* the model's neutral element */
void ecm_neutral(const struct ecmodel *M, struct ecmpt *P);
/* the Weierstrass curve of a model and the birational maps both ways:
 * model -> W and W -> model; exceptional points (where the rational
 * formula has a pole) are mapped by their limits.  0 when the model is
 * singular (or, for ECM_SHORT and Hessian, the characteristic does not
 * allow the formulas) */
int ecm_to_weierstrass(const struct ecmodel *M, struct ec *E);
int ecm_pt_to_w(const struct ecmodel *M, const struct ec *E,
    const struct ecmpt *P, struct ecpt *R);
int ecm_pt_from_w(const struct ecmodel *M, const struct ec *E,
    const struct ecpt *P, struct ecmpt *R);
/* a model of the given kind for E when one exists over E's field (a
 * Montgomery form needs a 2-torsion point (x0, *) with 3 x0^2 + 2 a2 x0 +
 * a4 a square after completing the square, Edwards needs Montgomery
 * plus a square, Legendre full 2-torsion and a square, ...).  0: none.
 * The maps are then ecm_pt_to_w / ecm_pt_from_w composed with *iso:
 * E --iso--> ecm_to_weierstrass(M). */
int ecm_from_weierstrass(int kind, const struct ec *E, struct ecmodel *M,
    struct eciso *iso);
/* the model's group law, through the Weierstrass curve (exact) */
int ecm_add(const struct ecmodel *M, struct ecmpt *R, const struct ecmpt *P,
    const struct ecmpt *Q);

/* ================================================================== */
/* commands (ec_cmd.c): parse.c calls ec_command                       */
/* ================================================================== */

int ec_command(struct P *p);

#endif
