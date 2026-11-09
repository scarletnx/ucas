/*
 * ecff.h - elliptic curves over finite fields: arithmetic, point counting,
 * cryptography and algorithms (E2 of elip.md).
 *
 * Layers, bottom up:
 *
 *   nn    fixed-width naturals (NW limbs = 1152 bits), with length-generic
 *         limb routines underneath (l_*), and zz = sign + nn     ecmp.c
 *   mf    Z/n in Montgomery form, n odd and up to FW = 9 limbs (521-bit
 *         fields, ECM and ECPP moduli); prime fields F_p        ecfp.c
 *   fk    F_{p^k} = F_p[x]/(m), k <= FKMAX, elements k*n limbs ecfp.c
 *   px    polynomials over F_p (arena-backed coefficient rows) ecpoly.c
 *   b2    binary fields GF(2^m), m <= 571, and binary curves  ecb2.c
 *   ew    curves y^2 = x^3 + a x + b over F_q (q = p or p^k), affine and
 *         Jacobian, wNAF, ladder; Montgomery and Edwards models ecw.c
 *
 * Elements are plain u64 limb arrays (little-endian limbs) in Montgomery
 * form; a field context says how many limbs are live.  Batches of
 * elements are limb columns (limb j of item i at a[j * stride + i]), so a
 * table of points is a few flat arrays.  Nothing here allocates per
 * element; scratch polynomials come from one arena with mark/release.
 *
 * This is mathematics for a CAS: the code is variable-time and makes no
 * claim to resist side channels.
 */
#ifndef ECFF_H
#define ECFF_H

#include <stddef.h>
#include <stdint.h>

#ifndef RING_H
typedef uint64_t u64;
typedef unsigned __int128 u128;
typedef __int128 i128;
#endif

/* ---- nn: naturals ---- */
#define NW 18
typedef struct {
	u64 w[NW];
} nn;
typedef struct {
	int neg;
	nn m;
} zz;

/* length-generic limb routines (n limbs each unless stated) */
u64 l_add(u64 *r, const u64 *a, const u64 *b, int n);
u64 l_sub(u64 *r, const u64 *a, const u64 *b, int n);
int l_cmp(const u64 *a, const u64 *b, int n);
int l_len(const u64 *a, int n);           /* limbs up to the top nonzero */
void l_mul(u64 *r, const u64 *a, int na, const u64 *b, int nb); /* r: na+nb */
/* q (na - nb + 1 limbs, or NULL) and r (nb limbs, or NULL); b's top limb
 * nonzero; fail()s on b = 0 */
void l_divmod(u64 *q, u64 *r, const u64 *a, int na, const u64 *b, int nb);
int l_bits(const u64 *a, int n);

void nn_zero(nn *a);
void nn_set(nn *a, u64 v);
int nn_is_zero(const nn *a);
int nn_is_one(const nn *a);
int nn_cmp(const nn *a, const nn *b);
int nn_cmp_u64(const nn *a, u64 v);
int nn_eq(const nn *a, const nn *b);
int nn_bits(const nn *a);
int nn_bit(const nn *a, int i);
int nn_fits_u64(const nn *a);
u64 nn_add(nn *r, const nn *a, const nn *b);       /* carry */
u64 nn_sub(nn *r, const nn *a, const nn *b);       /* borrow */
void nn_add_u64(nn *r, const nn *a, u64 v);
void nn_sub_u64(nn *r, const nn *a, u64 v);
int nn_mul(nn *r, const nn *a, const nn *b);       /* 0: overflow */
int nn_mul_u64(nn *r, const nn *a, u64 v);
void nn_shl(nn *r, const nn *a, int k);
void nn_shr(nn *r, const nn *a, int k);
void nn_divmod(nn *q, nn *r, const nn *a, const nn *b);
u64 nn_div_u64(nn *q, const nn *a, u64 m);         /* the remainder */
u64 nn_mod_u64(const nn *a, u64 m);
void nn_mod(nn *r, const nn *a, const nn *m);
void nn_gcd(nn *g, const nn *a, const nn *b);
int nn_invmod(nn *r, const nn *a, const nn *m);    /* 0: no inverse */
void nn_mulmod(nn *r, const nn *a, const nn *b, const nn *m);
void nn_addmod(nn *r, const nn *a, const nn *b, const nn *m);
void nn_submod(nn *r, const nn *a, const nn *b, const nn *m);
void nn_powmod(nn *r, const nn *a, const nn *e, const nn *m);
void nn_pow_u64(nn *r, const nn *a, u64 e);        /* fail()s on overflow */
void nn_sqrt(nn *r, const nn *a);                  /* floor */
int nn_is_square(const nn *a, nn *root);
int nn_jacobi(const nn *a, const nn *n);           /* n odd */
int nn_isprime(const nn *a);   /* BPSW: exact below 2^64, else probable */
void nn_nextprime(nn *r, const nn *a);             /* smallest prime >= a */
void nn_rand_below(nn *r, const nn *bound);        /* deterministic PRNG */
void nn_seed(u64 s);
u64 nn_rand64(void);
/* "123", "0x1f", "2^255 - 19", "(2^127-1)*3" */
int nn_parse(const char *s, nn *out, const char **end);
const char *nn_str(const nn *a);                   /* rotating buffers */
const char *nn_hex(const nn *a);                   /* 0x... */
void nn_from_bytes_be(nn *r, const unsigned char *b, size_t n);
void nn_to_bytes_be(unsigned char *b, size_t n, const nn *a);
void nn_from_bytes_le(nn *r, const unsigned char *b, size_t n);
void nn_to_bytes_le(unsigned char *b, size_t n, const nn *a);

void zz_set_i64(zz *a, int64_t v);
void zz_from_nn(zz *a, const nn *m, int neg);
void zz_add(zz *r, const zz *a, const zz *b);
void zz_sub(zz *r, const zz *a, const zz *b);
int zz_mul(zz *r, const zz *a, const zz *b);
void zz_mod(nn *r, const zz *a, const nn *m);      /* Euclidean residue */
int zz_cmp(const zz *a, const zz *b);
int zz_is_zero(const zz *a);
int zz_parse(const char *s, zz *out, const char **end);
const char *zz_str(const zz *a);
int64_t zz_to_i64(const zz *a, int *ok);

/* factorisation: trial division, Pollard-Brent rho, then ECM (ecalg.c);
 * every factor checked by multiplying back; prime[i] = 1 when BPSW says
 * prime, 0 when the search gave up on a composite */
#define NNFMAX 64
struct nnfac {
	int k;
	nn p[NNFMAX];
	int e[NNFMAX];
	int prime[NNFMAX];
	int complete;     /* every p is a (BPSW) prime */
};
void nn_factor(const nn *n, struct nnfac *f, int effort);
void nnfac_print(char *buf, size_t sz, const struct nnfac *f);
int nnfac_check(const struct nnfac *f, const nn *n);   /* product = n */

/* ---- mf: Z/n in Montgomery form ---- */
#define FW 9
struct mf {
	int n;              /* live limbs */
	int bits;
	u64 p[FW];
	u64 pinv;           /* -p^-1 mod 2^64 */
	u64 r2[FW];         /* R^2 mod p */
	u64 one[FW];        /* R mod p */
	nn P;
	int prime;          /* p is a (BPSW) prime */
	/* square roots: p - 1 = 2^s q */
	int s;
	nn q;
	u64 nonres[FW];     /* a quadratic non-residue (Montgomery) */
};
int mf_init(struct mf *F, const nn *p);   /* 0: even or too big */
/* BPSW-test p; for a prime, set up square roots (s, q, nonres) */
int mf_prime_setup(struct mf *F);
void mf_from_nn(const struct mf *F, u64 *r, const nn *a);  /* a mod p */
void mf_from_u64(const struct mf *F, u64 *r, u64 v);
void mf_from_i64(const struct mf *F, u64 *r, int64_t v);
void mf_from_zz(const struct mf *F, u64 *r, const zz *a);
void mf_to_nn(const struct mf *F, nn *r, const u64 *a);
void mf_copy(const struct mf *F, u64 *r, const u64 *a);
void mf_zero(const struct mf *F, u64 *r);
void mf_one(const struct mf *F, u64 *r);
int mf_is_zero(const struct mf *F, const u64 *a);
int mf_is_one(const struct mf *F, const u64 *a);
int mf_eq(const struct mf *F, const u64 *a, const u64 *b);
void mf_add(const struct mf *F, u64 *r, const u64 *a, const u64 *b);
void mf_sub(const struct mf *F, u64 *r, const u64 *a, const u64 *b);
void mf_neg(const struct mf *F, u64 *r, const u64 *a);
void mf_dbl(const struct mf *F, u64 *r, const u64 *a);
void mf_mul(const struct mf *F, u64 *r, const u64 *a, const u64 *b);
void mf_sqr(const struct mf *F, u64 *r, const u64 *a);
void mf_mul_u64(const struct mf *F, u64 *r, const u64 *a, u64 k);
void mf_pow(const struct mf *F, u64 *r, const u64 *a, const nn *e);
void mf_pow_l(const struct mf *F, u64 *r, const u64 *a, const u64 *e,
    int ne);
/* 1 and r = a^-1, or 0 and g = gcd(a, p) (as an integer) when not a unit */
int mf_inv(const struct mf *F, u64 *r, const u64 *a, nn *g);
int mf_legendre(const struct mf *F, const u64 *a);   /* p prime */
int mf_sqrt(const struct mf *F, u64 *r, const u64 *a);  /* 0: none */
const char *mf_str(const struct mf *F, const u64 *a);
const char *mf_hex(const struct mf *F, const u64 *a);
/* columns: limb j of item i at x[j * stride + i] */
void mfv_get(const struct mf *F, u64 *e, const u64 *col, size_t stride,
    size_t i);
void mfv_put(const struct mf *F, u64 *col, size_t stride, size_t i,
    const u64 *e);
/* invert cnt items of a column in place (Montgomery's trick); 0 and g if
 * one is not a unit */
int mfv_inv(const struct mf *F, u64 *col, size_t stride, size_t cnt, nn *g);

/* ---- fk: F_{p^k} = F_p[x]/(m) ---- */
#define FKMAX 12
#define FKE (FKMAX * FW)
struct fk {
	const struct mf *F;
	int k;
	int n;              /* F->n */
	u64 m[(FKMAX + 1) * FW];   /* the monic modulus' coefficients */
	int binom;          /* m = x^k - c: c = -m[0] */
};
void fk_init(struct fk *K, const struct mf *F, int k, const u64 *mod);
int fk_init_auto(struct fk *K, const struct mf *F, int k); /* finds one */
void fk_zero(const struct fk *K, u64 *r);
void fk_one(const struct fk *K, u64 *r);
void fk_copy(const struct fk *K, u64 *r, const u64 *a);
void fk_from_mf(const struct fk *K, u64 *r, const u64 *a);
int fk_is_zero(const struct fk *K, const u64 *a);
int fk_is_one(const struct fk *K, const u64 *a);
int fk_eq(const struct fk *K, const u64 *a, const u64 *b);
void fk_add(const struct fk *K, u64 *r, const u64 *a, const u64 *b);
void fk_sub(const struct fk *K, u64 *r, const u64 *a, const u64 *b);
void fk_neg(const struct fk *K, u64 *r, const u64 *a);
void fk_mul(const struct fk *K, u64 *r, const u64 *a, const u64 *b);
void fk_sqr(const struct fk *K, u64 *r, const u64 *a);
int fk_inv(const struct fk *K, u64 *r, const u64 *a);
void fk_pow_l(const struct fk *K, u64 *r, const u64 *a, const u64 *e,
    int ne);
void fk_frob(const struct fk *K, u64 *r, const u64 *a);   /* a^p */
void fk_print(const struct fk *K, char *buf, size_t sz, const u64 *a);
/* q - 1 = p^k - 1 as limbs (at most FKMAX * FW + 1) */
int fk_qm1(const struct fk *K, u64 *e);
int fk_sqrt(const struct fk *K, u64 *r, const u64 *a);

/* ---- ecw.c: short Weierstrass curves (ecw_impl.inc, twice) ---- */
#define ECW_API(P, FT, N) \
struct P##_curve { \
	const FT *K; \
	u64 a[N], b[N]; \
	int a_zero, a_m3; \
}; \
struct P##_pt { \
	u64 x[N], y[N]; \
	int inf; \
}; \
struct P##_jpt { \
	u64 X[N], Y[N], Z[N]; \
}; \
void P##_set(struct P##_curve *E, const FT *K, const u64 *a, const u64 *b); \
void P##_disc(const struct P##_curve *E, u64 *d); \
int P##_jinv(const struct P##_curve *E, u64 *j); \
void P##_rhs(const struct P##_curve *E, u64 *r, const u64 *x); \
int P##_on_curve(const struct P##_curve *E, const struct P##_pt *P); \
void P##_set_inf(const struct P##_curve *E, struct P##_pt *P); \
int P##_eq(const struct P##_curve *E, const struct P##_pt *P, \
    const struct P##_pt *Q); \
void P##_neg(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P); \
void P##_add(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, const struct P##_pt *Q); \
void P##_sub(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, const struct P##_pt *Q); \
void P##_to_jac(const struct P##_curve *E, struct P##_jpt *J, \
    const struct P##_pt *P); \
void P##_from_jac(const struct P##_curve *E, struct P##_pt *P, \
    const struct P##_jpt *J); \
void P##_jdbl(const struct P##_curve *E, struct P##_jpt *R, \
    const struct P##_jpt *P); \
void P##_jadd(const struct P##_curve *E, struct P##_jpt *R, \
    const struct P##_jpt *P, const struct P##_jpt *Q); \
int P##_jis_inf(const struct P##_curve *E, const struct P##_jpt *P); \
void P##_mul_affine(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, const nn *k); \
void P##_mul_ladder(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, const nn *k); \
void P##_mul_wnaf(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, const nn *k, int w); \
void P##_mul(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, const nn *k); \
void P##_mul_u64(const struct P##_curve *E, struct P##_pt *R, \
    const struct P##_pt *P, u64 k); \
int P##_lift_x(const struct P##_curve *E, struct P##_pt *P, const u64 *x); \
void P##_random_point(const struct P##_curve *E, struct P##_pt *P);

ECW_API(ecp, struct mf, FW)
ECW_API(eck, struct fk, FKE)
void fk_small(const struct fk *K, u64 *r, int64_t v);
void fk_rand(const struct fk *K, u64 *r);
void mf_rand(const struct mf *F, u64 *r);
const char *ecp_pt_str(const struct ecp_curve *E, const struct ecp_pt *P);
const char *eck_pt_str(const struct eck_curve *E, const struct eck_pt *P);

/* ---- ecnamed.c: named curves ---- */
enum { ECN_W = 1, ECN_MONT, ECN_ED };
struct ecnamed {
	const char *name, *aliases, *desc;
	int model;          /* ECN_W: y^2 = x^3 + a x + b; ECN_MONT: v^2 =
			     * u^3 + a u^2 + u; ECN_ED: a x^2 + y^2 = 1 + b x^2 y^2 */
	const char *p, *a, *b, *gx, *gy, *n, *h;
	const char *g2x, *g2y;   /* pairing curves: "c0,c1" over F_p[u]/(u^2+1) */
	const char *param;       /* BN u / BLS x */
	const char *xi;          /* the twist: xi = XI + u */
};
extern const struct ecnamed ecnamed_tab[];
struct ecload {
	const struct ecnamed *N;
	struct mf F;
	struct ecp_curve E;      /* Weierstrass model (E.K = &F: do not copy) */
	struct ecp_pt G;
	nn n, h, order;
	u64 A[FW];               /* Montgomery A (ECN_MONT, ECN_ED) */
	int has_A;
};
const struct ecnamed *ecnamed_find(const char *name);
void ecnamed_load(const struct ecnamed *N, struct ecload *L);
int ecnamed_check(struct ecload *L, char *why, size_t n);

/* ---- eccount.c: counting points ---- */
struct eccert {
	int ok;             /* N is the only candidate left */
	int complete;       /* the factorisations used were complete */
	int iters, npts;
	struct ecp_pt P[4]; /* points on E and their exact orders */
	nn ord[4];
	nn L, L2;           /* lcm of orders on E and on the twist */
};
void ec_hasse(const nn *p, nn *lo, nn *hi, nn *w2);
void ec_twist(const struct ecp_curve *E, struct ecp_curve *T, u64 *d);
int ec_count_naive(const struct ecp_curve *E, nn *N);
int ec_point_order(const struct ecp_curve *E, const struct ecp_pt *P,
    const nn *N, const struct nnfac *f, nn *ord);
int ec_certify(const struct ecp_curve *E, const nn *N, struct eccert *C);
int ec_count_bsgs(const struct ecp_curve *E, nn *N, struct eccert *C);
int cornacchia4(const nn *p, u64 d, nn *x, nn *y);
int ec_count_cm01(const struct ecp_curve *E, nn *N, struct eccert *C);
void ec_structure(const struct ecp_curve *E, const nn *N,
    const struct nnfac *f, nn *n1, nn *n2);
int ec_embedding_degree(const nn *p, const nn *r, int max);

/* ---- ecalg.c: ECM ---- */
size_t ecff_primes(u64 limit, const uint32_t **out);  /* primes <= limit */
struct ecm_res {
	int found;
	nn f;
	u64 sigma, B1, B2;
	int stage, curves;
	u64 ms;
};
int ecm_run(const nn *n, u64 B1, u64 B2, int maxcurves, u64 sigma0,
    struct ecm_res *r);
int ecm_find(const nn *n, nn *f, int effort);

/* ---- small helpers ---- */
void ecff_note_vartime(void);
u64 ecff_ms(void);    /* monotonic milliseconds, for timings */

#endif
