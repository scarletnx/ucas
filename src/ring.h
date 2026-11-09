/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Rings and finite fields: Z/n, GF(p^n), polynomials and matrices over
 * them (ntheory.c, gfield.c), the commands (ring.c) and their English
 * (ring_nlp.c).  Elements are u64 codes: a residue for Z/n, and for an
 * extension R[x]/(f) the coefficients packed base n (bits for n = 2), so
 * 0x57 in GF(2^8) is x^6 + x^4 + x^2 + x + 1.  Nothing here depends on
 * the node DAG; parse.c and nlp.c call the two entry points at the end.
 */
#ifndef RING_H
#define RING_H

#include <stdint.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
typedef __int128 i128;

/* ---- ntheory.c: arithmetic mod n < 2^64 ---- */
u64 nt_mulmod(u64 a, u64 b, u64 m);
u64 nt_addmod(u64 a, u64 b, u64 m);
u64 nt_submod(u64 a, u64 b, u64 m);
u64 nt_powmod(u64 a, u64 e, u64 m);
u64 nt_gcd(u64 a, u64 b);
u64 nt_inv(u64 a, u64 m, u64 *g);   /* a^-1 mod m, or 0 with *g = gcd > 1 */
int nt_isprime(u64 n);              /* deterministic Miller-Rabin */
struct nt_fac {
	int k;
	u64 p[64];
	int e[64];
};
void nt_factor(u64 n, struct nt_fac *f);   /* sorted primes; n = 1: k = 0 */
u64 nt_phi(u64 n);
u64 nt_lambda(u64 n);                 /* Carmichael */
u64 nt_order(u64 a, u64 n);           /* multiplicative; 0 if not a unit */
int nt_jacobi(u64 a, u64 n);          /* n odd */
int nt_sqrt_prime(u64 a, u64 p, u64 *r);   /* Tonelli-Shanks; 0: none */
int nt_sqrt_all(u64 a, u64 n, u64 *roots, int max);  /* sorted; -1: > max */
int nt_has_primroot(u64 n);
u64 nt_primroot(u64 n);               /* the smallest; 0 if none */
int nt_crt(u64 r1, u64 m1, u64 r2, u64 m2, u64 *r, u64 *m);  /* 0: none, -1: overflow */
u64 nt_isqrt(u64 n);

/* ---- gfield.c: Z/n and R[x]/(f) as rings of u64 codes ---- */
#define RMAXF 64     /* degree of an extension (n^deg <= 2^64) */
#define RMAXD 512    /* coefficients of a polynomial over a ring */
#define RMAXDEG 255  /* the degree of polynomials as given (products fit) */
struct ring {
	int deg;          /* 1: Z/n; > 1: Z/n[x]/(f) */
	u64 n;            /* the coefficient modulus */
	u64 f[RMAXF + 1]; /* deg > 1: the monic modulus, f[deg] = 1 */
	int nprime;       /* n is prime */
	int field;        /* n prime and (deg == 1 or f irreducible) */
	u64 qm1;          /* elements - 1 (n^deg - 1) */
	u64 gen;          /* a generator of the unit group (field), else 0 */
	uint32_t *lg, *ex;  /* log/antilog tables (fields of <= 2^16 elements) */
	const char *fname;  /* the modulus' name (aes, crc32 ...) or NULL */
};
void ring_zn(struct ring *R, u64 n);
int ring_ext(struct ring *R, u64 n, int deg, const u64 *f);  /* 0: too big */
void ring_free(struct ring *R);
void ring_tables(struct ring *R);     /* gen and log tables for fields */
int r_isunit(const struct ring *R, u64 a);
u64 r_add(const struct ring *R, u64 a, u64 b);
u64 r_sub(const struct ring *R, u64 a, u64 b);
u64 r_neg(const struct ring *R, u64 a);
u64 r_mul(const struct ring *R, u64 a, u64 b);
u64 r_mul_alt(const struct ring *R, u64 a, u64 b);  /* another algorithm */
u64 r_pow(const struct ring *R, u64 a, u64 e);
int r_inv(const struct ring *R, u64 a, u64 *out);   /* 0: not a unit */
u64 r_const(const struct ring *R, u64 k);  /* the integer k times 1 */
u64 r_x(const struct ring *R);             /* the class of x */
void r_digits(const struct ring *R, u64 a, u64 *d);   /* deg digits */
u64 r_pack(const struct ring *R, const u64 *d);
u64 r_frob(const struct ring *R, u64 a);   /* a^n (n = p for fields) */
u64 r_order(const struct ring *R, u64 a);  /* unit order, 0 if none */
int r_dlog(const struct ring *R, u64 g, u64 a, u64 *x, u64 *ord);
u64 r_charp(const struct ring *R);          /* the characteristic, n */

struct rpoly {
	int d;            /* degree; -1 for 0 */
	u64 c[RMAXD];
};
void rp_zero(struct rpoly *a);
void rp_const(const struct ring *R, struct rpoly *a, u64 c);
void rp_x(struct rpoly *a);
void rp_norm(struct rpoly *a);
int rp_eq(const struct rpoly *a, const struct rpoly *b);
void rp_add(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b);
void rp_sub(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b);
int rp_mul(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b);   /* 0: degree too high */
void rp_scale(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    u64 k);
int rp_divmod(const struct ring *R, const struct rpoly *a,
    const struct rpoly *b, struct rpoly *q, struct rpoly *r);  /* 0: lc not a unit */
int rp_monic(const struct ring *R, struct rpoly *a);
int rp_gcd(const struct ring *R, struct rpoly *g, const struct rpoly *a,
    const struct rpoly *b);   /* monic; fields */
int rp_xgcd(const struct ring *R, struct rpoly *g, struct rpoly *s,
    struct rpoly *t, const struct rpoly *a, const struct rpoly *b);
void rp_deriv(const struct ring *R, struct rpoly *r, const struct rpoly *a);
void rp_mulmod(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *b, const struct rpoly *f);
void rp_powmod(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    u64 e, const struct rpoly *f);
void rp_frobmod(const struct ring *R, struct rpoly *r, const struct rpoly *a,
    const struct rpoly *f);   /* a^q mod f, q = #R */
u64 rp_eval(const struct ring *R, const struct rpoly *a, u64 x);
int rp_irreducible(const struct ring *R, const struct rpoly *f);  /* Rabin */
int rp_benor(const struct ring *R, const struct rpoly *f);        /* Ben-Or */
int rp_berlekamp_count(const struct ring *R, const struct rpoly *f);
/* irreducible factors with multiplicities (monic); returns the count */
int rp_factor(const struct ring *R, const struct rpoly *f, struct rpoly *fac,
    int *mult, int max, u64 *lead);
/* the order of x mod f (f(0) != 0), 0 if unknown (too big to factor) */
u64 rp_xorder(const struct ring *R, const struct rpoly *f, int *overflow);

/* matrices over a ring, row major */
struct rmat {
	int r, c;
	u64 *a;
};
#define RM(m, i, j) ((m)->a[(size_t)(i) * (m)->c + (j)])
void rm_init(struct rmat *m, int r, int c);
void rm_free(struct rmat *m);
int rm_rref(const struct ring *R, struct rmat *m, int *piv);  /* fields: rank */
int rm_det(const struct ring *R, const struct rmat *m, u64 *det);
int rm_inverse(const struct ring *R, const struct rmat *m, struct rmat *inv);
int rm_nullspace(const struct ring *R, const struct rmat *m, struct rmat *ns);
int rm_mul(const struct ring *R, struct rmat *r, const struct rmat *a,
    const struct rmat *b);
void rm_charpoly(const struct ring *R, const struct rmat *m, struct rpoly *cp);

/* the magic number of `magic D` (ring_misc.inc): 0 if none is needed */
int ring_magic_number(u64 d, int w, int sign, u64 *m, int *s, int *add,
    u64 *e);

/* ---- ring.c: parse.c and nlp.c hooks ---- */
struct P;
int ring_command(struct P *p);
int ring_has_ring(const char *args);
int ring_parses(const char *text);     /* the whole text is a ring */   /* "... mod 7", "... over GF(2)" */
const char *ring_nlp_raw(const char *sent);   /* ring_nlp.c */
int ring_nlp_english(const char *s, unsigned long n);  /* 1, 0, -1 undecided */

#endif
