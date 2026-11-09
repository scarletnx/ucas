/*
 * absalg.h - exact abstract algebra: the core API (absalg.md, part A).
 *
 *   zint    arbitrary-precision integers: 64-bit handles, small values
 *           inline, big ones in one arena (SoA: sign/len/offset columns
 *           and a single limb array)                          zint.c
 *   zq      rationals as (num, den) zint pairs                zint.c
 *   zmat    dense integer matrices: HNF and SNF with unimodular
 *           transforms (verified), kernel/image/cokernel, solving over
 *           Z, lattices; qmat over Q; Z/p through ring.h's rmat zmat.c
 *   fgab    finitely generated abelian groups as subquotients A/B of
 *           Z^n, homomorphisms, kernels, images, cokernels, quotients,
 *           sums and intersections, m*G, G[m], T_p(G), membership with
 *           witness, element orders, isomorphism with explicit maps,
 *           direct sums, Hom, tensor, Tor, Ext, exactness       fgab.c
 *   pgrp    finite permutation groups (Schreier-Sims)         pgrp.c
 *   cgrp    finite groups by Cayley table                     pgrp.c
 *   mpl     polynomials over Q and F_p, Groebner bases with cofactors,
 *           ideal membership/inclusion/equality               ideal.c
 *   pidm    modules over F[x] (and Z via fgab): SNF            ideal.c
 *
 * Conventions used everywhere:
 *
 * - Exact: no floating point.  Integers never wrap: a zint whose value
 *   leaves the inline range moves to the arena.
 * - Matrices act on COLUMN vectors: a map Z^n -> Z^m is an m x n matrix,
 *   y = M x.  Generating sets of lattices are the COLUMNS of a matrix.
 * - Every function that claims a fact computes a certificate and checks
 *   it (U A V = D with U Ui = I, A x = b, a preimage, an inverse map) and
 *   fail()s with "internal: ..." if the check does not hold, so a wrong
 *   answer is never returned silently.  "_check" functions re-verify a
 *   certificate independently and return 1/0.
 * - Errors in the input (dimension mismatch, division by zero) fail()
 *   (ucas.h: longjmp to the statement); mathematical outcomes ("not
 *   solvable", "not a member", "not isomorphic") are return values.
 * - Memory: zints live in one global arena; z_mark()/z_release() scope
 *   temporaries, z_release_keep() keeps chosen results.  Matrices and
 *   groups own one malloc'd array per matrix (xmalloc), freed with the
 *   _free function; an all-zero struct is a valid empty object and an
 *   all-zero zint array is a zero vector.
 *
 * Stability: this header is the contract for homalg (B) and prove (C).
 * Changes are additive only.
 */
#ifndef ABSALG_H
#define ABSALG_H

#include <stddef.h>
#include <stdint.h>

#include "ring.h"   /* u64, struct ring, struct rmat (Z/p matrices) */

struct sbuf;
struct P;

/* ================================================================== */
/* zint: integers                                                      */
/* ================================================================== */

/*
 * A zint is a handle: even values are the integer v = h / 2 inline (|v| <
 * 2^61), odd values index the arena.  The integer 0 is the handle 0, so
 * zeroed memory holds zeros.  A value has exactly one handle form (small
 * values are always inline), so equal small values compare equal as
 * handles; use z_eq/z_cmp in general.
 */
typedef int64_t zint;
#define Z_ZERO ((zint)0)
#define Z_ONE ((zint)2)
#define Z_SMALL_LIM ((int64_t)1 << 61)   /* inline: |v| < Z_SMALL_LIM */
#define Z_IS_SMALL(h) (((h) & 1) == 0)

zint z_from_i64(int64_t v);
zint z_from_u64(uint64_t v);
zint z_from_i128(i128 v);
int z_fits_i64(zint a);
int64_t z_to_i64(zint a);          /* fail()s if it does not fit */
int z_to_u64(zint a, u64 *out);    /* 0: negative or too big */

zint z_add(zint a, zint b);
zint z_sub(zint a, zint b);
zint z_mul(zint a, zint b);
zint z_neg(zint a);
zint z_abs(zint a);
zint z_addmul(zint acc, zint a, zint b);   /* acc + a*b */
zint z_submul(zint acc, zint a, zint b);   /* acc - a*b */
int z_sign(zint a);                /* -1, 0, 1 */
int z_cmp(zint a, zint b);         /* -1, 0, 1 */
int z_cmpabs(zint a, zint b);
int z_eq(zint a, zint b);
int z_is_zero(zint a);
int z_is_one(zint a);
int z_is_pm1(zint a);              /* +-1 */

/* division; b != 0 (fail() otherwise) */
void z_divmod_trunc(zint a, zint b, zint *q, zint *r);   /* C: q toward 0 */
void z_divmod_floor(zint a, zint b, zint *q, zint *r);   /* r has b's sign */
void z_divmod_euclid(zint a, zint b, zint *q, zint *r);  /* 0 <= r < |b| */
zint z_div_floor(zint a, zint b);
zint z_mod(zint a, zint b);        /* Euclidean: 0 <= r < |b| */
zint z_divexact(zint a, zint b);   /* fail()s if b does not divide a */
int z_divides(zint d, zint a);     /* d | a (0 | a iff a = 0) */
/* round(a/b) with ties toward -inf: the remainder lies in (-|b|/2, |b|/2] */
zint z_div_round(zint a, zint b);

zint z_gcd(zint a, zint b);        /* >= 0; gcd(0, 0) = 0 */
zint z_xgcd(zint a, zint b, zint *s, zint *t);   /* g = s a + t b, g >= 0 */
zint z_lcm(zint a, zint b);        /* >= 0 */
zint z_pow(zint a, uint64_t e);
zint z_powmod(zint a, zint e, zint m);   /* e >= 0, m > 0 */
zint z_invmod(zint a, zint m);     /* a^-1 mod m; 0 if not a unit */
/* p-adic valuation of a != 0 (p >= 2); *rest = a / p^v (may be NULL) */
int z_val(zint a, zint p, zint *rest);
u64 z_mod_u64(zint a, u64 m);      /* Euclidean residue, m > 0 */
size_t z_bits(zint a);             /* bits of |a| (0 for 0) */
int z_isprime(zint a);   /* exact below 2^64 (deterministic Miller-Rabin);
			  * above: Miller-Rabin with 20 fixed bases plus a
			  * strong Lucas test (BPSW), documented as probable */
/* the prime factors of |a| (a != 0) with exponents, increasing; returns
 * the count, or -1 when a cofactor above 2^64 could not be split (then
 * the last entry is that composite with exponent 1 and *complete = 0) */
int z_factor(zint a, zint *p, int *e, int max, int *complete);

/* text: decimal with optional sign, 0x hex; *end after the number */
int z_parse(const char *s, zint *out, const char **end);   /* 0: none */
void z_print(struct sbuf *b, zint a);
const char *z_str(zint a);         /* rotating static buffers (8) */

/* the arena */
typedef struct {
	uint64_t n, limbs;
} zmark;
zmark z_mark(void);
void z_release(zmark m);           /* every zint made after m dies */
/* release, but keep keep[0..n-1] alive (their handles are rewritten) */
void z_release_keep(zmark m, zint *keep, size_t n);
size_t z_arena_bytes(void);
void z_arena_reset(void);          /* everything dies */

/* ---- rationals: d > 0, gcd(n, d) = 1 ---- */
struct zq {
	zint n, d;
};
struct zq zq_make(zint n, zint d);  /* normalises; d != 0 */
struct zq zq_int(zint n);
struct zq zq_add(struct zq a, struct zq b);
struct zq zq_sub(struct zq a, struct zq b);
struct zq zq_mul(struct zq a, struct zq b);
struct zq zq_div(struct zq a, struct zq b);
struct zq zq_neg(struct zq a);
struct zq zq_inv(struct zq a);
int zq_cmp(struct zq a, struct zq b);
int zq_is_zero(struct zq a);
int zq_eq(struct zq a, struct zq b);
int zq_parse(const char *s, struct zq *out, const char **end);  /* 3, -3/4 */
void zq_print(struct sbuf *b, struct zq a);
const char *zq_str(struct zq a);

/* ================================================================== */
/* zmat: integer matrices                                              */
/* ================================================================== */

struct zmat {
	int r, c;
	zint *a;        /* row major, r * c */
};
#define ZM(m, i, j) ((m)->a[(size_t)(i) * (size_t)(m)->c + (size_t)(j)])

void zm_init(struct zmat *m, int r, int c);     /* zero matrix */
void zm_free(struct zmat *m);
void zm_identity(struct zmat *m, int n);
void zm_copy(struct zmat *dst, const struct zmat *src);  /* dst fresh */
void zm_move(struct zmat *dst, struct zmat *src);  /* dst freed, src emptied */
void zm_from_i64(struct zmat *m, int r, int c, const int64_t *v);  /* rows */
/* "[[1, 2], [3, 4]]", "[1, 2, 3]" (one row), entries big or negative;
 * returns 0 on a syntax error (with *end where) */
int zm_parse(struct zmat *m, const char *s, const char **end);
void zm_print(struct sbuf *b, const struct zmat *m);   /* [[1, 2], [3, 4]] */
int zm_eq(const struct zmat *a, const struct zmat *b);
int zm_is_zero(const struct zmat *m);
int zm_is_identity(const struct zmat *m);

/* results go to a fresh r (freed first if it holds a matrix); r may not
 * alias an operand */
void zm_mul(struct zmat *r, const struct zmat *a, const struct zmat *b);
void zm_add(struct zmat *r, const struct zmat *a, const struct zmat *b);
void zm_sub(struct zmat *r, const struct zmat *a, const struct zmat *b);
void zm_scale(struct zmat *r, const struct zmat *a, zint k);
void zm_transpose(struct zmat *r, const struct zmat *a);
void zm_hcat(struct zmat *r, const struct zmat *a, const struct zmat *b);
void zm_vcat(struct zmat *r, const struct zmat *a, const struct zmat *b);
void zm_block_diag(struct zmat *r, const struct zmat *a, const struct zmat *b);
void zm_sub_cols(struct zmat *r, const struct zmat *a, int j0, int j1);  /* [j0, j1) */
void zm_sub_rows(struct zmat *r, const struct zmat *a, int i0, int i1);
void zm_mul_vec(zint *y, const struct zmat *a, const zint *x);   /* y = A x */
void zm_col(zint *x, const struct zmat *a, int j);                /* copy out */

int zm_rank(const struct zmat *a);
zint zm_det(const struct zmat *a);      /* square; Bareiss, exact */

/*
 * Hermite normal form, row style: H = U A with U unimodular (r x r), H in
 * row echelon form, every pivot > 0, the entries above a pivot reduced to
 * [0, pivot), zero rows last.  Returns the rank; pivcol[k] (if given, r
 * entries) is the pivot column of row k.  u may be NULL.
 */
int zm_hnf(const struct zmat *a, struct zmat *h, struct zmat *u, int *pivcol);
/* column style: H = A V, V unimodular (c x c), H = transpose of the row
 * HNF of A^T: column echelon, the nonzero columns first */
int zm_hnf_col(const struct zmat *a, struct zmat *h, struct zmat *v);
int zm_hnf_check(const struct zmat *a, const struct zmat *h,
    const struct zmat *u);   /* U A = H, det U = +-1, H in HNF */

/*
 * Smith normal form: U A V = D, U (r x r) and V (c x c) unimodular, D
 * diagonal with d_1 | d_2 | ... | d_rank > 0 and zeros after.  Ui = U^-1
 * and Vi = V^-1 are computed alongside (so a generator of a cyclic factor
 * is a column of Ui).  Verified before returning.
 */
struct zsnf {
	struct zmat D, U, V, Ui, Vi;
	int rank;
};
int zm_snf(const struct zmat *a, struct zsnf *s);   /* returns the rank */
void zsnf_free(struct zsnf *s);
int zm_snf_check(const struct zmat *a, const struct zsnf *s);
/* the diagonal d_1 .. d_min(r,c) (zeros included); returns the count */
int zm_snf_diag(const struct zsnf *s, zint *d);

/* a basis of ker A = {x : A x = 0} as the columns of k (c x dim);
 * returns dim.  The basis is saturated: Z^c / ker A is torsion free. */
int zm_kernel(const struct zmat *a, struct zmat *k);
/* a basis of the lattice A Z^c (columns of im, r x rank, column HNF) */
int zm_image(const struct zmat *a, struct zmat *im);
/* A x = b over Z for every column of b (r x k): returns 1 and x (c x k)
 * with A x = b checked, or 0 when some column has no integer solution */
int zm_solve(const struct zmat *a, const struct zmat *b, struct zmat *x);
int zm_solve_vec(const struct zmat *a, const zint *b, zint *x);
/* the inverse over Z of a unimodular matrix; 0 if det != +-1 */
int zm_inverse(const struct zmat *a, struct zmat *inv);
/* lattices given by generating columns (same row count) */
int zm_lattice_contains(const struct zmat *l, const zint *x, zint *w); /* x = L w */
int zm_lattice_le(const struct zmat *l1, const struct zmat *l2);   /* L1 in L2 */
int zm_lattice_eq(const struct zmat *l1, const struct zmat *l2);
void zm_lattice_sum(struct zmat *r, const struct zmat *l1, const struct zmat *l2);
void zm_lattice_meet(struct zmat *r, const struct zmat *l1, const struct zmat *l2);

/* ---- matrices over Q: SoA numerator and denominator columns ---- */
struct qmat {
	int r, c;
	zint *n, *d;    /* row major; d > 0, gcd(n, d) = 1 */
};
void qm_init(struct qmat *m, int r, int c);
void qm_free(struct qmat *m);
void qm_from_zmat(struct qmat *m, const struct zmat *a);
struct zq qm_get(const struct qmat *m, int i, int j);
void qm_set(struct qmat *m, int i, int j, struct zq v);
int qm_parse(struct qmat *m, const char *s, const char **end);  /* 1/2 ok */
void qm_print(struct sbuf *b, const struct qmat *m);
void qm_mul(struct qmat *r, const struct qmat *a, const struct qmat *b);
int qm_rref(struct qmat *m, int *pivcol);     /* in place; the rank */
int qm_rank(const struct qmat *a);
struct zq qm_det(const struct qmat *a);
int qm_inverse(const struct qmat *a, struct qmat *inv);   /* 0: singular */
int qm_nullspace(const struct qmat *a, struct qmat *ns);  /* columns; dim */
int qm_solve(const struct qmat *a, const struct qmat *b, struct qmat *x);

/* ---- over Z/p (p prime < 2^64): ring.h's rmat with ring_zn(R, p) ---- */
void zm_reduce_mod(const struct zmat *a, u64 m, struct rmat *out);
int zm_rank_mod(const struct zmat *a, u64 p);
int zm_nullspace_mod(const struct zmat *a, u64 p, struct rmat *ns);

/* ================================================================== */
/* fgab: finitely generated abelian groups                             */
/* ================================================================== */

/*
 * G = A / B with B <= A <= Z^n, the canonical object (a presentation
 * Z^g / R is A = Z^g, B = R Z^k; homology is cycles / boundaries).
 *
 *   n      the ambient rank; ELEMENTS are vectors x in Z^n with x in A,
 *          taken modulo B
 *   A      n x a, a basis of the lattice A (full column rank a)
 *   B      n x b, generators of B (B <= A is checked)
 *   R      a x b, B in A-coordinates: B = A R, so G = Z^a / R Z^b
 *   S      Smith form of R: U R V = D, with Ui = U^-1
 *   d      a entries: the invariant factor of each SNF coordinate,
 *          d[i] = D_ii (1 trivial, >= 2 torsion) or 0 (free), with
 *          d[0..ntriv-1] = 1, then torsion increasing by divisibility,
 *          then zeros
 *
 * A-coordinates c in Z^a: x = A c.  SNF coordinates y = U c, y_i taken
 * mod d[i]; the canonical form keeps i >= ntriv only (ninv of them, the
 * torsion ones reduced to [0, d[i])).  The i-th canonical generator is
 * A * (column ntriv + i of Ui).
 *
 * HOMOMORPHISMS f: G -> H are COORDINATE matrices M (H.a x G.a): column j
 * is f(A_G e_j) in H's A-coordinates.  Composition is matrix product
 * (ab_hom_compose).  Well-definedness: M R_G Z^b <= R_H Z^* (ab_hom_check).
 *
 * SUBGROUPS of G are fgab's with the same ambient n and the same B (and
 * A_S <= A_G); ab_kernel, ab_image, ab_sub return them that way, so the
 * lattice operations (sum, meet, inclusion) are lattice operations on A.
 */
struct fgab {
	int n, a, b;
	struct zmat A, B, R;
	struct zsnf S;
	zint *d;          /* a entries */
	int ntriv;        /* d[i] == 1 for i < ntriv */
	int ninv;         /* a - ntriv: the nontrivial cyclic factors */
	int rank;         /* free rank (d[i] == 0 entries) */
};

/* construction; every result is normalised (A a basis, SNF computed) */
void ab_present(struct fgab *G, const struct zmat *rel);   /* Z^g / rel Z^k */
/* A/B in Z^n: A, B given by generating columns (A NULL: Z^n, B NULL: 0).
 * Returns 0 (G untouched) when B is not inside A. */
int ab_subquotient(struct fgab *G, int n, const struct zmat *A,
    const struct zmat *B);
void ab_cyclic(struct fgab *G, zint m);       /* Z/m; m = 0: Z */
void ab_free_abelian(struct fgab *G, int r);  /* Z^r */
void ab_from_invariants(struct fgab *G, const zint *d, int k);  /* sum Z/d_i */
/* "Z/4 + Z/6", "Z^2 + Z/3", "Z/2 (+) Z/12", "0", "Z/2^3", "Z/4xZ/6" */
int ab_parse(struct fgab *G, const char *s, const char **end);
void ab_free(struct fgab *G);
void ab_copy(struct fgab *dst, const struct fgab *src);

/* invariants */
int ab_is_trivial(const struct fgab *G);
int ab_is_finite(const struct fgab *G);
int ab_is_free(const struct fgab *G);
int ab_is_cyclic(const struct fgab *G);
zint ab_order(const struct fgab *G);       /* 0 when infinite */
zint ab_exponent(const struct fgab *G);    /* 0 when infinite */
/* the nontrivial invariant factors d_1 | d_2 | ... then 0 per free Z;
 * returns ninv */
int ab_invariants(const struct fgab *G, zint *out);
/* elementary divisors: prime powers (sorted by prime, then power) then 0
 * per free Z; returns the count, -1 if a factor could not be factored */
int ab_primary(const struct fgab *G, zint *out, int max);
void ab_print(struct sbuf *b, const struct fgab *G);   /* Z/2 + Z/12 + Z^2, 0 */
void ab_print_primary(struct sbuf *b, const struct fgab *G);
const char *ab_str(const struct fgab *G);              /* rotating static */

/* elements: ambient vectors x (n entries) */
int ab_coords(const struct fgab *G, const zint *x, zint *c);  /* 0: x not in A */
void ab_from_coords(const struct fgab *G, const zint *c, zint *x);  /* x = A c */
/* canonical SNF coordinates (ninv entries, torsion ones reduced) */
int ab_canon(const struct fgab *G, const zint *x, zint *y);   /* 0: x not in A */
void ab_from_canon(const struct fgab *G, const zint *y, zint *x);
void ab_gen(const struct fgab *G, int i, zint *x);  /* i-th canonical generator */
int ab_elem_eq(const struct fgab *G, const zint *x, const zint *y);
int ab_elem_is_zero(const struct fgab *G, const zint *x);
zint ab_elem_order(const struct fgab *G, const zint *x);    /* 0: infinite */
/* x in B, with x = B w (w: b entries) as the certificate */
int ab_elem_zero_witness(const struct fgab *G, const zint *x, zint *w);

/* subgroups */
void ab_sub(struct fgab *S, const struct fgab *G, const struct zmat *gens);
/* x in S (S a subgroup, ambient vector): x = A_S w (w: S.a entries) */
int ab_member(const struct fgab *S, const zint *x, zint *w);
int ab_sub_le(const struct fgab *S, const struct fgab *T);   /* S <= T */
int ab_sub_eq(const struct fgab *S, const struct fgab *T);
void ab_sub_sum(struct fgab *r, const struct fgab *S, const struct fgab *T);
void ab_sub_meet(struct fgab *r, const struct fgab *S, const struct fgab *T);
void ab_mulsub(struct fgab *r, const struct fgab *G, zint m);     /* m*G */
void ab_torsion_m(struct fgab *r, const struct fgab *G, zint m);  /* G[m] */
void ab_torsion(struct fgab *r, const struct fgab *G);            /* T(G) */
void ab_ptorsion(struct fgab *r, const struct fgab *G, zint p);   /* T_p(G) */
/* G/S (S a subgroup of G): same ambient, A = A_G, B = A_S; the
 * projection G -> G/S is the identity in coordinates */
void ab_quotient(struct fgab *r, const struct fgab *G, const struct fgab *S);
/* the inclusion S -> G (S.a x ... : G.a x S.a coordinate matrix) */
void ab_inclusion(struct zmat *M, const struct fgab *S, const struct fgab *G);

/* homomorphisms (coordinate matrices, see above) */
int ab_hom_check(const struct fgab *G, const struct fgab *H,
    const struct zmat *M);
/* from ambient images: imgs is H.n x G.a (column j = f(A_G e_j)); 0 if an
 * image is not in A_H */
int ab_hom_from_images(struct zmat *M, const struct fgab *G,
    const struct fgab *H, const struct zmat *imgs);
/* induced by an ambient map F: Z^{G.n} -> Z^{H.n} (H.n x G.n) with
 * F(A_G) <= A_H and F(B_G) <= B_H; 0 if not (e.g. a chain map on
 * homology) */
int ab_hom_induced(struct zmat *M, const struct fgab *G, const struct fgab *H,
    const struct zmat *F);
/* from images of the canonical generators (H ambient vectors, H.n x
 * G.ninv); 0 if an image's order does not divide its generator's */
int ab_hom_from_gens(struct zmat *M, const struct fgab *G,
    const struct fgab *H, const struct zmat *imgs);
void ab_hom_apply(const struct fgab *G, const struct fgab *H,
    const struct zmat *M, const zint *x, zint *y);    /* ambient -> ambient */
void ab_hom_compose(struct zmat *r, const struct zmat *g,
    const struct zmat *f);                             /* g o f */
int ab_hom_eq(const struct fgab *G, const struct fgab *H,
    const struct zmat *M1, const struct zmat *M2);
int ab_hom_is_zero(const struct fgab *G, const struct fgab *H,
    const struct zmat *M);
void ab_hom_identity(struct zmat *M, const struct fgab *G);
void ab_hom_zero(struct zmat *M, const struct fgab *G, const struct fgab *H);
/* kernel (a subgroup of G), image (a subgroup of H), cokernel H/im f;
 * inc/proj (may be NULL) are the coordinate matrices of the inclusion
 * ker -> G, im -> H and the projection H -> coker */
void ab_kernel(struct fgab *K, struct zmat *inc, const struct fgab *G,
    const struct fgab *H, const struct zmat *M);
void ab_image(struct fgab *I, struct zmat *inc, const struct fgab *G,
    const struct fgab *H, const struct zmat *M);
void ab_cokernel(struct fgab *C, struct zmat *proj, const struct fgab *G,
    const struct fgab *H, const struct zmat *M);
int ab_hom_injective(const struct fgab *G, const struct fgab *H,
    const struct zmat *M);
int ab_hom_surjective(const struct fgab *G, const struct fgab *H,
    const struct zmat *M);
/* f an isomorphism: then inv (H.a x ... G.a x H.a) is its inverse,
 * checked both ways */
int ab_hom_iso(const struct fgab *G, const struct fgab *H,
    const struct zmat *M, struct zmat *inv);
/* G = H?  An explicit isomorphism phi: G -> H and its inverse psi: H -> G
 * (either may be NULL), both checked (psi phi = 1, phi psi = 1) */
int ab_isomorphic(const struct fgab *G, const struct fgab *H,
    struct zmat *phi, struct zmat *psi);

/* constructions */
/* G + H in Z^{G.n + H.n}, with the inclusions and projections (NULL ok) */
void ab_direct_sum(struct fgab *r, const struct fgab *G, const struct fgab *H,
    struct zmat *i1, struct zmat *i2, struct zmat *p1, struct zmat *p2);
/* Hom(G, H): ambient Z^{H.a * G.a}; an element is a coordinate matrix M
 * flattened row major (ab_hom_unflatten), A = the well-defined maps and B
 * = the maps that are zero (columns in R_H Z^*) */
void ab_hom_group(struct fgab *r, const struct fgab *G, const struct fgab *H);
void ab_hom_unflatten(struct zmat *M, const struct fgab *G,
    const struct fgab *H, const zint *x);
/* G (x) H = Z^{G.a H.a} / (R_G (x) 1, 1 (x) R_H); the element e_i (x) f_j
 * is the basis vector i * H.a + j */
void ab_tensor(struct fgab *r, const struct fgab *G, const struct fgab *H);
/* Tor(G, H) = ker(R (x) 1_H) and Ext(G, H) = coker(Hom(R, H)) for the
 * free resolution 0 -> Z^k -R-> Z^a -> G -> 0 (R of full column rank) */
void ab_tor(struct fgab *r, const struct fgab *G, const struct fgab *H);
void ab_ext(struct fgab *r, const struct fgab *G, const struct fgab *H);

/* exactness of G1 -f-> G2 -g-> G3 at G2: g f = 0 and ker g <= im f.
 * Certificate: pre (G1.a x k), f(pre) = the k generators of ker g
 * (coordinates, modulo R_G2).  why (may be NULL) gets the reason on 0. */
int ab_exact_at(const struct fgab *G1, const struct fgab *G2,
    const struct fgab *G3, const struct zmat *f, const struct zmat *g,
    struct zmat *pre, const char **why);
/* a sequence G[0] -f[0]-> G[1] -> ... -f[k-2]-> G[k-1], exact at every
 * inner group; *bad = the first index where it is not (or -1) */
int ab_exact_seq(const struct fgab *G, int k, const struct zmat *f, int *bad);

/* ================================================================== */
/* pgrp: finite permutation groups                                     */
/* ================================================================== */

/*
 * Points are 0 .. n-1; a permutation is n uint32_t images, p[i] = i^p
 * (acting on the right: (p q)[i] = q[p[i]], "first p, then q").  A group
 * is generated by gens (ngens * n images, one flat array) and carries a
 * base and strong generating set computed by Schreier-Sims (deterministic)
 * with explicit transversals in one flat array.
 */
struct pgrp {
	int n;
	int ngens;
	uint32_t *gens;     /* ngens * n */
	/* BSGS */
	int nbase;
	uint32_t *base;     /* nbase points */
	int nsgs;
	uint32_t *sgs;      /* nsgs * n strong generators */
	/* transversals: level l maps orbit point q to u = tr[(tro[l] + k) * n],
	 * k = trk[l * n + q] (NOTOK if q is not in the l-th basic orbit) */
	uint32_t *trk;      /* nbase * n */
	uint64_t *tro;      /* nbase + 1 offsets into tr, in elements */
	uint32_t *tr;
	uint32_t *orb;      /* the basic orbits, concatenated (tro offsets) */
	zint order;
};
void pg_init(struct pgrp *G, int n, const uint32_t *gens, int ngens);
void pg_free(struct pgrp *G);
void pg_copy(struct pgrp *dst, const struct pgrp *src);
/* "(1 2 3)(4 5), (1 2)" (1-based cycles; commas or "," between
 * generators), "S4", "A5", "C6", "D8" (dihedral of order 8), "Sym(4)";
 * n = 0: the largest point named */
int pg_parse(struct pgrp *G, const char *s, int n, const char **end);
void pg_symmetric(struct pgrp *G, int n);
void pg_alternating(struct pgrp *G, int n);
void pg_cyclic(struct pgrp *G, int n);
void pg_dihedral(struct pgrp *G, int n);   /* order 2n, on n points */
void perm_print(struct sbuf *b, const uint32_t *p, int n);   /* cycles, 1-based; () */
void perm_mul(uint32_t *r, const uint32_t *p, const uint32_t *q, int n);  /* p then q */
void perm_inv(uint32_t *r, const uint32_t *p, int n);
int perm_is_id(const uint32_t *p, int n);
zint perm_order(const uint32_t *p, int n);

zint pg_order(const struct pgrp *G);
/* membership by sifting.  The certificate k (nbase entries, may be NULL):
 * k[l] is the point of the l-th basic orbit whose transversal element u_l
 * was used, and p = u_{nbase-1} u_{nbase-2} ... u_0 (perm_mul order:
 * u_{nbase-1} acts first); pg_word_eval rebuilds p from k. */
int pg_contains(const struct pgrp *G, const uint32_t *p, uint32_t *k);
void pg_word_eval(const struct pgrp *G, const uint32_t *k, uint32_t *p);
int pg_orbits(const struct pgrp *G, uint32_t *orbit_of);   /* the count */
int pg_orbit(const struct pgrp *G, uint32_t pt, uint32_t *out);  /* size */
void pg_stabilizer(struct pgrp *r, const struct pgrp *G, uint32_t pt);
int pg_is_subgroup(const struct pgrp *H, const struct pgrp *G);
int pg_is_normal(const struct pgrp *H, const struct pgrp *G);
int pg_is_abelian(const struct pgrp *G);
/* the normal closure of the subgroup generated by gens (k * n) in G */
void pg_normal_closure(struct pgrp *r, const struct pgrp *G,
    const uint32_t *gens, int k);
void pg_commutator(struct pgrp *r, const struct pgrp *G, const struct pgrp *H,
    const struct pgrp *K);    /* [H, K], both normal in G */
void pg_derived(struct pgrp *r, const struct pgrp *G);    /* G' */
/* G = G0 > G1 > ... until it stops; the orders (len entries); returns len */
int pg_derived_series(const struct pgrp *G, zint *orders, int max);
int pg_lower_central(const struct pgrp *G, zint *orders, int max);
int pg_is_solvable(const struct pgrp *G);
int pg_is_nilpotent(const struct pgrp *G);
/* the centre: elements commuting with every generator (fails above
 * PG_ENUM_MAX elements: it enumerates) */
#define PG_ENUM_MAX 2000000
void pg_centre(struct pgrp *r, const struct pgrp *G);
void pg_centralizer(struct pgrp *r, const struct pgrp *G,
    const uint32_t *x);
/* the elements, in a deterministic order (|G| <= PG_ENUM_MAX) */
uint32_t pg_elements(const struct pgrp *G, uint32_t **out);
/* right coset representatives of H in G (H <= G): count = [G : H] */
uint32_t pg_cosets(const struct pgrp *G, const struct pgrp *H, uint32_t **reps);
zint pg_index(const struct pgrp *G, const struct pgrp *H);
/* is gens[i] -> imgs[i] a homomorphism G -> H (both permutation groups)?
 * Decided through the graph subgroup <(g_i, h_i)> of G x H: it is a
 * homomorphism iff its order is |G|. */
int pg_hom_check(const struct pgrp *G, const struct pgrp *H,
    const uint32_t *imgs);

/* ---- finite groups by Cayley table ---- */
struct cgrp {
	int n;              /* elements 0 .. n-1 */
	uint32_t *tab;      /* n * n: tab[a * n + b] = a b */
	uint32_t *inv;
	uint32_t e;
};
/* checks closure, identity, inverses and associativity (n^3); 0 and
 * *why on failure */
int cg_init(struct cgrp *G, int n, const uint32_t *tab, const char **why);
void cg_free(struct cgrp *G);
int cg_parse(struct cgrp *G, const char *s, const char **why);  /* [[0,1],[1,0]] */
void cg_from_pgrp(struct cgrp *r, const struct pgrp *G);  /* via pg_elements */
void cg_to_pgrp(struct pgrp *r, const struct cgrp *G);    /* regular rep */
uint32_t cg_elem_order(const struct cgrp *G, uint32_t a);
int cg_is_abelian(const struct cgrp *G);
/* f: n_G entries; 1 if f(ab) = f(a)f(b) for all a, b, else 0 with the
 * counterexample in *ba, *bb */
int cg_hom_check(const struct cgrp *G, const struct cgrp *H, const uint32_t *f,
    uint32_t *ba, uint32_t *bb);

/* ================================================================== */
/* mpl: multivariate polynomials over Q or F_p; Groebner bases         */
/* ================================================================== */

#define MPL_MAXV 16
enum { MPL_LEX, MPL_GRLEX, MPL_GREVLEX };
struct mring {
	int nv;
	int order;
	u64 p;                       /* 0: Q, else a prime */
	char name[MPL_MAXV][16];     /* variable names, x0 > x1 > ... */
};
/* terms in decreasing monomial order, SoA: exponents (n * nv), then the
 * coefficient as num/den (over F_p: num in [0, p), den = 1) */
struct mpl {
	uint32_t n, cap;
	uint32_t *e;
	zint *cn, *cd;
};
void mring_init(struct mring *R, int nv, const char *const *names, int order,
    u64 p);
void mpl_init(struct mpl *f);
void mpl_free(struct mpl *f);
void mpl_copy(struct mpl *dst, const struct mpl *src);
void mpl_const(const struct mring *R, struct mpl *f, struct zq c);
void mpl_var(const struct mring *R, struct mpl *f, int v);
int mpl_is_zero(const struct mpl *f);
int mpl_eq(const struct mring *R, const struct mpl *f, const struct mpl *g);
void mpl_add(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b);
void mpl_sub(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b);
void mpl_mul(const struct mring *R, struct mpl *r, const struct mpl *a,
    const struct mpl *b);
void mpl_scale(const struct mring *R, struct mpl *r, const struct mpl *a,
    struct zq c);
void mpl_monic(const struct mring *R, struct mpl *f);
int mpl_deg(const struct mring *R, const struct mpl *f);   /* total; -1 for 0 */
/* "x^2*y - 3/2 y + 1", juxtaposition x y, powers, parentheses */
int mpl_parse(const struct mring *R, const char *s, struct mpl *f,
    const char **end);
void mpl_print(struct sbuf *b, const struct mring *R, const struct mpl *f);
const char *mpl_str(const struct mring *R, const struct mpl *f);
/* f = sum q[i] g[i] + rem, no term of rem divisible by any lt(g[i]) */
void mpl_divide(const struct mring *R, const struct mpl *f,
    const struct mpl *g, int k, struct mpl *q, struct mpl *rem);

/* a reduced Groebner basis g[0..n-1] of <f[0..k-1]> with cofactors:
 * g[i] = sum_j cof[i * k + j] f[j] (verified) */
struct gbasis {
	int n, k;
	struct mpl *g;
	struct mpl *cof;
	long pairs, reductions, skipped;   /* statistics (criteria) */
};
int mpl_groebner(const struct mring *R, const struct mpl *f, int k,
    struct gbasis *gb);
void gb_free(struct gbasis *gb);
int gb_check(const struct mring *R, const struct mpl *f,
    const struct gbasis *gb);   /* cofactors and S-pairs reduce to 0 */
/* h in <f>: cof (k entries) with h = sum cof[j] f[j], verified */
int gb_member(const struct mring *R, const struct gbasis *gb,
    const struct mpl *f, const struct mpl *h, struct mpl *cof);
/* <f> <= <h>: each f in <h> (gbh its basis) */
int gb_ideal_le(const struct mring *R, const struct mpl *f, int kf,
    const struct gbasis *gbh, const struct mpl *h);
int gb_ideal_eq(const struct mring *R, const struct mpl *f, int kf,
    const struct mpl *h, int kh);

/* ---- modules over the PID F[x] (univariate mpl, R->nv == 1) ---- */
/* a r x c matrix of polynomials (row major) -> U A V = D, invariant
 * factors monic, d_i | d_{i+1}; U, V unimodular (det in F*) */
struct pmat {
	int r, c;
	struct mpl *a;
};
#define PM(m, i, j) (&(m)->a[(size_t)(i) * (size_t)(m)->c + (size_t)(j)])
void pm_init(struct pmat *m, int r, int c);
void pm_free(struct pmat *m);
void pm_mul(const struct mring *R, struct pmat *r, const struct pmat *a,
    const struct pmat *b);
int pm_snf(const struct mring *R, const struct pmat *a, struct pmat *D,
    struct pmat *U, struct pmat *V);   /* returns the rank */
/* the F[x]-module F^n with x acting by the matrix m: invariant factors
 * of xI - m (the rational canonical form's); returns their count */
int pm_charmodule(const struct mring *R, const struct qmat *m,
    struct mpl *inv, int max);

/* ================================================================== */
/* commands (absalg_cmd.c): parse.c calls absalg_command               */
/* ================================================================== */

int absalg_command(struct P *p);

#endif
