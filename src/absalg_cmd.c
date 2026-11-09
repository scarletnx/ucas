/*
 * absalg_cmd.c - the ucas commands of the abstract-algebra core (absalg.h).
 *
 *   bigint EXPR | bigint factor N       exact integers of any size
 *   snf M, hnf M                        Smith / Hermite forms + transforms
 *   zmat [det|rank|kernel|image|inverse|cokernel|solve] M [, b]
 *   qmat [rref|det|rank|inverse|nullspace|solve] M [, b]   over Q
 *   abgroup G                           invariants, primary parts, ...
 *   abiso G, H                          isomorphic? with explicit maps
 *   abhom|abtensor|abtor|abext G, H     Hom, tensor, Tor, Ext
 *   abmap G -> H by M                   a homomorphism: ker, im, coker
 *   abelem G, x                         canonical form and order
 *   absub G, x1, x2, ...                a subgroup and its quotient
 *   abexact G1 -> G2 -> G3 by f, g      exactness with preimages
 *
 * (pgrp/cgrp/ideal commands are in the second half.)  Arguments are the
 * rest of the statement as text, read by small scanners here, like
 * ring.c does, so group and matrix syntax never reaches the expression
 * grammar.  Groups: "Z/4 + Z/6", "Z^2 + Z/3", "(Z/2)^3", "0", or a
 * matrix: [[...]] or "coker [[...]]" = Z^rows / (columns).  Each answer
 * prints its certificate and the check that was run on it.
 */
#include "absalg.h"
#include "ucas.h"
#include "report.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* notes (from parsing) come before the answer */
#define printf(...) (notes_flush(), printf(__VA_ARGS__))

static char text[1 << 16];

/* ---- text helpers ---- */

static char *
trim(char *s)
{
	char *e;

	while (isspace((unsigned char)*s))
		s++;
	e = s + strlen(s);
	while (e > s && isspace((unsigned char)e[-1]))
		*--e = 0;
	return s;
}

/* split at top-level commas (outside brackets/parentheses); returns count */
static int
split_commas(char *s, char **part, int max)
{
	int n = 0, depth = 0;
	char *start = s;

	for (; *s; s++) {
		if (*s == '[' || *s == '(')
			depth++;
		else if (*s == ']' || *s == ')')
			depth--;
		else if (*s == ',' && depth == 0) {
			*s = 0;
			if (n < max)
				part[n++] = trim(start);
			start = s + 1;
		}
	}
	if (n < max)
		part[n++] = trim(start);
	return n;
}

/* the word w at top level in s (as a whole word): its position or NULL */
static char *
find_word(char *s, const char *w)
{
	size_t n = strlen(w);
	int depth = 0;
	char *p;

	for (p = s; *p; p++) {
		if (*p == '[' || *p == '(')
			depth++;
		else if (*p == ']' || *p == ')')
			depth--;
		else if (depth == 0 && !strncmp(p, w, n) && (p == s ||
		    !isalnum((unsigned char)p[-1])) && !isalnum((unsigned char)p[n]))
			return p;
	}
	return NULL;
}

static char *
find_arrow(char *s)
{
	int depth = 0;
	char *p;

	for (p = s; *p; p++) {
		if (*p == '[' || *p == '(')
			depth++;
		else if (*p == ']' || *p == ')')
			depth--;
		else if (depth == 0 && p[0] == '-' && p[1] == '>')
			return p;
	}
	return NULL;
}

static const char *
mat_str(const struct zmat *m)
{
	static struct sbuf b[4];
	static int k;

	k = (k + 1) & 3;
	sb_clear(&b[k]);
	zm_print(&b[k], m);
	return sb_str(&b[k]);
}

static const char *
vec_str(const zint *v, int n)
{
	static struct sbuf b[4];
	static int k;
	int i;

	k = (k + 1) & 3;
	sb_clear(&b[k]);
	sb_putc(&b[k], '(');
	for (i = 0; i < n; i++) {
		if (i)
			sb_puts(&b[k], ", ");
		z_print(&b[k], v[i]);
	}
	sb_putc(&b[k], ')');
	return sb_str(&b[k]);
}

static void need_matrix(struct zmat *m, const char *s, const char *what);
static const char *bp;
static zint bexpr(void);
static void bskip(void);

static void
need_vector(zint **v, int *n, const char *s, const char *what)
{
	struct zmat m;

	need_matrix(&m, s, what);
	if (m.r > 1)
		fail("%s: expected a vector like [1, 2], got \"%s\"", what, s);
	*n = m.c;
	*v = xcalloc((size_t)m.c + 1, sizeof **v);
	memcpy(*v, m.a, (size_t)m.c * sizeof **v);
	zm_free(&m);
}

/* a group: "Z/4 + Z/6", "[[2, 0], [0, 3]]", "coker [[...]]" */
static void
need_group(struct fgab *G, char *s, const char *what)
{
	const char *e;

	s = trim(s);
	memset(G, 0, sizeof *G);
	if (!strncmp(s, "coker", 5) || s[0] == '[') {
		struct zmat m;
		if (!strncmp(s, "coker", 5))
			s = trim(s + (strncmp(s, "cokernel", 8) ? 5 : 8));
		need_matrix(&m, s, what);
		ab_present(G, &m);
		zm_free(&m);
		return;
	}
	if (!ab_parse(G, s, &e) || *trim((char *)e) != 0)
		fail("%s: expected an abelian group like Z/4 + Z/6, Z^2 + Z/3, 0 "
		    "or coker [[2, 0], [0, 3]], got \"%s\"", what, s);
}

static void
rep_invariants(const char *key, const struct fgab *G)
{
	int i;

	rep_list(key);
	for (i = 0; i < G->ninv; i++)
		rep_str(NULL, z_str(G->d[G->ntriv + i]));
	rep_end();
}

/* the generators of a group as text: "e1 = (1, 0) of order 2, ..." */
static void
print_gens(const struct fgab *G)
{
	zint *x = xcalloc((size_t)G->n + 1, sizeof *x);
	int i;

	if (G->ninv == 0)
		return;
	printf("  generators:");
	for (i = 0; i < G->ninv; i++) {
		zint d = G->d[G->ntriv + i];
		ab_gen(G, i, x);
		ab_reduce(G, x);
		printf("%s %s of order %s", i ? "," : "", vec_str(x, G->n),
		    d == 0 ? "infinity" : z_str(d));
	}
	printf("\n");
	free(x);
}

/* ---- bigint: a small exact evaluator ---- */


static zint bexpr(void);

static void
bskip(void)
{
	while (isspace((unsigned char)*bp))
		bp++;
}

static zint
bprimary(void)
{
	zint v;
	const char *e;

	bskip();
	if (*bp == '(') {
		bp++;
		v = bexpr();
		bskip();
		if (*bp != ')')
			fail("bigint: missing )");
		bp++;
		return v;
	}
	if (isdigit((unsigned char)*bp)) {
		if (!z_parse(bp, &v, &e))
			fail("bigint: bad number");
		bp = e;
		return v;
	}
	if (isalpha((unsigned char)*bp)) {
		char name[16];
		int n = 0, k = 0;
		zint arg[2];
		while (isalnum((unsigned char)*bp) && n < 15)
			name[n++] = *bp++;
		name[n] = 0;
		bskip();
		if (*bp != '(')
			fail("bigint: %s is not a function (gcd, lcm, abs, isprime, "
			    "binomial)", name);
		bp++;
		for (;;) {
			if (k == 2)
				fail("bigint: too many arguments");
			arg[k++] = bexpr();
			bskip();
			if (*bp == ',') {
				bp++;
				continue;
			}
			if (*bp != ')')
				fail("bigint: missing )");
			bp++;
			break;
		}
		if (!strcmp(name, "gcd") && k == 2)
			return z_gcd(arg[0], arg[1]);
		if (!strcmp(name, "lcm") && k == 2)
			return z_lcm(arg[0], arg[1]);
		if (!strcmp(name, "abs") && k == 1)
			return z_abs(arg[0]);
		if (!strcmp(name, "isprime") && k == 1)
			return z_isprime(arg[0]) ? Z_ONE : 0;
		if (!strcmp(name, "binomial") && k == 2) {
			int64_t nn = z_to_i64(arg[0]), kk = z_to_i64(arg[1]), i;
			zint r = Z_ONE;
			if (kk < 0 || kk > nn || nn > 1000000)
				return 0;
			for (i = 1; i <= kk; i++)
				r = z_divexact(z_mul(r, z_from_i64(nn - kk + i)),
				    z_from_i64(i));
			return r;
		}
		fail("bigint: unknown function %s/%d", name, k);
	}
	fail("bigint: unexpected \"%s\"", bp);
	return 0;
}

static zint
bpostfix(void)
{
	zint v = bprimary();

	bskip();
	while (*bp == '!') {
		int64_t n = z_to_i64(v), i;
		if (n < 0 || n > 100000)
			fail("bigint: factorial of %lld", (long long)n);
		v = Z_ONE;
		for (i = 2; i <= n; i++)
			v = z_mul(v, z_from_i64(i));
		bp++;
		bskip();
	}
	return v;
}

static zint bunary(void);

static zint
bpower(void)
{
	zint v = bpostfix();

	bskip();
	if (*bp == '^' || (bp[0] == '*' && bp[1] == '*')) {
		zint e;
		bp += *bp == '^' ? 1 : 2;
		e = bunary();
		if (z_sign(e) < 0)
			fail("bigint: negative exponent");
		if (z_bits(e) > 40 && !(z_is_pm1(v) || v == 0))
			fail("bigint: the exponent %s is too large", z_str(e));
		return z_pow(v, (uint64_t)z_to_i64(e));
	}
	return v;
}

static zint
bunary(void)
{
	bskip();
	if (*bp == '-') {
		bp++;
		return z_neg(bunary());
	}
	if (*bp == '+') {
		bp++;
		return bunary();
	}
	return bpower();
}

static zint
bterm(void)
{
	zint v = bunary(), w, q, r;

	for (;;) {
		bskip();
		if ((bp[0] == '/' && bp[1] == '/') || (!strncmp(bp, "div", 3) &&
		    !isalnum((unsigned char)bp[3]))) {
			bp += bp[0] == '/' ? 2 : 3;
			w = bunary();
			v = z_div_floor(v, w);
		} else if (*bp == '/') {
			bp++;
			w = bunary();
			z_divmod_trunc(v, w, &q, &r);
			if (r != 0)
				fail("bigint: %s / %s is not exact (div is floor "
				    "division, %% the remainder)", z_str(v), z_str(w));
			v = q;
		} else if (*bp == '%' || !strncmp(bp, "mod", 3)) {
			bp += *bp == '%' ? 1 : 3;
			w = bunary();
			v = z_mod(v, w);
		} else if (*bp == '*' && bp[1] != '*') {
			bp++;
			v = z_mul(v, bunary());
		} else
			return v;
	}
}

static zint
bexpr(void)
{
	zint v = bterm();

	for (;;) {
		bskip();
		if (*bp == '+') {
			bp++;
			v = z_add(v, bterm());
		} else if (*bp == '-') {
			bp++;
			v = z_sub(v, bterm());
		} else
			return v;
	}
}

static void
cmd_bigint(char *s)
{
	zint v;

	s = trim(s);
	if (!strncmp(s, "factor ", 7)) {
		zint p[128];
		int e[128], n, i, comp;
		zint prod = Z_ONE;
		bp = s + 7;
		v = bexpr();
		bskip();
		if (*bp)
			fail("bigint: unexpected \"%s\"", bp);
		if (v == 0)
			fail("bigint factor: 0 has no factorization");
		n = z_factor(v, p, e, 128, &comp);
		printf("%s = %s", z_str(v), z_sign(v) < 0 ? "-" : "");
		rep_str("value", z_str(v));
		rep_list("factors");
		for (i = 0; i < n; i++) {
			printf("%s%s", i ? " * " : "", z_str(p[i]));
			if (e[i] > 1)
				printf("^%d", e[i]);
			prod = z_mul(prod, z_pow(p[i], (uint64_t)e[i]));
			rep_obj(NULL);
			rep_str("prime", z_str(p[i]));
			rep_int("exponent", e[i]);
			rep_end();
		}
		rep_end();
		if (n == 0)
			printf("1");
		printf("%s\n", !comp ? " (not fully factored)" : n == 1 &&
		    e[0] == 1 ? " (prime)" : "");
		printf("  how: trial division, then Pollard-Brent rho; primality "
		    "by Miller-Rabin (deterministic below 2^64, probable above)\n");
		if (!comp)
			printf("  not complete: a composite factor above 2^64 was not "
			    "split\n");
		printf("  check: the product is %s%s\n", z_eq(prod, z_abs(v)) ?
		    "the number" : "NOT the number", comp ? "; each factor "
		    "passes the primality test" : "");
		return;
	}
	bp = s;
	v = bexpr();
	bskip();
	if (*bp)
		fail("bigint: unexpected \"%s\"", bp);
	printf("%s = %s\n", s, z_str(v));
	rep_str("value", z_str(v));
	printf("  bits: %zu\n", z_bits(v));
	printf("  how: exact arbitrary-precision integers (32-bit limbs; "
	    "Knuth division)\n");
}

/* a matrix whose entries are bigint expressions: [[2^62, 1], [3, 4]] */
static void
need_matrix(struct zmat *m, const char *s, const char *what)
{
	zint *v = NULL;
	int nv = 0, cap = 0, r = 0, c = -1, row, nested;

	memset(m, 0, sizeof *m);
	bp = s;
	bskip();
	if (*bp != '[')
		goto bad;
	bp++;
	bskip();
	nested = *bp == '[';
	if (!nested)
		bp = s, bskip();
	for (;;) {
		bskip();
		if (*bp != '[')
			goto bad;
		bp++;
		row = 0;
		bskip();
		while (*bp != ']') {
			zint x = bexpr();
			if (nv == cap) {
				cap = cap ? cap * 2 : 16;
				v = xrealloc(v, (size_t)cap * sizeof *v);
			}
			v[nv++] = x;
			row++;
			bskip();
			if (*bp == ',')
				bp++;
			else if (*bp != ']')
				goto bad;
			bskip();
		}
		bp++;
		if (c < 0)
			c = row;
		else if (row != c)
			fail("%s: the rows have different lengths", what);
		r++;
		if (!nested)
			break;
		bskip();
		if (*bp == ',')
			bp++;
		else if (*bp == ']') {
			bp++;
			break;
		} else
			goto bad;
	}
	bskip();
	if (*bp)
		goto bad;
	zm_init(m, c ? r : 0, c);
	if (nv)
		memcpy(m->a, v, (size_t)nv * sizeof *v);
	free(v);
	return;
bad:
	free(v);
	fail("%s: expected an integer matrix like [[1, 2], [3, 4]], got \"%s\"",
	    what, s);
}

/* ---- matrices ---- */

static void
cmd_snf(char *s)
{
	struct zmat A;
	struct zsnf S;
	int i, n;
	struct fgab C;

	need_matrix(&A, trim(s), "snf");
	zm_snf(&A, &S);
	n = A.r < A.c ? A.r : A.c;
	printf("Smith normal form of the %dx%d matrix: diag(", A.r, A.c);
	rep_list("diagonal");
	for (i = 0; i < n; i++) {
		printf("%s%s", i ? ", " : "", z_str(ZM(&S.D, i, i)));
		rep_str(NULL, z_str(ZM(&S.D, i, i)));
	}
	rep_end();
	printf("), rank %d\n", S.rank);
	printf("  D: %s\n", mat_str(&S.D));
	printf("  U: %s\n", mat_str(&S.U));
	printf("  V: %s\n", mat_str(&S.V));
	printf("  U^-1: %s\n", mat_str(&S.Ui));
	printf("  V^-1: %s\n", mat_str(&S.Vi));
	memset(&C, 0, sizeof C);
	ab_present(&C, &A);
	printf("  cokernel: Z^%d / (columns) = %s\n", A.r, ab_str(&C));
	printf("  check: U A V = D, U U^-1 = I and V V^-1 = I (unimodular), "
	    "d_i | d_(i+1): %s\n", zm_snf_check(&A, &S) ? "verified" : "FAILED");
	ab_free(&C);
	zsnf_free(&S);
	zm_free(&A);
}

static void
cmd_hnf(char *s)
{
	struct zmat A, H, U;
	int rank, *piv;

	need_matrix(&A, trim(s), "hnf");
	memset(&H, 0, sizeof H);
	memset(&U, 0, sizeof U);
	piv = xcalloc((size_t)A.r + 1, sizeof *piv);
	rank = zm_hnf(&A, &H, &U, piv);
	printf("Hermite normal form (rows) of the %dx%d matrix: rank %d\n", A.r,
	    A.c, rank);
	printf("  H: %s\n", mat_str(&H));
	printf("  U: %s\n", mat_str(&U));
	printf("  check: U A = H, det U = +-1, H in echelon form with positive "
	    "pivots and reduced entries above them: %s\n", zm_hnf_check(&A, &H,
	    &U) ? "verified" : "FAILED");
	free(piv);
	zm_free(&A);
	zm_free(&H);
	zm_free(&U);
}

static void
cmd_zmat(char *s)
{
	static const char *const ops[] = {"det", "rank", "kernel", "image",
		"inverse", "cokernel", "solve", "snf", "hnf", NULL};
	char *part[3];
	int op = -1, k, np;
	struct zmat A, X;

	s = trim(s);
	for (k = 0; ops[k]; k++)
		if (!strncmp(s, ops[k], strlen(ops[k])) &&
		    !isalnum((unsigned char)s[strlen(ops[k])])) {
			op = k;
			s = trim(s + strlen(ops[k]));
			break;
		}
	if (op == 7) {
		cmd_snf(s);
		return;
	}
	if (op == 8) {
		cmd_hnf(s);
		return;
	}
	np = split_commas(s, part, 3);
	need_matrix(&A, part[0], "zmat");
	memset(&X, 0, sizeof X);
	if (op < 0) {
		struct zsnf S;
		int i, n = A.r < A.c ? A.r : A.c;
		printf("integer matrix %dx%d: rank %d", A.r, A.c, zm_rank(&A));
		if (A.r == A.c)
			printf(", det %s", z_str(zm_det(&A)));
		printf("\n");
		zm_snf(&A, &S);
		printf("  smith diagonal:");
		for (i = 0; i < n; i++)
			printf(" %s", z_str(ZM(&S.D, i, i)));
		printf("\n");
		zm_kernel(&A, &X);
		printf("  kernel basis (columns): %s\n", mat_str(&X));
		zm_image(&A, &X);
		printf("  image basis (columns): %s\n", mat_str(&X));
		zsnf_free(&S);
		printf("  check: A K = 0; the Smith form verified (U A V = D)\n");
	} else if (op == 0) {
		zint d;
		struct qmat q;
		if (A.r != A.c)
			fail("det: the matrix is %dx%d, not square", A.r, A.c);
		d = zm_det(&A);
		printf("det = %s\n", z_str(d));
		rep_str("value", z_str(d));
		printf("  how: Bareiss fraction-free elimination\n");
		qm_from_zmat(&q, &A);
		printf("  check: Gaussian elimination over Q gives %s\n",
		    zq_str(qm_det(&q)));
		qm_free(&q);
	} else if (op == 1) {
		int r = zm_rank(&A);
		printf("rank = %d\n", r);
		rep_int("value", r);
		printf("  how: Bareiss fraction-free elimination\n");
		{
			struct zsnf S;
			int rr = zm_snf(&A, &S);
			zsnf_free(&S);
			printf("  check: the Smith form has %d nonzero invariant "
			    "factors\n", rr);
		}
	} else if (op == 2) {
		int d = zm_kernel(&A, &X);
		printf("kernel of the %dx%d matrix over Z: rank %d\n", A.r, A.c, d);
		printf("  basis (columns): %s\n", mat_str(&X));
		printf("  how: U A^T = H (Hermite); the rows of U past the rank; a "
		    "saturated basis (Z^%d / ker is torsion free)\n", A.c);
		{
			struct zmat P;
			zm_init(&P, 0, 0);
			zm_mul(&P, &A, &X);
			printf("  check: A K = %s\n", zm_is_zero(&P) ? "0" : "NONZERO");
			zm_free(&P);
		}
	} else if (op == 3) {
		int d = zm_image(&A, &X);
		struct fgab C;
		printf("image of the %dx%d matrix (the lattice of its columns): "
		    "rank %d\n", A.r, A.c, d);
		printf("  basis (columns, Hermite form): %s\n", mat_str(&X));
		memset(&C, 0, sizeof C);
		ab_present(&C, &A);
		printf("  index / quotient: Z^%d / image = %s\n", A.r, ab_str(&C));
		ab_free(&C);
		printf("  check: each column of A lies in the lattice and each basis "
		    "vector in the column span: %s\n", zm_lattice_eq(&A, &X) ?
		    "verified" : "FAILED");
	} else if (op == 4) {
		if (!zm_inverse(&A, &X)) {
			printf("not invertible over Z: det = %s (must be +-1)\n",
			    A.r == A.c ? z_str(zm_det(&A)) : "(not square)");
			rep_bool("invertible", 0);
		} else {
			struct zmat P;
			printf("inverse over Z: %s\n", mat_str(&X));
			rep_bool("invertible", 1);
			rep_str("inverse", mat_str(&X));
			zm_init(&P, 0, 0);
			zm_mul(&P, &A, &X);
			printf("  check: A A^-1 = %s\n", zm_is_identity(&P) ? "I" :
			    "NOT I");
			zm_free(&P);
		}
	} else if (op == 5) {
		struct fgab C;
		memset(&C, 0, sizeof C);
		ab_present(&C, &A);
		printf("Z^%d / (columns) = %s\n", A.r, ab_str(&C));
		rep_str("group", ab_str(&C));
		rep_invariants("invariant_factors", &C);
		print_gens(&C);
		printf("  how: the Smith normal form of the relation matrix\n");
		ab_free(&C);
	} else if (op == 6) {
		struct zmat b;
		if (np < 2)
			fail("zmat solve: M, b");
		need_matrix(&b, part[1], "zmat solve");
		if (b.r == 1 && b.c == A.r && A.r != 1) {
			struct zmat t;
			zm_init(&t, 0, 0);
			zm_transpose(&t, &b);
			zm_move(&b, &t);
		}
		if (!zm_solve(&A, &b, &X)) {
			printf("no integer solution of A x = b\n");
			rep_bool("solvable", 0);
			{
				struct qmat qa, qb, qx;
				qm_from_zmat(&qa, &A);
				qm_from_zmat(&qb, &b);
				memset(&qx, 0, sizeof qx);
				if (qm_solve(&qa, &qb, &qx)) {
					struct sbuf t = {0};
					qm_print(&t, &qx);
					printf("  over Q: x = %s\n", sb_str(&t));
					free(t.p);
					qm_free(&qx);
				} else
					printf("  not even over Q\n");
				qm_free(&qa);
				qm_free(&qb);
			}
			printf("  how: U A V = D (Smith); D y = U b has no integer "
			    "solution\n");
		} else {
			struct zmat P, K;
			printf("x = %s\n", mat_str(&X));
			rep_bool("solvable", 1);
			rep_str("x", mat_str(&X));
			memset(&K, 0, sizeof K);
			if (zm_kernel(&A, &K) > 0)
				printf("  general solution: x + K t, K = %s\n",
				    mat_str(&K));
			zm_init(&P, 0, 0);
			zm_mul(&P, &A, &X);
			printf("  check: A x = %s\n", zm_eq(&P, &b) ? "b" : "NOT b");
			zm_free(&P);
			zm_free(&K);
		}
		zm_free(&b);
	}
	zm_free(&A);
	zm_free(&X);
}

static void
cmd_qmat(char *s)
{
	static const char *const ops[] = {"rref", "det", "rank", "inverse",
		"nullspace", "solve", NULL};
	char *part[3];
	int op = 0, k, np;
	struct qmat A, X;
	struct sbuf b = {0};
	const char *e;

	s = trim(s);
	for (k = 0; ops[k]; k++)
		if (!strncmp(s, ops[k], strlen(ops[k])) &&
		    !isalnum((unsigned char)s[strlen(ops[k])])) {
			op = k;
			s = trim(s + strlen(ops[k]));
			break;
		}
	np = split_commas(s, part, 3);
	memset(&A, 0, sizeof A);
	memset(&X, 0, sizeof X);
	if (!qm_parse(&A, part[0], &e) || *trim((char *)e))
		fail("qmat: expected a rational matrix like [[1/2, 1], [3, 4]]");
	switch (op) {
	case 0: {
		int *piv = xcalloc((size_t)A.r + 1, sizeof *piv), r;
		qm_init(&X, 0, 0);
		qm_free(&X);
		X = A;
		memset(&A, 0, sizeof A);
		r = qm_rref(&X, piv);
		qm_print(&b, &X);
		printf("rref = %s\n  rank: %d\n", sb_str(&b), r);
		rep_str("rref", sb_str(&b));
		free(piv);
		break;
	}
	case 1:
		printf("det = %s\n", zq_str(qm_det(&A)));
		rep_str("value", zq_str(qm_det(&A)));
		break;
	case 2:
		printf("rank = %d\n", qm_rank(&A));
		break;
	case 3:
		if (!qm_inverse(&A, &X))
			printf("singular: no inverse\n");
		else {
			qm_print(&b, &X);
			printf("inverse = %s\n  check: A A^-1 = I verified\n",
			    sb_str(&b));
			rep_str("inverse", sb_str(&b));
		}
		break;
	case 4: {
		int d = qm_nullspace(&A, &X);
		qm_print(&b, &X);
		printf("nullspace over Q: dimension %d\n  basis (columns): %s\n",
		    d, sb_str(&b));
		break;
	}
	case 5: {
		struct qmat B;
		if (np < 2 || !qm_parse(&B, part[1], &e))
			fail("qmat solve: M, b");
		if (B.r == 1 && B.c == A.r && A.r != 1) {
			struct qmat t;
			int i;
			qm_init(&t, B.c, 1);
			for (i = 0; i < B.c; i++)
				qm_set(&t, i, 0, qm_get(&B, 0, i));
			qm_free(&B);
			B = t;
		}
		if (!qm_solve(&A, &B, &X))
			printf("no solution over Q\n");
		else {
			struct qmat P;
			int i, j, ok = 1;
			qm_print(&b, &X);
			printf("x = %s\n", sb_str(&b));
			rep_str("x", sb_str(&b));
			qm_init(&P, 0, 0);
			qm_mul(&P, &A, &X);
			for (i = 0; i < P.r; i++)
				for (j = 0; j < P.c; j++)
					if (!zq_eq(qm_get(&P, i, j), qm_get(&B, i, j)))
						ok = 0;
			printf("  check: A x = %s\n", ok ? "b" : "NOT b");
			qm_free(&P);
		}
		qm_free(&B);
		break;
	}
	}
	free(b.p);
	qm_free(&A);
	qm_free(&X);
}

/* ---- abelian groups ---- */

static void
cmd_abgroup(char *s)
{
	struct fgab G, T;
	struct sbuf b = {0};
	zint o;

	need_group(&G, s, "abgroup");
	printf("%s = %s\n", trim(s), ab_str(&G));
	rep_str("group", ab_str(&G));
	rep_invariants("invariant_factors", &G);
	ab_print_primary(&b, &G);
	printf("  primary decomposition: %s\n", sb_str(&b));
	o = ab_order(&G);
	printf("  order: %s\n", o == 0 ? "infinite" : z_str(o));
	printf("  rank: %d\n", G.rank);
	if (G.rank == 0)
		printf("  exponent: %s\n", z_str(ab_exponent(&G)));
	memset(&T, 0, sizeof T);
	ab_torsion(&T, &G);
	printf("  torsion subgroup: %s\n", ab_str(&T));
	printf("  cyclic: %s\n  free: %s\n", ab_is_cyclic(&G) ? "yes" : "no",
	    ab_is_free(&G) ? "yes" : "no");
	print_gens(&G);
	printf("  how: the Smith normal form of the relation matrix (U R V = D, "
	    "verified)\n");
	free(b.p);
	ab_free(&G);
	ab_free(&T);
}

static void
print_map(const char *name, const struct fgab *G, const struct fgab *H,
    const struct zmat *M)
{
	zint *x = xcalloc((size_t)G->n + 1, sizeof *x);
	zint *y = xcalloc((size_t)H->n + 1, sizeof *y);
	int j;

	printf("  %s:", name);
	for (j = 0; j < G->a; j++) {
		int i;
		for (i = 0; i < G->n; i++)
			x[i] = ZM(&G->A, i, j);
		ab_hom_apply(G, H, M, x, y);
		ab_reduce(H, y);
		printf("%s %s -> %s", j ? "," : "", vec_str(x, G->n),
		    vec_str(y, H->n));
	}
	printf("\n");
	free(x);
	free(y);
}

static void
cmd_abiso(char *s)
{
	char *part[3];
	struct fgab G, H;
	struct zmat phi, psi;

	if (split_commas(s, part, 3) != 2)
		fail("abiso: two groups, G, H");
	need_group(&G, part[0], "abiso");
	need_group(&H, part[1], "abiso");
	memset(&phi, 0, sizeof phi);
	memset(&psi, 0, sizeof psi);
	if (!ab_isomorphic(&G, &H, &phi, &psi)) {
		printf("%s and %s are not isomorphic\n", part[0], part[1]);
		rep_bool("isomorphic", 0);
		printf("  invariant factors: %s versus %s\n", ab_str(&G),
		    ab_str(&H));
		printf("  how: the invariant factors of a finitely generated "
		    "abelian group determine it up to isomorphism\n");
	} else {
		printf("%s = %s: isomorphic, both %s\n", part[0], part[1],
		    ab_str(&G));
		rep_bool("isomorphic", 1);
		rep_str("phi_matrix", mat_str(&phi));
		rep_str("psi_matrix", mat_str(&psi));
		print_map("phi", &G, &H, &phi);
		print_map("psi = phi^-1", &H, &G, &psi);
		printf("  check: phi and psi are well defined (relations go to "
		    "relations), psi phi = 1 and phi psi = 1 on generators\n");
	}
	ab_free(&G);
	ab_free(&H);
	zm_free(&phi);
	zm_free(&psi);
}

static void
cmd_abfunctor(char *s, int which)
{
	static const char *const nm[] = {"Hom", "tensor", "Tor", "Ext"};
	char *part[3];
	struct fgab G, H, R;

	if (split_commas(s, part, 3) != 2)
		fail("%s: two groups, G, H", nm[which]);
	need_group(&G, part[0], nm[which]);
	need_group(&H, part[1], nm[which]);
	memset(&R, 0, sizeof R);
	switch (which) {
	case 0:
		ab_hom_group(&R, &G, &H);
		printf("Hom(%s, %s) = %s\n", part[0], part[1], ab_str(&R));
		printf("  how: the coordinate matrices M with M R_G inside R_H, "
		    "modulo those with every column in R_H\n");
		break;
	case 1:
		ab_tensor(&R, &G, &H);
		printf("(%s) (x) (%s) = %s\n", part[0], part[1], ab_str(&R));
		printf("  how: Z^(a b) / (R_G (x) 1, 1 (x) R_H)\n");
		break;
	case 2:
		ab_tor(&R, &G, &H);
		printf("Tor(%s, %s) = %s\n", part[0], part[1], ab_str(&R));
		printf("  how: ker(R (x) 1_H) for the free resolution 0 -> Z^b -R-> "
		    "Z^a -> G -> 0\n");
		break;
	case 3:
		ab_ext(&R, &G, &H);
		printf("Ext(%s, %s) = %s\n", part[0], part[1], ab_str(&R));
		printf("  how: coker Hom(R, H) for the free resolution 0 -> Z^b -R-> "
		    "Z^a -> G -> 0\n");
		break;
	}
	rep_str("group", ab_str(&R));
	rep_invariants("invariant_factors", &R);
	ab_free(&G);
	ab_free(&H);
	ab_free(&R);
}

/* "G -> H by M" */
static void
cmd_abmap(char *s)
{
	char *arrow = find_arrow(s), *by;
	struct fgab G, H, K, I, C;
	struct zmat M, inv;
	int inj, sur;

	by = find_word(s, "by");
	if (!arrow || !by || by < arrow)
		fail("abmap: G -> H by M (M: the images of G's generators, as "
		    "the columns, in H's generators)");
	*arrow = 0;
	*by = 0;
	need_group(&G, s, "abmap");
	need_group(&H, arrow + 2, "abmap");
	need_matrix(&M, trim(by + 2), "abmap");
	if (M.r != H.a || M.c != G.a)
		fail("abmap: a map from %d generators to %d needs a %dx%d matrix "
		    "(column j: the image of generator j)", G.a, H.a, H.a, G.a);
	if (!ab_hom_check(&G, &H, &M)) {
		printf("not a homomorphism: a relation of %s is not sent to a "
		    "relation of %s\n", ab_str(&G), ab_str(&H));
		rep_bool("well_defined", 0);
		goto out;
	}
	rep_bool("well_defined", 1);
	memset(&K, 0, sizeof K);
	memset(&I, 0, sizeof I);
	memset(&C, 0, sizeof C);
	memset(&inv, 0, sizeof inv);
	ab_kernel(&K, NULL, &G, &H, &M);
	ab_image(&I, NULL, &G, &H, &M);
	ab_cokernel(&C, NULL, &G, &H, &M);
	inj = ab_is_trivial(&K);
	sur = ab_is_trivial(&C);
	printf("a homomorphism %s -> %s\n", ab_str(&G), ab_str(&H));
	printf("  kernel: %s\n", ab_str(&K));
	print_gens(&K);
	printf("  image: %s\n", ab_str(&I));
	printf("  cokernel: %s\n", ab_str(&C));
	printf("  injective: %s\n  surjective: %s\n", inj ? "yes" : "no",
	    sur ? "yes" : "no");
	rep_str("kernel_group", ab_str(&K));
	rep_str("image_group", ab_str(&I));
	rep_str("cokernel_group", ab_str(&C));
	if (inj && sur && ab_hom_iso(&G, &H, &M, &inv))
		printf("  inverse: %s\n", mat_str(&inv));
	{
		struct fgab Q;
		int ok;
		memset(&Q, 0, sizeof Q);
		ab_quotient(&Q, &G, &K);
		ok = ab_isomorphic(&Q, &I, NULL, NULL);
		ab_free(&Q);
		printf("  check: well defined (M R_G inside R_H); G / ker = im: "
		    "%s\n", ok ? "isomorphic, as it must be" : "FAILED");
	}
	ab_free(&K);
	ab_free(&I);
	ab_free(&C);
	zm_free(&inv);
out:
	ab_free(&G);
	ab_free(&H);
	zm_free(&M);
}

static void
cmd_abelem(char *s)
{
	char *part[3];
	struct fgab G;
	zint *x, *y, o;
	int n;

	if (split_commas(s, part, 3) != 2)
		fail("abelem: G, [x1, x2, ...]");
	need_group(&G, part[0], "abelem");
	need_vector(&x, &n, part[1], "abelem");
	if (n != G.n)
		fail("abelem: the group's elements have %d coordinates", G.n);
	y = xcalloc((size_t)G.ninv + 1, sizeof *y);
	if (!ab_canon(&G, x, y))
		fail("abelem: not an element");
	o = ab_elem_order(&G, x);
	printf("%s in %s: canonical form %s\n", vec_str(x, n), ab_str(&G),
	    vec_str(y, G.ninv));
	rep_str("canonical", vec_str(y, G.ninv));
	printf("  order: %s\n", o == 0 ? "infinite" : z_str(o));
	printf("  zero: %s\n", ab_elem_is_zero(&G, x) ? "yes" : "no");
	if (o != 0)
		printf("  check: %s * x is a combination of the relations\n",
		    z_str(o));
	free(x);
	free(y);
	ab_free(&G);
}

static void
cmd_absub(char *s)
{
	char *part[64];
	int np, i, j, n;
	struct fgab G, S, Q;
	struct zmat gens;

	np = split_commas(s, part, 64);
	if (np < 2)
		fail("absub: G, x1, x2, ...");
	need_group(&G, part[0], "absub");
	if (np == 2 && !strncmp(trim(part[1]), "[[", 2)) {
		/* G, [[x1], [x2], ...]: the generators as rows */
		struct zmat rows;
		need_matrix(&rows, part[1], "absub");
		if (rows.c != G.n)
			fail("absub: the group's elements have %d coordinates", G.n);
		zm_init(&gens, 0, 0);
		zm_transpose(&gens, &rows);
		zm_free(&rows);
		goto have;
	}
	zm_init(&gens, G.n, np - 1);
	for (j = 1; j < np; j++) {
		zint *x;
		need_vector(&x, &n, part[j], "absub");
		if (n != G.n)
			fail("absub: the group's elements have %d coordinates", G.n);
		for (i = 0; i < n; i++)
			ZM(&gens, i, j - 1) = x[i];
		free(x);
	}
have:
	memset(&S, 0, sizeof S);
	memset(&Q, 0, sizeof Q);
	ab_sub(&S, &G, &gens);
	ab_quotient(&Q, &G, &S);
	printf("the subgroup generated in %s: %s\n", ab_str(&G), ab_str(&S));
	rep_str("subgroup", ab_str(&S));
	printf("  quotient: %s\n", ab_str(&Q));
	if (ab_is_finite(&Q))
		printf("  index: %s\n", z_str(ab_order(&Q)));
	else
		printf("  index: infinite\n");
	print_gens(&S);
	printf("  check: every generator is a member (with a witness): ");
	{
		zint *x = xcalloc((size_t)G.n + 1, sizeof *x);
		zint *w = xcalloc((size_t)S.a + 1, sizeof *w);
		int ok = 1;
		for (j = 0; j < gens.c; j++) {
			zm_col(x, &gens, j);
			if (!ab_member(&S, x, w))
				ok = 0;
		}
		printf("%s\n", ok ? "verified" : "FAILED");
		free(x);
		free(w);
	}
	zm_free(&gens);
	ab_free(&G);
	ab_free(&S);
	ab_free(&Q);
}

static void
cmd_abexact(char *s)
{
	char *by = find_word(s, "by"), *gp[3], *mp[3], *a1, *a2;
	struct fgab G[3];
	struct zmat f, g, pre;
	const char *why = NULL;
	int ok;

	if (!by)
		fail("abexact: G1 -> G2 -> G3 by f, g");
	*by = 0;
	a1 = find_arrow(s);
	if (!a1)
		fail("abexact: G1 -> G2 -> G3 by f, g");
	*a1 = 0;
	a2 = find_arrow(a1 + 2);
	if (!a2)
		fail("abexact: G1 -> G2 -> G3 by f, g");
	*a2 = 0;
	gp[0] = s;
	gp[1] = a1 + 2;
	gp[2] = a2 + 2;
	if (split_commas(by + 2, mp, 3) != 2)
		fail("abexact: two maps, f, g");
	need_group(&G[0], gp[0], "abexact");
	need_group(&G[1], gp[1], "abexact");
	need_group(&G[2], gp[2], "abexact");
	need_matrix(&f, mp[0], "abexact");
	need_matrix(&g, mp[1], "abexact");
	/* "[]" is a map to or from 0 */
	if (f.r == 0 && f.c == 0)
		zm_init(&f, G[1].a, G[0].a);
	if (g.r == 0 && g.c == 0)
		zm_init(&g, G[2].a, G[1].a);
	memset(&pre, 0, sizeof pre);
	ok = ab_exact_at(&G[0], &G[1], &G[2], &f, &g, &pre, &why);
	printf("%s -> %s -> %s: %s at %s\n", ab_str(&G[0]), ab_str(&G[1]),
	    ab_str(&G[2]), ok ? "exact" : "not exact", ab_str(&G[1]));
	rep_bool("exact", ok);
	if (!ok)
		printf("  why: %s\n", why);
	else {
		printf("  preimages of the generators of ker g: %s\n",
		    mat_str(&pre));
		printf("  check: g f = 0, and f(preimages) = the generators of ker "
		    "g modulo relations\n");
	}
	ab_free(&G[0]);
	ab_free(&G[1]);
	ab_free(&G[2]);
	zm_free(&f);
	zm_free(&g);
	zm_free(&pre);
}

/* ---- a randomized self-test of the identities (abselftest N) ---- */

static uint64_t rng = 88172645463325252ULL;

static int
rnd(int n)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return (int)(rng % (uint64_t)n);
}

static void
rand_group(struct fgab *G)
{
	struct zmat r;
	int i, g = 1 + rnd(3), k = rnd(4);

	zm_init(&r, g, k);
	for (i = 0; i < g * k; i++)
		r.a[i] = z_from_i64(rnd(3) ? rnd(13) - 6 : 0);
	ab_present(G, &r);
	zm_free(&r);
}

/* the formula for F(G, H) from the cyclic factors: 0 Hom, 1 tensor,
 * 2 Tor, 3 Ext */
static void
formula(struct fgab *R, const struct fgab *G, const struct fgab *H, int f)
{
	zint *d = xcalloc((size_t)(G->ninv * H->ninv) + 1, sizeof *d);
	int i, j, n = 0;

	for (i = 0; i < G->ninv; i++)
		for (j = 0; j < H->ninv; j++) {
			zint a = G->d[G->ntriv + i], b = H->d[H->ntriv + j], v = Z_ONE;
			switch (f) {
			case 0:	/* Hom(Z/a, Z/b) = Z/(a, b); Hom(Z, B) = B */
				v = a == 0 ? b : b == 0 ? Z_ONE : z_gcd(a, b);
				break;
			case 1:	/* Z/a (x) Z/b = Z/(a, b) (gcd(0, b) = b) */
				v = z_gcd(a, b);
				break;
			case 2:	/* Tor(Z/a, Z/b) = Z/(a, b) for a, b != 0 */
				v = a == 0 || b == 0 ? Z_ONE : z_gcd(a, b);
				break;
			case 3:	/* Ext(Z/a, Z/b) = Z/(a, b), Ext(Z/a, Z) = Z/a */
				v = a == 0 ? Z_ONE : b == 0 ? a : z_gcd(a, b);
				break;
			}
			d[n++] = v;
		}
	ab_from_invariants(R, d, n);
	free(d);
}

static void
cmd_selftest(char *s)
{
	int n = atoi(s), t, f, bad = 0, homs = 0;
	struct fgab G, H, R, F, K, I, C, X, Y;
	struct zmat M, inc, proj;
	const char *why;

	if (n <= 0)
		n = 100;
	memset(&G, 0, sizeof G);
	memset(&H, 0, sizeof H);
	memset(&R, 0, sizeof R);
	memset(&F, 0, sizeof F);
	memset(&K, 0, sizeof K);
	memset(&I, 0, sizeof I);
	memset(&C, 0, sizeof C);
	memset(&X, 0, sizeof X);
	memset(&Y, 0, sizeof Y);
	memset(&M, 0, sizeof M);
	memset(&inc, 0, sizeof inc);
	memset(&proj, 0, sizeof proj);
	for (t = 0; t < n; t++) {
		zmark mk = z_mark();
		int i;
		zint m = z_from_i64(2 + rnd(5));
		rand_group(&G);
		rand_group(&H);
		/* Hom, tensor, Tor, Ext against the formulas */
		for (f = 0; f < 4; f++) {
			if (f == 0)
				ab_hom_group(&R, &G, &H);
			else if (f == 1)
				ab_tensor(&R, &G, &H);
			else if (f == 2)
				ab_tor(&R, &G, &H);
			else
				ab_ext(&R, &G, &H);
			formula(&F, &G, &H, f);
			if (!ab_isomorphic(&R, &F, NULL, NULL)) {
				printf("  FAIL %s(%s, %s) = %s, formula %s\n", f == 0 ?
				    "Hom" : f == 1 ? "tensor" : f == 2 ? "Tor" : "Ext",
				    ab_str(&G), ab_str(&H), ab_str(&R), ab_str(&F));
				bad++;
			}
		}
		/* G[m] = Hom(Z/m, G) and m G = G / G[m] */
		ab_torsion_m(&X, &G, m);
		ab_cyclic(&Y, m);
		ab_hom_group(&R, &Y, &G);
		if (!ab_isomorphic(&X, &R, NULL, NULL))
			bad++, printf("  FAIL G[m]\n");
		ab_mulsub(&Y, &G, m);
		ab_quotient(&R, &G, &X);
		if (!ab_isomorphic(&Y, &R, NULL, NULL))
			bad++, printf("  FAIL mG\n");
		/* a random homomorphism G -> H, when well defined */
		zm_init(&M, H.a, G.a);
		for (i = 0; i < H.a * G.a; i++)
			M.a[i] = z_from_i64(rnd(7) - 3);
		if (ab_hom_check(&G, &H, &M)) {
			homs++;
			ab_kernel(&K, &inc, &G, &H, &M);
			ab_image(&I, NULL, &G, &H, &M);
			ab_cokernel(&C, &proj, &G, &H, &M);
			if (!ab_exact_at(&K, &G, &H, &inc, &M, NULL, &why))
				bad++, printf("  FAIL exact at G: %s\n", why);
			if (!ab_exact_at(&G, &H, &C, &M, &proj, NULL, &why))
				bad++, printf("  FAIL exact at H: %s\n", why);
			ab_quotient(&R, &G, &K);
			if (!ab_isomorphic(&R, &I, NULL, NULL))
				bad++, printf("  FAIL G/ker = im\n");
		}
		/* explicit isomorphism with the invariant-factor form */
		{
			zint d[16];
			int k = ab_invariants(&G, d);
			ab_from_invariants(&F, d, k);
			if (!ab_isomorphic(&G, &F, NULL, NULL))
				bad++, printf("  FAIL invariant form\n");
		}
		/* T(G) + G/T(G) and the order of every canonical generator */
		ab_torsion(&X, &G);
		for (i = 0; i < G.ninv; i++) {
			zint *x = xcalloc((size_t)G.n + 1, sizeof *x);
			ab_gen(&G, i, x);
			if (!z_eq(ab_elem_order(&G, x), G.d[G.ntriv + i]))
				bad++, printf("  FAIL generator order\n");
			free(x);
		}
		z_release(mk);
	}
	/* integers: identities on random numbers of 1 .. 600 bits */
	for (t = 0; t < n; t++) {
		zmark mk = z_mark();
		zint a = 0, b = 0, q, r, g, x, y;
		const char *e;
		int i, la = 1 + rnd(20), lb = 1 + rnd(20);
		for (i = 0; i < la; i++)
			a = z_add(z_mul(a, z_from_i64(1 << 30)), z_from_i64(rnd(1 <<
			    30)));
		for (i = 0; i < lb; i++)
			b = z_add(z_mul(b, z_from_i64(1 << 30)), z_from_i64(rnd(1 <<
			    30)));
		if (rnd(2))
			a = z_neg(a);
		if (rnd(2))
			b = z_neg(b);
		if (b == 0)
			b = Z_ONE;
		if (!z_eq(z_sub(z_add(a, b), b), a))
			bad++, printf("  FAIL a + b - b\n");
		if (!z_eq(z_divexact(z_mul(a, b), b), a))
			bad++, printf("  FAIL a b / b\n");
		z_divmod_floor(a, b, &q, &r);
		if (!z_eq(z_add(z_mul(q, b), r), a) || (r != 0 && z_sign(r) !=
		    z_sign(b)) || z_cmpabs(r, b) >= 0)
			bad++, printf("  FAIL floor division\n");
		g = z_xgcd(a, b, &x, &y);
		if (!z_eq(z_add(z_mul(x, a), z_mul(y, b)), g) || !z_divides(g, a) ||
		    !z_divides(g, b))
			bad++, printf("  FAIL xgcd\n");
		if (!z_parse(z_str(a), &q, &e) || *e || !z_eq(q, a))
			bad++, printf("  FAIL print/parse\n");
		z_release(mk);
	}
	printf("abselftest: %d random pairs of groups, %d random "
	    "homomorphisms, %d random integer pairs: %s\n", n, homs, n, bad ?
	    "FAILED" : "all identities hold");
	printf("  checked: Hom, tensor, Tor, Ext against the cyclic formulas; "
	    "G[m] = Hom(Z/m, G); mG = G/G[m]; 0 -> ker -> G -> H -> coker -> "
	    "0 exact (with preimages); G/ker = im; explicit isomorphisms with "
	    "the invariant-factor form; orders of the canonical generators; "
	    "for integers a + b - b, a b / b, floor division, xgcd, printing "
	    "and parsing back\n");
	ab_free(&G);
	ab_free(&H);
	ab_free(&R);
	ab_free(&F);
	ab_free(&K);
	ab_free(&I);
	ab_free(&C);
	ab_free(&X);
	ab_free(&Y);
	zm_free(&M);
	zm_free(&inc);
	zm_free(&proj);
}

/* ---- the entry point ---- */

void absalg_cmd2(const char *w, char *s);   /* below: groups, ideals */
int absalg_is_cmd2(const char *w);

static const struct {
	const char *w;
	int c;
} acmds[] = {
	{"bigint", 1}, {"snf", 2}, {"hnf", 3}, {"zmat", 4}, {"qmat", 5},
	{"abgroup", 6}, {"abiso", 7}, {"abhom", 8}, {"abtensor", 9},
	{"abtor", 10}, {"abext", 11}, {"abmap", 12}, {"abelem", 13},
	{"absub", 14}, {"abexact", 15}, {"abselftest", 16}, {NULL, 0}
};

int
absalg_command(struct P *p)
{
	char w[48];
	uint32_t pos = cmdp_pos(p);
	int k, c = 0;
	zmark m;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; acmds[k].w; k++)
		if (!strcmp(acmds[k].w, w))
			c = acmds[k].c;
	if ((!c && !absalg_is_cmd2(w)) || func_find(sym_cstr(w))) {
		cmdp_seek(p, pos);
		return 0;
	}
	text[0] = 0;
	if (!cmdp_eos(p))
		cmdp_rest(p, text, sizeof text, NULL);
	notes_flush();
	rep_begin(w);
	rep_text_fields();
	m = z_mark();
	switch (c) {
	case 0:
		absalg_cmd2(w, text);
		break;
	case 1:
		cmd_bigint(text);
		break;
	case 2:
		cmd_snf(text);
		break;
	case 3:
		cmd_hnf(text);
		break;
	case 4:
		cmd_zmat(text);
		break;
	case 5:
		cmd_qmat(text);
		break;
	case 6:
		cmd_abgroup(text);
		break;
	case 7:
		cmd_abiso(text);
		break;
	case 8: case 9: case 10: case 11:
		cmd_abfunctor(text, c - 8);
		break;
	case 12:
		cmd_abmap(text);
		break;
	case 13:
		cmd_abelem(text);
		break;
	case 14:
		cmd_absub(text);
		break;
	case 15:
		cmd_abexact(text);
		break;
	case 16:
		cmd_selftest(text);
		break;
	}
	z_release(m);
	return 1;
}

/* ==== the second half: permutation groups, Cayley tables, ideals ==== */

static const char *
perm_str(const uint32_t *p, int n)
{
	static struct sbuf b[4];
	static int k;

	k = (k + 1) & 3;
	sb_clear(&b[k]);
	perm_print(&b[k], p, n);
	return sb_str(&b[k]);
}

static void
need_pgroup(struct pgrp *G, char *s, int n, const char *what)
{
	const char *e;

	s = trim(s);
	memset(G, 0, sizeof *G);
	if (!pg_parse(G, s, n, &e) || *trim((char *)e))
		fail("%s: expected a permutation group: S4, A5, C6, D8, or "
		    "generators in cycle notation \"(1 2 3 4), (1 2)\", got \"%s\"",
		    what, s);
}

/* a permutation of degree n in cycle notation */
static uint32_t *
need_perm(char *s, int n, const char *what)
{
	struct pgrp T;
	const char *e;
	uint32_t *p;

	s = trim(s);
	memset(&T, 0, sizeof T);
	if (!strcmp(s, "()")) {
		int i;
		p = xmalloc((size_t)n * sizeof *p);
		for (i = 0; i < n; i++)
			p[i] = (uint32_t)i;
		return p;
	}
	if (!pg_parse(&T, s, n, &e) || *trim((char *)e) || T.ngens != 1)
		fail("%s: expected one permutation in cycle notation, like "
		    "(1 2 3)(4 5), got \"%s\"", what, s);
	p = xmalloc((size_t)n * sizeof *p);
	memcpy(p, T.gens, (size_t)n * sizeof *p);
	pg_free(&T);
	return p;
}

static void
print_series(const char *label, const zint *o, int len)
{
	int i;

	printf("  %s:", label);
	for (i = 0; i < len && i < 64; i++)
		printf("%s%s", i ? " > " : " ", z_str(o[i]));
	printf("\n");
}

static void
cmd_permgroup(char *s)
{
	struct pgrp G, Z;
	uint32_t *of, *k;
	zint ord[64];
	int i, j, no, len, n;
	u64 o64;

	need_pgroup(&G, s, 0, "permgroup");
	n = G.n;
	printf("%s: a permutation group of degree %d, order %s\n", trim(s), n,
	    z_str(G.order));
	rep_str("order_value", z_str(G.order));
	printf("  generators:");
	for (i = 0; i < G.ngens; i++)
		printf("%s %s", i ? "," : "", perm_str(G.gens + (size_t)i * n, n));
	printf("%s\n", G.ngens ? "" : " none (the trivial group)");
	printf("  base:");
	for (i = 0; i < G.nbase; i++)
		printf("%s %u", i ? "," : "", G.base[i] + 1);
	printf("%s; basic orbit lengths", G.nbase ? "" : " (empty)");
	for (i = 0; i < G.nbase; i++)
		printf(" %llu", (unsigned long long)(G.tro[i + 1] - G.tro[i]));
	printf(" (the order is their product); %d strong generators\n",
	    G.nsgs);
	of = xmalloc((size_t)n * sizeof *of);
	no = pg_orbits(&G, of);
	printf("  orbits:");
	for (j = 0; j < no; j++) {
		int first = 1;
		printf(" {");
		for (i = 0; i < n; i++)
			if (of[i] == (uint32_t)j) {
				printf("%s%d", first ? "" : ", ", i + 1);
				first = 0;
			}
		printf("}");
	}
	printf("%s\n", no == 1 ? " (transitive)" : "");
	printf("  abelian: %s\n", pg_is_abelian(&G) ? "yes" : "no");
	len = pg_derived_series(&G, ord, 64);
	print_series("derived series (orders)", ord, len);
	printf("  solvable: %s\n", z_is_one(ord[(len < 64 ? len : 64) - 1]) ?
	    "yes" : "no");
	rep_bool("solvable_value", z_is_one(ord[(len < 64 ? len : 64) - 1]));
	len = pg_lower_central(&G, ord, 64);
	print_series("lower central series (orders)", ord, len);
	printf("  nilpotent: %s\n", z_is_one(ord[(len < 64 ? len : 64) - 1]) ?
	    "yes" : "no");
	if (z_to_u64(G.order, &o64) && o64 <= 100000) {
		memset(&Z, 0, sizeof Z);
		pg_centre(&Z, &G);
		printf("  centre: order %s", z_str(Z.order));
		if (!z_is_one(Z.order)) {
			printf(", generated by");
			for (i = 0; i < Z.ngens; i++)
				printf("%s %s", i ? "," : "", perm_str(Z.gens +
				    (size_t)i * n, n));
		}
		printf("\n");
		pg_free(&Z);
	} else
		printf("  centre: not computed (it enumerates; the order is above "
		    "100000)\n");
	printf("  how: Schreier-Sims (deterministic): every Schreier generator "
	    "of every level sifts to the identity\n");
	k = xmalloc(((size_t)G.nbase + 1) * sizeof *k);
	{
		int ok = 1;
		for (i = 0; i < G.ngens; i++)
			ok &= pg_contains(&G, G.gens + (size_t)i * n, k);
		printf("  check: every generator sifts through the stabilizer chain "
		    "and its word rebuilds it: %s\n", ok ? "verified" : "FAILED");
	}
	free(k);
	free(of);
	pg_free(&G);
}

/* permelem G, p */
static void
cmd_permelem(char *s)
{
	char *part[3];
	struct pgrp G;
	uint32_t *p, *k;
	int n, l;

	if (split_commas(s, part, 3) < 2)
		fail("permelem: G, p (a group and a permutation)");
	/* the group may itself be a list of generators: the last part is p */
	{
		char *last = strrchr(s, 0);
		(void)last;
	}
	need_pgroup(&G, part[0], 0, "permelem");
	n = G.n;
	p = need_perm(part[1], n, "permelem");
	k = xmalloc(((size_t)G.nbase + 1) * sizeof *k);
	printf("%s: order %s; %s %s\n", perm_str(p, n), z_str(perm_order(p, n)),
	    pg_contains(&G, p, k) ? "in" : "not in", trim(part[0]));
	rep_bool("member", pg_contains(&G, p, NULL));
	rep_str("order_value", z_str(perm_order(p, n)));
	if (pg_contains(&G, p, k)) {
		printf("  word: p =");
		for (l = G.nbase - 1; l >= 0; l--)
			printf(" u%d(%u)", l + 1, k[l] + 1);
		printf(" (u_i(b): the transversal element of level i taking "
		    "the base point to b)\n");
		printf("  check: the product of the transversal elements is p\n");
	} else
		printf("  why: sifting through the stabilizer chain leaves a "
		    "non-identity residue or leaves a basic orbit\n");
	free(p);
	free(k);
	pg_free(&G);
}

/* permsub H in G: groups (S4, or generators) on the same points */
static void
cmd_permsub(char *s)
{
	char *in = find_word(s, "in");
	struct pgrp G, H, N;
	int n, i, normal;

	if (!in)
		fail("permsub: H in G (each a group: S4, or generators in cycle "
		    "notation separated by commas)");
	*in = 0;
	need_pgroup(&G, in + 2, 0, "permsub");
	n = G.n;
	need_pgroup(&H, s, n, "permsub");
	if (!pg_is_subgroup(&H, &G)) {
		printf("not a subgroup: a generator of H is not in G\n");
		rep_bool("subgroup", 0);
		pg_free(&G);
		pg_free(&H);
		return;
	}
	rep_bool("subgroup", 1);
	normal = pg_is_normal(&H, &G);
	printf("H <= G: |H| = %s, |G| = %s, index %s\n", z_str(H.order),
	    z_str(G.order), z_str(pg_index(&G, &H)));
	printf("  normal: %s\n", normal ? "yes" : "no");
	rep_bool("normal", normal);
	memset(&N, 0, sizeof N);
	pg_normal_closure(&N, &G, H.gens, H.ngens);
	printf("  normal closure of H in G: order %s\n", z_str(N.order));
	{
		u64 idx;
		if (z_to_u64(pg_index(&G, &H), &idx) && idx <= 24) {
			uint32_t *r, m = pg_cosets(&G, &H, &r);
			printf("  right coset representatives:");
			for (i = 0; i < (int)m; i++)
				printf("%s %s", i ? "," : "", perm_str(r + (size_t)i * n,
				    n));
			printf("\n");
			free(r);
		}
	}
	if (normal && !z_is_one(H.order)) {
		struct pgrp Q;
		(void)Q;
		printf("  quotient G/H: order %s\n", z_str(pg_index(&G, &H)));
	}
	printf("  check: each generator of H sifts through G's chain%s\n",
	    normal ? "; each conjugate g^-1 h g of generators sifts through H's" :
	    "");
	pg_free(&G);
	pg_free(&H);
	pg_free(&N);
}

/* permhom G -> H by p1, p2, ...: images of G's generators */
static void
cmd_permhom(char *s)
{
	char *arrow = find_arrow(s), *by = find_word(s, "by"), *part[64];
	struct pgrp G, H;
	uint32_t *imgs;
	int np, k, ok;

	if (!arrow || !by || by < arrow)
		fail("permhom: G -> H by images of G's generators");
	*arrow = 0;
	*by = 0;
	for (k = 0; arrow[2 + k] && arrow[2 + k] != ';'; k++)
		;
	need_pgroup(&G, s, 0, "permhom");
	need_pgroup(&H, arrow + 2, 0, "permhom");
	np = split_commas(by + 2, part, 64);
	if (np != G.ngens)
		fail("permhom: G has %d generators, %d images given", G.ngens, np);
	imgs = xmalloc(((size_t)np * H.n + 1) * sizeof *imgs);
	for (k = 0; k < np; k++) {
		uint32_t *p = need_perm(part[k], H.n, "permhom");
		memcpy(imgs + (size_t)k * H.n, p, (size_t)H.n * sizeof *p);
		free(p);
	}
	ok = pg_hom_check(&G, &H, imgs);
	printf("%s\n", ok ? "a homomorphism" : "not a homomorphism");
	rep_bool("homomorphism", ok);
	printf("  how: the graph <(g_i, f(g_i))> in Sym(%d + %d) has order |G| "
	    "= %s exactly when g_i -> f(g_i) extends to a homomorphism\n", G.n,
	    H.n, z_str(G.order));
	free(imgs);
	pg_free(&G);
	pg_free(&H);
}

static void
cmd_cayley(char *s)
{
	struct cgrp C;
	struct pgrp P;
	const char *why = "";
	zint ord[64];
	int a, len;

	if (!cg_parse(&C, trim(s), &why)) {
		printf("not a group: %s\n", why);
		rep_bool("group", 0);
		return;
	}
	rep_bool("group", 1);
	printf("a group of order %d (identity %u)\n", C.n, C.e);
	printf("  checked: closure, identity, inverses, associativity (all %d "
	    "triples)\n", C.n * C.n * C.n);
	printf("  abelian: %s\n", cg_is_abelian(&C) ? "yes" : "no");
	printf("  element orders:");
	for (a = 0; a < C.n; a++)
		printf(" %u", cg_elem_order(&C, (uint32_t)a));
	printf("\n");
	memset(&P, 0, sizeof P);
	cg_to_pgrp(&P, &C);
	len = pg_derived_series(&P, ord, 64);
	printf("  solvable: %s\n", z_is_one(ord[(len < 64 ? len : 64) - 1]) ?
	    "yes" : "no");
	len = pg_lower_central(&P, ord, 64);
	printf("  nilpotent: %s\n", z_is_one(ord[(len < 64 ? len : 64) - 1]) ?
	    "yes" : "no");
	printf("  how: the regular representation (x -> x a) as a permutation "
	    "group, order %s\n", z_str(P.order));
	pg_free(&P);
	cg_free(&C);
}

/* ---- polynomials and ideals ---- */

struct popts {
	int order;
	u64 p;
	int nv;
	char var[MPL_MAXV][16];
};

static int
is_kw_at(const char *s, const char *p, const char *w)
{
	size_t n = strlen(w);

	return !strncmp(p, w, n) && (p == s || !isalnum((unsigned char)p[-1])) &&
	    !isalnum((unsigned char)p[n]);
}

/* the options, from text like "order lex mod 7 vars x, y" */
static void
read_popts(const char *t, struct popts *o)
{
	const char *p = t;

	while (*p) {
		while (*p == ' ' || *p == ',')
			p++;
		if (!strncmp(p, "order", 5)) {
			p += 5;
			while (*p == ' ')
				p++;
			if (!strncmp(p, "grevlex", 7))
				o->order = MPL_GREVLEX, p += 7;
			else if (!strncmp(p, "grlex", 5) || !strncmp(p, "deglex", 6))
				o->order = MPL_GRLEX, p += *p == 'g' ? 5 : 6;
			else if (!strncmp(p, "lex", 3))
				o->order = MPL_LEX, p += 3;
			else
				fail("order: lex, grlex or grevlex");
		} else if (!strncmp(p, "mod", 3) || !strncmp(p, "over", 4)) {
			int over = p[0] == 'o';
			zint v;
			const char *e;
			p += over ? 4 : 3;
			while (*p == ' ')
				p++;
			if (over && (*p == 'Q' || !strncmp(p, "QQ", 2))) {
				o->p = 0;
				while (isalpha((unsigned char)*p))
					p++;
				continue;
			}
			if (over) {
				if (!strncmp(p, "GF(", 3))
					p += 3;
				else if (!strncmp(p, "F_", 2) || !strncmp(p, "Z/", 2))
					p += 2;
				else if (*p == 'F' || *p == 'Z')
					p++;
			}
			if (!z_parse(p, &v, &e) || !z_to_u64(v, &o->p) || !nt_isprime(
			    o->p))
				fail("polynomials: \"mod P\" or \"over GF(P)\" needs a "
				    "prime P < 2^64");
			p = e;
			if (*p == ')')
				p++;
		} else if (!strncmp(p, "vars", 4)) {
			p += 4;
			o->nv = 0;
			for (;;) {
				int n = 0;
				while (*p == ' ' || *p == ',')
					p++;
				if (!isalpha((unsigned char)*p) || is_kw_at(t, p, "order") ||
				    is_kw_at(t, p, "mod") || is_kw_at(t, p, "over"))
					break;
				if (o->nv == MPL_MAXV)
					fail("polynomials: at most %d variables", MPL_MAXV);
				while ((isalnum((unsigned char)*p) || *p == '_') && n < 15)
					o->var[o->nv][n++] = *p++;
				o->var[o->nv++][n] = 0;
			}
		} else if (*p)
			fail("polynomials: unexpected \"%s\" (options: order lex|grlex|"
			    "grevlex, mod P, over Q, vars x, y, ...)", p);
	}
}

/* the polynomial part and the options of s; the variables (if not given)
 * are the identifiers that occur, sorted */
static char *
popts(char *s, struct popts *o, const char *dflt_order)
{
	static const char *const kw[] = {"order", "mod", "over", "vars", NULL};
	char *cut = NULL, *p;
	int depth = 0, k;

	memset(o, 0, sizeof *o);
	o->order = !strcmp(dflt_order, "lex") ? MPL_LEX : MPL_GREVLEX;
	for (p = s; *p; p++) {
		if (*p == '[' || *p == '(')
			depth++;
		else if (*p == ']' || *p == ')')
			depth--;
		for (k = 0; depth == 0 && kw[k]; k++)
			if (is_kw_at(s, p, kw[k]) && !cut)
				cut = p;
	}
	if (cut) {
		read_popts(cut, o);
		*cut = 0;
	}
	if (o->nv == 0) {
		/* collect identifiers */
		for (p = s; *p;) {
			if (isalpha((unsigned char)*p) || *p == '_') {
				char w[16];
				int n = 0, i;
				while ((isalnum((unsigned char)*p) || *p == '_') && n < 15)
					w[n++] = *p++;
				while (isalnum((unsigned char)*p) || *p == '_')
					p++;
				w[n] = 0;
				if (!strcmp(w, "in"))
					continue;
				for (i = 0; i < o->nv && strcmp(o->var[i], w); i++)
					;
				if (i == o->nv) {
					if (o->nv == MPL_MAXV)
						fail("polynomials: at most %d variables",
						    MPL_MAXV);
					strcpy(o->var[o->nv++], w);
				}
			} else
				p++;
		}
		/* sorted: x > y > z, a > b ... (alphabetical) */
		{
			int i, j;
			for (i = 1; i < o->nv; i++)
				for (j = i; j > 0 && strcmp(o->var[j - 1], o->var[j]) > 0;
				    j--) {
					char t[16];
					strcpy(t, o->var[j]);
					strcpy(o->var[j], o->var[j - 1]);
					strcpy(o->var[j - 1], t);
				}
		}
	}
	return trim(s);
}

static void
ring_of(struct mring *R, const struct popts *o)
{
	const char *names[MPL_MAXV];
	int i;

	for (i = 0; i < o->nv; i++)
		names[i] = o->var[i];
	mring_init(R, o->nv, names, o->order, o->p);
}

static const char *
ring_desc(const struct mring *R)
{
	static char buf[512];
	int i, n;

	n = snprintf(buf, sizeof buf, "%s, ", R->order == MPL_LEX ? "lex" :
	    R->order == MPL_GRLEX ? "grlex" : "grevlex");
	for (i = 0; i < R->nv && n < (int)sizeof buf - 40; i++)
		n += snprintf(buf + n, sizeof buf - n, "%s%s", i ? " > " : "",
		    R->name[i]);
	if (R->p)
		snprintf(buf + n, sizeof buf - n, ", over F_%llu",
		    (unsigned long long)R->p);
	else
		snprintf(buf + n, sizeof buf - n, ", over Q");
	return buf;
}

/* polynomials separated by top-level commas */
static int
need_polys(const struct mring *R, char *s, struct mpl **out, const char *what)
{
	char *part[256];
	int n = split_commas(s, part, 256), i;
	struct mpl *f = xcalloc((size_t)n + 1, sizeof *f);

	for (i = 0; i < n; i++) {
		const char *e;
		if (!*part[i] || !mpl_parse(R, part[i], &f[i], &e) ||
		    *trim((char *)e))
			fail("%s: cannot read the polynomial \"%s\"", what, part[i]);
	}
	*out = f;
	return n;
}

static void
free_polys(struct mpl *f, int n)
{
	int i;

	for (i = 0; i < n; i++)
		mpl_free(&f[i]);
	free(f);
}

static void
cmd_groebner(char *s)
{
	struct popts o;
	struct mring R;
	struct mpl *f;
	struct gbasis gb;
	int k, i, j;

	s = trim(s);
	if (!strncmp(s, "basis of ", 9) || !strncmp(s, "basis for ", 10))
		s += s[6] == 'o' ? 9 : 10;	/* "groebner basis of ..." */
	s = popts(s, &o, "grevlex");
	ring_of(&R, &o);
	k = need_polys(&R, s, &f, "groebner");
	mpl_groebner(&R, f, k, &gb);
	printf("  input:");
	for (j = 0; j < k; j++)
		printf("%s f%d = %s", j ? "," : "", j + 1, mpl_str(&R, &f[j]));
	printf("\n");
	printf("reduced Groebner basis (%s): %d element%s\n", ring_desc(&R),
	    gb.n, gb.n == 1 ? "" : "s");
	rep_list("basis");
	for (i = 0; i < gb.n; i++)
		rep_str(NULL, mpl_str(&R, &gb.g[i]));
	rep_end();
	for (i = 0; i < gb.n; i++)
		printf("  g%d: %s\n", i + 1, mpl_str(&R, &gb.g[i]));
	for (i = 0; i < gb.n; i++) {
		int first = 1;
		printf("  g%d =", i + 1);
		for (j = 0; j < k; j++)
			if (gb.cof[(size_t)i * k + j].n) {
				printf("%s (%s)*f%d", first ? "" : " +", mpl_str(&R,
				    &gb.cof[(size_t)i * k + j]), j + 1);
				first = 0;
			}
		printf("\n");
	}
	printf("  pairs: %ld; reduced: %ld; skipped by the criteria: %ld\n",
	    gb.pairs, gb.reductions, gb.skipped);
	printf("  how: Buchberger (normal strategy, product and chain "
	    "criteria), every element carrying its cofactors\n");
	printf("  check: each g_i = sum c_ij f_j multiplied out; each f_j "
	    "reduces to 0; every S-pair of the basis reduces to 0: verified\n");
	gb_free(&gb);
	free_polys(f, k);
}

/* idealmember h in f1, f2, ... */
static void
cmd_idealmember(char *s)
{
	struct popts o;
	struct mring R;
	struct mpl *f, h, *cof, r;
	struct gbasis gb;
	char *in;
	int k, j;
	const char *e;

	s = popts(s, &o, "grevlex");
	in = find_word(s, "in");
	if (!in)
		fail("idealmember: h in f1, f2, ...");
	*in = 0;
	ring_of(&R, &o);
	memset(&h, 0, sizeof h);
	if (!mpl_parse(&R, trim(s), &h, &e) || *trim((char *)e))
		fail("idealmember: cannot read \"%s\"", trim(s));
	k = need_polys(&R, in + 2, &f, "idealmember");
	mpl_groebner(&R, f, k, &gb);
	cof = xcalloc((size_t)k + 1, sizeof *cof);
	if (gb_member(&R, &gb, f, &h, cof)) {
		printf("%s is in the ideal (%s)\n", mpl_str(&R, &h), ring_desc(&R));
		rep_bool("member", 1);
		printf("  certificate: %s =", mpl_str(&R, &h));
		rep_list("cofactors");
		for (j = 0; j < k; j++) {
			printf("%s (%s)*(%s)", j ? " +" : "", mpl_str(&R, &cof[j]),
			    mpl_str(&R, &f[j]));
			rep_str(NULL, mpl_str(&R, &cof[j]));
		}
		rep_end();
		printf("\n");
		printf("  check: the combination multiplied out equals the "
		    "polynomial\n");
	} else {
		memset(&r, 0, sizeof r);
		mpl_divide(&R, &h, gb.g, gb.n, NULL, &r);
		printf("%s is not in the ideal (%s)\n", mpl_str(&R, &h),
		    ring_desc(&R));
		rep_bool("member", 0);
		printf("  normal form: %s (nonzero)\n", mpl_str(&R, &r));
		printf("  how: the remainder on division by a reduced Groebner "
		    "basis (%d element%s, verified) is unique, and zero exactly "
		    "for members\n", gb.n, gb.n == 1 ? "" : "s");
		mpl_free(&r);
	}
	for (j = 0; j < k; j++)
		mpl_free(&cof[j]);
	free(cof);
	mpl_free(&h);
	gb_free(&gb);
	free_polys(f, k);
}

/* idealeq f1, f2 = g1, g2 (also <= for inclusion) */
static void
cmd_idealeq(char *s)
{
	struct popts o;
	struct mring R;
	struct mpl *f, *g;
	struct gbasis gf, gg;
	char *eq;
	int kf, kg, le, ge, i;

	s = popts(s, &o, "grevlex");
	for (eq = s; *eq && *eq != '='; eq++)
		;
	if (!*eq)
		fail("idealeq: f1, f2, ... = g1, g2, ...");
	*eq = 0;
	if (eq > s && eq[-1] == '<')
		eq[-1] = 0;
	ring_of(&R, &o);
	{
		static char ti[4096], tj[4096];
		snprintf(ti, sizeof ti, "%s", trim(s));
		snprintf(tj, sizeof tj, "%s", trim(eq + 1));
		kf = need_polys(&R, s, &f, "idealeq");
		kg = need_polys(&R, eq + 1, &g, "idealeq");
		s = ti;
		eq = tj - 1;
	}
	mpl_groebner(&R, f, kf, &gf);
	mpl_groebner(&R, g, kg, &gg);
	le = gb_ideal_le(&R, f, kf, &gg, g);
	ge = gb_ideal_le(&R, g, kg, &gf, f);
	printf("I = (%s), J = (%s) (%s): %s\n", trim(s), trim(eq + 1),
	    ring_desc(&R), le && ge ? "I = J" : le ? "I is strictly inside J" :
	    ge ? "J is strictly inside I" : "neither contains the other");
	rep_bool("equal", le && ge);
	rep_bool("i_in_j", le);
	rep_bool("j_in_i", ge);
	printf("  reduced Groebner basis of I:");
	for (i = 0; i < gf.n; i++)
		printf("%s %s", i ? "," : "", mpl_str(&R, &gf.g[i]));
	printf("\n  reduced Groebner basis of J:");
	for (i = 0; i < gg.n; i++)
		printf("%s %s", i ? "," : "", mpl_str(&R, &gg.g[i]));
	printf("\n  check: each generator of the smaller ideal is a combination "
	    "of the other's generators (cofactors multiplied out)\n");
	gb_free(&gf);
	gb_free(&gg);
	free_polys(f, kf);
	free_polys(g, kg);
}

/* fxmodule M [mod p]: F^n as an F[x]-module with x acting by M */
static void
cmd_fxmodule(char *s)
{
	struct popts o;
	struct mring R;
	struct qmat m;
	struct mpl inv[64], cp, t;
	const char *e, *names[1] = {"x"};
	int n, i;

	s = popts(s, &o, "grevlex");
	mring_init(&R, 1, names, MPL_LEX, o.p);
	memset(&m, 0, sizeof m);
	if (!qm_parse(&m, s, &e) || *trim((char *)e) || m.r != m.c)
		fail("fxmodule: a square matrix over Q (or with mod P)");
	n = pm_charmodule(&R, &m, inv, 64);
	printf("F^%d with x acting by the matrix (F = %s): F[x]-module ", m.r,
	    o.p ? "F_p" : "Q");
	for (i = 0; i < n; i++)
		printf("%sF[x]/(%s)", i ? " + " : "", mpl_str(&R, &inv[i]));
	if (n == 0)
		printf("0");
	printf("\n");
	rep_list("invariant_factors");
	for (i = 0; i < n; i++)
		rep_str(NULL, mpl_str(&R, &inv[i]));
	rep_end();
	memset(&cp, 0, sizeof cp);
	memset(&t, 0, sizeof t);
	mpl_const(&R, &cp, zq_int(Z_ONE));
	for (i = 0; i < n; i++)
		mpl_mul(&R, &cp, &cp, &inv[i]);
	printf("  minimal polynomial: %s\n", n ? mpl_str(&R, &inv[n - 1]) : "1");
	printf("  characteristic polynomial: %s\n", mpl_str(&R, &cp));
	printf("  rational canonical form: companion blocks of sizes");
	for (i = 0; i < n; i++)
		printf(" %d", mpl_deg(&R, &inv[i]));
	printf("\n");
	printf("  how: the Smith form U (xI - M) V = D over the PID F[x] "
	    "(U, V invertible: U U^-1 = I and V V^-1 = I checked)\n");
	for (i = 0; i < n; i++)
		mpl_free(&inv[i]);
	mpl_free(&cp);
	mpl_free(&t);
	qm_free(&m);
}

/* pmsnf [[polys]]: the Smith form over F[x] */
static void
cmd_pmsnf(char *s)
{
	struct popts o;
	struct mring R;
	struct pmat A, D, U, V;
	char *rows[64], *body;
	int nr, nc = -1, i, j, rank;

	s = popts(s, &o, "grevlex");
	if (o.nv > 1)
		fail("pmsnf: polynomials in one variable (found %d)", o.nv);
	if (o.nv == 0) {
		o.nv = 1;
		strcpy(o.var[0], "x");
	}
	ring_of(&R, &o);
	body = trim(s);
	if (body[0] != '[' || body[strlen(body) - 1] != ']')
		fail("pmsnf: a matrix [[x, 1], [0, x^2]]");
	body[strlen(body) - 1] = 0;
	nr = split_commas(body + 1, rows, 64);
	for (i = 0; i < nr; i++) {
		char *cells[64], *r = trim(rows[i]);
		int c;
		if (r[0] != '[' || r[strlen(r) - 1] != ']')
			fail("pmsnf: rows like [x, 1]");
		r[strlen(r) - 1] = 0;
		c = split_commas(r + 1, cells, 64);
		if (nc < 0) {
			nc = c;
			pm_init(&A, nr, nc);
		} else if (c != nc)
			fail("pmsnf: the rows have different lengths");
		for (j = 0; j < c; j++) {
			const char *e;
			if (!mpl_parse(&R, cells[j], PM(&A, i, j), &e) ||
			    *trim((char *)e))
				fail("pmsnf: cannot read \"%s\"", cells[j]);
		}
	}
	memset(&D, 0, sizeof D);
	memset(&U, 0, sizeof U);
	memset(&V, 0, sizeof V);
	rank = pm_snf(&R, &A, &D, &U, &V);
	printf("Smith form over %s[%s]: diag(", R.p ? "F_p" : "Q", R.name[0]);
	rep_list("diagonal");
	for (i = 0; i < (nr < nc ? nr : nc); i++) {
		printf("%s%s", i ? ", " : "", mpl_str(&R, PM(&D, i, i)));
		rep_str(NULL, mpl_str(&R, PM(&D, i, i)));
	}
	rep_end();
	printf("), rank %d\n", rank);
	printf("  U:");
	for (i = 0; i < U.r; i++) {
		printf(" [");
		for (j = 0; j < U.c; j++)
			printf("%s%s", j ? ", " : "", mpl_str(&R, PM(&U, i, j)));
		printf("]");
	}
	printf("\n  V:");
	for (i = 0; i < V.r; i++) {
		printf(" [");
		for (j = 0; j < V.c; j++)
			printf("%s%s", j ? ", " : "", mpl_str(&R, PM(&V, i, j)));
		printf("]");
	}
	printf("\n  check: U A V = D; U and V invertible over F[x] (their "
	    "inverses tracked and multiplied back); d_i | d_(i+1)\n");
	pm_free(&A);
	pm_free(&D);
	pm_free(&U);
	pm_free(&V);
}

static const struct {
	const char *w;
	int c;
} acmds2[] = {
	{"permgroup", 1}, {"permelem", 2}, {"permsub", 3}, {"permhom", 4},
	{"cayley", 5}, {"groebner", 6}, {"idealmember", 7}, {"idealeq", 8},
	{"fxmodule", 9}, {"pmsnf", 10}, {NULL, 0}
};

int
absalg_is_cmd2(const char *w)
{
	int k;

	for (k = 0; acmds2[k].w; k++)
		if (!strcmp(acmds2[k].w, w))
			return acmds2[k].c;
	return 0;
}

void
absalg_cmd2(const char *w, char *s)
{
	switch (absalg_is_cmd2(w)) {
	case 1:
		cmd_permgroup(s);
		break;
	case 2:
		cmd_permelem(s);
		break;
	case 3:
		cmd_permsub(s);
		break;
	case 4:
		cmd_permhom(s);
		break;
	case 5:
		cmd_cayley(s);
		break;
	case 6:
		cmd_groebner(s);
		break;
	case 7:
		cmd_idealmember(s);
		break;
	case 8:
		cmd_idealeq(s);
		break;
	case 9:
		cmd_fxmodule(s);
		break;
	case 10:
		cmd_pmsnf(s);
		break;
	}
}

/* ==== English (nlp.c): a few phrasings, read as the commands ==== */

static int
ci_prefix(const char *s, const char *w)
{
	size_t n = strlen(w);

	return !strncasecmp(s, w, n) && !isalnum((unsigned char)s[n]);
}

/* a group spec: permutation (S4, A5, cycles) or abelian (Z/.., 0, coker) */
static int
looks_perm(const char *g)
{
	return (strchr("SACD", g[0]) && isdigit((unsigned char)g[1])) ||
	    g[0] == '(' || !strncmp(g, "Sym(", 4) || !strncmp(g, "Alt(", 4);
}

static int
looks_ab(const char *g)
{
	return g[0] == 'Z' || (g[0] == '0' && !g[1]) || !strncmp(g, "coker", 5);
}

/* "find X" -> X; between the two words a and b */
static int
between(char *s, const char *a, const char *b, char **x, char **y)
{
	char *p = NULL, *q;
	size_t la = strlen(a);

	if (strncasecmp(s, a, la))
		return 0;
	*x = s + la;
	if (!b) {
		*x = trim(*x);
		return **x != 0;
	}
	for (q = *x; *q; q++)
		if (ci_prefix(q, b) && q > *x && q[-1] == ' ') {
			p = q;
			break;
		}
	if (!p)
		return 0;
	*p = 0;
	*y = trim(p + strlen(b));
	*x = trim(*x);
	return **x && **y;
}

const char *
absalg_nlp_raw(const char *sent)
{
	static char buf[8192], out[8300];
	static const char *const lead[] = {"what is ", "whats ", "what's ",
		"compute ", "find ", "give me ", "show me ", "show ", "the ",
		"please ", NULL};
	char *s = buf, *x, *y;
	size_t n;
	int k, again = 1;

	snprintf(buf, sizeof buf, "%s", sent);
	s = trim(s);
	n = strlen(s);
	while (n && (s[n - 1] == '?' || s[n - 1] == '.' || s[n - 1] == '!'))
		s[--n] = 0;
	s = trim(s);
	while (again) {
		again = 0;
		for (k = 0; lead[k]; k++)
			if (!strncasecmp(s, lead[k], strlen(lead[k]))) {
				s = trim(s + strlen(lead[k]));
				again = 1;
			}
	}
	if (between(s, "smith normal form of", NULL, &x, &y) ||
	    between(s, "smith form of", NULL, &x, &y))
		return snprintf(out, sizeof out, "snf %s", x), out;
	if (between(s, "hermite normal form of", NULL, &x, &y) ||
	    between(s, "hermite form of", NULL, &x, &y))
		return snprintf(out, sizeof out, "hnf %s", x), out;
	if ((between(s, "reduced groebner basis of", NULL, &x, &y) ||
	    between(s, "groebner basis of", NULL, &x, &y) ||
	    between(s, "groebner basis for", NULL, &x, &y) ||
	    between(s, "gröbner basis of", NULL, &x, &y)))
		return snprintf(out, sizeof out, "groebner %s", x), out;
	if (between(s, "is", "isomorphic to", &x, &y) && (looks_ab(x) ||
	    x[0] == '['))
		return snprintf(out, sizeof out, "abiso %s, %s", x, y), out;
	if (between(s, "tensor product of", "and", &x, &y) && looks_ab(x))
		return snprintf(out, sizeof out, "abtensor %s, %s", x, y), out;
	if (between(s, "tor of", "and", &x, &y) && looks_ab(x))
		return snprintf(out, sizeof out, "abtor %s, %s", x, y), out;
	if (between(s, "ext of", "and", &x, &y) && looks_ab(x))
		return snprintf(out, sizeof out, "abext %s, %s", x, y), out;
	if (between(s, "is", "in the ideal generated by", &x, &y) ||
	    between(s, "is", "in the ideal", &x, &y))
		return snprintf(out, sizeof out, "idealmember %s in %s", x, y), out;
	if (between(s, "invariant factors of", NULL, &x, &y) ||
	    between(s, "structure of", NULL, &x, &y) ||
	    between(s, "order of", NULL, &x, &y) ||
	    between(s, "is", NULL, &x, &y)) {
		/* "is A5 solvable": the group, then a property word */
		char *sp = x + strlen(x);
		static const char *const props[] = {"solvable", "soluble",
			"nilpotent", "abelian", "cyclic", "finite", "free", NULL};
		if (!strncasecmp(s, "is", 2)) {
			int ok = 0;
			size_t l = strlen(x);
			if (l > 6 && !strcasecmp(x + l - 6, " group"))
				x[l - 6] = 0;
			sp = x + strlen(x);
			while (sp > x && sp[-1] != ' ')
				sp--;
			for (k = 0; props[k]; k++)
				if (!strcasecmp(sp, props[k]))
					ok = 1;
			if (!ok || sp == x)
				return NULL;
			sp[-1] = 0;
			x = trim(x);
			l = strlen(x);
			if (l > 2 && (x[l - 1] == 'a' || x[l - 1] == 'A') &&
			    x[l - 2] == ' ')
				x[l - 2] = 0;
			x = trim(x);
		}
		if (looks_perm(x))
			return snprintf(out, sizeof out, "permgroup %s", x), out;
		if (looks_ab(x))
			return snprintf(out, sizeof out, "abgroup %s", x), out;
	}
	return NULL;
}

/* is this line (one of) our English phrasings: 1, ucas: 0, else -1 */
int
absalg_nlp_english(const char *s, unsigned long n)
{
	char line[2048], w[48];
	size_t k = 0;
	int c;

	if (n >= sizeof line)
		return -1;
	memcpy(line, s, n);
	line[n] = 0;
	while (k < n && k + 1 < sizeof w && (isalnum((unsigned char)line[k]) ||
	    line[k] == '_'))
		k++;
	memcpy(w, line, k);
	w[k] = 0;
	for (c = 0; acmds[c].w; c++)
		if (!strcmp(acmds[c].w, w))
			return 0;
	if (absalg_is_cmd2(w))
		return 0;
	return absalg_nlp_raw(line) ? 1 : -1;
}
