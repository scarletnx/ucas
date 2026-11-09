/*
 * ecff_cmd.c - the commands of elliptic curves over finite fields
 * (parse.c calls ecff_command).  A curve is
 *
 *   a name             secp256k1, P-256, Curve25519, BN254, ...
 *   an equation        y^2 = x^3 + 2x + 3 mod 97   (also "over F_97",
 *                      "over GF(97)"; general Weierstrass y^2 + a1 xy +
 *                      a3 y = x^3 + a2 x^2 + a4 x + a6 is moved to short
 *                      form, with the map printed)
 *   coefficients       [2, 3] mod 97, [a1, a2, a3, a4, a6] mod p
 *
 * and a point is (x, y), G (a named curve's generator), O, or random.
 * Every command prints a headline, "  label: value" lines (the JSON
 * result's fields) and "  check:" lines for what was verified.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>

#include "ucas.h"
#include "ring.h"
#include "ecff.h"

/* ---- the current curve ---- */

static struct {
	struct ecload L;        /* named, or custom (L.N == NULL) */
	int named;
	int mapped;             /* the input was not short Weierstrass */
	int maptype;            /* 1: x + a2/3; 2: (36x + 3b2, 108(2y+a1x+a3)) */
	u64 a1[FW], a2[FW], a3[FW], b2[FW];
	char desc[600];
} cc;

#define CF (&cc.L.F)
#define CE (&cc.L.E)

static const char *sp;     /* the argument scanner */

static void
skipws(void)
{
	while (*sp == ' ' || *sp == '\t' || *sp == ',' || *sp == ':')
		sp++;
}

static int
peekword(const char *w)
{
	size_t n = strlen(w);

	skipws();
	return !strncasecmp(sp, w, n) && !isalnum((unsigned char)sp[n]) &&
	    sp[n] != '_';
}

static int
takeword(const char *w)
{
	if (!peekword(w))
		return 0;
	sp += strlen(w);
	return 1;
}

static int
getword(char *buf, size_t n)
{
	size_t k = 0;

	skipws();
	while ((isalnum((unsigned char)*sp) || *sp == '_' || *sp == '-') &&
	    k + 1 < n)
		buf[k++] = *sp++;
	buf[k] = 0;
	return k > 0;
}

static int
getzz(zz *z)
{
	const char *e;

	skipws();
	if (!zz_parse(sp, z, &e))
		return 0;
	sp = e;
	return 1;
}

static int
getnn(nn *v)
{
	zz z;

	if (!getzz(&z) || z.neg)
		return 0;
	*v = z.m;
	return 1;
}

static void
need_nn(nn *v, const char *what)
{
	if (!getnn(v))
		fail("expected %s at \"%.30s\"", what, sp);
}

/* the field after mod / over: 97, F_97, GF(97), F_{2^255-19} */
static void
parse_modulus(nn *p, int *k)
{
	zz z;

	*k = 1;
	skipws();
	if (!strncasecmp(sp, "GF", 2) || ((*sp == 'F' || *sp == 'f') &&
	    (sp[1] == '_' || sp[1] == '(' || sp[1] == '{'))) {
		int paren;
		sp += (*sp == 'G' || *sp == 'g') ? 2 : 1;
		if (*sp == '_')
			sp++;
		paren = *sp == '(' || *sp == '{';
		if (paren)
			sp++;
		if (!zz_parse(sp, &z, &sp) || z.neg)
			fail("expected a prime after GF");
		*p = z.m;
		skipws();
		if (paren && (*sp == ')' || *sp == '}'))
			sp++;
		return;
	}
	if (!getzz(&z) || z.neg)
		fail("expected a modulus at \"%.30s\"", sp);
	*p = z.m;
}

/* terms c x^i y^j; coefficients in a[i][j] (i <= 3, j <= 2) */
static int
parse_side(zz a[4][3], int sign)
{
	int any = 0;

	for (;;) {
		zz c;
		int s = 1, i = 0, j = 0, hc = 0, hv = 0;
		skipws();
		if (*sp == '+' || *sp == '-') {
			s = *sp == '-' ? -1 : 1;
			sp++;
			skipws();
		} else if (any)
			return 1;
		zz_set_i64(&c, 1);
		if (isdigit((unsigned char)*sp) || *sp == '(') {
			/* a number (2x: the digits only; 0x1f: hex) or (expr) */
			zz v;
			char buf[400];
			size_t k = 0;
			if (*sp == '(') {
				const char *e;
				if (!zz_parse(sp, &v, &e))
					return 0;
				sp = e;
			} else {
				if (sp[0] == '0' && (sp[1] == 'x' || sp[1] == 'X') &&
				    isxdigit((unsigned char)sp[2]))
					for (k = 2; isxdigit((unsigned char)sp[k]); k++)
						;
				else
					for (k = 0; isdigit((unsigned char)sp[k]); k++)
						;
				if (k + 1 > sizeof buf)
					return 0;
				memcpy(buf, sp, k);
				buf[k] = 0;
				if (!zz_parse(buf, &v, NULL))
					return 0;
				sp += k;
			}
			c = v;
			hc = 1;
		}
		for (;;) {
			skipws();
			if (*sp == '*' && sp[1] != '*') {
				sp++;
				skipws();
			}
			if (*sp == 'x' || *sp == 'y') {
				int isx = *sp == 'x', e = 1;
				sp++;
				if (*sp == '^' || (*sp == '*' && sp[1] == '*')) {
					sp += *sp == '^' ? 1 : 2;
					if (!isdigit((unsigned char)*sp))
						return 0;
					e = (int)strtol(sp, (char **)&sp, 10);
				} else if (isdigit((unsigned char)*sp))
					e = (int)strtol(sp, (char **)&sp, 10);
				if (isx)
					i += e;
				else
					j += e;
				hv = 1;
				continue;
			}
			break;
		}
		if (!hc && !hv)
			return any;
		if (i > 3 || j > 2)
			fail("not a Weierstrass equation (term x^%d y^%d)", i, j);
		if (s * sign < 0)
			c.neg = !c.neg && !zz_is_zero(&c);
		zz_add(&a[i][j], &a[i][j], &c);
		any = 1;
	}
}

static void
custom_curve(const zz *A1, const zz *A2, const zz *A3, const zz *A4,
    const zz *A6, const nn *p)
{
	struct mf *F = CF;
	u64 a4[FW], a6[FW], t[FW], u[FW], i3[FW];

	memset(&cc, 0, sizeof cc);
	if (!mf_init(F, p))
		fail("the modulus must be an odd prime of at most %d bits",
		    64 * FW);
	if (!mf_prime_setup(F))
		fail("%s is not prime (curves over Z/n: ecm uses them; here "
		    "the field must be F_p)", nn_str(p));
	if (nn_cmp_u64(p, 3) <= 0)
		fail("characteristic 2 and 3: binary curves are ecb2; p >= 5 here");
	mf_from_zz(F, cc.a1, A1);
	mf_from_zz(F, cc.a2, A2);
	mf_from_zz(F, cc.a3, A3);
	mf_from_zz(F, a4, A4);
	mf_from_zz(F, a6, A6);
	mf_from_u64(F, i3, 3);
	mf_inv(F, i3, i3, NULL);
	if (mf_is_zero(F, cc.a1) && mf_is_zero(F, cc.a3)) {
		if (!mf_is_zero(F, cc.a2)) {
			/* x -> x - a2/3 */
			u64 a[FW], b[FW], h[FW];
			mf_mul(F, h, cc.a2, i3);           /* a2/3 */
			mf_mul(F, t, h, cc.a2);
			mf_sub(F, a, a4, t);               /* a4 - a2^2/3 */
			mf_mul(F, t, h, a4);
			mf_sub(F, b, a6, t);
			mf_sqr(F, u, h);
			mf_mul(F, u, u, h);
			mf_dbl(F, u, u);
			mf_add(F, b, b, u);                /* + 2 a2^3/27 */
			ecp_set(CE, F, a, b);
			cc.mapped = 1;
			cc.maptype = 1;
		} else
			ecp_set(CE, F, a4, a6);
	} else {
		u64 b2[FW], b4[FW], b6[FW], c4[FW], c6[FW], a[FW], b[FW];
		mf_sqr(F, b2, cc.a1);
		mf_mul_u64(F, t, cc.a2, 4);
		mf_add(F, b2, b2, t);
		mf_mul(F, b4, cc.a1, cc.a3);
		mf_mul_u64(F, t, a4, 2);
		mf_add(F, b4, b4, t);
		mf_sqr(F, b6, cc.a3);
		mf_mul_u64(F, t, a6, 4);
		mf_add(F, b6, b6, t);
		mf_sqr(F, c4, b2);
		mf_mul_u64(F, t, b4, 24);
		mf_sub(F, c4, c4, t);
		mf_sqr(F, c6, b2);
		mf_mul(F, c6, c6, b2);
		mf_neg(F, c6, c6);
		mf_mul(F, t, b2, b4);
		mf_mul_u64(F, t, t, 36);
		mf_add(F, c6, c6, t);
		mf_mul_u64(F, t, b6, 216);
		mf_sub(F, c6, c6, t);
		mf_mul_u64(F, a, c4, 27);
		mf_neg(F, a, a);
		mf_mul_u64(F, b, c6, 54);
		mf_neg(F, b, b);
		ecp_set(CE, F, a, b);
		mf_copy(F, cc.b2, b2);
		cc.mapped = 1;
		cc.maptype = 2;
	}
	{
		u64 d[FW];
		ecp_disc(CE, d);
		if (mf_is_zero(F, d))
			fail("singular: 4a^3 + 27b^2 = 0 mod p (not an elliptic curve)");
	}
	nn_zero(&cc.L.n);
}

static void
curve_desc(void)
{
	size_t o;

	if (cc.named) {
		snprintf(cc.desc, sizeof cc.desc, "%s", cc.L.N->name);
		return;
	}
	o = (size_t)snprintf(cc.desc, sizeof cc.desc, "y^2 = x^3");
	if (!mf_is_zero(CF, CE->a)) {
		nn v, w;
		mf_to_nn(CF, &v, CE->a);
		nn_sub(&w, &CF->P, &v);
		if (nn_bits(&w) < nn_bits(&v) - 8)
			o += (size_t)snprintf(cc.desc + o, sizeof cc.desc - o,
			    " - %sx", nn_is_one(&w) ? "" : nn_str(&w));
		else
			o += (size_t)snprintf(cc.desc + o, sizeof cc.desc - o,
			    " + %sx", nn_is_one(&v) ? "" : nn_str(&v));
	}
	if (!mf_is_zero(CF, CE->b)) {
		nn v, w;
		mf_to_nn(CF, &v, CE->b);
		nn_sub(&w, &CF->P, &v);
		if (nn_bits(&w) < nn_bits(&v) - 8)
			o += (size_t)snprintf(cc.desc + o, sizeof cc.desc - o,
			    " - %s", nn_str(&w));
		else
			o += (size_t)snprintf(cc.desc + o, sizeof cc.desc - o,
			    " + %s", nn_str(&v));
	}
	snprintf(cc.desc + o, sizeof cc.desc - o, " over F_%s",
	    CF->bits > 64 ? nn_hex(&CF->P) : nn_str(&CF->P));
}

/* a curve at sp */
static void
parse_curve(void)
{
	char w[64];
	const char *save;

	skipws();
	save = sp;
	if (getword(w, sizeof w)) {
		const struct ecnamed *N = ecnamed_find(w);
		if (N) {
			char why[200];
			memset(&cc, 0, sizeof cc);
			ecnamed_load(N, &cc.L);
			/* the pointer into cc.L must follow the struct */
			cc.L.E.K = &cc.L.F;
			cc.named = 1;
			if (!ecnamed_check(&cc.L, why, sizeof why))
				fail("ecff: the table entry for %s fails: %s",
				    N->name, why);
			curve_desc();
			return;
		}
	}
	sp = save;
	skipws();
	if (*sp == '[') {
		zz c[5];
		int n = 0;
		nn p;
		int k;
		sp++;
		while (n < 5) {
			if (!getzz(&c[n]))
				break;
			n++;
			skipws();
			if (*sp == ']')
				break;
		}
		if (*sp != ']' || (n != 2 && n != 5))
			fail("coefficients are [a, b] or [a1, a2, a3, a4, a6]");
		sp++;
		if (!takeword("mod") && !takeword("over") && !takeword("in"))
			fail("expected mod p after the coefficients");
		parse_modulus(&p, &k);
		if (n == 2) {
			zz z;
			zz_set_i64(&z, 0);
			custom_curve(&z, &z, &z, &c[0], &c[1], &p);
		} else
			custom_curve(&c[0], &c[1], &c[2], &c[3], &c[4], &p);
		curve_desc();
		return;
	}
	if (*sp == 'y') {
		zz L[4][3], R[4][3], z;
		nn p;
		int i, j, k;
		memset(L, 0, sizeof L);
		memset(R, 0, sizeof R);
		for (i = 0; i < 4; i++)
			for (j = 0; j < 3; j++) {
				zz_set_i64(&L[i][j], 0);
				zz_set_i64(&R[i][j], 0);
			}
		if (!parse_side(L, 1))
			fail("expected y^2 = x^3 + ...");
		skipws();
		if (*sp != '=')
			fail("expected = in the equation");
		sp++;
		if (!parse_side(L, -1))
			fail("expected the right side x^3 + a x + b");
		/* L = 0: y^2 + a1 xy + a3 y - x^3 - a2 x^2 - a4 x - a6 */
		zz_set_i64(&z, 1);
		if (zz_cmp(&L[0][2], &z) || (zz_set_i64(&z, -1),
		    zz_cmp(&L[3][0], &z)))
			fail("the equation must be y^2 (+ a1 xy + a3 y) = x^3 + ...");
		for (i = 0; i < 4; i++)
			for (j = 0; j < 3; j++) {
				int ok = (j == 0) || (i == 1 && j == 1) ||
				    (i == 0 && j == 1) || (i == 0 && j == 2);
				if (!ok && !zz_is_zero(&L[i][j]))
					fail("not a Weierstrass equation");
			}
		if (!takeword("mod") && !takeword("over") && !takeword("in"))
			fail("expected mod p (or over F_p) after the equation");
		parse_modulus(&p, &k);
		for (i = 0; i < 4; i++)
			L[i][0].neg = !L[i][0].neg && !zz_is_zero(&L[i][0]);
		custom_curve(&L[1][1], &L[2][0], &L[0][1], &L[1][0], &L[0][0], &p);
		curve_desc();
		(void)R;
		return;
	}
	fail("expected a curve: a name (secp256k1, P-256, ...), "
	    "y^2 = x^3 + a x + b mod p, or [a, b] mod p");
}

/* a point: (x, y), G, O, random; in the input's coordinates (mapped) */
static void
parse_point(struct ecp_pt *P, int check)
{
	const struct mf *F = CF;

	skipws();
	if (takeword("G")) {
		if (!cc.named)
			fail("G: only named curves have a generator");
		*P = cc.L.G;
		return;
	}
	if (takeword("O") || takeword("inf") || takeword("infinity")) {
		ecp_set_inf(CE, P);
		return;
	}
	if (takeword("random")) {
		ecp_random_point(CE, P);
		return;
	}
	if (*sp == '(') {
		zz x, y;
		u64 X[FW], Y[FW];
		sp++;
		if (!getzz(&x))
			fail("expected x in (x, y)");
		skipws();
		if (!getzz(&y))
			fail("expected y in (x, y)");
		skipws();
		if (*sp != ')')
			fail("expected ) after (x, y");
		sp++;
		mf_from_zz(F, X, &x);
		mf_from_zz(F, Y, &y);
		if (cc.maptype == 1) {
			u64 h[FW], i3[FW];
			mf_from_u64(F, i3, 3);
			mf_inv(F, i3, i3, NULL);
			mf_mul(F, h, cc.a2, i3);
			mf_add(F, X, X, h);
		} else if (cc.maptype == 2) {
			u64 t[FW], u[FW];
			mf_mul(F, t, cc.a1, X);
			mf_add(F, t, t, cc.a3);
			mf_add(F, t, t, Y);
			mf_add(F, t, t, Y);
			mf_mul_u64(F, Y, t, 108);
			mf_mul_u64(F, u, X, 36);
			mf_mul_u64(F, t, cc.b2, 3);
			mf_add(F, X, u, t);
		}
		mf_copy(F, P->x, X);
		mf_copy(F, P->y, Y);
		P->inf = 0;
		if (check && !ecp_on_curve(CE, P))
			fail("(%s, %s) is not on the curve", zz_str(&x),
			    zz_str(&y));
		return;
	}
	fail("expected a point: (x, y), G, O or random");
}

static const char *
fe(const u64 *a)
{
	return CF->bits > 64 ? mf_hex(CF, a) : mf_str(CF, a);
}

static const char *
pt(const struct ecp_pt *P)
{
	return ecp_pt_str(CE, P);
}

static void
print_curve_head(void)
{
	printf("  curve: %s\n", cc.desc);
	if (cc.named)
		printf("  standard: %s\n", cc.L.N->desc);
	if (cc.mapped)
		printf("  short form: y^2 = x^3 + %s x + %s (%s)\n", fe(CE->a),
		    fe(CE->b), cc.maptype == 1 ? "x -> x + a2/3" :
		    "(x, y) -> (36x + 3b2, 108(2y + a1 x + a3))");
	if (cc.named && cc.L.N->model != ECN_W)
		printf("  weierstrass: y^2 = x^3 + a x + b with a = %s, b = %s "
		    "(x = u + A/3)\n", fe(CE->a), fe(CE->b));
}

/* ---- counting ---- */

enum { CM_AUTO, CM_NAIVE, CM_BSGS, CM_CM, CM_SCHOOF, CM_SEA, CM_KNOWN };
static const char *const cmname[] = {"auto", "naive", "bsgs", "cm",
	"schoof", "sea", "known"};

int ec_count_schoof(const struct ecp_curve *E, nn *N, int verbose);
int ec_count_sea(const struct ecp_curve *E, nn *N, int verbose);

static int
count(int method, nn *N, struct eccert *C, char *how, size_t hn)
{
	const struct mf *F = CF;
	int ok = 0;
	u64 j[FW];

	if (method == CM_AUTO) {
		if (cc.named) {
			*N = cc.L.order;
			snprintf(how, hn, "the standard's n h, certified below");
			return ec_certify(CE, N, C);
		}
		if (F->bits <= 20)
			method = CM_NAIVE;
		else if (CE->a_zero || mf_is_zero(F, CE->b))
			method = CM_CM;
		else if (F->bits <= 56)
			method = CM_BSGS;
		else
			method = CM_SEA;
	}
	ecp_jinv(CE, j);
	switch (method) {
	case CM_NAIVE:
		if (!ec_count_naive(CE, N))
			fail("naive counting is for p < 2^26");
		snprintf(how, hn, "naive: 1 + sum over x of (1 + (x^3 + ax + b "
		    "| p)), with a table of squares");
		ok = 1;
		break;
	case CM_BSGS:
		ok = ec_count_bsgs(CE, N, NULL);
		snprintf(how, hn, "baby-step giant-step on random points of E and "
		    "of its quadratic twist (Mestre)");
		break;
	case CM_CM:
		if (!CE->a_zero && !mf_is_zero(F, CE->b))
			fail("cm counting here is for j = 0 (a = 0) and j = 1728 "
			    "(b = 0); other CM curves: eccm");
		ok = ec_count_cm01(CE, N, C);
		snprintf(how, hn, "complex multiplication by %s: Cornacchia's "
		    "4p = t^2 + %s v^2, the twist picked by multiplying points",
		    CE->a_zero ? "Z[(1+sqrt(-3))/2] (j = 0)" : "Z[i] (j = 1728)",
		    CE->a_zero ? "3" : "4");
		if (ok)
			return 1;
		break;
	case CM_SCHOOF:
		ok = ec_count_schoof(CE, N, 0);
		snprintf(how, hn, "Schoof: t mod l from the Frobenius on E[l] "
		    "(division polynomials), CRT");
		break;
	case CM_SEA:
		ok = ec_count_sea(CE, N, 0);
		snprintf(how, hn, "Schoof-Elkies-Atkin: Elkies primes through "
		    "modular polynomials and kernel polynomials, the rest by "
		    "Schoof, CRT and a final baby-step giant-step match");
		break;
	}
	if (!ok)
		return 0;
	return ec_certify(CE, N, C);
}

static void
print_cert(const struct eccert *C, const nn *N)
{
	nn lo, hi;
	int i;

	ec_hasse(&CF->P, &lo, &hi, NULL);
	for (i = 0; i < C->npts && i < 2; i++)
		printf("  check: %s has order %s (N P = O, and (N/q) P != O "
		    "stepwise for each prime q of N)\n", pt(&C->P[i]),
		    nn_str(&C->ord[i]));
	printf("  check: N = %s is the only multiple of %s%s%s in the Hasse "
	    "interval [%s, %s]%s\n", nn_str(N), nn_str(&C->L),
	    nn_is_one(&C->L2) ? "" : " (with 2p + 2 - N a multiple of ",
	    nn_is_one(&C->L2) ? "" : nn_str(&C->L2),
	    nn_str(&lo), nn_str(&hi), nn_is_one(&C->L2) ? "" : ", the twist)");
	if (!C->complete)
		printf("  note: a cofactor was only BPSW-tested (probable "
		    "prime)\n");
}

static int
parse_method(void)
{
	char w[32];
	int i;
	const char *save;

	skipws();
	save = sp;
	takeword("using") || takeword("by") || takeword("method");
	if (getword(w, sizeof w))
		for (i = 0; i <= CM_SEA; i++)
			if (!strcasecmp(w, cmname[i]))
				return i;
	sp = save;
	return CM_AUTO;
}

static void
cmd_ecorder(void)
{
	nn N, p1, lo, hi;
	struct eccert C;
	char how[300];
	int m;
	u64 t0;
	zz t, P1, NN;

	parse_curve();
	m = parse_method();
	t0 = ecff_ms();
	if (!count(m, &N, &C, how, sizeof how))
		fail("could not count (or certify) the points of %s", cc.desc);
	nn_add_u64(&p1, &CF->P, 1);
	zz_from_nn(&P1, &p1, 0);
	zz_from_nn(&NN, &N, 0);
	zz_sub(&t, &P1, &NN);
	printf("#E = %s\n", nn_str(&N));
	print_curve_head();
	printf("  trace: t = p + 1 - #E = %s\n", zz_str(&t));
	printf("  how: %s\n", how);
	ec_hasse(&CF->P, &lo, &hi, NULL);
	printf("  check: Hasse: |t| <= 2 sqrt(p)\n");
	print_cert(&C, &N);
	if (m != CM_AUTO || !cc.named)
		printf("  time: %llu ms\n", (unsigned long long)(ecff_ms() - t0));
	(void)P1;
}

static void
cmd_ec(void)
{
	nn N, p1, n1, n2, pm1;
	struct eccert C;
	struct nnfac f;
	char how[300], fb[2000];
	u64 j[FW], d[FW];
	zz t, P1, NN;
	int k;

	parse_curve();
	if (!count(CM_AUTO, &N, &C, how, sizeof how))
		fail("could not count the points of %s", cc.desc);
	printf("%s\n", cc.desc);
	print_curve_head();
	ecp_disc(CE, d);
	mf_mul_u64(CF, d, d, 16);
	mf_neg(CF, d, d);
	printf("  discriminant: %s\n", fe(d));
	ecp_jinv(CE, j);
	printf("  j: %s\n", fe(j));
	nn_add_u64(&p1, &CF->P, 1);
	zz_from_nn(&P1, &p1, 0);
	zz_from_nn(&NN, &N, 0);
	zz_sub(&t, &P1, &NN);
	printf("  order: #E = %s\n", nn_str(&N));
	nn_factor(&N, &f, 1);
	nnfac_print(fb, sizeof fb, &f);
	printf("  factored: %s\n", fb);
	printf("  trace: %s\n", zz_str(&t));
	printf("  how: %s\n", how);
	print_cert(&C, &N);
	printf("  supersingular: %s\n", nn_is_zero(&t.m) ? "yes (t = 0)" :
	    "no (t != 0 mod p)");
	ec_structure(CE, &N, &f, &n1, &n2);
	if (nn_is_one(&n2))
		printf("  group: Z/%s (cyclic)\n", nn_str(&n1));
	else
		printf("  group: Z/%s x Z/%s\n", nn_str(&n1), nn_str(&n2));
	{
		nn s2, NT;
		nn_shl(&s2, &p1, 1);
		nn_sub(&NT, &s2, &N);
		printf("  twist order: %s\n", nn_str(&NT));
	}
	/* the largest prime factor's embedding degree */
	if (f.k > 0 && f.prime[f.k - 1]) {
		k = ec_embedding_degree(&CF->P, &f.p[f.k - 1], 100);
		if (k > 0)
			printf("  embedding degree: %d (of r = %s: r | p^%d - 1)\n",
			    k, nn_str(&f.p[f.k - 1]), k);
		else if (k == 0)
			printf("  embedding degree: none (r = p: anomalous)\n");
		else
			printf("  embedding degree: > 100 (of r = %s)\n",
			    nn_str(&f.p[f.k - 1]));
	}
	if (nn_eq(&N, &CF->P))
		printf("  weak: anomalous (#E = p): Smart's attack solves "
		    "discrete logs in polynomial time\n");
	(void)pm1;
}

static void
cmd_ecpoints(void)
{
	const struct mf *F = CF;
	u64 p, x, cnt = 1;
	int shown = 0;

	parse_curve();
	if (!nn_fits_u64(&F->P) || F->P.w[0] > 100000)
		fail("ecpoints lists the points for p <= 100000");
	p = F->P.w[0];
	printf("E(F_%llu): %s\n", (unsigned long long)p, cc.desc);
	printf("  points: O");
	for (x = 0; x < p; x++) {
		u64 X[FW], r[FW], y[FW];
		mf_from_u64(F, X, x);
		ecp_rhs(CE, r, X);
		if (mf_is_zero(F, r)) {
			cnt++;
			if (shown++ < 60)
				printf(", (%llu, 0)", (unsigned long long)x);
		} else if (mf_sqrt(F, y, r)) {
			nn a, b;
			cnt += 2;
			mf_to_nn(F, &a, y);
			nn_sub(&b, &F->P, &a);
			if (shown < 60)
				printf(", (%llu, %s), (%llu, %s)",
				    (unsigned long long)x, nn_str(&a),
				    (unsigned long long)x, nn_str(&b));
			shown += 2;
		}
	}
	if (shown > 60)
		printf(", ... (%d more)", shown - 60);
	printf("\n  count: %llu\n", (unsigned long long)cnt);
	printf("  check: every listed (x, y) satisfies y^2 = x^3 + a x + b "
	    "(square roots by Tonelli-Shanks, squared back)\n");
}

static void
cmd_ecadd(void)
{
	struct ecp_pt P, Q, R, S;

	parse_curve();
	parse_point(&P, 1);
	skipws();
	if (*sp == '+')
		sp++;
	parse_point(&Q, 1);
	ecp_add(CE, &R, &P, &Q);
	printf("P + Q = %s\n", pt(&R));
	print_curve_head();
	printf("  P: %s\n  Q: %s\n", pt(&P), pt(&Q));
	printf("  check: on the curve: %s\n", ecp_on_curve(CE, &R) ? "yes" :
	    "NO");
	ecp_sub(CE, &S, &R, &Q);
	printf("  check: (P + Q) - Q = P: %s\n", ecp_eq(CE, &S, &P) ? "yes" :
	    "NO");
}

static void
cmd_ecmul(void)
{
	struct ecp_pt P, R, R2, R3, R4;
	nn k;
	zz kz;
	char w[32];
	int which = 0;     /* 0 all, 1 affine, 2 jacobian/wnaf, 3 ladder */
	u64 t0, t1, t2, t3;

	parse_curve();
	if (!getzz(&kz))
		fail("expected a scalar k");
	k = kz.m;
	skipws();
	if (*sp == '*')
		sp++;
	parse_point(&P, 1);
	if (kz.neg)
		ecp_neg(CE, &P, &P);
	skipws();
	if (takeword("using") && getword(w, sizeof w)) {
		if (!strcasecmp(w, "affine"))
			which = 1;
		else if (!strcasecmp(w, "wnaf") || !strcasecmp(w, "jacobian"))
			which = 2;
		else if (!strcasecmp(w, "ladder"))
			which = 3;
	}
	t0 = ecff_ms();
	ecp_mul_wnaf(CE, &R, &P, &k, 4);
	t1 = ecff_ms();
	printf("%s P = %s\n", zz_str(&kz), pt(&R));
	print_curve_head();
	printf("  P: %s\n", pt(&P));
	printf("  how: Jacobian coordinates, width-4 NAF (%llu ms)\n",
	    (unsigned long long)(t1 - t0));
	if (which == 0 || which == 1) {
		ecp_mul_affine(CE, &R2, &P, &k);
		t2 = ecff_ms();
		printf("  check: affine double-and-add agrees: %s (%llu ms)\n",
		    ecp_eq(CE, &R, &R2) ? "yes" : "NO",
		    (unsigned long long)(t2 - t1));
	}
	if (which == 0 || which == 3) {
		t2 = ecff_ms();
		ecp_mul_ladder(CE, &R3, &P, &k);
		t3 = ecff_ms();
		printf("  check: Montgomery ladder agrees: %s (%llu ms)\n",
		    ecp_eq(CE, &R, &R3) ? "yes" : "NO",
		    (unsigned long long)(t3 - t2));
	}
	ecp_mul_wnaf(CE, &R4, &P, &k, 2);
	printf("  check: plain NAF agrees: %s\n", ecp_eq(CE, &R, &R4) ? "yes" :
	    "NO");
	printf("  check: on the curve: %s\n", ecp_on_curve(CE, &R) ? "yes" :
	    "NO");
}

static void
cmd_ecm(void)
{
	nn n, f, q;
	struct ecm_res r;
	u64 B1 = 0, B2 = 0;
	struct nnfac F;
	char fb[3000];

	need_nn(&n, "the number to factor");
	skipws();
	if (takeword("B1")) {
		nn b;
		skipws();
		if (*sp == '=')
			sp++;
		need_nn(&b, "B1");
		B1 = b.w[0];
	}
	skipws();
	if (takeword("B2")) {
		nn b;
		skipws();
		if (*sp == '=')
			sp++;
		need_nn(&b, "B2");
		B2 = b.w[0];
	}
	if (nn_cmp_u64(&n, 4) < 0)
		fail("ecm: n must be at least 4");
	if (nn_isprime(&n)) {
		printf("%s is prime (BPSW)\n", nn_str(&n));
		printf("  note: ecpp gives a primality certificate\n");
		return;
	}
	if (B1) {
		if (!B2)
			B2 = 100 * B1;
		ecm_run(&n, B1, B2, 1000, 6, &r);
		if (!r.found)
			fail("no factor with B1 = %llu, B2 = %llu in 1000 curves",
			    (unsigned long long)B1, (unsigned long long)B2);
		f = r.f;
		nn_divmod(&q, NULL, &n, &f);
		printf("%s = %s * %s\n", nn_str(&n), nn_str(&f), nn_str(&q));
		printf("  factor: %s\n", nn_str(&f));
		printf("  cofactor: %s\n", nn_str(&q));
		printf("  how: Lenstra ECM, Montgomery curve by Suyama's sigma = "
		    "%llu, B1 = %llu, B2 = %llu, found in stage %d after %d "
		    "curve%s (%llu ms)\n", (unsigned long long)r.sigma,
		    (unsigned long long)B1, (unsigned long long)B2, r.stage,
		    r.curves, r.curves == 1 ? "" : "s",
		    (unsigned long long)r.ms);
		{
			nn t;
			nn_mul(&t, &f, &q);
			printf("  check: %s * %s = n: %s\n", nn_str(&f), nn_str(&q),
			    nn_eq(&t, &n) ? "yes" : "NO");
		}
		return;
	}
	{
		u64 t0 = ecff_ms();
		nn_factor(&n, &F, 2);
		nnfac_print(fb, sizeof fb, &F);
		printf("%s = %s\n", nn_str(&n), fb);
		printf("  how: trial division, Pollard-Brent rho, then ECM stage "
		    "1 + 2 with growing B1 (%llu ms)\n",
		    (unsigned long long)(ecff_ms() - t0));
		printf("  check: the factors multiply back to n: %s\n",
		    nnfac_check(&F, &n) ? "yes" : "NO");
		printf("  check: %s\n", F.complete ? "every factor is a BPSW "
		    "probable prime (ecpp certifies one)" : "a composite factor "
		    "is left (marked (c))");
	}
}

/* ---- dispatch ---- */

static const struct {
	const char *w;
	void (*fn)(void);
	const char *kind;
} ecmds[] = {
	{"ec", cmd_ec, "ec"},
	{"ecorder", cmd_ecorder, "ec_order"},
	{"ecpoints", cmd_ecpoints, "ec_points"},
	{"ecadd", cmd_ecadd, "ec_add"},
	{"ecmul", cmd_ecmul, "ec_mul"},
	{"ecm", cmd_ecm, "ecm"},
	{NULL, NULL, NULL}
};

/* nlp.c: a line starting with one of these words is ucas, not English */
int
ecff_is_cmd(const char *s, size_t n)
{
	size_t k = 0;
	int i;

	while (k < n && (isalnum((unsigned char)s[k]) || s[k] == '_'))
		k++;
	for (i = 0; ecmds[i].w; i++)
		if (strlen(ecmds[i].w) == k && !strncmp(ecmds[i].w, s, k))
			return k < n ? s[k] == ' ' || s[k] == '\t' : 1;
	return 0;
}

int
ecff_command(struct P *p)
{
	static char text[16384];
	char w[48];
	uint32_t pos = cmdp_pos(p);
	int k;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; ecmds[k].w; k++)
		if (!strcmp(ecmds[k].w, w))
			break;
	if (!ecmds[k].w || func_find(sym_cstr(w))) {
		cmdp_seek(p, pos);
		return 0;
	}
	text[0] = 0;
	if (!cmdp_eos(p))
		cmdp_rest(p, text, sizeof text, NULL);
	notes_flush();
	rep_begin(ecmds[k].kind);
	rep_text_fields();
	sp = text;
	nn_seed(1);
	ecmds[k].fn();
	return 1;
}
