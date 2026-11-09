/*
 * ec_cmd.c - the elliptic-curve commands over Q (parse.c calls
 * ec_command):
 *
 *   ec CURVE [OP ARGS]     a curve: [a1,a2,a3,a4,a6], [a4,a6], or an
 *                          equation y^2 + a1 x y + a3 y = x^3 + ...
 *       (none)             the summary: invariants, minimal model, local
 *                          data, conductor, torsion, rank, generators,
 *                          periods, L-values, BSD
 *       points [H]         points of naive height <= log H
 *       torsion, rank, local, minimal, analytic, periods
 *       on P, height P, add P Q, mul k P, order P
 *       wp z               x(z), y(z) by the Weierstrass p function
 *       ap p               a_p
 *       model KIND         short, montgomery, edwards, twisted_edwards,
 *                          legendre: the model and the maps
 *       integral [B]       integral points with |x| <= B
 *   diophantine EQ [positive] [integer] [nonzero] [count N] [maxmult K]
 *                          [search B]
 *   weierstrass EQ         the maps from a cubic or quartic curve
 *   congruent N            is N the area of a rational right triangle
 *   mordell K              y^2 = x^3 + K: the group and integral points
 *   cubesum K              x^3 + y^3 = K z^3
 */
#include "ecq.h"
#include "ecan.h"
#include "ucas.h"
#include "report.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IZ(v) z_from_i64(v)

static struct ecf QC;

static const struct ecf *
QQc(void)
{
	static int init;

	if (!init) {
		ecf_q(&QC);
		init = 1;
	}
	return &QC;
}

static const char *const ops[] = {"info", "points", "torsion", "rank",
	"gens", "local", "tate", "minimal", "analytic", "periods", "on",
	"height", "add", "mul", "order", "wp", "ap", "model", "integral",
	"lseries", NULL};

static const char *
skip(const char *s)
{
	while (*s == ' ' || *s == '\t')
		s++;
	return s;
}

/* the curve at the start of s; *rest after it */
static int
parse_curve(const char *s, struct ec *E, const char **rest)
{
	s = skip(s);
	if (*s == '[') {
		struct zq q[5];
		fe a[5];
		int n = 0, i;
		s++;
		for (;;) {
			const char *e;
			s = skip(s);
			if (n == 5 || !zq_parse(s, &q[n], &e))
				fail("ec: a curve [a1, a2, a3, a4, a6] or [a4, a6]");
			n++;
			s = skip(e);
			if (*s == ',') {
				s++;
				continue;
			}
			if (*s == ']') {
				s++;
				break;
			}
			fail("ec: expected , or ] in the coefficient list");
		}
		if (n != 2 && n != 5)
			fail("ec: give 5 coefficients [a1, a2, a3, a4, a6] or 2 "
			    "[a4, a6]");
		for (i = 0; i < 5; i++)
			a[i] = fe_from_q(zq_int(0));
		if (n == 2) {
			a[3] = fe_from_q(q[0]);
			a[4] = fe_from_q(q[1]);
		} else
			for (i = 0; i < 5; i++)
				a[i] = fe_from_q(q[i]);
		ec_init(E, QQc(), a);
		*rest = s;
	} else {
		/* an equation up to the first op word */
		char buf[2048];
		const char *end = s + strlen(s), *t;
		struct dio_eq D;
		const char *why = NULL;
		int i, vx = -1, vy = -1;
		struct zq c[7];
		static const int ex[7][2] = {{0, 2}, {1, 1}, {0, 1}, {3, 0},
		    {2, 0}, {1, 0}, {0, 0}};
		for (t = s; *t; t++) {
			int k;
			if (t > s && isalnum((unsigned char)t[-1]))
				continue;
			for (k = 0; ops[k]; k++) {
				size_t l = strlen(ops[k]);
				if (!strncmp(t, ops[k], l) && !isalnum((unsigned char)t[l])) {
					end = t;
					break;
				}
			}
			if (end != s + strlen(s))
				break;
		}
		if ((size_t)(end - s) >= sizeof buf)
			fail("ec: the equation is too long");
		memcpy(buf, s, (size_t)(end - s));
		buf[end - s] = 0;
		if (!dio_parse(buf, &D, &why))
			fail("ec: %s", why ? why : "cannot read the curve");
		for (i = 0; i < D.V.nv; i++) {
			if (!strcmp(D.V.name[i], "x"))
				vx = i;
			else if (!strcmp(D.V.name[i], "y"))
				vy = i;
			else
				fail("ec: the curve's variables are x and y (not %s)",
				    D.V.name[i]);
		}
		if (vx < 0 || vy < 0)
			fail("ec: the curve needs both x and y");
		for (i = 0; i < 7; i++)
			c[i] = zq_int(0);
		for (i = 0; i < D.F.n; i++) {
			const uint8_t *e = D.F.e + i * QP_MAXV;
			int k;
			for (k = 0; k < 7; k++)
				if (e[vx] == ex[k][0] && e[vy] == ex[k][1])
					break;
			if (k == 7)
				fail("ec: not in Weierstrass form (use weierstrass "
				    "for a general cubic)");
			c[k] = D.F.c[i];
		}
		if (zq_is_zero(c[0]) || !zq_eq(c[0], zq_neg(c[3])))
			fail("ec: not in Weierstrass form y^2 + a1 x y + a3 y = x^3 + "
			    "a2 x^2 + a4 x + a6 (use weierstrass for a general cubic)");
		{
			fe a[5];
			struct zq k = zq_inv(c[0]);
			a[0] = fe_from_q(zq_mul(c[1], k));
			a[2] = fe_from_q(zq_mul(c[2], k));
			a[1] = fe_from_q(zq_neg(zq_mul(c[4], k)));
			a[3] = fe_from_q(zq_neg(zq_mul(c[5], k)));
			a[4] = fe_from_q(zq_neg(zq_mul(c[6], k)));
			ec_init(E, QQc(), a);
		}
		dio_free(&D);
		*rest = end;
	}
	if (E->singular)
		fail("ec: the curve is singular (discriminant 0)");
	return 1;
}

static void
pt_need(const struct ec *E, struct ecpt *P, const char **s)
{
	const char *e;

	if (!ec_pt_parse(E, P, *s, &e))
		fail("ec: expected a point (x, y) or O");
	if (!ec_on(E, P))
		fail("ec: %s is not on the curve", ec_pt_str(E, P));
	*s = skip(e);
}

static void
print_curve_line(const char *label, const struct ec *E)
{
	struct sbuf s = {0};

	ec_print(&s, E);
	printf("%s%s\n", label, sb_str(&s));
	free(s.p);
}

static const char *
red_name(const struct ecq_local *L)
{
	if (L->kod == KOD_IN)
		return L->ap == 1 ? "split multiplicative" :
		    "non-split multiplicative";
	return "additive";
}

static void
print_local(const struct ecq_local *L, int nl, zint N)
{
	int i;
	zint prodc = Z_ONE;

	printf("  conductor: %s\n", z_str(N));
	rep_str("conductor", z_str(N));
	rep_list("local");
	for (i = 0; i < nl; i++) {
		printf("  p = %s: %s, v(disc) = %d, f = %d, c_p = %d, %s\n",
		    z_str(L[i].p), ecq_kodaira_str(&L[i]), L[i].vdisc, L[i].f,
		    L[i].cp, red_name(&L[i]));
		prodc = z_mul(prodc, IZ(L[i].cp));
		rep_obj(NULL);
		rep_str("p", z_str(L[i].p));
		rep_str("kodaira", ecq_kodaira_str(&L[i]));
		rep_int("f", L[i].f);
		rep_int("cp", L[i].cp);
		rep_str("reduction", red_name(&L[i]));
		rep_end();
	}
	rep_end();
	printf("  Tamagawa product: %s\n", z_str(prodc));
}

static void
print_torsion(const struct ec *E, const struct ecq_tors *T)
{
	int i;

	printf("  torsion: ");
	if (T->n == 1)
		printf("trivial");
	else if (T->n1 == 1)
		printf("Z/%d", T->n2);
	else
		printf("Z/2 x Z/%d", T->n2);
	rep_strf("torsion", T->n == 1 ? "0" : T->n1 == 1 ? "Z/%d" :
	    "Z/2 x Z/%d", T->n2);
	if (T->n > 1) {
		printf(", generated by");
		for (i = 0; i < T->ngen; i++)
			printf(" %s", ec_pt_str(E, &T->gen[i]));
	}
	printf("  [Nagell-Lutz, order bounded by gcd #E(F_p) = %lld]\n",
	    (long long)T->bound);
	if (T->n > 1) {
		printf("  torsion points:");
		for (i = 0; i < T->n; i++)
			printf(" %s", ec_pt_str(E, &T->pts[i]));
		printf("\n");
	}
}

static void
print_mw(const struct ecq_mw *M)
{
	uint32_t i;

	print_torsion(&M->E, &M->T);
	if (M->have_descent)
		printf("  2-isogeny descent: |Sel(phi)| = 2^%d, |Sel(phi')| = 2^%d; "
		    "images of points found 2^%d, 2^%d\n", M->D.nsel, M->D.nsel2,
		    M->D.nimg, M->D.nimg2);
	if (M->rank_hi >= 0 && M->rank_hi == M->rank_lo)
		printf("  rank: %d (proved by 2-isogeny descent)\n", M->rank_lo);
	else if (M->rank_hi >= 0) {
		printf("  rank: %d <= r <= %d", M->rank_lo, M->rank_hi);
		if (M->have_descent && M->D.sha_hint)
			printf(" (Selmer elements without points: Sha[2] or points "
			    "beyond the search)");
		printf("\n");
	} else
		printf("  rank: >= %d (no rational 2-torsion: no descent bound "
		    "here; see the analytic rank)\n", M->rank_lo);
	rep_int("rank_lo", M->rank_lo);
	if (M->rank_hi >= 0)
		rep_int("rank_hi", M->rank_hi);
	rep_list("generators");
	for (i = 0; i < M->gens.n; i++) {
		struct ecpt P;
		ecpts_get(&M->gens, i, &P);
		printf("  generator: %s, height %.12f\n", ec_pt_str(&M->E, &P),
		    M->h[i]);
		rep_obj(NULL);
		rep_str("point", ec_pt_str(&M->E, &P));
		rep_num("height", M->h[i]);
		rep_end();
	}
	rep_end();
	if (M->gens.n) {
		printf("  regulator: %.12f\n", M->reg);
		rep_num("regulator", M->reg);
		if (M->saturated_to)
			printf("  saturated at every prime <= %d (by reduction)\n",
			    M->saturated_to);
		else
			printf("  not proved saturated at the primes <= 7\n");
	}
}

static void
print_analytic(const struct ec *Em, const struct ecq_local *L, int nl,
    zint N, const struct ecq_mw *M)
{
	struct ecan_per P;
	struct ecan_L R;
	int i;
	double prodc = 1;

	ecan_periods(Em, &P);
	printf("  periods: w1 = %.15g, w2 = %.15g %+.15gi; Omega = %.15g "
	    "(%d component%s)\n", creal(P.w1), creal(P.w2), cimag(P.w2),
	    P.omega, P.ncomp, P.ncomp > 1 ? "s" : "");
	rep_num("omega", P.omega);
	if (!ecan_lseries(Em, L, nl, N, &R)) {
		printf("  L-series: the conductor is too large here (> 10^8)\n");
		return;
	}
	printf("  root number: %+d (functional equation mismatch %.1e for it, "
	    "%.1e for the other sign)\n", R.w, R.wfit, R.wtest);
	rep_int("root_number", R.w);
	printf("  a_n:");
	for (i = 1; i <= 20; i++)
		printf(" %lld", (long long)R.an[i]);
	printf("\n");
	for (i = 0; i < 4; i++)
		if (R.L[i] != 0 || (i == 0 && R.w == 1) || (i == 1 && R.w == -1)) {
			printf("  L%s(E, 1) = %.12f  (truncation error <= %.1e, %lld "
			    "terms)\n", i == 0 ? "" : i == 1 ? "'" : i == 2 ? "''" :
			    "'''", R.L[i], R.err, (long long)R.nterms);
		}
	if (R.rank <= 3) {
		printf("  analytic rank: %d\n", R.rank);
		rep_int("analytic_rank", R.rank);
	} else
		printf("  analytic rank: >= %d (the derivatives up to %d vanish to "
		    "1e-8)\n", R.rank - 1, R.rank - 2);
	if (R.rank <= 1)
		printf("  so the rank is %d (Gross-Zagier, Kolyvagin: L%s(E, 1) != "
		    "0)\n", R.rank, R.rank ? "'" : "");
	for (i = 0; i < nl; i++)
		prodc *= L[i].cp;
	if (M && R.rank <= 3 && (int)M->gens.n == R.rank) {
		double f = 1, sha;
		int k;
		for (k = 2; k <= R.rank; k++)
			f *= k;
		sha = R.L[R.rank] / f * M->T.n * M->T.n / (P.omega * M->reg *
		    prodc);
		printf("  BSD: #Sha_an = L^(%d)(1)/%d! * #T^2 / (Omega * R * prod "
		    "c_p) = %.9f\n", R.rank, R.rank, sha);
		rep_num("sha_an", sha);
	}
}

static void
ec_summary(const struct ec *E, int64_t search, int analytic)
{
	struct ecq_mw M;
	struct ecq_local L[64];
	int nl;
	zint N;

	print_curve_line("E: ", E);
	{
		struct sbuf s = {0};
		ec_print_ainv(&s, E);
		printf("  a-invariants: %s\n", sb_str(&s));
		rep_str("ainvs", sb_str(&s));
		free(s.p);
	}
	printf("  b2, b4, b6, b8: %s, %s, %s, %s\n", fe_str(E->F, &E->b2),
	    fe_str(E->F, &E->b4), fe_str(E->F, &E->b6), fe_str(E->F, &E->b8));
	printf("  c4, c6: %s, %s\n", fe_str(E->F, &E->c4), fe_str(E->F, &E->c6));
	printf("  discriminant: %s\n", fe_str(E->F, &E->disc));
	printf("  j-invariant: %s\n", fe_str(E->F, &E->j));
	rep_str("discriminant", fe_str(E->F, &E->disc));
	rep_str("j", fe_str(E->F, &E->j));
	ecq_mw_compute(E, &M, search);
	{
		struct sbuf s = {0};
		ec_print_ainv(&s, &M.E);
		printf("  minimal model: ");
		ec_print(&s, &M.E);
		printf("%s  ", strchr(sb_str(&s), ']') + 1);
		sb_clear(&s);
		eciso_print(&s, QQc(), &M.tomin);
		printf("(%s)\n", sb_str(&s));
		sb_clear(&s);
		ec_print_ainv(&s, &M.E);
		rep_str("minimal", sb_str(&s));
		printf("  minimal discriminant: %s\n", fe_str(QQc(), &M.E.disc));
		free(s.p);
	}
	nl = ecq_local_all(&M.E, L, 64, &N);
	print_local(L, nl, N);
	print_mw(&M);
	if (analytic)
		print_analytic(&M.E, L, nl, N, &M);
	ecq_mw_free(&M);
}

static void
ec_op(const struct ec *E, const char *op, const char *s)
{
	const struct ecf *F = E->F;

	s = skip(s);
	if (!strcmp(op, "on")) {
		struct ecpt P;
		const char *e;
		fe v;
		if (!ec_pt_parse(E, &P, s, &e))
			fail("ec on: expected a point");
		if (P.inf) {
			printf("O is on every curve\n");
			return;
		}
		ec_eval(E, &v, &P.x, &P.y);
		printf("%s %s on the curve (lhs - rhs = %s)\n", ec_pt_str(E, &P),
		    fe_is_zero(F, &v) ? "is" : "is not", fe_str(F, &v));
		rep_bool("on_curve", fe_is_zero(F, &v));
		return;
	}
	if (!strcmp(op, "add")) {
		struct ecpt P, Q, R;
		pt_need(E, &P, &s);
		if (*s == ',')
			s = skip(s + 1);
		pt_need(E, &Q, &s);
		ec_add(E, &R, &P, &Q);
		if (!ec_on(E, &R))
			fail("internal: P + Q is off the curve");
		printf("%s + %s = %s\n", ec_pt_str(E, &P), ec_pt_str(E, &Q),
		    ec_pt_str(E, &R));
		printf("  check: the sum is on the curve\n");
		rep_str("value", ec_pt_str(E, &R));
		return;
	}
	if (!strcmp(op, "mul")) {
		struct ecpt P, R;
		zint k;
		const char *e;
		if (!z_parse(s, &k, &e))
			fail("ec mul: expected k then a point");
		s = skip(e);
		if (*s == ',' || *s == '*')
			s = skip(s + 1);
		pt_need(E, &P, &s);
		ec_mul(E, &R, k, &P);
		if (!ec_on(E, &R))
			fail("internal: kP is off the curve");
		printf("%s * %s = %s\n", z_str(k), ec_pt_str(E, &P),
		    ec_pt_str(E, &R));
		rep_str("value", ec_pt_str(E, &R));
		return;
	}
	if (!strcmp(op, "order")) {
		struct ecpt P;
		int64_t o;
		pt_need(E, &P, &s);
		o = ec_order_upto(E, &P, 12);
		if (o)
			printf("%s has order %lld\n", ec_pt_str(E, &P), (long long)o);
		else
			printf("%s has infinite order (no multiple up to 12 is O; "
			    "Mazur: torsion orders are <= 12)\n", ec_pt_str(E, &P));
		rep_int("order", o);
		return;
	}
	if (!strcmp(op, "height")) {
		struct ecpt P;
		double h, err;
		pt_need(E, &P, &s);
		h = ecq_height(E, &P, &err);
		printf("canonical height of %s: %.15f (archimedean series error "
		    "<= %.1e)\n", ec_pt_str(E, &P), h, err);
		printf("  naive height log max(|num x|, |den x|): %.15f\n",
		    ecq_naive_height(E, &P));
		rep_num("height", h);
		rep_num("naive_height", ecq_naive_height(E, &P));
		return;
	}
	if (!strcmp(op, "points")) {
		struct ec Em;
		struct eciso m;
		struct ecpts T;
		int64_t H = 1000;
		uint32_t i;
		if (*s)
			H = (int64_t)strtoll(s, NULL, 10);
		ecq_minimal(E, &Em, &m);
		ecpts_init(&T);
		ecq_search(&Em, H, &T, 200);
		printf("points with x = a/d^2, |a| <= %lld, d^2 <= %lld (on the "
		    "given curve): %u\n", (long long)H, (long long)H, T.n);
		rep_list("points");
		for (i = 0; i < T.n; i++) {
			struct ecpt P, Q;
			ecpts_get(&T, i, &P);
			eciso_pt_inv(E, &m, &Q, &P);
			if (!ec_on(E, &Q))
				fail("internal: search point off the curve");
			printf("  %s\n", ec_pt_str(E, &Q));
			rep_str(NULL, ec_pt_str(E, &Q));
		}
		rep_end();
		ecpts_free(&T);
		return;
	}
	if (!strcmp(op, "torsion")) {
		struct ec Em;
		struct eciso m;
		struct ecq_tors T;
		int i;
		ecq_minimal(E, &Em, &m);
		ecq_torsion(&Em, &T);
		for (i = 0; i < T.n; i++)
			eciso_pt_inv(E, &m, &T.pts[i], &T.pts[i]);
		for (i = 0; i < T.ngen; i++)
			eciso_pt_inv(E, &m, &T.gen[i], &T.gen[i]);
		print_torsion(E, &T);
		return;
	}
	if (!strcmp(op, "rank") || !strcmp(op, "gens")) {
		struct ecq_mw M;
		int64_t H = *s ? (int64_t)strtoll(s, NULL, 10) : 1000;
		ecq_mw_compute(E, &M, H);
		print_curve_line("minimal model: ", &M.E);
		print_mw(&M);
		ecq_mw_free(&M);
		return;
	}
	if (!strcmp(op, "local") || !strcmp(op, "tate") ||
	    !strcmp(op, "minimal")) {
		struct ec Em;
		struct eciso m;
		struct ecq_local L[64];
		int nl;
		zint N;
		ecq_minimal(E, &Em, &m);
		print_curve_line("minimal model: ", &Em);
		{
			struct sbuf b = {0};
			eciso_print(&b, QQc(), &m);
			printf("  from the given curve by %s\n", sb_str(&b));
			free(b.p);
		}
		printf("  minimal discriminant: %s\n", fe_str(QQc(), &Em.disc));
		nl = ecq_local_all(&Em, L, 64, &N);
		print_local(L, nl, N);
		return;
	}
	if (!strcmp(op, "analytic") || !strcmp(op, "lseries") ||
	    !strcmp(op, "periods")) {
		struct ecq_mw M;
		struct ecq_local L[64];
		int nl;
		zint N;
		if (!strcmp(op, "periods")) {
			struct ec Em;
			struct eciso m;
			ecq_minimal(E, &Em, &m);
			nl = ecq_local_all(&Em, L, 64, &N);
			{
				struct ecan_per P;
				ecan_periods(&Em, &P);
				printf("periods of the minimal model: w1 = %.15g, w2 = "
				    "%.15g %+.15gi\n", creal(P.w1), creal(P.w2),
				    cimag(P.w2));
				printf("  Omega = %.15g (%d component%s of E(R))\n",
				    P.omega, P.ncomp, P.ncomp > 1 ? "s" : "");
				rep_num("omega", P.omega);
			}
			return;
		}
		ecq_mw_compute(E, &M, 1000);
		nl = ecq_local_all(&M.E, L, 64, &N);
		print_curve_line("minimal model: ", &M.E);
		printf("  conductor: %s\n", z_str(N));
		print_analytic(&M.E, L, nl, N, &M);
		ecq_mw_free(&M);
		return;
	}
	if (!strcmp(op, "wp")) {
		struct ec Em;
		struct eciso m;
		struct ecan_per P;
		double complex x, y;
		double re, im = 0, chk;
		char *e;
		ecq_minimal(E, &Em, &m);
		ecan_periods(&Em, &P);
		re = strtod(s, &e);
		e = (char *)skip(e);
		if (*e == '+' || *e == '-') {
			im = strtod(e, &e);
			if (*e != 'i')
				fail("ec wp: z as a or a+bi");
		}
		ecan_wp(&Em, &P, re + im * I, &x, &y, &chk);
		printf("on the minimal model, z = %g%+gi: x = %.15g %+.15gi, y = "
		    "%.15g %+.15gi\n", re, im, creal(x), cimag(x), creal(y),
		    cimag(y));
		printf("  check: p'^2 = 4p^3 - g2 p - g3 to %.1e (relative)\n", chk);
		rep_num("x_re", creal(x));
		rep_num("x_im", cimag(x));
		return;
	}
	if (!strcmp(op, "ap")) {
		int64_t p = strtoll(s, NULL, 10);
		struct ec Em;
		struct eciso m;
		if (p < 2 || p > 100000000 || !nt_isprime((u64)p))
			fail("ec ap: a prime below 10^8");
		ecq_minimal(E, &Em, &m);
		if (z_mod_u64(fe_qz(&Em.disc), (u64)p) == 0) {
			struct ecq_local L;
			ecq_tate(&Em, IZ(p), &L);
			printf("a_%lld = %d (bad reduction: %s)\n", (long long)p, L.ap,
			    red_name(&L));
			rep_int("ap", L.ap);
			return;
		}
		{
			int64_t a = ecq_ap(&Em, p);
			printf("a_%lld = %lld, #E(F_%lld) = %lld (|a_p| <= 2 sqrt p = "
			    "%.3f: Hasse)\n", (long long)p, (long long)a, (long long)p,
			    (long long)(p + 1 - a), 2 * sqrt((double)p));
			rep_int("ap", a);
		}
		return;
	}
	if (!strcmp(op, "model")) {
		char kind[32];
		int k = 0, mk;
		struct ecmodel Mo;
		struct eciso iso;
		while (isalpha((unsigned char)*s) || *s == '_')
			kind[k < 31 ? k++ : k] = *s++;
		kind[k] = 0;
		mk = ecm_kind_parse(kind);
		if (!mk)
			fail("ec model: short, montgomery, edwards, twisted_edwards, "
			    "legendre");
		if (!ecm_from_weierstrass(mk, E, &Mo, &iso)) {
			printf("no %s model over Q (", ecm_kind_name(mk));
			if (mk == ECM_MONTGOMERY || mk == ECM_EDWARDS ||
			    mk == ECM_TWISTED_EDWARDS)
				printf("it needs a rational 2-torsion point (x0, y0) "
				    "with 3 x0^2 + 2 a2 x0 + a4 a square, after "
				    "completing the square");
			else if (mk == ECM_LEGENDRE)
				printf("it needs full rational 2-torsion and e2 - e1 a "
				    "square");
			else
				printf("not available");
			printf(")\n");
			rep_bool("exists", 0);
			return;
		}
		{
			struct sbuf b = {0};
			struct ec W;
			struct ecmpt N0;
			ecm_print(&b, &Mo);
			printf("%s model: %s\n", ecm_kind_name(mk), sb_str(&b));
			rep_str("model", sb_str(&b));
			sb_clear(&b);
			eciso_print(&b, QQc(), &iso);
			printf("  E -> the model's Weierstrass form by %s, then the "
			    "rational map of the model\n", sb_str(&b));
			ecm_to_weierstrass(&Mo, &W);
			ecm_neutral(&Mo, &N0);
			printf("  neutral element: %s\n", N0.inf ? "the point at "
			    "infinity" : "");
			if (!N0.inf)
				printf("    (%s, %s)\n", fe_str(QQc(), &N0.x),
				    fe_str(QQc(), &N0.y));
			free(b.p);
			/* certificate: torsion and small points go across and back */
			{
				struct ec Em;
				struct eciso m;
				struct ecpts T;
				uint32_t i;
				int moved = 0;
				ecq_minimal(E, &Em, &m);
				ecpts_init(&T);
				ecq_search(&Em, 200, &T, 12);
				for (i = 0; i < T.n; i++) {
					struct ecpt P, Q, Wp, back, again;
					struct ecmpt mp;
					ecpts_get(&T, i, &P);
					eciso_pt_inv(E, &m, &Q, &P);
					eciso_pt(E, &iso, &Wp, &Q);
					if (!ecm_pt_from_w(&Mo, &W, &Wp, &mp))
						continue;
					if (!ecm_on(&Mo, &mp))
						fail("internal: model point off the model");
					if (!ecm_pt_to_w(&Mo, &W, &mp, &back) ||
					    !ec_pt_eq(&W, &back, &Wp))
						fail("internal: model maps do not compose");
					eciso_pt_inv(E, &iso, &again, &back);
					if (!ec_pt_eq(E, &again, &Q))
						fail("internal: isomorphism round trip");
					if (moved < 4)
						printf("  %s -> (%s, %s)\n", ec_pt_str(E, &Q),
						    fe_str(QQc(), &mp.x), fe_str(QQc(), &mp.y));
					moved++;
				}
				printf("  check: %d points mapped to the model, onto it, "
				    "and back to themselves\n", moved);
				ecpts_free(&T);
			}
		}
		return;
	}
	if (!strcmp(op, "integral")) {
		int64_t B = *s ? strtoll(s, NULL, 10) : 100000;
		struct ec Em;
		struct eciso m;
		int64_t x;
		int cnt = 0;
		const struct ecf *Fq = QQc();
		if (!ecq_is_integral(E))
			fail("ec integral: the curve's coefficients must be integers");
		if (B > 10000000)
			B = 10000000;
		(void)Em;
		(void)m;
		rep_list("points");
		for (x = -B; x <= B; x++) {
			fe X;
			struct ecpt P[2];
			int n, k;
			X = fe_from_q(zq_int(IZ(x)));
			{
				/* quick: the discriminant in y must be a square */
				zint a1 = fe_qz(&E->a1), a3 = fe_qz(&E->a3);
				zint L = z_add(z_mul(a1, IZ(x)), a3), R;
				R = z_add(z_mul(z_add(z_mul(z_add(IZ(x), fe_qz(&E->a2)),
				    IZ(x)), fe_qz(&E->a4)), IZ(x)), fe_qz(&E->a6));
				if (!zx_is_square(z_add(z_mul(L, L), z_mul(IZ(4), R)),
				    NULL))
					continue;
			}
			n = ec_lift_x(E, &X, P);
			for (k = 0; k < n; k++) {
				if (!z_is_one(fe_q(&P[k].y).d))
					continue;
				printf("  %s\n", ec_pt_str(E, &P[k]));
				rep_str(NULL, ec_pt_str(E, &P[k]));
				cnt++;
			}
			(void)Fq;
		}
		rep_end();
		printf("integral points with |x| <= %lld: %d (a search: complete "
		    "only up to the bound)\n", (long long)B, cnt);
		rep_int("count", cnt);
		return;
	}
	fail("ec: unknown operation %s", op);
}

/* ---- the families ---- */

static void
cmd_congruent(const char *s)
{
	zint n, sq;
	const char *e;
	struct ec E;
	struct ecq_mw M;
	fe a[5];

	if (!z_parse(skip(s), &n, &e) || z_sign(n) <= 0)
		fail("congruent: a positive integer");
	{
		zint part = zx_squarefree_part(n, &sq);
		if (!z_eq(part, n))
			printf("%s = %s * %s^2: the same question as for %s\n",
			    z_str(n), z_str(part), z_str(sq), z_str(part));
		n = part;
	}
	a[0] = a[1] = a[2] = a[4] = fe_from_q(zq_int(0));
	a[3] = fe_from_q(zq_int(z_neg(z_mul(n, n))));
	ec_init(&E, QQc(), a);
	print_curve_line("n is congruent iff E_n has positive rank, E_n: ", &E);
	ecq_mw_compute(&E, &M, 3000);
	print_mw(&M);
	rep_int("n", z_fits_i64(n) ? z_to_i64(n) : 0);
	if (M.gens.n) {
		struct ecpt P, Q;
		struct zq x, y, A, B, C;
		ecpts_get(&M.gens, 0, &P);
		eciso_pt_inv(&E, &M.tomin, &Q, &P);
		/* a point with y != 0: a = (x^2 - n^2)/y, b = 2nx/y,
		 * c = (x^2 + n^2)/y */
		x = fe_q(&Q.x);
		y = fe_q(&Q.y);
		{
			struct zq n2 = zq_int(z_mul(n, n));
			A = zq_div(zq_sub(zq_mul(x, x), n2), y);
			B = zq_div(zq_mul(zq_int(z_mul(IZ(2), n)), x), y);
			C = zq_div(zq_add(zq_mul(x, x), n2), y);
		}
		if (z_sign(A.n) < 0)
			A = zq_neg(A);
		if (z_sign(B.n) < 0)
			B = zq_neg(B);
		if (z_sign(C.n) < 0)
			C = zq_neg(C);
		if (!zq_eq(zq_add(zq_mul(A, A), zq_mul(B, B)), zq_mul(C, C)) ||
		    !zq_eq(zq_div(zq_mul(A, B), zq_int(IZ(2))), zq_int(n)))
			fail("internal: the triangle check failed");
		printf("%s is congruent: the right triangle with sides\n",
		    z_str(n));
		printf("  a = %s\n  b = %s\n  c = %s\n", zq_str(A), zq_str(B),
		    zq_str(C));
		printf("  check: a^2 + b^2 = c^2 and a b / 2 = %s (exact)\n",
		    z_str(n));
		rep_bool("congruent", 1);
		rep_str("a", zq_str(A));
		rep_str("b", zq_str(B));
		rep_str("c", zq_str(C));
	} else if (M.rank_hi == 0) {
		printf("%s is not congruent (proved: E_n has rank 0 by the "
		    "descent; its torsion Z/2 x Z/2 gives only degenerate "
		    "triangles)\n", z_str(n));
		rep_bool("congruent", 0);
	} else {
		/* Tunnell: n odd: #{2x^2 + y^2 + 8z^2 = n} = 2 #{... + 32 z^2} */
		int64_t nn = z_fits_i64(n) ? z_to_i64(n) : -1;
		if (nn > 0 && nn < 100000000) {
			int64_t A1 = 0, B1 = 0, x, y, z, m = nn & 1 ? nn : nn / 2;
			int64_t c1 = nn & 1 ? 2 : 4, c3 = nn & 1 ? 8 : 8, c3b = nn & 1 ?
			    32 : 32;
			for (x = -(int64_t)sqrt((double)m) - 1; x * x <= m + 1; x++)
				for (y = -(int64_t)sqrt((double)m) - 1; y * y <= m + 1; y++)
					for (z = -(int64_t)sqrt((double)m) - 1; z * z <= m + 1;
					    z++) {
						if (nn & 1) {
							if (2 * x * x + y * y + c3 * z * z == m)
								A1++;
							if (2 * x * x + y * y + c3b * z * z == m)
								B1++;
						} else {
							if (c1 * x * x + y * y + 8 * z * z == m)
								A1++;
							if (c1 * x * x + y * y + 32 * z * z == m)
								B1++;
						}
					}
			printf("Tunnell's counts: %lld and %lld: %s\n", (long long)A1,
			    (long long)B1, A1 == 2 * B1 ? "consistent with congruent "
			    "(a proof needs BSD, or a point: search further)" :
			    "not congruent (Tunnell, unconditionally)");
			rep_bool("tunnell_congruent", A1 == 2 * B1);
		}
	}
	ecq_mw_free(&M);
}

static void
cmd_mordell(const char *s)
{
	zint k;
	const char *e;
	struct ec E;
	fe a[5];
	int64_t B = 100000;

	if (!z_parse(skip(s), &k, &e) || z_is_zero(k))
		fail("mordell: a nonzero integer k (y^2 = x^3 + k)");
	e = skip(e);
	if (!strncmp(e, "bound", 5))
		B = strtoll(e + 5, NULL, 10);
	a[0] = a[1] = a[2] = a[3] = fe_from_q(zq_int(0));
	a[4] = fe_from_q(zq_int(k));
	ec_init(&E, QQc(), a);
	ec_summary(&E, 2000, 1);
	printf("integral points:\n");
	{
		char buf[64];
		snprintf(buf, sizeof buf, "%lld", (long long)B);
		ec_op(&E, "integral", buf);
	}
}

static void
cmd_cubesum(const char *s)
{
	zint k;
	const char *e;
	char eq[256];
	struct dio_opts o;

	if (!z_parse(skip(s), &k, &e) || z_sign(k) <= 0)
		fail("cubesum: a positive integer k (x^3 + y^3 = k z^3)");
	memset(&o, 0, sizeof o);
	o.want = 1;
	o.nonzero = 1;
	e = skip(e);
	if (!strncmp(e, "positive", 8))
		o.positive = 1;
	snprintf(eq, sizeof eq, "x^3 + y^3 = %s*z^3", z_str(k));
	printf("%s as a sum of two rational cubes: x^3 + y^3 = %s z^3 with "
	    "x, y, z != 0%s\n", z_str(k), z_str(k), o.positive ? ", x, y > 0" :
	    "");
	dio_solve(eq, &o);
}

static void
cmd_diophantine(const char *s, int maps_only)
{
	static const char *const kw[] = {"positive", "integer", "integral",
		"nonzero", "count", "maxmult", "search", NULL};
	char eq[4096];
	const char *t, *end = s + strlen(s);
	struct dio_opts o;

	memset(&o, 0, sizeof o);
	o.want = 1;
	for (t = s; *t; t++) {
		int k;
		if (t > s && isalnum((unsigned char)t[-1]))
			continue;
		for (k = 0; kw[k]; k++) {
			size_t l = strlen(kw[k]);
			if (!strncmp(t, kw[k], l) && !isalnum((unsigned char)t[l]))
				break;
		}
		if (kw[k]) {
			end = t;
			break;
		}
	}
	if ((size_t)(end - s) >= sizeof eq)
		fail("diophantine: the equation is too long");
	memcpy(eq, s, (size_t)(end - s));
	eq[end - s] = 0;
	for (t = end; *t;) {
		t = skip(t);
		if (!strncmp(t, "positive", 8)) {
			o.positive = 1;
			t += 8;
		} else if (!strncmp(t, "integral", 8)) {
			o.integral = 1;
			t += 8;
		} else if (!strncmp(t, "integer", 7)) {
			o.integral = 1;
			t += 7;
		} else if (!strncmp(t, "nonzero", 7)) {
			o.nonzero = 1;
			t += 7;
		} else if (!strncmp(t, "count", 5)) {
			o.want = (int)strtol(t + 5, (char **)&t, 10);
		} else if (!strncmp(t, "maxmult", 7)) {
			o.maxmult = (int)strtol(t + 7, (char **)&t, 10);
		} else if (!strncmp(t, "search", 6)) {
			o.search = strtoll(t + 6, (char **)&t, 10);
		} else if (*t == ',' )
			t++;
		else if (*t)
			fail("diophantine: unknown option at \"%s\"", t);
	}
	if (maps_only)
		o.want = -1;
	dio_solve(eq, &o);
}

int
ec_command(struct P *p)
{
	static char text[8192];
	char w[48];
	uint32_t pos = cmdp_pos(p);
	static const char *const words[] = {"ec", "ellcurve", "elliptic",
		"diophantine", "weierstrass", "congruent", "mordell", "cubesum",
		NULL};
	int k;
	zmark mk;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; words[k] && strcmp(words[k], w); k++)
		;
	if (!words[k] || func_find(sym_cstr(w))) {
		cmdp_seek(p, pos);
		return 0;
	}
	text[0] = 0;
	if (!cmdp_eos(p))
		cmdp_rest(p, text, sizeof text, NULL);
	/* curves over finite fields and Z/n are for a later hook */
	if (strstr(text, " mod ") || strstr(text, "over GF") ||
	    strstr(text, "over F_") || strstr(text, "over Z/")) {
		cmdp_seek(p, pos);
		return 0;
	}
	notes_flush();
	mk = z_mark();
	if (k <= 2) {
		struct ec E;
		const char *rest;
		char op[32];
		int n = 0;
		if (!text[0])
			fail("ec: give a curve: ec [a1, a2, a3, a4, a6], ec [a4, a6] "
			    "or ec y^2 = x^3 + ...");
		parse_curve(text, &E, &rest);
		rest = skip(rest);
		while (isalpha((unsigned char)*rest) && n < 31)
			op[n++] = *rest++;
		op[n] = 0;
		rep_begin(n ? op : "ec");
		if (!n || !strcmp(op, "info"))
			ec_summary(&E, 1000, 1);
		else
			ec_op(&E, op, rest);
	} else if (!strcmp(w, "diophantine")) {
		rep_begin("diophantine");
		cmd_diophantine(text, 0);
	} else if (!strcmp(w, "weierstrass")) {
		rep_begin("weierstrass");
		cmd_diophantine(text, 1);
	} else if (!strcmp(w, "congruent")) {
		rep_begin("congruent");
		cmd_congruent(text);
	} else if (!strcmp(w, "mordell")) {
		rep_begin("mordell");
		cmd_mordell(text);
	} else if (!strcmp(w, "cubesum")) {
		rep_begin("cubesum");
		cmd_cubesum(text);
	}
	z_release(mk);
	return 1;
}

/* nlp.c: is w one of these commands (so the line is ucas, not English) */
int
ec_word(const char *w)
{
	static const char *const words[] = {"ec", "ellcurve", "elliptic",
		"diophantine", "weierstrass", "congruent", "mordell", "cubesum",
		NULL};
	int k;

	for (k = 0; words[k]; k++)
		if (!strcmp(words[k], w))
			return 1;
	return 0;
}
