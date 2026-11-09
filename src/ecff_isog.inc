/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ecff_cmd.c, part 5: modular polynomials, isogenies, division
 * polynomials */

int modpoly_mod(const struct mf *F, int l, u64 *c);
int modpoly_z(int l, zz *c, int *nprimes, int *verified);
void modpoly_at(const struct mf *F, int l, const u64 *c, const u64 *j,
    struct px *r);
int velu(const struct ecp_curve *E, const struct ecp_pt *P, int l,
    struct ecp_curve *E2, struct px *h, u64 *v, u64 *w);
void velu_map(const struct ecp_curve *E, const struct ecp_pt *P, int l,
    const struct ecp_pt *R, struct ecp_pt *out);
void kohel(const struct ecp_curve *E, const struct px *h,
    struct ecp_curve *E2);
int px_ddf(const struct mf *F, const struct px *f, int *counts, int maxd);

static int
small_prime_arg(const char *what, int max)
{
	nn v;

	need_nn(&v, what);
	if (!nn_fits_u64(&v) || v.w[0] < 2 || v.w[0] > (u64)max ||
	    !nt_isprime(v.w[0]))
		fail("%s: a prime from 2 to %d", what, max);
	return (int)v.w[0];
}

static void
cmd_modpoly(void)
{
	int l = small_prime_arg("l", 31), L2 = l + 2, i, k, np, ver, first = 1;
	static char buf[200000];
	size_t o = 0;

	skipws();
	if (takeword("mod")) {
		nn p;
		struct mf F;
		u64 *c = xmalloc(sizeof(u64) * (size_t)(L2 * L2) * FW);
		need_nn(&p, "p");
		if (!mf_init(&F, &p) || !nn_isprime(&p))
			fail("modpoly: p must be an odd prime");
		if (!modpoly_mod(&F, l, c))
			fail("modpoly: the q-expansion system is singular mod %s "
			    "(p too small?)", nn_str(&p));
		for (i = l + 1; i >= 0; i--)
			for (k = l + 1; k >= 0; k--) {
				const u64 *v = c + (size_t)(i * L2 + k) * FW;
				if (mf_is_zero(&F, v) || o + 200 > sizeof buf)
					continue;
				o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%s*X^%d*Y^%d",
				    first ? "" : " + ", mf_str(&F, v), i, k);
				first = 0;
			}
		printf("Phi_%d(X, Y) mod %s = %s\n", l, nn_str(&p), buf);
		printf("  how: the unknown coefficients solve Phi(j(q), j(q^%d)) = 0 "
		    "for the q-expansion of j = E4^3/Delta mod p (linear algebra)\n",
		    l);
		free(c);
		return;
	}
	{
		zz *c = xcalloc((size_t)(L2 * L2), sizeof(zz));
		int ok = modpoly_z(l, c, &np, &ver);
		if (!ok)
			fail("modpoly: the coefficients of Phi_%d exceed %d bits "
			    "(use mod p)", l, 64 * NW);
		for (i = l + 1; i >= 0; i--)
			for (k = i; k >= 0; k--) {
				zz a = c[i * L2 + k];
				if (zz_is_zero(&a) || o + 1200 > sizeof buf)
					continue;
				o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%s",
				    first ? (a.neg ? "-" : "") : (a.neg ? " - " : " + "),
				    "");
				a.neg = 0;
				if (!(nn_is_one(&a.m) && (i || k)))
					o += (size_t)snprintf(buf + o, sizeof buf - o, "%s",
					    zz_str(&a));
				if (i == k && i)
					o += (size_t)snprintf(buf + o, sizeof buf - o, "%sX^%d"
					    "*Y^%d", nn_is_one(&a.m) ? "" : "*", i, k);
				else if (i) {
					const char *m = nn_is_one(&a.m) ? "" : "*";
					if (k)
						o += (size_t)snprintf(buf + o, sizeof buf - o,
						    "%s(X^%d*Y^%d + X^%d*Y^%d)", m, i, k, k, i);
					else
						o += (size_t)snprintf(buf + o, sizeof buf - o,
						    "%s(X^%d + Y^%d)", m, i, i);
				}
				first = 0;
			}
		printf("Phi_%d(X, Y) = %s\n", l, buf);
		printf("  how: Phi_%d mod %d primes of 62 bits (q-expansions of j, "
		    "linear algebra), CRT with symmetric residues until stable\n",
		    l, np);
		printf("  check: agrees mod 2^61 - 1 (an independent prime): %s\n",
		    ver ? "yes" : "NO");
		free(c);
	}
}

static void
print_curve_ab(const char *label, const struct ecp_curve *E)
{
	u64 j[FW];

	ecp_jinv(E, j);
	printf("  %s: y^2 = x^3 + %s x + %s (j = %s)\n", label, fe(E->a),
	    fe(E->b), fe(j));
}

static void
cmd_ecisogeny(void)
{
	struct ecp_pt P;
	struct ecp_curve E2;
	struct px h;
	u64 v[FW], w[FW];
	char hb[4000];
	int l, i, ok = 1;
	nn N;
	struct eccert C;
	char how[200];

	parse_curve();
	skipws();
	if (takeword("kernel")) {
		struct ecp_pt T;
		parse_point(&P, 1);
		/* the order of P (small) */
		T = P;
		for (l = 1; l < 2000 && !T.inf; l++)
			ecp_add(CE, &T, &T, &P);
		if (!T.inf || l < 2)
			fail("ecisogeny: the kernel point must have a small order "
			    "(found none below 2000)");
		px_init(&h);
		if (!velu(CE, &P, l, &E2, &h, v, w))
			fail("ecisogeny: Velu failed");
		px_print(CF, hb, sizeof hb, &h, "x");
		printf("E -> E' of degree %d (Velu)\n", l);
		print_curve_ab("E", CE);
		print_curve_ab("E'", &E2);
		printf("  kernel: <%s> of order %d\n", pt(&P), l);
		printf("  kernel polynomial: %s\n", hb);
		printf("  how: Velu: a' = a - 5v, b' = b - 7w, v and w sums over half "
		    "the kernel\n");
		/* checks: points map onto E', the map is a homomorphism */
		for (i = 0; i < 4; i++) {
			struct ecp_pt A, B, S, fa, fb, fs, t;
			ecp_random_point(CE, &A);
			ecp_random_point(CE, &B);
			ecp_add(CE, &S, &A, &B);
			velu_map(CE, &P, l, &A, &fa);
			velu_map(CE, &P, l, &B, &fb);
			velu_map(CE, &P, l, &S, &fs);
			ecp_add(&E2, &t, &fa, &fb);
			if (!ecp_on_curve(&E2, &fa) || !ecp_eq(&E2, &t, &fs))
				ok = 0;
		}
		printf("  check: random points map onto E' and phi(A + B) = phi(A) "
		    "+ phi(B): %s\n", ok ? "yes" : "NO");
		{
			struct ecp_curve K2;
			kohel(CE, &h, &K2);
			printf("  check: Kohel's formulas from the kernel polynomial "
			    "give the same E': %s\n", mf_eq(CF, K2.a, E2.a) &&
			    mf_eq(CF, K2.b, E2.b) ? "yes" : "NO");
		}
		if (count(CM_AUTO, &N, &C, how, sizeof how)) {
			struct ecp_pt R, T2;
			ecp_random_point(&E2, &R);
			ecp_mul(&E2, &T2, &R, &N);
			printf("  check: #E' = #E (isogenous over F_p): %s\n",
			    T2.inf ? "yes (N R' = O on E')" : "NO");
		}
		px_free(&h);
		return;
	}
	/* all F_p-rational l-isogenies: roots of Phi_l(X, j(E)) */
	l = small_prime_arg("l (or kernel (x, y))", 31);
	{
		int L2 = l + 2, nr;
		u64 *c = xmalloc(sizeof(u64) * (size_t)(L2 * L2) * FW), j[FW],
		    roots[40 * FW];
		struct px phi;
		if (!modpoly_mod(CF, l, c))
			fail("ecisogeny: no Phi_%d mod p", l);
		ecp_jinv(CE, j);
		px_init(&phi);
		modpoly_at(CF, l, c, j, &phi);
		nr = px_roots(CF, &phi, roots, 40);
		printf("%d rational %d-isogen%s from j = %s\n", nr, l, nr == 1 ?
		    "y" : "ies", fe(j));
		print_curve_ab("E", CE);
		for (i = 0; i < nr; i++) {
			u64 *jj = roots + (size_t)i * FW;
			printf("  j%d: %s\n", i + 1, fe(jj));
		}
		printf("  how: the roots in F_p of Phi_%d(X, j(E)) (the classical "
		    "modular polynomial mod p)\n", l);
		printf("  structure: %s\n", nr == 0 ? "Frobenius has no "
		    "eigenvalue mod l (an Atkin prime for SEA)" : nr == l + 1 ?
		    "Frobenius acts as a scalar on E[l]" : nr == 1 ? "one rational "
		    "kernel (l divides the discriminant t^2 - 4p)" : "two rational "
		    "kernels (an Elkies prime)");
		px_free(&phi);
		free(c);
	}
}

static void
cmd_ecvolcano(void)
{
	int l, L2, n = 0, i, maxn = 60, q = 0;
	u64 *c, *nodes, *nb;
	int *deg, *par;
	struct px phi;

	parse_curve();
	l = small_prime_arg("l", 13);
	L2 = l + 2;
	c = xmalloc(sizeof(u64) * (size_t)(L2 * L2) * FW);
	if (!modpoly_mod(CF, l, c))
		fail("ecvolcano: no Phi_%d mod p", l);
	nodes = xmalloc(sizeof(u64) * FW * (size_t)maxn);
	nb = xmalloc(sizeof(u64) * FW * 32);
	deg = xcalloc((size_t)maxn, sizeof(int));
	par = xcalloc((size_t)maxn, sizeof(int));
	ecp_jinv(CE, nodes);
	n = 1;
	par[0] = -1;
	px_init(&phi);
	printf("the %d-isogeny graph of j = %s (breadth first, at most %d "
	    "vertices)\n", l, fe(nodes), maxn);
	while (q < n) {
		int nr, k;
		char line[4000];
		size_t o = 0;
		modpoly_at(CF, l, c, nodes + (size_t)q * FW, &phi);
		nr = px_roots(CF, &phi, nb, 32);
		deg[q] = nr;
		for (k = 0; k < nr; k++) {
			u64 *jj = nb + (size_t)k * FW;
			int found = -1;
			for (i = 0; i < n; i++)
				if (mf_eq(CF, nodes + (size_t)i * FW, jj))
					found = i;
			if (found < 0 && n < maxn) {
				mf_copy(CF, nodes + (size_t)n * FW, jj);
				par[n] = q;
				found = n++;
			}
			o += (size_t)snprintf(line + o, sizeof line - o, "%s%s",
			    k ? ", " : "", found >= 0 ? fe(jj) : "...");
		}
		printf("  %s: degree %d -> %s\n", fe(nodes + (size_t)q * FW), nr,
		    line);
		q++;
	}
	{
		int floor_ = 0, top = 0;
		for (i = 0; i < n; i++) {
			if (deg[i] <= 1)
				floor_++;
			if (deg[i] == l + 1)
				top++;
		}
		printf("  vertices: %d%s\n", n, n >= maxn ? " (truncated)" : "");
		printf("  shape: %d of degree l + 1 = %d (above the floor), %d of "
		    "degree <= 1 (the floor's leaves); an ordinary component is a "
		    "volcano: a crater cycle with trees of depth d hanging\n",
		    top, l + 1, floor_);
	}
	px_free(&phi);
	free(c);
	free(nodes);
	free(nb);
	free(deg);
	free(par);
}

static void
cmd_ecdivpoly(void)
{
	struct px *f, psi;
	nn nv;
	int n, i, counts[600], top, nr;
	static char buf[100000];
	u64 roots[64 * FW];

	parse_curve();
	need_nn(&nv, "n");
	n = (int)nv.w[0];
	if (n < 2 || n > 60)
		fail("ecdivpoly: 2 <= n <= 60");
	f = xmalloc(sizeof(struct px) * (size_t)(n + 3));
	ec_divpolys(CE, f, n + 2);
	px_init(&psi);
	if (n == 2) {
		/* psi_2^2 / 4 = x^3 + a x + b: its roots are the 2-torsion x */
		px_reserve(CF, &psi, 3);
		mf_copy(CF, px_coef(CF, &psi, 0), CE->b);
		mf_copy(CF, px_coef(CF, &psi, 1), CE->a);
		mf_zero(CF, px_coef(CF, &psi, 2));
		mf_one(CF, px_coef(CF, &psi, 3));
		psi.d = 3;
	} else
		px_copy(CF, &psi, &f[n]);
	px_print(CF, buf, sizeof buf, &psi, "x");
	printf("%s_%d = %.4000s%s\n", n & 1 ? "psi" : "psi/(2y)", n, buf,
	    strlen(buf) > 4000 ? " ..." : "");
	printf("  curve: %s\n  degree: %d\n", cc.desc, psi.d);
	nr = px_roots(CF, &psi, roots, 64);
	printf("  rational roots: %d (x-coordinates of points of order dividing "
	    "%d over F_p)\n", nr, n);
	{
		struct px sq, g, d;
		px_init(&g);
		px_init(&d);
		px_init(&sq);
		px_deriv(CF, &d, &psi);
		px_gcd(CF, &g, &psi, &d);
		px_copy(CF, &sq, &psi);
		if (g.d > 0)
			px_divmod(CF, &sq, NULL, &psi, &g);
		top = px_ddf(CF, &sq, counts, 599);
		{
			size_t o = 0;
			for (i = 1; i <= 599 && i <= (top > 0 ? top : 599); i++)
				if (counts[i])
					o += (size_t)snprintf(buf + o, sizeof buf - o, "%s%d of "
					    "degree %d", o ? ", " : "", counts[i], i);
			printf("  irreducible factors: %s\n", buf);
		}
		px_free(&g);
		px_free(&d);
		px_free(&sq);
	}
	if (n & 1) {
		/* every root x gives points (x, +-y) of order n over F_p or F_p^2 */
		int full = psi.d > 0 && nr == psi.d;
		int L = 1;
		for (i = 1; i < 600; i++)
			if (counts[i]) {
				int g = L, b = i, t;
				while (b) {
					t = g % b;
					g = b;
					b = t;
				}
				L = L / g * i;
			}
		printf("  torsion field: the x-coordinates of E[%d] lie in F_p^%d (the "
		    "lcm of the factor degrees)%s\n", n, L, full ?
		    " (all rational)" : "");
	}
	for (i = 0; i <= n + 2; i++)
		px_free(&f[i]);
	free(f);
	px_free(&psi);
	printf("  how: the recurrences psi_2m+1 = psi_m+2 psi_m^3 - psi_m-1 "
	    "psi_m+1^3 and psi_2m = psi_m (psi_m+2 psi_m-1^2 - psi_m-2 "
	    "psi_m+1^2)/(2y)\n");
}
