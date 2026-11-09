/*
 * homalg_cmd.c - the homological algebra commands (README, "Homological
 * algebra").  parse.c calls homalg_command; the JSON frontend's templates
 * produce the same statements.
 *
 *   space X                         cells, Euler characteristic, d o d = 0
 *   homology X [with G] [generators]
 *   cohomology X [with G] [generators]   (+ universal coefficient checks)
 *   les X mod m                     0 -> C -m-> C -> C (x) Z/m -> 0 and its
 *                                   long exact sequence, delta by the snake
 *   bockstein X mod p [pages N]     the Bockstein spectral sequence
 *   bockstein X mod p trace n [a] [page r]   j_r(p^(r-1) a), ker/im d_r
 *   prop01 X mod p [pages N]        Proposition 0.1, every n and r
 *
 * X is a space (simp.c: S^n, T^n, RP^n, CP^n, K, L(m,q), pt, products
 * "x", wedges "v", susp(X), cells(...), facets(...)); G an abelian group
 * ("Z/2", "Z/4 + Z").
 */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "ucas.h"
#include "homalg.h"

enum { HC_SPACE = 1, HC_HOMOLOGY, HC_COHOMOLOGY, HC_LES, HC_BSS, HC_PROP };

static const struct {
	const char *w;
	int c;
} hcmds[] = {
	{"space", HC_SPACE}, {"homology", HC_HOMOLOGY},
	{"cohomology", HC_COHOMOLOGY}, {"les", HC_LES},
	{"bockstein", HC_BSS}, {"bss", HC_BSS}, {"prop01", HC_PROP},
	{NULL, 0}
};

static const char *const kws[] = {"with", "mod", "pages", "trace", "page",
	"check", "generators", "unreduced", NULL};

/* ---- options ---- */

struct hopt {
	char space[2048];
	char with[256];
	int has_with, has_mod, gens, check, has_trace, has_page, unreduced;
	long long mod, pages, tdeg, page;
	char tvec[512];
};

static int
is_kw_at(const char *s, size_t i, const char **kw)
{
	int k;

	if (i > 0 && !isspace((unsigned char)s[i - 1]))
		return 0;
	for (k = 0; kws[k]; k++) {
		size_t n = strlen(kws[k]);
		if (strncmp(s + i, kws[k], n) == 0 && (s[i + n] == 0 ||
		    isspace((unsigned char)s[i + n]))) {
			*kw = kws[k];
			return 1;
		}
	}
	return 0;
}

static void
trim_copy(char *dst, size_t n, const char *a, const char *b)
{
	while (a < b && isspace((unsigned char)*a))
		a++;
	while (b > a && isspace((unsigned char)b[-1]))
		b--;
	if ((size_t)(b - a) >= n)
		fail("homalg: an argument is too long");
	memcpy(dst, a, (size_t)(b - a));
	dst[b - a] = 0;
}

static long long
opt_int(const char *s, const char *what)
{
	char *end;
	long long v;

	while (isspace((unsigned char)*s))
		s++;
	v = strtoll(s, &end, 10);
	while (isspace((unsigned char)*end))
		end++;
	if (end == s || *end)
		fail("homalg: %s: an integer, not \"%s\"", what, s);
	return v;
}

static void
parse_opts(struct hopt *o, const char *text)
{
	size_t i, n = strlen(text), start = 0;
	const char *kw = NULL, *cur = NULL;
	char arg[512];

	memset(o, 0, sizeof *o);
	for (i = 0; i <= n; i++) {
		const char *k2 = NULL;
		if (i < n && !is_kw_at(text, i, &k2))
			continue;
		/* the text from start to i belongs to cur (or is the space) */
		if (cur == NULL)
			trim_copy(o->space, sizeof o->space, text + start,
			    text + i);
		else {
			trim_copy(arg, sizeof arg, text + start, text + i);
			if (!strcmp(cur, "with")) {
				snprintf(o->with, sizeof o->with, "%s", arg);
				o->has_with = 1;
			} else if (!strcmp(cur, "mod")) {
				o->mod = opt_int(arg, "mod");
				o->has_mod = 1;
			} else if (!strcmp(cur, "pages"))
				o->pages = opt_int(arg, "pages");
			else if (!strcmp(cur, "page")) {
				o->page = opt_int(arg, "page");
				o->has_page = 1;
			} else if (!strcmp(cur, "check") ||
			    !strcmp(cur, "generators") ||
			    !strcmp(cur, "unreduced")) {
				if (arg[0])
					fail("homalg: \"%s\" takes no argument",
					    cur);
				if (cur[0] == 'c')
					o->check = 1;
				else if (cur[0] == 'u')
					o->unreduced = 1;
				else
					o->gens = 1;
			} else if (!strcmp(cur, "trace")) {
				char *sp = arg;
				o->has_trace = 1;
				while (*sp && !isspace((unsigned char)*sp) &&
				    *sp != '[')
					sp++;
				{
					char d[32];
					trim_copy(d, sizeof d, arg, sp);
					o->tdeg = opt_int(d, "trace degree");
				}
				trim_copy(o->tvec, sizeof o->tvec, sp,
				    arg + strlen(arg));
			}
		}
		if (i == n)
			break;
		cur = k2;
		kw = k2;
		start = i + strlen(k2);
		i = start - 1;
	}
	(void)kw;
	if (!o->space[0])
		fail("homalg: which space? (S^2, RP^4, T^2, L(9;1), S^2 x RP^3, "
		    "K, cells(...), facets(...))");
}

static hspace
get_space(const char *text)
{
	const char *err = NULL;
	hspace s = hs_parse(text, &err);

	if (s == HNONE)
		fail("homalg: \"%s\": %s", text, err ? err : "not a space");
	return s;
}

/*
 * The command's space and, unless "unreduced", its reduction: every
 * complex below is the reduced one, and cocycles and cycles go back to
 * the space's own cells through f^* and g.
 */
static hspace cur_s;
static hred cur_red = HNONE;

static hspace
use_space(const struct hopt *o)
{
	cur_s = get_space(o->space);
	cur_red = o->unreduced ? HNONE : hred_space(cur_s);
	return cur_s;
}

static hcx
cochains(void)
{
	return cur_red == HNONE ? hcx_cochains(cur_s) : hred_cochains(cur_red);
}

static hcx
chains(void)
{
	return cur_red == HNONE ? hcx_chains(cur_s) : hred_chains(cur_red);
}

/* a (co)cycle of the complex in use, as a vector over the space's cells
 * (y: hs_ncells(k) entries) */
static void
to_cells(int k, int coh, const zint *x, zint *y)
{
	uint32_t c, n = hs_ncells(cur_s, k);

	if (cur_red == HNONE) {
		for (c = 0; c < n; c++)
			y[c] = x[c];
	} else if (coh)
		hred_cochain_back(cur_red, k, x, y);
	else
		hred_chain_back(cur_red, k, x, y);
}

static void
print_reduced(void)
{
	int k, top = hs_top(cur_s);
	struct sbuf a = {0}, b = {0};

	if (cur_red == HNONE)
		return;
	rep_obj("reduced");
	rep_list("cells_before");
	for (k = 0; k <= top; k++) {
		sb_printf(&a, "%s%u", k ? " " : "", (unsigned)hs_ncells(cur_s, k));
		rep_int(NULL, hs_ncells(cur_s, k));
	}
	rep_end();
	rep_list("cells_after");
	for (k = 0; k <= top; k++) {
		sb_printf(&b, "%s%u", k ? " " : "",
		    (unsigned)hred_ncells(cur_red, k));
		rep_int(NULL, hred_ncells(cur_red, k));
	}
	rep_end();
	rep_end();
	printf("  reduced: cells %s -> %s (unit pivots cancelled; f g = 1, "
	    "1 - g f = dh + hd and both chain maps checked on every cell)\n",
	    sb_str(&a), sb_str(&b));
	free(a.p);
	free(b.p);
}

/* ---- printing ---- */

/* the matrix of M: G -> H in canonical coordinates (rows H.ninv) */
static void
canon_matrix(struct sbuf *b, const struct fgab *G, const struct fgab *H,
    const struct zmat *M)
{
	zint *x = xcalloc((size_t)G->n + 1, sizeof *x);
	zint *y = xcalloc((size_t)H->n + 1, sizeof *y);
	zint *c = xcalloc((size_t)H->ninv + 1, sizeof *c);
	struct zmat C = {0};
	int i, j;

	if (G->ninv == 0 || H->ninv == 0) {
		sb_puts(b, "0");
		goto out;
	}
	zm_init(&C, H->ninv, G->ninv);
	for (j = 0; j < G->ninv; j++) {
		ab_gen(G, j, x);
		ab_hom_apply(G, H, M, x, y);
		if (!ab_canon(H, y, c))
			fail("internal: homalg: an image outside its group");
		for (i = 0; i < H->ninv; i++)
			ZM(&C, i, j) = c[i];
	}
	if (zm_is_zero(&C))
		sb_puts(b, "0");
	else
		zm_print(b, &C);
	zm_free(&C);
out:
	free(x);
	free(y);
	free(c);
}

static void
canon_vec(struct sbuf *b, const struct fgab *G, const zint *x)
{
	zint *c = xcalloc((size_t)G->ninv + 1, sizeof *c);
	int i;

	if (G->ninv == 0) {
		sb_puts(b, "0");
		free(c);
		return;
	}
	if (!ab_canon(G, x, c))
		fail("internal: homalg: an element outside its group");
	sb_putc(b, '[');
	for (i = 0; i < G->ninv; i++) {
		if (i)
			sb_puts(b, ", ");
		z_print(b, c[i]);
	}
	sb_putc(b, ']');
	free(c);
}

/* a (co)chain as a combination of cells (a == 1: one entry per cell) */
static void
chain_text(struct sbuf *b, hspace s, int k, const zint *x, int g)
{
	int i, first = 1, terms = 0;

	for (i = 0; i < g; i++) {
		if (z_is_zero(x[i]))
			continue;
		if (++terms > 8) {
			sb_puts(b, " + ...");
			return;
		}
		if (z_sign(x[i]) < 0)
			sb_puts(b, first ? "-" : " - ");
		else if (!first)
			sb_puts(b, " + ");
		if (!z_is_pm1(x[i])) {
			z_print(b, z_abs(x[i]));
			sb_putc(b, ' ');
		}
		hs_cell_name(b, s, k, (uint32_t)i);
		first = 0;
	}
	if (first)
		sb_puts(b, "0");
}

static void
print_checks(void)
{
	int k;
	struct sbuf b = {0};

	print_reduced();
	sb_puts(&b, "  check:");
	for (k = 0; k < HK_NKIND; k++)
		if (hal_checks[k]) {
			char t[160];
			sb_printf(&b, "%s %s (%ld)", b.n > 9 ? "," : "",
			    hal_check_name[k], hal_checks[k]);
			snprintf(t, sizeof t, "%s: verified %ld times",
			    hal_check_name[k], hal_checks[k]);
			rep_check(hal_check_name[k], hal_checks[k], -1, 1, t);
		}
	if (b.n > 9)
		printf("%s\n", sb_str(&b));
	free(b.p);
}

static void
coeff_group(struct fgab *G, const struct hopt *o)
{
	const char *end = NULL;

	if (!o->has_with) {
		ab_cyclic(G, Z_ZERO);
		return;
	}
	if (!ab_parse(G, o->with, &end) || (end && *end))
		fail("homalg: \"%s\" is not an abelian group (Z, Z/2, "
		    "Z/4 + Z ...)", o->with);
}

/* ---- space ---- */

static void
cmd_space(const struct hopt *o)
{
	hspace s = get_space(o->space);
	int k, top = hs_top(s);

	printf("%s: %s complex of dimension %d\n", hs_name(s),
	    hs_kind(s) == HS_SIMPLICIAL ? "simplicial" : "cell", top);
	rep_str("space", hs_name(s));
	rep_str("model", hs_kind(s) == HS_SIMPLICIAL ? "simplicial" :
	    "cellular");
	rep_int("dimension", top);
	printf("  %s:", hs_kind(s) == HS_SIMPLICIAL ? "simplices" : "cells");
	rep_list("cells");
	for (k = 0; k <= top; k++) {
		printf(" %u", (unsigned)hs_ncells(s, k));
		rep_int(NULL, hs_ncells(s, k));
	}
	rep_end();
	printf("  (dimensions 0..%d)\n", top);
	printf("  Euler characteristic: %lld\n", hs_euler(s));
	rep_int("euler", hs_euler(s));
	for (k = 1; k <= top && k <= 4; k++) {
		const uint32_t *r, *c;
		const int64_t *v;
		uint32_t m = hs_bd(s, k, &r, &c, &v), e;
		struct sbuf b = {0};
		if (hs_ncells(s, k) > 6 || hs_ncells(s, k - 1) > 6)
			continue;
		for (e = 0; e < m; e++) {
			uint32_t cc = c[e];
			if (e == 0 || c[e - 1] != cc) {
				uint32_t f;
				sb_puts(&b, b.n ? ", d " : "d ");
				hs_cell_name(&b, s, k, cc);
				sb_puts(&b, " = ");
				for (f = e; f < m && c[f] == cc; f++) {
					if (f > e)
						sb_puts(&b, v[f] < 0 ? " - " :
						    " + ");
					else if (v[f] < 0)
						sb_puts(&b, "-");
					if (v[f] != 1 && v[f] != -1)
						sb_printf(&b, "%lld ", (long long)
						    (v[f] < 0 ? -v[f] : v[f]));
					hs_cell_name(&b, s, k - 1, r[f]);
				}
			}
		}
		if (b.n)
			printf("  %s\n", sb_str(&b));
		free(b.p);
	}
	printf("  check: d o d = 0 in every dimension\n");
	rep_check("d o d = 0", top > 1 ? top - 1 : 0, -1, 1,
	    "d_{k-1} d_k = 0 for every k");
}

/* ---- (co)homology ---- */

static void
ucheck(const struct fgab *X, const struct fgab *Y, const char *what, int n)
{
	if (!ab_isomorphic(X, Y, NULL, NULL))
		fail("internal: homalg: universal coefficients fail in degree "
		    "%d: %s is %s, the formula gives %s", n, what, ab_str(X),
		    ab_str(Y));
	hal_checks[HK_UCT]++;
}

static void
cmd_homology(const struct hopt *o, int coh)
{
	hspace s = use_space(o);
	int top = hs_top(s), n;
	struct fgab G = {0}, *H, *HZ;
	hcx cz = chains(), c, cc;
	char gname[64];

	coeff_group(&G, o);
	snprintf(gname, sizeof gname, "%s", ab_str(&G));
	cc = coh ? cochains() : cz;
	c = o->has_with ? hcx_tensor(cc, &G) : cc;
	H = xcalloc((size_t)top + 2, sizeof *H);
	HZ = xcalloc((size_t)top + 2, sizeof *HZ);
	printf("%s(%s; %s)\n", coh ? "H^*" : "H_*", hs_name(s), gname);
	rep_str("space", hs_name(s));
	rep_str("coefficients", gname);
	rep_list("groups");
	for (n = 0; n <= top; n++) {
		struct sbuf b = {0};
		hcx_cohomology(&H[n], c, coh ? n : -n);
		hcx_cohomology(&HZ[n], cz, -n);
		printf("  H%s%d = %s\n", coh ? "^" : "_", n, ab_str(&H[n]));
		rep_obj(NULL);
		rep_int("degree", n);
		rep_str("group", ab_str(&H[n]));
		rep_int("rank", H[n].rank);
		{
			zint *inv = xcalloc((size_t)H[n].ninv + 1, sizeof *inv);
			int k, m = ab_invariants(&H[n], inv);
			rep_list("torsion");
			for (k = 0; k < m; k++)
				if (!z_is_zero(inv[k]))
					rep_str(NULL, z_str(inv[k]));
			rep_end();
			free(inv);
		}
		if (o->gens && H[n].ninv > 0 && G.a <= 1) {
			int k;
			zint *x = xcalloc((size_t)H[n].n + 1, sizeof *x);
			zint *y = xcalloc((size_t)hs_ncells(s, n) + 1, sizeof *y);
			rep_list("generators");
			for (k = 0; k < H[n].ninv; k++) {
				sb_clear(&b);
				ab_gen(&H[n], k, x);
				to_cells(n, coh, x, y);
				chain_text(&b, s, n, y, (int)hs_ncells(s, n));
				printf("    generator %d: %s\n", k + 1, sb_str(&b));
				rep_str(NULL, sb_str(&b));
			}
			rep_end();
			free(x);
			free(y);
		}
		rep_end();
		free(b.p);
	}
	rep_end();
	/* universal coefficients */
	for (n = 0; n <= top; n++) {
		struct fgab A = {0}, B = {0}, S = {0}, zero = {0};
		const struct fgab *prev = n > 0 ? &HZ[n - 1] : NULL;
		if (prev == NULL)
			ab_free_abelian(&zero, 0);
		if (coh) {
			/* H^n(X; G) = Hom(H_n, G) + Ext(H_{n-1}, G) */
			ab_hom_group(&A, &HZ[n], &G);
			ab_ext(&B, prev ? prev : &zero, &G);
		} else {
			/* H_n(X; G) = H_n (x) G + Tor(H_{n-1}, G) */
			ab_tensor(&A, &HZ[n], &G);
			ab_tor(&B, prev ? prev : &zero, &G);
		}
		ab_direct_sum(&S, &A, &B, NULL, NULL, NULL, NULL);
		ucheck(&H[n], &S, coh ? "H^n(X; G)" : "H_n(X; G)", n);
		ab_free(&A);
		ab_free(&B);
		ab_free(&S);
		ab_free(&zero);
	}
	printf("  universal coefficients: %s for n = 0..%d\n", coh ?
	    "H^n(X; G) = Hom(H_n X, G) + Ext(H_{n-1} X, G)" :
	    "H_n(X; G) = H_n X (x) G + Tor(H_{n-1} X, G)", top);
	if (coh && o->has_with) {
		/* and on the cochain complex: H^n(C (x) G) = H^n(C) (x) G +
		 * Tor(H^{n+1}(C), G) */
		struct fgab *HC = xcalloc((size_t)top + 2, sizeof *HC);
		for (n = 0; n <= top + 1; n++)
			hcx_cohomology(&HC[n], cc, n);
		for (n = 0; n <= top; n++) {
			struct fgab A = {0}, B = {0}, S = {0};
			ab_tensor(&A, &HC[n], &G);
			ab_tor(&B, &HC[n + 1], &G);
			ab_direct_sum(&S, &A, &B, NULL, NULL, NULL, NULL);
			ucheck(&H[n], &S, "H^n(C (x) G)", n);
			ab_free(&A);
			ab_free(&B);
			ab_free(&S);
		}
		for (n = 0; n <= top + 1; n++)
			ab_free(&HC[n]);
		free(HC);
		printf("  universal coefficients: H^n(X; G) = H^n(X) (x) G + "
		    "Tor(H^{n+1}(X), G) for n = 0..%d\n", top);
	}
	print_checks();
	for (n = 0; n <= top; n++) {
		ab_free(&H[n]);
		ab_free(&HZ[n]);
	}
	free(H);
	free(HZ);
	ab_free(&G);
}

/* ---- the long exact sequence of 0 -> C -m-> C -> C (x) Z/m -> 0 ---- */

static void
cmd_les(const struct hopt *o)
{
	hspace s = use_space(o);
	hcx c = cochains(), cm;
	struct fgab Zm = {0};
	struct hles L;
	hmap f, g;
	int n;
	zint m;

	if (!o->has_mod || o->mod < 2)
		fail("homalg: les X mod m (m >= 2)");
	m = z_from_i64(o->mod);
	ab_cyclic(&Zm, m);
	cm = hcx_tensor(c, &Zm);
	f = hmap_scalar(c, m);
	g = hmap_identity(c, cm);
	hles_compute(&L, f, g);
	printf("long exact sequence of 0 -> C^*(%s) -%lld-> C^*(%s) -> "
	    "C^*(%s; Z/%lld) -> 0\n", hs_name(s), o->mod, hs_name(s),
	    hs_name(s), o->mod);
	rep_str("space", hs_name(s));
	rep_int("modulus", o->mod);
	rep_list("degrees");
	for (n = L.lo; n <= L.hi; n++) {
		struct sbuf b = {0}, mf = {0}, mg = {0}, md = {0};
		int i = n - L.lo;
		canon_matrix(&mf, &L.HA[i], &L.HB[i], &L.fs[i]);
		canon_matrix(&mg, &L.HB[i], &L.HC[i], &L.gs[i]);
		if (n < L.hi)
			canon_matrix(&md, &L.HC[i], &L.HA[i + 1], &L.delta[i]);
		else
			sb_puts(&md, "0");
		printf("  H^%d(Z) = %s -%s-> H^%d(Z) = %s -%s-> H^%d(Z/%lld) = "
		    "%s -delta %s->\n", n, ab_str(&L.HA[i]), sb_str(&mf), n,
		    ab_str(&L.HB[i]), sb_str(&mg), n, o->mod, ab_str(&L.HC[i]),
		    sb_str(&md));
		rep_obj(NULL);
		rep_int("degree", n);
		rep_str("H_Z", ab_str(&L.HA[i]));
		rep_str("H_Zm", ab_str(&L.HC[i]));
		rep_str("times_m", sb_str(&mf));
		rep_str("reduction", sb_str(&mg));
		rep_str("delta", sb_str(&md));
		rep_end();
		free(b.p);
		free(mf.p);
		free(mg.p);
		free(md.p);
	}
	rep_end();
	printf("  (maps in the groups' canonical generators; delta by the "
	    "snake lemma: lift, apply d, divide by %lld)\n", o->mod);
	printf("  exact at all %d spots (0 -> H^0 ... H^%d(Z/%lld) -> 0)\n",
	    3 * (L.hi - L.lo + 1), L.hi, o->mod);
	print_checks();
	hles_free(&L);
	ab_free(&Zm);
}

/* ---- the Bockstein spectral sequence ---- */

static void
print_groups(const char *label, const struct hcouple *P, int D)
{
	int n;

	printf("  %s =", label);
	for (n = P->lo; n <= P->hi; n++)
		printf("%s %s", n > P->lo ? "," : "", ab_str(D ? hc_D(P, n) :
		    hc_E(P, n)));
	printf("   (degrees %d..%d)\n", P->lo, P->hi);
}

static void
rep_groups(const char *key, const struct hcouple *P, int D)
{
	int n;

	rep_list(key);
	for (n = P->lo; n <= P->hi; n++)
		rep_str(NULL, ab_str(D ? hc_D(P, n) : hc_E(P, n)));
	rep_end();
}

static void
bss_setup(struct hss *S, hspace *sp, const struct hopt *o)
{
	hspace s = use_space(o);
	long long maxp = o->pages ? o->pages : 64;

	if (!o->has_mod)
		fail("homalg: bockstein X mod p");
	if (maxp < 1 || maxp > 1000)
		fail("homalg: pages: 1..1000");
	*sp = s;
	hss_bockstein(S, cochains(), z_from_i64(o->mod), (int)maxp);
}

static void
print_stable(const struct hss *S)
{
	if (S->stable) {
		printf("  stable from page %d: i_%d is injective in every "
		    "degree, so k_%d = 0, d_r = 0 and E_r = E_%d for every "
		    "r >= %d (E_inf = E_%d)\n", S->stable, S->stable, S->stable,
		    S->stable, S->stable, S->stable);
		rep_int("stable_page", S->stable);
	} else {
		printf("  not stable by page %d\n", S->npages);
		rep_null("stable_page");
	}
}

static void
cmd_bss(const struct hopt *o)
{
	struct hss S;
	hspace s;
	int r, n;

	bss_setup(&S, &s, o);
	printf("Bockstein spectral sequence of %s at p = %lld\n", hs_name(s),
	    o->mod);
	printf("  (the exact couple of 0 -> C -%lld-> C -> C (x) Z/%lld -> 0: "
	    "D_1 = H^*(%s; Z), E_1 = H^*(%s; Z/%lld), i = %lld, j = "
	    "reduction, k = Bockstein)\n", o->mod, o->mod, hs_name(s),
	    hs_name(s), o->mod, o->mod);
	rep_str("space", hs_name(s));
	rep_int("p", o->mod);
	rep_list("pages");
	for (r = 1; r <= S.npages; r++) {
		const struct hcouple *P = &S.pg[r - 1];
		char lab[32];
		struct sbuf b = {0};
		rep_obj(NULL);
		rep_int("page", r);
		snprintf(lab, sizeof lab, "D_%d", r);
		print_groups(lab, P, 1);
		rep_groups("D", P, 1);
		snprintf(lab, sizeof lab, "E_%d", r);
		print_groups(lab, P, 0);
		rep_groups("E", P, 0);
		rep_list("d");
		for (n = P->lo; n <= P->hi; n++) {
			struct sbuf m = {0};
			if (n + 1 > P->hi)
				break;
			canon_matrix(&m, hc_E(P, n), hc_E(P, n + 1),
			    &P->d[n - P->lo]);
			if (strcmp(sb_str(&m), "0")) {
				sb_printf(&b, "%sE^%d -> E^%d %s", b.n ? ", " :
				    "", n, n + 1, sb_str(&m));
				rep_obj(NULL);
				rep_int("from", n);
				rep_int("to", n + 1);
				rep_str("matrix", sb_str(&m));
				rep_end();
			}
			free(m.p);
		}
		rep_end();
		printf("  d_%d: %s\n", r, b.n ? sb_str(&b) : "0");
		rep_end();
		free(b.p);
	}
	rep_end();
	print_stable(&S);
	if (S.stable) {
		const struct hcouple *P = &S.pg[S.stable - 1];
		rep_groups("E_inf", P, 0);
	}
	printf("  (d_r in the canonical generators of E_r^n; every page "
	    "derived exactly: D' = i D, E' = ker d / im d, i' j' k' checked "
	    "well defined, each couple checked exact)\n");
	print_checks();
	hss_free(&S);
}

static void
parse_vec(const char *t, zint *y, int n)
{
	const char *s = t;
	int i = 0;

	while (isspace((unsigned char)*s))
		s++;
	if (*s++ != '[')
		fail("homalg: an element is [c1, c2, ...] (canonical "
		    "coordinates)");
	for (;;) {
		const char *end;
		while (isspace((unsigned char)*s))
			s++;
		if (*s == ']' && i == 0)
			break;
		if (i >= n || !z_parse(s, &y[i], &end))
			fail("homalg: the element needs %d coordinates", n);
		i++;
		s = end;
		while (isspace((unsigned char)*s))
			s++;
		if (*s == ',') {
			s++;
			continue;
		}
		if (*s == ']')
			break;
		fail("homalg: an element is [c1, c2, ...]");
	}
	if (i != n)
		fail("homalg: the element needs %d coordinates (the canonical "
		    "generators of H^n), not %d", n, i);
}

static void
trace_one(const struct hss *S, hspace s, int n, const zint *a, int r)
{
	struct htrace T;
	struct sbuf b = {0}, e = {0}, w = {0};
	const struct fgab *Er;
	int rr = r > S->npages ? S->npages : r;

	hss_trace(&T, S, n, a, r);
	Er = hc_E(&S->pg[rr - 1], T.m);
	canon_vec(&e, Er, T.e);
	printf("    r = %d: j_%d(%s^%d a) = %s in E_%d^%d = %s;", r, r,
	    z_str(S->p), r - 1, sb_str(&e), r, T.m, ab_str(Er));
	printf(" %s ker d_%d,", T.in_ker ? "in" : "NOT in", r);
	if (T.zero)
		printf(" zero (so in im d_%d^%d)\n", r, T.m - 1);
	else if (T.in_im) {
		canon_vec(&w, hc_E(&S->pg[rr - 1], T.m - 1), T.wit);
		printf(" in im d_%d^%d: = d_%d %s\n", r, T.m - 1, r, sb_str(&w));
	} else
		printf(" not in im d_%d^%d\n", r, T.m - 1);
	rep_obj(NULL);
	rep_int("page", r);
	rep_int("degree", T.m);
	rep_str("element", sb_str(&e));
	rep_str("group", ab_str(Er));
	rep_bool("zero", T.zero);
	rep_bool("in_ker_d", T.in_ker);
	rep_bool("in_im_d", T.in_im);
	if (T.wit)
		rep_str("preimage", sb_str(&w));
	rep_end();
	hss_trace_free(&T);
	free(b.p);
	free(e.p);
	free(w.p);
	(void)s;
}

static void
cmd_trace(const struct hopt *o)
{
	struct hss S;
	hspace s;
	const struct fgab *H;
	int n = (int)o->tdeg, k, r, rmax;
	zint *y, *x;

	bss_setup(&S, &s, o);
	H = hc_D(&S.pg[0], n);
	printf("Bockstein spectral sequence of %s at p = %lld: tracing "
	    "H^%d(%s; Z) = %s\n", hs_name(s), o->mod, n, hs_name(s), ab_str(H));
	rep_str("space", hs_name(s));
	rep_int("p", o->mod);
	rep_int("degree", n);
	rep_str("H", ab_str(H));
	if (H->ninv == 0) {
		printf("  H^%d is 0: nothing to trace\n", n);
		hss_free(&S);
		return;
	}
	rmax = S.stable ? S.stable : S.npages;
	if (rmax < 2)
		rmax = 2;
	y = xcalloc((size_t)H->ninv + 1, sizeof *y);
	x = xcalloc((size_t)H->n + 1, sizeof *x);
	print_stable(&S);
	rep_list("traces");
	for (k = 0; k < H->ninv; k++) {
		struct sbuf b = {0};
		if (o->tvec[0]) {
			if (k > 0)
				break;
			parse_vec(o->tvec, y, H->ninv);
		} else {
			memset(y, 0, (size_t)H->ninv * sizeof *y);
			y[k] = Z_ONE;
		}
		ab_from_canon(H, y, x);
		canon_vec(&b, H, x);
		printf("  a = %s (canonical coordinates in %s; cocycle ", sb_str(&b),
		    ab_str(H));
		rep_obj(NULL);
		rep_str("coordinates", sb_str(&b));
		sb_clear(&b);
		{
			zint *yc = xcalloc((size_t)hs_ncells(s, n) + 1,
			    sizeof *yc);
			to_cells(n, 1, x, yc);
			chain_text(&b, s, n, yc, (int)hs_ncells(s, n));
			free(yc);
		}
		printf("%s)\n", sb_str(&b));
		rep_str("cocycle", sb_str(&b));
		rep_list("pages");
		if (o->has_page) {
			if (o->page < 1 || o->page > 100000)
				fail("homalg: page: 1 ..");
			trace_one(&S, s, n, x, (int)o->page);
		} else
			for (r = 1; r <= rmax; r++)
				trace_one(&S, s, n, x, r);
		rep_end();
		rep_end();
		free(b.p);
	}
	rep_end();
	free(x);
	free(y);
	print_checks();
	hss_free(&S);
}

static void
cmd_prop(const struct hopt *o)
{
	struct hss S;
	hspace s;
	int n, r, rmax, all = 1;

	bss_setup(&S, &s, o);
	printf("Proposition 0.1 on %s at p = %lld: if a in H^n(X; Z) generates "
	    "a Z summand, then abar = j_r(p^(r-1) a) is in ker d_r^n and not "
	    "in im d_r^(n-1), for every r >= 1\n", hs_name(s), o->mod);
	rep_str("space", hs_name(s));
	rep_int("p", o->mod);
	print_stable(&S);
	if (!S.stable)
		fail("homalg: the sequence did not stabilise by page %d: "
		    "raise pages", S.npages);
	rmax = S.stable;
	rep_list("degrees");
	for (n = S.pg[0].lo; n <= S.pg[0].hi; n++) {
		const struct fgab *H = hc_D(&S.pg[0], n);
		rep_obj(NULL);
		rep_int("degree", n);
		rep_str("H", ab_str(H));
		printf("  n = %d: H^%d = %s", n, n, ab_str(H));
		if (H->rank == 0)
			printf(" (no Z summand: nothing to prove; checked "
			    "anyway)");
		printf("\n");
		rep_list("pages");
		for (r = 1; r <= rmax; r++) {
			struct hprop01 R;
			hss_prop01(&R, &S, n, r);
			printf("    r = %d: d_%d j_%d %s^%d = 0 on H^%d %s; "
			    "ker(H^%d -> E_%d^%d / im d_%d) has content %s %s\n",
			    r, r, r, z_str(S.p), r - 1, n, R.ker_ok ? "[ok]" :
			    "[FAILS]", n, r, n, r, z_str(R.content), H->rank == 0
			    ? "[vacuous]" : R.im_ok ? "!= 1 [ok]" : "= 1 [FAILS]");
			rep_obj(NULL);
			rep_int("page", r);
			rep_bool("in_ker", R.ker_ok);
			rep_str("content", z_str(R.content));
			rep_bool("not_in_im", R.im_ok || H->rank == 0);
			rep_bool("holds", R.ker_ok && (R.im_ok || H->rank == 0));
			rep_end();
			if (!(R.ker_ok && (R.im_ok || H->rank == 0)))
				all = 0;
		}
		rep_end();
		rep_end();
	}
	rep_end();
	printf("  why this is complete: a generates a Z summand of H iff its "
	    "class in H/T(H) is primitive; psi_r(a) = [j_r(p^(r-1) a)] in "
	    "E_r^n / im d_r^(n-1) vanishes iff a is in K = ker psi_r, and the "
	    "image of K in the free group H/T(H) contains a primitive vector "
	    "iff the gcd of its coordinates (the content) is 1\n");
	printf("  r > %d: page %d is stable (i_%d injective: k = 0, and i_%d "
	    "is an isomorphism of exact couples onto page %d with "
	    "j_(r+1)(p^r a) = j_r(p^(r-1) a)), so every later page and every "
	    "abar is the one at r = %d\n", rmax, rmax, rmax, rmax, rmax + 1,
	    rmax);
	printf("  %s for every n and every r >= 1\n", all ? "holds" :
	    "FAILS");
	rep_bool("holds", all);
	rep_proof(all ? "exhaustive" : "counterexample", all ? "checked for "
	    "every degree and every page up to the stable one; later pages "
	    "are isomorphic to it" : "a degree and page where it fails");
	print_checks();
	hss_free(&S);
}

/* ---- the entry point ---- */

/* nlp.c: a line starting with one of these words and a space is ucas */
int
homalg_line(const char *s, size_t n)
{
	int k;

	for (k = 0; hcmds[k].w; k++) {
		size_t m = strlen(hcmds[k].w);
		if (n > m + 1 && !strncmp(s, hcmds[k].w, m) && s[m] == ' ')
			return !func_find(sym_intern(s, m));
	}
	return 0;
}

int
homalg_command(struct P *p)
{
	static char text[8192];
	char w[48];
	uint32_t pos = cmdp_pos(p);
	int k, c = 0;
	struct hopt o;
	zmark m;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (k = 0; hcmds[k].w; k++)
		if (!strcmp(hcmds[k].w, w)) {
			c = hcmds[k].c;
			break;
		}
	/* the space starts with a word (S^2, RP^4, cells(...)); "space = 3"
	 * or "homology(x) = ..." are not ours */
	if (!c || func_find(sym_cstr(w)) || !cmdp_at(p, T_ID)) {
		cmdp_seek(p, pos);
		return 0;
	}
	text[0] = 0;
	cmdp_rest(p, text, sizeof text, NULL);
	parse_opts(&o, text);
	notes_flush();
	hal_checks_reset();
	cur_red = HNONE;
	/* the integers of this command die with it (the complexes and
	 * couples it made are not used by later statements) */
	m = z_mark();
	switch (c) {
	case HC_SPACE:
		rep_begin("space");
		cmd_space(&o);
		break;
	case HC_HOMOLOGY:
	case HC_COHOMOLOGY:
		rep_begin(c == HC_HOMOLOGY ? "homology" : "cohomology");
		cmd_homology(&o, c == HC_COHOMOLOGY);
		break;
	case HC_LES:
		rep_begin("les");
		cmd_les(&o);
		break;
	case HC_BSS:
		if (o.has_trace) {
			rep_begin("bockstein_trace");
			cmd_trace(&o);
		} else if (o.check) {
			rep_begin("prop01");
			cmd_prop(&o);
		} else {
			rep_begin("bockstein");
			cmd_bss(&o);
		}
		break;
	case HC_PROP:
		rep_begin("prop01");
		cmd_prop(&o);
		break;
	}
	z_release(m);
	return 1;
}
