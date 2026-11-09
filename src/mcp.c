/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * mcp.c - a Model Context Protocol server over stdio (ucas --mcp).
 *
 * JSON-RPC 2.0, one message per line (MCP's stdio transport).  It is a
 * layer over the JSON frontend (jsonmode.c): a tool call becomes a JSON
 * request ({"cmd", "args"}), served by json_serve with its sessions,
 * ambiguity answers, files and timeouts, and the frontend's response
 * becomes the call's result: text content (what the CLI prints), images
 * for GIFs, and structuredContent (the response object).
 *
 *   ucas --mcp           the commands grouped by area: one tool per area
 *                        with an "op" (about 15 tools)
 *   ucas --mcp --flat    one tool per command (about 120 tools)
 *   ucas --mcp --dir D   also write the files into D
 *
 * Tools, prompts and resources are generated from the command table,
 * so a command added to jsonmode.c appears here without changes (in
 * the "misc" group until it is given one).
 */
#include "ucas.h"
#include "json.h"
#include "jsonmode.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *const versions[] = {"2025-06-18", "2025-03-26",
	"2024-11-05", NULL};
static char proto[16] = "2025-06-18";
static int flat;

/* ---- the groups ---- */

struct group {
	const char *name, *title, *desc;
	const char *ops;	/* space separated, in this order */
};

static const struct group groups[] = {
	{"analyze", "Analyse a function",
	    "Exhaustive or sampled analysis of a function over its domain "
	    "(machine types u8..u64, i8..i64, f32, f64, real, int): values, "
	    "range, where it is defined, wrapping and C UB, critical points, "
	    "equivalence of two functions (with a counterexample), error and "
	    "fit of an approximation, cost under a GPU cost model, minimal "
	    "gate networks (with a proof of minimality), cheaper equivalent "
	    "implementations, polynomial/rational approximations.",
	    "show table parse range domain wrap crit equiv fit error cost gates "
	    "find more approx approx_more"},
	{"calculus", "Calculus",
	    "Derivatives (higher, partial, mixed), gradient, Hessian, "
	    "Laplacian, Jacobian, Wronskian, limits (symbolic and numeric), "
	    "Taylor series, implicit derivatives, integrals (antiderivative "
	    "checked by differentiating back; definite also numerically; "
	    "contour integrals with around), closed forms of sums and loops, "
	    "evaluation schemes, ODEs.",
	    "diff grad hessian laplacian jacobian wronskian limit taylor "
	    "implicit integrate antiderivative closed scheme continuous dsolve"},
	{"algebra", "Symbolic algebra",
	    "Polynomials and rational functions over Q, exactly: expand, "
	    "factor, simplify (conditions where a factor cancelled), partial "
	    "fractions, gcd/lcm, resultants, discriminants, division, solving "
	    "equations and systems (every solution substituted back), "
	    "substitution.",
	    "expand factor simplify together cancel collect apart gcd lcm "
	    "resultant discriminant quo rem divide solve substitute"},
	{"properties", "Properties of a function",
	    "Monotonicity, period, classification (identify: polynomials, "
	    "conics, ODEs ...), zeros, parity, symmetry, bounds, injective/"
	    "surjective/bijective, continuity, convexity, asymptotes; each "
	    "tagged proved, exhaustive or numeric.",
	    "monotonic period identify zeros parity symmetry bounded injective "
	    "surjective bijective continuity convexity asymptotes"},
	{"complex", "Complex numbers",
	    "Complex numbers and functions: a + bi forms, polar form, real/"
	    "imaginary parts, trigonometric rewrites (each checked), n-th "
	    "roots, holomorphy (Cauchy-Riemann), residues, poles.  z and w are "
	    "complex variables (args.assume.z = \"complex\").",
	    "complex re im conj modulus arg polar trig nthroots unity "
	    "holomorphic residue poles"},
	{"hypercomplex", "Quaternions and octonions",
	    "Cayley-Dickson algebras (quaternion, octonion, sedenion, cd N): "
	    "multiplication tables, algebra properties with counterexamples, "
	    "zero divisors, associators, commutators, 3-D rotations by "
	    "quaternions, axis-angle, rotation matrices, slerp.",
	    "multiplication_table algebra zerodivisors associator commutator "
	    "rotate axisangle rotmatrix frommatrix slerp"},
	{"rings", "Rings and finite fields",
	    "Number theory and finite fields: Z/n, GF(p^n) (\"ring\": "
	    "\"GF(2^8)\", \"modulus\": \"aes\"), Z[i]: inverses, orders, phi, "
	    "factoring, Legendre/Jacobi, square roots mod n, primitive roots, "
	    "discrete logs, CRT, congruences, division magic numbers, "
	    "polynomials over GF(q) (factor, irreducible, primitive), "
	    "matrices mod n, GF(2) bit maps.  A ring argument is \"ring\" "
	    "and/or \"modulus\" (alone: mod N).",
	    "ring inverse_mod order_mod phi carmichael factorint legendre "
	    "jacobi sqrtmod qr primroot dlog crt congruence magic ring_factor "
	    "irreducible primitive ring_roots ring_poly ring_gcd ring_norm "
	    "matrix gf2matrix"},
	{"transforms", "Transforms and series",
	    "Laplace and Z transforms (with the region of convergence) and "
	    "their inverses, ODEs and recurrences through them, transfer "
	    "functions, Fourier transforms and series, DFT/DTFT, generating "
	    "functions, infinite sums and products (test, closed form, "
	    "numeric value), convergence, radius of convergence, power series, "
	    "Euler-Maclaurin asymptotics.",
	    "laplace ilaplace laplace_solve ztrans iztrans zsolve tf fourier "
	    "ifourier fourier_series dft idft dtft genfunc egf infinite_sum "
	    "infinite_product converge radius powerseries asymptotic"},
	{"solids", "Areas and solids of revolution",
	    "Areas under and between curves, solids of revolution (volume by "
	    "disks/washers/shells, surface area, Pappus), arc length, "
	    "centroids: exact and numeric values, checks, and a picture.",
	    "area revolve volume surfacearea arclength centroid"},
	{"plot", "Plots and animations",
	    "GIF plots of functions, curves, regions, surfaces, vector and "
	    "slope fields (bounds chosen from the function's features), and "
	    "animations of scans and SIMD execution.  The GIF comes back as "
	    "an image.",
	    "plot plot3d neighbourhood animate"},
	{"c", "C optimiser",
	    "C99 in, faster C out for a chosen x86-64 target (zen4, skylake, "
	    "alderlake, x86-64-v2/v3/v4 ...): each function or expression is "
	    "lowered with C99 LP64 semantics (promotions, wrap, UB as a domain "
	    "condition, IEEE float and double) and rewritten only by forms "
	    "proved equal on every input (exhaustively, by SAT, by a stated "
	    "derivation, or by induction for loops summed in closed form); "
	    "loops over arrays are rewritten per element, strlen-style scans "
	    "become strlen; c_lower shows each loop's IR for the vectoriser; "
	    "optional differential test and rdtsc benchmark with the host cc.",
	    "c_optimize c_lower c_targets"},
	{"abstract_algebra", "Abstract algebra, exactly",
	    "Exact algebra with certificates: integers of any size (bigint), "
	    "integer matrices (Smith and Hermite forms with unimodular "
	    "transforms, kernels, images, solving over Z; rational matrices), "
	    "finitely generated abelian groups (\"Z/4 + Z/6\", \"coker "
	    "[[2, 0], [0, 3]]\"): invariants, isomorphisms with explicit maps, "
	    "Hom, tensor, Tor, Ext, homomorphisms with kernel/image/cokernel, "
	    "subgroups, element orders, exactness; permutation groups by "
	    "Schreier-Sims (\"S4\", \"(1 2 3), (1 2)\": order, orbits, "
	    "solvable, nilpotent, centre, membership, subgroups, cosets, "
	    "homomorphisms) and Cayley tables; polynomial ideals over Q and "
	    "F_p (Groebner bases with cofactors, membership certificates, "
	    "ideal equality); modules over F[x] (Smith form, rational "
	    "canonical form).  Every claim is checked "
	    "(U A V = D, preimages, inverse maps) and the check is reported.  "
	    "For instance {\"op\": \"abiso\", \"args\": {\"g\": \"Z/4 + "
	    "Z/6\", \"h\": \"Z/2 + Z/12\"}} or {\"op\": \"snf\", \"args\": "
	    "{\"matrix\": [[2, 4], [6, 8]]}}.",
	    "bigint bigint_factor snf hnf zmatrix qmatrix abgroup abiso abhom "
	    "abtensor abtor abext abmap abelem absub abexact permgroup permelem "
	    "permsub permhom cayley groebner idealmember idealeq fxmodule "
	    "pmsnf"},
	{"elliptic", "Elliptic curves and Diophantine equations",
	    "Elliptic curves over Q, exactly (analytic values with error "
	    "bounds): invariants, minimal models, Tate's algorithm, conductor, "
	    "torsion, rank by 2-isogeny descent, generators, canonical "
	    "heights, the group law, models (Montgomery, Edwards, Legendre) "
	    "with their maps, periods, L-values, BSD; Diophantine equations "
	    "reduced to elliptic curves (\"a/(b+c) + b/(a+c) + c/(a+b) = 4\" "
	    "positive), congruent numbers, Mordell curves, sums of two cubes.  "
	    "A curve is \"[a1, a2, a3, a4, a6]\", \"[a4, a6]\" or \"y^2 = x^3 "
	    "- x\".  For instance {\"op\": \"diophantine\", \"args\": "
	    "{\"equation\": \"a/(b+c) + b/(a+c) + c/(a+b) = 4\", \"positive\": "
	    "true}}.",
	    "elliptic diophantine weierstrass congruent mordell cubesum"},
	{"elliptic_ff", "Elliptic curves over finite fields and cryptography",
	    "Curves over F_p (any size to 521 bits; named: secp256k1, P-256, "
	    "P-384, P-521, Curve25519, Ed25519, Curve448, BN254, BLS12-381) and "
	    "GF(2^m): #E by naive/BSGS/CM/Schoof with a certificate, group "
	    "structure, twists, embedding degrees, scalar multiplication, "
	    "ECDSA (RFC 6979), ECDH, X25519/X448, Ed25519, SEC 1 encodings, "
	    "test vectors; discrete logs (Pohlig-Hellman, rho, MOV, Smart); "
	    "Weil/Tate pairings; CM and Hilbert class polynomials; modular "
	    "polynomials, Velu isogenies, isogeny graphs, division "
	    "polynomials; ECM factoring and ECPP certificates.  A curve is "
	    "a name, \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\".  "
	    "Mathematics, not hardened (constant-time) cryptography.",
	    "ec_info ec_order ec_points ec_add ec_mul ecm ecdsa_sign "
	    "ecdsa_verify ecdh x25519 x448 ed25519_sign ed25519_verify "
	    "ec_compress ec_decompress ec_validate ec_vectors ec_dlog ec_weak "
	    "ec_pairing hilbert ec_cm ecpp modpoly ec_isogeny ec_volcano "
	    "ec_divpoly ec_binary"},
	{"session", "Session",
	    "The session: define a function (typed parameters), list the "
	    "state, keep answers to ambiguities (assume), reset, the interface "
	    "schema, the cost models.",
	    "define state assume reset schema funcs models use help"},
	{"homological", "Homological algebra",
	    "Exact homology and cohomology of spaces (spheres, tori, RP^n, "
	    "CP^n, Klein bottle, lens spaces L(m,q), products, wedges, "
	    "suspensions, simplicial or cellular complexes you give) with any "
	    "f.g. abelian coefficients; long exact sequences with the snake "
	    "lemma; the Bockstein spectral sequence (pages, d_r, E_inf), "
	    "element tracing, and the check of bockstein.pdf's Proposition "
	    "0.1.  Every claim carries its checks (d o d = 0, exactness, "
	    "well-defined induced maps, universal coefficients).",
	    "space homology cohomology les bockstein bockstein_trace prop01"},
	{"proofs", "Proofs",
	    "An LCF proof kernel for abstract algebra: state theorems "
	    "(theorem), prove them (prove: the library's proofs, replays, "
	    "automation for linear integer arithmetic, abelian-group "
	    "equalities, subgroup membership and diagram chases; on failure "
	    "a counterexample from concrete Bockstein spectral sequences), "
	    "print the proof tree and its axioms (proof), re-check with an "
	    "independent checker (proof_check).  The library proves "
	    "bockstein.pdf's Proposition 0.1 for all complexes, primes, n "
	    "and r, two ways (theorem bockstein: ...; prove bockstein).",
	    "theorem prove proof proof_check axioms lemmas"},
};
#define NGROUPS (int)(sizeof groups / sizeof groups[0])

/* the group of a command, -1 when it has none (a tool of its own) */
static int
group_of(const char *cmd)
{
	int g;
	size_t n = strlen(cmd);

	for (g = 0; g < NGROUPS; g++) {
		const char *p = groups[g].ops;
		while ((p = strstr(p, cmd)) != NULL) {
			if ((p == groups[g].ops || p[-1] == ' ') &&
			    (p[n] == ' ' || p[n] == 0))
				return g;
			p += n;
		}
	}
	return -1;
}

static int
cmd_index(const char *name)
{
	int k;

	for (k = 0; k < json_ncmds(); k++)
		if (!strcmp(json_cmd_name(k), name))
			return k;
	return -1;
}

/* ---- JSON-RPC ---- */

static struct jv *
copy(const struct jv *v)
{
	struct sbuf b = {0};
	struct jv *r;
	char err[64];

	jv_write(&b, v);
	r = jv_parse(b.p, err, sizeof err, NULL);
	free(b.p);
	return r;
}

static void
send(struct jv *msg)
{
	struct sbuf b = {0};
	FILE *o = json_out();

	jv_write(&b, msg);
	sb_putc(&b, '\n');
	fputs(b.p, o);
	fflush(o);
	free(b.p);
	jv_free(msg);
}

static void
rpc_result(const struct jv *id, struct jv *result)
{
	struct jv *m = jv_new(J_OBJ);

	jv_set(m, "jsonrpc", jv_str("2.0"));
	jv_set(m, "id", id ? copy(id) : jv_new(J_NULL));
	jv_set(m, "result", result);
	send(m);
}

static void
rpc_error(const struct jv *id, int code, const char *msg)
{
	struct jv *m = jv_new(J_OBJ), *e = jv_new(J_OBJ);

	jv_set(m, "jsonrpc", jv_str("2.0"));
	jv_set(m, "id", id ? copy(id) : jv_new(J_NULL));
	jv_set(e, "code", jv_int(code));
	jv_set(e, "message", jv_str(msg));
	jv_set(m, "error", e);
	send(m);
}

static int
new_proto(void)
{
	return strcmp(proto, "2025-06-18") >= 0;
}

/* ---- tools ---- */

static struct jv *
text_item(const char *s)
{
	struct jv *t = jv_new(J_OBJ);

	jv_set(t, "type", jv_str("text"));
	jv_set(t, "text", jv_str(s));
	return t;
}

/* only the $defs the schema refers to (directly or through each other) */
static struct jv *
prune_defs(struct jv *s)
{
	struct jv *defs = jv_get(s, "$defs"), *keep = jv_new(J_OBJ);
	struct sbuf b = {0};
	int k, again = 1;
	char ref[80];

	if (defs == NULL)
		return s;
	while (again) {
		again = 0;
		sb_clear(&b);
		for (k = 0; k < s->n; k++)
			if (strcmp(s->k[k], "$defs"))
				jv_write(&b, s->v[k]);
		for (k = 0; k < keep->n; k++)
			jv_write(&b, keep->v[k]);
		for (k = 0; k < defs->n; k++) {
			snprintf(ref, sizeof ref, "#/$defs/%s\"", defs->k[k]);
			if (!jv_get(keep, defs->k[k]) && strstr(b.p, ref)) {
				jv_set(keep, defs->k[k], copy(defs->v[k]));
				again = 1;
			}
		}
	}
	free(b.p);
	jv_set(s, "$defs", keep);
	return s;
}

static struct jv *
tool_list(void)
{
	struct jv *tools = jv_new(J_ARR), *t, *s, *p, *a, *e;
	struct sbuf d = {0};
	int g, k;

	for (g = 0; !flat && g < NGROUPS; g++) {
		const char *q = groups[g].ops;
		t = jv_new(J_OBJ);
		jv_set(t, "name", jv_str(groups[g].name));
		jv_set(t, "title", jv_str(groups[g].title));
		sb_clear(&d);
		sb_printf(&d, "%s\n\nCall with {\"op\": OP, \"args\": {...}}; "
		    "expressions are ucas syntax strings (x^2 sin(x), isqrt(x: u4)) "
		    "or an AST ({\"op\": \"+\", \"args\": [...]}).  Ops (* = "
		    "required arg):\n", groups[g].desc);
		e = jv_new(J_ARR);
		while (*q) {
			char op[48];
			size_t n = strcspn(q, " ");
			snprintf(op, sizeof op, "%.*s", (int)n, q);
			q += n + (q[n] == ' ');
			if ((k = cmd_index(op)) < 0)
				continue;
			jv_push(e, jv_str(op));
			sb_printf(&d, "- %s(", op);
			json_cmd_params(k, &d);
			sb_printf(&d, "): %s\n", json_cmd_desc(k));
		}
		sb_puts(&d, "Every op also takes assume (answers to ambiguities, "
		    "from an error's readings), defaults, timeout_s, files.  "
		    "Example: {\"op\": \"");
		sb_puts(&d, e->n ? e->v[0]->s : "");
		sb_puts(&d, "\", \"args\": {\"f\": \"x^3 - 3x\"}}");
		jv_set(t, "description", jv_str(d.p));
		s = jv_new(J_OBJ);
		jv_set(s, "type", jv_str("object"));
		p = jv_new(J_OBJ);
		a = jv_new(J_OBJ);
		jv_set(a, "type", jv_str("string"));
		jv_set(a, "enum", e);
		jv_set(a, "description", jv_str("the operation"));
		jv_set(p, "op", a);
		a = jv_new(J_OBJ);
		jv_set(a, "type", jv_str("object"));
		jv_set(a, "description", jv_str("the op's named arguments (see "
		    "the description; the exact schema of each op is in the "
		    "session tool's schema op)"));
		jv_set(p, "args", a);
		jv_set(s, "properties", p);
		a = jv_new(J_ARR);
		jv_push(a, jv_str("op"));
		jv_set(s, "required", a);
		jv_set(s, "additionalProperties", jv_bool(0));
		jv_set(t, "inputSchema", s);
		if (new_proto())
			jv_set(t, "outputSchema", json_response_schema());
		jv_push(tools, t);
	}
	for (k = 0; k < json_ncmds(); k++) {
		const char *nm = json_cmd_name(k);
		if (!flat && group_of(nm) >= 0)
			continue;
		t = jv_new(J_OBJ);
		jv_set(t, "name", jv_str(nm));
		sb_clear(&d);
		sb_puts(&d, json_cmd_desc(k));
		if (!strcmp(nm, "ucas"))
			sb_puts(&d, ".  Any ucas statements, as typed at the ucas> "
			    "prompt, e.g. \"f(x: u8) = x*x\\nrange f\" or \"x[0] = 1; "
			    "x[k+1] = 2x[k]\"");
		else if (!strcmp(nm, "english"))
			sb_puts(&d, ".  E.g. \"what is the range of x squared for a "
			    "signed byte x?\"");
		else if (!strcmp(nm, "spirv_simplify"))
			sb_puts(&d, "; float instructions too (bit-exact, within "
			    "Vulkan precision, or approximations).  code is SPIR-V "
			    "assembly or GLSL");
		jv_set(t, "description", jv_str(d.p));
		jv_set(t, "inputSchema", prune_defs(json_cmd_schema(k, 1)));
		/* the response schema on ~120 tools would be 260 KB: grouped only */
		if (new_proto() && !flat)
			jv_set(t, "outputSchema", json_response_schema());
		jv_push(tools, t);
	}
	free(d.p);
	return tools;
}

/* the call in progress: its JSON-RPC id */
static struct jv *call_id;
static struct jv *captured;	/* resources: the response kept */
static int capture;

static void
error_text(struct sbuf *b, const struct jv *err)
{
	const struct jv *rd = jv_get(err, "readings");
	int k;

	sb_printf(b, "error (%s): %s", jv_gets(err, "code"),
	    jv_gets(err, "message"));
	if (jv_gets(err, "key"))
		sb_printf(b, "\nambiguous: set args.assume.%s to one of:",
		    jv_gets(err, "key"));
	for (k = 0; rd && k < rd->n; k++) {
		const struct jv *r = rd->v[k], *df = jv_get(r, "default");
		sb_printf(b, "\n  %d) %s%s", (int)jv_get(r, "index")->num,
		    jv_gets(r, "reading"), df && df->t == J_TRUE ? " (default)" :
		    "");
		if (jv_gets(r, "how"))
			sb_printf(b, ": %s", jv_gets(r, "how"));
	}
	if (jv_gets(err, "how"))
		sb_printf(b, "\n%s", jv_gets(err, "how"));
}

#define MAXIMG (4 << 20)	/* base64 bytes of an image in a result */

/* the JSON frontend's response as a CallToolResult */
static void
answer(struct jv *resp)
{
	struct jv *r = jv_new(J_OBJ), *content = jv_new(J_ARR), *sc,
	    *files, *err = jv_get(resp, "error"), *ok = jv_get(resp, "ok");
	struct sbuf b = {0};
	const char *text = jv_gets(resp, "text");
	int k;

	if (capture) {
		jv_free(captured);
		captured = copy(resp);
		jv_free(r);
		jv_free(content);
		return;
	}
	if (err)
		error_text(&b, err);
	if (text && text[0]) {
		if (b.n)
			sb_puts(&b, "\n\n");
		sb_puts(&b, text);
	} else if (!err && jv_get(resp, "result")) {
		jv_write(&b, jv_get(resp, "result"));
	}
	jv_push(content, text_item(b.n ? b.p : ""));
	sc = copy(resp);
	files = jv_get(sc, "files");
	for (k = 0; files && k < files->n; k++) {
		struct jv *f = files->v[k], *im;
		const char *mime = jv_gets(f, "mime"), *data = jv_gets(f,
		    "base64");
		if (data == NULL)
			continue;
		if (mime && !strncmp(mime, "image/", 6) && strlen(data) <= MAXIMG) {
			im = jv_new(J_OBJ);
			jv_set(im, "type", jv_str("image"));
			jv_set(im, "data", jv_str(data));
			jv_set(im, "mimeType", jv_str(mime));
			jv_push(content, im);
		} else if (strlen(data) <= MAXIMG) {
			struct jv *res = jv_new(J_OBJ), *rr = jv_new(J_OBJ);
			char uri[200];
			snprintf(uri, sizeof uri, "ucas://file/%s", jv_gets(f, "name"));
			jv_set(rr, "uri", jv_str(uri));
			jv_set(rr, "mimeType", jv_str(mime ? mime :
			    "application/octet-stream"));
			jv_set(rr, "blob", jv_str(data));
			jv_set(res, "type", jv_str("resource"));
			jv_set(res, "resource", rr);
			jv_push(content, res);
		} else {
			char m[200];
			snprintf(m, sizeof m, "(%s is too large to return: %zu bytes "
			    "of base64; ask for fewer frames or seconds)",
			    jv_gets(f, "name"), strlen(data));
			jv_push(content, text_item(m));
		}
		jv_set(f, "base64", jv_new(J_NULL));	/* not twice */
		jv_set(f, "base64", jv_bool(1));
	}
	for (k = 0; files && k < files->n; k++) {	/* "base64": true -> gone */
		struct jv *f = files->v[k], *nf = jv_new(J_OBJ);
		int i;
		for (i = 0; i < f->n; i++)
			if (strcmp(f->k[i], "base64")) {
				jv_set(nf, f->k[i], f->v[i]);
				f->v[i] = jv_new(J_NULL);
			}
		jv_free(f);
		files->v[k] = nf;
	}
	jv_set(r, "content", content);
	if (new_proto())
		jv_set(r, "structuredContent", sc);
	else
		jv_free(sc);
	jv_set(r, "isError", jv_bool(!(ok && ok->t == J_TRUE)));
	rpc_result(call_id, r);
	free(b.p);
}

static void
tools_call(const struct jv *id, const struct jv *params)
{
	const char *name = jv_gets(params, "name"), *op = NULL;
	const struct jv *args = jv_get(params, "arguments"), *a = args;
	struct jv *rq;
	int g;

	if (name == NULL) {
		rpc_error(id, -32602, "tools/call needs a tool name");
		return;
	}
	if (args && args->t != J_OBJ) {
		rpc_error(id, -32602, "arguments must be an object");
		return;
	}
	g = -1;
	if (!flat)
		for (g = 0; g < NGROUPS && strcmp(groups[g].name, name); g++)
			;
	if (!flat && g < NGROUPS) {
		op = jv_gets(args, "op");
		if (op == NULL || group_of(op) != g || cmd_index(op) < 0) {
			char m[600];
			snprintf(m, sizeof m, "%s needs \"op\", one of: %s", name,
			    groups[g].ops);
			rpc_error(id, -32602, m);
			return;
		}
		a = jv_get(args, "args");
		if (a && a->t != J_OBJ) {
			rpc_error(id, -32602, "args must be an object");
			return;
		}
	} else {
		op = name;
		if (cmd_index(op) < 0 || (!flat && group_of(op) >= 0)) {
			char m[200];
			snprintf(m, sizeof m, "unknown tool \"%.100s\"", name);
			rpc_error(id, -32602, m);
			return;
		}
	}
	rq = jv_new(J_OBJ);
	jv_set(rq, "id", copy(id));
	jv_set(rq, "cmd", jv_str(op));
	if (a && a->n)
		jv_set(rq, "args", copy(a));
	jv_free(call_id);
	call_id = copy(id);
	json_serve(rq);
	jv_free(rq);
}

/* ---- resources and prompts ---- */

static struct jv *
run_capture(const char *cmd)
{
	struct jv *rq = jv_new(J_OBJ), *r;

	jv_set(rq, "cmd", jv_str(cmd));
	capture = 1;
	json_serve(rq);
	capture = 0;
	jv_free(rq);
	r = captured;
	captured = NULL;
	return r;
}

static struct jv *
resource(const char *uri, const char *name, const char *mime,
    const char *desc)
{
	struct jv *r = jv_new(J_OBJ);

	jv_set(r, "uri", jv_str(uri));
	jv_set(r, "name", jv_str(name));
	jv_set(r, "mimeType", jv_str(mime));
	jv_set(r, "description", jv_str(desc));
	return r;
}

static void
resources_read(const struct jv *id, const struct jv *params)
{
	const char *uri = jv_gets(params, "uri");
	struct jv *c = jv_new(J_OBJ), *list, *res, *r = NULL;
	struct sbuf b = {0};

	if (uri == NULL) {
		jv_free(c);
		rpc_error(id, -32602, "resources/read needs a uri");
		return;
	}
	if (!strcmp(uri, "ucas://state")) {
		r = run_capture("state");
		jv_write(&b, jv_get(r, "result"));
		jv_set(c, "mimeType", jv_str("application/json"));
	} else if (!strcmp(uri, "ucas://help")) {
		r = run_capture("help");
		sb_puts(&b, jv_gets(r, "text") ? jv_gets(r, "text") : "");
		jv_set(c, "mimeType", jv_str("text/plain"));
	} else if (!strcmp(uri, "ucas://schema")) {
		r = json_schema();
		jv_write(&b, r);
		jv_set(c, "mimeType", jv_str("application/schema+json"));
	} else {
		jv_free(c);
		rpc_error(id, -32002, "resource not found");
		return;
	}
	jv_free(r);
	jv_set(c, "uri", jv_str(uri));
	jv_set(c, "text", jv_str(b.p ? b.p : ""));
	list = jv_new(J_ARR);
	jv_push(list, c);
	res = jv_new(J_OBJ);
	jv_set(res, "contents", list);
	rpc_result(id, res);
	free(b.p);
}

static const struct {
	const char *name, *desc, *arg, *argdesc, *text;
} prompts[] = {
	{"cheaper_implementation", "Find and prove a cheaper implementation "
	    "of a function on its machine types", "f", "the function, with "
	    "types: \"days(m: u8 in 1..12) = match m { 2 => 28, 4 | 6 | 9 | 11 "
	    "=> 30, _ => 31 }\" or isqrt(x: u4)",
	    "Use the ucas tools to find a cheaper implementation of %s.  "
	    "Define it with session/define (typed parameters), then: analyze "
	    "range and wrap (what it computes, where it overflows), analyze "
	    "cost (under the active GPU model), analyze find (cheaper "
	    "equivalent implementations, verified on the whole domain) and, "
	    "for up to 6 input bits, analyze gates (a minimal gate network "
	    "with a proof).  Confirm the best candidate with analyze equiv "
	    "and report it with its cost and the proof each tool gave."},
	{"prove_identity", "Prove (or refute with a counterexample) that two "
	    "expressions are equal", "identity", "the claimed identity, e.g. "
	    "sin(x)^2 + cos(x)^2 = 1 or (x: u8) * 2 = x << 1",
	    "Decide whether %s holds, with the ucas tools.  For polynomial or "
	    "rational expressions use algebra simplify on the difference (an "
	    "exact normal form is a proof; note any conditions where a factor "
	    "cancelled).  For machine-typed expressions use analyze equiv "
	    "(exhaustive over the domain, or a counterexample).  For "
	    "trigonometric ones use complex trig, which checks each form.  "
	    "State which kind of proof you got (symbolic, exhaustive, "
	    "numeric)."},
	{"explore_function", "Study a function: its properties, critical "
	    "points, integral and a plot", "f", "the function, e.g. x^3 - 3x "
	    "or sin(x)/x",
	    "Explore %s with the ucas tools: properties identify, analyze crit "
	    "and range, calculus diff and integrate, properties asymptotes if "
	    "it is rational, and plot plot (the picture comes back as an "
	    "image).  Summarise what is proved and what is numeric."},
};
#define NPROMPTS (int)(sizeof prompts / sizeof prompts[0])

static void
prompts_get(const struct jv *id, const struct jv *params)
{
	const char *name = jv_gets(params, "name"), *v;
	const struct jv *args = jv_get(params, "arguments");
	struct jv *res, *msgs, *m, *c;
	char text[2048];
	int k;

	for (k = 0; k < NPROMPTS && (name == NULL ||
	    strcmp(prompts[k].name, name)); k++)
		;
	if (k == NPROMPTS) {
		rpc_error(id, -32602, "unknown prompt");
		return;
	}
	v = jv_gets(args, prompts[k].arg);
	if (v == NULL) {
		rpc_error(id, -32602, "missing the prompt's argument");
		return;
	}
	snprintf(text, sizeof text, prompts[k].text, v);
	c = jv_new(J_OBJ);
	jv_set(c, "type", jv_str("text"));
	jv_set(c, "text", jv_str(text));
	m = jv_new(J_OBJ);
	jv_set(m, "role", jv_str("user"));
	jv_set(m, "content", c);
	msgs = jv_new(J_ARR);
	jv_push(msgs, m);
	res = jv_new(J_OBJ);
	jv_set(res, "description", jv_str(prompts[k].desc));
	jv_set(res, "messages", msgs);
	rpc_result(id, res);
}

/* ---- the messages ---- */

static void
initialize(const struct jv *id, const struct jv *params)
{
	const char *want = jv_gets(params, "protocolVersion");
	struct jv *r = jv_new(J_OBJ), *cap = jv_new(J_OBJ), *t, *info;
	int k;

	snprintf(proto, sizeof proto, "%s", versions[0]);
	for (k = 0; want && versions[k]; k++)
		if (!strcmp(versions[k], want))
			snprintf(proto, sizeof proto, "%s", want);
	setenv("UCAS_MCP_PROTO", proto, 1);	/* kept across reset's exec */
	jv_set(r, "protocolVersion", jv_str(proto));
	t = jv_new(J_OBJ);
	jv_set(t, "listChanged", jv_bool(0));
	jv_set(cap, "tools", t);
	jv_set(cap, "resources", jv_new(J_OBJ));
	jv_set(cap, "prompts", jv_new(J_OBJ));
	jv_set(r, "capabilities", cap);
	info = jv_new(J_OBJ);
	jv_set(info, "name", jv_str("ucas"));
	jv_set(info, "title", jv_str("ucas, a computer algebra system for "
	    "programmers"));
	jv_set(info, "version", jv_str("1"));
	jv_set(r, "serverInfo", info);
	jv_set(r, "instructions", jv_str("ucas analyses functions (exhaustively "
	    "on machine types), does exact algebra and calculus, transforms, "
	    "number theory and finite fields, and finds cheaper "
	    "implementations, each answer checked or proved.  Expressions are "
	    "ucas syntax strings; give parameters types (x: u8, x: f32, x: "
	    "real) where it matters.  Definitions persist across calls.  An "
	    "ambiguous input is an error listing its readings: retry with "
	    "args.assume.KEY.  structuredContent holds the typed result, "
	    "checks and proof."));
	rpc_result(id, r);
}

static void
dispatch(const struct jv *msg)
{
	const struct jv *id = jv_get(msg, "id"), *params = jv_get(msg,
	    "params");
	const char *method = jv_gets(msg, "method"), *jr = jv_gets(msg,
	    "jsonrpc");
	int notif = id == NULL;

	if (msg->t != J_OBJ || jr == NULL || strcmp(jr, "2.0") || method == NULL ||
	    (id && id->t != J_STR && id->t != J_NUM)) {
		if (msg->t == J_OBJ && method == NULL && (jv_get(msg, "result") ||
		    jv_get(msg, "error")))
			return;		/* a response to us: we send no requests */
		rpc_error(id && (id->t == J_STR || id->t == J_NUM) ? id : NULL,
		    -32600, "invalid request");
		return;
	}
	if (params && params->t != J_OBJ) {
		if (!notif)
			rpc_error(id, -32602, "params must be an object");
		return;
	}
	if (notif)
		return;		/* notifications/initialized, cancelled ... */
	if (!strcmp(method, "initialize"))
		initialize(id, params);
	else if (!strcmp(method, "ping"))
		rpc_result(id, jv_new(J_OBJ));
	else if (!strcmp(method, "tools/list")) {
		struct jv *r = jv_new(J_OBJ);
		jv_set(r, "tools", tool_list());
		rpc_result(id, r);
	} else if (!strcmp(method, "tools/call"))
		tools_call(id, params);
	else if (!strcmp(method, "resources/list")) {
		struct jv *r = jv_new(J_OBJ), *l = jv_new(J_ARR);
		jv_push(l, resource("ucas://state", "state", "application/json",
		    "the session: defined functions, models, bases, assumptions"));
		jv_push(l, resource("ucas://help", "help", "text/plain",
		    "the ucas command summary"));
		jv_push(l, resource("ucas://schema", "schema",
		    "application/schema+json", "every command and its arguments "
		    "as JSON Schema"));
		jv_set(r, "resources", l);
		rpc_result(id, r);
	} else if (!strcmp(method, "resources/templates/list")) {
		struct jv *r = jv_new(J_OBJ);
		jv_set(r, "resourceTemplates", jv_new(J_ARR));
		rpc_result(id, r);
	} else if (!strcmp(method, "resources/read"))
		resources_read(id, params);
	else if (!strcmp(method, "prompts/list")) {
		struct jv *r = jv_new(J_OBJ), *l = jv_new(J_ARR), *p, *as, *a;
		int k;
		for (k = 0; k < NPROMPTS; k++) {
			p = jv_new(J_OBJ);
			jv_set(p, "name", jv_str(prompts[k].name));
			jv_set(p, "description", jv_str(prompts[k].desc));
			as = jv_new(J_ARR);
			a = jv_new(J_OBJ);
			jv_set(a, "name", jv_str(prompts[k].arg));
			jv_set(a, "description", jv_str(prompts[k].argdesc));
			jv_set(a, "required", jv_bool(1));
			jv_push(as, a);
			jv_set(p, "arguments", as);
			jv_push(l, p);
		}
		jv_set(r, "prompts", l);
		rpc_result(id, r);
	} else if (!strcmp(method, "prompts/get"))
		prompts_get(id, params);
	else {
		char m[160];
		snprintf(m, sizeof m, "method not found: %.100s", method);
		rpc_error(id, -32601, m);
	}
}

int
mcp_main(int argc, char **argv, int i)
{
	char *line, err[160];
	int fmode = REP_FILES_BASE64;
	const char *p;

	for (i++; i < argc; i++) {
		if (!strcmp(argv[i], "--flat"))
			flat = 1;
		else if (!strcmp(argv[i], "--dir") && i + 1 < argc) {
			fmode = REP_FILES_WRITE | REP_FILES_BASE64;
			if (chdir(argv[++i]) != 0) {
				perror(argv[i]);
				return 2;
			}
		} else {
			fprintf(stderr, "usage: ucas --mcp [--flat] [--dir DIR]\n");
			return 2;
		}
	}
	if ((p = getenv("UCAS_MCP_PROTO")) != NULL)	/* after reset */
		snprintf(proto, sizeof proto, "%s", p);
	if (json_setup(argc, argv, fmode) != 0)
		return 2;
	rep_small_files = 1;
	json_respond_hook = answer;
	while ((line = json_read_line()) != NULL) {
		struct jv *msg;
		size_t pos;
		const char *s = line;
		while (*s == ' ' || *s == '\t' || *s == '\r')
			s++;
		if (!*s)
			continue;
		msg = jv_parse(line, err, sizeof err, &pos);
		if (msg == NULL) {
			rpc_error(NULL, -32700, "parse error");
			continue;
		}
		if (msg->t == J_ARR)	/* batches are not in MCP 2025-06-18 */
			rpc_error(NULL, -32600, "batches are not supported");
		else
			dispatch(msg);
		jv_free(msg);
	}
	return 0;
}
