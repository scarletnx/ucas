/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * jsonmode.c - the JSON frontend, for programs and AI agents.
 *
 *   ucas --json [--dir D] [--files M]   JSON Lines: a request per line on
 *                                       stdin, a response per line on stdout
 *   ucas --json -e '{...}'              one request
 *   ucas --json-schema                  the protocol as JSON Schema, with
 *                                       every command and its parameters
 *
 * A request names a command and its arguments:
 *
 *   {"id": 1, "cmd": "integrate", "args": {"f": "x^2 sin(x)", "var": "x"}}
 *
 * Each command is a template that turns the arguments into one ucas
 * statement (shown in the response as "ucas"), which runs with stdout
 * captured into "text".  The commands fill "result" through the report
 * API (report.h); fail() becomes "error", a question the text frontends
 * would ask (or a guess they would note) becomes an "ambiguous" error
 * that lists the readings, and files go into "files".  Nothing is read
 * from the terminal and no English is parsed unless asked for ("cmd":
 * "english").  Definitions persist across requests; "reset" restarts the
 * process image (exec), so nothing of the old state survives.
 *
 * "timeout_s" runs the request in a forked child: on success the child
 * carries on as the server (it has the new state), on a timeout or crash
 * it is killed and the state is as before.  The first such request makes
 * the original process a waiter that keeps the pid until every worker is
 * gone, so a client sees one process throughout.
 */
#include "ucas.h"
#include "json.h"
#include "jsonmode.h"
#include "img.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __FreeBSD__
#include <sys/sysctl.h>
#endif
#include <sys/mman.h>
#include <fcntl.h>

#define PROTOCOL_VERSION 1

/* ---- parameters and commands ---- */

enum {
	P_FUNC,		/* a function: a name, an expression with inline types
			 * ("isqrt(x: u4)"), or {"body", "params", "name"} */
	P_EXPR,		/* an expression: ucas text, or the AST form */
	P_EXPRS,	/* one or more expressions, joined with ", " */
	P_NAME,		/* an identifier */
	P_NAMES,	/* identifiers, joined with ", " */
	P_NAMES_SP,	/* identifiers, joined with " " */
	P_INT,
	P_NUM,		/* a number, or a constant expression as text */
	P_BOOL,		/* the fragment is added when true */
	P_ENUM,
	P_TEXT,		/* free option text, one line */
	P_INTERVAL,	/* [lo, hi] -> lo..hi */
	P_SUBS,		/* {"x": expr, ...} -> x = expr, ... */
	P_PARAMS,	/* define's parameter list */
	P_STRS,		/* strings */
	P_OBJ,		/* an object */
	P_SOURCE,	/* ucas source, several lines allowed */
	P_RING,		/* a ring: "Z/12", "GF(2^8)", "Z[i]" (with "modulus":
			 * GF's reduction polynomial, or alone: mod N) */
	P_RINGMOD,	/* the modulus: a number, "2^61 - 1", or for GF(q) a
			 * polynomial or a name (aes, crc32) */
	P_MATRIX,	/* [[1, 2], [3, 4]] (or the text) */
	P_VALUES,	/* [x0, x1, ...]: numbers or "1+2i" */
	P_CONGS,	/* [{"residue": 2, "modulus": 3}, ...] or "x = 2 mod 3" */
	P_INPUTS	/* animate exec's image inputs: [{"color": "black"},
			 * {"png_base64": ...}, "noise", ...] */
};

static const char *const ptype_name[] = {"function", "expression",
	"expressions", "name", "names", "names", "integer", "number",
	"boolean", "enum", "string", "interval", "substitutions",
	"parameters", "strings", "object", "source", "ring", "modulus",
	"matrix", "values", "congruences", "image inputs"};

struct jparam {
	const char *name;
	int type;
	int req;
	const char *frag;	/* printf-like, one %s for the value */
	const char *desc;
	const char *values;	/* P_ENUM: "a|b|c" */
};

#define MAXJP 20
struct jcmd {
	const char *name;
	const char *stmt;	/* the statement's leading words */
	int structured;		/* the result has fields beyond "text" */
	int special;		/* handled by its own function */
	const char *desc;
	struct jparam p[MAXJP];
};

#define F(n, fr, d) {n, P_FUNC, 1, fr, d, NULL}
#define FO(n, fr, d) {n, P_FUNC, 0, fr, d, NULL}
#define E(n, fr, d) {n, P_EXPR, 1, fr, d, NULL}
#define EO(n, fr, d) {n, P_EXPR, 0, fr, d, NULL}
#define ES(n, fr, d) {n, P_EXPRS, 1, fr, d, NULL}
#define ESO(n, fr, d) {n, P_EXPRS, 0, fr, d, NULL}
#define NM(n, fr, d) {n, P_NAME, 1, fr, d, NULL}
#define NMO(n, fr, d) {n, P_NAME, 0, fr, d, NULL}
#define NSO(n, fr, d) {n, P_NAMES, 0, fr, d, NULL}
#define IO(n, fr, d) {n, P_INT, 0, fr, d, NULL}
#define IR(n, fr, d) {n, P_INT, 1, fr, d, NULL}
#define NUO(n, fr, d) {n, P_NUM, 0, fr, d, NULL}
#define BO(n, fr, d) {n, P_BOOL, 0, fr, d, NULL}
#define TO(n, fr, d) {n, P_TEXT, 0, fr, d, NULL}
#define IVO(n, fr, d) {n, P_INTERVAL, 0, fr, d, NULL}

#define NUR(n, fr, d) {n, P_NUM, 1, fr, d, NULL}
#define TX(n, fr, d) {n, P_TEXT, 1, fr, d, NULL}
#define TXO(n, fr, d) {n, P_TEXT, 0, fr, d, NULL}
#define RING(fr, d) {"ring", P_RING, 1, fr, d, NULL}
#define RINGO(fr, d) {"ring", P_RING, 0, fr, d, NULL}
#define RMOD {"modulus", P_RINGMOD, 0, "", "N (mod N); with a GF(q) ring " \
	"its polynomial or name (aes, crc32)", NULL}
#define FN F("f", " %s", "the function: a defined name, an expression " \
	"(inline types allowed: \"isqrt(x: u4)\"), or {\"body\", \"params\"}")
#define EXF E("f", " %s", "the expression")
#define VAR NMO("var", " %s", "the variable (default: the only one)")

static const struct jcmd cmds[] = {
	/* session */
	{"define", NULL, 1, 1, "define a function: name(params) = body "
	    "[where ...]; every parameter needs a type (or a range)",
	    {NM("name", "", "the function's name"),
	    {"params", P_PARAMS, 0, "", "[{\"name\", \"type\", \"range\": "
	    "[lo, hi], \"uniform\"}]; omitted: the body's free variables in "
	    "order of appearance (an ambiguity: free_vars)", NULL},
	    E("body", "", "the body"),
	    {"where", P_STRS, 0, "", "where-clauses: bindings (\"y = x + 1\") "
	    "and conditions (\"x != 0\")", NULL}}},
	{"state", NULL, 1, 1, "the session: functions, models, bases, "
	    "assumptions", {{0}}},
	{"reset", NULL, 1, 1, "drop every definition (the process image is "
	    "restarted)", {{0}}},
	{"assume", NULL, 1, 1, "session-wide answers to ambiguities "
	    "(merged under each request's args.assume)",
	    {{"set", P_OBJ, 0, "", "{key: reading}", NULL},
	    BO("clear", "", "drop the session's assumptions first")}},
	{"schema", NULL, 1, 1, "this interface as JSON Schema", {{0}}},
	{"ucas", NULL, 0, 1, "raw ucas statements (the escape hatch)",
	    {{"source", P_SOURCE, 1, "", "ucas source; several lines allowed",
	    NULL}, {"code", P_SOURCE, 0, "", "a code block for the statement "
	    "(SPIR-V assembly or GLSL)", NULL}}},
	{"english", NULL, 1, 1, "translate English (opt-in) and run it; the "
	    "translated statements are returned",
	    {{"text", P_TEXT, 1, "", "one English request", NULL},
	    {"answers", P_STRS, 0, "", "replies to follow-up questions "
	    "(needs_input), in order", NULL},
	    {"code", P_SOURCE, 0, "", "a code block the request is about "
	    "(\"SAT simplify\" on SPIR-V, GLSL or C)", NULL},
	    {"lang", P_ENUM, 0, "", "the code block's language (by default "
	    "from the code)", "spv|glsl|c"}}},
	{"help", "help", 0, 0, "the command summary", {{0}}},
	{"funcs", "funcs", 0, 0, "every defined function", {{0}}},
	{"models", "models", 0, 0, "the cost models", {{0}}},
	{"use", "use", 0, 0, "make a cost model active",
	    {NM("model", " %s", "the model")}},
	/* look */
	{"show", "show", 1, 0, "the function, typed", {FN}},
	{"table", "table", 1, 0, "values at the points of the domain",
	    {FN, IO("max", " max %s", "at most this many rows (64)")}},
	{"parse", "parse", 1, 0, "the expression fully parenthesised",
	    {E("expr", " %s", "the expression")}},
	{"range", "range", 1, 0, "min/max over the domain",
	    {FN, BO("nocrit", " nocrit", "grid only, no refinement at "
	    "critical points")}},
	{"domain", "domain", 1, 0, "the conditions of the partial operations "
	    "and the inputs where F is defined", {FN}},
	{"wrap", "wrap", 1, 0, "nuw/nsw per operation, C UB, end-to-end "
	    "agreement with the mathematical result", {FN}},
	{"crit", "crit", 1, 0, "critical points, extrema, inflection points",
	    {FN}},
	/* calculus */
	{"diff", "diff", 1, 0, "derivative (higher, partial, mixed)",
	    {FN, {"vars", P_NAMES_SP, 0, " %s", "variables, in order (x y: "
	    "d2/dydx)", NULL}, IO("order", " %s", "the order")}},
	{"grad", "grad", 1, 0, "the gradient", {FN, ESO("at", " at %s",
	    "a point")}},
	{"hessian", "hessian", 1, 0, "the Hessian (and at a point the "
	    "second-derivative test)", {FN, ESO("at", " at %s", "a point")}},
	{"laplacian", "laplacian", 1, 0, "the Laplacian", {FN, ESO("at",
	    " at %s", "a point")}},
	{"jacobian", "jacobian", 1, 0, "the Jacobian matrix",
	    {ES("fs", " %s", "the functions"), NSO("wrt", " wrt %s",
	    "the variables"), ESO("at", " at %s", "a point")}},
	{"wronskian", "wronskian", 1, 0, "the Wronskian and linear "
	    "independence", {ES("fs", " %s", "the functions"),
	    NMO("var", " wrt %s", "the variable")}},
	{"limit", NULL, 1, 1, "a limit, symbolically and numerically",
	    {FN, NMO("var", "", "the variable (needed with several)"),
	    E("to", "", "the point: a number, inf or -inf"),
	    {"side", P_ENUM, 0, "", "one-sided: + (from the right) or -",
	    "+|-"}}},
	{"taylor", "taylor", 1, 0, "Taylor polynomial, checked",
	    {FN, VAR, NUO("at", " at %s", "expansion point (0)"),
	    IO("order", " order %s", "the order")}},
	{"implicit", "implicit", 1, 0, "dy/dx of an implicit curve",
	    {E("equation", " %s", "L = R"), NMO("for", " for %s", "y"),
	    NMO("wrt", " wrt %s", "x"), IO("order", " order %s", "1 or 2")}},
	{"integrate", "integrate", 1, 0, "antiderivative (checked by "
	    "differentiating back) or definite integral",
	    {FN, NMO("var", " wrt %s", "the variable of integration"),
	    {"from", P_NUM, 0, " from %s", "lower bound (with to)", NULL},
	    {"to", P_NUM, 0, " to %s", "upper bound (with from)", NULL},
	    TO("around", " around %s", "a contour for a complex integrand: "
	    "\"|z| = 1\""),
	    BO("qmc", " numerically qmc", "a quasi-Monte Carlo estimate on "
	    "Sobol points: over from..to, or without them over the box of "
	    "every parameter's range"),
	    IO("points", " points %s", "qmc: points a replicate, a power of "
	    "two (4096)"), IO("replicates", " replicates %s", "qmc: "
	    "randomised replicates (8)"), IO("seed", " seed %s", "qmc: the "
	    "random seed (UCAS_SEED, else 1)"), BO("owen", " owen", "qmc: a "
	    "nested uniform (Owen) scrambling of each replicate, not a digital "
	    "shift")}},
	{"antiderivative", "antiderivative", 1, 0, "an antiderivative + C",
	    {FN, NMO("var", " wrt %s", "the variable")}},
	/* compare, search */
	{"equiv", "equiv", 1, 0, "equal on F's domain? (with a counterexample)",
	    {FN, F("g", " %s", "the other function")}},
	{"find", "find", 1, 0, "cheaper implementations under the cost model",
	    {FN, IO("n", " %s", "how many (5)"), IO("max", " max %s",
	    "maximum cost")}},
	{"texalu", "texalu", 1, 0, "table reads and weighted sums of "
	    "neighbouring entries as reads through the texture unit, where the "
	    "model prices them lower, with the pipeline change they need",
	    {FN, NUO("tol", " tol %s", "the error allowed on a filtered sum "
	    "(the sum's own rounding)"), NMO("model", " under %s",
	    "the model (the active one)")}},
	{"texenc", "texenc", 1, 0, "a constant table as BC4 blocks (two of "
	    "one shape as BC5): endpoints, selectors and bytes, exact or within "
	    "a tolerance",
	    {E("table", " %s", "the table: [t0, t1, ...] or its name"),
	    EO("green", ", %s", "a second table of the same shape: the two as "
	    "BC5, the first red and this one green"),
	    IO("rows", " rows %s", "texels in a row (4)"), NUO("tol", " tol %s",
	    "the error allowed on an entry")}},
	{"more", "more", 0, 0, "the next results of the last find",
	    {IO("n", " %s", "how many")}},
	{"stoke", "stoke", 1, 0, "MCMC (Metropolis-Hastings) over straight-"
	    "line integer programs priced by the model; what reaches distance "
	    "0 is verified on the whole domain",
	    {FN, IO("steps", " steps %s", "proposals a chain (200000)"),
	    IO("restarts", " restarts %s", "chains (4)"),
	    IO("len", " len %s", "instructions at most (8)"),
	    IO("seed", " seed %s", "the random seed (UCAS_SEED, else 1)"),
	    NMO("model", " model %s", "the model (the active one)")}},
	{"pit", "pit", 1, 0, "polynomial identity testing (Schwartz-Zippel): "
	    "a probabilistic answer with its error bound",
	    {FN, F("g", " %s", "the other polynomial"),
	    IO("trials", " trials %s", "random points (4)"),
	    IO("seed", " seed %s", "the random seed (UCAS_SEED, else 1)")}},
	{"phash", "phash", 1, 0, "a switch's keys hashed injectively into a "
	    "table, when the model prices the lookup below the compare chain",
	    {FN, IO("seed", " seed %s", "the random seed (UCAS_SEED, else 1)"),
	    NMO("model", " model %s", "the model (the active one)")}},
	{"fit", "fit", 1, 0, "choose G's extra parameters to approximate F",
	    {F("g", " %s", "the approximation, with extra parameters"),
	    F("f", " ~ %s", "the target"), {"metric", P_ENUM, 0, " metric %s",
	    "the error to minimise", "maxrel|maxabs|rms"}}},
	{"error", "error", 1, 0, "error of G as an approximation of F",
	    {F("g", " %s", "the approximation"), F("f", " ~ %s",
	    "the target")}},
	{"cost", "cost", 1, 0, "price under a cost model",
	    {FN, NMO("model", " model %s", "the model (the active one)"),
	    NMO("consecutive", " consecutive %s", "an integer parameter whose "
	    "values run along the lanes of a warp (a branch's expected cost "
	    "is then counted over those warps)")}},
	{"dword", "dword", 1, 0, "f64 arithmetic over f32 (alone, or as "
	    "double-word pairs) where the model prices that lower, with a "
	    "derived error bound and its conditions",
	    {FN, NUO("tol", " tol %s", "relative error allowed"),
	    NUO("atol", " atol %s", "absolute error allowed"),
	    NUO("ulp", " ulp %s", "for a final f32: ulp beyond its own rounding"),
	    NMO("model", " model %s", "the model (the active one)")}},
	{"foldproof", "foldproof", 1, 0, "the float round trips the model's "
	    "driver folds, and their rounding as integer arithmetic it cannot "
	    "fold, proved equal",
	    {FN, NMO("model", " model %s", "the model (the active one)")}},
	{"rop", "rop", 1, 0, "graphics fixed-function states (blending, logic "
	    "ops, stencil) that compute a per-cell update F(s, d), each proved "
	    "and priced",
	    {FN, NMO("model", " model %s", "the model (the active one)")}},
	{"banks", "banks", 1, 0, "the bank-conflict degree of shared-memory "
	    "index expressions over a warp, and a proved index swizzle that "
	    "lowers it when the model prices conflicts",
	    {ES("accesses", " %s", "the index expressions (functions of a "
	    "lane parameter and uniform ones)"), IO("size", " size %s",
	    "the array's length"), IO("elem", " elem %s", "bytes per element "
	    "(4)"), NMO("model", " model %s", "the model (the active one)")}},
	{"intatomic", "intatomic", 1, 0, "a float atomic add as an integer "
	    "atomic add on fixed point, when its values are proved exact "
	    "there (or within tol) and the model prices it lower",
	    {FN, IO("count", " count %s", "adds into one cell at most (1024)"),
	    NUO("tol", " tol %s", "absolute error allowed on the total"),
	    BO("global", " global", "a device-scope atomic (shared: "
	    "workgroup)"), NMO("model", " model %s", "the model (the active "
	    "one)")}},
	{"gates", "gates", 1, 0, "minimal gate network, with proof",
	    {FN, NMO("basis", " basis %s", "c, a64, a64sha, x86bmi, nand ..."),
	    IO("max", " max %s", "maximum cost (12)"), IO("nodes",
	    " nodes %s", "search budget"), BO("as_c", " as c",
	    "also a C function")}},
	{"approx", "approx", 1, 0, "polynomial/rational approximations, "
	    "measured", {FN, IVO("in", " in %s", "the interval"),
	    TO("options", " %s", "approx options as in the CLI: \"degree 5 "
	    "as f64 keep endpoints\"")}},
	{"approx_more", "approx more", 1, 0, "more candidates of the last "
	    "approx", {IO("n", " %s", "how many")}},
	/* loops */
	{"closed", "closed", 1, 0, "closed forms of sums, products, loops",
	    {FN}},
	{"scheme", "scheme", 0, 0, "evaluation schemes priced",
	    {FN, BO("more", " more", "a polynomial's forms beyond Horner "
	    "and Estrin too: Knuth, Paterson-Stockmeyer, compensated Horner"),
	    NMO("model", " model %s", "the model")}},
	{"continuous", "continuous", 0, 0, "sums as integrals, loops as ODEs",
	    {FN}},
	{"dsolve", "dsolve", 1, 0, "solve a defined ODE (symbolic or numeric)",
	    {NM("name", " %s", "the ODE's function"), NUO("from", " from %s",
	    "start"), NUO("to", " to %s", "end"), IO("steps", " steps %s",
	    "steps to print")}},
	/* complex and trig */
	{"complex", "complex", 1, 0, "as a + bi", {FN}},
	{"re", "re", 1, 0, "the real part", {FN}},
	{"im", "im", 1, 0, "the imaginary part", {FN}},
	{"conj", "conj", 1, 0, "the conjugate", {FN}},
	{"modulus", "modulus", 1, 0, "the modulus", {FN}},
	{"arg", "arg", 1, 0, "the argument", {FN}},
	{"polar", "polar", 1, 0, "r e^(i theta)", {FN}},
	{"trig", "trig", 1, 0, "equivalent trig forms",
	    {FN, {"mode", P_ENUM, 0, " %s", "one kind of form", "simplify|"
	    "sincos|exp|reduce|product|sum|expand|harmonic|half|weierstrass|"
	    "inverse|double|all"}}},
	{"nthroots", "nthroots", 1, 0, "the N complex N-th roots",
	    {IR("n", " %s", "N"), FN}},
	{"unity", "unity", 1, 0, "the N-th roots of unity",
	    {IR("n", " %s", "N")}},
	{"holomorphic", "holomorphic", 1, 0, "Cauchy-Riemann; the complex "
	    "derivative", {FN}},
	{"residue", "residue", 1, 0, "the residue at a point",
	    {FN, E("at", " at %s", "the point")}},
	{"poles", "poles", 1, 0, "the poles and their orders", {FN}},
	/* properties */
	{"monotonic", "monotonic", 1, 0, "increasing/decreasing, intervals",
	    {FN, VAR, IVO("in", " in %s", "an interval")}},
	{"period", "period", 1, 0, "the fundamental period", {FN, VAR}},
	{"identify", "identify", 1, 0, "classify an expression, equation, "
	    "conic or ODE", {E("f", " %s", "an expression, an equation L = "
	    "R, or a name")}},
	{"parity", "parity", 1, 0, "even/odd, symmetries", {FN}},
	{"bounded", "bounded", 1, 0, "bounds, poles, growth", {FN}},
	{"zeros", "zeros", 1, 0, "the zeros (exact for polynomials)", {FN}},
	{"injective", "injective", 1, 0, "one to one? (with a witness)",
	    {FN}},
	{"surjective", "surjective", 1, 0, "onto? (with a value never taken)",
	    {FN, IVO("onto", " onto %s", "the codomain")}},
	{"bijective", "bijective", 1, 0, "a bijection?", {FN}},
	{"continuity", "continuity", 1, 0, "discontinuities by kind", {FN}},
	{"convexity", "convexity", 1, 0, "convex/concave intervals", {FN}},
	{"asymptotes", "asymptotes", 1, 0, "vertical, horizontal, oblique",
	    {FN}},
	/* algebra */
	{"expand", "expand", 1, 0, "expanded over Q", {EXF}},
	{"factor", "factor", 1, 0, "factored over Q", {EXF}},
	{"simplify", "simplify", 1, 0, "the simplest normal form", {EXF}},
	{"together", "together", 1, 0, "one fraction", {EXF}},
	{"cancel", "cancel", 1, 0, "common factors cancelled", {EXF}},
	{"collect", "collect", 1, 0, "collected in a variable",
	    {EXF, NMO("var", " in %s", "the variable")}},
	{"apart", "apart", 1, 0, "partial fractions",
	    {EXF, NMO("var", " in %s", "the variable")}},
	{"gcd", "gcd", 1, 0, "polynomial gcd", {E("p", " %s", "P"),
	    E("q", ", %s", "Q")}},
	{"lcm", "lcm", 1, 0, "polynomial lcm", {E("p", " %s", "P"),
	    E("q", ", %s", "Q")}},
	{"resultant", "resultant", 1, 0, "the resultant", {E("p", " %s", "P"),
	    E("q", ", %s", "Q"), NMO("var", " in %s", "the variable")}},
	{"discriminant", "discriminant", 1, 0, "the discriminant",
	    {E("p", " %s", "P"), NMO("var", " in %s", "the variable")}},
	{"quo", "quo", 1, 0, "polynomial quotient", {E("p", " %s", "P"),
	    E("q", ", %s", "Q"), NMO("var", " in %s", "the variable")}},
	{"rem", "rem", 1, 0, "polynomial remainder", {E("p", " %s", "P"),
	    E("q", ", %s", "Q"), NMO("var", " in %s", "the variable")}},
	{"divide", "divide", 1, 0, "division with remainder",
	    {E("p", " %s", "P"), E("q", " by %s", "Q"), NMO("var", " in %s",
	    "the variable")}},
	{"solve", "solve", 1, 0, "equations and systems, every solution "
	    "substituted back", {ES("equations", " %s", "L = R (or F, "
	    "meaning F = 0)"), NSO("for", " for %s", "the unknowns")}},
	{"substitute", "substitute", 1, 0, "F at values",
	    {{"values", P_SUBS, 1, " %s", "{\"x\": expr, ...}", NULL},
	    E("f", " in %s", "the expression")}},
	/* pictures */
	{"plot", "plot", 1, 0, "a GIF plot (files)", {ES("fs", " %s",
	    "functions, an equation, a region, a curve ..."),
	    TO("options", " %s", "plot options as in the CLI: \"x in -1..1 "
	    "deriv logy\""), NMO("to", " to %s", "the file's name (plot)"),
	    BO("nofile", " nofile", "a checksum instead of the file")}},
	{"plot3d", "plot3d", 1, 0, "a surface (files)", {FN,
	    TO("options", " %s", "plot options"), NMO("to", " to %s", "name"),
	    BO("nofile", " nofile", "a checksum instead of the file")}},
	{"neighbourhood", "neighbourhood", 1, 0, "a plot around the "
	    "interesting part (files)", {FN, NUO("at", " at %s", "the point"),
	    BO("nofile", " nofile", "a checksum instead of the file")}},
	{"animate", "animate", 0, 0, "an animated GIF: scans, lockstep "
	    "execution, a shader over images (files)", {TO("spec", " %s",
	    "as in the CLI: \"hillis scan n 8\", \"exec of 32-bit x*x + 1\"; "
	    "with code, \"exec\" (the default)"),
	    {"code", P_SOURCE, 0, "", "SPIR-V assembly or GLSL for animate "
	    "exec", NULL},
	    {"inputs", P_INPUTS, 0, "%s", "the shader's image inputs, in "
	    "(set, binding) order: each a colour name, #rrggbb, \"noise\", "
	    "\"gradient\", \"checker\", or {\"color\" | \"pattern\" | "
	    "\"png_base64\" (any PNG/JPEG/GIF/BMP/PPM, base64) | \"file\", "
	    "\"format\": \"R8\", \"width\", \"height\", \"filter\": "
	    "\"nearest\", \"address\": \"clamp\"}; inputs not given are "
	    "noise", NULL},
	    {"output_format", P_TEXT, 0, " output %s", "the colour target's "
	    "format: RGBA8 (default), RGB8, RG8, R8, SRGB8_A8, RGBA16F, R32F, "
	    "RGBA32F ...", NULL},
	    TO("size", " size %s", "the image run's size, \"128x128\" (the "
	    "first input's, at most 256x256)"),
	    TO("sampler", " sampler %s", "the sampler state SPIR-V does not "
	    "carry: \"nearest clamp\", \"linear mirror\", \"border white\" "
	    "(linear repeat)"),
	    IO("seed", " seed %s", "the random inputs' seed"),
	    IO("lanes", " lanes %s", "4 (a 2x2 quad), 8, 16 or 32"),
	    NMO("to", " to %s", "the file's name (exec)"),
	    BO("nofile", " nofile", "a checksum instead of the file")}},
	{"symmetry", "symmetry", 1, 0, "axes x = c, centres, f(x, y) = f(y, "
	    "x)", {FN}},
	/* rings and finite fields (ring.c) */
	{"ring", "ring", 1, 0, "a ring's structure (units, zero divisors, "
	    "ideals ...), an element of it, or a value computed in it",
	    {RING(" %s", "Z/12, GF(2^8), GF(2)[x]/(x^4 + 1), Z[i], Q[x]/(x^2 "
	    "- 2) ..."), RMOD,
	    TO("what", " %s", "units | zero divisors | idempotents | table, an "
	    "element (\"4\", \"0x53\"), or an expression (\"0x57 * 0x83\")")}},
	{"inverse_mod", "inverse", 1, 0, "a multiplicative inverse (of a "
	    "number, an element, a matrix)", {TX("a", " %s", "the element "
	    "(\"3\", \"0x53\", \"[[1,2],[3,5]]\")"), RING(" in %s", "the ring"),
	    RMOD}},
	{"order_mod", "order", 1, 0, "multiplicative order (of a residue, an "
	    "element, a polynomial over GF)", {TX("a", " %s", "the element"),
	    RING(" over %s", "the ring"), RMOD}},
	{"phi", "phi", 1, 0, "Euler's totient", {NUR("n", " %s", "n")}},
	{"carmichael", "carmichael", 1, 0, "Carmichael's lambda",
	    {NUR("n", " %s", "n")}},
	{"factorint", "factorint", 1, 0, "integer factorisation (Pollard-Brent, "
	    "Miller-Rabin)", {NUR("n", " %s", "n")}},
	{"legendre", "legendre", 1, 0, "the Legendre symbol (a/p)",
	    {NUR("a", " %s", "a"), NUR("p", " %s", "p (an odd prime)")}},
	{"jacobi", "jacobi", 1, 0, "the Jacobi symbol (a/n)",
	    {NUR("a", " %s", "a"), NUR("n", " %s", "n (odd)")}},
	{"sqrtmod", "sqrtmod", 1, 0, "square roots mod n (Tonelli-Shanks, Hensel, "
	    "CRT)", {NUR("a", " %s", "a"), RING(" in %s", "the ring"), RMOD}},
	{"qr", "qr", 1, 0, "the quadratic residues mod n", {NUR("n", " %s",
	    "n")}},
	{"primroot", "primroot", 1, 0, "the smallest primitive root mod n, or "
	    "whether a is one", {NUO("a", " %s", "a candidate (with modulus)"),
	    NUO("n", " %s", "n (alone)"), RINGO(" in %s", "the ring"), RMOD}},
	{"dlog", "dlog", 1, 0, "the discrete logarithm", {TX("a", " %s",
	    "the value"), TXO("base", " base %s", "the base (default a "
	    "primitive root)"), RING(" in %s", "the ring"), RMOD}},
	{"crt", "crt", 1, 0, "the Chinese remainder theorem", {{"congruences",
	    P_CONGS, 1, " %s", "[{\"residue\": 2, \"modulus\": 3}, ...]",
	    NULL}}},
	{"congruence", "congruence", 1, 0, "a linear or polynomial congruence",
	    {TX("equation", " %s", "\"3x = 4\", \"x^2 + x + 1 = 0\""),
	    RING(" in %s", "the ring"), RMOD}},
	{"magic", "magic", 1, 0, "the multiply-shift for dividing by a constant",
	    {IR("divisor", " %s", "the divisor"), {"type", P_ENUM, 1, " %s",
	    "the integer type", "u8|u16|u32|u64|i8|i16|i32|i64"}}},
	{"ring_factor", "factor", 1, 0, "factor a polynomial over a finite "
	    "field or Z/n (or an element of Z[i])", {TX("p", " %s",
	    "the polynomial (or crc16, crc32 ...)"), RING(" over %s",
	    "GF(q), Z[i]"), RMOD}},
	{"irreducible", "irreducible", 1, 0, "Rabin's irreducibility test",
	    {TX("p", " %s", "the polynomial"), RING(" over %s", "GF(q)"), RMOD}},
	{"primitive", "primitive", 1, 0, "is the polynomial primitive",
	    {TX("p", " %s", "the polynomial"), RING(" over %s", "GF(q)"), RMOD}},
	{"ring_roots", "roots", 1, 0, "roots of a polynomial in Z/n or GF(q)",
	    {TX("p", " %s", "the polynomial"), RING(" over %s", "the ring"),
	    RMOD}},
	{"ring_poly", "poly", 1, 0, "a polynomial expression reduced over a "
	    "ring", {TX("p", " %s", "the expression"), RING(" over %s",
	    "the ring"), RMOD}},
	{"ring_gcd", "gcd", 1, 0, "polynomial (or Gaussian integer) gcd over a "
	    "ring", {TX("p", " %s", "P"), TX("q", ", %s", "Q"), RING(" over %s",
	    "the ring"), RMOD}},
	{"ring_norm", "norm", 1, 0, "the norm of a quadratic integer",
	    {TX("a", " %s", "the element"), RING(" in %s", "Z[i], Z[sqrt(2)]"),
	    RMOD}},
	{"matrix", NULL, 1, 1, "a matrix over Z/n or GF(q): its properties, "
	    "det, rank, nullspace, inverse, or a solve", {{"matrix", P_MATRIX, 1,
	    " %s", "[[a, b], [c, d]]", NULL}, {"op", P_ENUM, 0, "", "what "
	    "(default matrix: the summary)", "matrix|det|rank|nullspace|inverse|"
	    "solve"}, {"rhs", P_VALUES, 0, "", "solve: M x = rhs", NULL},
	    RING(" in %s", "the ring"), RMOD}},
	{"gf2matrix", "gf2matrix", 1, 0, "a GF(2)-linear bit map as a matrix",
	    {E("f", " %s", "the map of x (x ^ (x << 3))"), {"type", P_ENUM, 1,
	    " %s", "the word", "u8|u16|u32|u64"}}},
	/* transforms, series and sums (xform.c, series.c) */
	{"laplace", "laplace", 1, 0, "the one-sided Laplace transform, with "
	    "its region of convergence", {EXF, NMO("var", " %s", "t"),
	    NMO("to_var", " %s", "s"), NUO("period", " period %s", "the period "
	    "of a periodic f")}},
	{"ilaplace", "ilaplace", 1, 0, "the inverse Laplace transform",
	    {EXF, NMO("var", " %s", "s"), NMO("to_var", " %s", "t")}},
	{"laplace_solve", "laplace solve", 1, 0, "a linear ODE with constant "
	    "coefficients through the Laplace transform", {ES("equations", " %s",
	    "the ODE and its initial values: [\"y'' + y = sin(t)\", \"y(0) = 0\","
	    " \"y'(0) = 1\"]")}},
	{"ztrans", "ztrans", 1, 0, "the one-sided Z transform, with its ROC",
	    {EXF, NMO("var", " %s", "n"), NMO("to_var", " %s", "z")}},
	{"iztrans", "iztrans", 1, 0, "the inverse Z transform (causal)",
	    {EXF, NMO("var", " %s", "z"), NMO("to_var", " %s", "n")}},
	{"zsolve", "zsolve", 1, 0, "a linear recurrence through the Z "
	    "transform", {ES("equations", " %s", "[\"y[n+2] = y[n+1] + y[n]\", "
	    "\"y[0] = 0\", \"y[1] = 1\"]")}},
	{"tf", "tf", 1, 0, "a transfer function: poles, zeros, stability, "
	    "impulse response", {E("f", " %s", "a difference equation, an ODE, "
	    "or H(z)/H(s)")}},
	{"fourier", "fourier", 1, 0, "the Fourier transform", {EXF,
	    NMO("var", " %s", "x"), NMO("to_var", " %s", "w")}},
	{"ifourier", "ifourier", 1, 0, "the inverse Fourier transform", {EXF,
	    NMO("var", " %s", "w"), NMO("to_var", " %s", "x")}},
	{"fourier_series", "fourier series", 1, 0, "a Fourier series", {EXF,
	    NMO("var", " %s", "x"), IVO("on", " on %s", "the interval ([-pi, "
	    "pi])"), NUO("period", " period %s", "the period"), IO("terms",
	    " terms %s", "terms shown")}},
	{"dft", "dft", 1, 0, "the discrete Fourier transform (FFT for powers "
	    "of two)", {{"values", P_VALUES, 1, " %s", "[x0, x1, ...] (numbers "
	    "or \"1+2i\")", NULL}}},
	{"idft", "idft", 1, 0, "the inverse DFT", {{"values", P_VALUES, 1,
	    " %s", "[X0, X1, ...]", NULL}}},
	{"dtft", "dtft", 1, 0, "the discrete-time Fourier transform",
	    {{"values", P_VALUES, 0, " %s", "a finite sequence", NULL},
	    EO("f", " %s", "or a sequence x[n]")}},
	{"genfunc", "genfunc", 1, 0, "the ordinary generating function",
	    {TX("a", " %s", "a_n, a recurrence or a name (fibonacci, catalan)"),
	    NMO("var", " %s", "n")}},
	{"egf", "egf", 1, 0, "the exponential generating function",
	    {TX("a", " %s", "a_n or a name")}},
	{"infinite_sum", NULL, 1, 1, "an infinite sum: convergence test, "
	    "closed form, numeric value", {NM("index", "", "n"), NUO("from", "",
	    "the first index (1)"), E("f", "", "the term")}},
	{"infinite_product", NULL, 1, 1, "an infinite product", {NM("index", "",
	    "n"), NUO("from", "", "the first index (1)"), E("f", "",
	    "the factor")}},
	{"converge", "converge", 1, 0, "convergence tests of sum a_n",
	    {E("a", " %s", "the term a_n (in n)")}},
	{"radius", "radius", 1, 0, "the radius and interval of convergence of a "
	    "power series", {E("a", " %s", "the term, x^n/n!")}},
	{"powerseries", "powerseries", 1, 0, "the general term of a power "
	    "series", {EXF, NMO("var", " %s", "x"), NUO("at", " at %s",
	    "the centre")}},
	{"asymptotic", "asymptotic", 1, 0, "Euler-Maclaurin asymptotics of a "
	    "partial sum", {E("sum", " %s", "sum(k, 1, n, f)")}},
	/* integrals and solids of revolution (viz.c) */
	{"area", "area", 1, 0, "the area under or between curves (exact and "
	    "numeric; a GIF)", {E("f", " %s", "F, or \"F and G\""),
	    NUO("from", " from %s", "a"), NUO("to", " to %s", "b"),
	    TO("options", " %s", "riemann left 8, accumulate ..."),
	    BO("nofile", " nofile", "a checksum instead of the GIF")}},
	{"revolve", "revolve", 1, 0, "the solid of revolution: volume, surface, "
	    "centroid, Pappus (a GIF)", {E("f", " %s", "F, or \"F and G\""),
	    NUO("from", " from %s", "a"), NUO("to", " to %s", "b"),
	    TO("about", " about %s", "x, y, x = 2, y = -1"), {"method", P_ENUM, 0,
	    " %s", "the slicing", "disk|washer|shell"}, TO("options", " %s",
	    "still, rotate ..."), BO("nofile", " nofile", "no GIF")}},
	{"volume", "volume", 1, 0, "the volume of the solid of revolution",
	    {E("f", " %s", "F"), NUO("from", " from %s", "a"), NUO("to",
	    " to %s", "b"), TO("about", " about %s", "the axis"),
	    {"method", P_ENUM, 0, " %s", "the slicing", "disk|washer|shell"},
	    BO("nofile", " nofile", "no GIF")}},
	{"surfacearea", "surfacearea", 1, 0, "the lateral surface area of the "
	    "solid", {E("f", " %s", "F"), NUO("from", " from %s", "a"),
	    NUO("to", " to %s", "b"), TO("about", " about %s", "the axis"),
	    BO("nofile", " nofile", "no GIF")}},
	{"arclength", "arclength", 1, 0, "the arc length of a curve",
	    {E("f", " %s", "F"), NUO("from", " from %s", "a"), NUO("to",
	    " to %s", "b"), BO("nofile", " nofile", "no GIF")}},
	{"centroid", "centroid", 1, 0, "the centroid of the region under or "
	    "between curves", {E("f", " %s", "F, or \"F and G\""),
	    NUO("from", " from %s", "a"), NUO("to", " to %s", "b"),
	    BO("nofile", " nofile", "no GIF")}},
	/* hypercomplex algebras (hyper.c) */
	{"multiplication_table", "table", 1, 0, "an algebra's multiplication "
	    "table (and the Fano plane)", {{"algebra", P_TEXT, 1, " %s",
	    "complex | quaternion | octonion | sedenion | cd N", NULL}}},
	{"algebra", "algebra", 1, 0, "an algebra's properties with "
	    "counterexamples", {{"algebra", P_TEXT, 1, " %s", "complex | "
	    "quaternion | octonion | sedenion | cd N", NULL}, TO("property",
	    " %s", "one property: associative, alternative ...")}},
	{"zerodivisors", "zerodivisors", 1, 0, "a pair of zero divisors",
	    {TX("algebra", " %s", "sedenion | cd N (from the sedenions on)")}},
	{"associator", "associator", 1, 0, "(ab)c - a(bc)", {E("a", " %s", "a"),
	    E("b", ", %s", "b"), E("c", ", %s", "c")}},
	{"commutator", "commutator", 1, 0, "ab - ba", {E("a", " %s", "a"),
	    E("b", ", %s", "b")}},
	{"rotate", "rotate", 1, 0, "rotate a vector by a quaternion or an "
	    "angle about an axis", {TX("v", " %s", "(1, 0, 0)"), TX("by",
	    " by %s", "a quaternion, or an angle (90 degrees, pi/2)"),
	    TO("about", " about %s", "the axis: (0, 0, 1), x, y, z")}},
	{"axisangle", "axisangle", 1, 0, "axis and angle to a unit quaternion, "
	    "or back", {TX("q", " %s", "\"(0, 0, 1), pi/2\" or a quaternion")}},
	{"rotmatrix", "rotmatrix", 1, 0, "a quaternion's 3x3 rotation matrix",
	    {TX("q", " %s", "the quaternion")}},
	{"frommatrix", "frommatrix", 1, 0, "the quaternion of a rotation "
	    "matrix", {ES("matrix", " %s", "m11, ..., m33 (9 numbers, by rows)")}},
	{"slerp", "slerp", 1, 0, "spherical linear interpolation",
	    {TX("q1", " %s", "q1"), TX("q2", ", %s", "q2"), NUR("t", ", %s",
	    "t")}},
	{"spirv_simplify", NULL, 1, 1, "cheaper forms of a SPIR-V module's "
	    "integer instructions, each proved by SAT",
	    {{"code", P_SOURCE, 1, "", "SPIR-V assembly, GLSL, or C (run as "
	    "c_optimize)", NULL},
	    {"lang", P_ENUM, 0, "", "the code's language (by default from "
	    "the code; c: the C frontend, as c_optimize)", "spv|glsl|c"},
	    NUO("budget", "", "seconds (6)"),
	    {"ranges", P_OBJ, 0, "", "{\"n\": [lo, hi]}: values assumed for "
	    "an input", NULL},
	    BO("nofile", "", "a checksum instead of the rewritten module"),
	    BO("float_only", "", "only the float instructions"),
	    BO("exact", "", "float rewrites: bit-exact ones only"),
	    NUO("ulp", "", "float rewrites: within N ulp of the reference"),
	    BO("approx", "", "float rewrites: approximations too (with ulp)"),
	    BO("percomp", "", "float rewrites: per vector component"),
	    BO("advice", "", "also report what could leave the shader: values "
	    "of a draw's frequency, and in a fragment shader what is affine in "
	    "the varyings"),
	    {"sink", P_TEXT, 0, "", "float rewrites: the format the stored "
	    "and returned floats end in (unorm8, rgba8, r8g8b8a8_unorm, "
	    "rgb10a2, f16): one step of it is the error budget", NULL},
	    BO("functions", "", "GLSL: the function-level pass alone (true) or "
	    "not at all (false); both levels run by default"),
	    {"float_ranges", P_OBJ, 0, "", "{\"c\": [lo, hi]}: a float input's "
	    "range (closed)", NULL},
	    NMO("model", "", "the cost model to price and choose under (made "
	    "the active one; the dot products it offers are used)")}},
	/* cfront: C99 in, proved and x86-64-optimised C out (cf_*.c) */
	{"c_optimize", "c optimize", 1, 0, "C99 in, faster C out for an "
	    "x86-64 target: each rewrite proved (exhaustively, by SAT, by a "
	    "derivation or by induction) against the original under C99 LP64 "
	    "semantics with IEEE floats, unproved ones not applied; loops in "
	    "closed form, element-wise loop bodies, strlen scans, loops "
	    "vectorised with intrinsics (proved lane by lane)",
	    {{"code", P_SOURCE, 0, "", "C99 source (or file)", NULL},
	    TO("file", " %s", "a .c file instead of code"),
	    TO("target", " target %s", "x86-64-v3 (default), zen4, skylake, "
	    "alderlake, x86-64-v2 ... (c_targets)"),
	    NUO("budget", " budget %s", "seconds for proofs (10)"),
	    TO("range", " assume %s", "NAME LO HI: a contract on an input "
	    "(\"x 0 1000\")"),
	    BO("check", " check", "compile both with the host cc and compare "
	    "them on edge and random inputs (UB inputs skipped)"),
	    BO("bench", " bench", "check, and time both with rdtsc"),
	    BO("wrapv", " wrapv", "signed overflow wraps (do not exploit UB)"),
	    BO("nofile", " nofile", "print the code instead of a file"),
	    BO("timed", " timed", "say how many seconds the proofs took"),
	    BO("trace", " trace", "list every candidate and why it was not "
	    "taken"),
	    BO("vecstub", " vecstub", "a stub vectoriser that rebuilds each "
	    "counted loop from its IR (tests the loop extraction)"),
	    BO("vectorize", " vectorize", "apply the proved vectorised loops "
	    "(by default they are proposed; with bench, applied where they "
	    "beat clang's own code)"),
	    TO("engines", " engines %s", "which of ucas's engines generate "
	    "candidates: all (default), none, or a comma list of find gates "
	    "alg ring dio gf2 loops bound synth"),
	    NUO("steps", " steps %s", "sketches the synthesis may try per value "
	    "(a deterministic budget; default 1000)"),
	    NUO("ebudget", " ebudget %s", "CPU seconds for the engines' own "
	    "searches, in total (default: a quarter of budget)"),
	    BO("approx", " approx", "apply approximations (loops as integrals, "
	    "minimax) whose derived error bound is stated; by default they "
	    "are proposed"),
	    NUO("error", " error %s", "with approx: apply an approximation "
	    "only when its derived error bound is at most this")}},
	{"c_lower", "c lower", 1, 0, "each C function as one ucas value of "
	    "its parameters (with the C UB as a domain condition), or why not; "
	    "for the others, each counted loop in the vectoriser's IR",
	    {{"code", P_SOURCE, 0, "", "C99 source (or file)", NULL},
	    TO("file", " %s", "a .c file instead of code"),
	    BO("wrapv", " wrapv", "signed overflow wraps")}},
	{"c_targets", "c targets", 1, 0, "the x86-64 targets and their "
	    "instruction-set features", {{NULL, 0, 0, NULL, NULL, NULL}}},
	/* x86-64 (x86*.c, cfx86.c): cost models, mca, intrinsics */
	{"x86_models", "x86 models", 1, 0, "the x86 µarch cost models (skl "
	    "skx glc rpl zen4 x86v2 x86v3 x86v4)", {{NULL, 0, 0, NULL, NULL,
	    NULL}}},
	{"x86_model", "model", 1, 0, "one x86 µarch: features, ports, and per "
	    "instruction class latency, reciprocal throughput, uops and ports "
	    "at xmm/ymm/zmm (with where each number comes from)",
	    {TX("name", " %s", "skl skx glc rpl zen4 x86v2 x86v3 x86v4, or an "
	    "alias: skylake, znver4 ...")}},
	{"mca", "mca", 1, 0, "a cycle-level throughput estimate of a "
	    "function's x86 code (critical path, port pressure, front end; no "
	    "caches or mispredictions)", {F("f", " %s", "the function"),
	    TO("model", " on %s", "the µarch (default: the active x86 model, "
	    "else skl)")}},
	{"intrinsic", "intrinsic", 1, 0, "one x86 intrinsic: signature, "
	    "features, instruction class, exact lane semantics (as ucas "
	    "expressions) and costs per µarch", {TX("name", " %s",
	    "_mm256_add_epi32 ..."), TO("model", " on %s", "only this µarch")}},
	{"intrinsics", "intrinsics", 1, 0, "the intrinsics catalogue: count "
	    "per feature, or the names matching a filter (a substring, a "
	    "feature such as avx2, or a class)", {TO("filter", " %s",
	    "a substring, feature or class"), TO("model", " on %s",
	    "only those this µarch has")}},
	{"x86check", "x86check", 1, 0, "test the intrinsics' lane semantics "
	    "against the host C compiler's intrinsics (compiled and run on "
	    "random and edge inputs)", {TO("filter", " %s", "a substring or "
	    "feature"), IO("n", " n %s", "points per test (64)"),
	    BO("avx512", " avx512", "also the AVX-512 forms (lowered to run "
	    "on this host, and cross-checked)")}},
	/* abstract algebra (absalg_cmd.c): exact, every claim certified */
#define ZM_(n, fr, d) {n, P_MATRIX, 1, fr, d, NULL}
#define GRP(n, fr) TX(n, fr, "an abelian group: \"Z/4 + Z/6\", " \
	"\"Z^2 + Z/3\", \"(Z/2)^3\", \"0\", or \"coker [[2, 0], [0, 3]]\"")
	{"bigint", "bigint", 1, 0, "an exact integer expression of any size "
	    "(+ - * ^ !, / exact, div floor, % mod, gcd, lcm, abs, isprime, "
	    "binomial)", {TX("expr", " %s", "the expression: \"2^200 + 1\"")}},
	{"bigint_factor", "bigint factor", 1, 0, "prime factorization of an "
	    "integer of any size (Pollard-Brent rho; product and primality "
	    "checked)", {TX("n", " %s", "the integer or expression")}},
	{"snf", "snf", 1, 0, "Smith normal form U A V = D with U, V and their "
	    "inverses, verified; the cokernel", {ZM_("matrix", " %s",
	    "an integer matrix (big entries as strings: \"2^100\")")}},
	{"hnf", "hnf", 1, 0, "row Hermite normal form H = U A with U "
	    "unimodular, verified", {ZM_("matrix", " %s", "an integer "
	    "matrix")}},
	{"zmatrix", "zmat", 1, 0, "an integer matrix over Z: summary, det, "
	    "rank, kernel (saturated basis), image (lattice basis), inverse "
	    "over Z, cokernel group, or solve A x = b over Z",
	    {{"op", P_ENUM, 0, " %s", "what (default: a summary)", "det|rank|"
	    "kernel|image|inverse|cokernel|solve"}, ZM_("matrix", " %s",
	    "the matrix"), {"rhs", P_VALUES, 0, ", %s", "solve: b", NULL}}},
	{"qmatrix", "qmat", 1, 0, "a matrix over Q (entries like \"1/2\"): "
	    "rref, det, rank, inverse, nullspace, solve", {{"op", P_ENUM, 0,
	    " %s", "what (default rref)", "rref|det|rank|inverse|nullspace|"
	    "solve"}, ZM_("matrix", " %s", "the matrix"), {"rhs", P_VALUES, 0,
	    ", %s", "solve: b", NULL}}},
	{"abgroup", "abgroup", 1, 0, "a finitely generated abelian group: "
	    "invariant factors, primary decomposition, order, rank, exponent, "
	    "torsion, generators", {GRP("group", " %s")}},
	{"abiso", "abiso", 1, 0, "are two abelian groups isomorphic; if so an "
	    "explicit isomorphism and its inverse, both checked",
	    {GRP("g", " %s"), GRP("h", ", %s")}},
	{"abhom", "abhom", 1, 0, "the group Hom(G, H)", {GRP("g", " %s"),
	    GRP("h", ", %s")}},
	{"abtensor", "abtensor", 1, 0, "the tensor product G (x) H",
	    {GRP("g", " %s"), GRP("h", ", %s")}},
	{"abtor", "abtor", 1, 0, "Tor(G, H) from a free resolution",
	    {GRP("g", " %s"), GRP("h", ", %s")}},
	{"abext", "abext", 1, 0, "Ext(G, H) from a free resolution",
	    {GRP("g", " %s"), GRP("h", ", %s")}},
	{"abmap", "abmap", 1, 0, "a homomorphism G -> H given by the images of "
	    "G's generators (columns, in H's generators): well defined?, "
	    "kernel, image, cokernel, injective, surjective, inverse",
	    {GRP("from", " %s"), GRP("to", " -> %s"), ZM_("matrix", " by %s",
	    "column j = the image of generator j")}},
	{"abelem", "abelem", 1, 0, "an element of an abelian group: canonical "
	    "form and order", {GRP("group", " %s"), {"element", P_VALUES, 1,
	    ", %s", "its coordinates in the group's generators", NULL}}},
	{"absub", "absub", 1, 0, "the subgroup generated by elements, its "
	    "quotient and index (membership witnesses checked)",
	    {GRP("group", " %s"), ZM_("generators", ", %s", "the generators, "
	    "one per row")}},
	{"abexact", "abexact", 1, 0, "is G1 -f-> G2 -g-> G3 exact at G2 (with "
	    "preimages of ker g as the certificate)", {GRP("g1", " %s"),
	    GRP("g2", " -> %s"), GRP("g3", " -> %s"), TX("f", " by %s", "f as "
	    "a matrix ([[...]]; \"[]\" from 0)"), TX("g", ", %s", "g as a "
	    "matrix")}},
#define PGRP(n, fr) TX(n, fr, "a permutation group: \"S4\", \"A5\", " \
	"\"C6\", \"D8\" (order 8), or generators in cycle notation " \
	"\"(1 2 3 4), (1 2)\"")
	{"permgroup", "permgroup", 1, 0, "a permutation group by Schreier-Sims: "
	    "order (exact, any size), base and strong generators, orbits, "
	    "abelian, derived and lower central series, solvable, nilpotent, "
	    "centre", {PGRP("group", " %s")}},
	{"permelem", "permelem", 1, 0, "is a permutation in the group (sifting, "
	    "with the word in the transversals as the certificate); its order",
	    {PGRP("group", " %s"), TX("perm", ", %s", "the permutation: "
	    "\"(1 2 3)(4 5)\" or \"()\"")}},
	{"permsub", "permsub", 1, 0, "H in G: subgroup?, normal?, index, normal "
	    "closure, coset representatives", {PGRP("h", " %s"), PGRP("g",
	    " in %s")}},
	{"permhom", "permhom", 1, 0, "does sending G's generators to the given "
	    "permutations define a homomorphism G -> H", {PGRP("from", " %s"),
	    PGRP("to", " -> %s"), TX("images", " by %s", "the images of G's "
	    "generators, in order: \"(), (1 2)\"")}},
	{"cayley", "cayley", 1, 0, "a finite group by its Cayley table (entries "
	    "0..n-1): the axioms checked, abelian, element orders, solvable, "
	    "nilpotent", {ZM_("table", " %s", "the n x n table: row a, column "
	    "b holds a*b")}},
#undef PGRP
#define POPTS {"order", P_ENUM, 0, " order %s", "the monomial order " \
	"(default grevlex)", "lex|grlex|grevlex"}, {"modulus", P_INT, 0, \
	" mod %s", "a prime p: over F_p (default over Q)", NULL}, {"vars", \
	P_TEXT, 0, " vars %s", "the variables, largest first: \"x, y, z\" " \
	"(default: those that occur, alphabetical)", NULL}
	{"groebner", "groebner", 1, 0, "the reduced Groebner basis of an ideal "
	    "of polynomials over Q or F_p, each element with its cofactors in "
	    "the generators (Buchberger with criteria; verified)",
	    {TX("polys", " %s", "the generators: \"x^2 + y^2 - 1, x - y\""),
	    POPTS}},
	{"idealmember", "idealmember", 1, 0, "is h in the ideal: the cofactors "
	    "h = sum c_j f_j (multiplied out and checked), or the nonzero "
	    "normal form", {TX("h", " %s", "the polynomial"), TX("polys",
	    " in %s", "the ideal's generators"), POPTS}},
	{"idealeq", "idealeq", 1, 0, "compare two ideals: equal, one inside the "
	    "other, or neither (each inclusion certified by cofactors)",
	    {TX("i", " %s", "generators of I"), TX("j", " = %s", "generators "
	    "of J"), POPTS}},
	{"fxmodule", "fxmodule", 1, 0, "F^n as an F[x]-module with x acting by "
	    "a matrix: invariant factors, minimal and characteristic "
	    "polynomial, rational canonical form (Smith form over F[x])",
	    {ZM_("matrix", " %s", "a square matrix over Q (\"1/2\" allowed)"),
	    {"modulus", P_INT, 0, " mod %s", "a prime p: over F_p", NULL}}},
	{"pmsnf", "pmsnf", 1, 0, "the Smith normal form of a matrix of "
	    "polynomials in one variable over Q or F_p, with U and V",
	    {TX("matrix", " %s", "\"[[x, 1], [0, x^2]]\""), {"modulus",
	    P_INT, 0, " mod %s", "a prime p: over F_p", NULL}}},
#undef POPTS
#undef GRP
#undef ZM_
	/* homological algebra (homalg_cmd.c) */
#define HSPACE TX("space", " %s", "a space: S^n, T^n, RP^n, CP^n, K (Klein " \
	"bottle), L(m) or L(m,q), pt; _s for a simplicial model (RP^2_s, " \
	"T^2_s, K_s, S^3_s); X x Y, X v Y, susp(X); cells([[d1]], [[d2]], " \
	"...) by boundary matrices, facets([[0,1,2], ...])"), \
	BO("unreduced", " unreduced", "use the space's own cells (by " \
	"default unit pivots are cancelled first, with the homotopy " \
	"equivalence checked)")
	{"space", "space", 1, 0, "a cell complex: its cells, Euler "
	    "characteristic and boundaries (d o d = 0 checked)", {HSPACE}},
	{"homology", "homology", 1, 0, "H_n(X; G) for every n, exactly, with "
	    "the universal coefficient theorem as a check", {HSPACE,
	    TXO("coefficients", " with %s", "an abelian group: Z/2, Z/4 + Z "
	    "(default Z)"), BO("generators", " generators", "representative "
	    "cycles")}},
	{"cohomology", "cohomology", 1, 0, "H^n(X; G) for every n, exactly, "
	    "with both universal coefficient theorems as checks", {HSPACE,
	    TXO("coefficients", " with %s", "an abelian group: Z/2, Z/4 + Z "
	    "(default Z)"), BO("generators", " generators", "representative "
	    "cocycles")}},
	{"les", "les", 1, 0, "the long exact sequence of 0 -> C -m-> C -> "
	    "C (x) Z/m -> 0 on cochains, the connecting map by the snake "
	    "lemma, exactness verified at every spot", {HSPACE, IR("modulus",
	    " mod %s", "m >= 2")}},
	{"bockstein", "bockstein", 1, 0, "the Bockstein spectral sequence: "
	    "pages E_r, D_r and d_r of the exact couple of 0 -> C -p-> C -> "
	    "C (x) Z/p -> 0, derived exactly until stable", {HSPACE, IR("p",
	    " mod %s", "a prime"), IO("pages", " pages %s", "at most this "
	    "many pages (64)")}},
	{"bockstein_trace", "bockstein", 1, 0, "trace a in H^n(X; Z) "
	    "through the Bockstein spectral sequence: j_r(p^(r-1) a) in E_r, "
	    "in ker d_r?, in im d_r? (with a preimage)", {HSPACE, IR("p",
	    " mod %s", "a prime"), IR("degree", " trace %s", "n"),
	    TXO("element", " %s", "[c1, c2, ...]: canonical coordinates in "
	    "H^n(X; Z) (default: each generator)"), IO("page", " page %s",
	    "only page r")}},
	{"prop01", "prop01", 1, 0, "check Proposition 0.1 of bockstein.pdf "
	    "on a space: for a generating a Z summand of H^n(X; Z), "
	    "j_r(p^(r-1) a) is in ker d_r and not in im d_r, for every n and "
	    "every r (all r up to the stable page, which covers the rest)",
	    {HSPACE, IR("p", " mod %s", "a prime"), IO("pages", " pages %s",
	    "at most this many pages (64)")}},
#undef HSPACE
	/* proofs (prove_cmd.c): an LCF kernel, every theorem re-checked */
	{"theorem", "theorem", 1, 0, "state a theorem in the kernel's "
	    "language (forall x:el, n:int: ..., and or not -> <->, = <= <, "
	    "in; the library's sorts, functions and predicates: see lemmas "
	    "and axioms)", {NM("name", " %s", "its name"), TX("statement",
	    ": %s", "the formula")}},
	{"prove", "prove", 1, 0, "prove a stated theorem: by the library's "
	    "proofs of the same statement (every route, each re-checked "
	    "independently), else by replaying the library's proofs and by "
	    "automation; when it fails, a counterexample is searched in "
	    "concrete Bockstein spectral sequences", {NM("name", " %s",
	    "the theorem"), NSO("using", " using %s", "lemmas and axioms "
	    "for the automation")}},
	{"proof", "proof", 1, 0, "print a proved theorem's proof: each "
	    "route's script, the tree of lemmas it rests on with their "
	    "kernel inferences, and the axioms used", {NM("name", " %s",
	    "the theorem or library lemma")}},
	{"proof_check", "check", 1, 0, "re-check a proved theorem with the "
	    "independent checker (kcheck.c)", {NM("name", " %s", "the theorem "
	    "or library lemma")}},
	{"axioms", "axioms", 1, 0, "the axioms of the proof library "
	    "(definitions and the integer arithmetic of Z); with a name, the "
	    "ones that theorem's proofs rest on", {NMO("name", " %s",
	    "a proved theorem")}},
	{"lemmas", "lemmas", 1, 0, "the proof library's lemmas and theorems",
	    {{0}}},
	/* elliptic curves over Q, Diophantine equations (ec_cmd.c) */
	{"elliptic", "ec", 1, 0, "an elliptic curve over Q: by default the "
	    "summary (invariants, minimal model, Tate's algorithm: Kodaira "
	    "symbols, c_p, conductor; torsion; rank bounds by 2-isogeny "
	    "descent; generators and heights; periods, root number, L-values, "
	    "analytic rank, BSD quotient), or one operation",
	    {TX("curve", " %s", "[a1, a2, a3, a4, a6], [a4, a6], or "
	    "\"y^2 + x*y = x^3 - x + 1\""), {"op", P_ENUM, 0, " %s", "what "
	    "(default: the summary)", "info|points|torsion|rank|local|minimal|"
	    "analytic|periods|on|height|add|mul|order|wp|ap|model|integral"},
	    TXO("args", " %s", "the op's arguments: a point \"(x, y)\" (on, "
	    "height, order), two points (add), \"k (x, y)\" (mul), a height "
	    "bound (points, rank), a bound on |x| (integral), a prime (ap), z "
	    "as \"a+bi\" (wp), a model: short, montgomery, edwards, "
	    "twisted_edwards, legendre")}},
	{"diophantine", "diophantine", 1, 0, "solve a Diophantine equation "
	    "through an elliptic curve: a plane cubic (homogeneous in 3 "
	    "variables, or affine in 2), a curve quadratic in one variable "
	    "with a quartic discriminant, or z^2 = quartic(x, y); rational "
	    "point, birational map to Weierstrass form (kept with its "
	    "inverse), the group, combinations of generators mapped back, "
	    "each solution substituted into the equation; with no solution, "
	    "a proof where one is found (rank 0, or no rational point on the "
	    "real component the conditions need)",
	    {TX("equation", " %s", "\"a/(b+c) + b/(a+c) + c/(a+b) = 4\""),
	    BO("positive", " positive", "every variable > 0"),
	    BO("integer", " integer", "integer values (affine equations)"),
	    BO("nonzero", " nonzero", "every variable != 0"),
	    IO("count", " count %s", "solutions wanted (default 1)"),
	    IO("maxmult", " maxmult %s", "largest |coefficient| of a "
	    "generator (default 30 for rank 1)"),
	    IO("search", " search %s", "point search bound")}},
	{"weierstrass", "weierstrass", 1, 0, "the maps from a cubic or quartic "
	    "curve to Weierstrass form, the minimal model and the group",
	    {TX("equation", " %s", "\"x^3 + y^3 = 7*z^3\"")}},
	{"congruent", "congruent", 1, 0, "is n the area of a right triangle "
	    "with rational sides: the curve y^2 = x^3 - n^2 x, its rank by "
	    "descent, a triangle (checked) or a proof that none exists, "
	    "Tunnell's counts", {TX("n", " %s", "a positive integer")}},
	{"mordell", "mordell", 1, 0, "the Mordell curve y^2 = x^3 + k: the "
	    "summary and its integral points up to a bound",
	    {TX("k", " %s", "a nonzero integer"), IO("bound", " bound %s",
	    "|x| <= bound (default 100000)")}},
	{"cubesum", "cubesum", 1, 0, "k as a sum of two rational cubes: "
	    "x^3 + y^3 = k z^3 solved through its curve", {TX("k", " %s",
	    "a positive integer"), BO("positive", " positive", "x, y > 0")}},
	/* elliptic curves over finite fields, crypto, algorithms (ecff) */
	{"ec_info", "ec", 1, 0, "a curve over F_p: discriminant, j, #E "
	    "(certified by point orders), trace, group structure (certified "
	    "by a Weil pairing), twist, embedding degree, weaknesses",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\"")}},
	{"ec_order", "ecorder", 1, 0, "#E(F_p) by a chosen method, with the "
	    "certificate (points whose orders leave one candidate in the "
	    "Hasse interval)", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), {"method", P_ENUM, 0, " using %s",
	    "naive, baby-step giant-step (Mestre), CM (j = 0, 1728), Schoof, "
	    "SEA (Schoof with splitting here)", "auto|naive|bsgs|cm|schoof|sea"},
	    BO("timed", " timed", "print timings")}},
	{"ec_points", "ecpoints", 1, 0, "the points of E(F_p), p <= 100000",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\"")}},
	{"ec_add", "ecadd", 1, 0, "P + Q", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("P", " %s", "(x, y), "
	    "G or O"), TX("Q", " + %s", "(x, y), G or O")}},
	{"ec_mul", "ecmul", 1, 0, "k P by wNAF/Jacobian, checked against "
	    "affine double-and-add, the Montgomery ladder and plain NAF",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("k", " %s", "the scalar (an integer expression)"),
	    TX("P", " %s", "(x, y), G, random"), {"method", P_ENUM, 0,
	    " using %s", "only this cross-check", "affine|wnaf|ladder"}}},
	{"ecm", "ecm", 1, 0, "factor an integer (up to 576 bits): trial "
	    "division, Pollard rho, Lenstra ECM stage 1 + 2; with B1 one ECM "
	    "run", {TX("n", " %s", "the number (\"2^128 + 1\")"),
	    TXO("B1", " B1 %s", "stage 1 bound"), TXO("B2", " B2 %s",
	    "stage 2 bound")}},
	{"ecdsa_sign", "ecdsa sign", 1, 0, "ECDSA signature with RFC 6979 "
	    "deterministic nonce (variable-time code: mathematics, not a "
	    "hardened implementation)", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("key", " key %s",
	    "the private key d"), {"hash", P_ENUM, 0, " hash %s", "default by "
	    "the order's size", "sha256|sha384|sha512"}, TXO("nonce", " k %s",
	    "a given nonce instead of RFC 6979"), TX("msg", " msg %s",
	    "the message (text)")}},
	{"ecdsa_verify", "ecdsa verify", 1, 0, "verify an ECDSA signature",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("pub", " pub %s", "(x, y) or SEC 1 hex"), TX("r",
	    " r %s", "r"), TX("s", " s %s", "s"), {"hash", P_ENUM, 0,
	    " hash %s", "the hash", "sha256|sha384|sha512"}, TX("msg",
	    " msg %s", "the message (text)")}},
	{"ecdh", "ecdh", 1, 0, "Diffie-Hellman on a named curve: both sides "
	    "from two private keys, or one key and a peer's point (validated)",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("a", " a %s", "a private key"), TXO("b", " b %s",
	    "the other private key"), TXO("pub", " pub %s", "the peer's "
	    "point (x, y)")}},
	{"x25519", "x25519", 1, 0, "RFC 7748 X25519 (hex, little-endian)",
	    {TX("k", " %s", "the scalar, 32 bytes hex"), TXO("u", " %s",
	    "u, 32 bytes hex (default the base point 9)")}},
	{"x448", "x448", 1, 0, "RFC 7748 X448 (hex, little-endian)",
	    {TX("k", " %s", "the scalar, 56 bytes hex"), TXO("u", " %s",
	    "u, 56 bytes hex (default the base point 5)")}},
	{"ed25519_sign", "ed25519 sign", 1, 0, "RFC 8032 Ed25519 signature "
	    "(and the public key)", {TX("secret", " %s", "32 bytes hex"),
	    TX("msghex", " msghex %s", "the message as hex bytes")}},
	{"ed25519_verify", "ed25519 verify", 1, 0, "verify an Ed25519 "
	    "signature", {TX("pub", " %s", "32 bytes hex"), TX("sig", " %s",
	    "64 bytes hex"), TX("msghex", " msghex %s", "the message as hex")}},
	{"ec_compress", "eccompress", 1, 0, "SEC 1 encodings of a point",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("P", " %s", "(x, y) or G")}},
	{"ec_decompress", "ecdecompress", 1, 0, "decode a SEC 1 point",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("hex", " %s", "02/03/04 ...")}},
	{"ec_validate", "ecvalidate", 1, 0, "public-key validation (SEC 1 "
	    "3.2.2.1)", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("P", " %s", "(x, y)")}},
	{"ec_vectors", "ecvectors", 1, 0, "run the built-in RFC/FIPS test "
	    "vectors and the named curves' parameter checks", {{0}}},
	{"ec_dlog", "ecdlog", 1, 0, "a discrete logarithm k with k P = Q: "
	    "Pohlig-Hellman, BSGS, rho, MOV (pairing transfer), Smart "
	    "(anomalous); checked by k P = Q", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TX("P", " %s",
	    "(x, y) or random"), TXO("Q", " %s", "(x, y)"), TXO("secret",
	    " secret %s", "instead of Q: Q = secret P"), {"method", P_ENUM, 0,
	    " using %s", "default: by the curve", "bsgs|rho|ph|mov|smart"}}},
	{"ec_weak", "ecweak", 1, 0, "generate a weak curve (anomalous: #E = "
	    "p; supersingular: embedding degree 2); \"anomalous 64\" also "
	    "works as the curve of ec_dlog", {{"kind", P_ENUM, 1, " %s",
	    "which weakness", "anomalous|supersingular"}, IR("bits", " %s",
	    "the size of p")}},
	{"ec_pairing", "ecpairing", 1, 0, "Weil or reduced Tate pairing by "
	    "Miller's algorithm with bilinearity checks; on bn254/bls12_381 "
	    "alone: e(G1, G2) in F_p^12", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TXO("P", " %s", "(x, y)"),
	    TXO("Q", " %s", "(x, y)"), TXO("n", " %s", "the order n"),
	    {"kind", P_ENUM, 0, " %s", "default weil", "weil|tate"}}},
	{"hilbert", "hilbert", 1, 0, "the Hilbert class polynomial H_D (CM "
	    "j-invariants), rounding error bounded", {TX("D", " %s",
	    "a negative discriminant")}},
	{"ec_cm", "eccm", 1, 0, "a CM curve with discriminant D over F_p, or "
	    "a random prime-order CM curve of a given size", {TX("D", " %s",
	    "the discriminant"), TXO("p", " mod %s", "the prime"),
	    IO("bits", " bits %s", "instead of p: random p and prime #E")}},
	{"ecpp", "ecpp", 1, 0, "an Atkin-Morain ECPP primality certificate, "
	    "verified by an independent checker", {TX("n", " %s", "the "
	    "number (\"2^127 - 1\")")}},
	{"modpoly", "modpoly", 1, 0, "the classical modular polynomial Phi_l "
	    "(over Z, or mod p)", {IR("l", " %s", "a prime <= 31"), TXO("p",
	    " mod %s", "a prime modulus")}},
	{"ec_isogeny", "ecisogeny", 1, 0, "an isogeny from a kernel point "
	    "(Velu, Kohel, checked as a homomorphism), or the rational "
	    "l-isogenies (roots of Phi_l)", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), TXO("kernel", " kernel %s",
	    "a point (x, y) of small order"), TXO("l", " %s", "a prime l")}},
	{"ec_volcano", "ecvolcano", 1, 0, "the l-isogeny graph around j(E)",
	    {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), IR("l", " %s", "a prime <= 13")}},
	{"ec_divpoly", "ecdivpoly", 1, 0, "the division polynomial psi_n, its "
	    "roots and factor degrees (the torsion field)", {TX("curve", " %s", "secp256k1, P-256, Curve25519, BN254, ..., \"y^2 = x^3 + 2x + 3 mod 97\" or \"[a, b] mod p\""), IR("n",
	    " %s", "2 .. 60")}},
	{"ec_binary", "ecb2", 1, 0, "a binary curve y^2 + xy = x^3 + a x^2 + "
	    "b over GF(2^m): count (Koblitz, naive), FIPS parameters checked",
	    {TX("curve", " %s", "K-163, B-163, K-233, or \"[a, b] over "
	    "GF(2^13) poly 4 3 1 0\""), TXO("mul", " mul %s", "k: k G")}},
};
#define NCMDS (int)(sizeof cmds / sizeof cmds[0])

/* request-wide options, accepted by every command */
static const struct jparam opts[] = {
	{"assume", P_OBJ, 0, "", "answers to ambiguities: {key: reading or "
	    "index}, e.g. {\"caret\": \"power\", \"z\": \"complex\"}", NULL},
	{"defaults", P_BOOL, 0, "", "take the default reading of every "
	    "ambiguity (as the CLI does with a note) instead of an error", NULL},
	{"timeout_s", P_NUM, 0, "", "wall-clock limit; the request runs in a "
	    "forked child and the state is unchanged on a timeout", NULL},
	{"files", P_ENUM, 0, "", "what to do with files: write (to --dir, "
	    "default), base64 (inline, nothing written), both, none",
	    "write|base64|both|none"},
	{"answers", P_STRS, 0, "", "replies to follow-up questions, in order",
	    NULL},
};
#define NOPTS (int)(sizeof opts / sizeof opts[0])

/* ---- the server's state ---- */

static FILE *jout;		/* responses: the real stdout */
static struct jv *session;	/* the "assume" command's answers */
static char outdir[PATH_MAX] = ".";
static int default_fmode = REP_FILES_WRITE;
static char **sargv;
static int sargc;

/* ---- validation and rendering ---- */

struct req {
	const struct jcmd *c;
	const struct jv *args;
	struct sbuf pre;	/* statements before the command (definitions) */
	struct sbuf stmt;
	char err[600];
	const char *ecode;
	int nfunc;		/* function objects defined so far (it, that) */
};

static int
bad(struct req *r, const char *code, const char *fmt, ...)
{
	va_list ap;

	if (r->ecode)
		return 0;
	r->ecode = code;
	va_start(ap, fmt);
	vsnprintf(r->err, sizeof r->err, fmt, ap);
	va_end(ap);
	return 0;
}

static int
is_ident(const char *s)
{
	if (!(isalpha((unsigned char)*s) || *s == '_'))
		return 0;
	for (s++; *s; s++)
		if (!(isalnum((unsigned char)*s) || *s == '_'))
			return 0;
	return 1;
}

/* one line of ucas text: no newline (a second statement) or comment */
static int
text_ok(struct req *r, const char *what, const char *s)
{
	const char *p;

	for (p = s; *p; p++)
		if (*p == '\n' || *p == '\r' || *p == '\x1e' || *p == '#' ||
		    (unsigned char)*p < 0x20)
			return bad(r, "bad_request", "%s: one line of ucas text, "
			    "without newlines, control characters or '#'", what);
	if (!*s)
		return bad(r, "bad_request", "%s is empty", what);
	return 1;
}

static int render_expr(struct req *r, struct sbuf *b, const struct jv *v,
    const char *what, int top);

static const char *const binops[] = {"+", "-", "*", "/", "^", "%", "mod",
	"&", "|", "xor", "<<", ">>", "==", "!=", "<", "<=", ">", ">=", "&&",
	"||", "=", NULL};

/*
 * The AST form: {"num": "1/3"}, {"var": "x"}, {"op": "+", "args": [...]}
 * ({"op": "neg"|"not"|"~"} unary, {"op": "?:"} three arguments, any other
 * op a function call), {"expr": "ucas text"}.  Every operation is
 * parenthesised, so no precedence or juxtaposition rule is involved.
 */
static int
render_expr(struct req *r, struct sbuf *b, const struct jv *v,
    const char *what, int top)
{
	const struct jv *a, *x;
	const char *op, *s;
	int k;

	if (v->t == J_STR) {
		if (!text_ok(r, what, v->s))
			return 0;
		sb_printf(b, top ? "%s" : "(%s)", v->s);
		return 1;
	}
	if (v->t == J_NUM) {
		if (v->num < 0)
			sb_putc(b, '(');
		if (v->s)
			sb_puts(b, v->s);
		else
			json_num(b, v->num);
		if (v->num < 0)
			sb_putc(b, ')');
		return 1;
	}
	if (v->t != J_OBJ)
		return bad(r, "bad_request", "%s: an expression is a string or an "
		    "object ({\"op\", \"args\"}, {\"var\"}, {\"num\"}, {\"expr\"})",
		    what);
	if ((x = jv_get(v, "num")) != NULL) {
		if (x->t == J_NUM)
			return render_expr(r, b, x, what, top);
		if (x->t != J_STR || !x->s[0])
			return bad(r, "bad_request", "%s: num is a number or a "
			    "string", what);
		for (s = x->s; *s; s++)
			if (!isdigit((unsigned char)*s) && !strchr("./-+eE", *s))
				return bad(r, "bad_request", "%s: num \"%s\" is not a "
				    "number or a fraction", what, x->s);
		sb_printf(b, "(%s)", x->s);
		return 1;
	}
	if ((s = jv_gets(v, "var")) != NULL) {
		if (!is_ident(s))
			return bad(r, "bad_request", "%s: var \"%s\" is not a name",
			    what, s);
		sb_puts(b, s);
		return 1;
	}
	if ((s = jv_gets(v, "expr")) != NULL) {
		if (!text_ok(r, what, s))
			return 0;
		sb_printf(b, "(%s)", s);
		return 1;
	}
	if ((op = jv_gets(v, "op")) == NULL)
		return bad(r, "bad_request", "%s: an expression object has op, "
		    "var, num or expr", what);
	a = jv_get(v, "args");
	if (a == NULL || a->t != J_ARR || a->n == 0)
		return bad(r, "bad_request", "%s: op \"%s\" needs args (an array)",
		    what, op);
	if (!strcmp(op, "neg") || !strcmp(op, "not") || !strcmp(op, "!") ||
	    !strcmp(op, "~") || (!strcmp(op, "-") && a->n == 1)) {
		if (a->n != 1)
			return bad(r, "bad_request", "%s: %s takes one argument",
			    what, op);
		sb_puts(b, !strcmp(op, "neg") || !strcmp(op, "-") ? "(-" :
		    !strcmp(op, "~") ? "(~" : "(!");
		if (!render_expr(r, b, a->v[0], what, 0))
			return 0;
		sb_putc(b, ')');
		return 1;
	}
	if (!strcmp(op, "?:")) {
		if (a->n != 3)
			return bad(r, "bad_request", "%s: ?: takes three arguments",
			    what);
		sb_putc(b, '(');
		if (!render_expr(r, b, a->v[0], what, 0))
			return 0;
		sb_puts(b, " ? ");
		if (!render_expr(r, b, a->v[1], what, 0))
			return 0;
		sb_puts(b, " : ");
		if (!render_expr(r, b, a->v[2], what, 0))
			return 0;
		sb_putc(b, ')');
		return 1;
	}
	for (k = 0; binops[k]; k++)
		if (!strcmp(op, binops[k]))
			break;
	if (binops[k]) {
		int nary = !strcmp(op, "+") || !strcmp(op, "*") ||
		    !strcmp(op, "&") || !strcmp(op, "|") || !strcmp(op, "xor") ||
		    !strcmp(op, "&&") || !strcmp(op, "||");
		if (a->n < 2 || (!nary && a->n != 2))
			return bad(r, "bad_request", "%s: %s takes %s arguments",
			    what, op, nary ? "two or more" : "two");
		if (!strcmp(op, "=") && !top)
			return bad(r, "bad_request", "%s: = (an equation) only at "
			    "the top", what);
		if (!top || strcmp(op, "="))
			sb_putc(b, '(');
		for (k = 0; k < a->n; k++) {
			if (k)
				sb_printf(b, " %s ", op);
			if (!render_expr(r, b, a->v[k], what, 0))
				return 0;
		}
		if (!top || strcmp(op, "="))
			sb_putc(b, ')');
		return 1;
	}
	if (!is_ident(op))
		return bad(r, "bad_request", "%s: unknown op \"%s\"", what, op);
	sb_printf(b, "%s(", op);
	for (k = 0; k < a->n; k++) {
		if (k)
			sb_puts(b, ", ");
		if (!render_expr(r, b, a->v[k], what, 1))
			return 0;
	}
	sb_putc(b, ')');
	return 1;
}

static int
render_num(struct req *r, struct sbuf *b, const struct jv *v, const char *what)
{
	if (v->t == J_NUM) {
		if (v->s)
			sb_puts(b, v->s);
		else
			json_num(b, v->num);
		return 1;
	}
	if (v->t == J_STR)
		return text_ok(r, what, v->s) && (sb_puts(b, v->s), 1);
	return bad(r, "bad_request", "%s: a number (or a constant expression "
	    "as a string)", what);
}

/* define's parameters: x: u8 in 0..10 uniform */
static int
render_params(struct req *r, struct sbuf *b, const struct jv *ps)
{
	static const char *const tys[] = {"real", "int"};
	const struct jv *p, *rg, *u;
	const char *nm, *ty;
	char what[160];
	int k;

	if (ps->t != J_ARR)
		return bad(r, "bad_request", "params: an array of {\"name\", "
		    "\"type\", \"range\"}");
	sb_putc(b, '(');
	for (k = 0; k < ps->n; k++) {
		p = ps->v[k];
		if (k)
			sb_puts(b, ", ");
		if (p->t == J_STR)
			nm = p->s, p = NULL;
		else if (p->t == J_OBJ && (nm = jv_gets(p, "name")) != NULL)
			;
		else
			return bad(r, "bad_request", "params[%d]: {\"name\", "
			    "\"type\", \"range\"}", k);
		if (!is_ident(nm))
			return bad(r, "bad_request", "params[%d]: \"%s\" is not a "
			    "name", k, nm);
		sb_puts(b, nm);
		ty = p ? jv_gets(p, "type") : NULL;
		rg = p ? jv_get(p, "range") : NULL;
		if (ty) {
			if (!is_ident(ty) && strncmp(ty, "cd(", 3))
				return bad(r, "bad_request", "params[%d]: bad type", k);
			sb_printf(b, ": %s", ty);
		} else if (rg == NULL) {
			/* no type, no range: which? */
			const char *a;
			snprintf(what, sizeof what, "parameter %s has no type", nm);
			if ((a = rep_answer("param_type", what, tys, 2, 0)) != NULL)
				sb_printf(b, ": %s", tys[atoi(a) - 1]);
			else {
				rep_ask("param_type", what, tys, 2, 0, "give "
				    "params[].type (real, int, f16, f32, f64, u8, i32, bool, "
				    "complex ...)");
				return bad(r, "ambiguous", "%s", what);
			}
		}
		if (rg) {
			if (rg->t != J_ARR || rg->n != 2)
				return bad(r, "bad_request", "params[%d].range: [lo, hi]",
				    k);
			sb_puts(b, " in ");
			if (!render_num(r, b, rg->v[0], "range"))
				return 0;
			sb_puts(b, "..");
			if (!render_num(r, b, rg->v[1], "range"))
				return 0;
		}
		if (p && (u = jv_get(p, "uniform")) && u->t == J_TRUE)
			sb_puts(b, " uniform");
		if (p) {
			int i;
			for (i = 0; i < p->n; i++)
				if (strcmp(p->k[i], "name") && strcmp(p->k[i], "type") &&
				    strcmp(p->k[i], "range") && strcmp(p->k[i], "uniform"))
					return bad(r, "bad_request", "params[%d]: unknown "
					    "field \"%s\"", k, p->k[i]);
		}
	}
	sb_putc(b, ')');
	return 1;
}

/* a function argument; an object is defined first (as it, that, ...) */
static int
render_func(struct req *r, struct sbuf *b, const struct jv *v,
    const char *what)
{
	static const char *const names[] = {"it", "that", "it3", "it4"};
	const struct jv *body, *ps;
	const char *nm;

	if (v->t == J_STR)
		return text_ok(r, what, v->s) && (sb_puts(b, v->s), 1);
	if (v->t != J_OBJ || (body = jv_get(v, "body")) == NULL) {
		if (v->t == J_OBJ && (jv_get(v, "op") || jv_get(v, "var") ||
		    jv_get(v, "num") || jv_get(v, "expr")))
			return render_expr(r, b, v, what, 1);
		return bad(r, "bad_request", "%s: a name, an expression, or "
		    "{\"body\", \"params\"}", what);
	}
	nm = jv_gets(v, "name");
	if (nm == NULL)
		nm = names[r->nfunc++ & 3];
	if (!is_ident(nm))
		return bad(r, "bad_request", "%s.name: not a name", what);
	sb_puts(&r->pre, nm);
	if ((ps = jv_get(v, "params")) != NULL && !render_params(r, &r->pre, ps))
		return 0;
	sb_puts(&r->pre, " = ");
	if (!render_expr(r, &r->pre, body, what, 1))
		return 0;
	sb_putc(&r->pre, '\n');
	sb_puts(b, nm);
	return 1;
}

static int
render_list(struct req *r, struct sbuf *b, const struct jv *v,
    const struct jparam *p)
{
	const char *sep = p->type == P_NAMES_SP ? " " : ", ";
	int k;

	if (v->t != J_ARR) {
		if (p->type == P_EXPRS)
			return render_expr(r, b, v, p->name, 1);
		if (v->t == J_STR && is_ident(v->s))
			return sb_puts(b, v->s), 1;
		return bad(r, "bad_request", "%s: an array", p->name);
	}
	if (v->n == 0)
		return bad(r, "bad_request", "%s is empty", p->name);
	for (k = 0; k < v->n; k++) {
		if (k)
			sb_puts(b, sep);
		if (p->type == P_EXPRS) {
			if (!render_expr(r, b, v->v[k], p->name, 1))
				return 0;
		} else if (v->v[k]->t != J_STR || !is_ident(v->v[k]->s))
			return bad(r, "bad_request", "%s[%d]: a name", p->name, k);
		else
			sb_puts(b, v->v[k]->s);
	}
	return 1;
}

static int
enum_ok(const struct jparam *p, const char *s)
{
	const char *v = p->values, *e;
	size_t n = strlen(s);

	while (v && *v) {
		e = strchr(v, '|');
		if (e == NULL)
			e = v + strlen(v);
		if ((size_t)(e - v) == n && !strncmp(v, s, n))
			return 1;
		v = *e ? e + 1 : e;
	}
	return 0;
}

/* one argument's value as ucas text */
static int render_inputs(struct req *r, struct sbuf *b, const struct jv *v,
    const char *name);

static int
render_arg(struct req *r, struct sbuf *b, const struct jparam *p,
    const struct jv *v)
{
	int k;

	switch (p->type) {
	case P_FUNC:
		return render_func(r, b, v, p->name);
	case P_EXPR:
		return render_expr(r, b, v, p->name, 1);
	case P_EXPRS:
	case P_NAMES:
	case P_NAMES_SP:
		return render_list(r, b, v, p);
	case P_NAME:
		if (v->t != J_STR || !is_ident(v->s))
			return bad(r, "bad_request", "%s: a name", p->name);
		sb_puts(b, v->s);
		return 1;
	case P_INT:
		if (v->t != J_NUM || v->num != floor(v->num) ||
		    fabs(v->num) > 1e15)
			return bad(r, "bad_request", "%s: an integer", p->name);
		sb_printf(b, "%lld", (long long)v->num);
		return 1;
	case P_NUM:
		return render_num(r, b, v, p->name);
	case P_BOOL:
		if (v->t != J_TRUE && v->t != J_FALSE)
			return bad(r, "bad_request", "%s: true or false", p->name);
		return 1;
	case P_ENUM:
		if (v->t != J_STR || !enum_ok(p, v->s))
			return bad(r, "bad_request", "%s: one of %s", p->name,
			    p->values);
		sb_puts(b, v->s);
		return 1;
	case P_TEXT:
		if (v->t != J_STR)
			return bad(r, "bad_request", "%s: a string", p->name);
		return text_ok(r, p->name, v->s) && (sb_puts(b, v->s), 1);
	case P_INTERVAL:
		if (v->t != J_ARR || v->n != 2)
			return bad(r, "bad_request", "%s: [lo, hi]", p->name);
		if (!render_num(r, b, v->v[0], p->name))
			return 0;
		sb_puts(b, "..");
		return render_num(r, b, v->v[1], p->name);
	case P_SUBS:
		if (v->t != J_OBJ || v->n == 0)
			return bad(r, "bad_request", "%s: {\"x\": value, ...}",
			    p->name);
		for (k = 0; k < v->n; k++) {
			if (!is_ident(v->k[k]))
				return bad(r, "bad_request", "%s: \"%s\" is not a name",
				    p->name, v->k[k]);
			sb_printf(b, "%s%s = ", k ? ", " : "", v->k[k]);
			if (!render_expr(r, b, v->v[k], p->name, 1))
				return 0;
		}
		return 1;
	case P_STRS:
		if (v->t != J_ARR)
			return bad(r, "bad_request", "%s: an array of strings",
			    p->name);
		for (k = 0; k < v->n; k++)
			if (v->v[k]->t != J_STR || !text_ok(r, p->name, v->v[k]->s))
				return bad(r, "bad_request", "%s[%d]: a string",
				    p->name, k);
		return 1;
	case P_PARAMS:	/* checked as define renders them */
		return 1;
	case P_RING:
		if (v->t != J_STR)
			return bad(r, "bad_request", "%s: a ring, e.g. \"Z/12\", "
			    "\"GF(2^8)\", \"Z[i]\"", p->name);
		return text_ok(r, p->name, v->s) && (sb_puts(b, v->s), 1);
	case P_RINGMOD:
		return render_num(r, b, v, p->name);
	case P_MATRIX:
		if (v->t == J_STR)
			return text_ok(r, p->name, v->s) && (sb_puts(b, v->s), 1);
		if (v->t != J_ARR || v->n == 0)
			return bad(r, "bad_request", "%s: [[a, b], [c, d]]", p->name);
		sb_putc(b, '[');
		for (k = 0; k < v->n; k++) {
			int j;
			const struct jv *row = v->v[k];
			if (row->t != J_ARR || row->n == 0)
				return bad(r, "bad_request", "%s[%d]: a row (an array)",
				    p->name, k);
			sb_printf(b, "%s[", k ? ", " : "");
			for (j = 0; j < row->n; j++) {
				if (j)
					sb_puts(b, ", ");
				if (!render_num(r, b, row->v[j], p->name))
					return 0;
			}
			sb_putc(b, ']');
		}
		sb_putc(b, ']');
		return 1;
	case P_VALUES:
		if (v->t != J_ARR || v->n == 0)
			return bad(r, "bad_request", "%s: [x0, x1, ...]", p->name);
		sb_putc(b, '[');
		for (k = 0; k < v->n; k++) {
			if (k)
				sb_puts(b, ", ");
			if (!render_num(r, b, v->v[k], p->name))
				return 0;
		}
		sb_putc(b, ']');
		return 1;
	case P_CONGS:
		if (v->t != J_ARR || v->n == 0)
			return bad(r, "bad_request", "%s: [{\"residue\", "
			    "\"modulus\"}, ...]", p->name);
		for (k = 0; k < v->n; k++) {
			const struct jv *c = v->v[k], *rr, *mm;
			if (k)
				sb_puts(b, ", ");
			if (c->t == J_STR) {
				if (!text_ok(r, p->name, c->s))
					return 0;
				sb_puts(b, c->s);
				continue;
			}
			if ((rr = jv_get(c, "residue")) == NULL ||
			    (mm = jv_get(c, "modulus")) == NULL)
				return bad(r, "bad_request", "%s[%d]: {\"residue\", "
				    "\"modulus\"} or \"x = r mod m\"", p->name, k);
			sb_puts(b, "x = ");
			if (!render_num(r, b, rr, p->name))
				return 0;
			sb_puts(b, " mod ");
			if (!render_num(r, b, mm, p->name))
				return 0;
		}
		return 1;
	case P_OBJ:
		if (v->t != J_OBJ)
			return bad(r, "bad_request", "%s: an object", p->name);
		return 1;
	case P_INPUTS:
		return render_inputs(r, b, v, p->name);
	case P_SOURCE:
		if (v->t != J_STR || !v->s[0])
			return bad(r, "bad_request", "%s: a string", p->name);
		if (strchr(v->s, '\x1e'))
			return bad(r, "bad_request", "%s: no \\x1e directives",
			    p->name);
		return 1;
	}
	return bad(r, "internal", "parameter type");
}

/*
 * animate exec's inputs: " input K WORDS" for each; images in base64
 * go to the gallery and are named by their place in it
 */
static int
render_inputs(struct req *r, struct sbuf *b, const struct jv *v,
    const char *name)
{
	static const char *const keys[] = {"color", "colour", "pattern",
	    "format", "file", "png_base64", "base64", "image_base64", "name",
	    "width", "height", "filter", "address", NULL};
	int k, j;

	if (v->t != J_ARR)
		return bad(r, "bad_request", "%s: an array of inputs", name);
	if (v->n > IMG_MAXIN)
		return bad(r, "bad_request", "%s: at most %d inputs", name,
		    IMG_MAXIN);
	img_gallery_clear();	/* checked, then built: added once */
	for (k = 0; k < v->n; k++) {
		const struct jv *e = v->v[k], *x;
		struct imgin in;
		unsigned char c[4];
		const char *s;
		memset(&in, 0, sizeof in);
		in.fmt = in.filt = in.addr = -1;
		sb_printf(b, " input %d", k + 1);
		if (e->t == J_STR) {
			if (!img_in_word(&in, e->s) || strchr(e->s, ' '))
				return bad(r, "bad_request", "%s[%d]: \"%s\" is not a "
				    "colour, pattern or format", name, k, e->s);
			if (in.kind == IK_COLOR)
				sb_printf(b, " #%02x%02x%02x%02x", in.rgba[0],
				    in.rgba[1], in.rgba[2], in.rgba[3]);
			else
				sb_printf(b, " %s", e->s);
			continue;
		}
		if (e->t == J_NULL)
			continue;	/* noise */
		if (e->t != J_OBJ)
			return bad(r, "bad_request", "%s[%d]: a string or an object",
			    name, k);
		for (j = 0; j < e->n; j++) {
			int q;
			for (q = 0; keys[q] && strcmp(keys[q], e->k[j]); q++)
				;
			if (!keys[q])
				return bad(r, "bad_request", "%s[%d]: unknown field "
				    "\"%s\"", name, k, e->k[j]);
		}
		if ((s = jv_gets(e, "color")) != NULL ||
		    (s = jv_gets(e, "colour")) != NULL) {
			if (!img_color(s, c))
				return bad(r, "bad_request", "%s[%d].color: \"%s\" is "
				    "not a colour (a name, #rrggbb, rgba(...))", name, k, s);
			sb_printf(b, " #%02x%02x%02x%02x", c[0], c[1], c[2], c[3]);
		}
		if ((s = jv_gets(e, "pattern")) != NULL) {
			if (strcmp(s, "noise") && strcmp(s, "gradient") &&
			    strcmp(s, "checker") && strcmp(s, "checkerboard"))
				return bad(r, "bad_request", "%s[%d].pattern: noise, "
				    "gradient or checker", name, k);
			sb_printf(b, " %s", s);
		}
		if ((s = jv_gets(e, "file")) != NULL) {
			if (!s[0] || strpbrk(s, " \t\n\x1e"))
				return bad(r, "bad_request", "%s[%d].file: a path "
				    "without blanks", name, k);
			sb_printf(b, " file %s", s);
		}
		if ((s = jv_gets(e, "png_base64")) != NULL ||
		    (s = jv_gets(e, "base64")) != NULL ||
		    (s = jv_gets(e, "image_base64")) != NULL) {
			size_t n = strlen(s);
			unsigned char *d = xmalloc(n * 3 / 4 + 4);
			const char *nm = jv_gets(e, "name");
			char lab[64];
			size_t dn = img_b64(s, n, d);
			snprintf(lab, sizeof lab, "%s", nm ? nm : "");
			if (!lab[0])
				snprintf(lab, sizeof lab, "image %d", img_opt.ngal + 1);
			if (img_sniff(d, dn) == NULL || !img_gallery_add(lab, d, dn)) {
				free(d);
				return bad(r, "bad_request", "%s[%d]: the base64 is not "
				    "an image file I can read, or there are more than %d",
				    name, k, IMG_MAXGAL);
			}
			free(d);
			sb_printf(b, " gallery %d", img_opt.ngal);
		}
		if ((s = jv_gets(e, "format")) != NULL) {
			if (tex_fmt_find(s) < 0)
				return bad(r, "bad_request", "%s[%d].format: \"%s\" is "
				    "not a format (R8, RG8, RGB8, RGBA8, RGBA16F ...)",
				    name, k, s);
			sb_printf(b, " %s", tex_fmts[tex_fmt_find(s)].name);
		}
		if ((x = jv_get(e, "width")) != NULL || jv_get(e, "height")) {
			const struct jv *y = jv_get(e, "height");
			int w = x && x->t == J_NUM ? (int)x->num : 64,
			    h = y && y->t == J_NUM ? (int)y->num : w;
			if (w < 1 || h < 1 || w > 1024 || h > 1024)
				return bad(r, "bad_request", "%s[%d]: width and height "
				    "from 1 to 1024", name, k);
			sb_printf(b, " %dx%d", w, h);
		}
		if ((s = jv_gets(e, "filter")) != NULL) {
			if (strcmp(s, "nearest") && strcmp(s, "linear"))
				return bad(r, "bad_request", "%s[%d].filter: nearest or "
				    "linear", name, k);
			sb_printf(b, " %s", s);
		}
		if ((s = jv_gets(e, "address")) != NULL) {
			if (strcmp(s, "repeat") && strcmp(s, "clamp") &&
			    strcmp(s, "mirror") && strcmp(s, "border"))
				return bad(r, "bad_request", "%s[%d].address: repeat, "
				    "clamp, mirror or border", name, k);
			sb_printf(b, " %s", s);
		}
	}
	return 1;
}

static const struct jparam *
find_param(const struct jcmd *c, const char *name)
{
	int k;

	for (k = 0; k < MAXJP && c->p[k].name; k++)
		if (!strcmp(c->p[k].name, name))
			return &c->p[k];
	for (k = 0; k < NOPTS; k++)
		if (!strcmp(opts[k].name, name))
			return &opts[k];
	return NULL;
}

/* every argument known and well-typed, the required ones present */
static int
check_args(struct req *r)
{
	const struct jcmd *c = r->c;
	struct sbuf tmp = {0};
	int k, ok = 1;

	for (k = 0; r->args && k < r->args->n && ok; k++) {
		const struct jparam *p = find_param(c, r->args->k[k]);
		if (p == NULL) {
			ok = bad(r, "bad_request", "%s has no argument \"%s\"",
			    c->name, r->args->k[k]);
			break;
		}
		if (p->type != P_FUNC) {   /* functions define things: later */
			sb_clear(&tmp);
			ok = render_arg(r, &tmp, p, r->args->v[k]);
		}
	}
	for (k = 0; ok && k < MAXJP && c->p[k].name; k++)
		if (c->p[k].req && c->p[k].type != P_RING &&
		    jv_get(r->args, c->p[k].name) == NULL)
			ok = bad(r, "bad_request", "%s needs \"%s\" (%s)", c->name,
			    c->p[k].name, c->p[k].desc);
	free(tmp.p);
	return ok;
}

/*
 * A ring argument: "ring" (with "modulus" for GF's polynomial: GF(2^8,
 * aes)), or "modulus" alone (mod N).  The fragment's preposition is used
 * with a ring; Z/n and Z[i] style rings read better with "in".
 */
static int
ring_arg(struct req *r, const struct jparam *p)
{
	const struct jv *rv = jv_get(r->args, p->name),
	    *mv = jv_get(r->args, "modulus");
	struct sbuf m = {0}, t = {0};
	int ok = 1;

	if (mv && !render_num(r, &m, mv, "modulus"))
		ok = 0;
	if (ok && rv) {
		if (rv->t != J_STR || !text_ok(r, p->name, rv->s))
			ok = rv->t == J_STR ? 0 : bad(r, "bad_request", "%s: a ring",
			    p->name);
		else {
			size_t n = strlen(rv->s);
			if (mv && !strncmp(rv->s, "GF(", 3) && rv->s[n - 1] == ')' &&
			    !strchr(rv->s, ','))
				sb_printf(&t, "%.*s, %s)", (int)(n - 1), rv->s, m.p);
			else if (mv)
				ok = bad(r, "bad_request", "modulus goes with a GF(q) ring "
				    "(its polynomial), or alone (mod N)");
			else
				sb_puts(&t, rv->s);
			if (ok) {
				const char *fr = p->frag;
				if (!strncmp(fr, " over ", 6) && strncmp(rv->s, "GF", 2) &&
				    strncmp(rv->s, "gf", 2))
					fr = " in %s";
				sb_printf(&r->stmt, fr, t.p);
			}
		}
	} else if (ok && mv) {
		if (!strcmp(r->c->name, "ring"))
			sb_printf(&r->stmt, " Z/%s", m.p);
		else
			sb_printf(&r->stmt, " mod %s", m.p);
	} else if (ok && p->req)
		ok = bad(r, "bad_request", "%s needs \"ring\" or \"modulus\"",
		    r->c->name);
	free(m.p);
	free(t.p);
	return ok;
}

/* the statement of a template command */
static int
build(struct req *r)
{
	const struct jcmd *c = r->c;
	const struct jv *v;
	struct sbuf val = {0};
	int k;

	if (jv_get(r->args, "from") && jv_get(r->args, "to") == NULL &&
	    !strcmp(c->name, "integrate"))
		return bad(r, "bad_request", "integrate: from needs to");
	if (jv_get(r->args, "to") && jv_get(r->args, "from") == NULL &&
	    !strcmp(c->name, "integrate"))
		return bad(r, "bad_request", "integrate: to needs from");
	sb_puts(&r->stmt, c->stmt);
	for (k = 0; k < MAXJP && c->p[k].name; k++) {
		const struct jparam *p = &c->p[k];
		if (p->type == P_RING && !ring_arg(r, p))
			return 0;
		if (p->type == P_RING)
			continue;
		if ((v = jv_get(r->args, p->name)) == NULL)
			continue;
		if (p->type == P_BOOL) {
			if (v->t == J_TRUE)
				sb_puts(&r->stmt, p->frag);
			continue;
		}
		if (p->type == P_RINGMOD)
			continue;	/* with the ring */
		sb_clear(&val);
		sb_puts(&val, "");
		if (!render_arg(r, &val, p, v)) {
			free(val.p);
			return 0;
		}
		sb_printf(&r->stmt, p->frag, val.p);
	}
	free(val.p);
	return 1;
}

/* ---- capturing stdout ---- */

static int cap_saved = -1;
static FILE *cap_fp;
static int cap_shm;
#define CAP_SHM_MAX (64 << 20)

static void
cap_begin(void)
{
	fflush(stdout);
	cap_fp = tmpfile();
	cap_shm = 0;
#ifdef SHM_ANON
	/*
	 * No file system (the Discord bot's worker is in capability mode):
	 * an anonymous shared memory object, sized up front, since writes do
	 * not grow it.  The output ends at the shared offset.
	 */
	if (cap_fp == NULL) {
		int fd = shm_open(SHM_ANON, O_RDWR, 0600);
		if (fd >= 0 && ftruncate(fd, CAP_SHM_MAX) == 0 &&
		    (cap_fp = fdopen(fd, "r")) != NULL)
			cap_shm = 1;
		else if (fd >= 0)
			close(fd);
	}
#endif
	if (cap_fp == NULL)
		return;
	cap_saved = dup(1);
	dup2(fileno(cap_fp), 1);
}

static char *
cap_end(void)
{
	struct sbuf b = {0};
	char buf[4096];
	size_t n;
	off_t left = -1;

	fflush(stdout);
	sb_puts(&b, "");
	if (cap_fp == NULL)
		return b.p;
	if (cap_shm)
		left = lseek(fileno(cap_fp), 0, SEEK_CUR);
	dup2(cap_saved, 1);
	close(cap_saved);
	cap_saved = -1;
	rewind(cap_fp);
	while ((left < 0 || left > 0) && (n = fread(buf, 1, left >= 0 &&
	    left < (off_t)sizeof buf ? (size_t)left : sizeof buf, cap_fp)) > 0) {
		if (left > 0)
			left -= (off_t)n;
		size_t i;
		for (i = 0; i < n; i++)
			sb_putc(&b, buf[i]);
	}
	fclose(cap_fp);
	cap_fp = NULL;
	return b.p;
}

/*
 * Long lists in a result are cut at REP_LIST_MAX entries, with
 * "KEY_omitted": the number left out (a 20000-row table is not useful to
 * an agent and would flood an MCP client's context).
 */
#define REP_LIST_MAX 500

static void
cap_lists(struct jv *v)
{
	int i, n;
	char key[200];

	if (v == NULL)
		return;
	if (v->t == J_ARR) {
		for (i = 0; i < v->n; i++)
			cap_lists(v->v[i]);
		return;
	}
	if (v->t != J_OBJ)
		return;
	n = v->n;
	for (i = 0; i < n; i++) {
		struct jv *a = v->v[i];
		if (a->t == J_ARR && a->n > REP_LIST_MAX) {
			int j, cut = a->n - REP_LIST_MAX;
			for (j = REP_LIST_MAX; j < a->n; j++)
				jv_free(a->v[j]);
			a->n = REP_LIST_MAX;
			snprintf(key, sizeof key, "%s_omitted", v->k[i]);
			jv_set(v, key, jv_int(cut));
		}
		cap_lists(v->v[i]);
	}
}

/* ---- running ---- */

static void
run_stmts(const char *src, int echo)
{
	jmp_buf saved;

	nlp_asked = 0;
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		run_source(src, "<json>", echo);
	else
		rep_failed();
	memcpy(fail_jmp, saved, sizeof saved);
}

/* functions as data (define, state) */
static struct jv *
func_json(const struct func *f, const char *kind)
{
	struct jv *o = jv_new(J_OBJ), *ps = jv_new(J_ARR), *p, *rg;
	struct sbuf sig = {0};
	int k;

	if (kind)
		jv_set(o, "kind", jv_str(kind));
	jv_set(o, "name", jv_str(sym_str(f->name)));
	sb_printf(&sig, "%s(", sym_str(f->name));
	for (k = 0; k < f->np; k++) {
		const struct param *pp = &f->p[k];
		const char *ty = f->halg ? (f->hk[k] == 1 ? "complex" :
		    f->hk[k] == 2 ? "quat" : f->hk[k] == 3 ? "oct" :
		    f->hk[k] ? "cd" : ty_str(pp->ty)) : ty_str(pp->ty);
		p = jv_new(J_OBJ);
		jv_set(p, "name", jv_str(sym_str(pp->name)));
		jv_set(p, "type", jv_str(ty));
		sb_printf(&sig, "%s%s: %s", k ? ", " : "", sym_str(pp->name), ty);
		if (pp->ranged) {
			rg = jv_new(J_ARR);
			if (TY_FP(pp->ty) || pp->ty == TY_REAL) {
				jv_push(rg, jv_num(pp->rlo));
				jv_push(rg, jv_num(pp->rhi));
				sb_puts(&sig, " in ");
				json_num(&sig, pp->rlo);
				sb_puts(&sig, "..");
				json_num(&sig, pp->rhi);
			} else {
				jv_push(rg, jv_int(pp->lo));
				jv_push(rg, jv_int(pp->hi));
				sb_printf(&sig, " in %lld..%lld", (long long)pp->lo,
				    (long long)pp->hi);
			}
			jv_set(p, "range", rg);
		}
		if (pp->uniform)
			jv_set(p, "uniform", jv_bool(1));
		jv_push(ps, p);
	}
	sb_printf(&sig, ")");
	jv_set(o, "params", ps);
	if (f->body)
		jv_set(o, "body", jv_str(node_str(f->body)));
	if (f->where)
		jv_set(o, "where", jv_str(node_str(f->where)));
	jv_set(o, "signature", jv_str(sig.p));
	free(sig.p);
	return o;
}

static struct jv *
add_obj(struct jv *res, const char *key, struct jv *v)
{
	jv_set(res, key, v);
	return v;
}

/* ---- the schema ---- */

static struct jv *parse_const(const char *s);

static struct jv *
type_schema(const struct jparam *p)
{
	struct jv *o = jv_new(J_OBJ), *a;
	const char *v, *e;

	switch (p->type) {
	case P_FUNC:
		jv_set(o, "$ref", jv_str("#/$defs/function"));
		break;
	case P_EXPR:
		jv_set(o, "$ref", jv_str("#/$defs/expression"));
		break;
	case P_EXPRS:
		jv_set(o, "$ref", jv_str("#/$defs/expressions"));
		break;
	case P_NAME:
		jv_set(o, "$ref", jv_str("#/$defs/name"));
		break;
	case P_NAMES:
	case P_NAMES_SP:
		jv_set(o, "type", jv_str("array"));
		a = jv_new(J_OBJ);
		jv_set(a, "$ref", jv_str("#/$defs/name"));
		jv_set(o, "items", a);
		break;
	case P_INT:
		jv_set(o, "type", jv_str("integer"));
		break;
	case P_NUM:
		jv_set(o, "$ref", jv_str("#/$defs/number"));
		break;
	case P_BOOL:
		jv_set(o, "type", jv_str("boolean"));
		break;
	case P_ENUM:
		a = jv_new(J_ARR);
		for (v = p->values; *v; v = *e ? e + 1 : e) {
			e = strchr(v, '|');
			if (e == NULL)
				e = v + strlen(v);
			jv_push(a, jv_strn(v, (size_t)(e - v)));
		}
		jv_set(o, "enum", a);
		break;
	case P_TEXT:
	case P_SOURCE:
		jv_set(o, "type", jv_str("string"));
		break;
	case P_INTERVAL:
		jv_set(o, "$ref", jv_str("#/$defs/interval"));
		break;
	case P_SUBS:
		jv_set(o, "type", jv_str("object"));
		a = jv_new(J_OBJ);
		jv_set(a, "$ref", jv_str("#/$defs/expression"));
		jv_set(o, "additionalProperties", a);
		break;
	case P_PARAMS:
		jv_set(o, "$ref", jv_str("#/$defs/params"));
		break;
	case P_STRS:
		jv_set(o, "type", jv_str("array"));
		a = jv_new(J_OBJ);
		jv_set(a, "type", jv_str("string"));
		jv_set(o, "items", a);
		break;
	case P_OBJ:
		jv_set(o, "type", jv_str("object"));
		break;
	}
	switch (p->type) {
	case P_RING:
	case P_RINGMOD:
		jv_set(o, "type", jv_str(p->type == P_RING ? "string" : "number"));
		if (p->type == P_RINGMOD) {
			a = jv_new(J_ARR);
			jv_push(a, jv_str("number"));
			jv_push(a, jv_str("string"));
			jv_set(o, "type", a);
		}
		break;
	case P_MATRIX:
		jv_set(o, "type", jv_str("array"));
		jv_set(o, "items", parse_const("{\"type\": \"array\", \"items\": "
		    "{\"$ref\": \"#/$defs/number\"}}"));
		break;
	case P_VALUES:
		jv_set(o, "type", jv_str("array"));
		jv_set(o, "items", parse_const("{\"$ref\": \"#/$defs/number\"}"));
		break;
	case P_INPUTS:
		jv_set(o, "type", jv_str("array"));
		jv_set(o, "items", parse_const("{\"oneOf\": [{\"type\": "
		    "\"string\"}, {\"type\": \"null\"}, {\"type\": \"object\", "
		    "\"properties\": {\"color\": {\"type\": \"string\"}, "
		    "\"pattern\": {\"enum\": [\"noise\", \"gradient\", "
		    "\"checker\"]}, \"png_base64\": {\"type\": \"string\"}, "
		    "\"name\": {\"type\": \"string\"}, \"file\": {\"type\": "
		    "\"string\"}, \"format\": {\"type\": \"string\"}, "
		    "\"width\": {\"type\": \"integer\"}, \"height\": "
		    "{\"type\": \"integer\"}, \"filter\": {\"enum\": "
		    "[\"nearest\", \"linear\"]}, \"address\": {\"enum\": "
		    "[\"repeat\", \"clamp\", \"mirror\", \"border\"]}}, "
		    "\"additionalProperties\": false}]}"));
		break;
	case P_CONGS:
		jv_set(o, "type", jv_str("array"));
		jv_set(o, "items", parse_const("{\"oneOf\": [{\"type\": "
		    "\"string\"}, {\"type\": \"object\", \"required\": "
		    "[\"residue\", \"modulus\"], \"properties\": {\"residue\": "
		    "{\"$ref\": \"#/$defs/number\"}, \"modulus\": {\"$ref\": "
		    "\"#/$defs/number\"}}}]}"));
		break;
	}
	jv_set(o, "description", jv_str(p->desc));
	return o;
}

static struct jv *
parse_const(const char *s)
{
	char err[100];

	return jv_parse(s, err, sizeof err, NULL);
}

static const char schema_defs[] =
"{\"name\": {\"type\": \"string\", \"pattern\": \"^[A-Za-z_][A-Za-z0-9_]*$\"},"
" \"number\": {\"description\": \"a JSON number, or a constant expression "
"in ucas syntax (\\\"pi/2\\\", \\\"inf\\\")\", \"type\": [\"number\", "
"\"string\"]},"
" \"interval\": {\"type\": \"array\", \"minItems\": 2, \"maxItems\": 2, "
"\"items\": {\"$ref\": \"#/$defs/number\"}},"
" \"expression\": {\"description\": \"ucas syntax as a string (one line), "
"or the AST form, fully parenthesised when rendered\", \"oneOf\": ["
"{\"type\": \"string\"}, {\"type\": \"number\"},"
" {\"type\": \"object\", \"required\": [\"op\", \"args\"], \"properties\": "
"{\"op\": {\"description\": \"+ - * / ^ % mod & | xor << >> == != < <= > "
">= && || = (binary; + * & | xor && || n-ary), neg not ~ (unary), ?: "
"(three), or a function name (sin, atan2, isqrt, a defined function)\", "
"\"type\": \"string\"}, \"args\": {\"type\": \"array\", \"items\": "
"{\"$ref\": \"#/$defs/expression\"}}}, \"additionalProperties\": false},"
" {\"type\": \"object\", \"required\": [\"var\"], \"properties\": {\"var\": "
"{\"$ref\": \"#/$defs/name\"}}, \"additionalProperties\": false},"
" {\"type\": \"object\", \"required\": [\"num\"], \"properties\": {\"num\": "
"{\"description\": \"a number or an exact fraction \\\"1/3\\\"\", "
"\"type\": [\"string\", \"number\"]}}, \"additionalProperties\": false},"
" {\"type\": \"object\", \"required\": [\"expr\"], \"properties\": "
"{\"expr\": {\"type\": \"string\"}}, \"additionalProperties\": false}]},"
" \"expressions\": {\"oneOf\": [{\"$ref\": \"#/$defs/expression\"}, "
"{\"type\": \"array\", \"items\": {\"$ref\": \"#/$defs/expression\"}}]},"
" \"params\": {\"type\": \"array\", \"items\": {\"type\": \"object\", "
"\"required\": [\"name\"], \"properties\": {\"name\": {\"$ref\": "
"\"#/$defs/name\"}, \"type\": {\"description\": \"real int bool f16 f32 f64 "
"uN iN complex c64 quat oct sed cd(N)\", \"type\": \"string\"}, \"range\": "
"{\"$ref\": \"#/$defs/interval\"}, \"uniform\": {\"type\": \"boolean\"}}, "
"\"additionalProperties\": false}},"
" \"function\": {\"description\": \"a defined name, an expression with "
"inline annotations (isqrt(x: u4)), or an object defined first (as it, "
"that) and then named\", \"oneOf\": [{\"type\": \"string\"}, {\"type\": "
"\"object\", \"required\": [\"body\"], \"properties\": {\"body\": "
"{\"$ref\": \"#/$defs/expression\"}, \"params\": {\"$ref\": "
"\"#/$defs/params\"}, \"name\": {\"$ref\": \"#/$defs/name\"}}, "
"\"additionalProperties\": false}, {\"$ref\": \"#/$defs/expression\"}]}}";

static const char schema_response[] =
"{\"type\": \"object\", \"required\": [\"id\", \"ok\", \"cmd\"], "
"\"properties\": {"
"\"id\": {\"description\": \"the request's id, as given (null if none)\"},"
" \"ok\": {\"type\": \"boolean\"},"
" \"cmd\": {\"type\": [\"string\", \"null\"]},"
" \"ucas\": {\"description\": \"the ucas statements that ran (error "
"positions refer to them)\", \"type\": \"string\"},"
" \"structured\": {\"description\": \"result has fields beyond text\", "
"\"type\": \"boolean\"},"
" \"result\": {\"description\": \"the command's fields (kind, ...); "
"exact values as {value, exact}; expressions as ucas strings that parse "
"back; unstructured commands give {text}\", \"type\": \"object\"},"
" \"results\": {\"description\": \"every result, when the request ran "
"several commands\", \"type\": \"array\"},"
" \"notes\": {\"type\": \"array\", \"items\": {\"type\": \"object\", "
"\"properties\": {\"kind\": {\"enum\": [\"rule\", \"info\", \"assumed\", "
"\"default\"]}, \"text\": {\"type\": \"string\"}}}},"
" \"conditions\": {\"description\": \"where the answer holds\", "
"\"type\": \"array\", \"items\": {\"type\": \"string\"}},"
" \"checks\": {\"type\": \"array\", \"items\": {\"type\": \"object\", "
"\"properties\": {\"kind\": {\"type\": \"string\"}, \"points\": {\"type\": "
"\"integer\"}, \"max_diff\": {\"type\": \"number\"}, \"passed\": {\"type\": "
"\"boolean\"}, \"text\": {\"type\": \"string\"}}}},"
" \"proof\": {\"type\": \"object\", \"properties\": {\"kind\": {\"enum\": "
"[\"symbolic\", \"exhaustive\", \"sat\", \"numeric\", \"search\"]}, "
"\"text\": {\"type\": \"string\"}}},"
" \"files\": {\"type\": \"array\", \"items\": {\"type\": \"object\", "
"\"properties\": {\"name\": {\"type\": \"string\"}, \"mime\": {\"type\": "
"\"string\"}, \"bytes\": {\"type\": \"integer\"}, \"path\": {\"type\": "
"\"string\"}, \"base64\": {\"type\": \"string\"}}}},"
" \"text\": {\"description\": \"the output as the CLI prints it\", "
"\"type\": \"string\"},"
" \"error\": {\"type\": \"object\", \"required\": [\"code\", \"message\"], "
"\"properties\": {\"code\": {\"enum\": [\"bad_json\", \"bad_request\", "
"\"unknown_command\", \"parse_error\", \"domain_error\", \"unsupported\", "
"\"failed\", \"ambiguous\", \"needs_input\", \"not_understood\", "
"\"timeout\", \"internal\"]}, \"message\": {\"type\": \"string\"}, "
"\"pos\": {\"type\": \"object\", \"properties\": {\"line\": {\"type\": "
"\"integer\"}, \"col\": {\"type\": \"integer\"}}}, \"key\": {\"description\": "
"\"ambiguous: the args.assume key that picks a reading\", \"type\": "
"\"string\"}, \"readings\": {\"type\": \"array\", \"items\": {\"type\": "
"\"object\", \"properties\": {\"index\": {\"type\": \"integer\"}, "
"\"reading\": {\"type\": \"string\"}, \"default\": {\"type\": \"boolean\"}, "
"\"how\": {\"type\": \"string\"}}}}, \"how\": {\"type\": \"string\"}}}}}";

static struct jv *
schema(void)
{
	struct jv *s = jv_new(J_OBJ), *defs, *req, *one, *list, *c, *a, *pr,
	    *rq, *ps, *pi;
	int k, i;

	jv_set(s, "$schema", jv_str("https://json-schema.org/draft/2020-12/"
	    "schema"));
	jv_set(s, "title", jv_str("ucas JSON frontend"));
	jv_set(s, "protocol", jv_int(PROTOCOL_VERSION));
	jv_set(s, "transport", jv_str("JSON Lines: `ucas --json` reads one "
	    "request object per line on stdin and writes one response object "
	    "per line on stdout; `ucas --json -e REQUEST` answers one"));
	defs = add_obj(s, "$defs", parse_const(schema_defs));
	/* request: one shape per command */
	req = jv_new(J_OBJ);
	one = jv_new(J_ARR);
	list = jv_new(J_ARR);
	for (k = 0; k < NCMDS; k++) {
		c = jv_new(J_OBJ);
		pr = jv_new(J_OBJ);
		rq = jv_new(J_ARR);
		a = jv_new(J_OBJ);
		jv_set(a, "const", jv_str(cmds[k].name));
		jv_set(pr, "cmd", a);
		a = jv_new(J_OBJ);
		jv_set(a, "type", jv_str("object"));
		ps = jv_new(J_OBJ);
		{
			struct jv *areq = jv_new(J_ARR);
			for (i = 0; i < MAXJP && cmds[k].p[i].name; i++) {
				jv_set(ps, cmds[k].p[i].name, type_schema(&cmds[k].p[i]));
				if (cmds[k].p[i].req && cmds[k].p[i].type != P_RING)
					jv_push(areq, jv_str(cmds[k].p[i].name));
			}
			for (i = 0; i < NOPTS; i++)
				jv_set(ps, opts[i].name, type_schema(&opts[i]));
			jv_set(a, "properties", ps);
			if (areq->n)
				jv_set(a, "required", areq);
			else
				jv_free(areq);
		}
		jv_set(a, "additionalProperties", jv_bool(0));
		jv_set(pr, "args", a);
		jv_set(pr, "id", jv_new(J_OBJ));
		jv_set(c, "properties", pr);
		jv_push(rq, jv_str("cmd"));
		jv_set(c, "required", rq);
		jv_push(one, c);
		/* the plain list */
		c = jv_new(J_OBJ);
		jv_set(c, "cmd", jv_str(cmds[k].name));
		jv_set(c, "description", jv_str(cmds[k].desc));
		if (cmds[k].stmt)
			jv_set(c, "ucas", jv_str(cmds[k].stmt));
		jv_set(c, "structured", jv_bool(cmds[k].structured));
		ps = jv_new(J_ARR);
		for (i = 0; i < MAXJP && cmds[k].p[i].name; i++) {
			pi = jv_new(J_OBJ);
			jv_set(pi, "name", jv_str(cmds[k].p[i].name));
			jv_set(pi, "type", jv_str(ptype_name[cmds[k].p[i].type]));
			jv_set(pi, "required", jv_bool(cmds[k].p[i].req));
			jv_set(pi, "description", jv_str(cmds[k].p[i].desc));
			if (cmds[k].p[i].values)
				jv_set(pi, "values", jv_str(cmds[k].p[i].values));
			jv_push(ps, pi);
		}
		jv_set(c, "params", ps);
		jv_push(list, c);
	}
	jv_set(req, "type", jv_str("object"));
	jv_set(req, "oneOf", one);
	jv_set(defs, "request", req);
	jv_set(defs, "response", parse_const(schema_response));
	{
		struct jv *o = jv_new(J_ARR);
		for (i = 0; i < NOPTS; i++) {
			pi = jv_new(J_OBJ);
			jv_set(pi, "name", jv_str(opts[i].name));
			jv_set(pi, "type", jv_str(ptype_name[opts[i].type]));
			jv_set(pi, "description", jv_str(opts[i].desc));
			jv_push(o, pi);
		}
		jv_set(s, "options", o);
	}
	jv_set(s, "commands", list);
	{
		struct jv *r = jv_new(J_OBJ);
		jv_set(r, "$ref", jv_str("#/$defs/request"));
		jv_set(s, "request", r);
		r = jv_new(J_OBJ);
		jv_set(r, "$ref", jv_str("#/$defs/response"));
		jv_set(s, "response", r);
	}
	return s;
}

/* ---- special commands ---- */

static int
cmd_define(struct req *r)
{
	const struct jv *ps, *w;
	const char *nm = jv_gets(r->args, "name");
	int k;

	if (!is_ident(nm))
		return bad(r, "bad_request", "name: not a name");
	sb_puts(&r->stmt, nm);
	if ((ps = jv_get(r->args, "params")) != NULL &&
	    !render_params(r, &r->stmt, ps))
		return 0;
	sb_puts(&r->stmt, " = ");
	if (!render_expr(r, &r->stmt, jv_get(r->args, "body"), "body", 1))
		return 0;
	if ((w = jv_get(r->args, "where")) != NULL)
		for (k = 0; k < w->n; k++)
			sb_printf(&r->stmt, "%s%s", k ? ", " : " where ", w->v[k]->s);
	return 1;
}

static int
cmd_limit(struct req *r)
{
	const char *side = jv_gets(r->args, "side");
	const struct jv *v;

	sb_puts(&r->stmt, "limit ");
	if (!render_func(r, &r->stmt, jv_get(r->args, "f"), "f"))
		return 0;
	if ((v = jv_get(r->args, "var")) != NULL) {
		if (v->t != J_STR || !is_ident(v->s))
			return bad(r, "bad_request", "var: a name");
		sb_printf(&r->stmt, " as %s -> ", v->s);
	} else
		sb_puts(&r->stmt, " at ");
	if (!render_expr(r, &r->stmt, jv_get(r->args, "to"), "to", 0))
		return 0;
	if (side)
		sb_puts(&r->stmt, side);
	return 1;
}

/* spirv_simplify given C (lang "c", or C by its content): c optimize */
static int spirv_is_c;

static int
cmd_spirv(struct req *r, const char **code)
{
	const struct jv *v, *rg;
	const char *lang = jv_gets(r->args, "lang");
	int k;

	*code = jv_gets(r->args, "code");
	spirv_is_c = (lang && !strcmp(lang, "c")) || (!lang && *code &&
	    code_lang(*code, "", NULL) == CL_C);
	if (spirv_is_c) {
		/* the options that mean the same; target, check: c_optimize */
		sb_puts(&r->stmt, "c optimize");
		if ((v = jv_get(r->args, "budget")) != NULL) {
			sb_puts(&r->stmt, " budget ");
			if (!render_num(r, &r->stmt, v, "budget"))
				return 0;
		}
		if ((v = jv_get(r->args, "ranges")) != NULL)
			for (k = 0; k < v->n; k++) {
				rg = v->v[k];
				if (!is_ident(v->k[k]) || rg->t != J_ARR || rg->n != 2 ||
				    rg->v[0]->t != J_NUM || rg->v[1]->t != J_NUM)
					return bad(r, "bad_request", "ranges: {\"name\": "
					    "[lo, hi]} with integers");
				sb_printf(&r->stmt, " assume %s %lld %lld", v->k[k],
				    (long long)rg->v[0]->num, (long long)rg->v[1]->num);
			}
		if ((v = jv_get(r->args, "nofile")) != NULL && v->t == J_TRUE)
			sb_puts(&r->stmt, " nofile");
		return 1;
	}
	if ((v = jv_get(r->args, "model")) != NULL && v->t == J_STR) {
		if (!is_ident(v->s))
			return bad(r, "bad_request", "model: a model's name");
		sb_printf(&r->stmt, "use %s; ", v->s);
	}
	sb_puts(&r->stmt, "spvsimplify");
	if ((v = jv_get(r->args, "budget")) != NULL) {
		sb_puts(&r->stmt, " budget ");
		if (!render_num(r, &r->stmt, v, "budget"))
			return 0;
	}
	if ((v = jv_get(r->args, "nofile")) != NULL && v->t == J_TRUE)
		sb_puts(&r->stmt, " nofile");
	if ((v = jv_get(r->args, "ranges")) != NULL)
		for (k = 0; k < v->n; k++) {
			rg = v->v[k];
			if (!is_ident(v->k[k]) || rg->t != J_ARR || rg->n != 2 ||
			    rg->v[0]->t != J_NUM || rg->v[1]->t != J_NUM)
				return bad(r, "bad_request", "ranges: {\"name\": [lo, "
				    "hi]} with integers");
			sb_printf(&r->stmt, " assume %s %lld %lld", v->k[k],
			    (long long)rg->v[0]->num, (long long)rg->v[1]->num);
		}
	if ((v = jv_get(r->args, "float_only")) != NULL && v->t == J_TRUE)
		sb_puts(&r->stmt, " float");
	if ((v = jv_get(r->args, "exact")) != NULL && v->t == J_TRUE)
		sb_puts(&r->stmt, " exact");
	if ((v = jv_get(r->args, "ulp")) != NULL) {
		sb_puts(&r->stmt, " ulp ");
		if (!render_num(r, &r->stmt, v, "ulp"))
			return 0;
	}
	if ((v = jv_get(r->args, "approx")) != NULL && v->t == J_TRUE)
		sb_puts(&r->stmt, " approx");
	if ((v = jv_get(r->args, "percomp")) != NULL && v->t == J_TRUE)
		sb_puts(&r->stmt, " percomp");
	if ((v = jv_get(r->args, "advice")) != NULL && v->t == J_TRUE)
		sb_puts(&r->stmt, " advice");
	if ((v = jv_get(r->args, "sink")) != NULL) {
		const char *q;
		if (v->t != J_STR || !v->s[0])
			return bad(r, "bad_request", "sink: a format's name");
		for (q = v->s; *q; q++)
			if (!isalnum((unsigned char)*q) && *q != '_')
				return bad(r, "bad_request", "sink: a format's name "
				    "(letters, digits, _)");
		sb_printf(&r->stmt, " sink %s", v->s);
	}
	if ((v = jv_get(r->args, "functions")) != NULL)
		sb_puts(&r->stmt, v->t == J_TRUE ? " functions" : " nofunctions");
	if ((v = jv_get(r->args, "float_ranges")) != NULL)
		for (k = 0; k < v->n; k++) {
			rg = v->v[k];
			if (!is_ident(v->k[k]) || rg->t != J_ARR || rg->n != 2 ||
			    rg->v[0]->t != J_NUM || rg->v[1]->t != J_NUM)
				return bad(r, "bad_request", "float_ranges: {\"name\": "
				    "[lo, hi]}");
			sb_printf(&r->stmt, " fassume %s ", v->k[k]);
			json_num(&r->stmt, rg->v[0]->num);
			sb_putc(&r->stmt, ' ');
			json_num(&r->stmt, rg->v[1]->num);
			sb_puts(&r->stmt, " 0 0");
		}
	if (lang && !strcmp(lang, "glsl"))
		code_block_lang[0] = 0;
	return 1;
}

/* matrix: "OP M [x = RHS]" and the ring */
static int
cmd_matrix(struct req *r)
{
	const char *op = jv_gets(r->args, "op");
	const struct jv *rhs = jv_get(r->args, "rhs");
	int k;

	if (op == NULL)
		op = "matrix";
	if (!strcmp(op, "solve") != (rhs != NULL))
		return bad(r, "bad_request", "matrix: rhs goes with op solve");
	sb_puts(&r->stmt, op);
	for (k = 0; k < MAXJP && r->c->p[k].name; k++) {
		const struct jparam *p = &r->c->p[k];
		if (p->type == P_RING) {
			if (!ring_arg(r, p))
				return 0;
		} else if (!strcmp(p->name, "matrix")) {
			sb_putc(&r->stmt, ' ');
			if (!render_arg(r, &r->stmt, p, jv_get(r->args, "matrix")))
				return 0;
		} else if (!strcmp(p->name, "rhs") && rhs) {
			sb_puts(&r->stmt, " x = ");
			if (!render_arg(r, &r->stmt, p, rhs))
				return 0;
		}
	}
	return 1;
}

/* infinite_sum/product: sum(n, lo, inf, f) */
static int
cmd_infinite(struct req *r)
{
	const struct jv *lo = jv_get(r->args, "from");

	sb_printf(&r->stmt, "%s(%s, ", r->c->name[9] == 's' ? "sum" : "prod",
	    jv_gets(r->args, "index"));
	if (lo) {
		if (!render_num(r, &r->stmt, lo, "from"))
			return 0;
	} else
		sb_puts(&r->stmt, "1");
	sb_puts(&r->stmt, ", inf, ");
	if (!render_expr(r, &r->stmt, jv_get(r->args, "f"), "f", 1))
		return 0;
	sb_putc(&r->stmt, ')');
	return 1;
}

/* ---- one request ---- */

static const struct jcmd *
find_cmd(const char *name)
{
	int k;

	for (k = 0; k < NCMDS; k++)
		if (!strcmp(cmds[k].name, name))
			return &cmds[k];
	return NULL;
}

static const char *
note_kind(const char *t)
{
	if (strstr(t, "read as") || strstr(t, "taken as") ||
	    strstr(t, "is read") || strstr(t, "inferred") ||
	    strstr(t, "is power") || strstr(t, "bind tighter"))
		return "rule";
	return "info";
}

/* notes the text has that the report does not */
static void
scan_notes(struct jv *notes, const char *text)
{
	const char *p = text, *e, *s;
	struct jv *o;
	int i, dup;
	char buf[1024];

	while (*p) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		for (s = p; s < e && *s == ' '; s++)
			;
		if (e - s > 6 && !strncmp(s, "note: ", 6)) {
			s += 6;
			snprintf(buf, sizeof buf, "%.*s", (int)(e - s), s);
			for (i = 0, dup = 0; i < notes->n; i++)
				if (!strcmp(jv_gets(notes->v[i], "text"), buf))
					dup = 1;
			if (!dup) {
				o = jv_new(J_OBJ);
				jv_set(o, "kind", jv_str(note_kind(buf)));
				jv_set(o, "text", jv_str(buf));
				jv_push(notes, o);
			}
		}
		p = *e ? e + 1 : e;
	}
}

/* mcp.c: responses go through it (a JSON-RPC result instead) */
void (*json_respond_hook)(struct jv *resp);

static void
respond(struct jv *resp)
{
	struct sbuf b = {0};

	if (json_respond_hook) {
		json_respond_hook(resp);
		return;
	}
	jv_write(&b, resp);
	sb_putc(&b, '\n');
	fputs(b.p, jout);
	fflush(jout);
	free(b.p);
}

static struct jv *
error_response(const struct jv *id, const char *cmd, const char *code,
    const char *msg)
{
	struct jv *o = jv_new(J_OBJ), *e = jv_new(J_OBJ);
	struct sbuf b = {0};

	if (id) {   /* a copy */
		jv_write(&b, id);
		jv_set(o, "id", parse_const(b.p));
		free(b.p);
	} else
		jv_set(o, "id", jv_new(J_NULL));
	jv_set(o, "ok", jv_bool(0));
	jv_set(o, "cmd", cmd ? jv_str(cmd) : jv_new(J_NULL));
	jv_set(e, "code", jv_str(code));
	jv_set(e, "message", jv_str(msg));
	jv_set(o, "error", e);
	return o;
}

static int reset_wanted, oneshot;

static struct jv *
handle(const struct jv *rq)
{
	struct req r;
	const struct jv *id, *a, *v;
	const char *cmd, *code = NULL;
	struct jv *resp, *pieces, *res, *results, *notes;
	char *text = NULL;
	int fmode = default_fmode, defaults = 0, echo = 0, k;
	struct sbuf src = {0};

	memset(&r, 0, sizeof r);
	id = rq->t == J_OBJ ? jv_get(rq, "id") : NULL;
	if (rq->t != J_OBJ)
		return error_response(NULL, NULL, "bad_request", "a request is a "
		    "JSON object {\"id\", \"cmd\", \"args\"}");
	for (k = 0; k < rq->n; k++)
		if (strcmp(rq->k[k], "id") && strcmp(rq->k[k], "cmd") &&
		    strcmp(rq->k[k], "args")) {
			char m[200];
			snprintf(m, sizeof m, "unknown field \"%s\" (a request has "
			    "id, cmd, args)", rq->k[k]);
			return error_response(id, NULL, "bad_request", m);
		}
	if ((cmd = jv_gets(rq, "cmd")) == NULL)
		return error_response(id, NULL, "bad_request", "\"cmd\" (a string) "
		    "is missing");
	if ((r.c = find_cmd(cmd)) == NULL) {
		char m[200];
		snprintf(m, sizeof m, "unknown command \"%.64s\" (\"schema\" lists "
		    "them)", cmd);
		return error_response(id, cmd, "unknown_command", m);
	}
	a = jv_get(rq, "args");
	if (a && a->t != J_OBJ)
		return error_response(id, cmd, "bad_request", "\"args\" is an "
		    "object");
	r.args = a;
	if ((v = jv_get(a, "files")) != NULL && v->t == J_STR)
		fmode = !strcmp(v->s, "base64") ? REP_FILES_BASE64 :
		    !strcmp(v->s, "both") ? REP_FILES_WRITE | REP_FILES_BASE64 :
		    !strcmp(v->s, "none") ? 0 : REP_FILES_WRITE;
	defaults = (v = jv_get(a, "defaults")) != NULL && v->t == J_TRUE;
	if ((v = jv_get(a, "assume")) != NULL && v->t != J_OBJ)
		return error_response(id, cmd, "bad_request", "args.assume is an "
		    "object {key: reading}");

	rep_request(jv_get(a, "assume"), session, defaults, fmode, outdir);
	nlp_interactive = 0;
	nlp_asked = 0;
	img_gallery_clear();
	check_args(&r);
	if (!r.ecode) {
		code_block = NULL;
		code_block_lang[0] = 0;
		code_block_used = 0;
		if (!strcmp(cmd, "define"))
			cmd_define(&r);
		else if (!strcmp(cmd, "limit"))
			cmd_limit(&r);
		else if (!strcmp(cmd, "spirv_simplify"))
			cmd_spirv(&r, &code);
		else if (!strcmp(cmd, "matrix"))
			cmd_matrix(&r);
		else if (!strcmp(cmd, "infinite_sum") ||
		    !strcmp(cmd, "infinite_product"))
			cmd_infinite(&r);
		else if (!strcmp(cmd, "ucas")) {
			const char *s = jv_gets(a, "source");
			sb_puts(&r.stmt, s);
			code = jv_gets(a, "code");
			echo = strchr(s, '\n') != NULL;
		} else if (!strcmp(cmd, "english")) {
			const struct jv *ans = jv_get(a, "answers");
			for (k = 0; ans && k < ans->n; k++)
				sb_printf(&r.stmt, "\x1e" "answer %s\n", ans->v[k]->s);
			sb_puts(&r.stmt, jv_gets(a, "text"));
			if (jv_gets(a, "code") != NULL)	/* as a fenced block */
			{
				sb_printf(&r.stmt, "\n```%s\n", jv_gets(a, "lang") ?
				    jv_gets(a, "lang") : "");
				sb_puts(&r.stmt, jv_gets(a, "code"));
				sb_puts(&r.stmt, "\n```");
			}
		} else if (!r.c->special) {
			build(&r);
			if (!strncmp(cmd, "c_", 2))	/* cfront: C source */
				code = jv_gets(a, "code");
			if (!strcmp(cmd, "animate") && (code = jv_gets(a, "code")) &&
			    !r.ecode && !jv_get(a, "spec")) {
				/* a shader: exec is what it can be */
				struct sbuf t = {0};
				sb_printf(&t, "animate exec%s", r.stmt.p + 7);
				free(r.stmt.p);
				r.stmt = t;
			}
		}
	}
	if (r.ecode) {
		rep_error(r.ecode, r.err);   /* unless an ambiguity is there */
		pieces = rep_collect();
		resp = error_response(id, cmd, r.ecode, r.err);
		for (k = 0; k < pieces->n; k++)   /* the error object, moved */
			if (!strcmp(pieces->k[k], "error")) {
				jv_set(resp, "error", pieces->v[k]);
				pieces->v[k] = jv_new(J_NULL);
			}
		jv_free(pieces);
		free(r.pre.p);
		free(r.stmt.p);
		return resp;
	}

	/* run */
	sb_puts(&src, r.pre.p ? r.pre.p : "");
	sb_puts(&src, r.stmt.p ? r.stmt.p : "");
	resp = jv_new(J_OBJ);
	{
		struct sbuf b = {0};
		if (id) {
			jv_write(&b, id);
			jv_set(resp, "id", parse_const(b.p));
			free(b.p);
		} else
			jv_set(resp, "id", jv_new(J_NULL));
	}
	jv_set(resp, "ok", jv_bool(1));
	jv_set(resp, "cmd", jv_str(cmd));
	cap_begin();
	if (!strcmp(cmd, "schema")) {
		rep_begin("schema");
		res = schema();
	} else if (!strcmp(cmd, "state") || !strcmp(cmd, "reset") ||
	    !strcmp(cmd, "assume")) {
		res = NULL;
	} else {
		static struct sbuf glsl;
		res = NULL;
		if (code) {
			code_block = code;
			int as_c = !strncmp(cmd, "c_", 2) ||
			    (!strcmp(cmd, "spirv_simplify") && spirv_is_c);
			if (as_c && strncmp(cmd, "c_", 2))
				printf("  note: the code is C: optimised by the C frontend "
				    "(c optimize; the c_optimize tool also takes a target "
				    "and check)\n");
			if (!as_c && glsl_like(code, "")) {
				struct sbuf nt = {0};
				sb_clear(&glsl);
				if (glsl_to_spvasm(code, &glsl, &nt)) {
					printf("  note: the code was compiled to SPIR-V "
					    "with glslangValidator\n%s", nt.p ? nt.p : "");
					code_block = glsl.p;
				} else
					printf("  note: the GLSL did not compile:\n%s",
					    nt.p ? nt.p : "");
				free(nt.p);
			}
			snprintf(code_block_lang, sizeof code_block_lang, "%s", as_c ?
			    "c" : "spv");
		}
		jv_set(resp, "ucas", jv_str(sb_str(&src)));
		if (!strcmp(cmd, "english")) {
			jmp_buf saved;
			memcpy(saved, fail_jmp, sizeof saved);
			nlp_asked = 0;
			if (setjmp(fail_jmp) == 0)
				run_text(src.p, "<english>", 0);
			else
				rep_failed();
			memcpy(fail_jmp, saved, sizeof saved);
		} else
			run_stmts(src.p, echo);
		code_block = NULL;
	}
	text = cap_end();
	text = rep_text_cap(text);
	rep_text_fill(text);
	pieces = rep_collect();
	if (!strcmp(cmd, "schema"))
		;
	else if (!strcmp(cmd, "state")) {
		struct jv *fs = jv_new(J_ARR), *ms = jv_new(J_ARR),
		    *bs = jv_new(J_ARR);
		struct model *m;
		struct basis *bb;
		res = jv_new(J_OBJ);
		jv_set(res, "kind", jv_str("state"));
		for (k = 0; k < nfuncs; k++)
			if (!funcs[k].anon && sym_str(funcs[k].name)[0] != '_')
				jv_push(fs, func_json(&funcs[k], NULL));
		jv_set(res, "functions", fs);
		for (k = 0; (m = model_nth(k)) != NULL; k++)
			jv_push(ms, jv_str(sym_str(m->name)));
		jv_set(res, "models", ms);
		jv_set(res, "active_model", model_active() ?
		    jv_str(sym_str(model_active()->name)) : jv_new(J_NULL));
		for (k = 0; (bb = basis_nth(k)) != NULL; k++)
			jv_push(bs, jv_str(sym_str(bb->name)));
		jv_set(res, "bases", bs);
		{
			struct sbuf b = {0};
			jv_write(&b, session);
			jv_set(res, "assumptions", parse_const(b.p));
			free(b.p);
		}
	} else if (!strcmp(cmd, "assume")) {
		const struct jv *set = jv_get(a, "set"), *cl = jv_get(a, "clear");
		struct sbuf b = {0};
		if (cl && cl->t == J_TRUE) {
			jv_free(session);
			session = jv_new(J_OBJ);
		}
		for (k = 0; set && k < set->n; k++) {
			sb_clear(&b);
			jv_write(&b, set->v[k]);
			jv_set(session, set->k[k], parse_const(b.p));
		}
		sb_clear(&b);
		jv_write(&b, session);
		res = jv_new(J_OBJ);
		jv_set(res, "kind", jv_str("assume"));
		jv_set(res, "assumptions", parse_const(b.p));
		free(b.p);
	} else if (!strcmp(cmd, "reset")) {
		res = jv_new(J_OBJ);
		jv_set(res, "kind", jv_str("reset"));
		reset_wanted = 1;
	} else if (!strcmp(cmd, "define")) {
		struct func *f = func_find(sym_cstr(jv_gets(a, "name")));
		if (f && !jv_get(pieces, "error"))
			res = func_json(f, "define");
	} else if (!strcmp(cmd, "english")) {
		/* the translated statements, as data */
		const char *p = text, *e, *s;
		struct jv *st = jv_new(J_ARR);
		res = jv_new(J_OBJ);
		jv_set(res, "kind", jv_str("english"));
		while (*p) {
			e = strchr(p, '\n');
			if (e == NULL)
				e = p + strlen(p);
			for (s = p; s < e && *s == ' '; s++)
				;
			if (e - s > 6 && !strncmp(s, "ucas: ", 6))
				jv_push(st, jv_strn(s + 6, (size_t)(e - s - 6)));
			if (e - s > 7 && !strncmp(s, "sorry: ", 7) &&
			    !jv_get(pieces, "error")) {
				struct jv *er = jv_new(J_OBJ);
				jv_set(er, "code", jv_str("not_understood"));
				jv_set(er, "message", jv_strn(s + 7, (size_t)(e - s - 7)));
				jv_set(pieces, "error", er);
			}
			p = *e ? e + 1 : e;
		}
		jv_set(res, "statements", st);
		jv_set(res, "understood", jv_bool(st->n > 0));
		results = jv_get(pieces, "results");
		if (results && results->n) {   /* what the statements reported */
			struct sbuf b = {0};
			jv_write(&b, results);
			jv_set(res, "results", parse_const(b.p));
			free(b.p);
		}
	}
	results = jv_get(pieces, "results");
	if (res == NULL && results && results->n > 0) {
		struct sbuf b = {0};
		jv_write(&b, results->v[results->n - 1]);
		res = parse_const(b.p);
		free(b.p);
	}
	jv_set(resp, "structured", jv_bool(res != NULL));
	if (res == NULL && !jv_get(pieces, "error")) {
		res = jv_new(J_OBJ);
		jv_set(res, "text", jv_str(text));
	}
	if (res) {
		cap_lists(res);
		jv_set(resp, "result", res);
	}
	if (results && results->n > 1 && strcmp(cmd, "english")) {
		struct sbuf b = {0};
		jv_write(&b, results);
		jv_set(resp, "results", parse_const(b.p));
		free(b.p);
	}
	/* the rest of the pieces, moved over */
	notes = jv_get(pieces, "notes");
	scan_notes(notes, text);
	for (k = 0; k < pieces->n; k++) {
		const char *key = pieces->k[k];
		if (!strcmp(key, "results"))
			continue;
		if (!strcmp(key, "error"))
			jv_set(resp, "ok", jv_bool(0));
		jv_set(resp, key, pieces->v[k]);
		pieces->v[k] = jv_new(J_NULL);
	}
	jv_set(resp, "text", jv_str(text));
	/* the error last, for reading */
	for (k = 0; k < resp->n; k++)
		if (!strcmp(resp->k[k], "error") && k != resp->n - 1) {
			struct jv *e = resp->v[k];
			char *kk = resp->k[k];
			memmove(resp->v + k, resp->v + k + 1,
			    (size_t)(resp->n - k - 1) * sizeof *resp->v);
			memmove(resp->k + k, resp->k + k + 1,
			    (size_t)(resp->n - k - 1) * sizeof *resp->k);
			resp->v[resp->n - 1] = e;
			resp->k[resp->n - 1] = kk;
			break;
		}
	jv_free(pieces);
	free(text);
	free(src.p);
	free(r.pre.p);
	free(r.stmt.p);
	return resp;
}

/* ---- the process: timeouts, reset ---- */

static void
self_exec(void)
{
	char path[PATH_MAX];
	int ok = 0;

	fflush(NULL);
#ifdef __FreeBSD__
	{
		int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
		size_t len = sizeof path;
		ok = sysctl(mib, 4, path, &len, NULL, 0) == 0;
	}
#else
	{
		ssize_t n = readlink("/proc/self/exe", path, sizeof path - 1);
		if (n > 0) {
			path[n] = 0;
			ok = 1;
		}
	}
#endif
	if (ok)
		execv(path, sargv);
	execvp(sargv[0], sargv);
	fprintf(stderr, "ucas: reset: cannot restart: %s\n", strerror(errno));
	_exit(1);
}

static int chained;

/*
 * The first request with a timeout: this process stays as the waiter
 * (the pid the client knows) and a child carries on as the server.  The
 * waiter exits when the last worker has closed the lifeline.
 */
static void
chain_start(void)
{
	int lp[2];
	pid_t pid;
	char c;

	if (chained || pipe(lp) != 0)
		return;
	fflush(NULL);
	pid = fork();
	if (pid < 0) {
		close(lp[0]);
		close(lp[1]);
		return;
	}
	if (pid > 0) {
		close(lp[1]);
		close(0);
		while (read(lp[0], &c, 1) > 0 || errno == EINTR)
			;
		while (waitpid(-1, NULL, WNOHANG) > 0)
			;
		_exit(0);
	}
	close(lp[0]);
	chained = 1;
}

/* 1: this process is the child that ran the request (the new server) */
static int
run_with_timeout(const struct jv *rq, double secs)
{
	int sp[2], st;
	pid_t pid;
	struct pollfd pf;
	struct timespec t0, t1;
	char c;
	struct jv *resp;
	double left;

	chain_start();
	if (pipe(sp) != 0)
		return -1;
	fflush(NULL);
	pid = fork();
	if (pid < 0) {
		close(sp[0]);
		close(sp[1]);
		return -1;
	}
	if (pid == 0) {
		close(sp[0]);
		resp = handle(rq);
		respond(resp);
		jv_free(resp);
		(void)!write(sp[1], "k", 1);
		close(sp[1]);
		if (reset_wanted && !oneshot)
			self_exec();
		return 1;
	}
	close(sp[1]);
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (;;) {
		clock_gettime(CLOCK_MONOTONIC, &t1);
		left = secs - (t1.tv_sec - t0.tv_sec) -
		    (t1.tv_nsec - t0.tv_nsec) / 1e9;
		if (left <= 0) {
			kill(pid, SIGKILL);
			waitpid(pid, NULL, 0);
			close(sp[0]);
			{
				char m[120];
				snprintf(m, sizeof m, "timed out after %g s; the state "
				    "is unchanged", secs);
				resp = error_response(jv_get(rq, "id"),
				    jv_gets(rq, "cmd"), "timeout", m);
			}
			respond(resp);
			jv_free(resp);
			return 0;
		}
		pf.fd = sp[0];
		pf.events = POLLIN;
		if (poll(&pf, 1, (int)(left * 1000) + 1) <= 0)
			continue;
		if (read(sp[0], &c, 1) == 1)
			_exit(0);   /* the child is the server now */
		/* closed without the byte: it died */
		waitpid(pid, &st, 0);
		close(sp[0]);
		{
			char m[120];
			snprintf(m, sizeof m, "the evaluation crashed (%s %d); the "
			    "state is unchanged", WIFSIGNALED(st) ? "signal" :
			    "status", WIFSIGNALED(st) ? WTERMSIG(st) :
			    WEXITSTATUS(st));
			resp = error_response(jv_get(rq, "id"), jv_gets(rq, "cmd"),
			    "internal", m);
		}
		respond(resp);
		jv_free(resp);
		return 0;
	}
}

/*
 * One request, answered through respond (under timeout_s in a forked
 * child, which then carries on as the server).  rq stays the caller's.
 */
void
json_serve(const struct jv *rq)
{
	struct jv *resp, *t, *a;

	a = jv_get(rq, "args");
	t = jv_get(a, "timeout_s");
	if (t && t->t == J_NUM && t->num > 0) {
		if (run_with_timeout(rq, t->num) >= 0)
			return;
	}
	resp = handle(rq);
	respond(resp);
	jv_free(resp);
	if (reset_wanted && !oneshot)
		self_exec();
	reset_wanted = 0;
}

static void
serve_line(const char *line)
{
	char err[160];
	size_t pos;
	struct jv *rq, *resp;

	rq = jv_parse(line, err, sizeof err, &pos);
	if (rq == NULL) {
		char m[240];
		snprintf(m, sizeof m, "%s at byte %zu", err, pos);
		resp = error_response(NULL, NULL, "bad_json", m);
		respond(resp);
		jv_free(resp);
		return;
	}
	json_serve(rq);
	jv_free(rq);
}

/* a line from fd 0, unbuffered: nothing is read ahead (fork, exec) */
char *
json_read_line(void)
{
	static struct sbuf b;
	char c;
	ssize_t n;

	sb_clear(&b);
	sb_puts(&b, "");
	for (;;) {
		n = read(0, &c, 1);
		if (n < 0 && errno == EINTR)
			continue;
		if (n <= 0)
			return b.n ? b.p : NULL;
		if (c == '\n')
			return b.p;
		sb_putc(&b, c);
	}
}

int
json_main(int argc, char **argv, int i)
{
	const char *one = NULL;
	char *line;
	int fd;

	sargc = argc;
	sargv = argv;
	session = jv_new(J_OBJ);
	if (!strcmp(argv[i], "--json-schema")) {
		struct jv *s = schema();
		struct sbuf b = {0};
		jv_write(&b, s);
		printf("%s\n", b.p);
		return 0;
	}
	for (i++; i < argc; i++) {
		if (!strcmp(argv[i], "-e") && i + 1 < argc)
			one = argv[++i];
		else if (!strcmp(argv[i], "--dir") && i + 1 < argc)
			snprintf(outdir, sizeof outdir, "%s", argv[++i]);
		else if (!strcmp(argv[i], "--files") && i + 1 < argc) {
			const char *m = argv[++i];
			default_fmode = !strcmp(m, "base64") ? REP_FILES_BASE64 :
			    !strcmp(m, "both") ? REP_FILES_WRITE | REP_FILES_BASE64 :
			    !strcmp(m, "none") ? 0 : REP_FILES_WRITE;
		} else {
			fprintf(stderr, "usage: ucas --json [--dir DIR] [--files "
			    "write|base64|both|none] [-e REQUEST]\n");
			return 2;
		}
	}
	/* responses go to the real stdout; fd 1 is swapped per request */
	fd = dup(1);
	jout = fdopen(fd, "w");
	if (jout == NULL)
		return 2;
	nlp_interactive = 0;
	signal(SIGPIPE, SIG_DFL);
	run_source("", "<prelude>", 0);   /* the prelude, outside requests */
	if (one) {
		oneshot = 1;
		serve_line(one);
		return 0;
	}
	while ((line = json_read_line()) != NULL) {
		const char *s = line;
		while (*s == ' ' || *s == '\t' || *s == '\r')
			s++;
		if (!*s)
			continue;
		serve_line(line);
	}
	return 0;
}

/* ---- for mcp.c: the server's setup and the commands as data ---- */

/* the process' setup for serving: argv for reset, stdout for answers */
int
json_setup(int argc, char **argv, int fmode)
{
	int fd;

	sargc = argc;
	sargv = argv;
	session = jv_new(J_OBJ);
	default_fmode = fmode;
	fd = dup(1);
	jout = fdopen(fd, "w");
	if (jout == NULL)
		return -1;
	nlp_interactive = 0;
	signal(SIGPIPE, SIG_DFL);
	run_source("", "<prelude>", 0);
	return 0;
}

FILE *
json_out(void)
{
	return jout;
}

int
json_ncmds(void)
{
	return NCMDS;
}

const char *
json_cmd_name(int k)
{
	return cmds[k].name;
}

const char *
json_cmd_desc(int k)
{
	return cmds[k].desc;
}

/* "f*, var, from, to": the parameters, * when required */
void
json_cmd_params(int k, struct sbuf *b)
{
	int i;

	for (i = 0; i < MAXJP && cmds[k].p[i].name; i++)
		sb_printf(b, "%s%s%s", i ? ", " : "", cmds[k].p[i].name,
		    cmds[k].p[i].req && cmds[k].p[i].type != P_RING ? "*" : "");
}

/* a command's args as a JSON Schema object; with defs, self-contained */
struct jv *
json_cmd_schema(int k, int defs)
{
	struct jv *a = jv_new(J_OBJ), *ps = jv_new(J_OBJ),
	    *areq = jv_new(J_ARR);
	int i;

	jv_set(a, "type", jv_str("object"));
	for (i = 0; i < MAXJP && cmds[k].p[i].name; i++) {
		jv_set(ps, cmds[k].p[i].name, type_schema(&cmds[k].p[i]));
		if (cmds[k].p[i].req && cmds[k].p[i].type != P_RING)
			jv_push(areq, jv_str(cmds[k].p[i].name));
	}
	for (i = 0; i < NOPTS; i++)
		jv_set(ps, opts[i].name, type_schema(&opts[i]));
	jv_set(a, "properties", ps);
	if (areq->n)
		jv_set(a, "required", areq);
	else
		jv_free(areq);
	jv_set(a, "additionalProperties", jv_bool(0));
	if (defs)
		jv_set(a, "$defs", parse_const(schema_defs));
	return a;
}

struct jv *
json_response_schema(void)
{
	return parse_const(schema_response);
}

struct jv *
json_schema(void)
{
	return schema();
}

/*
 * The Discord bot's JSON path (discord/bot.c, "JSON over Discord"): body
 * is a request or an array of them, as the bot passed them on (files set
 * to base64, GLSL already compiled); each is answered by handle() and the
 * response, or the array of them, is printed as one line after a
 * "\x1ejson" line.  This runs in the bot's sandboxed channel worker: no
 * file system, no exec, no timeout_s fork (the bot's limits apply), and
 * reset and schema are the bot's own.  The worker's text-path state
 * (nlp_defer) is set aside meanwhile, so the answers are those of
 * `ucas --json`: an ambiguity is an error with its readings, never a
 * question.
 */
void
json_bot_serve(const char *body)
{
	struct jv *rq, *out;
	struct sbuf b = {0};
	char err[160];
	size_t pos;
	int k, defer = nlp_defer, inter = nlp_interactive,
	    small = rep_small_files;

	if (session == NULL)
		session = jv_new(J_OBJ);
	default_fmode = REP_FILES_BASE64;
	rep_small_files = 1;
	nlp_interactive = 0;
	nlp_defer = 0;
	spvcross_defer = 1;
	rq = jv_parse(body, err, sizeof err, &pos);
	if (rq == NULL) {
		char m[240];
		snprintf(m, sizeof m, "%s at byte %zu", err, pos);
		out = error_response(NULL, NULL, "bad_json", m);
	} else if (rq->t == J_ARR) {
		out = jv_new(J_ARR);
		for (k = 0; k < rq->n; k++)
			jv_push(out, handle(rq->v[k]));
	} else
		out = handle(rq);
	reset_wanted = 0;
	jv_write(&b, out);
	fflush(stdout);
	printf("\x1e" "json\n%s\n", b.p ? b.p : "null");
	fflush(stdout);
	free(b.p);
	jv_free(out);
	jv_free(rq);
	nlp_defer = defer;
	nlp_interactive = inter;
	rep_small_files = small;
	spvcross_defer = 0;
}
