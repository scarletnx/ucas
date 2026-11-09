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
	P_SOURCE	/* ucas source, several lines allowed */
};

static const char *const ptype_name[] = {"function", "expression",
	"expressions", "name", "names", "names", "integer", "number",
	"boolean", "enum", "string", "interval", "substitutions",
	"parameters", "strings", "object", "source"};

struct jparam {
	const char *name;
	int type;
	int req;
	const char *frag;	/* printf-like, one %s for the value */
	const char *desc;
	const char *values;	/* P_ENUM: "a|b|c" */
};

#define MAXJP 8
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
	    "(needs_input), in order", NULL}}},
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
	{"implicit", "implicit", 0, 0, "dy/dx of an implicit curve",
	    {E("equation", " %s", "L = R"), NMO("for", " for %s", "y"),
	    NMO("wrt", " wrt %s", "x"), IO("order", " order %s", "1 or 2")}},
	{"integrate", "integrate", 1, 0, "antiderivative (checked by "
	    "differentiating back) or definite integral",
	    {FN, NMO("var", " wrt %s", "the variable of integration"),
	    {"from", P_NUM, 0, " from %s", "lower bound (with to)", NULL},
	    {"to", P_NUM, 0, " to %s", "upper bound (with from)", NULL},
	    TO("around", " around %s", "a contour for a complex integrand: "
	    "\"|z| = 1\"")}},
	{"antiderivative", "antiderivative", 1, 0, "an antiderivative + C",
	    {FN, NMO("var", " wrt %s", "the variable")}},
	/* compare, search */
	{"equiv", "equiv", 1, 0, "equal on F's domain? (with a counterexample)",
	    {FN, F("g", " %s", "the other function")}},
	{"find", "find", 1, 0, "cheaper implementations under the cost model",
	    {FN, IO("n", " %s", "how many (5)"), IO("max", " max %s",
	    "maximum cost")}},
	{"more", "more", 0, 0, "the next results of the last find",
	    {IO("n", " %s", "how many")}},
	{"fit", "fit", 1, 0, "choose G's extra parameters to approximate F",
	    {F("g", " %s", "the approximation, with extra parameters"),
	    F("f", " ~ %s", "the target"), {"metric", P_ENUM, 0, " metric %s",
	    "the error to minimise", "maxrel|maxabs|rms"}}},
	{"error", "error", 1, 0, "error of G as an approximation of F",
	    {F("g", " %s", "the approximation"), F("f", " ~ %s",
	    "the target")}},
	{"cost", "cost", 1, 0, "price under a cost model",
	    {FN, NMO("model", " model %s", "the model (the active one)")}},
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
	    {FN, NMO("model", " model %s", "the model")}},
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
	{"nthroots", "nthroots", 0, 0, "the N complex N-th roots",
	    {IR("n", " %s", "N"), FN}},
	{"unity", "unity", 0, 0, "the N-th roots of unity",
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
	{"parity", "parity", 0, 0, "even/odd, symmetries", {FN}},
	{"bounded", "bounded", 0, 0, "bounds, poles, growth", {FN}},
	{"zeros", "zeros", 1, 0, "the zeros (exact for polynomials)", {FN}},
	{"injective", "injective", 0, 0, "one to one? (with a witness)",
	    {FN}},
	{"surjective", "surjective", 0, 0, "onto? (with a value never taken)",
	    {FN, IVO("onto", " onto %s", "the codomain")}},
	{"bijective", "bijective", 0, 0, "a bijection?", {FN}},
	{"continuity", "continuity", 0, 0, "discontinuities by kind", {FN}},
	{"convexity", "convexity", 0, 0, "convex/concave intervals", {FN}},
	{"asymptotes", "asymptotes", 0, 0, "vertical, horizontal, oblique",
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
	    "execution (files)", {TO("spec", " %s", "as in the CLI: \"hillis "
	    "scan n 8\", \"exec of 32-bit x*x + 1\""),
	    {"code", P_SOURCE, 0, "", "SPIR-V assembly for animate exec",
	    NULL}}},
	{"spirv_simplify", NULL, 1, 1, "cheaper forms of a SPIR-V module's "
	    "integer instructions, each proved by SAT",
	    {{"code", P_SOURCE, 1, "", "SPIR-V assembly or GLSL", NULL},
	    {"lang", P_ENUM, 0, "", "the code's language (by default from "
	    "the code)", "spv|glsl"},
	    NUO("budget", "", "seconds (6)"),
	    {"ranges", P_OBJ, 0, "", "{\"n\": [lo, hi]}: values assumed for "
	    "an input", NULL},
	    BO("nofile", "", "a checksum instead of the rewritten module")}},
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
				    "params[].type (real, int, f32, f64, u8, i32, bool, "
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
	case P_OBJ:
		if (v->t != J_OBJ)
			return bad(r, "bad_request", "%s: an object", p->name);
		return 1;
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
		if (c->p[k].req && jv_get(r->args, c->p[k].name) == NULL)
			ok = bad(r, "bad_request", "%s needs \"%s\" (%s)", c->name,
			    c->p[k].name, c->p[k].desc);
	free(tmp.p);
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
		if ((v = jv_get(r->args, p->name)) == NULL)
			continue;
		if (p->type == P_BOOL) {
			if (v->t == J_TRUE)
				sb_puts(&r->stmt, p->frag);
			continue;
		}
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

static void
cap_begin(void)
{
	fflush(stdout);
	cap_fp = tmpfile();
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

	fflush(stdout);
	sb_puts(&b, "");
	if (cap_fp == NULL)
		return b.p;
	dup2(cap_saved, 1);
	close(cap_saved);
	cap_saved = -1;
	rewind(cap_fp);
	while ((n = fread(buf, 1, sizeof buf, cap_fp)) > 0) {
		size_t i;
		for (i = 0; i < n; i++)
			sb_putc(&b, buf[i]);
	}
	fclose(cap_fp);
	cap_fp = NULL;
	return b.p;
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
func_json(const struct func *f)
{
	struct jv *o = jv_new(J_OBJ), *ps = jv_new(J_ARR), *p, *rg;
	struct sbuf sig = {0};
	int k;

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
				sb_printf(&sig, " in %.17g..%.17g", pp->rlo, pp->rhi);
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
"\"#/$defs/name\"}, \"type\": {\"description\": \"real int bool f32 f64 "
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
				if (cmds[k].p[i].req)
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

static int
cmd_spirv(struct req *r, const char **code)
{
	const struct jv *v, *rg;
	const char *lang = jv_gets(r->args, "lang");
	int k;

	*code = jv_gets(r->args, "code");
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
	if (lang && !strcmp(lang, "glsl"))
		code_block_lang[0] = 0;
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

static void
respond(struct jv *resp)
{
	struct sbuf b = {0};

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
		} else if (!r.c->special)
			build(&r);
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
			if (glsl_like(code, "")) {
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
			snprintf(code_block_lang, sizeof code_block_lang, "spv");
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
				jv_push(fs, func_json(&funcs[k]));
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
		if (f && !jv_get(pieces, "error")) {
			res = func_json(f);
			jv_set(res, "kind", jv_str("define"));
		}
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
	if (res)
		jv_set(resp, "result", res);
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

static void
serve_line(const char *line)
{
	char err[160];
	size_t pos;
	struct jv *rq, *resp, *t, *a;

	rq = jv_parse(line, err, sizeof err, &pos);
	if (rq == NULL) {
		char m[240];
		snprintf(m, sizeof m, "%s at byte %zu", err, pos);
		resp = error_response(NULL, NULL, "bad_json", m);
		respond(resp);
		jv_free(resp);
		return;
	}
	a = jv_get(rq, "args");
	t = jv_get(a, "timeout_s");
	if (t && t->t == J_NUM && t->num > 0) {
		if (run_with_timeout(rq, t->num) >= 0) {
			jv_free(rq);
			return;
		}
	}
	resp = handle(rq);
	respond(resp);
	jv_free(resp);
	jv_free(rq);
	if (reset_wanted && !oneshot)
		self_exec();
	reset_wanted = 0;
}

/* a line from fd 0, unbuffered: nothing is read ahead (fork, exec) */
static char *
read_line(void)
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
	while ((line = read_line()) != NULL) {
		const char *s = line;
		while (*s == ' ' || *s == '\t' || *s == '\r')
			s++;
		if (!*s)
			continue;
		serve_line(line);
	}
	return 0;
}
