/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * adamfit.c - cheap approximations by a search over expression shapes,
 * each shape's constants fitted by Adam (adamfit.h).
 *
 *   adamfit F [in a..b | x in a..b, y in c..d ...] [options]
 *   adamfit [NAME] [x in a..b ...] [options]     (a ```c or ```glsl block)
 *
 *   options: as f32|f64|real, ops + - * / fma sqrt exp log sin cos abs,
 *   count N, size N, steps N, lr R, restarts N, wcost W, werr W, cref C,
 *   error e | ulp u, relative | absolute, auto, list N, seed K, seconds S,
 *   name P, nocert, budget N (the integer search's evaluations)
 *
 * The target: a ucas function or expression of 1 to 4 parameters (integer,
 * float or real), or a function of a C or GLSL code block.  A ucas target
 * is evaluated by ucas (a long double reference for one real or float
 * parameter, the machine evaluator otherwise); code is sampled natively: a
 * harness that calls the function is compiled with the host's cc (binary32
 * and binary64 as written: no contraction, -fwrapv), GLSL's scalar subset
 * through a shim (its casts, float literals and overloaded built-ins as C).
 *
 * The samples: a domain of at most 2^22 points (integer parameters, or the
 * floats of one interval) is enumerated, so the error measured on it holds
 * for every input; a larger one is sampled, uniformly and independently,
 * and the error is reported with what the sample supports (with N points
 * all within e, the part of the domain that errs more is at most
 * q = 1 - 10^(-6/N) with probability 1 - 10^-6).  Training points are
 * spread by smoothstep (one parameter) or scrambled Sobol points with the
 * box's corners (several).
 *
 * Float or real results: thousands of expression shapes over the
 * parameters and trainable constants (polynomials, odd and even ones,
 * rationals, sqrt forms; affine, bilinear and quadratic forms, products
 * and quotients of several parameters; random expressions of the grammar),
 * each fitted by Adam on (1/p) log sum |r|^p, the best at each cost and the
 * best scores refitted and polished (Levenberg-Marquardt, Lawson).
 *
 * Integer results (integer parameters): fixed-point forms
 * (sum C_k m_k(x) + B) >> s over monomials m_k, their real relaxation
 * fitted by Adam, then the integers C_k, B and the shift s searched with
 * CMA-ES (numsearch.c) on the exact integer error; 32-bit arithmetic when
 * the interval bound of the sum allows it, else 64-bit.  An exact or
 * bounded candidate of a ucas target (or of C that ucas lowers) is proved
 * by enumeration or SAT (equiv_decide on |F - G| <= e).
 *
 * The ranking: score = wcost * cost / C0 + werr * log2(1 + error / tol),
 * C0 the target's own cost under the active model.  One parameter, a real
 * or float target: the listed candidates' error is certified on the whole
 * interval (acert.c).
 */
#include "ucas.h"
#include "approx.h"
#include "numsearch.h"
#include "report.h"
#include "stoch.h"
#include "cfront.h"
#include "adamfit.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#define AF_MAXN 40	/* nodes of a shape */
#define AF_MAXK 8	/* trainable constants */
#define AF_MAXV 4	/* parameters */
#define AF_T 264	/* training points at most */
#define AF_STAGE2 96	/* shapes trained longer */
#define AF_PERCOST 3	/* of them, the best at each cost */
#define AF_NV 65536	/* measuring sample when not enumerated */
#define AF_EXH (1L << 22)	/* domains enumerated up to this */
#define AF_DELTA 1e-6	/* the sampled statement's failure probability */

enum {
	Q_X, Q_C, Q_ADD, Q_SUB, Q_MUL, Q_DIV, Q_FMA, Q_SQRT, Q_EXP, Q_LOG,
	Q_SIN, Q_COS, Q_ABS, Q_NOPS
};
static const char *const qname[Q_NOPS] = {"x", "c", "+", "-", "*", "/",
	"fma", "sqrt", "exp", "log", "sin", "cos", "abs"};
static const int qarity[Q_NOPS] = {0, 0, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1};

enum { TK_UCAS, TK_C, TK_GLSL };
enum { PF_NONE, PF_CERT, PF_EXH, PF_SAT, PF_SAMPLED };

struct shape {
	uint8_t op[AF_MAXN], a[AF_MAXN], b[AF_MAXN], c[AF_MAXN], k[AF_MAXN];
	int n, nk;		/* root: n - 1; nodes 0 .. nv-1: the parameters */
	char kind[12];		/* poly, odd, even, rational, sqrt, affine ... */
	/* the fit */
	double th[AF_MAXK];
	double terr;		/* the largest weighted residual on the sample */
	double cost;
	double score1;
	int ok;
	/* an integer form: (sum C_k m_k + B) >> sft; m_k = prod x_v^e[k][v] */
	int nm;
	uint8_t e[AF_MAXK][AF_MAXV];
	int64_t C[AF_MAXK], B;
	int sft, wide;
	/* stage 2 */
	double merr;		/* measured in the metric */
	int proof;		/* PF_* */
	double bound;		/* proved or certified, or -1 */
	char why[200];		/* the proof's line, or why there is none */
	double score;
	long bad;		/* points where it is undefined */
	int pick;		/* chosen for stage 2 */
};

static struct {
	/* the target */
	int kind;
	struct func f;		/* TK_UCAS: the target; code: the lowered C */
	int has_f;		/* f can be evaluated (TK_UCAS) or proved with */
	char fname[200];
	struct lref ref;
	int lref_ok;
	int nv;
	char vname[AF_MAXV][40];
	ty_t vty[AF_MAXV];
	int vint[AF_MAXV], vw[AF_MAXV];	/* integer; float width (0 real) */
	double vlo[AF_MAXV], vhi[AF_MAXV];
	int vranged[AF_MAXV];
	ty_t oty;
	int oint, ow, osigned;	/* integer result: width, signed */
	char *code;		/* C (GLSL already as C) */
	char cfn[64];
	char lang[8];
	char dir[PATH_MAX], bin[PATH_MAX];
	/* the candidates' format, the metric */
	int w, prec, emin, rel;
	unsigned ops;		/* 1 << Q_* */
	int count, size, steps, restarts, list, autom, cert, ibudget;
	double lr, wcost, werr, tol, secs, cref, tol_err, tol_ulp;
	char tolname[32];
	uint64_t seed;
	char name[24];
	/* the training sample */
	double tx[AF_MAXV][AF_T], ty[AF_T], tw[AF_T];
	int nt;
	/* the measuring sample */
	double *mx[AF_MAXV];
	ld *my;
	long nm;
	int exhaustive;		/* every point of the domain */
	double c0;		/* the target's cost */
	char c0why[120];
	struct shape *sh;
	int nsh, capsh;
	uint64_t *seen;		/* hash set of the shapes' keys */
	uint32_t capseen;
	long tried;		/* integer forms evaluated */
	struct nsrng rng;
	struct timespec t0;
} F;

/* ---- small helpers ---- */

static double
rnd(int w, double v)
{
	return w == 32 ? (double)(float)v : v;
}

static ld
ulp_of(ld y)
{
	int e;

	y = fabsl(y);
	if (y == 0)
		return ldexpl(1, F.emin - (F.prec - 1));
	e = ilogbl(y);
	if (e < F.emin)
		e = F.emin;
	return ldexpl(1, e - (F.prec - 1));
}

static void
kfmt(char *buf, size_t n, double v)
{
	snprintf(buf, n, F.w == 64 || F.w == 0 ? "%.17g" : "%.9g", v);
	if (!strpbrk(buf, ".eni"))
		strncat(buf, ".0", n - strlen(buf) - 1);
}

static void
vfmt(char *buf, size_t n, int v, double x)
{
	if (F.vint[v])
		snprintf(buf, n, "%.0f", x);
	else
		snprintf(buf, n, F.vw[v] == 32 ? "%.9g" : "%.17g", x);
	if (!F.vint[v] && !strpbrk(buf, ".eni"))
		strncat(buf, ".0", n - strlen(buf) - 1);
}

static double
elapsed(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)(t.tv_sec - F.t0.tv_sec) +
	    (t.tv_nsec - F.t0.tv_nsec) / 1e9;
}

/* the error in its unit (ulp, relative, absolute; integer units) */
static void
efmt(char *buf, size_t n, double e)
{
	if (!isfinite(e))
		snprintf(buf, n, "inf");
	else if (F.oint)
		snprintf(buf, n, "%.0f", e);
	else if (F.w && F.rel)
		snprintf(buf, n, "%.3g ulp", e);
	else
		snprintf(buf, n, "%.3g%s", e, F.rel ? " rel" : "");
}

/* live feedback: stdout, and stderr too inside a JSON request, where
 * stdout is captured until the answer (a client shows those lines live) */
static void
say(const char *fmt, ...)
{
	char line[700];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(line, sizeof line, fmt, ap);
	va_end(ap);
	fputs(line, stdout);
	fputc('\n', stdout);
	fflush(stdout);
	if (rep_on) {
		fprintf(stderr, "ucas: adamfit: %s\n", line);
		fflush(stderr);
	}
}

static const char *
fmtname(int w)
{
	return w == 32 ? "f32" : w == 64 ? "f64" : "real";
}

/* ---- shapes ---- */

static int
sh_add(struct shape *s, int op, int a, int b, int c)
{
	int i = s->n;

	if (i >= AF_MAXN)
		return -1;
	s->op[i] = (uint8_t)op;
	s->a[i] = (uint8_t)a;
	s->b[i] = (uint8_t)b;
	s->c[i] = (uint8_t)c;
	s->k[i] = 0;
	s->n++;
	return i;
}

static int
sh_const(struct shape *s)
{
	int i;

	if (s->nk >= AF_MAXK)
		return -1;
	if ((i = sh_add(s, Q_C, 0, 0, 0)) < 0)
		return -1;
	s->k[i] = (uint8_t)s->nk++;
	return i;
}

static void
sh_init(struct shape *s, const char *kind)
{
	int v;

	memset(s, 0, sizeof *s);
	for (v = 0; v < F.nv; v++) {
		s->op[v] = Q_X;
		s->k[v] = (uint8_t)v;
	}
	s->n = F.nv;
	snprintf(s->kind, sizeof s->kind, "%s", kind);
}

/* does node i depend on a parameter */
static int
has_x(const struct shape *s, int i)
{
	int j, ar;

	if (s->op[i] == Q_X)
		return 1;
	if (s->op[i] == Q_C)
		return 0;
	ar = qarity[s->op[i]];
	for (j = 0; j < ar; j++)
		if (has_x(s, j == 0 ? s->a[i] : j == 1 ? s->b[i] : s->c[i]))
			return 1;
	return 0;
}

/*
 * The canonical text of node i, constants as c, the operands of + and *
 * (and fma's first two) in order: two shapes are the same when their
 * texts are.  0 when too long.
 */
static int
canon(const struct shape *s, int i, char *out, size_t n)
{
	char x[2][512], y[512];
	int op = s->op[i];

	if (op == Q_X)
		return snprintf(out, n, "x%d", s->k[i]) < (int)n;
	if (op == Q_C)
		return snprintf(out, n, "c") < (int)n;
	if (qarity[op] == 1) {
		if (!canon(s, s->a[i], x[0], sizeof x[0]))
			return 0;
		return snprintf(out, n, "%s(%s)", qname[op], x[0]) < (int)n;
	}
	if (!canon(s, s->a[i], x[0], sizeof x[0]) ||
	    !canon(s, s->b[i], x[1], sizeof x[1]))
		return 0;
	if ((op == Q_ADD || op == Q_MUL || op == Q_FMA) &&
	    strcmp(x[0], x[1]) > 0) {
		memcpy(y, x[0], sizeof y);
		memcpy(x[0], x[1], sizeof y);
		memcpy(x[1], y, sizeof y);
	}
	if (op == Q_FMA) {
		if (!canon(s, s->c[i], y, sizeof y))
			return 0;
		return snprintf(out, n, "fma(%s,%s,%s)", x[0], x[1], y) <
		    (int)n;
	}
	return snprintf(out, n, "(%s%s%s)", x[0], qname[op], x[1]) < (int)n;
}

static uint64_t
hash_str(const char *s)
{
	uint64_t h = 1469598103934665603ull;

	while (*s)
		h = (h ^ (uint8_t)*s++) * 1099511628211ull;
	return h | 1;
}

/* the shape is new: remember it */
static int
seen_add(uint64_t h)
{
	uint32_t i, m;

	if ((uint32_t)F.nsh * 2 + 2 > F.capseen) {
		uint64_t *old = F.seen;
		uint32_t oc = F.capseen, j;
		F.capseen = F.capseen ? F.capseen * 2 : 8192;
		F.seen = xcalloc(F.capseen, sizeof *F.seen);
		for (j = 0; j < oc; j++)
			if (old[j]) {
				m = F.capseen - 1;
				for (i = (uint32_t)old[j] & m; F.seen[i];
				    i = (i + 1) & m)
					;
				F.seen[i] = old[j];
			}
		free(old);
	}
	m = F.capseen - 1;
	for (i = (uint32_t)h & m; F.seen[i]; i = (i + 1) & m)
		if (F.seen[i] == h)
			return 0;
	F.seen[i] = h;
	return 1;
}

/* a shape worth training: a parameter in it, no operation on constants
 * alone, no x - x, x / x; only enabled operations; new */
static int
sh_keep(const struct shape *s)
{
	char key[2048];
	int i;

	if (s->n <= F.nv || !has_x(s, s->n - 1))
		return 0;
	for (i = F.nv; i < s->n; i++) {
		int op = s->op[i];
		if (op == Q_C || op == Q_X)
			continue;
		if (!has_x(s, i))
			return 0;	/* constants folded: one constant does */
		if ((op == Q_SUB || op == Q_DIV) && s->a[i] == s->b[i])
			return 0;
		if (!F.oint && !(F.ops & (1u << op)))
			return 0;
	}
	if (!canon(s, s->n - 1, key, sizeof key))
		return 0;
	return seen_add(hash_str(key));
}

static struct shape *
sh_push(const struct shape *s)
{
	if (!sh_keep(s))
		return NULL;
	if (F.nsh == F.capsh) {
		F.capsh = F.capsh ? F.capsh * 2 : 1024;
		F.sh = xrealloc(F.sh, (size_t)F.capsh * sizeof *F.sh);
	}
	F.sh[F.nsh] = *s;
	return &F.sh[F.nsh++];
}

/* Horner in v (a node) with nk constants: c0 + v*(c1 + v*(...)) */
static int
horner(struct shape *s, int v, int nk)
{
	int acc = sh_const(s), j;

	for (j = 1; j < nk && acc >= 0; j++) {
		int m = sh_add(s, Q_MUL, acc, v, 0), c = sh_const(s);
		if (m < 0 || c < 0)
			return -1;
		acc = sh_add(s, Q_ADD, m, c, 0);
	}
	return acc;
}

/* c0 + c1 x_0 + c2 x_1 ...: the parameters but skip (-1: none), onto acc */
static int
affine_onto(struct shape *s, int acc, int skip)
{
	int v;

	for (v = 0; v < F.nv && acc >= 0; v++) {
		int c, m;
		if (v == skip)
			continue;
		c = sh_const(s);
		m = c < 0 ? -1 : sh_add(s, Q_MUL, c, v, 0);
		acc = m < 0 ? -1 : acc == INT_MAX ? m : sh_add(s, Q_ADD, acc, m, 0);
	}
	return acc;
}

static void
seeds1(void)
{
	struct shape s;
	int d, e, r, sq;

	for (d = 1; d <= AF_MAXK; d++) {
		/* c0 + x*(...) of d constants */
		sh_init(&s, "poly");
		if (horner(&s, 0, d) >= 0)
			sh_push(&s);
		/* odd: x*P(x*x); even: P(x*x) */
		sh_init(&s, "odd");
		sq = sh_add(&s, Q_MUL, 0, 0, 0);
		if ((r = horner(&s, sq, d)) >= 0 && sh_add(&s, Q_MUL, r, 0, 0) >= 0)
			sh_push(&s);
		if (d > 1) {
			sh_init(&s, "even");
			sq = sh_add(&s, Q_MUL, 0, 0, 0);
			if (horner(&s, sq, d) >= 0)
				sh_push(&s);
		}
	}
	/* rationals P/Q */
	for (d = 1; d <= 5; d++)
		for (e = 1; e <= 4 && d + e <= AF_MAXK; e++) {
			int p, q;
			sh_init(&s, "rational");
			p = horner(&s, 0, d);
			q = horner(&s, 0, e + 1);
			if (p >= 0 && q >= 0 && s.nk <= AF_MAXK &&
			    sh_add(&s, Q_DIV, p, q, 0) >= 0)
				sh_push(&s);
		}
	if (F.ops & (1u << Q_SQRT))
		for (d = 1; d <= 4; d++) {
			int t, p;
			/* P(x) * sqrt(x), P(sqrt(x)) */
			sh_init(&s, "sqrt");
			t = sh_add(&s, Q_SQRT, 0, 0, 0);
			p = horner(&s, 0, d);
			if (t >= 0 && p >= 0 && sh_add(&s, Q_MUL, p, t, 0) >= 0)
				sh_push(&s);
			sh_init(&s, "sqrt");
			t = sh_add(&s, Q_SQRT, 0, 0, 0);
			if (horner(&s, t, d + 1) >= 0)
				sh_push(&s);
		}
}

/* several parameters: affine, bilinear and quadratic forms, one
 * parameter's polynomial plus the others' linear terms, products and
 * quotients of pairs, a norm under sqrt */
static void
seedsn(void)
{
	struct shape s;
	int i, j, d, c, t, u;

	sh_init(&s, "affine");
	if ((t = affine_onto(&s, INT_MAX, -1)) >= 0 && (c = sh_const(&s)) >= 0 &&
	    sh_add(&s, Q_ADD, t, c, 0) >= 0)
		sh_push(&s);
	for (i = 0; i < F.nv; i++)
		for (d = 2; d <= 5; d++) {
			sh_init(&s, "poly+lin");
			t = horner(&s, i, d);
			if (t >= 0 && F.nv > 1)
				t = affine_onto(&s, t, i);
			if (t >= 0)
				sh_push(&s);
		}
	for (i = 0; i < F.nv; i++)
		for (j = i + 1; j < F.nv; j++) {
			int xy, a, b;
			/* c0 + c1 x + c2 y + c3 x*y (+ c4 x x + c5 y y) */
			for (d = 0; d < 2; d++) {
				sh_init(&s, d ? "quadratic" : "bilinear");
				xy = sh_add(&s, Q_MUL, i, j, 0);
				c = sh_const(&s);
				t = c < 0 ? -1 : sh_add(&s, Q_MUL, c, xy, 0);
				t = t < 0 ? -1 : affine_onto(&s, t, -1);
				if (d && t >= 0) {
					int xx = sh_add(&s, Q_MUL, i, i, 0),
					    yy = sh_add(&s, Q_MUL, j, j, 0), c1, c2;
					c1 = sh_const(&s);
					c2 = sh_const(&s);
					if (xx < 0 || yy < 0 || c1 < 0 || c2 < 0)
						t = -1;
					else {
						t = sh_add(&s, Q_ADD, t, sh_add(&s, Q_MUL, c1, xx,
						    0), 0);
						t = t < 0 ? -1 : sh_add(&s, Q_ADD, t,
						    sh_add(&s, Q_MUL, c2, yy, 0), 0);
					}
				}
				c = sh_const(&s);
				if (t >= 0 && c >= 0 && sh_add(&s, Q_ADD, t, c, 0) >= 0)
					sh_push(&s);
			}
			/* c x y; (a + b x)(c + d y); x / (a + b y); (a + b x)/(c + d y) */
			sh_init(&s, "product");
			xy = sh_add(&s, Q_MUL, i, j, 0);
			c = sh_const(&s);
			if (sh_add(&s, Q_MUL, c, xy, 0) >= 0)
				sh_push(&s);
			for (u = 0; u < 2; u++) {
				int p = u ? j : i, q = u ? i : j;
				sh_init(&s, "product");
				a = horner(&s, p, 2);
				b = horner(&s, q, 2);
				if (a >= 0 && b >= 0 && sh_add(&s, Q_MUL, a, b, 0) >= 0 &&
				    !u)
					sh_push(&s);
				sh_init(&s, "quotient");
				b = horner(&s, q, 2);
				if (b >= 0 && sh_add(&s, Q_DIV, p, b, 0) >= 0)
					sh_push(&s);
				sh_init(&s, "quotient");
				a = horner(&s, p, 2);
				b = horner(&s, q, 2);
				if (a >= 0 && b >= 0 && sh_add(&s, Q_DIV, a, b, 0) >= 0)
					sh_push(&s);
			}
			if (F.ops & (1u << Q_SQRT)) {
				/* sqrt(c0 + c1 x x + c2 y y) */
				int xx, yy, r;
				sh_init(&s, "sqrt");
				xx = sh_add(&s, Q_MUL, i, i, 0);
				yy = sh_add(&s, Q_MUL, j, j, 0);
				a = sh_const(&s);
				b = sh_const(&s);
				c = sh_const(&s);
				r = xx < 0 || yy < 0 || a < 0 || b < 0 || c < 0 ? -1 :
				    sh_add(&s, Q_ADD, sh_add(&s, Q_MUL, a, xx, 0),
				    sh_add(&s, Q_MUL, b, yy, 0), 0);
				r = r < 0 ? -1 : sh_add(&s, Q_ADD, r, c, 0);
				if (r >= 0 && sh_add(&s, Q_SQRT, r, 0, 0) >= 0)
					sh_push(&s);
			}
		}
}

/* a random expression of about `size` nodes from the enabled operations */
static int
gen(struct shape *s, int size)
{
	int ops[Q_NOPS], nops = 0, op, i, a, b, c, left;
	double u;

	if (size <= 1 || s->n >= AF_MAXN - 2) {
		u = ns_unif(&F.rng);
		if (u < 0.55 || s->nk >= AF_MAXK)
			return (int)(ns_unif(&F.rng) * F.nv) % F.nv;	/* a parameter */
		return sh_const(s);
	}
	for (i = Q_ADD; i < Q_NOPS; i++)
		if ((F.ops & (1u << i)) && qarity[i] < size)
			ops[nops++] = i;
	if (nops == 0)
		return 0;
	op = ops[(int)(ns_unif(&F.rng) * nops) % nops];
	left = size - 1;
	if (qarity[op] == 1) {
		a = gen(s, left);
		return a < 0 ? -1 : sh_add(s, op, a, 0, 0);
	}
	if (qarity[op] == 2) {
		int la = 1 + (int)(ns_unif(&F.rng) * (left - 1));
		a = gen(s, la);
		b = gen(s, left - la);
		return a < 0 || b < 0 ? -1 : sh_add(s, op, a, b, 0);
	}
	{
		int la = 1 + (int)(ns_unif(&F.rng) * (left - 2));
		int lb = 1 + (int)(ns_unif(&F.rng) * (left - la - 1));
		a = gen(s, la);
		b = gen(s, lb);
		c = gen(s, left - la - lb);
		return a < 0 || b < 0 || c < 0 ? -1 : sh_add(s, op, a, b, c);
	}
}

static void
randoms(void)
{
	struct shape s;
	long tries;

	for (tries = 0; F.nsh < F.count && tries < (long)F.count * 60;
	    tries++) {
		int size = 2 + (int)(ns_unif(&F.rng) * (F.size - 1));
		sh_init(&s, "random");
		if (gen(&s, size) < 0 || s.op[s.n - 1] == Q_X)
			continue;
		sh_push(&s);
	}
}

/* ---- integer forms: (sum C_k m_k + B) >> s ---- */

/* the node of monomial e (a product of parameters) */
static int
mono_node(struct shape *s, const uint8_t *e)
{
	int acc = -1, v, p;

	for (v = 0; v < F.nv; v++)
		for (p = 0; p < e[v]; p++)
			acc = acc < 0 ? v : sh_add(s, Q_MUL, acc, v, 0);
	return acc;
}

/* the shape of a fixed-point form over nm monomials: c_k m_k summed, plus
 * c_nm (B); its real relaxation is what Adam fits */
static void
int_form(const uint8_t (*e)[AF_MAXV], int nm)
{
	struct shape s;
	int k, sum = -1;

	sh_init(&s, "fixed");
	s.nm = nm;
	for (k = 0; k < nm; k++) {
		int m = mono_node(&s, e[k]), c = sh_const(&s), t;
		memcpy(s.e[k], e[k], sizeof s.e[k]);
		if (m < 0 || c < 0)
			return;
		t = sh_add(&s, Q_MUL, c, m, 0);
		sum = sum < 0 ? t : sh_add(&s, Q_ADD, sum, t, 0);
		if (t < 0 || sum < 0)
			return;
	}
	k = sh_const(&s);
	if (k < 0 || sh_add(&s, Q_ADD, sum, k, 0) < 0)
		return;
	sh_push(&s);
}

/* every set of up to 4 monomials of degree 1..D (D: 3 for one parameter, 2
 * for two, 1 more) */
static void
int_seeds(void)
{
	uint8_t mon[64][AF_MAXV], pick[AF_MAXK][AF_MAXV];
	int nmon = 0, D = F.nv == 1 ? 3 : F.nv == 2 ? 2 : 1, v, k, deg;
	uint32_t set;

	/* monomials, by degree */
	for (deg = 1; deg <= D; deg++) {
		int e[AF_MAXV] = {0};
		for (;;) {
			int sum = 0;
			for (v = 0; v < F.nv; v++)
				sum += e[v];
			if (sum == deg && nmon < 64) {
				for (v = 0; v < AF_MAXV; v++)
					mon[nmon][v] = (uint8_t)(v < F.nv ? e[v] : 0);
				nmon++;
			}
			for (v = 0; v < F.nv && ++e[v] > deg; v++)
				e[v] = 0;
			if (v == F.nv)
				break;
		}
	}
	if (nmon > 20)
		nmon = 20;
	for (set = 1; set < (1u << nmon) && F.nsh < F.count; set++) {
		int nm = 0;
		for (k = 0; k < nmon; k++)
			if (set & (1u << k)) {
				if (nm == 4) {
					nm = 99;
					break;
				}
				memcpy(pick[nm++], mon[k], AF_MAXV);
			}
		if (nm <= 4)
			int_form((const uint8_t (*)[AF_MAXV])pick, nm);
	}
}

/* ---- the samples ---- */

static int64_t
tmin(ty_t t)
{
	if (TYK(t) == K_SINT)
		return TYW(t) >= 64 ? INT64_MIN : -((int64_t)1 << (TYW(t) - 1));
	return 0;
}

static int64_t
tmax(ty_t t)
{
	if (TYK(t) == K_SINT)
		return TYW(t) >= 64 ? INT64_MAX : ((int64_t)1 << (TYW(t) - 1)) - 1;
	if (TYK(t) == K_BOOL)
		return 1;
	return TYW(t) >= 63 ? INT64_MAX : ((int64_t)1 << TYW(t)) - 1;
}

/* the result's integer value wrapped to its type */
static int64_t
owrap(int64_t v)
{
	if (TYK(F.oty) == K_INT || F.ow >= 64)
		return v;
	if (F.osigned) {
		int sh = 64 - F.ow;
		return (int64_t)((uint64_t)v << sh) >> sh;
	}
	return (int64_t)((uint64_t)v & (((uint64_t)1 << F.ow) - 1));
}

/* the domain's size, inf beyond enumeration's reach */
static double
dom_count(void)
{
	double n = 1;
	int v;

	for (v = 0; v < F.nv; v++) {
		if (F.vint[v])
			n *= F.vhi[v] - F.vlo[v] + 1;
		else if (F.nv == 1 && F.vw[v])
			n *= (double)(fkey(F.vw[v], F.vhi[v]) -
			    fkey(F.vw[v], F.vlo[v]) + 1);
		else
			return INFINITY;
	}
	return n;
}

static void freeze_point(double *x, int v)
{
	if (F.vint[v])
		*x = floor(*x + 0.5);
	else if (F.vw[v])
		*x = rnd(F.vw[v], *x);
	if (*x < F.vlo[v])
		*x = F.vlo[v];
	if (*x > F.vhi[v])
		*x = F.vhi[v];
}

static void ref_batch(double *const *xs, long n, ld *ys);

/*
 * n training points: one parameter, spread by smoothstep (denser near the
 * ends); several, the box's corners then scrambled Sobol points; values
 * from the reference.  Points where F is undefined (or 0, for a relative
 * error) are left out.
 */
static int
sample(int n)
{
	double *xs[AF_MAXV], tmp[AF_MAXV][AF_T];
	ld ys[AF_T];
	int i, v, k = 0, nc;
	uint32_t sh[AF_MAXV], u[8];

	for (v = 0; v < F.nv; v++) {
		xs[v] = tmp[v];
		sh[v] = (uint32_t)ns_u64(&F.rng);
	}
	nc = F.nv > 1 ? 1 << F.nv : 0;
	for (i = 0; i < n; i++) {
		for (v = 0; v < F.nv; v++) {
			double t;
			if (F.nv == 1) {
				double q = (double)i / (n - 1);
				t = q * q * (3 - 2 * q);
			} else if (i < nc)
				t = (i >> v) & 1;
			else {
				sobol_point((uint32_t)(i - nc + 1), F.nv, u);
				t = (double)(u[v] ^ sh[v]) / 4294967296.0;
			}
			tmp[v][i] = F.vlo[v] + (F.vhi[v] - F.vlo[v]) * t;
			freeze_point(&tmp[v][i], v);
		}
	}
	ref_batch(xs, n, ys);
	for (i = 0; i < n; i++) {
		double y = (double)ys[i];
		if (!isfinite(y) || (F.rel && y == 0))
			continue;
		for (v = 0; v < F.nv; v++)
			F.tx[v][k] = tmp[v][i];
		F.ty[k] = y;
		F.tw[k] = F.rel ? 1 / fabs(y) : 1;
		k++;
	}
	F.nt = k;
	return k;
}

/* the measuring points: every point of a domain of at most AF_EXH, else
 * AF_NV independent uniform ones */
static void
measuring(void)
{
	double cnt = dom_count();
	long i;
	int v;

	F.exhaustive = cnt <= AF_EXH;
	F.nm = F.exhaustive ? (long)cnt : AF_NV;
	for (v = 0; v < F.nv; v++)
		F.mx[v] = xmalloc((size_t)F.nm * sizeof *F.mx[v]);
	F.my = xmalloc((size_t)F.nm * sizeof *F.my);
	for (i = 0; i < F.nm; i++) {
		long r = i;
		for (v = 0; v < F.nv; v++) {
			double x;
			if (F.exhaustive && F.vint[v]) {
				long span = (long)(F.vhi[v] - F.vlo[v] + 1);
				x = F.vlo[v] + (double)(r % span);
				r /= span;
			} else if (F.exhaustive)
				x = fromkey(F.vw[v], fkey(F.vw[v], F.vlo[v]) + r);
			else if (F.vint[v]) {
				double span = F.vhi[v] - F.vlo[v] + 1;
				x = F.vlo[v] + floor(ns_unif(&F.rng) * span);
				if (x > F.vhi[v])
					x = F.vhi[v];
			} else {
				x = F.vlo[v] + (F.vhi[v] - F.vlo[v]) * ns_unif(&F.rng);
				freeze_point(&x, v);
			}
			F.mx[v][i] = x;
		}
	}
	ref_batch(F.mx, F.nm, F.my);
}

/* ---- the reference: ucas, or the code run natively ---- */

static void
plan_batch(nid body, double *const *xs, long n, ld *ys, int isint)
{
	struct plan p;
	nid roots[1];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t i, np, loc;
	long base;
	int v;

	roots[0] = body;
	plan_build(&p, roots, 1, 0);
	loc = p.root[0];
	for (base = 0; base < n; base += CH) {
		np = n - base < CH ? (uint32_t)(n - base) : CH;
		for (v = 0; v < F.nv; v++)
			for (i = 0; i < np; i++) {
				in_r[(size_t)v * CH + i] = xs[v][base + i];
				in_i[(size_t)v * CH + i] = (int64_t)xs[v][base + i];
			}
		plan_run(&p, np, in_i, in_r);
		for (i = 0; i < np; i++) {
			if (COLF(&p, loc)[i] & F_UNDEF)
				ys[base + i] = NAN;
			else if (isint)
				ys[base + i] = (ld)COLI(&p, loc)[i];
			else
				ys[base + i] = COLR(&p, loc)[i];
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
}

static void native_batch(double *const *xs, long n, ld *ys);

static void
ref_batch(double *const *xs, long n, ld *ys)
{
	long i;

	if (F.kind != TK_UCAS)
		native_batch(xs, n, ys);
	else if (F.lref_ok)
		for (i = 0; i < n; i++)
			ys[i] = lref_eval(&F.ref, xs[0][i]);
	else
		plan_batch(F.f.body, xs, n, ys, F.oint);
}

/* GLSL's scalar subset as C: its types, the qualifiers that mean nothing
 * here, and its built-ins overloaded with _Generic */
static const char glsl_shim[] =
"#include <math.h>\n#include <stdint.h>\n#include <stdbool.h>\n#include <string.h>\n"
"typedef uint32_t uint;\n"
"#define in\n#define highp\n#define mediump\n#define lowp\n#define precise\n"
"#define GF1(n, x) _Generic((x), float: n##f, default: n)(x)\n"
"static inline int32_t gl_iabs(int32_t a) { return a < 0 ? -a : a; }\n"
"static inline uint32_t gl_uabs(uint32_t a) { return a; }\n"
"#undef abs\n"
"#define abs(x) _Generic((x), float: fabsf, double: fabs, int32_t: gl_iabs, default: gl_uabs)(x)\n"
"static inline float gl_signf(float a) { return a > 0 ? 1.0f : a < 0 ? -1.0f : 0.0f; }\n"
"static inline double gl_signd(double a) { return a > 0 ? 1.0 : a < 0 ? -1.0 : 0.0; }\n"
"static inline int32_t gl_signi(int32_t a) { return (a > 0) - (a < 0); }\n"
"#define sign(x) _Generic((x), float: gl_signf, double: gl_signd, default: gl_signi)(x)\n"
"static inline float gl_minf(float a, float b) { return b < a ? b : a; }\n"
"static inline double gl_mind(double a, double b) { return b < a ? b : a; }\n"
"static inline int32_t gl_mini(int32_t a, int32_t b) { return b < a ? b : a; }\n"
"static inline uint32_t gl_minu(uint32_t a, uint32_t b) { return b < a ? b : a; }\n"
"static inline float gl_maxf(float a, float b) { return a < b ? b : a; }\n"
"static inline double gl_maxd(double a, double b) { return a < b ? b : a; }\n"
"static inline int32_t gl_maxi(int32_t a, int32_t b) { return a < b ? b : a; }\n"
"static inline uint32_t gl_maxu(uint32_t a, uint32_t b) { return a < b ? b : a; }\n"
"#define min(a, b) _Generic((a), float: gl_minf, double: gl_mind, int32_t: gl_mini, default: gl_minu)(a, b)\n"
"#define max(a, b) _Generic((a), float: gl_maxf, double: gl_maxd, int32_t: gl_maxi, default: gl_maxu)(a, b)\n"
"#define clamp(x, a, b) min(max(x, a), b)\n"
"static inline float gl_mixf(float x, float y, float a) { return x * (1.0f - a) + y * a; }\n"
"static inline double gl_mixd(double x, double y, double a) { return x * (1.0 - a) + y * a; }\n"
"#define mix(x, y, a) _Generic((x), float: gl_mixf, default: gl_mixd)(x, y, a)\n"
"static inline float gl_stepf(float e, float x) { return x < e ? 0.0f : 1.0f; }\n"
"static inline double gl_stepd(double e, double x) { return x < e ? 0.0 : 1.0; }\n"
"#define step(e, x) _Generic((x), float: gl_stepf, default: gl_stepd)(e, x)\n"
"static inline float gl_ssf(float a, float b, float x) { float t = (x - a) / (b - a); t = t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t; return t * t * (3.0f - 2.0f * t); }\n"
"static inline double gl_ssd(double a, double b, double x) { double t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); }\n"
"#define smoothstep(a, b, x) _Generic((x), float: gl_ssf, default: gl_ssd)(a, b, x)\n"
"static inline float gl_fractf(float x) { return x - floorf(x); }\n"
"static inline double gl_fractd(double x) { return x - floor(x); }\n"
"#define fract(x) _Generic((x), float: gl_fractf, default: gl_fractd)(x)\n"
"static inline float gl_modf(float x, float y) { return x - y * floorf(x / y); }\n"
"static inline double gl_modd(double x, double y) { return x - y * floor(x / y); }\n"
"#define mod(x, y) _Generic((x), float: gl_modf, default: gl_modd)(x, y)\n"
"static inline float gl_isqrtf(float x) { return 1.0f / sqrtf(x); }\n"
"static inline double gl_isqrtd(double x) { return 1.0 / sqrt(x); }\n"
"#define inversesqrt(x) _Generic((x), float: gl_isqrtf, default: gl_isqrtd)(x)\n"
"#define floor(x) GF1(floor, x)\n#define ceil(x) GF1(ceil, x)\n#define trunc(x) GF1(trunc, x)\n"
"#define round(x) GF1(round, x)\n#define roundEven(x) GF1(rint, x)\n#define sqrt(x) GF1(sqrt, x)\n"
"#define exp(x) GF1(exp, x)\n#define exp2(x) GF1(exp2, x)\n#define log(x) GF1(log, x)\n#define log2(x) GF1(log2, x)\n"
"#define sin(x) GF1(sin, x)\n#define cos(x) GF1(cos, x)\n#define tan(x) GF1(tan, x)\n#define asin(x) GF1(asin, x)\n"
"#define acos(x) GF1(acos, x)\n#define sinh(x) GF1(sinh, x)\n#define cosh(x) GF1(cosh, x)\n#define tanh(x) GF1(tanh, x)\n"
"#define asinh(x) GF1(asinh, x)\n#define acosh(x) GF1(acosh, x)\n#define atanh(x) GF1(atanh, x)\n"
"#define pow(x, y) _Generic((x), float: powf, default: pow)(x, y)\n"
"#define fma(a, b, c) _Generic((a), float: fmaf, default: fma)(a, b, c)\n"
"#define GL_ATAN(_1, _2, NAME, ...) NAME\n"
"#define gl_atan1(y) _Generic((y), float: atanf, default: atan)(y)\n"
"#define gl_atan2(y, x) _Generic((y), float: atan2f, default: atan2)(y, x)\n"
"#define atan(...) GL_ATAN(__VA_ARGS__, gl_atan2, gl_atan1, 0)(__VA_ARGS__)\n"
"#define radians(x) ((x) * (float)(3.14159265358979323846 / 180))\n"
"#define degrees(x) ((x) * (float)(180 / 3.14159265358979323846))\n"
"static inline uint32_t floatBitsToUint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }\n"
"static inline int32_t floatBitsToInt(float f) { int32_t u; memcpy(&u, &f, 4); return u; }\n"
"static inline float uintBitsToFloat(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }\n"
"static inline float intBitsToFloat(int32_t u) { float f; memcpy(&f, &u, 4); return f; }\n"
"static inline int32_t bitCount(uint32_t x) { return __builtin_popcount(x); }\n"
"static inline int32_t findMSB(uint32_t x) { return x ? 31 - __builtin_clz(x) : -1; }\n"
"static inline int32_t findLSB(uint32_t x) { return x ? __builtin_ctz(x) : -1; }\n"
"static inline uint32_t bitfieldReverse(uint32_t x) { uint32_t r = 0; int i; for (i = 0; i < 32; i++) r |= ((x >> i) & 1u) << (31 - i); return r; }\n";

static int
is_word(char c)
{
	return isalnum((unsigned char)c) || c == '_';
}

static const char *const glsl_types[] = {"float", "double", "int", "uint",
	"bool", NULL};

/* GLSL as C: constructor casts float(x) -> (float)(x), float literals with
 * their f, double literals' lf dropped; #version, #extension, precision
 * and layout lines left out.  Vectors and matrices are refused. */
static char *
glsl_as_c(const char *s)
{
	struct sbuf b = {0};
	size_t i = 0, n = strlen(s);
	/* vector, matrix and opaque types: a prefix and a digit or capital */
	static const char *const refuse[] = {"vec", "ivec", "uvec", "bvec",
		"dvec", "mat", "dmat", "sampler", "image", "texture", "isampler",
		"usampler", "iimage", "uimage", NULL};
	static const char *const refuse_word[] = {"out", "inout", "buffer",
		"uniform", "shared", NULL};
	int bol = 1;

	while (i < n) {
		char c = s[i];
		if (bol) {
			size_t j = i;
			while (j < n && (s[j] == ' ' || s[j] == '\t'))
				j++;
			if (!strncmp(s + j, "#version", 8) || !strncmp(s + j,
			    "#extension", 10) || !strncmp(s + j, "#pragma", 7) ||
			    !strncmp(s + j, "precision ", 10) || !strncmp(s + j,
			    "layout", 6)) {
				while (i < n && s[i] != '\n')
					i++;
				continue;
			}
		}
		bol = c == '\n';
		if (c == '/' && i + 1 < n && s[i + 1] == '/') {
			while (i < n && s[i] != '\n')
				sb_putc(&b, s[i++]);
			continue;
		}
		if (c == '/' && i + 1 < n && s[i + 1] == '*') {
			const char *e = strstr(s + i + 2, "*/");
			size_t m = e ? (size_t)(e - s) + 2 : n;
			for (; i < m; i++)
				sb_putc(&b, s[i]);
			continue;
		}
		if (isalpha((unsigned char)c) || c == '_') {
			size_t j = i, k;
			int t;
			while (j < n && is_word(s[j]))
				j++;
			for (t = 0; refuse[t]; t++) {
				size_t l = strlen(refuse[t]);
				if (j - i > l && !strncmp(s + i, refuse[t], l) &&
				    (isdigit((unsigned char)s[i + l]) ||
				    isupper((unsigned char)s[i + l]))) {
					free(b.p);
					fail("adamfit: GLSL's scalar subset only (float, double, "
					    "int, uint, bool): %.*s is not", (int)(j - i), s + i);
				}
			}
			for (t = 0; refuse_word[t]; t++)
				if (j - i == strlen(refuse_word[t]) && !strncmp(s + i,
				    refuse_word[t], j - i)) {
					free(b.p);
					fail("adamfit: %s: the function's own values only (no "
					    "out parameters, buffers or uniforms)", refuse_word[t]);
				}
			k = j;
			while (k < n && (s[k] == ' ' || s[k] == '\t'))
				k++;
			for (t = 0; glsl_types[t]; t++)
				if (j - i == strlen(glsl_types[t]) &&
				    !strncmp(s + i, glsl_types[t], j - i) && k < n &&
				    s[k] == '(')
					break;
			if (glsl_types[t]) {
				/* a constructor: a cast */
				sb_printf(&b, "(%s)", glsl_types[t]);
				i = k;
				continue;
			}
			sb_printf(&b, "%.*s", (int)(j - i), s + i);
			i = j;
			continue;
		}
		if (isdigit((unsigned char)c) || (c == '.' && i + 1 < n &&
		    isdigit((unsigned char)s[i + 1]))) {
			size_t j = i;
			int flt = 0;
			if (c == '0' && i + 1 < n && (s[i + 1] == 'x' || s[i + 1] == 'X')) {
				j += 2;
				while (j < n && isxdigit((unsigned char)s[j]))
					j++;
			} else {
				while (j < n && (isdigit((unsigned char)s[j]) || s[j] == '.')) {
					flt |= s[j] == '.';
					j++;
				}
				if (j < n && (s[j] == 'e' || s[j] == 'E')) {
					flt = 1;
					j++;
					if (j < n && (s[j] == '+' || s[j] == '-'))
						j++;
					while (j < n && isdigit((unsigned char)s[j]))
						j++;
				}
			}
			sb_printf(&b, "%.*s", (int)(j - i), s + i);
			i = j;
			if (j + 1 < n && (s[j] == 'l' || s[j] == 'L') &&
			    (s[j + 1] == 'f' || s[j + 1] == 'F')) {
				i += 2;		/* a double */
				continue;
			}
			if (flt && !(j < n && (s[j] == 'f' || s[j] == 'F')))
				sb_putc(&b, 'f');
			continue;
		}
		sb_putc(&b, c);
		i++;
	}
	sb_puts(&b, "");
	return b.p;
}

/* compile the harness: the code, then a main that calls F.cfn on records
 * of nv 8-byte values (int64 or double) and writes 8-byte results */
static void
native_build(void)
{
	struct sbuf b = {0}, out = {0};
	const char *tmp = getenv("TMPDIR"), *cc = getenv("UCAS_CC");
	char src[PATH_MAX + 16];
	int v, rc;

	snprintf(F.dir, sizeof F.dir, "%s/ucas-adamfit-XXXXXX", tmp && *tmp ? tmp :
	    "/tmp");
	if (mkdtemp(F.dir) == NULL)
		fail("adamfit: cannot make a temporary folder in %s", tmp && *tmp ?
		    tmp : "/tmp");
	if (F.kind == TK_GLSL)
		sb_puts(&b, glsl_shim);
	else
		sb_puts(&b, "#include <stdint.h>\n#include <math.h>\n");
	sb_puts(&b, "#define main ucas_user_main_\n#line 1 \"code\"\n");
	sb_puts(&b, F.code);
	/* the shim's in is a GLSL qualifier: the harness's names avoid it */
	sb_printf(&b, "\n#undef main\n#undef in\n#include <stdio.h>\n"
	    "int main(int argc, char **argv)\n"
	    "{\n\tFILE *ucas_fi = fopen(argv[1], \"rb\"), *ucas_fo = fopen(argv[2], \"wb\");\n"
	    "\tlong long ucas_n, ucas_k;\n"
	    "\tif (!ucas_fi || !ucas_fo || fread(&ucas_n, 8, 1, ucas_fi) != 1)\n"
	    "\t\treturn 2;\n\tfor (ucas_k = 0; ucas_k < ucas_n; ucas_k++) {\n"
	    "\t\tunion { double d; long long i; } a[%d];\n"
	    "\t\tif (fread(a, 8, %d, ucas_fi) != %d)\n\t\t\treturn 3;\n", F.nv, F.nv,
	    F.nv);
	if (F.oint)
		sb_puts(&b, "\t\tlong long r = (long long)");
	else
		sb_puts(&b, "\t\tdouble r = (double)");
	sb_printf(&b, "%s(", F.cfn);
	for (v = 0; v < F.nv; v++)
		sb_printf(&b, "%sa[%d].%c", v ? ", " : "", v, F.vint[v] ? 'i' : 'd');
	sb_puts(&b, ");\n\t\tfwrite(&r, 8, 1, ucas_fo);\n\t}\n\tfclose(ucas_fo);\n"
	    "\treturn 0;\n}\n");
	snprintf(src, sizeof src, "%s/h.c", F.dir);
	snprintf(F.bin, sizeof F.bin, "%s/h", F.dir);
	{
		FILE *fp = fopen(src, "w");
		if (fp == NULL || fputs(b.p, fp) < 0) {
			if (fp)
				fclose(fp);
			free(b.p);
			fail("adamfit: cannot write %s", src);
		}
		fclose(fp);
	}
	free(b.p);
	{
		char *argv[] = {(char *)(cc && *cc ? cc : "cc"), "-std=gnu11", "-O1",
			"-ffp-contract=off", "-fwrapv", "-fno-fast-math", "-w", "-o",
			F.bin, src, "-lm", NULL};
		rc = tool_run(argv, &out);
	}
	if (rc != 0) {
		char m[1600];
		snprintf(m, sizeof m, "%s", out.p ? out.p : "");
		free(out.p);
		fail("adamfit: the %s did not compile with %s:\n%s", F.kind ==
		    TK_GLSL ? "GLSL (as C)" : "C", cc && *cc ? cc : "cc", m);
	}
	free(out.p);
}

static void
native_batch(double *const *xs, long n, ld *ys)
{
	char pin[PATH_MAX + 8], pout[PATH_MAX + 8];
	struct sbuf out = {0};
	FILE *fp;
	long i;
	int v, rc;
	long long nn = n;

	snprintf(pin, sizeof pin, "%s/in", F.dir);
	snprintf(pout, sizeof pout, "%s/out", F.dir);
	if ((fp = fopen(pin, "wb")) == NULL)
		fail("adamfit: cannot write %s", pin);
	fwrite(&nn, 8, 1, fp);
	for (i = 0; i < n; i++)
		for (v = 0; v < F.nv; v++) {
			if (F.vint[v]) {
				long long q = (long long)xs[v][i];
				fwrite(&q, 8, 1, fp);
			} else
				fwrite(&xs[v][i], 8, 1, fp);
		}
	fclose(fp);
	{
		char *argv[] = {F.bin, pin, pout, NULL};
		rc = tool_run(argv, &out);
	}
	if (rc != 0) {
		char m[600];
		snprintf(m, sizeof m, "%s", out.p ? out.p : "");
		free(out.p);
		fail("adamfit: running %s on the samples failed (status %d)%s%s",
		    F.cfn, rc, m[0] ? ":\n" : "", m);
	}
	free(out.p);
	if ((fp = fopen(pout, "rb")) == NULL)
		fail("adamfit: no results from %s", F.cfn);
	for (i = 0; i < n; i++) {
		union {
			double d;
			long long q;
		} r;
		if (fread(&r, 8, 1, fp) != 1) {
			fclose(fp);
			fail("adamfit: %s stopped after %ld of %ld samples", F.cfn, i, n);
		}
		if (F.oint)
			ys[i] = F.osigned || F.ow < 64 ? (ld)r.q : (ld)(uint64_t)r.q;
		else
			ys[i] = r.d;
	}
	fclose(fp);
	unlink(pin);
	unlink(pout);
}

static void
native_free(void)
{
	char p[PATH_MAX + 16];

	if (!F.dir[0])
		return;
	snprintf(p, sizeof p, "%s/h.c", F.dir);
	unlink(p);
	unlink(F.bin);
	rmdir(F.dir);
	F.dir[0] = 0;
}

/* ---- the fit: forward, one reverse sweep, Adam ---- */

static double vv[AF_MAXN][AF_T], gg[AF_MAXN][AF_T];

/* forward on the training sample; 0 where something is not finite */
static int
forward(const struct shape *s, const double *th)
{
	int i, t;

	for (i = 0; i < F.nv; i++)
		for (t = 0; t < F.nt; t++)
			vv[i][t] = F.tx[i][t];
	for (i = F.nv; i < s->n; i++) {
		const double *a = vv[s->a[i]], *b = vv[s->b[i]], *c = vv[s->c[i]];
		double *v = vv[i];
		switch (s->op[i]) {
		case Q_C:
			for (t = 0; t < F.nt; t++)
				v[t] = th[s->k[i]];
			break;
		case Q_ADD: for (t = 0; t < F.nt; t++) v[t] = a[t] + b[t]; break;
		case Q_SUB: for (t = 0; t < F.nt; t++) v[t] = a[t] - b[t]; break;
		case Q_MUL: for (t = 0; t < F.nt; t++) v[t] = a[t] * b[t]; break;
		case Q_DIV: for (t = 0; t < F.nt; t++) v[t] = a[t] / b[t]; break;
		case Q_FMA:
			for (t = 0; t < F.nt; t++)
				v[t] = a[t] * b[t] + c[t];
			break;
		case Q_SQRT:
			for (t = 0; t < F.nt; t++)
				v[t] = a[t] >= 0 ? sqrt(a[t]) : NAN;
			break;
		case Q_EXP: for (t = 0; t < F.nt; t++) v[t] = exp(a[t]); break;
		case Q_LOG:
			for (t = 0; t < F.nt; t++)
				v[t] = a[t] > 0 ? log(a[t]) : NAN;
			break;
		case Q_SIN: for (t = 0; t < F.nt; t++) v[t] = sin(a[t]); break;
		case Q_COS: for (t = 0; t < F.nt; t++) v[t] = cos(a[t]); break;
		case Q_ABS: for (t = 0; t < F.nt; t++) v[t] = fabs(a[t]); break;
		}
	}
	for (t = 0; t < F.nt; t++)
		if (!isfinite(vv[s->n - 1][t]))
			return 0;
	return 1;
}

/*
 * The loss (1/p) log sum |r_t|^p, r_t = (R(x_t) - y_t) w_t, and its
 * gradient over the constants in g; returns max |r_t| (inf: not finite).
 */
static double
loss_grad(const struct shape *s, const double *th, double p, double *g)
{
	int i, t, root = s->n - 1;
	double M = 0, sum = 0, r[AF_T];

	if (!forward(s, th))
		return INFINITY;
	for (t = 0; t < F.nt; t++) {
		r[t] = (vv[root][t] - F.ty[t]) * F.tw[t];
		if (fabs(r[t]) > M)
			M = fabs(r[t]);
	}
	for (i = 0; i < s->nk; i++)
		g[i] = 0;
	if (M == 0)
		return 0;
	for (t = 0; t < F.nt; t++)
		sum += pow(fabs(r[t]) / M, p);
	for (i = 0; i < s->n; i++)
		memset(gg[i], 0, sizeof(double) * (size_t)F.nt);
	for (t = 0; t < F.nt; t++) {
		double q = r[t] / M;
		/* d/dr (1/p) log sum |r|^p = |r|^(p-1) sign(r) / sum |r|^p */
		gg[root][t] = pow(fabs(q), p - 1) * (q < 0 ? -1 : 1) / (sum * M) *
		    F.tw[t];
	}
	for (i = root; i >= F.nv; i--) {
		double *gi = gg[i], *ga = gg[s->a[i]], *gb = gg[s->b[i]],
		    *gc = gg[s->c[i]];
		const double *va = vv[s->a[i]], *vb = vv[s->b[i]], *v = vv[i];
		switch (s->op[i]) {
		case Q_C:
			for (t = 0; t < F.nt; t++)
				g[s->k[i]] += gi[t];
			break;
		case Q_ADD:
			for (t = 0; t < F.nt; t++)
				ga[t] += gi[t], gb[t] += gi[t];
			break;
		case Q_SUB:
			for (t = 0; t < F.nt; t++)
				ga[t] += gi[t], gb[t] -= gi[t];
			break;
		case Q_MUL:
			for (t = 0; t < F.nt; t++) {
				ga[t] += gi[t] * vb[t];
				gb[t] += gi[t] * va[t];
			}
			break;
		case Q_DIV:
			for (t = 0; t < F.nt; t++) {
				ga[t] += gi[t] / vb[t];
				gb[t] -= gi[t] * v[t] / vb[t];
			}
			break;
		case Q_FMA:
			for (t = 0; t < F.nt; t++) {
				ga[t] += gi[t] * vb[t];
				gb[t] += gi[t] * va[t];
				gc[t] += gi[t];
			}
			break;
		case Q_SQRT:
			for (t = 0; t < F.nt; t++)
				ga[t] += v[t] > 0 ? gi[t] * 0.5 / v[t] : 0;
			break;
		case Q_EXP:
			for (t = 0; t < F.nt; t++)
				ga[t] += gi[t] * v[t];
			break;
		case Q_LOG:
			for (t = 0; t < F.nt; t++)
				ga[t] += gi[t] / va[t];
			break;
		case Q_SIN:
			for (t = 0; t < F.nt; t++)
				ga[t] += gi[t] * cos(va[t]);
			break;
		case Q_COS:
			for (t = 0; t < F.nt; t++)
				ga[t] -= gi[t] * sin(va[t]);
			break;
		case Q_ABS:
			for (t = 0; t < F.nt; t++)
				ga[t] += va[t] < 0 ? -gi[t] : gi[t];
			break;
		}
	}
	for (i = 0; i < s->nk; i++)
		if (!isfinite(g[i]))
			return INFINITY;
	return M;
}

/* a normal deviate from twelve uniforms (no libm in the loop) */
static double
gauss(void)
{
	double z = -6;
	int i;

	for (i = 0; i < 12; i++)
		z += ns_unif(&F.rng);
	return z;
}

/*
 * Adam over the constants from th: steps steps, the step lr decaying to a
 * tenth, p from p0 to p1 at 40%.  th ends at the best point seen (by the
 * largest residual); returns that residual.
 */
static double
adam(const struct shape *s, double *th, int steps, double lr, double p0,
    double p1)
{
	double m[AF_MAXK] = {0}, v[AF_MAXK] = {0}, g[AF_MAXK], best[AF_MAXK];
	double be = INFINITY, b1t = 1, b2t = 1;
	const double b1 = 0.9, b2 = 0.999;
	int st, i;

	memcpy(best, th, sizeof best);
	for (st = 0; st <= steps; st++) {
		double p = st < steps * 4 / 10 ? p0 : p1;
		double e = loss_grad(s, th, p, g), a;
		if (!isfinite(e)) {
			if (st == 0)
				return INFINITY;
			break;	/* left the domain: keep the best so far */
		}
		if (e < be) {
			be = e;
			memcpy(best, th, sizeof best);
		}
		if (st == steps || e == 0)
			break;
		b1t *= b1;
		b2t *= b2;
		a = lr * (1 - 0.9 * st / steps);
		for (i = 0; i < s->nk; i++) {
			double mh, vh;
			m[i] = b1 * m[i] + (1 - b1) * g[i];
			v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
			mh = m[i] / (1 - b1t);
			vh = v[i] / (1 - b2t);
			th[i] -= a * mh / (sqrt(vh) + 1e-300);
		}
	}
	memcpy(th, best, sizeof best);
	return be;
}

static double polish(const struct shape *s, double *th);

/* stage 1: a few starts, the best kept */
static void
train(struct shape *s)
{
	double th[AF_MAXK], e;
	int r, i;

	s->terr = INFINITY;
	for (r = 0; r < F.restarts; r++) {
		for (i = 0; i < AF_MAXK; i++)
			th[i] = r == 0 ? 0.5 : gauss();
		e = adam(s, th, F.steps, F.lr, 2, 8);
		if (F.oint) {
			/* linear in its constants: Levenberg-Marquardt solves what
			 * Adam's small steps cannot reach (large offsets) */
			double e2 = polish(s, th);
			if (e2 < e)
				e = e2;
		}
		if (e < s->terr) {
			s->terr = e;
			memcpy(s->th, th, sizeof th);
		}
	}
	s->ok = isfinite(s->terr);
}

/* the training residual in the metric's unit: ulps of f's format when
 * relative (one ulp of a relative 2^(1 - prec) at most) */
static double
tunit(double r)
{
	return !F.oint && F.w && F.rel ? r * ldexp(1, F.prec - 1) : r;
}

/* ---- the polish: Levenberg-Marquardt, then Lawson's reweighting ---- */

static double dv[AF_MAXK][AF_MAXN][AF_T];

/* the residuals e_t = (R - y) w_t and their Jacobian J[k][t]; 0 when not
 * finite */
static int
jacobian(const struct shape *s, const double *th, double *e, double J[][AF_T])
{
	int i, k, t, root = s->n - 1;

	if (!forward(s, th))
		return 0;
	for (k = 0; k < s->nk; k++) {
		double (*d)[AF_T] = dv[k];
		for (i = 0; i < F.nv; i++)
			for (t = 0; t < F.nt; t++)
				d[i][t] = 0;
		for (i = F.nv; i < s->n; i++) {
			const double *a = vv[s->a[i]], *b = vv[s->b[i]], *v = vv[i];
			const double *da = d[s->a[i]], *db = d[s->b[i]], *dc = d[s->c[i]];
			double *o = d[i];
			switch (s->op[i]) {
			case Q_C:
				for (t = 0; t < F.nt; t++)
					o[t] = s->k[i] == k;
				break;
			case Q_ADD: for (t = 0; t < F.nt; t++) o[t] = da[t] + db[t]; break;
			case Q_SUB: for (t = 0; t < F.nt; t++) o[t] = da[t] - db[t]; break;
			case Q_MUL:
				for (t = 0; t < F.nt; t++)
					o[t] = da[t] * b[t] + a[t] * db[t];
				break;
			case Q_DIV:
				for (t = 0; t < F.nt; t++)
					o[t] = (da[t] - v[t] * db[t]) / b[t];
				break;
			case Q_FMA:
				for (t = 0; t < F.nt; t++)
					o[t] = da[t] * b[t] + a[t] * db[t] + dc[t];
				break;
			case Q_SQRT:
				for (t = 0; t < F.nt; t++)
					o[t] = v[t] > 0 ? da[t] * 0.5 / v[t] : 0;
				break;
			case Q_EXP: for (t = 0; t < F.nt; t++) o[t] = da[t] * v[t]; break;
			case Q_LOG: for (t = 0; t < F.nt; t++) o[t] = da[t] / a[t]; break;
			case Q_SIN:
				for (t = 0; t < F.nt; t++)
					o[t] = da[t] * cos(a[t]);
				break;
			case Q_COS:
				for (t = 0; t < F.nt; t++)
					o[t] = -da[t] * sin(a[t]);
				break;
			case Q_ABS:
				for (t = 0; t < F.nt; t++)
					o[t] = a[t] < 0 ? -da[t] : da[t];
				break;
			}
		}
		for (t = 0; t < F.nt; t++) {
			J[k][t] = d[root][t] * F.tw[t];
			if (!isfinite(J[k][t]))
				return 0;
		}
	}
	for (t = 0; t < F.nt; t++)
		e[t] = (vv[root][t] - F.ty[t]) * F.tw[t];
	return 1;
}

/* solve A x = b (n <= AF_MAXK) by elimination with partial pivoting */
static int
solve(int n, double A[][AF_MAXK], double *b, double *x)
{
	int i, j, k;

	for (k = 0; k < n; k++) {
		int pv = k;
		for (i = k + 1; i < n; i++)
			if (fabs(A[i][k]) > fabs(A[pv][k]))
				pv = i;
		if (!(fabs(A[pv][k]) > 0))
			return 0;
		if (pv != k) {
			for (j = 0; j < n; j++) {
				double t = A[k][j];
				A[k][j] = A[pv][j];
				A[pv][j] = t;
			}
			{
				double t = b[k];
				b[k] = b[pv];
				b[pv] = t;
			}
		}
		for (i = k + 1; i < n; i++) {
			double f = A[i][k] / A[k][k];
			for (j = k; j < n; j++)
				A[i][j] -= f * A[k][j];
			b[i] -= f * b[k];
		}
	}
	for (i = n - 1; i >= 0; i--) {
		double v = b[i];
		for (j = i + 1; j < n; j++)
			v -= A[i][j] * x[j];
		x[i] = v / A[i][i];
	}
	for (i = 0; i < n; i++)
		if (!isfinite(x[i]))
			return 0;
	return 1;
}

/* weighted squares sum u_t e_t^2 */
static double
wsse(const double *e, const double *u)
{
	double s2 = 0;
	int t;

	for (t = 0; t < F.nt; t++)
		s2 += u[t] * e[t] * e[t];
	return s2;
}

/* Levenberg-Marquardt on sum u_t e_t^2 from th, iters steps at most */
static void
lm(const struct shape *s, double *th, const double *u, int iters)
{
	static double J[AF_MAXK][AF_T], NJ[AF_MAXK][AF_T];
	double e[AF_T], lam = 1e-3, cur;
	int it, i, j, t, n = s->nk;

	if (!jacobian(s, th, e, J))
		return;
	cur = wsse(e, u);
	for (it = 0; it < iters; it++) {
		double A[AF_MAXK][AF_MAXK], g[AF_MAXK], d[AF_MAXK], tr[AF_MAXK];
		double ne[AF_T];
		for (i = 0; i < n; i++) {
			g[i] = 0;
			for (t = 0; t < F.nt; t++)
				g[i] -= u[t] * J[i][t] * e[t];
			for (j = 0; j < n; j++) {
				double a = 0;
				for (t = 0; t < F.nt; t++)
					a += u[t] * J[i][t] * J[j][t];
				A[i][j] = a;
			}
		}
		for (;;) {
			double B[AF_MAXK][AF_MAXK], c[AF_MAXK];
			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++)
					B[i][j] = A[i][j];
				B[i][i] += lam * (A[i][i] > 0 ? A[i][i] : 1);
				c[i] = g[i];
			}
			if (solve(n, B, c, d)) {
				for (i = 0; i < n; i++)
					tr[i] = th[i] + d[i];
				if (jacobian(s, tr, ne, NJ) && wsse(ne, u) < cur) {
					memcpy(th, tr, sizeof(double) * (size_t)n);
					memcpy(e, ne, sizeof(double) * (size_t)F.nt);
					memcpy(J, NJ, sizeof J);
					cur = wsse(e, u);
					lam = lam / 3 < 1e-12 ? 1e-12 : lam / 3;
					break;
				}
			}
			lam *= 4;
			if (lam > 1e12)
				return;	/* no step helps: a minimum */
		}
	}
}

/* LM, then Lawson's reweighting towards minimax; th ends at the best (by
 * the largest |e_t|), which is returned */
static double
polish(const struct shape *s, double *th)
{
	static double J[AF_MAXK][AF_T];
	double u[AF_T], e[AF_T], best[AF_MAXK], be = INFINITY, sum;
	int r, t;

	for (t = 0; t < F.nt; t++)
		u[t] = 1.0 / F.nt;
	memcpy(best, th, sizeof best);
	for (r = 0; r < 24; r++) {
		double m = 0;
		lm(s, th, u, r == 0 ? 60 : 12);
		if (!jacobian(s, th, e, J))
			break;
		for (t = 0; t < F.nt; t++)
			if (fabs(e[t]) > m)
				m = fabs(e[t]);
		if (m < be) {
			be = m;
			memcpy(best, th, sizeof best);
		}
		if (m == 0)
			break;
		/* Lawson: weights times |e|, normalised */
		sum = 0;
		for (t = 0; t < F.nt; t++)
			sum += u[t] *= fabs(e[t]) / m + 1e-300;
		if (!(sum > 0))
			break;
		for (t = 0; t < F.nt; t++)
			u[t] /= sum;
	}
	memcpy(th, best, sizeof best);
	return be;
}

/* ---- integer forms: rounding the relaxation, CMA-ES on the integers ---- */

/* the integer search's points: an even stride of at most AF_IFULL points
 * of an enumerated domain (all of it when that small), else the training
 * sample; the candidates are measured on the whole domain after */
#define AF_IFULL 8192
static int64_t *mval[AF_MAXK], *yint, *ipt[AF_MAXV];
static long nip;
static double mmax[AF_MAXK];

struct ictx {
	const struct shape *s;
	int64_t C[AF_MAXK + 1];	/* C_0 .. C_nm-1, then B */
	int sft;
};

/* floor(v / 2^s) */
static int64_t
fshr(int64_t v, int s)
{
	if (s == 0)
		return v;
	return v >= 0 ? v >> s : -((-v + (((int64_t)1 << s) - 1)) >> s);
}

/* the largest |sum C_k m_k + B| over the box */
static double
ibound(const struct shape *s, const int64_t *C)
{
	double b = fabs((double)C[s->nm]);
	int k;

	for (k = 0; k < s->nm; k++)
		b += fabs((double)C[k]) * mmax[k];
	return b;
}

/* the largest error on the training points (huge when the sum could leave
 * 64 bits); plus a tenth of the mean, to break ties */
static double
ierr(const struct shape *s, const int64_t *C, int sft)
{
	double m = 0, sum = 0;
	int k;

	F.tried++;
	if (ibound(s, C) > 4.0e18)
		return 1e30;
	for (long t = 0; t < nip; t++) {
		int64_t a = C[s->nm], d;
		for (k = 0; k < s->nm; k++)
			a += C[k] * mval[k][t];
		d = owrap(fshr(a, sft)) - yint[t];
		if (d < 0)
			d = -d;
		if ((double)d > m)
			m = (double)d;
		sum += (double)d;
	}
	return m + 0.1 * sum / (double)nip;
}

static double
iobj(void *ctx, const int *off)
{
	const struct ictx *c = ctx;
	int64_t C[AF_MAXK + 1];
	int k;

	for (k = 0; k <= c->s->nm; k++)
		C[k] = c->C[k] + off[k];
	return ierr(c->s, C, c->sft);
}

/*
 * B where the form is exact: each point asks y 2^s <= S + B < (y + 1) 2^s
 * (S the monomials' sum), an interval for B; when the intervals meet on
 * every measuring point (the whole domain when enumerated), B moves into
 * it (the nearest end).  Results that wrap are left out.
 */
static void
fix_b(struct shape *s)
{
	double lo = -9.2e18, hi = 9.2e18;
	long n = F.nm, t;
	int k, v, p;

	for (t = 0; t < n; t++) {
		double S = 0, y = (double)F.my[t], sc;
		if (!isfinite(y))
			continue;
		if (y < (double)tmin(F.oty) || y > (double)tmax(F.oty))
			continue;
		for (k = 0; k < s->nm; k++) {
			double m = (double)s->C[k];
			for (v = 0; v < F.nv; v++)
				for (p = 0; p < s->e[k][v]; p++)
					m *= F.mx[v][t];
			S += m;
		}
		sc = ldexp(1, s->sft);
		if (y * sc - S > lo)
			lo = y * sc - S;
		if ((y + 1) * sc - 1 - S < hi)
			hi = (y + 1) * sc - 1 - S;
		if (lo > hi)
			return;		/* no B is exact */
	}
	if ((double)s->B < lo)
		s->B = (int64_t)lo;
	else if ((double)s->B > hi)
		s->B = (int64_t)hi;
}

/* the largest error of an integer form on the measuring points */
static double
merr_int(const struct shape *s)
{
	double m = 0;
	long t;
	int k, v, p;

	for (t = 0; t < F.nm; t++) {
		int64_t a = s->B, d;
		if (!isfinite(F.my[t]))
			continue;
		for (k = 0; k < s->nm; k++) {
			int64_t q = s->C[k];
			for (v = 0; v < F.nv; v++)
				for (p = 0; p < s->e[k][v]; p++)
					q *= (int64_t)F.mx[v][t];
			a += q;
		}
		d = owrap(fshr(a, s->sft)) - (int64_t)F.my[t];
		if (d < 0)
			d = -d;
		if ((double)d > m)
			m = (double)d;
	}
	return m;
}

/* the B interval of a form on the search's points: 1 and [*lo, *hi] when
 * some B is exact on every one */
static int
b_interval(const struct shape *s, const int64_t *C, int sft, double *lo,
    double *hi)
{
	double sc = ldexp(1, sft);
	long t;
	int k;

	*lo = -9.2e18;
	*hi = 9.2e18;
	for (t = 0; t < nip; t++) {
		double S = 0, y = (double)yint[t];
		if (y < (double)tmin(F.oty) || y > (double)tmax(F.oty))
			continue;
		for (k = 0; k < s->nm; k++)
			S += (double)C[k] * (double)mval[k][t];
		if (y * sc - S > *lo)
			*lo = y * sc - S;
		if ((y + 1) * sc - 1 - S < *hi)
			*hi = (y + 1) * sc - 1 - S;
		if (*lo > *hi)
			return 0;
	}
	return 1;
}

/*
 * An exact form near the relaxation: for forms of up to two monomials,
 * every shift and every coefficient near the rounded relaxation (within 64
 * for one monomial, 8 for two: a floor's sawtooth biases the fitted slope),
 * the B that is exact (an interval) when there is one; the smallest shift
 * that has one, confirmed on every measuring point.  1 when found.
 */
static int
exact_search(struct shape *s)
{
	int sft, d0, d1, k, r0 = s->nm == 1 ? 64 : 8, r1 = s->nm > 1 ? 8 : 0;

	if (s->nm > 2)
		return 0;
	for (sft = 0; sft < 48; sft++) {
		double sc = ldexp(1, sft);
		int64_t base[2] = {0, 0}, C[AF_MAXK + 1];
		for (k = 0; k < s->nm; k++)
			base[k] = (int64_t)llround(s->th[k] * sc);
		for (d0 = -r0; d0 <= r0; d0++)
			for (d1 = -r1; d1 <= r1; d1++) {
				double lo, hi;
				C[0] = base[0] + d0;
				C[1] = base[1] + d1;
				C[s->nm] = 0;
				F.tried++;
				if (ibound(s, C) + sc > 4.0e18)
					return 0;
				if (!b_interval(s, C, sft, &lo, &hi))
					continue;
				{
					struct shape t = *s;
					memcpy(t.C, C, sizeof(int64_t) * (size_t)s->nm);
					t.B = (int64_t)lo;
					t.sft = sft;
					fix_b(&t);	/* the measuring points */
					C[s->nm] = t.B;
					if (ierr(&t, C, sft) == 0 && merr_int(&t) == 0) {
						*s = t;
						return 1;
					}
				}
			}
	}
	return 0;
}

/* the fitted relaxation as integers: for each shift the rounded form; the
 * best few shifts searched with CMA-ES; the best kept, reduced (a form
 * with every integer even is the same at one shift less) */
static void
integerize(struct shape *s, int idx)
{
	double best = INFINITY;
	int top[4], ntop = 0, sft, k, j;
	long t;
	double terr[64];
	int64_t Cs[64][AF_MAXK + 1];

	/* the points: the whole (small, enumerated) domain, or the training
	 * sample; the monomials' values there and bounds on the box */
	if (nip == 0) {
		int full = F.exhaustive, v;
		long q = 0, stride = full ? (F.nm + AF_IFULL - 1) / AF_IFULL : 1;
		nip = full ? (F.nm + stride - 1) / stride : F.nt;
		for (k = 0; k < AF_MAXK; k++)
			mval[k] = xmalloc((size_t)nip * sizeof *mval[k]);
		yint = xmalloc((size_t)nip * sizeof *yint);
		for (v = 0; v < F.nv; v++)
			ipt[v] = xmalloc((size_t)nip * sizeof *ipt[v]);
		for (t = 0; t < nip; t++) {
			long at = full ? t * stride : t;
			ld y = full ? F.my[at] : F.ty[at];
			if (!isfinite(y))
				continue;
			for (v = 0; v < F.nv; v++)
				ipt[v][q] = (int64_t)(full ? F.mx[v][at] : F.tx[v][at]);
			yint[q++] = (int64_t)y;
		}
		nip = q;
	}
	for (k = 0; k < s->nm; k++) {
		double mb = 1;
		int v;
		for (t = 0; t < nip; t++) {
			int64_t m = 1;
			for (v = 0; v < F.nv; v++) {
				int p;
				for (p = 0; p < s->e[k][v]; p++)
					m *= ipt[v][t];
			}
			mval[k][t] = m;
		}
		for (v = 0; v < F.nv; v++) {
			double a = fabs(F.vlo[v]) > fabs(F.vhi[v]) ? fabs(F.vlo[v]) :
			    fabs(F.vhi[v]);
			int p;
			for (p = 0; p < s->e[k][v]; p++)
				mb *= a;
		}
		mmax[k] = mb;
	}
	/* every shift: the relaxation rounded (to nearest, and the
	 * coefficients up, as a division's magic numbers are), the constant
	 * term at nearest or at the relaxation's own; the best of those */
	for (sft = 0; sft < 63; sft++) {
		double sc = ldexp(1, sft), be = INFINITY;
		int var;
		int64_t C[AF_MAXK + 1];
		for (var = 0; var < 4; var++) {
			double e;
			for (k = 0; k < s->nm; k++)
				C[k] = (int64_t)(var & 1 ? ceil(s->th[k] * sc) :
				    llround(s->th[k] * sc));
			C[s->nm] = (int64_t)llround(s->th[s->nm] * sc +
			    (var & 2 || !sft ? 0 : sc / 2));
			e = ibound(s, C) > 4.0e18 ? INFINITY : ierr(s, C, sft);
			if (e < be) {
				be = e;
				memcpy(Cs[sft], C, sizeof C);
			}
		}
		terr[sft] = be;
		if (!isfinite(terr[sft]))
			break;
	}
	{
		int n = sft;
		for (j = 0; j < 4; j++) {
			int b = -1;
			for (sft = 0; sft < n; sft++) {
				int used = 0, q;
				for (q = 0; q < ntop; q++)
					used |= top[q] == sft;
				if (!used && isfinite(terr[sft]) && (b < 0 ||
				    terr[sft] < terr[b]))
					b = sft;
			}
			if (b < 0)
				break;
			top[ntop++] = b;
		}
	}
	for (j = 0; j < ntop; j++) {
		struct ictx c;
		struct nsopt o;
		struct nsres r;
		int off[AF_MAXK + 1], q;
		double e;
		c.s = s;
		c.sft = top[j];
		memcpy(c.C, Cs[top[j]], sizeof c.C);
		ns_defaults(&o);
		o.kind = NS_CMAES;
		o.budget = F.ibudget;
		o.seed = F.seed + (uint64_t)idx * 7919 + (uint64_t)j;
		o.sigma = 2;
		o.radius = 1 << 16;
		ns_run(s->nm + 1, iobj, &c, &o, off, &r);
		for (q = 0; q <= s->nm; q++)
			c.C[q] += off[q];
		e = ierr(s, c.C, c.sft);
		/* the same error at fewer bits wins */
		if (e < best || (e == best && c.sft < s->sft)) {
			best = e;
			memcpy(s->C, c.C, sizeof(int64_t) * (size_t)s->nm);
			s->B = c.C[s->nm];
			s->sft = c.sft;
		}
	}
	fix_b(s);
	if (best > 0 && exact_search(s))
		best = 0;
	/* even integers: one bit less */
	while (s->sft > 0) {
		int even = (s->B & 1) == 0;
		for (k = 0; k < s->nm; k++)
			even &= (s->C[k] & 1) == 0;
		if (!even)
			break;
		for (k = 0; k < s->nm; k++)
			s->C[k] /= 2;
		s->B /= 2;
		s->sft--;
	}
	{
		int64_t C[AF_MAXK + 1];
		memcpy(C, s->C, sizeof(int64_t) * (size_t)s->nm);
		C[s->nm] = s->B;
		s->wide = ibound(s, C) >= 2147483647.0;
		{
			int v;
			for (v = 0; v < F.nv; v++)
				if (F.vhi[v] > 2147483647.0 || F.vlo[v] < -2147483648.0)
					s->wide = 1;
		}
		s->terr = ierr(s, C, s->sft);
	}
	s->ok = isfinite(best);
}

/* ---- ucas expressions: text, definition, price ---- */

static void text(const struct shape *s, int i, const double *th, int round,
    struct sbuf *b);

/* a parameter as an operand of the candidate's arithmetic */
static void
var_text(int v, struct sbuf *b)
{
	if (F.vint[v] && !F.oint)
		sb_printf(b, "%s(%s)", fmtname(F.w), F.vname[v]);
	else
		sb_puts(b, F.vname[v]);
}

static void
text(const struct shape *s, int i, const double *th, int round, struct sbuf *b)
{
	char k[48];
	int op = s->op[i];

	switch (op) {
	case Q_X:
		var_text(s->k[i], b);
		return;
	case Q_C:
		if (th == NULL)
			snprintf(k, sizeof k, "c%d", s->k[i]);
		else
			kfmt(k, sizeof k, round ? rnd(F.w, th[s->k[i]]) : th[s->k[i]]);
		sb_puts(b, k[0] == '-' ? "(" : "");
		sb_puts(b, k);
		sb_puts(b, k[0] == '-' ? ")" : "");
		return;
	}
	if (qarity[op] == 1) {
		sb_printf(b, "%s(", qname[op]);
		text(s, s->a[i], th, round, b);
		sb_puts(b, ")");
		return;
	}
	if (op == Q_FMA) {
		sb_puts(b, "fma(");
		text(s, s->a[i], th, round, b);
		sb_puts(b, ", ");
		text(s, s->b[i], th, round, b);
		sb_puts(b, ", ");
		text(s, s->c[i], th, round, b);
		sb_puts(b, ")");
		return;
	}
	{
		/* parentheses only where precedence needs them */
		int pr = op == Q_ADD || op == Q_SUB ? 1 : 2;
		int la = s->op[s->a[i]], lb = s->op[s->b[i]];
		int pa = la == Q_ADD || la == Q_SUB ? 1 : la == Q_MUL || la == Q_DIV ? 2 : 3;
		int pb = lb == Q_ADD || lb == Q_SUB ? 1 : lb == Q_MUL || lb == Q_DIV ? 2 : 3;
		int qa = pa < pr, qb = pb < pr || (pb == pr && (op == Q_SUB ||
		    op == Q_DIV));
		sb_puts(b, qa ? "(" : "");
		text(s, s->a[i], th, round, b);
		sb_printf(b, "%s %s ", qa ? ")" : "", qname[op]);
		sb_puts(b, qb ? "(" : "");
		text(s, s->b[i], th, round, b);
		sb_puts(b, qb ? ")" : "");
	}
}

/* an integer form: T((C0 * m0 + ... + B) >> s), in i32 or i64 */
static void
int_text(const struct shape *s, int symbolic, struct sbuf *b)
{
	const char *W = s->wide ? "i64" : "i32";
	int k, v, p, first = 1, any;
	char o[24];

	snprintf(o, sizeof o, "%s", ty_str(F.oty));
	sb_printf(b, "%s(", o);
	if (s->sft)
		sb_puts(b, "(");
	for (k = 0; k < s->nm; k++) {
		if (s->C[k] == 0 && (!symbolic || s->ok))
			continue;
		if (symbolic)
			sb_printf(b, "%sC%d", first ? "" : " + ", k);
		else if (s->C[k] < 0)
			sb_printf(b, "%s%lld", first ? "-" : " - ",
			    (long long)-s->C[k]);
		else
			sb_printf(b, "%s%lld", first ? "" : " + ", (long long)s->C[k]);
		for (v = 0; v < F.nv; v++)
			for (p = 0; p < s->e[k][v]; p++)
				sb_printf(b, " * %s(%s)", W, F.vname[v]);
		first = 0;
	}
	any = !first;
	if (symbolic)
		sb_printf(b, "%sB", any ? " + " : "");
	else if (s->B || !any) {
		if (!any)
			sb_printf(b, "%s(%lld)", W, (long long)s->B);
		else if (s->B < 0)
			sb_printf(b, " - %lld", (long long)-s->B);
		else
			sb_printf(b, " + %lld", (long long)s->B);
	}
	if (s->sft) {
		if (symbolic)
			sb_puts(b, ") >> s");
		else
			sb_printf(b, ") >> %d", s->sft);
	}
	sb_puts(b, ")");
}

/* a shape's text with its constants (th) or as c0, c1 ... (NULL) */
static char *
shape_text(const struct shape *s, const double *th)
{
	struct sbuf b = {0};

	if (s->nm && F.oint)
		int_text(s, th == NULL, &b);
	else
		text(s, s->n - 1, th, th != NULL && F.w != 0, &b);
	return b.p;
}

/* the parameter list of a candidate: the target's, typed and ranged */
static void
params_text(struct sbuf *b)
{
	int v;

	for (v = 0; v < F.nv; v++) {
		char lo[48], hi[48];
		vfmt(lo, sizeof lo, v, F.vlo[v]);
		vfmt(hi, sizeof hi, v, F.vhi[v]);
		if (F.vint[v])
			sb_printf(b, "%s%s: %s in %s..%s", v ? ", " : "", F.vname[v],
			    ty_str(F.vty[v]), lo, hi);
		else
			sb_printf(b, "%s%s: %s in %s..%s", v ? ", " : "", F.vname[v],
			    fmtname(F.vw[v]), lo, hi);
	}
}

static struct func *
define(const char *name, const char *body)
{
	jmp_buf saved;
	struct toks *st = diag_toks;
	const char *sf = diag_file;
	uint32_t stok = diag_tok;
	struct sbuf b = {0};
	struct func *f;
	uint32_t sym = sym_cstr(name);

	sb_printf(&b, "%s(", name);
	params_text(&b);
	sb_printf(&b, ") = %s\n", body);
	if ((f = func_find(sym)) != NULL)
		f->body = 0;
	note_mute++;
	fail_mute++;
	memcpy(saved, fail_jmp, sizeof saved);
	run_source(b.p, "<adamfit>", 0);
	memcpy(fail_jmp, saved, sizeof saved);
	fail_mute--;
	note_mute--;
	diag_toks = st;
	diag_file = sf;
	diag_tok = stok;
	free(b.p);
	f = func_find(sym);
	if (f == NULL || f->body == 0)
		return NULL;
	return f;
}

/* drop a function this command made for its own use */
static void
forget(const char *name)
{
	uint32_t sym = sym_cstr(name);
	int i;

	for (i = 0; i < nfuncs; i++)
		if (funcs[i].name == sym) {
			memmove(&funcs[i], &funcs[i + 1],
			    (size_t)(nfuncs - i - 1) * sizeof *funcs);
			nfuncs--;
			return;
		}
}

#define TMP "_adamfit_"

static double
shape_cost(const struct shape *s)
{
	double th[AF_MAXK], c;
	struct func *f;
	char *t;
	int i;

	for (i = 0; i < AF_MAXK; i++)
		th[i] = 0.75 + i;	/* any constants: the price is the shape's */
	t = shape_text(s, th);
	f = define(TMP, t);
	free(t);
	c = f ? approx_price(f->body) : INFINITY;
	return c;
}

/* ---- measuring ---- */

static void
measure(struct shape *s, nid body)
{
	ld *ys = xmalloc((size_t)F.nm * sizeof *ys);
	double e = 0;
	long i;

	plan_batch(body, F.mx, F.nm, ys, F.oint);
	s->bad = 0;
	for (i = 0; i < F.nm; i++) {
		ld y = F.my[i], v = ys[i], ae, d;
		if (!isfinite(y))
			continue;
		if (!isfinite(v)) {
			s->bad++;
			continue;
		}
		ae = fabsl(v - y);
		if (F.oint)
			d = ae;
		else if (F.w && F.rel)
			d = ae / ulp_of(y);
		else if (F.rel)
			d = y != 0 ? ae / fabsl(y) : (ae == 0 ? 0 : INFINITY);
		else
			d = ae;
		if (d > e)
			e = (double)d;
	}
	free(ys);
	s->merr = s->bad ? INFINITY : e;
}

static double
score_of(double cost, double err)
{
	if (!isfinite(err) || !isfinite(cost))
		return INFINITY;
	return F.wcost * cost / F.c0 + F.werr * log2(1 + err / F.tol);
}

/* the sampled statement: with N independent uniform points all within the
 * measured error, more of the domain errs on at most q with probability
 * 1 - AF_DELTA */
static double
sampled_q(void)
{
	return 1 - exp(log(AF_DELTA) / (double)F.nm);
}

/*
 * Prove the integer candidate's error bound e on the target's whole
 * domain, |F - G| <= e, by enumeration or SAT (equiv_decide); 0 when it
 * could not be decided.
 */
static int
prove_int(struct shape *s, nid cb)
{
	struct eqproof pr;
	struct sbuf w = {0}, h = {0};
	struct func tf = F.f;
	nid a, b;
	int v, m;

	if (!F.has_f)
		return 0;
	for (v = 0; v < F.nv; v++)
		if (F.vint[v]) {
			tf.p[v].ranged = 1;
			tf.p[v].lo = (int64_t)F.vlo[v];
			tf.p[v].hi = (int64_t)F.vhi[v];
		}
	if (s->merr == 0) {
		a = F.f.body;
		b = cb;
	} else {
		/* max(|F - G| - e, 0) == 0 */
		ty_t t64 = TY(K_SINT, 64);
		nid d = mk_bin(O_SUB, mk_cast(t64, F.f.body), mk_cast(t64, cb));
		a = mk_const_i(t64, 0);
		b = mk_bin(O_MAX, mk_bin(O_SUB, mk_un(O_ABS, d), mk_const_i(t64,
		    (int64_t)s->merr)), a);
	}
	m = equiv_decide(&tf, a, b, &w, &pr);
	free(w.p);
	if (m != 0 || (pr.how != EQ_SAT && pr.how != EQ_EXHAUSTIVE &&
	    pr.how != EQ_ALGEBRA)) {
		eqproof_text(&h, &pr);
		snprintf(s->why, sizeof s->why, "%s", h.p ? h.p : "");
		free(h.p);
		return 0;
	}
	eqproof_text(&h, &pr);
	snprintf(s->why, sizeof s->why, "%s", h.p ? h.p : "");
	free(h.p);
	s->proof = pr.how == EQ_SAT || pr.how == EQ_ALGEBRA ? PF_SAT : PF_EXH;
	s->bound = s->merr;
	return 1;
}

/* the proof (or the statement) behind a listed candidate */
static void
prove(struct shape *s, struct func *cf)
{
	char e[40];

	s->proof = PF_NONE;
	s->bound = -1;
	s->why[0] = 0;
	if (!isfinite(s->merr))
		return;
	if (F.exhaustive) {
		/* every point of the domain was measured */
		s->proof = PF_EXH;
		s->bound = s->merr;
		snprintf(s->why, sizeof s->why, "every one of the domain's %ld "
		    "points measured", F.nm);
		return;
	}
	if (F.oint && F.cert && cf && prove_int(s, cf->body))
		return;
	if (!F.oint && F.cert && cf && F.kind == TK_UCAS && F.nv == 1 &&
	    F.lref_ok) {
		struct acert ac;
		memset(&ac, 0, sizeof ac);
		acert_run(F.f.body, cf->body, F.vlo[0], F.vhi[0], F.w, F.rel, 0, 0,
		    &ac);
		if (ac.ok) {
			s->proof = PF_CERT;
			s->bound = ac.bound;
			snprintf(s->why, sizeof s->why, "derived on the whole interval "
			    "(%ld pieces)", ac.pieces);
			return;
		}
		snprintf(s->why, sizeof s->why, "%s", ac.why);
	}
	s->proof = PF_SAMPLED;
	efmt(e, sizeof e, s->merr);
	{
		char note[200];
		snprintf(note, sizeof note, "sampled (not a proof): %ld uniform "
		    "points; with probability 1 - 1e-6 at most %.2g%% of the domain "
		    "errs more than %s", F.nm, 100 * sampled_q(), e);
		if (s->why[0]) {
			char t[200];
			snprintf(t, sizeof t, "%s", s->why);
			snprintf(s->why, sizeof s->why, "%s; not certified: %s", note, t);
		} else
			snprintf(s->why, sizeof s->why, "%s", note);
	}
}

/* ---- the run ---- */

static int
by_score1(const void *a, const void *b)
{
	const struct shape *x = a, *y = b;

	return (x->score1 > y->score1) - (x->score1 < y->score1);
}

/* stage 2's choice first, then by score1 */
static int
by_pick(const void *a, const void *b)
{
	const struct shape *x = a, *y = b;

	if (x->pick != y->pick)
		return y->pick - x->pick;
	return (x->score1 > y->score1) - (x->score1 < y->score1);
}

static int
by_cost_err(const void *a, const void *b)
{
	const struct shape *x = a, *y = b;

	if (x->cost != y->cost)
		return (x->cost > y->cost) - (x->cost < y->cost);
	return (x->terr > y->terr) - (x->terr < y->terr);
}

static int
by_score(const void *a, const void *b)
{
	const struct shape *x = a, *y = b;

	return (x->score > y->score) - (x->score < y->score);
}

static int
by_pair(const void *a, const void *b)
{
	const double *x = a, *y = b;

	return (x[0] > y[0]) - (x[0] < y[0]);
}

/* the candidates no other beats in both cost and error, cheapest first */
static void
front_text(struct sbuf *b, struct shape *s, int n, int stage2)
{
	int i, j, k = 0, np = 0;
	char e[32];
	double (*pr)[2] = xmalloc((size_t)(n + 1) * sizeof *pr);

	for (i = 0; i < n; i++) {
		double ci = s[i].cost, ei = stage2 ? s[i].merr :
		    tunit(s[i].terr);
		int dom = !isfinite(ei) || !s[i].ok;
		for (j = 0; j < n && !dom; j++) {
			double cj = s[j].cost, ej = stage2 ? s[j].merr :
			    tunit(s[j].terr);
			if (j != i && s[j].ok && cj <= ci && ej <= ei &&
			    (cj < ci || ej < ei || j < i))
				dom = 1;
		}
		if (dom)
			continue;
		pr[np][0] = ci;
		pr[np][1] = ei;
		np++;
	}
	xsort(pr, (size_t)np, sizeof *pr, by_pair);
	for (i = 0; i < np; i++) {
		efmt(e, sizeof e, pr[i][1]);
		sb_printf(b, "%s%g/%s", k++ ? ", " : "", pr[i][0], e);
		if (k >= 9 && i + 1 < np) {
			sb_puts(b, " ...");
			break;
		}
	}
	if (np == 0)
		sb_puts(b, "-");
	free(pr);
}

static const char *
proof_name(int k)
{
	switch (k) {
	case PF_CERT: return "certified";
	case PF_EXH: return "exhaustive";
	case PF_SAT: return "sat";
	case PF_SAMPLED: return "sampled";
	}
	return "none";
}

static void
run(void)
{
	int i, done = 0, nt = 0, n2, m2, n1;
	double best = INFINITY, best2 = INFINITY, next_report;
	struct shape *s;
	char e[32], e2[32];
	struct sbuf b = {0};

	clock_gettime(CLOCK_MONOTONIC, &F.t0);
	ns_seed(&F.rng, F.seed);
	if (F.kind != TK_UCAS)
		native_build();
	measuring();
	/* relative unless F is 0 somewhere (integers: absolute) */
	if (F.rel < 0) {
		long k;
		F.rel = !F.oint;
		for (k = 0; k < F.nm && F.rel; k++)
			if (F.my[k] == 0)
				F.rel = 0;
	}
	n1 = F.oint ? AF_T : F.nv == 1 ? 66 : 128;
	if (sample(n1) < 8)
		fail("adamfit: %s is undefined (or 0, for a relative error) at most "
		    "points of its domain", F.fname);
	if (F.tol_ulp > 0 && F.w && F.rel && !F.oint) {
		F.tol = F.tol_ulp;
		snprintf(F.tolname, sizeof F.tolname, "%g ulp", F.tol_ulp);
	} else if (F.tol_err > 0) {
		F.tol = F.w && F.rel && !F.oint ? F.tol_err * ldexp(1, F.prec - 1) :
		    F.tol_err;
		snprintf(F.tolname, sizeof F.tolname, "%g", F.tol_err);
	} else {
		if (F.oint) {
			F.tol = 1;
			snprintf(F.tolname, sizeof F.tolname, "1");
		} else if (F.w && F.rel) {
			F.tol = 1;
			snprintf(F.tolname, sizeof F.tolname, "1 ulp");
		} else {
			F.tol = F.w == 32 ? 1e-7 : 1e-15;
			snprintf(F.tolname, sizeof F.tolname, "%g", F.tol);
		}
	}

	if (F.oint)
		int_seeds();
	else {
		if (F.nv == 1)
			seeds1();
		else
			seedsn();
		randoms();
	}
	printf("adamfit %s", F.fname);
	sb_clear(&b);
	params_text(&b);
	printf(" (%s) -> %s: %d %s", b.p, F.oint ? ty_str(F.oty) : fmtname(F.w),
	    F.nsh, F.oint ? "fixed-point forms" : "shapes");
	if (!F.oint) {
		printf(" of up to %d nodes, ops", F.size);
		for (i = Q_ADD; i < Q_NOPS; i++)
			if (F.ops & (1u << i))
				printf(" %s", qname[i]);
	}
	printf("\n");
	if (F.kind != TK_UCAS)
		printf("  the %s function %s, sampled natively (%s, %s; no "
		    "contraction, -fwrapv)\n", F.kind == TK_GLSL ? "GLSL" : "C", F.cfn,
		    getenv("UCAS_CC") && *getenv("UCAS_CC") ? getenv("UCAS_CC") : "cc",
		    F.kind == TK_GLSL ? "its scalar subset as C, binary32" :
		    "as written");
	if (F.exhaustive)
		printf("  the domain's %ld points, every one measured\n", F.nm);
	else
		printf("  measured on %ld independent uniform points of the domain\n",
		    F.nm);
	printf("  fit: Adam (b1 0.9, b2 0.999, step %g decaying to a tenth), %d "
	    "steps from %d starts on %d points, the %s residual%s\n", F.lr,
	    F.steps, F.restarts, F.nt, F.oint ? "absolute" : F.rel ? "relative" :
	    "absolute", F.oint ? "; then the integers C, B and the shift by CMA-ES "
	    "on the exact error" : "; the best again longer, then polished "
	    "(Levenberg-Marquardt, Lawson)");
	printf("  score = %g * cost / %g (%s) + %g * log2(1 + error / %s)\n",
	    F.wcost, F.c0, F.c0why, F.werr, F.tolname);
	fflush(stdout);
	rep_begin("adamfit");
	rep_str("target", F.fname);
	rep_str("language", F.kind == TK_C ? "c" : F.kind == TK_GLSL ? "glsl" :
	    "ucas");
	rep_str("params", b.p);
	rep_str("result", F.oint ? ty_str(F.oty) : fmtname(F.w));
	rep_bool("integer", F.oint);
	rep_bool("exhaustive", F.exhaustive);
	rep_int("points", F.nm);
	rep_int("shapes", F.nsh);
	rep_num("wcost", F.wcost);
	rep_num("werr", F.werr);
	rep_num("cost_reference", F.c0);
	rep_num("tolerance", F.tol);

	/* stage 1: every shape, fitted by Adam (integers: and integerized) */
	next_report = F.nsh / 10.0;
	for (i = 0; i < F.nsh; i++) {
		s = &F.sh[i];
		train(s);
		if (F.oint && s->ok) {
			s->terr = INFINITY;
			integerize(s, i);
		}
		s->cost = shape_cost(s);
		done++;
		s->score1 = s->ok ? score_of(s->cost, tunit(s->terr)) : INFINITY;
		if (s->ok)
			nt++;
		if (!F.autom && s->score1 < best) {
			char *t = shape_text(s, s->th);
			efmt(e, sizeof e, tunit(s->terr));
			say("  [%5d/%d] score %.3f  cost %g  error ~%s  %s  (%s)", done,
			    F.nsh, s->score1, s->cost, e, t, s->kind);
			free(t);
		}
		if (s->score1 < best)
			best = s->score1;
		if (!F.autom && done >= next_report && done < F.nsh) {
			sb_clear(&b);
			front_text(&b, F.sh, done, 0);
			say("  ... %d/%d shapes, %d fitted, best score %.3f, front "
			    "(cost/error): %s", done, F.nsh, nt, best, b.p ? b.p : "");
			next_report += F.nsh / 10.0;
		}
		if (F.secs > 0 && elapsed() > F.secs) {
			say("  stopped after %.1f s: %d of %d shapes fitted", F.secs,
			    done, F.nsh);
			break;
		}
	}
	F.nsh = done;
	rep_int("fitted", nt);
	if (F.oint)
		rep_int("integer_forms_tried", F.tried);

	/*
	 * stage 2: the best at every cost, then the best scores; floats are
	 * trained again and polished; every one measured
	 */
	xsort(F.sh, (size_t)F.nsh, sizeof *F.sh, by_cost_err);
	m2 = F.count / 4 < 8 ? 8 : F.count / 4 > AF_STAGE2 ? AF_STAGE2 : F.count / 4;
	if (F.oint)
		m2 = F.nsh;
	n2 = 0;
	for (i = 0; i < F.nsh && n2 < m2 * 2 / 3; i++) {
		int j = i, k = 0;
		while (j < F.nsh && F.sh[j].cost == F.sh[i].cost) {
			if (F.sh[j].ok && k < AF_PERCOST && n2 < m2 * 2 / 3) {
				F.sh[j].pick = 1;
				k++;
				n2++;
			}
			j++;
		}
		i = j - 1;
	}
	xsort(F.sh, (size_t)F.nsh, sizeof *F.sh, by_score1);
	for (i = 0; i < F.nsh && n2 < m2; i++)
		if (F.sh[i].ok && !F.sh[i].pick) {
			F.sh[i].pick = 1;
			n2++;
		}
	xsort(F.sh, (size_t)F.nsh, sizeof *F.sh, by_pick);
	if (!F.oint) {
		sample(F.nv == 1 ? 257 : AF_T);
		if (!F.autom)
			say("  stage 2: the best %d shapes on %d points: Adam %d more "
			    "steps, then Levenberg-Marquardt and Lawson's reweighting",
			    n2, F.nt, F.steps * 10);
	} else if (!F.autom)
		say("  stage 2: the %d integer forms measured on the domain", n2);
	for (i = 0; i < n2; i++) {
		struct func *f;
		char *t;
		s = &F.sh[i];
		if (!F.oint) {
			double th[AF_MAXK], er;
			/* Adam on, nearer the largest error, then the polish from
			 * where it ends; the better of the two */
			s->terr = adam(s, s->th, F.steps * 10, F.lr * 0.2, 8, 32);
			memcpy(th, s->th, sizeof th);
			er = polish(s, th);
			if (er < s->terr) {
				s->terr = er;
				memcpy(s->th, th, sizeof th);
			}
		}
		t = shape_text(s, s->th);
		f = define(TMP, t);
		free(t);
		if (f == NULL) {
			s->merr = INFINITY;
		} else {
			s->cost = approx_price(f->body);
			measure(s, f->body);
		}
		s->bound = -1;
		s->score = score_of(s->cost, s->merr);
		if (!F.autom && s->score < best2) {
			t = shape_text(s, s->th);
			efmt(e2, sizeof e2, s->merr);
			say("  [stage 2 %2d/%d] score %.3f  cost %g  error %s  %s  (%s)",
			    i + 1, n2, s->score, s->cost, e2, t, s->kind);
			free(t);
		}
		if (s->score < best2)
			best2 = s->score;
	}
	forget(TMP);
	xsort(F.sh, (size_t)n2, sizeof *F.sh, by_score);
	{
		/* shapes that came to the same expression (an integer form's zero
		 * terms, a constant fitted to 1): once */
		uint64_t *h = xcalloc((size_t)n2 + 1, sizeof *h);
		int j, q;
		for (i = 0; i < n2; i++) {
			char *t = shape_text(&F.sh[i], F.sh[i].th);
			h[i] = hash_str(t);
			free(t);
			for (j = 0; j < i; j++)
				if (h[j] == h[i] && isfinite(F.sh[j].score)) {
					F.sh[i].score = INFINITY;
					break;
				}
		}
		free(h);
		xsort(F.sh, (size_t)n2, sizeof *F.sh, by_score);
		for (q = n2; q > 0 && !isfinite(F.sh[q - 1].score); q--)
			;
		n2 = q;
	}
	while (n2 > 0 && !isfinite(F.sh[n2 - 1].score))
		n2--;
	if (n2 == 0) {
		free(b.p);
		fail("adamfit: no shape could be fitted on the domain");
	}
	sb_clear(&b);
	front_text(&b, F.sh, n2, 1);
	printf("  measured: front (cost/error) %s\n", b.p);

	/* the closest candidates, proved where they can be; ranked by it */
	{
		int nl = n2 < F.list ? n2 : F.list;
		for (i = 0; i < nl; i++) {
			char nm[40], *t;
			struct func *f;
			s = &F.sh[i];
			snprintf(nm, sizeof nm, "%s%d", F.name, i + 1);
			t = shape_text(s, s->th);
			f = define(nm, t);
			free(t);
			prove(s, f);
			s->score = score_of(s->cost, s->bound >= 0 ? s->bound :
			    s->merr);
		}
		xsort(F.sh, (size_t)nl, sizeof *F.sh, by_score);
		printf("  %s candidates (defined as %s1 ...):\n", F.autom ? "closest" :
		    "best", F.name);
		printf("  #   score   cost  measured      proved             shape\n");
		rep_list("candidates");
		for (i = 0; i < nl; i++) {
			char nm[40], me[32], ce[48], *t, *k;
			s = &F.sh[i];
			snprintf(nm, sizeof nm, "%s%d", F.name, i + 1);
			t = shape_text(s, s->th);
			k = shape_text(s, NULL);
			define(nm, t);
			efmt(me, sizeof me, s->merr);
			if (s->bound >= 0) {
				char bb[32];
				efmt(bb, sizeof bb, s->bound);
				snprintf(ce, sizeof ce, "%s%s (%s)", s->bound == 0 ? "" :
				    "<= ", s->bound == 0 ? "exact" : bb, proof_name(s->proof));
			} else
				snprintf(ce, sizeof ce, s->proof == PF_SAMPLED ? "sampled" :
				    "-");
			printf("  %-3d %-7.3f %-5g %-13s %-18s %s (%s)\n", i + 1,
			    s->score, s->cost, me, ce, k, s->kind);
			{
				int v;
				printf("      %s(", nm);
				for (v = 0; v < F.nv; v++)
					printf("%s%s", v ? ", " : "", F.vname[v]);
				printf(") = %s\n", t);
			}
			if (s->why[0] && (s->proof == PF_SAMPLED || s->proof == PF_SAT ||
			    s->proof == PF_NONE))
				printf("      %s\n", s->why);
			rep_obj(NULL);
			rep_str("name", nm);
			rep_str("expr", t);
			rep_str("shape", k);
			rep_str("kind", s->kind);
			rep_num("cost", s->cost);
			rep_num("error", s->merr);
			rep_str("proof", proof_name(s->proof));
			rep_bool("certified", s->proof == PF_CERT);
			if (s->bound >= 0)
				rep_num("bound", s->bound);
			if (s->why[0])
				rep_str("why", s->why);
			rep_num("score", s->score);
			rep_end();
			free(t);
			free(k);
		}
		rep_end();
		if (nl && F.sh[0].bound >= 0)
			rep_proof(F.sh[0].proof == PF_SAT ? "sat" : F.sh[0].proof ==
			    PF_EXH ? "exhaustive" : "symbolic", F.sh[0].proof == PF_CERT ?
			    "the best candidate's error bound is derived on the whole "
			    "interval (acert)" : F.sh[0].why);
		else if (nl)
			rep_proof("numeric", F.sh[0].why);
	}
	if (F.oint)
		printf("  %ld integer forms evaluated in the search\n", F.tried);
	if (getenv("UCAS_NO_TIMES") == NULL)
		printf("  %.2f s\n", elapsed());
	free(b.p);
}

static void
reset(void)
{
	int v;

	free(F.sh);
	free(F.seen);
	for (v = 0; v < AF_MAXV; v++)
		free(F.mx[v]);
	free(F.my);
	free(F.code);
	if (F.ref.n)
		lref_free(&F.ref);
	native_free();
	for (v = 0; v < AF_MAXK; v++) {
		free(mval[v]);
		mval[v] = NULL;
	}
	for (v = 0; v < AF_MAXV; v++) {
		free(ipt[v]);
		ipt[v] = NULL;
	}
	free(yint);
	yint = NULL;
	nip = 0;
	memset(&F, 0, sizeof F);
}

static const char *const optwords[] = {"in", "as", "ops", "count", "size",
	"steps", "lr", "restarts", "wcost", "werr", "error", "ulp", "relative",
	"absolute", "auto", "list", "seed", "seconds", "name", "nocert", "cref",
	"budget", NULL};

/* number parsing stops at the options' words and the parameters' names
 * ("list 2 r in 0..1" is not 2*r) */
static const char *af_stop[64];

static void
stops(void)
{
	int k = 0, v;

	while (optwords[k] && k < 56) {
		af_stop[k] = optwords[k];
		k++;
	}
	for (v = 0; v < F.nv; v++)
		af_stop[k++] = F.vname[v];
	af_stop[k] = NULL;
}

static unsigned
op_bit(const char *w)
{
	int i;

	if (!strcmp(w, "add") || !strcmp(w, "+"))
		return 1u << Q_ADD;
	if (!strcmp(w, "sub") || !strcmp(w, "-"))
		return 1u << Q_SUB;
	if (!strcmp(w, "mul") || !strcmp(w, "*"))
		return 1u << Q_MUL;
	if (!strcmp(w, "div") || !strcmp(w, "/"))
		return 1u << Q_DIV;
	for (i = Q_FMA; i < Q_NOPS; i++)
		if (!strcmp(w, qname[i]))
			return 1u << i;
	return 0;
}

/* ---- the targets ---- */

/* a parameter of the target, from its ucas type */
static void
add_param(const char *name, ty_t t, int ranged, double lo, double hi)
{
	int v = F.nv;

	if (v >= AF_MAXV)
		fail("adamfit: at most %d parameters", AF_MAXV);
	snprintf(F.vname[v], sizeof F.vname[v], "%s", name);
	if (t == 0)
		fail("adamfit: %s's type is not a number type", name);
	F.vty[v] = t;
	F.vint[v] = TY_ISINT(t) && TYK(t) != K_LIT;
	F.vw[v] = TYK(t) == K_FLOAT ? TYW(t) : 0;
	if (TYK(t) == K_FLOAT && TYW(t) == 16)
		fail("adamfit: %s is f16, which it has no format for", name);
	F.vranged[v] = ranged;
	F.vlo[v] = lo;
	F.vhi[v] = hi;
	F.nv++;
}

static void
set_result(ty_t t)
{
	F.oty = t;
	F.oint = TY_ISINT(t) && TYK(t) != K_LIT;
	F.ow = TYK(t) == K_INT ? 64 : TYW(t);
	F.osigned = TYK(t) == K_SINT || TYK(t) == K_INT;
	if (TYK(t) == K_INT)
		F.oty = TY(K_SINT, 64);
}

/* a ucas function or expression */
static void
target_ucas(struct func *f)
{
	int v;

	F.kind = TK_UCAS;
	F.f = *f;
	F.has_f = 1;
	snprintf(F.fname, sizeof F.fname, "%s", F.f.anon ? node_str(F.f.body) :
	    sym_str(F.f.name));
	if (F.f.np < 1)
		fail("adamfit needs a function of 1 to %d parameters", AF_MAXV);
	for (v = 0; v < F.f.np; v++) {
		const struct param *p = &F.f.p[v];
		int ranged = p->ranged;
		double lo = TY_ISINT(p->ty) ? (double)p->lo : p->rlo;
		double hi = TY_ISINT(p->ty) ? (double)p->hi : p->rhi;
		add_param(sym_str(p->name), p->ty, ranged, lo, hi);
	}
	set_result(N.ty[F.f.body]);
	F.c0 = approx_price(F.f.body);
	snprintf(F.c0why, sizeof F.c0why, "%s's cost under %s", F.fname,
	    model_active() ? sym_str(model_active()->name) : "-");
}

/* a GLSL function's signature: "float name(float x, uint n)" */
struct gsig {
	char ret[16], name[64];
	int np;
	char pty[AF_MAXV][16], pname[AF_MAXV][40];
};

static ty_t
glsl_ty(const char *t)
{
	if (!strcmp(t, "float"))
		return TY_F32;
	if (!strcmp(t, "double"))
		return TY_F64;
	if (!strcmp(t, "int"))
		return TY(K_SINT, 32);
	if (!strcmp(t, "uint"))
		return TY(K_UINT, 32);
	if (!strcmp(t, "bool"))
		return TY(K_BOOL, 1);
	return 0;
}

/* the functions defined in GLSL source (comments and preprocessor lines
 * skipped): up to max */
static int
glsl_funcs(const char *s, struct gsig *out, int max)
{
	char tok[3][64];
	int n = 0, depth = 0, nt = 0;
	const char *p = s;

	while (*p && n < max) {
		if (p[0] == '/' && p[1] == '/') {
			while (*p && *p != '\n')
				p++;
			continue;
		}
		if (p[0] == '/' && p[1] == '*') {
			const char *e = strstr(p + 2, "*/");
			p = e ? e + 2 : p + strlen(p);
			continue;
		}
		if (*p == '#') {
			while (*p && *p != '\n')
				p++;
			continue;
		}
		if (*p == '{') {
			depth++;
			p++;
			nt = 0;
			continue;
		}
		if (*p == '}') {
			depth--;
			p++;
			nt = 0;
			continue;
		}
		if (depth == 0 && (isalpha((unsigned char)*p) || *p == '_')) {
			const char *q = p;
			while (is_word(*q))
				q++;
			memmove(tok[0], tok[1], sizeof tok[0]);
			memmove(tok[1], tok[2], sizeof tok[0]);
			snprintf(tok[2], sizeof tok[2], "%.*s", (int)(q - p), p);
			nt++;
			p = q;
			continue;
		}
		if (depth == 0 && *p == '(' && nt >= 2) {
			/* ret name ( params ) { : a definition */
			const char *e = strchr(p, ')'), *b;
			struct gsig g;
			memset(&g, 0, sizeof g);
			if (e == NULL)
				break;
			b = e + 1;
			while (*b == ' ' || *b == '\t' || *b == '\n' || *b == '\r')
				b++;
			if (*b == '{') {
				char ps[400], *save = NULL, *it;
				snprintf(g.ret, sizeof g.ret, "%s", tok[1]);
				snprintf(g.name, sizeof g.name, "%s", tok[2]);
				snprintf(ps, sizeof ps, "%.*s", (int)(e - p - 1), p + 1);
				for (it = strtok_r(ps, ",", &save); it; it = strtok_r(NULL,
				    ",", &save)) {
					char w[6][40];
					int nw = 0;
					char *s2 = NULL, *x;
					for (x = strtok_r(it, " \t\n\r", &s2); x && nw < 6;
					    x = strtok_r(NULL, " \t\n\r", &s2))
						snprintf(w[nw++], sizeof w[0], "%s", x);
					if (nw == 0 || (nw == 1 && !strcmp(w[0], "void")))
						continue;
					if (g.np >= AF_MAXV) {
						g.np = AF_MAXV + 1;
						break;
					}
					{
						int k;
						for (k = 0; k < nw; k++)
							if (!strcmp(w[k], "out") || !strcmp(w[k], "inout"))
								g.np = AF_MAXV + 2;
					}
					if (g.np > AF_MAXV)
						break;
					snprintf(g.pty[g.np], sizeof g.pty[0], "%s", nw >= 2 ?
					    w[nw - 2] : w[0]);
					snprintf(g.pname[g.np], sizeof g.pname[0], "%s", nw >= 2 ?
					    w[nw - 1] : "x");
					g.np++;
				}
				out[n++] = g;
				p = b;
				nt = 0;
				continue;
			}
		}
		if (!isspace((unsigned char)*p))
			nt = *p == '(' || *p == ')' || *p == ',' ? nt : 0;
		p++;
	}
	return n;
}

static void
target_glsl(const char *src, const char *want)
{
	struct gsig g[64];
	int n = glsl_funcs(src, g, 64), k = -1, v;

	if (n == 0)
		fail("adamfit: no function in the GLSL block");
	for (v = 0; v < n; v++)
		if (want[0] ? !strcmp(g[v].name, want) : strcmp(g[v].name, "main"))
			k = v;		/* named, else the last that is not main */
	if (k < 0)
		fail("adamfit: no function %s in the GLSL block", want[0] ? want :
		    "but main");
	if (g[k].np > AF_MAXV)
		fail("adamfit: %s takes %s", g[k].name, g[k].np == AF_MAXV + 1 ?
		    "more than 4 parameters" : "an out or inout parameter");
	if (g[k].np == 0)
		fail("adamfit: %s takes no parameters", g[k].name);
	F.kind = TK_GLSL;
	snprintf(F.cfn, sizeof F.cfn, "%s", g[k].name);
	snprintf(F.fname, sizeof F.fname, "GLSL %s", g[k].name);
	for (v = 0; v < g[k].np; v++) {
		ty_t t = glsl_ty(g[k].pty[v]);
		if (t == 0)
			fail("adamfit: %s's parameter %s is %s: the scalar types only "
			    "(float, double, int, uint)", g[k].name, g[k].pname[v],
			    g[k].pty[v]);
		add_param(g[k].pname[v], t, 0, (double)tmin(t), (double)tmax(t));
	}
	if (glsl_ty(g[k].ret) == 0)
		fail("adamfit: %s returns %s: the scalar types only", g[k].name,
		    g[k].ret);
	set_result(glsl_ty(g[k].ret));
	F.code = glsl_as_c(src);
	F.c0 = F.cref > 0 ? F.cref : 8;
	snprintf(F.c0why, sizeof F.c0why, F.cref > 0 ? "cref" : "8: the GLSL's own "
	    "cost is not known (cref C sets it)");
}

static void
target_c(const char *src, const char *want)
{
	struct cf_unit u;
	int k, pick = -1, v;
	uint32_t ty, fn = 0;

	memset(&u, 0, sizeof u);
	if (!cf_preprocess(&u, "<block>", src, cf_target_default()) ||
	    !cf_parse(&u)) {
		char m[600];
		snprintf(m, sizeof m, "%s", u.err.p ? u.err.p : "?");
		cf_unit_free(&u);
		fail("adamfit: the C does not parse: %s", m);
	}
	for (k = 0; k < u.nfn; k++) {
		uint32_t f = u.fns[k];
		const char *nm = sym_str(u.S.name[u.A.iv[f]]);
		if (u.tk.file[u.A.t0[f]] != 0)
			continue;
		if (want[0] ? !strcmp(nm, want) : strcmp(nm, "main") != 0)
			pick = k;
	}
	if (pick < 0) {
		cf_unit_free(&u);
		fail("adamfit: no function %s in the C block", want[0] ? want :
		    "but main");
	}
	fn = u.fns[pick];
	F.kind = TK_C;
	snprintf(F.cfn, sizeof F.cfn, "%s", sym_str(u.S.name[u.A.iv[fn]]));
	snprintf(F.fname, sizeof F.fname, "C %s", F.cfn);
	ty = ct_unqual(u.S.ty[u.A.iv[fn]]);
	if (CT.kind[ty] != CT_FUNC || CT.nf[ty] == 0 || CT.nf[ty] > AF_MAXV) {
		cf_unit_free(&u);
		fail("adamfit: %s takes 1 to %d parameters", F.cfn, AF_MAXV);
	}
	for (v = 0; v < (int)CT.nf[ty]; v++) {
		uint32_t pt = ct_unqual(CT.fty[CT.fst[ty] + v]);
		uint32_t pn = CT.fname[CT.fst[ty] + v];
		ty_t t = ct_to_ucas(pt);
		char nm[40];
		snprintf(nm, sizeof nm, "%s", pn ? sym_str(pn) : "x");
		if (t == 0 || TYK(t) == K_BOOL) {
			cf_unit_free(&u);
			fail("adamfit: %s's parameter %s is not an integer or float", F.cfn,
			    nm);
		}
		add_param(nm, t, 0, (double)tmin(t), (double)tmax(t));
	}
	{
		ty_t rt = ct_to_ucas(ct_unqual(CT.base[ty]));
		if (rt == 0) {
			cf_unit_free(&u);
			fail("adamfit: %s returns something that is not a number", F.cfn);
		}
		set_result(rt);
	}
	/* lowered by ucas: its price, and proofs over it */
	{
		struct cf_value cv;
		memset(&cv, 0, sizeof cv);
		if (cf_lower_func(&u, fn, &cv, 1)) {
			struct func lf;
			int same = 1;
			memset(&lf, 0, sizeof lf);
			cf_value_func(&cv, &lf);
			same = lf.np == F.nv;
			for (v = 0; same && v < F.nv; v++)
				same = !strcmp(sym_str(lf.p[v].name), F.vname[v]) &&
				    lf.p[v].ty == F.vty[v];
			F.c0 = approx_price(cv.e);
			snprintf(F.c0why, sizeof F.c0why, "%s's cost under %s, as ucas "
			    "lowers it", F.cfn, model_active() ?
			    sym_str(model_active()->name) : "-");
			if (same) {
				lf.body = cv.e;
				F.f = lf;
				F.has_f = 1;
			}
		} else {
			F.c0 = F.cref > 0 ? F.cref : 8;
			snprintf(F.c0why, sizeof F.c0why, F.cref > 0 ? "cref" : "8: %s "
			    "does not lower to one value (cref C sets it)", F.cfn);
			free((char *)cv.why);
		}
	}
	if (F.cref > 0) {
		F.c0 = F.cref;
		snprintf(F.c0why, sizeof F.c0why, "cref");
	}
	cf_unit_free(&u);
	F.code = xmalloc(strlen(src) + 1);
	strcpy(F.code, src);
}

/* the code for adamfit: the GLSL before compilation, or C */
static int
code_target(const char **src, int *glsl)
{
	int cl;

	if (code_block_glsl) {
		*src = code_block_glsl;
		*glsl = 1;
		return 1;
	}
	if (code_block == NULL)
		return 0;
	cl = code_lang(code_block, code_block_lang, NULL);
	if (cl == CL_SPV)
		fail("adamfit samples C or GLSL; SPIR-V assembly is not run here "
		    "(give the GLSL)");
	*src = code_block;
	*glsl = cl == CL_GLSL;
	return 1;
}

int
adamfit_wants_glsl(const char *stmt)
{
	while (*stmt == ' ' || *stmt == '\t' || *stmt == '\n')
		stmt++;
	return !strncmp(stmt, "adamfit", 7) && (stmt[7] == 0 || stmt[7] == ' ' ||
	    stmt[7] == '\n' || stmt[7] == '\t');
}

/* "x in a..b" for parameter x; 1 when it was one */
static int
var_range(struct P *p)
{
	uint32_t at = cmdp_pos(p);
	char w[40];
	int v;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	for (v = 0; v < F.nv; v++)
		if (!strcmp(w, F.vname[v]))
			break;
	if (v == F.nv || !cmdp_kw(p, "in")) {
		cmdp_seek(p, at);
		return 0;
	}
	F.vlo[v] = cmdp_number(p, af_stop);
	if (!cmdp_tok(p, T_DOTDOT))
		fail("expected '..' in %s's range", w);
	F.vhi[v] = cmdp_number(p, af_stop);
	F.vranged[v] = 1;
	(void)cmdp_tok(p, T_COMMA);
	return 1;
}

int
adamfit_command(struct P *p)
{
	struct func *f;
	uint32_t at = cmdp_pos(p);
	char w[64];
	int fmt = -1, v, glsl = 0;
	double tol_err = 0, tol_ulp = 0;
	const char *src = NULL;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (strcmp(w, "adamfit") != 0 || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	reset();
	F.ops = 1u << Q_ADD | 1u << Q_SUB | 1u << Q_MUL | 1u << Q_DIV |
	    1u << Q_FMA | 1u << Q_SQRT;
	F.count = 2000;
	F.size = 9;
	F.steps = 200;
	F.restarts = 3;
	F.lr = 0.05;
	F.wcost = 0.8;
	F.werr = 0.5;
	F.list = 6;
	F.cert = 1;
	F.ibudget = 1500;
	F.rel = -1;
	F.seed = stoch_seed();
	snprintf(F.name, sizeof F.name, "af");

	/*
	 * the target: a ucas function or expression when the statement names
	 * one (a defined function, a call like sin(x), an expression in
	 * parentheses); else, with a code block, one of the code's functions
	 * (a block follows several statements in a file: it is not every
	 * one's)
	 */
	{
		uint32_t q = cmdp_pos(p);
		char nw[64];
		int ucas_named = 0;
		if (!cmdp_word(p, nw, sizeof nw))
			ucas_named = !cmdp_eos(p);	/* "(x: u8) / 3", "2*x" */
		else {
			int k;
			for (k = 0; optwords[k] && strcmp(nw, optwords[k]); k++)
				;
			/* a code function's name is followed by options or ranges
			 * (words) or nothing; anything else is an expression */
			ucas_named = !optwords[k] && (func_find(sym_cstr(nw)) != NULL ||
			    (!cmdp_eos(p) && !cmdp_at(p, T_ID)));
		}
		cmdp_seek(p, q);
		if (ucas_named)
			src = NULL;
		else if (!code_target(&src, &glsl))
			src = NULL;
	}
	if (src) {
		char want[64] = "";
		uint32_t q = cmdp_pos(p);
		code_block_used = 1;
		if (cmdp_word(p, want, sizeof want)) {
			int k;
			for (k = 0; optwords[k]; k++)
				if (!strcmp(want, optwords[k]))
					break;
			if (optwords[k] || !strcmp(want, "this") || !strcmp(want, "it")) {
				if (optwords[k])
					cmdp_seek(p, q);
				want[0] = 0;
			}
		}
		if (glsl)
			target_glsl(src, want);
		else
			target_c(src, want);
	} else {
		f = cmdp_func(p, optwords);
		target_ucas(f);
	}
	stops();
	if (cmdp_kw(p, "in")) {
		double lo = cmdp_number(p, af_stop), hi;
		if (!cmdp_tok(p, T_DOTDOT))
			fail("expected '..' in the interval");
		hi = cmdp_number(p, af_stop);
		for (v = 0; v < F.nv; v++) {
			F.vlo[v] = lo;
			F.vhi[v] = hi;
			F.vranged[v] = 1;
		}
		(void)cmdp_tok(p, T_COMMA);
	}
	while (var_range(p))
		;
	while (!cmdp_eos(p)) {
		cmdp_fail_here(p);
		if (var_range(p))
			continue;	/* "x in a..b" among the options too */
		if (cmdp_kw(p, "as")) {
			if (cmdp_kw(p, "f32") || cmdp_kw(p, "float"))
				fmt = 32;
			else if (cmdp_kw(p, "f64") || cmdp_kw(p, "double"))
				fmt = 64;
			else if (cmdp_kw(p, "real"))
				fmt = 0;
			else
				fail("as f32, f64 or real");
		} else if (cmdp_kw(p, "ops")) {
			/* + - * / are tokens, the rest words */
			F.ops = 0;
			for (;;) {
				unsigned bit = 0;
				if (cmdp_tok(p, T_PLUS))
					bit = 1u << Q_ADD;
				else if (cmdp_tok(p, T_MINUS))
					bit = 1u << Q_SUB;
				else if (cmdp_tok(p, T_STAR))
					bit = 1u << Q_MUL;
				else if (cmdp_tok(p, T_SLASH))
					bit = 1u << Q_DIV;
				else {
					uint32_t q = cmdp_pos(p);
					char ow[16];
					if (!cmdp_word(p, ow, sizeof ow))
						break;
					if ((bit = op_bit(ow)) == 0) {
						cmdp_seek(p, q);
						break;
					}
				}
				F.ops |= bit;
				(void)cmdp_tok(p, T_COMMA);
			}
			if (!(F.ops & (1u << Q_ADD | 1u << Q_MUL | 1u << Q_FMA)))
				fail("ops: at least + or * (or fma) among + - * / fma "
				    "sqrt exp log sin cos abs");
		} else if (cmdp_kw(p, "count")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 1) || x > 1e6)
				fail("count: shapes from 1 to 1000000, not %g", x);
			F.count = (int)x;
		} else if (cmdp_kw(p, "size")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 2) || x > 20)
				fail("size: nodes of a random shape, 2 to 20, not %g", x);
			F.size = (int)x;
		} else if (cmdp_kw(p, "steps")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 10) || x > 100000)
				fail("steps: Adam's steps per start, 10 to 100000, not %g",
				    x);
			F.steps = (int)x;
		} else if (cmdp_kw(p, "lr")) {
			F.lr = cmdp_number(p, af_stop);
			if (!(F.lr > 0) || F.lr > 10)
				fail("lr: Adam's step, above 0 and at most 10");
		} else if (cmdp_kw(p, "restarts")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 1) || x > 64)
				fail("restarts: starts per shape, 1 to 64, not %g", x);
			F.restarts = (int)x;
		} else if (cmdp_kw(p, "wcost")) {
			F.wcost = cmdp_number(p, af_stop);
			if (!(F.wcost >= 0))
				fail("wcost: a weight, 0 or more");
		} else if (cmdp_kw(p, "werr")) {
			F.werr = cmdp_number(p, af_stop);
			if (!(F.werr >= 0))
				fail("werr: a weight, 0 or more");
		} else if (cmdp_kw(p, "cref")) {
			F.cref = cmdp_number(p, af_stop);
			if (!(F.cref > 0))
				fail("cref: the target's cost, above 0");
			F.c0 = F.cref;
			snprintf(F.c0why, sizeof F.c0why, "cref");
		} else if (cmdp_kw(p, "budget")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 10) || x > 1e7)
				fail("budget: the integer search's evaluations, 10 to "
				    "10000000, not %g", x);
			F.ibudget = (int)x;
		} else if (cmdp_kw(p, "error"))
			tol_err = cmdp_number(p, af_stop);
		else if (cmdp_kw(p, "ulp"))
			tol_ulp = cmdp_number(p, af_stop);
		else if (cmdp_kw(p, "relative"))
			F.rel = 1;
		else if (cmdp_kw(p, "absolute"))
			F.rel = 0;
		else if (cmdp_kw(p, "auto"))
			F.autom = 1;
		else if (cmdp_kw(p, "nocert"))
			F.cert = 0;
		else if (cmdp_kw(p, "list")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 1) || x > 64)
				fail("list: 1 to 64 candidates, not %g", x);
			F.list = (int)x;
		} else if (cmdp_kw(p, "seed")) {
			double x = cmdp_number(p, af_stop);
			if (!(x >= 0) || x > 9e15)
				fail("seed: a whole number, not %g", x);
			F.seed = (uint64_t)x;
		} else if (cmdp_kw(p, "seconds")) {
			F.secs = cmdp_number(p, af_stop);
			if (!(F.secs > 0))
				fail("seconds: a time limit above 0");
		} else if (cmdp_kw(p, "name")) {
			if (!cmdp_word(p, F.name, sizeof F.name))
				fail("name needs an identifier");
		} else
			fail("unknown adamfit option (see help)");
	}
	notes_flush();
	/* the domain */
	for (v = 0; v < F.nv; v++) {
		if (!F.vranged[v] && !F.vint[v])
			fail(F.nv == 1 ? "give an interval: adamfit F in a..b" :
			    "give %s a range: %s in a..b", F.vname[v], F.vname[v]);
		if (!F.vranged[v]) {
			if (TYK(F.vty[v]) == K_INT)
				fail("give %s a range: %s in a..b (an int is unbounded)",
				    F.vname[v], F.vname[v]);
			F.vlo[v] = (double)tmin(F.vty[v]);
			F.vhi[v] = (double)tmax(F.vty[v]);
		}
		if (!(F.vlo[v] < F.vhi[v]) || !isfinite(F.vlo[v]) ||
		    !isfinite(F.vhi[v]))
			fail("%s's range must be finite and non-empty", F.vname[v]);
		if (F.vint[v]) {
			F.vlo[v] = ceil(F.vlo[v]);
			F.vhi[v] = floor(F.vhi[v]);
		} else if (F.vw[v]) {
			F.vlo[v] = rnd(F.vw[v], F.vlo[v]);
			F.vhi[v] = rnd(F.vw[v], F.vhi[v]);
		}
	}
	if (F.oint) {
		for (v = 0; v < F.nv; v++)
			if (!F.vint[v])
				fail("adamfit: an integer result of a float parameter (%s) is "
				    "not searched yet: integer results take integer parameters",
				    F.vname[v]);
		if (TYK(F.oty) == K_BOOL)
			fail("adamfit: a bool result is a decision, not a value to fit");
		F.w = 0;
		F.rel = 0;
	} else {
		/* the candidates' format: as, else the result's, else f32 */
		F.w = fmt >= 0 ? fmt : TYK(F.oty) == K_FLOAT ? TYW(F.oty) : 32;
		if (TYK(F.oty) == K_FLOAT && TYW(F.oty) == 16)
			fail("adamfit fits f32, f64 or real formats: the result is f16");
		/* real parameters take the candidates' format */
		for (v = 0; v < F.nv; v++)
			if (!F.vint[v] && TYK(F.vty[v]) == K_REAL) {
				F.vw[v] = F.w;
				F.vlo[v] = rnd(F.w, F.vlo[v]);
				F.vhi[v] = rnd(F.w, F.vhi[v]);
			}
		if (F.kind == TK_UCAS && F.nv == 1 && !F.vint[0]) {
			lref_init(&F.ref, F.f.body, 0);
			F.lref_ok = 1;
		}
	}
	F.prec = F.w == 32 ? 24 : 53;
	F.emin = F.w == 32 ? -126 : -1022;
	F.tol_err = tol_err;
	F.tol_ulp = tol_ulp;
	run();
	native_free();
	return 1;
}

static char *
ci_find(char *s, const char *w)
{
	size_t n = strlen(w);

	for (; *s; s++)
		if (!strncasecmp(s, w, n))
			return s;
	return NULL;
}

/*
 * English: "fit sin(x) with adam on 0..1", "adam fit exp(x) in 0..1",
 * "search for cheap approximations of sin(x) in 0..1 by gradient descent",
 * "fit this with adam" (a code block) ... becomes adamfit; "and just
 * report the closest" / "automatically" adds auto.
 */
const char *
adamfit_nlp_raw(const char *sent)
{
	static char out[1200];
	char buf[1000], tail[80] = "", *s, *q;
	size_t n;
	static const char *const pre[] = {"adam fit ", "adamfit ", "fit with adam ",
		"neural fit ", "train expressions for ", "train an expression for ",
		"evolve expressions for ", "search expressions for ",
		"search for cheap expressions for ", "search for cheap approximations of ",
		"find cheap expressions for ", "find cheap approximations of ", NULL};
	static const char *const post[] = {" with adam", " using adam", " by adam",
		" with gradient descent", " by gradient descent",
		" using gradient descent", " with a neural fitter", NULL};
	static const char *const autos[] = {" and just report the closest",
		" and report the closest", " automatically", " in auto mode",
		" quietly", NULL};
	int i, adam = 0;

	while (*sent == ' ' || *sent == '\t')
		sent++;
	if (strlen(sent) >= sizeof buf)
		return NULL;
	strcpy(buf, sent);
	n = strlen(buf);
	while (n && (buf[n - 1] == ' ' || buf[n - 1] == '?' || buf[n - 1] == '.' ||
	    buf[n - 1] == '\n'))
		buf[--n] = 0;
	s = buf;
	if (!strncasecmp(s, "please ", 7))
		s += 7;
	for (i = 0; autos[i]; i++)
		if ((q = ci_find(s, autos[i])) != NULL) {
			memmove(q, q + strlen(autos[i]), strlen(q + strlen(autos[i])) +
			    1);
			snprintf(tail, sizeof tail, " auto");
		}
	for (i = 0; post[i]; i++)
		if ((q = ci_find(s, post[i])) != NULL) {
			memmove(q, q + strlen(post[i]), strlen(q + strlen(post[i])) + 1);
			adam = 1;
		}
	for (i = 0; pre[i]; i++)
		if (!strncasecmp(s, pre[i], strlen(pre[i]))) {
			s += strlen(pre[i]);
			adam = 1;
			break;
		}
	if (!adam && !strncasecmp(s, "fit ", 4) && strstr(sent, "adam")) {
		s += 4;
		adam = 1;
	}
	{
		/* "fit F with adam": the verb before F */
		static const char *const verbs[] = {"fit ", "approximate ",
			"approximations of ", "search for ", "search ", "train ",
			"find ", NULL};
		for (i = 0; adam && verbs[i]; i++)
			if (!strncasecmp(s, verbs[i], strlen(verbs[i]))) {
				s += strlen(verbs[i]);
				break;
			}
	}
	if (!adam || !*s)
		return NULL;
	/* a code block's function: "this", "it", "the function" */
	if (!strcasecmp(s, "this") || !strcasecmp(s, "it") ||
	    !strcasecmp(s, "the function") || !strcasecmp(s, "this function")) {
		snprintf(out, sizeof out, "adamfit%s", tail);
		return out;
	}
	/* "on 0..1", "from 0 to 1", "over [0, 1]" as in a..b */
	if ((q = strstr(s, " from ")) != NULL && strstr(q, " to ")) {
		char *t = strstr(q, " to ");
		*q = 0;
		*t = 0;
		snprintf(out, sizeof out, "adamfit %s in %s..%s%s", s, q + 6, t + 4,
		    tail);
		return out;
	}
	if ((q = strstr(s, " on ")) != NULL && strstr(q, ".."))
		memcpy(q, " in ", 4);
	if ((q = strstr(s, " over ")) != NULL && strstr(q, "..")) {
		memmove(q + 4, q + 6, strlen(q + 6) + 1);
		memcpy(q, " in ", 4);
	}
	snprintf(out, sizeof out, "adamfit %s%s", s, tail);
	return out;
}
