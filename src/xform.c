/*
 * xform.c: integral and discrete transforms (xform.h).
 *
 *   laplace F [t] [s] [period T]   L{f}(s) = int_0^inf f(t) e^(-st) dt
 *   ilaplace F [s] [t]             by partial fractions, steps e^(-as)
 *   laplace solve ODE, ICs         linear, constant coefficients
 *   ztrans X [n] [z]               Z{x}(z) = sum_{n>=0} x[n] z^-n
 *   iztrans X [z] [n]              partial fractions of X(z)/z, and long
 *                                  division for the first terms
 *   zsolve REC, ICs                linear recurrences
 *   tf EQ | H                      transfer function: poles, zeros,
 *                                  stability, impulse/frequency response
 *   fourier series F [x] [on a..b] [terms N]
 *   fourier F [x] [w], ifourier F  continuous transform (table + rules)
 *   dft|fft|idft|ifft [x0, x1, ...]
 *   dtft X [n] | dtft [x0, ...]
 *   genfunc A [n] [x], egf A       ordinary/exponential generating
 *                                  functions; genfunc REC, ICs
 *
 * Every result is checked numerically: a Laplace transform against the
 * integral at sample s, an inverse by transforming it back (the integral
 * again), a Z transform by summing the series at sample z, an inverse Z
 * transform against long division, ODE/recurrence solutions by their
 * residual and an independent numeric solution.
 */
#include "xform.h"
#include "calc2.h"
#include "props.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- text helpers ---- */

static char *
xstrdup(const char *s)
{
	char *r = xmalloc(strlen(s) + 1);

	strcpy(r, s);
	return r;
}

static void
trim(char *s)
{
	size_t n = strlen(s), i = 0;

	while (n > 0 && isspace((unsigned char)s[n - 1]))
		s[--n] = 0;
	while (isspace((unsigned char)s[i]))
		i++;
	if (i)
		memmove(s, s + i, n - i + 1);
}

static int
wch(int c)
{
	return isalnum(c) || c == '_' || (c & 0x80);
}

/*
 * The first top-level (depth 0) occurrence of a whole word of list[] in
 * s at or after from: its offset, else -1
 */
static long
find_word(const char *s, const char *const *list, size_t from)
{
	size_t i, n = strlen(s);
	int depth = 0, k;

	for (i = 0; i < n; i++) {
		char c = s[i];
		if (c == '(' || c == '[' || c == '{')
			depth++;
		else if (c == ')' || c == ']' || c == '}')
			depth--;
		if (i < from || depth != 0 || !wch((unsigned char)c) ||
		    (i > 0 && wch((unsigned char)s[i - 1])))
			continue;
		for (k = 0; list[k]; k++) {
			size_t l = strlen(list[k]);
			if (!strncmp(s + i, list[k], l) &&
			    !wch((unsigned char)s[i + l]))
				return (long)i;
		}
	}
	return -1;
}

/* the text after option word w (up to the next option word), cut from s */
static int
take_opt(char *s, const char *w, const char *const *opts, char *val,
    size_t vn)
{
	const char *one[2] = {w, NULL};
	long at = find_word(s, one, 0), end;
	size_t l = strlen(w);

	if (at < 0)
		return 0;
	end = find_word(s, opts, (size_t)at + l);
	if (end < 0)
		end = (long)strlen(s);
	snprintf(val, vn, "%.*s", (int)(end - at - (long)l), s + at + l);
	trim(val);
	memmove(s + at, s + end, strlen(s + end) + 1);
	trim(s);
	return 1;
}

static int
has_ident(const char *s, const char *w)
{
	const char *one[2] = {w, NULL};
	size_t i, n = strlen(s), l = strlen(w);

	(void)one;
	for (i = 0; i + l <= n; i++)
		if (!strncmp(s + i, w, l) && (i == 0 || !wch((unsigned char)s[i - 1])) &&
		    !wch((unsigned char)s[i + l]))
			return 1;
	return 0;
}

static int
builtin_name(const char *w)
{
	static const char *const b[] = {"min", "max", "atan2", "fma",
		"copysign", "heaviside", "dirac", "kron", "fact", "binom", "u",
		"delta", "H", "step", NULL};
	int k;

	for (k = 0; b[k]; k++)
		if (!strcmp(b[k], w))
			return 1;
	return 0;
}

static const char *const unary_fns[] = {"sin", "cos", "tan", "exp", "log",
	"ln", "sqrt", "abs", "sinh", "cosh", "tanh", "sec", "csc", "cot",
	"atan", "asin", "acos", "floor", "ceil", "sign", "arctan", "arcsin",
	"arccos", NULL};

static int
in_list(const char *const *l, const char *w)
{
	int k;

	for (k = 0; l[k]; k++)
		if (!strcmp(l[k], w))
			return 1;
	return 0;
}

/*
 * Trailing variable names: "F t s".  in: the input variable (must occur
 * in F), out: the output variable (one of outs[]).  Removes them.
 */
static void
trail_vars(char *s, char *in, size_t inn, char *out, size_t outn,
    const char *const *outs)
{
	int pass;

	for (pass = 0; pass < 2; pass++) {
		size_t n = strlen(s), e = n, b;
		char w[32], rest_last[32];
		while (e > 0 && s[e - 1] == ' ')
			e--;
		b = e;
		while (b > 0 && (isalpha((unsigned char)s[b - 1]) ||
		    (unsigned char)s[b - 1] >= 0x80))
			b--;
		if (b == e || b == 0 || s[b - 1] != ' ' || e - b >= sizeof w)
			return;
		snprintf(w, sizeof w, "%.*s", (int)(e - b), s + b);
		{
			size_t r = b;
			while (r > 0 && s[r - 1] == ' ')
				r--;
			if (r == 0)
				return;
			/* the text before must end an expression */
			if (!(isalnum((unsigned char)s[r - 1]) || s[r - 1] == ')' ||
			    s[r - 1] == ']' || s[r - 1] == '!' ||
			    (unsigned char)s[r - 1] >= 0x80))
				return;
			{
				size_t q = r;
				while (q > 0 && isalpha((unsigned char)s[q - 1]))
					q--;
				snprintf(rest_last, sizeof rest_last, "%.*s",
				    (int)(r - q), s + q);
				if (in_list(unary_fns, rest_last))
					return;
			}
			if (func_find(sym_cstr(w)))
				return;
			{
				char save = s[b];
				int inrest;
				s[b] = 0;
				inrest = has_ident(s, w);
				s[b] = save;
				if (inrest && !in[0]) {
					snprintf(in, inn, "%s", w);
					s[r] = 0;
					continue;
				}
				if (!inrest && !out[0] && !in[0] && in_list(outs, w)) {
					snprintf(out, outn, "%s", w);
					s[r] = 0;
					continue;
				}
			}
		}
		return;
	}
}

/* split at top-level commas (and " and ", " with "): the pieces */
static int
split_list(char *s, char **part, int max)
{
	int n = 0, depth = 0;
	char *p = s, *start = s;

	for (; *p; p++) {
		if (*p == '(' || *p == '[' || *p == '{')
			depth++;
		else if (*p == ')' || *p == ']' || *p == '}')
			depth--;
		if (depth == 0 && (*p == ',' || *p == ';')) {
			*p = 0;
			if (n < max)
				part[n++] = start;
			start = p + 1;
		}
	}
	if (n < max)
		part[n++] = start;
	for (depth = 0; depth < n; depth++)
		trim(part[depth]);
	return n;
}

void
xf_head(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	putchar('\n');
}

/* ---- terms and factors ---- */

#define MAXT 96
struct terms {
	int n;
	nid t[MAXT];
};

static int
is_sum(nid e)
{
	while (N.op[e] == O_CAST && TY_FP(N.ty[N.a[e]]))
		e = N.a[e];
	return N.op[e] == O_ADD || N.op[e] == O_SUB ||
	    (N.op[e] == O_NEG && is_sum(N.a[e]));
}

static void terms_of(nid e, struct terms *T);

static void
add_terms(struct terms *T, nid e, int neg)
{
	while (N.op[e] == O_CAST && TY_FP(N.ty[N.a[e]]))
		e = N.a[e];
	switch (N.op[e]) {
	case O_ADD:
		add_terms(T, N.a[e], neg);
		add_terms(T, N.b[e], neg);
		return;
	case O_SUB:
		add_terms(T, N.a[e], neg);
		add_terms(T, N.b[e], !neg);
		return;
	case O_NEG:
		add_terms(T, N.a[e], !neg);
		return;
	case O_MUL:
	case O_DIV:
		/* distribute: (a + b) c, (a + b)/c */
		if (is_sum(N.a[e]) || (N.op[e] == O_MUL && is_sum(N.b[e]))) {
			struct terms *A = xmalloc(sizeof *A), *B = xmalloc(sizeof *B);
			int i, j;
			terms_of(N.a[e], A);
			if (N.op[e] == O_MUL)
				terms_of(N.b[e], B);
			else {
				B->n = 1;
				B->t[0] = N.b[e];
			}
			if (A->n * B->n + T->n <= MAXT) {
				for (i = 0; i < A->n; i++)
					for (j = 0; j < B->n; j++) {
						nid t = N.op[e] == O_MUL ? xmul(A->t[i], B->t[j]) :
						    xdiv(A->t[i], B->t[j]);
						T->t[T->n++] = neg ? xneg(t) : t;
					}
				free(A);
				free(B);
				return;
			}
			free(A);
			free(B);
		}
		break;
	}
	if (T->n < MAXT)
		T->t[T->n++] = neg ? xneg(e) : e;
	else
		fail("too many terms");
}

/* the terms of a sum, products of sums multiplied out */
static void
terms_of(nid e, struct terms *T)
{
	T->n = 0;
	add_terms(T, e, 0);
}

#define MAXFC 24
struct facs {
	nid coef;                 /* no var */
	int nf;
	nid f[MAXFC];             /* factors with the var */
	double pw[MAXFC];         /* their exponents */
};

static void
fac_rec(nid e, int slot, double pw, struct facs *F)
{
	double v;

	while (N.op[e] == O_CAST && TY_FP(N.ty[N.a[e]]))
		e = N.a[e];
	if (!has_slot(e, slot)) {
		F->coef = xmul(F->coef, pw == 1 ? e : xpow(e, xk(pw)));
		return;
	}
	switch (N.op[e]) {
	case O_NEG:
		F->coef = xneg(F->coef);
		fac_rec(N.a[e], slot, pw, F);
		return;
	case O_MUL:
		fac_rec(N.a[e], slot, pw, F);
		fac_rec(N.b[e], slot, pw, F);
		return;
	case O_DIV:
		fac_rec(N.a[e], slot, pw, F);
		fac_rec(N.b[e], slot, -pw, F);
		return;
	case O_POW:
		/* f^k: the factor f with exponent k (a numeric k) */
		if (xnum(N.b[e], &v) && v != 0) {
			fac_rec(N.a[e], slot, pw * v, F);
			return;
		}
		break;
	case O_SQRT:
		fac_rec(N.a[e], slot, pw * 0.5, F);
		return;
	}
	if (F->nf == MAXFC)
		fail("too many factors");
	F->f[F->nf] = e;
	F->pw[F->nf++] = pw;
}

static nid
unc(nid e)
{
	while (N.op[e] == O_CAST)
		e = N.a[e];
	return e;
}

static int
is_var(nid e, int slot)
{
	e = unc(e);
	return N.op[e] == O_VAR && N.ki[e] == slot;
}

/* u = c1 x + c0 (x in slot), c1 != 0: real nodes */
static int
lin_of(nid u, int slot, nid *c1, nid *c0)
{
	struct npoly p;

	if (!has_slot(u, slot) || !npoly_of(u, slot, &p, 1) || p.deg != 1)
		return 0;
	*c0 = xadd(xk(0), p.c[0]);
	*c1 = xadd(xk(0), p.c[1]);
	return 1;
}

/* the same, numeric */
static int
lin_num(nid u, int slot, double *c1, double *c0)
{
	nid a, b;

	return lin_of(u, slot, &a, &b) && xnum(a, c1) && xnum(b, c0) &&
	    *c1 != 0;
}

/* coef times the factors except those in skip (a bit mask) */
static nid
facs_except(const struct facs *F, uint32_t skip)
{
	nid r = F->coef;
	int k;

	for (k = 0; k < F->nf; k++)
		if (!(skip & (1u << k)))
			r = xmul(r, F->pw[k] == 1 ? F->f[k] : xpow(F->f[k],
			    xk(F->pw[k])));
	return r;
}

/* the indicator of a condition on slot, as steps u(x - a) (x >= 0) */
static nid
indicator(nid c, int slot, nid x)
{
	int op;
	double c1, c0, a;
	nid H;

	c = unc(c);
	op = N.op[c];
	if (op == O_LAND) {
		nid A = indicator(N.a[c], slot, x), B = A ? indicator(N.b[c],
		    slot, x) : 0;
		return A && B ? xmul(A, B) : 0;
	}
	if (op == O_LOR) {
		nid A = indicator(N.a[c], slot, x), B = A ? indicator(N.b[c],
		    slot, x) : 0;
		return A && B ? xsub(xadd(A, B), xmul(A, B)) : 0;
	}
	if (op == O_LNOT) {
		nid A = indicator(N.a[c], slot, x);
		return A ? xsub(xk(1), A) : 0;
	}
	if (!(opinfo[op].fl & OF_CMP))
		return 0;
	if (!lin_num(xsub(N.a[c], N.b[c]), slot, &c1, &c0))
		return 0;
	a = -c0 / c1;
	H = xheav(xsub(x, xk(a)));   /* x >= a */
	if (op == O_EQ)
		return xk(0);   /* one point */
	if (op == O_NE)
		return xk(1);
	if ((op == O_GE || op == O_GT) == (c1 > 0))
		return H;
	return xsub(xk(1), H);
}

/* (x + i b)^m: real and imaginary parts */
static void
cbinom(nid x, nid b, int m, nid *re, nid *im)
{
	double C = 1;
	int j;

	*re = xk(0);
	*im = xk(0);
	for (j = 0; j <= m; j++) {
		nid t = xmul(xk(C), xmul(xpow(x, xk(m - j)), xpow(b, xk(j))));
		int sg = (j / 2) & 1 ? -1 : 1;
		if (j & 1)
			*im = sg > 0 ? xadd(*im, t) : xsub(*im, t);
		else
			*re = sg > 0 ? xadd(*re, t) : xsub(*re, t);
		C = C * (m - j) / (j + 1);
	}
}

static double
factd(int n)
{
	double r = 1;

	while (n > 1)
		r *= n--;
	return r;
}

/* ---- Laplace ---- */

enum {
	RL_LIN = 1, RL_EXP = 2, RL_TPOW = 4, RL_TRIG = 8, RL_HYP = 16,
	RL_STEP = 32, RL_DELTA = 64, RL_GAMMA = 128, RL_PIECE = 256,
	RL_P2S = 512, RL_POWRED = 1024, RL_POLY = 2048, RL_PERIOD = 4096,
	RL_ABS = 8192
};
static const char *const rl_names[] = {"linearity",
	"e^(at) f(t) -> F(s - a)", "t^n f(t) -> (-1)^n F^(n)(s)",
	"sin/cos table", "sinh/cosh table", "u(t - a) f(t - a) -> e^(-as) F(s)",
	"delta(t - a) -> e^(-as)", "t^p -> Gamma(p + 1)/s^(p + 1)",
	"piecewise as steps", "product to sum", "power reduction",
	"polynomial expanded", "periodic: F(s) = F_T(s)/(1 - e^(-Ts))",
	"|t - a| as steps", NULL};

struct lctx {
	int ts, ss;
	nid t, s;
	int n;
	nid R[MAXT];
	double sh[MAXT];
	int dl[MAXT];
	int nt[MAXT];       /* table terms merged into R[k] */
	nid roc;            /* max abscissa; 0: entire */
	unsigned rules;
	int depth;
};

static void
lc_add(struct lctx *C, nid R, double sh, int dl)
{
	int k;

	for (k = 0; k < C->n; k++)
		if (fabs(C->sh[k] - sh) <= 1e-12 * fmax(1, fabs(sh)) &&
		    C->dl[k] == dl) {
			C->R[k] = xadd(C->R[k], R);
			C->nt[k]++;
			return;
		}
	if (C->n == MAXT)
		fail("too many terms");
	C->R[C->n] = R;
	C->sh[C->n] = sh;
	C->dl[C->n] = dl;
	C->nt[C->n++] = 1;
}

static void
lc_roc(struct lctx *C, nid a)
{
	C->roc = C->roc ? mk_bin(O_MAX, C->roc, xadd(xk(0), a)) :
	    xadd(xk(0), a);
}

static void lap_rec(struct lctx *C, nid e, double sh);

/* L{t^n e^(at) trig(bt + phi)} */
static nid
lap_core(struct lctx *C, double pw, nid alpha, int trig, nid beta, nid phi)
{
	nid x = xsub(C->s, alpha), R;
	int n;
	double ph;

	if (trig == 0) {
		lc_roc(C, alpha);
		if (pw == floor(pw) && pw >= 0) {
			n = (int)pw;
			if (n > 0)
				C->rules |= RL_TPOW;
			return xdiv(xk(factd(n)), xpow(x, xk(n + 1)));
		}
		if (pw > -1) {
			C->rules |= RL_GAMMA;
			return xdiv(xk(tgamma(pw + 1)), xpow(x, xk(pw + 1)));
		}
		fail("t^%s: the integral diverges at t = 0 (p <= -1)",
		    xf_num(pw));
	}
	if (pw != floor(pw) || pw < 0)
		fail("t^%s times a sine or cosine: no rule", xf_num(pw));
	n = (int)pw;
	if (n > 0)
		C->rules |= RL_TPOW;
	if (trig == O_SIN || trig == O_COS) {
		nid re, im, den, Ls, Lc;
		C->rules |= RL_TRIG;
		lc_roc(C, alpha);
		cbinom(x, beta, n + 1, &re, &im);
		den = xpow(xadd(xpow(x, xk(2)), xpow(beta, xk(2))), xk(n + 1));
		Ls = xdiv(xmul(xk(factd(n)), im), den);
		Lc = xdiv(xmul(xk(factd(n)), re), den);
		if (phi == 0 || (xnum(phi, &ph) && ph == 0))
			return trig == O_SIN ? Ls : Lc;
		{
			nid cp = mk_un(O_COS, phi), sp = mk_un(O_SIN, phi);
			if (trig == O_SIN)
				return xadd(xmul(cp, Ls), xmul(sp, Lc));
			return xsub(xmul(cp, Lc), xmul(sp, Ls));
		}
	}
	/* sinh, cosh */
	C->rules |= RL_HYP;
	lc_roc(C, xadd(alpha, mk_un(O_ABS, beta)));
	{
		nid Lsh, Lch;
		if (n == 0) {
			nid den = xsub(xpow(x, xk(2)), xpow(beta, xk(2)));
			Lsh = xdiv(beta, den);
			Lch = xdiv(x, den);
		} else {
			nid A = xdiv(xk(factd(n)), xpow(xsub(x, beta), xk(n + 1)));
			nid B = xdiv(xk(factd(n)), xpow(xadd(x, beta), xk(n + 1)));
			Lsh = xdiv(xsub(A, B), xk(2));
			Lch = xdiv(xadd(A, B), xk(2));
		}
		if (phi == 0 || (xnum(phi, &ph) && ph == 0))
			R = trig == O_SINH ? Lsh : Lch;
		else {
			nid ch = mk_un(O_COSH, phi), sh = mk_un(O_SINH, phi);
			R = trig == O_SINH ? xadd(xmul(ch, Lsh), xmul(sh, Lch)) :
			    xadd(xmul(ch, Lch), xmul(sh, Lsh));
		}
	}
	return R;
}

static int
trig_op(int op)
{
	return op == O_SIN || op == O_COS || op == O_SINH || op == O_COSH;
}

/* f(u) g(v) for f, g in sin, cos as a sum (product to sum) */
static nid
prod2sum(int f, nid u, int g, nid v)
{
	nid p = xadd(u, v), m = xsub(u, v);

	if (f == O_SIN && g == O_SIN)
		return xdiv(xsub(mk_un(O_COS, m), mk_un(O_COS, p)), xk(2));
	if (f == O_COS && g == O_COS)
		return xdiv(xadd(mk_un(O_COS, m), mk_un(O_COS, p)), xk(2));
	if (f == O_SIN && g == O_COS)
		return xdiv(xadd(mk_un(O_SIN, p), mk_un(O_SIN, m)), xk(2));
	if (f == O_COS && g == O_SIN)
		return xdiv(xsub(mk_un(O_SIN, p), mk_un(O_SIN, m)), xk(2));
	/* hyperbolic: through exponentials */
	{
		nid a = f == O_SINH ? xdiv(xsub(xexp(u), xexp(xneg(u))), xk(2)) :
		    f == O_COSH ? xdiv(xadd(xexp(u), xexp(xneg(u))), xk(2)) :
		    mk_un(f, u);
		nid b = g == O_SINH ? xdiv(xsub(xexp(v), xexp(xneg(v))), xk(2)) :
		    g == O_COSH ? xdiv(xadd(xexp(v), xexp(xneg(v))), xk(2)) :
		    mk_un(g, v);
		nid r;
		/* expand (a1 + a2)(b1 + b2) */
		struct terms A, B;
		int i, j;
		terms_of(a, &A);
		terms_of(b, &B);
		r = xk(0);
		for (i = 0; i < A.n; i++)
			for (j = 0; j < B.n; j++)
				r = xadd(r, xmul(A.t[i], B.t[j]));
		return r;
	}
}

static void
lap_term(struct lctx *C, nid term, double sh)
{
	struct facs F;
	double pw = 0, step = 0, da = 0, dw = 0, v, hv1, hv0;
	nid alpha = xk(0), beta = 0, phi = 0, c1, c0, arg, u;
	int k, trig = 0, trigk = -1, dirac = 0;
	uint32_t steps = 0;

	memset(&F, 0, sizeof F);
	F.coef = xk(1);
	fac_rec(term, C->ts, 1, &F);
	for (k = 0; k < F.nf; k++) {
		nid f = unc(F.f[k]);
		double p = F.pw[k];
		int op = N.op[f];
		if (is_var(f, C->ts)) {
			pw += p;
			continue;
		}
		if (xis_exp(f, &u) && lin_of(u, C->ts, &c1, &c0)) {
			alpha = xadd(alpha, xmul(xk(p), c1));
			F.coef = xmul(F.coef, xexp(xmul(xk(p), c0)));
			C->rules |= RL_EXP;
			continue;
		}
		if (trig_op(op) && p == 1 && lin_of(N.a[f], C->ts, &c1, &c0)) {
			if (trig) {
				nid g = facs_except(&F, (1u << k) | (1u << trigk));
				C->rules |= RL_P2S;
				lap_rec(C, xmul(g, prod2sum(N.op[unc(F.f[trigk])],
				    N.a[unc(F.f[trigk])], op, N.a[f])), sh);
				return;
			}
			trig = op;
			trigk = k;
			beta = c1;
			phi = c0;
			continue;
		}
		if (trig_op(op) && p == floor(p) && p >= 2 && p <= 6) {
			nid b = f, w = N.a[b], red, rest;
			int bop = N.op[b];
			nid c2 = mk_un(bop == O_SIN || bop == O_COS ? O_COS : O_COSH,
			    xmul(xk(2), w));
			if (bop == O_SIN)
				red = xdiv(xsub(xk(1), c2), xk(2));
			else if (bop == O_COS)
				red = xdiv(xadd(xk(1), c2), xk(2));
			else if (bop == O_SINH)
				red = xdiv(xsub(c2, xk(1)), xk(2));
			else
				red = xdiv(xadd(c2, xk(1)), xk(2));
			rest = facs_except(&F, 1u << k);
			if (p > 2)
				rest = xmul(rest, xpow(b, xk(p - 2)));
			C->rules |= RL_POWRED;
			{
				struct terms T;
				int i;
				terms_of(red, &T);
				for (i = 0; i < T.n; i++)
					lap_rec(C, xmul(rest, T.t[i]), sh);
			}
			return;
		}
		if (xis_heav(f, &arg) && p == 1 && lin_num(arg, C->ts, &hv1,
		    &hv0)) {
			double a1 = hv1, a0 = hv0;
			if (a1 > 0) {
				double a = -a0 / a1;
				if (a > step)
					step = a;
				steps |= 1u << k;
				continue;
			}
			{
				/* u(a - t) = 1 - u(t - a) */
				nid g = facs_except(&F, 1u << k);
				double a = -a0 / a1;
				C->rules |= RL_STEP;
				lap_rec(C, xsub(g, xmul(g, xheav(xsub(C->t, xk(a))))),
				    sh);
				return;
			}
		}
		if (xis_dirac(f, &arg) && p == 1 && !dirac) {
			double a1, a0;
			if (!lin_num(arg, C->ts, &a1, &a0))
				fail("delta(%s): the argument must be linear in %s",
				    xf_str(arg), sym_str(N.a[C->t]));
			da = -a0 / a1;
			dw = 1 / fabs(a1);
			dirac = 1;
			steps |= 1u << k;   /* left out of the rest */
			continue;
		}
		if (op == O_SEL && p == 1) {
			nid H = indicator(N.a[f], C->ts, C->t);
			if (H) {
				nid g = facs_except(&F, 1u << k);
				nid A = N.b[f], B = N.c[f];
				C->rules |= RL_PIECE;
				lap_rec(C, xmul(g, xadd(B, xmul(xsub(A, B), H))), sh);
				return;
			}
		}
		if (op == O_ABS && p == 1 && lin_num(N.a[f], C->ts, &hv1, &hv0)) {
			double a1 = hv1, a0 = hv0, a;
			nid g = facs_except(&F, 1u << k), lin = N.a[f];
			a = -a0 / a1;
			C->rules |= RL_ABS;
			if (a <= 0) {
				lap_rec(C, xmul(g, a1 > 0 ? lin : xneg(lin)), sh);
				return;
			}
			/* |lin| = -s lin + 2 s lin u(t - a), s = sign(a1) */
			{
				nid sl = a1 > 0 ? lin : xneg(lin);
				lap_rec(C, xadd(xmul(g, xneg(sl)), xmul(g, xmul(xk(2),
				    xmul(sl, xheav(xsub(C->t, xk(a))))))), sh);
			}
			return;
		}
		if (p > 0 && p == floor(p) && p <= 12) {
			struct npoly np;
			nid fp = p == 1 ? f : xpow(f, xk(p));
			if (npoly_of(fp, C->ts, &np, 12) && np.deg >= 1) {
				nid g = facs_except(&F, 1u << k);
				int j;
				C->rules |= RL_POLY;
				for (j = 0; j <= np.deg; j++)
					if (!(is_const(np.c[j]) && const_is(np.c[j], 0)))
						lap_rec(C, xmul(xmul(g, xadd(xk(0), np.c[j])),
						    xpow(C->t, xk(j))), sh);
				return;
			}
		}
		fail("laplace: no rule for the factor %s", xf_str(F.pw[k] == 1 ?
		    F.f[k] : xpow(F.f[k], xk(F.pw[k]))));
	}
	if (dirac) {
		nid g, val;
		/* the rest of the term at t = a: steps included */
		{
			int j;
			g = F.coef;
			for (j = 0; j < F.nf; j++) {
				nid ff = unc(F.f[j]), a2;
				if (xis_dirac(ff, &a2))
					continue;
				g = xmul(g, F.pw[j] == 1 ? F.f[j] : xpow(F.f[j],
				    xk(F.pw[j])));
			}
		}
		C->rules |= RL_DELTA;
		if (da < 0)
			return;   /* outside [0, inf) */
		val = xmul(xk(dw), xsubst1(g, C->ts, xk(da)));
		lc_add(C, val, sh + da, 1);
		return;
	}
	if (step > 0) {
		nid g = facs_except(&F, steps);
		C->rules |= RL_STEP;
		lap_rec(C, xsubst1(g, C->ts, xadd(C->t, xk(step))), sh + step);
		return;
	}
	lc_add(C, xmul(F.coef, lap_core(C, pw, alpha, trig, beta, phi)), sh, 0);
}

static void
lap_rec(struct lctx *C, nid e, double sh)
{
	struct terms T;
	int i;

	if (++C->depth > 40)
		fail("laplace: the expression is too deep");
	terms_of(e, &T);
	if (T.n > 1)
		C->rules |= RL_LIN;
	for (i = 0; i < T.n; i++)
		lap_term(C, T.t[i], sh);
	C->depth--;
}

/* ---- common: polishing, parameters, variables ---- */

/* a rational function of slot: canonical (factored); else as is */
nid
xf_polish(nid R, int slot, nid x)
{
	struct xrat r;

	if (node_slots(R) & ~(1ULL << slot))
		return R;
	if (!xq_of(R, slot, &r))
		return R;
	return xq_node(&r, x);
}

/* sample values for the other parameters of f (symbols), in slots[] */
static int
samples(const struct func *f, const int *skip, int nskip, int *slots,
    double *vals, struct sbuf *txt)
{
	static const double sv[] = {0.7, 1.3, 0.45, 1.9, 0.85, 2.3, 0.3, 1.1};
	int k, j, n = 0;

	for (k = 0; k < f->np; k++) {
		for (j = 0; j < nskip; j++)
			if (skip[j] == k)
				break;
		if (j < nskip)
			continue;
		slots[n] = k;
		vals[n] = sv[n & 7];
		if (txt)
			sb_printf(txt, "%s%s = %s", n ? ", " : "",
			    sym_str(f->p[k].name), xf_num(vals[n]));
		n++;
	}
	return n;
}

/* e with the sample values substituted */
static nid
with_samples(nid e, const int *slots, const double *vals, int n)
{
	nid map[NSLOT];
	int k;

	memset(map, 0, sizeof map);
	for (k = 0; k < n; k++)
		map[slots[k]] = xk(vals[k]);
	return subst(e, map, NSLOT);
}

static const char *
fname_of(const struct func *f)
{
	return f->anon ? xf_str(f->body) : sym_str(f->name);
}

/* pick the variable: the name given, else pref[], else the only one */
static int
pick_slot(const struct func *f, const char *given, const char *const *pref)
{
	int k;

	if (given && given[0]) {
		k = xf_slot(f, given);
		if (k < 0)
			fail("%s does not depend on %s", fname_of(f), given);
		return k;
	}
	for (k = 0; pref[k]; k++)
		if (xf_slot(f, pref[k]) >= 0)
			return xf_slot(f, pref[k]);
	if (f->np == 1)
		return 0;
	return -1;
}

static int
max_slot(const struct func *f)
{
	return f->np;
}

/* ---- Laplace: the command ---- */

static nid
lap_total(struct lctx *C, int nodelta, int *ndelta)
{
	nid total = xk(0);
	int k;

	if (ndelta)
		*ndelta = 0;
	for (k = 0; k < C->n; k++) {
		nid R;
		if (C->dl[k] && ndelta)
			(*ndelta)++;
		if (C->dl[k] && nodelta)
			continue;
		R = xf_polish(C->R[k], C->ss, C->s);
		if (C->sh[k] != 0)
			R = xmul(R, xexp(xmul(xk(-C->sh[k]), C->s)));
		total = xadd(total, R);
	}
	return total;
}

static void
rules_line(unsigned rules)
{
	int k, n = 0;

	printf("  rules:");
	for (k = 0; rl_names[k]; k++)
		if (rules & (1u << k))
			printf("%s %s", n++ ? "," : "", rl_names[k]);
	if (n == 0)
		printf(" table");
	printf("\n");
}

static double
reldiff(double a, double b)
{
	double sc = fmax(fmax(fabs(a), fabs(b)), 1e-12);

	return fabs(a - b) / sc;
}

/*
 * The Laplace integral of e (in slot ts) at s0, numerically; periodic
 * with period P when P > 0
 */
static int
lap_numeric(nid e, int ts, nid t, double s0, double P, double decay,
    double *v)
{
	double err, a, b, X, sum = 0, piece;
	int nev;
	nid g = e;

	if (P > 0)
		g = xsubst1(e, ts, xsub(t, xmul(xk(P), mk_un(O_FLOOR,
		    xdiv(t, xk(P))))));
	g = xmul(g, xexp(xmul(xk(-s0), t)));
	if (P > 0) {
		/* one period: sum_k e^(-k P s0) int_0^P */
		if (!gk_integrate(NULL, g, ts, 0, P, &piece, &err, &nev))
			return 0;
		*v = piece / (1 - exp(-P * s0));
		return 1;
	}
	/* [0, X] in doubling pieces: the integrand is e^(-decay t) small */
	X = 80 / fmax(decay, 0.05) + 40;
	for (a = 0, b = 0.5; a < X; a = b, b *= 2) {
		if (!gk_integrate(NULL, g, ts, a, b, &piece, &err, &nev))
			return 0;
		sum += piece;
	}
	*v = sum;
	return 1;
}

static void
lap_check(const struct func *f, nid body, struct lctx *C, nid F, double P,
    int ndelta)
{
	int skip[2] = {C->ts, C->ss}, sl[MAXP], ns, i, ok, good = 1;
	double vals[MAXP], roc = 0, worst = 0, s0[3];
	struct sbuf txt = {0};
	nid fb, Fb;

	ns = samples(f, skip, 2, sl, vals, &txt);
	fb = with_samples(xf_nodirac(body), sl, vals, ns);
	Fb = with_samples(F, sl, vals, ns);
	if (C->roc) {
		nid r = with_samples(C->roc, sl, vals, ns);
		if (!xnum(r, &roc))
			roc = xf_eval(r, NULL, NULL, 0, &ok);
	}
	if (P > 0 && roc < 0)
		roc = 0;
	if (!C->roc && P <= 0)
		roc = 0;
	s0[0] = roc + 0.5;
	s0[1] = roc + 1.5;
	s0[2] = roc + 4;
	for (i = 0; i < 3; i++) {
		double Iv, Fv;
		if (!lap_numeric(fb, C->ts, C->t, s0[i], P, s0[i] - roc, &Iv)) {
			good = 0;
			break;
		}
		Fv = xf_eval1(Fb, C->ss, s0[i], &ok);
		if (!ok) {
			good = 0;
			break;
		}
		worst = fmax(worst, reldiff(Iv, Fv));
	}
	if (!good)
		printf("  check: the integral could not be evaluated numerically\n");
	else
		printf("  check: %s int_0^inf f(t) e^(-st) dt (Gauss-Kronrod) at "
		    "s = %s, %s, %s%s%s%s: max relative difference %.2g\n",
		    worst < 1e-7 ? "matches" : "DIFFERS FROM", xf_num(s0[0]),
		    xf_num(s0[1]), xf_num(s0[2]), ns ? " with " : "",
		    ns ? txt.p : "", ndelta ? " (delta terms left out)" : "",
		    worst);
	free(txt.p);
}

static void
cmd_laplace(char *text)
{
	static const char *const opts[] = {"period", NULL};
	static const char *const outs[] = {"s", "p", "w", NULL};
	static const char *const tpref[] = {"t", "x", NULL};
	char tv[32] = "", sv[32] = "", per[96] = "";
	struct func *f;
	struct lctx C;
	nid body, F, Fnd;
	double P = 0;
	int ndelta;

	take_opt(text, "period", opts, per, sizeof per);
	trail_vars(text, tv, sizeof tv, sv, sizeof sv, outs);
	f = xf_parse(text);
	memset(&C, 0, sizeof C);
	C.ts = pick_slot(f, tv, tpref);
	if (C.ts < 0 && f->np > 0)
		fail("laplace: which variable is time?  (laplace F t s)");
	if (C.ts < 0)
		C.ts = max_slot(f);
	C.t = xv(C.ts, C.ts < f->np ? sym_str(f->p[C.ts].name) : "t");
	C.ss = C.ts == f->np ? f->np + 1 : f->np;
	if (C.ss >= MAXP)
		fail("too many variables");
	if (!sv[0])
		snprintf(sv, sizeof sv, "%s", xf_slot(f, "s") >= 0 ? "p" : "s");
	C.s = xv(C.ss, sv);
	body = f->body;
	if (per[0]) {
		struct func *pf = xf_parse(per);
		if (!xnum(pf->body, &P) || P <= 0)
			fail("laplace: the period must be a positive number");
	}
	lap_rec(&C, body, 0);
	if (P > 0) {
		lap_rec(&C, xneg(xsubst1(body, C.ts, xadd(C.t, xk(P)))), P);
		C.rules |= RL_PERIOD;
	}
	F = lap_total(&C, 0, &ndelta);
	Fnd = lap_total(&C, 1, NULL);
	if (P > 0) {
		nid den = xsub(xk(1), xexp(xmul(xk(-P), C.s)));
		F = xdiv(F, den);
		Fnd = xdiv(Fnd, den);
	}
	printf("L{%s}(%s) = %s\n", fname_of(f), sym_str(N.a[C.s]), xf_str(F));
	if (P > 0)
		printf("  f is the periodic extension of its values on [0, %s)\n",
		    xf_num(P));
	if (P > 0)
		printf("  region of convergence: Re(%s) > %s\n", sv,
		    C.roc && xnum(C.roc, &(double){0}) ? "0" : "0");
	else if (!C.roc)
		printf("  region of convergence: every %s (f is zero after a "
		    "finite time)\n", sv);
	else {
		double r;
		if (xnum(C.roc, &r))
			printf("  region of convergence: Re(%s) > %s\n", sv, xf_num(r));
		else
			printf("  region of convergence: Re(%s) > %s\n", sv,
			    xf_str(C.roc));
	}
	printf("  one-sided: f(%s) for %s >= 0 (0 before)\n",
	    sym_str(N.a[C.t]), sym_str(N.a[C.t]));
	rules_line(C.rules);
	lap_check(f, body, &C, Fnd, P, ndelta);
}

/* ---- inverse Laplace ---- */

/* the time function of a partial fraction expansion (no polynomial) */
static nid
pf_time(const struct xpf *pf, nid t, int discrete)
{
	nid y = xk(0);
	int i, j, k, used[XD] = {0};

	(void)discrete;
	for (i = 0; i < pf->np; i++) {
		const struct xpole *P = &pf->pole[i];
		double re = creal(P->p), im = cimag(P->p);
		if (used[i])
			continue;
		used[i] = 1;
		if (im == 0) {
			struct xpoly pl;
			nid term;
			/* a pair +-r with one coefficient sum zero: cosh/sinh */
			if (P->m == 1 && re != 0)
				for (j = i + 1; j < pf->np; j++) {
					const struct xpole *Q = &pf->pole[j];
					double A, B, r;
					if (used[j] || Q->m != 1 || cimag(Q->p) != 0 ||
					    fabs(creal(Q->p) + re) > 1e-12 * fabs(re))
						continue;
					A = creal(re > 0 ? P->a[1] : Q->a[1]);
					B = creal(re > 0 ? Q->a[1] : P->a[1]);
					r = fabs(re);
					if (fabs(A + B) > 1e-12 * fabs(A) &&
					    fabs(A - B) > 1e-12 * fabs(A))
						break;
					used[j] = 1;
					y = xadd(y, xadd(xmul(xk(A + B), mk_un(O_COSH,
					    xmul(xk(r), t))), xmul(xk(A - B), mk_un(O_SINH,
					    xmul(xk(r), t)))));
					goto next;
				}
			xp_zero(&pl);
			for (k = 1; k <= P->m; k++)
				pl.c[k - 1] = creal(P->a[k]) / factd(k - 1);
			pl.deg = P->m - 1;
			xp_trim(&pl);
			term = xp_node(&pl, t);
			if (re != 0)
				term = xmul(term, xexp(xmul(xk(re), t)));
			y = xadd(y, term);
		} else {
			struct xpoly pc, ps;
			nid term;
			xp_zero(&pc);
			xp_zero(&ps);
			for (k = 1; k <= P->m; k++) {
				pc.c[k - 1] = 2 * creal(P->a[k]) / factd(k - 1);
				ps.c[k - 1] = -2 * cimag(P->a[k]) / factd(k - 1);
			}
			pc.deg = ps.deg = P->m - 1;
			xp_trim(&pc);
			xp_trim(&ps);
			term = xadd(xmul(xp_node(&pc, t), mk_un(O_COS, xmul(xk(im), t))),
			    xmul(xp_node(&ps, t), mk_un(O_SIN, xmul(xk(im), t))));
			if (re != 0)
				term = xmul(xexp(xmul(xk(re), t)), term);
			y = xadd(y, term);
		}
next:		;
	}
	return y;
}

static void
poles_text(struct sbuf *b, const struct xpf *pf, const char *v)
{
	int i;

	for (i = 0; i < pf->np; i++) {
		double complex p = pf->pole[i].p;
		const char *ord = pf->pole[i].m == 1 ? "" : pf->pole[i].m == 2 ?
		    " (double)" : pf->pole[i].m == 3 ? " (triple)" : " (repeated)";
		if (cimag(p) == 0)
			sb_printf(b, "%s%s%s", i ? ", " : "", xf_num(creal(p)), ord);
		else if (creal(p) == 0)
			sb_printf(b, "%s+-%s%s", i ? ", " : "",
			    xf_cnum(I * cimag(p)), ord);
		else
			sb_printf(b, "%s%s +- %s%s", i ? ", " : "", xf_num(creal(p)),
			    xf_cnum(I * cimag(p)), ord);
	}
	(void)v;
}

static double
pf_abscissa(const struct xpf *pf)
{
	double m = -INFINITY;
	int i;

	for (i = 0; i < pf->np; i++)
		m = fmax(m, creal(pf->pole[i].p));
	return m;
}

/* symbolic: c/(s - a)^m and (n1 s + n0)/(A s^2 + B s + C) */
static nid
ilap_symbolic(nid R, int ss, nid t, struct sbuf *assume)
{
	struct facs F;
	struct npoly D, Nm;
	nid den = 0, num = xk(1);
	double m = 0;
	int k;

	memset(&F, 0, sizeof F);
	F.coef = xk(1);
	fac_rec(R, ss, 1, &F);
	for (k = 0; k < F.nf; k++) {
		if (F.pw[k] < 0) {
			if (den)
				return 0;
			den = F.f[k];
			m = -F.pw[k];
		} else if (F.pw[k] == 1)
			num = xmul(num, F.f[k]);
		else
			return 0;
	}
	if (!den)
		return 0;
	if (N.op[unc(den)] == O_POW && xnum(N.b[unc(den)], &(double){0})) {
		double e;
		xnum(N.b[unc(den)], &e);
		m *= e;
		den = N.a[unc(den)];
	}
	if (m != floor(m) || !npoly_of(den, ss, &D, 2) ||
	    !npoly_of(num, ss, &Nm, 1))
		return 0;
	if (D.deg == 1 && Nm.deg == 0) {
		nid c1 = xadd(xk(0), D.c[1]), a = xneg(xdiv(xadd(xk(0), D.c[0]),
		    c1));
		nid r = xdiv(xmul(F.coef, xadd(xk(0), Nm.c[0])), xpow(c1, xk(m)));
		r = xmul(r, xdiv(xpow(t, xk(m - 1)), xk(factd((int)m - 1))));
		return xmul(r, xexp(xmul(a, t)));
	}
	if (D.deg == 2 && m == 1) {
		nid A = xadd(xk(0), D.c[2]), B = xadd(xk(0), D.c[1]),
		    Cc = xadd(xk(0), D.c[0]);
		nid al = xneg(xdiv(B, xmul(xk(2), A)));
		nid w2 = xsub(xdiv(Cc, A), xpow(al, xk(2))), w;
		nid n1 = Nm.deg >= 1 ? xadd(xk(0), Nm.c[1]) : xk(0);
		nid n0 = xadd(xk(0), Nm.c[0]);
		nid r;
		if (N.op[unc(w2)] == O_POW && xnum(N.b[unc(w2)], &(double){0}) &&
		    const_is(unc(N.b[unc(w2)]), 2)) {
			w = N.a[unc(w2)];
			sb_printf(assume, "%s > 0", xf_str(w));
		} else {
			w = mk_un(O_SQRT, w2);
			sb_printf(assume, "%s > 0", xf_str(w2));
		}
		r = xadd(xmul(n1, mk_un(O_COS, xmul(w, t))),
		    xmul(xdiv(xadd(n0, xmul(n1, al)), w), mk_un(O_SIN,
		    xmul(w, t))));
		r = xmul(xdiv(F.coef, A), r);
		if (!(is_const(al) && const_is(al, 0)))
			r = xmul(xexp(xmul(al, t)), r);
		return r;
	}
	return 0;
}

/* (c1 s + c0)^(-p), p not an integer: t^(p-1) e^(at)/Gamma(p) */
static nid
ilap_power(nid R, int ss, nid t)
{
	struct facs F;
	double e = 0, c1, c0, v;
	nid lin = 0;
	int k;

	memset(&F, 0, sizeof F);
	F.coef = xk(1);
	fac_rec(R, ss, 1, &F);
	if (F.nf != 1 || node_slots(F.coef))
		return 0;
	{
		nid f = unc(F.f[0]);
		e = F.pw[0];
		if (N.op[f] == O_SQRT)
			lin = N.a[f], e *= 0.5;
		else if (N.op[f] == O_POW && xnum(N.b[f], &v))
			lin = N.a[f], e *= v;
		else
			lin = f;
	}
	if (e >= 0 || e == floor(e) || !lin_num(lin, ss, &c1, &c0) || c1 <= 0)
		return 0;
	k = 0;
	(void)k;
	{
		double p = -e, a = -c0 / c1;
		nid r = xmul(xdiv(F.coef, xk(pow(c1, p) * tgamma(p))),
		    xpow(t, xk(p - 1)));
		if (a != 0)
			r = xmul(r, xexp(xmul(xk(a), t)));
		return r;
	}
}

struct igroup {
	double sh;
	nid R;
};

/* the inverse transform of F (slot ss) as a node in t; *sigma the
 * abscissa (right of every pole); deltas as dirac nodes */
static nid
ilap_node(nid F, int ss, nid t, int ts, double *sigma, struct sbuf *poles,
    struct sbuf *assume, int *nd)
{
	struct terms T;
	struct igroup G[MAXT];
	int ng = 0, i, k;
	nid y = xk(0);

	(void)ts;
	*sigma = -INFINITY;
	*nd = 0;
	terms_of(F, &T);
	for (i = 0; i < T.n; i++) {
		struct facs Fc;
		double sh = 0, c1, c0;
		nid rest;
		memset(&Fc, 0, sizeof Fc);
		Fc.coef = xk(1);
		fac_rec(T.t[i], ss, 1, &Fc);
		{
			uint32_t skip = 0;
			for (k = 0; k < Fc.nf; k++) {
				nid u;
				if (xis_exp(Fc.f[k], &u) && lin_num(u, ss, &c1, &c0)) {
					c1 *= Fc.pw[k];
					c0 *= Fc.pw[k];
					if (c1 > 0)
						fail("e^(%ss) with a > 0 is an advance: no causal "
						    "inverse", xf_num(c1));
					sh += -c1;
					Fc.coef = xmul(Fc.coef, xk(exp(c0)));
					skip |= 1u << k;
				}
			}
			rest = facs_except(&Fc, skip);
		}
		for (k = 0; k < ng; k++)
			if (fabs(G[k].sh - sh) <= 1e-12 * fmax(1, sh))
				break;
		if (k == ng) {
			if (ng == MAXT)
				fail("too many terms");
			G[ng].sh = sh;
			G[ng++].R = rest;
		} else
			G[k].R = xadd(G[k].R, rest);
	}
	for (k = 0; k < ng; k++) {
		struct xrat r;
		struct xpf pf;
		nid g = 0;
		if (!(node_slots(G[k].R) & ~(1ULL << ss)) && xq_of(G[k].R, ss, &r)) {
			if (!xf_apart(&r, &pf))
				fail("ilaplace: the denominator's roots were not found "
				    "(degree > 14?)");
			if (pf.poly.deg > 0)
				fail("ilaplace: %s grows like s^%d: its inverse has "
				    "derivatives of delta", xf_str(G[k].R), pf.poly.deg);
			g = pf_time(&pf, t, 0);
			if (pf.poly.deg == 0) {
				g = xadd(xmul(xk(pf.poly.c[0]), xdirac(t)), g);
				(*nd)++;
			}
			*sigma = fmax(*sigma, pf_abscissa(&pf));
			if (pf.np && poles)
				poles_text(poles, &pf, "s");
		} else if ((g = ilap_power(G[k].R, ss, t)) != 0) {
			double v;
			*sigma = fmax(*sigma, 0);
			(void)v;
		} else if ((g = ilap_symbolic(G[k].R, ss, t, assume)) != 0) {
			*sigma = fmax(*sigma, 0);
		} else
			fail("ilaplace: no rule for %s (rational functions of s, "
			    "times e^(-as); (s - a)^(-p))", xf_str(G[k].R));
		if (G[k].sh > 0)
			g = xmul(xsubst1(g, N.ki[t], xsub(t, xk(G[k].sh))),
			    xheav(xsub(t, xk(G[k].sh))));
		y = xadd(y, g);
	}
	return y;
}

/* forward-transform y numerically and compare with F at s right of sigma */
static void
ilap_check(const struct func *f, nid F, int ss, nid y, int ts, nid t,
    double sigma, int nd)
{
	int skip[2] = {ss, ts}, sl[MAXP], ns, i, ok, good = 1;
	double vals[MAXP], worst = 0, s0[3], base = isfinite(sigma) ?
	    fmax(sigma, 0) : 0;
	struct sbuf txt = {0};
	nid yb, Fb;

	ns = samples(f, skip, 2, sl, vals, &txt);
	yb = with_samples(y, sl, vals, ns);
	Fb = with_samples(F, sl, vals, ns);
	if (isfinite(sigma) && sigma > base)
		base = sigma;
	s0[0] = base + 1;
	s0[1] = base + 2;
	s0[2] = base + 3.5;
	for (i = 0; i < 3; i++) {
		double Iv = 0, Fv;
		if (!lap_numeric(xf_nodirac(yb), ts, t, s0[i], 0, s0[i] - base,
		    &Iv)) {
			good = 0;
			break;
		}
		if (nd) {
			/* the delta terms: c delta(t - a) -> c e^(-a s) */
			struct terms T;
			int j;
			terms_of(yb, &T);
			for (j = 0; j < T.n; j++) {
				struct facs Fc;
				int k;
				memset(&Fc, 0, sizeof Fc);
				Fc.coef = xk(1);
				fac_rec(T.t[j], ts, 1, &Fc);
				for (k = 0; k < Fc.nf; k++) {
					nid a;
					double a1, a0, cv;
					if (xis_dirac(Fc.f[k], &a) &&
					    lin_num(a, ts, &a1, &a0) && Fc.nf == 1 &&
					    xnum(Fc.coef, &cv))
						Iv += cv * exp(a0 / a1 * s0[i]) / fabs(a1);
				}
			}
		}
		Fv = xf_eval1(Fb, ss, s0[i], &ok);
		if (!ok) {
			good = 0;
			break;
		}
		worst = fmax(worst, reldiff(Iv, Fv));
	}
	if (!good)
		printf("  check: the transform back could not be evaluated\n");
	else
		printf("  check: transformed back (Gauss-Kronrod), it %s F at "
		    "s = %s, %s, %s%s%s: max relative difference %.2g\n",
		    worst < 1e-7 ? "matches" : "DIFFERS FROM", xf_num(s0[0]),
		    xf_num(s0[1]), xf_num(s0[2]), ns ? " with " : "",
		    ns ? txt.p : "", worst);
	free(txt.p);
}

static void
cmd_ilaplace(char *text)
{
	static const char *const outs[] = {"t", "x", NULL};
	static const char *const spref[] = {"s", "p", NULL};
	char sv[32] = "", tv[32] = "";
	struct func *f;
	struct sbuf poles = {0}, assume = {0};
	int ss, ts, nd;
	double sigma;
	nid s, t, y;

	trail_vars(text, sv, sizeof sv, tv, sizeof tv, outs);
	f = xf_parse(text);
	ss = pick_slot(f, sv, spref);
	if (ss < 0 && f->np > 0)
		fail("ilaplace: which variable is s?  (ilaplace F s t)");
	if (ss < 0)
		ss = max_slot(f);
	s = xv(ss, ss < f->np ? sym_str(f->p[ss].name) : "s");
	ts = ss == f->np ? f->np + 1 : f->np;
	if (ts >= MAXP)
		fail("too many variables");
	if (!tv[0])
		snprintf(tv, sizeof tv, "%s", xf_slot(f, "t") >= 0 ? "x" : "t");
	t = xv(ts, tv);
	y = ilap_node(f->body, ss, t, ts, &sigma, &poles, &assume, &nd);
	printf("L^-1{%s}(%s) = %s\n", fname_of(f), tv, xf_str(y));
	if (poles.n)
		printf("  poles: %s = %s\n", sym_str(N.a[s]), poles.p);
	printf("  for %s >= 0 (0 before): the right-sided inverse, region of "
	    "convergence Re(%s) > %s%s\n", tv, sym_str(N.a[s]),
	    isfinite(sigma) ? xf_num(sigma) : "-inf",
	    isfinite(sigma) ? " (right of every pole)" : "");
	if (assume.n)
		printf("  assuming %s\n", assume.p);
	if (nd)
		printf("  the constant part of F gives delta(%s)\n", tv);
	ilap_check(f, f->body, ss, y, ts, t, sigma, nd);
	(void)s;
	free(poles.p);
	free(assume.p);
}

/* ---- linear equations: y'' + 3y' = f(t), y[n+1] - a y[n] = f(n) ---- */

/*
 * Rewrites the unknown's derivatives (y'', d^2y/dt^2, y(t)) or shifts
 * (y[n + k]) into placeholder names __d<k> (k + 16 for shifts, so that
 * negative shifts are names too); returns the unknown's name in nm.
 */
static char upre = 'd';   /* placeholder prefix: __d1, __e1 */
static const char *uxname = "x";   /* never the unknown (tf's input) */

static void
unknowns(const char *in, struct sbuf *o, char *nm, size_t nmn, int shifts,
    const char *idx, int *kmin, int *kmax, const char *xname)
{
	size_t i = 0, n = strlen(in);

	sb_clear(o);
	*kmin = 1000;
	*kmax = -1000;
	while (i < n) {
		if (isdigit((unsigned char)in[i]) && (i == 0 ||
		    !wch((unsigned char)in[i - 1]))) {
			/* a number: 2y is 2 y */
			while (i < n && (isdigit((unsigned char)in[i]) ||
			    in[i] == '.'))
				sb_putc(o, in[i++]);
			continue;
		}
		if (wch((unsigned char)in[i]) && !isdigit((unsigned char)in[i]) &&
		    (i == 0 || !wch((unsigned char)in[i - 1]) ||
		    isdigit((unsigned char)in[i - 1]))) {
			size_t j = i;
			char w[32];
			while (j < n && wch((unsigned char)in[j]))
				j++;
			snprintf(w, sizeof w, "%.*s", (int)(j - i), in + i);
			/* dy/dt, d^2y/dt^2 */
			if (!shifts && w[0] == 'd' && (in[j] == '/' || in[j] == '^')) {
				size_t k = j;
				int ord = 1;
				char un[32];
				if (in[k] == '^') {
					ord = atoi(in + k + 1);
					k++;
					while (k < n && isdigit((unsigned char)in[k]))
						k++;
				}
				if (w[1]) {
					/* dy/dt: the name is in w */
					snprintf(un, sizeof un, "%s", w + 1);
				} else {
					size_t b = k;
					while (k < n && wch((unsigned char)in[k]))
						k++;
					snprintf(un, sizeof un, "%.*s", (int)(k - b), in + b);
				}
				if (k < n && in[k] == '/' && in[k + 1] == 'd' && un[0]) {
					k += 2;
					while (k < n && wch((unsigned char)in[k]))
						k++;
					if (in[k] == '^') {
						k++;
						while (k < n && isdigit((unsigned char)in[k]))
							k++;
					}
					if (!nm[0])
						snprintf(nm, nmn, "%s", un);
					if (!strcmp(un, nm)) {
						if (o->n && (isdigit((unsigned char)o->p[o->n - 1]) ||
						    o->p[o->n - 1] == ')'))
							sb_putc(o, '*');
						sb_printf(o, "__%c%d", upre, ord);
						if (ord > *kmax)
							*kmax = ord;
						if (ord < *kmin)
							*kmin = ord;
						i = k;
						continue;
					}
				}
			}
			if ((nm[0] && !strcmp(w, nm)) || (!nm[0] && !shifts &&
			    in[j] == '\'') || (!nm[0] && shifts && in[j] == '[' &&
			    strcmp(w, xname))) {
				size_t k = j;
				int ord = 0;
				if (!nm[0])
					snprintf(nm, nmn, "%s", w);
				if (!shifts) {
					while (k < n && in[k] == '\'')
						ord++, k++;
					/* y(t) */
					if (in[k] == '(') {
						size_t e = k;
						while (e < n && in[e] != ')')
							e++;
						if (e < n && has_ident(in + k, idx) &&
						    e - k - 1 == strlen(idx))
							k = e + 1;
					}
				} else if (in[k] == '[') {
					/* y[n + k] */
					size_t e = k + 1;
					char sub[64];
					int depth = 1;
					while (e < n && depth) {
						if (in[e] == '[')
							depth++;
						if (in[e] == ']')
							depth--;
						e++;
					}
					snprintf(sub, sizeof sub, "%.*s", (int)(e - k - 2),
					    in + k + 1);
					{
						char *p = sub, *q;
						while (*p == ' ')
							p++;
						if (strncmp(p, idx, strlen(idx)))
							fail("%s[%s]: only %s[%s + k] is understood",
							    nm, sub, nm, idx);
						p += strlen(idx);
						while (*p == ' ')
							p++;
						if (*p == 0)
							ord = 0;
						else {
							int sg = *p == '-' ? -1 : 1;
							if (*p != '+' && *p != '-')
								fail("%s[%s]: only %s[%s + k]", nm, sub, nm,
								    idx);
							ord = sg * (int)strtol(p + 1, &q, 10);
							while (*q == ' ')
								q++;
							if (*q)
								fail("%s[%s]: the shift must be a number",
								    nm, sub);
						}
					}
					k = e;
				}
				if (o->n && (isdigit((unsigned char)o->p[o->n - 1]) ||
				    o->p[o->n - 1] == ')'))
					sb_putc(o, '*');
				sb_printf(o, "__%c%d", upre, ord + (shifts ? 16 : 0));
				if (ord > *kmax)
					*kmax = ord;
				if (ord < *kmin)
					*kmin = ord;
				i = k;
				continue;
			}
			sb_printf(o, "%s", w);
			i = j;
			continue;
		}
		sb_putc(o, in[i]);
		i++;
	}
}

#define MAXORD 12
struct lineq {
	int order, kmin;          /* coefficients a[0..order] of shifts kmin.. */
	double a[MAXORD + 1];
	nid g;                    /* the forcing, in the independent slot */
	int xs;                   /* its slot */
	nid x;
	char nm[32], xn[32];
	struct func *f;
};

/* E = sum a_k d_k - g(x): the coefficients, checked to be linear */
static void
lineq_of(struct lineq *L, const char *eqtext, int shifts)
{
	struct sbuf o = {0}, t = {0};
	char *lhs, *rhs, *eq = xstrdup(eqtext), *p;
	int kmin, kmax, k, depth = 0;
	struct func *f;
	nid E, z;

	/* the = of the equation (not ==, <=, >=) */
	for (p = eq; *p; p++) {
		if (*p == '(' || *p == '[')
			depth++;
		if (*p == ')' || *p == ']')
			depth--;
		if (*p == '=' && depth == 0 && p[1] != '=' && (p == eq ||
		    (p[-1] != '<' && p[-1] != '>' && p[-1] != '!' && p[-1] != '=')))
			break;
	}
	if (!*p)
		fail("expected an equation with =");
	*p = 0;
	lhs = eq;
	rhs = p + 1;
	sb_printf(&t, "(%s) - (%s)", lhs, rhs);
	unknowns(t.p, &o, L->nm, sizeof L->nm, shifts, L->xn, &kmin, &kmax,
	    uxname);
	if (!L->nm[0] || kmax < kmin)
		fail("no unknown function (y'' + y = 0, y[n+1] = 2y[n])");
	if (kmax - kmin > MAXORD)
		fail("order above %d", MAXORD);
	f = xf_parse(o.p);
	L->f = f;
	L->kmin = kmin;
	L->order = kmax - kmin;
	L->xs = xf_slot(f, L->xn);
	if (L->xs < 0) {
		/* the independent variable does not occur: add it */
		L->xs = f->np;
		if (L->xs >= MAXP - 1)
			fail("too many variables");
	}
	L->x = xv(L->xs, L->xn);
	E = f->body;
	z = E;
	for (k = 0; k < f->np; k++)
		if (!strncmp(sym_str(f->p[k].name), "__d", 3))
			z = xsubst1(z, k, xk(0));
	L->g = xneg(z);
	for (k = 0; k <= L->order; k++) {
		char nmk[16];
		int sl;
		double v;
		snprintf(nmk, sizeof nmk, "__d%d", k + kmin + (shifts ? 16 : 0));
		sl = xf_slot(f, nmk);
		L->a[k] = 0;
		if (sl < 0)
			continue;
		{
			nid d = diff_node(E, (uint32_t)sl);
			if (!xnum(d, &v)) {
				/* constant after all?  evaluate at two points */
				int ok1, ok2, sl2[MAXP], j, nn = 0;
				double x1[MAXP], x2[MAXP], v2;
				for (j = 0; j < f->np; j++) {
					sl2[nn] = j;
					x1[nn] = 0.37 + 0.11 * j;
					x2[nn++] = 1.91 - 0.23 * j;
				}
				v = xf_eval(d, sl2, x1, nn, &ok1);
				v2 = xf_eval(d, sl2, x2, nn, &ok2);
				if (!ok1 || !ok2 || fabs(v - v2) > 1e-9 * fmax(1, fabs(v)))
					fail("not linear with constant coefficients: the "
					    "coefficient of %s is %s", k + kmin == 0 ? L->nm :
					    nmk, xf_str(d));
			}
		}
		L->a[k] = v;
	}
	if (L->a[L->order] == 0)
		fail("the highest order term vanishes");
	/* linear: E = sum a_k d_k - g exactly, at random points */
	{
		int j, sl2[MAXP], nn = 0, ok;
		double x1[MAXP];
		nid lin = xneg(L->g);
		for (k = 0; k <= L->order; k++) {
			char nmk[16];
			int sl;
			snprintf(nmk, sizeof nmk, "__d%d", k + kmin + (shifts ? 16 : 0));
			sl = xf_slot(f, nmk);
			if (sl >= 0)
				lin = xadd(lin, xmul(xk(L->a[k]), xv(sl, nmk)));
		}
		for (j = 0; j < f->np; j++) {
			sl2[nn] = j;
			x1[nn++] = 0.53 + 0.29 * j;
		}
		if (fabs(xf_eval(xsub(E, lin), sl2, x1, nn, &ok)) > 1e-9 ||
		    !ok) {
			for (j = 0; j < f->np; j++)
				x1[j] = 1.7 - 0.41 * j;
			if (fabs(xf_eval(xsub(E, lin), sl2, x1, nn, &ok)) > 1e-9)
				fail("the equation is not linear in %s", L->nm);
		}
	}
	free(o.p);
	free(t.p);
	free(eq);
}

/* initial values: y(0) = 1, y'(0) = 0, y[1] = 2 */
static void
initial_values(char **part, int np, const char *nm, int shifts, double *iv,
    int *have, int max)
{
	int k;

	for (k = 0; k < np; k++) {
		char *p = part[k], *eqp, *q;
		int ord = 0;
		size_t l = strlen(nm);
		struct func *f;
		double v, at;
		if (strncmp(p, nm, l))
			fail("expected an initial value like %s, not '%s'",
			    shifts ? "y[0] = 1" : "y(0) = 1", p);
		q = p + l;
		while (*q == '\'')
			ord++, q++;
		if (*q != (shifts ? '[' : '('))
			fail("expected %s(0) = value", nm);
		eqp = strchr(q, '=');
		if (!eqp)
			fail("%s: expected = value", p);
		{
			char *close = strchr(q, shifts ? ']' : ')');
			char tmp[64];
			if (!close || close > eqp)
				fail("%s: expected %s(0)", p, nm);
			snprintf(tmp, sizeof tmp, "%.*s", (int)(close - q - 1), q + 1);
			f = xf_parse(tmp);
			if (!xnum(f->body, &at))
				fail("%s: the point must be a number", p);
		}
		if (shifts) {
			if (at != floor(at) || at < 0)
				fail("%s: initial values are y[0], y[1], ...", p);
			ord = (int)at;
		} else if (at != 0)
			fail("%s: initial values are taken at t = 0", p);
		f = xf_parse(eqp + 1);
		if (!xnum(f->body, &v))
			fail("%s: the value must be a number", p);
		if (ord >= max)
			fail("%s: beyond the order of the equation", p);
		iv[ord] = v;
		have[ord] = 1;
	}
}

static nid
coef_poly_node(const double *a, int order, nid s)
{
	struct xpoly P;
	int k;

	xp_zero(&P);
	for (k = 0; k <= order; k++)
		P.c[k] = a[k];
	P.deg = order;
	xp_trim(&P);
	return xp_node(&P, s);
}

static void
cmd_lsolve(char *text)
{
	char *part[16];
	int np, k, have[MAXORD] = {0}, nd, ss;
	double iv[MAXORD] = {0}, sigma;
	struct lineq L;
	struct lctx C;
	struct xpoly P, Q;
	nid Pn, Qn, F, Y = xk(0), y, s;
	struct sbuf missing = {0}, poles = {0}, assume = {0};

	{
		/* "with" and "and" between the conditions */
		static const char *const w[] = {"with", "where", "given", "and",
			NULL};
		long at;
		while ((at = find_word(text, w, 0)) >= 0) {
			size_t l = strncmp(text + at, "with", 4) == 0 ? 4 :
			    strncmp(text + at, "where", 5) == 0 ? 5 :
			    strncmp(text + at, "given", 5) == 0 ? 5 : 3;
			text[at] = ',';
			memset(text + at + 1, ' ', l - 1);
		}
	}
	np = split_list(text, part, 16);
	memset(&L, 0, sizeof L);
	snprintf(L.xn, sizeof L.xn, "t");
	if (!has_ident(part[0], "t") && has_ident(part[0], "x"))
		snprintf(L.xn, sizeof L.xn, "x");
	lineq_of(&L, part[0], 0);
	if (L.kmin != 0)
		fail("expected derivatives of %s", L.nm);
	initial_values(part + 1, np - 1, L.nm, 0, iv, have, L.order);
	for (k = 0; k < L.order; k++)
		if (!have[k])
			sb_printf(&missing, "%s%s%.*s(0)", missing.n ? ", " : "", L.nm,
			    k, "''''''''''''");
	/* transform the forcing */
	memset(&C, 0, sizeof C);
	C.ts = L.xs;
	C.t = L.x;
	ss = L.f->np > L.xs ? L.f->np : L.xs + 1;
	if (ss >= MAXP)
		fail("too many variables");
	C.ss = ss;
	C.s = s = xv(ss, "s");
	if (!(is_const(L.g) && const_is(L.g, 0)))
		lap_rec(&C, L.g, 0);
	F = lap_total(&C, 0, NULL);
	/* P(s) Y - Q(s) = F */
	xp_zero(&P);
	xp_zero(&Q);
	for (k = 0; k <= L.order; k++) {
		int j;
		P.c[k] = L.a[k];
		for (j = 0; j < k; j++)
			Q.c[k - 1 - j] += L.a[k] * iv[j];
	}
	P.deg = L.order;
	Q.deg = L.order - 1;
	xp_trim(&P);
	xp_trim(&Q);
	Pn = xp_node(&P, s);
	Qn = xp_node(&Q, s);
	{
		/* echo the problem */
		int j;
		printf("%s", part[0]);
		for (j = 0; j < L.order; j++)
			printf(", %s%.*s(0) = %s", L.nm, j, "''''''''''''",
			    xf_num(iv[j]));
		printf("\n");
		if (missing.n)
			printf("  note: %s not given: taken as 0\n", missing.p);
	}
	{
		struct sbuf l = {0};
		if (P.deg >= 1 && (P.c[0] != 0 || P.deg > 1 || P.c[1] != 1))
			sb_printf(&l, "(%s)Y(s)", xf_str(Pn));
		else
			sb_printf(&l, "%sY(s)", P.deg == 1 ? "s" : xf_str(Pn));
		if (Q.deg >= 0) {
			const char *qs = xf_str(Qn);
			if (qs[0] == '-')
				sb_printf(&l, " + %s", qs + 1);
			else if (Q.deg == 0 || (Q.deg >= 1 && 0))
				sb_printf(&l, " - %s", qs);
			else
				sb_printf(&l, " - (%s)", qs);
		}
		printf("  transformed: %s = %s\n", l.p, xf_str(F));
		free(l.p);
	}
	/* Y = (F + Q)/P, group by group */
	{
		int g;
		int any0 = 0;
		for (g = 0; g < C.n; g++) {
			nid R = xf_polish(C.R[g], ss, s);
			if (C.sh[g] == 0 && !C.dl[g]) {
				R = xadd(R, Qn);
				any0 = 1;
			}
			R = xf_polish(xdiv(R, Pn), ss, s);
			if (C.sh[g] != 0)
				R = xmul(R, xexp(xmul(xk(-C.sh[g]), s)));
			Y = xadd(Y, R);
		}
		if (!any0 && Q.deg >= 0)
			Y = xadd(xf_polish(xdiv(Qn, Pn), ss, s), Y);
	}
	printf("  Y(s) = %s\n", xf_str(Y));
	y = ilap_node(Y, ss, L.x, L.xs, &sigma, &poles, &assume, &nd);
	printf("  %s(%s) = %s    for %s >= 0\n", L.nm, L.xn, xf_str(y), L.xn);
	if (poles.n)
		printf("  poles of Y: %s\n", poles.p);
	/* checks: the residual, the initial values, a numeric solution */
	{
		nid d[MAXORD + 1];
		double res = 0, scale = 0, ivd = 0;
		int ok, n = 0, i;
		d[0] = y;
		for (k = 1; k <= L.order; k++)
			d[k] = diff_node(d[k - 1], (uint32_t)L.xs);
		for (i = 0; i <= 200; i++) {
			double tt = 10.0 * i / 200 + 1e-9, r, gv;
			int ok2;
			r = 0;
			for (k = 0; k <= L.order; k++) {
				double v = xf_eval1(d[k], L.xs, tt, &ok);
				if (!ok)
					break;
				r += L.a[k] * v;
				scale = fmax(scale, fabs(L.a[k] * v));
			}
			if (k <= L.order)
				continue;
			gv = xf_eval1(L.g, L.xs, tt, &ok2);
			if (!ok2)
				continue;
			res = fmax(res, fabs(r - gv));
			n++;
		}
		for (k = 0; k < L.order; k++) {
			double v = xf_eval1(d[k], L.xs, 0, &ok);
			ivd = fmax(ivd, ok ? fabs(v - iv[k]) : INFINITY);
		}
		printf("  check: residual of the equation at %d points of [0, 10]: "
		    "max %.2g%s; initial values %s (max difference %.2g)\n", n, res,
		    res <= 1e-8 * fmax(1, scale) ? "" : "  (TOO LARGE)",
		    ivd <= 1e-9 ? "hold" : "DO NOT HOLD", ivd);
	}
	{
		/* RK4 on the system, h = 1/1000 */
		int m = L.order, steps = 10000, i, j, okk, maxd = 0;
		double h = 10.0 / steps, z[MAXORD], worst = 0, scale = 1;
		/* the forcing at each stage: just inside the step, so that a
		 * jump at a grid point is seen from the correct side */
		double *ts = xmalloc((size_t)(3 * steps) * sizeof *ts);
		double *gv = xmalloc((size_t)(3 * steps) * sizeof *gv);
		uint8_t *gok = xmalloc((size_t)(3 * steps));
		for (i = 0; i < steps; i++) {
			ts[3 * i] = (i + 1e-9) * h;
			ts[3 * i + 1] = (i + 0.5) * h;
			ts[3 * i + 2] = (i + 1 - 1e-9) * h;
		}
		xf_evalv(L.g, L.xs, ts, 3 * steps, NULL, NULL, 0, gv, gok);
		for (i = 0; i < 3 * steps; i++)
			if (!gok[i])
				gv[i] = i > 0 ? gv[i - 1] : 0;
		for (j = 0; j < m; j++)
			z[j] = iv[j];
		for (i = 0; i < steps; i++) {
			double k1[MAXORD], k2[MAXORD], k3[MAXORD], k4[MAXORD],
			    w[MAXORD];
			int st;
			for (st = 0; st < 4; st++) {
				double *kk = st == 0 ? k1 : st == 1 ? k2 : st == 2 ? k3 : k4;
				double gg = gv[3 * i + (st == 0 ? 0 : st == 3 ? 2 : 1)], acc;
				for (j = 0; j < m; j++)
					w[j] = z[j] + (st == 0 ? 0 : st == 3 ? h : h / 2) *
					    (st == 0 ? 0 : st == 1 ? k1[j] : st == 2 ? k2[j] :
					    k3[j]);
				for (j = 0; j + 1 < m; j++)
					kk[j] = w[j + 1];
				acc = gg;
				for (j = 0; j < m; j++)
					acc -= L.a[j] * w[j];
				kk[m - 1] = acc / L.a[m];
			}
			for (j = 0; j < m; j++)
				z[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
			if ((i + 1) % 500 == 0) {
				double yv = xf_eval1(y, L.xs, (i + 1) * h, &okk);
				if (okk) {
					worst = fmax(worst, fabs(yv - z[0]));
					scale = fmax(scale, fabs(yv));
					maxd++;
				}
			}
		}
		printf("  check: a numeric solution (RK4, h = 1/1000) agrees at %d "
		    "points of [0, 10]: max difference %.2g%s\n", maxd, worst,
		    worst <= 1e-6 * scale ? "" : "  (LARGE: a step in f?)");
		free(ts);
		free(gv);
		free(gok);
	}
	free(missing.p);
	free(poles.p);
	free(assume.p);
}

/* ---- Z transform ---- */

/* a failure that a probe (genfunc trying the Z transform) catches quietly */
static int zquiet;
static jmp_buf zjmp;

static void
zfail(const char *fmt, ...)
{
	va_list ap;
	char buf[480];

	if (zquiet)
		longjmp(zjmp, 1);
	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	fail("%s", buf);
}

enum {
	ZR_LIN = 1, ZR_SCALE = 2, ZR_N = 4, ZR_TRIG = 8, ZR_DELAY = 16,
	ZR_DELTA = 32, ZR_P2S = 64, ZR_POLY = 128, ZR_FACT = 256,
	ZR_PIECE = 512, ZR_POWRED = 1024
};
static const char *const zr_names[] = {"linearity",
	"a^n x[n] -> X(z/a)", "n x[n] -> -z X'(z)", "cos/sin table",
	"x[n - k] u[n - k] -> z^-k X(z)", "delta[n - k] -> z^-k",
	"product to sum", "polynomial expanded", "a^n/n! -> e^(a/z)",
	"piecewise as steps", "power reduction", NULL};

struct zctx {
	int ns, zs;
	nid n, z;
	nid X;
	nid roc;            /* max |r|: |z| > roc */
	int hasroc;
	unsigned rules;
	int depth;
};

static void
zc_roc(struct zctx *C, nid r)
{
	double v;
	nid a = xnum(r, &v) ? xk(fabs(v)) : mk_un(O_ABS, r);

	C->roc = C->hasroc ? mk_bin(O_MAX, C->roc, a) : a;
	C->hasroc = 1;
}

/* the discrete indicator of a condition on n: steps u[n - k] */
static nid
indicator_d(nid c, int slot, nid x)
{
	int op;
	double c1, c0, a;

	c = unc(c);
	op = N.op[c];
	if (op == O_LAND) {
		nid A = indicator_d(N.a[c], slot, x), B = A ? indicator_d(N.b[c],
		    slot, x) : 0;
		return A && B ? xmul(A, B) : 0;
	}
	if (op == O_LNOT) {
		nid A = indicator_d(N.a[c], slot, x);
		return A ? xsub(xk(1), A) : 0;
	}
	if (!(opinfo[op].fl & OF_CMP))
		return 0;
	if (!lin_num(xsub(N.a[c], N.b[c]), slot, &c1, &c0))
		return 0;
	a = -c0 / c1;
	if (c1 < 0) {
		/* flip: -n + c > 0 is n < c */
		op = op == O_GE ? O_LE : op == O_GT ? O_LT : op == O_LE ? O_GE :
		    op == O_LT ? O_GT : op;
	}
	switch (op) {
	case O_GE:
		return xheav(xsub(x, xk(ceil(a))));
	case O_GT:
		return xheav(xsub(x, xk(floor(a) + 1)));
	case O_LT:
		return xsub(xk(1), xheav(xsub(x, xk(ceil(a)))));
	case O_LE:
		return xsub(xk(1), xheav(xsub(x, xk(floor(a) + 1))));
	case O_EQ:
		return a == floor(a) ? xkron(xsub(x, xk(a))) : xk(0);
	case O_NE:
		return a == floor(a) ? xsub(xk(1), xkron(xsub(x, xk(a)))) : xk(1);
	}
	return 0;
}

/* Eulerian numbers E(k, j) */
static double
euler_num(int k, int j)
{
	static double E[16][16];
	static int done;

	if (!done) {
		int a, b;
		E[0][0] = 1;
		for (a = 1; a < 16; a++)
			for (b = 0; b < a; b++)
				E[a][b] = (b + 1) * E[a - 1][b] +
				    (a - b) * (b > 0 ? E[a - 1][b - 1] : 0);
		done = 1;
	}
	return k < 16 && j < 16 && j >= 0 ? E[k][j] : 0;
}

/* Z{n^k r^n} */
static nid
z_npow(nid z, nid r, int k)
{
	nid num = xk(0);
	int j;

	if (k == 0)
		return xdiv(z, xsub(z, r));
	for (j = 0; j < k; j++)
		num = xadd(num, xmul(xk(euler_num(k, j)), xmul(xpow(r, xk(j + 1)),
		    xpow(z, xk(k - j)))));
	return xdiv(num, xpow(xsub(z, r), xk(k + 1)));
}

static void zt_rec(struct zctx *C, nid e, int delay);

/* Z{c n^k r^n trig(b n + phi)} */
static nid
zt_core(struct zctx *C, int k, nid r, int trig, nid beta, nid phi)
{
	nid z = C->z;
	double ph = 0, rv, bv;

	zc_roc(C, r);
	if (k > 0)
		C->rules |= ZR_N;
	if (trig == 0)
		return z_npow(z, r, k);
	C->rules |= ZR_TRIG;
	if (trig == O_SINH || trig == O_COSH) {
		/* (r e^b)^n and (r e^-b)^n */
		nid r1 = xmul(r, xexp(beta)), r2 = xmul(r, xexp(xneg(beta)));
		nid A = z_npow(z, r1, k), B = z_npow(z, r2, k), R;
		zc_roc(C, r1);
		zc_roc(C, r2);
		if (phi && !(xnum(phi, &ph) && ph == 0)) {
			A = xmul(xexp(phi), A);
			B = xmul(xexp(xneg(phi)), B);
		}
		R = trig == O_SINH ? xdiv(xsub(A, B), xk(2)) :
		    xdiv(xadd(A, B), xk(2));
		return R;
	}
	if (k > 0 && xnum(r, &rv) && xnum(beta, &bv)) {
		/* numeric: n^k w^n, w = r e^(ib), as a complex rational */
		double complex w = rv * cexp(I * bv), Nc[XD + 1], Dc[XD + 1];
		struct xpoly Pr, Pi, Dd;
		int j, i, nd = 0;
		double complex ph1 = 1;
		/* numerator sum_j E(k, j) w^(j+1) z^(k-j), denominator (z-w)^(k+1) */
		for (i = 0; i <= XD; i++)
			Nc[i] = Dc[i] = 0;
		for (j = 0; j < k; j++)
			Nc[k - j] += euler_num(k, j) * cpow(w, j + 1);
		Dc[0] = 1;
		for (j = 0; j <= k; j++) {
			for (i = nd + 1; i >= 1; i--)
				Dc[i] = Dc[i - 1] - w * Dc[i];
			Dc[0] = -w * Dc[0];
			nd++;
		}
		/* N/D = N conj(D) / |D|^2 for real z */
		xp_zero(&Pr);
		xp_zero(&Pi);
		xp_zero(&Dd);
		for (i = 0; i <= k; i++)
			for (j = 0; j <= nd; j++) {
				double complex t = Nc[i] * conj(Dc[j]);
				Pr.c[i + j] += creal(t);
				Pi.c[i + j] += cimag(t);
			}
		for (i = 0; i <= nd; i++)
			for (j = 0; j <= nd; j++)
				Dd.c[i + j] += creal(Dc[i] * conj(Dc[j]));
		Pr.deg = Pi.deg = k + nd;
		Dd.deg = 2 * nd;
		xp_trim(&Pr);
		xp_trim(&Pi);
		xp_trim(&Dd);
		if (phi)
			xnum(phi, &ph);
		ph1 = cexp(I * ph);
		{
			/* cos(bn + ph) = Re(e^(i ph) w^n), sin = Im */
			struct xpoly num;
			double cr = creal(ph1), ci = cimag(ph1);
			xp_zero(&num);
			for (i = 0; i <= XD; i++)
				num.c[i] = trig == O_COS ? cr * Pr.c[i] - ci * Pi.c[i] :
				    ci * Pr.c[i] + cr * Pi.c[i];
			num.deg = XD;
			xp_trim(&num);
			return xdiv(xp_node(&num, z), xp_node(&Dd, z));
		}
	}
	{
		nid cb = mk_un(O_COS, beta), sb = mk_un(O_SIN, beta);
		nid den = xadd(xsub(xpow(z, xk(2)), xmul(xmul(xk(2), r),
		    xmul(z, cb))), xpow(r, xk(2)));
		nid Zc = xdiv(xmul(z, xsub(z, xmul(r, cb))), den);
		nid Zs = xdiv(xmul(xmul(r, z), sb), den);
		nid R;
		if (phi && !(xnum(phi, &ph) && ph == 0)) {
			nid cp = mk_un(O_COS, phi), sp = mk_un(O_SIN, phi);
			R = trig == O_COS ? xsub(xmul(cp, Zc), xmul(sp, Zs)) :
			    xadd(xmul(cp, Zs), xmul(sp, Zc));
		} else
			R = trig == O_COS ? Zc : Zs;
		for (; k > 0; k--)
			R = xneg(xmul(z, diff_node(R, (uint32_t)C->zs)));
		return R;
	}
}

static void
zt_term(struct zctx *C, nid term, int delay)
{
	struct facs F;
	double pw = 0, v, c1, c0;
	nid r = xk(1), beta = 0, phi = 0, u, arg;
	int k, trig = 0, trigk = -1, fact = 0;
	int64_t step = 0;
	uint32_t steps = 0;

	memset(&F, 0, sizeof F);
	F.coef = xk(1);
	fac_rec(term, C->ns, 1, &F);
	for (k = 0; k < F.nf; k++) {
		nid f = unc(F.f[k]);
		double p = F.pw[k];
		int op = N.op[f];
		if (is_var(f, C->ns)) {
			pw += p;
			continue;
		}
		if (op == O_POW && !has_slot(N.a[f], C->ns) &&
		    lin_of(N.b[f], C->ns, &u, &arg)) {
			/* a^(c1 n + c0) */
			r = xmul(r, xpow(N.a[f], xmul(xk(p), u)));
			F.coef = xmul(F.coef, xpow(N.a[f], xmul(xk(p), arg)));
			C->rules |= ZR_SCALE;
			continue;
		}
		if (xis_exp(f, &u) && lin_of(u, C->ns, &beta, &arg)) {
			r = xmul(r, xexp(xmul(xk(p), beta)));
			F.coef = xmul(F.coef, xexp(xmul(xk(p), arg)));
			beta = 0;
			C->rules |= ZR_SCALE;
			continue;
		}
		if (trig_op(op) && p == 1 && lin_of(N.a[f], C->ns, &u, &arg)) {
			if (trig) {
				nid g = facs_except(&F, (1u << k) | (1u << trigk));
				C->rules |= ZR_P2S;
				zt_rec(C, xmul(g, prod2sum(N.op[unc(F.f[trigk])],
				    N.a[unc(F.f[trigk])], op, N.a[f])), delay);
				return;
			}
			trig = op;
			trigk = k;
			beta = u;
			phi = arg;
			continue;
		}
		if (trig_op(op) && p == floor(p) && p >= 2 && p <= 6) {
			nid w = N.a[f], red, rest;
			nid c2 = mk_un(op == O_SIN || op == O_COS ? O_COS : O_COSH,
			    xmul(xk(2), w));
			if (op == O_SIN)
				red = xdiv(xsub(xk(1), c2), xk(2));
			else if (op == O_COS)
				red = xdiv(xadd(xk(1), c2), xk(2));
			else if (op == O_SINH)
				red = xdiv(xsub(c2, xk(1)), xk(2));
			else
				red = xdiv(xadd(c2, xk(1)), xk(2));
			rest = facs_except(&F, 1u << k);
			if (p > 2)
				rest = xmul(rest, xpow(f, xk(p - 2)));
			C->rules |= ZR_POWRED;
			zt_rec(C, xmul(rest, red), delay);
			return;
		}
		if (xis_heav(f, &arg) && p == 1 && lin_num(arg, C->ns, &c1, &c0)) {
			double a = -c0 / c1;
			if (c1 > 0) {
				int64_t s2 = (int64_t)ceil(a - 1e-12);
				if (s2 > step)
					step = s2;
				steps |= 1u << k;
				continue;
			}
			{
				/* u[a - n] = 1 - u[n - floor(a) - 1] */
				nid g = facs_except(&F, 1u << k);
				C->rules |= ZR_DELAY;
				zt_rec(C, xsub(g, xmul(g, xheav(xsub(C->n,
				    xk(floor(a) + 1))))), delay);
				return;
			}
		}
		if (xis_kron(f, &arg) && p == 1 && lin_num(arg, C->ns, &c1, &c0)) {
			double a = -c0 / c1;
			nid g = facs_except(&F, 1u << k);
			C->rules |= ZR_DELTA;
			if (a != floor(a) || a < 0)
				return;
			C->X = xadd(C->X, xmul(xsubst1(g, C->ns, xk(a)),
			    xpow(C->z, xk(-(a + delay)))));
			if (!C->hasroc) {
				C->roc = xk(0);
				C->hasroc = 1;
			}
			return;
		}
		if (op == O_SEL && p == 1) {
			nid H = indicator_d(N.a[f], C->ns, C->n);
			if (H) {
				nid g = facs_except(&F, 1u << k);
				C->rules |= ZR_PIECE;
				zt_rec(C, xmul(g, xadd(N.c[f], xmul(xsub(N.b[f], N.c[f]),
				    H))), delay);
				return;
			}
		}
		if (xis_fact(f, &arg) && p == -1 && is_var(arg, C->ns)) {
			fact = 1;
			continue;
		}
		if (p > 0 && p == floor(p) && p <= 12) {
			struct npoly np;
			nid fp = p == 1 ? f : xpow(f, xk(p));
			if (npoly_of(fp, C->ns, &np, 12) && np.deg >= 1) {
				nid g = facs_except(&F, 1u << k);
				int j;
				C->rules |= ZR_POLY;
				for (j = 0; j <= np.deg; j++)
					if (!(is_const(np.c[j]) && const_is(np.c[j], 0)))
						zt_rec(C, xmul(xmul(g, xadd(xk(0), np.c[j])),
						    xpow(C->n, xk(j))), delay);
				return;
			}
		}
		zfail("ztrans: no rule for the factor %s", xf_str(F.pw[k] == 1 ?
		    F.f[k] : xpow(F.f[k], xk(F.pw[k]))));
	}
	if (step > 0) {
		nid g = facs_except(&F, steps);
		C->rules |= ZR_DELAY;
		zt_rec(C, xsubst1(g, C->ns, xadd(C->n, xk((double)step))),
		    delay + (int)step);
		return;
	}
	if (pw != floor(pw) || pw < 0)
		zfail("ztrans: n^%s: no rule (powers n^k, k = 0, 1, 2, ...)",
		    xf_num(pw));
	{
		nid R;
		if (fact) {
			if (pw != 0 || trig)
				zfail("ztrans: n^k or a sine times 1/n!: no rule");
			C->rules |= ZR_FACT;
			R = xexp(xdiv(r, C->z));
			if (!C->hasroc) {
				C->roc = xk(0);
				C->hasroc = 1;
			}
		} else
			R = zt_core(C, (int)pw, r, trig, beta, phi);
		R = xmul(F.coef, R);
		if (delay)
			R = xmul(R, xpow(C->z, xk(-delay)));
		C->X = xadd(C->X, R);
	}
}

static void
zt_rec(struct zctx *C, nid e, int delay)
{
	struct terms T;
	int i;

	if (++C->depth > 40)
		zfail("ztrans: the expression is too deep");
	terms_of(e, &T);
	if (T.n > 1)
		C->rules |= ZR_LIN;
	for (i = 0; i < T.n; i++)
		zt_term(C, T.t[i], delay);
	C->depth--;
}

/* a coefficient before z^-k: 1 omitted, 1/2 in parentheses */
static const char *
zcoef(double c, int before)
{
	const char *t = xf_num(c);
	static char buf[4][64];
	static int k;
	char *b = buf[k++ & 3];

	if (!before)
		return t;
	if (c == 1)
		return "";
	if (c == -1)
		return "-";
	if (strchr(t, '/') || strchr(t, ' ') || strchr(t, '^')) {
		snprintf(b, 64, "(%s)", t);
		return b;
	}
	return t;
}

/* X(z) in z^-1: (b0 + b1 z^-1 ...)/(1 + a1 z^-1 ...) as text */
static void
zinv_text(struct sbuf *b, const struct xrat *r)
{
	int dd = r->d.deg, i, first;
	double a0 = r->d.c[dd];

	for (i = 0, first = 1; i <= dd; i++) {
		/* numerator coefficient of z^(dd - i): z^-i */
		double c = dd - i <= r->n.deg && dd - i >= 0 ? r->n.c[dd - i] /
		    a0 : 0;
		if (c == 0)
			continue;
		sb_printf(b, "%s%s", first ? "" : c < 0 ? " - " : " + ",
		    zcoef(first ? c : fabs(c), i > 0));
		if (i)
			sb_printf(b, "z^-%d", i);
		first = 0;
	}
	if (first)
		sb_puts(b, "0");
	{
		struct sbuf d = {0};
		for (i = 0, first = 1; i <= dd; i++) {
			double c = r->d.c[dd - i] / a0;
			if (c == 0)
				continue;
			sb_printf(&d, "%s%s", first ? "" : c < 0 ? " - " : " + ",
			    zcoef(first ? c : fabs(c), i > 0));
			if (i)
				sb_printf(&d, "z^-%d", i);
			first = 0;
		}
		if (strcmp(d.p, "1")) {
			const char *num = b->p;
			int paren = strstr(num, " + ") || strstr(num + 1, " - ");
			char *nb = xstrdup(num);
			sb_clear(b);
			sb_printf(b, paren ? "(%s)/(%s)" : "%s/(%s)", nb, d.p);
			free(nb);
		}
		free(d.p);
	}
}

static double
roc_value(nid roc, const int *sl, const double *vals, int ns)
{
	double r;
	int ok;
	nid v = with_samples(roc, sl, vals, ns);

	if (xnum(v, &r))
		return r;
	r = xf_eval(v, NULL, NULL, 0, &ok);
	return ok ? r : 1;
}

/* sum x[n] z0^-n numerically; 0 when it did not settle */
static int
zsum(nid x, int ns, double z0, double *val, int *nterms)
{
	int n, chunk = 256, quiet = 0, total = 0;
	double xs[256], out[256], S = 0, zp = 1;
	uint8_t ok[256];

	for (n = 0; n < 20000; n += chunk) {
		int i;
		for (i = 0; i < chunk; i++)
			xs[i] = n + i;
		xf_evalv(x, ns, xs, chunk, NULL, NULL, 0, out, ok);
		for (i = 0; i < chunk; i++) {
			double t;
			if (!ok[i] || !isfinite(out[i]))
				return 0;
			t = out[i] * zp;
			S += t;
			zp /= z0;
			total++;
			if (fabs(t) <= 1e-18 * fmax(fabs(S), 1e-300) || zp == 0)
				quiet++;
			else
				quiet = 0;
			if (quiet > 40) {
				*val = S;
				*nterms = total;
				return 1;
			}
		}
	}
	return 0;
}

static void
zt_check(const struct func *f, nid body, struct zctx *C, nid X)
{
	int skip[2] = {C->ns, C->zs}, sl[MAXP], ns, i, ok, good = 1, nt = 0;
	double vals[MAXP], R, worst = 0, z0[3];
	struct sbuf txt = {0};
	nid xb, Xb;

	ns = samples(f, skip, 2, sl, vals, &txt);
	xb = with_samples(body, sl, vals, ns);
	Xb = with_samples(X, sl, vals, ns);
	R = C->hasroc ? roc_value(C->roc, sl, vals, ns) : 0;
	z0[0] = R + 1;
	z0[1] = 2 * R + 2;
	z0[2] = 3 * R + 5;
	for (i = 0; i < 3; i++) {
		double S, Xv;
		int n;
		if (!zsum(xb, C->ns, z0[i], &S, &n)) {
			good = 0;
			break;
		}
		nt = n > nt ? n : nt;
		Xv = xf_eval1(Xb, C->zs, z0[i], &ok);
		if (!ok) {
			good = 0;
			break;
		}
		worst = fmax(worst, reldiff(S, Xv));
	}
	if (!good)
		printf("  check: the series could not be summed numerically\n");
	else
		printf("  check: %s sum_n x[n] z^-n (%d terms) at z = %s, %s, %s%s%s: "
		    "max relative difference %.2g\n", worst < 1e-9 ? "matches" :
		    "DIFFERS FROM", nt, xf_num(z0[0]), xf_num(z0[1]),
		    xf_num(z0[2]), ns ? " with " : "", ns ? txt.p : "", worst);
	free(txt.p);
}

static void
zrules_line(unsigned rules)
{
	int k, n = 0;

	printf("  rules:");
	for (k = 0; zr_names[k]; k++)
		if (rules & (1u << k))
			printf("%s %s", n++ ? "," : "", zr_names[k]);
	if (n == 0)
		printf(" a^n -> z/(z - a)");
	printf("\n");
}

/* the Z transform of body in slot ns; zs the slot of z */
static nid
zt_of(nid body, struct zctx *C)
{
	C->X = xk(0);
	zt_rec(C, body, 0);
	return xf_polish(C->X, C->zs, C->z);
}

static void
roc_print(struct zctx *C, const char *zn)
{
	double r;

	if (!C->hasroc)
		printf("  region of convergence: every z\n");
	else if (xnum(C->roc, &r))
		printf("  region of convergence: |%s| > %s\n", zn, xf_num(r));
	else
		printf("  region of convergence: |%s| > %s\n", zn, xf_str(C->roc));
}

static void
cmd_ztrans(char *text)
{
	static const char *const outs[] = {"z", "w", NULL};
	static const char *const npref[] = {"n", "k", NULL};
	char nv[32] = "", zv[32] = "";
	struct func *f;
	struct zctx C;
	struct xrat r;
	nid X;

	trail_vars(text, nv, sizeof nv, zv, sizeof zv, outs);
	f = xf_parse(text);
	memset(&C, 0, sizeof C);
	C.ns = pick_slot(f, nv, npref);
	if (C.ns < 0 && f->np > 0)
		fail("ztrans: which variable is n?  (ztrans X n z)");
	if (C.ns < 0)
		C.ns = max_slot(f);
	C.n = xv(C.ns, C.ns < f->np ? sym_str(f->p[C.ns].name) : "n");
	C.zs = C.ns == f->np ? f->np + 1 : f->np;
	if (C.zs >= MAXP)
		fail("too many variables");
	if (!zv[0])
		snprintf(zv, sizeof zv, "%s", xf_slot(f, "z") >= 0 ? "w" : "z");
	C.z = xv(C.zs, zv);
	xf_discrete = 1;
	X = zt_of(f->body, &C);
	printf("Z{%s}(%s) = %s\n", fname_of(f), zv, xf_str(X));
	if (!(node_slots(X) & ~(1ULL << C.zs)) && xq_of(X, C.zs, &r) &&
	    r.d.deg > 0) {
		struct sbuf b = {0};
		zinv_text(&b, &r);
		printf("  in powers of %s^-1: %s\n", zv, b.p);
		free(b.p);
	}
	roc_print(&C, zv);
	printf("  one-sided: x[%s] for %s >= 0\n", sym_str(N.a[C.n]),
	    sym_str(N.a[C.n]));
	zrules_line(C.rules);
	zt_check(f, f->body, &C, X);
	xf_discrete = 0;
}

/* ---- inverse Z ---- */

/* the sequence of a partial fraction expansion of X(z)/z */
static nid
pf_seq(const struct xpf *pf, nid n)
{
	nid x = xk(0);
	int i, k, j;

	for (i = 0; i < pf->np; i++) {
		const struct xpole *P = &pf->pole[i];
		double complex p = P->p;
		if (cabs(p) == 0) {
			/* a_k / z^k in X/z: a_k delta[n - (k - 1)] */
			for (k = 1; k <= P->m; k++)
				if (creal(P->a[k]) != 0)
					x = xadd(x, xmul(xk(creal(P->a[k])), xkron(xsub(n,
					    xk(k - 1)))));
			continue;
		}
		{
			/* sum_k a_k C(n, k-1) p^(n-k+1) = Q(n) p^n */
			double complex Qc[XD + 1];
			struct xpoly Qr, Qi;
			for (j = 0; j <= XD; j++)
				Qc[j] = 0;
			for (k = 1; k <= P->m; k++) {
				/* C(n, k-1) as a polynomial in n */
				double complex bc[XD + 1];
				int d = 0;
				for (j = 0; j <= XD; j++)
					bc[j] = 0;
				bc[0] = 1;
				for (j = 0; j < k - 1; j++) {
					int i2;
					/* times (n - j)/(j + 1) */
					for (i2 = d + 1; i2 >= 1; i2--)
						bc[i2] = (bc[i2 - 1] - j * bc[i2]) / (j + 1);
					bc[0] = -j * bc[0] / (j + 1);
					d++;
				}
				for (j = 0; j <= d; j++)
					Qc[j] += P->a[k] * cpow(p, 1 - k) * bc[j];
			}
			xp_zero(&Qr);
			xp_zero(&Qi);
			for (j = 0; j < P->m; j++) {
				Qr.c[j] = creal(Qc[j]);
				Qi.c[j] = cimag(Qc[j]);
			}
			Qr.deg = Qi.deg = P->m - 1;
			xp_trim(&Qr);
			xp_trim(&Qi);
			if (cimag(p) == 0) {
				nid term = xp_node(&Qr, n);
				if (creal(p) != 1)
					term = xmul(term, xpow(xk(creal(p)), n));
				x = xadd(x, term);
			} else {
				/* 2 Re(Q(n) p^n) = 2 rho^n (Re Q cos - Im Q sin)(theta n) */
				double rho = cabs(p), th = carg(p);
				nid tn = xmul(xk(th), n), term;
				xp_scale(&Qr, 2);
				xp_scale(&Qi, -2);
				term = xadd(xmul(xp_node(&Qr, n), mk_un(O_COS, tn)),
				    xmul(xp_node(&Qi, n), mk_un(O_SIN, tn)));
				if (fabs(rho - 1) > 1e-14)
					term = xmul(xpow(xk(rho), n), term);
				x = xadd(x, term);
			}
		}
	}
	return x;
}

/* long division: x[0..cnt-1] of N(z)/D(z), deg N <= deg D */
static int
long_div(const struct xrat *r, double *x, int cnt)
{
	int dd = r->d.deg, n, j;
	double a[XD + 1], b[XD + 1];

	if (r->n.deg > dd)
		return 0;
	for (j = 0; j <= dd; j++) {
		a[j] = r->d.c[dd - j];
		b[j] = dd - j <= r->n.deg ? r->n.c[dd - j] : 0;
	}
	for (n = 0; n < cnt; n++) {
		double v = n <= dd ? b[n] : 0;
		for (j = 1; j <= dd && j <= n; j++)
			v -= a[j] * x[n - j];
		x[n] = v / a[0];
	}
	return 1;
}

/* x[n] from a rational X(z) in slot zs (numeric): 0 if not rational */
static nid
iz_rational(nid X, int zs, nid n, struct xrat *r, struct sbuf *poles,
    double *R)
{
	struct xrat xz, zz;
	struct xpf pf;
	int i;

	if ((node_slots(X) & ~(1ULL << zs)) || !xq_of(X, zs, r))
		return 0;
	if (r->n.deg > r->d.deg)
		fail("iztrans: X grows like z^%d: not the transform of a causal "
		    "sequence", r->n.deg - r->d.deg);
	xp_const(&zz.n, 1);
	xp_zero(&zz.d);
	zz.d.c[1] = 1;
	zz.d.deg = 1;
	if (!xq_mul(r, &zz, &xz) || !xf_apart(&xz, &pf))
		fail("iztrans: the roots of the denominator were not found");
	*R = 0;
	for (i = 0; i < pf.np; i++) {
		*R = fmax(*R, cabs(pf.pole[i].p));
	}
	if (poles) {
		struct xpf p2;
		if (xf_apart(r, &p2))
			poles_text(poles, &p2, "z");
	}
	return pf_seq(&pf, n);
}

/* c z^k/(z - a)^m and e^(a/z), symbolically */
static nid
iz_symbolic(nid X, int zs, nid n)
{
	struct terms T;
	nid x = xk(0);
	int i, k;

	terms_of(X, &T);
	for (i = 0; i < T.n; i++) {
		struct facs F;
		nid den = 0, u;
		double m = 0, zk = 0, v;
		int ok = 1;
		memset(&F, 0, sizeof F);
		F.coef = xk(1);
		fac_rec(T.t[i], zs, 1, &F);
		for (k = 0; k < F.nf && ok; k++) {
			nid f = unc(F.f[k]);
			if (is_var(f, zs))
				zk += F.pw[k];
			else if (F.pw[k] < 0 && !den)
				den = f, m = -F.pw[k];
			else if (xis_exp(f, &u) && F.nf == 1 && F.pw[k] == 1) {
				/* e^(c/z): c^n/n! */
				struct facs G;
				memset(&G, 0, sizeof G);
				G.coef = xk(1);
				fac_rec(u, zs, 1, &G);
				if (G.nf != 1 || !is_var(G.f[0], zs) || G.pw[0] != -1)
					return 0;
				x = xadd(x, xmul(F.coef, xdiv(xpow(G.coef, n), xfact(n))));
				ok = 2;
			} else
				ok = 0;
		}
		if (ok == 2)
			continue;
		if (!ok || !den || m != floor(m))
			return 0;
		{
			struct npoly D;
			nid a, c1;
			if (!npoly_of(den, zs, &D, 1) || D.deg != 1)
				return 0;
			c1 = xadd(xk(0), D.c[1]);
			a = xneg(xdiv(xadd(xk(0), D.c[0]), c1));
			if (m == 1 && zk == 1)
				x = xadd(x, xmul(xdiv(F.coef, c1), xpow(a, n)));
			else if (m == 1 && zk == 0)
				x = xadd(x, xmul(xdiv(F.coef, c1), xmul(xpow(a, xsub(n,
				    xk(1))), xheav(xsub(n, xk(1))))));
			else if (m == 2 && zk == 1)
				x = xadd(x, xmul(xdiv(F.coef, xpow(c1, xk(2))), xmul(n,
				    xpow(a, xsub(n, xk(1))))));
			else
				return 0;
			(void)v;
		}
	}
	return x;
}

static void
cmd_iztrans(char *text)
{
	static const char *const outs[] = {"n", "k", NULL};
	static const char *const zpref[] = {"z", NULL};
	char zv[32] = "", nv[32] = "";
	struct func *f;
	struct xrat r;
	struct sbuf poles = {0};
	int zs, ns, i, ok;
	double R = 0;
	nid z, n, x;

	trail_vars(text, zv, sizeof zv, nv, sizeof nv, outs);
	f = xf_parse(text);
	zs = pick_slot(f, zv, zpref);
	if (zs < 0 && f->np > 0)
		fail("iztrans: which variable is z?  (iztrans X z n)");
	if (zs < 0)
		zs = max_slot(f);
	z = xv(zs, zs < f->np ? sym_str(f->p[zs].name) : "z");
	ns = zs == f->np ? f->np + 1 : f->np;
	if (ns >= MAXP)
		fail("too many variables");
	if (!nv[0])
		snprintf(nv, sizeof nv, "n");
	n = xv(ns, nv);
	xf_discrete = 1;
	x = iz_rational(f->body, zs, n, &r, &poles, &R);
	if (!x) {
		x = iz_symbolic(f->body, zs, n);
		if (!x)
			fail("iztrans: no rule for %s (rational functions of z; "
			    "c z/(z - a), e^(a/z))", xf_str(f->body));
		printf("Z^-1{%s}[%s] = %s    for %s >= 0\n", fname_of(f), nv,
		    xf_str(x), nv);
		printf("  taken as causal (right-sided): region of convergence "
		    "outside every pole\n");
	} else {
		double ld[32];
		printf("Z^-1{%s}[%s] = %s    for %s >= 0\n", fname_of(f), nv,
		    xf_str(x), nv);
		if (poles.n)
			printf("  poles: %s = %s\n", sym_str(N.a[z]), poles.p);
		printf("  taken as causal (right-sided): region of convergence "
		    "|%s| > %s\n", sym_str(N.a[z]), xf_num(R));
		if (long_div(&r, ld, 32)) {
			double worst = 0, sc = 0;
			printf("  long division in %s^-1: ", sym_str(N.a[z]));
			for (i = 0; i < 8; i++)
				printf("%s%s", i ? ", " : "", xf_num(ld[i]));
			printf(", ...\n");
			for (i = 0; i < 32; i++) {
				double v = xf_eval1(x, ns, i, &ok);
				worst = fmax(worst, ok ? fabs(v - ld[i]) : INFINITY);
				sc = fmax(sc, fabs(ld[i]));
			}
			printf("  check: the closed form %s the long division for "
			    "%s = 0..31 (max difference %.2g)\n",
			    worst <= 1e-9 * fmax(1, sc) ? "equals" : "DIFFERS FROM", nv,
			    worst);
		}
	}
	free(poles.p);
	xf_discrete = 0;
}

/* ---- recurrences by Z ---- */

/* the solution of a linear recurrence: y[n] as a node; prints the steps */
static nid
zsolve_core(char *text, int quiet, nid *Yout, struct zctx *Cout,
    struct lineq *Lout)
{
	char *part[16];
	int np, k, have[MAXORD] = {0}, i, zs;
	double iv[MAXORD] = {0};
	struct lineq L;
	struct zctx C;
	struct xpoly P, Q;
	struct xrat Yr;
	struct sbuf missing = {0}, poles = {0};
	nid Pn, Qn, G, Y, y, z;
	double R;

	np = split_list(text, part, 16);
	memset(&L, 0, sizeof L);
	snprintf(L.xn, sizeof L.xn, "n");
	if (!strchr(part[0], '['))
		fail("zsolve: expected a recurrence like y[n+1] = 2y[n] + 1");
	{
		/* the index: the name inside the first [ ] */
		const char *b = strchr(part[0], '[');
		size_t l = 0;
		b++;
		while (*b == ' ')
			b++;
		while (isalpha((unsigned char)b[l]))
			l++;
		if (l && l < sizeof L.xn)
			snprintf(L.xn, sizeof L.xn, "%.*s", (int)l, b);
	}
	upre = 'd';
	uxname = "";
	lineq_of(&L, part[0], 1);
	uxname = "x";
	initial_values(part + 1, np - 1, L.nm, 1, iv, have, L.order);
	for (k = 0; k < L.order; k++)
		if (!have[k])
			sb_printf(&missing, "%s%s[%d]", missing.n ? ", " : "", L.nm, k);
	/* m = n + kmin: sum_k a_k y[m + k] = g(m - kmin) for m >= 0 */
	memset(&C, 0, sizeof C);
	C.ns = L.xs;
	C.n = L.x;
	zs = L.f->np > L.xs ? L.f->np : L.xs + 1;
	if (zs >= MAXP)
		fail("too many variables");
	C.zs = zs;
	C.z = z = xv(zs, "z");
	{
		nid g = L.kmin ? xsubst1(L.g, L.xs, xsub(L.x, xk(L.kmin))) : L.g;
		L.g = g;
		G = is_const(g) && const_is(g, 0) ? xk(0) : zt_of(g, &C);
	}
	xp_zero(&P);
	xp_zero(&Q);
	for (k = 0; k <= L.order; k++) {
		int j;
		P.c[k] = L.a[k];
		for (j = 0; j < k; j++)
			Q.c[k - j] += L.a[k] * iv[j];
	}
	P.deg = L.order;
	Q.deg = L.order;
	xp_trim(&P);
	xp_trim(&Q);
	Pn = xp_node(&P, z);
	Qn = xp_node(&Q, z);
	Y = xf_polish(xdiv(xadd(G, Qn), Pn), zs, z);
	if (!quiet) {
		printf("%s", part[0]);
		for (k = 0; k < L.order; k++)
			printf(", %s[%d] = %s", L.nm, k, xf_num(iv[k]));
		printf("\n");
		if (missing.n)
			printf("  note: %s not given: taken as 0\n", missing.p);
		if (L.kmin)
			printf("  the recurrence holds for %s >= %d\n", L.xn, -L.kmin);
		printf("  transformed: (%s)Y(z)%s%s%s = %s\n", xf_str(Pn),
		    Q.deg >= 0 ? " - (" : "", Q.deg >= 0 ? xf_str(Qn) : "",
		    Q.deg >= 0 ? ")" : "", xf_str(G));
		printf("  Y(z) = %s\n", xf_str(Y));
	}
	y = iz_rational(Y, zs, L.x, &Yr, &poles, &R);
	if (!y)
		fail("zsolve: Y(z) is not a rational function of z");
	if (!quiet) {
		printf("  %s[%s] = %s    for %s >= 0\n", L.nm, L.xn, xf_str(y), L.xn);
		if (poles.n)
			printf("  poles of Y: %s\n", poles.p);
	}
	{
		/* iterate the recurrence */
		double it[64], worst = 0, sc = 0;
		int ok, nn = 40;
		for (i = 0; i < L.order; i++)
			it[i] = iv[i];
		for (i = 0; i + L.order < nn; i++) {
			double v = xf_eval1(L.g, L.xs, i, &ok);
			int j;
			if (!ok)
				v = 0;
			for (j = 0; j < L.order; j++)
				v -= L.a[j] * it[i + j];
			it[i + L.order] = v / L.a[L.order];
		}
		for (i = 0; i < nn; i++) {
			double v = xf_eval1(y, L.xs, i, &ok);
			worst = fmax(worst, ok ? fabs(v - it[i]) / fmax(1,
			    fabs(it[i])) : INFINITY);
			sc = fmax(sc, fabs(it[i]));
		}
		if (!quiet)
			printf("  check: iterating the recurrence gives the same %s[0..%d] "
			    "(max relative difference %.2g)%s\n", L.nm, nn - 1, worst,
			    worst <= 1e-9 ? "" : "  DIFFERS");
	}
	if (Yout)
		*Yout = Y;
	if (Cout)
		*Cout = C;
	if (Lout)
		*Lout = L;
	free(missing.p);
	free(poles.p);
	return y;
}

static void
cmd_zsolve(char *text)
{
	xf_discrete = 1;
	zsolve_core(text, 0, NULL, NULL, NULL);
	xf_discrete = 0;
}

/* ---- transfer functions ---- */

/* the names used as sequences (x[...]) or functions (y', y(t)) */
static int
seq_names(const char *s, int discrete, char names[4][32])
{
	size_t i = 0, n = strlen(s);
	int k = 0, j;

	while (i < n && k < 4) {
		if (isalpha((unsigned char)s[i]) && (i == 0 ||
		    !wch((unsigned char)s[i - 1]) || isdigit((unsigned char)s[i - 1]))) {
			size_t e = i;
			char w[32];
			while (e < n && wch((unsigned char)s[e]))
				e++;
			snprintf(w, sizeof w, "%.*s", (int)(e - i), s + i);
			if ((discrete && s[e] == '[') || (!discrete && (s[e] == '\'' ||
			    (s[e] == '(' && !in_list(unary_fns, w) &&
			    !builtin_name(w))))) {
				for (j = 0; j < k; j++)
					if (!strcmp(names[j], w))
						break;
				if (j == k)
					snprintf(names[k++], 32, "%s", w);
			}
			i = e;
			continue;
		}
		i++;
	}
	return k;
}

static void
freq_table(const struct xrat *H, int discrete)
{
	static const double wd[] = {0, M_PI / 8, M_PI / 4, M_PI / 2,
		3 * M_PI / 4, M_PI};
	double wc[6], w0 = 1;
	int i;

	if (!discrete) {
		double complex z[XD];
		int m[XD], nz = xf_roots(&H->d, z, m);
		double mx = 0;
		for (i = 0; i < nz; i++)
			mx = fmax(mx, cabs(z[i]));
		if (mx > 0)
			w0 = pow(10, floor(log10(mx) + 0.5));
		wc[0] = 0;
		wc[1] = w0 / 10;
		wc[2] = w0 / 2;
		wc[3] = w0;
		wc[4] = 2 * w0;
		wc[5] = 10 * w0;
	}
	printf("  frequency response %s:\n", discrete ? "H(e^(iw)), w in "
	    "radians per sample" : "H(iw)");
	printf("    %-10s %-12s %-10s %s\n", "w", "|H|", "dB", "phase (deg)");
	for (i = 0; i < 6; i++) {
		double w = discrete ? wd[i] : wc[i];
		double complex h = xq_ceval(H, discrete ? cexp(I * w) : I * w);
		double mag = cabs(h), ph = carg(h) * 180 / M_PI;
		if (fabs(ph) < 1e-9)
			ph = 0;
		if (!isfinite(mag) || isnan(mag))
			printf("    %-10s %-12s %-10s %s\n", xf_num(w), "inf", "inf",
			    "(a pole)");
		else
			printf("    %-10s %-12.6g %-10.4g %.4g\n", xf_num(w), mag,
			    mag > 0 ? 20 * log10(mag) : -INFINITY, ph);
	}
	/* -3 dB point when the gain falls monotonically from w = 0 */
	{
		double lo = 0, hi = discrete ? M_PI : wc[5] * 10;
		double g0 = cabs(xq_ceval(H, discrete ? 1 : 0)), it;
		double gh = cabs(xq_ceval(H, discrete ? cexp(I * hi) : I * hi));
		if (g0 > 0 && isfinite(g0) && gh < g0 / sqrt(2)) {
			for (it = 0; it < 80; it++) {
				double mid = (lo + hi) / 2;
				double g = cabs(xq_ceval(H, discrete ? cexp(I * mid) :
				    I * mid));
				if (g > g0 / sqrt(2))
					lo = mid;
				else
					hi = mid;
			}
			printf("  -3 dB (gain %s/sqrt(2)) at w = %.6g%s\n", xf_num(g0), lo,
			    discrete ? " rad/sample" : " rad/s");
			if (discrete)
				printf("    = %.6g of the sample rate\n", lo / (2 * M_PI));
		}
	}
}

/* y[n] = sum b_i x[n-i] - sum a_i y[n-i] as C */
static void
c_filter(const double *b, int nb, const double *a, int na)
{
	struct sbuf e = {0};
	int i;

	for (i = 0; i < nb; i++) {
		if (b[i] == 0)
			continue;
		if (e.n)
			sb_puts(&e, b[i] < 0 ? " - " : " + ");
		else if (b[i] < 0)
			sb_putc(&e, '-');
		if (fabs(b[i]) != 1)
			sb_printf(&e, "%.9gf*", fabs(b[i]));
		sb_printf(&e, i ? "x%d" : "x", i);
	}
	for (i = 1; i < na; i++) {
		if (a[i] == 0)
			continue;
		sb_puts(&e, -a[i] < 0 ? " - " : " + ");
		if (fabs(a[i]) != 1)
			sb_printf(&e, "%.9gf*", fabs(a[i]));
		sb_printf(&e, "y%d", i);
	}
	printf("  as C (direct form I; x1 = x[n-1], y1 = y[n-1], ...):\n");
	printf("    float y = %s;\n", e.n ? e.p : "0");
	if (nb > 1 || na > 1) {
		printf("   ");
		for (i = nb - 1; i >= 1; i--)
			printf(" x%d = %s;", i, i == 1 ? "x" : (char[8]){'x',
			    (char)('0' + i - 1), 0});
		for (i = na - 1; i >= 1; i--)
			printf(" y%d = %s;", i, i == 1 ? "y" : (char[8]){'y',
			    (char)('0' + i - 1), 0});
		printf("\n");
	}
	free(e.p);
}

static void
tf_analyse(struct xrat *H, int discrete, const char *vn, struct func *f)
{
	struct xpf pf;
	double complex z[XD];
	int m[XD], nz, i;
	double maxp = -INFINITY;
	nid v = xv(f ? f->np : 0, vn), Hn;
	int vs = f ? f->np : 0;

	xq_reduce(H);
	Hn = xq_node(H, v);
	printf("  H(%s) = %s\n", vn, xf_str(Hn));
	if (discrete && H->d.deg > 0) {
		struct sbuf b = {0};
		zinv_text(&b, H);
		printf("       = %s\n", b.p);
		free(b.p);
	}
	nz = xf_roots(&H->d, z, m);
	if (nz > 0) {
		struct sbuf b = {0};
		if (xf_apart(H, &pf))
			poles_text(&b, &pf, vn);
		printf("  poles: %s\n", b.p ? b.p : "none");
		free(b.p);
	} else
		printf("  poles: none\n");
	for (i = 0; i < nz; i++)
		maxp = fmax(maxp, discrete ? cabs(z[i]) : creal(z[i]));
	{
		double complex zz[XD];
		int mm[XD], k = H->n.deg > 0 ? xf_roots(&H->n, zz, mm) : 0;
		struct sbuf b = {0};
		for (i = 0; i < k; i++) {
			if (cimag(zz[i]) == 0)
				sb_printf(&b, "%s%s", i ? ", " : "", xf_num(creal(zz[i])));
			else
				sb_printf(&b, "%s%s +- %s", i ? ", " : "",
				    xf_num(creal(zz[i])), xf_cnum(I * cimag(zz[i])));
			if (mm[i] > 1)
				sb_printf(&b, " (x%d)", mm[i]);
		}
		printf("  zeros: %s\n", k > 0 ? b.p : "none");
		free(b.p);
	}
	if (discrete) {
		if (nz <= 0)
			printf("  stable: FIR (no poles but at 0)\n");
		else if (maxp < 1 - 1e-12)
			printf("  stable (BIBO): every pole inside the unit circle "
			    "(max |p| = %s)\n", xf_num(maxp));
		else if (maxp <= 1 + 1e-12)
			printf("  marginally stable at best: a pole on the unit circle "
			    "(|p| = 1): bounded input can give unbounded output\n");
		else
			printf("  UNSTABLE: a pole outside the unit circle (|p| = %s)\n",
			    xf_num(maxp));
		printf("  causal: %s\n", H->n.deg <= H->d.deg ? "yes (deg numerator "
		    "<= deg denominator)" : "no (it needs future inputs)");
	} else {
		if (nz <= 0)
			printf("  stable: no poles\n");
		else if (maxp < -1e-12)
			printf("  stable (BIBO): every pole in the left half-plane "
			    "(max Re(p) = %s)\n", xf_num(maxp));
		else if (maxp <= 1e-12)
			printf("  marginally stable at best: a pole on the imaginary "
			    "axis\n");
		else
			printf("  UNSTABLE: a pole in the right half-plane (Re(p) = "
			    "%s)\n", xf_num(maxp));
		printf("  proper: %s\n", H->n.deg <= H->d.deg ? "yes" : "no");
	}
	/* impulse response */
	{
		jmp_buf saved;
		memcpy(saved, fail_jmp, sizeof saved);
		if (setjmp(fail_jmp) == 0) {
			if (discrete) {
				struct xrat r2;
				double R;
				nid n = xv(vs + 1, "n"), h;
				xf_discrete = 1;
				h = iz_rational(Hn, vs, n, &r2, NULL, &R);
				if (h)
					printf("  impulse response h[n] = %s    for n >= 0\n",
					    xf_str(h));
				xf_discrete = 0;
			} else if (H->n.deg < H->d.deg || H->n.deg == H->d.deg) {
				double sg;
				int nd;
				nid t = xv(vs + 1, "t"), h;
				h = ilap_node(Hn, vs, t, vs + 1, &sg, NULL, NULL, &nd);
				printf("  impulse response h(t) = %s    for t >= 0\n",
				    xf_str(h));
			}
		}
		memcpy(fail_jmp, saved, sizeof saved);
		xf_discrete = 0;
	}
	{
		double complex g = xq_ceval(H, discrete ? 1 : 0);
		if (isfinite(creal(g)))
			printf("  DC gain H(%s) = %s\n", discrete ? "1" : "0",
			    xf_cnum(g));
	}
	freq_table(H, discrete);
	if (discrete && H->n.deg <= H->d.deg && H->d.deg <= 8) {
		double b[XD + 1], a[XD + 1];
		int dd = H->d.deg;
		for (i = 0; i <= dd; i++) {
			a[i] = H->d.c[dd - i] / H->d.c[dd];
			b[i] = dd - i <= H->n.deg ? H->n.c[dd - i] / H->d.c[dd] : 0;
		}
		{
			int nb = dd + 1, na = dd + 1;
			while (nb > 1 && b[nb - 1] == 0)
				nb--;
			while (na > 1 && a[na - 1] == 0)
				na--;
			c_filter(b, nb, a, na);
		}
	}
}

static void
cmd_tf(char *text)
{
	struct xrat H;
	int discrete;

	if (strchr(text, '=') && !strstr(text, "==")) {
		char names[4][32], out[32] = "", in[32] = "";
		int nn, k, kminy, kmaxy, kminx, kmaxx;
		struct sbuf t = {0}, o1 = {0}, o2 = {0};
		char *eq = xstrdup(text), *p;
		struct func *f;
		struct xpoly A, B;
		int lo;
		discrete = strchr(text, '[') != NULL;
		nn = seq_names(text, discrete, names);
		for (k = 0; k < nn; k++) {
			if (!strcmp(names[k], "y"))
				snprintf(out, sizeof out, "y");
			if (!strcmp(names[k], "x") || !strcmp(names[k], "u"))
				snprintf(in, sizeof in, "%s", names[k]);
		}
		if (!discrete && !in[0]) {
			static const char *const ins[] = {"x", "u", "f", "r", NULL};
			for (k = 0; ins[k]; k++)
				if (has_ident(text, ins[k]) && strcmp(ins[k], out)) {
					snprintf(in, sizeof in, "%s", ins[k]);
					break;
				}
		}
		for (k = 0; k < nn && (!out[0] || !in[0]); k++) {
			if (!out[0] && strcmp(names[k], in))
				snprintf(out, sizeof out, "%s", names[k]);
			else if (!in[0] && strcmp(names[k], out))
				snprintf(in, sizeof in, "%s", names[k]);
		}
		if (!out[0] || !in[0])
			fail("tf: expected an output and an input, y[n] = 0.9y[n-1] + "
			    "x[n] or y'' + y' = x");
		p = strchr(eq, '=');
		*p = 0;
		sb_printf(&t, "(%s) - (%s)", eq, p + 1);
		upre = 'd';
		uxname = in;
		unknowns(t.p, &o1, out, sizeof out, discrete, discrete ? "n" : "t",
		    &kminy, &kmaxy, in);
		upre = 'e';
		uxname = "";
		unknowns(o1.p, &o2, in, sizeof in, discrete, discrete ? "n" : "t",
		    &kminx, &kmaxx, "");
		upre = 'd';
		uxname = "x";
		f = xf_parse(o2.p);
		xp_zero(&A);
		xp_zero(&B);
		lo = kminy < kminx ? kminy : kminx;
		if (kmaxy < kminy)
			fail("tf: %s does not occur", out);
		if (kmaxx < kminx)
			fail("tf: %s does not occur", in);
		if ((kmaxy > kmaxx ? kmaxy : kmaxx) - lo > XD)
			fail("tf: order too high");
		for (k = 0; k < f->np; k++) {
			const char *nm = sym_str(f->p[k].name);
			double v;
			int off;
			nid d;
			if (strncmp(nm, "__", 2) || (nm[2] != 'd' && nm[2] != 'e'))
				fail("tf: %s is neither %s nor %s", nm, out, in);
			off = atoi(nm + 3) - (discrete ? 16 : 0);
			d = diff_node(f->body, (uint32_t)k);
			if (!xnum(d, &v))
				fail("tf: not linear with constant coefficients (%s)",
				    xf_str(d));
			if (nm[2] == 'd')
				A.c[off - lo] += v;
			else
				B.c[off - lo] -= v;
		}
		A.deg = B.deg = XD;
		xp_trim(&A);
		xp_trim(&B);
		H.n = B;
		H.d = A;
		printf("transfer function of %s (zero initial state)\n", text);
		printf("  input %s, output %s: H(%s) = %c(%s)/%c(%s)\n", in, out,
		    discrete ? "z" : "s", toupper((unsigned char)out[0]),
		    discrete ? "z" : "s", toupper((unsigned char)in[0]),
		    discrete ? "z" : "s");
		tf_analyse(&H, discrete, discrete ? "z" : "s", NULL);
		free(eq);
		free(t.p);
		free(o1.p);
		free(o2.p);
		return;
	}
	{
		struct func *f = xf_parse(text);
		int zs = xf_slot(f, "z"), ss = xf_slot(f, "s"), vs;
		if (zs >= 0)
			discrete = 1, vs = zs;
		else if (ss >= 0)
			discrete = 0, vs = ss;
		else if (f->np == 1)
			discrete = 0, vs = 0;
		else
			fail("tf: H should be a function of z (discrete) or s");
		if (f->np > 1)
			fail("tf: H has symbols besides %s", sym_str(f->p[vs].name));
		if (!xq_of(f->body, vs, &H))
			fail("tf: %s is not a rational function", xf_str(f->body));
		printf("transfer function %s\n", fname_of(f));
		tf_analyse(&H, discrete, sym_str(f->p[vs].name), f);
	}
}

/* ---- Fourier series ---- */

#define GLN 20
static double glx[GLN], glw[GLN];

static void
gl_init(void)
{
	int i, j, it;

	if (glw[0] != 0)
		return;
	for (i = 0; i < GLN; i++) {
		double x = cos(M_PI * (i + 0.75) / (GLN + 0.5)), p0, p1, dp = 1;
		for (it = 0; it < 100; it++) {
			double dx;
			p0 = 1;
			p1 = x;
			for (j = 2; j <= GLN; j++) {
				double p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
				p0 = p1;
				p1 = p2;
			}
			dp = GLN * (x * p1 - p0) / (x * x - 1);
			dx = p1 / dp;
			x -= dx;
			if (fabs(dx) < 1e-16)
				break;
		}
		glx[i] = x;
		glw[i] = 2 / ((1 - x * x) * dp * dp);
	}
}

/* quadrature nodes and weights on [a, b] split at the points br[] */
struct quad {
	int n;
	double *x, *w, *f;
	uint8_t *ok;
};

static void
quad_make(struct quad *Q, nid e, int slot, double a, double b,
    const double *br, int nbr, int panels)
{
	double cut[80];
	int nc = 0, i, j, k;

	gl_init();
	cut[nc++] = a;
	for (i = 0; i < nbr && nc < 78; i++)
		if (br[i] > a + 1e-12 * (b - a) && br[i] < b - 1e-12 * (b - a))
			cut[nc++] = br[i];
	cut[nc++] = b;
	/* sort */
	for (i = 1; i < nc; i++)
		for (j = i; j > 0 && cut[j] < cut[j - 1]; j--) {
			double t = cut[j];
			cut[j] = cut[j - 1];
			cut[j - 1] = t;
		}
	Q->n = 0;
	Q->x = xmalloc((size_t)(panels + nc) * GLN * sizeof *Q->x);
	Q->w = xmalloc((size_t)(panels + nc) * GLN * sizeof *Q->w);
	for (i = 0; i + 1 < nc; i++) {
		double L = cut[i + 1] - cut[i];
		int m = (int)ceil(panels * L / (b - a));
		if (m < 1)
			m = 1;
		for (k = 0; k < m; k++) {
			double p = cut[i] + L * k / m, q = cut[i] + L * (k + 1) / m;
			for (j = 0; j < GLN; j++) {
				Q->x[Q->n] = (p + q) / 2 + (q - p) / 2 * glx[j];
				Q->w[Q->n++] = (q - p) / 2 * glw[j];
			}
		}
	}
	Q->f = xmalloc((size_t)Q->n * sizeof *Q->f);
	Q->ok = xmalloc((size_t)Q->n);
	xf_evalv(e, slot, Q->x, Q->n, NULL, NULL, 0, Q->f, Q->ok);
	for (i = 0; i < Q->n; i++)
		if (!Q->ok[i] || !isfinite(Q->f[i]))
			Q->f[i] = 0;
}

static void
quad_free(struct quad *Q)
{
	free(Q->x);
	free(Q->w);
	free(Q->f);
	free(Q->ok);
}

/* c_n ~ sum_k (A_k + B_k (-1)^n)/n^k, k = 1..4, with exact coefficients */
static nid
coef_form(const double *c, int nmax, nid n, double scale)
{
	double M[48 * 8], rhs[48], sol[8], snapv[8];
	int m = 0, i, k;
	nid r = xk(0);

	for (i = 1; i <= 24 && i <= nmax; i++, m++) {
		for (k = 0; k < 4; k++) {
			M[m * 8 + 2 * k] = pow(i, -(k + 1));
			M[m * 8 + 2 * k + 1] = (i & 1 ? -1 : 1) * pow(i, -(k + 1));
		}
		rhs[m] = c[i];
	}
	if (!solve_lsq(M, rhs, m, 8, sol))
		return 0;
	for (k = 0; k < 8; k++) {
		const char *t;
		if (fabs(sol[k]) <= 1e-9 * fmax(scale, 1e-300)) {
			snapv[k] = 0;
			continue;
		}
		t = xf_num(sol[k]);
		if (strchr(t, 'e') && !strstr(t, "e^") && isdigit((unsigned char)t[0]))
			return 0;   /* 1.234e-05 */
		if (strchr(t, '.'))
			return 0;
		snapv[k] = sol[k];
		/* re-read the exact value */
		{
			struct func *f = xf_parse(t);
			double v;
			if (!xnum(f->body, &v))
				return 0;
			snapv[k] = v;
		}
	}
	for (i = 1; i <= nmax; i++) {
		double v = 0;
		for (k = 0; k < 4; k++)
			v += (snapv[2 * k] + (i & 1 ? -1 : 1) * snapv[2 * k + 1]) *
			    pow(i, -(k + 1));
		if (fabs(v - c[i]) > 1e-8 * fmax(scale, 1e-300))
			return 0;
	}
	for (k = 0; k < 4; k++) {
		double A = snapv[2 * k], B = snapv[2 * k + 1];
		nid num, pm = xpow(xk(-1), n);
		if (A == 0 && B == 0)
			continue;
		if (A == 0)
			num = xmul(xk(B), pm);
		else if (B == 0)
			num = xk(A);
		else if (fabs(A + B) <= 1e-12 * fabs(A))
			num = xmul(xk(A), xsub(xk(1), pm));
		else if (fabs(A - B) <= 1e-12 * fabs(A))
			num = xmul(xk(A), xadd(xk(1), pm));
		else
			num = xadd(xk(A), xmul(xk(B), pm));
		r = xadd(r, xdiv(num, k == 0 ? n : xpow(n, xk(k + 1))));
	}
	return r;
}

#define FSN 48

/* a one-sided limit at x0 from the side of h: 2 f(x0 + h) - f(x0 + 2h) */
static double
side_lim(nid e, int slot, double x0, double h, int *ok)
{
	int ok2;
	double a = xf_eval1(e, slot, x0 + h, ok);
	double b = xf_eval1(e, slot, x0 + 2 * h, &ok2);

	if (!ok2)
		return a;
	{
		double v = 2 * a - b;
		struct q q;
		if (q_from_double(v, &q) && q.d <= 1000)
			v = q_val(q);
		return v;
	}
}

static void
cmd_fseries(char *text)
{
	static const char *const opts[] = {"on", "over", "from", "period",
		"terms", NULL};
	static const char *const xpref[] = {"x", "t", "theta", NULL};
	char xv_[32] = "", dummy[32] = "", on[96] = "", per[64] = "",
	    terms[32] = "";
	struct func *f;
	int xs, i, nshow = 5, odd, even, nb = 0, kinds[64];
	double a = -M_PI, b = M_PI, P, w, an[FSN + 1], bn[FSN + 1], scale = 0,
	    br[64];
	struct quad Q;
	nid x, nn;

	if (!take_opt(text, "on", opts, on, sizeof on))
		take_opt(text, "over", opts, on, sizeof on);
	if (!on[0] && take_opt(text, "from", opts, on, sizeof on)) {
		/* from a to b */
		static const char *const to[] = {"to", NULL};
		long at = find_word(on, to, 0);
		if (at > 0) {
			on[at] = '.';
			on[at + 1] = '.';
		}
	}
	take_opt(text, "period", opts, per, sizeof per);
	take_opt(text, "terms", opts, terms, sizeof terms);
	{
		static const char *const outs[] = {NULL};
		trail_vars(text, xv_, sizeof xv_, dummy, sizeof dummy, outs);
	}
	if (on[0]) {
		char *dd = strstr(on, ".."), *l = on, *r;
		struct func *fa, *fb2;
		if (!dd) {
			/* a to b */
			static const char *const to[] = {"to", NULL};
			long at2 = find_word(on, to, 0);
			if (at2 > 0) {
				on[at2] = '.';
				on[at2 + 1] = '.';
				dd = on + at2;
			}
		}
		if (!dd) {
			/* [a, b] */
			char *c = strchr(on, ',');
			if (!c)
				fail("fourier series: the interval is a..b or [a, b]");
			*c = 0;
			r = c + 1;
			if (*l == '[' || *l == '(')
				l++;
			if (r[strlen(r) - 1] == ']' || r[strlen(r) - 1] == ')')
				r[strlen(r) - 1] = 0;
		} else {
			*dd = 0;
			r = dd + 2;
		}
		fa = xf_parse(l);
		fb2 = xf_parse(r);
		if (!xnum(fa->body, &a) || !xnum(fb2->body, &b) || b <= a)
			fail("fourier series: the interval must be a..b with a < b");
	} else if (per[0]) {
		struct func *fp = xf_parse(per);
		if (!xnum(fp->body, &P) || P <= 0)
			fail("fourier series: the period must be positive");
		a = -P / 2;
		b = P / 2;
	}
	if (terms[0])
		nshow = atoi(terms) > 0 && atoi(terms) <= 20 ? atoi(terms) : 5;
	f = xf_parse(text);
	xs = pick_slot(f, xv_, xpref);
	if (xs < 0 && f->np > 0)
		fail("fourier series: which variable?  (fourier series F x)");
	if (f->np > 1)
		fail("fourier series: give the other parameters values");
	if (xs < 0)
		xs = 0;
	x = xv(xs, f->np ? sym_str(f->p[xs].name) : "x");
	nn = xv(xs + 1, "n");
	P = b - a;
	w = 2 * M_PI / P;
	if (f->np == 1) {
		struct func t = *f;
		t.p[0].ty = TY_REAL;
		nb = fn_discontinuities(&t, a, b, br, kinds, 64);
	}
	quad_make(&Q, f->body, xs, a, b, br, nb, 256);
	for (i = 0; i <= FSN; i++) {
		double sa = 0, sb = 0;
		int j;
		for (j = 0; j < Q.n; j++) {
			sa += Q.w[j] * Q.f[j] * cos(i * w * Q.x[j]);
			sb += Q.w[j] * Q.f[j] * sin(i * w * Q.x[j]);
		}
		an[i] = 2 / P * sa;
		bn[i] = 2 / P * sb;
		scale = fmax(scale, fmax(fabs(an[i]), fabs(bn[i])));
	}
	for (i = 0; i <= FSN; i++) {
		if (fabs(an[i]) < 1e-11 * scale)
			an[i] = 0;
		if (fabs(bn[i]) < 1e-11 * scale)
			bn[i] = 0;
	}
	odd = 1;
	even = 1;
	for (i = 0; i <= FSN; i++) {
		if (an[i] != 0)
			odd = 0;
		if (bn[i] != 0)
			even = 0;
	}
	printf("Fourier series of %s on [%s, %s] (period %s):\n", fname_of(f),
	    xf_num(a), xf_num(b), xf_num(P));
	{
		const char *wx = xf_str(xmul(xmul(xk(w), nn), x));
		printf("  f(%s) ~ a0/2 + sum_{n>=1} a_n cos(%s) + b_n sin(%s)\n",
		    sym_str(N.a[x]), wx, wx);
		printf("  a_n = %s int f(%s) cos(%s) d%s over the period, b_n "
		    "likewise with sin\n", xf_num(2 / P), sym_str(N.a[x]), wx,
		    sym_str(N.a[x]));
	}
	printf("  a0 = %s\n", xf_num(an[0]));
	{
		int last = 0;
		for (i = 1; i <= FSN; i++)
			if (an[i] != 0 || bn[i] != 0)
				last = i;
		if (last <= 12) {
			/* a trigonometric polynomial */
			printf("  finite: every coefficient after n = %d is 0\n", last);
			for (i = 1; i <= last; i++)
				if (an[i] != 0 || bn[i] != 0)
					printf("  a_%d = %s, b_%d = %s\n", i, xf_num(an[i]), i,
					    xf_num(bn[i]));
		} else {
			nid fa = odd ? xk(0) : coef_form(an, FSN, nn, scale);
			nid fb = even ? xk(0) : coef_form(bn, FSN, nn, scale);
			if (odd)
				printf("  a_n = 0 (f is odd)\n");
			else if (fa)
				printf("  a_n = %s\n", xf_str(fa));
			else
				printf("  a_n: no closed form recognised; a_1..a_4 = %s, %s, "
				    "%s, %s\n", xf_num(an[1]), xf_num(an[2]), xf_num(an[3]),
				    xf_num(an[4]));
			if (even)
				printf("  b_n = 0 (f is even)\n");
			else if (fb)
				printf("  b_n = %s\n", xf_str(fb));
			else
				printf("  b_n: no closed form recognised; b_1..b_4 = %s, %s, "
				    "%s, %s\n", xf_num(bn[1]), xf_num(bn[2]), xf_num(bn[3]),
				    xf_num(bn[4]));
			if ((fa || odd) && (fb || even))
				printf("  check: the closed forms equal the numeric "
				    "coefficients (Gauss-Legendre, %d nodes) for n = 1..%d\n",
				    Q.n, FSN);
		}
	}
	{
		/* the first terms */
		nid S = xk(0);
		int shown = 0;
		for (i = 1; i <= FSN && shown < nshow; i++) {
			nid arg = xmul(xk(i * w), x);
			if (an[i] != 0)
				S = xadd(S, xmul(xk(an[i]), mk_un(O_COS, arg)));
			if (bn[i] != 0)
				S = xadd(S, xmul(xk(bn[i]), mk_un(O_SIN, arg)));
			if (an[i] != 0 || bn[i] != 0)
				shown++;
		}
		{
			const char *st = xf_str(S);
			if (an[0] != 0 && !(xnum(S, &(double){0}) && shown == 0))
				printf("  f(%s) ~ %s %s %s%s\n", sym_str(N.a[x]),
				    xf_num(an[0] / 2), st[0] == '-' ? "-" : "+",
				    st[0] == '-' ? st + 1 : st, shown == nshow ? " + ..." : "");
			else
				printf("  f(%s) ~ %s%s\n", sym_str(N.a[x]), shown ? st :
				    xf_num(an[0] / 2), shown == nshow ? " + ..." : "");
		}
	}
	{
		/* L2 error of partial sums */
		int Ns[2] = {5, FSN}, k2, j;
		double nf = 0;
		for (j = 0; j < Q.n; j++)
			nf += Q.w[j] * Q.f[j] * Q.f[j];
		printf("  partial sums S_N: ||f - S_N||/||f|| =");
		for (k2 = 0; k2 < 2; k2++) {
			double e2 = 0;
			for (j = 0; j < Q.n; j++) {
				double sv = an[0] / 2, d;
				for (i = 1; i <= Ns[k2]; i++)
					sv += an[i] * cos(i * w * Q.x[j]) + bn[i] * sin(i * w *
					    Q.x[j]);
				d = Q.f[j] - sv;
				e2 += Q.w[j] * d * d;
			}
			printf("%s %.3g (N = %d)", k2 ? "," : "", nf > 0 ?
			    sqrt(e2 / nf) : 0, Ns[k2]);
		}
		printf("\n");
	}
	{
		/* jumps of the periodic extension and the Gibbs overshoot */
		double jx[66], jsz[66];
		int nj = 0, ok1, ok2;
		double fa = side_lim(f->body, xs, a, 1e-7 * P, &ok1);
		double fb = side_lim(f->body, xs, b, -1e-7 * P, &ok2);
		for (i = 0; i < nb; i++)
			if (kinds[i] == 2) {
				double l = side_lim(f->body, xs, br[i], -1e-7 * P, &ok1);
				double r = side_lim(f->body, xs, br[i], 1e-7 * P, &ok2);
				jx[nj] = br[i];
				jsz[nj++] = r - l;
			}
		if (ok1 && ok2 && fabs(fa - fb) > 1e-6 * fmax(1, scale)) {
			jx[nj] = b;
			jsz[nj++] = fa - fb;
		}
		if (nj == 0)
			printf("  f is continuous as a periodic function: S_N -> f "
			    "uniformly (no Gibbs overshoot)\n");
		else {
			double x0 = jx[0], J = jsz[0], mx = -INFINITY, at = x0, h;
			int Nn = FSN, j;
			double fl = side_lim(f->body, xs, x0, -1e-7 * P, &ok1);
			double fr = x0 >= b ? fa : side_lim(f->body, xs, x0, 1e-7 * P,
			    &ok2);
			if (x0 >= b)
				fl = fb;
			printf("  converges to f where f is continuous, and to the "
			    "midpoint of the jump where it jumps:\n");
			for (j = 0; j < nj && j < 4; j++)
				printf("    at %s = %s it jumps by %s: S_N -> %s\n",
				    sym_str(N.a[x]), xf_num(jx[j]), xf_num(jsz[j]),
				    j == 0 ? xf_num((fl + fr) / 2) : "the midpoint");
			/* overshoot of S_48 on the high side of the first jump */
			{
				double hiv, side, top;
				if (x0 >= b) {
					fl = fb;
					fr = fa;
				}
				side = fr > fl ? 1 : -1;
				top = fmax(fl, fr);
				hiv = -INFINITY;
				h = P / (4.0 * Nn);
				for (j = 1; j <= 400; j++) {
					double xx = x0 + side * h * j / 100.0, sv;
					sv = an[0] / 2;
					for (i = 1; i <= Nn; i++)
						sv += an[i] * cos(i * w * xx) + bn[i] * sin(i * w * xx);
					if (sv > hiv) {
						hiv = sv;
						at = xx;
					}
				}
				mx = hiv;
				printf("  Gibbs: near %s = %s, S_%d overshoots by %.4g = %.1f%% "
				    "of the jump (at %s = %.4g); the overshoot tends to "
				    "8.95%% of the jump and does not shrink as N grows\n",
				    sym_str(N.a[x]), xf_num(x0), Nn, mx - top,
				    100 * (mx - top) / fabs(J), sym_str(N.a[x]), at);
			}
		}
	}
	quad_free(&Q);
}

/* ---- the continuous Fourier transform ---- */

struct cpair {
	nid re, im;
};

static struct cpair
cp(nid re, nid im)
{
	struct cpair r;

	r.re = re;
	r.im = im;
	return r;
}

static struct cpair
cp_add(struct cpair a, struct cpair b)
{
	return cp(xadd(a.re, b.re), xadd(a.im, b.im));
}

static struct cpair
cp_scale(struct cpair a, nid c)
{
	return cp(xmul(c, a.re), xmul(c, a.im));
}

/* times i */
static struct cpair
cp_i(struct cpair a)
{
	return cp(xneg(a.im), a.re);
}

static struct cpair
cp_sub_w(struct cpair a, int ws, nid w, nid by)
{
	return cp(xsubst1(a.re, ws, xsub(w, by)), xsubst1(a.im, ws, xsub(w, by)));
}

struct fctx {
	int xs, ws;
	nid x, w;
	int ndelta;
	int depth;
};

static struct cpair ft_rec(struct fctx *C, nid e);

/* an interval [c, d] where cond holds (c may be -inf, d inf) */
static int
cond_interval(nid c, int slot, double *lo, double *hi)
{
	int op;
	double c1, c0, a;

	c = unc(c);
	op = N.op[c];
	if (op == O_LAND) {
		double l1, h1, l2, h2;
		if (!cond_interval(N.a[c], slot, &l1, &h1) ||
		    !cond_interval(N.b[c], slot, &l2, &h2))
			return 0;
		*lo = fmax(l1, l2);
		*hi = fmin(h1, h2);
		return 1;
	}
	if (!(opinfo[op].fl & OF_CMP) || op == O_EQ || op == O_NE)
		return 0;
	{
		nid L = unc(N.a[c]), R = unc(N.b[c]);
		/* |x| < a */
		if (N.op[L] == O_ABS && is_var(N.a[L], slot) && xnum(R, &a) &&
		    (op == O_LT || op == O_LE)) {
			*lo = -a;
			*hi = a;
			return 1;
		}
	}
	if (!lin_num(xsub(N.a[c], N.b[c]), slot, &c1, &c0))
		return 0;
	a = -c0 / c1;
	if (c1 < 0)
		op = op == O_GE ? O_LE : op == O_GT ? O_LT : op == O_LE ? O_GE :
		    O_GT;
	if (op == O_GE || op == O_GT) {
		*lo = a;
		*hi = INFINITY;
	} else {
		*lo = -INFINITY;
		*hi = a;
	}
	return 1;
}

static struct cpair
ft_term(struct fctx *C, nid term)
{
	struct facs F;
	double pw = 0, v, c1, c0;
	nid u, arg, w = C->w;
	int k;

	memset(&F, 0, sizeof F);
	F.coef = xk(1);
	fac_rec(term, C->xs, 1, &F);
	if (F.nf == 0) {
		/* a constant: 2 pi c delta(w) */
		C->ndelta++;
		return cp(xmul(xmul(xk(2 * M_PI), F.coef), xdirac(w)), xk(0));
	}
	/* x^k: i^k d^k/dw^k */
	for (k = 0; k < F.nf; k++)
		if (is_var(F.f[k], C->xs) && F.pw[k] > 0 && F.pw[k] == floor(F.pw[k]))
			break;
	if (k < F.nf) {
		int m = (int)F.pw[k];
		struct cpair G = ft_rec(C, facs_except(&F, 1u << k));
		int j;
		for (j = 0; j < m; j++)
			G = cp_i(cp(diff_node(G.re, (uint32_t)C->ws),
			    diff_node(G.im, (uint32_t)C->ws)));
		return G;
	}
	if (F.nf == 2) {
		/* sin(a x)/x */
			for (k = 0; k < 2; k++) {
				nid f = unc(F.f[k]), g = unc(F.f[1 - k]);
				if (N.op[f] == O_SIN && F.pw[k] == 1 && is_var(g, C->xs) &&
				    F.pw[1 - k] == -1 && lin_of(N.a[f], C->xs, &u, &arg) &&
				    xnum(arg, &v) && v == 0) {
					/* pi for |w| < a */
					nid a = mk_un(O_ABS, u);
					return cp(xmul(F.coef, mk_sel(mk_bin(O_LT, mk_un(O_ABS, w),
					    a), xk(M_PI), xk(0))), xk(0));
				}
			}
	}
	/* modulation: cos(bx) g, sin(bx) g */
	for (k = 0; k < F.nf; k++) {
		nid f = unc(F.f[k]);
		if ((N.op[f] == O_COS || N.op[f] == O_SIN) && F.pw[k] == 1 &&
		    lin_of(N.a[f], C->xs, &u, &arg) && xnum(arg, &v) && v == 0 &&
		    F.nf > 1) {
			struct cpair G = ft_rec(C, facs_except(&F, 1u << k));
			struct cpair A = cp_sub_w(G, C->ws, w, u),
			    B = cp_sub_w(G, C->ws, w, xneg(u));
			if (N.op[f] == O_COS)
				return cp_scale(cp_add(A, B), xk(0.5));
			/* (A - B)/(2i) = -i (A - B)/2 */
			return cp_scale(cp_i(cp(xsub(A.re, B.re), xsub(A.im, B.im))),
			    xk(-0.5));
		}
	}
	if (F.nf == 1) {
		nid f = unc(F.f[0]);
		double p = F.pw[0];
		/* e^(q(x)): gaussian, e^(-a|x|) */
		if (xis_exp(f, &u) && p == 1) {
			struct npoly q;
			if (npoly_of(u, C->xs, &q, 2) && q.deg == 2 &&
			    xnum(xadd(xk(0), q.c[1]), &v) && v == 0) {
				nid A = xneg(xadd(xk(0), q.c[2]));   /* e^(-A x^2) */
				nid c = xmul(F.coef, xexp(xadd(xk(0), q.c[0])));
				return cp(xmul(c, xmul(mk_un(O_SQRT, xdiv(xk(M_PI), A)),
				    xexp(xneg(xdiv(xpow(w, xk(2)), xmul(xk(4), A)))))),
				    xk(0));
			}
			{
				/* -a |x| */
				struct facs G;
				memset(&G, 0, sizeof G);
				G.coef = xk(1);
				fac_rec(u, C->xs, 1, &G);
				if (G.nf == 1 && G.pw[0] == 1 && N.op[unc(G.f[0])] == O_ABS &&
				    is_var(N.a[unc(G.f[0])], C->xs)) {
					nid a = xneg(G.coef);
					return cp(xmul(F.coef, xdiv(xmul(xk(2), a),
					    xadd(xpow(a, xk(2)), xpow(w, xk(2))))), xk(0));
				}
			}
		}
		/* 1/(x^2 + a^2), x/(x^2 + a^2) */
		if (p == -1) {
			struct npoly q;
			if (npoly_of(f, C->xs, &q, 2) && q.deg == 2 &&
			    xnum(xadd(xk(0), q.c[1]), &v) && v == 0) {
				nid A = xadd(xk(0), q.c[2]), B = xadd(xk(0), q.c[0]);
				nid a = mk_un(O_SQRT, xdiv(B, A));
				return cp(xmul(xdiv(F.coef, A), xmul(xdiv(xk(M_PI), a),
				    xexp(xneg(xmul(a, mk_un(O_ABS, w)))))), xk(0));
			}
		}
		/* sin(a x)/x */
		if (N.op[f] == O_SIN && p == 1) {
			/* only with a 1/x factor: handled below */
		}
		/* delta(x - c) */
		if (xis_dirac(f, &arg) && p == 1 && lin_num(arg, C->xs, &c1, &c0)) {
			double c = -c0 / c1;
			nid cw = xmul(xk(c), w);
			return cp(xmul(xdiv(F.coef, xk(fabs(c1))), mk_un(O_COS, cw)),
			    xneg(xmul(xdiv(F.coef, xk(fabs(c1))), mk_un(O_SIN, cw))));
		}
		/* a box: [c, d] */
		if (N.op[f] == O_SEL && p == 1 && xnum(N.c[f], &v) && v == 0 &&
		    !has_slot(N.b[f], C->xs)) {
			double lo, hi;
			if (cond_interval(N.a[f], C->xs, &lo, &hi) && isfinite(lo) &&
			    isfinite(hi)) {
				nid c = xmul(F.coef, N.b[f]);
				if (lo == -hi)
					return cp(xmul(c, xdiv(xmul(xk(2), mk_un(O_SIN,
					    xmul(xk(hi), w))), w)), xk(0));
				/* (e^(-i w lo) - e^(-i w hi))/(i w) */
				nid re = xdiv(xsub(mk_un(O_SIN, xmul(xk(hi), w)),
				    mk_un(O_SIN, xmul(xk(lo), w))), w);
				nid im = xdiv(xsub(mk_un(O_COS, xmul(xk(hi), w)),
				    mk_un(O_COS, xmul(xk(lo), w))), w);
				return cp(xmul(c, re), xmul(c, im));
			}
		}
		/* cos(b x), sin(b x) alone: deltas */
		if ((N.op[f] == O_COS || N.op[f] == O_SIN) && p == 1 &&
		    lin_of(N.a[f], C->xs, &u, &arg) && xnum(arg, &v) && v == 0) {
			nid d1 = xdirac(xsub(w, u)), d2 = xdirac(xadd(w, u));
			C->ndelta++;
			if (N.op[f] == O_COS)
				return cp(xmul(xmul(F.coef, xk(M_PI)), xadd(d1, d2)), xk(0));
			return cp(xk(0), xmul(xmul(F.coef, xk(M_PI)), xsub(d2, d1)));
		}
	}
	if (F.nf == 2) {
		/* e^(-a x) u(x) */
		int ke = -1, kh = -1;
		for (k = 0; k < 2; k++) {
			nid f = unc(F.f[k]);
			if (xis_exp(f, &u) && F.pw[k] == 1)
				ke = k;
			if (xis_heav(f, &arg) && F.pw[k] == 1)
				kh = k;
		}
		if (ke >= 0 && kh >= 0) {
			double h1, h0;
			nid a, b0;
			xis_exp(unc(F.f[ke]), &u);
			xis_heav(unc(F.f[kh]), &arg);
			if (lin_of(u, C->xs, &a, &b0) && lin_num(arg, C->xs, &h1, &h0) &&
			    h0 == 0) {
				/* e^(a x + b0): a < 0 with u(x), a > 0 with u(-x) */
				nid c = xmul(F.coef, xexp(b0)), al = h1 > 0 ? xneg(a) : a;
				nid den = xadd(xpow(al, xk(2)), xpow(w, xk(2)));
				/* u(x): 1/(al + i w); u(-x): 1/(al - i w) */
				return cp(xmul(c, xdiv(al, den)), xmul(c, xdiv(h1 > 0 ?
				    xneg(w) : w, den)));
			}
		}
	}
	fail("fourier: no rule for %s (e^(-ax^2), e^(-a|x|), e^(-ax) u(x), "
	    "1/(x^2 + a^2), boxes, sin(ax)/x, delta, x^n, cos/sin modulation)",
	    xf_str(term));
	return cp(0, 0);
}

static struct cpair
ft_rec(struct fctx *C, nid e)
{
	struct terms T;
	struct cpair r = cp(xk(0), xk(0));
	int i;

	if (++C->depth > 30)
		fail("fourier: too deep");
	terms_of(e, &T);
	for (i = 0; i < T.n; i++)
		r = cp_add(r, ft_term(C, T.t[i]));
	C->depth--;
	return r;
}

/* a top-level + or - in printed text (not inside parentheses) */
static int
top_sum(const char *s)
{
	int depth = 0;
	size_t i;

	for (i = 1; s[i]; i++) {
		if (s[i] == '(' || s[i] == '[')
			depth++;
		else if (s[i] == ')' || s[i] == ']')
			depth--;
		else if (depth == 0 && s[i - 1] == ' ' && (s[i] == '+' ||
		    s[i] == '-') && s[i + 1] == ' ')
			return 1;
	}
	return 0;
}

static const char *
cp_str(struct cpair z)
{
	double v;
	int rz = xnum(z.re, &v) && v == 0, iz = xnum(z.im, &v) && v == 0;
	static struct sbuf b[4];
	static int k;
	struct sbuf *o = &b[k++ & 3];

	sb_clear(o);
	if (iz)
		sb_puts(o, xf_str(z.re));
	else {
		const char *is = xf_str(z.im);
		int sum = top_sum(is);
		int neg = is[0] == '-' && !sum;
		const char *body = neg ? is + 1 : is;
		int paren = sum;
		if (!rz)
			sb_printf(o, "%s %c ", xf_str(z.re), neg ? '-' : '+');
		else if (neg)
			sb_putc(o, '-');
		sb_printf(o, paren ? "i*(%s)" : "i*%s", body);
	}
	return sb_str(o);
}

/* int f(x) e^(-iwx) dx numerically over [-X, X]: re and im */
static void
ft_numeric_on(nid e, int xs, double w0, double X, double *re, double *im)
{
	int j;
	struct quad Q;
	double brk[66];
	int kinds[64], nb = 0, panels = (int)fmin(60000, 2 * X * fmax(2, w0));
	struct func t;

	memset(&t, 0, sizeof t);
	t.np = xs + 1;
	t.p[xs].ty = TY_REAL;
	t.body = e;
	if (xs == 0 && X <= 64)
		nb = fn_discontinuities(&t, -X, X, brk, kinds, 64);
	brk[nb++] = 0;
	quad_make(&Q, e, xs, -X, X, brk, nb, panels);
	*re = 0;
	*im = 0;
	for (j = 0; j < Q.n; j++) {
		*re += Q.w[j] * Q.f[j] * cos(w0 * Q.x[j]);
		*im -= Q.w[j] * Q.f[j] * sin(w0 * Q.x[j]);
	}
	quad_free(&Q);
}

/* the cut-off where |f| is negligible (at most 2^15) */
static double
ft_cutoff(nid e, int xs)
{
	double X = 8, mx = 0;
	int k, ok;

	for (k = 0; k < 13; k++, X *= 2) {
		double v1 = fabs(xf_eval1(e, xs, X, &ok)), v2 = fabs(xf_eval1(e, xs,
		    -X, &ok));
		double v0 = fabs(xf_eval1(e, xs, 0.3, &ok));
		mx = fmax(mx, v0);
		if (fmax(v1, v2) < 1e-12 * fmax(mx, 1e-300))
			break;
	}
	return X;
}

static int
ft_numeric(nid e, int xs, double w0, double *re, double *im)
{
	double X = 8, mx = 0, cut[2];
	int k, ok, j;
	struct quad Q;

	for (k = 0; k < 13; k++, X *= 2) {
		double v1 = fabs(xf_eval1(e, xs, X, &ok)), v2 = fabs(xf_eval1(e, xs,
		    -X, &ok));
		double v0 = fabs(xf_eval1(e, xs, 0.3, &ok));
		mx = fmax(mx, v0);
		if (fmax(v1, v2) < 1e-12 * fmax(mx, 1e-300))
			break;
	}
	{
		double brk[66];
		int kinds[64], nb = 0, panels = (int)fmin(40000, 2 * X * fmax(2, w0));
		struct func t;
		memset(&t, 0, sizeof t);
		t.np = xs + 1;
		t.p[xs].ty = TY_REAL;
		t.body = e;
		if (xs == 0)
			nb = fn_discontinuities(&t, -X, X, brk, kinds, 64);
		brk[nb++] = 0;
		(void)cut;
		quad_make(&Q, e, xs, -X, X, brk, nb, panels);
	}
	*re = 0;
	*im = 0;
	for (j = 0; j < Q.n; j++) {
		*re += Q.w[j] * Q.f[j] * cos(w0 * Q.x[j]);
		*im -= Q.w[j] * Q.f[j] * sin(w0 * Q.x[j]);
	}
	quad_free(&Q);
	return 1;
}

static void
cmd_fourier(char *text, int inverse)
{
	static const char *const outs[] = {"w", "k", "omega", "f", "x", "t",
		NULL};
	static const char *const xpref[] = {"x", "t", NULL};
	static const char *const wpref[] = {"w", "omega", "k", NULL};
	char xn[32] = "", wn[32] = "";
	struct func *f;
	struct fctx C;
	struct cpair G;
	int i;

	trail_vars(text, xn, sizeof xn, wn, sizeof wn, outs);
	f = xf_parse(text);
	memset(&C, 0, sizeof C);
	C.xs = pick_slot(f, xn, inverse ? wpref : xpref);
	if (C.xs < 0 && f->np > 0)
		fail("fourier: which variable?  (fourier F x w)");
	if (C.xs < 0)
		C.xs = max_slot(f);
	C.x = xv(C.xs, C.xs < f->np ? sym_str(f->p[C.xs].name) : "x");
	C.ws = C.xs == f->np ? f->np + 1 : f->np;
	if (C.ws >= MAXP)
		fail("too many variables");
	if (!wn[0])
		snprintf(wn, sizeof wn, "%s", inverse ? "x" : "w");
	C.w = xv(C.ws, wn);
	G = ft_rec(&C, f->body);
	if (inverse) {
		/* f(x) = (1/2pi) F{G}(-x) */
		/* G real: F{G}(-x) = conj(F{G}(x)) */
		G.re = xmul(xk(1 / (2 * M_PI)), G.re);
		G.im = xmul(xk(-1 / (2 * M_PI)), G.im);
	}
	G.re = xf_polish(G.re, C.ws, C.w);
	G.im = xf_polish(G.im, C.ws, C.w);
	printf("%s{%s}(%s) = %s\n", inverse ? "F^-1" : "F", fname_of(f), wn,
	    cp_str(G));
	if (inverse)
		printf("  convention: f(x) = (1/(2pi)) int F(w) e^(iwx) dw\n");
	else
		printf("  convention: F(w) = int f(x) e^(-iwx) dx over the real line "
		    "(angular frequency; the inverse has 1/(2pi))\n");
	if (C.ndelta) {
		printf("  delta terms: f does not decay, the transform is a "
		    "distribution (not checked numerically)\n");
		return;
	}
	{
		int skip[2] = {C.xs, C.ws}, sl[MAXP], ns, ok;
		double vals[MAXP], worst = 0, ws[3] = {0.5, 1.3, 2.7};
		struct sbuf txt = {0};
		nid fb, Gr, Gi;
		ns = samples(f, skip, 2, sl, vals, &txt);
		fb = with_samples(f->body, sl, vals, ns);
		Gr = with_samples(G.re, sl, vals, ns);
		Gi = with_samples(G.im, sl, vals, ns);
		{
			double X = ft_cutoff(fb, C.xs), trunc = 0, gmax = 0;
			for (i = 0; i < 3; i++) {
				double re, im, re2, im2, gr, gi, sg = inverse ? -1 : 1;
				ft_numeric_on(fb, C.xs, sg * ws[i], X, &re, &im);
				ft_numeric_on(fb, C.xs, sg * ws[i], X / 2, &re2, &im2);
				if (inverse) {
					re /= 2 * M_PI;
					im /= 2 * M_PI;
					re2 /= 2 * M_PI;
					im2 /= 2 * M_PI;
				}
				gr = xf_eval1(Gr, C.ws, ws[i], &ok);
				gi = xf_eval1(Gi, C.ws, ws[i], &ok);
				gmax = fmax(gmax, fmax(cabs(gr + I * gi), cabs(re + I * im)));
				trunc = fmax(trunc, cabs((re + I * im) - (re2 + I * im2)));
				worst = fmax(worst, cabs((re + I * im) - (gr + I * gi)));
			}
			worst /= fmax(gmax, 1e-300);
			trunc /= fmax(gmax, 1e-300);
			{
				/* the tail beyond X: the difference between X/2 and X,
				 * scaled by how much f fell from X/2 to X (envelopes) */
				int ok2, j;
				double eX = 0, eH = 0;
				for (j = 0; j < 16; j++) {
					double u = 0.9 + 0.1 * j / 15;
					eX = fmax(eX, fabs(xf_eval1(fb, C.xs, u * X, &ok2)));
					eX = fmax(eX, fabs(xf_eval1(fb, C.xs, -u * X, &ok2)));
					eH = fmax(eH, fabs(xf_eval1(fb, C.xs, u * X / 2, &ok2)));
					eH = fmax(eH, fabs(xf_eval1(fb, C.xs, -u * X / 2, &ok2)));
				}
				trunc *= eH > 0 ? fmin(1, eX / eH) : 0;
			}
			if (trunc > 1e-8)
				printf("  check: %s the integral over [-%s, %s] computed "
				    "numerically at %s = 1/2, 13/10, 27/10%s%s: max relative "
				    "difference %.2g (f decays slowly: the numeric integral "
				    "itself is uncertain by about %.2g)\n",
				    worst < 10 * trunc + 1e-6 ? "consistent with" :
				    "DIFFERS FROM", xf_num(X), xf_num(X), wn, ns ? " with " : "",
				    ns ? txt.p : "", worst, trunc);
			else
				printf("  check: %s the integral computed numerically at %s = "
				    "1/2, 13/10, 27/10%s%s: max relative difference %.2g\n",
				    worst < 1e-6 ? "matches" : "DIFFERS FROM", wn,
				    ns ? " with " : "", ns ? txt.p : "", worst);
		}
		free(txt.p);
	}
}

/* ---- DFT ---- */

/* the items of [a, b, ...]: complex values (a + bi allowed) */
static int
list_values(const char *text, double complex *v, int max)
{
	char *s = xstrdup(text), *p = s, *part[1024];
	int n, k;

	trim(p);
	if (*p == '[' || *p == '(')
		p++;
	if (strlen(p) && (p[strlen(p) - 1] == ']' || p[strlen(p) - 1] == ')'))
		p[strlen(p) - 1] = 0;
	n = split_list(p, part, max < 1024 ? max : 1024);
	for (k = 0; k < n; k++) {
		struct func *f = xf_parse(part[k]);
		double re, im, v1;
		int ok;
		if (f->np == 0 && xnum(f->body, &re)) {
			v[k] = re;
			continue;
		}
		if (f->np == 1 && (!strcmp(sym_str(f->p[0].name), "i") ||
		    !strcmp(sym_str(f->p[0].name), "j"))) {
			re = xf_eval1(f->body, 0, 0, &ok);
			v1 = xf_eval1(f->body, 0, 1, &ok);
			im = v1 - re;
			if (fabs(xf_eval1(f->body, 0, 2, &ok) - (re + 2 * im)) > 1e-9 *
			    fmax(1, fabs(re) + fabs(im)))
				fail("%s is not a + bi", part[k]);
			v[k] = re + I * im;
			continue;
		}
		fail("dft: %s is not a number", part[k]);
	}
	free(s);
	return n;
}

static void
fft_rec(double complex *x, int n, int sign)
{
	double complex *e, *o;
	int k;

	if (n == 1)
		return;
	e = xmalloc((size_t)n / 2 * sizeof *e);
	o = xmalloc((size_t)n / 2 * sizeof *o);
	for (k = 0; k < n / 2; k++) {
		e[k] = x[2 * k];
		o[k] = x[2 * k + 1];
	}
	fft_rec(e, n / 2, sign);
	fft_rec(o, n / 2, sign);
	for (k = 0; k < n / 2; k++) {
		double complex t = cexp(sign * 2 * M_PI * I * k / n) * o[k];
		x[k] = e[k] + t;
		x[k + n / 2] = e[k] - t;
	}
	free(e);
	free(o);
}

static void
dft(const double complex *x, double complex *X, int n, int inverse)
{
	int k, j;

	if ((n & (n - 1)) == 0) {
		memcpy(X, x, (size_t)n * sizeof *X);
		fft_rec(X, n, inverse ? 1 : -1);
	} else
		for (k = 0; k < n; k++) {
			double complex s = 0;
			for (j = 0; j < n; j++)
				s += x[j] * cexp((inverse ? 1 : -1) * 2 * M_PI * I *
				    (double)((int64_t)k * j % n) / n);
			X[k] = s;
		}
	if (inverse)
		for (k = 0; k < n; k++)
			X[k] /= n;
	for (k = 0; k < n; k++) {
		double sc = cabs(X[k]) + 1e-300;
		if (fabs(creal(X[k])) < 1e-13 * fmax(sc, 1))
			X[k] = I * cimag(X[k]);
		if (fabs(cimag(X[k])) < 1e-13 * fmax(sc, 1))
			X[k] = creal(X[k]);
	}
}

static void
cmd_dft(char *text, int inverse)
{
	double complex x[1024], X[1024], back[1024];
	int n = list_values(text, x, 1024), k;
	double worst = 0, p1 = 0, p2 = 0;

	if (n < 1)
		fail("dft: expected a list [x0, x1, ...]");
	dft(x, X, n, inverse);
	printf("%s of %d values (%s; %s)\n", inverse ? "inverse DFT" : "DFT", n,
	    inverse ? "x[n] = (1/N) sum_k X[k] e^(2 pi i k n/N)" :
	    "X[k] = sum_n x[n] e^(-2 pi i k n/N)", (n & (n - 1)) == 0 ?
	    "radix-2 FFT, N log2 N operations" : "direct sum, N^2 operations");
	printf("  %-4s %-28s %-14s %s\n", "k", inverse ? "x[k]" : "X[k]",
	    "|.|", "arg");
	for (k = 0; k < n && k < 64; k++) {
		double ph = carg(X[k]);
		printf("  %-4d %-28s %-14s %s\n", k, xf_cnum(X[k]), xf_num(cabs(X[k])),
		    cabs(X[k]) == 0 ? "-" : xf_num(fabs(ph) < 1e-15 ? 0 : ph));
	}
	if (n > 64)
		printf("  ... (%d more)\n", n - 64);
	dft(X, back, n, !inverse);
	for (k = 0; k < n; k++) {
		worst = fmax(worst, cabs(back[k] - x[k]));
		p1 += cabs(x[k]) * cabs(x[k]);
		p2 += cabs(X[k]) * cabs(X[k]);
	}
	printf("  check: the %s transform gives the input back (max difference "
	    "%.2g); Parseval: sum |x|^2 = %s, sum |X|^2/N = %s\n",
	    inverse ? "forward" : "inverse", worst, xf_num(inverse ? p2 : p1),
	    xf_num(inverse ? p1 / n : p2 / n));
	if (!inverse)
		printf("  bin k is the frequency k/N cycles per sample; k > N/2 are "
		    "the negative frequencies k - N\n");
}

/* ---- DTFT ---- */

static void
cmd_dtft(char *text)
{
	static const double wd[] = {0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI};
	int i;

	trim(text);
	if (text[0] == '[') {
		double complex x[1024];
		int n = list_values(text, x, 1024), k;
		nid w = xv(0, "w"), re = xk(0), im = xk(0);
		for (k = 0; k < n; k++) {
			if (cimag(x[k]) != 0)
				fail("dtft: complex values in the list: use dft");
			re = xadd(re, xmul(xk(creal(x[k])), k ? mk_un(O_COS, xmul(xk(k),
			    w)) : xk(1)));
			im = xsub(im, xmul(xk(creal(x[k])), mk_un(O_SIN, xmul(xk(k), w))));
		}
		printf("DTFT of %d values: X(w) = sum_n x[n] e^(-iwn)\n", n);
		printf("  X(w) = %s\n", cp_str(cp(re, im)));
		printf("  %-8s %-12s %s\n", "w", "|X|", "arg");
		for (i = 0; i < 5; i++) {
			double complex s = 0;
			for (k = 0; k < n; k++)
				s += x[k] * cexp(-I * wd[i] * k);
			printf("  %-8s %-12.6g %.6g\n", xf_num(wd[i]), cabs(s),
			    cabs(s) < 1e-14 || fabs(carg(s)) < 1e-12 ? 0 : carg(s));
		}
		printf("  periodic in w with period 2pi; the DFT samples it at w = "
		    "2pi k/N\n");
		return;
	}
	{
		struct func *f = xf_parse(text);
		static const char *const npref[] = {"n", "k", NULL};
		struct zctx C;
		struct xrat r;
		nid X;
		double R = 0;
		memset(&C, 0, sizeof C);
		C.ns = pick_slot(f, NULL, npref);
		if (C.ns < 0)
			C.ns = 0;
		if (f->np > 1)
			fail("dtft: give the other parameters values");
		C.n = xv(C.ns, f->np ? sym_str(f->p[C.ns].name) : "n");
		C.zs = C.ns + 1;
		C.z = xv(C.zs, "z");
		xf_discrete = 1;
		X = zt_of(f->body, &C);
		printf("DTFT of %s: X(e^(iw)) = X(z) at z = e^(iw)\n", fname_of(f));
		printf("  X(z) = %s\n", xf_str(X));
		if (C.hasroc)
			xnum(C.roc, &R);
		if (R >= 1) {
			printf("  the region of convergence |z| > %s excludes the unit "
			    "circle: the DTFT does not exist as a function\n", xf_num(R));
			xf_discrete = 0;
			return;
		}
		if (!xq_of(X, C.zs, &r))
			fail("dtft: X(z) is not rational");
		printf("  %-8s %-12s %s\n", "w", "|X|", "arg");
		for (i = 0; i < 5; i++) {
			double complex v = xq_ceval(&r, cexp(I * wd[i]));
			printf("  %-8s %-12.6g %.6g\n", xf_num(wd[i]), cabs(v),
			    fabs(carg(v)) < 1e-14 ? 0 : carg(v));
		}
		xf_discrete = 0;
	}
}

/* ---- generating functions ---- */

/* c0 + c1 x + ... in ascending powers, as text */
static void
asc_text(struct sbuf *b, const struct xpoly *p, const char *x)
{
	int i, first = 1;

	for (i = 0; i <= p->deg; i++) {
		double c = p->c[i];
		const char *cs;
		if (c == 0)
			continue;
		cs = xf_num(fabs(c));
		if (first)
			sb_puts(b, c < 0 ? "-" : "");
		else
			sb_puts(b, c < 0 ? " - " : " + ");
		if (i == 0)
			sb_puts(b, cs);
		else {
			if (strcmp(cs, "1")) {
				if (strchr(cs, '/') || strchr(cs, ' '))
					sb_printf(b, "(%s)", cs);
				else
					sb_puts(b, cs);
			}
			sb_puts(b, x);
			if (i > 1)
				sb_printf(b, "^%d", i);
		}
		first = 0;
	}
	if (first)
		sb_puts(b, "0");
}

/* a rational G(x) with G's denominator normalised to 1 + ... */
static void
gf_text(struct sbuf *b, struct xrat *r, const char *x)
{
	struct sbuf n = {0}, d = {0};
	double c0;

	xq_reduce(r);
	c0 = r->d.deg >= 0 ? r->d.c[0] : 1;
	if (c0 != 0 && r->d.deg > 0) {
		xp_scale(&r->n, 1 / c0);
		xp_scale(&r->d, 1 / c0);
	}
	asc_text(&n, &r->n, x);
	if (r->d.deg <= 0) {
		sb_puts(b, n.p);
	} else {
		asc_text(&d, &r->d, x);
		sb_printf(b, top_sum(n.p) ? "(%s)/(%s)" : "%s/(%s)", n.p, d.p);
	}
	free(n.p);
	free(d.p);
}

/* Stirling numbers of the second kind S(k, j) */
static double
stirling2(int k, int j)
{
	static double S[16][16];
	static int done;

	if (!done) {
		int a, b;
		S[0][0] = 1;
		for (a = 1; a < 16; a++)
			for (b = 1; b <= a; b++)
				S[a][b] = b * S[a - 1][b] + S[a - 1][b - 1];
		done = 1;
	}
	return k < 16 && j < 16 ? S[k][j] : 0;
}

/* the EGF of a(n) = sum of c n^k r^n [cos|sin(b n)], delta[n - k]: a node */
static nid
egf_of(nid a, int ns, nid x)
{
	struct terms T;
	nid G = xk(0);
	int i, k;

	terms_of(a, &T);
	for (i = 0; i < T.n; i++) {
		struct facs F;
		double pw = 0, c1, c0;
		nid r = xk(1), u, arg;
		int trig = 0;
		nid beta = 0;
		memset(&F, 0, sizeof F);
		F.coef = xk(1);
		fac_rec(T.t[i], ns, 1, &F);
		for (k = 0; k < F.nf; k++) {
			nid f = unc(F.f[k]);
			if (is_var(f, ns) && F.pw[k] > 0 && F.pw[k] == floor(F.pw[k]))
				pw += F.pw[k];
			else if (N.op[f] == O_POW && !has_slot(N.a[f], ns) &&
			    lin_of(N.b[f], ns, &u, &arg)) {
				r = xmul(r, xpow(N.a[f], xmul(xk(F.pw[k]), u)));
				F.coef = xmul(F.coef, xpow(N.a[f], xmul(xk(F.pw[k]), arg)));
			} else if (xis_exp(f, &u) && lin_of(u, ns, &c1 == 0 ? &u : &u,
			    &arg)) {
				nid s1, s0;
				lin_of(u, ns, &s1, &s0);
				r = xmul(r, xexp(xmul(xk(F.pw[k]), s1)));
				F.coef = xmul(F.coef, xexp(xmul(xk(F.pw[k]), s0)));
			} else if ((N.op[f] == O_COS || N.op[f] == O_SIN) && !trig &&
			    F.pw[k] == 1 && lin_of(N.a[f], ns, &u, &arg) &&
			    xnum(arg, &c0) && c0 == 0) {
				trig = N.op[f];
				beta = u;
			} else if (xis_kron(f, &arg) && F.pw[k] == 1 &&
			    lin_num(arg, ns, &c1, &c0)) {
				/* delta[n - m]: x^m/m! */
				double m = -c0 / c1;
				pw = -1;
				G = xadd(G, xmul(F.coef, xdiv(xpow(x, xk(m)), xk(factd(
				    (int)m)))));
				break;
			} else
				fail("egf: no rule for %s", xf_str(f));
		}
		if (pw < 0)
			continue;
		if (pw > 15)
			fail("egf: n^%d: too high a power", (int)pw);
		{
			/* sum n^k (r x)^n/n! = e^(rx) sum_j S(k, j) (rx)^j */
			int kk = (int)pw, j;
			nid rx, poly = xk(0), e;
			if (trig) {
				/* r^n cos(b n): Re e^(r e^(ib) x) = e^(r cos b x) cos(r sin b x) */
				nid rc = xmul(r, mk_un(O_COS, beta)), rs = xmul(r, mk_un(O_SIN,
				    beta));
				if (kk)
					fail("egf: n^k times a sine: no rule");
				e = xmul(xexp(xmul(rc, x)), mk_un(trig, xmul(rs, x)));
				G = xadd(G, xmul(F.coef, e));
				continue;
			}
			rx = xmul(r, x);
			if (kk == 0)
				poly = xk(1);
			for (j = 1; j <= kk; j++)
				poly = xadd(poly, xmul(xk(stirling2(kk, j)), xpow(rx, xk(j))));
			G = xadd(G, xmul(F.coef, xmul(poly, xexp(rx))));
		}
	}
	return G;
}

static const struct {
	const char *name;
	const char *def;
} named_seq[] = {
	{"fibonacci", "a[n+2] = a[n+1] + a[n], a[0] = 0, a[1] = 1"},
	{"fib", "a[n+2] = a[n+1] + a[n], a[0] = 0, a[1] = 1"},
	{"the fibonacci numbers", "a[n+2] = a[n+1] + a[n], a[0] = 0, a[1] = 1"},
	{"fibonacci numbers", "a[n+2] = a[n+1] + a[n], a[0] = 0, a[1] = 1"},
	{"lucas", "a[n+2] = a[n+1] + a[n], a[0] = 2, a[1] = 1"},
	{"lucas numbers", "a[n+2] = a[n+1] + a[n], a[0] = 2, a[1] = 1"},
	{"pell", "a[n+2] = 2a[n+1] + a[n], a[0] = 0, a[1] = 1"},
	{"pell numbers", "a[n+2] = 2a[n+1] + a[n], a[0] = 0, a[1] = 1"},
	{"tribonacci", "a[n+3] = a[n+2] + a[n+1] + a[n], a[0] = 0, a[1] = 0, "
	    "a[2] = 1"},
	{"catalan", "binom(2n, n)/(n + 1)"},
	{"catalan numbers", "binom(2n, n)/(n + 1)"},
	{"central binomial", "binom(2n, n)"},
	{"central binomial coefficients", "binom(2n, n)"},
	{"squares", "n^2"},
	{"cubes", "n^3"},
	{"triangular", "n(n + 1)/2"},
	{"triangular numbers", "n(n + 1)/2"},
	{"powers of 2", "2^n"},
	{"powers of two", "2^n"},
	{"natural numbers", "n"},
	{"ones", "1"},
	{"factorials", "n!"},
	{NULL, NULL}
};

static void
cmd_genfunc(char *text, int egf)
{
	char *t = text;
	struct func *f = NULL;
	nid a = 0, x, G = 0;
	int ns, xsl, k, isrec = 0;
	struct sbuf gtxt = {0};
	double R = INFINITY;
	struct xrat gr;
	int rational = 0;
	const char *libname = NULL;

	trim(t);
	for (k = 0; named_seq[k].name; k++)
		if (!strcasecmp(t, named_seq[k].name)) {
			printf("%s: %s\n", named_seq[k].name, named_seq[k].def);
			t = xstrdup(named_seq[k].def);
			break;
		}
	xf_discrete = 1;
	if (strchr(t, '[') && strchr(t, '=')) {
		/* a recurrence: Y(z) from zsolve, G(x) = Y(1/x) */
		struct zctx C;
		struct lineq L;
		nid Y;
		a = zsolve_core(t, 1, &Y, &C, &L);
		ns = L.xs;
		xsl = C.zs;
		x = xv(xsl, "x");
		isrec = 1;
		printf("  a[%s] = %s\n", L.xn, xf_str(a));
		if (!egf) {
			nid Gx = xsubst1(Y, C.zs, xdiv(xk(1), x));
			if (xq_of(Gx, xsl, &gr)) {
				rational = 1;
				G = Gx;
			}
		}
	} else {
		static const char *const npref[] = {"n", "k", "i", NULL};
		f = xf_parse(t);
		ns = pick_slot(f, NULL, npref);
		if (ns < 0)
			ns = f->np;
		if (f->np > (xf_slot(f, sym_str(f->np ? f->p[0].name : 0)) >= 0 &&
		    ns < f->np ? 1 : 0))
			fail("genfunc: the terms have other variables");
		a = f->body;
		xsl = ns + 1;
		if (xsl >= MAXP)
			fail("too many variables");
		x = xv(xsl, "x");
		if (!egf) {
			jmp_buf saved;
			struct zctx C;
			memset(&C, 0, sizeof C);
			C.ns = ns;
			C.n = xv(ns, f->np ? sym_str(f->p[ns].name) : "n");
			C.zs = xsl;
			C.z = xv(xsl, "z");
			(void)saved;
			zquiet = 1;
			if (setjmp(zjmp) == 0) {
				nid X = zt_of(a, &C);
				nid Gx = xsubst1(X, xsl, xdiv(xk(1), x));
				if (xq_of(Gx, xsl, &gr)) {
					rational = 1;
					G = Gx;
				} else
					G = Gx;
			}
			zquiet = 0;
			notes_drop();
		}
	}
	if (!egf) {
		if (rational) {
			double complex z[XD];
			int m[XD], nz = xf_roots(&gr.d, z, m), i;
			for (i = 0; i < nz; i++)
				R = fmin(R, cabs(z[i]));
			gf_text(&gtxt, &gr, "x");
		} else if (!G) {
			/* the library on the terms */
			double c[30];
			int i, ok;
			for (i = 0; i < 30; i++) {
				c[i] = xf_eval1(a, ns, i, &ok);
				if (!ok || !isfinite(c[i]))
					fail("genfunc: a_%d is undefined", i);
			}
			G = xf_libsum(c, 30, 0, x, &R, &libname);
			if (!G) {
				/* radius 0? */
				double r1 = fabs(c[29] / c[28]);
				if (r1 > 20)
					printf("  the ordinary generating function has radius of "
					    "convergence 0 (a_n grows faster than any r^n); try "
					    "egf\n");
				else
					printf("  no closed form recognised (C-finite sequences, "
					    "and the library: exp, log, atan, binomial, Catalan)\n");
				xf_discrete = 0;
				return;
			}
			sb_puts(&gtxt, xf_str(G));
		} else
			sb_puts(&gtxt, xf_str(G));
		printf("  G(x) = sum_{n>=0} a_n x^n = %s", gtxt.p);
		if (isinf(R))
			printf("    for every x\n");
		else
			printf("    for |x| < %s\n", xf_num(R));
		if (libname)
			printf("  from the library: %s\n", libname);
		{
			/* check: sum a_n x^n at x = R/2 */
			double x0 = isinf(R) ? 0.5 : R / 2, s = 0, xp = 1, gv;
			int i, ok, okg;
			for (i = 0; i < 400; i++, xp *= x0) {
				double v = xf_eval1(a, ns, i, &ok);
				if (!ok || !isfinite(v))
					break;
				s += v * xp;
				if (fabs(v * xp) < 1e-18 * fabs(s) && i > 20)
					break;
			}
			gv = xf_eval1(G, xsl, x0, &okg);
			printf("  check: sum_n a_n x^n at x = %s is %.15g, G = %.15g "
			    "(difference %.2g)\n", xf_num(x0), s, gv, fabs(s - gv));
		}
	} else {
		nid E;
		{
			/* the closed form of a_n first */
			jmp_buf saved;
			memcpy(saved, fail_jmp, sizeof saved);
			if (!isrec && setjmp(fail_jmp) == 0) {
				struct zctx C;
				struct xrat r2;
				double R2;
				nid X;
				memset(&C, 0, sizeof C);
				C.ns = ns;
				C.n = xv(ns, f && f->np ? sym_str(f->p[ns].name) : "n");
				C.zs = xsl;
				C.z = xv(xsl, "z");
				(void)X;
				(void)r2;
				(void)R2;
			}
			memcpy(fail_jmp, saved, sizeof saved);
		}
		{
			/* n!: 1/(1 - x) */
			nid fa;
			if (xis_fact(a, &fa) && is_var(fa, ns))
				E = xdiv(xk(1), xsub(xk(1), x));
			else
				E = egf_of(a, ns, x);
		}
		printf("  E(x) = sum_{n>=0} a_n x^n/n! = %s\n", xf_str(E));
		{
			double xs2[2] = {0.25, 0.5};
			int j, i, ok;
			double worst = 0;
			for (j = 0; j < 2; j++) {
				double s = 0, xp = 1, fct = 1, ev;
				for (i = 0; i < 160; i++) {
					double v = xf_eval1(a, ns, i, &ok);
					if (!ok || !isfinite(v))
						break;
					s += v * xp / fct;
					xp *= xs2[j];
					fct *= i + 1;
				}
				ev = xf_eval1(E, xsl, xs2[j], &ok);
				worst = fmax(worst, fabs(s - ev) / fmax(1, fabs(ev)));
			}
			printf("  check: the series at x = 1/4 and 1/2 matches E (max "
			    "relative difference %.2g)\n", worst);
		}
	}
	free(gtxt.p);
	xf_discrete = 0;
}

/* ---- the command dispatcher ---- */

static const char *const xcmds[] = {"laplace", "ilaplace", "ztrans",
	"iztrans", "zsolve", "tf", "fourier", "ifourier", "fseries", "dft",
	"idft", "fft", "ifft", "dtft", "genfunc", "ogf", "egf", "converge",
	"converges", "radius", "powerseries", "coeffs", "asymptotic", NULL};

int
xform_command(struct P *p)
{
	char w[32], *text;
	uint32_t at = cmdp_pos(p);
	size_t cap = 4096;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (!in_list(xcmds, w) && strcmp(w, "sum") && strcmp(w, "prod") &&
	    strcmp(w, "Σ") && strcmp(w, "∏")) {
		cmdp_seek(p, at);
		return 0;
	}
	if (func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	/* sum(...) with an infinite bound is ours (series.c) */
	if (!in_list(xcmds, w)) {
		cmdp_seek(p, at);
		return series_command(p, w);
	}
	text = xmalloc(cap);
	if (!cmdp_rest(p, text, cap, NULL))
		text[0] = 0;
	xf_helpers();
	notes_flush();
	if (!strcmp(w, "laplace")) {
		char *r = text;
		while (*r == ' ')
			r++;
		if (!strncmp(r, "solve ", 6))
			cmd_lsolve(r + 6);
		else
			cmd_laplace(text);
	} else if (!strcmp(w, "ilaplace"))
		cmd_ilaplace(text);
	else if (!strcmp(w, "ztrans"))
		cmd_ztrans(text);
	else if (!strcmp(w, "iztrans"))
		cmd_iztrans(text);
	else if (!strcmp(w, "zsolve"))
		cmd_zsolve(text);
	else if (!strcmp(w, "tf"))
		cmd_tf(text);
	else if (!strcmp(w, "fseries"))
		cmd_fseries(text);
	else if (!strcmp(w, "fourier") || !strcmp(w, "ifourier")) {
		char *r = text;
		while (*r == ' ')
			r++;
		if (!strncmp(r, "series ", 7))
			cmd_fseries(r + 7);
		else
			cmd_fourier(text, w[0] == 'i');
	} else if (!strcmp(w, "dft") || !strcmp(w, "fft"))
		cmd_dft(text, 0);
	else if (!strcmp(w, "idft") || !strcmp(w, "ifft"))
		cmd_dft(text, 1);
	else if (!strcmp(w, "dtft"))
		cmd_dtft(text);
	else if (!strcmp(w, "genfunc") || !strcmp(w, "ogf"))
		cmd_genfunc(text, 0);
	else if (!strcmp(w, "egf"))
		cmd_genfunc(text, 1);
	else {
		free(text);
		cmdp_seek(p, at);
		return series_command(p, w);
	}
	free(text);
	return 1;
}
