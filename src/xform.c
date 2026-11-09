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
static nid
polish(nid R, int slot, nid x)
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
		R = polish(C->R[k], C->ss, C->s);
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
						sb_printf(o, "__d%d", ord);
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
				sb_printf(o, "__d%d", ord + (shifts ? 16 : 0));
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
	    "x");
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
			nid R = polish(C.R[g], ss, s);
			if (C.sh[g] == 0 && !C.dl[g]) {
				R = xadd(R, Qn);
				any0 = 1;
			}
			R = polish(xdiv(R, Pn), ss, s);
			if (C.sh[g] != 0)
				R = xmul(R, xexp(xmul(xk(-C.sh[g]), s)));
			Y = xadd(Y, R);
		}
		if (!any0 && Q.deg >= 0)
			Y = xadd(polish(xdiv(Qn, Pn), ss, s), Y);
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

/* ---- the command dispatcher ---- */

static const char *const xcmds[] = {"laplace", "ilaplace", "ztrans",
	"iztrans", "zsolve", "tf", "fourier", "ifourier", "fseries", "dft",
	"idft", "fft", "ifft", "dtft", "genfunc", "egf", "converge",
	"converges", "radius", "powerseries", "asymptotic", NULL};

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
	else {
		free(text);
		cmdp_seek(p, at);
		cmdp_word(p, w, sizeof w);
		return series_command(p, w);
	}
	free(text);
	return 1;
}
