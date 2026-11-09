/*
 * Cancelling quotients, keeping the conditions.
 *
 * x/x is 1 only where x != 0; (x^2 - 1)/(x - 1) is x + 1 only where
 * x != 1.  cancel_quotients() rewrites every real quotient in an
 * expression by
 *   - collecting the factors of numerator and denominator (products,
 *     quotients, negation, integer powers, numeric constants) and
 *     cancelling equal ones (equal nodes: the DAG is hash-consed), and
 *   - dividing exactly when both sides are polynomials in the variable
 *     with numeric coefficients and the remainder is zero,
 * and appends one "factor != 0" condition per cancelled factor, so the
 * caller can say where the simpler form is valid.
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXF 32

struct fac {
	nid base;
	int e;
	int wasden;	/* appeared in a denominator */
	int wasnum;	/* appeared in a numerator */
};

static int
add_fac(struct fac *f, int *nf, nid base, int e)
{
	int i;

	for (i = 0; i < *nf; i++)
		if (f[i].base == base) {
			f[i].e += e;
			if (e < 0)
				f[i].wasden = 1;
			else
				f[i].wasnum = 1;
			return 1;
		}
	if (*nf == MAXF)
		return 0;
	f[*nf].base = base;
	f[*nf].e = e;
	f[*nf].wasden = e < 0;
	f[*nf].wasnum = e > 0;
	(*nf)++;
	return 1;
}

static int
collect(nid n, int sign, struct fac *f, int *nf, double *k)
{
	double v;

	switch (N.op[n]) {
	case O_MUL:
		return collect(N.a[n], sign, f, nf, k) &&
		    collect(N.b[n], sign, f, nf, k);
	case O_DIV:
		return collect(N.a[n], sign, f, nf, k) &&
		    collect(N.b[n], -sign, f, nf, k);
	case O_NEG:
		if (sign % 2)   /* (-x)^2 = x^2 */
			*k = -*k;
		return collect(N.a[n], sign, f, nf, k);
	case O_POW:
		if (num_of(N.b[n], &v) && v == floor(v) && fabs(v) <= 64)
			return collect(N.a[n], sign * (int)v, f, nf, k);
		break;
	case O_CAST:
		if (TY_FP(N.ty[n]) && !TY_FP(N.ty[N.a[n]]))
			break;
		return collect(N.a[n], sign, f, nf, k);
	}
	if (num_of(n, &v)) {
		if (v == 0 && sign < 0)
			return 0;   /* x/0 stays as it is */
		*k *= pow(v, sign);
		return 1;
	}
	return add_fac(f, nf, n, sign);
}

static nid
product(const struct fac *f, int nf, int neg)
{
	nid r = 0, t;
	int i, e;

	for (i = 0; i < nf; i++) {
		e = neg ? -f[i].e : f[i].e;
		if (e <= 0)
			continue;
		t = e == 1 ? f[i].base : mk_bin(O_POW, f[i].base,
		    mk_const_r((double)e));
		r = r ? mk_bin(O_MUL, r, t) : t;
	}
	return r;
}

static void
cond(struct sbuf *c, nid x)
{
	char buf[256];

	snprintf(buf, sizeof buf, "%s != 0", node_str(x));
	if (c->p && strstr(c->p, buf))
		return;
	if (c->n)
		sb_puts(c, ", ");
	sb_puts(c, buf);
}

/* exact division of polynomials with numeric coefficients, else 0 */
static nid
polydiv(nid num, nid den, int slot, nid x, struct sbuf *c)
{
	struct npoly pn, pd;
	double a[13], b[13], q[13];
	int i, j;

	if (!npoly_of(num, slot, &pn, 12) || !npoly_of(den, slot, &pd, 12) ||
	    pd.deg < 1 || pn.deg < pd.deg)
		return 0;
	for (i = 0; i <= pn.deg; i++)
		if (!num_of(pn.c[i], &a[i]))
			return 0;
	for (i = 0; i <= pd.deg; i++)
		if (!num_of(pd.c[i], &b[i]))
			return 0;
	if (b[pd.deg] == 0)
		return 0;
	memset(q, 0, sizeof q);
	for (i = pn.deg - pd.deg; i >= 0; i--) {
		q[i] = a[i + pd.deg] / b[pd.deg];
		for (j = 0; j <= pd.deg; j++)
			a[i + j] -= q[i] * b[j];
	}
	{
		double scale = 0;
		for (i = 0; i <= pn.deg; i++)
			if (pn.c[i] && num_of(pn.c[i], &b[0]))
				scale = fmax(scale, fabs(b[0]));
		for (i = 0; i < pd.deg; i++)
			if (fabs(a[i]) > 1e-12 * fmax(1, scale))
				return 0;   /* a remainder: not exact */
	}
	{
		nid r = 0, t;
		for (i = pn.deg - pd.deg; i >= 0; i--) {
			if (q[i] == 0)
				continue;
			t = i == 0 ? mk_const_r(q[i]) : mk_bin(O_MUL, mk_const_r(q[i]),
			    i == 1 ? x : mk_bin(O_POW, x, mk_const_r(i)));
			r = r ? mk_bin(O_ADD, r, t) : t;
		}
		cond(c, den);
		return r ? r : mk_const_r(0);
	}
}

static nid *memo;
static uint32_t memon;

static nid
walk(nid n, int slot, nid x, struct sbuf *c)
{
	nid a, b, r;
	struct fac f[MAXF];
	int nf = 0, i;
	double k = 1;

	if (n == 0 || n >= memon)
		return n;
	if (memo[n])
		return memo[n];
	switch (N.op[n]) {
	case O_CONST: case O_VAR: case O_UNDEF: case O_TABLE: case O_LOOP:
	case O_INDEX: case O_ATOMIC: case O_BARRIER:
		return memo[n] = n;
	}
	a = walk(N.a[n], slot, x, c);
	b = N.b[n] ? walk(N.b[n], slot, x, c) : 0;
	if (N.op[n] == O_SEL)
		r = mk_sel(a, b, walk(N.c[n], slot, x, c));
	else if (N.op[n] == O_CAST)
		r = mk_cast(N.ty[n], a);
	else if (opinfo[N.op[n]].arity == 1)
		r = mk_un(N.op[n], a);
	else if (opinfo[N.op[n]].arity == 2)
		r = mk_bin(N.op[n], a, b);
	else
		r = n;
	if (N.op[r] != O_DIV || TYK(N.ty[r]) != K_REAL)
		return memo[n] = r;
	if (collect(r, 1, f, &nf, &k)) {
		nid num, den;
		int cancelled = 0;
		for (i = 0; i < nf; i++) {
			if (f[i].wasden && f[i].wasnum)
				cancelled = 1;   /* x^2/x^3 is 1/x */
			if (f[i].wasden && f[i].e >= 0)
				cond(c, f[i].base);   /* gone from the denominator */
		}
		if (cancelled) {
			num = product(f, nf, 0);
			den = product(f, nf, 1);
			if (num == 0)
				num = mk_const_r(k);
			else if (k != 1)
				num = mk_bin(O_MUL, mk_const_r(k), num);
			r = den ? mk_bin(O_DIV, num, den) : num;
		}
	}
	if (N.op[r] == O_DIV && x != 0) {
		nid q = polydiv(N.a[r], N.b[r], slot, x, c);
		if (q)
			r = q;
	}
	return memo[n] = r;
}

nid
cancel_quotients(nid e, int slot, nid x, struct sbuf *conds)
{
	nid r;

	memon = N.n;
	memo = xcalloc(memon, sizeof *memo);
	r = walk(e, slot, x, conds);
	free(memo);
	memo = NULL;
	return r;
}
