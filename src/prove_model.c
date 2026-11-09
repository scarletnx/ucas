/*
 * prove_model.c - counterexamples for statements that fail to prove.
 *
 * The statement is evaluated in a concrete model of the theory's
 * vocabulary: cochain complexes are the cellular cochains of standard
 * spaces (homalg.h), bock(C,p) is homalg's Bockstein spectral sequence of
 * C at p, computed exactly, H(C,n) is H^n(C;Z) with elements in canonical
 * coordinates.  Leading universal quantifiers are enumerated over small
 * ranges; atoms the model cannot decide are "unknown", and a
 * counterexample is reported only when the body is definitely false.
 * This is a computation, not a kernel proof: the report says so.
 */
#include "ucas.h"
#include "prove.h"
#include "homalg.h"

#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

static const char *const spaces[] = {"S^1", "S^2", "RP^2", "T^2", "RP^3",
	"L(4;1)", "L(9;1)", "RP^4", NULL};
#define NSP 8
#define MAXPG 8

struct bcache {
	int sp, p, ok;
	struct hss S;
};
static struct bcache bc[64];
static int nbc;

/* run f under a private fail() trap */
static struct hss *
bss(int sp, int p)
{
	int i;
	jmp_buf save;

	for (i = 0; i < nbc; i++)
		if (bc[i].sp == sp && bc[i].p == p)
			return bc[i].ok ? &bc[i].S : NULL;
	if (nbc == 64)
		return NULL;
	bc[nbc].sp = sp;
	bc[nbc].p = p;
	bc[nbc].ok = 0;
	memcpy(save, fail_jmp, sizeof save);
	if (setjmp(fail_jmp) == 0) {
		hspace s = hs_parse(spaces[sp], NULL);
		if (s != HNONE) {
			hss_bockstein(&bc[nbc].S, hcx_cochains(s), z_from_i64(p),
			    MAXPG);
			bc[nbc].ok = 1;
		}
	}
	memcpy(fail_jmp, save, sizeof save);
	notes_drop();
	nbc++;
	return bc[nbc - 1].ok ? &bc[nbc - 1].S : NULL;
}

struct val {
	int sort;
	int64_t i;		/* int; cx: the space */
	int sp, n;		/* el: the group H^n(spaces[sp]) */
	int64_t y[8];		/* el: canonical coordinates */
	int ny;
};

#define UNK 2
static struct val env[32];
static int nenv;
static int budget;

static int
isprime(int64_t p)
{
	int64_t d;

	if (p < 2)
		return 0;
	for (d = 2; d * d <= p; d++)
		if (p % d == 0)
			return 0;
	return 1;
}

static struct val *
bv(kt t)
{
	if (KOP(t) != KO_BVAR || (int)KT.aux[t] >= nenv)
		return NULL;
	return &env[nenv - 1 - KT.aux[t]];
}

static int
ival(kt t, int64_t *v)
{
	int64_t a, b;
	struct val *x;

	switch (KOP(t)) {
	case KO_NUM:
		*v = KT.num[t];
		return 1;
	case KO_BVAR:
		x = bv(t);
		if (!x || x->sort != KS_INT)
			return 0;
		*v = x->i;
		return 1;
	case KO_ADD:
	case KO_MUL:
		if (!ival(KA(t, 0), &a) || !ival(KA(t, 1), &b))
			return 0;
		*v = KOP(t) == KO_ADD ? a + b : a * b;
		return 1;
	case KO_NEG:
		if (!ival(KA(t, 0), &a))
			return 0;
		*v = -a;
		return 1;
	case KO_POW:
		if (!ival(KA(t, 0), &a) || !ival(KA(t, 1), &b) || b < 0 || b > 40)
			return 0;
		*v = 1;
		while (b-- > 0)
			*v *= a;
		return 1;
	}
	return 0;
}

static int
symis(kt t, const char *name)
{
	return (KOP(t) == KO_APP || KOP(t) == KO_PRED) &&
	    !strcmp(sym_str(ksyms[KT.aux[t]].name), name);
}

/* H(C, N): the space and degree */
static int
hgroup(kt t, int *sp, int *n)
{
	struct val *c;
	int64_t v;

	if (!symis(t, "H") || !(c = bv(KA(t, 0))) || c->sort == KS_INT ||
	    c->sort == KS_EL || !ival(KA(t, 1), &v))
		return 0;
	*sp = (int)c->i;
	*n = (int)v;
	return 1;
}

/* bock(C, P) */
static int
bockof(kt t, int *sp, int *p)
{
	struct val *c;
	int64_t v;

	if (!symis(t, "bock") || !(c = bv(KA(t, 0))) || !ival(KA(t, 1), &v))
		return 0;
	*sp = (int)c->i;
	*p = (int)v;
	return 1;
}

static const struct fgab *
hn(int sp, int p, int n)
{
	struct hss *S = bss(sp, p ? p : 2);

	if (!S)
		return NULL;
	return hc_D(&S->pg[0], n);
}

/* trace a-bar at page r: what = 0 zero, 1 in ker d, 2 in im d */
static int
trace(int sp, int p, struct val *a, int r, int what)
{
	struct hss *S = bss(sp, p);
	const struct fgab *H;
	struct htrace T;
	zint y[8], *x;
	int k, res;
	jmp_buf save;

	if (!S || r < 1 || r > MAXPG || a->n < S->pg[0].lo ||
	    a->n > S->pg[0].hi)
		return UNK;
	H = hc_D(&S->pg[0], a->n);
	if (H->ninv != a->ny)
		return UNK;
	for (k = 0; k < a->ny; k++)
		y[k] = z_from_i64(a->y[k]);
	x = xcalloc((size_t)H->n + 1, sizeof *x);
	res = UNK;
	memcpy(save, fail_jmp, sizeof save);
	if (setjmp(fail_jmp) == 0) {
		ab_from_canon(H, y, x);
		hss_trace(&T, S, a->n, x, r);
		res = what == 0 ? T.zero : what == 1 ? T.in_ker :
		    (T.in_im || T.zero);
		hss_trace_free(&T);
	}
	memcpy(fail_jmp, save, sizeof save);
	notes_drop();
	free(x);
	return res;
}

static int64_t
gcd64(int64_t a, int64_t b)
{
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	while (b) {
		int64_t t = a % b;
		a = b;
		b = t;
	}
	return a;
}

static int eval(kt f);

static int
atom(kt f)
{
	int64_t a, b, P, K;
	int sp, n, p, sp2, p2;
	struct val *x;
	const struct fgab *H;
	int k;

	switch (KOP(f)) {
	case KO_TRUE:
		return 1;
	case KO_FALSE:
		return 0;
	case KO_EQ:
	case KO_LE:
	case KO_LT:
		if (KSORT(KA(f, 0)) != KS_INT || !ival(KA(f, 0), &a) ||
		    !ival(KA(f, 1), &b))
			return UNK;
		return KOP(f) == KO_EQ ? a == b : KOP(f) == KO_LE ? a <= b : a < b;
	case KO_IN:
		if (!(x = bv(KA(f, 0))) || x->sort != KS_EL ||
		    !hgroup(KA(f, 1), &sp, &n))
			return UNK;
		return x->sp == sp && x->n == n ? 1 : UNK;
	case KO_PRED:
		break;
	default:
		return UNK;
	}
	if (symis(f, "prime"))
		return ival(KA(f, 0), &a) ? isprime(a) : UNK;
	if (symis(f, "fintype"))
		return 1;
	if (symis(f, "zsummand") || symis(f, "zpksummand")) {
		if (!(x = bv(KA(f, 0))) || x->sort != KS_EL ||
		    !hgroup(KA(f, 1), &sp, &n) || x->sp != sp || x->n != n ||
		    !(H = hn(sp, 2, n)))
			return UNK;
		if (symis(f, "zsummand")) {
			int64_t g = 0;
			for (k = 0; k < H->ninv; k++)
				if (z_is_zero(H->d[H->ntriv + k]))
					g = gcd64(g, x->y[k]);
			return g == 1;
		}
		if (!ival(KA(f, 2), &P) || !ival(KA(f, 3), &K) || K < 0 || K > 30)
			return UNK;
		{
			int64_t q = 1, ord = 1;
			int ones = 0, which = -1;
			while (K-- > 0)
				q *= P;
			for (k = 0; k < H->ninv; k++) {
				int64_t d = z_to_i64(H->d[H->ntriv + k]), o;
				if (!x->y[k])
					continue;
				if (d == 0)
					return 0;	/* infinite order */
				o = d / gcd64(d, x->y[k]);
				ord = ord / gcd64(ord, o) * o;
				if (x->y[k] == 1) {
					ones++;
					which = k;
				}
			}
			if (ord != q)
				return 0;
			if (ones == 1 && z_to_i64(H->d[H->ntriv + which]) == q)
				return 1;
			return UNK;
		}
	}
	/* Zr / Br / dzero / dimg of j(bock(C,p), a) */
	if (symis(f, "Zr") || symis(f, "Br") || symis(f, "dzero") ||
	    symis(f, "dimg")) {
		int last = KN(f) - 1;
		kt jt = KA(f, last);
		int64_t r, m;
		if (!bockof(KA(f, 0), &sp, &p) || !symis(jt, "j") ||
		    !bockof(KA(jt, 0), &sp2, &p2) || sp2 != sp || p2 != p ||
		    !(x = bv(KA(jt, 1))) || x->sort != KS_EL || x->sp != sp ||
		    !ival(KA(f, 1), &r) || !isprime(p))
			return UNK;
		if (symis(f, "Zr"))
			return r >= 1 ? 1 : UNK;
		if (symis(f, "Br"))
			return trace(sp, p, x, (int)r, 0);
		if (symis(f, "dzero"))
			return trace(sp, p, x, (int)r, 1);
		if (!ival(KA(f, 2), &m) || m != x->n - 1)
			return UNK;
		return trace(sp, p, x, (int)r, 2);
	}
	return UNK;
}

/* the group an el variable ranges over: an atom "v in H(C, N)" */
static int
find_group(kt f, uint32_t depth, int *sp, int *n)
{
	int i;

	if (KOP(f) == KO_IN && KOP(KA(f, 0)) == KO_BVAR &&
	    KT.aux[KA(f, 0)] == depth) {
		/* evaluate H(C, N) with the variable's own binder pushed */
		return hgroup(KA(f, 1), sp, n);
	}
	for (i = 0; i < KN(f); i++)
		if (KSORT(KA(f, i)) == KS_PROP && find_group(KA(f, i), depth +
		    (KOP(f) == KO_ALL || KOP(f) == KO_EX), sp, n))
			return 1;
	return 0;
}

static int
quant(kt f)
{
	int s = (int)KT.aux[f], all = KOP(f) == KO_ALL, res = all ? 1 : 0, r;
	int64_t v, lo = 0, hi = 4;
	struct val *x = &env[nenv];

	if (nenv >= 31 || budget-- <= 0)
		return UNK;
	memset(x, 0, sizeof *x);
	x->sort = s;
	if (s == KS_INT) {
		for (v = lo; v <= hi; v++) {
			x->i = v;
			nenv++;
			r = eval(KA(f, 0));
			nenv--;
			if (all && r == 0)
				return 0;
			if (!all && r == 1)
				return 1;
			if (r == UNK)
				res = UNK;
		}
		return all ? (res == 1 ? UNK : res) : (res == 0 ? UNK : res);
	}
	if (s == KS_EL) {
		int sp, n, k, l, m;
		const struct fgab *H;
		nenv++;
		x->sort = KS_EL;
		x->sp = -1;
		/* the group is fixed by the outer binders only */
		if (!find_group(KA(f, 0), 0, &sp, &n)) {
			nenv--;
			return UNK;
		}
		nenv--;
		H = hn(sp, 2, n);
		if (!H || H->ninv > 8)
			return UNK;
		/* 0, each generator, twice and three times it, sums of two */
		for (m = 0; m < 1 + 3 * H->ninv + H->ninv * H->ninv; m++) {
			memset(x->y, 0, sizeof x->y);
			x->sp = sp;
			x->n = n;
			x->ny = H->ninv;
			if (m == 0)
				;
			else if (m <= 3 * H->ninv) {
				k = (m - 1) % H->ninv;
				x->y[k] = (m - 1) / H->ninv + 1;
			} else {
				k = (m - 1 - 3 * H->ninv) / H->ninv;
				l = (m - 1 - 3 * H->ninv) % H->ninv;
				if (k >= l)
					continue;
				x->y[k] = 1;
				x->y[l] = 1;
			}
			nenv++;
			r = eval(KA(f, 0));
			nenv--;
			if (all && r == 0)
				return 0;
			if (!all && r == 1)
				return 1;
		}
		return UNK;
	}
	/* a complex: the standard spaces */
	for (v = 0; v < NSP; v++) {
		x->i = v;
		nenv++;
		r = eval(KA(f, 0));
		nenv--;
		if (all && r == 0)
			return 0;
		if (!all && r == 1)
			return 1;
	}
	return UNK;
}

static int
eval(kt f)
{
	int a, b;

	switch (KOP(f)) {
	case KO_NOT:
		a = eval(KA(f, 0));
		return a == UNK ? UNK : !a;
	case KO_AND:
		a = eval(KA(f, 0));
		if (a == 0)
			return 0;
		b = eval(KA(f, 1));
		return b == 0 ? 0 : a == 1 && b == 1 ? 1 : UNK;
	case KO_OR:
		a = eval(KA(f, 0));
		if (a == 1)
			return 1;
		b = eval(KA(f, 1));
		return b == 1 ? 1 : a == 0 && b == 0 ? 0 : UNK;
	case KO_IMP:
		a = eval(KA(f, 0));
		if (a == 0)
			return 1;
		b = eval(KA(f, 1));
		return b == 1 ? 1 : a == 1 && b == 0 ? 0 : UNK;
	case KO_IFF:
		a = eval(KA(f, 0));
		b = eval(KA(f, 1));
		return a == UNK || b == UNK ? UNK : a == b;
	case KO_ALL:
	case KO_EX:
		return quant(f);
	default:
		return atom(f);
	}
}

/* describe the assignment that falsifies the body (leading foralls) */
static int
search(kt f, struct sbuf *out, int depth)
{
	int r;

	if (KOP(f) != KO_ALL)
		return eval(f) == 0;
	{
		int s = (int)KT.aux[f];
		struct val *x = &env[nenv];
		const char *nm = KT.hint[f] ? sym_str(KT.hint[f]) : "x";
		int64_t v;
		int m, sp = 0, n = 0, cnt;
		const struct fgab *H = NULL;

		memset(x, 0, sizeof *x);
		x->sort = s;
		if (s == KS_EL) {
			nenv++;
			r = find_group(KA(f, 0), 0, &sp, &n);
			nenv--;
			if (!r || !(H = hn(sp, 2, n)) || H->ninv > 8)
				return 0;
		}
		cnt = s == KS_INT ? 5 : s == KS_EL ? 1 + 3 * H->ninv +
		    H->ninv * H->ninv : NSP;
		for (m = 0; m < cnt; m++) {
			size_t mark = out->n;
			if (s == KS_INT) {
				x->i = m;
				sb_printf(out, "%s = %d, ", nm, m);
			} else if (s == KS_EL) {
				int k, l;
				memset(x->y, 0, sizeof x->y);
				x->sp = sp;
				x->n = n;
				x->ny = H->ninv;
				if (m > 0 && m <= 3 * H->ninv) {
					k = (m - 1) % H->ninv;
					x->y[k] = (m - 1) / H->ninv + 1;
				} else if (m > 3 * H->ninv) {
					k = (m - 1 - 3 * H->ninv) / H->ninv;
					l = (m - 1 - 3 * H->ninv) % H->ninv;
					if (k >= l)
						continue;
					x->y[k] = x->y[l] = 1;
				}
				sb_printf(out, "%s = [", nm);
				for (k = 0; k < x->ny; k++)
					sb_printf(out, "%s%lld", k ? ", " : "",
					    (long long)x->y[k]);
				sb_printf(out, "] in H^%d(%s; Z) = %s, ", n,
				    spaces[sp], ab_str(H));
			} else {
				x->i = m;
				sb_printf(out, "%s = the cochains of %s, ", nm,
				    spaces[m]);
			}
			nenv++;
			r = search(KA(f, 0), out, depth + 1);
			nenv--;
			if (r)
				return 1;
			out->n = mark;
			if (out->p)
				out->p[mark] = 0;
			if (budget <= 0)
				return 0;
		}
		(void)v;
		return 0;
	}
}

int
pm_refute(kt phi, struct sbuf *out)
{
	int r;

	nenv = 0;
	budget = 200000;
	sb_clear(out);
	r = search(phi, out, 0);
	/* the computations' zints die with the command: drop the cache */
	{
		int i;
		for (i = 0; i < nbc; i++)
			if (bc[i].ok)
				hss_free(&bc[i].S);
		nbc = 0;
	}
	if (r && out->n >= 2)
		out->p[out->n -= 2] = 0;
	return r;
}
