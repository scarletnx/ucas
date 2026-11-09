/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * knorm.c - normal forms for the kernel's conversion rule.
 *
 * int terms: polynomials with int64 coefficients over atoms (variables,
 * applications, powers with a symbolic exponent), monomials as products
 * of atoms sorted by id.  el terms: linear combinations sum P_k * a_k of
 * el atoms with polynomial coefficients (a module over Z[atoms]); a
 * symbol flagged KSF_HOM is additive and Z-linear in its last argument,
 * so f(c, x + 2y) = f(c, x) + 2 f(c, y).  Relations: s = t becomes
 * N(s - t) = 0 with a positive leading coefficient, s <= t becomes
 * 0 <= N(t - s), s < t becomes 0 <= N(t - s - 1) (integers); a relation
 * between constants becomes true or false.  The justification is the
 * commutative ring axioms of Z, the module axioms of el, homomorphism
 * additivity and integer order on numerals.
 */
#include "ucas.h"
#include "prove.h"

#include <stdlib.h>
#include <string.h>

struct mono {
	kt m;
	int64_t c;
};
struct poly {
	struct mono *t;
	int n, cap;
};
struct lterm {
	kt a;
	struct poly c;
};
struct lin {
	struct lterm *t;
	int n, cap;
};

static kt *memo;
static uint32_t memocap;

void
kn_reset(void)
{
	if (memo)
		memset(memo, 0, memocap * sizeof *memo);
}

static int64_t
cadd(int64_t a, int64_t b)
{
	__int128 r = (__int128)a + b;

	if (r > INT64_MAX || r < -INT64_MAX)
		fail("integer overflow in a proof (coefficients beyond 64 bits)");
	return (int64_t)r;
}

static int64_t
cmul(int64_t a, int64_t b)
{
	__int128 r = (__int128)a * b;

	if (r > INT64_MAX || r < -INT64_MAX)
		fail("integer overflow in a proof (coefficients beyond 64 bits)");
	return (int64_t)r;
}

static void
pfree(struct poly *p)
{
	free(p->t);
	memset(p, 0, sizeof *p);
}

static void
ppush(struct poly *p, kt m, int64_t c)
{
	if (p->n == p->cap) {
		p->cap = p->cap ? p->cap * 2 : 4;
		p->t = xrealloc(p->t, (size_t)p->cap * sizeof *p->t);
	}
	p->t[p->n].m = m;
	p->t[p->n++].c = c;
}

static int
mcmp(const void *a, const void *b)
{
	const struct mono *x = a, *y = b;

	return x->m < y->m ? -1 : x->m > y->m;
}

/* sort and merge equal monomials, drop zeros */
static void
pcanon(struct poly *p)
{
	int i, k = 0;

	qsort(p->t, (size_t)p->n, sizeof *p->t, mcmp);
	for (i = 0; i < p->n; i++) {
		if (k > 0 && p->t[k - 1].m == p->t[i].m)
			p->t[k - 1].c = cadd(p->t[k - 1].c, p->t[i].c);
		else
			p->t[k++] = p->t[i];
	}
	p->n = k;
	for (i = k = 0; i < p->n; i++)
		if (p->t[i].c != 0)
			p->t[k++] = p->t[i];
	p->n = k;
}

static kt one(void) { return kt_num(1); }

/* the atoms of a monomial, in order */
static int
mflat(kt m, kt *out, int max)
{
	kt st[256];
	int n = 0, i;

	if (m == one())
		return 0;
	while (KOP(m) == KO_MUL && n < 256) {
		st[n++] = KA(m, 1);
		m = KA(m, 0);
	}
	if (n >= max)
		fail("monomial too large in a proof");
	out[0] = m;
	for (i = 0; i < n; i++)
		out[i + 1] = st[n - 1 - i];
	return n + 1;
}

static int
kcmp(const void *a, const void *b)
{
	kt x = *(const kt *)a, y = *(const kt *)b;

	return x < y ? -1 : x > y;
}

static kt
mbuild(kt *a, int n)
{
	kt m;
	int i;

	if (n == 0)
		return one();
	qsort(a, (size_t)n, sizeof *a, kcmp);
	m = a[0];
	for (i = 1; i < n; i++)
		m = kt_mul(m, a[i]);
	return m;
}

static kt
mmul(kt x, kt y)
{
	kt a[512];
	int n;

	if (x == one())
		return y;
	if (y == one())
		return x;
	n = mflat(x, a, 256);
	n += mflat(y, a + n, 256);
	return mbuild(a, n);
}

static struct poly
pconst(int64_t v)
{
	struct poly p = {0};

	if (v)
		ppush(&p, one(), v);
	return p;
}

static struct poly
patom(kt a)
{
	struct poly p = {0};

	ppush(&p, a, 1);
	return p;
}

static void
paddto(struct poly *r, const struct poly *b, int64_t s)
{
	int i;

	for (i = 0; i < b->n; i++)
		ppush(r, b->t[i].m, cmul(b->t[i].c, s));
	pcanon(r);
}

static struct poly
pmul(const struct poly *a, const struct poly *b)
{
	struct poly r = {0};
	int i, j;

	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++)
			ppush(&r, mmul(a->t[i].m, b->t[j].m),
			    cmul(a->t[i].c, b->t[j].c));
	pcanon(&r);
	return r;
}

static struct poly
pcopy(const struct poly *a)
{
	struct poly r = {0};
	int i;

	for (i = 0; i < a->n; i++)
		ppush(&r, a->t[i].m, a->t[i].c);
	return r;
}

static kt
pterm(const struct poly *p)
{
	kt r = KNONE, s;
	int i;

	for (i = 0; i < p->n; i++) {
		if (p->t[i].m == one())
			s = kt_num(p->t[i].c);
		else if (p->t[i].c == 1)
			s = p->t[i].m;
		else
			s = kt_mul(kt_num(p->t[i].c), p->t[i].m);
		r = r == KNONE ? s : kt_add(r, s);
	}
	return r == KNONE ? kt_num(0) : r;
}

static struct poly npoly(kt t);
static struct lin nlin(kt t);
static kt nterm(kt t);

/* an application with normalised arguments */
static kt
napp(kt t, int skiplast)
{
	kt x[16];
	int i;

	for (i = 0; i < KN(t); i++)
		x[i] = (skiplast && i == KN(t) - 1) ? KA(t, i) : nterm(KA(t, i));
	return kt_mk(KOP(t), KSORT(t), KT.aux[t], KT.num[t], x, KN(t));
}

static struct poly
npoly(kt t)
{
	struct poly a, b, r;
	int64_t e;

	switch (KOP(t)) {
	case KO_NUM:
		return pconst(KT.num[t]);
	case KO_ADD:
		a = npoly(KA(t, 0));
		b = npoly(KA(t, 1));
		paddto(&a, &b, 1);
		pfree(&b);
		return a;
	case KO_NEG:
		a = npoly(KA(t, 0));
		r = pconst(0);
		paddto(&r, &a, -1);
		pfree(&a);
		return r;
	case KO_MUL:
		a = npoly(KA(t, 0));
		b = npoly(KA(t, 1));
		r = pmul(&a, &b);
		pfree(&a);
		pfree(&b);
		return r;
	case KO_POW:
		a = npoly(KA(t, 0));
		b = npoly(KA(t, 1));
		if (b.n == 0 || (b.n == 1 && b.t[0].m == one() &&
		    b.t[0].c >= 0 && b.t[0].c <= 64)) {
			e = b.n ? b.t[0].c : 0;
			r = pconst(1);
			while (e-- > 0) {
				struct poly q = pmul(&r, &a);
				pfree(&r);
				r = q;
			}
			pfree(&a);
			pfree(&b);
			return r;
		}
		if (a.n == 1 && a.t[0].m == one() && a.t[0].c == 1) {
			pfree(&a);
			pfree(&b);
			return pconst(1);
		}
		{
			kt base = pterm(&a), ex = pterm(&b);
			pfree(&a);
			pfree(&b);
			return patom(kt_pow(base, ex));
		}
	case KO_APP:
		return patom(napp(t, 0));
	default:
		return patom(t);
	}
}

static void
lfree(struct lin *l)
{
	int i;

	for (i = 0; i < l->n; i++)
		pfree(&l->t[i].c);
	free(l->t);
	memset(l, 0, sizeof *l);
}

static void
lpush(struct lin *l, kt a, struct poly c)
{
	if (l->n == l->cap) {
		l->cap = l->cap ? l->cap * 2 : 4;
		l->t = xrealloc(l->t, (size_t)l->cap * sizeof *l->t);
	}
	l->t[l->n].a = a;
	l->t[l->n++].c = c;
}

static int
lcmp(const void *x, const void *y)
{
	const struct lterm *a = x, *b = y;

	return a->a < b->a ? -1 : a->a > b->a;
}

static void
lcanon(struct lin *l)
{
	int i, k = 0;

	qsort(l->t, (size_t)l->n, sizeof *l->t, lcmp);
	for (i = 0; i < l->n; i++) {
		if (k > 0 && l->t[k - 1].a == l->t[i].a) {
			paddto(&l->t[k - 1].c, &l->t[i].c, 1);
			pfree(&l->t[i].c);
		} else
			l->t[k++] = l->t[i];
	}
	l->n = k;
	for (i = k = 0; i < l->n; i++) {
		if (l->t[i].c.n == 0)
			pfree(&l->t[i].c);
		else
			l->t[k++] = l->t[i];
	}
	l->n = k;
}

static void
laddto(struct lin *r, const struct lin *b, const struct poly *s)
{
	int i;

	for (i = 0; i < b->n; i++)
		lpush(r, b->t[i].a, pmul(&b->t[i].c, s));
	lcanon(r);
}

static struct lin
nlin(kt t)
{
	struct lin a = {0}, b, r = {0};
	struct poly s;
	int i;

	switch (KOP(t)) {
	case KO_ZERO:
		return a;
	case KO_ADD:
		a = nlin(KA(t, 0));
		b = nlin(KA(t, 1));
		s = pconst(1);
		laddto(&a, &b, &s);
		pfree(&s);
		lfree(&b);
		return a;
	case KO_NEG:
		a = nlin(KA(t, 0));
		s = pconst(-1);
		laddto(&r, &a, &s);
		pfree(&s);
		lfree(&a);
		return r;
	case KO_MUL:
		s = npoly(KA(t, 0));
		b = nlin(KA(t, 1));
		laddto(&r, &b, &s);
		pfree(&s);
		lfree(&b);
		return r;
	case KO_APP:
		if (ksyms[KT.aux[t]].flags & KSF_HOM) {
			kt base = napp(t, 1), x[16];
			int n = KN(t), j;
			b = nlin(KA(t, n - 1));
			for (j = 0; j < n - 1; j++)
				x[j] = KA(base, j);
			for (i = 0; i < b.n; i++) {
				x[n - 1] = b.t[i].a;
				lpush(&r, kt_mk(KO_APP, KSORT(t), KT.aux[t], 0, x,
				    n), pcopy(&b.t[i].c));
			}
			lfree(&b);
			lcanon(&r);
			return r;
		}
		lpush(&a, napp(t, 0), pconst(1));
		return a;
	default:
		lpush(&a, t, pconst(1));
		return a;
	}
}

static kt
lterm_(const struct lin *l)
{
	kt r = KNONE, s, c;
	int i;

	for (i = 0; i < l->n; i++) {
		c = pterm(&l->t[i].c);
		s = c == one() ? l->t[i].a : kt_mul(c, l->t[i].a);
		r = r == KNONE ? s : kt_add(r, s);
	}
	return r == KNONE ? kt_zero() : r;
}

static kt
nterm(kt t)
{
	if (KSORT(t) == KS_INT) {
		struct poly p = npoly(t);
		kt r = pterm(&p);
		pfree(&p);
		return r;
	}
	if (KSORT(t) == KS_EL) {
		struct lin l = nlin(t);
		kt r = lterm_(&l);
		lfree(&l);
		return r;
	}
	if (KOP(t) == KO_APP)
		return napp(t, 0);
	return t;
}

static kt
nrel(kt t)
{
	int op = KOP(t);
	kt a = KA(t, 0), b = KA(t, 1);

	if (op == KO_EQ && KSORT(a) == KS_EL) {
		struct lin l = nlin(a), m = nlin(b);
		struct poly s = pconst(-1);
		kt r;
		laddto(&l, &m, &s);
		pfree(&s);
		lfree(&m);
		if (l.n == 0) {
			lfree(&l);
			return kf_true();
		}
		if (l.t[0].c.t[0].c < 0) {
			struct lin q = {0};
			s = pconst(-1);
			laddto(&q, &l, &s);
			pfree(&s);
			lfree(&l);
			l = q;
		}
		r = kf_eq(lterm_(&l), kt_zero());
		lfree(&l);
		return r;
	}
	if (KSORT(a) == KS_INT) {
		struct poly p, q;
		kt r;
		if (op == KO_EQ) {
			p = npoly(a);
			q = npoly(b);
			paddto(&p, &q, -1);
		} else {
			p = npoly(b);
			q = npoly(a);
			paddto(&p, &q, -1);
			if (op == KO_LT) {
				struct poly m1 = pconst(-1);
				paddto(&p, &m1, 1);
				pfree(&m1);
			}
		}
		pfree(&q);
		if (p.n == 0 || (p.n == 1 && p.t[0].m == one())) {
			int64_t v = p.n ? p.t[0].c : 0;
			pfree(&p);
			return (op == KO_EQ ? v == 0 : v >= 0) ? kf_true() :
			    kf_false();
		}
		if (op == KO_EQ && p.t[0].c < 0) {
			struct poly z = pconst(0);
			paddto(&z, &p, -1);
			pfree(&p);
			p = z;
		}
		r = pterm(&p);
		pfree(&p);
		return op == KO_EQ ? kf_eq(r, kt_num(0)) : kf_le(kt_num(0), r);
	}
	a = nterm(a);
	b = nterm(b);
	if (op == KO_EQ) {
		if (a == b)
			return kf_true();
		return a < b ? kf_eq(a, b) : kf_eq(b, a);
	}
	return kt_mk(op, KS_PROP, 0, 0, (kt[]){a, b}, 2);
}

kt
kn_norm(kt t)
{
	kt r, x[16];
	int i;

	if (t >= memocap) {
		uint32_t c = memocap ? memocap : 4096;
		while (c <= t)
			c *= 2;
		memo = xrealloc(memo, c * sizeof *memo);
		memset(memo + memocap, 0, (c - memocap) * sizeof *memo);
		memocap = c;
	}
	if (memo[t])
		return memo[t];
	switch (KOP(t)) {
	case KO_EQ:
	case KO_LE:
	case KO_LT:
		r = nrel(t);
		break;
	case KO_IN:
	case KO_PRED:
		for (i = 0; i < KN(t); i++)
			x[i] = nterm(KA(t, i));
		r = kt_mk(KOP(t), KS_PROP, KT.aux[t], 0, x, KN(t));
		break;
	case KO_NOT:
	case KO_AND:
	case KO_OR:
	case KO_IMP:
	case KO_IFF:
	case KO_ALL:
	case KO_EX:
		for (i = 0; i < KN(t); i++)
			x[i] = kn_norm(KA(t, i));
		r = kt_mk(KOP(t), KS_PROP, KT.aux[t], 0, x, KN(t));
		if (!KT.hint[r])
			KT.hint[r] = KT.hint[t];
		break;
	default:
		r = KSORT(t) == KS_PROP ? t : nterm(t);
	}
	if (t < memocap)	/* a reset may not shrink it; keep it simple */
		memo[t] = r;
	return r;
}

int
kn_equiv(kt a, kt b)
{
	return a == b || kn_norm(a) == kn_norm(b);
}

int
kn_poly_const(kt t, int64_t *v)
{
	struct poly p = npoly(t);
	int ok = p.n == 0 || (p.n == 1 && p.t[0].m == one());

	if (ok)
		*v = p.n ? p.t[0].c : 0;
	pfree(&p);
	return ok;
}

/* the summands of an int or el term's normal form: coef[i]*atom[i]; for
 * int, atom is a monomial (the numeral 1 for the constant) and coef a
 * numeral; for el, coef is a normalised int term.  -1 if too many. */
int
kn_split(kt t, kt *atom, kt *coef, int max)
{
	int i, n;

	if (KSORT(t) == KS_INT) {
		struct poly p = npoly(t);
		n = p.n;
		for (i = 0; i < p.n && i < max; i++) {
			atom[i] = p.t[i].m;
			coef[i] = kt_num(p.t[i].c);
		}
		pfree(&p);
		return n > max ? -1 : n;
	}
	{
		struct lin l = nlin(t);
		n = l.n;
		for (i = 0; i < l.n && i < max; i++) {
			atom[i] = l.t[i].a;
			coef[i] = pterm(&l.t[i].c);
		}
		lfree(&l);
		return n > max ? -1 : n;
	}
}
