/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * kcheck.c - an independent re-checker for kernel proofs.
 *
 * It shares only the term table (hash-consed ids) and the axiom list
 * with the kernel.  Everything else is re-done here, written separately:
 * substitution and abstraction, the hypothesis sets, the side
 * conditions of every rule, and the normal forms behind conv and lin,
 * which are computed as canonical strings (monomials and atoms sorted
 * as text) rather than knorm.c's id-ordered term DAG.  The theorems a
 * proof depends on are replayed in id order (premises always have
 * smaller ids); each must re-derive exactly its stored sequent.
 */
#include "ucas.h"
#include "prove.h"

#include <stdlib.h>
#include <string.h>

/* ---- substitution, written again ---- */

static kt
c_mk(kt t, const kt *x)
{
	return kt_mk(KOP(t), KSORT(t), KT.aux[t], KT.num[t], x, KN(t));
}

static int
binder(kt t)
{
	return KOP(t) == KO_ALL || KOP(t) == KO_EX;
}

static kt
c_shift(kt s, uint32_t by, uint32_t above)
{
	kt x[16];
	int i;

	if (KOP(s) == KO_BVAR)
		return KT.aux[s] >= above ? kt_bvar(KT.aux[s] + by, KSORT(s)) : s;
	if (KN(s) == 0)
		return s;
	for (i = 0; i < KN(s); i++)
		x[i] = c_shift(KA(s, i), by, above + binder(s));
	return c_mk(s, x);
}

static kt
c_inst(kt t, kt s, uint32_t d)
{
	kt x[16];
	int i;

	if (KOP(t) == KO_BVAR) {
		if (KT.aux[t] == d)
			return c_shift(s, d, 0);
		return KT.aux[t] > d ? kt_bvar(KT.aux[t] - 1, KSORT(t)) : t;
	}
	if (KN(t) == 0)
		return t;
	for (i = 0; i < KN(t); i++)
		x[i] = c_inst(KA(t, i), s, d + binder(t));
	return c_mk(t, x);
}

static kt
c_abs(kt t, kt v, uint32_t d)
{
	kt x[16];
	int i;

	if (t == v)
		return kt_bvar(d, KSORT(v));
	if (KN(t) == 0)
		return t;
	for (i = 0; i < KN(t); i++)
		x[i] = c_abs(KA(t, i), v, d + binder(t));
	return c_mk(t, x);
}

static int
c_occ(kt v, kt t)
{
	int i;

	if (t == v)
		return 1;
	for (i = 0; i < KN(t); i++)
		if (c_occ(v, KA(t, i)))
			return 1;
	return 0;
}

static int
c_closed(kt t, uint32_t d)
{
	int i;

	if (KOP(t) == KO_BVAR)
		return KT.aux[t] < d;
	if (KOP(t) == KO_META)
		return 0;
	for (i = 0; i < KN(t); i++)
		if (!c_closed(KA(t, i), d + binder(t)))
			return 0;
	return 1;
}

/* ---- canonical strings ---- */

static uint32_t now;	/* the theorem being checked: homs declared before */

static int
ishom(uint32_t sym)
{
	return (ksyms[sym].flags & KSF_HOM) &&
	    (ksyms[sym].homthm == 0 || ksyms[sym].homthm < now);
}

struct cm {
	char *m;	/* atoms sorted, joined by \1 ("" = 1) */
	long long c;
};
struct cp {
	struct cm *t;
	int n;
};
struct ce {
	char *a;
	struct cp c;
};
struct cl {
	struct ce *t;
	int n;
};

static int cbad;	/* overflow */

/* Use the exact same bound as knorm.c's cadd()/cmul(): a coefficient must
 * fit in [-INT64_MAX, INT64_MAX], checked with __int128 so no precision is
 * lost near the 64-bit edge.  The old long-double 9.2e18 guard rejected
 * coefficients in (9.2e18, INT64_MAX] that knorm.c accepts, so a proof the
 * prover built could be rejected by its own independent re-check. */
static long long
ovadd(long long a, long long b)
{
	__int128 r = (__int128)a + b;

	if (r > INT64_MAX || r < -INT64_MAX) {
		cbad = 1;
		return 0;
	}
	return (long long)r;
}

static long long
ovmul(long long a, long long b)
{
	__int128 r = (__int128)a * b;

	if (r > INT64_MAX || r < -INT64_MAX) {
		cbad = 1;
		return 0;
	}
	return (long long)r;
}

static char *
sdup(const char *s)
{
	size_t n = strlen(s) + 1;
	char *r = xmalloc(n);

	memcpy(r, s, n);
	return r;
}

static int
cmcmp(const void *a, const void *b)
{
	return strcmp(((const struct cm *)a)->m, ((const struct cm *)b)->m);
}

static void
cp_norm(struct cp *p)
{
	int i, k = 0;

	xsort(p->t, (size_t)p->n, sizeof *p->t, cmcmp);
	for (i = 0; i < p->n; i++) {
		if (k && !strcmp(p->t[k - 1].m, p->t[i].m)) {
			p->t[k - 1].c = ovadd(p->t[k - 1].c, p->t[i].c);
			free(p->t[i].m);
		} else
			p->t[k++] = p->t[i];
	}
	p->n = k;
	for (i = k = 0; i < p->n; i++) {
		if (p->t[i].c)
			p->t[k++] = p->t[i];
		else
			free(p->t[i].m);
	}
	p->n = k;
}

static void
cp_add1(struct cp *p, const char *m, long long c)
{
	p->t = xrealloc(p->t, (size_t)(p->n + 1) * sizeof *p->t);
	p->t[p->n].m = sdup(m);
	p->t[p->n++].c = c;
}

static void
cp_free(struct cp *p)
{
	int i;

	for (i = 0; i < p->n; i++)
		free(p->t[i].m);
	free(p->t);
	p->t = NULL;
	p->n = 0;
}

static int
scmp(const void *a, const void *b)
{
	return strcmp(*(char *const *)a, *(char *const *)b);
}

static char *
mono_mul(const char *a, const char *b)
{
	char *buf, *parts[1024], *s;
	int n = 0, i;
	size_t len = strlen(a) + strlen(b) + 2;
	struct sbuf o = {0};

	buf = xmalloc(len);
	snprintf(buf, len, "%s%s%s", a, *a && *b ? "\1" : "", b);
	if (!*buf)
		return buf;
	for (s = buf; n < 1024;) {
		parts[n++] = s;
		s = strchr(s, '\1');
		if (!s)
			break;
		*s++ = 0;
	}
	xsort(parts, (size_t)n, sizeof *parts, scmp);
	for (i = 0; i < n; i++) {
		if (i)
			sb_putc(&o, '\1');
		sb_puts(&o, parts[i]);
	}
	free(buf);
	return o.p ? o.p : sdup("");
}

static struct cp
cp_mul(const struct cp *a, const struct cp *b)
{
	struct cp r = {0};
	int i, j;

	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			char *m = mono_mul(a->t[i].m, b->t[j].m);
			cp_add1(&r, m, ovmul(a->t[i].c, b->t[j].c));
			free(m);
		}
	cp_norm(&r);
	return r;
}

static void
cp_addto(struct cp *r, const struct cp *b, long long s)
{
	int i;

	for (i = 0; i < b->n; i++)
		cp_add1(r, b->t[i].m, ovmul(b->t[i].c, s));
	cp_norm(r);
}

static char *cs_term(kt t);

static void
cp_str(struct sbuf *o, const struct cp *p)
{
	int i;

	if (!p->n)
		sb_puts(o, "0");
	for (i = 0; i < p->n; i++)
		sb_printf(o, "%+lld{%s}", p->t[i].c, p->t[i].m);
}

static struct cp cpoly(kt t);

static char *
c_app(kt t, int skiplast)
{
	struct sbuf o = {0};
	int i;

	sb_printf(&o, "%c%s(", KOP(t) == KO_APP ? 'A' : 'P',
	    sym_str(ksyms[KT.aux[t]].name));
	for (i = 0; i < KN(t) - skiplast; i++) {
		char *s = cs_term(KA(t, i));
		sb_puts(&o, s);
		sb_putc(&o, ',');
		free(s);
	}
	return o.p;
}

static char *
c_atom(kt t)
{
	struct sbuf o = {0};

	switch (KOP(t)) {
	case KO_VAR:
		sb_printf(&o, "V%s", sym_str(KT.aux[t]));
		break;
	case KO_BVAR:
		sb_printf(&o, "B%u", KT.aux[t]);
		break;
	case KO_APP: {
		char *s = c_app(t, 0);
		sb_puts(&o, s);
		sb_putc(&o, ')');
		free(s);
		break;
	}
	default:
		sb_printf(&o, "?%u", t);
	}
	return o.p;
}

static struct cp
cpoly(kt t)
{
	struct cp a, b, r = {0};
	long long e;

	switch (KOP(t)) {
	case KO_NUM:
		if (KT.num[t])
			cp_add1(&r, "", KT.num[t]);
		return r;
	case KO_ADD:
		a = cpoly(KA(t, 0));
		b = cpoly(KA(t, 1));
		cp_addto(&a, &b, 1);
		cp_free(&b);
		return a;
	case KO_NEG:
		a = cpoly(KA(t, 0));
		cp_addto(&r, &a, -1);
		cp_free(&a);
		return r;
	case KO_MUL:
		a = cpoly(KA(t, 0));
		b = cpoly(KA(t, 1));
		r = cp_mul(&a, &b);
		cp_free(&a);
		cp_free(&b);
		return r;
	case KO_POW:
		a = cpoly(KA(t, 0));
		b = cpoly(KA(t, 1));
		if (b.n == 0 || (b.n == 1 && !*b.t[0].m && b.t[0].c >= 0 &&
		    b.t[0].c <= 64)) {
			e = b.n ? b.t[0].c : 0;
			cp_add1(&r, "", 1);
			while (e-- > 0) {
				struct cp q = cp_mul(&r, &a);
				cp_free(&r);
				r = q;
			}
		} else if (a.n == 1 && !*a.t[0].m && a.t[0].c == 1) {
			cp_add1(&r, "", 1);
		} else {
			struct sbuf o = {0};
			sb_puts(&o, "W(");
			cp_str(&o, &a);
			sb_puts(&o, ")^(");
			cp_str(&o, &b);
			sb_puts(&o, ")");
			cp_add1(&r, o.p, 1);
			free(o.p);
		}
		cp_free(&a);
		cp_free(&b);
		return r;
	default: {
		char *s = c_atom(t);
		cp_add1(&r, s, 1);
		free(s);
		return r;
	}
	}
}

static int
cecmp(const void *a, const void *b)
{
	return strcmp(((const struct ce *)a)->a, ((const struct ce *)b)->a);
}

static void
cl_free(struct cl *l)
{
	int i;

	for (i = 0; i < l->n; i++) {
		free(l->t[i].a);
		cp_free(&l->t[i].c);
	}
	free(l->t);
	l->t = NULL;
	l->n = 0;
}

static void
cl_push(struct cl *l, char *a, struct cp c)
{
	l->t = xrealloc(l->t, (size_t)(l->n + 1) * sizeof *l->t);
	l->t[l->n].a = a;
	l->t[l->n++].c = c;
}

static void
cl_norm(struct cl *l)
{
	int i, k = 0;

	xsort(l->t, (size_t)l->n, sizeof *l->t, cecmp);
	for (i = 0; i < l->n; i++) {
		if (k && !strcmp(l->t[k - 1].a, l->t[i].a)) {
			cp_addto(&l->t[k - 1].c, &l->t[i].c, 1);
			free(l->t[i].a);
			cp_free(&l->t[i].c);
		} else
			l->t[k++] = l->t[i];
	}
	l->n = k;
	for (i = k = 0; i < l->n; i++) {
		if (l->t[i].c.n)
			l->t[k++] = l->t[i];
		else {
			free(l->t[i].a);
			cp_free(&l->t[i].c);
		}
	}
	l->n = k;
}

static void
cl_addto(struct cl *r, const struct cl *b, const struct cp *s)
{
	int i;

	for (i = 0; i < b->n; i++)
		cl_push(r, sdup(b->t[i].a), cp_mul(&b->t[i].c, s));
	cl_norm(r);
}

static struct cl
clin(kt t)
{
	struct cl a = {0}, b, r = {0};
	struct cp s = {0};
	int i;

	switch (KOP(t)) {
	case KO_ZERO:
		return a;
	case KO_ADD:
		a = clin(KA(t, 0));
		b = clin(KA(t, 1));
		cp_add1(&s, "", 1);
		cl_addto(&a, &b, &s);
		cp_free(&s);
		cl_free(&b);
		return a;
	case KO_NEG:
		a = clin(KA(t, 0));
		cp_add1(&s, "", -1);
		cl_addto(&r, &a, &s);
		cp_free(&s);
		cl_free(&a);
		return r;
	case KO_MUL:
		s = cpoly(KA(t, 0));
		b = clin(KA(t, 1));
		cl_addto(&r, &b, &s);
		cp_free(&s);
		cl_free(&b);
		return r;
	case KO_APP:
		if (ishom(KT.aux[t])) {
			char *pre = c_app(t, 1);
			b = clin(KA(t, KN(t) - 1));
			for (i = 0; i < b.n; i++) {
				struct sbuf o = {0};
				struct cp one = {0};
				sb_puts(&o, pre);
				sb_printf(&o, "L[%s])", b.t[i].a);
				cp_add1(&one, "", 1);
				cl_push(&r, o.p, cp_mul(&b.t[i].c, &one));
				cp_free(&one);
			}
			free(pre);
			cl_free(&b);
			cl_norm(&r);
			return r;
		}
		/* FALLTHROUGH */
	default: {
		struct cp one = {0};
		cp_add1(&one, "", 1);
		cl_push(&a, c_atom(t), one);
		return a;
	}
	}
}

static void
cl_str(struct sbuf *o, const struct cl *l)
{
	int i;

	if (!l->n)
		sb_puts(o, "0");
	for (i = 0; i < l->n; i++) {
		sb_puts(o, "+[");
		cp_str(o, &l->t[i].c);
		sb_printf(o, "]%s", l->t[i].a);
	}
}

static char *
cs_term(kt t)
{
	struct sbuf o = {0};

	if (KSORT(t) == KS_INT) {
		struct cp p = cpoly(t);
		sb_puts(&o, "I:");
		cp_str(&o, &p);
		cp_free(&p);
	} else if (KSORT(t) == KS_EL) {
		struct cl l = clin(t);
		sb_puts(&o, "E:");
		cl_str(&o, &l);
		cl_free(&l);
	} else {
		char *s = c_atom(t);
		sb_puts(&o, s);
		free(s);
	}
	return o.p;
}

static char *cs_formula(kt t);

static char *
cs_rel(kt t)
{
	struct sbuf o = {0};
	kt a = KA(t, 0), b = KA(t, 1);
	int op = KOP(t);

	if (op == KO_EQ && KSORT(a) == KS_EL) {
		struct cl l = clin(a), m = clin(b);
		struct cp s = {0};
		cp_add1(&s, "", -1);
		cl_addto(&l, &m, &s);
		if (l.n == 0)
			sb_puts(&o, "T");
		else {
			if (l.t[0].c.t[0].c < 0) {
				struct cl q = {0};
				cl_addto(&q, &l, &s);
				cl_free(&l);
				l = q;
			}
			sb_puts(&o, "Z(");
			cl_str(&o, &l);
			sb_puts(&o, ")");
		}
		cp_free(&s);
		cl_free(&l);
		cl_free(&m);
		return o.p;
	}
	if (KSORT(a) == KS_INT) {
		struct cp p, q;
		if (op == KO_EQ) {
			p = cpoly(a);
			q = cpoly(b);
		} else {
			p = cpoly(b);
			q = cpoly(a);
		}
		cp_addto(&p, &q, -1);
		if (op == KO_LT) {
			struct cp m = {0};
			cp_add1(&m, "", -1);
			cp_addto(&p, &m, 1);
			cp_free(&m);
		}
		if (p.n == 0 || (p.n == 1 && !*p.t[0].m)) {
			long long v = p.n ? p.t[0].c : 0;
			sb_puts(&o, (op == KO_EQ ? v == 0 : v >= 0) ? "T" : "F");
		} else {
			if (op == KO_EQ && p.t[0].c < 0) {
				struct cp z = {0};
				cp_addto(&z, &p, -1);
				cp_free(&p);
				p = z;
			}
			sb_puts(&o, op == KO_EQ ? "Z(" : "G(");
			cp_str(&o, &p);
			sb_puts(&o, ")");
		}
		cp_free(&p);
		cp_free(&q);
		return o.p;
	}
	{
		char *x = cs_term(a), *y = cs_term(b);
		if (op == KO_EQ) {
			if (!strcmp(x, y))
				sb_puts(&o, "T");
			else if (strcmp(x, y) < 0)
				sb_printf(&o, "Q(%s,%s)", x, y);
			else
				sb_printf(&o, "Q(%s,%s)", y, x);
		} else
			sb_printf(&o, "R%d(%s,%s)", op, x, y);
		free(x);
		free(y);
	}
	return o.p;
}

static char **memo;
static uint32_t memon;

static char *
cs_formula(kt t)
{
	struct sbuf o = {0};
	int i;

	if (t < memon && memo[t])
		return sdup(memo[t]);
	switch (KOP(t)) {
	case KO_TRUE:
		sb_puts(&o, "T");
		break;
	case KO_FALSE:
		sb_puts(&o, "F");
		break;
	case KO_EQ:
	case KO_LE:
	case KO_LT:
		free(o.p);
		o.p = cs_rel(t);
		break;
	case KO_IN:
	case KO_PRED:
		sb_printf(&o, "%c%u(", KOP(t) == KO_IN ? 'M' : 'P', KT.aux[t]);
		for (i = 0; i < KN(t); i++) {
			char *s = cs_term(KA(t, i));
			sb_puts(&o, s);
			sb_putc(&o, ';');
			free(s);
		}
		sb_putc(&o, ')');
		break;
	default:
		sb_printf(&o, "O%d.%u(", KOP(t), binder(t) ? KT.aux[t] : 0);
		for (i = 0; i < KN(t); i++) {
			char *s = cs_formula(KA(t, i));
			sb_puts(&o, s);
			sb_putc(&o, ';');
			free(s);
		}
		sb_putc(&o, ')');
	}
	if (t >= memon) {
		uint32_t c = memon ? memon : 1024;
		while (c <= t)
			c *= 2;
		memo = xrealloc(memo, c * sizeof *memo);
		memset(memo + memon, 0, (c - memon) * sizeof *memo);
		memon = c;
	}
	memo[t] = sdup(o.p ? o.p : "");
	return o.p ? o.p : sdup("");
}

static void
memo_clear(void)
{
	uint32_t i;

	for (i = 0; i < memon; i++) {
		free(memo[i]);
		memo[i] = NULL;
	}
}

static int
same(kt a, kt b)
{
	char *x, *y;
	int r;

	if (a == b)
		return 1;
	x = cs_formula(a);
	y = cs_formula(b);
	r = !strcmp(x, y);
	free(x);
	free(y);
	return r;
}

/* ---- hypothesis sets ---- */

struct hset {
	kt *h;
	int n;
};

static void
hs_put(struct hset *s, kt h)
{
	int i;

	for (i = 0; i < s->n; i++)
		if (s->h[i] == h)
			return;
	s->h = xrealloc(s->h, (size_t)(s->n + 1) * sizeof *s->h);
	s->h[s->n++] = h;
}

static void
hs_of(struct hset *s, kthm th, kt except)
{
	uint32_t i;

	for (i = 0; i < kthm_nhyps(th); i++)
		if (kthm_hyp(th, i) != except)
			hs_put(s, kthm_hyp(th, i));
}

static int
hs_eq(struct hset *s, kthm th)
{
	uint32_t i;
	int j, found;

	if ((uint32_t)s->n != kthm_nhyps(th))
		return 0;
	for (i = 0; i < kthm_nhyps(th); i++) {
		found = 0;
		for (j = 0; j < s->n; j++)
			if (s->h[j] == kthm_hyp(th, i))
				found = 1;
		if (!found)
			return 0;
	}
	return 1;
}

static int
hyps_occ(kthm th, kt v, kt except)
{
	uint32_t i;

	for (i = 0; i < kthm_nhyps(th); i++)
		if (kthm_hyp(th, i) != except && c_occ(v, kthm_hyp(th, i)))
			return 1;
	return 0;
}

static kt
relterm(kt c, int *ge)
{
	if (KOP(c) == KO_EQ && (KSORT(KA(c, 0)) == KS_INT ||
	    KSORT(KA(c, 0)) == KS_EL)) {
		*ge = 0;
		return kt_add(KA(c, 0), kt_neg(KA(c, 1)));
	}
	if (KOP(c) == KO_LE) {
		*ge = 1;
		return kt_add(KA(c, 1), kt_neg(KA(c, 0)));
	}
	if (KOP(c) == KO_LT) {
		*ge = 1;
		return kt_add(kt_add(KA(c, 1), kt_neg(KA(c, 0))), kt_num(-1));
	}
	return KNONE;
}

static kt
c_homlaw(int sym, int which)
{
	struct ksym *s = &ksyms[sym];
	int k = s->arity - 1, i;
	kt a[8], f;

	for (i = 0; i < k; i++)
		a[i] = kt_bvar((uint32_t)(k + 1 - i), s->arg[i]);
	if (!which) {
		kt x = kt_bvar(1, KS_EL), y = kt_bvar(0, KS_EL), l, r;
		a[k] = kt_add(x, y);
		l = kt_app(sym, a, k + 1);
		a[k] = x;
		r = kt_app(sym, a, k + 1);
		a[k] = y;
		f = kf_eq(l, kt_add(r, kt_app(sym, a, k + 1)));
		f = kf_bind(KO_ALL, KS_EL, 0, kf_bind(KO_ALL, KS_EL, 0, f));
	} else {
		kt n = kt_bvar(1, KS_INT), x = kt_bvar(0, KS_EL), l;
		a[k] = kt_mul(n, x);
		l = kt_app(sym, a, k + 1);
		a[k] = x;
		f = kf_eq(l, kt_mul(n, kt_app(sym, a, k + 1)));
		f = kf_bind(KO_ALL, KS_INT, 0, kf_bind(KO_ALL, KS_EL, 0, f));
	}
	for (i = k - 1; i >= 0; i--)
		f = kf_bind(KO_ALL, s->arg[i], 0, f);
	return f;
}

static const char *why;

static int
step(kthm th)
{
	int r = kthm_rule(th), ge, i;
	kt c = kthm_concl(th), x, y, v, s;
	kthm p0 = kthm_nprem(th) > 0 ? kthm_prem(th, 0) : 0;
	kthm p1 = kthm_nprem(th) > 1 ? kthm_prem(th, 1) : 0;
	kthm p2 = kthm_nprem(th) > 2 ? kthm_prem(th, 2) : 0;
	struct hset h = {0};
	int ok = 0;

	for (i = 0; i < (int)kthm_nprem(th); i++)
		if (kthm_prem(th, (uint32_t)i) >= th) {
			why = "a premise is not older";
			return 0;
		}
	if (!c_closed(c, 0)) {
		why = "the conclusion is not closed";
		return 0;
	}
	switch (r) {
	case KR_ASSUME:
		x = kthm_arg(th, 0);
		hs_put(&h, x);
		ok = c == x;
		break;
	case KR_AXIOM:
		ok = kthm_aux(th) < nkaxioms && c == kaxioms[kthm_aux(th)].phi;
		break;
	case KR_IMPI:
		x = kthm_arg(th, 0);
		hs_of(&h, p0, x);
		ok = c == kf_imp(x, kthm_concl(p0));
		break;
	case KR_IMPE:
		x = kthm_concl(p0);
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KNONE);
		ok = KOP(x) == KO_IMP && KA(x, 0) == kthm_concl(p1) &&
		    c == KA(x, 1);
		break;
	case KR_ANDI:
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KNONE);
		ok = c == kf_and(kthm_concl(p0), kthm_concl(p1));
		break;
	case KR_ANDE1:
	case KR_ANDE2:
		x = kthm_concl(p0);
		hs_of(&h, p0, KNONE);
		ok = KOP(x) == KO_AND && c == KA(x, r == KR_ANDE2);
		break;
	case KR_ORI1:
		hs_of(&h, p0, KNONE);
		ok = c == kf_or(kthm_concl(p0), kthm_arg(th, 0));
		break;
	case KR_ORI2:
		hs_of(&h, p0, KNONE);
		ok = c == kf_or(kthm_arg(th, 0), kthm_concl(p0));
		break;
	case KR_ORE:
		x = kthm_concl(p0);
		if (KOP(x) != KO_OR)
			break;
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KA(x, 0));
		hs_of(&h, p2, KA(x, 1));
		ok = c == kthm_concl(p1) && c == kthm_concl(p2);
		break;
	case KR_FALSEE:
		hs_of(&h, p0, KNONE);
		ok = KOP(kthm_concl(p0)) == KO_FALSE && c == kthm_arg(th, 0);
		break;
	case KR_NOTI:
		x = kthm_arg(th, 0);
		hs_of(&h, p0, x);
		ok = KOP(kthm_concl(p0)) == KO_FALSE && c == kf_not(x);
		break;
	case KR_NOTE:
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KNONE);
		ok = KOP(c) == KO_FALSE && kthm_concl(p0) ==
		    kf_not(kthm_concl(p1));
		break;
	case KR_RAA:
		x = kthm_arg(th, 0);
		hs_of(&h, p0, kf_not(x));
		ok = KOP(kthm_concl(p0)) == KO_FALSE && c == x;
		break;
	case KR_IFFI:
		x = kthm_concl(p0);
		y = kthm_concl(p1);
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KNONE);
		ok = KOP(x) == KO_IMP && y == kf_imp(KA(x, 1), KA(x, 0)) &&
		    c == kf_iff(KA(x, 0), KA(x, 1));
		break;
	case KR_IFFE1:
	case KR_IFFE2:
		x = kthm_concl(p0);
		hs_of(&h, p0, KNONE);
		ok = KOP(x) == KO_IFF && c == (r == KR_IFFE1 ?
		    kf_imp(KA(x, 0), KA(x, 1)) : kf_imp(KA(x, 1), KA(x, 0)));
		break;
	case KR_ALLI:
		v = kthm_arg(th, 0);
		hs_of(&h, p0, KNONE);
		ok = KOP(v) == KO_VAR && !hyps_occ(p0, v, KNONE) &&
		    KOP(c) == KO_ALL && KT.aux[c] == KSORT(v) &&
		    KA(c, 0) == c_abs(kthm_concl(p0), v, 0);
		break;
	case KR_ALLE:
		x = kthm_concl(p0);
		v = kthm_arg(th, 0);
		hs_of(&h, p0, KNONE);
		ok = KOP(x) == KO_ALL && KSORT(v) == KT.aux[x] &&
		    c_closed(v, 0) && c == c_inst(KA(x, 0), v, 0);
		break;
	case KR_EXI:
		x = kthm_arg(th, 0);
		v = kthm_arg(th, 1);
		hs_of(&h, p0, KNONE);
		ok = c == x && KOP(x) == KO_EX && KSORT(v) == KT.aux[x] &&
		    c_closed(v, 0) && kthm_concl(p0) == c_inst(KA(x, 0), v, 0);
		break;
	case KR_EXE:
		x = kthm_concl(p0);
		v = kthm_arg(th, 0);
		if (KOP(x) != KO_EX || KOP(v) != KO_VAR || KSORT(v) != KT.aux[x])
			break;
		y = c_inst(KA(x, 0), v, 0);
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, y);
		ok = c == kthm_concl(p1) && !c_occ(v, x) && !c_occ(v, c) &&
		    !hyps_occ(p0, v, KNONE) && !hyps_occ(p1, v, y);
		break;
	case KR_REFL:
		x = kthm_arg(th, 0);
		ok = c_closed(x, 0) && c == kf_eq(x, x);
		break;
	case KR_SUBST:
		x = kthm_concl(p0);
		s = kthm_arg(th, 0);
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KNONE);
		ok = KOP(x) == KO_EQ && c_closed(s, 1) &&
		    kthm_concl(p1) == c_inst(s, KA(x, 0), 0) &&
		    c == c_inst(s, KA(x, 1), 0);
		break;
	case KR_CONV:
		hs_of(&h, p0, KNONE);
		ok = c == kthm_arg(th, 0) && same(kthm_concl(p0), c);
		break;
	case KR_LIN: {
		kt sum = KNONE, t;
		int anyge = 0, sort = -1;
		uint32_t k;
		for (k = 0; k < kthm_nprem(th); k++) {
			kthm q = kthm_prem(th, k);
			kt co = kthm_arg(th, k);
			hs_of(&h, q, KNONE);
			t = relterm(kthm_concl(q), &ge);
			if (t == KNONE || KSORT(co) != KS_INT || !c_closed(co, 0))
				goto out;
			if (sort < 0)
				sort = KSORT(t);
			if (KSORT(t) != sort)
				goto out;
			if (ge) {
				struct cp cp = cpoly(co);
				int okc = cp.n == 0 || (cp.n == 1 && !*cp.t[0].m &&
				    cp.t[0].c >= 0);
				cp_free(&cp);
				if (!okc)
					goto out;
				anyge = 1;
			}
			t = kt_mul(co, t);
			sum = sum == KNONE ? t : kt_add(sum, t);
		}
		if (sum == KNONE)
			break;
		x = anyge ? kf_le(kt_num(0), sum) : kf_eq(sum, sort == KS_EL ?
		    kt_zero() : kt_num(0));
		ok = KOP(c) == KOP(x) && same(c, x);
		break;
	}
	case KR_INDUCT: {
		kt base = kthm_arg(th, 0), pat = kthm_arg(th, 1);
		kt m = kt_bvar(0, KS_INT);
		hs_of(&h, p0, KNONE);
		hs_of(&h, p1, KNONE);
		if (!c_closed(base, 0) || !c_closed(pat, 1))
			break;
		x = kf_bind(KO_ALL, KS_INT, 0, kf_imp(kf_and(kf_le(base, m),
		    pat), c_inst(pat, kt_add(m, kt_num(1)), 0)));
		ok = kthm_concl(p0) == c_inst(pat, base, 0) &&
		    kthm_concl(p1) == x && c == kf_bind(KO_ALL, KS_INT, 0,
		    kf_imp(kf_le(base, m), pat));
		break;
	}
	case KR_HOMDECL: {
		uint32_t sym = kthm_aux(th);
		ok = sym > 0 && sym < nksyms && ksyms[sym].homthm == th &&
		    kthm_nhyps(p0) == 0 && kthm_nhyps(p1) == 0 &&
		    kthm_concl(p0) == c_homlaw((int)sym, 0) &&
		    kthm_concl(p1) == c_homlaw((int)sym, 1) && KOP(c) == KO_TRUE;
		memo_clear();	/* the normal forms change from here on */
		break;
	}
	default:
		why = "unknown rule";
	}
	if (ok && !hs_eq(&h, th)) {
		why = "the hypotheses differ";
		ok = 0;
	} else if (!ok && !why)
		why = "the conclusion does not follow";
out:
	free(h.h);
	if (cbad) {
		why = "integer overflow";
		cbad = 0;
		ok = 0;
	}
	return ok;
}

int
kcheck(kthm top, char *msg, int n, uint32_t *nsteps)
{
	uint8_t *mark;
	kthm *st, th;
	uint32_t ns = 0, cnt = 0, i, cap = kthm_count();

	if (top == KNONE || top >= cap) {
		snprintf(msg, (size_t)n, "no such theorem");
		return 0;
	}
	mark = xcalloc(cap, 1);
	st = xmalloc(cap * sizeof *st);
	st[ns++] = top;
	mark[top] = 1;
	while (ns) {
		th = st[--ns];
		for (i = 0; i < kthm_nprem(th); i++) {
			kthm q = kthm_prem(th, i);
			if (q && q < cap && !mark[q]) {
				mark[q] = 1;
				st[ns++] = q;
			}
		}
		/* a hom declaration a conversion relies on is checked too */
		if (kthm_rule(th) == KR_CONV) {
			uint32_t s;
			for (s = 1; s < nksyms; s++)
				if (ksyms[s].homthm && ksyms[s].homthm < th &&
				    !mark[ksyms[s].homthm]) {
					mark[ksyms[s].homthm] = 1;
					st[ns++] = ksyms[s].homthm;
				}
		}
	}
	memo_clear();
	now = 0;
	for (th = 1; th < cap; th++) {
		if (!mark[th])
			continue;
		if (kthm_rule(th) == KR_HOMDECL || now == 0 ||
		    th > now) {
			/* homs active strictly before th: refresh the memo
			 * when crossing a declaration */
			uint32_t s, cross = 0;
			for (s = 1; s < nksyms; s++)
				if (ksyms[s].homthm && ksyms[s].homthm >= now &&
				    ksyms[s].homthm < th)
					cross = 1;
			if (cross)
				memo_clear();
			now = th;
		}
		why = NULL;
		if (!step(th)) {
			snprintf(msg, (size_t)n, "theorem %u (%s): %s: %s", th,
			    kr_name[kthm_rule(th)], why ? why : "invalid",
			    kt_str(kthm_concl(th)));
			free(mark);
			free(st);
			return 0;
		}
		cnt++;
	}
	free(mark);
	free(st);
	if (nsteps)
		*nsteps = cnt;
	snprintf(msg, (size_t)n, "%u inferences re-checked", cnt);
	return 1;
}

/* ---- PAC / Nullstellensatz certificates ---------------------------------
 *
 * Kaufmann & Biere (STTT 2023, AMulet2) distinguish two algebraic proof
 * formats.  PAC is a sequence of small steps, each checked on its own, so a
 * bad step is localized; Nullstellensatz is one concise list of cofactors
 * a_i with f = sum a_i g_i, where a single wrong cofactor ruins the whole
 * check.  Their LPAC sits between the two.  Here a kernel proof already
 * carries the cofactors: every KR_LIN inference is a Nullstellensatz
 * combination (premise relations g_i, kernel args a_i) and every KR_CONV
 * inference is a ring normalization.  We expose those steps in either
 * format and re-check them with the canonical arithmetic above, which is
 * independent of knorm.c.  A deliberately corrupted cofactor is reported
 * with its step number, the way PAC localizes an error.
 */

enum { CERT_OTHER, CERT_LIN, CERT_RING };

struct cert_step {
	int kind;		/* CERT_LIN | CERT_RING | CERT_OTHER */
	int rule;		/* kernel rule id */
	int supported;		/* all terms in the integer polynomial ring */
	int injected;		/* fault injected by cert_check() */
	uint32_t id;		/* kernel theorem id */
	uint32_t alg;		/* 1-based algebraic step number */
	uint32_t np;		/* premises / cofactors (kr_lin allows <= 64) */
	kt f;			/* claimed relation */
	kt g[64];		/* premise relations, materialized */
	kt a[64];		/* cofactors */
	int ge[64];		/* premise is an inequality */
};

struct ucas_cert {
	struct cert_step *s;
	int n;			/* all stored steps */
	int nlin, nring;	/* algebraic steps */
	int supported;
};

static struct cp
cert_poly(kt t)
{
	struct cp r = {0};

	/* the module's zero is a ring zero when read as a polynomial */
	if (t == KNONE || KOP(t) == KO_ZERO)
		return r;
	return cpoly(t);
}

struct ucas_cert *
cert_build(kthm top, int *nall)
{
	uint32_t cap = kthm_count(), ns = 0, i, th;
	uint8_t *mark;
	kthm *st;
	struct ucas_cert *c = xcalloc(1, sizeof *c);
	int n = 0, alloc = 0;

	if (nall)
		*nall = 0;
	if (top == KNONE || top >= cap)
		return c;
	mark = xcalloc((size_t)cap, 1);
	st = xmalloc((size_t)cap * sizeof *st);
	st[ns++] = top;
	mark[top] = 1;
	while (ns) {	/* mark the whole proof DAG, premises first */
		th = st[--ns];
		for (i = 0; i < kthm_nprem(th); i++) {
			kthm q = kthm_prem(th, i);
			if (q && q < cap && !mark[q]) {
				mark[q] = 1;
				st[ns++] = q;
			}
		}
	}
	c->supported = 1;
	for (th = 1; th < cap; th++) {
		struct cert_step *s;
		struct sbuf b = {0};
		int r, j, k, nt = 0;
		kt terms[130];

		if (!mark[th])
			continue;
		r = kthm_rule(th);
		if (nall)
			(*nall)++;
		if (n == alloc) {
			alloc = alloc ? alloc * 2 : 16;
			c->s = xrealloc(c->s, (size_t)alloc * sizeof *c->s);
		}
		s = &c->s[n++];
		memset(s, 0, sizeof *s);
		s->supported = 1;
		s->rule = r;
		s->id = th;
		s->f = kthm_concl(th);
		if (r != KR_LIN && r != KR_CONV) {
			s->kind = CERT_OTHER;
			continue;
		}
		/* a conversion is a ring-normalization step only when its
		 * conclusion is an equation whose polynomial is zero; other
		 * conversions are propositional/order scaffolding */
		if (r == KR_CONV) {
			int ge = 0;
			kt f = relterm(kthm_concl(th), &ge);
			struct cp p;

			if (f == KNONE || ge) {
				s->kind = CERT_OTHER;
				continue;
			}
			p = cert_poly(f);
			if (p.n) {
				cp_free(&p);
				s->kind = CERT_OTHER;
				continue;
			}
			cp_free(&p);
		}
		s->kind = r == KR_LIN ? CERT_LIN : CERT_RING;
		s->alg = (uint32_t)(c->nlin + c->nring + 1);
		if (r == KR_LIN) {
			uint32_t np = kthm_nprem(th);

			c->nlin++;

			for (j = 0; j < (int)np; j++) {
				int ge = 0;
				kt g = relterm(kthm_concl(kthm_prem(th,
				    (uint32_t)j)), &ge);

				if (g == KNONE) {
					s->supported = 0;
					break;
				}
				s->g[j] = g;
				s->ge[j] = ge;
				s->a[j] = kthm_arg(th, (uint32_t)j);
				s->np++;
			}
		} else {
			/* a conv establishes the relation by ring normalization;
			 * its premise may be `true`, so only the conclusion is a
			 * ring fact: the relation normalizes to zero. */
			c->nring++;
			s->np = 0;
		}
		/* the ring supports the certificate only if every term has a
		 * canonical polynomial form (no opaque powers or operators) */
		{
			int ge = 0;
			kt f = relterm(s->f, &ge);
			terms[nt++] = f == KNONE ? s->f : f;
		}
		for (j = 0; j < (int)s->np; j++) {
			terms[nt++] = s->g[j];
			if (s->kind == CERT_LIN)
				terms[nt++] = s->a[j];
		}
		for (k = 0; s->supported && k < nt; k++) {
			struct cp p = cert_poly(terms[k]);

			cp_str(&b, &p);
			cp_free(&p);
			if (strstr(b.p ? b.p : "", "W(") ||
			    strstr(b.p ? b.p : "", "?"))
				s->supported = 0;
		}
		free(b.p);
		if (!s->supported)
			c->supported = 0;
	}
	free(mark);
	free(st);
	c->n = n;
	return c;
}

void
cert_free(struct ucas_cert *c)
{
	if (!c)
		return;
	free(c->s);
	free(c);
}

/* The checker has three outcomes, not two: a check that passed, a checked
 * step that failed, and "there is nothing checkable / the request is bad".
 * cmd_check() in prove_cmd.c mirrors these codes so it can print an honest
 * status instead of a bogus failure at step 0. */
enum { CERT_UNCHECKABLE = -1, CERT_FAILED = 0, CERT_VERIFIED = 1 };

int
cert_nalg(const struct ucas_cert *c)
{
	return c ? c->nlin + c->nring : 0;
}

int
cert_supported(const struct ucas_cert *c)
{
	return c && c->supported;
}

static int
cp_same(const struct cp *a, const struct cp *b)
{
	struct sbuf x = {0}, y = {0};
	int r;

	cp_str(&x, a);
	cp_str(&y, b);
	r = !strcmp(x.p ? x.p : "", y.p ? y.p : "");
	free(x.p);
	free(y.p);
	return r;
}

/* Verify every algebraic step.  corrupt = 0 checks the proof as emitted;
 * otherwise the 1-based algebraic step with that number is sabotaged and
 * the checker must name it.  A request beyond the last algebraic step is an
 * error, never a silent pass.  Returns CERT_VERIFIED only when every step
 * checks, CERT_FAILED when a checked step fails (badstep names it), and
 * CERT_UNCHECKABLE when there is nothing checkable or the request is invalid
 * (msg says why, no step is named). */
int
cert_check(struct ucas_cert *c, int corrupt, char *msg, int n,
    uint32_t *badstep)
{
	int i, alg = 0, j;

	if (badstep)
		*badstep = 0;
	cbad = 0;	/* never inherit an overflow flag from a previous check */
	if (!c || !c->nlin && !c->nring) {
		snprintf(msg, (size_t)n, "no algebraic steps to check");
		return CERT_UNCHECKABLE;
	}
	if (!c->supported) {
		snprintf(msg, (size_t)n,
		    "not supported: terms outside the integer polynomial ring");
		return CERT_UNCHECKABLE;
	}
	for (i = 0; i < c->n; i++) {
		struct cert_step *s = &c->s[i];
		struct cp sum = {0}, p, q, r, target, one = {0};
		struct sbuf got = {0}, want = {0};
		char wbuf[96];
		int ge = 0;
		const char *why = NULL;

		if (s->kind == CERT_OTHER)
			continue;
		alg++;
		if (corrupt == alg) {
			s->injected = 1;
			if (s->kind == CERT_LIN && s->np)
				s->a[0] = kt_add(s->a[0], kt_num(1));
		}
		target = cert_poly(relterm(s->f, &ge));
		if (s->kind == CERT_RING) {
			if (target.n)
				why = "does not reduce to zero in the ring";
			if (s->injected) {
				cp_add1(&one, "", 1);
				cp_addto(&target, &one, 1);
				if (!why)
					why = "normalization fault injected";
			}
		} else {
			int anyge = 0;

			/* relterm() maps s = 0 and 0 <= s to the same polynomial,
			 * so a polynomial match alone cannot tell an equality
			 * conclusion from an inequality one.  kr_lin() makes the
			 * conclusion an inequality exactly when some premise is
			 * one; reject a conclusion with the wrong relation kind. */
			for (j = 0; j < (int)s->np; j++)
				if (s->ge[j])
					anyge = 1;
			if (ge != anyge || KOP(s->f) != (anyge ? KO_LE : KO_EQ))
				why = "conclusion relation does not match the premises";
			for (j = 0; j < (int)s->np && !why; j++) {
				p = cert_poly(s->a[j]);
				q = cert_poly(s->g[j]);
				if (s->ge[j]) {
					int nonneg = p.n == 0 ||
					    (p.n == 1 && !*p.t[0].m &&
					    p.t[0].c >= 0);
					if (!nonneg) {
						snprintf(wbuf, sizeof wbuf,
						    "cofactor a%d is not a nonnegative numeral "
						    "(inequality premise)", j + 1);
						why = wbuf;
						cp_free(&p);
						cp_free(&q);
						break;
					}
				}
				r = cp_mul(&p, &q);
				cp_addto(&sum, &r, 1);
				cp_free(&r);
				cp_free(&p);
				cp_free(&q);
			}
			if (!why && s->injected && !s->np)
				cp_add1(&sum, "", 1);
			if (!why && !cp_same(&sum, &target))
				why = s->injected ?
				    "cofactor a1 fault: combination no longer equals f" :
				    "sum of cofactors differs from f";
		}
		if (cbad) {	/* overflowed arithmetic is not a verification */
			cbad = 0;
			cp_free(&sum);
			cp_free(&target);
			free(got.p);
			free(want.p);
			snprintf(msg, (size_t)n,
			    "integer overflow while checking the certificate");
			return CERT_UNCHECKABLE;
		}
		cp_str(&got, &sum);
		cp_str(&want, &target);
		if (why) {
			char detail[256];
			snprintf(detail, sizeof detail,
			    "sum [%s], f [%s]", got.p ? got.p : "0",
			    want.p ? want.p : "0");
			snprintf(msg, (size_t)n,
			    "step %u (%s): %s; %s", s->alg,
			    s->kind == CERT_LIN ? "lin" : "ring", why, detail);
			if (badstep)
				*badstep = s->alg;
			cp_free(&sum);
			cp_free(&target);
			free(got.p);
			free(want.p);
			return CERT_FAILED;
		}
		cp_free(&sum);
		cp_free(&target);
		free(got.p);
		free(want.p);
	}
	/* No fault was injected: the requested step does not exist. */
	if (corrupt > alg) {
		snprintf(msg, (size_t)n,
		    "broken step %d is out of range: the proof has only "
		    "%d algebraic step(s)", corrupt, alg);
		return CERT_UNCHECKABLE;
	}
	if (cbad) {	/* the last product may have overflowed */
		cbad = 0;
		snprintf(msg, (size_t)n,
		    "integer overflow while checking the certificate");
		return CERT_UNCHECKABLE;
	}
	snprintf(msg, (size_t)n,
	    "%d algebraic step(s) verified (%d Nullstellensatz combination(s), "
	    "%d ring normalization(s))", c->nlin + c->nring, c->nlin, c->nring);
	return CERT_VERIFIED;
}

void
cert_emit(struct ucas_cert *c, int mode, struct sbuf *o)
{
	char msg[512];
	int ok = cert_check(c, 0, msg, sizeof msg, NULL);
	int i, j;

	if (mode == 0 || mode == 2) {
		sb_puts(o, "Nullstellensatz certificate (a_i, f = sum a_i*g_i):\n");
		if (!c->nlin && !c->nring) {
			sb_puts(o, "  no algebraic steps in this proof\n");
		} else if (!c->supported) {
			sb_puts(o, "  not supported: terms outside the integer "
			    "polynomial ring\n");
		} else {
			for (i = 0; i < c->n; i++) {
				struct cert_step *s = &c->s[i];

				if (s->kind == CERT_OTHER)
					continue;
				sb_printf(o, "  step %u [%s]\n", s->alg,
				    s->kind == CERT_LIN ? "lin" : "ring");
				if (s->kind == CERT_LIN) {
					for (j = 0; j < (int)s->np; j++)
						sb_printf(o, "    g%d = %s\n",
						    j + 1, kt_str(s->g[j]));
					for (j = 0; j < (int)s->np; j++)
						sb_printf(o, "    a%d = %s\n",
						    j + 1, kt_str(s->a[j]));
				}
				sb_printf(o, "    f  = %s\n", kt_str(s->f));
			}
			sb_printf(o, "  [%s] %s\n", ok ? "ok" : "FAILED", msg);
		}
	}
	if (mode == 1 || mode == 2) {
		sb_puts(o, "PAC step proof (one line per kernel inference):\n");
		for (i = 0; i < c->n; i++) {
			struct cert_step *s = &c->s[i];

			sb_printf(o, "  step %u [%s]", s->id,
			    kr_name[s->rule]);
			if (s->kind == CERT_LIN) {
				for (j = 0; j < (int)s->np; j++)
					sb_printf(o, " a%d=%s", j + 1,
					    kt_str(s->a[j]));
			}
			sb_printf(o, ": %s\n", kt_str(s->f));
		}
	}
}
