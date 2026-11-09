/*
 * kterm.c - the proof kernel's terms and formulas: sorts, signature,
 * a hash-consed DAG in SoA columns, de Bruijn binders, substitution,
 * a printer and a parser (prove.h).
 */
#include "ucas.h"
#include "prove.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

struct kterms KT;
struct ksym *ksyms;
uint32_t nksyms;
static uint32_t capsyms;

static char sortname[KS_MAX][16] = {"prop", "int", "el"};
static int nsorts = KS_USER;

int
ks_lookup(const char *name)
{
	int i;

	for (i = 0; i < nsorts; i++)
		if (!strcmp(sortname[i], name))
			return i;
	return -1;
}

int
ks_declare(const char *name)
{
	int i = ks_lookup(name);

	if (i >= 0)
		return i;
	if (nsorts >= KS_MAX)
		fail("too many sorts");
	snprintf(sortname[nsorts], sizeof sortname[0], "%s", name);
	return nsorts++;
}

const char *
ks_name(int s)
{
	return s >= 0 && s < nsorts ? sortname[s] : "?";
}

int
ksym_lookup(const char *name)
{
	uint32_t id = sym_cstr(name), i;

	for (i = 1; i < nksyms; i++)
		if (ksyms[i].name == id)
			return (int)i;
	return -1;
}

int
ksym_declare(const char *name, int kind, int arity, const uint8_t *args,
    int res, int flags)
{
	struct ksym *s;
	int i;

	if (nksyms == 0)
		nksyms = 1;	/* 0 is no symbol */
	if ((i = ksym_lookup(name)) >= 0)
		return i;
	if (nksyms >= capsyms) {
		capsyms = capsyms ? capsyms * 2 : 64;
		ksyms = xrealloc(ksyms, capsyms * sizeof *ksyms);
	}
	s = &ksyms[nksyms];
	memset(s, 0, sizeof *s);
	s->name = sym_cstr(name);
	s->kind = (uint8_t)kind;
	s->arity = (uint8_t)arity;
	s->res = (uint8_t)res;
	s->flags = (uint8_t)flags;
	for (i = 0; i < arity && i < 8; i++)
		s->arg[i] = args[i];
	return (int)nksyms++;
}

/* ---- the table ---- */

static uint32_t *htab;
static uint32_t hcap;

static uint32_t
hmix(uint32_t h, uint64_t v)
{
	h ^= (uint32_t)v + 0x9e3779b9u + (h << 6) + (h >> 2);
	h ^= (uint32_t)(v >> 32) + 0x9e3779b9u + (h << 6) + (h >> 2);
	return h;
}

static uint32_t
thash(int op, int sort, uint32_t aux, int64_t num, const kt *args, int n)
{
	uint32_t h = 2166136261u;
	int i;

	h = hmix(h, (uint64_t)op | (uint64_t)sort << 8);
	h = hmix(h, aux);
	h = hmix(h, (uint64_t)num);
	for (i = 0; i < n; i++)
		h = hmix(h, args[i]);
	return h;
}

static void
grow(void)
{
	uint32_t c = KT.cap ? KT.cap * 2 : 4096;

	KT.op = xrealloc(KT.op, c);
	KT.sort = xrealloc(KT.sort, c);
	KT.n = xrealloc(KT.n, c);
	KT.lb = xrealloc(KT.lb, c * sizeof *KT.lb);
	KT.aux = xrealloc(KT.aux, c * sizeof *KT.aux);
	KT.a0 = xrealloc(KT.a0, c * sizeof *KT.a0);
	KT.hint = xrealloc(KT.hint, c * sizeof *KT.hint);
	KT.fv = xrealloc(KT.fv, c * sizeof *KT.fv);
	KT.num = xrealloc(KT.num, c * sizeof *KT.num);
	KT.cap = c;
}

static void
rehash(void)
{
	uint32_t c = hcap ? hcap * 2 : 8192, i, h;

	free(htab);
	htab = xcalloc(c, sizeof *htab);
	hcap = c;
	for (i = 1; i < KT.len; i++) {
		h = thash(KT.op[i], KT.sort[i], KT.aux[i], KT.num[i],
		    KT.args + KT.a0[i], KT.n[i]) & (c - 1);
		while (htab[h])
			h = (h + 1) & (c - 1);
		htab[h] = i;
	}
}

kt
kt_mk(int op, int sort, uint32_t aux, int64_t num, const kt *args, int n)
{
	uint32_t h, t;
	int i;
	unsigned lb = 0;
	uint32_t fv = 0;

	if (KT.len == 0) {
		grow();
		KT.len = 1;	/* id 0 is KNONE */
		KT.nargs = 0;
	}
	if ((KT.len + 1) * 2 >= hcap)
		rehash();
	h = thash(op, sort, aux, num, args, n) & (hcap - 1);
	while ((t = htab[h]) != 0) {
		if (KT.op[t] == op && KT.sort[t] == sort && KT.aux[t] == aux &&
		    KT.num[t] == num && KT.n[t] == n &&
		    (n == 0 || !memcmp(KT.args + KT.a0[t], args,
		    (size_t)n * sizeof *args)))
			return t;
		h = (h + 1) & (hcap - 1);
	}
	if (KT.len == KT.cap)
		grow();
	if (KT.nargs + (uint32_t)n > KT.capargs) {
		KT.capargs = newcap(KT.capargs, KT.nargs + (uint32_t)n + 1024);
		KT.args = xrealloc(KT.args, KT.capargs * sizeof *KT.args);
	}
	t = KT.len++;
	KT.op[t] = (uint8_t)op;
	KT.sort[t] = (uint8_t)sort;
	KT.aux[t] = aux;
	KT.num[t] = num;
	KT.n[t] = (uint8_t)n;
	KT.a0[t] = KT.nargs;
	KT.hint[t] = 0;
	for (i = 0; i < n; i++) {
		KT.args[KT.nargs++] = args[i];
		if (KT.lb[args[i]] > lb)
			lb = KT.lb[args[i]];
		fv |= KT.fv[args[i]];
	}
	if (op == KO_BVAR)
		lb = aux + 1;
	if (op == KO_ALL || op == KO_EX)
		lb = lb ? lb - 1 : 0;
	if (op == KO_VAR || op == KO_META)
		fv = 1u << ((aux * 2654435761u + (op == KO_META)) >> 27);
	KT.lb[t] = (uint16_t)lb;
	KT.fv[t] = fv;
	htab[h] = t;
	return t;
}

kt
kt_bvar(uint32_t i, int sort)
{
	return kt_mk(KO_BVAR, sort, i, 0, NULL, 0);
}

kt
kt_var(uint32_t name, int sort)
{
	return kt_mk(KO_VAR, sort, name, 0, NULL, 0);
}

kt
kt_varc(const char *name, int sort)
{
	return kt_var(sym_cstr(name), sort);
}

kt
kt_meta(uint32_t i, int sort)
{
	return kt_mk(KO_META, sort, i, 0, NULL, 0);
}

kt
kt_num(int64_t v)
{
	return kt_mk(KO_NUM, KS_INT, 0, v, NULL, 0);
}

kt
kt_zero(void)
{
	return kt_mk(KO_ZERO, KS_EL, 0, 0, NULL, 0);
}

kt
kt_app(int sym, const kt *args, int n)
{
	return kt_mk(KO_APP, ksyms[sym].res, (uint32_t)sym, 0, args, n);
}

static kt
bin(int op, int sort, kt a, kt b)
{
	kt x[2];

	x[0] = a;
	x[1] = b;
	return kt_mk(op, sort, 0, 0, x, 2);
}

kt
kt_add(kt a, kt b)
{
	return bin(KO_ADD, KSORT(a), a, b);
}

kt
kt_neg(kt a)
{
	return kt_mk(KO_NEG, KSORT(a), 0, 0, &a, 1);
}

kt
kt_sub(kt a, kt b)
{
	return kt_add(a, kt_neg(b));
}

kt
kt_mul(kt a, kt b)
{
	return bin(KO_MUL, KSORT(b), a, b);
}

kt
kt_pow(kt a, kt b)
{
	return bin(KO_POW, KS_INT, a, b);
}

kt
kf_true(void)
{
	return kt_mk(KO_TRUE, KS_PROP, 0, 0, NULL, 0);
}

kt
kf_false(void)
{
	return kt_mk(KO_FALSE, KS_PROP, 0, 0, NULL, 0);
}

kt
kf_pred(int sym, const kt *args, int n)
{
	return kt_mk(KO_PRED, KS_PROP, (uint32_t)sym, 0, args, n);
}

kt
kf_eq(kt a, kt b)
{
	return bin(KO_EQ, KS_PROP, a, b);
}

kt
kf_le(kt a, kt b)
{
	return bin(KO_LE, KS_PROP, a, b);
}

kt
kf_lt(kt a, kt b)
{
	return bin(KO_LT, KS_PROP, a, b);
}

kt
kf_in(kt x, kt s)
{
	return bin(KO_IN, KS_PROP, x, s);
}

kt
kf_not(kt a)
{
	return kt_mk(KO_NOT, KS_PROP, 0, 0, &a, 1);
}

kt
kf_and(kt a, kt b)
{
	return bin(KO_AND, KS_PROP, a, b);
}

kt
kf_or(kt a, kt b)
{
	return bin(KO_OR, KS_PROP, a, b);
}

kt
kf_imp(kt a, kt b)
{
	return bin(KO_IMP, KS_PROP, a, b);
}

kt
kf_iff(kt a, kt b)
{
	return bin(KO_IFF, KS_PROP, a, b);
}

kt
kf_bind(int op, int sort, uint32_t hint, kt body)
{
	kt t = kt_mk(op, KS_PROP, (uint32_t)sort, 0, &body, 1);

	if (!KT.hint[t])
		KT.hint[t] = hint;
	return t;
}

int
kt_isformula(kt t)
{
	return KSORT(t) == KS_PROP;
}

/* ---- substitution ---- */

/* rebuild t with new arguments */
static kt
remk(kt t, const kt *x)
{
	kt r = kt_mk(KOP(t), KSORT(t), KT.aux[t], KT.num[t], x, KN(t));

	if ((KOP(t) == KO_ALL || KOP(t) == KO_EX) && !KT.hint[r])
		KT.hint[r] = KT.hint[t];
	return r;
}

/* a small memo for one traversal */
struct memo {
	uint32_t *k, *v, cap, n;
};

static uint32_t
memo_get(struct memo *m, uint32_t key)
{
	uint32_t h;

	if (!m->cap)
		return 0;
	h = (key * 2654435761u) & (m->cap - 1);
	while (m->k[h]) {
		if (m->k[h] == key)
			return m->v[h];
		h = (h + 1) & (m->cap - 1);
	}
	return 0;
}

static void
memo_put(struct memo *m, uint32_t key, uint32_t val)
{
	uint32_t h, i;

	if ((m->n + 1) * 2 >= m->cap) {
		struct memo o = *m;
		m->cap = m->cap ? m->cap * 2 : 256;
		m->k = xcalloc(m->cap, sizeof *m->k);
		m->v = xcalloc(m->cap, sizeof *m->v);
		m->n = 0;
		for (i = 0; i < o.cap; i++)
			if (o.k[i])
				memo_put(m, o.k[i], o.v[i]);
		free(o.k);
		free(o.v);
	}
	h = (key * 2654435761u) & (m->cap - 1);
	while (m->k[h] && m->k[h] != key)
		h = (h + 1) & (m->cap - 1);
	if (!m->k[h])
		m->n++;
	m->k[h] = key;
	m->v[h] = val;
}

static void
memo_free(struct memo *m)
{
	free(m->k);
	free(m->v);
	memset(m, 0, sizeof *m);
}

/* lift the loose bound variables of s (index >= c) by k */
static kt
lift_rec(kt s, uint32_t k, uint32_t c, struct memo *m)
{
	kt x[16], r;
	uint32_t key;
	int i, n = KN(s), ch = 0;

	if (KT.lb[s] <= c)
		return s;
	if (KOP(s) == KO_BVAR)
		return kt_bvar(KT.aux[s] + k, KSORT(s));
	key = s * 64u + c + 1;
	if ((r = memo_get(m, key)) != 0)
		return r;
	for (i = 0; i < n; i++) {
		x[i] = lift_rec(KA(s, i), k, c + (KOP(s) == KO_ALL ||
		    KOP(s) == KO_EX), m);
		ch |= x[i] != KA(s, i);
	}
	r = ch ? remk(s, x) : s;
	memo_put(m, key, r);
	return r;
}

static kt
lift(kt s, uint32_t k)
{
	struct memo m = {0};
	kt r;

	if (k == 0 || KT.lb[s] == 0)
		return s;
	r = lift_rec(s, k, 0, &m);
	memo_free(&m);
	return r;
}

static kt
inst_rec(kt t, uint32_t d, kt s, struct memo *m)
{
	kt x[16], r;
	uint32_t key;
	int i, n = KN(t), ch = 0;

	if (KT.lb[t] <= d)
		return t;
	if (KOP(t) == KO_BVAR)
		return KT.aux[t] == d ? lift(s, d) :
		    kt_bvar(KT.aux[t] - 1, KSORT(t));
	key = t * 64u + d + 1;
	if ((r = memo_get(m, key)) != 0)
		return r;
	for (i = 0; i < n; i++) {
		x[i] = inst_rec(KA(t, i), d + (KOP(t) == KO_ALL ||
		    KOP(t) == KO_EX), s, m);
		ch |= x[i] != KA(t, i);
	}
	r = ch ? remk(t, x) : t;
	memo_put(m, key, r);
	return r;
}

kt
kt_inst(kt body, kt s)
{
	struct memo m = {0};
	kt r = inst_rec(body, 0, s, &m);

	memo_free(&m);
	return r;
}

static kt
abs_rec(kt t, uint32_t d, kt v, struct memo *m)
{
	kt x[16], r;
	uint32_t key;
	int i, n = KN(t), ch = 0;

	if (t == v)
		return kt_bvar(d, KSORT(v));
	if (!(KT.fv[t] & KT.fv[v]) || n == 0)
		return t;
	key = t * 64u + d + 1;
	if ((r = memo_get(m, key)) != 0)
		return r;
	for (i = 0; i < n; i++) {
		x[i] = abs_rec(KA(t, i), d + (KOP(t) == KO_ALL ||
		    KOP(t) == KO_EX), v, m);
		ch |= x[i] != KA(t, i);
	}
	r = ch ? remk(t, x) : t;
	memo_put(m, key, r);
	return r;
}

kt
kt_abstract(kt t, kt var)
{
	struct memo m = {0};
	kt r = abs_rec(t, 0, var, &m);

	memo_free(&m);
	return r;
}

static int
occ_rec(kt v, kt t, struct memo *m)
{
	int i, r = 0;

	if (t == v)
		return 1;
	if (!(KT.fv[t] & KT.fv[v]) || KN(t) == 0)
		return 0;
	if (memo_get(m, t))
		return memo_get(m, t) - 1;
	for (i = 0; i < KN(t) && !r; i++)
		r = occ_rec(v, KA(t, i), m);
	memo_put(m, t, (uint32_t)r + 1);
	return r;
}

int
kt_occurs(kt v, kt t)
{
	struct memo m = {0};
	int r = occ_rec(v, t, &m);

	memo_free(&m);
	return r;
}

static int
hasmeta_rec(kt t, struct memo *m)
{
	int i, r = 0;

	if (KOP(t) == KO_META)
		return 1;
	if (KN(t) == 0 || !KT.fv[t])
		return 0;
	if (memo_get(m, t))
		return memo_get(m, t) - 1;
	for (i = 0; i < KN(t) && !r; i++)
		r = hasmeta_rec(KA(t, i), m);
	memo_put(m, t, (uint32_t)r + 1);
	return r;
}

int
kt_hasmeta(kt t)
{
	struct memo m = {0};
	int r = hasmeta_rec(t, &m);

	memo_free(&m);
	return r;
}

static kt
msub_rec(kt t, const kt *val, uint32_t nval, struct memo *m)
{
	kt x[16], r;
	int i, ch = 0;

	if (KOP(t) == KO_META)
		return KT.aux[t] < nval && val[KT.aux[t]] ?
		    msub_rec(val[KT.aux[t]], val, nval, m) : t;
	if (KN(t) == 0 || !KT.fv[t])
		return t;
	if ((r = memo_get(m, t)) != 0)
		return r;
	for (i = 0; i < KN(t); i++) {
		x[i] = msub_rec(KA(t, i), val, nval, m);
		ch |= x[i] != KA(t, i);
	}
	r = ch ? remk(t, x) : t;
	memo_put(m, t, r);
	return r;
}

kt
kt_metasub(kt t, const kt *val, uint32_t nval)
{
	struct memo m = {0};
	kt r = msub_rec(t, val, nval, &m);

	memo_free(&m);
	return r;
}

kt
kt_fresh(const char *base, int sort)
{
	static uint32_t ctr;
	char b[64];
	size_t n = strcspn(base, "'");

	snprintf(b, sizeof b, "%.*s'%u", (int)(n > 40 ? 40 : n), base, ++ctr);
	return kt_varc(b, sort);
}

/* ---- printer ---- */

/* precedence: 1 iff, 2 imp, 3 or, 4 and, 5 not/quant, 6 relation,
 * 7 sum, 8 product, 9 unary minus, 10 power, 11 atom */
struct kpctx {
	const char *bn[64];
	int nb;
};

static void pr(struct sbuf *b, kt t, struct kpctx *c, int prec);

static void
popen_(struct sbuf *b, int need, int prec)
{
	if (need < prec)
		sb_putc(b, '(');
}

static void
pclose_(struct sbuf *b, int need, int prec)
{
	if (need < prec)
		sb_putc(b, ')');
}

static const char *
bname(struct kpctx *c, kt t)
{
	static char buf[64][32];
	static int k;
	const char *h = KT.hint[t] ? sym_str(KT.hint[t]) : "x";
	int i, clash, n = 0;
	char *o = buf[k = (k + 1) & 63];

	snprintf(o, 32, "%s", h);
	for (;;) {
		clash = 0;
		for (i = 0; i < c->nb; i++)
			if (!strcmp(c->bn[i], o))
				clash = 1;
		if (!clash)
			break;
		snprintf(o, 32, "%s%d", h, ++n);
	}
	return o;
}

static void
pr_args(struct sbuf *b, kt t, struct kpctx *c)
{
	int i;

	sb_putc(b, '(');
	for (i = 0; i < KN(t); i++) {
		if (i)
			sb_puts(b, ", ");
		pr(b, KA(t, i), c, 0);
	}
	sb_putc(b, ')');
}

static int
isneg(kt t)
{
	return KOP(t) == KO_NEG || (KOP(t) == KO_NUM && KT.num[t] < 0) ||
	    (KOP(t) == KO_MUL && KOP(KA(t, 0)) == KO_NUM &&
	    KT.num[KA(t, 0)] < 0);
}

static void
pr_bin(struct sbuf *b, kt t, struct kpctx *c, int prec, int my,
    const char *op, int lp, int rp)
{
	popen_(b, my, prec);
	pr(b, KA(t, 0), c, lp);
	sb_puts(b, op);
	pr(b, KA(t, 1), c, rp);
	pclose_(b, my, prec);
}

static void
pr(struct sbuf *b, kt t, struct kpctx *c, int prec)
{
	static const char *const rel[] = {" = ", " <= ", " < ", " in "};
	kt u;
	int op = KOP(t);

	switch (op) {
	case KO_BVAR:
		if (KT.aux[t] < (uint32_t)c->nb)
			sb_puts(b, c->bn[c->nb - 1 - KT.aux[t]]);
		else
			sb_printf(b, "#%u", KT.aux[t]);
		return;
	case KO_VAR:
		sb_puts(b, sym_str(KT.aux[t]));
		return;
	case KO_META:
		sb_printf(b, "?%u", KT.aux[t]);
		return;
	case KO_NUM:
		popen_(b, KT.num[t] < 0 ? 9 : 11, prec);
		sb_printf(b, "%lld", (long long)KT.num[t]);
		pclose_(b, KT.num[t] < 0 ? 9 : 11, prec);
		return;
	case KO_ZERO:
		sb_puts(b, "0");
		return;
	case KO_APP:
	case KO_PRED:
		sb_puts(b, sym_str(ksyms[KT.aux[t]].name));
		if (KN(t))
			pr_args(b, t, c);
		return;
	case KO_ADD:
		popen_(b, 7, prec);
		pr(b, KA(t, 0), c, 7);
		u = KA(t, 1);
		if (isneg(u)) {
			sb_puts(b, " - ");
			if (KOP(u) == KO_NEG)
				pr(b, KA(u, 0), c, 8);
			else if (KOP(u) == KO_NUM)
				sb_printf(b, "%lld", -(long long)KT.num[u]);
			else {
				if (KT.num[KA(u, 0)] != -1)
					sb_printf(b, "%lld*",
					    -(long long)KT.num[KA(u, 0)]);
				pr(b, KA(u, 1), c, 8);
			}
		} else {
			sb_puts(b, " + ");
			pr(b, u, c, 8);
		}
		pclose_(b, 7, prec);
		return;
	case KO_NEG:
		popen_(b, 9, prec);
		sb_putc(b, '-');
		pr(b, KA(t, 0), c, 9);
		pclose_(b, 9, prec);
		return;
	case KO_MUL:
		if (KOP(KA(t, 0)) == KO_NUM && KT.num[KA(t, 0)] == -1) {
			popen_(b, 9, prec);
			sb_putc(b, '-');
			pr(b, KA(t, 1), c, 9);
			pclose_(b, 9, prec);
			return;
		}
		pr_bin(b, t, c, prec, 8, "*", 8, 9);
		return;
	case KO_POW:
		pr_bin(b, t, c, prec, 10, "^", 11, 10);
		return;
	case KO_TRUE:
		sb_puts(b, "true");
		return;
	case KO_FALSE:
		sb_puts(b, "false");
		return;
	case KO_EQ:
	case KO_LE:
	case KO_LT:
	case KO_IN:
		pr_bin(b, t, c, prec, 6, rel[op - KO_EQ], 7, 7);
		return;
	case KO_NOT:
		u = KA(t, 0);
		if (KOP(u) == KO_EQ) {
			pr_bin(b, u, c, prec, 6, " != ", 7, 7);
			return;
		}
		popen_(b, 5, prec);
		sb_puts(b, "not ");
		pr(b, u, c, 5);
		pclose_(b, 5, prec);
		return;
	case KO_AND:
		pr_bin(b, t, c, prec, 4, " and ", 4, 5);
		return;
	case KO_OR:
		pr_bin(b, t, c, prec, 3, " or ", 3, 4);
		return;
	case KO_IMP:
		pr_bin(b, t, c, prec, 2, " -> ", 3, 2);
		return;
	case KO_IFF:
		pr_bin(b, t, c, prec, 1, " <-> ", 2, 2);
		return;
	case KO_ALL:
	case KO_EX: {
		int s = (int)KT.aux[t], nb0 = c->nb, first = 1;

		popen_(b, 1, prec);
		sb_puts(b, op == KO_ALL ? "forall " : "exists ");
		/* group a run of the same quantifier */
		for (;;) {
			const char *nm = bname(c, t);
			if (!first && KT.aux[t] == (uint32_t)s)
				sb_putc(b, ' ');
			else if (!first) {
				sb_printf(b, ":%s, ", ks_name(s));
				s = (int)KT.aux[t];
			}
			sb_puts(b, nm);
			if (c->nb < 64)
				c->bn[c->nb++] = nm;
			first = 0;
			u = KA(t, 0);
			if (KOP(u) != op)
				break;
			t = u;
		}
		sb_printf(b, ":%s: ", ks_name(s));
		pr(b, KA(t, 0), c, 1);
		c->nb = nb0;
		pclose_(b, 1, prec);
		return;
	}
	}
	sb_puts(b, "?");
}

void
kt_print(struct sbuf *b, kt t)
{
	struct kpctx c;

	c.nb = 0;
	pr(b, t, &c, 0);
}

const char *
kt_str(kt t)
{
	static struct sbuf b[8];
	static int k;

	k = (k + 1) & 7;
	sb_clear(&b[k]);
	if (t == KNONE)
		sb_puts(&b[k], "(none)");
	else
		kt_print(&b[k], t);
	return sb_str(&b[k]);
}

/* ---- parser ---- */

enum {
	PT_END, PT_ID, PT_NUM, PT_LP, PT_RP, PT_COMMA, PT_COLON, PT_PLUS,
	PT_MINUS, PT_STAR, PT_CARET, PT_EQ, PT_NE, PT_LT, PT_LE, PT_GT, PT_GE,
	PT_IMP, PT_IFF, PT_AND, PT_OR, PT_NOT, PT_LB, PT_RB, PT_BAD
};

struct kp {
	const char *s;
	size_t i;
	int tk;
	size_t ts, tl;		/* token text */
	int64_t num;
	const struct kscope *sc;
	const char *bn[64];
	int bs[64];
	int nb;
	char *err;
	int errn;
	int failed;
};

static int
utf(const char *s, const char *u)
{
	return !strncmp(s, u, strlen(u));
}

static void
next(struct kp *p)
{
	const char *s = p->s;
	size_t i = p->i;
	static const struct {
		const char *t;
		int k;
	} ops[] = {
		{"<->", PT_IFF}, {"->", PT_IMP}, {"<=", PT_LE}, {">=", PT_GE},
		{"!=", PT_NE}, {"==", PT_EQ}, {"&&", PT_AND}, {"||", PT_OR},
		{"=", PT_EQ}, {"<", PT_LT}, {">", PT_GT}, {"+", PT_PLUS},
		{"-", PT_MINUS}, {"*", PT_STAR}, {"^", PT_CARET}, {"(", PT_LP},
		{")", PT_RP}, {",", PT_COMMA}, {":", PT_COLON}, {"!", PT_NOT},
		{"[", PT_LB}, {"]", PT_RB},
		{"\xe2\x86\x94", PT_IFF}, {"\xe2\x86\x92", PT_IMP},
		{"\xe2\x89\xa4", PT_LE}, {"\xe2\x89\xa5", PT_GE},
		{"\xe2\x89\xa0", PT_NE}, {"\xe2\x88\xa7", PT_AND},
		{"\xe2\x88\xa8", PT_OR}, {"\xc2\xac", PT_NOT},
		{"\xc2\xb7", PT_STAR}, {"\xe2\x88\x92", PT_MINUS},
	};
	size_t k;

	while (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')
		i++;
	p->ts = i;
	if (!s[i] || s[i] == '#') {
		p->tk = PT_END;
		p->tl = 0;
		p->i = i;
		return;
	}
	if (isdigit((unsigned char)s[i])) {
		int64_t v = 0;
		while (isdigit((unsigned char)s[i])) {
			if (v > (INT64_MAX - 9) / 10)
				p->failed = 1;
			v = v * 10 + (s[i++] - '0');
		}
		p->tk = PT_NUM;
		p->num = v;
	} else if (isalpha((unsigned char)s[i]) || s[i] == '_') {
		while (isalnum((unsigned char)s[i]) || s[i] == '_' ||
		    s[i] == '\'')
			i++;
		p->tk = PT_ID;
	} else if (utf(s + i, "\xe2\x88\x80")) {	/* forall */
		i += 3;
		p->tk = PT_ID;
	} else if (utf(s + i, "\xe2\x88\x83")) {	/* exists */
		i += 3;
		p->tk = PT_ID;
	} else if (utf(s + i, "\xe2\x88\x88")) {	/* in */
		i += 3;
		p->tk = PT_ID;
	} else {
		p->tk = PT_BAD;
		for (k = 0; k < sizeof ops / sizeof ops[0]; k++)
			if (utf(s + i, ops[k].t)) {
				p->tk = ops[k].k;
				i += strlen(ops[k].t);
				break;
			}
		if (p->tk == PT_BAD)
			i++;
	}
	p->tl = i - p->ts;
	p->i = i;
}

static int
isw(struct kp *p, const char *w)
{
	size_t n = strlen(w);

	if (p->tk != PT_ID)
		return 0;
	if (p->tl == n && !strncmp(p->s + p->ts, w, n))
		return 1;
	if (!strcmp(w, "forall") && utf(p->s + p->ts, "\xe2\x88\x80"))
		return 1;
	if (!strcmp(w, "exists") && utf(p->s + p->ts, "\xe2\x88\x83"))
		return 1;
	if (!strcmp(w, "in") && utf(p->s + p->ts, "\xe2\x88\x88"))
		return 1;
	return 0;
}

static kt
perr(struct kp *p, const char *fmt, const char *a)
{
	if (!p->failed) {
		char m[200];
		snprintf(m, sizeof m, fmt, a);
		snprintf(p->err, (size_t)p->errn, "%s at column %zu", m,
		    p->ts + 1);
	}
	p->failed = 1;
	return KNONE;
}

static const char *
tokstr(struct kp *p)
{
	static char b[64];

	if (p->tk == PT_END)
		return "the end";
	snprintf(b, sizeof b, "'%.*s'", (int)(p->tl > 40 ? 40 : p->tl),
	    p->s + p->ts);
	return b;
}

static kt pformula(struct kp *p);
static kt pterm(struct kp *p, int want);

/* a numeral where an el is wanted: only 0 */
static kt
coerce(struct kp *p, kt t, int want)
{
	if (t == KNONE || want < 0 || KSORT(t) == want)
		return t;
	if (want == KS_EL && KOP(t) == KO_NUM && KT.num[t] == 0)
		return kt_zero();
	return perr(p, "expected %s", ks_name(want));
}

static kt
pident(struct kp *p, int want)
{
	char name[64];
	int i, s;
	kt args[8];
	uint32_t id;

	snprintf(name, sizeof name, "%.*s", (int)(p->tl > 63 ? 63 : p->tl),
	    p->s + p->ts);
	next(p);
	/* bound variables, innermost first */
	for (i = p->nb - 1; i >= 0; i--)
		if (!strcmp(p->bn[i], name))
			return coerce(p, kt_bvar((uint32_t)(p->nb - 1 - i),
			    p->bs[i]), want);
	id = sym_cstr(name);
	if (p->sc)
		for (i = p->sc->n - 1; i >= 0; i--)
			if (p->sc->name[i] == id)
				return coerce(p, p->sc->var[i], want);
	s = ksym_lookup(name);
	if (s < 0 || ksyms[s].kind != KSY_FUNC)
		return perr(p, "unknown name '%s'", name);
	if (ksyms[s].arity == 0)
		return coerce(p, kt_app(s, NULL, 0), want);
	if (p->tk != PT_LP)
		return perr(p, "'%s' needs its arguments", name);
	next(p);
	for (i = 0; i < ksyms[s].arity; i++) {
		if (i) {
			if (p->tk != PT_COMMA)
				return perr(p, "expected ',' in %s(...)", name);
			next(p);
		}
		args[i] = pterm(p, ksyms[s].arg[i]);
		if (p->failed)
			return KNONE;
	}
	if (p->tk != PT_RP)
		return perr(p, "expected ')' after %s's arguments", name);
	next(p);
	return coerce(p, kt_app(s, args, ksyms[s].arity), want);
}

static kt
pprimary(struct kp *p, int want)
{
	kt t;

	if (p->tk == PT_NUM) {
		t = kt_num(p->num);
		next(p);
		return coerce(p, t, want);
	}
	if (p->tk == PT_ID)
		return pident(p, want);
	if (p->tk == PT_LP) {
		next(p);
		t = pterm(p, want);
		if (p->failed)
			return KNONE;
		if (p->tk != PT_RP)
			return perr(p, "expected ')', not %s", tokstr(p));
		next(p);
		return t;
	}
	return perr(p, "expected a term, not %s", tokstr(p));
}

static kt punary(struct kp *p, int want);

static kt
ppow(struct kp *p, int want)
{
	kt a = pprimary(p, want == KS_EL ? -1 : want), b;

	if (p->failed)
		return KNONE;
	if (p->tk == PT_CARET) {
		next(p);
		b = punary(p, KS_INT);
		if (p->failed)
			return KNONE;
		if (KSORT(a) != KS_INT)
			return perr(p, "%s", "^ needs integers");
		return kt_pow(a, b);
	}
	return a;
}

static kt
punary(struct kp *p, int want)
{
	kt a;

	if (p->tk == PT_MINUS) {
		next(p);
		a = punary(p, want);
		if (p->failed)
			return KNONE;
		if (KOP(a) == KO_NUM)
			return kt_num(-KT.num[a]);
		return kt_neg(a);
	}
	return ppow(p, want);
}

/* a product: int * ... * int [* el] */
static kt
pprod(struct kp *p, int want)
{
	kt a = punary(p, -1), b;

	if (p->failed)
		return KNONE;
	while (p->tk == PT_STAR) {
		next(p);
		b = punary(p, -1);
		if (p->failed)
			return KNONE;
		if (KSORT(a) != KS_INT)
			return perr(p, "%s", "only an integer can multiply");
		if (KSORT(b) != KS_INT && KSORT(b) != KS_EL)
			return perr(p, "%s", "cannot multiply that");
		a = kt_mul(a, b);
	}
	return coerce(p, a, want);
}

static kt
pterm(struct kp *p, int want)
{
	kt a = pprod(p, want), b;
	int neg;

	if (p->failed)
		return KNONE;
	while (p->tk == PT_PLUS || p->tk == PT_MINUS) {
		neg = p->tk == PT_MINUS;
		next(p);
		b = pprod(p, KSORT(a));
		if (p->failed)
			return KNONE;
		if (KSORT(a) != KS_INT && KSORT(a) != KS_EL)
			return perr(p, "%s", "cannot add that");
		a = neg ? kt_sub(a, b) : kt_add(a, b);
	}
	return a;
}

static int
relop(int tk)
{
	return tk == PT_EQ || tk == PT_NE || tk == PT_LT || tk == PT_LE ||
	    tk == PT_GT || tk == PT_GE;
}

/* t1 REL t2 or t1 in S */
static kt
prel(struct kp *p, kt a)
{
	int op = p->tk;
	kt b;

	if (isw(p, "in")) {
		next(p);
		b = pterm(p, -1);
		if (p->failed)
			return KNONE;
		if (KSORT(a) != KS_EL)
			a = coerce(p, a, KS_EL);
		return kf_in(a, b);
	}
	next(p);
	b = pterm(p, KSORT(a) == KS_INT && KOP(a) == KO_NUM ? -1 : KSORT(a));
	if (p->failed)
		return KNONE;
	if (KSORT(a) != KSORT(b)) {
		a = coerce(p, a, KSORT(b));
		if (p->failed)
			return KNONE;
	}
	if (op != PT_EQ && op != PT_NE && KSORT(a) != KS_INT)
		return perr(p, "%s", "order needs integers");
	switch (op) {
	case PT_EQ:
		return kf_eq(a, b);
	case PT_NE:
		return kf_not(kf_eq(a, b));
	case PT_LT:
		return kf_lt(a, b);
	case PT_LE:
		return kf_le(a, b);
	case PT_GT:
		return kf_lt(b, a);
	default:
		return kf_le(b, a);
	}
}

static kt
patom(struct kp *p)
{
	size_t save = p->i, ts = p->ts, tl = p->tl;
	int tk = p->tk, s;
	int64_t num = p->num;
	char err0[8];
	kt t, args[8];
	int i;

	if (isw(p, "true")) {
		next(p);
		return kf_true();
	}
	if (isw(p, "false")) {
		next(p);
		return kf_false();
	}
	if (p->tk == PT_ID) {
		char name[64];
		snprintf(name, sizeof name, "%.*s", (int)(p->tl > 63 ? 63 :
		    p->tl), p->s + p->ts);
		s = ksym_lookup(name);
		if (s >= 0 && ksyms[s].kind == KSY_PRED) {
			int bound = 0;
			for (i = 0; i < p->nb; i++)
				if (!strcmp(p->bn[i], name))
					bound = 1;
			if (!bound) {
				next(p);
				if (ksyms[s].arity == 0)
					return kf_pred(s, NULL, 0);
				if (p->tk != PT_LP)
					return perr(p, "'%s' needs its arguments",
					    name);
				next(p);
				for (i = 0; i < ksyms[s].arity; i++) {
					if (i) {
						if (p->tk != PT_COMMA)
							return perr(p, "expected ',' in "
							    "%s(...)", name);
						next(p);
					}
					args[i] = pterm(p, ksyms[s].arg[i]);
					if (p->failed)
						return KNONE;
				}
				if (p->tk != PT_RP)
					return perr(p, "expected ')' after %s's "
					    "arguments", name);
				next(p);
				return kf_pred(s, args, ksyms[s].arity);
			}
		}
	}
	/* a term followed by a relation; else a parenthesised formula */
	if (p->tk == PT_LP) {
		char *e = p->err;
		int en = p->errn;
		p->err = err0;
		p->errn = sizeof err0;
		t = pterm(p, -1);
		p->err = e;
		p->errn = en;
		if (!p->failed && (relop(p->tk) || isw(p, "in")))
			return prel(p, t);
		p->failed = 0;
		p->i = save;
		p->ts = ts;
		p->tl = tl;
		p->tk = tk;
		p->num = num;
		next(p);
		t = pformula(p);
		if (p->failed)
			return KNONE;
		if (p->tk != PT_RP)
			return perr(p, "expected ')', not %s", tokstr(p));
		next(p);
		return t;
	}
	t = pterm(p, -1);
	if (p->failed)
		return KNONE;
	if (!relop(p->tk) && !isw(p, "in"))
		return perr(p, "expected a relation (= < <= in ...), not %s",
		    tokstr(p));
	return prel(p, t);
}

static kt
pquant(struct kp *p, int op)
{
	const char *names[32];
	int sorts[32], n = 0, k, nb0 = p->nb, s;
	kt body;

	next(p);
	for (;;) {
		int first = n;
		char sn[32];
		while (p->tk == PT_ID) {
			char *nm = xmalloc(p->tl + 1);
			memcpy(nm, p->s + p->ts, p->tl);
			nm[p->tl] = 0;
			if (n >= 32)
				return perr(p, "%s", "too many variables");
			names[n++] = nm;
			next(p);
		}
		if (n == first || p->tk != PT_COLON)
			return perr(p, "expected 'x:sort', not %s", tokstr(p));
		next(p);
		if (p->tk != PT_ID)
			return perr(p, "expected a sort, not %s", tokstr(p));
		snprintf(sn, sizeof sn, "%.*s", (int)(p->tl > 31 ? 31 : p->tl),
		    p->s + p->ts);
		s = ks_lookup(sn);
		if (s < 0 || s == KS_PROP)
			return perr(p, "unknown sort '%s'", sn);
		next(p);
		for (k = first; k < n; k++)
			sorts[k] = s;
		if (p->tk == PT_COMMA) {
			next(p);
			continue;
		}
		if (p->tk != PT_COLON)
			return perr(p, "expected ':' before the body, not %s",
			    tokstr(p));
		next(p);
		break;
	}
	for (k = 0; k < n; k++) {
		if (p->nb >= 64)
			return perr(p, "%s", "nested too deeply");
		p->bn[p->nb] = names[k];
		p->bs[p->nb++] = sorts[k];
	}
	body = pformula(p);
	p->nb = nb0;
	if (p->failed)
		return KNONE;
	for (k = n - 1; k >= 0; k--)
		body = kf_bind(op, sorts[k], sym_cstr(names[k]), body);
	return body;
}

static kt
punf(struct kp *p)
{
	kt a;

	if (p->tk == PT_NOT || isw(p, "not")) {
		next(p);
		a = punf(p);
		return p->failed ? KNONE : kf_not(a);
	}
	if (isw(p, "forall"))
		return pquant(p, KO_ALL);
	if (isw(p, "exists"))
		return pquant(p, KO_EX);
	return patom(p);
}

static kt
pand(struct kp *p)
{
	kt a = punf(p), b;

	while (!p->failed && (p->tk == PT_AND || isw(p, "and"))) {
		next(p);
		b = punf(p);
		if (p->failed)
			return KNONE;
		a = kf_and(a, b);
	}
	return a;
}

static kt
por(struct kp *p)
{
	kt a = pand(p), b;

	while (!p->failed && (p->tk == PT_OR || isw(p, "or"))) {
		next(p);
		b = pand(p);
		if (p->failed)
			return KNONE;
		a = kf_or(a, b);
	}
	return a;
}

static kt
pimp(struct kp *p)
{
	kt a = por(p), b;

	if (!p->failed && (p->tk == PT_IMP || isw(p, "implies"))) {
		next(p);
		b = pimp(p);
		if (p->failed)
			return KNONE;
		return kf_imp(a, b);
	}
	return a;
}

static kt
pformula(struct kp *p)
{
	kt a = pimp(p), b;

	if (!p->failed && (p->tk == PT_IFF || isw(p, "iff"))) {
		next(p);
		b = pimp(p);
		if (p->failed)
			return KNONE;
		return kf_iff(a, b);
	}
	return a;
}

kt
kt_parse(const char *text, const struct kscope *sc, char *err, int errn)
{
	struct kp p;
	kt t;

	memset(&p, 0, sizeof p);
	p.s = text;
	p.sc = sc;
	p.err = err;
	p.errn = errn;
	if (errn > 0)
		err[0] = 0;
	next(&p);
	t = pformula(&p);
	if (!p.failed && p.tk != PT_END)
		perr(&p, "unexpected %s", tokstr(&p));
	return p.failed ? KNONE : t;
}

kt
kt_parse_term(const char *text, const struct kscope *sc, int sort,
    char *err, int errn)
{
	struct kp p;
	kt t;

	memset(&p, 0, sizeof p);
	p.s = text;
	p.sc = sc;
	p.err = err;
	p.errn = errn;
	if (errn > 0)
		err[0] = 0;
	next(&p);
	t = pterm(&p, sort);
	if (!p.failed && p.tk != PT_END)
		perr(&p, "unexpected %s", tokstr(&p));
	return p.failed ? KNONE : t;
}
