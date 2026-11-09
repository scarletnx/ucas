/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: targets, costs, proofs and the rewrite pipeline.
 *
 * Per function: if the whole function lowers to one value of its
 * parameters (cf_lower_func), that value is optimised and, when a proved
 * form is cheaper on the target, the body is replaced by it.  Otherwise
 * each maximal pure integer expression in the body is lowered on its own
 * (its variables and loads are the inputs) and optimised in place.
 *
 * Candidates, each checked against the original value on its whole
 * domain (enumerated when small, else sampled and then proved by SAT;
 * "sampled, not proved" is never applied):
 *   - the value is a constant, or one of its inputs;
 *   - bit idioms of one input: popcount, clz, ctz, bit length,
 *     x & -x, x & (x - 1), parity, abs, x != 0 ...;
 *   - a switch-like value of one input (a chain of x == c tests) as a
 *     range check and a table, a bit mask or a linear function;
 *   - a value of one small input as a table;
 *   - dropping an operation that never bites (a mask, clamp, min, max,
 *     ?: or conversion replaced by an operand);
 *   - divisions: 64-bit ones whose operands fit in 32 bits narrowed,
 *     signed ones of a non-negative dividend made unsigned, division by
 *     a constant on a bounded dividend as one multiply and shift;
 *   - more candidates from the x86 side (cf_add_cand_hook).
 * The cheapest proved candidate under the target's cost model wins.
 */
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfront.h"

/* ---- targets ---- */

#define V2 (CF_F(CF_SSE2) | CF_F(CF_SSE3) | CF_F(CF_SSSE3) | CF_F(CF_SSE41) | \
    CF_F(CF_SSE42) | CF_F(CF_POPCNT))
#define V3 (V2 | CF_F(CF_AVX) | CF_F(CF_AVX2) | CF_F(CF_BMI1) | \
    CF_F(CF_BMI2) | CF_F(CF_FMA) | CF_F(CF_LZCNT) | CF_F(CF_MOVBE))
#define V4 (V3 | CF_F(CF_AVX512F) | CF_F(CF_AVX512BW) | CF_F(CF_AVX512DQ) | \
    CF_F(CF_AVX512VL))

/*
 * Default scalar numbers (cycles, a blend of latency and reciprocal
 * throughput): Agner Fog's instruction tables and uops.info for the
 * divisions (32/64-bit DIV: Skylake 26/35-88, Ice Lake and later 12/15,
 * Zen 3/4 10-14/14-18); the x86 side's model replaces these.
 */
static struct cf_target targets[32] = {
	{"x86-64", "x86-64", CF_F(CF_SSE2), "baseline x86-64 (SSE2)",
	    1, 3, 26, 40, 1, 12, 4, 15},
	{"x86-64-v2", "x86-64-v2", V2, "SSE4.2, POPCNT", 1, 3, 26, 40, 1, 12,
	    4, 15},
	{"x86-64-v3", "x86-64-v3", V3, "AVX2, BMI1/2, FMA, LZCNT", 1, 3, 26,
	    40, 1, 12, 4, 15},
	{"x86-64-v4", "x86-64-v4", V4, "AVX-512 F/BW/DQ/VL", 1, 3, 14, 18, 1,
	    12, 4, 15},
	{"skylake", "skylake", V3, "Intel Skylake / Coffee Lake client",
	    1, 3, 26, 42, 1, 12, 4, 16},
	{"skylake-avx512", "skylake-avx512", V4, "Intel Skylake-SP/X",
	    1, 3, 26, 42, 1, 12, 4, 16},
	{"icelake", "icelake-client", V4 | CF_F(CF_AVX512VBMI) |
	    CF_F(CF_AVX512VPOPCNTDQ) | CF_F(CF_GFNI), "Intel Ice Lake client",
	    1, 3, 12, 15, 1, 12, 5, 16},
	{"alderlake", "alderlake", V3 | CF_F(CF_GFNI), "Intel Alder Lake / "
	    "Raptor Lake P-core (Golden Cove)", 1, 3, 12, 15, 1, 12, 5, 17},
	{"raptorlake", "raptorlake", V3 | CF_F(CF_GFNI), "Intel Raptor Lake "
	    "P-core", 1, 3, 12, 15, 1, 12, 5, 17},
	{"sapphirerapids", "sapphirerapids", V4 | CF_F(CF_AVX512VBMI) |
	    CF_F(CF_AVX512VPOPCNTDQ) | CF_F(CF_GFNI), "Intel Sapphire Rapids "
	    "(Golden Cove)", 1, 3, 12, 15, 1, 12, 5, 17},
	{"zen3", "znver3", V3, "AMD Zen 3", 1, 3, 10, 14, 1, 12, 4, 13},
	{"zen4", "znver4", V4 | CF_F(CF_AVX512VBMI) | CF_F(CF_AVX512VPOPCNTDQ) |
	    CF_F(CF_GFNI), "AMD Zen 4", 1, 3, 10, 14, 1, 12, 4, 13},
	{NULL, NULL, 0, NULL, 0, 0, 0, 0, 0, 0, 0, 0}
};

const struct cf_target *
cf_target_find(const char *name)
{
	int k;

	for (k = 0; targets[k].name; k++)
		if (!strcmp(targets[k].name, name) ||
		    !strcmp(targets[k].march, name))
			return &targets[k];
	if (!strcmp(name, "goldencove"))
		return cf_target_find("alderlake");
	if (!strcmp(name, "native") || !strcmp(name, "coffeelake"))
		return cf_target_find("skylake");
	return NULL;
}

const struct cf_target *
cf_target_nth(int i)
{
	int k;

	for (k = 0; targets[k].name; k++)
		if (k == i)
			return &targets[k];
	return NULL;
}

const struct cf_target *
cf_target_default(void)
{
	return cf_target_find("x86-64-v3");
}

void
cf_target_register(const struct cf_target *t)
{
	int k;

	for (k = 0; targets[k].name; k++)
		if (!strcmp(targets[k].name, t->name)) {
			targets[k] = *t;
			return;
		}
	if (k < 31) {
		targets[k] = *t;
		memset(&targets[k + 1], 0, sizeof targets[k + 1]);
	}
}

static const char *const feat_names[CF_NFEAT][2] = {
	{"sse2", "__SSE2__"}, {"sse3", "__SSE3__"}, {"ssse3", "__SSSE3__"},
	{"sse4.1", "__SSE4_1__"}, {"sse4.2", "__SSE4_2__"},
	{"popcnt", "__POPCNT__"}, {"lzcnt", "__LZCNT__"}, {"bmi", "__BMI__"},
	{"bmi2", "__BMI2__"}, {"avx", "__AVX__"}, {"avx2", "__AVX2__"},
	{"fma", "__FMA__"}, {"avx512f", "__AVX512F__"},
	{"avx512bw", "__AVX512BW__"}, {"avx512dq", "__AVX512DQ__"},
	{"avx512vl", "__AVX512VL__"}, {"avx512vbmi", "__AVX512VBMI__"},
	{"avx512vpopcntdq", "__AVX512VPOPCNTDQ__"}, {"gfni", "__GFNI__"},
	{"pclmul", "__PCLMUL__"}, {"aes", "__AES__"}, {"movbe", "__MOVBE__"},
	{"adx", "__ADX__"}
};

const char *
cf_feat_macro(int f)
{
	return f >= 0 && f < CF_NFEAT ? feat_names[f][1] : "";
}

const char *
cf_feat_name(int f)
{
	return f >= 0 && f < CF_NFEAT ? feat_names[f][0] : "";
}

/* ---- hooks ---- */

static cf_cost_fn cost_hook;
static cf_cand_fn cand_hooks[8];
static int ncand_hooks;
static cf_vec_fn vec_hook;

void
cf_set_cost_hook(cf_cost_fn f)
{
	cost_hook = f;
}

void
cf_add_cand_hook(cf_cand_fn f)
{
	if (ncand_hooks < 8)
		cand_hooks[ncand_hooks++] = f;
}

void
cf_set_vectoriser(cf_vec_fn f)
{
	vec_hook = f;
}

void
cf_cand_add(struct cf_cands *cs, nid sem, const char *text,
    const char *fallback, uint64_t feat, double cost, const char *what)
{
	struct cf_cand *c;

	if (sem == 0)
		return;
	if (cs->n == cs->cap) {
		cs->cap = cs->cap ? cs->cap * 2 : 32;
		cs->c = xrealloc(cs->c, (size_t)cs->cap * sizeof *cs->c);
	}
	c = &cs->c[cs->n++];
	c->sem = sem;
	c->text = text ? strdup(text) : NULL;
	c->fallback = fallback ? strdup(fallback) : NULL;
	c->feat = feat;
	c->cost = cost;
	snprintf(c->what, sizeof c->what, "%s", what ? what : "");
	c->lemma[0] = 0;
}

static void
last_lemma(struct cf_cands *cs, nid sem, const char *fmt, ...)
{
	va_list ap;

	if (sem == 0 || cs->n == 0 || cs->c[cs->n - 1].sem != sem)
		return;
	va_start(ap, fmt);
	vsnprintf(cs->c[cs->n - 1].lemma, sizeof cs->c[cs->n - 1].lemma, fmt,
	    ap);
	va_end(ap);
}

static void
cands_free(struct cf_cands *cs)
{
	int k;

	for (k = 0; k < cs->n; k++) {
		free((char *)cs->c[k].text);
		free((char *)cs->c[k].fallback);
	}
	free(cs->c);
	memset(cs, 0, sizeof *cs);
}

const char *
cf_proof_name(int k)
{
	static const char *const n[] = {"not proved", "exhaustive", "sat",
	    "algebra", "induction", "structural", "lanes", "tested"};

	return k >= 0 && k <= CF_PROOF_TESTED ? n[k] : "?";
}

/* ---- costs ---- */

static double
op_cost(const struct cf_target *t, nid n)
{
	int op = N.op[n], w = TYW(N.ty[n]);
	nid b = N.b[n];

	switch (op) {
	case O_CONST:
	case O_VAR:
	case O_TABLE:
	case O_UNDEF:
		return 0;
	case O_CAST:
		/* a zero/sign extension or a truncation: often free */
		return 0.25;
	case O_ADD: case O_SUB: case O_AND: case O_OR: case O_XOR:
	case O_NOT: case O_NEG: case O_SHL: case O_SHR: case O_LNOT:
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE:
		return t->c_alu;
	case O_LAND:
	case O_LOR:
		return t->c_alu;
	case O_SEL:
		return t->c_alu * 1.5;
	case O_MIN:
	case O_MAX:
	case O_ABS:
	case O_SIGN:
		return t->c_alu * 2;
	case O_MUL:
		return t->c_mul;
	case O_DIV:
	case O_MOD:
		if (is_const(b)) {
			int64_t d = N.ki[b];
			if (d > 0 && (d & (d - 1)) == 0)
				return TYK(N.ty[n]) == K_SINT ? 3 : 1;
			return t->c_mul + 3;	/* the compiler's magic number */
		}
		return w > 32 ? t->c_div64 : t->c_div32;
	case O_POPCNT:
		return (t->feat & CF_F(CF_POPCNT)) ? t->c_bitcnt_hw :
		    t->c_bitcnt_sw;
	case O_CLZ:
		return (t->feat & CF_F(CF_LZCNT)) ? t->c_bitcnt_hw :
		    t->c_bitcnt_hw + 2;
	case O_CTZ:
		return (t->feat & CF_F(CF_BMI1)) ? t->c_bitcnt_hw :
		    t->c_bitcnt_hw + 2;
	case O_INDEX:
		return t->c_load;
	}
	return 8;
}

double
cf_cost_default(const struct cf_target *t, nid root)
{
	nid *st = xmalloc((N.n + 1) * sizeof *st);
	uint8_t *seen = xcalloc(N.n, 1);
	uint32_t sp = 0;
	double c = 0;

	st[sp++] = root;
	while (sp) {
		nid n = st[--sp];
		if (n == 0 || seen[n])
			continue;
		seen[n] = 1;
		c += op_cost(t, n);
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_TABLE)
			continue;
		if (N.op[n] == O_INDEX) {
			st[sp++] = N.b[n];
			continue;
		}
		if (N.a[n])
			st[sp++] = N.a[n];
		if (N.b[n])
			st[sp++] = N.b[n];
		if (N.c[n] && (N.op[n] == O_SEL || N.op[n] == O_FMA))
			st[sp++] = N.c[n];
	}
	free(st);
	free(seen);
	return c;
}

double
cf_cost(const struct cf_target *t, nid root)
{
	if (cost_hook)
		return cost_hook(t, root);
	return cf_cost_default(t, root);
}

/* ---- the run's state ---- */

static struct {
	const struct cf_opts *o;
	struct cf_result *r;
	struct cf_unit *u;
	clock_t t0;
	double deadline;	/* seconds of CPU from t0 */
	char fname[64];
	int nproofs;
	uint32_t fnstart;
} R;

static double
elapsed(void)
{
	return (double)(clock() - R.t0) / CLOCKS_PER_SEC;
}

static int
out_of_time(void)
{
	return elapsed() > R.deadline;
}

/* ---- proofs ---- */

#define QPTS 512

/* cand agrees with f's body on a sample (a quick filter) */
static int
quick_ok(const struct func *f, nid cand)
{
	struct diter d;
	struct plan p;
	nid roots[3];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t np, i;
	int ok = 1;

	roots[0] = f->body;
	roots[1] = mk_cast(N.ty[f->body], cand);
	roots[2] = f->where ? f->where : f->body;
	plan_build(&p, roots, 3, 0);
	diter_init(&d, f, QPTS, 2, QPTS);
	while (ok && (np = diter_next(&d, in_i, in_r)) != 0) {
		uint8_t v[CH];
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[2] : NOTOK, np, v);
		for (i = 0; i < np; i++) {
			if (!v[i] || (COLF(&p, p.root[0])[i] & F_UNDEF))
				continue;
			if ((COLF(&p, p.root[1])[i] & F_UNDEF) ||
			    COLI(&p, p.root[0])[i] != COLI(&p, p.root[1])[i]) {
				ok = 0;
				break;
			}
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return ok;
}

struct proof {
	int kind;		/* CF_PROOF_* */
	char text[160];
};

/* cand refines the value: on every input where it is defined */
static int
prove(const struct cf_value *v, nid cand, struct proof *pf)
{
	struct func f;
	struct eqproof pr;
	struct sbuf w = {0}, how = {0};
	int bad;

	cf_value_func(v, &f);
	pf->kind = CF_PROOF_NONE;
	pf->text[0] = 0;
	if (!quick_ok(&f, cand)) {
		snprintf(pf->text, sizeof pf->text, "differs on a sample");
		return 0;
	}
	if (out_of_time()) {
		snprintf(pf->text, sizeof pf->text, "out of time");
		return 0;
	}
	R.nproofs++;
	bad = equiv_decide(&f, v->e, cand, &w, &pr);
	eqproof_text(&how, &pr);
	if (bad)
		snprintf(pf->text, sizeof pf->text, "differs: %s", w.p ? w.p : "");
	else if (pr.how == EQ_EXHAUSTIVE) {
		pf->kind = CF_PROOF_EXHAUSTIVE;
		snprintf(pf->text, sizeof pf->text, "exhaustive: %llu inputs",
		    (unsigned long long)pr.points);
	} else if (pr.how == EQ_SAT) {
		pf->kind = CF_PROOF_SAT;
		snprintf(pf->text, sizeof pf->text, "%s", how.p);
	} else
		snprintf(pf->text, sizeof pf->text, "%s", how.p);
	free(w.p);
	free(how.p);
	if (!bad && pf->kind == CF_PROOF_NONE)
		return 2;	/* sampled: no difference, no proof */
	return !bad && pf->kind != CF_PROOF_NONE;
}

/* ---- rewriting helpers ---- */

static nid
safe_replace(nid root, nid from, nid to)
{
	jmp_buf saved;
	nid r = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = xreplace(root, from, to);
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return r;
}

static nid
safe_cast(ty_t t, nid a)
{
	jmp_buf saved;
	nid r = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = mk_cast(t, a);
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	return r;
}

/* the nodes reachable from root, in id order */
static uint32_t
reach(nid root, nid **out)
{
	uint8_t *seen = xcalloc(N.n, 1);
	nid *st = xmalloc((N.n + 1) * sizeof *st), *r;
	uint32_t sp = 0, n = 0, k;

	st[sp++] = root;
	while (sp) {
		nid x = st[--sp];
		if (x == 0 || seen[x])
			continue;
		seen[x] = 1;
		if (N.op[x] == O_CONST || N.op[x] == O_VAR || N.op[x] == O_TABLE)
			continue;
		if (N.op[x] == O_INDEX) {
			st[sp++] = N.b[x];
			continue;
		}
		if (N.a[x]) st[sp++] = N.a[x];
		if (N.b[x]) st[sp++] = N.b[x];
		if (N.c[x] && (N.op[x] == O_SEL || N.op[x] == O_FMA))
			st[sp++] = N.c[x];
	}
	r = xmalloc((N.n + 1) * sizeof *r);
	for (k = 0; k < N.n; k++)
		if (seen[k])
			r[n++] = k;
	free(seen);
	free(st);
	*out = r;
	return n;
}

/* the one input slot e depends on, or -1 */
static int
single_slot(nid e)
{
	uint64_t s = node_slots(e);
	int k;

	if (s == 0 || (s & (s - 1)))
		return -1;
	for (k = 0; k < 64; k++)
		if (s >> k & 1)
			return k;
	return -1;
}

/* the leaf node of slot k */
static nid
leaf_node(const struct cf_value *v, int k)
{
	return mk_var((uint32_t)k, sym_intern(v->leaf[k].text,
	    strlen(v->leaf[k].text)), ct_to_ucas(v->leaf[k].cty));
}

/* values of e at x = lo..hi of slot k (other inputs absent); undefined
 * points give 0 and *undef counts them */
static int
eval_line(const struct cf_value *v, nid e, int k, int64_t lo, int64_t hi,
    int64_t *out, int *undef)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int64_t x = lo;
	uint32_t i, np, loc;
	ty_t t = ct_to_ucas(v->leaf[k].cty);

	plan_build(&p, &e, 1, 0);
	loc = p.root[0];
	*undef = 0;
	while (x <= hi) {
		np = 0;
		while (np < CH && x <= hi) {
			in_i[(size_t)k * CH + np] = int_norm(t, x);
			np++;
			x++;
		}
		plan_run(&p, np, in_i, in_r);
		for (i = 0; i < np; i++) {
			if (COLF(&p, loc)[i] & F_UNDEF) {
				out[x - np + i - lo] = 0;
				(*undef)++;
			} else
				out[x - np + i - lo] = COLI(&p, loc)[i];
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	return 1;
}

/* ---- candidate generators ---- */

static void
gen_basic(const struct cf_value *v, nid e, struct cf_cands *cs)
{
	ty_t t = N.ty[e];
	int k;

	/* a constant: the value at the first sample point */
	{
		struct plan p;
		int64_t in_i[MAXP * CH];
		double in_r[MAXP * CH];
		memset(in_i, 0, sizeof in_i);
		memset(in_r, 0, sizeof in_r);
		plan_build(&p, &e, 1, 0);
		plan_run(&p, 1, in_i, in_r);
		if (!(COLF(&p, p.root[0])[0] & F_UNDEF))
			cf_cand_add(cs, mk_const_i(t, COLI(&p, p.root[0])[0]),
			    NULL, NULL, 0, 0, "the value is a constant");
		plan_free(&p);
	}
	for (k = 0; k < v->nleaf; k++) {
		nid x = leaf_node(v, k), c;
		char w[96];
		if (!has_slot(e, k))
			continue;
		if ((c = safe_cast(t, x)) != 0) {
			snprintf(w, sizeof w, "the value is %s", v->leaf[k].text);
			cf_cand_add(cs, c, NULL, NULL, 0, 0, w);
		}
	}
}

static void
gen_idioms(const struct cf_value *v, nid e, struct cf_cands *cs)
{
	ty_t t = N.ty[e];
	int k, have0 = 0;
	int64_t at0 = 0;

	{
		/* the value with every input 0 */
		struct plan p;
		int64_t in_i[MAXP * CH];
		double in_r[MAXP * CH];
		memset(in_i, 0, sizeof in_i);
		memset(in_r, 0, sizeof in_r);
		plan_build(&p, &e, 1, 0);
		plan_run(&p, 1, in_i, in_r);
		if (!(COLF(&p, p.root[0])[0] & F_UNDEF)) {
			have0 = 1;
			at0 = COLI(&p, p.root[0])[0];
		}
		plan_free(&p);
	}

	for (k = 0; k < v->nleaf; k++) {
		nid x = leaf_node(v, k), ux, c;
		ty_t xt = N.ty[x];
		int w = TYW(xt);
		char d[96];
		const char *nm = v->leaf[k].text;
		if (!has_slot(e, k) || !TY_SIZED(xt) || xt == TY_BOOL)
			continue;
		ux = TYK(xt) == K_SINT ? mk_cast(TY(K_UINT, w), x) : x;
		struct {
			nid e;
			const char *f;
		} f[12];
		int nf = 0, j;
		f[nf].e = mk_un(O_POPCNT, ux); f[nf++].f = "popcount(%s)";
		f[nf].e = mk_un(O_CLZ, ux); f[nf++].f = "clz(%s)";
		f[nf].e = mk_un(O_CTZ, ux); f[nf++].f = "ctz(%s)";
		f[nf].e = mk_bin(O_SUB, mk_const_i(TY(K_UINT, w), w),
		    mk_un(O_CLZ, ux));
		f[nf++].f = "bit length of %s (width - clz)";
		f[nf].e = mk_bin(O_SUB, mk_const_i(TY(K_UINT, w), w - 1),
		    mk_un(O_CLZ, ux));
		f[nf++].f = "floor(log2(%s)) (width - 1 - clz)";
		f[nf].e = mk_bin(O_AND, mk_un(O_POPCNT, ux), mk_const_i(TY(K_UINT,
		    w), 1));
		f[nf++].f = "parity of %s";
		f[nf].e = mk_bin(O_AND, ux, mk_un(O_NEG, ux));
		f[nf++].f = "%s & -%s (lowest set bit)";
		f[nf].e = mk_bin(O_AND, ux, mk_bin(O_SUB, ux, mk_const_i(TY(K_UINT,
		    w), 1)));
		f[nf++].f = "%s & (%s - 1) (clear the lowest set bit)";
		f[nf].e = mk_bin(O_NE, x, mk_const_i(xt, 0));
		f[nf++].f = "%s != 0";
		if (TYK(xt) == K_SINT) {
			f[nf].e = mk_un(O_ABS, x);
			f[nf++].f = "abs(%s)";
		}
		for (j = 0; j < nf; j++) {
			if ((c = safe_cast(t, f[j].e)) == 0)
				continue;
			snprintf(d, sizeof d, f[j].f, nm, nm);
			cf_cand_add(cs, c, NULL, NULL, 0, 0, d);
			/* the loop forms often special-case 0 */
			if (have0 && j < 6) {
				char d2[96];
				nid z = mk_sel(mk_bin(O_EQ, x, mk_const_i(xt, 0)),
				    mk_const_i(t, at0), c);
				snprintf(d2, sizeof d2, "%.60s, %lld at 0", d,
				    (long long)at0);
				cf_cand_add(cs, z, NULL, NULL, 0, 0, d2);
			}
		}
	}
}

/* a value of one input with x == c tests: range check + table/mask/line */
static void
gen_switch(const struct cf_value *v, nid e, struct cf_cands *cs)
{
	int k = single_slot(e), undef, j, nvals = 0;
	nid *rn, x;
	uint32_t nr, i;
	int64_t lo = INT64_MAX, hi = INT64_MIN, span, *tab, out[8], d;
	ty_t t = N.ty[e], xt;
	int64_t probe[8];

	if (k < 0 || k >= v->nleaf || !TY_SIZED(t) || t == TY_BOOL)
		return;
	x = leaf_node(v, k);
	xt = N.ty[x];
	if (!TY_SIZED(xt) || xt == TY_BOOL)
		return;
	nr = reach(e, &rn);
	for (i = 0; i < nr; i++) {
		nid n = rn[i], a = N.a[n], b = N.b[n];
		int64_t c;
		if (N.op[n] != O_EQ && N.op[n] != O_NE)
			continue;
		if (!is_const(b) || !(a == x || (N.op[a] == O_CAST &&
		    N.a[a] == x)))
			continue;
		c = N.ki[b];
		if (N.op[a] == O_CAST && int_norm(xt, c) != c)
			continue;
		nvals++;
		if (c < lo)
			lo = c;
		if (c > hi)
			hi = c;
	}
	free(rn);
	if (nvals < 3 || hi - lo > 1023)
		return;
	span = hi - lo + 1;
	tab = xmalloc((size_t)span * sizeof *tab);
	eval_line(v, e, k, lo, hi, tab, &undef);
	/* the value outside [lo, hi]: must be one constant */
	{
		int64_t tmin = TYK(xt) == K_UINT ? 0 : (TYW(xt) >= 64 ? INT64_MIN :
		    -((int64_t)1 << (TYW(xt) - 1)));
		int64_t tmax = TYK(xt) == K_UINT ? (TYW(xt) >= 64 ? -1 :
		    (int64_t)((1ULL << TYW(xt)) - 1)) : (TYW(xt) >= 64 ? INT64_MAX :
		    ((int64_t)1 << (TYW(xt) - 1)) - 1);
		int np = 0, u2;
		if (lo > tmin)
			probe[np++] = lo - 1, probe[np++] = tmin;
		if (hi < tmax || (TYK(xt) == K_UINT && TYW(xt) >= 64))
			probe[np++] = hi + 1, probe[np++] = tmax;
		probe[np++] = int_norm(xt, (int64_t)0x5bd1e995);
		probe[np++] = int_norm(xt, (int64_t)0x7fffff17);
		d = 0;
		for (j = 0; j < np; j++) {
			if (probe[j] >= lo && probe[j] <= hi)
				continue;
			eval_line(v, e, k, probe[j], probe[j], &out[0], &u2);
			if (u2)
				continue;
			if (j == 0 || nvals == -1)
				d = out[0];
			if (out[0] != d) {
				free(tab);
				return;
			}
			nvals = -1;
		}
	}
	{
		/* the index: (unsigned)(x - lo), and the range check */
		ty_t ut = TYW(xt) > 32 ? TY(K_UINT, 64) : TY(K_UINT, 32);
		nid xi = mk_cast(ut, x), off, inr, idx, body, cand;
		int bits01 = 1, linear = 1;
		int64_t a1 = span > 1 ? tab[1] - tab[0] : 0;
		char w[96];
		off = mk_bin(O_SUB, xi, mk_const_i(ut, lo));
		inr = mk_bin(O_LE, off, mk_const_i(ut, span - 1));
		idx = mk_cast(TY(K_UINT, 64), off);
		for (j = 0; j < span; j++) {
			if (tab[j] != 0 && tab[j] != 1)
				bits01 = 0;
			if (tab[j] != tab[0] + a1 * j)
				linear = 0;
		}
		if (linear) {
			body = mk_cast(t, mk_bin(O_ADD, mk_bin(O_MUL,
			    mk_const_i(ut, a1), off), mk_const_i(ut, tab[0])));
			cand = mk_sel(inr, body, mk_const_i(t, d));
			snprintf(w, sizeof w, "switch on %s: a range check and "
			    "%lld*(x - %lld) + %lld", v->leaf[k].text, (long long)a1,
			    (long long)lo, (long long)tab[0]);
			cf_cand_add(cs, cand, NULL, NULL, 0, 0, w);
			if (d == tab[0] - a1 || d == tab[0] + a1 * span) {
				/* the line continues outside: no check at all? */
				cand = mk_cast(t, mk_bin(O_ADD, mk_bin(O_MUL,
				    mk_const_i(ut, a1), off), mk_const_i(ut, tab[0])));
				cf_cand_add(cs, cand, NULL, NULL, 0, 0, w);
			}
		}
		if (bits01 && span <= 64) {
			uint64_t m = 0;
			ty_t mt = span <= 32 ? TY(K_UINT, 32) : TY(K_UINT, 64);
			for (j = 0; j < span; j++)
				if (tab[j])
					m |= 1ULL << j;
			body = mk_cast(t, mk_bin(O_AND, mk_bin(O_SHR,
			    mk_const_i(mt, (int64_t)m), mk_cast(mt, off)),
			    mk_const_i(mt, 1)));
			cand = mk_sel(inr, body, mk_const_i(t, d));
			snprintf(w, sizeof w, "switch on %s: a range check and the "
			    "bit mask 0x%llx", v->leaf[k].text, (unsigned long long)m);
			cf_cand_add(cs, cand, NULL, NULL, 0, 0, w);
		}
		{
			nid *el = xmalloc((size_t)span * sizeof *el);
			for (j = 0; j < span; j++)
				el[j] = mk_const_i(t, tab[j]);
			body = mk_index(mk_table(el, (uint32_t)span), idx);
			free(el);
			cand = mk_sel(inr, body, mk_const_i(t, d));
			snprintf(w, sizeof w, "switch on %s: a range check and a "
			    "table of %lld", v->leaf[k].text, (long long)span);
			cf_cand_add(cs, cand, NULL, NULL, 0, 0, w);
		}
	}
	free(tab);
}

/* a value of one input with at most 256 values: a table */
static void
gen_table(const struct cf_value *v, nid e, struct cf_cands *cs)
{
	int k = single_slot(e), undef, j;
	int64_t lo, hi, *tab;
	ty_t t = N.ty[e], xt;
	nid x, *el, idx, c;
	char w[96];

	if (k < 0 || k >= v->nleaf || !TY_SIZED(t))
		return;
	x = leaf_node(v, k);
	xt = N.ty[x];
	if (v->leaf[k].ranged) {
		lo = v->leaf[k].lo;
		hi = v->leaf[k].hi;
	} else if (TYW(xt) <= 8 && TY_SIZED(xt)) {
		lo = TYK(xt) == K_SINT ? -128 : 0;
		hi = TYK(xt) == K_SINT ? 127 : (xt == TY_BOOL ? 1 : 255);
	} else
		return;
	if (hi - lo > 255 || hi < lo)
		return;
	tab = xmalloc((size_t)(hi - lo + 1) * sizeof *tab);
	eval_line(v, e, k, lo, hi, tab, &undef);
	el = xmalloc((size_t)(hi - lo + 1) * sizeof *el);
	for (j = 0; j <= hi - lo; j++)
		el[j] = mk_const_i(t, tab[j]);
	idx = mk_cast(TY(K_SINT, 64), x);
	if (lo)
		idx = mk_bin(O_SUB, idx, mk_const_i(TY(K_SINT, 64), lo));
	c = mk_index(mk_table(el, (uint32_t)(hi - lo + 1)), idx);
	snprintf(w, sizeof w, "a table of %lld values indexed by %s",
	    (long long)(hi - lo + 1), v->leaf[k].text);
	cf_cand_add(cs, c, NULL, NULL, 0, 0, w);
	free(el);
	free(tab);
}

/*
 * A value of one bounded input that is a polynomial of degree <= 3 in it
 * (up to a point after which it stays constant, as a loop with a capped
 * trip count gives): Newton's forward differences are integers, so
 * f0 + D1 u + D2 u(u-1)/2 + D3 u(u-1)(u-2)/6 is exact in 64 bits.
 */
static void
gen_poly(const struct cf_value *v, nid e, struct cf_cands *cs)
{
	int k = single_slot(e), undef, deg, j;
	int64_t lo, hi, *tab, c, n;
	ty_t t = N.ty[e], xt, w = TY(K_SINT, 64);
	nid x;

	if (k < 0 || k >= v->nleaf || !TY_SIZED(t) || t == TY_BOOL)
		return;
	x = leaf_node(v, k);
	xt = N.ty[x];
	if (v->leaf[k].ranged) {
		lo = v->leaf[k].lo;
		hi = v->leaf[k].hi;
	} else if (TY_SIZED(xt) && TYW(xt) <= 16 && xt != TY_BOOL) {
		lo = TYK(xt) == K_SINT ? -((int64_t)1 << (TYW(xt) - 1)) : 0;
		hi = TYK(xt) == K_SINT ? ((int64_t)1 << (TYW(xt) - 1)) - 1 :
		    ((int64_t)1 << TYW(xt)) - 1;
	} else
		return;
	if (hi - lo > 65535 || hi - lo < 8)
		return;
	n = hi - lo + 1;
	tab = xmalloc((size_t)n * sizeof *tab);
	eval_line(v, e, k, lo, hi, tab, &undef);
	if (undef) {
		free(tab);
		return;
	}
	/* the plateau: tab[c..] all equal */
	for (c = n - 1; c > 0 && tab[c - 1] == tab[n - 1]; c--)
		;
	for (deg = 1; deg <= 3; deg++) {
		int64_t d[4], ok = 1;
		if (c < deg + 2)
			break;
		d[0] = tab[0];
		d[1] = tab[1] - tab[0];
		d[2] = tab[2] - 2 * tab[1] + tab[0];
		d[3] = tab[3] - 3 * tab[2] + 3 * tab[1] - tab[0];
		for (j = deg + 1; j <= 3; j++)
			d[j] = 0;
		for (j = 0; j <= c && ok; j++) {
			/* u = j: the polynomial at j, exactly */
			int64_t u = j, p = d[0] + d[1] * u + d[2] * (u * (u - 1) / 2) +
			    d[3] * (u * (u - 1) * (u - 2) / 6);
			if (int_norm(t, p) != tab[j])
				ok = 0;
		}
		if (ok) {
			nid u, r, u1, u2;
			char what[120];
			u = mk_cast(w, x);
			if (c < n - 1)
				u = mk_bin(O_MIN, u, mk_const_i(w, lo + c));
			if (lo)
				u = mk_bin(O_SUB, u, mk_const_i(w, lo));
			u1 = mk_bin(O_SUB, u, mk_const_i(w, 1));
			u2 = mk_bin(O_SUB, u, mk_const_i(w, 2));
			r = mk_const_i(w, d[0]);
			if (d[1])
				r = mk_bin(O_ADD, r, mk_bin(O_MUL, mk_const_i(w, d[1]), u));
			if (d[2])
				r = mk_bin(O_ADD, r, mk_bin(O_MUL, mk_const_i(w, d[2]),
				    mk_bin(O_DIV, mk_bin(O_MUL, u, u1), mk_const_i(w, 2))));
			if (d[3])
				r = mk_bin(O_ADD, r, mk_bin(O_MUL, mk_const_i(w, d[3]),
				    mk_bin(O_DIV, mk_bin(O_MUL, mk_bin(O_MUL, u, u1), u2),
				    mk_const_i(w, 6))));
			r = mk_cast(t, r);
			snprintf(what, sizeof what, "a polynomial of degree %d in %s%s",
			    deg, v->leaf[k].text, c < n - 1 ? " (capped: constant "
			    "from there on)" : "");
			cf_cand_add(cs, r, NULL, NULL, 0, 0, what);
			break;
		}
	}
	free(tab);
}

/* ---- local rewrites (one node at a time) ---- */

struct ivals {
	struct plan p;
	struct ival *iv;
	int ok;
};

static void
ivals_build(struct ivals *I, const struct cf_value *v, nid e)
{
	struct func f;

	cf_value_func(v, &f);
	plan_build(&I->p, &e, 1, 0);
	I->iv = xcalloc(I->p.n + 1, sizeof *I->iv);
	plan_ival(&I->p, &f, I->iv);
	I->ok = 1;
}

static void
ivals_free(struct ivals *I)
{
	plan_free(&I->p);
	free(I->iv);
}

static int
ival_of(struct ivals *I, nid n, int64_t *lo, int64_t *hi)
{
	uint32_t l = plan_loc(&I->p, n);

	/* a wrapped interval is the type's full range: still sound, except
	 * for u64, whose range int64 cannot hold */
	if (l == NOTOK || I->iv[l].top || (I->iv[l].wrap &&
	    TYK(N.ty[n]) == K_UINT && TYW(N.ty[n]) == 64) ||
	    I->iv[l].lo > I->iv[l].hi)
		return 0;
	*lo = I->iv[l].lo;
	*hi = I->iv[l].hi;
	return 1;
}

/* the smallest (M, s) with floor(x*M / 2^s) = floor(x/d) on [0, X] */
static int
magic(uint64_t d, uint64_t X, uint64_t *M, int *s, int *w32)
{
	int sh;

	for (sh = 0; sh < 64; sh++) {
		uint64_t p = 1ULL << sh, m = (p + d - 1) / d, e = m * d - p;
		if (p / d + 1 < m)	/* m overflowed */
			break;
		if (e != 0 && X > (p - 1) / e)
			continue;
		if (m != 0 && X > UINT64_MAX / m)
			continue;
		*M = m;
		*s = sh;
		*w32 = m == 0 || X <= 0xffffffffULL / m;
		return 1;
	}
	return 0;
}

static void
gen_local(const struct cf_value *v, nid e, struct cf_cands *cs)
{
	nid *rn;
	uint32_t nr, i;
	struct ivals I;
	char w[120];

	ivals_build(&I, v, e);
	nr = reach(e, &rn);
	for (i = 0; i < nr; i++) {
		nid n = rn[i], a = N.a[n], b = N.b[n], r;
		int op = N.op[n];
		ty_t t = N.ty[n];
		int64_t alo, ahi, blo, bhi;
		if (!TY_SIZED(t) || t == TY_BOOL)
			goto drops;
		if (getenv("CF_DEBUG") && (op == O_DIV || op == O_MOD)) {
			uint32_t la = plan_loc(&I.p, a), lb = plan_loc(&I.p, b);
			fprintf(stderr, "cfront: div %s: a [%lld, %lld] top %d wrap %d, "
			    "b [%lld, %lld] top %d wrap %d\n", shared_str(n, 100),
			    la == NOTOK ? 0LL : (long long)I.iv[la].lo,
			    la == NOTOK ? 0LL : (long long)I.iv[la].hi,
			    la == NOTOK ? -1 : I.iv[la].top, la == NOTOK ? -1 :
			    I.iv[la].wrap, lb == NOTOK ? 0LL : (long long)I.iv[lb].lo,
			    lb == NOTOK ? 0LL : (long long)I.iv[lb].hi,
			    lb == NOTOK ? -1 : I.iv[lb].top, lb == NOTOK ? -1 :
			    I.iv[lb].wrap);
		}
		if ((op == O_DIV || op == O_MOD) && ival_of(&I, a, &alo, &ahi) &&
		    alo >= 0) {
			int have_b = ival_of(&I, b, &blo, &bhi);
			/* 64-bit operands that fit in 32 bits */
			if (TYW(t) == 64 && have_b && blo >= 0 &&
			    (uint64_t)ahi <= 0xffffffffULL &&
			    (uint64_t)bhi <= 0xffffffffULL && !is_const(b)) {
				ty_t u = TY(K_UINT, 32);
				r = mk_cast(t, mk_bin(op, mk_cast(u, a), mk_cast(u, b)));
				snprintf(w, sizeof w, "a 64-bit %s on values below 2^32 "
				    "done in 32 bits", op == O_DIV ? "division" :
				    "remainder");
				r = safe_replace(e, n, r);
				cf_cand_add(cs, r, NULL, NULL, 0, 0, w);
				last_lemma(cs, r, "intervals: dividend in [%lld, %lld], "
				    "divisor in [%lld, %lld], so the 32-bit %s is exact",
				    (long long)alo, (long long)ahi, (long long)blo,
				    (long long)bhi, op == O_DIV ? "quotient" : "remainder");
			}
			/* signed with a non-negative dividend and divisor */
			if (TYK(t) == K_SINT && have_b && blo > 0) {
				ty_t u = TY(K_UINT, TYW(t));
				r = mk_cast(t, mk_bin(op, mk_cast(u, a), mk_cast(u, b)));
				snprintf(w, sizeof w, "a signed %s of non-negative values "
				    "done unsigned", op == O_DIV ? "division" :
				    "remainder");
				r = safe_replace(e, n, r);
				cf_cand_add(cs, r, NULL, NULL, 0, 0, w);
				last_lemma(cs, r, "intervals: dividend in [%lld, %lld], "
				    "divisor in [%lld, %lld]: signed and unsigned agree",
				    (long long)alo, (long long)ahi, (long long)blo,
				    (long long)bhi);
			}
			/* a constant divisor on a bounded dividend */
			if (is_const(b) && N.ki[b] > 1 && (N.ki[b] & (N.ki[b] - 1)) &&
			    TYW(t) <= 64) {
				uint64_t M;
				int s, w32;
				if (magic((uint64_t)N.ki[b], (uint64_t)ahi, &M, &s,
				    &w32)) {
					ty_t u = w32 ? TY(K_UINT, 32) : TY(K_UINT, 64);
					nid q = mk_bin(O_SHR, mk_bin(O_MUL, mk_cast(u, a),
					    mk_const_i(u, (int64_t)M)), mk_const_i(u, s));
					if (op == O_MOD)
						q = mk_bin(O_SUB, mk_cast(u, a), mk_bin(O_MUL, q,
						    mk_const_i(u, N.ki[b])));
					r = mk_cast(t, q);
					snprintf(w, sizeof w, "x %s %lld for x <= %lld: "
					    "(x * %llu) >> %d in %d bits", op == O_DIV ?
					    "/" : "%", (long long)N.ki[b], (long long)ahi,
					    (unsigned long long)M, s, w32 ? 32 : 64);
					r = safe_replace(e, n, r);
					cf_cand_add(cs, r, NULL, NULL, 0, 0, w);
					last_lemma(cs, r, "interval: x in [%lld, %lld]; "
					    "e = M*d - 2^s = %llu and e*%lld < 2^%d, so "
					    "(x*M) >> s = x/d", (long long)alo, (long long)ahi,
					    (unsigned long long)(M * (uint64_t)N.ki[b] -
					    (1ULL << s)), (long long)ahi, s);
				}
			}
		}
drops:
		/* drop an operation: replace it by an operand */
		if (op == O_SEL) {
			cf_cand_add(cs, safe_replace(e, n, N.b[n]), NULL, NULL, 0, 0,
			    "a ?: that always takes its first arm");
			cf_cand_add(cs, safe_replace(e, n, N.c[n]), NULL, NULL, 0, 0,
			    "a ?: that always takes its second arm");
		} else if (op == O_CAST && TY_ISINT(N.ty[a]) && TY_SIZED(N.ty[a])) {
			cf_cand_add(cs, safe_replace(e, n, safe_cast(t, a)), NULL,
			    NULL, 0, 0, "a conversion that never changes the value");
		} else if (op == O_AND || op == O_OR || op == O_MIN ||
		    op == O_MAX || op == O_LAND || op == O_LOR) {
			const char *nm = op == O_AND ? "a mask" : op == O_OR ? "an or" :
			    op == O_MIN ? "a min (clamp)" : op == O_MAX ?
			    "a max (clamp)" : "a logical test";
			snprintf(w, sizeof w, "%s that never bites", nm);
			cf_cand_add(cs, safe_replace(e, n, a), NULL, NULL, 0, 0, w);
			cf_cand_add(cs, safe_replace(e, n, b), NULL, NULL, 0, 0, w);
		}
	}
	free(rn);
	ivals_free(&I);
}

/* ---- the driver for one value ---- */

struct step {
	char what[120];
	struct proof pf;
	double before, after;
};

static int
cmp_cand(const void *x, const void *y)
{
	const struct cf_cand *a = x, *b = y;

	if (a->cost != b->cost)
		return a->cost < b->cost ? -1 : 1;
	return (int)a->sem - (int)b->sem;
}

/*
 * The cheapest proved form of v->e.  Global candidates first, then local
 * rewrites until none helps.  Steps are recorded.
 */
static nid
optimize_value(struct cf_value *v, struct step *steps, int *nsteps, int max,
    int allow_table)
{
	const struct cf_target *t = R.o->t;
	nid cur = v->e;
	double ccur = cf_cost(t, cur);
	int round, k;

	*nsteps = 0;
	for (round = 0; round < 12 && *nsteps < max && !out_of_time(); round++) {
		struct cf_cands cs = {0};
		int took = 0;
		if (round == 0) {
			gen_basic(v, cur, &cs);
			gen_idioms(v, cur, &cs);
			gen_switch(v, cur, &cs);
			if (allow_table)
				gen_table(v, cur, &cs);
			gen_poly(v, cur, &cs);
			for (k = 0; k < ncand_hooks; k++) {
				const char *lt[CF_MAXLEAF];
				int j;
				for (j = 0; j < v->nleaf; j++)
					lt[j] = v->leaf[j].text;
				cand_hooks[k](t, cur, v->nleaf, lt, &cs);
			}
		}
		gen_local(v, cur, &cs);
		for (k = 0; k < cs.n; k++)
			cs.c[k].cost = cf_cost(t, cs.c[k].sem);
		if (getenv("CF_DEBUG"))
			fprintf(stderr, "cfront: round %d: %d candidates, cost %.1f\n",
			    round, cs.n, ccur);
		if (getenv("CF_DEBUG") && atoi(getenv("CF_DEBUG")) > 1)
			for (k = 0; k < cs.n; k++)
				fprintf(stderr, "cfront:   %.1f %s\n", cs.c[k].cost,
				    cs.c[k].what);
		qsort(cs.c, (size_t)cs.n, sizeof *cs.c, cmp_cand);
		for (k = 0; k < cs.n && !out_of_time(); k++) {
			struct proof pf;
			if (cs.c[k].cost >= ccur - 0.01 || cs.c[k].sem == cur)
				continue;
			int pr = prove(v, cs.c[k].sem, &pf);
			if (pr == 2 && cs.c[k].lemma[0]) {
				char t[160];
				snprintf(t, sizeof t, "%s; sampled inputs agree (SAT "
				    "gave up)", cs.c[k].lemma);
				pf.kind = CF_PROOF_ALGEBRA;
				snprintf(pf.text, sizeof pf.text, "%s", t);
				pr = 1;
			}
			if (pr != 1) {
				if (getenv("CF_DEBUG"))
					fprintf(stderr, "cfront: %s: %s [%s]\n",
					    cs.c[k].what, pf.text, shared_str(cs.c[k].sem, 200));
				continue;
			}
			snprintf(steps[*nsteps].what, sizeof steps[*nsteps].what, "%s",
			    cs.c[k].what);
			steps[*nsteps].pf = pf;
			steps[*nsteps].before = ccur;
			steps[*nsteps].after = cs.c[k].cost;
			(*nsteps)++;
			cur = cs.c[k].sem;
			ccur = cs.c[k].cost;
			took = 1;
			break;
		}
		cands_free(&cs);
		if (!took)
			break;
	}
	return cur;
}

/* ---- output: replacements in the main file ---- */

struct repl {
	uint32_t from, to;
	char *text;
	uint32_t fnstart;	/* helpers go before this offset */
};

static struct repl *reps;
static int nreps, repcap;

static void
add_repl(uint32_t from, uint32_t to, const char *text, uint32_t fnstart)
{
	if (nreps == repcap) {
		repcap = repcap ? repcap * 2 : 16;
		reps = xrealloc(reps, (size_t)repcap * sizeof *reps);
	}
	reps[nreps].from = from;
	reps[nreps].to = to;
	reps[nreps].text = strdup(text);
	reps[nreps].fnstart = fnstart;
	nreps++;
}

static struct cf_rewrite *
add_rw(const char *kind, int line)
{
	struct cf_result *r = R.r;
	struct cf_rewrite *w;

	if (r->nrw == r->rwcap) {
		r->rwcap = r->rwcap ? r->rwcap * 2 : 16;
		r->rw = xrealloc(r->rw, (size_t)r->rwcap * sizeof *r->rw);
	}
	w = &r->rw[r->nrw++];
	memset(w, 0, sizeof *w);
	snprintf(w->func, sizeof w->func, "%s", R.fname);
	snprintf(w->kind, sizeof w->kind, "%s", kind);
	w->line = line;
	return w;
}

static void
apply_assumptions(struct cf_value *v)
{
	int k, j;

	for (k = 0; k < v->nleaf; k++)
		for (j = 0; j < R.o->nassume; j++)
			if (!strcmp(v->leaf[k].text, R.o->assume[j].name)) {
				v->leaf[k].ranged = 1;
				v->leaf[k].lo = R.o->assume[j].lo;
				v->leaf[k].hi = R.o->assume[j].hi;
			}
}

/* ---- call-site domains ---- */

/*
 * A static function whose address never escapes is only entered through
 * the calls in this file, so each parameter ranges over the union of
 * what those calls pass (each argument lowered and bounded by sound
 * interval arithmetic over its own inputs' types).  As in the function
 * level of simplify, the domain comes from the call sites; the rewrites
 * are proved over it.
 */
struct csdom {
	int n;			/* call sites seen */
	uint8_t known[MAXP];
	int64_t lo[MAXP], hi[MAXP];
};
static struct csdom *csd;

static void
cs_walk(struct cf_unit *u, uint32_t caller, uint32_t n, uint8_t *escape)
{
	uint32_t k;
	int kind;

	if (n == 0 || n >= u->A.n)
		return;
	kind = u->A.kind[n];
	if (kind == CA_ID) {
		uint32_t s = (uint32_t)u->A.iv[n];
		if (u->S.kind[s] == CS_FUNC)
			escape[s] = 1;	/* a function used as a value */
		return;
	}
	if (kind == CA_CALL) {
		uint32_t callee = u->A.a[n], s, fn, j;
		while (u->A.kind[callee] == CA_CAST && u->A.iv[callee] == 1)
			callee = u->A.a[callee];
		for (k = 0; k < u->A.c[n]; k++)
			cs_walk(u, caller, u->A.lst[u->A.b[n] + k], escape);
		if (u->A.kind[callee] != CA_ID) {
			cs_walk(u, caller, callee, escape);
			return;
		}
		s = (uint32_t)u->A.iv[callee];
		fn = u->S.decl[s];
		if (u->S.kind[s] != CS_FUNC || fn == 0 || u->A.kind[fn] != CA_FUNC)
			return;
		for (j = 0; j < (uint32_t)u->nfn && u->fns[j] != fn; j++)
			;
		if (j == (uint32_t)u->nfn || u->A.c[n] != u->A.d[fn] ||
		    u->A.d[fn] > MAXP)
			return;
		{
			struct csdom *d = &csd[j];
			for (k = 0; k < u->A.c[n]; k++) {
				struct cf_value v;
				uint32_t a = u->A.lst[u->A.b[n] + k];
				int64_t lo = 0, hi = 0;
				int ok = 0;
				if (cf_lower_expr(u, caller, a, &v)) {
					struct ivals I;
					ivals_build(&I, &v, v.e);
					ok = ival_of(&I, v.e, &lo, &hi);
					ivals_free(&I);
				} else
					free((char *)v.why);
				if (!ok) {
					d->known[k] = 2;	/* unknown at some site */
					continue;
				}
				if (d->known[k] == 0) {
					d->known[k] = 1;
					d->lo[k] = lo;
					d->hi[k] = hi;
				} else if (d->known[k] == 1) {
					if (lo < d->lo[k])
						d->lo[k] = lo;
					if (hi > d->hi[k])
						d->hi[k] = hi;
				}
			}
			d->n++;
		}
		return;
	}
	if (kind == CA_BLOCK || kind == CA_INIT) {
		for (k = 0; k < u->A.b[n]; k++)
			cs_walk(u, caller, u->A.lst[u->A.a[n] + k], escape);
		return;
	}
	if (kind == CA_NUM || kind == CA_FNUM || kind == CA_CHARLIT ||
	    kind == CA_STR || kind == CA_BREAK || kind == CA_CONTINUE ||
	    kind == CA_GOTO || kind == CA_EMPTY || kind == CA_TYPEDEF)
		return;
	if (kind == CA_DECL) {
		cs_walk(u, caller, u->A.a[n], escape);
		return;
	}
	if (kind == CA_MEMBER || kind == CA_ARROW || kind == CA_CASE ||
	    kind == CA_DEFAULT || kind == CA_LABEL || kind == CA_SIZEOF_E ||
	    kind == CA_CAST || kind == CA_COMPOUND) {
		cs_walk(u, caller, u->A.a[n], escape);
		return;
	}
	cs_walk(u, caller, u->A.a[n], escape);
	cs_walk(u, caller, u->A.b[n], escape);
	cs_walk(u, caller, u->A.c[n], escape);
	if (kind == CA_FOR)
		cs_walk(u, caller, u->A.d[n], escape);
}

static void
callsite_domains(struct cf_unit *u)
{
	uint8_t *escape = xcalloc(u->S.n + 1, 1);
	int k;

	free(csd);
	csd = xcalloc((size_t)u->nfn + 1, sizeof *csd);
	for (k = 0; k < u->nfn; k++)
		cs_walk(u, u->fns[k], u->A.a[u->fns[k]], escape);
	/* global initializers can take a function's address too */
	for (k = 0; k < (int)u->A.d[0]; k++) {
		uint32_t n = u->A.lst[u->top + k];
		if (u->A.kind[n] != CA_FUNC)
			cs_walk(u, 0, n, escape);
	}
	for (k = 0; k < u->nfn; k++) {
		uint32_t s = (uint32_t)u->A.iv[u->fns[k]];
		if (u->S.stclass[s] != 's' || escape[s] || csd[k].n == 0)
			csd[k].n = 0;	/* entered from outside: no domain */
	}
	free(escape);
}

static int cur_fn_index;

static void
apply_callsites(struct cf_value *v, int whole)
{
	struct cf_unit *u = R.u;
	uint32_t fn = u->fns[cur_fn_index], k;
	const struct csdom *d = &csd[cur_fn_index];
	int j;

	if (d->n == 0)
		return;
	for (k = 0; k < u->A.d[fn] && k < MAXP; k++) {
		uint32_t ps = u->A.lst[u->A.c[fn] + k];
		const char *nm = sym_str(u->S.name[ps]);
		if (d->known[k] != 1 || (!whole && u->S.assigned[ps]))
			continue;
		for (j = 0; j < v->nleaf; j++)
			if (!strcmp(v->leaf[j].text, nm) && !v->leaf[j].ranged) {
				ty_t t = ct_to_ucas(v->leaf[j].cty);
				int64_t lo = int_norm(t, d->lo[k]), hi = int_norm(t,
				    d->hi[k]);
				if (lo != d->lo[k] || hi != d->hi[k] || lo > hi)
					continue;
				v->leaf[j].ranged = 1;
				v->leaf[j].lo = lo;
				v->leaf[j].hi = hi;
				v->leaf[j].from_calls = 1;
			}
	}
}

static void
record_steps(const struct step *s, int ns, nid from, nid to, int line,
    const char *before_c, const char *after_c, int applied)
{
	int k, best = CF_PROOF_SAT;
	struct cf_rewrite *w;
	struct sbuf what = {0};

	for (k = 0; k < ns; k++) {
		int rep = 1;
		while (k + rep < ns && !strcmp(s[k + rep].what, s[k].what))
			rep++;
		sb_printf(&what, "%s%s", k ? "; " : "", s[k].what);
		if (rep > 1)
			sb_printf(&what, " (x%d)", rep);
		k += rep - 1;
		if (s[k].pf.kind == CF_PROOF_EXHAUSTIVE && best != CF_PROOF_SAT)
			best = CF_PROOF_EXHAUSTIVE;
	}
	w = add_rw(ns ? s[0].what : "rewrite", line);
	(void)from;
	(void)to;
	w->before = strdup(before_c);
	w->after = strdup(after_c);
	/* the last step is proved against the original, so it is the proof */
	w->proof = ns ? s[ns - 1].pf.kind : CF_PROOF_NONE;
	snprintf(w->proof_text, sizeof w->proof_text, "%s", ns ?
	    s[ns - 1].pf.text : "");
	w->cost_before = ns ? s[0].before : 0;
	w->cost_after = ns ? s[ns - 1].after : 0;
	w->applied = applied;
	if (ns > 1) {
		free(w->before);
		w->before = strdup(before_c);
		snprintf(w->kind, sizeof w->kind, "%d steps", ns);
	}
	w->steps = what.p ? strdup(what.p) : NULL;
	free(what.p);
	(void)best;
}

/* the inputs' ranges that did not come from their types */
static void
note_domain(const struct cf_value *v)
{
	struct cf_rewrite *w;
	char buf[160];
	size_t l;
	int j, any = 0;

	if (R.r->nrw == 0)
		return;
	w = &R.r->rw[R.r->nrw - 1];
	buf[0] = 0;
	for (j = 0; j < v->nleaf; j++) {
		if (!v->leaf[j].ranged)
			continue;
		l = strlen(buf);
		snprintf(buf + l, sizeof buf - l, "%s%s in [%lld, %lld] (%s)",
		    any++ ? ", " : "", v->leaf[j].text, (long long)v->leaf[j].lo,
		    (long long)v->leaf[j].hi, v->leaf[j].from_calls ?
		    "from the call sites" : "assumed");
	}
	if (any) {
		free(w->domain);
		w->domain = strdup(buf);
	}
}

static uint32_t
fn_line(uint32_t n)
{
	return R.u->tk.line[R.u->A.t0[n]];
}

/* the whole function: one value */
static int
whole_function(uint32_t fn, struct cf_emit *em)
{
	struct cf_unit *u = R.u;
	struct cf_value v;
	struct step steps[16];
	int ns;
	nid best;
	uint32_t body = u->A.a[fn], from, to, ffrom, fto, s = (uint32_t)u->A.iv[fn];
	struct sbuf out = {0}, before = {0};

	if (!cf_lower_func(u, fn, &v, !R.o->wrapv)) {
		struct cf_rewrite *w = add_rw("not lowered", fn_line(fn));
		w->before = strdup("");
		w->after = strdup("");
		snprintf(w->proof_text, sizeof w->proof_text, "%s", v.why ?
		    v.why : "?");
		w->applied = 0;
		w->info = 1;
		free((char *)v.why);
		return 0;
	}
	apply_assumptions(&v);
	apply_callsites(&v, 1);
	best = optimize_value(&v, steps, &ns, 16, 1);
	if (!cf_span(u, body, &from, &to) || !cf_span(u, fn, &ffrom, &fto))
		return 1;
	sb_printf(&before, "%.*s", (int)(to - from), u->ftext[0] + from);
	if (best == v.e || ns == 0) {
		struct cf_rewrite *w = add_rw("no cheaper form", fn_line(fn));
		w->before = strdup("");
		w->after = strdup("");
		snprintf(w->proof_text, sizeof w->proof_text, "lowered: %d "
		    "inputs, cost %.1f", v.nleaf, cf_cost(R.o->t, v.e));
		w->info = 1;
		free(before.p);
		return 1;
	}
	if (!cf_emit_body(em, &v, best, CT.base[u->S.ty[s]], &out)) {
		struct cf_rewrite *w = add_rw("not emitted", fn_line(fn));
		w->before = strdup("");
		w->after = strdup("");
		snprintf(w->proof_text, sizeof w->proof_text, "the C emitter "
		    "cannot print the form without UB");
		w->info = 1;
		free(before.p);
		return 1;
	}
	record_steps(steps, ns, v.e, best, (int)fn_line(fn), before.p, out.p, 1);
	note_domain(&v);
	add_repl(from, to, out.p, ffrom);
	R.r->cost_before += steps[0].before;
	R.r->cost_after += steps[ns - 1].after;
	free(out.p);
	free(before.p);
	return 1;
}

/* ---- expressions inside functions that do not lower whole ---- */

static void visit_stmt(uint32_t fn, uint32_t n, struct cf_emit *em);

static int
is_expr_kind(int k)
{
	return k <= CA_INIT;
}

static void
visit_expr(uint32_t fn, uint32_t n, struct cf_emit *em)
{
	struct cf_unit *u = R.u;
	struct cf_value v;
	uint32_t k;
	int kind;

	if (n == 0 || out_of_time())
		return;
	kind = u->A.kind[n];
	if (!is_expr_kind(kind))
		return;
	if (ct_is_int(u->A.ty[n]) && kind != CA_ID && kind != CA_NUM &&
	    kind != CA_CHARLIT && kind != CA_ASSIGN && kind != CA_CALL &&
	    cf_lower_expr(u, fn, n, &v)) {
		struct step steps[16];
		int ns;
		nid best;
		uint32_t from, to;
		if (cf_cost(R.o->t, v.e) <= 1.5 || !cf_span(u, n, &from, &to))
			return;
		apply_assumptions(&v);
		apply_callsites(&v, 0);
		best = optimize_value(&v, steps, &ns, 16, 0);
		if (best != v.e && ns > 0) {
			struct sbuf out = {0}, before = {0};
			if (cf_emit_expr(em, &v, best, u->A.ty[n], &out)) {
				sb_printf(&before, "%.*s", (int)(to - from),
				    u->ftext[0] + from);
				record_steps(steps, ns, v.e, best,
				    (int)u->tk.line[u->A.t0[n]], before.p, out.p, 1);
				note_domain(&v);
				{
					struct sbuf par = {0};
					sb_printf(&par, "(%s)", out.p);
					add_repl(from, to, par.p, R.fnstart);
					free(par.p);
				}
				R.r->cost_before += steps[0].before;
				R.r->cost_after += steps[ns - 1].after;
			}
			free(out.p);
			free(before.p);
		}
		return;
	}
	if (v.why)
		free((char *)v.why);
	switch (kind) {
	case CA_CALL:
		for (k = 0; k < u->A.c[n]; k++)
			visit_expr(fn, u->A.lst[u->A.b[n] + k], em);
		return;
	case CA_INIT:
		for (k = 0; k < u->A.b[n]; k++)
			visit_expr(fn, u->A.lst[u->A.a[n] + k], em);
		return;
	case CA_COND:
		visit_expr(fn, u->A.c[n], em);
		/* fall through */
	case CA_BIN: case CA_LAND: case CA_LOR: case CA_COMMA: case CA_ASSIGN:
	case CA_INDEX:
		visit_expr(fn, u->A.a[n], em);
		visit_expr(fn, u->A.b[n], em);
		return;
	case CA_MEMBER: case CA_ARROW: case CA_POSTINC: case CA_POSTDEC:
	case CA_PREINC: case CA_PREDEC: case CA_ADDR: case CA_DEREF: case CA_PLUS:
	case CA_NEG: case CA_BNOT: case CA_LNOT: case CA_CAST:
		visit_expr(fn, u->A.a[n], em);
		return;
	}
}

static void
visit_stmt(uint32_t fn, uint32_t n, struct cf_emit *em)
{
	struct cf_unit *u = R.u;
	uint32_t k;

	if (n == 0)
		return;
	switch (u->A.kind[n]) {
	case CA_BLOCK:
		for (k = 0; k < u->A.b[n]; k++)
			visit_stmt(fn, u->A.lst[u->A.a[n] + k], em);
		return;
	case CA_EXPR:
	case CA_RETURN:
		visit_expr(fn, u->A.a[n], em);
		return;
	case CA_DECL:
		if (u->S.kind[u->A.iv[n]] == CS_LOCAL)
			visit_expr(fn, u->A.a[n], em);
		return;
	case CA_IF:
		visit_expr(fn, u->A.a[n], em);
		visit_stmt(fn, u->A.b[n], em);
		visit_stmt(fn, u->A.c[n], em);
		return;
	case CA_WHILE:
		visit_expr(fn, u->A.a[n], em);
		visit_stmt(fn, u->A.b[n], em);
		return;
	case CA_DO:
		visit_stmt(fn, u->A.a[n], em);
		visit_expr(fn, u->A.b[n], em);
		return;
	case CA_FOR:
		visit_stmt(fn, u->A.a[n], em);
		visit_expr(fn, u->A.b[n], em);
		visit_expr(fn, u->A.c[n], em);
		visit_stmt(fn, u->A.d[n], em);
		return;
	case CA_SWITCH:
		visit_expr(fn, u->A.a[n], em);
		visit_stmt(fn, u->A.b[n], em);
		return;
	case CA_CASE:
	case CA_DEFAULT:
	case CA_LABEL:
		visit_stmt(fn, u->A.a[n], em);
		return;
	}
}

static int
cmp_repl(const void *x, const void *y)
{
	const struct repl *a = x, *b = y;

	return a->from < b->from ? -1 : a->from > b->from;
}

int
cf_optimize(struct cf_unit *u, const struct cf_opts *o, struct cf_result *r)
{
	struct cf_emit em;
	int k;
	uint32_t first = UINT32_MAX;
	const char *src = u->ftext[0];
	size_t pos = 0, len = strlen(src);

#ifdef CF_HAVE_X86
	{
		static int inited;
		if (!inited++)
			cfx86_init();
	}
#endif
	memset(r, 0, sizeof *r);
	memset(&R, 0, sizeof R);
	memset(&em, 0, sizeof em);
	em.t = o->t;
	R.o = o;
	R.r = r;
	R.u = u;
	R.t0 = clock();
	R.deadline = o->budget > 0 ? o->budget : 10;
	nreps = 0;
	callsite_domains(u);
	for (k = 0; k < u->nfn; k++) {
		uint32_t fn = u->fns[k], s = (uint32_t)u->A.iv[fn], ff, ft;
		int before = nreps;
		snprintf(R.fname, sizeof R.fname, "%s", sym_str(u->S.name[s]));
		if (u->tk.file[u->A.t0[fn]] != 0)
			continue;	/* from an included file */
		if (!cf_span(u, fn, &ff, &ft))
			continue;
		R.fnstart = ff;
		cur_fn_index = k;
		if (!whole_function(fn, &em) || nreps == before)
			if (nreps == before)
				visit_stmt(fn, u->A.a[fn], &em);
		if (nreps > before && ff < first)
			first = ff;
	}
	/* the output: the main file with the replacements, helpers before
	 * the first rewritten function */
	qsort(reps, (size_t)nreps, sizeof *reps, cmp_repl);
	sb_printf(&r->out, "/* optimised by ucas cfront for %s (%s); every "
	    "rewrite proved, see the report */\n", o->t->name, o->t->march);
	if (em.helpers.n || nreps) {
		int has_stdint = strstr(u->includes.p ? u->includes.p : "",
		    "<stdint.h>") != NULL || strstr(u->includes.p ?
		    u->includes.p : "", "<inttypes.h>") != NULL;
		if (!has_stdint)
			sb_puts(&r->out, "#include <stdint.h>\n");
	}
	for (k = 0; k < nreps; k++) {
		if (reps[k].from < pos)
			continue;	/* overlapping: the outer one won */
		if (reps[k].from >= first && pos <= first && em.helpers.n) {
			sb_printf(&r->out, "%.*s", (int)(first - pos), src + pos);
			sb_puts(&r->out, "#if defined(__POPCNT__) || "
			    "defined(__LZCNT__) || defined(__BMI__)\n"
			    "#include <immintrin.h>\n#endif\n");
			sb_puts(&r->out, em.helpers.p);
			pos = first;
		}
		sb_printf(&r->out, "%.*s", (int)(reps[k].from - pos), src + pos);
		sb_puts(&r->out, reps[k].text);
		pos = reps[k].to;
	}
	sb_printf(&r->out, "%.*s", (int)(len - pos), src + pos);
	for (k = 0; k < nreps; k++)
		free(reps[k].text);
	free(em.helpers.p);
	r->nproofs = R.nproofs;
	r->calls_only = xcalloc((size_t)u->nfn + 1, 1);
	for (k = 0; k < u->nfn; k++)
		r->calls_only[k] = csd[k].n > 0;
	r->seconds = elapsed();
	return 1;
}

void
cf_result_free(struct cf_result *r)
{
	int k;

	for (k = 0; k < r->nrw; k++) {
		free(r->rw[k].before);
		free(r->rw[k].after);
		free(r->rw[k].steps);
		free(r->rw[k].domain);
	}
	free(r->rw);
	free(r->out.p);
	free(r->calls_only);
	memset(r, 0, sizeof *r);
}
