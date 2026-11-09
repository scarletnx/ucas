/*
 * Cost models: a cost per operation class, declared with `model`.  The
 * prelude defines gt1030 (NVIDIA GP108 through SPIR-V), which is active by
 * default.  `cost f` prices a function as a DAG (each shared subterm once)
 * and prices every ?: three ways: as a select, as a uniform branch, and as
 * a divergent branch, where the warp runs both arms ("keep-up").
 */
#include "calc.h"

#include <stdlib.h>
#include <string.h>

const char *const cclass_name[C_NCLASS] = {
	"iadd", "logic", "shift", "imul", "idiv", "idivk", "cmp", "select",
	"minmax", "bitcnt", "conv", "bitcast",
	"fadd", "fmul", "fdiv", "fsqrt", "ftrans", "fpow", "f64",
	"load", "branch", "diverge", "barrier", "atomic_shared",
	"atomic_global", "store",
};

#define MAXMODEL 16
static struct model models[MAXMODEL];
static int nmodels;
static struct model *active;

struct model *
model_find(uint32_t name)
{
	int i;

	for (i = 0; i < nmodels; i++)
		if (models[i].name == name)
			return &models[i];
	return NULL;
}

struct model *
model_new(uint32_t name)
{
	struct model *m = model_find(name);
	int k;

	if (m == NULL) {
		if (nmodels == MAXMODEL)
			fail("too many models");
		m = &models[nmodels++];
	}
	memset(m, 0, sizeof *m);
	m->name = name;
	for (k = 0; k < C_NCLASS; k++)
		m->c[k] = 1;
	m->c[C_BITCAST] = 0;
	m->c[C_F64] = 1;
	return m;
}

/* the i-th model, NULL past the last (jsonmode.c's state) */
struct model *
model_nth(int i)
{
	return i >= 0 && i < nmodels ? &models[i] : NULL;
}

struct model *
model_active(void)
{
	return active;
}

void
model_set_active(struct model *m)
{
	active = m;
}

int
cclass_find(const char *s, size_t n)
{
	int k;

	for (k = 0; k < C_NCLASS; k++)
		if (strlen(cclass_name[k]) == n && !memcmp(cclass_name[k], s, n))
			return k;
	return -2;
}

static int
fpclass(int k)
{
	return k >= C_FADD && k <= C_FPOW;
}

float
node_cost(const struct model *m, nid n, int *cls)
{
	int op = N.op[n], k = -1;
	ty_t t = N.ty[n], ta = N.a[n] ? N.ty[N.a[n]] : 0;
	int fp = TY_FP(t) || (ta && TY_FP(ta) && (opinfo[op].fl & OF_CMP));
	float mul = 1, c;

	switch (op) {
	case O_CONST: case O_VAR: case O_UNDEF: case O_TABLE:
		*cls = -1;
		return 0;
	case O_ADD: case O_SUB: case O_NEG:
		k = fp ? C_FADD : C_IADD;
		break;
	case O_MUL:
		if (!fp && is_const(N.a[n]) && N.ki[N.a[n]] > 0 &&
		    !(N.ki[N.a[n]] & (N.ki[N.a[n]] - 1)))
			k = C_SHIFT;
		else
			k = fp ? C_FMUL : C_IMUL;
		break;
	case O_DIV: case O_MOD:
		k = fp ? C_FDIV : is_const(N.b[n]) ? C_IDIVK : C_IDIV;
		break;
	case O_POW:
		if (fp) {
			k = C_FPOW;
			if (is_const(N.b[n]) && N.kr[N.b[n]] == 2)
				k = C_FMUL;
			if (is_const(N.b[n]) && N.kr[N.b[n]] == 0.5)
				k = C_FSQRT;
		} else {
			k = C_IMUL;
			if (is_const(N.b[n]) && N.ki[N.b[n]] > 1)
				mul = (float)(N.ki[N.b[n]] - 1);
			else if (!is_const(N.b[n])) {
				/* square and multiply over the exponent's bits */
				int w = TY_SIZED(t) ? TYW(t) : 64, bits = 0;
				while ((1 << bits) < w)
					bits++;
				*cls = C_IMUL;
				return (float)bits * (2 * m->c[C_IMUL] +
				    m->c[C_BRANCH]);
			}
		}
		break;
	case O_AND: case O_OR: case O_XOR: case O_NOT: case O_LNOT:
	case O_LAND: case O_LOR:
		k = C_LOGIC;
		break;
	case O_SHL: case O_SHR:
		k = C_SHIFT;
		break;
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE:
		k = C_CMP;
		break;
	case O_SEL:
		k = C_SELECT;
		break;
	case O_MIN: case O_MAX: case O_ABS: case O_SIGN:
		k = C_MINMAX;
		break;
	case O_SQRT:
		k = C_FSQRT;
		break;
	case O_ISQRT:
		k = C_FSQRT;
		*cls = k;
		return m->c[C_FSQRT] + 2 * m->c[C_CONV];
	case O_EXP: case O_LOG: case O_SIN: case O_COS:
		k = C_FTRANS;
		break;
	case O_TAN:
		*cls = C_FTRANS;
		return 2 * m->c[C_FTRANS] + m->c[C_FDIV];
	case O_SEC: case O_CSC:		/* SFU op and a divide */
		*cls = C_FTRANS;
		return m->c[C_FTRANS] + m->c[C_FDIV];
	case O_COT:
		*cls = C_FTRANS;
		return 2 * m->c[C_FTRANS] + m->c[C_FDIV];
	case O_LAMBERTW:		/* Halley iterations of exp: a library call */
	case O_LAMBERTWM1:
		k = C_FTRANS;
		mul = 8;
		break;
	case O_ATAN:			/* no SFU instruction: a library call */
	case O_ASIN: case O_ACOS: case O_ATAN2: case O_SINH: case O_COSH:
	case O_TANH: case O_ASINH: case O_ACOSH: case O_ATANH:
		k = C_FTRANS;
		mul = 4;
		break;
	case O_COPYSIGN:		/* one bitwise select */
		k = C_LOGIC;
		break;
	case O_FMA:			/* full rate, like a multiply */
		k = C_FMUL;
		break;
	case O_FLOOR: case O_CEIL:
		k = C_CONV;
		break;
	case O_CLZ: case O_CTZ: case O_POPCNT:
		k = C_BITCNT;
		break;
	case O_CAST:
		if (TY_FP(t) || TY_FP(ta))
			k = C_CONV;
		else if (TY_SIZED(t) && TYW(t) < 32 &&
		    (!TY_SIZED(ta) || TYW(ta) > TYW(t)))
			k = C_LOGIC;    /* narrowing: a mask or sign extend */
		else {
			*cls = -1;
			return 0;
		}
		break;
	case O_ASUINT: case O_ASINT: case O_ASFLOAT:
		k = C_BITCAST;
		break;
	case O_INDEX:
		k = C_LOAD;
		break;
	case O_BARRIER:
		k = C_BARRIER;
		break;
	case O_ATOMIC:
		k = (N.ki[n] & AT_SHARED) ? C_ATOM_SH : C_ATOM_GL;
		break;
	case O_LOOP:
		return loop_cost(m, n, cls);
	default:
		k = C_IADD;
	}
	*cls = k;
	c = m->c[k] * mul;
	if (fpclass(k) && TYK(t) == K_FLOAT && TYW(t) == 64)
		c *= m->c[C_F64];
	return c;
}

void
cmd_models(void)
{
	int i, k;

	for (i = 0; i < nmodels; i++) {
		printf("model %s%s {", sym_str(models[i].name),
		    &models[i] == active ? " (active)" : "");
		for (k = 0; k < C_NCLASS; k++)
			printf("%s %s: %g", k ? "," : "", cclass_name[k],
			    (double)models[i].c[k]);
		printf(" }\n");
	}
}

/* ---- pricing a function ---- */

/* every node reachable from root, not into loop entries (a loop is a node) */
static void
reach(nid n, uint8_t *mark, nid skip_from, nid skip_to)
{
	uint32_t j;

	if (n == 0 || mark[n])
		return;
	mark[n] = 1;
	if (N.op[n] == O_TABLE) {
		for (j = 0; j < N.b[n]; j++)
			reach(N.lst[N.a[n] + j], mark, skip_from, skip_to);
		return;
	}
	if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF)
		return;
	if (!(n == skip_from && N.a[n] == skip_to))
		reach(N.a[n], mark, skip_from, skip_to);
	if (!(n == skip_from && N.b[n] == skip_to))
		reach(N.b[n], mark, skip_from, skip_to);
	if (!(n == skip_from && N.c[n] == skip_to))
		reach(N.c[n], mark, skip_from, skip_to);
}

static int
depends_nonuniform(const struct func *f, nid n, uint8_t *seen)
{
	uint32_t j;

	if (n == 0 || seen[n])
		return 0;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_VAR:
		/* a loop's own variables follow its (possibly varying) inputs */
		return N.ki[n] >= f->np || !f->p[N.ki[n]].uniform;
	case O_CONST:
	case O_UNDEF:
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			if (depends_nonuniform(f, N.lst[N.a[n] + j], seen))
				return 1;
		return 0;
	case O_ATOMIC:
		return 1;   /* the old value differs per invocation */
	}
	return depends_nonuniform(f, N.a[n], seen) ||
	    depends_nonuniform(f, N.b[n], seen) ||
	    depends_nonuniform(f, N.c[n], seen);
}

static float
sum_cost(const struct model *m, const uint8_t *mark, uint32_t lim)
{
	float s = 0;
	uint32_t g;
	int cls;

	for (g = 1; g < lim; g++)
		if (mark[g])
			s += node_cost(m, g, &cls);
	return s;
}

/*
 * A loop priced as a program would run it: the initial values and the
 * loop-invariant subterms of the body once (hoisted), the rest of the
 * body plus the loop overhead per iteration (for: the index increment,
 * the compare and the branch; while: the branch, the compare being in the
 * condition), times the trip count: exact for constant bounds, otherwise
 * the worst case over the parameters' ranges (a sound interval bound,
 * or the domain's maximum when enumerated) and the average over the
 * domain's points.
 */
struct lprice {
	nid n;
	uint32_t e;
	int known, exh, sound, measured;
	double worst, avg, best;
	uint64_t pts;
	float once, body, over;
	uint8_t *var;        /* per-iteration nodes */
};

static void
trip_range(const struct func *f, nid lo, nid hi, struct lprice *lp)
{
	struct diter d;
	struct plan p;
	nid roots[3] = {lo, hi, f->where};
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint32_t np, i;
	double sum = 0, mx = 0, mn = -1;
	struct ival *iv;
	struct func g = *f;
	int64_t blo[MAXP], bhi[MAXP];
	int j;

	plan_build(&p, roots, 3, 0);
	diter_init(&d, f, 1u << 16, 2, 4096);
	lp->exh = d.exhaustive;
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		uint8_t ok[CH];
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[2] : NOTOK, np, ok);
		for (i = 0; i < np; i++) {
			double t;
			if (!ok[i] || ((COLF(&p, p.root[0])[i] |
			    COLF(&p, p.root[1])[i]) & F_UNDEF))
				continue;
			t = (double)COLI(&p, p.root[1])[i] -
			    (double)COLI(&p, p.root[0])[i] + 1;
			if (t < 0)
				t = 0;
			sum += t;
			if (t > mx)
				mx = t;
			if (mn < 0 || t < mn)
				mn = t;
			lp->pts++;
		}
	}
	lp->avg = lp->pts ? sum / (double)lp->pts : 0;
	lp->best = mn < 0 ? 0 : mn;
	lp->worst = mx;
	/* a sound bound from intervals, the where clause narrowing first */
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &blo[j], &bhi[j]);
	where_bounds(f, blo, bhi, NULL);
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty) && !(TYK(f->p[j].ty) == K_UINT &&
		    TYW(f->p[j].ty) == 64)) {
			g.p[j].ranged = 1;
			g.p[j].lo = blo[j];
			g.p[j].hi = bhi[j];
		}
	iv = xcalloc(p.n ? p.n : 1, sizeof *iv);
	plan_ival(&p, &g, iv);
	if (!lp->exh && !iv[p.root[0]].top && !iv[p.root[1]].top) {
		double w = (double)iv[p.root[1]].hi - (double)iv[p.root[0]].lo + 1;
		lp->worst = w < 0 ? 0 : w;
		lp->sound = 1;
	}
	if (lp->exh)
		lp->sound = 1;
	free(iv);
	plan_free(&p);
	free(in_i);
	free(in_r);
}

/*
 * A while loop's trip count, measured: the same loop with one more state
 * counting the iterations, evaluated over the domain (exact when it is
 * enumerated, a sample otherwise).
 */
static void
trip_while(const struct func *f, nid n, struct lprice *lp)
{
	struct loopdef d;
	uint32_t e = loop_ent(n);
	int slot = KSLOT - 1;
	struct diter it;
	struct plan p;
	nid cntr, roots[2];
	int64_t *in_i, mx = 0, mn = -1;
	double *in_r, sum = 0;
	uint32_t np, i;

	lp->worst = lp->avg = lp->best = -1;
	loop_get(e, &d);
	if (d.nst >= MAXST || ((loops.bound[e] | loops.free[e]) >> slot) & 1)
		return;
	d.sslot[d.nst] = slot;
	d.sname[d.nst] = sym_cstr("_trips");
	d.sty[d.nst] = TY_INT;
	d.init[d.nst] = mk_const_i(TY_INT, 0);
	d.next[d.nst] = mk_bin(O_ADD, var_of(slot, d.sname[d.nst], TY_INT),
	    mk_const_i(TY_INT, 1));
	d.nst++;
	cntr = mk_loop(loop_intern(&d), d.nst - 1, N.a[n], N.b[n], N.c[n]);
	roots[0] = cntr;
	roots[1] = f->where;
	in_i = xcalloc((size_t)NSLOT * CH, sizeof *in_i);
	in_r = xcalloc((size_t)NSLOT * CH, sizeof *in_r);
	plan_build(&p, roots, 2, 0);
	diter_init(&it, f, 1u << 16, 2, 4096);
	lp->exh = it.exhaustive;
	while ((np = diter_next(&it, in_i, in_r)) != 0) {
		uint8_t ok[CH];
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, f->where ? p.root[1] : NOTOK, np, ok);
		for (i = 0; i < np; i++) {
			int64_t t;
			if (!ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF))
				continue;
			t = COLI(&p, p.root[0])[i];
			sum += (double)t;
			if (t > mx)
				mx = t;
			if (mn < 0 || t < mn)
				mn = t;
			lp->pts++;
		}
	}
	if (lp->pts) {
		lp->worst = (double)mx;
		lp->best = (double)mn;
		lp->avg = sum / (double)lp->pts;
		lp->sound = lp->exh;
		lp->measured = 1;
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
}

static void
price_loop(const struct model *m, const struct func *f, nid n,
    const uint8_t *outer, uint32_t lim, struct lprice *lp, float *bycls,
    int *cnt)
{
	uint32_t e = loop_ent(n), s = loops.st[e], k, g;
	uint8_t *ini = xcalloc(lim, 1), *bod = xcalloc(lim, 1);
	uint64_t bound = loops.bound[e];
	int64_t lo, hi;
	int cls, isfor = loops.kind[e] == LK_FOR;

	memset(lp, 0, sizeof *lp);
	lp->n = n;
	lp->e = e;
	for (k = 0; k < loops.nst[e]; k++) {
		reach(loops.init[s + k], ini, 0, 0);
		reach(loops.next[s + k], bod, 0, 0);
	}
	reach(loops.cond[e], bod, 0, 0);
	lp->var = xcalloc(lim, 1);
	for (g = 1; g < lim; g++) {
		float c;
		if (outer[g] || !(ini[g] || bod[g]))
			continue;
		c = node_cost(m, g, &cls);
		if (cls < 0)
			continue;
		if (bod[g] && (node_slots(g) & bound)) {
			lp->var[g] = 1;
			lp->body += c;
		} else
			lp->once += c;
	}
	if (isfor)
		lp->over = m->c[C_IADD] + m->c[C_CMP] + m->c[C_BRANCH];
	else
		lp->over = m->c[C_BRANCH];
	if (!isfor)
		trip_while(f, n, lp);
	else if (int_of(N.a[n], &lo) && int_of(N.b[n], &hi)) {
		lp->known = lp->sound = lp->exh = 1;
		lp->worst = lp->avg = lp->best = hi >= lo ? (double)(hi - lo + 1) :
		    0;
	} else
		trip_range(f, N.a[n], N.b[n], lp);
	/* by class: once, and per iteration times the worst trip count */
	{
		double t = lp->worst < 0 ? 1 : lp->worst;
		for (g = 1; g < lim; g++) {
			float c;
			if (outer[g] || !(ini[g] || bod[g]))
				continue;
			c = node_cost(m, g, &cls);
			if (cls < 0)
				continue;
			bycls[cls] += lp->var[g] ? (float)(c * t) : c;
			cnt[cls] += lp->var[g] ? (int)t : 1;
		}
		if (isfor) {
			bycls[C_IADD] += (float)(m->c[C_IADD] * t);
			cnt[C_IADD] += (int)t;
			bycls[C_CMP] += (float)(m->c[C_CMP] * t);
			cnt[C_CMP] += (int)t;
		}
		bycls[C_BRANCH] += (float)(m->c[C_BRANCH] * t);
		cnt[C_BRANCH] += (int)t;
	}
	free(ini);
	free(bod);
}

static float
lp_total(const struct lprice *lp, double trips)
{
	if (trips < 0)
		trips = 1;
	return lp->once + (float)(trips * (lp->body + lp->over));
}

static void
print_loop(const struct func *f, const struct lprice *lp)
{
	uint32_t e = lp->e;
	char head[160];
	struct sbuf b = {0};
	float tw = lp_total(lp, lp->worst);

	if (loops.kind[e] == LK_FOR) {
		sb_printf(&b, "%s in ", sym_str(loops.name[e]));
		print_node(&b, N.a[lp->n], NULL);
		sb_puts(&b, "..");
		print_node(&b, N.b[lp->n], NULL);
	} else {
		sb_puts(&b, "while ");
		print_shared(&b, loops.cond[e], 60);
	}
	cap_text(&b, 0, 80);
	snprintf(head, sizeof head, "%s", sb_str(&b));
	free(b.p);
	printf("  loop %s:\n", head);
	printf("    per iteration: body %g + overhead %g (%s) = %g\n",
	    (double)lp->body, (double)lp->over, loops.kind[e] == LK_FOR ?
	    "index add, compare, branch" : "branch; the compare is in the "
	    "condition", (double)(lp->body + lp->over));
	if (lp->worst < 0)
		printf("    trip count: not known (a while loop that could not "
		    "be measured); priced as one iteration\n");
	else if (lp->measured)
		printf("    trip count: %s %.0f, average %.4g, measured by running "
		    "it on %s%llu points\n", lp->exh ? "at most" : "up to",
		    lp->worst, lp->avg, lp->exh ? "all " : "a sample of ",
		    (unsigned long long)lp->pts);
	else if (lp->known)
		printf("    trip count: %.0f (constant bounds)\n", lp->worst);
	else {
		printf("    trip count: at most %.0f%s, average %.4g over %s%llu "
		    "points\n", lp->worst, lp->sound ? "" : " (sampled, not a "
		    "bound)", lp->avg, lp->exh ? "all " : "a sample of ",
		    (unsigned long long)lp->pts);
	}
	printf("    once: %g (initial values and loop-invariant subterms, "
	    "hoisted)\n", (double)lp->once);
	if (lp->worst >= 0 && !lp->known && lp->avg != lp->worst)
		printf("    loop total: %g worst case, %g on average\n",
		    (double)tw, (double)lp_total(lp, lp->avg));
	else
		printf("    loop total: %g\n", (double)tw);
	(void)f;
	rep_obj(NULL);
	rep_str("loop", head);
	rep_num("body", lp->body);
	rep_num("overhead", lp->over);
	rep_num("once", lp->once);
	if (lp->worst >= 0) {
		rep_num("trips_worst", lp->worst);
		rep_num("trips_average", lp->avg);
		rep_bool("trips_known", lp->known);
		rep_bool("trips_bound_sound", lp->sound);
	}
	rep_num("total_worst", tw);
	rep_num("total_average", lp_total(lp, lp->avg));
	rep_end();
}

#define MAXSEL_SHOWN 12

void
cmd_cost(struct func *f, struct model *m)
{
	uint32_t lim = N.n, g;
	uint8_t *all = xcalloc(lim, 1), *cut = xcalloc(lim, 1);
	uint8_t *seen = xcalloc(lim, 1);
	float bycls[C_NCLASS] = {0}, total = 0, avg_total, best_total, c;
	int cnt[C_NCLASS] = {0}, cls, k, nops = 0, nsel = 0, nloop = 0;
	int varying = 0;
	struct lprice *lps = NULL;
	struct shmap *sh;
	struct sbuf cb = {0};

	reach(f->body, all, 0, 0);
	for (g = 1; g < lim; g++) {
		if (!all[g] || N.op[g] == O_LOOP)
			continue;
		c = node_cost(m, g, &cls);
		if (cls < 0)
			continue;
		bycls[cls] += c;
		cnt[cls]++;
		total += c;
		nops++;
	}
	avg_total = total;
	for (g = 1; g < lim; g++)
		if (all[g] && N.op[g] == O_LOOP) {
			lps = xrealloc(lps, (size_t)(nloop + 1) * sizeof *lps);
			price_loop(m, f, g, all, lim, &lps[nloop], bycls, cnt);
			total += lp_total(&lps[nloop], lps[nloop].worst);
			avg_total += lp_total(&lps[nloop], lps[nloop].avg);
			if (!lps[nloop].known && lps[nloop].worst >= 0 &&
			    lps[nloop].avg != lps[nloop].worst)
				varying = 1;
			nloop++;
		}
	printf("cost of %s under %s: %g (", f->anon ? shared_str(f->body, 200) :
	    sym_str(f->name), sym_str(m->name), (double)total);
	if (nloop)
		printf("%s%d operation%s outside loops and %d loop%s, ",
		    varying ? "worst case; " : "", nops, nops == 1 ? "" : "s",
		    nloop, nloop == 1 ? "" : "s");
	else
		printf("%d operations, ", nops);
	printf("shared subterms counted once)\n");
	if (varying)
		printf("  average over the domain: %g\n", (double)avg_total);
	sh = shmap_new(f->body);
	if (shmap_count(sh)) {
		struct sbuf bb = {0};
		shmap_bindings(&bb, sh, "\n    ", 300);
		printf("  shared subterms (named once, used below):\n    %s\n",
		    sb_str(&bb));
		free(bb.p);
	}
	rep_begin("cost");
	if (shmap_count(sh)) {
		struct sbuf bb = {0};
		shmap_bindings(&bb, sh, "\n", EXPR_CAP);
		rep_str("shared", sb_str(&bb));
		free(bb.p);
	}
	rep_str("model", sym_str(m->name));
	rep_num("total", total);
	if (varying)
		rep_num("average_total", avg_total);
	rep_int("operations", nops);
	rep_list("by_class");
	for (k = 0; k < C_NCLASS; k++)
		if (cnt[k]) {
			printf("  %-14s %3d x %-5g = %g\n", cclass_name[k], cnt[k],
			    (double)(bycls[k] / (float)cnt[k]), (double)bycls[k]);
			rep_obj(NULL);
			rep_str("class", cclass_name[k]);
			rep_int("count", cnt[k]);
			rep_num("each", bycls[k] / (float)cnt[k]);
			rep_num("cost", bycls[k]);
			rep_end();
		}
	rep_end();
	if (nloop) {
		rep_list("loops");
		for (k = 0; k < nloop; k++)
			print_loop(f, &lps[k]);
		rep_end();
	}
	rep_list("selects");
	best_total = total;
	/*
	 * Every ?: outside loops (its arms priced against the whole body)
	 * and inside a loop's body (against that body, times its trips).
	 */
	for (k = -1; k < nloop; k++) {
		const uint8_t *in = k < 0 ? all : lps[k].var;
		double times = k < 0 ? 1 : lps[k].worst < 0 ? 1 : lps[k].worst;
		for (g = 1; g < lim; g++) {
			nid s = g, arm[2];
			float ex[2], sel, uni, div;
			int i, nonuni;

			if (!in[g] || N.op[g] != O_SEL)
				continue;
			nsel++;
			arm[0] = N.b[s];
			arm[1] = N.c[s];
			for (i = 0; i < 2; i++) {
				uint8_t *only = xcalloc(lim, 1);
				uint32_t h, q;
				memset(cut, 0, lim);
				if (k < 0)
					reach(f->body, cut, s, arm[i]);
				else {
					uint32_t e = lps[k].e, st = loops.st[e];
					for (q = 0; q < loops.nst[e]; q++)
						reach(loops.next[st + q], cut, s, arm[i]);
					reach(loops.cond[e], cut, s, arm[i]);
				}
				reach(arm[i], only, 0, 0);
				for (h = 0; h < lim; h++)
					if (cut[h] || (k >= 0 && !lps[k].var[h]))
						only[h] = 0;
				ex[i] = sum_cost(m, only, lim);
				free(only);
			}
			memset(seen, 0, lim);
			nonuni = depends_nonuniform(f, N.a[s], seen);
			sel = m->c[C_SELECT];
			uni = m->c[C_BRANCH] + (ex[0] > ex[1] ? ex[0] : ex[1]) -
			    ex[0] - ex[1];
			div = m->c[C_BRANCH] + m->c[C_DIVERGE];
			{
				float pick = nonuni ? (sel < div ? sel : div) :
				    (sel < uni ? sel : uni);
				const char *name = nonuni ? (sel <= div ? "select" :
				    "divergent branch") : (sel <= uni ? "select" :
				    "uniform branch");
				best_total += (float)((pick - sel) * times);
				if (nsel > MAXSEL_SHOWN)
					continue;
				sb_clear(&cb);
				print_with(&cb, N.a[s], sh, 160);
				printf("  ?: on %s  (arms cost %g and %g)%s\n",
				    sb_str(&cb), (double)ex[0],
				    (double)ex[1], k < 0 ? "" :
				    "  [in a loop body: per iteration]");
				printf("      select: %+g   uniform branch: %+g   "
				    "divergent branch: %+g%s\n", (double)sel,
				    (double)(uni), (double)div, nonuni ? "" :
				    "  (condition is uniform)");
				rep_obj(NULL);
				sb_clear(&cb);
				print_with(&cb, N.a[s], sh, 300);
				rep_str("condition", sb_str(&cb));
				rep_num("select", sel);
				rep_num("uniform_branch", uni);
				rep_num("divergent_branch", div);
				rep_bool("uniform_condition", !nonuni);
				rep_str("best", name);
				if (k >= 0)
					rep_bool("in_loop", 1);
				rep_end();
				printf("      -> %s%s\n", name, nonuni ?
				    " (condition varies per invocation; mark "
				    "parameters 'uniform' if it does not)" : "");
			}
		}
	}
	rep_end();
	if (nsel > MAXSEL_SHOWN) {
		printf("  ... %d more ?: not listed (each counted in the "
		    "estimate)\n", nsel - MAXSEL_SHOWN);
		rep_int("selects_not_listed", nsel - MAXSEL_SHOWN);
	}
	rep_num("best_total", best_total);
	if (nsel)
		printf("  estimate with the best form per ?: : %g\n",
		    (double)best_total);
	for (k = 0; k < nloop; k++)
		free(lps[k].var);
	free(lps);
	shmap_free(sh);
	free(cb.p);
	free(all);
	free(cut);
	free(seen);
}
