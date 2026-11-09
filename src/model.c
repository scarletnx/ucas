/*
 * Cost models: a cost per operation class, declared with `model`.  The
 * prelude defines gt1030 (NVIDIA GP108 through SPIR-V), which is active by
 * default.  `cost f` prices a function as a DAG (each shared subterm once)
 * and prices every ?: three ways: as a select, as a uniform branch, and as
 * a divergent branch, where the warp runs both arms ("keep-up").
 */
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

const char *const cclass_name[C_NCLASS] = {
	"iadd", "logic", "shift", "imul", "idiv", "idivk", "cmp", "select",
	"minmax", "bitcnt", "conv", "bitcast",
	"fadd", "fmul", "fdiv", "fsqrt", "ftrans", "fpow", "f64",
	"load", "branch", "diverge", "barrier", "atomic_shared",
	"atomic_global",
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
	case O_ATAN:			/* no SFU instruction: a library call */
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
		return !f->p[N.ki[n]].uniform;
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

void
cmd_cost(struct func *f, struct model *m)
{
	uint32_t lim = N.n, g;
	uint8_t *all = xcalloc(lim, 1), *cut = xcalloc(lim, 1);
	uint8_t *seen = xcalloc(lim, 1);
	float bycls[C_NCLASS] = {0}, total = 0, best_total, c;
	int cnt[C_NCLASS] = {0}, cls, k, nops = 0, nsel = 0;
	nid roots[2] = {f->body, 0};

	reach(f->body, all, 0, 0);
	for (g = 1; g < lim; g++) {
		if (!all[g])
			continue;
		c = node_cost(m, g, &cls);
		if (cls < 0)
			continue;
		bycls[cls] += c;
		cnt[cls]++;
		total += c;
		nops++;
	}
	printf("cost of %s under %s: %g (%d operations, shared subterms "
	    "counted once)\n", f->anon ? node_str(f->body) : sym_str(f->name),
	    sym_str(m->name), (double)total, nops);
	for (k = 0; k < C_NCLASS; k++)
		if (cnt[k])
			printf("  %-14s %3d x %-5g = %g\n", cclass_name[k], cnt[k],
			    (double)(bycls[k] / (float)cnt[k]), (double)bycls[k]);
	best_total = total;
	for (g = 1; g < lim; g++) {
		nid s = g, arm[2];
		float ex[2], sel, uni, div;
		int i, nonuni;

		if (!all[g] || N.op[g] != O_SEL)
			continue;
		nsel++;
		arm[0] = N.b[s];
		arm[1] = N.c[s];
		for (i = 0; i < 2; i++) {
			uint8_t *only = xcalloc(lim, 1);
			uint32_t h;
			memset(cut, 0, lim);
			reach(f->body, cut, s, arm[i]);
			reach(arm[i], only, 0, 0);
			for (h = 0; h < lim; h++)
				if (cut[h])
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
		printf("  ?: on %s  (arms cost %g and %g)\n", node_str(N.a[s]),
		    (double)ex[0], (double)ex[1]);
		printf("      select: %+g   uniform branch: %+g   divergent "
		    "branch: %+g%s\n", (double)sel, (double)(uni),
		    (double)div, nonuni ? "" : "  (condition is uniform)");
		{
			float pick = nonuni ? (sel < div ? sel : div) :
			    (sel < uni ? sel : uni);
			const char *name = nonuni ? (sel <= div ? "select" :
			    "divergent branch") : (sel <= uni ? "select" :
			    "uniform branch");
			best_total += pick - sel;
			printf("      -> %s%s\n", name, nonuni ?
			    " (condition varies per invocation; mark "
			    "parameters 'uniform' if it does not)" : "");
		}
	}
	if (nsel)
		printf("  estimate with the best form per ?: : %g\n",
		    (double)best_total);
	(void)roots;
	free(all);
	free(cut);
	free(seen);
}
