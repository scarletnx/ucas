/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Cost models: a cost per operation class, declared with `model`.  The
 * prelude defines gt1030 (NVIDIA GP108 through SPIR-V), which is active by
 * default.  `cost f` prices a function as a DAG (each shared subterm once)
 * and prices every ?: three ways: as a select, as a uniform branch, and as
 * a divergent branch, where the warp runs both arms ("keep-up").
 */
#include "calc.h"
#include "fuse.h"
#include "dword.h"
#include "memk.h"
#include "texalu.h"

#include <math.h>
#include "stoch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *const cclass_name[C_NCLASS] = {
	"iadd", "logic", "shift", "imul", "idiv", "idivk", "cmp", "select",
	"minmax", "bitcnt", "conv", "bitcast",
	"fadd", "fmul", "fdiv", "fsqrt", "ftrans", "fpow", "f64",
	"load", "branch", "diverge", "barrier", "atomic_shared",
	"atomic_global", "store",
	"sdiv", "sdivk", "irem", "iremk", "srem", "sremk",
	/* in the order of the O_*DOT* ops (DOT_CLASS) */
	"udot4x8", "sdot4x8", "sudot4x8", "udot2x16", "sdot2x16", "sudot2x16",
	"udot8x4", "sdot8x4", "sudot8x4",
	"udot4x8sat", "sdot4x8sat", "sudot4x8sat", "udot2x16sat",
	"sdot2x16sat", "sudot2x16sat", "udot8x4sat", "sdot8x4sat",
	"sudot8x4sat",
	/* the memory kinds (memk.c) */
	"load_shared", "store_shared", "load_ubo", "load_ubo_div",
	"load_stream", "store_stream", "bank_conflict",
	"atomic_shared_same", "atomic_shared_f", "atomic_global_f",
	"atomic_global_64", "atomic_global_cas", "atomic_global_stream",
	"tex_fetch", "tex_sample", "tex_sample_f32", "image_load",
	"tex_nearest",
	"sg_ballot", "sg_vote", "sg_add", "sg_shuffle", "sg_scan",
	"mul16",
	"rop_blend", "rop_adv", "rop_logic", "rop_stencil",
	"bfe", "brev", "mulhi", "funnel", "med3", "sad4x8", "byteperm", "bfi",
	"mulwide",
};

/*
 * A form's three signednesses under one name, as models are usually
 * written: `dot4x8: 1` sets udot4x8, sdot4x8 and sudot4x8.
 */
static const char *const dot_group[6] = {
	"dot4x8", "dot2x16", "dot8x4", "dot4x8sat", "dot2x16sat", "dot8x4sat",
};

const char *const mparam_name[MP_NPAR] = {"banks", "bank_bytes", "warp",
	"subtexel", "tex_within"};

#define MAXMODEL 32
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
	/* a memory kind the model does not list: its fallback's price */
	for (k = C_MEM_FIRST; k <= C_MEM_LAST; k++)
		m->c[k] = NAN;
	m->c[C_BITCAST] = 0;
	m->c[C_F64] = 1;
	/* an instruction the target may not have: absent unless listed */
	for (k = 0; k < C_NCLASS; k++)
		if (C_ABSENT(k))
			m->c[k] = INFINITY;
	return m;
}

int
cclass_available(const struct model *m, int k)
{
	return isfinite(m->c[k]);
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
	/* a group of three: -(100 + the first class) */
	for (k = 0; k < 6; k++)
		if (strlen(dot_group[k]) == n && !memcmp(dot_group[k], s, n))
			return -(100 + C_DOT_FIRST + 3 * k);
	return -2;
}

/* a model's classes as `models` prints them: dot forms only when listed,
 * a group's three by its name when they agree */
void
model_classes_text(struct sbuf *b, const struct model *m)
{
	int k, first = 1;

	for (k = 0; k < C_NBASE; k++) {
		sb_printf(b, "%s %s: %g", first ? "" : ",", cclass_name[k],
		    (double)m->c[k]);
		first = 0;
	}
	for (k = C_DOT_FIRST; k <= C_DOT_LAST; k += 3) {
		int j;
		if (m->c[k] == m->c[k + 1] && m->c[k] == m->c[k + 2]) {
			if (cclass_available(m, k))
				sb_printf(b, ", %s: %g", dot_group[(k - C_DOT_FIRST) / 3],
				    (double)m->c[k]);
			continue;
		}
		for (j = k; j < k + 3; j++)
			if (cclass_available(m, j))
				sb_printf(b, ", %s: %g", cclass_name[j], (double)m->c[j]);
	}
	/* the memory kinds and counts it lists (memk.c) */
	for (k = C_MEM_FIRST; k <= C_MEM_LAST; k++)
		if (model_lists(m, k))
			sb_printf(b, ", %s: %g", cclass_name[k], (double)m->c[k]);
	for (k = 0; k < MP_NPAR; k++)
		if (m->par[k] != 0)
			sb_printf(b, ", %s: %g", mparam_name[k], (double)m->par[k]);
	for (k = C_SG_FIRST; k < C_NCLASS; k++)
		if (cclass_available(m, k))
			sb_printf(b, ", %s: %g", cclass_name[k], (double)m->c[k]);
}

/* an operation's class the model cannot use: a dot form or a subgroup
 * operation it does not list, or a read through the texture unit it
 * does not price */
static int
class_missing(const struct model *m, int k)
{
	if (C_ABSENT(k))
		return !cclass_available(m, k);
	return (k == C_TEX_NEAREST || k == C_TEX_SAMPLE ||
	    k == C_TEX_SAMPLE_F32) && !model_lists(m, k);
}

static int
fpclass(int k)
{
	return k >= C_FADD && k <= C_FPOW;
}

/*
 * An integer division's class: unsigned (uN, bool) or signed (iN, int),
 * quotient or remainder, by a variable or by a constant.  An unsigned
 * division by a power of two is a shift and its remainder a mask, which
 * is what every compiler makes of them.
 */
int
div_class(nid n)
{
	ty_t t = N.ty[n];
	int sgn = TYK(t) == K_SINT || TYK(t) == K_INT || TYK(t) == K_LIT;
	int k = N.op[n] == O_DIV ? (sgn ? C_SDIV : C_IDIV) :
	    (sgn ? C_SREM : C_IREM);
	nid b = N.b[n];

	if (!is_const(b))
		return k;
	if (!sgn && N.ki[b] > 0 && !(N.ki[b] & (N.ki[b] - 1)))
		return N.op[n] == O_DIV ? C_SHIFT : C_LOGIC;
	return C_DIVK(k);
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
		k = fp ? C_FDIV : div_class(n);
		if (!fp && (TYK(t) == K_SINT || TYK(t) == K_INT) &&
		    is_const(N.b[n]) && N.ki[N.b[n]] > 1 &&
		    !(N.ki[N.b[n]] & (N.ki[N.b[n]] - 1))) {
			/*
			 * signed by 2^k: (x + ((x >> w-1) >>> (w-k))) >> k, three
			 * shifts and an add; the remainder x - ((x + that bias) &
			 * -2^k), two shifts, two adds and a mask
			 */
			*cls = C_SHIFT;
			return op == O_DIV ? 3 * m->c[C_SHIFT] + m->c[C_IADD] :
			    2 * m->c[C_SHIFT] + 2 * m->c[C_IADD] + m->c[C_LOGIC];
		}
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
		/* a float add where the model prices it apart (memk.c) */
		if (TY_FP(t) && (N.ki[n] & 15) == AT_ADD) {
			int kf = (N.ki[n] & AT_SHARED) ? C_ATOM_SH_F : C_ATOM_GL_F;
			if (model_lists(m, kf)) {
				*cls = kf;
				return m->c[kf];
			}
		}
		break;
	case O_LOOP:
		return loop_cost(m, n, cls);
	default:
		if (IS_TEX(op))
			return tex_cost(m, n, cls);
		k = IS_DOT(op) ? DOT_CLASS(op) : IS_SG(op) ? sg_class(op) :
		    IS_INST(op) ? inst_class(op) : C_IADD;
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
	struct sbuf b = {0};

	for (i = 0; i < nmodels; i++) {
		sb_clear(&b);
		model_classes_text(&b, &models[i]);
		printf("model %s%s {%s }\n", sym_str(models[i].name),
		    &models[i] == active ? " (active)" : "", sb_str(&b));
		fuse_print_models(&models[i]);
		for (k = 0; k < models[i].nfold; k++)
			printf("  folds %s -> %s -> %s\n",
			    ty_str(models[i].fold[k][0]),
			    ty_str(models[i].fold[k][1]),
			    ty_str(models[i].fold[k][0]));
	}
	free(b.p);
}

/*
 * The cover of the function being priced (fuse.c): each node's share of
 * the price (a fusion's at its root, 0 for what it absorbs), and which
 * nodes a fusion took (they are tallied apart from the classes).
 */
static const float *cur_eff;
static const uint8_t *cur_in, *cur_fused;
static struct fgraph cur_fg;
static struct fuse_tally cur_tally;

static float
ncost(const struct model *m, nid g, int *cls)
{
	float c = node_cost(m, g, cls);

	return cur_eff && cur_in[g] ? cur_eff[g] : c;
}

static int
fused(nid g)
{
	return cur_fused && cur_fused[g];
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

/* vslots: the loop variables (by slot) that vary; a caller that does not
 * look at their inputs itself takes them all as varying */
static int
depends_nonuniform(const struct func *f, nid n, uint8_t *seen,
    uint64_t vslots)
{
	uint32_t j;

	if (n == 0 || seen[n])
		return 0;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_VAR:
		if (N.ki[n] >= f->np)
			return N.ki[n] >= NSLOT || (vslots >> N.ki[n] & 1);
		return !f->p[N.ki[n]].uniform;
	case O_CONST:
	case O_UNDEF:
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			if (depends_nonuniform(f, N.lst[N.a[n] + j], seen,
			    vslots))
				return 1;
		return 0;
	case O_ATOMIC:
		return 1;   /* the old value differs per invocation */
	case O_SGBALLOT: case O_SGANY: case O_SGALL: case O_SGADD:
		/* one value for the subgroup, whatever its lanes hold */
		return 0;
	case O_SGSHUF: case O_SGSCAN:
		return 1;
	}
	return depends_nonuniform(f, N.a[n], seen, vslots) ||
	    depends_nonuniform(f, N.b[n], seen, vslots) ||
	    depends_nonuniform(f, N.c[n], seen, vslots);
}

static float
sum_cost(const struct model *m, const uint8_t *mark, uint32_t lim)
{
	float s = 0;
	uint32_t g;
	int cls;

	for (g = 1; g < lim; g++)
		if (mark[g])
			s += ncost(m, g, &cls);
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
	/* a warp whose lanes run different trip counts (wmode as warp_trips
	 * returns it): the most trips of its lanes on average, the chance
	 * they differ, how many states the vote keeps by a select */
	int wmode, wexact, nkeep;
	double wmax, wsplit;
	uint64_t wpts;
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
/* the while loop n with one more state counting its iterations, its
 * result; 0 when no state is left for it */
static nid
trip_counter(nid n)
{
	struct loopdef d;
	uint32_t e = loop_ent(n);
	int slot = KSLOT - 1;

	loop_get(e, &d);
	if (d.nst >= MAXST || ((loops.bound[e] | loops.free[e]) >> slot) & 1)
		return 0;
	d.sslot[d.nst] = slot;
	d.sname[d.nst] = sym_cstr("_trips");
	d.sty[d.nst] = TY_INT;
	d.init[d.nst] = mk_const_i(TY_INT, 0);
	d.next[d.nst] = mk_bin(O_ADD, var_of(slot, d.sname[d.nst], TY_INT),
	    mk_const_i(TY_INT, 1));
	d.nst++;
	return mk_loop(loop_intern(&d), d.nst - 1, N.a[n], N.b[n], N.c[n]);
}

static void
trip_while(const struct func *f, nid n, struct lprice *lp)
{
	struct diter it;
	struct plan p;
	nid cntr, roots[2];
	int64_t *in_i, mx = 0, mn = -1;
	double *in_r, sum = 0;
	uint32_t np, i;

	lp->worst = lp->avg = lp->best = -1;
	if ((cntr = trip_counter(n)) == 0)
		return;
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

/* a loop's trip count may differ between the lanes (1): its bounds, or
 * the condition with the states that vary (a state varies when its
 * initial or next value reads a varying parameter or state; a vote does
 * not); -1 when states vary and the count does not */
static int
loop_varies(const struct func *f, nid n)
{
	uint32_t e = loop_ent(n), s = loops.st[e], k;
	uint8_t *seen = xcalloc(N.n, 1);
	uint64_t vs = 0, was;
	int v;

	do {
		was = vs;
		for (k = 0; k < loops.nst[e]; k++) {
			memset(seen, 0, N.n);
			if (depends_nonuniform(f, loops.init[s + k], seen, vs) ||
			    depends_nonuniform(f, loops.next[s + k], seen, vs))
				vs |= 1ULL << loops.sslot[s + k];
		}
	} while (vs != was);
	memset(seen, 0, N.n);
	v = loops.kind[e] == LK_FOR ? depends_nonuniform(f, N.a[n], seen, vs) ||
	    depends_nonuniform(f, N.b[n], seen, vs) :
	    depends_nonuniform(f, loops.cond[e], seen, vs);
	free(seen);
	return v ? 1 : vs ? -1 : 0;
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
		c = ncost(m, g, &cls);
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
	/*
	 * Under a model with a warp, a loop whose lanes run different trip
	 * counts (its inputs vary per invocation, its count over the domain):
	 * the warp's count, the most of its lanes'.
	 */
	if (m->par[MP_WARP] >= 2 && lp->worst >= 0 && !lp->known &&
	    lp->best != lp->worst && !(loops.free[e] >> MAXP))
		lp->wmode = loop_varies(f, n);
	if (lp->wmode > 0) {
		nid tc = isfor ? 0 : trip_counter(n);
		lp->wmode = 0;
		if (isfor || tc)
			lp->wmode = warp_trips(f, tc, isfor ? N.a[n] : 0,
			    isfor ? N.b[n] : 0, (int)m->par[MP_WARP], &lp->wmax,
			    &lp->wsplit, &lp->wexact, &lp->wpts);
		for (k = 0; k < loops.nst[e]; k++) {
			nid nx = loops.next[s + k];
			lp->nkeep += !(N.op[nx] == O_VAR &&
			    N.ki[nx] == loops.sslot[s + k]);
		}
	}
	/* by class: once, and per iteration times the worst trip count */
	{
		double t = lp->worst < 0 ? 1 : lp->worst;
		uint8_t *once = xcalloc(lim, 1);
		for (g = 1; g < lim; g++) {
			float c;
			if (outer[g] || !(ini[g] || bod[g]))
				continue;
			c = ncost(m, g, &cls);
			if (cls < 0)
				continue;
			once[g] = !lp->var[g];
			if (fused(g))
				continue;
			bycls[cls] += lp->var[g] ? (float)(c * t) : c;
			cnt[cls] += lp->var[g] ? (int)t : 1;
		}
		if (cur_fused) {
			fg_tally(&cur_fg, lp->var, t, &cur_tally);
			fg_tally(&cur_fg, once, 1, &cur_tally);
		}
		free(once);
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

/*
 * A warp runs a loop until its last lane leaves.  Leaving per lane, the
 * lanes that left are masked off and wait for the rest (the keep-up),
 * and the warps whose lanes leave at different counts pay the
 * divergence once, on reconverging after the loop.  Leaving on a vote,
 * sg_any(c), every lane runs every iteration and nothing diverges, but
 * each state takes a select to stay put in the lanes where c fails.
 */
static void
print_warp(const struct model *m, const struct lprice *lp, const char *cond)
{
	uint32_t e = lp->e, s = loops.st[e], k;
	int i;
	float per = lp->body + lp->over, vc = m->c[C_SG_VOTE];
	double lane = lp->once + lp->wmax * per + lp->wsplit * m->c[C_DIVERGE];
	double vote = lp->once + lp->wmax * (per + vc + lp->nkeep *
	    m->c[C_SELECT]);
	int novote = class_missing(m, C_SG_VOTE), isvote;
	struct sbuf b = {0};

	if (lp->wmode < 0) {
		printf("    in a warp of %d lanes: its lanes leave together, "
		    "the condition being uniform\n", (int)m->par[MP_WARP]);
		rep_obj("warp");
		rep_int("lanes", (int)m->par[MP_WARP]);
		rep_bool("leave_together", 1);
		rep_end();
		return;
	}
	isvote = !novote && vote < lane;
	printf("    in a warp of %d lanes (%s %s%llu %s): it runs until its "
	    "last lane leaves, %.4g iterations on average, and its lanes leave "
	    "at different counts with probability %.4g\n", (int)m->par[MP_WARP],
	    lp->wmode == 2 ? "aligned runs of the consecutive index," :
	    "independent draws from the domain,", lp->wexact ? "all " :
	    "a sample of ", (unsigned long long)lp->wpts, lp->wmode == 2 ?
	    "warps" : "points", lp->wmax, lp->wsplit);
	printf("      exit per lane: %g + %.4g x %g + %.4g x %g diverge (the "
	    "lanes that left keep up) = %g\n", (double)lp->once, lp->wmax,
	    (double)per, lp->wsplit, (double)m->c[C_DIVERGE], lane);
	for (k = 0, i = 0; k < loops.nst[e]; k++) {
		nid nx = loops.next[s + k];
		if (N.op[nx] == O_VAR && N.ki[nx] == loops.sslot[s + k])
			continue;
		i++;
		sb_printf(&b, "%s%s", i == 1 ? "" : i == lp->nkeep ? " and " :
		    ", ", sym_str(loops.sname[s + k]));
	}
	if (novote)
		printf("      exit on a vote: sg_vote is not available under %s "
		    "(the model lists no such instruction)\n", sym_str(m->name));
	else
		printf("      exit on a vote: %g + %.4g x (%g + %g sg_vote + %d "
		    "x %g select) = %g  (while sg_any(%s)%s%s%s)\n",
		    (double)lp->once, lp->wmax, (double)per, (double)vc,
		    lp->nkeep, (double)m->c[C_SELECT], vote, cond, b.n ?
		    ", a select keeping " : "", b.n ? sb_str(&b) : "", b.n ?
		    " where it fails" : "");
	printf("      -> %s\n", isvote ? "exit on a vote" : "exit per lane");
	rep_obj("warp");
	rep_int("lanes", (int)m->par[MP_WARP]);
	rep_str("lanes_are", lp->wmode == 2 ? "consecutive" : "independent");
	rep_int(lp->wmode == 2 ? "warps" : "points", (long long)lp->wpts);
	rep_bool("all_counted", lp->wexact);
	rep_num("trips_most_average", lp->wmax);
	rep_num("p_split", lp->wsplit);
	rep_num("exit_per_lane", lane);
	if (novote)
		rep_null("exit_on_vote");
	else
		rep_num("exit_on_vote", vote);
	rep_strf("vote", "while sg_any(%s)", cond);
	rep_int("selects_kept", lp->nkeep);
	rep_str("best", isvote ? "exit on a vote" : "exit per lane");
	rep_end();
	free(b.p);
}

static void
print_loop(const struct model *m, const struct func *f,
    const struct lprice *lp)
{
	uint32_t e = lp->e;
	char head[160], cond[160];
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
	/* the condition a vote reads: a for loop's is its index within hi */
	sb_clear(&b);
	if (loops.kind[e] == LK_FOR) {
		sb_printf(&b, "%s <= ", sym_str(loops.name[e]));
		print_node(&b, N.b[lp->n], NULL);
	} else
		print_shared(&b, loops.cond[e], 60);
	cap_text(&b, 0, 80);
	snprintf(cond, sizeof cond, "%s", sb_str(&b));
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
	if (lp->wmode)
		print_warp(m, lp, cond);
	rep_end();
}

#define MAXSEL_SHOWN 12

static void
cost_report(struct func *f, struct model *m, const struct func *src)
{
	uint32_t lim = N.n, g;
	uint8_t *all, *cut, *seen;
	float bycls[C_NCLASS] = {0}, total = 0, avg_total, best_total, c;
	int cnt[C_NCLASS] = {0}, cls, k, nops = 0, nsel = 0, nloop = 0;
	int varying = 0;
	struct lprice *lps = NULL;
	struct shmap *sh;
	struct sbuf cb = {0};
	struct func flat;

	/* subgroup operations: priced as one lane runs them, a value per
	 * lane once whether an operation or the lane itself reads it; new
	 * nodes, so before anything is sized by N.n */
	if (N.fl[f->body] & NF_SG) {
		flat = *f;
		flat.body = sg_priced(f, f->body);
		f = &flat;
		lim = N.n;
	}
	all = xcalloc(lim, 1);
	cut = xcalloc(lim, 1);
	seen = xcalloc(lim, 1);
	reach(f->body, all, 0, 0);
	/* the cover by the model's fusions: outside loops and in their
	 * bodies, each loop's state a result of its own */
	{
		uint8_t *cm = xcalloc(lim, 1), *fz = xcalloc(lim, 1);
		nid *pins = xmalloc((size_t)lim * sizeof *pins);
		int np = 0, i;
		float *eff = xcalloc(lim, sizeof *eff);
		memcpy(cm, all, lim);
		pins[np++] = f->body;
		for (g = 1; g < lim; g++)
			if (all[g] && N.op[g] == O_LOOP) {
				uint32_t e = loop_ent(g), s = loops.st[e], q;
				for (q = 0; q < loops.nst[e]; q++) {
					reach(loops.init[s + q], cm, 0, 0);
					reach(loops.next[s + q], cm, 0, 0);
					pins[np++] = loops.init[s + q];
					pins[np++] = loops.next[s + q];
				}
				reach(loops.cond[e], cm, 0, 0);
				pins[np++] = loops.cond[e];
			}
		memset(&cur_tally, 0, sizeof cur_tally);
		fuse_domain(f);
		fuse_cover_nodes(m, cm, lim, pins, np, 0, eff, &cur_fg);
		fuse_domain(NULL);
		for (i = 0; i < cur_fg.n; i++)
			if (cur_fg.fz[i] >= 0)
				fz[cur_fg.id[i]] = 1;
		cur_eff = eff;
		cur_in = cm;
		cur_fused = fz;
		free(pins);
	}
	for (g = 1; g < lim; g++) {
		if (!all[g] || N.op[g] == O_LOOP)
			continue;
		c = ncost(m, g, &cls);
		if (cls < 0)
			continue;
		if (!fused(g)) {
			bycls[cls] += c;
			cnt[cls]++;
		}
		total += c;
		nops++;
	}
	fg_tally(&cur_fg, all, 1, &cur_tally);
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
	if (cur_tally.n) {
		printf("  fused by the model (one instruction for several "
		    "operations):\n");
		rep_list("fusions");
		fuse_tally_print(&cur_tally, "    ");
		rep_end();
	}
	for (k = 0; k < C_NCLASS; k++)
		if (cnt[k] && class_missing(m, k))
			break;
	if (k < C_NCLASS) {
		rep_list("unavailable");
		for (k = 0; k < C_NCLASS; k++)
			if (cnt[k] && class_missing(m, k)) {
				printf("  %s is not available under %s (the model "
				    "lists no such instruction)\n", cclass_name[k],
				    sym_str(m->name));
				rep_str(NULL, cclass_name[k]);
			}
		rep_end();
	}
	if (nloop) {
		rep_list("loops");
		for (k = 0; k < nloop; k++)
			print_loop(m, f, &lps[k]);
		rep_end();
	}
	if (src != f)
		fold_cost_notes(src, m);
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
			float ex[2], sel, uni, div, expd = 0;
			double dp = 0, dat = 0, daf = 0;
			uint64_t dpts = 0;
			int i, nonuni, dmode, dexact = 0;

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
			nonuni = depends_nonuniform(f, N.a[s], seen, ~0ULL);
			sel = m->c[C_SELECT];
			uni = m->c[C_BRANCH] + (ex[0] > ex[1] ? ex[0] : ex[1]) -
			    ex[0] - ex[1];
			div = m->c[C_BRANCH] + m->c[C_DIVERGE];
			/*
			 * A branch on a varying condition, on average: the warps
			 * that agree skip the other arm, the ones that split pay
			 * the divergence (stoch.c counts or estimates how many).
			 */
			dmode = 0;
			if (nonuni && k < 0 && m->par[MP_WARP] >= 2)
				dmode = diverge_stats(f, N.a[s], (int)m->par[MP_WARP], &dp, &dat,
				    &daf, &dexact, &dpts);
			if (dmode)
				expd = m->c[C_BRANCH] + (float)((1 - dat - daf) *
				    (double)m->c[C_DIVERGE] - dat * (double)ex[1] -
				    daf * (double)ex[0]);
			{
				float pick = nonuni ? (dmode ? (sel <= expd ? sel : expd) :
				    (sel < div ? sel : div)) : (sel < uni ? sel : uni);
				const char *name = nonuni ? (dmode ? (sel <= expd ?
				    "select" : "branch (its expected cost)") :
				    (sel <= div ? "select" : "divergent branch")) :
				    (sel <= uni ? "select" : "uniform branch");
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
				if (dmode == 2)
					printf("      expected branch: %+g   (%llu warps of "
					    "%d consecutive lanes%s: %.4g all true, %.4g "
					    "all false, %.4g split)\n", (double)expd,
					    (unsigned long long)dpts, (int)m->par[MP_WARP],
					    dexact ? ", all counted" : ", sampled", dat, daf,
					    1 - dat - daf);
				else if (dmode == 1)
					printf("      expected branch: %+g   (P(true) %s "
					    "%.4g%s%s%llu points; %d independent lanes split "
					    "with probability 1 - p^%d - (1 - p)^%d = %.4g)\n",
					    (double)expd, dexact < 0 ? (dp < 0.5 ? "<=" : ">=") :
					    "=", dp, dexact < 0 ? " at 95% confidence: " : " over ",
					    dexact < 0 ? (dp < 0.5 ? "never in a sample of " :
					    "always in a sample of ") : dexact ? "all " :
					    "a sample of ", (unsigned long long)dpts,
					    (int)m->par[MP_WARP], (int)m->par[MP_WARP],
					    (int)m->par[MP_WARP], 1 - dat - daf);
				rep_obj(NULL);
				sb_clear(&cb);
				print_with(&cb, N.a[s], sh, 300);
				rep_str("condition", sb_str(&cb));
				rep_num("select", sel);
				rep_num("uniform_branch", uni);
				rep_num("divergent_branch", div);
				rep_bool("uniform_condition", !nonuni);
				if (dmode) {
					rep_num("expected_branch", expd);
					rep_num("p_true", dp);
					rep_num("p_split", 1 - dat - daf);
					rep_str("lanes", dmode == 2 ? "consecutive" :
					    "independent");
				}
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
	free((void *)cur_eff);
	free((void *)cur_in);
	free((void *)cur_fused);
	cur_eff = NULL;
	cur_in = cur_fused = NULL;
	fg_free(&cur_fg);
}

/*
 * F is priced as the model's driver runs it: the float round trips the
 * model says are folded (fpfold.c) are gone before anything is counted,
 * so what they sat between fuses as it would; the report then says which
 * and whether F leaned on their rounding.
 */
void
cmd_cost(struct func *f, struct model *m)
{
	struct func v = *f;

	v.body = fold_apply(m, f->body);
	cost_report(v.body != f->body ? &v : f, m, f);
}
