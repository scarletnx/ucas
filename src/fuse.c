/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Fusions: what a cost model says one instruction does that the classes
 * price as several (a multiply-add in one XMAD chain or FFMA, a tree of
 * bitwise logic in one LOP3, a negation as a free source modifier).  A
 * model declares them (fuse lines, src/fuse.h); nothing here knows a GPU.
 *
 * Pricing covers the DAG with them: instruction selection by dynamic
 * programming over trees.  A node used more than once (or pinned: a
 * result, a loop's state) is computed once and is a leaf to every user,
 * so the DAG falls into trees; in each tree a node's best price is the
 * cheapest of
 *   - itself, plus its operands' best prices;
 *   - a pattern rooted at it, plus the best prices of the pattern's
 *     leaves (its interior nodes must be used only inside it);
 *   - a lut: a cut of up to k leaves of the bitwise-logic cone below it
 *     (k-feasible cuts, as technology mappers enumerate them), priced
 *     once, plus the leaves' best prices.
 * That is the exact cheapest cover of each tree, without duplicating a
 * shared value into two instructions.
 */
#include "calc.h"
#include "fuse.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static struct fusion fz[MAXFUSE * 4];
static int nfz;

void
fuse_reset(const struct model *m)
{
	int i, j = 0;

	for (i = 0; i < nfz; i++)
		if (fz[i].model != m->name)
			fz[j++] = fz[i];
	nfz = j;
}

void
fuse_remove(const struct model *m, uint32_t name)
{
	int i, j = 0, found = 0;

	for (i = 0; i < nfz; i++)
		if (fz[i].model == m->name && fz[i].name == name)
			found = 1;
		else
			fz[j++] = fz[i];
	nfz = j;
	if (!found)
		fail("model %s has no fusion %s", sym_str(m->name),
		    sym_str(name));
}

void
fuse_add(const struct model *m, const struct fusion *f)
{
	int i;

	for (i = 0; i < nfz; i++)
		if (fz[i].model == m->name && fz[i].name == f->name) {
			fz[i] = *f;
			fz[i].model = m->name;
			return;
		}
	if (nfz == MAXFUSE * 4 || fuse_count(m) == MAXFUSE)
		fail("too many fusions");
	fz[nfz] = *f;
	fz[nfz++].model = m->name;
}

void
fuse_inherit(const struct model *m, const struct model *parent)
{
	int i, n = nfz;

	fuse_reset(m);
	n = nfz;
	for (i = 0; i < n; i++)
		if (fz[i].model == parent->name) {
			struct fusion c = fz[i];
			fuse_add(m, &c);
		}
}

int
fuse_count(const struct model *m)
{
	int i, n = 0;

	for (i = 0; i < nfz; i++)
		n += fz[i].model == m->name;
	return n;
}

static const char *const pkname[] = {"int", "float", "const", "any"};

void
fuse_text(struct sbuf *b, const struct fusion *f)
{
	int i, k;
	struct pctx pc;
	const char *vn[FUSE_MAXP];

	sb_printf(b, "fuse %s", sym_str(f->name));
	if (f->np) {
		sb_putc(b, '(');
		for (i = 0; i < f->np; i++)
			sb_printf(b, "%s%s: %s", i ? ", " : "",
			    sym_str(f->pname[i]), pkname[f->pkind[i]]);
		sb_putc(b, ')');
	}
	sb_puts(b, " = ");
	if (f->lut)
		sb_printf(b, "lut %d", f->lut);
	else {
		for (i = 0; i < f->np; i++)
			vn[i] = sym_str(f->pname[i]);
		memset(&pc, 0, sizeof pc);
		pc.vn = vn;
		print_node(b, f->pat, &pc);
	}
	sb_puts(b, " : ");
	for (i = 0; i < f->nt; i++) {
		if (i)
			sb_puts(b, " + ");
		if (f->tmul[i] != 1)
			sb_printf(b, "%g*", (double)f->tmul[i]);
		sb_puts(b, cclass_name[f->tcls[i]]);
	}
	if (f->k != 0 || !f->nt)
		sb_printf(b, "%s%g", f->nt ? " + " : "", (double)f->k);
	if (f->contract)
		sb_puts(b, " contract");
	if (f->users) {
		sb_puts(b, " in");
		for (k = 0; k < C_NCLASS && k < 64; k++)
			if (f->users >> k & 1)
				sb_printf(b, " %s", cclass_name[k]);
	}
}

void
fuse_print_models(const struct model *m)
{
	int i;
	struct sbuf b = {0};

	for (i = 0; i < nfz; i++)
		if (fz[i].model == m->name) {
			sb_clear(&b);
			fuse_text(&b, &fz[i]);
			printf("  %s\n", sb_str(&b));
		}
	free(b.p);
}

static int
fpclass(int k)
{
	return k >= C_FADD && k <= C_FPOW;
}

/* the fusion's price under m (fp64 operations cost f64 times more) */
float
fuse_price(const struct model *m, const struct fusion *f, int f64)
{
	float c = f->k;
	int i;

	for (i = 0; i < f->nt; i++)
		c += f->tmul[i] * m->c[f->tcls[i]] *
		    (f64 && fpclass(f->tcls[i]) ? m->c[C_F64] : 1);
	return c;
}

/* ---- graphs ---- */

void
fg_init(struct fgraph *g, int n)
{
	memset(g, 0, sizeof *g);
	g->n = n;
	if (n < 1)
		n = 1;
	g->op = xcalloc((size_t)n, 1);
	g->fp = xcalloc((size_t)n, 1);
	g->f64 = xcalloc((size_t)n, 1);
	g->isk = xcalloc((size_t)n, 1);
	g->uns = xcalloc((size_t)n, 1);
	g->isbool = xcalloc((size_t)n, 1);
	g->nocontract = xcalloc((size_t)n, 1);
	g->pinned = xcalloc((size_t)n, 1);
	g->kv = xcalloc((size_t)n, sizeof *g->kv);
	g->kr = xcalloc((size_t)n, sizeof *g->kr);
	g->arg = xmalloc((size_t)n * sizeof *g->arg);
	memset(g->arg, 0xff, (size_t)n * sizeof *g->arg);
	g->uses = xcalloc((size_t)n, sizeof *g->uses);
	g->own = xcalloc((size_t)n, sizeof *g->own);
	g->cls = xmalloc((size_t)n * sizeof *g->cls);
	memset(g->cls, 0xff, (size_t)n * sizeof *g->cls);
	g->eff = xcalloc((size_t)n, sizeof *g->eff);
	g->fz = xmalloc((size_t)n * sizeof *g->fz);
	memset(g->fz, 0xff, (size_t)n * sizeof *g->fz);
	g->by = xmalloc((size_t)n * sizeof *g->by);
	memset(g->by, 0xff, (size_t)n * sizeof *g->by);
	g->own0 = xcalloc((size_t)n, sizeof *g->own0);
	g->id = xcalloc((size_t)n, sizeof *g->id);
}

void
fg_free(struct fgraph *g)
{
	free(g->op);
	free(g->fp);
	free(g->f64);
	free(g->isk);
	free(g->uns);
	free(g->isbool);
	free(g->nocontract);
	free(g->pinned);
	free(g->kv);
	free(g->kr);
	free(g->arg);
	free(g->uses);
	free(g->own);
	free(g->cls);
	free(g->eff);
	free(g->fz);
	free(g->by);
	free(g->own0);
	free(g->id);
	memset(g, 0, sizeof *g);
}

/* ---- matching a pattern at a node ---- */

struct mst {
	int bind[FUSE_MAXP];
	int inter[32], ninter;
};

static int
inner_ok(const struct fgraph *g, int i)
{
	return g->uses[i] == 1 && !g->pinned[i];
}

static int
is_logic(const struct fgraph *g, int i)
{
	int op = g->op[i];

	return !g->fp[i] && !g->isbool[i] && (op == O_AND || op == O_OR ||
	    op == O_XOR || op == O_NOT);
}

static int fmatch(const struct fgraph *g, const struct fusion *f, nid P,
    int i, int root, struct mst *s);

static int
fmatch_args(const struct fgraph *g, const struct fusion *f, nid P, int i,
    int root, struct mst *s, int swap)
{
	nid pa[3] = {N.a[P], N.b[P], N.c[P]};
	int ga[3] = {g->arg[i][0], g->arg[i][1], g->arg[i][2]}, k, ar;

	ar = opinfo[N.op[P]].arity;
	if (swap) {
		int t = ga[0];
		ga[0] = ga[1];
		ga[1] = t;
	}
	for (k = 0; k < ar && k < 3; k++)
		if (ga[k] < 0 || !fmatch(g, f, pa[k], ga[k], root, s))
			return 0;
	return 1;
}

static int
log2k(int64_t v)
{
	int k = 0;

	if (v <= 0 || (v & (v - 1)))
		return -1;
	while (v > 1) {
		v >>= 1;
		k++;
	}
	return k;
}

/* the shift amount of a pattern's x << k against a power of two 2^e */
static int
match_amount(const struct fgraph *g, const struct fusion *f, nid P,
    int kn, int e, struct mst *s)
{
	if (N.op[P] == O_CONST)
		return N.ki[P] == e;
	if (N.op[P] == O_VAR && (f->pkind[N.ki[P]] == FP_CONST ||
	    f->pkind[N.ki[P]] == FP_ANY)) {
		int sl = (int)N.ki[P];
		if (s->bind[sl] >= 0)
			return s->bind[sl] == kn;
		s->bind[sl] = kn;	/* the constant 2^e stands for k */
		return 1;
	}
	return 0;
}

static int
fmatch(const struct fgraph *g, const struct fusion *f, nid P, int i,
    int root, struct mst *s)
{
	int op = N.op[P];

	if (op == O_VAR) {
		int sl = (int)N.ki[P];
		switch (f->pkind[sl]) {
		case FP_INT:
			if (g->fp[i] || g->isbool[i])
				return 0;
			break;
		case FP_FLOAT:
			if (!g->fp[i])
				return 0;
			break;
		case FP_CONST:
			if (!g->isk[i])
				return 0;
			break;
		}
		if (s->bind[sl] >= 0)
			return s->bind[sl] == i;
		s->bind[sl] = i;
		return 1;
	}
	if (op == O_CONST) {
		if (!g->isk[i])
			return 0;
		return TY_FP(N.ty[P]) ? g->fp[i] && g->kr[i] == N.kr[P] :
		    !g->fp[i] && g->kv[i] == N.ki[P];
	}
	if (g->isk[i] || (i != root && !inner_ok(g, i)))
		return 0;
	if ((g->fp[i] != 0) != (TY_FP(N.ty[P]) != 0) || g->isbool[i])
		return 0;
	if (f->contract && g->nocontract[i])
		return 0;
	if (i != root) {
		if (s->ninter == 32)
			return 0;
		s->inter[s->ninter++] = i;
	}
	if (g->op[i] != op) {
		/* x << k is also x * 2^k; x >> k also x / 2^k (unsigned) */
		int j, e;
		if ((op == O_SHL && g->op[i] == O_MUL && !g->fp[i]) ||
		    (op == O_SHR && g->op[i] == O_DIV && g->uns[i]))
			for (j = op == O_SHL ? 0 : 1; j < 2; j++) {
				int kn = g->arg[i][j], xn = g->arg[i][1 - j];
				struct mst save = *s;
				if (kn < 0 || xn < 0 || !g->isk[kn] ||
				    (e = log2k(g->kv[kn])) < 0)
					continue;
				if (fmatch(g, f, N.a[P], xn, root, s) &&
				    match_amount(g, f, N.b[P], kn, e, s))
					return 1;
				*s = save;
			}
		return 0;
	}
	{
		struct mst save = *s;
		if (fmatch_args(g, f, P, i, root, s, 0))
			return 1;
		*s = save;
		if ((opinfo[op].fl & OF_COMM) && opinfo[op].arity == 2 &&
		    fmatch_args(g, f, P, i, root, s, 1))
			return 1;
		*s = save;
	}
	return 0;
}

static void
mst_init(struct mst *s)
{
	int k;

	for (k = 0; k < FUSE_MAXP; k++)
		s->bind[k] = -1;
	s->ninter = 0;
}

/* ---- cuts for lut fusions ---- */

#define MAXCUT 24
struct cut {
	int n;
	int l[6];
};

static int
cut_merge(const struct cut *a, const struct cut *b, int k, struct cut *r)
{
	int i, j;

	*r = *a;
	for (j = 0; j < b->n; j++) {
		for (i = 0; i < r->n && r->l[i] != b->l[j]; i++)
			;
		if (i == r->n) {
			if (r->n == k)
				return 0;
			r->l[r->n++] = b->l[j];
		}
	}
	return 1;
}

static int
cut_same(const struct cut *a, const struct cut *b)
{
	int i, j;

	if (a->n != b->n)
		return 0;
	for (i = 0; i < a->n; i++) {
		for (j = 0; j < b->n && b->l[j] != a->l[i]; j++)
			;
		if (j == b->n)
			return 0;
	}
	return 1;
}

/* the logic nodes between root and the cut (the cone it covers) */
static int
cut_cone(const struct fgraph *g, int root, const struct cut *c, int *out,
    int max)
{
	int st[64], sp = 0, n = 0, i, k, x;

	st[sp++] = root;
	while (sp) {
		x = st[--sp];
		for (k = 0; k < c->n; k++)
			if (c->l[k] == x)
				break;
		if (k < c->n)
			continue;
		for (i = 0; i < n; i++)
			if (out[i] == x)
				break;
		if (i < n)
			continue;
		if (n == max)
			return -1;
		out[n++] = x;
		for (k = 0; k < 3; k++)
			if (g->arg[x][k] >= 0 && sp < 64)
				st[sp++] = g->arg[x][k];
	}
	return n;
}

/* ---- the cover ---- */

enum { CH_PLAIN, CH_PAT, CH_LUT };

float
fg_cover(const struct model *m, struct fgraph *g)
{
	int n = g->n, i, j, k, q;
	float *best = xcalloc((size_t)(n ? n : 1), sizeof *best);
	int *choice = xcalloc((size_t)(n ? n : 1), sizeof *choice);
	int *which = xcalloc((size_t)(n ? n : 1), sizeof *which);
	struct cut (*cuts)[MAXCUT] = NULL;
	int *ncut = NULL, lutk = 0, lutf = -1;
	struct cut *bestcut = xcalloc((size_t)(n ? n : 1), sizeof *bestcut);
	float total = 0;

	g->nfl = 0;
	memcpy(g->own0, g->own, (size_t)n * sizeof *g->own);
	for (i = 0; i < nfz; i++)
		if (fz[i].model == m->name && g->nfl < MAXFUSE) {
			if (fz[i].lut && fz[i].lut > lutk && fz[i].lut <= 6) {
				lutk = fz[i].lut;
				lutf = g->nfl;
			}
			g->fl[g->nfl++] = &fz[i];
		}
	/* modifiers: their own price when every use absorbs them */
	for (q = 0; q < g->nfl; q++) {
		const struct fusion *f = g->fl[q];
		uint8_t *bad;
		if (!f->users || f->lut)
			continue;
		bad = xcalloc((size_t)(n ? n : 1), 1);
		for (j = 0; j < n; j++)
			for (k = 0; k < 3; k++)
				if (g->arg[j][k] >= 0 && (g->cls[j] < 0 ||
				    g->cls[j] >= 64 ||	/* the mask's width */
				    !(f->users >> g->cls[j] & 1)))
					bad[g->arg[j][k]] = 1;
		for (i = 0; i < n; i++) {
			struct mst s;
			float c;
			if (bad[i] || g->pinned[i] || g->uses[i] < 1 || g->isk[i])
				continue;
			mst_init(&s);
			if (!fmatch(g, f, f->pat, i, i, &s) || s.ninter)
				continue;
			c = fuse_price(m, f, g->f64[i]);
			if (c < g->own[i]) {
				g->own[i] = c;
				g->fz[i] = q;
			}
		}
		free(bad);
	}
	if (lutk) {
		cuts = xmalloc((size_t)(n ? n : 1) * sizeof *cuts);
		ncut = xcalloc((size_t)(n ? n : 1), sizeof *ncut);
	}
	for (i = 0; i < n; i++) {
		float plain = g->own[i];
		for (k = 0; k < 3; k++) {
			int a = g->arg[i][k];
			if (a >= 0 && inner_ok(g, a))
				plain += best[a];
		}
		best[i] = plain;
		choice[i] = CH_PLAIN;
		if (g->isk[i] || g->op[i] == O_NONE)
			continue;
		/* patterns rooted here */
		for (q = 0; q < g->nfl; q++) {
			const struct fusion *f = g->fl[q];
			struct mst s;
			float c;
			int l;
			if (f->lut || f->users)
				continue;
			mst_init(&s);
			if (!fmatch(g, f, f->pat, i, i, &s))
				continue;
			c = fuse_price(m, f, g->f64[i]);
			for (l = 0; l < f->np; l++) {
				int b = s.bind[l], d;
				/* a leaf bound twice is counted once */
				for (d = 0; d < l; d++)
					if (s.bind[d] == b)
						break;
				if (b >= 0 && d == l && inner_ok(g, b))
					c += best[b];
			}
			if (c < best[i] - 1e-6f) {
				best[i] = c;
				choice[i] = CH_PAT;
				which[i] = q;
			}
		}
		/* lut: the cuts of the logic cone */
		if (lutk && is_logic(g, i)) {
			struct cut acc[MAXCUT], tmp[MAXCUT], self;
			int na = 1, nt;
			acc[0].n = 0;
			for (k = 0; k < 3; k++) {
				int a = g->arg[i][k], ca, cb;
				struct cut opts[MAXCUT + 1];
				int no = 0;
				if (a < 0)
					continue;
				opts[no].n = 1;
				opts[no++].l[0] = a;
				if (is_logic(g, a) && inner_ok(g, a))
					for (ca = 0; ca < ncut[a] && no <= MAXCUT; ca++)
						opts[no++] = cuts[a][ca];
				nt = 0;
				for (ca = 0; ca < na; ca++)
					for (cb = 0; cb < no && nt < MAXCUT; cb++) {
						struct cut r;
						int d;
						if (!cut_merge(&acc[ca], &opts[cb], lutk, &r))
							continue;
						for (d = 0; d < nt && !cut_same(&tmp[d], &r); d++)
							;
						if (d == nt)
							tmp[nt++] = r;
					}
				memcpy(acc, tmp, (size_t)nt * sizeof *acc);
				na = nt;
			}
			ncut[i] = na;
			memcpy(cuts[i], acc, (size_t)na * sizeof *acc);
			self.n = 0;
			for (j = 0; j < na; j++) {
				float c = fuse_price(m, g->fl[lutf], 0);
				int l, cone[32], nc = cut_cone(g, i, &acc[j], cone, 32);
				if (nc < 2)
					continue;	/* one operation: not a fusion */
				for (l = 0; l < acc[j].n; l++)
					if (inner_ok(g, acc[j].l[l]))
						c += best[acc[j].l[l]];
				if (c < best[i] - 1e-6f) {
					best[i] = c;
					choice[i] = CH_LUT;
					which[i] = lutf;
					bestcut[i] = acc[j];
				}
			}
			(void)self;
		}
	}
	/* the trees' roots, then down each tree: who pays for what */
	{
		int *st = xmalloc((size_t)(n ? n : 1) * sizeof *st), sp = 0;
		for (i = 0; i < n; i++) {
			if (inner_ok(g, i))
				continue;
			total += best[i];
			st[sp++] = i;
			while (sp) {
				int x = st[--sp];
				if (choice[x] == CH_PLAIN) {
					g->eff[x] = g->own[x];
					for (k = 0; k < 3; k++) {
						int a = g->arg[x][k];
						if (a >= 0 && inner_ok(g, a))
							st[sp++] = a;
					}
				} else if (choice[x] == CH_PAT) {
					const struct fusion *f = g->fl[which[x]];
					struct mst s;
					int l;
					mst_init(&s);
					fmatch(g, f, f->pat, x, x, &s);
					g->eff[x] = fuse_price(m, f, g->f64[x]);
					g->fz[x] = which[x];
					for (l = 0; l < s.ninter; l++) {
						g->eff[s.inter[l]] = 0;
						g->by[s.inter[l]] = x;
						g->fz[s.inter[l]] = which[x];
					}
					for (l = 0; l < f->np; l++) {
						int b = s.bind[l], d;
						for (d = 0; d < l; d++)
							if (s.bind[d] == b)
								break;
						if (b >= 0 && d == l && inner_ok(g, b))
							st[sp++] = b;
					}
				} else {
					int cone[32], nc, l;
					nc = cut_cone(g, x, &bestcut[x], cone, 32);
					g->eff[x] = fuse_price(m, g->fl[which[x]], 0);
					g->fz[x] = which[x];
					for (l = 1; l < nc; l++) {
						g->eff[cone[l]] = 0;
						g->by[cone[l]] = x;
						g->fz[cone[l]] = which[x];
					}
					for (l = 0; l < bestcut[x].n; l++)
						if (inner_ok(g, bestcut[x].l[l]))
							st[sp++] = bestcut[x].l[l];
				}
			}
		}
		free(st);
	}
	free(best);
	free(choice);
	free(which);
	free(bestcut);
	free(cuts);
	free(ncut);
	return total;
}

/*
 * What the cover fused among the nodes in `in` (all when NULL), each
 * fusion's count, price and saving, times `times` (a loop's trips).
 */
void
fg_tally(const struct fgraph *g, const uint8_t *in, double times,
    struct fuse_tally *t)
{
	int i, j;

	for (i = 0; i < g->n; i++) {
		const struct fusion *f;
		double saved;
		if (g->fz[i] < 0 || g->by[i] >= 0 || (in && !in[g->id[i]]))
			continue;
		f = g->fl[g->fz[i]];
		saved = g->own0[i] - g->eff[i];
		for (j = 0; j < g->n; j++)
			if (g->by[j] == i)
				saved += g->own0[j];
		for (j = 0; j < t->n && t->f[j] != f; j++)
			;
		if (j == t->n) {
			if (t->n == MAXFUSE)
				continue;
			t->f[t->n] = f;
			t->cnt[j] = t->cost[j] = t->saved[j] = 0;
			t->n++;
		}
		t->cnt[j] += times;
		t->cost[j] += g->eff[i] * times;
		t->saved[j] += saved * times;
	}
}

void
fuse_tally_print(const struct fuse_tally *t, const char *ind)
{
	int j;
	struct sbuf b = {0};

	for (j = 0; j < t->n; j++) {
		const struct fusion *f = t->f[j];
		sb_clear(&b);
		if (f->lut)
			sb_printf(&b, "bitwise logic of up to %d inputs", f->lut);
		else {
			const char *vn[FUSE_MAXP];
			struct pctx pc;
			int i;
			for (i = 0; i < f->np; i++)
				vn[i] = sym_str(f->pname[i]);
			memset(&pc, 0, sizeof pc);
			pc.vn = vn;
			print_node(&b, f->pat, &pc);
		}
		printf("%s%-14s %3g x %-5g = %g  (%s%s; saves %g)\n", ind,
		    sym_str(f->name), t->cnt[j], t->cnt[j] ? t->cost[j] /
		    t->cnt[j] : 0, t->cost[j], sb_str(&b), f->users ?
		    ", a source modifier" : f->contract ? ", contracted: one "
		    "rounding" : "", t->saved[j]);
		rep_obj(NULL);
		rep_str("fusion", sym_str(f->name));
		rep_str("pattern", sb_str(&b));
		rep_num("count", t->cnt[j]);
		rep_num("cost", t->cost[j]);
		rep_num("saved", t->saved[j]);
		if (f->contract)
			rep_bool("contracted", 1);
		rep_end();
	}
	free(b.p);
}

/* ---- ucas nodes ---- */

/* the index of nid g among the sorted nodes, or -1 */
static int
ix_of(const nid *v, int n, nid g)
{
	int lo = 0, hi = n - 1;

	while (lo <= hi) {
		int mid = (lo + hi) / 2;
		if (v[mid] == g)
			return mid;
		if (v[mid] < g)
			lo = mid + 1;
		else
			hi = mid - 1;
	}
	return -1;
}

/* the cover of the nodes v[0..cnt) (ascending nids: operands first) */
static float
cover_list(const struct model *m, const nid *v, int cnt, const nid *pins,
    int npins, int loops_own, struct fgraph *G)
{
	uint32_t j;
	int i, k;

	fg_init(G, cnt);
	for (i = 0; i < cnt; i++) {
		nid g = v[i];
		int op = N.op[g], cls;
		ty_t t = N.ty[g];
		G->id[i] = g;
		G->op[i] = (uint8_t)op;
		G->fp[i] = TY_FP(t) != 0;
		G->f64[i] = TYK(t) == K_FLOAT && TYW(t) == 64;
		G->isbool[i] = t == TY_BOOL;
		G->uns[i] = TYK(t) == K_UINT || t == TY_BOOL;
		if (op == O_CONST) {
			G->isk[i] = 1;
			G->kv[i] = N.ki[g];
			G->kr[i] = N.kr[g];
		}
		if (op == O_LOOP && !loops_own) {
			G->own[i] = 0;
			G->cls[i] = -1;
		} else {
			G->own[i] = node_cost(m, g, &cls);
			G->cls[i] = cls;
		}
		if (op == O_CONST || op == O_VAR || op == O_UNDEF)
			continue;
		if (op == O_TABLE) {
			G->op[i] = O_NONE;
			for (j = 0; j < N.b[g]; j++) {
				int e = ix_of(v, cnt, N.lst[N.a[g] + j]);
				if (e >= 0) {
					G->uses[e]++;
					G->pinned[e] = 1;
				}
			}
			continue;
		}
		if (op == O_LOOP || op == O_BARRIER || op == O_ATOMIC ||
		    op == O_INDEX)
			G->op[i] = O_NONE;	/* priced as they are */
		{
			nid ar[3] = {N.a[g], N.b[g], N.c[g]};
			for (k = 0; k < 3; k++) {
				int e = ar[k] ? ix_of(v, cnt, ar[k]) : -1;
				if (e < 0)
					continue;
				G->uses[e]++;
				if (G->op[i] != O_NONE)
					G->arg[i][k] = e;
				else
					G->pinned[e] = 1;
			}
		}
	}
	for (k = 0; k < npins; k++) {
		int e = pins[k] ? ix_of(v, cnt, pins[k]) : -1;
		if (e >= 0)
			G->pinned[e] = 1;
	}
	return fg_cover(m, G);
}

float
fuse_cover_nodes(const struct model *m, const uint8_t *mark, uint32_t lim,
    const nid *pins, int npins, int loops_own, float *eff,
    struct fgraph *keep)
{
	uint32_t g;
	int cnt = 0, i;
	nid *v = xmalloc((size_t)(lim ? lim : 1) * sizeof *v);
	struct fgraph G;
	float total;

	for (g = 1; g < lim; g++)
		if (mark[g])
			v[cnt++] = g;
	total = cover_list(m, v, cnt, pins, npins, loops_own, &G);
	if (eff)
		for (i = 0; i < cnt; i++)
			eff[v[i]] = G.eff[i];
	if (keep)
		*keep = G;	/* id[] maps its nodes back to nids */
	else
		fg_free(&G);
	free(v);
	return total;
}

static int
cmp_nid(const void *x, const void *y)
{
	nid a = *(const nid *)x, b = *(const nid *)y;

	return a < b ? -1 : a > b;
}

/*
 * Every node reachable from root (into tables, a loop priced as a node):
 * its cheapest cover.  The visited set is kept between calls and cleared
 * by its list, so a small expression costs little in a big node table.
 */
float
fuse_expr_cost(const struct model *m, nid root)
{
	static uint8_t *mark;
	static uint32_t cap;
	uint32_t sp = 0, j;
	int cnt = 0, i;
	nid *st, *v, n;
	size_t vcap = 64, stcap = 64;
	struct fgraph G;
	float c;

	if (cap < N.n) {
		uint32_t nc = N.n + N.n / 2 + 1024;
		mark = xrealloc(mark, nc);
		memset(mark + cap, 0, nc - cap);
		cap = nc;
	}
	st = xmalloc(stcap * sizeof *st);
	v = xmalloc(vcap * sizeof *v);
	st[sp++] = root;
	mark[root] = 1;
	while (sp) {
		n = st[--sp];
		if ((size_t)cnt == vcap)
			v = xrealloc(v, (vcap *= 2) * sizeof *v);
		v[cnt++] = n;
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF)
			continue;
		for (j = 0; j < (N.op[n] == O_TABLE ? N.b[n] : 3); j++) {
			nid a = N.op[n] == O_TABLE ? N.lst[N.a[n] + j] :
			    j == 0 ? N.a[n] : j == 1 ? N.b[n] : N.c[n];
			if (!a || mark[a])
				continue;
			mark[a] = 1;
			if (sp == stcap)
				st = xrealloc(st, (stcap *= 2) * sizeof *st);
			st[sp++] = a;
		}
	}
	for (i = 0; i < cnt; i++)
		mark[v[i]] = 0;
	qsort(v, (size_t)cnt, sizeof *v, cmp_nid);
	c = cover_list(m, v, cnt, &root, 1, 1, &G);
	fg_free(&G);
	free(st);
	free(v);
	return c;
}
