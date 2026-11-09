/*
 * Gate-level exact synthesis (the general form of the LOP3 reduction in
 * basis.md).
 *
 * A function of at most 6 input bits is a set of 64-bit truth tables, one
 * per output bit, plus a care mask: inputs outside the declared ranges or
 * the where-clause are don't-cares.  A basis is a set of gates declared as
 * expressions over 1-bit inputs; each gate's truth table and argument
 * symmetries are derived from its expression.  The search enumerates
 * straight-line programs by iterative deepening on cost, so the first
 * program found is minimal, and an exhausted bound is the certificate.
 *
 * Pruning (each keeps at least one minimal program):
 *   - a value equal on the care set to an existing value is dropped;
 *   - a value constant on the care set is dropped;
 *   - interchangeable arguments are ordered (a <= b);
 *   - a step independent of the previous step must have a larger key
 *     than it (adjacent independent steps commute);
 *   - cost + (unsatisfied outputs) * (cheapest gate) must fit the bound.
 */
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

#define MAXBASIS 32
static struct basis bases[MAXBASIS];
static int nbases;

struct basis *
basis_find(uint32_t name)
{
	int i;

	for (i = 0; i < nbases; i++)
		if (bases[i].name == name)
			return &bases[i];
	return NULL;
}

/* the i-th basis, NULL past the last (jsonmode.c's state) */
struct basis *
basis_nth(int i)
{
	return i >= 0 && i < nbases ? &bases[i] : NULL;
}

struct basis *
basis_new(uint32_t name)
{
	struct basis *b = basis_find(name);

	if (b == NULL) {
		if (nbases == MAXBASIS)
			fail("too many bases");
		b = &bases[nbases++];
	}
	memset(b, 0, sizeof *b);
	b->name = name;
	return b;
}

void
basis_finish_gate(struct basis *b, struct gate *g)
{
	struct plan p;
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];
	uint32_t m, i, j, n = 1u << g->arity;

	(void)b;
	memset(in_r, 0, sizeof in_r);
	for (m = 0; m < n; m++)
		for (i = 0; i < g->arity; i++)
			in_i[(size_t)i * CH + m] = (m >> i) & 1;
	plan_build(&p, &g->body, 1, 0);
	plan_run(&p, n, in_i, in_r);
	g->tt = 0;
	for (m = 0; m < n; m++)
		if (COLI(&p, p.root[0])[m] & 1)
			g->tt |= (uint8_t)(1u << m);
	plan_free(&p);
	g->symm = 0;
	for (i = 0; i < g->arity; i++)
		for (j = i + 1; j < g->arity; j++) {
			int same = 1;
			for (m = 0; m < n; m++) {
				uint32_t bi = (m >> i) & 1, bj = (m >> j) & 1;
				uint32_t s = (m & ~((1u << i) | (1u << j))) |
				    (bj << i) | (bi << j);
				if (((g->tt >> m) & 1) != ((g->tt >> s) & 1))
					same = 0;
			}
			if (same)
				g->symm |= (uint8_t)(1u << (i * 3 + j));
		}
}

/* ---- search ---- */

#define MAXV 250
#define MAXSTEP 244

uint64_t gates_budget = 1ULL << 27;   /* search nodes; "nodes N" sets it */

struct gs {
	const struct basis *bs;
	int nin, ntg;
	uint64_t full, care;
	uint64_t tg[MAXV];
	uint8_t active[MAXV];
	uint64_t val[MAXV];
	int nval;
	uint8_t sg[MAXSTEP], sa[MAXSTEP][3];
	uint32_t key[MAXSTEP];
	int nstep;
	int sat[MAXV];
	int nunsat;
	int bound, mincost;
	uint64_t nodes, limit;
	int aborted, found;
	uint8_t bsg[MAXSTEP], bsa[MAXSTEP][3];
	int bnstep;
};

static uint64_t
gapply(const struct gate *g, const uint64_t *x, uint64_t full)
{
	uint64_t r = 0, t;
	uint32_t m, i;

	switch (g->arity == 2 ? g->tt & 15 : 0x100) {
	case 0x8: return x[0] & x[1];
	case 0xe: return x[0] | x[1];
	case 0x6: return x[0] ^ x[1];
	case 0x7: return ~(x[0] & x[1]) & full;
	}
	for (m = 0; m < (1u << g->arity); m++) {
		if (!((g->tt >> m) & 1))
			continue;
		t = full;
		for (i = 0; i < g->arity; i++)
			t &= ((m >> i) & 1) ? x[i] : ~x[i];
		r |= t;
	}
	return r & full;
}

static int
symm(const struct gate *g, int i, int j)
{
	return (g->symm >> (i * 3 + j)) & 1;
}

static int
covers(const struct gs *s, uint64_t v, int t)
{
	return ((v ^ s->tg[t]) & s->care) == 0;
}

static void
record(struct gs *s)
{
	s->found = 1;
	s->bnstep = s->nstep;
	memcpy(s->bsg, s->sg, sizeof s->sg);
	memcpy(s->bsa, s->sa, sizeof s->sa);
}

static int dfs(struct gs *s, int cost);

static int
try_step(struct gs *s, int cost, int gi, const uint8_t *args)
{
	const struct gate *g = &s->bs->g[gi];
	uint64_t x[3], v, vc;
	uint32_t key;
	int i, t, ns = 0, uses_last = 0, r;

	key = (uint32_t)gi << 24 | (uint32_t)args[0] << 16 |
	    (uint32_t)(g->arity > 1 ? args[1] : 0) << 8 |
	    (uint32_t)(g->arity > 2 ? args[2] : 0);
	for (i = 0; i < g->arity; i++) {
		x[i] = s->val[args[i]];
		if (args[i] == s->nval - 1 && s->nstep > 0)
			uses_last = 1;
	}
	if (s->nstep > 0 && !uses_last && key <= s->key[s->nstep - 1])
		return 0;
	if (++s->nodes > s->limit) {
		s->aborted = 1;
		return 0;
	}
	v = gapply(g, x, s->full);
	vc = v & s->care;
	if (vc == 0 || vc == s->care)
		return 0;
	for (i = 0; i < s->nval; i++)
		if (((s->val[i] ^ v) & s->care) == 0)
			return 0;
	for (t = 0; t < s->ntg; t++)
		if (s->active[t] && !s->sat[t] && covers(s, v, t))
			ns++;
	if (cost + g->cost + (s->nunsat - ns) * s->mincost > s->bound)
		return 0;
	if (s->nval == MAXV || s->nstep == MAXSTEP)
		return 0;
	/* push */
	s->val[s->nval++] = v;
	s->sg[s->nstep] = (uint8_t)gi;
	memcpy(s->sa[s->nstep], args, 3);
	s->key[s->nstep++] = key;
	for (t = 0; t < s->ntg; t++)
		if (s->active[t] && covers(s, v, t) && !s->sat[t]++)
			s->nunsat--;
	r = dfs(s, cost + g->cost);
	for (t = 0; t < s->ntg; t++)
		if (s->active[t] && covers(s, v, t) && !--s->sat[t])
			s->nunsat++;
	s->nstep--;
	s->nval--;
	return r;
}

static int
dfs(struct gs *s, int cost)
{
	int gi, a, b, c;
	uint8_t args[3];

	if (s->nunsat == 0) {
		record(s);
		return 1;
	}
	if (cost + s->nunsat * s->mincost > s->bound)
		return 0;
	for (gi = 0; gi < s->bs->ng; gi++) {
		const struct gate *g = &s->bs->g[gi];
		if (cost + g->cost > s->bound)
			continue;
		for (a = 0; a < s->nval; a++) {
			args[0] = (uint8_t)a;
			if (g->arity == 1) {
				args[1] = args[2] = 0;
				if (try_step(s, cost, gi, args))
					return 1;
				if (s->aborted)
					return 0;
				continue;
			}
			for (b = 0; b < s->nval; b++) {
				if (b == a || (symm(g, 0, 1) && b < a))
					continue;
				args[1] = (uint8_t)b;
				if (g->arity == 2) {
					args[2] = 0;
					if (try_step(s, cost, gi, args))
						return 1;
					if (s->aborted)
						return 0;
					continue;
				}
				for (c = 0; c < s->nval; c++) {
					if (c == a || c == b)
						continue;
					if ((symm(g, 1, 2) && c < b) ||
					    (symm(g, 0, 2) && c < a))
						continue;
					args[2] = (uint8_t)c;
					if (try_step(s, cost, gi, args))
						return 1;
					if (s->aborted)
						return 0;
				}
			}
		}
	}
	return 0;
}

static void
reset_sat(struct gs *s)
{
	int t, i;

	s->nunsat = 0;
	for (t = 0; t < s->ntg; t++) {
		s->sat[t] = 0;
		for (i = 0; i < s->nval; i++)
			if (covers(s, s->val[i], t))
				s->sat[t]++;
		if (s->active[t] && !s->sat[t])
			s->nunsat++;
	}
}

/* iterative deepening from the current values; returns the cost or -1 */
static int
deepen(struct gs *s, int from, int maxcost, uint64_t budget)
{
	int b;

	s->found = 0;
	s->aborted = 0;
	s->limit = s->nodes + budget;   /* one budget for the whole deepening */
	for (b = from; b <= maxcost; b++) {
		s->bound = b;
		reset_sat(s);
		if (dfs(s, 0))
			return b;
		if (s->aborted)
			return -1;
	}
	return -1;
}


/*
 * Fallback when the exact search runs out of budget: Shannon expansion.
 * Restricting the care set to x = 1 and to x = 0 gives two smaller
 * problems; their results are joined by an exact search over just
 * {x, v1, v0}, so the join is the cheapest one the basis allows.  Steps
 * are shared: any value already computed that fits is reused.
 */
static int
combine(struct gs *s, const int *idx, int n, uint64_t tgt, uint64_t care,
    uint64_t budget, int maxc)
{
	struct gs *m = xcalloc(1, sizeof *m);
	int i, k, r = -1, map[MAXV];

	m->bs = s->bs;
	m->nin = n;
	m->full = s->full;
	m->care = care;
	for (i = 0; i < n; i++)
		m->val[m->nval++] = s->val[idx[i]];
	m->tg[0] = tgt;
	m->active[0] = 1;
	m->ntg = 1;
	m->mincost = s->mincost;
	if (deepen(m, 0, maxc, budget) >= 0) {
		for (i = 0; i < n; i++)
			map[i] = idx[i];
		for (i = 0; i < m->bnstep; i++) {
			const struct gate *g = &s->bs->g[m->bsg[i]];
			uint64_t x[3];
			uint8_t a[3] = {0, 0, 0};
			if (s->nval >= MAXV || s->bnstep >= MAXSTEP) {
				free(m);
				return -1;
			}
			uint64_t v;
			int dup = -1, j;
			for (k = 0; k < g->arity; k++) {
				a[k] = (uint8_t)map[m->bsa[i][k]];
				x[k] = s->val[a[k]];
			}
			v = gapply(g, x, s->full);
			for (j = 0; j < s->nval; j++)
				if (s->val[j] == v) {
					dup = j;
					break;
				}
			if (dup >= 0) {
				map[n + i] = dup;
				continue;
			}
			s->bsg[s->bnstep] = m->bsg[i];
			memcpy(s->bsa[s->bnstep], a, 3);
			s->bnstep++;
			map[n + i] = s->nval;
			s->val[s->nval++] = v;
		}
		for (i = s->nval; i-- > 0;)
			if (((s->val[i] ^ tgt) & care) == 0) {
				r = i;
				break;
			}
	}
	free(m);
	return r;
}

/* index of a value equal to tgt on care; -2 if tgt is constant there */
static int
shannon(struct gs *s, uint64_t tgt, uint64_t care)
{
	int i, n = 0, idx[MAXV], v1, v0, split = -1, r;

	tgt &= care;
	for (i = s->nval; i-- > 0;)
		if (((s->val[i] ^ tgt) & care) == 0)
			return i;
	if (tgt == 0 || tgt == care)
		return -2;
	for (i = 0; i < s->nin; i++)
		idx[i] = i;
	if ((r = combine(s, idx, s->nin, tgt, care, 1u << 18, 5)) >= 0)
		return r;
	for (i = s->nin; i-- > 0;) {
		uint64_t h1 = care & s->val[i], h0 = care & ~s->val[i];
		if (h1 && h0 && (tgt & h1) != (h1 & ~0ULL ? tgt & h1 : 0)) {
			split = i;
			break;
		}
		if (h1 && h0 && split < 0)
			split = i;
	}
	if (split < 0)
		return -1;
	v1 = shannon(s, tgt, care & s->val[split]);
	v0 = shannon(s, tgt, care & ~s->val[split]);
	if (v1 == -1 || v0 == -1)
		return -1;
	idx[n++] = split;
	if (v1 >= 0)
		idx[n++] = v1;
	if (v0 >= 0 && v0 != v1)
		idx[n++] = v0;
	return combine(s, idx, n, tgt, care, 1u << 22, 8);
}

static int
gate_cost_sum(const struct gs *s, int nstep, const uint8_t *sg)
{
	int i, c = 0;

	for (i = 0; i < nstep; i++)
		c += s->bs->g[sg[i]].cost;
	return c;
}

static void
print_program(const struct gs *s, const struct func *f, int nstep,
    const uint8_t *sg, uint8_t (*sa)[3], const int *outv, const int *outk,
    int nout, int width, const uint64_t *constbits, int emit)
{
	char **names = xcalloc(MAXV, sizeof *names);
	struct sbuf b = {0};
	int i, j, k, v = 0;

	for (j = 0; j < f->np; j++) {
		int w = TYW(f->p[j].ty);
		for (k = 0; k < w; k++) {
			char buf[64];
			if (w == 1)
				snprintf(buf, sizeof buf, "%s", sym_str(f->p[j].name));
			else
				snprintf(buf, sizeof buf, "%s%d",
				    sym_str(f->p[j].name), k);
			names[v++] = strdup(buf);
		}
	}
	for (i = 0; i < nstep; i++) {
		char buf[32];
		snprintf(buf, sizeof buf, "t%d", i);
		names[s->nin + i] = strdup(buf);
	}
	if (!emit)
		rep_list("program");
	for (i = 0; i < nstep; i++) {
		const struct gate *g = &s->bs->g[sg[i]];
		const char *vn[3];
		struct pctx pc;
		memset(&pc, 0, sizeof pc);
		for (k = 0; k < g->arity; k++)
			vn[k] = names[sa[i][k]];
		pc.mode = emit ? PM_CBITS : PM_UCAS;
		pc.vn = vn;
		sb_clear(&b);
		print_node(&b, g->body, &pc);
		if (emit)
			printf("\tuint32_t t%d = %s;\n", i, sb_str(&b));
		else {
			printf("  t%d = %s(", i, sym_str(g->name));
			for (k = 0; k < g->arity; k++)
				printf("%s%s", k ? ", " : "", vn[k]);
			printf(")");
			if (g->arity >= 2 || strcmp(sb_str(&b), "") != 0)
				printf("%*s= %s", 26 - (int)strlen(sym_str(g->name)) -
				    g->arity * 4 > 2 ? 26 - (int)strlen(sym_str(
				    g->name)) - g->arity * 4 : 2, "", sb_str(&b));
			printf("\n");
			rep_obj(NULL);
			rep_strf("dst", "t%d", i);
			rep_str("gate", sym_str(g->name));
			rep_list("args");
			for (k = 0; k < g->arity; k++)
				rep_str(NULL, vn[k]);
			rep_end();
			rep_str("expr", sb_str(&b));
			rep_end();
		}
	}
	if (emit) {
		printf("\treturn ");
		for (k = 0, i = 0; k < width; k++) {
			int o;
			for (o = 0; o < nout; o++)
				if (outk[o] == k)
					break;
			if (o < nout) {
				printf("%s(%s & 1) << %d", i++ ? " | " : "",
				    names[outv[o]], k);
			} else if ((constbits[0] >> k) & 1)
				printf("%s1u << %d", i++ ? " | " : "", k);
		}
		if (!i)
			printf("0");
		printf(";\n");
	} else {
		rep_end();
		rep_obj("outputs");
		for (k = width; k-- > 0;) {
			int o;
			char rn[16];
			for (o = 0; o < nout; o++)
				if (outk[o] == k)
					break;
			snprintf(rn, sizeof rn, "r%d", k);
			if (o < nout)
				rep_str(rn, names[outv[o]]);
			else
				rep_strf(rn, "%d", (int)((constbits[0] >> k) & 1));
		}
		rep_end();
		for (k = width; k-- > 0;) {
			int o;
			for (o = 0; o < nout; o++)
				if (outk[o] == k)
					break;
			if (o < nout)
				printf("  r%d = %s\n", k, names[outv[o]]);
			else
				printf("  r%d = %d\n", k,
				    (int)((constbits[0] >> k) & 1));
		}
	}
	for (i = 0; i < MAXV; i++)
		free(names[i]);
	free(names);
	free(b.p);
}

void
cmd_gates(struct func *f, struct basis *bs, int maxcost, int emit)
{
	struct gs *s = xcalloc(1, sizeof *s);
	struct plan p;
	nid roots[2];
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];
	uint32_t m, npts, j, k, i;
	int nin = 0, width, outk[64], nout = 0, t, cost, gi;
	uint64_t tbit[64] = {0}, constbits = 0, care = 0;
	ty_t rt = N.ty[f->body];
	int64_t vmin = 0, vmax = 0;
	int any = 0, uniq[64];

	for (j = 0; j < (uint32_t)f->np; j++) {
		ty_t pt = f->p[j].ty;
		if (!TY_SIZED(pt))
			fail("gates needs bit-sized parameters (uN, iN, bool); "
			    "%s is %s", sym_str(f->p[j].name), ty_str(pt));
		nin += TYW(pt);
	}
	if (nin > 6)
		fail("gates handles up to 6 input bits (%d here); exact "
		    "synthesis beyond that needs SAT, not implemented yet", nin);
	if (!TY_ISINT(rt))
		fail("gates needs an integer or boolean result");
	if (bs->ng == 0)
		fail("basis %s has no gates", sym_str(bs->name));
	npts = 1u << nin;
	s->bs = bs;
	s->nin = nin;
	s->full = npts == 64 ? ~0ULL : (1ULL << npts) - 1;

	/* evaluate every input combination */
	memset(in_r, 0, sizeof in_r);
	for (m = 0; m < npts; m++) {
		int at = 0;
		for (j = 0; j < (uint32_t)f->np; j++) {
			int w = TYW(f->p[j].ty);
			uint64_t bitsv = (m >> at) & ((1u << w) - 1);
			in_i[(size_t)j * CH + m] = int_norm(f->p[j].ty, (int64_t)bitsv);
			at += w;
		}
	}
	roots[0] = f->body;
	roots[1] = f->where;
	plan_build(&p, roots, 2, 0);
	plan_run(&p, npts, in_i, in_r);
	for (m = 0; m < npts; m++) {
		int ok = 1;
		int64_t v;
		for (j = 0; j < (uint32_t)f->np; j++) {
			int64_t x = in_i[(size_t)j * CH + m];
			if (f->p[j].ranged && (x < f->p[j].lo || x > f->p[j].hi))
				ok = 0;
		}
		if (f->where && (COLF(&p, p.root[1])[m] & F_UNDEF ||
		    !COLI(&p, p.root[1])[m]))
			ok = 0;
		if (COLF(&p, p.root[0])[m] & F_UNDEF)
			ok = 0;
		if (!ok)
			continue;
		care |= 1ULL << m;
		v = COLI(&p, p.root[0])[m];
		if (!any || v < vmin)
			vmin = v;
		if (!any || v > vmax)
			vmax = v;
		any = 1;
	}
	if (!any)
		fail("the function is undefined on every input");
	if (TY_SIZED(rt))
		width = TYW(rt);
	else if (vmin >= 0)
		width = bits_for((uint64_t)vmax);
	else
		width = 1 + bits_for((uint64_t)(vmin < 0 ? ~vmin : vmin) >
		    (uint64_t)vmax ? (uint64_t)~vmin : (uint64_t)vmax);
	if (width > 64)
		width = 64;
	for (m = 0; m < npts; m++)
		if ((care >> m) & 1) {
			uint64_t v = (uint64_t)COLI(&p, p.root[0])[m];
			for (k = 0; k < (uint32_t)width; k++)
				if ((v >> k) & 1)
					tbit[k] |= 1ULL << m;
		}
	plan_free(&p);
	s->care = care;

	/* inputs */
	for (i = 0; i < (uint32_t)nin; i++) {
		uint64_t proj = 0;
		for (m = 0; m < npts; m++)
			if ((m >> i) & 1)
				proj |= 1ULL << m;
		s->val[s->nval++] = proj;
	}
	/* outputs: constants drop out, equal outputs share a target */
	for (k = 0; k < (uint32_t)width; k++) {
		uint64_t tb = tbit[k] & care;
		if (tb == 0)
			continue;
		if (tb == care) {
			constbits |= 1ULL << k;
			continue;
		}
		for (t = 0; t < s->ntg; t++)
			if (((s->tg[t] ^ tb) & care) == 0)
				break;
		if (t == s->ntg) {
			s->tg[s->ntg] = tb;
			s->active[s->ntg++] = 1;
		}
		outk[nout] = (int)k;
		uniq[nout++] = t;
	}
	s->mincost = 255;
	for (gi = 0; gi < bs->ng; gi++)
		if (bs->g[gi].cost < s->mincost)
			s->mincost = bs->g[gi].cost;

	printf("gates for %s: %d input bit%s -> %d output bit%s", f->anon ?
	    node_str(f->body) : sym_str(f->name), nin, nin == 1 ? "" : "s",
	    width, width == 1 ? "" : "s");
	if (constbits || nout < width)
		printf(" (%d constant)", width - nout);
	printf(", basis %s\n", sym_str(bs->name));
	rep_begin("gates");
	rep_str("basis", sym_str(bs->name));
	rep_int("input_bits", nin);
	rep_int("output_bits", width);
	rep_int("constant_bits", width - nout);
	if (care != s->full)
		printf("  %d of %u input combinations are don't-cares (outside "
		    "the domain)\n", (int)(npts - (uint32_t)__builtin_popcountll(care)),
		    npts);

	cost = deepen(s, 0, maxcost, gates_budget);
	if (cost >= 0) {
		printf("  minimal: cost %d, %d step%s (no program of cost <= %d "
		    "exists; %llu candidates searched)\n", cost, s->bnstep,
		    s->bnstep == 1 ? "" : "s", cost - 1,
		    (unsigned long long)s->nodes);
		rep_bool("minimal", 1);
		rep_int("cost", cost);
		rep_int("steps", s->bnstep);
		rep_int("candidates", (long long)s->nodes);
		rep_proof("exhaustive", "iterative deepening on cost: no program "
		    "of a lower cost exists");
	} else {
		/* upper bound by Shannon expansion, one output at a time */
		int o, total;
		printf("  exact search stopped (%s at cost %d); Shannon "
		    "decomposition instead:\n", s->aborted ? "node budget" :
		    "cost limit", s->bound);
		s->nval = nin;
		s->bnstep = 0;
		for (t = 0; t < s->ntg; t++) {
			if (shannon(s, s->tg[t], care) == -1) {
				printf("  no program found for output %d (basis not "
				    "complete, or more than %d steps)\n", t, MAXSTEP);
				free(s);
				return;
			}
		}
		/* drop steps no output uses */
		{
			uint8_t live[MAXV] = {0};
			int remap[MAXV], ns = 0, at;
			for (t = 0; t < s->ntg; t++)
				for (at = s->nval; at-- > 0;)
					if (((s->val[at] ^ s->tg[t]) & care) == 0) {
						live[at] = 1;
						break;
					}
			for (at = s->bnstep; at-- > 0;) {
				const struct gate *g = &bs->g[s->bsg[at]];
				if (!live[nin + at])
					continue;
				for (k = 0; k < g->arity; k++)
					live[s->bsa[at][k]] = 1;
			}
			for (at = 0; at < nin; at++)
				remap[at] = at;
			for (at = 0; at < s->bnstep; at++) {
				const struct gate *g = &bs->g[s->bsg[at]];
				if (!live[nin + at])
					continue;
				s->bsg[ns] = s->bsg[at];
				for (k = 0; k < g->arity; k++)
					s->bsa[ns][k] = (uint8_t)remap[s->bsa[at][k]];
				s->val[nin + ns] = s->val[nin + at];
				remap[nin + at] = nin + ns;
				ns++;
			}
			s->bnstep = ns;
			s->nval = nin + ns;
		}
		total = gate_cost_sum(s, s->bnstep, s->bsg);
		printf("  cost %d, %d steps (an upper bound, not proved minimal)\n",
		    total, s->bnstep);
		rep_bool("minimal", 0);
		rep_int("cost", total);
		rep_int("steps", s->bnstep);
		for (o = 0; o < s->ntg; o++)
			s->active[o] = 1;
	}
	/* which value is each output */
	{
		int vout[64];
		uint64_t v[MAXV];
		int nv = 0;
		for (i = 0; i < (uint32_t)nin; i++)
			v[nv++] = s->val[i];
		for (i = 0; i < (uint32_t)s->bnstep; i++) {
			const struct gate *g = &bs->g[s->bsg[i]];
			uint64_t x[3];
			for (k = 0; k < g->arity; k++)
				x[k] = v[s->bsa[i][k]];
			v[nv++] = gapply(g, x, s->full);
		}
		for (t = 0; t < nout; t++) {
			vout[t] = -1;
			for (i = (uint32_t)nv; i-- > 0;)
				if (((v[i] ^ s->tg[uniq[t]]) & care) == 0) {
					vout[t] = (int)i;
					break;
				}
		}
		for (t = 0; t < nout; t++)
			if (vout[t] < 0) {
				printf("  internal error: output bit %d not produced\n",
				    outk[t]);
				free(s);
				return;
			}
		printf("  checked: the program reproduces every output bit on all "
		    "%d care inputs\n", __builtin_popcountll(care));
		rep_check("truth_table", __builtin_popcountll(care), -1, 1,
		    "every output bit on every care input");
		print_program(s, f, s->bnstep, s->bsg, s->bsa, vout, outk, nout,
		    width, &constbits, 0);
		if (emit) {
			printf("\n/* %s, basis %s: bits are independent lanes, so the "
			    "same code also\n   runs bit-sliced on whole words */\n",
			    f->anon ? "_" : sym_str(f->name), sym_str(bs->name));
			printf("static inline uint32_t\n%s_gates(", f->anon ? "f" :
			    sym_str(f->name));
			for (j = 0; j < (uint32_t)f->np; j++)
				printf("%suint32_t %s", j ? ", " : "",
				    sym_str(f->p[j].name));
			printf(")\n{\n");
			for (j = 0; j < (uint32_t)f->np; j++) {
				int w = TYW(f->p[j].ty);
				if (w == 1)
					continue;
				for (k = 0; k < (uint32_t)w; k++)
					printf("\tuint32_t %s%u = %s >> %u & 1;\n",
					    sym_str(f->p[j].name), k,
					    sym_str(f->p[j].name), k);
			}
			print_program(s, f, s->bnstep, s->bsg, s->bsa, vout, outk,
			    nout, width, &constbits, 1);
			printf("}\n");
		}
	}
	free(s);
}
