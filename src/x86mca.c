/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * mca: what a block of x86 instructions costs on a µarch, in the manner
 * of llvm-mca, simplified.
 *
 * A function is lowered to scalar x86 instruction classes (one or two per
 * operation, as a compiler would select them: a compare feeding ?: is a
 * cmp + cmov, x / 7 is a mulx and a shift, x * 8 a shift ...), its first
 * loop's body becoming a block that repeats: index add, compare+branch,
 * and the states carried from one iteration to the next.
 *
 * The estimate is a cycle-level simulation: in-order dispatch of up to
 * `width` fused uops a cycle into a reorder buffer of `rob` entries,
 * out-of-order issue of each instruction's uops (oldest first, within
 * a scheduler window) to a free port of its class once its operands are
 * ready, completion after its latency, in-order retirement.  A port takes
 * one uop a cycle, or is held for rthroughput * ports / uops cycles when
 * the class is slower than that; a class whose throughput is not
 * explained by its uops (a divider, a gather) holds its first port for
 * the whole rthroughput.  Loops run 160 iterations and report the
 * steady-state cycles per iteration (iterations 40..160).  Next to it,
 * three bounds explain the result: the latency bound (the critical path,
 * or for a loop the loop-carried recurrence per iteration), the port
 * bound (the busiest port after spreading every class's port-cycles over
 * its ports, least loaded first) and the front-end bound (uops / width).
 *
 * Not modelled: caches (every load hits L1), branch mispredictions (the
 * loop branch is predicted), memory dependences, macro-fusion beyond
 * cmp+jcc, move elimination, the uop cache and LSD, partial-register and
 * flag merges.
 */
#include "x86.h"
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void
mblock_add(struct mblock *b, int cls, int vw, const char *text,
    int32_t d0, int32_t d1, int32_t d2)
{
	struct mins *m;

	if (b->n == b->cap) {
		b->cap = newcap(b->cap, b->n + 1);
		b->in = xrealloc(b->in, b->cap * sizeof *b->in);
	}
	m = &b->in[b->n++];
	memset(m, 0, sizeof *m);
	m->cls = (uint8_t)cls;
	m->vw = (uint8_t)vw;
	m->dep[0] = d0;
	m->dep[1] = d1;
	m->dep[2] = d2;
	snprintf(m->text, sizeof m->text, "%s", text ? text : xc_name[cls]);
}

static int
popc16(uint16_t m)
{
	int n = 0;

	for (; m; m &= m - 1)
		n++;
	return n;
}

static const struct xcost *
cost_of(int ua, const struct mins *m)
{
	const struct xcost *c = xcost_get(ua, m->cls, m->vw);
	static struct xcost none = {1, 1, 1, 1, 1, 'E'};

	if (!c && m->vw > 0)
		c = xcost_get(ua, m->cls, m->vw - 1);
	if (!c)
		c = xcost_get(ua, m->cls, VW_X);
	if (!c) {
		none.ports = 1;
		return &none;
	}
	return c;
}

/* a unit that is busy for the whole rthroughput (divider, gather) */
static int
unit_bound(const struct xcost *c)
{
	return c->rtp >= 2 && c->rtp * (float)popc16(c->ports) >
	    1.5f * (float)c->uops;
}

/* ---- the analytic bounds ---- */

static void
pressure(const struct mblock *b, int ua, double *pr)
{
	uint32_t i;
	int pass, p;

	memset(pr, 0, MAXPORT * sizeof *pr);
	/* single-port loads first, then wider sets, least loaded first */
	for (pass = 1; pass <= MAXPORT; pass++)
		for (i = 0; i < b->n; i++) {
			const struct xcost *c = cost_of(ua, &b->in[i]);
			int np = popc16(c->ports);
			double left;
			if (np != pass || b->in[i].cls == XC_NONE)
				continue;
			if (unit_bound(c)) {
				for (p = 0; p < MAXPORT; p++)
					if ((c->ports >> p) & 1) {
						pr[p] += c->rtp;
						break;
					}
				left = c->uops - 1;
			} else
				left = c->rtp * np > c->uops ? c->rtp * np : c->uops;
			/* water-filling in steps of 1/16 */
			while (left > 1e-9) {
				double step = left < 0.0625 ? left : 0.0625;
				int best = -1;
				for (p = 0; p < MAXPORT; p++)
					if (((c->ports >> p) & 1) &&
					    (best < 0 || pr[p] < pr[best]))
						best = p;
				if (best < 0)
					break;
				pr[best] += step;
				left -= step;
			}
		}
}

#define LITERS 160
#define LFROM 40

/* critical path (block) or recurrence per iteration (loop) */
static double
latency_bound(const struct mblock *b, int ua)
{
	uint32_t n = b->n, it, i, iters = b->loop ? LITERS : 1;
	double *t = xcalloc((size_t)n * iters, sizeof *t), best = 0, r;
	int k;

	for (it = 0; it < iters; it++)
		for (i = 0; i < n; i++) {
			double s = 0;
			for (k = 0; k < MCA_MAXDEP; k++) {
				int32_t d = b->in[i].dep[k];
				double v;
				if (d < 0)
					continue;
				if (b->in[i].carried[k]) {
					if (it == 0)
						continue;
					v = t[(size_t)(it - 1) * n + d];
				} else
					v = t[(size_t)it * n + d];
				if (v > s)
					s = v;
			}
			t[(size_t)it * n + i] = s + cost_of(ua, &b->in[i])->lat;
		}
	if (!b->loop) {
		for (i = 0; i < n; i++)
			if (t[i] > best)
				best = t[i];
		free(t);
		return best;
	}
	{
		double a = 0, z = 0;
		for (i = 0; i < n; i++) {
			if (t[(size_t)(LFROM - 1) * n + i] > a)
				a = t[(size_t)(LFROM - 1) * n + i];
			if (t[(size_t)(iters - 1) * n + i] > z)
				z = t[(size_t)(iters - 1) * n + i];
		}
		r = (z - a) / (double)(iters - LFROM);
	}
	free(t);
	return r;
}

/* ---- the simulation ---- */

void
mca_run(const struct mblock *b, int ua, struct mres *r)
{
	const struct uarch *u = &uarchs[ua];
	uint32_t n = b->n, iters = b->loop ? LITERS : 1, tot = n * iters;
	double *done, pfree[MAXPORT], pload[MAXPORT], cyc = 0;
	uint8_t *left, *mainleft;
	uint32_t ret = 0, disp = 0, first = 0, i, k;
	double t_from = 0, t_to = 0, fe_uops = 0;
	int p;

	memset(r, 0, sizeof *r);
	if (n == 0)
		return;
	x86_init();
	done = xcalloc(tot, sizeof *done);
	left = xcalloc(tot, 1);
	mainleft = xcalloc(tot, 1);
	for (i = 0; i < tot; i++) {
		const struct xcost *c = cost_of(ua, &b->in[i % n]);
		done[i] = HUGE_VAL;
		left[i] = c->uops ? c->uops : 1;
		mainleft[i] = (uint8_t)unit_bound(c);
		if (b->in[i % n].cls == XC_NONE)
			left[i] = 0, done[i] = 0;
	}
	for (i = 0; i < n; i++)
		fe_uops += b->in[i].cls == XC_NONE ? 0 : cost_of(ua, &b->in[i])->uops;
	for (p = 0; p < MAXPORT; p++)
		pfree[p] = pload[p] = 0;
	while (ret < tot && cyc < (double)tot * 400 + 1000) {
		int budget = u->width, inwin = 0;
		/* retire in order */
		while (ret < disp && done[ret] <= cyc)
			ret++;
		/* dispatch in order */
		while (disp < tot && disp - ret < (uint32_t)u->rob) {
			const struct mins *m = &b->in[disp % n];
			int uo = m->cls == XC_NONE ? 0 : cost_of(ua, m)->uops;
			if (uo > budget && budget < u->width)
				break;
			budget -= uo;
			disp++;
			if (budget <= 0)
				break;
		}
		/* issue, oldest first */
		while (first < disp && left[first] == 0)
			first++;
		for (i = first; i < disp && inwin < u->sched; i++) {
			const struct mins *m = &b->in[i % n];
			const struct xcost *c;
			uint32_t it = i / n;
			int ready = 1, np;
			if (left[i] == 0)
				continue;
			inwin++;
			for (k = 0; k < MCA_MAXDEP && ready; k++) {
				int32_t d = m->dep[k];
				uint32_t gi;
				if (d < 0)
					continue;
				if (m->carried[k]) {
					if (it == 0)
						continue;
					gi = (it - 1) * n + (uint32_t)d;
				} else
					gi = it * n + (uint32_t)d;
				if (gi >= disp || done[gi] > cyc)
					ready = 0;
			}
			if (!ready)
				continue;
			c = cost_of(ua, m);
			np = popc16(c->ports);
			while (left[i] > 0) {
				int best = -1;
				double occ;
				for (p = 0; p < MAXPORT; p++)
					if (((c->ports >> p) & 1) && pfree[p] <= cyc &&
					    (best < 0 || pload[p] < pload[best]))
						best = p;
				if (best < 0)
					break;
				if (mainleft[i]) {
					/* the unit's own uop on the first port */
					for (p = 0; p < MAXPORT; p++)
						if ((c->ports >> p) & 1)
							break;
					if (pfree[p] > cyc)
						break;
					best = p;
					occ = c->rtp;
					mainleft[i] = 0;
				} else if (unit_bound(c))
					occ = 1;
				else {
					occ = c->rtp * np / (c->uops ? c->uops : 1);
					if (occ < 1)
						occ = 1;
				}
				pfree[best] = cyc + occ;
				pload[best] += occ;
				left[i]--;
			}
			if (left[i] == 0)
				done[i] = cyc + (c->lat > 0 ? c->lat : 1);
		}
		cyc += 1;
	}
	if (b->loop) {
		for (i = (LFROM - 1) * n; i < LFROM * n; i++)
			if (done[i] > t_from)
				t_from = done[i];
		for (i = (iters - 1) * n; i < tot; i++)
			if (done[i] > t_to)
				t_to = done[i];
		r->cycles = (t_to - t_from) / (double)(iters - LFROM);
	} else {
		for (i = 0; i < tot; i++)
			if (done[i] > r->cycles && done[i] < HUGE_VAL)
				r->cycles = done[i];
	}
	free(done);
	free(left);
	free(mainleft);
	r->uops = fe_uops;
	r->fe_bound = fe_uops / u->width;
	pressure(b, ua, r->pressure);
	r->busiest = 0;
	for (p = 0; p < MAXPORT; p++)
		if (r->pressure[p] > r->pressure[r->busiest])
			r->busiest = p;
	r->port_bound = r->pressure[r->busiest];
	r->lat_bound = latency_bound(b, ua);
	if (r->lat_bound >= r->port_bound && r->lat_bound >= r->fe_bound)
		r->limit = b->loop ? "the loop-carried recurrence" :
		    "the critical path";
	else if (r->port_bound >= r->fe_bound)
		r->limit = "port pressure";
	else
		r->limit = "the front end (dispatch width)";
}

void
mca_print(const struct mblock *b, int ua, const struct mres *r,
    const char *title)
{
	const struct uarch *u = &uarchs[ua];
	struct sbuf pb = {0};
	uint32_t i;
	int k, p;

	printf("  %s: %u instruction%s, %g uops\n", title, b->n,
	    b->n == 1 ? "" : "s", r->uops);
	printf("     #  %-30s %4s %5s %4s  %-9s %s\n", "instruction", "lat",
	    "rtp", "uops", "ports", "inputs");
	rep_obj(NULL);
	rep_str("block", title);
	rep_bool("loop", b->loop);
	rep_list("instructions");
	for (i = 0; i < b->n; i++) {
		const struct mins *m = &b->in[i];
		const struct xcost *c = xcost_get(ua, m->cls, m->vw);
		char deps[64] = "";
		size_t dl = 0;
		sb_clear(&pb);
		if (c)
			xports_text(&pb, ua, c->ports);
		for (k = 0; k < MCA_MAXDEP; k++)
			if (m->dep[k] >= 0 && dl < sizeof deps - 12)
				dl += (size_t)snprintf(deps + dl, sizeof deps - dl,
				    "%s%d%s", dl ? " " : "", m->dep[k],
				    m->carried[k] ? "'" : "");
		if (c)
			printf("    %2u  %-30s %4g %5g %4d  %-9s %s\n", i, m->text,
			    (double)c->lat, (double)c->rtp, c->uops, sb_str(&pb),
			    deps);
		else
			printf("    %2u  %-30s (not on %s)\n", i, m->text, u->name);
		rep_obj(NULL);
		rep_str("text", m->text);
		rep_str("class", xc_name[m->cls]);
		if (c) {
			rep_num("latency", c->lat);
			rep_num("rthroughput", c->rtp);
			rep_int("uops", c->uops);
			rep_str("ports", sb_str(&pb));
		}
		rep_str("inputs", deps);
		rep_end();
	}
	rep_end();
	printf("    (inputs: instruction numbers; ' = from the previous "
	    "iteration)\n");
	printf("  port pressure%s:", b->loop ? " per iteration" : "");
	rep_obj("pressure");
	for (p = 0; p < u->nport; p++)
		if (r->pressure[p] > 0.005) {
			printf(" %s %.2f", u->port[p], r->pressure[p]);
			rep_num(u->port[p], floor(r->pressure[p] * 100 + 0.5) / 100);
		}
	rep_end();
	printf("\n");
	printf("  bounds: %s %.2f, ports %.2f (%s), front end %.2f (%g uops / "
	    "%d)\n", b->loop ? "recurrence" : "critical path", r->lat_bound,
	    r->port_bound, u->port[r->busiest], r->fe_bound, r->uops, u->width);
	printf("  simulated: %.2f cycles%s, limited by %s\n", r->cycles,
	    b->loop ? " per iteration (steady state over 120 iterations)" : "",
	    r->limit);
	rep_num(b->loop ? "cycles_per_iteration" : "cycles",
	    floor(r->cycles * 100 + 0.5) / 100);
	rep_num("latency_bound", floor(r->lat_bound * 100 + 0.5) / 100);
	rep_num("port_bound", floor(r->port_bound * 100 + 0.5) / 100);
	rep_str("busiest_port", u->port[r->busiest]);
	rep_num("frontend_bound", floor(r->fe_bound * 100 + 0.5) / 100);
	rep_str("limited_by", r->limit);
	rep_end();
	free(pb.p);
}

/* ---- lowering a function's nodes to instruction classes ---- */

struct low {
	int ua;
	struct mblock *b;
	nid *key;		/* memo: node -> code */
	int32_t *val;
	uint32_t lim;
	uint8_t *invar;		/* loop: node is loop-invariant (hoisted) */
	uint64_t bound;		/* loop: slots bound by the loop */
	int sslot[MAXST], nst, islot;
	int nested;
	struct low *outer;	/* where hoisted nodes go */
	int32_t *flag;		/* compare nodes: the cmp (flags) */
};

#define PH_STATE (-100)	/* placeholder: state k is PH_STATE - k */
#define PH_INDEX (-200)

static int has_feat(int ua, int f)
{
	return (uarchs[ua].feat & XB(f)) != 0;
}

static int32_t lower(struct low *L, nid n);

static int32_t
emit(struct low *L, int cls, nid n, int32_t d0, int32_t d1, int32_t d2)
{
	char txt[48];
	struct sbuf b = {0};

	print_node(&b, n, NULL);
	cap_text(&b, 0, 30);
	snprintf(txt, sizeof txt, "%-9s %s", xc_name[cls], sb_str(&b));
	free(b.p);
	mblock_add(L->b, cls, VW_X, txt, d0, d1, d2);
	return (int32_t)(L->b->n - 1);
}

static int
is_cmp(nid n)
{
	return (opinfo[N.op[n]].fl & OF_CMP) != 0;
}

static int32_t
lower1(struct low *L, nid n)
{
	int op = N.op[n];
	ty_t t = N.ty[n], ta = N.a[n] ? N.ty[N.a[n]] : 0;
	int fp = TY_FP(t), w = TY_SIZED(t) ? TYW(t) : 64;
	int32_t a = -1, b = -1, c = -1, x;

	switch (op) {
	case O_CONST: case O_UNDEF: case O_TABLE:
		return -1;
	case O_VAR: {
		int s = (int)N.ki[n], k;
		if (L->bound && ((L->bound >> s) & 1)) {
			if (s == L->islot)
				return PH_INDEX;
			for (k = 0; k < L->nst; k++)
				if (L->sslot[k] == s)
					return PH_STATE - k;
		}
		return -1;
	}
	case O_LOOP:
		L->nested++;
		return -1;
	case O_INDEX:
		b = lower(L, N.b[n]);
		return emit(L, XC_LOAD, n, b, -1, -1);
	case O_SEL:
		if (is_cmp(N.a[n])) {
			lower(L, N.a[n]);
			a = L->flag[N.a[n]] >= 0 ? L->flag[N.a[n]] : -1;
		} else {
			a = lower(L, N.a[n]);
			a = emit(L, XC_ALU, N.a[n], a, -1, -1);   /* test */
		}
		b = lower(L, N.b[n]);
		c = lower(L, N.c[n]);
		return emit(L, fp ? XC_VBLENDV : XC_CMOV, n, a, b, c);
	}
	if (N.a[n])
		a = lower(L, N.a[n]);
	if (N.b[n] && opinfo[op].arity >= 2)
		b = lower(L, N.b[n]);
	if (N.c[n] && opinfo[op].arity >= 3)
		c = lower(L, N.c[n]);
	/* a compare used as a value (not by ?:): setcc after the cmp */
	if (a >= 0 && N.a[n] && is_cmp(N.a[n]) && L->flag[N.a[n]] == a)
		a = emit(L, XC_SETCC, N.a[n], a, -1, -1);
	if (b >= 0 && N.b[n] && is_cmp(N.b[n]) && L->flag[N.b[n]] == b)
		b = emit(L, XC_SETCC, N.b[n], b, -1, -1);
	switch (op) {
	case O_ADD: case O_SUB:
		return emit(L, fp ? XC_VFADD : XC_ALU, n, a, b, -1);
	case O_NEG:
		return emit(L, fp ? XC_VALU : XC_ALU, n, a, -1, -1);
	case O_AND: case O_OR: case O_XOR: case O_NOT: case O_LNOT:
	case O_LAND: case O_LOR:
		return emit(L, XC_ALU, n, a, b, -1);
	case O_MUL:
		if (fp)
			return emit(L, XC_VFMUL, n, a, b, -1);
		if (is_const(N.a[n]) && N.ki[N.a[n]] > 0 &&
		    !(N.ki[N.a[n]] & (N.ki[N.a[n]] - 1)))
			return emit(L, XC_SHIFT, n, b, -1, -1);
		if (is_const(N.a[n]) && (N.ki[N.a[n]] == 3 || N.ki[N.a[n]] == 5 ||
		    N.ki[N.a[n]] == 9))
			return emit(L, XC_LEA, n, b, -1, -1);
		return emit(L, XC_IMUL, n, a, b, -1);
	case O_FMA:
		return emit(L, XC_VFMA, n, a, b, c);
	case O_DIV: case O_MOD:
		if (fp)
			return emit(L, TYW(t) == 64 ? XC_VFDIVD : XC_VFDIVS, n, a, b,
			    -1);
		if (is_const(N.b[n])) {
			int64_t k = N.ki[N.b[n]];
			if (k > 0 && !(k & (k - 1)) && TYK(t) == K_UINT)
				return emit(L, op == O_DIV ? XC_SHIFT : XC_ALU, n, a,
				    -1, -1);
			x = emit(L, XC_MULX, n, a, -1, -1);
			x = emit(L, XC_SHIFT, n, x, -1, -1);
			if (TYK(t) != K_UINT)
				x = emit(L, XC_ALU, n, x, a, -1);
			if (op == O_MOD) {
				x = emit(L, XC_IMUL, n, x, -1, -1);
				x = emit(L, XC_ALU, n, x, a, -1);
			}
			return x;
		}
		return emit(L, w > 32 ? XC_DIV64 : XC_DIV32, n, a, b, -1);
	case O_SHL: case O_SHR:
		if (is_const(N.b[n]))
			return emit(L, XC_SHIFT, n, a, -1, -1);
		return emit(L, has_feat(L->ua, XF_BMI2) ? XC_SHIFTX : XC_SHIFTCL,
		    n, a, b, -1);
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE:
		x = emit(L, ta && TY_FP(ta) ? XC_VFCMP : XC_ALU, n, a, b, -1);
		L->flag[n] = x;
		return x;
	case O_MIN: case O_MAX:
		if (fp)
			return emit(L, XC_VFMINMAX, n, a, b, -1);
		x = emit(L, XC_ALU, n, a, b, -1);
		return emit(L, XC_CMOV, n, x, a, b);
	case O_ABS:
		if (fp)
			return emit(L, XC_VALU, n, a, -1, -1);
		x = emit(L, XC_ALU, n, a, -1, -1);
		return emit(L, XC_CMOV, n, x, a, -1);
	case O_SIGN:
		x = emit(L, XC_ALU, n, a, -1, -1);
		x = emit(L, XC_SETCC, n, x, -1, -1);
		return emit(L, XC_ALU, n, x, -1, -1);
	case O_CLZ: case O_CTZ: case O_POPCNT:
		x = emit(L, XC_BITCNT, n, a, -1, -1);
		if ((op == O_CLZ && !has_feat(L->ua, XF_LZCNT)) ||
		    (op == O_CTZ && !has_feat(L->ua, XF_BMI1)))
			x = emit(L, XC_CMOV, n, x, -1, -1);   /* bsr/bsf + zero fix */
		return x;
	case O_CAST:
		if (fp && ta && TY_FP(ta))
			return emit(L, XC_VCVTW, n, a, -1, -1);
		if (fp)
			return emit(L, XC_CVTI2F, n, a, -1, -1);
		if (ta && TY_FP(ta))
			return emit(L, XC_CVTF2I, n, a, -1, -1);
		if (ta && TY_SIZED(ta) && TY_SIZED(t) && TYW(t) > TYW(ta) &&
		    TYW(ta) > 1)
			return emit(L, XC_ALU, n, a, -1, -1);   /* movzx/movsx */
		return a;
	case O_ASUINT: case O_ASINT:
		return emit(L, XC_MOVX2G, n, a, -1, -1);
	case O_ASFLOAT:
		return emit(L, XC_MOVG2X, n, a, -1, -1);
	case O_SQRT:
		return emit(L, fp && TYW(t) == 64 ? XC_VFSQRTD : XC_VFSQRTS, n, a,
		    -1, -1);
	case O_ISQRT:
		x = emit(L, XC_CVTI2F, n, a, -1, -1);
		x = emit(L, XC_VFSQRTD, n, x, -1, -1);
		return emit(L, XC_CVTF2I, n, x, -1, -1);
	case O_FLOOR: case O_CEIL: case O_ROUNDEVEN:
		return emit(L, XC_VROUND, n, a, -1, -1);
	case O_BARRIER:
		return a;
	case O_ATOMIC:
		return emit(L, XC_LOCK, n, a, b, -1);
	default:
		if (opinfo[op].fl & OF_REALFN)
			return emit(L, XC_LIBM, n, a, b, -1);
		return emit(L, fp ? XC_VFADD : XC_ALU, n, a, b, -1);
	}
}

static int32_t
lower(struct low *L, nid n)
{
	int32_t v;

	if (n == 0)
		return -1;
	if (L->outer && L->invar[n])
		return lower(L->outer, n) >= 0 ? -1 : -1;   /* hoisted: ready */
	if (L->key[n] == n)
		return L->val[n];
	v = lower1(L, n);
	L->key[n] = n;
	L->val[n] = v;
	return v;
}

/* mark the nodes of a loop body that do not read the loop's slots */
static void
mark_invar(struct low *L, nid n, uint8_t *seen)
{
	uint32_t j;

	if (n == 0 || seen[n])
		return;
	seen[n] = 1;
	L->invar[n] = !(node_slots(n) & L->bound);
	if (N.op[n] == O_TABLE) {
		for (j = 0; j < N.b[n]; j++)
			mark_invar(L, N.lst[N.a[n] + j], seen);
		return;
	}
	if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_LOOP)
		return;
	mark_invar(L, N.a[n], seen);
	mark_invar(L, N.b[n], seen);
	mark_invar(L, N.c[n], seen);
}

static nid
first_loop(nid n, uint8_t *seen)
{
	uint32_t j;
	nid r;

	if (n == 0 || seen[n])
		return 0;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_LOOP:
		return n;
	case O_CONST: case O_VAR: case O_UNDEF:
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			if ((r = first_loop(N.lst[N.a[n] + j], seen)) != 0)
				return r;
		return 0;
	}
	if ((r = first_loop(N.a[n], seen)) != 0)
		return r;
	if ((r = first_loop(N.b[n], seen)) != 0)
		return r;
	return first_loop(N.c[n], seen);
}

static void
low_init(struct low *L, int ua, struct mblock *b)
{
	memset(L, 0, sizeof *L);
	L->ua = ua;
	L->b = b;
	L->lim = N.n + 64;
	L->key = xcalloc(L->lim, sizeof *L->key);
	L->val = xcalloc(L->lim, sizeof *L->val);
	L->flag = xcalloc(L->lim, sizeof *L->flag);
	memset(L->flag, 0xff, L->lim * sizeof *L->flag);
	L->islot = -1;
}

static void
low_free(struct low *L)
{
	free(L->key);
	free(L->val);
	free(L->flag);
	free(L->invar);
}

/*
 * Lower f: the code outside its first loop to `outer`, the loop's body
 * (per iteration) to `loop`; trips = the trip count when it is constant
 * (-1 otherwise, 0 when there is no loop).  Returns the number of loops
 * left out (nested or after the first).
 */
int
x86_lower(struct func *f, int ua, struct mblock *outer, struct mblock *loop,
    double *trips)
{
	struct low O, Lp;
	uint8_t *seen = xcalloc(N.n + 1, 1);
	nid ln;
	int skipped = 0;

	x86_init();
	*trips = 0;
	low_init(&O, ua, outer);
	ln = first_loop(f->body, seen);
	memset(seen, 0, N.n + 1);
	if (ln) {
		uint32_t e = loop_ent(ln), s = loops.st[e], k;
		int32_t prod[MAXST], idx = -1, cnd;
		int64_t lo, hi;
		low_init(&Lp, ua, loop);
		Lp.outer = &O;
		Lp.bound = loops.bound[e];
		Lp.nst = loops.nst[e];
		Lp.islot = loops.kind[e] == LK_FOR ? loops.islot[e] : -1;
		for (k = 0; k < loops.nst[e]; k++)
			Lp.sslot[k] = loops.sslot[s + k];
		Lp.invar = xcalloc(Lp.lim, 1);
		for (k = 0; k < loops.nst[e]; k++) {
			mark_invar(&Lp, loops.next[s + k], seen);
			lower(&O, loops.init[s + k]);
		}
		mark_invar(&Lp, loops.cond[e], seen);
		for (k = 0; k < loops.nst[e]; k++)
			prod[k] = lower(&Lp, loops.next[s + k]);
		if (loops.kind[e] == LK_FOR) {
			mblock_add(loop, XC_ALU, VW_X, "add       index, 1", PH_INDEX,
			    -1, -1);
			idx = (int32_t)loop->n - 1;
			mblock_add(loop, XC_BR, VW_X, "cmp+jcc   index, end", idx,
			    -1, -1);
		} else {
			cnd = lower(&Lp, loops.cond[e]);
			mblock_add(loop, XC_BR, VW_X, "jcc       while", cnd, -1, -1);
		}
		/* resolve the placeholders: carried from the last iteration */
		{
			uint32_t i;
			int j;
			for (i = 0; i < loop->n; i++)
				for (j = 0; j < MCA_MAXDEP; j++) {
					int32_t d = loop->in[i].dep[j], to = -1;
					if (d > -100)
						continue;
					if (d == PH_INDEX)
						to = idx;
					else if (d <= PH_STATE && d > PH_STATE - MAXST)
						to = prod[PH_STATE - d];
					/* a state that is another state's value */
					while (to <= PH_STATE && to > PH_STATE - MAXST)
						to = -1;
					loop->in[i].dep[j] = to;
					loop->in[i].carried[j] = to >= 0;
				}
		}
		loop->loop = 1;
		if (loops.kind[e] == LK_FOR && int_of(N.a[ln], &lo) &&
		    int_of(N.b[ln], &hi))
			*trips = hi >= lo ? (double)(hi - lo + 1) : 0;
		else
			*trips = -1;
		skipped += Lp.nested;
		low_free(&Lp);
	}
	lower(&O, f->body);
	skipped += O.nested - (ln ? 1 : 0);
	low_free(&O);
	free(seen);
	return skipped;
}

void
cmd_mca(struct func *f, int ua)
{
	struct mblock ob = {0}, lb = {0};
	struct mres ro, rl;
	double trips;
	int skipped;
	const char *fname = f->anon ? shared_str(f->body, 80) : sym_str(f->name);

	skipped = x86_lower(f, ua, &ob, &lb, &trips);
	printf("mca of %s on %s (%s)\n", fname, uarchs[ua].name,
	    uarchs[ua].alias);
	rep_begin("mca");
	rep_str("model", uarchs[ua].name);
	rep_list("blocks");
	if (ob.n) {
		mca_run(&ob, ua, &ro);
		mca_print(&ob, ua, &ro, lb.n ? "outside the loop (once)" :
		    "straight-line code");
	}
	if (lb.n) {
		mca_run(&lb, ua, &rl);
		mca_print(&lb, ua, &rl, "loop body (per iteration)");
	}
	rep_end();
	if (lb.n && trips > 0) {
		double tot = (ob.n ? ro.cycles : 0) + trips * rl.cycles;
		printf("  total: about %.0f cycles (%g iterations)\n", tot, trips);
		rep_num("trips", trips);
		rep_num("total_cycles", floor(tot + 0.5));
	}
	if (skipped > 0) {
		printf("  note: %d further loop%s (nested or later) priced as "
		    "inputs, not simulated\n", skipped, skipped == 1 ? "" : "s");
		rep_int("loops_not_simulated", skipped);
	}
	free(ob.in);
	free(lb.in);
}
