/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Loops in the expression DAG: the entry table, the O_LOOP node, its
 * evaluation, printing and substitution.
 *
 * Evaluation is column-wise like the rest of the evaluator: the lanes of
 * a chunk are the domain points, and every iteration runs the loop's
 * step plan once over all lanes.  Lanes that finish early are masked.  A
 * lane that needs more than MAXTRIP iterations is reported undefined and
 * counted in loop_skipped, so a caller can say how many points it could
 * not check.
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct ltab loops;
uint64_t loop_skipped;

/* ---- small helpers ---- */

int
num_of(nid n, double *v)
{
	if (!is_const(n))
		return 0;
	if (TY_FP(N.ty[n]))
		*v = N.kr[n];
	else if (TYK(N.ty[n]) == K_UINT && TYW(N.ty[n]) == 64)
		*v = (double)(uint64_t)N.ki[n];
	else
		*v = (double)N.ki[n];
	return 1;
}

int
int_of(nid n, int64_t *v)
{
	double d;

	if (!num_of(n, &d))
		return 0;
	if (!TY_FP(N.ty[n])) {
		*v = N.ki[n];
		return 1;
	}
	if (d != floor(d) || fabs(d) > 9.0e15)
		return 0;
	*v = (int64_t)d;
	return 1;
}

nid
var_of(int slot, uint32_t name, ty_t t)
{
	return mk_var((uint32_t)slot, name, t);
}

/* ---- slot masks: which variables a node reads ---- */

static uint64_t *smask;
static uint8_t *sdone;
static uint32_t scap;

uint64_t
node_slots(nid n)
{
	uint64_t m = 0;
	uint32_t j;

	if (n == 0)
		return 0;
	if (n >= scap) {
		uint32_t c = newcap(scap, N.n + 1);
		smask = xrealloc(smask, c * sizeof *smask);
		sdone = xrealloc(sdone, c);
		memset(sdone + scap, 0, c - scap);
		scap = c;
	}
	if (sdone[n])
		return smask[n];
	switch (N.op[n]) {
	case O_CONST:
	case O_UNDEF:
		break;
	case O_VAR:
		m = N.ki[n] < 64 ? 1ULL << N.ki[n] : 0;
		break;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			m |= node_slots(N.lst[N.a[n] + j]);
		break;
	case O_LOOP:
		m = loops.free[(uint32_t)N.ki[n]];
		/* FALLTHROUGH */
	default:
		m |= node_slots(N.a[n]) | node_slots(N.b[n]) | node_slots(N.c[n]);
	}
	if (n >= scap)
		(void)node_slots(n);   /* cannot happen: grown above */
	smask[n] = m;
	sdone[n] = 1;
	return m;
}

int
has_slot(nid n, int slot)
{
	return (node_slots(n) >> slot) & 1;
}

/* highest slot read, or -1 */
int
loop_top_slot(nid n)
{
	uint64_t m = node_slots(n);
	int s = -1;

	while (m) {
		s++;
		m >>= 1;
	}
	return s;
}

/* ---- the entry table ---- */

static uint32_t
lhash(const struct loopdef *d)
{
	uint64_t h = 1469598103934665603ULL;
	int j;

#define H(x) (h = (h ^ (uint64_t)(x)) * 1099511628211ULL)
	H(d->kind); H(d->orig); H(d->flags); H(d->name); H(d->islot);
	H(d->ity); H(d->nst); H(d->cond); H(d->aux);
	for (j = 0; j < d->nst; j++) {
		H(d->sslot[j]); H(d->sname[j]); H(d->sty[j]);
		H(d->init[j]); H(d->next[j]);
	}
#undef H
	return (uint32_t)(h ^ (h >> 32));
}

static int
lsame(uint32_t e, const struct loopdef *d)
{
	uint32_t s = loops.st[e];
	int j;

	if (loops.kind[e] != d->kind || loops.orig[e] != d->orig ||
	    loops.flags[e] != d->flags || loops.name[e] != d->name ||
	    loops.islot[e] != d->islot || loops.ity[e] != d->ity ||
	    loops.nst[e] != d->nst || loops.cond[e] != d->cond || loops.aux[e] != d->aux)
		return 0;
	for (j = 0; j < d->nst; j++)
		if (loops.sslot[s + j] != d->sslot[j] || loops.sname[s + j] != d->sname[j] ||
		    loops.sty[s + j] != d->sty[j] || loops.init[s + j] != d->init[j] ||
		    loops.next[s + j] != d->next[j])
			return 0;
	return 1;
}

static void
lgrow(void)
{
	uint32_t c = newcap(loops.cap, loops.n + 1);

	loops.kind = xrealloc(loops.kind, c);
	loops.orig = xrealloc(loops.orig, c);
	loops.flags = xrealloc(loops.flags, c);
	loops.nst = xrealloc(loops.nst, c);
	loops.islot = xrealloc(loops.islot, c);
	loops.ity = xrealloc(loops.ity, c * sizeof *loops.ity);
	loops.name = xrealloc(loops.name, c * sizeof *loops.name);
	loops.st = xrealloc(loops.st, c * sizeof *loops.st);
	loops.cond = xrealloc(loops.cond, c * sizeof *loops.cond);
	loops.aux = xrealloc(loops.aux, c * sizeof *loops.aux);
	loops.bound = xrealloc(loops.bound, c * sizeof *loops.bound);
	loops.free = xrealloc(loops.free, c * sizeof *loops.free);
	loops.cap = c;
}

static void
lrehash(void)
{
	uint32_t e, j;
	struct loopdef d;

	free(loops.ht);
	loops.htcap = loops.htcap ? loops.htcap * 2 : 256;
	loops.ht = xcalloc(loops.htcap, sizeof *loops.ht);
	for (e = 0; e < loops.n; e++) {
		loop_get(e, &d);
		j = lhash(&d) & (loops.htcap - 1);
		while (loops.ht[j])
			j = (j + 1) & (loops.htcap - 1);
		loops.ht[j] = e + 1;
	}
}

uint32_t
loop_intern(const struct loopdef *d)
{
	uint32_t j, e, s;
	uint64_t bound = 0, fr = 0;
	int k;

	if ((loops.n + 1) * 2 > loops.htcap)
		lrehash();
	j = lhash(d) & (loops.htcap - 1);
	while (loops.ht[j]) {
		if (lsame(loops.ht[j] - 1, d))
			return loops.ht[j] - 1;
		j = (j + 1) & (loops.htcap - 1);
	}
	if (loops.n == loops.cap)
		lgrow();
	if (loops.ns + MAXST > loops.scap) {
		uint32_t c = newcap(loops.scap, loops.ns + MAXST);
		loops.sslot = xrealloc(loops.sslot, c);
		loops.sname = xrealloc(loops.sname, c * sizeof *loops.sname);
		loops.sty = xrealloc(loops.sty, c * sizeof *loops.sty);
		loops.init = xrealloc(loops.init, c * sizeof *loops.init);
		loops.next = xrealloc(loops.next, c * sizeof *loops.next);
		loops.scap = c;
	}
	e = loops.n++;
	s = loops.ns;
	loops.ns += (uint32_t)d->nst;
	loops.kind[e] = (uint8_t)d->kind;
	loops.orig[e] = (uint8_t)d->orig;
	loops.flags[e] = (uint8_t)d->flags;
	loops.nst[e] = (uint8_t)d->nst;
	loops.islot[e] = (uint8_t)d->islot;
	loops.ity[e] = d->ity;
	loops.name[e] = d->name;
	loops.st[e] = s;
	loops.cond[e] = d->cond;
	loops.aux[e] = d->aux;
	if (d->kind == LK_FOR)
		bound |= 1ULL << d->islot;
	for (k = 0; k < d->nst; k++) {
		loops.sslot[s + k] = (uint8_t)d->sslot[k];
		loops.sname[s + k] = d->sname[k];
		loops.sty[s + k] = d->sty[k];
		loops.init[s + k] = d->init[k];
		loops.next[s + k] = d->next[k];
		bound |= 1ULL << d->sslot[k];
		fr |= node_slots(d->init[k]) | node_slots(d->next[k]);
	}
	fr |= node_slots(d->cond) | node_slots(d->aux);
	loops.bound[e] = bound;
	loops.free[e] = fr & ~bound;
	loops.ht[j] = e + 1;
	return e;
}

void
loop_get(uint32_t e, struct loopdef *d)
{
	uint32_t s = loops.st[e];
	int k;

	memset(d, 0, sizeof *d);
	d->kind = loops.kind[e];
	d->orig = loops.orig[e];
	d->flags = loops.flags[e];
	d->name = loops.name[e];
	d->islot = loops.islot[e];
	d->ity = loops.ity[e];
	d->nst = loops.nst[e];
	d->cond = loops.cond[e];
	d->aux = loops.aux[e];
	for (k = 0; k < d->nst; k++) {
		d->sslot[k] = loops.sslot[s + k];
		d->sname[k] = loops.sname[s + k];
		d->sty[k] = loops.sty[s + k];
		d->init[k] = loops.init[s + k];
		d->next[k] = loops.next[s + k];
	}
}

uint32_t
loop_ent(nid n)
{
	return (uint32_t)N.ki[n];
}

int
loop_res(nid n)
{
	return (int)((uint64_t)N.ki[n] >> 32);
}

/* ---- the node ---- */

static int
fold_loop(nid n, int64_t *vi, double *vr)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int ok;

	plan_build(&p, &n, 1, 0);
	plan_run(&p, 1, in_i, in_r);
	ok = !(COLF(&p, p.root[0])[0] & F_UNDEF);
	*vi = COLI(&p, p.root[0])[0];
	*vr = COLR(&p, p.root[0])[0];
	plan_free(&p);
	free(in_i);
	free(in_r);
	return ok;
}

nid
mk_loop(uint32_t e, int res, nid lo, nid hi, nid q)
{
	ty_t t = loops.sty[loops.st[e] + res];
	uint32_t s = loops.st[e];
	int fl = 0, k;
	nid n;
	int64_t vi;
	double vr;

	if (loops.kind[e] == LK_FOR) {
		lo = mk_cast(TY_INT, lo);
		hi = mk_cast(TY_INT, hi);
		if (N.op[lo] == O_UNDEF || N.op[hi] == O_UNDEF)
			return mk_undef(t);
		fl |= (N.fl[lo] | N.fl[hi]) & NF_PARTIAL;
	} else {
		lo = hi = 0;
		fl |= NF_PARTIAL;   /* it may not terminate */
	}
	if (loops.flags[e] & LF_NEGUNDEF)
		fl |= NF_PARTIAL;
	for (k = 0; k < loops.nst[e]; k++)
		fl |= (N.fl[loops.init[s + k]] | N.fl[loops.next[s + k]]) & NF_PARTIAL;
	if (loops.cond[e])
		fl |= N.fl[loops.cond[e]] & NF_PARTIAL;
	n = mk_raw(O_LOOP, t, lo, hi, q,
	    (int64_t)((uint64_t)e | ((uint64_t)res << 32)), fl);
	if (node_slots(n) != 0 && !(N.fl[n] & NF_VAR))
		N.fl[n] |= NF_VAR;
	if (node_slots(n) == 0) {
		uint64_t sk = loop_skipped;
		if (fold_loop(n, &vi, &vr)) {
			loop_skipped = sk;
			return TY_FP(t) ? (TYK(t) == K_FLOAT ?
			    mk_cast(t, mk_const_r(vr)) : mk_const_r(vr)) :
			    mk_const_i(t, vi);
		}
		loop_skipped = sk;
	}
	return n;
}

/* ---- evaluation ---- */

struct lpc {
	struct plan p;
	uint32_t init[MAXST], next[MAXST], cond;
	int built;
};
static struct lpc *lpcs;
static uint32_t nlpc;

static struct lpc *
lpc_get(uint32_t e, int ideal)
{
	uint32_t i = e * 2 + (ideal != 0), k, m = loops.nst[e], s = loops.st[e];
	struct lpc *c;
	nid roots[2 * MAXST + 1];

	if (i >= nlpc) {
		uint32_t nn = newcap(nlpc, i + 1);
		lpcs = xrealloc(lpcs, nn * sizeof *lpcs);
		memset(lpcs + nlpc, 0, (nn - nlpc) * sizeof *lpcs);
		nlpc = nn;
	}
	c = &lpcs[i];
	if (c->built)
		return c;
	for (k = 0; k < m; k++) {
		roots[k] = loops.init[s + k];
		roots[m + k] = loops.next[s + k];
	}
	roots[2 * m] = loops.cond[e];
	plan_build(&c->p, roots, 2 * m + 1, ideal);
	for (k = 0; k < m; k++) {
		c->init[k] = c->p.root[k];
		c->next[k] = c->p.root[m + k];
	}
	c->cond = c->p.root[2 * m];
	c->built = 1;
	return c;
}

#define MAXDEPTH 24
struct lbuf {
	int64_t *vi, *si, *trip, *lo;
	double *vr, *sr;
	uint8_t *dead, *fin;
};
static struct lbuf bufs[MAXDEPTH];
static int depth;

static struct lbuf *
lbuf_get(int d)
{
	struct lbuf *b = &bufs[d];

	if (b->vi == NULL) {
		b->vi = xcalloc((size_t)NSLOT * CH, sizeof *b->vi);
		b->vr = xcalloc((size_t)NSLOT * CH, sizeof *b->vr);
		b->si = xcalloc((size_t)MAXST * CH, sizeof *b->si);
		b->sr = xcalloc((size_t)MAXST * CH, sizeof *b->sr);
		b->trip = xcalloc(CH, sizeof *b->trip);
		b->lo = xcalloc(CH, sizeof *b->lo);
		b->dead = xcalloc(CH, 1);
		b->fin = xcalloc(CH, 1);
	}
	return b;
}

static void
put_states(struct lbuf *B, uint32_t e, uint32_t np)
{
	uint32_t k, s = loops.st[e], m = loops.nst[e];

	for (k = 0; k < m; k++) {
		size_t at = (size_t)loops.sslot[s + k] * CH;
		if (TY_FP(loops.sty[s + k]))
			memcpy(B->vr + at, B->sr + (size_t)k * CH, np * sizeof *B->vr);
		else
			memcpy(B->vi + at, B->si + (size_t)k * CH, np * sizeof *B->vi);
	}
}

void
loop_run(struct plan *p, uint32_t k, uint32_t np, const int64_t *in_i,
    const double *in_r)
{
	uint64_t key = (uint64_t)p->ki[k];
	uint32_t e = (uint32_t)key, l, j, m = loops.nst[e], s = loops.st[e];
	uint32_t res = (uint32_t)(key >> 32);
	int W = MAXP, isfor = loops.kind[e] == LK_FOR;
	struct lpc *c = lpc_get(e, p->ideal);
	struct lbuf *B;
	int64_t *O = COLI(p, k), maxt = 0, it;
	double *OR = COLR(p, k);
	uint8_t *OF = COLF(p, k);
	size_t is = (size_t)loops.islot[e] * CH;

	if (depth >= MAXDEPTH)
		fail("loops nested too deeply to evaluate");
	if (depth > 0 && in_i == bufs[depth - 1].vi)
		W = NSLOT;
	B = lbuf_get(depth);
	memcpy(B->vi, in_i, (size_t)W * CH * sizeof *B->vi);
	memcpy(B->vr, in_r, (size_t)W * CH * sizeof *B->vr);
	for (l = 0; l < np; l++) {
		B->dead[l] = 0;
		B->fin[l] = 0;
		B->trip[l] = 0;
		B->lo[l] = 0;
		if (!isfor)
			continue;
		if ((COLF(p, p->a[k])[l] | COLF(p, p->b[k])[l]) & F_UNDEF) {
			B->dead[l] = 1;
			continue;
		}
		B->lo[l] = COLI(p, p->a[k])[l];
		B->trip[l] = COLI(p, p->b[k])[l] - B->lo[l] + 1;
		if (B->trip[l] < 0) {
			if (loops.flags[e] & LF_NEGUNDEF)
				B->dead[l] = 1;
			B->trip[l] = 0;
		}
		if (B->trip[l] > (int64_t)MAXTRIP) {
			B->dead[l] = 2;
			loop_skipped++;
		}
		if (!B->dead[l] && B->trip[l] > maxt)
			maxt = B->trip[l];
	}
	for (l = 0; l < np; l++) {
		B->vi[is + l] = B->lo[l];
		B->vr[is + l] = (double)B->lo[l];
	}
	for (j = 0; j < m; j++) {
		memset(B->si + (size_t)j * CH, 0, CH * sizeof *B->si);
		memset(B->sr + (size_t)j * CH, 0, CH * sizeof *B->sr);
	}
	put_states(B, e, np);
	depth++;
	plan_run(&c->p, np, B->vi, B->vr);
	for (j = 0; j < m; j++) {
		uint32_t loc = c->init[j];
		for (l = 0; l < np; l++) {
			if (COLF(&c->p, loc)[l] & F_UNDEF)
				B->dead[l] |= 1;
			B->si[(size_t)j * CH + l] = COLI(&c->p, loc)[l];
			B->sr[(size_t)j * CH + l] = COLR(&c->p, loc)[l];
		}
	}
	for (it = 0;; it++) {
		int any = 0;
		for (l = 0; l < np; l++)
			if (!B->dead[l] && !B->fin[l] && (!isfor || it < B->trip[l]))
				any = 1;
		if (!any)
			break;
		if (!isfor && it >= (int64_t)MAXTRIP) {
			for (l = 0; l < np; l++)
				if (!B->dead[l] && !B->fin[l]) {
					B->dead[l] = 2;
					loop_skipped++;
				}
			break;
		}
		if (isfor)
			for (l = 0; l < np; l++) {
				B->vi[is + l] = B->lo[l] + it;
				B->vr[is + l] = (double)(B->lo[l] + it);
			}
		put_states(B, e, np);
		plan_run(&c->p, np, B->vi, B->vr);
		for (l = 0; l < np; l++) {
			if (B->dead[l] || B->fin[l] || (isfor && it >= B->trip[l]))
				continue;
			if (!isfor) {
				uint32_t cl = c->cond;
				if (COLF(&c->p, cl)[l] & F_UNDEF) {
					B->dead[l] = 1;
					continue;
				}
				if (!COLI(&c->p, cl)[l]) {
					B->fin[l] = 1;
					continue;
				}
			}
			for (j = 0; j < m; j++) {
				uint32_t loc = c->next[j];
				if (COLF(&c->p, loc)[l] & F_UNDEF)
					B->dead[l] = 1;
			}
			if (B->dead[l])
				continue;
			for (j = 0; j < m; j++) {
				uint32_t loc = c->next[j];
				B->si[(size_t)j * CH + l] = COLI(&c->p, loc)[l];
				B->sr[(size_t)j * CH + l] = COLR(&c->p, loc)[l];
			}
		}
	}
	depth--;
	(void)maxt;
	(void)s;
	for (l = 0; l < np; l++) {
		O[l] = B->si[(size_t)res * CH + l];
		OR[l] = B->sr[(size_t)res * CH + l];
		OF[l] = B->dead[l] ? F_UNDEF : 0;
	}
}

/* ---- printing ---- */

void
loop_print(struct sbuf *b, nid n, const struct pctx *pc)
{
	uint32_t e = loop_ent(n), s = loops.st[e];
	int res = loop_res(n), k;

	switch (loops.orig[e]) {
	case LO_SUM:
	case LO_PROD:
		sb_printf(b, "%s(%s, ", loops.orig[e] == LO_SUM ? "sum" : "prod",
		    sym_str(loops.name[e]));
		print_node(b, N.a[n], pc);
		sb_puts(b, ", ");
		print_node(b, N.b[n], pc);
		sb_puts(b, ", ");
		print_node(b, loops.aux[e], pc);
		sb_putc(b, ')');
		return;
	case LO_REC:
		sb_printf(b, "%s[", sym_str(loops.name[e]));
		print_node(b, N.c[n] ? N.c[n] : N.b[n], pc);
		sb_putc(b, ']');
		return;
	}
	sb_puts(b, "loop(");
	if (loops.kind[e] == LK_FOR) {
		sb_printf(b, "%s in ", sym_str(loops.name[e]));
		print_node(b, N.a[n], pc);
		sb_puts(b, "..");
		print_node(b, N.b[n], pc);
	} else {
		sb_puts(b, "while ");
		print_node(b, loops.cond[e], pc);
	}
	for (k = 0; k < loops.nst[e]; k++) {
		sb_printf(b, "; %s = ", sym_str(loops.sname[s + k]));
		print_node(b, loops.init[s + k], pc);
		sb_puts(b, " then ");
		print_node(b, loops.next[s + k], pc);
	}
	sb_printf(b, ").%s", sym_str(loops.sname[s + res]));
}

/* ---- substitution: f(x) = sum(...) called with arguments ---- */

static void
remap_ent(uint32_t e, const nid *map, int nmap, uint32_t *ne)
{
	struct loopdef d;
	nid m2[NSLOT];
	uint64_t vals = 0;
	int j, k, lo = 64, hi = -1, delta = 0, any = 0;

	loop_get(e, &d);
	memset(m2, 0, sizeof m2);
	for (j = 0; j < nmap && j < NSLOT; j++)
		if (map[j] && ((loops.free[e] >> j) & 1)) {
			m2[j] = map[j];
			vals |= node_slots(map[j]);
			any = 1;
		}
	if (!any) {
		*ne = e;
		return;
	}
	for (j = 0; j < 64; j++)
		if ((loops.bound[e] >> j) & 1) {
			if (j < lo)
				lo = j;
			if (j > hi)
				hi = j;
		}
	/* the arguments read slots this loop binds: move the loop up */
	if (vals & loops.bound[e]) {
		for (j = 63; j >= 0; j--)
			if ((vals >> j) & 1)
				break;
		delta = j + 1 - lo;
		/* nested loops inside go up by the same amount */
		for (k = 0; k < d.nst; k++) {
			int t = loop_top_slot(d.next[k]);
			if (t > hi)
				hi = t;
		}
		if (hi + delta >= KSLOT)
			fail("loops nested too deeply");
		if (d.kind == LK_FOR)
			m2[d.islot] = var_of(d.islot + delta, d.name, d.ity);
		for (k = 0; k < d.nst; k++)
			m2[d.sslot[k]] = var_of(d.sslot[k] + delta, d.sname[k],
			    d.sty[k]);
		if (d.kind == LK_FOR)
			d.islot += delta;
		for (k = 0; k < d.nst; k++)
			d.sslot[k] += delta;
	}
	for (k = 0; k < d.nst; k++) {
		d.init[k] = subst(d.init[k], m2, NSLOT);
		d.next[k] = mk_cast(d.sty[k], subst(d.next[k], m2, NSLOT));
	}
	if (d.cond)
		d.cond = subst(d.cond, m2, NSLOT);
	if (d.aux)
		d.aux = subst(d.aux, m2, NSLOT);
	*ne = loop_intern(&d);
}

nid
loop_remake(nid n, nid a, nid b, nid c, const nid *map, int nmap)
{
	uint32_t ne;

	remap_ent(loop_ent(n), map, nmap, &ne);
	return mk_loop(ne, loop_res(n), a, b, c);
}

/* ---- generic rebuild with a replacement ---- */

struct xr {
	nid from, to;
	nid *memo;
	uint32_t lim;
};

static nid xr1(struct xr *x, nid n);

static uint32_t
xr_ent(struct xr *x, uint32_t e)
{
	struct loopdef d;
	int k, ch = 0;
	nid t;

	loop_get(e, &d);
	for (k = 0; k < d.nst; k++) {
		t = xr1(x, d.init[k]);
		ch |= t != d.init[k];
		d.init[k] = t;
		t = mk_cast(d.sty[k], xr1(x, d.next[k]));
		ch |= t != d.next[k];
		d.next[k] = t;
	}
	if (d.cond) {
		t = xr1(x, d.cond);
		ch |= t != d.cond;
		d.cond = t;
	}
	if (d.aux) {
		t = xr1(x, d.aux);
		ch |= t != d.aux;
		d.aux = t;
	}
	return ch ? loop_intern(&d) : e;
}

static nid
xr1(struct xr *x, nid n)
{
	nid r, *e;
	uint32_t i;

	if (n == 0)
		return 0;
	if (n == x->from)
		return x->to;
	if (n < x->lim && x->memo[n])
		return x->memo[n];
	switch (N.op[n]) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
		r = n;
		break;
	case O_TABLE:
		e = xmalloc(N.b[n] * sizeof *e + 1);
		for (i = 0; i < N.b[n]; i++)
			e[i] = xr1(x, N.lst[N.a[n] + i]);
		r = mk_table(e, N.b[n]);
		free(e);
		break;
	case O_SEL:
		r = mk_sel(xr1(x, N.a[n]), xr1(x, N.b[n]), xr1(x, N.c[n]));
		break;
	case O_CAST:
		r = mk_cast(N.ty[n], xr1(x, N.a[n]));
		break;
	case O_INDEX:
		r = mk_index(xr1(x, N.a[n]), xr1(x, N.b[n]));
		break;
	case O_BARRIER:
	case O_ATOMIC:
		r = n;
		break;
	case O_LOOP:
		r = mk_loop(xr_ent(x, loop_ent(n)), loop_res(n), xr1(x, N.a[n]),
		    xr1(x, N.b[n]), xr1(x, N.c[n]));
		break;
	default:
		if (opinfo[N.op[n]].arity == 1)
			r = mk_un(N.op[n], xr1(x, N.a[n]));
		else
			r = mk_bin(N.op[n], xr1(x, N.a[n]), xr1(x, N.b[n]));
	}
	if (n < x->lim)
		x->memo[n] = r;
	return r;
}

nid
xreplace(nid root, nid from, nid to)
{
	struct xr x;
	nid r;

	x.from = from;
	x.to = to;
	x.lim = N.n;
	x.memo = xcalloc(x.lim, sizeof *x.memo);
	r = xr1(&x, root);
	free(x.memo);
	return r;
}

/* ---- cost: per iteration times the trip count ---- */

static float
body_cost(const struct model *m, nid n, uint8_t *mark, uint32_t lim)
{
	float c = 0;
	int cls;
	uint32_t j;

	if (n == 0 || n >= lim || mark[n])
		return 0;
	mark[n] = 1;
	if (N.op[n] == O_TABLE) {
		for (j = 0; j < N.b[n]; j++)
			c += body_cost(m, N.lst[N.a[n] + j], mark, lim);
		return c;
	}
	c = node_cost(m, n, &cls);
	if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF)
		return 0;
	return c + body_cost(m, N.a[n], mark, lim) +
	    body_cost(m, N.b[n], mark, lim) + body_cost(m, N.c[n], mark, lim);
}

float
loop_iter_cost(const struct model *m, nid n)
{
	uint32_t e = loop_ent(n), s = loops.st[e], lim = N.n;
	uint8_t *mark = xcalloc(lim, 1);
	float c = 0;
	int k;

	for (k = 0; k < loops.nst[e]; k++)
		c += body_cost(m, loops.next[s + k], mark, lim);
	c += body_cost(m, loops.cond[e], mark, lim);
	free(mark);
	return c + m->c[C_BRANCH];
}

float
loop_cost(const struct model *m, nid n, int *cls)
{
	float per = loop_iter_cost(m, n);
	int64_t lo, hi;

	*cls = C_BRANCH;
	if (loops.kind[loop_ent(n)] == LK_FOR && int_of(N.a[n], &lo) &&
	    int_of(N.b[n], &hi)) {
		note("loop %s priced as %lld iterations x %g (listed under "
		    "branch)", shared_str(n, 120), (long long)(hi - lo + 1),
		    (double)per);
		return per * (float)(hi >= lo ? hi - lo + 1 : 0);
	}
	note("loop %s priced per iteration (%g, listed under branch); "
	    "multiply by the trip count; see 'scheme'", shared_str(n, 120),
	    (double)per);
	return per;
}

/*
 * d/dx of a loop: forward mode.  Each real state s gets a companion ds
 * with ds0 = d init/dx and ds' = sum_l (d next/d s_l) ds_l + d next/dx.
 * The trip count and branches are piecewise constant in x and contribute
 * nothing where the derivative exists.
 */
nid
loop_diff(nid n, uint32_t slot)
{
	struct loopdef d, e;
	int j, l, m, top = MAXP, res = loop_res(n), k;
	int dpos[MAXST];

	if (!((node_slots(n) >> slot) & 1))
		return mk_const_i(TY_FP(N.ty[n]) ? N.ty[n] : TY_REAL, 0);
	loop_get(loop_ent(n), &d);
	if (!TY_FP(d.sty[res]))
		fail("%s is integer valued: no derivative (use 'diff' on an "
		    "integer parameter for the forward difference)",
		    node_str(n));
	for (j = 0; j < 64; j++)
		if ((loops.bound[loop_ent(n)] >> j) & 1 && j > top)
			top = j;
	for (j = 0; j < d.nst; j++) {
		int t = loop_top_slot(d.next[j]);
		if (t > top)
			top = t;
	}
	if (d.orig == LO_SUM && d.nst == 1) {
		/* d/dx sum(i, lo, hi, g) = sum(i, lo, hi, dg/dx) */
		nid dg = diff_node(d.aux, slot), sv;
		e = d;
		sv = var_of(e.sslot[0], e.sname[0], e.sty[0]);
		e.aux = dg;
		e.next[0] = mk_cast(e.sty[0], mk_bin(O_ADD, sv, dg));
		return mk_loop(loop_intern(&e), 0, N.a[n], N.b[n], N.c[n]);
	}
	e = d;
	e.orig = LO_BLOCK;
	m = d.nst;
	for (j = 0; j < d.nst; j++) {
		char nm[64];
		dpos[j] = -1;
		if (!TY_FP(d.sty[j]))
			continue;
		if (m == MAXST || top + 1 + m >= KSLOT - 1)
			fail("loop too large to differentiate");
		snprintf(nm, sizeof nm, "d%s", sym_str(d.sname[j]));
		dpos[j] = m;
		e.sslot[m] = top + 1 + m;
		e.sname[m] = sym_cstr(nm);
		e.sty[m] = d.sty[j];
		e.init[m] = mk_cast(d.sty[j], diff_node(d.init[j], slot));
		m++;
	}
	e.nst = m;
	for (j = 0; j < d.nst; j++) {
		nid acc;
		if (dpos[j] < 0)
			continue;
		acc = diff_node(d.next[j], slot);
		for (l = 0; l < d.nst; l++) {
			nid ds;
			if (dpos[l] < 0 || !has_slot(d.next[j], d.sslot[l]))
				continue;
			ds = var_of(e.sslot[dpos[l]], e.sname[dpos[l]], e.sty[dpos[l]]);
			acc = mk_bin(O_ADD, acc, mk_bin(O_MUL,
			    diff_node(d.next[j], (uint32_t)d.sslot[l]), ds));
		}
		k = dpos[j];
		e.next[k] = mk_cast(e.sty[k], acc);
	}
	return mk_loop(loop_intern(&e), dpos[res], N.a[n], N.b[n], N.c[n]);
}
