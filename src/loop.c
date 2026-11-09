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

struct ltab L;
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
		m = L.free[(uint32_t)N.ki[n]];
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
	uint32_t s = L.st[e];
	int j;

	if (L.kind[e] != d->kind || L.orig[e] != d->orig ||
	    L.flags[e] != d->flags || L.name[e] != d->name ||
	    L.islot[e] != d->islot || L.ity[e] != d->ity ||
	    L.nst[e] != d->nst || L.cond[e] != d->cond || L.aux[e] != d->aux)
		return 0;
	for (j = 0; j < d->nst; j++)
		if (L.sslot[s + j] != d->sslot[j] || L.sname[s + j] != d->sname[j] ||
		    L.sty[s + j] != d->sty[j] || L.init[s + j] != d->init[j] ||
		    L.next[s + j] != d->next[j])
			return 0;
	return 1;
}

static void
lgrow(void)
{
	uint32_t c = newcap(L.cap, L.n + 1);

	L.kind = xrealloc(L.kind, c);
	L.orig = xrealloc(L.orig, c);
	L.flags = xrealloc(L.flags, c);
	L.nst = xrealloc(L.nst, c);
	L.islot = xrealloc(L.islot, c);
	L.ity = xrealloc(L.ity, c * sizeof *L.ity);
	L.name = xrealloc(L.name, c * sizeof *L.name);
	L.st = xrealloc(L.st, c * sizeof *L.st);
	L.cond = xrealloc(L.cond, c * sizeof *L.cond);
	L.aux = xrealloc(L.aux, c * sizeof *L.aux);
	L.bound = xrealloc(L.bound, c * sizeof *L.bound);
	L.free = xrealloc(L.free, c * sizeof *L.free);
	L.cap = c;
}

static void
lrehash(void)
{
	uint32_t e, j;
	struct loopdef d;

	free(L.ht);
	L.htcap = L.htcap ? L.htcap * 2 : 256;
	L.ht = xcalloc(L.htcap, sizeof *L.ht);
	for (e = 0; e < L.n; e++) {
		loop_get(e, &d);
		j = lhash(&d) & (L.htcap - 1);
		while (L.ht[j])
			j = (j + 1) & (L.htcap - 1);
		L.ht[j] = e + 1;
	}
}

uint32_t
loop_intern(const struct loopdef *d)
{
	uint32_t j, e, s;
	uint64_t bound = 0, fr = 0;
	int k;

	if ((L.n + 1) * 2 > L.htcap)
		lrehash();
	j = lhash(d) & (L.htcap - 1);
	while (L.ht[j]) {
		if (lsame(L.ht[j] - 1, d))
			return L.ht[j] - 1;
		j = (j + 1) & (L.htcap - 1);
	}
	if (L.n == L.cap)
		lgrow();
	if (L.ns + MAXST > L.scap) {
		uint32_t c = newcap(L.scap, L.ns + MAXST);
		L.sslot = xrealloc(L.sslot, c);
		L.sname = xrealloc(L.sname, c * sizeof *L.sname);
		L.sty = xrealloc(L.sty, c * sizeof *L.sty);
		L.init = xrealloc(L.init, c * sizeof *L.init);
		L.next = xrealloc(L.next, c * sizeof *L.next);
		L.scap = c;
	}
	e = L.n++;
	s = L.ns;
	L.ns += (uint32_t)d->nst;
	L.kind[e] = (uint8_t)d->kind;
	L.orig[e] = (uint8_t)d->orig;
	L.flags[e] = (uint8_t)d->flags;
	L.nst[e] = (uint8_t)d->nst;
	L.islot[e] = (uint8_t)d->islot;
	L.ity[e] = d->ity;
	L.name[e] = d->name;
	L.st[e] = s;
	L.cond[e] = d->cond;
	L.aux[e] = d->aux;
	if (d->kind == LK_FOR)
		bound |= 1ULL << d->islot;
	for (k = 0; k < d->nst; k++) {
		L.sslot[s + k] = (uint8_t)d->sslot[k];
		L.sname[s + k] = d->sname[k];
		L.sty[s + k] = d->sty[k];
		L.init[s + k] = d->init[k];
		L.next[s + k] = d->next[k];
		bound |= 1ULL << d->sslot[k];
		fr |= node_slots(d->init[k]) | node_slots(d->next[k]);
	}
	fr |= node_slots(d->cond) | node_slots(d->aux);
	L.bound[e] = bound;
	L.free[e] = fr & ~bound;
	L.ht[j] = e + 1;
	return e;
}

void
loop_get(uint32_t e, struct loopdef *d)
{
	uint32_t s = L.st[e];
	int k;

	memset(d, 0, sizeof *d);
	d->kind = L.kind[e];
	d->orig = L.orig[e];
	d->flags = L.flags[e];
	d->name = L.name[e];
	d->islot = L.islot[e];
	d->ity = L.ity[e];
	d->nst = L.nst[e];
	d->cond = L.cond[e];
	d->aux = L.aux[e];
	for (k = 0; k < d->nst; k++) {
		d->sslot[k] = L.sslot[s + k];
		d->sname[k] = L.sname[s + k];
		d->sty[k] = L.sty[s + k];
		d->init[k] = L.init[s + k];
		d->next[k] = L.next[s + k];
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
	ty_t t = L.sty[L.st[e] + res];
	uint32_t s = L.st[e];
	int fl = 0, k;
	nid n;
	int64_t vi;
	double vr;

	if (L.kind[e] == LK_FOR) {
		lo = mk_cast(TY_INT, lo);
		hi = mk_cast(TY_INT, hi);
		if (N.op[lo] == O_UNDEF || N.op[hi] == O_UNDEF)
			return mk_undef(t);
		fl |= (N.fl[lo] | N.fl[hi]) & NF_PARTIAL;
	} else {
		lo = hi = 0;
		fl |= NF_PARTIAL;   /* it may not terminate */
	}
	if (L.flags[e] & LF_NEGUNDEF)
		fl |= NF_PARTIAL;
	for (k = 0; k < L.nst[e]; k++)
		fl |= (N.fl[L.init[s + k]] | N.fl[L.next[s + k]]) & NF_PARTIAL;
	if (L.cond[e])
		fl |= N.fl[L.cond[e]] & NF_PARTIAL;
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
	uint32_t i = e * 2 + (ideal != 0), k, m = L.nst[e], s = L.st[e];
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
		roots[k] = L.init[s + k];
		roots[m + k] = L.next[s + k];
	}
	roots[2 * m] = L.cond[e];
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
	uint32_t k, s = L.st[e], m = L.nst[e];

	for (k = 0; k < m; k++) {
		size_t at = (size_t)L.sslot[s + k] * CH;
		if (TY_FP(L.sty[s + k]))
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
	uint32_t e = (uint32_t)key, l, j, m = L.nst[e], s = L.st[e];
	uint32_t res = (uint32_t)(key >> 32);
	int W = MAXP, isfor = L.kind[e] == LK_FOR;
	struct lpc *c = lpc_get(e, p->ideal);
	struct lbuf *B;
	int64_t *O = COLI(p, k), maxt = 0, it;
	double *OR = COLR(p, k);
	uint8_t *OF = COLF(p, k);
	size_t is = (size_t)L.islot[e] * CH;

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
			if (L.flags[e] & LF_NEGUNDEF)
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
	uint32_t e = loop_ent(n), s = L.st[e];
	int res = loop_res(n), k;

	switch (L.orig[e]) {
	case LO_SUM:
	case LO_PROD:
		sb_printf(b, "%s(%s, ", L.orig[e] == LO_SUM ? "sum" : "prod",
		    sym_str(L.name[e]));
		print_node(b, N.a[n], pc);
		sb_puts(b, ", ");
		print_node(b, N.b[n], pc);
		sb_puts(b, ", ");
		print_node(b, L.aux[e], pc);
		sb_putc(b, ')');
		return;
	case LO_REC:
		sb_printf(b, "%s[", sym_str(L.name[e]));
		print_node(b, N.c[n] ? N.c[n] : N.b[n], pc);
		sb_putc(b, ']');
		return;
	}
	sb_puts(b, "loop(");
	if (L.kind[e] == LK_FOR) {
		sb_printf(b, "%s in ", sym_str(L.name[e]));
		print_node(b, N.a[n], pc);
		sb_puts(b, "..");
		print_node(b, N.b[n], pc);
	} else {
		sb_puts(b, "while ");
		print_node(b, L.cond[e], pc);
	}
	for (k = 0; k < L.nst[e]; k++) {
		sb_printf(b, "; %s = ", sym_str(L.sname[s + k]));
		print_node(b, L.init[s + k], pc);
		sb_puts(b, " then ");
		print_node(b, L.next[s + k], pc);
	}
	sb_printf(b, ").%s", sym_str(L.sname[s + res]));
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
		if (map[j] && ((L.free[e] >> j) & 1)) {
			m2[j] = map[j];
			vals |= node_slots(map[j]);
			any = 1;
		}
	if (!any) {
		*ne = e;
		return;
	}
	for (j = 0; j < 64; j++)
		if ((L.bound[e] >> j) & 1) {
			if (j < lo)
				lo = j;
			if (j > hi)
				hi = j;
		}
	/* the arguments read slots this loop binds: move the loop up */
	if (vals & L.bound[e]) {
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
	uint32_t e = loop_ent(n), s = L.st[e], lim = N.n;
	uint8_t *mark = xcalloc(lim, 1);
	float c = 0;
	int k;

	for (k = 0; k < L.nst[e]; k++)
		c += body_cost(m, L.next[s + k], mark, lim);
	c += body_cost(m, L.cond[e], mark, lim);
	free(mark);
	return c + m->c[C_BRANCH];
}

float
loop_cost(const struct model *m, nid n, int *cls)
{
	float per = loop_iter_cost(m, n);
	int64_t lo, hi;

	*cls = C_BRANCH;
	if (L.kind[loop_ent(n)] == LK_FOR && int_of(N.a[n], &lo) &&
	    int_of(N.b[n], &hi)) {
		note("loop %s priced as %lld iterations x %g (listed under "
		    "branch)", node_str(n), (long long)(hi - lo + 1),
		    (double)per);
		return per * (float)(hi >= lo ? hi - lo + 1 : 0);
	}
	note("loop %s priced per iteration (%g, listed under branch); "
	    "multiply by the trip count; see 'scheme'", node_str(n),
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
		if ((L.bound[loop_ent(n)] >> j) & 1 && j > top)
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
