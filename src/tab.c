/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * tab.c: a function of few input bits as a table.
 *
 * find's closed forms already offer the plain lookup table for one
 * parameter of at most 256 values.  This is the general case: when the
 * parameters the function reads have, by their ranges and the where
 * clause, at most 2^16 points between them, the function is evaluated at
 * every one and becomes one load, the index packed from the parameters
 * (fields side by side when every range is a power of two, mixed radix
 * otherwise).  It is offered only where the active model prices the load
 * and the index below the function as written, and the proof is the
 * tabulation itself: entry k is the function at point k, every point
 * evaluated.  The check on the domain afterwards is exhaustive, since the
 * domain is that small.
 *
 * The price is the model's `load`, a cache-resident access; the report
 * gives the table's size so that it can be judged against that.
 */
#include "calc.h"
#include "fuse.h"
#include "stoch.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TAB_MAX 65536	/* 16 input bits */

/*
 * A table for part of the domain, the function computed elsewhere: x <
 * K ? T[x - lo] : f(x), for a function of one parameter whose range is too
 * large for a whole table.  T is f at every point of lo..K-1, each
 * evaluated, so the select is f everywhere (there by the table, past it
 * by f itself).  Its price is an expectation over the domain (each value
 * as likely): a select computes both sides; a branch on a uniform
 * parameter takes one side for the whole warp, the table's with the
 * probability p that x < K; on a parameter with a value in each lane,
 * a warp whose lanes split runs both and pays the divergence
 * (diverge_stats, as cost prices a ?:).  K is searched over the table
 * sizes 2^k up to TAB_MAX; the cheapest is offered when it is below the
 * function as written.
 */
static void
tab_split(const struct func *f)
{
	const struct model *m = model_active();
	const char *mn = sym_str(m->name);
	nid body = f->body, x, idxv, cond;
	ty_t t = N.ty[body];
	uint64_t slots;
	int64_t lo, hi, total, n, bestn = 0, in_i[MAXP * CH], undef = 0;
	double in_r[MAXP * CH], fcost, lcost, before, best = INFINITY;
	double bsel = 0, bp = 0, bsplit = 0;
	int j = -1, k, W = (int)m->par[MP_WARP], uni, bexact = 0;
	struct plan p;
	nid roots[1];

	if (!(TYK(t) == K_UINT || TYK(t) == K_SINT) || (N.fl[body] & NF_SG) ||
	    !cclass_available(m, C_LOAD) || f->where ||
	    N.op[body] == O_INDEX || is_const(body))
		return;
	slots = node_slots(body);
	if (!slots || (slots & (slots - 1)) || (slots >> MAXP))
		return;
	while (!((slots >> ++j) & 1))
		;
	if (j >= f->np || !TY_ISINT(f->p[j].ty))
		return;
	param_bounds(&f->p[j], &lo, &hi);
	if (hi < lo || (uint64_t)(hi - lo) < TAB_MAX || (uint64_t)(hi - lo) >=
	    (1ULL << 32))
		return;		/* a whole table, or more than 32 bits */
	total = hi - lo + 1;
	uni = f->p[j].uniform;
	before = fuse_expr_cost(m, body);
	x = mk_var((uint32_t)j, f->p[j].name, f->p[j].ty);
	idxv = lo ? mk_bin(O_SUB, x, mk_const_i(f->p[j].ty, lo)) : x;
	/* the table's side: a load, and the index's subtract */
	lcost = m->c[C_LOAD] + (lo ? m->c[C_IADD] : 0);
	fcost = before;
	for (n = 2; n <= TAB_MAX; n *= 2) {
		double p = (double)n / (double)total, dat, daf, cost, sel, dp;
		int exact = 0;
		uint64_t pts;
		cond = mk_bin(O_LT, x, mk_const_i(f->p[j].ty, lo + n));
		sel = m->c[C_CMP] + m->c[C_SELECT] + lcost + fcost;
		if (uni || W < 2) {
			/* the whole warp on one side */
			cost = m->c[C_CMP] + m->c[C_BRANCH] + p * lcost + (1 - p) *
			    fcost;
			dat = p;
			daf = 1 - p;
		} else {
			double reached;
			if (!diverge_stats(f, cond, 0, LOOP_NONE, W, &dp, &dat, &daf,
			    &reached, &exact, &pts))
				continue;
			cost = m->c[C_CMP] + m->c[C_BRANCH] + dat * lcost + daf *
			    fcost + (1 - dat - daf) * (lcost + fcost +
			    m->c[C_DIVERGE]);
		}
		if (sel < cost)
			cost = sel;
		if (cost < best) {
			best = cost;
			bestn = n;
			bsel = sel;
			bp = p;
			bsplit = 1 - dat - daf;
			bexact = exact;
		}
	}
	if (!(best < before) || bestn == 0)
		return;
	/* the table: f at every point of lo..lo+bestn-1 */
	roots[0] = body;
	plan_build(&p, roots, 1, 0);
	for (k = 0; k < bestn; k += CH) {
		uint32_t np = (uint32_t)(bestn - k < CH ? bestn - k : CH), i;
		memset(in_i, 0, sizeof in_i);
		memset(in_r, 0, sizeof in_r);
		for (i = 0; i < np; i++)
			in_i[j * CH + i] = lo + k + i;
		plan_run(&p, np, in_i, in_r);
		for (i = 0; i < np; i++)
			undef += (COLF(&p, p.root[0])[i] & F_UNDEF) != 0;
	}
	plan_free(&p);
	printf("  a table for part of the domain under %s (the function past it; "
	    "cost %g as written):\n", mn, before);
	printf("  cost %-3g %s < %lld ? T[%s] : %s   (expected; T the %lld "
	    "values at %s = %lld..%lld)\n", best, sym_str(f->p[j].name),
	    (long long)(lo + bestn), node_str(idxv), shared_str(body, 80),
	    (long long)bestn, sym_str(f->p[j].name), (long long)lo,
	    (long long)(lo + bestn - 1));
	printf("           entry i is the function at %s = ", sym_str(
	    f->p[j].name));
	if (lo)
		printf("%lld + ", (long long)lo);
	printf("i, all %lld evaluated%s; past them the function itself, so the "
	    "select is the function everywhere\n", (long long)bestn, undef ?
	    " (0 where it is undefined)" : "");
	if (uni || W < 2)
		printf("           %s is the same in every lane: the warp takes one "
		    "side, the table's with probability %.4g (each of the %lld "
		    "values as likely): compare and branch, then %g or %g\n",
		    sym_str(f->p[j].name), bp, (long long)total, lcost, fcost);
	else
		printf("           %s has a value in each lane: a warp of %d splits "
		    "with probability %.4g%s and then runs both sides and the "
		    "divergence\n", sym_str(f->p[j].name), W, bsplit, bexact ?
		    " (counted)" : "");
	printf("           as a select (both sides computed) %g; K searched over "
	    "the table sizes 2^k up to %d\n", bsel, TAB_MAX);
	rep_obj("table_part");
	rep_str("model", mn);
	rep_num("cost_before", before);
	rep_num("cost", best);
	rep_int("entries", bestn);
	rep_int("split", lo + bestn);
	rep_num("p_table", bp);
	rep_bool("uniform", uni || W < 2);
	rep_end();
}

void
tab_find(const struct func *f)
{
	const struct model *m = model_active();
	const char *mn = sym_str(m->name);
	int64_t blo[MAXP], bhi[MAXP], in_i[MAXP * CH];
	double in_r[MAXP * CH];
	uint64_t size[MAXP], total = 1, slots, k;
	int rd[MAXP], nrd = 0, j, pow2 = 1, bytes;
	nid body = f->body, roots[2], *e, idx = 0, cand, where = f->where;
	ty_t t = N.ty[body];
	struct plan p;
	struct eqproof pr;
	struct sbuf w = {0}, dom = {0}, ix = {0};
	float before, cost;
	uint64_t undef = 0;

	tab_split(f);
	if (!(TYK(t) == K_UINT || TYK(t) == K_SINT) || (N.fl[body] & NF_SG) ||
	    !cclass_available(m, C_LOAD))
		return;
	/* already a table read, or nothing to it */
	if (N.op[body] == O_INDEX || N.op[body] == O_VAR || is_const(body) ||
	    (N.op[body] == O_CAST && N.op[N.a[body]] == O_INDEX))
		return;
	slots = node_slots(body) | (where ? node_slots(where) : 0);
	if (slots >> MAXP)
		return;		/* a free loop variable: not a function of its parameters alone */
	for (j = 0; j < f->np; j++) {
		if (!((slots >> j) & 1))
			continue;
		if (!TY_ISINT(f->p[j].ty) || (TYK(f->p[j].ty) == K_UINT &&
		    TYW(f->p[j].ty) == 64 && !f->p[j].ranged))
			return;
		param_bounds(&f->p[j], &blo[j], &bhi[j]);
	}
	where_bounds(f, blo, bhi, NULL);
	for (j = 0; j < f->np; j++) {
		if (!((slots >> j) & 1))
			continue;
		if (bhi[j] < blo[j] || (uint64_t)(bhi[j] - blo[j]) >= TAB_MAX)
			return;
		size[j] = (uint64_t)(bhi[j] - blo[j]) + 1;
		total *= size[j];
		if (total > TAB_MAX)
			return;
		if (size[j] & (size[j] - 1))
			pow2 = 0;
		rd[nrd++] = j;
	}
	/* one parameter of at most 256 values is find's own baseline */
	if (total < 2 || nrd == 0 || (nrd == 1 && total <= 256))
		return;
	before = fuse_expr_cost(m, body);

	/* the index: parameter rd[0] in the low part */
	{
		uint64_t stride = 1;
		int sh = 0;
		for (j = 0; j < nrd; j++) {
			int s = rd[j];
			nid x = mk_var((uint32_t)s, f->p[s].name, f->p[s].ty), part;
			if (blo[s] > 0)
				x = mk_bin(O_SUB, x, mk_const_i(N.ty[x], blo[s]));
			else if (blo[s] < 0)
				x = mk_bin(O_ADD, x, mk_const_i(N.ty[x], -blo[s]));
			x = mk_cast(TY(K_UINT, 32), x);
			if (pow2)
				part = sh ? mk_bin(O_SHL, x, mk_const_i(TY(K_UINT, 32),
				    sh)) : x;
			else
				part = stride > 1 ? mk_bin(O_MUL, x, mk_const_i(
				    TY(K_UINT, 32), (int64_t)stride)) : x;
			idx = !idx ? part : pow2 ? mk_bin(O_OR, idx, part) :
			    mk_bin(O_ADD, idx, part);
			sb_printf(&dom, "%s%s in %lld..%lld", j ? ", " : "",
			    sym_str(f->p[s].name), (long long)blo[s],
			    (long long)bhi[s]);
			stride *= size[s];
			while (pow2 && (1ULL << sh) < stride)
				sh++;
		}
	}
	/* a quick bound before the work: the load and the index alone */
	if (!(m->c[C_LOAD] < before)) {
		free(dom.p);
		return;
	}

	/* the function at every point, in index order */
	roots[0] = body;
	roots[1] = where;
	plan_build(&p, roots, 2, 0);
	e = xmalloc((size_t)total * sizeof *e);
	for (k = 0; k < total; k += CH) {
		uint32_t np = (uint32_t)(total - k < CH ? total - k : CH), i;
		uint8_t ok[CH];
		memset(in_i, 0, sizeof in_i);
		memset(in_r, 0, sizeof in_r);
		for (i = 0; i < np; i++) {
			uint64_t q = k + i;
			for (j = 0; j < nrd; j++) {
				in_i[rd[j] * CH + i] = blo[rd[j]] + (int64_t)(q % size[rd[j]]);
				q /= size[rd[j]];
			}
		}
		plan_run(&p, np, in_i, in_r);
		valid_mask(&p, where ? p.root[1] : NOTOK, np, ok);
		for (i = 0; i < np; i++) {
			int bad = !ok[i] || (COLF(&p, p.root[0])[i] & F_UNDEF);
			undef += bad;
			e[k + i] = mk_const_i(t, bad ? 0 : COLI(&p, p.root[0])[i]);
		}
	}
	plan_free(&p);
	cand = mk_cast(t, mk_index(mk_table(e, (uint32_t)total), idx));
	free(e);
	cost = fuse_expr_cost(m, cand);
	if (!(cost < before) || undef == total)
		goto out;
	if (equiv_decide(f, body, cand, &w, &pr)) {
		printf("  a table under %s: (cost %g) does not hold: %s\n", mn,
		    (double)cost, sb_str(&w));
		goto out;
	}
	bytes = (TYW(t) + 7) / 8;
	sb_puts(&ix, node_str(idx));
	printf("  a table under %s (the function at every point of its domain; "
	    "cost %g as written):\n", mn, (double)before);
	printf("  cost %-3g %s\n", (double)cost, shared_str(cand, 160));
	printf("           %llu entries of %s (%llu bytes of constant data), one "
	    "for each point of %s; entry k is the function at the point with "
	    "index k = %s, all %llu evaluated%s\n", (unsigned long long)total,
	    ty_str(t), (unsigned long long)total * (unsigned)bytes, sb_str(&dom),
	    sb_str(&ix), (unsigned long long)total, undef ?
	    " (0 where the function is undefined or the where clause fails)" :
	    "");
	printf("           priced as %s's load, a cache-resident access: the "
	    "table has to stay in cache for that\n", mn);
	printf("           %s\n", pr.how == EQ_EXHAUSTIVE ?
	    "checked on every point of the domain" : pr.how == EQ_SAT ?
	    "checked by SAT" : "checked on a sample of the domain");
	rep_obj("table");
	rep_str("model", mn);
	rep_num("cost_before", before);
	rep_num("cost", cost);
	rep_int("entries", (long long)total);
	rep_int("bytes", (long long)total * bytes);
	rep_str("index", sb_str(&ix));
	rep_str("domain", sb_str(&dom));
	rep_str("check", pr.how == EQ_EXHAUSTIVE ? "exhaustive" : pr.how ==
	    EQ_SAT ? "sat" : "sampled");
	rep_end();
out:
	free(w.p);
	free(dom.p);
	free(ix.p);
}
