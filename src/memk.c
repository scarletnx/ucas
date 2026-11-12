/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * memk: the memory side of a cost model.
 *
 * The memory kinds (shared loads and stores, block reads at a uniform or
 * a divergent address, streams, the atomic kinds, texture reads) are cost
 * classes; a model that lists none of them prices each as its fallback,
 * so nothing here fires under it.  Two rewrites come out of the prices:
 *
 * banks.  Workgroup memory is `banks` banks of `bank_bytes` bytes; the
 * lanes of a warp that reach different addresses on one bank are served
 * one after the other, each after the first at `bank_conflict`.  The
 * conflict degree of an index expression is the largest number of
 * distinct addresses on one bank over the `warp` consecutive values of
 * its lane parameter: gcd arithmetic when the index is affine in the
 * lane, an enumeration of every warp and every value of the uniform
 * parameters otherwise.  A swizzle of the index (the row number xored or
 * added into the bank bits, a padded row) is searched over the accesses
 * of the array together; the one that makes their total price lowest is
 * taken when it beats the price as written, and it is proved to keep the
 * indices apart (an involution, or strictly increasing: by enumeration
 * or SAT), so every read still finds what a write put there.
 *
 * intatomic.  A float atomic add where the model prices it above an
 * integer one (a compare-and-swap loop) becomes an integer atomic add of
 * the addend scaled to fixed point, when the addend is proved to lie on a
 * grid k 2^g with its f32 evaluation exact (from the structure of the
 * expression and its ranges) and the sum of `count` addends fits 32
 * bits; with `tol` a rounded scaling is taken and its error derived.
 */
#include "memk.h"
#include "nodesat.h"
#include "fuse.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static const int fallback[C_NCLASS] = {
	[C_LOAD_SHARED] = C_LOAD, [C_STORE_SHARED] = C_STORE,
	[C_LOAD_UBO] = C_LOAD, [C_LOAD_UBO_DIV] = C_LOAD_UBO,
	[C_LOAD_STREAM] = C_LOAD, [C_STORE_STREAM] = C_STORE,
	[C_BANK_CONFLICT] = -1,
	[C_ATOM_SH_SAME] = C_ATOM_SH, [C_ATOM_SH_F] = C_ATOM_SH,
	[C_ATOM_GL_F] = C_ATOM_GL, [C_ATOM_GL_64] = C_ATOM_GL,
	[C_ATOM_GL_CAS] = C_ATOM_GL, [C_ATOM_GL_STREAM] = C_ATOM_GL,
	[C_TEX_FETCH] = C_LOAD, [C_TEX_SAMPLE] = C_LOAD,
	[C_TEX_SAMPLE_F32] = C_TEX_SAMPLE, [C_IMAGE_LOAD] = C_LOAD,
};

int
model_lists(const struct model *m, int cls)
{
	return cls < C_MEM_FIRST || cls > C_MEM_LAST || !isnan(m->c[cls]);
}

/* the class a price comes from: cls, or the fallback m lists */
int
model_class(const struct model *m, int cls)
{
	while (cls >= C_MEM_FIRST && cls <= C_MEM_LAST && isnan(m->c[cls]) &&
	    fallback[cls] >= 0)
		cls = fallback[cls];
	return cls;
}

float
model_cost(const struct model *m, int cls)
{
	while (cls >= C_MEM_FIRST && cls <= C_MEM_LAST && isnan(m->c[cls])) {
		cls = fallback[cls];
		if (cls < 0)
			return 0;	/* no price: the model has no such effect */
	}
	return m->c[cls];
}

/* ---- bank conflicts ---- */

struct mrun {
	struct plan p;
	int64_t *in_i;
	double *in_r;
};

/* the lane parameter: the first integer parameter that is not uniform */
static int
lane_slot(const struct func *f, int *others)
{
	int j, lane = -1;

	*others = 0;
	for (j = 0; j < f->np; j++) {
		if (f->p[j].uniform)
			continue;
		if (lane < 0 && TY_ISINT(f->p[j].ty))
			lane = j;
		else
			(*others)++;
	}
	return lane;
}

static int64_t
gcd64(int64_t a, int64_t b)
{
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	while (b) {
		int64_t t = a % b;
		a = b;
		b = t;
	}
	return a;
}

/* the degree of 32 indices: distinct addresses on the fullest bank */
static int
degree_of(const int64_t *idx, int n, int elem, int bbytes, int banks)
{
	int i, j, best = 1;

	for (i = 0; i < n; i++) {
		int64_t bi = (idx[i] * elem / bbytes) % banks, cnt = 1;
		for (j = 0; j < i; j++)
			if (idx[j] == idx[i])
				break;
		if (j < i)
			continue;	/* the same address: served together */
		for (j = i + 1; j < n; j++) {
			int k;
			if ((idx[j] * elem / bbytes) % banks != bi || idx[j] == idx[i])
				continue;
			for (k = i + 1; k < j; k++)
				if (idx[k] == idx[j])
					break;
			if (k == j)
				cnt++;
		}
		if (cnt > best)
			best = (int)cnt;
	}
	return best;
}

#define BANK_MAXCASES (1u << 16)

int
bank_degree(const struct func *f, nid e, const struct model *m, int elem,
    struct bankres *r)
{
	int banks = (int)m->par[MP_BANKS], bbytes = (int)m->par[MP_BANK_BYTES];
	int warp = (int)m->par[MP_WARP], others, lane, j, np, w;
	int64_t lo[MAXP], hi[MAXP], cur[MAXP], idx[CH], llo, lhi;
	uint64_t total = 1, c, wins;
	struct plan p;
	nid roots[2];
	int64_t *in_i;
	double *in_r;
	uint32_t loc;
	uint64_t rng = 0x9e3779b97f4a7c15ULL;
	int affine = 1, first = 1;
	int64_t stride = 0;

	memset(r, 0, sizeof *r);
	r->degree = 1;
	if (banks < 1 || warp < 1)
		return 0;
	if (bbytes < 1)
		bbytes = 4;
	if (warp > CH)
		warp = CH;	/* a warp is one chunk of the evaluator */
	lane = lane_slot(f, &others);
	if (lane < 0 || others) {
		r->degree = warp < banks ? warp : banks;	/* nothing known */
		return -1;
	}
	for (j = 0; j < f->np; j++) {
		if (!TY_ISINT(f->p[j].ty))
			return -1;
		param_bounds(&f->p[j], &lo[j], &hi[j]);
		if (hi[j] < lo[j])
			return -1;
	}
	llo = lo[lane];
	lhi = hi[lane];
	/* warps are aligned windows of the lane's values */
	wins = (uint64_t)((lhi / warp) - (llo / warp) + 1);
	for (j = 0; j < f->np; j++)
		if (j != lane) {
			uint64_t sz = (uint64_t)(hi[j] - lo[j]) + 1;
			total = sz > BANK_MAXCASES || total * sz > BANK_MAXCASES ?
			    BANK_MAXCASES + 1 : total * sz;
		}
	total = total > BANK_MAXCASES || total * wins > BANK_MAXCASES ?
	    BANK_MAXCASES + 1 : total * wins;
	r->exact = total <= BANK_MAXCASES;
	roots[0] = e;
	roots[1] = f->where;
	plan_build(&p, roots, 2, 0);
	loc = plan_loc(&p, e);
	in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	for (j = 0; j < f->np; j++)
		cur[j] = lo[j];
	cur[lane] = llo / warp;	/* the window */
	for (c = 0; c < (r->exact ? total : 4096); c++) {
		int64_t base;
		if (!r->exact) {
			for (j = 0; j < f->np; j++) {
				uint64_t span;
				rng ^= rng << 13;
				rng ^= rng >> 7;
				rng ^= rng << 17;
				if (j == lane) {
					cur[j] = llo / warp + (int64_t)(rng % wins);
					continue;
				}
				span = (uint64_t)(hi[j] - lo[j]) + 1;
				cur[j] = lo[j] + (int64_t)(span ? rng % span : rng);
			}
		}
		base = cur[lane] * warp;
		np = 0;
		for (w = 0; w < warp; w++) {
			int64_t l = base + w;
			if (l < llo || l > lhi)
				continue;
			for (j = 0; j < f->np; j++)
				in_i[(size_t)j * CH + np] = j == lane ? l : cur[j];
			np++;
		}
		if (np == 0) {
			r->cases++;
			goto next;
		}
		plan_run(&p, (uint32_t)np, in_i, in_r);
		for (w = 0; w < np; w++)
			idx[w] = COLI(&p, loc)[w];
		{
			int d = degree_of(idx, np, elem, bbytes, banks);
			if (d > r->degree)
				r->degree = d;
		}
		if (affine) {
			for (w = 2; w < np; w++)
				if (idx[w] - idx[w - 1] != idx[1] - idx[0])
					affine = 0;
			if (affine && np >= 2) {
				int64_t st = idx[1] - idx[0];
				if (first) {
					stride = st;
					first = 0;
				} else if (st != stride)
					affine = 0;	/* a stride per case: enumerated */
			}
		}
		r->cases++;
next:
		if (r->exact) {
			/* the next case: an odometer over the parameters */
			for (j = 0; j < f->np; j++) {
				int64_t top = j == lane ? lhi / warp : hi[j];
				if (cur[j] < top) {
					cur[j]++;
					break;
				}
				cur[j] = j == lane ? llo / warp : lo[j];
			}
		}
	}
	r->affine = affine && !first;
	r->stride = stride;
	plan_free(&p);
	free(in_i);
	free(in_r);
	return 1;
}

double
bank_price(const struct model *m, int store, int degree)
{
	return (double)model_cost(m, store ? C_STORE_SHARED : C_LOAD_SHARED) +
	    (degree - 1) * (double)model_cost(m, C_BANK_CONFLICT);
}

/* ---- swizzles ---- */

struct swz {
	int kind, s;	/* the row is i >> s */
};

static nid
ku(ty_t t, int64_t v)
{
	return mk_const_i(t, v);
}

/* the swizzled index: i's row number xored or added into its bank bits,
 * or i with one element of padding after every 2^s */
static nid
swz_apply(const struct swz *z, nid i, int banks)
{
	ty_t t = N.ty[i];
	nid row = mk_bin(O_SHR, i, ku(t, z->s));
	nid mask = ku(t, banks - 1);

	switch (z->kind) {
	case SW_XOR:
		return mk_bin(O_XOR, i, mk_bin(O_AND, row, mask));
	case SW_ADD:
		return mk_bin(O_OR, mk_bin(O_AND, i, ku(t, ~(int64_t)(banks - 1))),
		    mk_bin(O_AND, mk_bin(O_ADD, i, row), mask));
	}
	return mk_bin(O_ADD, i, row);
}

static void
swz_text(const struct swz *z, int banks, char *b, size_t n)
{
	switch (z->kind) {
	case SW_XOR:
		snprintf(b, n, "i xor (i >> %d & %d)", z->s, banks - 1);
		break;
	case SW_ADD:
		snprintf(b, n, "(i & ~%d) | (i + (i >> %d) & %d)", banks - 1, z->s,
		    banks - 1);
		break;
	default:
		snprintf(b, n, "i + (i >> %d)", z->s);
	}
}

/* what root costs beyond stop (0: nothing), under the model's cover
 * with its fusions */
static double
expr_cost_under(const struct model *m, nid root, nid stop)
{
	return (double)fuse_expr_cost(m, root) - (stop ? (double)
	    fuse_expr_cost(m, stop) : 0);
}

/*
 * The swizzle keeps indices apart on [0, size): xor and add are proved
 * involutions or inverted (so bijections of each row of `banks`
 * elements), the padding strictly increasing.  *how: EQ_*.
 */
static int
swz_prove(const struct swz *z, int banks, int64_t size, int *how,
    struct sbuf *text)
{
	struct func t;
	struct sbuf w = {0};
	struct eqproof pr;
	ty_t u = TY(K_UINT, 32);
	nid i, a, b;
	int bad;

	memset(&t, 0, sizeof t);
	t.name = sym_cstr("_swz");
	t.anon = 1;
	t.np = 1;
	t.p[0].name = sym_cstr("i");
	t.p[0].ty = u;
	t.p[0].ranged = 1;
	t.p[0].lo = 0;
	t.p[0].hi = size - 1;
	i = mk_var(0, t.p[0].name, u);
	if (z->kind == SW_XOR) {
		/* applied twice it is i: the row bits are not touched */
		a = i;
		b = swz_apply(z, swz_apply(z, i, banks), banks);
		sb_printf(text, "applied twice it gives i back (the row i >> %d "
		    "is not changed by it), so it is a bijection of [0, %lld)",
		    z->s, (long long)size);
	} else if (z->kind == SW_ADD) {
		/* undone by subtracting the row from the bank bits */
		nid y = swz_apply(z, i, banks);
		nid row = mk_bin(O_SHR, y, ku(u, z->s));
		a = i;
		b = mk_bin(O_OR, mk_bin(O_AND, y, ku(u, ~(int64_t)(banks - 1))),
		    mk_bin(O_AND, mk_bin(O_SUB, y, row), ku(u, banks - 1)));
		sb_printf(text, "subtracting the row from the bank bits gives i "
		    "back, so it is a bijection of [0, %lld)", (long long)size);
	} else {
		/* i + 1 lands after i */
		t.p[0].hi = size - 2;
		a = mk_const_i(TY_BOOL, 1);
		b = mk_bin(O_GT, swz_apply(z, mk_bin(O_ADD, i, ku(u, 1)), banks),
		    swz_apply(z, i, banks));
		sb_printf(text, "it is strictly increasing (i + 1 lands after "
		    "i), so no two indices meet; the array grows to %lld "
		    "elements", (long long)(size + ((size - 1) >> z->s)));
	}
	t.body = a;
	if (size < 2) {
		*how = EQ_EXHAUSTIVE;
		free(w.p);
		return 1;
	}
	bad = equiv_decide(&t, a, b, &w, &pr);
	*how = pr.how;
	free(w.p);
	return !bad && pr.how != EQ_SAMPLED;
}

#define MAXACC 32

static const char *
how_text(int how)
{
	return how == EQ_SAT ? "proved by SAT" : how == EQ_EXHAUSTIVE ?
	    "every index enumerated" : "sampled only: NOT proved";
}

static void
degree_text(const struct bankres *r, int banks, int warp, int same,
    char *b, size_t n)
{
	int64_t g = gcd64(r->stride, banks);
	int64_t d = g ? ((int64_t)warp * g + banks - 1) / banks : 1;

	if (r->affine && r->exact && r->stride != 0 && same && d == r->degree)
		snprintf(b, n, "index = %lld lane + b: two lanes share a bank "
		    "when %d divides %lld (l - l'), that is when %lld divides l - "
		    "l', so %d gcd(%lld, %d) / %d = %lld lane%s a bank; all "
		    "%llu cases enumerated too", (long long)r->stride, banks,
		    (long long)r->stride, (long long)(banks / g), warp,
		    (long long)r->stride, banks, banks, (long long)d,
		    d == 1 ? "" : "s", (unsigned long long)r->cases);
	else if (r->exact)
		snprintf(b, n, "every warp and every value of the uniform "
		    "parameters enumerated: %llu cases",
		    (unsigned long long)r->cases);
	else
		snprintf(b, n, "sampled: %llu of the cases, the degree is at "
		    "least this (not a bound)", (unsigned long long)r->cases);
}

int
memk_banks(const uint32_t *names, int nf, int64_t size, int elem,
    int64_t row, double extra, const struct model *m, struct bankswz *out)
{
	struct func *fs[MAXACC];
	int same = elem == (int)m->par[MP_BANK_BYTES];
	int banks = (int)m->par[MP_BANKS], warp = (int)m->par[MP_WARP];
	struct bankres r0[MAXACC], r1[MAXACC];
	struct swz best = {0, 0}, z;
	double cost0 = 0, bestc, cconf = (double)model_cost(m, C_BANK_CONFLICT);
	int k, kind, s, found = 0, allexact = 1, how, smax;
	int64_t maxidx = 0;
	char tb[400];

	for (k = 0; k < nf; k++)
		fs[k] = func_find(names[k]);
	rep_begin("banks");
	rep_str("model", sym_str(m->name));
	if (banks < 1 || warp < 1) {
		printf("banks: model %s declares no banks (banks: N, bank_bytes: "
		    "B, warp: W in its block): shared memory has no bank "
		    "conflicts under it, nothing to do\n", sym_str(m->name));
		rep_bool("rewritten", 0);
		return 0;
	}
	printf("banks under %s: %d banks of %d bytes, a warp of %d lanes, "
	    "%d-byte elements; a shared load %g, each further address on a "
	    "bank %+g\n", sym_str(m->name), banks, (int)m->par[MP_BANK_BYTES],
	    warp, elem, (double)model_cost(m, C_LOAD_SHARED), cconf);
	rep_list("accesses");
	for (k = 0; k < nf; k++) {
		struct func *f = fs[k];
		int others, lane = lane_slot(f, &others), rc;
		rc = bank_degree(f, f->body, m, elem, &r0[k]);
		printf("  %s = %s\n", f->anon ? "index" : sym_str(f->name),
		    shared_str(f->body, 200));
		if (rc < 0) {
			printf("    not analysed: it needs one integer lane "
			    "parameter (not `uniform`), the others uniform "
			    "integers; priced at the worst case, degree %d\n",
			    r0[k].degree);
			allexact = 0;
		} else {
			degree_text(&r0[k], banks, warp, same, tb, sizeof tb);
			printf("    lane parameter %s; conflict degree %d: %s\n",
			    sym_str(f->p[lane].name), r0[k].degree, tb);
			if (!r0[k].exact)
				allexact = 0;
		}
		printf("    price %g\n", bank_price(m, 0, r0[k].degree));
		cost0 += bank_price(m, 0, r0[k].degree);
		rep_obj(NULL);
		rep_expr("index", f->body);
		rep_int("degree", r0[k].degree);
		rep_bool("exact", r0[k].exact);
		rep_num("price", bank_price(m, 0, r0[k].degree));
		rep_end();
		/* the largest index, for the array's size */
		{
			struct plan p;
			struct ival *iv;
			nid roots[1];
			roots[0] = f->body;
			plan_build(&p, roots, 1, 1);
			iv = xcalloc(p.n, sizeof *iv);
			plan_ival(&p, f, iv);
			if (!iv[plan_loc(&p, f->body)].top &&
			    iv[plan_loc(&p, f->body)].hi > maxidx)
				maxidx = iv[plan_loc(&p, f->body)].hi;
			free(iv);
			plan_free(&p);
		}
	}
	rep_end();
	if (size > 0 && maxidx >= size)
		fail("banks: an index reaches %lld, past size %lld",
		    (long long)maxidx, (long long)size);
	if (size <= 0) {
		/* whole rows of `banks` elements */
		size = (maxidx / banks + 1) * banks;
		printf("  array size taken as %lld (the largest index %lld, in "
		    "whole rows of %d; `size N` to say)\n", (long long)size,
		    (long long)maxidx, banks);
	}
	rep_num("price", cost0);
	if (!allexact) {
		printf("  -> no swizzle searched: a degree above is not exact "
		    "(give the parameters ranges small enough to enumerate)\n");
		rep_bool("rewritten", 0);
		return 0;
	}
	if (banks & (banks - 1)) {
		printf("  -> no swizzle searched: %d banks is not a power of "
		    "two\n", banks);
		rep_bool("rewritten", 0);
		return 0;
	}
	bestc = cost0;
	for (smax = 0; smax < 30 && ((int64_t)1 << smax) < size; smax++)
		;
	for (kind = SW_XOR; kind <= SW_PAD; kind++)
		for (s = 1; s <= smax; s++) {
			double c = 0;
			int ok = 1;
			/* xor and add need the row above the bank bits; a padded
			 * row is the array's own when it has rows */
			if (kind != SW_PAD && ((1 << s) < banks || size % banks))
				continue;
			if (kind == SW_PAD && row > 0 && ((int64_t)1 << s) != row)
				continue;
			z.kind = kind;
			z.s = s;
			for (k = 0; k < nf && ok; k++) {
				nid e = swz_apply(&z, mk_cast(TY(K_UINT, 32),
				    fs[k]->body), banks);
				if (bank_degree(fs[k], e, m, elem, &r1[k]) < 0 ||
				    !r1[k].exact)
					ok = 0;
				c += bank_price(m, 0, r1[k].degree) +
				    expr_cost_under(m, e, fs[k]->body) + extra;
			}
			if (ok && c < bestc - 1e-9) {
				bestc = c;
				best = z;
				found = 1;
			}
		}
	if (!found) {
		printf("  -> not rewritten: no swizzle (row xored or added into "
		    "the bank bits, a padded row) makes the %d access%s cheaper "
		    "than %g under %s\n", nf, nf == 1 ? "" : "es", cost0,
		    sym_str(m->name));
		rep_bool("rewritten", 0);
		return 0;
	}
	{
		struct sbuf pt = {0};
		int proved = swz_prove(&best, banks, size, &how, &pt);
		swz_text(&best, banks, tb, sizeof tb);
		printf("  -> swizzle every index i of the array: %s\n", tb);
		printf("     %s: %s\n", sb_str(&pt), how_text(how));
		free(pt.p);
		if (!proved) {
			printf("     not applied: that step is not proved\n");
			rep_bool("rewritten", 0);
			return 0;
		}
		rep_str("swizzle", tb);
		rep_num("price_after", bestc);
		rep_list("rewritten_accesses");
		for (k = 0; k < nf; k++) {
			struct func *fk = func_find(names[k]), copy = *fk, *h;
			char nm[96], db[400];
			nid e = swz_apply(&best, mk_cast(TY(K_UINT, 32), fk->body),
			    banks), src = fk->body;
			bank_degree(fk, e, m, elem, &r1[k]);
			degree_text(&r1[k], banks, warp, same, db, sizeof db);
			if (fk->anon)
				snprintf(nm, sizeof nm, "_sw%d", k + 1);
			else
				snprintf(nm, sizeof nm, "%s_sw", sym_str(fk->name));
			printf("     %s = %s\n       conflict degree %d (%s): "
			    "price %g + %g for the swizzle\n", nm,
			    shared_str(e, 200), r1[k].degree, db,
			    bank_price(m, 0, r1[k].degree), expr_cost_under(m, e,
			    src) + extra);
			rep_obj(NULL);
			rep_str("defined", nm);
			rep_expr("index", e);
			rep_int("degree", r1[k].degree);
			rep_end();
			copy.name = sym_cstr(nm);
			copy.anon = 0;
			copy.body = e;
			h = func_new(copy.name);
			*h = copy;
		}
		rep_end();
		printf("     total %g against %g (%+.1f%%); every access of the "
		    "array takes the swizzle, reads and writes alike\n", bestc,
		    cost0, 100 * (bestc - cost0) / cost0);
		rep_bool("rewritten", 1);
		rep_proof(how == EQ_SAT ? "sat" : "exhaustive", "the swizzle "
		    "keeps the indices apart; the degrees are enumerated over "
		    "every warp");
		if (out) {
			out->kind = best.kind;
			out->s = best.s;
			out->size = best.kind == SW_PAD ? size + ((size - 1) >>
			    best.s) : size;
		}
	}
	return 1;
}

/* ---- fixed point ---- */

struct grid {
	int ok, g;		/* values are k 2^g */
	double mag;		/* |value| <= mag */
	int zero;		/* identically 0 */
};

static struct grid
grid_of(const struct func *f, nid n, struct sbuf *why, int depth)
{
	struct grid r = {0, 0, 0, 0}, a, b;
	ty_t t = N.ty[n];

	if (depth > 40)
		return r;
	switch (N.op[n]) {
	case O_CONST: {
		double v = TY_FP(t) ? N.kr[n] : (double)N.ki[n];
		int e;
		if (isnan(v) || isinf(v))
			return r;
		r.ok = 1;
		r.mag = fabs(v);
		if (v == 0) {
			r.zero = 1;
			r.g = 0;
			return r;
		}
		/* the lowest set bit of v */
		{
			double mnt = frexp(fabs(v), &e);
			int k = 0;
			while (mnt != floor(mnt) && k < 64) {
				mnt *= 2;
				k++;
			}
			r.g = e - k;
		}
		return r;
	}
	case O_VAR:
		if (!TY_ISINT(t) || N.ki[n] < 0 || N.ki[n] >= f->np)
			break;
		{
			int64_t lo, hi;
			param_bounds(&f->p[N.ki[n]], &lo, &hi);
			r.ok = 1;
			r.g = 0;
			r.mag = fabs((double)lo) > fabs((double)hi) ?
			    fabs((double)lo) : fabs((double)hi);
			return r;
		}
	case O_CAST:
		a = grid_of(f, N.a[n], why, depth + 1);
		if (!a.ok)
			break;
		if (TY_FP(t) || a.g >= 0)
			r = a;	/* exact while it fits 24 bits: checked below */
		break;
	case O_NEG: case O_ABS:
		return grid_of(f, N.a[n], why, depth + 1);
	case O_ADD: case O_SUB:
		a = grid_of(f, N.a[n], why, depth + 1);
		b = grid_of(f, N.b[n], why, depth + 1);
		if (!a.ok || !b.ok)
			break;
		r.ok = 1;
		r.g = a.zero ? b.g : b.zero ? a.g : a.g < b.g ? a.g : b.g;
		r.mag = a.mag + b.mag;
		r.zero = a.zero && b.zero;
		break;
	case O_MUL:
		a = grid_of(f, N.a[n], why, depth + 1);
		b = grid_of(f, N.b[n], why, depth + 1);
		if (!a.ok || !b.ok)
			break;
		r.ok = 1;
		r.g = a.g + b.g;
		r.mag = a.mag * b.mag;
		r.zero = a.zero || b.zero;
		break;
	case O_DIV:
		a = grid_of(f, N.a[n], why, depth + 1);
		if (!a.ok || N.op[N.b[n]] != O_CONST || !TY_FP(t))
			break;
		{
			double d = fabs(N.kr[N.b[n]]);
			int e;
			if (d == 0 || frexp(d, &e) != 0.5)
				break;	/* not a power of two */
			r = a;
			r.g = a.g - (e - 1);
			r.mag = a.mag / d;
		}
		break;
	case O_SHL:
		a = grid_of(f, N.a[n], why, depth + 1);
		if (!a.ok || N.op[N.b[n]] != O_CONST)
			break;
		r = a;
		r.g = a.g + (int)N.ki[N.b[n]];
		r.mag = ldexp(a.mag, (int)N.ki[N.b[n]]);
		break;
	case O_AND: case O_OR: case O_XOR: case O_SHR: case O_MOD: case O_MIN:
	case O_MAX: case O_INDEX: case O_SEL:
		if (!TY_ISINT(t) || !TY_SIZED(t))
			break;
		{
			struct ity it;
			ity_of(&it, t, 0);
			r.ok = 1;
			r.g = 0;
			r.mag = TYK(t) == K_SINT ? -(double)it.smin : (double)it.mask;
			if (N.op[n] == O_AND && N.op[N.b[n]] == O_CONST &&
			    N.ki[N.b[n]] >= 0)
				r.mag = (double)N.ki[N.b[n]];
			return r;
		}
	}
	if (!r.ok && why && why->n == 0)
		sb_printf(why, "%s is not on a binary grid (only integers, "
		    "binary constants, + - * and powers of two keep one)",
		    shared_str(n, 60));
	/* a float evaluation rounds past 24 bits of the grid */
	if (r.ok && TY_FP(t) && TYK(t) == K_FLOAT && TYW(t) == 32 && !r.zero &&
	    r.mag / ldexp(1.0, r.g) > 16777216.0) {
		if (why && why->n == 0)
			sb_printf(why, "%s reaches %g on the grid 2^%d: more than "
			    "24 bits, its f32 value rounds", shared_str(n, 60),
			    r.mag, r.g);
		r.ok = 0;
	}
	return r;
}

int
fix_grid(const struct func *f, nid e, int *g, double *kmax, struct sbuf *why)
{
	struct grid r = grid_of(f, e, why, 0);

	if (!r.ok)
		return 0;
	*g = r.g;
	*kmax = r.zero ? 0 : r.mag / ldexp(1.0, r.g);
	return 1;
}

/* the float atomic add under root, or 0 */
static nid
find_fatomic(nid root)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc((size_t)N.n * sizeof *st), n, hit = 0;
	uint32_t sp = 0;

	st[sp++] = root;
	mark[root] = 1;
	while (sp && !hit) {
		n = st[--sp];
		if (N.op[n] == O_ATOMIC && TY_FP(N.ty[n]) && (N.ki[n] & 15) ==
		    AT_ADD)
			hit = n;
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF
		    || N.op[n] == O_TABLE)
			continue;
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
	}
	free(mark);
	free(st);
	return hit;
}

/* an integer expression equal to e 2^-g, when e's structure gives one:
 * the conversions dropped and the constants scaled; 0 otherwise */
static nid
int_form(nid e, int g, ty_t it)
{
	nid a, b;

	switch (N.op[e]) {
	case O_CONST: {
		double v = (TY_FP(N.ty[e]) ? N.kr[e] : (double)N.ki[e]) *
		    ldexp(1.0, -g);
		if (v != floor(v) || fabs(v) > 2147483647.0)
			return 0;
		return mk_const_i(it, (int64_t)v);
	}
	case O_VAR:
		return g == 0 && TY_ISINT(N.ty[e]) ? mk_cast(it, e) : 0;
	case O_CAST:
		if (TY_ISINT(N.ty[N.a[e]]))
			return g == 0 ? mk_cast(it, N.a[e]) : g < 0 ? 0 :
			    mk_bin(O_SHR, mk_cast(it, N.a[e]), mk_const_i(it, g));
		return int_form(N.a[e], g, it);
	case O_NEG:
		a = int_form(N.a[e], g, it);
		return a ? mk_un(O_NEG, a) : 0;
	case O_ADD: case O_SUB:
		a = int_form(N.a[e], g, it);
		b = int_form(N.b[e], g, it);
		return a && b ? mk_bin(N.op[e], a, b) : 0;
	case O_MUL:
		/* a constant factor carries the scale */
		if (N.op[N.a[e]] == O_CONST) {
			double v = N.kr[N.a[e]];
			int ex;
			if (TY_FP(N.ty[N.a[e]]) && v > 0 && frexp(v, &ex) == 0.5) {
				b = int_form(N.b[e], g - (ex - 1), it);
				return b;
			}
			a = int_form(N.a[e], g, it);
			b = int_form(N.b[e], 0, it);
			return a && b ? mk_bin(O_MUL, a, b) : 0;
		}
		return 0;
	case O_DIV:
		if (N.op[N.b[e]] == O_CONST && TY_FP(N.ty[N.b[e]])) {
			double v = N.kr[N.b[e]];
			int ex;
			if (v > 0 && frexp(v, &ex) == 0.5)
				return int_form(N.a[e], g + (ex - 1), it);
		}
		return 0;
	}
	return 0;
}

nid
memk_fixform(nid v, int g)
{
	ty_t it = TY(K_SINT, 32);
	nid form = int_form(v, g, it);

	/* the scaled value converted: exact, it is an integer */
	if (form == 0)
		form = mk_cast(it, g == 0 ? v : mk_bin(O_MUL, v,
		    mk_cast(N.ty[v], mk_const_r(ldexp(1.0, -g)))));
	return form;
}

void
memk_intatomic(struct func *f, long count, double tol, int global,
    const struct model *m, struct intatom *out)
{
	nid at = find_fatomic(f->body), v, cell = 0, k, form;
	struct sbuf why = {0};
	const char *fn = f->anon ? shared_str(f->body, 120) : sym_str(f->name);
	int shared = !global, g = 0, havegrid, s;
	double kmax = 0, cf, ci, cconv, vmag = 0, fits = 0;
	ty_t it = TY(K_SINT, 32);
	const char *sc;
	struct intatom none;

	if (out == NULL)
		out = &none;
	memset(out, 0, sizeof *out);
	if (at) {
		v = N.b[at];
		cell = N.a[at];
		shared = (N.ki[at] & AT_SHARED) != 0;
	} else
		v = f->body;
	sc = shared ? "shared" : "global";
	rep_begin("intatomic");
	rep_str("model", sym_str(m->name));
	rep_str("scope", sc);
	if (!TY_FP(N.ty[v])) {
		printf("intatomic %s: the addend %s is not a float: its atomic "
		    "add is an integer one already\n", fn, shared_str(v, 100));
		rep_bool("rewritten", 0);
		return;
	}
	cf = (double)model_cost(m, shared ? C_ATOM_SH_F : C_ATOM_GL_F);
	ci = (double)model_cost(m, shared ? C_ATOM_SH : C_ATOM_GL);
	if (count > 0)
		printf("intatomic %s under %s: a %s float atomic add of %s, at "
		    "most %ld into one cell\n", fn, sym_str(m->name), sc,
		    shared_str(v, 160), count);
	else
		printf("intatomic %s under %s: a %s float atomic add of %s; how "
		    "many reach one cell is not known here\n", fn,
		    sym_str(m->name), sc, shared_str(v, 160));
	printf("  the float add costs %g, an integer one %g\n", cf, ci);
	rep_num("cost_float", cf);
	if (cf <= ci) {
		printf("  -> not rewritten: under %s a %s float atomic add is "
		    "priced as an integer one (%g): nothing to gain\n",
		    sym_str(m->name), sc, ci);
		rep_bool("rewritten", 0);
		return;
	}
	havegrid = fix_grid(f, v, &g, &kmax, &why);
	if (havegrid) {
		double total = kmax * (double)count;
		printf("  the addend is k 2^%d with |k| <= %.0f, exactly: every "
		    "leaf is an integer or a binary constant, the\n  operations "
		    "keep the grid, and no f32 value on the way needs more than "
		    "24 bits, so none rounds\n", g, kmax);
		if (count <= 0) {
			fits = kmax > 0 ? floor(2147483647.0 / kmax) : HUGE_VAL;
			if (kmax > 0)
				printf("  exact while at most %.0f of them reach one "
				    "cell: their sum then fits 32 bits\n", fits);
			else
				printf("  the addend is 0: any number of them fits\n");
		} else if (total > 2147483647.0) {
			printf("  -> not rewritten: %ld addends of up to %.0f can "
			    "reach %.0f, past 2^31 - 1; fewer adds (`count N`) or "
			    "a narrower range\n", count, kmax, total);
			rep_bool("rewritten", 0);
			free(why.p);
			return;
		} else
			printf("  the sum of %ld of them is at most %.0f 2^%d: it "
			    "fits 32 bits (%.0f < 2^31), so the integer sum is the "
			    "exact sum\n", count, total, g, total);
		form = memk_fixform(v, g);
		cconv = expr_cost_under(m, form, 0) - expr_cost_under(m, v, 0);
		if (count > 0 && total <= 16777216.0)
			printf("  the float sum would be exact too (it stays "
			    "within 24 bits): the two agree bit for bit, in "
			    "every order\n");
		else if (count > 0)
			printf("  the float sum rounds once it passes 24 bits "
			    "(%.0f here), by an amount that depends on the order "
			    "the\n  invocations arrive in; the integer sum does "
			    "not, and is exact\n", total);
		s = -g;
	} else {
		double lo = 0, hi = 0, total;
		struct plan p;
		struct ival *iv;
		nid roots[1];

		printf("  no exact fixed point: %s\n", why.n ? why.p : "the "
		    "addend is not on a binary grid");
		if (!(tol > 0)) {
			printf("  -> not rewritten: `tol T` allows a rounded "
			    "scaling (T: absolute error on the sum)\n");
			rep_bool("rewritten", 0);
			free(why.p);
			return;
		}
		if (count <= 0) {
			printf("  -> not rewritten: a rounded addend's error grows "
			    "with the number of adds, which is not known here\n");
			rep_bool("rewritten", 0);
			free(why.p);
			return;
		}
		/* the addend's magnitude, from the parameters' ranges */
		roots[0] = v;
		plan_build(&p, roots, 1, 1);
		iv = xcalloc(p.n, sizeof *iv);
		plan_ival(&p, f, iv);
		if (iv[plan_loc(&p, v)].top) {
			int ok = N.op[v] == O_VAR && N.ki[v] < f->np &&
			    f->p[N.ki[v]].ranged;
			if (ok) {
				lo = f->p[N.ki[v]].rlo;
				hi = f->p[N.ki[v]].rhi;
			}
			if (!ok) {
				printf("  -> not rewritten: the addend's range is not "
				    "known (give its parameters ranges)\n");
				rep_bool("rewritten", 0);
				free(iv);
				plan_free(&p);
				free(why.p);
				return;
			}
		} else {
			lo = iv[plan_loc(&p, v)].rlo;
			hi = iv[plan_loc(&p, v)].rhi;
		}
		free(iv);
		plan_free(&p);
		vmag = fabs(lo) > fabs(hi) ? fabs(lo) : fabs(hi);
		/* the largest scale whose sum fits, each scaled addend below
		 * 2^22 so that adding 0.5 to it is exact in f32 */
		for (s = 30; s > -30; s--)
			if ((vmag * ldexp(1.0, s) + 0.5) * (double)count <=
			    2147483647.0 && vmag * ldexp(1.0, s) <= 4194304.0)
				break;
		total = (double)count * ldexp(0.5, -s);
		printf("  |addend| <= %g: scaled by 2^%d (exact; below 2^22, so "
		    "+ 0.5 and floor are exact too) and rounded to nearest\n  each "
		    "is off by at most 2^%d, %ld of them by %g; the sum fits 32 "
		    "bits\n", vmag, s, -s - 1, count, total);
		if (total > tol) {
			printf("  -> not rewritten: that error %g exceeds tol %g\n",
			    total, tol);
			rep_bool("rewritten", 0);
			free(why.p);
			return;
		}
		g = -s;
		{
			nid sv = mk_bin(O_MUL, v, mk_cast(N.ty[v],
			    mk_const_r(ldexp(1.0, s))));
			nid half = mk_cast(N.ty[v], mk_const_r(0.5));
			form = mk_cast(it, mk_un(O_FLOOR, mk_bin(O_ADD, sv, half)));
		}
		cconv = expr_cost_under(m, form, 0) - expr_cost_under(m, v, 0);
		rep_num("error_bound", total);
	}
	free(why.p);
	if (cconv < 0)
		cconv = 0;
	/* each side whole: its addend and its atomic */
	cf += expr_cost_under(m, v, 0);
	ci += expr_cost_under(m, form, 0);
	(void)cconv;
	printf("  the integer addend: %s\n", shared_str(form, 200));
	if (ci >= cf) {
		printf("  -> not rewritten: the integer addend and its atomic "
		    "cost %g, the float ones %g under %s\n", ci, cf,
		    sym_str(m->name));
		rep_bool("rewritten", 0);
		return;
	}
	out->ok = 1;
	out->exact = havegrid;
	out->g = g;
	out->kmax = kmax;
	out->fits = count > 0 ? (double)count : fits;
	out->cost_f = cf;
	out->cost_i = ci;
	out->form = form;
	if (count <= 0) {
		printf("  -> the integer add of %s: addend and atomic cost %g "
		    "against %g (%+.1f%%), exact for up to %.0f adds into one "
		    "cell\n", shared_str(form, 200), ci, cf, 100 * (ci - cf) / cf,
		    fits);
		rep_bool("rewritten", 0);
		rep_num("fits", fits);
		rep_num("cost", ci);
		return;
	}
	{
		struct func copy = *f, *h;
		char nm[96];
		nid icell, body;
		int slot = -1;

		/* the cell: an i32 parameter in the float cell's place */
		if (cell && N.op[cell] == O_VAR && N.ki[cell] < copy.np) {
			slot = (int)N.ki[cell];
			copy.p[slot].ty = it;
			copy.p[slot].ranged = 0;
		} else if (copy.np < MAXP) {
			slot = copy.np++;
			memset(&copy.p[slot], 0, sizeof copy.p[slot]);
			copy.p[slot].name = sym_cstr("cell");
			copy.p[slot].ty = it;
		}
		if (slot < 0) {
			printf("  -> not rewritten: no room for the cell "
			    "parameter\n");
			rep_bool("rewritten", 0);
			return;
		}
		icell = mk_var((uint32_t)slot, copy.p[slot].name, it);
		k = mk_atomic(AT_ADD | (shared ? AT_SHARED : 0), icell, form);
		/* the old value, read back as the float it stands for */
		body = mk_cast(TY_F32, k);
		if (g != 0)
			body = mk_bin(O_MUL, body, mk_cast(TY_F32,
			    mk_const_r(ldexp(1.0, g))));
		snprintf(nm, sizeof nm, "%s_ia", f->anon ? "" : sym_str(f->name));
		copy.name = sym_cstr(nm);
		copy.anon = 0;
		copy.body = body;
		printf("  -> %s: addend and atomic cost %g against %g (%+.1f%%)\n",
		    shared_str(k, 200), ci, cf, 100 * (ci - cf) / cf);
		if (g != 0)
			printf("     the cell is an i32 holding the sum times 2^%d; "
			    "every other access of it changes too: a store of v\n"
			    "     stores i32(v * 2^%d), a read is f32(cell) * 2^%d (a "
			    "conversion and a multiply: %g a read)\n", -g, -g, g,
			    (double)m->c[C_CONV] + (double)m->c[C_FMUL]);
		else
			printf("     the cell is an i32 holding the sum; every "
			    "other access of it changes too: a store of v stores\n"
			    "     i32(v), a read is f32(cell) (a conversion: %g a "
			    "read)\n", (double)m->c[C_CONV]);
		h = func_new(copy.name);
		*h = copy;
		printf("  defined %s: the atomic's old value read back as a "
		    "float\n", nm);
		rep_bool("rewritten", 1);
		rep_expr("atomic", k);
		rep_int("scale_log2", -g);
		rep_num("cost", ci);
		rep_str("defined", nm);
		rep_proof("derivation", havegrid ? "the addend's binary grid "
		    "and magnitude from its structure and ranges; the sum "
		    "fits 32 bits" : "a rounded scaling within the tolerance");
	}
}

/* ---- commands ---- */

int
memk_command(struct P *p)
{
	static const char *const stop[] = {"size", "elem", "model", "count",
		"tol", "global", NULL};
	struct model *m = model_active();
	char w[48];
	uint32_t at = cmdp_pos(p);
	int64_t v;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if ((strcmp(w, "banks") != 0 && strcmp(w, "intatomic") != 0) ||
	    func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	if (!strcmp(w, "banks")) {
		uint32_t names[MAXACC];
		int nf = 0, elem = 4, k;
		int64_t size = 0;

		do {
			struct func *f = NULL;
			uint32_t at2 = cmdp_pos(p);
			if (nf == MAXACC)
				fail("banks: at most %d accesses", MAXACC);
			/* a name before a comma: cmd_func would read on */
			if (cmdp_word(p, w, sizeof w) && (f = func_find(sym_cstr(w)))
			    != NULL && (cmdp_at(p, T_COMMA) || cmdp_eos(p)))
				;
			else {
				cmdp_seek(p, at2);
				f = cmdp_func(p, stop);
			}
			if (f->anon) {
				/* keep it: the next anonymous one reuses its slot */
				char nm[24];
				struct func copy = *f, *h;
				snprintf(nm, sizeof nm, "_acc%d", nf + 1);
				copy.name = sym_cstr(nm);
				h = func_new(copy.name);
				*h = copy;
				h->anon = 1;
				names[nf++] = h->name;
			} else
				names[nf++] = f->name;
		} while (cmdp_tok(p, T_COMMA));
		while (!cmdp_eos(p)) {
			if (cmdp_kw(p, "size")) {
				if (!cmdp_int(p, &size) || size < 1)
					fail("banks: size N, the array's length");
			} else if (cmdp_kw(p, "elem")) {
				if (!cmdp_int(p, &v) || v < 1 || v > 64)
					fail("banks: elem B, bytes per element (1 to 64)");
				elem = (int)v;
			} else if (cmdp_kw(p, "model")) {
				if (!cmdp_word(p, w, sizeof w) ||
				    (m = model_find(sym_cstr(w))) == NULL)
					fail("banks: unknown model (see 'models')");
			} else {
				cmdp_fail_here(p);
				fail("banks F [, G ...] [size N] [elem B] [model M]");
			}
		}
		for (k = 0; k < nf; k++)
			if (TY_FP(N.ty[func_find(names[k])->body]))
				fail("banks: an index is an integer, %s is not",
				    shared_str(func_find(names[k])->body, 80));
		notes_flush();
		memk_banks(names, nf, size, elem, 0, 0, m, NULL);
		return 1;
	}
	{
		struct func *f = cmdp_func(p, stop);
		long count = 1024;
		double tol = 0;
		int global = 0;

		while (!cmdp_eos(p)) {
			if (cmdp_kw(p, "count")) {
				if (!cmdp_int(p, &v) || v < 1)
					fail("intatomic: count N, the adds into one cell");
				count = (long)v;
			} else if (cmdp_kw(p, "tol")) {
				tol = cmdp_number(p, stop);
				if (!(tol > 0))
					fail("intatomic: tol must be positive, not %g",
					    tol);
			} else if (cmdp_kw(p, "global"))
				global = 1;
			else if (cmdp_kw(p, "model")) {
				if (!cmdp_word(p, w, sizeof w) ||
				    (m = model_find(sym_cstr(w))) == NULL)
					fail("intatomic: unknown model (see 'models')");
			} else {
				cmdp_fail_here(p);
				fail("intatomic F [count N] [tol T] [global] [model M]");
			}
		}
		notes_flush();
		memk_intatomic(f, count, tol, global, m, NULL);
	}
	return 1;
}

/* ---- English ---- */

/*
 * bank conflicts of | in F [and G]; conflict degree of F; swizzle F;
 * avoid bank conflicts in F; make F an integer atomic; integer atomic
 * for F; F as an integer atomic; fixed-point atomic for F; each with
 * [under | on MODEL].
 */
const char *
memk_nlp_raw(const char *sent)
{
	static char out[1200];
	char buf[1000], tail[80] = "", *s, *q, *e;
	size_t n;
	const char *cmd = NULL;
	static const char *const bpre[] = {"bank conflicts of ",
		"bank conflicts in ", "the bank conflicts of ",
		"the bank conflicts in ", "conflict degree of ",
		"the conflict degree of ", "avoid bank conflicts in ",
		"remove bank conflicts from ", "swizzle ", "what are the bank "
		"conflicts of ", NULL};
	static const char *const apre[] = {"integer atomic for ",
		"an integer atomic for ", "fixed-point atomic for ",
		"fixed point atomic for ", "integer atomics for ", NULL};
	static const char *const apost[] = {" as an integer atomic",
		" an integer atomic", " with an integer atomic",
		" in fixed point", NULL};
	int i;

	while (*sent == ' ' || *sent == '\t')
		sent++;
	if (strlen(sent) >= sizeof buf)
		return NULL;
	strcpy(buf, sent);
	n = strlen(buf);
	while (n && (buf[n - 1] == ' ' || buf[n - 1] == '?' || buf[n - 1] == '.'
	    || buf[n - 1] == '\n'))
		buf[--n] = 0;
	s = buf;
	if (!strncasecmp(s, "please ", 7))
		s += 7;
	q = NULL;
	for (e = s; (e = strstr(e, " under ")) != NULL; e++)
		q = e;
	if (q == NULL)
		for (e = s; (e = strstr(e, " on ")) != NULL; e++)
			q = e;
	if (q != NULL) {
		const char *w = strrchr(q, ' ') + 1;
		size_t wn = strlen(w);
		if (wn && wn < 40 && strchr(q + 1, ' ') == strrchr(q, ' ') &&
		    model_find(sym_intern(w, wn)) != NULL) {
			snprintf(tail, sizeof tail, " model %s", w);
			*q = 0;
		}
	}
	for (i = 0; bpre[i] && !cmd; i++)
		if (!strncasecmp(s, bpre[i], strlen(bpre[i]))) {
			s += strlen(bpre[i]);
			cmd = "banks";
		}
	for (i = 0; apre[i] && !cmd; i++)
		if (!strncasecmp(s, apre[i], strlen(apre[i]))) {
			s += strlen(apre[i]);
			cmd = "intatomic";
		}
	if (!cmd) {
		if (!strncasecmp(s, "make ", 5) || !strncasecmp(s, "rewrite ", 8))
			s = strchr(s, ' ') + 1;
		n = strlen(s);
		for (i = 0; apost[i]; i++) {
			size_t k = strlen(apost[i]);
			if (n > k && !strcasecmp(s + n - k, apost[i])) {
				s[n - k] = 0;
				cmd = "intatomic";
				break;
			}
		}
	}
	if (!cmd || *s == 0 || !strcmp(s, "it"))
		return NULL;
	if (!strcmp(cmd, "banks")) {
		/* "f and g": two accesses of the array */
		char *a;
		while ((a = strstr(s, " and ")) != NULL)
			memcpy(a, ",    ", 5);
	}
	snprintf(out, sizeof out, "%s %s%s", cmd, s, tail);
	return out;
}
