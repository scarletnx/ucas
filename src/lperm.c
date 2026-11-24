/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * lperm.c: lane maps as xor exchange stages (src/lperm.h says how).
 * Nothing here names a target: the exchanges are the model's classes
 * sg_shuffle_xor, sg_quad and sg_shuffle, the selects and predicates its
 * select, logic, cmp and bitcnt, and the cheapest proved realisation
 * under the active model is the one taken.
 */
#include "calc.h"
#include "fuse.h"
#include "lperm.h"
#include "report.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int
parity8(unsigned x)
{
	x ^= x >> 8;
	x ^= x >> 4;
	x ^= x >> 2;
	x ^= x >> 1;
	return (int)(x & 1);
}

static int
log2w(int w)
{
	int b = 0;

	if (w < 2 || w > SG_MAXW || (w & (w - 1)))
		return -1;
	while ((1 << b) < w)
		b++;
	return b;
}

/* A x over n bits, A by its columns */
static unsigned
apply(const uint8_t *col, int n, unsigned x)
{
	unsigned r = 0;
	int j;

	for (j = 0; j < n; j++)
		if (x >> j & 1)
			r ^= col[j];
	return r;
}

static int
rank_of(const uint8_t *v, int n)
{
	uint8_t a[8];
	int r = 0, i, j, b;

	memcpy(a, v, (size_t)n);
	for (b = 7; b >= 0; b--) {
		for (i = r; i < n && !(a[i] >> b & 1); i++)
			;
		if (i == n)
			continue;
		{
			uint8_t t = a[i];
			a[i] = a[r];
			a[r] = t;
		}
		for (j = 0; j < n; j++)
			if (j != r && (a[j] >> b & 1))
				a[j] ^= a[r];
		r++;
	}
	return r;
}

/*
 * Each output bit j of map is an affine function of the lane's n bits:
 * <a_j, l> ^ c_j on every defined lane, a linear system over F2 in n + 1
 * unknowns, solved by elimination.  A free unknown takes the identity's
 * value (a_j = e_j, c_j = 0), so an unread lane does not add stages.
 */
int
lperm_affine(const int16_t *map, int w, uint8_t *col, uint8_t *c)
{
	int n = log2w(w), j, l, i, r, q, piv[8];
	uint8_t row[SG_MAXW];	/* unknowns: bits 0..n-1 a_j, bit n c_j */
	uint8_t rhs[SG_MAXW];

	if (n < 0)
		return 0;
	for (l = 0; l < w; l++)
		if (map[l] >= w)
			return 0;
	memset(col, 0, 6);
	*c = 0;
	for (j = 0; j < n; j++) {
		uint8_t sol = 0;
		int nr = 0;
		for (l = 0; l < w; l++)
			if (map[l] >= 0) {
				row[nr] = (uint8_t)(l | 1 << n);
				rhs[nr++] = (uint8_t)(map[l] >> j & 1);
			}
		/* reduced row echelon form */
		r = 0;
		for (q = 0; q <= n; q++) {
			piv[q] = -1;
			for (i = r; i < nr && !(row[i] >> q & 1); i++)
				;
			if (i == nr)
				continue;
			{
				uint8_t t = row[i], u = rhs[i];
				row[i] = row[r];
				rhs[i] = rhs[r];
				row[r] = t;
				rhs[r] = u;
			}
			for (i = 0; i < nr; i++)
				if (i != r && (row[i] >> q & 1)) {
					row[i] ^= row[r];
					rhs[i] ^= rhs[r];
				}
			piv[q] = r++;
		}
		for (i = r; i < nr; i++)
			if (rhs[i])
				return 0;	/* inconsistent: not affine */
		/* free unknowns as the identity's, then the pivots */
		for (q = 0; q <= n; q++)
			if (piv[q] < 0 && q == j)
				sol |= (uint8_t)(1 << q);
		for (q = 0; q <= n; q++)
			if (piv[q] >= 0) {
				int v = rhs[piv[q]], t;
				for (t = 0; t <= n; t++)
					if (t != q && piv[t] < 0 && (row[piv[q]] >> t & 1) &&
					    (sol >> t & 1))
						v ^= 1;
				if (v)
					sol |= (uint8_t)(1 << q);
			}
		for (q = 0; q < n; q++)
			if (sol >> q & 1)
				col[q] |= (uint8_t)(1 << j);
		if (sol >> n & 1)
			*c |= (uint8_t)(1 << j);
	}
	for (l = 0; l < w; l++)
		if (map[l] >= 0 && (int)(apply(col, n, (unsigned)l) ^ *c) !=
		    map[l])
			return 0;
	return 1;
}

/* ---- prices ---- */

static double
cls(const struct model *m, int k)
{
	return cclass_available(m, k) ? (double)m->c[k] : INFINITY;
}

/* the exchange with lane xor k, and the class that does it */
static double
xchg_price(const struct model *m, int k, int *how)
{
	double b = cls(m, C_SG_SHUF) + m->c[C_LOGIC], x = cls(m, C_SG_SHUFXOR),
	    q = k < 4 ? cls(m, C_SG_QUAD) : INFINITY;

	*how = LPX_SHUF;
	if (x < b) {
		b = x;
		*how = LPX_XOR;
	}
	if (q < b) {
		b = q;
		*how = LPX_QUAD;
	}
	return b;
}

static double
pred_price(const struct model *m, int msk)
{
	if (!msk)
		return 0;
	if (!(msk & (msk - 1)))
		return m->c[C_LOGIC] + m->c[C_CMP] + m->c[C_SELECT];
	return 2 * m->c[C_LOGIC] + m->c[C_BITCNT] + m->c[C_CMP] +
	    m->c[C_SELECT];
}

/* a stage's lanes as a table: (T >> lane) & 1, compared, selected */
static double
tab_price(const struct model *m)
{
	return m->c[C_SHIFT] + m->c[C_LOGIC] + m->c[C_CMP] + m->c[C_SELECT];
}

/* ---- the search ---- */

#define BEAM 4

struct srch {
	const struct model *m;
	int n;
	uint8_t A[6];
	int lim;
	/* the current sequence */
	uint8_t k[LPERM_MAXST], msk[LPERM_MAXST];
	int d;
	/* the best */
	int bn;
	uint8_t bk[LPERM_MAXST], bm[LPERM_MAXST];
	double bp;
	long work;
};

struct cand {
	uint8_t k, m;
	double p;
};

static int
cmp_cand(const void *a, const void *b)
{
	const struct cand *x = a, *y = b;

	return x->p < y->p ? -1 : x->p > y->p;
}

static void
dfs(struct srch *s, const uint8_t *M, double sofar)
{
	uint8_t D[6], M2[6];
	int n = s->n, r, k, mk, j, nc = 0, i, how;
	struct cand *cs;

	for (j = 0; j < n; j++)
		D[j] = M[j] ^ s->A[j];
	r = rank_of(D, n);
	if (r == 0) {
		if (sofar < s->bp) {
			s->bp = sofar;
			s->bn = s->d;
			memcpy(s->bk, s->k, (size_t)s->d);
			memcpy(s->bm, s->msk, (size_t)s->d);
		}
		return;
	}
	if (s->d + r > s->lim || s->work > 2000000)
		return;
	cs = xmalloc(sizeof *cs << (2 * n));
	for (k = 1; k < 1 << n; k++) {
		unsigned u = apply(M, n, (unsigned)k);
		double xp = xchg_price(s->m, k, &how);
		if (!u || !isfinite(xp))
			continue;	/* the stage moves nothing */
		for (mk = 1; mk < 1 << n; mk++) {
			s->work++;
			for (j = 0; j < n; j++) {
				M2[j] = (uint8_t)(M[j] ^ (mk >> j & 1 ? u : 0));
				D[j] = M2[j] ^ s->A[j];
			}
			if (rank_of(D, n) != r - 1)
				continue;
			cs[nc].k = (uint8_t)k;
			cs[nc].m = (uint8_t)mk;
			cs[nc++].p = xp + pred_price(s->m, mk);
		}
	}
	xsort(cs, (size_t)nc, sizeof *cs, cmp_cand);
	for (i = 0; i < nc && i < BEAM; i++) {
		unsigned u = apply(M, n, cs[i].k);
		if (sofar + cs[i].p >= s->bp)
			break;
		for (j = 0; j < n; j++)
			M2[j] = (uint8_t)(M[j] ^ (cs[i].m >> j & 1 ? u : 0));
		s->k[s->d] = cs[i].k;
		s->msk[s->d] = cs[i].m;
		s->d++;
		dfs(s, M2, sofar + cs[i].p);
		s->d--;
	}
	free(cs);
}

/* the lane whose value lane l holds after the stages: g_1(...g_n(l)) */
static unsigned
compose(const struct lperm_plan *p, unsigned l)
{
	int i;

	for (i = p->nst - 1; i >= 0; i--) {
		const struct lperm_stage *s = &p->st[i];
		if (s->istab ? (int)(s->tab >> l & 1) : parity8(l & s->m) ^ s->e)
			l ^= s->k;
	}
	return l;
}

/* the stages for the affine map p->col, p->c: fewest first, then the
 * cheapest found; 0 when the model has no exchange */
static int
plan_stages(const struct model *m, struct lperm_plan *p)
{
	struct srch s;
	uint8_t I[6], u[LPERM_MAXST], M[6];
	int n = p->bits, j, i, how, best = -1;
	unsigned d, e, want;

	memset(&s, 0, sizeof s);
	s.m = m;
	s.n = n;
	memcpy(s.A, p->col, 6);
	for (j = 0; j < n; j++)
		I[j] = (uint8_t)(1 << j);
	{
		uint8_t D[6];
		for (j = 0; j < n; j++)
			D[j] = I[j] ^ p->col[j];
		p->rank = rank_of(D, n);
	}
	s.bp = INFINITY;
	for (s.lim = p->rank; s.lim <= 2 * n && !isfinite(s.bp); s.lim++)
		dfs(&s, I, 0);
	if (!isfinite(s.bp))
		return 0;
	/* the constant: u_i = M_(i-1) k_i, d = xor of the e_i u_i */
	memcpy(M, I, 6);
	for (i = 0; i < s.bn; i++) {
		u[i] = (uint8_t)apply(M, n, s.bk[i]);
		for (j = 0; j < n; j++)
			M[j] = (uint8_t)(M[j] ^ (s.bm[i] >> j & 1 ? u[i] : 0));
	}
	want = p->c;
	for (e = 0; e < 1u << s.bn; e++) {
		d = 0;
		for (i = 0; i < s.bn; i++)
			if (e >> i & 1)
				d ^= u[i];
		if (d == want) {
			best = (int)e;
			break;
		}
	}
	p->nst = 0;
	if (best < 0) {
		/* one unconditional exchange first by want (g_0 outermost:
		 * map = g_0 o T, so the constant is added at the end) */
		p->st[0].k = (uint8_t)want;
		p->st[0].m = 0;
		p->st[0].e = 1;
		p->nst = 1;
		best = 0;
	}
	for (i = 0; i < s.bn; i++) {
		struct lperm_stage *t = &p->st[p->nst++];
		t->k = s.bk[i];
		t->m = s.bm[i];
		t->e = (uint8_t)(best >> i & 1);
	}
	for (i = 0; i < p->nst; i++) {
		struct lperm_stage *t = &p->st[i];
		t->price = xchg_price(m, t->k, &how);
		t->how = (uint8_t)how;
		t->price += pred_price(m, t->m);
	}
	return 1;
}


/* ---- maps that are not affine: a Benes network ---- */

/*
 * Route the values of the lanes in L (nl of them, a subcube over the
 * bits ord[0..r-1], the others fixed) to their destinations' bits of
 * that cube: cur[x] is the destination of the value at lane x.  Stage
 * s0 and s0 + 2r - 2 exchange by bit ord[0], the stages between them are
 * the two halves' (the lanes whose bit ord[0] is 0, and 1), routed the
 * same way at once.  tab[s] collects the lanes that take the exchange.
 */
static void
benes_route(int *cur, const int *ord, int r, const int *L, int nl, int s0,
    uint64_t *tab)
{
	int B = 1 << ord[0], i, x, rm = 0, n0 = 0, n1 = 0;
	int col[SG_MAXW], keypos[SG_MAXW], L0[SG_MAXW], L1[SG_MAXW];

	for (i = 0; i < r; i++)
		rm |= 1 << ord[i];
	if (r == 1) {
		for (i = 0; i < nl; i++) {
			x = L[i];
			if (!(x & B) && ((cur[x] ^ x) & B)) {
				int t = cur[x];
				cur[x] = cur[x | B];
				cur[x | B] = t;
				tab[s0] |= 1ULL << x | 1ULL << (x | B);
			}
		}
		return;
	}
	for (i = 0; i < nl; i++) {
		col[L[i]] = -1;
		keypos[cur[L[i]] & rm] = L[i];
	}
	/* the looping algorithm: a lane and its partner by B go to
	 * different halves, and so do two values whose destinations are
	 * partners by B; the constraints are even cycles */
	for (i = 0; i < nl; i++) {
		int y = L[i], c = 0;
		if (col[y] >= 0)
			continue;
		for (;;) {
			int z;
			col[y] = c;
			col[y ^ B] = 1 - c;
			z = keypos[(cur[y ^ B] & rm) ^ B];
			if (col[z] >= 0)
				break;
			y = z;
		}
	}
	for (i = 0; i < nl; i++) {
		x = L[i];
		if (!(x & B) && col[x] == 1) {
			int t = cur[x];
			cur[x] = cur[x | B];
			cur[x | B] = t;
			tab[s0] |= 1ULL << x | 1ULL << (x | B);
		}
	}
	for (i = 0; i < nl; i++)
		if (L[i] & B)
			L1[n1++] = L[i];
		else
			L0[n0++] = L[i];
	benes_route(cur, ord + 1, r - 1, L0, n0, s0 + 1, tab);
	benes_route(cur, ord + 1, r - 1, L1, n1, s0 + 1, tab);
	for (i = 0; i < nl; i++) {
		x = L[i];
		if (!(x & B) && (cur[x] & B)) {
			int t = cur[x];
			cur[x] = cur[x | B];
			cur[x | B] = t;
			tab[s0 + 2 * r - 2] |= 1ULL << x | 1ULL << (x | B);
		}
	}
}

/* a table of the lanes as a parity of their bits: m, e; 0 when none */
static int
tab_parity(uint64_t tab, int w, uint8_t *pm, uint8_t *pe)
{
	int m, e, l;

	for (m = 0; m < w; m++)
		for (e = 0; e < 2; e++) {
			for (l = 0; l < w && (int)(tab >> l & 1) == (parity8((unsigned)(l &
			    m)) ^ e); l++)
				;
			if (l == w) {
				*pm = (uint8_t)m;
				*pe = (uint8_t)e;
				return 1;
			}
		}
	return 0;
}

static int
next_order(int *ord, int n)
{
	int i = n - 2, j, t;

	while (i >= 0 && ord[i] >= ord[i + 1])
		i--;
	if (i < 0)
		return 0;
	for (j = n - 1; ord[j] <= ord[i]; j--)
		;
	t = ord[i];
	ord[i] = ord[j];
	ord[j] = t;
	for (i++, j = n - 1; i < j; i++, j--) {
		t = ord[i];
		ord[i] = ord[j];
		ord[j] = t;
	}
	return 1;
}

/*
 * A map that sends each lane's value to one lane at most, as a Benes
 * network: every order of the bits routed, its stages priced, the
 * cheapest kept.  The unread lanes take the sources nobody reads (their
 * own first).  0 when the map copies a lane to two.
 */
static int
plan_benes(const struct model *m, const int16_t *map, struct lperm_plan *p)
{
	int w = p->w, n = p->bits, l, ord[6], src[SG_MAXW], used[SG_MAXW];
	int i, L[SG_MAXW], how, best_n = 0;
	double best = INFINITY;
	struct lperm_stage bst[LPERM_MAXST];

	if (n < 1 || 2 * n - 1 > LPERM_MAXST)
		return 0;
	memset(used, 0, sizeof used);
	for (l = 0; l < w; l++) {
		src[l] = map[l];
		if (map[l] >= 0 && used[map[l]]++)
			return 0;
	}
	for (l = 0; l < w; l++)
		if (src[l] < 0 && !used[l])
			used[src[l] = l] = 1;
	for (l = 0, i = 0; l < w; l++)
		if (src[l] < 0) {
			while (used[i])
				i++;
			used[src[l] = i] = 1;
		}
	for (i = 0; i < n; i++)
		ord[i] = i;
	for (l = 0; l < w; l++)
		L[l] = l;
	do {
		int cur[SG_MAXW], ns = 0, s;
		uint64_t tab[LPERM_MAXST];
		struct lperm_stage st[LPERM_MAXST];
		double pr = 0;
		memset(tab, 0, sizeof tab);
		for (l = 0; l < w; l++)
			cur[src[l]] = l;
		benes_route(cur, ord, n, L, w, 0, tab);
		for (s = 0; s < 2 * n - 1; s++) {
			struct lperm_stage *t;
			uint64_t all = w == 64 ? ~0ULL : (1ULL << w) - 1;
			if (!tab[s])
				continue;
			t = &st[ns++];
			memset(t, 0, sizeof *t);
			t->k = (uint8_t)(1 << ord[s < n ? s : 2 * n - 2 - s]);
			if (tab[s] == all) {
				t->e = 1;	/* every lane */
			} else if (!tab_parity(tab[s], w, &t->m, &t->e) || !t->m) {
				t->m = 0;
				t->e = 0;
				t->istab = 1;
				t->tab = tab[s];
			}
			t->price = xchg_price(m, t->k, &how);
			t->how = (uint8_t)how;
			t->price += t->istab ? tab_price(m) : pred_price(m, t->m);
			pr += t->price;
		}
		if (pr < best) {
			best = pr;
			best_n = ns;
			memcpy(bst, st, sizeof st);
		}
	} while (next_order(ord, n));
	if (!isfinite(best))
		return 0;
	/* in the order the values go through them, as plan_stages's */
	p->nst = best_n;
	for (i = 0; i < best_n; i++)
		p->st[i] = bst[i];
	p->benes = 1;
	return 1;
}

/* ---- expressions ---- */

/* (tab >> l) & 1 != 0 */
static nid
pred_tab(nid l, uint64_t tab, int w)
{
	ty_t u = TY(K_UINT, w > 32 ? 64 : 32);

	return mk_bin(O_NE, mk_bin(O_AND, mk_bin(O_SHR, mk_const_i(u,
	    (int64_t)tab), mk_cast(u, l)), mk_const_i(u, 1)), mk_const_i(u, 0));
}

static nid
pred_of(nid l, int msk)
{
	ty_t u = TY(K_UINT, 32);
	nid a = mk_bin(O_AND, l, mk_const_i(u, msk));

	if (msk & (msk - 1))
		a = mk_bin(O_AND, mk_cast(u, mk_un(O_POPCNT, a)), mk_const_i(u, 1));
	return mk_bin(O_NE, a, mk_const_i(u, 0));
}

/* the affine index A l ^ c of lane l, by shifts, masks and xors */
static nid
affine_index(const struct lperm_plan *p, nid l)
{
	ty_t u = TY(K_UINT, 32);
	nid r = 0;
	int n = p->bits, s, i;

	for (s = -(n - 1); s < n; s++) {
		unsigned msk = 0;
		nid t;
		for (i = 0; i < n; i++)
			if (i + s >= 0 && i + s < n && (p->col[i] >> (i + s) & 1))
				msk |= 1u << i;
		if (!msk)
			continue;
		t = msk == (unsigned)(p->w - 1) ? l : mk_bin(O_AND, l,
		    mk_const_i(u, msk));
		if (s > 0)
			t = mk_bin(O_SHL, t, mk_const_i(u, s));
		else if (s < 0)
			t = mk_bin(O_SHR, t, mk_const_i(u, -s));
		r = r ? mk_bin(O_XOR, r, t) : t;
	}
	if (!r)
		return mk_const_i(u, p->c);
	return p->c ? mk_bin(O_XOR, r, mk_const_i(u, p->c)) : r;
}

nid
lperm_expr(const struct lperm_plan *p, nid v, nid at)
{
	ty_t u = TY(K_UINT, 32);
	nid w = mk_const_i(TY_LIT, p->w), L = sg_lane(), sb = v, sl;
	nid map[SG_LANE_SLOT + 1];
	int i;

	at = mk_cast(u, at);
	memset(map, 0, sizeof map);
	map[SG_LANE_SLOT] = at;
	sl = subst(v, map, SG_LANE_SLOT + 1);
	switch (p->kind) {
	case LPH_IDENTITY:
		return sl;
	case LPH_QUADBCAST:
		return mk_sg(O_SGSHUF, v, mk_bin(O_OR, mk_bin(O_AND, at,
		    mk_const_i(u, ~3u)), mk_const_i(u, p->qbi)), w);
	case LPH_SHUFFLE:
		return p->affine ? mk_sg(O_SGSHUF, v, affine_index(p, at), w) : 0;
	case LPH_STAGES:
		break;
	default:
		return 0;
	}
	for (i = 0; i < p->nst; i++) {
		const struct lperm_stage *s = &p->st[i];
		nid k = mk_const_i(u, s->k);
		nid shl = mk_sg(O_SGSHUF, sb, mk_bin(O_XOR, at, k), w), nl;
		nid shb = i + 1 < p->nst ? mk_sg(O_SGSHUF, sb, mk_bin(O_XOR, L,
		    k), w) : 0, nb = 0;
		if (LPERM_ALWAYS(s)) {
			nl = shl;
			nb = shb;
		} else if (s->istab) {
			nid pl = pred_tab(at, s->tab, p->w);
			nl = mk_sel(pl, shl, sl);
			if (shb)
				nb = mk_sel(pred_tab(L, s->tab, p->w), shb, sb);
		} else {
			nid pl = pred_of(at, s->m);
			nl = s->e ? mk_sel(pl, sl, shl) : mk_sel(pl, shl, sl);
			if (shb) {
				nid pb = pred_of(L, s->m);
				nb = s->e ? mk_sel(pb, sb, shb) : mk_sel(pb, shb, sb);
			}
		}
		sl = nl;
		sb = nb;
	}
	return sl;
}

static double
expr_price(const struct model *m, const struct lperm_plan *p)
{
	nid v = mk_var(1, sym_cstr("v"), TY(K_UINT, 32));
	nid at = mk_var(0, sym_cstr("lane"), TY(K_UINT, 32));
	nid e = lperm_expr(p, v, at);

	if (!e)
		return INFINITY;
	return fuse_expr_cost(m, sg_flat(e, at));
}

int
lperm_plan(const struct model *m, const int16_t *map, int w,
    double idx_price, struct lperm_plan *p)
{
	int l, any = 0, ident = 1, qb = -1, isqb = 1;

	memset(p, 0, sizeof *p);
	p->w = w;
	p->bits = log2w(w);
	p->qbi = -1;
	p->price = p->stages_price = p->shuffle_price = INFINITY;
	if (p->bits < 0)
		return p->kind = LPH_NONE;
	for (l = 0; l < w; l++) {
		if (map[l] < 0)
			continue;
		if (map[l] >= w)
			return p->kind = LPH_NONE;
		any = 1;
		ident &= map[l] == l;
		if ((map[l] & ~3) != (l & ~3))
			isqb = 0;
		else if (qb < 0)
			qb = map[l] & 3;
		else if ((map[l] & 3) != qb)
			isqb = 0;
	}
	if (!any || ident) {
		p->kind = LPH_IDENTITY;
		p->price = 0;
		p->proved = 1;
		p->affine = 1;
		for (l = 0; l < p->bits; l++)
			p->col[l] = (uint8_t)(1 << l);
		return p->kind;
	}
	p->affine = lperm_affine(map, w, p->col, &p->c);
	if (p->affine) {
		uint8_t D[6];
		for (l = 0; l < p->bits; l++)
			D[l] = (uint8_t)(p->col[l] ^ (1 << l));
		p->rank = rank_of(D, p->bits);
	}
	/* one shuffle by an index */
	if (idx_price < 0 && p->affine) {
		p->kind = LPH_SHUFFLE;
		p->shuffle_price = expr_price(m, p);
	} else if (idx_price >= 0)
		p->shuffle_price = cls(m, C_SG_SHUF) + idx_price;
	p->kind = LPH_SHUFFLE;
	p->price = p->shuffle_price;
	/* a quad's broadcast */
	if (isqb && qb >= 0 && w >= 4 && cls(m, C_SG_QUAD) < p->price) {
		p->kind = LPH_QUADBCAST;
		p->qbi = qb;
		p->price = cls(m, C_SG_QUAD);
	}
	/* the stages */
	if (p->affine) {
		struct lperm_plan q = *p;
		q.kind = LPH_STAGES;
		if (plan_stages(m, &q)) {
			q.proved = 1;
			for (l = 0; l < w; l++)
				if (map[l] >= 0 && (int)compose(&q, (unsigned)l) != map[l])
					q.proved = 0;
			q.stages_price = expr_price(m, &q);
			p->nst = q.nst;
			memcpy(p->st, q.st, sizeof p->st);
			p->rank = q.rank;
			p->stages_price = q.stages_price;
			/* one exchange by xor in every lane is the shuffle by
			 * lane xor c itself: the same price, said as what it is */
			if (q.proved && (q.stages_price < p->price ||
			    (q.stages_price <= p->price && q.nst == 1 &&
			    !q.st[0].m))) {
				p->kind = LPH_STAGES;
				p->price = q.stages_price;
			}
		}
	}
	/* a Benes network, for a map that sends each lane to one at most */
	{
		struct lperm_plan q = *p;
		q.kind = LPH_STAGES;
		q.affine = 0;
		if (plan_benes(m, map, &q)) {
			q.proved = 1;
			for (l = 0; l < w; l++)
				if (map[l] >= 0 && (int)compose(&q, (unsigned)l) != map[l])
					q.proved = 0;
			q.stages_price = expr_price(m, &q);
			if (q.proved && (!p->nst || q.stages_price < p->stages_price)) {
				p->nst = q.nst;
				memcpy(p->st, q.st, sizeof p->st);
				p->benes = 1;
				p->stages_price = q.stages_price;
				if (q.stages_price < p->price) {
					p->kind = LPH_STAGES;
					p->price = q.stages_price;
				}
			}
		}
	}
	/* a shuffle and a quad op are their definitions */
	p->proved = 1;
	if (p->kind == LPH_STAGES) {
		p->proved = 1;
		for (l = 0; l < w; l++)
			if (map[l] >= 0 && (int)compose(p, (unsigned)l) != map[l])
				p->proved = 0;
	}
	if (!isfinite(p->price))
		p->kind = LPH_NONE;
	return p->kind;
}

double
lperm_price(const struct model *m, const int16_t map[32], int *stages)
{
	struct lperm_plan p;

	lperm_plan(m, map, 32, -1, &p);
	if (stages)
		*stages = p.kind == LPH_STAGES ? p.nst : p.kind == LPH_IDENTITY ?
		    0 : p.kind == LPH_NONE ? -1 : 1;
	return p.kind == LPH_NONE ? INFINITY : p.price;
}

static const char *const how_name[] = {"sg_shuffle_xor", "sg_quad",
    "sg_shuffle"};

static void
stage_text(struct sbuf *b, const struct lperm_stage *s)
{
	sb_printf(b, "lane xor %d", s->k);
	if (s->istab)
		sb_printf(b, " in the lanes of 0x%llx", (unsigned long long)s->tab);
	else if (!s->m)
		sb_puts(b, " in every lane");
	else if (!(s->m & (s->m - 1)))
		sb_printf(b, " where lane & %d is %s", s->m, s->e ? "0" : "not 0");
	else
		sb_printf(b, " where popcount(lane & %d) is %s", s->m, s->e ?
		    "even" : "odd");
	sb_printf(b, " (%s)", how_name[s->how]);
}

void
lperm_text(struct sbuf *b, const struct lperm_plan *p)
{
	int i;

	switch (p->kind) {
	case LPH_IDENTITY:
		sb_puts(b, "the identity: no exchange");
		return;
	case LPH_QUADBCAST:
		sb_printf(b, "the quad's broadcast of its lane %d (sg_quad)", p->qbi);
		return;
	case LPH_SHUFFLE:
		sb_puts(b, "one shuffle by an index (sg_shuffle)");
		return;
	case LPH_NONE:
		sb_puts(b, "no realisation under the model");
		return;
	}
	sb_printf(b, "%d stage%s%s:", p->nst, p->nst == 1 ? "" : "s", p->benes ?
	    " (a Benes network: the map is not affine)" : "");
	for (i = 0; i < p->nst; i++) {
		sb_puts(b, i ? "; " : " ");
		stage_text(b, &p->st[i]);
	}
}

/* ---- the command ---- */

/* the map of idx, an integer expression of parameter slot (the lane),
 * at lanes 0..w-1: -1 where it has no value */
static int
map_of(nid idx, int slot, int w, int16_t *map)
{
	int l, ok;
	double v;

	for (l = 0; l < w; l++) {
		v = eval_const_at(idx, slot, (double)l, &ok);
		map[l] = (int16_t)(ok && v >= 0 && v < w && v == floor(v) ?
		    (int)v : -1);
		if (!ok || v < 0 || v >= w || v != floor(v))
			return 0;
	}
	return 1;
}

static void
plan_report(const struct model *m, const struct lperm_plan *p)
{
	struct sbuf b = {0};
	int j;

	lperm_text(&b, p);
	if (p->affine) {
		printf("  affine: lane -> A lane xor %d, A's columns (lane bit j "
		    "goes to)", p->c);
		for (j = 0; j < p->bits; j++)
			printf(" %d", p->col[j]);
		printf("; rank(I xor A) = %d\n", p->rank);
	} else if (p->benes)
		printf("  not affine over F2, a permutation of the lanes (an unread "
		    "lane's source any the others leave): a Benes network\n");
	else
		printf("  not affine over F2, and it copies a lane to two: one "
		    "shuffle by its index\n");
	if (p->nst) {
		struct sbuf s = {0};
		struct lperm_plan q = *p;
		q.kind = LPH_STAGES;
		lperm_text(&s, &q);
		printf("  stages: %s, cost %g%s\n", sb_str(&s), p->stages_price,
		    p->kind == LPH_STAGES ? "" : " (not the cheapest)");
		free(s.p);
	}
	if (isfinite(p->shuffle_price))
		printf("  one shuffle by the index: cost %g\n", p->shuffle_price);
	printf("  chosen under %s: %s, cost %g\n", sym_str(m->name), sb_str(&b),
	    p->price);
	if (p->kind == LPH_STAGES)
		printf("  proved: the stages composed lane by lane give the map on "
		    "every one of the %d lanes (lane l ends with the value of lane "
		    "g_1(...g_n(l)), by the shuffle's and the select's "
		    "definitions)\n",
		    p->w);
	rep_str("model", sym_str(m->name));
	rep_int("lanes", p->w);
	rep_bool("affine", p->affine);
	rep_bool("benes", p->benes);
	if (p->affine) {
		rep_int("constant", p->c);
		rep_list("columns");
		for (j = 0; j < p->bits; j++)
			rep_int(NULL, p->col[j]);
		rep_end();
		rep_int("rank", p->rank);
	}
	rep_str("chosen", p->kind == LPH_STAGES ? "stages" : p->kind ==
	    LPH_QUADBCAST ? "quad_broadcast" : p->kind == LPH_SHUFFLE ?
	    "shuffle" : p->kind == LPH_IDENTITY ? "identity" : "none");
	rep_str("plan", sb_str(&b));
	rep_num("cost", isfinite(p->price) ? p->price : -1);
	if (isfinite(p->stages_price))
		rep_num("stages_cost", p->stages_price);
	if (isfinite(p->shuffle_price))
		rep_num("shuffle_cost", p->shuffle_price);
	rep_list("stages");
	for (j = 0; j < p->nst; j++) {
		rep_obj(NULL);
		rep_int("xor", p->st[j].k);
		rep_int("mask", p->st[j].m);
		rep_int("invert", p->st[j].e);
		if (p->st[j].istab)
			rep_int("lanes_table", (long long)p->st[j].tab);
		rep_str("by", how_name[p->st[j].how]);
		rep_end();
	}
	rep_end();
	if (p->kind == LPH_STAGES)
		rep_proof("exhaustive", "the stages composed on every lane give "
		    "the map");
	free(b.p);
}

int
lperm_command(struct P *p)
{
	static const char *const stop[] = {"model", "under", "lanes", NULL};
	char w[48];
	uint32_t at = cmdp_pos(p);
	const struct model *m = model_active();
	struct lperm_plan pl;
	struct func *f;
	int16_t map[SG_MAXW];
	int W = 0, slot = -1, j, l;
	int64_t v, lo, hi;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (strcmp(w, "lperm") != 0 || func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	f = cmdp_func(p, stop);
	while (!cmdp_eos(p)) {
		if (cmdp_kw(p, "lanes")) {
			if (!cmdp_int(p, &v) || v < 2 || v > SG_MAXW || (v & (v - 1)))
				fail("lperm: lanes N, a power of two from 2 to %d",
				    SG_MAXW);
			W = (int)v;
		} else if (cmdp_kw(p, "model") || cmdp_kw(p, "under")) {
			if (!cmdp_word(p, w, sizeof w) ||
			    (m = model_find(sym_cstr(w))) == NULL)
				fail("lperm: unknown model (see 'models')");
		} else {
			cmdp_fail_here(p);
			fail("lperm INDEX [lanes N] [model M]: INDEX an expression of "
			    "lane");
		}
	}
	for (j = 0; j < f->np; j++)
		if (f->p[j].name == sym_cstr("lane"))
			slot = j;
	if (slot < 0 && f->np == 1)
		slot = 0;
	if (slot < 0 || !TY_ISINT(N.ty[f->body]))
		fail("lperm: the index is an integer expression of lane, the "
		    "lane it reads for each lane");
	if (!W) {
		param_bounds(&f->p[slot], &lo, &hi);
		if (f->p[slot].ranged && lo == 0 && hi >= 1 && hi < SG_MAXW &&
		    !((hi + 1) & hi))
			W = (int)hi + 1;
		else
			W = (int)m->par[MP_WARP];
	}
	if (W < 2 || W > SG_MAXW || (W & (W - 1)))
		fail("lperm: no subgroup size (lanes N, a lane: u32 in 0..N-1, or "
		    "the model's warp)");
	notes_flush();
	if (!map_of(f->body, slot, W, map))
		fail("lperm: %s is not a lane of the %d for every lane",
		    shared_str(f->body, 80), W);
	rep_begin("lperm");
	printf("lperm: the lane map %s over %d lanes\n  map:",
	    shared_str(f->body, 120), W);
	for (l = 0; l < W; l++)
		printf(" %d", map[l]);
	printf("\n");
	rep_expr("index", f->body);
	lperm_plan(m, map, W, -1, &pl);
	if (!pl.affine)		/* one shuffle, its index priced as written */
		lperm_plan(m, map, W, fuse_expr_cost(m, f->body), &pl);
	plan_report(m, &pl);
	rep_end();
	return 1;
}

/* ---- English ---- */

/*
 * "the lane map lane xor (lane & 1) << 1 as xor stages under rdna2",
 * "decompose the lane permutation (lane & 3) * 8 + (lane >> 2) into
 * butterflies", "lane shuffle index lane & 28 as exchanges by xor"
 */
const char *
lperm_nlp_raw(const char *sent)
{
	static char out[400];
	static const char *const keys[] = {"lane map", "lane permutation",
	    "lane shuffle index", "shuffle index", "lane perm", NULL};
	static const char *const ends[] = {" as ", " into ", " under ",
	    " on rdna", " for ", " with ", " model ", " using ", " by ", NULL};
	char buf[400], model[48] = "";
	const char *h = NULL, *e, *q;
	size_t n = strlen(sent), kn = 0;
	int i, w = 0;

	if (n >= sizeof buf)
		return NULL;
	for (i = 0; (size_t)i <= n; i++)
		buf[i] = (char)(sent[i] >= 'A' && sent[i] <= 'Z' ? sent[i] + 32 :
		    sent[i]);
	if (!strncmp(buf, "lperm", 5))
		return NULL;
	for (i = 0; keys[i] && !h; i++)
		if ((h = strstr(buf, keys[i])) != NULL)
			kn = strlen(keys[i]);
	if (!h || strchr(buf, '='))
		return NULL;
	h += kn;
	if (!strncmp(h, " of ", 4))
		h += 3;
	while (*h == ' ')
		h++;
	e = h + strlen(h);
	for (i = 0; ends[i]; i++)
		if ((q = strstr(h, ends[i])) != NULL && q < e)
			e = q;
	while (e > h && (e[-1] == ' ' || e[-1] == '?' || e[-1] == '.'))
		e--;
	if (e == h || !strstr(h, "lane") || (size_t)(e - h) > 200)
		return NULL;
	for (q = buf; (q = strpbrk(q, " ")) != NULL; q++) {
		const char *wd = q + 1;
		size_t wn = strcspn(wd, " ?.,");
		if (wn && wn < sizeof model && q - buf >= 3 &&
		    (!strncmp(q - 3, " on", 3) || !strncmp(q - 3, "der", 3) ||
		    !strncmp(q - 3, "for", 3) || !strncmp(q - 3, "del", 3)) &&
		    model_find(sym_intern(wd, wn)) != NULL) {
			memcpy(model, wd, wn);
			model[wn] = 0;
		}
	}
	if ((q = strstr(buf, " lanes")) != NULL && q > buf) {
		const char *d = q;
		while (d > buf && d[-1] >= '0' && d[-1] <= '9')
			d--;
		if (d < q && d > buf && d[-1] == ' ')
			w = atoi(d);
	}
	snprintf(out, sizeof out, "lperm %.*s", (int)(e - h), h);
	if (w)
		snprintf(out + strlen(out), sizeof out - strlen(out), " lanes %d", w);
	if (model[0])
		snprintf(out + strlen(out), sizeof out - strlen(out), " model %s",
		    model);
	return out;
}
