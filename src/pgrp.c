/*
 * pgrp.c - finite permutation groups (deterministic Schreier-Sims) and
 * finite groups given by Cayley tables.
 *
 * A permutation is n uint32_t images in one flat array; products act on
 * the right (perm_mul(r, p, q): first p, then q).  The base and strong
 * generating set come from the textbook Schreier-Sims algorithm (Holt,
 * Handbook of Computational Group Theory, 4.4.2): every Schreier
 * generator of every level is sifted, so the result is exact, not
 * randomised.  Transversals are explicit (an element per orbit point,
 * all of them in one flat array per level while building, packed at the
 * end), so sifting is a few array lookups per level.  The order is the
 * product of the basic orbit lengths, as a zint.
 */
#include "absalg.h"
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

#define TR_LIMIT ((uint64_t)1 << 27)   /* transversal entries (512 MB) */

/* ---- permutations ---- */

void
perm_mul(uint32_t *r, const uint32_t *p, const uint32_t *q, int n)
{
	int i;

	for (i = 0; i < n; i++)
		r[i] = q[p[i]];
}

void
perm_inv(uint32_t *r, const uint32_t *p, int n)
{
	int i;

	for (i = 0; i < n; i++)
		r[p[i]] = (uint32_t)i;
}

int
perm_is_id(const uint32_t *p, int n)
{
	int i;

	for (i = 0; i < n; i++)
		if (p[i] != (uint32_t)i)
			return 0;
	return 1;
}

static void
perm_id(uint32_t *p, int n)
{
	int i;

	for (i = 0; i < n; i++)
		p[i] = (uint32_t)i;
}

zint
perm_order(const uint32_t *p, int n)
{
	uint8_t *seen = xcalloc((size_t)n + 1, 1);
	zint o = Z_ONE;
	int i;

	for (i = 0; i < n; i++) {
		int len = 0;
		uint32_t j = (uint32_t)i;
		if (seen[i])
			continue;
		while (!seen[j]) {
			seen[j] = 1;
			j = p[j];
			len++;
		}
		o = z_lcm(o, z_from_i64(len));
	}
	free(seen);
	return o;
}

void
perm_print(struct sbuf *b, const uint32_t *p, int n)
{
	uint8_t *seen = xcalloc((size_t)n + 1, 1);
	int i, any = 0;

	for (i = 0; i < n; i++) {
		uint32_t j;
		if (seen[i] || p[i] == (uint32_t)i)
			continue;
		sb_putc(b, '(');
		for (j = (uint32_t)i; !seen[j]; j = p[j]) {
			seen[j] = 1;
			sb_printf(b, "%s%u", j == (uint32_t)i ? "" : " ", j + 1);
		}
		sb_putc(b, ')');
		any = 1;
	}
	if (!any)
		sb_puts(b, "()");
	free(seen);
}

/* ---- Schreier-Sims ---- */

/* the working state: per level an orbit, a point -> index map and the
 * transversal elements (flat) */
struct lvl {
	uint32_t *orb, *tr;     /* orbit points; transversal (norb * n) */
	uint32_t norb, cap;
};

struct ss {
	int n;
	int nb;
	uint32_t *base;
	uint32_t *idx;          /* nb_cap * n: point -> orbit index or NOTOK */
	struct lvl *L;
	int nbcap;
	uint32_t *S;            /* strong generators, ns * n */
	int ns, scap;
	uint64_t trsize;
	uint32_t *tmp, *tmp2, *tmp3;
};

static void
ss_add_gen(struct ss *s, const uint32_t *g)
{
	if (s->ns == s->scap) {
		s->scap = s->scap ? s->scap * 2 : 8;
		s->S = xrealloc(s->S, (size_t)s->scap * s->n * sizeof *s->S);
	}
	memcpy(s->S + (size_t)s->ns * s->n, g, (size_t)s->n * sizeof *g);
	s->ns++;
}

static void
ss_add_base(struct ss *s, uint32_t pt)
{
	int i;

	if (s->nb == s->nbcap) {
		s->nbcap = s->nbcap ? s->nbcap * 2 : 8;
		s->base = xrealloc(s->base, (size_t)s->nbcap * sizeof *s->base);
		s->idx = xrealloc(s->idx, (size_t)s->nbcap * s->n * sizeof *s->idx);
		s->L = xrealloc(s->L, (size_t)s->nbcap * sizeof *s->L);
	}
	s->base[s->nb] = pt;
	memset(&s->L[s->nb], 0, sizeof s->L[s->nb]);
	for (i = 0; i < s->n; i++)
		s->idx[(size_t)s->nb * s->n + i] = NOTOK;
	s->nb++;
}

/* does g fix base points 0 .. l-1 */
static int
fixes_upto(const struct ss *s, const uint32_t *g, int l)
{
	int i;

	for (i = 0; i < l; i++)
		if (g[s->base[i]] != s->base[i])
			return 0;
	return 1;
}

static void
lvl_push(struct ss *s, int l, uint32_t pt, const uint32_t *u)
{
	struct lvl *L = &s->L[l];
	int n = s->n;

	if (L->norb == L->cap) {
		L->cap = L->cap ? L->cap * 2 : 8;
		L->orb = xrealloc(L->orb, (size_t)L->cap * sizeof *L->orb);
		L->tr = xrealloc(L->tr, (size_t)L->cap * n * sizeof *L->tr);
	}
	s->trsize += (uint64_t)n;
	if (s->trsize > TR_LIMIT)
		fail("permutation group: the transversals need more than %llu "
		    "entries (degree %d too large for explicit transversals)",
		    (unsigned long long)TR_LIMIT, n);
	L->orb[L->norb] = pt;
	memcpy(L->tr + (size_t)L->norb * n, u, (size_t)n * sizeof *u);
	s->idx[(size_t)l * n + pt] = L->norb;
	L->norb++;
}

/* the basic orbit and transversal of level l under S^(l) */
static void
ss_orbit(struct ss *s, int l)
{
	struct lvl *L = &s->L[l];
	int n = s->n, k;
	uint32_t i;

	for (i = 0; i < L->norb; i++)
		s->idx[(size_t)l * n + L->orb[i]] = NOTOK;
	s->trsize -= (uint64_t)L->norb * n;
	L->norb = 0;
	perm_id(s->tmp, n);
	lvl_push(s, l, s->base[l], s->tmp);
	for (i = 0; i < L->norb; i++)
		for (k = 0; k < s->ns; k++) {
			const uint32_t *g = s->S + (size_t)k * n;
			uint32_t q;
			if (!fixes_upto(s, g, l))
				continue;
			q = g[L->orb[i]];
			if (s->idx[(size_t)l * n + q] != NOTOK)
				continue;
			perm_mul(s->tmp, L->tr + (size_t)i * n, g, n);
			lvl_push(s, l, q, s->tmp);
		}
}

/* sift g (in place) from level l; returns the level where it stopped
 * (nb when it went through) */
static int
ss_strip(struct ss *s, uint32_t *g, int l)
{
	int n = s->n;

	for (; l < s->nb; l++) {
		uint32_t q = g[s->base[l]], k = s->idx[(size_t)l * n + q];
		if (k == NOTOK)
			return l;
		perm_inv(s->tmp2, s->L[l].tr + (size_t)k * n, n);
		perm_mul(s->tmp3, g, s->tmp2, n);
		memcpy(g, s->tmp3, (size_t)n * sizeof *g);
	}
	return l;
}

static uint32_t
moved_point(const uint32_t *g, int n)
{
	int i;

	for (i = 0; i < n; i++)
		if (g[i] != (uint32_t)i)
			return (uint32_t)i;
	return NOTOK;
}

static void
ss_run(struct ss *s, const uint32_t *gens, int ngens, const uint32_t *pre,
    int npre)
{
	int n = s->n, i, k, l;
	uint32_t *h = xmalloc((size_t)n * sizeof *h);
	uint32_t *u = xmalloc((size_t)n * sizeof *u);

	s->tmp = xmalloc((size_t)n * sizeof *s->tmp);
	s->tmp2 = xmalloc((size_t)n * sizeof *s->tmp2);
	s->tmp3 = xmalloc((size_t)n * sizeof *s->tmp3);
	for (i = 0; i < npre; i++)
		ss_add_base(s, pre[i]);
	for (k = 0; k < ngens; k++) {
		const uint32_t *g = gens + (size_t)k * n;
		if (perm_is_id(g, n))
			continue;
		ss_add_gen(s, g);
		if (fixes_upto(s, g, s->nb))
			ss_add_base(s, moved_point(g, n));
	}
	for (l = 0; l < s->nb; l++)
		ss_orbit(s, l);
	l = s->nb - 1;
	while (l >= 0) {
		struct lvl *L = &s->L[l];
		int restart = 0;
		uint32_t j;
		for (j = 0; j < L->norb && !restart; j++)
			for (k = 0; k < s->ns && !restart; k++) {
				const uint32_t *g = s->S + (size_t)k * n;
				uint32_t q, kk;
				int stop;
				if (!fixes_upto(s, g, l))
					continue;
				/* the Schreier generator u_j g u_{j^g}^-1 */
				q = g[L->orb[j]];
				kk = s->idx[(size_t)l * n + q];
				perm_mul(u, L->tr + (size_t)j * n, g, n);
				perm_inv(s->tmp, L->tr + (size_t)kk * n, n);
				perm_mul(h, u, s->tmp, n);
				stop = ss_strip(s, h, l + 1);
				if (stop < s->nb || !perm_is_id(h, n)) {
					int m;
					ss_add_gen(s, h);
					if (stop == s->nb)
						ss_add_base(s, moved_point(h, n));
					for (m = l + 1; m <= stop && m < s->nb; m++)
						ss_orbit(s, m);
					l = stop < s->nb ? stop : s->nb - 1;
					restart = 1;
				}
			}
		if (!restart)
			l--;
	}
	free(h);
	free(u);
}

/* pack the state into G */
static void
ss_pack(struct pgrp *G, struct ss *s)
{
	int n = s->n, l;
	uint64_t tot = 0, o;

	G->nbase = s->nb;
	G->base = xmalloc(((size_t)s->nb + 1) * sizeof *G->base);
	memcpy(G->base, s->base, (size_t)s->nb * sizeof *G->base);
	G->nsgs = s->ns;
	G->sgs = xmalloc(((size_t)s->ns * n + 1) * sizeof *G->sgs);
	memcpy(G->sgs, s->S, (size_t)s->ns * n * sizeof *G->sgs);
	G->tro = xmalloc(((size_t)s->nb + 1) * sizeof *G->tro);
	for (l = 0; l < s->nb; l++) {
		G->tro[l] = tot;
		tot += s->L[l].norb;
	}
	G->tro[s->nb] = tot;
	G->tr = xmalloc((tot * n + 1) * sizeof *G->tr);
	G->orb = xmalloc((tot + 1) * sizeof *G->orb);
	G->trk = xmalloc(((size_t)s->nb * n + 1) * sizeof *G->trk);
	memcpy(G->trk, s->idx, (size_t)s->nb * n * sizeof *G->trk);
	G->order = Z_ONE;
	for (l = 0, o = 0; l < s->nb; l++) {
		memcpy(G->orb + o, s->L[l].orb, s->L[l].norb * sizeof *G->orb);
		memcpy(G->tr + o * n, s->L[l].tr, (size_t)s->L[l].norb * n *
		    sizeof *G->tr);
		o += s->L[l].norb;
		G->order = z_mul(G->order, z_from_i64(s->L[l].norb));
		free(s->L[l].orb);
		free(s->L[l].tr);
	}
	free(s->base);
	free(s->idx);
	free(s->L);
	free(s->S);
	free(s->tmp);
	free(s->tmp2);
	free(s->tmp3);
}

static void
pg_build(struct pgrp *G, int n, const uint32_t *gens, int ngens,
    const uint32_t *pre, int npre)
{
	struct ss s;
	struct pgrp T;
	int k, i;

	if (n <= 0)
		n = 1;
	for (k = 0; k < ngens; k++) {
		uint8_t *seen = xcalloc((size_t)n, 1);
		for (i = 0; i < n; i++) {
			uint32_t v = gens[(size_t)k * n + i];
			if (v >= (uint32_t)n || seen[v])
				fail("generator %d is not a permutation of 1..%d", k + 1,
				    n);
			seen[v] = 1;
		}
		free(seen);
	}
	memset(&s, 0, sizeof s);
	memset(&T, 0, sizeof T);
	s.n = n;
	ss_run(&s, gens, ngens, pre, npre);
	T.n = n;
	T.ngens = ngens;
	T.gens = xmalloc(((size_t)ngens * n + 1) * sizeof *T.gens);
	memcpy(T.gens, gens, (size_t)ngens * n * sizeof *T.gens);
	ss_pack(&T, &s);
	pg_free(G);
	*G = T;
}

void
pg_init(struct pgrp *G, int n, const uint32_t *gens, int ngens)
{
	pg_build(G, n, gens, ngens, NULL, 0);
}

void
pg_free(struct pgrp *G)
{
	free(G->gens);
	free(G->base);
	free(G->sgs);
	free(G->trk);
	free(G->tro);
	free(G->tr);
	free(G->orb);
	memset(G, 0, sizeof *G);
}

void
pg_copy(struct pgrp *dst, const struct pgrp *src)
{
	pg_build(dst, src->n, src->gens, src->ngens, src->base, src->nbase);
}

/* ---- standard groups ---- */

void
pg_symmetric(struct pgrp *G, int n)
{
	uint32_t *g = xmalloc((size_t)2 * (n > 0 ? n : 1) * sizeof *g);
	int i, k = n >= 2 ? 2 : 0;

	for (i = 0; i < n; i++) {
		g[i] = (uint32_t)((i + 1) % n);
		g[n + i] = (uint32_t)i;
	}
	if (n >= 2) {
		g[n + 0] = 1;
		g[n + 1] = 0;
	}
	pg_init(G, n, g, k);
	free(g);
}

void
pg_alternating(struct pgrp *G, int n)
{
	uint32_t *g;
	int i, k = 0;

	if (n < 3) {
		pg_init(G, n > 0 ? n : 1, NULL, 0);
		return;
	}
	g = xmalloc((size_t)(n - 2) * n * sizeof *g);
	/* the 3-cycles (1 2 i), i = 3 .. n */
	for (k = 0; k < n - 2; k++) {
		uint32_t *p = g + (size_t)k * n;
		perm_id(p, n);
		p[0] = 1;
		p[1] = (uint32_t)(k + 2);
		p[k + 2] = 0;
	}
	(void)i;
	pg_init(G, n, g, n - 2);
	free(g);
}

void
pg_cyclic(struct pgrp *G, int n)
{
	uint32_t *g = xmalloc((size_t)(n > 0 ? n : 1) * sizeof *g);
	int i;

	for (i = 0; i < n; i++)
		g[i] = (uint32_t)((i + 1) % n);
	pg_init(G, n, g, n >= 2 ? 1 : 0);
	free(g);
}

void
pg_dihedral(struct pgrp *G, int n)
{
	uint32_t *g = xmalloc((size_t)2 * (n > 0 ? n : 1) * sizeof *g);
	int i;

	for (i = 0; i < n; i++) {
		g[i] = (uint32_t)((i + 1) % n);
		g[n + i] = (uint32_t)((n - i) % n);
	}
	pg_init(G, n, g, n >= 2 ? 2 : 0);
	free(g);
}

/* "(1 2 3)(4 5), (1 2)", "(1,2,3)", "S4", "A5", "C6", "D8", "Sym(4)" */
int
pg_parse(struct pgrp *G, const char *s, int n, const char **end)
{
	uint32_t *cyc = NULL, *gens = NULL;
	int ncyc = 0, ccap = 0, ngens = 0, gcap = 0, maxpt = 0, i;
	/* generators as lists of cycles: (len, points...) records, -1 ends */
	int *rec = NULL, nrec = 0, rcap = 0;

#define RPUSH(v) do { \
		if (nrec == rcap) { \
			rcap = rcap ? rcap * 2 : 64; \
			rec = xrealloc(rec, (size_t)rcap * sizeof *rec); \
		} \
		rec[nrec++] = (v); \
	} while (0)
	(void)cyc;
	(void)ncyc;
	(void)ccap;
	while (*s == ' ')
		s++;
	if ((s[0] == 'S' || s[0] == 'A' || s[0] == 'C' || s[0] == 'D') &&
	    ((s[1] >= '0' && s[1] <= '9') || !strncmp(s, "Sym(", 4) ||
	    !strncmp(s, "Alt(", 4))) {
		char kind = s[0];
		long v;
		char *e;
		const char *t = s + 1;
		int paren = 0;
		if (!strncmp(s, "Sym(", 4) || !strncmp(s, "Alt(", 4)) {
			t = s + 4;
			paren = 1;
		}
		v = strtol(t, &e, 10);
		if (paren) {
			if (*e != ')')
				return 0;
			e++;
		}
		if (v < 1 || v > 100000)
			return 0;
		if (kind == 'S')
			pg_symmetric(G, (int)v);
		else if (kind == 'A')
			pg_alternating(G, (int)v);
		else if (kind == 'C')
			pg_cyclic(G, (int)v);
		else {
			if (v % 2 || v < 4)
				fail("D%ld: the dihedral group of order %ld needs an even "
				    "order >= 4 (D8 is the symmetries of a square)", v, v);
			pg_dihedral(G, (int)(v / 2));
		}
		if (end)
			*end = e;
		return 1;
	}
	for (;;) {
		int any = 0;
		while (*s == ' ')
			s++;
		while (*s == '(') {
			int len = 0, at = nrec;
			s++;
			RPUSH(0);
			for (;;) {
				long v;
				char *e;
				while (*s == ' ' || *s == ',')
					s++;
				if (*s == ')') {
					s++;
					break;
				}
				v = strtol(s, &e, 10);
				if (e == s || v < 1 || v > 1000000) {
					free(rec);
					return 0;
				}
				s = e;
				RPUSH((int)v - 1);
				if (v > maxpt)
					maxpt = (int)v;
				len++;
			}
			rec[at] = len;
			any = 1;
			while (*s == ' ')
				s++;
		}
		if (!any)
			break;
		RPUSH(-1);
		ngens++;
		while (*s == ' ')
			s++;
		if (*s == ',')
			s++;
		else
			break;
	}
	if (ngens == 0) {
		free(rec);
		return 0;
	}
	if (n <= 0)
		n = maxpt > 0 ? maxpt : 1;
	if (maxpt > n)
		fail("permutation: point %d is beyond the degree %d", maxpt, n);
	gcap = ngens;
	gens = xmalloc((size_t)gcap * n * sizeof *gens);
	{
		int r = 0, k;
		for (k = 0; k < ngens; k++) {
			uint32_t *g = gens + (size_t)k * n;
			uint32_t *c = xmalloc((size_t)n * sizeof *c);
			perm_id(g, n);
			while (rec[r] != -1) {
				int len = rec[r++], j;
				uint32_t *pts = xmalloc((size_t)(len + 1) * sizeof *pts);
				for (j = 0; j < len; j++)
					pts[j] = (uint32_t)rec[r++];
				/* the cycle as a permutation, then g = g * cycle */
				perm_id(c, n);
				for (j = 0; j < len; j++) {
					if (c[pts[j]] != pts[j])
						fail("a point repeats in a cycle");
					c[pts[j]] = pts[(j + 1) % len];
				}
				{
					uint32_t *t = xmalloc((size_t)n * sizeof *t);
					perm_mul(t, g, c, n);
					memcpy(g, t, (size_t)n * sizeof *t);
					free(t);
				}
				free(pts);
			}
			r++;
			free(c);
		}
	}
	(void)i;
	pg_init(G, n, gens, ngens);
	free(gens);
	free(rec);
#undef RPUSH
	if (end)
		*end = s;
	return 1;
}

/* ---- queries ---- */

zint
pg_order(const struct pgrp *G)
{
	return G->order;
}

static const uint32_t *
tr_elem(const struct pgrp *G, int l, uint32_t k)
{
	return G->tr + (G->tro[l] + k) * (uint64_t)G->n;
}

int
pg_contains(const struct pgrp *G, const uint32_t *p, uint32_t *kout)
{
	int n = G->n, l, ok = 1;
	uint32_t *g = xmalloc((size_t)n * sizeof *g);
	uint32_t *t = xmalloc((size_t)n * sizeof *t);
	uint32_t *iv = xmalloc((size_t)n * sizeof *iv);

	memcpy(g, p, (size_t)n * sizeof *g);
	for (l = 0; l < G->nbase; l++) {
		uint32_t q = g[G->base[l]], k = G->trk[(size_t)l * n + q];
		if (k == NOTOK) {
			ok = 0;
			break;
		}
		if (kout)
			kout[l] = q;
		perm_inv(iv, tr_elem(G, l, k), n);
		perm_mul(t, g, iv, n);
		memcpy(g, t, (size_t)n * sizeof *g);
	}
	if (ok && !perm_is_id(g, n))
		ok = 0;
	if (ok && kout) {
		/* the certificate rebuilds p */
		pg_word_eval(G, kout, t);
		if (memcmp(t, p, (size_t)n * sizeof *t))
			fail("internal: membership certificate failed");
	}
	free(g);
	free(t);
	free(iv);
	return ok;
}

void
pg_word_eval(const struct pgrp *G, const uint32_t *k, uint32_t *p)
{
	int n = G->n, l;
	uint32_t *t = xmalloc((size_t)n * sizeof *t);

	perm_id(p, n);
	for (l = 0; l < G->nbase; l++) {
		uint32_t kk = G->trk[(size_t)l * n + k[l]];
		if (kk == NOTOK)
			fail("internal: a word with a point outside a basic orbit");
		/* p <- u_l p (u_l acts first) */
		perm_mul(t, tr_elem(G, l, kk), p, n);
		memcpy(p, t, (size_t)n * sizeof *t);
	}
	free(t);
}

int
pg_orbits(const struct pgrp *G, uint32_t *orbit_of)
{
	int n = G->n, i, k, c = 0;
	uint32_t *q = xmalloc((size_t)n * sizeof *q);

	for (i = 0; i < n; i++)
		orbit_of[i] = NOTOK;
	for (i = 0; i < n; i++) {
		int h = 0, t = 0;
		if (orbit_of[i] != NOTOK)
			continue;
		orbit_of[i] = (uint32_t)c;
		q[t++] = (uint32_t)i;
		while (h < t) {
			uint32_t x = q[h++];
			for (k = 0; k < G->ngens; k++) {
				uint32_t y = G->gens[(size_t)k * n + x];
				if (orbit_of[y] == NOTOK) {
					orbit_of[y] = (uint32_t)c;
					q[t++] = y;
				}
			}
		}
		c++;
	}
	free(q);
	return c;
}

int
pg_orbit(const struct pgrp *G, uint32_t pt, uint32_t *out)
{
	uint32_t *of = xmalloc((size_t)G->n * sizeof *of);
	int i, m = 0;

	pg_orbits(G, of);
	for (i = 0; i < G->n; i++)
		if (of[i] == of[pt])
			out[m++] = (uint32_t)i;
	free(of);
	return m;
}

void
pg_stabilizer(struct pgrp *r, const struct pgrp *G, uint32_t pt)
{
	struct pgrp T;
	uint32_t *g;
	int k, m = 0, n = G->n;

	memset(&T, 0, sizeof T);
	pg_build(&T, n, G->gens, G->ngens, &pt, 1);
	/* the strong generators fixing the first base point (pt) */
	g = xmalloc(((size_t)T.nsgs * n + 1) * sizeof *g);
	for (k = 0; k < T.nsgs; k++)
		if (T.sgs[(size_t)k * n + pt] == pt)
			memcpy(g + (size_t)m++ * n, T.sgs + (size_t)k * n,
			    (size_t)n * sizeof *g);
	pg_init(r, n, g, m);
	if (!z_eq(z_mul(r->order, z_from_i64(T.tro[1] - T.tro[0])), G->order))
		fail("internal: orbit-stabilizer check failed");
	free(g);
	pg_free(&T);
}

int
pg_is_subgroup(const struct pgrp *H, const struct pgrp *G)
{
	int k;

	if (H->n != G->n)
		return 0;
	for (k = 0; k < H->ngens; k++)
		if (!pg_contains(G, H->gens + (size_t)k * H->n, NULL))
			return 0;
	return 1;
}

/* r = g^-1 h g */
static void
conj(uint32_t *r, const uint32_t *h, const uint32_t *g, int n)
{
	uint32_t *gi = xmalloc((size_t)n * sizeof *gi);
	uint32_t *t = xmalloc((size_t)n * sizeof *t);

	perm_inv(gi, g, n);
	perm_mul(t, gi, h, n);
	perm_mul(r, t, g, n);
	free(gi);
	free(t);
}

int
pg_is_normal(const struct pgrp *H, const struct pgrp *G)
{
	int n = G->n, i, k, ok = 1;
	uint32_t *c = xmalloc((size_t)n * sizeof *c);

	if (!pg_is_subgroup(H, G))
		ok = 0;
	for (i = 0; ok && i < G->ngens; i++)
		for (k = 0; ok && k < H->ngens; k++) {
			conj(c, H->gens + (size_t)k * n, G->gens + (size_t)i * n, n);
			if (!pg_contains(H, c, NULL))
				ok = 0;
		}
	free(c);
	return ok;
}

int
pg_is_abelian(const struct pgrp *G)
{
	int n = G->n, i, k, ok = 1;
	uint32_t *a = xmalloc((size_t)n * sizeof *a);
	uint32_t *b = xmalloc((size_t)n * sizeof *b);

	for (i = 0; ok && i < G->ngens; i++)
		for (k = i + 1; ok && k < G->ngens; k++) {
			perm_mul(a, G->gens + (size_t)i * n, G->gens + (size_t)k * n, n);
			perm_mul(b, G->gens + (size_t)k * n, G->gens + (size_t)i * n, n);
			if (memcmp(a, b, (size_t)n * sizeof *a))
				ok = 0;
		}
	free(a);
	free(b);
	return ok;
}

void
pg_normal_closure(struct pgrp *r, const struct pgrp *G, const uint32_t *gens,
    int k)
{
	int n = G->n, cap = k + 8, m = 0, i, j, changed = 1;
	uint32_t *g = xmalloc((size_t)cap * n * sizeof *g);
	uint32_t *c = xmalloc((size_t)n * sizeof *c);
	struct pgrp N;

	memset(&N, 0, sizeof N);
	for (i = 0; i < k; i++)
		if (!perm_is_id(gens + (size_t)i * n, n))
			memcpy(g + (size_t)m++ * n, gens + (size_t)i * n,
			    (size_t)n * sizeof *g);
	pg_init(&N, n, g, m);
	while (changed) {
		changed = 0;
		for (i = 0; i < G->ngens && !changed; i++)
			for (j = 0; j < m && !changed; j++) {
				conj(c, g + (size_t)j * n, G->gens + (size_t)i * n, n);
				if (pg_contains(&N, c, NULL))
					continue;
				if (m == cap) {
					cap *= 2;
					g = xrealloc(g, (size_t)cap * n * sizeof *g);
				}
				memcpy(g + (size_t)m++ * n, c, (size_t)n * sizeof *c);
				pg_init(&N, n, g, m);
				changed = 1;
			}
	}
	pg_free(r);
	*r = N;
	free(g);
	free(c);
}

void
pg_commutator(struct pgrp *r, const struct pgrp *G, const struct pgrp *H,
    const struct pgrp *K)
{
	int n = G->n, i, j, m = 0;
	uint32_t *g = xmalloc(((size_t)H->ngens * K->ngens + 1) * n * sizeof *g);
	uint32_t *hi = xmalloc((size_t)n * sizeof *hi);
	uint32_t *ki = xmalloc((size_t)n * sizeof *ki);
	uint32_t *t = xmalloc((size_t)n * sizeof *t);
	uint32_t *t2 = xmalloc((size_t)n * sizeof *t2);

	/* [h, k] = h^-1 k^-1 h k */
	for (i = 0; i < H->ngens; i++)
		for (j = 0; j < K->ngens; j++) {
			const uint32_t *h = H->gens + (size_t)i * n;
			const uint32_t *k = K->gens + (size_t)j * n;
			perm_inv(hi, h, n);
			perm_inv(ki, k, n);
			perm_mul(t, hi, ki, n);
			perm_mul(t2, t, h, n);
			perm_mul(g + (size_t)m++ * n, t2, k, n);
		}
	pg_normal_closure(r, G, g, m);
	free(g);
	free(hi);
	free(ki);
	free(t);
	free(t2);
}

void
pg_derived(struct pgrp *r, const struct pgrp *G)
{
	pg_commutator(r, G, G, G);
}

int
pg_derived_series(const struct pgrp *G, zint *orders, int max)
{
	struct pgrp A, B;
	int len = 0;

	memset(&A, 0, sizeof A);
	memset(&B, 0, sizeof B);
	pg_copy(&A, G);
	for (;;) {
		if (len < max)
			orders[len] = A.order;
		len++;
		if (z_is_one(A.order))
			break;
		pg_derived(&B, &A);
		if (z_eq(B.order, A.order))
			break;
		pg_free(&A);
		A = B;
		memset(&B, 0, sizeof B);
	}
	pg_free(&A);
	pg_free(&B);
	return len;
}

int
pg_lower_central(const struct pgrp *G, zint *orders, int max)
{
	struct pgrp A, B;
	int len = 0;

	memset(&A, 0, sizeof A);
	memset(&B, 0, sizeof B);
	pg_copy(&A, G);
	for (;;) {
		if (len < max)
			orders[len] = A.order;
		len++;
		if (z_is_one(A.order))
			break;
		pg_commutator(&B, G, &A, G);
		if (z_eq(B.order, A.order))
			break;
		pg_free(&A);
		A = B;
		memset(&B, 0, sizeof B);
	}
	pg_free(&A);
	pg_free(&B);
	return len;
}

int
pg_is_solvable(const struct pgrp *G)
{
	zint o[64];
	int len = pg_derived_series(G, o, 64);

	return z_is_one(o[(len < 64 ? len : 64) - 1]);
}

int
pg_is_nilpotent(const struct pgrp *G)
{
	zint o[64];
	int len = pg_lower_central(G, o, 64);

	return z_is_one(o[(len < 64 ? len : 64) - 1]);
}

uint32_t
pg_elements(const struct pgrp *G, uint32_t **out)
{
	int n = G->n, l, m = G->nbase;
	u64 total;
	uint32_t *e, *k, *t, cnt = 0;

	if (!z_to_u64(G->order, &total) || total > PG_ENUM_MAX)
		fail("the group has %s elements, more than I enumerate (%d)",
		    z_str(G->order), PG_ENUM_MAX);
	e = xmalloc(((size_t)total * n + 1) * sizeof *e);
	k = xcalloc((size_t)m + 1, sizeof *k);
	t = xmalloc((size_t)n * sizeof *t);
	/* every choice of transversal elements, as a mixed-radix counter */
	for (;;) {
		uint32_t *p = e + (size_t)cnt * n;
		perm_id(p, n);
		for (l = 0; l < m; l++) {
			perm_mul(t, tr_elem(G, l, k[l]), p, n);
			memcpy(p, t, (size_t)n * sizeof *t);
		}
		cnt++;
		for (l = m - 1; l >= 0; l--) {
			if (++k[l] < G->tro[l + 1] - G->tro[l])
				break;
			k[l] = 0;
		}
		if (l < 0)
			break;
	}
	free(k);
	free(t);
	*out = e;
	return cnt;
}

static void
subgroup_of_elems(struct pgrp *r, const struct pgrp *G, const uint32_t *e,
    uint32_t m)
{
	/* generate greedily: add an element when it is not yet inside */
	int n = G->n, k = 0, cap = 8;
	uint32_t *g = xmalloc((size_t)cap * n * sizeof *g), i;
	struct pgrp S;

	memset(&S, 0, sizeof S);
	pg_init(&S, n, g, 0);
	for (i = 0; i < m; i++) {
		if (pg_contains(&S, e + (size_t)i * n, NULL))
			continue;
		if (k == cap) {
			cap *= 2;
			g = xrealloc(g, (size_t)cap * n * sizeof *g);
		}
		memcpy(g + (size_t)k++ * n, e + (size_t)i * n, (size_t)n * sizeof *g);
		pg_init(&S, n, g, k);
	}
	pg_free(r);
	*r = S;
	free(g);
}

static int
commutes(const uint32_t *a, const uint32_t *b, int n, uint32_t *t1,
    uint32_t *t2)
{
	perm_mul(t1, a, b, n);
	perm_mul(t2, b, a, n);
	return !memcmp(t1, t2, (size_t)n * sizeof *t1);
}

void
pg_centralizer(struct pgrp *r, const struct pgrp *G, const uint32_t *x)
{
	uint32_t *e, m, i, keep = 0;
	int n = G->n;
	uint32_t *t1 = xmalloc((size_t)n * sizeof *t1);
	uint32_t *t2 = xmalloc((size_t)n * sizeof *t2);

	m = pg_elements(G, &e);
	for (i = 0; i < m; i++)
		if (commutes(e + (size_t)i * n, x, n, t1, t2))
			memmove(e + (size_t)keep++ * n, e + (size_t)i * n,
			    (size_t)n * sizeof *e);
	subgroup_of_elems(r, G, e, keep);
	free(e);
	free(t1);
	free(t2);
}

void
pg_centre(struct pgrp *r, const struct pgrp *G)
{
	uint32_t *e, m, i, keep = 0;
	int n = G->n, k;
	uint32_t *t1 = xmalloc((size_t)n * sizeof *t1);
	uint32_t *t2 = xmalloc((size_t)n * sizeof *t2);

	m = pg_elements(G, &e);
	for (i = 0; i < m; i++) {
		int ok = 1;
		for (k = 0; k < G->ngens && ok; k++)
			ok = commutes(e + (size_t)i * n, G->gens + (size_t)k * n, n,
			    t1, t2);
		if (ok)
			memmove(e + (size_t)keep++ * n, e + (size_t)i * n,
			    (size_t)n * sizeof *e);
	}
	subgroup_of_elems(r, G, e, keep);
	free(e);
	free(t1);
	free(t2);
}

zint
pg_index(const struct pgrp *G, const struct pgrp *H)
{
	if (!pg_is_subgroup(H, G))
		fail("index: not a subgroup");
	return z_divexact(G->order, H->order);
}

uint32_t
pg_cosets(const struct pgrp *G, const struct pgrp *H, uint32_t **reps)
{
	uint32_t *e, m, i, j, nr = 0;
	int n = G->n;
	uint32_t *t = xmalloc((size_t)n * sizeof *t);
	uint32_t *iv = xmalloc((size_t)n * sizeof *iv);
	uint32_t *r;
	u64 idx;

	if (!z_to_u64(pg_index(G, H), &idx) || idx > 100000)
		fail("cosets: the index %s is too large to list", z_str(pg_index(G,
		    H)));
	m = pg_elements(G, &e);
	r = xmalloc(((size_t)idx * n + 1) * sizeof *r);
	/* right cosets H g: g, g' together iff g g'^-1 in H */
	for (i = 0; i < m && nr < idx; i++) {
		const uint32_t *g = e + (size_t)i * n;
		int found = 0;
		for (j = 0; j < nr && !found; j++) {
			perm_inv(iv, r + (size_t)j * n, n);
			perm_mul(t, g, iv, n);
			if (pg_contains(H, t, NULL))
				found = 1;
		}
		if (!found)
			memcpy(r + (size_t)nr++ * n, g, (size_t)n * sizeof *g);
	}
	free(e);
	free(t);
	free(iv);
	*reps = r;
	return nr;
}

int
pg_hom_check(const struct pgrp *G, const struct pgrp *H, const uint32_t *imgs)
{
	int n = G->n, m = H->n, k, i, ok;
	uint32_t *g;
	struct pgrp Gr;

	for (k = 0; k < G->ngens; k++)
		if (!pg_contains(H, imgs + (size_t)k * m, NULL))
			return 0;
	/* the graph <(g_i, h_i)> in Sym(n + m) */
	g = xmalloc(((size_t)G->ngens * (n + m) + 1) * sizeof *g);
	for (k = 0; k < G->ngens; k++) {
		uint32_t *p = g + (size_t)k * (n + m);
		for (i = 0; i < n; i++)
			p[i] = G->gens[(size_t)k * n + i];
		for (i = 0; i < m; i++)
			p[n + i] = (uint32_t)n + imgs[(size_t)k * m + i];
	}
	memset(&Gr, 0, sizeof Gr);
	pg_init(&Gr, n + m, g, G->ngens);
	ok = z_eq(Gr.order, G->order);
	pg_free(&Gr);
	free(g);
	return ok;
}

/* ---- Cayley tables ---- */

int
cg_init(struct cgrp *G, int n, const uint32_t *tab, const char **why)
{
	int a, b, c, e = -1;
	uint32_t *inv;

	memset(G, 0, sizeof *G);
	if (n <= 0) {
		*why = "the table is empty";
		return 0;
	}
	for (a = 0; a < n * n; a++)
		if (tab[a] >= (uint32_t)n) {
			*why = "an entry is not an element (entries are 0 .. n-1)";
			return 0;
		}
	for (a = 0; a < n && e < 0; a++) {
		int ok = 1;
		for (b = 0; b < n && ok; b++)
			if (tab[a * n + b] != (uint32_t)b || tab[b * n + a] !=
			    (uint32_t)b)
				ok = 0;
		if (ok)
			e = a;
	}
	if (e < 0) {
		*why = "no identity element";
		return 0;
	}
	inv = xmalloc((size_t)n * sizeof *inv);
	for (a = 0; a < n; a++) {
		inv[a] = NOTOK;
		for (b = 0; b < n; b++)
			if (tab[a * n + b] == (uint32_t)e && tab[b * n + a] ==
			    (uint32_t)e) {
				inv[a] = (uint32_t)b;
				break;
			}
		if (inv[a] == NOTOK) {
			free(inv);
			*why = "an element has no inverse";
			return 0;
		}
	}
	for (a = 0; a < n; a++)
		for (b = 0; b < n; b++) {
			uint32_t ab = tab[a * n + b];
			for (c = 0; c < n; c++)
				if (tab[ab * n + c] != tab[a * n + tab[b * n + c]]) {
					free(inv);
					*why = "the operation is not associative";
					return 0;
				}
		}
	G->n = n;
	G->tab = xmalloc((size_t)n * n * sizeof *G->tab);
	memcpy(G->tab, tab, (size_t)n * n * sizeof *G->tab);
	G->inv = inv;
	G->e = (uint32_t)e;
	return 1;
}

void
cg_free(struct cgrp *G)
{
	free(G->tab);
	free(G->inv);
	memset(G, 0, sizeof *G);
}

int
cg_parse(struct cgrp *G, const char *s, const char **why)
{
	struct zmat m;
	uint32_t *t;
	int i, ok;
	const char *e;

	memset(&m, 0, sizeof m);
	if (!zm_parse(&m, s, &e) || m.r != m.c || m.r == 0) {
		*why = "expected a square table [[0, 1], [1, 0]]";
		zm_free(&m);
		return 0;
	}
	t = xmalloc((size_t)m.r * m.c * sizeof *t);
	for (i = 0; i < m.r * m.c; i++) {
		int64_t v = z_fits_i64(m.a[i]) ? z_to_i64(m.a[i]) : -1;
		if (v < 0 || v >= m.r) {
			*why = "an entry is not an element (entries are 0 .. n-1)";
			free(t);
			zm_free(&m);
			return 0;
		}
		t[i] = (uint32_t)v;
	}
	ok = cg_init(G, m.r, t, why);
	free(t);
	zm_free(&m);
	return ok;
}

void
cg_from_pgrp(struct cgrp *r, const struct pgrp *G)
{
	uint32_t *e, m, i, j, k;
	int n = G->n;
	uint32_t *tab, *t = xmalloc((size_t)n * sizeof *t);
	const char *why;

	m = pg_elements(G, &e);
	if (m > 4096)
		fail("a Cayley table of %u elements is too large", m);
	tab = xmalloc((size_t)m * m * sizeof *tab);
	for (i = 0; i < m; i++)
		for (j = 0; j < m; j++) {
			perm_mul(t, e + (size_t)i * n, e + (size_t)j * n, n);
			for (k = 0; k < m; k++)
				if (!memcmp(t, e + (size_t)k * n, (size_t)n * sizeof *t))
					break;
			tab[i * m + j] = k;
		}
	if (!cg_init(r, (int)m, tab, &why))
		fail("internal: Cayley table of a permutation group: %s", why);
	free(tab);
	free(e);
	free(t);
}

void
cg_to_pgrp(struct pgrp *r, const struct cgrp *G)
{
	/* the right regular representation: a acts by x -> x a */
	int n = G->n, a, x;
	uint32_t *g = xmalloc((size_t)n * n * sizeof *g);

	for (a = 0; a < n; a++)
		for (x = 0; x < n; x++)
			g[(size_t)a * n + x] = G->tab[x * n + a];
	pg_init(r, n, g, n);
	free(g);
	if (!z_eq(r->order, z_from_i64(n)))
		fail("internal: the regular representation has the wrong order");
}

uint32_t
cg_elem_order(const struct cgrp *G, uint32_t a)
{
	uint32_t x = a, k = 1;

	while (x != G->e) {
		x = G->tab[x * G->n + a];
		k++;
	}
	return k;
}

int
cg_is_abelian(const struct cgrp *G)
{
	int a, b, n = G->n;

	for (a = 0; a < n; a++)
		for (b = a + 1; b < n; b++)
			if (G->tab[a * n + b] != G->tab[b * n + a])
				return 0;
	return 1;
}

int
cg_hom_check(const struct cgrp *G, const struct cgrp *H, const uint32_t *f,
    uint32_t *ba, uint32_t *bb)
{
	int a, b, n = G->n, m = H->n;

	for (a = 0; a < n; a++)
		if (f[a] >= (uint32_t)m)
			fail("homomorphism: %u is not an element of the target",
			    f[a]);
	for (a = 0; a < n; a++)
		for (b = 0; b < n; b++)
			if (f[G->tab[a * n + b]] != H->tab[f[a] * m + f[b]]) {
				if (ba)
					*ba = (uint32_t)a;
				if (bb)
					*bb = (uint32_t)b;
				return 0;
			}
	return 1;
}
