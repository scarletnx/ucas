/*
 * simp.c - finite cell complexes for homalg (homalg.h, "spaces").
 *
 * Storage (struct of arrays, one arena for the process):
 *   spaces      top[] kind[] name[] cnt_o[] bd_o[] sx_o[]   (per handle)
 *   cnt[]       cell counts, top+1 per space from cnt_o
 *   bdo[]       top+2 offsets per space from bd_o into the triplets:
 *               d_k is tr[bdo[o+k] .. bdo[o+k+1]) (d_0 is empty)
 *   tr_r tr_c tr_v   boundary triplets, sorted by (col, row) per block
 *   sxo[] sxv[] simplicial: dimension k at sxv[sxo[o+k]], k+1 columns of
 *               cnt[k] vertices each (column j at + j*cnt[k])
 *
 * A space is built in the one static builder B and committed; the
 * builder's buffers are reused, so a fail() halfway leaks nothing.
 */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "ucas.h"
#include "homalg.h"

typedef __int128 i128;

/* ---- the arena ---- */

static struct {
	uint32_t n, cap;
	int8_t *top;
	uint8_t *kind;
	uint32_t *name, *cnt_o, *bd_o, *sx_o;
} S;

struct u32v {
	uint32_t *p;
	uint32_t n, cap;
};
struct i64v {
	int64_t *p;
	uint32_t n, cap;
};

static struct u32v cnt, bdo, tr_r, tr_c, sxo, sxv;
static struct i64v tr_v;

static void
u32_push(struct u32v *a, uint32_t x)
{
	if (a->n == a->cap) {
		a->cap = newcap(a->cap, a->n + 1);
		a->p = xrealloc(a->p, a->cap * sizeof *a->p);
	}
	a->p[a->n++] = x;
}

static void
i64_push(struct i64v *a, int64_t x)
{
	if (a->n == a->cap) {
		a->cap = newcap(a->cap, a->n + 1);
		a->p = xrealloc(a->p, a->cap * sizeof *a->p);
	}
	a->p[a->n++] = x;
}

static void
chk(hspace s)
{
	if (s >= S.n)
		fail("homalg: no space %u", (unsigned)s);
}

int
hs_top(hspace s)
{
	chk(s);
	return S.top[s];
}

int
hs_kind(hspace s)
{
	chk(s);
	return S.kind[s];
}

const char *
hs_name(hspace s)
{
	chk(s);
	return sym_str(S.name[s]);
}

uint32_t
hs_ncells(hspace s, int k)
{
	chk(s);
	if (k < 0 || k > S.top[s])
		return 0;
	return cnt.p[S.cnt_o[s] + k];
}

uint32_t
hs_bd(hspace s, int k, const uint32_t **row, const uint32_t **col,
    const int64_t **val)
{
	uint32_t a, b;

	chk(s);
	if (k < 1 || k > S.top[s]) {
		*row = *col = NULL;
		*val = NULL;
		return 0;
	}
	a = bdo.p[S.bd_o[s] + k];
	b = bdo.p[S.bd_o[s] + k + 1];
	*row = tr_r.p + a;
	*col = tr_c.p + a;
	*val = tr_v.p + a;
	return b - a;
}

const uint32_t *
hs_simplex_col(hspace s, int k, int j)
{
	chk(s);
	if (S.kind[s] != HS_SIMPLICIAL || k < 0 || k > S.top[s] || j < 0 ||
	    j > k)
		return NULL;
	return sxv.p + sxo.p[S.sx_o[s] + k] + (uint32_t)j * hs_ncells(s, k);
}

uint32_t
hs_nvert(hspace s)
{
	return hs_ncells(s, 0);
}

long long
hs_euler(hspace s)
{
	long long e = 0;
	int k;

	for (k = 0; k <= hs_top(s); k++)
		e += (k & 1 ? -1 : 1) * (long long)hs_ncells(s, k);
	return e;
}

/* ---- the builder ---- */

struct trip {
	uint32_t *r, *c;
	int64_t *v;
	uint32_t n, cap;
};

static struct {
	int top, simp;
	uint32_t cnt[HMAXDIM + 1];
	struct trip bd[HMAXDIM + 2];
	uint32_t *sx[HMAXDIM + 1];	/* row-major, k+1 per simplex */
	uint32_t sxcap[HMAXDIM + 1];
} B;

static void
b_reset(int top, int simp)
{
	int k;

	if (top > HMAXDIM)
		fail("homalg: dimension %d (at most %d)", top, HMAXDIM);
	B.top = top;
	B.simp = simp;
	for (k = 0; k <= HMAXDIM; k++)
		B.cnt[k] = 0;
	for (k = 0; k < HMAXDIM + 2; k++)
		B.bd[k].n = 0;
}

static void
b_add(int k, uint32_t r, uint32_t c, int64_t v)
{
	struct trip *t = &B.bd[k];

	if (v == 0)
		return;
	if (t->n == t->cap) {
		t->cap = newcap(t->cap, t->n + 1);
		t->r = xrealloc(t->r, t->cap * sizeof *t->r);
		t->c = xrealloc(t->c, t->cap * sizeof *t->c);
		t->v = xrealloc(t->v, t->cap * sizeof *t->v);
	}
	t->r[t->n] = r;
	t->c[t->n] = c;
	t->v[t->n] = v;
	t->n++;
}

static const struct trip *sort_t;

static int
trip_cmp(const void *a, const void *b)
{
	uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b;

	if (sort_t->c[i] != sort_t->c[j])
		return sort_t->c[i] < sort_t->c[j] ? -1 : 1;
	if (sort_t->r[i] != sort_t->r[j])
		return sort_t->r[i] < sort_t->r[j] ? -1 : 1;
	return 0;
}

#define VMAX ((int64_t)1 << 40)

/* d_{k-1} d_k = 0, on the committed blocks of space s */
static void
check_dd(hspace s)
{
	static i128 *acc;
	static uint32_t acap;
	int k;

	for (k = 2; k <= S.top[s]; k++) {
		const uint32_t *r1, *c1, *r2, *c2;
		const int64_t *v1, *v2;
		uint32_t n1 = hs_bd(s, k - 1, &r1, &c1, &v1);
		uint32_t n2 = hs_bd(s, k, &r2, &c2, &v2);
		uint32_t m = hs_ncells(s, k - 2), a, b, e, *start;

		if (m > acap) {
			acap = m;
			acc = xrealloc(acc, acap * sizeof *acc);
		}
		/* column starts of d_{k-1} */
		start = xmalloc((hs_ncells(s, k - 1) + 1) * sizeof *start);
		for (a = 0, e = 0; a <= hs_ncells(s, k - 1); a++) {
			while (e < n1 && c1[e] < a)
				e++;
			start[a] = e;
		}
		for (a = 0; a < n2; a = b) {
			for (b = a; b < n2 && c2[b] == c2[a]; b++)
				;
			memset(acc, 0, m * sizeof *acc);
			for (e = a; e < b; e++) {
				uint32_t f;
				for (f = start[r2[e]]; f < start[r2[e] + 1]; f++)
					acc[r1[f]] += (i128)v2[e] * v1[f];
			}
			for (e = 0; e < m; e++)
				if (acc[e] != 0) {
					free(start);
					fail("homalg: %s: d%d d%d != 0 (cell %u "
					    "of dimension %d)", hs_name(s), k - 1,
					    k, (unsigned)c2[a], k);
				}
		}
		free(start);
	}
}

static hspace
b_commit(const char *name)
{
	static uint32_t *idx;
	static uint32_t icap;
	hspace s;
	int k;

	if (S.n == S.cap) {
		S.cap = newcap(S.cap, S.n + 1);
		S.top = xrealloc(S.top, S.cap);
		S.kind = xrealloc(S.kind, S.cap);
		S.name = xrealloc(S.name, S.cap * sizeof *S.name);
		S.cnt_o = xrealloc(S.cnt_o, S.cap * sizeof *S.cnt_o);
		S.bd_o = xrealloc(S.bd_o, S.cap * sizeof *S.bd_o);
		S.sx_o = xrealloc(S.sx_o, S.cap * sizeof *S.sx_o);
	}
	s = S.n;
	S.top[s] = (int8_t)B.top;
	S.kind[s] = B.simp ? HS_SIMPLICIAL : HS_CELLULAR;
	S.name[s] = sym_cstr(name);
	S.cnt_o[s] = cnt.n;
	for (k = 0; k <= B.top; k++)
		u32_push(&cnt, B.cnt[k]);
	S.bd_o[s] = bdo.n;
	for (k = 0; k <= B.top + 1; k++) {
		struct trip *t = &B.bd[k];
		uint32_t i, j;

		u32_push(&bdo, tr_r.n);
		if (k == 0 || k > B.top)
			continue;
		if (t->n > icap) {
			icap = t->n;
			idx = xrealloc(idx, icap * sizeof *idx);
		}
		for (i = 0; i < t->n; i++) {
			if (t->r[i] >= B.cnt[k - 1] || t->c[i] >= B.cnt[k])
				fail("homalg: %s: a boundary entry outside "
				    "the cells", name);
			idx[i] = i;
		}
		sort_t = t;
		qsort(idx, t->n, sizeof *idx, trip_cmp);
		for (i = 0; i < t->n; i = j) {
			int64_t v = 0;
			for (j = i; j < t->n && t->r[idx[j]] == t->r[idx[i]] &&
			    t->c[idx[j]] == t->c[idx[i]]; j++)
				v += t->v[idx[j]];	/* |v| < 2^40 each */
			if (v != 0) {
				u32_push(&tr_r, t->r[idx[i]]);
				u32_push(&tr_c, t->c[idx[i]]);
				i64_push(&tr_v, v);
			}
		}
	}
	S.sx_o[s] = sxo.n;
	if (B.simp)
		for (k = 0; k <= B.top; k++) {
			uint32_t i, j, n = B.cnt[k];
			u32_push(&sxo, sxv.n);
			for (j = 0; j <= (uint32_t)k; j++)
				for (i = 0; i < n; i++)
					u32_push(&sxv, B.sx[k][i * (k + 1) + j]);
		}
	S.n++;
	check_dd(s);
	return s;
}

/* ---- simplicial complexes ---- */

static uint32_t sx_stride;
static const uint32_t *sx_base;

static int
row_cmp(const void *a, const void *b)
{
	const uint32_t *x = a, *y = b;
	uint32_t j;

	for (j = 0; j < sx_stride; j++)
		if (x[j] != y[j])
			return x[j] < y[j] ? -1 : 1;
	return 0;
}

static int
u32_cmp(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;

	return x < y ? -1 : x > y;
}

static uint32_t
sx_find(int k, const uint32_t *v)
{
	uint32_t lo = 0, hi = B.cnt[k];

	sx_stride = (uint32_t)k + 1;
	while (lo < hi) {
		uint32_t mid = lo + (hi - lo) / 2;
		int c = row_cmp(B.sx[k] + mid * sx_stride, v);
		if (c == 0)
			return mid;
		if (c < 0)
			lo = mid + 1;
		else
			hi = mid;
	}
	return HNONE;
}

#define MAXFACET 14	/* vertices of a facet (2^14 faces each) */

hspace
hs_simplicial(const char *name, const uint32_t *v, const uint32_t *off,
    uint32_t nfacets)
{
	uint32_t f, i, top = 0, buf[MAXFACET], face[MAXFACET];
	int k;

	(void)sx_base;
	if (nfacets == 0)
		fail("homalg: a simplicial complex needs a facet");
	for (f = 0; f < nfacets; f++) {
		uint32_t m = off[f + 1] - off[f];
		if (m == 0 || m > MAXFACET)
			fail("homalg: a facet has 1 to %d vertices", MAXFACET);
		if (m - 1 > top)
			top = m - 1;
	}
	b_reset((int)top, 1);
	/* every face of every facet, then sort and remove duplicates */
	for (f = 0; f < nfacets; f++) {
		uint32_t m = off[f + 1] - off[f], mask;
		memcpy(buf, v + off[f], m * sizeof *buf);
		qsort(buf, m, sizeof *buf, u32_cmp);
		for (i = 1; i < m; i++)
			if (buf[i] == buf[i - 1])
				fail("homalg: facet %u repeats vertex %u",
				    (unsigned)f, (unsigned)buf[i]);
		for (mask = 1; mask < (1u << m); mask++) {
			uint32_t n = 0, kk;
			for (i = 0; i < m; i++)
				if (mask >> i & 1)
					face[n++] = buf[i];
			kk = n - 1;
			if (B.cnt[kk] + 1 > B.sxcap[kk]) {
				B.sxcap[kk] = newcap(B.sxcap[kk], B.cnt[kk] + 1);
				B.sx[kk] = xrealloc(B.sx[kk], (size_t)B.sxcap[kk] *
				    n * sizeof *B.sx[kk]);
			}
			memcpy(B.sx[kk] + (size_t)B.cnt[kk] * n, face,
			    n * sizeof *face);
			B.cnt[kk]++;
		}
	}
	for (k = 0; k <= (int)top; k++) {
		uint32_t n = 0, st = (uint32_t)k + 1;
		sx_stride = st;
		qsort(B.sx[k], B.cnt[k], st * sizeof *B.sx[k], row_cmp);
		for (i = 0; i < B.cnt[k]; i++)
			if (n == 0 || row_cmp(B.sx[k] + (n - 1) * st,
			    B.sx[k] + i * st) != 0) {
				memmove(B.sx[k] + n * st, B.sx[k] + i * st,
				    st * sizeof *B.sx[k]);
				n++;
			}
		B.cnt[k] = n;
	}
	/* boundaries */
	for (k = 1; k <= (int)top; k++) {
		uint32_t st = (uint32_t)k + 1, s, j;
		for (s = 0; s < B.cnt[k]; s++) {
			const uint32_t *row = B.sx[k] + s * st;
			for (j = 0; j < st; j++) {
				uint32_t n = 0, t, r;
				for (t = 0; t < st; t++)
					if (t != j)
						face[n++] = row[t];
				r = sx_find(k - 1, face);
				b_add(k, r, s, j & 1 ? -1 : 1);
			}
		}
	}
	return b_commit(name);
}

/* ---- cellular complexes ---- */

hspace
hs_cellular(const char *name, int top, const uint32_t *ncells,
    const int64_t *const *d)
{
	int k;

	if (top < 0)
		fail("homalg: a cell complex needs a cell");
	b_reset(top, 0);
	for (k = 0; k <= top; k++)
		B.cnt[k] = ncells[k];
	for (k = 1; k <= top; k++) {
		uint32_t r, c, nr = ncells[k - 1], nc = ncells[k];
		for (r = 0; r < nr; r++)
			for (c = 0; c < nc; c++) {
				int64_t x = d[k - 1][(size_t)r * nc + c];
				if (x >= VMAX || x <= -VMAX)
					fail("homalg: boundary coefficients are "
					    "below 2^40");
				b_add(k, r, c, x);
			}
	}
	return b_commit(name);
}

/* ---- standard models ---- */

static char nbuf[256];

hspace
hs_point(void)
{
	uint32_t v = 0, off[2] = {0, 1};

	return hs_simplicial("pt", &v, off, 1);
}

hspace
hs_sphere(int n, int simplicial)
{
	if (n < 0 || n > HMAXDIM - 1)
		fail("homalg: S^%d: 0 <= n < %d", n, HMAXDIM);
	if (simplicial) {
		/* the boundary of the (n+1)-simplex: drop one vertex each */
		uint32_t *v, *off, f, i, nv;
		hspace s;
		if (n + 2 > MAXFACET + 1)
			fail("homalg: simplicial S^%d: n <= %d", n, MAXFACET - 1);
		v = xmalloc((size_t)(n + 2) * (n + 1) * sizeof *v);
		off = xmalloc((size_t)(n + 3) * sizeof *off);
		for (f = 0, nv = 0; f < (uint32_t)n + 2; f++) {
			off[f] = nv;
			for (i = 0; i < (uint32_t)n + 2; i++)
				if (i != f)
					v[nv++] = i;
		}
		off[n + 2] = nv;
		snprintf(nbuf, sizeof nbuf, "S^%d_s", n);
		s = hs_simplicial(nbuf, v, off, (uint32_t)n + 2);
		free(v);
		free(off);
		return s;
	}
	b_reset(n, 0);
	B.cnt[0] = n == 0 ? 2 : 1;
	if (n > 0)
		B.cnt[n] = 1;
	snprintf(nbuf, sizeof nbuf, "S^%d", n);
	return b_commit(nbuf);
}

hspace
hs_torus(int n)
{
	hspace s, c;
	int i;

	if (n < 0 || n > HMAXDIM)
		fail("homalg: T^%d: 0 <= n <= %d", n, HMAXDIM);
	if (n > 12)
		fail("homalg: T^%d has 2^%d cells (n <= 12)", n, n);
	if (n == 0)
		return hs_point();
	c = hs_sphere(1, 0);
	for (s = c, i = 1; i < n; i++)
		s = hs_product(s, c);
	snprintf(nbuf, sizeof nbuf, "T^%d", n);
	S.name[s] = sym_cstr(nbuf);
	return s;
}

static hspace
grid_surface(const char *name, int nx, int ny, int flip)
{
	/* the square [0,nx] x [0,ny], (x,0) ~ (x,ny), and (nx,y) ~ (0,y) or
	 * with flip (nx,y) ~ (0,ny-y): torus or Klein bottle */
	uint32_t *v = xmalloc((size_t)nx * ny * 6 * sizeof *v);
	uint32_t *off = xmalloc(((size_t)nx * ny * 2 + 1) * sizeof *off);
	uint32_t nf = 0, nv = 0;
	int x, y, t;
	hspace s;

	for (x = 0; x < nx; x++)
		for (y = 0; y < ny; y++) {
			int px[2][3] = {{x, x + 1, x + 1}, {x, x, x + 1}};
			int py[2][3] = {{y, y, y + 1}, {y, y + 1, y + 1}};
			for (t = 0; t < 2; t++) {
				int c;
				off[nf++] = nv;
				for (c = 0; c < 3; c++) {
					int a = px[t][c], b = py[t][c];
					if (a == nx) {
						a = 0;
						b = flip ? ny - b : b;
					}
					b %= ny;
					v[nv++] = (uint32_t)(a * ny + b);
				}
			}
		}
	off[nf] = nv;
	s = hs_simplicial(name, v, off, nf);
	free(v);
	free(off);
	if (hs_ncells(s, 0) != (uint32_t)(nx * ny) ||
	    hs_ncells(s, 1) != (uint32_t)(3 * nx * ny) ||
	    hs_ncells(s, 2) != (uint32_t)(2 * nx * ny))
		fail("homalg: %s: the grid triangulation collapsed", name);
	return s;
}

hspace
hs_torus_simplicial(void)
{
	/* Moebius' 7-vertex torus: {i, i+1, i+3}, {i, i+2, i+3} mod 7 */
	uint32_t v[42], off[15];
	int i;
	hspace s;

	for (i = 0; i < 7; i++) {
		off[2 * i] = 6 * i;
		v[6 * i] = i;
		v[6 * i + 1] = (i + 1) % 7;
		v[6 * i + 2] = (i + 3) % 7;
		off[2 * i + 1] = 6 * i + 3;
		v[6 * i + 3] = i;
		v[6 * i + 4] = (i + 2) % 7;
		v[6 * i + 5] = (i + 3) % 7;
	}
	off[14] = 42;
	s = hs_simplicial("T^2_s", v, off, 14);
	if (hs_ncells(s, 0) != 7 || hs_ncells(s, 1) != 21 ||
	    hs_ncells(s, 2) != 14)
		fail("homalg: the 7-vertex torus is not (7, 21, 14)");
	return s;
}

hspace
hs_rp(int n)
{
	int k;

	if (n < 0 || n > HMAXDIM)
		fail("homalg: RP^%d: 0 <= n <= %d", n, HMAXDIM);
	/* one cell e_k per dimension; d e_k = (1 + (-1)^k) e_{k-1} */
	b_reset(n, 0);
	for (k = 0; k <= n; k++)
		B.cnt[k] = 1;
	for (k = 2; k <= n; k += 2)
		b_add(k, 0, 0, 2);
	snprintf(nbuf, sizeof nbuf, "RP^%d", n);
	return b_commit(nbuf);
}

/* index of a sign vector in {-1,0,1}^m, canonical up to sign */
static uint32_t
signvec_id(const int *s, int m, int canon)
{
	uint32_t id = 0;
	int j, flip = 0, first = 1;

	for (j = 0; j < m; j++) {
		if (canon && first && s[j] != 0) {
			flip = s[j] < 0;
			first = 0;
		}
	}
	for (j = 0; j < m; j++)
		id = id * 3 + (uint32_t)((flip ? -s[j] : s[j]) + 1);
	return id;
}

hspace
hs_rp_simplicial(int n)
{
	if (n == 2) {
		/* the hemi-icosahedron: 6 vertices, 15 edges, 10 triangles */
		static const uint32_t f[30] = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4,
			5, 0, 5, 1, 1, 2, 4, 2, 3, 5, 3, 4, 1, 4, 5, 2, 5, 1, 3};
		uint32_t off[11];
		int i;
		hspace s;
		for (i = 0; i <= 10; i++)
			off[i] = 3 * i;
		s = hs_simplicial("RP^2_s", f, off, 10);
		if (hs_ncells(s, 0) != 6 || hs_ncells(s, 1) != 15 ||
		    hs_ncells(s, 2) != 10)
			fail("homalg: the 6-vertex RP^2 is not (6, 15, 10)");
		return s;
	}
	if (n < 1 || n > 4)
		fail("homalg: simplicial RP^%d: 1 <= n <= 4", n);
	{
		/* sd of the boundary of the (n+1)-cross-polytope: a facet is a
		 * flag of sign vectors, by a permutation and signs; its
		 * quotient by -1 identifies v with -v */
		int m = n + 1, perm[6], sg[6], vec[6], j, k;
		uint32_t nfac = 1, f, *vu, *vd, *off;
		hspace up, dn;

		for (j = 2; j <= m; j++)
			nfac *= (uint32_t)j;
		nfac <<= m;
		vu = xmalloc((size_t)nfac * m * sizeof *vu);
		vd = xmalloc((size_t)nfac * m * sizeof *vd);
		off = xmalloc(((size_t)nfac + 1) * sizeof *off);
		for (f = 0; f < nfac; f++) {
			uint32_t q = f >> m;
			int used[6] = {0};
			/* the permutation: q in the factorial number system */
			for (j = 0; j < m; j++) {
				int r = (int)(q % (uint32_t)(m - j)), c;
				q /= (uint32_t)(m - j);
				for (c = 0; c < m; c++)
					if (!used[c] && r-- == 0)
						break;
				used[c] = 1;
				perm[j] = c;
			}
			for (j = 0; j < m; j++)
				sg[j] = (f >> j & 1) ? -1 : 1;
			for (j = 0; j < m; j++)
				vec[j] = 0;
			off[f] = f * (uint32_t)m;
			for (k = 0; k < m; k++) {
				vec[perm[k]] = sg[perm[k]];
				vu[f * m + k] = signvec_id(vec, m, 0);
				vd[f * m + k] = signvec_id(vec, m, 1);
			}
		}
		off[nfac] = nfac * (uint32_t)m;
		snprintf(nbuf, sizeof nbuf, "sd S^%d", n);
		up = hs_simplicial(nbuf, vu, off, nfac);
		snprintf(nbuf, sizeof nbuf, "RP^%d_s", n);
		dn = hs_simplicial(nbuf, vd, off, nfac);
		free(vu);
		free(vd);
		free(off);
		/* a 2-to-1 covering: no simplex meets v and -v (a flag never
		 * does), and every count halves */
		for (k = 0; k <= n; k++)
			if (hs_ncells(up, k) != 2 * hs_ncells(dn, k))
				fail("homalg: RP^%d_s: S^%d -> RP^%d is not 2 to "
				    "1 on %d-simplices", n, n, n, k);
		return dn;
	}
}

hspace
hs_klein(int simplicial)
{
	if (simplicial)
		return grid_surface("K_s", 3, 3, 1);
	/* a, b; the 2-cell along a b a^-1 b: d e2 = 2b */
	b_reset(2, 0);
	B.cnt[0] = 1;
	B.cnt[1] = 2;
	B.cnt[2] = 1;
	b_add(2, 1, 0, 2);
	return b_commit("K");
}

static uint64_t
gcd64(uint64_t a, uint64_t b)
{
	while (b) {
		uint64_t t = a % b;
		a = b;
		b = t;
	}
	return a;
}

hspace
hs_lens(uint64_t m, uint64_t q)
{
	/* e0 .. e3; d e2 = m e1 (the 2-cell wraps m times), d e1 = d e3 = 0
	 * (for every q prime to m) */
	if (m < 1 || m >= (uint64_t)VMAX)
		fail("homalg: L(m;q): 1 <= m < 2^40");
	if (q == 0 || gcd64(m, q) != 1)
		fail("homalg: L(%llu;%llu): q is prime to m",
		    (unsigned long long)m, (unsigned long long)q);
	b_reset(3, 0);
	B.cnt[0] = B.cnt[1] = B.cnt[2] = B.cnt[3] = 1;
	b_add(2, 0, 0, (int64_t)m);
	snprintf(nbuf, sizeof nbuf, "L(%llu;%llu)", (unsigned long long)m,
	    (unsigned long long)(q % m == 0 ? q : q % m));
	return b_commit(nbuf);
}

hspace
hs_cp(int n)
{
	int k;

	if (n < 0 || 2 * n > HMAXDIM)
		fail("homalg: CP^%d: 0 <= n <= %d", n, HMAXDIM / 2);
	b_reset(2 * n, 0);
	for (k = 0; k <= n; k++)
		B.cnt[2 * k] = 1;
	snprintf(nbuf, sizeof nbuf, "CP^%d", n);
	return b_commit(nbuf);
}

/* ---- constructions ---- */

static uint32_t
bd_col_start(hspace s, int k, uint32_t col, const uint32_t *c, uint32_t n)
{
	uint32_t lo = 0, hi = n;

	(void)s;
	(void)k;
	while (lo < hi) {
		uint32_t mid = lo + (hi - lo) / 2;
		if (c[mid] < col)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

hspace
hs_product(hspace a, hspace b)
{
	int ta = hs_top(a), tb = hs_top(b), top = ta + tb, n, p;
	/* off[n][p]: the first cell of dimension n with an a-part of dim p */
	static uint32_t off[2 * HMAXDIM + 2][HMAXDIM + 2];
	char name[256];

	if (top > HMAXDIM)
		fail("homalg: %s x %s: dimension %d (at most %d)", hs_name(a),
		    hs_name(b), top, HMAXDIM);
	snprintf(name, sizeof name, "%s x %s", hs_name(a), hs_name(b));
	b_reset(top, 0);
	for (n = 0; n <= top; n++) {
		uint64_t t = 0;
		for (p = 0; p <= n; p++) {
			off[n][p] = (uint32_t)t;
			t += (uint64_t)hs_ncells(a, p) * hs_ncells(b, n - p);
			if (t > 50000000)
				fail("homalg: %s has too many cells", name);
		}
		off[n][n + 1] = (uint32_t)t;
		B.cnt[n] = (uint32_t)t;
	}
	for (n = 1; n <= top; n++)
		for (p = 0; p <= n; p++) {
			int q = n - p;
			uint32_t na = hs_ncells(a, p), nb = hs_ncells(b, q), x, y;
			const uint32_t *ra, *ca, *rb, *cb;
			const int64_t *va, *vb;
			uint32_t ka = hs_bd(a, p, &ra, &ca, &va);
			uint32_t kb = hs_bd(b, q, &rb, &cb, &vb);
			for (x = 0; x < na; x++)
				for (y = 0; y < nb; y++) {
					uint32_t col = off[n][p] + x * nb + y, e;
					/* d(x * y) = dx * y + (-1)^p x * dy */
					if (p >= 1)
						for (e = bd_col_start(a, p, x, ca,
						    ka); e < ka && ca[e] == x; e++)
							b_add(n, off[n - 1][p - 1] +
							    ra[e] * nb + y, col,
							    va[e]);
					if (q >= 1)
						for (e = bd_col_start(b, q, y, cb,
						    kb); e < kb && cb[e] == y; e++)
							b_add(n, off[n - 1][p] +
							    x * hs_ncells(b, q - 1) +
							    rb[e], col, p & 1 ?
							    -vb[e] : vb[e]);
				}
		}
	return b_commit(name);
}

hspace
hs_wedge(hspace a, hspace b)
{
	int ta = hs_top(a), tb = hs_top(b), top = ta > tb ? ta : tb, k;
	char name[256];

	if (hs_ncells(a, 0) == 0 || hs_ncells(b, 0) == 0)
		fail("homalg: a wedge needs a 0-cell on each side");
	snprintf(name, sizeof name, "%s v %s", hs_name(a), hs_name(b));
	b_reset(top, 0);
	for (k = 0; k <= top; k++)
		B.cnt[k] = hs_ncells(a, k) + hs_ncells(b, k) - (k == 0);
	for (k = 1; k <= top; k++) {
		const uint32_t *r, *c;
		const int64_t *v;
		uint32_t n = hs_bd(a, k, &r, &c, &v), e;
		uint32_t na = hs_ncells(a, k), nar = hs_ncells(a, k - 1);
		for (e = 0; e < n; e++)
			b_add(k, r[e], c[e], v[e]);
		n = hs_bd(b, k, &r, &c, &v);
		for (e = 0; e < n; e++) {
			/* b's 0-cell 0 is a's 0-cell 0 */
			uint32_t row = k == 1 ? (r[e] == 0 ? 0 : nar + r[e] - 1)
			    : nar + r[e];
			b_add(k, row, na + c[e], v[e]);
		}
	}
	return b_commit(name);
}

hspace
hs_suspension(hspace a)
{
	int ta = hs_top(a), k;
	char name[256];

	if (ta + 1 > HMAXDIM)
		fail("homalg: the suspension of %s is too big", hs_name(a));
	snprintf(name, sizeof name, "susp(%s)", hs_name(a));
	b_reset(ta + 1, 0);
	B.cnt[0] = 2;		/* N = 0, S = 1 */
	for (k = 0; k <= ta; k++)
		B.cnt[k + 1] = hs_ncells(a, k);
	{
		uint32_t i;
		for (i = 0; i < hs_ncells(a, 0); i++) {
			b_add(1, 1, i, 1);	/* d(v x I) = S - N */
			b_add(1, 0, i, -1);
		}
	}
	for (k = 1; k <= ta; k++) {
		const uint32_t *r, *c;
		const int64_t *v;
		uint32_t n = hs_bd(a, k, &r, &c, &v), e;
		for (e = 0; e < n; e++)
			b_add(k + 1, r[e], c[e], v[e]);
	}
	return b_commit(name);
}

/* ---- names ---- */

/*
 * space := wedge ;  wedge := prod (("v" | "\/" | "wedge") prod)* ;
 * prod := unary (("x" | "*" | "×") unary)* ;
 * unary := "susp" "(" space ")" | "(" space ")" | atom ;
 * atom := S^n | T^n | RP^n | CP^n | K | Klein | L(m;q) | L(m) | pt,
 * with "_s" for the simplicial model (S^n_s, T^2_s, RP^n_s, K_s).
 */
static const char *ps, *perr;

static void
ws(void)
{
	while (*ps == ' ' || *ps == '\t')
		ps++;
}

static int
lit(const char *w)
{
	size_t n = strlen(w);

	ws();
	if (strncmp(ps, w, n) != 0)
		return 0;
	if (isalpha((unsigned char)w[n - 1]) && (isalnum((unsigned char)ps[n])
	    || ps[n] == '_'))
		return 0;
	ps += n;
	return 1;
}

static int
num(uint64_t *v)
{
	ws();
	if (!isdigit((unsigned char)*ps))
		return 0;
	*v = 0;
	while (isdigit((unsigned char)*ps)) {
		if (*v > 100000000000ULL)
			return 0;
		*v = *v * 10 + (uint64_t)(*ps++ - '0');
	}
	return 1;
}

static hspace p_space(void);

static int
suffix_s(void)
{
	if (ps[0] == '_' && ps[1] == 's' && !isalnum((unsigned char)ps[2])) {
		ps += 2;
		return 1;
	}
	return 0;
}

static int
power(uint64_t *n)
{
	ws();
	if (*ps != '^')
		return 0;
	ps++;
	return num(n);
}

static hspace
p_atom(void)
{
	uint64_t n, q;

	ws();
	if (lit("(")) {
		hspace s = p_space();
		if (s == HNONE)
			return HNONE;
		if (!lit(")"))
			return perr = "missing )", HNONE;
		return s;
	}
	if (lit("susp") || lit("Σ")) {
		hspace s;
		if (lit("(")) {
			s = p_space();
			if (s == HNONE)
				return HNONE;
			if (!lit(")"))
				return perr = "missing )", HNONE;
		} else if ((s = p_atom()) == HNONE)
			return HNONE;
		return hs_suspension(s);
	}
	if (lit("pt") || lit("point"))
		return hs_point();
	if (lit("Klein") || lit("klein"))
		return hs_klein(suffix_s());
	if (strncmp(ps, "RP", 2) == 0) {
		ps += 2;
		if (!power(&n))
			return perr = "RP^n", HNONE;
		return suffix_s() ? hs_rp_simplicial((int)n) : hs_rp((int)n);
	}
	if (strncmp(ps, "CP", 2) == 0) {
		ps += 2;
		if (!power(&n))
			return perr = "CP^n", HNONE;
		return hs_cp((int)n);
	}
	if (*ps == 'S' && (ps[1] == '^' || ps[1] == ' ')) {
		ps++;
		if (!power(&n))
			return perr = "S^n", HNONE;
		return hs_sphere((int)n, suffix_s());
	}
	if (*ps == 'T' && (ps[1] == '^' || ps[1] == ' ')) {
		ps++;
		if (!power(&n))
			return perr = "T^n", HNONE;
		if (suffix_s()) {
			if (n != 2)
				return perr = "T^2_s only", HNONE;
			return hs_torus_simplicial();
		}
		return hs_torus((int)n);
	}
	if (*ps == 'K' && !isalnum((unsigned char)ps[1])) {
		ps++;
		return hs_klein(suffix_s());
	}
	if (*ps == 'L' && (ps[1] == '(' || ps[1] == ' ')) {
		ps++;
		if (!lit("(") || !num(&n))
			return perr = "L(m;q)", HNONE;
		q = 1;
		if (lit(";") || lit(","))
			if (!num(&q))
				return perr = "L(m;q)", HNONE;
		if (!lit(")"))
			return perr = "L(m;q): missing )", HNONE;
		return hs_lens(n, q);
	}
	return perr = "a space: S^n, T^n, RP^n, CP^n, K, L(m;q), pt, "
	    "susp(X), X x Y, X v Y", HNONE;
}

static hspace
p_prod(void)
{
	hspace s = p_atom(), t;

	while (s != HNONE && (lit("x") || lit("*") || lit("×"))) {
		if ((t = p_atom()) == HNONE)
			return HNONE;
		s = hs_product(s, t);
	}
	return s;
}

static hspace
p_space(void)
{
	hspace s = p_prod(), t;

	while (s != HNONE && (lit("v") || lit("\\/") || lit("∨") ||
	    lit("wedge"))) {
		if ((t = p_prod()) == HNONE)
			return HNONE;
		s = hs_wedge(s, t);
	}
	return s;
}

hspace
hs_parse(const char *text, const char **err)
{
	hspace s;

	ps = text;
	perr = NULL;
	s = p_space();
	ws();
	if (s != HNONE && *ps) {
		perr = "unexpected text after the space";
		s = HNONE;
	}
	if (err)
		*err = perr;
	return s;
}
