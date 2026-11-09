/*
 * zmat.c - integer matrices: products, Bareiss determinant and rank,
 * Hermite and Smith normal forms with unimodular transforms, kernels,
 * images, solving over Z, lattices; matrices over Q (qmat) and the
 * bridge to ring.h's matrices over Z/p.
 *
 * Every entry is a zint handle in one dense row-major array.  The normal
 * forms work in place on copies with 2x2 unimodular row and column
 * operations (Euclid steps from z_xgcd) applied to the matrix and to its
 * transforms together; after each pivot the arena is compacted down to
 * the live entries (keep()), so a long elimination does not accumulate
 * garbage.  Results are verified by multiplication before they are
 * returned (internal failure otherwise).
 */
#include "absalg.h"
#include "ucas.h"

#include <stdlib.h>
#include <string.h>

/* ---- basics ---- */

void
zm_init(struct zmat *m, int r, int c)
{
	size_t n;

	if (r < 0 || c < 0)
		fail("internal: negative matrix size");
	m->r = r;
	m->c = c;
	n = (size_t)r * (size_t)c;
	m->a = xcalloc(n ? n : 1, sizeof *m->a);
}

void
zm_free(struct zmat *m)
{
	free(m->a);
	m->a = NULL;
	m->r = m->c = 0;
}

void
zm_identity(struct zmat *m, int n)
{
	int i;

	zm_init(m, n, n);
	for (i = 0; i < n; i++)
		ZM(m, i, i) = Z_ONE;
}

void
zm_copy(struct zmat *dst, const struct zmat *src)
{
	struct zmat t;

	zm_init(&t, src->r, src->c);
	memcpy(t.a, src->a, (size_t)src->r * (size_t)src->c * sizeof *t.a);
	*dst = t;
}

void
zm_move(struct zmat *dst, struct zmat *src)
{
	if (dst == src)
		return;
	free(dst->a);
	*dst = *src;
	src->a = NULL;
	src->r = src->c = 0;
}

void
zm_from_i64(struct zmat *m, int r, int c, const int64_t *v)
{
	size_t i;

	zm_init(m, r, c);
	for (i = 0; i < (size_t)r * (size_t)c; i++)
		m->a[i] = z_from_i64(v[i]);
}

static const char *
skipws(const char *s)
{
	while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')
		s++;
	return s;
}

/* a row: [a, b, c] (also with spaces only between entries) */
static int
parse_row(const char *s, zint **v, int *n, int *cap, const char **end)
{
	s = skipws(s);
	if (*s != '[')
		return 0;
	s = skipws(s + 1);
	*n = 0;
	if (*s == ']') {
		*end = s + 1;
		return 1;
	}
	for (;;) {
		zint x;
		const char *e;
		if (!z_parse(s, &x, &e))
			return 0;
		if (*n == *cap) {
			*cap = *cap ? *cap * 2 : 16;
			*v = xrealloc(*v, (size_t)*cap * sizeof **v);
		}
		(*v)[(*n)++] = x;
		s = skipws(e);
		if (*s == ',')
			s = skipws(s + 1);
		else if (*s == ']') {
			*end = s + 1;
			return 1;
		} else if (!((*s >= '0' && *s <= '9') || *s == '-' || *s == '+'))
			return 0;
	}
}

int
zm_parse(struct zmat *m, const char *s, const char **end)
{
	zint *all = NULL, *row = NULL;
	int nall = 0, capall = 0, nrow = 0, caprow = 0, r = 0, c = -1;
	const char *e, *s0 = s;

	s = skipws(s);
	if (*s != '[')
		goto bad;
	e = skipws(s + 1);
	if (*e != '[') {
		/* one row */
		if (!parse_row(s, &row, &nrow, &caprow, &e))
			goto bad;
		zm_init(m, nrow ? 1 : 0, nrow);
		memcpy(m->a, row, (size_t)nrow * sizeof *row);
		free(row);
		if (end)
			*end = e;
		return 1;
	}
	s = e;
	for (;;) {
		int k;
		if (!parse_row(s, &row, &nrow, &caprow, &e))
			goto bad;
		if (c < 0)
			c = nrow;
		else if (nrow != c)
			goto bad;
		if (nall + nrow > capall) {
			capall = (nall + nrow) * 2 + 16;
			all = xrealloc(all, (size_t)capall * sizeof *all);
		}
		for (k = 0; k < nrow; k++)
			all[nall++] = row[k];
		r++;
		s = skipws(e);
		if (*s == ',')
			s = skipws(s + 1);
		else if (*s == ']') {
			s++;
			break;
		} else if (*s != '[')
			goto bad;
	}
	zm_init(m, r, c);
	if (nall)
		memcpy(m->a, all, (size_t)nall * sizeof *all);
	free(all);
	free(row);
	if (end)
		*end = s;
	return 1;
bad:
	free(all);
	free(row);
	if (end)
		*end = s > s0 ? s : s0;
	return 0;
}

void
zm_print(struct sbuf *b, const struct zmat *m)
{
	int i, j;

	if (m->r == 0) {
		sb_puts(b, "[]");
		return;
	}
	sb_putc(b, '[');
	for (i = 0; i < m->r; i++) {
		sb_puts(b, i ? ", [" : "[");
		for (j = 0; j < m->c; j++) {
			if (j)
				sb_puts(b, ", ");
			z_print(b, ZM(m, i, j));
		}
		sb_putc(b, ']');
	}
	sb_putc(b, ']');
}

int
zm_eq(const struct zmat *a, const struct zmat *b)
{
	size_t i, n = (size_t)a->r * (size_t)a->c;

	if (a->r != b->r || a->c != b->c)
		return 0;
	for (i = 0; i < n; i++)
		if (!z_eq(a->a[i], b->a[i]))
			return 0;
	return 1;
}

int
zm_is_zero(const struct zmat *m)
{
	size_t i, n = (size_t)m->r * (size_t)m->c;

	for (i = 0; i < n; i++)
		if (m->a[i] != 0)
			return 0;
	return 1;
}

int
zm_is_identity(const struct zmat *m)
{
	int i, j;

	if (m->r != m->c)
		return 0;
	for (i = 0; i < m->r; i++)
		for (j = 0; j < m->c; j++)
			if (ZM(m, i, j) != (i == j ? Z_ONE : 0))
				return 0;
	return 1;
}

/* ---- arithmetic ---- */

/* bits of the largest entry when every entry is inline, else 1000 */
static int
small_bits(const struct zmat *m)
{
	size_t i, n = (size_t)m->r * (size_t)m->c;
	uint64_t mx = 0;

	for (i = 0; i < n; i++) {
		int64_t v;
		if (!Z_IS_SMALL(m->a[i]))
			return 1000;
		v = m->a[i] >> 1;
		if ((uint64_t)(v < 0 ? -v : v) > mx)
			mx = (uint64_t)(v < 0 ? -v : v);
	}
	return bits_for(mx);
}

void
zm_mul(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	struct zmat t;
	int i, j, k;

	if (a->c != b->r)
		fail("internal: matrix product %dx%d * %dx%d", a->r, a->c, b->r,
		    b->c);
	zm_init(&t, a->r, b->c);
	if (small_bits(a) + small_bits(b) + bits_for((uint64_t)a->c) < 126) {
		/* every entry inline and no sum can overflow 128 bits */
		i128 *acc = xcalloc((size_t)b->c + 1, sizeof *acc);
		for (i = 0; i < a->r; i++) {
			memset(acc, 0, (size_t)b->c * sizeof *acc);
			for (k = 0; k < a->c; k++) {
				int64_t x = ZM(a, i, k) >> 1;
				const zint *br = b->a + (size_t)k * b->c;
				if (x == 0)
					continue;
				for (j = 0; j < b->c; j++)
					if (br[j] != 0)
						acc[j] += (i128)x * (br[j] >> 1);
			}
			for (j = 0; j < b->c; j++)
				ZM(&t, i, j) = z_from_i128(acc[j]);
		}
		free(acc);
		zm_move(r, &t);
		return;
	}
	for (i = 0; i < a->r; i++)
		for (k = 0; k < a->c; k++) {
			zint x = ZM(a, i, k);
			if (x == 0)
				continue;
			for (j = 0; j < b->c; j++)
				if (ZM(b, k, j) != 0)
					ZM(&t, i, j) = z_addmul(ZM(&t, i, j), x,
					    ZM(b, k, j));
		}
	zm_move(r, &t);
}

static void
ew(struct zmat *r, const struct zmat *a, const struct zmat *b, int sub)
{
	struct zmat t;
	size_t i, n;

	if (a->r != b->r || a->c != b->c)
		fail("internal: matrix sizes differ");
	zm_init(&t, a->r, a->c);
	n = (size_t)a->r * (size_t)a->c;
	for (i = 0; i < n; i++)
		t.a[i] = sub ? z_sub(a->a[i], b->a[i]) : z_add(a->a[i], b->a[i]);
	zm_move(r, &t);
}

void
zm_add(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	ew(r, a, b, 0);
}

void
zm_sub(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	ew(r, a, b, 1);
}

void
zm_scale(struct zmat *r, const struct zmat *a, zint k)
{
	struct zmat t;
	size_t i, n = (size_t)a->r * (size_t)a->c;

	zm_init(&t, a->r, a->c);
	for (i = 0; i < n; i++)
		t.a[i] = z_mul(a->a[i], k);
	zm_move(r, &t);
}

void
zm_transpose(struct zmat *r, const struct zmat *a)
{
	struct zmat t;
	int i, j;

	zm_init(&t, a->c, a->r);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < a->c; j++)
			ZM(&t, j, i) = ZM(a, i, j);
	zm_move(r, &t);
}

void
zm_hcat(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	struct zmat t;
	int i, j;

	if (a->r != b->r)
		fail("internal: hcat of %d and %d rows", a->r, b->r);
	zm_init(&t, a->r, a->c + b->c);
	for (i = 0; i < a->r; i++) {
		for (j = 0; j < a->c; j++)
			ZM(&t, i, j) = ZM(a, i, j);
		for (j = 0; j < b->c; j++)
			ZM(&t, i, a->c + j) = ZM(b, i, j);
	}
	zm_move(r, &t);
}

void
zm_vcat(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	struct zmat t;
	int i, j;

	if (a->c != b->c)
		fail("internal: vcat of %d and %d columns", a->c, b->c);
	zm_init(&t, a->r + b->r, a->c);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < a->c; j++)
			ZM(&t, i, j) = ZM(a, i, j);
	for (i = 0; i < b->r; i++)
		for (j = 0; j < a->c; j++)
			ZM(&t, a->r + i, j) = ZM(b, i, j);
	zm_move(r, &t);
}

void
zm_block_diag(struct zmat *r, const struct zmat *a, const struct zmat *b)
{
	struct zmat t;
	int i, j;

	zm_init(&t, a->r + b->r, a->c + b->c);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < a->c; j++)
			ZM(&t, i, j) = ZM(a, i, j);
	for (i = 0; i < b->r; i++)
		for (j = 0; j < b->c; j++)
			ZM(&t, a->r + i, a->c + j) = ZM(b, i, j);
	zm_move(r, &t);
}

void
zm_sub_cols(struct zmat *r, const struct zmat *a, int j0, int j1)
{
	struct zmat t;
	int i, j;

	if (j0 < 0 || j1 > a->c || j0 > j1)
		fail("internal: columns %d..%d of %d", j0, j1, a->c);
	zm_init(&t, a->r, j1 - j0);
	for (i = 0; i < a->r; i++)
		for (j = j0; j < j1; j++)
			ZM(&t, i, j - j0) = ZM(a, i, j);
	zm_move(r, &t);
}

void
zm_sub_rows(struct zmat *r, const struct zmat *a, int i0, int i1)
{
	struct zmat t;
	int i, j;

	if (i0 < 0 || i1 > a->r || i0 > i1)
		fail("internal: rows %d..%d of %d", i0, i1, a->r);
	zm_init(&t, i1 - i0, a->c);
	for (i = i0; i < i1; i++)
		for (j = 0; j < a->c; j++)
			ZM(&t, i - i0, j) = ZM(a, i, j);
	zm_move(r, &t);
}

void
zm_mul_vec(zint *y, const struct zmat *a, const zint *x)
{
	int i, j;

	for (i = 0; i < a->r; i++) {
		zint s = 0;
		for (j = 0; j < a->c; j++)
			if (x[j] != 0 && ZM(a, i, j) != 0)
				s = z_addmul(s, ZM(a, i, j), x[j]);
		y[i] = s;
	}
}

void
zm_col(zint *x, const struct zmat *a, int j)
{
	int i;

	for (i = 0; i < a->r; i++)
		x[i] = ZM(a, i, j);
}

/* ---- keeping the live entries when the arena is compacted ---- */

/*
 * Compaction copies every live handle, so it is done lazily: only when
 * the arena has grown past twice what was live after the last compaction
 * under the same mark (plus 1M limbs), or when forced at the end of an
 * elimination.  Small entries never touch the arena, so a matrix of small
 * numbers is never copied at all.
 */
static struct {
	uint64_t mn, ml;	/* the mark it refers to */
	uint64_t live;		/* limbs above the mark after the last one */
} kst;

static void
keep_force(zmark m, struct zmat **ms, int k, int force)
{
	static zint *buf;
	static size_t cap;
	size_t n = 0, o;
	int i;
	zmark now = z_mark();

	if (kst.mn != m.n || kst.ml != m.limbs) {
		kst.mn = m.n;
		kst.ml = m.limbs;
		kst.live = 0;
	}
	if (now.n == m.n)
		return;		/* nothing was allocated */
	if (!force && now.limbs - m.limbs < 2 * kst.live + ((uint64_t)1 << 20))
		return;
	for (i = 0; i < k; i++)
		if (ms[i])
			n += (size_t)ms[i]->r * (size_t)ms[i]->c;
	if (n > cap) {
		cap = n * 2;
		buf = xrealloc(buf, cap * sizeof *buf);
	}
	for (o = 0, i = 0; i < k; i++)
		if (ms[i]) {
			size_t c = (size_t)ms[i]->r * (size_t)ms[i]->c;
			memcpy(buf + o, ms[i]->a, c * sizeof *buf);
			o += c;
		}
	z_release_keep(m, buf, n);
	for (o = 0, i = 0; i < k; i++)
		if (ms[i]) {
			size_t c = (size_t)ms[i]->r * (size_t)ms[i]->c;
			memcpy(ms[i]->a, buf + o, c * sizeof *buf);
			o += c;
		}
	kst.live = z_mark().limbs - m.limbs;
}

static void
keep(zmark m, struct zmat **ms, int k)
{
	keep_force(m, ms, k, 0);
}

/* ---- Bareiss: determinant and rank ---- */

static int
bareiss(const struct zmat *a, zint *det)
{
	struct zmat m;
	zmark mk = z_mark();
	zint prev = Z_ONE;
	int r = 0, c, i, j, sign = 1, rows = a->r;

	zm_copy(&m, a);
	for (c = 0; c < m.c && r < rows; c++) {
		int p = -1;
		for (i = r; i < rows; i++)
			if (ZM(&m, i, c) != 0) {
				p = i;
				break;
			}
		if (p < 0)
			continue;
		if (p != r) {
			for (j = 0; j < m.c; j++) {
				zint t = ZM(&m, p, j);
				ZM(&m, p, j) = ZM(&m, r, j);
				ZM(&m, r, j) = t;
			}
			sign = -sign;
		}
		for (i = r + 1; i < rows; i++) {
			for (j = c + 1; j < m.c; j++)
				ZM(&m, i, j) = z_divexact(z_sub(z_mul(ZM(&m, r, c),
				    ZM(&m, i, j)), z_mul(ZM(&m, i, c), ZM(&m, r, j))),
				    prev);
			ZM(&m, i, c) = 0;
		}
		prev = ZM(&m, r, c);
		r++;
		{
			struct zmat *ms[1];
			ms[0] = &m;
			keep(mk, ms, 1);
			prev = ZM(&m, r - 1, c);
		}
	}
	if (det) {
		/* square: the determinant is the last pivot (or 0) */
		zint d = (r == a->r && a->r == a->c) ? (a->r ? ZM(&m, a->r - 1,
		    a->c - 1) : Z_ONE) : 0;
		if (sign < 0)
			d = z_neg(d);
		zm_free(&m);
		z_release_keep(mk, &d, 1);
		*det = d;
	} else {
		zm_free(&m);
		z_release(mk);
	}
	return r;
}

int
zm_rank(const struct zmat *a)
{
	return bareiss(a, NULL);
}

zint
zm_det(const struct zmat *a)
{
	zint d;

	if (a->r != a->c)
		fail("det: the matrix is %dx%d, not square", a->r, a->c);
	bareiss(a, &d);
	return d;
}

/* ---- elementary operations, on a matrix and its transforms ---- */



static void
swap_rows(struct zmat *m, int a, int b)
{
	int j;

	if (!m || !m->a || a == b)
		return;
	for (j = 0; j < m->c; j++) {
		zint t = ZM(m, a, j);
		ZM(m, a, j) = ZM(m, b, j);
		ZM(m, b, j) = t;
	}
}

static void
swap_cols(struct zmat *m, int a, int b)
{
	int j;

	if (!m || !m->a || a == b)
		return;
	for (j = 0; j < m->r; j++) {
		zint t = ZM(m, j, a);
		ZM(m, j, a) = ZM(m, j, b);
		ZM(m, j, b) = t;
	}
}

static void
neg_row(struct zmat *m, int i)
{
	int j;

	if (!m || !m->a)
		return;
	for (j = 0; j < m->c; j++)
		ZM(m, i, j) = z_neg(ZM(m, i, j));
}

static void
neg_col(struct zmat *m, int i)
{
	int j;

	if (!m || !m->a)
		return;
	for (j = 0; j < m->r; j++)
		ZM(m, j, i) = z_neg(ZM(m, j, i));
}

/* row k -= q * row i */
static void
row_addmul(struct zmat *m, int k, int i, zint q)
{
	int j;

	if (!m || !m->a || q == 0)
		return;
	for (j = 0; j < m->c; j++)
		if (ZM(m, i, j) != 0)
			ZM(m, k, j) = z_submul(ZM(m, k, j), q, ZM(m, i, j));
}

/* col k -= q * col i */
static void
col_addmul(struct zmat *m, int k, int i, zint q)
{
	int j;

	if (!m || !m->a || q == 0)
		return;
	for (j = 0; j < m->r; j++)
		if (ZM(m, j, i) != 0)
			ZM(m, j, k) = z_submul(ZM(m, j, k), q, ZM(m, j, i));
}

/*
 * A row operation on a with its transform u (u <- E u) and the inverse ui
 * (ui <- ui E^-1).  E = [p q; r s] with det 1.
 */
struct rt {
	struct zmat *a, *u, *ui;     /* row side */
	struct zmat *v, *vi;         /* column side */
};


static void
rop_swap(struct rt *t, int k, int i)
{
	swap_rows(t->a, k, i);
	swap_rows(t->u, k, i);
	swap_cols(t->ui, k, i);
}

static void
rop_neg(struct rt *t, int k)
{
	neg_row(t->a, k);
	neg_row(t->u, k);
	neg_col(t->ui, k);
}

/* row k -= q row i: E = I - q e_k e_i^T, E^-1 = I + q e_k e_i^T, so
 * ui <- ui E^-1 adds q * column k to column i */
static void
rop_addmul(struct rt *t, int k, int i, zint q)
{
	row_addmul(t->a, k, i, q);
	row_addmul(t->u, k, i, q);
	col_addmul(t->ui, i, k, z_neg(q));
}

/* column operation: a <- a F, v <- v F, vi <- F^-1 vi */

static void
cop_swap(struct rt *t, int k, int i)
{
	swap_cols(t->a, k, i);
	swap_cols(t->v, k, i);
	swap_rows(t->vi, k, i);
}

/* col k -= q col i: F = I - q e_i e_k^T, F^-1 = I + q e_i e_k^T: row i of
 * vi += q row k */
static void
cop_addmul(struct rt *t, int k, int i, zint q)
{
	col_addmul(t->a, k, i, q);
	col_addmul(t->v, k, i, q);
	row_addmul(t->vi, i, k, z_neg(q));
}

/*
 * Clear column c below row k (rows k+1 ..), leaving gcd at (k, c): Euclid
 * across the rows, the smallest entry as the pivot and the others reduced
 * by rounded quotients, which keeps the numbers far smaller than chained
 * 2x2 gcd steps.
 */
static void
clear_col(struct rt *t, int k, int c)
{
	struct zmat *a = t->a;
	int i;

	for (;;) {
		int p = -1, more = 0;
		for (i = k; i < a->r; i++)
			if (ZM(a, i, c) != 0 && (p < 0 || z_cmpabs(ZM(a, i, c),
			    ZM(a, p, c)) < 0))
				p = i;
		if (p < 0)
			return;
		if (p != k)
			rop_swap(t, k, p);
		for (i = k + 1; i < a->r; i++) {
			zint y = ZM(a, i, c);
			if (y == 0)
				continue;
			rop_addmul(t, i, k, z_div_round(y, ZM(a, k, c)));
			if (ZM(a, i, c) != 0)
				more = 1;
		}
		if (!more)
			return;
	}
}

/* clear row k right of column c, the same way with columns */
static void
clear_row(struct rt *t, int k, int c)
{
	struct zmat *a = t->a;
	int j;

	for (;;) {
		int p = -1, more = 0;
		for (j = c; j < a->c; j++)
			if (ZM(a, k, j) != 0 && (p < 0 || z_cmpabs(ZM(a, k, j),
			    ZM(a, k, p)) < 0))
				p = j;
		if (p < 0)
			return;
		if (p != c)
			cop_swap(t, c, p);
		for (j = c + 1; j < a->c; j++) {
			zint y = ZM(a, k, j);
			if (y == 0)
				continue;
			cop_addmul(t, j, c, z_div_round(y, ZM(a, k, c)));
			if (ZM(a, k, j) != 0)
				more = 1;
		}
		if (!more)
			return;
	}
}

static void
keep_rt_force(zmark m, struct rt *t, int force)
{
	struct zmat *ms[5];

	ms[0] = t->a;
	ms[1] = t->u;
	ms[2] = t->ui;
	ms[3] = t->v;
	ms[4] = t->vi;
	keep_force(m, ms, 5, force);
}

static void
keep_rt(zmark m, struct rt *t)
{
	keep_rt_force(m, t, 0);
}

/* ---- Hermite normal form ---- */

static int
hnf_core(struct rt *t, int *pivcol)
{
	struct zmat *a = t->a;
	zmark mk = z_mark();
	int k = 0, c, i;

	for (c = 0; c < a->c && k < a->r; c++) {
		int p = -1;
		for (i = k; i < a->r; i++)
			if (ZM(a, i, c) != 0) {
				p = i;
				break;
			}
		if (p < 0)
			continue;
		rop_swap(t, k, p);
		clear_col(t, k, c);
		if (z_sign(ZM(a, k, c)) < 0)
			rop_neg(t, k);
		for (i = 0; i < k; i++) {
			zint q = z_div_floor(ZM(a, i, c), ZM(a, k, c));
			rop_addmul(t, i, k, q);
		}
		if (pivcol)
			pivcol[k] = c;
		k++;
		keep_rt(mk, t);
	}
	keep_rt_force(mk, t, 1);
	return k;
}

int
zm_hnf(const struct zmat *a, struct zmat *h, struct zmat *u, int *pivcol)
{
	struct zmat H, U;
	struct rt t;
	int rank;

	memset(&t, 0, sizeof t);
	zm_copy(&H, a);
	t.a = &H;
	if (u) {
		zm_identity(&U, a->r);
		t.u = &U;
	}
	rank = hnf_core(&t, pivcol);
	if (u) {
		struct zmat chk;
		zm_init(&chk, 0, 0);
		zm_mul(&chk, &U, a);
		if (!zm_eq(&chk, &H))
			fail("internal: HNF transform check failed");
		zm_free(&chk);
		zm_move(u, &U);
	}
	if (h)
		zm_move(h, &H);
	else
		zm_free(&H);
	return rank;
}

int
zm_hnf_col(const struct zmat *a, struct zmat *h, struct zmat *v)
{
	struct zmat at, ht, u;
	int rank;

	memset(&ht, 0, sizeof ht);
	memset(&u, 0, sizeof u);
	zm_init(&at, 0, 0);
	zm_transpose(&at, a);
	rank = zm_hnf(&at, &ht, v ? &u : NULL, NULL);
	if (h)
		zm_transpose(h, &ht);
	if (v)
		zm_transpose(v, &u);
	zm_free(&at);
	zm_free(&ht);
	zm_free(&u);
	return rank;
}

int
zm_hnf_check(const struct zmat *a, const struct zmat *h, const struct zmat *u)
{
	struct zmat p;
	zmark mk = z_mark();
	int ok = 1, k, c = 0, i, j;
	zint d;

	zm_init(&p, 0, 0);
	if (u->r != a->r || u->c != a->r || h->r != a->r || h->c != a->c)
		return 0;
	zm_mul(&p, u, a);
	if (!zm_eq(&p, h))
		ok = 0;
	d = zm_det(u);
	if (!z_is_pm1(d))
		ok = 0;
	/* echelon shape, positive pivots, reduced above */
	for (k = 0; ok && k < h->r; k++) {
		while (c < h->c && ZM(h, k, c) == 0) {
			for (i = k + 1; i < h->r; i++)
				if (ZM(h, i, c) != 0)
					ok = 0;
			c++;
		}
		if (c == h->c) {
			for (j = 0; j < h->c; j++)
				if (ZM(h, k, j) != 0)
					ok = 0;
			continue;
		}
		if (z_sign(ZM(h, k, c)) <= 0)
			ok = 0;
		for (i = 0; i < h->r; i++) {
			if (i == k)
				continue;
			if (i > k && ZM(h, i, c) != 0)
				ok = 0;
			if (i < k && (z_sign(ZM(h, i, c)) < 0 ||
			    z_cmp(ZM(h, i, c), ZM(h, k, c)) >= 0))
				ok = 0;
		}
		c++;
	}
	zm_free(&p);
	z_release(mk);
	return ok;
}

/* ---- Smith normal form ---- */

/*
 * Unit pivots in (approximate) Markowitz order: the column with the
 * fewest nonzeros that holds a +-1, then in it the +-1 whose row is
 * sparsest.  Row and column counts are kept up to date from the support
 * of the pivot row, so a pivot costs O(rows touched * its support), not a
 * scan of the whole matrix.
 */
static void
unit_pivots(struct rt *t, zmark mk)
{
	struct zmat *a = t->a;
	int n = a->r < a->c ? a->r : a->c, k, i, j, q;
	int *rn = xcalloc((size_t)a->r + 1, sizeof *rn);
	int *cn = xcalloc((size_t)a->c + 1, sizeof *cn);
	uint8_t *nounit = xcalloc((size_t)a->c + 1, 1);
	int *sup = xcalloc((size_t)a->c + 1, sizeof *sup);
	uint8_t *was = xcalloc((size_t)a->c + 1, 1);

	for (i = 0; i < a->r; i++)
		for (j = 0; j < a->c; j++)
			if (ZM(a, i, j) != 0) {
				rn[i]++;
				cn[j]++;
			}
	for (k = 0; k < n; k++) {
		int bi = -1, bj = -1, ns = 0;
		/* the sparsest column with a unit */
		for (;;) {
			bj = -1;
			for (j = k; j < a->c; j++)
				if (cn[j] > 0 && !nounit[j] && (bj < 0 || cn[j] < cn[bj]))
					bj = j;
			if (bj < 0)
				break;
			bi = -1;
			for (i = k; i < a->r; i++)
				if (z_is_pm1(ZM(a, i, bj)) && (bi < 0 || rn[i] < rn[bi]))
					bi = i;
			if (bi >= 0)
				break;
			nounit[bj] = 1;
		}
		if (bj < 0)
			break;
		rop_swap(t, k, bi);
		q = rn[k], rn[k] = rn[bi], rn[bi] = q;
		cop_swap(t, k, bj);
		q = cn[k], cn[k] = cn[bj], cn[bj] = q;
		q = nounit[k], nounit[k] = nounit[bj], nounit[bj] = (uint8_t)q;
		if (z_sign(ZM(a, k, k)) < 0)
			rop_neg(t, k);
		/* the support of the pivot row (columns > k) */
		for (j = k + 1; j < a->c; j++)
			if (ZM(a, k, j) != 0)
				sup[ns++] = j;
		for (i = k + 1; i < a->r; i++) {
			int s2;
			if (ZM(a, i, k) == 0)
				continue;
			for (s2 = 0; s2 < ns; s2++)
				was[s2] = ZM(a, i, sup[s2]) != 0;
			rop_addmul(t, i, k, ZM(a, i, k));
			/* column k loses this entry */
			cn[k]--;
			rn[i]--;
			for (s2 = 0; s2 < ns; s2++) {
				int now = ZM(a, i, sup[s2]) != 0;
				cn[sup[s2]] += now - was[s2];
				rn[i] += now - was[s2];
				nounit[sup[s2]] = 0;
			}
		}
		/* the pivot row clears with column additions (column k is e_k) */
		for (q = 0; q < ns; q++) {
			j = sup[q];
			cop_addmul(t, j, k, ZM(a, k, j));
			cn[j]--;
		}
		keep_rt(mk, t);
	}
	free(rn);
	free(cn);
	free(nounit);
	free(sup);
	free(was);
}

int
zm_snf(const struct zmat *a, struct zsnf *s)
{
	struct zmat D, U, Ui, V, Vi;
	struct rt t;
	zmark mk = z_mark();
	int k, i, j, n = a->r < a->c ? a->r : a->c, rank = 0;

	zm_copy(&D, a);
	zm_identity(&U, a->r);
	zm_identity(&Ui, a->r);
	zm_identity(&V, a->c);
	zm_identity(&Vi, a->c);
	t.a = &D;
	t.u = &U;
	t.ui = &Ui;
	t.v = &V;
	t.vi = &Vi;
	/* unit pivots first (boundary matrices are mostly +-1), in Markowitz
	 * order to limit fill: a +-1 divides everything, so its row and
	 * column clear with additions only */
	unit_pivots(&t, mk);
	/* then Hermite: echelon form with reduced entries keeps the numbers
	 * (and the transforms) small, and leaves most pivots at 1 */
	hnf_core(&t, NULL);
	for (k = 0; k < n; k++) {
		/* the smallest nonzero entry of the lower right block (the
		 * diagonal one on ties) */
		int pi = -1, pj = -1, unit = 0;
		for (i = k; i < D.r && !unit; i++)
			for (j = k; j < D.c; j++)
				if (ZM(&D, i, j) != 0 && (pi < 0 ||
				    z_cmpabs(ZM(&D, i, j), ZM(&D, pi, pj)) < 0)) {
					pi = i;
					pj = j;
					if (z_is_pm1(ZM(&D, i, j))) {
						unit = 1;	/* nothing is smaller */
						break;
					}
				}
		if (pi < 0)
			break;
		rop_swap(&t, k, pi);
		cop_swap(&t, k, pj);
		for (;;) {
			int dirty = 0;
			clear_col(&t, k, k);
			clear_row(&t, k, k);
			for (i = k + 1; i < D.r; i++)
				if (ZM(&D, i, k) != 0)
					dirty = 1;
			if (dirty)
				continue;
			/* divisibility: d_k must divide the rest (a unit does) */
			if (z_is_pm1(ZM(&D, k, k)))
				break;
			for (i = k + 1; i < D.r && !dirty; i++)
				for (j = k + 1; j < D.c; j++)
					if (ZM(&D, i, j) != 0 &&
					    !z_divides(ZM(&D, k, k), ZM(&D, i, j))) {
						/* row k += row i */
						rop_addmul(&t, k, i, z_from_i64(-1));
						dirty = 1;
						break;
					}
			if (!dirty)
				break;
		}
		if (z_sign(ZM(&D, k, k)) < 0)
			rop_neg(&t, k);
		rank++;
		keep_rt(mk, &t);
	}
	keep_rt_force(mk, &t, 1);
	s->D = D;
	s->U = U;
	s->Ui = Ui;
	s->V = V;
	s->Vi = Vi;
	s->rank = rank;
	if (!zm_snf_check(a, s))
		fail("internal: Smith normal form check failed");
	return rank;
}

void
zsnf_free(struct zsnf *s)
{
	zm_free(&s->D);
	zm_free(&s->U);
	zm_free(&s->V);
	zm_free(&s->Ui);
	zm_free(&s->Vi);
}

int
zm_snf_check(const struct zmat *a, const struct zsnf *s)
{
	struct zmat p, q;
	zmark mk = z_mark();
	int ok = 1, i, j, n;

	if (s->U.r != a->r || s->U.c != a->r || s->V.r != a->c ||
	    s->V.c != a->c || s->D.r != a->r || s->D.c != a->c)
		return 0;
	zm_init(&p, 0, 0);
	zm_init(&q, 0, 0);
	zm_mul(&p, &s->U, a);
	zm_mul(&q, &p, &s->V);
	if (!zm_eq(&q, &s->D))
		ok = 0;
	if (ok && s->Ui.a) {
		zm_mul(&p, &s->U, &s->Ui);
		if (!zm_is_identity(&p))
			ok = 0;
	} else if (ok && !z_is_pm1(zm_det(&s->U)))
		ok = 0;
	if (ok && s->Vi.a) {
		zm_mul(&p, &s->V, &s->Vi);
		if (!zm_is_identity(&p))
			ok = 0;
	} else if (ok && !z_is_pm1(zm_det(&s->V)))
		ok = 0;
	n = a->r < a->c ? a->r : a->c;
	for (i = 0; ok && i < s->D.r; i++)
		for (j = 0; j < s->D.c; j++) {
			zint x = ZM(&s->D, i, j);
			if (i != j && x != 0)
				ok = 0;
			if (i == j && (z_sign(x) < 0 || (i < s->rank) != (x != 0)))
				ok = 0;
			if (i == j && i + 1 < n && x != 0 &&
			    !z_divides(x, ZM(&s->D, i + 1, i + 1)))
				ok = 0;
		}
	zm_free(&p);
	zm_free(&q);
	z_release(mk);
	return ok;
}

int
zm_snf_diag(const struct zsnf *s, zint *d)
{
	int i, n = s->D.r < s->D.c ? s->D.r : s->D.c;

	for (i = 0; i < n; i++)
		d[i] = ZM(&s->D, i, i);
	return n;
}

/* ---- kernel, image, solving ---- */

/* dense: U A^T = H; the rows of U past the rank span the left kernel of
 * A^T, i.e. ker A, saturated because U is unimodular */
static int
kernel_hnf(const struct zmat *a, struct zmat *k)
{
	struct zmat at, h, u, kt, kh;
	int rank, dim, i, j;

	zm_init(&at, 0, 0);
	memset(&h, 0, sizeof h);
	memset(&u, 0, sizeof u);
	zm_transpose(&at, a);
	rank = zm_hnf(&at, &h, &u, NULL);
	dim = a->c - rank;
	zm_init(&kt, dim, a->c);
	for (i = 0; i < dim; i++)
		for (j = 0; j < a->c; j++)
			ZM(&kt, i, j) = ZM(&u, rank + i, j);
	memset(&kh, 0, sizeof kh);
	zm_hnf(&kt, &kh, NULL, NULL);
	zm_transpose(k, &kh);
	{
		struct zmat chk;
		zm_init(&chk, 0, 0);
		zm_mul(&chk, a, k);
		if (!zm_is_zero(&chk))
			fail("internal: kernel check failed");
		zm_free(&chk);
	}
	zm_free(&at);
	zm_free(&h);
	zm_free(&u);
	zm_free(&kt);
	zm_free(&kh);
	return dim;
}

int
zm_kernel(const struct zmat *a, struct zmat *k)
{
	struct zsnf S;
	struct zmat kt, kh;
	int dim, i, j;

	size_t nz = 0, e, ne = (size_t)a->r * (size_t)a->c;

	for (e = 0; e < ne; e++)
		nz += a->a[e] != 0;
	if (4 * nz > ne)
		return kernel_hnf(a, k);
	/* sparse: U A V = D, the columns of V past the rank span ker A, and V
	 * unimodular makes the basis saturated; then the Hermite form of the
	 * basis rows makes it canonical (the same basis as kernel_hnf) */
	zm_snf(a, &S);
	dim = a->c - S.rank;
	zm_init(&kt, dim, a->c);
	for (i = 0; i < dim; i++)
		for (j = 0; j < a->c; j++)
			ZM(&kt, i, j) = ZM(&S.V, j, S.rank + i);
	memset(&kh, 0, sizeof kh);
	zm_hnf(&kt, &kh, NULL, NULL);
	zm_transpose(k, &kh);
	{
		struct zmat chk;
		zm_init(&chk, 0, 0);
		zm_mul(&chk, a, k);
		if (!zm_is_zero(&chk))
			fail("internal: kernel check failed");
		zm_free(&chk);
	}
	zsnf_free(&S);
	zm_free(&kt);
	zm_free(&kh);
	return dim;
}

int
zm_image(const struct zmat *a, struct zmat *im)
{
	struct zmat at, h, hr;
	int rank;

	zm_init(&at, 0, 0);
	memset(&h, 0, sizeof h);
	memset(&hr, 0, sizeof hr);
	zm_transpose(&at, a);
	rank = zm_hnf(&at, &h, NULL, NULL);
	zm_sub_rows(&hr, &h, 0, rank);
	zm_transpose(im, &hr);
	zm_free(&at);
	zm_free(&h);
	zm_free(&hr);
	return rank;
}

int
zm_solve(const struct zmat *a, const struct zmat *b, struct zmat *x)
{
	struct zsnf s;
	struct zmat c, y, chk;
	int i, j, ok = 1;

	if (b->r != a->r)
		fail("solve: %d equations, right-hand side has %d rows", a->r,
		    b->r);
	zm_snf(a, &s);
	zm_init(&c, 0, 0);
	zm_mul(&c, &s.U, b);
	zm_init(&y, a->c, b->c);
	for (j = 0; j < b->c && ok; j++)
		for (i = 0; i < a->r; i++) {
			zint ci = ZM(&c, i, j);
			if (i < s.rank) {
				zint d = ZM(&s.D, i, i), q, r;
				z_divmod_trunc(ci, d, &q, &r);
				if (r != 0) {
					ok = 0;
					break;
				}
				ZM(&y, i, j) = q;
			} else if (ci != 0) {
				ok = 0;
				break;
			}
		}
	if (ok) {
		zm_init(&chk, 0, 0);
		zm_mul(x, &s.V, &y);
		zm_mul(&chk, a, x);
		if (!zm_eq(&chk, b))
			fail("internal: solution check failed");
		zm_free(&chk);
	}
	zm_free(&c);
	zm_free(&y);
	zsnf_free(&s);
	return ok;
}

int
zm_solve_vec(const struct zmat *a, const zint *b, zint *x)
{
	struct zmat B, X;
	int i, ok;

	zm_init(&B, a->r, 1);
	for (i = 0; i < a->r; i++)
		ZM(&B, i, 0) = b[i];
	memset(&X, 0, sizeof X);
	ok = zm_solve(a, &B, &X);
	if (ok)
		for (i = 0; i < a->c; i++)
			x[i] = ZM(&X, i, 0);
	zm_free(&B);
	zm_free(&X);
	return ok;
}

int
zm_inverse(const struct zmat *a, struct zmat *inv)
{
	struct zmat h, u;
	int rank;

	if (a->r != a->c)
		return 0;
	memset(&h, 0, sizeof h);
	memset(&u, 0, sizeof u);
	rank = zm_hnf(a, &h, &u, NULL);
	if (rank != a->r || !zm_is_identity(&h)) {
		zm_free(&h);
		zm_free(&u);
		return 0;
	}
	zm_move(inv, &u);
	zm_free(&h);
	return 1;
}

int
zm_lattice_contains(const struct zmat *l, const zint *x, zint *w)
{
	zint *t = xcalloc((size_t)l->c + 1, sizeof *t);
	int ok = zm_solve_vec(l, x, t);

	if (ok && w)
		memcpy(w, t, (size_t)l->c * sizeof *t);
	free(t);
	return ok;
}

int
zm_lattice_le(const struct zmat *l1, const struct zmat *l2)
{
	struct zmat x;
	int ok;

	if (l1->r != l2->r)
		fail("internal: lattices in Z^%d and Z^%d", l1->r, l2->r);
	if (l1->c == 0)
		return 1;
	memset(&x, 0, sizeof x);
	ok = zm_solve(l2, l1, &x);
	zm_free(&x);
	return ok;
}

int
zm_lattice_eq(const struct zmat *l1, const struct zmat *l2)
{
	return zm_lattice_le(l1, l2) && zm_lattice_le(l2, l1);
}

void
zm_lattice_sum(struct zmat *r, const struct zmat *l1, const struct zmat *l2)
{
	struct zmat t;

	zm_init(&t, 0, 0);
	zm_hcat(&t, l1, l2);
	zm_image(&t, r);
	zm_free(&t);
}

void
zm_lattice_meet(struct zmat *r, const struct zmat *l1, const struct zmat *l2)
{
	struct zmat b1, b2, nb2, cat, k, top, m;

	memset(&b1, 0, sizeof b1);
	memset(&b2, 0, sizeof b2);
	memset(&k, 0, sizeof k);
	zm_init(&nb2, 0, 0);
	zm_init(&cat, 0, 0);
	zm_init(&top, 0, 0);
	zm_init(&m, 0, 0);
	zm_image(l1, &b1);
	zm_image(l2, &b2);
	zm_scale(&nb2, &b2, z_from_i64(-1));
	zm_hcat(&cat, &b1, &nb2);
	zm_kernel(&cat, &k);
	zm_sub_rows(&top, &k, 0, b1.c);
	zm_mul(&m, &b1, &top);
	zm_image(&m, r);
	zm_free(&b1);
	zm_free(&b2);
	zm_free(&nb2);
	zm_free(&cat);
	zm_free(&k);
	zm_free(&top);
	zm_free(&m);
}

/* ---- matrices over Q ---- */

void
qm_init(struct qmat *m, int r, int c)
{
	size_t n = (size_t)r * (size_t)c, i;

	m->r = r;
	m->c = c;
	m->n = xcalloc(n ? n : 1, sizeof *m->n);
	m->d = xcalloc(n ? n : 1, sizeof *m->d);
	for (i = 0; i < n; i++)
		m->d[i] = Z_ONE;
}

void
qm_free(struct qmat *m)
{
	free(m->n);
	free(m->d);
	m->n = m->d = NULL;
	m->r = m->c = 0;
}

static void
qm_move(struct qmat *dst, struct qmat *src)
{
	if (dst == src)
		return;
	free(dst->n);
	free(dst->d);
	*dst = *src;
	src->n = src->d = NULL;
}

void
qm_from_zmat(struct qmat *m, const struct zmat *a)
{
	size_t i, n = (size_t)a->r * (size_t)a->c;

	qm_init(m, a->r, a->c);
	for (i = 0; i < n; i++)
		m->n[i] = a->a[i];
}

struct zq
qm_get(const struct qmat *m, int i, int j)
{
	struct zq q;
	size_t k = (size_t)i * (size_t)m->c + (size_t)j;

	q.n = m->n[k];
	q.d = m->d[k];
	return q;
}

void
qm_set(struct qmat *m, int i, int j, struct zq v)
{
	size_t k = (size_t)i * (size_t)m->c + (size_t)j;

	m->n[k] = v.n;
	m->d[k] = v.d;
}

int
qm_parse(struct qmat *m, const char *s, const char **end)
{
	/* the integer parser, with a / allowed: parse as text rows */
	const char *p = skipws(s);
	struct zq *v = NULL;
	int nv = 0, cap = 0, r = 0, c = -1, row = 0, inner;

	if (*p != '[')
		return 0;
	p = skipws(p + 1);
	inner = *p == '[';
	if (!inner)
		p = skipws(s), r = 0;
	for (;;) {
		p = skipws(p);
		if (*p != '[')
			goto bad;
		p = skipws(p + 1);
		row = 0;
		while (*p != ']') {
			struct zq q;
			const char *e;
			if (!zq_parse(p, &q, &e))
				goto bad;
			if (nv == cap) {
				cap = cap ? cap * 2 : 16;
				v = xrealloc(v, (size_t)cap * sizeof *v);
			}
			v[nv++] = q;
			row++;
			p = skipws(e);
			if (*p == ',')
				p = skipws(p + 1);
		}
		p++;
		if (c < 0)
			c = row;
		else if (c != row)
			goto bad;
		r++;
		if (!inner)
			break;
		p = skipws(p);
		if (*p == ',')
			p++;
		else if (*p == ']') {
			p++;
			break;
		}
	}
	qm_init(m, c ? r : 0, c);
	{
		int k;
		for (k = 0; k < nv; k++) {
			m->n[k] = v[k].n;
			m->d[k] = v[k].d;
		}
	}
	free(v);
	if (end)
		*end = p;
	return 1;
bad:
	free(v);
	return 0;
}

void
qm_print(struct sbuf *b, const struct qmat *m)
{
	int i, j;

	if (m->r == 0) {
		sb_puts(b, "[]");
		return;
	}
	sb_putc(b, '[');
	for (i = 0; i < m->r; i++) {
		sb_puts(b, i ? ", [" : "[");
		for (j = 0; j < m->c; j++) {
			if (j)
				sb_puts(b, ", ");
			zq_print(b, qm_get(m, i, j));
		}
		sb_putc(b, ']');
	}
	sb_putc(b, ']');
}

void
qm_mul(struct qmat *r, const struct qmat *a, const struct qmat *b)
{
	struct qmat t;
	int i, j, k;

	if (a->c != b->r)
		fail("internal: matrix product %dx%d * %dx%d", a->r, a->c, b->r,
		    b->c);
	qm_init(&t, a->r, b->c);
	for (i = 0; i < a->r; i++)
		for (j = 0; j < b->c; j++) {
			struct zq s = zq_int(0);
			for (k = 0; k < a->c; k++)
				if (a->n[(size_t)i * a->c + k] != 0)
					s = zq_add(s, zq_mul(qm_get(a, i, k),
					    qm_get(b, k, j)));
			qm_set(&t, i, j, s);
		}
	qm_move(r, &t);
}

int
qm_rref(struct qmat *m, int *pivcol)
{
	int r = 0, c, i, j;

	for (c = 0; c < m->c && r < m->r; c++) {
		struct zq inv;
		for (i = r; i < m->r && m->n[(size_t)i * m->c + c] == 0; i++)
			;
		if (i == m->r)
			continue;
		if (i != r)
			for (j = 0; j < m->c; j++) {
				struct zq x = qm_get(m, i, j);
				qm_set(m, i, j, qm_get(m, r, j));
				qm_set(m, r, j, x);
			}
		inv = zq_inv(qm_get(m, r, c));
		for (j = 0; j < m->c; j++)
			qm_set(m, r, j, zq_mul(qm_get(m, r, j), inv));
		for (i = 0; i < m->r; i++) {
			struct zq f;
			if (i == r || m->n[(size_t)i * m->c + c] == 0)
				continue;
			f = qm_get(m, i, c);
			for (j = 0; j < m->c; j++)
				qm_set(m, i, j, zq_sub(qm_get(m, i, j), zq_mul(f,
				    qm_get(m, r, j))));
		}
		if (pivcol)
			pivcol[r] = c;
		r++;
	}
	return r;
}

static void
qm_copy(struct qmat *d, const struct qmat *s)
{
	size_t n = (size_t)s->r * (size_t)s->c;

	qm_init(d, s->r, s->c);
	memcpy(d->n, s->n, n * sizeof *d->n);
	memcpy(d->d, s->d, n * sizeof *d->d);
}

int
qm_rank(const struct qmat *a)
{
	struct qmat m;
	int r;

	qm_copy(&m, a);
	r = qm_rref(&m, NULL);
	qm_free(&m);
	return r;
}

struct zq
qm_det(const struct qmat *a)
{
	struct qmat m;
	struct zq det = zq_int(Z_ONE);
	int c, i, j, n = a->r;

	if (a->r != a->c)
		fail("det: the matrix is %dx%d, not square", a->r, a->c);
	qm_copy(&m, a);
	for (c = 0; c < n; c++) {
		for (i = c; i < n && m.n[(size_t)i * n + c] == 0; i++)
			;
		if (i == n) {
			qm_free(&m);
			return zq_int(0);
		}
		if (i != c) {
			for (j = 0; j < n; j++) {
				struct zq x = qm_get(&m, i, j);
				qm_set(&m, i, j, qm_get(&m, c, j));
				qm_set(&m, c, j, x);
			}
			det = zq_neg(det);
		}
		det = zq_mul(det, qm_get(&m, c, c));
		for (i = c + 1; i < n; i++) {
			struct zq f;
			if (m.n[(size_t)i * n + c] == 0)
				continue;
			f = zq_div(qm_get(&m, i, c), qm_get(&m, c, c));
			for (j = c; j < n; j++)
				qm_set(&m, i, j, zq_sub(qm_get(&m, i, j), zq_mul(f,
				    qm_get(&m, c, j))));
		}
	}
	qm_free(&m);
	return det;
}

int
qm_inverse(const struct qmat *a, struct qmat *inv)
{
	struct qmat m, chk;
	int n = a->r, i, j, r;

	if (a->r != a->c)
		return 0;
	qm_init(&m, n, 2 * n);
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			qm_set(&m, i, j, qm_get(a, i, j));
		qm_set(&m, i, n + i, zq_int(Z_ONE));
	}
	r = qm_rref(&m, NULL);
	for (i = 0; i < n && r >= n; i++)
		if (!zq_eq(qm_get(&m, i, i), zq_int(Z_ONE)))
			r = 0;
	if (r < n) {
		qm_free(&m);
		return 0;
	}
	{
		struct qmat t;
		qm_init(&t, n, n);
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				qm_set(&t, i, j, qm_get(&m, i, n + j));
		qm_move(inv, &t);
	}
	qm_free(&m);
	qm_init(&chk, 0, 0);
	qm_mul(&chk, a, inv);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			if (!zq_eq(qm_get(&chk, i, j), zq_int(i == j ? Z_ONE : 0)))
				fail("internal: inverse check failed");
	qm_free(&chk);
	return 1;
}

int
qm_nullspace(const struct qmat *a, struct qmat *ns)
{
	struct qmat m, t;
	int *piv = xcalloc((size_t)a->r + 1, sizeof *piv);
	int *isp = xcalloc((size_t)a->c + 1, sizeof *isp);
	int rank, j, k, f = 0;

	qm_copy(&m, a);
	rank = qm_rref(&m, piv);
	for (k = 0; k < rank; k++)
		isp[piv[k]] = 1;
	qm_init(&t, a->c, a->c - rank);
	for (j = 0; j < a->c; j++) {
		if (isp[j])
			continue;
		qm_set(&t, j, f, zq_int(Z_ONE));
		for (k = 0; k < rank; k++)
			qm_set(&t, piv[k], f, zq_neg(qm_get(&m, k, j)));
		f++;
	}
	qm_move(ns, &t);
	qm_free(&m);
	free(piv);
	free(isp);
	return f;
}

int
qm_solve(const struct qmat *a, const struct qmat *b, struct qmat *x)
{
	struct qmat m, t;
	int *piv = xcalloc((size_t)a->r + 1, sizeof *piv);
	int rank, i, j, k, ok = 1;

	if (b->r != a->r)
		fail("solve: %d equations, right-hand side has %d rows", a->r,
		    b->r);
	qm_init(&m, a->r, a->c + b->c);
	for (i = 0; i < a->r; i++) {
		for (j = 0; j < a->c; j++)
			qm_set(&m, i, j, qm_get(a, i, j));
		for (j = 0; j < b->c; j++)
			qm_set(&m, i, a->c + j, qm_get(b, i, j));
	}
	rank = qm_rref(&m, piv);
	for (k = 0; k < rank; k++)
		if (piv[k] >= a->c)
			ok = 0;
	if (ok) {
		qm_init(&t, a->c, b->c);
		for (k = 0; k < rank; k++)
			for (j = 0; j < b->c; j++)
				qm_set(&t, piv[k], j, qm_get(&m, k, a->c + j));
		qm_move(x, &t);
	}
	qm_free(&m);
	free(piv);
	return ok;
}

/* ---- over Z/m through ring.h ---- */

void
zm_reduce_mod(const struct zmat *a, u64 m, struct rmat *out)
{
	size_t i, n = (size_t)a->r * (size_t)a->c;

	rm_init(out, a->r, a->c);
	for (i = 0; i < n; i++)
		out->a[i] = z_mod_u64(a->a[i], m);
}

int
zm_rank_mod(const struct zmat *a, u64 p)
{
	struct ring R;
	struct rmat m;
	int r;

	if (!nt_isprime(p))
		fail("rank mod %llu: the modulus must be prime",
		    (unsigned long long)p);
	ring_zn(&R, p);
	zm_reduce_mod(a, p, &m);
	r = rm_rref(&R, &m, NULL);
	rm_free(&m);
	return r;
}

int
zm_nullspace_mod(const struct zmat *a, u64 p, struct rmat *ns)
{
	struct ring R;
	struct rmat m, rows;
	int dim, i, j;

	if (!nt_isprime(p))
		fail("nullspace mod %llu: the modulus must be prime",
		    (unsigned long long)p);
	ring_zn(&R, p);
	zm_reduce_mod(a, p, &m);
	dim = rm_nullspace(&R, &m, &rows);
	(void)dim;
	dim = rows.r;
	/* columns, as everywhere in absalg */
	rm_init(ns, a->c, dim);
	for (i = 0; i < dim; i++)
		for (j = 0; j < a->c; j++)
			RM(ns, j, i) = RM(&rows, i, j);
	rm_free(&rows);
	rm_free(&m);
	return dim;
}
