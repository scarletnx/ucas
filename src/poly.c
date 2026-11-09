/*
 * poly: the algebra core (alg.h).  Rationals with 128-bit intermediates,
 * an arena, sparse multivariate polynomials over Q in atoms, their gcd
 * (primitive PRS, recursive content), exact division, resultants by
 * fraction-free elimination of the Sylvester matrix, factorisation
 * (content, monomials, square-free, univariate over Q by rational roots
 * and Kronecker's method, homogeneous bivariate through t = x/y),
 * rational functions, dense univariate polynomials, Q(sqrt d), and the
 * conversions node <-> polynomial.
 */
#include "alg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int alg_ovf;

/* ---- arena ---- */

struct arblk {
	char *p;
	size_t n, used;
};
static struct arblk *blk;
static int nblk, cur;

size_t
ar_mark(void)
{
	if (nblk == 0)
		return 0;
	return ((size_t)cur << 40) | blk[cur].used;
}

void
ar_release(size_t m)
{
	if (nblk == 0)
		return;
	cur = (int)(m >> 40);
	blk[cur].used = m & (((size_t)1 << 40) - 1);
}

void *
ar_alloc(size_t n)
{
	void *r;

	n = (n + 15) & ~(size_t)15;
	if (n == 0)
		n = 16;
	if (nblk == 0) {
		blk = xcalloc(64, sizeof *blk);
		nblk = 1;
		blk[0].n = (size_t)1 << 22;
		blk[0].p = xmalloc(blk[0].n);
	}
	if (blk[cur].used + n > blk[cur].n) {
		cur++;
		if (cur == 64)
			fail("algebra: out of memory (the expression is too large)");
		if (cur == nblk) {
			nblk++;
			blk[cur].n = n > ((size_t)1 << 22) ? n : (size_t)1 << 22;
			blk[cur].p = xmalloc(blk[cur].n);
		} else if (blk[cur].n < n) {
			free(blk[cur].p);
			blk[cur].n = n;
			blk[cur].p = xmalloc(n);
		}
		blk[cur].used = 0;
	}
	r = blk[cur].p + blk[cur].used;
	blk[cur].used += n;
	return r;
}

/* ---- rationals ---- */

typedef __int128 i128;

static aq
qred(i128 n, i128 d)
{
	i128 a, b, t;

	if (d == 0) {
		alg_ovf = 1;
		return (aq){0, 1};
	}
	if (d < 0)
		n = -n, d = -d;
	a = n < 0 ? -n : n;
	b = d;
	while (b) {
		t = a % b;
		a = b;
		b = t;
	}
	if (a > 1)
		n /= a, d /= a;
	if (n == 0)
		d = 1;
	if (n > INT64_MAX || n < -INT64_MAX || d > INT64_MAX) {
		alg_ovf = 1;
		return (aq){0, 1};
	}
	return (aq){(int64_t)n, (int64_t)d};
}

aq
qi(int64_t n)
{
	return (aq){n, 1};
}

aq
qmk(int64_t n, int64_t d)
{
	return qred(n, d);
}

aq
qadd(aq a, aq b)
{
	if (a.d == 1 && b.d == 1)
		return qred((i128)a.n + b.n, 1);
	return qred((i128)a.n * b.d + (i128)b.n * a.d, (i128)a.d * b.d);
}

aq
qsub(aq a, aq b)
{
	return qadd(a, qneg(b));
}

aq
qmul(aq a, aq b)
{
	return qred((i128)a.n * b.n, (i128)a.d * b.d);
}

aq
qdiv(aq a, aq b)
{
	if (b.n == 0) {
		alg_ovf = 1;
		return (aq){0, 1};
	}
	return qred((i128)a.n * b.d, (i128)a.d * b.n);
}

aq
qneg(aq a)
{
	a.n = -a.n;
	return a;
}

int
qeq(aq a, aq b)
{
	return a.n == b.n && a.d == b.d;
}

int
qsgn(aq a)
{
	return a.n > 0 ? 1 : a.n < 0 ? -1 : 0;
}

double
qd(aq a)
{
	return (double)a.n / (double)a.d;
}

int64_t
lcm64(int64_t a, int64_t b)
{
	i128 r;

	if (a == 0 || b == 0)
		return a ? a : b;
	r = (i128)(a / gcd64(a, b)) * b;
	if (r < 0)
		r = -r;
	if (r > INT64_MAX) {
		alg_ovf = 1;
		return 1;
	}
	return (int64_t)r;
}

/* v is exactly n/d (in double) with d <= 10^6 */
int
qexact(double v, aq *r)
{
	double x, h0 = 0, h1 = 1, k0 = 1, k1 = 0, a;
	int i;

	if (!isfinite(v) || fabs(v) > 9e15)
		return 0;
	if (v == floor(v)) {
		*r = qi((int64_t)v);
		return 1;
	}
	x = fabs(v);
	for (i = 0; i < 40; i++) {
		double h, k;
		a = floor(x);
		h = a * h1 + h0;
		k = a * k1 + k0;
		if (k > 1e6)
			return 0;
		h0 = h1, h1 = h, k0 = k1, k1 = k;
		if (h1 / k1 == fabs(v)) {
			*r = qmk((int64_t)(v < 0 ? -h1 : h1), (int64_t)k1);
			return 1;
		}
		if (x - a < 1e-300)
			return 0;
		x = 1 / (x - a);
	}
	return 0;
}

/* ---- atoms ---- */

nid at_node[AV];
int nat;

void
atoms_reset(void)
{
	nat = 0;
}

int
atom_find(nid e)
{
	int i;

	for (i = 0; i < nat; i++)
		if (at_node[i] == e)
			return i;
	return -1;
}

int
atom_of(nid e)
{
	int i = atom_find(e);

	if (i >= 0)
		return i;
	if (nat == AV)
		return -1;
	at_node[nat] = e;
	return nat++;
}

/* ---- multivariate polynomials ---- */

static int
mcmp(const struct mono *a, const struct mono *b)
{
	int i;

	for (i = 0; i < AV; i++)
		if (a->e[i] != b->e[i])
			return a->e[i] > b->e[i] ? -1 : 1;
	return 0;
}

static int
mcmp_q(const void *a, const void *b)
{
	return mcmp(a, b);
}

static struct mp
mp_new(int n)
{
	struct mp r;

	r.n = 0;
	r.t = ar_alloc((size_t)(n > 0 ? n : 1) * sizeof *r.t);
	return r;
}

struct mp
mp_zero(void)
{
	struct mp r = {0, NULL};
	return r;
}

struct mp
mp_const(aq c)
{
	struct mp r = mp_new(1);

	if (c.n == 0)
		return mp_zero();
	memset(&r.t[0], 0, sizeof r.t[0]);
	r.t[0].c = c;
	r.n = 1;
	return r;
}

struct mp
mp_atom(int v)
{
	struct mp r = mp_new(1);

	memset(&r.t[0], 0, sizeof r.t[0]);
	r.t[0].c = qi(1);
	r.t[0].e[v] = 1;
	r.n = 1;
	return r;
}

struct mp
mp_copy(struct mp a)
{
	struct mp r = mp_new(a.n);

	memcpy(r.t, a.t, (size_t)a.n * sizeof *r.t);
	r.n = a.n;
	return r;
}

int
mp_iszero(struct mp a)
{
	return a.n == 0;
}

int
mp_isconst(struct mp a, aq *c)
{
	int i;

	if (a.n == 0) {
		if (c)
			*c = qi(0);
		return 1;
	}
	if (a.n > 1)
		return 0;
	for (i = 0; i < AV; i++)
		if (a.t[0].e[i])
			return 0;
	if (c)
		*c = a.t[0].c;
	return 1;
}

int
mp_eq(struct mp a, struct mp b)
{
	int i;

	if (a.n != b.n)
		return 0;
	for (i = 0; i < a.n; i++)
		if (mcmp(&a.t[i], &b.t[i]) || !qeq(a.t[i].c, b.t[i].c))
			return 0;
	return 1;
}

static struct mp
mp_addsc(struct mp a, struct mp b, int sg)
{
	struct mp r = mp_new(a.n + b.n);
	int i = 0, j = 0, c;

	while (i < a.n || j < b.n) {
		if (i == a.n)
			c = 1;
		else if (j == b.n)
			c = -1;
		else
			c = mcmp(&a.t[i], &b.t[j]);
		if (c < 0)
			r.t[r.n++] = a.t[i++];
		else if (c > 0) {
			r.t[r.n] = b.t[j++];
			if (sg < 0)
				r.t[r.n].c = qneg(r.t[r.n].c);
			r.n++;
		} else {
			aq s = sg > 0 ? qadd(a.t[i].c, b.t[j].c) :
			    qsub(a.t[i].c, b.t[j].c);
			if (s.n) {
				r.t[r.n] = a.t[i];
				r.t[r.n++].c = s;
			}
			i++, j++;
		}
	}
	return r;
}

struct mp
mp_add(struct mp a, struct mp b)
{
	return mp_addsc(a, b, 1);
}

struct mp
mp_sub(struct mp a, struct mp b)
{
	return mp_addsc(a, b, -1);
}

struct mp
mp_neg(struct mp a)
{
	return mp_scale(a, qi(-1));
}

struct mp
mp_scale(struct mp a, aq c)
{
	struct mp r;
	int i;

	if (c.n == 0)
		return mp_zero();
	r = mp_copy(a);
	for (i = 0; i < r.n; i++)
		r.t[i].c = qmul(r.t[i].c, c);
	return r;
}

/* a times the monomial m */
static struct mp
mp_mulmono(struct mp a, const struct mono *m)
{
	struct mp r = mp_new(a.n);
	int i, k;

	for (i = 0; i < a.n; i++) {
		r.t[i] = a.t[i];
		r.t[i].c = qmul(a.t[i].c, m->c);
		for (k = 0; k < AV; k++) {
			int e = a.t[i].e[k] + m->e[k];
			if (e > 255) {
				alg_ovf = 1;
				e = 255;
			}
			r.t[i].e[k] = (uint8_t)e;
		}
	}
	r.n = a.n;
	return r;
}

struct mp
mp_mul(struct mp a, struct mp b)
{
	struct mp r;
	int i, j, k, n = 0;

	if (a.n == 0 || b.n == 0)
		return mp_zero();
	if (b.n == 1)
		return mp_mulmono(a, &b.t[0]);
	if (a.n == 1)
		return mp_mulmono(b, &a.t[0]);
	if ((size_t)a.n * b.n > 2000000) {
		alg_ovf = 1;
		return mp_zero();
	}
	r = mp_new(a.n * b.n);
	for (i = 0; i < a.n; i++)
		for (j = 0; j < b.n; j++) {
			struct mono *m = &r.t[n++];
			m->c = qmul(a.t[i].c, b.t[j].c);
			for (k = 0; k < AV; k++) {
				int e = a.t[i].e[k] + b.t[j].e[k];
				if (e > 255) {
					alg_ovf = 1;
					e = 255;
				}
				m->e[k] = (uint8_t)e;
			}
		}
	qsort(r.t, (size_t)n, sizeof *r.t, mcmp_q);
	for (i = 0, j = -1; i < n; i++) {
		if (j >= 0 && !mcmp(&r.t[j], &r.t[i]))
			r.t[j].c = qadd(r.t[j].c, r.t[i].c);
		else {
			if (j >= 0 && r.t[j].c.n == 0)
				r.t[j] = r.t[i];
			else
				r.t[++j] = r.t[i];
		}
	}
	r.n = j + 1;
	if (r.n && r.t[r.n - 1].c.n == 0)
		r.n--;
	/* zero coefficients left in the middle */
	for (i = 0, j = 0; i < r.n; i++)
		if (r.t[i].c.n)
			r.t[j++] = r.t[i];
	r.n = j;
	return r;
}

struct mp
mp_pow(struct mp a, int n)
{
	struct mp r = mp_const(qi(1));

	if (n < 0) {
		alg_ovf = 1;
		return mp_zero();
	}
	while (n > 0 && !alg_ovf) {
		if (n & 1)
			r = mp_mul(r, a);
		n >>= 1;
		if (n)
			a = mp_mul(a, a);
	}
	return r;
}

int
mp_deg(struct mp a, int v)
{
	int i, d = a.n ? 0 : -1;

	for (i = 0; i < a.n; i++)
		if (a.t[i].e[v] > d)
			d = a.t[i].e[v];
	return d;
}

int
mp_tdeg(struct mp a)
{
	int i, k, d = a.n ? 0 : -1;

	for (i = 0; i < a.n; i++) {
		int s = 0;
		for (k = 0; k < AV; k++)
			s += a.t[i].e[k];
		if (s > d)
			d = s;
	}
	return d;
}

uint32_t
mp_vars(struct mp a)
{
	uint32_t m = 0;
	int i, k;

	for (i = 0; i < a.n; i++)
		for (k = 0; k < AV; k++)
			if (a.t[i].e[k])
				m |= 1u << k;
	return m;
}

struct mp
mp_coef(struct mp a, int v, int k)
{
	struct mp r = mp_new(a.n);
	int i;

	for (i = 0; i < a.n; i++)
		if (a.t[i].e[v] == k) {
			r.t[r.n] = a.t[i];
			r.t[r.n++].e[v] = 0;
		}
	/* zeroing a column can break the order when v is not first */
	qsort(r.t, (size_t)r.n, sizeof *r.t, mcmp_q);
	return r;
}

struct mp
mp_lcv(struct mp a, int v)
{
	return mp_coef(a, v, mp_deg(a, v));
}

int
mp_divexact(struct mp a, struct mp b, struct mp *q)
{
	struct mp r = a, acc = mp_zero();
	int it = 0, k;

	if (b.n == 0)
		return 0;
	while (r.n && !alg_ovf) {
		struct mono m;
		if (++it > 20000)
			return 0;
		for (k = 0; k < AV; k++) {
			if (r.t[0].e[k] < b.t[0].e[k])
				return 0;
			m.e[k] = r.t[0].e[k] - b.t[0].e[k];
		}
		m.c = qdiv(r.t[0].c, b.t[0].c);
		{
			struct mp mm;
			mm.n = 1;
			mm.t = &m;
			acc = mp_add(acc, mp_copy(mm));
		}
		r = mp_sub(r, mp_mulmono(b, &m));
	}
	if (alg_ovf)
		return 0;
	*q = acc;
	return 1;
}

static struct mp
mp_vpow(int v, int k)
{
	struct mp r = mp_const(qi(1));

	if (k > 0)
		r.t[0].e[v] = (uint8_t)k;
	return r;
}

struct mp
mp_prem(struct mp a, struct mp b, int v)
{
	int db = mp_deg(b, v), dr;
	struct mp lb = mp_lcv(b, v), r = a;

	while (r.n && (dr = mp_deg(r, v)) >= db && !alg_ovf) {
		struct mp lr = mp_lcv(r, v);
		r = mp_sub(mp_mul(lb, r), mp_mul(mp_mul(lr, mp_vpow(v, dr - db)),
		    b));
		r = mp_prim(r);
	}
	return r;
}

aq
mp_qcontent(struct mp a)
{
	int64_t g = 0, l = 1;
	int i;

	if (a.n == 0)
		return qi(1);
	for (i = 0; i < a.n; i++) {
		g = gcd64(g, a.t[i].c.n);
		l = lcm64(l, a.t[i].c.d);
	}
	if (a.t[0].c.n < 0)
		g = -g;
	return qmk(g, l);
}

struct mp
mp_prim(struct mp a)
{
	if (a.n == 0)
		return a;
	return mp_scale(a, qdiv(qi(1), mp_qcontent(a)));
}

static int
topvar(uint32_t m)
{
	int k;

	for (k = AV - 1; k >= 0; k--)
		if (m & (1u << k))
			return k;
	return -1;
}

struct mp
mp_content(struct mp a, int v)
{
	struct mp g = mp_zero();
	int k, d = mp_deg(a, v);

	for (k = d; k >= 0 && !alg_ovf; k--) {
		struct mp c = mp_coef(a, v, k);
		if (c.n == 0)
			continue;
		g = g.n ? mp_gcd(g, c) : mp_prim(c);
		if (mp_isconst(g, NULL))
			return mp_const(qi(1));
	}
	return g.n ? g : mp_const(qi(1));
}

static int gcd_depth;

struct mp
mp_gcd(struct mp a, struct mp b)
{
	uint32_t ma, mb;
	int v;
	struct mp ca, cb, pa, pb, g, t;

	if (alg_ovf)
		return mp_const(qi(1));
	if (a.n == 0)
		return mp_prim(b);
	if (b.n == 0)
		return mp_prim(a);
	ma = mp_vars(a);
	mb = mp_vars(b);
	if (ma == 0 || mb == 0)
		return mp_const(qi(1));
	if (++gcd_depth > 40) {
		gcd_depth--;
		alg_ovf = 1;
		return mp_const(qi(1));
	}
	v = topvar(ma | mb);
	if (!(ma & (1u << v))) {
		g = mp_gcd(a, mp_content(b, v));
		goto out;
	}
	if (!(mb & (1u << v))) {
		g = mp_gcd(mp_content(a, v), b);
		goto out;
	}
	ca = mp_content(a, v);
	cb = mp_content(b, v);
	if (!mp_divexact(a, ca, &pa) || !mp_divexact(b, cb, &pb)) {
		alg_ovf = 1;
		g = mp_const(qi(1));
		goto out;
	}
	g = mp_gcd(ca, cb);
	if (mp_deg(pa, v) < mp_deg(pb, v))
		t = pa, pa = pb, pb = t;
	pa = mp_prim(pa);
	pb = mp_prim(pb);
	while (pb.n && !alg_ovf) {
		struct mp r = mp_prem(pa, pb, v), cr, pr;
		if (r.n == 0)
			break;
		if (mp_deg(r, v) == 0) {
			pb = mp_const(qi(1));
			break;
		}
		cr = mp_content(r, v);
		if (!mp_divexact(r, cr, &pr)) {
			alg_ovf = 1;
			break;
		}
		pa = pb;
		pb = mp_prim(pr);
	}
	g = mp_prim(mp_mul(g, pb));
out:
	gcd_depth--;
	return g;
}

struct mp
mp_diff(struct mp a, int v)
{
	struct mp r = mp_new(a.n);
	int i;

	for (i = 0; i < a.n; i++)
		if (a.t[i].e[v]) {
			r.t[r.n] = a.t[i];
			r.t[r.n].c = qmul(a.t[i].c, qi(a.t[i].e[v]));
			r.t[r.n++].e[v]--;
		}
	qsort(r.t, (size_t)r.n, sizeof *r.t, mcmp_q);
	return r;
}

struct mp
mp_subst(struct mp a, int v, struct mp val)
{
	int d = mp_deg(a, v), k;
	struct mp r = mp_zero(), pw = mp_const(qi(1));

	for (k = 0; k <= d && !alg_ovf; k++) {
		struct mp c = mp_coef(a, v, k);
		if (c.n)
			r = mp_add(r, mp_mul(c, pw));
		if (k < d)
			pw = mp_mul(pw, val);
	}
	return r;
}

double
mp_eval(struct mp a, const double *x)
{
	double s = 0;
	int i, k;

	for (i = 0; i < a.n; i++) {
		double t = qd(a.t[i].c);
		for (k = 0; k < AV; k++)
			if (a.t[i].e[k])
				t *= pow(x[k], a.t[i].e[k]);
		s += t;
	}
	return s;
}

/* ---- resultants ---- */

struct mp
mp_resultant(struct mp a, struct mp b, int v)
{
	int da = mp_deg(a, v), db = mp_deg(b, v), n, i, j, k, sg = 1;
	struct mp *M, prev, r;

	if (a.n == 0 || b.n == 0)
		return mp_zero();
	if (da == 0)
		return mp_pow(a, db);
	if (db == 0)
		return mp_pow(b, da);
	n = da + db;
	if (n > 2 * UD) {
		alg_ovf = 1;
		return mp_zero();
	}
	M = ar_alloc((size_t)n * n * sizeof *M);
	for (i = 0; i < n * n; i++)
		M[i] = mp_zero();
	for (i = 0; i < db; i++)
		for (k = 0; k <= da; k++)
			M[i * n + i + k] = mp_coef(a, v, da - k);
	for (i = 0; i < da; i++)
		for (k = 0; k <= db; k++)
			M[(db + i) * n + i + k] = mp_coef(b, v, db - k);
	/* Bareiss */
	prev = mp_const(qi(1));
	for (k = 0; k < n - 1 && !alg_ovf; k++) {
		if (M[k * n + k].n == 0) {
			for (i = k + 1; i < n && M[i * n + k].n == 0; i++)
				;
			if (i == n)
				return mp_zero();
			for (j = 0; j < n; j++) {
				struct mp t = M[k * n + j];
				M[k * n + j] = M[i * n + j];
				M[i * n + j] = t;
			}
			sg = -sg;
		}
		for (i = k + 1; i < n; i++)
			for (j = k + 1; j < n; j++) {
				struct mp t = mp_sub(mp_mul(M[k * n + k], M[i * n + j]),
				    mp_mul(M[i * n + k], M[k * n + j]));
				if (!mp_divexact(t, prev, &M[i * n + j])) {
					alg_ovf = 1;
					return mp_zero();
				}
			}
		prev = M[k * n + k];
	}
	r = M[(n - 1) * n + n - 1];
	return sg < 0 ? mp_neg(r) : r;
}

struct mp
mp_discriminant(struct mp a, int v)
{
	int n = mp_deg(a, v);
	struct mp r, q;

	if (n < 1)
		return mp_zero();
	if (n == 1)
		return mp_const(qi(1));
	r = mp_resultant(a, mp_diff(a, v), v);
	if (!mp_divexact(r, mp_lcv(a, v), &q)) {
		alg_ovf = 1;
		return mp_zero();
	}
	if ((n * (n - 1) / 2) & 1)
		q = mp_neg(q);
	return q;
}

/* ---- dense univariate ---- */

void
up_zero(struct up *r)
{
	r->d = -1;
}

void
up_set(struct up *r, const struct up *a)
{
	int i;

	r->d = a->d;
	for (i = 0; i <= a->d; i++)
		r->c[i] = a->c[i];
}

void
up_trim(struct up *r)
{
	while (r->d >= 0 && r->c[r->d].n == 0)
		r->d--;
}

int
up_iszero(const struct up *a)
{
	return a->d < 0;
}

int
up_of_mp(struct mp a, int v, struct up *r)
{
	int i, k;

	r->d = -1;
	for (i = 0; i <= UD; i++)
		r->c[i] = qi(0);
	for (i = 0; i < a.n; i++) {
		for (k = 0; k < AV; k++)
			if (k != v && a.t[i].e[k])
				return 0;
		k = a.t[i].e[v];
		if (k > UD)
			return 0;
		r->c[k] = qadd(r->c[k], a.t[i].c);
		if (k > r->d)
			r->d = k;
	}
	up_trim(r);
	return !alg_ovf;
}

struct mp
mp_of_up(const struct up *a, int v)
{
	struct mp r = mp_new(a->d + 1);
	int k;

	for (k = a->d; k >= 0; k--)
		if (a->c[k].n) {
			memset(&r.t[r.n], 0, sizeof r.t[r.n]);
			r.t[r.n].c = a->c[k];
			r.t[r.n++].e[v] = (uint8_t)k;
		}
	return r;
}

void
up_add(struct up *r, const struct up *a, const struct up *b)
{
	int i, d = a->d > b->d ? a->d : b->d;
	struct up t;

	for (i = 0; i <= d; i++)
		t.c[i] = qadd(i <= a->d ? a->c[i] : qi(0), i <= b->d ? b->c[i] :
		    qi(0));
	t.d = d;
	up_trim(&t);
	up_set(r, &t);
}

void
up_sub(struct up *r, const struct up *a, const struct up *b)
{
	int i, d = a->d > b->d ? a->d : b->d;
	struct up t;

	for (i = 0; i <= d; i++)
		t.c[i] = qsub(i <= a->d ? a->c[i] : qi(0), i <= b->d ? b->c[i] :
		    qi(0));
	t.d = d;
	up_trim(&t);
	up_set(r, &t);
}

void
up_mul(struct up *r, const struct up *a, const struct up *b)
{
	struct up t;
	int i, j;

	if (a->d < 0 || b->d < 0) {
		r->d = -1;
		return;
	}
	if (a->d + b->d > UD) {
		alg_ovf = 1;
		r->d = -1;
		return;
	}
	t.d = a->d + b->d;
	for (i = 0; i <= t.d; i++)
		t.c[i] = qi(0);
	for (i = 0; i <= a->d; i++)
		if (a->c[i].n)
			for (j = 0; j <= b->d; j++)
				t.c[i + j] = qadd(t.c[i + j], qmul(a->c[i], b->c[j]));
	up_trim(&t);
	up_set(r, &t);
}

void
up_scale(struct up *r, const struct up *a, aq c)
{
	int i;

	r->d = a->d;
	for (i = 0; i <= a->d; i++)
		r->c[i] = qmul(a->c[i], c);
	up_trim(r);
}

int
up_divmod(const struct up *a, const struct up *b, struct up *q, struct up *rem)
{
	struct up r, qq;
	int i, k;

	if (b->d < 0) {
		alg_ovf = 1;
		return 0;
	}
	up_set(&r, a);
	qq.d = a->d - b->d;
	for (i = 0; i <= (qq.d > 0 ? qq.d : 0); i++)
		qq.c[i] = qi(0);
	for (k = a->d - b->d; k >= 0; k--) {
		aq c = r.d == k + b->d ? qdiv(r.c[k + b->d], b->c[b->d]) : qi(0);
		qq.c[k] = c;
		if (c.n)
			for (i = 0; i <= b->d; i++)
				r.c[k + i] = qsub(r.c[k + i], qmul(c, b->c[i]));
		r.c[k + b->d] = qi(0);
		r.d = k + b->d - 1;
		up_trim(&r);
	}
	if (qq.d < 0)
		qq.d = -1;
	up_trim(&qq);
	if (q)
		up_set(q, &qq);
	if (rem)
		up_set(rem, &r);
	return !alg_ovf;
}

void
up_monic(struct up *r, const struct up *a)
{
	if (a->d < 0) {
		r->d = -1;
		return;
	}
	up_scale(r, a, qdiv(qi(1), a->c[a->d]));
}

void
up_gcd(struct up *g, const struct up *a, const struct up *b)
{
	struct up x, y, r;

	up_set(&x, a);
	up_set(&y, b);
	while (y.d >= 0 && !alg_ovf) {
		up_divmod(&x, &y, NULL, &r);
		up_set(&x, &y);
		up_monic(&y, &r);
	}
	up_monic(g, &x);
}

int
up_gcdex(const struct up *a, const struct up *b, struct up *s, struct up *t,
    struct up *g)
{
	struct up r0, r1, s0, s1, t0, t1, q, tmp, tmp2;
	aq lc;

	up_set(&r0, a);
	up_set(&r1, b);
	s0.d = 0;
	s0.c[0] = qi(1);
	s1.d = -1;
	t0.d = -1;
	t1.d = 0;
	t1.c[0] = qi(1);
	while (r1.d >= 0 && !alg_ovf) {
		up_divmod(&r0, &r1, &q, &tmp);
		up_set(&r0, &r1);
		up_set(&r1, &tmp);
		up_mul(&tmp2, &q, &s1);
		up_sub(&tmp, &s0, &tmp2);
		up_set(&s0, &s1);
		up_set(&s1, &tmp);
		up_mul(&tmp2, &q, &t1);
		up_sub(&tmp, &t0, &tmp2);
		up_set(&t0, &t1);
		up_set(&t1, &tmp);
	}
	if (r0.d < 0)
		return 0;
	lc = qdiv(qi(1), r0.c[r0.d]);
	up_scale(g, &r0, lc);
	up_scale(s, &s0, lc);
	up_scale(t, &t0, lc);
	return !alg_ovf;
}

void
up_diff(struct up *r, const struct up *a)
{
	struct up t;
	int i;

	t.d = a->d - 1;
	for (i = 1; i <= a->d; i++)
		t.c[i - 1] = qmul(a->c[i], qi(i));
	if (t.d < 0)
		t.d = -1;
	up_trim(&t);
	up_set(r, &t);
}

double
up_eval(const struct up *a, double x)
{
	double s = 0;
	int i;

	for (i = a->d; i >= 0; i--)
		s = s * x + qd(a->c[i]);
	return s;
}

aq
up_evalq(const struct up *a, aq x)
{
	aq s = qi(0);
	int i;

	for (i = a->d; i >= 0; i--)
		s = qadd(qmul(s, x), a->c[i]);
	return s;
}

/* Yun: a = c prod s[i]^(i+1), s[i] monic squarefree, pairwise coprime */
int
up_sqfree(const struct up *a, struct up *s, int max)
{
	struct up b, d, g, c, y, t;
	int i = 0, n = 0;

	up_monic(&b, a);
	up_diff(&d, &b);
	up_gcd(&g, &b, &d);
	up_divmod(&b, &g, &c, NULL);
	up_divmod(&d, &g, &y, NULL);
	up_diff(&t, &c);
	up_sub(&y, &y, &t);
	while (c.d > 0 && i < max && !alg_ovf) {
		struct up w;
		up_gcd(&w, &c, &y);
		up_set(&s[i], &w);
		if (w.d > 0)
			n = i + 1;
		up_divmod(&c, &w, &c, NULL);
		up_divmod(&y, &w, &y, NULL);
		up_diff(&t, &c);
		up_sub(&y, &y, &t);
		i++;
	}
	return n;
}

/* the primitive integer coefficients of a (lc > 0) into A; 0 on overflow */
static int
up_int(const struct up *a, int64_t *A)
{
	int64_t l = 1, g = 0;
	int i;

	for (i = 0; i <= a->d; i++)
		l = lcm64(l, a->c[i].d);
	for (i = 0; i <= a->d; i++) {
		i128 v = (i128)a->c[i].n * (l / a->c[i].d);
		if (v > INT64_MAX || v < -INT64_MAX)
			return 0;
		A[i] = (int64_t)v;
		g = gcd64(g, A[i]);
	}
	if (g == 0)
		return 0;
	if (A[a->d] < 0)
		g = -g;
	for (i = 0; i <= a->d; i++)
		A[i] /= g;
	return !alg_ovf;
}

static int
divs(int64_t v, int64_t *out, int max)
{
	int64_t i;
	int n = 0, k, m;

	if (v < 0)
		v = -v;
	if (v == 0)
		return 0;
	for (i = 1; i * i <= v && i <= 3000000; i++)
		if (v % i == 0) {
			if (n >= max)
				return -1;
			out[n++] = i;
		}
	if (i <= 3000000 || i * i > v) {
		m = n;
		for (k = m - 1; k >= 0; k--) {
			int64_t o = v / out[k];
			if (o != out[k]) {
				if (n >= max)
					return -1;
				out[n++] = o;
			}
		}
	} else
		return -1;
	/* ascending */
	for (k = 1; k < n; k++) {
		int64_t t = out[k];
		int j = k;
		while (j > 0 && out[j - 1] > t) {
			out[j] = out[j - 1];
			j--;
		}
		out[j] = t;
	}
	return n;
}

/* the integer polynomial value, 0 on overflow (*ok = 0) */
static int64_t
ipeval(const int64_t *A, int d, int64_t x, int *ok)
{
	i128 s = 0;
	int i;

	for (i = d; i >= 0; i--) {
		s = s * x + A[i];
		if (s > ((i128)1 << 100) || s < -((i128)1 << 100)) {
			*ok = 0;
			return 0;
		}
	}
	if (s > INT64_MAX || s < -INT64_MAX) {
		*ok = 0;
		return 0;
	}
	return (int64_t)s;
}

/* a monic degree-k factor of g by Kronecker's method; 1 found, 0 none, -1 too costly */
static int
kronecker(const struct up *g, int k, struct up *h)
{
	int64_t A[UD + 1], pts[16], val[16], dv[16][512];
	int nd[16], d = g->d, np = 0, i, j, ok = 1;
	long long combos = 1, it;
	int idx[16], saved_ovf;

	if (!up_int(g, A))
		return -1;
	/* points 0, 1, -1, 2, -2, ... where g != 0, few divisors first */
	{
		int64_t cand[40], cv[40];
		int nc = 0, cnd[40];
		for (i = 0; nc < 2 * k + 6 && i < 80; i++) {
			int64_t x = (i & 1) ? (i + 1) / 2 : -(i / 2);
			int64_t v = ipeval(A, d, x, &ok), tmp[512];
			int c;
			if (!ok)
				return -1;
			if (v == 0)
				continue;
			c = divs(v, tmp, 512);
			if (c < 0)
				continue;
			cand[nc] = x;
			cv[nc] = v;
			cnd[nc++] = c;
		}
		if (nc < k + 1)
			return -1;
		/* choose the k+1 with the fewest divisors */
		for (j = 0; j <= k; j++) {
			int best = -1;
			for (i = 0; i < nc; i++)
				if (cnd[i] >= 0 && (best < 0 || cnd[i] < cnd[best]))
					best = i;
			pts[np] = cand[best];
			val[np] = cv[best];
			nd[np] = divs(cv[best], dv[np], 512);
			cnd[best] = -1;
			np++;
		}
	}
	for (i = 0; i < np; i++) {
		combos *= (i == 0 ? 1 : 2) * nd[i];
		if (combos > 400000)
			return -1;
	}
	memset(idx, 0, sizeof idx);
	saved_ovf = alg_ovf;
	for (it = 0; it < combos; it++) {
		aq y[16], dd[16];
		struct up p;
		int64_t c = it;
		/* decode the combination */
		for (i = 0; i < np; i++) {
			int base = (i == 0 ? 1 : 2) * nd[i];
			int r = (int)(c % base);
			int64_t dvv;
			c /= base;
			dvv = dv[i][r % nd[i]];
			if (i > 0 && r >= nd[i])
				dvv = -dvv;
			y[i] = qi(dvv);
		}
		/* Newton interpolation */
		for (i = 0; i < np; i++)
			dd[i] = y[i];
		for (j = 1; j < np; j++)
			for (i = np - 1; i >= j; i--)
				dd[i] = qdiv(qsub(dd[i], dd[i - 1]), qi(pts[i] - pts[i - j]));
		p.d = 0;
		p.c[0] = dd[np - 1];
		for (i = np - 2; i >= 0; i--) {
			struct up lin, t;
			lin.d = 1;
			lin.c[1] = qi(1);
			lin.c[0] = qi(-pts[i]);
			up_mul(&t, &p, &lin);
			t.c[0] = qadd(t.c[0], dd[i]);
			if (t.d < 0) {
				t.d = 0;
			}
			up_trim(&t);
			up_set(&p, &t);
		}
		if (alg_ovf) {
			alg_ovf = saved_ovf;
			continue;
		}
		if (p.d != k)
			continue;
		for (i = 0; i <= p.d; i++)
			if (p.c[i].d != 1)
				break;
		if (i <= p.d)
			continue;
		if (A[d] % p.c[k].n)
			continue;
		{
			struct up rem, q;
			up_divmod(g, &p, &q, &rem);
			if (rem.d < 0 && !alg_ovf) {
				up_monic(h, &p);
				return 1;
			}
			alg_ovf = saved_ovf;
		}
	}
	alg_ovf = saved_ovf;
	return 0;
}

/* g monic squarefree: its irreducible factors */
static int
factor_sqf(const struct up *g0, struct up *f, int max, int *partial)
{
	struct up g;
	int64_t A[UD + 1], dp[512], dq[512];
	int n = 0, np, nq, i, j, s, k;

	up_set(&g, g0);
	if (g.d <= 1) {
		if (g.d == 1 && n < max)
			up_set(&f[n++], &g);
		return n;
	}
	/* x as a factor */
	if (g.c[0].n == 0) {
		struct up x;
		x.d = 1;
		x.c[1] = qi(1);
		x.c[0] = qi(0);
		up_set(&f[n++], &x);
		up_divmod(&g, &x, &g, NULL);
	}
	if (g.d >= 1 && up_int(&g, A)) {
		np = divs(A[0], dp, 512);
		nq = divs(A[g.d], dq, 512);
		if (np < 0 || nq < 0)
			*partial = 1;
		for (i = 0; i < np && g.d >= 1; i++)
			for (j = 0; j < nq && g.d >= 1; j++)
				for (s = -1; s <= 1 && g.d >= 1; s += 2) {
					aq r;
					if (gcd64(dp[i], dq[j]) != 1)
						continue;
					int so = alg_ovf;
					r = qmk(s * dp[i], dq[j]);
					if (up_evalq(&g, r).n != 0 || alg_ovf) {
						alg_ovf = so;
						continue;
					}
					{
						struct up lin;
						lin.d = 1;
						lin.c[1] = qi(1);
						lin.c[0] = qneg(r);
						if (n < max)
							up_set(&f[n++], &lin);
						up_divmod(&g, &lin, &g, NULL);
					}
				}
	} else if (g.d >= 1)
		*partial = 1;
	if (g.d <= 0)
		return n;
	if (g.d <= 3) {
		if (n < max)
			up_set(&f[n++], &g);
		return n;
	}
	for (k = 2; k <= g.d / 2; k++) {
		struct up h, q;
		int r = kronecker(&g, k, &h);
		if (r < 0) {
			*partial = 1;
			continue;
		}
		if (r == 1) {
			up_divmod(&g, &h, &q, NULL);
			n += factor_sqf(&h, f + n, max - n, partial);
			n += factor_sqf(&q, f + n, max - n, partial);
			return n;
		}
	}
	if (n < max)
		up_set(&f[n++], &g);
	return n;
}

int
up_factor(const struct up *a, struct up *f, int *m, int max, int *partial)
{
	struct up s[UD];
	int ns, i, n = 0, k, j;

	*partial = 0;
	if (a->d <= 0)
		return 0;
	ns = up_sqfree(a, s, UD);
	for (i = 0; i < ns; i++) {
		if (s[i].d <= 0)
			continue;
		k = factor_sqf(&s[i], f + n, max - n, partial);
		for (j = 0; j < k; j++)
			m[n + j] = i + 1;
		n += k;
	}
	return n;
}

/* ---- multivariate factorisation ---- */

static int
fadd(struct mp g, int mult, struct mp *f, int *m, int n, int max)
{
	int i;

	if (mp_isconst(g, NULL))
		return n;
	g = mp_prim(g);
	for (i = 0; i < n; i++)
		if (mp_eq(f[i], g)) {
			m[i] += mult;
			return n;
		}
	if (n < max) {
		f[n] = g;
		m[n++] = mult;
	}
	return n;
}

static int
is_homog(struct mp a)
{
	int i, k, d = -1;

	for (i = 0; i < a.n; i++) {
		int s = 0;
		for (k = 0; k < AV; k++)
			s += a.t[i].e[k];
		if (d >= 0 && s != d)
			return 0;
		d = s;
	}
	return 1;
}

/* a primitive and squarefree, no monomial or variable content */
static int
factor_prim(struct mp a, int mult, struct mp *f, int *m, int n, int max,
    int *partial)
{
	uint32_t vs = mp_vars(a);
	int nv = __builtin_popcount(vs), v, w, i, k;
	struct up u, uf[UD];
	int um[UD], up_n, pp = 0;

	if (nv == 0)
		return n;
	if (nv == 1) {
		v = topvar(vs);
		if (!up_of_mp(a, v, &u))
			return fadd(a, mult, f, m, n, max);
		up_n = up_factor(&u, uf, um, UD, &pp);
		if (pp)
			*partial = 1;
		for (i = 0; i < up_n; i++)
			n = fadd(mp_of_up(&uf[i], v), mult * um[i], f, m, n, max);
		return n;
	}
	if (nv == 2 && is_homog(a)) {
		int td = mp_tdeg(a), dsum = 0;
		struct mp one = mp_const(qi(1)), b;
		v = topvar(vs);
		w = topvar(vs & ~(1u << v));
		b = mp_subst(a, w, one);
		if (!up_of_mp(b, v, &u))
			return fadd(a, mult, f, m, n, max);
		up_n = up_factor(&u, uf, um, UD, &pp);
		if (pp)
			*partial = 1;
		for (i = 0; i < up_n; i++) {
			/* homogenise: sum c_j v^j w^(k - j) */
			struct mp h = mp_zero();
			int kk = uf[i].d;
			for (k = 0; k <= kk; k++)
				if (uf[i].c[k].n) {
					struct mp t = mp_const(uf[i].c[k]);
					t.t[0].e[v] = (uint8_t)k;
					t.t[0].e[w] = (uint8_t)(kk - k);
					h = mp_add(h, t);
				}
			dsum += kk * um[i];
			n = fadd(h, mult * um[i], f, m, n, max);
		}
		if (td > dsum)
			n = fadd(mp_atom(w), mult * (td - dsum), f, m, n, max);
		return n;
	}
	/* linear in some variable: irreducible (it is primitive) */
	for (k = 0; k < AV; k++)
		if ((vs & (1u << k)) && mp_deg(a, k) == 1)
			return fadd(a, mult, f, m, n, max);
	*partial = 1;
	return fadd(a, mult, f, m, n, max);
}

static int
factor_rec(struct mp a, int mult, struct mp *f, int *m, int n, int max,
    int *partial, int depth)
{
	uint32_t vs;
	int k, i;

	if (alg_ovf || depth > 12)
		return fadd(a, mult, f, m, n, max);
	a = mp_prim(a);
	vs = mp_vars(a);
	if (vs == 0)
		return n;
	/* monomial content */
	for (k = 0; k < AV; k++) {
		int mn = 255;
		if (!(vs & (1u << k)))
			continue;
		for (i = 0; i < a.n; i++)
			if (a.t[i].e[k] < mn)
				mn = a.t[i].e[k];
		if (mn > 0) {
			struct mp q, mo = mp_const(qi(1));
			mo.t[0].e[k] = (uint8_t)mn;
			if (mp_divexact(a, mo, &q))
				a = q;
			n = fadd(mp_atom(k), mult * mn, f, m, n, max);
		}
	}
	vs = mp_vars(a);
	if (__builtin_popcount(vs) >= 2) {
		/* the content in each variable */
		for (k = 0; k < AV; k++) {
			struct mp c, q;
			if (!(vs & (1u << k)))
				continue;
			c = mp_content(a, k);
			if (!mp_isconst(c, NULL) && mp_divexact(a, c, &q)) {
				n = factor_rec(c, mult, f, m, n, max, partial, depth + 1);
				return factor_rec(q, mult, f, m, n, max, partial,
				    depth + 1);
			}
		}
		/* square-free by Yun in the top variable */
		{
			int v = topvar(vs), j = 1;
			struct mp b = a, db = mp_diff(a, v), g = mp_gcd(b, db), c, d,
			    y, q;
			if (!mp_isconst(g, NULL)) {
				if (!mp_divexact(b, g, &c) || !mp_divexact(db, g, &d))
					return fadd(a, mult, f, m, n, max);
				d = mp_sub(d, mp_diff(c, v));
				while (!mp_isconst(c, NULL) && !alg_ovf && j < 64) {
					y = mp_gcd(c, d);
					n = factor_prim(y, mult * j, f, m, n, max, partial);
					if (!mp_divexact(c, y, &q))
						break;
					c = q;
					if (!mp_divexact(d, y, &q))
						break;
					d = mp_sub(q, mp_diff(c, v));
					j++;
				}
				return n;
			}
		}
		return factor_prim(a, mult, f, m, n, max, partial);
	}
	return factor_prim(a, mult, f, m, n, max, partial);
}

static int
fsort_key(struct mp a, double *k2)
{
	aq c;
	struct mp t = a;
	int i;

	*k2 = 0;
	for (i = 0; i < t.n; i++) {
		int k, z = 1;
		for (k = 0; k < AV; k++)
			if (t.t[i].e[k])
				z = 0;
		if (z) {
			c = t.t[i].c;
			*k2 = qd(c);
		}
	}
	return mp_tdeg(a);
}

int
mp_factor(struct mp a, aq *c, struct mp *f, int *m, int max, int *partial)
{
	int n, i, j;
	struct mp prod, q;
	aq cc;

	*partial = 0;
	if (mp_isconst(a, c))
		return 0;
	n = factor_rec(a, 1, f, m, 0, max, partial, 0);
	/* the constant: a / prod */
	prod = mp_const(qi(1));
	for (i = 0; i < n; i++)
		prod = mp_mul(prod, mp_pow(f[i], m[i]));
	if (!mp_divexact(a, prod, &q) || !mp_isconst(q, &cc)) {
		/* should not happen: give the input back whole */
		f[0] = mp_prim(a);
		m[0] = 1;
		*c = mp_qcontent(a);
		*partial = 1;
		return 1;
	}
	*c = cc;
	/* by degree, then constant term */
	for (i = 1; i < n; i++)
		for (j = i; j > 0; j--) {
			double k1, k2;
			int d1 = fsort_key(f[j - 1], &k1), d2 = fsort_key(f[j], &k2);
			if (d1 < d2 || (d1 == d2 && k1 <= k2))
				break;
			{
				struct mp t = f[j];
				int tm = m[j];
				f[j] = f[j - 1];
				m[j] = m[j - 1];
				f[j - 1] = t;
				m[j - 1] = tm;
			}
		}
	return n;
}

/* ---- rational functions ---- */

struct rf
rf_poly(struct mp a)
{
	struct rf r;

	r.n = a;
	r.d = mp_const(qi(1));
	return r;
}

struct rf
rf_norm(struct rf a)
{
	struct mp g, q;
	aq c;

	if (a.n.n == 0) {
		a.d = mp_const(qi(1));
		return a;
	}
	if (!mp_isconst(a.d, NULL)) {
		g = mp_gcd(a.n, a.d);
		if (!mp_isconst(g, NULL)) {
			if (mp_divexact(a.n, g, &q))
				a.n = q;
			if (mp_divexact(a.d, g, &q))
				a.d = q;
		}
	}
	c = mp_qcontent(a.d);
	a.n = mp_scale(a.n, qdiv(qi(1), c));
	a.d = mp_scale(a.d, qdiv(qi(1), c));
	return a;
}

struct rf
rf_add(struct rf a, struct rf b)
{
	struct rf r;
	struct mp g = mp_gcd(a.d, b.d), ad, bd;

	if (!mp_divexact(a.d, g, &ad) || !mp_divexact(b.d, g, &bd)) {
		ad = a.d;
		bd = b.d;
		g = mp_const(qi(1));
	}
	r.n = mp_add(mp_mul(a.n, bd), mp_mul(b.n, ad));
	r.d = mp_mul(mp_mul(ad, bd), g);
	return rf_norm(r);
}

struct rf
rf_sub(struct rf a, struct rf b)
{
	b.n = mp_neg(b.n);
	return rf_add(a, b);
}

struct rf
rf_mul(struct rf a, struct rf b)
{
	struct rf r;

	r.n = mp_mul(a.n, b.n);
	r.d = mp_mul(a.d, b.d);
	return rf_norm(r);
}

struct rf
rf_div(struct rf a, struct rf b)
{
	struct rf r;

	if (b.n.n == 0) {
		alg_ovf = 1;
		return a;
	}
	r.n = mp_mul(a.n, b.d);
	r.d = mp_mul(a.d, b.n);
	return rf_norm(r);
}

struct rf
rf_pow(struct rf a, int n)
{
	struct rf r;

	if (n < 0) {
		r.n = a.d;
		r.d = a.n;
		n = -n;
		if (a.n.n == 0) {
			alg_ovf = 1;
			return a;
		}
	} else
		r = a;
	r.n = mp_pow(r.n, n);
	r.d = mp_pow(r.d, n);
	return rf_norm(r);
}

/* ---- node -> rational function ---- */

static int
conv(nid e, struct rf *r, struct mp *dall, int depth)
{
	double v;
	struct rf a, b;
	int k;

	if (depth > 200 || alg_ovf)
		return 0;
	switch (N.op[e]) {
	case O_CONST:
		num_of(e, &v);
		{
			aq c;
			if (qexact(v, &c)) {
				*r = rf_poly(mp_const(c));
				return 1;
			}
		}
		goto atom;
	case O_ADD:
	case O_SUB:
		if (!TY_FP(N.ty[e]) && TYK(N.ty[e]) != K_INT && TYK(N.ty[e]) !=
		    K_LIT)
			goto atom;
		if (!conv(N.a[e], &a, dall, depth + 1) ||
		    !conv(N.b[e], &b, dall, depth + 1))
			return 0;
		*r = N.op[e] == O_ADD ? rf_add(a, b) : rf_sub(a, b);
		return !alg_ovf;
	case O_NEG:
		if (!conv(N.a[e], &a, dall, depth + 1))
			return 0;
		a.n = mp_neg(a.n);
		*r = a;
		return 1;
	case O_MUL:
		if (!TY_FP(N.ty[e]) && TYK(N.ty[e]) != K_INT && TYK(N.ty[e]) !=
		    K_LIT)
			goto atom;
		if (!conv(N.a[e], &a, dall, depth + 1) ||
		    !conv(N.b[e], &b, dall, depth + 1))
			return 0;
		*r = rf_mul(a, b);
		return !alg_ovf;
	case O_DIV:
		if (!TY_FP(N.ty[e]))
			goto atom;
		if (!conv(N.a[e], &a, dall, depth + 1) ||
		    !conv(N.b[e], &b, dall, depth + 1))
			return 0;
		if (b.n.n == 0)
			return 0;
		if (dall && !mp_isconst(b.n, NULL))
			*dall = mp_mul(*dall, b.n);
		*r = rf_div(a, b);
		return !alg_ovf;
	case O_CAST:
		if (TY_FP(N.ty[e]) && TY_FP(N.ty[N.a[e]]))
			return conv(N.a[e], r, dall, depth + 1);
		if (TY_FP(N.ty[e]) && is_const(N.a[e])) {
			aq c;
			num_of(N.a[e], &v);
			if (qexact(v, &c)) {
				*r = rf_poly(mp_const(c));
				return 1;
			}
		}
		goto atom;
	case O_POW:
		if (num_of(N.b[e], &v) && v == floor(v) && fabs(v) <= 64 &&
		    !(num_of(N.a[e], &v) && v == M_E)) {
			num_of(N.b[e], &v);
			k = (int)v;
			if (!conv(N.a[e], &a, dall, depth + 1))
				return 0;
			if (k < 0) {
				if (a.n.n == 0)
					return 0;
				if (dall && !mp_isconst(a.n, NULL))
					*dall = mp_mul(*dall, mp_pow(a.n, -k));
			}
			*r = rf_pow(a, k);
			return !alg_ovf;
		}
		/* u^(k + p/q) = u^k u^(p/q) */
		if (num_of(N.b[e], &v) && fabs(v) < 64 && v != floor(v)) {
			aq ex;
			if (qexact(v, &ex) && ex.d <= 12) {
				int64_t kk = ex.n / ex.d;
				aq fr;
				nid at;
				if (ex.n < 0 && ex.n % ex.d)
					kk--;
				fr = qsub(ex, qi(kk));
				if (kk != 0) {
					at = fr.n * 2 == fr.d ? mk_un(O_SQRT, N.a[e]) :
					    POW(N.a[e], q_cnode(fr));
					if (!conv(N.a[e], &a, dall, depth + 1))
						return 0;
					if (kk < 0) {
						if (a.n.n == 0)
							return 0;
						if (dall && !mp_isconst(a.n, NULL))
							*dall = mp_mul(*dall, mp_pow(a.n, (int)-kk));
					}
					a = rf_pow(a, (int)kk);
					k = atom_of(at);
					if (k < 0)
						return 0;
					*r = rf_mul(a, rf_poly(mp_atom(k)));
					return !alg_ovf;
				}
				if (fr.n * 2 == fr.d)
					e = mk_un(O_SQRT, N.a[e]);
			}
		}
		goto atom;
	}
atom:
	if (!TY_FP(N.ty[e]) && TYK(N.ty[e]) != K_INT && TYK(N.ty[e]) != K_LIT)
		return 0;
	k = atom_of(e);
	if (k < 0)
		return 0;
	*r = rf_poly(mp_atom(k));
	return 1;
}

int
rf_of(nid e, struct rf *r, struct mp *dall)
{
	alg_ovf = 0;
	if (dall)
		*dall = mp_const(qi(1));
	if (!conv(e, r, dall, 0) || alg_ovf)
		return 0;
	return 1;
}

int
mp_of(nid e, struct mp *r)
{
	struct rf f;
	aq c;

	if (!rf_of(e, &f, NULL) || !mp_isconst(f.d, &c))
		return 0;
	*r = mp_scale(f.n, qdiv(qi(1), c));
	return !alg_ovf;
}

/* ---- printing ---- */

nid
q_cnode(aq c)
{
	nid r;

	if (c.d == 1)
		return KR((double)c.n);
	r = mk_raw(O_DIV, TY_REAL, KR((double)(c.n < 0 ? -c.n : c.n)),
	    KR((double)c.d), 0, 0, 0);
	return c.n < 0 ? mk_un(O_NEG, r) : r;
}

nid
sqrt_cnode(int64_t m)
{
	return mk_raw(O_SQRT, TY_REAL, KR((double)m), 0, 0, 0, 0);
}

/* k * x, as 3x*sin(x) rather than 3(x*sin(x)) */
static nid
kmul(double k, nid x)
{
	if (N.op[x] == O_MUL && !is_const(N.a[x]))
		return MUL(kmul(k, N.a[x]), N.b[x]);
	return MUL(KR(k), x);
}

nid
qmul_node(aq c, nid x)
{
	int neg = c.n < 0;
	nid r;

	if (c.n == 0)
		return KR(0);
	if (neg)
		c.n = -c.n;
	if (c.n == 1 && c.d == 1)
		r = x;
	else if (c.d == 1)
		r = kmul((double)c.n, x);
	else
		r = DIV(c.n == 1 ? x : kmul((double)c.n, x), KR((double)c.d));
	return neg ? mk_un(O_NEG, r) : r;
}

/* the monomial with |coefficient| */
static nid
mono_node(const struct mono *m)
{
	nid r = 0;
	aq c = m->c;
	int k;

	if (c.n < 0)
		c.n = -c.n;
	if (c.n != 1)
		r = KR((double)c.n);
	for (k = 0; k < AV; k++)
		if (m->e[k]) {
			nid f = m->e[k] == 1 ? at_node[k] : POW(at_node[k],
			    KI(m->e[k]));
			r = r ? MUL(r, f) : f;
		}
	if (r == 0)
		return q_cnode((aq){1, c.d});
	if (c.d != 1)
		r = DIV(r, KR((double)c.d));
	return r;
}

nid
mp_node(struct mp a)
{
	nid r = 0;
	int i;

	for (i = 0; i < a.n; i++) {
		nid t = mono_node(&a.t[i]);
		if (r == 0)
			r = a.t[i].c.n < 0 ? mk_un(O_NEG, t) : t;
		else
			r = a.t[i].c.n < 0 ? SUB(r, t) : ADD(r, t);
	}
	return r ? r : KR(0);
}

nid
mp_node_in(struct mp a, int v)
{
	int d = mp_deg(a, v), k;
	nid r = 0;

	for (k = d; k >= 0; k--) {
		struct mp c = mp_coef(a, v, k), t;
		nid tn;
		int neg = 0;
		if (c.n == 0)
			continue;
		if (c.n == 1 || k == 0) {
			t = c;
			if (k) {
				t = mp_copy(c);
				t.t[0].e[v] = (uint8_t)k;
			}
			tn = mp_node(t);
			if (c.n == 1 && c.t[0].c.n < 0) {
				t.t[0].c = qneg(t.t[0].c);
				tn = mp_node(t);
				neg = 1;
			}
		} else {
			aq cc = mp_qcontent(c);
			nid pw = k == 1 ? at_node[v] : POW(at_node[v], KI(k));
			if (cc.n < 0) {
				neg = 1;
				cc.n = -cc.n;
			}
			tn = MUL(mp_node(mp_prim(c)), pw);
			tn = qmul_node(cc, tn);
		}
		if (r == 0)
			r = neg ? mk_un(O_NEG, tn) : tn;
		else
			r = neg ? SUB(r, tn) : ADD(r, tn);
	}
	return r ? r : KR(0);
}

/* c * prod f^m, left associated so 2(x - 1)*(x + 1) prints that way */
static nid
fprod_node(aq c, const struct mp *f, const int *m, int n)
{
	nid r = 0;
	int i, neg = c.n < 0;
	aq ac = c;

	if (neg)
		ac.n = -ac.n;
	if (ac.n != 1)
		r = KR((double)ac.n);
	for (i = 0; i < n; i++) {
		nid t = mp_node(f[i]);
		if (m[i] > 1)
			t = POW(t, KI(m[i]));
		r = r ? MUL(r, t) : t;
	}
	if (r == 0)
		r = KR(1);
	if (ac.d != 1)
		r = DIV(r, KR((double)ac.d));
	return neg ? mk_un(O_NEG, r) : r;
}

nid
mp_fnode(struct mp a)
{
	struct mp f[32];
	int m[32], n, partial;
	aq c;
	size_t mk = ar_mark();
	nid r;

	if (mp_isconst(a, &c))
		return q_cnode(c);
	n = mp_factor(a, &c, f, m, 32, &partial);
	r = alg_ovf ? mp_node(a) : fprod_node(c, f, m, n);
	ar_release(mk);
	return r;
}

nid
rf_node(struct rf a, int factored)
{
	aq cn, cd;
	nid num, den;
	struct mp n, d;

	if (mp_isconst(a.d, &cd))
		return factored == 1 ? mp_fnode(mp_scale(a.n, qdiv(qi(1), cd))) :
		    mp_node(mp_scale(a.n, qdiv(qi(1), cd)));
	cn = mp_iszero(a.n) ? qi(1) : mp_qcontent(a.n);
	cd = mp_qcontent(a.d);
	n = mp_scale(a.n, qdiv(qi(1), cn));
	d = mp_scale(a.d, qdiv(qi(1), cd));
	cn = qdiv(cn, cd);     /* the value is cn * n / d */
	if (factored) {
		num = mp_isconst(n, NULL) ? 0 : factored == 2 ? mp_node(n) :
		    mp_fnode(n);
		den = mp_fnode(d);
	} else {
		num = mp_isconst(n, NULL) ? 0 : mp_node(n);
		den = mp_node(d);
	}
	{
		int neg = cn.n < 0;
		if (neg)
			cn.n = -cn.n;
		if (num == 0)
			num = KR((double)cn.n);
		else if (cn.n != 1)
			num = MUL(KR((double)cn.n), num);
		if (cn.d != 1)
			den = MUL(KR((double)cn.d), den);
		num = DIV(num, den);
		return neg ? mk_un(O_NEG, num) : num;
	}
}

const char *
mp_str(struct mp a)
{
	return node_str(mp_node(a));
}

void
rf_conds(struct mp dall, struct mp d, struct sbuf *out)
{
	struct mp f[32], q;
	int m[32], n, i, partial;
	aq c;

	if (mp_isconst(dall, NULL))
		return;
	n = mp_factor(dall, &c, f, m, 32, &partial);
	for (i = 0; i < n; i++)
		if (!mp_divexact(d, f[i], &q)) {
			if (out->n)
				sb_puts(out, ", ");
			sb_printf(out, "%s != 0", mp_str(f[i]));
		}
}

nid
up_node(const struct up *a, nid x)
{
	nid r = 0;
	int k;

	for (k = a->d; k >= 0; k--) {
		aq c = a->c[k];
		nid t;
		int neg = c.n < 0;
		if (c.n == 0)
			continue;
		if (neg)
			c.n = -c.n;
		if (k == 0)
			t = q_cnode(c);
		else
			t = qmul_node(c, k == 1 ? x : POW(x, KI(k)));
		if (r == 0)
			r = neg ? mk_un(O_NEG, t) : t;
		else
			r = neg ? SUB(r, t) : ADD(r, t);
	}
	return r ? r : KR(0);
}

nid
up_pnode(const struct up *a, nid x)
{
	int64_t A[UD + 1];
	struct up p;
	int i;

	if (a->d < 0 || !up_int(a, A))
		return up_node(a, x);
	p.d = a->d;
	for (i = 0; i <= a->d; i++)
		p.c[i] = qi(A[i]);
	return up_node(&p, x);
}

const char *
up_str(const struct up *a, nid x)
{
	return node_str(up_node(a, x));
}

/* ---- Q(sqrt d) ---- */

int64_t qs_d = 2;

struct qs
qs_add(struct qs x, struct qs y)
{
	return (struct qs){qadd(x.a, y.a), qadd(x.b, y.b)};
}

struct qs
qs_sub(struct qs x, struct qs y)
{
	return (struct qs){qsub(x.a, y.a), qsub(x.b, y.b)};
}

struct qs
qs_mul(struct qs x, struct qs y)
{
	return (struct qs){qadd(qmul(x.a, y.a), qmul(qi(qs_d), qmul(x.b, y.b))),
	    qadd(qmul(x.a, y.b), qmul(x.b, y.a))};
}

struct qs
qs_div(struct qs x, struct qs y)
{
	aq n = qsub(qmul(y.a, y.a), qmul(qi(qs_d), qmul(y.b, y.b)));
	struct qs c = {qdiv(y.a, n), qneg(qdiv(y.b, n))};

	return qs_mul(x, c);
}

double
qs_val(struct qs x)
{
	return qd(x.a) + qd(x.b) * sqrt((double)qs_d);
}

nid
qs_node(struct qs x)
{
	nid b;

	if (x.b.n == 0)
		return q_cnode(x.a);
	b = qmul_node(x.b, sqrt_cnode(qs_d));
	if (x.a.n == 0)
		return b;
	if (x.b.n < 0)
		return SUB(q_cnode(x.a), qmul_node(qneg(x.b), sqrt_cnode(qs_d)));
	return ADD(q_cnode(x.a), b);
}

static int
isqrt_exact(int64_t v, int64_t *r)
{
	int64_t s;

	if (v < 0)
		return 0;
	s = (int64_t)floor(sqrt((double)v) + 0.5);
	while (s > 0 && s * s > v)
		s--;
	while ((s + 1) * (s + 1) <= v)
		s++;
	if (s * s != v)
		return 0;
	*r = s;
	return 1;
}

static int
qsqrt_exact(aq v, aq *r)
{
	int64_t a, b;

	if (!isqrt_exact(v.n, &a) || !isqrt_exact(v.d, &b))
		return 0;
	*r = qmk(a, b);
	return 1;
}

/* sqrt(v) = k sqrt(m), m squarefree positive integer, v >= 0 */
void
sqrt_split(aq v, aq *k, int64_t *m)
{
	i128 x = (i128)v.n * v.d;   /* sqrt(n/d) = sqrt(n d)/d */
	int64_t out = 1, mm = 1, p;

	if (v.n <= 0) {
		*k = qi(0);
		*m = 1;
		return;
	}
	if (x > INT64_MAX) {
		alg_ovf = 1;
		*k = qi(0);
		*m = 1;
		return;
	}
	{
		int64_t y = (int64_t)x;
		for (p = 2; p * p <= y && p < 200000; p++)
			while (y % (p * p) == 0) {
				y /= p * p;
				out *= p;
			}
		mm = y;
	}
	*k = qmk(out, v.d);
	*m = mm;
}

/* (u + v sqrt d)^2 = a + b sqrt d */
int
qs_sqrt(struct qs x, struct qs *r)
{
	aq a = x.a, b = x.b, t, u2, u, v, D;
	int s;

	if (b.n == 0) {
		if (qsgn(a) < 0)
			return 0;
		if (qsqrt_exact(a, &u)) {
			*r = (struct qs){u, qi(0)};
			return 1;
		}
		/* a = d v^2 */
		if (qsqrt_exact(qdiv(a, qi(qs_d)), &v)) {
			*r = (struct qs){qi(0), v};
			return 1;
		}
		return 0;
	}
	D = qsub(qmul(a, a), qmul(qi(qs_d), qmul(b, b)));
	if (!qsqrt_exact(D, &t))
		return 0;
	for (s = 1; s >= -1; s -= 2) {
		u2 = qdiv(qadd(a, s > 0 ? t : qneg(t)), qi(2));
		if (qsgn(u2) <= 0 || !qsqrt_exact(u2, &u))
			continue;
		v = qdiv(b, qmul(qi(2), u));
		*r = (struct qs){u, v};
		if (qs_val(*r) < 0)
			*r = (struct qs){qneg(u), qneg(v)};
		return !alg_ovf;
	}
	return 0;
}
