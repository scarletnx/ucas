/*
 * ecq.c - elliptic curves over Q: integral and minimal models, Tate's
 * algorithm, the conductor, a_p, torsion (reduction bound + Nagell-Lutz),
 * heights (naive and canonical: Tate's series at infinity, Silverman's
 * local formulas at the bad primes), point search, 2-isogeny descent
 * with local solubility of quartics, independence and saturation.
 */
#include "ecq.h"
#include "ucas.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static struct ecf QF;
static int qf_init;

static const struct ecf *
QQ(void)
{
	if (!qf_init) {
		ecf_q(&QF);
		qf_init = 1;
	}
	return &QF;
}

zint
fe_qz(const fe *a)
{
	struct zq q = fe_q(a);

	if (!z_is_one(q.d))
		fail("internal: %s is not an integer", zq_str(q));
	return q.n;
}

fe
fe_from_z(zint a)
{
	return fe_from_q(zq_int(a));
}

#define ZA(E, f) fe_qz(&(E)->f)
#define I(v) z_from_i64(v)

static void
ec_from_z(struct ec *E, zint a1, zint a2, zint a3, zint a4, zint a6)
{
	fe a[5];

	a[0] = fe_from_z(a1);
	a[1] = fe_from_z(a2);
	a[2] = fe_from_z(a3);
	a[3] = fe_from_z(a4);
	a[4] = fe_from_z(a6);
	ec_init(E, QQ(), a);
}

static void
iso_rst(struct eciso *m, zint r, zint s, zint t)
{
	m->u = fe_from_z(Z_ONE);
	m->r = fe_from_z(r);
	m->s = fe_from_z(s);
	m->t = fe_from_z(t);
}

int
ecq_is_integral(const struct ec *E)
{
	return z_is_one(fe_q(&E->a1).d) && z_is_one(fe_q(&E->a2).d) &&
	    z_is_one(fe_q(&E->a3).d) && z_is_one(fe_q(&E->a4).d) &&
	    z_is_one(fe_q(&E->a6).d);
}

void
ecq_integral(const struct ec *E, struct ec *Ei, struct eciso *m)
{
	zint k = Z_ONE;
	fe a[5];
	int i, w[5] = {1, 2, 3, 4, 6};

	ec_coeffs(E, a);
	/* the smallest k with a_i k^i integral: per prime of a denominator */
	for (i = 0; i < 5; i++) {
		zint d = fe_q(&a[i]).d, p[64];
		int e[64], n, j, comp;
		if (z_is_one(d))
			continue;
		n = z_factor(d, p, e, 64, &comp);
		for (j = 0; j < n; j++) {
			int need = (e[j] + w[i] - 1) / w[i], have = 0;
			zint kk = k;
			while (z_divides(p[j], kk)) {
				kk = z_divexact(kk, p[j]);
				have++;
			}
			if (need > have)
				k = z_mul(k, z_pow(p[j], (uint64_t)(need - have)));
		}
	}
	eciso_identity(QQ(), m);
	m->u = fe_from_q(zq_make(Z_ONE, k));
	eciso_apply(E, m, Ei);
}

/* ---- minimal models ---- */

int
ecq_from_c4c6(zint c4, zint c6, struct ec *E)
{
	zint b2, b4, b6, a1, a2, a3, a4, a6, t;

	b2 = z_mod(z_neg(c6), I(12));
	if (z_cmp(b2, I(6)) > 0)
		b2 = z_sub(b2, I(12));
	t = z_sub(z_mul(b2, b2), c4);
	if (!z_divides(I(24), t))
		return 0;
	b4 = z_divexact(t, I(24));
	t = z_sub(z_add(z_neg(z_pow(b2, 3)), z_mul(I(36), z_mul(b2, b4))), c6);
	if (!z_divides(I(216), t))
		return 0;
	b6 = z_divexact(t, I(216));
	a1 = z_mod(b2, I(2));
	a3 = z_mod(b6, I(2));
	t = z_sub(b2, a1);
	if (!z_divides(I(4), t))
		return 0;
	a2 = z_divexact(t, I(4));
	t = z_sub(b4, z_mul(a1, a3));
	if (!z_divides(I(2), t))
		return 0;
	a4 = z_divexact(t, I(2));
	t = z_sub(b6, a3);
	if (!z_divides(I(4), t))
		return 0;
	a6 = z_divexact(t, I(4));
	ec_from_z(E, a1, a2, a3, a4, a6);
	if (!z_eq(ZA(E, c4), c4) || !z_eq(ZA(E, c6), c6))
		return 0;
	return 1;
}

static int
zval(zint a, zint p)
{
	if (z_is_zero(a))
		return 1 << 20;
	return z_val(a, p, NULL);
}

void
ecq_minimal(const struct ec *E0, struct ec *Em, struct eciso *m)
{
	struct ec E;
	struct eciso mi, mf;
	zint c4, c6, D, p[64], u0 = Z_ONE, best = 0;
	int e[64], n, j, comp, d2max = 0, d3max = 0, d2, d3;
	struct ec cand, bestE;

	if (E0->singular)
		fail("the curve is singular (discriminant 0)");
	ecq_integral(E0, &E, &mi);
	c4 = ZA(&E, c4);
	c6 = ZA(&E, c6);
	D = ZA(&E, disc);
	n = z_factor(D, p, e, 64, &comp);
	if (!comp)
		note("the discriminant %s was not completely factored; the "
		    "model may not be minimal at its large factors", z_str(D));
	for (j = 0; j < n; j++) {
		int d = e[j] / 12, v4 = zval(c4, p[j]) / 4, v6 = zval(c6, p[j]) / 6;
		if (v4 < d)
			d = v4;
		if (v6 < d)
			d = v6;
		if (d <= 0)
			continue;
		if (z_eq(p[j], I(2)))
			d2max = d;
		else if (z_eq(p[j], I(3)))
			d3max = d;
		else
			u0 = z_mul(u0, z_pow(p[j], (uint64_t)d));
	}
	for (d2 = d2max; d2 >= 0; d2--)
		for (d3 = d3max; d3 >= 0; d3--) {
			zint u = z_mul(u0, z_mul(z_pow(I(2), (uint64_t)d2),
			    z_pow(I(3), (uint64_t)d3)));
			zint u4 = z_pow(u, 4), u6 = z_pow(u, 6);
			if (!ecq_from_c4c6(z_divexact(c4, u4), z_divexact(c6, u6),
			    &cand))
				continue;
			if (best == 0 || z_cmp(u, best) > 0) {
				best = u;
				bestE = cand;
			}
		}
	if (best == 0) {
		/* E itself as the fallback (cannot happen: u = 1 works for
		 * an integral model up to the reduction of a1, a2, a3) */
		if (!ecq_from_c4c6(c4, c6, &bestE))
			bestE = E;
	}
	*Em = bestE;
	if (!eciso_find(&E, Em, &mf))
		fail("internal: the minimal model is not isomorphic");
	eciso_compose(QQ(), m, &mi, &mf);
	/* certificate: E0 under m is Em */
	{
		struct ec T;
		eciso_apply(E0, m, &T);
		if (!fe_eq(QQ(), &T.a1, &Em->a1) || !fe_eq(QQ(), &T.a2, &Em->a2) ||
		    !fe_eq(QQ(), &T.a3, &Em->a3) || !fe_eq(QQ(), &T.a4, &Em->a4) ||
		    !fe_eq(QQ(), &T.a6, &Em->a6))
			fail("internal: minimal model check");
	}
}

/* ---- Tate's algorithm ---- */

static zint
zmodp(zint a, zint p)
{
	return z_mod(a, p);
}

/* roots of a X^2 + b X + c mod p: the number of distinct roots in F_p,
 * whether the discriminant vanishes (*dbl, with the root in *r) */
static int
quad_roots(zint a, zint b, zint c, zint p, int *dbl, zint *r)
{
	*dbl = 0;
	a = zmodp(a, p);
	b = zmodp(b, p);
	c = zmodp(c, p);
	if (z_eq(p, I(2))) {
		int n = 0, k, first = -1;
		for (k = 0; k < 2; k++) {
			zint v = z_add(z_mul(z_add(z_mul(a, I(k)), b), I(k)), c);
			if (z_mod_u64(v, 2) == 0) {
				n++;
				if (first < 0)
					first = k;
			}
		}
		if (z_is_zero(b)) {
			/* a X^2 + c = a (X + c/a)^2 in char 2: a double root */
			*dbl = 1;
			*r = z_is_zero(a) ? 0 : c;   /* a = 1 here */
			return 1;
		}
		return n;
	}
	{
		zint d = zmodp(z_sub(z_mul(b, b), z_mul(I(4), z_mul(a, c))), p);
		if (z_is_zero(d)) {
			*dbl = 1;
			/* -b / (2a) */
			*r = zmodp(z_mul(z_neg(b), z_invmod(z_mul(I(2), a), p)), p);
			return 1;
		}
		{
			zint e = z_div_floor(z_sub(p, Z_ONE), I(2));
			return z_is_one(z_powmod(d, e, p)) ? 2 : 0;
		}
	}
}


/* (a * b) mod f, f = T^3 + f[2] T^2 + f[1] T + f[0], degrees < 3 */
static void
pmulmod3(zint *r, const zint *a, const zint *b, const zint *f, zint p)
{
	zint t[5];
	int i, j;

	for (i = 0; i < 5; i++)
		t[i] = 0;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			t[i + j] = z_add(t[i + j], z_mul(a[i], b[j]));
	for (i = 4; i >= 3; i--) {
		zint c = zmodp(t[i], p);
		t[i] = 0;
		t[i - 1] = z_sub(t[i - 1], z_mul(c, f[2]));
		t[i - 2] = z_sub(t[i - 2], z_mul(c, f[1]));
		t[i - 3] = z_sub(t[i - 3], z_mul(c, f[0]));
	}
	for (i = 0; i < 3; i++)
		r[i] = zmodp(t[i], p);
}

/* polynomials mod p as zint coefficient arrays (low first), degree < 8 */
static int
pdeg(const zint *a, int n)
{
	while (n > 0 && z_is_zero(a[n - 1]))
		n--;
	return n - 1;
}

/* r = a mod b over F_p (b != 0), in place on a copy; returns deg r */
static int
pmod(zint *a, int da, const zint *b, int db, zint p)
{
	zint inv = z_invmod(b[db], p);
	int i, k;

	for (i = da; i >= db; i--) {
		zint c = zmodp(z_mul(a[i], inv), p);
		if (z_is_zero(c))
			continue;
		for (k = 0; k <= db; k++)
			a[i - db + k] = zmodp(z_sub(a[i - db + k], z_mul(c, b[k])),
			    p);
	}
	for (i = 0; i < db && i <= da; i++)
		a[i] = zmodp(a[i], p);
	for (i = db; i <= da; i++)
		a[i] = 0;
	return pdeg(a, db);
}

/* degree of gcd(a, b) over F_p */
static int
pgcd_deg(zint *a, int da, zint *b, int db, zint p)
{
	zint t[8];
	int i;

	while (db >= 0) {
		int dr;
		for (i = 0; i < 8; i++)
			t[i] = i <= da ? a[i] : 0;
		dr = pmod(t, da, b, db, p);
		for (i = 0; i < 8; i++)
			a[i] = b[i];
		da = db;
		for (i = 0; i < 8; i++)
			b[i] = t[i];
		db = dr;
	}
	return da;
}

static zint
cubv(zint a, zint b, zint c, zint x)
{
	return z_add(z_mul(z_add(z_mul(z_add(x, a), x), b), x), c);
}

/* the roots in F_p of T^3 + a T^2 + b T + c: *kind 0 squarefree, 1 a
 * double root (in *r), 2 a triple root (in *r); returns the number of
 * distinct roots in F_p */
static int
cubic_roots_modp(zint a, zint b, zint c, zint p, int *kind, zint *r)
{
	a = zmodp(a, p);
	b = zmodp(b, p);
	c = zmodp(c, p);
	*kind = 0;
	if (z_cmp(p, I(1000)) <= 0) {
		int64_t pp = z_to_i64(p), x;
		int n = 0;
		for (x = 0; x < pp; x++) {
			zint X = I(x), q1, q0;
			int m;
			if (!z_is_zero(zmodp(cubv(a, b, c, X), p)))
				continue;
			n++;
			/* multiplicity by synthetic division: f = (T - X) q,
			 * q = T^2 + q1 T + q0 */
			q1 = zmodp(z_add(a, X), p);
			q0 = zmodp(z_add(b, z_mul(q1, X)), p);
			m = 1;
			if (z_is_zero(zmodp(z_add(z_mul(z_add(X, q1), X), q0), p))) {
				m = 2;
				/* q / (T - X) = T + q1 + X: root X again? */
				if (z_is_zero(zmodp(z_add(q1, z_add(X, X)), p)))
					m = 3;
			}
			if (m >= 2) {
				*kind = m - 1;
				*r = X;
			}
		}
		return n;
	}
	{
		/* p >= 5 */
		zint d = zmodp(z_add(z_sub(z_sub(z_sub(z_mul(z_mul(a, a), z_mul(b,
		    b)), z_mul(I(4), z_pow(b, 3))), z_mul(I(4), z_mul(z_pow(a, 3),
		    c))), z_mul(I(27), z_mul(c, c))), z_mul(I(18), z_mul(a, z_mul(b,
		    c)))), p);
		if (z_is_zero(d)) {
			zint s = zmodp(z_sub(z_mul(a, a), z_mul(I(3), b)), p);
			if (z_is_zero(s)) {
				*kind = 2;
				*r = zmodp(z_mul(z_neg(a), z_invmod(I(3), p)), p);
				return 1;
			}
			*kind = 1;
			*r = zmodp(z_mul(z_sub(z_mul(I(9), c), z_mul(a, b)),
			    z_invmod(z_mul(I(2), s), p)), p);
			return 2;
		}
		{
			/* deg gcd(f, T^p - T) */
			zint f[3] = {c, b, a}, acc[3] = {Z_ONE, 0, 0},
			    base[3] = {0, Z_ONE, 0}, e = p, A[8], B[8];
			int i;
			while (!z_is_zero(e)) {
				if (z_mod_u64(e, 2))
					pmulmod3(acc, acc, base, f, p);
				pmulmod3(base, base, base, f, p);
				e = z_div_floor(e, I(2));
			}
			for (i = 0; i < 8; i++)
				A[i] = B[i] = 0;
			A[0] = c;
			A[1] = b;
			A[2] = a;
			A[3] = Z_ONE;
			B[0] = acc[0];
			B[1] = zmodp(z_sub(acc[1], Z_ONE), p);
			B[2] = acc[2];
			return pgcd_deg(A, 3, B, pdeg(B, 3), p);
		}
	}
}

static void
rst(struct ec *E, zint r, zint s, zint t)
{
	struct eciso m;
	struct ec T;

	iso_rst(&m, r, s, t);
	eciso_apply(E, &m, &T);
	*E = T;
}

static const char *const kodname[] = {"?", "I0", "I", "II", "III", "IV",
	"I0*", "I*", "IV*", "III*", "II*"};

const char *
ecq_kodaira_str(const struct ecq_local *L)
{
	static char buf[4][32];
	static int k;
	char *b = buf[k++ & 3];

	if (L->kod == KOD_IN)
		snprintf(b, 32, "I%d", L->n);
	else if (L->kod == KOD_INS)
		snprintf(b, 32, "I%d*", L->n);
	else
		snprintf(b, 32, "%s", kodname[L->kod]);
	return b;
}

void
ecq_tate(const struct ec *E0, zint p, struct ecq_local *L)
{
	struct ec E = *E0;
	zint a1, a2, a3, a4, a6, b2, b6, b8, r, t, s, pk;
	int vD, nroots, kind, dbl;

	memset(L, 0, sizeof *L);
	L->p = p;
	for (;;) {
		vD = zval(ZA(&E, disc), p);
		L->vdisc = vD;
		if (vD == 0) {
			L->kod = KOD_I0;
			L->f = 0;
			L->cp = 1;
			return;
		}
		b2 = ZA(&E, b2);
		a1 = ZA(&E, a1);
		a2 = ZA(&E, a2);
		a3 = ZA(&E, a3);
		a4 = ZA(&E, a4);
		a6 = ZA(&E, a6);
		/* move the singular point to (0, 0) */
		if (z_eq(p, I(2))) {
			if (z_mod_u64(b2, 2) == 0) {
				r = zmodp(a4, p);
				t = zmodp(z_add(z_mul(r, z_add(z_add(Z_ONE, a2), a4)),
				    a6), p);
			} else {
				r = zmodp(a3, p);
				t = zmodp(z_add(r, a4), p);
			}
		} else if (z_eq(p, I(3))) {
			if (z_mod_u64(b2, 3) == 0)
				r = zmodp(z_neg(ZA(&E, b6)), p);
			else
				r = zmodp(z_neg(z_mul(b2, ZA(&E, b4))), p);
			t = zmodp(z_add(z_mul(a1, r), a3), p);
		} else {
			zint c4 = ZA(&E, c4), c6 = ZA(&E, c6);
			if (z_divides(p, c4))
				r = z_mul(z_neg(z_invmod(I(12), p)), b2);
			else
				r = z_mul(z_neg(z_invmod(zmodp(z_mul(I(12), c4), p),
				    p)), z_add(c6, z_mul(b2, c4)));
			r = zmodp(r, p);
			t = zmodp(z_mul(z_neg(z_invmod(I(2), p)), z_add(z_mul(a1, r),
			    a3)), p);
		}
		rst(&E, r, 0, t);
		a1 = ZA(&E, a1);
		a2 = ZA(&E, a2);
		a3 = ZA(&E, a3);
		a4 = ZA(&E, a4);
		a6 = ZA(&E, a6);
		b2 = ZA(&E, b2);
		b6 = ZA(&E, b6);
		b8 = ZA(&E, b8);
		if (!z_divides(p, a3) || !z_divides(p, a4) || !z_divides(p, a6))
			fail("internal: Tate: the singular point is not at (0, 0)");
		/* multiplicative */
		if (!z_divides(p, b2)) {
			int n = quad_roots(Z_ONE, a1, z_neg(a2), p, &dbl, &r);
			L->kod = KOD_IN;
			L->n = vD;
			L->f = 1;
			L->ap = n ? 1 : -1;
			L->cp = n ? vD : (vD & 1 ? 1 : 2);
			return;
		}
		L->ap = 0;
		if (!z_divides(z_mul(p, p), a6)) {
			L->kod = KOD_II;
			L->f = vD;
			L->cp = 1;
			return;
		}
		if (!z_divides(z_pow(p, 3), b8)) {
			L->kod = KOD_III;
			L->f = vD - 1;
			L->cp = 2;
			return;
		}
		if (!z_divides(z_pow(p, 3), b6)) {
			int n = quad_roots(Z_ONE, z_divexact(a3, p),
			    z_neg(z_divexact(a6, z_mul(p, p))), p, &dbl, &r);
			L->kod = KOD_IV;
			L->f = vD - 2;
			L->cp = n ? 3 : 1;
			return;
		}
		/* p | a1, a2; p^2 | a3, a4; p^3 | a6 */
		if (z_eq(p, I(2))) {
			s = z_mod(a2, I(2));
			t = z_mul(I(2), z_mod(z_divexact(a6, I(4)), I(2)));
		} else {
			zint h = z_invmod(I(2), p);
			s = zmodp(z_mul(z_neg(a1), h), p);
			t = zmodp(z_mul(z_neg(a3), h), z_mul(p, p));
		}
		rst(&E, 0, s, t);
		a1 = ZA(&E, a1);
		a2 = ZA(&E, a2);
		a3 = ZA(&E, a3);
		a4 = ZA(&E, a4);
		a6 = ZA(&E, a6);
		pk = z_mul(p, p);
		if (!z_divides(p, a1) || !z_divides(p, a2) || !z_divides(pk, a3) ||
		    !z_divides(pk, a4) || !z_divides(z_mul(pk, p), a6))
			fail("internal: Tate: step 6 (p = %s)", z_str(p));
		nroots = cubic_roots_modp(z_divexact(a2, p), z_divexact(a4, pk),
		    z_divexact(a6, z_mul(pk, p)), p, &kind, &r);
		if (kind == 0) {
			L->kod = KOD_I0S;
			L->f = vD - 4;
			L->cp = 1 + nroots;
			return;
		}
		if (kind == 1) {
			/* I_m*: the double root to 0 */
			zint mx = pk, my = pk, root;
			int m = 1;
			rst(&E, z_mul(r, p), 0, 0);
			for (;;) {
				zint qa, qb, qc;
				int n;
				a2 = ZA(&E, a2);
				a3 = ZA(&E, a3);
				a4 = ZA(&E, a4);
				a6 = ZA(&E, a6);
				/* Y^2 + a3/my Y - a6/(mx my) */
				qb = z_divexact(a3, my);
				qc = z_neg(z_divexact(a6, z_mul(mx, my)));
				n = quad_roots(Z_ONE, qb, qc, p, &dbl, &root);
				if (!dbl) {
					L->cp = n ? 4 : 2;
					break;
				}
				rst(&E, 0, 0, z_mul(my, root));
				my = z_mul(my, p);
				m++;
				a2 = ZA(&E, a2);
				a4 = ZA(&E, a4);
				a6 = ZA(&E, a6);
				/* a2/p X^2 + a4/(p mx) X + a6/(p mx my) */
				qa = z_divexact(a2, p);
				qb = z_divexact(a4, z_mul(p, mx));
				qc = z_divexact(a6, z_mul(p, z_mul(mx, my)));
				n = quad_roots(qa, qb, qc, p, &dbl, &root);
				if (!dbl) {
					L->cp = n ? 4 : 2;
					break;
				}
				if (z_eq(p, I(2)))
					root = qc;   /* qa odd: X = c */
				rst(&E, z_mul(mx, root), 0, 0);
				mx = z_mul(mx, p);
				m++;
			}
			L->kod = KOD_INS;
			L->n = m;
			L->f = vD - m - 4;
			return;
		}
		/* a triple root */
		rst(&E, z_mul(r, p), 0, 0);
		a3 = ZA(&E, a3);
		a4 = ZA(&E, a4);
		a6 = ZA(&E, a6);
		{
			zint root;
			int n = quad_roots(Z_ONE, z_divexact(a3, pk),
			    z_neg(z_divexact(a6, z_mul(pk, pk))), p, &dbl, &root);
			if (!dbl) {
				L->kod = KOD_IVS;
				L->f = vD - 6;
				L->cp = n ? 3 : 1;
				return;
			}
			rst(&E, 0, 0, z_mul(root, pk));
		}
		a4 = ZA(&E, a4);
		a6 = ZA(&E, a6);
		if (!z_divides(z_pow(p, 4), a4)) {
			L->kod = KOD_IIIS;
			L->f = vD - 7;
			L->cp = 2;
			return;
		}
		if (!z_divides(z_pow(p, 6), a6)) {
			L->kod = KOD_IIS;
			L->f = vD - 8;
			L->cp = 1;
			return;
		}
		/* not minimal at p: u = p */
		{
			struct eciso m;
			struct ec T;
			eciso_identity(QQ(), &m);
			m.u = fe_from_z(p);
			eciso_apply(&E, &m, &T);
			E = T;
			if (!ecq_is_integral(&E))
				fail("internal: Tate: scaling");
		}
	}
}

int
ecq_local_all(const struct ec *Em, struct ecq_local *L, int max,
    zint *conductor)
{
	zint p[64], N = Z_ONE;
	int e[64], n, j, comp, k = 0;

	n = z_factor(ZA(Em, disc), p, e, 64, &comp);
	if (!comp)
		fail("the discriminant %s could not be factored completely",
		    z_str(ZA(Em, disc)));
	for (j = 0; j < n && k < max; j++) {
		ecq_tate(Em, p[j], &L[k]);
		N = z_mul(N, z_pow(p[j], (uint64_t)L[k].f));
		k++;
	}
	if (conductor)
		*conductor = N;
	return k;
}

/* ---- a_p by Legendre sums ---- */

int64_t
ecq_ap(const struct ec *E, int64_t p)
{
	int64_t A1 = z_mod_u64(fe_qz(&E->a1), (u64)p),
	    A2 = z_mod_u64(fe_qz(&E->a2), (u64)p),
	    A3 = z_mod_u64(fe_qz(&E->a3), (u64)p),
	    A4 = z_mod_u64(fe_qz(&E->a4), (u64)p),
	    A6 = z_mod_u64(fe_qz(&E->a6), (u64)p);
	int64_t x, cnt = 1;   /* O */
	static signed char *qr;
	static int64_t qrp;

	if (p == 2) {
		int64_t y;
		for (x = 0; x < 2; x++)
			for (y = 0; y < 2; y++)
				if (((y * y + A1 * x * y + A3 * y) - (x * x * x + A2 * x *
				    x + A4 * x + A6)) % 2 == 0)
					cnt++;
		return p + 1 - cnt;
	}
	if (qrp != p) {
		qr = xrealloc(qr, (size_t)p);
		memset(qr, -1, (size_t)p);
		qr[0] = 0;
		for (x = 1; x < p; x++)
			qr[(x * x) % p] = 1;
		qrp = p;
	}
	for (x = 0; x < p; x++) {
		/* (a1 x + a3)^2 + 4 (x^3 + a2 x^2 + a4 x + a6) */
		int64_t u = (A1 * x + A3) % p, v;
		v = ((x + A2) % p * x + A4) % p;
		v = (v * x + A6) % p;
		v = (u * u + 4 * v) % p;
		cnt += 1 + qr[v];
	}
	return p + 1 - cnt;
}

/* ---- torsion ---- */

static int
pt_in(const struct ec *E, const struct ecpt *pts, int n, const struct ecpt *P)
{
	int i;

	for (i = 0; i < n; i++)
		if (ec_pt_eq(E, &pts[i], P))
			return 1;
	return 0;
}

void
ecq_torsion(const struct ec *E, struct ecq_tors *T)
{
	struct ec S;
	struct eciso ms;
	zint A, B, D, fp[64], disc;
	int fe_[64], nf, comp, i, j;
	int64_t bound = 0, p;
	int cnt = 0;

	memset(T, 0, sizeof *T);
	ec_O(&T->pts[0]);
	T->n = 1;
	if (!ecq_is_integral(E))
		fail("internal: torsion needs an integral model");
	disc = ZA(E, disc);
	/* the bound: gcd of #E(F_p), good p >= 3 */
	for (p = 3; cnt < 12 && p < 2000; p += 2) {
		int64_t q, np;
		for (q = 3; q * q <= p && p % q; q += 2)
			;
		if (q * q <= p)
			continue;
		if (z_mod_u64(disc, (u64)p) == 0)
			continue;
		np = p + 1 - ecq_ap(E, p);
		bound = bound ? (int64_t)nt_gcd((u64)bound, (u64)np) : np;
		cnt++;
		if (bound == 1)
			break;
	}
	T->bound = bound;
	T->n1 = 1;
	T->n2 = 1;
	if (bound == 1)
		return;
	/* Nagell-Lutz on y^2 = x^3 + A x + B, A = -27 c4, B = -54 c6 */
	eciso_to_short(E, &ms, &S);
	A = ZA(&S, a4);
	B = ZA(&S, a6);
	D = z_add(z_mul(I(4), z_pow(A, 3)), z_mul(I(27), z_mul(B, B)));
	nf = z_factor(D, fp, fe_, 64, &comp);
	if (!comp)
		fail("torsion: %s could not be factored", z_str(D));
	{
		/* every d with d^2 | D (and d = 0) */
		int idx[64];
		memset(idx, 0, sizeof idx);
		for (;;) {
			zint d = Z_ONE, roots[3];
			int nr, k, sgn;
			for (i = 0; i < nf; i++)
				d = z_mul(d, z_pow(fp[i], (uint64_t)idx[i]));
			for (sgn = 0; sgn < 2; sgn++) {
				zint y = sgn ? z_neg(d) : d;
				/* x^3 + A x + B - y^2 = 0 */
				nr = zx_cubic_int_roots(0, A, z_sub(B, z_mul(y, y)),
				    roots);
				for (k = 0; k < nr; k++) {
					struct ecpt P, Q;
					int64_t o;
					P.x = fe_from_z(roots[k]);
					P.y = fe_from_z(y);
					P.inf = 0;
					eciso_pt_inv(E, &ms, &Q, &P);
					o = ec_order_upto(E, &Q, 12);
					if (o > 0 && bound % o == 0 && !pt_in(E, T->pts,
					    T->n, &Q) && T->n < 16)
						T->pts[T->n++] = Q;
				}
			}
			for (i = 0; i < nf; i++) {
				if (2 * (idx[i] + 1) <= fe_[i]) {
					idx[i]++;
					break;
				}
				idx[i] = 0;
			}
			if (i == nf)
				break;
		}
		/* y = 0 */
		{
			zint roots[3];
			int nr = zx_cubic_int_roots(0, A, B, roots), k;
			for (k = 0; k < nr; k++) {
				struct ecpt P, Q;
				P.x = fe_from_z(roots[k]);
				P.y = fe_from_z(0);
				P.inf = 0;
				eciso_pt_inv(E, &ms, &Q, &P);
				if (!pt_in(E, T->pts, T->n, &Q) && T->n < 16)
					T->pts[T->n++] = Q;
			}
		}
	}
	/* structure */
	{
		int n2tors = 0, maxo = 1;
		struct ecpt g = T->pts[0];
		for (i = 0; i < T->n; i++) {
			int64_t o = ec_order_upto(E, &T->pts[i], 12);
			if (!ec_on(E, &T->pts[i]) || o <= 0)
				fail("internal: torsion point check");
			if (o <= 2)
				n2tors++;
			if (o > maxo) {
				maxo = (int)o;
				g = T->pts[i];
			}
		}
		if (n2tors == 4) {
			T->n1 = 2;
			T->n2 = T->n / 2;
			T->gen[0] = g;
			T->ngen = 1;
			/* a 2-torsion point outside <g> */
			for (j = 0; j < T->n; j++) {
				struct ecpt Q, R;
				int64_t o = ec_order_upto(E, &T->pts[j], 2);
				int k, inside = 0;
				if (o != 2)
					continue;
				ec_O(&R);
				for (k = 0; k < maxo; k++) {
					if (ec_pt_eq(E, &R, &T->pts[j]))
						inside = 1;
					ec_add(E, &Q, &R, &g);
					R = Q;
				}
				if (!inside) {
					T->gen[T->ngen++] = T->pts[j];
					break;
				}
			}
		} else {
			T->n1 = 1;
			T->n2 = T->n;
			if (maxo != T->n)
				fail("internal: torsion of order %d is not cyclic",
				    T->n);
			if (T->n > 1) {
				T->gen[0] = g;
				T->ngen = 1;
			}
		}
		/* Mazur */
		if (!((T->n1 == 1 && (T->n2 <= 10 || T->n2 == 12)) ||
		    (T->n1 == 2 && (T->n2 == 2 || T->n2 == 4 || T->n2 == 6 ||
		    T->n2 == 8))))
			fail("internal: torsion Z/%d x Z/%d is not in Mazur's "
			    "list", T->n1, T->n2);
	}
}

/* ================================================================== */
/* heights                                                             */
/* ================================================================== */

double
ecq_naive_height(const struct ec *E, const struct ecpt *P)
{
	struct zq x;
	double a, b;

	if (P->inf)
		return 0;
	x = fe_q(&P->x);
	a = z_is_zero(x.n) ? 0 : zx_log(x.n);
	b = zx_log(x.d);
	return a > b ? a : b;
}

/* the real roots of 4x^3 + b2 x^2 + 2 b4 x + b6 (increasing); count */
static int
real_roots_2div(long double b2, long double b4, long double b6,
    long double *r)
{
	/* x^3 + A x^2 + B x + C */
	long double A = b2 / 4, B = b4 / 2, C = b6 / 4;
	long double Q = (A * A - 3 * B) / 9;
	long double R = (2 * A * A * A - 9 * A * B + 27 * C) / 54;
	long double t;
	int n, i, k;

	if (R * R < Q * Q * Q) {
		long double th = acosl(R / sqrtl(Q * Q * Q));
		long double s = -2 * sqrtl(Q);
		r[0] = s * cosl(th / 3) - A / 3;
		r[1] = s * cosl((th + 2 * 3.14159265358979323846264338L) / 3) -
		    A / 3;
		r[2] = s * cosl((th - 2 * 3.14159265358979323846264338L) / 3) -
		    A / 3;
		n = 3;
	} else {
		long double AA = -cbrtl(R + (R >= 0 ? 1 : -1) *
		    sqrtl(R * R - Q * Q * Q));
		long double BB = AA != 0 ? Q / AA : 0;
		r[0] = AA + BB - A / 3;
		n = 1;
	}
	/* polish by Newton */
	for (i = 0; i < n; i++)
		for (k = 0; k < 4; k++) {
			long double x = r[i];
			long double f = ((x + A) * x + B) * x + C;
			long double d = (3 * x + 2 * A) * x + B;
			if (d != 0)
				r[i] = x - f / d;
		}
	for (i = 1; i < n; i++)
		for (k = i; k > 0 && r[k - 1] > r[k]; k--) {
			t = r[k];
			r[k] = r[k - 1];
			r[k - 1] = t;
		}
	return n;
}

/* the archimedean local height (Silverman's normalisation, without the
 * log|Delta| / 12 term), by Tate's series after shifting x so that every
 * real point has x >= 1 */
static double
lambda_inf(const struct ec *E, const struct ecpt *P, double *err)
{
	long double b2 = zq_to_double(fe_q(&E->b2)),
	    b4 = zq_to_double(fe_q(&E->b4)), b6 = zq_to_double(fe_q(&E->b6)),
	    b8 = zq_to_double(fe_q(&E->b8));
	long double rt[3], r, B2, B4, B6, B8, x, t, sum = 0, w4 = 1, mx = 0;
	int n, k;

	n = real_roots_2div(b2, b4, b6, rt);
	(void)n;
	r = rt[0] - 1;
	/* x = x' + r */
	B2 = b2 + 12 * r;
	B4 = b4 + r * b2 + 6 * r * r;
	B6 = b6 + 2 * r * b4 + r * r * b2 + 4 * r * r * r;
	B8 = b8 + 3 * r * b6 + 3 * r * r * b4 + r * r * r * b2 + 3 * r * r * r * r;
	/* x(P) - r, carefully when x is huge */
	{
		struct zq q = fe_q(&P->x);
		double lx = (z_is_zero(q.n) ? -1e300 : zx_log(q.n)) - zx_log(q.d);
		if (lx > 40) {
			/* t = 1/(x - r) ~ 1/x; log x' = log x + log(1 - r/x) */
			long double xv = expl((long double)lx);
			long double lxp = lx + log1pl(-r / (z_sign(q.n) < 0 ? -xv :
			    xv));
			t = expl(-lxp);
			if (z_sign(q.n) < 0)
				fail("internal: lambda_inf: x < real roots");
			sum = 0;
			x = 0;
			{
				long double lam = 0.5L * lxp;
				for (k = 0; k < 40; k++) {
					long double z = 1 - B4 * t * t - 2 * B6 * t * t * t -
					    B8 * t * t * t * t;
					long double w = 4 * t + B2 * t * t + 2 * B4 * t * t *
					    t + B6 * t * t * t * t;
					long double lz = logl(fabsl(z));
					sum += w4 * lz;
					if (fabsl(lz) > mx)
						mx = fabsl(lz);
					w4 /= 4;
					t = w / z;
				}
				if (err)
					*err = (double)(mx * w4 / 24 + 1e-15L * fabsl(lam +
					    sum / 8));
				return (double)(lam + sum / 8);
			}
		}
		x = (long double)zq_to_double(q) - r;
	}
	if (x < 0.5L)
		fail("internal: lambda_inf: x' = %Lg < 1", x);
	t = 1 / x;
	for (k = 0; k < 40; k++) {
		long double z = 1 - B4 * t * t - 2 * B6 * t * t * t -
		    B8 * t * t * t * t;
		long double w = 4 * t + B2 * t * t + 2 * B4 * t * t * t +
		    B6 * t * t * t * t;
		long double lz = logl(fabsl(z));
		sum += w4 * lz;
		if (fabsl(lz) > mx)
			mx = fabsl(lz);
		w4 /= 4;
		t = w / z;
	}
	if (err)
		*err = (double)(mx * w4 / 24 + 1e-15L * fabsl(0.5L * logl(x) +
		    sum / 8));
	return (double)(0.5L * logl(x) + sum / 8);
}

/* the local height correction at a bad prime p (Silverman 1988, Thm
 * 5.2), the model minimal at p; P with integral x there */
static double
lambda_bad(const struct ec *E, const struct ecpt *P, zint p)
{
	const struct ecf *F = E->F;
	fe t, u;
	struct zq A, B, C, x = fe_q(&P->x);
	int va, vb, vc, N, vc4;
	double lp = zx_log(p);

	if (z_divides(p, x.d))
		return 0.5 * z_val(x.d, p, NULL) * lp;
	/* A = 3x^2 + 2 a2 x + a4 - a1 y */
	fe_mul(F, &t, &P->x, &P->x);
	fe_mul_i64(F, &t, &t, 3);
	fe_mul(F, &u, &E->a2, &P->x);
	fe_mul_i64(F, &u, &u, 2);
	fe_add(F, &t, &t, &u);
	fe_add(F, &t, &t, &E->a4);
	fe_mul(F, &u, &E->a1, &P->y);
	fe_sub(F, &t, &t, &u);
	A = fe_q(&t);
	/* B = 2y + a1 x + a3 */
	fe_mul_i64(F, &t, &P->y, 2);
	fe_mul(F, &u, &E->a1, &P->x);
	fe_add(F, &t, &t, &u);
	fe_add(F, &t, &t, &E->a3);
	B = fe_q(&t);
	va = zval(A.n, p) - zval(A.d, p);
	vb = zval(B.n, p) - zval(B.d, p);
	if (va <= 0 || vb <= 0)
		return 0;
	N = zval(ZA(E, disc), p);
	vc4 = zval(ZA(E, c4), p);
	if (vc4 == 0) {
		double M = vb < N / 2.0 ? vb : N / 2.0;
		return M * (M - N) / (2.0 * N) * lp;
	}
	/* C = 3x^4 + b2 x^3 + 3 b4 x^2 + 3 b6 x + b8 */
	{
		fe c;
		fe_mul_i64(F, &c, &P->x, 3);
		fe_add(F, &c, &c, &E->b2);
		fe_mul(F, &c, &c, &P->x);
		fe_mul_i64(F, &t, &E->b4, 3);
		fe_add(F, &c, &c, &t);
		fe_mul(F, &c, &c, &P->x);
		fe_mul_i64(F, &t, &E->b6, 3);
		fe_add(F, &c, &c, &t);
		fe_mul(F, &c, &c, &P->x);
		fe_add(F, &c, &c, &E->b8);
		C = fe_q(&c);
	}
	vc = z_is_zero(C.n) ? 1 << 20 : zval(C.n, p) - zval(C.d, p);
	if (vc >= 3 * vb)
		return -(vb / 3.0) * lp;
	return -(vc / 8.0) * lp;
}

/* a cache keyed by the curve's a-invariants (text, so it survives the
 * arena): is the model minimal, and the primes of its discriminant */
static struct {
	char key[512];
	int minimal;
	int np;
	char primes[64][64];
} hc;

static int
hcache(const struct ec *E)
{
	struct sbuf s = {0};
	int same;

	ec_print_ainv(&s, E);
	same = strlen(sb_str(&s)) < sizeof hc.key && !strcmp(hc.key, sb_str(&s));
	if (!same && strlen(sb_str(&s)) < sizeof hc.key) {
		struct ec Em;
		struct eciso m;
		zint p[64];
		int e[64], comp, j;
		zmark mk = z_mark();
		snprintf(hc.key, sizeof hc.key, "%s", sb_str(&s));
		hc.minimal = 0;
		hc.np = 0;
		if (ecq_is_integral(E)) {
			ecq_minimal(E, &Em, &m);
			hc.minimal = fe_is_one(E->F, &m.u);
		}
		if (hc.minimal) {
			hc.np = z_factor(ZA(E, disc), p, e, 64, &comp);
			for (j = 0; j < hc.np; j++)
				snprintf(hc.primes[j], sizeof hc.primes[j], "%s",
				    z_str(p[j]));
		}
		z_release(mk);
		same = 1;
	}
	free(s.p);
	return same && hc.minimal;
}

/* the canonical height on a minimal model (BSD normalisation: twice
 * Silverman's sum of local heights, h^(P) ~ h(x(P))) */
static double
height_min(const struct ec *E, const struct ecpt *P, double *err)
{
	int j;
	double h, er = 0;
	struct zq x;
	zmark m = z_mark();

	if (P->inf || ec_order_upto(E, P, 12) > 0) {
		if (err)
			*err = 0;
		z_release(m);
		return 0;
	}
	h = lambda_inf(E, P, &er);
	x = fe_q(&P->x);
	h += 0.5 * zx_log(x.d);
	for (j = 0; j < hc.np; j++) {
		zint p;
		z_parse(hc.primes[j], &p, NULL);
		if (!z_divides(p, x.d))
			h += lambda_bad(E, P, p);
	}
	z_release(m);
	if (err)
		*err = 2 * er;
	return 2 * h;
}

double
ecq_height(const struct ec *E, const struct ecpt *P, double *err)
{
	struct ec Em;
	struct eciso m;
	struct ecpt Q;
	double h;

	if (hcache(E))
		return height_min(E, P, err);
	ecq_minimal(E, &Em, &m);
	eciso_pt(E, &m, &Q, P);
	if (!hcache(&Em))
		fail("internal: the minimal model is not minimal");
	h = height_min(&Em, &Q, err);
	return h;
}

double
ecq_height_pairing(const struct ec *E, const struct ecpt *P,
    const struct ecpt *Q)
{
	struct ecpt S;

	ec_add(E, &S, P, Q);
	return (ecq_height(E, &S, NULL) - ecq_height(E, P, NULL) -
	    ecq_height(E, Q, NULL)) / 2;
}

static double
det_sym(double *a, int n)
{
	double d = 1;
	int i, j, k;

	for (i = 0; i < n; i++) {
		int piv = i;
		for (j = i + 1; j < n; j++)
			if (fabs(a[j * n + i]) > fabs(a[piv * n + i]))
				piv = j;
		if (a[piv * n + i] == 0)
			return 0;
		if (piv != i) {
			for (k = 0; k < n; k++) {
				double t = a[i * n + k];
				a[i * n + k] = a[piv * n + k];
				a[piv * n + k] = t;
			}
			d = -d;
		}
		d *= a[i * n + i];
		for (j = i + 1; j < n; j++) {
			double f = a[j * n + i] / a[i * n + i];
			for (k = i; k < n; k++)
				a[j * n + k] -= f * a[i * n + k];
		}
	}
	return d;
}

double
ecq_regulator(const struct ec *E, const struct ecpt *P, int n)
{
	double m[64];
	int i, j;

	if (n == 0)
		return 1;
	if (n > 8)
		fail("regulator: more than 8 points");
	for (i = 0; i < n; i++) {
		m[i * n + i] = ecq_height(E, &P[i], NULL);
		for (j = 0; j < i; j++)
			m[i * n + j] = m[j * n + i] = ecq_height_pairing(E, &P[i],
			    &P[j]);
	}
	return det_sym(m, n);
}

/* ================================================================== */
/* point search                                                        */
/* ================================================================== */

static int
i128_is_square(i128 v, i128 *root)
{
	static const unsigned char sq64[64] = {1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,
	    1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,
	    1,0,0,0,0,0,0,0,1,0,0,0,0,0,0};
	long double s;
	i128 r;

	if (v < 0)
		return 0;
	if (!sq64[(int)(v & 63)])
		return 0;
	{
		static unsigned char sq63[63], sq65[65], sq11[11], ok;
		int k;
		if (!ok) {
			for (k = 0; k < 63; k++)
				sq63[(k * k) % 63] = 1;
			for (k = 0; k < 65; k++)
				sq65[(k * k) % 65] = 1;
			for (k = 0; k < 11; k++)
				sq11[(k * k) % 11] = 1;
			ok = 1;
		}
		if (!sq63[(int)(v % 63)] || !sq65[(int)(v % 65)] ||
		    !sq11[(int)(v % 11)])
			return 0;
	}
	s = sqrtl((long double)v);
	r = (i128)s;
	while (r > 0 && r * r > v)
		r--;
	while ((r + 1) * (r + 1) <= v)
		r++;
	if (r * r != v)
		return 0;
	*root = r;
	return 1;
}

static int
fits_small(zint a, int64_t lim)
{
	return z_fits_i64(a) && z_to_i64(a) < lim && z_to_i64(a) > -lim;
}

int
ecq_search(const struct ec *E, int64_t H, struct ecpts *T, int max)
{
	int64_t d, a, a1, a2, a3, a4, a6, lim = (int64_t)1 << 30;
	int found = 0;

	if (!ecq_is_integral(E))
		fail("internal: search needs an integral model");
	if (!fits_small(ZA(E, a1), lim) || !fits_small(ZA(E, a2), lim) ||
	    !fits_small(ZA(E, a3), lim) || !fits_small(ZA(E, a4), lim) ||
	    !fits_small(ZA(E, a6), (int64_t)1 << 50))
		fail("point search: the coefficients are too large for the "
		    "sieve (|a6| < 2^50, the others < 2^30)");
	a1 = z_to_i64(ZA(E, a1));
	a2 = z_to_i64(ZA(E, a2));
	a3 = z_to_i64(ZA(E, a3));
	a4 = z_to_i64(ZA(E, a4));
	a6 = z_to_i64(ZA(E, a6));
	if (H > 2000000)
		H = 2000000;
	for (d = 1; d * d <= H; d++) {
		i128 d2 = (i128)d * d, d3 = d2 * d, d4 = d2 * d2, d6 = d3 * d3;
		for (a = -H; a <= H; a++) {
			i128 L, R, D, s;
			if (nt_gcd((u64)(a < 0 ? -a : a), (u64)d) != 1)
				continue;
			L = (i128)a1 * a * d + (i128)a3 * d3;
			R = (i128)a * a * a + (i128)a2 * a * a * d2 +
			    (i128)a4 * a * d4 + (i128)a6 * d6;
			D = L * L + 4 * R;
			if (!i128_is_square(D, &s))
				continue;
			{
				int k;
				for (k = 0; k < (s ? 2 : 1); k++) {
					i128 b2 = -L + (k ? -s : s);
					struct ecpt P;
					/* y = b2 / (2 d^3) */
					P.x = fe_from_q(zq_make(I(a), z_from_i128(d2)));
					P.y = fe_from_q(zq_make(z_from_i128(b2),
					    z_from_i128(2 * d3)));
					P.inf = 0;
					if (!ec_on(E, &P))
						fail("internal: search found a point off "
						    "the curve");
					if (ecpts_find(E, T, &P) == UINT32_MAX) {
						ecpts_push(T, &P);
						found++;
						if (found >= max)
							return found;
					}
				}
			}
		}
	}
	return found;
}

/* ================================================================== */
/* local solubility of y^2 = quartic                                   */
/* ================================================================== */

static int
zv(zint a, zint p)
{
	return z_is_zero(a) ? 1 << 20 : z_val(a, p, NULL);
}

/* is f(x0 + p^n t) a square in Q_p for some t in Z_p (f = c0 + ... + c4
 * x^4)?  Decided on discs where the square class is constant, else
 * split; an undecided disc at depth 40 counts as soluble (conservative:
 * it can only make a Selmer group larger) */
static int
disc_soluble(const zint *c, zint p, zint x0, int n)
{
	zint T[5];
	int k, j, v0, mn = 1 << 30;

	/* Taylor coefficients at x0 */
	for (k = 0; k <= 4; k++) {
		zint s = 0;
		for (j = 4; j >= k; j--) {
			/* binom(j, k) c_j x0^(j - k) */
			static const int bin[5][5] = {{1}, {1, 1}, {1, 2, 1},
			    {1, 3, 3, 1}, {1, 4, 6, 4, 1}};
			s = z_add(s, z_mul(z_mul(I(bin[j][k]), c[j]),
			    z_pow(x0, (uint64_t)(j - k))));
		}
		T[k] = s;
	}
	if (z_is_zero(T[0]))
		return 1;
	v0 = zv(T[0], p);
	for (k = 1; k <= 4; k++) {
		int b = z_is_zero(T[k]) ? 1 << 30 : zv(T[k], p) + n * k;
		if (b < mn)
			mn = b;
	}
	if (v0 < mn) {
		zint u;
		if (v0 & 1)
			return 0;
		z_val(T[0], p, &u);
		if (!z_eq(p, I(2))) {
			zint e = z_div_floor(z_sub(p, Z_ONE), I(2));
			return z_is_one(z_powmod(z_mod(u, p), e, p));
		}
		if (mn - v0 >= 3)
			return z_mod_u64(u, 8) == 1;
	}
	if (n >= 40)
		return 1;
	{
		zint pn = z_pow(p, (uint64_t)n);
		int64_t i, pp = z_fits_i64(p) ? z_to_i64(p) : 0;
		if (pp == 0 || pp > 1000000)
			return 1;   /* too large to split: conservative */
		for (i = 0; i < pp; i++)
			if (disc_soluble(c, p, z_add(x0, z_mul(I(i), pn)), n + 1))
				return 1;
	}
	return 0;
}

static int
real_soluble(const zint *c)
{
	long double a[5], cr[3], best;
	int i, n;

	for (i = 0; i < 5; i++)
		a[i] = zx_to_double(c[i]);
	if (a[4] > 0 || a[0] >= 0)
		return 1;
	/* leading coefficient < 0: the maximum is at a critical point */
	if (a[4] == 0) {
		/* degree <= 3: odd degree takes every sign */
		if (a[3] != 0)
			return 1;
		if (a[2] > 0)
			return 1;
		if (a[2] == 0)
			return a[1] != 0 || a[0] >= 0;
		/* a quadratic opening downwards: max at -a1/(2 a2) */
		{
			long double x = -a[1] / (2 * a[2]);
			return (a[2] * x + a[1]) * x + a[0] >= 0;
		}
	}
	/* f' = 4 a4 x^3 + 3 a3 x^2 + 2 a2 x + a1; as 4x^3 + B2 x^2 + 2 B4 x
	 * + B6 with B2 = 3 a3 / a4, B4 = a2 / a4, B6 = a1 / a4 */
	n = real_roots_2div(3 * a[3] / a[4], a[2] / a[4], a[1] / a[4], cr);
	best = -1e300L;
	for (i = 0; i < n; i++) {
		long double x = cr[i];
		long double v = (((a[4] * x + a[3]) * x + a[2]) * x + a[1]) * x +
		    a[0];
		if (v > best)
			best = v;
	}
	return best >= -1e-12L * (fabsl(a[0]) + 1);
}

int
ecq_quartic_locsol(const zint *c, zint p)
{
	zint rc[5];
	int i;

	if (z_is_zero(p))
		return real_soluble(c);
	if (disc_soluble(c, p, 0, 0))
		return 1;
	/* x = 1 / z with z in p Z_p: z^4 f(1/z) */
	for (i = 0; i < 5; i++)
		rc[i] = c[4 - i];
	return disc_soluble(rc, p, 0, 1);
}

/* ================================================================== */
/* 2-isogeny descent                                                   */
/* ================================================================== */

/* the squarefree part of a nonzero integer (sign kept) */
static zint
sqfree(zint a)
{
	return zx_squarefree_part(a, NULL);
}

/* the squarefree divisors of b (both signs) */
static int
sqfree_divisors(zint b, zint *out, int max)
{
	zint p[64];
	int e[64], n, comp, i, k, cnt = 0;

	n = z_factor(b, p, e, 64, &comp);
	if (!comp)
		fail("descent: %s could not be factored", z_str(b));
	if (n > 20 || (2 << n) > max)
		fail("descent: too many prime factors in %s", z_str(b));
	for (k = 0; k < (1 << n); k++) {
		zint d = Z_ONE;
		for (i = 0; i < n; i++)
			if (k >> i & 1)
				d = z_mul(d, p[i]);
		out[cnt++] = d;
		out[cnt++] = z_neg(d);
	}
	return cnt;
}

/* the subgroup of Q* mod squares generated by g[0..ng-1] (squarefree ints) */
static int
sq_span(const zint *g, int ng, zint *out, int max)
{
	int n = 1, i, j, k;

	out[0] = Z_ONE;
	for (i = 0; i < ng; i++) {
		int have = 0;
		for (j = 0; j < n; j++)
			if (z_eq(out[j], g[i]))
				have = 1;
		if (have)
			continue;
		k = n;
		for (j = 0; j < k && n < max; j++)
			out[n++] = sqfree(z_mul(out[j], g[i]));
	}
	return n;
}

static int
ilog2i(int n)
{
	int k = 0;

	while ((1 << k) < n)
		k++;
	return k;
}

/* the primes where local solubility must be tested: those dividing
 * 2 b (a^2 - 4b), and infinity (0) */
static int
bad_primes(zint a, zint b, zint *ps, int max)
{
	zint p[64], v = z_mul(I(2), z_mul(b, z_sub(z_mul(a, a), z_mul(I(4), b))));
	int e[64], n, comp, i, k = 0;

	n = z_factor(v, p, e, 64, &comp);
	if (!comp)
		fail("descent: %s could not be factored", z_str(v));
	ps[k++] = 0;
	for (i = 0; i < n && k < max; i++)
		ps[k++] = p[i];
	return k;
}

/* search the quartic N^2 = d M^4 + a M^2 e^2 + (b/d) e^4 for a point
 * with |M|, e <= B, coprime; 1 and (M, e, N) when found */
static int
quartic_search(zint d, zint a, zint b, int64_t B, zint *Mo, zint *eo,
    zint *No)
{
	zint bd = z_divexact(b, d);
	int64_t M, e;

	if (z_fits_i64(d) && z_fits_i64(a) && z_fits_i64(bd) &&
	    z_bits(d) < 40 && z_bits(a) < 40 && z_bits(bd) < 40 && B <= 3000) {
		int64_t D = z_to_i64(d), A = z_to_i64(a), BD = z_to_i64(bd);
		for (e = 0; e <= B; e++)
			for (M = (e == 0 ? 1 : -B); M <= B; M++) {
				i128 M2, e2, v, s;
				if (e == 0 && M != 1)
					continue;
				if (nt_gcd((u64)(M < 0 ? -M : M), (u64)e) != 1)
					continue;
				M2 = (i128)M * M;
				e2 = (i128)e * e;
				v = (i128)D * M2 * M2 + (i128)A * M2 * e2 + (i128)BD *
				    e2 * e2;
				if (i128_is_square(v, &s)) {
					*Mo = I(M);
					*eo = I(e);
					*No = z_from_i128(s);
					return 1;
				}
			}
		return 0;
	}
	for (e = 0; e <= B; e++)
		for (M = (e == 0 ? 1 : -B); M <= B; M++) {
			zint M2, e2, v, s;
			if (e == 0 && M != 1)
				continue;
			if (nt_gcd((u64)(M < 0 ? -M : M), (u64)e) != 1)
				continue;
			M2 = z_mul(I(M), I(M));
			e2 = z_mul(I(e), I(e));
			v = z_add(z_add(z_mul(d, z_mul(M2, M2)), z_mul(a, z_mul(M2,
			    e2))), z_mul(bd, z_mul(e2, e2)));
			if (zx_is_square(v, &s)) {
				*Mo = I(M);
				*eo = I(e);
				*No = s;
				return 1;
			}
		}
	return 0;
}

/* one side of the descent: the Selmer group of y^2 = x^3 + a x^2 + b x
 * for the isogeny with kernel (0, 0), and the found images with points */
static void
descent_side(zint a, zint b, int64_t B, zint *sel, int *found, int *nsel,
    int *nimg, struct ecpts *pts, const struct ec *Ec)
{
	zint cand[256], ps[64], img[64], ng[64];
	int nc, np, i, j, ns = 0, nfound = 0;

	nc = sqfree_divisors(b, cand, 256);
	np = bad_primes(a, b, ps, 64);
	for (i = 0; i < nc; i++) {
		zint q[5];
		int ok = 1;
		q[4] = cand[i];
		q[3] = 0;
		q[2] = a;
		q[1] = 0;
		q[0] = z_divexact(b, cand[i]);
		for (j = 0; j < np && ok; j++)
			ok = ecq_quartic_locsol(q, ps[j]);
		if (ok && ns < 64) {
			found[ns] = 0;
			sel[ns++] = cand[i];
		}
	}
	/* the images: 1 and b (from (0, 0)) are always there */
	ng[nfound++] = sqfree(b);
	for (i = 0; i < ns; i++) {
		zint M, e, N;
		int in = 0, k, n;
		n = sq_span(ng, nfound, img, 64);
		for (k = 0; k < n; k++)
			if (z_eq(img[k], sel[i]))
				in = 1;
		if (in)
			continue;
		if (quartic_search(sel[i], a, b, B, &M, &e, &N)) {
			struct ecpt P;
			ng[nfound++] = sel[i];
			/* x = d M^2 / e^2, y = d M N / e^3 */
			P.x = fe_from_q(zq_make(z_mul(sel[i], z_mul(M, M)),
			    z_mul(e, e)));
			P.y = fe_from_q(zq_make(z_mul(sel[i], z_mul(M, N)),
			    z_pow(e, 3)));
			P.inf = z_is_zero(e);
			if (!P.inf) {
				if (!ec_on(Ec, &P))
					fail("internal: descent point off the curve");
				ecpts_push(pts, &P);
			}
		}
	}
	{
		int n = sq_span(ng, nfound, img, 64), k;
		for (i = 0; i < ns; i++)
			for (k = 0; k < n; k++)
				if (z_eq(img[k], sel[i]))
					found[i] = 1;
		*nimg = ilog2i(n);
	}
	*nsel = ilog2i(ns);
	if ((1 << *nsel) != ns)
		fail("internal: the Selmer set has %d elements (not a power of "
		    "2)", ns);
}

void
ecq_descent2(const struct ec *E, struct ecq_descent *D, int64_t search)
{
	struct eciso m1, m2;
	struct ec W, W2;
	zint b2, b4, b6, roots[3], a, b;
	int nr, i;

	memset(D, 0, sizeof *D);
	ecpts_init(&D->pts);
	if (!ecq_is_integral(E))
		fail("internal: descent needs an integral model");
	b2 = ZA(E, b2);
	b4 = ZA(E, b4);
	b6 = ZA(E, b6);
	nr = zx_cubic_int_roots(b2, z_mul(I(8), b4), z_mul(I(16), b6), roots);
	if (nr == 0)
		return;
	D->ok = 1;
	D->rank_hi = 1 << 20;
	for (i = 0; i < nr; i++) {
		struct ecq_descent T;
		struct ec E1, E2;
		struct ecpts p1, p2;
		zint e = roots[i], g;
		int k;
		memset(&T, 0, sizeof T);
		/* [1/2, 0, -a1/2, -a3/2] then x -> x + e */
		eciso_identity(QQ(), &m1);
		m1.u = fe_from_q(zq_make(Z_ONE, I(2)));
		m1.s = fe_from_q(zq_make(z_neg(ZA(E, a1)), I(2)));
		m1.t = fe_from_q(zq_make(z_neg(ZA(E, a3)), I(2)));
		iso_rst(&m2, e, 0, 0);
		eciso_compose(QQ(), &T.to, &m1, &m2);
		eciso_apply(E, &T.to, &W);
		a = ZA(&W, a2);
		b = ZA(&W, a4);
		if (!z_is_zero(ZA(&W, a1)) || !z_is_zero(ZA(&W, a3)) ||
		    !z_is_zero(ZA(&W, a6)))
			fail("internal: descent model");
		/* remove k^2, k^4 from (a, b) */
		g = z_gcd(a, b);
		if (!z_is_zero(g) && !z_is_one(g)) {
			zint p[64];
			int ee[64], n, comp, j;
			n = z_factor(g, p, ee, 64, &comp);
			for (j = 0; j < n; j++)
				while (z_divides(z_mul(p[j], p[j]), a) &&
				    z_divides(z_pow(p[j], 4), b)) {
					struct eciso s;
					eciso_identity(QQ(), &s);
					s.u = fe_from_z(p[j]);
					eciso_compose(QQ(), &T.to, &T.to, &s);
					a = z_divexact(a, z_mul(p[j], p[j]));
					b = z_divexact(b, z_pow(p[j], 4));
				}
			eciso_apply(E, &T.to, &W);
		}
		T.a = a;
		T.b = b;
		ec_from_z(&E1, 0, a, 0, b, 0);
		ec_from_z(&E2, 0, z_mul(I(-2), a), 0, z_sub(z_mul(a, a),
		    z_mul(I(4), b)), 0);
		ecpts_init(&p1);
		ecpts_init(&p2);
		descent_side(a, b, search, T.sel, T.selfound, &T.nsel, &T.nimg,
		    &p1, &E1);
		descent_side(z_mul(I(-2), a), z_sub(z_mul(a, a), z_mul(I(4), b)),
		    search, T.sel2, T.sel2found, &T.nsel2, &T.nimg2, &p2, &E2);
		T.rank_lo = T.nimg + T.nimg2 - 2;
		T.rank_hi = T.nsel + T.nsel2 - 2;
		T.positive_only = 1;
		for (k = 0; k < (1 << T.nsel); k++)
			if (z_sign(T.sel[k]) < 0)
				T.positive_only = 0;
		/* points of E': to E by the dual isogeny, then to the given
		 * curve */
		ecpts_init(&T.pts);
		for (k = 0; k < (int)p1.n; k++) {
			struct ecpt P, Q;
			ecpts_get(&p1, k, &P);
			eciso_pt_inv(E, &T.to, &Q, &P);
			if (!ec_on(E, &Q))
				fail("internal: descent point map");
			ecpts_push(&T.pts, &Q);
		}
		for (k = 0; k < (int)p2.n; k++) {
			struct ecpt P, Q, R;
			struct zq X, Y, bp;
			ecpts_get(&p2, k, &P);
			X = fe_q(&P.x);
			Y = fe_q(&P.y);
			if (zq_is_zero(X))
				continue;
			bp = zq_int(z_sub(z_mul(a, a), z_mul(I(4), b)));
			/* x = Y^2 / (4 X^2), y = Y (b' - X^2) / (8 X^2) */
			Q.x = fe_from_q(zq_div(zq_mul(Y, Y), zq_mul(zq_int(I(4)),
			    zq_mul(X, X))));
			Q.y = fe_from_q(zq_div(zq_mul(Y, zq_sub(bp, zq_mul(X, X))),
			    zq_mul(zq_int(I(8)), zq_mul(X, X))));
			Q.inf = 0;
			if (!ec_on(&E1, &Q))
				fail("internal: dual isogeny image off the curve");
			eciso_pt_inv(E, &T.to, &R, &Q);
			if (!ec_on(E, &R))
				fail("internal: descent point map");
			ecpts_push(&T.pts, &R);
		}
		ecpts_free(&p1);
		ecpts_free(&p2);
		T.ok = 1;
		T.sha_hint = T.rank_hi - T.rank_lo;
		(void)W2;
		if (T.rank_hi < D->rank_hi) {
			ecpts_free(&D->pts);
			*D = T;
		} else
			ecpts_free(&T.pts);
		if (D->rank_hi == D->rank_lo)
			break;
	}
}

void
ecq_descent_free(struct ecq_descent *D)
{
	ecpts_free(&D->pts);
}

/* ================================================================== */
/* generators: independence, reduction, saturation                     */
/* ================================================================== */

int
ecq_add_indep(const struct ec *E, struct ecpts *G, const struct ecpt *P)
{
	struct ecpt pts[9];
	double r0, r1, h;
	uint32_t i;

	if (P->inf || ec_order_upto(E, P, 12) > 0 || G->n >= 8)
		return 0;
	for (i = 0; i < G->n; i++)
		ecpts_get(G, i, &pts[i]);
	pts[G->n] = *P;
	h = ecq_height(E, P, NULL);
	r0 = ecq_regulator(E, pts, (int)G->n);
	r1 = ecq_regulator(E, pts, (int)G->n + 1);
	if (r1 <= 1e-9 * r0 * h)
		return 0;
	ecpts_push(G, P);
	return 1;
}

/* size reduction of the generators with the height pairing (exact on
 * the points, floating on the Gram matrix): P_i -= round(<P_i, P_j> /
 * <P_j, P_j>) P_j until nothing changes, then sort by height */
static void
reduce_gens(const struct ec *E, struct ecpts *G)
{
	int n = (int)G->n, i, j, changed = 1, rounds = 0;
	struct ecpt P[8], Q;
	double h[8];

	for (i = 0; i < n; i++)
		ecpts_get(G, i, &P[i]);
	while (changed && rounds++ < 50) {
		changed = 0;
		for (i = 0; i < n; i++)
			h[i] = ecq_height(E, &P[i], NULL);
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++) {
				double c;
				int64_t k;
				if (i == j)
					continue;
				c = ecq_height_pairing(E, &P[i], &P[j]) / h[j];
				k = (int64_t)floor(c + 0.5);
				if (k == 0)
					continue;
				ec_mul_i64(E, &Q, k, &P[j]);
				ec_sub(E, &P[i], &P[i], &Q);
				h[i] = ecq_height(E, &P[i], NULL);
				changed = 1;
			}
	}
	/* sort by height; the sign: prefer y >= the other point's y? keep */
	for (i = 1; i < n; i++)
		for (j = i; j > 0 && h[j - 1] > h[j]; j--) {
			double t = h[j];
			h[j] = h[j - 1];
			h[j - 1] = t;
			Q = P[j];
			P[j] = P[j - 1];
			P[j - 1] = Q;
		}
	for (i = 0; i < n; i++)
		ecpts_set(G, (uint32_t)i, &P[i]);
}

static int64_t
small_prime_next(int64_t p)
{
	for (p++;; p++) {
		int64_t q;
		if (p < 2)
			continue;
		for (q = 2; q * q <= p && p % q; q++)
			;
		if (q * q > p)
			return p;
	}
}

/* the rank of a matrix over F_l (rows of r entries), in place */
static int
rank_mod(int64_t *m, int rows, int r, int64_t l)
{
	int rk = 0, c, i, j, k;

	for (c = 0; c < r && rk < rows; c++) {
		int piv = -1;
		for (i = rk; i < rows; i++)
			if (m[i * r + c] % l) {
				piv = i;
				break;
			}
		if (piv < 0)
			continue;
		for (k = 0; k < r; k++) {
			int64_t t = m[rk * r + k];
			m[rk * r + k] = m[piv * r + k];
			m[piv * r + k] = t;
		}
		{
			int64_t inv = (int64_t)nt_inv((u64)((m[rk * r + c] % l + l) %
			    l), (u64)l, NULL);
			for (j = 0; j < rows; j++) {
				int64_t f;
				if (j == rk)
					continue;
				f = (m[j * r + c] % l + l) % l * inv % l;
				for (k = 0; k < r; k++)
					m[j * r + k] = ((m[j * r + k] - f * m[rk * r + k]) %
					    l + l) % l;
			}
		}
		rk++;
	}
	return rk;
}

/* is the subgroup generated by P[0..r-1] l-saturated (Z/l-independent
 * images in E(F_p)/l E(F_p) at primes with l || #E(F_p))? */
static int
l_saturated(const struct ec *E, const struct ecpt *P, int r, int64_t l)
{
	int64_t m[64 * 8], p = 2;
	int rows = 0, tries = 0;
	zint D = ZA(E, disc);

	if (r == 0)
		return 1;
	while (rows < 64 && tries < 400) {
		struct ring R;
		struct ecf F;
		struct ec Ep;
		int64_t np, cof, i;
		struct ecpt g;
		int haveg = 0, ok = 1;
		int64_t row[8];
		p = small_prime_next(p);
		tries++;
		if (p < 3 || z_mod_u64(D, (u64)p) == 0)
			continue;
		np = p + 1 - ecq_ap(E, p);
		if (np % l || (np / l) % l == 0)
			continue;
		cof = np / l;
		ring_zn(&R, (u64)p);
		ecf_fq(&F, &R);
		if (!ec_reduce(&Ep, E, &F))
			continue;
		for (i = 0; i < r && ok; i++) {
			struct ecpt Q, V, W;
			int64_t k;
			if (!fe_set_zq(&F, &Q.x, fe_q(&P[i].x)) ||
			    !fe_set_zq(&F, &Q.y, fe_q(&P[i].y))) {
				ok = 0;
				break;
			}
			Q.inf = P[i].inf;
			ec_mul_i64(&Ep, &V, cof, &Q);
			if (V.inf) {
				row[i] = 0;
				continue;
			}
			if (!haveg) {
				g = V;
				haveg = 1;
				row[i] = 1;
				continue;
			}
			ec_O(&W);
			for (k = 0; k < l; k++) {
				if (ec_pt_eq(&Ep, &W, &V))
					break;
				ec_add(&Ep, &W, &W, &g);
			}
			if (k == l)
				fail("internal: saturation: not in the cyclic group");
			row[i] = k;
		}
		ring_free(&R);
		if (!ok)
			continue;
		for (i = 0; i < r; i++)
			m[rows * r + i] = row[i];
		rows++;
		{
			int64_t c[64 * 8];
			memcpy(c, m, sizeof(int64_t) * (size_t)(rows * r));
			if (rank_mod(c, rows, r, l) == r)
				return 1;
		}
	}
	return 0;
}

/* 2-saturation with full rational 2-torsion, exactly: R is in 2E(Q) iff
 * x(R) - e_i is a square for each root e_i of the 2-division polynomial
 * (at R = (e_j, *) the j-th entry is (e_j - e_k)(e_j - e_l)); checked for
 * every nonzero combination of the points mod 2.  -1: not full 2-torsion */
static int
two_saturated_full(const struct ec *E, const struct ecpt *P, int r)
{
	zint X[3];
	struct zq e[3];
	int nr, mask, i, j;

	nr = zx_cubic_int_roots(ZA(E, b2), z_mul(I(8), ZA(E, b4)),
	    z_mul(I(16), ZA(E, b6)), X);
	if (nr < 3)
		return -1;
	for (i = 0; i < 3; i++)
		e[i] = zq_make(X[i], I(4));
	for (mask = 1; mask < (1 << r); mask++) {
		struct ecpt R;
		int in2 = 1;
		ec_O(&R);
		for (i = 0; i < r; i++)
			if (mask >> i & 1)
				ec_add(E, &R, &R, &P[i]);
		if (R.inf)
			return 0;
		for (i = 0; i < 3 && in2; i++) {
			struct zq v = zq_sub(fe_q(&R.x), e[i]);
			if (zq_is_zero(v)) {
				v = zq_int(Z_ONE);
				for (j = 0; j < 3; j++)
					if (j != i)
						v = zq_mul(v, zq_sub(e[i], e[j]));
			}
			if (!zq_is_square(v, NULL))
				in2 = 0;
		}
		if (in2)
			return 0;
	}
	return 1;
}

int
ecq_saturate(const struct ec *E, struct ecpts *G, int pmax)
{
	struct ecpt P[10];
	struct ecq_tors T;
	int64_t l;
	int r = (int)G->n, all = 1, i;

	ecq_torsion(E, &T);
	for (l = 2; l <= pmax; l = small_prime_next(l)) {
		int n = r;
		for (i = 0; i < r; i++)
			ecpts_get(G, (uint32_t)i, &P[i]);
		/* torsion generators whose order l divides */
		for (i = 0; i < T.ngen; i++) {
			int64_t o = ec_order_upto(E, &T.gen[i], 12);
			if (o % l == 0)
				P[n++] = T.gen[i];
		}
		if (l == 2) {
			int t = two_saturated_full(E, P, n);
			if (t >= 0) {
				if (!t)
					all = 0;
				continue;
			}
		}
		if (!l_saturated(E, P, n, l))
			all = 0;
	}
	return all;
}

/* if Q is not in the lattice spanned by G (modulo torsion), replace G by
 * a basis of G + Z Q: Q = sum (a_j / b) P_j numerically (Gram matrix),
 * then the Hermite form of [b I; a] with its unimodular transform gives
 * the new basis as integer combinations of the P_j and Q */
int
ecq_enlarge(const struct ec *E, struct ecpts *G, const struct ecpt *Q)
{
	int r = (int)G->n, i, j, k;
	double A[64], rhs[8], c[8];
	int64_t b, an[8];
	struct ecpt P[8];

	if (r == 0 || r > 8 || Q->inf || ec_order_upto(E, Q, 12) > 0)
		return 0;
	for (i = 0; i < r; i++)
		ecpts_get(G, (uint32_t)i, &P[i]);
	for (i = 0; i < r; i++) {
		for (j = 0; j < r; j++)
			A[i * r + j] = i == j ? ecq_height(E, &P[i], NULL) :
			    ecq_height_pairing(E, &P[i], &P[j]);
		rhs[i] = ecq_height_pairing(E, Q, &P[i]);
	}
	/* solve A c = rhs (Gauss, partial pivoting) */
	for (i = 0; i < r; i++) {
		int piv = i;
		for (j = i + 1; j < r; j++)
			if (fabs(A[j * r + i]) > fabs(A[piv * r + i]))
				piv = j;
		for (k = 0; k < r; k++) {
			double t = A[i * r + k];
			A[i * r + k] = A[piv * r + k];
			A[piv * r + k] = t;
		}
		{
			double t = rhs[i];
			rhs[i] = rhs[piv];
			rhs[piv] = t;
		}
		for (j = i + 1; j < r; j++) {
			double f = A[j * r + i] / A[i * r + i];
			for (k = i; k < r; k++)
				A[j * r + k] -= f * A[i * r + k];
			rhs[j] -= f * rhs[i];
		}
	}
	for (i = r - 1; i >= 0; i--) {
		double s = rhs[i];
		for (k = i + 1; k < r; k++)
			s -= A[i * r + k] * c[k];
		c[i] = s / A[i * r + i];
	}
	for (b = 1; b <= 1000; b++) {
		int ok = 1;
		for (i = 0; i < r && ok; i++) {
			double v = c[i] * (double)b;
			an[i] = (int64_t)floor(v + 0.5);
			if (fabs(v - (double)an[i]) > 1e-5 * (1 + fabs(v)))
				ok = 0;
		}
		if (ok)
			break;
	}
	if (b > 1000 || b == 1)
		return 0;
	{
		struct zmat M, H, U;
		int64_t *rows = xmalloc(sizeof(int64_t) * (size_t)((r + 1) * r));
		struct ecpt N[8];
		for (i = 0; i < r; i++)
			for (j = 0; j < r; j++)
				rows[i * r + j] = i == j ? b : 0;
		for (j = 0; j < r; j++)
			rows[r * r + j] = an[j];
		memset(&M, 0, sizeof M);
		memset(&H, 0, sizeof H);
		memset(&U, 0, sizeof U);
		zm_from_i64(&M, r + 1, r, rows);
		zm_hnf(&M, &H, &U, NULL);
		for (i = 0; i < r; i++) {
			struct ecpt S, T;
			ec_O(&S);
			for (k = 0; k <= r; k++) {
				zint u = ZM(&U, i, k);
				if (z_is_zero(u))
					continue;
				ec_mul(E, &T, u, k < r ? &P[k] : Q);
				ec_add(E, &S, &S, &T);
			}
			N[i] = S;
		}
		zm_free(&M);
		zm_free(&H);
		zm_free(&U);
		free(rows);
		/* the new points must be independent (the HNF rows i < r) */
		{
			double r0 = ecq_regulator(E, P, r), r1 = ecq_regulator(E, N, r);
			if (!(r1 > 0) || fabs(r1 * (double)b * (double)b / (r0 > 0 ?
			    r0 : 1) - 1) > 1e-3 * (double)b * (double)b)
				;   /* the index check below decides */
			if (!(r1 > 1e-12) || r1 >= r0 * (1 - 1e-9))
				return 0;
		}
		for (i = 0; i < r; i++)
			ecpts_set(G, (uint32_t)i, &N[i]);
	}
	return 1;
}

void
ecq_mw_compute(const struct ec *E0, struct ecq_mw *M, int64_t search)
{
	struct ecpts cand;
	uint32_t i;

	memset(M, 0, sizeof *M);
	ecpts_init(&M->gens);
	ecq_minimal(E0, &M->E, &M->tomin);
	ecq_torsion(&M->E, &M->T);
	ecpts_init(&cand);
	ecq_descent2(&M->E, &M->D, search > 0 ? (search < 300 ? search : 300) :
	    100);
	M->rank_hi = -1;
	if (M->D.ok) {
		M->have_descent = 1;
		M->rank_lo = M->D.rank_lo;
		M->rank_hi = M->D.rank_hi;
		M->rank_how = 1;
		for (i = 0; i < M->D.pts.n; i++) {
			struct ecpt P;
			ecpts_get(&M->D.pts, i, &P);
			ecpts_push(&cand, &P);
		}
	}
	if (M->rank_hi != 0) {
		int64_t H = search > 0 ? search : 1000;
		int fits = fits_small(ZA(&M->E, a6), (int64_t)1 << 50) &&
		    fits_small(ZA(&M->E, a4), (int64_t)1 << 30) &&
		    fits_small(ZA(&M->E, a2), (int64_t)1 << 30);
		if (fits)
			ecq_search(&M->E, H, &cand, 4000);
	}
	/* by canonical height (the first 400 by naive height) */
	{
		uint32_t a, b, n = cand.n;
		double *h;
		for (a = 1; a < n; a++)
			for (b = a; b > 0; b--) {
				struct ecpt P, Q;
				ecpts_get(&cand, b - 1, &P);
				ecpts_get(&cand, b, &Q);
				if (ecq_naive_height(&M->E, &P) <= ecq_naive_height(&M->E,
				    &Q))
					break;
				ecpts_set(&cand, b - 1, &Q);
				ecpts_set(&cand, b, &P);
			}
		if (n > 400)
			n = cand.n = 400;
		h = xmalloc(sizeof(double) * (n + 1));
		for (a = 0; a < n; a++) {
			struct ecpt P;
			ecpts_get(&cand, a, &P);
			h[a] = ecq_height(&M->E, &P, NULL);
		}
		for (a = 1; a < n; a++)
			for (b = a; b > 0 && h[b - 1] > h[b]; b--) {
				struct ecpt P, Q;
				double t = h[b];
				h[b] = h[b - 1];
				h[b - 1] = t;
				ecpts_get(&cand, b - 1, &P);
				ecpts_get(&cand, b, &Q);
				ecpts_set(&cand, b - 1, &Q);
				ecpts_set(&cand, b, &P);
			}
		free(h);
	}
	for (i = 0; i < cand.n; i++) {
		struct ecpt P;
		if (M->rank_hi >= 0 && (int)M->gens.n >= M->rank_hi)
			break;
		ecpts_get(&cand, i, &P);
		ecq_add_indep(&M->E, &M->gens, &P);
	}
	/* every candidate must lie in the lattice of the generators: enlarge
	 * it where one does not (Q = sum (a_j / b) P_j modulo torsion) */
	for (i = 0; i < cand.n && M->gens.n > 0; i++) {
		struct ecpt Q;
		ecpts_get(&cand, i, &Q);
		ecq_enlarge(&M->E, &M->gens, &Q);
	}
	ecpts_free(&cand);
	if ((int)M->gens.n > M->rank_lo)
		M->rank_lo = (int)M->gens.n;
	if (M->gens.n > 1)
		reduce_gens(&M->E, &M->gens);
	M->saturated_to = 0;
	if (M->gens.n > 0 && ecq_saturate(&M->E, &M->gens, 7))
		M->saturated_to = 7;
	for (i = 0; i < M->gens.n && i < 8; i++) {
		struct ecpt P;
		ecpts_get(&M->gens, i, &P);
		M->h[i] = ecq_height(&M->E, &P, NULL);
	}
	{
		struct ecpt P[8];
		for (i = 0; i < M->gens.n && i < 8; i++)
			ecpts_get(&M->gens, i, &P[i]);
		M->reg = ecq_regulator(&M->E, P, (int)M->gens.n);
	}
}

void
ecq_mw_free(struct ecq_mw *M)
{
	ecpts_free(&M->gens);
	if (M->have_descent)
		ecq_descent_free(&M->D);
}

int
ecq_real_roots(long double b2, long double b4, long double b6, long double *r)
{
	return real_roots_2div(b2, b4, b6, r);
}

/* no rational point lies on the bounded real component (Delta > 0):
 * with the largest root e3 of the 2-division polynomial rational, every
 * rational point has x - e3 = d * square with d in the Selmer group of
 * the descent; when every Selmer element is positive, x > e3 for every
 * rational point (other than (e3, *)), and the egg lies in x < e3 */
int
ecq_egg_empty(const struct ec *E, char *why, size_t n)
{
	zint roots[3], a, b, sel[64];
	int nr, found[64], nsel, nimg, k;
	struct ecpts pts;
	struct ec W;

	if (z_sign(ZA(E, disc)) <= 0) {
		snprintf(why, n, "the discriminant is negative: E(R) is connected");
		return 0;
	}
	nr = zx_cubic_int_roots(ZA(E, b2), z_mul(I(8), ZA(E, b4)),
	    z_mul(I(16), ZA(E, b6)), roots);
	if (nr < 3) {
		snprintf(why, n, "the 2-division polynomial does not split over "
		    "Q: its largest root may be irrational");
		if (nr == 0)
			return 0;
		{
			long double r[3];
			ecq_real_roots(zx_to_double(ZA(E, b2)), zx_to_double(ZA(E,
			    b4)), zx_to_double(ZA(E, b6)), r);
			/* X = 4x: is the rational root the largest? */
			if (fabsl(zx_to_double(roots[nr - 1]) / 4 - r[2]) > 1e-6L *
			    (1 + fabsl(r[2])))
				return 0;
		}
	}
	{
		zint e3 = roots[nr - 1];
		zint b2 = ZA(E, b2), b4 = z_mul(I(8), ZA(E, b4));
		a = z_add(z_mul(I(3), e3), b2);
		b = z_add(z_add(z_mul(I(3), z_mul(e3, e3)), z_mul(z_mul(I(2), b2),
		    e3)), b4);
	}
	ec_from_z(&W, 0, a, 0, b, 0);
	ecpts_init(&pts);
	descent_side(a, b, 1, sel, found, &nsel, &nimg, &pts, &W);
	ecpts_free(&pts);
	for (k = 0; k < (1 << nsel); k++)
		if (z_sign(sel[k]) < 0) {
			snprintf(why, n, "the Selmer group of the descent at the "
			    "largest 2-torsion point contains %s < 0", z_str(sel[k]));
			return 0;
		}
	{
		struct sbuf s = {0};
		for (k = 0; k < (1 << nsel); k++)
			sb_printf(&s, "%s%s", k ? ", " : "", z_str(sel[k]));
		snprintf(why, n, "on y^2 = x^3 + %s x^2 + %s x (X = 4x shifted by the "
		    "largest root of the 2-division polynomial) the 2-isogeny Selmer group "
		    "for x mod squares is {%s}: all positive, so every rational "
		    "point has x >= 0, while the bounded component lies in x < 0",
		    z_str(a), z_str(b), sb_str(&s));
		free(s.p);
	}
	return 1;
}
