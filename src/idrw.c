/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Identities that are solved for the function at hand, not written
 * down: each candidate's constants are computed from the divisor or
 * from the values the subterm takes on its domain, and its derivation
 * names them.  As in irw.c nothing here decides: find keeps a candidate
 * only when the active cost model prices it lower.
 *
 *   - a divisibility test by a constant, n % d == r, as a multiply and
 *     a compare (Granlund and Montgomery; Lemire): with d = 2^k e, e
 *     odd, n -> rotr((n - r) e^-1, k) is a bijection of the machine
 *     word that takes the n with that remainder onto 0..B;
 *   - a division or remainder by a constant whose dividend has a proved
 *     bound N, as one multiply and one shift in u32: M = ceil(2^s / d)
 *     with N (M d - 2^s) < 2^s and N M < 2^32;
 *   - a map of one variable with a small range (compares and selects, a
 *     table, quotients), from its values on that range:
 *       a table in one constant, (C >> b (x - lo)) & (2^b - 1), when
 *       its entries fit 32 bits;
 *       a multiply and a shift, (x M + A) >> s, with M, A and s solved
 *       so that v 2^s <= x M + A < (v + 1) 2^s at every x.
 */
#include "calc.h"
#include "irw.h"

#include <stdlib.h>
#include <string.h>

#define U32 TY(K_UINT, 32)
#define MAP_POINTS 4096		/* values of the variable a map is read at */
#define MAP_LISTED 16		/* of them, printed in the derivation */

static const char *
sstr(nid e)
{
	return shared_str(e, 60);
}

static int
pow2(int64_t v)
{
	return v > 0 && !(v & (v - 1));
}

/* the inverse of an odd e mod 2^64 (Newton: each step doubles the bits) */
static uint64_t
inv_odd(uint64_t e)
{
	uint64_t x = e;	/* e * e = 1 mod 8 */
	int i;

	for (i = 0; i < 5; i++)
		x *= 2 - e * x;
	return x;
}

/* ---- n % d == r by a multiply ---- */

static void
rw_divis(struct ivc *c, nid root, nid n, struct irw **out, int *cnt, int *cap)
{
	nid m = N.a[n], rc = N.b[n], x, t1, to;
	ty_t t, U;
	int W, k, sg, eq = N.op[n] == O_EQ;
	int64_t d, r;
	uint64_t mask, e, einv, cc, B, A = 0, qmax = 0;
	char what[200], cond[200], tn[120];
	struct sbuf pf = {0};
	struct irw *rw;

	if (N.op[m] != O_MOD) {
		m = N.b[n];
		rc = N.a[n];
	}
	if (N.op[m] != O_MOD || !is_const(rc) || !is_const(N.b[m]) ||
	    TY_FP(N.ty[rc]))
		return;
	t = N.ty[m];
	W = TYW(t);
	sg = TYK(t) == K_SINT;
	if ((TYK(t) != K_UINT && !sg) || W < 8 || W > 32)
		return;
	x = N.a[m];
	d = N.ki[N.b[m]];
	r = N.ki[rc];
	if (d < 3 || pow2(d) || r < 0 || r >= d || (sg && r != 0))
		return;
	U = TY(K_UINT, W);
	mask = (1ULL << W) - 1;
	for (k = 0; !((d >> k) & 1); k++)
		;
	e = (uint64_t)d >> k;
	einv = inv_odd(e) & mask;
	if (!sg) {
		cc = (0 - (uint64_t)r * einv) & mask;
		B = (mask - (uint64_t)r) / (uint64_t)d;
	} else {
		A = (1ULL << (W - 1)) / (uint64_t)d;
		qmax = ((1ULL << (W - 1)) - 1) / (uint64_t)d;
		cc = (A << k) & mask;
		B = qmax + A;
	}
	t1 = mk_bin(O_MUL, mk_cast(U, x), mk_const_i(U, (int64_t)einv));
	if (cc)
		t1 = mk_bin(O_ADD, t1, mk_const_i(U, (int64_t)cc));
	if (k)
		t1 = mk_bin(O_OR, mk_bin(O_SHR, t1, mk_const_i(U, k)),
		    mk_bin(O_SHL, t1, mk_const_i(U, W - k)));
	to = mk_bin(eq ? O_LE : O_GT, t1, mk_const_i(U, (int64_t)B));

	snprintf(what, sizeof what, "the divisibility test %s as a multiply%s "
	    "and a compare", sstr(n), k ? ", a rotation" : "");
	snprintf(cond, sizeof cond, "%s is any %s (the multiply wraps mod "
	    "2^%d)", sstr(x), ty_str(t), W);
	snprintf(tn, sizeof tn, "%sn*0x%llx%s", k ? "rotr(" : "",
	    (unsigned long long)einv, cc ? " + c" : "");
	if (k)
		snprintf(tn + strlen(tn), sizeof tn - strlen(tn), ", %d)", k);
	if (k)
		sb_printf(&pf, "%lld = 2^%d * %llu with %llu odd, and %llu * "
		    "0x%llx = 1 mod 2^%d", (long long)d, k, (unsigned long long)e,
		    (unsigned long long)e, (unsigned long long)e,
		    (unsigned long long)einv, W);
	else
		sb_printf(&pf, "%lld is odd, and %lld * 0x%llx = 1 mod 2^%d",
		    (long long)d, (long long)d, (unsigned long long)einv, W);
	if (cc)
		sb_printf(&pf, "; c = 0x%llx is %s mod 2^%d",
		    (unsigned long long)cc, sg ? "A * 2^k" : "-r * that inverse",
		    W);
	sb_printf(&pf, "\nt(n) = %s takes each %d-bit word to a different one "
	    "(a bijection): a multiply by an odd number%s%s %s", tn, W,
	    cc ? ", an add" : "", k ? " and a rotation" : "", cc || k ?
	    "each do" : "does");
	if (!sg && r) {
		sb_printf(&pf, "\nthe n with n %% %lld = %lld are %lld + q*%lld "
		    "for q = 0..%llu (floor((2^%d - 1 - %lld)/%lld)); there (n - "
		    "%lld) * 0x%llx = q%s mod 2^%d", (long long)d, (long long)r,
		    (long long)r, (long long)d, (unsigned long long)B, W,
		    (long long)r, (long long)d, (long long)r,
		    (unsigned long long)einv, k ? " * 2^k" : "", W);
	} else if (!sg) {
		sb_printf(&pf, "\nthe multiples of %lld are q*%lld for q = "
		    "0..%llu (floor((2^%d - 1)/%lld)); there n * 0x%llx = q%s "
		    "mod 2^%d", (long long)d, (long long)d, (unsigned long long)B,
		    W, (long long)d, (unsigned long long)einv, k ? " * 2^k" : "",
		    W);
	} else {
		sb_printf(&pf, "\nthe multiples of %lld in %s are q*%lld for q = "
		    "-A..%llu (A = floor(2^%d/%lld) = %llu); there n * 0x%llx + "
		    "c = (q + A)%s mod 2^%d", (long long)d, ty_str(t),
		    (long long)d, (unsigned long long)qmax, W - 1, (long long)d,
		    (unsigned long long)A, (unsigned long long)einv, k ?
		    " * 2^k" : "", W);
	}
	if (k)
		sb_printf(&pf, ", below 2^%d because %s*%lld is, and rotated "
		    "right by %d it is %s", W, sg ? "(q + A)" : "q", (long long)d,
		    k, sg ? "q + A" : "q");
	sb_printf(&pf, ": t takes them onto 0..%llu", (unsigned long long)B);
	sb_printf(&pf, "\na bijection takes every other n outside 0..%llu: "
	    "t(n) <= %llu exactly where %s", (unsigned long long)B,
	    (unsigned long long)B, sstr(eq ? n : mk_bin(O_EQ, m, rc)));
	rw = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	if (rw)
		rw->nosat = rw->derived = 1;
	free(pf.p);
}

/* ---- x / d and x % d, x <= N, by one multiply and one shift ---- */

/* the least s with M = ceil(2^s/d), N (M d - 2^s) < 2^s and N M < 2^32 */
static int
divk_solve(uint64_t N_, uint64_t d, uint64_t *M)
{
	int s;

	for (s = 1; s < 32; s++) {
		uint64_t m = ((1ULL << s) + d - 1) / d, e = m * d - (1ULL << s);
		if (N_ * e < (1ULL << s) && N_ * m < (1ULL << 32)) {
			*M = m;
			return s;
		}
	}
	return 0;
}

/* the largest N some s serves, and that s */
static uint64_t
divk_reach(uint64_t d, int *sbest)
{
	uint64_t best = 0;
	int s;

	for (s = 1; s < 32; s++) {
		uint64_t m = ((1ULL << s) + d - 1) / d, e = m * d - (1ULL << s);
		uint64_t n1 = ((1ULL << s) - 1) / e, n2 = 0xffffffffULL / m;
		if (n2 < n1)
			n1 = n2;
		if (n1 > best) {
			best = n1;
			*sbest = s;
		}
	}
	return best;
}

static void
rw_divk(struct ivc *c, nid root, nid n, struct irw **out, int *cnt, int *cap)
{
	ty_t t = N.ty[n];
	nid a = N.a[n], b = N.b[n], xu, q, to;
	int64_t alo, ahi, d;
	uint64_t M = 0, e, reach = 0;
	int s, div = N.op[n] == O_DIV, full;
	char wa[200], what[200], cond[400];
	struct sbuf pf = {0};
	struct irw *rw;

	if (!TY_ISINT(t) || t == TY_BOOL || !is_const(b))
		return;
	d = N.ki[b];
	if (d < 3 || pow2(d) || d >= (1LL << 32))
		return;
	if (!irw_range(c, a, 0, &alo, &ahi, wa, sizeof wa) || alo < 0 ||
	    ahi < d || ahi >= (1LL << 32))
		return;
	/* a dividend that takes its whole type: nothing was proved of it */
	full = TY_SIZED(t) && TYW(t) <= 32 && ahi == (TYK(t) == K_SINT ?
	    (1LL << (TYW(t) - 1)) - 1 : (1LL << TYW(t)) - 1);
	s = divk_solve((uint64_t)ahi, (uint64_t)d, &M);
	if (!s) {
		if (full && TYW(t) == 32)
			return;
		reach = divk_reach((uint64_t)d, &s);
		M = ((1ULL << s) + (uint64_t)d - 1) / (uint64_t)d;
	}
	e = M * (uint64_t)d - (1ULL << s);
	xu = mk_cast(U32, a);
	q = mk_bin(O_SHR, mk_bin(O_MUL, xu, mk_const_i(U32, (int64_t)M)),
	    mk_const_i(U32, s));
	to = mk_cast(t, div ? q : mk_bin(O_SUB, xu, mk_bin(O_MUL, q,
	    mk_const_i(U32, d))));
	snprintf(what, sizeof what, "the %s %s as a multiply and a shift in "
	    "u32", div ? "division" : "remainder", sstr(n));
	sb_printf(&pf, "M = ceil(2^%d/%lld) = %llu, so M*%lld = 2^%d + %llu", s,
	    (long long)d, (unsigned long long)M, (long long)d, s,
	    (unsigned long long)e);
	sb_printf(&pf, "\nx*M/2^%d = x/%lld + x*%llu/(%lld*2^%d): the excess is "
	    "below 1/%lld while x*%llu < 2^%d, and x/%lld is at least 1/%lld "
	    "below the next integer, so floor(x*M/2^%d) = floor(x/%lld)", s,
	    (long long)d, (unsigned long long)e, (long long)d, s, (long long)d,
	    (unsigned long long)e, s, (long long)d, (long long)d, s,
	    (long long)d);
	if (reach) {
		snprintf(cond, sizeof cond, "%s <= %llu (it is in [%lld, %lld]: "
		    "%s)", sstr(a), (unsigned long long)reach, (long long)alo,
		    (long long)ahi, wa);
		sb_printf(&pf, "\nthat and a product below 2^32 need x <= %llu "
		    "(the most any shift allows, with 2^%d); here x reaches "
		    "%lld", (unsigned long long)reach, s, (long long)ahi);
		irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 0);
		free(pf.p);
		return;
	}
	snprintf(cond, sizeof cond, "%s in [%lld, %lld] (%s)", sstr(a),
	    (long long)alo, (long long)ahi, wa);
	sb_printf(&pf, "\nx <= %lld: x*%llu <= %llu < 2^%d, and x*M <= %llu < "
	    "2^32, so the product does not wrap in u32", (long long)ahi,
	    (unsigned long long)e, (unsigned long long)ahi * e, s,
	    (unsigned long long)ahi * M);
	if (!div)
		sb_printf(&pf, "\nand x %% %lld = x - %lld*floor(x/%lld)",
		    (long long)d, (long long)d, (long long)d);
	rw = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	if (rw)
		rw->nosat = rw->derived = 1;
	free(pf.p);
}

void
idrw_node(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	int op = N.op[n];

	if (op == O_EQ || op == O_NE)
		rw_divis(c, root, n, out, cnt, cap);
	if (op == O_DIV || op == O_MOD)
		rw_divk(c, root, n, out, cnt, cap);
}

/* ---- maps of one small-range variable ---- */

struct mapv {
	nid s, x;		/* the map and its variable */
	nid from, key;		/* in the function: the subterm that is the map,
				 * and what its variable stands for (x itself,
				 * or the subterm a fresh x was put in place
				 * of) */
	ty_t t;			/* the type the result is computed in */
	char why[120];		/* why the key stays in lo..hi */
	int64_t lo, hi;		/* the variable's range */
	int K;
	int64_t *v;		/* the map's value at lo + i */
	int64_t vmin, vmax;
};

/* s at every value of its variable; 0 when one of them is undefined */
static int
map_values(struct ivc *c, struct mapv *m, int slot)
{
	struct plan p;
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int base, i, ok = 1;

	plan_build(&p, &m->s, 1, 0);
	m->v = xmalloc((size_t)m->K * sizeof *m->v);
	for (base = 0; ok && base < m->K; base += CH) {
		int np = m->K - base < CH ? m->K - base : CH;
		for (i = 0; i < np; i++)
			in_i[(size_t)slot * CH + i] = m->lo + base + i;
		plan_run(&p, (uint32_t)np, in_i, in_r);
		for (i = 0; i < np; i++) {
			if (COLF(&p, p.root[0])[i] & F_UNDEF) {
				ok = 0;
				break;
			}
			m->v[base + i] = COLI(&p, p.root[0])[i];
		}
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	if (!ok)
		return 0;
	m->vmin = m->vmax = m->v[0];
	for (i = 1; i < m->K; i++) {
		if (m->v[i] < m->vmin)
			m->vmin = m->v[i];
		if (m->v[i] > m->vmax)
			m->vmax = m->v[i];
	}
	return 1;
}

static void
values_text(struct sbuf *b, const struct mapv *m)
{
	int i;

	sb_printf(b, "%s at %s = %lld..%lld is [", sstr(m->from), sstr(m->key),
	    (long long)m->lo, (long long)m->hi);
	for (i = 0; i < m->K && i < MAP_LISTED; i++)
		sb_printf(b, "%s%lld", i ? ", " : "", (long long)m->v[i]);
	if (i < m->K)
		sb_printf(b, ", ... (%d more)", m->K - i);
	sb_printf(b, "] (evaluated at each of the %d)", m->K);
}

/* a table in one u32 constant */
static int
map_lut(struct ivc *c, nid root, const struct mapv *m, struct irw **out,
    int *cnt, int *cap)
{
	static const int widths[] = {1, 2, 4, 8, 16};
	ty_t t = m->t;
	uint64_t span = (uint64_t)(m->vmax - m->vmin), C = 0, bmask;
	int b = bits_for(span), bw = 0, i, direct, off;
	nid idx, sh, r, to;
	struct irw *rw;
	char what[200], cond[200];
	struct sbuf pf = {0};

	for (i = 0; i < 5 && !bw; i++)
		if (widths[i] >= b && (int64_t)m->K * widths[i] <= 32)
			bw = widths[i];
	if (!bw && (int64_t)m->K * b <= 32)
		bw = b;
	if (!bw)
		return 0;
	direct = m->lo >= 0 && (m->hi + 1) * bw <= 32;
	off = direct ? (int)m->lo : 0;
	bmask = (1ULL << bw) - 1;
	for (i = 0; i < m->K; i++)
		C |= (uint64_t)(m->v[i] - m->vmin) << (bw * (i + off));
	/* what is built is what is claimed: read every entry back */
	for (i = 0; i < m->K; i++)
		if ((int64_t)((C >> (bw * (i + off))) & bmask) + m->vmin !=
		    m->v[i])
			return 0;
	idx = mk_cast(U32, direct ? m->x : m->lo < 0 ? mk_bin(O_ADD, m->x,
	    mk_const_i(N.ty[m->x], -m->lo)) : mk_bin(O_SUB, m->x,
	    mk_const_i(N.ty[m->x], m->lo)));
	sh = bw == 1 ? idx : pow2(bw) ? mk_bin(O_SHL, idx, mk_const_i(U32,
	    ilog2u((uint64_t)bw))) : mk_bin(O_MUL, idx, mk_const_i(U32, bw));
	r = mk_bin(O_SHR, mk_const_i(U32, (int64_t)C), sh);
	if (t == TY_BOOL)
		to = mk_cast(t, r);	/* a bool is the low bit */
	else {
		to = mk_cast(t, mk_bin(O_AND, r, mk_const_i(U32,
		    (int64_t)bmask)));
		if (m->vmin)
			to = mk_bin(O_ADD, to, mk_const_i(t, m->vmin));
	}
	snprintf(what, sizeof what, "the map %s of %s as a table in one "
	    "constant (%d entries of %d bit%s)", sstr(m->from), sstr(m->key), m->K,
	    bw, bw > 1 ? "s" : "");
	snprintf(cond, sizeof cond, "%s in [%lld, %lld] (%s)", sstr(m->key),
	    (long long)m->lo, (long long)m->hi, m->why);
	values_text(&pf, m);
	sb_printf(&pf, "\nbits [%d*i, %d*i + %d) of 0x%llx hold the value at "
	    "%s = %s%s: shifting right by %s%s%s reads it", bw, bw, bw,
	    (unsigned long long)C, sstr(m->key), direct ? "i" : "lo + i",
	    m->vmin ? ", less the least value" : "", bw > 1 ? "that many "
	    "bits times " : "", direct ? sstr(m->key) : "the offset from lo",
	    t == TY_BOOL ? " and taking the low bit" : " and masking");
	if (m->key != m->x)
		to = xreplace(to, m->x, m->key);
	rw = irw_add(out, cnt, cap, root, m->from, to, what, cond, sb_str(&pf), 1);
	if (rw)
		rw->derived = 1;
	free(pf.p);
	return 1;
}

/*
 * For a shift s and a multiplier M, the offsets A with v 2^s <= i M + A <
 * (v + 1) 2^s at every i are Alo..Ahi; the width Ahi - Alo is concave in
 * M (a minimum of lines less a maximum of lines).
 */
static int64_t
ms_width(const struct mapv *m, int s, int64_t M, int64_t base, int64_t *alo,
    int64_t *ahi, int *ilo, int *ihi)
{
	int64_t lo = INT64_MIN, hi = INT64_MAX;
	int i;

	for (i = 0; i < m->K; i++) {
		int64_t w = m->v[i] - base, p = (int64_t)i * M;
		int64_t l = w * (1LL << s) - p, h = (w + 1) * (1LL << s) - 1 - p;
		if (l > lo) {
			lo = l;
			if (ilo)
				*ilo = i;
		}
		if (h < hi) {
			hi = h;
			if (ihi)
				*ihi = i;
		}
	}
	if (alo)
		*alo = lo;
	if (ahi)
		*ahi = hi;
	return hi - lo;
}

/* (x M + A) >> s: s, M and A solved from the values */
static int
map_mulshift(struct ivc *c, nid root, const struct mapv *m, struct irw **out,
    int *cnt, int *cap)
{
	ty_t t = m->t;
	int64_t base = m->vmin < 0 ? m->vmin : 0, wmax = m->vmax - base;
	int64_t dv = m->v[m->K - 1] - m->v[0], M = 0, alo = 0, ahi = 0, A;
	int s, found = 0, i, ilo = 0, ihi = 0;
	uint32_t Mu, Au;
	nid r, to;
	char what[200], cond[200];
	struct sbuf pf = {0};
	struct irw *rw;

	if (wmax >= (1LL << 31) || m->K < 3)
		return 0;
	/* affine (a shift of 0 serves): the affine fit's, not ours */
	if (ms_width(m, 0, dv / (m->K - 1), base, NULL, NULL, NULL, NULL) >= 0
	    && dv % (m->K - 1) == 0)
		return 0;
	for (s = 1; s < 32 && !found; s++) {
		int64_t one = 1LL << s, a, b, mm;
		if ((wmax + 1) * one > (1LL << 32))
			break;	/* x M + A would leave u32 */
		/* from the first and the last point: (K-1) M is within 2^s
		 * of dv 2^s */
		a = ((dv - 1) * one) / (m->K - 1) - 1;
		b = ((dv + 1) * one) / (m->K - 1) + 1;
		while (b - a > 2) {
			int64_t m1 = a + (b - a) / 3, m2 = b - (b - a) / 3;
			int64_t w1 = ms_width(m, s, m1, base, NULL, NULL, NULL,
			    NULL), w2 = ms_width(m, s, m2, base, NULL, NULL,
			    NULL, NULL);
			if (w1 < w2)
				a = m1 + 1;
			else if (w1 > w2)
				b = m2 - 1;
			else {
				a = m1;
				b = m2;
			}
		}
		for (mm = a; mm <= b; mm++)
			if (ms_width(m, s, mm, base, &alo, &ahi, &ilo, &ihi) >= 0) {
				found = s;
				M = mm;
				break;
			}
	}
	if (!found)
		return 0;
	s = found;
	/* in terms of x = lo + i the offset is A - lo M; 0 when it can be */
	A = m->lo * M >= alo && m->lo * M <= ahi ? m->lo * M : alo;
	Mu = (uint32_t)(uint64_t)M;
	Au = (uint32_t)(uint64_t)(A - m->lo * M);
	for (i = 0; i < m->K; i++) {
		uint32_t xu = (uint32_t)(uint64_t)(m->lo + i);
		if ((int64_t)((uint32_t)(xu * Mu + Au) >> s) + base != m->v[i])
			return 0;
	}
	r = mk_bin(O_MUL, mk_cast(U32, m->x), mk_const_i(U32, (int64_t)Mu));
	if (Au)
		r = mk_bin(O_ADD, r, mk_const_i(U32, (int64_t)Au));
	to = mk_cast(t, mk_bin(O_SHR, r, mk_const_i(U32, s)));
	if (base)
		to = mk_bin(O_ADD, to, mk_const_i(t, base));
	snprintf(what, sizeof what, "the map %s of %s as a multiply and a "
	    "shift in u32", sstr(m->from), sstr(m->key));
	snprintf(cond, sizeof cond, "%s in [%lld, %lld] (%s)", sstr(m->key),
	    (long long)m->lo, (long long)m->hi, m->why);
	values_text(&pf, m);
	sb_printf(&pf, "\nwith i = %s%s, M = %lld and A = %lld: each value%s "
	    "w has w*2^%d <= i*M + A < (w + 1)*2^%d (the closest: %lld above at "
	    "%s = %lld, %lld below at %s = %lld), so (i*M + A) >> %d is w",
	    sstr(m->key), m->lo ? " - lo" : "", (long long)M, (long long)A,
	    base ? ", less the least one," : "", s, s, (long long)(A - alo),
	    sstr(m->key), (long long)(m->lo + ilo), (long long)(ahi - A),
	    sstr(m->key), (long long)(m->lo + ihi), s);
	sb_printf(&pf, "\ni*M + A stays in 0..%lld, inside u32: computed mod "
	    "2^32 from %s itself, as %s*0x%x", (long long)((wmax + 1) *
	    (1LL << s) - 1), sstr(m->key), sstr(m->key), Mu);
	if (Au)
		sb_printf(&pf, " + 0x%x", Au);
	sb_printf(&pf, ", it is the same number");
	if (m->key != m->x)
		to = xreplace(to, m->x, m->key);
	rw = irw_add(out, cnt, cap, root, m->from, to, what, cond, sb_str(&pf),
	    1);
	if (rw)
		rw->nosat = rw->derived = 1;
	free(pf.p);
	return 1;
}

/* what makes a subterm worth reading as a map: a decision or a quotient */
static int
map_seed(int op)
{
	return op == O_SEL || op == O_EQ || op == O_NE || op == O_LT ||
	    op == O_LE || op == O_GT || op == O_GE || op == O_LAND ||
	    op == O_LOR || op == O_DIV || op == O_MOD || op == O_INDEX;
}

struct mapinfo {
	uint64_t sup;		/* parameter slots it reads */
	uint8_t ok, seed, reach;
};

/* n's entry from its operands' (ids are in topological order) */
static void
map_info(nid n, struct mapinfo *mi, nid *var)
{
	nid kid[3];
	uint32_t j;
	int op = N.op[n], i;

	mi[n].ok = 1;
	if (op == O_CONST)
		return;
	if (op == O_VAR) {
		if (N.ki[n] < 0 || N.ki[n] >= MAXP)
			mi[n].ok = 0;
		else {
			mi[n].sup = 1ULL << N.ki[n];
			var[N.ki[n]] = n;
		}
		return;
	}
	if (op == O_UNDEF || op == O_LOOP || op == O_ATOMIC || op == O_BARRIER
	    || TY_FP(N.ty[n])) {
		mi[n].ok = 0;
		return;
	}
	mi[n].seed = (uint8_t)map_seed(op);
	if (op == O_TABLE) {
		for (j = 0; j < N.b[n]; j++) {
			nid e = N.lst[N.a[n] + j];
			mi[n].sup |= mi[e].sup;
			mi[n].ok &= mi[e].ok;
		}
		return;
	}
	kid[0] = N.a[n];
	kid[1] = N.b[n];
	kid[2] = N.c[n];
	for (i = 0; i < 3; i++)
		if (kid[i]) {
			mi[n].sup |= mi[kid[i]].sup;
			mi[n].ok &= mi[kid[i]].ok;
			mi[n].seed |= mi[kid[i]].seed;
		}
}

/* m: the map (s over x in lo..hi, from and key set); slot is x's */
static int
map_try(struct ivc *c, nid root, struct mapv *m, int slot, struct irw **out,
    int *cnt, int *cap)
{
	nid s = m->s, x = m->x;
	int added = 0;

	if (N.ty[x] == TY_BOOL || N.ty[x] == TY_LIT)
		return 0;
	/* literal arms: the map is computed as an int where it is the whole
	 * function (find searches in a machine type anyway) */
	m->t = N.ty[m->from];
	if (m->t == TY_LIT) {
		if (m->from != root)
			return 0;
		m->t = TY_INT;
	}
	if (m->hi < m->lo || m->hi - m->lo >= MAP_POINTS || m->hi - m->lo < 1 ||
	    m->lo <= -(1LL << 31) || m->hi >= (1LL << 32))
		return 0;
	m->K = (int)(m->hi - m->lo + 1);
	/* one compare of the variable: a threshold is already one operation */
	if ((opinfo[N.op[s]].fl & OF_CMP) && (N.a[s] == x || is_const(N.a[s])) &&
	    (N.b[s] == x || is_const(N.b[s])))
		return 0;
	if (map_values(c, m, slot) && m->vmin != m->vmax) {
		added |= map_lut(c, root, m, out, cnt, cap);
		/* a quotient of the variable itself is rw_divk's, and one
		 * by a power of two is already a shift */
		if (!((N.op[s] == O_DIV || N.op[s] == O_MOD) && is_const(N.b[s])
		    && (N.a[s] == x || pow2(N.ki[N.b[s]]))))
			added |= map_mulshift(c, root, m, out, cnt, cap);
	}
	free(m->v);
	m->v = NULL;
	return added;
}

/* n's operands onto the stack (a loop's are its own business) */
static uint32_t
map_push(nid n, nid *st, uint32_t sp)
{
	uint32_t j;

	if (N.op[n] == O_LOOP || N.op[n] == O_CONST || N.op[n] == O_VAR)
		return sp;
	if (N.op[n] == O_TABLE) {
		for (j = 0; j < N.b[n]; j++)
			st[sp++] = N.lst[N.a[n] + j];
		return sp;
	}
	if (N.a[n])
		st[sp++] = N.a[n];
	if (N.b[n])
		st[sp++] = N.b[n];
	if (N.c[n])
		st[sp++] = N.c[n];
	return sp;
}

/*
 * Maps whose key is not a parameter but a subterm with a small proved
 * range, (v >> 3) & 7: the largest subterms that read the parameters
 * only through that key are evaluated with a fresh variable in its
 * place, at each value of its range, and the table or the multiply-
 * shift is written back on the key itself.  Keys are taken innermost
 * first, so the largest map that fits is the one offered; a subterm
 * that already has a candidate is left alone.
 */
#define MAP_KEYS 24

static void
map_keys(struct ivc *c, nid root, const struct mapinfo *mi, uint8_t *took,
    nid *st, struct irw **out, int *cnt, int *cap)
{
	uint32_t lim = (uint32_t)root + 1, sp;
	uint8_t *fk, *seedk, *mark;
	nid k, n;
	int nkey = 0, slot = c->f->np;

	if (slot >= MAXP)
		return;		/* no parameter slot left for the key */
	fk = xmalloc(lim);
	seedk = xmalloc(lim);
	mark = xmalloc(lim);
	for (k = 1; k < root && nkey < MAP_KEYS; k++) {
		int64_t lo, hi;
		struct mapv m;
		nid xv;

		if (!mi[k].reach || !mi[k].ok || !mi[k].sup || N.op[k] == O_CONST
		    || N.op[k] == O_VAR || N.ty[k] == TY_BOOL || N.ty[k] == TY_LIT ||
		    TY_FP(N.ty[k]))
			continue;
		memset(&m, 0, sizeof m);
		if (!irw_range(c, k, INT64_MIN, &lo, &hi, m.why, sizeof m.why) ||
		    hi - lo < 1 || hi - lo >= MAP_POINTS)
			continue;
		nkey++;
		/* what is a function of k alone, above it */
		memset(fk, 0, lim);
		memset(seedk, 0, lim);
		memset(mark, 0, lim);
		fk[k] = 1;
		for (n = k + 1; n < lim; n++) {
			nid o[3] = {N.a[n], N.b[n], N.c[n]};
			int i, dep = 0, bad = 0, sd = map_seed(N.op[n]);
			if (!mi[n].reach || !mi[n].ok || N.op[n] == O_CONST ||
			    N.op[n] == O_VAR || N.op[n] == O_TABLE)
				continue;
			for (i = 0; i < 3; i++) {
				if (!o[i])
					continue;
				if (o[i] < lim && fk[o[i]]) {
					dep = 1;
					sd |= seedk[o[i]];
				} else if (!is_const(o[i]) && !(N.op[o[i]] == O_TABLE &&
				    !mi[o[i]].sup))
					bad = 1;	/* reads something else */
			}
			fk[n] = (uint8_t)(dep && !bad);
			seedk[n] = (uint8_t)(fk[n] && sd);
		}
		xv = 0;
		sp = 0;
		st[sp++] = root;
		while (sp) {
			n = st[--sp];
			if (n <= k || n >= lim || mark[n] || took[n])
				continue;
			mark[n] = 1;
			if (fk[n] && seedk[n]) {
				if (!xv)
					xv = mk_var((uint32_t)slot, sym_cstr("key"), N.ty[k]);
				m.from = n;
				m.key = k;
				m.x = xv;
				m.s = xreplace(n, k, xv);
				m.lo = lo;
				m.hi = hi;
				if (m.s && m.s != n && map_try(c, root, &m, slot, out,
				    cnt, cap)) {
					took[n] = 1;
					continue;
				}
			}
			sp = map_push(n, st, sp);
		}
	}
	free(fk);
	free(seedk);
	free(mark);
}

/* the largest subterms that read one small-range parameter and decide
 * or divide; where neither form fits, their operands in turn */
void
idrw_maps(struct ivc *c, nid root, struct irw **out, int *cnt, int *cap)
{
	uint32_t lim = N.n, sp = 0;
	struct mapinfo *mi = xcalloc(lim, sizeof *mi);
	uint8_t *mark = xcalloc(lim, 1), *took = xcalloc(lim, 1);
	nid *st = xmalloc(((size_t)lim * 3 + N.nlst + 4) * sizeof *st);
	nid var[MAXP] = {0}, n;

	st[sp++] = root;
	while (sp) {
		n = st[--sp];
		if (n == 0 || n >= lim || mi[n].reach)
			continue;
		mi[n].reach = 1;
		sp = map_push(n, st, sp);
	}
	for (n = 1; n < lim; n++)
		if (mi[n].reach)
			map_info(n, mi, var);
	st[sp++] = root;
	while (sp) {
		n = st[--sp];
		if (n == 0 || n >= lim || mark[n])
			continue;
		mark[n] = 1;
		if (mi[n].ok && mi[n].seed && mi[n].sup && !(mi[n].sup &
		    (mi[n].sup - 1))) {
			struct mapv m;
			int slot = 0;
			while (!((mi[n].sup >> slot) & 1))
				slot++;
			memset(&m, 0, sizeof m);
			m.s = m.from = n;
			m.x = m.key = var[slot];
			m.lo = c->g.p[slot].lo;
			m.hi = c->g.p[slot].hi;
			snprintf(m.why, sizeof m.why, "its range");
			if (c->g.p[slot].ranged && map_try(c, root, &m, slot, out,
			    cnt, cap)) {
				took[n] = 1;
				continue;
			}
		}
		sp = map_push(n, st, sp);
	}
	map_keys(c, root, mi, took, st, out, cnt, cap);
	free(mi);
	free(mark);
	free(took);
	free(st);
}
