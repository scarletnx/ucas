/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Packed integer dot products, as SPV_KHR_integer_dot_product defines
 * them (OpUDot, OpSDot, OpSUDot and their AccSat forms), with the lanes
 * packed in a 32-bit word:
 *
 *   4 x 8 bits   PackedVectorFormat4x8Bit (DP4A on NVIDIA sm_61 and later,
 *                vpdpbusd on x86 with VNNI: signed x unsigned)
 *   2 x 16 bits  a 2-component vector of 16-bit integers (vpdpwssd)
 *   8 x 4 bits   int4 lanes (AMD v_dot8_u32_u4 / v_dot8_i32_i4); SPIR-V
 *                has no form for these, so they stay in expressions
 *
 * Lane k of a word w is bits [k*bits, (k+1)*bits), zero-extended for an
 * unsigned operand, sign-extended for a signed one; su: the first
 * operand's lanes are signed, the second's unsigned.  The result is u32
 * when both operands are unsigned, i32 otherwise.
 *
 *   udot4x8(a, b)        = sum_k a_k * b_k, its low 32 bits ("the
 *                          low-order N bits of the correct result")
 *   udot4x8sat(a, b, c)  = the exact sum plus c, clamped to the result's
 *                          range (c has the result's type)
 *
 * so for 4x8 and 8x4 the plain forms never wrap (|sum| <= 4 * 255^2 =
 * 260100 and 8 * 15^2); 2 x 16 can (2 * 65535^2 > 2^32).
 *
 * The second half is the search: sums of products laid into the lanes of
 * the forms the active cost model offers (dot_find).
 */
#include "calc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const struct dotform forms[O_DOT_LAST - O_DOT_FIRST + 1] = {
	{"udot4x8", 4, 8, 0, 0, 0}, {"sdot4x8", 4, 8, 1, 1, 0},
	{"sudot4x8", 4, 8, 1, 0, 0}, {"udot2x16", 2, 16, 0, 0, 0},
	{"sdot2x16", 2, 16, 1, 1, 0}, {"sudot2x16", 2, 16, 1, 0, 0},
	{"udot8x4", 8, 4, 0, 0, 0}, {"sdot8x4", 8, 4, 1, 1, 0},
	{"sudot8x4", 8, 4, 1, 0, 0},
	{"udot4x8sat", 4, 8, 0, 0, 1}, {"sdot4x8sat", 4, 8, 1, 1, 1},
	{"sudot4x8sat", 4, 8, 1, 0, 1}, {"udot2x16sat", 2, 16, 0, 0, 1},
	{"sdot2x16sat", 2, 16, 1, 1, 1}, {"sudot2x16sat", 2, 16, 1, 0, 1},
	{"udot8x4sat", 8, 4, 0, 0, 1}, {"sdot8x4sat", 8, 4, 1, 1, 1},
	{"sudot8x4sat", 8, 4, 1, 0, 1},
};

const struct dotform *
dot_form(int op)
{
	return IS_DOT(op) ? &forms[op - O_DOT_FIRST] : NULL;
}

/* the op for a form; s x u has no op of its own (swap the operands) */
int
dot_op(int lanes, int bits, int sa, int sb, int sat)
{
	int k;

	for (k = 0; k <= O_DOT_LAST - O_DOT_FIRST; k++)
		if (forms[k].lanes == lanes && forms[k].bits == bits &&
		    forms[k].sa == sa && forms[k].sb == sb && forms[k].sat == sat)
			return O_DOT_FIRST + k;
	return 0;
}

static int64_t
lane(uint64_t w, int k, int bits, int s)
{
	uint64_t m = (1ULL << bits) - 1, v = (w >> (k * bits)) & m;

	if (s && (v >> (bits - 1)))
		return (int64_t)v - (int64_t)(1ULL << bits);
	return (int64_t)v;
}

int64_t
dot_scalar(int op, int ideal, int64_t a, int64_t b, int64_t acc,
    uint8_t *fl)
{
	const struct dotform *df = dot_form(op);
	uint64_t ua = (uint64_t)a & 0xffffffffULL, ub = (uint64_t)b & 0xffffffffULL;
	int64_t sum = 0, lo, hi;
	int k, sres = df->sa || df->sb;

	for (k = 0; k < df->lanes; k++)
		sum += lane(ua, k, df->bits, df->sa) * lane(ub, k, df->bits, df->sb);
	lo = sres ? -(int64_t)0x80000000LL : 0;
	hi = sres ? 0x7fffffffLL : 0xffffffffLL;
	if (df->sat) {
		sum += acc;
		return sum < lo ? lo : sum > hi ? hi : sum;
	}
	if (ideal || (sum >= lo && sum <= hi))
		return sum;
	*fl |= sres ? F_SW : F_UW;
	sum &= 0xffffffffLL;
	return sres && sum > hi ? sum - 0x100000000LL : sum;
}

/*
 * Lane k's range from the word's [lo, hi] (as a u32): the top lane is
 * monotone in the word; a lower one is when the bits above it are the
 * same at lo and hi, otherwise it can be anything.
 */
static void
lane_range(int64_t lo, int64_t hi, int k, int lanes, int bits, int s,
    int64_t *l, int64_t *h)
{
	int64_t m = ((int64_t)1 << bits) - 1, half = (int64_t)1 << (bits - 1);
	int sh = k * bits;

	if (lo < 0 || hi > 0xffffffffLL || lo > hi) {
		*l = 0;
		*h = m;
	} else if (k == lanes - 1 || (hi >> (sh + bits)) == (lo >> (sh + bits))) {
		*l = (lo >> sh) & m;
		*h = (hi >> sh) & m;
	} else {
		*l = 0;
		*h = m;
	}
	if (!s)
		return;
	if (*h < half)
		return;
	if (*l >= half) {
		*l -= m + 1;
		*h -= m + 1;
		return;
	}
	*l = -half;
	*h = half - 1;
}

int
dot_ival(int op, int64_t alo, int64_t ahi, int64_t blo, int64_t bhi,
    int64_t clo, int64_t chi, int64_t *lo, int64_t *hi)
{
	const struct dotform *df = dot_form(op);
	int64_t slo = 0, shi = 0, rlo, rhi;
	int k, j, sres = df->sa || df->sb;

	for (k = 0; k < df->lanes; k++) {
		int64_t al, ah, bl, bh, x[4], pl, ph;
		lane_range(alo, ahi, k, df->lanes, df->bits, df->sa, &al, &ah);
		lane_range(blo, bhi, k, df->lanes, df->bits, df->sb, &bl, &bh);
		x[0] = al * bl;
		x[1] = al * bh;
		x[2] = ah * bl;
		x[3] = ah * bh;
		pl = ph = x[0];
		for (j = 1; j < 4; j++) {
			if (x[j] < pl)
				pl = x[j];
			if (x[j] > ph)
				ph = x[j];
		}
		slo += pl;
		shi += ph;
	}
	rlo = sres ? -(int64_t)0x80000000LL : 0;
	rhi = sres ? 0x7fffffffLL : 0xffffffffLL;
	if (df->sat) {
		slo += clo;
		shi += chi;
		*lo = slo < rlo ? rlo : slo > rhi ? rhi : slo;
		*hi = shi < rlo ? rlo : shi > rhi ? rhi : shi;
		return 1;
	}
	*lo = slo;
	*hi = shi;
	return slo >= rlo && shi <= rhi;
}

/* ---- the search ---- */

/*
 * A sum is normalised to sum_i s_i f_i g_i + c (s_i = +-1): through +,
 * - and unary -, and through conversions that cannot change the value
 * mod 2^W of the sum's type (narrowing ones, and widening ones of a
 * subterm the intervals show does not wrap).  A product's factors are
 * its operands; t << k is t * 2^k; any other term is t * 1.  In a type
 * of at most 32 bits everything is mod 2^W, so wrapping inside the sum
 * is harmless; a wider type needs the exact values (no wrap anywhere).
 *
 * A factor is a lane value when its proved range fits a lane: [0, 2^b)
 * for unsigned lanes, [-2^(b-1), 2^(b-1)) for signed.  A bit field of a
 * word, (w >> b*k) & (2^b - 1) (or w >> b*k reaching w's top, or a
 * conversion to ub), is already lane k of w: the dot takes w itself, and
 * the other side's zero lanes cancel w's other lanes.  Other values are
 * packed: a constant into a constant word, a value into lane 0 as it
 * is, into lane k by a shift and an or (each priced).
 *
 * Groups of terms are laid into the forms the active model has; a
 * rewrite is kept when the whole function is cheaper under the model,
 * and then proved (equiv: exhaustive or SAT).
 */

#define MAXTERM 12
#define MAXGROUP 4
#define MAXROOT 16

struct fac {
	nid n;
	int64_t lo, hi;
	int ok;              /* a proved range */
	int isk;
	int64_t k;
	nid word;            /* a bit field: (word >> sh) & (2^w - 1) */
	int sh, w;
};

struct term {
	nid n;               /* as written */
	int sign;
	int prod;            /* has two factors */
	struct fac f[2];
};

struct sum {
	nid root;
	ty_t t;
	int exact;           /* a type wider than 32 bits: no wrap allowed */
	struct term tm[MAXTERM];
	int nt;
	int64_t k;           /* the constant part */
	int ok;
};

/* one group of terms in one dot */
struct group {
	int op;              /* the dot op */
	int swap;            /* operands of the op are (B, A) */
	uint32_t set;        /* terms */
	uint32_t orient;     /* bit i: term i's second factor on side A */
	int lane[MAXTERM];
	nid wa, wb;          /* the words */
	int moda, modb;      /* side is a word's lanes (1) or built (0) */
	int sign;
	uint32_t neg;        /* terms of the other sign: their constant negated */
	int spare;           /* lanes no term uses */
	int64_t slo, shi;    /* the exact sum's range */
};

static struct {
	const struct func *f;
	struct plan p;
	struct ival *iv;
	struct model *m;
} S;

static int
iv_of(nid n, int64_t *lo, int64_t *hi, int *wrap)
{
	uint32_t l;

	if (is_const(n) && TY_ISINT(N.ty[n])) {
		*lo = *hi = N.ki[n];
		*wrap = 0;
		return 1;
	}
	l = plan_loc(&S.p, n);
	if (l == NOTOK || S.iv[l].top)
		return 0;
	*lo = S.iv[l].lo;
	*hi = S.iv[l].hi;
	*wrap = S.iv[l].wrap;
	return 1;
}

static int
mwidth(ty_t t)
{
	return TYK(t) == K_UINT || TYK(t) == K_SINT ? TYW(t) : 0;
}

/* (w >> sh) & (2^wd - 1), possibly through conversions */
static void
field_of(struct fac *x)
{
	nid n = x->n, w;
	int mw = 64, s = 0;
	int64_t m;

	while (N.op[n] == O_CAST && TY_ISINT(N.ty[N.a[n]]) &&
	    mwidth(N.ty[N.a[n]]) && mwidth(N.ty[n])) {
		ty_t from = N.ty[N.a[n]], to = N.ty[n];
		if (TYW(to) < TYW(from) && TYK(to) == K_UINT) {
			if (TYW(to) < mw)
				mw = TYW(to);
		} else if (!(TYW(to) >= TYW(from) && (TYK(from) == K_UINT ||
		    TYK(to) == K_SINT)))
			return;
		n = N.a[n];
	}
	if (N.op[n] == O_AND && (is_const(N.b[n]) || is_const(N.a[n]))) {
		nid kk = is_const(N.b[n]) ? N.b[n] : N.a[n];
		m = N.ki[kk];
		if (m <= 0 || (m & (m + 1)))
			return;
		if (bits_for((uint64_t)m) < mw)
			mw = bits_for((uint64_t)m);
		n = kk == N.b[n] ? N.a[n] : N.b[n];
	}
	if (N.op[n] == O_SHR && is_const(N.b[n]) && mwidth(N.ty[n])) {
		w = N.a[n];
		s = (int)N.ki[N.b[n]];
		if (s <= 0 || s >= mwidth(N.ty[w]))
			return;
		if (TYK(N.ty[w]) == K_SINT && s + mw > mwidth(N.ty[w]))
			return;   /* an arithmetic shift brings in sign bits */
	} else if (mw < 64 && mwidth(N.ty[n])) {
		w = n;
		s = 0;
	} else
		return;
	if (mwidth(N.ty[w]) == 0 || mwidth(N.ty[w]) > 32)
		return;
	if (mw > mwidth(N.ty[w]) - s)
		mw = mwidth(N.ty[w]) - s;
	x->word = w;
	x->sh = s;
	x->w = mw;
}

static void
fac_init(struct fac *x, nid n)
{
	int wrap;

	memset(x, 0, sizeof *x);
	x->n = n;
	if (is_const(n) && TY_ISINT(N.ty[n])) {
		x->isk = 1;
		x->k = N.ki[n];
	}
	x->ok = TY_ISINT(N.ty[n]) && iv_of(n, &x->lo, &x->hi, &wrap);
	field_of(x);
}

static void
add_term(struct sum *s, nid n, int sign, nid fa, nid fb)
{
	struct term *t;

	if (s->nt == MAXTERM) {
		s->ok = 0;
		return;
	}
	t = &s->tm[s->nt++];
	memset(t, 0, sizeof *t);
	t->n = n;
	t->sign = sign;
	t->prod = fb != 0;
	fac_init(&t->f[0], fa);
	if (fb)
		fac_init(&t->f[1], fb);
	else {
		memset(&t->f[1], 0, sizeof t->f[1]);
		t->f[1].isk = t->f[1].ok = 1;
		t->f[1].k = t->f[1].lo = t->f[1].hi = 1;
	}
}

/* exact: the subterm must not wrap (its arithmetic is not mod 2^W) */
static void
collect(struct sum *s, nid n, int sign, int exact)
{
	int op = N.op[n], wrap = 0;
	int64_t lo, hi;

	if (!s->ok)
		return;
	if (!TY_ISINT(N.ty[n]) || N.ty[n] == TY_BOOL) {
		s->ok = 0;
		return;
	}
	if (op == O_CONST) {
		s->k += sign * N.ki[n];
		return;
	}
	if ((op == O_ADD || op == O_SUB || op == O_NEG || op == O_MUL ||
	    op == O_SHL) && (exact || !mwidth(N.ty[n]))) {
		if (!iv_of(n, &lo, &hi, &wrap) || wrap) {
			add_term(s, n, sign, n, 0);
			return;
		}
	}
	switch (op) {
	case O_ADD:
		collect(s, N.a[n], sign, exact);
		collect(s, N.b[n], sign, exact);
		return;
	case O_SUB:
		collect(s, N.a[n], sign, exact);
		collect(s, N.b[n], -sign, exact);
		return;
	case O_NEG:
		collect(s, N.a[n], -sign, exact);
		return;
	case O_MUL:
		add_term(s, n, sign, N.a[n], N.b[n]);
		return;
	case O_SHL:
		if (is_const(N.b[n]) && N.ki[N.b[n]] >= 0 && N.ki[N.b[n]] < 31) {
			add_term(s, n, sign, N.a[n], mk_const_i(N.ty[n],
			    (int64_t)1 << N.ki[N.b[n]]));
			return;
		}
		break;
	case O_CAST: {
		ty_t from = N.ty[N.a[n]], to = N.ty[n];
		if (!TY_ISINT(from) || from == TY_BOOL)
			break;
		if (mwidth(to) && mwidth(from) >= mwidth(to)) {
			/* narrowing: mod 2^W of the inner value */
			collect(s, N.a[n], sign, exact);
			return;
		}
		if (iv_of(N.a[n], &lo, &hi, &wrap) && !wrap) {
			struct ity it;
			ity_of(&it, to, 0);
			if (!mwidth(to) || (it.k == K_UINT ? lo >= 0 &&
			    (uint64_t)hi <= it.mask : lo >= it.smin &&
			    hi <= it.smax)) {
				collect(s, N.a[n], sign, 1);
				return;
			}
		}
		break;
	}
	}
	add_term(s, n, sign, n, 0);
}

/* the range of lane values a factor allows: unsigned (1) and/or signed (2) */
static int
fac_signs(const struct fac *x, int b)
{
	int64_t lo = x->isk ? x->k : x->lo, hi = x->isk ? x->k : x->hi;
	int r = 0;

	if (!x->ok)
		return 0;
	if (lo >= 0 && hi < ((int64_t)1 << b))
		r |= 1;
	if (lo >= -((int64_t)1 << (b - 1)) && hi < ((int64_t)1 << (b - 1)))
		r |= 2;
	return r;
}

/* the factor is lane sh/b of its word, exactly */
static int
fac_lane(const struct fac *x, int L, int b)
{
	nid w = x->word;

	if (!w || x->sh % b || x->sh / b >= L)
		return -1;
	if (x->w == b)
		return x->sh / b;
	if (x->w < b && x->sh + x->w == mwidth(N.ty[w]) &&
	    TYK(N.ty[w]) == K_UINT)
		return x->sh / b;
	return -1;
}

/* a word's lane k is 0 (by the word's range) */
static int
lane_zero(nid w, int k, int L, int b)
{
	int64_t lo, hi, l, h;
	int wrap;

	if (!iv_of(w, &lo, &hi, &wrap) || lo < 0)
		return 0;
	lane_range(lo, hi, k, L, b, 0, &l, &h);
	return l == 0 && h == 0;
}

static nid
ku32(int64_t v)
{
	return mk_const_i(TY(K_UINT, 32), (int64_t)((uint64_t)v & 0xffffffffULL));
}

/*
 * Lay out the terms in set (orientation orient) into one dot of L lanes
 * of b bits; fills g, 0 when they do not fit or the model has no form.
 */
static int
layout(const struct sum *s, uint32_t set, uint32_t orient, int L, int b,
    int sat, struct group *g)
{
	const struct fac *side[2][MAXTERM];
	struct fac negk[MAXTERM];
	int idx[MAXTERM], n = 0, i, k, sd, used[8] = {0};
	int signs[2] = {3, 3}, mode[2];
	nid word[2] = {0, 0};
	int64_t slo = 0, shi = 0;
	int gsign = set ? s->tm[__builtin_ctz(set)].sign : 1, dirty = 0;

	memset(g, 0, sizeof *g);
	for (i = 0; i < s->nt; i++)
		if ((set >> i) & 1) {
			int o = (orient >> i) & 1;
			idx[n] = i;
			side[0][n] = &s->tm[i].f[o];
			side[1][n] = &s->tm[i].f[1 - o];
			if (s->tm[i].sign != gsign) {
				/* - f * K is f * -K */
				int q = side[1][n]->isk ? 1 : side[0][n]->isk ? 0 : -1;
				if (q < 0)
					return 0;
				negk[n] = *side[q][n];
				negk[n].k = negk[n].lo = negk[n].hi = -negk[n].k;
				side[q][n] = &negk[n];
				g->neg |= 1u << i;
			}
			n++;
		}
	if (n == 0 || n > L)
		return 0;
	/* a side is a word's lanes when every factor on it is a lane of
	 * the same word; otherwise it is built from the values */
	for (sd = 0; sd < 2; sd++) {
		mode[sd] = 1;
		for (i = 0; i < n; i++) {
			if (fac_lane(side[sd][i], L, b) < 0 ||
			    (word[sd] && side[sd][i]->word != word[sd])) {
				mode[sd] = 0;
				break;
			}
			word[sd] = side[sd][i]->word;
		}
		if (!mode[sd])
			word[sd] = 0;
	}
	for (i = 0; i < n; i++) {
		int la = mode[0] ? fac_lane(side[0][i], L, b) : -1;
		int lb = mode[1] ? fac_lane(side[1][i], L, b) : -1;
		if (la >= 0 && lb >= 0 && la != lb)
			return 0;
		g->lane[idx[i]] = la >= 0 ? la : lb;
		if (g->lane[idx[i]] >= 0) {
			if (used[g->lane[idx[i]]])
				return 0;
			used[g->lane[idx[i]]] = 1;
		}
	}
	for (i = 0; i < n; i++)
		if (g->lane[idx[i]] < 0) {
			for (k = 0; k < L && used[k]; k++)
				;
			if (k == L)
				return 0;
			g->lane[idx[i]] = k;
			used[k] = 1;
		}
	/* both sides words: the unused lanes must be zero on one of them */
	if (mode[0] && mode[1])
		for (k = 0; k < L; k++)
			if (!used[k] && !lane_zero(word[0], k, L, b) &&
			    !lane_zero(word[1], k, L, b))
				return 0;
	/* signedness: what every factor on the side allows */
	for (sd = 0; sd < 2; sd++)
		for (i = 0; i < n; i++) {
			const struct fac *x = side[sd][i];
			int r = fac_signs(x, b);
			if (mode[sd])
				r &= 1 | (x->hi < ((int64_t)1 << (b - 1)) ? 2 : 0);
			signs[sd] &= r;
		}
	if (!signs[0] || !signs[1])
		return 0;
	/* the cheapest op the model has (s x u is su with the sides swapped) */
	{
		float best = INFINITY;
		int sa, sb;
		for (sa = 0; sa < 2; sa++)
			for (sb = 0; sb < 2; sb++) {
				int op, sw = 0;
				if (!((signs[0] >> sa) & 1) || !((signs[1] >> sb) & 1))
					continue;
				if (!sa && sb) {
					op = dot_op(L, b, 1, 0, sat);
					sw = 1;
				} else
					op = dot_op(L, b, sa, sb, sat);
				if (!op || !cclass_available(S.m, DOT_CLASS(op)) ||
				    !(S.m->c[DOT_CLASS(op)] < best))
					continue;
				best = S.m->c[DOT_CLASS(op)];
				g->op = op;
				g->swap = sw;
			}
		if (!g->op)
			return 0;
	}
	/* the words */
	for (sd = 0; sd < 2; sd++) {
		int64_t kw = 0;
		nid acc = 0;
		int sgn;
		if (g->swap)
			sgn = sd == 0 ? dot_form(g->op)->sb : dot_form(g->op)->sa;
		else
			sgn = sd == 0 ? dot_form(g->op)->sa : dot_form(g->op)->sb;
		if (mode[sd]) {
			if (sd == 0)
				g->wa = word[0], g->moda = 1;
			else
				g->wb = word[1], g->modb = 1;
			continue;
		}
		for (i = 0; i < n; i++) {
			const struct fac *x = side[sd][i];
			int ln = g->lane[idx[i]];
			int64_t m = ((int64_t)1 << b) - 1;
			nid e;
			if (x->isk) {
				kw |= (x->k & m) << (b * ln);
				continue;
			}
			e = mk_cast(TY(K_UINT, 32), x->n);
			if (x->lo < 0 && sgn) {
				/* a negative value's sign bits fill the lanes
				 * above it: masked unless nothing above it is
				 * used on either side */
				int above = 0, j;
				for (j = 0; j < n; j++)
					if (g->lane[idx[j]] > ln)
						above = 1;
				if (mode[1 - sd]) {
					for (j = ln + 1; j < L; j++)
						if (!lane_zero(word[1 - sd], j, L, b))
							above = 1;
				} else if (sd == 1)
					above |= dirty;	/* side A left its sign bits */
				else
					for (j = 0; j < n; j++)
						if (!side[1][j]->isk && side[1][j]->lo < 0)
							above = 1;
				if (above)
					e = mk_bin(O_AND, e, ku32(m));
				else if (sd == 0)
					dirty = 1;
			}
			if (ln)
				e = mk_bin(O_SHL, e, ku32(b * ln));
			acc = acc ? mk_bin(O_OR, acc, e) : e;
		}
		if (kw)
			acc = acc ? mk_bin(O_OR, acc, ku32(kw)) : ku32(kw);
		if (!acc)
			acc = ku32(0);
		if (sd == 0)
			g->wa = acc;
		else
			g->wb = acc;
	}
	if (is_const(g->wa) && is_const(g->wb))
		return 0;
	/* the exact sum's range, lane by lane */
	for (i = 0; i < n; i++) {
		const struct fac *x = side[0][i], *y = side[1][i];
		int64_t xl = x->isk ? x->k : x->lo, xh = x->isk ? x->k : x->hi;
		int64_t yl = y->isk ? y->k : y->lo, yh = y->isk ? y->k : y->hi;
		int64_t p[4] = {xl * yl, xl * yh, xh * yl, xh * yh}, pl, ph;
		int j;
		pl = ph = p[0];
		for (j = 1; j < 4; j++) {
			if (p[j] < pl)
				pl = p[j];
			if (p[j] > ph)
				ph = p[j];
		}
		slo += pl;
		shi += ph;
	}
	g->slo = slo;
	g->shi = shi;
	g->set = set;
	g->orient = orient;
	g->sign = gsign;
	for (k = 0; k < L; k++)
		if (!used[k])
			g->spare = 1;
	return 1;
}

static nid
group_dot(const struct group *g, nid acc)
{
	const struct dotform *df = dot_form(g->op);
	int sw = g->swap;

	/* a constant word second when the order does not matter */
	if (df->sa == df->sb && is_const(g->wa) && !is_const(g->wb))
		sw = !sw;
	return sw ? mk_dot(g->op, g->wb, g->wa, acc) :
	    mk_dot(g->op, g->wa, g->wb, acc);
}

/* the rest of the sum: the terms in no group, and the constant */
static nid
rest_of(const struct sum *s, uint32_t taken, int64_t *lo, int64_t *hi,
    int *exact)
{
	nid r = mk_const_i(s->t, int_norm(s->t, s->k));
	int i, wrap;
	int64_t l, h;

	*lo = *hi = s->k;
	*exact = 1;
	for (i = 0; i < s->nt; i++) {
		if ((taken >> i) & 1)
			continue;
		r = mk_bin(s->tm[i].sign > 0 ? O_ADD : O_SUB, r,
		    mk_cast(s->t, s->tm[i].n));
		if (!iv_of(s->tm[i].n, &l, &h, &wrap) || wrap)
			*exact = 0;
		else if (s->tm[i].sign > 0)
			*lo += l, *hi += h;
		else
			*lo -= h, *hi -= l;
	}
	if (*exact && mwidth(s->t)) {
		struct ity it;
		ity_of(&it, s->t, 0);
		if (it.k == K_UINT ? *lo < 0 || (uint64_t)*hi > it.mask :
		    *lo < it.smin || *hi > it.smax)
			*exact = 0;
	}
	return r;
}

/* the sum with its groups as dots: plain, or chained through _SAT forms
 * when no partial sum can saturate */
static nid
rebuild_sum(const struct sum *s, const struct group *g, int ng, int sat,
    char *cond, size_t ncond)
{
	uint32_t taken = 0;
	int i, ex;
	int64_t lo, hi;
	nid r;

	for (i = 0; i < ng; i++)
		taken |= g[i].set;
	r = rest_of(s, taken, &lo, &hi, &ex);
	if (cond)
		cond[0] = 0;
	if (sat) {
		/* acc must be the exact rest, every partial sum in range */
		for (i = 0; i < ng; i++) {
			const struct dotform *df = dot_form(g[i].op);
			int64_t rlo = df->sa || df->sb ? -(int64_t)0x80000000LL : 0;
			int64_t rhi = df->sa || df->sb ? 0x7fffffffLL : 0xffffffffLL;
			if (!ex || !df->sat || g[i].sign < 0 || lo < rlo || hi > rhi)
				return 0;
			lo += g[i].slo;
			hi += g[i].shi;
			if (lo < rlo || hi > rhi)
				return 0;
			r = group_dot(&g[i], mk_cast(df->sa || df->sb ?
			    TY(K_SINT, 32) : TY(K_UINT, 32), r));
		}
		if (cond)
			snprintf(cond, ncond, "every partial sum, accumulator "
			    "included, is in %lld..%lld: nothing saturates",
			    (long long)lo, (long long)hi);
		return mk_cast(s->t, r);
	}
	{
		nid d = 0;
		for (i = 0; i < ng; i++) {
			nid x = mk_cast(s->t, group_dot(&g[i], 0));
			if (dot_form(g[i].op)->sat)
				return 0;
			if (s->exact) {
				/* the dot's exact sum must fit its result */
				const struct dotform *df = dot_form(g[i].op);
				if (g[i].slo < (df->sa || df->sb ?
				    -(int64_t)0x80000000LL : 0) || g[i].shi >
				    (df->sa || df->sb ? 0x7fffffffLL : 0xffffffffLL))
					return 0;
			}
			if (!d)
				d = g[i].sign > 0 ? x : mk_un(O_NEG, x);
			else
				d = mk_bin(g[i].sign > 0 ? O_ADD : O_SUB, d, x);
		}
		if (cond) {
			struct sbuf b = {0};
			for (i = 0; i < ng; i++)
			{
				const struct dotform *df = dot_form(g[i].op);
				int sr = df->sa || df->sb;
				int fits = g[i].slo >= (sr ? -(int64_t)0x80000000LL :
				    0) && g[i].shi <= (sr ? 0x7fffffffLL :
				    0xffffffffLL);
				sb_printf(&b, "%s%s's sum is in %lld..%lld: %s", i ?
				    "; " : "", df->name, (long long)g[i].slo,
				    (long long)g[i].shi, fits ? "it does not wrap" :
				    "it wraps mod 2^32 as the sum does");
			}
			snprintf(cond, ncond, "%s", sb_str(&b));
			free(b.p);
		}
		return is_const(r) && N.ki[r] == 0 ? d : mk_bin(O_ADD, d, r);
	}
}

/* the cost of a DAG under the model, each node once */
static float
price_dag(nid e)
{
	uint32_t cap = 256, n = 0, sp = 0, j;
	nid *seen = xcalloc(cap, sizeof *seen), *st = NULL, x;
	uint32_t stcap = 0;
	float c = 0;
	int cls;

#define PUSH(v) do { nid v_ = (v); uint32_t h_; \
	if (!v_) break; \
	if ((n + 1) * 2 > cap) { nid *o_ = seen; uint32_t oc_ = cap, q_; \
		cap *= 2; seen = xcalloc(cap, sizeof *seen); \
		for (q_ = 0; q_ < oc_; q_++) if (o_[q_]) { \
			h_ = (o_[q_] * 2654435761u) & (cap - 1); \
			while (seen[h_]) h_ = (h_ + 1) & (cap - 1); \
			seen[h_] = o_[q_]; } free(o_); } \
	h_ = (v_ * 2654435761u) & (cap - 1); \
	while (seen[h_] && seen[h_] != v_) h_ = (h_ + 1) & (cap - 1); \
	if (seen[h_]) break; \
	seen[h_] = v_; n++; \
	if (sp == stcap) { stcap = stcap ? stcap * 2 : 64; \
		st = xrealloc(st, stcap * sizeof *st); } \
	st[sp++] = v_; } while (0)

	PUSH(e);
	while (sp) {
		x = st[--sp];
		c += node_cost(S.m, x, &cls);
		if (N.op[x] == O_CONST || N.op[x] == O_VAR || N.op[x] == O_UNDEF)
			continue;
		if (N.op[x] == O_TABLE) {
			for (j = 0; j < N.b[x]; j++)
				PUSH(N.lst[N.a[x] + j]);
			continue;
		}
		PUSH(N.a[x]);
		PUSH(N.b[x]);
		PUSH(N.c[x]);
	}
#undef PUSH
	free(seen);
	free(st);
	return c;
}

struct cand {
	nid body;
	float cost;
	struct sum *s;
	struct group g[MAXGROUP];
	int ng, sat;
	char cond[256];
};

static void
consider(struct cand *best, const struct func *f, struct sum *s,
    const struct group *g, int ng)
{
	int sat;

	for (sat = 0; sat < 2; sat++) {
		char cond[256];
		nid r = rebuild_sum(s, g, ng, sat, cond, sizeof cond), body;
		float c;
		if (!r)
			continue;
		body = xreplace(f->body, s->root, mk_cast(N.ty[s->root], r));
		if (!body)
			continue;
		c = price_dag(body);
		if (c < best->cost) {
			best->body = body;
			best->cost = c;
			best->s = s;
			memcpy(best->g, g, (size_t)ng * sizeof *g);
			best->ng = ng;
			best->sat = sat;
			snprintf(best->cond, sizeof best->cond, "%s", cond);
		}
	}
}

/* the forms of the model: (lanes, bits) with any op available */
static int
model_formats(int (*fmt)[2])
{
	static const int all[3][2] = {{4, 8}, {2, 16}, {8, 4}};
	int i, op, n = 0;

	for (i = 0; i < 3; i++)
		for (op = O_DOT_FIRST; op <= O_DOT_LAST; op++)
			if (dot_form(op)->lanes == all[i][0] &&
			    cclass_available(S.m, DOT_CLASS(op))) {
				fmt[n][0] = all[i][0];
				fmt[n][1] = all[i][1];
				n++;
				break;
			}
	return n;
}

/*
 * Greedy over groups: the cheapest single group of any format, then
 * another among the terms left, while the whole gets cheaper.
 */
static void
search_sum(struct cand *best, const struct func *f, struct sum *s,
    int (*fmt)[2], int nfmt)
{
	struct group g[MAXGROUP];
	int ng = 0;
	uint32_t taken = 0;

	while (ng < MAXGROUP) {
		struct cand here = *best;
		struct group bestg;
		int found = 0, fi, sat;
		uint32_t set;
		for (fi = 0; fi < nfmt; fi++)
			for (sat = 0; sat < 2; sat++)
			for (set = 1; set < (1u << s->nt); set++) {
				int pc = __builtin_popcount(set), j;
				uint32_t o, no;
				if ((set & taken) || pc > fmt[fi][0])
					continue;
				for (j = 0; j < s->nt; j++)
					if (((set >> j) & 1) && (!s->tm[j].f[0].ok ||
					    !s->tm[j].f[1].ok))
						break;
				if (j < s->nt)
					continue;
				no = pc <= 6 ? 1u << pc : 1;
				for (o = 0; o < no; o++) {
					uint32_t orient = 0;
					int q = 0;
					for (j = 0; j < s->nt; j++)
						if ((set >> j) & 1)
							orient |= ((o >> q++) & 1u) << j;
					if (!layout(s, set, orient, fmt[fi][0], fmt[fi][1],
					    sat, &g[ng]))
						continue;
					{
						float before = here.cost;
						consider(&here, f, s, g, ng + 1);
						if (here.cost < before) {
							bestg = g[ng];
							found = 1;
						}
					}
				}
			}
		if (!found)
			break;
		g[ng++] = bestg;
		taken |= bestg.set;
		*best = here;
	}
}

static void
fac_text(struct sbuf *b, const struct fac *x)
{
	int op = N.op[x->n];

	if (x->isk)
		sb_printf(b, "%lld", (long long)x->k);
	else if (op == O_VAR || op == O_CONST || op == O_CAST)
		sb_puts(b, shared_str(x->n, 60));
	else
		sb_printf(b, "(%s)", shared_str(x->n, 60));
}

/* what the rewrite relies on, for the report */
static void
describe(struct sbuf *b, const struct cand *c)
{
	int i, j;

	for (i = 0; i < c->ng; i++) {
		const struct group *g = &c->g[i];
		const struct dotform *df = dot_form(g->op);
		sb_printf(b, "%s%s (%d x %d-bit lanes, %s): ", i ? "\n" : "",
		    df->name, df->lanes, df->bits, df->sa && df->sb ? "signed" :
		    df->sa ? "signed x unsigned" : "unsigned");
		int first = 1;
		for (j = 0; j < c->s->nt; j++) {
			const struct term *t = &c->s->tm[j];
			int o = (g->orient >> j) & 1;
			if (!((g->set >> j) & 1))
				continue;
			sb_printf(b, "%slane %d: ", first ? "" : ", ", g->lane[j]);
			first = 0;
			if (t->f[o].isk && !t->f[1 - o].isk)
				o = 1 - o;
			fac_text(b, &t->f[o]);
			sb_puts(b, " * ");
			if ((g->neg >> j) & 1)
				sb_printf(b, "%lld", (long long)-t->f[1 - o].k);
			else
				fac_text(b, &t->f[1 - o]);
		}
	}
	for (i = 0; i < c->ng; i++) {
		const struct group *g = &c->g[i];
		for (j = 0; j < c->s->nt; j++) {
			const struct term *t = &c->s->tm[j];
			int o = (g->orient >> j) & 1, q;
			if (!((g->set >> j) & 1))
				continue;
			for (q = 0; q < 2; q++) {
				const struct fac *x = &t->f[q ? 1 - o : o];
				int fromword = (q == 0 ? g->moda : g->modb);
				if (x->isk)
					continue;
				char line[300];
				if (fromword)
					snprintf(line, sizeof line, "\n%s is bits %d..%d of "
					    "%s: lane %d as it is", shared_str(x->n, 60),
					    x->sh, x->sh + x->w - 1, shared_str(x->word, 40),
					    g->lane[j]);
				else
					snprintf(line, sizeof line, "\n%s in %lld..%lld "
					    "(proved range)", shared_str(x->n, 60),
					    (long long)x->lo, (long long)x->hi);
				if (!strstr(sb_str(b), line))
					sb_puts(b, line);
			}
		}
		if ((g->moda || g->modb) && g->spare)
			sb_printf(b, "\nthe word's other lanes meet zero lanes");
	}
	if (c->cond[0])
		sb_printf(b, "\n%s", c->cond);
}

/* the sums worth trying: maximal +, -, * subterms (and the body) */
static int
roots_of(nid body, nid *out, int max)
{
	uint32_t cap = 1024, sp = 0, n = 0, i;
	nid *st = xmalloc(cap * sizeof *st), x;
	uint8_t *seen = xcalloc(N.n, 1), *inner = xcalloc(N.n, 1);

	st[sp++] = body;
	seen[body] = 1;
	while (sp) {
		int op;
		x = st[--sp];
		op = N.op[x];
		if (op == O_CONST || op == O_VAR || op == O_UNDEF ||
		    op == O_TABLE || op == O_LOOP)
			continue;
		if (op == O_ADD || op == O_SUB || op == O_NEG) {
			inner[N.a[x]] = 1;
			if (N.b[x] && op != O_NEG)
				inner[N.b[x]] = 1;
		}
		for (i = 0; i < 3; i++) {
			nid y = i == 0 ? N.a[x] : i == 1 ? N.b[x] : N.c[x];
			if (!y || seen[y])
				continue;
			seen[y] = 1;
			if (sp == cap) {
				cap *= 2;
				st = xrealloc(st, cap * sizeof *st);
			}
			st[sp++] = y;
		}
	}
	for (x = 1; x < N.n && (int)n < max; x++) {
		int op = N.op[x];
		if (!seen[x] || !TY_ISINT(N.ty[x]) || N.ty[x] == TY_BOOL)
			continue;
		if (op != O_ADD && op != O_SUB && op != O_MUL && op != O_NEG)
			continue;
		if (inner[x] && x != body)
			continue;
		out[n++] = x;
	}
	/* the body first */
	for (i = 0; i < n; i++)
		if (out[i] == body && i) {
			out[i] = out[0];
			out[0] = body;
		}
	free(st);
	free(seen);
	free(inner);
	return (int)n;
}

/*
 * The proof: the rewritten function against the original, exhaustively
 * or by SAT.  When SAT gives up (the original's adders associate
 * differently from the dots'), by parts: each dot against the sum of its
 * own terms (exhaustive or SAT: a small circuit), and the regrouping of
 * the sum's terms, which is algebra: + and - are associative and
 * commutative mod 2^W, and the conversions the normalisation passed keep
 * the value mod 2^W (narrowing) or exactly (widening a subterm the
 * intervals show does not wrap).  0 proved, 1 a counterexample, 2 not
 * proved; *kind is "exhaustive", "sat" or "parts".
 */
static int
prove_cand(const struct func *f, const struct cand *c, struct sbuf *how,
    struct sbuf *wit, const char **kind)
{
	struct eqproof pr;
	int bad = equiv_decide(f, f->body, c->body, wit, &pr), i, j;
	struct sbuf parts = {0};

	eqproof_text(how, &pr);
	*kind = pr.how == EQ_EXHAUSTIVE ? "exhaustive" : "sat";
	if (bad)
		return 1;
	if (pr.how != EQ_SAMPLED)
		return 0;
	if (c->sat)
		return 2;
	for (i = 0; i < c->ng; i++) {
		const struct group *g = &c->g[i];
		const struct dotform *df = dot_form(g->op);
		ty_t rt = df->sa || df->sb ? TY(K_SINT, 32) : TY(K_UINT, 32);
		nid sum = 0, d = group_dot(g, 0);
		struct eqproof q;
		struct sbuf h = {0};
		for (j = 0; j < c->s->nt; j++) {
			const struct term *t = &c->s->tm[j];
			nid p;
			if (!((g->set >> j) & 1))
				continue;
			p = mk_bin(O_MUL, mk_cast(rt, t->f[0].n), t->prod ?
			    mk_cast(rt, t->f[1].n) : mk_const_i(rt, 1));
			if (!sum)
				sum = t->sign == g->sign ? p : mk_un(O_NEG, p);
			else
				sum = mk_bin(t->sign == g->sign ? O_ADD : O_SUB, sum, p);
		}
		if (equiv_decide(f, sum, d, NULL, &q) || q.how == EQ_SAMPLED) {
			free(parts.p);
			return 2;
		}
		eqproof_text(&h, &q);
		sb_printf(&parts, "%s%s = %s (%s)", i ? "; " : "",
		    shared_str(d, 120), shared_str(sum, 200), sb_str(&h));
		free(h.p);
	}
	sb_clear(how);
	sb_printf(how, "by parts: %s; the other terms are the sum's own, "
	    "regrouped (+ and - associate and commute mod 2^%d; the "
	    "conversions passed keep the value mod 2^%d or exactly); whole: "
	    "SAT gave up, sampled without a difference", sb_str(&parts),
	    mwidth(c->s->t) ? mwidth(c->s->t) : 64,
	    mwidth(c->s->t) ? mwidth(c->s->t) : 64);
	free(parts.p);
	*kind = "parts";
	return 0;
}

void
dot_find(const struct func *f)
{
	struct func g = *f;
	int64_t blo[MAXP], bhi[MAXP];
	nid roots[2], rt[MAXROOT];
	int j, nr, nfmt, any = 0, shown = 0;
	int fmt[3][2];
	struct sum *sums;
	struct cand best;
	float base;

	if (!TY_ISINT(N.ty[f->body]) || N.ty[f->body] == TY_BOOL)
		return;
	S.f = f;
	S.m = model_active();
	nfmt = model_formats(fmt);
	/* proved ranges: the parameters' (the where clause narrowing them),
	 * then intervals, sound for machine values */
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &blo[j], &bhi[j]);
	where_bounds(f, blo, bhi, NULL);
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty) && !(TYK(f->p[j].ty) == K_UINT &&
		    TYW(f->p[j].ty) == 64)) {
			g.p[j].ranged = 1;
			g.p[j].lo = blo[j];
			g.p[j].hi = bhi[j];
		}
	roots[0] = f->body;
	roots[1] = f->where;
	plan_build(&S.p, roots, 2, 0);
	S.iv = xcalloc(S.p.n ? S.p.n : 1, sizeof *S.iv);
	plan_ival(&S.p, &g, S.iv);
	nr = roots_of(f->body, rt, MAXROOT);
	sums = xcalloc((size_t)(nr ? nr : 1), sizeof *sums);
	base = price_dag(f->body);
	memset(&best, 0, sizeof best);
	best.cost = base;
	for (j = 0; j < nr; j++) {
		struct sum *s = &sums[j];
		int i, lanes = 0;
		s->root = rt[j];
		s->t = N.ty[rt[j]];
		s->ok = 1;
		s->exact = !mwidth(s->t) || mwidth(s->t) > 32;
		collect(s, rt[j], 1, s->exact);
		if (!s->ok)
			continue;
		/* worth it with a product of two lane-sized factors */
		for (i = 0; i < s->nt; i++)
			if (N.op[s->tm[i].n] == O_MUL &&
			    fac_signs(&s->tm[i].f[0], 16) &&
			    fac_signs(&s->tm[i].f[1], 16))
				lanes = 1;
		if (!lanes)
			continue;
		any = 1;
		if (nfmt)
			search_sum(&best, f, s, fmt, nfmt);
	}
	if (any && !nfmt) {
		printf("  dot products: %s has no dot product forms (products "
		    "of lane-sized factors are left as they are)\n",
		    sym_str(S.m->name));
		rep_obj("dot_products");
		rep_str("model", sym_str(S.m->name));
		rep_bool("available", 0);
		rep_end();
	}
	if (best.body) {
		struct sbuf w = {0}, how = {0}, d = {0};
		const char *kind;
		int bad = prove_cand(f, &best, &how, &w, &kind);
		describe(&d, &best);
		rep_obj("dot_products");
		rep_str("model", sym_str(S.m->name));
		rep_bool("available", 1);
		rep_num("cost_before", base);
		if (bad) {
			printf("  dot products under %s: %s (cost %g) %s: %s\n",
			    sym_str(S.m->name), shared_str(best.body, 300),
			    (double)best.cost, bad == 1 ? "does not hold" :
			    "is not proved", bad == 1 ? sb_str(&w) : sb_str(&how));
			rep_expr("candidate", best.body);
			rep_str("not_proved", bad ? sb_str(&w) : sb_str(&how));
		} else {
			const char *s = sb_str(&d), *e;
			shown = 1;
			printf("  dot products under %s (sums of products laid "
			    "into lanes; cost %g as written):\n",
			    sym_str(S.m->name), (double)base);
			printf("  cost %-3g %s\n", (double)best.cost,
			    shared_str(best.body, 300));
			while (*s) {
				e = strchr(s, '\n');
				if (!e)
					e = s + strlen(s);
				printf("           %.*s\n", (int)(e - s), s);
				s = *e ? e + 1 : e;
			}
			printf("           %s\n", sb_str(&how));
			rep_num("cost", best.cost);
			rep_expr("expr", best.body);
			rep_str("relies_on", sb_str(&d));
			rep_str("proof", kind);
		}
		rep_end();
		free(w.p);
		free(how.p);
		free(d.p);
	}
	(void)shown;
	free(sums);
	free(S.iv);
	plan_free(&S.p);
	memset(&S, 0, sizeof S);
}
