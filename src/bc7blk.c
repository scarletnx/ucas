/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * bc7blk.c: a BC7 block (BPTC_UNORM) decoded, and the block for one
 * table in its red channel, or two in red and alpha, searched.  Khronos
 * Data Format Specification 1.3, 20.1.
 *
 * A block is 16 bytes read least significant bit first.  Its mode m is
 * the number of zero bits below the first set one (no bit set in the
 * first byte: a reserved block, every channel 0).  After the mode come
 * the partition (PB bits), the rotation (RB), the index selection (ISB),
 * the colour endpoints (CB bits each: red of every endpoint of every
 * subset, then green, then blue), the alpha ones (AB), the p-bits (one
 * an endpoint, or one a subset shared by its two), then a primary index
 * per texel (IB bits) and, in modes 4 and 5, a secondary one (IB2).  A
 * subset's anchor texel (texel 0, and for the other subsets the anchor
 * tables' entry for the partition) has one bit less: its top bit is 0.
 * An endpoint's channel is its bits with the p-bit below, widened to 8
 * by repeating its top bits; no alpha bits means alpha 255.  A texel of
 * subset s reads ((64 - w) e0 + w e1 + 32) >> 6 with w its index's
 * weight; in modes 4 and 5 colour takes the secondary index when ISB is
 * set and alpha the other one, and the rotation then swaps alpha with
 * red, green or blue.  A channel reads c/255.
 *
 *   mode  NS PB RB ISB CB AB EPB SPB IB IB2
 *   0      3  4  0  0   4  0  1   0   3  0
 *   1      2  6  0  0   6  0  0   1   3  0
 *   2      3  6  0  0   5  0  0   0   2  0
 *   3      2  6  0  0   7  0  1   0   2  0
 *   4      1  0  2  1   5  6  0   0   2  3
 *   5      1  0  2  0   7  8  0   0   2  2
 *   6      1  0  0  0   7  7  1   0   4  0
 *   7      2  6  0  0   5  5  1   0   2  0
 */
#include "calc.h"
#include "texenc.h"

#include <math.h>
#include <string.h>

struct bc7mode {
	int ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2;
};

static const struct bc7mode bc7m[8] = {
	{3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0},
	{3, 6, 0, 0, 5, 0, 0, 0, 2, 0}, {2, 6, 0, 0, 7, 0, 1, 0, 2, 0},
	{1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
	{1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0}};

/*
 * The partitions (2 bits a texel, texel x + 4y at bit 2(x + 4y): its
 * subset) and the anchors of the second and third subsets, as the
 * specification's tables 114 to 118 give them
 */
static const uint32_t bc7_part2[64] = {
	0x50505050, 0x40404040, 0x54545454, 0x54505040, 0x50404000,
	0x55545450, 0x55545040, 0x54504000, 0x50400000, 0x55555450,
	0x55544000, 0x54400000, 0x55555440, 0x55550000, 0x55555500,
	0x55000000, 0x55150100, 0x00004054, 0x15010000, 0x00405054,
	0x00004050, 0x15050100, 0x05010000, 0x40505054, 0x00404050,
	0x05010100, 0x14141414, 0x05141450, 0x01155440, 0x00555500,
	0x15014054, 0x05414150, 0x44444444, 0x55005500, 0x11441144,
	0x05055050, 0x05500550, 0x11114444, 0x41144114, 0x44111144,
	0x15055054, 0x01055040, 0x05041050, 0x05455150, 0x14414114,
	0x50050550, 0x41411414, 0x00141400, 0x00041504, 0x00105410,
	0x10541000, 0x04150400, 0x50410514, 0x41051450, 0x05415014,
	0x14054150, 0x41050514, 0x41505014, 0x40011554, 0x54150140,
	0x50505500, 0x00555050, 0x15151010, 0x54540404
};
static const uint32_t bc7_part3[64] = {
	0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000,
	0xa0a05050, 0x5555a0a0, 0x5a5a5050, 0xaa550000, 0xaa555500,
	0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450,
	0x2a0a4250, 0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0,
	0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500, 0x0050a4a4,
	0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414,
	0x50a4a450, 0x6a5a0200, 0xa9a58000, 0x5090a0a8, 0xa8a09050,
	0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
	0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0,
	0x69286928, 0x44aaaa44, 0x66666600, 0xaa444444, 0x54a854a8,
	0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414,
	0x96960000, 0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
	0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254
};
static const uint8_t bc7_anchor2[64] = {
	15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2,
	8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15,
	2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15,
	15, 2, 2, 15
};
static const uint8_t bc7_anchor32[64] = {
	3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3,
	6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15,
	5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3,
	3
};
static const uint8_t bc7_anchor33[64] = {
	15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15,
	3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9,
	10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15,
	15, 15, 3, 15, 15, 8
};

static const int bc7_w2[4] = {0, 21, 43, 64};
static const int bc7_w3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
static const int bc7_w4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47,
	51, 55, 60, 64};

static const int *
bc7_weights(int bits)
{
	return bits == 2 ? bc7_w2 : bits == 3 ? bc7_w3 : bc7_w4;
}

static int
lerp7(int a, int b, int w)
{
	return ((64 - w) * a + w * b + 32) >> 6;
}

static int
widen(int v, int bits)
{
	return (v << (8 - bits)) | (v >> (2 * bits - 8));
}

/* subset of texel i under partition p of an ns-subset mode */
static int
subset_of(int ns, int p, int i)
{
	return ns == 1 ? 0 : (int)(((ns == 2 ? bc7_part2[p] : bc7_part3[p]) >>
	    (2 * i)) & 3);
}

static int
is_anchor(int ns, int p, int i)
{
	return i == 0 || (ns == 2 && i == bc7_anchor2[p]) || (ns == 3 &&
	    (i == bc7_anchor32[p] || i == bc7_anchor33[p]));
}

static int
getbits(const uint8_t q[16], int *pos, int n)
{
	int v = 0, k;

	for (k = 0; k < n; k++, (*pos)++)
		v |= (q[*pos >> 3] >> (*pos & 7) & 1) << k;
	return v;
}

static void
putbits(uint8_t q[16], int *pos, int n, int v)
{
	int k;

	for (k = 0; k < n; k++, (*pos)++)
		if (v >> k & 1)
			q[*pos >> 3] |= (uint8_t)(1 << (*pos & 7));
}

int
bc7_mode(const uint8_t q[16])
{
	int m;

	for (m = 0; m < 8 && !(q[0] >> m & 1); m++)
		;
	return m;	/* 8: reserved */
}

void
bc7_decode(const uint8_t q[16], uint8_t px[16][4])
{
	int m = bc7_mode(q), pos, part, rot, isb, s, e, c, i, ep[6][4];
	int idx[16], idx2[16];
	const struct bc7mode *M;

	if (m == 8) {
		memset(px, 0, 16 * 4);
		return;
	}
	M = &bc7m[m];
	pos = m + 1;
	part = getbits(q, &pos, M->pb);
	rot = getbits(q, &pos, M->rb);
	isb = getbits(q, &pos, M->isb);
	for (c = 0; c < 3; c++)
		for (s = 0; s < M->ns; s++)
			for (e = 0; e < 2; e++)
				ep[2 * s + e][c] = getbits(q, &pos, M->cb);
	for (s = 0; s < M->ns; s++)
		for (e = 0; e < 2; e++)
			ep[2 * s + e][3] = M->ab ? getbits(q, &pos, M->ab) : 0;
	for (s = 0; s < M->ns; s++) {
		int sp = M->spb ? getbits(q, &pos, 1) : 0;
		for (e = 0; e < 2; e++) {
			int p = M->epb ? getbits(q, &pos, 1) : sp;
			for (c = 0; c < 4; c++)
				if (M->epb || M->spb)
					ep[2 * s + e][c] = ep[2 * s + e][c] << 1 | p;
		}
	}
	for (s = 0; s < 2 * M->ns; s++) {
		for (c = 0; c < 3; c++)
			ep[s][c] = widen(ep[s][c], M->cb + M->epb + M->spb);
		ep[s][3] = M->ab ? widen(ep[s][3], M->ab + M->epb + M->spb) : 255;
	}
	for (i = 0; i < 16; i++)
		idx[i] = getbits(q, &pos, M->ib - is_anchor(M->ns, part, i));
	for (i = 0; i < 16; i++)
		idx2[i] = M->ib2 ? getbits(q, &pos, M->ib2 - (i == 0)) : 0;
	for (i = 0; i < 16; i++) {
		int sub = subset_of(M->ns, part, i), t;
		int ci = M->ib2 && isb ? idx2[i] : idx[i];
		int cbits = M->ib2 && isb ? M->ib2 : M->ib;
		int ai = M->ib2 && !isb ? idx2[i] : idx[i];
		int abits = M->ib2 && !isb ? M->ib2 : M->ib;
		for (c = 0; c < 3; c++)
			px[i][c] = (uint8_t)lerp7(ep[2 * sub][c], ep[2 * sub + 1][c],
			    bc7_weights(cbits)[ci]);
		px[i][3] = (uint8_t)lerp7(ep[2 * sub][3], ep[2 * sub + 1][3],
		    bc7_weights(abits)[ai]);
		if (rot) {
			t = px[i][rot - 1];
			px[i][rot - 1] = px[i][3];
			px[i][3] = (uint8_t)t;
		}
	}
}

/* ---- the search ---- */

static double
rd8(int c)
{
	return (double)(float)(c / 255.0);
}

/*
 * One channel's endpoint pair for the texels of a subset (mask), of the
 * values a field of `bits` bits with a p-bit (pb: 0 none, 1 one an
 * endpoint, 2 one shared by the pair) widens to, interpolated with
 * `ibits` weights.  The pair with the smallest largest error, then the
 * smallest sum of squares, the first found; with lim >= 0, the smallest
 * sum of squares among the pairs within lim.  Codes (field and p-bit) and
 * the texels' indices into *r.
 */
struct bc7pair {
	double mx, sq;
	int c[2], p[2], idx[16];
};

static void
pair_search(const double *t, int mask, int bits, int pb, int ibits,
    double lim, struct bc7pair *r)
{
	const int *wt = bc7_weights(ibits);
	int nl = 1 << ibits, nv = 1 << (bits + (pb ? 1 : 0)), a, b, i, k;
	int idx[16] = {0}, found = 0;
	double tmin = INFINITY, tmax = -INFINITY, best = lim >= 0 ? lim :
	    INFINITY, bestsq = INFINITY;

	memset(r, 0, sizeof *r);
	for (i = 0; i < 16; i++)
		if (mask >> i & 1) {
			if (t[i] < tmin)
				tmin = t[i];
			if (t[i] > tmax)
				tmax = t[i];
		}
	if (!mask)
		return;
	/* v = the field with its p-bit below; a pair is unordered, and the
	 * anchor's top bit is set by swapping afterwards */
	for (a = 0; a < nv; a++) {
		int va = widen(a, bits + (pb ? 1 : 0));
		if (rd8(va) - tmin > best)
			continue;
		for (b = 0; b < nv; b++) {
			int vb = widen(b, bits + (pb ? 1 : 0)), pal[16];
			double mx = 0, sq = 0;
			if (pb == 2 && (a & 1) != (b & 1))
				continue;	/* one p-bit for the pair */
			if (vb < va || tmax - rd8(vb) > best)
				continue;
			for (k = 0; k < nl; k++)
				pal[k] = lerp7(va, vb, wt[k]);
			for (i = 0; i < 16 && mx <= best; i++) {
				double be = INFINITY;
				if (!(mask >> i & 1))
					continue;
				for (k = 0; k < nl; k++) {
					double e = fabs(rd8(pal[k]) - t[i]);
					if (e < be) {
						be = e;
						idx[i] = k;
					}
				}
				if (be > mx)
					mx = be;
				sq += be * be;
			}
			if (i < 16 || mx > best)
				continue;
			if (lim < 0 ? found && (mx > best || (mx == best && sq >=
			    bestsq)) : found && sq >= bestsq)
				continue;
			found = 1;
			if (lim < 0)
				best = mx;
			bestsq = sq;
			r->mx = mx;
			r->sq = sq;
			r->c[0] = pb ? a >> 1 : a;
			r->c[1] = pb ? b >> 1 : b;
			r->p[0] = pb ? a & 1 : 0;
			r->p[1] = pb ? b & 1 : 0;
			memcpy(r->idx, idx, sizeof idx);
		}
	}
}

/* the pair's anchor index top bit cleared by swapping its endpoints */
static void
pair_anchor(struct bc7pair *r, int mask, int anchor, int ibits)
{
	int i, t;

	if (!(r->idx[anchor] >> (ibits - 1) & 1))
		return;
	t = r->c[0], r->c[0] = r->c[1], r->c[1] = t;
	t = r->p[0], r->p[0] = r->p[1], r->p[1] = t;
	for (i = 0; i < 16; i++)
		if (mask >> i & 1)
			r->idx[i] = (1 << ibits) - 1 - r->idx[i];
}

/* the anchor texel of subset s under partition p */
static int
anchor_of(int ns, int p, int s)
{
	return s == 0 ? 0 : ns == 2 ? bc7_anchor2[p] : s == 1 ?
	    bc7_anchor32[p] : bc7_anchor33[p];
}

/*
 * A channel's best over one way of storing it: the mode's subsets under
 * partition p, each a pair search; the block's largest error, then the
 * sum of squares at that largest error over every subset
 */
static void
subsets_search(const double *t, const uint8_t *care, int ns, int p,
    int bits, int pb, int ibits, struct bc7pair *pr, double *mx, double *sq)
{
	int s, i;
	double big = 0;

	*sq = 0;
	for (s = 0; s < ns; s++) {
		int mask = 0;
		for (i = 0; i < 16; i++)
			if (care[i] && subset_of(ns, p, i) == s)
				mask |= 1 << i;
		pair_search(t, mask, bits, pb, ibits, -1, &pr[s]);
		if (pr[s].mx > big)
			big = pr[s].mx;
	}
	for (s = 0; s < ns; s++) {
		int mask = 0;
		for (i = 0; i < 16; i++)
			if (care[i] && subset_of(ns, p, i) == s)
				mask |= 1 << i;
		/* below the block's largest error, the fewest squares */
		if (pr[s].mx < big)
			pair_search(t, mask, bits, pb, ibits, big, &pr[s]);
		pair_anchor(&pr[s], mask, anchor_of(ns, p, s), ibits);
		*sq += pr[s].sq;
	}
	*mx = big;
}

/*
 * The block: mode m with partition part, rotation rot and isb, the
 * colour slot's red pairs and the alpha slot's pairs (NULL: 0), the
 * other channels 0
 */
static void
bc7_write(int m, int part, int rot, int isb, const struct bc7pair *col,
    const struct bc7pair *alp, uint8_t q[16])
{
	const struct bc7mode *M = &bc7m[m];
	int pos = 0, s, e, c, i, cidx[16], aidx[16];

	memset(q, 0, 16);
	putbits(q, &pos, m + 1, 1 << m);
	putbits(q, &pos, M->pb, part);
	putbits(q, &pos, M->rb, rot);
	putbits(q, &pos, M->isb, isb);
	for (c = 0; c < 3; c++)
		for (s = 0; s < M->ns; s++)
			for (e = 0; e < 2; e++)
				putbits(q, &pos, M->cb, c == 0 && col ? col[s].c[e] : 0);
	for (s = 0; s < M->ns; s++)
		for (e = 0; e < 2; e++)
			putbits(q, &pos, M->ab, alp ? alp[s].c[e] : 0);
	for (s = 0; s < M->ns; s++) {
		if (M->spb)
			putbits(q, &pos, 1, col ? col[s].p[0] : 0);
		for (e = 0; e < 2 && M->epb; e++)
			putbits(q, &pos, 1, col ? col[s].p[e] : 0);
	}
	for (i = 0; i < 16; i++) {
		int sub = subset_of(M->ns, part, i);
		cidx[i] = col ? col[sub].idx[i] : 0;
		aidx[i] = alp ? alp[0].idx[i] : 0;
	}
	/* modes 4 and 5: colour reads the secondary set when isb is set */
	for (i = 0; i < 16; i++)
		putbits(q, &pos, M->ib - is_anchor(M->ns, part, i), M->ib2 &&
		    isb ? aidx[i] : cidx[i]);
	for (i = 0; i < 16 && M->ib2; i++)
		putbits(q, &pos, M->ib2 - (i == 0), isb ? cidx[i] : aidx[i]);
}

/*
 * The red channel's best block over every mode and every way it can
 * hold red: in modes 0 to 3, 6 and 7 the colour's red; in 4 and 5 the
 * colour's red on either index set, or the alpha rotated into red; with
 * ta, alpha too, in modes 4 and 5 only (where alpha has indices of its
 * own).  The largest error of both, then the sum of squares, the first
 * found; *aerr alpha's largest error.
 */
double
bc7_encode(const double *tr, const double *ta, const uint8_t care[16],
    uint8_t out[16], struct bc7_info *info, double *aerr)
{
	struct bc7pair pr[3], pa[1];
	double bmx = INFINITY, bsq = INFINITY;
	int m, part, found = 0, i, s, e;

	memset(info, 0, sizeof *info);
	memset(out, 0, 16);
	memset(pa, 0, sizeof pa);
	*aerr = 0;
	for (m = 0; m < 8; m++) {
		const struct bc7mode *M = &bc7m[m];
		int nparts = M->pb ? 1 << M->pb : 1, way, nways;
		/* mode 4: colour red on the primary set, on the secondary
		 * (isb), alpha rotated into red on the secondary, on the
		 * primary (isb); mode 5: colour red, alpha rotated into red */
		nways = m == 4 ? 4 : m == 5 ? 2 : 1;
		if (ta && m != 4 && m != 5)
			continue;
		for (part = 0; part < nparts; part++)
			for (way = 0; way < nways; way++) {
				int rot = way >= 2 || (m == 5 && way == 1);
				int isb = m == 4 && (way == 1 || way == 3);
				/* red's slot, field and index set; the other slot's */
				int rbits = rot ? M->ab : M->cb;
				int ribits = !M->ib2 ? M->ib : rot ? (isb ? M->ib :
				    M->ib2) : (isb ? M->ib2 : M->ib);
				int rpb = M->epb ? 1 : M->spb ? 2 : 0;
				int abits = rot ? M->cb : M->ab;
				int aibits = !M->ib2 ? M->ib : rot ? (isb ? M->ib2 :
				    M->ib) : (isb ? M->ib : M->ib2);
				double mx, sq, amx = 0, asq = 0;
				subsets_search(tr, care, M->ns, part, rbits, rpb, ribits,
				    pr, &mx, &sq);
				if (ta) {
					subsets_search(ta, care, 1, 0, abits, 0, aibits, pa,
					    &amx, &asq);
					if (amx > mx)
						mx = amx;
					sq += asq;
				}
				if (found && (mx > bmx || (mx == bmx && sq >= bsq)))
					continue;
				found = 1;
				bmx = mx;
				bsq = sq;
				*aerr = amx;
				bc7_write(m, part, rot, isb, rot ? (ta ? pa : NULL) : pr,
				    rot ? pr : (ta ? pa : NULL), out);
				info->mode = m;
				info->part = part;
				info->rot = rot;
				info->isb = isb;
				info->ns = M->ns;
				info->ribits = ribits;
				for (s = 0; s < M->ns; s++)
					for (e = 0; e < 2; e++)
						info->red[s][e] = rpb ? widen(pr[s].c[e] << 1 |
						    pr[s].p[e], rbits + 1) : widen(pr[s].c[e],
						    rbits);
				for (e = 0; e < 2; e++)
					info->alpha[e] = ta ? widen(pa[0].c[e], abits) : 0;
				for (i = 0; i < 16; i++) {
					info->ridx[i] = pr[subset_of(M->ns, part, i)].idx[i];
					info->aidx[i] = ta ? pa[0].idx[i] : 0;
				}
			}
	}
	return bmx;
}
