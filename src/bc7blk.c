/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * bc7blk.c: a BC7 block (BPTC_UNORM) decoded, and the block for one
 * table in its red channel, or two in red and alpha, searched.  Khronos
 * Data Format Specification 1.3, 20.1.  BC6H, the other BPTC format,
 * below.
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
 * red, green or blue.  A channel reads c/255; in BC7_SRGB red, green and
 * blue read the sRGB EOTF of c/255 (after the rotation), alpha not.
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
#include <stdlib.h>
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

/* what a read of channel value c returns, through the sRGB EOTF or not */
static double
rd8(int srgb, int c)
{
	return srgb ? texenc_srgb8(c) : (double)(float)(c / 255.0);
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
    double lim, int srgb, struct bc7pair *r)
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
		if (rd8(srgb, va) - tmin > best)
			continue;
		for (b = 0; b < nv; b++) {
			int vb = widen(b, bits + (pb ? 1 : 0)), pal[16];
			double mx = 0, sq = 0;
			if (pb == 2 && (a & 1) != (b & 1))
				continue;	/* one p-bit for the pair */
			if (vb < va || tmax - rd8(srgb, vb) > best)
				continue;
			for (k = 0; k < nl; k++)
				pal[k] = lerp7(va, vb, wt[k]);
			for (i = 0; i < 16 && mx <= best; i++) {
				double be = INFINITY;
				if (!(mask >> i & 1))
					continue;
				for (k = 0; k < nl; k++) {
					double e = fabs(rd8(srgb, pal[k]) - t[i]);
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
    int bits, int pb, int ibits, int srgb, struct bc7pair *pr, double *mx,
    double *sq)
{
	int s, i;
	double big = 0;

	*sq = 0;
	for (s = 0; s < ns; s++) {
		int mask = 0;
		for (i = 0; i < 16; i++)
			if (care[i] && subset_of(ns, p, i) == s)
				mask |= 1 << i;
		pair_search(t, mask, bits, pb, ibits, -1, srgb, &pr[s]);
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
			pair_search(t, mask, bits, pb, ibits, big, srgb, &pr[s]);
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
    int srgb, uint8_t out[16], struct bc7_info *info, double *aerr)
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
				/* sRGB converts the red a read returns, rotated or not;
				 * alpha stays linear */
				subsets_search(tr, care, M->ns, part, rbits, rpb, ribits,
				    srgb, pr, &mx, &sq);
				if (ta) {
					subsets_search(ta, care, 1, 0, abits, 0, aibits, 0, pa,
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

/* ---- BC6H ---- */

/*
 * A BC6H block (BC6H_UFLOAT, BC6H_SFLOAT; Khronos Data Format
 * Specification 1.3, 20.2): a mode (2 bits when they are below 2, else
 * 5), endpoints of the mode's bits scattered over the block as the
 * specification's tables 123 and 124 place them (bc6runs, read from
 * them), the first endpoint stored whole and in the transformed modes the
 * others as signed differences from it, a partition of BC7's first 32
 * two-subset ones, and 3 or 4 index bits a texel from bit 82 (two
 * subsets) or 65.  An endpoint is unquantized to 16 bits, two are
 * interpolated with BC7's weights, and the result is finished to half
 * float bits: times 31/64 unsigned, 31/32 of the magnitude signed.
 * Reserved modes read 0.
 */
/* the spec's tables 123 and 124 as runs: block bit, field (0 R, 1 G, 2 B,
 * 3 partition), endpoint (a partition bit: its number), first bit,
 * count, step (+1 or -1) */
struct bc6run {
	uint8_t at, f, e, b, n;
	int8_t d;
};

static const struct bc6run bc6runs[14][32] = {
	/* mode 0 */
	{{2, 1, 2, 4, 1, 1}, {3, 2, 2, 4, 1, 1}, {4, 2, 3, 4, 1, 1},
	    {5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 5, 1}, {40, 1, 3, 4, 1, 1}, {41, 1, 2, 0, 4, 1},
	    {45, 1, 1, 0, 5, 1}, {50, 2, 3, 0, 1, 1}, {51, 1, 3, 0, 4, 1},
	    {55, 2, 1, 0, 5, 1}, {60, 2, 3, 1, 1, 1}, {61, 2, 2, 0, 4, 1},
	    {65, 0, 2, 0, 5, 1}, {70, 2, 3, 2, 1, 1}, {71, 0, 3, 0, 5, 1},
	    {76, 2, 3, 3, 1, 1}, {77, 3, 0, 0, 1, 1}, {78, 3, 1, 0, 1, 1},
	    {79, 3, 2, 0, 1, 1}, {80, 3, 3, 0, 1, 1}, {81, 3, 4, 0, 1, 1}},
	/* mode 1 */
	{{2, 1, 2, 5, 1, 1}, {3, 1, 3, 4, 2, 1}, {5, 0, 0, 0, 7, 1},
	    {12, 2, 3, 0, 2, 1}, {14, 2, 2, 4, 1, 1}, {15, 1, 0, 0, 7, 1},
	    {22, 2, 2, 5, 1, 1}, {23, 2, 3, 2, 1, 1}, {24, 1, 2, 4, 1, 1},
	    {25, 2, 0, 0, 7, 1}, {32, 2, 3, 3, 1, 1}, {33, 2, 3, 5, 2, -1},
	    {35, 0, 1, 0, 6, 1}, {41, 1, 2, 0, 4, 1}, {45, 1, 1, 0, 6, 1},
	    {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 6, 1}, {61, 2, 2, 0, 4, 1},
	    {65, 0, 2, 0, 6, 1}, {71, 0, 3, 0, 6, 1}, {77, 3, 0, 0, 1, 1},
	    {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1}, {80, 3, 3, 0, 1, 1},
	    {81, 3, 4, 0, 1, 1}},
	/* mode 2 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 5, 1}, {40, 0, 0, 10, 1, 1}, {41, 1, 2, 0, 4, 1},
	    {45, 1, 1, 0, 4, 1}, {49, 1, 0, 10, 1, 1}, {50, 2, 3, 0, 1, 1},
	    {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 4, 1}, {59, 2, 0, 10, 1, 1},
	    {60, 2, 3, 1, 1, 1}, {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 5, 1},
	    {70, 2, 3, 2, 1, 1}, {71, 0, 3, 0, 5, 1}, {76, 2, 3, 3, 1, 1},
	    {77, 3, 0, 0, 1, 1}, {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1},
	    {80, 3, 3, 0, 1, 1}, {81, 3, 4, 0, 1, 1}},
	/* mode 6 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 4, 1}, {39, 0, 0, 10, 1, 1}, {40, 1, 3, 4, 1, 1},
	    {41, 1, 2, 0, 4, 1}, {45, 1, 1, 0, 5, 1}, {50, 1, 0, 10, 1, 1},
	    {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 4, 1}, {59, 2, 0, 10, 1, 1},
	    {60, 2, 3, 1, 1, 1}, {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 4, 1},
	    {69, 2, 3, 0, 1, 1}, {70, 2, 3, 2, 1, 1}, {71, 0, 3, 0, 4, 1},
	    {75, 1, 2, 4, 1, 1}, {76, 2, 3, 3, 1, 1}, {77, 3, 0, 0, 1, 1},
	    {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1}, {80, 3, 3, 0, 1, 1},
	    {81, 3, 4, 0, 1, 1}},
	/* mode 10 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 4, 1}, {39, 0, 0, 10, 1, 1}, {40, 2, 2, 4, 1, 1},
	    {41, 1, 2, 0, 4, 1}, {45, 1, 1, 0, 4, 1}, {49, 1, 0, 10, 1, 1},
	    {50, 2, 3, 0, 1, 1}, {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 5, 1},
	    {60, 2, 0, 10, 1, 1}, {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 4, 1},
	    {69, 2, 3, 1, 2, 1}, {71, 0, 3, 0, 4, 1}, {75, 2, 3, 4, 2, -1},
	    {77, 3, 0, 0, 1, 1}, {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1},
	    {80, 3, 3, 0, 1, 1}, {81, 3, 4, 0, 1, 1}},
	/* mode 14 */
	{{5, 0, 0, 0, 9, 1}, {14, 2, 2, 4, 1, 1}, {15, 1, 0, 0, 9, 1},
	    {24, 1, 2, 4, 1, 1}, {25, 2, 0, 0, 9, 1}, {34, 2, 3, 4, 1, 1},
	    {35, 0, 1, 0, 5, 1}, {40, 1, 3, 4, 1, 1}, {41, 1, 2, 0, 4, 1},
	    {45, 1, 1, 0, 5, 1}, {50, 2, 3, 0, 1, 1}, {51, 1, 3, 0, 4, 1},
	    {55, 2, 1, 0, 5, 1}, {60, 2, 3, 1, 1, 1}, {61, 2, 2, 0, 4, 1},
	    {65, 0, 2, 0, 5, 1}, {70, 2, 3, 2, 1, 1}, {71, 0, 3, 0, 5, 1},
	    {76, 2, 3, 3, 1, 1}, {77, 3, 0, 0, 1, 1}, {78, 3, 1, 0, 1, 1},
	    {79, 3, 2, 0, 1, 1}, {80, 3, 3, 0, 1, 1}, {81, 3, 4, 0, 1, 1}},
	/* mode 18 */
	{{5, 0, 0, 0, 8, 1}, {13, 1, 3, 4, 1, 1}, {14, 2, 2, 4, 1, 1},
	    {15, 1, 0, 0, 8, 1}, {23, 2, 3, 2, 1, 1}, {24, 1, 2, 4, 1, 1},
	    {25, 2, 0, 0, 8, 1}, {33, 2, 3, 3, 2, 1}, {35, 0, 1, 0, 6, 1},
	    {41, 1, 2, 0, 4, 1}, {45, 1, 1, 0, 5, 1}, {50, 2, 3, 0, 1, 1},
	    {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 5, 1}, {60, 2, 3, 1, 1, 1},
	    {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 6, 1}, {71, 0, 3, 0, 6, 1},
	    {77, 3, 0, 0, 1, 1}, {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1},
	    {80, 3, 3, 0, 1, 1}, {81, 3, 4, 0, 1, 1}},
	/* mode 22 */
	{{5, 0, 0, 0, 8, 1}, {13, 2, 3, 0, 1, 1}, {14, 2, 2, 4, 1, 1},
	    {15, 1, 0, 0, 8, 1}, {23, 1, 2, 5, 2, -1}, {25, 2, 0, 0, 8, 1},
	    {33, 1, 3, 5, 1, 1}, {34, 2, 3, 4, 1, 1}, {35, 0, 1, 0, 5, 1},
	    {40, 1, 3, 4, 1, 1}, {41, 1, 2, 0, 4, 1}, {45, 1, 1, 0, 6, 1},
	    {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 5, 1}, {60, 2, 3, 1, 1, 1},
	    {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 5, 1}, {70, 2, 3, 2, 1, 1},
	    {71, 0, 3, 0, 5, 1}, {76, 2, 3, 3, 1, 1}, {77, 3, 0, 0, 1, 1},
	    {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1}, {80, 3, 3, 0, 1, 1},
	    {81, 3, 4, 0, 1, 1}},
	/* mode 26 */
	{{5, 0, 0, 0, 8, 1}, {13, 2, 3, 1, 1, 1}, {14, 2, 2, 4, 1, 1},
	    {15, 1, 0, 0, 8, 1}, {23, 2, 2, 5, 1, 1}, {24, 1, 2, 4, 1, 1},
	    {25, 2, 0, 0, 8, 1}, {33, 2, 3, 5, 2, -1}, {35, 0, 1, 0, 5, 1},
	    {40, 1, 3, 4, 1, 1}, {41, 1, 2, 0, 4, 1}, {45, 1, 1, 0, 5, 1},
	    {50, 2, 3, 0, 1, 1}, {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 6, 1},
	    {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 5, 1}, {70, 2, 3, 2, 1, 1},
	    {71, 0, 3, 0, 5, 1}, {76, 2, 3, 3, 1, 1}, {77, 3, 0, 0, 1, 1},
	    {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1}, {80, 3, 3, 0, 1, 1},
	    {81, 3, 4, 0, 1, 1}},
	/* mode 30 */
	{{5, 0, 0, 0, 6, 1}, {11, 1, 3, 4, 1, 1}, {12, 2, 3, 0, 2, 1},
	    {14, 2, 2, 4, 1, 1}, {15, 1, 0, 0, 6, 1}, {21, 1, 2, 5, 1, 1},
	    {22, 2, 2, 5, 1, 1}, {23, 2, 3, 2, 1, 1}, {24, 1, 2, 4, 1, 1},
	    {25, 2, 0, 0, 6, 1}, {31, 1, 3, 5, 1, 1}, {32, 2, 3, 3, 1, 1},
	    {33, 2, 3, 5, 2, -1}, {35, 0, 1, 0, 6, 1}, {41, 1, 2, 0, 4, 1},
	    {45, 1, 1, 0, 6, 1}, {51, 1, 3, 0, 4, 1}, {55, 2, 1, 0, 6, 1},
	    {61, 2, 2, 0, 4, 1}, {65, 0, 2, 0, 6, 1}, {71, 0, 3, 0, 6, 1},
	    {77, 3, 0, 0, 1, 1}, {78, 3, 1, 0, 1, 1}, {79, 3, 2, 0, 1, 1},
	    {80, 3, 3, 0, 1, 1}, {81, 3, 4, 0, 1, 1}},
	/* mode 3 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 10, 1}, {45, 1, 1, 0, 10, 1}, {55, 2, 1, 0, 10, 1}},
	/* mode 7 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 9, 1}, {44, 0, 0, 10, 1, 1}, {45, 1, 1, 0, 9, 1},
	    {54, 1, 0, 10, 1, 1}, {55, 2, 1, 0, 9, 1}, {64, 2, 0, 10, 1, 1}},
	/* mode 11 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 8, 1}, {43, 0, 0, 11, 2, -1}, {45, 1, 1, 0, 8, 1},
	    {53, 1, 0, 11, 2, -1}, {55, 2, 1, 0, 8, 1}, {63, 2, 0, 11, 2, -1}},
	/* mode 15 */
	{{5, 0, 0, 0, 10, 1}, {15, 1, 0, 0, 10, 1}, {25, 2, 0, 0, 10, 1},
	    {35, 0, 1, 0, 4, 1}, {39, 0, 0, 15, 6, -1}, {45, 1, 1, 0, 4, 1},
	    {49, 1, 0, 15, 6, -1}, {55, 2, 1, 0, 4, 1}, {59, 2, 0, 15, 6, -1}},
};

struct bc6mode {
	int code, trans, pb, epb, d[3], ib;
};

/* table 121 */
static const struct bc6mode bc6m[14] = {
	{0, 1, 5, 10, {5, 5, 5}, 3}, {1, 1, 5, 7, {6, 6, 6}, 3},
	{2, 1, 5, 11, {5, 4, 4}, 3}, {6, 1, 5, 11, {4, 5, 4}, 3},
	{10, 1, 5, 11, {4, 4, 5}, 3}, {14, 1, 5, 9, {5, 5, 5}, 3},
	{18, 1, 5, 8, {6, 5, 5}, 3}, {22, 1, 5, 8, {5, 6, 5}, 3},
	{26, 1, 5, 8, {5, 5, 6}, 3}, {30, 0, 5, 6, {6, 6, 6}, 3},
	{3, 0, 0, 10, {10, 10, 10}, 4}, {7, 1, 0, 11, {9, 9, 9}, 4},
	{11, 1, 0, 12, {8, 8, 8}, 4}, {15, 1, 0, 16, {4, 4, 4}, 4}};

static int
bc6_sext(int v, int bits)
{
	return bits >= 32 ? v : (v ^ (1 << (bits - 1))) - (1 << (bits - 1));
}

static int
bc6_unq(int x, int epb, int sgn)
{
	int s = 0;

	if (!sgn) {
		if (epb >= 15)
			return x;
		if (x == 0)
			return 0;
		if (x == (1 << epb) - 1)
			return 0xffff;
		return ((x << 15) + 0x4000) >> (epb - 1);
	}
	if (epb >= 16)
		return x;
	if (x < 0) {
		s = 1;
		x = -x;
	}
	if (x == 0)
		x = 0;
	else if (x >= (1 << (epb - 1)) - 1)
		x = 0x7fff;
	else
		x = ((x << 15) + 0x4000) >> (epb - 1);
	return s ? -x : x;
}

static int
bc6_finish(int i, int sgn)
{
	if (!sgn)
		return (i * 31) >> 6;
	return i < 0 ? (((-i) * 31) >> 5) | 0x8000 : (i * 31) >> 5;
}

/* the mode's index into bc6m, -1 reserved */
static int
bc6_mode(const uint8_t q[16])
{
	int c = (q[0] & 3) < 2 ? q[0] & 3 : q[0] & 31, i;

	for (i = 0; i < 14; i++)
		if (bc6m[i].code == c)
			return i;
	return -1;
}

/* each texel's R, G, B as half-float bits (a reserved block: 0) */
void
bc6h_decode(const uint8_t q[16], int sgn, uint16_t px[16][3])
{
	int mi = bc6_mode(q), e[4][3], part = 0, i, c, r, pos, ns;
	const struct bc6mode *M;

	memset(px, 0, 16 * 3 * sizeof px[0][0]);
	if (mi < 0)
		return;
	M = &bc6m[mi];
	memset(e, 0, sizeof e);
	for (r = 0; r < 32 && bc6runs[mi][r].n; r++) {
		const struct bc6run *R = &bc6runs[mi][r];
		for (i = 0; i < R->n; i++) {
			int bit = q[(R->at + i) >> 3] >> ((R->at + i) & 7) & 1;
			if (R->f == 3)	/* a partition bit: its number in e */
				part |= bit << (R->e + i);
			else
				e[R->e][R->f] |= bit << (R->b + R->d * i);
		}
	}
	ns = M->pb ? 2 : 1;
	for (c = 0; c < 3; c++) {
		if (sgn)
			e[0][c] = bc6_sext(e[0][c], M->epb);
		for (i = 1; i < 2 * ns; i++) {
			if (M->trans) {
				e[i][c] = (e[0][c] + bc6_sext(e[i][c], M->d[c])) &
				    ((1 << M->epb) - 1);
				if (sgn)
					e[i][c] = bc6_sext(e[i][c], M->epb);
			} else if (sgn)
				e[i][c] = bc6_sext(e[i][c], M->epb);
		}
		for (i = 0; i < 2 * ns; i++)
			e[i][c] = bc6_unq(e[i][c], M->epb, sgn);
	}
	pos = ns == 2 ? 82 : 65;
	for (i = 0; i < 16; i++) {
		int anchor = i == 0 || (ns == 2 && i == bc7_anchor2[part]);
		int nb = M->ib - anchor, idx = 0, k, s;
		const int *w = M->ib == 3 ? bc7_w3 : bc7_w4;
		for (k = 0; k < nb; k++, pos++)
			idx |= (q[pos >> 3] >> (pos & 7) & 1) << k;
		s = ns == 2 ? (int)(bc7_part2[part] >> (2 * i) & 3) : 0;
		for (c = 0; c < 3; c++)
			px[i][c] = (uint16_t)bc6_finish(((64 - w[idx]) *
			    e[2 * s][c] + w[idx] * e[2 * s + 1][c] + 32) >> 6, sgn);
	}
}

#define BC6W 6

/* the f32 a half's bits read as */
double
bc6h_half(int h)
{
	int s = h >> 15 & 1, e = h >> 10 & 31, m = h & 1023;
	double v;

	if (e == 31)
		v = m ? NAN : INFINITY;
	else if (e == 0)
		v = ldexp(m, -24);
	else
		v = ldexp(m | 1024, e - 25);
	return s ? -v : v;
}

/* the read of quantized endpoint x alone (weight 0) */
static double
bc6_endval(int x, int epb, int sgn)
{
	return bc6h_half(bc6_finish(bc6_unq(x, epb, sgn), sgn));
}

/* the quantized value whose read is closest to v (the reads rise with x) */
static int
bc6_nearest(double v, int epb, int sgn)
{
	int lo = sgn ? -((1 << (epb - 1)) - 1) : 0, hi = sgn ? (1 << (epb -
	    1)) - 1 : (1 << epb) - 1, mid;

	while (lo < hi) {
		mid = lo + (hi - lo) / 2;
		if (bc6_endval(mid, epb, sgn) < v)
			lo = mid + 1;
		else
			hi = mid;
	}
	if (lo > (sgn ? -((1 << (epb - 1)) - 1) : 0) &&
	    fabs(bc6_endval(lo - 1, epb, sgn) - v) <= fabs(bc6_endval(lo, epb,
	    sgn) - v))
		lo--;
	return lo;
}

struct bc6pair {
	int a, b;
	double mx, sq;
	int idx[16];
};

/* every ordered pair of the subset (mask) near its least and greatest
 * target, the anchor's top bit clear, into P (at most np); how many */
static int
bc6_pairs(const double *t, int mask, int anchor, const struct bc6mode *M,
    int sgn, struct bc6pair *P, int np)
{
	int lo = sgn ? -((1 << (M->epb - 1)) - 1) : 0, hi = sgn ? (1 << (M->epb -
	    1)) - 1 : (1 << M->epb) - 1, cand[4 * BC6W + 2], nc = 0, i, j, k, n = 0;
	const int *w = M->ib == 3 ? bc7_w3 : bc7_w4;
	double tmin = INFINITY, tmax = -INFINITY;

	for (i = 0; i < 16; i++)
		if (mask >> i & 1) {
			tmin = fmin(tmin, t[i]);
			tmax = fmax(tmax, t[i]);
		}
	if (!mask)
		return 0;
	for (k = 0; k < 2; k++) {
		int c = bc6_nearest(k ? tmax : tmin, M->epb, sgn);
		for (i = c - BC6W; i <= c + BC6W; i++) {
			if (i < lo || i > hi)
				continue;
			for (j = 0; j < nc && cand[j] != i; j++)
				;
			if (j == nc)
				cand[nc++] = i;
		}
	}
	for (i = 0; i < nc; i++)
		for (j = 0; j < nc && n < np; j++) {
			int ua = bc6_unq(cand[i], M->epb, sgn), ub = bc6_unq(cand[j],
			    M->epb, sgn), nl = 1 << M->ib, x;
			double pal[16], mx = 0, sq = 0;
			struct bc6pair *p = &P[n];
			for (k = 0; k < nl; k++)
				pal[k] = bc6h_half(bc6_finish(((64 - w[k]) * ua + w[k] * ub +
				    32) >> 6, sgn));
			for (x = 0; x < 16; x++) {
				double be = INFINITY;
				p->idx[x] = 0;
				if (!(mask >> x & 1))
					continue;
				for (k = 0; k < nl; k++)
					if (fabs(pal[k] - t[x]) < be) {
						be = fabs(pal[k] - t[x]);
						p->idx[x] = k;
					}
				mx = fmax(mx, be);
				sq += be * be;
			}
			if (p->idx[anchor] >> (M->ib - 1) & 1)
				continue;	/* the other order has it */
			p->a = cand[i];
			p->b = cand[j];
			p->mx = mx;
			p->sq = sq;
			n++;
		}
	return n;
}

/* does v - base fit the delta's bits, read back as the endpoint */
static int
bc6_delta_ok(int base, int v, int epb, int d, int sgn)
{
	int m = (1 << epb) - 1, delta, back;

	delta = ((v - base) & m);
	delta = bc6_sext(delta, epb);	/* the shortest difference */
	if (delta < -(1 << (d - 1)) || delta > (1 << (d - 1)) - 1)
		return 0;
	back = (base + delta) & m;
	if (sgn)
		back = bc6_sext(back, epb);
	return back == v;
}

/* the block of mode mi: red's endpoints (raw: e0, then e1..e3 as the
 * mode stores them), G and B 0, the partition, the indices */
static void
bc6_write(int mi, int part, const int er[4], const int *idx, uint8_t q[16])
{
	const struct bc6mode *M = &bc6m[mi];
	int r, i, pos, ns = M->pb ? 2 : 1, code = M->code;

	memset(q, 0, 16);
	for (i = 0; i < (code < 2 ? 2 : 5); i++)
		q[0] |= (uint8_t)((code >> i & 1) << i);
	for (r = 0; r < 32 && bc6runs[mi][r].n; r++) {
		const struct bc6run *R = &bc6runs[mi][r];
		for (i = 0; i < R->n; i++) {
			int src = R->f == 3 ? R->e + i : R->b + R->d * i;
			int v = R->f == 3 ? part : R->f == 0 ? er[R->e] : 0;
			int at = R->at + i;
			if (v >> src & 1)
				q[at >> 3] |= (uint8_t)(1 << (at & 7));
		}
	}
	pos = ns == 2 ? 82 : 65;
	for (i = 0; i < 16; i++) {
		int nb = M->ib - (i == 0 || (ns == 2 && i == bc7_anchor2[part])), k;
		for (k = 0; k < nb; k++, pos++)
			if (idx[i] >> k & 1)
				q[pos >> 3] |= (uint8_t)(1 << (pos & 7));
	}
}

/* red's stored fields from the quantized endpoints: e0 raw in EPB bits,
 * the others as deltas (transformed) or raw */
static void
bc6_fields(const struct bc6mode *M, const int x[4], int ns, int er[4])
{
	int m = (1 << M->epb) - 1, i;

	er[0] = x[0] & m;
	for (i = 1; i < 2 * ns; i++)
		er[i] = M->trans ? (bc6_sext((x[i] - x[0]) & m, M->epb) &
		    ((1 << M->d[0]) - 1)) : x[i] & m;
}

double
bc6h_encode_r(const double t[16], const uint8_t care[16], int sgn,
    uint8_t out[16], struct bc6_info *info)
{
	enum { NP = (4 * BC6W + 2) * (4 * BC6W + 2) };
	struct bc6pair *P[2];
	double bmx = INFINITY, bsq = INFINITY;
	int mi, part, found = 0;

	P[0] = xcalloc(2 * NP, sizeof *P[0]);
	P[1] = P[0] + NP;

	memset(out, 0, 16);
	memset(info, 0, sizeof *info);
	for (mi = 0; mi < 14; mi++) {
		const struct bc6mode *M = &bc6m[mi];
		int ns = M->pb ? 2 : 1, nparts = M->pb ? 32 : 1;
		for (part = 0; part < nparts; part++) {
			int n[2] = {0, 0}, s, i, j, mask[2] = {0, 0};
			for (i = 0; i < 16; i++)
				if (care[i])
					mask[ns == 2 ? (int)(bc7_part2[part] >> (2 * i) & 3) :
					    0] |= 1 << i;
			for (s = 0; s < ns; s++)
				n[s] = bc6_pairs(t, mask[s], s ? bc7_anchor2[part] : 0, M,
				    sgn, P[s], NP);
			if (n[0] == 0 && mask[0])
				continue;
			for (i = 0; i < (mask[0] ? n[0] : 1); i++)
				for (j = 0; j < (ns == 2 && mask[1] ? n[1] : 1); j++) {
					const struct bc6pair *p0 = mask[0] ? &P[0][i] : NULL;
					const struct bc6pair *p1 = ns == 2 && mask[1] ?
					    &P[1][j] : NULL;
					double mx = fmax(p0 ? p0->mx : 0, p1 ? p1->mx : 0);
					double sq = (p0 ? p0->sq : 0) + (p1 ? p1->sq : 0);
					int x[4], k, idx[16], er[4];
					if (found && (mx > bmx || (mx == bmx && sq >= bsq)))
						continue;
					x[0] = p0 ? p0->a : 0;
					x[1] = p0 ? p0->b : 0;
					x[2] = p1 ? p1->a : 0;
					x[3] = p1 ? p1->b : 0;
					/* every endpoint within the deltas of e0 */
					if (M->trans) {
						for (k = 1; k < 2 * ns; k++)
							if (!bc6_delta_ok(x[0], x[k], M->epb, M->d[0],
							    sgn))
								break;
						if (k < 2 * ns)
							continue;
					}
					for (k = 0; k < 16; k++)
						idx[k] = ns == 2 && (bc7_part2[part] >> (2 * k) & 3) ?
						    (p1 ? p1->idx[k] : 0) : (p0 ? p0->idx[k] : 0);
					found = 1;
					bmx = mx;
					bsq = sq;
					bc6_fields(M, x, ns, er);
					bc6_write(mi, part, er, idx, out);
					info->mode = M->code;
					info->part = part;
					info->ns = ns;
					for (k = 0; k < 4; k++)
						info->x[k] = x[k];
					for (k = 0; k < 16; k++)
						info->idx[k] = idx[k];
				}
		}
	}
	free(P[0]);
	return bmx;
}
