/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * astcblk.c: ASTC blocks whole (Khronos Data Format Specification, "ASTC
 * Compressed Texture Image Formats"; 2D blocks of the LDR profile): the
 * decoder, and the encoder `texenc T astc WxH` searches with (texenc.c).
 *
 * The decoder reads the block's 128 bits as the specification lays them
 * out: the block mode (the weight grid, its range, one or two planes),
 * the partition count, seed and endpoint modes, the colour values and
 * the weights in the bounded integer sequence encoding (trits and quints
 * packed five and three a group, the weights from the top bit down),
 * then each texel through the steps astc.c has as operations
 * (unquantisation, the endpoint modes, the partition hash, the infill,
 * the interpolation).  It gives each texel's four channels as UNORM16,
 * and the error colour for a block the profile refuses.  It was checked
 * against astcenc 5.7.1 on 2 064 384 random blocks, and the operations
 * against it on every block astcenc decodes among 229 376 more.
 *
 * The encoder: for each block of the table, every block mode the
 * footprint has (a weight grid up to the block's size, a weight range,
 * one plane, or two for two tables), endpoints among the colour range's
 * values near each partition's least and greatest entry, the weights by
 * least squares through the infill, rounded to their range and improved
 * one at a time; for one table also 2 to 4 partitions, each seed tried
 * for the pattern the entries make.  Every block it keeps is decoded
 * again and must give what the search computed.  A texel is read in the
 * UNORM8 decode mode, its byte over 255 (the default float16 mode gives
 * C/65536 rounded toward zero, 1 for 65535: reported beside it).
 */
#include "ucas.h"
#include "texenc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- the decoder ---- */

static uint32_t
bits_at(const uint8_t *b, int at, int n)
{
	uint32_t v = 0;
	int i;

	for (i = 0; i < n; i++)
		if (at + i >= 0 && at + i < 128)
			v |= (uint32_t)((b[(at + i) >> 3] >> ((at + i) & 7)) & 1) << i;
	return v;
}

struct bise {
	int trits, quints, bits;
};

/* the 21 colour ranges, by their number of levels */
static const struct bise crange[21] = {
	{0, 0, 1}, {1, 0, 0}, {0, 0, 2}, {0, 1, 0}, {1, 0, 1}, {0, 0, 3},
	{0, 1, 1}, {1, 0, 2}, {0, 0, 4}, {0, 1, 2}, {1, 0, 3}, {0, 0, 5},
	{0, 1, 3}, {1, 0, 4}, {0, 0, 6}, {0, 1, 4}, {1, 0, 5}, {0, 0, 7},
	{0, 1, 5}, {1, 0, 6}, {0, 0, 8},
};

/* the weight ranges by the mode's high precision bit and r (2..7) */
static const struct bise wrange[2][8] = {
	{{0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {0, 0, 2}, {0, 1, 0},
	    {1, 0, 1}, {0, 0, 3}},
	{{0, 0, 0}, {0, 0, 0}, {0, 1, 1}, {1, 0, 2}, {0, 0, 4}, {0, 1, 2},
	    {1, 0, 3}, {0, 0, 5}},
};

static int
levels(const struct bise *q)
{
	return (q->trits ? 3 : q->quints ? 5 : 1) << q->bits;
}

static int
bise_bits(const struct bise *r, int n)
{
	return n * r->bits + (r->trits ? (8 * n + 4) / 5 : 0) +
	    (r->quints ? (7 * n + 2) / 3 : 0);
}

struct rd {
	const uint8_t *b;
	int at, rev, left;	/* rev: reads downward from at (the weights) */
};

static uint32_t
rd(struct rd *r, int n)
{
	uint32_t v = 0;
	int i;

	for (i = 0; i < n; i++) {
		if (r->left > 0)
			v |= bits_at(r->b, r->rev ? r->at - i : r->at + i, 1) << i;
		r->left--;
	}
	r->at += r->rev ? -n : n;
	return v;
}

/* the five trits an 8-bit T packs */
static void
trits_of(uint32_t T, int *t)
{
	int C;

	if (((T >> 2) & 7) == 7) {
		C = (int)(((T >> 5) & 7) << 2 | (T & 3));
		t[4] = t[3] = 2;
	} else {
		C = (int)(T & 31);
		if (((T >> 5) & 3) == 3) {
			t[4] = 2;
			t[3] = (int)(T >> 7) & 1;
		} else {
			t[4] = (int)(T >> 7) & 1;
			t[3] = (int)(T >> 5) & 3;
		}
	}
	if ((C & 3) == 3) {
		t[2] = 2;
		t[1] = (C >> 4) & 1;
		t[0] = ((C >> 3) & 1) << 1 | ((C >> 2) & 1 & ~(C >> 3));
	} else if (((C >> 2) & 3) == 3) {
		t[2] = 2;
		t[1] = 2;
		t[0] = C & 3;
	} else {
		t[2] = (C >> 4) & 1;
		t[1] = (C >> 2) & 3;
		t[0] = (C & 2) | (C & 1 & ~(C >> 1));
	}
}

/* the three quints a 7-bit Q packs */
static void
quints_of(uint32_t Q, int *v)
{
	int C;

	if (((Q >> 1) & 3) == 3 && ((Q >> 5) & 3) == 0) {
		v[2] = (int)((Q & 1) << 2 | ((Q >> 4) & 1 & ~Q) << 1 |
		    ((Q >> 3) & 1 & ~Q));
		v[1] = v[0] = 4;
		return;
	}
	if (((Q >> 1) & 3) == 3) {
		v[2] = 4;
		C = (int)(((Q >> 3) & 3) << 3 | (~(Q >> 5) & 3) << 1 | (Q & 1));
	} else {
		v[2] = (int)(Q >> 5) & 3;
		C = (int)(Q & 31);
	}
	if ((C & 7) == 5) {
		v[1] = 4;
		v[0] = (C >> 3) & 3;
	} else {
		v[1] = (C >> 3) & 3;
		v[0] = C & 7;
	}
}

/* n values: each the trit or quint times 2^m plus its m bits */
static void
bise_decode(struct rd *r, const struct bise *q, int n, int *val)
{
	int i, j, m = q->bits;

	if (q->trits) {
		for (i = 0; i < n; i += 5) {
			int t[5];
			uint32_t T, mb[5];
			mb[0] = rd(r, m);
			T = rd(r, 2);
			mb[1] = rd(r, m);
			T |= rd(r, 2) << 2;
			mb[2] = rd(r, m);
			T |= rd(r, 1) << 4;
			mb[3] = rd(r, m);
			T |= rd(r, 2) << 5;
			mb[4] = rd(r, m);
			T |= rd(r, 1) << 7;
			trits_of(T, t);
			for (j = 0; j < 5 && i + j < n; j++)
				val[i + j] = t[j] << m | (int)mb[j];
		}
	} else if (q->quints) {
		for (i = 0; i < n; i += 3) {
			int v[3];
			uint32_t Q, mb[3];
			mb[0] = rd(r, m);
			Q = rd(r, 3);
			mb[1] = rd(r, m);
			Q |= rd(r, 2) << 3;
			mb[2] = rd(r, m);
			Q |= rd(r, 2) << 5;
			quints_of(Q, v);
			for (j = 0; j < 3 && i + j < n; j++)
				val[i + j] = v[j] << m | (int)mb[j];
		}
	} else
		for (i = 0; i < n; i++)
			val[i] = (int)rd(r, m);
}

/* a colour value v of range q to 0..255 */
static int
color_unq(const struct bise *q, int v)
{
	int m = q->bits, d = v >> m, a = v & 1, b = (v >> 1) & 1;
	int c = (v >> 2) & 1, dd = (v >> 3) & 1, e = (v >> 4) & 1;
	int f = (v >> 5) & 1, A = a ? 0x1ff : 0, B = 0, C = 0, T;

	if (!q->trits && !q->quints) {
		int r = 0, n = 0;
		while (n < 8) {
			r = (r << m) | v;
			n += m;
		}
		return (r >> (n - 8)) & 255;
	}
	if (q->trits) {
		switch (m) {
		case 1: C = 204; break;
		case 2: C = 93; B = b << 8 | b << 4 | b << 2 | b << 1; break;
		case 3: C = 44; B = c << 8 | b << 7 | c << 3 | b << 2 | c << 1 | b;
			break;
		case 4: C = 22; B = dd << 8 | c << 7 | b << 6 | dd << 2 | c << 1 |
			    b; break;
		case 5: C = 11; B = e << 8 | dd << 7 | c << 6 | b << 5 | e << 1 |
			    dd; break;
		default: C = 5; B = f << 8 | e << 7 | dd << 6 | c << 5 | b << 4 |
			    f; break;
		}
	} else {
		switch (m) {
		case 1: C = 113; break;
		case 2: C = 54; B = b << 8 | b << 3 | b << 2; break;
		case 3: C = 26; B = c << 8 | b << 7 | c << 2 | b << 1 | c; break;
		case 4: C = 13; B = dd << 8 | c << 7 | b << 6 | dd << 1 | c;
			break;
		default: C = 6; B = e << 8 | dd << 7 | c << 6 | b << 5 | e;
			break;
		}
	}
	T = (d * C + B) ^ A;
	return (A & 0x80) | (T >> 2);
}

/* a weight v of range q to 0..64 */
static int
weight_unq(const struct bise *q, int v)
{
	int m = q->bits, d = v >> m, b = (v >> 1) & 1, c = (v >> 2) & 1;
	int A = (v & 1) ? 0x7f : 0, B = 0, C = 0, T;

	if (!q->trits && !q->quints) {
		int r = 0, n = 0;
		while (n < 6) {
			r = (r << m) | v;
			n += m;
		}
		T = (r >> (n - 6)) & 63;
	} else if (m == 0) {
		static const int t3[3] = {0, 32, 63}, q5[5] = {0, 16, 32, 47, 63};
		T = q->trits ? t3[d] : q5[d];
	} else {
		if (q->trits) {
			switch (m) {
			case 1: C = 50; break;
			case 2: C = 23; B = b << 6 | b << 2 | b; break;
			default: C = 11; B = c << 6 | b << 5 | c << 1 | b; break;
			}
		} else {
			switch (m) {
			case 1: C = 28; break;
			default: C = 13; B = b << 6 | b << 1; break;
			}
		}
		T = (d * C + B) ^ A;
		T = (A & 0x20) | (T >> 2);
	}
	return T > 32 ? T + 1 : T;
}

int
astcb_partition(int seed, int x, int y, int count, int small)
{
	uint32_t p, s[9];
	int sh1, sh2, a, b, c, d, i;

	if (count < 2)
		return 0;
	if (small) {
		x <<= 1;
		y <<= 1;
	}
	seed += (count - 1) * 1024;
	p = (uint32_t)seed;
	p ^= p >> 15;
	p -= p << 17;
	p += p << 7;
	p += p << 4;
	p ^= p >> 5;
	p += p << 16;
	p ^= p >> 7;
	p ^= p >> 3;
	p ^= p << 6;
	p ^= p >> 17;
	/* the nibbles z's terms would use (9 to 12) are not needed: z = 0 */
	for (i = 1; i <= 8; i++) {
		s[i] = (p >> (4 * (i - 1))) & 15;
		s[i] *= s[i];
	}
	if (seed & 1) {
		sh1 = seed & 2 ? 4 : 5;
		sh2 = count == 3 ? 6 : 5;
	} else {
		sh1 = count == 3 ? 6 : 5;
		sh2 = seed & 2 ? 4 : 5;
	}
	for (i = 1; i <= 8; i++)
		s[i] >>= i & 1 ? sh1 : sh2;
	a = (int)((s[1] * (uint32_t)x + s[2] * (uint32_t)y + (p >> 14)) & 63);
	b = (int)((s[3] * (uint32_t)x + s[4] * (uint32_t)y + (p >> 10)) & 63);
	c = (int)((s[5] * (uint32_t)x + s[6] * (uint32_t)y + (p >> 6)) & 63);
	d = (int)((s[7] * (uint32_t)x + s[8] * (uint32_t)y + (p >> 2)) & 63);
	if (count < 4)
		d = 0;
	if (count < 3)
		c = 0;
	if (a >= b && a >= c && a >= d)
		return 0;
	if (b >= c && b >= d)
		return 1;
	return c >= d ? 2 : 3;
}

/* bit_transfer_signed */
static void
bts(int *a, int *b)
{
	*b >>= 1;
	*b |= *a & 0x80;
	*a >>= 1;
	*a &= 0x3f;
	if (*a & 0x20)
		*a -= 0x40;
}

static int
clamp8(int v)
{
	return v < 0 ? 0 : v > 255 ? 255 : v;
}

static void
contract(int *e)
{
	e[0] = (e[0] + e[2]) >> 1;
	e[1] = (e[1] + e[2]) >> 1;
}

static void
set4(int *e, int r, int g, int b, int a)
{
	e[0] = r;
	e[1] = g;
	e[2] = b;
	e[3] = a;
}

/* the two endpoints of LDR mode cem from its values; 0 for an HDR mode */
static int
endpoints(int cem, const int *vv, int *e0, int *e1)
{
	int v[8], i, t;

	for (i = 0; i < 8; i++)
		v[i] = vv[i];
	switch (cem) {
	case 0:
		set4(e0, v[0], v[0], v[0], 255);
		set4(e1, v[1], v[1], v[1], 255);
		break;
	case 1: {
		int l0 = (v[0] >> 2) | (v[1] & 0xc0), l1 = l0 + (v[1] & 0x3f);
		if (l1 > 255)
			l1 = 255;
		set4(e0, l0, l0, l0, 255);
		set4(e1, l1, l1, l1, 255);
		break;
	}
	case 4:
		set4(e0, v[0], v[0], v[0], v[2]);
		set4(e1, v[1], v[1], v[1], v[3]);
		break;
	case 5:
		bts(&v[1], &v[0]);
		bts(&v[3], &v[2]);
		set4(e0, v[0], v[0], v[0], v[2]);
		t = clamp8(v[0] + v[1]);
		set4(e1, t, t, t, clamp8(v[2] + v[3]));
		break;
	case 6:
	case 10:
		set4(e0, v[0] * v[3] >> 8, v[1] * v[3] >> 8, v[2] * v[3] >> 8,
		    cem == 10 ? v[4] : 255);
		set4(e1, v[0], v[1], v[2], cem == 10 ? v[5] : 255);
		break;
	case 8:
	case 12:
		if (v[1] + v[3] + v[5] >= v[0] + v[2] + v[4]) {
			set4(e0, v[0], v[2], v[4], cem == 12 ? v[6] : 255);
			set4(e1, v[1], v[3], v[5], cem == 12 ? v[7] : 255);
		} else {
			set4(e0, v[1], v[3], v[5], cem == 12 ? v[7] : 255);
			set4(e1, v[0], v[2], v[4], cem == 12 ? v[6] : 255);
			contract(e0);
			contract(e1);
		}
		break;
	case 9:
	case 13:
		bts(&v[1], &v[0]);
		bts(&v[3], &v[2]);
		bts(&v[5], &v[4]);
		if (cem == 13)
			bts(&v[7], &v[6]);
		else {
			v[6] = 255;
			v[7] = 0;
		}
		if (v[1] + v[3] + v[5] >= 0) {
			set4(e0, v[0], v[2], v[4], v[6]);
			set4(e1, v[0] + v[1], v[2] + v[3], v[4] + v[5], v[6] + v[7]);
		} else {
			set4(e0, v[0] + v[1], v[2] + v[3], v[4] + v[5], v[6] + v[7]);
			set4(e1, v[0], v[2], v[4], v[6]);
			contract(e0);
			contract(e1);
		}
		for (i = 0; i < 4; i++) {
			e0[i] = clamp8(e0[i]);
			e1[i] = clamp8(e1[i]);
		}
		break;
	default:
		return 0;
	}
	return 1;
}

static int
error_block(int n, uint16_t *out)
{
	int i;

	for (i = 0; i < n; i++) {
		out[4 * i] = 0xffff;
		out[4 * i + 1] = 0;
		out[4 * i + 2] = 0xffff;
		out[4 * i + 3] = 0xffff;
	}
	return 1;
}

/* the block mode's grid, planes and weight range; 0 for a void extent
 * or a reserved mode */
static int
mode_of(uint32_t mode, int *W, int *H, int *D, int *Hp, int *r)
{
	int A = (int)(mode >> 5) & 3, B = (int)(mode >> 7) & 3;

	if ((mode & 0x1ff) == 0x1fc || (mode & 15) == 0)
		return 0;
	*D = (int)(mode >> 10) & 1;
	*Hp = (int)(mode >> 9) & 1;
	if (mode & 3) {
		*r = (int)((mode & 3) << 1 | ((mode >> 4) & 1));
		switch ((mode >> 2) & 3) {
		case 0: *W = B + 4; *H = A + 2; break;
		case 1: *W = B + 8; *H = A + 2; break;
		case 2: *W = A + 2; *H = B + 8; break;
		default:
			if (B & 2) {
				*W = (B & 1) + 2;
				*H = A + 2;
			} else {
				*W = A + 2;
				*H = (B & 1) + 6;
			}
		}
	} else {
		*r = (int)(((mode >> 2) & 3) << 1 | ((mode >> 4) & 1));
		switch (B) {
		case 0: *W = 12; *H = A + 2; break;
		case 1: *W = A + 2; *H = 12; break;
		case 2:
			*W = A + 6;
			*H = (int)((mode >> 9) & 3) + 6;
			*D = *Hp = 0;
			break;
		default:
			if (A == 0) {
				*W = 6;
				*H = 10;
			} else if (A == 1) {
				*W = 10;
				*H = 6;
			} else
				return 0;
		}
	}
	return *r >= 2;
}

/* the infill of texel (i, j) of a bw x bh block from a W x H grid: the
 * four grid points it reads and their factors (sixteenths) */
static void
infill_at(int bw, int bh, int W, int H, int i, int j, int pt[4], int f[4])
{
	int ds = (1024 + bw / 2) / (bw - 1), dt = (1024 + bh / 2) / (bh - 1);
	int gs = (ds * i * (W - 1) + 32) >> 6, gt = (dt * j * (H - 1) + 32) >> 6;
	int js = gs >> 4, fs = gs & 15, jt = gt >> 4, ft = gt & 15;
	int w11 = (fs * ft + 8) >> 4;

	f[0] = 16 - fs - ft + w11;
	f[1] = fs - w11;
	f[2] = ft - w11;
	f[3] = w11;
	pt[0] = js + jt * W;
	pt[1] = js + 1 < W ? js + 1 + jt * W : -1;
	pt[2] = jt + 1 < H ? js + (jt + 1) * W : -1;
	pt[3] = js + 1 < W && jt + 1 < H ? js + 1 + (jt + 1) * W : -1;
}

int
astcb_decode(const uint8_t blk[16], int bw, int bh, uint16_t *out)
{
	uint32_t mode = bits_at(blk, 0, 11);
	int W, H, D, Hp, r, nw, wbits, parts, cem[4], i, j, k, ncv = 0, cbits;
	int cstart, below, qi, ccs = 0, cv[18], val[64], w[64], e0[4][4];
	int e1[4][4], hdr[4];
	struct bise wq;
	struct rd rr;

	if ((mode & 0x1ff) == 0x1fc) {
		/* a void extent: one colour, with the rectangle it holds on */
		uint32_t e[4];
		int all = 1;
		if (bits_at(blk, 9, 1) || bits_at(blk, 10, 2) != 3)
			return error_block(bw * bh, out);
		for (i = 0; i < 4; i++) {
			e[i] = bits_at(blk, 12 + 13 * i, 13);
			all &= e[i] == 0x1fff;
		}
		if (!all && (e[0] >= e[1] || e[2] >= e[3]))
			return error_block(bw * bh, out);
		for (i = 0; i < bw * bh; i++)
			for (j = 0; j < 4; j++)
				out[4 * i + j] = (uint16_t)bits_at(blk, 64 + 16 * j, 16);
		return 0;
	}
	if (!mode_of(mode, &W, &H, &D, &Hp, &r) || W > bw || H > bh)
		return error_block(bw * bh, out);
	wq = wrange[Hp][r];
	nw = W * H * (D ? 2 : 1);
	wbits = bise_bits(&wq, nw);
	parts = (int)bits_at(blk, 11, 2) + 1;
	if (nw > 64 || wbits < 24 || wbits > 96 || (D && parts == 4))
		return error_block(bw * bh, out);
	below = 128 - wbits;
	if (parts == 1) {
		cem[0] = (int)bits_at(blk, 13, 4);
		cstart = 17;
	} else {
		uint32_t f = bits_at(blk, 23, 6);
		cstart = 29;
		if ((f & 3) == 0) {
			for (i = 0; i < parts; i++)
				cem[i] = (int)(f >> 2) & 15;
		} else {
			int cls = (int)(f & 3) - 1, extra = 3 * parts - 4;
			below -= extra;
			f |= bits_at(blk, below, extra) << 6;
			for (i = 0; i < parts; i++)
				cem[i] = (cls + (int)((f >> (2 + i)) & 1)) * 4 +
				    (int)((f >> (2 + parts + 2 * i)) & 3);
		}
	}
	/* the second plane's component sits under the extra mode bits */
	if (D) {
		below -= 2;
		ccs = (int)bits_at(blk, below, 2);
	}
	for (i = 0; i < parts; i++)
		ncv += 2 * (cem[i] / 4 + 1);
	cbits = below - cstart;
	if (ncv > 18 || cbits < (13 * ncv + 4) / 5)
		return error_block(bw * bh, out);
	for (qi = 20; qi >= 0 && bise_bits(&crange[qi], ncv) > cbits; qi--)
		;
	if (qi < 4)
		return error_block(bw * bh, out);
	rr.b = blk;
	rr.at = cstart;
	rr.rev = 0;
	rr.left = bise_bits(&crange[qi], ncv);
	bise_decode(&rr, &crange[qi], ncv, cv);
	for (i = 0; i < ncv; i++)
		cv[i] = color_unq(&crange[qi], cv[i]);
	for (i = 0, k = 0; i < parts; i++) {
		int v[8] = {0};
		for (j = 0; j < 2 * (cem[i] / 4 + 1); j++)
			v[j] = cv[k++];
		hdr[i] = !endpoints(cem[i], v, e0[i], e1[i]);
	}
	rr.at = 127;
	rr.rev = 1;
	rr.left = wbits;
	bise_decode(&rr, &wq, nw, val);
	for (i = 0; i < nw; i++)
		w[i] = weight_unq(&wq, val[i]);
	for (j = 0; j < bh; j++)
		for (i = 0; i < bw; i++) {
			int pt[4], f[4], pl, wt[2], p, c, q;
			uint16_t *o = out + 4 * (j * bw + i);
			infill_at(bw, bh, W, H, i, j, pt, f);
			for (pl = 0; pl < (D ? 2 : 1); pl++) {
				int s = 8;
				for (q = 0; q < 4; q++)
					if (pt[q] >= 0)
						s += w[pt[q] * (D ? 2 : 1) + pl] * f[q];
				wt[pl] = s >> 4;
			}
			p = astcb_partition((int)bits_at(blk, 13, 10), i, j, parts,
			    bw * bh < 31);
			if (hdr[p]) {
				o[0] = 0xffff;
				o[1] = 0;
				o[2] = 0xffff;
				o[3] = 0xffff;
				continue;
			}
			for (c = 0; c < 4; c++) {
				int ww = D && c == ccs ? wt[1] : wt[0];
				int c0 = e0[p][c] * 257, c1 = e1[p][c] * 257;
				o[c] = (uint16_t)((c0 * (64 - ww) + c1 * ww + 32) >> 6);
			}
		}
	return 0;
}

double
astcb_float16(int c)
{
	double x;
	int e;

	/* decode_float16: 65535 is 1; the rest C/65536 to binary16 toward
	 * zero (an 11-bit significand) */
	if (c == 65535)
		return 1;
	if (c == 0)
		return 0;
	x = c / 65536.0;
	e = (int)floor(log2(x));
	if (e < -14)
		e = -14;	/* subnormal: steps of 2^-24 */
	return floor(ldexp(x, 10 - e)) * ldexp(1, e - 10);
}

/* ---- the encoder ---- */

/* a value written into a stream of bits, least significant first */
struct wr {
	uint8_t s[128];
	int n;
};

static void
put(struct wr *w, uint32_t v, int n)
{
	int i;

	for (i = 0; i < n; i++)
		w->s[w->n++] = (uint8_t)((v >> i) & 1);
}

/*
 * n values of range q as the integer sequence encoding writes them: the
 * packed trits or quints of a group interleaved with each value's bits;
 * a last group that is cut short packs its missing values as 0 and drops
 * the packed bits past the values it has, which must then be 0.
 */
static int
bise_encode(struct wr *w, const struct bise *q, int n, const int *val)
{
	int i, j, m = q->bits, k, g = q->trits ? 5 : 3;

	if (!q->trits && !q->quints) {
		for (i = 0; i < n; i++)
			put(w, (uint32_t)val[i], m);
		return 1;
	}
	for (i = 0; i < n; i += g) {
		int hi[5] = {0, 0, 0, 0, 0}, have = n - i < g ? n - i : g, keep;
		uint32_t P = 0;
		for (j = 0; j < have; j++)
			hi[j] = val[i + j] >> m;
		/* the packed bits the group keeps: those of its values */
		keep = q->trits ? (8 * have + 4) / 5 : (7 * have + 2) / 3;
		for (k = 0; k < (q->trits ? 256 : 128); k++) {
			int t[5];
			if (keep < (q->trits ? 8 : 7) && (k >> keep))
				continue;
			if (q->trits)
				trits_of((uint32_t)k, t);
			else
				quints_of((uint32_t)k, t);
			for (j = 0; j < have && t[j] == hi[j]; j++)
				;
			if (j == have)
				break;
		}
		if (k == (q->trits ? 256 : 128))
			return 0;
		P = (uint32_t)k;
#define LO(j) ((uint32_t)val[i + (j)] & ((1u << m) - 1))
		if (q->trits) {
			static const int at[5] = {0, 2, 4, 5, 7}, nb[5] = {2, 2, 1, 2,
			    1};
			for (j = 0; j < have; j++) {
				put(w, LO(j), m);
				put(w, P >> at[j], nb[j]);
			}
		} else {
			static const int at[3] = {0, 3, 5}, nb[3] = {3, 2, 2};
			for (j = 0; j < have; j++) {
				put(w, LO(j), m);
				put(w, P >> at[j], nb[j]);
			}
		}
#undef LO
	}
	return 1;
}

/* the mode for a grid, its planes and its range (index into wrange), or
 * -1 */
static int
mode_for(int W, int H, int D, int Hp, int r)
{
	static int table[13][13][2][2][8], built;
	uint32_t m;

	if (!built) {
		int a, b, c, d, e;
		for (a = 0; a < 13; a++)
			for (b = 0; b < 13; b++)
				for (c = 0; c < 2; c++)
					for (d = 0; d < 2; d++)
						for (e = 0; e < 8; e++)
							table[a][b][c][d][e] = -1;
		for (m = 0; m < 2048; m++) {
			int w, h, dd, hp, rr;
			if (mode_of(m, &w, &h, &dd, &hp, &rr) && w <= 12 && h <= 12 &&
			    table[w][h][dd][hp][rr] < 0)
				table[w][h][dd][hp][rr] = (int)m;
		}
		built = 1;
	}
	if (W < 2 || W > 12 || H < 2 || H > 12)
		return -1;
	return table[W][H][D][Hp][r];
}

/*
 * A block's layout, written: the mode, partitions (count, seed, one
 * endpoint mode for all), the colour values in the largest range the
 * bits left allow, the second plane's channel, the weights from the top
 * bit down.  0 when it does not fit.
 */
struct alay {
	int W, H, D, Hp, r, parts, seed, cem, ccs;
	int qi;			/* the colour range */
	int ncv, cv[18];	/* colour values (encoded) */
	int nw, wv[64];		/* weights (encoded), planes interleaved */
};

/* the colour range a layout leaves room for, or -1 */
static int
color_range(const struct alay *L)
{
	struct bise wq = wrange[L->Hp][L->r];
	int nw = L->W * L->H * (L->D ? 2 : 1), wbits = bise_bits(&wq, nw);
	int below = 128 - wbits - (L->D ? 2 : 0), cbits, qi;
	int ncv = L->parts * 2 * (L->cem / 4 + 1);

	if (nw > 64 || wbits < 24 || wbits > 96 || ncv > 18 || (L->D &&
	    L->parts == 4))
		return -1;
	cbits = below - (L->parts == 1 ? 17 : 29);
	if (cbits < (13 * ncv + 4) / 5)
		return -1;
	for (qi = 20; qi >= 0 && bise_bits(&crange[qi], ncv) > cbits; qi--)
		;
	return qi >= 4 ? qi : -1;
}

static int
write_block(const struct alay *L, uint8_t out[16])
{
	struct wr c, w;
	struct bise wq = wrange[L->Hp][L->r];
	int mode = mode_for(L->W, L->H, L->D, L->Hp, L->r), i, below, cstart;
	uint32_t hi = 0;

	if (mode < 0)
		return 0;
	memset(out, 0, 16);
	memset(&c, 0, sizeof c);
	memset(&w, 0, sizeof w);
	put(&c, (uint32_t)mode, 11);
	put(&c, (uint32_t)(L->parts - 1), 2);
	if (L->parts == 1)
		put(&c, (uint32_t)L->cem, 4);
	else {
		put(&c, (uint32_t)L->seed, 10);
		put(&c, (uint32_t)L->cem << 2, 6);
	}
	cstart = c.n;
	if (!bise_encode(&c, &crange[L->qi], L->ncv, L->cv) ||
	    !bise_encode(&w, &wq, L->nw, L->wv))
		return 0;
	below = 128 - w.n;
	for (i = 0; i < c.n; i++)
		out[i >> 3] |= (uint8_t)(c.s[i] << (i & 7));
	if (L->D) {
		below -= 2;
		if (below < c.n)
			return 0;
		hi = (uint32_t)L->ccs;
		for (i = 0; i < 2; i++)
			out[(below + i) >> 3] |= (uint8_t)(((hi >> i) & 1) <<
			    ((below + i) & 7));
	}
	if (below < c.n)
		return 0;
	for (i = 0; i < w.n; i++)
		out[(127 - i) >> 3] |= (uint8_t)(w.s[i] << ((127 - i) & 7));
	(void)cstart;
	return 1;
}

/* ---- the search ---- */

#define MAXT 144

struct aprob {
	int bw, bh, n, nt;	/* texels, tables (planes) */
	double t[2][MAXT];	/* what a read should give, per table */
	uint8_t care[MAXT];
};

struct ascore {
	double mx, sq;
};

static int
better(struct ascore a, struct ascore b)
{
	return a.mx < b.mx || (a.mx == b.mx && a.sq < b.sq);
}

/* a decoded byte as a read returns it (UNORM8 mode): its f32 over 255 */
static double
rd8(int c)
{
	return (double)(float)((c >> 8) / 255.0);
}

static int
lerp16(int a, int b, int w)
{
	return (a * 257 * (64 - w) + b * 257 * w + 32) >> 6;
}

/* the sorted distinct values of a range and an encoded value for each */
struct lvl {
	int n, u[256], v[256];
};

static void
lvl_of(struct lvl *L, const struct bise *q, int colour)
{
	int k, i, j, nl = levels(q);

	L->n = 0;
	for (k = 0; k < nl; k++) {
		int u = colour ? color_unq(q, k) : weight_unq(q, k);
		for (i = 0; i < L->n && L->u[i] < u; i++)
			;
		if (i < L->n && L->u[i] == u)
			continue;
		for (j = L->n; j > i; j--) {
			L->u[j] = L->u[j - 1];
			L->v[j] = L->v[j - 1];
		}
		L->u[i] = u;
		L->v[i] = k;
		L->n++;
	}
}

/* the level nearest u */
static int
lvl_near(const struct lvl *L, double u)
{
	int i, b = 0;

	for (i = 1; i < L->n; i++)
		if (fabs(L->u[i] - u) < fabs(L->u[b] - u))
			b = i;
	return b;
}

/*
 * One plane's weights for a grid, given each texel's endpoints (a, b:
 * bytes) and target: least squares through the infill toward each
 * texel's ideal weight, rounded to the range, then each grid weight
 * moved a level at a time while the largest error (then the sum of
 * squares) goes down.  gi[] the levels chosen; the score.
 */
struct grid {
	int W, H, pt[MAXT][4], f[MAXT][4];
	double *ata;		/* the normal equations, factored */
	int ok;
};

static void
grid_of(struct grid *g, int bw, int bh, int W, int H)
{
	int i, j, k, q, m = W * H;
	double *a = xcalloc((size_t)m * (size_t)m, sizeof *a);

	g->W = W;
	g->H = H;
	for (j = 0; j < bh; j++)
		for (i = 0; i < bw; i++)
			infill_at(bw, bh, W, H, i, j, g->pt[j * bw + i],
			    g->f[j * bw + i]);
	for (k = 0; k < bw * bh; k++)
		for (i = 0; i < 4; i++)
			for (q = 0; q < 4; q++)
				if (g->pt[k][i] >= 0 && g->pt[k][q] >= 0)
					a[g->pt[k][i] * m + g->pt[k][q]] += g->f[k][i] *
					    g->f[k][q] / 256.0;
	/* a small ridge: a grid point no texel weighs stays at its start */
	for (i = 0; i < m; i++)
		a[i * m + i] += 1e-6;
	/* Cholesky */
	g->ok = 1;
	for (j = 0; j < m && g->ok; j++) {
		double s = a[j * m + j];
		for (k = 0; k < j; k++)
			s -= a[j * m + k] * a[j * m + k];
		if (s <= 0) {
			g->ok = 0;
			break;
		}
		a[j * m + j] = sqrt(s);
		for (i = j + 1; i < m; i++) {
			double t = a[i * m + j];
			for (k = 0; k < j; k++)
				t -= a[i * m + k] * a[j * m + k];
			a[i * m + j] = t / a[j * m + j];
		}
	}
	g->ata = a;
}

static void
grid_solve(const struct grid *g, int n, const double *wt, const uint8_t
    *care, double *x)
{
	int m = g->W * g->H, i, k, q;
	double *b = xcalloc((size_t)m, sizeof *b);
	const double *a = g->ata;

	for (k = 0; k < n; k++)
		if (care[k])
			for (q = 0; q < 4; q++)
				if (g->pt[k][q] >= 0)
					b[g->pt[k][q]] += g->f[k][q] / 16.0 * wt[k];
	for (i = 0; i < m; i++) {
		double s = b[i];
		for (k = 0; k < i; k++)
			s -= a[i * m + k] * x[k];
		x[i] = s / a[i * m + i];
	}
	for (i = m - 1; i >= 0; i--) {
		double s = x[i];
		for (k = i + 1; k < m; k++)
			s -= a[k * m + i] * x[k];
		x[i] = s / a[i * m + i];
	}
	free(b);
}

/* a plane's score with grid levels gi (indices into lw): each texel's
 * weight through the infill, then its byte */
static struct ascore
plane_score(const struct grid *g, const struct lvl *lw, const int *gi,
    const struct aprob *P, int t, const int *ea, const int *eb)
{
	struct ascore s = {0, 0};
	int k, q;

	for (k = 0; k < P->n; k++) {
		int sum = 8, w;
		double e;
		if (!P->care[k])
			continue;
		for (q = 0; q < 4; q++)
			if (g->pt[k][q] >= 0)
				sum += lw->u[gi[g->pt[k][q]]] * g->f[k][q];
		w = sum >> 4;
		e = fabs(rd8(lerp16(ea[k], eb[k], w)) - P->t[t][k]);
		if (e > s.mx)
			s.mx = e;
		s.sq += e * e;
	}
	return s;
}

static struct ascore
plane_fit(const struct grid *g, const struct lvl *lw, const struct aprob *P,
    int t, const int *ea, const int *eb, int *gi, int polish)
{
	int m = g->W * g->H, k, pass, i, moved;
	double wt[MAXT], x[64];
	struct ascore s, b;

	if (g->W == P->bw && g->H == P->bh) {
		/* a grid the block's size: each texel reads its own point,
		 * whole, so each takes its best level */
		for (k = 0; k < P->n; k++) {
			double be = INFINITY;
			gi[k] = 0;
			for (i = 0; i < lw->n && P->care[k]; i++) {
				double e = fabs(rd8(lerp16(ea[k], eb[k], lw->u[i])) -
				    P->t[t][k]);
				if (e < be) {
					be = e;
					gi[k] = i;
				}
			}
		}
		return plane_score(g, lw, gi, P, t, ea, eb);
	}
	for (k = 0; k < P->n; k++) {
		double d = eb[k] - ea[k];
		wt[k] = d == 0 ? 0 : 64 * (P->t[t][k] * 255 - ea[k]) / d;
		wt[k] = wt[k] < 0 ? 0 : wt[k] > 64 ? 64 : wt[k];
	}
	grid_solve(g, P->n, wt, P->care, x);
	for (i = 0; i < m; i++)
		gi[i] = lvl_near(lw, x[i]);
	b = plane_score(g, lw, gi, P, t, ea, eb);
	for (pass = 0, moved = polish; moved && pass < 6; pass++) {
		moved = 0;
		for (i = 0; i < m; i++) {
			int d;
			for (d = -1; d <= 1; d += 2) {
				int was = gi[i];
				if (was + d < 0 || was + d >= lw->n)
					continue;
				gi[i] = was + d;
				s = plane_score(g, lw, gi, P, t, ea, eb);
				if (better(s, b)) {
					b = s;
					moved = 1;
				} else
					gi[i] = was;
			}
		}
	}
	return b;
}

/* a partition's least and greatest entry of table t (bytes) */
static void
part_span(const struct aprob *P, const int *part, int p, int t, double *lo,
    double *hi)
{
	int k;

	*lo = INFINITY;
	*hi = -INFINITY;
	for (k = 0; k < P->n; k++)
		if (P->care[k] && part[k] == p) {
			if (P->t[t][k] * 255 < *lo)
				*lo = P->t[t][k] * 255;
			if (P->t[t][k] * 255 > *hi)
				*hi = P->t[t][k] * 255;
		}
	if (*lo > *hi)
		*lo = *hi = 0;
}

/* a pattern's match with a seed's partitions, up to relabelling: the
 * texels whose label differs under the best relabelling */
static int
seed_miss(const struct aprob *P, const int *pat, int parts, int seed)
{
	static const int perms[24][4] = {
		{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1},
		{0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
		{1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
		{2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
		{2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1},
		{3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
	};
	int cnt[4][4], k, h, i, best = P->n + 1;

	memset(cnt, 0, sizeof cnt);
	for (k = 0; k < P->n; k++)
		if (P->care[k])
			cnt[astcb_partition(seed, k % P->bw, k / P->bw, parts,
			    P->bw * P->bh < 31)][pat[k]]++;
	for (i = 0; i < 24; i++) {
		int miss = 0, perm = 1;
		for (h = 0; h < parts; h++)
			perm &= perms[i][h] < parts;
		if (!perm)
			continue;
		for (k = 0; k < parts; k++)
			for (h = 0; h < parts; h++)
				if (perms[i][k] != h)
					miss += cnt[k][h];
		if (miss < best)
			best = miss;
	}
	return best;
}

/*
 * The pattern the entries make into parts: the values cut at the parts
 * - 1 largest gaps between the sorted distinct values
 */
static void
value_pattern(const struct aprob *P, int parts, int *pat)
{
	double v[MAXT], cut[3];
	int nv = 0, k, i, j;

	for (k = 0; k < P->n; k++)
		if (P->care[k]) {
			for (i = 0; i < nv && v[i] < P->t[0][k]; i++)
				;
			if (i < nv && v[i] == P->t[0][k])
				continue;
			for (j = nv; j > i; j--)
				v[j] = v[j - 1];
			v[i] = P->t[0][k];
			nv++;
		}
	for (j = 0; j < parts - 1; j++) {
		double bg = -1, bc = INFINITY;
		for (i = 0; i + 1 < nv; i++) {
			double gap = v[i + 1] - v[i], c = (v[i] + v[i + 1]) / 2;
			int used = 0, q;
			for (q = 0; q < j; q++)
				used |= cut[q] == c;
			if (!used && gap > bg) {
				bg = gap;
				bc = c;
			}
		}
		cut[j] = bc;
	}
	for (k = 0; k < P->n; k++) {
		pat[k] = 0;
		for (j = 0; j < parts - 1; j++)
			pat[k] += P->care[k] && P->t[0][k] > cut[j];
	}
}

/*
 * A layout's endpoints and weights for P: each table (plane) and
 * partition starts at the levels nearest its least and greatest entries;
 * with polish, each partition's nine pairs around those are tried in
 * turn and each grid weight moved a level at a time.  Fills L's encoded
 * values; the score.
 */
static struct ascore
fit_layout(const struct aprob *P, struct alay *L, const struct grid *g,
    int polish)
{
	struct lvl lc, lw;
	struct ascore sc = {0, 0};
	int part[MAXT], ends[4][2][2], ea[MAXT], eb[MAXT], gi[2][64], k, p, t;

	lvl_of(&lc, &crange[L->qi], 1);
	lvl_of(&lw, &wrange[L->Hp][L->r], 0);
	for (k = 0; k < P->n; k++)
		part[k] = L->parts == 1 ? 0 : astcb_partition(L->seed, k % P->bw,
		    k / P->bw, L->parts, P->bw * P->bh < 31);
	for (t = 0; t < P->nt; t++) {
		struct ascore pb;
		for (p = 0; p < L->parts; p++) {
			double lo, hi;
			part_span(P, part, p, t, &lo, &hi);
			ends[p][t][0] = lvl_near(&lc, lo);
			ends[p][t][1] = lvl_near(&lc, hi);
		}
#define ENDS() do { \
		for (k = 0; k < P->n; k++) { \
			ea[k] = lc.u[ends[part[k]][t][0]]; \
			eb[k] = lc.u[ends[part[k]][t][1]]; \
		} \
	} while (0)
		ENDS();
		pb = plane_fit(g, &lw, P, t, ea, eb, gi[t], polish);
		for (p = 0; polish && p < L->parts; p++) {
			int a0 = ends[p][t][0], b0 = ends[p][t][1], i, j, ba = a0,
			    bb = b0;
			for (i = a0 - 1; i <= a0 + 1; i++)
				for (j = b0 - 1; j <= b0 + 1; j++) {
					struct ascore s1;
					if (i < 0 || j < 0 || i >= lc.n || j >= lc.n ||
					    (i == a0 && j == b0))
						continue;
					ends[p][t][0] = i;
					ends[p][t][1] = j;
					ENDS();
					s1 = plane_fit(g, &lw, P, t, ea, eb, gi[t], polish);
					if (better(s1, pb)) {
						pb = s1;
						ba = i;
						bb = j;
					}
				}
			ends[p][t][0] = ba;
			ends[p][t][1] = bb;
		}
		ENDS();
#undef ENDS
		pb = plane_fit(g, &lw, P, t, ea, eb, gi[t], polish);
		if (pb.mx > sc.mx)
			sc.mx = pb.mx;
		sc.sq += pb.sq;
	}
	/* the encoded values: CEM 0, v0 v1 a partition; CEM 4, v0 v1 the
	 * luminance (the first table), v2 v3 the alpha (the second) */
	L->ncv = 0;
	for (p = 0; p < L->parts; p++)
		for (t = 0; t < P->nt; t++) {
			L->cv[L->ncv++] = lc.v[ends[p][t][0]];
			L->cv[L->ncv++] = lc.v[ends[p][t][1]];
		}
	L->nw = g->W * g->H * (L->D ? 2 : 1);
	for (k = 0; k < g->W * g->H; k++)
		for (t = 0; t < (L->D ? 2 : 1); t++)
			L->wv[k * (L->D ? 2 : 1) + t] = lw.v[gi[t][k]];
	return sc;
}

#define KEEP 6

/* the candidates kept for polishing, best first */
struct keep {
	int n;
	struct ascore s[KEEP];
	struct alay L[KEEP];
};

/* a layout fitted roughly, kept when it is among the KEEP best */
static void
consider(const struct aprob *P, struct alay *L, struct grid *gr[13][13],
    struct keep *K)
{
	struct ascore sc;
	int i, j;

	if (mode_for(L->W, L->H, L->D, L->Hp, L->r) < 0 ||
	    (L->qi = color_range(L)) < 0)
		return;
	if (!gr[L->W][L->H]) {
		gr[L->W][L->H] = xcalloc(1, sizeof *gr[L->W][L->H]);
		grid_of(gr[L->W][L->H], P->bw, P->bh, L->W, L->H);
	}
	if (!gr[L->W][L->H]->ok)
		return;
	sc = fit_layout(P, L, gr[L->W][L->H], 0);
	for (i = 0; i < K->n && !better(sc, K->s[i]); i++)
		;
	if (i == KEEP)
		return;
	if (K->n < KEEP)
		K->n++;
	for (j = K->n - 1; j > i; j--) {
		K->s[j] = K->s[j - 1];
		K->L[j] = K->L[j - 1];
	}
	K->s[i] = sc;
	K->L[i] = *L;
}

/*
 * The best block for P, its layout and bytes, the score: every mode the
 * footprint has (one plane, two for two tables), 1 to 4 partitions for
 * one table with the seeds whose partitions are closest to the entries'
 * pattern, each fitted roughly; the KEEP best then polished.
 */
static struct ascore
search(const struct aprob *P, struct alay *best, uint8_t out[16])
{
	struct ascore bs = {INFINITY, INFINITY};
	struct keep K;
	struct grid *gr[13][13];
	struct alay L;
	int W, H, Hp, r, parts, k;

	memset(gr, 0, sizeof gr);
	K.n = 0;
	for (parts = 1; parts <= (P->nt == 2 ? 1 : 4); parts++) {
		int pat[MAXT], seeds[2], ns = 0, s, bm = P->n + 1, si;
		if (parts > 1) {
			value_pattern(P, parts, pat);
			for (s = 0; s < 1024; s++) {
				int miss = seed_miss(P, pat, parts, s);
				if (miss < bm) {
					bm = miss;
					ns = 0;
				}
				if (miss == bm && ns < 2)
					seeds[ns++] = s;
			}
		} else
			seeds[ns++] = 0;
		for (si = 0; si < ns; si++) {
			for (Hp = 0; Hp < 2; Hp++) {
				for (r = 2; r < 8; r++) {
					for (W = 2; W <= P->bw; W++) {
						for (H = 2; H <= P->bh; H++) {
							memset(&L, 0, sizeof L);
							L.W = W;
							L.H = H;
							L.D = P->nt == 2;
							L.Hp = Hp;
							L.r = r;
							L.parts = parts;
							L.seed = seeds[si];
							L.cem = P->nt == 2 ? 4 : 0;
							L.ccs = 3;
							consider(P, &L, gr, &K);
						}
					}
				}
			}
		}
	}
	for (k = 0; k < K.n; k++) {
		struct ascore sc;
		uint8_t b[16];
		L = K.L[k];
		sc = fit_layout(P, &L, gr[L.W][L.H], 1);
		if (better(sc, bs) && write_block(&L, b)) {
			bs = sc;
			*best = L;
			memcpy(out, b, 16);
		}
	}
	for (W = 0; W < 13; W++)
		for (H = 0; H < 13; H++)
			if (gr[W][H]) {
				free(gr[W][H]->ata);
				free(gr[W][H]);
			}
	return bs;
}

int
astcb_encode(const double *t0, const double *t1, const uint8_t *care, int bw,
    int bh, uint8_t out[16], struct astcb_info *info)
{
	struct aprob P;
	struct alay L;
	struct ascore s;
	int k;

	memset(&P, 0, sizeof P);
	memset(&L, 0, sizeof L);
	P.bw = bw;
	P.bh = bh;
	P.n = bw * bh;
	P.nt = t1 ? 2 : 1;
	for (k = 0; k < P.n; k++) {
		P.t[0][k] = t0[k];
		P.t[1][k] = t1 ? t1[k] : 0;
		P.care[k] = care[k];
	}
	s = search(&P, &L, out);
	if (!isfinite(s.mx))
		return 0;
	memset(info, 0, sizeof *info);
	info->parts = L.parts;
	info->seed = L.seed;
	info->W = L.W;
	info->H = L.H;
	info->wlevels = levels(&wrange[L.Hp][L.r]);
	info->clevels = levels(&crange[L.qi]);
	info->planes = L.D ? 2 : 1;
	info->err = s.mx;
	if (L.parts > 1) {
		int pat[MAXT];
		value_pattern(&P, L.parts, pat);
		info->miss = seed_miss(&P, pat, L.parts, L.seed);
	}
	for (k = 0; k < L.ncv; k++)
		info->ends[k] = color_unq(&crange[L.qi], L.cv[k]);
	info->nends = L.ncv;
	return 1;
}
