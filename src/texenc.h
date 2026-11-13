/* SPDX-License-Identifier: GPL-3.0-or-later */
/* texenc.c: constant tables as compressed textures (see there) */
#ifndef UCAS_TEXENC_H
#define UCAS_TEXENC_H

#include "ucas.h"

/* a BC4 block's palette as a read returns it: eight f32 values */
void bc4_palette(int r0, int r1, double pal[8]);
void bc4_decode(const unsigned char b[8], double out[16]);
/* the best block for 16 targets (care[i] = 0: don't care); the largest
 * error */
double bc4_encode(const double t[16], const unsigned char care[16],
    unsigned char out[8]);
/*
 * A table of n values as BC4 blocks, rows of W: blocks (8 bytes each,
 * row by row) into *bytes (malloc'd), the decoded values into dec[n];
 * the largest error and where.  0 when the table is not one of numbers
 * in [0, 1] (or whole numbers 0..255, read as k/255).
 */
struct texenc {
	int n, W, H, bw, bh;	/* texels, rows of W, blocks across and down */
	int isint;		/* whole numbers 0..255, stored as k/255 */
	unsigned char *bytes;
	double *target, *dec;	/* what a read should return, and does */
	double err;
	int at;			/* the entry with the largest error */
	int sgn;		/* EAC: signed */
	double err16;		/* EAC: the largest error read at 16 bits */
};
int texenc_bc4(nid tab, int W, struct texenc *e);
void texenc_free(struct texenc *e);
/*
 * A table ucas emits (find's, phash's) searched for a lossless BC4
 * layout: rows of natw (a width the table has, 0: none) and of 4, 8, ...
 * up to its size, the exact one with the fewest bytes (fewer than the
 * table's own) into *W and *bytes; where BC4 has none, whole numbers
 * up to 2047 (signed: -1023..1023) as EAC's 11-bit values.  The format
 * into *fmt; 0 when there is none.
 */
enum { TEXENC_BC4, TEXENC_EAC, TEXENC_EAC_SIGNED };
int texenc_lossless(nid tab, int natw, int *W, int *bytes, int *fmt);
/* its line in a report: under ind, the entries named by what */
void texenc_note(nid tab, int natw, const char *what, const char *ind);

/*
 * EAC (R11_EAC, signed or not): the 11-bit value of a block's selector
 * (base as stored, a byte), what a read returns (x16: the decoder that
 * extends it to 16 bits), a block's sixteen values (texel x + 4y), the
 * best block for 16 targets (the largest error), and a table as blocks
 * as texenc_bc4 does (0 when it is not what EAC holds)
 */
int eac_value(int base, int mul, int tab, int sel, int sgn);
double eac_read(int v, int sgn, int x16);
void eac_decode(const unsigned char b[8], int sgn, int v[16]);
double eac_encode(const double t[16], const unsigned char care[16], int sgn,
    unsigned char out[8]);
int texenc_eac(nid tab, int W, int sgn, struct texenc *e);
int eac_mod(int tab, int sel);		/* modifier sel of table tab */

/*
 * etc2blk.c: an RGB8_ETC2 block's mode and its sixteen texels (x + 4y,
 * R, G, B), and the block for 16 targets in its red channel, every
 * encoding of that channel tried (G and B whatever the mode needs): the
 * largest error, and in info the mode, the flip of a two-colour block
 * and its values (individual and differential: the two reds and the two
 * modifier tables; T and H: the two reds and the distance; planar: O,
 * H, V).  ETC2's 8-bit alpha block likewise (RGBA8_ETC2_EAC's first 8
 * bytes): clamp(base + m mod, 0, 255).
 */
enum { ETC2_INDIVIDUAL, ETC2_DIFFERENTIAL, ETC2_T, ETC2_H, ETC2_PLANAR };
struct etc2_info {
	int mode, flip, v[4];
};
int etc2_mode(const uint8_t q[8]);
int etc1_mod(int tab, int k);		/* +a, +b, -a, -b of table tab */
void etc2_decode(const uint8_t q[8], uint8_t px[16][3]);
double etc2_encode_r(const double t[16], const uint8_t care[16], int srgb,
    uint8_t out[8], struct etc2_info *info);
double texenc_srgb8(int c);	/* the sRGB EOTF of code c, an f32 */
int etc2_alpha_value(int base, int mul, int tab, int sel);
void etc2_alpha_decode(const uint8_t q[8], int v[16]);
double etc2_alpha_encode(const double t[16], const uint8_t care[16],
    uint8_t out[8]);

/*
 * bc7blk.c: a BC7 block's mode (8: reserved) and its sixteen texels (x +
 * 4y, RGBA), and the block for 16 targets in red (tr), or red and alpha
 * (ta, modes 4 and 5): every mode, partition and way of holding red,
 * each subset's pair of endpoints searched over every value its field
 * has.  The largest error, alpha's in *aerr; in info the mode,
 * partition, rotation, index selection, red's endpoints per subset (and
 * alpha's), and each texel's index into red's (and alpha's) weights.
 */
struct bc7_info {
	int mode, part, rot, isb, ns, ribits;
	int red[3][2], alpha[2], ridx[16], aidx[16];
};
int bc7_mode(const uint8_t q[16]);
void bc7_decode(const uint8_t q[16], uint8_t px[16][4]);
double bc7_encode(const double *tr, const double *ta, const uint8_t care[16],
    int srgb, uint8_t out[16], struct bc7_info *info, double *aerr);

/*
 * bc7blk.c, BC6H: a block's texels (x + 4y, R, G, B as half bits;
 * reserved: 0), what half bits read as, and the block for 16 targets in
 * red (sgn: SFLOAT), each mode and partition searched near each subset's
 * least and greatest target: the largest error; in info the mode number,
 * the partition, the subsets, red's quantized endpoints and the indices
 */
struct bc6_info {
	int mode, part, ns, x[4], idx[16];
};
void bc6h_decode(const uint8_t q[16], int sgn, uint16_t px[16][3]);
double bc6h_half(int h);
double bc6h_encode_r(const double t[16], const uint8_t care[16], int sgn,
    uint8_t out[16], struct bc6_info *info);

/*
 * astcblk.c: an ASTC block (2D, LDR) decoded, each texel's four channels
 * as UNORM16 (1 for a block the profile refuses: the error colour), and
 * a block searched for one table (t1 NULL) or two (two planes); 0 when
 * none fits.  srgb 1 decodes as ASTC's sRGB formats (the first table's
 * endpoints widened with 0x80 below their byte, read as that byte
 * through the EOTF), srgb 2 widens alpha that way too, as astcenc and
 * Mesa do against the specification.  A texel reads (C >> 8)/255 in the UNORM8 decode mode,
 * astcb_float16(C) in the float16 one.
 */
struct astcb_info {
	int parts, seed, miss;	/* partitions; the seed, and how many texels
				 * its partitions put off the entries' pattern */
	int W, H, wlevels, clevels, planes;
	int nends, ends[18];	/* the endpoint values, unquantised */
	double err;		/* the largest error the search computed */
};
int astcb_decode(const uint8_t blk[16], int bw, int bh, int srgb,
    uint16_t *out);
int astcb_encode(const double *t0, const double *t1, const uint8_t *care,
    int bw, int bh, int srgb, uint8_t out[16], struct astcb_info *info);
int astcb_partition(int seed, int x, int y, int count, int small);
double astcb_float16(int c);

struct P;
int texenc_command(struct P *p);		/* parse.c */
const char *texenc_nlp_raw(const char *sent);	/* nlp.c */
int texenc_nlp_english(const char *line);	/* is it such a sentence */

#endif
