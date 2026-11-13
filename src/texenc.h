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
};
int texenc_bc4(nid tab, int W, struct texenc *e);
void texenc_free(struct texenc *e);
/*
 * A table ucas emits (find's, phash's) searched for a lossless BC4
 * layout: rows of natw (a width the table has, 0: none) and of 4, 8, ...
 * up to its size, the exact one with the fewest bytes (fewer than the
 * table's own) into *W and *bytes; 0 when there is none.
 */
int texenc_lossless(nid tab, int natw, int *W, int *bytes);
/* its line in a report: under ind, the entries named by what */
void texenc_note(nid tab, int natw, const char *what, const char *ind);

/*
 * astcblk.c: an ASTC block (2D, LDR) decoded, each texel's four channels
 * as UNORM16 (1 for a block the profile refuses: the error colour), and
 * a block searched for one table (t1 NULL) or two (two planes); 0 when
 * none fits.  A texel reads (C >> 8)/255 in the UNORM8 decode mode,
 * astcb_float16(C) in the float16 one.
 */
struct astcb_info {
	int parts, seed, miss;	/* partitions; the seed, and how many texels
				 * its partitions put off the entries' pattern */
	int W, H, wlevels, clevels, planes;
	int nends, ends[18];	/* the endpoint values, unquantised */
	double err;		/* the largest error the search computed */
};
int astcb_decode(const uint8_t blk[16], int bw, int bh, uint16_t *out);
int astcb_encode(const double *t0, const double *t1, const uint8_t *care,
    int bw, int bh, uint8_t out[16], struct astcb_info *info);
int astcb_partition(int seed, int x, int y, int count, int small);
double astcb_float16(int c);

struct P;
int texenc_command(struct P *p);		/* parse.c */
const char *texenc_nlp_raw(const char *sent);	/* nlp.c */
int texenc_nlp_english(const char *line);	/* is it such a sentence */

#endif
