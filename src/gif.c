/*
 * A small animated-GIF encoder (GIF89a) writing into memory.
 *
 * Indexed frames of one fixed size with a global palette of up to 256
 * colours.  Each frame stores only the bounding box of the pixels that
 * changed since the previous one (disposal "do not dispose"), and a frame
 * identical to the previous one is folded into its delay, so slowly
 * changing diagrams stay small.  LZW as in the GIF89a specification:
 * variable code width from min+1 to 12 bits, a clear code when the table
 * is full, codes packed LSB first into sub-blocks of up to 255 bytes.
 */
#include "gif.h"

#include <stdlib.h>
#include <string.h>

static void
put8(struct gif *g, int v)
{
	sb_putc(&g->out, v & 0xff);
}

static void
put16(struct gif *g, int v)
{
	put8(g, v);
	put8(g, v >> 8);
}

void
gif_begin(struct gif *g, int w, int h, const unsigned char (*pal)[3],
    int ncolors)
{
	int bits = 1, i;

	memset(g, 0, sizeof *g);
	g->w = w;
	g->h = h;
	while ((1 << bits) < ncolors)
		bits++;
	g->bits = bits < 2 ? 2 : bits;   /* LZW minimum code size is >= 2 */
	g->prev = xcalloc((size_t)w * h, 1);
	sb_puts(&g->out, "GIF89a");
	put16(g, w);
	put16(g, h);
	put8(g, 0x80 | ((bits - 1) << 4) | (bits - 1));  /* global table */
	put8(g, 0);         /* background colour */
	put8(g, 0);         /* aspect */
	for (i = 0; i < (1 << bits); i++) {
		put8(g, i < ncolors ? pal[i][0] : 0);
		put8(g, i < ncolors ? pal[i][1] : 0);
		put8(g, i < ncolors ? pal[i][2] : 0);
	}
	/* loop forever (NETSCAPE2.0 application extension) */
	put8(g, 0x21);
	put8(g, 0xff);
	put8(g, 11);
	sb_puts(&g->out, "NETSCAPE2.0");
	put8(g, 3);
	put8(g, 1);
	put16(g, 0);
	put8(g, 0);
}

/* ---- LZW ---- */

struct bits {
	struct gif *g;
	unsigned long acc;
	int n;
	unsigned char blk[255];
	int nb;
};

static void
flush_block(struct bits *b)
{
	int i;

	if (b->nb == 0)
		return;
	put8(b->g, b->nb);
	for (i = 0; i < b->nb; i++)
		put8(b->g, b->blk[i]);
	b->nb = 0;
}

static void
put_code(struct bits *b, int code, int width)
{
	b->acc |= (unsigned long)code << b->n;
	b->n += width;
	while (b->n >= 8) {
		b->blk[b->nb++] = (unsigned char)(b->acc & 0xff);
		if (b->nb == 255)
			flush_block(b);
		b->acc >>= 8;
		b->n -= 8;
	}
}

#define HSIZE 8192

static void
lzw(struct gif *g, const unsigned char *pix, int x0, int y0, int w, int h)
{
	int min = g->bits, clear = 1 << min, eoi = clear + 1;
	int next = eoi + 1, width = min + 1, prefix = -1, x, y;
	int *hkey = xmalloc(HSIZE * sizeof *hkey);
	short *hval = xmalloc(HSIZE * sizeof *hval);
	struct bits b;

	memset(&b, 0, sizeof b);
	b.g = g;
	memset(hkey, 0xff, HSIZE * sizeof *hkey);
	put8(g, min);
	put_code(&b, clear, width);
	for (y = y0; y < y0 + h; y++)
		for (x = x0; x < x0 + w; x++) {
			int k = pix[(size_t)y * g->w + x], key, slot;
			if (prefix < 0) {
				prefix = k;
				continue;
			}
			key = (prefix << 8) | k;
			slot = (key * 2654435761u) >> 19 & (HSIZE - 1);
			while (hkey[slot] != -1 && hkey[slot] != key)
				slot = (slot + 1) & (HSIZE - 1);
			if (hkey[slot] == key) {
				prefix = hval[slot];
				continue;
			}
			put_code(&b, prefix, width);
			if (next < 4096) {
				hkey[slot] = key;
				hval[slot] = (short)next;
				if (next == (1 << width) && width < 12)
					width++;
				next++;
			} else {
				put_code(&b, clear, width);
				memset(hkey, 0xff, HSIZE * sizeof *hkey);
				next = eoi + 1;
				width = min + 1;
			}
			prefix = k;
		}
	if (prefix >= 0)
		put_code(&b, prefix, width);
	put_code(&b, eoi, width);
	if (b.n > 0)
		put_code(&b, 0, 8 - b.n);
	flush_block(&b);
	put8(g, 0);   /* block terminator */
	free(hkey);
	free(hval);
}

void
gif_frame(struct gif *g, const unsigned char *pix, int delay_cs)
{
	int x0 = g->w, y0 = g->h, x1 = -1, y1 = -1, x, y;

	if (g->nframes == 0) {
		x0 = y0 = 0;
		x1 = g->w - 1;
		y1 = g->h - 1;
	} else {
		for (y = 0; y < g->h; y++)
			for (x = 0; x < g->w; x++)
				if (pix[(size_t)y * g->w + x] !=
				    g->prev[(size_t)y * g->w + x]) {
					if (x < x0)
						x0 = x;
					if (x > x1)
						x1 = x;
					if (y < y0)
						y0 = y;
					if (y > y1)
						y1 = y;
				}
		if (x1 < 0) {
			/* nothing changed: the previous frame shows longer */
			unsigned char *d = (unsigned char *)g->out.p + g->delay_at;
			int v = d[0] | d[1] << 8;
			v += delay_cs;
			d[0] = (unsigned char)(v & 0xff);
			d[1] = (unsigned char)(v >> 8);
			return;
		}
	}
	/* graphic control: disposal 1 (keep), delay in 1/100 s */
	put8(g, 0x21);
	put8(g, 0xf9);
	put8(g, 4);
	put8(g, 1 << 2);
	g->delay_at = g->out.n;
	put16(g, delay_cs);
	put8(g, 0);
	put8(g, 0);
	/* image descriptor: the changed rectangle, no local palette */
	put8(g, 0x2c);
	put16(g, x0);
	put16(g, y0);
	put16(g, x1 - x0 + 1);
	put16(g, y1 - y0 + 1);
	put8(g, 0);
	lzw(g, pix, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
	memcpy(g->prev, pix, (size_t)g->w * g->h);
	g->nframes++;
}

void
gif_end(struct gif *g)
{
	put8(g, 0x3b);
	free(g->prev);
	g->prev = NULL;
}
