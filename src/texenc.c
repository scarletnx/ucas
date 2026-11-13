/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * texenc.c: constant tables as compressed textures.
 *
 * A block-compressed format decodes in fixed function: a few parameters
 * a block, interpolated by the decoder.  So a table that the decoder's
 * palette reproduces is stored in less than its bytes, and read at the
 * same price a fetch.  The decoder's semantics are exact here, and the
 * encoder is a search over them with the table as its target, not an
 * image: every pair of endpoints, each texel on its nearest palette
 * entry.
 *
 * BC4 (RGTC1, unsigned; BC5 is two of them): a 4 x 4 block is 8 bytes,
 * two 8-bit endpoints r0, r1 and sixteen 3-bit selectors, least
 * significant first, texel (x, y) at bits 3(4y + x).  With r0 > r1 the
 * palette is r0, r1 and the six values ((8 - i) r0 + (i - 1) r1) / 7,
 * i = 2..7; otherwise r0, r1, the four ((6 - i) r0 + (i - 1) r1) / 5,
 * i = 2..5, then 0 and 255.  A texel is its palette entry over 255, as a
 * rational: 0 to 1785 over 1785, or over 1275.  A read gives its f32.
 *
 *   texenc T [rows W] [tol E]
 *       T as BC4 blocks (the table read as rows of W texels, 4 by
 *       default; a last column or row that does not fill a block is
 *       padded with don't-cares): each block's endpoints, selectors and
 *       bytes, the table it decodes to, and whether that is the table
 *       exactly (as f32) or within E
 *   texenc T, U [rows W] [tol E]
 *       T and U, two tables of one shape, as BC5 (RGTC2): a block is 16
 *       bytes, T's BC4 block (red) then U's (green), each channel
 *       encoded on its own
 *   texenc T astc BWxBH [rows W] [tol E]
 *       T as ASTC blocks of BW x BH texels (2D, LDR; astcblk.c searches
 *       each): a block's grid, ranges, partitions and seed, endpoints and
 *       bytes, read in the UNORM8 decode mode (a texel's byte over 255);
 *       the default float16 mode's reads beside
 *   texenc T, U astc BWxBH [rows W] [tol E]
 *       T and U in one block's two planes, T the luminance (R, G, B) and
 *       U the alpha
 *   texenc T [, U] astc BWxBH srgb [rows W] [tol E]
 *       the same as ASTC_BWxBH_SRGB: the luminance's endpoints widened
 *       with 0x80 below them, and read through the sRGB EOTF of the top
 *       byte; alpha as before
 *   texenc T [, U] eac [rows W] [tol E]
 *       T as R11_EAC blocks (T and U as RG11_EAC), signed when an entry
 *       is negative: each block's base, multiplier, modifier table and
 *       selectors, every one of the 65536 headers tried; whole numbers
 *       are the format's 11-bit values, k/2047 (signed, k/1023)
 *   texenc T [, U] etc2 [rows W] [tol E]
 *       T in the red channel of RGB8_ETC2 blocks (etc2blk.c), every
 *       encoding of that channel tried (individual, differential, T, H,
 *       planar); T and U as RGBA8_ETC2_EAC, U in the 8-bit alpha block
 *   texenc T [, U] etc2 srgb [rows W] [tol E]
 *       the same as SRGB8_ETC2 (SRGB8_ALPHA8_ETC2_EAC): red reads the sRGB
 *       EOTF of its code over 255, alpha stays linear
 *   texenc T [, U] bc7 [rows W] [tol E]
 *       T in the red channel of BC7 blocks (bc7blk.c), every mode,
 *       partition and way of holding red tried, each subset's endpoints
 *       over every value of their field; T and U in red and alpha, in
 *       the modes where alpha has indices of its own (4 and 5)
 *   texenc T [, U] bc7 srgb [rows W] [tol E]
 *       the same as BC7_SRGB: red reads the sRGB EOTF of its code over
 *       255, alpha stays linear
 *   texenc T bc6h [rows W] [tol E]
 *       T, numbers of a half float's range, in the red channel of BC6H
 *       blocks (UFLOAT, or SFLOAT when an entry is negative): each mode
 *       and partition searched near each subset's least and greatest
 *       entry, not every encoding
 */
#include "calc.h"
#include "report.h"
#include "texenc.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void
bc4_palette(int r0, int r1, double pal[8])
{
	int i;

	pal[0] = r0 / 255.0;
	pal[1] = r1 / 255.0;
	if (r0 > r1) {
		for (i = 2; i < 8; i++)
			pal[i] = ((8 - i) * r0 + (i - 1) * r1) / (7 * 255.0);
	} else {
		for (i = 2; i < 6; i++)
			pal[i] = ((6 - i) * r0 + (i - 1) * r1) / (5 * 255.0);
		pal[6] = 0;
		pal[7] = 1;
	}
	/* what a read returns: the rational's f32 (a quotient of integers
	 * below 2^11 is never close enough to a midpoint of two floats for
	 * the double in between to matter) */
	for (i = 0; i < 8; i++)
		pal[i] = (double)(float)pal[i];
}

void
bc4_decode(const unsigned char b[8], double out[16])
{
	double pal[8];
	uint64_t bits = 0;
	int i;

	bc4_palette(b[0], b[1], pal);
	for (i = 0; i < 6; i++)
		bits |= (uint64_t)b[2 + i] << (8 * i);
	for (i = 0; i < 16; i++)
		out[i] = pal[bits >> (3 * i) & 7];
}

/*
 * The block for 16 targets (care[i] = 0: any value will do): the
 * endpoints with the smallest largest error, then the smallest sum of
 * squares, then the first found, so the answer does not depend on
 * anything but the targets.  Returns the largest error.
 */
double
bc4_encode(const double t[16], const unsigned char care[16],
    unsigned char out[8])
{
	double best = INFINITY, bestsq = INFINITY, pal[8];
	int r0, r1, i, k, sel[16], bsel[16], b0 = 0, b1 = 0;
	uint64_t bits = 0;

	memset(bsel, 0, sizeof bsel);
	for (r0 = 0; r0 < 256; r0++)
		for (r1 = 0; r1 < 256; r1++) {
			double mx = 0, sq = 0;
			bc4_palette(r0, r1, pal);
			for (i = 0; i < 16 && mx <= best; i++) {
				double be = INFINITY;
				sel[i] = 0;
				if (!care[i])
					continue;
				for (k = 0; k < 8; k++) {
					double e = fabs(pal[k] - t[i]);
					if (e < be) {
						be = e;
						sel[i] = k;
					}
				}
				if (be > mx)
					mx = be;
				sq += be * be;
			}
			if (i < 16)
				continue;
			if (mx < best || (mx == best && sq < bestsq)) {
				best = mx;
				bestsq = sq;
				b0 = r0;
				b1 = r1;
				memcpy(bsel, sel, sizeof sel);
			}
		}
	out[0] = (unsigned char)b0;
	out[1] = (unsigned char)b1;
	for (i = 0; i < 16; i++)
		bits |= (uint64_t)bsel[i] << (3 * i);
	for (i = 0; i < 6; i++)
		out[2 + i] = (unsigned char)(bits >> (8 * i));
	return best;
}

void
texenc_free(struct texenc *e)
{
	free(e->bytes);
	free(e->target);
	free(e->dec);
	memset(e, 0, sizeof *e);
}

int
texenc_bc4(nid tab, int W, struct texenc *e)
{
	uint32_t j, n;
	int bx, by, i, isint = 1;

	memset(e, 0, sizeof *e);
	if (N.op[tab] != O_TABLE || N.b[tab] == 0)
		return 0;
	n = N.b[tab];
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v;
		if (!is_const(x))
			return 0;
		v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (!TY_ISINT(N.ty[x]) || v > 255)
			isint = 0;
		if (!(v >= 0))
			return 0;
	}
	e->n = (int)n;
	e->W = W;
	e->H = (int)((n + (uint32_t)W - 1) / (uint32_t)W);
	e->bw = (W + 3) / 4;
	e->bh = (e->H + 3) / 4;
	e->isint = isint;
	e->target = xcalloc(n, sizeof *e->target);
	e->dec = xcalloc(n, sizeof *e->dec);
	e->bytes = xcalloc((size_t)e->bw * (size_t)e->bh, 8);
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (!isint && v > 1) {
			texenc_free(e);
			return 0;
		}
		e->target[j] = (double)(float)(isint ? v / 255 : v);
	}
	for (by = 0; by < e->bh; by++)
		for (bx = 0; bx < e->bw; bx++) {
			double t[16], d[16];
			unsigned char care[16], *b = e->bytes + 8 * (by * e->bw + bx);
			for (i = 0; i < 16; i++) {
				int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
				long at = (long)y * W + x;
				care[i] = x < W && at < (long)n;
				t[i] = care[i] ? e->target[at] : 0;
			}
			bc4_encode(t, care, b);
			bc4_decode(b, d);
			for (i = 0; i < 16; i++) {
				int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
				long at = (long)y * W + x;
				if (!care[i])
					continue;
				e->dec[at] = d[i];
				if (fabs(d[i] - t[i]) > e->err) {
					e->err = fabs(d[i] - t[i]);
					e->at = (int)at;
				}
			}
		}
	return 1;
}

/* block k's endpoints, selectors and bytes, printed and reported */
static void
pr_block(const unsigned char *q, struct sbuf *b)
{
	uint64_t bits = 0;
	int i;

	for (i = 0; i < 6; i++)
		bits |= (uint64_t)q[2 + i] << (8 * i);
	printf("endpoints %d, %d (%s), selectors", q[0], q[1], q[0] > q[1] ?
	    "six steps of 1/7 between them" : "four steps of 1/5, then 0 and 1");
	sb_clear(b);
	for (i = 0; i < 16; i++) {
		printf(" %d", (int)(bits >> (3 * i) & 7));
		sb_printf(b, "%s%d", i ? " " : "", (int)(bits >> (3 * i) & 7));
	}
	rep_int("r0", q[0]);
	rep_int("r1", q[1]);
	rep_str("selectors", sb_str(b));
}

/* a channel's decoded table, and how close it is */
static void
pr_channel(const char *name, const struct texenc *e, double tol, int has_tol,
    struct sbuf *b)
{
	int i;

	sb_clear(b);
	sb_putc(b, '[');
	for (i = 0; i < e->n; i++)
		sb_printf(b, "%s%.9g", i ? ", " : "", e->dec[i]);
	sb_putc(b, ']');
	printf("  %sdecodes to %s\n", name, sb_str(b));
	rep_str("decoded", sb_str(b));
	rep_bool("exact", e->err == 0);
	rep_num("max_error", e->err);
	if (e->err == 0)
		printf("  %sexact: every texel is its entry's f32%s\n", name,
		    e->isint ? " over 255" : "");
	else {
		printf("  %snot exact: the closest blocks are within %.3g of the "
		    "table (entry %d: %.9g as %.9g), every pair of endpoints "
		    "tried\n", name, e->err, e->at, e->target[e->at],
		    e->dec[e->at]);
		if (has_tol) {
			printf("  %s%s the %g allowed (tol)\n", name, e->err <= tol ?
			    "within" : "more than", tol);
			rep_bool("within_tol", e->err <= tol);
		}
	}
}

/*
 * Two tables of one shape as BC5: each its own BC4 channel, red the
 * first; a block is the red channel's 8 bytes, then the green's.
 */
static void
cmd_texenc5(const struct func *f, const struct func *g, int W, double tol,
    int has_tol)
{
	struct texenc e[2];
	const struct func *fs[2] = {f, g};
	const char *nm[2], *ch[2] = {"red", "green"};
	struct sbuf b = {0};
	int c, k, i, nb;

	for (c = 0; c < 2; c++) {
		nm[c] = fs[c]->anon ? (c ? "the second table" : "the first table") :
		    sym_str(fs[c]->name);
		if (fs[c]->np != 0 || N.op[fs[c]->body] != O_TABLE)
			fail("texenc takes tables [t0, t1, ...] (or their names), not "
			    "%s", fs[c]->anon ? node_str(fs[c]->body) :
			    sym_str(fs[c]->name));
	}
	if (N.b[f->body] != N.b[g->body])
		fail("texenc: BC5's two channels are tables of one shape; %s has "
		    "%u entries and %s %u", nm[0], N.b[f->body], nm[1],
		    N.b[g->body]);
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	for (c = 0; c < 2; c++)
		if (!texenc_bc4(fs[c]->body, W, &e[c])) {
			if (c)
				texenc_free(&e[0]);
			fail("texenc: a BC5 channel holds numbers from 0 to 1 (or "
			    "whole numbers 0 to 255, read as k/255); %s has an entry "
			    "that is neither", nm[c]);
		}
	nb = e[0].bw * e[0].bh;
	rep_begin("texenc");
	rep_str("format", "BC5_UNORM");
	rep_int("entries", e[0].n);
	rep_int("width", e[0].W);
	rep_int("height", e[0].H);
	rep_int("bytes", nb * 16);
	printf("texenc %s, %s: %d entr%s a channel as BC5 (red %s, green %s), "
	    "rows of %d (%d x %d texels): %d block%s, %d bytes (%d as "
	    "R8G8_UNORM, %d as R32G32_SFLOAT)\n", nm[0], nm[1], e[0].n,
	    e[0].n == 1 ? "y" : "ies", nm[0], nm[1], e[0].W, e[0].W, e[0].H, nb,
	    nb == 1 ? "" : "s", nb * 16, 2 * e[0].W * e[0].H,
	    8 * e[0].W * e[0].H);
	rep_list("blocks");
	for (k = 0; k < nb; k++) {
		rep_obj(NULL);
		rep_int("x", k % e[0].bw);
		rep_int("y", k / e[0].bw);
		for (c = 0; c < 2; c++) {
			printf("  block (%d, %d) %s: ", k % e[0].bw, k / e[0].bw, ch[c]);
			rep_obj(ch[c]);
			pr_block(e[c].bytes + 8 * k, &b);
			rep_end();
			printf("\n");
		}
		sb_clear(&b);
		printf("  block (%d, %d) bytes", k % e[0].bw, k / e[0].bw);
		for (c = 0; c < 2; c++)
			for (i = 0; i < 8; i++) {
				printf(" %02x", e[c].bytes[8 * k + i]);
				sb_printf(&b, "%s%02x", c || i ? " " : "",
				    e[c].bytes[8 * k + i]);
			}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	for (c = 0; c < 2; c++) {
		char pre[16];
		snprintf(pre, sizeof pre, "%s: ", ch[c]);
		rep_obj(ch[c]);
		pr_channel(pre, &e[c], tol, has_tol, &b);
		rep_end();
	}
	rep_bool("exact", e[0].err == 0 && e[1].err == 0);
	if (has_tol)
		rep_bool("within_tol", e[0].err <= tol && e[1].err <= tol);
	printf("  a read of the texture is that read of the decoded tables, red "
	    "and green; no model prices a BC5 read apart from its uncompressed "
	    "one: not measured\n");
	free(b.p);
	texenc_free(&e[0]);
	texenc_free(&e[1]);
}

static int eac_lossless(nid tab, int natw, int stored, int *W, int *bytes,
    int *sgn);

int
texenc_lossless(nid tab, int natw, int *W, int *bytes, int *fmt)
{
	struct texenc e;
	int w, n, k, best = 0, stored, sgn;

	*fmt = TEXENC_BC4;
	if (N.op[tab] != O_TABLE || N.b[tab] < 2)
		return 0;
	n = (int)N.b[tab];
	stored = n * ((TYW(N.ty[N.lst[N.a[tab]]]) + 7) / 8);
	for (k = -1, w = natw; k <= 12; w = 4 << ++k) {
		if (w < 1 || w > 4096 || (k >= 0 && w > 4 && w / 2 >= n) ||
		    (k >= 0 && w == natw))
			continue;
		if (!texenc_bc4(tab, w, &e))
			break;	/* not numbers BC4 holds */
		if (e.err == 0 && e.bw * e.bh * 8 < stored && (!best ||
		    e.bw * e.bh * 8 < *bytes)) {
			best = 1;
			*W = w;
			*bytes = e.bw * e.bh * 8;
		}
		texenc_free(&e);
	}
	/* EAC's 11-bit values where BC4 has no exact layout */
	if (!best && eac_lossless(tab, natw, stored, W, bytes, &sgn)) {
		*fmt = sgn ? TEXENC_EAC_SIGNED : TEXENC_EAC;
		best = 1;
	}
	return best;
}

void
texenc_note(nid tab, int natw, const char *what, const char *ind)
{
	int W, bytes, n, fmt;
	ty_t t;

	if (!texenc_lossless(tab, natw, &W, &bytes, &fmt))
		return;
	n = (int)N.b[tab];
	t = N.ty[N.lst[N.a[tab]]];
	printf("%sthe %s are exact as %s blocks, rows of %d: %d bytes against "
	    "%d as %s (texenc%s; a read returns k/%s; a compressed read has no "
	    "price of its own, so what is priced stays as it is)\n", ind, what,
	    fmt == TEXENC_BC4 ? "BC4_UNORM" : fmt == TEXENC_EAC ? "R11_EAC" :
	    "SIGNED_R11_EAC", W, bytes, n * ((TYW(t) + 7) / 8), ty_str(t),
	    fmt == TEXENC_BC4 ? "" : " ... eac", fmt == TEXENC_BC4 ? "255" :
	    fmt == TEXENC_EAC ? "2047" : "1023");
	rep_obj(fmt == TEXENC_BC4 ? "bc4" : "eac");
	if (fmt != TEXENC_BC4)
		rep_bool("signed", fmt == TEXENC_EAC_SIGNED);
	rep_str("entries", what);
	rep_int("width", W);
	rep_int("bytes", bytes);
	rep_int("bytes_as_stored", (long long)n * ((TYW(t) + 7) / 8));
	rep_end();
}

static void
cmd_texenc(const struct func *f, int W, double tol, int has_tol)
{
	struct texenc e;
	const char *fn = f->anon ? "the table" : sym_str(f->name);
	struct sbuf b = {0};
	int i, k, nb;

	if (f->np != 0 || N.op[f->body] != O_TABLE)
		fail("texenc takes a table [t0, t1, ...] (or its name), not %s",
		    f->anon ? node_str(f->body) : sym_str(f->name));
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	if (!texenc_bc4(f->body, W, &e))
		fail("texenc: BC4 holds numbers from 0 to 1 (or whole numbers 0 "
		    "to 255, read as k/255); %s has an entry that is neither",
		    fn);
	nb = e.bw * e.bh;
	rep_begin("texenc");
	rep_str("format", "BC4_UNORM");
	rep_int("entries", e.n);
	rep_int("width", e.W);
	rep_int("height", e.H);
	rep_int("bytes", nb * 8);
	printf("texenc %s: %d entr%s as BC4, rows of %d (%d x %d texels): %d "
	    "block%s, %d bytes (%d as R8_UNORM, %d as R32_SFLOAT)%s\n", fn, e.n,
	    e.n == 1 ? "y" : "ies", e.W, e.W, e.H, nb, nb == 1 ? "" : "s",
	    nb * 8, e.W * e.H, 4 * e.W * e.H, e.isint ? "; whole numbers k are "
	    "stored as k/255" : "");
	rep_list("blocks");
	for (k = 0; k < nb; k++) {
		const unsigned char *q = e.bytes + 8 * k;
		uint64_t bits = 0;
		for (i = 0; i < 6; i++)
			bits |= (uint64_t)q[2 + i] << (8 * i);
		printf("  block (%d, %d): endpoints %d, %d (%s), selectors", k %
		    e.bw, k / e.bw, q[0], q[1], q[0] > q[1] ? "six steps of 1/7 "
		    "between them" : "four steps of 1/5, then 0 and 1");
		sb_clear(&b);
		for (i = 0; i < 16; i++) {
			printf(" %d", (int)(bits >> (3 * i) & 7));
			sb_printf(&b, "%s%d", i ? " " : "", (int)(bits >> (3 * i) & 7));
		}
		printf(", bytes");
		rep_obj(NULL);
		rep_int("x", k % e.bw);
		rep_int("y", k / e.bw);
		rep_int("r0", q[0]);
		rep_int("r1", q[1]);
		rep_str("selectors", sb_str(&b));
		sb_clear(&b);
		for (i = 0; i < 8; i++) {
			printf(" %02x", q[i]);
			sb_printf(&b, "%s%02x", i ? " " : "", q[i]);
		}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	sb_clear(&b);
	sb_putc(&b, '[');
	for (i = 0; i < e.n; i++)
		sb_printf(&b, "%s%.9g", i ? ", " : "", e.dec[i]);
	sb_putc(&b, ']');
	printf("  decodes to %s\n", sb_str(&b));
	rep_str("decoded", sb_str(&b));
	rep_bool("exact", e.err == 0);
	rep_num("max_error", e.err);
	if (e.err == 0)
		printf("  exact: every texel is its entry's f32%s (16 texels a "
		    "block against 8 palette values, every pair of endpoints "
		    "tried)\n", e.isint ? " over 255" : "");
	else {
		printf("  not exact: the closest blocks are within %.3g of the "
		    "table (entry %d: %.9g as %.9g), every pair of endpoints "
		    "tried\n", e.err, e.at, e.target[e.at], e.dec[e.at]);
		if (has_tol) {
			printf("  %s the %g allowed (tol)\n", e.err <= tol ? "within" :
			    "more than", tol);
			rep_bool("within_tol", e.err <= tol);
		}
	}
	printf("  a nearest or filtered read of the texture is that read of the "
	    "decoded table (fetch_..., tex_linear...); no model prices a BC4 "
	    "read apart from its uncompressed one: not measured\n");
	free(b.p);
	texenc_free(&e);
}

/*
 * EAC: ETC2's single-channel blocks, R11_EAC (RG11_EAC is two of them),
 * signed or not.  A 4 x 4 block is 8 bytes read as one big-endian word:
 * a base codeword (bits 63..56), a multiplier m (55..52), a table index
 * (51..48) that chooses eight modifiers, and sixteen 3-bit selectors
 * column by column, texel (x, y) at bits 47 - 3(4x + y) and the two
 * below.  A texel is the 11-bit value clamp(8 base + 4 + 8 m mod, 0,
 * 2047) over 2047, 8 base + 4 + mod when m is 0; signed, clamp(8 base +
 * 8 m mod, -1023, 1023) over 1023, the base a two's-complement byte with
 * -128 read as -127 (Khronos Data Format Specification 1.3, 22.5 and
 * 22.7).  That quotient is the format's value; a decoder may extend the
 * 11 bits to 16 by bit replication instead, (v << 5) + (v >> 6) over
 * 65535 (signed: the magnitude's (v << 5) + (v >> 5) over 32767), which
 * is a different number, so both are reported.
 */
static const signed char eac_mods[16][8] = {
	{-3, -6, -9, -15, 2, 5, 8, 14}, {-3, -7, -10, -13, 2, 6, 9, 12},
	{-2, -5, -8, -13, 1, 4, 7, 12}, {-2, -4, -6, -13, 1, 3, 5, 12},
	{-3, -6, -8, -12, 2, 5, 7, 11}, {-3, -7, -9, -11, 2, 6, 8, 10},
	{-4, -7, -8, -11, 3, 6, 7, 10}, {-3, -5, -8, -11, 2, 4, 7, 10},
	{-2, -6, -8, -10, 1, 5, 7, 9}, {-2, -5, -8, -10, 1, 4, 7, 9},
	{-2, -4, -8, -10, 1, 3, 7, 9}, {-2, -5, -7, -10, 1, 4, 6, 9},
	{-3, -4, -7, -10, 2, 3, 6, 9}, {-1, -2, -3, -10, 0, 1, 2, 9},
	{-4, -6, -8, -9, 3, 5, 7, 8}, {-3, -5, -7, -9, 2, 4, 6, 8}};

int
eac_value(int base, int mul, int tab, int sel, int sgn)
{
	int mod = eac_mods[tab & 15][sel & 7], v;

	if (sgn) {
		int b = base >= 128 ? base - 256 : base;
		if (b == -128)
			b = -127;
		v = 8 * b + (mul ? 8 * mod * mul : mod);
		return v < -1023 ? -1023 : v > 1023 ? 1023 : v;
	}
	v = 8 * base + 4 + (mul ? 8 * mod * mul : mod);
	return v < 0 ? 0 : v > 2047 ? 2047 : v;
}

int
eac_mod(int tab, int sel)
{
	return eac_mods[tab & 15][sel & 7];
}

double
eac_read(int v, int sgn, int x16)
{
	int a = v < 0 ? -v : v;

	if (!x16)
		return (double)(float)(v / (sgn ? 1023.0 : 2047.0));
	a = sgn ? (a << 5) + (a >> 5) : (a << 5) + (a >> 6);
	return (double)(float)((v < 0 ? -a : a) / (sgn ? 32767.0 : 65535.0));
}

void
eac_decode(const unsigned char b[8], int sgn, int v[16])
{
	uint64_t w = 0;
	int i;

	for (i = 0; i < 8; i++)
		w = w << 8 | b[i];
	for (i = 0; i < 16; i++)
		v[i] = eac_value(b[0], (int)(w >> 52 & 15), (int)(w >> 48 & 15),
		    (int)(w >> (45 - 3 * (4 * (i & 3) + (i >> 2))) & 7), sgn);
}

double
eac_encode(const double t[16], const unsigned char care[16], int sgn,
    unsigned char out[8])
{
	double best = INFINITY, bestsq = INFINITY, pal[8];
	int b, mul, tab, i, k, sel[16], bsel[16], bb = 0, bm = 0, bt = 0;
	uint64_t w;

	memset(bsel, 0, sizeof bsel);
	/* signed, -127..127 (-128 is a second spelling of -127) */
	for (b = sgn ? -127 : 0; b <= (sgn ? 127 : 255); b++)
		for (mul = 0; mul < 16; mul++)
			for (tab = 0; tab < 16; tab++) {
				double mx = 0, sq = 0;
				for (k = 0; k < 8; k++)
					pal[k] = eac_read(eac_value(b & 255, mul, tab, k,
					    sgn), sgn, 0);
				for (i = 0; i < 16 && mx <= best; i++) {
					double be = INFINITY;
					sel[i] = 0;
					if (!care[i])
						continue;
					for (k = 0; k < 8; k++) {
						double e = fabs(pal[k] - t[i]);
						if (e < be) {
							be = e;
							sel[i] = k;
						}
					}
					if (be > mx)
						mx = be;
					sq += be * be;
				}
				if (i < 16)
					continue;
				if (mx < best || (mx == best && sq < bestsq)) {
					best = mx;
					bestsq = sq;
					bb = b & 255;
					bm = mul;
					bt = tab;
					memcpy(bsel, sel, sizeof sel);
				}
			}
	w = (uint64_t)bb << 56 | (uint64_t)bm << 52 | (uint64_t)bt << 48;
	for (i = 0; i < 16; i++)
		w |= (uint64_t)bsel[i] << (45 - 3 * (4 * (i & 3) + (i >> 2)));
	for (i = 0; i < 8; i++)
		out[i] = (unsigned char)(w >> (56 - 8 * i));
	return best;
}

/* does a table hold what EAC (signed or not) stores: whole numbers
 * 0..2047 (-1023..1023), read as k/2047 (k/1023), or numbers in [0, 1]
 * ([-1, 1]); 1 when it has a negative entry, in *neg */
static int
eac_holds(nid tab, int sgn, int *neg)
{
	uint32_t j, n = N.b[tab];
	int isint = 1;

	*neg = 0;
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v;
		if (!is_const(x))
			return 0;
		v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (!(v == v))
			return 0;
		if (v < 0)
			*neg = 1;
		if (!TY_ISINT(N.ty[x]))
			isint = 0;
	}
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (isint ? (sgn ? v < -1023 || v > 1023 : v < 0 || v > 2047) :
		    (sgn ? v < -1 || v > 1 : v < 0 || v > 1))
			return 0;
	}
	return 1;
}

int
texenc_eac(nid tab, int W, int sgn, struct texenc *e)
{
	uint32_t j, n;
	int bx, by, i, neg;

	memset(e, 0, sizeof *e);
	if (N.op[tab] != O_TABLE || N.b[tab] == 0 || !eac_holds(tab, sgn,
	    &neg))
		return 0;
	n = N.b[tab];
	e->n = (int)n;
	e->W = W;
	e->H = (int)((n + (uint32_t)W - 1) / (uint32_t)W);
	e->bw = (W + 3) / 4;
	e->bh = (e->H + 3) / 4;
	e->sgn = sgn;
	e->isint = 1;
	for (j = 0; j < n; j++)
		if (!TY_ISINT(N.ty[N.lst[N.a[tab] + j]]))
			e->isint = 0;
	e->target = xcalloc(n, sizeof *e->target);
	e->dec = xcalloc(n, sizeof *e->dec);
	e->bytes = xcalloc((size_t)e->bw * (size_t)e->bh, 8);
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		e->target[j] = (double)(float)(e->isint ? v / (sgn ? 1023.0 :
		    2047.0) : v);
	}
	for (by = 0; by < e->bh; by++)
		for (bx = 0; bx < e->bw; bx++) {
			double t[16];
			unsigned char care[16], *b = e->bytes + 8 * (by * e->bw + bx);
			int v[16];
			for (i = 0; i < 16; i++) {
				int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
				long at = (long)y * W + x;
				care[i] = x < W && at < (long)n;
				t[i] = care[i] ? e->target[at] : 0;
			}
			eac_encode(t, care, sgn, b);
			eac_decode(b, sgn, v);
			for (i = 0; i < 16; i++) {
				int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
				long at = (long)y * W + x;
				double d16;
				if (!care[i])
					continue;
				e->dec[at] = eac_read(v[i], sgn, 0);
				d16 = eac_read(v[i], sgn, 1);
				if (fabs(e->dec[at] - t[i]) > e->err) {
					e->err = fabs(e->dec[at] - t[i]);
					e->at = (int)at;
				}
				if (fabs(d16 - t[i]) > e->err16)
					e->err16 = fabs(d16 - t[i]);
			}
		}
	return 1;
}

/* an EAC block whose 11-bit values are the 16 targets exactly (care[i]
 * = 0: any), the first in the encoder's order; 0 when there is none */
static int
eac_exact(const int t[16], const unsigned char care[16], int sgn,
    unsigned char out[8])
{
	int b, mul, tab, i, k, v[8], sel[16];
	uint64_t w;

	for (b = sgn ? -127 : 0; b <= (sgn ? 127 : 255); b++)
		for (mul = 0; mul < 16; mul++)
			for (tab = 0; tab < 16; tab++) {
				for (k = 0; k < 8; k++)
					v[k] = eac_value(b & 255, mul, tab, k,
					    sgn);
				for (i = 0; i < 16; i++) {
					sel[i] = 0;
					if (!care[i])
						continue;
					for (k = 0; k < 8 && v[k] != t[i]; k++)
						;
					if (k == 8)
						break;
					sel[i] = k;
				}
				if (i < 16)
					continue;
				w = (uint64_t)(b & 255) << 56 |
				    (uint64_t)mul << 52 | (uint64_t)tab << 48;
				for (i = 0; i < 16; i++)
					w |= (uint64_t)sel[i] << (45 - 3 *
					    (4 * (i & 3) + (i >> 2)));
				for (i = 0; i < 8; i++)
					out[i] = (unsigned char)(w >> (56 -
					    8 * i));
				return 1;
			}
	return 0;
}

/*
 * A table of whole numbers 0..2047 (signed: -1023..1023, one of them
 * negative) as EAC blocks whose values are its entries exactly, in the
 * row widths texenc_lossless tries: the one with the fewest bytes, fewer
 * than stored; 0 when there is none.
 */
static int
eac_lossless(nid tab, int natw, int stored, int *W, int *bytes, int *sgn)
{
	uint32_t j, n = N.b[tab];
	int w, k, best = 0, *t, lo = 0, hi = 0;

	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		if (!is_const(x) || !TY_ISINT(N.ty[x]))
			return 0;
		if (j == 0 || N.ki[x] < lo)
			lo = N.ki[x] < -2048 ? -2048 : (int)N.ki[x];
		if (j == 0 || N.ki[x] > hi)
			hi = N.ki[x] > 4096 ? 4096 : (int)N.ki[x];
	}
	*sgn = lo < 0;
	if (*sgn ? lo < -1023 || hi > 1023 : hi > 2047)
		return 0;
	t = xcalloc(n, sizeof *t);
	for (j = 0; j < n; j++)
		t[j] = (int)N.ki[N.lst[N.a[tab] + j]];
	for (k = -1, w = natw; k <= 12; w = 4 << ++k) {
		int bw, bh, bx, by, ok = 1;
		if (w < 1 || w > 4096 || (k >= 0 && w > 4 && (uint32_t)w / 2 >=
		    n) || (k >= 0 && w == natw))
			continue;
		bw = (w + 3) / 4;
		bh = (int)((n + (uint32_t)w - 1) / (uint32_t)w + 3) / 4;
		if (bw * bh * 8 >= stored || (best && bw * bh * 8 >= *bytes))
			continue;
		for (by = 0; by < bh && ok; by++)
			for (bx = 0; bx < bw && ok; bx++) {
				int bt[16], i;
				unsigned char care[16], q[8];
				for (i = 0; i < 16; i++) {
					int x = bx * 4 + (i & 3);
					int y = by * 4 + (i >> 2);
					long at = (long)y * w + x;
					care[i] = x < w && at < (long)n;
					bt[i] = care[i] ? t[at] : 0;
				}
				ok = eac_exact(bt, care, *sgn, q);
			}
		if (ok) {
			best = 1;
			*W = w;
			*bytes = bw * bh * 8;
		}
	}
	free(t);
	return best;
}

/*
 * One table as R11_EAC, or two of one shape as RG11_EAC (16 bytes a
 * block, red's 8 then green's), signed when an entry is negative: each
 * block's base, multiplier, modifiers and selectors, every header tried.
 */
static void
cmd_texenc_eac(const struct func *f, const struct func *g, int W,
    double tol, int has_tol)
{
	const struct func *fs[2] = {f, g};
	const char *nm[2], *ch[2] = {"red", "green"};
	int nt = g ? 2 : 1, c, k, i, nb, sgn = 0, neg;
	struct texenc e[2];
	struct sbuf b = {0};
	char fmt[24];

	for (c = 0; c < nt; c++) {
		nm[c] = fs[c]->anon ? (c ? "the second table" : nt == 2 ?
		    "the first table" : "the table") : sym_str(fs[c]->name);
		if (fs[c]->np != 0 || N.op[fs[c]->body] != O_TABLE)
			fail("texenc takes tables [t0, t1, ...] (or their names), not "
			    "%s", fs[c]->anon ? node_str(fs[c]->body) :
			    sym_str(fs[c]->name));
		/* a negative entry anywhere makes both channels signed */
		(void)eac_holds(fs[c]->body, 1, &neg);
		if (neg)
			sgn = 1;
	}
	if (nt == 2 && N.b[f->body] != N.b[g->body])
		fail("texenc: RG11_EAC's two channels are tables of one shape; %s "
		    "has %u entries and %s %u", nm[0], N.b[f->body], nm[1],
		    N.b[g->body]);
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	for (c = 0; c < nt; c++)
		if (!texenc_eac(fs[c]->body, W, sgn, &e[c])) {
			if (c)
				texenc_free(&e[0]);
			fail("texenc: %s holds whole numbers from %s (read as k/%s) "
			    "or numbers from %s; %s has an entry that is neither",
			    sgn ? "signed EAC" : "EAC", sgn ? "-1023 to 1023" :
			    "0 to 2047", sgn ? "1023" : "2047", sgn ? "-1 to 1" :
			    "0 to 1", nm[c]);
		}
	snprintf(fmt, sizeof fmt, "%s%s_EAC", sgn ? "SIGNED_" : "", nt == 2 ?
	    "RG11" : "R11");
	nb = e[0].bw * e[0].bh;
	rep_begin("texenc");
	rep_str("format", fmt);
	rep_int("entries", e[0].n);
	rep_int("width", e[0].W);
	rep_int("height", e[0].H);
	rep_int("bytes", nb * 8 * nt);
	sb_clear(&b);
	if (nt == 2)
		sb_printf(&b, " (red %s, green %s)", nm[0], nm[1]);
	printf("texenc %s%s%s eac: %d entr%s%s as %s%s, rows of %d (%d x %d "
	    "texels): %d block%s, %d bytes (%d as %s, %d as %s)%s\n", nm[0],
	    nt == 2 ? ", " : "", nt == 2 ? nm[1] : "", e[0].n, e[0].n == 1 ?
	    "y" : "ies", nt == 2 ? " a channel" : "", fmt, sb_str(&b), e[0].W,
	    e[0].W, e[0].H, nb, nb == 1 ? "" : "s", nb * 8 * nt, 2 * nt *
	    e[0].W * e[0].H, nt == 2 ? (sgn ? "R16G16_SNORM" : "R16G16_UNORM") :
	    (sgn ? "R16_SNORM" : "R16_UNORM"), 4 * nt * e[0].W * e[0].H,
	    nt == 2 ? "R32G32_SFLOAT" : "R32_SFLOAT", e[0].isint || (nt == 2 &&
	    e[1].isint) ? (sgn ? "; whole numbers k are stored as k/1023" :
	    "; whole numbers k are stored as k/2047") : "");
	rep_list("blocks");
	for (k = 0; k < nb; k++) {
		rep_obj(NULL);
		rep_int("x", k % e[0].bw);
		rep_int("y", k / e[0].bw);
		for (c = 0; c < nt; c++) {
			const unsigned char *q = e[c].bytes + 8 * k;
			uint64_t w = 0;
			int mul, tab, base = sgn && q[0] >= 128 ? q[0] - 256 : q[0];
			for (i = 0; i < 8; i++)
				w = w << 8 | q[i];
			mul = (int)(w >> 52 & 15);
			tab = (int)(w >> 48 & 15);
			printf("  block (%d, %d)%s%s: base %d, multiplier %d, modifiers",
			    k % e[0].bw, k / e[0].bw, nt == 2 ? " " : "", nt == 2 ?
			    ch[c] : "", base, mul);
			for (i = 0; i < 8; i++)
				printf(" %d", eac_mods[tab][i]);
			printf(" (table %d), selectors", tab);
			sb_clear(&b);
			for (i = 0; i < 16; i++) {
				int s = (int)(w >> (45 - 3 * (4 * (i & 3) + (i >> 2))) & 7);
				printf(" %d", s);
				sb_printf(&b, "%s%d", i ? " " : "", s);
			}
			if (nt == 2)
				rep_obj(ch[c]);
			rep_int("base", base);
			rep_int("multiplier", mul);
			rep_int("table", tab);
			rep_str("selectors", sb_str(&b));
			if (nt == 2) {
				rep_end();
				printf("\n");
			}
		}
		sb_clear(&b);
		if (nt == 2)
			printf("  block (%d, %d) bytes", k % e[0].bw, k / e[0].bw);
		else
			printf(", bytes");
		for (c = 0; c < nt; c++)
			for (i = 0; i < 8; i++) {
				printf(" %02x", e[c].bytes[8 * k + i]);
				sb_printf(&b, "%s%02x", c || i ? " " : "",
				    e[c].bytes[8 * k + i]);
			}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	for (c = 0; c < nt; c++) {
		char pre[16];
		snprintf(pre, sizeof pre, "%s", nt == 2 ? (c ? "green: " :
		    "red: ") : "");
		if (nt == 2)
			rep_obj(ch[c]);
		sb_clear(&b);
		sb_putc(&b, '[');
		for (i = 0; i < e[c].n; i++)
			sb_printf(&b, "%s%.9g", i ? ", " : "", e[c].dec[i]);
		sb_putc(&b, ']');
		printf("  %sdecodes to %s\n", pre, sb_str(&b));
		rep_str("decoded", sb_str(&b));
		rep_bool("exact", e[c].err == 0);
		rep_num("max_error", e[c].err);
		rep_num("x16_max_error", e[c].err16);
		if (e[c].err == 0)
			printf("  %sexact: every texel is its entry's f32%s (16 texels "
			    "a block against 8 values, every base, multiplier and "
			    "table tried)\n", pre, !e[c].isint ? "" : sgn ?
			    " over 1023" : " over 2047");
		else {
			printf("  %snot exact: the closest blocks are within %.3g of "
			    "the table (entry %d: %.9g as %.9g), every base, multiplier "
			    "and table tried\n", pre, e[c].err, e[c].at,
			    e[c].target[e[c].at], e[c].dec[e[c].at]);
			if (has_tol) {
				printf("  %s%s the %g allowed (tol)\n", pre, e[c].err <= tol ?
				    "within" : "more than", tol);
				rep_bool("within_tol", e[c].err <= tol);
			}
		}
		printf("  %sa decoder that extends the 11 bits to 16 (the "
		    "specification allows it) reads %s over %s: within %.3g of "
		    "the table\n", pre, sgn ? "|v|'s (|v| << 5) + (|v| >> 5), "
		    "with v's sign," : "(v << 5) + (v >> 6)", sgn ? "32767" :
		    "65535", e[c].err16);
		if (nt == 2)
			rep_end();
	}
	if (nt == 2) {
		rep_bool("exact", e[0].err == 0 && e[1].err == 0);
		if (has_tol)
			rep_bool("within_tol", e[0].err <= tol && e[1].err <= tol);
	}
	printf("  a nearest or filtered read of the texture is that read of the "
	    "decoded table%s; no model prices an EAC read apart from its "
	    "uncompressed one: not measured\n", nt == 2 ? "s, red and green" :
	    "");
	free(b.p);
	for (c = 0; c < nt; c++)
		texenc_free(&e[c]);
}

/* the 2D footprints ASTC has */
static int
astc_footprint(int bw, int bh)
{
	static const int fp[14][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6},
		{8, 5}, {8, 6}, {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10},
		{12, 10}, {12, 12}};
	int i;

	for (i = 0; i < 14; i++)
		if (fp[i][0] == bw && fp[i][1] == bh)
			return 1;
	return 0;
}

/* a table's entries as what a read should return (k/255 for whole
 * numbers 0..255); 0 when they are not numbers ASTC holds */
static int
astc_targets(nid tab, double *t, int *isint)
{
	uint32_t j, n = N.b[tab];

	*isint = 1;
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v;
		if (!is_const(x))
			return 0;
		v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (!TY_ISINT(N.ty[x]) || v > 255)
			*isint = 0;
		if (!(v >= 0))
			return 0;
	}
	for (j = 0; j < n; j++) {
		nid x = N.lst[N.a[tab] + j];
		double v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (!*isint && v > 1)
			return 0;
		t[j] = (double)(float)(*isint ? v / 255 : v);
	}
	return 1;
}

/*
 * One table, or two in a block's two planes, as ASTC blocks of bw x bh:
 * each block searched (astcblk.c), then decoded again, and what it
 * decodes to is what is reported and judged.
 */
static void
cmd_texenc_astc(const struct func *f, const struct func *g, int bw, int bh,
    int W, double tol, int has_tol, int srgb)
{
	const struct func *fs[2] = {f, g};
	const char *nm[2];
	int nt = g ? 2 : 1, n, H, nbx, nby, c, k, i, isint[2], at[2] = {0, 0};
	double *t[2] = {NULL, NULL}, *dec[2], *d16[2], err[2] = {0, 0},
	    err16[2] = {0, 0}, *alt = NULL, erralt = 0;
	struct sbuf b = {0};
	uint16_t px[144 * 4], px2[144 * 4];

	for (c = 0; c < nt; c++) {
		nm[c] = fs[c]->anon ? (c ? "the second table" : nt == 2 ?
		    "the first table" : "the table") : sym_str(fs[c]->name);
		if (fs[c]->np != 0 || N.op[fs[c]->body] != O_TABLE)
			fail("texenc takes tables [t0, t1, ...] (or their names), not "
			    "%s", fs[c]->anon ? node_str(fs[c]->body) :
			    sym_str(fs[c]->name));
	}
	if (!astc_footprint(bw, bh))
		fail("texenc: ASTC's 2D blocks are 4x4, 5x4, 5x5, 6x5, 6x6, 8x5, "
		    "8x6, 8x8, 10x5, 10x6, 10x8, 10x10, 12x10 and 12x12, not %dx%d",
		    bw, bh);
	if (nt == 2 && N.b[f->body] != N.b[g->body])
		fail("texenc: a block's two planes are tables of one shape; %s has "
		    "%u entries and %s %u", nm[0], N.b[f->body], nm[1],
		    N.b[g->body]);
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	n = (int)N.b[f->body];
	H = (n + W - 1) / W;
	nbx = (W + bw - 1) / bw;
	nby = (H + bh - 1) / bh;
	for (c = 0; c < nt; c++) {
		t[c] = xcalloc((size_t)n, sizeof *t[c]);
		dec[c] = xcalloc((size_t)n, sizeof *dec[c]);
		d16[c] = xcalloc((size_t)n, sizeof *d16[c]);
		if (!astc_targets(fs[c]->body, t[c], &isint[c]))
			fail("texenc: ASTC holds numbers from 0 to 1 (or whole numbers "
			    "0 to 255, read as k/255); %s has an entry that is "
			    "neither", nm[c]);
		/* sRGB luminance: a whole number is the code, read through the
		 * EOTF */
		if (srgb && c == 0 && isint[0])
			for (i = 0; i < n; i++)
				t[0][i] = texenc_srgb8((int)lround(t[0][i] * 255));
	}
	rep_begin("texenc");
	sb_printf(&b, "ASTC_%dx%d_%s", bw, bh, srgb ? "SRGB" : "UNORM");
	rep_str("format", sb_str(&b));
	rep_int("entries", n);
	rep_int("width", W);
	rep_int("height", H);
	rep_int("bytes", nbx * nby * 16);
	if (nt == 2)
		printf("texenc %s, %s astc %dx%d%s: %d entr%s a table in ASTC %dx%d%s "
		    "blocks' two planes (%s the luminance, %s the alpha), rows "
		    "of %d (%d x %d texels): %d block%s, %d bytes (%d as "
		    "R8G8_UNORM, %d as R32G32_SFLOAT)%s\n", nm[0], nm[1], bw, bh,
		    srgb ? " srgb" : "", n, n == 1 ? "y" : "ies", bw, bh, srgb ?
		    " sRGB" : "", nm[0], nm[1], W, W, H, nbx * nby, nbx * nby == 1 ?
		    "" : "s", nbx * nby * 16, 2 * W * H, 8 * W * H, srgb ? "; the "
		    "luminance reads the sRGB EOTF of its code over 255, so a "
		    "whole number k in it is the code k" : "");
	else
		printf("texenc %s astc %dx%d%s: %d entr%s as ASTC %dx%d%s, rows of "
		    "%d (%d x %d texels): %d block%s, %d bytes (%d as R8_UNORM, %d "
		    "as R32_SFLOAT)%s\n", nm[0], bw, bh, srgb ? " srgb" : "", n,
		    n == 1 ? "y" : "ies", bw, bh, srgb ? " sRGB" : "", W, W, H,
		    nbx * nby, nbx * nby == 1 ? "" : "s", nbx * nby * 16, W * H,
		    4 * W * H, srgb ? "; the luminance reads the sRGB EOTF of its "
		    "code over 255, so a whole number k in it is the code k" :
		    isint[0] ? "; whole numbers k are stored as k/255" : "");
	rep_list("blocks");
	for (k = 0; k < nbx * nby; k++) {
		double bt[2][144];
		uint8_t care[144], q[16];
		struct astcb_info in;
		double got = 0;
		int bx = k % nbx, by = k / nbx;
		for (i = 0; i < bw * bh; i++) {
			int x = bx * bw + i % bw, y = by * bh + i / bw;
			long a = (long)y * W + x;
			care[i] = x < W && a < n;
			for (c = 0; c < nt; c++)
				bt[c][i] = care[i] ? t[c][a] : 0;
		}
		if (!astcb_encode(bt[0], nt == 2 ? bt[1] : NULL, care, bw, bh, srgb,
		    q, &in))
			fail("texenc: no ASTC %dx%d block layout holds block (%d, %d) "
			    "of %s", bw, bh, bx, by, nm[0]);
		if (astcb_decode(q, bw, bh, srgb, px))
			fail("texenc: block (%d, %d) as written decodes to the error "
			    "colour (a block the LDR profile refuses): an encoder "
			    "error", bx, by);
		if (srgb && nt == 2) {
			/* alpha as astcenc and Mesa widen it */
			if (!alt)
				alt = xcalloc((size_t)n, sizeof *alt);
			(void)astcb_decode(q, bw, bh, 2, px2);
			for (i = 0; i < bw * bh; i++) {
				int x = bx * bw + i % bw, y = by * bh + i / bw;
				long a = (long)y * W + x;
				if (!care[i])
					continue;
				alt[a] = (double)(float)((px2[4 * i + 3] >> 8) / 255.0);
				if (fabs(alt[a] - t[1][a]) > erralt)
					erralt = fabs(alt[a] - t[1][a]);
			}
		}
		for (i = 0; i < bw * bh; i++) {
			int x = bx * bw + i % bw, y = by * bh + i / bw;
			long a = (long)y * W + x;
			if (!care[i])
				continue;
			for (c = 0; c < nt; c++) {
				/* the first table in R, the second in A */
				int C = px[4 * i + (c ? 3 : 0)];
				double e;
				dec[c][a] = srgb && !c ? texenc_srgb8(C >> 8) :
				    (double)(float)((C >> 8) / 255.0);
				d16[c][a] = astcb_float16(C);
				e = fabs(dec[c][a] - t[c][a]);
				if (e > got)
					got = e;
				if (e > err[c]) {
					err[c] = e;
					at[c] = (int)a;
				}
				if (fabs(d16[c][a] - t[c][a]) > err16[c])
					err16[c] = fabs(d16[c][a] - t[c][a]);
			}
		}
		if (got != in.err)
			fail("texenc: block (%d, %d) decodes within %.9g of its entries "
			    "where the search computed %.9g: an encoder error", bx, by,
			    got, in.err);
		printf("  block (%d, %d): %dx%d weights of %d levels in %s, ", bx, by,
		    in.W, in.H, in.wlevels, in.planes == 2 ? "two planes" :
		    "one plane");
		if (in.parts == 1)
			printf("1 partition");
		else if (in.miss == 0)
			printf("%d partitions (seed %d: its partitions are the "
			    "entries' pattern)", in.parts, in.seed);
		else
			printf("%d partitions (seed %d: the closest to the entries' "
			    "pattern, %d texel%s off it)", in.parts, in.seed, in.miss,
			    in.miss == 1 ? "" : "s");
		printf(", endpoints");
		sb_clear(&b);
		for (i = 0; i < in.nends; i++) {
			printf("%s %d", i ? "," : "", in.ends[i]);
			sb_printf(&b, "%s%d", i ? " " : "", in.ends[i]);
		}
		printf(" (%s, %d levels), bytes", nt == 2 ? "luminance, luminance, "
		    "alpha, alpha" : "luminance", in.clevels);
		rep_obj(NULL);
		rep_int("x", bx);
		rep_int("y", by);
		rep_int("grid_width", in.W);
		rep_int("grid_height", in.H);
		rep_int("weight_levels", in.wlevels);
		rep_int("colour_levels", in.clevels);
		rep_int("planes", in.planes);
		rep_int("partitions", in.parts);
		if (in.parts > 1) {
			rep_int("seed", in.seed);
			rep_int("pattern_misses", in.miss);
		}
		rep_str("endpoints", sb_str(&b));
		sb_clear(&b);
		for (i = 0; i < 16; i++) {
			printf(" %02x", q[i]);
			sb_printf(&b, "%s%02x", i ? " " : "", q[i]);
		}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	for (c = 0; c < nt; c++) {
		const char *pre = nt == 2 ? (c ? "alpha: " : "luminance: ") : "";
		rep_obj(nt == 2 ? (c ? "alpha" : "luminance") : "table");
		sb_clear(&b);
		sb_putc(&b, '[');
		for (i = 0; i < n; i++)
			sb_printf(&b, "%s%.9g", i ? ", " : "", dec[c][i]);
		sb_putc(&b, ']');
		printf("  %sdecodes to %s\n", pre, sb_str(&b));
		rep_str("decoded", sb_str(&b));
		rep_bool("exact", err[c] == 0);
		rep_num("max_error", err[c]);
		rep_num("float16_max_error", err16[c]);
		if (err[c] == 0)
			printf("  %sexact: every texel is its entry's f32%s\n", pre,
			    !isint[c] ? "" : srgb && !c ? ", the EOTF of its code" :
			    " over 255");
		else {
			printf("  %snot exact: the closest block found is within %.3g of "
			    "the table (entry %d: %.9g as %.9g)\n", pre, err[c], at[c],
			    t[c][at[c]], dec[c][at[c]]);
			if (has_tol) {
				printf("  %s%s the %g allowed (tol)\n", pre, err[c] <= tol ?
				    "within" : "more than", tol);
				rep_bool("within_tol", err[c] <= tol);
			}
		}
		/* sRGB has no decode mode: 8 bits, then the curve */
		if (!srgb)
			printf("  %sin the default float16 decode mode a read is "
			    "C/65536 toward zero (1 for 65535): within %.3g of the "
			    "table\n", pre, err16[c]);
		else if (c == 1) {
			sb_clear(&b);
			sb_putc(&b, '[');
			for (i = 0; i < n; i++)
				sb_printf(&b, "%s%.9g", i ? ", " : "", alt[i]);
			sb_putc(&b, ']');
			printf("  alpha as astcenc 5.7.1 and Mesa decode it (its bytes "
			    "widened with 0x80 too, where the specification repeats "
			    "them): %s, within %.3g of the table\n", sb_str(&b),
			    erralt);
			rep_str("decoded_x80", sb_str(&b));
			rep_num("max_error_x80", erralt);
		}
		rep_end();
	}
	printf("  searched: every block mode the footprint has (a weight grid up "
	    "to the block, each weight range, %s), %sthe endpoints near each "
	    "partition's least and greatest entry, the weights by least "
	    "squares through the infill and then a level at a time; not every "
	    "encoding, so a closer block may exist where it is not exact.  "
	    "Every block written is decoded again\n", nt == 2 ? "two planes" :
	    "one plane", nt == 2 ? "" : "1 to 4 partitions with the seeds "
	    "whose partitions are closest to the entries' pattern, ");
	printf("  a nearest or filtered read of the texture is that read of the "
	    "decoded table; no model prices an ASTC read: not measured\n");
	free(b.p);
	for (c = 0; c < nt; c++) {
		free(t[c]);
		free(dec[c]);
		free(d16[c]);
	}
	free(alt);
}

/* an RGB8_ETC2 block's red channel as words, printed and reported */
static void
pr_etc2(const uint8_t *q, const struct etc2_info *in, struct sbuf *b)
{
	static const char *const mname[] = {"individual", "differential", "T",
	    "H", "planar"};
	uint64_t w = 0;
	int i;

	for (i = 0; i < 8; i++)
		w = w << 8 | q[i];
	rep_str("mode", mname[in->mode]);
	switch (in->mode) {
	case ETC2_INDIVIDUAL: case ETC2_DIFFERENTIAL:
		printf("%s, sub-blocks %s: red %d with modifiers %d %d %d %d, red "
		    "%d with %d %d %d %d", mname[in->mode], in->flip ? "one over "
		    "the other" : "side by side", in->v[0], etc1_mod(in->v[2], 0),
		    etc1_mod(in->v[2], 1), etc1_mod(in->v[2], 2),
		    etc1_mod(in->v[2], 3), in->v[1], etc1_mod(in->v[3], 0),
		    etc1_mod(in->v[3], 1), etc1_mod(in->v[3], 2),
		    etc1_mod(in->v[3], 3));
		rep_bool("flip", in->flip);
		rep_int("red0", in->v[0]);
		rep_int("red1", in->v[1]);
		rep_int("table0", in->v[2]);
		rep_int("table1", in->v[3]);
		break;
	case ETC2_T:
		printf("T mode: red %d, then %d + %d, %d, %d - %d", in->v[0],
		    in->v[1], in->v[2], in->v[1], in->v[1], in->v[2]);
		rep_int("red0", in->v[0]);
		rep_int("red1", in->v[1]);
		rep_int("distance", in->v[2]);
		break;
	case ETC2_H:
		printf("H mode: red %d and %d, each + and - %d", in->v[0], in->v[1],
		    in->v[2]);
		rep_int("red0", in->v[0]);
		rep_int("red1", in->v[1]);
		rep_int("distance", in->v[2]);
		break;
	default:
		printf("planar: red %d at (0, 0), %d at (4, 0), %d at (0, 4)",
		    in->v[0], in->v[1], in->v[2]);
		rep_int("red_o", in->v[0]);
		rep_int("red_h", in->v[1]);
		rep_int("red_v", in->v[2]);
		return;
	}
	printf(", selectors");
	sb_clear(b);
	for (i = 0; i < 16; i++) {
		int k = (i >> 2) + 4 * (i & 3);
		int s = (int)((w >> (16 + k) & 1) << 1 | (w >> k & 1));
		printf(" %d", s);
		sb_printf(b, "%s%d", i ? " " : "", s);
	}
	rep_str("selectors", sb_str(b));
}

/*
 * One table in the red channel of RGB8_ETC2 blocks, or two of one shape
 * as RGBA8_ETC2_EAC (16 bytes a block: the second table's 8-bit alpha
 * block, then the first's colour block): every encoding of the red
 * channel tried, every alpha header.
 */
static void
cmd_texenc_etc2(const struct func *f, const struct func *g, int W,
    double tol, int has_tol, int srgb)
{
	const struct func *fs[2] = {f, g};
	const char *nm[2], *ch[2] = {"red", "alpha"}, *fmt;
	int nt = g ? 2 : 1, n, H, nbx, nby, c, k, i, isint[2], at[2] = {0, 0};
	double *t[2] = {NULL, NULL}, *dec[2] = {NULL, NULL}, err[2] = {0, 0};
	struct sbuf b = {0};

	for (c = 0; c < nt; c++) {
		nm[c] = fs[c]->anon ? (c ? "the second table" : nt == 2 ?
		    "the first table" : "the table") : sym_str(fs[c]->name);
		if (fs[c]->np != 0 || N.op[fs[c]->body] != O_TABLE)
			fail("texenc takes tables [t0, t1, ...] (or their names), not "
			    "%s", fs[c]->anon ? node_str(fs[c]->body) :
			    sym_str(fs[c]->name));
	}
	if (nt == 2 && N.b[f->body] != N.b[g->body])
		fail("texenc: RGBA8_ETC2_EAC's red and alpha are tables of one "
		    "shape; %s has %u entries and %s %u", nm[0], N.b[f->body], nm[1],
		    N.b[g->body]);
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	n = (int)N.b[f->body];
	H = (n + W - 1) / W;
	nbx = (W + 3) / 4;
	nby = (H + 3) / 4;
	for (c = 0; c < nt; c++) {
		t[c] = xcalloc((size_t)n, sizeof *t[c]);
		dec[c] = xcalloc((size_t)n, sizeof *dec[c]);
		if (!astc_targets(fs[c]->body, t[c], &isint[c]))
			fail("texenc: ETC2 holds numbers from 0 to 1 (or whole numbers "
			    "0 to 255, read as k/255); %s has an entry that is "
			    "neither", nm[c]);
		/* sRGB red: a whole number is the code, read through the EOTF */
		if (srgb && c == 0 && isint[0])
			for (i = 0; i < n; i++)
				t[0][i] = texenc_srgb8((int)lround(t[0][i] * 255));
	}
	fmt = nt == 2 ? (srgb ? "SRGB8_ALPHA8_ETC2_EAC" : "RGBA8_ETC2_EAC") :
	    srgb ? "SRGB8_ETC2" : "RGB8_ETC2";
	rep_begin("texenc");
	rep_str("format", fmt);
	rep_int("entries", n);
	rep_int("width", W);
	rep_int("height", H);
	rep_int("bytes", nbx * nby * 8 * nt);
	sb_clear(&b);
	if (nt == 2)
		sb_printf(&b, " (%s the red, %s the alpha)", nm[0], nm[1]);
	printf("texenc %s%s%s etc2%s: %d entr%s%s as %s%s, rows of %d (%d x %d "
	    "texels): %d block%s, %d bytes (%d as %s, %d as %s)%s%s\n", nm[0],
	    nt == 2 ? ", " : "", nt == 2 ? nm[1] : "", srgb ? " srgb" : "", n,
	    n == 1 ? "y" : "ies", nt == 2 ? " a channel" : "", fmt, sb_str(&b),
	    W, W, H, nbx * nby, nbx * nby == 1 ? "" : "s", nbx * nby * 8 * nt,
	    nt * W * H, nt == 2 ? "R8G8_UNORM" : "R8_UNORM", 4 * nt * W * H,
	    nt == 2 ? "R32G32_SFLOAT" : "R32_SFLOAT", srgb ? "; red reads the "
	    "sRGB EOTF of its code over 255, so a whole number k in it is the "
	    "code k" : isint[0] || (nt == 2 && isint[1]) ? "; whole numbers k "
	    "are stored as k/255" : "", srgb && nt == 2 && isint[1] ? "; alpha "
	    "stays linear, k/255" : "");
	rep_list("blocks");
	for (k = 0; k < nbx * nby; k++) {
		double bt[2][16], got[2] = {0, 0}, want[2];
		uint8_t care[16], q[16], px[16][3];
		struct etc2_info in;
		int bx = k % nbx, by = k / nbx, av[16];
		for (i = 0; i < 16; i++) {
			int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
			long a = (long)y * W + x;
			care[i] = x < W && a < n;
			for (c = 0; c < nt; c++)
				bt[c][i] = care[i] ? t[c][a] : 0;
		}
		/* RGBA8_ETC2_EAC: the alpha block first, then the colour one */
		want[0] = etc2_encode_r(bt[0], care, srgb, q + (nt == 2 ? 8 : 0),
		    &in);
		etc2_decode(q + (nt == 2 ? 8 : 0), px);
		if (nt == 2) {
			want[1] = etc2_alpha_encode(bt[1], care, q);
			etc2_alpha_decode(q, av);
		}
		for (i = 0; i < 16; i++) {
			int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
			long a = (long)y * W + x;
			if (!care[i])
				continue;
			for (c = 0; c < nt; c++) {
				double e;
				dec[c][a] = c ? (double)(float)(av[i] / 255.0) : srgb ?
				    texenc_srgb8(px[i][0]) : (double)(float)(px[i][0] /
				    255.0);
				e = fabs(dec[c][a] - t[c][a]);
				if (e > got[c])
					got[c] = e;
				if (e > err[c]) {
					err[c] = e;
					at[c] = (int)a;
				}
			}
		}
		for (c = 0; c < nt; c++)
			if (got[c] != want[c])
				fail("texenc: block (%d, %d) %s decodes within %.9g of its "
				    "entries where the search computed %.9g: an encoder "
				    "error", bx, by, ch[c], got[c], want[c]);
		rep_obj(NULL);
		rep_int("x", bx);
		rep_int("y", by);
		printf("  block (%d, %d)%s: ", bx, by, nt == 2 ? " red" : "");
		if (nt == 2)
			rep_obj("red");
		pr_etc2(q + (nt == 2 ? 8 : 0), &in, &b);
		if (nt == 2) {
			uint64_t w = 0;
			int mul, tab;
			rep_end();
			for (i = 0; i < 8; i++)
				w = w << 8 | q[i];
			mul = (int)(w >> 52 & 15);
			tab = (int)(w >> 48 & 15);
			printf("\n  block (%d, %d) alpha: base %d, multiplier %d, "
			    "modifiers", bx, by, q[0], mul);
			for (i = 0; i < 8; i++)
				printf(" %d", eac_mod(tab, i));
			printf(" (table %d), selectors", tab);
			sb_clear(&b);
			for (i = 0; i < 16; i++) {
				int s = (int)(w >> (45 - 3 * (4 * (i & 3) + (i >> 2))) &
				    7);
				printf(" %d", s);
				sb_printf(&b, "%s%d", i ? " " : "", s);
			}
			rep_obj("alpha");
			rep_int("base", q[0]);
			rep_int("multiplier", mul);
			rep_int("table", tab);
			rep_str("selectors", sb_str(&b));
			rep_end();
			printf("\n  block (%d, %d) bytes", bx, by);
		} else
			printf(", bytes");
		sb_clear(&b);
		for (i = 0; i < 8 * nt; i++) {
			printf(" %02x", q[i]);
			sb_printf(&b, "%s%02x", i ? " " : "", q[i]);
		}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	for (c = 0; c < nt; c++) {
		const char *pre = nt == 2 ? (c ? "alpha: " : "red: ") : "";
		if (nt == 2)
			rep_obj(ch[c]);
		sb_clear(&b);
		sb_putc(&b, '[');
		for (i = 0; i < n; i++)
			sb_printf(&b, "%s%.9g", i ? ", " : "", dec[c][i]);
		sb_putc(&b, ']');
		printf("  %sdecodes to %s\n", pre, sb_str(&b));
		rep_str("decoded", sb_str(&b));
		rep_bool("exact", err[c] == 0);
		rep_num("max_error", err[c]);
		if (err[c] == 0)
			printf("  %sexact: every texel is its entry's f32%s (%s)\n",
			    pre, !isint[c] ? "" : srgb && !c ? ", the EOTF of its "
			    "code" : " over 255", c ? "every base, multiplier and table "
			    "tried" : "every encoding of the red channel tried");
		else {
			printf("  %snot exact: the closest blocks are within %.3g of "
			    "the table (entry %d: %.9g as %.9g), %s\n", pre, err[c],
			    at[c], t[c][at[c]], dec[c][at[c]], c ? "every base, "
			    "multiplier and table tried" : "every encoding of the red "
			    "channel tried");
			if (has_tol) {
				printf("  %s%s the %g allowed (tol)\n", pre, err[c] <= tol ?
				    "within" : "more than", tol);
				rep_bool("within_tol", err[c] <= tol);
			}
		}
		if (nt == 2)
			rep_end();
	}
	if (nt == 2) {
		rep_bool("exact", err[0] == 0 && err[1] == 0);
		if (has_tol)
			rep_bool("within_tol", err[0] <= tol && err[1] <= tol);
	}
	if (srgb)
		printf("  the EOTF here is the specification's formula rounded to "
		    "an f32 once; an implementation's conversion may differ in "
		    "its last bits%s\n", nt == 2 ? " (alpha is not converted)" :
		    "");
	printf("  a nearest or filtered read of the texture is that read of the "
	    "decoded table%s (green and blue are whatever the block's mode "
	    "needs); no model prices an ETC2 read apart from its uncompressed "
	    "one: not measured\n", nt == 2 ? "s, red and alpha" : " in red");
	free(b.p);
	for (c = 0; c < nt; c++) {
		free(t[c]);
		free(dec[c]);
	}
}

/*
 * One table in the red channel of BC7 blocks, or two of one shape in red
 * and alpha (modes 4 and 5, where alpha has indices of its own): every
 * mode, partition and way of holding red, each subset's endpoints
 * searched over every value its field has.
 */
static void
cmd_texenc_bc7(const struct func *f, const struct func *g, int W,
    double tol, int has_tol, int srgb)
{
	const struct func *fs[2] = {f, g};
	const char *nm[2], *ch[2] = {"red", "alpha"};
	int nt = g ? 2 : 1, n, H, nbx, nby, c, k, i, s, isint[2], at[2] = {0, 0};
	double *t[2] = {NULL, NULL}, *dec[2] = {NULL, NULL}, err[2] = {0, 0};
	struct sbuf b = {0};

	for (c = 0; c < nt; c++) {
		nm[c] = fs[c]->anon ? (c ? "the second table" : nt == 2 ?
		    "the first table" : "the table") : sym_str(fs[c]->name);
		if (fs[c]->np != 0 || N.op[fs[c]->body] != O_TABLE)
			fail("texenc takes tables [t0, t1, ...] (or their names), not "
			    "%s", fs[c]->anon ? node_str(fs[c]->body) :
			    sym_str(fs[c]->name));
	}
	if (nt == 2 && N.b[f->body] != N.b[g->body])
		fail("texenc: BC7's red and alpha are tables of one shape; %s has "
		    "%u entries and %s %u", nm[0], N.b[f->body], nm[1],
		    N.b[g->body]);
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	n = (int)N.b[f->body];
	H = (n + W - 1) / W;
	nbx = (W + 3) / 4;
	nby = (H + 3) / 4;
	for (c = 0; c < nt; c++) {
		t[c] = xcalloc((size_t)n, sizeof *t[c]);
		dec[c] = xcalloc((size_t)n, sizeof *dec[c]);
		if (!astc_targets(fs[c]->body, t[c], &isint[c]))
			fail("texenc: BC7 holds numbers from 0 to 1 (or whole numbers 0 "
			    "to 255, read as k/255); %s has an entry that is neither",
			    nm[c]);
		/* sRGB red: a whole number is the code, read through the EOTF */
		if (srgb && c == 0 && isint[0])
			for (i = 0; i < n; i++)
				t[0][i] = texenc_srgb8((int)lround(t[0][i] * 255));
	}
	rep_begin("texenc");
	rep_str("format", srgb ? "BC7_SRGB" : "BC7_UNORM");
	rep_int("entries", n);
	rep_int("width", W);
	rep_int("height", H);
	rep_int("bytes", nbx * nby * 16);
	sb_clear(&b);
	if (nt == 2)
		sb_printf(&b, " (%s the red, %s the alpha)", nm[0], nm[1]);
	printf("texenc %s%s%s bc7%s: %d entr%s%s as %s%s, rows of %d (%d x %d "
	    "texels): %d block%s, %d bytes (%d as %s, %d as %s)%s%s\n", nm[0],
	    nt == 2 ? ", " : "", nt == 2 ? nm[1] : "", srgb ? " srgb" : "", n,
	    n == 1 ? "y" : "ies", nt == 2 ? " a channel" : "", srgb ?
	    "BC7_SRGB" : "BC7_UNORM", sb_str(&b), W, W, H, nbx * nby,
	    nbx * nby == 1 ? "" : "s", nbx * nby * 16, nt * W * H, nt == 2 ?
	    "R8G8_UNORM" : "R8_UNORM", 4 * nt * W * H, nt == 2 ?
	    "R32G32_SFLOAT" : "R32_SFLOAT", srgb ? "; red reads the sRGB EOTF "
	    "of its code over 255, so a whole number k in it is the code k" :
	    isint[0] || (nt == 2 && isint[1]) ? "; whole numbers k are stored "
	    "as k/255" : "", srgb && nt == 2 && isint[1] ? "; alpha stays "
	    "linear, k/255" : "");
	rep_list("blocks");
	for (k = 0; k < nbx * nby; k++) {
		double bt[2][16], got[2] = {0, 0}, want[2];
		uint8_t care[16], q[16], px[16][4];
		struct bc7_info in;
		int bx = k % nbx, by = k / nbx;
		for (i = 0; i < 16; i++) {
			int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
			long a = (long)y * W + x;
			care[i] = x < W && a < n;
			for (c = 0; c < nt; c++)
				bt[c][i] = care[i] ? t[c][a] : 0;
		}
		want[0] = bc7_encode(bt[0], nt == 2 ? bt[1] : NULL, care, srgb, q,
		    &in, &want[1]);
		bc7_decode(q, px);
		for (i = 0; i < 16; i++) {
			int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
			long a = (long)y * W + x;
			if (!care[i])
				continue;
			for (c = 0; c < nt; c++) {
				double e;
				dec[c][a] = !c && srgb ? texenc_srgb8(px[i][0]) :
				    (double)(float)(px[i][c ? 3 : 0] / 255.0);
				e = fabs(dec[c][a] - t[c][a]);
				if (e > got[c])
					got[c] = e;
				if (e > err[c]) {
					err[c] = e;
					at[c] = (int)a;
				}
			}
		}
		/* the search's largest error is of both channels together */
		if (nt == 2 ? (got[0] > want[0] || got[1] != want[1] ||
		    fmax(got[0], got[1]) != want[0]) : got[0] != want[0])
			fail("texenc: block (%d, %d) decodes within %.9g of its entries "
			    "where the search computed %.9g: an encoder error", bx, by,
			    fmax(got[0], got[1]), want[0]);
		rep_obj(NULL);
		rep_int("x", bx);
		rep_int("y", by);
		rep_int("mode", in.mode);
		printf("  block (%d, %d): mode %d", bx, by, in.mode);
		if (in.ns > 1) {
			printf(", partition %d", in.part);
			rep_int("partition", in.part);
		}
		if (in.mode == 4 || in.mode == 5) {
			printf(", %s", in.rot ? "red the rotated alpha" : "red the "
			    "colour's");
			rep_int("rotation", in.rot);
			if (in.mode == 4) {
				printf(" (index selection %d)", in.isb);
				rep_int("index_selection", in.isb);
			}
		}
		sb_clear(&b);
		printf(", red");
		for (s = 0; s < in.ns; s++) {
			printf("%s %d to %d", s ? "," : "", in.red[s][0], in.red[s][1]);
			sb_printf(&b, "%s%d %d", s ? " " : "", in.red[s][0],
			    in.red[s][1]);
		}
		rep_str("red_endpoints", sb_str(&b));
		printf(" (%d weights)", 1 << in.ribits);
		if (nt == 2) {
			printf(", alpha %d to %d", in.alpha[0], in.alpha[1]);
			sb_clear(&b);
			sb_printf(&b, "%d %d", in.alpha[0], in.alpha[1]);
			rep_str("alpha_endpoints", sb_str(&b));
		}
		printf(", selectors");
		sb_clear(&b);
		for (i = 0; i < 16; i++) {
			printf(" %d", in.ridx[i]);
			sb_printf(&b, "%s%d", i ? " " : "", in.ridx[i]);
		}
		rep_str("selectors", sb_str(&b));
		if (nt == 2) {
			printf(", alpha's");
			sb_clear(&b);
			for (i = 0; i < 16; i++) {
				printf(" %d", in.aidx[i]);
				sb_printf(&b, "%s%d", i ? " " : "", in.aidx[i]);
			}
			rep_str("alpha_selectors", sb_str(&b));
		}
		printf(", bytes");
		sb_clear(&b);
		for (i = 0; i < 16; i++) {
			printf(" %02x", q[i]);
			sb_printf(&b, "%s%02x", i ? " " : "", q[i]);
		}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	for (c = 0; c < nt; c++) {
		const char *pre = nt == 2 ? (c ? "alpha: " : "red: ") : "";
		if (nt == 2)
			rep_obj(ch[c]);
		sb_clear(&b);
		sb_putc(&b, '[');
		for (i = 0; i < n; i++)
			sb_printf(&b, "%s%.9g", i ? ", " : "", dec[c][i]);
		sb_putc(&b, ']');
		printf("  %sdecodes to %s\n", pre, sb_str(&b));
		rep_str("decoded", sb_str(&b));
		rep_bool("exact", err[c] == 0);
		rep_num("max_error", err[c]);
		if (err[c] == 0)
			printf("  %sexact: every texel is its entry's f32%s\n", pre,
			    !isint[c] ? "" : srgb && !c ? ", the EOTF of its code" :
			    " over 255");
		else {
			printf("  %snot exact: the closest blocks are within %.3g of "
			    "the table (entry %d: %.9g as %.9g)\n", pre, err[c], at[c],
			    t[c][at[c]], dec[c][at[c]]);
			if (has_tol) {
				printf("  %s%s the %g allowed (tol)\n", pre, err[c] <= tol ?
				    "within" : "more than", tol);
				rep_bool("within_tol", err[c] <= tol);
			}
		}
		if (nt == 2)
			rep_end();
	}
	if (nt == 2) {
		rep_bool("exact", err[0] == 0 && err[1] == 0);
		if (has_tol)
			rep_bool("within_tol", err[0] <= tol && err[1] <= tol);
	}
	printf("  searched: %s, each subset's two endpoints over every value "
	    "its field has%s; green and blue are 0\n", nt == 2 ? "modes 4 and 5 "
	    "(the two with alpha's own indices), both rotations and index "
	    "selections" : "every mode, partition, and way of holding red (in "
	    "modes 4 and 5 the colour's or the rotated alpha, either index "
	    "set)", nt == 2 ? ", red and alpha each on its own" : "");
	if (srgb)
		printf("  the EOTF here is the specification's formula rounded to "
		    "an f32 once; an implementation's conversion may differ in "
		    "its last bits%s\n", nt == 2 ? " (alpha is not converted)" :
		    "");
	printf("  a BC7 block is 16 bytes for 16 texels: %s; a nearest or "
	    "filtered read is that read of the decoded table%s; no model prices "
	    "a BC7 read apart from its uncompressed one: not measured\n",
	    nt == 2 ? "half of R8G8_UNORM, as BC5 is" : "as many as R8_UNORM, "
	    "so one table is no smaller than R8 (BC4, EAC and ETC2 are half)",
	    nt == 2 ? "s, red and alpha" : " in red");
	free(b.p);
	for (c = 0; c < nt; c++) {
		free(t[c]);
		free(dec[c]);
	}
}

/*
 * One table of numbers in the red channel of BC6H blocks, UFLOAT (every
 * entry >= 0) or SFLOAT: a texel reads a half float (bc6hblk.c).  Not
 * every encoding is tried; a closer block may exist where it is not
 * exact.
 */
static void
cmd_texenc_bc6h(const struct func *f, int W, double tol, int has_tol)
{
	const char *nm = f->anon ? "the table" : sym_str(f->name);
	int n, H, nbx, nby, k, i, sgn = 0, at = 0;
	double *t, *dec, err = 0;
	struct sbuf b = {0};

	if (f->np != 0 || N.op[f->body] != O_TABLE)
		fail("texenc takes a table [t0, t1, ...] (or its name), not %s",
		    f->anon ? node_str(f->body) : sym_str(f->name));
	if (W < 1 || W > 4096)
		fail("texenc: rows W, the texels in a row, is 1 to 4096, not %d", W);
	n = (int)N.b[f->body];
	H = (n + W - 1) / W;
	nbx = (W + 3) / 4;
	nby = (H + 3) / 4;
	t = xcalloc((size_t)n, sizeof *t);
	dec = xcalloc((size_t)n, sizeof *dec);
	for (i = 0; i < n; i++) {
		nid x = N.lst[N.a[f->body] + i];
		double v;
		if (!is_const(x))
			fail("texenc: %s has an entry that is not a number: %s", nm,
			    node_str(x));
		v = TY_FP(N.ty[x]) ? N.kr[x] : (double)N.ki[x];
		if (!(fabs(v) <= 65504))
			fail("texenc: BC6H holds half floats, -65504 to 65504; %s has "
			    "%g (entry %d)", nm, v, i);
		t[i] = (double)(float)v;
		if (v < 0)
			sgn = 1;
	}
	rep_begin("texenc");
	rep_str("format", sgn ? "BC6H_SFLOAT" : "BC6H_UFLOAT");
	rep_int("entries", n);
	rep_int("width", W);
	rep_int("height", H);
	rep_int("bytes", nbx * nby * 16);
	printf("texenc %s bc6h: %d entr%s as %s, rows of %d (%d x %d texels): "
	    "%d block%s, %d bytes (%d as R16_SFLOAT, %d as R32_SFLOAT)\n", nm, n,
	    n == 1 ? "y" : "ies", sgn ? "BC6H_SFLOAT" : "BC6H_UFLOAT", W, W, H,
	    nbx * nby, nbx * nby == 1 ? "" : "s", nbx * nby * 16, 2 * W * H,
	    4 * W * H);
	rep_list("blocks");
	for (k = 0; k < nbx * nby; k++) {
		double bt[16], got = 0, want;
		uint8_t care[16], q[16];
		uint16_t px[16][3];
		struct bc6_info in;
		int bx = k % nbx, by = k / nbx;
		for (i = 0; i < 16; i++) {
			int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
			long a = (long)y * W + x;
			care[i] = x < W && a < n;
			bt[i] = care[i] ? t[a] : 0;
		}
		want = bc6h_encode_r(bt, care, sgn, q, &in);
		bc6h_decode(q, sgn, px);
		for (i = 0; i < 16; i++) {
			int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
			long a = (long)y * W + x;
			double e;
			if (!care[i])
				continue;
			dec[a] = bc6h_half(px[i][0]);
			e = fabs(dec[a] - t[a]);
			if (e > got)
				got = e;
			if (e > err) {
				err = e;
				at = (int)a;
			}
		}
		if (got != want)
			fail("texenc: block (%d, %d) decodes within %.9g of its entries "
			    "where the search computed %.9g: an encoder error", bx, by,
			    got, want);
		rep_obj(NULL);
		rep_int("x", bx);
		rep_int("y", by);
		rep_int("mode", in.mode);
		printf("  block (%d, %d): mode %d", bx, by, in.mode);
		if (in.ns == 2) {
			printf(", partition %d, red %d to %d, %d to %d", in.part,
			    in.x[0], in.x[1], in.x[2], in.x[3]);
			rep_int("partition", in.part);
		} else
			printf(", red %d to %d", in.x[0], in.x[1]);
		sb_clear(&b);
		for (i = 0; i < 2 * in.ns; i++)
			sb_printf(&b, "%s%d", i ? " " : "", in.x[i]);
		rep_str("red_endpoints", sb_str(&b));
		printf(" (quantized), selectors");
		sb_clear(&b);
		for (i = 0; i < 16; i++) {
			printf(" %d", in.idx[i]);
			sb_printf(&b, "%s%d", i ? " " : "", in.idx[i]);
		}
		rep_str("selectors", sb_str(&b));
		printf(", bytes");
		sb_clear(&b);
		for (i = 0; i < 16; i++) {
			printf(" %02x", q[i]);
			sb_printf(&b, "%s%02x", i ? " " : "", q[i]);
		}
		printf("\n");
		rep_str("bytes", sb_str(&b));
		rep_end();
	}
	rep_end();
	sb_clear(&b);
	sb_putc(&b, '[');
	for (i = 0; i < n; i++)
		sb_printf(&b, "%s%.9g", i ? ", " : "", dec[i]);
	sb_putc(&b, ']');
	printf("  decodes to %s\n", sb_str(&b));
	rep_str("decoded", sb_str(&b));
	rep_bool("exact", err == 0);
	rep_num("max_error", err);
	if (err == 0)
		printf("  exact: every texel is its entry's f32\n");
	else {
		printf("  not exact: the closest block found is within %.3g of the "
		    "table (entry %d: %.9g as %.9g)\n", err, at, t[at], dec[at]);
		if (has_tol) {
			printf("  %s the %g allowed (tol)\n", err <= tol ? "within" :
			    "more than", tol);
			rep_bool("within_tol", err <= tol);
		}
	}
	printf("  searched: every mode and partition, each subset's endpoints "
	    "among the %d quantized values each side of those nearest its least "
	    "and greatest entry, under the mode's deltas; not every encoding, so "
	    "a closer block may exist where it is not exact; green and blue are "
	    "0\n", 6);
	printf("  a nearest or filtered read of the texture is that read of the "
	    "decoded table in red; no model prices a BC6H read apart from its "
	    "uncompressed one: not measured\n");
	free(b.p);
	free(t);
	free(dec);
}

/* a table: a defined one by its name (before a comma too), or an
 * expression */
static struct func *
texenc_table(struct P *p, const char *const *stop)
{
	uint32_t at = cmdp_pos(p);
	char w[64];
	struct func *f;

	if (cmdp_word(p, w, sizeof w) && (f = func_find(sym_cstr(w))) != NULL &&
	    cmdp_at(p, T_COMMA))
		return f;
	cmdp_seek(p, at);
	return cmdp_func(p, stop);
}

int
texenc_command(struct P *p)
{
	static const char *const stop[] = {"rows", "tol", "astc", "eac", "etc2",
	    "srgb", "bc7", "bc6h", NULL};
	struct func *f, *g = NULL;
	char w[48];
	uint32_t at = cmdp_pos(p);
	double tol = 0;
	int has_tol = 0, astc = 0, eac = 0, etc2 = 0, srgb = 0, bc7 = 0, bc6h = 0;
	int64_t W = 4, bw = 0, bh = 0;

	if (!cmdp_word(p, w, sizeof w) || strcmp(w, "texenc") != 0 ||
	    func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	f = texenc_table(p, stop);
	if (cmdp_tok(p, T_COMMA))
		g = cmdp_func(p, stop);
	if (cmdp_kw(p, "astc")) {
		/* the footprint: 6x6 reads as 6 and x6, or 6 x 6 */
		astc = 1;
		if (!cmdp_int(p, &bw) || !cmdp_word(p, w, sizeof w) || w[0] != 'x' ||
		    (w[1] ? (bh = atoi(w + 1)) <= 0 : !cmdp_int(p, &bh)))
			fail("texenc: astc takes a block's footprint, BWxBH (6x6, 8x5, "
			    "...)");
		W = bw;
		srgb = cmdp_kw(p, "srgb");
	} else if (cmdp_kw(p, "eac"))
		eac = 1;
	else if (cmdp_kw(p, "etc2")) {
		etc2 = 1;
		srgb = cmdp_kw(p, "srgb");
	} else if (cmdp_kw(p, "bc7")) {
		bc7 = 1;
		srgb = cmdp_kw(p, "srgb");
	} else if (cmdp_kw(p, "bc6h"))
		bc6h = 1;
	while (!cmdp_eos(p)) {
		if (cmdp_kw(p, "rows")) {
			if (!cmdp_int(p, &W))
				fail("texenc: rows W, the texels in a row");
		} else if (cmdp_kw(p, "tol")) {
			tol = cmdp_number(p, stop);
			has_tol = 1;
			if (!(tol >= 0) || !isfinite(tol))
				fail("texenc: tol E, the error allowed on an entry, is "
				    "0 or more, not %g", tol);
		} else {
			cmdp_fail_here(p);
			fail("texenc T [, U] [astc BWxBH [srgb] | eac | etc2 [srgb] | "
			    "bc7 [srgb] | bc6h] [rows W] [tol E]");
		}
	}
	notes_flush();
	if (bc6h) {
		if (g)
			fail("texenc: bc6h takes one table (its red channel), not two");
		cmd_texenc_bc6h(f, (int)W, tol, has_tol);
	} else if (bc7)
		cmd_texenc_bc7(f, g, (int)W, tol, has_tol, srgb);
	else if (etc2)
		cmd_texenc_etc2(f, g, (int)W, tol, has_tol, srgb);
	else if (eac)
		cmd_texenc_eac(f, g, (int)W, tol, has_tol);
	else if (astc)
		cmd_texenc_astc(f, g, (int)bw, (int)bh, (int)W, tol, has_tol, srgb);
	else if (g)
		cmd_texenc5(f, g, (int)W, tol, has_tol);
	else
		cmd_texenc(f, (int)W, tol, has_tol);
	return 1;
}

/*
 * "compress K as BC4", "K as a BC4 texture", "encode K as bc4 within
 * 0.01, rows of 8", "compress K as ASTC 6x6", "store K as EAC", "K and L
 * as RG11", "compress K as ETC2": texenc T [astc BWxBH | eac | etc2]
 * [rows W] [tol E].  The sentence must name BC4 (or RGTC), ASTC with a
 * footprint, EAC (R11, RG11) or ETC2, and a defined table; NULL
 * otherwise.
 */
const char *
texenc_nlp_raw(const char *sent)
{
	static char out[200];
	char w[48], prev[48] = "", prev2[48] = "", fn[48] = "", fn2[48] = "",
	    tol[32] = "", rows[16] = "", fp[16] = "";
	const char *s;
	size_t k, n = strlen(sent);
	int hit = 0, bc5 = 0, astc = 0, eac = 0, etc2 = 0, srgb = 0, bc7 = 0,
	    bc6h = 0, a, b;

	if (n > 400)
		return NULL;
	for (s = sent; *s;) {
		while (*s && !(*s == '_' || *s == '.' || (*s >= '0' && *s <= '9') ||
		    (*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z')))
			s++;
		for (k = 0; *s && (*s == '_' || *s == '.' || (*s >= '0' &&
		    *s <= '9') || (*s >= 'a' && *s <= 'z') || (*s >= 'A' &&
		    *s <= 'Z')); s++)
			if (k + 1 < sizeof w)
				w[k++] = *s;
		while (k > 0 && w[k - 1] == '.')
			k--;
		w[k] = 0;
		if (!k)
			continue;
		if (!strcmp(w, "bc4") || !strcmp(w, "BC4") || !strcmp(w, "rgtc") ||
		    !strcmp(w, "RGTC") || !strcmp(w, "rgtc1") || !strcmp(w, "RGTC1"))
			hit = 1;
		else if (!strcmp(w, "bc5") || !strcmp(w, "BC5") || !strcmp(w,
		    "rgtc2") || !strcmp(w, "RGTC2"))
			hit = bc5 = 1;
		else if (!strcmp(w, "astc") || !strcmp(w, "ASTC"))
			hit = astc = 1;
		else if (!strcmp(w, "eac") || !strcmp(w, "EAC") || !strcmp(w,
		    "r11") || !strcmp(w, "R11"))
			hit = eac = 1;
		else if (!strcmp(w, "rg11") || !strcmp(w, "RG11"))
			hit = eac = bc5 = 1;
		else if (!strcmp(w, "etc2") || !strcmp(w, "ETC2"))
			hit = etc2 = 1;
		else if (!strcmp(w, "bc7") || !strcmp(w, "BC7"))
			hit = bc7 = 1;
		else if (!strcmp(w, "bc6h") || !strcmp(w, "BC6H"))
			hit = bc6h = 1;
		else if (!strcmp(w, "srgb") || !strcmp(w, "sRGB") || !strcmp(w,
		    "SRGB"))
			srgb = 1;
		else if (sscanf(w, "%dx%d", &a, &b) == 2 && a > 0 && b > 0 &&
		    a < 100 && b < 100)
			snprintf(fp, sizeof fp, "%dx%d", a, b);
		else if ((!strcmp(prev, "within") || !strcmp(prev, "tol")) &&
		    (w[0] == '.' || (w[0] >= '0' && w[0] <= '9')))
			snprintf(tol, sizeof tol, "%s", w);
		else if ((!strcmp(prev, "rows") || (!strcmp(prev, "of") &&
		    !strcmp(prev2, "rows"))) && w[0] >= '0' && w[0] <= '9')
			snprintf(rows, sizeof rows, "%s", w);
		else if (!fn2[0] && func_find(sym_cstr(w)) &&
		    func_find(sym_cstr(w))->np == 0 &&
		    N.op[func_find(sym_cstr(w))->body] == O_TABLE)
			snprintf(fn[0] ? fn2 : fn, sizeof fn, "%s", w);
		snprintf(prev2, sizeof prev2, "%s", prev);
		snprintf(prev, sizeof prev, "%s", w);
	}
	if (!hit || !fn[0] || (bc5 && !fn2[0]) || (astc && !fp[0]))
		return NULL;
	snprintf(out, sizeof out, "texenc %s%s%s%s%s%s%s%s%s%s", fn, bc5 ||
	    ((astc || eac || etc2 || bc7) && fn2[0]) ? ", " : "", bc5 ||
	    ((astc || eac || etc2 || bc7) && fn2[0]) ? fn2 : "", astc ?
	    " astc " : "", astc ? fp : "", astc && srgb ? " srgb" : eac ? " eac" :
	    etc2 ? (srgb ?
	    " etc2 srgb" : " etc2") : bc7 ? (srgb ? " bc7 srgb" : " bc7") :
	    bc6h ? " bc6h" : "", rows[0] ? " rows " : "", rows, tol[0] ?
	    " tol " : "", tol);
	return out;
}

/* w (lower case) as a word of line, in either case: "eac" is inside
 * "each" and "reach" */
static int
has_word(const char *line, const char *w)
{
	size_t n = strlen(w), i, k;

	for (i = 0; line[i]; i++) {
		for (k = 0; k < n && line[i + k] && (line[i + k] | 32) == w[k]; k++)
			;
		if (k == n && (i == 0 || !isalnum((unsigned char)line[i - 1])) &&
		    !isalnum((unsigned char)line[i + n]))
			return 1;
	}
	return 0;
}

/*
 * Before anything is defined: a line that starts with a verb for it and
 * names the format is such a sentence ("compress K as BC4"), not a
 * product of names.
 */
int
texenc_nlp_english(const char *line)
{
	static const char *const verbs[] = {"compress", "encode", "pack",
		"store", "write", "put", "can", "is", "does", "how", "what", NULL};
	char w[16];
	size_t k = 0;
	int i, verb = 0;

	for (; line[k] && line[k] != ' ' && k + 1 < sizeof w; k++)
		w[k] = (char)(line[k] >= 'A' && line[k] <= 'Z' ? line[k] + 32 :
		    line[k]);
	w[k] = 0;
	for (i = 0; verbs[i]; i++)
		if (!strcmp(w, verbs[i]))
			verb = 1;
	return verb && (strstr(line, "BC4") || strstr(line, "bc4") ||
	    strstr(line, "BC5") || strstr(line, "bc5") || strstr(line, "RGTC") ||
	    strstr(line, "rgtc") || strstr(line, "ASTC") || strstr(line, "astc") ||
	    has_word(line, "eac") || strstr(line, "R11") || strstr(line, "r11") ||
	    strstr(line, "RG11") || strstr(line, "rg11") || strstr(line, "ETC2") ||
	    strstr(line, "etc2") || strstr(line, "BC7") || strstr(line, "bc7") ||
	    strstr(line, "BC6H") || strstr(line, "bc6h"));
}
