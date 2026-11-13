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
 */
#include "calc.h"
#include "report.h"
#include "texenc.h"

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

int
texenc_lossless(nid tab, int natw, int *W, int *bytes)
{
	struct texenc e;
	int w, n, k, best = 0, stored;

	if (N.op[tab] != O_TABLE || N.b[tab] < 2)
		return 0;
	n = (int)N.b[tab];
	stored = n * ((TYW(N.ty[N.lst[N.a[tab]]]) + 7) / 8);
	for (k = -1, w = natw; k <= 12; w = 4 << ++k) {
		if (w < 1 || w > 4096 || (k >= 0 && w > 4 && w / 2 >= n) ||
		    (k >= 0 && w == natw))
			continue;
		if (!texenc_bc4(tab, w, &e))
			return 0;	/* not numbers BC4 holds */
		if (e.err == 0 && e.bw * e.bh * 8 < stored && (!best ||
		    e.bw * e.bh * 8 < *bytes)) {
			best = 1;
			*W = w;
			*bytes = e.bw * e.bh * 8;
		}
		texenc_free(&e);
	}
	return best;
}

void
texenc_note(nid tab, int natw, const char *what, const char *ind)
{
	int W, bytes, n;
	ty_t t;

	if (!texenc_lossless(tab, natw, &W, &bytes))
		return;
	n = (int)N.b[tab];
	t = N.ty[N.lst[N.a[tab]]];
	printf("%sthe %s are exact as BC4_UNORM blocks, rows of %d: %d bytes "
	    "against %d as %s (texenc; a read returns k/255; a compressed "
	    "read has no price of its own, so what is priced stays as it is)\n",
	    ind, what, W, bytes, n * ((TYW(t) + 7) / 8), ty_str(t));
	rep_obj("bc4");
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
	static const char *const stop[] = {"rows", "tol", NULL};
	struct func *f, *g = NULL;
	char w[48];
	uint32_t at = cmdp_pos(p);
	double tol = 0;
	int has_tol = 0;
	int64_t W = 4;

	if (!cmdp_word(p, w, sizeof w) || strcmp(w, "texenc") != 0 ||
	    func_find(sym_cstr(w))) {
		cmdp_seek(p, at);
		return 0;
	}
	f = texenc_table(p, stop);
	if (cmdp_tok(p, T_COMMA))
		g = cmdp_func(p, stop);
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
			fail("texenc T [, U] [rows W] [tol E]");
		}
	}
	notes_flush();
	if (g)
		cmd_texenc5(f, g, (int)W, tol, has_tol);
	else
		cmd_texenc(f, (int)W, tol, has_tol);
	return 1;
}

/*
 * "compress K as BC4", "K as a BC4 texture", "encode K as bc4 within
 * 0.01, rows of 8": texenc T [rows W] [tol E].  The sentence must name
 * BC4 (or RGTC) and a defined table; NULL otherwise.
 */
const char *
texenc_nlp_raw(const char *sent)
{
	static char out[200];
	char w[48], prev[48] = "", prev2[48] = "", fn[48] = "", fn2[48] = "",
	    tol[32] = "", rows[16] = "";
	const char *s;
	size_t k, n = strlen(sent);
	int hit = 0, bc5 = 0;

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
	if (!hit || !fn[0] || (bc5 && !fn2[0]))
		return NULL;
	snprintf(out, sizeof out, "texenc %s%s%s%s%s%s%s", fn, bc5 ? ", " : "",
	    bc5 ? fn2 : "", rows[0] ? " rows " : "", rows, tol[0] ? " tol " : "",
	    tol);
	return out;
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
	    strstr(line, "rgtc"));
}
