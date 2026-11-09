/*
 * animate scan: an animated GIF of a parallel prefix sum (scan), laid out
 * as one row of cells per level, with time measured by the active cost
 * model.
 *
 *   animate [hillis | blelloch | sequential] scan [n N] [fps F]
 *       [seconds S] [random [seed K] | values a, b, ...] [f32]
 *       [MODEL] [to NAME]
 *
 * The argument parser spots keywords and skips filler, so the sentence
 * "animate random sample prefix sum scan n=8 at 30 fps using gt1030" is
 * ucas input as it stands.
 *
 * Cost model (one work group of n threads on one SM; n <= 100 is at most
 * four warps, which Pascal's four schedulers issue side by side):
 *   a parallel level costs one lane's work, loads + add + store, plus
 *   branch + diverge when some lanes of a warp sit the level out, plus a
 *   barrier before the next level reads what this one wrote;
 *   the sequential scan keeps the running sum in a register: one load,
 *   one add and one store per element, no barriers.
 * Work is the sum over all lanes.  The time axis of the animation is the
 * critical path, in the model's units.
 *
 * In the bot the GIF travels inside the output after a "\x1efile" marker,
 * base64-encoded, and the bot attaches it; the evaluator cannot write
 * files there.  On the command line it is written to NAME.gif.
 */
#include "ucas.h"
#include "gif.h"
#include "font6x10.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#define MAXN 100
#define MAXCELLS 128		/* Blelloch pads to a power of two */
#define MAXLV MAXN		/* sequential: one per element; parallel: 2 log n + 1 */
#define MAXGIF (700 << 10)	/* fits the bot's 1 MiB output as base64 */

enum { AL_HILLIS, AL_BLELLOCH, AL_SEQ };
static const char *const alname[] = {"Hillis-Steele", "Blelloch",
	"sequential"};

struct aop {
	short dst, src;
	unsigned char swap;	/* down-sweep: a[src] = a[dst], a[dst] += old */
};

struct level {
	char label[16];
	char what[64];
	int nops, lanes, diverged;
	struct aop ops[MAXCELLS];
	int64_t v[MAXCELLS];	/* values after the level */
	double t0, tc, t1;	/* start, compute done, barrier done */
	double lane, work;
};

static struct {
	int alg, n, cells, fp, nlv;
	int64_t in[MAXCELLS];
	struct level lv[MAXLV + 1];
	double T, work;
	double cl, ca, cb, cbr, cdv;
} S;

/* ---- the algorithms, as levels of independent operations ---- */

static void
costs(const struct model *m)
{
	S.cl = m->c[C_LOAD];
	S.ca = S.fp ? m->c[C_FADD] : m->c[C_IADD];
	S.cb = m->c[C_BARRIER];
	S.cbr = m->c[C_BRANCH];
	S.cdv = m->c[C_DIVERGE];
}

static struct level *
newlevel(const int64_t *prev, const char *label, const char *what)
{
	struct level *L = &S.lv[++S.nlv];

	memset(L, 0, sizeof *L);
	snprintf(L->label, sizeof L->label, "%s", label);
	snprintf(L->what, sizeof L->what, "%s", what);
	memcpy(L->v, prev, sizeof L->v);
	return L;
}

static void
apply(struct level *L, const int64_t *old)
{
	int k;

	for (k = 0; k < L->nops; k++) {
		struct aop *o = &L->ops[k];
		if (o->swap) {
			L->v[o->src] = old[o->dst];
			L->v[o->dst] = old[o->dst] + old[o->src];
		} else
			L->v[o->dst] = old[o->dst] + old[o->src];
	}
}

static void
build(int alg)
{
	struct level *L;
	char lab[16], what[64];
	int d, k, i;

	S.alg = alg;
	S.nlv = 0;
	S.cells = S.n;
	if (alg == AL_BLELLOCH)
		while (S.cells & (S.cells - 1))
			S.cells++;
	memset(&S.lv[0], 0, sizeof S.lv[0]);
	snprintf(S.lv[0].label, sizeof S.lv[0].label, "in");
	for (i = 0; i < S.cells; i++)
		S.lv[0].v[i] = i < S.n ? S.in[i] : 0;
	if (alg == AL_HILLIS) {
		for (d = 1; d < S.n; d *= 2) {
			snprintf(lab, sizeof lab, "d=%d", d);
			snprintf(what, sizeof what, "a[i] += a[i-%d] for i >= %d", d, d);
			L = newlevel(S.lv[S.nlv].v, lab, what);
			for (i = d; i < S.n; i++) {
				L->ops[L->nops].dst = (short)i;
				L->ops[L->nops++].src = (short)(i - d);
			}
			L->lanes = S.n;
			apply(L, S.lv[S.nlv - 1].v);
		}
	} else if (alg == AL_BLELLOCH) {
		for (d = 1; d < S.cells; d *= 2) {
			snprintf(lab, sizeof lab, "up %d", d);
			snprintf(what, sizeof what, "up-sweep: a[i] += a[i-%d]", d);
			L = newlevel(S.lv[S.nlv].v, lab, what);
			for (i = 2 * d - 1; i < S.cells; i += 2 * d) {
				L->ops[L->nops].dst = (short)i;
				L->ops[L->nops++].src = (short)(i - d);
			}
			L->lanes = S.cells / 2;
			apply(L, S.lv[S.nlv - 1].v);
		}
		L = newlevel(S.lv[S.nlv].v, "clear", "a[n-1] = 0 (exclusive scan)");
		L->v[S.cells - 1] = 0;
		L->lanes = 1;
		L->nops = 0;
		for (d = S.cells / 2; d >= 1; d /= 2) {
			snprintf(lab, sizeof lab, "down %d", d);
			snprintf(what, sizeof what, "down-sweep: swap and add, stride %d",
			    d);
			L = newlevel(S.lv[S.nlv].v, lab, what);
			for (i = 2 * d - 1; i < S.cells; i += 2 * d) {
				L->ops[L->nops].dst = (short)i;
				L->ops[L->nops].src = (short)(i - d);
				L->ops[L->nops++].swap = 1;
			}
			L->lanes = S.cells / 2;
			apply(L, S.lv[S.nlv - 1].v);
		}
	} else {
		/* one row that fills left to right; each step is a level */
		for (i = 1; i < S.n; i++) {
			snprintf(lab, sizeof lab, "i=%d", i);
			snprintf(what, sizeof what, "s += a[%d]; out[%d] = s", i, i);
			L = newlevel(S.lv[S.nlv].v, lab, what);
			L->ops[0].dst = (short)i;
			L->ops[0].src = (short)(i - 1);
			L->nops = 1;
			L->lanes = 1;
			apply(L, S.lv[S.nlv - 1].v);
		}
	}
	/* times */
	{
		double t = 0;
		S.work = 0;
		for (k = 1; k <= S.nlv; k++) {
			L = &S.lv[k];
			L->t0 = t;
			if (alg == AL_SEQ) {
				L->lane = S.cl + S.ca + S.cl + (k == 1 ? S.cl : 0);
				L->work = L->lane;
				L->tc = L->t1 = t + L->lane;
			} else {
				int swaps = L->nops && L->ops[0].swap;
				L->lane = L->nops == 0 ? S.cl :   /* the clear: a store */
				    swaps ? 4 * S.cl + S.ca : 3 * S.cl + S.ca;
				L->diverged = L->nops > 0 && L->nops < L->lanes;
				L->work = L->nops ? L->nops * L->lane : S.cl;
				if (alg == AL_HILLIS)   /* idle lanes copy their cell */
					L->work += (L->lanes - L->nops) * 2 * S.cl;
				L->tc = t + L->lane + (L->diverged ? S.cbr + S.cdv : 0);
				L->t1 = k < S.nlv ? L->tc + S.cb : L->tc;
			}
			S.work += L->work;
			t = L->t1;
		}
		S.T = t;
	}
}

/* ---- drawing ---- */

enum { C_BG, C_TEXT, C_GRID, C_GRAY, C_CELL, C_HI, C_ARROW, C_BAR_ORANGE,
	C_RED, C_DONE, C_PEND, C_BAR, C_FAINT, C_TLBG, NCOL };
static const unsigned char pal[16][3] = {
	{255, 255, 255}, {25, 25, 30}, {205, 205, 210}, {130, 130, 140},
	{226, 236, 250}, {255, 212, 90}, {40, 100, 220}, {240, 140, 30},
	{215, 60, 60}, {60, 160, 80}, {170, 188, 220}, {95, 135, 205},
	{196, 204, 218}, {236, 236, 238}, {0, 0, 0}, {0, 0, 0},
};

static struct {
	int W, H, cw, ch, gap, left, top, text;
	unsigned char *px;
	int64_t vmax;
} D;

static void
dot(int x, int y, int c)
{
	if (x >= 0 && y >= 0 && x < D.W && y < D.H)
		D.px[(size_t)y * D.W + x] = (unsigned char)c;
}

static void
box(int x, int y, int w, int h, int c)
{
	int i, j;

	for (j = y; j < y + h; j++)
		for (i = x; i < x + w; i++)
			dot(i, j, c);
}

static void
frame_rect(int x, int y, int w, int h, int c)
{
	box(x, y, w, 1, c);
	box(x, y + h - 1, w, 1, c);
	box(x, y, 1, h, c);
	box(x + w - 1, y, 1, h, c);
}

/* the first fraction f of the segment (x0, y0)-(x1, y1) */
static void
seg(int x0, int y0, int x1, int y1, double f, int c)
{
	int n = abs(x1 - x0) > abs(y1 - y0) ? abs(x1 - x0) : abs(y1 - y0), i;

	if (n == 0) {
		dot(x0, y0, c);
		return;
	}
	for (i = 0; i <= (int)(n * f + 0.5); i++)
		dot(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, c);
}

static int
text(int x, int y, const char *s, int c)
{
	int i, r, b;

	for (; *s; s++, x += FONT_6X10_W) {
		unsigned ch = (unsigned char)*s;
		if (ch < 32 || ch > 126)
			ch = '?';
		for (r = 0; r < FONT_6X10_H; r++) {
			unsigned bits = font_6x10[ch - 32][r];
			for (b = 0; b < FONT_6X10_W; b++)
				if (bits & (0x80 >> b))
					dot(x + b, y + r, c);
		}
	}
	(void)i;
	return x;
}

/* a legend entry: a colour swatch, then the label in grey */
static int
swatch(int x, int y, int col, const char *label)
{
	box(x, y + 2, 8, 7, col);
	return text(x + 11, y, label, C_GRAY) + FONT_6X10_W;
}

static int
row_y(int k)
{
	return D.top + k * (D.ch + D.gap);
}

static int
cell_x(int i)
{
	return D.left + i * D.cw;
}

static void
draw_row(int k, const int64_t *v, const unsigned char *hi, int pending)
{
	int i, y = row_y(k);
	char buf[24];

	text(4, y + (D.ch - FONT_6X10_H) / 2 + 1, S.lv[k].label, C_GRAY);
	for (i = 0; i < S.cells; i++) {
		int x = cell_x(i), pad = i >= S.n;
		if (pending) {
			frame_rect(x + 1, y, D.cw - 2, D.ch, C_PEND);
			continue;
		}
		box(x + 1, y, D.cw - 2, D.ch, pad ? C_TLBG : hi && hi[i] ? C_HI :
		    C_CELL);
		frame_rect(x + 1, y, D.cw - 2, D.ch, C_GRID);
		if (D.text) {
			int len = snprintf(buf, sizeof buf, "%lld", (long long)v[i]);
			text(x + (D.cw - len * FONT_6X10_W) / 2 + 1,
			    y + (D.ch - FONT_6X10_H) / 2 + 1, buf, pad ? C_GRAY : C_TEXT);
		} else if (D.vmax > 0 && v[i] > 0) {
			int h = (int)((D.ch - 2) * (double)v[i] / (double)D.vmax + 0.5);
			if (h < 1)
				h = 1;
			box(x + 2, y + D.ch - 1 - h, D.cw - 4 > 1 ? D.cw - 4 : 1, h,
			    hi && hi[i] ? C_HI : C_BAR);
		}
	}
}

static void
draw_arrows(int k, double f, int c)
{
	const struct level *L = &S.lv[k];
	int j, y0 = row_y(k - 1) + D.ch, y1 = row_y(k) - 1;

	for (j = 0; j < L->nops; j++) {
		int xs = cell_x(L->ops[j].src) + D.cw / 2;
		int xd = cell_x(L->ops[j].dst) + D.cw / 2;
		seg(xs, y0, xd, y1, f, c);
		if (L->ops[j].swap)   /* the left cell takes the right's value */
			seg(xd, y0, xs, y1, f, c);
		if (f >= 1) {
			dot(xd - 1, y1 - 1, c);
			dot(xd + 1, y1 - 1, c);
			dot(xd, y1, c);
		}
	}
}

static void
draw_frame(double t)
{
	int k, i, cur = 0, x0, x1, tlw, ty;
	char buf[160];
	unsigned char hi[MAXCELLS];
	const struct level *L;

	memset(D.px, C_BG, (size_t)D.W * D.H);
	/* which level is running */
	for (k = 1; k <= S.nlv; k++)
		if (t >= S.lv[k].t0)
			cur = k;
	if (t >= S.T)
		cur = S.nlv + 1;
	/* header */
	snprintf(buf, sizeof buf, "%s %s scan, n = %d%s, %s, cost model %s",
	    alname[S.alg], S.alg == AL_BLELLOCH ? "exclusive" : "inclusive", S.n,
	    S.cells != S.n ? S.alg == AL_BLELLOCH ? " (padded to 2^k)" : "" : "",
	    S.fp ? "f32" : "u32", sym_str(model_active()->name));
	text(4, 4, buf, C_TEXT);
	{
		double w = 0, tt = t < S.T ? t : S.T;
		for (k = 1; k <= S.nlv; k++) {
			L = &S.lv[k];
			if (tt >= L->t1)
				w += L->work;
			else if (tt > L->t0)
				w += L->work * (tt - L->t0) / (L->t1 - L->t0);
		}
		snprintf(buf, sizeof buf, "time %6.1f / %.0f (critical path)   "
		    "work %6.0f / %.0f", tt, S.T, w, S.work);
		text(4, D.H - 26, buf, C_TEXT);
	}
	if (cur == 0)
		snprintf(buf, sizeof buf, "input");
	else if (cur > S.nlv)
		snprintf(buf, sizeof buf, "done");
	else {
		L = &S.lv[cur];
		if (S.alg != AL_SEQ && t >= L->tc && cur < S.nlv)
			snprintf(buf, sizeof buf, "level %d/%d: barrier (%g)", cur,
			    S.nlv, S.cb);
		else if (S.alg == AL_SEQ)
			snprintf(buf, sizeof buf, "step %d/%d: %s", cur, S.nlv, L->what);
		else
			snprintf(buf, sizeof buf, "level %d/%d: %s, %d lanes%s", cur,
			    S.nlv, L->what, L->nops, L->diverged ?
			    ", divergent warp" : "");
	}
	text(4, D.H - 13, buf, cur > S.nlv ? C_DONE : C_ARROW);

	if (S.alg == AL_SEQ) {
		/* input row, and the output row filling in */
		int done = 0;
		draw_row(0, S.lv[0].v, NULL, 0);
		for (k = 1; k <= S.nlv; k++)
			if (t >= S.lv[k].tc)
				done = k;
		memset(hi, 0, sizeof hi);
		if (done >= 1 && done < S.nlv + 1)
			hi[done] = 1;
		{
			int64_t out[MAXCELLS];
			int y = row_y(1);
			memcpy(out, S.lv[done].v, sizeof out);
			text(4, y + (D.ch - FONT_6X10_H) / 2 + 1, "out", C_GRAY);
			for (i = 0; i < S.n; i++) {
				int x = cell_x(i);
				char nb[24];
				if (i > done) {
					frame_rect(x + 1, y, D.cw - 2, D.ch, C_PEND);
					continue;
				}
				box(x + 1, y, D.cw - 2, D.ch, hi[i] ? C_HI : C_CELL);
				frame_rect(x + 1, y, D.cw - 2, D.ch, C_GRID);
				if (D.text) {
					int len = snprintf(nb, sizeof nb, "%lld",
					    (long long)out[i]);
					text(x + (D.cw - len * FONT_6X10_W) / 2 + 1,
					    y + (D.ch - FONT_6X10_H) / 2 + 1, nb, C_TEXT);
				} else if (D.vmax > 0 && out[i] > 0) {
					int h = (int)((D.ch - 2) * (double)out[i] /
					    (double)D.vmax + 0.5);
					box(x + 2, y + D.ch - 1 - (h < 1 ? 1 : h),
					    D.cw - 4 > 1 ? D.cw - 4 : 1, h < 1 ? 1 : h,
					    hi[i] ? C_HI : C_BAR);
				}
			}
			if (cur >= 1 && cur <= S.nlv) {
				/* the element being added flows down */
				L = &S.lv[cur];
				i = L->ops[0].dst;
				seg(cell_x(i) + D.cw / 2, row_y(0) + D.ch,
				    cell_x(i) + D.cw / 2, row_y(1) - 1,
				    (t - L->t0) / (L->tc - L->t0), C_ARROW);
			}
		}
	} else {
		draw_row(0, S.lv[0].v, NULL, 0);
		for (k = 1; k <= S.nlv; k++) {
			L = &S.lv[k];
			if (t < L->t0) {
				draw_row(k, L->v, NULL, 1);
				continue;
			}
			if (t < L->tc) {   /* computing: arrows grow */
				draw_row(k, L->v, NULL, 1);
				draw_arrows(k, (t - L->t0) / (L->tc - L->t0), C_ARROW);
				continue;
			}
			memset(hi, 0, sizeof hi);
			if (k == cur && cur <= S.nlv)
				for (i = 0; i < L->nops; i++) {
					hi[L->ops[i].dst] = 1;
					if (L->ops[i].swap)
						hi[L->ops[i].src] = 1;
				}
			if (L->nops == 0 && k == cur)
				hi[S.cells - 1] = 1;
			draw_row(k, L->v, hi, 0);
			draw_arrows(k, 1, k == cur ? C_ARROW : C_FAINT);
			if (k == cur && t < L->t1 && k < S.nlv) {   /* barrier */
				int y = row_y(k) + D.ch + D.gap / 2;
				for (i = D.left; i < D.left + S.cells * D.cw; i += 6)
					box(i, y, 3, 2, C_BAR_ORANGE);
				text(D.left + S.cells * D.cw + 4 - 7 * FONT_6X10_W > D.left ?
				    D.left + S.cells * D.cw - 7 * FONT_6X10_W : D.left,
				    y - FONT_6X10_H - 1, "barrier", C_BAR_ORANGE);
			}
		}
	}

	/* timeline: compute (blue), divergence (red), barriers (orange) */
	x0 = D.left;
	x1 = D.W - 10;
	tlw = x1 - x0;
	ty = D.H - 56;
	box(x0, ty, tlw, 10, C_TLBG);
	for (k = 1; k <= S.nlv; k++) {
		int a, b, c;
		L = &S.lv[k];
		a = x0 + (int)(tlw * L->t0 / S.T);
		c = x0 + (int)(tlw * L->tc / S.T);
		b = L->diverged ? x0 + (int)(tlw * (L->t0 + L->lane) / S.T) : c;
		box(a, ty, b - a, 10, C_BAR);
		if (c > b)
			box(b, ty, c - b, 10, C_RED);
		box(c, ty, x0 + (int)(tlw * L->t1 / S.T) - c, 10, C_BAR_ORANGE);
		box(a, ty, 1, 10, C_BG);
	}
	{
		int cx = x0 + (int)(tlw * (t < S.T ? t : S.T) / S.T);
		box(cx - 1, ty - 3, 2, 16, C_TEXT);
	}
	text(x0, ty + 12, "0", C_GRAY);
	snprintf(buf, sizeof buf, "%.0f", S.T);
	text(x1 - (int)strlen(buf) * FONT_6X10_W, ty + 12, buf, C_GRAY);
	{
		/* legend: what the colours of the timeline mean */
		int lx = x0 + tlw / 2 - 15 * FONT_6X10_W;
		lx = swatch(lx, ty + 12, C_BAR, "compute");
		lx = swatch(lx, ty + 12, C_RED, "diverge");
		swatch(lx, ty + 12, C_BAR_ORANGE, "barrier");
	}
}

/* ---- options ---- */

static const char *const stopw[] = {
	"n", "fps", "at", "using", "with", "under", "seconds", "second", "s",
	"elements", "element", "items", "values", "random", "sample", "seed",
	"to", "hillis", "steele", "blelloch", "sequential", "serial", "scan",
	"prefix", "sum", "sums", "f32", "u32", "for", "frames", "per", "and",
	"on", "of", "a", "the", "in", "nofile", "blelloch", "sequential",
	"efficient", "work", "values", "seconds", NULL
};
static const char *const filler[] = {
	"a", "an", "the", "of", "prefix", "sum", "sums", "scan", "scans",
	"parallel", "at", "using", "with", "under", "model", "cost", "for",
	"frames", "per", "second", "and", "on", "in", "gpu", "algorithm",
	"animation", "animate", "show", "me", "please", "it", "inclusive",
	"exclusive", "workgroup", "threads", "sample", "sampled", "data",
	"of", "lanes",
	"input", "array", "fps", "frame", "rate", "nofile", NULL
};

static int in_words(const char *w, const char *const *list);

/* an unseeded sample's seed: the clock, or UCAS_SEED (tests) */
static uint64_t
time_seed(void)
{
	const char *e = getenv("UCAS_SEED");

	if (e != NULL && *e)
		return strtoull(e, NULL, 10);
	return (uint64_t)time(NULL) % 100000 + 1;
}

/* the next token is a word from list (it is not consumed) */
static int
in_words_at(struct P *p, const char *const *list)
{
	char w[64];
	uint32_t pos = cmdp_pos(p);
	int r = cmdp_word(p, w, sizeof w) && in_words(w, list);

	cmdp_seek(p, pos);
	return r;
}

static int
in_words(const char *w, const char *const *list)
{
	int i;

	for (i = 0; list[i]; i++)
		if (!strcmp(w, list[i]))
			return 1;
	return 0;
}

static void
b64(struct sbuf *out, const unsigned char *s, size_t n)
{
	static const char tab[] =
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	size_t i;
	int col = 0;

	for (i = 0; i < n; i += 3) {
		unsigned v = (unsigned)s[i] << 16 | (i + 1 < n ? s[i + 1] << 8 : 0) |
		    (i + 2 < n ? s[i + 2] : 0);
		sb_putc(out, tab[v >> 18 & 63]);
		sb_putc(out, tab[v >> 12 & 63]);
		sb_putc(out, i + 1 < n ? tab[v >> 6 & 63] : '=');
		sb_putc(out, i + 2 < n ? tab[v & 63] : '=');
		if ((col += 4) >= 76) {
			sb_putc(out, '\n');
			col = 0;
		}
	}
	if (col)
		sb_putc(out, '\n');
}

static void deliver(struct gif *gp, const char *name, int nofile);

/* ---- animate exec: an expression, instruction by instruction ---- */

/*
 * The expression is lowered to a serial instruction list (one per
 * operation; casts that cost nothing alias their operand) and run on a
 * group of lanes in lockstep, SIMT style: a 2x2 quad by default, up to a
 * warp of 32.  Every instruction takes its cost under the model for all
 * lanes at once.  A ?: whose lanes disagree is a branch: the "then" side
 * runs with the other lanes masked, then the "else" side, then the lanes
 * reconverge, so every lane pays for both sides (keep-up); when all lanes
 * agree, only the taken side runs.  barrier() stalls every lane; an
 * atomic is serialized, one lane after another.
 */
#define MAXINS 1024
#define MAXLANE 32

struct ins {
	nid n;
	int kind, cls, lane, reg;
	char text[96];
	char dst[16];			/* the register it writes, "t3" or "%20" */
	char val[MAXLANE][24];		/* each lane's value after it */
	double cost, t0, t1;
	uint32_t mask;
};

static struct {
	struct ins in[MAXINS];
	int nins, nregs, nl, cols, rows;
	nid rnode[MAXINS];
	int rreg[MAXINS];
	int nr;
	struct func *f;
	struct plan p;
	double T;
	const struct model *m;
	int64_t vi[MAXP][MAXLANE];
	double vr[MAXP][MAXLANE];
	char title[200];
	char lanein[MAXLANE][48];	/* what each lane starts with */
} X;

static int
xreg(nid n)
{
	int i;

	for (i = 0; i < X.nr; i++)
		if (X.rnode[i] == n)
			return X.rreg[i];
	return -1;
}

static void
xsetreg(nid n, int r)
{
	if (X.nr < MAXINS) {
		X.rnode[X.nr] = n;
		X.rreg[X.nr++] = r;
	}
}

static void
xoperand(struct sbuf *b, nid a)
{
	int r = xreg(a);

	if (r >= 0)
		sb_printf(b, "t%d", r);
	else
		print_node(b, a, NULL);
}

static void
xtext(struct sbuf *b, nid n)
{
	int op = N.op[n], ar = opinfo[op].arity;

	if (op == O_INDEX) {
		sb_puts(b, "load table[");
		xoperand(b, N.b[n]);
		sb_puts(b, "]");
	} else if (op == O_CAST) {
		sb_printf(b, "%s(", ty_str(N.ty[n]));
		xoperand(b, N.a[n]);
		sb_puts(b, ")");
	} else if (ar == 1) {
		if (op == O_NEG || op == O_NOT || op == O_LNOT)
			sb_puts(b, opinfo[op].name);
		else
			sb_printf(b, "%s(", opinfo[op].name);
		xoperand(b, N.a[n]);
		if (!(op == O_NEG || op == O_NOT || op == O_LNOT))
			sb_puts(b, ")");
	} else if (op == O_ATOMIC && xreg(N.a[n]) < 0 && xreg(N.b[n]) < 0) {
		print_node(b, n, NULL);   /* atomic_add_shared(x, 1) */
	} else if (ar == 2 && (op == O_MIN || op == O_MAX || op == O_COPYSIGN ||
	    op == O_ATOMIC)) {
		sb_printf(b, "%s(", op == O_ATOMIC ? "atomic" : opinfo[op].name);
		xoperand(b, N.a[n]);
		sb_puts(b, ", ");
		xoperand(b, N.b[n]);
		sb_puts(b, ")");
	} else if (ar == 2) {
		xoperand(b, N.a[n]);
		sb_printf(b, " %s ", opinfo[op].name);
		xoperand(b, N.b[n]);
	} else if (ar == 3) {
		sb_printf(b, "%s(", opinfo[op].name);
		xoperand(b, N.a[n]);
		sb_puts(b, ", ");
		xoperand(b, N.b[n]);
		sb_puts(b, ", ");
		xoperand(b, N.c[n]);
		sb_puts(b, ")");
	}
}

static struct ins *
xadd(int kind, nid n, double cost, int cls, uint32_t mask)
{
	struct ins *I;

	if (X.nins == MAXINS)
		fail("animate exec: more than %d instructions", MAXINS);
	I = &X.in[X.nins++];
	memset(I, 0, sizeof *I);
	I->kind = kind;
	I->n = n;
	I->cost = cost;
	I->cls = cls;
	I->mask = mask;
	I->reg = -1;
	I->lane = -1;
	return I;
}

static void
lanes_text(char *buf, size_t n, uint32_t mask)
{
	int i, k = 0;

	buf[0] = 0;
	for (i = 0; i < X.nl; i++)
		if (mask >> i & 1)
			k += snprintf(buf + k, n - (size_t)k, "%s%d", k ? "," : "", i);
}

static void
xemit(nid n, uint32_t mask)
{
	struct sbuf b = {0};
	struct ins *I;
	int cls, i;
	double c;

	if (n == 0 || xreg(n) >= 0)
		return;
	switch (N.op[n]) {
	case O_CONST: case O_VAR: case O_UNDEF: case O_TABLE:
		return;
	}
	if (N.op[n] == O_SEL) {
		uint32_t mt = 0, mf;
		uint32_t loc;
		char lt[96], lf[96];
		int rt, rf;
		xemit(N.a[n], mask);
		loc = plan_loc(&X.p, N.a[n]);
		for (i = 0; i < X.nl; i++)
			if ((mask >> i & 1) && loc != NOTOK && COLI(&X.p, loc)[i])
				mt |= 1u << i;
		mf = mask & ~mt;
		lanes_text(lt, sizeof lt, mt);
		lanes_text(lf, sizeof lf, mf);
		I = xadd(I_BRANCH, n, X.m->c[C_BRANCH], C_BRANCH, mask);
		xoperand(&b, N.a[n]);
		snprintf(I->text, sizeof I->text, "branch on %s: then {%s} else {%s}%s",
		    b.p, lt, lf, mt && mf ? " diverges" : " uniform");
		if (mt)
			xemit(N.b[n], mt);
		if (mf)
			xemit(N.c[n], mf);
		if (mt && mf) {
			I = xadd(I_RECONV, n, X.m->c[C_DIVERGE], C_DIVERGE, mask);
			snprintf(I->text, sizeof I->text, "reconverge (keep-up)");
		}
		/* the result: each lane's side; a register, no instruction */
		I = xadd(I_MERGE, n, 0, -1, mask);
		I->reg = X.nregs++;
		sb_clear(&b);
		sb_printf(&b, "t%d = ", I->reg);
		if (mt)
			xoperand(&b, N.b[n]);
		if (mt && mf)
			sb_puts(&b, " | ");
		if (mf)
			xoperand(&b, N.c[n]);
		snprintf(I->text, sizeof I->text, "%s  (per lane)", b.p);
		xsetreg(n, I->reg);
		(void)rt;
		(void)rf;
		free(b.p);
		return;
	}
	xemit(N.a[n], mask);
	if (opinfo[N.op[n]].arity >= 2)
		xemit(N.b[n], mask);
	if (opinfo[N.op[n]].arity >= 3)
		xemit(N.c[n], mask);
	c = node_cost(X.m, n, &cls);
	if (N.op[n] == O_BARRIER) {
		I = xadd(I_BARRIER, n, c, cls, mask);
		snprintf(I->text, sizeof I->text, "barrier()");
		xsetreg(n, xreg(N.a[n]));
		return;
	}
	if (cls < 0 || c == 0) {   /* free: an alias of its operand */
		if (xreg(N.a[n]) >= 0)
			xsetreg(n, xreg(N.a[n]));
		else if (N.op[n] != O_CAST) {
			I = xadd(I_OP, n, 0, cls, mask);
			I->reg = X.nregs++;
			xtext(&b, n);
			snprintf(I->text, sizeof I->text, "t%d = %s", I->reg, b.p);
			xsetreg(n, I->reg);
		}
		free(b.p);
		return;
	}
	xtext(&b, n);
	if (N.op[n] == O_ATOMIC) {
		int r = X.nregs++;
		for (i = 0; i < X.nl; i++) {
			if (!(mask >> i & 1))
				continue;
			I = xadd(I_ATOMIC, n, c, cls, 1u << i);
			I->lane = i;
			I->reg = r;
			snprintf(I->text, sizeof I->text, "t%d = %s  (lane %d)", r, b.p,
			    i);
		}
		xsetreg(n, r);
		free(b.p);
		return;
	}
	I = xadd(I_OP, n, c, cls, mask);
	I->reg = X.nregs++;
	snprintf(I->text, sizeof I->text, "t%d = %s", I->reg, b.p);
	xsetreg(n, I->reg);
	free(b.p);
}

/* lanes x instruction: the value lane i holds after instruction k */
static void
xvalue(char *buf, size_t n, nid g, int lane)
{
	uint32_t loc = plan_loc(&X.p, g);
	struct sbuf b = {0};

	if (loc == NOTOK) {
		snprintf(buf, n, "?");
		return;
	}
	if (COLF(&X.p, loc)[lane] & F_UNDEF)
		snprintf(buf, n, "undef");
	else {
		fmt_val(&b, X.p.ty[loc], COLI(&X.p, loc)[lane], COLR(&X.p, loc)[lane]);
		snprintf(buf, n, "%s", b.p);
	}
	free(b.p);
}

#define LBW 132		/* lane box */
#define LBH 64

#define LISTMAX 24		/* serial-form lines shown at once */

static void
xdraw(double t, int listx, int listw, int gridx, int top)
{
	int k, i, cur = -1, y, x0, x1, tlw, ty, first, lastk;
	char buf[160];
	const struct ins *I;

	memset(D.px, C_BG, (size_t)D.W * D.H);
	text(4, 4, X.title, C_TEXT);
	for (k = 0; k < X.nins; k++)
		if (X.in[k].cost > 0 && t >= X.in[k].t0 && t < X.in[k].t1)
			cur = k;
	lastk = -1;
	for (k = 0; k < X.nins; k++)
		if (t >= X.in[k].t1)
			lastk = k;
	/* the serial form, scrolled to keep the current line in view */
	first = 0;
	if (X.nins > LISTMAX) {
		int at = cur >= 0 ? cur : lastk >= 0 ? lastk : 0;
		first = at - LISTMAX / 3;
		if (first > X.nins - LISTMAX)
			first = X.nins - LISTMAX;
		if (first < 0)
			first = 0;
	}
	snprintf(buf, sizeof buf, "serial form%s", X.nins > LISTMAX ?
	    " (scrolling)" : "");
	text(listx, top - 14, buf, C_GRAY);
	for (k = first, y = top; k < X.nins && k < first + LISTMAX; k++) {
		int col;
		I = &X.in[k];
		col = k == cur ? C_ARROW : t >= I->t1 ? C_TEXT : C_GRAY;
		if (I->kind == I_BRANCH || I->kind == I_RECONV)
			col = k == cur || t >= I->t1 ? C_RED : C_GRAY;
		if (I->kind == I_BARRIER)
			col = k == cur || t >= I->t1 ? C_BAR_ORANGE : C_GRAY;
		if (k == cur)
			box(listx - 2, y - 1, listw, FONT_6X10_H + 2, C_HI);
		snprintf(buf, sizeof buf, "%c %s", t >= I->t1 && I->cost > 0 ? '*' :
		    k == cur ? '>' : ' ', I->text);
		buf[(listw - 40) / FONT_6X10_W] = 0;
		text(listx, y, buf, col);
		if (I->cost > 0) {
			snprintf(buf, sizeof buf, "%g", I->cost);
			text(listx + listw - 6 - (int)strlen(buf) * FONT_6X10_W, y, buf,
			    C_GRAY);
		}
		y += FONT_6X10_H + 3;
	}
	/* the lanes */
	for (i = 0; i < X.nl; i++) {
		int bx = gridx + (i % X.cols) * (LBW + 8);
		int by = top + (i / X.cols) * (LBH + 8);
		int active = cur >= 0 && (X.in[cur].mask >> i & 1);
		int stall = cur >= 0 && X.in[cur].kind == I_BARRIER;
		int atom = cur >= 0 && X.in[cur].kind == I_ATOMIC;
		int last = -1, j, fill;
		fill = cur < 0 ? C_CELL : stall ? C_TLBG : atom ? (active ? C_HI :
		    C_TLBG) : active ? C_CELL : C_TLBG;
		box(bx, by, LBW, LBH, fill);
		frame_rect(bx, by, LBW, LBH, stall ? C_BAR_ORANGE : active && cur >= 0 ?
		    C_ARROW : C_GRID);
		snprintf(buf, sizeof buf, "lane %d", i);
		text(bx + 4, by + 3, buf, C_TEXT);
		snprintf(buf, sizeof buf, "%s", X.lanein[i]);
		buf[(LBW - 8 - 8 * FONT_6X10_W) / FONT_6X10_W] = 0;
		text(bx + 4 + 7 * FONT_6X10_W, by + 3, buf, C_GRAY);
		/* the latest values this lane computed */
		for (k = 0; k < X.nins; k++)
			if (t >= X.in[k].t1 && X.in[k].dst[0] && (X.in[k].mask >> i & 1))
				last = k;
		for (k = last, j = 0; last >= 0 && k >= 0 && j < 3; k--) {
			I = &X.in[k];
			if (!I->dst[0] || !(I->mask >> i & 1) || t < I->t1)
				continue;
			snprintf(buf, sizeof buf, "%s = %s", I->dst, I->val[i]);
			buf[(LBW - 8) / FONT_6X10_W] = 0;
			text(bx + 4, by + 16 + (2 - j) * 12, buf, j ? C_GRAY : C_TEXT);
			j++;
		}
		if (cur >= 0 && !active && !stall)
			text(bx + LBW - 7 * FONT_6X10_W, by + 3, atom ? "waits" :
			    "masked", C_RED);
		if (stall)
			text(bx + LBW - 7 * FONT_6X10_W, by + 3, "barrier", C_BAR_ORANGE);
	}
	/* timeline */
	x0 = 46;
	x1 = D.W - 10;
	tlw = x1 - x0;
	ty = D.H - 56;
	box(x0, ty, tlw, 10, C_TLBG);
	for (k = 0; k < X.nins; k++) {
		int a, b2, col;
		I = &X.in[k];
		if (I->cost <= 0)
			continue;
		a = x0 + (int)(tlw * I->t0 / X.T);
		b2 = x0 + (int)(tlw * I->t1 / X.T);
		col = I->kind == I_BRANCH || I->kind == I_RECONV ? C_RED :
		    I->kind == I_BARRIER ? C_BAR_ORANGE : I->kind == I_ATOMIC ? C_HI :
		    C_BAR;
		box(a, ty, b2 - a, 10, col);
		if (b2 - a > 2)
			box(a, ty, 1, 10, C_BG);
	}
	{
		int cx = x0 + (int)(tlw * (t < X.T ? t : X.T) / X.T);
		box(cx - 1, ty - 3, 2, 16, C_TEXT);
	}
	text(x0, ty + 12, "0", C_GRAY);
	snprintf(buf, sizeof buf, "%g", X.T);
	text(x1 - (int)strlen(buf) * FONT_6X10_W, ty + 12, buf, C_GRAY);
	{
		int lx = x0 + tlw / 2 - 22 * FONT_6X10_W;
		lx = swatch(lx, ty + 12, C_BAR, "op");
		lx = swatch(lx, ty + 12, C_RED, "branch/keep-up");
		lx = swatch(lx, ty + 12, C_BAR_ORANGE, "barrier");
		swatch(lx, ty + 12, C_HI, "atomic");
	}
	{
		double busy = 0, tt = t < X.T ? t : X.T;
		for (k = 0; k < X.nins; k++) {
			double d;
			I = &X.in[k];
			if (I->cost <= 0 || tt <= I->t0)
				continue;
			d = (tt < I->t1 ? tt : I->t1) - I->t0;
			for (i = 0; i < X.nl; i++)
				if (I->mask >> i & 1 && I->kind != I_BARRIER)
					busy += d;
		}
		snprintf(buf, sizeof buf, "time %6.1f / %g   lanes busy %3.0f%%", tt,
		    X.T, tt > 0 ? 100 * busy / (X.nl * tt) : 100.0);
		text(4, D.H - 26, buf, C_TEXT);
	}
	if (cur >= 0)
		snprintf(buf, sizeof buf, "%s", X.in[cur].text);
	else
		snprintf(buf, sizeof buf, t >= X.T ? "done" : "start");
	buf[(D.W - 8) / FONT_6X10_W] = 0;
	text(4, D.H - 13, buf, cur < 0 ? C_DONE : C_ARROW);
}

/*
 * The shared back end: X.in[] holds the executed instructions with their
 * costs, masks and per-lane values; this times them, prints the serial
 * form and renders the animation.
 */
static void
xr_finish(int fps, double seconds, const char *name, int nofile)
{
	int i, k, nframes, hold, fr, listw, gridx, top, h1, h2;
	double t = 0, per, keep = 0, busy = 0;
	struct gif g;

	for (k = 0; k < X.nins; k++) {
		X.in[k].t0 = t;
		t += X.in[k].cost;
		X.in[k].t1 = t;
	}
	X.T = t;
	printf("  serial form:\n");
	for (k = 0; k < X.nins; k++) {
		const struct ins *I = &X.in[k];
		char lt[96];
		int pc = 0;
		lanes_text(lt, sizeof lt, I->mask);
		for (i = 0; i < X.nl; i++)
			pc += I->mask >> i & 1;
		if (I->cost > 0 && I->kind != I_BARRIER) {
			busy += I->cost * pc;
			if ((I->kind == I_OP || I->kind == I_ATOMIC) && pc < X.nl)
				keep += I->cost * (X.nl - pc);
		}
		if (X.nins > 200 && k == 160) {
			printf("    ... %d more\n", X.nins - 180);
			k = X.nins - 21;
			continue;
		}
		printf("    %-44s %-10s %5g  t = %g..%g%s%s%s", I->text,
		    I->cls >= 0 ? cclass_name[I->cls] : "", I->cost, I->t0, I->t1,
		    pc < X.nl ? "  lanes {" : "", pc < X.nl ? lt : "",
		    pc < X.nl ? "}" : "");
		if (I->dst[0] && X.nl <= 8) {   /* each lane's value */
			printf("  ->");
			for (i = 0; i < X.nl; i++)
				printf(" %s", I->mask >> i & 1 ? I->val[i] : "-");
		}
		printf("\n");
	}
	printf("  time %g for every lane; lanes busy %.0f%% (%g lane-units "
	    "idle: masked, or waiting for an atomic)\n", X.T, X.T > 0 ?
	    100 * busy / (X.nl * X.T) : 100.0, keep);
	if (X.T <= 0) {
		printf("  nothing to execute\n");
		return;
	}
	listw = 300;
	{
		int maxt = 0;
		for (k = 0; k < X.nins; k++)
			if ((int)strlen(X.in[k].text) > maxt)
				maxt = (int)strlen(X.in[k].text);
		if (maxt * FONT_6X10_W + 60 > listw)
			listw = maxt * FONT_6X10_W + 60;
		if (listw > 520)
			listw = 520;
	}
	top = 36;
	gridx = 8 + listw + 16;
	D.W = gridx + X.cols * (LBW + 8) + 4;
	h1 = (X.nins < LISTMAX ? X.nins : LISTMAX) * (FONT_6X10_H + 3);
	h2 = X.rows * (LBH + 8);
	D.H = top + (h1 > h2 ? h1 : h2) + 70;
	D.px = xmalloc((size_t)D.W * D.H);
	nframes = (int)(fps * seconds + 0.5);
	hold = (int)(fps * 1.5 + 0.5);
	per = X.T / (nframes - 1);
	gif_begin(&g, D.W, D.H, pal, 16);
	for (fr = 0; fr < nframes + hold; fr++) {
		double tt = fr < nframes ? fr * per : X.T;
		int d = (int)((fr + 1) * 100.0 / fps + 0.5) -
		    (int)(fr * 100.0 / fps + 0.5);
		xdraw(tt, 8, listw, gridx, top);
		gif_frame(&g, D.px, d);
	}
	gif_end(&g);
	free(D.px);
	printf("  animation: %dx%d, %d frames at %d fps (%.1f s, then the result "
	    "for 1.5 s), 1 s = %.1f cost units; %zu bytes\n", D.W, D.H,
	    nframes + hold, fps, seconds, X.T / seconds, g.out.n);
	deliver(&g, name, nofile);
	free(g.out.p);
}

/* ---- the API spirv.c uses ---- */

void
xr_begin(int lanes, const char *title)
{
	memset(&X, 0, sizeof X);
	X.nl = lanes;
	X.m = model_active();
	X.cols = lanes == 4 ? 2 : lanes == 8 ? 4 : lanes == 16 ? 4 : 8;
	X.rows = (lanes + X.cols - 1) / X.cols;
	snprintf(X.title, sizeof X.title, "%s", title);
}

int
xr_add(int kind, const char *txt, int cls, double cost, uint32_t mask,
    const char *dst)
{
	struct ins *I = xadd(kind, 0, cost, cls, mask);

	snprintf(I->text, sizeof I->text, "%s", txt);
	snprintf(I->dst, sizeof I->dst, "%s", dst ? dst : "");
	return X.nins - 1;
}

void
xr_val(int k, int lane, const char *v)
{
	snprintf(X.in[k].val[lane], sizeof X.in[k].val[lane], "%s", v);
}

void
xr_lanein(int lane, const char *v)
{
	snprintf(X.lanein[lane], sizeof X.lanein[lane], "%s", v);
}

int
xr_count(void)
{
	return X.nins;
}

void
xr_done(int fps, double seconds, const char *name, int nofile)
{
	xr_finish(fps, seconds, name, nofile);
}

static void
animate_exec(struct func *f, int lanes, int fps, double seconds, int random,
    uint64_t seed, const char *name, int nofile)
{
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];
	nid root = f->body;
	int i, j, k;
	uint64_t r = seed * 0x9e3779b97f4a7c15ULL + 1;

	memset(&X, 0, sizeof X);
	X.f = f;
	X.nl = lanes;
	X.m = model_active();
	X.cols = lanes == 4 ? 2 : lanes == 8 ? 4 : lanes == 16 ? 4 : 8;
	X.rows = (lanes + X.cols - 1) / X.cols;
	/* each lane's inputs */
	memset(in_i, 0, sizeof in_i);
	memset(in_r, 0, sizeof in_r);
	for (i = 0; i < lanes; i++)
		for (j = 0; j < f->np; j++) {
			int64_t lo, hi;
			r ^= r << 13;
			r ^= r >> 7;
			r ^= r << 17;
			if (TY_FP(f->p[j].ty)) {
				double v = random ? (double)(r % 2001) / 100.0 - 10 :
				    (double)(i + j);
				/* a ranged real (an approximation's domain): spread over it */
				if (f->p[j].ranged && f->p[j].rlo < f->p[j].rhi) {
					double lo_ = f->p[j].rlo, w_ = f->p[j].rhi - lo_;
					v = random ? lo_ + w_ * (double)(r % 10001) / 10000.0 :
					    lo_ + w_ * (double)i / (double)(lanes > 1 ?
					    lanes - 1 : 1);
				}
				X.vr[j][i] = in_r[(size_t)j * CH + i] = v;
			} else {
				int64_t v = random ? (int64_t)(r % 32) : i + j;
				param_bounds(&f->p[j], &lo, &hi);
				if (v < lo)
					v = lo;
				if (v > hi)
					v = hi;
				X.vi[j][i] = in_i[(size_t)j * CH + i] = v;
			}
		}
	plan_build(&X.p, &root, 1, 0);
	plan_run(&X.p, (uint32_t)lanes, in_i, in_r);
	xemit(root, lanes == 32 ? 0xffffffffu : (1u << lanes) - 1);
	/* each instruction's register and every lane's value, as text */
	for (k = 0; k < X.nins; k++) {
		struct ins *I = &X.in[k];
		if (I->reg < 0)
			continue;
		snprintf(I->dst, sizeof I->dst, "t%d", I->reg);
		for (i = 0; i < lanes; i++)
			xvalue(I->val[i], sizeof I->val[i], I->n, i);
	}
	for (i = 0; i < lanes; i++) {
		int at = 0;
		for (j = 0; j < f->np && j < 3; j++) {
			struct sbuf b = {0};
			fmt_val(&b, f->p[j].ty, X.vi[j][i], X.vr[j][i]);
			at += snprintf(X.lanein[i] + at, sizeof X.lanein[i] - (size_t)at,
			    "%s%s=%s", j ? " " : "", sym_str(f->p[j].name), b.p);
			free(b.p);
		}
	}
	snprintf(X.title, sizeof X.title, "exec of %s on %d lanes (%s), cost "
	    "model %s", node_str(root), X.nl, X.nl == 4 ? "a 2x2 quad" :
	    X.nl == 32 ? "a warp" : "lockstep", sym_str(X.m->name));
	/* report */
	printf("exec of %s, cost model %s, %d lanes in lockstep", node_str(root),
	    sym_str(X.m->name), lanes);
	for (j = 0; j < f->np; j++)
		printf("%s %s: %s", j ? "," : ";", sym_str(f->p[j].name),
		    ty_str(f->p[j].ty));
	printf("\n");
	if (random)
		printf("  random inputs (seed %llu)\n", (unsigned long long)seed);
	printf("  result per lane:");
	for (i = 0; i < lanes; i++) {
		char v[48];
		xvalue(v, sizeof v, root, i);
		printf(" %s", v);
	}
	printf("\n");
	xr_finish(fps, seconds, name, nofile);
	plan_free(&X.p);
}

/* the GIF: a file on the command line, an attachment in the bot */
static void
deliver(struct gif *gp, const char *name, int nofile)
{
	struct gif g = *gp;
	struct sbuf out = {0};

	if (nofile) {
		uint32_t hsh = 2166136261u;
		size_t q;
		for (q = 0; q < g.out.n; q++)
			hsh = (hsh ^ (unsigned char)g.out.p[q]) * 16777619u;
		printf("  (nofile: GIF checksum %08x)\n", hsh);
	} else if (nlp_defer) {
		if (g.out.n > MAXGIF) {
			printf("  the GIF is too large for the chat (%zu bytes): ask "
			    "for fewer seconds, a lower fps or a smaller n\n", g.out.n);
		} else {
			b64(&out, (const unsigned char *)g.out.p, g.out.n);
			printf("\x1e" "file %s.gif image/gif %zu\n%s\x1e" "end\n", name,
			    g.out.n, out.p);
		}
	} else {
		char path[80];
		FILE *fp;
		snprintf(path, sizeof path, "%s.gif", name);
		if ((fp = fopen(path, "wb")) == NULL ||
		    fwrite(g.out.p, 1, g.out.n, fp) != g.out.n) {
			if (fp)
				fclose(fp);
			fail("cannot write %s", path);
		}
		fclose(fp);
		printf("  wrote %s\n", path);
	}
	free(out.p);
}

void
cmd_animate(struct P *p)
{
	char w[64], name[64] = "scan", unk[256] = "";
	int alg = AL_HILLIS, n = 8, fps = 30, random = 0, nvals = 0, k, i;
	int last_n = 0;   /* the previous word was "n" or "size" */
	double seconds = 6, num;
	uint64_t seed = 1;
	int64_t vals[MAXN];
	struct model *m = model_active(), *saved = model_active();
	struct gif g;
	int nframes, hold, f, nofile = 0, exec = 0, lanes = 4, seeded = 0;
	int spirv = 0;
	struct func *xf = NULL;
	double per;

	S.fp = 0;
	nspv_set = 0;
	while (!cmdp_eos(p)) {
		if (cmdp_at(p, T_NUM) || cmdp_at(p, T_MINUS)) {
			num = cmdp_number(p, stopw);
			if (last_n) {   /* n 500, n = 8: the number is n */
				n = (int)num;
				last_n = 0;
				continue;
			}
			if (cmdp_at(p, T_ID)) {
				cmdp_word(p, w, sizeof w);
				for (k = 0; w[k]; k++)
					w[k] = (char)tolower((unsigned char)w[k]);
				if (!strcmp(w, "fps") || !strcmp(w, "frames")) {
					fps = (int)num;
					continue;
				}
				if (!strcmp(w, "seconds") || !strcmp(w, "second") ||
				    !strcmp(w, "s")) {
					seconds = num;
					continue;
				}
				if (!strcmp(w, "elements") || !strcmp(w, "element") ||
				    !strcmp(w, "items") || !strcmp(w, "values")) {
					n = (int)num;
					continue;
				}
				/* a bare number is n; the word is read as usual */
				n = (int)num;
				goto word;
			}
			n = (int)num;
			continue;
		}
		if (cmdp_tok(p, T_PCT)) {
			/* "%19 = 2.5" or "%x = lane": a SPIR-V input's value */
			char id[40];
			int64_t iv;
			if (cmdp_int(p, &iv))
				snprintf(id, sizeof id, "%lld", (long long)iv);
			else if (!cmdp_word(p, id, sizeof id))
				fail("animate: %%name = value");
			if (!cmdp_tok(p, T_ASSIGN))
				fail("animate: %%%s = value", id);
			if (nspv_set < 16) {
				struct spvset *e = &spv_set[nspv_set++];
				snprintf(e->name, sizeof e->name, "%s", id);
				if (cmdp_at(p, T_ID)) {
					cmdp_word(p, w, sizeof w);
					if (strcmp(w, "lane") && strcmp(w, "index"))
						fail("animate: %%%s = a number or lane", id);
					e->mode = 1;
				} else {
					e->mode = 4;
					e->v = cmdp_number(p, stopw);
				}
			}
			continue;
		}
		if (cmdp_tok(p, T_ASSIGN) || cmdp_tok(p, T_COMMA) ||
		    cmdp_tok(p, T_COLON) || cmdp_tok(p, T_MINUS))
			continue;
		if (!cmdp_at(p, T_ID))
			fail("animate: unexpected input (try: animate scan n 8 fps 30)");
		cmdp_word(p, w, sizeof w);
		for (k = 0; w[k]; k++)
			w[k] = (char)tolower((unsigned char)w[k]);
word:
		if (!strcmp(w, "exec") || !strcmp(w, "execute") ||
		    !strcmp(w, "execution") || !strcmp(w, "run")) {
			static const char *const exstop[] = {"at", "fps", "lanes",
			    "lane", "using", "with", "random", "seed", "seconds",
			    "to", "nofile", "on", "quad", "warp", "under", "for", NULL};
			ty_t ty = 0;
			int sign = 0, bits = 0, fpw = 0, asked_ty = 0;
			exec = 1;
			if (cmdp_at(p, T_ID)) {   /* "of", "for" */
				uint32_t pos = cmdp_pos(p);
				cmdp_word(p, w, sizeof w);
				if (strcmp(w, "of") && strcmp(w, "for"))
					cmdp_seek(p, pos);
			}
			/*
			 * "animate exec of best found cos [-pi/2, pi/2] with crit +
			 * cheby poly": find the approximation first (through the
			 * English translator and approx), then run the best one.
			 */
			if (cmdp_at(p, T_ID)) {
				static const char *const apw[] = {"best", "found",
				    "approximation", "approximate", "approx", "cheapest",
				    "fastest", "cheby", "chebyshev", "minimax", "taylor",
				    NULL};
				/* animate's own options end the request; "with" does not */
				static const char *const apstop[] = {"at", "fps", "lanes",
				    "lane", "random", "seed", "seconds", "nofile", "quad",
				    "warp", "to", NULL};
				char rest[512], req[600], *r = rest;
				uint32_t pos = cmdp_pos(p);
				int approx_req = 0;
				size_t q;
				if (cmdp_rest(p, rest, sizeof rest, apstop)) {
					char lw[512];
					
					for (q = 0; rest[q] && q + 1 < sizeof lw; q++)
						lw[q] = (char)tolower((unsigned char)rest[q]);
					lw[q] = 0;
					for (q = 0; apw[q]; q++)
						if (strstr(lw, apw[q]))
							approx_req = 1;
				}
				if (approx_req) {
					struct sbuf st = {0};
					struct func *best;
					static const char *const drop[] = {"the ", "best ",
					    "found ", "approximation of ", "approximation ",
					    "approx of ", "approx ", "approximate ", "a ", "an ",
					    NULL};
					int again = 1;
					while (again) {   /* "best found", "the best" ... */
						again = 0;
						for (q = 0; drop[q]; q++) {
							size_t dl = strlen(drop[q]);
							if (!strncasecmp(r, drop[q], dl)) {
								r += dl;
								again = 1;
							}
						}
					}
					snprintf(req, sizeof req, "approximate %s", r);
					printf("  note: finding the approximation first: %s\n",
					    req);
					notes_flush();
					if (!nlp_translate(req, &st)) {
						free(st.p);
						if (nlp_asked)
							return;
						fail("animate exec: could not read \"%s\" as an "
						    "approximation request", r);
					}
					{
						char *l = st.p, *nl;
						while (*l && (nl = strchr(l, '\n')) != NULL) {
							printf("  ucas: %.*s\n", (int)(nl - l), l);
							l = nl + 1;
						}
					}
					run_source(st.p, "<english>", 0);
					free(st.p);
					if (nlp_asked)
						return;
					if ((best = func_find(sym_cstr("ap1"))) == NULL)
						fail("animate exec: the approximation found no "
						    "candidate");
					printf("  animating the best candidate, ap1 = %s\n",
					    node_str(best->body));
					xf = best;
					continue;
				}
				cmdp_seek(p, pos);
			}
			/*
			 * "exec of this thing", "of the attached shader", "of the
			 * code above": the code block (the bot adds attachments and
			 * the replied-to message's code as blocks)
			 */
			{
				static const char *const deict[] = {"this", "that",
				    "these", "those", "it", "thing", "things", "stuff",
				    "code", "shader", "snippet", "file", "attachment",
				    "attached", "above", "below", "here", "one", "the",
				    "my", "following", "program", "kernel", "spirv",
				    "embedded", "embed", "posted", NULL};
				int nd = 0;
				while (cmdp_at(p, T_ID) && in_words_at(p, deict)) {
					uint32_t pos = cmdp_pos(p);
					cmdp_word(p, w, sizeof w);
					/* a function named "it" (approx's target) is itself */
					if (code_block == NULL && func_find(sym_cstr(w)) !=
					    NULL) {
						cmdp_seek(p, pos);
						break;
					}
					nd++;
				}
				if (nd && code_block == NULL && (cmdp_eos(p) ||
				    (cmdp_at(p, T_ID) && in_words_at(p, exstop))))
					fail("animate exec: I see no code to run here: attach "
					    "the file (.spv, .spvasm, .comp/.frag GLSL), paste "
					    "it in a ``` block, or reply to the message that "
					    "has it");
			}
			/* nothing after it but a code block: run the SPIR-V */
			if (code_block != NULL && (cmdp_eos(p) || cmdp_at(p, T_PCT) ||
			    cmdp_at(p, T_NUM) || (cmdp_at(p, T_ID) &&
			    in_words_at(p, exstop)))) {
				spirv = 1;
				continue;
			}
			/* the type: "32-bit", "signed 16-bit", "u32", "float" */
			for (;;) {
				uint32_t pos = cmdp_pos(p);
				int64_t v;
				ty_t t2;
				if (cmdp_int(p, &v)) {
					cmdp_tok(p, T_MINUS);
					if (cmdp_word(p, w, sizeof w) && (!strcmp(w, "bit") ||
					    !strcmp(w, "bits") || !strcmp(w, "wide"))) {
						bits = (int)v;
						continue;
					}
					cmdp_seek(p, pos);
					break;
				}
				if (!cmdp_word(p, w, sizeof w))
					break;
				for (k = 0; w[k]; k++)
					w[k] = (char)tolower((unsigned char)w[k]);
				if (!strcmp(w, "signed"))
					sign = 1;
				else if (!strcmp(w, "unsigned"))
					sign = 2;
				else if (!strcmp(w, "float") || !strcmp(w, "floats"))
					fpw = 32;
				else if (!strcmp(w, "double") || !strcmp(w, "doubles"))
					fpw = 64;
				else if (!strcmp(w, "integer") || !strcmp(w, "integers") ||
				    !strcmp(w, "int") || !strcmp(w, "ints"))
					;
				else if (ty_parse(w, strlen(w), &t2) && strcmp(w, "real"))
					ty = t2;
				else {
					cmdp_seek(p, pos);
					break;
				}
			}
			{
				/* a defined function by name keeps its parameters' types */
				uint32_t pos = cmdp_pos(p);
				if (!ty && cmdp_word(p, w, sizeof w) &&
				    (xf = func_find(sym_cstr(w))) != NULL &&
				    (cmdp_eos(p) || (cmdp_at(p, T_ID) &&
				    in_words_at(p, exstop))))
					continue;
				xf = NULL;
				cmdp_seek(p, pos);
			}
			if (!ty && !fpw && !bits && !sign) {
				/* no type: GPUs have no unbounded integers; ask */
				static const char *const opts[] = {
				    "u32 (32-bit unsigned, the GPU's int)",
				    "i32 (32-bit signed)", "f32 (float)",
				    "as written (mathematical int or real)"};
				static const ty_t tys[] = {TY(K_UINT, 32), TY(K_SINT, 32),
				    TY(K_FLOAT, 32), 0};
				int k2 = 0;
				{
					/* real functions or decimals: f32 is the default */
					char rest[512];
					uint32_t pos = cmdp_pos(p);
					static const char *const realw[] = {"sin", "cos", "tan",
					    "exp", "log", "sqrt", "pow", "atan", ".", "pi",
					    NULL};
					if (cmdp_rest(p, rest, sizeof rest, exstop)) {
						int q;
						for (q = 0; realw[q]; q++)
							if (strstr(rest, realw[q]))
								k2 = 2;
					}
					cmdp_seek(p, pos);
				}
				if (nlp_can_ask()) {
					k2 = nlp_ask_choice("animate exec: no type given for the "
					    "variables; which", opts, 4, k2);
					if (nlp_asked)
						return;
				} else
					note("no type given: every variable is %s (say "
					    "\"32-bit\", \"signed 32-bit\", \"f32\" ...)",
					    k2 == 2 ? "f32, as the expression is real" :
					    "u32, a GPU's integer");
				ty = tys[k2];
				asked_ty = 1;
			}
			if (!ty && fpw)
				ty = TY(K_FLOAT, bits ? bits : fpw);
			if (!ty && bits) {
				if (bits < 1 || bits > 64)
					fail("animate exec: %d-bit is not a type", bits);
				ty = TY(sign == 1 ? K_SINT : K_UINT, bits);
				if (sign == 0)
					note("%d-bit: every variable is u%d (say \"signed "
					    "%d-bit\" for i%d)", bits, bits, bits, bits);
				else
					note("%d-bit: every variable is %s", bits, ty_str(ty));
			} else if (ty && !asked_ty)
				note("every variable is %s", ty_str(ty));
			xf = cmdp_func_typed(p, exstop, ty);
			continue;
		}
		if (!strcmp(w, "lanes") || !strcmp(w, "lane")) {
			cmdp_tok(p, T_ASSIGN);
			if (cmdp_at(p, T_NUM))
				lanes = (int)cmdp_number(p, stopw);
			continue;
		}
		if (!strcmp(w, "quad")) {
			lanes = 4;
			continue;
		}
		if (!strcmp(w, "warp")) {
			lanes = 32;
			continue;
		}
		if (!strcmp(w, "n") || !strcmp(w, "size") || !strcmp(w, "length")) {
			last_n = 1;
			continue;
		}
		if (!strcmp(w, "fps")) {   /* fps = 30 */
			cmdp_tok(p, T_ASSIGN);
			if (cmdp_at(p, T_NUM))
				fps = (int)cmdp_number(p, stopw);
			continue;
		}
		if (!strcmp(w, "seconds")) {
			cmdp_tok(p, T_ASSIGN);
			if (cmdp_at(p, T_NUM))
				seconds = cmdp_number(p, stopw);
			continue;
		}
		if (!strcmp(w, "hillis") || !strcmp(w, "steele")) {
			alg = AL_HILLIS;
			continue;
		}
		if (!strcmp(w, "blelloch") || !strcmp(w, "efficient")) {
			alg = AL_BLELLOCH;
			continue;
		}
		if (!strcmp(w, "work")) {
			continue;
		}
		if (!strcmp(w, "sequential") || !strcmp(w, "serial") ||
		    !strcmp(w, "naive")) {
			alg = AL_SEQ;
			continue;
		}
		if (!strcmp(w, "random") || !strcmp(w, "randomly")) {
			random = 1;
			continue;
		}
		if (!strcmp(w, "seed")) {
			cmdp_tok(p, T_ASSIGN);
			seed = (uint64_t)cmdp_number(p, stopw);
			random = 1;
			seeded = 1;
			continue;
		}
		if (!strcmp(w, "values")) {
			cmdp_tok(p, T_ASSIGN);
			nvals = 0;
			while (cmdp_at(p, T_NUM) || cmdp_at(p, T_MINUS)) {
				if (nvals < MAXN)
					vals[nvals++] = (int64_t)cmdp_number(p, stopw);
				if (!cmdp_tok(p, T_COMMA))
					break;
			}
			continue;
		}
		if (!strcmp(w, "f32") || !strcmp(w, "float")) {
			S.fp = 1;
			continue;
		}
		if (!strcmp(w, "u32") || !strcmp(w, "int")) {
			S.fp = 0;
			continue;
		}
		if (!strcmp(w, "nofile")) {   /* report only (tests) */
			nofile = 1;
			continue;
		}
		if (!strcmp(w, "to")) {
			if (!cmdp_word(p, name, sizeof name))
				fail("animate ... to NAME writes NAME.gif");
			continue;
		}
		{
			struct model *mm = model_find(sym_cstr(w));
			if (mm != NULL) {
				m = mm;
				continue;
			}
		}
		if (in_words(w, filler))
			continue;
		if (strlen(unk) + strlen(w) + 3 < sizeof unk)
			snprintf(unk + strlen(unk), sizeof unk - strlen(unk), "%s%s",
			    unk[0] ? ", " : "", w);
	}
	if (unk[0])
		note("animate: ignored %s", unk);
	if (random && !seeded)   /* a new sample each time; the seed is shown */
		seed = time_seed();
	if (exec && !strcmp(name, "scan"))
		snprintf(name, sizeof name, "exec");
	if (exec && spirv && !seeded)   /* buffers: a new sample each run */
		seed = time_seed();
	if (exec && spirv) {
		if (lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
			fail("animate exec: lanes 4 (a 2x2 quad), 8, 16 or 32 (a warp)");
		notes_flush();
		model_set_active(m);
		spirv_exec(code_block, lanes, fps, seconds, random, seed, name,
		    nofile);
		model_set_active(saved);
		return;
	}
	if (exec) {
		if (xf == NULL)
			fail("animate exec of EXPR: which expression?");
		if (lanes != 4 && lanes != 8 && lanes != 16 && lanes != 32)
			fail("animate exec: lanes 4 (a 2x2 quad), 8, 16 or 32 (a warp)");
		if (fps < 1 || fps > 60)
			fail("animate: fps from 1 to 60");
		if (seconds < 1 || seconds > 20)
			fail("animate: seconds from 1 to 20");
		notes_flush();
		model_set_active(m);
		animate_exec(xf, lanes, fps, seconds, random, seed, name, nofile);
		model_set_active(saved);
		return;
	}
	if (nvals)
		n = nvals;
	if (n < 2)
		fail("animate: n must be at least 2");
	if (n > MAXN) {
		note("n = %d is over the cap: using %d", n, MAXN);
		n = MAXN;
	}
	if (fps < 1 || fps > 60)
		fail("animate: fps from 1 to 60");
	if (seconds < 1 || seconds > 20)
		fail("animate: seconds from 1 to 20");
	notes_flush();

	S.n = n;
	if (nvals)
		for (i = 0; i < n; i++)
			S.in[i] = vals[i];
	else if (random) {
		uint64_t r = seed * 0x9e3779b97f4a7c15ULL + 1;
		for (i = 0; i < n; i++) {
			r ^= r << 13;
			r ^= r >> 7;
			r ^= r << 17;
			S.in[i] = (int64_t)(r % 10);
		}
	} else
		for (i = 0; i < n; i++)
			S.in[i] = i + 1;

	model_set_active(m);
	costs(m);
	/* the three algorithms' totals, for comparison */
	printf("prefix sum of %d %s values (%s), cost model %s\n", n,
	    S.fp ? "f32" : "u32", nvals ? "given" : random ? "random" : "1..n",
	    sym_str(m->name));
	if (random && !nvals)
		printf("  random values (seed %llu):", (unsigned long long)seed);
	else
		printf("  values:");
	for (i = 0; i < n && i < 24; i++)
		printf(" %lld", (long long)S.in[i]);
	printf("%s\n", n > 24 ? " ..." : "");
	printf("  per element: load %g, add %g, barrier %g, divergence %g\n",
	    S.cl, S.ca, S.cb, S.cbr + S.cdv);
	printf("    %-14s %6s %8s %8s\n", "algorithm", "levels", "time", "work");
	for (k = 0; k < 3; k++) {
		build(k);
		printf("    %-14s %6d %8.0f %8.0f%s\n", alname[k], S.nlv, S.T, S.work,
		    k == alg ? "   <- animated" : "");
	}
	build(alg);
	printf("  %s levels:\n", alname[alg]);
	if (alg != AL_SEQ)
		for (k = 1; k <= S.nlv; k++) {
			const struct level *L = &S.lv[k];
			printf("    %-8s %3d ops  lane %4g%s%s  t = %g..%g\n", L->label,
			    L->nops, L->lane, L->diverged ? " +diverge" : "",
			    k < S.nlv ? " +barrier" : "", L->t0, L->t1);
		}
	else
		printf("    %d steps of load + add + store (the running sum stays "
		    "in a register), no barriers\n", S.nlv);
	printf("  result:");
	for (i = 0; i < S.cells && i < 24; i++)
		printf(" %lld", (long long)S.lv[S.nlv].v[i]);
	printf("%s\n", S.cells > 24 ? " ..." : "");

	/* layout */
	D.cw = S.cells <= 16 ? 36 : S.cells <= 32 ? 22 : S.cells <= 64 ? 11 : 7;
	D.text = D.cw >= 22;
	D.ch = 16;
	D.gap = 24;
	D.left = 46;
	D.top = 22;
	D.W = D.left + S.cells * D.cw + 10;
	if (D.W < 420)
		D.W = 420;
	D.H = D.top + (S.alg == AL_SEQ ? 2 : S.nlv + 1) * (D.ch + D.gap) + 60;
	D.vmax = 1;
	for (k = 0; k <= S.nlv; k++)
		for (i = 0; i < S.cells; i++)
			if (S.lv[k].v[i] > D.vmax)
				D.vmax = S.lv[k].v[i];
	D.px = xmalloc((size_t)D.W * D.H);

	/* frames: the run lasts `seconds`, then the result holds 1.5 s */
	nframes = (int)(fps * seconds + 0.5);
	hold = (int)(fps * 1.5 + 0.5);
	per = S.T / (nframes - 1);
	gif_begin(&g, D.W, D.H, pal, 16);
	for (f = 0; f < nframes + hold; f++) {
		double t = f < nframes ? f * per : S.T;
		/* GIF delays are whole 1/100 s: alternate to average 1/fps */
		int d = (int)((f + 1) * 100.0 / fps + 0.5) - (int)(f * 100.0 / fps +
		    0.5);
		draw_frame(t);
		gif_frame(&g, D.px, d);
	}
	gif_end(&g);
	free(D.px);
	model_set_active(saved);
	printf("  animation: %dx%d, %d frames at %d fps (%.1f s, then the result "
	    "for 1.5 s), 1 s = %.1f cost units; %zu bytes\n", D.W, D.H,
	    nframes + hold, fps, seconds, S.T / seconds, g.out.n);
	deliver(&g, name, nofile);
	free(g.out.p);
}
