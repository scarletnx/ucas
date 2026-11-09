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
		int lx = x0 + tlw / 2 - 12 * FONT_6X10_W;
		lx = text(lx, ty + 12, "compute", C_BAR) + FONT_6X10_W;
		lx = text(lx, ty + 12, "diverge", C_RED) + FONT_6X10_W;
		text(lx, ty + 12, "barrier", C_BAR_ORANGE);
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
	"input", "array", "fps", "frame", "rate", "nofile", NULL
};

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
	int nframes, hold, f, nofile = 0;
	struct sbuf out = {0};
	double per;

	S.fp = 0;
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
		if (cmdp_tok(p, T_ASSIGN) || cmdp_tok(p, T_COMMA) ||
		    cmdp_tok(p, T_COLON) || cmdp_tok(p, T_MINUS))
			continue;
		if (!cmdp_at(p, T_ID))
			fail("animate: unexpected input (try: animate scan n 8 fps 30)");
		cmdp_word(p, w, sizeof w);
		for (k = 0; w[k]; k++)
			w[k] = (char)tolower((unsigned char)w[k]);
word:
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
	free(g.out.p);
}
