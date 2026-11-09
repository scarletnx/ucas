/*
 * viz: integrals and solids of revolution drawn (viz.c), on plot.c's
 * renderer, which plot.c exports here (the pr_ functions).
 */
#ifndef UCAS_VIZ_H
#define UCAS_VIZ_H

#include "ucas.h"

/* ---- plot.c: the canvas, the palette, curves, text, axes, 3-D ---- */

/* the fixed colours, for pr_ink */
enum {
	PR_BG, PR_TEXT, PR_GRID, PR_AXIS, PR_FRAME, PR_DIM, PR_ZERO, PR_MARK,
	PR_PANE, PR_PANEGRID, PR_LEGBG, PR_LEGBORDER, PR_UNDEF, PR_ASYM, PR_WIRE
};

struct pr_ax {			/* a linear 2-D chart */
	double lo[2], hi[2];
	int L, T, R, B;		/* the plot rectangle, after pr_layout */
};

struct pr_v3 {			/* a view: azimuth, elevation, centre, scale */
	double az, el, cx, cy, S;
};

void pr_begin(int w, int h);	/* the palette and a cleared canvas */
void pr_end(void);
void pr_clear(void);		/* background, z-buffer, no clip */
unsigned char *pr_pixels(void);
float *pr_zbuf(void);
/* plot's palette into out; [*ramp0, *ramp0 + *nramp) may be redefined */
int pr_palette(unsigned char out[256][3], int *ramp0, int *nramp);
int pr_ink(int role);
int pr_ser(int s, int tint);	/* series colour s, tint 0 full .. 3 faint */

void pr_clip(int x0, int y0, int x1, int y1);
void pr_put(int x, int y, int c);
int pr_get(int x, int y);
void pr_fill(int x, int y, int w, int h, int c);
void pr_rect(int x, int y, int w, int h, int c);
void pr_line1(int x0, int y0, int x1, int y1, int c);
int pr_text(int x, int y, const char *s, int c);
int pr_textw(const char *s);
void pr_text_bg(int x, int y, const char *s, int c, int bg);
void pr_ascii(char *dst, size_t n, const char *s);
void pr_title(int cx, int y, const char *s, int maxw);
/* anti-aliased: coverage first (pr_seg, pr_disc), then inked */
void pr_seg(double x0, double y0, double x1, double y1, double w);
void pr_disc(double x, double y, double r);
void pr_ink_ser(int s);
void pr_ink_solid(int c);
void pr_ink_tints(const int *t);	/* t[0] full coverage .. t[3] faint */

const char *pr_g4(double v);	/* %.4g */
const char *pr_gpi(double v);	/* 3.142 (pi) */
int pr_ticks(double lo, double hi, int maxn, double *v, char (*lab)[20]);

void pr_layout(struct pr_ax *a, int x0, int y0, int w, int h, int small);
void pr_axes(const struct pr_ax *a, const char *xn, const char *yn,
    int small);
/* style: 0 line, 1 dot, 2 swatch, 3 thin line; col: a series */
void pr_legend(int n, const char (*lab)[80], const int *col,
    const int *style, const struct pr_ax *a);

void pr_proj(const struct pr_v3 *V, double X, double Y, double Z,
    double *sx, double *sy, double *dep);
/* a filled triangle; with zt depth-tested (smaller dep is nearer) */
void pr_tri(const double *x, const double *y, const double *z, int col,
    int zt);

void pr_eval1(const struct func *f, nid body, int n, const double *xs,
    double *ys, uint8_t *ok);
void pr_auto_range(const struct func *f, double *lo, double *hi, char *why,
    size_t n);
int pr_vrange(double *v, int n, double *lo, double *hi, double *vmin,
    double *vmax);

/* ---- viz.c: the commands ---- */

struct P;
/* area, revolve, volume, surfacearea, arclength, centroid */
int viz_command(struct P *p);

#endif
