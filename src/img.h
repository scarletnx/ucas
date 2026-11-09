/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Images for animate exec: files (img.c, stb_image), textures and their
 * sampling (tex.c), and the choices a request makes for a shader's image
 * inputs (struct imgopts, filled by anim.c's options, the English
 * translator, the JSON frontend and the bot's attachments).
 */
#ifndef UCAS_IMG_H
#define UCAS_IMG_H

#include <stddef.h>
#include <stdint.h>

/* ---- img.c: image files ---- */
#define IMG_MAXDIM 16384		/* width or height */
#define IMG_MAXPIX (24u << 20)		/* decoded pixels (w * h) */
#define IMG_MAXFILE (12u << 20)		/* encoded bytes */

struct imgdec {
	int w, h;
	int channels;		/* in the file: 1 gray, 2 gray + alpha, 3, 4 */
	char format[8];		/* "PNG", "JPEG", "GIF", "BMP", "PNM" */
	unsigned char *rgba;	/* w * h * 4 bytes, 8-bit RGBA, top row first */
};

const char *img_sniff(const unsigned char *p, size_t n);  /* NULL: none */
/* 1, or 0 with a one-line reason in err */
int img_decode(const unsigned char *p, size_t n, struct imgdec *out,
    char *err, size_t errn);
void img_free(struct imgdec *im);

/* ---- tex.c: formats ---- */
enum { TC_UNORM, TC_SNORM, TC_FLOAT, TC_UINT, TC_SINT, TC_SRGB };
struct texfmt {
	const char *name;	/* ucas's name: "RGBA8", "R16F", "R32UI" */
	const char *spv;	/* SPIR-V's ImageFormat: "Rgba8", "" if none */
	unsigned char nc, cls, b[4];	/* channels, TC_, bits per channel */
};
extern const struct texfmt tex_fmts[];
extern const int tex_nfmts;
int tex_fmt_find(const char *name);	/* either name, any case; -1 */
int tex_fmt_rgba8(void);

/* ---- tex.c: textures ---- */
#define TEX_MAXLV 15
#define TEX_MAXSIDE 1024		/* an input larger is box-filtered down */
enum { TD_1D, TD_2D, TD_3D, TD_CUBE, TD_RECT, TD_BUFFER };

struct tex {
	int w, h, d, layers;	/* level 0; a cube has 6 layers per element */
	int levels, fmt, dim;
	int lw[TEX_MAXLV], lh[TEX_MAXLV], ld[TEX_MAXLV];
	size_t lo[TEX_MAXLV];	/* each level's first texel in t */
	uint32_t *t;		/* 4 words per texel: float bits or integers */
	size_t ntexel;
};

void tex_alloc(struct tex *t, int dim, int w, int h, int d, int layers,
    int fmt, int mips);
void tex_free(struct tex *t);
void tex_copy(struct tex *dst, const struct tex *src);
/* level 0 (every layer) from 8-bit RGBA, converted to the format */
void tex_from_rgba8(struct tex *t, const unsigned char *rgba, int w, int h);
void tex_mips(struct tex *t);	/* levels 1.. by a 2x2(x2) box filter */
/* a texel as the shader sees it (missing channels 0, 0, 0, 1; sRGB
 * decoded); ints are exact in the doubles */
void tex_load(const struct tex *t, int lv, int x, int y, int z, int layer,
    double v[4]);
/* a write: quantized to the format (unorm clamped and rounded, half
 * floats rounded to nearest even, integers truncated to their width) */
void tex_store(struct tex *t, int lv, int x, int y, int z, int layer,
    const double v[4]);
uint32_t *tex_word(struct tex *t, int lv, int x, int y, int z, int layer);
/* 8-bit RGBA of level 0, layer 0, slice 0 (for display); unorm/sRGB as
 * they are, floats clamped, integers scaled to their width */
void tex_to_rgba8(const struct tex *t, unsigned char *out);

/* ---- tex.c: sampling ---- */
enum { SF_NEAREST, SF_LINEAR };
enum { SA_REPEAT, SA_MIRROR, SA_CLAMP, SA_BORDER, SA_MIRROR_CLAMP };
enum { SB_TRANSPARENT, SB_BLACK, SB_WHITE };

struct samp {
	unsigned char mag, min, mip;	/* SF_ */
	unsigned char addr[3];		/* SA_ per axis */
	unsigned char border;		/* SB_ */
	unsigned char unnorm;		/* unnormalized coordinates */
	unsigned char cmpop;		/* Dref: 0 never .. 7 always, as Vulkan */
	double bias, minlod, maxlod;
};
void samp_default(struct samp *s);

/* what the sampler computes for one lane (tex_sample) */
struct texreq {
	double c[4];		/* coordinates (s, t, r) and the array layer */
	int nc;			/* coordinates before the layer (1, 2, 3) */
	int arrayed;
	double lod;		/* the level of detail, already selected */
	int off[3];		/* ConstOffset/Offset, in texels */
	int dref;		/* compare against ref */
	double ref;
	int gather;		/* -1, or the component gathered */
};
/* lambda from the derivatives of the normalized coordinates */
double tex_lambda(const struct tex *t, int nc, const double *dx,
    const double *dy);
void tex_sample(const struct tex *t, const struct samp *s,
    const struct texreq *r, double out[4]);
int tex_level_size(const struct tex *t, int lv, int *w, int *h, int *d);

/* ---- tex.c: what fills an input ---- */
enum { IK_DEFAULT, IK_NOISE, IK_COLOR, IK_GRADIENT, IK_CHECKER, IK_FILE,
	IK_GALLERY };
int img_color(const char *name, unsigned char rgba[4]);  /* name, #hex, rgba() */
const char *img_color_name(const unsigned char rgba[4]);  /* NULL if none */
/* w * h * 4 bytes of a pattern */
void img_pattern(unsigned char *rgba, int w, int h, int kind,
    const unsigned char col[4], uint64_t seed);
/* box-filter 8-bit RGBA from (w, h) to (nw, nh) */
void img_downscale(const unsigned char *src, int w, int h, unsigned char *dst,
    int nw, int nh);

/* ---- the request's choices ---- */
#define IMG_MAXIN 16
#define IMG_MAXGAL 8
struct imgin {			/* one input slot */
	int kind;		/* IK_ */
	unsigned char rgba[4];	/* IK_COLOR */
	int fmt;		/* -1: RGBA8 */
	char file[200];		/* IK_FILE: a path */
	int w, h;		/* 0: 64 x 64 (or the gallery image's size) */
	int filt, addr;		/* -1: the sampler's defaults */
};
struct imggal {			/* an image the front end handed over */
	char name[96];
	unsigned char *data;
	size_t n;
};
struct imgopts {
	int any;		/* an image option was given */
	struct imgin in[IMG_MAXIN];
	int all;		/* in[0] applies to every input */
	int outfmt;		/* -1: RGBA8 */
	int w, h;		/* the output's size, 0: the first input's */
	int filt, mipf, addr, border;	/* the default sampler, -1 unset */
	int use_gallery;	/* "use my images": the gallery, asked for */
	struct imggal gal[IMG_MAXGAL];
	int ngal;
};
extern struct imgopts img_opt;
void img_opt_reset(void);	/* keeps the gallery */
void img_gallery_clear(void);
int img_gallery_add(const char *name, const unsigned char *data, size_t n);
/* "noise", "black", "#ff0000", "rgba(1,0,0,1)", "gradient", "checker",
 * a format ("R8") or a filter/address word: into *in; 0 if not one */
int img_in_word(struct imgin *in, const char *w);
/* base64 text (blanks skipped) to bytes; the count */
size_t img_b64(const char *s, size_t n, unsigned char *out);

#endif
