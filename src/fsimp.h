/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * fsimp: the mathematics of float simplify (fsimp.c), for spirv.c.
 *
 * A float expression is one scalar function per lane: spirv.c reads each
 * component of a vector expression into its own graph (the lane's
 * constants), lanes with the same graph are analysed once.  Every claim a
 * rewrite makes is derived: IEEE binary32 interval arithmetic over the
 * total order -inf < ... < -0 < +0 < ... < +inf (rounding is monotone, so
 * the image of an interval under a monotone chain is the interval of its
 * end points' images), monotonicity rules per operation, preimages of
 * monotone functions by bisection on that order, the standard model
 * fl(a op b) = (a op b)(1 + d), |d| <= u = 2^-24, and the Vulkan
 * precision table composed through each rewrite.  Nothing is sampled for
 * a proof; witnesses of a difference are constructed from the analysis
 * and then evaluated.
 */
#ifndef FSIMP_H
#define FSIMP_H

#include "ucas.h"

enum fxop {
	FX_IN, FX_K, FX_ADD, FX_SUB, FX_MUL, FX_DIV, FX_FMA, FX_NEG, FX_POW,
	FX_EXP2, FX_LOG2, FX_SQRT, FX_ISQRT, FX_STEP, FX_MIX, FX_SELGE,
	FX_MIN, FX_MAX, FX_ABS, FX_CVT, FX_OPQ
};

struct fxn {
	int op, a, b, c, d;	/* operands (node indices), -1; SELGE:
				 * !(a < b) ? c : d */
	float k;		/* FX_K: the lane's value */
	int in;			/* FX_IN, FX_CVT, FX_OPQ: the input index */
	int ins;		/* the SPIR-V instruction (spirv.c), -1 */
	int sid;		/* the SPIR-V id of the (vector) value */
	int uses;		/* uses of that id in the module */
};

#define FX_MAXN 256
#define FX_MAXIN 8

/* an input: a float load (with its assumed range), or an integer's
 * conversion (the integer type's range) */
struct fxin {
	char name[40];
	float lo, hi;		/* in the total order (-0 < +0) */
	int nan;		/* NaN possible */
	int assumed;
	int cvt;		/* float(an integer) */
	int sid, lane;		/* where it came from (spirv.c) */
};

struct fxg {
	struct fxn n[FX_MAXN];
	int nn;
	struct fxin in[FX_MAXIN];
	int nin;
	int preserve;		/* SignedZeroInfNanPreserve in the module */
	const char *srcline;	/* the GLSL line of the root, or NULL */
	/*
	 * The sink the root ends in: one step of its format, absolute
	 * (sink) or relative to the value (sinkrel, for values of at least
	 * sinkmin in magnitude); 0: none.  fx_rewrites fills the rest: the
	 * budget in the value's units, how much of it the accepted rewrites
	 * use, and why nothing was taken.
	 */
	double sink, sinkrel, sinkmin;
	char sinkname[48];
	double sinkbudget, sinkdev;
	char sinknote[240];
};

/* an interval of f32 values in the total order, plus NaN / undefined */
struct fiv {
	float lo, hi;
	int empty, nan, undef;
};

enum {
	FXR_SELECT = 1,	/* mix(a, b, step(e, x)) -> x >= e (or NaN) ? b : a */
	FXR_RECIP,	/* x / d -> x * fl(1/d) */
	FXR_DIVNOTE,	/* spirv-opt already did x / d -> x * fl(1/d) */
	FXR_FMA,	/* (x + A) * K -> fma(x, K1, K2) */
	FXR_FMA2,	/* x * K + B -> fma(x, K, B) */
	FXR_POWEXP,	/* pow(q, y) -> exp2(y * log2(q)) */
	FXR_SQRT,	/* sqrt(x) -> x * inversesqrt(x) */
	FXR_POLY,	/* pow(q, y) (or exp2, log2, sqrt, inversesqrt of q)
			 * ~ P(q - c), an approximation */
	FXR_LOGDOM	/* a product of powers -> exp2 of a sum of logs */
};

#define FX_MAXF 8

#define FX_MAXK 16

struct fxrw {
	int kind;
	int node;		/* the node replaced */
	int x, e, a, b;		/* operands, by kind */
	float k[FX_MAXK];	/* new constants for this lane */
	int nk;
	int exact;		/* 1 bit-exact (outside exc), 0 not */
	int vulkan;		/* 1 within the original's Vulkan precision */
	int refine;		/* 1 a refinement (defined where the original is) */
	int approx;		/* an approximation */
	int dep;		/* valid only with rewrite rw[dep] applied, -1 */
	int deg;		/* FXR_POLY: degree (k[0] the centre, k[1..]
				 * the coefficients from the highest) */
	double ulp;		/* derived bound against the reference, -1 */
	double gain;		/* per component, under the model */
	double rel;		/* FXR_POLY: certified relative error (0: none,
				 * the function is 0 inside the range) */
	double abserr;		/* FXR_POLY: certified absolute error, or 0
				 * (pow's: relative only) */
	double cost;		/* of the replacement, per component */
	char why[80];
	char exc[400];		/* where it is not bit-exact (and why) */
	char wit[300];		/* a witness: input, both values */
	char fail[300];		/* why it is not within precision */
	int sink;		/* accepted by the root's sink budget */
	double sinkdev;		/* the root's deviation with it (and the rest) */
	char sinkfail[200];	/* why the budget refuses it */
	/* FXR_LOGDOM: factor j is base fa[j] (its log2 taken) to the power
	 * fy[j] (a node, or -1 for the constant k[j]), or, when fe[j], the
	 * exponent fa[j] of an exp2 (no log); k[FX_MAXF] the log2 of the
	 * constant factors */
	int nf, fa[FX_MAXF], fy[FX_MAXF];
	unsigned char fe[FX_MAXF];
	struct sbuf proof;	/* the derivation, lines */
};

struct fxopt {
	int mode;		/* 0 within Vulkan precision, 1 exact, 2 ulp N */
	double ulp;
	int approx;		/* list approximations; apply with ulp */
};

/* the reference value (IEEE binary32; pow, exp2, log2 correctly
 * rounded through long double); *undef when GLSL leaves it undefined */
float fx_eval(const struct fxg *g, int node, const float *x, int *undef);
struct fiv fx_range(const struct fxg *g, int node, const struct fiv *dom);
/* the rewrites of the graph's nodes under root, at most max */
int fx_rewrites(struct fxg *g, int root, const struct fxopt *o,
    struct fxrw *rw, int max);
int fx_accept(const struct fxrw *r, const struct fxopt *o);
void fx_free(struct fxrw *rw, int n);
/* the operations a rewrite of a whole (a refitted polynomial, a product
 * in the log domain) replaces, its root among them, marked in
 * in[FX_MAXN]; 0 for any other rewrite */
int fx_inner(const struct fxg *g, const struct fxrw *r, unsigned char *in);
/* "%.9g" with -0, inf, nan spelt */
const char *fx_str(float v, char *b, size_t n);
/* the per-component cost of a node's operation as lowered */
double fx_price(int op);

#endif
