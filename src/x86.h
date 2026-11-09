/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * x86.h - x86-64 cost models, the intrinsics catalogue, an llvm-mca-like
 * throughput estimate and the vectoriser (agent F2 of cfront.md).
 *
 *   x86data.c  µarchs, instruction classes and their costs per µarch and
 *              vector width (latency, reciprocal throughput, uops, ports),
 *              the `model` declarations derived from them
 *   x86mca.c   lowering a function to x86 instruction classes and a
 *              cycle-level port/latency simulation (`mca F`)
 *   x86intr.c  the intrinsics catalogue: exact lane semantics in ucas
 *              nodes, feature flags, classes; differential tests with cc
 *   x86.c      commands, the prelude hook, x86 gate bases, idioms
 */
#ifndef UCAS_X86_H
#define UCAS_X86_H

#include "ucas.h"

/* ---- ISA features ---- */
enum {
	XF_SSE2, XF_SSE3, XF_SSSE3, XF_SSE41, XF_SSE42, XF_POPCNT, XF_LZCNT,
	XF_BMI1, XF_BMI2, XF_AVX, XF_AVX2, XF_FMA, XF_F16C, XF_PCLMUL, XF_AES,
	XF_ADX, XF_MOVBE, XF_GFNI, XF_AVX512F, XF_AVX512BW, XF_AVX512DQ,
	XF_AVX512VL, XF_AVX512CD, XF_AVX512VBMI, XF_AVX512VPOPCNTDQ,
	XF_AVX512BITALG, XF_VPCLMUL, XF_VAES,
	XF_N
};
typedef uint64_t xfeat;
#define XB(f) (1ULL << (f))
extern const char *const xfeat_name[XF_N];	/* "sse4.1": the -m flag */
extern const char *const xfeat_macro[XF_N];	/* "__SSE4_1__" */
int xfeat_find(const char *s);
void xfeat_text(struct sbuf *b, xfeat f);	/* "avx2 fma bmi2" */

/* ---- µarchs ---- */
enum { UA_SKL, UA_SKX, UA_GLC, UA_RPL, UA_ZEN4, UA_V2, UA_V3, UA_V4, UA_N };
#define MAXPORT 12
struct uarch {
	const char *name, *alias, *desc;
	xfeat feat;
	int width;		/* uops dispatched (renamed) per cycle */
	int rob;		/* reorder buffer entries */
	int sched;		/* scheduler entries (the issue window) */
	int mispredict;		/* branch misprediction penalty, cycles */
	int nport;
	const char *port[MAXPORT];
};
extern const struct uarch uarchs[UA_N];
int uarch_find(const char *s);			/* -1 if none */
int uarch_of_model(const struct model *m);	/* -1: not an x86 model */

/* ---- instruction classes: forms that cost the same ---- */
enum {
	XC_NONE,
	/* scalar integer */
	XC_ALU, XC_LEA, XC_SHIFT, XC_SHIFTX, XC_SHIFTCL, XC_IMUL, XC_MULX,
	XC_DIV32, XC_DIV64, XC_BITCNT, XC_BMI, XC_PDEP, XC_CMOV, XC_SETCC,
	XC_ADC, XC_BR, XC_LOAD, XC_STORE, XC_CRC32, XC_LOCK, XC_LIBM,
	XC_CVTI2F, XC_CVTF2I, XC_MOVX2G, XC_MOVG2X,
	/* vector integer */
	XC_VALU, XC_VALU01, XC_VSHIFT, XC_VSHIFTX, XC_VSHUF, XC_VPERM,
	XC_VBLEND, XC_VBLENDV, XC_VMUL, XC_VMULLD, XC_VMULLQ, XC_VSAD,
	XC_VPCMPQ, XC_VMOVMSK, XC_VTEST, XC_VEXTR, XC_VINSR, XC_VPMOVX,
	XC_VPMOVN, XC_VHADD, XC_VCMPK,
	/* vector float */
	XC_VFADD, XC_VFMUL, XC_VFMA, XC_VFDIVS, XC_VFDIVD, XC_VFSQRTS,
	XC_VFSQRTD, XC_VRCP, XC_VROUND, XC_VFMINMAX, XC_VFCMP, XC_VCVT,
	XC_VCVTW,
	/* memory, special units */
	XC_VGATHERD, XC_VGATHERQ, XC_VTERNLOG, XC_VPOPCNT, XC_VLZCNT,
	XC_VGF2P8, XC_PCLMUL, XC_AES, XC_VCOMPRESS, XC_KOP, XC_KMOV,
	XC_VMPSAD, XC_VLOAD, XC_VSTORE, XC_VMASKLD, XC_VMASKST, XC_VBCASTG,
	XC_N
};
extern const char *const xc_name[XC_N];		/* "vpaddd" style */
extern const char *const xc_desc[XC_N];		/* the forms in the class */
enum { VW_X, VW_Y, VW_Z };	/* scalar or xmm, ymm, zmm */
struct xcost {
	float lat, rtp;		/* latency; reciprocal throughput (cycles) */
	uint8_t uops;		/* fused-domain uops */
	uint8_t have;		/* 0: the µarch lacks it */
	uint16_t ports;		/* port mask (uarchs[].port names) */
	char src;		/* where the numbers come from (x86data.c) */
};
const struct xcost *xcost_get(int ua, int cls, int vw);
void xports_text(struct sbuf *b, int ua, uint16_t mask);
const char *xsrc_text(char src);

void x86_init(void);
void x86_prelude(void);		/* parse.c: after the prelude */
void cmd_xmodel(int ua);	/* `model zen4` */
void cmd_xmodels(void);

/* ---- mca: instruction sequences and the simulator ---- */
#define MCA_MAXDEP 3
struct mins {			/* one instruction of a block */
	uint8_t cls, vw;
	int32_t dep[MCA_MAXDEP];	/* producer index, -1 none */
	uint8_t carried[MCA_MAXDEP];	/* dep is from the previous iteration */
	char text[48];
};
struct mblock {
	struct mins *in;
	uint32_t n, cap;
	int loop;			/* the block repeats */
};
void mblock_add(struct mblock *b, int cls, int vw, const char *text,
    int32_t d0, int32_t d1, int32_t d2);
struct mres {
	double cycles;		/* block: total; loop: per iteration */
	double lat_bound;	/* critical path / recurrence per iteration */
	double port_bound;	/* busiest port */
	double fe_bound;	/* uops / dispatch width */
	double uops;
	double pressure[MAXPORT];
	int busiest;
	const char *limit;	/* "latency", "port N", "front end" */
};
void mca_run(const struct mblock *b, int ua, struct mres *r);
void mca_print(const struct mblock *b, int ua, const struct mres *r,
    const char *title);
void cmd_mca(struct func *f, int ua);
int x86_lower(struct func *f, int ua, struct mblock *outer,
    struct mblock *loop, double *trips);

/* ---- intrinsics ---- */
void x86_intr_init(void);
uint32_t xi_count(void);
int xi_find(const char *name);		/* -1 */
void cmd_intrinsic(const char *name, int ua);
void cmd_intrinsics(const char *filter, int ua);
void cmd_x86check(const char *filter, int npts, int avx512);
/*
 * One lane of an intrinsic's result as a node: ops[k](lane) gives the
 * k-th operand's lane (or the scalar), imm the immediate.
 */
typedef nid (*xi_lane_fn)(void *ctx, int opnd, int lane);
nid xi_lane(int id, int lane, int64_t imm, xi_lane_fn fn, void *ctx);
int xi_lanes(int id);			/* lanes of the result (1: scalar) */
ty_t xi_rtype(int id);
ty_t xi_otype(int id, int k);
int xi_nops(int id);
int xi_cls(int id);
int xi_vw(int id);
xfeat xi_feat(int id);
const char *xi_name(int id);
int xi_okind(int id, int k);

/* ---- parse.c hook ---- */
struct P;
int x86_command(struct P *p);

#endif
