/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * x86-64 µarchs and the cost of each instruction class on each.
 *
 * Numbers per (µarch, class, vector width): latency in cycles, reciprocal
 * throughput in cycles (the average time between two independent
 * instructions of the form when nothing else runs), fused-domain uops,
 * and the execution ports the uops can go to.  A class groups the forms
 * that cost the same on these cores (vpaddb/w/d/q, vpand, vpxor ... are
 * XC_VALU).  Each row says where its numbers come from:
 *
 *   U  uops.info measurements (Abel & Reineke, "uops.info: Characterizing
 *      Latency, Throughput, and Port Usage of Instructions on Intel
 *      Microarchitectures", ASPLOS 2019; tables for SKL, SKX, ADL-P, ZEN4)
 *   A  Agner Fog, "Instruction tables" (2022 edition), and his
 *      "microarchitecture" manual for front-end widths and penalties
 *   I  Intel 64 and IA-32 Architectures Optimization Reference Manual
 *      (vol. 1, 2023: Skylake, Golden Cove chapters)
 *   D  AMD Software Optimization Guide for the AMD Zen 4 Microarchitecture
 *      (pub. 57647, rev. 3.0): latency/throughput appendix
 *   M  measured on this host's Skylake-family core (an i5-9400, Coffee
 *      Lake) with rdtsc chains, `x86check bench` (README)
 *   E  an estimate: no published figure for this exact form; taken from
 *      the closest form and marked so
 *
 * The rows were transcribed by hand from those tables (not scraped), and
 * rounded where a form has a data-dependent range (div: the typical
 * case for 64-bit operands; gathers: all lanes active).  The generic
 * levels x86-64-v2/v3/v4 take, per class, the worst latency and the worst
 * throughput over the cores that implement the level (v2, v3: Skylake,
 * Golden Cove, Zen 4; v4: Skylake-X, Golden Cove, Zen 4), so code tuned
 * for them is never priced optimistically.
 *
 * Widths: a row for xmm (and scalar) forms; ymm and zmm rows only where
 * they differ.  Missing ymm = xmm.  Missing zmm on an AVX-512 core:
 * Intel runs 512-bit uops on port 0 (ports 0+1 fused) and port 5, so the
 * port set loses port 1 and the throughput scales by the ports lost;
 * Zen 4 double-pumps 512-bit uops through its 256-bit pipes (Zen 4 SOG
 * 2.11), so the zmm throughput is twice the ymm one.
 */
#include "x86.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

const char *const xfeat_name[XF_N] = {
	"sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "lzcnt",
	"bmi", "bmi2", "avx", "avx2", "fma", "f16c", "pclmul", "aes",
	"adx", "movbe", "gfni", "avx512f", "avx512bw", "avx512dq",
	"avx512vl", "avx512cd", "avx512vbmi", "avx512vpopcntdq",
	"avx512bitalg", "vpclmulqdq", "vaes",
};
const char *const xfeat_macro[XF_N] = {
	"__SSE2__", "__SSE3__", "__SSSE3__", "__SSE4_1__", "__SSE4_2__",
	"__POPCNT__", "__LZCNT__", "__BMI__", "__BMI2__", "__AVX__",
	"__AVX2__", "__FMA__", "__F16C__", "__PCLMUL__", "__AES__",
	"__ADX__", "__MOVBE__", "__GFNI__", "__AVX512F__", "__AVX512BW__",
	"__AVX512DQ__", "__AVX512VL__", "__AVX512CD__", "__AVX512VBMI__",
	"__AVX512VPOPCNTDQ__", "__AVX512BITALG__", "__VPCLMULQDQ__",
	"__VAES__",
};

int
xfeat_find(const char *s)
{
	int i;

	for (i = 0; i < XF_N; i++)
		if (!strcmp(xfeat_name[i], s))
			return i;
	if (!strcmp(s, "bmi1"))
		return XF_BMI1;
	if (!strcmp(s, "sse41"))
		return XF_SSE41;
	if (!strcmp(s, "sse42"))
		return XF_SSE42;
	return -1;
}

void
xfeat_text(struct sbuf *b, xfeat f)
{
	int i, n = 0;

	for (i = 0; i < XF_N; i++)
		if (f & XB(i))
			sb_printf(b, "%s%s", n++ ? " " : "", xfeat_name[i]);
	if (!n)
		sb_puts(b, "x86-64");
}

#define V2F (XB(XF_SSE2) | XB(XF_SSE3) | XB(XF_SSSE3) | XB(XF_SSE41) | \
	XB(XF_SSE42) | XB(XF_POPCNT))
#define V3F (V2F | XB(XF_AVX) | XB(XF_AVX2) | XB(XF_FMA) | XB(XF_F16C) | \
	XB(XF_BMI1) | XB(XF_BMI2) | XB(XF_LZCNT) | XB(XF_MOVBE))
#define V4F (V3F | XB(XF_AVX512F) | XB(XF_AVX512BW) | XB(XF_AVX512DQ) | \
	XB(XF_AVX512VL) | XB(XF_AVX512CD))
#define CRYPTO (XB(XF_PCLMUL) | XB(XF_AES) | XB(XF_ADX))
#define ICLX (XB(XF_AVX512VBMI) | XB(XF_AVX512VPOPCNTDQ) | \
	XB(XF_AVX512BITALG) | XB(XF_GFNI) | XB(XF_VPCLMUL) | XB(XF_VAES))

/*
 * Ports.  Intel: 0 1 5 6 (and 10 on Golden Cove) integer ALUs, 0 1 5
 * vector, 2 3 (11) loads, 4 (9) store data, 7 (8) store address.  Zen 4:
 * ALU0-3, AGU0-2, the vector pipes FP0-FP3 and FP45 (stores, moves to
 * and from integer registers) -- D, 2.11-2.12.
 */
const struct uarch uarchs[UA_N] = {
	{"skl", "skylake", "Intel Skylake/Kaby/Coffee Lake client core "
	    "(this host's i5-9400 is one): AVX2, no AVX-512",
	    V3F | CRYPTO, 4, 224, 97, 16, 8,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"}},
	{"skx", "skylake-x", "Intel Skylake-SP/X server core: AVX-512 "
	    "F/CD/BW/DQ/VL, two 512-bit FMA units (ports 0+1 fused, 5)",
	    V4F | CRYPTO, 4, 224, 97, 16, 8,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"}},
	{"glc", "goldencove", "Intel Golden Cove P-core as in Sapphire "
	    "Rapids: AVX-512 with VBMI, VPOPCNTDQ, BITALG, GFNI",
	    V4F | CRYPTO | ICLX, 6, 512, 160, 17, 12,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9",
	    "p10", "p11"}},
	{"rpl", "raptorlake", "Intel Raptor Lake / Alder Lake P-core "
	    "(Golden Cove/Raptor Cove): Golden Cove costs, AVX-512 fused "
	    "off; GFNI, VAES and VPCLMULQDQ on xmm/ymm",
	    V3F | CRYPTO | XB(XF_GFNI) | XB(XF_VPCLMUL) | XB(XF_VAES), 6, 512,
	    160, 17, 12,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9",
	    "p10", "p11"}},
	{"zen4", "znver4", "AMD Zen 4: AVX-512 (VBMI, VPOPCNTDQ, BITALG, "
	    "GFNI) on 256-bit pipes, 512-bit ops double-pumped",
	    V4F | CRYPTO | ICLX, 6, 320, 96, 13, 12,
	    {"ALU0", "ALU1", "ALU2", "ALU3", "AGU0", "AGU1", "AGU2", "FP0",
	    "FP1", "FP2", "FP3", "FP45"}},
	{"x86v2", "x86-64-v2", "generic x86-64-v2 (SSE4.2, POPCNT): the "
	    "worst of Skylake, Golden Cove and Zen 4 per class",
	    V2F, 4, 224, 97, 17, 8,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"}},
	{"x86v3", "x86-64-v3", "generic x86-64-v3 (AVX2, FMA, BMI1/2, "
	    "LZCNT, MOVBE): the worst of Skylake, Golden Cove and Zen 4",
	    V3F, 4, 224, 97, 17, 8,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"}},
	{"x86v4", "x86-64-v4", "generic x86-64-v4 (AVX-512 F/BW/CD/DQ/VL): "
	    "the worst of Skylake-X, Golden Cove and Zen 4",
	    V4F, 4, 224, 97, 17, 8,
	    {"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"}},
};

int
uarch_find(const char *s)
{
	int i;

	for (i = 0; i < UA_N; i++)
		if (!strcmp(uarchs[i].name, s) || !strcmp(uarchs[i].alias, s))
			return i;
	if (!strcmp(s, "coffeelake") || !strcmp(s, "kabylake"))
		return UA_SKL;
	if (!strcmp(s, "alderlake") || !strcmp(s, "adl"))
		return UA_RPL;
	if (!strcmp(s, "sapphirerapids") || !strcmp(s, "spr"))
		return UA_GLC;
	return -1;
}

int
uarch_of_model(const struct model *m)
{
	return m ? uarch_find(sym_str(m->name)) : -1;
}

const char *const xc_name[XC_N] = {
	"-",
	"add", "lea", "shl", "shlx", "shl cl", "imul", "mulx",
	"div r32", "div r64", "popcnt", "andn", "pext", "cmov", "setcc",
	"adc", "jcc", "load", "store", "crc32", "lock add", "call libm",
	"cvtsi2ss", "cvttss2si", "movd r,x", "movd x,r",
	"vpaddd", "vpminsd", "vpslld", "vpsllw x", "vpshufb", "vpermd",
	"vpblendd", "vpblendvb", "vpmullw", "vpmulld", "vpmullq", "vpsadbw",
	"vpcmpgtq", "vpmovmskb", "vptest", "vpextrd", "vpinsrd", "vpmovzx",
	"vpmovdb", "vphaddd", "vpcmpd k",
	"vaddps", "vmulps", "vfmadd", "vdivps", "vdivpd", "vsqrtps",
	"vsqrtpd", "vrcpps", "vroundps", "vminps", "vcmpps", "vcvtdq2ps",
	"vcvtps2pd",
	"vpgatherdd", "vpgatherqq", "vpternlogd", "vpopcntd", "vplzcntd",
	"gf2p8affineqb", "pclmulqdq", "aesenc", "vpcompressd", "kandw",
	"kmovw", "mpsadbw", "vmovdqu ld", "vmovdqu st", "vpmaskmovd ld",
	"vpmaskmovd st", "vpbroadcastd r",
};

const char *const xc_desc[XC_N] = {
	"no instruction",
	"add sub and or xor cmp test inc dec neg not mov r,r",
	"lea with base+index*scale+disp",
	"shl shr sar rol ror by an immediate",
	"shlx shrx sarx rorx (BMI2)",
	"shl shr sar by cl",
	"imul r, r (low half)",
	"mulx / mul r64 (the full 128-bit product)",
	"div idiv r32",
	"div idiv r64",
	"popcnt lzcnt tzcnt",
	"andn blsr blsi blsmsk bzhi",
	"pdep pext",
	"cmovcc",
	"setcc",
	"adc sbb adcx adox",
	"jcc, macro-fused cmp+jcc",
	"mov r, [m]",
	"mov [m], r",
	"crc32",
	"lock add/xadd/cmpxchg [m]",
	"a libm call (exp, log, sin ...), estimated",
	"cvtsi2ss cvtsi2sd",
	"cvttss2si cvttsd2si cvtss2si",
	"movd/movq r, xmm",
	"movd/movq xmm, r",
	"paddb/w/d/q psub pand pandn por pxor",
	"pcmpeq pcmpgt(b/w/d) pmin pmax pabs pavg padds paddus psign",
	"psll/psrl/psra by an immediate; vpsllv/vpsrlv/vpsrav",
	"psll/psrl/psra by a count in an xmm",
	"pshufb pshufd pshuflw punpck palignr pack pslldq shufps unpcklps "
	    "pblendw insertps movhlps",
	"vpermd vpermq vperm2i128 vpermb/w vpermt2 vinserti128 "
	    "vextracti128 vpbroadcast ymm",
	"vpblendd blendps blendpd vpblendm (k)",
	"pblendvb blendvps blendvpd",
	"pmullw pmulhw pmulhuw pmulhrsw pmuludq pmuldq pmaddwd pmaddubsw",
	"pmulld",
	"vpmullq (AVX512DQ)",
	"psadbw",
	"pcmpgtq",
	"pmovmskb movmskps movmskpd",
	"ptest vtestps",
	"pextrb/w/d/q",
	"pinsrb/w/d/q",
	"pmovzx pmovsx",
	"vpmovdb vpmovqd vpmovsdb (narrowing, AVX-512)",
	"phaddw phaddd phsub haddps",
	"vpcmp/vcmpps into a mask register",
	"addps subps addpd addss (and sd)",
	"mulps mulpd mulss",
	"vfmadd/vfmsub/vfnmadd 132/213/231",
	"divps divss",
	"divpd divsd",
	"sqrtps sqrtss",
	"sqrtpd sqrtsd",
	"rcpps rsqrtps vrcp14ps vrsqrt14ps (approximate)",
	"roundps roundpd vrndscaleps",
	"minps maxps minpd maxpd",
	"cmpps cmppd (into a vector)",
	"cvtdq2ps cvtps2dq cvttps2dq",
	"cvtps2pd cvtpd2ps cvtdq2pd (width changes)",
	"vpgatherdd vgatherdps (all lanes active)",
	"vpgatherqq vpgatherdq vgatherqpd",
	"vpternlogd/q",
	"vpopcntb/w/d/q",
	"vplzcntd/q (AVX512CD)",
	"gf2p8affineqb gf2p8affineinvqb gf2p8mulb",
	"pclmulqdq",
	"aesenc aesenclast aesdec aesdeclast",
	"vpcompressd/q vpexpandd/q (register)",
	"kand kor kxor knot kshift",
	"kmov between k and a general register",
	"mpsadbw",
	"vmovdqu/vmovups load",
	"vmovdqu/vmovups store",
	"vpmaskmovd vmaskmovps load",
	"vpmaskmovd vmaskmovps store",
	"vpbroadcastd from a general register (or movd + broadcast)",
};

struct xrow {
	uint8_t ua, cls, vw;
	float lat, rtp;
	uint8_t uops;
	const char *ports;
	char src;
};

#define S(c, l, t, u, p, s) {UA_SKL, XC_##c, VW_X, l, t, u, p, s},
#define SY(c, l, t, u, p, s) {UA_SKL, XC_##c, VW_Y, l, t, u, p, s},
#define K(c, l, t, u, p, s) {UA_SKX, XC_##c, VW_X, l, t, u, p, s},
#define KY(c, l, t, u, p, s) {UA_SKX, XC_##c, VW_Y, l, t, u, p, s},
#define KZ(c, l, t, u, p, s) {UA_SKX, XC_##c, VW_Z, l, t, u, p, s},
#define G(c, l, t, u, p, s) {UA_GLC, XC_##c, VW_X, l, t, u, p, s},
#define GY(c, l, t, u, p, s) {UA_GLC, XC_##c, VW_Y, l, t, u, p, s},
#define GZ(c, l, t, u, p, s) {UA_GLC, XC_##c, VW_Z, l, t, u, p, s},
#define Z(c, l, t, u, p, s) {UA_ZEN4, XC_##c, VW_X, l, t, u, p, s},
#define ZY(c, l, t, u, p, s) {UA_ZEN4, XC_##c, VW_Y, l, t, u, p, s},
#define ZZ(c, l, t, u, p, s) {UA_ZEN4, XC_##c, VW_Z, l, t, u, p, s},

/*
 * Port strings: one character per port index, '0'..'9', 'a' = 10,
 * 'b' = 11.  Zen 4: 0-3 ALU0-3, 4-6 AGU0-2, 7-a FP0-FP3, b FP45.
 */
static const struct xrow rows[] = {
	/* ---- Skylake client (SKL, this host) ---- */
	S(ALU, 1, 0.25, 1, "0156", 'M')
	S(LEA, 3, 1, 1, "1", 'U')
	S(SHIFT, 1, 0.5, 1, "06", 'U')
	S(SHIFTX, 1, 0.5, 1, "06", 'U')
	S(SHIFTCL, 2, 1.5, 3, "06", 'U')
	S(IMUL, 3, 1, 1, "1", 'M')
	S(MULX, 4, 1, 2, "15", 'U')
	S(DIV32, 26, 6, 10, "0156", 'U')
	S(DIV64, 36, 21, 36, "0156", 'M')	/* 35-88: by the dividend's size */
	S(BITCNT, 3, 1, 1, "1", 'M')
	S(BMI, 1, 0.5, 1, "15", 'U')
	S(PDEP, 3, 1, 1, "1", 'M')
	S(CMOV, 1, 0.5, 1, "06", 'U')
	S(SETCC, 1, 0.5, 1, "06", 'U')
	S(ADC, 1, 0.5, 1, "06", 'U')
	S(BR, 1, 0.5, 1, "06", 'U')
	S(LOAD, 5, 0.5, 1, "23", 'U')
	S(STORE, 1, 1, 1, "4", 'U')
	S(CRC32, 3, 1, 1, "1", 'U')
	S(LOCK, 18, 18, 9, "0156", 'A')
	S(LIBM, 20, 15, 20, "0156", 'E')
	S(CVTI2F, 5, 1, 2, "015", 'U')
	S(CVTF2I, 6, 1, 2, "015", 'U')
	S(MOVX2G, 2, 1, 1, "0", 'U')
	S(MOVG2X, 2, 1, 1, "5", 'U')
	S(VALU, 1, 0.33f, 1, "015", 'M')
	S(VALU01, 1, 0.5, 1, "01", 'U')
	S(VSHIFT, 1, 0.5, 1, "01", 'U')
	S(VSHIFTX, 2, 1, 2, "015", 'U')
	SY(VSHIFTX, 4, 1, 2, "015", 'U')
	S(VSHUF, 1, 1, 1, "5", 'M')	/* vpshufb ymm: 1 and 1.00 here */
	S(VPERM, 3, 1, 1, "5", 'U')
	S(VBLEND, 1, 0.33f, 1, "015", 'U')
	S(VBLENDV, 2, 1, 2, "015", 'U')
	S(VMUL, 5, 0.5, 1, "01", 'U')
	S(VMULLD, 10, 1, 2, "01", 'U')
	S(VSAD, 3, 1, 1, "5", 'U')
	S(VPCMPQ, 3, 1, 1, "5", 'U')
	S(VMOVMSK, 3, 1, 1, "0", 'U')
	S(VTEST, 3, 1, 2, "05", 'U')
	S(VEXTR, 3, 1, 2, "05", 'U')
	S(VINSR, 3, 2, 2, "5", 'U')
	S(VPMOVX, 1, 1, 1, "5", 'U')
	SY(VPMOVX, 3, 1, 1, "5", 'U')
	S(VHADD, 3, 2, 3, "015", 'U')
	S(VFADD, 4, 0.5, 1, "01", 'M')
	S(VFMUL, 4, 0.5, 1, "01", 'U')
	S(VFMA, 4, 0.5, 1, "01", 'M')
	S(VFDIVS, 11, 3, 1, "0", 'U')
	SY(VFDIVS, 11, 5, 1, "0", 'U')
	S(VFDIVD, 14, 4, 1, "0", 'U')
	SY(VFDIVD, 14, 8, 1, "0", 'U')
	S(VFSQRTS, 12, 3, 1, "0", 'U')
	SY(VFSQRTS, 12, 6, 1, "0", 'U')
	S(VFSQRTD, 16, 6, 1, "0", 'U')
	SY(VFSQRTD, 16, 12, 1, "0", 'U')
	S(VRCP, 4, 1, 1, "0", 'U')
	S(VROUND, 8, 1, 2, "01", 'U')
	S(VFMINMAX, 4, 0.5, 1, "01", 'U')
	S(VFCMP, 4, 0.5, 1, "01", 'U')
	S(VCVT, 4, 0.5, 1, "01", 'U')
	S(VCVTW, 5, 1, 2, "015", 'U')
	SY(VCVTW, 7, 1, 2, "015", 'U')
	S(VGATHERD, 20, 4, 4, "0523", 'U')
	SY(VGATHERD, 22, 5, 4, "0523", 'U')
	S(VGATHERQ, 20, 3, 4, "0523", 'U')
	SY(VGATHERQ, 22, 4, 4, "0523", 'U')
	S(PCLMUL, 7, 1, 1, "5", 'U')
	S(AES, 4, 1, 1, "0", 'U')
	S(VMPSAD, 4, 2, 2, "5", 'U')
	S(VLOAD, 6, 0.5, 1, "23", 'U')
	SY(VLOAD, 7, 0.5, 1, "23", 'U')
	S(VSTORE, 1, 1, 1, "4", 'U')
	S(VMASKLD, 8, 0.5, 2, "01523", 'U')
	S(VMASKST, 1, 1, 3, "0457", 'U')
	S(VBCASTG, 3, 1, 2, "5", 'U')

	/* ---- Skylake-X: SKL plus AVX-512 (U: SKX tables) ---- */
	K(VMULLQ, 15, 1.5, 3, "01", 'U')
	KZ(VMULLQ, 15, 1.5, 3, "0", 'U')
	K(VPMOVN, 4, 2, 2, "5", 'U')
	K(VCMPK, 3, 1, 1, "5", 'U')
	KZ(VFDIVS, 18, 10, 3, "05", 'U')
	KZ(VFDIVD, 23, 16, 3, "05", 'U')
	KZ(VFSQRTS, 19, 12, 3, "05", 'U')
	KZ(VFSQRTD, 25, 24, 3, "05", 'U')
	KZ(VRCP, 4, 2, 3, "05", 'U')
	KZ(VGATHERD, 24, 8, 4, "0523", 'U')
	KZ(VGATHERQ, 22, 5, 4, "0523", 'U')
	K(VTERNLOG, 1, 0.33f, 1, "015", 'U')
	KZ(VTERNLOG, 1, 0.5, 1, "05", 'U')
	K(VLZCNT, 4, 0.5, 1, "01", 'U')
	K(VCOMPRESS, 3, 2, 2, "5", 'U')
	K(KOP, 1, 1, 1, "0", 'U')
	K(KMOV, 3, 1, 1, "0", 'U')
	KZ(VLOAD, 8, 0.5, 1, "23", 'U')
	K(VBCASTG, 3, 1, 1, "5", 'U')

	/* ---- Golden Cove (U: ADL-P P-core tables; I: SPR chapter) ---- */
	G(ALU, 1, 0.2f, 1, "0156a", 'U')
	G(LEA, 2, 0.5, 1, "15", 'E')
	G(SHIFT, 1, 0.5, 1, "06", 'U')
	G(SHIFTX, 1, 0.5, 1, "06", 'U')
	G(SHIFTCL, 2, 1, 2, "06", 'U')
	G(IMUL, 3, 1, 1, "1", 'U')
	G(MULX, 4, 1, 2, "15", 'U')
	G(DIV32, 12, 6, 4, "0156", 'U')
	G(DIV64, 15, 10, 4, "0156", 'U')
	G(BITCNT, 3, 1, 1, "1", 'U')
	G(BMI, 1, 0.5, 1, "15", 'U')
	G(PDEP, 3, 1, 1, "1", 'U')
	G(CMOV, 1, 0.5, 1, "06", 'U')
	G(SETCC, 1, 0.5, 1, "06", 'U')
	G(ADC, 1, 0.5, 1, "06", 'U')
	G(BR, 1, 0.5, 1, "06", 'U')
	G(LOAD, 5, 0.33f, 1, "23b", 'U')
	G(STORE, 1, 0.5, 1, "49", 'U')
	G(CRC32, 3, 1, 1, "1", 'U')
	G(LOCK, 20, 20, 9, "0156", 'A')
	G(LIBM, 18, 12, 20, "0156a", 'E')
	G(CVTI2F, 5, 1, 2, "015", 'U')
	G(CVTF2I, 6, 1, 2, "015", 'U')
	G(MOVX2G, 3, 1, 1, "0", 'U')
	G(MOVG2X, 3, 1, 1, "5", 'U')
	G(VALU, 1, 0.33f, 1, "015", 'U')
	G(VALU01, 1, 0.5, 1, "01", 'U')
	G(VSHIFT, 1, 0.5, 1, "01", 'U')
	G(VSHIFTX, 2, 1, 2, "015", 'U')
	GY(VSHIFTX, 3, 1, 2, "015", 'U')
	G(VSHUF, 1, 0.5, 1, "15", 'U')
	GZ(VSHUF, 1, 1, 1, "5", 'U')
	G(VPERM, 3, 1, 1, "5", 'U')
	G(VBLEND, 1, 0.33f, 1, "015", 'U')
	G(VBLENDV, 3, 1, 3, "015", 'U')
	G(VMUL, 5, 0.5, 1, "01", 'U')
	G(VMULLD, 10, 1, 2, "01", 'U')
	G(VMULLQ, 15, 1.5, 3, "01", 'U')
	GZ(VMULLQ, 15, 1.5, 3, "0", 'U')
	G(VSAD, 3, 1, 1, "5", 'U')
	G(VPCMPQ, 3, 1, 1, "5", 'U')
	G(VMOVMSK, 3, 1, 1, "0", 'U')
	G(VTEST, 3, 1, 2, "05", 'U')
	G(VEXTR, 3, 1, 2, "05", 'U')
	G(VINSR, 3, 2, 2, "5", 'U')
	G(VPMOVX, 1, 0.5, 1, "15", 'U')
	GY(VPMOVX, 3, 1, 1, "5", 'U')
	G(VPMOVN, 4, 2, 2, "5", 'U')
	G(VHADD, 3, 2, 3, "015", 'U')
	G(VCMPK, 3, 1, 1, "5", 'U')
	/* the new FADD units on ports 1 and 5 (I: Golden Cove, "fast adder") */
	G(VFADD, 2, 0.5, 1, "15", 'U')
	GZ(VFADD, 4, 0.5, 1, "05", 'E')
	G(VFMUL, 4, 0.5, 1, "01", 'U')
	G(VFMA, 4, 0.5, 1, "01", 'U')
	G(VFDIVS, 11, 3, 1, "0", 'U')
	GY(VFDIVS, 11, 5, 1, "0", 'U')
	GZ(VFDIVS, 18, 10, 3, "05", 'E')
	G(VFDIVD, 13, 4, 1, "0", 'U')
	GY(VFDIVD, 13, 8, 1, "0", 'U')
	GZ(VFDIVD, 22, 16, 3, "05", 'E')
	G(VFSQRTS, 12, 3, 1, "0", 'U')
	GY(VFSQRTS, 12, 6, 1, "0", 'U')
	GZ(VFSQRTS, 19, 12, 3, "05", 'E')
	G(VFSQRTD, 16, 4.5f, 1, "0", 'U')
	GY(VFSQRTD, 16, 9, 1, "0", 'U')
	GZ(VFSQRTD, 23, 18, 3, "05", 'E')
	G(VRCP, 4, 1, 1, "0", 'U')
	G(VROUND, 8, 1, 2, "01", 'U')
	G(VFMINMAX, 4, 0.5, 1, "01", 'U')
	G(VFCMP, 4, 0.5, 1, "01", 'U')
	G(VCVT, 4, 0.5, 1, "01", 'U')
	G(VCVTW, 5, 1, 2, "015", 'U')
	GY(VCVTW, 7, 1, 2, "015", 'U')
	/* gathers before the 2023 GDS microcode (which makes them ~3x slower) */
	G(VGATHERD, 20, 3, 5, "0523b", 'U')
	GY(VGATHERD, 22, 5, 5, "0523b", 'U')
	GZ(VGATHERD, 24, 8, 5, "0523b", 'E')
	G(VGATHERQ, 20, 3, 5, "0523b", 'U')
	GY(VGATHERQ, 22, 4, 5, "0523b", 'U')
	G(VTERNLOG, 1, 0.33f, 1, "015", 'U')
	GZ(VTERNLOG, 1, 0.5, 1, "05", 'U')
	G(VPOPCNT, 3, 1, 1, "5", 'U')
	G(VLZCNT, 4, 0.5, 1, "01", 'U')
	G(VGF2P8, 5, 0.5, 1, "01", 'U')
	GZ(VGF2P8, 5, 1, 1, "0", 'U')
	G(PCLMUL, 3, 1, 1, "5", 'U')
	G(AES, 3, 0.5, 1, "01", 'U')
	G(VCOMPRESS, 3, 2, 2, "5", 'U')
	G(KOP, 1, 1, 1, "0", 'U')
	G(KMOV, 3, 1, 1, "0", 'U')
	G(VMPSAD, 4, 2, 2, "5", 'U')
	G(VLOAD, 6, 0.33f, 1, "23b", 'U')
	GY(VLOAD, 7, 0.33f, 1, "23b", 'U')
	GZ(VLOAD, 8, 0.5, 1, "23", 'U')
	G(VSTORE, 1, 0.5, 1, "49", 'U')
	GZ(VSTORE, 1, 1, 1, "49", 'U')
	G(VMASKLD, 8, 0.5, 2, "01523", 'U')
	G(VMASKST, 1, 1, 3, "0459", 'U')
	G(VBCASTG, 3, 1, 2, "5", 'U')

	/* ---- Zen 4 (D: Zen 4 SOG appendix; U: ZEN4 tables) ---- */
	Z(ALU, 1, 0.25, 1, "0123", 'D')
	Z(LEA, 2, 0.5, 1, "0123", 'D')
	Z(SHIFT, 1, 0.5, 1, "12", 'D')
	Z(SHIFTX, 1, 0.5, 1, "12", 'D')
	Z(SHIFTCL, 1, 0.5, 1, "12", 'D')
	Z(IMUL, 3, 1, 1, "1", 'D')
	Z(MULX, 4, 1, 2, "12", 'D')
	Z(DIV32, 12, 6, 2, "1", 'D')
	Z(DIV64, 14, 9, 2, "1", 'D')
	Z(BITCNT, 1, 0.25, 1, "0123", 'D')
	Z(BMI, 1, 0.25, 1, "0123", 'D')
	Z(PDEP, 3, 1, 1, "1", 'D')
	Z(CMOV, 1, 0.5, 1, "03", 'D')
	Z(SETCC, 1, 0.5, 1, "03", 'D')
	Z(ADC, 1, 0.5, 1, "03", 'D')
	Z(BR, 1, 0.5, 1, "03", 'D')
	Z(LOAD, 4, 0.33f, 1, "456", 'D')
	Z(STORE, 1, 0.5, 1, "56", 'D')
	Z(CRC32, 3, 1, 1, "1", 'D')
	Z(LOCK, 8, 8, 9, "0123", 'A')
	Z(LIBM, 18, 12, 20, "0123", 'E')
	Z(CVTI2F, 4, 1, 2, "9ab", 'D')
	Z(CVTF2I, 5, 1, 2, "9ab", 'D')
	Z(MOVX2G, 3, 1, 1, "b", 'D')
	Z(MOVG2X, 3, 1, 1, "b", 'D')
	Z(VALU, 1, 0.25, 1, "789a", 'D')
	Z(VALU01, 1, 0.25, 1, "789a", 'D')
	Z(VSHIFT, 1, 0.5, 1, "89", 'D')
	Z(VSHIFTX, 1, 0.5, 1, "89", 'D')
	Z(VSHUF, 1, 0.5, 1, "89", 'D')
	Z(VPERM, 3, 1, 1, "89", 'D')
	ZY(VPERM, 4, 1, 1, "89", 'D')
	ZZ(VPERM, 6, 2, 1, "89", 'D')
	Z(VBLEND, 1, 0.25, 1, "789a", 'D')
	Z(VBLENDV, 1, 0.5, 1, "78", 'D')
	Z(VMUL, 3, 0.5, 1, "7a", 'D')
	Z(VMULLD, 3, 0.5, 1, "7a", 'D')
	Z(VMULLQ, 3, 0.5, 1, "7a", 'D')
	Z(VSAD, 3, 0.5, 1, "7a", 'D')
	Z(VPCMPQ, 1, 0.25, 1, "789a", 'D')
	Z(VMOVMSK, 5, 1, 1, "b", 'U')
	Z(VTEST, 4, 1, 2, "7b", 'D')
	Z(VEXTR, 4, 1, 2, "8b", 'D')
	Z(VINSR, 4, 1, 2, "8b", 'D')
	Z(VPMOVX, 1, 0.5, 1, "89", 'D')
	ZY(VPMOVX, 4, 1, 1, "89", 'D')
	Z(VPMOVN, 4, 2, 2, "89", 'D')
	Z(VHADD, 2, 1, 3, "789a", 'D')
	Z(VCMPK, 3, 0.5, 1, "7a", 'D')
	Z(VFADD, 3, 0.5, 1, "9a", 'D')
	Z(VFMUL, 3, 0.5, 1, "78", 'D')
	Z(VFMA, 4, 0.5, 1, "78", 'D')
	Z(VFDIVS, 10, 3, 1, "8", 'D')
	ZY(VFDIVS, 10, 5, 1, "8", 'D')
	ZZ(VFDIVS, 11, 10, 1, "8", 'D')
	Z(VFDIVD, 13, 5, 1, "8", 'D')
	ZY(VFDIVD, 13, 8, 1, "8", 'D')
	ZZ(VFDIVD, 13, 16, 1, "8", 'D')
	Z(VFSQRTS, 15, 5, 1, "8", 'D')
	ZY(VFSQRTS, 15, 8, 1, "8", 'D')
	ZZ(VFSQRTS, 15, 16, 1, "8", 'D')
	Z(VFSQRTD, 21, 9, 1, "8", 'D')
	ZY(VFSQRTD, 21, 13, 1, "8", 'D')
	ZZ(VFSQRTD, 21, 26, 1, "8", 'D')
	Z(VRCP, 3, 0.5, 1, "78", 'D')
	Z(VROUND, 3, 0.5, 1, "9a", 'D')
	Z(VFMINMAX, 2, 0.5, 1, "78", 'D')
	Z(VFCMP, 1, 0.5, 1, "78", 'D')
	Z(VCVT, 3, 0.5, 1, "9a", 'D')
	Z(VCVTW, 4, 1, 2, "89a", 'D')
	/* gathers are microcoded on Zen 4: about a lane per cycle */
	Z(VGATHERD, 20, 4, 9, "456", 'U')
	ZY(VGATHERD, 22, 8, 17, "456", 'U')
	ZZ(VGATHERD, 25, 16, 33, "456", 'U')
	Z(VGATHERQ, 20, 3, 6, "456", 'U')
	ZY(VGATHERQ, 22, 5, 10, "456", 'U')
	ZZ(VGATHERQ, 25, 8, 18, "456", 'U')
	Z(VTERNLOG, 1, 0.25, 1, "789a", 'D')
	Z(VPOPCNT, 2, 0.5, 1, "78", 'D')
	Z(VLZCNT, 2, 0.5, 1, "78", 'D')
	Z(VGF2P8, 3, 0.5, 1, "78", 'D')
	Z(PCLMUL, 4, 1, 1, "78", 'D')
	Z(AES, 4, 0.5, 1, "78", 'D')
	Z(VCOMPRESS, 4, 2, 2, "89", 'D')
	Z(KOP, 1, 0.5, 1, "78", 'D')
	Z(KMOV, 3, 1, 1, "b", 'D')
	Z(VMPSAD, 4, 2, 3, "789a", 'D')
	Z(VLOAD, 7, 0.5, 1, "456", 'D')
	ZZ(VLOAD, 8, 1, 1, "456", 'D')
	Z(VSTORE, 1, 0.5, 1, "56", 'D')
	ZY(VSTORE, 1, 1, 1, "56", 'D')
	ZZ(VSTORE, 1, 2, 1, "56", 'D')
	Z(VMASKLD, 8, 0.5, 1, "456", 'D')
	Z(VMASKST, 1, 6, 4, "56", 'U')
	Z(VBCASTG, 4, 1, 2, "8b", 'D')
};

#undef S
#undef SY
#undef K
#undef KY
#undef KZ
#undef G
#undef GY
#undef GZ
#undef Z
#undef ZY
#undef ZZ

/* the cost tables, (µarch, class, width) */
static struct xcost X[UA_N][XC_N][3];
static int x86_ready;

const char *
xsrc_text(char src)
{
	switch (src) {
	case 'U': return "uops.info";
	case 'A': return "Agner Fog's tables";
	case 'I': return "Intel optimization manual";
	case 'D': return "AMD Zen 4 SOG";
	case 'M': return "measured on this host (SKL), agrees with uops.info";
	case 'E': return "estimate (closest documented form)";
	case 'G': return "generic: worst of the level's cores";
	case 'V': return "derived from the ymm form";
	}
	return "?";
}

static uint16_t
port_mask(const char *s)
{
	uint16_t m = 0;

	for (; *s; s++)
		m |= (uint16_t)(1u << (isdigit((unsigned char)*s) ? *s - '0' :
		    *s - 'a' + 10));
	return m;
}

static int
popc16(uint16_t m)
{
	int n = 0;

	for (; m; m &= m - 1)
		n++;
	return n;
}

/* Intel: 512-bit uops do not use port 1 (ports 0+1 fuse into one unit) */
static void
derive_zmm_intel(struct xcost *z, const struct xcost *y)
{
	uint16_t m = y->ports;
	int before = popc16(m), after;

	*z = *y;
	z->src = 'V';
	if ((m & 0x23) && (m & ~2u))	/* vector ports 0 1 5: drop 1 */
		m &= ~2u;
	else if (m == 2)
		m = 1;
	after = popc16(m);
	z->ports = m;
	if (after && after < before)
		z->rtp = y->rtp * (float)before / (float)after;
}

void
x86_init(void)
{
	size_t i;
	int ua, c, w, k;
	static const int members[3][3] = {
		{UA_SKL, UA_GLC, UA_ZEN4},	/* v2 */
		{UA_SKL, UA_GLC, UA_ZEN4},	/* v3 */
		{UA_SKX, UA_GLC, UA_ZEN4},	/* v4 */
	};

	if (x86_ready)
		return;
	x86_ready = 1;
	memset(X, 0, sizeof X);
	/* SKX and RPL start as copies of SKL and GLC */
	for (i = 0; i < sizeof rows / sizeof rows[0]; i++) {
		const struct xrow *r = &rows[i];
		int t[2], nt = 1, j;
		t[0] = r->ua;
		if (r->ua == UA_SKL)
			t[nt++] = UA_SKX;
		if (r->ua == UA_GLC)
			t[nt++] = UA_RPL;
		for (j = 0; j < nt; j++) {
			struct xcost *x = &X[t[j]][r->cls][r->vw];
			if (j > 0 && x->have)
				continue;
			x->lat = r->lat;
			x->rtp = r->rtp;
			x->uops = r->uops;
			x->ports = port_mask(r->ports);
			x->src = r->src;
			x->have = 1;
		}
	}
	/* SKX rows given after SKL ones override the copies */
	for (i = 0; i < sizeof rows / sizeof rows[0]; i++) {
		const struct xrow *r = &rows[i];
		struct xcost *x;
		if (r->ua != UA_SKX)
			continue;
		x = &X[UA_SKX][r->cls][r->vw];
		x->lat = r->lat;
		x->rtp = r->rtp;
		x->uops = r->uops;
		x->ports = port_mask(r->ports);
		x->src = r->src;
		x->have = 1;
	}
	for (ua = 0; ua < UA_V2; ua++) {
		int avx512 = (uarchs[ua].feat & XB(XF_AVX512F)) != 0;
		for (c = 1; c < XC_N; c++) {
			struct xcost *x = X[ua][c];
			if (!x[VW_X].have && !x[VW_Y].have && !x[VW_Z].have)
				continue;
			if (!x[VW_X].have && x[VW_Z].have && avx512) {
				/* an AVX-512-only class given at zmm: VL forms */
				x[VW_X] = x[VW_Z];
				x[VW_X].src = 'V';
			}
			if (!x[VW_Y].have && x[VW_X].have)
				x[VW_Y] = x[VW_X];
			if (!x[VW_Z].have && avx512) {
				if (ua == UA_ZEN4) {
					x[VW_Z] = x[VW_Y];
					x[VW_Z].rtp = x[VW_Y].rtp * 2;
					x[VW_Z].src = 'V';
				} else
					derive_zmm_intel(&x[VW_Z], &x[VW_Y]);
			}
			if (!avx512)
				x[VW_Z].have = 0;
		}
		/* RPL: no AVX-512 */
	}
	for (k = 0; k < XC_N; k++)
		for (w = 0; w < 3; w++)
			if (w == VW_Z || k == XC_VMULLQ || k == XC_VCMPK ||
			    k == XC_VTERNLOG || k == XC_VCOMPRESS || k == XC_KOP ||
			    k == XC_KMOV || k == XC_VPMOVN || k == XC_VLZCNT)
				X[UA_RPL][k][w].have = 0;
	X[UA_RPL][XC_VPOPCNT][0].have = X[UA_RPL][XC_VPOPCNT][1].have = 0;
	/* generic levels: the worst of the members, ports of the first */
	for (ua = UA_V2; ua <= UA_V4; ua++) {
		const int *mb = members[ua - UA_V2];
		for (c = 1; c < XC_N; c++)
			for (w = 0; w < 3; w++) {
				struct xcost *x = &X[ua][c][w];
				const struct xcost *b = &X[mb[0]][c][w];
				if (!b->have)
					continue;
				*x = *b;
				x->src = 'G';
				for (k = 1; k < 3; k++) {
					const struct xcost *o = &X[mb[k]][c][w];
					if (!o->have)
						continue;
					if (o->lat > x->lat)
						x->lat = o->lat;
					if (o->rtp > x->rtp)
						x->rtp = o->rtp;
				}
				if (ua == UA_V2 && w > VW_X)
					x->have = 0;
				if (ua == UA_V3 && w == VW_Z)
					x->have = 0;
			}
	}
}

const struct xcost *
xcost_get(int ua, int cls, int vw)
{
	x86_init();
	if (ua < 0 || ua >= UA_N || cls <= 0 || cls >= XC_N || vw < 0 || vw > 2)
		return NULL;
	return X[ua][cls][vw].have ? &X[ua][cls][vw] : NULL;
}

void
xports_text(struct sbuf *b, int ua, uint16_t mask)
{
	int p, n = 0;

	for (p = 0; p < uarchs[ua].nport; p++)
		if ((mask >> p) & 1) {
			const char *nm = uarchs[ua].port[p];
			/* p0 p1 p5 -> p015 */
			if (n && nm[0] == 'p' && nm[2] == 0 &&
			    uarchs[ua].port[0][0] == 'p')
				sb_puts(b, nm + 1);
			else
				sb_printf(b, "%s%s", n ? " " : "", nm);
			n++;
		}
	if (!n)
		sb_puts(b, "-");
}

/*
 * The `model` declaration of a µarch: ucas's per-class costs (README
 * "Cost models"), which `cost` and `find` use.  For a CPU the class cost
 * is the latency of the scalar form in cycles (expressions are chains of
 * dependent operations); `mca` gives throughput.  ftrans and fpow are
 * libm calls; diverge is half the misprediction penalty (a branch on
 * data that goes either way); barrier is an mfence.
 */
static double
lat_of(int ua, int cls)
{
	const struct xcost *c = xcost_get(ua, cls, VW_X);

	return c ? (double)c->lat : 1.0;
}

static void
model_text(struct sbuf *b, int ua)
{
#define L(cls) lat_of(ua, XC_##cls)
	sb_printf(b, "model %s {\n", uarchs[ua].name);
	sb_printf(b, "  iadd: %g, logic: %g, shift: %g, imul: %g, idiv: %g, "
	    "idivk: %g, cmp: %g,\n", L(ALU), L(ALU), L(SHIFT), L(IMUL),
	    L(DIV64), L(MULX) + L(SHIFT), L(ALU));
	sb_printf(b, "  select: %g, minmax: %g, bitcnt: %g, conv: %g, "
	    "bitcast: %g,\n", L(CMOV), L(ALU) + L(CMOV), L(BITCNT),
	    L(CVTI2F), L(MOVX2G));
	sb_printf(b, "  fadd: %g, fmul: %g, fdiv: %g, fsqrt: %g, ftrans: %g, "
	    "fpow: %g, f64: 1,\n", L(VFADD), L(VFMUL), L(VFDIVS), L(VFSQRTS),
	    L(LIBM), 3 * L(LIBM));
	sb_printf(b, "  load: %g, branch: 1, diverge: %g, barrier: 33,\n",
	    L(LOAD), (double)uarchs[ua].mispredict / 2);
	sb_printf(b, "  atomic_shared: %g, atomic_global: %g, store: 1", L(LOCK),
	    L(LOCK));
	/*
	 * VNNI (AVX-VNNI on rpl, AVX512-VNNI on glc and zen4): vpdpbusd
	 * multiplies unsigned bytes by signed bytes and adds the four
	 * products to a 32-bit accumulator (sudot4x8 with the operands
	 * swapped), vpdpwssd signed 16-bit pairs (sdot2x16); the ...s forms
	 * saturate (the *sat classes).  The scalar form is a movd to xmm,
	 * the dot and a movd back.  ESTIMATES: the dot's latency is the
	 * published one (uops.info: 5 cycles on Ice Lake and Golden Cove,
	 * 4 on Zen 4), not measured here.  No VNNI form multiplies
	 * unsigned by unsigned or signed by signed bytes before
	 * AVX-VNNI-INT8, which none of these has.
	 */
	{
		static const struct { const char *ua; double lat; } vnni[] = {
			{"glc", 5}, {"rpl", 5}, {"zen4", 4},
		};
		size_t i;
		for (i = 0; i < sizeof vnni / sizeof vnni[0]; i++)
			if (!strcmp(uarchs[ua].name, vnni[i].ua)) {
				double c = L(MOVG2X) + vnni[i].lat + L(MOVX2G);
				sb_printf(b, ",\n  sudot4x8: %g, sudot4x8sat: %g, "
				    "sdot2x16: %g, sdot2x16sat: %g", c, c, c, c);
			}
	}
	sb_puts(b, "\n}\n");
#undef L
}

void
x86_prelude(void)
{
	struct sbuf b = {0};
	int ua;

	x86_init();
	for (ua = 0; ua < UA_N; ua++)
		model_text(&b, ua);
	run_source(sb_str(&b), "<x86 prelude>", 0);
	free(b.p);
}

static void
row_text(struct sbuf *b, int ua, int c, int w)
{
	const struct xcost *x = xcost_get(ua, c, w);

	if (!x) {
		sb_puts(b, "       -          ");
		return;
	}
	sb_printf(b, "%4g %5g %2d ", (double)x->lat, (double)x->rtp, x->uops);
	{
		struct sbuf pb = {0};
		xports_text(&pb, ua, x->ports);
		sb_printf(b, "%-7s", sb_str(&pb));
		free(pb.p);
	}
}

void
cmd_xmodel(int ua)
{
	const struct uarch *u = &uarchs[ua];
	struct sbuf b = {0};
	int c, w;
	int zmm = (u->feat & XB(XF_AVX512F)) != 0;

	x86_init();
	printf("model %s (%s): %s\n", u->name, u->alias, u->desc);
	xfeat_text(&b, u->feat);
	printf("  features: %s\n", sb_str(&b));
	printf("  dispatch %d uops/cycle, ROB %d, scheduler %d, branch "
	    "mispredict %d cycles\n", u->width, u->rob, u->sched, u->mispredict);
	rep_begin("x86_model");
	rep_str("model", u->name);
	rep_str("alias", u->alias);
	rep_str("description", u->desc);
	rep_str("features", sb_str(&b));
	rep_int("dispatch_width", u->width);
	rep_int("rob", u->rob);
	rep_int("mispredict", u->mispredict);
	sb_clear(&b);
	printf("  ports:");
	for (c = 0; c < u->nport; c++)
		printf(" %s", u->port[c]);
	printf("\n");
	printf("  %-12s %-25s %-25s%s\n", "class", "xmm/scalar: lat rtp uops "
	    "ports", "ymm", zmm ? " zmm" : "");
	rep_list("classes");
	for (c = 1; c < XC_N; c++) {
		const struct xcost *x0 = xcost_get(ua, c, VW_X);
		const struct xcost *x1 = xcost_get(ua, c, VW_Y);
		const struct xcost *x2 = xcost_get(ua, c, VW_Z);
		if (!x0 && !x1 && !x2)
			continue;
		sb_clear(&b);
		for (w = 0; w < (zmm ? 3 : 2); w++) {
			if (w == VW_Y && c < XC_VALU)
				break;
			row_text(&b, ua, c, w);
			sb_puts(&b, "  ");
		}
		printf("  %-12s %s [%c]\n", xc_name[c], sb_str(&b),
		    x0 ? x0->src : x1 ? x1->src : x2->src);
		rep_obj(NULL);
		rep_str("class", xc_name[c]);
		rep_str("forms", xc_desc[c]);
		for (w = 0; w < 3; w++) {
			const struct xcost *x = xcost_get(ua, c, w);
			static const char *const wn[3] = {"xmm", "ymm", "zmm"};
			struct sbuf pb = {0};
			if (!x || (w > 0 && c < XC_VALU))
				continue;
			rep_obj(wn[w]);
			rep_num("latency", x->lat);
			rep_num("rthroughput", x->rtp);
			rep_int("uops", x->uops);
			xports_text(&pb, ua, x->ports);
			rep_str("ports", sb_str(&pb));
			rep_str("source", xsrc_text(x->src));
			rep_end();
			free(pb.p);
		}
		rep_end();
	}
	rep_end();
	printf("  sources: U uops.info, A Agner Fog, I Intel manual, D AMD Zen 4 "
	    "SOG, M measured here, E estimate, V derived from ymm, G worst of "
	    "the level's cores\n");
	sb_clear(&b);
	model_text(&b, ua);
	printf("  as a ucas cost model (scalar latencies; see `models`):\n");
	{
		const char *s = sb_str(&b);
		while (*s) {
			const char *e = strchr(s, '\n');
			if (!e)
				e = s + strlen(s);
			printf("    %.*s\n", (int)(e - s), s);
			s = *e ? e + 1 : e;
		}
	}
	free(b.p);
}

void
cmd_xmodels(void)
{
	int ua;

	rep_begin("x86_models");
	rep_list("models");
	for (ua = 0; ua < UA_N; ua++) {
		printf("  %-6s %-14s %s\n", uarchs[ua].name, uarchs[ua].alias,
		    uarchs[ua].desc);
		rep_obj(NULL);
		rep_str("model", uarchs[ua].name);
		rep_str("alias", uarchs[ua].alias);
		rep_str("description", uarchs[ua].desc);
		rep_end();
	}
	rep_end();
}
