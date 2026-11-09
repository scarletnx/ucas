/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * x86intr.c's tables: intrinsic families and explicit rows, the
 * generator, the commands and the differential test against cc.
 *
 * A family row: name stem ("%s" the element suffix, "%d" the width),
 * suffixes, signature, semantics, class (and one for pd when it
 * differs), the features at 128/256/512 bits ("-" none; "vl" the 512-bit
 * form's features plus AVX512VL; an "avx512f" 512-bit form of 8- and
 * 16-bit integer elements is AVX512BW), flags, the largest immediate.
 *
 * Signatures: "R = A B ..." with each R/A a width letter, ':' and an
 * element.  Widths: v the row's width W, h W/2, q W/4, e W/8 (at least
 * 128), x 128, y 256, z 512, s a scalar, i an immediate, k a mask
 * (as many bits as lanes, at least 8), p a pointer.  Elements: * the
 * suffix's, u* its unsigned type, or i8 .. f64.
 */

#define FM_MASK 1	/* AVX-512 mask/maskz variants */
#define FM_MASKA 2	/* the masked form's src is operand a (fma, ternlog) */
#define FM_KMASK 4	/* a mask result: mask_ variant ands it with k */
#define FM_NOBW 8	/* 8/16-bit elements at 512 bits stay avx512f */
#define FM_MASK3 16	/* also mask3 (fma: src is c) */

struct fam {
	const char *stem, *sfx, *spec, *sem;
	uint8_t cls, clspd;
	const char *isa;
	uint8_t fl, imax, fa;
};

#define C(x) XC_##x
static const struct fam fams[] = {
	/* ---- integer arithmetic ---- */
	{"add_%s", "epi8 epi16 epi32 epi64", "v:* = v:* v:*", "a + b",
	    C(VALU), 0, "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"sub_%s", "epi8 epi16 epi32 epi64", "v:* = v:* v:*", "a - b",
	    C(VALU), 0, "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"adds_%s", "epi8 epi16 epu8 epu16", "v:* = v:* v:*",
	    "sat(int(a) + int(b))", C(VALU01), 0, "sse2 avx2 avx512f", FM_MASK,
	    0, 0},
	{"subs_%s", "epi8 epi16 epu8 epu16", "v:* = v:* v:*",
	    "sat(int(a) - int(b))", C(VALU01), 0, "sse2 avx2 avx512f", FM_MASK,
	    0, 0},
	{"avg_%s", "epu8 epu16", "v:* = v:* v:*", "(int(a) + int(b) + 1) >> 1",
	    C(VALU01), 0, "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"max_%s", "epi16 epu8", "v:* = v:* v:*", "max(a, b)", C(VALU01), 0,
	    "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"max_%s", "epi8 epu16 epi32 epu32", "v:* = v:* v:*", "max(a, b)",
	    C(VALU01), 0, "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"max_%s", "epi64 epu64", "v:* = v:* v:*", "max(a, b)", C(VALU01), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"min_%s", "epi16 epu8", "v:* = v:* v:*", "min(a, b)", C(VALU01), 0,
	    "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"min_%s", "epi8 epu16 epi32 epu32", "v:* = v:* v:*", "min(a, b)",
	    C(VALU01), 0, "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"min_%s", "epi64 epu64", "v:* = v:* v:*", "min(a, b)", C(VALU01), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"abs_%s", "epi8 epi16 epi32", "v:* = v:*", "a < 0 ? -a : a",
	    C(VALU01), 0, "ssse3 avx2 avx512f", FM_MASK, 0, 0},
	{"abs_%s", "epi64", "v:* = v:*", "a < 0 ? -a : a", C(VALU01), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"sign_%s", "epi8 epi16 epi32", "v:* = v:* v:*",
	    "b < 0 ? -a : b == 0 ? 0 : a", C(VALU01), 0, "ssse3 avx2 -", 0, 0, 0},
	{"mullo_%s", "epi16", "v:* = v:* v:*", "a * b", C(VMUL), 0,
	    "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"mullo_%s", "epi32", "v:* = v:* v:*", "a * b", C(VMULLD), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"mullo_%s", "epi64", "v:* = v:* v:*", "a * b", C(VMULLQ), 0,
	    "vl vl avx512dq", FM_MASK, 0, 0},
	{"mulhi_%s", "epi16 epu16", "v:* = v:* v:*", "(int(a) * int(b)) >> 16",
	    C(VMUL), 0, "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"mulhrs_%s", "epi16", "v:* = v:* v:*",
	    "((int(a) * int(b) >> 14) + 1) >> 1", C(VMUL), 0,
	    "ssse3 avx2 avx512f", FM_MASK, 0, 0},
	{"mul_%s", "epu32", "v:u64 = v:u32 v:u32",
	    "u64(a[2*i]) * u64(b[2*i])", C(VMUL), 0, "sse2 avx2 avx512f",
	    FM_MASK, 0, 0},
	{"mul_%s", "epi32", "v:i64 = v:i32 v:i32",
	    "i64(a[2*i]) * i64(b[2*i])", C(VMUL), 0, "sse4.1 avx2 avx512f",
	    FM_MASK, 0, 0},
	{"madd_%s", "epi16", "v:i32 = v:i16 v:i16",
	    "int(a[2*i]) * int(b[2*i]) + int(a[2*i+1]) * int(b[2*i+1])",
	    C(VMUL), 0, "sse2 avx2 avx512bw", FM_MASK, 0, 0},
	{"maddubs_%s", "epi16", "v:i16 = v:u8 v:i8",
	    "sat(int(a[2*i]) * int(b[2*i]) + int(a[2*i+1]) * int(b[2*i+1]))",
	    C(VMUL), 0, "ssse3 avx2 avx512bw", FM_MASK, 0, 0},
	{"sad_%s", "epu8", "v:u64 = v:u8 v:u8",
	    "add(j, 0, 7, abs(int(a[8*i+j]) - int(b[8*i+j])))", C(VSAD), 0,
	    "sse2 avx2 avx512bw", 0, 0, 0},
	{"hadd_%s", "epi16 epi32", "v:* = v:* v:*",
	    "@def k = i % L; @def o = i - i % L; "
	    "k < L/2 ? a[o + 2*k] + a[o + 2*k + 1] : "
	    "b[o + 2*(k - L/2)] + b[o + 2*(k - L/2) + 1]",
	    C(VHADD), 0, "ssse3 avx2 -", 0, 0, 0},
	{"hsub_%s", "epi16 epi32", "v:* = v:* v:*",
	    "@def k = i % L; @def o = i - i % L; "
	    "k < L/2 ? a[o + 2*k] - a[o + 2*k + 1] : "
	    "b[o + 2*(k - L/2)] - b[o + 2*(k - L/2) + 1]",
	    C(VHADD), 0, "ssse3 avx2 -", 0, 0, 0},
	{"hadds_%s", "epi16", "v:* = v:* v:*",
	    "@def k = i % L; @def o = i - i % L; "
	    "k < L/2 ? sat(int(a[o + 2*k]) + int(a[o + 2*k + 1])) : "
	    "sat(int(b[o + 2*(k - L/2)]) + int(b[o + 2*(k - L/2) + 1]))",
	    C(VHADD), 0, "ssse3 avx2 -", 0, 0, 0},
	/* ---- compares ---- */
	{"cmpeq_%s", "epi8 epi16 epi32", "v:* = v:* v:*", "m(a == b)",
	    C(VALU01), 0, "sse2 avx2 -", 0, 0, 0},
	{"cmpeq_%s", "epi64", "v:* = v:* v:*", "m(a == b)", C(VALU01), 0,
	    "sse4.1 avx2 -", 0, 0, 0},
	{"cmpgt_%s", "epi8 epi16 epi32", "v:* = v:* v:*", "m(a > b)",
	    C(VALU01), 0, "sse2 avx2 -", 0, 0, 0},
	{"cmpgt_%s", "epi64", "v:* = v:* v:*", "m(a > b)", C(VPCMPQ), 0,
	    "sse4.2 avx2 -", 0, 0, 0},
	{"cmplt_%s", "epi8 epi16 epi32", "v:* = v:* v:*", "m(a < b)",
	    C(VALU01), 0, "sse2 - -", 0, 0, 0},
	{"cmpeq_%s_mask", "epi8 epi16 epi32 epi64 epu8 epu16 epu32 epu64",
	    "k = v:* v:*", "or(j, 0, n-1, u64(a[j] == b[j]) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"cmpneq_%s_mask", "epi8 epi16 epi32 epi64 epu8 epu16 epu32 epu64",
	    "k = v:* v:*", "or(j, 0, n-1, u64(a[j] != b[j]) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"cmplt_%s_mask", "epi8 epi16 epi32 epi64 epu8 epu16 epu32 epu64",
	    "k = v:* v:*", "or(j, 0, n-1, u64(a[j] < b[j]) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"cmple_%s_mask", "epi8 epi16 epi32 epi64 epu8 epu16 epu32 epu64",
	    "k = v:* v:*", "or(j, 0, n-1, u64(a[j] <= b[j]) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"cmpgt_%s_mask", "epi8 epi16 epi32 epi64 epu8 epu16 epu32 epu64",
	    "k = v:* v:*", "or(j, 0, n-1, u64(a[j] > b[j]) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"cmpge_%s_mask", "epi8 epi16 epi32 epi64 epu8 epu16 epu32 epu64",
	    "k = v:* v:*", "or(j, 0, n-1, u64(a[j] >= b[j]) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"test_%s_mask", "epi8 epi16 epi32 epi64", "k = v:* v:*",
	    "or(j, 0, n-1, u64((a[j] & b[j]) != 0) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 0, 0},
	{"movepi%s_mask", "8 16 32 64", "k = v:*",
	    "or(j, 0, n-1, u64(a[j] < 0) << j)", C(VCMPK), 0,
	    "vl+avx512dq vl+avx512dq avx512dq", 0, 0, 0},
	/* ---- logic ---- */
	{"and_si%d", "si", "v:i64 = v:i64 v:i64", "a & b", C(VALU), 0,
	    "sse2 avx2 avx512f", 0, 0, 0},
	{"or_si%d", "si", "v:i64 = v:i64 v:i64", "a | b", C(VALU), 0,
	    "sse2 avx2 avx512f", 0, 0, 0},
	{"xor_si%d", "si", "v:i64 = v:i64 v:i64", "a ^ b", C(VALU), 0,
	    "sse2 avx2 avx512f", 0, 0, 0},
	{"andnot_si%d", "si", "v:i64 = v:i64 v:i64", "~a & b", C(VALU), 0,
	    "sse2 avx2 avx512f", 0, 0, 0},
	{"and_%s", "epi32 epi64", "v:* = v:* v:*", "a & b", C(VALU), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"or_%s", "epi32 epi64", "v:* = v:* v:*", "a | b", C(VALU), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"xor_%s", "epi32 epi64", "v:* = v:* v:*", "a ^ b", C(VALU), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"andnot_%s", "epi32 epi64", "v:* = v:* v:*", "~a & b", C(VALU), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"ternarylogic_%s", "epi32 epi64", "v:* = v:* v:* v:* i",
	    "or(j, 0, 7, bit(imm, j) ? (bit(j, 2) ? a : ~a) & "
	    "(bit(j, 1) ? b : ~b) & (bit(j, 0) ? c : ~c) : 0)",
	    C(VTERNLOG), 0, "vl vl avx512f", FM_MASK | FM_MASKA, 255, 0},
	/* ---- shifts ---- */
	{"slli_%s", "epi16 epi32 epi64", "v:* = v:* i",
	    "imm > w - 1 ? 0 : a << imm", C(VSHIFT), 0, "sse2 avx2 avx512f",
	    FM_MASK, 255, 0},
	{"srli_%s", "epi16 epi32 epi64", "v:* = v:* i",
	    "imm > w - 1 ? 0 : u(a) >> imm", C(VSHIFT), 0, "sse2 avx2 avx512f",
	    FM_MASK, 255, 0},
	{"srai_%s", "epi16 epi32", "v:* = v:* i",
	    "imm > w - 1 ? a >> (w - 1) : a >> imm", C(VSHIFT), 0,
	    "sse2 avx2 avx512f", FM_MASK, 255, 0},
	{"srai_%s", "epi64", "v:* = v:* i",
	    "imm > w - 1 ? a >> (w - 1) : a >> imm", C(VSHIFT), 0,
	    "vl vl avx512f", FM_MASK, 255, 0},
	{"sll_%s", "epi16 epi32 epi64", "v:* = v:* x:u64",
	    "b[0] > w - 1 ? 0 : a << b[0]", C(VSHIFTX), 0, "sse2 avx2 avx512f",
	    FM_MASK, 0, 0},
	{"srl_%s", "epi16 epi32 epi64", "v:* = v:* x:u64",
	    "b[0] > w - 1 ? 0 : u(a) >> b[0]", C(VSHIFTX), 0,
	    "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"sra_%s", "epi16 epi32", "v:* = v:* x:u64",
	    "b[0] > w - 1 ? a >> (w - 1) : a >> b[0]", C(VSHIFTX), 0,
	    "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"sllv_%s", "epi32 epi64", "v:* = v:* v:*",
	    "u(b) > w - 1 ? 0 : a << u(b)", C(VSHIFT), 0, "avx2 avx2 avx512f",
	    FM_MASK, 0, 0},
	{"srlv_%s", "epi32 epi64", "v:* = v:* v:*",
	    "u(b) > w - 1 ? 0 : u(a) >> u(b)", C(VSHIFT), 0,
	    "avx2 avx2 avx512f", FM_MASK, 0, 0},
	{"srav_%s", "epi32", "v:* = v:* v:*",
	    "u(b) > w - 1 ? a >> (w - 1) : a >> u(b)", C(VSHIFT), 0,
	    "avx2 avx2 avx512f", FM_MASK, 0, 0},
	{"srav_%s", "epi64", "v:* = v:* v:*",
	    "u(b) > w - 1 ? a >> (w - 1) : a >> u(b)", C(VSHIFT), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"sllv_%s", "epi16", "v:* = v:* v:*", "u(b) > w - 1 ? 0 : a << u(b)",
	    C(VSHIFT), 0, "vl vl avx512bw", FM_MASK, 0, 0},
	{"srlv_%s", "epi16", "v:* = v:* v:*",
	    "u(b) > w - 1 ? 0 : u(a) >> u(b)", C(VSHIFT), 0, "vl vl avx512bw",
	    FM_MASK, 0, 0},
	{"srav_%s", "epi16", "v:* = v:* v:*",
	    "u(b) > w - 1 ? a >> (w - 1) : a >> u(b)", C(VSHIFT), 0,
	    "vl vl avx512bw", FM_MASK, 0, 0},
	{"rol_%s", "epi32 epi64", "v:* = v:* i",
	    "imm % w == 0 ? a : (a << (imm % w)) | (u(a) >> (w - imm % w))",
	    C(VSHIFT), 0, "vl vl avx512f", FM_MASK, 255, 0},
	{"ror_%s", "epi32 epi64", "v:* = v:* i",
	    "imm % w == 0 ? a : (u(a) >> (imm % w)) | (a << (w - imm % w))",
	    C(VSHIFT), 0, "vl vl avx512f", FM_MASK, 255, 0},
	{"rolv_%s", "epi32 epi64", "v:* = v:* v:*",
	    "u(b) % w == 0 ? a : (a << (u(b) % w)) | (u(a) >> (w - u(b) % w))",
	    C(VSHIFT), 0, "vl vl avx512f", FM_MASK, 0, 0},
	{"rorv_%s", "epi32 epi64", "v:* = v:* v:*",
	    "u(b) % w == 0 ? a : (u(a) >> (u(b) % w)) | (a << (w - u(b) % w))",
	    C(VSHIFT), 0, "vl vl avx512f", FM_MASK, 0, 0},
	/* ---- shuffles ---- */
	{"shuffle_%s", "epi8", "v:* = v:* v:*",
	    "b < 0 ? 0 : a[(i - i % 16) + (b & 15)]", C(VSHUF), 0,
	    "ssse3 avx2 avx512f", FM_MASK, 0, 0},
	{"shuffle_%s", "epi32", "v:* = v:* i",
	    "a[(i - i % 4) + ((imm >> (2 * (i % 4))) & 3)]", C(VSHUF), 0,
	    "sse2 avx2 avx512f", FM_MASK, 255, 0},
	{"shufflelo_%s", "epi16", "v:* = v:* i",
	    "i % 8 < 4 ? a[(i - i % 8) + ((imm >> (2 * (i % 8))) & 3)] : a",
	    C(VSHUF), 0, "sse2 avx2 avx512f", FM_MASK, 255, 0},
	{"shufflehi_%s", "epi16", "v:* = v:* i",
	    "i % 8 >= 4 ? a[(i - i % 8) + 4 + ((imm >> (2 * (i % 8 - 4))) & 3)] "
	    ": a", C(VSHUF), 0, "sse2 avx2 avx512f", FM_MASK, 255, 0},
	{"unpacklo_%s", "epi8 epi16 epi32 epi64", "v:* = v:* v:*",
	    "(i % L) % 2 ? b[(i - i % L) + (i % L) / 2] : "
	    "a[(i - i % L) + (i % L) / 2]", C(VSHUF), 0, "sse2 avx2 avx512f",
	    FM_MASK, 0, 0},
	{"unpackhi_%s", "epi8 epi16 epi32 epi64", "v:* = v:* v:*",
	    "(i % L) % 2 ? b[(i - i % L) + L/2 + (i % L) / 2] : "
	    "a[(i - i % L) + L/2 + (i % L) / 2]", C(VSHUF), 0,
	    "sse2 avx2 avx512f", FM_MASK, 0, 0},
	{"alignr_%s", "epi8", "v:* = v:* v:* i",
	    "@def k = i % 16 + imm; @def o = i - i % 16; "
	    "k < 16 ? b[o + k] : k < 32 ? a[o + k - 16] : 0", C(VSHUF), 0,
	    "ssse3 avx2 avx512f", FM_MASK, 255, 0},
	{"packs_%s", "epi16", "v:i8 = v:i16 v:i16",
	    "i % 16 < 8 ? sat(int(a[(i / 16) * 8 + i % 16])) : "
	    "sat(int(b[(i / 16) * 8 + i % 16 - 8]))", C(VSHUF), 0,
	    "sse2 avx2 avx512bw", FM_MASK, 0, 0},
	{"packs_%s", "epi32", "v:i16 = v:i32 v:i32",
	    "i % 8 < 4 ? sat(int(a[(i / 8) * 4 + i % 8])) : "
	    "sat(int(b[(i / 8) * 4 + i % 8 - 4]))", C(VSHUF), 0,
	    "sse2 avx2 avx512bw", FM_MASK, 0, 0},
	{"packus_%s", "epi16", "v:u8 = v:i16 v:i16",
	    "i % 16 < 8 ? sat(int(a[(i / 16) * 8 + i % 16])) : "
	    "sat(int(b[(i / 16) * 8 + i % 16 - 8]))", C(VSHUF), 0,
	    "sse2 avx2 avx512bw", FM_MASK, 0, 0},
	{"packus_%s", "epi32", "v:u16 = v:i32 v:i32",
	    "i % 8 < 4 ? sat(int(a[(i / 8) * 4 + i % 8])) : "
	    "sat(int(b[(i / 8) * 4 + i % 8 - 4]))", C(VSHUF), 0,
	    "sse4.1 avx2 avx512bw", FM_MASK, 0, 0},
	{"blend_%s", "epi16", "v:* = v:* v:* i", "bit(imm, i % 8) ? b : a",
	    C(VSHUF), 0, "sse4.1 avx2 -", 0, 255, 0},
	{"blend_%s", "epi32", "v:* = v:* v:* i", "bit(imm, i % 8) ? b : a",
	    C(VBLEND), 0, "avx2 avx2 -", 0, 255, 0},
	{"blendv_%s", "epi8", "v:* = v:* v:* v:*", "c < 0 ? b : a",
	    C(VBLENDV), 0, "sse4.1 avx2 -", 0, 0, 0},
	{"mask_blend_%s", "epi8 epi16 epi32 epi64", "v:* = k v:* v:*",
	    "bit(a, i) ? c : b", C(VBLEND), 0, "vl vl avx512f", 0, 0, 0},
	{"maskz_compress_%s", "epi32 epi64", "v:* = k v:*",
	    "xor(j, 0, n-1, bit(a, j) && popcnt(a & ((u64(1) << j) - 1)) == i "
	    "? b[j] : 0)", C(VCOMPRESS), 0, "vl vl avx512f", 0, 0, 0},
	{"mask_compress_%s", "epi32 epi64", "v:* = v:* k v:*",
	    "popcnt(b) <= i ? a : xor(j, 0, n-1, bit(b, j) && "
	    "popcnt(b & ((u64(1) << j) - 1)) == i ? c[j] : 0)",
	    C(VCOMPRESS), 0, "vl vl avx512f", 0, 0, 0},
	{"maskz_expand_%s", "epi32 epi64", "v:* = k v:*",
	    "bit(a, i) ? b[popcnt(a & ((u64(1) << i) - 1))] : 0",
	    C(VCOMPRESS), 0, "vl vl avx512f", 0, 0, 0},
	{"permutexvar_%s", "epi16", "v:* = v:* v:*", "b[u(a) % n]", C(VPERM), 0,
	    "vl vl avx512bw", FM_MASK, 0, 0},
	{"permutexvar_%s", "epi8", "v:* = v:* v:*", "b[u(a) % n]", C(VPERM), 0,
	    "vl+avx512vbmi vl+avx512vbmi avx512vbmi", FM_MASK, 0, 0},
	{"permutexvar_%s", "epi32", "v:* = v:* v:*", "b[u(a) % n]", C(VPERM), 0,
	    "- vl avx512f", FM_MASK, 0, 0},
	{"permutexvar_%s", "epi64", "v:* = v:* v:*", "b[u(a) % n]", C(VPERM), 0,
	    "- vl avx512f", FM_MASK, 0, 0},
	{"permutex2var_%s", "epi32 epi64", "v:* = v:* v:* v:*",
	    "bit(u(b), ilog(n)) ? c[u(b) % n] : a[u(b) % n]", C(VPERM), 0,
	    "vl vl avx512f", 0, 0, 0},
	{"permutex2var_%s", "epi16", "v:* = v:* v:* v:*",
	    "bit(u(b), ilog(n)) ? c[u(b) % n] : a[u(b) % n]", C(VPERM), 0,
	    "vl vl avx512bw", 0, 0, 0},
	{"permutex2var_%s", "epi8", "v:* = v:* v:* v:*",
	    "bit(u(b), ilog(n)) ? c[u(b) % n] : a[u(b) % n]", C(VPERM), 0,
	    "vl+avx512vbmi vl+avx512vbmi avx512vbmi", 0, 0, 0},
	{"broadcastb_%s", "epi8", "v:* = x:*", "a[0]", C(VPERM), 0,
	    "avx2 avx2 avx512bw", FM_MASK, 0, 0},
	{"broadcastw_%s", "epi16", "v:* = x:*", "a[0]", C(VPERM), 0,
	    "avx2 avx2 avx512bw", FM_MASK, 0, 0},
	{"broadcastd_%s", "epi32", "v:* = x:*", "a[0]", C(VPERM), 0,
	    "avx2 avx2 avx512f", FM_MASK, 0, 0},
	{"broadcastq_%s", "epi64", "v:* = x:*", "a[0]", C(VPERM), 0,
	    "avx2 avx2 avx512f", FM_MASK, 0, 0},
	/* ---- widening, narrowing ---- */
	{"cvtepi8_%s", "epi16", "v:* = h:i8", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512bw", FM_MASK, 0, 0},
	{"cvtepu8_%s", "epi16", "v:* = h:u8", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512bw", FM_MASK, 0, 0},
	{"cvtepi8_%s", "epi32", "v:* = q:i8", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepu8_%s", "epi32", "v:* = q:u8", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepi8_%s", "epi64", "v:* = e:i8", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepu8_%s", "epi64", "v:* = e:u8", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepi16_%s", "epi32", "v:* = h:i16", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepu16_%s", "epi32", "v:* = h:u16", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepi16_%s", "epi64", "v:* = q:i16", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepu16_%s", "epi64", "v:* = q:u16", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepi32_%s", "epi64", "v:* = h:i32", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepu32_%s", "epi64", "v:* = h:u32", "a", C(VPMOVX), 0,
	    "sse4.1 avx2 avx512f", FM_MASK, 0, 0},
	{"cvtepi32_%s", "epi8", "q:* = v:i32", "i < na ? a : 0", C(VPMOVN), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtepi32_%s", "epi16", "h:* = v:i32", "i < na ? a : 0", C(VPMOVN), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtepi64_%s", "epi8", "e:* = v:i64", "i < na ? a : 0", C(VPMOVN), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtepi64_%s", "epi16", "q:* = v:i64", "i < na ? a : 0", C(VPMOVN), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtepi64_%s", "epi32", "h:* = v:i64", "i < na ? a : 0", C(VPMOVN), 0,
	    "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtepi16_%s", "epi8", "h:* = v:i16", "i < na ? a : 0", C(VPMOVN), 0,
	    "vl vl avx512bw", FM_MASK, 0, 0},
	{"cvtsepi32_%s", "epi8", "q:* = v:i32", "i < na ? sat(int(a)) : 0",
	    C(VPMOVN), 0, "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtsepi16_%s", "epi8", "h:* = v:i16", "i < na ? sat(int(a)) : 0",
	    C(VPMOVN), 0, "vl vl avx512bw", FM_MASK, 0, 0},
	{"cvtusepi32_%s", "epi8", "q:u8 = v:u32", "i < na ? sat(int(a)) : 0",
	    C(VPMOVN), 0, "vl vl avx512f", FM_MASK, 0, 0},
	{"cvtusepi16_%s", "epi8", "h:u8 = v:u16", "i < na ? sat(int(a)) : 0",
	    C(VPMOVN), 0, "vl vl avx512bw", FM_MASK, 0, 0},
	/* ---- to and from scalars ---- */
	{"movemask_%s", "epi8", "s:i32 = v:*",
	    "or(j, 0, n-1, i32(a[j] < 0) << j)", C(VMOVMSK), 0, "sse2 avx2 -", 0,
	    0, 0},
	{"testz_si%d", "si", "s:i32 = v:i64 v:i64",
	    "or(j, 0, n-1, a[j] & b[j]) == 0", C(VTEST), 0, "sse4.1 avx -", 0, 0,
	    0},
	{"testc_si%d", "si", "s:i32 = v:i64 v:i64",
	    "or(j, 0, n-1, ~a[j] & b[j]) == 0", C(VTEST), 0, "sse4.1 avx -", 0, 0,
	    0},
	{"set1_%s", "epi8 epi16 epi32", "v:* = s:*", "a", C(VBCASTG), 0,
	    "sse2 avx avx512f", 0, 0, 0},
	/* ---- bit counts (AVX-512) ---- */
	{"popcnt_%s", "epi32 epi64", "v:* = v:*", "popcnt(u(a))", C(VPOPCNT), 0,
	    "vl+avx512vpopcntdq vl+avx512vpopcntdq avx512vpopcntdq", FM_MASK, 0,
	    0},
	{"popcnt_%s", "epi8 epi16", "v:* = v:*", "popcnt(u(a))", C(VPOPCNT), 0,
	    "vl+avx512bitalg vl+avx512bitalg avx512bitalg", FM_MASK | FM_NOBW,
	    0, 0},
	{"lzcnt_%s", "epi32 epi64", "v:* = v:*", "clz(u(a))", C(VLZCNT), 0,
	    "vl+avx512cd vl+avx512cd avx512cd", FM_MASK, 0, 0},
	/* ---- GFNI ---- */
	{"gf2p8affine_%s", "epi64_epi8", "v:u8 = v:u8 v:u8 i",
	    "xor(k, 0, 7, (popcnt(b[(i - i % 8) + 7 - k] & a) & 1) << k) ^ imm",
	    C(VGF2P8), 0, "gfni gfni+avx gfni+avx512f", FM_MASK | FM_NOBW, 255,
	    0},
	{"gf2p8affineinv_%s", "epi64_epi8", "v:u8 = v:u8 v:u8 i",
	    "xor(k, 0, 7, (popcnt(b[(i - i % 8) + 7 - k] & ginv(a)) & 1) << k) "
	    "^ imm", C(VGF2P8), 0, "gfni gfni+avx gfni+avx512f",
	    FM_MASK | FM_NOBW, 255, 0},
	{"gf2p8mul_%s", "epi8", "v:u8 = v:u8 v:u8", "gfmul(a, b)", C(VGF2P8), 0,
	    "gfni gfni+avx gfni+avx512f", FM_MASK | FM_NOBW, 0, 0},
	/* ---- floating point ---- */
	{"add_%s", "ps pd", "v:* = v:* v:*", "a + b", C(VFADD), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"sub_%s", "ps pd", "v:* = v:* v:*", "a - b", C(VFADD), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"mul_%s", "ps pd", "v:* = v:* v:*", "a * b", C(VFMUL), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"div_%s", "ps pd", "v:* = v:* v:*", "a / b", C(VFDIVS), C(VFDIVD),
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"min_%s", "ps pd", "v:* = v:* v:*", "a < b ? a : b", C(VFMINMAX), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"max_%s", "ps pd", "v:* = v:* v:*", "a > b ? a : b", C(VFMINMAX), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"sqrt_%s", "ps pd", "v:* = v:*", "sqrt(a)", C(VFSQRTS), C(VFSQRTD),
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"and_%s", "ps pd", "v:* = v:* v:*", "asf(asu(a) & asu(b))", C(VALU), 0,
	    "sse2 avx avx512dq", FM_MASK, 0, 0},
	{"or_%s", "ps pd", "v:* = v:* v:*", "asf(asu(a) | asu(b))", C(VALU), 0,
	    "sse2 avx avx512dq", FM_MASK, 0, 0},
	{"xor_%s", "ps pd", "v:* = v:* v:*", "asf(asu(a) ^ asu(b))", C(VALU), 0,
	    "sse2 avx avx512dq", FM_MASK, 0, 0},
	{"andnot_%s", "ps pd", "v:* = v:* v:*", "asf(~asu(a) & asu(b))",
	    C(VALU), 0, "sse2 avx avx512dq", FM_MASK, 0, 0},
	{"cmpeq_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 0))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmplt_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 1))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmple_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 2))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmpunord_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 3))", C(VFCMP),
	    0, "sse2 - -", 0, 0, 0},
	{"cmpneq_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 4))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmpnlt_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 5))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmpnle_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 6))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmpord_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 7))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmpgt_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 14))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmpge_%s", "ps pd", "v:* = v:* v:*", "m(fcmp(a, b, 13))", C(VFCMP), 0,
	    "sse2 - -", 0, 0, 0},
	{"cmp_%s", "ps pd", "v:* = v:* v:* i", "m(fcmp(a, b, imm))", C(VFCMP), 0,
	    "avx avx -", 0, 31, 0},
	{"cmp_%s_mask", "ps pd", "k = v:* v:* i",
	    "or(j, 0, n-1, u64(fcmp(a[j], b[j], imm)) << j)", C(VCMPK), 0,
	    "vl vl avx512f", FM_KMASK, 31, 0},
	{"fmadd_%s", "ps pd", "v:* = v:* v:* v:*", "fma(a, b, c)", C(VFMA), 0,
	    "fma fma avx512f", FM_MASK | FM_MASKA | FM_MASK3, 0, 0},
	{"fmsub_%s", "ps pd", "v:* = v:* v:* v:*", "fma(a, b, -c)", C(VFMA), 0,
	    "fma fma avx512f", FM_MASK | FM_MASKA | FM_MASK3, 0, 0},
	{"fnmadd_%s", "ps pd", "v:* = v:* v:* v:*", "fma(-a, b, c)", C(VFMA), 0,
	    "fma fma avx512f", FM_MASK | FM_MASKA | FM_MASK3, 0, 0},
	{"fnmsub_%s", "ps pd", "v:* = v:* v:* v:*", "fma(-a, b, -c)", C(VFMA),
	    0, "fma fma avx512f", FM_MASK | FM_MASKA | FM_MASK3, 0, 0},
	{"fmaddsub_%s", "ps pd", "v:* = v:* v:* v:*",
	    "i % 2 == 0 ? fma(a, b, -c) : fma(a, b, c)", C(VFMA), 0,
	    "fma fma avx512f", FM_MASK | FM_MASKA | FM_MASK3, 0, 0},
	{"fmsubadd_%s", "ps pd", "v:* = v:* v:* v:*",
	    "i % 2 == 0 ? fma(a, b, c) : fma(a, b, -c)", C(VFMA), 0,
	    "fma fma avx512f", FM_MASK | FM_MASKA | FM_MASK3, 0, 0},
	{"addsub_%s", "ps pd", "v:* = v:* v:*", "i % 2 == 0 ? a - b : a + b",
	    C(VFADD), 0, "sse3 avx -", 0, 0, 0},
	{"hadd_%s", "ps pd", "v:* = v:* v:*",
	    "@def k = i % L; @def o = i - i % L; "
	    "k < L/2 ? a[o + 2*k] + a[o + 2*k + 1] : "
	    "b[o + 2*(k - L/2)] + b[o + 2*(k - L/2) + 1]",
	    C(VHADD), 0, "sse3 avx -", 0, 0, 0},
	{"hsub_%s", "ps pd", "v:* = v:* v:*",
	    "@def k = i % L; @def o = i - i % L; "
	    "k < L/2 ? a[o + 2*k] - a[o + 2*k + 1] : "
	    "b[o + 2*(k - L/2)] - b[o + 2*(k - L/2) + 1]",
	    C(VHADD), 0, "sse3 avx -", 0, 0, 0},
	{"round_%s", "ps pd", "v:* = v:* i", "rnd(a, imm)", C(VROUND), 0,
	    "sse4.1 avx -", 0, 15, 0},
	{"floor_%s", "ps pd", "v:* = v:*", "floor(a)", C(VROUND), 0,
	    "sse4.1 avx avx512f", 0, 0, 0},
	{"ceil_%s", "ps pd", "v:* = v:*", "ceil(a)", C(VROUND), 0,
	    "sse4.1 avx avx512f", 0, 0, 0},
	{"rcp_%s", "ps", "v:* = v:*", "f32(1.0) / a", C(VRCP), 0,
	    "sse2 avx -", 0, 0, FA_APPROX12},
	{"rsqrt_%s", "ps", "v:* = v:*", "f32(1.0) / sqrt(a)", C(VRCP), 0,
	    "sse2 avx -", 0, 0, FA_APPROX12},
	{"rcp14_%s", "ps pd", "v:* = v:*", "1.0 / a", C(VRCP), 0,
	    "vl vl avx512f", FM_MASK, 0, FA_APPROX14},
	{"rsqrt14_%s", "ps pd", "v:* = v:*", "1.0 / sqrt(a)", C(VRCP), 0,
	    "vl vl avx512f", FM_MASK, 0, FA_APPROX14},
	{"cvtepi32_%s", "ps", "v:* = v:i32", "f32(a)", C(VCVT), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvtps_%s", "epi32", "v:* = v:f32", "cvti(a, 32)", C(VCVT), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvttps_%s", "epi32", "v:* = v:f32", "cvtti(a, 32)", C(VCVT), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvtps_%s", "pd", "v:* = h:f32", "f64(a)", C(VCVTW), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvtepi32_%s", "pd", "v:* = h:i32", "f64(a)", C(VCVTW), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvtpd_%s", "ps", "h:* = v:f64", "i < na ? f32(a) : f32(0)", C(VCVTW),
	    0, "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvtpd_%s", "epi32", "h:* = v:f64", "i < na ? cvti(a, 32) : 0",
	    C(VCVTW), 0, "sse2 avx avx512f", FM_MASK, 0, 0},
	{"cvttpd_%s", "epi32", "h:* = v:f64", "i < na ? cvtti(a, 32) : 0",
	    C(VCVTW), 0, "sse2 avx avx512f", FM_MASK, 0, 0},
	{"shuffle_%s", "ps", "v:* = v:* v:* i",
	    "i % 4 < 2 ? a[(i - i % 4) + ((imm >> (2 * (i % 4))) & 3)] : "
	    "b[(i - i % 4) + ((imm >> (2 * (i % 4))) & 3)]", C(VSHUF), 0,
	    "sse2 avx avx512f", FM_MASK, 255, 0},
	{"shuffle_%s", "pd", "v:* = v:* v:* i",
	    "i % 2 == 0 ? a[(i - i % 2) + bit(imm, i)] : "
	    "b[(i - i % 2) + bit(imm, i)]", C(VSHUF), 0, "sse2 avx avx512f",
	    FM_MASK, 255, 0},
	{"unpacklo_%s", "ps pd", "v:* = v:* v:*",
	    "(i % L) % 2 ? b[(i - i % L) + (i % L) / 2] : "
	    "a[(i - i % L) + (i % L) / 2]", C(VSHUF), 0, "sse2 avx avx512f",
	    FM_MASK, 0, 0},
	{"unpackhi_%s", "ps pd", "v:* = v:* v:*",
	    "(i % L) % 2 ? b[(i - i % L) + L/2 + (i % L) / 2] : "
	    "a[(i - i % L) + L/2 + (i % L) / 2]", C(VSHUF), 0,
	    "sse2 avx avx512f", FM_MASK, 0, 0},
	{"moveldup_%s", "ps", "v:* = v:*", "a[i - i % 2]", C(VSHUF), 0,
	    "sse3 avx avx512f", FM_MASK, 0, 0},
	{"movehdup_%s", "ps", "v:* = v:*", "a[i - i % 2 + 1]", C(VSHUF), 0,
	    "sse3 avx avx512f", FM_MASK, 0, 0},
	{"movedup_%s", "pd", "v:* = v:*", "a[i - i % 2]", C(VSHUF), 0,
	    "sse3 avx avx512f", FM_MASK, 0, 0},
	{"permute_%s", "ps", "v:* = v:* i",
	    "a[(i - i % 4) + ((imm >> (2 * (i % 4))) & 3)]", C(VSHUF), 0,
	    "avx avx avx512f", FM_MASK, 255, 0},
	{"permute_%s", "pd", "v:* = v:* i", "a[(i - i % 2) + bit(imm, i)]",
	    C(VSHUF), 0, "avx avx avx512f", FM_MASK, 255, 0},
	{"permutevar_%s", "ps", "v:* = v:* v:i32", "a[(i - i % 4) + (b & 3)]",
	    C(VSHUF), 0, "avx avx avx512f", FM_MASK, 0, 0},
	{"blend_%s", "ps pd", "v:* = v:* v:* i", "bit(imm, i) ? b : a",
	    C(VBLEND), 0, "sse4.1 avx -", 0, 255, 0},
	{"blendv_%s", "ps pd", "v:* = v:* v:* v:*", "s(asu(c)) < 0 ? b : a",
	    C(VBLENDV), 0, "sse4.1 avx -", 0, 0, 0},
	{"mask_blend_%s", "ps pd", "v:* = k v:* v:*", "bit(a, i) ? c : b",
	    C(VBLEND), 0, "vl vl avx512f", 0, 0, 0},
	{"movemask_%s", "ps pd", "s:i32 = v:*",
	    "or(j, 0, n-1, i32(s(asu(a[j])) < 0) << j)", C(VMOVMSK), 0,
	    "sse2 avx -", 0, 0, 0},
	{"set1_%s", "ps pd", "v:* = s:*", "a", C(VBCASTG), 0,
	    "sse2 avx avx512f", 0, 0, 0},
	/* scalar lane 0, the others from a */
	{"add_%s", "ss sd", "x:* = x:* x:*", "i == 0 ? a + b : a", C(VFADD), 0,
	    "sse2 - -", 0, 0, 0},
	{"sub_%s", "ss sd", "x:* = x:* x:*", "i == 0 ? a - b : a", C(VFADD), 0,
	    "sse2 - -", 0, 0, 0},
	{"mul_%s", "ss sd", "x:* = x:* x:*", "i == 0 ? a * b : a", C(VFMUL), 0,
	    "sse2 - -", 0, 0, 0},
	{"div_%s", "ss sd", "x:* = x:* x:*", "i == 0 ? a / b : a", C(VFDIVS),
	    C(VFDIVD), "sse2 - -", 0, 0, 0},
	{"min_%s", "ss sd", "x:* = x:* x:*", "i == 0 ? (a < b ? a : b) : a",
	    C(VFMINMAX), 0, "sse2 - -", 0, 0, 0},
	{"max_%s", "ss sd", "x:* = x:* x:*", "i == 0 ? (a > b ? a : b) : a",
	    C(VFMINMAX), 0, "sse2 - -", 0, 0, 0},
	{"sqrt_%s", "sd", "x:* = x:* x:*", "i == 0 ? sqrt(b[0]) : a",
	    C(VFSQRTD), 0, "sse2 - -", 0, 0, 0},
	{"fmadd_%s", "ss sd", "x:* = x:* x:* x:*", "i == 0 ? fma(a, b, c) : a",
	    C(VFMA), 0, "fma - -", 0, 0, 0},
	{"fmsub_%s", "ss sd", "x:* = x:* x:* x:*", "i == 0 ? fma(a, b, -c) : a",
	    C(VFMA), 0, "fma - -", 0, 0, 0},
	{"fnmadd_%s", "ss sd", "x:* = x:* x:* x:*", "i == 0 ? fma(-a, b, c) : a",
	    C(VFMA), 0, "fma - -", 0, 0, 0},
	{NULL, NULL, NULL, NULL, 0, 0, NULL, 0, 0, 0}
};

/* irregular intrinsics: name, features, signature (fixed widths), sem */
struct xrow1 {
	const char *name, *isa, *spec, *sem, *sem2;
	uint8_t cls, imax, fa;
};
static const struct xrow1 xrows[] = {
	/* scalar: POPCNT LZCNT BMI1 BMI2 ADX SSE4.2 */
	{"_mm_popcnt_u32", "popcnt", "s:i32 = s:u32", "popcnt(a)", NULL,
	    C(BITCNT), 0, 0},
	{"_mm_popcnt_u64", "popcnt", "s:i64 = s:u64", "popcnt(a)", NULL,
	    C(BITCNT), 0, 0},
	{"_lzcnt_u32", "lzcnt", "s:u32 = s:u32", "clz(a)", NULL, C(BITCNT), 0, 0},
	{"_lzcnt_u64", "lzcnt", "s:u64 = s:u64", "clz(a)", NULL, C(BITCNT), 0, 0},
	{"_tzcnt_u32", "bmi", "s:u32 = s:u32", "ctz(a)", NULL, C(BITCNT), 0, 0},
	{"_tzcnt_u64", "bmi", "s:u64 = s:u64", "ctz(a)", NULL, C(BITCNT), 0, 0},
	{"_andn_u32", "bmi", "s:u32 = s:u32 s:u32", "~a & b", NULL, C(BMI), 0, 0},
	{"_andn_u64", "bmi", "s:u64 = s:u64 s:u64", "~a & b", NULL, C(BMI), 0, 0},
	{"_blsr_u32", "bmi", "s:u32 = s:u32", "a & (a - 1)", NULL, C(BMI), 0, 0},
	{"_blsr_u64", "bmi", "s:u64 = s:u64", "a & (a - 1)", NULL, C(BMI), 0, 0},
	{"_blsi_u32", "bmi", "s:u32 = s:u32", "a & -a", NULL, C(BMI), 0, 0},
	{"_blsi_u64", "bmi", "s:u64 = s:u64", "a & -a", NULL, C(BMI), 0, 0},
	{"_blsmsk_u32", "bmi", "s:u32 = s:u32", "a ^ (a - 1)", NULL, C(BMI), 0,
	    0},
	{"_blsmsk_u64", "bmi", "s:u64 = s:u64", "a ^ (a - 1)", NULL, C(BMI), 0,
	    0},
	{"_bextr_u32", "bmi", "s:u32 = s:u32 s:u32 s:u32",
	    "(b & 255) > 31 ? 0 : (c & 255) >= 32 ? a >> (b & 255) : "
	    "u64(a >> (b & 255)) & ((u64(1) << (c & 255)) - 1)", NULL, C(BMI), 0,
	    0},
	{"_bextr_u64", "bmi", "s:u64 = s:u64 s:u32 s:u32",
	    "(b & 255) > 63 ? 0 : (c & 255) >= 64 ? a >> (b & 255) : "
	    "(a >> (b & 255)) & ((u64(1) << (c & 255)) - 1)", NULL, C(BMI), 0, 0},
	{"_bzhi_u32", "bmi2", "s:u32 = s:u32 s:u32",
	    "(b & 255) >= 32 ? a : u64(a) & ((u64(1) << (b & 255)) - 1)", NULL,
	    C(BMI), 0, 0},
	{"_bzhi_u64", "bmi2", "s:u64 = s:u64 s:u32",
	    "(b & 255) >= 64 ? a : a & ((u64(1) << (b & 255)) - 1)", NULL,
	    C(BMI), 0, 0},
	{"_pdep_u32", "bmi2", "s:u32 = s:u32 s:u32",
	    "or(j, 0, 31, bit(b, j) != 0 && bit(a, popcnt(b & ((u64(1) << j) - 1)))"
	    " != 0 ? u64(1) << j : 0)", NULL, C(PDEP), 0, 0},
	{"_pdep_u64", "bmi2", "s:u64 = s:u64 s:u64",
	    "or(j, 0, 63, bit(b, j) != 0 && bit(a, popcnt(b & ((u64(1) << j) - 1)))"
	    " != 0 ? u64(1) << j : 0)", NULL, C(PDEP), 0, 0},
	{"_pext_u32", "bmi2", "s:u32 = s:u32 s:u32",
	    "or(j, 0, 31, bit(b, j) != 0 && bit(a, j) != 0 ? "
	    "u64(1) << popcnt(b & ((u64(1) << j) - 1)) : 0)", NULL, C(PDEP), 0, 0},
	{"_pext_u64", "bmi2", "s:u64 = s:u64 s:u64",
	    "or(j, 0, 63, bit(b, j) != 0 && bit(a, j) != 0 ? "
	    "u64(1) << popcnt(b & ((u64(1) << j) - 1)) : 0)", NULL, C(PDEP), 0, 0},
	{"_mulx_u32", "bmi2", "s:u32 = s:u32 s:u32 p:u32",
	    "u64(a) * u64(b)", "(u64(a) * u64(b)) >> 32", C(MULX), 0, FA_PTROUT},
	{"_mulx_u64", "bmi2", "s:u64 = s:u64 s:u64 p:u64", "a * b",
	    "umulhi64(a, b)", C(MULX), 0, FA_PTROUT},
	{"_addcarry_u32", "", "s:u8 = s:u8 s:u32 s:u32 p:u32",
	    "u64(b) + u64(c) + u64(a != 0) > 0xffffffff",
	    "u64(b) + u64(c) + u64(a != 0)", C(ADC), 0, FA_PTROUT},
	{"_addcarry_u64", "", "s:u8 = s:u8 s:u64 s:u64 p:u64",
	    "b + c < b || (a != 0 && b + c == 0xffffffffffffffff)",
	    "b + c + u64(a != 0)", C(ADC), 0, FA_PTROUT},
	{"_subborrow_u32", "", "s:u8 = s:u8 s:u32 s:u32 p:u32",
	    "u64(b) < u64(c) + u64(a != 0)", "b - c - u32(a != 0)", C(ADC), 0,
	    FA_PTROUT},
	{"_subborrow_u64", "", "s:u8 = s:u8 s:u64 s:u64 p:u64",
	    "b < c || (a != 0 && b == c)", "b - c - u64(a != 0)", C(ADC), 0,
	    FA_PTROUT},
	{"_addcarryx_u32", "adx", "s:u8 = s:u8 s:u32 s:u32 p:u32",
	    "u64(b) + u64(c) + u64(a != 0) > 0xffffffff",
	    "u64(b) + u64(c) + u64(a != 0)", C(ADC), 0, FA_PTROUT},
	{"_addcarryx_u64", "adx", "s:u8 = s:u8 s:u64 s:u64 p:u64",
	    "b + c < b || (a != 0 && b + c == 0xffffffffffffffff)",
	    "b + c + u64(a != 0)", C(ADC), 0, FA_PTROUT},
	{"_bswap", "", "s:i32 = s:i32",
	    "or(j, 0, 3, ((u(a) >> (8*j)) & 255) << (8*(3-j)))", NULL, C(ALU), 0,
	    0},
	{"_bswap64", "", "s:i64 = s:i64",
	    "or(j, 0, 7, ((u(a) >> (8*j)) & 255) << (8*(7-j)))", NULL, C(ALU), 0,
	    0},
	{"_mm_crc32_u8", "sse4.2", "s:u32 = s:u32 s:u8", "crc32c(a, b, 8)",
	    NULL, C(CRC32), 0, 0},
	{"_mm_crc32_u16", "sse4.2", "s:u32 = s:u32 s:u16", "crc32c(a, b, 16)",
	    NULL, C(CRC32), 0, 0},
	{"_mm_crc32_u32", "sse4.2", "s:u32 = s:u32 s:u32", "crc32c(a, b, 32)",
	    NULL, C(CRC32), 0, 0},
	{"_mm_crc32_u64", "sse4.2", "s:u64 = s:u64 s:u64", "crc32c(a, b, 64)",
	    NULL, C(CRC32), 0, 0},
	/* vector <-> scalar */
	{"_mm_cvtsi32_si128", "sse2", "x:i32 = s:i32", "i == 0 ? a : 0", NULL,
	    C(MOVG2X), 0, 0},
	{"_mm_cvtsi64_si128", "sse2", "x:i64 = s:i64", "i == 0 ? a : 0", NULL,
	    C(MOVG2X), 0, 0},
	{"_mm_cvtsi128_si32", "sse2", "s:i32 = x:i32", "a[0]", NULL, C(MOVX2G),
	    0, 0},
	{"_mm_cvtsi128_si64", "sse2", "s:i64 = x:i64", "a[0]", NULL, C(MOVX2G),
	    0, 0},
	{"_mm_extract_epi8", "sse4.1", "s:i32 = x:u8 i", "a[imm % 16]", NULL,
	    C(VEXTR), 255, 0},
	{"_mm_extract_epi16", "sse2", "s:i32 = x:u16 i", "a[imm % 8]", NULL,
	    C(VEXTR), 255, 0},
	{"_mm_extract_epi32", "sse4.1", "s:i32 = x:i32 i", "a[imm % 4]", NULL,
	    C(VEXTR), 255, 0},
	{"_mm_extract_epi64", "sse4.1", "s:i64 = x:i64 i", "a[imm % 2]", NULL,
	    C(VEXTR), 255, 0},
	{"_mm_insert_epi8", "sse4.1", "x:i8 = x:i8 s:i32 i",
	    "i == imm % 16 ? i8(b) : a", NULL, C(VINSR), 255, 0},
	{"_mm_insert_epi16", "sse2", "x:i16 = x:i16 s:i32 i",
	    "i == imm % 8 ? i16(b) : a", NULL, C(VINSR), 255, 0},
	{"_mm_insert_epi32", "sse4.1", "x:i32 = x:i32 s:i32 i",
	    "i == imm % 4 ? b : a", NULL, C(VINSR), 255, 0},
	{"_mm_insert_epi64", "sse4.1", "x:i64 = x:i64 s:i64 i",
	    "i == imm % 2 ? b : a", NULL, C(VINSR), 255, 0},
	{"_mm256_extract_epi32", "avx", "s:i32 = y:i32 i", "a[imm % 8]", NULL,
	    C(VEXTR), 255, 0},
	{"_mm256_extract_epi64", "avx", "s:i64 = y:i64 i", "a[imm % 4]", NULL,
	    C(VEXTR), 255, 0},
	{"_mm_set1_epi64x", "sse2", "x:i64 = s:i64", "a", NULL, C(VBCASTG), 0, 0},
	{"_mm256_set1_epi64x", "avx", "y:i64 = s:i64", "a", NULL, C(VBCASTG), 0,
	    0},
	{"_mm512_set1_epi64", "avx512f", "z:i64 = s:i64", "a", NULL, C(VBCASTG),
	    0, 0},
	{"_mm_setzero_si128", "sse2", "x:i64 =", "0", NULL, C(VALU), 0, 0},
	{"_mm256_setzero_si256", "avx", "y:i64 =", "0", NULL, C(VALU), 0, 0},
	{"_mm512_setzero_si512", "avx512f", "z:i64 =", "0", NULL, C(VALU), 0, 0},
	{"_mm_cvtss_f32", "sse2", "s:f32 = x:f32", "a[0]", NULL, C(VALU), 0, 0},
	{"_mm_cvtsd_f64", "sse2", "s:f64 = x:f64", "a[0]", NULL, C(VALU), 0, 0},
	{"_mm_cvtsi32_ss", "sse2", "x:f32 = x:f32 s:i32",
	    "i == 0 ? f32(b) : a", NULL, C(CVTI2F), 0, 0},
	{"_mm_cvtsi32_sd", "sse2", "x:f64 = x:f64 s:i32",
	    "i == 0 ? f64(b) : a", NULL, C(CVTI2F), 0, 0},
	{"_mm_cvtsi64_sd", "sse2", "x:f64 = x:f64 s:i64",
	    "i == 0 ? f64(b) : a", NULL, C(CVTI2F), 0, 0},
	{"_mm_cvtss_si32", "sse2", "s:i32 = x:f32", "cvti(a[0], 32)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvttss_si32", "sse2", "s:i32 = x:f32", "cvtti(a[0], 32)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvtss_si64", "sse2", "s:i64 = x:f32", "cvti(a[0], 64)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvttss_si64", "sse2", "s:i64 = x:f32", "cvtti(a[0], 64)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvtsd_si32", "sse2", "s:i32 = x:f64", "cvti(a[0], 32)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvttsd_si32", "sse2", "s:i32 = x:f64", "cvtti(a[0], 32)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvtsd_si64", "sse2", "s:i64 = x:f64", "cvti(a[0], 64)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_cvttsd_si64", "sse2", "s:i64 = x:f64", "cvtti(a[0], 64)", NULL,
	    C(CVTF2I), 0, 0},
	{"_mm_sqrt_ss", "sse2", "x:f32 = x:f32", "i == 0 ? sqrt(a) : a", NULL,
	    C(VFSQRTS), 0, 0},
	{"_mm_movehl_ps", "sse2", "x:f32 = x:f32 x:f32", "i < 2 ? b[i + 2] : a",
	    NULL, C(VSHUF), 0, 0},
	{"_mm_movelh_ps", "sse2", "x:f32 = x:f32 x:f32", "i < 2 ? a : b[i - 2]",
	    NULL, C(VSHUF), 0, 0},
	{"_mm_minpos_epu16", "sse4.1", "x:u16 = x:u16",
	    "@def mv = min(min(min(a[0], a[1]), min(a[2], a[3])), "
	    "min(min(a[4], a[5]), min(a[6], a[7]))); "
	    "i == 0 ? mv : i == 1 ? (a[0] == mv ? 0 : a[1] == mv ? 1 : "
	    "a[2] == mv ? 2 : a[3] == mv ? 3 : a[4] == mv ? 4 : a[5] == mv ? 5 :"
	    " a[6] == mv ? 6 : 7) : 0", NULL, C(VMUL), 0, 0},
	{"_mm_mpsadbw_epu8", "sse4.1", "x:u16 = x:u8 x:u8 i",
	    "add(j, 0, 3, abs(int(a[((imm >> 2) & 1) * 4 + i + j]) - "
	    "int(b[(imm & 3) * 4 + j])))", NULL, C(VMPSAD), 7, 0},
	/* byte shifts of each 128-bit lane */
	{"_mm_bslli_si128", "sse2", "x:u8 = x:u8 i",
	    "i % 16 < imm ? 0 : a[i - imm]", NULL, C(VSHUF), 255, 0},
	{"_mm_bsrli_si128", "sse2", "x:u8 = x:u8 i",
	    "i % 16 + imm > 15 ? 0 : a[i + imm]", NULL, C(VSHUF), 255, 0},
	{"_mm_slli_si128", "sse2", "x:u8 = x:u8 i",
	    "i % 16 < imm ? 0 : a[i - imm]", NULL, C(VSHUF), 255, 0},
	{"_mm_srli_si128", "sse2", "x:u8 = x:u8 i",
	    "i % 16 + imm > 15 ? 0 : a[i + imm]", NULL, C(VSHUF), 255, 0},
	{"_mm256_bslli_epi128", "avx2", "y:u8 = y:u8 i",
	    "i % 16 < imm ? 0 : a[i - imm]", NULL, C(VSHUF), 255, 0},
	{"_mm256_bsrli_epi128", "avx2", "y:u8 = y:u8 i",
	    "i % 16 + imm > 15 ? 0 : a[i + imm]", NULL, C(VSHUF), 255, 0},
	{"_mm512_bslli_epi128", "avx512bw", "z:u8 = z:u8 i",
	    "i % 16 < imm ? 0 : a[i - imm]", NULL, C(VSHUF), 255, 0},
	{"_mm512_bsrli_epi128", "avx512bw", "z:u8 = z:u8 i",
	    "i % 16 + imm > 15 ? 0 : a[i + imm]", NULL, C(VSHUF), 255, 0},
	/* AVX2 cross-lane */
	{"_mm256_permutevar8x32_epi32", "avx2", "y:i32 = y:i32 y:i32",
	    "a[u(b) % 8]", NULL, C(VPERM), 0, 0},
	{"_mm256_permutevar8x32_ps", "avx2", "y:f32 = y:f32 y:i32",
	    "a[u(b) % 8]", NULL, C(VPERM), 0, 0},
	{"_mm256_permute4x64_epi64", "avx2", "y:i64 = y:i64 i",
	    "a[(imm >> (2 * i)) & 3]", NULL, C(VPERM), 255, 0},
	{"_mm256_permute4x64_pd", "avx2", "y:f64 = y:f64 i",
	    "a[(imm >> (2 * i)) & 3]", NULL, C(VPERM), 255, 0},
	{"_mm256_permute2x128_si256", "avx2", "y:i64 = y:i64 y:i64 i",
	    "@def sl = (imm >> (4 * (i / 2))) & 15; "
	    "sl & 8 ? 0 : sl & 2 ? b[2 * (sl & 1) + i % 2] : "
	    "a[2 * (sl & 1) + i % 2]", NULL, C(VPERM), 255, 0},
	{"_mm256_inserti128_si256", "avx2", "y:i64 = y:i64 x:i64 i",
	    "i / 2 == imm % 2 ? b[i % 2] : a", NULL, C(VPERM), 1, 0},
	{"_mm256_extracti128_si256", "avx2", "x:i64 = y:i64 i",
	    "a[2 * (imm % 2) + i]", NULL, C(VPERM), 1, 0},
	{"_mm256_movemask_epi8", "avx2", "s:i32 = y:i8",
	    "or(j, 0, 31, i32(a[j] < 0) << j)", NULL, C(VMOVMSK), 0, 0},
	/* carry-less multiply, AES */
	{"_mm_clmulepi64_si128", "pclmul", "x:u64 = x:u64 x:u64 i",
	    "@def X = a[imm & 1]; @def Y = b[(imm >> 4) & 1]; "
	    "i == 0 ? xor(j, 0, 63, bit(Y, j) != 0 ? X << j : 0) : "
	    "xor(j, 1, 63, bit(Y, j) != 0 ? X >> (64 - j) : 0)", NULL,
	    C(PCLMUL), 255, 0},
	{"_mm256_clmulepi64_epi128", "vpclmulqdq+avx", "y:u64 = y:u64 y:u64 i",
	    "@def X = a[(i - i % 2) + (imm & 1)]; "
	    "@def Y = b[(i - i % 2) + ((imm >> 4) & 1)]; "
	    "i % 2 == 0 ? xor(j, 0, 63, bit(Y, j) != 0 ? X << j : 0) : "
	    "xor(j, 1, 63, bit(Y, j) != 0 ? X >> (64 - j) : 0)", NULL,
	    C(PCLMUL), 255, 0},
	{"_mm512_clmulepi64_epi128", "vpclmulqdq+avx512f",
	    "z:u64 = z:u64 z:u64 i",
	    "@def X = a[(i - i % 2) + (imm & 1)]; "
	    "@def Y = b[(i - i % 2) + ((imm >> 4) & 1)]; "
	    "i % 2 == 0 ? xor(j, 0, 63, bit(Y, j) != 0 ? X << j : 0) : "
	    "xor(j, 1, 63, bit(Y, j) != 0 ? X >> (64 - j) : 0)", NULL,
	    C(PCLMUL), 255, 0},
	{"_mm_aesenclast_si128", "aes", "x:u8 = x:u8 x:u8",
	    "sbox(a[aes_sr(i)]) ^ b", NULL, C(AES), 0, 0},
	{"_mm_aesenc_si128", "aes", "x:u8 = x:u8 x:u8",
	    "@def S(q) = sbox(a[aes_sr(q)]); @def c = i - i % 4; @def r = i % 4; "
	    "xtime(S(c + r)) ^ xtime(S(c + (r + 1) % 4)) ^ S(c + (r + 1) % 4) ^ "
	    "S(c + (r + 2) % 4) ^ S(c + (r + 3) % 4) ^ b", NULL, C(AES), 0, 0},
	{"_mm_aesdeclast_si128", "aes", "x:u8 = x:u8 x:u8",
	    "isbox(a[aes_isr(i)]) ^ b", NULL, C(AES), 0, 0},
	{"_mm_aesdec_si128", "aes", "x:u8 = x:u8 x:u8",
	    "@def S(q) = isbox(a[aes_isr(q)]); @def c = i - i % 4; "
	    "@def r = i % 4; "
	    "gfmul(S(c + r), 14) ^ gfmul(S(c + (r + 1) % 4), 11) ^ "
	    "gfmul(S(c + (r + 2) % 4), 13) ^ gfmul(S(c + (r + 3) % 4), 9) ^ b",
	    NULL, C(AES), 0, 0},
	{"_mm_aesimc_si128", "aes", "x:u8 = x:u8",
	    "@def c = i - i % 4; @def r = i % 4; "
	    "gfmul(a[c + r], 14) ^ gfmul(a[c + (r + 1) % 4], 11) ^ "
	    "gfmul(a[c + (r + 2) % 4], 13) ^ gfmul(a[c + (r + 3) % 4], 9)",
	    NULL, C(AES), 0, 0},
	/* memory: costs and features; semantics in words */
	{"_mm_loadu_si128", "sse2", "x:i64 = p:i64", NULL, NULL, C(VLOAD), 0,
	    FA_MEM | FA_NOEXEC},
	{"_mm256_loadu_si256", "avx", "y:i64 = p:i64", NULL, NULL, C(VLOAD), 0,
	    FA_MEM | FA_NOEXEC},
	{"_mm512_loadu_si512", "avx512f", "z:i64 = p:i64", NULL, NULL, C(VLOAD),
	    0, FA_MEM | FA_NOEXEC},
	{"_mm_storeu_si128", "sse2", "s:i32 = p:i64 x:i64", NULL, NULL,
	    C(VSTORE), 0, FA_MEM | FA_NOEXEC},
	{"_mm256_storeu_si256", "avx", "s:i32 = p:i64 y:i64", NULL, NULL,
	    C(VSTORE), 0, FA_MEM | FA_NOEXEC},
	{"_mm512_storeu_si512", "avx512f", "s:i32 = p:i64 z:i64", NULL, NULL,
	    C(VSTORE), 0, FA_MEM | FA_NOEXEC},
	{"_mm_maskload_epi32", "avx2", "x:i32 = p:i32 x:i32", NULL, NULL,
	    C(VMASKLD), 0, FA_MEM | FA_NOEXEC},
	{"_mm256_maskload_epi32", "avx2", "y:i32 = p:i32 y:i32", NULL, NULL,
	    C(VMASKLD), 0, FA_MEM | FA_NOEXEC},
	{"_mm256_maskstore_epi32", "avx2", "s:i32 = p:i32 y:i32 y:i32", NULL,
	    NULL, C(VMASKST), 0, FA_MEM | FA_NOEXEC},
	{"_mm256_i32gather_epi32", "avx2", "y:i32 = p:i32 y:i32 i", NULL, NULL,
	    C(VGATHERD), 8, FA_MEM | FA_NOEXEC},
	{"_mm256_i64gather_epi64", "avx2", "y:i64 = p:i64 y:i64 i", NULL, NULL,
	    C(VGATHERQ), 8, FA_MEM | FA_NOEXEC},
	{"_mm512_i32gather_epi32", "avx512f", "z:i32 = z:i32 p:i32 i", NULL,
	    NULL, C(VGATHERD), 8, FA_MEM | FA_NOEXEC},
	{"_mm512_maskz_loadu_epi32", "avx512f", "z:i32 = k16 p:i32", NULL, NULL,
	    C(VLOAD), 0, FA_MEM | FA_NOEXEC},
	{"_mm512_mask_storeu_epi32", "avx512f", "s:i32 = p:i32 k16 z:i32", NULL,
	    NULL, C(VSTORE), 0, FA_MEM | FA_NOEXEC},
	{NULL, NULL, NULL, NULL, NULL, 0, 0, 0}
};
#undef C

/* ---- generation ---- */

static int
sfx_elem(const char *s)
{
	static const struct { const char *s; int e; } m[] = {
		{"epi8", E_I8}, {"epu8", E_U8}, {"epi16", E_I16},
		{"epu16", E_U16}, {"epi32", E_I32}, {"epu32", E_U32},
		{"epi64", E_I64}, {"epu64", E_U64}, {"ps", E_F32}, {"pd", E_F64},
		{"ss", E_F32}, {"sd", E_F64}, {"si", E_I64}, {"8", E_I8},
		{"16", E_I16}, {"32", E_I32}, {"64", E_I64},
		{"epi64_epi8", E_U8}, {NULL, 0}};
	int k;

	for (k = 0; m[k].s; k++)
		if (!strcmp(m[k].s, s))
			return m[k].e;
	return E_I32;
}

static xfeat
feats(const char *tok)
{
	xfeat f = 0;
	char buf[64], *p, *q;

	if (!tok || !strcmp(tok, "-"))
		return 0;
	snprintf(buf, sizeof buf, "%s", tok);
	for (p = buf; p && *p; p = q) {
		int k;
		q = strchr(p, '+');
		if (q)
			*q++ = 0;
		if (!*p)
			continue;
		k = xfeat_find(p);
		if (k < 0)
			fail("x86 catalogue: unknown feature %s", p);
		f |= XB(k);
	}
	return f;
}

static int
wbits(char c, int W)
{
	int b;

	switch (c) {
	case 'v': return W;
	case 'h': b = W / 2; break;
	case 'q': b = W / 4; break;
	case 'e': b = W / 8; break;
	case 'x': return 128;
	case 'y': return 256;
	case 'z': return 512;
	default: return 0;
	}
	return b < 128 ? 128 : b;
}

/* parse "v:i32 = h:u8 i k" for width W and the family element fe */
static void
parse_spec(const char *spec, int W, int fe, struct ispec *sp)
{
	const char *s = spec;
	int k = -1, lanes0 = 0;

	memset(sp, 0, sizeof *sp);
	while (*s) {
		char tk[16];
		size_t n = 0;
		uint8_t kind, el = (uint8_t)fe;
		uint16_t bits = 0;
		while (*s == ' ')
			s++;
		if (!*s)
			break;
		while (*s && *s != ' ' && n + 1 < sizeof tk)
			tk[n++] = *s++;
		tk[n] = 0;
		if (!strcmp(tk, "=")) {
			k = 0;
			continue;
		}
		switch (tk[0]) {
		case 's': kind = OK_S; break;
		case 'i': kind = OK_I; el = E_I32; break;
		case 'k': kind = OK_K; bits = (uint16_t)atoi(tk + 1); break;
		case 'p': kind = OK_P; break;
		default: kind = OK_V; bits = (uint16_t)wbits(tk[0], W);
		}
		if (tk[1] == ':' ) {
			const char *e = tk + 2;
			if (!strcmp(e, "*"))
				el = (uint8_t)fe;
			else if (!strcmp(e, "u*"))
				el = (uint8_t)(fe | 1);
			else {
				int x = ename_find(e, strlen(e));
				if (x < 0)
					fail("x86 catalogue: element %s", e);
				el = (uint8_t)x;
			}
		}
		if (k < 0) {
			sp->rk = kind;
			sp->re = el;
			sp->rbits = bits;
		} else {
			if (k == MAXOP)
				fail("x86 catalogue: too many operands");
			sp->ok[k] = kind;
			sp->oe[k] = el;
			sp->obits[k] = bits;
			if (kind == OK_V && !lanes0)
				lanes0 = bits / ebits[el];
			k++;
		}
	}
	sp->nop = (uint8_t)(k < 0 ? 0 : k);
	/* masks: as many bits as lanes, at least 8 */
	{
		int lanes = sp->rk == OK_V ? sp->rbits / ebits[sp->re] : lanes0, j;
		int mb = lanes <= 8 ? 8 : lanes;
		if (sp->rk == OK_K && !sp->rbits) {
			sp->rbits = (uint16_t)mb;
			sp->re = mb == 8 ? E_U8 : mb == 16 ? E_U16 : mb == 32 ? E_U32 :
			    E_U64;
		}
		for (j = 0; j < sp->nop; j++)
			if (sp->ok[j] == OK_K) {
				int b = sp->obits[j] ? sp->obits[j] : mb;
				sp->obits[j] = (uint16_t)b;
				sp->oe[j] = b == 8 ? E_U8 : b == 16 ? E_U16 : b == 32 ?
				    E_U32 : E_U64;
			}
	}
}

static const char *
wprefix(int W)
{
	return W == 512 ? "_mm512_" : W == 256 ? "_mm256_" : "_mm_";
}

static void
fam_name(char *buf, size_t n, const char *pre, const char *stem,
    const char *sfx, int W)
{
	char st[64];
	const char *p;
	size_t k = 0;

	for (p = stem; *p && k + 8 < sizeof st; p++) {
		if (p[0] == '%' && p[1] == 's') {
			k += (size_t)snprintf(st + k, sizeof st - k, "%s", sfx);
			p++;
		} else if (p[0] == '%' && p[1] == 'd') {
			k += (size_t)snprintf(st + k, sizeof st - k, "%d", W);
			p++;
		} else
			st[k++] = *p;
	}
	st[k] = 0;
	snprintf(buf, n, "%s%s", pre, st);
}

static void
gen_family(const struct fam *f)
{
	char sfxs[96], *sx, *save = NULL;

	snprintf(sfxs, sizeof sfxs, "%s", f->sfx);
	for (sx = strtok_r(sfxs, " ", &save); sx; sx = strtok_r(NULL, " ",
	    &save)) {
		int fe = sfx_elem(sx), wi;
		char isa[3][48];
		int intsmall = !efloat(fe) && ebits[fe] <= 16 &&
		    !(f->fl & FM_NOBW);
		xfeat f512;
		if (sscanf(f->isa, "%47s %47s %47s", isa[0], isa[1], isa[2]) != 3)
			fail("x86 catalogue: isa of %s", f->stem);
		f512 = feats(isa[2]);
		if ((f512 & XB(XF_AVX512F)) && intsmall)
			f512 = (f512 & ~XB(XF_AVX512F)) | XB(XF_AVX512BW);
		if (!strcmp(sx, "ss") || !strcmp(sx, "sd"))
			wi = 0;
		for (wi = 0; wi < 3; wi++) {
			int W = 128 << wi, avx512form;
			xfeat ft;
			struct ispec sp;
			char nm[96];
			uint32_t id;
			const char *tok = isa[wi];
			if (!strcmp(tok, "-"))
				continue;
			if (!strncmp(tok, "vl", 2)) {
				ft = f512 | XB(XF_AVX512VL) | feats(tok[2] == '+' ?
				    tok + 3 : NULL);
				avx512form = 1;
			} else if (wi == 2) {
				ft = f512;
				avx512form = 1;
			} else {
				ft = feats(tok);
				avx512form = 0;
			}
			if (fe == E_F32 && !strcmp(sx, "ss") && wi)
				continue;
			parse_spec(f->spec, W, fe, &sp);
			fam_name(nm, sizeof nm, wprefix(W), f->stem, sx, W);
			if (xi_find_raw(nm) >= 0)
				continue;
			id = I.n;
			xi_add(nm, ft, &sp, f->sem, NULL, efloat(fe) &&
			    fe == E_F64 && f->clspd ? f->clspd : f->cls, f->fa,
			    f->imax, NULL);
			/* AVX-512 masked forms (the 512-bit form must exist) */
			if (!(f->fl & (FM_MASK | FM_KMASK)) || !strcmp(isa[2], "-"))
				continue;
			{
				xfeat mf = f512 | (wi < 2 ? XB(XF_AVX512VL) : 0) |
				    (avx512form ? ft : 0);
				struct ispec ms;
				char mn[112];
				int j, kind;
				for (kind = 0; kind < 3; kind++) {
					int mm;
					memset(&ms, 0, sizeof ms);
					ms.rk = sp.rk;
					ms.re = sp.re;
					ms.rbits = sp.rbits;
					if (f->fl & FM_KMASK) {
						if (kind)
							break;
						mm = MM_KMASK;
						ms.ok[0] = OK_K;
						ms.oe[0] = sp.re;
						ms.obits[0] = sp.rbits;
						for (j = 0; j < sp.nop; j++) {
							ms.ok[j + 1] = sp.ok[j];
							ms.oe[j + 1] = sp.oe[j];
							ms.obits[j + 1] = sp.obits[j];
						}
						ms.nop = (uint8_t)(sp.nop + 1);
						fam_name(mn, sizeof mn, wprefix(W), f->stem, sx,
						    W);
						snprintf(mn, sizeof mn, "%smask_%s", wprefix(W),
						    nm + strlen(wprefix(W)));
					} else {
						int lanes = sp.rbits / ebits[sp.re];
						int mb = lanes <= 8 ? 8 : lanes;
						uint8_t ke = mb == 8 ? E_U8 : mb == 16 ? E_U16 :
						    mb == 32 ? E_U32 : E_U64;
						if (kind == 2 && !(f->fl & FM_MASK3))
							break;
						if (sp.rk != OK_V)
							break;
						if (kind == 0) {
							mm = f->fl & FM_MASKA ? MM_MASKA : MM_MASK;
							j = 0;
							if (mm == MM_MASK) {
								ms.ok[j] = OK_V;
								ms.oe[j] = sp.re;
								ms.obits[j++] = sp.rbits;
							} else {
								ms.ok[j] = sp.ok[0];
								ms.oe[j] = sp.oe[0];
								ms.obits[j++] = sp.obits[0];
							}
							ms.ok[j] = OK_K;
							ms.oe[j] = ke;
							ms.obits[j++] = (uint16_t)mb;
							{
								int s0 = mm == MM_MASKA ? 1 : 0, q;
								for (q = s0; q < sp.nop; q++) {
									ms.ok[j] = sp.ok[q];
									ms.oe[j] = sp.oe[q];
									ms.obits[j++] = sp.obits[q];
								}
							}
							ms.nop = (uint8_t)j;
							snprintf(mn, sizeof mn, "%smask_%s",
							    wprefix(W), nm + strlen(wprefix(W)));
						} else if (kind == 1) {
							mm = MM_MASKZ;
							ms.ok[0] = OK_K;
							ms.oe[0] = ke;
							ms.obits[0] = (uint16_t)mb;
							for (j = 0; j < sp.nop; j++) {
								ms.ok[j + 1] = sp.ok[j];
								ms.oe[j + 1] = sp.oe[j];
								ms.obits[j + 1] = sp.obits[j];
							}
							ms.nop = (uint8_t)(sp.nop + 1);
							snprintf(mn, sizeof mn, "%smaskz_%s",
							    wprefix(W), nm + strlen(wprefix(W)));
						} else {
							mm = MM_MASK3;
							for (j = 0; j < sp.nop; j++) {
								ms.ok[j] = sp.ok[j];
								ms.oe[j] = sp.oe[j];
								ms.obits[j] = sp.obits[j];
							}
							ms.ok[j] = OK_K;
							ms.oe[j] = ke;
							ms.obits[j++] = (uint16_t)mb;
							ms.nop = (uint8_t)j;
							snprintf(mn, sizeof mn, "%smask3_%s",
							    wprefix(W), nm + strlen(wprefix(W)));
						}
					}
					if (xi_find_raw(mn) >= 0)
						continue;
					xi_add(mn, mf, &ms, NULL, NULL, I.cls[id], I.fl[id],
					    I.imax[id], NULL);
					I.base[I.n - 1] = id;
					I.mmode[I.n - 1] = (uint8_t)mm;
				}
			}
		}
	}
}

static int intr_ready;

void
x86_intr_init(void)
{
	int k;

	if (intr_ready)
		return;
	intr_ready = 1;
	x86_init();
	for (k = 0; xrows[k].name; k++) {
		struct ispec sp;
		parse_spec(xrows[k].spec, 128, E_I32, &sp);
		xi_add(xrows[k].name, feats(xrows[k].isa[0] ? xrows[k].isa : NULL),
		    &sp, xrows[k].sem, xrows[k].sem2, xrows[k].cls, xrows[k].fa,
		    xrows[k].imax, NULL);
	}
	for (k = 0; fams[k].stem; k++)
		gen_family(&fams[k]);
}

/* ---- showing an intrinsic ---- */

static const char *
ctype_v(int bits, int e)
{
	static char b[4][16];
	static int r;
	const char *suf = e == E_F32 ? "" : e == E_F64 ? "d" : "i";

	r = (r + 1) & 3;
	snprintf(b[r], sizeof b[r], "__m%d%s", bits, suf);
	return b[r];
}

static const char *
ctype_s(int e)
{
	static const char *const t[E_N] = {"char", "unsigned char", "short",
		"unsigned short", "int", "unsigned int", "long long",
		"unsigned long long", "float", "double"};
	return t[e];
}

static const char *
ctype_k(int bits)
{
	return bits <= 8 ? "__mmask8" : bits <= 16 ? "__mmask16" : bits <= 32 ?
	    "__mmask32" : "__mmask64";
}

static void
sig_text(struct sbuf *b, int id)
{
	int k;

	if (I.rk[id] == OK_V)
		sb_printf(b, "%s ", ctype_v(I.rbits[id], I.re[id]));
	else if (I.rk[id] == OK_K)
		sb_printf(b, "%s ", ctype_k(I.rbits[id]));
	else
		sb_printf(b, "%s ", ctype_s(I.re[id]));
	sb_printf(b, "%s(", I.name[id]);
	for (k = 0; k < I.nop[id]; k++) {
		const char *t;
		switch (I.ok[id][k]) {
		case OK_V: t = ctype_v(I.obits[id][k], I.oe[id][k]); break;
		case OK_K: t = ctype_k(I.obits[id][k]); break;
		case OK_I: t = "const int"; break;
		case OK_P: t = "void *"; break;
		default: t = ctype_s(I.oe[id][k]);
		}
		sb_printf(b, "%s%s %c", k ? ", " : "", t, 'a' + k);
	}
	sb_puts(b, ")");
}

/* display/eval variables: operand k lane l -> slot k*64 + l, "a3" */
static nid
var_lane(void *ctx, int k, int lane)
{
	int id = *(int *)ctx;
	char nm[16];
	ty_t t;

	(void)ctx;
	if (I.ok[id][k] == OK_K)
		t = TY(K_UINT, I.obits[id][k] <= 8 ? 8 : I.obits[id][k]);
	else
		t = ety(I.oe[id][k]);
	if (I.ok[id][k] == OK_V)
		snprintf(nm, sizeof nm, "%c%d", 'a' + k, lane);
	else
		snprintf(nm, sizeof nm, "%c", 'a' + k);
	return mk_var((uint32_t)(k * 64 + lane), sym_cstr(nm), t);
}

static int
has_imm(int id)
{
	int k;

	for (k = 0; k < I.nop[id]; k++)
		if (I.ok[id][k] == OK_I)
			return 1;
	return 0;
}

static void
show_lane(int id, int lane, int64_t imm)
{
	int ctx = id;
	nid e = xi_lane(id, lane, imm, var_lane, &ctx);
	const char *s = shared_str(e, 600);

	printf("    r%d = %s\n", lane, s);
	rep_obj(NULL);
	rep_int("lane", lane);
	rep_str("expr", s);
	rep_end();
}

void
cmd_intrinsic(const char *name, int ua)
{
	int id = xi_find(name), u, lanes;
	struct sbuf b = {0};
	int64_t imm = 0x1b;

	if (id < 0) {
		/* suggest by substring */
		uint32_t k;
		int n = 0;
		for (k = 0; k < I.n && n < 8; k++)
			if (strstr(I.name[k], name))
				sb_printf(&b, "%s%s", n++ ? ", " : "", I.name[k]);
		fail("unknown intrinsic %s%s%s", name, n ? "; similar: " : "",
		    sb_str(&b));
	}
	rep_begin("intrinsic");
	sig_text(&b, id);
	printf("%s\n", sb_str(&b));
	rep_str("name", I.name[id]);
	rep_str("signature", sb_str(&b));
	sb_clear(&b);
	xfeat_text(&b, I.feat[id]);
	printf("  needs: %s\n", sb_str(&b));
	rep_str("features", sb_str(&b));
	printf("  class: %s (%s)\n", xc_name[I.cls[id]], xc_desc[I.cls[id]]);
	rep_str("class", xc_name[I.cls[id]]);
	if (I.fl[id] & FA_MEM)
		printf("  memory: reads or writes memory; no lane semantics, "
		    "costs only\n");
	else {
		lanes = rlanes(id);
		if (I.mmode[id]) {
			static const char *const mt[] = {"", "r = k[i] ? base : src (the first operand)", "r = k[i] ? base : 0",
			    "r = k[i] ? base : a", "r = k[i] ? base : c",
			    "r = base & k"};
			printf("  semantics: %s, base %s: %s\n", mt[I.mmode[id]],
			    I.name[I.base[id]], I.sem[I.base[id]]);
			rep_str("semantics", I.sem[I.base[id]]);
			rep_str("masking", mt[I.mmode[id]]);
			rep_str("base", I.name[I.base[id]]);
		} else {
			printf("  semantics (each lane i of %d): %s\n", lanes,
			    I.sem[id]);
			rep_str("semantics", I.sem[id]);
			if (I.sem2[id]) {
				printf("  second result (through the pointer): %s\n",
				    I.sem2[id]);
				rep_str("semantics_ptr", I.sem2[id]);
			}
		}
		if (I.fl[id] & (FA_APPROX12 | FA_APPROX14))
			printf("  approximate: relative error at most %s; the "
			    "semantics is the exact value\n", I.fl[id] & FA_APPROX12 ?
			    "1.5 * 2^-12" : "2^-14");
		if (has_imm(id))
			printf("  lanes with imm = 0x1b (operand lanes named a0, "
			    "b3 ...):\n");
		else
			printf("  lanes (operand lanes named a0, b3 ...):\n");
		rep_list("lanes");
		show_lane(id, 0, imm);
		if (lanes > 1)
			show_lane(id, lanes - 1, imm);
		rep_end();
	}
	printf("  cost       lat   rtp uops  ports\n");
	rep_list("costs");
	for (u = 0; u < UA_N; u++) {
		const struct xcost *c;
		if (ua >= 0 && u != ua)
			continue;
		if ((I.feat[id] & uarchs[u].feat) != I.feat[id]) {
			if (ua >= 0)
				printf("  %-8s not available (lacks a feature)\n",
				    uarchs[u].name);
			continue;
		}
		c = xcost_get(u, I.cls[id], xi_vw(id));
		if (!c)
			continue;
		sb_clear(&b);
		xports_text(&b, u, c->ports);
		printf("  %-8s %5g %5g %4d  %s\n", uarchs[u].name, (double)c->lat,
		    (double)c->rtp, c->uops, sb_str(&b));
		rep_obj(NULL);
		rep_str("model", uarchs[u].name);
		rep_num("latency", c->lat);
		rep_num("rthroughput", c->rtp);
		rep_int("uops", c->uops);
		rep_str("ports", sb_str(&b));
		rep_end();
	}
	rep_end();
	free(b.p);
}

void
cmd_intrinsics(const char *filter, int ua)
{
	uint32_t k;
	int nf[XF_N] = {0}, shown = 0, total = 0, f;
	int fk = filter ? xfeat_find(filter) : -1;

	x86_intr_init();
	rep_begin("intrinsics");
	rep_list("intrinsics");
	for (k = 0; k < I.n; k++) {
		if (filter && fk < 0 && !strstr(I.name[k], filter) &&
		    strcmp(xc_name[I.cls[k]], filter))
			continue;
		if (fk >= 0 && !(I.feat[k] & XB(fk)))
			continue;
		if (ua >= 0 && (I.feat[k] & uarchs[ua].feat) != I.feat[k])
			continue;
		total++;
		for (f = 0; f < XF_N; f++)
			if (I.feat[k] & XB(f))
				nf[f]++;
		if (filter && shown < 200) {
			printf("  %s\n", I.name[k]);
			rep_str(NULL, I.name[k]);
			shown++;
		}
	}
	rep_end();
	if (filter && total > shown)
		printf("  ... %d more\n", total - shown);
	printf("%d intrinsic%s%s%s%s%s", total, total == 1 ? "" : "s",
	    filter ? " matching " : "", filter ? filter : "",
	    ua >= 0 ? " available on " : "", ua >= 0 ? uarchs[ua].name : "");
	printf("; by feature:");
	for (f = 0; f < XF_N; f++)
		if (nf[f])
			printf(" %s %d", xfeat_name[f], nf[f]);
	printf("\n");
	rep_int("count", total);
	rep_obj("by_feature");
	for (f = 0; f < XF_N; f++)
		if (nf[f])
			rep_int(xfeat_name[f], nf[f]);
	rep_end();
}

/* ---- differential testing against the C compiler ---- */

#define CK_OPB 64		/* bytes per operand slot */
#define CK_PT (MAXOP * CK_OPB)	/* input bytes per point */
#define CK_OUT 128		/* output bytes per point: result, pointer result */

static const int imm_list[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 27,
	31, 32, 33, 63, 64, 78, 85, 127, 128, 147, 170, 177, 200, 216, 228,
	255, -1};

struct ck_test {
	int id;
	int imm;
	int line;		/* first line of its function in the C file */
	int dropped;		/* 1 rejected by cc, 2 not runnable here */
};

static uint64_t ck_rng;

static uint64_t
rnd64(void)
{
	ck_rng ^= ck_rng << 13;
	ck_rng ^= ck_rng >> 7;
	ck_rng ^= ck_rng << 17;
	return ck_rng;
}

/* one element of type e: random bits or an edge value */
static uint64_t
gen_elem(int e)
{
	uint64_t r = rnd64(), m = ebits[e] == 64 ? ~0ULL : (1ULL << ebits[e]) - 1;
	int pick = (int)(rnd64() % 16);

	if (e == E_F32) {
		static const float fv[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f,
			2.5f, -2.5f, 3.5f, 1e10f, -1e10f, 3e9f, -3e9f, 2147483648.0f,
			-2147483648.0f, 1e-40f, 65535.5f};
		float f;
		uint32_t u;
		if (pick < 6) {
			f = fv[rnd64() % (sizeof fv / sizeof fv[0])];
			memcpy(&u, &f, 4);
			return u;
		}
		if (pick == 6)
			return rnd64() % 3 == 0 ? 0x7fc00000u : rnd64() % 2 ?
			    0x7f800000u : 0xff800000u;
		if (pick < 12) {	/* moderate values with fractions */
			f = (float)((double)(int32_t)(r & 0xffffff) / 256.0);
			memcpy(&u, &f, 4);
			return u;
		}
		/* random finite bits */
		u = (uint32_t)r;
		if (((u >> 23) & 0xff) == 0xff)
			u &= ~(1u << 30);
		return u;
	}
	if (e == E_F64) {
		static const double dv[] = {0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5,
			-2.5, 1e20, -1e20, 3e9, -3e9, 2147483648.0, -2147483649.0,
			9.3e18, 1e-310, 0.1};
		double d;
		uint64_t u;
		if (pick < 6) {
			d = dv[rnd64() % (sizeof dv / sizeof dv[0])];
			memcpy(&u, &d, 8);
			return u;
		}
		if (pick == 6)
			return rnd64() % 3 == 0 ? 0x7ff8000000000000ULL : rnd64() % 2 ?
			    0x7ff0000000000000ULL : 0xfff0000000000000ULL;
		if (pick < 12) {
			d = (double)(int64_t)(r & 0xffffffffffLL) / 65536.0 -
			    8388608.0;
			memcpy(&u, &d, 8);
			return u;
		}
		u = r;
		if (((u >> 52) & 0x7ff) == 0x7ff)
			u &= ~(1ULL << 62);
		return u;
	}
	switch (pick) {
	case 0: return 0;
	case 1: return m;			/* -1 */
	case 2: return 1;
	case 3: return (m >> 1) + 1;		/* min */
	case 4: return m >> 1;			/* max */
	case 5: return rnd64() % 72;		/* small: shift counts, indices */
	case 6: return (m >> 1) - rnd64() % 4;
	case 7: return m - rnd64() % 4;
	}
	return r & m;
}

static void
put_elem(unsigned char *p, int bits, uint64_t v)
{
	memcpy(p, &v, (size_t)bits / 8);	/* little endian */
}

static uint64_t
get_bits(const unsigned char *p, int bits)
{
	uint64_t v = 0;

	memcpy(&v, p, (size_t)bits / 8);
	return v;
}

static void
gen_point(int id, unsigned char *pt)
{
	int k, j;

	memset(pt, 0, CK_PT);
	for (k = 0; k < I.nop[id]; k++) {
		unsigned char *o = pt + k * CK_OPB;
		switch (I.ok[id][k]) {
		case OK_V: {
			int e = I.oe[id][k], n = I.obits[id][k] / ebits[e];
			for (j = 0; j < n; j++)
				put_elem(o + j * ebits[e] / 8, ebits[e], gen_elem(e));
			break;
		}
		case OK_S:
			put_elem(o, ebits[I.oe[id][k]], gen_elem(I.oe[id][k]));
			break;
		case OK_K: {
			uint64_t v = rnd64();
			int pick = (int)(rnd64() % 8);
			if (pick == 0)
				v = 0;
			if (pick == 1)
				v = ~0ULL;
			put_elem(o, I.obits[id][k], v);
			break;
		}
		}
	}
}

static void
c_test_fn(struct sbuf *b, int t, const struct ck_test *ct)
{
	int id = ct->id, k;
	const char *rt;

	sb_printf(b, "__attribute__((noinline)) void t%d(const unsigned char "
	    "*in, unsigned char *out) {\n", t);
	for (k = 0; k < I.nop[id]; k++) {
		const char *ty = NULL;
		switch (I.ok[id][k]) {
		case OK_V: ty = ctype_v(I.obits[id][k], I.oe[id][k]); break;
		case OK_S: ty = ctype_s(I.oe[id][k]); break;
		case OK_K: ty = ctype_k(I.obits[id][k]); break;
		}
		if (ty)
			sb_printf(b, "  %s o%d; memcpy(&o%d, in + %d, sizeof o%d);\n",
			    ty, k, k, k * CK_OPB, k);
		else if (I.ok[id][k] == OK_P)
			sb_printf(b, "  %s o%d = 0;\n", ctype_s(I.oe[id][k]), k);
	}
	rt = I.rk[id] == OK_V ? ctype_v(I.rbits[id], I.re[id]) :
	    I.rk[id] == OK_K ? ctype_k(I.rbits[id]) : ctype_s(I.re[id]);
	sb_printf(b, "  %s r = %s(", rt, I.name[id]);
	for (k = 0; k < I.nop[id]; k++) {
		if (I.ok[id][k] == OK_I)
			sb_printf(b, "%s%d", k ? ", " : "", ct->imm);
		else if (I.ok[id][k] == OK_P)
			sb_printf(b, "%s&o%d", k ? ", " : "", k);
		else
			sb_printf(b, "%so%d", k ? ", " : "", k);
	}
	sb_puts(b, ");\n  memcpy(out, &r, sizeof r);\n");
	if (I.fl[id] & FA_PTROUT)
		sb_printf(b, "  memcpy(out + 64, &o%d, sizeof o%d);\n",
		    I.nop[id] - 1, I.nop[id] - 1);
	sb_puts(b, "}\n");
}

static int
count_lines(const char *s)
{
	int n = 0;

	for (; *s; s++)
		n += *s == '\n';
	return n;
}

static int
run_sh(const char *cmd, struct sbuf *out)
{
	FILE *f = popen(cmd, "r");
	char buf[4096];
	size_t n;

	if (!f)
		return -1;
	while ((n = fread(buf, 1, sizeof buf, f)) > 0)
		if (out && out->n < (4u << 20)) {
			size_t k;
			for (k = 0; k < n; k++)
				sb_putc(out, buf[k]);
		}
	return pclose(f);
}

static xfeat
host_feats(void)
{
	static xfeat f;
	static int done;
	struct sbuf o = {0};
	int k;

	if (done)
		return f;
	done = 1;
	run_sh("cc -march=native -dM -E - </dev/null 2>/dev/null", &o);
	for (k = 0; k < XF_N; k++) {
		char m[48];
		snprintf(m, sizeof m, "#define %s ", xfeat_macro[k]);
		if (o.p && strstr(o.p, m))
			f |= XB(k);
	}
	free(o.p);
	return f;
}

#define ALLM "-msse4.2 -mpopcnt -mlzcnt -mbmi -mbmi2 -mavx2 -mfma -mf16c " \
	"-mpclmul -maes -madx -mgfni -mavx512f -mavx512bw -mavx512dq " \
	"-mavx512vl -mavx512cd -mavx512vbmi -mavx512vpopcntdq " \
	"-mavx512bitalg -mvpclmulqdq -mvaes"

/*
 * In the LLVM IR of the AVX-512 tests, a function that calls an x86
 * target intrinsic (llvm.x86.*) cannot run on a core without the
 * feature: its body becomes `ret void` and it is marked not runnable.
 * Generic vector IR (what clang's headers lower most intrinsics to)
 * compiles for the host's AVX2 after the target features are replaced,
 * and so do the host's own intrinsics (llvm.x86.avx2.* ...).
 */
static void
strip_ir(const char *ll, struct sbuf *out, struct ck_test *ct, int nt,
    xfeat host)
{
	const char *s = ll;

	while (*s) {
		const char *e = strchr(s, '\n');
		size_t n = e ? (size_t)(e - s) + 1 : strlen(s);
		int t;
		if (!strncmp(s, "define ", 7) && sscanf(strstr(s, "@t") ?
		    strstr(s, "@t") : "", "@t%d(", &t) == 1 && t >= 0 && t < nt) {
			const char *end = strstr(s, "\n}\n"), *x;
			int bad = 0;
			if (!end)
				break;
			for (x = s; x < end; x++)
				if (!strncmp(x, "@llvm.x86.", 10) &&
				    (I.feat[ct[t].id] & ~host)) {
					bad = 1;
					break;
				}
			if (bad) {
				const char *br = strchr(s, '{');
				ct[t].dropped = 2;
				for (; s <= br; s++)
					sb_putc(out, *s);
				sb_puts(out, "\n  ret void\n}\n");
				s = end + 3;
				continue;
			}
			for (; s < end + 3; s++)
				sb_putc(out, *s);
			continue;
		}
		if (strstr(s, "\"target-features\"=") && strstr(s, "\"target-features\"=") <
		    s + n) {
			const char *a = strstr(s, "\"target-features\"=\"");
			const char *b = strchr(a + 19, '"');
			for (; s < a; s++)
				sb_putc(out, *s);
			sb_puts(out, "\"target-features\"=\"+avx2,+avx,+fma,+bmi,+bmi2,"
			    "+popcnt,+lzcnt,+sse4.2,+crc32,+movbe,+aes,+pclmul,+f16c,+adx\"");
			s = b + 1;
			continue;
		}
		if (!strncmp(s, "attributes ", 11) && strstr(s, "\"target-cpu\"")) {
			/* drop the cpu: the features above are what counts */
			const char *a = strstr(s, "\"target-cpu\"=\"");
			if (a && a < s + n) {
				const char *b = strchr(a + 14, '"');
				for (; s < a; s++)
					sb_putc(out, *s);
				s = b + 1;
				continue;
			}
		}
		while (n--)
			sb_putc(out, *s++);
	}
}

static volatile int ck_semerr;

static nid
ck_var(void *c, int k, int lane)
{
	return var_lane(c, k, lane);
}

/* the value of operand slot (k, lane) for point p, as the evaluator wants */
static void
load_inputs(int id, const unsigned char *pt, int64_t *in_i, double *in_r,
    int p)
{
	int k, j;

	for (k = 0; k < I.nop[id]; k++) {
		const unsigned char *o = pt + k * CK_OPB;
		int n = 1, e = I.oe[id][k], bits;
		if (I.ok[id][k] == OK_V)
			n = I.obits[id][k] / ebits[e];
		bits = I.ok[id][k] == OK_K ? I.obits[id][k] : ebits[e];
		for (j = 0; j < n; j++) {
			size_t slot = (size_t)(k * 64 + j) * CH + (size_t)p;
			uint64_t v = get_bits(o + j * bits / 8, bits);
			if (I.ok[id][k] != OK_K && efloat(e)) {
				if (e == E_F32) {
					float f;
					uint32_t u = (uint32_t)v;
					memcpy(&f, &u, 4);
					in_r[slot] = f;
				} else {
					double d;
					memcpy(&d, &v, 8);
					in_r[slot] = d;
				}
			} else if (I.ok[id][k] != OK_K && !(e & 1) && bits < 64 &&
			    (v >> (bits - 1)) & 1)
				in_i[slot] = (int64_t)(v | ~((1ULL << bits) - 1));
			else
				in_i[slot] = (int64_t)v;
		}
	}
}

/*
 * The AVX-512 cross-check: an AVX-512 intrinsic that could not run here
 * (it calls an llvm.x86.avx512.* builtin) against the narrower form of
 * the same name (_mm512_X against _mm256_X, _mm256_X against _mm_X)
 * that did run and agree.  Lane i of the wide form, in half h = i / n,
 * must equal lane i - h n of the narrow form applied to half h of each
 * vector operand (and the mask bits of that half).  Equal node DAGs
 * (hash-consed: the same id) are a proof; otherwise the two are
 * evaluated on random and edge points: a difference means the wide form
 * moves data across halves (or a catalogue error), agreement without
 * equal DAGs is only sampled.
 */
struct half_ctx {
	int wide, narrow, h;
};

static nid
half_fn(void *c, int k, int lane)
{
	struct half_ctx *hc = c;
	int n = hc->narrow, e = I.oe[n][k];

	switch (I.ok[n][k]) {
	case OK_V:	/* a full-width operand: half h; a same-width one
			 * (a shift count in an xmm): as it is */
		if (I.obits[hc->wide][k] == I.obits[n][k])
			return var_lane(&hc->wide, k, lane);
		return var_lane(&hc->wide, k, lane + hc->h *
		    lanes_of(I.obits[n][k], e));
	case OK_K: {
		int nb = I.obits[n][k] <= 8 ? 8 : I.obits[n][k];
		nid v = var_lane(&hc->wide, k, 0);
		if (hc->h)
			v = mk_bin(O_SHR, v, mk_const_i(N.ty[v],
			    (int64_t)hc->h * rlanes(n)));
		return mk_cast(TY(K_UINT, nb), v);
	}
	}
	return var_lane(&hc->wide, k, 0);
}

static const char *
narrow_name(const char *w, char *buf, size_t nb)
{
	if (!strncmp(w, "_mm512_", 7))
		snprintf(buf, nb, "_mm256_%s", w + 7);
	else if (!strncmp(w, "_mm256_", 7))
		snprintf(buf, nb, "_mm_%s", w + 7);
	else
		return NULL;
	return buf;
}

/* same operands, every vector twice as wide (but an xmm count operand
 * after the first) */
static int
half_shape(int w, int n)
{
	int k;

	if (I.nop[w] != I.nop[n] || I.rk[w] != OK_V || I.rk[n] != OK_V ||
	    I.re[w] != I.re[n] || I.rbits[w] != 2 * I.rbits[n] ||
	    I.imax[w] != I.imax[n])
		return 0;
	for (k = 0; k < I.nop[w]; k++) {
		if (I.ok[w][k] != I.ok[n][k] || I.oe[w][k] != I.oe[n][k])
			return 0;
		if (I.ok[w][k] == OK_V && I.obits[w][k] != 2 * I.obits[n][k] &&
		    (I.obits[w][k] != 128 || I.obits[n][k] != 128 || !k))
			return 0;
		if (I.ok[w][k] == OK_P)
			return 0;
	}
	return 1;
}

#define AVX512_BITS (XB(XF_AVX512F) | XB(XF_AVX512BW) | XB(XF_AVX512DQ) | \
	XB(XF_AVX512VL) | XB(XF_AVX512CD) | XB(XF_AVX512VBMI) | \
	XB(XF_AVX512VPOPCNTDQ) | XB(XF_AVX512BITALG))

/* 0 proved (equal DAGs), 1 agree on samples, 2 differ */
static int
cross_one(int w, int n, int64_t *in_i, double *in_r, int npts)
{
	const int *im;
	int verdict = 0, wctx = w;

	for (im = imm_list; *im >= 0; im++) {
		int lanes = rlanes(w), l, p, nroot = 0;
		nid roots[128];
		if (*im > I.imax[w] || (!has_imm(w) && im != imm_list))
			break;
		for (l = 0; l < lanes; l++) {
			struct half_ctx hc;
			nid a, b;
			hc.wide = w;
			hc.narrow = n;
			hc.h = l / rlanes(n);
			a = xi_lane(w, l, *im, var_lane, &wctx);
			b = xi_lane(n, l % rlanes(n), *im, half_fn, &hc);
			if (a != b) {
				roots[nroot++] = a;
				roots[nroot++] = b;
			}
		}
		if (nroot) {
			struct plan pl;
			unsigned char pt[CK_PT];
			if (verdict < 1)
				verdict = 1;
			memset(in_i, 0, (size_t)MAXOP * 64 * CH * sizeof *in_i);
			memset(in_r, 0, (size_t)MAXOP * 64 * CH * sizeof *in_r);
			for (p = 0; p < npts; p++) {
				gen_point(w, pt);
				load_inputs(w, pt, in_i, in_r, p);
			}
			plan_build(&pl, roots, (uint32_t)nroot, 0);
			plan_run(&pl, (uint32_t)npts, in_i, in_r);
			for (l = 0; l < nroot; l += 2) {
				uint32_t x = pl.root[l], y = pl.root[l + 1];
				for (p = 0; p < npts; p++) {
					double u = COLR(&pl, x)[p], v = COLR(&pl, y)[p];
					if ((COLF(&pl, x)[p] ^ COLF(&pl, y)[p]) & F_UNDEF)
						verdict = 2;
					else if (TY_FP(N.ty[roots[l]]) ? !(u == v ||
					    (u != u && v != v) ) || (u == 0 && v == 0 &&
					    signbit(u) != signbit(v)) :
					    COLI(&pl, x)[p] != COLI(&pl, y)[p])
						verdict = 2;
				}
			}
			plan_free(&pl);
		}
		if (verdict == 2)
			break;
	}
	return verdict;
}

/* idst per intrinsic: 1 a test agreed, 2 a test differed, 4 not run */
static void
ck_cross(const uint8_t *idst, const char *filter, int fk)
{
	int64_t *in_i = xcalloc((size_t)MAXOP * 64 * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXOP * 64 * CH, sizeof *in_r);
	uint32_t id;
	int nnot = 0, nproved = 0, nsampled = 0, ndiffer = 0, nnone = 0,
	    nmask = 0, shown = 0, nonone = 0;
	struct sbuf diff = {0}, none = {0};

	/* vst: 1 proved, 2 sampled, 3 differ, 5 no narrower form */
	uint8_t *vst = xcalloc(I.n, 1);
	int pass;

	for (pass = 0; pass < 2; pass++)
	for (id = 0; id < I.n; id++) {
		char nb[96];
		int n, v;
		if (!(I.feat[id] & AVX512_BITS) || idst[id] != 4)
			continue;
		if (filter && fk < 0 && !strstr(I.name[id], filter))
			continue;
		if (fk >= 0 && !(I.feat[id] & XB(fk)))
			continue;
		if (pass == 1) {
			/* sel(k bit i, base, src) over its base (xi_lane) */
			uint32_t b = I.base[id];
			if (!I.mmode[id])
				continue;
			nnot++;
			if (idst[b] == 1 || vst[b] == 1)
				nmask++;
			else {
				nnone++;
				if (nonone++ < 40)
					sb_printf(&none, " %s", I.name[id]);
			}
			continue;
		}
		if (I.mmode[id])
			continue;
		nnot++;
		if (!narrow_name(I.name[id], nb, sizeof nb) ||
		    (n = xi_find(nb)) < 0 || idst[n] != 1 || !half_shape(id, n)) {
			vst[id] = 5;
			nnone++;
			if (nonone++ < 40)
				sb_printf(&none, " %s", I.name[id]);
			continue;
		}
		ck_rng = 0x2545f4914f6cdd1dULL ^ id;
		{
			jmp_buf saved;
			memcpy(saved, fail_jmp, sizeof saved);
			if (setjmp(fail_jmp)) {
				memcpy(fail_jmp, saved, sizeof saved);
				v = 2;
			} else {
				v = cross_one((int)id, n, in_i, in_r, 64);
				memcpy(fail_jmp, saved, sizeof saved);
			}
		}
		vst[id] = (uint8_t)(v + 1);
		if (v == 0)
			nproved++;
		else if (v == 1)
			nsampled++;
		else {
			ndiffer++;
			if (shown++ < 40)
				sb_printf(&diff, " %s", I.name[id]);
		}
	}
	free(vst);
	{
		int nran = 0;
		for (id = 0; id < I.n; id++)
			if ((I.feat[id] & AVX512_BITS) && idst[id] == 1 &&
			    (!filter || (fk < 0 ? strstr(I.name[id], filter) != NULL :
			    (I.feat[id] & XB(fk)) != 0)))
				nran++;
		printf("  AVX-512: %d intrinsics ran as clang's header definitions "
		    "lowered to generic vector IR and executed as AVX2 code here: "
		    "checked against clang, not against AVX-512 hardware\n", nran);
		rep_int("avx512_ran_lowered", nran);
	}
	printf("  AVX-512 cross-check of the %d intrinsics that could not "
	    "run (wide form against the run-validated narrower form, half by "
	    "half):\n", nnot);
	printf("    %d proved lane by lane (equal DAGs), %d agree on 64 sampled "
	    "points only, %d differ (they move data across halves)%s\n",
	    nproved, nsampled, ndiffer, ndiffer ? ":" : "");
	if (ndiffer)
		printf("     %s%s\n", sb_str(&diff), ndiffer > 40 ? " ..." : "");
	printf("    %d mask/maskz forms whose base ran or is proved above: "
	    "sel(k bit i, base, src) by construction; the operand order of each "
	    "mask kind is checked by the tests that ran\n", nmask);
	printf("    %d without a run-validated narrower form (not validated):%s%s\n",
	    nnone, sb_str(&none), nnone > 40 ? " ..." : "");
	rep_int("avx512_not_run", nnot);
	rep_int("avx512_cross_proved", nproved);
	rep_int("avx512_cross_sampled", nsampled);
	rep_int("avx512_cross_differ", ndiffer);
	rep_int("avx512_mask_forms", nmask);
	rep_int("avx512_unvalidated", nnone);
	free(diff.p);
	free(none.p);
	free(in_i);
	free(in_r);
}

/* a mismatch's inputs: each operand's lane (or the scalar, the mask) */
static void
ck_show_inputs(int id, const unsigned char *pt, int lane)
{
	int k;

	printf("    inputs:");
	for (k = 0; k < I.nop[id]; k++) {
		const unsigned char *o = pt + k * CK_OPB;
		int e = I.oe[id][k], bits = ebits[e], j = 0;
		if (I.ok[id][k] == OK_K)
			bits = I.obits[id][k];
		else if (I.ok[id][k] == OK_V && lane < I.obits[id][k] / bits)
			j = lane;
		printf(" %c%s=0x%llx", 'a' + k, I.ok[id][k] == OK_V ? "[lane]" : "",
		    (unsigned long long)get_bits(o + j * bits / 8, bits));
	}
	printf("\n");
}

static int
same_val(ty_t t, int64_t vi, double vr, uint8_t fl, uint64_t cbits, int fa,
    double *relerr)
{
	if (fl & F_UNDEF)
		return 0;
	if (TY_FP(t)) {
		int w = TYW(t);
		double c;
		uint64_t ub = fbits(w, vr);
		if (w == 32) {
			float f;
			uint32_t u = (uint32_t)cbits;
			memcpy(&f, &u, 4);
			c = f;
		} else
			memcpy(&c, &cbits, 8);
		if (vr != vr && c != c)
			return 1;
		if (fa & (FA_APPROX12 | FA_APPROX14)) {
			double tol = fa & FA_APPROX12 ? 1.5 / 4096 : 1.0 / 16384;
			double lim = w == 32 ? 1.17549435e-38 : 2.2250738585072014e-308;
			if (c == vr)
				return 1;
			/* denormal inputs read as zero: 1/0 */
			if (isinf(c) && (isinf(vr) || fabs(vr) > 4.6e18))
				return 1;
			if (c == 0 && fabs(vr) < lim * 2)
				return 1;
			if (fabs(vr) < lim * 4 && fabs(c) < lim * 4)
				return 1;
			*relerr = fabs(c - vr) / fabs(vr);
			return *relerr <= tol;
		}
		return ub == (w == 32 ? (cbits & 0xffffffffu) : cbits);
	}
	{
		int w = TY_SIZED(t) ? TYW(t) : 64;
		uint64_t m = w == 64 ? ~0ULL : (1ULL << w) - 1;
		return ((uint64_t)vi & m) == (cbits & m);
	}
}

void
cmd_x86check(const char *filter, int npts, int avx512)
{
	xfeat host = host_feats(), allow;
	struct ck_test *ct = NULL;
	int nt = 0, capt = 0, round, t, k, fk = filter ? xfeat_find(filter) : -1;
	char dir[] = "/tmp/ucas-x86check-XXXXXX";
	char path[256], cmd[1024];
	struct sbuf src = {0}, out = {0};
	unsigned char *in = NULL, *res = NULL;
	int64_t *in_i;
	double *in_r;
	int nrej = 0, nrange = 0, nnorun = 0, nok = 0, nbad = 0, nnosem = 0, shownbad = 0;
	long cmp_lanes = 0;
	double worst_rel = 0;
	uint32_t id;
	uint8_t *idst = NULL;

	x86_intr_init();
	ck_semerr = 0;
	if (npts < 1)
		npts = 1;
	if (npts > CH)
		npts = CH;
	allow = avx512 ? ~0ULL : host;
	if (!mkdtemp(dir))
		fail("x86check: no temporary directory");
	/* the tests: each intrinsic, each immediate */
	for (id = 0; id < I.n; id++) {
		const int *im;
		if (filter && fk < 0 && !strstr(I.name[id], filter))
			continue;
		if (fk >= 0 && !(I.feat[id] & XB(fk)))
			continue;
		if ((I.feat[id] & allow) != I.feat[id])
			continue;
		if (I.fl[id] & FA_NOEXEC) {
			nnosem++;
			continue;
		}
		for (im = imm_list; *im >= 0; im++) {
			if (*im > I.imax[id] || (!has_imm(id) && im != imm_list))
				break;
			if (nt == capt) {
				capt = capt ? capt * 2 : 256;
				ct = xrealloc(ct, (size_t)capt * sizeof *ct);
			}
			ct[nt].id = (int)id;
			ct[nt].imm = *im;
			ct[nt].dropped = 0;
			nt++;
		}
	}
	if (!nt) {
		printf("x86check: no intrinsic matches%s%s\n", filter ? " " : "",
		    filter ? filter : "");
		rmdir(dir);
		free(ct);
		return;
	}
	/* write, compile; tests cc rejects are dropped and it compiles again */
	for (round = 0; round < 10; round++) {
		int ok;
		sb_clear(&src);
		sb_puts(&src, "#include <immintrin.h>\n#include <x86intrin.h>\n"
		    "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n");
		for (t = 0; t < nt; t++) {
			ct[t].line = count_lines(sb_str(&src)) + 1;
			if (ct[t].dropped == 1 || ct[t].dropped == 3) {
				sb_printf(&src, "void t%d(const unsigned char *in, unsigned "
				    "char *out) { (void)in; (void)out; }\n", t);
				continue;
			}
			c_test_fn(&src, t, &ct[t]);
		}
		sb_printf(&src, "typedef void (*tf)(const unsigned char *, unsigned "
		    "char *);\nstatic tf tab[%d] = {", nt);
		for (t = 0; t < nt; t++)
			sb_printf(&src, "%st%d", t ? ", " : "", t);
		sb_printf(&src, "};\nint main(int argc, char **argv) {\n"
		    "  int np = %d, nt = %d, t, p; size_t pb = %d;\n"
		    "  FILE *fi = fopen(argv[1], \"rb\"), *fo = fopen(argv[2], \"wb\");\n"
		    "  unsigned char *in = malloc(pb * np * nt), out[%d];\n"
		    "  if (!fi || !fo || fread(in, pb, (size_t)np * nt, fi) != "
		    "(size_t)np * nt) return 2;\n"
		    "  for (t = 0; t < nt; t++) for (p = 0; p < np; p++) {\n"
		    "    memset(out, 0, sizeof out);\n"
		    "    tab[t](in + pb * ((size_t)t * np + p), out);\n"
		    "    fwrite(out, sizeof out, 1, fo); }\n"
		    "  fclose(fo); (void)argc; return 0; }\n", npts, nt, CK_PT, CK_OUT);
		snprintf(path, sizeof path, "%s/t.c", dir);
		{
			FILE *f = fopen(path, "w");
			if (!f)
				fail("x86check: cannot write %s", path);
			fwrite(src.p, 1, src.n, f);
			fclose(f);
		}
		sb_clear(&out);
		if (avx512) {
			snprintf(cmd, sizeof cmd, "cd %s && cc -O1 -ferror-limit=0 " ALLM " -S -emit-llvm "
			    "-o t.ll t.c 2>&1", dir);
			ok = run_sh(cmd, &out) == 0;
		} else {
			snprintf(cmd, sizeof cmd, "cd %s && cc -O1 -ferror-limit=0 -march=native -o t t.c "
			    "2>&1", dir);
			ok = run_sh(cmd, &out) == 0;
		}
		if (ok)
			break;
		/* map the errors to tests */
		{
			const char *s = sb_str(&out);
			int any = 0;
			while ((s = strstr(s, "t.c:")) != NULL) {
				int line = atoi(s + 4), lo = 0, hi = nt - 1;
				s += 4;
				if (!strstr(s, "error") || strstr(s, "error") >
				    (strchr(s, '\n') ? strchr(s, '\n') : s + strlen(s)))
					continue;
				while (lo < hi) {
					int mid = (lo + hi + 1) / 2;
					if (ct[mid].line <= line)
						lo = mid;
					else
						hi = mid - 1;
				}
				if (ct[lo].dropped != 1 && ct[lo].dropped != 3) {
					const char *nl = strchr(s, '\n');
					const char *rv = strstr(s, "argument value");
					/* 1: an immediate out of its range; 3: anything else */
					ct[lo].dropped = rv && (!nl || rv < nl) ? 1 : 3;
					any = 1;
				}
			}
			if (!any) {
				printf("x86check: cc failed:\n%.2000s\n", sb_str(&out));
				goto done;
			}
		}
	}
	if (round == 10) {
		printf("x86check: cc kept failing\n");
		goto done;
	}
	if (avx512) {
		/* strip the AVX-512 target features from the IR, run on AVX2 */
		struct sbuf ll = {0}, st = {0};
		FILE *f;
		long sz;
		snprintf(path, sizeof path, "%s/t.ll", dir);
		f = fopen(path, "r");
		if (!f)
			fail("x86check: no IR");
		fseek(f, 0, SEEK_END);
		sz = ftell(f);
		fseek(f, 0, SEEK_SET);
		ll.p = xmalloc((size_t)sz + 1);
		ll.n = fread(ll.p, 1, (size_t)sz, f);
		ll.p[ll.n] = 0;
		ll.cap = (size_t)sz + 1;
		fclose(f);
		strip_ir(ll.p, &st, ct, nt, host);
		snprintf(path, sizeof path, "%s/u.ll", dir);
		f = fopen(path, "w");
		fwrite(st.p, 1, st.n, f);
		fclose(f);
		free(ll.p);
		free(st.p);
		sb_clear(&out);
		snprintf(cmd, sizeof cmd, "cd %s && cc -O1 -march=native -o t u.ll "
		    "2>&1", dir);
		if (run_sh(cmd, &out) != 0) {
			printf("x86check: the AVX-512 tests did not compile for this "
			    "host after lowering:\n%.1500s\n", sb_str(&out));
			goto done;
		}
	}
	/* inputs */
	ck_rng = 0x9e3779b97f4a7c15ULL;
	in = xcalloc((size_t)nt * npts, CK_PT);
	for (t = 0; t < nt; t++)
		for (k = 0; k < npts; k++)
			gen_point(ct[t].id, in + ((size_t)t * npts + k) * CK_PT);
	snprintf(path, sizeof path, "%s/in.bin", dir);
	{
		FILE *f = fopen(path, "wb");
		fwrite(in, CK_PT, (size_t)nt * npts, f);
		fclose(f);
	}
	snprintf(cmd, sizeof cmd, "cd %s && ./t in.bin out.bin 2>&1", dir);
	sb_clear(&out);
	if (run_sh(cmd, &out) != 0) {
		printf("x86check: the test program failed: %.500s\n", sb_str(&out));
		goto done;
	}
	res = xcalloc((size_t)nt * npts, CK_OUT);
	snprintf(path, sizeof path, "%s/out.bin", dir);
	{
		FILE *f = fopen(path, "rb");
		if (!f || fread(res, CK_OUT, (size_t)nt * npts, f) !=
		    (size_t)nt * npts)
			fail("x86check: short output");
		fclose(f);
	}
	/* compare with the lane semantics, evaluated */
	in_i = xcalloc((size_t)MAXOP * 64 * CH, sizeof *in_i);
	in_r = xcalloc((size_t)MAXOP * 64 * CH, sizeof *in_r);
	idst = xcalloc(I.n, 1);
	for (t = 0; t < nt; t++) {
		int id2 = ct[t].id, lanes, nroot, p, l, bad = 0;
		nid roots[65];
		struct plan pl;
		int ctx = id2;
		if (ct[t].dropped == 1) {
			nrange++;
			continue;
		}
		if (ct[t].dropped == 3) {
			nrej++;
			continue;
		}
		if (ct[t].dropped == 2) {
			nnorun++;
			idst[ct[t].id] |= 4;
			continue;
		}
		lanes = rlanes(id2);
		nroot = 0;
		{
			jmp_buf saved;
			memcpy(saved, fail_jmp, sizeof saved);
			if (setjmp(fail_jmp)) {
				memcpy(fail_jmp, saved, sizeof saved);
				printf("  ^ in the semantics of %s\n", I.name[ct[t].id]);
				ck_semerr++;
				continue;
			}
			for (l = 0; l < lanes; l++)
				roots[nroot++] = xi_lane(id2, l, ct[t].imm, ck_var, &ctx);
			if (I.fl[id2] & FA_PTROUT)
				roots[nroot++] = xi_lane2(id2, 0, ct[t].imm, ck_var, &ctx);
			memcpy(fail_jmp, saved, sizeof saved);
		}
		memset(in_i, 0, (size_t)MAXOP * 64 * CH * sizeof *in_i);
		memset(in_r, 0, (size_t)MAXOP * 64 * CH * sizeof *in_r);
		for (p = 0; p < npts; p++)
			load_inputs(id2, in + ((size_t)t * npts + p) * CK_PT, in_i,
			    in_r, p);
		plan_build(&pl, roots, (uint32_t)nroot, 0);
		plan_run(&pl, (uint32_t)npts, in_i, in_r);
		for (p = 0; p < npts && !bad; p++) {
			const unsigned char *o = res + ((size_t)t * npts + p) * CK_OUT;
			for (l = 0; l < nroot && !bad; l++) {
				ty_t ty = N.ty[roots[l]];
				int w = TY_SIZED(ty) || TY_FP(ty) ? TYW(ty) : 64, off;
				uint64_t cb;
				double rel = 0;
				uint32_t loc = pl.root[l];
				if (w < 8)
					w = 8;
				off = l < lanes ? l * w / 8 : 64;
				cb = get_bits(o + off, w);
				cmp_lanes++;
				if (same_val(ty, COLI(&pl, loc)[p], COLR(&pl, loc)[p],
				    COLF(&pl, loc)[p], cb, I.fl[id2], &rel)) {
					if (rel > worst_rel)
						worst_rel = rel;
					continue;
				}
				bad = 1;
				if (shownbad++ < 12) {
					struct sbuf vb = {0};
					fmt_val(&vb, ty, COLI(&pl, loc)[p], COLR(&pl, loc)[p]);
					printf("  MISMATCH %s", I.name[id2]);
					if (has_imm(id2))
						printf(" imm %d", ct[t].imm);
					printf(" lane %d: ucas %s%s, cc 0x%llx\n", l,
					    sb_str(&vb), COLF(&pl, loc)[p] & F_UNDEF ?
					    " (undefined)" : "", (unsigned long long)cb);
					printf("    lane semantics: %s\n",
					    shared_str(roots[l], 300));
					ck_show_inputs(id2, in + ((size_t)t * npts + p) *
					    CK_PT, l);
					free(vb.p);
				}
			}
		}
		plan_free(&pl);
		idst[id2] |= bad ? 2 : 1;
		if (bad)
			nbad++;
		else
			nok++;
	}
	free(in_i);
	free(in_r);
	{
		int nint = 0;
		uint32_t q;
		/* distinct intrinsics checked */
		uint8_t *seen = xcalloc(I.n, 1);
		for (t = 0; t < nt; t++)
			if (!ct[t].dropped && !seen[ct[t].id]) {
				seen[ct[t].id] = 1;
				nint++;
			}
		(void)q;
		free(seen);
		printf("x86check%s%s: %d intrinsics, %d tests (with immediates) x %d "
		    "points, %ld lanes compared: %d agree, %d differ\n",
		    filter ? " " : "", filter ? filter : "", nint, nok + nbad, npts,
		    cmp_lanes, nok, nbad);
		if (avx512)
			printf("  AVX-512: compiled with every feature, the IR's "
			    "target features replaced by the host's (AVX2), run here; "
			    "%d tests call an x86 intrinsic of the IR and cannot run on "
			    "this host\n", nnorun);
		if (ck_semerr)
			printf("  %d tests: the semantics did not build (above)\n",
			    ck_semerr);
		if (nrange)
			printf("  %d immediates outside an intrinsic's range (cc rejects "
			    "them; not tests)\n", nrange);
		if (nrej) {
			int shown = 0, last = -1;
			printf("  %d tests rejected by cc (a name or signature cc does "
			    "not know: a catalogue error):", nrej);
			for (t = 0; t < nt; t++)
				if (ct[t].dropped == 3 && ct[t].id != last) {
					last = ct[t].id;
					if (shown++ < 24)
						printf(" %s", I.name[last]);
				}
			printf("%s\n", shown > 24 ? " ..." : "");
		}
		if (nnosem)
			printf("  %d memory intrinsics have no lane semantics (not "
			    "run)\n", nnosem);
		if (worst_rel > 0)
			printf("  approximations: worst relative error %.3g\n",
			    worst_rel);
		rep_begin("x86check");
		rep_int("intrinsics", nint);
		rep_int("tests", nok + nbad);
		rep_int("points", npts);
		rep_int("lanes", cmp_lanes);
		rep_int("agree", nok);
		rep_int("differ", nbad);
		rep_int("rejected_by_cc", nrej);
		rep_int("immediates_out_of_range", nrange);
		rep_int("not_runnable", nnorun);
		if (avx512 && nnorun)
			ck_cross(idst, filter, fk);
		rep_check("cc", cmp_lanes, 0, nbad == 0 && !nrej, "lane semantics against "
		    "the C compiler's intrinsics on this host");
	}
done:
	snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
	run_sh(cmd, NULL);
	free(src.p);
	free(out.p);
	free(in);
	free(res);
	free(ct);
	free(idst);
}

/* names cc rejects outright (for the catalogue's own consistency) */
