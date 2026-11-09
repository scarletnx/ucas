/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * x86bench: latency and reciprocal throughput of a few instruction
 * classes on this host, in core cycles, to check the cost tables
 * (src/x86data.c, rows marked M).  rdtsc counts reference cycles, so each
 * figure is scaled by a 1-cycle add chain measured the same way.
 *
 *   cc -O2 -march=native -o x86bench tools/x86bench.c && ./x86bench
 *
 * Each line: class latency rthroughput.  Run by `x86 bench` (x86.c),
 * which compares with the table of the host's µarch.
 */
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

#define N 4000
#define R16(x) x x x x x x x x x x x x x x x x

static double best(double a, double b) { return a < b ? a : b; }

#define LAT(name, init, body, vars) \
static double lat_##name(void) { \
	double m = 1e30; int k; \
	for (k = 0; k < 20; k++) { \
		init; uint64_t t0 = __rdtsc(); int i; \
		for (i = 0; i < N; i++) __asm__ volatile(R16(body) : vars); \
		m = best(m, (double)(__rdtsc() - t0) / (N * 16.0)); } \
	return m; }

#define TP(name, init, body, vars) \
static double tp_##name(void) { \
	double m = 1e30; int k; \
	for (k = 0; k < 20; k++) { \
		init; uint64_t t0 = __rdtsc(); int i; \
		for (i = 0; i < N; i++) __asm__ volatile(R16(body) : vars); \
		m = best(m, (double)(__rdtsc() - t0) / (N * 16.0 * 8)); } \
	return m; }

#define I8 "+r"(a), "+r"(b), "+r"(c), "+r"(d), "+r"(e), "+r"(f), "+r"(g), "+r"(h)
#define X8 "+x"(xa), "+x"(xb), "+x"(xc), "+x"(xd), "+x"(xe), "+x"(xf), "+x"(xg), "+x"(xh)
#define IINIT uint64_t a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8
#define XINIT __m256i xa = _mm256_set1_epi32(0x3f800000), xb = xa, xc = xa, xd = xa, \
	xe = xa, xf = xa, xg = xa, xh = xa
#define T8(op) op " %0, %0\n" op " %1, %1\n" op " %2, %2\n" op " %3, %3\n" \
	op " %4, %4\n" op " %5, %5\n" op " %6, %6\n" op " %7, %7\n"
#define V8(op) op " %0, %0, %0\n" op " %1, %1, %1\n" op " %2, %2, %2\n" \
	op " %3, %3, %3\n" op " %4, %4, %4\n" op " %5, %5, %5\n" \
	op " %6, %6, %6\n" op " %7, %7, %7\n"

LAT(add, uint64_t a = 1, "add %0, %0\n", "+r"(a))
TP(add, IINIT, T8("add"), I8)
LAT(imul, uint64_t a = 1, "imul %0, %0\n", "+r"(a))
TP(imul, IINIT, T8("imul"), I8)
LAT(popcnt, uint64_t a = 1, "popcnt %0, %0\n", "+r"(a))
TP(popcnt, IINIT, T8("popcnt"), I8)
LAT(pdep, uint64_t a = 1, "pdep %0, %0, %0\n", "+r"(a))
TP(pdep, IINIT, "pdep %0, %0, %0\npdep %1, %1, %1\npdep %2, %2, %2\n"
    "pdep %3, %3, %3\npdep %4, %4, %4\npdep %5, %5, %5\npdep %6, %6, %6\n"
    "pdep %7, %7, %7\n", I8)
LAT(vpaddd, __m256i xa = _mm256_set1_epi32(1), "vpaddd %0, %0, %0\n", "+x"(xa))
TP(vpaddd, XINIT, V8("vpaddd"), X8)
LAT(vpshufb, __m256i xa = _mm256_set1_epi32(1), "vpshufb %0, %0, %0\n",
    "+x"(xa))
TP(vpshufb, XINIT, V8("vpshufb"), X8)
LAT(vpmulld, __m256i xa = _mm256_set1_epi32(1), "vpmulld %0, %0, %0\n",
    "+x"(xa))
TP(vpmulld, XINIT, V8("vpmulld"), X8)
LAT(vaddps, __m256i xa = _mm256_set1_epi32(0x3f800000), "vaddps %0, %0, %0\n",
    "+x"(xa))
TP(vaddps, XINIT, V8("vaddps"), X8)
LAT(vfmadd, __m256i xa = _mm256_set1_epi32(0x3f800000), "vfmadd231ps %0, %0, %0\n",
    "+x"(xa))
TP(vfmadd, XINIT, V8("vfmadd231ps"), X8)

static double
lat_div(void)
{
	double m = 1e30;
	int k;

	for (k = 0; k < 10; k++) {
		uint64_t a = 1000000007, t0 = __rdtsc();
		int i;
		for (i = 0; i < N; i++)
			__asm__ volatile("xor %%edx, %%edx\n\tmov %0, %%rax\n\t"
			    "div %0\n\tmov %%rax, %0\n\tor $123456789, %0\n"
			    : "+r"(a) : : "rax", "rdx");
		m = best(m, (double)(__rdtsc() - t0) / N);
	}
	return m;
}

int
main(void)
{
	double s = lat_add();	/* reference cycles per core cycle */

	printf("scale %.4f\n", s);
	printf("add %.2f %.2f\n", lat_add() / s, tp_add() / s);
	printf("imul %.2f %.2f\n", lat_imul() / s, tp_imul() / s);
	printf("popcnt %.2f %.2f\n", lat_popcnt() / s, tp_popcnt() / s);
	printf("pext %.2f %.2f\n", lat_pdep() / s, tp_pdep() / s);
	printf("vpaddd %.2f %.2f\n", lat_vpaddd() / s, tp_vpaddd() / s);
	printf("vpshufb %.2f %.2f\n", lat_vpshufb() / s, tp_vpshufb() / s);
	/* 8 chains of a 10-cycle latency: the throughput reads 10/8 at best */
	printf("vpmulld %.2f %.2f\n", lat_vpmulld() / s, tp_vpmulld() / s);
	printf("vaddps %.2f %.2f\n", lat_vaddps() / s, tp_vaddps() / s);
	printf("vfmadd %.2f %.2f\n", lat_vfmadd() / s, tp_vfmadd() / s);
	/* div chain: div + 3 single-cycle ops (mov, mov, or): subtract them */
	printf("div64 %.2f -\n", lat_div() / s - 3);
	return 0;
}
