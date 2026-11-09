/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: end-to-end checks with the host C compiler (optional).
 *
 * The original and the optimised text are each compiled as their own
 * translation unit (every function renamed with a macro, so static
 * functions are reachable through a wrapper), and a generated harness
 * calls both on edge and random inputs and compares the results and
 * every buffer passed by pointer.  The original is compiled with UBSan's
 * minimal runtime for signed overflow, shifts and division, whose
 * handlers the harness defines: an input on which the original has UB
 * is skipped (the rewrites are only claimed where C defines a value).
 *
 * Pointer parameters get 64-element buffers of random values (a `char *`
 * a NUL-terminated string); an integer parameter named n, len, count,
 * size or num gets a length 0..64.  Functions with float, struct or
 * other parameters are not tested (and say so).
 *
 * bench: rdtsc around a loop of calls through the non-inlined wrappers,
 * the median of 15 runs each, original and optimised interleaved.
 *
 * same code: both units are also compiled to assembly; a function whose
 * instructions are identical (labels and comments aside) was rewritten
 * into what clang -O2 produces anyway, and the report says so.
 */
#include <sys/types.h>
#include <sys/stat.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cfront.h"

static int have_cc = -1;

int
cf_have_cc(void)
{
	if (have_cc < 0) {
		char *argv[] = {"cc", "--version", NULL};
		struct sbuf o = {0};
		have_cc = tool_run(argv, &o) == 0;
		free(o.p);
	}
	return have_cc;
}

/* a function the harness can call: 0 none, else its parameter kinds */
enum { PK_INT, PK_PTR, PK_STR, PK_LEN };

struct tfn {
	uint32_t fn;
	const char *name;
	int np, kind[16];
	uint32_t pty[16];
	uint32_t rty;
	int bufs;
	uint32_t ft;
};

static int
is_len_name(const char *s)
{
	return !strcmp(s, "n") || !strcmp(s, "len") || !strcmp(s, "count") ||
	    !strcmp(s, "size") || !strcmp(s, "num") || !strcmp(s, "length") ||
	    !strcmp(s, "w") || !strcmp(s, "width");
}

static int
testable(struct cf_unit *u, uint32_t fn, struct tfn *t)
{
	uint32_t s = (uint32_t)u->A.iv[fn], ft = u->S.ty[s], k;
	uint32_t rt = CT.base[ft];

	memset(t, 0, sizeof *t);
	t->fn = fn;
	t->name = sym_str(u->S.name[s]);
	t->rty = rt;
	if (CT.kind[rt] != CT_VOID && !ct_is_int(rt))
		return 0;
	if (CT.nf[ft] > 16 || CT.variadic[ft])
		return 0;
	t->np = (int)CT.nf[ft];
	t->ft = ft;
	for (k = 0; k < CT.nf[ft]; k++) {
		uint32_t pt = CT.fty[CT.fst[ft] + k], b;
		const char *nm = sym_str(CT.fname[CT.fst[ft] + k]);
		t->pty[k] = pt;
		if (ct_is_int(pt)) {
			t->kind[k] = is_len_name(nm) ? PK_LEN : PK_INT;
			continue;
		}
		if (CT.kind[ct_unqual(pt)] != CT_PTR)
			return 0;
		b = CT.base[ct_unqual(pt)];
		if (!ct_is_int(b) && CT.kind[ct_unqual(b)] != CT_VOID)
			return 0;
		t->kind[k] = (CT.kind[ct_unqual(b)] == CT_CHAR) ? PK_STR : PK_PTR;
		t->bufs++;
	}
	return 1;
}

static void
tydecl(struct sbuf *b, uint32_t t, const char *name)
{
	ct_str(b, t, name);
}

/* the wrappers: one unit per version */
static void
unit_text(struct sbuf *b, struct cf_unit *u, const char *path,
    const char *pfx, const struct tfn *tf, int ntf)
{
	int k, j;

	for (k = 0; k < u->nfn; k++) {
		const char *nm = sym_str(u->S.name[u->A.iv[u->fns[k]]]);
		sb_printf(b, "#define %s %s_%s\n", nm, pfx, nm);
	}
	sb_printf(b, "#include \"%s\"\n", path);
	for (k = 0; k < u->nfn; k++) {
		const char *nm = sym_str(u->S.name[u->A.iv[u->fns[k]]]);
		sb_printf(b, "#undef %s\n", nm);
	}
	for (k = 0; k < ntf; k++) {
		struct sbuf proto = {0}, nm = {0};
		sb_printf(&nm, "cfw_%s_%s(", pfx, tf[k].name);
		for (j = 0; j < tf[k].np; j++) {
			char pn[16];
			snprintf(pn, sizeof pn, "a%d", j);
			if (j)
				sb_puts(&nm, ", ");
			tydecl(&nm, tf[k].pty[j], pn);
		}
		if (tf[k].np == 0)
			sb_puts(&nm, "void");
		sb_puts(&nm, ")");
		tydecl(&proto, tf[k].rty, nm.p);
		sb_printf(b, "__attribute__((noinline)) %s\n{\n\t%s%s_%s(",
		    proto.p, CT.kind[tf[k].rty] == CT_VOID ? "" : "return ", pfx,
		    tf[k].name);
		for (j = 0; j < tf[k].np; j++)
			sb_printf(b, "%sa%d", j ? ", " : "", j);
		sb_puts(b, ");\n}\n");
		free(proto.p);
		free(nm.p);
	}
}

static const char *ub_handlers =
    "volatile int cf_ub;\n"
#define H(x) "void __ubsan_handle_" x "_minimal(void) { cf_ub = 1; }\n" \
    "void __ubsan_handle_" x "_minimal_abort(void) { cf_ub = 1; }\n"
    H("add_overflow") H("sub_overflow") H("mul_overflow")
    H("negate_overflow") H("divrem_overflow") H("shift_out_of_bounds")
#undef H
    ;

static const struct cf_opts *H_o;

/* an assumed range for parameter j of t: the harness stays inside it */
static int
assumed(const struct tfn *t, int j, int64_t *lo, int64_t *hi)
{
	int k;
	const char *nm = sym_str(CT.fname[CT.fst[t->ft] + j]);

	for (k = 0; H_o && k < H_o->nassume; k++)
		if (!strcmp(H_o->assume[k].name, nm)) {
			*lo = H_o->assume[k].lo;
			*hi = H_o->assume[k].hi;
			return 1;
		}
	return 0;
}

/*
 * bench for a function with buffers (added with the vectoriser): every
 * pointer gets 4096 (+ 64) elements, every length 4096, other integers
 * 3; the median of 15 runs of 8 calls each, original and optimised
 * interleaved.  "BENCHN name c0 c1 4096": cycles per call over 4096
 * elements.  Not for strings (they need their terminator).
 */
#define BENCH_N 4096
static void
bench_bufs(struct sbuf *b, const struct tfn *t)
{
	int j, v;

	for (j = 0; j < t->np; j++)
		if (t->kind[j] == PK_STR)
			return;
	sb_puts(b, "\t\t{\n\t\t\tdouble c[2][15];\n");
	for (j = 0; j < t->np; j++)
		if (t->kind[j] == PK_PTR) {
			uint32_t et = CT.base[ct_unqual(t->pty[j])];
			struct sbuf d = {0};
			if (CT.kind[ct_unqual(et)] == CT_VOID)
				sb_puts(&d, "unsigned char");
			else
				ct_str(&d, ct_unqual(et), "");
			sb_printf(b, "\t\t\tstatic %s g%d[2][%d];\n\t\t\tfor (int q "
			    "= 0; q < %d; q++) g%d[0][q] = g%d[1][q] = (%s)(rnd() >> "
			    "(rnd() & 63));\n", d.p, j, BENCH_N + 64, BENCH_N + 64, j, j,
			    d.p);
			free(d.p);
		}
	sb_puts(b, "\t\t\tfor (int run = 0; run < 15; run++) for (int v = 0; "
	    "v < 2; v++) {\n\t\t\t\tuint64_t t0 = __rdtsc();\n\t\t\t\t"
	    "for (int rep = 0; rep < 8; rep++) {\n");
	for (v = 0; v < 2; v++) {
		sb_printf(b, "\t\t\t\t\t%scfw_%s_%s(", v ? "else " : "if (v == 0) ",
		    v ? "opt" : "orig", t->name);
		for (j = 0; j < t->np; j++) {
			struct sbuf d = {0};
			ct_str(&d, ct_unqual(t->pty[j]), "");
			if (t->kind[j] == PK_PTR)
				sb_printf(b, "%sg%d[%d]", j ? ", " : "", j, v);
			else
				sb_printf(b, "%s(%s)%d", j ? ", " : "", d.p,
				    t->kind[j] == PK_LEN ? BENCH_N : 3);
			free(d.p);
		}
		sb_puts(b, ");\n");
	}
	sb_printf(b, "\t\t\t\t}\n\t\t\t\tc[v][run] = (double)(__rdtsc() - "
	    "t0) / 8.0;\n\t\t\t}\n\t\t\tqsort(c[0], 15, sizeof c[0][0], "
	    "cmpd);\n\t\t\tqsort(c[1], 15, sizeof c[1][0], cmpd);\n\t\t\t"
	    "printf(\"BENCHN %s %%.0f %%.0f %d\\n\", c[0][7], c[1][7]);\n"
	    "\t\t}\n", t->name, BENCH_N);
}

static void
harness(struct sbuf *b, const struct tfn *tf, int ntf, int bench,
    long iters)
{
	int k, j;

	sb_puts(b, "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
	    "#include <stdlib.h>\n#include <setjmp.h>\n#include <signal.h>\n"
	    "#include <x86intrin.h>\n");
	sb_puts(b, "static sigjmp_buf cf_jb;\nstatic void cf_sig(int s) { (void)s; "
	    "siglongjmp(cf_jb, 1); }\n");
	sb_puts(b, ub_handlers);
	sb_puts(b, "static uint64_t rs = 0x9e3779b97f4a7c15ull;\n"
	    "static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; "
	    "rs ^= rs << 17; return rs; }\n"
	    "static const int64_t edge[] = {0, 1, -1, 2, 3, 7, 8, 15, 16, 31, 32, "
	    "33, 63, 64, 100, 127, 128, 255, 256, 1000, 32767, 32768, 65535, "
	    "65536, 2147483647, -2147483647-1, 4294967295ll, "
	    "9223372036854775807ll, -9223372036854775807ll-1, -2, -128, -256};\n"
	    "#define NEDGE (int)(sizeof edge / sizeof edge[0])\n"
	    "static int64_t pick(int i) { return i < NEDGE ? edge[i] : (i & 1) ? "
	    "(int64_t)(rnd() & ((1ull << (rnd() % 64)) - 1)) : (int64_t)rnd(); }\n"
	    "static int cmpd(const void *a, const void *b) { double x = "
	    "*(const double *)a, y = *(const double *)b; return (x > y) - "
	    "(x < y); }\n");
	for (k = 0; k < ntf; k++) {
		const struct tfn *t = &tf[k];
		struct sbuf proto = {0};
		for (j = 0; j < 2; j++) {
			int q;
			struct sbuf nm = {0};
			sb_printf(&nm, "cfw_%s_%s(", j ? "opt" : "orig", t->name);
			for (q = 0; q < t->np; q++) {
				if (q)
					sb_puts(&nm, ", ");
				tydecl(&nm, t->pty[q], "");
			}
			if (t->np == 0)
				sb_puts(&nm, "void");
			sb_puts(&nm, ")");
			sb_clear(&proto);
			tydecl(&proto, t->rty, nm.p);
			sb_printf(b, "%s;\n", proto.p);
			free(nm.p);
		}
		free(proto.p);
	}
	sb_puts(b, "int main(void)\n{\n\tlong bad = 0, skipped = 0, n = 0;\n"
	    "\tvolatile long i;\n\tsignal(SIGFPE, cf_sig);\n"
	    "\tsignal(SIGSEGV, cf_sig);\n\tsignal(SIGBUS, cf_sig);\n");
	for (k = 0; k < ntf; k++) {
		const struct tfn *t = &tf[k];
		int nonvoid = CT.kind[t->rty] != CT_VOID;
		sb_printf(b, "\t{ /* %s */\n", t->name);
		for (j = 0; j < t->np; j++) {
			if (t->kind[j] == PK_PTR || t->kind[j] == PK_STR) {
				uint32_t et = CT.base[ct_unqual(t->pty[j])];
				const char *e = CT.kind[ct_unqual(et)] == CT_VOID ?
				    "unsigned char" : NULL;
				struct sbuf d = {0};
				if (e == NULL)
					ct_str(&d, ct_unqual(et), "");
				sb_printf(b, "\t\t%s b%d[2][80];\n", e ? e : d.p, j);
				free(d.p);
			}
		}
		sb_printf(b, "\t\tlong tb = 0, ts = 0;\n\t\tfor (i = 0; i < %ld; "
		    "i++) {\n", iters);
		for (j = 0; j < t->np; j++) {
			struct sbuf d = {0};
			ct_str(&d, ct_unqual(t->pty[j]), "");
			int64_t alo, ahi;
			if (t->kind[j] == PK_INT && assumed(t, j, &alo, &ahi)) {
				sb_printf(b, "\t\t\t%s a%d = (%s)(%lldll + (long long)(rnd() "
				    "%% %lluull));\n", d.p, j, d.p, (long long)alo,
				    (unsigned long long)(ahi - alo + 1));
				free(d.p);
				continue;
			}
			switch (t->kind[j]) {
			case PK_INT:
				sb_printf(b, "\t\t\t%s a%d = (%s)pick((int)((i * %d + "
				    "%d) %% %d < NEDGE ? (i * %d + %d) %% %d : NEDGE + "
				    "i));\n", d.p, j, d.p, j * 7 + 1, j * 3, 64, j * 7 + 1,
				    j * 3, 64);
				break;
			case PK_LEN:
				sb_printf(b, "\t\t\t%s a%d = (%s)(i < 65 ? i : rnd() %% "
				    "65);\n", d.p, j, d.p);
				break;
			case PK_PTR:
			case PK_STR: {
				sb_printf(b, "\t\t\tfor (int q = 0; q < 80; q++) b%d[0][q] = "
				    "b%d[1][q] = (%s)(rnd() >> (rnd() & 63));\n", j, j,
				    t->kind[j] == PK_STR ? "char" : "unsigned char");
				if (t->kind[j] == PK_STR)
					sb_printf(b, "\t\t\t{ int z = (int)(rnd() %% 64); "
					    "b%d[0][z] = b%d[1][z] = 0; b%d[0][79] = b%d[1][79] "
					    "= 0; }\n", j, j, j, j);
				break;
			}
			}
			free(d.p);
		}
		/* call the original first, under the UB handlers */
		sb_puts(b, "\t\t\tcf_ub = 0;\n\t\t\tvolatile long long r0 = 0;\n"
		    "\t\t\tif (sigsetjmp(cf_jb, 1)) { ts++; continue; }\n\t\t\t");
		if (nonvoid)
			sb_puts(b, "r0 = (long long)");
		sb_printf(b, "cfw_orig_%s(", t->name);
		for (j = 0; j < t->np; j++)
			sb_printf(b, "%s%s%d%s", j ? ", " : "",
			    t->kind[j] >= PK_PTR && t->kind[j] != PK_LEN ? "b" : "a",
			    j, t->kind[j] == PK_PTR || t->kind[j] == PK_STR ? "[0]" :
			    "");
		sb_puts(b, ");\n\t\t\tif (cf_ub) { ts++; continue; }\n"
		    "\t\t\tif (sigsetjmp(cf_jb, 1)) { if (tb++ < 3) printf(\"CRASH %s\\n\", "
		    "__func__); continue; }\n\t\t\t");
		if (nonvoid)
			sb_puts(b, "long long r1 = (long long)");
		sb_printf(b, "cfw_opt_%s(", t->name);
		for (j = 0; j < t->np; j++)
			sb_printf(b, "%s%s%d%s", j ? ", " : "",
			    t->kind[j] == PK_PTR || t->kind[j] == PK_STR ? "b" : "a",
			    j, t->kind[j] == PK_PTR || t->kind[j] == PK_STR ? "[1]" :
			    "");
		sb_puts(b, ");\n\t\t\tn++;\n\t\t\tif (0");
		if (nonvoid)
			sb_puts(b, " || r0 != r1");
		for (j = 0; j < t->np; j++)
			if (t->kind[j] == PK_PTR || t->kind[j] == PK_STR)
				sb_printf(b, " || memcmp(b%d[0], b%d[1], sizeof b%d[0])",
				    j, j, j);
		sb_printf(b, ") {\n\t\t\t\tif (tb++ < 3) printf(\"MISMATCH %s:",
		    t->name);
		for (j = 0; j < t->np; j++)
			if (t->kind[j] == PK_INT || t->kind[j] == PK_LEN)
				sb_printf(b, " a%d=%%lld", j);
		if (nonvoid)
			sb_puts(b, " orig=%lld opt=%lld");
		sb_puts(b, "\\n\"");
		for (j = 0; j < t->np; j++)
			if (t->kind[j] == PK_INT || t->kind[j] == PK_LEN)
				sb_printf(b, ", (long long)a%d", j);
		if (nonvoid)
			sb_puts(b, ", r0, r1");
		sb_puts(b, ");\n\t\t\t}\n\t\t}\n");
		sb_printf(b, "\t\tprintf(\"TEST %s %%ld %%ld\\n\", tb, ts);\n"
		    "\t\tbad += tb; skipped += ts;\n", t->name);
		if (bench && t->bufs == 0) {
			/* rdtsc: the median of 15 runs of 4096 calls */
			sb_puts(b, "\t\t{\n\t\t\tdouble c[2][15];\n\t\t\tstatic "
			    "long long in[4096][8];\n\t\t\tvolatile long long sink "
			    "= 0;\n\t\t\tfor (i = 0; i < 4096; i++) for (int q = 0; q "
			    "< 8; q++) in[i][q] = (long long)rnd();\n");
			sb_puts(b, "\t\t\tfor (int run = 0; run < 15; run++) for "
			    "(int v = 0; v < 2; v++) {\n\t\t\t\tlong long acc = 0;\n"
			    "\t\t\t\tuint64_t t0 = __rdtsc();\n\t\t\t\tfor (int rep = 0; "
			    "rep < 4; rep++) for (i = 0; i < 4096; i++) {\n");
			for (j = 0; j < 2; j++) {
				int q;
				sb_printf(b, "\t\t\t\t\t%sacc += (long long)cfw_%s_%s(",
				    j ? "else " : "if (v == 0) ", j ? "opt" : "orig",
				    t->name);
				for (q = 0; q < t->np; q++) {
					struct sbuf d = {0};
					ct_str(&d, ct_unqual(t->pty[q]), "");
					sb_printf(b, "%s(%s)(in[i][%d]%s)", q ? ", " : "", d.p,
					    q, t->kind[q] == PK_LEN ? " & 63" : "");
					free(d.p);
				}
				sb_puts(b, ");\n");
			}
			sb_puts(b, "\t\t\t\t}\n\t\t\t\tc[v][run] = (double)(__rdtsc() "
			    "- t0) / (4 * 4096.0);\n\t\t\t\tsink += acc;\n\t\t\t}\n"
			    "\t\t\tqsort(c[0], 15, sizeof c[0][0], cmpd);\n"
			    "\t\t\tqsort(c[1], 15, sizeof c[1][0], cmpd);\n");
			sb_printf(b, "\t\t\tprintf(\"BENCH %s %%.2f %%.2f\\n\", "
			    "c[0][7], c[1][7]);\n\t\t}\n", t->name);
		}
		if (bench && t->bufs > 0)
			bench_bufs(b, t);
		sb_puts(b, "\t}\n");
	}
	sb_puts(b, "\tprintf(\"TOTAL %ld %ld %ld\\n\", n, bad, skipped);\n"
	    "\treturn bad != 0;\n}\n");
}

static int
write_file(const char *path, const char *s)
{
	FILE *fp = fopen(path, "w");

	if (fp == NULL)
		return 0;
	fputs(s, fp);
	fclose(fp);
	return 1;
}

static int
run(char **argv, struct sbuf *out)
{
	sb_clear(out);
	return tool_run(argv, out);
}

/* the host can run code for the target: its features are a subset */
static const char *
host_march(const struct cf_target *t, struct sbuf *note)
{
	static char buf[64];
	char *argv[] = {"cc", "-march=native", "-dM", "-E", "-x", "c",
	    "/dev/null", NULL};
	struct sbuf o = {0};
	int k;

	if (tool_run(argv, &o) != 0) {
		free(o.p);
		snprintf(buf, sizeof buf, "-march=%s", t->march);
		return buf;
	}
	for (k = 0; k < CF_NFEAT; k++)
		if ((t->feat & CF_F(k)) && !strstr(o.p, cf_feat_macro(k))) {
			sb_printf(note, "  note: this host lacks %s, so the checks "
			    "ran with -march=native (the scalar fallbacks of "
			    "%s-only paths)\n", cf_feat_name(k), t->name);
			free(o.p);
			return "-march=native";
		}
	free(o.p);
	snprintf(buf, sizeof buf, "-march=%s", t->march);
	return buf;
}

/* the instructions of one function in an assembly listing */
static void
asm_body(const char *s, const char *fn, struct sbuf *out)
{
	char lab[160];
	const char *p;
	size_t l;

	snprintf(lab, sizeof lab, "\n%s:", fn);
	p = strstr(s, lab);
	sb_clear(out);
	sb_puts(out, "");
	if (p == NULL)
		return;
	p += strlen(lab);
	while (*p) {
		const char *e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		l = (size_t)(e - p);
		while (l && isspace((unsigned char)*p))
			p++, l--;
		if (l >= 11 && !strncmp(p, ".Lfunc_end", 10))
			break;
		if (l && *p != '#' && *p != '.' && !(l && p[l - 1] == ':' &&
		    p[0] == '.')) {
			/* an instruction: drop label operands' numbers */
			size_t i;
			for (i = 0; i < l; i++) {
				if (p[i] == '#')
					break;
				if (p[i] == '.' && i + 2 < l && p[i + 1] == 'L') {
					sb_puts(out, ".L");
					while (i + 1 < l && (isalnum((unsigned char)p[i + 1]) ||
					    p[i + 1] == '_' || p[i + 1] == '.'))
						i++;
					continue;
				}
				sb_putc(out, p[i]);
			}
			sb_putc(out, '\n');
		}
		p = *e ? e + 1 : e;
	}
}

static void
strip_pfx(struct sbuf *s, const char *pfx)
{
	struct sbuf o = {0};
	const char *p = s->p ? s->p : "", *q;
	size_t l = strlen(pfx);

	sb_puts(&o, "");
	while ((q = strstr(p, pfx)) != NULL) {
		sb_printf(&o, "%.*s", (int)(q - p), p);
		p = q + l;
	}
	sb_puts(&o, p);
	free(s->p);
	*s = o;
}

static int
check(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep, int bench, struct cf_result *r)
{
	char dir[] = "/tmp/ucas-cfront-XXXXXX";
	char p_orig[256], p_opt[256], p_u0[256], p_u1[256], p_h[256], p_bin[300],
	    o0[300], o1[300], oh[300], s0[300], s1[300];
	struct tfn tf[64];
	int ntf = 0, k, ok = 0;
	struct sbuf b = {0}, out = {0}, note = {0}, bout = {0};
	const char *march;
	char mflag[80];

	if (!cf_have_cc()) {
		sb_puts(rep, "  check: no host cc\n");
		return -1;
	}
	if (mkdtemp(dir) == NULL)
		return -1;
	march = host_march(o->t, &note);
	snprintf(mflag, sizeof mflag, "%s", march);
	sb_puts(rep, note.p ? note.p : "");
	H_o = o;
	for (k = 0; k < u->nfn && ntf < 64; k++)
		if (r && r->calls_only && r->calls_only[k])
			sb_printf(rep, "  check: %s is static and only called here: "
			    "tested through its callers\n", sym_str(u->S.name[
			    u->A.iv[u->fns[k]]]));
		else if (u->tk.file[u->A.t0[u->fns[k]]] == 0 &&
		    testable(u, u->fns[k], &tf[ntf]))
			ntf++;
		else if (u->tk.file[u->A.t0[u->fns[k]]] == 0)
			sb_printf(rep, "  check: %s not tested (a parameter or result "
			    "the harness cannot make)\n", sym_str(u->S.name[
			    u->A.iv[u->fns[k]]]));
	snprintf(p_orig, sizeof p_orig, "%s/orig.c", dir);
	snprintf(p_opt, sizeof p_opt, "%s/opt.c", dir);
	snprintf(p_u0, sizeof p_u0, "%s/u0.c", dir);
	snprintf(p_u1, sizeof p_u1, "%s/u1.c", dir);
	snprintf(p_h, sizeof p_h, "%s/h.c", dir);
	snprintf(p_bin, sizeof p_bin, "%s/h", dir);
	snprintf(o0, sizeof o0, "%s/u0.o", dir);
	snprintf(o1, sizeof o1, "%s/u1.o", dir);
	snprintf(oh, sizeof oh, "%s/h.o", dir);
	snprintf(s0, sizeof s0, "%s/u0.s", dir);
	snprintf(s1, sizeof s1, "%s/u1.s", dir);
	write_file(p_orig, orig);
	write_file(p_opt, opt);
	unit_text(&b, u, p_orig, "orig", tf, ntf);
	write_file(p_u0, b.p);
	sb_clear(&b);
	unit_text(&b, u, p_opt, "opt", tf, ntf);
	write_file(p_u1, b.p);
	sb_clear(&b);
	harness(&b, tf, ntf, bench, 20000);
	write_file(p_h, b.p);
	{
		char *a0[] = {"cc", "-std=c99", "-O2", mflag, "-w",
		    "-fsanitize=signed-integer-overflow,shift,integer-divide-by-zero",
		    "-fsanitize-minimal-runtime", "-c", p_u0, "-o", o0, NULL};
		char *a1[] = {"cc", "-std=c99", "-O2", mflag, "-Wall", "-Wextra",
		    "-Wno-unused-function", "-Wno-unused-parameter", "-c", p_u1,
		    "-o", o1, NULL};
		char *ah[] = {"cc", "-std=gnu99", "-O1", mflag, "-w", "-c", p_h,
		    "-o", oh, NULL};
		char *al[] = {"cc", "-o", p_bin, oh, o0, o1, NULL};
		char *ax[] = {p_bin, NULL};
		/* bench: the original without the UBSan checks (they would
		 * slow it and keep clang from vectorising it) */
		char o0n[310], p_bin2[310];
		char *a0n[] = {"cc", "-std=c99", "-O2", mflag, "-w", "-c", p_u0,
		    "-o", o0n, NULL};
		char *al2[] = {"cc", "-o", p_bin2, oh, o0n, o1, NULL};
		char *ax2[] = {p_bin2, NULL};
		char *as0[] = {"cc", "-std=c99", "-O2", mflag, "-w", "-S", p_u0,
		    "-o", s0, NULL};
		char *as1[] = {"cc", "-std=c99", "-O2", mflag, "-w", "-S", p_u1,
		    "-o", s1, NULL};
		if (run(a0, &out) != 0) {
			sb_printf(rep, "  check: the original does not compile:\n%.600s",
			    out.p);
			goto done;
		}
		if (run(a1, &out) != 0) {
			sb_printf(rep, "  check: the OUTPUT does not compile:\n%.1200s",
			    out.p);
			ok = 0;
			goto done;
		}
		if (out.n > 0 && strstr(out.p, "warning"))
			sb_printf(rep, "  check: the output compiles with warnings:\n"
			    "%.800s", out.p);
		if (!strcmp(mflag, "-march=native")) {
			/* the target's own paths are #if'ed out here: compile
			 * them for the target at least (they cannot run) */
			char tm[80];
			char *ac[] = {"cc", "-std=c99", "-O2", tm, "-Wall", "-Wextra",
			    "-Wno-unused-function", "-Wno-unused-parameter",
			    "-fsyntax-only", p_u1, NULL};
			struct sbuf co = {0};
			snprintf(tm, sizeof tm, "-march=%s", o->t->march);
			if (run(ac, &co) != 0) {
				sb_printf(rep, "  check: the OUTPUT does not compile for %s:"
				    "\n%.1200s", tm, co.p);
				free(co.p);
				ok = 0;
				goto done;
			}
			sb_printf(rep, "  check: the output also compiles with %s%s (its "
			    "paths for that target cannot run on this host)\n", tm,
			    co.n && strstr(co.p, "warning") ? ", with warnings" : "");
			free(co.p);
		}
		if (run(ah, &out) != 0 || run(al, &out) != 0) {
			sb_printf(rep, "  check: the harness does not build:\n%.600s",
			    out.p);
			goto done;
		}
		run(ax, &out);
		snprintf(o0n, sizeof o0n, "%s/u0n.o", dir);
		snprintf(p_bin2, sizeof p_bin2, "%s/hb", dir);
		if (bench && (run(a0n, &bout) != 0 || run(al2, &bout) != 0 ||
		    run(ax2, &bout) < 0))
			sb_clear(&bout);
		{
			/* TEST name bad skipped / BENCH name c0 c1 / TOTAL n bad skip */
			const char *p = out.p ? out.p : "";
			long n = -1, bad = 0, skip = 0;
			const char *t = strstr(p, "TOTAL ");
			if (t)
				sscanf(t, "TOTAL %ld %ld %ld", &n, &bad, &skip);
			if (n < 0) {
				sb_printf(rep, "  check: the harness failed:\n%.600s", p);
				goto done;
			}
			if (bad)
				sb_printf(rep, "  check: DIFFERENCES between the original "
				    "and the output:\n%.1200s", p);
			else
				sb_printf(rep, "  check: compiled both with cc -O2 %s; %ld "
				    "calls of %d functions agree (edge and random inputs%s)\n",
				    mflag, n, ntf, skip ? "" : "");
			if (skip)
				sb_printf(rep, "  check: %ld inputs skipped: UB in the "
				    "original (UBSan)\n", skip);
			rep_obj("check");
			rep_int("calls", n);
			rep_int("functions", ntf);
			rep_int("mismatches", bad);
			rep_int("skipped_ub", skip);
			rep_end();
			if (bench) {
				const char *q = bout.p ? bout.p : "";
				p = q;
				rep_list("bench");
				while ((q = strstr(q, "BENCH ")) != NULL) {
					char nm[96];
					double c0, c1;
					if (sscanf(q, "BENCH %95s %lf %lf", nm, &c0, &c1) == 3) {
						sb_printf(rep, "  bench %s: %.2f -> %.2f cycles per "
						    "call (rdtsc, median of 15; x%.2f)\n", nm, c0, c1,
						    c1 > 0 ? c0 / c1 : 0);
						rep_obj(NULL);
						rep_str("function", nm);
						rep_num("cycles_before", c0);
						rep_num("cycles_after", c1);
						rep_end();
					}
					q += 6;
				}
				q = p;
				while ((q = strstr(q, "BENCHN ")) != NULL) {
					char nm[96];
					double c0, c1;
					int ne;
					if (sscanf(q, "BENCHN %95s %lf %lf %d", nm, &c0, &c1,
					    &ne) == 4) {
						sb_printf(rep, "  bench %s: %.0f -> %.0f cycles per "
						    "call of %d elements (%.2f -> %.2f per element; "
						    "rdtsc, median of 15; x%.2f)\n", nm, c0, c1, ne,
						    c0 / ne, c1 / ne, c1 > 0 ? c0 / c1 : 0);
						rep_obj(NULL);
						rep_str("function", nm);
						rep_num("cycles_before", c0);
						rep_num("cycles_after", c1);
						rep_int("elements", ne);
						rep_end();
					}
					q += 7;
				}
				rep_end();
			}
			ok = bad == 0;
		}
		/* the same code as clang -O2 makes of the original? */
		if (r && run(as0, &out) == 0) {
			struct sbuf a = {0}, bb = {0}, t0 = {0}, t1 = {0};
			char *argv0[] = {"cat", s0, NULL}, *argv1[] = {"cat", s1, NULL};
			run(argv0, &t0);
			if (run(as1, &out) == 0 && run(argv1, &t1) == 0) {
				int i;
				for (i = 0; i < r->nrw; i++) {
					char f0[96], f1[96];
					if (!r->rw[i].applied)
						continue;
					snprintf(f0, sizeof f0, "cfw_orig_%s", r->rw[i].func);
					snprintf(f1, sizeof f1, "cfw_opt_%s", r->rw[i].func);
					asm_body(t0.p ? t0.p : "", f0, &a);
					asm_body(t1.p ? t1.p : "", f1, &bb);
					strip_pfx(&a, "orig_");
					strip_pfx(&bb, "opt_");
					if (a.n > 0 && a.n == bb.n && !strcmp(a.p, bb.p))
						r->rw[i].trivial = 1;
				}
			}
			free(a.p);
			free(bb.p);
			free(t0.p);
			free(t1.p);
		}
	}
done:
	if (o->keep)
		sb_printf(rep, "  check: kept the build directory %s\n", dir);
	else {
		char *rm[] = {"rm", "-rf", dir, NULL};
		run(rm, &out);
	}
	free(b.p);
	free(out.p);
	free(bout.p);
	free(note.p);
	return ok;
}

int
cf_difftest(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep)
{
	return check(u, orig, opt, o, rep, 0, NULL);
}

int
cf_bench(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep)
{
	return check(u, orig, opt, o, rep, 1, NULL);
}

/* both, and the clang -O2 comparison per rewritten function */
int cf_check_all(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep, int bench,
    struct cf_result *r);

int
cf_check_all(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep, int bench, struct cf_result *r)
{
	return check(u, orig, opt, o, rep, bench, r);
}
