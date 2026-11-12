/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * GLSL to SPIR-V assembly through glslangValidator and spirv-dis, with a
 * snippet completed first: a #version line; when there is no main(), a
 * compute shader whose main() calls every function with its arguments
 * read from a buffer (so nothing is optimized away and the parameters are
 * inputs); and a declaration for each identifier the compiler says is
 * undeclared, its type guessed from use (".w" makes a 4-vector, "1u" or
 * bit operators make it unsigned).  What was assumed goes to notes.
 *
 * It runs programs, so it is for the command line and the bot's
 * supervisor; inside the bot's sandbox it fails (and says so).
 */
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/wait.h>

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ucas.h"

/*
 * A tool's output is kept whole or the run fails: a disassembly cut short
 * reads as a module whose branches lead nowhere.  A large shader's is a
 * few megabytes (BC7 with its ASTC decoder: 0.5 MiB; with the ASTC
 * encoder: 1.1 MiB), and spirv-opt -O takes 8 s and 480 MiB on that one.
 */
#define MAXTOOL (32UL << 20)

/* run argv with limits; its output (both streams) into out; -2 when the
 * output is more than MAXTOOL */
int
tool_run(char *const argv[], struct sbuf *out)
{
	int fd[2], st = 0, over = 0;
	pid_t pid;
	char tmp[4096];
	ssize_t r;

	if (pipe(fd) < 0)
		return -1;
	if ((pid = fork()) < 0) {
		close(fd[0]);
		close(fd[1]);
		return -1;
	}
	if (pid == 0) {
		struct rlimit cpu = {60, 60}, mem = {2048UL << 20, 2048UL << 20},
		    fsz = {32UL << 20, 32UL << 20};
		int nul = open("/dev/null", O_RDONLY);
		setrlimit(RLIMIT_CPU, &cpu);
		setrlimit(RLIMIT_AS, &mem);
		setrlimit(RLIMIT_FSIZE, &fsz);
		if (nul >= 0)
			dup2(nul, 0);
		dup2(fd[1], 1);
		dup2(fd[1], 2);
		close(fd[0]);
		close(fd[1]);
		execvp(argv[0], argv);
		_exit(127);
	}
	close(fd[1]);
	while ((r = read(fd[0], tmp, sizeof tmp)) > 0) {
		if (out->n + (size_t)r > MAXTOOL) {
			over = 1;
			continue;	/* drained, so the tool ends */
		}
		sb_printf(out, "%.*s", (int)r, tmp);
	}
	close(fd[0]);
	while (waitpid(pid, &st, 0) < 0)
		;
	if (over)
		return -2;
	return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

/* data to a new temporary file named like tmpl ("...XXXXXX.ext") */
int
tool_tmp(char *tmpl, const char *data, size_t n)
{
	const char *dot = strrchr(tmpl, '.');
	int fd = mkstemps(tmpl, dot ? (int)strlen(dot) : 0);

	if (fd < 0)
		return -1;
	if (write(fd, data, n) != (ssize_t)n) {
		close(fd);
		unlink(tmpl);
		return -1;
	}
	close(fd);
	return 0;
}

/* binary SPIR-V to text; 0 on failure (the reason in out) */
int
spv_disassemble(const char *data, size_t n, struct sbuf *out)
{
	char path[] = "/tmp/ucas-spv-XXXXXX.spv";
	char *argv[] = {"spirv-dis", "--no-color", path, NULL};
	int rc;

	if (tool_tmp(path, data, n) < 0)
		return 0;
	rc = tool_run(argv, out);
	unlink(path);
	return rc == 0;
}

static const char *const kw[] = {"if", "for", "while", "switch", "return",
	"else", "do", "layout", "struct", "uniform", "buffer", "in", "out",
	"inout", "const", "shared", NULL};

static int
is_kw(const char *w, size_t n)
{
	int k;

	for (k = 0; kw[k]; k++)
		if (strlen(kw[k]) == n && !strncmp(kw[k], w, n))
			return 1;
	return 0;
}

/*
 * Top-level functions "TYPE NAME(PARAMS) {": a compute main() calling
 * each, its arguments from a buffer, its result stored back
 */
static void
harness(const char *src, struct sbuf *h, struct sbuf *notes)
{
	struct sbuf blk = {0}, calls = {0};
	const char *p = src;
	int depth = 0, nf = 0;

	while (*p) {
		if (*p == '{') {
			depth++;
			p++;
			continue;
		}
		if (*p == '}') {
			depth--;
			p++;
			continue;
		}
		if (*p == '/' && p[1] == '/') {
			while (*p && *p != '\n')
				p++;
			continue;
		}
		if (depth == 0 && (isalpha((unsigned char)*p) || *p == '_') &&
		    (p == src || !(isalnum((unsigned char)p[-1]) || p[-1] == '_'))) {
			const char *t0 = p, *t1, *n0, *n1, *q;
			while (isalnum((unsigned char)*p) || *p == '_')
				p++;
			t1 = p;
			while (*p == ' ' || *p == '\t')
				p++;
			n0 = p;
			while (isalnum((unsigned char)*p) || *p == '_')
				p++;
			n1 = p;
			q = p;
			while (*q == ' ' || *q == '\t')
				q++;
			if (n1 > n0 && *q == '(' && !is_kw(t0, (size_t)(t1 - t0)) &&
			    !(n1 - n0 == 4 && !strncmp(n0, "main", 4))) {
				const char *pe = strchr(q, ')'), *b;
				const char *a;
				int np = 0;
				if (pe == NULL)
					continue;
				b = pe + 1;
				while (*b == ' ' || *b == '\t' || *b == '\n' || *b == '\r')
					b++;
				if (*b != '{')
					continue;
				/* parameters: [in|out|inout|const] TYPE NAME */
				if (t1 - t0 == 4 && !strncmp(t0, "void", 4))
					sb_printf(&calls, "\t%.*s(", (int)(n1 - n0), n0);
				else {	/* the call stores its result */
					sb_printf(&blk, "\t%.*s %.*s_ret;\n", (int)(t1 - t0), t0,
					    (int)(n1 - n0), n0);
					sb_printf(&calls, "\tucas_args.%.*s_ret = %.*s(",
					    (int)(n1 - n0), n0, (int)(n1 - n0), n0);
				}
				a = q + 1;
				while (a < pe) {
					const char *w[4], *we[4];
					int nw = 0;
					while (a < pe && nw < 4) {
						while (a < pe && !(isalnum((unsigned char)*a) ||
						    *a == '_'))
							a++;
						if (a >= pe)
							break;
						w[nw] = a;
						while (a < pe && (isalnum((unsigned char)*a) ||
						    *a == '_'))
							a++;
						we[nw++] = a;
						while (a < pe && (*a == ' ' || *a == '\t'))
							a++;
						if (a < pe && *a == ',') {
							a++;
							break;
						}
					}
					if (nw >= 2) {	/* the last two: type, name */
						sb_printf(&blk, "\t%.*s %.*s_%.*s;\n",
						    (int)(we[nw - 2] - w[nw - 2]), w[nw - 2],
						    (int)(n1 - n0), n0,
						    (int)(we[nw - 1] - w[nw - 1]), w[nw - 1]);
						sb_printf(&calls, "%sucas_args.%.*s_%.*s",
						    np ? ", " : "", (int)(n1 - n0), n0,
						    (int)(we[nw - 1] - w[nw - 1]), w[nw - 1]);
						np++;
					}
				}
				sb_puts(&calls, ");\n");
				nf++;
				sb_printf(notes, "  %.*s: called from a compute shader's "
				    "main(), its arguments read from a buffer\n",
				    (int)(n1 - n0), n0);
				p = pe + 1;
			}
			continue;
		}
		p++;
	}
	sb_puts(h, "\nlayout(local_size_x = 1) in;\n");
	if (blk.n)
		sb_printf(h, "layout(std430, binding = 7) buffer UcasArgs {\n%s}"
		    " ucas_args;\n", blk.p);
	sb_printf(h, "void main() {\n%s}\n", calls.p ? calls.p : "");
	(void)nf;
	free(blk.p);
	free(calls.p);
}

/* a declaration for an undeclared NAME, from how the snippet uses it */
static void
guess_decl(const char *src, const char *name, struct sbuf *decl,
    struct sbuf *notes)
{
	size_t nl = strlen(name);
	const char *p = src;
	int comps = 0, arr = 0, uns = 0, flt = 0, sgn = 0;
	char ty[32];

	while ((p = strstr(p, name)) != NULL) {
		const char *e = p + nl;
		if ((p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) ||
		    isalnum((unsigned char)*e) || *e == '_') {
			p = e;
			continue;
		}
		if (*e == '.') {
			const char *s = e + 1;
			while (strchr("xyzwrgbastpq", *s) && *s) {
				int c = strchr("xyzw", *s) ? (int)(strchr("xyzw", *s) -
				    "xyzw") : strchr("rgba", *s) ? (int)(strchr("rgba",
				    *s) - "rgba") : (int)(strchr("stpq", *s) - "stpq");
				if (c + 1 > comps)
					comps = c + 1;
				s++;
			}
		}
		if (*e == '[')
			arr = 1;
		p = e;
	}
	/* the snippet's flavour */
	for (p = src; *p; p++) {
		if (isdigit((unsigned char)*p) && (p[1] == 'u' || p[1] == 'U') &&
		    !isalnum((unsigned char)p[2]))
			uns = 1;
		if (isdigit((unsigned char)*p) && p[1] == '.' &&
		    isdigit((unsigned char)p[2]))
			flt = 1;
	}
	if (strstr(src, "uint") || strstr(src, "uvec") || strstr(src, ">>") ||
	    strstr(src, "<<") || strchr(src, '^') || strchr(src, '|'))
		uns = 1;
	if (strstr(src, "ivec") || strstr(src, "int "))
		sgn = 1;
	if (comps >= 2)
		snprintf(ty, sizeof ty, "%svec%d", flt && !uns ? "" : uns ? "u" :
		    sgn ? "i" : "", comps);
	else
		snprintf(ty, sizeof ty, "%s", flt && !uns ? "float" : uns ? "uint" :
		    "int");
	sb_printf(decl, "layout(std430, binding = %d) buffer UcasGlobal_%s { %s "
	    "%s%s; };\n", 6 - (int)(decl->n % 5), name, ty, name, arr ? "[64]" :
	    "");
	sb_printf(notes, "  %s is not declared: assumed `%s %s%s;` (a global in a "
	    "buffer, so the optimizer keeps what is written to it; from %s)\n",
	    name, ty, name, arr ? "[64]" : "", comps >= 2 ? "its swizzles" :
	    "how it is used");
}

/* word w in src, as a whole word */
static int
has_word(const char *src, const char *w)
{
	size_t n = strlen(w);
	const char *p = src;

	while ((p = strstr(p, w)) != NULL) {
		if ((p == src || !(isalnum((unsigned char)p[-1]) || p[-1] == '_')) &&
		    !(isalnum((unsigned char)p[n]) || p[n] == '_'))
			return 1;
		p += n;
	}
	return 0;
}

/*
 * The stage src is compiled as: what it says ("#pragma shader_stage(X)",
 * glslang's own way), else the block's tag (```vert, ```frag, ```comp: a
 * file's extension), else what only one stage has: a workgroup, a written
 * gl_Position or the vertex's index, a fragment's built-ins.
 */
static const char *
glsl_stage(const char *src, int has_main)
{
	static const char *const st[][2] = {{"vertex", "vert"}, {"fragment",
	    "frag"}, {"compute", "comp"}, {"geometry", "geom"}, {"tesscontrol",
	    "tesc"}, {"tesseval", "tese"}, {"vert", "vert"}, {"frag", "frag"},
	    {"comp", "comp"}, {"geom", "geom"}, {"tesc", "tesc"}, {"tese",
	    "tese"}};
	const char *p = strstr(src, "#pragma shader_stage");
	size_t k;

	if (p != NULL && (p = strchr(p, '(')) != NULL) {
		p++;
		while (*p == ' ')
			p++;
		for (k = 0; k < 6; k++)
			if (!strncmp(p, st[k][0], strlen(st[k][0])))
				return st[k][1];
	}
	if (!has_main)
		return "comp";
	for (k = 6; k < 12; k++)
		if (!strcmp(code_block_lang, st[k][0]))
			return st[k][1];
	if (strstr(src, "local_size") || has_word(src, "gl_GlobalInvocationID") ||
	    has_word(src, "gl_LocalInvocationID") ||
	    has_word(src, "gl_LocalInvocationIndex") ||
	    has_word(src, "gl_WorkGroupID"))
		return "comp";
	if (has_word(src, "gl_Position") || has_word(src, "gl_VertexIndex") ||
	    has_word(src, "gl_InstanceIndex") || has_word(src, "gl_VertexID"))
		return "vert";
	return "frag";
}

/*
 * src (GLSL) to SPIR-V assembly in out; 1 on success, else 0 with the
 * compiler's errors in notes
 */
int
glsl_to_spvasm(const char *src, struct sbuf *out, struct sbuf *notes)
{
	struct sbuf body = {0}, decl = {0}, full = {0}, log = {0};
	const char *stage, *env = "vulkan1.1";
	int tries, ok = 0, has_main = strstr(src, "main") != NULL &&
	    strstr(src, "void main") != NULL;

	stage = glsl_stage(src, has_main);
	sb_puts(&body, src);
	{
		/* #line directives blanked: the lines reported are the text's own
		 * (a preprocessed shader names its included files there) */
		char *l = body.p;
		while (l != NULL && *l) {
			char *e = strchr(l, '\n'), *c = l;
			while (*c == ' ' || *c == '\t')
				c++;
			if (*c == '#') {
				c++;
				while (*c == ' ' || *c == '\t')
					c++;
				if (!strncmp(c, "line", 4) && !isalnum((unsigned char)c[4]) &&
				    c[4] != '_')
					for (c = l; *c && *c != '\n'; c++)
						*c = ' ';
			}
			l = e ? e + 1 : NULL;
		}
	}
	for (tries = 0; tries < 8; tries++) {
		char path[] = "/tmp/ucas-glsl-XXXXXX.glsl";
		char spv[] = "/tmp/ucas-glsl-XXXXXX.spv";
		char st[8];
		char tenv[16];
		char *argv[] = {"glslangValidator", "-g", "-V", "--target-env",
		    tenv, "-S", st, "-o", spv, path, NULL};
		const char *e, *l;
		int changed = 0;

		snprintf(st, sizeof st, "%s", stage);
		snprintf(tenv, sizeof tenv, "%s", env);
		sb_clear(&full);
		sb_puts(&full, "");
		{
			/* #version first (the source's own, or 450), declarations,
			 * then the source with its own line numbers in OpLine */
			const char *v = strstr(body.p, "#version"), *rest = body.p;
			int line = 1;
			if (v != NULL) {
				const char *ve = strchr(v, '\n'), *c;
				for (c = body.p; c < v; c++)
					if (*c == '\n')
						line++;
				/* what came before #version is comments or blank */
				sb_printf(&full, "%.*s\n", (int)(ve ? ve - v :
				    (long)strlen(v)), v);
				rest = ve ? ve + 1 : v + strlen(v);
				line++;
			} else
				sb_puts(&full, "#version 450\n");
			sb_puts(&full, decl.p ? decl.p : "");
			sb_printf(&full, "#line %d\n", line);
			sb_puts(&full, rest);
		}
		if (!has_main) {
			sb_puts(&full, "\n#line 10000\n");
			harness(src, &full, tries == 0 ? notes : &log);
		}
		sb_clear(&log);
		if (tool_tmp(path, full.p, full.n) < 0 || tool_tmp(spv, "", 0) < 0) {
			sb_puts(notes, "  cannot write a temporary file here\n");
			break;
		}
		{
			int rc = tool_run(argv, &log);
			if (rc == 0) {
				/* the binary, disassembled */
				FILE *fp = fopen(spv, "rb");
				char *bin = NULL;
				size_t nb = 0, cap = 0, r;
				/* raw bytes: the binary is full of zeros */
				while (fp) {
					if (nb == cap) {
						cap = cap ? cap * 2 : 65536;
						bin = xrealloc(bin, cap);
					}
					if ((r = fread(bin + nb, 1, cap - nb, fp)) == 0)
						break;
					nb += r;
				}
				if (fp)
					fclose(fp);
				/* optimized as a real pipeline would: spirv-opt -O */
				if (nb > 0) {
					char opath[] = "/tmp/ucas-opt-XXXXXX.spv";
					char oenv[32];
					char *oargv[] = {"spirv-opt", "-O", oenv, spv, "-o",
					    opath, NULL};
					struct sbuf olog = {0};
					snprintf(oenv, sizeof oenv, "--target-env=%s", env);
					if (tool_tmp(opath, "", 0) == 0) {
						if (tool_run(oargv, &olog) == 0) {
							FILE *of = fopen(opath, "rb");
							size_t r2;
							nb = 0;
							while (of) {
								if (nb == cap) {
									cap = cap ? cap * 2 : 65536;
									bin = xrealloc(bin, cap);
								}
								if ((r2 = fread(bin + nb, 1, cap - nb, of)) == 0)
									break;
								nb += r2;
							}
							if (of)
								fclose(of);
							sb_puts(notes, "  optimized with spirv-opt -O (what "
							    "follows is beyond that)\n");
						} else
							sb_printf(notes, "  spirv-opt -O failed: the "
							    "unoptimized module is used\n");
						unlink(opath);
					}
					free(olog.p);
				}
				ok = nb > 0 && spv_disassemble(bin, nb, out);
				if (nb > 0 && !ok) {
					/* not the compiler's error: the reason is said */
					sb_printf(notes, "  the module could not be disassembled "
					    "(spirv-dis missing, or its text is more than %lu "
					    "MiB)\n", MAXTOOL >> 20);
					sb_clear(out);
				}
				free(bin);
				/* the temporary file's name (in OpString) as a fixed one */
				if (ok && out->p) {
					char *at;
					size_t pl = strlen(path);
					while ((at = strstr(out->p, path)) != NULL) {
						memmove(at + 12, at + pl, strlen(at + pl) + 1);
						memcpy(at, "snippet.glsl", 12);
						out->n = strlen(out->p);
					}
				}
			} else if (rc == 127 || rc < 0)
				sb_puts(notes, "  glslangValidator cannot run here\n");
		}
		unlink(path);
		unlink(spv);
		if (ok)
			break;
		/* "ERROR: 0:3: 's' : undeclared identifier": declare it */
		for (l = log.p ? log.p : ""; (e = strstr(l, "' : undeclared "
		    "identifier")) != NULL; l = e + 1) {
			const char *b = e;
			char name[64];
			while (b > l && b[-1] != '\'')
				b--;
			if ((size_t)(e - b) >= sizeof name || e == b)
				continue;
			memcpy(name, b, (size_t)(e - b));
			name[e - b] = 0;
			if (decl.p && strstr(decl.p, name))
				continue;
			guess_decl(src, name, &decl, notes);
			changed = 1;
		}
		if (!changed && !strcmp(env, "vulkan1.1") && log.p != NULL &&
		    (strstr(log.p, "required extension not requested") ||
		    strstr(log.p, "not supported for this version") ||
		    strstr(log.p, "requires SPIR-V") || strstr(log.p, "SPIR-V 1."))) {
			/* what Vulkan 1.1 has only as an extension (gl_Layer written
			 * by a vertex shader, the integer dot products): 1.3 has it */
			env = "vulkan1.3";
			continue;
		}
		if (!changed) {
			/* the errors, without the banner */
			for (l = log.p ? log.p : ""; *l; ) {
				const char *nl = strchr(l, '\n');
				size_t len = nl ? (size_t)(nl - l) : strlen(l);
				if (!strncmp(l, "ERROR:", 6))
					sb_printf(notes, "  %.*s\n", (int)len, l);
				l += len + (nl != NULL);
			}
			break;
		}
	}
	free(body.p);
	free(decl.p);
	free(full.p);
	free(log.p);
	return ok;
}
