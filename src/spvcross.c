/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * spirv-cross: a simplified module back to GLSL, to read what simplify
 * did.  The rewritten module is assembled (spirv-as), validated
 * (spirv-val) and cross-compiled (spirv-cross --vulkan-semantics, at the
 * source's #version, ES when it was ES); the module as spirv-opt -O left
 * it goes the same way, and a unified diff of the two shows the lines
 * that changed.  The GLSL is spirv-cross's reconstruction (its names, its
 * temporaries), not the user's source.
 *
 * spirv-cross is an external tool like glslangValidator: looked up on
 * PATH (UCAS_SPIRV_CROSS names another one, a path or a name on PATH;
 * "none" turns the view off without a word, as the tests do).  When it is missing or fails, a note
 * says so and simplify goes on without the view.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ucas.h"
#include "json.h"
#include "report.h"

/* the spirv-cross to run, or NULL (*silent: turned off on purpose) */
const char *
spvcross_tool(int *silent)
{
	static char path[1024];
	const char *env = getenv("UCAS_SPIRV_CROSS"), *p, *e, *name =
	    "spirv-cross";

	*silent = 0;
	if (env != NULL) {
		if (*env == 0 || !strcmp(env, "none")) {
			*silent = 1;
			return NULL;
		}
		if (strchr(env, '/'))	/* a path; else a name on PATH */
			return access(env, X_OK) == 0 ? env : NULL;
		name = env;
	}
	for (p = getenv("PATH"); p && *p; p = *e ? e + 1 : e) {
		e = strchr(p, ':');
		if (e == NULL)
			e = p + strlen(p);
		snprintf(path, sizeof path, "%.*s/%s", (int)(e - p), p, name);
		if (e > p && access(path, X_OK) == 0)
			return path;
	}
	return NULL;
}

static char *
read_file(const char *path)
{
	FILE *fp = fopen(path, "rb");
	struct sbuf b = {0};
	char buf[4096];
	size_t n;

	if (fp == NULL)
		return NULL;
	while ((n = fread(buf, 1, sizeof buf, fp)) > 0)
		sb_printf(&b, "%.*s", (int)n, buf);
	fclose(fp);
	sb_puts(&b, "");
	return b.p;
}

/* the first line of a tool's complaint */
static void
first_line(struct sbuf *why, const char *what, const char *log)
{
	const char *e;
	size_t l;

	log = log ? log : "";
	while (*log == '\n' || *log == ' ')
		log++;
	e = strchr(log, '\n');
	l = e ? (size_t)(e - log) : strlen(log);
	if (l > 200)
		l = 200;
	sb_printf(why, "%s%s%.*s", what, l ? ": " : "", (int)l, log);
}

/* one module (assembly) to GLSL; 0 with why on failure */
static int
cross(const char *tool, const char *asmtext, struct sbuf *glsl,
    struct sbuf *why)
{
	char src[] = "/tmp/ucas-xs-XXXXXX.spvasm";
	char bin[] = "/tmp/ucas-xs-XXXXXX.spv";
	char out[] = "/tmp/ucas-xs-XXXXXX.glsl";
	char ver[16] = "450", env[32] = "";
	const char *v;
	int es = 0, ok = 0, rc;
	struct sbuf log = {0};

	if ((v = strstr(asmtext, "OpSource ESSL ")) != NULL) {
		es = 1;
		snprintf(ver, sizeof ver, "%d", atoi(v + 14));
	} else if ((v = strstr(asmtext, "OpSource GLSL ")) != NULL)
		snprintf(ver, sizeof ver, "%d", atoi(v + 14));
	if (atoi(ver) <= 0)
		snprintf(ver, sizeof ver, "%s", es ? "310" : "450");
	if ((v = strstr(asmtext, "target-env vulkan1.")) != NULL &&
	    isdigit((unsigned char)v[19]))
		snprintf(env, sizeof env, "vulkan1.%c", v[19]);
	if (tool_tmp(src, asmtext, strlen(asmtext)) < 0 ||
	    tool_tmp(bin, "", 0) < 0 || tool_tmp(out, "", 0) < 0) {
		sb_puts(why, "cannot write a temporary file here");
		unlink(src);
		unlink(bin);
		return 0;
	}
	{
		char *a1[] = {"spirv-as", "--target-env", env, src, "-o", bin, NULL};
		char *a2[] = {"spirv-as", src, "-o", bin, NULL};
		rc = tool_run(env[0] ? a1 : a2, &log);
	}
	if (rc != 0) {
		first_line(why, rc == 127 || rc < 0 ? "spirv-as cannot run here" :
		    "spirv-as rejects it", rc == 127 || rc < 0 ? "" : log.p);
		goto done;
	}
	sb_clear(&log);
	{
		char *a1[] = {"spirv-val", "--target-env", env, bin, NULL};
		char *a2[] = {"spirv-val", bin, NULL};
		rc = tool_run(env[0] ? a1 : a2, &log);
	}
	if (rc != 0) {
		first_line(why, rc == 127 || rc < 0 ? "spirv-val cannot run here" :
		    "spirv-val rejects it", rc == 127 || rc < 0 ? "" : log.p);
		goto done;
	}
	sb_clear(&log);
	{
		char *a1[] = {(char *)tool, "--version", ver, "--es",
		    "--vulkan-semantics", "--output", out, bin, NULL};
		char *a2[] = {(char *)tool, "--version", ver, "--vulkan-semantics",
		    "--output", out, bin, NULL};
		rc = tool_run(es ? a1 : a2, &log);
	}
	if (rc != 0) {
		first_line(why, rc == 127 || rc < 0 ? "spirv-cross cannot run "
		    "here" : "spirv-cross fails", rc == 127 || rc < 0 ? "" :
		    log.p);
		goto done;
	}
	{
		char *g = read_file(out);
		if (g != NULL && *g) {
			sb_puts(glsl, g);
			ok = 1;
		} else
			sb_puts(why, "spirv-cross wrote nothing");
		free(g);
	}
done:
	unlink(src);
	unlink(bin);
	unlink(out);
	free(log.p);
	return ok;
}

/* ---- a unified diff of two texts, by lines (LCS) ---- */

static int
split_lines(const char *s, const char ***lines, int **lens)
{
	int n = 0, cap = 0;
	const char *p = s, *e;

	*lines = NULL;
	*lens = NULL;
	while (*p) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		if (n == cap) {
			cap = cap ? cap * 2 : 64;
			*lines = xrealloc(*lines, (size_t)cap * sizeof **lines);
			*lens = xrealloc(*lens, (size_t)cap * sizeof **lens);
		}
		(*lines)[n] = p;
		(*lens)[n++] = (int)(e - p);
		p = *e ? e + 1 : e;
	}
	return n;
}

#define DIFF_CTX 2

void
text_udiff(const char *a, const char *b, const char *na, const char *nb,
    int maxlines, struct sbuf *out)
{
	const char **la, **lb;
	int *ea, *eb, n, m, i, j, nop = 0, k, shown = 0, more = 0;
	int *dp;
	char *op;	/* ' ', '-', '+' */
	int *ai, *bi;

	n = split_lines(a, &la, &ea);
	m = split_lines(b, &lb, &eb);
	if ((double)(n + 1) * (m + 1) > 4e6) {
		sb_printf(out, "(too long to diff here: %d and %d lines)\n", n, m);
		goto out;
	}
	dp = xcalloc((size_t)(n + 1) * (size_t)(m + 1), sizeof *dp);
#define DP(x, y) dp[(size_t)(x) * (size_t)(m + 1) + (size_t)(y)]
	for (i = n - 1; i >= 0; i--)
		for (j = m - 1; j >= 0; j--)
			DP(i, j) = ea[i] == eb[j] && !memcmp(la[i], lb[j],
			    (size_t)ea[i]) ? DP(i + 1, j + 1) + 1 :
			    DP(i + 1, j) >= DP(i, j + 1) ? DP(i + 1, j) : DP(i, j + 1);
	op = xmalloc((size_t)(n + m + 1));
	ai = xmalloc((size_t)(n + m + 1) * sizeof *ai);
	bi = xmalloc((size_t)(n + m + 1) * sizeof *bi);
	for (i = j = 0; i < n || j < m; ) {
		ai[nop] = i;
		bi[nop] = j;
		if (i < n && j < m && ea[i] == eb[j] && !memcmp(la[i], lb[j],
		    (size_t)ea[i])) {
			op[nop++] = ' ';
			i++;
			j++;
		} else if (i < n && (j == m || DP(i + 1, j) >= DP(i, j + 1))) {
			op[nop++] = '-';
			i++;
		} else {
			op[nop++] = '+';
			j++;
		}
	}
#undef DP
	free(dp);
	for (k = 0; k < nop && op[k] == ' '; k++)
		;
	if (k == nop)
		goto done;
	sb_printf(out, "--- %s\n+++ %s\n", na, nb);
	while (k < nop) {
		/* a hunk: from DIFF_CTX before the change to DIFF_CTX after the
		 * last change closer than 2 * DIFF_CTX + 1 */
		int s = k - DIFF_CTX < 0 ? 0 : k - DIFF_CTX, e = k, q, ca = 0,
		    cb = 0;
		for (q = k; q < nop; q++)
			if (op[q] != ' ') {
				if (q - e > 2 * DIFF_CTX)
					break;
				e = q;
			}
		e = e + DIFF_CTX + 1 > nop ? nop : e + DIFF_CTX + 1;
		for (q = s; q < e; q++) {
			ca += op[q] != '+';
			cb += op[q] != '-';
		}
		if (shown < maxlines) {
			sb_printf(out, "@@ -%d,%d +%d,%d @@\n", ai[s] + 1, ca,
			    bi[s] + 1, cb);
			shown++;
		} else
			more++;
		for (q = s; q < e; q++) {
			if (shown >= maxlines) {
				more++;
				continue;
			}
			if (op[q] == '+')
				sb_printf(out, "+%.*s\n", eb[bi[q]], lb[bi[q]]);
			else
				sb_printf(out, "%c%.*s\n", op[q], ea[ai[q]], la[ai[q]]);
			shown++;
		}
		for (k = e; k < nop && op[k] == ' '; k++)
			;
	}
	if (more)
		sb_printf(out, "... %d more diff line%s\n", more, more == 1 ? "" :
		    "s");
done:
	free(op);
	free(ai);
	free(bi);
out:
	free(la);
	free(lb);
	free(ea);
	free(eb);
}

/*
 * The view: report gets a note and the diff (indented, capped), oglsl and
 * sglsl the two GLSL texts.  1 when the simplified GLSL is there.
 */
int
spvcross_view(const char *orig, const char *simp, struct sbuf *report,
    struct sbuf *oglsl, struct sbuf *sglsl)
{
	struct sbuf why = {0}, d = {0};
	const char *p, *e;
	int silent, have_o;
	const char *tool = spvcross_tool(&silent);

	if (tool == NULL) {
		if (!silent)
			sb_puts(report, "  GLSL view: spirv-cross not found (install "
			    "it, pkg install spirv-cross, or put it on PATH): "
			    "skipped\n");
		return 0;
	}
	if (!cross(tool, simp, sglsl, &why)) {
		sb_printf(report, "  GLSL view: skipped (the rewritten module: "
		    "%s)\n", why.p ? why.p : "?");
		free(why.p);
		return 0;
	}
	sb_clear(&why);
	have_o = orig != NULL && cross(tool, orig, oglsl, &why);
	sb_puts(report, "  GLSL view (spirv-cross's reconstruction of the "
	    "SPIR-V, with its own names, not your source):\n");
	if (!have_o) {
		sb_printf(report, "    the module before the rewrites did not "
		    "cross-compile (%s); simplified.glsl only\n", why.p ? why.p :
		    "?");
		free(why.p);
		return 1;
	}
	text_udiff(oglsl->p, sglsl->p, "original.glsl (spirv-opt -O)",
	    "simplified.glsl", 40, &d);
	if (d.n == 0)
		sb_puts(report, "    the same GLSL before and after (the rewrites "
		    "do not show at this level)\n");
	for (p = d.p ? d.p : ""; *p; p = *e ? e + 1 : e) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		const char *t = e;
		while (t > p && (t[-1] == ' ' || t[-1] == '\t'))
			t--;
		if (t > p)
			sb_printf(report, "    %.*s\n", (int)(t - p), p);
		else
			sb_puts(report, "\n");
	}
	free(d.p);
	free(why.p);
	return 1;
}

/*
 * After simplify wrote simplified.spvasm: the view, printed, and the two
 * GLSL files delivered (written, attached, or in the response's files).
 * The bot's engine runs in a sandbox that cannot run programs: it hands
 * both modules to the bot in an "\x1exspv" block and the bot makes the
 * view (discord/bot.c, with the same spvcross_view).  The bot's JSON path
 * sets spvcross_defer: the block then lands in the response's text, and
 * the bot expands it there.
 */
int spvcross_defer;

void
spvcross_after_simplify(const char *orig, const char *simp, int nofile)
{
	struct sbuf rep = {0}, og = {0}, sg = {0};

	if ((nlp_defer && !rep_on) || spvcross_defer) {
		int silent;
		if (spvcross_tool(&silent) == NULL && silent)
			return;
		printf("\x1e" "xspv %d\n", nofile);
		b64_encode(&rep, (const unsigned char *)(orig ? orig : ""),
		    orig ? strlen(orig) : 0);
		printf("%s\n\x1e" "and\n", rep.p ? rep.p : "");
		sb_clear(&rep);
		b64_encode(&rep, (const unsigned char *)simp, strlen(simp));
		printf("%s\n\x1e" "end\n", rep.p ? rep.p : "");
		free(rep.p);
		return;
	}
	if (spvcross_view(orig, simp, &rep, &og, &sg)) {
		fputs(rep.p, stdout);
		rep_str("glsl_view", rep.p);
		rep_str("glsl", sg.p);
		deliver_file("simplified.glsl", "text/x-glsl", sg.p, sg.n, nofile);
		if (og.n)
			deliver_file("original.glsl", "text/x-glsl", og.p, og.n,
			    nofile);
	} else if (rep.n) {
		fputs(rep.p, stdout);
		rep_str("glsl_view", rep.p);
	}
	free(rep.p);
	free(og.p);
	free(sg.p);
}
