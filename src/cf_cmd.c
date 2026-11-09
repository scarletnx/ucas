/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: the commands.
 *
 *   c optimize [FILE] [target T] [budget S] [assume NAME LO HI]
 *              [wrapv] [check] [bench] [nofile]
 *   c lower [FILE] [wrapv]      each function as one ucas value, or why not
 *   c targets                   the x86-64 targets and their features
 *
 * FILE, or a ```c block after the statement.  English ("optimize this C
 * for zen4") reaches here through nlp.c's code-block hook.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfront.h"

int cf_check_all(struct cf_unit *u, const char *orig, const char *opt,
    const struct cf_opts *o, struct sbuf *rep, int bench,
    struct cf_result *r);

static char *
slurp(const char *path)
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

static int
load(struct cf_unit *u, const char *src, const char *path,
    const struct cf_target *t)
{
	memset(u, 0, sizeof *u);
	if (!cf_preprocess(u, path, src, t) || !cf_parse(u)) {
		printf("error: %s\n", u->err.p ? u->err.p : "?");
		rep_error("c_syntax", u->err.p ? u->err.p : "?");
		return 0;
	}
	return 1;
}

static void
print_lines(const char *pfx, const char *s, int max)
{
	int n = 0;

	while (*s && n < max) {
		const char *e = strchr(s, '\n');
		if (e == NULL)
			e = s + strlen(s);
		printf("%s%.*s\n", pfx, (int)(e - s), s);
		n++;
		s = *e ? e + 1 : e;
	}
	if (*s)
		printf("%s...\n", pfx);
}

static void
cmd_lower(const char *src, const char *path, const struct cf_opts *o)
{
	struct cf_unit u;
	int k;

	if (!load(&u, src, path, o->t))
		return;
	rep_begin("c_lower");
	rep_list("functions");
	for (k = 0; k < u.nfn; k++) {
		uint32_t fn = u.fns[k];
		const char *nm = sym_str(u.S.name[u.A.iv[fn]]);
		struct cf_value v;
		if (u.tk.file[u.A.t0[fn]] != 0)
			continue;
		rep_obj(NULL);
		rep_str("function", nm);
		if (cf_lower_func(&u, fn, &v, !o->wrapv)) {
			int j;
			printf("  %s(", nm);
			for (j = 0; j < v.nleaf; j++)
				printf("%s%s: %s", j ? ", " : "", v.leaf[j].text,
				    ty_str(ct_to_ucas(v.leaf[j].cty)));
			printf(") = %s\n", shared_str(v.e, 1500));
			rep_expr("value", v.e);
			if (v.where && !is_const(v.where)) {
				printf("    defined where (no C UB): %s\n",
				    shared_str(v.where, 600));
				rep_expr("where", v.where);
			}
		} else {
			printf("  %s: not lowered: %s\n", nm, v.why);
			rep_str("not_lowered", v.why);
			free((char *)v.why);
		}
		rep_end();
	}
	rep_end();
	cf_unit_free(&u);
}

void
cfront_run(const char *src, const char *path, const struct cf_opts *o)
{
	struct cf_unit u;
	struct cf_result r;
	struct sbuf rep = {0};
	int k, napplied = 0, nex = 0, nsat = 0, ntriv = 0;
	int checked = 0;

	if (!load(&u, src, path, o->t))
		return;
	cf_optimize(&u, o, &r);
	if (o->check || o->bench)
		checked = cf_check_all(&u, src, r.out.p, o, &rep, o->bench, &r);
	for (k = 0; k < r.nrw; k++)
		if (r.rw[k].applied) {
			napplied++;
			nex += r.rw[k].proof == CF_PROOF_EXHAUSTIVE;
			nsat += r.rw[k].proof == CF_PROOF_SAT;
			ntriv += r.rw[k].trivial;
		}
	rep_begin("c_optimize");
	rep_str("target", o->t->name);
	printf("c optimize: %d function%s, %d rewrite%s proved (%d exhaustive, "
	    "%d by SAT) for %s in %.1f s\n", u.nfn, u.nfn == 1 ? "" : "s",
	    napplied, napplied == 1 ? "" : "s", nex, nsat, o->t->name,
	    r.seconds);
	if (napplied)
		printf("  estimated cost of the rewritten values on %s: %.1f -> "
		    "%.1f cycles\n", o->t->name, r.cost_before, r.cost_after);
	if (ntriv)
		printf("  of these, %d give the same machine code as clang -O2 "
		    "makes of the original (listed apart)\n", ntriv);
	rep_int("rewrites", napplied);
	rep_int("exhaustive", nex);
	rep_int("sat", nsat);
	rep_int("same_as_clang", ntriv);
	rep_num("cost_before", r.cost_before);
	rep_num("cost_after", r.cost_after);
	rep_list("rewrites");
	{
		int pass;
		for (pass = 0; pass < 3; pass++)
			for (k = 0; k < r.nrw; k++) {
				struct cf_rewrite *w = &r.rw[k];
				int cls = w->info ? 2 : w->trivial ? 1 : 0;
				if (cls != pass)
					continue;
				if (pass == 1 && k == 0)
					;
				rep_obj(NULL);
				rep_str("function", w->func);
				rep_int("line", w->line);
				if (w->info) {
					printf("  %s (line %d): %s: %s\n", w->func, w->line,
					    w->kind, w->proof_text);
					rep_str("status", w->kind);
					rep_str("why", w->proof_text);
					rep_end();
					continue;
				}
				printf("  %s (line %d)%s: %s\n", w->func, w->line,
				    w->trivial ? " [same code as clang -O2]" : "",
				    w->steps ? w->steps : w->kind);
				printf("    proof: %s (%s); cost %.1f -> %.1f\n",
				    cf_proof_name(w->proof), w->proof_text,
				    w->cost_before, w->cost_after);
				if (strchr(w->before, '\n') == NULL &&
				    strlen(w->before) < 200) {
					printf("    before: %s\n", w->before);
				}
				if (strchr(w->after, '\n') == NULL)
					printf("    after:  %s\n", w->after);
				rep_str("what", w->steps ? w->steps : w->kind);
				rep_str("proof", cf_proof_name(w->proof));
				rep_str("proof_text", w->proof_text);
				rep_num("cost_before", w->cost_before);
				rep_num("cost_after", w->cost_after);
				rep_str("before", w->before);
				rep_str("after", w->after);
				rep_bool("same_as_clang", w->trivial);
				rep_end();
			}
	}
	rep_end();
	if (rep.n)
		fputs(rep.p, stdout);
	if (o->check || o->bench)
		rep_bool("check_passed", checked > 0);
	if (o->nofile || !path || !strcmp(path, "<block>")) {
		printf("optimised C:\n");
		print_lines("  ", r.out.p, 400);
		rep_str("code", r.out.p);
	} else {
		char outp[1024];
		const char *dot = strrchr(path, '.');
		FILE *fp;
		snprintf(outp, sizeof outp, "%.*s.opt.c", dot ? (int)(dot - path) :
		    (int)strlen(path), path);
		if (rep_on) {
			rep_file(strrchr(outp, '/') ? strrchr(outp, '/') + 1 : outp,
			    "text/x-c", r.out.p, r.out.n);
			rep_str("code", r.out.p);
		} else if ((fp = fopen(outp, "w")) != NULL) {
			fputs(r.out.p, fp);
			fclose(fp);
			printf("wrote %s\n", outp);
		} else
			printf("error: cannot write %s\n", outp);
	}
	free(rep.p);
	cf_result_free(&r);
	cf_unit_free(&u);
}

static void
cmd_targets(void)
{
	int k, f;
	const struct cf_target *t;

	rep_begin("c_targets");
	rep_list("targets");
	for (k = 0; (t = cf_target_nth(k)) != NULL; k++) {
		printf("  %-15s -march=%-15s %s:", t->name, t->march, t->desc);
		rep_obj(NULL);
		rep_str("name", t->name);
		rep_str("march", t->march);
		for (f = 0; f < CF_NFEAT; f++)
			if (t->feat & CF_F(f))
				printf(" %s", cf_feat_name(f));
		printf("\n");
		rep_end();
	}
	rep_end();
}

/* options from the rest of the line; 0 on a bad option */
static int
parse_opts(char *s, struct cf_opts *o, char *file, size_t fn, int *lower)
{
	char *w, *save = NULL;

	file[0] = 0;
	for (w = strtok_r(s, " \t", &save); w; w = strtok_r(NULL, " \t", &save)) {
		if (!strcmp(w, "target") || !strcmp(w, "for") || !strcmp(w, "on")) {
			const struct cf_target *t;
			w = strtok_r(NULL, " \t", &save);
			if (w == NULL || (t = cf_target_find(w)) == NULL) {
				printf("error: c: unknown target %s (c targets lists "
				    "them)\n", w ? w : "");
				return 0;
			}
			o->t = t;
		} else if (!strcmp(w, "budget")) {
			w = strtok_r(NULL, " \t", &save);
			o->budget = w ? atof(w) : 10;
			if (o->budget < 0.5)
				o->budget = 0.5;
			if (o->budget > 120)
				o->budget = 120;
		} else if (!strcmp(w, "assume")) {
			char *nm = strtok_r(NULL, " \t", &save), *a, *b;
			a = strtok_r(NULL, " \t", &save);
			if (a && !strcmp(a, "in"))
				a = strtok_r(NULL, " \t", &save);
			b = a && strstr(a, "..") ? strstr(a, "..") + 2 :
			    strtok_r(NULL, " \t", &save);
			if (!nm || !a || !b || o->nassume == 16) {
				printf("error: c: assume NAME LO HI\n");
				return 0;
			}
			snprintf(o->assume[o->nassume].name,
			    sizeof o->assume[o->nassume].name, "%s", nm);
			o->assume[o->nassume].lo = strtoll(a, NULL, 0);
			o->assume[o->nassume].hi = strtoll(b, NULL, 0);
			o->nassume++;
		} else if (!strcmp(w, "wrapv"))
			o->wrapv = 1;
		else if (!strcmp(w, "check"))
			o->check = 1;
		else if (!strcmp(w, "bench"))
			o->bench = 1;
		else if (!strcmp(w, "nofile"))
			o->nofile = 1;
		else if (!strcmp(w, "lower"))
			*lower = 1;
		else if (!strcmp(w, "this") || !strcmp(w, "code") ||
		    !strcmp(w, "C") || !strcmp(w, "c"))
			;
		else if (file[0] == 0 && (strchr(w, '.') || strchr(w, '/')))
			snprintf(file, fn, "%s", w);
		else {
			printf("error: c: unknown option '%s'\n", w);
			return 0;
		}
	}
	return 1;
}

int
cfront_line(const char *line)
{
	char rest[2048], file[1024];
	struct cf_opts o;
	int lower = 0;
	char *src, *w;

	memset(&o, 0, sizeof o);
	o.t = cf_target_default();
	o.budget = 10;
	snprintf(rest, sizeof rest, "%s", line);
	w = rest;
	while (*w == ' ')
		w++;
	if (!strncmp(w, "targets", 7)) {
		cmd_targets();
		return 1;
	}
	if (!strncmp(w, "lower", 5)) {
		lower = 1;
		w += 5;
	} else if (!strncmp(w, "optimize", 8) || !strncmp(w, "optimise", 8))
		w += 8;
	else
		fail("c: optimize, lower or targets");
	if (!parse_opts(w, &o, file, sizeof file, &lower))
		return 1;
	if (file[0]) {
		src = slurp(file);
		if (src == NULL)
			fail("c: cannot read %s", file);
	} else if (code_block) {
		src = strdup(code_block);
		code_block_used = 1;
		snprintf(file, sizeof file, "<block>");
	} else
		fail("c: a FILE or a ```c block");
	if (lower)
		cmd_lower(src, file, &o);
	else
		cfront_run(src, file, &o);
	free(src);
	return 1;
}

/*
 * nlp.c: a ```c block whose statement is a C command or reads as one
 * ("optimize this C", "c optimize for zen4"): it stays C, not GLSL.
 */
int
cfront_claims(const char *stmt, const char *lang)
{
	const char *s = stmt;

	if (s == NULL)
		return 0;
	while (*s && isspace((unsigned char)*s))
		s++;
	if ((s[0] == 'c' || s[0] == 'C') && isspace((unsigned char)s[1]))
		return 1;
	if (lang && (!strcmp(lang, "c") || !strcmp(lang, "h")) &&
	    (strstr(s, "optimi") || strstr(s, "lower")) &&
	    !strstr(s, "glsl") && !strstr(s, "GLSL") && !strstr(s, "shader"))
		return 1;
	return 0;
}

/* English: "optimize this C [for zen4] [with check]" -> "c optimize ..." */
int cfront_english(const char *stmt, struct sbuf *out);

int
cfront_english(const char *stmt, struct sbuf *out)
{
	const char *const words[] = {"check", "bench", "nofile", "wrapv", NULL};
	int k;
	const struct cf_target *t;

	if (!cfront_claims(stmt, "c"))
		return 0;
	while (*stmt && isspace((unsigned char)*stmt))
		stmt++;
	if ((stmt[0] == 'c' || stmt[0] == 'C') && isspace((unsigned char)stmt[1]))
		return 0;	/* already a command */
	sb_puts(out, strstr(stmt, "lower") ? "c lower" : "c optimize");
	for (k = 0; (t = cf_target_nth(k)) != NULL; k++) {
		const char *q = strstr(stmt, t->name);
		if (q && (q == stmt || !isalnum((unsigned char)q[-1])) &&
		    !isalnum((unsigned char)q[strlen(t->name)]) &&
		    q[strlen(t->name)] != '-') {
			sb_printf(out, " target %s", t->name);
			break;
		}
	}
	for (k = 0; words[k]; k++)
		if (strstr(stmt, words[k]))
			sb_printf(out, " %s", words[k]);
	if (strstr(stmt, "test") && !strstr(out->p, "check"))
		sb_puts(out, " check");
	return 1;
}
