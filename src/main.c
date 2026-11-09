/*
 * ucas [file ...]     run files (echoing each statement), or
 * ucas                read statements from standard input
 * ucas -e 'stmt'      run one statement
 *
 * A line that reads as English is translated to ucas statements first
 * (nlp.c).  -y never asks about ambiguous words (the default reading is
 * taken and noted); -i always asks, even when not on a terminal.
 */
#include "ucas.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *
slurp(FILE *fp)
{
	size_t n = 0, cap = 4096, r;
	char *s = xmalloc(cap);

	while ((r = fread(s + n, 1, cap - n - 1, fp)) > 0) {
		n += r;
		if (cap - n - 1 == 0) {
			cap *= 2;
			s = xrealloc(s, cap);
		}
	}
	s[n] = 0;
	return s;
}

static void
repl(void)
{
	char line[8192];
	struct sbuf buf = {0};
	int depth = 0, tty = isatty(0);
	char *p;

	for (;;) {
		if (tty) {
			fputs(depth > 0 ? "... " : "ucas> ", stdout);
			fflush(stdout);
		}
		if (fgets(line, sizeof line, stdin) == NULL)
			break;
		sb_puts(&buf, line);
		for (p = line; *p; p++) {
			if (*p == '{' || *p == '(' || *p == '[')
				depth++;
			if (*p == '}' || *p == ')' || *p == ']')
				depth--;
		}
		if (depth > 0)
			continue;
		depth = 0;
		run_text(buf.p, "<stdin>", 0);
		sb_clear(&buf);
	}
	if (buf.n)
		run_text(buf.p, "<stdin>", 0);
}

int
main(int argc, char **argv)
{
	int i, ran = 0, echo = 1;

	setvbuf(stdout, NULL, _IOLBF, 0);
	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-q")) {
			echo = 0;
		} else if (!strcmp(argv[i], "-y")) {
			nlp_interactive = 0;
		} else if (!strcmp(argv[i], "-i")) {
			nlp_interactive = 1;
		} else if (!strcmp(argv[i], "-e") && i + 1 < argc) {
			run_text(argv[++i], "<arg>", 0);
			ran = 1;
		} else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
			run_source("help", "<arg>", 0);
			return 0;
		} else {
			FILE *fp = fopen(argv[i], "r");
			char *src;
			if (fp == NULL) {
				perror(argv[i]);
				return 1;
			}
			src = slurp(fp);
			fclose(fp);
			run_text(src, argv[i], echo);
			free(src);
			ran = 1;
		}
	}
	if (!ran)
		repl();
	return 0;
}
