/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfront: the preprocessor and lexer (a C99 subset, enough for kernels).
 *
 * Object and function-like macros (# and ##, __VA_ARGS__), #if/#ifdef/
 * #ifndef/#elif/#else/#endif with defined(), #include "file" (read
 * relative to the including file) and #include <header>: the header's
 * line is kept for the output, and known headers (stdint.h, stddef.h,
 * stdbool.h, limits.h, string.h, math.h, immintrin.h ...) bring a small
 * built-in prelude of declarations.  #pragma, #line, #error, #warning are
 * skipped (#error inside a live group is an error).
 *
 * Every output token remembers where it came from: its own file offset,
 * and for tokens out of a macro expansion the source extent of the
 * outermost invocation in the main file, so that an AST node can be
 * mapped back to one contiguous piece of the original text (cf_span).
 *
 * Predefined: __STDC__, __STDC_VERSION__ 199901L, __STDC_HOSTED__,
 * __x86_64__, __x86_64, __amd64__, __LP64__, _LP64, __CHAR_BIT__,
 * __SIZEOF_{INT,LONG,POINTER,...}__, __ORDER_LITTLE_ENDIAN__,
 * __BYTE_ORDER__, and the target's feature macros (__SSE2__, __AVX2__,
 * __BMI2__, __POPCNT__ ...).  __GNUC__ is not defined: GNU extensions
 * are not parsed, so headers should take their portable paths.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfront.h"

struct ptok {
	uint8_t kind, fl;
	uint16_t file;
	uint32_t off, len, line, col, sym;
	uint32_t o0, o1, grp;	/* origin extent in the main file, group */
};

struct ptbuf {
	struct ptok *t;
	uint32_t n, cap;
};

static void
pt_push(struct ptbuf *b, const struct ptok *t)
{
	if (b->n == b->cap) {
		b->cap = newcap(b->cap, b->n + 1);
		b->t = xrealloc(b->t, b->cap * sizeof *b->t);
	}
	b->t[b->n++] = *t;
}

struct macro {
	uint32_t name;
	int fn, variadic, nparam;
	uint32_t param[32];
	struct ptbuf body;
	int active;
	int undef;
};

static struct {
	struct cf_unit *u;
	struct macro *m;
	int nm, mcap;
	uint32_t *ht;		/* name -> macro + 1 */
	uint32_t htcap;
	struct ptbuf syn;	/* synthetic text: file nfiles-1 */
	struct sbuf syntext;
	uint16_t synfile;
	uint32_t ngrp;
	uint8_t *grpbad;
	uint32_t grpcap;
	int depth;		/* include depth */
	const char *err;
	char errbuf[300];
	int known_hdr;		/* bit set of preludes already included */
} PP;

static uint32_t s_define, s_undef, s_include, s_if, s_ifdef, s_ifndef,
    s_elif, s_else, s_endif, s_defined, s_va, s_pragma, s_line, s_error,
    s_warning, s_hash, s_hashhash, s_lp, s_rp, s_comma, s_ellipsis;

static void
pp_syms(void)
{
	s_define = sym_cstr("define");
	s_undef = sym_cstr("undef");
	s_include = sym_cstr("include");
	s_if = sym_cstr("if");
	s_ifdef = sym_cstr("ifdef");
	s_ifndef = sym_cstr("ifndef");
	s_elif = sym_cstr("elif");
	s_else = sym_cstr("else");
	s_endif = sym_cstr("endif");
	s_defined = sym_cstr("defined");
	s_va = sym_cstr("__VA_ARGS__");
	s_pragma = sym_cstr("pragma");
	s_line = sym_cstr("line");
	s_error = sym_cstr("error");
	s_warning = sym_cstr("warning");
	s_hash = sym_cstr("#");
	s_hashhash = sym_cstr("##");
	s_lp = sym_cstr("(");
	s_rp = sym_cstr(")");
	s_comma = sym_cstr(",");
	s_ellipsis = sym_cstr("...");
}

static void
pp_error(const struct ptok *t, const char *fmt, const char *a)
{
	char m[200];

	if (PP.err)
		return;
	snprintf(m, sizeof m, fmt, a);
	if (t)
		snprintf(PP.errbuf, sizeof PP.errbuf, "%s:%u:%u: %s",
		    PP.u->fname[t->file], t->line, t->col, m);
	else
		snprintf(PP.errbuf, sizeof PP.errbuf, "%s", m);
	PP.err = PP.errbuf;
}

/* ---- files ---- */

static int
add_file(const char *name, char *text)
{
	struct cf_unit *u = PP.u;

	u->ftext = xrealloc(u->ftext, (size_t)(u->nfiles + 1) * sizeof *u->ftext);
	u->fname = xrealloc(u->fname, (size_t)(u->nfiles + 1) * sizeof *u->fname);
	u->ftext[u->nfiles] = text;
	u->fname[u->nfiles] = strdup(name);
	return u->nfiles++;
}

/* ---- the raw lexer ---- */

static const char *const puncts[] = {
	"...", "<<=", ">>=", "->", "++", "--", "<<", ">>", "<=", ">=", "==",
	"!=", "&&", "||", "*=", "/=", "%=", "+=", "-=", "&=", "^=", "|=", "##",
	"[", "]", "(", ")", "{", "}", ".", "&", "*", "+", "-", "~", "!", "/",
	"%", "<", ">", "^", "|", "?", ":", ";", "=", ",", "#", NULL
};

static void
lex_file(int file, struct ptbuf *out)
{
	const char *s = PP.u->ftext[file], *p = s;
	uint32_t line = 1;
	const char *lstart = s;
	int bol = 1, space = 0;

	while (*p) {
		struct ptok t;
		const char *st;
		int k;
		if (*p == '\n') {
			p++;
			line++;
			lstart = p;
			bol = 1;
			space = 0;
			continue;
		}
		if (*p == '\\' && p[1] == '\n') {
			p += 2;
			line++;
			lstart = p;
			continue;
		}
		if (*p == '\\' && p[1] == '\r' && p[2] == '\n') {
			p += 3;
			line++;
			lstart = p;
			continue;
		}
		if (isspace((unsigned char)*p)) {
			p++;
			space = 1;
			continue;
		}
		if (p[0] == '/' && p[1] == '/') {
			while (*p && *p != '\n')
				p++;
			space = 1;
			continue;
		}
		if (p[0] == '/' && p[1] == '*') {
			p += 2;
			while (*p && !(p[0] == '*' && p[1] == '/')) {
				if (*p == '\n') {
					line++;
					lstart = p + 1;
				}
				p++;
			}
			if (*p)
				p += 2;
			space = 1;
			continue;
		}
		memset(&t, 0, sizeof t);
		t.file = (uint16_t)file;
		t.off = (uint32_t)(p - s);
		t.line = line;
		t.col = (uint32_t)(p - lstart) + 1;
		t.fl = (uint8_t)((bol ? CTF_BOL : 0) | (space ? CTF_SPACE : 0));
		st = p;
		if (isalpha((unsigned char)*p) || *p == '_' ||
		    ((*p == 'L' || *p == 'u' || *p == 'U') && (p[1] == '\'' ||
		    p[1] == '"'))) {
			if ((*p == 'L' || *p == 'u' || *p == 'U') &&
			    (p[1] == '\'' || p[1] == '"'))
				goto quoted;
			while (isalnum((unsigned char)*p) || *p == '_' || *p == '$')
				p++;
			t.kind = CK_ID;
		} else if (isdigit((unsigned char)*p) || (*p == '.' &&
		    isdigit((unsigned char)p[1]))) {
			p++;
			for (;;) {
				if ((*p == '+' || *p == '-') && (p[-1] == 'e' ||
				    p[-1] == 'E' || p[-1] == 'p' || p[-1] == 'P'))
					p++;
				else if (isalnum((unsigned char)*p) || *p == '.' ||
				    *p == '_')
					p++;
				else
					break;
			}
			t.kind = CK_NUM;
		} else if (*p == '\'' || *p == '"') {
			char q;
quoted:
			if (*p != '\'' && *p != '"')
				p++;
			q = *p++;
			while (*p && *p != q && *p != '\n') {
				if (*p == '\\' && p[1])
					p++;
				p++;
			}
			if (*p == q)
				p++;
			t.kind = q == '"' ? CK_STR : CK_CHAR;
		} else {
			for (k = 0; puncts[k]; k++) {
				size_t l = strlen(puncts[k]);
				if (!strncmp(p, puncts[k], l)) {
					p += l;
					break;
				}
			}
			if (!puncts[k])
				p++;	/* a stray character: its own token */
			t.kind = CK_PUNCT;
		}
		t.len = (uint32_t)(p - st);
		t.sym = sym_intern(st, t.len);
		t.o0 = t.off;
		t.o1 = t.off + t.len;
		pt_push(out, &t);
		bol = 0;
		space = 0;
	}
}

/* ---- macros ---- */

static struct macro *
mac_find(uint32_t name)
{
	uint32_t j;

	if (!PP.htcap)
		return NULL;
	j = (name * 2654435761u) & (PP.htcap - 1);
	while (PP.ht[j]) {
		struct macro *m = &PP.m[PP.ht[j] - 1];
		if (m->name == name)
			return m->undef ? NULL : m;
		j = (j + 1) & (PP.htcap - 1);
	}
	return NULL;
}

static struct macro *
mac_new(uint32_t name)
{
	uint32_t j, i;
	struct macro *m;

	for (i = 0; i < (uint32_t)PP.nm; i++)
		if (PP.m[i].name == name) {
			m = &PP.m[i];
			free(m->body.t);
			memset(m, 0, sizeof *m);
			m->name = name;
			return m;
		}
	if ((uint32_t)(PP.nm + 1) * 2 > PP.htcap) {
		uint32_t cap = PP.htcap ? PP.htcap * 2 : 256;
		free(PP.ht);
		PP.ht = xcalloc(cap, sizeof *PP.ht);
		PP.htcap = cap;
		for (i = 0; i < (uint32_t)PP.nm; i++) {
			j = (PP.m[i].name * 2654435761u) & (cap - 1);
			while (PP.ht[j])
				j = (j + 1) & (cap - 1);
			PP.ht[j] = i + 1;
		}
	}
	if (PP.nm == PP.mcap) {
		PP.mcap = PP.mcap ? PP.mcap * 2 : 128;
		PP.m = xrealloc(PP.m, (size_t)PP.mcap * sizeof *PP.m);
	}
	m = &PP.m[PP.nm];
	memset(m, 0, sizeof *m);
	m->name = name;
	j = (name * 2654435761u) & (PP.htcap - 1);
	while (PP.ht[j])
		j = (j + 1) & (PP.htcap - 1);
	PP.ht[j] = (uint32_t)++PP.nm;
	{
		struct cf_unit *u = PP.u;
		u->macros = xrealloc(u->macros, (size_t)(u->nmacro + 1) *
		    sizeof *u->macros);
		u->macros[u->nmacro++] = name;
	}
	return m;
}

/* a token with new text (pasting, stringizing, numbers from #if) */
static struct ptok
synth(const char *s, size_t n, const struct ptok *like)
{
	struct ptok t = *like;
	struct cf_unit *u = PP.u;
	char *base0 = PP.syntext.p;

	t.file = PP.synfile;
	t.off = (uint32_t)PP.syntext.n;
	t.len = (uint32_t)n;
	sb_printf(&PP.syntext, "%.*s", (int)n, s);
	sb_putc(&PP.syntext, 0);
	if (PP.syntext.p != base0)
		u->ftext[PP.synfile] = PP.syntext.p;
	t.sym = sym_intern(s, n);
	if (n && (isalpha((unsigned char)*s) || *s == '_'))
		t.kind = CK_ID;
	else if (n && (isdigit((unsigned char)*s) || (*s == '.' && n > 1)))
		t.kind = CK_NUM;
	else if (n && *s == '"')
		t.kind = CK_STR;
	else if (n && *s == '\'')
		t.kind = CK_CHAR;
	else
		t.kind = CK_PUNCT;
	return t;
}

static const char *
ptext(const struct ptok *t)
{
	return PP.u->ftext[t->file] + t->off;
}

/* ---- the reader: a stack of token frames ---- */

struct frame {
	struct ptok *t;
	uint32_t n, i;
	struct macro *m;	/* re-enabled when the frame ends */
	int owned, file;	/* file frame: directives are seen */
};

struct reader {
	struct frame f[256];
	int nf;
};

static void
rd_push(struct reader *r, struct ptok *t, uint32_t n, struct macro *m,
    int owned, int file)
{
	struct frame *f;

	if (r->nf == 256) {
		pp_error(NULL, "macro expansion too deep%s", "");
		if (owned)
			free(t);
		return;
	}
	f = &r->f[r->nf++];
	f->t = t;
	f->n = n;
	f->i = 0;
	f->m = m;
	f->owned = owned;
	f->file = file;
	if (m)
		m->active = 1;
}

static void
rd_pop(struct reader *r)
{
	struct frame *f = &r->f[--r->nf];

	if (f->m)
		f->m->active = 0;
	if (f->owned)
		free(f->t);
}

/* the next token, popping finished expansion frames (not file frames) */
static struct ptok *
rd_peek(struct reader *r)
{
	while (r->nf > 0) {
		struct frame *f = &r->f[r->nf - 1];
		if (f->i < f->n)
			return &f->t[f->i];
		if (f->file)
			return NULL;
		rd_pop(r);
	}
	return NULL;
}

static int
rd_infile(struct reader *r)
{
	return r->nf > 0 && r->f[r->nf - 1].file;
}

static void expand_into(struct ptbuf *in, struct ptbuf *out, uint32_t grp,
    uint32_t o0, uint32_t o1);

static uint32_t
new_grp(void)
{
	if (PP.ngrp + 1 >= PP.grpcap) {
		uint32_t c = newcap(PP.grpcap, PP.ngrp + 2);
		PP.grpbad = xrealloc(PP.grpbad, c);
		memset(PP.grpbad + PP.grpcap, 0, c - PP.grpcap);
		PP.grpcap = c;
	}
	return ++PP.ngrp;
}

/* stringize raw argument tokens */
static struct ptok
stringize(const struct ptok *t, uint32_t n, const struct ptok *like)
{
	struct sbuf b = {0};
	struct ptok r;
	uint32_t i, k;

	sb_putc(&b, '"');
	for (i = 0; i < n; i++) {
		const char *s = ptext(&t[i]);
		if (i && (t[i].fl & CTF_SPACE))
			sb_putc(&b, ' ');
		for (k = 0; k < t[i].len; k++) {
			if ((t[i].kind == CK_STR || t[i].kind == CK_CHAR) &&
			    (s[k] == '"' || s[k] == '\\'))
				sb_putc(&b, '\\');
			sb_putc(&b, s[k]);
		}
	}
	sb_putc(&b, '"');
	r = synth(b.p, b.n, like);
	free(b.p);
	return r;
}

/*
 * Expand macro m whose name token is nt; for a function-like macro the
 * reader is at the '(' (already checked).  The replacement goes on the
 * reader as a new frame.
 */
static void
expand_macro(struct reader *r, struct macro *m, struct ptok nt)
{
	struct ptbuf args[33], body = {0}, *res;
	uint32_t i, nargs = 0, grp, o0, o1;
	int bad = 0;

	memset(args, 0, sizeof args);
	if (nt.grp) {
		grp = nt.grp;
		o0 = nt.o0;
		o1 = nt.o1;
	} else {
		grp = nt.file == 0 ? new_grp() : 0;
		o0 = nt.off;
		o1 = nt.off + nt.len;
	}
	if (m->fn) {
		int depth = 0;
		struct ptok *t;
		/* the '(' */
		t = rd_peek(r);
		r->f[r->nf - 1].i++;
		for (;;) {
			t = rd_peek(r);
			if (t == NULL) {
				pp_error(&nt, "unterminated call of macro %s",
				    sym_str(m->name));
				break;
			}
			r->f[r->nf - 1].i++;
			if (t->sym == s_lp)
				depth++;
			else if (t->sym == s_rp) {
				if (depth-- == 0) {
					if (t->grp != nt.grp)
						bad = 1;
					if (!t->grp && t->file == 0)
						o1 = t->off + t->len;
					break;
				}
			} else if (t->sym == s_comma && depth == 0 &&
			    !(m->variadic && nargs + 1 >= (uint32_t)m->nparam)) {
				if (nargs < 32)
					nargs++;
				continue;
			}
			if (t->grp != nt.grp)
				bad = 1;
			pt_push(&args[nargs], t);
		}
		nargs++;
		if (m->nparam == 0 && nargs == 1 && args[0].n == 0)
			nargs = 0;
		if ((int)nargs != m->nparam && !(m->variadic &&
		    (int)nargs == m->nparam - 1))
			pp_error(&nt, "macro %s: wrong number of arguments",
			    sym_str(m->name));
	}
	if (bad && nt.grp)
		PP.grpbad[nt.grp] = 1;
	/* substitute */
	for (i = 0; i < m->body.n; i++) {
		struct ptok t = m->body.t[i];
		int pi = -1, k;
		t.fl = (uint8_t)((t.fl & CTF_SPACE) | CTF_MACRO);
		if (m->fn && t.sym == s_hash && i + 1 < m->body.n) {
			for (k = 0; k < m->nparam; k++)
				if (m->param[k] == m->body.t[i + 1].sym)
					pi = k;
			if (pi >= 0) {
				struct ptok s = stringize(args[pi].t, args[pi].n, &t);
				s.fl = t.fl;
				pt_push(&body, &s);
				i++;
				continue;
			}
		}
		if (m->fn && t.kind == CK_ID)
			for (k = 0; k < m->nparam; k++)
				if (m->param[k] == t.sym)
					pi = k;
		if (pi >= 0) {
			int raw = (i > 0 && m->body.t[i - 1].sym == s_hashhash) ||
			    (i + 1 < m->body.n && m->body.t[i + 1].sym == s_hashhash);
			uint32_t j0 = body.n, j;
			if (raw) {
				for (j = 0; j < args[pi].n; j++)
					pt_push(&body, &args[pi].t[j]);
			} else {
				struct ptbuf ex = {0};
				expand_into(&args[pi], &ex, 0, 0, 0);
				for (j = 0; j < ex.n; j++)
					pt_push(&body, &ex.t[j]);
				free(ex.t);
			}
			for (j = j0; j < body.n; j++)
				body.t[j].fl |= CTF_MACRO;
			if (j0 < body.n)
				body.t[j0].fl = (uint8_t)((body.t[j0].fl & ~CTF_SPACE) |
				    (t.fl & CTF_SPACE));
			if (raw && args[pi].n == 0) {
				/* a placemarker: an empty token */
				struct ptok e = synth("", 0, &t);
				pt_push(&body, &e);
			}
			continue;
		}
		pt_push(&body, &t);
	}
	/* ## */
	for (i = 0; i < body.n; i++) {
		if (body.t[i].sym == s_hashhash && i > 0 && i + 1 < body.n &&
		    body.t[i].file != PP.synfile + 1) {
			struct sbuf b = {0};
			struct ptok p;
			sb_printf(&b, "%.*s%.*s", (int)body.t[i - 1].len,
			    ptext(&body.t[i - 1]), (int)body.t[i + 1].len,
			    ptext(&body.t[i + 1]));
			p = synth(b.p ? b.p : "", b.n, &body.t[i - 1]);
			free(b.p);
			body.t[i - 1] = p;
			memmove(&body.t[i], &body.t[i + 2], (body.n - i - 2) *
			    sizeof *body.t);
			body.n -= 2;
			i--;
		}
	}
	/* drop placemarkers */
	{
		uint32_t j = 0;
		for (i = 0; i < body.n; i++)
			if (body.t[i].len || body.t[i].kind != CK_PUNCT ||
			    body.t[i].file != PP.synfile)
				body.t[j++] = body.t[i];
		body.n = j;
	}
	for (i = 0; i < body.n; i++) {
		body.t[i].grp = grp;
		body.t[i].o0 = o0;
		body.t[i].o1 = o1;
		body.t[i].fl |= CTF_MACRO;
		if (nt.file != 0 && !nt.grp)
			body.t[i].file = body.t[i].file;	/* not main file */
	}
	if (body.n)
		body.t[0].fl = (uint8_t)((body.t[0].fl & ~CTF_SPACE) |
		    (nt.fl & CTF_SPACE));
	for (i = 0; i < 33; i++)
		free(args[i].t);
	res = &body;
	rd_push(r, res->t, res->n, m, 1, 0);
}

/* macro-expand a token list in isolation (arguments, #if lines) */
static void
expand_into(struct ptbuf *in, struct ptbuf *out, uint32_t grp, uint32_t o0,
    uint32_t o1)
{
	struct reader r;
	struct ptok *t;

	r.nf = 0;
	rd_push(&r, in->t, in->n, NULL, 0, 1);
	for (;;) {
		struct macro *m;
		t = rd_peek(&r);
		if (t == NULL) {
			/* the base frame is a "file": pop it at its end */
			if (r.nf == 1)
				break;
			continue;
		}
		r.f[r.nf - 1].i++;
		if (t->kind == CK_ID && (m = mac_find(t->sym)) != NULL &&
		    !m->active) {
			struct ptok nt = *t, *q;
			if (m->fn) {
				q = rd_peek(&r);
				if (q == NULL || q->sym != s_lp) {
					pt_push(out, &nt);
					continue;
				}
			}
			expand_macro(&r, m, nt);
			if (PP.err)
				break;
			continue;
		}
		pt_push(out, t);
	}
	while (r.nf > 0)
		rd_pop(&r);
}

/* ---- #if expressions ---- */

struct cexp {
	struct ptok *t;
	uint32_t n, i;
	int err;
};

static int64_t cx_cond(struct cexp *c);

static int
cx_is(struct cexp *c, const char *s)
{
	if (c->i < c->n && c->t[c->i].len == strlen(s) &&
	    !strncmp(ptext(&c->t[c->i]), s, strlen(s))) {
		c->i++;
		return 1;
	}
	return 0;
}

static int64_t
cx_prim(struct cexp *c)
{
	struct ptok *t;
	int64_t v;

	if (c->i >= c->n) {
		c->err = 1;
		return 0;
	}
	if (cx_is(c, "(")) {
		v = cx_cond(c);
		if (!cx_is(c, ")"))
			c->err = 1;
		return v;
	}
	if (cx_is(c, "!"))
		return !cx_prim(c);
	if (cx_is(c, "-"))
		return -cx_prim(c);
	if (cx_is(c, "+"))
		return cx_prim(c);
	if (cx_is(c, "~"))
		return ~cx_prim(c);
	t = &c->t[c->i++];
	if (t->kind == CK_NUM) {
		char buf[64];
		snprintf(buf, sizeof buf, "%.*s", (int)t->len, ptext(t));
		return (int64_t)strtoull(buf, NULL, 0);
	}
	if (t->kind == CK_CHAR) {
		const char *s = ptext(t);
		while (*s != '\'')
			s++;
		s++;
		if (*s == '\\') {
			s++;
			switch (*s) {
			case 'n': return 10;
			case 't': return 9;
			case '0': return 0;
			case '\\': return '\\';
			case '\'': return '\'';
			default: return *s;
			}
		}
		return (unsigned char)*s;
	}
	if (t->kind == CK_ID)
		return 0;	/* an identifier left after expansion */
	c->err = 1;
	return 0;
}

static int64_t
cx_bin(struct cexp *c, int lev)
{
	static const char *const ops[][5] = {
		{"||"}, {"&&"}, {"|"}, {"^"}, {"&"}, {"==", "!="},
		{"<=", ">=", "<", ">"}, {"<<", ">>"}, {"+", "-"},
		{"*", "/", "%"}
	};
	int64_t a, b;
	int k;

	if (lev == 10)
		return cx_prim(c);
	a = cx_bin(c, lev + 1);
	for (;;) {
		const char *op = NULL;
		for (k = 0; k < 5 && ops[lev][k]; k++)
			if (cx_is(c, ops[lev][k])) {
				op = ops[lev][k];
				break;
			}
		if (op == NULL)
			return a;
		b = cx_bin(c, lev + 1);
		if (!strcmp(op, "||")) a = a || b;
		else if (!strcmp(op, "&&")) a = a && b;
		else if (!strcmp(op, "|")) a |= b;
		else if (!strcmp(op, "^")) a ^= b;
		else if (!strcmp(op, "&")) a &= b;
		else if (!strcmp(op, "==")) a = a == b;
		else if (!strcmp(op, "!=")) a = a != b;
		else if (!strcmp(op, "<=")) a = a <= b;
		else if (!strcmp(op, ">=")) a = a >= b;
		else if (!strcmp(op, "<")) a = a < b;
		else if (!strcmp(op, ">")) a = a > b;
		else if (!strcmp(op, "<<")) a = (int64_t)((uint64_t)a << (b & 63));
		else if (!strcmp(op, ">>")) a >>= (b & 63);
		else if (!strcmp(op, "+")) a = (int64_t)((uint64_t)a + (uint64_t)b);
		else if (!strcmp(op, "-")) a = (int64_t)((uint64_t)a - (uint64_t)b);
		else if (!strcmp(op, "*")) a = (int64_t)((uint64_t)a * (uint64_t)b);
		else if (b == 0) c->err = 1;
		else if (!strcmp(op, "/")) a = (b == -1) ? -a : a / b;
		else a = (b == -1) ? 0 : a % b;
	}
}

static int64_t
cx_cond(struct cexp *c)
{
	int64_t a = cx_bin(c, 0), b, d;

	if (!cx_is(c, "?"))
		return a;
	b = cx_cond(c);
	if (!cx_is(c, ":"))
		c->err = 1;
	d = cx_cond(c);
	return a ? b : d;
}

/* evaluate the directive line t[0..n) */
static int
pp_ifexpr(struct ptok *t, uint32_t n, const struct ptok *at)
{
	struct ptbuf in = {0}, out = {0};
	struct cexp c;
	uint32_t i;
	int64_t v;

	for (i = 0; i < n; i++) {
		if (t[i].sym == s_defined) {
			uint32_t nm;
			int par = 0;
			struct ptok one;
			if (i + 1 < n && t[i + 1].sym == s_lp)
				par = 1, i++;
			if (i + 1 >= n) {
				pp_error(at, "#if: defined needs a name%s", "");
				break;
			}
			nm = t[++i].sym;
			if (par && i + 1 < n && t[i + 1].sym == s_rp)
				i++;
			one = synth(mac_find(nm) ? "1" : "0", 1, &t[i]);
			pt_push(&in, &one);
			continue;
		}
		pt_push(&in, &t[i]);
	}
	expand_into(&in, &out, 0, 0, 0);
	c.t = out.t;
	c.n = out.n;
	c.i = 0;
	c.err = 0;
	v = cx_cond(&c);
	if (c.err || c.i != c.n)
		pp_error(at, "#if: cannot evaluate the expression%s", "");
	free(in.t);
	free(out.t);
	return v != 0;
}

/* ---- the built-in preludes for system headers ---- */

static const struct {
	const char *name;
	const char *text;
} preludes[] = {
	{"stdint.h",
	    "typedef signed char int8_t; typedef short int16_t; "
	    "typedef int int32_t; typedef long int64_t; "
	    "typedef unsigned char uint8_t; typedef unsigned short uint16_t; "
	    "typedef unsigned int uint32_t; typedef unsigned long uint64_t; "
	    "typedef signed char int_least8_t; typedef short int_least16_t; "
	    "typedef int int_least32_t; typedef long int_least64_t; "
	    "typedef unsigned char uint_least8_t; "
	    "typedef unsigned short uint_least16_t; "
	    "typedef unsigned int uint_least32_t; "
	    "typedef unsigned long uint_least64_t; "
	    "typedef signed char int_fast8_t; typedef long int_fast16_t; "
	    "typedef long int_fast32_t; typedef long int_fast64_t; "
	    "typedef unsigned char uint_fast8_t; "
	    "typedef unsigned long uint_fast16_t; "
	    "typedef unsigned long uint_fast32_t; "
	    "typedef unsigned long uint_fast64_t; "
	    "typedef long intptr_t; typedef unsigned long uintptr_t; "
	    "typedef long intmax_t; typedef unsigned long uintmax_t;\n"
	    "#define INT8_MIN (-128)\n#define INT8_MAX 127\n"
	    "#define UINT8_MAX 255\n#define INT16_MIN (-32767-1)\n"
	    "#define INT16_MAX 32767\n#define UINT16_MAX 65535\n"
	    "#define INT32_MIN (-2147483647-1)\n#define INT32_MAX 2147483647\n"
	    "#define UINT32_MAX 4294967295U\n"
	    "#define INT64_MIN (-9223372036854775807L-1)\n"
	    "#define INT64_MAX 9223372036854775807L\n"
	    "#define UINT64_MAX 18446744073709551615UL\n"
	    "#define SIZE_MAX 18446744073709551615UL\n"
	    "#define INTPTR_MAX 9223372036854775807L\n"
	    "#define UINTPTR_MAX 18446744073709551615UL\n"
	    "#define INT8_C(x) x\n#define INT16_C(x) x\n#define INT32_C(x) x\n"
	    "#define INT64_C(x) x ## L\n#define UINT8_C(x) x\n"
	    "#define UINT16_C(x) x\n#define UINT32_C(x) x ## U\n"
	    "#define UINT64_C(x) x ## UL\n"},
	{"inttypes.h", "#include <stdint.h>\n"},
	{"stddef.h",
	    "typedef unsigned long size_t; typedef long ptrdiff_t; "
	    "typedef int wchar_t;\n#define NULL ((void*)0)\n"
	    "#define offsetof(t, m) ((size_t)&((t*)0)->m)\n"},
	{"stdbool.h",
	    "#define bool _Bool\n#define true 1\n#define false 0\n"
	    "#define __bool_true_false_are_defined 1\n"},
	{"limits.h",
	    "#define CHAR_BIT 8\n#define SCHAR_MIN (-128)\n#define SCHAR_MAX 127\n"
	    "#define UCHAR_MAX 255\n#define CHAR_MIN (-128)\n#define CHAR_MAX 127\n"
	    "#define SHRT_MIN (-32768)\n#define SHRT_MAX 32767\n"
	    "#define USHRT_MAX 65535\n#define INT_MIN (-2147483647-1)\n"
	    "#define INT_MAX 2147483647\n#define UINT_MAX 4294967295U\n"
	    "#define LONG_MIN (-9223372036854775807L-1)\n"
	    "#define LONG_MAX 9223372036854775807L\n"
	    "#define ULONG_MAX 18446744073709551615UL\n"
	    "#define LLONG_MIN (-9223372036854775807LL-1)\n"
	    "#define LLONG_MAX 9223372036854775807LL\n"
	    "#define ULLONG_MAX 18446744073709551615ULL\n"},
	{"string.h",
	    "#include <stddef.h>\n"
	    "void *memcpy(void *restrict d, const void *restrict s, size_t n); "
	    "void *memmove(void *d, const void *s, size_t n); "
	    "void *memset(void *d, int c, size_t n); "
	    "int memcmp(const void *a, const void *b, size_t n); "
	    "size_t strlen(const char *s); "
	    "int strcmp(const char *a, const char *b); "
	    "char *strcpy(char *restrict d, const char *restrict s); "
	    "void *memchr(const void *s, int c, size_t n); "
	    "char *strchr(const char *s, int c);\n"},
	{"stdlib.h",
	    "#include <stddef.h>\n"
	    "void *malloc(size_t n); void *calloc(size_t n, size_t m); "
	    "void *realloc(void *p, size_t n); void free(void *p); "
	    "void abort(void); void exit(int s); int abs(int x); "
	    "long labs(long x); int rand(void);\n"},
	{"stdio.h",
	    "#include <stddef.h>\n"
	    "typedef struct __FILE FILE; extern FILE *stdout, *stderr; "
	    "int printf(const char *restrict f, ...); "
	    "int fprintf(FILE *restrict o, const char *restrict f, ...); "
	    "int puts(const char *s); int putchar(int c);\n"},
	{"math.h",
	    "double sqrt(double); float sqrtf(float); double fabs(double); "
	    "float fabsf(float); double floor(double); float floorf(float); "
	    "double ceil(double); float ceilf(float); double sin(double); "
	    "double cos(double); double exp(double); double log(double); "
	    "float sinf(float); float cosf(float); float expf(float); "
	    "float logf(float); double pow(double, double); "
	    "float powf(float, float); double fmin(double, double); "
	    "double fmax(double, double); float fminf(float, float); "
	    "float fmaxf(float, float); double fma(double, double, double); "
	    "float fmaf(float, float, float); double round(double); "
	    "float roundf(float); double trunc(double); "
	    "float truncf(float);\n"},
	{"assert.h", "#define assert(e) ((void)0)\n"},
	{"immintrin.h",
	    "typedef struct __cf_m128i { long __v[2]; } __m128i; "
	    "typedef struct __cf_m256i { long __v[4]; } __m256i; "
	    "typedef struct __cf_m512i { long __v[8]; } __m512i; "
	    "typedef struct __cf_m128 { float __v[4]; } __m128; "
	    "typedef struct __cf_m256 { float __v[8]; } __m256; "
	    "typedef struct __cf_m512 { float __v[16]; } __m512; "
	    "typedef struct __cf_m128d { double __v[2]; } __m128d; "
	    "typedef struct __cf_m256d { double __v[4]; } __m256d; "
	    "typedef struct __cf_m512d { double __v[8]; } __m512d;\n"},
	{"x86intrin.h", "#include <immintrin.h>\n"},
	{"emmintrin.h", "#include <immintrin.h>\n"},
	{"smmintrin.h", "#include <immintrin.h>\n"},
	{"nmmintrin.h", "#include <immintrin.h>\n"},
	{"tmmintrin.h", "#include <immintrin.h>\n"},
	{"xmmintrin.h", "#include <immintrin.h>\n"},
	{"float.h",
	    "#define FLT_MAX 3.40282347e+38F\n#define DBL_MAX 1.7976931348623157e+308\n"
	    "#define FLT_EPSILON 1.19209290e-7F\n"
	    "#define DBL_EPSILON 2.2204460492503131e-16\n"},
	{NULL, NULL}
};

static void process(struct reader *r, struct ptbuf *out);

static void
include_text(struct reader *r, const char *name, const char *text,
    int builtin, struct ptbuf *out)
{
	struct ptbuf raw = {0};
	int f = add_file(name, strdup(text));
	uint32_t i;
	struct reader sub;

	lex_file(f, &raw);
	for (i = 0; i < raw.n; i++)
		if (builtin)
			raw.t[i].fl |= CTF_BUILTIN;
	if (++PP.depth > 64) {
		pp_error(NULL, "#include nested too deeply (%s)", name);
		free(raw.t);
		PP.depth--;
		return;
	}
	sub.nf = 0;
	rd_push(&sub, raw.t, raw.n, NULL, 1, 1);
	process(&sub, out);
	while (sub.nf > 0)
		rd_pop(&sub);
	PP.depth--;
	(void)r;
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

static void
do_include(struct reader *r, struct ptok *t, uint32_t n, struct ptok *at,
    struct ptbuf *out)
{
	char name[256];
	const char *s;
	int sys, k;

	if (n == 0) {
		pp_error(at, "#include needs a file%s", "");
		return;
	}
	s = ptext(&t[0]);
	if (t[0].kind == CK_STR) {
		snprintf(name, sizeof name, "%.*s", (int)t[0].len - 2, s + 1);
		sys = 0;
	} else if (t[0].sym == sym_cstr("<")) {
		/* <a/b.h>: the text up to '>' */
		const char *e = strchr(s, '>');
		if (e == NULL || (size_t)(e - s) > sizeof name) {
			pp_error(at, "#include <...>: no '>'%s", "");
			return;
		}
		snprintf(name, sizeof name, "%.*s", (int)(e - s - 1), s + 1);
		sys = 1;
	} else {
		pp_error(at, "#include: a \"file\" or <header>%s", "");
		return;
	}
	if (sys) {
		if (at->file == 0 || !(at->fl & CTF_BUILTIN))
			if (PP.depth == 0 || at->file == 0)
				sb_printf(&PP.u->includes, "#include <%s>\n", name);
		for (k = 0; preludes[k].name; k++)
			if (!strcmp(preludes[k].name, name)) {
				if (PP.known_hdr & (1 << k))
					return;
				PP.known_hdr |= 1 << k;
				include_text(r, name, preludes[k].text, 1, out);
				return;
			}
		return;	/* unknown: its declarations are not known */
	}
	{
		char path[1024];
		const char *cur = PP.u->fname[at->file], *sl = strrchr(cur, '/');
		char *text;
		if (name[0] == '/' || sl == NULL)
			snprintf(path, sizeof path, "%s", name);
		else
			snprintf(path, sizeof path, "%.*s/%s", (int)(sl - cur), cur,
			    name);
		text = read_file(path);
		if (text == NULL) {
			pp_error(at, "#include \"%s\": cannot read it", name);
			return;
		}
		include_text(r, path, text, 0, out);
		free(text);
	}
}

static void
do_define(struct ptok *t, uint32_t n, struct ptok *at)
{
	struct macro *m;
	uint32_t i = 1;

	if (n == 0 || t[0].kind != CK_ID) {
		pp_error(at, "#define needs a name%s", "");
		return;
	}
	m = mac_new(t[0].sym);
	if (n > 1 && t[1].sym == s_lp && !(t[1].fl & CTF_SPACE)) {
		m->fn = 1;
		i = 2;
		while (i < n && t[i].sym != s_rp) {
			if (t[i].sym == s_ellipsis) {
				m->variadic = 1;
				if (m->nparam < 32)
					m->param[m->nparam++] = s_va;
			} else if (t[i].kind == CK_ID && m->nparam < 32)
				m->param[m->nparam++] = t[i].sym;
			i++;
			if (i < n && t[i].sym == s_comma)
				i++;
		}
		i++;
	}
	for (; i < n; i++)
		pt_push(&m->body, &t[i]);
}

/* the main loop over a reader whose bottom frame is a file */
static void
process(struct reader *r, struct ptbuf *out)
{
	/* conditional stack: live, taken (some branch taken), else seen */
	uint8_t live[256], taken[256];
	int nc = 0;
	struct ptok *t;

	for (;;) {
		struct macro *m;
		int on = 1, k;
		if (PP.err)
			return;
		for (k = 0; k < nc; k++)
			on &= live[k];
		t = rd_peek(r);
		if (t == NULL) {
			if (r->nf > 1) {	/* a file frame inside: never */
				rd_pop(r);
				continue;
			}
			break;
		}
		if (rd_infile(r) && t->sym == s_hash && (t->fl & CTF_BOL)) {
			/* a directive: the rest of the line */
			struct frame *f = &r->f[r->nf - 1];
			uint32_t s0 = f->i + 1, e = s0;
			struct ptok at = *t;
			uint32_t d;
			while (e < f->n && !(f->t[e].fl & CTF_BOL))
				e++;
			f->i = e;
			if (s0 == e)
				continue;	/* the null directive */
			d = f->t[s0].sym;
			if (d == s_if || d == s_ifdef || d == s_ifndef) {
				int v = 0;
				if (nc == 256) {
					pp_error(&at, "#if nested too deeply%s", "");
					return;
				}
				if (on) {
					if (d == s_if)
						v = pp_ifexpr(&f->t[s0 + 1], e - s0 - 1, &at);
					else if (e > s0 + 1)
						v = (mac_find(f->t[s0 + 1].sym) != NULL) ==
						    (d == s_ifdef);
				}
				live[nc] = (uint8_t)v;
				taken[nc] = (uint8_t)(v || !on);
				nc++;
			} else if (d == s_elif) {
				if (nc == 0) {
					pp_error(&at, "#elif without #if%s", "");
					return;
				}
				nc--;
				on = 1;
				for (k = 0; k < nc; k++)
					on &= live[k];
				if (taken[nc] || !on)
					live[nc] = 0;
				else {
					live[nc] = (uint8_t)pp_ifexpr(&f->t[s0 + 1],
					    e - s0 - 1, &at);
					taken[nc] = live[nc];
				}
				nc++;
			} else if (d == s_else) {
				if (nc == 0) {
					pp_error(&at, "#else without #if%s", "");
					return;
				}
				live[nc - 1] = !taken[nc - 1];
				taken[nc - 1] = 1;
			} else if (d == s_endif) {
				if (nc == 0) {
					pp_error(&at, "#endif without #if%s", "");
					return;
				}
				nc--;
			} else if (!on)
				continue;
			else if (d == s_define)
				do_define(&f->t[s0 + 1], e - s0 - 1, &at);
			else if (d == s_undef) {
				if (e > s0 + 1 && (m = mac_find(f->t[s0 + 1].sym)))
					m->undef = 1;
			} else if (d == s_include)
				do_include(r, &f->t[s0 + 1], e - s0 - 1, &at, out);
			else if (d == s_error)
				pp_error(&at, "#error%s", "");
			else if (d == s_pragma || d == s_line || d == s_warning ||
			    f->t[s0].kind == CK_NUM)
				;
			else
				pp_error(&at, "unknown directive #%s",
				    sym_str(d));
			continue;
		}
		r->f[r->nf - 1].i++;
		if (!on)
			continue;
		if (t->kind == CK_ID && (m = mac_find(t->sym)) != NULL &&
		    !m->active) {
			struct ptok nt = *t, *q;
			if (m->fn) {
				q = rd_peek(r);
				if (q == NULL || q->sym != s_lp) {
					pt_push(out, &nt);
					continue;
				}
			}
			expand_macro(r, m, nt);
			continue;
		}
		pt_push(out, t);
	}
	if (nc && !PP.err)
		pp_error(NULL, "#if without #endif%s", "");
}

static void
predefine(const char *name, const char *val)
{
	struct macro *m = mac_new(sym_cstr(name));
	struct ptok like;
	struct ptok t;

	memset(&like, 0, sizeof like);
	like.fl = CTF_BUILTIN;
	t = synth(val, strlen(val), &like);
	t.fl = CTF_BUILTIN;
	pt_push(&m->body, &t);
}

int
cf_preprocess(struct cf_unit *u, const char *path, const char *text,
    const struct cf_target *tg)
{
	struct ptbuf raw = {0}, out = {0};
	struct reader r;
	uint32_t i;
	int k;
	static const char *const pre[][2] = {
		{"__STDC__", "1"}, {"__STDC_VERSION__", "199901L"},
		{"__STDC_HOSTED__", "1"}, {"__x86_64__", "1"}, {"__x86_64", "1"},
		{"__amd64__", "1"}, {"__LP64__", "1"}, {"_LP64", "1"},
		{"__CHAR_BIT__", "8"}, {"__SIZEOF_INT__", "4"},
		{"__SIZEOF_LONG__", "8"}, {"__SIZEOF_LONG_LONG__", "8"},
		{"__SIZEOF_SHORT__", "2"}, {"__SIZEOF_POINTER__", "8"},
		{"__SIZEOF_SIZE_T__", "8"}, {"__ORDER_LITTLE_ENDIAN__", "1234"},
		{"__ORDER_BIG_ENDIAN__", "4321"}, {"__BYTE_ORDER__", "1234"},
		{"__ucas_cfront__", "1"}
	};

	memset(&PP, 0, sizeof PP);
	PP.u = u;
	pp_syms();
	u->path = path;
	u->text = strdup(text);
	add_file(path, u->text);
	PP.synfile = (uint16_t)add_file("<macro>", strdup(""));
	sb_puts(&PP.syntext, "");
	u->ftext[PP.synfile] = PP.syntext.p;
	for (i = 0; i < sizeof pre / sizeof pre[0]; i++)
		predefine(pre[i][0], pre[i][1]);
	if (tg)
		for (k = 0; k < CF_NFEAT; k++)
			if (tg->feat & CF_F(k))
				predefine(cf_feat_macro(k), "1");
	new_grp();
	lex_file(0, &raw);
	r.nf = 0;
	rd_push(&r, raw.t, raw.n, NULL, 1, 1);
	process(&r, &out);
	while (r.nf > 0)
		rd_pop(&r);
	if (PP.err) {
		sb_puts(&u->err, PP.err);
		free(out.t);
		return 0;
	}
	/* to SoA; a token from a bad group, or of another file inside a
	 * main-file invocation, keeps grp but gets file 0xffff origin */
	{
		struct cf_toks *tk = &u->tk;
		tk->n = 0;
		tk->cap = out.n + 1;
		tk->kind = xmalloc(tk->cap);
		tk->fl = xmalloc(tk->cap);
		tk->file = xmalloc(tk->cap * sizeof *tk->file);
		tk->off = xmalloc(tk->cap * sizeof *tk->off);
		tk->len = xmalloc(tk->cap * sizeof *tk->len);
		tk->line = xmalloc(tk->cap * sizeof *tk->line);
		tk->col = xmalloc(tk->cap * sizeof *tk->col);
		tk->sym = xmalloc(tk->cap * sizeof *tk->sym);
		u->tk_o0 = xmalloc(tk->cap * sizeof *u->tk_o0);
		u->tk_o1 = xmalloc(tk->cap * sizeof *u->tk_o1);
		u->tk_grp = xmalloc(tk->cap * sizeof *u->tk_grp);
		for (i = 0; i <= out.n; i++) {
			struct ptok e, *p;
			if (i == out.n) {
				memset(&e, 0, sizeof e);
				e.kind = CK_EOF;
				e.file = 0;
				e.sym = sym_cstr("<eof>");
				if (out.n) {
					e.line = out.t[out.n - 1].line;
					e.col = out.t[out.n - 1].col;
					e.file = out.t[out.n - 1].file;
				}
				p = &e;
			} else
				p = &out.t[i];
			tk->kind[i] = p->kind;
			tk->fl[i] = p->fl;
			tk->file[i] = p->file;
			tk->off[i] = p->off;
			tk->len[i] = p->len;
			tk->line[i] = p->line;
			tk->col[i] = p->col;
			tk->sym[i] = p->sym;
			u->tk_o0[i] = p->o0;
			u->tk_o1[i] = p->o1;
			u->tk_grp[i] = p->grp && PP.grpbad[p->grp] ? 0xffffffffu :
			    p->grp;
			tk->n++;
		}
	}
	u->synfile = PP.synfile;
	free(out.t);
	return 1;
}

const char *
cf_tok_text(const struct cf_unit *u, uint32_t i, size_t *n)
{
	*n = u->tk.len[i];
	return u->ftext[u->tk.file[i]] + u->tk.off[i];
}

int
cf_is_macro(const struct cf_unit *u, uint32_t sym)
{
	int i;

	for (i = 0; i < u->nmacro; i++)
		if (u->macros[i] == sym)
			return 1;
	return 0;
}

/*
 * The main-file text of tokens t0..t1: both ends must be main-file
 * tokens or the first/last token of a main-file macro invocation, and
 * no directive may sit in between.
 */
int
cf_tspan(const struct cf_unit *u, uint32_t t0, uint32_t t1, uint32_t *from,
    uint32_t *to)
{
	uint32_t g0 = u->tk_grp[t0], g1 = u->tk_grp[t1], i;
	const char *s;

	if (t1 < t0 || t1 >= u->tk.n)
		return 0;
	if (g0 == 0xffffffffu || g1 == 0xffffffffu)
		return 0;
	if (g0 == 0 && u->tk.file[t0] != 0)
		return 0;
	if (g1 == 0 && u->tk.file[t1] != 0)
		return 0;
	if (g0 && t0 > 0 && u->tk_grp[t0 - 1] == g0)
		return 0;
	if (g1 && t1 + 1 < u->tk.n && u->tk_grp[t1 + 1] == g1)
		return 0;
	for (i = t0; i <= t1; i++)
		if (u->tk_grp[i] == 0xffffffffu ||
		    (u->tk_grp[i] == 0 && u->tk.file[i] != 0))
			return 0;
	*from = u->tk_o0[t0];
	*to = u->tk_o1[t1];
	if (*to <= *from)
		return 0;
	/* no directive inside */
	s = u->ftext[0];
	for (i = *from; i < *to; i++)
		if (s[i] == '\n') {
			uint32_t j = i + 1;
			while (j < *to && (s[j] == ' ' || s[j] == '\t'))
				j++;
			if (j < *to && s[j] == '#')
				return 0;
		}
	return 1;
}
