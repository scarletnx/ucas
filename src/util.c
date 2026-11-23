/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "ucas.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

/*
 * OOM is fatal everywhere in ucas: there is no recovery path that can run
 * after an allocation returns NULL.  xmalloc/xcalloc/xrealloc only protect
 * the sites that call them, but the parser and symbol code also call raw
 * malloc/calloc/realloc/strdup and dereference the result unchecked.  Route
 * every allocation through a wrapper so a failed one always ends in the
 * same clean exit instead of a random SIGSEGV.
 *
 * dlsym(RTLD_NEXT) reaches the real allocator (or an LD_PRELOAD fault-
 * injection shim), so injected failures are still observed here.
 *
 * On glibc, dlsym() itself may re-enter one of the wrappers below while it
 * resolves a symbol; when that happens real_fn() falls back to glibc's
 * private, non-interposed allocator.  Those symbols are GLIBC_PRIVATE and do
 * not exist on musl or FreeBSD libc, so they are referenced only there.
 * Everywhere else a plain dlsym(RTLD_NEXT, ...) is enough: those libcs'
 * dlsym() does not allocate and therefore cannot recurse into a wrapper.
 */
/* a sanitizer brings its own allocator: no interposing under it */
#if defined(__SANITIZE_ADDRESS__)
#define UCAS_NO_INTERPOSE 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer)
#define UCAS_NO_INTERPOSE 1
#endif
#endif

#ifndef UCAS_NO_INTERPOSE
#if defined(__GLIBC__)
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
#endif

static void
oom(void)
{
	fputs("ucas: out of memory\n", stderr);
	exit(2);
}

static void *
real_fn(const char *name, void *fallback)
{
	static int busy;
	void *fn;

	if (busy)
		return fallback;
	busy = 1;
	fn = dlsym(RTLD_NEXT, name);
	busy = 0;
	return fn ? fn : fallback;
}

void *
malloc(size_t n)
{
	static void *(*fn)(size_t);
	void *p;

	if (fn == NULL) {
#if defined(__GLIBC__)
		fn = real_fn("malloc", __libc_malloc);
#else
		fn = dlsym(RTLD_NEXT, "malloc");
#endif
		if (fn == NULL)
			oom();		/* no allocator could be resolved */
	}
	p = fn(n ? n : 1);  /* malloc(0) may legitimately return NULL on BSDs */
	if (p == NULL)
		oom();
	return p;
}

void *
calloc(size_t n, size_t sz)
{
	static void *(*fn)(size_t, size_t);
	void *p;

	if (fn == NULL) {
#if defined(__GLIBC__)
		fn = real_fn("calloc", __libc_calloc);
#else
		fn = dlsym(RTLD_NEXT, "calloc");
#endif
		if (fn == NULL)
			oom();
	}
	p = fn(n ? n : 1, sz ? sz : 1);  /* see the malloc(0) note above */
	if (p == NULL)
		oom();
	return p;
}

void *
realloc(void *p, size_t n)
{
	static void *(*fn)(void *, size_t);
	void *q;

	if (fn == NULL) {
#if defined(__GLIBC__)
		fn = real_fn("realloc", __libc_realloc);
#else
		fn = dlsym(RTLD_NEXT, "realloc");
#endif
		if (fn == NULL)
			oom();
	}
	q = fn(p, n);
	if (q == NULL && n != 0)    /* realloc(p, 0) frees and returns NULL */
		oom();
	return q;
}

char *
strdup(const char *s)
{
	static char *(*fn)(const char *);
	char *p;

	if (fn == NULL) {
		fn = real_fn("strdup", NULL);
		if (fn == NULL) {
			size_t n = strlen(s) + 1;
			p = malloc(n);
			if (p)
				memcpy(p, s, n);
			return p;
		}
	}
	p = fn(s);
	if (p == NULL)
		oom();
	return p;
}
#endif /* UCAS_NO_INTERPOSE */

void *
xmalloc(size_t n)
{
	return malloc(n ? n : 1);
}

void *
xcalloc(size_t n, size_t sz)
{
	return calloc(n ? n : 1, sz ? sz : 1);
}

void *
xrealloc(void *p, size_t n)
{
	return realloc(p, n ? n : 1);
}

uint32_t
newcap(uint32_t cap, uint32_t need)
{
	if (cap >= need)
		return cap;
	if (cap == 0)
		cap = 64;
	while (cap < need)
		cap *= 2;
	return cap;
}

/* ---- string buffers ---- */

static void
sb_reserve(struct sbuf *b, size_t extra)
{
	if (b->n + extra + 1 <= b->cap)
		return;
	if (b->cap == 0)
		b->cap = 64;
	while (b->n + extra + 1 > b->cap)
		b->cap *= 2;
	b->p = xrealloc(b->p, b->cap);
}

void
sb_putc(struct sbuf *b, int c)
{
	sb_reserve(b, 1);
	b->p[b->n++] = (char)c;
	b->p[b->n] = 0;
}

void
sb_puts(struct sbuf *b, const char *s)
{
	size_t n = strlen(s);

	sb_reserve(b, n);
	memcpy(b->p + b->n, s, n + 1);
	b->n += n;
}

void
sb_printf(struct sbuf *b, const char *fmt, ...)
{
	va_list ap;
	int n;

	va_start(ap, fmt);
	n = vsnprintf(NULL, 0, fmt, ap);
	va_end(ap);
	if (n < 0)
		return;
	sb_reserve(b, (size_t)n);
	va_start(ap, fmt);
	vsnprintf(b->p + b->n, (size_t)n + 1, fmt, ap);
	va_end(ap);
	b->n += (size_t)n;
}

void
sb_clear(struct sbuf *b)
{
	b->n = 0;
	if (b->p)
		b->p[0] = 0;
}

const char *
sb_str(struct sbuf *b)
{
	if (b->p == NULL)
		sb_reserve(b, 0), b->p[0] = 0;
	return b->p;
}

/* ---- symbols ---- */

static char **syms;
static uint32_t nsyms, symcap;
static uint32_t *symht;
static uint32_t symhtcap;

static uint32_t
fnv(const char *s, size_t n)
{
	uint32_t h = 2166136261u;
	size_t i;

	for (i = 0; i < n; i++)
		h = (h ^ (unsigned char)s[i]) * 16777619u;
	return h;
}

static void
sym_rehash(void)
{
	uint32_t i, j;

	free(symht);
	symhtcap = symhtcap ? symhtcap * 2 : 256;
	symht = xcalloc(symhtcap, sizeof *symht);
	for (i = 0; i < nsyms; i++) {
		j = fnv(syms[i], strlen(syms[i])) & (symhtcap - 1);
		while (symht[j])
			j = (j + 1) & (symhtcap - 1);
		symht[j] = i + 1;
	}
}

uint32_t
sym_intern(const char *s, size_t n)
{
	uint32_t j;

	if ((nsyms + 1) * 2 > symhtcap)
		sym_rehash();
	j = fnv(s, n) & (symhtcap - 1);
	while (symht[j]) {
		const char *t = syms[symht[j] - 1];
		if (strlen(t) == n && memcmp(t, s, n) == 0)
			return symht[j] - 1;
		j = (j + 1) & (symhtcap - 1);
	}
	if (nsyms == symcap) {
		symcap = newcap(symcap, nsyms + 1);
		syms = xrealloc(syms, symcap * sizeof *syms);
	}
	syms[nsyms] = xmalloc(n + 1);
	memcpy(syms[nsyms], s, n);
	syms[nsyms][n] = 0;
	symht[j] = ++nsyms;
	return nsyms - 1;
}

uint32_t
sym_cstr(const char *s)
{
	return sym_intern(s, strlen(s));
}

const char *
sym_str(uint32_t id)
{
	return id < nsyms ? syms[id] : "?";
}

/* ---- diagnostics ---- */

jmp_buf fail_jmp;
uint32_t diag_tok = NOTOK;
struct toks *diag_toks;
const char *diag_file = "<input>";
static struct sbuf notes;

static void
location(struct sbuf *b)
{
	const char *src, *ls, *p;
	uint32_t off, line = 1, col, i;

	if (diag_toks == NULL || diag_tok == NOTOK || diag_tok >= diag_toks->n) {
		sb_printf(b, "%s: ", diag_file);
		return;
	}
	src = diag_toks->src;
	off = diag_toks->off[diag_tok];
	ls = src;
	for (p = src; p < src + off; p++)
		if (*p == '\n')
			line++, ls = p + 1;
	col = (uint32_t)(src + off - ls) + 1;
	sb_printf(b, "%s:%u:%u: ", diag_file, line, col);
	(void)i;
}

/* the line and column of the token in error (0 when there is none) */
static void
where(uint32_t *line, uint32_t *col)
{
	const char *src, *ls, *p;
	uint32_t off;

	if (diag_toks == NULL || diag_tok == NOTOK || diag_tok >= diag_toks->n)
		return;
	src = diag_toks->src;
	off = diag_toks->off[diag_tok];
	ls = src;
	*line = 1;
	for (p = src; p < src + off; p++)
		if (*p == '\n')
			++*line, ls = p + 1;
	*col = (uint32_t)(src + off - ls) + 1;
}

static void
caret(struct sbuf *b)
{
	const char *src, *ls, *le;
	uint32_t off, i;

	if (diag_toks == NULL || diag_tok == NOTOK || diag_tok >= diag_toks->n)
		return;
	src = diag_toks->src;
	off = diag_toks->off[diag_tok];
	ls = src + off;
	while (ls > src && ls[-1] != '\n')
		ls--;
	le = src + off;
	while (*le && *le != '\n')
		le++;
	sb_puts(b, "    ");
	for (i = 0; ls + i < le; i++)
		sb_putc(b, ls[i]);
	sb_puts(b, "\n    ");
	for (i = 0; ls + i < src + off; i++)
		sb_putc(b, ls[i] == '\t' ? '\t' : ' ');
	sb_puts(b, "^\n");
}

int fail_mute;   /* > 0: fail() says nothing (a trial parse) */

void
fail(const char *fmt, ...)
{
	struct sbuf b = {0};
	va_list ap;
	char tmp[512];

	if (fail_mute) {
		notes_drop();
		longjmp(fail_jmp, 1);
	}
	location(&b);
	va_start(ap, fmt);
	vsnprintf(tmp, sizeof tmp, fmt, ap);
	va_end(ap);
	if (rep_on) {
		uint32_t line = 0, col = 0;
		where(&line, &col);
		rep_fail(tmp, line, col);
	}
	sb_printf(&b, "error: %s\n", tmp);
	caret(&b);
	fputs(sb_str(&b), stdout);
	free(b.p);
	notes_drop();
	longjmp(fail_jmp, 1);
}

/*
 * End the statement without an error message: a deferred question has
 * been printed and the run stops until it is answered.
 */
void
fail_quiet(void)
{
	notes_flush();
	longjmp(fail_jmp, 1);
}

int note_mute;   /* > 0: notes are dropped (trial rewrites) */

void
note(const char *fmt, ...)
{
	va_list ap, ap2;
	char *tmp;
	int len;
	struct sbuf line = {0}, t = {0};

	if (note_mute > 0)
		return;
	va_start(ap, fmt);
	va_copy(ap2, ap);
	len = vsnprintf(NULL, 0, fmt, ap);
	va_end(ap);
	tmp = xmalloc((size_t)(len < 0 ? 0 : len) + 1);
	vsnprintf(tmp, (size_t)len + 1, fmt, ap2);
	va_end(ap2);
	/* a note is one line: a long expression in it is cut */
	sb_puts(&t, tmp);
	cap_text(&t, 0, NOTE_CAP);
	if (rep_on)
		rep_note_hook(sb_str(&t));	/* a guess: an ambiguity in JSON mode */
	sb_printf(&line, "  note: %s\n", sb_str(&t));
	if (notes.p == NULL || strstr(notes.p, line.p) == NULL)
		sb_puts(&notes, line.p);
	free(line.p);
	free(t.p);
	free(tmp);
}

void
notes_flush(void)
{
	if (notes.n)
		fputs(notes.p, stdout);
	sb_clear(&notes);
}

void
notes_drop(void)
{
	sb_clear(&notes);
}

int
ilog2u(uint64_t v)
{
	int r = -1;

	while (v) {
		v >>= 1;
		r++;
	}
	return r;
}

int
bits_for(uint64_t v)
{
	return v ? ilog2u(v) + 1 : 1;
}
