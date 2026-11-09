#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "text.h"

static int
is_space(int c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
	    c == '\v' || c == '\f';
}

static int
is_lang(int c)
{
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
	    (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '+';
}

/*
 * "$stmt", "$ stmt", "$`stmt`", "$```[lang]\n...\n```" and "$reset".
 * Everything after the prefix goes to the engine unchanged; only the
 * code fence around it is removed.
 */
int
dc_command(const char *content, struct sbuf *code, int *echo)
{
	const char *p, *e, *q;
	int lines = 0, blank = 1;

	sb_clear(code);
	*echo = 0;
	if (content == NULL || content[0] != '$')
		return DC_IGNORE;
	p = content + 1;
	e = p + strlen(p);
	while (p < e && is_space((unsigned char)*p))
		p++;
	while (e > p && is_space((unsigned char)e[-1]))
		e--;
	if (e - p == 5 && memcmp(p, "reset", 5) == 0)
		return DC_RESET;
	if (e - p >= 3 && memcmp(p, "```", 3) == 0) {
		p += 3;
		/* a language tag alone on the opening line */
		for (q = p; q < e && is_lang((unsigned char)*q); q++)
			;
		if (q < e && *q == '\n')
			p = q + 1;
		else if (q < e && *q == '\r' && q + 1 < e && q[1] == '\n')
			p = q + 2;
		/* the closing fence is the last ``` */
		for (q = e - 3; q >= p; q--)
			if (memcmp(q, "```", 3) == 0) {
				e = q;
				break;
			}
	} else if (e - p >= 2 && *p == '`' && e[-1] == '`') {
		p++;
		e--;
	}
	while (p < e && is_space((unsigned char)*p))
		p++;
	while (e > p && is_space((unsigned char)e[-1]))
		e--;
	if (p == e)
		return DC_IGNORE;
	for (q = p; q < e; q++) {
		if (*q == '\r')
			continue;	/* the engine gets plain newlines */
		sb_putc(code, *q);
		if (*q == '\n') {
			blank = 1;
		} else if (!is_space((unsigned char)*q) && blank) {
			blank = 0;
			lines++;
		}
	}
	*echo = lines > 1;
	return DC_EVAL;
}

/* ``` inside the output would end the code block */
static void
escape(struct sbuf *b, const char *s, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++) {
		if (i + 2 < n && s[i] == '`' && s[i + 1] == '`' &&
		    s[i + 2] == '`') {
			sb_puts(b, "``\xe2\x80\x8b`");	/* zero-width space */
			i += 2;
		} else if (s[i] == 0) {
			sb_putc(b, '?');
		} else {
			sb_putc(b, s[i]);
		}
	}
	while (b->n > 0 && (b->p[b->n - 1] == '\n' || b->p[b->n - 1] == ' '))
		b->p[--b->n] = 0;
}

/*
 * The longest prefix of s[0..n) of at most max bytes, ending at a line
 * break when there is one, else at a UTF-8 character boundary.  *next
 * is where the rest starts (after the line break).
 */
static size_t
cut(const char *s, size_t n, size_t max, size_t *next)
{
	size_t i;

	if (n <= max) {
		*next = n;
		return n;
	}
	for (i = max; i > 0; i--)
		if (s[i] == '\n') {
			*next = i + 1;
			return i;
		}
	for (i = max; i > 0 && ((unsigned char)s[i] & 0xc0) == 0x80; i--)
		;
	if (i == 0)
		i = max;
	*next = i;
	return i;
}

static char *
block(const char *head, const char *s, size_t n)
{
	struct sbuf b = {0};

	if (head != NULL)
		sb_puts(&b, head);
	sb_puts(&b, "```\n");
	sb_printf(&b, "%.*s\n```", (int)n, s);
	return b.p;
}

void
dc_format(struct dc_reply *r, const char *head, const char *out, size_t n)
{
	struct sbuf body = {0}, h = {0};
	size_t hl, pos, len, next, lines;
	const char *s;
	int k;

	memset(r, 0, sizeof *r);
	if (head != NULL && *head)
		sb_printf(&h, "%s\n", head);
	hl = h.n;
	escape(&body, out ? out : "", out ? n : 0);
	s = sb_str(&body);
	if (body.n == 0) {
		r->n = 1;
		r->msg[0] = strdup(hl ? h.p : "(no output)");
		if (hl && r->msg[0][hl - 1] == '\n')
			r->msg[0][hl - 1] = 0;
		goto done;
	}
	if (hl + 8 + body.n <= DC_LIMIT) {
		r->n = 1;
		r->msg[0] = block(hl ? h.p : NULL, s, body.n);
		goto done;
	}
	/* split at line breaks into at most DC_MAXMSG messages */
	pos = 0;
	for (k = 0; k < DC_MAXMSG && pos < body.n; k++) {
		size_t room = DC_LIMIT - 8 - (k == 0 ? hl : 0);
		len = cut(s + pos, body.n - pos, room, &next);
		r->msg[k] = block(k == 0 && hl ? h.p : NULL, s + pos, len);
		pos += next;
	}
	r->n = k;
	if (pos >= body.n)
		goto done;
	/* too long: a preview and the whole output as a file */
	dc_reply_free(r);
	for (lines = 0, pos = 0; pos < n; pos++)
		if (out[pos] == '\n')
			lines++;
	if (n > 0 && out[n - 1] != '\n')
		lines++;
	sb_printf(&h, "%zu lines (%zu bytes); the full output is attached "
	    "as " DC_FILENAME "\n", lines, n);
	len = cut(s, body.n, DC_LIMIT - 8 - h.n, &next);
	r->n = 1;
	r->msg[0] = block(h.p, s, len);
	r->file = xmalloc(n + 1);
	memcpy(r->file, out, n);
	r->file[n] = 0;
	r->filelen = n;
done:
	free(body.p);
	free(h.p);
}

void
dc_reply_free(struct dc_reply *r)
{
	int i;

	for (i = 0; i < r->n; i++)
		free(r->msg[i]);
	free(r->file);
	memset(r, 0, sizeof *r);
}

const char *
dc_status(uint32_t status, uint32_t arg, const struct sup_limits *l,
    char *buf, size_t sz)
{
	const char *keep = "this channel's state is unchanged";
	int n = 0;

	buf[0] = 0;
	switch (status & SR_MASK) {
	case SR_OK:
		break;
	case SR_TIMEOUT:
		n = snprintf(buf, sz, "timed out after %d s; %s", l->wall_s,
		    keep);
		break;
	case SR_CPU:
		n = snprintf(buf, sz, "timed out (CPU limit %d s); %s",
		    l->cpu_s, keep);
		break;
	case SR_MEM:
		n = snprintf(buf, sz, "out of memory (limit %d MiB); %s",
		    l->mem_mb, keep);
		break;
	case SR_CRASH:
		if (arg >= 1000)
			n = snprintf(buf, sz, "evaluation exited with status "
			    "%u; %s", arg - 1000, keep);
		else
			n = snprintf(buf, sz, "evaluation crashed (signal %u); "
			    "%s", arg, keep);
		break;
	case SR_LOST:
		n = snprintf(buf, sz, "the evaluator for this channel was "
		    "restarted; its state was reset");
		break;
	case SR_RESET:
		n = snprintf(buf, sz, "state reset for this channel");
		break;
	case SR_BUSY:
		n = snprintf(buf, sz, "busy: too many requests waiting here; "
		    "try again in a moment");
		break;
	default:
		n = snprintf(buf, sz, "internal error");
		break;
	}
	if (status & SR_TRUNC)
		snprintf(buf + n, sz - (size_t)n, "%s(output cut at %zu KiB)",
		    n ? "; " : "", l->max_out >> 10);
	return buf[0] ? buf : NULL;
}
