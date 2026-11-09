/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * json.c - JSON values: a strict parser (RFC 8259: no comments, no
 * trailing commas, \u escapes with surrogate pairs) and a one-line writer.
 */
#include "ucas.h"
#include "json.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct jp {
	const char *s, *p;
	char *err;
	size_t errn;
	int depth;
	int bad;
};

static void
jerr(struct jp *j, const char *msg)
{
	if (!j->bad) {
		snprintf(j->err, j->errn, "%s", msg);
		j->bad = 1;
	}
}

static void
ws(struct jp *j)
{
	while (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')
		j->p++;
}

struct jv *
jv_new(int t)
{
	struct jv *v = xcalloc(1, sizeof *v);

	v->t = t;
	return v;
}

struct jv *
jv_strn(const char *s, size_t n)
{
	struct jv *v = jv_new(J_STR);

	v->s = xmalloc(n + 1);
	memcpy(v->s, s, n);
	v->s[n] = 0;
	return v;
}

struct jv *
jv_str(const char *s)
{
	return jv_strn(s, strlen(s));
}

struct jv *
jv_num(double d)
{
	struct jv *v;

	if (!isfinite(d))
		return jv_new(J_NULL);
	v = jv_new(J_NUM);
	v->num = d;
	return v;
}

struct jv *
jv_int(long long i)
{
	struct jv *v = jv_new(J_NUM);
	char buf[32];

	v->num = (double)i;
	snprintf(buf, sizeof buf, "%lld", i);
	v->s = xmalloc(strlen(buf) + 1);
	strcpy(v->s, buf);
	return v;
}

struct jv *
jv_bool(int b)
{
	return jv_new(b ? J_TRUE : J_FALSE);
}

static void
grow(struct jv *a)
{
	if (a->n < a->cap)
		return;
	a->cap = a->cap ? a->cap * 2 : 8;
	a->v = xrealloc(a->v, (size_t)a->cap * sizeof *a->v);
	if (a->t == J_OBJ)
		a->k = xrealloc(a->k, (size_t)a->cap * sizeof *a->k);
}

void
jv_push(struct jv *a, struct jv *v)
{
	grow(a);
	a->v[a->n++] = v;
}

void
jv_set(struct jv *o, const char *key, struct jv *v)
{
	int i;

	for (i = 0; i < o->n; i++)
		if (!strcmp(o->k[i], key)) {
			jv_free(o->v[i]);
			o->v[i] = v;
			return;
		}
	grow(o);
	o->k[o->n] = xmalloc(strlen(key) + 1);
	strcpy(o->k[o->n], key);
	o->v[o->n++] = v;
}

struct jv *
jv_get(const struct jv *o, const char *key)
{
	int i;

	if (o == NULL || o->t != J_OBJ)
		return NULL;
	for (i = 0; i < o->n; i++)
		if (!strcmp(o->k[i], key))
			return o->v[i];
	return NULL;
}

const char *
jv_gets(const struct jv *o, const char *key)
{
	struct jv *v = jv_get(o, key);

	return v && v->t == J_STR ? v->s : NULL;
}

void
jv_free(struct jv *v)
{
	int i;

	if (v == NULL)
		return;
	for (i = 0; i < v->n; i++) {
		jv_free(v->v[i]);
		if (v->k)
			free(v->k[i]);
	}
	free(v->v);
	free(v->k);
	free(v->s);
	free(v);
}

/* ---- parser ---- */

static struct jv *value(struct jp *j);

static int
hex4(const char *p, unsigned *u)
{
	int i;

	*u = 0;
	for (i = 0; i < 4; i++) {
		int c = p[i], d;
		if (c >= '0' && c <= '9')
			d = c - '0';
		else if (c >= 'a' && c <= 'f')
			d = c - 'a' + 10;
		else if (c >= 'A' && c <= 'F')
			d = c - 'A' + 10;
		else
			return 0;
		*u = *u * 16 + (unsigned)d;
	}
	return 1;
}

static void
utf8(struct sbuf *b, unsigned u)
{
	if (u < 0x80)
		sb_putc(b, (int)u);
	else if (u < 0x800) {
		sb_putc(b, (int)(0xc0 | u >> 6));
		sb_putc(b, (int)(0x80 | (u & 0x3f)));
	} else if (u < 0x10000) {
		sb_putc(b, (int)(0xe0 | u >> 12));
		sb_putc(b, (int)(0x80 | (u >> 6 & 0x3f)));
		sb_putc(b, (int)(0x80 | (u & 0x3f)));
	} else {
		sb_putc(b, (int)(0xf0 | u >> 18));
		sb_putc(b, (int)(0x80 | (u >> 12 & 0x3f)));
		sb_putc(b, (int)(0x80 | (u >> 6 & 0x3f)));
		sb_putc(b, (int)(0x80 | (u & 0x3f)));
	}
}

static char *
string(struct jp *j)
{
	struct sbuf b = {0};
	unsigned u, u2;

	j->p++;   /* the quote */
	sb_puts(&b, "");
	for (;;) {
		unsigned char c = (unsigned char)*j->p;
		if (c == 0) {
			jerr(j, "unterminated string");
			break;
		}
		if (c < 0x20) {
			jerr(j, "control character in a string");
			break;
		}
		if (c == '"') {
			j->p++;
			return b.p;
		}
		if (c != '\\') {
			sb_putc(&b, c);
			j->p++;
			continue;
		}
		j->p++;
		switch (*j->p) {
		case '"': sb_putc(&b, '"'); break;
		case '\\': sb_putc(&b, '\\'); break;
		case '/': sb_putc(&b, '/'); break;
		case 'b': sb_putc(&b, '\b'); break;
		case 'f': sb_putc(&b, '\f'); break;
		case 'n': sb_putc(&b, '\n'); break;
		case 'r': sb_putc(&b, '\r'); break;
		case 't': sb_putc(&b, '\t'); break;
		case 'u':
			if (!hex4(j->p + 1, &u)) {
				jerr(j, "bad \\u escape");
				goto out;
			}
			j->p += 4;
			if (u >= 0xd800 && u < 0xdc00) {
				if (j->p[1] != '\\' || j->p[2] != 'u' ||
				    !hex4(j->p + 3, &u2) || u2 < 0xdc00 ||
				    u2 >= 0xe000) {
					jerr(j, "unpaired surrogate");
					goto out;
				}
				j->p += 6;
				u = 0x10000 + ((u - 0xd800) << 10) + (u2 - 0xdc00);
			} else if (u >= 0xdc00 && u < 0xe000) {
				jerr(j, "unpaired surrogate");
				goto out;
			}
			if (u == 0) {
				jerr(j, "\\u0000 is not allowed");
				goto out;
			}
			utf8(&b, u);
			break;
		default:
			jerr(j, "bad escape");
			goto out;
		}
		j->p++;
	}
out:
	free(b.p);
	return NULL;
}

static struct jv *
number(struct jp *j)
{
	const char *s = j->p;
	struct jv *v;

	if (*j->p == '-')
		j->p++;
	if (*j->p == '0')
		j->p++;
	else if (*j->p >= '1' && *j->p <= '9')
		while (*j->p >= '0' && *j->p <= '9')
			j->p++;
	else {
		jerr(j, "bad number");
		return NULL;
	}
	if (*j->p == '.') {
		j->p++;
		if (!(*j->p >= '0' && *j->p <= '9')) {
			jerr(j, "bad number");
			return NULL;
		}
		while (*j->p >= '0' && *j->p <= '9')
			j->p++;
	}
	if (*j->p == 'e' || *j->p == 'E') {
		j->p++;
		if (*j->p == '+' || *j->p == '-')
			j->p++;
		if (!(*j->p >= '0' && *j->p <= '9')) {
			jerr(j, "bad number");
			return NULL;
		}
		while (*j->p >= '0' && *j->p <= '9')
			j->p++;
	}
	v = jv_new(J_NUM);
	v->s = xmalloc((size_t)(j->p - s) + 1);
	memcpy(v->s, s, (size_t)(j->p - s));
	v->s[j->p - s] = 0;
	v->num = strtod(v->s, NULL);
	return v;
}

static struct jv *
value(struct jp *j)
{
	struct jv *v, *e;
	char *k;

	ws(j);
	if (++j->depth > 64) {
		jerr(j, "nested too deeply");
		return NULL;
	}
	switch (*j->p) {
	case '{':
		j->p++;
		v = jv_new(J_OBJ);
		ws(j);
		if (*j->p == '}') {
			j->p++;
			break;
		}
		for (;;) {
			ws(j);
			if (*j->p != '"') {
				jerr(j, "expected a key");
				goto bad;
			}
			if ((k = string(j)) == NULL)
				goto bad;
			if (jv_get(v, k)) {
				free(k);
				jerr(j, "duplicate key");
				goto bad;
			}
			ws(j);
			if (*j->p != ':') {
				free(k);
				jerr(j, "expected ':'");
				goto bad;
			}
			j->p++;
			if ((e = value(j)) == NULL) {
				free(k);
				goto bad;
			}
			jv_set(v, k, e);
			free(k);
			ws(j);
			if (*j->p == ',') {
				j->p++;
				continue;
			}
			if (*j->p == '}') {
				j->p++;
				break;
			}
			jerr(j, "expected ',' or '}'");
			goto bad;
		}
		break;
	case '[':
		j->p++;
		v = jv_new(J_ARR);
		ws(j);
		if (*j->p == ']') {
			j->p++;
			break;
		}
		for (;;) {
			if ((e = value(j)) == NULL)
				goto bad;
			jv_push(v, e);
			ws(j);
			if (*j->p == ',') {
				j->p++;
				continue;
			}
			if (*j->p == ']') {
				j->p++;
				break;
			}
			jerr(j, "expected ',' or ']'");
			goto bad;
		}
		break;
	case '"':
		if ((k = string(j)) == NULL)
			return NULL;
		v = jv_new(J_STR);
		v->s = k;
		break;
	case 't':
		if (strncmp(j->p, "true", 4)) {
			jerr(j, "bad literal");
			return NULL;
		}
		j->p += 4;
		v = jv_new(J_TRUE);
		break;
	case 'f':
		if (strncmp(j->p, "false", 5)) {
			jerr(j, "bad literal");
			return NULL;
		}
		j->p += 5;
		v = jv_new(J_FALSE);
		break;
	case 'n':
		if (strncmp(j->p, "null", 4)) {
			jerr(j, "bad literal");
			return NULL;
		}
		j->p += 4;
		v = jv_new(J_NULL);
		break;
	default:
		if (*j->p == '-' || (*j->p >= '0' && *j->p <= '9'))
			return j->depth--, number(j);
		jerr(j, *j->p ? "unexpected character" : "unexpected end");
		return NULL;
	}
	j->depth--;
	return v;
bad:
	jv_free(v);
	return NULL;
}

struct jv *
jv_parse(const char *s, char *err, size_t errn, size_t *pos)
{
	struct jp j;
	struct jv *v;

	memset(&j, 0, sizeof j);
	j.s = j.p = s;
	j.err = err;
	j.errn = errn;
	v = value(&j);
	if (v) {
		ws(&j);
		if (*j.p) {
			jerr(&j, "text after the value");
			jv_free(v);
			v = NULL;
		}
	}
	if (v == NULL && !j.bad)
		jerr(&j, "bad JSON");
	if (pos)
		*pos = (size_t)(j.p - s);
	return v;
}

/* ---- writer ---- */

void
json_quoten(struct sbuf *b, const char *s, size_t n)
{
	size_t i;

	sb_putc(b, '"');
	for (i = 0; i < n; i++) {
		unsigned char c = (unsigned char)s[i];
		switch (c) {
		case '"': sb_puts(b, "\\\""); break;
		case '\\': sb_puts(b, "\\\\"); break;
		case '\n': sb_puts(b, "\\n"); break;
		case '\r': sb_puts(b, "\\r"); break;
		case '\t': sb_puts(b, "\\t"); break;
		default:
			if (c < 0x20 || c == 0x7f)
				sb_printf(b, "\\u%04x", c);
			else
				sb_putc(b, c);
		}
	}
	sb_putc(b, '"');
}

void
json_quote(struct sbuf *b, const char *s)
{
	json_quoten(b, s ? s : "", s ? strlen(s) : 0);
}

void
json_num(struct sbuf *b, double d)
{
	char buf[40];
	int prec;

	if (!isfinite(d)) {
		sb_puts(b, "null");
		return;
	}
	if (d == 0) {
		sb_puts(b, "0");
		return;
	}
	for (prec = 15; prec <= 17; prec++) {
		snprintf(buf, sizeof buf, "%.*g", prec, d);
		if (strtod(buf, NULL) == d)
			break;
	}
	sb_puts(b, buf);
}

void
jv_write(struct sbuf *b, const struct jv *v)
{
	int i;

	switch (v->t) {
	case J_NULL: sb_puts(b, "null"); break;
	case J_FALSE: sb_puts(b, "false"); break;
	case J_TRUE: sb_puts(b, "true"); break;
	case J_NUM:
		if (v->s)
			sb_puts(b, v->s);
		else
			json_num(b, v->num);
		break;
	case J_STR: json_quote(b, v->s); break;
	case J_ARR:
		sb_putc(b, '[');
		for (i = 0; i < v->n; i++) {
			if (i)
				sb_puts(b, ", ");
			jv_write(b, v->v[i]);
		}
		sb_putc(b, ']');
		break;
	case J_OBJ:
		sb_putc(b, '{');
		for (i = 0; i < v->n; i++) {
			if (i)
				sb_puts(b, ", ");
			json_quote(b, v->k[i]);
			sb_puts(b, ": ");
			jv_write(b, v->v[i]);
		}
		sb_putc(b, '}');
		break;
	}
}

void
b64_encode(struct sbuf *b, const unsigned char *p, size_t n)
{
	static const char t[] =
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	size_t i;

	for (i = 0; i + 2 < n; i += 3) {
		sb_putc(b, t[p[i] >> 2]);
		sb_putc(b, t[(p[i] & 3) << 4 | p[i + 1] >> 4]);
		sb_putc(b, t[(p[i + 1] & 15) << 2 | p[i + 2] >> 6]);
		sb_putc(b, t[p[i + 2] & 63]);
	}
	if (i < n) {
		sb_putc(b, t[p[i] >> 2]);
		if (i + 1 < n) {
			sb_putc(b, t[(p[i] & 3) << 4 | p[i + 1] >> 4]);
			sb_putc(b, t[(p[i + 1] & 15) << 2]);
		} else {
			sb_putc(b, t[(p[i] & 3) << 4]);
			sb_putc(b, '=');
		}
		sb_putc(b, '=');
	}
}
