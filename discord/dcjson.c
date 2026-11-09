/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * JSON over Discord (dcjson.h).  The requests are the JSON frontend's
 * (src/jsonmode.c); the channel's worker answers them with the same
 * handle() as `ucas --json` (json_bot_serve).  What the worker's sandbox
 * cannot do happens here, around it:
 *
 *   before   GLSL in args.code compiled to SPIR-V (glslangValidator);
 *            {"attachment": N} in args.code or args.inputs fetched;
 *            "files" made base64 (nothing is written on the host);
 *            reset and schema answered by the bot itself
 *   after    the GLSL view (spirv-cross) put into "text"; files decoded
 *            and attached, listed by name; the bot's notes added
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "discord.h"

#include "ucas.h"
#include "json.h"
#include "jsonmode.h"
#include "sup.h"
#include "text.h"
#include "attach.h"
#include "dcjson.h"

size_t
dj_b64dec(const char *s, size_t n, char *out, size_t cap)
{
	unsigned v = 0;
	int bits = 0;
	size_t i, o = 0;

	for (i = 0; i < n; i++) {
		int c = (unsigned char)s[i], d;
		if (c >= 'A' && c <= 'Z')
			d = c - 'A';
		else if (c >= 'a' && c <= 'z')
			d = c - 'a' + 26;
		else if (c >= '0' && c <= '9')
			d = c - '0' + 52;
		else if (c == '+')
			d = 62;
		else if (c == '/')
			d = 63;
		else
			continue;
		v = (v << 6) | (unsigned)d;
		bits += 6;
		if (bits >= 8) {
			bits -= 8;
			if (o < cap)
				out[o++] = (char)(v >> bits & 0xff);
		}
	}
	return o;
}

int
xspv_expand(const char *out, size_t olen, struct sbuf *t, struct dfile *files,
    int *nfiles, int maxf, struct sbuf *report)
{
	const char *xs = memmem(out, olen, "\x1e" "xspv ", 6);
	const char *xa = xs ? memmem(xs, (size_t)(out + olen - xs),
	    "\x1e" "and\n", 5) : NULL;
	const char *xe = xa ? memmem(xa, (size_t)(out + olen - xa),
	    "\x1e" "end\n", 5) : NULL;

	if (xs == NULL)
		return 0;
	if (xe == NULL) {	/* cut by the output limit */
		sb_printf(t, "%.*s", (int)(xs - out), out);
		return 1;
	}
	{
		struct sbuf rep = {0}, og = {0}, sg = {0};
		const char *b1 = memchr(xs, '\n', (size_t)(xa - xs));
		size_t c1 = (size_t)(xa - xs) * 3 / 4 + 4,
		    c2 = (size_t)(xe - xa) * 3 / 4 + 4, n1 = 0, n2;
		char *om = xmalloc(c1 + 1), *sm = xmalloc(c2 + 1);
		int nofile = atoi(xs + 6);
		if (b1 != NULL)
			n1 = dj_b64dec(b1 + 1, (size_t)(xa - b1 - 1), om, c1);
		n2 = dj_b64dec(xa + 5, (size_t)(xe - xa - 5), sm, c2);
		om[n1] = 0;
		sm[n2] = 0;
		sb_printf(t, "%.*s", (int)(xs - out), out);
		spvcross_view(n1 ? om : NULL, sm, &rep, &og, &sg);
		sb_puts(t, rep.p ? rep.p : "");
		if (report != NULL && rep.n)
			sb_puts(report, rep.p);
		if (sg.n && nofile) {
			const struct sbuf *g[2] = {&sg, &og};
			const char *gn[2] = {"simplified.glsl", "original.glsl"};
			int k;
			for (k = 0; k < 2; k++) {
				uint32_t hsh = 2166136261u;
				size_t q;
				if (g[k]->n == 0)
					continue;
				for (q = 0; q < g[k]->n; q++)
					hsh = (hsh ^ (unsigned char)g[k]->p[q]) * 16777619u;
				sb_printf(t, "  (nofile: %s, %zu bytes, checksum "
				    "%08x)\n", gn[k], g[k]->n, hsh);
			}
		} else if (sg.n && *nfiles < maxf) {
			struct dfile *f = &files[(*nfiles)++];
			snprintf(f->name, sizeof f->name, "simplified.glsl");
			snprintf(f->type, sizeof f->type, "text/plain");
			f->data = sg.p;
			f->len = sg.n;
			sg.p = NULL;
			sb_puts(t, "  spirv-cross's GLSL of it is attached as "
			    "simplified.glsl\n");
		}
		sb_printf(t, "%.*s", (int)(out + olen - (xe + 5)), xe + 5);
		free(rep.p);
		free(og.p);
		free(sg.p);
		free(om);
		free(sm);
	}
	return 1;
}

/* ---- reading the request ---- */

int
dj_find(const char *s, struct sbuf *out)
{
	const char *a, *b, *p;

	for (p = s; p && (a = strstr(p, "```")) != NULL; p = a + 3) {
		const char *t = a + 3;
		if (strncasecmp(t, "json", 4) != 0 || !isspace((unsigned char)t[4]))
			continue;
		t += 4;
		if ((b = strstr(t, "```")) == NULL)
			return 0;
		sb_clear(out);
		sb_printf(out, "%.*s", (int)(b - t), t);
		return 1;
	}
	return 0;
}

static struct jv *
jv_copy(const struct jv *v)
{
	struct sbuf b = {0};
	struct jv *c;
	char err[80];
	size_t pos;

	if (v == NULL)
		return jv_new(J_NULL);
	jv_write(&b, v);
	c = jv_parse(b.p, err, sizeof err, &pos);
	free(b.p);
	return c ? c : jv_new(J_NULL);
}

struct jv *
dj_error(const struct jv *id, const char *cmd, const char *code,
    const char *msg)
{
	struct jv *o = jv_new(J_OBJ), *e = jv_new(J_OBJ);

	jv_set(o, "id", jv_copy(id));
	jv_set(o, "ok", jv_bool(0));
	jv_set(o, "cmd", cmd ? jv_str(cmd) : jv_new(J_NULL));
	jv_set(e, "code", jv_str(code));
	jv_set(e, "message", jv_str(msg));
	jv_set(o, "error", e);
	return o;
}

/* a block from attach_block: its fence tag and body */
static void
split_block(const char *blk, struct sbuf *tag, struct sbuf *body)
{
	const char *p = strstr(blk, "```"), *nl, *e;

	sb_clear(tag);
	sb_clear(body);
	sb_puts(tag, "");
	sb_puts(body, "");
	if (p == NULL || (nl = strchr(p, '\n')) == NULL)
		return;
	sb_printf(tag, "%.*s", (int)(nl - p - 3), p + 3);
	e = blk + strlen(blk);
	if (e - nl >= 4 && !strncmp(e - 4, "\n```", 4))
		e -= 4;
	if (e < nl + 1)
		e = nl + 1;
	sb_printf(body, "%.*s", (int)(e - nl - 1), nl + 1);
}

/* {"attachment": N} -> N - 1, else -1 (-2: a malformed one) */
static int
att_ref(const struct jv *v)
{
	const struct jv *a;

	if (v == NULL || v->t != J_OBJ || (a = jv_get(v, "attachment")) == NULL)
		return -1;
	if (v->n != 1 || a->t != J_NUM || a->num < 1 || a->num > 10 ||
	    a->num != (int)a->num)
		return -2;
	return (int)a->num - 1;
}

/* the fetched attachment k: 1 with its tag and body, else 0 with why */
static int
fetch_att(const struct discord_message *m, int k, struct sbuf *tag,
    struct sbuf *body, struct sbuf *why)
{
	struct sbuf blk = {0}, notes = {0};
	int ok;

	if (m->attachments == NULL || k >= m->attachments->size) {
		sb_printf(why, "attachment %d: the message has %d", k + 1,
		    m->attachments ? m->attachments->size : 0);
		return 0;
	}
	ok = attach_block(m, k, &blk, &notes);
	if (ok)
		split_block(blk.p, tag, body);
	else
		sb_printf(why, "attachment %d: %s", k + 1, notes.p ? notes.p :
		    "not readable");
	while (why->n && why->p[why->n - 1] == '\n')
		why->p[--why->n] = 0;
	free(blk.p);
	free(notes.p);
	return ok;
}

/* one request's args, made ready; 0 with *reply on an error */
static int
prep_one(const struct discord_message *m, struct jv *rq, struct jv *notes,
    struct jv **reply)
{
	struct jv *a = rq->t == J_OBJ ? jv_get(rq, "args") : NULL, *v;
	const char *cmd = rq->t == J_OBJ ? jv_gets(rq, "cmd") : NULL;
	const struct jv *id = rq->t == J_OBJ ? jv_get(rq, "id") : NULL;
	struct sbuf tag = {0}, body = {0}, why = {0};
	const char *code, *lang;
	int k, n, ok = 1;

	if (a == NULL || a->t != J_OBJ || cmd == NULL)
		return 1;	/* handle() says what is wrong */
	if ((v = jv_get(a, "files")) != NULL && !(v->t == J_STR &&
	    !strcmp(v->s, "none")))
		jv_set(a, "files", jv_str("base64"));
	if (jv_get(a, "timeout_s") != NULL)
		jv_push(notes, jv_str("timeout_s is not used here: the bot's own "
		    "time limits apply"));
	/* {"attachment": N} as the code */
	if ((n = att_ref(jv_get(a, "code"))) != -1) {
		if (n < 0 || !fetch_att(m, n, &tag, &body, &why) ||
		    !strncmp(tag.p, "image", 5)) {
			if (n < 0)
				sb_printf(&why, "args.code: {\"attachment\": N} with N "
				    "from 1");
			else if (why.n == 0)
				sb_printf(&why, "attachment %d is an image, not code",
				    n + 1);
			*reply = dj_error(id, cmd, "bad_attachment", why.p);
			ok = 0;
			goto out;
		}
		jv_set(a, "code", jv_str(body.p));
		if (!strcmp(tag.p, "c") && (jv_gets(a, "lang") != NULL ||
		    !strcmp(cmd, "spirv_simplify")))
			jv_set(a, "lang", jv_str("c"));
		else if (!strcmp(tag.p, "spv") && jv_gets(a, "lang") != NULL)
			jv_set(a, "lang", jv_str("spv"));
		jv_push(notes, jv_str(!strcmp(tag.p, "spv") ? "args.code: the "
		    "attachment, as SPIR-V assembly" : !strcmp(tag.p, "c") ?
		    "args.code: the attachment, as C" : "args.code: the "
		    "attachment"));
	}
	/* {"attachment": N} among the image inputs */
	if ((v = jv_get(a, "inputs")) != NULL && v->t == J_ARR)
		for (k = 0; k < v->n; k++) {
			struct jv *e;
			if ((n = att_ref(v->v[k])) == -1)
				continue;
			sb_clear(&why);
			if (n < 0 || !fetch_att(m, n, &tag, &body, &why) ||
			    strncmp(tag.p, "image", 5) != 0) {
				if (n < 0)
					sb_printf(&why, "args.inputs[%d]: {\"attachment\": "
					    "N} with N from 1", k);
				else if (why.n == 0)
					sb_printf(&why, "attachment %d is not an image",
					    n + 1);
				*reply = dj_error(id, cmd, "bad_attachment", why.p);
				ok = 0;
				goto out;
			}
			e = jv_new(J_OBJ);
			{
				/* the base64 without its line breaks */
				struct sbuf b = {0};
				const char *s;
				for (s = body.p; *s; s++)
					if (*s != '\n')
						sb_putc(&b, *s);
				jv_set(e, "image_base64", jv_str(sb_str(&b)));
				free(b.p);
			}
			if (tag.n > 6)
				jv_set(e, "name", jv_str(tag.p + 6));
			jv_free(v->v[k]);
			v->v[k] = e;
		}
	/* GLSL: compiled here, as the engine's sandbox cannot */
	code = jv_gets(a, "code");
	lang = jv_gets(a, "lang");
	if (code != NULL && strncmp(cmd, "c_", 2) != 0 &&
	    !(lang && !strcmp(lang, "c")) && !(!strcmp(cmd, "spirv_simplify") &&
	    !lang && code_lang(code, "", NULL) == CL_C) &&
	    glsl_like(code, lang ? lang : "")) {
		struct sbuf spv = {0}, nt = {0};
		if (glsl_to_spvasm(code, &spv, &nt)) {
			struct sbuf s = {0};
			sb_puts(&s, "the code was compiled to SPIR-V with "
			    "glslangValidator");
			if (nt.n)
				sb_printf(&s, "\n%s", nt.p);
			while (s.n && s.p[s.n - 1] == '\n')
				s.p[--s.n] = 0;
			jv_push(notes, jv_str(s.p));
			jv_set(a, "code", jv_str(sb_str(&spv)));
			if (lang != NULL)
				jv_set(a, "lang", jv_str("spv"));
			free(s.p);
		} else {
			struct sbuf s = {0};
			sb_printf(&s, "the GLSL did not compile:\n%s", nt.p ? nt.p :
			    "");
			while (s.n && s.p[s.n - 1] == '\n')
				s.p[--s.n] = 0;
			*reply = dj_error(id, cmd, "bad_code", s.p);
			free(s.p);
			ok = 0;
		}
		free(spv.p);
		free(nt.p);
	}
out:
	free(tag.p);
	free(body.p);
	free(why.p);
	return ok;
}

/* user text must not carry engine directives (\x1e lines) */
static void
strip_rs(struct jv *v)
{
	int k;

	if (v == NULL)
		return;
	if (v->t == J_STR && v->s && strchr(v->s, '\x1e')) {
		char *d = v->s, *p;
		for (p = v->s; *p; p++)
			if (*p != '\x1e')
				*d++ = *p;
		*d = 0;
	}
	if (v->t == J_ARR || v->t == J_OBJ)
		for (k = 0; k < v->n; k++)
			strip_rs(v->v[k]);
}

/* an array refused for its request k: that error, "not_run" for the rest */
static struct jv *
refuse_array(const struct jv *rq, int k, struct jv *err)
{
	struct jv *a = jv_new(J_ARR);
	char msg[120];
	int i;

	snprintf(msg, sizeof msg, "not run: request %d of this message was "
	    "refused", k + 1);
	for (i = 0; i < rq->n; i++) {
		const struct jv *q = rq->v[i];
		if (i == k)
			jv_push(a, err);
		else
			jv_push(a, dj_error(q->t == J_OBJ ? jv_get(q, "id") : NULL,
			    q->t == J_OBJ ? jv_gets(q, "cmd") : NULL, "not_run",
			    msg));
	}
	return a;
}

int
dj_prepare(const struct discord_message *m, const char *body,
    struct djreq *r, struct jv **reply)
{
	char err[160], msg[240];
	size_t pos;
	int k, n;

	memset(r, 0, sizeof *r);
	*reply = NULL;
	r->rq = jv_parse(body, err, sizeof err, &pos);
	if (r->rq == NULL) {
		snprintf(msg, sizeof msg, "%s at byte %zu of the json block",
		    err, pos);
		*reply = dj_error(NULL, NULL, "bad_json", msg);
		return DJ_REPLY;
	}
	strip_rs(r->rq);
	r->arr = r->rq->t == J_ARR;
	n = r->arr ? r->rq->n : 1;
	if (r->arr && (n == 0 || n > DJ_MAXREQ)) {
		snprintf(msg, sizeof msg, "an array of 1 to %d requests",
		    DJ_MAXREQ);
		*reply = dj_error(NULL, NULL, "bad_request", msg);
		return DJ_REPLY;
	}
	r->notes = jv_new(J_ARR);
	for (k = 0; k < n; k++) {
		struct jv *q = r->arr ? r->rq->v[k] : r->rq;
		const char *cmd = q->t == J_OBJ ? jv_gets(q, "cmd") : NULL;
		jv_push(r->notes, jv_new(J_ARR));
		if (cmd && (!strcmp(cmd, "reset") || !strcmp(cmd, "schema"))) {
			if (r->arr) {
				snprintf(msg, sizeof msg, "%s goes alone in a message, "
				    "not in an array", cmd);
				*reply = refuse_array(r->rq, k, dj_error(jv_get(q, "id"),
				    cmd, "bad_request", msg));
				return DJ_REPLY;
			}
			return !strcmp(cmd, "reset") ? DJ_RESET : DJ_SCHEMA;
		}
		if (!prep_one(m, q, r->notes->v[k], reply)) {
			if (r->arr)	/* the whole message is refused */
				*reply = refuse_array(r->rq, k, *reply);
			return DJ_REPLY;
		}
	}
	return DJ_WORKER;
}

void
dj_free(struct djreq *r)
{
	jv_free(r->rq);
	jv_free(r->notes);
	memset(r, 0, sizeof *r);
}

/* ---- the answer ---- */

static const char *
status_code(uint32_t st)
{
	switch (st & SR_MASK) {
	case SR_TIMEOUT:
		return "timeout";
	case SR_CPU:
		return "cpu_limit";
	case SR_MEM:
		return "memory_limit";
	case SR_BUSY:
		return "busy";
	case SR_LOST:
		return "lost";
	default:
		return "internal";
	}
}

/* every request answered by the same error */
static struct jv *
all_error(const struct djreq *r, const char *code, const char *msg)
{
	struct jv *a;
	int k;

	if (!r->arr)
		return dj_error(r->rq->t == J_OBJ ? jv_get(r->rq, "id") : NULL,
		    r->rq->t == J_OBJ ? jv_gets(r->rq, "cmd") : NULL, code, msg);
	a = jv_new(J_ARR);
	for (k = 0; k < r->rq->n; k++) {
		const struct jv *q = r->rq->v[k];
		jv_push(a, dj_error(q->t == J_OBJ ? jv_get(q, "id") : NULL,
		    q->t == J_OBJ ? jv_gets(q, "cmd") : NULL, code, msg));
	}
	return a;
}

/* a file name not yet among files */
static void
unique_name(char *name, size_t sz, const char *want, const struct dfile *fs,
    int n)
{
	int k, i, used;

	snprintf(name, sz, "%s", want);
	for (i = 2;; i++) {
		for (used = 0, k = 0; k < n; k++)
			used |= !strcmp(fs[k].name, name);
		if (!used)
			return;
		snprintf(name, sz, "%d-%s", i, want);
	}
}

/* one response: notes, the GLSL view, files */
static void
finish_one(struct jv *resp, const struct jv *notes, struct dfile *files,
    int *nfiles, int maxf)
{
	struct jv *nt, *fs, *res, *nfs;
	const char *text;
	int k;

	if (resp->t != J_OBJ)
		return;
	nt = jv_get(resp, "notes");
	if (notes && notes->n) {
		if (nt == NULL || nt->t != J_ARR) {
			nt = jv_new(J_ARR);
			jv_set(resp, "notes", nt);
			nt = jv_get(resp, "notes");
		}
		for (k = 0; k < notes->n; k++) {
			struct jv *o = jv_new(J_OBJ);
			jv_set(o, "kind", jv_str("info"));
			jv_set(o, "text", jv_str(notes->v[k]->s));
			jv_push(nt, o);
		}
	}
	fs = jv_get(resp, "files");
	nfs = jv_new(J_ARR);
	/* the engine's files: decoded, attached, named */
	for (k = 0; fs && fs->t == J_ARR && k < fs->n; k++) {
		const struct jv *f = fs->v[k];
		const char *b64 = jv_gets(f, "base64"), *nm = jv_gets(f, "name"),
		    *mime = jv_gets(f, "mime");
		struct jv *o = jv_new(J_OBJ);
		char name[64];
		unique_name(name, sizeof name, nm ? nm : "file", files, *nfiles);
		jv_set(o, "name", jv_str(name));
		jv_set(o, "mime", jv_str(mime ? mime : "application/octet-stream"));
		jv_set(o, "bytes", jv_copy(jv_get(f, "bytes")));
		if (b64 != NULL && *nfiles < maxf) {
			struct dfile *d = &files[(*nfiles)++];
			size_t cap = strlen(b64) * 3 / 4 + 4;
			d->data = xmalloc(cap);
			d->len = dj_b64dec(b64, strlen(b64), d->data, cap);
			snprintf(d->name, sizeof d->name, "%s", name);
			snprintf(d->type, sizeof d->type, "%s", mime ? mime :
			    "application/octet-stream");
			jv_set(o, "attached", jv_bool(1));
		} else {
			jv_set(o, "attached", jv_bool(0));
			if (b64 != NULL)
				jv_set(o, "why", jv_str("too many files for one "
				    "message"));
		}
		jv_push(nfs, o);
	}
	/* the GLSL view, made here (spirv-cross) */
	text = jv_gets(resp, "text");
	if (text != NULL && strstr(text, "\x1e" "xspv ") != NULL) {
		struct sbuf t = {0}, rep = {0};
		int before = *nfiles;
		char *old = strdup(text);
		xspv_expand(old, strlen(old), &t, files, nfiles, maxf, &rep);
		jv_set(resp, "text", jv_str(sb_str(&t)));
		res = jv_get(resp, "result");
		if (res && res->t == J_OBJ) {
			const char *rt = jv_gets(res, "text");
			if (rt && !strcmp(rt, old))
				jv_set(res, "text", jv_str(sb_str(&t)));
			else if (rep.n)
				jv_set(res, "glsl_view", jv_str(rep.p));
		}
		for (k = before; k < *nfiles; k++) {
			struct jv *o = jv_new(J_OBJ);
			jv_set(o, "name", jv_str(files[k].name));
			jv_set(o, "mime", jv_str("text/x-glsl"));
			jv_set(o, "bytes", jv_int((long long)files[k].len));
			jv_set(o, "attached", jv_bool(1));
			jv_push(nfs, o);
		}
		free(old);
		free(t.p);
		free(rep.p);
	}
	jv_set(resp, "files", nfs);
	/* the error last, as in JSON mode */
	for (k = 0; k < resp->n - 1; k++)
		if (!strcmp(resp->k[k], "error")) {
			struct jv *e = resp->v[k];
			char *kk = resp->k[k];
			memmove(resp->v + k, resp->v + k + 1,
			    (size_t)(resp->n - k - 1) * sizeof *resp->v);
			memmove(resp->k + k, resp->k + k + 1,
			    (size_t)(resp->n - k - 1) * sizeof *resp->k);
			resp->v[resp->n - 1] = e;
			resp->k[resp->n - 1] = kk;
			break;
		}
}

struct jv *
dj_result(const struct djreq *r, uint32_t status, const char *smsg,
    const char *out, size_t n, struct dfile *files, int *nfiles, int maxf)
{
	const char *mk = out ? memmem(out, n, "\x1e" "json\n", 6) : NULL;
	struct jv *resp;
	char err[160];
	size_t pos;
	int k;

	if ((status & SR_MASK) != SR_OK)
		return all_error(r, status_code(status), smsg ? smsg : "failed");
	if (mk == NULL || (resp = jv_parse(mk + 6, err, sizeof err, &pos)) ==
	    NULL)
		return all_error(r, (status & SR_TRUNC) ? "output_limit" :
		    "internal", (status & SR_TRUNC) && smsg ? smsg :
		    "the evaluator gave no response");
	if (r->arr && resp->t == J_ARR)
		for (k = 0; k < resp->n; k++)
			finish_one(resp->v[k], k < r->notes->n ? r->notes->v[k] :
			    NULL, files, nfiles, maxf);
	else
		finish_one(resp, r->notes && r->notes->n ? r->notes->v[0] : NULL,
		    files, nfiles, maxf);
	return resp;
}

struct jv *
dj_reset_response(const struct djreq *r)
{
	struct jv *o = jv_new(J_OBJ), *res = jv_new(J_OBJ);

	jv_set(o, "id", jv_copy(jv_get(r->rq, "id")));
	jv_set(o, "ok", jv_bool(1));
	jv_set(o, "cmd", jv_str("reset"));
	jv_set(res, "kind", jv_str("reset"));
	jv_set(o, "result", res);
	jv_set(o, "text", jv_str("state reset for this channel"));
	return o;
}

struct jv *
dj_schema_response(const struct djreq *r, struct dfile *f)
{
	struct jv *o = jv_new(J_OBJ), *res = jv_new(J_OBJ), *fs = jv_new(J_ARR),
	    *fo = jv_new(J_OBJ), *s = json_schema();
	struct sbuf b = {0};

	jv_write(&b, s);
	sb_putc(&b, '\n');
	jv_free(s);
	snprintf(f->name, sizeof f->name, DJ_SCHEMA_FILE);
	snprintf(f->type, sizeof f->type, "application/json");
	f->data = b.p;
	f->len = b.n;
	jv_set(o, "id", jv_copy(jv_get(r->rq, "id")));
	jv_set(o, "ok", jv_bool(1));
	jv_set(o, "cmd", jv_str("schema"));
	jv_set(res, "kind", jv_str("schema"));
	jv_set(res, "attached", jv_str(DJ_SCHEMA_FILE));
	jv_set(o, "result", res);
	jv_set(fo, "name", jv_str(DJ_SCHEMA_FILE));
	jv_set(fo, "mime", jv_str("application/json"));
	jv_set(fo, "bytes", jv_int((long long)b.n));
	jv_set(fo, "attached", jv_bool(1));
	jv_push(fs, fo);
	jv_set(o, "files", fs);
	return o;
}

int
dj_all_ok(const struct jv *resp)
{
	const struct jv *ok;
	int k;

	if (resp == NULL)
		return 0;
	if (resp->t == J_ARR) {
		for (k = 0; k < resp->n; k++)
			if (!dj_all_ok(resp->v[k]))
				return 0;
		return 1;
	}
	ok = jv_get(resp, "ok");
	return ok && ok->t == J_TRUE;
}

/* ---- the message ---- */

/* ``` in a string would end the code block: ` as ` */
static void
fence(struct sbuf *msg, const struct jv *v)
{
	struct sbuf b = {0};
	size_t i;

	jv_write(&b, v);
	sb_clear(msg);
	sb_puts(msg, "```json\n");
	for (i = 0; i < b.n; i++)
		if (b.p[i] == '`')
			sb_puts(msg, "\\u0060");
		else
			sb_putc(msg, b.p[i]);
	sb_puts(msg, "\n```");
	free(b.p);
}

/* s cut to at most max bytes, at a character boundary */
static struct jv *
cut_str(const char *s, size_t max)
{
	size_t n = strlen(s);

	if (n <= max)
		return jv_str(s);
	while (max > 0 && ((unsigned char)s[max] & 0xc0) == 0x80)
		max--;
	return jv_strn(s, max);
}

/* resp with every "text" (and result.text) at most max bytes */
static struct jv *
with_text_cut(const struct jv *resp, size_t max)
{
	struct jv *c = jv_copy(resp);
	int k, n = c->t == J_ARR ? c->n : 1;

	for (k = 0; k < n; k++) {
		struct jv *o = c->t == J_ARR ? c->v[k] : c, *res;
		const char *t;
		if (o->t != J_OBJ)
			continue;
		if ((t = jv_gets(o, "text")) != NULL && strlen(t) > max) {
			jv_set(o, "text", cut_str(t, max));
			jv_set(o, "text_truncated", jv_bool(1));
		}
		res = jv_get(o, "result");
		if (res && res->t == J_OBJ && (t = jv_gets(res, "text")) != NULL &&
		    strlen(t) > max) {
			jv_set(res, "text", cut_str(t, max));
			jv_set(o, "text_truncated", jv_bool(1));
		}
		jv_set(o, "full", jv_str(DJ_FULL));
	}
	return c;
}

/* the first two lines of text that are not notes */
static void
headline(const char *text, struct sbuf *h)
{
	const char *p = text, *e, *s;
	int got = 0;

	while (p && *p && got < 2) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		for (s = p; s < e && (*s == ' ' || *s == '\t'); s++)
			;
		if (e > s && strncmp(s, "note:", 5) != 0) {
			sb_printf(h, "%s%.*s", got ? " | " : "", (int)(e - s), s);
			got++;
		}
		p = *e ? e + 1 : e;
	}
}

static struct jv *
summary_one(const struct jv *o)
{
	struct jv *s = jv_new(J_OBJ), *fs = jv_new(J_ARR);
	const struct jv *e = jv_get(o, "error"), *f = jv_get(o, "files");
	struct sbuf h = {0};
	int k;

	jv_set(s, "id", jv_copy(jv_get(o, "id")));
	jv_set(s, "ok", jv_copy(jv_get(o, "ok")));
	jv_set(s, "cmd", jv_copy(jv_get(o, "cmd")));
	headline(jv_gets(o, "text"), &h);
	if (h.n == 0 && e)
		sb_puts(&h, jv_gets(e, "message") ? jv_gets(e, "message") : "");
	jv_set(s, "headline", cut_str(sb_str(&h), 300));
	if (e && e->t == J_OBJ) {
		struct jv *ee = jv_new(J_OBJ);
		jv_set(ee, "code", jv_copy(jv_get(e, "code")));
		jv_set(ee, "message", jv_copy(jv_get(e, "message")));
		if (jv_get(e, "key"))
			jv_set(ee, "key", jv_copy(jv_get(e, "key")));
		jv_set(s, "error", ee);
	}
	for (k = 0; f && f->t == J_ARR && k < f->n; k++)
		if (jv_gets(f->v[k], "name"))
			jv_push(fs, jv_str(jv_gets(f->v[k], "name")));
	jv_set(s, "files", fs);
	jv_set(s, "full", jv_str(DJ_FULL));
	free(h.p);
	return s;
}

void
dj_format(const struct jv *resp, struct sbuf *msg, struct dfile *files,
    int *nfiles, int maxf)
{
	struct jv *c;
	struct sbuf full = {0};
	size_t lo, hi, mid;
	int k;

	fence(msg, resp);
	if (msg->n <= DC_LIMIT)
		return;
	/* the whole response, attached */
	jv_write(&full, resp);
	sb_putc(&full, '\n');
	if (*nfiles < maxf) {
		struct dfile *f = &files[(*nfiles)++];
		snprintf(f->name, sizeof f->name, DJ_FULL);
		snprintf(f->type, sizeof f->type, "application/json");
		f->data = full.p;
		f->len = full.n;
		full.p = NULL;
	}
	free(full.p);
	/* the response with its text cut, as long as it fits */
	c = with_text_cut(resp, 0);
	fence(msg, c);
	jv_free(c);
	if (msg->n <= DC_LIMIT) {
		lo = 0;
		hi = DC_LIMIT;
		while (lo < hi) {	/* the longest cut that fits */
			mid = (lo + hi + 1) / 2;
			c = with_text_cut(resp, mid);
			fence(msg, c);
			jv_free(c);
			if (msg->n <= DC_LIMIT)
				lo = mid;
			else
				hi = mid - 1;
		}
		c = with_text_cut(resp, lo);
		fence(msg, c);
		jv_free(c);
		return;
	}
	/* a summary */
	if (resp->t == J_ARR) {
		c = jv_new(J_ARR);
		for (k = 0; k < resp->n; k++)
			jv_push(c, summary_one(resp->v[k]));
	} else
		c = summary_one(resp);
	fence(msg, c);
	jv_free(c);
	if (msg->n <= DC_LIMIT)
		return;
	c = jv_new(J_OBJ);
	jv_set(c, "full", jv_str(DJ_FULL));
	jv_set(c, "ok", jv_bool(dj_all_ok(resp)));
	fence(msg, c);
	jv_free(c);
}
