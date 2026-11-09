/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * JSON over Discord: a ```json block holding JSON frontend requests (the
 * protocol of `ucas --json`), answered with a ```json block of the
 * responses, files attached.  See README.md, "Bots and programs".
 */
#ifndef UCAS_DISCORD_DCJSON_H
#define UCAS_DISCORD_DCJSON_H

#include <stddef.h>
#include <stdint.h>

struct discord_message;
struct sbuf;
struct jv;

/* a file for a reply: the engine's (a GIF, simplified.spvasm, ...) */
struct dfile {
	char name[64], type[64];
	char *data;
	size_t len;
};
#define MAXDFILES 6		/* the engine's files on a text reply */
#define MAXPOST 8		/* files on any reply */

#define DJ_MAXREQ 8		/* requests in one message */
#define DJ_FULL "response.json"
#define DJ_SCHEMA_FILE "schema.json"

enum {
	DJ_WORKER,	/* send r->rq to the channel's worker */
	DJ_REPLY,	/* answered here: *reply */
	DJ_RESET,	/* {"cmd": "reset"} */
	DJ_SCHEMA	/* {"cmd": "schema"} */
};

struct djreq {
	struct jv *rq;		/* the request (or array) as sent on */
	struct jv *notes;	/* per request, the bot's notes (strings) */
	int arr;		/* rq is an array */
};

/* base64 to bytes, skipping what is not base64; returns the length */
size_t dj_b64dec(const char *s, size_t n, char *out, size_t cap);

/*
 * "\x1exspv ..." in out (the simplified module and the one before, for
 * spirv-cross, which the engine's sandbox cannot run): the text with the
 * GLSL view in its place goes to t and simplified.glsl to files.  0 if
 * there is no such block (t untouched).  report, if not NULL, gets the
 * view's report.
 */
int xspv_expand(const char *out, size_t olen, struct sbuf *t,
    struct dfile *files, int *nfiles, int maxf, struct sbuf *report);

/* the body of the first ```json block of s; 1 if there is one */
int dj_find(const char *s, struct sbuf *out);

/* read the request(s) and fetch what they point at; returns DJ_* */
int dj_prepare(const struct discord_message *m, const char *body,
    struct djreq *r, struct jv **reply);

/*
 * The response(s) for the worker's answer: status is the supervisor's
 * (SR_*), smsg its status line.  Files are decoded into files (at most
 * maxf); the response lists them by name, without base64.
 */
struct jv *dj_result(const struct djreq *r, uint32_t status,
    const char *smsg, const char *out, size_t n, struct dfile *files,
    int *nfiles, int maxf);

/* the response for a reset, the schema (file in f), or an error */
struct jv *dj_reset_response(const struct djreq *r);
struct jv *dj_schema_response(const struct djreq *r, struct dfile *f);
struct jv *dj_error(const struct jv *id, const char *cmd, const char *code,
    const char *msg);

/* 1 if every response is ok */
int dj_all_ok(const struct jv *resp);

/*
 * The message for resp: a ```json block of it whole when it fits in a
 * Discord message, else (with the whole of it attached as response.json)
 * the response with its "text" cut, or a summary.
 */
void dj_format(const struct jv *resp, struct sbuf *msg, struct dfile *files,
    int *nfiles, int maxf);

void dj_free(struct djreq *r);

#endif
