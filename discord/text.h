/*
 * Discord message text: the `$` prefix, code fences, and fitting output
 * into Discord's 2000-character messages.  No Concord types here, so
 * this is unit tested on its own (selftest.c).
 */
#ifndef UCAS_DISCORD_TEXT_H
#define UCAS_DISCORD_TEXT_H

#include <stddef.h>
#include <stdint.h>

#include "ucas.h"
#include "sup.h"

#define DC_LIMIT 2000	/* bytes per message (Discord counts characters) */
#define DC_MAXMSG 3	/* split into at most this many, else attach */
#define DC_FILENAME "ucas-output.txt"

enum { DC_IGNORE, DC_EVAL, DC_RESET };

/* classify a message; for DC_EVAL, code receives the input to run */
int dc_command(const char *content, struct sbuf *code, int *echo);

struct dc_reply {
	int n;
	char *msg[DC_MAXMSG];
	char *file;		/* attachment (raw output) or NULL */
	size_t filelen;
};

/* head: status line or NULL; out: engine output (n bytes) */
void dc_format(struct dc_reply *r, const char *head, const char *out,
    size_t n);
void dc_reply_free(struct dc_reply *r);

/* status line for a supervisor response, or NULL for a plain success */
const char *dc_status(uint32_t status, uint32_t arg,
    const struct sup_limits *l, char *buf, size_t sz);

#endif
