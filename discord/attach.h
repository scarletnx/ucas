/* a message's attachments and replied-to message as code blocks */
#ifndef UCAS_DISCORD_ATTACH_H
#define UCAS_DISCORD_ATTACH_H

#include <stdint.h>

struct discord_message;
struct sbuf;

extern int attach_allow_file;	/* --stdin tests: file:// URLs */

/*
 * Appends "\n```lang\n...\n```" for each usable attachment of m and, for
 * a reply to someone else's message, that message's ``` blocks (unless
 * m has its own, have_block) and attachments.  Lines for the user go to
 * notes.  Returns 1 if anything was added.
 */
int attach_gather(const struct discord_message *m, uint64_t self,
    int have_block, struct sbuf *blocks, struct sbuf *notes);

/* ```glsl blocks in the request, compiled to ```spv in place */
int attach_convert(struct sbuf *code, struct sbuf *notes);

#endif
