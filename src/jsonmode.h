/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * jsonmode.h - the JSON frontend's entry points for other frontends
 * (mcp.c), which drive the same commands, sessions and timeouts.
 */
#ifndef JSONMODE_H
#define JSONMODE_H

#include <stdio.h>

struct jv;
struct sbuf;

int json_setup(int argc, char **argv, int fmode);
FILE *json_out(void);			/* the real stdout */
char *json_read_line(void);		/* unbuffered, NULL at the end */
void json_serve(const struct jv *rq);	/* answered through respond */
extern void (*json_respond_hook)(struct jv *resp);

int json_ncmds(void);
const char *json_cmd_name(int k);
const char *json_cmd_desc(int k);
void json_cmd_params(int k, struct sbuf *b);
struct jv *json_cmd_schema(int k, int defs);
struct jv *json_response_schema(void);
struct jv *json_schema(void);

int mcp_main(int argc, char **argv, int i);	/* mcp.c */

#endif
