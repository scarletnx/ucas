/*
 * report.h - structured results for the JSON frontend (jsonmode.c).
 *
 * A command calls these next to the printf that shows the same thing.
 * Outside a JSON request (rep_on == 0) every call returns at once, so the
 * text frontends are unchanged.  Inside one, the calls build the
 * response's "result" object (and its conditions, checks, proof, files),
 * while the printed text is captured into "text".
 *
 *   rep_begin("integrate");            result.kind; marks it structured
 *   rep_expr("antiderivative", F);     an expression, in ucas syntax
 *   rep_str("result", text);           a string field
 *   rep_num("value", v);               a number
 *   rep_exact("value", v, "1/3");      {"value": 0.333.., "exact": "1/3"}
 *   rep_list("roots"); ... rep_end();  an array; inside it pass key NULL
 *   rep_obj(NULL); ... rep_end();      an object (in an array: key NULL)
 *   rep_cond("x != 0");                a condition the answer holds under
 *   rep_check("derivative", 401, 8.9e-16, 1, "...");
 *   rep_proof("symbolic", "...");
 *   rep_note("rule", "...");           (note() adds its notes itself)
 *
 * New commands adopt it with a rep_begin and a field per printed result.
 */
#ifndef REPORT_H
#define REPORT_H

#include <stddef.h>
#include <stdint.h>

extern int rep_on;		/* a JSON request is being answered */
extern int rep_small_files;	/* files travel inline (MCP): keep them small */

void rep_begin(const char *kind);
void rep_str(const char *key, const char *s);
void rep_strf(const char *key, const char *fmt, ...);
void rep_num(const char *key, double v);
void rep_int(const char *key, long long v);
void rep_bool(const char *key, int b);
void rep_null(const char *key);
void rep_exact(const char *key, double v, const char *exact);
void rep_expr(const char *key, uint32_t e);   /* a nid */
void rep_list(const char *key);
void rep_obj(const char *key);
void rep_end(void);
void rep_cond(const char *fmt, ...);
void rep_check(const char *kind, long points, double max_diff, int passed,
    const char *text);
void rep_proof(const char *kind, const char *text);
void rep_note(const char *kind, const char *text);
/* a file a command made: written, inlined as base64, or both (by request) */
void rep_file(const char *name, const char *mime, const char *data, size_t n);
/* the strings of a list of lines ("  = x + 1\n  check: ..."), trimmed */
void rep_lines(const char *key, const char *text);
/* the lines printed from here to the end of the command, as a list */
void rep_text_list(const char *key);
/* the lines printed from here on, read as labelled fields ("  label:
 * value" -> "label": value, "check: ..." -> checks, "how:" -> method) */
void rep_text_fields(void);
void rep_text_fill(const char *text);	/* jsonmode.c */

/* hooks for util.c, nlp.c, anim.c */
void rep_fail(const char *msg, unsigned line, unsigned col);
void rep_failed(void);   /* the statement ended in that fail() */
void rep_note_hook(const char *text);
const char *rep_answer(const char *key, const char *what,
    const char *const *opt, int nopt, int dflt);
void rep_ask(const char *key, const char *what, const char *const *opt,
    int nopt, int dflt, const char *hint);
void rep_ask_text(const char *what);
int rep_deliver(const char *name, const char *mime, const char *data,
    size_t n, int nofile);

/* jsonmode.c: one request */
struct jv;
#define REP_FILES_WRITE 1
#define REP_FILES_BASE64 2
void rep_request(const struct jv *assume, const struct jv *session,
    int defaults, int fmode, const char *outdir);
struct jv *rep_collect(void);
int rep_has_error(void);
void rep_error(const char *code, const char *msg);
int json_main(int argc, char **argv, int i);   /* jsonmode.c */

#endif
