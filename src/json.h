/*
 * json.h - a small JSON value, parser and writer (for jsonmode.c and
 * report.c).  Values are trees; objects keep their keys in order.
 */
#ifndef JSON_H
#define JSON_H

#include <stddef.h>

struct sbuf;

enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ };

struct jv {
	int t;
	double num;
	char *s;            /* J_STR: the string; J_NUM: the literal as written */
	int n, cap;         /* J_ARR, J_OBJ */
	struct jv **v;
	char **k;           /* J_OBJ keys */
};

/* NULL on error: err gets a message and *pos the byte offset */
struct jv *jv_parse(const char *s, char *err, size_t errn, size_t *pos);
void jv_free(struct jv *v);

struct jv *jv_new(int t);
struct jv *jv_str(const char *s);
struct jv *jv_strn(const char *s, size_t n);
struct jv *jv_num(double d);         /* non-finite numbers become null */
struct jv *jv_int(long long i);
struct jv *jv_bool(int b);
void jv_push(struct jv *arr, struct jv *v);
void jv_set(struct jv *obj, const char *key, struct jv *v);  /* replaces */
struct jv *jv_get(const struct jv *obj, const char *key);
const char *jv_gets(const struct jv *obj, const char *key); /* string or NULL */

void jv_write(struct sbuf *b, const struct jv *v);   /* one line */
void json_quote(struct sbuf *b, const char *s);
void json_quoten(struct sbuf *b, const char *s, size_t n);
void json_num(struct sbuf *b, double d);    /* shortest round-trip form */
void b64_encode(struct sbuf *b, const unsigned char *p, size_t n);

#endif
