/*
 * report.c - structured results for the JSON frontend (see report.h).
 *
 * While a JSON request runs (rep_on), commands add fields to a result
 * object, and the hooks in util.c (fail, note), nlp.c (questions) and
 * anim.c (files) turn errors, ambiguities and files into data instead of
 * text, prompts or \x1e directives.  jsonmode.c starts a request with
 * rep_request() and takes the pieces with rep_collect().
 */
#include "ucas.h"
#include "json.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int rep_on;

static struct {
	struct jv *results;	/* every result object of the request */
	struct jv *stack[32];	/* containers being filled; [0] the result */
	int sp;
	struct jv *conds, *checks, *proof, *files, *notes, *error;
	/* the last fail(): it is the error only if the statement ended so */
	char pend[512];
	unsigned pline, pcol;
	int pending;
	/* how questions are answered */
	const struct jv *assume, *session;
	int defaults;
	int fmode;		/* REP_FILES_* */
	char outdir[PATH_MAX];
} R;

void
rep_request(const struct jv *assume, const struct jv *session, int defaults,
    int fmode, const char *outdir)
{
	memset(&R, 0, sizeof R);
	R.results = jv_new(J_ARR);
	R.conds = jv_new(J_ARR);
	R.checks = jv_new(J_ARR);
	R.files = jv_new(J_ARR);
	R.notes = jv_new(J_ARR);
	R.assume = assume;
	R.session = session;
	R.defaults = defaults;
	R.fmode = fmode;
	snprintf(R.outdir, sizeof R.outdir, "%s", outdir ? outdir : ".");
	rep_on = 1;
}

/* the pieces, as one object; the request is over */
struct jv *
rep_collect(void)
{
	struct jv *o = jv_new(J_OBJ);

	jv_set(o, "results", R.results);
	jv_set(o, "notes", R.notes);
	jv_set(o, "conditions", R.conds);
	jv_set(o, "checks", R.checks);
	if (R.proof)
		jv_set(o, "proof", R.proof);
	jv_set(o, "files", R.files);
	if (R.error)
		jv_set(o, "error", R.error);
	rep_on = 0;
	memset(&R, 0, sizeof R);
	return o;
}

int
rep_has_error(void)
{
	return R.error != NULL;
}

/* jsonmode.c's own errors (bad requests ...) */
void
rep_error(const char *code, const char *msg)
{
	if (!rep_on || R.error)
		return;
	R.error = jv_new(J_OBJ);
	jv_set(R.error, "code", jv_str(code));
	jv_set(R.error, "message", jv_str(msg));
}

/* ---- the result object ---- */

static void
add(const char *key, struct jv *v)
{
	struct jv *c;

	if (!rep_on) {
		jv_free(v);
		return;
	}
	if (R.sp == 0)
		rep_begin(NULL);
	c = R.stack[R.sp - 1];
	if (c->t == J_ARR)
		jv_push(c, v);
	else
		jv_set(c, key ? key : "value", v);
}

void
rep_begin(const char *kind)
{
	struct jv *o;

	if (!rep_on)
		return;
	/* an empty object is reused; a later command gets a new one */
	if (R.sp > 0 && R.stack[0]->n == 0) {
		if (kind)
			jv_set(R.stack[0], "kind", jv_str(kind));
		R.sp = 1;
		return;
	}
	o = jv_new(J_OBJ);
	if (kind)
		jv_set(o, "kind", jv_str(kind));
	jv_push(R.results, o);
	R.stack[0] = o;
	R.sp = 1;
}

void
rep_str(const char *key, const char *s)
{
	if (rep_on)
		add(key, s ? jv_str(s) : jv_new(J_NULL));
}

void
rep_strf(const char *key, const char *fmt, ...)
{
	char buf[1024];
	va_list ap;

	if (!rep_on)
		return;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	add(key, jv_str(buf));
}

void
rep_num(const char *key, double v)
{
	if (rep_on)
		add(key, jv_num(v));
}

void
rep_int(const char *key, long long v)
{
	if (rep_on)
		add(key, jv_int(v));
}

void
rep_bool(const char *key, int b)
{
	if (rep_on)
		add(key, jv_bool(b));
}

void
rep_null(const char *key)
{
	if (rep_on)
		add(key, jv_new(J_NULL));
}

void
rep_exact(const char *key, double v, const char *exact)
{
	struct jv *o;

	if (!rep_on)
		return;
	o = jv_new(J_OBJ);
	jv_set(o, "value", jv_num(v));
	if (exact)
		jv_set(o, "exact", jv_str(exact));
	add(key, o);
}

void
rep_expr(const char *key, uint32_t e)
{
	if (rep_on)
		add(key, jv_str(node_str(e)));
}

static void
push(struct jv *v)
{
	if (R.sp < (int)(sizeof R.stack / sizeof R.stack[0]))
		R.stack[R.sp++] = v;
}

void
rep_list(const char *key)
{
	struct jv *v;

	if (!rep_on)
		return;
	v = jv_new(J_ARR);
	add(key, v);
	push(v);
}

void
rep_obj(const char *key)
{
	struct jv *v;

	if (!rep_on)
		return;
	v = jv_new(J_OBJ);
	add(key, v);
	push(v);
}

void
rep_end(void)
{
	if (rep_on && R.sp > 1)
		R.sp--;
}

void
rep_lines(const char *key, const char *text)
{
	struct jv *a;
	const char *p = text, *e, *s;

	if (!rep_on || text == NULL)
		return;
	a = jv_new(J_ARR);
	while (*p) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		for (s = p; s < e && isspace((unsigned char)*s); s++)
			;
		while (e > s && isspace((unsigned char)e[-1]))
			e--;
		if (e > s)
			jv_push(a, jv_strn(s, (size_t)(e - s)));
		p = strchr(p, '\n') ? strchr(p, '\n') + 1 : p + strlen(p);
	}
	add(key, a);
}

void
rep_cond(const char *fmt, ...)
{
	char buf[512];
	va_list ap;
	int i;

	if (!rep_on)
		return;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	for (i = 0; i < R.conds->n; i++)
		if (!strcmp(R.conds->v[i]->s, buf))
			return;
	jv_push(R.conds, jv_str(buf));
}

void
rep_check(const char *kind, long points, double max_diff, int passed,
    const char *text)
{
	struct jv *o;

	if (!rep_on)
		return;
	o = jv_new(J_OBJ);
	jv_set(o, "kind", jv_str(kind));
	if (points >= 0)
		jv_set(o, "points", jv_int(points));
	if (max_diff >= 0)
		jv_set(o, "max_diff", jv_num(max_diff));
	jv_set(o, "passed", jv_bool(passed));
	if (text)
		jv_set(o, "text", jv_str(text));
	jv_push(R.checks, o);
}

void
rep_proof(const char *kind, const char *text)
{
	if (!rep_on)
		return;
	jv_free(R.proof);
	R.proof = jv_new(J_OBJ);
	jv_set(R.proof, "kind", jv_str(kind));
	if (text)
		jv_set(R.proof, "text", jv_str(text));
}

void
rep_note(const char *kind, const char *text)
{
	struct jv *o;
	int i;

	if (!rep_on)
		return;
	for (i = 0; i < R.notes->n; i++)
		if (!strcmp(jv_gets(R.notes->v[i], "text"), text))
			return;
	o = jv_new(J_OBJ);
	jv_set(o, "kind", jv_str(kind));
	jv_set(o, "text", jv_str(text));
	jv_push(R.notes, o);
}

/* ---- errors ---- */

void
rep_fail(const char *msg, unsigned line, unsigned col)
{
	if (!rep_on)
		return;
	snprintf(R.pend, sizeof R.pend, "%s", msg);
	R.pline = line;
	R.pcol = col;
	R.pending = 1;
}

static const char *
classify(const char *m)
{
	static const struct {
		const char *w, *code;
	} t[] = {
		{"expected", "parse_error"}, {"unexpected", "parse_error"},
		{"unknown type", "parse_error"}, {"unterminated", "parse_error"},
		{"syntax", "parse_error"}, {"unknown character", "parse_error"},
		{"unknown function", "parse_error"},
		{"is not defined", "parse_error"}, {"undefined name", "parse_error"},
		{"unknown model", "parse_error"}, {"unknown basis", "parse_error"},
		{"outside the domain", "domain_error"},
		{"undefined at", "domain_error"}, {"not defined at", "domain_error"},
		{"division by zero", "domain_error"}, {"empty range", "domain_error"},
		{"no point of the domain", "domain_error"},
		{"not supported", "unsupported"}, {"only ", "unsupported"},
		{"at most", "unsupported"}, {"too many", "unsupported"},
		{"too large", "unsupported"}, {"too long", "unsupported"},
		{"cannot", "unsupported"}, {"can't", "unsupported"},
		{"refuse", "unsupported"}, {"not done", "unsupported"},
		{"needs ", "unsupported"}, {"does not fit", "unsupported"},
	};
	size_t i;

	for (i = 0; i < sizeof t / sizeof t[0]; i++)
		if (strstr(m, t[i].w))
			return t[i].code;
	return "failed";
}

void
rep_failed(void)
{
	struct jv *pos;

	if (!rep_on || !R.pending || R.error)
		return;
	R.error = jv_new(J_OBJ);
	jv_set(R.error, "code", jv_str(classify(R.pend)));
	jv_set(R.error, "message", jv_str(R.pend));
	if (R.pline) {
		pos = jv_new(J_OBJ);
		jv_set(pos, "line", jv_int(R.pline));
		jv_set(pos, "col", jv_int(R.pcol));
		jv_set(R.error, "pos", pos);
	}
	R.pending = 0;
}

/* ---- questions: answered from args.assume, or ambiguities ---- */

/* a stable key for a question asked without one */
static const char *
qkey(const char *key, const char *what)
{
	static const struct {
		const char *a, *b, *key;
	} t[] = {
		{"solve for which variable", "", "solve_for"},
		{"integrate ", "with respect to which variable", "wrt"},
		{"animate exec: no type", "", "anim_type"},
		{"", "has special points at the interval's ends", "keep_ends"},
		{"SPIR-V: ", "does not say its type", "spv_type"},
		{"SPIR-V: ", "its value", "spv_value"},
		{"", "'s parameter ", "i_param"},
		{"what does \"", "apply to", "scope"},
		{"", "is not a word I know", "spelling"},
		{"the ", "derivative of", "derivative_var"},
	};
	size_t i;

	if (key)
		return key;
	for (i = 0; i < sizeof t / sizeof t[0]; i++)
		if (!strncmp(what, t[i].a, strlen(t[i].a)) &&
		    strstr(what, t[i].b))
			return t[i].key;
	return "question";
}

static const struct jv *
assumed(const char *key, const char *what)
{
	const struct jv *v;

	if ((v = jv_get(R.assume, key)) || (v = jv_get(R.assume, what)))
		return v;
	if ((v = jv_get(R.session, key)) || (v = jv_get(R.session, what)))
		return v;
	return NULL;
}

/* the index of the option a value names, -1 if none */
static int
match_option(const struct jv *v, const char *const *opt, int nopt)
{
	const char *s;
	char *e;
	long n;
	int k, hit = -1;

	if (v->t == J_NUM) {
		n = (long)v->num;
		return n >= 1 && n <= nopt && n == v->num ? (int)n - 1 : -1;
	}
	if (v->t != J_STR || !v->s[0])
		return -1;
	s = v->s;
	n = strtol(s, &e, 10);
	if (*e == 0)
		return n >= 1 && n <= nopt ? (int)n - 1 : -1;
	for (k = 0; k < nopt; k++)
		if (!strcasecmp(opt[k], s))
			return k;
	/* a unique prefix, or the option's first word ("ln" for "ln (base e)") */
	for (k = 0; k < nopt; k++) {
		size_t l = strlen(s);
		if (!strncasecmp(opt[k], s, l) && (opt[k][l] == 0 ||
		    opt[k][l] == ' ' || opt[k][l] == '(' || opt[k][l] == ',')) {
			if (hit >= 0)
				return -1;
			hit = k;
		}
	}
	return hit;
}

static char badassume[256];

const char *
rep_answer(const char *key, const char *what, const char *const *opt,
    int nopt, int dflt)
{
	static char buf[16];
	const struct jv *v;
	const char *k = qkey(key, what);
	int i;

	badassume[0] = 0;
	if (!rep_on)
		return NULL;
	if ((v = assumed(k, what)) != NULL) {
		if ((i = match_option(v, opt, nopt)) < 0) {
			struct sbuf b = {0};
			jv_write(&b, v);
			snprintf(badassume, sizeof badassume, "assume.%s = %s is not "
			    "one of the readings", k, b.p);
			free(b.p);
			return NULL;
		}
		{
			char t[600];
			snprintf(t, sizeof t, "%s: %s (args.assume.%s)", what, opt[i],
			    k);
			rep_note("assumed", t);
		}
		snprintf(buf, sizeof buf, "%d", i + 1);
		return buf;
	}
	if (R.defaults) {
		char t[600];
		snprintf(t, sizeof t, "%s: the default, %s (args.defaults)", what,
		    opt[dflt]);
		rep_note("default", t);
		snprintf(buf, sizeof buf, "%d", dflt + 1);
		return buf;
	}
	return NULL;
}

static void
ambiguity(const char *key, const char *what, const char *const *opt,
    const char *const *how, int nopt, int dflt, const char *hint)
{
	struct jv *rd, *o;
	char m[700];
	int k;

	if (!rep_on || R.error)
		return;
	R.error = jv_new(J_OBJ);
	jv_set(R.error, "code", jv_str("ambiguous"));
	if (badassume[0])
		snprintf(m, sizeof m, "%s (%s)", what, badassume);
	else
		snprintf(m, sizeof m, "%s", what);
	jv_set(R.error, "message", jv_str(m));
	jv_set(R.error, "key", jv_str(key));
	rd = jv_new(J_ARR);
	for (k = 0; k < nopt; k++) {
		o = jv_new(J_OBJ);
		jv_set(o, "index", jv_int(k + 1));
		jv_set(o, "reading", jv_str(opt[k]));
		jv_set(o, "default", jv_bool(k == dflt));
		if (how && how[k])
			jv_set(o, "how", jv_str(how[k]));
		jv_push(rd, o);
	}
	jv_set(R.error, "readings", rd);
	snprintf(m, sizeof m, "repeat the request with args.assume.%s set to a "
	    "reading (or its index), or args.defaults = true to take the "
	    "default%s%s", key, hint ? "; or: " : "", hint ? hint : "");
	jv_set(R.error, "how", jv_str(m));
	badassume[0] = 0;
}

void
rep_ask(const char *key, const char *what, const char *const *opt, int nopt,
    int dflt, const char *hint)
{
	ambiguity(qkey(key, what), what, opt, NULL, nopt, dflt, hint);
}

void
rep_ask_text(const char *what)
{
	if (!rep_on || R.error)
		return;
	R.error = jv_new(J_OBJ);
	jv_set(R.error, "code", jv_str("needs_input"));
	jv_set(R.error, "message", jv_str(what));
	jv_set(R.error, "how", jv_str("repeat the request with args.answers = "
	    "[\"...\"], the missing details as they would be typed"));
}

/*
 * Notes that report a guess the text frontends make silently: in JSON
 * mode each is an ambiguity unless args.assume names the reading taken
 * (the first) or args.defaults is set.  The other readings say how to
 * get them: they need different input, not a flag.
 */
static const struct {
	const char *pat, *key;
	const char *opt[2], *how[2];
} guesses[] = {
	{" taken as complex (no x or y", "z", {"complex", "real"},
	    {"assume.z = \"complex\"", "annotate the variable, e.g. z: real"}},
	{" has no type: inferred ", "untyped", {"inferred", "real"},
	    {"assume.untyped = \"inferred\"", "give the type (params[].type, or "
	    "x: real in the text)"}},
	{" is not defined: read as a product", "juxtaposition",
	    {"product", "one name"}, {"assume.juxtaposition = \"product\", or "
	    "write the * yourself", "define the name first"}},
	{" (single-letter variables)", "juxtaposition", {"product", "one name"},
	    {"assume.juxtaposition = \"product\", or write the * yourself",
	    "define the name first, or annotate it (xy: real)"}},
	{"read as floor square root", "isqrt", {"isqrt", "real sqrt"},
	    {"assume.isqrt = \"isqrt\", or write isqrt(n)",
	    "write sqrt(real(n))"}},
	{"j is read as the imaginary unit", "j", {"unit", "variable"},
	    {"assume.j = \"unit\"", "use another name for the variable"}},
	{"is read as degrees", "angle", {"degrees", "radians"},
	    {"assume.angle = \"degrees\", or write deg", "write the angle in "
	    "radians with rad (or as a multiple of pi)"}},
	{" has free variables: defined as ", "free_vars",
	    {"in order of appearance", "explicit"},
	    {"assume.free_vars = \"in order of appearance\"", "give params"}},
	{"2e1 is read as 2*e1", "e_units", {"unit", "number"},
	    {"assume.e_units = \"unit\", or write 2*e1", "write 20 or 2.0e1"}},
};

void
rep_note_hook(const char *text)
{
	size_t i;
	const struct jv *v;

	if (!rep_on)
		return;
	for (i = 0; i < sizeof guesses / sizeof guesses[0]; i++) {
		if (!strstr(text, guesses[i].pat))
			continue;
		if ((v = assumed(guesses[i].key, text)) != NULL) {
			if (match_option(v, guesses[i].opt, 2) == 0) {
				rep_note("assumed", text);
				return;
			}
			snprintf(badassume, sizeof badassume, "assume.%s: the other "
			    "reading needs different input: %s", guesses[i].key,
			    guesses[i].how[1]);
		} else if (R.defaults) {
			rep_note("default", text);
			return;
		}
		ambiguity(guesses[i].key, text, guesses[i].opt, guesses[i].how, 2,
		    0, NULL);
		nlp_asked = 1;
		fail_quiet();
	}
}

/* ---- files ---- */

void
rep_file(const char *name, const char *mime, const char *data, size_t n)
{
	struct jv *o;
	char path[PATH_MAX], real[PATH_MAX];
	const char *base;
	FILE *fp;

	if (!rep_on)
		return;
	base = strrchr(name, '/') ? strrchr(name, '/') + 1 : name;
	o = jv_new(J_OBJ);
	jv_set(o, "name", jv_str(base));
	jv_set(o, "mime", jv_str(mime));
	jv_set(o, "bytes", jv_int((long long)n));
	if (R.fmode & REP_FILES_WRITE) {
		snprintf(path, sizeof path, "%s/%s", R.outdir, base);
		if ((fp = fopen(path, "wb")) == NULL ||
		    fwrite(data, 1, n, fp) != n) {
			if (fp)
				fclose(fp);
			jv_set(o, "error", jv_str(strerror(errno)));
		} else {
			fclose(fp);
			jv_set(o, "path", jv_str(realpath(path, real) ? real : path));
		}
	}
	if (R.fmode & REP_FILES_BASE64) {
		struct sbuf b = {0};
		b64_encode(&b, (const unsigned char *)data, n);
		jv_set(o, "base64", jv_str(sb_str(&b)));
		free(b.p);
	}
	jv_push(R.files, o);
}

/* anim.c's deliver: 1 when the file went into the response */
int
rep_deliver(const char *name, const char *mime, const char *data, size_t n,
    int nofile)
{
	if (!rep_on || nofile)
		return 0;
	if (R.fmode == 0) {
		printf("  (files: none) %s, %zu bytes, not written\n", name, n);
		return 1;
	}
	rep_file(name, mime, data, n);
	if (R.fmode & REP_FILES_WRITE)
		printf("  wrote %s\n", name);
	else
		printf("  attached %s (%zu bytes)\n", name, n);
	return 1;
}
