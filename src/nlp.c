/*
 * nlp.c - English in, ucas statements out.
 *
 * A line of English is translated into ordinary ucas statements, which are
 * printed ("  ucas: ...") and then run, so every answer also shows the
 * syntax that produced it.  The translator is a recursive-descent parser
 * over words, with no dictionary beyond the tables below:
 *
 *   request     "what is the range of ...", "does ... overflow", "is A the
 *               same as B", "how accurate is G as an approximation of F",
 *               "let f of x be ...", "assume bytes are signed", "help"
 *   target      a function name, a pronoun (it, its, that) or an expression
 *   expression  English operators ("x squared plus one", "the sum of a and
 *               b", "the integer square root of a 4-bit unsigned x") mixed
 *               freely with ucas notation ("x^2 + 1", "isqrt(x)")
 *   clauses     "where x is an unsigned byte", "for x between 1 and 12",
 *               "when x is 3", "with respect to y", "using arm64 gates"
 *
 * The sentence is parsed twice.  The first, silent pass only learns the
 * variables' types (declarations usually come last: "the square root of x
 * where x is a u8"), so that the second pass can tell an integer square
 * root from a real one while it builds the expression.
 *
 * Some words have more than one reasonable reading: "byte" (u8 or i8),
 * "log" (ln, log2, log10), "int" (C's i32 or ucas' unbounded int), "the
 * square root of x plus 1" (sqrt(x) + 1 or sqrt(x + 1)).  When stdin and
 * stdout are both terminals the user is asked, and the meaning of a term is
 * kept for the session.  Otherwise the default reading is taken and a note
 * says how to choose the other one ("assume ..." or a rephrasing).
 */
#include "ucas.h"
#include "props.h"

#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

int nlp_interactive = -1;    /* -1: ask only when stdin and stdout are ttys */

/*
 * Deferred questions, for front ends with no terminal (the Discord bot).
 * A "\x1edefer" line turns them on.  A question is then printed after a
 * "\x1eask choice N" or "\x1eask text" marker line, and the run stops
 * there: nothing translated runs.  The front end asks the user and runs
 * the same text again, preceded by one "\x1eanswer TEXT" line per answer
 * so far.  Questions take queued answers in order, which works because
 * translation is deterministic.
 */
int nlp_defer;
int nlp_asked;
#define MAXANS 16
static char *answers[MAXANS];
static int nans, ansi;

static const char *
next_answer(void)
{
	return ansi < nans ? answers[ansi++] : NULL;
}

static int clarify(const char *key, const char *what,
    const char *const *opt, int nopt, int dflt, const char *hint);
static int interactive(void);
static int dry, quiet;

/* for commands that need a decision at run time (approx's endpoints) */
int
nlp_can_ask(void)
{
	return !dry && !quiet && (nlp_defer || interactive());
}

int
nlp_ask_choice(const char *what, const char *const *opt, int nopt, int dflt)
{
	return clarify(NULL, what, opt, nopt, dflt, NULL);
}

/* the same, kept for the session under key (asked once) */
int
nlp_ask_pref(const char *key, const char *what, const char *const *opt,
    int nopt, int dflt, const char *hint)
{
	return clarify(key, what, opt, nopt, dflt, hint);
}

/* a reply to a numbered question: a number, a prefix of an option, or
 * empty for the default; -1 if it is none of these */
static int
pick_option(const char *p, const char *const *opt, int nopt, int dflt)
{
	size_t n;
	int k, v;

	while (isspace((unsigned char)*p))
		p++;
	n = strcspn(p, "\r\n");
	while (n > 0 && isspace((unsigned char)p[n - 1]))
		n--;
	if (n == 0)
		return dflt;
	if (isdigit((unsigned char)*p)) {
		v = atoi(p) - 1;
		return v >= 0 && v < nopt ? v : -1;
	}
	for (k = 0; k < nopt; k++)
		if (!strncasecmp(opt[k], p, n))
			return k;
	return -1;
}

static jmp_buf nl_jmp;
static int dry;              /* first pass: learn declarations, say nothing */
static int quiet;            /* lookahead: never ask */
static int incond;           /* parsing a condition: "x is 3" compares */

static void nfail(const char *fmt, ...);

/* ---- arena: the strings of one sentence ---- */

static char arena[1 << 18];
static size_t arn;

static const char *
S(const char *fmt, ...)
{
	va_list ap;
	char *r = arena + arn;
	int n;

	va_start(ap, fmt);
	n = vsnprintf(r, sizeof arena - arn, fmt, ap);
	va_end(ap);
	if (n < 0 || (size_t)n + 1 >= sizeof arena - arn)
		nfail("that sentence is too long for me");
	arn += (size_t)n + 1;
	return r;
}

static void
nfail(const char *fmt, ...)
{
	va_list ap;

	if (!dry && !quiet) {
		fputs("  sorry: ", stdout);
		va_start(ap, fmt);
		vprintf(fmt, ap);
		va_end(ap);
		putchar('\n');
	}
	longjmp(nl_jmp, 1);
}

/* ---- session: remembered meanings and assumed variables ---- */

#define MAXPREF 16
static struct {
	char key[16];
	int val;
} prefs[MAXPREF];
static int nprefs;

static int
pref_get(const char *k)
{
	int i;

	for (i = 0; i < nprefs; i++)
		if (!strcmp(prefs[i].key, k))
			return prefs[i].val;
	return -1;
}

static void
pref_set(const char *k, int v)
{
	int i;

	for (i = 0; i < nprefs; i++)
		if (!strcmp(prefs[i].key, k)) {
			prefs[i].val = v;
			return;
		}
	if (nprefs < MAXPREF) {
		snprintf(prefs[nprefs].key, sizeof prefs[0].key, "%s", k);
		prefs[nprefs++].val = v;
	}
}

static int
interactive(void)
{
	if (nlp_interactive < 0)
		nlp_interactive = isatty(0) && isatty(1);
	return nlp_interactive;
}

/*
 * Which reading of an ambiguous phrase is meant?  key names a term whose
 * meaning is kept for the session (NULL: a one-off, such as the scope of
 * "of").  hint tells how to say it unambiguously.
 */
static int
clarify(const char *key, const char *what, const char *const *opt, int nopt,
    int dflt, const char *hint)
{
	char line[128], *p;
	int k, v, first;

	if (key && (v = pref_get(key)) >= 0)
		return v;
	if (dry || quiet)
		return dflt;
	if (nlp_defer) {
		const char *a;
		if (nlp_asked)
			return dflt;   /* one question at a time */
		if ((a = next_answer()) == NULL) {
			printf("\x1e" "ask choice %d\n? %s:\n", nopt, what);
			for (k = 0; k < nopt; k++)
				printf("  %d) %s%s\n", k + 1, opt[k],
				    k == dflt ? "  (default)" : "");
			nlp_asked = 1;
			return dflt;
		}
		if ((v = pick_option(a, opt, nopt, dflt)) < 0) {
			printf("  note: \"%s\" is not one of the %d choices for "
			    "%s: read as %s\n", a, nopt, what, opt[dflt]);
			v = dflt;
		}
		if (key) {
			pref_set(key, v);
			printf("  note: %s: %s (kept for this session)\n", what,
			    opt[v]);
		}
		return v;
	}
	if (!interactive()) {
		printf("  note: %s: read as %s", what, opt[dflt]);
		if (nopt > 1) {
			fputs(", not ", stdout);
			for (k = 0, first = 1; k < nopt; k++)
				if (k != dflt) {
					printf("%s%s", first ? "" : " or ", opt[k]);
					first = 0;
				}
		}
		if (hint)
			printf(" (%s)", hint);
		putchar('\n');
		if (key)
			pref_set(key, dflt);
		return dflt;
	}
	printf("  ? %s:\n", what);
	for (k = 0; k < nopt; k++)
		printf("      %d) %s\n", k + 1, opt[k]);
	for (;;) {
		printf("    which? [%d] ", dflt + 1);
		fflush(stdout);
		if (fgets(line, sizeof line, stdin) == NULL) {
			putchar('\n');
			v = dflt;
			break;
		}
		for (p = line; isspace((unsigned char)*p); p++)
			;
		if (*p == 0) {
			v = dflt;
			break;
		}
		v = atoi(p) - 1;
		if (isdigit((unsigned char)*p) && v >= 0 && v < nopt)
			break;
		p[strcspn(p, "\r\n")] = 0;
		for (k = 0; k < nopt; k++)
			if (!strncasecmp(opt[k], p, strlen(p)))
				break;
		if (k < nopt) {
			v = k;
			break;
		}
		printf("    (a number from 1 to %d)\n", nopt);
	}
	if (key) {
		pref_set(key, v);
		printf("    (kept for this session%s%s)\n", hint ? "; " : "",
		    hint ? hint : "");
	}
	return v;
}

/* ---- words ---- */

enum { W_END, W_WORD, W_NUM, W_SYM };
#define MAXW 320
struct word {
	uint8_t k;
	uint8_t sp;      /* whitespace before */
	uint8_t poss;    /* followed by 's */
	char s[40];      /* lower case; symbols "|" "||" "?" are BAR OROR QUES */
	char o[40];      /* as written */
};
static struct word W[MAXW + 4];
static int nw, wi;

static void
push(int k, const char *s, size_t n, int sp)
{
	struct word *w;
	size_t i;

	if (nw >= MAXW)
		nfail("that sentence is too long for me");
	w = &W[nw++];
	memset(w, 0, sizeof *w);
	w->k = k;
	w->sp = sp;
	if (n >= sizeof w->s)
		n = sizeof w->s - 1;
	memcpy(w->o, s, n);
	for (i = 0; i < n; i++)
		w->s[i] = tolower((unsigned char)s[i]);
	if (k == W_SYM) {
		if (!strcmp(w->s, "|"))
			strcpy(w->s, "BAR");
		else if (!strcmp(w->s, "||"))
			strcpy(w->s, "OROR");
		else if (!strcmp(w->s, "?"))
			strcpy(w->s, "QUES");
	}
}

static int
in_list(const char *const *l, const char *w)
{
	int i;

	for (i = 0; l[i]; i++)
		if (!strcmp(l[i], w))
			return 1;
	return 0;
}

/* a word, with contractions undone: what's, doesn't, f's */
static void
push_word(const char *s, size_t n, int sp)
{
	static const char *const is_s[] = {"what", "it", "that", "there",
		"here", "who", "where", "how", "which", NULL};
	const char *ap = memchr(s, '\'', n);
	char lw[40];
	size_t a, i;

	if (ap == NULL) {
		push(W_WORD, s, n, sp);
		return;
	}
	a = (size_t)(ap - s);
	for (i = 0; i < a && i < sizeof lw - 1; i++)
		lw[i] = tolower((unsigned char)s[i]);
	lw[i] = 0;
	if (n - a == 2 && tolower((unsigned char)ap[1]) == 's') {
		if (in_list(is_s, lw)) {
			push(W_WORD, s, a, sp);
			push(W_WORD, "is", 2, 1);
		} else if (!strcmp(lw, "let")) {
			push(W_WORD, s, a, sp);
			push(W_WORD, "us", 2, 1);
		} else {
			push(W_WORD, s, a, sp);
			W[nw - 1].poss = 1;
		}
	} else if (n - a == 2 && tolower((unsigned char)ap[1]) == 't' &&
	    a > 0 && tolower((unsigned char)s[a - 1]) == 'n') {
		if (!strcmp(lw, "can"))
			push(W_WORD, "can", 3, sp);
		else if (!strcmp(lw, "won"))
			push(W_WORD, "will", 4, sp);
		else
			push(W_WORD, s, a - 1, sp);
		push(W_WORD, "not", 3, 1);
	} else if (n - a == 3 && !strncasecmp(ap + 1, "re", 2)) {
		push(W_WORD, s, a, sp);
		push(W_WORD, "are", 3, 1);
	} else {
		push(W_WORD, s, a, sp);   /* I'd, I'm, we'll: the rest adds nothing */
	}
}

static void
tokenize(const char *s)
{
	static const struct {
		const char *u, *t;
		int k;
	} uni[] = {
		{"\xc2\xb2", "\xc2\xb2", W_SYM}, {"\xc2\xb3", "\xc2\xb3", W_SYM},
		{"\xe2\x88\x9a", "sqrt", W_WORD}, {"\xcf\x80", "pi", W_WORD},
		{"\xe2\x8a\x95", "xor", W_WORD}, {"\xc3\x97", "*", W_SYM},
		{"\xc2\xb7", "*", W_SYM}, {"\xc3\xb7", "/", W_SYM},
		{"\xe2\x89\xa4", "<=", W_SYM}, {"\xe2\x89\xa5", ">=", W_SYM},
		{"\xe2\x89\xa0", "!=", W_SYM}, {"\xe2\x88\x92", "-", W_SYM},
		{"\xc2\xb0", "deg", W_WORD}, {"\xe2\x88\xa0", "\xe2\x88\xa0", W_SYM},
		/* calc2: ∂ ∇ → ∞ */
		{"\xe2\x88\x82", "\xe2\x88\x82", W_SYM},
		{"\xe2\x88\x87", "\xe2\x88\x87", W_SYM},
		{"\xe2\x86\x92", "\xe2\x86\x92", W_SYM},
		{"\xe2\x88\x9e", "infinity", W_WORD},
		{NULL, NULL, 0}
	};
	static const char *const greeku[] = {"alpha", "beta", "gamma",
		"delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
		"lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma",
		"sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"};
	static const char *const sy[] = {"..", "<=", ">=", "==", "!=", "<<",
		">>", "&&", "||", "**", "+", "-", "*", "/", "^", "%", "&", "|",
		"~", "!", "<", ">", "=", "(", ")", "[", "]", ",", ":", ";", "?",
		".", "{", "}", NULL};
	size_t i = 0, j, n = strlen(s), k;
	int sp;

	nw = 0;
	while (i < n) {
		sp = 0;
		while (i < n && isspace((unsigned char)s[i])) {
			i++;
			sp = 1;
		}
		if (i >= n)
			break;
		if (isdigit((unsigned char)s[i]) || (s[i] == '.' && i + 1 < n &&
		    isdigit((unsigned char)s[i + 1]))) {
			j = i;
			if (s[j] == '0' && (s[j + 1] == 'x' || s[j + 1] == 'X')) {
				j += 2;
				while (isxdigit((unsigned char)s[j]))
					j++;
			} else {
				while (isdigit((unsigned char)s[j]))
					j++;
				if (s[j] == '.' && isdigit((unsigned char)s[j + 1])) {
					j++;
					while (isdigit((unsigned char)s[j]))
						j++;
				}
				if ((s[j] == 'e' || s[j] == 'E') &&
				    (isdigit((unsigned char)s[j + 1]) ||
				    ((s[j + 1] == '-' || s[j + 1] == '+') &&
				    isdigit((unsigned char)s[j + 2])))) {
					j += 2;
					while (isdigit((unsigned char)s[j]))
						j++;
				}
				if (s[j] == 'f' && !isalnum((unsigned char)s[j + 1]))
					j++;
			}
			push(W_NUM, s + i, j - i, sp);
			i = j;
			continue;
		}
		if (isalpha((unsigned char)s[i]) || s[i] == '_') {
			j = i;
			for (;;) {
				if (isalnum((unsigned char)s[j]) || s[j] == '_') {
					j++;
				} else if (s[j] == '\'' && isalpha((unsigned char)s[j + 1]) &&
				    isalpha((unsigned char)s[j - 1])) {
					j++;
				} else if (s[j] == '-' && j - i >= 3 &&
				    isalpha((unsigned char)s[j - 1]) &&
				    isalpha((unsigned char)s[j + 1]) &&
				    isalpha((unsigned char)s[j + 2])) {
					j++;   /* thirty-two, non-negative */
				} else
					break;
			}
			push_word(s + i, j - i, sp);
			i = j;
			continue;
		}
		/* lower-case Greek letters are names: θ is theta */
		if (((unsigned char)s[i] == 0xce || (unsigned char)s[i] == 0xcf) &&
		    i + 1 < n) {
			int cp = (((unsigned char)s[i] & 0x1f) << 6) |
			    ((unsigned char)s[i + 1] & 0x3f);
			if (cp >= 0x3b1 && cp <= 0x3c9) {
				push(W_WORD, greeku[cp - 0x3b1],
				    strlen(greeku[cp - 0x3b1]), sp);
				i += 2;
				continue;
			}
		}
		for (k = 0; uni[k].u; k++)
			if (!strncmp(s + i, uni[k].u, strlen(uni[k].u)))
				break;
		if (uni[k].u) {
			push(uni[k].k, uni[k].t, strlen(uni[k].t), sp);
			i += strlen(uni[k].u);
			continue;
		}
		for (k = 0; sy[k]; k++)
			if (!strncmp(s + i, sy[k], strlen(sy[k])))
				break;
		/* the period of an abbreviation: "crit. points" */
		if (sy[k] && !strcmp(sy[k], ".") && nw > 0 &&
		    W[nw - 1].k == W_WORD && !isdigit((unsigned char)s[i + 1])) {
			i++;
			continue;
		}
		if (sy[k]) {
			push(W_SYM, sy[k], strlen(sy[k]), sp);
			i += strlen(sy[k]);
			continue;
		}
		nfail("I can't read the character '%c'", s[i]);
	}
	for (k = 0; k < 4; k++)
		memset(&W[nw + k], 0, sizeof W[0]);
}

static const char *
words_text(int a, int b)
{
	struct sbuf sb = {0};
	const char *r;
	int i;

	for (i = a; i < b && i < nw; i++) {
		if (i > a && W[i].sp)
			sb_putc(&sb, ' ');
		sb_puts(&sb, W[i].o);
		if (W[i].poss)
			sb_puts(&sb, "'s");
	}
	r = S("%s", sb.p ? sb.p : "");
	free(sb.p);
	return r;
}

/*
 * Match a phrase: words separated by spaces, "a|b" alternatives, "?w"
 * optional.  Advances past the phrase on success.
 */
static int
mat(const char *pat)
{
	const char *p = pat, *e, *a, *b, *q;
	int save = wi, opt, ok;
	size_t len;

	while (*p) {
		while (*p == ' ')
			p++;
		if (!*p)
			break;
		for (e = p; *e && *e != ' '; e++)
			;
		q = p;
		opt = *q == '?' && q + 1 < e;
		if (opt)
			q++;
		ok = 0;
		if (W[wi].k != W_END) {
			len = strlen(W[wi].s);
			for (a = q; a < e; a = b + 1) {
				for (b = a; b < e && *b != '|'; b++)
					;
				if ((size_t)(b - a) == len && !strncmp(W[wi].s, a, len)) {
					ok = 1;
					break;
				}
			}
		}
		if (ok)
			wi++;
		else if (!opt) {
			wi = save;
			return 0;
		}
		p = e;
	}
	return 1;
}

static int
peek(const char *pat)
{
	int s = wi, r = mat(pat);

	wi = s;
	return r;
}

static int
issym(const char *s)
{
	return W[wi].k == W_SYM && !strcmp(W[wi].s, s);
}

static int
acc_sym(const char *s)
{
	if (issym(s)) {
		wi++;
		return 1;
	}
	return 0;
}

static int
isw(int i, const char *w)
{
	return W[i].k == W_WORD && !strcmp(W[i].s, w);
}

/* ---- functions known to ucas ---- */

static struct func *
user_fn(const char *name)
{
	int i;

	for (i = 0; i < nfuncs; i++)
		if (!funcs[i].anon && !strcmp(sym_str(funcs[i].name), name))
			return &funcs[i];
	if (!strcmp(name, "_"))
		for (i = 0; i < nfuncs; i++)
			if (funcs[i].anon && !strcmp(sym_str(funcs[i].name), "_"))
				return &funcs[i];
	return NULL;
}

static const char *const builtins[] = {"sin", "cos", "tan", "exp", "log",
	"ln", "sqrt", "isqrt", "abs", "floor", "ceil", "sign", "min", "max",
	"clz", "ctz", "popcount", "asuint", "asint", "asfloat", "bits",
	"barrier", "atan", "sec", "csc", "cot", "copysign", "fma",
	"asin", "acos", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh",
	"atanh", "arcsin", "arccos", "arctan",
	"w", "w0", "lambertw", "productlog", "wm1", "lambertwm1", NULL};

static int
builtin_call(int i)
{
	ty_t t;

	if (W[i].k != W_WORD || !(W[i + 1].k == W_SYM &&
	    !strcmp(W[i + 1].s, "(")))
		return 0;
	return in_list(builtins, W[i].s) || !strncmp(W[i].s, "atomic_", 7) ||
	    ty_parse(W[i].s, strlen(W[i].s), &t);
}

/* "it", "its", "that": the function the last question was about */
static char last_fn[64];
static int last_nfuncs;

static const char *
last_name(void)
{
	int i;

	if (*last_fn && nfuncs == last_nfuncs && user_fn(last_fn))
		return last_fn;
	for (i = nfuncs - 1; i >= 0; i--)
		if (!funcs[i].anon)
			return sym_str(funcs[i].name);
	if (*last_fn && user_fn(last_fn))
		return last_fn;
	return NULL;
}

/* ---- variables ---- */

#define MAXV 16
#define A_POS 1
#define A_NEG 2
#define A_NONNEG 4
#define A_NONZERO 8
#define A_EVEN 16
#define A_ODD 32
struct var {
	char name[32];
	char ty[24];      /* ucas type, "" when untyped */
	int ity;          /* 1 integer, 2 real or IEEE, 0 unknown */
	char lo[48], hi[48];
	int bc;           /* bit 1, 2: lo, hi is a constant expression */
	char cond[4][96];
	int ncond;
	int adj;
	char val[96];     /* bound to a value: substituted */
	int order;        /* first use, from 1; 0 unused */
	int uses;         /* bit per target */
	int decl;         /* declared in this sentence */
};
static struct var V[MAXV], SV[MAXV], P1[MAXV];
static int nv, nsv, np1, norder, cur_mask = 1;

static struct var *
var_find(const char *name)
{
	int i;

	for (i = 0; i < nv; i++)
		if (!strcmp(V[i].name, name))
			return &V[i];
	return NULL;
}

static struct var *
var_get(const char *name)
{
	struct var *v = var_find(name);

	if (v)
		return v;
	if (nv == MAXV)
		nfail("too many variables");
	v = &V[nv++];
	memset(v, 0, sizeof *v);
	snprintf(v->name, sizeof v->name, "%s", name);
	return v;
}

static int
ty_ity(const char *t)
{
	if (!*t)
		return 0;
	if (!strcmp(t, "real") || t[0] == 'f')
		return 2;
	return 1;
}

/* the value range of a sized integer type */
static int
type_limits(const char *t, int64_t *lo, int64_t *hi)
{
	int bits;

	if (!strcmp(t, "bool")) {
		*lo = 0;
		*hi = 1;
		return 1;
	}
	if ((t[0] != 'u' && t[0] != 'i') || !isdigit((unsigned char)t[1]))
		return 0;
	bits = atoi(t + 1);
	if (bits < 1 || bits > 64)
		return 0;
	if (t[0] == 'u') {
		*lo = 0;
		*hi = bits == 64 ? INT64_MAX : (int64_t)((1ULL << bits) - 1);
	} else {
		*hi = bits == 64 ? INT64_MAX : (int64_t)((1ULL << (bits - 1)) - 1);
		*lo = -*hi - 1;
	}
	return 1;
}

/* ---- expressions ---- */

enum {
	P_TERN = 1, P_OR, P_AND, P_CMP, P_BOR, P_BXOR, P_BAND, P_SHIFT, P_ADD,
	P_MUL, P_UNARY, P_POW, P_ATOM = 20
};

struct ex {
	const char *s;
	int prec;
	int ity;         /* 0 no variables, 1 integer, 2 real, 3 mixed/unknown */
	int isbool;
	int lit;         /* numeric literal */
	int closed;      /* delimited: parentheses, "the quantity", a call */
	int pword;       /* x squared: a word-form power of pb */
	const char *pb, *pe;
	int pbprec;
	int hv;          /* a constant with value v: pi/2, 2^10 */
	double v;
};

static int
icomb(int a, int b)
{
	if (!a)
		return b;
	if (!b)
		return a;
	return a == b ? a : 3;
}

static struct ex
mkatom(const char *s, int ity, int lit)
{
	struct ex e;

	memset(&e, 0, sizeof e);
	e.s = s;
	e.prec = P_ATOM;
	e.ity = ity;
	e.lit = lit;
	return e;
}

static const char *
paren(struct ex e, int need, int strict)
{
	if (e.prec < need || (strict && e.prec == need))
		return S("(%s)", e.s);
	return e.s;
}

static struct ex
mkbin(struct ex a, const char *op, struct ex b, int prec)
{
	struct ex r;
	int ra = prec == P_POW;
	const char *l = paren(a, prec, ra || prec == P_CMP);
	const char *rr = paren(b, prec, !ra || prec == P_CMP);

	if (rr == b.s && b.s[0] == '-')
		rr = S("(%s)", b.s);
	if (ra && l == a.s && a.s[0] == '-')
		l = S("(%s)", a.s);
	memset(&r, 0, sizeof r);
	if (ra)
		r.s = S("%s^%s", l, rr);
	else if (!strcmp(op, "*") || !strcmp(op, "/"))
		r.s = S("%s%s%s", l, op, rr);
	else
		r.s = S("%s %s %s", l, op, rr);
	r.prec = prec;
	r.ity = icomb(a.ity, b.ity);
	r.isbool = prec == P_CMP || ((prec == P_AND || prec == P_OR) &&
	    a.isbool && b.isbool);
	r.lit = a.lit && b.lit;
	if (a.hv && b.hv) {
		r.hv = 1;
		if (!strcmp(op, "+"))
			r.v = a.v + b.v;
		else if (!strcmp(op, "-"))
			r.v = a.v - b.v;
		else if (!strcmp(op, "*"))
			r.v = a.v * b.v;
		else if (!strcmp(op, "/") && b.v != 0)
			r.v = a.v / b.v;
		else if (ra)
			r.v = pow(a.v, b.v);
		else
			r.hv = 0;
	}
	return r;
}

static struct ex
mkcall(const char *fn, struct ex a)
{
	static const struct {
		const char *n;
		double (*f)(double);
	} cf[] = {{"sin", sin}, {"cos", cos}, {"tan", tan}, {"exp", exp},
		{"log", log}, {"sqrt", sqrt}, {"abs", fabs}, {"floor", floor},
		{"ceil", ceil}, {"real", NULL}, {NULL, NULL}};
	struct ex r = mkatom(S("%s(%s)", fn, a.s), a.ity, a.lit);
	int k;

	r.closed = 1;
	for (k = 0; a.hv && cf[k].n; k++)
		if (!strcmp(cf[k].n, fn)) {
			r.hv = 1;
			r.v = cf[k].f ? cf[k].f(a.v) : a.v;
		}
	return r;
}

static struct ex
mkun(const char *op, struct ex a)
{
	struct ex r;
	const char *x = paren(a, P_UNARY, 0);

	if (x == a.s && (a.s[0] == '-' || a.s[0] == '!' || a.s[0] == '~'))
		x = S("(%s)", a.s);
	r = mkatom(S("%s%s", op, x), a.ity, a.lit);
	r.prec = P_UNARY;
	r.isbool = op[0] == '!';
	if (a.hv && op[0] == '-') {
		r.hv = 1;
		r.v = -a.v;
	}
	return r;
}

static struct ex
num(const char *s)
{
	struct ex e = mkatom(s, 0, 1);
	char *end;

	if (!strcmp(s, "pi")) {
		e.hv = 1;
		e.v = M_PI;
	} else if (!strcmp(s, "e")) {
		e.hv = 1;
		e.v = M_E;
	} else {
		e.v = strtod(s, &end);
		e.hv = *end == 0 || (end[0] == 'f' && end[1] == 0);
		if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
			e.v = (double)strtoull(s, NULL, 16), e.hv = 1;
	}
	return e;
}

static struct ex
mkpow(struct ex b, struct ex e, int word)
{
	struct ex r = mkbin(b, "^", e, P_POW);

	if (word && !b.closed) {
		r.pword = 1;
		r.pb = b.s;
		r.pbprec = b.prec;
		r.pe = e.s;
	}
	return r;
}

static struct ex parse_expr(int minprec);

/* ---- numbers in words ---- */

static const char *const ones[] = {"zero", "one", "two", "three", "four",
	"five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
	"thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
	"nineteen", NULL};
static const char *const tensw[] = {"twenty", "thirty", "forty", "fifty",
	"sixty", "seventy", "eighty", "ninety", NULL};
static const char *const ordw[] = {"zeroth", "first", "second", "third",
	"fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
	"eleventh", "twelfth", NULL};

static int
lookup(const char *const *l, const char *w)
{
	int i;

	for (i = 0; l[i]; i++)
		if (!strcmp(l[i], w))
			return i;
	return -1;
}

static int
numword_val(const char *w, int64_t *v)
{
	const char *h = strchr(w, '-');
	char t[24];
	int a, b;

	if (h) {
		snprintf(t, sizeof t, "%.*s", (int)(h - w), w);
		if ((a = lookup(tensw, t)) >= 0 && (b = lookup(ones, h + 1)) >= 1 &&
		    b <= 9) {
			*v = (a + 2) * 10 + b;
			return 1;
		}
		return 0;
	}
	if ((a = lookup(ones, w)) >= 0) {
		*v = a;
		return 1;
	}
	if ((a = lookup(tensw, w)) >= 0) {
		*v = (a + 2) * 10;
		return 1;
	}
	return 0;
}

static int
mult_word(const char *w, int64_t *m)
{
	if (!strcmp(w, "thousand"))
		*m = 1000;
	else if (!strcmp(w, "million"))
		*m = 1000000;
	else if (!strcmp(w, "billion"))
		*m = 1000000000;
	else
		return 0;
	return 1;
}

/* "two hundred fifty-six", "a thousand" */
static int
number_words(int64_t *out)
{
	int s = wi, any = 0, small = 0;
	int64_t total = 0, cur = 0, v, m;

	for (;;) {
		const char *w = W[wi].s;
		if (W[wi].k != W_WORD)
			break;
		if (numword_val(w, &v) && !(small && v < 10 && cur % 10)) {
			if (any && cur % 100 && cur % 100 < 20)
				break;   /* "three four": two numbers */
			cur += v;
			any = small = 1;
			wi++;
			continue;
		}
		if (!any && !strcmp(w, "a") && (isw(wi + 1, "hundred") ||
		    mult_word(W[wi + 1].s, &m))) {
			cur = 1;
			any = 1;
			wi++;
			continue;
		}
		if (any && !strcmp(w, "hundred")) {
			cur = (cur ? cur : 1) * 100;
			small = 0;
			wi++;
			continue;
		}
		if (any && mult_word(w, &m)) {
			total += (cur ? cur : 1) * m;
			cur = 0;
			small = 0;
			wi++;
			continue;
		}
		if (any && !strcmp(w, "and") && (isw(wi - 1, "hundred") ||
		    isw(wi - 1, "thousand")) && numword_val(W[wi + 1].s, &v)) {
			wi++;
			continue;
		}
		break;
	}
	if (!any) {
		wi = s;
		return 0;
	}
	*out = total + cur;
	return 1;
}

/* "half", "thirds": the denominator */
static int
fraction_word(int i)
{
	static const char *const fr[] = {"half", "halves", "third", "thirds",
		"quarter", "quarters", "fourth", "fourths", "fifth", "fifths",
		"sixth", "sixths", "eighth", "eighths", "tenth", "tenths", NULL};
	static const int den[] = {2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 8, 8,
		10, 10};
	int k;

	if (W[i].k != W_WORD || (k = lookup(fr, W[i].s)) < 0)
		return 0;
	return den[k];
}

/* an exponent written as an ordinal: fourth, 4th, nth */
static int
ordinal(struct ex *e)
{
	int k;

	if (W[wi].k == W_WORD && (k = lookup(ordw, W[wi].s)) >= 2) {
		wi++;
		*e = num(S("%d", k));
		return 1;
	}
	if (W[wi].k == W_NUM && !W[wi + 1].sp && (isw(wi + 1, "th") ||
	    isw(wi + 1, "st") || isw(wi + 1, "nd") || isw(wi + 1, "rd"))) {
		*e = num(S("%s", W[wi].o));
		wi += 2;
		return 1;
	}
	if (isw(wi, "nth")) {
		wi++;
		*e = mkatom("n", 3, 0);
		return 2;
	}
	return 0;
}

/* ---- types ---- */

struct decl {
	char ty[24];
	int ity;
	char lo[48], hi[48];
	int bc;
	char cond[4][96];   /* "%s" stands for the variable */
	int ncond;
	int adj;
	char val[96];
	int any;
};

static void
decl_type(struct decl *d, const char *t)
{
	snprintf(d->ty, sizeof d->ty, "%s", t);
	d->ity = ty_ity(t);
	d->any = 1;
}

static int
sign_of(int sgn, const char *key, const char *what, const char *uopt,
    const char *sopt, const char *hint)
{
	const char *opt[2];

	if (sgn >= 0)
		return sgn;
	opt[0] = uopt;
	opt[1] = sopt;
	return clarify(key, what, opt, 2, 0, hint);
}

/*
 * A type: "unsigned 8-bit integer", "byte", "u8", "uint8_t", "double",
 * "real number", "digit".  Asks when the signedness or width is unclear.
 */
static int
parse_type(struct decl *d)
{
	int s = wi, sgn = -1, bits = 0, k;
	int64_t v;
	ty_t t;
	const char *w;

	if (mat("unsigned"))
		sgn = 0;
	else if (mat("signed"))
		sgn = 1;
	if (W[wi].k == W_NUM && (isw(wi + 1, "bit") || isw(wi + 1, "bits") ||
	    (W[wi + 1].k == W_SYM && !strcmp(W[wi + 1].s, "-") &&
	    (isw(wi + 2, "bit") || isw(wi + 2, "bits"))))) {
		bits = atoi(W[wi].s);
		wi += isw(wi + 1, "bit") || isw(wi + 1, "bits") ? 2 : 3;
	} else if (W[wi].k == W_WORD && (w = strstr(W[wi].s, "-bit")) != NULL &&
	    (!strcmp(w, "-bit") || !strcmp(w, "-bits"))) {
		char t2[24];
		snprintf(t2, sizeof t2, "%.*s", (int)(w - W[wi].s), W[wi].s);
		if (numword_val(t2, &v)) {
			bits = (int)v;
			wi++;
		}
	}
	if (sgn < 0) {
		if (mat("unsigned"))
			sgn = 0;
		else if (mat("signed"))
			sgn = 1;
	}
	if (bits) {
		if (mat("float|floats|floating-point|fp") ||
		    mat("floating point ?number|numbers|value|values")) {
			if (bits != 32 && bits != 64)
				nfail("ucas has 32- and 64-bit floats, not %d-bit", bits);
			decl_type(d, bits == 32 ? "f32" : "f64");
			return 1;
		}
		mat("?integer|integers|int|ints|number|numbers|value|values|word|words|"
		    "quantity|quantities|variable|variables|input|inputs");
		if (bits < 1 || bits > 64)
			nfail("integers have 1 to 64 bits, not %d", bits);
		sgn = sign_of(sgn, "nbit", S("\"%d-bit\" without signed or unsigned",
		    bits), S("unsigned, u%d", bits), S("signed, i%d", bits),
		    "say \"unsigned\" or \"signed\", or \"assume N-bit numbers are "
		    "signed\"");
		decl_type(d, S("%c%d", sgn ? 'i' : 'u', bits));
		return 1;
	}
	if (W[wi].k != W_WORD) {
		if (sgn >= 0) {
			decl_type(d, sgn ? "i32" : "u32");
			return 1;
		}
		wi = s;
		return 0;
	}
	w = W[wi].s;
	if (mat("byte|bytes")) {
		sgn = sign_of(sgn, "byte", "\"byte\" without signed or unsigned",
		    "unsigned, u8", "signed, i8",
		    "say \"unsigned byte\", or \"assume bytes are signed\"");
		decl_type(d, sgn ? "i8" : "u8");
	} else if (mat("octet|octets")) {
		decl_type(d, "u8");
	} else if (mat("char|chars|character|characters")) {
		if (sgn < 0) {
			static const char *const o[] = {"signed, i8 (as on x86)",
				"unsigned, u8 (as on ARM)"};
			sgn = clarify("char", "\"char\" without signed or unsigned",
			    o, 2, 0, "say \"signed char\", or \"assume chars are "
			    "unsigned\"") ? 0 : 1;
		}
		decl_type(d, sgn ? "i8" : "u8");
	} else if (mat("short|shorts")) {
		mat("?int|integer|ints|integers");
		decl_type(d, sgn == 0 ? "u16" : "i16");
	} else if (mat("long|longs")) {
		mat("?long|longs");
		mat("?int|integer|ints|integers");
		decl_type(d, sgn == 0 ? "u64" : "i64");
	} else if (mat("size_t|uintptr_t")) {
		decl_type(d, "u64");
	} else if (mat("ssize_t|ptrdiff_t|intptr_t|off_t")) {
		decl_type(d, "i64");
	} else if (mat("int|ints")) {
		if (sgn >= 0) {
			decl_type(d, sgn ? "i32" : "u32");
		} else {
			static const char *const o[] = {"a C int, i32",
				"an unbounded integer, int"};
			k = clarify("int", "\"int\"", o, 2, 0,
			    "say \"32-bit int\" or \"unbounded integer\", or "
			    "\"assume int means unbounded\"");
			decl_type(d, k ? "int" : "i32");
		}
	} else if (mat("integer|integers")) {
		if (sgn >= 0) {
			decl_type(d, sgn ? "i32" : "u32");
		} else {
			static const char *const o[] = {"an unbounded integer, int",
				"a C int, i32"};
			k = clarify("integer", "\"integer\" without a width", o, 2, 0,
			    "say \"32-bit integer\", or \"assume integers are 32-bit\"");
			decl_type(d, k ? "i32" : "int");
		}
	} else if (mat("word|words")) {
		static const char *const o[] = {"16 bits (x86 WORD)",
			"32 bits (ARM word)", "64 bits (machine word)"};
		static const int wb[] = {16, 32, 64};
		k = clarify("word", "\"word\"", o, 3, 1,
		    "say \"16-bit\", \"32-bit\" or \"64-bit\", or \"assume words "
		    "are 64-bit\"");
		decl_type(d, S("%c%d", sgn == 1 ? 'i' : 'u', wb[k]));
	} else if (mat("dword|dwords|doubleword|doublewords")) {
		decl_type(d, sgn == 1 ? "i32" : "u32");
	} else if (mat("qword|qwords|quadword|quadwords")) {
		decl_type(d, sgn == 1 ? "i64" : "u64");
	} else if (mat("nibble|nibbles|nybble|nybbles")) {
		decl_type(d, sgn == 1 ? "i4" : "u4");
	} else if (mat("bit|bits")) {
		decl_type(d, "u1");
	} else if (mat("bool|bools|boolean|booleans|flag|flags")) {
		decl_type(d, "bool");
	} else if (mat("float|floats|single|singles|single-precision")) {
		mat("?precision");
		mat("?float|floats|number|numbers|value|values");
		decl_type(d, "f32");
	} else if (mat("double|doubles|double-precision")) {
		mat("?precision");
		mat("?float|floats|number|numbers|value|values");
		decl_type(d, "f64");
	} else if (mat("real|reals")) {
		mat("?number|numbers|value|values");
		decl_type(d, "real");
	} else if (mat("natural|whole number|numbers")) {
		decl_type(d, "int");
		strcpy(d->lo, "0");
	} else if (mat("?decimal digit|digits")) {
		decl_type(d, "int");
		strcpy(d->lo, "0");
		strcpy(d->hi, "9");
	} else if (mat("hex|hexadecimal digit|digits")) {
		decl_type(d, "int");
		strcpy(d->lo, "0");
		strcpy(d->hi, "15");
	} else if (mat("octal digit|digits")) {
		decl_type(d, "int");
		strcpy(d->lo, "0");
		strcpy(d->hi, "7");
	} else if (mat("probability|probabilities")) {
		decl_type(d, "real");
		strcpy(d->lo, "0");
		strcpy(d->hi, "1");
	} else if (mat("percentage|percentages")) {
		decl_type(d, "real");
		strcpy(d->lo, "0");
		strcpy(d->hi, "100");
	} else if (mat("angle|angles")) {
		decl_type(d, "real");
	} else if (mat("number|numbers|value|values|quantity|quantities")) {
		d->any = 1;   /* untyped: ucas decides */
	} else if (ty_parse(W[wi].s, strlen(W[wi].s), &t) &&
	    strcmp(w, "uint") && strcmp(w, "sint")) {
		wi++;
		decl_type(d, ty_str(t));
	} else if (sgn >= 0) {
		decl_type(d, sgn ? "i32" : "u32");
	} else {
		wi = s;
		return 0;
	}
	return 1;
}

static int
peek_type(void)
{
	struct decl d;
	int s = wi, r;

	memset(&d, 0, sizeof d);
	quiet++;
	mat("?a|an");
	r = parse_type(&d) && d.ty[0];
	quiet--;
	wi = s;
	return r;
}

/* ---- variable names ---- */

static const char *const reserved[] = {
	"a", "an", "the", "of", "is", "to", "by", "in", "as", "at", "or", "be",
	"if", "it", "on", "we", "me", "my", "no", "so", "do", "up", "us", "vs",
	"and", "not", "for", "are", "its", "all", "any", "how", "who", "why",
	"what", "when", "where", "which", "with", "that", "this", "than",
	"then", "else", "from", "over", "into", "onto", "does", "can", "will",
	"would", "could", "should", "might", "may", "must", "each", "every",
	"some", "both", "plus", "minus", "times", "squared", "cubed", "half",
	"twice", "value", "values", "number", "numbers", "between", "equal",
	"equals", "less", "more", "greater", "smaller", "larger", "bigger",
	"sum", "product", "difference", "quotient", "ratio", "power", "root",
	"square", "cube", "divided", "multiplied", "mod", "modulo", "xor",
	"bitwise", "shifted", "left", "right", "true", "false", "pi",
	"range", "domain", "derivative", "function", "expression", "formula",
	"please", "pls", "plz", "also", "only", "just", "always", "ever", "never", "same",
	"defined", "undefined", "overflow", "wrap", "using", "via", "where",
	"given", "assuming", "such", "type", "unsigned", "signed", "byte",
	"bytes", "integer", "integers", "float", "double", "real", "bits",
	"bit", "otherwise", "quantity", "ones", "zeros", "zeroes", "leading",
	"trailing", "negative", "positive", "absolute", "log", "logarithm",
	"natural", "sine", "cosine", "tangent", "exponential", "floor",
	"ceiling", "sign", "average", "mean", "minimum", "maximum", "min",
	"max", "remainder", "per", "percent", "variable", "let", "define",
	"set", "call", "name", "show", "find", "give", "tell", "compute",
	"calculate", "evaluate", "there", "here", "these", "those", "has",
	"have", "had", "was", "were", "been", "being", "get", "gets", "up",
	"down", "out", "about", "like", "approximately", "roughly", NULL
};

static int
reserved_word(const char *w)
{
	return in_list(reserved, w) || in_list(ones, w) || in_list(tensw, w) ||
	    in_list(ordw, w);
}

/* Greek letter names are variables: theta, alpha (pi is the constant) */
static const char *const greek[] = {"alpha", "beta", "gamma", "delta",
	"epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu",
	"nu", "xi", "omicron", "rho", "sigma", "tau", "upsilon", "phi",
	"chi", "psi", "omega", "varphi", "vartheta", NULL};

static const char *known_names[32];
static int nknown;

static int
known_name(const char *w)
{
	int i;

	for (i = 0; i < nknown; i++)
		if (!strcmp(known_names[i], w))
			return 1;
	return 0;
}

/* can word i name a variable? */
static int
varlike(int i)
{
	const char *w = W[i].s, *p;

	if (W[i].k != W_WORD)
		return 0;
	if (var_find(W[i].o) || known_name(W[i].o) || in_list(greek, w))
		return 1;
	if (reserved_word(w) || user_fn(W[i].o) || in_list(builtins, w))
		return 0;
	if (strlen(w) <= 2)
		return 1;
	for (p = w; *p; p++)
		if (isdigit((unsigned char)*p) || *p == '_')
			return 1;
	return 0;
}

/* ---- the request being built ---- */

enum {
	C_NONE, C_EVAL, C_SHOW, C_TABLE, C_RANGE, C_DOMAIN, C_WRAP, C_CRIT,
	C_DIFF, C_EQUIV, C_GATES, C_FIND, C_FIT, C_ERROR, C_COST, C_DEF,
	C_RAW, C_PROP, C_APPROX, C_INTEG, C_CLOSED, C_SCHEME, C_CONT, C_DSOLVE,
	C_PCMD, C_PLOT
};
enum { PR_COMM, PR_ASSOC, PR_ADD, PR_IDEM, PR_ODD, PR_EVEN, PR_ODDV,
	PR_EVENV };

struct tgt {
	int has, named, wrapped;
	char fn[64];
	struct ex e;
};

static struct {
	int cmd;
	struct tgt a, b;
	char basis[16];
	int emit_c, maxc, count, metric;
	char wrt[32], model[32], defname[64];
	char dparam[MAXP][32];
	int ndparam, dexplicit;
	const char *scond[8];
	int nscond;
	const char *raw[4];
	int nraw;
	const char *why;      /* a note printed before the answer */
	int nocrit;           /* range without critical points */
	int interval;         /* an implicit variable over an interval */
	const char *ilo, *ihi; /* "on [a, b]" for the only variable */
	int prop;             /* C_PROP: which property */
	char degree[32], aty[8];
	char acc[32];          /* approx: "error 1e-6" or "ulp 2" */
	int ends;              /* approx: 1 keep endpoints, -1 free */
	char method[16];       /* approx: cheb, econ, minimax */
	int crit;             /* critical points asked for explicitly */
	int steps;            /* dsolve: output steps */
	char ode[32];         /* dsolve: the unknown */
	const char *pcmd;     /* C_PCMD: monotonic, period, ... (props.c) */
	/* plot (plot.c): the functions, the command, raw options */
	struct tgt pl[6];
	int npl, pmode;
	char popt[256];
	const char *pat;      /* neighbourhood at */
} Q;

/* a question the sentence leaves open; asked on a terminal (nlp_run) */
static const char *need_more;
static int no_more_prompts;
static int spell_quiet;      /* a retranslation: notes were given */

static struct sbuf out;

static void
emit(const char *line)
{
	sb_puts(&out, line);
	sb_putc(&out, '\n');
}

/* ---- expressions in words ---- */

struct bop {
	const char *pat, *op;
	int prec, kind;
};
enum { B_NORM, B_SWAP, B_BETWEEN, B_IN, B_IF, B_CAST, B_AND, B_OR, B_IS,
	B_OVER };

static const struct bop bops[] = {
	{"is greater than or equal to", ">=", P_CMP, B_NORM},
	{"greater than or equal to", ">=", P_CMP, B_NORM},
	{"is at least", ">=", P_CMP, B_NORM},
	{"is no less than", ">=", P_CMP, B_NORM},
	{"is not less than", ">=", P_CMP, B_NORM},
	{">=", ">=", P_CMP, B_NORM},
	{"is less than or equal to", "<=", P_CMP, B_NORM},
	{"less than or equal to", "<=", P_CMP, B_NORM},
	{"is at most", "<=", P_CMP, B_NORM},
	{"is no more than", "<=", P_CMP, B_NORM},
	{"is not more than", "<=", P_CMP, B_NORM},
	{"is not greater than", "<=", P_CMP, B_NORM},
	{"<=", "<=", P_CMP, B_NORM},
	{"is greater|more|bigger|larger|higher than", ">", P_CMP, B_NORM},
	{"is above", ">", P_CMP, B_NORM},
	{"exceeds", ">", P_CMP, B_NORM},
	{"greater|more|bigger|larger than", ">", P_CMP, B_NORM},
	{">", ">", P_CMP, B_NORM},
	{"is less|smaller|lower than", "<", P_CMP, B_NORM},
	{"is below|under", "<", P_CMP, B_NORM},
	{"less|smaller than", "<", P_CMP, B_NORM},
	{"<", "<", P_CMP, B_NORM},
	{"is not equal to", "!=", P_CMP, B_NORM},
	{"does not equal", "!=", P_CMP, B_NORM},
	{"!=", "!=", P_CMP, B_NORM},
	{"is equal to", "==", P_CMP, B_NORM},
	{"equals", "==", P_CMP, B_NORM},
	{"==", "==", P_CMP, B_NORM},
	{"=", "==", P_CMP, B_NORM},
	{"is between", NULL, P_CMP, B_BETWEEN},
	{"lies between", NULL, P_CMP, B_BETWEEN},
	{"is in", NULL, P_CMP, B_IN},
	{"in", NULL, P_CMP, B_IN},
	{"is not", "!=", P_CMP, B_IS},
	{"is", "==", P_CMP, B_IS},
	{"if", NULL, P_TERN, B_IF},
	{"logical or", "||", P_OR, B_NORM},
	{"OROR", "||", P_OR, B_NORM},
	{"or", NULL, P_OR, B_OR},
	{"logical and", "&&", P_AND, B_NORM},
	{"&&", "&&", P_AND, B_NORM},
	{"and", NULL, P_AND, B_AND},
	{"bitwise or", "|", P_BOR, B_NORM},
	{"ored with", "|", P_BOR, B_NORM},
	{"BAR", "|", P_BOR, B_NORM},
	{"bitwise xor", "xor", P_BXOR, B_NORM},
	{"exclusive or|exclusive-or", "xor", P_BXOR, B_NORM},
	{"xored with", "xor", P_BXOR, B_NORM},
	{"xor", "xor", P_BXOR, B_NORM},
	{"bitwise and", "&", P_BAND, B_NORM},
	{"anded with", "&", P_BAND, B_NORM},
	{"masked with|by", "&", P_BAND, B_NORM},
	{"&", "&", P_BAND, B_NORM},
	{"shifted left by", "<<", P_SHIFT, B_NORM},
	{"shifted to the left by", "<<", P_SHIFT, B_NORM},
	{"left shifted by", "<<", P_SHIFT, B_NORM},
	{"left-shifted by", "<<", P_SHIFT, B_NORM},
	{"shift left by", "<<", P_SHIFT, B_NORM},
	{"<<", "<<", P_SHIFT, B_NORM},
	{"shifted right by", ">>", P_SHIFT, B_NORM},
	{"shifted to the right by", ">>", P_SHIFT, B_NORM},
	{"right shifted by", ">>", P_SHIFT, B_NORM},
	{"right-shifted by", ">>", P_SHIFT, B_NORM},
	{"shift right by", ">>", P_SHIFT, B_NORM},
	{">>", ">>", P_SHIFT, B_NORM},
	{"plus|+", "+", P_ADD, B_NORM},
	{"added to", "+", P_ADD, B_SWAP},
	{"minus|-", "-", P_ADD, B_NORM},
	{"subtracted from", "-", P_ADD, B_SWAP},
	{"times|*", "*", P_MUL, B_NORM},
	{"multiplied by|with", "*", P_MUL, B_NORM},
	{"divided by", "/", P_MUL, B_NORM},
	{"over", "/", P_MUL, B_OVER},
	{"/", "/", P_MUL, B_NORM},
	{"mod|modulo|%", "%", P_MUL, B_NORM},
	{"as", NULL, P_TERN, B_CAST},
	{"converted|cast|casted|truncated to|into", NULL, P_TERN, B_CAST},
	{NULL, NULL, 0, 0}
};

static int parse_varlist(char names[][32], int *n);

/* words that start a clause or an option rather than an operand */
static int
clause_word(int i)
{
	static const char *const cw[] = {"for", "where", "with", "when",
		"whenever", "given", "assuming", "on", "using", "over", "such",
		"so", "as", "in", "letting", "call", "name", "then", "please",
		"and", "if", "to", "against", "versus", "vs", NULL};

	return W[i].k == W_END || (W[i].k == W_WORD && in_list(cw, W[i].s));
}

/* after "and": does a new clause start here rather than an operand? */
static int
and_starts_clause(void)
{
	char names[8][32];
	int s = wi, r = 0, n;

	if (peek("it|its|then|also|show|give|print|tell|what|with|for|where|"
	    "call|name|use|using|emit|write|both|all|each|every|is|are|whether|"
	    "how|does|do|can|in|as"))
		return 1;
	quiet++;
	if (parse_varlist(names, &n) && (issym(":") || issym("=") ||
	    peek("is|are|be|between|from|in|ranges|range|goes|runs|varies")))
		r = 1;
	quiet--;
	wi = s;
	return r;
}

static const struct bop *
match_bop(void)
{
	int k, s = wi;
	const struct bop *o;

	for (k = 0; bops[k].pat; k++) {
		if (!mat(bops[k].pat))
			continue;
		o = &bops[k];
		if (((o->kind == B_IN || o->kind == B_IS) && !incond) ||
		    (o->kind == B_AND && and_starts_clause()) ||
		    (o->kind == B_OVER && peek("all|every|each|any|its|their")) ||
		    (o->kind == B_CAST && !peek_type()) ||
		    (o->kind == B_IS && peek("a|an|the|both|all|defined|safe|"
		    "positive|negative|nonzero|non-zero|even|odd|between|from"))) {
			wi = s;
			continue;
		}
		return o;
	}
	return NULL;
}

static const struct bop *
peek_bop(void)
{
	int s = wi;
	const struct bop *o = match_bop();

	wi = s;
	return o;
}

/*
 * A word operator was applied to an operand parsed up to the next binary
 * operator.  If one follows that could also belong inside ("the square
 * root of x plus 1"), ask which was meant.
 */
static int
want_wide(struct ex arg, int minprec, const char *pre, const char *post,
    const char *phrase)
{
	const struct bop *o;
	const char *opt[2];
	int lo = minprec > P_BOR ? minprec : P_BOR;

	if (arg.closed || W[wi].k == W_END || issym(","))
		return 0;
	o = peek_bop();
	if (o == NULL || (o->kind != B_NORM && o->kind != B_SWAP &&
	    o->kind != B_OVER) || o->prec < lo || o->prec >= P_POW)
		return 0;
	opt[0] = S("%s%s%s %s ...", pre, arg.s, post, o->op);
	opt[1] = S("%s%s %s ...%s", pre, arg.s, o->op, post);
	return clarify(NULL, S("what does \"%s\" apply to", phrase), opt, 2, 0,
	    S("say \"%s the quantity ...\" for the second", phrase)) == 1;
}

static struct ex
var_atom(const char *name)
{
	struct var *v = var_get(name);

	if (!v->order)
		v->order = ++norder;
	v->uses |= cur_mask;
	return mkatom(S("%s", v->name), v->ity ? v->ity : 3, 0);
}

static struct var Vsave[MAXV];

/*
 * The constant ends of an interval: [a, b], (a, b), a to b, a..b, and
 * after "between", a and b.  Nothing is consumed when there is none.
 */
static int
const_interval(struct ex *a, struct ex *b, int between)
{
	jmp_buf saved;
	int s = wi, ok = 0, br, nvs = nv, nos = norder;

	memcpy(saved, nl_jmp, sizeof saved);
	memcpy(Vsave, V, sizeof V);
	quiet++;
	if (setjmp(nl_jmp) == 0) {
		br = acc_sym("[") || acc_sym("(");
		*a = parse_expr(P_ADD);
		if ((br && acc_sym(",")) || acc_sym("..") ||
		    mat("to|through|thru|until|till") || (between && mat("and"))) {
			*b = parse_expr(P_ADD);
			ok = a->hv && b->hv && (!br || acc_sym("]") || acc_sym(")"));
		}
	}
	quiet--;
	memcpy(nl_jmp, saved, sizeof saved);
	if (!ok) {
		wi = s;
		memcpy(V, Vsave, sizeof V);
		nv = nvs;
		norder = nos;
		return 0;
	}
	if (a->v > b->v) {
		struct ex t = *a;
		*a = *b;
		*b = t;
		if (!dry)
			printf("  note: read the interval as [%s, %s]\n", a->s, b->s);
	}
	return 1;
}

/* the variable a bare function name stands for: cos on [a, b] is cos(x) */
static struct ex
implicit_var(const char *lo, const char *hi)
{
	static const char *const cand[] = {"x", "t", "u", "v", NULL};
	struct var *v;
	const char *name = "x";
	int k;

	for (k = 0; cand[k]; k++)
		if ((v = var_find(cand[k])) == NULL || !v->order ||
		    (lo == NULL && !v->lo[0])) {
			name = cand[k];
			break;
		}
	v = var_get(name);
	if (lo) {
		snprintf(v->lo, sizeof v->lo, "%s", lo);
		snprintf(v->hi, sizeof v->hi, "%s", hi);
		v->bc = 3;
		v->decl = 1;
		Q.interval = 1;
	}
	return var_atom(name);
}

static void
seed_params(struct func *f)
{
	struct var *v;
	int k;

	for (k = 0; k < f->np; k++) {
		v = var_get(sym_str(f->p[k].name));
		if (!v->ty[0]) {
			snprintf(v->ty, sizeof v->ty, "%s", ty_str(f->p[k].ty));
			v->ity = ty_ity(v->ty);
			if (f->p[k].ranged && !v->lo[0]) {
				if (TY_FP(f->p[k].ty)) {
					snprintf(v->lo, sizeof v->lo, "%.17g", f->p[k].rlo);
					snprintf(v->hi, sizeof v->hi, "%.17g", f->p[k].rhi);
				} else {
					snprintf(v->lo, sizeof v->lo, "%lld",
					    (long long)f->p[k].lo);
					snprintf(v->hi, sizeof v->hi, "%lld",
					    (long long)f->p[k].hi);
				}
			}
		}
		if (!v->order)
			v->order = ++norder;
		v->uses |= cur_mask;
	}
}

/* f(a, b), f of a and b, f at 3, or plain f meaning f(its parameters) */
static struct ex
call_user(struct func *f, const char *name, int minprec)
{
	struct ex a[MAXP], r;
	struct sbuf sb = {0};
	int n = 0, k, s;

	if (acc_sym("(")) {
		if (!acc_sym(")")) {
			do {
				if (n == MAXP)
					nfail("too many arguments");
				a[n++] = parse_expr(P_TERN);
			} while (acc_sym(","));
			if (!acc_sym(")"))
				nfail("expected ')' after the arguments of %s", name);
		}
	} else if (mat("of|at|on") || mat("applied to") || mat("evaluated at")) {
		s = wi;
		if (f->np == 1) {
			a[n++] = parse_expr(P_POW);
			if (want_wide(a[0], minprec, S("%s(", name), ")",
			    words_text(s - 1, s))) {
				wi = s;
				a[0] = parse_expr(minprec > P_BOR ? minprec : P_BOR);
			}
		} else {
			a[n++] = parse_expr(P_BOR);
			while (n < f->np && n < MAXP && (acc_sym(",") || mat("and")))
				a[n++] = parse_expr(P_BOR);
		}
	} else {
		seed_params(f);
		for (k = 0; k < f->np; k++)
			a[n++] = var_atom(sym_str(f->p[k].name));
	}
	if (n == 0)
		return mkatom(S("%s()", name), 0, 0);
	sb_printf(&sb, "%s(", name);
	r = mkatom("", 0, 1);
	for (k = 0; k < n; k++) {
		sb_printf(&sb, "%s%s", k ? ", " : "", a[k].s);
		r.ity = icomb(r.ity, a[k].ity);
		r.lit = r.lit && a[k].lit;
	}
	sb_putc(&sb, ')');
	r.s = S("%s", sb.p);
	r.closed = 1;
	r.lit = 0;
	free(sb.p);
	return r;
}

/* word operators taking one operand */
enum {
	U_FN, U_SQRT, U_SQRTR, U_LOG, U_LOG2, U_LOG10, U_CBRT, U_RSQRT, U_SQ,
	U_CUBE, U_NEG, U_NOT, U_BNOT, U_DIV, U_MUL, U_RECIP
};
struct uop {
	const char *pat;
	int kind;
	const char *fn;
	int k;
};
static const struct uop uops[] = {
	/* Lambert W, the inverse of w e^w */
	{"?the principal|main|upper branch of ?the lambert w|W ?function ?of",
	    U_FN, "W", 0},
	{"?the lower|negative|secondary|other branch of ?the lambert w|W "
	    "?function ?of", U_FN, "Wm1", 0},
	{"?the lambert w|W ?function ?of", U_FN, "W", 0},
	{"?the product log|productlog ?function ?of", U_FN, "W", 0},
	{"?the omega function of", U_FN, "W", 0},
	{"?the integer|floor square root of", U_FN, "isqrt", 0},
	{"?the integer|floor sqrt of", U_FN, "isqrt", 0},
	{"isqrt ?of", U_FN, "isqrt", 0},
	{"?the real square root of", U_SQRTR, NULL, 0},
	{"?the square root of", U_SQRT, NULL, 0},
	{"?the sqrt of", U_SQRT, NULL, 0},
	{"sqrt", U_SQRT, NULL, 0},
	{"?the cube root of", U_CBRT, NULL, 0},
	{"cbrt ?of", U_CBRT, NULL, 0},
	{"?the reciprocal|inverse square root of", U_RSQRT, NULL, 0},
	{"rsqrt ?of", U_RSQRT, NULL, 0},
	{"?the absolute value of", U_FN, "abs", 0},
	{"?the magnitude of", U_FN, "abs", 0},
	{"abs ?of", U_FN, "abs", 0},
	{"?the sine of", U_FN, "sin", 0},
	{"sin|sine ?of", U_FN, "sin", 0},
	{"?the cosine of", U_FN, "cos", 0},
	{"cos|cosine ?of", U_FN, "cos", 0},
	{"?the tangent of", U_FN, "tan", 0},
	{"tan|tangent ?of", U_FN, "tan", 0},
	{"?the exponential of", U_FN, "exp", 0},
	{"exp ?of", U_FN, "exp", 0},
	{"e to the power of", U_FN, "exp", 0},
	{"e to the", U_FN, "exp", 0},
	{"e raised to ?the ?power ?of", U_FN, "exp", 0},
	{"?the natural log|logarithm of", U_FN, "log", 0},
	{"ln ?of", U_FN, "log", 0},
	{"?the log|logarithm base|to-base 2|two of", U_LOG2, NULL, 0},
	{"?the log|logarithm to base 2|two of", U_LOG2, NULL, 0},
	{"?the log|logarithm to the base 2|two of", U_LOG2, NULL, 0},
	{"?the base 2|two log|logarithm of", U_LOG2, NULL, 0},
	{"?the base-2|base-two log|logarithm of", U_LOG2, NULL, 0},
	{"?the binary log|logarithm of", U_LOG2, NULL, 0},
	{"log2|lg ?of", U_LOG2, NULL, 0},
	{"?the log|logarithm base|to-base 10|ten of", U_LOG10, NULL, 0},
	{"?the log|logarithm to base 10|ten of", U_LOG10, NULL, 0},
	{"?the log|logarithm to the base 10|ten of", U_LOG10, NULL, 0},
	{"?the base 10|ten log|logarithm of", U_LOG10, NULL, 0},
	{"?the base-10|base-ten log|logarithm of", U_LOG10, NULL, 0},
	{"?the common|decimal log|logarithm of", U_LOG10, NULL, 0},
	{"log10 ?of", U_LOG10, NULL, 0},
	{"?the log|logarithm of", U_LOG, NULL, 0},
	{"log|logarithm", U_LOG, NULL, 0},
	{"?the floor of", U_FN, "floor", 0},
	{"floor ?of", U_FN, "floor", 0},
	{"?the ceiling|ceil of", U_FN, "ceil", 0},
	{"ceil|ceiling ?of", U_FN, "ceil", 0},
	{"?the sign|signum of", U_FN, "sign", 0},
	{"signum ?of", U_FN, "sign", 0},
	{"?the number|count of set|one|1 bits in|of", U_FN, "popcount", 0},
	{"?the number|count of ones|1s in|of", U_FN, "popcount", 0},
	{"?the population count of", U_FN, "popcount", 0},
	{"?the hamming weight of", U_FN, "popcount", 0},
	{"?the bit count of", U_FN, "popcount", 0},
	{"?the popcount|popcnt of", U_FN, "popcount", 0},
	{"popcount|popcnt ?of", U_FN, "popcount", 0},
	{"?the number|count of leading zeros|zeroes in|of", U_FN, "clz", 0},
	{"?the number|count of leading zero bits in|of", U_FN, "clz", 0},
	{"?the leading zero count of", U_FN, "clz", 0},
	{"clz ?of", U_FN, "clz", 0},
	{"?the number|count of trailing zeros|zeroes in|of", U_FN, "ctz", 0},
	{"?the number|count of trailing zero bits in|of", U_FN, "ctz", 0},
	{"?the trailing zero count of", U_FN, "ctz", 0},
	{"ctz ?of", U_FN, "ctz", 0},
	{"?the bits|representation of", U_FN, "asuint", 0},
	{"?the bit pattern of", U_FN, "asuint", 0},
	{"?the float|floating-point ?number|value with bits", U_FN, "asfloat", 0},
	{"?the float whose bits are", U_FN, "asfloat", 0},
	{"?the negative|negation|opposite of", U_NEG, NULL, 0},
	{"negative|minus", U_NEG, NULL, 0},
	{"?the logical negation of", U_NOT, NULL, 0},
	{"logical not", U_NOT, NULL, 0},
	{"?the bitwise not|complement|inverse of", U_BNOT, NULL, 0},
	{"?the complement|ones-complement of", U_BNOT, NULL, 0},
	{"bitwise not", U_BNOT, NULL, 0},
	{"not", U_NOT, NULL, 0},
	{"?the square of", U_SQ, NULL, 0},
	{"?the cube of", U_CUBE, NULL, 0},
	{"?the reciprocal|inverse of", U_RECIP, NULL, 0},
	{"half of", U_DIV, NULL, 2},
	{"half", U_DIV, NULL, 2},
	{"a|one half of", U_DIV, NULL, 2},
	{"a|one third of", U_DIV, NULL, 3},
	{"a|one quarter|fourth of", U_DIV, NULL, 4},
	{"twice|double", U_MUL, NULL, 2},
	{"triple|thrice", U_MUL, NULL, 3},
	{NULL, 0, NULL, 0}
};

static struct ex
apply_uop(const struct uop *u, struct ex a)
{
	static const char *const logo[] = {"the natural log, ln", "log base 2",
		"log base 10"};
	static const char *const sq[] = {"the floor square root, isqrt",
		"the real square root, sqrt(real(...))"};
	static const char *const no[] = {"bitwise not, ~", "logical not, !"};
	int k;

	switch (u->kind) {
	case U_FN:
		return mkcall(u->fn, a);
	case U_SQRT:
		if (a.ity != 1)
			return mkcall("sqrt", a);
		k = clarify("sqrt", "\"square root\" of an integer", sq, 2, 0,
		    "say \"integer square root\" or \"real square root\"");
		return k ? mkcall("sqrt", mkcall("real", a)) : mkcall("isqrt", a);
	case U_SQRTR:
		return mkcall("sqrt", a.ity == 1 ? mkcall("real", a) : a);
	case U_LOG:
		k = clarify("log", "\"log\" without a base", logo, 3, 0,
		    "say \"natural log\", \"log base 2\" or \"log base 10\", or "
		    "\"assume log means base 2\"");
		if (k == 0)
			return mkcall("log", a);
		return mkbin(mkcall("log", a), "/", mkatom(k == 1 ? "log(2)" :
		    "log(10)", 0, 1), P_MUL);
	case U_LOG2:
		return mkbin(mkcall("log", a), "/", mkatom("log(2)", 0, 1), P_MUL);
	case U_LOG10:
		return mkbin(mkcall("log", a), "/", mkatom("log(10)", 0, 1), P_MUL);
	case U_CBRT:
		return mkbin(a, "^", mkbin(num("1"), "/", num("3"), P_MUL), P_POW);
	case U_RSQRT:
		return mkbin(num("1"), "/", mkcall("sqrt", a), P_MUL);
	case U_SQ:
		return mkbin(a, "^", num("2"), P_POW);
	case U_CUBE:
		return mkbin(a, "^", num("3"), P_POW);
	case U_NEG:
		return mkun("-", a);
	case U_NOT:
		if (a.isbool)
			return mkun("!", a);
		k = clarify("not", "\"not\" of a number", no, 2, 0,
		    "say \"bitwise not\" or \"logical not\"");
		return mkun(k ? "!" : "~", a);
	case U_BNOT:
		return mkun("~", a);
	case U_DIV:
		return mkbin(a, "/", num(S("%d", u->k)), P_MUL);
	case U_MUL:
		return mkbin(num(S("%d", u->k)), "*", a, P_MUL);
	case U_RECIP:
		return mkbin(num("1"), "/", a, P_MUL);
	}
	return a;
}

/* how a word operator is shown when asking about its scope */
static void
uop_shape(const struct uop *u, const char **pre, const char **post)
{
	*post = ")";
	switch (u->kind) {
	case U_FN:
		*pre = S("%s(", u->fn);
		break;
	case U_SQRT:
	case U_SQRTR:
		*pre = "sqrt(";
		break;
	case U_LOG:
	case U_LOG2:
	case U_LOG10:
		*pre = "log(";
		break;
	case U_CBRT:
		*pre = "cbrt(";
		break;
	case U_RSQRT:
		*pre = "1/sqrt(";
		break;
	case U_SQ:
		*pre = "(";
		*post = ")^2";
		break;
	case U_CUBE:
		*pre = "(";
		*post = ")^3";
		break;
	case U_DIV:
		*pre = "(";
		*post = S(")/%d", u->k);
		break;
	case U_MUL:
		*pre = S("%d*(", u->k);
		break;
	case U_RECIP:
		*pre = "1/(";
		break;
	default:
		*pre = "(";
	}
}

/* n-ary phrases: the sum of a, b and c */
enum { N_FOLD, N_TWO, N_AVG, N_MIN, N_MAX };
struct nop {
	const char *pat, *op;
	int prec, kind;
};
static const struct nop nops[] = {
	{"?the sum of", "+", P_ADD, N_FOLD},
	{"?the sum over|for", "+", P_ADD, N_FOLD},
	{"?the product over|for", "*", P_MUL, N_FOLD},
	{"?the product of", "*", P_MUL, N_FOLD},
	{"?the difference between|of", "-", P_ADD, N_TWO},
	{"?the quotient|ratio of", "/", P_MUL, N_TWO},
	{"?the average|mean of", "+", P_ADD, N_AVG},
	{"?the min|minimum|smaller|smallest|lesser|least of", NULL, 0, N_MIN},
	{"?the max|maximum|larger|largest|greater|greatest|bigger|biggest of",
	    NULL, 0, N_MAX},
	{"?the bitwise and of", "&", P_BAND, N_FOLD},
	{"?the bitwise or of", "|", P_BOR, N_FOLD},
	{"?the bitwise xor of", "xor", P_BXOR, N_FOLD},
	{"?the xor|exclusive-or of", "xor", P_BXOR, N_FOLD},
	{"?the exclusive or of", "xor", P_BXOR, N_FOLD},
	{NULL, NULL, 0, 0}
};

static const char *subst_ident(const char *s, const char *name,
    const char *val);

/* sum(i, lo, hi, body): i is bound there, not a parameter */
static struct ex
index_sum(const struct nop *o, const char *iv, struct ex lo, struct ex hi,
    struct ex body)
{
	struct var *v = var_find(iv);
	struct ex r;

	int k;

	if (v && !v->decl) {
		v->order = 0;
		v->uses = 0;
	}
	/* a free bound counts: an int, 0..100 as ucas would take it */
	for (k = 0; k < nv; k++) {
		struct var *b = &V[k];
		if (b == v || b->ty[0] || !b->order)
			continue;
		if (strcmp(subst_ident(S("%s %s", lo.s, hi.s), b->name, ""),
		    S("%s %s", lo.s, hi.s))) {
			strcpy(b->ty, "int");
			b->ity = 1;
			if (!b->lo[0]) {
				strcpy(b->lo, "0");
				strcpy(b->hi, "100");
			}
		}
	}
	r = mkatom(S("%s(%s, %s, %s, %s)", o->op[0] == '+' ? "sum" : "prod",
	    iv, lo.s, hi.s, body.s), icomb(lo.ity, hi.ity), 0);
	r.closed = 1;
	return r;
}

static struct ex
parse_nary(const struct nop *o)
{
	struct ex a[16], r;
	int n = 0, k;
	const char *opt[2];

	/* "the sum over i from 1 to n of i squared" */
	if ((o->kind == N_FOLD && (!strcmp(o->op, "+") || !strcmp(o->op, "*"))) &&
	    varlike(wi) && isw(wi + 1, "from")) {
		const char *iv = S("%s", W[wi].o);
		struct ex lo, hi, body;
		wi += 2;
		lo = parse_expr(P_ADD);
		if (!mat("to|through|until") && !acc_sym(".."))
			nfail("from %s to what?", lo.s);
		hi = parse_expr(P_ADD);
		if (!mat("of") && !acc_sym(","))
			nfail("the %s of what?", o->op[0] == '+' ? "sum" : "product");
		body = parse_expr(P_BOR);
		return index_sum(o, iv, lo, hi, body);
	}
	a[n++] = parse_expr(P_BOR);
	/* "the sum of i squared for i from 1 to n" */
	if ((o->kind == N_FOLD && (!strcmp(o->op, "+") || !strcmp(o->op, "*"))) &&
	    peek("for|over|as") && varlike(wi + 1) &&
	    (isw(wi + 2, "from") || isw(wi + 2, "between") ||
	    isw(wi + 2, "in") || (W[wi + 2].k == W_SYM &&
	    !strcmp(W[wi + 2].s, "=")))) {
		const char *iv = S("%s", W[wi + 1].o);
		struct ex lo, hi;
		wi += 3;
		lo = parse_expr(P_ADD);
		if (!mat("to|through|until|and") && !acc_sym(".."))
			nfail("from %s to what?", lo.s);
		hi = parse_expr(P_ADD);
		return index_sum(o, iv, lo, hi, a[0]);
	}
	for (;;) {
		int s = wi;
		if (n == 16)
			break;
		if (acc_sym(",")) {
			if (mat("and")) {
				if (and_starts_clause()) {
					wi = s;
					break;
				}
			} else if (clause_word(wi)) {
				wi = s;
				break;
			}
		} else if (mat("and")) {
			if (and_starts_clause()) {
				wi = s;
				break;
			}
		} else
			break;
		if (W[wi].k == W_END) {
			wi = s;
			break;
		}
		a[n++] = parse_expr(P_BOR);
	}
	if (n < 2)
		nfail("\"%s\" needs two or more things, joined by \"and\"",
		    o->kind == N_MIN ? "minimum" : o->kind == N_MAX ? "maximum" :
		    o->op);
	if (o->kind == N_TWO && n != 2)
		nfail("a difference or ratio is of two things");
	/* "the sum of x and y squared": (x + y)^2 or x + y^2? */
	if (a[n - 1].pword && a[n - 1].prec == P_POW) {
		struct ex last = a[n - 1];
		struct ex base = mkatom(last.pb, last.ity, 0);
		struct ex t;
		base.prec = last.pbprec;
		t = a[0];
		for (k = 1; k < n - 1; k++)
			t = mkbin(t, o->op ? o->op : ",", a[k], o->prec ? o->prec :
			    P_ADD);
		opt[0] = S("... %s %s", o->op ? o->op : ",", last.s);
		opt[1] = S("(... %s %s)^%s", o->op ? o->op : ",", base.s, last.pe);
		if (o->kind != N_MIN && o->kind != N_MAX &&
		    clarify(NULL, S("what does \"%s\" apply to", "squared/power"),
		    opt, 2, 0, "say \"the square of the sum of ...\" for the second")) {
			a[n - 1] = base;
			r = a[0];
			for (k = 1; k < n; k++)
				r = mkbin(r, o->op, a[k], o->prec);
			if (o->kind == N_AVG)
				r = mkbin(r, "/", num(S("%d", n)), P_MUL);
			r = mkbin(r, "^", num(last.pe), P_POW);
			return r;
		}
	}
	if (o->kind == N_MIN || o->kind == N_MAX) {
		r = a[0];
		for (k = 1; k < n; k++) {
			struct ex t = mkatom(S("%s(%s, %s)", o->kind == N_MIN ? "min" :
			    "max", r.s, a[k].s), icomb(r.ity, a[k].ity), 0);
			t.closed = 1;
			r = t;
		}
		return r;
	}
	r = a[0];
	for (k = 1; k < n; k++)
		r = mkbin(r, o->op, a[k], o->prec);
	if (o->kind == N_AVG)
		r = mkbin(r, "/", num(S("%d", n)), P_MUL);
	r.closed = 1;
	return r;
}

static struct ex parse_postfix(struct ex e, int minprec);

static struct ex
parse_prefix(int minprec)
{
	struct ex e, a, b, c;
	struct func *f;
	struct decl d;
	int64_t v;
	int s = wi, s1, k, den;

	switch (W[wi].k) {
	case W_END:
		nfail("the sentence ends where I expected a value");
		break;
	case W_SYM:
		if (issym("[") && const_interval(&a, &b, 0))
			return implicit_var(a.s, b.s);
		if (acc_sym("(")) {
			e = parse_expr(0);
			if (!acc_sym(")"))
				nfail("expected ')'");
			e.closed = 1;
			e.pword = 0;
			return e;
		}
		if (acc_sym("BAR")) {
			e = parse_expr(P_BXOR);
			if (!acc_sym("BAR"))
				nfail("expected '|' to close the absolute value");
			return mkcall("abs", e);
		}
		if (acc_sym("-"))
			return mkun("-", parse_expr(P_UNARY));
		if (acc_sym("+"))
			return parse_expr(P_UNARY);
		if (acc_sym("~"))
			return mkun("~", parse_expr(P_UNARY));
		if (acc_sym("!"))
			return mkun("!", parse_expr(P_UNARY));
		nfail("I didn't expect '%s' here", W[wi].o);
		break;
	case W_NUM:
		e = num(S("%s", W[wi].o));
		wi++;
		/* units: 90r, 1.5 rad are radians; 90deg, 90 degrees, 90° */
		if (W[wi].k == W_WORD && !W[wi].sp && (isw(wi, "r") ||
		    isw(wi, "rad") || isw(wi, "rads")) && !var_find("r"))
			wi++;
		else if (mat("radian|radians"))
			;
		else if (W[wi + 1].k != W_NUM && mat("deg|degs|degree|degrees")) {
			/* not "degree 4": that is a polynomial's degree */
			e = mkbin(mkbin(e, "*", num("pi"), P_MUL), "/", num("180"),
			    P_MUL);
			return e;
		}
		goto fraction;
	}
	if (number_words(&v)) {
		e = num(S("%lld", (long long)v));
fraction:
		if ((den = fraction_word(wi)) != 0) {
			wi++;
			e = mkbin(e, "/", num(S("%d", den)), P_MUL);
			e.lit = 1;
			if (mat("of"))
				e = mkbin(e, "*", parse_expr(P_UNARY), P_MUL);
		}
		return e;
	}
	if (mat("?the quantity")) {
		e = parse_expr(P_BOR);
		acc_sym(",");
		e.closed = 1;
		return e;
	}
	if (mat("if")) {
		incond++;
		c = parse_expr(P_OR);
		incond--;
		acc_sym(",");
		mat("?then");
		a = parse_expr(P_TERN + 1);
		acc_sym(",");
		if (!mat("else|otherwise"))
			return mkatom(S("{ %s if %s }", a.s, c.s), a.ity, 0);
		b = parse_expr(P_TERN);
		e = mkatom(S("%s ? %s : %s", paren(c, P_TERN, 1), paren(a, P_TERN, 1),
		    paren(b, P_TERN, 0)), icomb(a.ity, b.ity), 0);
		e.prec = P_TERN;
		return e;
	}
	if (mat("pi"))
		return num("pi");
	if (mat("infinity|inf")) {
		e = mkatom("inf", 0, 1);
		e.hv = 1;
		e.v = INFINITY;
		return e;
	}
	if (isw(wi, "e") && !varlike(wi + 1) && !peek("e to") &&
	    !var_find("e")) {
		wi++;
		return num("e");
	}
	if (mat("true|false"))
		return num(S("%s", W[wi - 1].s));
	/* calls in ucas notation: sin(x), min(a, b), u8(x) */
	if (builtin_call(wi)) {
		const char *fn = W[wi].o;
		struct sbuf sb = {0};
		wi += 2;
		sb_printf(&sb, "%s(", fn);
		e = mkatom("", 0, 1);
		if (!acc_sym(")")) {
			k = 0;
			do {
				a = parse_expr(P_TERN);
				sb_printf(&sb, "%s%s", k++ ? ", " : "", a.s);
				e.ity = icomb(e.ity, a.ity);
				e.lit = e.lit && a.lit;
			} while (acc_sym(","));
			if (!acc_sym(")"))
				nfail("expected ')' after the arguments of %s", fn);
		}
		sb_putc(&sb, ')');
		e.s = S("%s", sb.p);
		e.closed = 1;
		free(sb.p);
		return e;
	}
	/* user functions */
	if (W[wi].k == W_WORD && (f = user_fn(W[wi].o)) != NULL &&
	    !var_find(W[wi].o)) {
		const char *name = S("%s", W[wi].o);
		wi++;
		return call_user(f, name, minprec);
	}
	if (mat("it|this|that") && !varlike(wi - 1)) {
		const char *name = last_name();
		if (name == NULL)
			nfail("\"%s\" has nothing to refer to yet", W[wi - 1].o);
		return call_user(user_fn(name), S("%s", name), minprec);
	}
	/* phrases */
	for (k = 0; nops[k].pat; k++)
		if (mat(nops[k].pat))
			return parse_nary(&nops[k]);
	if (mat("?the remainder of|when")) {
		a = parse_expr(P_MUL + 1);
		if (!mat("?is divided by") && !mat("modulo|mod|over|by"))
			nfail("expected \"divided by\" in the remainder");
		b = parse_expr(P_MUL + 1);
		return mkbin(a, "%", b, P_MUL);
	}
	for (k = 0; uops[k].pat; k++) {
		const char *pre, *post;
		s1 = wi;
		if (!mat(uops[k].pat))
			continue;
		/* "minus 3" is a number, "double x" is twice x, "not" waits */
		if (uops[k].kind == U_NEG || uops[k].kind == U_NOT ||
		    uops[k].kind == U_BNOT)
			return apply_uop(&uops[k], parse_expr(P_UNARY));
		s = wi;
		uop_shape(&uops[k], &pre, &post);
		if (const_interval(&a, &b, 0)) {
			/* cos of -pi/2 to pi/2: over an interval */
			e = apply_uop(&uops[k], implicit_var(a.s, b.s));
			e.closed = 1;
			return e;
		}
		if (W[wi].k == W_END || issym(",") || (clause_word(wi) &&
		    !isw(wi, "if") && !isw(wi, "to")) || isw(wi, "from") ||
		    isw(wi, "using") || isw(wi, "via") || isw(wi, "across") ||
		    isw(wi, "within")) {
			/* just "cos": of an implicit x */
			e = apply_uop(&uops[k], implicit_var(NULL, NULL));
			e.closed = 1;
			return e;
		}
		a = parse_expr(P_POW);
		if (want_wide(a, minprec, pre, post, words_text(s1, s))) {
			wi = s;
			a = parse_expr(minprec > P_BOR ? minprec : P_BOR);
			a.closed = 1;
		}
		e = apply_uop(&uops[k], a);
		e.closed = 1;
		return e;
	}
	/* "a 4-bit unsigned x", "an unsigned byte x" */
	if (mat("a|an|every|each|any|some|all")) {
		s1 = wi;
		memset(&d, 0, sizeof d);
		if (parse_type(&d) && d.ty[0] && varlike(wi)) {
			struct var *vv = var_get(W[wi].o);
			snprintf(vv->ty, sizeof vv->ty, "%s", d.ty);
			vv->ity = d.ity;
			vv->decl = 1;
			if (d.lo[0]) {
				strcpy(vv->lo, d.lo);
				strcpy(vv->hi, d.hi);
			}
			wi++;
			return var_atom(W[wi - 1].o);
		}
		wi = s1;
		if (isw(s, "a") && (W[wi].k == W_END || W[wi].k == W_SYM ||
		    peek_bop() || peek("squared|cubed|to|raised") ||
		    clause_word(wi))) {
			wi = s + 1;
			return var_atom(W[s].o);
		}
		mat("?variable|value|parameter|input|argument|number|constant");
		return parse_prefix(minprec);
	}
	if (mat("?the value of") || mat("the variable|parameter|input|argument|"
	    "number|constant") || mat("the"))
		return parse_prefix(minprec);
	if (W[wi].k == W_WORD) {
		if (varlike(wi)) {
			wi++;
			/* "n-th" */
			return var_atom(W[wi - 1].o);
		}
		if (!reserved_word(W[wi].s) && !in_list(builtins, W[wi].s)) {
			static const char *const o[] = {"a variable",
				"a mistake (stop here)"};
			char what[64];
			snprintf(what, sizeof what, "I don't know the word \"%s\"",
			    W[wi].o);
			if (clarify(NULL, S("%s", what), o, 2, 0, "rephrase it, or "
			    "name variables with one or two letters") == 1)
				nfail("I don't know the word \"%s\"", W[wi].o);
			if (!dry && nknown < 32)
				known_names[nknown++] = strdup(W[wi].o);
			wi++;
			return var_atom(W[wi - 1].o);
		}
	}
	nfail("I don't understand \"%s\" here", words_text(wi, wi + 3));
	return num("0");
}

static int c2_juxt(int i);	/* nlp_calc2.inc */

static struct ex
parse_postfix(struct ex e, int minprec)
{
	struct ex x;
	int s, k, s1;

	for (;;) {
		s = wi;
		if (mat("squared") || acc_sym("\xc2\xb2")) {
			e = mkpow(e, num("2"), W[s].k == W_WORD);
			continue;
		}
		if (mat("cubed") || acc_sym("\xc2\xb3")) {
			e = mkpow(e, num("3"), W[s].k == W_WORD);
			continue;
		}
		if (mat("to the") && (k = ordinal(&x)) != 0) {
			if (k == 2)
				x = var_atom("n");
			mat("?power");
			e = mkpow(e, x, 1);
			continue;
		}
		wi = s;
		if (mat("to the power ?of") || mat("raised to ?the ?power ?of") ||
		    mat("to the") || mat("raised to")) {
			s1 = wi;
			x = parse_expr(P_POW);
			if (want_wide(x, minprec, S("%s^(", paren(e, P_POW, 1)), ")",
			    words_text(s, s1))) {
				wi = s1;
				x = parse_expr(minprec > P_BOR ? minprec : P_BOR);
			}
			e = mkpow(e, x, 0);
			continue;
		}
		if (acc_sym("^") || acc_sym("**")) {
			x = parse_expr(P_POW);
			e = mkbin(e, "^", x, P_POW);
			continue;
		}
		/* 2x, 3(x + 1), 2pi */
		if (e.lit && e.prec == P_ATOM && !W[wi].sp && (((varlike(wi) ||
		    isw(wi, "pi")) &&
		    !peek("th|st|nd|rd")) || issym("("))) {
			x = parse_expr(P_POW);
			e = mkbin(e, "*", x, P_MUL);
			continue;
		}
		/* x^2 y, x sin x, x e^x: juxtaposition multiplies (calc2) */
		if (minprec <= P_MUL && W[wi].sp && c2_juxt(wi)) {
			x = parse_expr(P_MUL + 1);
			e = mkbin(e, "*", x, P_MUL);
			continue;
		}
		if (e.lit && mat("percent of|%")) {
			if (W[wi - 1].k == W_SYM && !mat("of")) {
				wi = s;
				break;
			}
			x = parse_expr(P_UNARY);
			e = mkbin(mkbin(x, "*", e, P_MUL), "/", num("100"), P_MUL);
			continue;
		}
		if (e.lit && mat("percent")) {
			e = mkbin(e, "/", num("100"), P_MUL);
			continue;
		}
		break;
	}
	return e;
}

static struct ex
parse_expr(int minprec)
{
	struct ex l, r, c;
	const struct bop *o;
	struct decl d;
	int s, k;

	l = parse_postfix(parse_prefix(minprec), minprec);
	for (;;) {
		s = wi;
		/* in a condition: "x is positive", "n is not even" */
		if (incond && P_CMP >= minprec && mat("is|are")) {
			int neg = mat("not");
			const char *c = NULL;
			if (mat("positive"))
				c = "%s > 0";
			else if (mat("negative"))
				c = "%s < 0";
			else if (mat("nonnegative|non-negative"))
				c = "%s >= 0";
			else if (mat("nonpositive|non-positive"))
				c = "%s <= 0";
			else if (mat("zero"))
				c = "%s == 0";
			else if (mat("nonzero|non-zero"))
				c = "%s != 0";
			else if (mat("even"))
				c = "%s %% 2 == 0";
			else if (mat("odd"))
				c = "%s %% 2 != 0";
			if (c) {
				l = mkatom(S(c, paren(l, P_MUL, 1)), l.ity, 0);
				l.prec = P_CMP;
				l.isbool = 1;
				if (neg)
					l = mkun("!", l);
				continue;
			}
			wi = s;
		}
		if (issym(",")) {
			wi++;
			o = peek_bop();
			if (o == NULL || (o->kind != B_NORM && o->kind != B_SWAP &&
			    o->kind != B_OVER) || o->prec < P_CMP) {
				wi = s;
				break;
			}
			l.closed = 1;
		}
		o = match_bop();
		if (o == NULL || o->prec < minprec) {
			wi = s;
			break;
		}
		switch (o->kind) {
		case B_IF:
			incond++;
			c = parse_expr(P_OR);
			incond--;
			if (!c.isbool) {
				wi = s;
				return l;
			}
			acc_sym(",");
			if (mat("else|otherwise") || mat("and ?is otherwise")) {
				mat("?it ?is");
				r = parse_expr(P_TERN);
				l = mkatom(S("%s ? %s : %s", paren(c, P_TERN, 1),
				    paren(l, P_TERN, 1), paren(r, P_TERN, 0)),
				    icomb(l.ity, r.ity), 0);
				l.prec = P_TERN;
			} else {
				l = mkatom(S("{ %s if %s }", l.s, c.s), l.ity, 0);
			}
			continue;
		case B_CAST:
			memset(&d, 0, sizeof d);
			mat("?a|an");
			parse_type(&d);
			l = mkcall(d.ty, l);
			l.ity = d.ity;
			continue;
		case B_BETWEEN:
			r = parse_expr(P_ADD);
			if (!mat("and") && !acc_sym(".."))
				nfail("expected \"and\" after \"between %s\"", r.s);
			c = parse_expr(P_ADD);
			l = mkatom(S("%s in %s..%s", paren(l, P_ADD, 0), r.s, c.s),
			    l.ity, 0);
			l.prec = P_CMP;
			l.isbool = 1;
			continue;
		case B_IN:
			mat("?the range|interval");
			acc_sym("[");
			r = parse_expr(P_ADD);
			if (!acc_sym("..") && !mat("to|through") && !acc_sym(","))
				nfail("expected a range like 0..9");
			c = parse_expr(P_ADD);
			acc_sym("]");
			l = mkatom(S("%s in %s..%s", paren(l, P_ADD, 0), r.s, c.s),
			    l.ity, 0);
			l.prec = P_CMP;
			l.isbool = 1;
			continue;
		case B_AND:
		case B_OR:
			r = parse_expr(o->prec + 1);
			if (l.isbool || r.isbool) {
				l = mkbin(l, o->kind == B_AND ? "&&" : "||", r, o->prec);
				l.isbool = 1;
			} else {
				static const char *const ao[] = {"bitwise (& and |)",
					"logical (&& and ||)"};
				k = clarify("andor", "\"and\"/\"or\" between numbers",
				    ao, 2, 0, "say \"bitwise and\" or \"logical and\"");
				if (k)
					l = mkbin(l, o->kind == B_AND ? "&&" : "||", r,
					    o->prec);
				else
					l = mkbin(l, o->kind == B_AND ? "&" : "|", r,
					    o->kind == B_AND ? P_BAND : P_BOR);
			}
			continue;
		case B_SWAP:
			r = parse_expr(o->prec + 1);
			l = mkbin(r, o->op, l, o->prec);
			continue;
		default:
			r = parse_expr(o->prec + 1);
			l = mkbin(l, o->op, r, o->prec);
			continue;
		}
	}
	return l;
}

/* ---- declarations and clauses ---- */

static int
parse_varlist(char names[][32], int *n)
{
	int s = wi;

	*n = 0;
	mat("?both|all|each|every|any");
	for (;;) {
		if (!varlike(wi) || user_fn(W[wi].o))
			break;
		snprintf(names[*n], 32, "%s", W[wi].o);
		wi++;
		if (++*n == 8)
			break;
		if (issym(",") && varlike(wi + 1)) {
			wi++;
			continue;
		}
		if (issym(",") && isw(wi + 1, "and") && varlike(wi + 2)) {
			wi += 2;
			continue;
		}
		if (isw(wi, "and") && varlike(wi + 1)) {
			wi++;
			continue;
		}
		break;
	}
	if (*n == 0) {
		wi = s;
		return 0;
	}
	return 1;
}

static int
is_number(const char *s)
{
	char *e;

	if (!*s)
		return 0;
	(void)strtod(s, &e);
	return *e == 0;
}

static int
is_int(const char *s)
{
	char *e;

	if (!*s)
		return 0;
	(void)strtoll(s, &e, 10);
	return *e == 0;
}

static void
dcond(struct decl *d, const char *fmt, const char *arg)
{
	if (d->ncond < 4) {
		snprintf(d->cond[d->ncond++], sizeof d->cond[0], fmt, "%s", arg);
		d->any = 1;
	}
}

/* a bound: "x < 10" is x <= 9 on integers */
static void
set_bound(struct decl *d, int hi, struct ex b, int strict)
{
	char *dst = hi ? d->hi : d->lo;

	d->any = 1;
	if (!strict && b.hv && !is_number(b.s))
		d->bc |= hi ? 2 : 1;
	if (strict) {
		if (is_int(b.s)) {
			snprintf(dst, 48, "%lld", strtoll(b.s, NULL, 10) +
			    (hi ? -1 : 1));
			if (d->ity == 1 || !d->ty[0])
				return;
			dst[0] = 0;
		}
		dcond(d, hi ? "%s < %s" : "%s > %s", paren(b, P_SHIFT, 0));
		return;
	}
	snprintf(dst, 48, "%s", b.s);
}

static int
parse_rangemod(struct decl *d)
{
	struct ex a, b;
	int s = wi, br;

	if (mat("between") || mat("from") || mat("ranging|going|running|varying "
	    "from") || mat("in|within ?the range|interval ?of|from") ||
	    mat("in")) {
		br = acc_sym("[");
		a = parse_expr(P_ADD);
		if (!(acc_sym("..") || mat("and|to|through|thru|until|till") ||
		    (br && acc_sym(",")))) {
			wi = s;
			return 0;
		}
		b = parse_expr(P_ADD);
		if (br)
			acc_sym("]");
		mat("?inclusive|inclusively");
		if (a.hv && b.hv && a.v > b.v) {
			struct ex t = a;
			a = b;
			b = t;
		}
		set_bound(d, 0, a, 0);
		set_bound(d, 1, b, mat("exclusive"));
		return 1;
	}
	if (mat("less than or equal to") || mat("smaller than or equal to") ||
	    mat("up to") || mat("at most") || mat("no more|greater|larger than") ||
	    mat("not more|greater|larger than") || mat("<=")) {
		set_bound(d, 1, parse_expr(P_ADD), 0);
		return 1;
	}
	if (mat("less|smaller|fewer than") || mat("below|under|<")) {
		set_bound(d, 1, parse_expr(P_ADD), 1);
		return 1;
	}
	if (mat("greater than or equal to") || mat("at least") ||
	    mat("no less|smaller than") || mat("not less|smaller than") ||
	    mat(">=")) {
		set_bound(d, 0, parse_expr(P_ADD), 0);
		return 1;
	}
	if (mat("greater|more|bigger|larger than") || mat("above|over|>")) {
		set_bound(d, 0, parse_expr(P_ADD), 1);
		return 1;
	}
	if (mat("nonzero|non-zero") || mat("not zero") || mat("!= 0")) {
		d->adj |= A_NONZERO;
		d->any = 1;
		return 1;
	}
	return 0;
}

static int
parse_adj(struct decl *d)
{
	if (mat("strictly ?positive") || mat("positive")) {
		d->adj |= A_POS;
	} else if (mat("negative")) {
		d->adj |= A_NEG;
	} else if (mat("nonnegative|non-negative") || mat("not negative")) {
		d->adj |= A_NONNEG;
	} else if (mat("nonzero|non-zero")) {
		d->adj |= A_NONZERO;
	} else if (mat("even")) {
		d->adj |= A_EVEN;
	} else if (mat("odd")) {
		d->adj |= A_ODD;
	} else
		return 0;
	d->any = 1;
	return 1;
}

/* what follows "x is" / "x:" / "x ranges": type, bounds, or a value */
static int
parse_descr(struct decl *d)
{
	int verb, s = wi;
	struct ex e;

	verb = acc_sym(":") || acc_sym("=") || mat("is|are|be|being|was|were|"
	    "equals|ranges|range|goes|go|runs|run|varies|vary|lies|lie") ||
	    mat("has|have ?type") || mat("of type");
	mat("?a|an|both|all|each|any|some|always");
	mat("?of type");
	while (parse_adj(d))
		;
	parse_type(d);
	while (parse_adj(d))
		;
	while (parse_rangemod(d) || parse_adj(d))
		mat("?and ?also");
	if (!d->any && verb && !isw(s, "ranges") && !isw(s, "range")) {
		incond++;
		e = parse_expr(P_BOR);
		incond--;
		if (!e.isbool) {
			snprintf(d->val, sizeof d->val, "%s", e.s);
			d->any = 1;
		}
	}
	if (!d->any)
		wi = s;
	return d->any;
}

static void
apply_decl(struct var *v, const struct decl *d)
{
	int k;

	v->decl = 1;
	if (d->ty[0]) {
		snprintf(v->ty, sizeof v->ty, "%s", d->ty);
		v->ity = d->ity;
	}
	if (d->lo[0]) {
		snprintf(v->lo, sizeof v->lo, "%s", d->lo);
		v->bc = (v->bc & ~1) | (d->bc & 1);
	}
	if (d->hi[0]) {
		snprintf(v->hi, sizeof v->hi, "%s", d->hi);
		v->bc = (v->bc & ~2) | (d->bc & 2);
	}
	for (k = 0; k < d->ncond && v->ncond < 4; k++)
		snprintf(v->cond[v->ncond++], sizeof v->cond[0], d->cond[k],
		    v->name);
	v->adj |= d->adj;
	if (d->val[0])
		snprintf(v->val, sizeof v->val, "%s", d->val);
}

static int session_decl;   /* "assume x is ...": also keep it */

static void
declare(char names[][32], int n, const struct decl *d)
{
	int k, j;

	for (k = 0; k < n; k++) {
		apply_decl(var_get(names[k]), d);
		if (session_decl && !dry) {
			for (j = 0; j < nsv; j++)
				if (!strcmp(SV[j].name, names[k]))
					break;
			if (j == nsv && nsv < MAXV) {
				memset(&SV[nsv], 0, sizeof SV[0]);
				snprintf(SV[nsv].name, sizeof SV[0].name, "%s", names[k]);
				nsv++;
			}
			if (j < MAXV) {
				apply_decl(&SV[j], d);
				SV[j].decl = 0;
				SV[j].val[0] = 0;
			}
		}
	}
}

/* "where x is an unsigned byte", "for every u8 x", "x between 1 and 12" */
static int
parse_clause(void)
{
	char names[8][32];
	struct decl d;
	struct ex e;
	int s = wi, st, s2, n;

	st = mat("for|where|with|given|assuming|when|whenever|on|letting") ||
	    mat("such|so that") || mat("over all|every|each|any") ||
	    mat("let") || mat("and");
	/* "on [a, b]", "from a to b", "between a and b": the only variable */
	{
		struct ex a, b;
		int s3 = wi, btw;
		if (!st)
			mat("from|over|across|within|in|between|along");
		btw = isw(wi - 1, "between");
		if (st || wi > s3) {
			mat("?the interval|range|domain|segment");
			if (mat("between"))
				btw = 1;
			if (const_interval(&a, &b, btw)) {
				Q.ilo = a.s;
				Q.ihi = b.s;
				return 1;
			}
		}
		wi = st ? s3 : s;
	}
	mat("?all|every|each|any|some|both");
	if ((isw(wi, "a") || isw(wi, "an")) && peek_type())
		wi++;
	s2 = wi;
	memset(&d, 0, sizeof d);
	if (parse_type(&d) && parse_varlist(names, &n)) {
		while (parse_rangemod(&d) || parse_adj(&d))
			;
		declare(names, n, &d);
		return 1;
	}
	wi = s2;
	memset(&d, 0, sizeof d);
	if (parse_varlist(names, &n)) {
		if (parse_descr(&d)) {
			declare(names, n, &d);
			return 1;
		}
		wi = s2;
	}
	if (st && !isw(s, "and")) {
		incond++;
		e = parse_expr(P_OR);
		incond--;
		if (e.isbool && Q.nscond < 8) {
			Q.scond[Q.nscond++] = e.s;
			return 1;
		}
	}
	wi = s;
	return 0;
}

static const struct {
	const char *pat, *basis;
} basis_names[] = {
	{"arm|arm64|aarch64|a64|armv8 with|and ?the sha3|sha", "a64sha"},
	{"a64sha|sha3", "a64sha"},
	{"arm|arm64|aarch64|a64|armv8", "a64"},
	{"x86|x86-64|x64|amd64|bmi|bmi1|x86bmi", "x86bmi"},
	{"nand", "nand"},
	{"spirv|spir-v|vulkan", "spirv"},
	{NULL, NULL}
};

static int
parse_basis(void)
{
	int s = wi, k;

	if (mat("basis") && W[wi].k == W_WORD &&
	    basis_find(sym_cstr(W[wi].s))) {
		snprintf(Q.basis, sizeof Q.basis, "%s", W[wi].s);
		wi++;
		return 1;
	}
	wi = s;
	mat("?only");
	for (k = 0; basis_names[k].pat; k++)
		if (mat(basis_names[k].pat)) {
			snprintf(Q.basis, sizeof Q.basis, "%s", basis_names[k].basis);
			mat("?basis|gates|instructions|ops|operations|logic|isa|cpu|"
			    "cpus|architecture|instruction");
			mat("?set");
			return 1;
		}
	if (W[wi].k == W_WORD && basis_find(sym_cstr(W[wi].s)) &&
	    peek("? basis|gates")) {
		snprintf(Q.basis, sizeof Q.basis, "%s", W[wi].s);
		wi++;
		mat("basis|gates");
		return 1;
	}
	wi = s;
	return 0;
}

static int
parse_model(void)
{
	int s = wi;
	char name[48];

	mat("?the|a|an");
	if (W[wi].k == W_WORD) {
		snprintf(name, sizeof name, "%s", W[wi].s);
		if (W[wi + 1].k == W_NUM && !model_find(sym_cstr(name)))
			snprintf(name, sizeof name, "%s%s", W[wi].s, W[wi + 1].s);
		if (model_find(sym_cstr(name))) {
			wi += strcmp(name, W[wi].s) ? 2 : 1;
			mat("?cost ?model|gpu|card");
			snprintf(Q.model, sizeof Q.model, "%s", name);
			return 1;
		}
	}
	wi = s;
	return 0;
}

static int
small_number(int *n)
{
	int64_t v;
	int s = wi;

	if (W[wi].k == W_NUM && is_int(W[wi].s)) {
		*n = atoi(W[wi].s);
		wi++;
		return 1;
	}
	if (number_words(&v)) {
		*n = (int)v;
		return 1;
	}
	wi = s;
	return 0;
}

static const char *subst_call(const char *s, const char *from,
    const char *to);
static int osa(const char *a, const char *b, int max);
static int spell_limit(size_t n);

static int
parse_option(void)
{
	int s = wi, n;

	/* critical points: "w/o crit. points", "nocrit", "yes critical" */
	if (mat("nocrit") || ((mat("without|ignoring|ignore|skipping|skip|"
	    "excluding|sans|no|not") || mat("w / o")) &&
	    (mat("?any|considering|consideration|regard|refining|refinement|using"),
	    mat("?for|of|to|at|by"), mat("?the|any"),
	    mat("critical|crit|stationary|turning ?points|point|pts|values")))) {
		Q.nocrit = 1;
		Q.crit = 0;
		return 1;
	}
	wi = s;
	if (mat("?yes|with|use|using|consider|considering|keep|keeping|include|"
	    "including|exact|preserve|preserving ?at|the|its critical|crit|"
	    "stationary ?points|point|pts|values") && (isw(wi - 1, "critical") ||
	    isw(wi - 1, "crit") || isw(wi - 1, "points") ||
	    isw(wi - 1, "point") || isw(wi - 1, "pts") ||
	    isw(wi - 1, "stationary") || isw(wi - 1, "values"))) {
		Q.crit = 1;
		Q.nocrit = 0;
		return 1;
	}
	wi = s;
	/* approximation strategy: "with cheby poly", "using minimax" */
	mat("?with|using|via|by|use|try|start ?with ?a|an|the");
	mat("?initial|first");
	if (mat("chebyshev|cheby|cheb|tschebyscheff|tchebychev")) {
		if (mat("economization|economisation|econ|economized|economised"))
			strcpy(Q.method, "econ");
		else
			strcpy(Q.method, "cheb");
		mat("?polynomial|polynomials|poly|polys|interpolation|series|nodes|"
		    "approximation|approximations|expansion|fit|fits");
		mat("?interpolation|approximation|fit|strategy|method");
		return 1;
	}
	if (mat("minimax|remez")) {
		strcpy(Q.method, "minimax");
		mat("?polynomial|polynomials|poly|polys|approximation|fit|exchange|"
		    "algorithm|strategy|method");
		return 1;
	}
	if (mat("taylor ?series|polynomial|poly|expansion")) {
		strcpy(Q.method, "econ");
		Q.why = "Taylor: using the Chebyshev economisation of the Taylor "
		    "series (method econ)";
		return 1;
	}
	if (mat("polynomial|polynomials|poly|polys ?approximation|fit") &&
	    !varlike(wi)) {
		if (!Q.method[0])
			strcpy(Q.method, "minimax");
		return 1;
	}
	wi = s;
	/* approx: degree and format */
	if (mat("?of|with|at|to ?a|polynomial degree|deg|degrees") &&
	    W[wi].k == W_NUM) {
		int d1 = atoi(W[wi].s), d2;
		wi++;
		if ((acc_sym("..") || mat("to|through") || acc_sym("-")) &&
		    small_number(&d2))
			snprintf(Q.degree, sizeof Q.degree, "%d..%d", d1, d2);
		else
			snprintf(Q.degree, sizeof Q.degree, "%d", d1);
		return 1;
	}
	wi = s;
	/* approx: accuracy, "within 1e-6", "error of 1e-6", "2 ulp" */
	{
		int kw = 0;
		if (mat("within|below|under"))
			kw = 1;
		else if (mat("?to|with|at ?an|a|the ?max|maximum|absolute|relative "
		    "error|accuracy|tolerance|precision|tol")) {
			kw = 1;
			mat("?of|below|under|at|within");
		} else {
			wi = s;
			/* "to 2 ulp": unambiguous with the unit after it */
			if (!mat("?to|at|in"))
				wi = s;
		}
		if (W[wi].k == W_NUM) {
			const char *num = W[wi].s;
			wi++;
			if (mat("ulp|ulps")) {
				snprintf(Q.acc, sizeof Q.acc, "ulp %s", num);
				return 1;
			}
			if (kw) {
				snprintf(Q.acc, sizeof Q.acc, "error %s", num);
				return 1;
			}
		}
	}
	wi = s;
	/* Lambert W's branch, said after the expression */
	if (mat("?on|in|using|with|for ?the principal|main|upper|lower|negative|"
	    "secondary ?real branch|branches ?of ?w|lambert ?w")) {
		int lower = 0, k;
		for (k = s; k < wi; k++)
			if (isw(k, "lower") || isw(k, "negative") ||
			    isw(k, "secondary"))
				lower = 1;
		if (Q.a.has && Q.a.e.s && !lower) {
			Q.why = "principal branch: W (the -1 branch is Wm1)";
			return 1;
		}
		if (Q.a.has && Q.a.e.s && lower) {
			Q.a.e.s = subst_call(Q.a.e.s, "W", "Wm1");
			Q.why = "lower branch: Wm1 (the principal branch is W)";
			return 1;
		}
	}
	wi = s;
	/* approx: special endpoints kept exact or not */
	if (mat("?keep|keeping|preserve|preserving|with|exact ?the|its|both "
	    "endpoints|ends|edges ?exact|fixed")) {
		Q.ends = 1;
		return 1;
	}
	wi = s;
	if (mat("free|loose|nokeep ?the ?endpoints|ends|edges")) {
		Q.ends = -1;
		return 1;
	}
	wi = s;
	/* a bare whole number answering "which degree?" */
	if (Q.cmd == C_APPROX && W[wi].k == W_NUM && is_int(W[wi].s) &&
	    (W[wi + 1].k == W_END || (W[wi + 1].k == W_SYM &&
	    !strcmp(W[wi + 1].s, ",")))) {
		snprintf(Q.degree, sizeof Q.degree, "%s", W[wi].s);
		wi++;
		return 1;
	}
	wi = s;
	if (mat("?as|in|using|with|for|at f32|f64|float|floats|single|double|"
	    "doubles|real|reals ?precision") && (isw(wi - 1, "precision") ||
	    !varlike(wi))) {
		const char *w = isw(wi - 1, "precision") ? W[wi - 2].s : W[wi - 1].s;
		strcpy(Q.aty, !strcmp(w, "f64") || !strncmp(w, "double", 6) ? "f64" :
		    !strncmp(w, "real", 4) ? "real" : "f32");
		return 1;
	}
	wi = s;
	if ((mat("in|with|using|over") || 1) && small_number(&n) &&
	    mat("steps|step")) {
		Q.steps = n;
		return 1;
	}
	wi = s;
	if (mat("steps") && small_number(&n)) {
		Q.steps = n;
		return 1;
	}
	wi = s;
	if (mat("with respect to") || mat("wrt") || mat("w . r . t ?.")) {
		if (!varlike(wi))
			nfail("with respect to which variable?");
		snprintf(Q.wrt, sizeof Q.wrt, "%s", W[wi].o);
		wi++;
		return 1;
	}
	if (mat("?and ?then ?also as|in c ?code|source|function") ||
	    mat("with|and ?the|a c code|source|function|version") ||
	    mat("and|then emit|write|give|show|print ?me ?it ?as ?the|a c ?code|"
	    "source|function|version")) {
		Q.emit_c = 1;
		return 1;
	}
	if (mat("using|with|on|in|for|over|targeting|targetting|via|from") &&
	    (mat("?the|a|an") , 1) && parse_basis())
		return 1;
	wi = s;
	if (Q.cmd == C_GATES && parse_basis())
		return 1;
	if (mat("on|under|with|using|for") && parse_model())
		return 1;
	wi = s;
	if ((mat("with ?a ?gate ?budget|limit|cost of") || mat("max|maximum ?cost") ||
	    mat("at most") || mat("up to") || mat("no more than")) &&
	    small_number(&n)) {
		if (mat("gates|gate|steps|step|operations|ops|instructions|cost") ||
		    !isw(s, "at")) {
			Q.maxc = n;
			return 1;
		}
	}
	wi = s;
	if (mat("?the first|top") && small_number(&n) &&
	    mat("results|implementations|options|candidates|alternatives|ways|"
	    "rows|values|entries|lines|programs|versions")) {
		Q.count = n;
		return 1;
	}
	wi = s;
	if (mat("minimizing|minimising|using|by|with|under|for ?the ?max|maximum|"
	    "worst-case|worst ?case relative error") || mat("metric maxrel")) {
		Q.metric = 0;
		return 1;
	}
	wi = s;
	if (mat("minimizing|minimising|using|by|with|under|for ?the ?max|maximum|"
	    "worst-case|worst ?case absolute error") || mat("metric maxabs")) {
		Q.metric = 1;
		return 1;
	}
	wi = s;
	if (mat("minimizing|minimising|using|by|with|under|for ?the rms|"
	    "root-mean-square ?error") || mat("minimizing|minimising|using|by|"
	    "with|under|for ?the root mean square|squared ?error") ||
	    mat("metric rms")) {
		Q.metric = 2;
		return 1;
	}
	wi = s;
	if (mat("?and call|name it|this|that") ||
	    mat("?and call|name the result|function") ||
	    mat("calling|naming it")) {
		if (W[wi].k != W_WORD)
			nfail("call it what?");
		snprintf(Q.defname, sizeof Q.defname, "%s", W[wi].o);
		wi++;
		return 1;
	}
	wi = s;
	return 0;
}

static void
parse_tail(void)
{
	for (;;) {
		if (W[wi].k == W_END)
			return;
		if (acc_sym(",") || acc_sym(";"))
			continue;
		/* options joined by "+", "&", "and", "plus": "crit + cheby poly" */
		if ((acc_sym("+") || acc_sym("&") || mat("and|plus|also")) &&
		    W[wi].k != W_END)
			continue;
		if (parse_option())
			continue;
		{
			struct ex a, b;
			if (const_interval(&a, &b, 0)) {
				Q.ilo = a.s;
				Q.ihi = b.s;
				continue;
			}
		}
		if (parse_clause())
			continue;
		if (mat("please|thanks|again|too|now") || mat("thank you") ||
		    mat("for me") || mat("as well") || mat("if possible") ||
		    mat("if you can"))
			continue;
		return;
	}
}

/* "x: u8 in 0..10" for parameter k of f */
static const char *
fparam_decl(struct func *f, int k)
{
	const struct param *p = &f->p[k];
	const char *s = S("%s: %s", sym_str(p->name), ty_str(p->ty));

	if (p->ranged) {
		if (TY_FP(p->ty))
			s = S("%s in %.17g..%.17g", s, p->rlo, p->rhi);
		else
			s = S("%s in %lld..%lld", s, (long long)p->lo,
			    (long long)p->hi);
	}
	return s;
}

/* ---- targets ---- */

static void
named_target(struct tgt *t, const char *name)
{
	struct func *f = user_fn(name);

	memset(t, 0, sizeof *t);
	t->has = t->named = 1;
	snprintf(t->fn, sizeof t->fn, "%s", name);
	if (f)
		seed_params(f);
}

static void
target(struct tgt *t, int minprec)
{
	int s = wi;
	const struct bop *o;

	memset(t, 0, sizeof *t);
	if (mat("it|this|that|them ?function|expression|formula|one") ||
	    mat("the|that|this same|last|previous ?function|expression|formula|"
	    "one")) {
		const char *n = last_name();
		if (n == NULL)
			nfail("\"%s\" has nothing to refer to yet", W[s].o);
		if (!peek_bop() && !issym("(") && !peek("of|at|applied")) {
			named_target(t, n);
			return;
		}
		wi = s;
	}
	if (W[wi].k == W_WORD && user_fn(W[wi].o) && !var_find(W[wi].o)) {
		wi++;
		o = peek_bop();
		if (!issym("(") && !peek("of|at|applied|evaluated") &&
		    (o == NULL || o->prec < minprec)) {
			named_target(t, W[wi - 1].o);
			return;
		}
		wi = s;
	}
	t->e = parse_expr(minprec);
	t->has = 1;
}

/* ---- plots (plot.c): "plot sin x and cos x from 0 to pi" ---- */

static const char *subst_ident(const char *s, const char *name,
    const char *val);
static const char *annotate(const char *s, const char *name, const char *ty);
static const char *value_text(const char *val);
static const char *fn_used;

static void
popt_add(const char *o)
{
	size_t n = strlen(Q.popt);

	snprintf(Q.popt + n, sizeof Q.popt - n, " %s", o);
}

/* a constant expression, or NULL (nothing consumed) */
static const char *
plot_const(void)
{
	jmp_buf saved;
	int s = wi, ok = 0, nvs = nv, nos = norder;
	struct ex e;

	memset(&e, 0, sizeof e);
	memcpy(saved, nl_jmp, sizeof saved);
	memcpy(Vsave, V, sizeof V);
	quiet++;
	if (W[wi].k != W_END && setjmp(nl_jmp) == 0) {
		e = parse_expr(P_ADD);
		ok = e.hv;
	}
	quiet--;
	memcpy(nl_jmp, saved, sizeof saved);
	if (!ok) {
		wi = s;
		memcpy(V, Vsave, sizeof V);
		nv = nvs;
		norder = nos;
		return NULL;
	}
	return e.s;
}

/* a variable name as a word: x, t, theta */
static const char *
plot_var(void)
{
	if (W[wi].k == W_WORD && (strlen(W[wi].s) <= 2 || var_find(W[wi].s) ||
	    !strcmp(W[wi].s, "theta") || !strcmp(W[wi].s, "time")) &&
	    !reserved_word(W[wi].s)) {
		const char *n = !strcmp(W[wi].s, "time") ? "t" : W[wi].o;
		wi++;
		return S("%s", n);
	}
	return NULL;
}

static int
plot_option(void)
{
	int s = wi, n;
	const char *c, *v;

	if (mat("nofile")) {	/* tests: the report without the file */
		popt_add("nofile");
		return 1;
	}
	mat("?and|with|plus|also|then|but");
	/* the view */
	if (mat("?in|as|into ?a|an|the 3 d ?plot|plots|graph|view|surface|"
	    "picture|image|render|rendering") ||
	    mat("?in|as ?a|an|the 3 - d ?plot|graph|view|surface") ||
	    mat("?in|as ?a|an|the three-dimensional|3-dimensional|three-d ?plot|"
	    "graph|view|surface|picture") ||
	    mat("in three|3 dimensions") ||
	    mat("?as|in ?a|an|the surface|surf ?plot|plots|graph|view|picture")) {
		if (Q.pmode == 0)
			Q.pmode = 1;
		return 1;
	}
	if (mat("?as|in|with ?a|an|the contour|contours ?plot|plots|graph|map|"
	    "lines|view|diagram") ||
	    mat("?as|with ?a|an|the level curves|sets|lines")) {
		popt_add("contour");
		return 1;
	}
	if (mat("?as|in ?a|an|the heatmap|heatmaps|heat-map|density|colormap|"
	    "colourmap ?plot|map|graph|view|image") ||
	    mat("?as|in ?a|an|the heat|color|colour map|plot|image")) {
		popt_add("heatmap");
		return 1;
	}
	if (mat("?as ?a|an|the wireframe|wire-frame|mesh ?plot|graph|surface|view|"
	    "model") || mat("?as ?a|an|the wire frame|mesh")) {
		popt_add("wireframe");
		return 1;
	}
	if (mat("?in|as|using ?a|an|the polar ?plot|graph|coordinates|form|"
	    "curve|coordinate")) {
		popt_add("polar");
		return 1;
	}
	if (mat("?as ?a|an|the vector|gradient field|fields ?plot|graph")) {
		return 1;	/* a tuple (P, Q) of two variables is one */
	}
	if (mat("?as ?a|an|the slope|direction field|fields ?plot|graph")) {
		popt_add("field");
		return 1;
	}
	if (mat("?as|in|using ?a|an|the parallel coordinates|coordinate ?plot|"
	    "graph|view|chart")) {
		popt_add("parallel");
		return 1;
	}
	if (mat("?as|in ?a|an|the small multiples") ||
	    mat("?as|in ?a|an|the slices|panels") ||
	    mat("?as|in ?a|an|the grid of slices|panels|plots")) {
		popt_add("slices");
		return 1;
	}
	if (mat("?and ?make|let ?it rotating|rotate|rotates|spinning|spin|spins|"
	    "turning|turn|revolving ?around|round ?the ?vertical|z ?axis") ||
	    mat("?as ?a|an|the turntable|rotation|spinning|rotating ?animation|"
	    "gif|view")) {
		popt_add("rotate");
		return 1;
	}
	/* animation: "animated over t", "over time", "as t goes from 0 to 1" */
	if (mat("animated|animate|animating|animation|moving|evolving|frames ?it "
	    "?over|along|through|in|with|by|for ?the ?variable|parameter")) {
		if (mat("time"))
			popt_add("animate t");
		else if ((v = plot_var()) != NULL)
			popt_add(S("animate %s", v));
		else
			popt_add("animate");
		return 1;
	}
	if (mat("over|through|in ?the time")) {
		popt_add("animate t");
		return 1;
	}
	{
		int s2 = wi;
		if (mat("as|while|when") && (v = plot_var()) != NULL &&
		    mat("varies|changes|goes|runs|increases|moves|grows|ranges|"
		    "evolves|sweeps")) {
			struct ex a, b;
			popt_add(S("animate %s", v));
			mat("?from|over|across|between|in");
			if (const_interval(&a, &b, 1))
				popt_add(S("%s in %s..%s", v, a.s, b.s));
			return 1;
		}
		wi = s2;
	}
	/* frames */
	if (mat("?at|with ?a|an ?frame ?rate ?of") && W[wi].k == W_NUM &&
	    (isw(wi + 1, "fps") || (isw(wi + 1, "frames") && isw(wi + 2, "per")))) {
		n = atoi(W[wi].s);
		wi++;
		mat("fps") || mat("frames per second");
		popt_add(S("fps %d", n));
		return 1;
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	if (mat("?for|lasting|over") && W[wi].k == W_NUM &&
	    (isw(wi + 1, "seconds") || isw(wi + 1, "second") ||
	    isw(wi + 1, "secs") || isw(wi + 1, "sec") || isw(wi + 1, "s"))) {
		popt_add(S("seconds %s", W[wi].s));
		wi += 2;
		return 1;
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	if (mat("?in|with|using") && W[wi].k == W_NUM && isw(wi + 1, "frames")) {
		popt_add(S("frames %s", W[wi].s));
		wi += 2;
		return 1;
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	if (mat("?with|using") && W[wi].k == W_NUM && (isw(wi + 1, "samples") ||
	    isw(wi + 1, "points"))) {
		popt_add(S("points %s", W[wi].s));
		wi += 2;
		return 1;
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	/* derivative and tangent */
	if (mat("?together|along|alongside ?with ?its|the|their first|1st "
	    "derivative|derivatives") ||
	    mat("?together|along|alongside ?with ?its|the|their derivative|"
	    "derivatives ?too") ||
	    mat("?together|along|alongside ?with ?its|the|their slope function")) {
		popt_add("deriv");
		return 1;
	}
	if (mat("?together|along ?with ?its|the|a tangent ?line ?at|through|in "
	    "?the ?point")) {
		if (W[wi].k == W_WORD && W[wi + 1].k == W_SYM &&
		    (!strcmp(W[wi + 1].s, "=") || !strcmp(W[wi + 1].s, "==")))
			wi += 2;
		if ((c = plot_const()) == NULL)
			nfail("the tangent at which point? (tangent at 2)");
		popt_add(S("tangent %s", c));
		return 1;
	}
	/* log scales */
	wi = s;
	mat("?and|with|plus|also|then|but");
	mat("?on|with|using|in ?a|an|the");
	if (mat("log-log|loglog|double-log|double-logarithmic ?scale|scales|axes|"
	    "axis|plot|graph") || mat("log log ?scale|scales|axes|plot")) {
		popt_add("loglog");
		return 1;
	}
	if (mat("log|logarithmic|semilog|semi-log|log-scale|log-scaled")) {
		if (mat("x|x-axis|horizontal ?scale|axis")) {
			popt_add("logx");
			return 1;
		}
		if (mat("y|y-axis|vertical ?scale|axis")) {
			popt_add("logy");
			return 1;
		}
		if (mat("axes|scales")) {
			popt_add("loglog");
			return 1;
		}
		if (mat("scale|axis|plot|graph|scaling")) {
			if (!dry)
				printf("  note: a log scale read as a logarithmic y axis "
				    "(say log x axis, or log-log, for the others)\n");
			popt_add("logy");
			return 1;
		}
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	if ((isw(wi, "x") || isw(wi, "y")) && (mat("x on|in ?a|an|the log|"
	    "logarithmic scale|axis") || mat("y on|in ?a|an|the log|logarithmic "
	    "scale|axis"))) {
		popt_add(isw(s, "x") || isw(s + 1, "x") ? "logx" : "logy");
		return 1;
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	/* neighbourhood: "as a neighbourhood", "around 2", "near its interesting part" */
	if (mat("as ?a|an|the|its neighbourhood|neighborhood|neighbourhoods|"
	    "neighborhoods|surroundings")) {
		if (Q.pmode != 1)
			Q.pmode = 2;
		mat("?of|around|at|near");
		if ((c = plot_const()) != NULL)
			Q.pat = c;
		return 1;
	}
	if (mat("near|around|about|nearby|at") || mat("close to") ||
	    mat("in|within the neighbourhood|neighborhood|vicinity of") ||
	    mat("in|within a|the small ?neighbourhood|neighborhood|window around")) {
		if (mat("its|the interesting|interestin|important|relevant|main "
		    "part|region|bit|parts|features|bits|stuff|area|range")) {
			if (Q.pmode != 1)
				Q.pmode = 2;
			return 1;
		}
		mat("?the ?point");
		if (W[wi].k == W_WORD && W[wi + 1].k == W_SYM &&
		    (!strcmp(W[wi + 1].s, "=") || !strcmp(W[wi + 1].s, "==")))
			wi += 2;
		if (acc_sym("(")) {
			const char *a, *b;
			if ((a = plot_const()) != NULL && acc_sym(",") &&
			    (b = plot_const()) != NULL && acc_sym(")")) {
				Q.pat = S("(%s, %s)", a, b);
				if (Q.pmode != 1)
					Q.pmode = 2;
				return 1;
			}
		} else if ((c = plot_const()) != NULL && !isw(wi, "fps") &&
		    !isw(wi, "frames")) {
			Q.pat = c;
			if (Q.pmode != 1)
				Q.pmode = 2;
			return 1;
		}
	}
	wi = s;
	mat("?and|with|plus|also|then|but");
	if (mat("where ?it is|it's|its interesting") ||
	    mat("?on|over|in|at|showing ?its|the interesting|important|relevant "
	    "part|region|bit|parts|features|bits|stuff|area|range") ||
	    mat("?with|using automatic|auto|implicit|natural|sensible|good|nice|"
	    "default ?bounds|range|ranges|window|limits|axes")) {
		if (Q.pmode != 1)
			Q.pmode = 2;
		return 1;
	}
	/* the file: "save it as wave", "to wave.gif" */
	wi = s;
	mat("?and|with|plus|also|then|but");
	if (mat("save|write|store|put ?it|the|that ?plot|image|picture|gif|graph "
	    "?as|to|into|in") || mat("into|named|called")) {
		if (W[wi].k == W_WORD) {
			popt_add(S("to %s", W[wi].o));
			wi++;
			if (issym(".") && isw(wi + 1, "gif"))
				wi += 2;
			return 1;
		}
	}
	wi = s;
	return 0;
}

/* plot_option without its effect: does an option start here? */
static int
plot_peek(void)
{
	char save[sizeof Q.popt];
	int s = wi, pm = Q.pmode, r, q = quiet;
	const char *pa = Q.pat;

	memcpy(save, Q.popt, sizeof save);
	quiet = 1;
	dry++;
	r = plot_option();
	dry--;
	quiet = q;
	memcpy(Q.popt, save, sizeof save);
	Q.pmode = pm;
	Q.pat = pa;
	wi = s;
	return r;
}

/* "a contour plot of", "the graph of", "a surface plot for" */
static int
plot_noun(void)
{
	char save[sizeof Q.popt];
	int s = wi, pm = Q.pmode, any = 0;
	const char *pa = Q.pat;

	memcpy(save, Q.popt, sizeof save);
	for (;;) {
		int s1 = wi;
		mat("?a|an|the|some|nice|quick|simple|little|small|big|animated");
		if (mat("plot|plots|graph|graphs|chart|picture|drawing|sketch|figure|"
		    "visualization|visualisation|diagram|rendering|render|"
		    "animation|gif|view") && peek("of|for")) {
			any = 1;
			break;
		}
		wi = s1;
		mat("?a|an|the");
		if (mat("neighbourhood|neighborhood") && peek("of|for")) {
			if (Q.pmode != 1)
				Q.pmode = 2;
			any = 1;
			break;
		}
		wi = s1;
		{
			/* only a view names a plot: "the derivative of" does not */
			static const char *const views[] = {" contour", " heatmap",
				" wireframe", " polar", " field", " parallel", " slices",
				" rotate", NULL};
			size_t n0 = strlen(Q.popt);
			int pm0 = Q.pmode;
			if (!plot_option() || W[wi].k == W_END ||
			    (Q.pmode == pm0 && !in_list(views, Q.popt + n0))) {
				Q.popt[n0] = 0;
				Q.pmode = pm0;
				wi = s1;
				break;
			}
		}
		if (peek("of|for")) {	/* "a heatmap of", "the slope field of" */
			any = 1;
			break;
		}
	}
	if (any && mat("of|for"))
		return 1;
	memcpy(Q.popt, save, sizeof save);
	Q.pmode = pm;
	Q.pat = pa;
	wi = s;
	return 0;
}

static void
plot_tail(void)
{
	for (;;) {
		struct ex a, b;
		int s = wi;
		if (W[wi].k == W_END)
			return;
		if (acc_sym(",") || acc_sym(";"))
			continue;
		if (plot_option())
			continue;
		if (const_interval(&a, &b, 0)) {
			Q.ilo = a.s;
			Q.ihi = b.s;
			continue;
		}
		if (parse_clause())
			continue;
		if (mat("please|thanks|again|too|now") || mat("thank you") ||
		    mat("for me") || mat("as well") || mat("if possible") ||
		    mat("if you can"))
			continue;
		if (mat("and|also|plus|then") && W[wi].k != W_END)
			continue;
		wi = s;
		return;
	}
}

/* the functions: "sin x and cos x", "f, g and h" */
static void
plot_targets(void)
{
	for (;;) {
		int s;
		if (Q.npl == 6)
			nfail("at most six functions in one plot");
		cur_mask = 1;
		if (issym("(")) {
			/* a tuple (x(t), y(t)) or (P, Q): its parts one by one */
			int k, d = 0, comma = 0;
			for (k = wi; W[k].k != W_END; k++) {
				if (W[k].k == W_SYM && !strcmp(W[k].s, "("))
					d++;
				else if (W[k].k == W_SYM && !strcmp(W[k].s, ")") && --d == 0)
					break;
				else if (d == 1 && W[k].k == W_SYM && !strcmp(W[k].s, ","))
					comma = 1;
			}
			if (comma && W[k].k != W_END) {
				struct tgt *t = &Q.pl[Q.npl++];
				const char *parts = "";
				int np = 0;
				memset(t, 0, sizeof *t);
				acc_sym("(");
				do {
					struct tgt c;
					target(&c, P_CMP);
					parts = np++ ? S("%s, %s", parts, c.e.s) : c.e.s;
				} while (acc_sym(","));
				if (!acc_sym(")"))
					nfail("I didn't follow \"%s\"", words_text(wi, nw));
				t->has = 1;
				t->e = mkatom(S("(%s)", parts), 2, 0);
				goto next;
			}
		}
		target(&Q.pl[Q.npl++], P_CMP);
next:
		s = wi;
		if (acc_sym(",") || acc_sym("&") || mat("and|plus|vs|versus") ||
		    mat("together|along|alongside with")) {
			if (W[wi].k == W_END || plot_peek() ||
			    peek("for|where|with|on|from|over|between|when|whenever|"
			    "its|it|their|in|at|near|around")) {
				wi = s;
				break;
			}
			continue;
		}
		break;
	}
}

static void
plot_sentence(void)
{
	Q.cmd = C_PLOT;
	mat("?me|us|out");
	/* "plot 2d", "plot 3d", "plot Nd", "an n-dimensional plot of" */
	for (;;) {
		int s1 = wi;
		mat("?in|as ?a|an|the");
		if (W[wi].k == W_NUM && isw(wi + 1, "d") && !W[wi + 1].sp) {
			int dim = atoi(W[wi].s);
			wi += 2;
			if (dim == 2)
				popt_add("2d");
			else if (dim == 3 && Q.pmode == 0)
				Q.pmode = 1;
			/* 4d and up: by the number of variables (plot.c) */
			continue;
		}
		if (isw(wi, "n") && W[wi + 1].k == W_SYM &&
		    !strcmp(W[wi + 1].s, "-") && isw(wi + 2, "d")) {
			wi += 3;
			continue;
		}
		if (mat("nd|n-dimensional|multidimensional|multi-dimensional|"
		    "multivariate|higher-dimensional"))
			continue;
		wi = s1;
		break;
	}
	while (plot_noun())
		;
	mat("?of|for");
	/* "the unit circle x^2 + y^2 = 1", "the function f" */
	mat("?the|a|an ?unit circle|curve|curves|function|functions|parabola|"
	    "line|lines|ellipse|hyperbola|cardioid|spiral|rose|equation|"
	    "equations|expression|expressions|formula|wave|surface|helix|field");
	mat("?the|a|an ?vector|gradient field|fields");
	mat("?of|for|given by|defined by");
	plot_targets();
	plot_tail();
}

/* is W[i] the start of "a neighbourhood", "a contour plot" after "as"? */
static int
plot_as_at(int i)
{
	static const char *const w[] = {"neighbourhood", "neighborhood",
		"neighbourhoods", "neighborhoods", "plot", "graph", "surface",
		"contour", "contours", "heatmap", "heat", "wireframe", "picture",
		"chart", "curve", "drawing", "sketch", "polar", "small", "slices",
		"panels", "parallel", "mesh", "density", "rotating", "animation",
		"spinning", "level", "slope", "direction", NULL};
	int k;

	if (isw(i, "a") || isw(i, "an") || isw(i, "the") || isw(i, "its"))
		i++;
	if (isw(i, "3") && isw(i + 1, "d"))
		return 1;
	for (k = 0; w[k]; k++)
		if (isw(i, w[k]))
			return 1;
	return 0;
}

/* "F as a neighbourhood": the functions are what comes before "as" */
static int
plot_as(void)
{
	int k, depth = 0;

	for (k = wi; W[k].k != W_END; k++) {
		if (W[k].k == W_SYM && (!strcmp(W[k].s, "(") || !strcmp(W[k].s, "[")))
			depth++;
		if (W[k].k == W_SYM && (!strcmp(W[k].s, ")") || !strcmp(W[k].s, "]")))
			depth--;
		if (depth == 0 && k > wi && isw(k, "as") && plot_as_at(k + 1))
			break;
	}
	if (W[k].k == W_END)
		return 0;
	{
		struct word save = W[k];
		W[k].k = W_END;
		Q.cmd = C_PLOT;
		plot_targets();
		W[k] = save;
		if (wi != k)
			nfail("I didn't follow \"%s\"", words_text(wi, k));
	}
	plot_tail();
	return 1;
}

/* "what does sin x look like" */
static int
plot_looks(void)
{
	int s = wi, k;

	if (!mat("what|how does|do|would|will|might"))
		return 0;
	for (k = wi; W[k].k != W_END; k++)
		if (isw(k, "look") || isw(k, "looks"))
			break;
	if (W[k].k == W_END || k == wi || !(W[k + 1].k == W_END ||
	    isw(k + 1, "like") || plot_peek())) {
		wi = s;
		return 0;
	}
	{
		struct word save = W[k];
		W[k].k = W_END;
		Q.cmd = C_PLOT;
		plot_targets();
		W[k] = save;
		if (wi != k)
			nfail("I didn't follow \"%s\"", words_text(wi, k));
	}
	wi = k + 1;
	mat("?like");
	plot_tail();
	return 1;
}

/* top-level "==" of an equation back to "=" (plot x^2 + y^2 = 1) */
static const char *
plot_eq(const char *s)
{
	int d = 0;
	const char *p;

	for (p = s; *p; p++) {
		if (*p == '(' || *p == '[')
			d++;
		else if (*p == ')' || *p == ']')
			d--;
		else if (d == 0 && p[0] == '=' && p[1] == '=')
			return S("%.*s=%s", (int)(p - s), s, p + 2);
	}
	return s;
}

static void
emit_plot(void)
{
	struct sbuf sb = {0};
	char typed[MAXV];
	int k, j, hv = -1;
	const char *cmd = Q.pmode == 1 ? "plot3d" : Q.pmode == 2 ?
	    "neighbourhood" : "plot";

	memset(typed, 0, sizeof typed);
	sb_printf(&sb, "%s ", cmd);
	for (k = 0; k < Q.npl; k++) {
		struct tgt *t = &Q.pl[k];
		const char *body;
		if (t->named)
			body = t->fn;
		else {
			body = t->e.s;
			for (j = 0; j < nv; j++)
				if (V[j].val[0])
					body = subst_ident(body, V[j].name, value_text(V[j].val));
			for (j = 0; j < nv; j++)
				if (V[j].ty[0] && !typed[j] && !V[j].val[0] &&
				    strcmp(annotate(body, V[j].name, V[j].ty), body)) {
					body = annotate(body, V[j].name, V[j].ty);
					typed[j] = 1;
				}
			body = plot_eq(body);
		}
		sb_printf(&sb, "%s%s", k ? ", " : "", body);
	}
	/* ranges in words: "for x between 0 and 2" */
	for (j = 0; j < nv; j++) {
		struct var *v = &V[j];
		if (!v->order || v->val[0] || !v->lo[0] || !v->hi[0])
			continue;
		sb_printf(&sb, " %s in %s..%s", v->name, v->lo, v->hi);
		if (!strcmp(v->name, "x"))
			hv = -2;
	}
	for (j = 0; j < nv; j++)
		if (V[j].order && !V[j].val[0] && V[j].ncond && !dry)
			printf("  note: the condition on %s is not used by the plot "
			    "(give it a range: %s in a..b)\n", V[j].name, V[j].name);
	/* "from a to b" without a variable: the horizontal one */
	if (Q.ilo && hv == -1) {
		const char *name = NULL;
		int best = 1 << 30;
		for (j = 0; j < nv; j++)
			if (V[j].order && !V[j].val[0] && !strcmp(V[j].name, "x"))
				name = "x";
		for (j = 0; j < nv && !name; j++)
			if (V[j].order && !V[j].val[0] && V[j].order < best) {
				best = V[j].order;
				name = V[j].name;
			}
		if (Q.npl == 1 && Q.pl[0].named) {
			struct func *f = user_fn(Q.pl[0].fn);
			if (f && f->np)
				name = sym_str(f->p[0].name);
		}
		if (!name)
			name = "x";
		sb_printf(&sb, " %s in %s..%s", name, Q.ilo, Q.ihi);
	}
	if (Q.pat)
		sb_printf(&sb, " at %s", Q.pat);
	sb_puts(&sb, Q.popt);
	emit(sb.p);
	free(sb.p);
	if (Q.npl == 1 && Q.pl[0].named)
		fn_used = Q.pl[0].fn;
}

/* ---- requests ---- */

/*
 * Properties props.c answers: the command for a noun ("the period of"),
 * an adjective ("is f periodic") or a where-question ("where is f
 * decreasing").
 */
static const char *
props_noun(void)
{
	if (mat("?fundamental|smallest|least|minimal|primitive|prime|basic "
	    "period|periods|wavelength"))
		return "period";
	if (mat("periodicity"))
		return "periodic";
	if (mat("intervals|regions|parts of increase and|or decrease") ||
	    mat("intervals|regions|parts of monotonicity|increase|decrease") ||
	    mat("increasing and|or decreasing intervals|regions|parts") ||
	    mat("monotonicity|monotony"))
		return "monotonic";
	if (mat("parity|evenness|oddness"))
		return "parity";
	if (mat("symmetry|symmetries") || mat("axis|axes|centre|center|centres|"
	    "centers of symmetry"))
		return "symmetry";
	if (mat("?real zeros|zeroes|roots|zero|root|x-intercepts|intercepts") ||
	    mat("x - intercepts|intercept"))
		return "zeros";
	if (mat("?vertical|horizontal|oblique|slant asymptote|asymptotes"))
		return "asymptotes";
	if (mat("discontinuities|discontinuity|singularities|singularity|poles|"
	    "pole|jumps|jump"))
		return "discontinuities";
	if (mat("continuity"))
		return "continuity";
	if (mat("intervals|regions of convexity|concavity") ||
	    mat("convexity|concavity"))
		return "convexity";
	if (mat("boundedness"))
		return "bounded";
	if (mat("injectivity"))
		return "injective";
	if (mat("surjectivity"))
		return "surjective";
	if (mat("bijectivity"))
		return "bijective";
	if (mat("classification|nature|properties|characteristics|category") &&
	    peek("of"))
		return "identify";
	return NULL;
}

static const char *
props_adj(void)
{
	int s = wi;
	const char *c;

	mat("?a|an");
	mat("?strictly|monotonically|always|everywhere|globally");
	if (mat("monotonic|monotone|increasing|decreasing|nondecreasing|"
	    "nonincreasing|ascending|descending|rising|falling|growing") ||
	    mat("non - decreasing|increasing") || mat("order preserving|reversing")) {
		mat("?function|map|mapping|sequence");
		return "monotonic";
	}
	if (mat("periodic|cyclic|cyclical") || mat("repeating ?itself") ||
	    mat("repeat|repeats|recur|recurs|cycle|cycles ?itself|periodically")) {
		mat("?function|map");
		return "periodic";
	}
	if (mat("convex|concave ?up|down|upward|upwards|downward|downwards")) {
		mat("?function|map");
		return "convexity";
	}
	if (mat("bounded|limited ?from ?above|below")) {
		mat("?function|map");
		return "bounded";
	}
	if (mat("injective|invertible|reversible") || mat("one to one") ||
	    mat("one - to - one") || mat("1 - 1") || mat("1 - to - 1")) {
		mat("?function|map|mapping");
		return "injective";
	}
	if (mat("surjective|onto")) {
		mat("?function|map|mapping");
		return "surjective";
	}
	if (mat("bijective|bijection|permutation")) {
		mat("?function|map|mapping");
		return "bijective";
	}
	if (mat("continuous|discontinuous")) {
		mat("?everywhere|function|map");
		mat("?on its domain");
		return "continuity";
	}
	if (mat("polynomial|rational|algebraic|transcendental|linear|quadratic|"
	    "cubic|trigonometric|exponential|logarithmic|elementary ?function|"
	    "map|expression"))
		return "identify";
	if (mat("have|has|possess|possesses ?a|an|any|some|vertical|horizontal|"
	    "oblique") && (c = props_noun()) != NULL)
		return c;
	wi = s;
	if (mat("cross|crosses|hit|hits|touch|touches|intersect|intersects|meet|"
	    "meets zero|0") || mat("cross|crosses|hit|hits|touch|touches|"
	    "intersect|intersects|meet|meets the x - axis|axis"))
		return "zeros";
	wi = s;
	return NULL;
}

static const char *
props_where(void)
{
	if (mat("?strictly|monotonically increasing|decreasing|rising|falling|"
	    "growing|shrinking|monotonic|monotone|nondecreasing|nonincreasing"))
		return "monotonic";
	if (mat("?strictly convex|concave"))
		return "convexity";
	if (mat("discontinuous|continuous|singular") || mat("not continuous"))
		return "discontinuities";
	if (mat("zero|0|vanishing|vanish|vanishes") || mat("equal|equals ?to "
	    "zero|0") || mat("cross|crosses|hit|hits|touch|touches|meet|meets|"
	    "intersect|intersects zero|0") || mat("cross|crosses|hit|hits|touch|"
	    "touches|meet|meets|intersect|intersects the x - axis|axis"))
		return "zeros";
	return NULL;
}

static int
noun_cmd(void)
{
	int n, s0;

	if (plot_noun())
		return C_PLOT;
	mat("?the|a|an|some|any|its|their");
	if (small_number(&n)) {
		Q.count = n;
		mat("?more");
	}
	if ((Q.pcmd = props_noun()) != NULL)
		return C_PCMD;
	if (mat("range|image|codomain|bounds|extent|span") ||
	    mat("?set ?of possible|output|attainable values") ||
	    mat("min|minimum|smallest|lowest and max|maximum|largest|highest "
	    "?values|value") ||
	    mat("max|maximum|largest|highest and min|minimum|smallest|lowest "
	    "?values|value") ||
	    mat("values ?that it can take|takes"))
		return C_RANGE;
	s0 = wi;
	if (mat("maximum|max|largest|biggest|highest|minimum|min|smallest|"
	    "lowest ?possible|output value|values") && peek("of|for|in")) {
		/* "the maximum of a and b" is max(a, b): look for "and" */
		int s = wi, depth = 0, i;
		for (i = wi + 1; W[i].k != W_END; i++) {
			if (W[i].k == W_SYM && !strcmp(W[i].s, "("))
				depth++;
			if (W[i].k == W_SYM && !strcmp(W[i].s, ")"))
				depth--;
			if (depth == 0 && (isw(i, "for") || isw(i, "where") ||
			    isw(i, "when") || isw(i, "with") || isw(i, "given")))
				break;
			if (depth == 0 && (isw(i, "and") || (W[i].k == W_SYM &&
			    !strcmp(W[i].s, ",")))) {
				wi = i + 1;
				if (!and_starts_clause()) {
					wi = s0;
					return 0;
				}
				break;
			}
		}
		wi = s;
		return C_RANGE;
	}
	wi = s0;
	if (mat("domain") || mat("valid|legal|allowed inputs|values|arguments"))
		return C_DOMAIN;
	if (mat("overflow|wraparound|wrapping|wrap|ub|nsw|nuw|overflows ?behavior|"
	    "behaviour|analysis|check|checks|flags|cases|points") ||
	    mat("undefined behavior|behaviour ?analysis"))
		return C_WRAP;
	if (mat("critical|stationary|turning|inflection points|point") ||
	    mat("local|global ?extrema|extremes|maxima|minima|maximum|minimum") ||
	    mat("maxima and minima") || mat("minima and maxima"))
		return C_CRIT;
	if (mat("?first derivative|derivatives|gradient|differential|slope") ||
	    mat("forward difference"))
		return C_DIFF;
	if (mat("continuous approximation|version|analogue|analog|limit|form") ||
	    mat("integral approximation|approximations") ||
	    mat("euler-maclaurin|stirling ?approximation|expansion|form|formula") ||
	    mat("euler maclaurin ?approximation|expansion|form|formula"))
		return C_CONT;
	if (mat("?definite|indefinite integral|integrals|antiderivative|"
	    "antiderivatives|primitive|area")) {
		mat("?under");
		return C_INTEG;
	}
	if (mat("closed|explicit|exact form|forms|formula|expression|solution") ||
	    mat("closed-form|explicit ?formula|expression|solution"))
		return C_CLOSED;
	if (mat("?best|cheapest|good|efficient evaluation|evaluating scheme|"
	    "schemes|strategy|order|method") || mat("?evaluation scheme|schemes"))
		return C_SCHEME;
	if (mat("?minimal|smallest|cheapest|optimal|best|minimum ?logic|boolean "
	    "gate network|circuit|netlist|implementation|count") ||
	    mat("?minimal|smallest|cheapest|optimal|best|minimum ?logic|boolean "
	    "circuit|circuits|netlist|gates"))
		return C_GATES;
	if (mat("best|optimal ?magic constant|constants|parameter|parameters|k") ||
	    mat("best|optimal value|values of|for the|its constant|constants|"
	    "parameter|parameters"))
		return C_FIT;
	if (mat("?best|cheap|cheapest|good|fast|fastest ?chebyshev|cheby|cheb|"
	    "minimax|remez|polynomial|poly|taylor|rational approximation|"
	    "approximations|approximant|approximants|expansion|fit|fits|series") ||
	    mat("chebyshev|cheby|cheb|minimax|remez polynomial|polynomials|poly|"
	    "polys|interpolation|fit|fits")) {
		int i;
		for (i = s0; i < wi; i++) {
			if (isw(i, "chebyshev") || isw(i, "cheby") || isw(i, "cheb"))
				strcpy(Q.method, "cheb");
			if (isw(i, "minimax") || isw(i, "remez"))
				strcpy(Q.method, "minimax");
			if (isw(i, "taylor"))
				strcpy(Q.method, "econ");
		}
		return C_APPROX;
	}
	/* a method alone: "minimax of sin ...", "chebyshev for exp ..." */
	{
		int s1 = wi;
		if (mat("chebyshev|cheby|cheb|tschebyscheff|minimax|remez|taylor") &&
		    (isw(wi, "of") || isw(wi, "for") || isw(wi, "on"))) {
			if (isw(wi - 1, "minimax") || isw(wi - 1, "remez"))
				strcpy(Q.method, "minimax");
			else if (isw(wi - 1, "taylor"))
				strcpy(Q.method, "econ");
			else
				strcpy(Q.method, "cheb");
			return C_APPROX;
		}
		wi = s1;
	}
	if (mat("cheaper|faster|better|alternative|other|equivalent|different|"
	    "simpler|cheapest|fastest|best ?ways|way|implementations|"
	    "implementation|formulas|formulae|formula|expressions|expression|"
	    "programs|program|versions|version|rewrites|rewrite|alternatives") ||
	    mat("implementations|alternatives|rewrites"))
		return C_FIND;
	if (mat("cost|price|expense|latency") || mat("gpu cost"))
		return C_COST;
	if (mat("?truth table|tabulation") || mat("list|table of values") ||
	    mat("values")) {
		mat("?of values|outputs");
		return C_TABLE;
	}
	if (mat("?approximation|relative|maximum|max error|accuracy|precision"))
		return C_ERROR;
	if (mat("simplified form|version|expression") || mat("simplification|"
	    "definition|formula|type"))
		return C_SHOW;
	return 0;
}

static int
float_target(const struct tgt *t)
{
	struct func *f;
	int k;

	if (!t->named)
		return t->e.ity == 2 || t->e.ity == 3;
	if ((f = user_fn(t->fn)) == NULL)
		return 0;
	for (k = 0; k < f->np; k++)
		if (TY_FP(f->p[k].ty))
			return 1;
	return TY_FP(f->p[0].ty);
}

static void
two_targets(const char *sep)
{
	cur_mask = 1;
	target(&Q.a, P_BOR);
	if (!mat(sep))
		nfail("expected \"%s\" between the two functions", sep);
	mat("?the");
	cur_mask = 2;
	target(&Q.b, P_BOR);
	cur_mask = 1;
}

static const char *const vs_words =
    "to|against|~|vs|versus|with|and|for|of|approximating|from";

/* "as an approximation of", "compared to": 1 when it ended the phrase */
static int
vs_sep(void)
{
	int s = wi;

	mat("?as ?an|a ?approximation|estimate ?to|of|for");
	mat("?when ?compared|comparing ?to|with");
	mat("?compared|relative|close ?to|with");
	return wi > s && (isw(wi - 1, "to") || isw(wi - 1, "of") ||
	    isw(wi - 1, "for") || isw(wi - 1, "with"));
}

static void
second_target(void)
{
	cur_mask = 2;
	mat("?the ?function");
	target(&Q.b, P_BOR);
	cur_mask = 1;
}

static int
yes_no(int s0)
{
	int s;

	cur_mask = 1;
	if (mat("there ?any|some")) {
		/* "is there overflow in f", "are there critical points of f" */
		int c = noun_cmd();
		if (c) {
			Q.cmd = c;
			mat("in|of|for|when computing|when evaluating");
			target(&Q.a, P_TERN);
			return 1;
		}
		wi = s0;
		return 0;
	}
	mat("?the function|map");
	target(&Q.a, P_BOR);
	s = wi;
	if (mat("and")) {
		second_target();
		mat("?always|ever|really");
		if (mat("equal|equivalent|identical|interchangeable") ||
		    mat("the same ?function|functions|thing")) {
			mat("?to each other|functions");
			Q.cmd = C_EQUIV;
			return 1;
		}
		wi = s;
	}
	mat("?always|ever|possibly|really");
	mat("?guaranteed|going to");
	if ((Q.pcmd = props_adj()) != NULL) {
		Q.cmd = C_PCMD;
		return 1;
	}
	/* algebraic properties, checked with equiv */
	{
		static const char *const pw[] = {"commutative", "symmetric",
			"associative", "additive", "idempotent", NULL};
		static const int pk[] = {PR_COMM, PR_COMM, PR_ASSOC, PR_ADD,
			PR_IDEM};
		int s5 = wi, k;
		mat("?a|an");
		if (W[wi].k == W_WORD && (k = lookup(pw, W[wi].s)) >= 0) {
			wi++;
			mat("?function|operation|operator|map|mapping|op");
			Q.prop = pk[k];
			Q.cmd = C_PROP;
			if (k == 1)   /* symmetric: about an axis for one argument */
				Q.pcmd = "symmetry";
			return 1;
		}
		if (mat("odd|even")) {
			int odd = isw(wi - 1, "odd");
			if (!mat("function|map|mapping")) {
				const char *o[2];
				o[0] = odd ? "an odd function, f(-x) = -f(x)" :
				    "an even function, f(-x) = f(x)";
				o[1] = odd ? "an odd number for every input" :
				    "an even number for every input";
				if (clarify(NULL, S("\"%s\"", odd ? "odd" : "even"), o,
				    2, 0, "say \"an odd function\" or \"always odd\"")) {
					Q.prop = odd ? PR_ODDV : PR_EVENV;
					Q.cmd = C_PROP;
					return 1;
				}
			}
			mat("?function|map|mapping");
			Q.pcmd = "parity";
			Q.cmd = C_PCMD;
			return 1;
		}
		wi = s5;
	}
	if (mat("overflow|overflows|underflow|underflows|wrap|wraps ?around") ||
	    mat("have|has|invoke|invokes|cause|causes|trigger|triggers ub") ||
	    mat("have|has|invoke|invokes|cause|causes|trigger|triggers undefined "
	    "behavior|behaviour") || mat("lose|loses precision") ||
	    mat("truncate|truncates") || mat("be ub")) {
		Q.cmd = C_WRAP;
		return 1;
	}
	if (mat("safe|ub-free|overflow-free") || mat("free of|from ub|overflow") ||
	    mat("safe from overflow|ub")) {
		Q.cmd = C_WRAP;
		return 1;
	}
	if (mat("defined|well-defined|total|valid")) {
		mat("?everywhere|for all inputs|on its domain");
		Q.cmd = C_DOMAIN;
		return 1;
	}
	if (mat("equal|equals|equivalent|identical ?to|with") ||
	    mat("interchangeable with") || mat("?the same ?function|thing as") ||
	    mat("==")) {
		second_target();
		mat("?always|everywhere|for all inputs");
		Q.cmd = C_EQUIV;
		return 1;
	}
	if (mat("a|an good|accurate|close|decent ?approximation|estimate of|to|"
	    "for") || mat("close|accurate to|for")) {
		second_target();
		Q.cmd = C_ERROR;
		return 1;
	}
	if (mat("fit|fits in|into") || mat("fit|fits")) {
		struct decl d;
		memset(&d, 0, sizeof d);
		mat("?a|an");
		parse_type(&d);
		Q.cmd = C_RANGE;
		Q.why = d.ty[0] ? S("see \"fits in\" below; you asked about %s",
		    d.ty) : NULL;
		return 1;
	}
	if (mat("positive|negative|zero|nonnegative|non-negative|nonzero|"
	    "non-zero|bounded|even|odd") || mat("bigger|larger|greater|smaller|"
	    "less|more than")) {
		wi = s0;
		/* a comparison with a function in it: answer with its range */
		incond++;
		target(&Q.a, P_TERN);
		incond--;
		if (!Q.a.e.isbool) {
			wi = s;
			mat("?always|ever");
			while (W[wi].k != W_END && !peek("for|where|when|with|given"))
				wi++;
			Q.cmd = C_RANGE;
			return 1;
		}
		Q.cmd = C_EVAL;
		return 1;
	}
	/* "is 2^10 greater than 1000": evaluate the comparison */
	wi = s0;
	incond++;
	target(&Q.a, P_TERN);
	incond--;
	Q.cmd = C_EVAL;
	return 1;
}

static int
how_question(void)
{
	if (mat("big|large|small|high or small|big|large|low") ||
	    mat("big|large|small|high|low|negative|positive|far|wide")) {
		mat("?can|could|does|do|will|would|might|is|are");
		target(&Q.a, P_TERN);
		mat("?get|be|become|go|grow|reach|possibly|range");
		mat("?be|get");
		Q.cmd = C_RANGE;
		return 1;
	}
	if (mat("many bits|bytes")) {
		mat("?does|do|will|would|is|are");
		mat("?it ?take|need ?to ?store|hold|represent");
		target(&Q.a, P_TERN);
		mat("?need|needs|take|takes|use|uses|require|requires|occupy|"
		    "occupies|needed|required");
		Q.cmd = C_RANGE;
		Q.why = "see \"fits in\"";
		return 1;
	}
	if (mat("accurate|good|close|precise|exact|wrong|bad|well") ||
	    mat("far off")) {
		mat("?is|are|does|do");
		target(&Q.a, P_BOR);
		if (!vs_sep() && !mat(vs_words))
			nfail("compared with what?");
		second_target();
		mat("?approximate|approximates");
		Q.cmd = C_ERROR;
		return 1;
	}
	if (mat("expensive|costly|fast|slow|cheap|quick|slowly|quickly")) {
		mat("?is|are|would|will|does|do|be|it");
		mat("?it ?to ?compute|evaluate|run");
		target(&Q.a, P_TERN);
		Q.cmd = C_COST;
		return 1;
	}
	if (mat("much does")) {
		target(&Q.a, P_TERN);
		mat("cost");
		Q.cmd = C_COST;
		return 1;
	}
	if (mat("many|few gates|steps|ops|operations|instructions")) {
		mat("?does|do|would|will|are|is ?it ?take|need|needed|required");
		mat("?for|to ?compute|implement|build|make");
		target(&Q.a, P_TERN);
		mat("?need|needs|take|takes");
		Q.cmd = C_GATES;
		return 1;
	}
	if (mat("can|could|do|should|would|might|to") ) {
		mat("?i|we|you|one");
		mat("?best ?compute|implement|calculate|write|evaluate|express|"
		    "do|get|build|make");
		target(&Q.a, P_TERN);
		Q.cmd = C_FIND;
		if (mat("with|using|in|from ?only ?logic|boolean gates|gate|a|"
		    "circuit|circuits"))
			Q.cmd = C_GATES;
		mat("?more faster|cheaper|quicker|better|cheaply|efficiently|quickly");
		mat("?with fewer operations|ops|instructions");
		return 1;
	}
	nfail("I don't know how to answer that \"how\" question");
	return 0;
}

static int
where_question(void)
{
	int c = 0;

	mat("?inputs|values|arguments|points");
	if (mat("of") && varlike(wi))
		wi++;
	mat("?is|are|does|do|will|can");
	target(&Q.a, P_BOR);
	if ((Q.pcmd = props_where()) != NULL) {
		Q.cmd = C_PCMD;
		return 1;
	}
	if (mat("?not defined|undefined|valid|invalid|ok|well-defined"))
		c = C_DOMAIN;
	else if (mat("maximized|minimized|maximised|minimised|maximal|minimal|"
	    "largest|smallest|biggest|increasing|decreasing|flat|stationary|"
	    "extremal") || mat("reach|reaches|attain|attains|hit|hits|take|"
	    "takes|have|has ?its ?a maximum|minimum|max|min|extreme|extremes|"
	    "extrema|peak|peaks|extremum"))
		c = C_CRIT;
	else if (mat("overflow|overflows|wrap|wraps|underflow|underflows|ub") ||
	    mat("have|has|invoke|cause undefined behavior|behaviour"))
		c = C_WRAP;
	else if (mat("equal|agree|match ?with")) {
		second_target();
		c = C_EQUIV;
	}
	if (!c)
		return 0;
	Q.cmd = c;
	return 1;
}

static void
show_help(void)
{
	puts(
"English: type a question or request; it is translated to ucas and run.\n"
"  what is the range of the integer square root of a 4-bit unsigned x?\n"
"  does x times 2 plus 1 overflow for a signed byte x?\n"
"  is (a + b) / 2 the same as the average of a and b, for u8 a and b?\n"
"  what are the critical points of x cubed minus 3x?\n"
"  the derivative of sin x squared with respect to x\n"
"  let f of x be x squared plus 1 where x is between 0 and 10\n"
"  how accurate is quake as an approximation of rsqrt?\n"
"  find cheaper implementations of days;  gates for it using arm64\n"
"  what is 2 to the 10th?   what is f when x is 3?\n"
"  cos of pi/2 to -pi/2 w/o crit. points      (range ... nocrit)\n"
"  alt ((1 << 3) & x) + x*5                   (find ... 10)\n"
"  cos from -90deg to 90deg using cheby poly  (approx ... method cheb)\n"
"  is avg commutative / associative / additive / idempotent / odd?\n"
"  is sin(2x) periodic?  where is x^3 - 3x decreasing?  is f bounded?\n"
"  what kind of equation is y'' + y = 0?   identify x^2 + y^2 = 1\n"
"  the integral of x squared times sin x from 0 to pi\n"
"  the closed form of the sum of i squared for i from 1 to n\n"
"  the best evaluation scheme for fb;   solve y from 0 to 2 in 4 steps\n"
"  assume bytes are signed / log means base 2 / int means unbounded\n"
"  assume x is an unsigned byte;   forget everything\n"
"A line is read as English when it ends in '?', starts with '?', or reads\n"
"like a sentence.  Unclear words (byte, log, int, word, \"the square root\n"
"of x plus 1\") are asked about on a terminal, otherwise the default is\n"
"taken and a note says so; so is a missing approx interval, where the\n"
"reply can add options (\"cheby poly nocrit\", \"0 to 1 degree 5 crit\").\n"
"Misspelt words are corrected towards the words ucas knows (also two\n"
"halves of one word); Greek letter names (theta, alpha) are variables;\n"
"90deg and 90r are angles.");
}

static int
assume(void)
{
	static const struct {
		const char *subj, *val, *key;
		int v;
	} tab[] = {
		{"log|logs|logarithm|logarithms|log(x)", "natural|ln|base-e", "log", 0},
		{"log|logs|logarithm|logarithms|log(x)", "?base 2|two|binary|log2|lg",
		    "log", 1},
		{"log|logs|logarithm|logarithms|log(x)",
		    "?base 10|ten|common|decimal|log10", "log", 2},
		{"byte|bytes", "unsigned", "byte", 0},
		{"byte|bytes", "signed", "byte", 1},
		{"char|chars|character|characters", "signed", "char", 0},
		{"char|chars|character|characters", "unsigned", "char", 1},
		{"int|ints", "32-bit|i32|c|32", "int", 0},
		{"int|ints", "unbounded|mathematical|math|arbitrary|bignum|bignums",
		    "int", 1},
		{"integer|integers", "unbounded|mathematical|math|arbitrary", "integer",
		    0},
		{"integer|integers", "32-bit|i32|c|32", "integer", 1},
		{"word|words", "16-bit|16|u16", "word", 0},
		{"word|words", "32-bit|32|u32", "word", 1},
		{"word|words", "64-bit|64|u64", "word", 2},
		{"n-bit|sized|fixed-width ?numbers|integers|values|types", "unsigned",
		    "nbit", 0},
		{"n-bit|sized|fixed-width ?numbers|integers|values|types", "signed",
		    "nbit", 1},
		{"and|or|and/or", "bitwise", "andor", 0},
		{"and|or|and/or", "logical", "andor", 1},
		{"not", "bitwise", "not", 0},
		{"not", "logical", "not", 1},
		{"sqrt|square ?root|roots ?of ?integers", "floor|integer|rounded",
		    "sqrt", 0},
		{"sqrt|square ?root|roots ?of ?integers", "real|exact", "sqrt", 1},
		{NULL, NULL, NULL, 0}
	};
	char names[8][32];
	struct decl d;
	int s = wi, k, n;

	for (k = 0; tab[k].subj; k++) {
		wi = s;
		mat("?that ?the ?word");
		if (W[wi].k == W_NUM && isw(wi + 1, "bit")) /* "N-bit" written out */
			;
		if (!mat(tab[k].subj) && !(isw(wi, "n") && mat("n - bit")))
			continue;
		mat("?numbers|integers|values|types");
		mat("?on|of|between integers|numbers");
		mat("?always|should");
		mat("?means|mean|is|are|=|denote|denotes|be|refers|refer|default|"
		    "defaults");
		mat("?to");
		mat("?the|a|an|be");
		mat("?by default");
		if (!mat(tab[k].val))
			continue;
		mat("?ones|one|numbers|integers|logs|log|logarithm|logarithms|bits|"
		    "operators|operations");
		if (W[wi].k != W_END)
			continue;
		if (!dry) {
			pref_set(tab[k].key, tab[k].v);
			printf("  ok: %s\n", words_text(s, wi));
		}
		Q.cmd = C_NONE;
		return 1;
	}
	wi = s;
	mat("?that");
	if (parse_varlist(names, &n)) {
		memset(&d, 0, sizeof d);
		session_decl = 1;
		if (parse_descr(&d)) {
			declare(names, n, &d);
			session_decl = 0;
			while (W[wi].k != W_END && (acc_sym(",") || mat("and"))) {
				memset(&d, 0, sizeof d);
				session_decl = 1;
				if (!parse_varlist(names, &n) || !parse_descr(&d))
					break;
				declare(names, n, &d);
				session_decl = 0;
			}
			session_decl = 0;
			if (W[wi].k != W_END)
				nfail("I didn't follow \"%s\"", words_text(wi, nw));
			if (!dry)
				printf("  ok: from now on %s\n", words_text(s, nw));
			Q.cmd = C_NONE;
			return 1;
		}
		session_decl = 0;
	}
	nfail("I don't know how to assume \"%s\"", words_text(s, nw));
	return 0;
}

static int
meta(void)
{
	int s = wi, n;

	if ((mat("help ?me|english|nlp ?with ?english") || mat("what can you do") || mat("how do i use "
	    "?this|you|ucas") || mat("examples")) && W[wi].k == W_END) {
		if (!dry)
			show_help();
		Q.cmd = C_NONE;
		return 1;
	}
	wi = s;
	if (mat("assume|suppose") || mat("from now on ?assume") ||
	    mat("by default") || mat("let us assume|say")) {
		mat("?that|,");
		return assume();
	}
	if (mat("forget|reset|clear")) {
		if (mat("?all|everything|the|your|my ?assumptions|preferences|"
		    "answers|choices|defaults|variables|meanings") &&
		    W[wi].k == W_END) {
			if (!dry) {
				nprefs = nsv = nknown = 0;
				puts("  ok: forgot every assumption");
			}
			Q.cmd = C_NONE;
			return 1;
		}
		if (W[wi].k == W_WORD && W[wi + 1].k == W_END) {
			if (!dry) {
				for (n = 0; n < nsv; n++)
					if (!strcmp(SV[n].name, W[wi].o)) {
						memmove(&SV[n], &SV[n + 1],
						    (size_t)(nsv - n - 1) * sizeof SV[0]);
						nsv--;
						break;
					}
				printf("  ok: forgot %s\n", W[wi].o);
			}
			wi++;
			Q.cmd = C_NONE;
			return 1;
		}
		wi = s;
	}
	if (mat("use|select ?the") || mat("switch to ?the")) {
		if (parse_model() && (mat("?cost ?model") , W[wi].k == W_END)) {
			Q.raw[Q.nraw++] = S("use %s", Q.model);
			Q.cmd = C_RAW;
			return 1;
		}
		wi = s;
	}
	if ((mat("list|show ?all|the|my functions|definitions") ||
	    mat("what are ?all ?the|my functions|definitions") ||
	    mat("what functions are|have ?been ?defined")) && W[wi].k == W_END) {
		Q.raw[Q.nraw++] = "funcs";
		Q.cmd = C_RAW;
		return 1;
	}
	wi = s;
	if ((mat("list|show ?all|the ?cost models") ||
	    mat("what are ?all ?the ?cost models") ||
	    mat("which cost models ?are ?there")) && W[wi].k == W_END) {
		Q.raw[Q.nraw++] = "models";
		Q.cmd = C_RAW;
		return 1;
	}
	wi = s;
	if (mat("?show|give ?me more|next") || mat("keep going") ||
	    mat("continue")) {
		n = 5;
		small_number(&n);
		mat("?more|results|implementations|of them|please");
		if (W[wi].k == W_END) {
			Q.raw[Q.nraw++] = S("more %d", n);
			Q.cmd = C_RAW;
			return 1;
		}
	}
	wi = s;
	/* "call it myabs": name the last function */
	if ((mat("?and call|name it|this|that") ||
	    mat("?and call|name the result|function")) && W[wi].k == W_WORD &&
	    W[wi + 1].k == W_END) {
		const char *ln = last_name();
		struct func *f;
		struct sbuf sb = {0};
		if (ln == NULL || (f = user_fn(ln)) == NULL)
			nfail("there is no function to name yet");
		snprintf(Q.defname, sizeof Q.defname, "%s", W[wi].o);
		sb_printf(&sb, "%s(", Q.defname);
		for (n = 0; n < f->np; n++)
			sb_printf(&sb, "%s%s", n ? ", " : "", fparam_decl(f, n));
		sb_printf(&sb, ") = %s(", ln);
		for (n = 0; n < f->np; n++)
			sb_printf(&sb, "%s%s", n ? ", " : "", sym_str(f->p[n].name));
		sb_putc(&sb, ')');
		Q.raw[Q.nraw++] = S("%s", sb.p);
		Q.raw[Q.nraw++] = S("show %s", Q.defname);
		free(sb.p);
		wi++;
		Q.cmd = C_RAW;
		return 1;
	}
	wi = s;
	if (mat("?show|give ?me") && small_number(&n) &&
	    mat("more ?results|implementations") && W[wi].k == W_END) {
		Q.raw[Q.nraw++] = S("more %d", n);
		Q.cmd = C_RAW;
		return 1;
	}
	wi = s;
	return 0;
}

/* let f(x) be ..., define f of x and y as ..., f(x) is ... */
static int
definition(void)
{
	int s = wi, let, k;
	char names[8][32];
	struct decl d;

	let = mat("let|define|set|declare|suppose");
	mat("?the function|functions");
	if (W[wi].k != W_WORD || reserved_word(W[wi].s) ||
	    (!let && !(issym("(") || 1)))
		goto no;
	snprintf(Q.defname, sizeof Q.defname, "%s", W[wi].o);
	wi++;
	Q.ndparam = 0;
	if (acc_sym("(")) {
		Q.dexplicit = 1;
		if (!acc_sym(")")) {
			do {
				if (!varlike(wi) || Q.ndparam == MAXP)
					goto no;
				snprintf(Q.dparam[Q.ndparam++], 32, "%s", W[wi].o);
				var_atom(W[wi].o);
				wi++;
				if (acc_sym(":")) {
					memset(&d, 0, sizeof d);
					if (!parse_type(&d))
						goto no;
					while (parse_rangemod(&d))
						;
					apply_decl(var_get(Q.dparam[Q.ndparam - 1]), &d);
				}
			} while (acc_sym(","));
			if (!acc_sym(")"))
				goto no;
		}
	} else if (mat("of") && parse_varlist(names, &k)) {
		Q.dexplicit = 1;
		for (Q.ndparam = 0; Q.ndparam < k && Q.ndparam < MAXP; Q.ndparam++) {
			snprintf(Q.dparam[Q.ndparam], 32, "%s", names[Q.ndparam]);
			var_atom(names[Q.ndparam]);
		}
	}
	if (!(acc_sym("=") || (acc_sym(":") && acc_sym("=")) ||
	    mat("be equal to") || mat("to be") || mat("stand|stands for") ||
	    mat("be|equal|equals|as|denote|mean|is|means|denotes")))
		goto no;
	if (!let && !Q.dexplicit && !isw(wi - 1, "be"))
		goto no;
	mat("?the function|expression|formula");
	if (Q.dexplicit && mat("the function that maps|takes"))
		mat("?it|them ?to");
	cur_mask = 1;
	target(&Q.a, P_TERN);
	Q.cmd = C_DEF;
	return 1;
no:
	wi = s;
	Q.ndparam = 0;
	Q.dexplicit = 0;
	Q.defname[0] = 0;
	return 0;
}

static void
fillers(void)
{
	for (;;) {
		if (mat("please|pls|plz|plox|pleas|kindly|hey|hi|hello|ok|okay|so|"
		    "now|also|and|then|"
		    "next|well|alright|right|hmm|um|uh|ucas|quick|question") ||
		    mat("can|could|would|will you ?please") ||
		    mat("i|we ?would|really|just like|want|need|wanna|wish ?you ?to "
		    "?know|see|understand|learn") ||
		    mat("i|we ?am|are wondering|curious ?about") ||
		    mat("i|we wonder") || mat("do you know") ||
		    mat("let me|us know|see") || mat("tell|show|give|find|get me|us") ||
		    mat("help me|us ?find|understand|compute") || mat("let us") ||
		    mat("to"))
			continue;
		if (isw(wi, "let") && isw(wi + 1, "us")) {
			wi += 2;
			continue;
		}
		break;
	}
}

static void
cmd_after_noun(int c)
{
	Q.cmd = c;
	switch (c) {
	case C_PLOT:
		plot_targets();
		plot_tail();
		break;
	case C_EQUIV:
		break;
	case C_ERROR:
		if (Q.a.has) {
			mat("?of|for|in");
		} else {
			mat("of|in|for");
			cur_mask = 1;
			target(&Q.a, P_BOR);
		}
		if (vs_sep() || mat(vs_words))
			second_target();
		else
			nfail("the error compared with what?");
		break;
	case C_FIT:
		mat("for|in|of");
		cur_mask = 1;
		target(&Q.a, P_BOR);
		if (!mat("so that it") && !mat("so it") && !mat("to make it"))
			mat("?that|which|to");
		mat("?best");
		if (!mat("approximate|approximates|approximating|match|matches|fit|"
		    "fits|~|against|for|to"))
			nfail("fit it to what?");
		mat("?the ?function");
		second_target();
		break;
	default:
		if (!Q.a.has) {
			if (!mat("of|for|in|on") && !mat("when computing|evaluating") &&
			    !mat("to compute|implement|evaluate|calculate|write|do|get"))
				nfail("of what?");
			mat("?computing|evaluating");
			cur_mask = 1;
			target(&Q.a, P_TERN);
		}
	}
}

/* ---- complex, polar and trig forms (forms.c) ---- */

enum { FN_NONE, FN_RECT, FN_POLAR, FN_EXPF, FN_TRIGF, FN_RE, FN_IM, FN_CONJ,
	FN_MOD, FN_ARG, FN_TRIG };

static const struct {
	const char *pat;
	int cls;
	const char *opt;	/* trig mode */
} form_phr[] = {
	{"real and imaginary parts|part|components|component", FN_RECT, ""},
	{"imaginary and real parts|part|components", FN_RECT, ""},
	{"real part|component|value", FN_RE, ""},
	{"imaginary part|component", FN_IM, ""},
	{"complex conjugate|conjugates", FN_CONJ, ""},
	{"conjugate|conj", FN_CONJ, ""},
	{"modulus|magnitude|norm|length|absolute and argument|arg|phase|angle",
	    FN_POLAR, ""},
	{"modulus-argument|mod-arg|magnitude-phase ?form", FN_POLAR, ""},
	{"polar|phasor|cis|versor ?form|forms|representation|notation|"
	    "coordinates", FN_POLAR, ""},
	{"exponentials", FN_TRIG, " exp"},
	{"complex exponentials|exponential", FN_TRIG, " exp"},
	{"exponential|euler|eulers ?form|representation|notation", FN_EXPF, ""},
	{"rectangular|cartesian|algebraic|binomial|standard ?form|forms|"
	    "representation|notation|coordinates", FN_RECT, ""},
	{"a + bi|ib ?form", FN_RECT, "e"},	/* not at the start */
	{"a + b i ?form", FN_RECT, "e"},
	{"x + iy|yi ?form", FN_RECT, "e"},
	{"complex ?number form|representation", FN_RECT, ""},
	{"trigonometric|trig form|representation", FN_TRIGF, ""},
	{"modulus|magnitude|norm", FN_MOD, ""},
	{"absolute value", FN_MOD, "u"},	/* only with i */
	{"principal argument|value", FN_ARG, ""},
	{"trig|trigonometric rewrite|rewrites|identities|identity|forms|"
	    "alternatives|equivalents|simplification", FN_TRIG, ""},
	{"alternative|equivalent|other trig|trigonometric ?forms|form|"
	    "expressions", FN_TRIG, ""},
	{"double|double-angle ?angle ?identity|identities|formula|formulas|"
	    "formulae", FN_TRIG, " double"},
	{"power reduction|reducing ?formula|formulas|identity|identities",
	    FN_TRIG, " reduce"},
	{"power-reduction|power-reducing ?formula|formulas|identity|identities",
	    FN_TRIG, " reduce"},
	{"lower|first powers", FN_TRIG, " reduce"},
	{"product to sum|sums ?formula|formulas|identity|identities", FN_TRIG,
	    " product"},
	{"products to sums", FN_TRIG, " product"},
	{"product-to-sum ?formula|formulas|identity|identities", FN_TRIG,
	    " product"},
	{"a sum|sums", FN_TRIG, " product"},
	{"sum to product|products ?formula|formulas|identity|identities",
	    FN_TRIG, " sum"},
	{"sums to products", FN_TRIG, " sum"},
	{"sum-to-product ?formula|formulas|identity|identities", FN_TRIG, " sum"},
	{"a product", FN_TRIG, " sum"},
	{"sin|sine|sines and cos|cosine|cosines", FN_TRIG, " sincos"},
	{"sin / cos", FN_TRIG, " sincos"},
	{"compound|multiple|multiple-angle|triple|addition angle|angles|"
	    "formula|formulas|identity|identities", FN_TRIG, " expand"},
	{"angle addition|sum ?formula|formulas|identity|identities", FN_TRIG,
	    " expand"},
	{"tangent|tan half-angle|half ?angle ?substitution|formula|formulas",
	    FN_TRIG, " half"},
	{"half-angle|half angle ?formula|formulas|identity|identities", FN_TRIG,
	    " half"},
	{"weierstrass|t-substitution ?substitution|form", FN_TRIG, " half"},
	{"inverse trig|trigonometric ?functions|identities|form", FN_TRIG,
	    " inverse"},
	{"terms of arcsin|asin|arccos|acos|arctan|atan|atan2", FN_TRIG,
	    " inverse"},
	{"terms of inverse trig|trigonometric ?functions", FN_TRIG, " inverse"},
	{"harmonic|amplitude-phase ?form|addition", FN_TRIG, " harmonic"},
	{"amplitude phase ?form", FN_TRIG, " harmonic"},
	{"single sine|sinusoid|cosine|wave", FN_TRIG, " harmonic"},
	{"the form r|a sin|cos", FN_TRIG, " harmonic"},
	{"alternative|equivalent|other forms|form|expressions|rewrites|"
	    "identities", FN_TRIG, "t"},	/* only for trig */
	{"argument|arg|phase|angle", FN_ARG, ""},
	{NULL, 0, NULL}
};

static int
issym_at(int i, const char *s)
{
	return W[i].k == W_SYM && !strcmp(W[i].s, s);
}

static int
form_lead(int *verb)
{
	int k = 0;

	*verb = 0;
	for (;;) {
		if (mat("expand|expanding")) {
			*verb = 1;
		} else if (mat("simplify|simplifying")) {
			*verb = 2;
		} else if (mat("linearize|linearise|linearizing|linearising")) {
			*verb = 3;
		} else if (mat("rewrite|express|write|convert|put|transform|turn|"
		    "change|represent|rewriting|writing|converting|expressing|"
		    "use|apply|using|applying")) {
			if (!*verb)
				*verb = 4;
		} else if (isw(wi, "i") && (!strcmp(W[wi].o, "I") ||
		    peek("i want|need|would|have|am|wanna|write|rewrite|express|"
		    "convert|put|find|get|compute|calculate|turn|change|expand|"
		    "simplify"))) {
			wi++;
		} else if (mat("please|pls|plz|kindly|can|could|would|will|you|me|us|"
		    "we|want|need|like|to|how|do|does|what|whats|is|are|find|"
		    "compute|calculate|evaluate|give|get|show|tell|determine|list|"
		    "work|out|the|an|all|now|so|ok|okay|hey|hi|also|then|its|both|"
		    "and|in")) {
			;
		} else
			break;
		k++;
	}
	return k;
}

/* a form phrase at wi: its class; wi moves past it */
static int
form_phrase(const char **opt, int start)
{
	int k, s = wi;

	for (k = 0; form_phr[k].pat; k++) {
		wi = s;
		if (start && !strcmp(form_phr[k].opt, "e"))
			continue;	/* "x + iy to polar": the expression */
		if (mat(form_phr[k].pat) || (wi = s, mat("the|a|an|its") &&
		    mat(form_phr[k].pat))) {
			*opt = form_phr[k].opt;
			return form_phr[k].cls;
		}
	}
	wi = s;
	return FN_NONE;
}

/* an ordinal: 2, 2nd, second, square, cube, 5th, fifth */
static int
form_ordinal(int *n)
{
	static const char *const ow[] = {"", "", "square", "cube", "fourth",
		"fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
		"twelfth", NULL};
	int k;

	if (W[wi].k == W_NUM) {
		*n = atoi(W[wi].s);
		wi++;
		if (W[wi].k == W_WORD && !W[wi].sp && (isw(wi, "th") ||
		    isw(wi, "st") || isw(wi, "nd") || isw(wi, "rd")))
			wi++;
		else if (issym("-") && (isw(wi + 1, "th") || isw(wi + 1, "st") ||
		    isw(wi + 1, "nd") || isw(wi + 1, "rd")))
			wi += 2;
		return *n > 0;
	}
	for (k = 2; ow[k]; k++)
		if (isw(wi, ow[k])) {
			*n = k;
			wi++;
			return 1;
		}
	if (isw(wi, "second")) {
		*n = 2;
		wi++;
		return 1;
	}
	if (isw(wi, "third")) {
		*n = 3;
		wi++;
		return 1;
	}
	return 0;
}

static int
has_trig_word(int a, int b)
{
	static const char *const tw[] = {"sin", "cos", "tan", "sec", "csc",
		"cot", "sinh", "cosh", "tanh", "asin", "acos", "atan", "atan2",
		"arcsin", "arccos", "arctan", "asinh", "acosh", "atanh", "sine",
		"cosine", "tangent", NULL};
	int i, k;

	for (i = a; i < b; i++)
		for (k = 0; tw[k]; k++)
			if (isw(i, tw[k]))
				return 1;
	return 0;
}

static int
has_unit_word(int a, int b)
{
	int i;

	for (i = a; i < b; i++)
		if ((W[i].k == W_WORD && (!strcmp(W[i].o, "i") ||
		    !strcmp(W[i].o, "j") || !strcmp(W[i].o, "I") ||
		    !strcmp(W[i].s, "im") || !strcmp(W[i].s, "cis"))) ||
		    (W[i].k == W_SYM && !strcmp(W[i].o, "\xe2\x88\xa0")))
			return 1;
	return 0;
}

/* the words a..b as ucas text: as written, or translated when English */
static const char *
form_expr(int a, int b)
{
	static const char *const ew[] = {"plus", "minus", "times", "over",
		"divided", "squared", "cubed", "root", "power", "half", "twice",
		"of", "sine", "cosine", "tangent", "logarithm", "quantity", "sum",
		"product", "raised", "multiplied", "one", "two", "three", "four",
		"five", "six", "seven", "eight", "nine", "ten", NULL};
	int i, k, eng = 0;

	while (b > a && W[b - 1].k == W_SYM && (!strcmp(W[b - 1].s, "?") ||
	    !strcmp(W[b - 1].s, ".") || !strcmp(W[b - 1].s, ",")))
		b--;
	for (i = a; i < b; i++)
		for (k = 0; ew[k]; k++)
			if (isw(i, ew[k]))
				eng = 1;
	if (eng) {
		jmp_buf saved;
		struct word wend = W[b];
		struct ex e;
		int s = wi, ok = 0;
		memcpy(saved, nl_jmp, sizeof saved);
		memset(&W[b], 0, sizeof W[b]);
		wi = a;
		quiet++;
		if (setjmp(nl_jmp) == 0) {
			e = parse_expr(P_TERN);
			ok = wi == b && e.s;
		}
		quiet--;
		memcpy(nl_jmp, saved, sizeof saved);
		W[b] = wend;
		wi = s;
		if (ok)
			return e.s;
	}
	return words_text(a, b);
}

static int
forms_sentence(void)
{
	int s = wi, verb, cls = FN_NONE, ps = -1, pe = -1, i, a, b, n = 0;
	const char *opt = "", *ex, *cmd = NULL;

	/* "polar plot of 1 + cos theta": plot.c's */
	for (i = wi; i < nw; i++)
		if (isw(i, "plot") || isw(i, "plots") || isw(i, "graph") ||
		    isw(i, "draw") || isw(i, "sketch") || isw(i, "chart") ||
		    isw(i, "visualize") || isw(i, "visualise") || isw(i, "plot3d"))
			return 0;
	form_lead(&verb);
	a = wi;
	/* "solve z^3 = 8i": the cube roots */
	wi = s;
	if (mat("solve") && W[wi].k == W_WORD && strlen(W[wi].s) == 1 &&
	    W[wi + 1].k == W_SYM && !strcmp(W[wi + 1].s, "^") &&
	    W[wi + 2].k == W_NUM && W[wi + 3].k == W_SYM &&
	    !strcmp(W[wi + 3].s, "=")) {
		n = atoi(W[wi + 2].s);
		ex = form_expr(wi + 4, nw);
		if (n < 1 || !*ex)
			goto no;
		Q.raw[0] = S("nthroots %d %s", n, ex);
		goto done;
	}
	/* "the 5th roots of unity", "the cube roots of 8i" */
	for (i = a; i < nw; i++) {
		wi = i;
		mat("?all ?the|of");
		/* a plain number needs a suffix: "the 2 roots of" is not this */
		if ((W[wi].k != W_NUM || (W[wi + 1].k == W_WORD && !W[wi + 1].sp) ||
		    issym_at(wi + 1, "-")) && form_ordinal(&n) &&
		    (mat("roots ?of|for") || (mat("root ?of|for") &&
		    has_unit_word(wi, nw)))) {
			mat("?the");
			if (mat("unity|one|1") && W[wi].k == W_END) {
				Q.raw[0] = S("unity %d", n);
				goto done;
			}
			mat("?the ?complex ?number");
			ex = form_expr(wi, nw);
			if (!*ex)
				goto no;
			Q.raw[0] = S("nthroots %d %s", n, ex);
			goto done;
		}
	}
	/* a form phrase: the first one, from the left */
	for (i = a; i < nw && cls == FN_NONE; i++) {
		wi = i;
		/* "a" or "i" inside the expression are not phrases */
		if (i > a && !(isw(i, "in") || isw(i, "into") || isw(i, "to") ||
		    isw(i, "as") || isw(i, "using") || isw(i, "with") ||
		    isw(i, "via") || isw(i, "by") || isw(i, "terms") ||
		    isw(i, "the") || isw(i, "of") || isw(i, "for") || isw(i, "its") ||
		    isw(i - 1, "in") || isw(i - 1, "into") || isw(i - 1, "to") ||
		    isw(i - 1, "as") || isw(i - 1, "using") || isw(i - 1, "of") ||
		    isw(i - 1, "with") || isw(i - 1, "via") || isw(i - 1, "the")))
			continue;
		mat("?in|into|to|as|using|with|via|by");
		if (!peek("terms of arcsin|asin|arccos|acos|arctan|atan|atan2|"
		    "inverse"))
			mat("?terms of");
		ps = i;
		cls = form_phrase(&opt, i == a);
		pe = wi;
	}
	if (cls == FN_NONE) {
		/* "expand sin(a + b)", "simplify sin x cos x" */
		if (verb < 1 || verb > 3)
			goto no;
		mat("?the ?expression|function");
		b = nw;
		if (!has_trig_word(a, b) && !has_unit_word(a, b))
			goto no;
		ex = form_expr(a, b);
		if (has_unit_word(a, b) && !has_trig_word(a, b))
			Q.raw[0] = S("complex %s", ex);
		else
			Q.raw[0] = S("trig %s%s", ex, verb == 1 ? " expand" :
			    verb == 3 ? " reduce" : "");
		goto done;
	}
	/* the expression: after the phrase ("polar form of z"), or before
	 * it ("write z in polar form") */
	if (ps == a) {
		wi = pe;
		mat("?form|forms|representation|notation|parts|part");
		mat("?of|for|on");
		mat("?the ?complex ?number|expression|function");
		b = nw;
		a = wi;
	} else {
		b = ps;
		while (b > a && (isw(b - 1, "in") || isw(b - 1, "into") ||
		    isw(b - 1, "to") || isw(b - 1, "as") || isw(b - 1, "using") ||
		    isw(b - 1, "with") || isw(b - 1, "via") || isw(b - 1, "by") ||
		    isw(b - 1, "the") || isw(b - 1, "of") || isw(b - 1, "terms") ||
		    isw(b - 1, "its") || isw(b - 1, "form") || isw(b - 1, "for")))
			b--;
		/* "find the polar form ... of": no */
	}
	if (b <= a)
		goto no;
	if (!strcmp(opt, "u")) {
		if (!has_unit_word(a, b))
			goto no;	/* |x|: an expression, not a form */
		opt = "";
	}
	if (!strcmp(opt, "t")) {
		if (!has_trig_word(a, b))
			goto no;	/* other forms of x + 1: find */
		opt = "";
	}
	if (!strcmp(opt, "e"))
		opt = "";
	ex = form_expr(a, b);
	if (!*ex)
		goto no;
	switch (cls) {
	case FN_RECT: cmd = "complex"; break;
	case FN_POLAR: cmd = "polar"; break;
	case FN_RE: cmd = "re"; break;
	case FN_IM: cmd = "im"; break;
	case FN_CONJ: cmd = "conj"; break;
	case FN_MOD: cmd = "modulus"; break;
	case FN_ARG: cmd = "arg"; break;
	case FN_EXPF:
		if (!has_unit_word(a, b) && has_trig_word(a, b)) {
			Q.why = "an exponential form of a real trig expression: read as "
			    "its Euler form (say \"polar form\" for r e^(i theta))";
			cmd = "trig";
			opt = " exp";
		} else
			cmd = "polar";
		break;
	case FN_TRIGF:
		if (has_unit_word(a, b))
			cmd = "polar";
		else
			cmd = "trig";
		break;
	default:
		cmd = "trig";
	}
	if (strcmp(cmd, "trig"))
		opt = "";
	else if (verb == 1 && !strcmp(opt, " double"))
		opt = " expand";
	Q.raw[0] = S("%s %s%s", cmd, ex, opt);
done:
	Q.cmd = C_RAW;
	Q.nraw = 1;
	wi = nw;
	return 1;
no:
	wi = s;
	return 0;
}

/* ---- integrals drawn, solids of revolution (viz.c) ---- */

/*
 * "slice of integral f(x) = x^2 at 0 till 4", "area under x^2 from 0 to
 * 4", "shade the integral of sin x from 0 to pi", "the area between x^2
 * and x", "a left riemann sum of x^2 with 8 rectangles", "revolve f(x) =
 * x^2 in x-axis", "rotate x^2 around the y axis", "solid of revolution
 * of sqrt x from 0 to 4", "the volume of revolution of ...", "the surface
 * area of revolution of ...", "the arc length of ...", "the centroid of
 * the region between ...": one area/revolve/volume/... line.
 */
static const char *const viz_cmds[] = {"area", "revolve", "volume",
	"surfacearea", "arclength", "centroid", NULL};

static int word_at(const char *s, size_t n, size_t *i, char *w, size_t wn);

/* the raw command's text read as English? (nlp_is_english) */
static int
viz_english(const char *s, size_t n)
{
	static const char *const en[] = {"of", "under", "beneath", "below",
		"the", "a", "an", "till", "until", "axis", "line", "enclosed",
		"bounded", "curve", "curves", "graph", "function", "region",
		"solid", "revolution", "using", "method", "rectangles",
		"subintervals", "me", "please", "its", "it", "with", "sum", "sums",
		"rule", "obtained", "generated", NULL};
	size_t j = 0, k;
	char w[40];

	for (k = 0; k + 1 < n; k++)	/* f(x) = ... */
		if (s[k] == ')') {
			size_t q = k + 1;
			while (q < n && s[q] == ' ')
				q++;
			if (q < n && s[q] == '=' && (q + 1 >= n || s[q + 1] != '='))
				return 1;
		}
	while (j < n) {
		if (word_at(s, n, &j, w, sizeof w)) {
			if (in_list(en, w) && !user_fn(w))
				return 1;
			if (!strcmp(w, "in")) {	/* in x-axis */
				size_t q = j;
				while (q < n && s[q] == ' ')
					q++;
				if (q + 1 < n && (s[q] == 'x' || s[q] == 'y') &&
				    (s[q + 1] == '-' || s[q + 1] == ' '))
					return 1;
			}
			continue;
		}
		j++;
	}
	return 0;
}

/* "f(x) = x^2", "y = x^2": the body only */
static void
viz_skip_def(void)
{
	int k, d = 0;

	if (W[wi].k != W_WORD)
		return;
	if (issym_at(wi + 1, "=")) {
		wi += 2;
		return;
	}
	if (!issym_at(wi + 1, "("))
		return;
	for (k = wi + 1; W[k].k != W_END; k++)
		if (issym_at(k, "("))
			d++;
		else if (issym_at(k, ")") && --d == 0)
			break;
	if (W[k].k != W_END && issym_at(k + 1, "="))
		wi = k + 2;
}

/* about the x-axis, around the y axis, in x-axis, about the line y = 2 */
static const char *
viz_axis(void)
{
	int s = wi;
	const char *c;

	if (!mat("about|around|in|along|over|across"))
		return NULL;
	mat("?the");
	if (mat("x-axis|xaxis|abscissa") || mat("x - axis") || mat("x axis") ||
	    mat("horizontal axis"))
		return "x";
	if (mat("y-axis|yaxis|ordinate") || mat("y - axis") || mat("y axis") ||
	    mat("vertical axis"))
		return "y";
	mat("?line|lines");
	if ((isw(wi, "x") || isw(wi, "y")) && issym_at(wi + 1, "=")) {
		const char *v = W[wi].s;
		wi += 2;
		if ((c = plot_const()) != NULL)
			return S("%s = %s", v, c);
	}
	if ((isw(wi, "x") || isw(wi, "y")) && (isw(s, "about") ||
	    isw(s, "around")) && (W[wi + 1].k == W_END || W[wi + 1].k == W_WORD)) {
		wi++;
		return W[wi - 1].s;
	}
	wi = s;
	return NULL;
}

static int
viz_sentence(void)
{
	enum { A, REV, VOL, SURF, ARC, CEN };
	int s = wi, cmd = -1, verb, riemann = 0, accum = 0, still = 0, rot = 0;
	int nofile = 0, anim = 0, rev = 0, k;
	const char *kind = NULL, *n = NULL, *method = NULL, *axis = NULL;
	const char *lo = NULL, *hi = NULL, *nm = NULL, *anim_opt = NULL;
	struct sbuf sb = {0};
	static const char *const kinds = "left|right|midpoint|mid|middle|"
	    "trapezoid|trapezoidal|trapezium|simpson|simpsons|lower|upper";

	verb = mat("show|display|draw|plot|graph|shade|visualize|visualise|"
	    "illustrate|sketch|render|depict|animate|highlight|picture") ? 1 :
	    mat("find|compute|calculate|get|give|determine|evaluate|estimate|"
	    "tell") || mat("work out") || mat("what|whats ?is|are") ? 2 : 0;
	mat("?me|us");
	mat("?the|a|an");
	while (mat(kinds) || mat("-|endpoint|endpoints|point|hand"))
		if (!kind && W[wi - 1].k == W_WORD && strcmp(W[wi - 1].s,
		    "endpoint") && strcmp(W[wi - 1].s, "endpoints") &&
		    strcmp(W[wi - 1].s, "point") && strcmp(W[wi - 1].s, "hand"))
			kind = W[wi - 1].s;
	if (mat("slice|slices|sliver ?of ?the|an|a ?definite integral|integrals "
	    "?of|for")) {
		if (!isw(wi - 1, "integral") && !isw(wi - 1, "integrals") &&
		    !isw(wi - 2, "integral") && !isw(wi - 2, "integrals"))
			goto no;
		cmd = A;
	} else if (verb == 1 && mat("?definite integral|integrals ?of|for")) {
		cmd = A;
	} else if (mat("area|region|areas under|beneath|below|of")) {
		cmd = A;
	} else if (mat("area|region|areas ?that|which ?is|lies ?enclosed|bounded|"
	    "contained|trapped|caught between|by")) {
		cmd = A;
	} else if (mat("riemann|rectangle|rectangles|trapezoid|trapezoidal|"
	    "simpson|simpsons ?sum|sums|rule|approximation|approximations "
	    "?of|for") || (kind && mat("sum|sums|rule|approximation ?of|for"))) {
		cmd = A;
		riemann = 1;
		if (!kind && !isw(s, "riemann") && W[wi - 1].k == W_WORD) {
			for (k = s; k < wi; k++)
				if (isw(k, "trapezoid") || isw(k, "trapezoidal") ||
				    isw(k, "simpson") || isw(k, "simpsons"))
					kind = W[k].s;
		}
	} else if (mat("accumulation|accumulated ?area ?function ?of|for") ||
	    mat("area function ?of|for")) {
		cmd = A;
		accum = 1;
	} else if (mat("solid|solids ?of revolution ?of|for|from")) {
		cmd = verb == 2 ? VOL : REV;
		rev = 1;
	} else if (mat("volume|volumes") || (mat("surface area|areas") &&
	    (cmd = SURF))) {
		if (cmd != SURF)
			cmd = VOL;
		mat("?of ?the ?solid ?of revolution");
		rev = isw(wi - 1, "revolution");
		if (mat("?that|which ?is obtained|generated|formed|swept|made ?by|when"
		    " ?revolving|rotating|turning|spinning|sweeping") &&
		    W[wi - 1].k == W_WORD && strstr(W[wi - 1].s, "ing"))
			rev = 1;
		mat("?of|for");
		if (verb == 1 && cmd == VOL)
			cmd = REV;
	} else if (mat("arc|curve length|lengths") || mat("arclength") ||
	    mat("length of the curve|graph")) {
		cmd = ARC;
	} else if (mat("centroid|centroids") || mat("center|centre of mass|"
	    "gravity")) {
		cmd = CEN;
	} else if (mat("revolve|revolving|rotate|rotating|spin|spinning|turn|"
	    "turning|sweep|sweeping")) {
		cmd = REV;
		rev = isw(wi - 1, "revolve") || isw(wi - 1, "revolving");
		for (k = wi; k < nw && !rev; k++)
			if (isw(k, "axis") || isw(k, "about") || isw(k, "around") ||
			    isw(k, "x-axis") || isw(k, "y-axis"))
				rev = 1;
		if (!rev)
			goto no;
	} else
		goto no;
	if ((cmd == VOL || cmd == SURF) && !rev) {
		for (k = wi; k < nw && !rev; k++)
			if (isw(k, "about") || isw(k, "around") || isw(k, "axis") ||
			    isw(k, "x-axis") || isw(k, "y-axis") || isw(k, "revolution"))
				rev = 1;
		if (!rev)
			goto no;
	}
	mat("?of|for");
	mat("?the ?region|area|curve|curves|graph|graphs|function|functions|"
	    "parabola|line");
	mat("?that|which ?is|lies");
	mat("?under|beneath|below|between|of|enclosed|bounded|by");
	mat("?by|between");
	mat("?the ?curve|curves|graph|graphs|function|functions");
	viz_skip_def();
	plot_targets();
	for (;;) {
		struct ex a, b;
		const char *c;
		int s2 = wi, btw;
		if (W[wi].k == W_END)
			break;
		if (acc_sym(",") || acc_sym(";") || acc_sym(".") ||
		    mat("please|thanks|now|too|again|then|also|and") ||
		    mat("for me") || mat("as well"))
			continue;
		if ((c = viz_axis()) != NULL) {
			axis = c;
			continue;
		}
		if (mat("at|from|on|over|between|across|within|for|in")) {
			btw = isw(wi - 1, "between");
			mat("?the interval|range|segment");
			if ((isw(wi, "x") || isw(wi, "t")) && (mat("x|t from|in|between") ||
			    mat("x|t =")))
				btw |= isw(wi - 1, "between");
			if (const_interval(&a, &b, btw)) {
				lo = a.s;
				hi = b.s;
				continue;
			}
			wi = s2;
		}
		if (mat(kinds)) {
			kind = W[wi - 1].s;
			riemann = 1;
			continue;
		}
		if (mat("?riemann sum|sums|rule|approximation|approximations") &&
		    wi > s2) {
			riemann = 1;
			continue;
		}
		if (mat("riemann|endpoint|endpoints|point|points|hand")) {
			riemann |= isw(wi - 1, "riemann");
			continue;
		}
		if (mat("?with|using|in|into ?n ?=") && W[wi].k == W_NUM) {
			n = W[wi].s;
			wi++;
			mat("?rectangles|rectangle|trapezoids|trapezoid|subintervals|"
			    "intervals|pieces|strips|slices|parts|boxes|bars|steps");
			riemann = 1;
			continue;
		}
		wi = s2;
		if (mat("?using|by|with|via ?the disk|disc|disks|discs|washer|washers|"
		    "shell|shells ?method|methods")) {
			for (k = s2; k < wi; k++)
				if (W[k].s[0] == 'd' || W[k].s[0] == 'w')
					method = "washer";
				else if (W[k].s[0] == 's' && strncmp(W[k].s, "shell", 5) == 0)
					method = "shell";
			if (method)
				continue;
			wi = s2;
		}
		if (mat("?as ?an|a animated|animate|animation|animations") &&
		    wi > s2) {
			anim = 1;
			continue;
		}
		if (mat("?as ?a|an still|static|without|no ?animation|image|picture") &&
		    wi > s2) {
			still = 1;
			continue;
		}
		if (mat("?and ?then rotate|rotating|spin|spinning|turn|turning ?it|the "
		    "?solid ?around|round") && wi > s2) {
			rot = 1;
			continue;
		}
		if (mat("?with|and|plus ?its|the accumulation|accumulated ?area "
		    "?function") && wi > s2) {
			accum = 1;
			continue;
		}
		if (mat("nofile")) {
			nofile = 1;
			continue;
		}
		/* frames 12, in 12 frames, 30 fps, at 30 fps */
		if ((mat("frames|fps") && W[wi].k == W_NUM) ||
		    (mat("?in|with|at") && W[wi].k == W_NUM && (isw(wi + 1, "frames") ||
		    isw(wi + 1, "fps")))) {
			int f = isw(wi - 1, "frames") || isw(wi + 1, "frames");
			anim_opt = S("%s %s %s", anim_opt ? anim_opt : "", f ? "frames" :
			    "fps", W[wi].s);
			wi += isw(wi - 1, "frames") || isw(wi - 1, "fps") ? 1 : 2;
			continue;
		}
		wi = s2;
		if (isw(wi, "to") && W[wi + 1].k == W_WORD && W[wi + 2].k == W_END) {
			nm = W[wi + 1].o;	/* to NAME, the last words */
			wi += 2;
			continue;
		}
		if (mat("save|call|name ?it|that ?as") && W[wi].k == W_WORD) {
			nm = W[wi].o;
			wi++;
			continue;
		}
		if (parse_clause())
			continue;
		nfail("I didn't follow \"%s\"", words_text(wi, nw));
	}
	if (Q.npl == 0)
		nfail("of what? (a function)");
	if (Q.npl > 2)
		nfail("one or two functions, please");
	if (!lo && Q.ilo) {
		lo = Q.ilo;
		hi = Q.ihi;
	}
	for (k = 0; k < nv && !lo; k++)
		if (V[k].order && V[k].lo[0] && V[k].hi[0] && !V[k].ty[0]) {
			lo = S("%s", V[k].lo);
			hi = S("%s", V[k].hi);
		}
	sb_printf(&sb, "%s", cmd == A ? "area" : cmd == REV ? "revolve" : cmd ==
	    VOL ? "volume" : cmd == SURF ? "surfacearea" : cmd == ARC ?
	    "arclength" : "centroid");
	for (k = 0; k < Q.npl; k++)
		sb_printf(&sb, "%s%s", k ? " and " : " ", Q.pl[k].named ?
		    Q.pl[k].fn : Q.pl[k].e.s);
	if (lo)
		sb_printf(&sb, " from %s to %s", lo, hi);
	if (cmd == A && (riemann || kind)) {
		const char *kk = !kind ? NULL : !strcmp(kind, "middle") ||
		    !strcmp(kind, "mid") ? "mid" : !strncmp(kind, "trap", 4) ?
		    "trapezoid" : !strncmp(kind, "simpson", 7) ? "simpson" :
		    !strcmp(kind, "lower") ? "left" : !strcmp(kind, "upper") ? "right" :
		    kind;
		sb_printf(&sb, " riemann");
		if (n && !anim)
			sb_printf(&sb, " %s", n);
		if (kk)
			sb_printf(&sb, " %s", kk);
		if (kind && (!strcmp(kind, "lower") || !strcmp(kind, "upper")) && !dry)
			printf("  note: read \"%s sum\" as the %s endpoint sum (the %s "
			    "sum for an increasing function)\n", kind, kk, kind);
	}
	if (cmd == A && accum)
		sb_printf(&sb, " accumulate");
	if (axis && cmd != A && cmd != ARC && cmd != CEN)
		sb_printf(&sb, " about %s", axis);
	if (method && (cmd == REV || cmd == VOL))
		sb_printf(&sb, " %s", method);
	if (cmd == REV && still)
		sb_printf(&sb, " still");
	if (cmd == REV && rot)
		sb_printf(&sb, " rotate");
	if (anim_opt && cmd != VOL && cmd != SURF && cmd != ARC && cmd != CEN)
		sb_puts(&sb, anim_opt);
	if (nm)
		sb_printf(&sb, " to %s", nm);
	if (nofile)
		sb_printf(&sb, " nofile");
	Q.raw[0] = S("%s", sb.p);
	free(sb.p);
	Q.cmd = C_RAW;
	Q.nraw = 1;
	wi = nw;
	return 1;
no:
	wi = s;
	return 0;
}

#include "calc2.h"
#include "nlp_calc2.inc"   /* derivatives, limits, series ... (calc2) */

static void
sentence(void)
{
	int s, c, verb = 0, wordy;

	fillers();
	if (W[wi].k == W_END) {
		Q.cmd = C_NONE;
		return;
	}
	if (meta())
		return;
	if (viz_sentence())	/* area under, revolve, volume of revolution */
		return;
	if (definition())
		return;
	if (c2_sentence())	/* derivatives, limits, series (calc2) */
		return;
	if (forms_sentence())	/* polar form of 1 + i, expand sin(a + b) */
		return;
	s = wi;
	/* "sin x as a neighbourhood", "what does sin x look like" */
	mat("?show|display|draw|plot|graph|render|give|print|visualize|"
	    "visualise|sketch");
	mat("?me|us");
	if (plot_as() || plot_looks())
		return;
	wi = s;
	/* commands named by a verb */
	if (mat("differentiate|derive")) {
		target(&Q.a, P_TERN);
		Q.cmd = C_DIFF;
		return;
	}
	if (mat("tabulate|enumerate")) {
		target(&Q.a, P_TERN);
		Q.cmd = C_TABLE;
		return;
	}
	/* "alt ((1 << 3) & x) + x*5": cheaper alternatives, up to 10 */
	if (mat("alt|alts|alternatives|alternative|alternate|alternates")) {
		mat("?implementations|forms|versions|ways|rewrites|programs");
		mat("?to|of|for");
		mat("?computing|compute");
		target(&Q.a, P_TERN);
		Q.cmd = C_FIND;
		if (!Q.count)
			Q.count = 10;
		return;
	}
	if (mat("approximate|approx")) {
		mat("?the ?function");
		target(&Q.a, P_TERN);
		Q.cmd = C_APPROX;
		return;
	}
	/* "classify f", "what kind of function is f" (props.c: identify) */
	if (mat("identify|classify|characterize|characterise|categorize|"
	    "categorise|recognize|recognise|describe") ||
	    mat("what|which kind|type|sort|class|category of function|functions|"
	    "expression|map|thing|equation ?is|are") ||
	    mat("what|which kind|type|sort|class is|are") ||
	    mat("what properties|characteristics does|do|has|have")) {
		mat("?the|this ?function|expression|map");
		target(&Q.a, P_TERN);
		mat("?have|has|possess|satisfy");
		mat("?function|map");
		Q.cmd = C_PCMD;
		Q.pcmd = "identify";
		return;
	}
	if (mat("plot|graph|chart|draw|sketch|visualize|visualise|render|"
	    "depict|illustrate|picture")) {
		plot_sentence();
		return;
	}
	if (mat("simplify|expand|normalize|normalise|canonicalize|show|display|"
	    "print")) {
		if ((c = noun_cmd()) != 0) {
			cmd_after_noun(c);
			return;
		}
		wi = s + 1;
		target(&Q.a, P_TERN);
		Q.cmd = C_SHOW;
		return;
	}
	if (mat("optimize|optimise|improve") || mat("speed up")) {
		mat("?the ?implementation|computation|evaluation|code ?of|for");
		target(&Q.a, P_TERN);
		Q.cmd = C_FIND;
		return;
	}
	if (mat("synthesize|synthesise|build|design")) {
		mat("?a|an|the ?minimal|smallest|optimal ?logic ?circuit|gates|"
		    "network ?for|of");
		target(&Q.a, P_TERN);
		Q.cmd = C_GATES;
		return;
	}
	if (mat("fit|tune|calibrate")) {
		mat("?the ?constants|constant ?of|in");
		cur_mask = 1;
		target(&Q.a, P_BOR);
		if (!mat("so it matches") && !mat("to match|approximate") &&
		    !mat("to|against|~|for"))
			nfail("fit it to what?");
		second_target();
		Q.cmd = C_FIT;
		return;
	}
	if (mat("compare")) {
		two_targets("to|with|and|against|vs|versus");
		Q.cmd = float_target(&Q.a) || float_target(&Q.b) ? C_ERROR :
		    C_EQUIV;
		return;
	}
	if (mat("integrate|antidifferentiate")) {
		mat("?the ?function");
		target(&Q.a, P_TERN);
		Q.cmd = C_INTEG;
		return;
	}
	/* "solve y from 0 to 2": an ODE or PDE given in ucas syntax */
	if (mat("solve|dsolve")) {
		mat("?the ?ode|odes|pde|pdes|equation|equations|system");
		mat("?for");
		if (W[wi].k != W_WORD)
			nfail("solve what? (the ODE's unknown, e.g. solve y from 0 "
			    "to 2)");
		snprintf(Q.ode, sizeof Q.ode, "%s", W[wi].o);
		wi++;
		Q.cmd = C_DSOLVE;
		return;
	}
	if (mat("factor|factorize|factorise|prove"))
		nfail("ucas can't %s yet: it measures functions", W[s].o);
	if (mat("check|verify|test|confirm|decide|determine") &&
	    mat("whether|if|that")) {
		s = wi;
		if (yes_no(s))
			return;
	}
	wi = s;
	if (mat("how")) {
		how_question();
		return;
	}
	if (mat("where|when|wherever") || mat("for|on which|what ?inputs|values|"
	    "arguments|points") || mat("at what|which ?inputs|values|points")) {
		if (where_question())
			return;
		wi = s;
	}
	if (mat("is|are|does|do|can|could|will|would|might|may|did") &&
	    !peek("the range|domain|derivative")) {
		if (yes_no(wi))
			return;
	}
	wi = s;
	/* "what is the range of ...", "find faster ways to compute ..." */
	if (mat("what|which ?is|are|was|were") || mat("whats")) {
		verb = 1;
		mat("?is|are");
	} else if (mat("compute|calculate|evaluate|find|get|determine|give|list|"
	    "report|tell|estimate") || mat("work|figure out")) {
		verb = 2;
		mat("?me|us");
		if (isw(s, "find") || isw(s, "get") || isw(s, "list"))
			verb = 3;
	}
	s = wi;
	/* its range, f's derivative */
	if (mat("its|their")) {
		named_target(&Q.a, last_name() ? last_name() : "");
		if (!last_name())
			nfail("\"its\" has nothing to refer to yet");
		if ((c = noun_cmd()) != 0) {
			cmd_after_noun(c);
			return;
		}
		wi = s;
		memset(&Q.a, 0, sizeof Q.a);
	}
	if (W[wi].k == W_WORD && W[wi].poss && user_fn(W[wi].o)) {
		named_target(&Q.a, W[wi].o);
		wi++;
		if ((c = noun_cmd()) != 0) {
			cmd_after_noun(c);
			return;
		}
		wi = s;
		memset(&Q.a, 0, sizeof Q.a);
	}
	if ((c = noun_cmd()) != 0 && (c != C_TABLE || verb || peek("of|for"))) {
		cmd_after_noun(c);
		return;
	}
	wi = s;
	/* "x is an unsigned byte" on its own: an assumption */
	if (!verb) {
		char names[8][32];
		struct decl d;
		int n, ok;
		memset(&d, 0, sizeof d);
		quiet++;
		ok = parse_varlist(names, &n) && (peek("is|are") || issym(":")) &&
		    parse_descr(&d) && !d.val[0] && W[wi].k == W_END;
		quiet--;
		wi = s;
		if (ok) {
			assume();
			return;
		}
	}
	/* anything else is a value to work out */
	wordy = verb;
	(void)wordy;
	cur_mask = 1;
	target(&Q.a, P_TERN);
	Q.cmd = C_EVAL;
}

/* ---- output ---- */

static int
is_ident(char c)
{
	return isalnum((unsigned char)c) || c == '_';
}

/* rename calls: "W(" -> "Wm1(" as a whole identifier */
static const char *
subst_call(const char *s, const char *from, const char *to)
{
	struct sbuf sb = {0};
	size_t n = strlen(from);
	const char *p;

	for (p = s; *p;) {
		if (!strncmp(p, from, n) && p[n] == '(' && (p == s ||
		    !(isalnum((unsigned char)p[-1]) || p[-1] == '_'))) {
			sb_puts(&sb, to);
			p += n;
			continue;
		}
		sb_putc(&sb, *p++);
	}
	return sb.p ? sb.p : s;
}

/* replace the variable name in s by val, as a whole identifier */
static const char *
subst_ident(const char *s, const char *name, const char *val)
{
	struct sbuf sb = {0};
	size_t n = strlen(name);
	const char *p = s, *r;

	while (*p) {
		if (!strncmp(p, name, n) && (p == s || !is_ident(p[-1])) &&
		    !is_ident(p[n]) && p[n] != '(' && !(p > s && p[-1] == '.')) {
			sb_puts(&sb, val);
			p += n;
		} else
			sb_putc(&sb, *p++);
	}
	r = S("%s", sb.p ? sb.p : "");
	free(sb.p);
	return r;
}

/* the first use of the variable carries its type: isqrt(x: u4) */
static const char *
annotate(const char *s, const char *name, const char *ty)
{
	size_t n = strlen(name);
	const char *p;

	for (p = s; *p; p++) {
		if (strncmp(p, name, n) || (p > s && (is_ident(p[-1]) ||
		    p[-1] == '.')) || is_ident(p[n]) || p[n] == '(')
			continue;
		if (p > s + 1 && p[-1] == '(' && is_ident(p[-2]) &&
		    (p[n] == ')' || p[n] == ','))
			return S("%.*s%s: %s%s", (int)(p - s), s, name, ty, p + n);
		if (p > s && p[-1] == ' ' && p > s + 2 && p[-2] == ',' &&
		    (p[n] == ')' || p[n] == ','))
			return S("%.*s%s: %s%s", (int)(p - s), s, name, ty, p + n);
		return S("%.*s(%s: %s)%s", (int)(p - s), s, name, ty, p + n);
	}
	return s;
}

static void
addcond(struct var *v, const char *fmt)
{
	if (v->ncond < 4)
		snprintf(v->cond[v->ncond++], sizeof v->cond[0], fmt, v->name,
		    v->name);
}

static void
var_final(struct var *v)
{
	int64_t tlo, thi;
	int sized = type_limits(v->ty, &tlo, &thi), isint = v->ity == 1;

	if (v->adj & A_POS) {
		if (isint && !v->lo[0])
			strcpy(v->lo, "1");
		else if (!isint)
			addcond(v, "%s > 0");
	}
	if (v->adj & A_NEG) {
		if (isint && !v->hi[0])
			strcpy(v->hi, "-1");
		else if (!isint)
			addcond(v, "%s < 0");
	}
	if (v->adj & A_NONNEG) {
		if (isint && !v->lo[0])
			strcpy(v->lo, "0");
		else if (!isint)
			addcond(v, "%s >= 0");
	}
	if (v->adj & A_NONZERO)
		addcond(v, "%s != 0");
	if (v->adj & A_EVEN)
		addcond(v, "%s %% 2 == 0");
	if (v->adj & A_ODD)
		addcond(v, "%s %% 2 != 0");
	v->adj = 0;
	if (v->lo[0] && !v->hi[0] && sized && is_number(v->lo))
		snprintf(v->hi, sizeof v->hi, "%lld", (long long)thi);
	if (v->hi[0] && !v->lo[0] && sized && is_number(v->hi))
		snprintf(v->lo, sizeof v->lo, "%lld", (long long)tlo);
	if (v->lo[0] && (!v->hi[0] || !(is_number(v->lo) || (v->bc & 1)) ||
	    !(is_number(v->hi) || (v->bc & 2)))) {
		if (v->ncond < 4)
			snprintf(v->cond[v->ncond++], sizeof v->cond[0], "%s >= %s",
			    v->name, v->lo);
		if (v->hi[0] && v->ncond < 4)
			snprintf(v->cond[v->ncond++], sizeof v->cond[0], "%s <= %s",
			    v->name, v->hi);
		v->lo[0] = v->hi[0] = 0;
	} else if (v->hi[0] && !v->lo[0]) {
		if (v->ncond < 4)
			snprintf(v->cond[v->ncond++], sizeof v->cond[0], "%s <= %s",
			    v->name, v->hi);
		v->hi[0] = 0;
	}
}

static const char *
value_text(const char *val)
{
	if (is_number(val) && val[0] != '-')
		return val;
	return S("(%s)", val);
}

/* the parameters of a definition: mask_first's variables, then the rest */
static int
collect(int mask_first, int mask_all, struct var **ps)
{
	int n = 0, k, o, pass;

	for (pass = 0; pass < 2; pass++)
		for (o = 1; o <= norder; o++)
			for (k = 0; k < nv; k++) {
				struct var *v = &V[k];
				int want = pass == 0 ? mask_first : mask_all;
				if (v->order != o || v->val[0] || !(v->uses & want))
					continue;
				if (pass == 1 && (v->uses & mask_first))
					continue;
				if (n < MAXP)
					ps[n++] = v;
			}
	return n;
}

static int
any_scope_on(struct func *f)
{
	struct var *v;
	int k;

	if (Q.nscond)
		return 1;
	for (k = 0; k < f->np; k++)
		if ((v = var_find(sym_str(f->p[k].name))) != NULL && (v->decl ||
		    v->val[0]))
			return 1;
	return 0;
}

static const char *fn_used;

/*
 * The ucas argument for a target: a function name, an annotated
 * anonymous expression (inline_ok), or a definition emitted first.
 */
static const char *
target_fn(struct tgt *t, int mask, const char *defname, int inline_ok,
    int mask_first, int mask_all)
{
	struct var *ps[MAXP];
	struct func *f;
	struct sbuf sb = {0};
	const char *body, *r, *name;
	int np, k, j, needdef, ncond = 0;

	if (t->named) {
		f = user_fn(t->fn);
		if (f == NULL || !any_scope_on(f)) {
			fn_used = t->fn;
			return t->fn;
		}
		for (k = 0; k < f->np; k++) {
			struct var *v = var_get(sym_str(f->p[k].name));
			v->uses |= mask;
		}
		sb_printf(&sb, "%s(", t->fn);
		for (k = 0; k < f->np; k++)
			sb_printf(&sb, "%s%s", k ? ", " : "", sym_str(f->p[k].name));
		sb_putc(&sb, ')');
		t->e = mkatom(S("%s", sb.p), 0, 0);
		sb_clear(&sb);
	}
	body = t->e.s;
	for (k = 0; k < nv; k++)
		if (V[k].val[0] && (V[k].uses & mask))
			body = subst_ident(body, V[k].name, value_text(V[k].val));
	np = collect(mask_first, mask_all, ps);
	if (np == 0) {
		free(sb.p);
		if (Q.nscond)
			nfail("there is no variable for \"%s\" to restrict",
			    Q.scond[0]);
		return body;
	}
	/*
	 * Always a definition: ucas alternates its anonymous slots, so "it"
	 * could not name an anonymous function reliably.
	 */
	needdef = 1;
	for (k = 0; k < np; k++) {
		var_final(ps[k]);
		if (ps[k]->lo[0] || ps[k]->ncond)
			needdef = 1;
	}
	if (!needdef) {
		for (k = 0; k < np; k++)
			if (ps[k]->ty[0])
				body = annotate(body, ps[k]->name, ps[k]->ty);
		free(sb.p);
		fn_used = "_";
		return body;
	}
	name = Q.defname[0] && mask == 1 ? Q.defname : defname;
	sb_printf(&sb, "%s(", name);
	for (k = 0; k < np; k++) {
		sb_printf(&sb, "%s%s", k ? ", " : "", ps[k]->name);
		if (ps[k]->ty[0])
			sb_printf(&sb, ": %s", ps[k]->ty);
		if (ps[k]->lo[0])
			sb_printf(&sb, " in %s..%s", ps[k]->lo, ps[k]->hi);
	}
	sb_printf(&sb, ") = %s", body);
	for (k = 0; k < np; k++)
		for (j = 0; j < ps[k]->ncond; j++)
			sb_printf(&sb, "%s%s", ncond++ ? ", " : " where ",
			    ps[k]->cond[j]);
	for (k = 0; k < Q.nscond; k++)
		sb_printf(&sb, "%s%s", ncond++ ? ", " : " where ", Q.scond[k]);
	emit(sb.p);
	free(sb.p);
	r = S("%s", name);
	if (mask == 1)
		fn_used = r;
	return r;
}

/* "on [a, b]" without a variable: it is for the only one */
static void
apply_interval(void)
{
	struct var *only = NULL;
	int k, n = 0;

	if (!Q.ilo)
		return;
	for (k = 0; k < nv; k++)
		if (V[k].order && !V[k].val[0] && (V[k].uses & 1)) {
			n++;
			only = &V[k];
		}
	if (n == 0)
		return;
	if (n > 1)
		nfail("which variable is [%s, %s] for?", Q.ilo, Q.ihi);
	snprintf(only->lo, sizeof only->lo, "%s", Q.ilo);
	snprintf(only->hi, sizeof only->hi, "%s", Q.ihi);
	only->bc = 3;
	only->decl = 1;
	Q.interval = 1;
	/* [0, 1] is an interval of reals, not the integers 0 and 1 */
	if (!only->ty[0] && only->ity != 1 && is_int(Q.ilo) && is_int(Q.ihi)) {
		snprintf(only->ty, sizeof only->ty, "real");
		if (!dry)
			printf("  note: %s taken as real on [%s, %s] (say \"integer "
			    "%s\" for whole numbers)\n", only->name, Q.ilo, Q.ihi,
			    only->name);
	}
}

/* a sensible interval for a function nobody gave one for */
static void
default_interval(const char *body, const char **lo, const char **hi)
{
	if (strstr(body, "tan(")) {
		*lo = "-pi/4";
		*hi = "pi/4";
	} else if (strstr(body, "csc(") || strstr(body, "cot(")) {
		*lo = "pi/4";
		*hi = "3*pi/4";
	} else if (strstr(body, "sin(") || strstr(body, "cos(") ||
	    strstr(body, "sec(")) {
		*lo = "-pi/2";
		*hi = "pi/2";
	} else if (strstr(body, "log(") || strstr(body, "sqrt(")) {
		*lo = "1";
		*hi = "2";
	} else {
		*lo = "-1";
		*hi = "1";
	}
}

static const char *
var_decl(const struct var *v, const char *name)
{
	const char *s = name;

	if (v->ty[0])
		s = S("%s: %s", s, v->ty);
	if (v->lo[0])
		s = S("%s in %s..%s", s, v->lo, v->hi);
	return s;
}

/* a property as two functions that must agree: equiv lhs rhs */
static void
emit_prop(void)
{
	struct var *ps[MAXP];
	struct func *f;
	const char *F, *pn[MAXP], *pd[MAXP], *x, *y, *z, *dx, *dy, *dz;
	const char *law;
	struct sbuf l = {0}, r = {0};
	int np, k;

	F = target_fn(&Q.a, 1, "it", 0, 1, 1);
	if ((f = user_fn(F)) != NULL && fn_used == Q.a.fn) {
		np = f->np;
		for (k = 0; k < np; k++) {
			pn[k] = sym_str(f->p[k].name);
			pd[k] = fparam_decl(f, k);
		}
	} else {
		np = collect(1, 1, ps);
		for (k = 0; k < np; k++) {
			pn[k] = ps[k]->name;
			pd[k] = var_decl(ps[k], ps[k]->name);
		}
	}
	if (np == 0)
		nfail("%s is a constant", F);
	x = pn[0];
	dx = pd[0];
	/* a second and third variable of the first one's type */
	y = np > 1 ? pn[1] : S("%s2", x);
	dy = np > 1 ? pd[1] : S("%s2%s", x, dx + strlen(x));
	z = S("%s3", x);
	dz = S("%s3%s", x, dx + strlen(x));
	switch (Q.prop) {
	case PR_COMM:
		if (np < 2 && Q.pcmd) {
			/* "is f symmetric" for one argument: about an axis */
			emit(S("symmetry %s", F));
			return;
		}
		if (np < 2)
			nfail("commutative needs two arguments; %s has one", F);
		sb_printf(&l, "lhs(%s", dx);
		for (k = 1; k < np; k++)
			sb_printf(&l, ", %s", pd[k]);
		sb_printf(&l, ") = %s(%s, %s", F, x, y);
		sb_printf(&r, "rhs(%s", dx);
		for (k = 1; k < np; k++)
			sb_printf(&r, ", %s", pd[k]);
		sb_printf(&r, ") = %s(%s, %s", F, y, x);
		for (k = 2; k < np; k++) {
			sb_printf(&l, ", %s", pn[k]);
			sb_printf(&r, ", %s", pn[k]);
		}
		sb_putc(&l, ')');
		sb_putc(&r, ')');
		law = S("commutative: %s(%s, %s) = %s(%s, %s) everywhere?", F, x, y,
		    F, y, x);
		break;
	case PR_ASSOC:
		if (np != 2)
			nfail("associative needs two arguments");
		sb_printf(&l, "lhs(%s, %s, %s) = %s(%s(%s, %s), %s)", dx, dy, dz, F,
		    F, x, y, z);
		sb_printf(&r, "rhs(%s, %s, %s) = %s(%s, %s(%s, %s))", dx, dy, dz, F,
		    x, F, y, z);
		law = S("associative: %s(%s(a, b), c) = %s(a, %s(b, c)) everywhere?",
		    F, F, F, F);
		break;
	case PR_ADD:
		if (np != 1)
			nfail("additive is about functions of one argument");
		sb_printf(&l, "lhs(%s, %s) = %s(%s + %s)", dx, dy, F, x, y);
		sb_printf(&r, "rhs(%s, %s) = %s(%s) + %s(%s)", dx, dy, F, x, F, y);
		law = S("additive: %s(a + b) = %s(a) + %s(b) everywhere?", F, F, F);
		break;
	case PR_IDEM:
		if (np != 1)
			nfail("idempotent is about functions of one argument");
		sb_printf(&l, "lhs(%s) = %s(%s(%s))", dx, F, F, x);
		sb_printf(&r, "rhs(%s) = %s(%s)", dx, F, x);
		law = S("idempotent: %s(%s(x)) = %s(x) everywhere?", F, F, F);
		break;
	case PR_ODD:
	case PR_EVEN:
		if (np != 1)
			nfail("odd and even are about functions of one argument");
		sb_printf(&l, "lhs(%s) = %s(-%s)", dx, F, x);
		sb_printf(&r, "rhs(%s) = %s%s(%s)", dx, Q.prop == PR_ODD ? "-" : "",
		    F, x);
		law = S("%s: %s(-x) = %s%s(x) everywhere?", Q.prop == PR_ODD ? "odd" :
		    "even", F, Q.prop == PR_ODD ? "-" : "", F);
		break;
	default:
		sb_printf(&l, "lhs(%s", dx);
		for (k = 1; k < np; k++)
			sb_printf(&l, ", %s", pd[k]);
		sb_printf(&l, ") = %s(%s", F, x);
		for (k = 1; k < np; k++)
			sb_printf(&l, ", %s", pn[k]);
		sb_printf(&l, ") %% 2");
		emit(l.p);
		emit("range lhs");
		if (!dry)
			printf("  note: %s for every input when the range is %s\n",
			    Q.prop == PR_ODDV ? "odd" : "even", Q.prop == PR_ODDV ?
			    "1..1" : "0..0");
		free(l.p);
		return;
	}
	if (!dry)
		printf("  note: %s\n", law);
	emit(l.p);
	emit(r.p);
	emit("equiv lhs rhs");
	free(l.p);
	free(r.p);
}

static void
emit_approx(void)
{
	struct var *ps[MAXP];
	struct func *f;
	const char *F, *lo = NULL, *hi = NULL, *body;
	int np, ranged;
	struct sbuf sb = {0};

	F = target_fn(&Q.a, 1, "it", 0, 1, 1);
	if ((f = user_fn(F)) != NULL && fn_used == Q.a.fn) {
		np = f->np;
		ranged = np == 1 && f->p[0].ranged;
		body = node_str(f->body);
	} else {
		np = collect(1, 1, ps);
		ranged = np == 1 && ps[0]->lo[0];
		body = Q.a.e.s;
	}
	if (np != 1)
		nfail("approx needs a function of one variable (%s has %d)", F, np);
	if (!ranged) {
		default_interval(body, &lo, &hi);
		if (!no_more_prompts && (interactive() || nlp_defer) && !dry) {
			need_more = S(
"approximating %s needs an interval (default %s to %s).\n"
"    Reply with any of these, in any order (empty: the defaults):\n"
"      an interval      -pi/2 to pi/2, [0, 1], -90deg to 90deg, -1r to 1r\n"
"      a strategy       cheby poly (Chebyshev, the default here: %s),\n"
"                       econ (Chebyshev-economised Taylor), minimax\n"
"      critical points  crit (keep the extrema exact) or nocrit (%s)\n"
"      degree N, f32 or f64, in C",
			    body, lo, hi, Q.method[0] ? Q.method : "minimax",
			    Q.crit ? "crit" : "nocrit");
			longjmp(nl_jmp, 1);
		}
		if (!dry)
			printf("  note: no interval given: using %s to %s (say \"on a "
			    "to b\")\n", lo, hi);
	}
	if (!Q.degree[0] && !Q.acc[0] && !no_more_prompts &&
	    (interactive() || nlp_defer) && !dry) {
		need_more = S(
"how precise should the approximation of %s be?\n"
"    Reply with a degree (6, degree 6, degree 4 to 8) or an accuracy\n"
"    (within 1e-6, 2 ulp); empty: every degree up to the format's precision",
		    body);
		longjmp(nl_jmp, 1);
	}
	sb_printf(&sb, "approx %s", F);
	if (lo)
		sb_printf(&sb, " in %s..%s", lo, hi);
	if (Q.method[0])
		sb_printf(&sb, " method %s", Q.method);
	if (Q.degree[0])
		sb_printf(&sb, " degree %s", Q.degree);
	if (Q.acc[0])
		sb_printf(&sb, " %s", Q.acc);
	if (Q.ends > 0)
		sb_printf(&sb, " keep endpoints");
	else if (Q.ends < 0)
		sb_printf(&sb, " free");
	if (Q.aty[0] || Q.emit_c)
		sb_printf(&sb, " as%s%s", Q.aty[0] ? S(" %s", Q.aty) : "",
		    Q.emit_c ? " c" : "");
	if (Q.crit)
		sb_printf(&sb, " keep crit");
	emit(sb.p);
	free(sb.p);
}

static void
finish(void)
{
	struct sbuf sb = {0};
	const char *a, *b;
	int k, np;
	struct var *ps[MAXP];

	sb_clear(&out);
	fn_used = NULL;
	if (Q.why && !dry)
		printf("  note: %s\n", Q.why);
	if (c2_finish())
		return;
	if (Q.cmd == C_PLOT) {
		emit_plot();
		return;
	}
	if (Q.cmd == C_DSOLVE) {
		sb_printf(&sb, "dsolve %s", Q.ode);
		if (Q.ilo)
			sb_printf(&sb, " from %s to %s", Q.ilo, Q.ihi);
		if (Q.steps)
			sb_printf(&sb, " steps %d", Q.steps);
		emit(sb.p);
		free(sb.p);
		return;
	}
	if (Q.cmd == C_INTEG) {
		/* the interval is the integral's bounds, not the domain */
		const char *lo = Q.ilo, *hi = Q.ihi;
		for (k = 0; k < nv && !lo; k++)
			if (V[k].order && (V[k].uses & 1) && V[k].lo[0] &&
			    V[k].hi[0] && !V[k].ty[0]) {
				lo = S("%s", V[k].lo);
				hi = S("%s", V[k].hi);
				V[k].lo[0] = V[k].hi[0] = 0;
			}
		a = target_fn(&Q.a, 1, "it", 0, 1, 1);
		sb_printf(&sb, "integrate %s", a);
		if (Q.wrt[0])
			sb_printf(&sb, " wrt %s", Q.wrt);
		if (lo)
			sb_printf(&sb, " from %s to %s", lo, hi);
		emit(sb.p);
		free(sb.p);
		return;
	}
	if (Q.cmd != C_RAW && Q.cmd != C_NONE)
		apply_interval();
	if (Q.method[0] && (Q.cmd == C_EVAL || Q.cmd == C_RANGE ||
	    Q.cmd == C_SHOW || Q.cmd == C_FIT))
		Q.cmd = C_APPROX;
	else if (Q.cmd == C_EVAL && (Q.interval || Q.nocrit))
		Q.cmd = C_RANGE;
	if (Q.cmd == C_FIND && !Q.count)
		Q.count = 10;
	switch (Q.cmd) {
	case C_NONE:
		return;
	case C_RAW:
		for (k = 0; k < Q.nraw; k++)
			emit(Q.raw[k]);
		if (Q.defname[0])
			fn_used = Q.defname;
		return;
	case C_PROP:
		emit_prop();
		return;
	case C_APPROX:
		emit_approx();
		return;
	case C_DEF:
		for (k = 0; k < nv; k++)
			if (V[k].val[0])
				Q.a.e.s = subst_ident(Q.a.e.s, V[k].name,
				    value_text(V[k].val));
		if (Q.a.named) {
			struct func *f = user_fn(Q.a.fn);
			sb_printf(&sb, "%s", Q.a.fn);
			if (f && f->np) {
				sb_putc(&sb, '(');
				for (k = 0; k < f->np; k++)
					sb_printf(&sb, "%s%s", k ? ", " : "",
					    sym_str(f->p[k].name));
				sb_putc(&sb, ')');
			}
			Q.a.e = mkatom(S("%s", sb.p), 0, 0);
			sb_clear(&sb);
		}
		/* explicit parameters first, in their order */
		for (k = 0; k < Q.ndparam; k++) {
			struct var *v = var_get(Q.dparam[k]);
			v->uses |= 1;
		}
		np = collect(1, 1, ps);
		sb_printf(&sb, "%s", Q.defname);
		if (np || Q.dexplicit) {
			int j, nc = 0;
			sb_putc(&sb, '(');
			for (k = 0; k < np; k++) {
				var_final(ps[k]);
				sb_printf(&sb, "%s%s", k ? ", " : "", ps[k]->name);
				if (ps[k]->ty[0])
					sb_printf(&sb, ": %s", ps[k]->ty);
				if (ps[k]->lo[0])
					sb_printf(&sb, " in %s..%s", ps[k]->lo, ps[k]->hi);
			}
			sb_printf(&sb, ") = %s", Q.a.e.s);
			for (k = 0; k < np; k++)
				for (j = 0; j < ps[k]->ncond; j++)
					sb_printf(&sb, "%s%s", nc++ ? ", " : " where ",
					    ps[k]->cond[j]);
			for (k = 0; k < Q.nscond; k++)
				sb_printf(&sb, "%s%s", nc++ ? ", " : " where ",
				    Q.scond[k]);
		} else
			sb_printf(&sb, " = %s", Q.a.e.s);
		emit(sb.p);
		sb_printf(&sb, "");
		free(sb.p);
		sb.p = NULL;
		emit(S("show %s", Q.defname));
		fn_used = S("%s", Q.defname);
		return;
	case C_EQUIV:
	case C_ERROR:
	case C_FIT:
		if (Q.cmd == C_EQUIV) {
			a = target_fn(&Q.a, 1, "it", 0, 3, 3);
			b = target_fn(&Q.b, 2, "that", 0, 3, 3);
			emit(S("equiv %s %s", a, b));
		} else {
			/* G (a) takes F's (b) parameters first, then its own */
			b = target_fn(&Q.b, 2, "that", 0, 2, 2);
			a = target_fn(&Q.a, 1, "it", 0, 2, 3);
			if (Q.cmd == C_FIT)
				emit(S("fit %s ~ %s%s", a, b, Q.metric == 1 ?
				    " metric maxabs" : Q.metric == 2 ? " metric rms" : ""));
			else
				emit(S("error %s ~ %s", a, b));
		}
		return;
	default:
		break;
	}
	if (Q.cmd == C_PCMD)
		/* "on [0, 7]": a real interval unless a type was given */
		for (k = 0; k < nv; k++)
			if (!V[k].ty[0] && V[k].lo[0] && (V[k].uses & 1))
				snprintf(V[k].ty, sizeof V[k].ty, "real");
	a = target_fn(&Q.a, 1, "it", !(Q.cmd == C_DIFF && Q.wrt[0]), 1, 1);
	switch (Q.cmd) {
	case C_EVAL:
		np = 0;
		for (k = 0; k < nv; k++)
			if (V[k].order && !V[k].val[0] && (V[k].uses & 1))
				np++;
		if (Q.a.named && fn_used == Q.a.fn)
			emit(S("show %s", a));
		else if (np == 0)
			emit(a);
		else
			emit(S("show %s", a));
		break;
	case C_SHOW:
		emit(S("show %s", a));
		break;
	case C_TABLE:
		emit(Q.count ? S("table %s max %d", a, Q.count) : S("table %s", a));
		break;
	case C_RANGE:
		emit(S("range %s%s", a, Q.nocrit ? " nocrit" : ""));
		break;
	case C_DOMAIN:
		emit(S("domain %s", a));
		break;
	case C_WRAP:
		emit(S("wrap %s", a));
		break;
	case C_CRIT:
		emit(S("crit %s", a));
		break;
	case C_CLOSED:
		emit(S("closed %s", a));
		break;
	case C_SCHEME:
		emit(Q.model[0] ? S("scheme %s model %s", a, Q.model) :
		    S("scheme %s", a));
		break;
	case C_CONT:
		emit(S("continuous %s", a));
		break;
	case C_DIFF:
		emit(Q.wrt[0] ? S("diff %s %s", a, Q.wrt) : S("diff %s", a));
		break;
	case C_COST:
		emit(Q.model[0] ? S("cost %s model %s", a, Q.model) :
		    S("cost %s", a));
		break;
	case C_FIND:
		emit(S("find %s%s%s", a, Q.count ? S(" %d", Q.count) : "",
		    Q.maxc ? S(" max %d", Q.maxc) : ""));
		break;
	case C_GATES:
		emit(S("gates %s%s%s%s", a, Q.basis[0] ? S(" basis %s", Q.basis) :
		    "", Q.maxc ? S(" max %d", Q.maxc) : "", Q.emit_c ? " as c" : ""));
		break;
	case C_PCMD:
		emit(S("%s %s", Q.pcmd, a));
		break;
	}
}

/* ---- driver ---- */

static void
reset_sentence(int pass)
{
	int k;

	arn = 0;
	memset(&Q, 0, sizeof Q);
	wi = 0;
	norder = 0;
	cur_mask = 1;
	incond = quiet = 0;
	session_decl = 0;
	nv = 0;
	for (k = 0; k < nsv; k++) {
		V[nv] = SV[k];
		V[nv].decl = 0;
		V[nv].order = V[nv].uses = 0;
		nv++;
	}
	if (pass == 2)
		for (k = 0; k < np1; k++)
			if (P1[k].decl && P1[k].ty[0]) {
				struct var *v = var_get(P1[k].name);
				snprintf(v->ty, sizeof v->ty, "%s", P1[k].ty);
				v->ity = P1[k].ity;
			}
}

/* ---- misspelt words ---- */

/* every word of four or more letters in this file's strings (Makefile) */
static const char *const dict[] = {
#include "nlp_words.h"
	NULL
};

static int
in_dict(const char *w)
{
	return in_list(dict, w);
}

/* optimal string alignment distance, or more than max */
static int
osa(const char *a, const char *b, int max)
{
	int la = (int)strlen(a), lb = (int)strlen(b), i, j, v, rowmin;
	static int d[48][48];

	if (la >= 48 || lb >= 48 || abs(la - lb) > max)
		return max + 1;
	for (i = 0; i <= la; i++)
		d[i][0] = i;
	for (j = 0; j <= lb; j++)
		d[0][j] = j;
	for (i = 1; i <= la; i++) {
		rowmin = max + 1;
		for (j = 1; j <= lb; j++) {
			v = d[i - 1][j] + 1;
			if (d[i][j - 1] + 1 < v)
				v = d[i][j - 1] + 1;
			if (d[i - 1][j - 1] + (a[i - 1] != b[j - 1]) < v)
				v = d[i - 1][j - 1] + (a[i - 1] != b[j - 1]);
			if (i > 1 && j > 1 && a[i - 1] == b[j - 2] &&
			    a[i - 2] == b[j - 1] && d[i - 2][j - 2] + 1 < v)
				v = d[i - 2][j - 2] + 1;
			d[i][j] = v;
			if (v < rowmin)
				rowmin = v;
		}
		if (rowmin > max)
			return max + 1;
	}
	return d[la][lb];
}

/* edits allowed for a word of n letters */
static int
spell_limit(size_t n)
{
	return n >= 7 ? 2 : n >= 5 ? 1 : 0;
}

/* the known words nearest to w (same first letter preferred) */
static int
closest(const char *w, const char **best, int nbest)
{
	const char *const *lists[2] = {dict, greek};
	int lim = spell_limit(strlen(w)), bd = lim + 1, n = 0, d, l, k, same;

	for (l = 0; l < 2; l++)
		for (k = 0; lists[l][k]; k++) {
			if ((d = osa(w, lists[l][k], lim)) > lim || d > bd)
				continue;
			if (d < bd) {
				bd = d;
				n = 0;
			}
			if (n < nbest)
				best[n++] = lists[l][k];
		}
	for (k = same = 0; k < n; k++)
		if (best[k][0] == w[0])
			best[same++] = best[k];
	return same ? same : n;
}

static int
protected_word(int i)
{
	const char *w = W[i].s, *p;

	if (W[i].k != W_WORD || in_dict(w) || reserved_word(w) ||
	    in_list(builtins, w) || in_list(greek, w) || user_fn(W[i].o) ||
	    var_find(W[i].o) || known_name(W[i].o))
		return 1;
	for (p = w; *p; p++)
		if (!isalpha((unsigned char)*p) && *p != '-')
			return 1;
	/* a name being given: let foo be, call it foo, foo(x) = */
	if (i > 0 && (isw(i - 1, "let") || isw(i - 1, "define") ||
	    (isw(i - 1, "it") && i > 1 && (isw(i - 2, "call") ||
	    isw(i - 2, "name")))))
		return 1;
	if (W[i + 1].k == W_SYM && (!strcmp(W[i + 1].s, "(") ||
	    !strcmp(W[i + 1].s, "=") || !strcmp(W[i + 1].s, ":")))
		return 1;
	return 0;
}

static void
fix_word(int i, int span, const char **best, int n)
{
	const char *was = S("%s", words_text(i, i + span)), *opt[5];
	int k, c = 0;

	if (n > 1) {
		for (k = 0; k < n && k < 4; k++)
			opt[k] = best[k];
		opt[k] = S("\"%s\" as written", was);
		c = clarify(NULL, S("\"%s\" is not a word I know", was), opt, k + 1,
		    0, NULL);
		if (c == k)
			return;
	} else if (!spell_quiet)
		printf("  note: read \"%s\" as \"%s\"\n", was, best[0]);
	snprintf(W[i].s, sizeof W[i].s, "%s", best[c]);
	snprintf(W[i].o, sizeof W[i].o, "%s", best[c]);
	if (span == 2) {
		memmove(&W[i + 1], &W[i + 2], (size_t)(nw + 2 - i) * sizeof W[0]);
		nw--;
	}
}

/* comutative is commutative, add ivite is additive */
static void
spellfix(void)
{
	const char *best[4];
	char cat[80];
	int i, n;

	for (i = 0; i < nw; i++) {
		if (protected_word(i))
			continue;
		if (i + 1 < nw && W[i + 1].sp && !protected_word(i + 1) &&
		    strlen(W[i].s) >= 3 && strlen(W[i + 1].s) >= 3) {
			snprintf(cat, sizeof cat, "%s%s", W[i].s, W[i + 1].s);
			if ((n = closest(cat, best, 4)) > 0) {
				fix_word(i, 2, best, n);
				continue;
			}
		}
		if ((n = closest(W[i].s, best, 4)) > 0)
			fix_word(i, 1, best, n);
	}
}

/* translate one sentence; 0 when it failed (and said why) */
static int
translate(const char *sent)
{
	need_more = NULL;
	{
		/* an equation or ODE to classify: props.c reads it as written */
		const char *r = props_nlp_raw(sent);
		if (r) {
			sb_clear(&out);
			emit(r);
			return 1;
		}
	}
	if (setjmp(nl_jmp)) {
		dry = 0;
		quiet = 0;
		return 0;
	}
	tokenize(sent);
	if (nw == 0)
		return 0;
	reset_sentence(1);
	spellfix();
	dry = 1;
	if (setjmp(nl_jmp) == 0) {
		sentence();
		parse_tail();
	}
	dry = 0;
	memcpy(P1, V, sizeof V);
	np1 = nv;
	if (setjmp(nl_jmp))
		return 0;
	reset_sentence(2);
	sentence();
	parse_tail();
	if (W[wi].k != W_END)
		nfail("I didn't follow \"%s\"", words_text(wi, nw));
	finish();
	return 1;
}

/*
 * translate, and when the sentence leaves a question open (an interval
 * for approx), ask it and translate again with the reply appended: the
 * reply is read like the end of the sentence ("cheby poly nocrit").
 */
static int
translate_asking(const char *sent)
{
	char *cur = strdup(sent), reply[512], *p;
	int ok, tries;

	{
		/* conversational openers mean "what is" */
		static const char *const open[] = {"what do you make of ",
		    "what would you make of ", "what do you think of ",
		    "what do you think about ", "what would you say about ",
		    "what about ", "how about ", "tell me about ", NULL};
		int k;
		while (*cur == ' ')
			memmove(cur, cur + 1, strlen(cur));
		for (k = 0; open[k]; k++) {
			size_t n = strlen(open[k]);
			if (!strncasecmp(cur, open[k], n)) {
				char *t = xmalloc(strlen(cur) - n + 9);
				sprintf(t, "what is %s", cur + n);
				free(cur);
				cur = t;
				break;
			}
		}
	}

	for (tries = 0;; tries++) {
		no_more_prompts = tries >= 1;
		spell_quiet = tries > 0;
		ok = translate(cur);
		if (nlp_asked) {
			ok = 0;
			break;
		}
		if (ok || need_more == NULL)
			break;
		if (nlp_defer) {
			const char *a = next_answer();
			if (a == NULL) {
				printf("\x1e" "ask text\n? %s\n", need_more);
				nlp_asked = 1;
				ok = 0;
				break;
			}
			snprintf(reply, sizeof reply, "%s", a);
		} else {
			printf("  ? %s\n    > ", need_more);
			fflush(stdout);
			if (fgets(reply, sizeof reply, stdin) == NULL)
				reply[0] = 0;
		}
		reply[strcspn(reply, "\r\n")] = 0;
		for (p = reply; isspace((unsigned char)*p); p++)
			;
		if (*p == 0)
			no_more_prompts = 1;
		p = xmalloc(strlen(cur) + strlen(reply) + 4);
		sprintf(p, "%s , %s", cur, reply);
		free(cur);
		cur = p;
		if (!reply[0]) {
			spell_quiet = 1;
			ok = translate(cur);
			break;
		}
	}
	no_more_prompts = 0;
	spell_quiet = 0;
	free(cur);
	return ok;
}

/*
 * Translate one request and hand back the ucas statements instead of
 * running them (animate exec of ...).  Questions are asked as usual;
 * returns 0 when the request did not translate or a question is out.
 */
int
nlp_translate(const char *req, struct sbuf *dst)
{
	int ok;

	run_source("", "<english>", 0);   /* the prelude */
	ok = translate_asking(req);
	if (!ok || nlp_asked || out.n == 0)
		return 0;
	sb_clear(dst);
	sb_puts(dst, out.p);
	return 1;
}

void
nlp_run(const char *line, size_t n, const char *file, int echo)
{
	char *buf = xmalloc(n + 1), *p, *q, *start;
	size_t i, j;

	/* curly quotes to plain ones */
	for (i = j = 0; i < n; i++) {
		if ((unsigned char)line[i] == 0xe2 && i + 2 < n &&
		    (unsigned char)line[i + 1] == 0x80 &&
		    ((unsigned char)line[i + 2] == 0x98 ||
		    (unsigned char)line[i + 2] == 0x99)) {
			buf[j++] = '\'';
			i += 2;
		} else if ((unsigned char)line[i] == 0xe2 && i + 2 < n &&
		    (unsigned char)line[i + 1] == 0x80 &&
		    ((unsigned char)line[i + 2] == 0x9c ||
		    (unsigned char)line[i + 2] == 0x9d)) {
			buf[j++] = '"';
			i += 2;
		} else
			buf[j++] = line[i];
	}
	buf[j] = 0;
	for (p = buf; isspace((unsigned char)*p); p++)
		;
	if (*p == '?')
		p++;
	q = p + strlen(p);
	while (q > p && isspace((unsigned char)q[-1]))
		*--q = 0;
	if (echo)
		printf("> %s\n", p);
	run_source("", file, 0);   /* the prelude, before looking up names */
	/* one sentence at a time: split at ; and at . ? ! before a capital */
	start = p;
	for (q = p;; q++) {
		int end = *q == 0 || *q == ';';
		if ((*q == '.' || *q == '?' || *q == '!') && q[1] != '=' &&
		    !(q > p && isdigit((unsigned char)q[-1]) &&
		    isdigit((unsigned char)q[1])) && q[1] != '.' &&
		    !(q > p && q[-1] == '.')) {
			char *r = q + 1;
			while (isspace((unsigned char)*r))
				r++;
			if (*r == 0 || isupper((unsigned char)*r) || *q == '?')
				end = 1;
		}
		if (end) {
			char c = *q;
			*q = 0;
			while (*start == '"')
				start++;
			if (strspn(start, " \t\"") < strlen(start) &&
			    translate_asking(start) && out.n) {
				char *l = out.p, *nl;
				while (*l) {
					nl = strchr(l, '\n');
					printf("  ucas: %.*s\n", (int)(nl - l), l);
					l = nl + 1;
				}
				run_source(out.p, "<english>", 0);
				if (fn_used && user_fn(fn_used))
					snprintf(last_fn, sizeof last_fn, "%s", fn_used);
				last_nfuncs = nfuncs;
			}
			fflush(stdout);
			if (c == 0 || nlp_asked)
				break;
			start = q + 1;
		}
	}
	free(buf);
}

/* ---- telling English from ucas ---- */

static const char *const ucas_cmds[] = {"show", "table", "range", "domain",
	"wrap", "crit", "diff", "equiv", "gates", "find", "fit", "error",
	"cost", "use", "more", "parse", "funcs", "models", "help", "basis",
	"model", "approx", "closed", "scheme", "continuous", "integrate",
	"dsolve", "pde", "animate",
	/* props.c */
	"monotonic", "monotone", "period", "periodic", "identify", "parity",
	"symmetry", "bounded", "injective", "surjective", "bijective",
	"continuity", "discontinuities", "convexity", "convex", "concave",
	"asymptotes", "zeros", "roots", NULL};
static const char *const cmd_followers[] = {"of", "the", "a", "an", "me",
	"us", "all", "for", "out", "whether", "if", "how", "when", "where",
	"what", "some", "any", "its", "it", "my", "this", "that", "every",
	"each", "possible", "both", "another", "other", "cheaper", "faster",
	"better", "alternative", "alternatives", "implementations", "ways",
	"minimal", "smallest", "english", "form", "forms", NULL};
static const char *const forms_cmds[] = {"complex", "rect", "polar",
	"conj", "re", "im", "modulus", "arg", "nthroots", "unity", "trig", NULL};
static const char *const forms_followers[] = {"conjugate", "rewrite",
	"rewrites", "identity", "identities", "representation", "notation",
	"coordinates", "part", "parts", "roots", "root", "number", "numbers",
	"and", "simplification", "functions", "of", "the", "for", "form",
	"forms", "me", NULL};
static const char *const starters[] = {"what", "whats", "how", "is", "are",
	"does", "do", "can", "could", "would", "will", "which", "why", "who",
	"compute", "calculate", "evaluate", "simplify", "solve", "let",
	"define", "suppose", "assume", "give", "tell", "please", "pls", "plz",
	"i", "we",
	"differentiate", "derive", "check", "list", "explain", "write",
	"express", "determine", "prove", "plot", "graph", "optimize",
	"optimise", "estimate", "convert", "tabulate", "enumerate", "compare",
	"synthesize", "synthesise", "factor", "forget", "hey",
	"hi", "ok", "okay", "so", "now", "also", "the", "whether", "get",
	"work", "figure", "tune", "calibrate", "speed", "improve", "name",
	"call", "declare", "thanks", "verify", "alt", "alts", "alternatives",
	"alternative", "approximate", "expand", "rewrite", "linearize",
	"linearise", "classify", "characterize", "characterise", "describe",
	"draw", "sketch", "visualize", "visualise", "render", "depict",
	"illustrate", "chart", "shade", "rotate", "spin", NULL};
static const char *const vocab[] = {"principal", "branch", "lambert",
	"productlog", "lower", "upper", "plus", "minus", "times", "multiplied",
	"divided", "squared", "cubed", "square", "root", "cube", "sum",
	"product", "difference", "quotient", "ratio", "average", "half",
	"twice", "of", "the", "is", "are", "equal", "equals", "greater",
	"less", "than", "between", "power", "raised", "modulo", "remainder",
	"absolute", "value", "sine", "cosine", "tangent", "logarithm",
	"imaginary", "polar", "rectangular", "cartesian", "modulus",
	"conjugate", "exponential", "unity", "roots", "identities",
	"trigonometric", "magnitude",
	"natural", "negative", "positive", "bitwise", "shifted", "exclusive",
	"number", "count", "leading", "trailing", "zeros", "ones", "derivative",
	"maximum", "minimum", "unsigned", "signed", "byte", "bytes", "integer",
	"integers", "bit", "from", "to", "with", "respect", "over", "one",
	"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
	"hundred", "thousand", "overflow", "range", "from", "using", "with",
	"cheby", "cheb", "chebyshev", "poly", "polynomial", "minimax", "remez",
	"crit", "critical", "nocrit", "points", "radians", "degrees", "deg",
	"approximation", "commutative", "additive", "associative",
	"idempotent", "plot", "graph", "neighbourhood", "neighborhood",
	"heatmap", "contour", "surface", NULL};

/*
 * Names the text being run defines (f(x) = ..., x[0] = ..., y' = ...,
 * pde heat {): the lines are run in blocks, so a name can be used before
 * ucas has seen its definition; "range sum" is then not English.
 */
static char pending[64][32];
static int npending;

static int
pending_name(const char *w)
{
	int k;

	for (k = 0; k < npending; k++)
		if (!strcmp(pending[k], w))
			return 1;
	return 0;
}

static void
scan_definitions(const char *src)
{
	const char *p = src, *q;
	size_t n;

	for (; *p; p = *q ? q + 1 : q) {
		q = strchr(p, '\n');
		if (q == NULL)
			q = p + strlen(p);
		if (!strncmp(p, "pde ", 4))
			p += 4;
		if (!isalpha((unsigned char)*p) && *p != '_')
			continue;
		for (n = 0; p + n < q && (isalnum((unsigned char)p[n]) ||
		    p[n] == '_'); n++)
			;
		if (n >= 32 || npending == 64)
			continue;
		/* a definition head: name, then ( [ ' = or { */
		{
			const char *r = p + n;
			while (r < q && *r == ' ')
				r++;
			if (r < q && (*r == '(' || *r == '[' || *r == '\'' ||
			    *r == '{' || (*r == '=' && r[1] != '=')) &&
			    memchr(p, '=', (size_t)(q - p)) != NULL) {
				memcpy(pending[npending], p, n);
				pending[npending][n] = 0;
				if (!pending_name(pending[npending]))
					npending++;
			}
		}
	}
}

/* words that are also options of ucas commands: integrate f from 0 to 1 */
static const char *const option_words[] = {"from", "to", "with", "using",
	"wrt", "steps", "degree", "method", "keep", "crit", "nocrit", "range",
	"deg",
	/* approx's options: ucas, never English */
	"error", "ulp", "relative", "absolute", "odd", "even", "nosym",
	"endpoints", "ends", "weight", "fma", "nofma", "norefine",
	"norational", "noreduce", "nosplit", "list", "points", "name",
	"compare", "free", "more",
	/* gates, find, cost */
	"basis", "max", "nodes", "model", "metric", NULL};

static int word_at(const char *s, size_t n, size_t *i, char *w, size_t wn);

/*
 * English words in s that ucas has no use for; a word followed by '(' is
 * a call, sum(i, 1, n, i^2).  strong: leave out ucas option words.
 */
static int
english_words(const char *s, size_t n, int strong)
{
	char w[40];
	size_t j = 0, k;
	int count = 0;

	while (j < n) {
		if (s[j] == '#')
			break;
		if (word_at(s, n, &j, w, sizeof w)) {
			for (k = j; k < n && s[k] == ' '; k++)
				;
			if ((k < n && s[k] == '(') || user_fn(w) || pending_name(w) ||
			    (strong && in_list(option_words, w)))
				continue;
			if (!in_list(vocab, w)) {
				/* a misspelt English word ("princpal") counts too,
				 * but not one of ucas's own option words */
				int v, lim = in_list(option_words, w) ? 0 :
				    spell_limit(strlen(w));
				for (v = 0; lim > 0 && vocab[v]; v++)
					if (osa(w, vocab[v], lim) <= lim)
						break;
				if (lim == 0 || vocab[v] == NULL)
					continue;
			}
			count++;
			continue;
		}
		j++;
	}
	return count;
}

static int
word_at(const char *s, size_t n, size_t *i, char *w, size_t wn)
{
	size_t k = 0;

	while (*i < n && (s[*i] == ' ' || s[*i] == '\t'))
		(*i)++;
	if (*i >= n || !(isalpha((unsigned char)s[*i]) || s[*i] == '_'))
		return 0;
	while (*i < n && (isalnum((unsigned char)s[*i]) || s[*i] == '_' ||
	    s[*i] == '\'' || s[*i] == '-')) {
		if (k + 1 < wn)
			w[k++] = tolower((unsigned char)s[*i]);
		(*i)++;
	}
	w[k] = 0;
	return 1;
}

int
nlp_is_english(const char *s, size_t n)
{
	char w1[40], w2[40], w[40];
	size_t i = 0, j;
	int count = 0, words = 0;

	while (n > 0 && isspace((unsigned char)s[n - 1]))
		n--;
	while (i < n && isspace((unsigned char)s[i]))
		i++;
	if (i >= n || s[i] == '#' || (s[i] == '/' && s[i + 1] == '/'))
		return 0;
	if (s[i] == '?')
		return 1;
	if (s[n - 1] == '?')
		return 1;
	/* calc2: d/dx F, d²/dx² F, d^2/dx^2 F, dy/dx, ∂f/∂x, ∇f */
	if (s[i] == 'd' && i + 4 < n && isalpha((unsigned char)s[i + 1]) &&
	    s[i + 2] == '/' && s[i + 3] == 'd' && isalpha((unsigned char)s[i + 4]))
		return 1;
	if ((s[i] == 'd' && i + 2 < n && (s[i + 1] == '/' || s[i + 1] == '^' ||
	    !strncmp(s + i + 1, "\xc2\xb2", 2) || (isdigit((unsigned char)s[i + 1])
	    && s[i + 2] == '/'))) || !strncmp(s + i, "\xe2\x88\x82", 3) ||
	    !strncmp(s + i, "\xe2\x88\x87", 3))
		return 1;
	for (j = i; j < n; j++) {
		if (s[j] == '#')
			break;
		/* a contraction (what's), not a derivative (y' = ...) */
		if (j > i && isalpha((unsigned char)s[j - 1]) &&
		    ((s[j] == '\'' && j + 1 < n &&
		    isalpha((unsigned char)s[j + 1])) ||
		    ((unsigned char)s[j] == 0xe2 && j + 3 < n &&
		    (unsigned char)s[j + 1] == 0x80 &&
		    isalpha((unsigned char)s[j + 3]))))
			return 1;
	}
	j = i;
	w1[0] = 0;
	if (word_at(s, n, &j, w1, sizeof w1)) {
		size_t k = j;
		int depth = 0;
		while (k < n && s[k] == ' ')
			k++;
		/* a ucas definition, f(x) = ..., x[0] = ... or g = ..., stays ucas */
		if (k < n && (s[k] == '(' || s[k] == '[')) {
			char open = s[k], close = open == '(' ? ')' : ']';
			for (; k < n; k++) {
				if (s[k] == open)
					depth++;
				else if (s[k] == close && --depth == 0)
					break;
			}
			for (k++; k < n && s[k] == ' '; k++)
				;
			if (k < n && s[k] == '=' && (k + 1 >= n || s[k + 1] != '='))
				return 0;
		}
		if (k < n && s[k] == '=' && (k + 1 >= n || s[k + 1] != '=') &&
		    !in_list(starters, w1))
			return 0;
		/* animate reads its own wordy options ("n=8 at 30 fps") */
		if (!strcmp(w1, "animate"))
			return 0;
		/* forms.c: "polar 1 + i" is ucas, "polar form of 1 + i" English */
		if (in_list(forms_cmds, w1)) {
			size_t j2 = j;
			return word_at(s, n, &j2, w2, sizeof w2) &&
			    in_list(forms_followers, w2);
		}
		if ((in_list(c2_cmds, w1) || !strcmp(w1, "range")) &&
		    word_at(s, n, &(size_t){j}, w2,
		    sizeof w2) && (!strcmp(w2, "of") || !strcmp(w2, "matrix") ||
		    !strcmp(w2, "series") || !strcmp(w2, "derivative") ||
		    !strcmp(w2, "as") || !strcmp(w2, "expansion")))
			return 1;   /* calc2: "limit of", "taylor series of" */
		/*
		 * plot's raw options are words too ("plot sin x tangent 2"):
		 * English after "plot of", or with an English word that is not
		 * one of them ("plot x squared", "plot it with its derivative")
		 */
		if ((!strcmp(w1, "plot") || !strcmp(w1, "plot3d") ||
		    !strcmp(w1, "neighbourhood") || !strcmp(w1, "neighborhood")) &&
		    !user_fn(w1)) {
			static const char *const raw[] = {"tangent", "derivative",
				"derivatives", "over", "from", "to", "with", "points",
				"at", "near", "around", "contour", "surface", "field",
				"polar", NULL};
			size_t j2 = j;
			if (word_at(s, n, &j2, w2, sizeof w2) &&
			    (in_list(cmd_followers, w2) || !strcmp(w2, "me")))
				return 1;
			for (j2 = j; j2 < n;) {
				if (s[j2] == '#')
					break;
				if (word_at(s, n, &j2, w, sizeof w)) {
					if (!in_list(raw, w) && in_list(vocab, w) &&
					    !user_fn(w))
						return 1;
					{
						static const char *const en[] = {"its", "it",
							"as", "the", "and", "interesting", "on", "a",
							"an", "of", "into", "using", "scale", "axis",
							"axes", "coordinates", "neighbourhood",
							"neighborhood", "me", "show", "part", "save",
							"called", "named", "between", "together",
							"along", "their", "logarithmic", "multiples", "dimensions", "three", "time",
							"look", "like", "is", "where", "per", "lines",
							"map", "plot", "graph", "circle", "curve",
							"function", "then", "also", NULL};
						if (in_list(en, w))
							return 1;
					}
					continue;
				}
				j2++;
			}
			return 0;
		}
		/* viz.c: "area x^2 from 0 to 4" is ucas, "area under x^2" not */
		if (in_list(viz_cmds, w1) && !user_fn(w1))
			return viz_english(s + j, n - j);
		if (in_list(ucas_cmds, w1) || in_list(c2_cmds, w1)) {
			size_t j2 = j, j3;
			/* "find 2 cheaper ways": look past a count */
			for (j3 = j2; j3 < n && (s[j3] == ' ' ||
			    isdigit((unsigned char)s[j3])); j3++)
				;
			if (j3 > j2 && j3 < n && isalpha((unsigned char)s[j3]) &&
			    isdigit((unsigned char)s[j3 - 1]) == 0)
				j2 = j3;
			if (!word_at(s, n, &j2, w2, sizeof w2))
				return 0;
			/* "integrate x squared" is English, "integrate x^2" not */
			if (!in_list(cmd_followers, w2))
				return english_words(s + j, n - j, 1) >= 1;
			/*
			 * "parse a < b", "dsolve a to 4" are ucas; "show a
			 * table of ..." is not
			 */
			if (strlen(w2) <= 2)
				return english_words(s + j2, n - j2, 1) >= 1;
			return 1;
		} else if (!user_fn(w1) && in_list(starters, w1)) {
			return 1;
		} else if (!strcmp(w1, "where")) {
			size_t j2 = j;
			return word_at(s, n, &j2, w2, sizeof w2) &&
			    (!strcmp(w2, "is") || !strcmp(w2, "are") ||
			    !strcmp(w2, "does") || !strcmp(w2, "do"));
		}
	}
	/* otherwise: count English words that ucas has no use for */
	count = english_words(s + i, n - i, 0);
	for (j = i; j < n;) {
		if (s[j] == '#')
			break;
		if (word_at(s, n, &j, w, sizeof w)) {
			words++;
			continue;
		}
		j++;
	}
	return count >= 2 || (count == 1 && words >= 2);
}

/*
 * Run a file or a line: ucas statements go to run_source in blocks, and a
 * line that reads as English goes through the translator.  Blocks keep
 * their line numbers by starting with empty lines.
 */
/*
 * A fenced code block (```lang ... ```) after a statement is that
 * statement's input: "animate exec" followed by SPIR-V.  The block is
 * taken out of the source (its lines left empty, so line numbers stay)
 * and kept here for the command that wants it.
 */
const char *code_block;
char code_block_lang[24];
int code_block_used;
static struct sbuf code_buf;

/*
 * Split off the text up to the first fenced block: *piece gets that text
 * (the statement the block belongs to), the block goes to code_block, and
 * the return value points after the closing fence (NULL: no block).  Lines
 * taken out are kept as empty lines so later line numbers stay right.
 */
static const char *
next_code_block(const char *src, struct sbuf *piece)
{
	const char *p = src, *e, *q;
	int in = 0;

	sb_clear(piece);
	sb_puts(piece, "");
	sb_clear(&code_buf);
	code_block = NULL;
	code_block_lang[0] = 0;
	code_block_used = 0;
	while (*p) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		for (q = p; q < e && (*q == ' ' || *q == '\t'); q++)
			;
		if (!in && e - q >= 3 && !strncmp(q, "```", 3)) {
			size_t n = 0;
			q += 3;
			while (q < e && n + 1 < sizeof code_block_lang &&
			    !isspace((unsigned char)*q))
				code_block_lang[n++] = (char)tolower((unsigned char)*q++);
			code_block_lang[n] = 0;
			in = 1;
			sb_putc(piece, '\n');
		} else if (in) {
			sb_putc(piece, '\n');
			if (e - q >= 3 && !strncmp(q, "```", 3)) {
				code_block = code_buf.p ? code_buf.p : "";
				return *e ? e + 1 : e;
			}
			sb_printf(&code_buf, "%.*s\n", (int)(e - p), p);
		} else
			sb_printf(piece, "%.*s\n", (int)(e - p), p);
		p = *e ? e + 1 : e;
	}
	if (in) {   /* no closing fence: the block runs to the end */
		code_block = code_buf.p ? code_buf.p : "";
		return p;
	}
	return NULL;
}

static void run_text_plain(const char *src, const char *file, int echo);

/* the block looks like GLSL source (and is tagged so, or untagged) */
int
glsl_like(const char *c, const char *lang)
{
	static const char *const tags[] = {"", "glsl", "comp", "frag", "vert",
	    "c", "cpp", "c++", "hlsl", "shader", "geom", "tesc", "tese", NULL};
	int k, tagged = 0;

	if (c == NULL)
		return 0;
	for (k = 0; tags[k]; k++)
		if (!strcmp(lang ? lang : "", tags[k]))
			tagged = 1;
	if (!tagged || strstr(c, "= Op") || strstr(c, "OpFunction"))
		return 0;
	return strchr(c, '{') && strchr(c, ';') && (strstr(c, "uint") ||
	    strstr(c, "vec") || strstr(c, "float") || strstr(c, "int ") ||
	    strstr(c, "void ") || strstr(c, "#version"));
}

/* the block looks like SPIR-V assembly */
static int
spv_like(const char *c)
{
	return c != NULL && (strstr(c, "= Op") != NULL ||
	    strstr(c, "OpFunction") != NULL || strstr(c, "OpCapability") != NULL ||
	    strstr(c, "OpStore") != NULL);
}

/* an integer literal: decimal or 0x hex, with a sign */
static int
int_lit(const char **pp, int64_t *v)
{
	const char *p = *pp;
	int neg = 0;
	char *e;

	while (*p == ' ' || *p == '\t')
		p++;
	if (*p == '-' || *p == '+')
		neg = *p++ == '-';
	if (!isdigit((unsigned char)*p))
		return 0;
	*v = (int64_t)strtoull(p, &e, 0);
	if (neg)
		*v = -*v;
	*pp = e;
	return 1;
}

/*
 * Assumptions in the request, each as " assume NAME LO HI" (inclusive):
 * "n in [0, 32)", "n in (a, b]", "n in 0..31", "0 <= n < 32", "n < 32"
 * (0 <= n), "n <= 31".
 */
static void
find_assumptions(const char *t, struct sbuf *opt)
{
	const char *p = t;

	while (*p) {
		const char *q, *ns;
		char name[40];
		size_t nl;
		int64_t lo, hi;
		if (!(isalpha((unsigned char)*p) || *p == '_') ||
		    (p > t && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))) {
			p++;
			continue;
		}
		ns = p;
		while (isalnum((unsigned char)*p) || *p == '_')
			p++;
		nl = (size_t)(p - ns);
		if (nl >= sizeof name)
			continue;
		memcpy(name, ns, nl);
		name[nl] = 0;
		q = p;
		while (*q == ' ')
			q++;
		/* NAME in [lo, hi) / (lo, hi] / lo..hi */
		if (!strncmp(q, "in", 2) && !isalnum((unsigned char)q[2])) {
			char open, close;
			q += 2;
			while (*q == ' ')
				q++;
			open = *q;
			if (open == '[' || open == '(') {
				q++;
				if (!int_lit(&q, &lo))
					continue;
				while (*q == ' ')
					q++;
				if (*q != ',' && strncmp(q, "..", 2) != 0)
					continue;
				q += *q == ',' ? 1 : 2;
				if (!int_lit(&q, &hi))
					continue;
				while (*q == ' ')
					q++;
				close = *q;
				if (close != ']' && close != ')')
					continue;
				if (open == '(')
					lo++;
				if (close == ')')
					hi--;
			} else {
				if (!int_lit(&q, &lo))
					continue;
				if (strncmp(q, "..", 2) != 0)
					continue;
				q += 2;
				if (!int_lit(&q, &hi))
					continue;
			}
			if (lo <= hi)
				sb_printf(opt, " assume %s %lld %lld", name,
				    (long long)lo, (long long)hi);
			p = q;
			continue;
		}
		/* NAME < hi / NAME <= hi, and lo <= NAME before it */
		if (*q == '<') {
			int le = q[1] == '=';
			const char *b = ns;
			q += le ? 2 : 1;
			if (!int_lit(&q, &hi))
				continue;
			if (!le)
				hi--;
			lo = 0;
			while (b > t && b[-1] == ' ')
				b--;
			if (b - 2 >= t && b[-1] == '=' && b[-2] == '<') {
				const char *e = b - 2, *st;
				int64_t v;
				while (e > t && e[-1] == ' ')
					e--;
				st = e;
				while (st > t && (isxdigit((unsigned char)st[-1]) ||
				    st[-1] == 'x' || st[-1] == '-'))
					st--;
				if (int_lit(&st, &v))
					lo = v;
			} else if (b - 1 >= t && b[-1] == '<') {
				const char *e = b - 1, *st;
				int64_t v;
				while (e > t && e[-1] == ' ')
					e--;
				st = e;
				while (st > t && (isxdigit((unsigned char)st[-1]) ||
				    st[-1] == 'x' || st[-1] == '-'))
					st--;
				if (int_lit(&st, &v))
					lo = v + 1;
			}
			if (lo <= hi)
				sb_printf(opt, " assume %s %lld %lld", name,
				    (long long)lo, (long long)hi);
			p = q;
		}
	}
}

/*
 * "SAT simplify", "optimize this shader", "peephole it with smt" next to
 * a SPIR-V block: the text lines become one `spvsimplify` statement
 * (directive lines and the line count stay), options carried over
 */
static int
simplify_piece(struct sbuf *piece)
{
	static const char *const w[] = {"simplify", "simplified",
	    "simplification", "simplifications", "simp", "optimize", "optimise",
	    "optimized", "optimised", "optimization", "optimisation",
	    "superoptimize", "superoptimise", "superoptimization", "peephole",
	    "sat", "smt", "minimize", "minimise", "shorten", "cheaper", "souper",
	    NULL};
	char *p = piece->p, budget[32] = "", lw[64] = "", *assume_opts = NULL;
	int hit = 0, anim = 0, nofile = 0, k, first = 1;
	struct sbuf out = {0};

	if (p == NULL)
		return 0;
	/* the words */
	{
		const char *q = p;
		while (*q) {
			size_t n = 0;
			if (*q == '\x1e') {	/* directives */
				while (*q && *q != '\n')
					q++;
				continue;
			}
			if (!isalpha((unsigned char)*q)) {
				if (isdigit((unsigned char)*q) && !strcmp(lw, "budget")) {
					size_t b = 0;
					while ((isdigit((unsigned char)*q) || *q == '.') &&
					    b + 1 < sizeof budget)
						budget[b++] = *q++;
					budget[b] = 0;
					continue;
				}
				q++;
				continue;
			}
			while (isalpha((unsigned char)*q) && n + 1 < sizeof lw)
				lw[n++] = (char)tolower((unsigned char)*q++);
			lw[n] = 0;
			while (isalpha((unsigned char)*q))
				q++;
			for (k = 0; w[k]; k++)
				if (!strcmp(lw, w[k]))
					hit = 1;
			if (!strcmp(lw, "animate") || !strcmp(lw, "exec") ||
			    !strcmp(lw, "run"))
				anim = 1;
			if (!strcmp(lw, "nofile"))
				nofile = 1;
		}
	}
	if (!hit || anim)
		return 0;
	/* rebuild: directives kept, the text on one line, blank lines after */
	{
		struct sbuf as = {0};
		const char *q0 = p;
		/* assumptions from the text lines (not directives) */
		while (*q0) {
			const char *e = strchr(q0, '\n');
			size_t ln = e ? (size_t)(e - q0) : strlen(q0);
			if (*q0 != '\x1e' && *q0 != '#') {	/* not comments */
				char *line = xmalloc(ln + 1);
				memcpy(line, q0, ln);
				line[ln] = 0;
				find_assumptions(line, &as);
				free(line);
			}
			q0 += ln + (e != NULL);
		}
		assume_opts = as.p;
	}
	{
		const char *q = p, *e;
		while (*q) {
			e = strchr(q, '\n');
			if (e == NULL)
				e = q + strlen(q);
			if (*q == '\x1e')
				sb_printf(&out, "%.*s\n", (int)(e - q), q);
			else if (first && e > q) {
				sb_printf(&out, "spvsimplify%s%s%s%s\n", budget[0] ?
				    " budget " : "", budget, nofile ? " nofile" : "",
				    assume_opts ? assume_opts : "");
				first = 0;
			} else
				sb_putc(&out, '\n');
			q = *e ? e + 1 : e;
		}
	}
	free(assume_opts);
	free(piece->p);
	*piece = out;
	return 1;
}

/* statements, each fenced block going to the statement before it */
void
run_text(const char *src, const char *file, int echo)
{
	struct sbuf piece = {0}, pad = {0};
	const char *rest = src, *next;
	int lines = 0;

	/* one run: the questions' answers are queued once, for all pieces */
	nlp_asked = 0;
	while (nans > 0)
		free(answers[--nans]);
	ansi = 0;
	for (;;) {
		const char *c;
		next = next_code_block(rest, &piece);
		if (code_block && glsl_like(code_block, code_block_lang)) {
			/* GLSL: compiled to SPIR-V (the command line; the bot
			 * converts before it sends) */
			static struct sbuf spv;
			struct sbuf notes = {0};
			sb_clear(&spv);
			if (glsl_to_spvasm(code_block, &spv, &notes)) {
				printf("  note: the ```%s block was compiled to SPIR-V with "
				    "glslangValidator\n%s", code_block_lang[0] ?
				    code_block_lang : "glsl", notes.p ? notes.p : "");
				code_block = spv.p;
				snprintf(code_block_lang, sizeof code_block_lang, "spv");
			} else
				printf("  note: the GLSL block did not compile:\n%s",
				    notes.p ? notes.p : "");
			free(notes.p);
		}
		if (code_block && spv_like(code_block))
			simplify_piece(&piece);
		/* earlier pieces' lines come first, as empty lines */
		sb_clear(&pad);
		sb_puts(&pad, "");
		{
			int i;
			for (i = 0; i < lines; i++)
				sb_putc(&pad, '\n');
		}
		sb_puts(&pad, piece.p ? piece.p : "");
		run_text_plain(pad.p, file, echo);
		if (code_block && !code_block_used && !nlp_asked)
			printf("  note: the ```%s code block was not used (animate "
			    "exec and simplify take SPIR-V)\n", code_block_lang);
		code_block = NULL;
		if (next == NULL || *next == 0 || nlp_asked)
			break;
		for (c = rest; c < next; c++)
			if (*c == '\n')
				lines++;
		rest = next;
	}
	free(piece.p);
	free(pad.p);
}

static void
run_text_plain(const char *src, const char *file, int echo)
{
	struct sbuf seg = {0};
	const char *p = src, *e, *c;
	int depth = 0, lineno = 1, have = 0, k;
	size_t len;

	npending = 0;
	scan_definitions(src);
	while (*p) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		len = (size_t)(e - p);
		/* front-end directives (see nlp_defer); not counted as lines */
		if (len > 0 && p[0] == '\x1e') {
			if (len >= 6 && !strncmp(p + 1, "defer", 5))
				nlp_defer = 1;
			else if (len >= 7 && !strncmp(p + 1, "answer", 6) &&
			    nans < MAXANS) {
				const char *a = p + 7;
				if (a < e && *a == ' ')
					a++;
				answers[nans] = xmalloc((size_t)(e - a) + 1);
				memcpy(answers[nans], a, (size_t)(e - a));
				answers[nans++][e - a] = 0;
			}
			p = *e ? e + 1 : e;
			continue;
		}
		if (depth <= 0 && len > 0 && !isspace((unsigned char)p[0]) &&
		    nlp_is_english(p, len)) {
			if (have) {
				run_source(seg.p, file, echo);
				have = 0;
			}
			nlp_run(p, len, file, echo);
			if (nlp_asked)
				break;   /* the rest waits for the answer */
		} else {
			if (!have) {
				sb_clear(&seg);
				for (k = 1; k < lineno; k++)
					sb_putc(&seg, '\n');
				have = 1;
			}
			for (c = p; c < e; c++) {
				if (*c == '#')
					break;
				if (*c == '{' || *c == '(' || *c == '[')
					depth++;
				if (*c == '}' || *c == ')' || *c == ']')
					depth--;
			}
			sb_printf(&seg, "%.*s\n", (int)len, p);
		}
		lineno++;
		p = *e ? e + 1 : e;
	}
	if (have)
		run_source(seg.p, file, echo);
	free(seg.p);
}
