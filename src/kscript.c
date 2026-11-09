/*
 * kscript.c - theories and structured proofs.
 *
 * A theory is text: declarations (sort, func, pred), definitions and
 * arithmetic axioms (the only places axioms come from), and lemmas with
 * Isar-like proofs.  A proof is a block of steps on a goal:
 *
 *   fix x y            introduce universally quantified variables
 *   intro h1 h2        assume the antecedent (its conjuncts named)
 *   have h: F by R     a fact, proved by kauto from the context and R
 *   obtain x:s st h: F by R    an existential, opened
 *   show F by R        the goal (F up to conversion; "show by R")
 *   contra h           assume the negation of the goal
 *   cases h: A or B by R / case h1 proof .. qed / case h2 proof .. qed
 *   induct n from B / base proof .. qed / step proof .. qed
 *
 * R lists facts and lemmas, an instance L[t1, t2], and "with t1, t2"
 * (witnesses for existentials).  Any "by R" may be a nested
 * "proof ... qed" instead.  Every step ends in kernel rules; the script
 * only steers them.
 */
#include "ucas.h"
#include "prove.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

struct klemma *klemmas;
uint32_t nklemmas;
static uint32_t capl;

int
kl_find(const char *name)
{
	uint32_t id = sym_cstr(name), i;

	for (i = 0; i < nklemmas; i++)
		if (klemmas[i].name == id)
			return (int)i;
	return -1;
}

static int
kl_add(const char *name, kt phi, const char *src, int lib)
{
	int i = kl_find(name);

	if (i >= 0) {
		klemmas[i].phi = phi;
		klemmas[i].th = KNONE;
		klemmas[i].src = src;
		return i;
	}
	if (nklemmas == capl) {
		capl = capl ? capl * 2 : 64;
		klemmas = xrealloc(klemmas, capl * sizeof *klemmas);
	}
	memset(&klemmas[nklemmas], 0, sizeof klemmas[0]);
	klemmas[nklemmas].name = sym_cstr(name);
	klemmas[nklemmas].phi = phi;
	klemmas[nklemmas].src = src;
	klemmas[nklemmas].lib = lib;
	return (int)nklemmas++;
}

/* ---- logical lines ---- */

struct line {
	char *text;
	int no;
};

struct src {
	struct line *l;
	int n, i;
	const char *file;
	char *msg;
	int msgn;
	int failed;
};

static int
ends_open(const char *s)
{
	static const char *const tail[] = {"->", "<->", "and", "or", ",",
		":", "(", "=", "+", "-", "*", "iff", "not", "[", NULL};
	size_t n = strlen(s), k;
	int i;

	while (n && isspace((unsigned char)s[n - 1]))
		n--;
	for (i = 0; tail[i]; i++) {
		k = strlen(tail[i]);
		if (n >= k && !strncmp(s + n - k, tail[i], k) &&
		    (k == 1 || !isalpha((unsigned char)tail[i][0]) || n == k ||
		    !isalnum((unsigned char)s[n - k - 1])))
			return 1;
	}
	return 0;
}

static int
depth_of(const char *s)
{
	int d = 0;

	for (; *s && *s != '#'; s++)
		d += (*s == '(' || *s == '[') - (*s == ')' || *s == ']');
	return d;
}

static void
split_lines(struct src *S, const char *text)
{
	const char *p = text;
	int no = 0, cap = 0;
	struct sbuf cur = {0};
	int curno = 0, depth = 0, open = 0;

	S->n = 0;
	while (*p) {
		const char *e = strchr(p, '\n');
		size_t len = e ? (size_t)(e - p) : strlen(p);
		char *ln = xmalloc(len + 1), *h;
		no++;
		memcpy(ln, p, len);
		ln[len] = 0;
		if ((h = strchr(ln, '#')) != NULL)
			*h = 0;
		p = e ? e + 1 : p + len;
		{
			char *t = ln;
			while (isspace((unsigned char)*t))
				t++;
			if (!*t) {
				free(ln);
				continue;
			}
			if (cur.n == 0)
				curno = no;
			else
				sb_putc(&cur, ' ');
			sb_puts(&cur, t);
			depth += depth_of(t);
			open = ends_open(t);
		}
		free(ln);
		if (depth <= 0 && !open) {
			if (S->n == cap) {
				cap = cap ? cap * 2 : 256;
				S->l = xrealloc(S->l, (size_t)cap * sizeof *S->l);
			}
			S->l[S->n].text = cur.p;
			S->l[S->n++].no = curno;
			memset(&cur, 0, sizeof cur);
			depth = 0;
		}
	}
	if (cur.n) {
		if (S->n == cap) {
			cap = cap ? cap * 2 : 256;
			S->l = xrealloc(S->l, (size_t)cap * sizeof *S->l);
		}
		S->l[S->n].text = cur.p;
		S->l[S->n++].no = curno;
	}
}

static int
serr(struct src *S, const char *fmt, ...)
{
	va_list ap;
	char m[1024];
	int no = S->i < S->n ? S->l[S->i].no : S->n ? S->l[S->n - 1].no : 0;

	if (S->failed)
		return 0;
	va_start(ap, fmt);
	vsnprintf(m, sizeof m, fmt, ap);
	va_end(ap);
	snprintf(S->msg, (size_t)S->msgn, "%s:%d: %s", S->file, no, m);
	S->failed = 1;
	return 0;
}

/* word at s (an identifier), advancing */
static int
word(const char **s, char *w, size_t n)
{
	const char *p = *s;
	size_t k = 0;

	while (isspace((unsigned char)*p))
		p++;
	if (!isalpha((unsigned char)*p) && *p != '_')
		return 0;
	while ((isalnum((unsigned char)*p) || *p == '_' || *p == '\'') &&
	    k + 1 < n)
		w[k++] = *p++;
	w[k] = 0;
	*s = p;
	return 1;
}

static int
iskw(const char *s, const char *w)
{
	size_t n = strlen(w);

	while (isspace((unsigned char)*s))
		s++;
	return !strncmp(s, w, n) && !isalnum((unsigned char)s[n]) &&
	    s[n] != '_';
}

static const char *
skipws(const char *s)
{
	while (isspace((unsigned char)*s))
		s++;
	return s;
}

/* find the top-level keyword w (not inside parentheses or brackets) */
static const char *
find_kw(const char *s, const char *w)
{
	int d = 0;
	size_t n = strlen(w);
	const char *p;

	for (p = s; *p; p++) {
		if (*p == '(' || *p == '[')
			d++;
		else if (*p == ')' || *p == ']')
			d--;
		else if (d == 0 && !strncmp(p, w, n) &&
		    (p == s || (!isalnum((unsigned char)p[-1]) && p[-1] != '_')) &&
		    !isalnum((unsigned char)p[n]) && p[n] != '_')
			return p;
	}
	return NULL;
}

/* ---- declarations ---- */

static int
decl_sorts(struct src *S, const char *s, uint8_t *out, int max)
{
	char w[32];
	int n = 0, k;

	s = skipws(s);
	if (*s != '(')
		return 0;
	s++;
	for (;;) {
		if (!word(&s, w, sizeof w))
			return serr(S, "expected a sort") - 1;
		if ((k = ks_lookup(w)) < 0)
			return serr(S, "unknown sort %s", w) - 1;
		if (n == max)
			return serr(S, "too many arguments") - 1;
		out[n++] = (uint8_t)k;
		s = skipws(s);
		if (*s == ',') {
			s++;
			continue;
		}
		if (*s == ')')
			break;
		return serr(S, "expected , or )") - 1;
	}
	return n;
}

static int
do_decl(struct src *S, const char *t)
{
	char kw[16], name[64], w[32];
	uint8_t args[8];
	int n = 0, res = KS_PROP, flags = 0, k;
	const char *s = t;

	word(&s, kw, sizeof kw);
	if (!strcmp(kw, "sort")) {
		while (word(&s, w, sizeof w))
			ks_declare(w);
		return 1;
	}
	if (!word(&s, name, sizeof name))
		return serr(S, "expected a name");
	s = skipws(s);
	if (*s == '(') {
		n = decl_sorts(S, s, args, 8);
		if (n < 0)
			return 0;
		s = strchr(s, ')') + 1;
	}
	if (!strcmp(kw, "func")) {
		s = skipws(s);
		if (*s != ':')
			return serr(S, "expected ': sort'");
		s++;
		if (!word(&s, w, sizeof w) || (res = ks_lookup(w)) < 0)
			return serr(S, "expected a sort");
		if (word(&s, w, sizeof w)) {
			if (strcmp(w, "hom"))
				return serr(S, "unexpected %s", w);
			if (n < 1 || args[n - 1] != KS_EL || res != KS_EL)
				return serr(S, "a hom maps el to el");
			flags = KSF_HOM;
		}
		k = ksym_declare(name, KSY_FUNC, n, args, res, flags);
	} else
		k = ksym_declare(name, KSY_PRED, n, args, KS_PROP, 0);
	(void)k;
	return 1;
}

/* ---- proofs ---- */

struct ps {
	struct src *S;
	struct kctx ctx;
	struct kscope sc;
	int steps;
};

static kthm block(struct ps *P, kt goal);

static void
scope_add(struct kscope *sc, const char *name, kt v)
{
	if (sc->n < 64) {
		sc->name[sc->n] = sym_cstr(name);
		sc->var[sc->n++] = v;
	} else {
		memmove(sc->name, sc->name + 1, 63 * sizeof sc->name[0]);
		memmove(sc->var, sc->var + 1, 63 * sizeof sc->var[0]);
		sc->name[63] = sym_cstr(name);
		sc->var[63] = v;
	}
}

static kt
parse_f(struct ps *P, const char *text)
{
	char err[256];
	kt f = kt_parse(text, &P->sc, err, sizeof err);

	if (!f)
		serr(P->S, "%s in: %s", err, text);
	else if (KSORT(f) != KS_PROP) {
		serr(P->S, "not a formula: %s", text);
		f = KNONE;
	}
	return f;
}

/* a fresh variable named as the user wrote, if that is free */
static kt
freshvar(struct ps *P, const char *name, int sort)
{
	kt v = kt_varc(name, sort);
	int i;
	uint32_t h;

	for (i = 0; i < P->sc.n; i++)
		if (P->sc.var[i] == v)
			return kt_fresh(name, sort);
	for (i = 0; i < P->ctx.n; i++) {
		kthm th = P->ctx.fact[i];
		if (kt_occurs(v, kthm_concl(th)))
			return kt_fresh(name, sort);
		for (h = 0; h < kthm_nhyps(th); h++)
			if (kt_occurs(v, kthm_hyp(th, h)))
				return kt_fresh(name, sort);
	}
	if (ksym_lookup(name) >= 0)
		return kt_fresh(name, sort);
	return v;
}

static kthm
fact_named(struct ps *P, const char *name)
{
	uint32_t id = sym_cstr(name);
	int i;

	for (i = P->ctx.n - 1; i >= 0; i--)
		if (P->ctx.name[i] == id)
			return P->ctx.fact[i];
	if ((i = kl_find(name)) >= 0 && klemmas[i].th)
		return klemmas[i].th;
	if (kax_find(name) >= 0)
		return kr_axiom(name);
	return KNONE;
}

/* split "a, b, c" at top-level commas */
static int
split_commas(const char *s, char out[][512], int max)
{
	int n = 0, d = 0;
	size_t k = 0;

	for (; *s; s++) {
		if (*s == '(' || *s == '[')
			d++;
		if (*s == ')' || *s == ']')
			d--;
		if (*s == ',' && d == 0) {
			out[n][k] = 0;
			if (++n == max)
				return n;
			k = 0;
			continue;
		}
		if (k < 511)
			out[n][k++] = *s;
	}
	out[n][k] = 0;
	return n + 1;
}

/* prove f from the context and the references in text */
static kthm
by(struct ps *P, kt f, const char *refs)
{
	struct kctx *c = xmalloc(sizeof *c);
	const char *s = refs, *w;
	char name[64], err[256];
	kthm th;

	*c = P->ctx;
	c->nref = 0;
	c->nwith = 0;
	c->depth = 3;
	if ((w = find_kw(s, "with")) != NULL) {
		static char parts[8][512];
		int n = split_commas(w + 4, parts, 8), i;
		for (i = 0; i < n; i++) {
			kt t = kt_parse_term(parts[i], &P->sc, -1, err, sizeof err);
			if (!t) {
				free(c);
				serr(P->S, "%s in witness %s", err, parts[i]);
				return KNONE;
			}
			c->with[c->nwith++] = t;
		}
	}
	for (;;) {
		s = skipws(s);
		if (w && s >= w)
			break;
		if (*s == ',') {
			s++;
			continue;
		}
		if (iskw(s, "depth")) {
			s += 5;
			c->depth = (int)strtol(s, (char **)&s, 10);
			continue;
		}
		if (!word(&s, name, sizeof name))
			break;
		th = fact_named(P, name);
		if (!th) {
			free(c);
			serr(P->S, "no fact, lemma or axiom named %s", name);
			return KNONE;
		}
		if (*s == '[') {
			static char parts[8][512];
			const char *e = s + 1;
			int d = 1, n, i;
			char inner[1024];
			while (*e && d) {
				d += (*e == '[') - (*e == ']');
				e++;
			}
			snprintf(inner, sizeof inner, "%.*s", (int)(e - s - 2), s + 1);
			s = e;
			n = split_commas(inner, parts, 8);
			for (i = 0; i < n && th; i++) {
				kt c0 = kthm_concl(th), t;
				if (KOP(c0) != KO_ALL) {
					free(c);
					serr(P->S, "%s has fewer quantifiers", name);
					return KNONE;
				}
				t = kt_parse_term(parts[i], &P->sc, (int)KT.aux[c0], err,
				    sizeof err);
				if (!t) {
					free(c);
					serr(P->S, "%s in %s", err, parts[i]);
					return KNONE;
				}
				th = kr_alle(th, t);
				if (!th) {
					free(c);
					serr(P->S, "%s", kerr());
					return KNONE;
				}
			}
		}
		ka_add(c, th, 0);
	}
	th = ka_prove(c, f);
	free(c);
	if (!th)
		serr(P->S, "%s", ka_why);
	return th;
}

static int
cur_is(struct ps *P, const char *w)
{
	return P->S->i < P->S->n && iskw(P->S->l[P->S->i].text, w) &&
	    strlen(skipws(P->S->l[P->S->i].text)) == strlen(w);
}

/* the justification after a step: "by ..." on the line, or a block */
static kthm
justify(struct ps *P, kt f, const char *rest)
{
	const char *b = find_kw(rest, "by");

	if (b)
		return by(P, f, b + 2);
	P->S->i++;
	if (!cur_is(P, "proof")) {
		P->S->i--;
		serr(P->S, "expected 'by ...' or a proof block");
		return KNONE;
	}
	{
		struct ps Q = *P;
		kthm th;
		P->S->i++;
		Q.S = P->S;
		th = block(&Q, f);
		P->steps += Q.steps;
		return th;	/* at qed: the caller's loop moves past it */
	}
}

/* the text of a formula before " by " (or the end) */
static char *
formula_text(const char *s)
{
	const char *b = find_kw(s, "by");
	size_t n = b ? (size_t)(b - s) : strlen(s);
	char *r = xmalloc(n + 1);

	memcpy(r, s, n);
	r[n] = 0;
	return r;
}

/* flatten a conjunction into the named facts */
static void
add_named(struct ps *P, kthm th, char names[][64], int nn, int *k)
{
	kt f = kthm_concl(th);

	if (KOP(f) == KO_AND && *k < nn - 1) {
		add_named(P, kr_ande1(th), names, nn, k);
		add_named(P, kr_ande2(th), names, nn, k);
		return;
	}
	ka_add(&P->ctx, th, *k < nn ? sym_cstr(names[*k]) : 0);
	(*k)++;
}

static int
names_of(const char *s, char out[][64], int max)
{
	int n = 0;

	while (n < max && word(&s, out[n], 64))
		n++;
	return n;
}

enum { C_ALLI, C_IMPI, C_NOTI, C_RAA, C_EXE };
struct closer {
	int kind;
	kt f;
	kt var;
	kthm ex;
};

static kthm
block(struct ps *P, kt goal)
{
	struct src *S = P->S;
	struct closer cl[128];
	int ncl = 0, i;
	kthm done = KNONE;
	kt G = goal, G0 = goal;

	for (; S->i < S->n && !S->failed; S->i++) {
		const char *t = skipws(S->l[S->i].text), *s;
		char kw[16], names[16][64];
		int nn, k;

		s = t;
		word(&s, kw, sizeof kw);
		P->steps++;
		if (!strcmp(kw, "qed")) {
			if (!done)
				return serr(S, "the block ends before showing %s",
				    kt_str(G)), KNONE;
			for (i = ncl - 1; i >= 0 && done; i--) {
				switch (cl[i].kind) {
				case C_ALLI:
					done = kr_alli(done, cl[i].var);
					break;
				case C_IMPI:
					done = kr_impi(done, cl[i].f);
					break;
				case C_NOTI:
					done = kr_noti(done, cl[i].f);
					break;
				case C_RAA:
					done = kr_raa(done, cl[i].f);
					break;
				case C_EXE:
					done = kr_exe(cl[i].ex, cl[i].var, done);
					break;
				}
				if (!done)
					return serr(S, "closing the block: %s", kerr()),
					    KNONE;
			}
			done = kr_conv(done, G0);
			if (!done)
				serr(S, "%s", kerr());
			return done;
		}
		if (done)
			return serr(S, "the goal is already shown; expected qed"),
			    KNONE;
		if (!strcmp(kw, "fix")) {
			char w[64];
			while (word(&s, w, sizeof w)) {
				kt v;
				if (KOP(G) != KO_ALL)
					return serr(S, "fix %s: the goal %s is not "
					    "universal", w, kt_str(G)), KNONE;
				v = freshvar(P, w, (int)KT.aux[G]);
				cl[ncl].kind = C_ALLI;
				cl[ncl++].var = v;
				G = kt_inst(KA(G, 0), v);
				scope_add(&P->sc, w, v);
			}
		} else if (!strcmp(kw, "intro")) {
			kthm a;
			nn = names_of(s, names, 16);
			if (KOP(G) == KO_IMP) {
				cl[ncl].kind = C_IMPI;
				cl[ncl++].f = KA(G, 0);
				a = kr_assume(KA(G, 0));
				G = KA(G, 1);
			} else if (KOP(G) == KO_NOT) {
				cl[ncl].kind = C_NOTI;
				cl[ncl++].f = KA(G, 0);
				a = kr_assume(KA(G, 0));
				G = kf_false();
			} else
				return serr(S, "intro: the goal %s is not an "
				    "implication", kt_str(G)), KNONE;
			k = 0;
			add_named(P, a, names, nn ? nn : 1, &k);
		} else if (!strcmp(kw, "contra")) {
			kthm a;
			nn = names_of(s, names, 16);
			if (KOP(G) == KO_NOT) {
				cl[ncl].kind = C_NOTI;
				cl[ncl++].f = KA(G, 0);
				a = kr_assume(KA(G, 0));
			} else {
				cl[ncl].kind = C_RAA;
				cl[ncl++].f = G;
				a = kr_assume(kf_not(G));
			}
			G = kf_false();
			k = 0;
			add_named(P, a, names, nn ? nn : 1, &k);
		} else if (!strcmp(kw, "have")) {
			char name[64];
			char *ft;
			kt f;
			kthm th;
			const char *colon;
			if (!word(&s, name, sizeof name))
				return serr(S, "have NAME: formula"), KNONE;
			colon = strchr(s, ':');
			if (!colon)
				return serr(S, "have NAME: formula"), KNONE;
			ft = formula_text(colon + 1);
			f = parse_f(P, ft);
			free(ft);
			if (!f)
				return KNONE;
			th = justify(P, f, colon + 1);
			if (!th)
				return KNONE;
			ka_add(&P->ctx, kr_conv(th, f), sym_cstr(name));
		} else if (!strcmp(kw, "show")) {
			char *ft;
			kt f = G;
			kthm th;
			const char *rest = s;
			s = skipws(s);
			if (!iskw(s, "by") && *s) {
				ft = formula_text(s);
				f = parse_f(P, ft);
				free(ft);
				if (!f)
					return KNONE;
			}
			th = justify(P, f, rest);
			if (!th)
				return KNONE;
			done = kr_conv(th, G);
			if (!done)
				return serr(S, "show: %s", kerr()), KNONE;
		} else if (!strcmp(kw, "obtain")) {
			/* obtain x y:sort, z:sort st h1 h2: F by ... */
			const char *st = find_kw(s, "st"), *colon;
			char vars[512], *ft, buf[2048];
			kt f, cur;
			kthm th;
			int nv = 0;
			char vn[16][64];
			if (!st)
				return serr(S, "obtain VARS st NAMES: formula"), KNONE;
			snprintf(vars, sizeof vars, "%.*s", (int)(st - s), s);
			colon = strchr(st, ':');
			if (!colon)
				return serr(S, "obtain VARS st NAMES: formula"), KNONE;
			{
				char nb[512];
				snprintf(nb, sizeof nb, "%.*s", (int)(colon - st - 2),
				    st + 2);
				nn = names_of(nb, names, 16);
			}
			ft = formula_text(colon + 1);
			snprintf(buf, sizeof buf, "exists %s: (%s)", vars, ft);
			free(ft);
			f = parse_f(P, buf);
			if (!f)
				return KNONE;
			th = justify(P, f, colon + 1);
			if (!th)
				return KNONE;
			th = kr_conv(th, f);
			/* the variable names, in order (sorts skipped) */
			{
				const char *q = vars;
				char w[64], srt[64];
				while (nv < 16 && word(&q, w, sizeof w)) {
					snprintf(vn[nv++], 64, "%s", w);
					q = skipws(q);
					if (*q == ':') {
						q++;
						word(&q, srt, sizeof srt);
						q = skipws(q);
						if (*q == ',')
							q++;
					}
				}
			}
			cur = f;
			for (i = 0; i < nv; i++) {
				kt v, a;
				if (KOP(cur) != KO_EX)
					return serr(S, "obtain: too many variables"),
					    KNONE;
				v = freshvar(P, vn[i], (int)KT.aux[cur]);
				a = kt_inst(KA(cur, 0), v);
				cl[ncl].kind = C_EXE;
				cl[ncl].ex = th;
				cl[ncl++].var = v;
				th = kr_assume(a);
				cur = a;
				scope_add(&P->sc, vn[i], v);
			}
			k = 0;
			add_named(P, th, names, nn ? nn : 1, &k);
		} else if (!strcmp(kw, "cases")) {
			char name[64], *ft;
			const char *colon = strchr(s, ':');
			kt f;
			kthm th, c1, c2;
			if (!word(&s, name, sizeof name) || !colon)
				return serr(S, "cases NAME: A or B by ..."), KNONE;
			ft = formula_text(colon + 1);
			f = parse_f(P, ft);
			free(ft);
			if (!f)
				return KNONE;
			if (KOP(f) != KO_OR)
				return serr(S, "cases: not a disjunction"), KNONE;
			th = justify(P, f, colon + 1);
			if (!th)
				return KNONE;
			th = kr_conv(th, f);
			for (k = 0; k < 2; k++) {
				struct ps Q;
				kthm a;
				int kk = 0;
				S->i++;
				if (S->i >= S->n || !iskw(S->l[S->i].text, "case"))
					return serr(S, "expected case"), KNONE;
				s = skipws(S->l[S->i].text) + 4;
				nn = names_of(s, names, 16);
				Q = *P;
				a = kr_assume(KA(f, k));
				add_named(&Q, a, names, nn ? nn : 1, &kk);
				S->i++;
				if (!cur_is(&Q, "proof"))
					return serr(S, "expected proof"), KNONE;
				S->i++;
				if (k == 0)
					c1 = block(&Q, G);
				else
					c2 = block(&Q, G);
				P->steps += Q.steps;
				if (!(k ? c2 : c1))
					return KNONE;
			}
			done = kr_ore(th, c1, c2);
			if (!done)
				return serr(S, "cases: %s", kerr()), KNONE;
		} else if (!strcmp(kw, "induct")) {
			char v[64], err[256];
			const char *fr = find_kw(s, "from");
			kt base, pat, want, stepg;
			kthm tb, ts;
			struct ps Q;
			if (!word(&s, v, sizeof v) || !fr)
				return serr(S, "induct VAR from BASE"), KNONE;
			base = kt_parse_term(fr + 4, &P->sc, KS_INT, err, sizeof err);
			if (!base)
				return serr(S, "%s", err), KNONE;
			if (KOP(G) != KO_ALL || KT.aux[G] != KS_INT ||
			    KOP(KA(G, 0)) != KO_IMP)
				return serr(S, "induct: the goal is not forall %s:int: "
				    "B <= %s -> ...", v, v), KNONE;
			want = KA(KA(G, 0), 0);
			if (kn_norm(want) != kn_norm(kf_le(base, kt_bvar(0, KS_INT))))
				return serr(S, "induct: the goal's hypothesis is %s",
				    kt_str(want)), KNONE;
			pat = KA(KA(G, 0), 1);
			stepg = kf_bind(KO_ALL, KS_INT, sym_cstr(v),
			    kf_imp(kf_and(kf_le(base, kt_bvar(0, KS_INT)), pat),
			    kt_inst(pat, kt_add(kt_bvar(0, KS_INT), kt_num(1)))));
			S->i++;
			if (!cur_is(P, "base"))
				return serr(S, "expected base"), KNONE;
			S->i++;
			if (!cur_is(P, "proof"))
				return serr(S, "expected proof"), KNONE;
			S->i++;
			Q = *P;
			tb = block(&Q, kt_inst(pat, base));
			P->steps += Q.steps;
			if (!tb)
				return KNONE;
			S->i++;
			if (!cur_is(P, "step"))
				return serr(S, "expected step"), KNONE;
			S->i++;
			if (!cur_is(P, "proof"))
				return serr(S, "expected proof"), KNONE;
			S->i++;
			Q = *P;
			ts = block(&Q, stepg);
			P->steps += Q.steps;
			if (!ts)
				return KNONE;
			tb = kr_conv(tb, kt_inst(pat, base));
			ts = kr_conv(ts, stepg);
			done = tb && ts ? kr_induct(base, pat, tb, ts) : KNONE;
			if (!done)
				return serr(S, "induct: %s", kerr()), KNONE;
			done = kr_conv(done, G);
			if (!done)
				return serr(S, "induct: %s", kerr()), KNONE;
		} else
			return serr(S, "unknown step '%s'", kw), KNONE;
	}
	serr(S, "missing qed");
	return KNONE;
}

/* ---- theory files ---- */

static int
statement(struct src *S, const char *s, char *name, size_t nn, kt *phi)
{
	const char *colon;
	char err[256];

	if (!word(&s, name, nn))
		return serr(S, "expected a name");
	colon = strchr(s, ':');
	if (!colon)
		return serr(S, "expected ':'");
	*phi = kt_parse(colon + 1, NULL, err, sizeof err);
	if (!*phi)
		return serr(S, "%s", err);
	if (KSORT(*phi) != KS_PROP || KT.lb[*phi])
		return serr(S, "not a closed formula");
	return 1;
}

static char *
xstrdup(const char *s)
{
	size_t n = strlen(s) + 1;
	char *r = xmalloc(n);

	memcpy(r, s, n);
	return r;
}

static int
run(struct src *S)
{
	char kw[16], name[64];
	const char *s;
	kt phi;

	for (S->i = 0; S->i < S->n && !S->failed; S->i++) {
		const char *t = S->l[S->i].text;
		s = t;
		if (!word(&s, kw, sizeof kw))
			return serr(S, "expected a declaration");
		if (!strcmp(kw, "sort") || !strcmp(kw, "func") ||
		    !strcmp(kw, "pred")) {
			if (!do_decl(S, t))
				return 0;
		} else if (!strcmp(kw, "def") || !strcmp(kw, "arith")) {
			if (!statement(S, s, name, sizeof name, &phi))
				return 0;
			if (kax_add(name, phi, !strcmp(kw, "def") ? "definition" :
			    "integer arithmetic", xstrdup(skipws(strchr(s, ':') +
			    1))) < 0)
				return serr(S, "axiom %s: duplicate or not closed",
				    name);
		} else if (!strcmp(kw, "lemma") || !strcmp(kw, "theorem")) {
			struct ps P;
			int li;
			kthm th;
			uint32_t k0 = kthm_count();
			int l0 = S->i;
			if (!statement(S, s, name, sizeof name, &phi))
				return 0;
			li = kl_add(name, phi, xstrdup(skipws(strchr(s, ':') + 1)),
			    1);
			memset(&P, 0, sizeof P);
			P.S = S;
			if (S->i + 1 < S->n && iskw(S->l[S->i + 1].text, "by")) {
				S->i++;
				th = by(&P, phi, skipws(S->l[S->i].text) + 2);
				P.steps = 1;
			} else {
				S->i++;
				if (!cur_is(&P, "proof"))
					return serr(S, "expected proof or by");
				S->i++;
				th = block(&P, phi);
			}
			if (!th)
				return 0;
			klemmas[li].l0 = l0;
			klemmas[li].l1 = S->i;
			th = kr_conv(th, phi);
			if (!th || kthm_nhyps(th))
				return serr(S, "%s: the proof leaves hypotheses",
				    name);
			klemmas[li].th = th;
			klemmas[li].steps = (uint32_t)P.steps;
			klemmas[li].kfirst = k0;
			klemmas[li].klast = kthm_count();
			klemmas[li].route = !strcmp(kw, "theorem") ? "theorem" :
			    "lemma";
		} else if (!strcmp(kw, "homdecl")) {
			/* homdecl f by add_law smul_law */
			char f[64], a[64], m[64];
			int sym, ia, im;
			kthm th;
			if (!word(&s, f, sizeof f) || !iskw(s, "by"))
				return serr(S, "homdecl F by ADD SMUL");
			s = skipws(s) + 2;
			if (!word(&s, a, sizeof a) || !word(&s, m, sizeof m))
				return serr(S, "homdecl F by ADD SMUL");
			sym = ksym_lookup(f);
			ia = kl_find(a);
			im = kl_find(m);
			if (sym < 0 || ia < 0 || im < 0)
				return serr(S, "homdecl: unknown name");
			th = kr_homdecl(sym, klemmas[ia].th, klemmas[im].th);
			if (!th)
				return serr(S, "%s", kerr());
		} else
			return serr(S, "unknown declaration '%s'", kw);
	}
	return !S->failed;
}

int
ks_run(const char *text, const char *file, char *msg, int n)
{
	struct src S;
	int ok;

	memset(&S, 0, sizeof S);
	S.file = file;
	S.msg = msg;
	S.msgn = n;
	msg[0] = 0;
	split_lines(&S, text);
	ok = run(&S);
	return ok;
}

static const char lib_text[] =
#include "thy_absalg.inc"
;

static int loaded;
char ks_load_msg[1024];
static struct src libsrc;

int
ks_add_user(const char *name, kt phi, const char *src)
{
	return kl_add(name, phi, src, 0);
}

/* the proof of a library lemma as written (indented by nesting) */
void
ks_print_script(int li)
{
	int i, d = 2;

	if (!klemmas[li].lib || klemmas[li].l1 <= klemmas[li].l0 ||
	    klemmas[li].l1 >= libsrc.n)
		return;
	for (i = klemmas[li].l0 + 1; i <= klemmas[li].l1; i++) {
		const char *t = skipws(libsrc.l[i].text);
		if (iskw(t, "qed"))
			d--;
		printf("%*s%s\n", 2 * d, "", t);
		if (iskw(t, "proof"))
			d++;
	}
}

void
ks_load_library(void)
{
	if (loaded)
		return;
	loaded = 1;
	memset(&libsrc, 0, sizeof libsrc);
	libsrc.file = "thy_absalg.thy";
	libsrc.msg = ks_load_msg;
	libsrc.msgn = sizeof ks_load_msg;
	split_lines(&libsrc, lib_text);
	if (!run(&libsrc))
		fail("the proof library does not load: %s", ks_load_msg);
}
