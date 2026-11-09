/*
 * English for the ring commands: the sentence is normalised (lower case,
 * "modulo" and "(mod n)" to "mod n", "≡" and "is congruent to" to "=",
 * "with the AES polynomial" to "aes"), then matched against phrase
 * templates whose slots must look like numbers, rings or polynomials.
 * nlp.c calls ring_nlp_raw before its own parser, and ring_nlp_english
 * when deciding whether a line is English at all.
 */
#include "ring.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define MAXWD 160
static char nbuf[2048];
static char *wd[MAXWD];
static int nwd;
static char cap[6][600];
static char out[2400];

/* what "it" refers to */
static char last_ring[200], last_poly[600];

static void
put_space_around(char *s, size_t cap_, const char *tok)
{
	char tmp[2048];
	size_t n = strlen(tok), o = 0;
	const char *p;

	for (p = s; *p && o + n + 3 < sizeof tmp;) {
		if (!strncmp(p, tok, n)) {
			tmp[o++] = ' ';
			memcpy(tmp + o, tok, n);
			o += n;
			tmp[o++] = ' ';
			p += n;
		} else
			tmp[o++] = *p++;
	}
	tmp[o] = 0;
	snprintf(s, cap_, "%s", tmp);
}

/* replace every whole-word occurrence of from (words separated by spaces) */
static void
replace_words(char *s, size_t cap_, const char *from, const char *to)
{
	char tmp[2048];
	size_t n = strlen(from), o = 0;
	const char *p = s;

	while (*p && o + strlen(to) + 2 < sizeof tmp) {
		if (!strncmp(p, from, n) && (p == s || p[-1] == ' ') &&
		    (p[n] == ' ' || p[n] == 0)) {
			memcpy(tmp + o, to, strlen(to));
			o += strlen(to);
			p += n;
		} else
			tmp[o++] = *p++;
	}
	tmp[o] = 0;
	snprintf(s, cap_, "%s", tmp);
}

static void
normalise(const char *sent)
{
	char *p, *q;
	size_t o = 0;
	const unsigned char *u;
	int k;

	for (u = (const unsigned char *)sent; *u && o + 8 < sizeof nbuf; u++) {
		if (u[0] == 0xe2 && u[1] == 0x89 && u[2] == 0xa1) {   /* ≡ */
			memcpy(nbuf + o, " = ", 3);
			o += 3;
			u += 2;
		} else if (u[0] == 0xe2 && u[1] == 0x80 && (u[2] == 0x99 ||
		    u[2] == 0x98)) {
			nbuf[o++] = '\'';
			u += 2;
		} else
			nbuf[o++] = (char)tolower(*u);
	}
	nbuf[o] = 0;
	/* trim ? . ! and leading ? */
	for (p = nbuf; *p == ' ' || *p == '?'; p++)
		;
	memmove(nbuf, p, strlen(p) + 1);
	o = strlen(nbuf);
	while (o && (nbuf[o - 1] == '?' || nbuf[o - 1] == '.' ||
	    nbuf[o - 1] == '!' || nbuf[o - 1] == ' '))
		nbuf[--o] = 0;
	put_space_around(nbuf, sizeof nbuf, ",");
	put_space_around(nbuf, sizeof nbuf, "=");
	/* "(mod 7)" -> "mod 7" */
	for (p = nbuf; (p = strstr(p, "(mod")) != NULL;) {
		q = strchr(p, ')');
		*p = ' ';
		if (q)
			*q = ' ';
	}
	{
		static const char *const rep[][2] = {
			{"modulo", "mod"}, {"is congruent to", "="},
			{"congruent to", "="}, {"equals", "="}, {"is equal to", "="},
			{"with the aes polynomial", "aes"},
			{"using the aes polynomial", "aes"}, {"with aes polynomial", "aes"},
			{"with the aes modulus", "aes"}, {"aes polynomial", "aes"},
			{"with the rijndael polynomial", "aes"},
			{"the aes field", "gf(2^8) aes"}, {"aes field", "gf(2^8) aes"},
			{"the rijndael field", "gf(2^8) aes"},
			{"with the crc-32 polynomial", "crc32"},
			{"the crc-32 polynomial", "crc32"}, {"crc-32", "crc32"},
			{"the crc32 polynomial", "crc32"}, {"crc32 polynomial", "crc32"},
			{"the integers mod", "z/"}, {"integers mod", "z/"},
			{"the integers modulo", "z/"}, {"galois field", "gf"},
			{"the field", ""}, {"the ring", ""}, {"ring of integers mod", "z/"},
			{"the gaussian integers", "z[i]"}, {"gaussian integers", "z[i]"},
			{"multiplicative inverse", "inverse"}, {"modular inverse", "inverse"},
			{"discrete logarithm", "dlog"}, {"discrete log", "dlog"},
			{"euler's totient", "phi"}, {"euler totient", "phi"},
			{"euler's phi", "phi"}, {"euler phi", "phi"}, {"totient", "phi"},
			{"carmichael function", "carmichael"},
			{"carmichael lambda", "carmichael"},
			{"multiplicative order", "order"}, {"zero divisors", "zerodivisors"},
			{"zero-divisors", "zerodivisors"}, {"null space", "nullspace"},
			{"integral domain", "domain"}, {"primitive roots", "primroots"},
			{"primitive root", "primroot"}, {"magic numbers", "magic"},
			{"magic number", "magic"}, {"magic constant", "magic"},
			{"multiply-high", "magic"}, {"quadratic residues", "qrs"},
			{"quadratic residue", "qr"}, {"square roots", "sqrt"},
			{"square root", "sqrt"}, {"log and antilog tables", "table"},
			{"log/antilog tables", "table"}, {"log table", "table"},
			{"antilog table", "table"}, {"logarithm table", "table"},
			{"log tables", "table"}, {"legendre symbol", "legendre"},
			{"jacobi symbol", "jacobi"}, {"chinese remainder theorem", "crt"},
			{"the chinese remainder", "crt"},
			{NULL, NULL}};
		for (k = 0; rep[k][0]; k++)
			replace_words(nbuf, sizeof nbuf, rep[k][0], rep[k][1]);
	}
	/* words */
	nwd = 0;
	for (p = nbuf; *p && nwd < MAXWD;) {
		while (*p == ' ')
			*p++ = 0;
		if (!*p)
			break;
		wd[nwd++] = p;
		while (*p && *p != ' ')
			p++;
	}
}

static int
wis(int i, const char *alts)
{
	const char *a = alts;
	size_t n;

	if (i >= nwd)
		return 0;
	for (;;) {
		const char *b = strchr(a, '|');
		n = b ? (size_t)(b - a) : strlen(a);
		if (strlen(wd[i]) == n && !strncmp(wd[i], a, n))
			return 1;
		if (!b)
			return 0;
		a = b + 1;
	}
}

/* pattern: words, "a|b" alternatives, "a?" optional, "{k}" a capture */
static char pw[64][48];
static int npw;

static int
m_rec(int pi, int wi)
{
	if (pi == npw)
		return wi == nwd;
	if (pw[pi][0] == '{') {
		int k = pw[pi][1] - '0', e;
		for (e = wi + 1; e <= nwd; e++)
			if (m_rec(pi + 1, e)) {
				size_t o = 0;
				int j;
				cap[k][0] = 0;
				for (j = wi; j < e && o + 2 < sizeof cap[k]; j++)
					o += (size_t)snprintf(cap[k] + o, sizeof cap[k] - o,
					    "%s%s", j > wi ? " " : "", wd[j]);
				return 1;
			}
		return 0;
	}
	{
		size_t n = strlen(pw[pi]);
		if (n > 1 && pw[pi][n - 1] == '?') {
			char a[48];
			snprintf(a, sizeof a, "%.*s", (int)(n - 1), pw[pi]);
			if (wis(wi, a) && m_rec(pi + 1, wi + 1))
				return 1;
			return m_rec(pi + 1, wi);
		}
		return wis(wi, pw[pi]) && m_rec(pi + 1, wi + 1);
	}
}

static int
match(const char *pat)
{
	const char *p = pat;

	npw = 0;
	while (*p && npw < 64) {
		size_t n = strcspn(p, " ");
		snprintf(pw[npw++], sizeof pw[0], "%.*s", (int)n, p);
		p += n;
		while (*p == ' ')
			p++;
	}
	return m_rec(0, 0);
}

/* slot checks */
static int
is_num(const char *s)
{
	int dig = 0;

	for (; *s; s++) {
		if (isdigit((unsigned char)*s))
			dig = 1;
		else if (!strchr(" +-*^()_xabcdef", *s))
			return 0;
	}
	return dig;
}

static int
is_ring(const char *s)
{
	if (!strncmp(s, "z/", 2) || !strncmp(s, "gf", 2) || !strncmp(s, "zmod", 4) ||
	    !strncmp(s, "f_", 2) || !strncmp(s, "q[x]", 4) || !strncmp(s, "z[", 2) ||
	    !strncmp(s, "z_", 2) || !strncmp(s, "z /", 3))
		return 1;
	return 0;
}

static int
is_poly(const char *s)
{
	int x = 0;

	for (; *s; s++) {
		if (*s == 'x')
			x = 1;
		else if (!isdigit((unsigned char)*s) && !strchr(" +-*^()", *s))
			return 0;
	}
	return x;
}

static int
is_matrix(const char *s)
{
	return s[0] == '[';
}

/* "a u32" / "an i64" / "u32" / "unsigned 32-bit" -> "u32" */
static const char *
type_of(const char *s)
{
	static char t[16];
	const char *p = s;
	int sg = -1, w = 0;

	if (!strncmp(p, "a ", 2))
		p += 2;
	if (!strncmp(p, "an ", 3))
		p += 3;
	if ((p[0] == 'u' || p[0] == 'i') && isdigit((unsigned char)p[1])) {
		snprintf(t, sizeof t, "%.*s", (int)strcspn(p, " "), p);
		return t;
	}
	if (strstr(p, "unsigned"))
		sg = 0;
	else if (strstr(p, "signed"))
		sg = 1;
	for (; *p; p++)
		if (isdigit((unsigned char)*p)) {
			w = atoi(p);
			break;
		}
	if (strstr(s, "byte"))
		w = 8;
	if (strstr(s, "short"))
		w = 16;
	if (strstr(s, "int") && !w)
		w = 32;
	if (strstr(s, "long"))
		w = 64;
	if (!w)
		return NULL;
	snprintf(t, sizeof t, "%s%d", sg == 1 ? "i" : "u", w);
	if (sg < 0 && !strstr(s, "uint"))
		return t;
	return t;
}

static const char *
R(const char *ring)
{
	snprintf(last_ring, sizeof last_ring, "%s", ring);
	return ring;
}

static const char *
P(const char *poly)
{
	snprintf(last_poly, sizeof last_poly, "%s", poly);
	return poly;
}

static void
strip_lead(void)
{
	static const char *const lead[] = {"what", "whats", "what's", "is", "are",
		"the", "compute", "calculate", "find", "give", "me", "tell", "show",
		"list", "determine", "please", "can", "could", "you", "us", "get",
		"work", "out", "evaluate", "all", "a", "an", "of", "do", "does",
		"i", "need", "want", "to", "know", NULL};
	int k, i = 0;

	for (;;) {
		for (k = 0; lead[k]; k++)
			if (i < nwd && !strcmp(wd[i], lead[k]))
				break;
		if (!lead[k])
			break;
		i++;
	}
	/* keep "is" when it starts a yes/no question */
	if (i > 0 && (!strcmp(wd[i - 1], "is") || !strcmp(wd[i - 1], "are")) &&
	    i < nwd && strcmp(wd[i], "the"))
		i--;
	memmove(wd, wd + i, sizeof(char *) * (size_t)(nwd - i));
	nwd -= i;
}

static const char *
join_from(int a)
{
	static char b[2048];
	size_t o = 0;
	int j;

	b[0] = 0;
	for (j = a; j < nwd && o + 2 < sizeof b; j++)
		o += (size_t)snprintf(b + o, sizeof b - o, "%s%s", j > a ? " " : "",
		    wd[j]);
	return b;
}

static int
has_word(const char *w)
{
	int i;

	for (i = 0; i < nwd; i++)
		if (!strcmp(wd[i], w))
			return 1;
	return 0;
}

#define OUT(...) (snprintf(out, sizeof out, __VA_ARGS__), out)

/* a congruence or a system of them: "x = 2 mod 3 , x = 3 mod 5" */
static const char *
congruences(const char *s)
{
	char t[2048];
	int neq = 0, nmod = 0;
	const char *p;

	snprintf(t, sizeof t, "%s", s);
	replace_words(t, sizeof t, "and", ",");
	for (p = t; *p; p++)
		neq += *p == '=';
	for (p = t; (p = strstr(p, "mod")) != NULL; p += 3)
		nmod++;
	if (neq == 0 || nmod == 0)
		return NULL;
	/* x^... = a mod n with a numeric base: a discrete log */
	{
		char g[64], a[200], n[200];
		if (sscanf(t, " %63[0-9] ^ x = %199[^m] mod %199[^,]", g, a, n) == 3 &&
		    !strchr(t, ','))
			return OUT("dlog %s base %s mod %s", a, g, n);
	}
	return OUT("%s %s", neq > 1 || strchr(t, ',') ? "crt" : "congruence", t);
}

const char *
ring_nlp_raw(const char *sent)
{
	const char *t;

	normalise(sent);
	if (nwd == 0)
		return NULL;
	strip_lead();
	if (nwd == 0)
		return NULL;

	/* pronouns: it = the last ring or polynomial */
	if ((match("is it primitive") || match("is it a primitive polynomial")) &&
	    last_poly[0])
		return OUT("primitive %s over %s", last_poly, last_ring[0] ? last_ring :
		    "gf(2)");
	if (match("is it irreducible") && last_poly[0])
		return OUT("irreducible %s over %s", last_poly, last_ring[0] ?
		    last_ring : "gf(2)");
	if (match("is it a field") && last_ring[0])
		return OUT("ring %s field", last_ring);
	if (match("is it a|an? domain") && last_ring[0])
		return OUT("ring %s domain", last_ring);
	if (match("its|the? units|zerodivisors|nilpotents|idempotents|ideals|"
	    "subfields|generators|characteristic|table") && last_ring[0])
		return OUT("ring %s %s", last_ring, wd[nwd - 1]);

	/* inverses */
	if ((match("inverse of? {0} mod {1}") || match("{0} inverse mod {1}") ||
	    match("{0} ^ -1 mod {1}") || match("1 / {0} mod {1}")) &&
	    (is_num(cap[0]) || is_matrix(cap[0])) && is_num(cap[1]))
		return OUT("inverse %s mod %s", cap[0], cap[1]);
	if (match("inverse of? {0} in|over {1}") && is_ring(cap[1]))
		return OUT("inverse %s in %s", cap[0], R(cap[1]));
	if (match("inverse of? {0} mod {1}") && is_matrix(cap[0]))
		return OUT("inverse %s mod %s", cap[0], cap[1]);

	/* order, phi, lambda */
	if ((match("order of? {0} mod {1}") || match("order of? {0} in {1}")) &&
	    (is_num(cap[0]) || is_poly(cap[0])) &&
	    (is_num(cap[1]) || is_ring(cap[1])))
		return OUT("order %s %s %s", cap[0], is_num(cap[1]) ? "mod" : "in",
		    cap[1]);
	if ((match("period of? the? lfsr with? feedback? polynomial? {0}") ||
	    match("order of? x mod {0}")) && is_poly(cap[0]))
		return OUT("order %s over gf(2)", P(cap[0]));
	if ((match("phi of? {0}") || match("phi function of? {0}")) &&
	    is_num(cap[0]))
		return OUT("phi %s", cap[0]);
	if (nwd == 1 && !strncmp(wd[0], "phi(", 4) && is_num(wd[0] + 4))
		return OUT("phi %.*s", (int)strlen(wd[0] + 4) - 1, wd[0] + 4);
	if ((match("carmichael of? {0}") || match("lambda of? {0}")) &&
	    is_num(cap[0]))
		return OUT("carmichael %s", cap[0]);

	/* primitive roots, generators */
	if ((match("is {0} a primroot mod {1}") ||
	    match("is {0} a primroot of {1}") || match("is {0} a generator mod {1}"))
	    && is_num(cap[0]) && is_num(cap[1]))
		return OUT("primroot %s mod %s", cap[0], cap[1]);
	if ((match("is {0} a generator of|in {1}") ||
	    match("is {0} a primitive element of|in {1}")) && is_ring(cap[1]))
		return OUT("primroot %s in %s", cap[0], R(cap[1]));
	if ((match("smallest? primroot|primroots mod|of|for {0}") ||
	    match("generators of|for {0}")) && is_num(cap[0]))
		return OUT("primroot %s", cap[0]);
	if (match("generators of|in|for {0}") && is_ring(cap[0]))
		return OUT("ring %s generators", R(cap[0]));

	/* discrete logs */
	if ((match("dlog of? {0} base|to {1} mod {2}") ||
	    match("dlog of? {0} to the base {1} mod {2}") ||
	    match("dlog base {1} of {0} mod {2}") ||
	    match("log base {1} of {0} mod {2}")) && is_num(cap[2]))
		return OUT("dlog %s base %s mod %s", cap[0], cap[1], cap[2]);
	if ((match("dlog of? {0} base|to {1} in {2}") ||
	    match("dlog base {1} of {0} in {2}")) && is_ring(cap[2]))
		return OUT("dlog %s base %s in %s", cap[0], cap[1], R(cap[2]));
	if (match("dlog of? {0} in {1}") && is_ring(cap[1]))
		return OUT("dlog %s in %s", cap[0], R(cap[1]));

	/* Legendre, Jacobi, squares */
	if ((match("legendre of? {0} and|over|mod {1}") ||
	    match("legendre ( {0} / {1} )")) && is_num(cap[0]) && is_num(cap[1]))
		return OUT("legendre %s %s", cap[0], cap[1]);
	if (match("jacobi of? {0} and|over|mod {1}") && is_num(cap[0]) &&
	    is_num(cap[1]))
		return OUT("jacobi %s %s", cap[0], cap[1]);
	if ((match("is {0} a qr mod {1}") || match("is {0} a square mod {1}") ||
	    match("sqrt of? {0} mod {1}") || match("sqrts of? {0} mod {1}")) &&
	    is_num(cap[0]) && is_num(cap[1]))
		return OUT("sqrtmod %s mod %s", cap[0], cap[1]);
	if (match("sqrt of? {0} in {1}") && is_ring(cap[1]))
		return OUT("sqrt %s in %s", cap[0], R(cap[1]));
	if (match("qrs mod|of|for {0}") && is_num(cap[0]))
		return OUT("qr %s", cap[0]);

	/* magic numbers for division */
	if (has_word("magic") || (has_word("divide") && has_word("multiply"))) {
		const char *ty = NULL;
		char d[64] = "";
		int i;
		for (i = 0; i < nwd; i++) {
			if ((wd[i][0] == 'u' || wd[i][0] == 'i') &&
			    isdigit((unsigned char)wd[i][1]))
				ty = type_of(wd[i]);
			if (!strcmp(wd[i], "by") && i + 1 < nwd && is_num(wd[i + 1]))
				snprintf(d, sizeof d, "%s", wd[i + 1]);
			if ((!strcmp(wd[i], "byte") || !strcmp(wd[i], "bytes")) && !ty)
				ty = "u8";
		}
		if (!d[0])
			for (i = 0; i < nwd; i++)
				if (is_num(wd[i]) && strspn(wd[i], "0123456789") ==
				    strlen(wd[i])) {
					snprintf(d, sizeof d, "%s", wd[i]);
					break;
				}
		if (d[0])
			return OUT("magic %s%s%s%s", d, ty ? " " : "", ty ? ty : "",
			    has_word("signed") && !ty ? " signed" : "");
	}

	/* rings: field? domain? units ... */
	if (match("is {0} a field") && is_ring(cap[0]))
		return OUT("ring %s field", R(cap[0]));
	if (match("is {0} a|an? domain") && is_ring(cap[0]))
		return OUT("ring %s domain", R(cap[0]));
	{
		static const char *const subs[] = {"units", "zerodivisors",
			"nilpotents", "idempotents", "ideals", "subfields", "generators",
			"characteristic", "table", "elements", "unit", "nilpotent",
			"idempotent", NULL};
		int k;
		for (k = 0; subs[k]; k++) {
			char pat[96];
			snprintf(pat, sizeof pat, "%s of|in|for {0}", subs[k]);
			if (match(pat) && is_ring(cap[0]))
				return OUT("ring %s %s", R(cap[0]), subs[k]);
			snprintf(pat, sizeof pat, "{0} %s", subs[k]);
			if (match(pat) && is_ring(cap[0]))
				return OUT("ring %s %s", R(cap[0]), subs[k]);
		}
	}
	if ((match("describe {0}") || match("about {0}") || match("{0}")) &&
	    is_ring(cap[0]) && !strchr(cap[0], '='))
		return OUT("ring %s", R(cap[0]));

	/* arithmetic in a ring */
	if ((match("in {0} multiply {1} by|and|with {2}") ||
	    match("multiply {1} by|and|with {2} in {0}") ||
	    match("product of {1} and {2} in {0}") ||
	    match("{1} times {2} in {0}")) && is_ring(cap[0]))
		return OUT("ring %s (%s) * (%s)", R(cap[0]), cap[1], cap[2]);
	if ((match("in {0} divide {1} by {2}") || match("divide {1} by {2} in {0}"))
	    && is_ring(cap[0]))
		return OUT("ring %s (%s) / (%s)", R(cap[0]), cap[1], cap[2]);
	if ((match("in {0} add {1} and|to {2}") || match("add {1} and|to {2} in {0}")
	    || match("sum of {1} and {2} in {0}")) && is_ring(cap[0]))
		return OUT("ring %s (%s) + (%s)", R(cap[0]), cap[1], cap[2]);
	if ((match("in {0} invert {1}") || match("invert {1} in {0}")) &&
	    is_ring(cap[0]))
		return OUT("inverse %s in %s", cap[1], R(cap[0]));
	if ((match("minimal polynomial of {1} in|over {0}") ||
	    match("minpoly of {1} in {0}")) && is_ring(cap[0]))
		return OUT("ring %s minpoly %s", R(cap[0]), cap[1]);
	if ((match("in {0} {1}") || match("{1} in {0}")) && is_ring(cap[0]) &&
	    !is_ring(cap[1]) && !strstr(cap[1], " is "))
		return OUT("ring %s %s", R(cap[0]), cap[1]);

	/* polynomials */
	if ((match("factor|factorise|factorize {0} over|mod|in {1}") ||
	    match("factorization|factorisation of {0} over|mod|in {1}") ||
	    match("factors of {0} over|mod|in {1}")) && (is_ring(cap[1]) ||
	    is_num(cap[1])))
		return OUT("factor %s %s %s", P(cap[0]), is_num(cap[1]) ? "mod" :
		    "over", R(cap[1]));
	if ((match("factor|factorise|factorize {0}") ||
	    match("prime factors of {0}") || match("factorization of {0}")) &&
	    is_num(cap[0]) && !strchr(cap[0], 'x'))
		return OUT("factorint %s", cap[0]);
	if (match("is {0} prime") && is_num(cap[0]) && !strchr(cap[0], 'x'))
		return OUT("factorint %s", cap[0]);
	if ((match("is {0} irreducible over|mod|in {1}") ||
	    match("is {0} an? irreducible polynomial over|mod|in {1}")) &&
	    is_poly(cap[0]))
		return OUT("irreducible %s %s %s", P(cap[0]), is_num(cap[1]) ? "mod" :
		    "over", R(cap[1]));
	if (match("is {0} irreducible") && is_poly(cap[0])) {
		last_ring[0] = 0;
		return OUT("irreducible %s", P(cap[0]));
	}
	if ((match("is {0} primitive over|mod|in {1}") ||
	    match("is {0} a primitive polynomial over|mod|in {1}")) &&
	    is_poly(cap[0]))
		return OUT("primitive %s %s %s", P(cap[0]), is_num(cap[1]) ? "mod" :
		    "over", R(cap[1]));
	if ((match("is {0} primitive") || match("is {0} a primitive polynomial"))
	    && is_poly(cap[0]))
		return OUT("primitive %s over gf(2)", P(cap[0]));
	if ((match("roots of {0} mod|over|in {1}") ||
	    match("zeros of {0} mod|over|in {1}")) && is_poly(cap[0]))
		return OUT("roots %s %s %s", P(cap[0]), is_num(cap[1]) ? "mod" : "in",
		    R(cap[1]));
	if ((match("gcd of? {0} and {1} mod|over|in {2}") ||
	    match("greatest common divisor of {0} and {1} mod|over|in {2}")))
		return OUT("gcd %s, %s %s %s", cap[0], cap[1], is_num(cap[2]) ? "mod" :
		    "in", cap[2]);
	if (match("norm of {0} in {1}") && is_ring(cap[1]))
		return OUT("norm %s in %s", cap[0], R(cap[1]));

	/* matrices */
	if ((match("determinant|det of {0} mod|over {1}")) && is_matrix(cap[0]))
		return OUT("det %s %s %s", cap[0], is_num(cap[1]) ? "mod" : "over",
		    cap[1]);
	if (match("rank of {0} mod|over {1}") && is_matrix(cap[0]))
		return OUT("rank %s %s %s", cap[0], is_num(cap[1]) ? "mod" : "over",
		    cap[1]);
	if ((match("nullspace|kernel of {0} mod|over {1}")) && is_matrix(cap[0]))
		return OUT("nullspace %s %s %s", cap[0], is_num(cap[1]) ? "mod" :
		    "over", cap[1]);
	if (match("solve {0} x = {1} mod|over {2}") && is_matrix(cap[0]))
		return OUT("matsolve %s, %s %s %s", cap[0], cap[1], is_num(cap[2]) ?
		    "mod" : "over", cap[2]);

	/* GF(2) matrices of bit expressions */
	if (match("gf(2) matrix of {0}") || match("gf2 matrix of {0}") ||
	    match("matrix of {0} over gf(2)") || match("is {0} linear over gf(2)") ||
	    match("{0} as a gf(2) matrix") || match("gf(2)-matrix of {0}")) {
		char e[600];
		const char *ty = NULL, *p;
		snprintf(e, sizeof e, "%s", cap[0]);
		/* "... for a u8", "... on u16", "... as a u32" */
		for (p = e; (p = strstr(p, " ")) != NULL; p++) {
			const char *q = p + 1;
			if (!strncmp(q, "for ", 4) || !strncmp(q, "on ", 3) ||
			    !strncmp(q, "as ", 3) || !strncmp(q, "in ", 3)) {
				const char *r = strchr(q, ' ') + 1;
				if ((ty = type_of(r)) != NULL) {
					e[p - e] = 0;
					break;
				}
			}
		}
		return OUT("gf2matrix %s%s%s", e, ty ? " " : "", ty ? ty : "");
	}

	/* congruences and systems */
	if (has_word("mod") && strchr(join_from(0), '=')) {
		int i = 0;
		while (i < nwd && (wis(i, "solve|find|for|x|such|that|with|all|"
		    "the|system|crt|:") && !(i + 1 < nwd && !strcmp(wd[i + 1], "="))))
			i++;
		if (i < nwd && (t = congruences(join_from(i))) != NULL)
			return t;
	}
	if (match("crt {0}") && strstr(cap[0], "mod"))
		return OUT("crt %s", cap[0]);
	return NULL;
}

/* 0: a ring command as ucas, 1: English this module reads, -1: undecided */
int
ring_nlp_english(const char *s, unsigned long n)
{
	static const char *const raw[] = {"ring", "zmod", "gf", "phi", "totient",
		"carmichael", "factorint", "legendre", "jacobi", "sqrtmod", "qr",
		"primroot", "primroots", "dlog", "crt", "congruence", "magic",
		"irreducible", "matinv", "matsolve", "gf2matrix", "gf2map", "invmod",
		"modinv", "multorder", "lambda", NULL};
	static const char *const shared[] = {"inverse", "inv", "order", "ord",
		"factor", "gcd", "primitive", "roots", "solve", "det", "rank",
		"matrix", "poly", "norm", "log", "sqrt", "nullspace", "kernel", NULL};
	static const char *const english[] = {"of", "the", "what", "whats",
		"is", "are", "does", "how", "please", "me", "which", "whether",
		"an", "find", "compute", "number", "such", "that", "for", "by",
		"dividing", "divide", "symbol", "root", "using", "polynomial",
		"with", "and", NULL};
	char line[2048], w[64];
	size_t i = 0, k, after = 0;
	int first = 1, israw = 0, isshared = 0, eng = 0, ctx = 0;

	if (n >= sizeof line)
		return -1;
	memcpy(line, s, n);
	line[n] = 0;
	while (i < n) {
		if (isalpha((unsigned char)line[i])) {
			k = 0;
			while (i < n && (isalnum((unsigned char)line[i]) ||
			    line[i] == '_') && k + 1 < sizeof w)
				w[k++] = (char)tolower((unsigned char)line[i++]);
			w[k] = 0;
			if (first) {
				int j;
				for (j = 0; raw[j]; j++)
					if (!strcmp(raw[j], w))
						israw = 1;
				for (j = 0; shared[j]; j++)
					if (!strcmp(shared[j], w))
						isshared = 1;
				first = 0;
				after = i;
				if (!strcmp(w, "gf") && i < n && line[i] == '(')
					israw = 1;
				continue;
			}
			{
				int j;
				for (j = 0; english[j]; j++)
					if (!strcmp(english[j], w))
						eng = 1;
			}
			continue;
		}
		if ((unsigned char)line[i] == 0xe2 && i + 2 < n &&
		    (unsigned char)line[i + 1] == 0x89 &&
		    (unsigned char)line[i + 2] == 0xa1)
			eng = 1;    /* ≡: only English reads it */
		i++;
	}
	if (first)
		return -1;
	if (isshared && !eng)
		ctx = ring_has_ring(line + after);
	if (israw && !eng)
		return 0;
	if (isshared && ctx && !eng)
		return 0;
	if (line[n - 1] == '?' || eng || strstr(line, "\xe2\x89\xa1")) {
		if (ring_nlp_raw(line) != NULL)
			return 1;
	}
	return -1;
}
