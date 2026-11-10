/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Tokenizer.  One pass over the source into four parallel arrays; numbers
 * and identifiers are kept as (offset, length) and decoded on demand.
 */
#include "ucas.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static const char *const names[T_NTOK] = {
	"end of input", "end of line", "number", "identifier", "(", ")", "[",
	"]", "{", "}", ",", ":", "?", "=", "==", "!=", "<", "<=", ">", ">=",
	"+", "-", "*", "/", "%", "^", "&", "|", "~", "!", "<<", ">>", "&&",
	"||", "=>", "..", "xor", "sqrt", "superscript",
};

const char *
tok_name(int kind)
{
	return kind >= 0 && kind < T_NTOK ? names[kind] : "?";
}

static void
push(struct toks *t, int kind, uint32_t off, uint32_t len, int fl)
{
	if (t->n == t->cap) {
		t->cap = newcap(t->cap, t->n + 1);
		t->kind = xrealloc(t->kind, t->cap);
		t->fl = xrealloc(t->fl, t->cap);
		t->off = xrealloc(t->off, t->cap * sizeof *t->off);
		t->len = xrealloc(t->len, t->cap * sizeof *t->len);
	}
	t->kind[t->n] = (uint8_t)kind;
	t->fl[t->n] = (uint8_t)fl;
	t->off[t->n] = off;
	t->len[t->n] = len;
	t->n++;
}

void
toks_free(struct toks *t)
{
	free(t->kind);
	free(t->fl);
	free(t->off);
	free(t->len);
	memset(t, 0, sizeof *t);
}

int
tok_is(const struct toks *t, uint32_t i, const char *word)
{
	size_t n = strlen(word);

	return t->kind[i] == T_ID && t->len[i] == n &&
	    memcmp(t->src + t->off[i], word, n) == 0;
}

/* UTF-8 operators: returns bytes consumed, 0 if none */
static int
utf8_op(const unsigned char *s, int *kind)
{
	static const struct {
		const char *u;
		int kind;
	} tab[] = {
		{"\xc2\xb7", T_STAR},          /* · */
		{"\xc3\x97", T_STAR},          /* × */
		{"\xc3\xb7", T_SLASH},         /* ÷ */
		{"\xe2\x89\xa4", T_LE},        /* ≤ */
		{"\xe2\x89\xa5", T_GE},        /* ≥ */
		{"\xe2\x89\xa0", T_NE},        /* ≠ */
		{"\xe2\x8a\x95", T_XOR},       /* ⊕ */
		{"\xe2\x88\x9a", T_SQRT},      /* √ */
		{"\xe2\x88\xa7", T_ANDAND},    /* ∧ */
		{"\xe2\x88\xa8", T_OROR},      /* ∨ */
		{"\xc2\xb2", T_SUP},           /* ² */
		{"\xc2\xb3", T_SUP},           /* ³ */
		{"\xe2\x86\x92", T_ARROW},     /* → */
		{"\xe2\x88\x92", T_MINUS},     /* − */
	};
	size_t i, n;

	for (i = 0; i < sizeof tab / sizeof tab[0]; i++) {
		n = strlen(tab[i].u);
		if (strncmp((const char *)s, tab[i].u, n) == 0) {
			*kind = tab[i].kind;
			return (int)n;
		}
	}
	return 0;
}

void
lex(struct toks *t, const char *src)
{
	const unsigned char *s = (const unsigned char *)src;
	uint32_t i = 0, st;
	int depth = 0, fl = TF_SPACE, k, n;

	t->n = 0;
	t->src = src;
	for (;;) {
		unsigned c = s[i];

		if (c == ' ' || c == '\t' || c == '\r') {
			i++;
			fl = TF_SPACE;
			continue;
		}
		if (t->n > 0 && t->kind[t->n - 1] == T_ID && t->len[t->n - 1] == 4 &&
		    !strncmp(src + t->off[t->n - 1], "file", 4) && c != '\n' &&
		    c != ';' && c != '#' && c != 0) {
			/* "input 1 file a/b.png": the path is one word */
			for (n = 0; s[i + n] && !isspace(s[i + n]) && s[i + n] != ';';
			    n++)
				;
			push(t, T_ID, i, (uint32_t)n, fl);
			i += (uint32_t)n;
			fl = 0;
			continue;
		}
		if (c == '#' && t->n > 0 && t->kind[t->n - 1] != T_NL) {
			/* a colour after a word, "input 1 #ff8800": a word */
			for (n = 1; isxdigit(s[i + n]); n++)
				;
			if ((n == 4 || n == 5 || n == 7 || n == 9) &&
			    !isalnum(s[i + n]) && s[i + n] != '_') {
				push(t, T_ID, i, (uint32_t)n, fl);
				i += (uint32_t)n;
				fl = 0;
				continue;
			}
		}
		if (c == '#' || (c == '/' && s[i + 1] == '/')) {
			while (s[i] && s[i] != '\n')
				i++;
			continue;
		}
		if (c == '\\' && s[i + 1] == '\n') {
			i += 2;
			fl = TF_SPACE;
			continue;
		}
		if (c == '\n' || c == ';') {
			/*
			 * Inside ( or [ a newline continues the statement, but
			 * only onto an indented line: a line starting in column
			 * 0 always starts a new statement, so one unclosed
			 * parenthesis cannot swallow the rest of a file.
			 */
			if (c == '\n' && depth > 0 && s[i + 1] != ' ' &&
			    s[i + 1] != '\t')
				depth = 0;
			if (!(c == '\n' && depth > 0))
				push(t, T_NL, i, 1, fl);
			i++;
			fl = TF_SPACE;
			continue;
		}
		if (c == 0) {
			push(t, T_EOF, i, 0, fl);
			return;
		}
		st = i;
		if (isdigit(c) || (c == '.' && isdigit(s[i + 1]))) {
			if (c == '0' && (s[i + 1] == 'x' || s[i + 1] == 'X' ||
			    s[i + 1] == 'b' || s[i + 1] == 'B')) {
				i += 2;
				while (isxdigit(s[i]) || s[i] == '_')
					i++;
			} else {
				while (isdigit(s[i]) || s[i] == '_')
					i++;
				if (s[i] == '.' && s[i + 1] != '.') {
					i++;
					while (isdigit(s[i]))
						i++;
				}
				/* 2e3 is a number; 2e and 2ex are 2*e... */
				if ((s[i] == 'e' || s[i] == 'E') &&
				    (isdigit(s[i + 1]) ||
				    ((s[i + 1] == '-' || s[i + 1] == '+') &&
				    isdigit(s[i + 2])))) {
					i += 2;
					while (isdigit(s[i]))
						i++;
				}
				/* C float suffix on a decimal: 1.5f is an f32 */
				if ((s[i] == 'f' || s[i] == 'F') &&
				    memchr(s + st, '.', i - st) != NULL &&
				    !isalnum(s[i + 1]) && s[i + 1] != '_')
					i++;
			}
			push(t, T_NUM, st, i - st, fl);
		} else if (isalpha(c) || c == '_') {
			while (isalnum(s[i]) || s[i] == '_')
				i++;
			while (s[i] == '\'')   /* calc: y' y'' (derivatives) */
				i++;
			push(t, T_ID, st, i - st, fl);
		} else if ((c == 0xce && s[i + 1] == 0xa3) || (c == 0xe2 &&
		    s[i + 1] == 0x88 && (s[i + 2] == 0x8f || s[i + 2] == 0xab))) {
			/* calc: Σ ∏ ∫ are names (sum, prod, integral) */
			n = c == 0xce ? 2 : 3;
			i += (uint32_t)n;
			push(t, T_ID, st, (uint32_t)n, fl);
		} else if ((c == 0xcf && s[i + 1] == 0x80) ||   /* π */
		    (c == 0xc2 && s[i + 1] == 0xb0)) {        /* ° (forms.c) */
			i += 2;
			push(t, T_ID, st, 2, fl);
		} else if (c == 0xe2 && s[i + 1] == 0x88 && s[i + 2] == 0xa0) {
			i += 3;   /* ∠: r∠theta, polar notation (forms.c) */
			push(t, T_ID, st, 3, fl);
		} else if ((n = utf8_op(s + i, &k)) > 0) {
			i += (uint32_t)n;
			push(t, k, st, (uint32_t)n, fl);
		} else {
			k = -1;
			n = 1;
			switch (c) {
			case '(': k = T_LP; depth++; break;
			case ')': k = T_RP; depth -= depth > 0; break;
			case '[': k = T_LB; depth++; break;
			case ']': k = T_RB; depth -= depth > 0; break;
			case '{': k = T_LC; break;
			case '}': k = T_RC; break;
			case ',': k = T_COMMA; break;
			case ':': k = T_COLON; break;
			case '?': k = T_QUES; break;
			case '+': k = T_PLUS; break;
			case '*':
				k = T_STAR;
				if (s[i + 1] == '*')   /* ** is power */
					k = T_CARET, n = 2;
				break;
			case '/': k = T_SLASH; break;
			case '%': k = T_PCT; break;
			case '^': k = T_CARET; break;
			case '~': k = T_TILDE; break;
			case '-':
				k = T_MINUS;
				if (s[i + 1] == '>')
					k = T_ARROW, n = 2;
				break;
			case '=':
				k = T_ASSIGN;
				if (s[i + 1] == '=')
					k = T_EQ, n = 2;
				else if (s[i + 1] == '>')
					k = T_ARROW, n = 2;
				break;
			case '!':
				k = T_BANG;
				if (s[i + 1] == '=')
					k = T_NE, n = 2;
				break;
			case '<':
				k = T_LT;
				if (s[i + 1] == '=')
					k = T_LE, n = 2;
				else if (s[i + 1] == '<')
					k = T_SHL, n = 2;
				break;
			case '>':
				k = T_GT;
				if (s[i + 1] == '=')
					k = T_GE, n = 2;
				else if (s[i + 1] == '>')
					k = T_SHR, n = 2;
				break;
			case '&':
				k = T_AMP;
				if (s[i + 1] == '&')
					k = T_ANDAND, n = 2;
				break;
			case '|':
				/*
				 * "a || b" is logical or; "||x| - 1|" opens two
				 * absolute values, "|y||" closes two.  || is
				 * only an operator with blanks on both sides.
				 */
				k = T_BAR;
				if (s[i + 1] == '|' && i > 0 &&
				    (s[i - 1] == ' ' || s[i - 1] == '\t') &&
				    (s[i + 2] == ' ' || s[i + 2] == '\t'))
					k = T_OROR, n = 2;
				break;
			case '.':
				if (s[i + 1] == '.')
					k = T_DOTDOT, n = 2;
				break;
			}
			if (k < 0) {
				push(t, T_EOF, i, 1, fl);
				diag_toks = t;
				diag_tok = t->n - 1;
				fail("unexpected character '%c'", c >= 32 && c < 127 ? c : '?');
			}
			i += (uint32_t)n;
			push(t, k, st, (uint32_t)n, fl);
		}
		fl = 0;
	}
}
