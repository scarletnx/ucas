/*
 * The algebra commands (alg.c, solve.c), included at the end of parse.c:
 *
 *   expand F    factor F    simplify F    together F    cancel F
 *   collect F [in|for|wrt x]     apart F [in|for|wrt x]
 *   gcd P, Q    lcm P, Q    resultant P, Q [in x]    discriminant P [in x]
 *   quo P, Q [in x]    rem P, Q [in x]    divide P by Q [in x]
 *   solve L = R [, L2 = R2 ...] [for x [, y]]
 *   substitute x = a [, y = b] in F     (also subst)
 *
 * Every expression is parsed on its own (cmd_func_text); variables with
 * the same name are the same variable (alg_canon).
 */
#include "alg.h"

static const char *const alg_words[] = {"expand", "factor", "collect",
	"simplify", "together", "apart", "cancel", "gcd", "lcm", "resultant",
	"discriminant", "quo", "rem", "divide", "solve", "substitute", "subst",
	NULL};

/* the tokens [a, b) as an expression in the command's variables */
static nid
alg_expr(struct P *p, uint32_t a, uint32_t b)
{
	char *t;
	struct func *f;
	nid e;

	if (b <= a)
		fail("expected an expression");
	t = cs_text(p, a, b);
	f = cmd_func_text(t);
	free(t);
	e = alg_canon(f);
	return e;
}

/* the top-level commas in [a, b): piece boundaries into at[] */
static int
alg_split(struct P *p, uint32_t a, uint32_t b, uint32_t *at, int max,
    int kind)
{
	int depth = 0, n = 0;
	uint32_t i;

	at[n++] = a;
	for (i = a; i < b; i++) {
		int k = p->t->kind[i];
		if (k == T_LP || k == T_LB || k == T_LC)
			depth++;
		else if (k == T_RP || k == T_RB || k == T_RC)
			depth--;
		else if (depth == 0 && k == kind && n < max)
			at[n++] = i + 1;
	}
	at[n] = b + 1;
	return n;
}

/* an identifier list [a, b) (x, y): canonical variables */
static int
alg_varlist(struct P *p, uint32_t a, uint32_t b, nid *v, int max)
{
	uint32_t i;
	int n = 0;

	for (i = a; i < b; i++) {
		if (p->t->kind[i] == T_COMMA)
			continue;
		if (p->t->kind[i] != T_ID || n == max) {
			diag_tok = i;
			fail("expected a variable name");
		}
		v[n++] = alg_var_named(tsym(p, i));
	}
	return n;
}

static int
alg_command(struct P *p)
{
	static const char *const kws[] = {"for", "wrt", "in", "by", NULL};
	char cmd[16];
	uint32_t s, e, k1, at[17];
	int w, n, i;
	nid var = 0;

	if (pk(p) != T_ID || func_find(tsym(p, p->i)))
		return 0;
	for (w = 0; alg_words[w]; w++)
		if (is_kw(p, alg_words[w]))
			break;
	if (!alg_words[w])
		return 0;
	snprintf(cmd, sizeof cmd, "%s", alg_words[w]);
	s = p->i + 1;
	e = cs_stmt_end(p, s);
	if (!strcmp(cmd, "solve")) {
		/* an equation is needed; "solve y from 0 to 2" is dsolve's */
		for (i = (int)s; i < (int)e; i++)
			if (p->t->kind[i] == T_ASSIGN || p->t->kind[i] == T_EQ)
				break;
		if (i == (int)e)
			return 0;
	}
	if (!strcmp(cmd, "subst"))
		strcpy(cmd, "substitute");
	adv(p);
	alg_vars_reset();
	if (!strcmp(cmd, "substitute")) {
		/* substitute x = a, y = b in F */
		nid vars[8], vals[8], body;
		int nv = 0;
		k1 = c2_find_kw(p, s, e, (const char *const[]){"in", "into", NULL});
		if (k1 == e)
			fail("substitute x = a in F");
		body = alg_expr(p, k1 + 1, e);
		n = alg_split(p, s, k1, at, 8, T_COMMA);
		for (i = 0; i < n; i++) {
			uint32_t a = at[i], b = at[i + 1] - 1;
			if (b < a + 3 || p->t->kind[a] != T_ID ||
			    p->t->kind[a + 1] != T_ASSIGN)
				fail("substitute: write name = value");
			vars[nv] = alg_var_named(tsym(p, a));
			vals[nv++] = alg_expr(p, a + 2, b);
		}
		p->i = e;
		notes_flush();
		alg_subst(body, vars, vals, nv);
		return 1;
	}
	k1 = c2_find_kw(p, s, e, kws);
	if (!strcmp(cmd, "solve")) {
		nid eq[8], vars[8];
		static char eqt[8][160];
		const char *eqtp[8];
		int neq = 0, nv = 0;
		n = alg_split(p, s, k1, at, 8, T_COMMA);
		for (i = 0; i < n; i++) {
			uint32_t a = at[i], b = at[i + 1] - 1, j;
			int depth = 0;
			for (j = a; j < b; j++) {
				int k = p->t->kind[j];
				if (k == T_LP || k == T_LB)
					depth++;
				else if (k == T_RP || k == T_RB)
					depth--;
				else if (depth == 0 && (k == T_ASSIGN || k == T_EQ))
					break;
			}
			if (j == b)
				eq[neq] = alg_expr(p, a, b);   /* F: F = 0 */
			else
				eq[neq] = SUB(alg_expr(p, a, j), alg_expr(p, j + 1, b));
			{
				char *t = cs_text(p, a, b);
				snprintf(eqt[neq], sizeof eqt[neq], "%s%s", t, j == b ?
				    " = 0" : "");
				free(t);
				eqtp[neq] = eqt[neq];
			}
			neq++;
		}
		if (k1 < e)
			nv = alg_varlist(p, k1 + 1, e, vars, 8);
		p->i = e;
		notes_flush();
		if (nv == 0) {
			/* all the variables; one equation: x, or ask */
			int k;
			extern int alg_nvars(uint32_t *names);
			uint32_t names[MAXP];
			int nn = alg_nvars(names);
			if (neq == 1 && nn > 1) {
				const char *opt[MAXP];
				int dflt = 0;
				for (k = 0; k < nn; k++) {
					opt[k] = sym_str(names[k]);
					if (!strcmp(opt[k], "x"))
						dflt = k;
				}
				k = nlp_ask_choice("solve for which variable?", opt, nn,
				    dflt);
				if (nlp_asked)
					return 1;
				vars[nv++] = alg_var_named(names[k]);
			} else
				for (k = 0; k < nn && nv < 8; k++)
					vars[nv++] = alg_var_named(names[k]);
		}
		alg_solve(eq, eqtp, neq, vars, nv);
		return 1;
	}
	if (!strcmp(cmd, "divide")) {
		uint32_t kb = c2_find_kw(p, s, e, (const char *const[]){"by", NULL});
		nid a, b;
		if (kb == e)
			fail("divide P by Q");
		k1 = c2_find_kw(p, kb + 1, e, (const char *const[]){"in", "wrt",
		    "for", NULL});
		a = alg_expr(p, s, kb);
		b = alg_expr(p, kb + 1, k1);
		if (k1 < e)
			alg_varlist(p, k1 + 1, e, &var, 1);
		p->i = e;
		notes_flush();
		alg_cmd2("divide", a, b, var);
		return 1;
	}
	n = alg_split(p, s, k1, at, 8, T_COMMA);
	if (k1 < e)
		alg_varlist(p, k1 + 1, e, &var, 1);
	{
		/* "collect F, x": a trailing single name is the variable */
		int two = !strcmp(cmd, "gcd") || !strcmp(cmd, "lcm") ||
		    !strcmp(cmd, "resultant") || !strcmp(cmd, "quo") ||
		    !strcmp(cmd, "rem");
		int want = two ? 2 : 1;
		if (n == want + 1 && at[n] - 1 == at[n - 1] + 1 &&
		    p->t->kind[at[n - 1]] == T_ID && !var) {
			var = alg_var_named(tsym(p, at[n - 1]));
			n--;
		}
		if (n != want)
			fail("%s takes %s", cmd, two ? "two polynomials: P, Q" :
			    "one expression");
		if (two) {
			nid a = alg_expr(p, at[0], at[1] - 1), b = alg_expr(p, at[1],
			    at[2] - 1);
			p->i = e;
			notes_flush();
			alg_cmd2(cmd, a, b, var);
		} else {
			nid a = alg_expr(p, at[0], at[1] - 1);
			p->i = e;
			notes_flush();
			alg_cmd1(cmd, a, var);
		}
	}
	return 1;
}
