/*
 * English for the algebra commands (alg_syntax.inc), included by nlp.c
 * before sentence().  alg_sentence() writes one ucas line:
 *
 *   factor x^4 - 1, factorise ..., the factors of P
 *   expand (x+1)^5, multiply out ...           (not trig: forms.c's)
 *   simplify (x^2-1)/(x-1), reduce, cancel     (not trig: forms.c's)
 *   (the) partial fraction decomposition|expansion of F [in x],
 *   decompose F into partial fractions, apart F
 *   combine F into one|a single fraction, put F over a common denominator
 *   collect (the terms of) F in|by|with respect to x
 *   solve x^2 - 5x + 6 = 0 [for x], solve sin x = 1/2,
 *   solve the system x + y = 3, x - y = 1 [for x and y], solve A and B
 *   the gcd|hcf|greatest common divisor of P and Q, the lcm|least common
 *   multiple of P and Q, the resultant of P and Q [in x], the
 *   discriminant of P [in x], divide P by Q, the quotient|remainder of P
 *   divided by Q, the remainder when P is divided by Q
 *   substitute|plug x = 2 [and y = 3] in|into F
 */

static int
alg_depth_at(int a, int i)
{
	int d = 0, k;

	for (k = a; k < i; k++)
		if (W[k].k == W_SYM && (!strcmp(W[k].s, "(") || !strcmp(W[k].s, "[")))
			d++;
		else if (W[k].k == W_SYM && (!strcmp(W[k].s, ")") ||
		    !strcmp(W[k].s, "]")))
			d--;
	return d;
}

/* the first word of the |-list at depth 0 in [a, b), else b */
static int
alg_find(int a, int b, const char *list)
{
	int i;

	for (i = a; i < b; i++) {
		char buf[256], *p, *q;
		if (alg_depth_at(a, i) != 0)
			continue;
		snprintf(buf, sizeof buf, "%s", list);
		for (p = buf; p; p = q) {
			q = strchr(p, '|');
			if (q)
				*q++ = 0;
			if ((W[i].k == W_WORD && !strcmp(W[i].s, p)) ||
			    (W[i].k == W_SYM && !strcmp(W[i].s, p)))
				return i;
		}
	}
	return b;
}

/* "in x", "for x", "wrt x", "with respect to x", "in terms of x" at the end */
static const char *
alg_tailvar(int a, int *b)
{
	int e = *b;

	while (e > a && W[e - 1].k == W_SYM && (!strcmp(W[e - 1].s, "?") ||
	    !strcmp(W[e - 1].s, ".")))
		e--;
	if (e - a >= 2 && W[e - 1].k == W_WORD && varlike(e - 1)) {
		int s = e - 1;
		if (isw(s - 1, "in") || isw(s - 1, "for") || isw(s - 1, "wrt") ||
		    isw(s - 1, "by")) {
			if (isw(s - 1, "in") && s - 3 >= a && isw(s - 2, "terms") &&
			    isw(s - 3, "in"))
				;
			*b = s - 1;
			if (isw(*b - 1, "terms") && isw(*b - 2, "in"))
				*b -= 2;
			else if (isw(*b - 1, "terms") && isw(*b - 2, "of") &&
			    isw(*b - 3, "in"))
				*b -= 3;
			return W[s].o;
		}
		if (s - 3 >= a && isw(s - 1, "to") && isw(s - 2, "respect") &&
		    isw(s - 3, "with")) {
			*b = s - 3;
			return W[s].o;
		}
	}
	return NULL;
}

/* trig, hyperbolic or complex words: forms.c rewrites those */
static int
alg_trigish(int a, int b)
{
	static const char *const w[] = {"sin", "cos", "tan", "sec", "csc", "cot",
		"sinh", "cosh", "tanh", "asin", "acos", "atan", "arcsin", "arccos",
		"arctan", "sine", "cosine", "tangent", "trig", "polar", "complex",
		"cis", "using", "angle", "identity", "identities", "into", "as",
		"form", NULL};
	int i;

	for (i = a; i < b; i++)
		if (W[i].k == W_WORD && (in_list(w, W[i].s) || !strcmp(W[i].o, "i")))
			return 1;
	return 0;
}

static int
alg_out(const char *line)
{
	Q.raw[0] = S("%s", line);
	Q.cmd = C_RAW;
	Q.nraw = 1;
	wi = nw;
	return 1;
}

/* end of the words, without a trailing ? or . */
static int
alg_end(void)
{
	int b = nw;

	while (b > wi && W[b - 1].k == W_SYM && (!strcmp(W[b - 1].s, "?") ||
	    !strcmp(W[b - 1].s, ".")))
		b--;
	return b;
}

/* a, b, ... and z in [a, b): ucas texts joined with ", " */
static const char *
alg_list(int a, int b)
{
	struct sbuf sb = {0};
	const char *r;
	int s = a, i;

	for (i = a; i <= b; i++) {
		if (i == b || (alg_depth_at(a, i) == 0 && ((W[i].k == W_SYM &&
		    !strcmp(W[i].s, ",")) || isw(i, "and")))) {
			int e = i;
			if (e > s) {
				if (sb.n)
					sb_puts(&sb, ", ");
				sb_puts(&sb, form_expr(s, e));
			}
			s = i + 1;
		}
	}
	r = S("%s", sb.p ? sb.p : "");
	free(sb.p);
	return r;
}

static int
alg_sentence(void)
{
	int s = wi, a, b, k;
	const char *v;

	/* y' = ..., f''(x): a differential equation (dsolve), not algebra */
	for (k = wi; k < nw; k++)
		if (W[k].k == W_WORD && W[k].prime)
			return 0;
	mat("?please|pls|kindly");
	mat("?can|could|would|will you");
	mat("?please");
	mat("?find|compute|calculate|determine|give|get|show|tell|evaluate|"
	    "work|what|whats|what's ?me|us|out|is|are");
	mat("?the");
	a = wi;
	b = alg_end();
	/* factor */
	if (mat("factor|factorise|factorize|factorization|factorisation|"
	    "factors|factorizations") || mat("factored form")) {
		mat("?of");
		mat("?the ?polynomial|expression");
		a = wi;
		k = alg_find(a, b, "over|completely");
		if (k < b && (isw(k, "completely") || isw(k + 1, "the") ||
		    isw(k + 1, "rationals") || isw(k + 1, "q")))
			b = k;
		if (a >= b)
			goto no;
		return alg_out(S("factor %s", form_expr(a, b)));
	}
	wi = a;
	/* partial fractions */
	if (mat("partial fraction|fractions ?decomposition|expansion|form ?of|for") ||
	    mat("apart")) {
		mat("?the");
		a = wi;
		v = alg_tailvar(a, &b);
		if (a >= b)
			goto no;
		return alg_out(v ? S("apart %s in %s", form_expr(a, b), v) :
		    S("apart %s", form_expr(a, b)));
	}
	wi = a;
	if (mat("decompose|split")) {
		a = wi;
		k = alg_find(a, b, "into|in");
		if (k < b && (isw(k + 1, "partial") || isw(k + 2, "partial"))) {
			if (a >= k)
				goto no;
			return alg_out(S("apart %s", form_expr(a, k)));
		}
		goto no;
	}
	wi = a;
	/* expand, simplify, cancel, together, collect */
	if (mat("expand|expanding") || mat("multiply out")) {
		mat("?out");
		a = wi;
		if (alg_trigish(a, b) || a >= b)
			goto no;
		return alg_out(S("expand %s", form_expr(a, b)));
	}
	wi = a;
	if (mat("simplify|simplified|simplification|reduce|cancel|"
	    "normalize|normalise")) {
		int cancel = isw(wi - 1, "cancel");
		mat("?of");
		mat("?the ?expression|fraction|quotient");
		a = wi;
		if (alg_trigish(a, b) || a >= b || code_block)
			goto no;
		return alg_out(S("%s %s", cancel ? "cancel" : "simplify",
		    form_expr(a, b)));
	}
	wi = a;
	if (mat("combine|put|write|bring")) {
		a = wi;
		k = alg_find(a, b, "into|over|as|to|on");
		if (k < b && (alg_find(k, b, "fraction") < b || alg_find(k, b,
		    "denominator") < b)) {
			if (a >= k)
				goto no;
			return alg_out(S("together %s", form_expr(a, k)));
		}
		goto no;
	}
	wi = a;
	if (mat("together|collect")) {
		int col = isw(wi - 1, "collect");
		mat("?like ?the ?terms ?of|in");
		a = wi;
		v = col ? alg_tailvar(a, &b) : NULL;
		if (a >= b)
			goto no;
		return alg_out(v ? S("collect %s in %s", form_expr(a, b), v) :
		    S("%s %s", col ? "collect" : "together", form_expr(a, b)));
	}
	wi = a;
	/* solve */
	if (mat("solve")) {
		const char *vars = NULL;
		int fk;
		/* "solve z^3 = 8i": the complex roots (forms.c) */
		if (W[wi].k == W_WORD && strlen(W[wi].s) == 1 && W[wi + 1].k ==
		    W_SYM && !strcmp(W[wi + 1].s, "^") && W[wi + 2].k == W_NUM &&
		    W[wi + 3].k == W_SYM && !strcmp(W[wi + 3].s, "=") &&
		    (!strcmp(W[wi].s, "z") || !strcmp(W[wi].s, "w") ||
		    has_unit_word(wi + 4, b)))
			goto no;
		mat("?the ?system|equations|equation ?of");
		mat("?equations|equation");
		a = wi;
		if (alg_find(a, b, "=") == b)
			goto no;      /* an ODE's unknown: dsolve */
		for (k = a; k < b; k++)
			if (W[k].k == W_SYM && !strcmp(W[k].s, "'"))
				goto no;
		/* "for x and y" at the end */
		fk = alg_find(a, b, "for");
		if (fk < b) {
			vars = alg_list(fk + 1, b);
			b = fk;
		}
		mat("?:");
		if (a >= b)
			goto no;
		return alg_out(vars ? S("solve %s for %s", alg_list(a, b), vars) :
		    S("solve %s", alg_list(a, b)));
	}
	wi = a;
	/* gcd, lcm, resultant, discriminant */
	if (mat("gcd|hcf|gcf") || mat("greatest|highest common divisor|factor") ||
	    mat("lcm") || mat("least|lowest common multiple|denominator") ||
	    mat("resultant")) {
		const char *cmd = isw(wi - 1, "resultant") ? "resultant" :
		    (isw(wi - 1, "lcm") || isw(wi - 1, "multiple") ||
		    isw(wi - 1, "denominator")) ? "lcm" : "gcd";
		mat("?of|between");
		a = wi;
		v = !strcmp(cmd, "resultant") ? alg_tailvar(a, &b) : NULL;
		k = alg_find(a, b, "and|,");
		if (k >= b || a >= k)
			goto no;
		return alg_out(v ? S("%s %s, %s in %s", cmd, form_expr(a, k),
		    form_expr(k + 1, b), v) : S("%s %s, %s", cmd, form_expr(a, k),
		    form_expr(k + 1, b)));
	}
	wi = a;
	if (mat("discriminant")) {
		mat("?of");
		mat("?the ?polynomial|quadratic|cubic|equation");
		a = wi;
		v = alg_tailvar(a, &b);
		if (a >= b)
			goto no;
		return alg_out(v ? S("discriminant %s in %s", form_expr(a, b), v) :
		    S("discriminant %s", form_expr(a, b)));
	}
	wi = a;
	/* division */
	if (mat("quotient|remainder|divide|division")) {
		const char *cmd = isw(wi - 1, "quotient") ? "quo" :
		    isw(wi - 1, "remainder") ? "rem" : "divide";
		mat("?of|when|from|in");
		a = wi;
		k = alg_find(a, b, "divided|by|/");
		if (k >= b)
			goto no;
		{
			int e1 = k, s2 = k + 1;
			if (isw(k, "divided"))
				s2 = isw(k + 1, "by") ? k + 2 : k + 1;
			if (e1 > a && isw(e1 - 1, "is"))
				e1--;
			if (a >= e1 || s2 >= b)
				goto no;
			v = alg_tailvar(s2, &b);
			if (!strcmp(cmd, "divide"))
				return alg_out(v ? S("divide %s by %s in %s", form_expr(a,
				    e1), form_expr(s2, b), v) : S("divide %s by %s",
				    form_expr(a, e1), form_expr(s2, b)));
			return alg_out(v ? S("%s %s, %s in %s", cmd, form_expr(a, e1),
			    form_expr(s2, b), v) : S("%s %s, %s", cmd, form_expr(a, e1),
			    form_expr(s2, b)));
		}
	}
	wi = a;
	/* substitution */
	if (mat("substitute|plug|put|set") || mat("plug in")) {
		mat("?in");
		a = wi;
		k = alg_find(a, b, "in|into");
		if (k >= b || alg_find(a, k, "=") == k)
			goto no;
		return alg_out(S("substitute %s in %s", alg_list(a, k),
		    form_expr(k + 1, b)));
	}
no:
	wi = s;
	return 0;
}
