/*
 * English for hyper.c (complex, quaternion, octonion numbers), included by
 * nlp.c after forms_sentence.  hy_sentence() reads the request and writes
 * one ucas line into Q.raw:
 *
 *   multiply (1+2i) by (3-i), divide A by B, the [quaternion] product of
 *   i and j, e to the i pi, is z^2 holomorphic | analytic | entire, the
 *   complex derivative of F, the residue of F at a, the poles of F,
 *   the contour integral of F around the unit circle | |z - 1| = 2,
 *   rotate (1, 0, 0) by 90 degrees about the z axis [using quaternions],
 *   the rotation matrix of q, the quaternion for 90 degrees about (0, 0, 1),
 *   slerp between q1 and q2 at t, is octonion multiplication associative,
 *   are the quaternions commutative, is the sedenion algebra a division
 *   algebra, the octonion multiplication table, the Fano plane, zero
 *   divisors of the sedenions, the associator of e1, e2 and e4, the
 *   commutator of i and j, domain colouring of F.
 */

/* the line is a formula with units (i^i, i*j, 3 + 4i): not English */
static int
hy_formula(const char *s, size_t n)
{
	static const char *const ok[] = {"i", "j", "k", "I", "z", "w", "pi", "e",
		"sin", "cos", "tan", "exp", "log", "ln", "sqrt", "abs", "conj", "re",
		"im", "arg", "norm", "cis", "sinh", "cosh", "tanh", "asin", "acos",
		"atan", NULL};
	size_t j = 0;
	int unit = 0, op = 0;
	char w[40];

	while (j < n) {
		unsigned char c = s[j];
		if (isalpha(c)) {
			size_t k = 0;
			while (j < n && (isalnum((unsigned char)s[j]) || s[j] == '_')) {
				if (k + 1 < sizeof w)
					w[k++] = s[j];
				j++;
			}
			w[k] = 0;
			if (!strcmp(w, "i") || !strcmp(w, "j") || !strcmp(w, "k") ||
			    (w[0] == 'e' && isdigit((unsigned char)w[1])))
				unit = 1;
			else if (!in_list(ok, w) && strlen(w) > 1)
				return 0;
			continue;
		}
		if (strchr("^*/+-()=", c))
			op = 1;
		else if (!isdigit(c) && !strchr(" .,|", c))
			return 0;
		j++;
	}
	return unit && op;
}

/* hyper.c's own commands are ucas, not English */
static int
hy_own_cmd(const char *s, size_t n)
{
	static const char *const c[] = {"axisangle", "rotmatrix", "frommatrix",
		"slerp", "associator", "commutator", "zerodivisors", "algebra",
		"rotate", NULL};
	size_t k = 0;
	char w[24];

	while (k < n && k + 1 < sizeof w && isalpha((unsigned char)s[k])) {
		w[k] = s[k];
		k++;
	}
	w[k] = 0;
	if (!in_list(c, w) || k == n || s[k] != ' ')
		return 0;
	while (k < n && s[k] == ' ')
		k++;
	return strncmp(s + k, "of ", 3) && strncmp(s + k, "the ", 4) &&
	    strncmp(s + k, "me ", 3);
}

/* English that starts like a command or not at all like ucas */
static int
hy_starter(const char *s, size_t n)
{
	static const char *const st[] = {"multiply ", "divide ",
		"domain colouring", "domain coloring", NULL};
	int k;

	for (k = 0; st[k]; k++)
		if (n > strlen(st[k]) && !strncasecmp(s, st[k], strlen(st[k])))
			return 1;
	return 0;
}

/* the end of an operand list item: the first of words at depth 0 */
static int
hy_upto(int a, const char *const *words)
{
	int i, depth = 0;

	for (i = a; i < nw; i++) {
		if (W[i].k == W_SYM && (!strcmp(W[i].s, "(") || !strcmp(W[i].s, "[")))
			depth++;
		if (W[i].k == W_SYM && (!strcmp(W[i].s, ")") || !strcmp(W[i].s, "]")))
			depth--;
		if (depth == 0 && W[i].k == W_WORD && in_list(words, W[i].s))
			return i;
	}
	return nw;
}

/* the end of the sentence, without a trailing ? or . */
static int
hy_end(void)
{
	int b = nw;

	while (b > 0 && W[b - 1].k == W_SYM && (!strcmp(W[b - 1].s, "?") ||
	    !strcmp(W[b - 1].s, ".") || !strcmp(W[b - 1].s, "!")))
		b--;
	return b;
}

static int
hy_algword(int i, const char **name)
{
	if (W[i].k != W_WORD)
		return 0;
	if (!strncmp(W[i].s, "quaternion", 10))
		*name = "quaternion";
	else if (!strncmp(W[i].s, "octonion", 8))
		*name = "octonion";
	else if (!strncmp(W[i].s, "sedenion", 8))
		*name = "sedenion";
	else if (!strcmp(W[i].s, "complex"))
		*name = "complex";
	else
		return 0;
	return 1;
}

static int
hy_sentence(void)
{
	static const char *const props[] = {"commutative", "associative",
		"alternative", "division", "normed", "composition", "flexible",
		NULL};
	static const char *const by_w[] = {"by", "and", "with", NULL};
	static const char *const at_w[] = {"at", "about", "around", "near",
		NULL};
	int s = wi, k, b = hy_end(), i, j;
	const char *ex, *ex2, *alg = NULL, *prop = NULL;

	mat("?please|now");
	mat("?can|could|would ?you");
	mat("?what|whats|which ?is|are");
	mat("?compute|calculate|find|evaluate|give|show|determine|get|draw|"
	    "make|plot");
	mat("?me|us");
	mat("?the|a|an");
	k = wi;

	/* the algebras: "is octonion multiplication associative" */
	for (i = s; i < b; i++)
		if (hy_algword(i, &alg))
			break;
	for (j = s; j < b; j++)
		if (W[j].k == W_WORD && in_list(props, W[j].s)) {
			prop = W[j].s;
			break;
		}
	if (alg && prop && (isw(s, "is") || isw(s, "are") || isw(s, "check") ||
	    isw(s, "whether") || isw(s, "show") || isw(s, "prove"))) {
		/* "is (1+i)*(2) commutative": not about the algebra */
		int other = 0;
		for (i = s; i < b; i++)
			if (W[i].k == W_SYM || W[i].k == W_NUM)
				other = 1;
		if (!other) {
			Q.raw[0] = S("algebra %s %s", alg, prop);
			goto done;
		}
	}
	wi = k;
	if (alg && (mat("?octonion|quaternion|sedenion|complex|octonions|"
	    "quaternions|sedenions multiplication|cayley table|tables ?of|for "
	    "?the ?octonions|quaternions|sedenions|octonion|quaternion|sedenion") ||
	    mat("?octonion|quaternion|sedenion|complex|octonions|quaternions|"
	    "sedenions multiplication|cayley table|tables"))) {
		if (wi == b) {
			Q.raw[0] = S("table %s", alg);
			goto done;
		}
	}
	wi = k;
	if (mat("fano plane ?of|for ?the ?octonions|octonion")) {
		Q.raw[0] = S("table octonion");
		goto done;
	}
	wi = s;
	for (i = s; i < b; i++)
		if (isw(i, "zero") && isw(i + 1, "divisors")) {
			Q.raw[0] = S("zerodivisors %s", alg ? alg : "sedenion");
			goto done;
		}
	wi = k;
	if (alg && !prop && (mat("properties of ?the ?octonions|quaternions|"
	    "sedenions|octonion|quaternion|sedenion|complex ?numbers|algebra") ||
	    mat("?octonion|quaternion|sedenion algebra"))) {
		Q.raw[0] = S("algebra %s", alg);
		goto done;
	}

	/* arithmetic in words */
	wi = s;
	mat("?please");
	if (mat("multiply|divide")) {
		int div = isw(wi - 1, "divide");
		int a = wi, m = hy_upto(a, by_w);
		if (m >= b || (div && !isw(m, "by")))
			goto no;
		ex = form_expr(a, m);
		ex2 = form_expr(m + 1, b);
		if (!*ex || !*ex2)
			goto no;
		Q.raw[0] = S("(%s)%s(%s)", ex, div ? "/" : "*", ex2);
		goto done;
	}
	wi = k;
	if (mat("?quaternion|octonion|sedenion|complex|hypercomplex product of")) {
		static const char *const and_w[] = {"and", NULL};
		int a = wi, m = hy_upto(a, and_w);
		if (m >= b)
			goto no;
		ex = form_expr(a, m);
		ex2 = form_expr(m + 1, b);
		if (!hy_formula(ex, strlen(ex)) && !hy_formula(ex2, strlen(ex2)) &&
		    !alg)
			goto no;
		Q.raw[0] = S("(%s)*(%s)", ex, ex2);
		goto done;
	}
	/* "e to the i pi", "2 to the i" */
	for (i = k + 1; i + 2 < b; i++)
		if (isw(i, "to") && isw(i + 1, "the") && !isw(i + 2, "power") &&
		    has_unit_word(i + 2, b)) {
			ex = form_expr(k, i);
			ex2 = form_expr(i + 2, b);
			if (!*ex || !*ex2)
				goto no;
			Q.raw[0] = S("(%s)^(%s)", ex, ex2);
			goto done;
		}

	/* holomorphy */
	wi = s;
	if (mat("is|are|check ?if|whether") || mat("where is")) {
		int a = wi;
		mat("?the ?function");
		a = wi;
		if (b - 1 > a && (isw(b - 1, "holomorphic") ||
		    isw(b - 1, "analytic") || isw(b - 1, "entire") ||
		    isw(b - 1, "differentiable") || isw(b - 1, "meromorphic"))) {
			int e = b - 1;
			if (e > a && isw(e - 1, "complex"))
				e--;
			if (e > a && (isw(e - 1, "function") || isw(e - 1, "an") ||
			    isw(e - 1, "a")))
				e--;
			if (e > a && isw(e - 1, "an"))
				e--;
			ex = form_expr(a, e);
			if (!*ex)
				goto no;
			Q.raw[0] = S("holomorphic %s", ex);
			goto done;
		}
	}
	wi = k;
	if (mat("complex derivative of")) {
		ex = form_expr(wi, b);
		Q.raw[0] = S("diff %s", ex);
		goto done;
	}
	/* residues and poles */
	wi = k;
	if (mat("residue|residues ?of")) {
		int m = hy_upto(wi, at_w);
		ex = form_expr(wi, m);
		if (!*ex)
			goto no;
		if (m < b) {
			ex2 = form_expr(m + 1, b);
			Q.raw[0] = S("residue %s at %s", ex, ex2);
		} else
			Q.raw[0] = S("poles %s", ex);
		goto done;
	}
	wi = k;
	if (mat("poles|singularities of")) {
		ex = form_expr(wi, b);
		if (!*ex)
			goto no;
		Q.raw[0] = S("poles %s", ex);
		goto done;
	}
	/* contour integrals */
	wi = k;
	if (mat("?contour|line|closed integral of") || mat("integrate")) {
		static const char *const ar_w[] = {"around", "along", NULL};
		int m = hy_upto(wi, ar_w), c;
		if (m >= b)
			goto no;
		ex = form_expr(wi, m);
		wi = m + 1;
		mat("?the|a");
		c = wi;
		if (mat("unit circle")) {
			Q.raw[0] = S("integrate %s around |z| = 1%s", ex,
			    isw(b - 1, "clockwise") ? " clockwise" : "");
			goto done;
		}
		if (mat("circle")) {
			/* the circle |z - 1| = 2, the circle of radius 2 about 1 */
			if (mat("of radius")) {
				int r0 = wi, r1 = hy_upto(wi, at_w);
				const char *rr = form_expr(r0, r1), *cc = "0";
				if (r1 < b)
					cc = form_expr(r1 + 1, b);
				Q.raw[0] = S("integrate %s around |z - (%s)| = %s", ex, cc,
				    rr);
				goto done;
			}
			c = wi;
		}
		Q.raw[0] = S("integrate %s around %s", ex, words_text(c, b));
		goto done;
	}
	/* rotations */
	wi = s;
	if (mat("rotate")) {
		int e = b;
		for (i = wi; i < b; i++)
			if (isw(i, "using") || (isw(i, "with") && (isw(i + 1, "a") ||
			    isw(i + 1, "quaternions") || isw(i + 1, "quaternion"))) ||
			    (isw(i, "by") && isw(i + 1, "quaternion") &&
			    isw(i + 2, "multiplication")) || isw(i, "via"))
				e = i;
		Q.raw[0] = S("rotate %s", words_text(wi, e));
		goto done;
	}
	wi = k;
	if (mat("rotation matrix of|for ?the ?quaternion")) {
		Q.raw[0] = S("rotmatrix %s", form_expr(wi, b));
		goto done;
	}
	wi = k;
	if (mat("?unit quaternion for|of ?a ?rotation ?by")) {
		static const char *const ab[] = {"about", "around", NULL};
		int m = hy_upto(wi, ab), a0 = wi;
		if (m >= b)
			goto no;
		wi = m + 1;
		mat("?the");
		if (W[wi].k == W_WORD && strlen(W[wi].s) == 1 && (isw(wi + 1, "axis") ||
		    wi + 1 == b))
			Q.raw[0] = S("axisangle (%s), %s", !strcmp(W[wi].s, "x") ?
			    "1, 0, 0" : !strcmp(W[wi].s, "y") ? "0, 1, 0" : "0, 0, 1",
			    words_text(a0, m));
		else
			Q.raw[0] = S("axisangle %s, %s", words_text(wi, b),
			    words_text(a0, m));
		goto done;
	}
	wi = k;
	if (mat("slerp|interpolate ?between|from")) {
		static const char *const and_w[] = {"and", "to", NULL};
		int m = hy_upto(wi, and_w), m2;
		if (m >= b)
			goto no;
		m2 = hy_upto(m + 1, at_w);
		if (m2 >= b)
			goto no;
		Q.raw[0] = S("slerp %s, %s, %s", form_expr(wi, m),
		    form_expr(m + 1, m2), form_expr(m2 + 1, b));
		goto done;
	}
	/* associator, commutator */
	wi = k;
	if (mat("associator|commutator of")) {
		int com = isw(wi - 2, "commutator");
		const char *t = words_text(wi, b);
		Q.raw[0] = S("%s %s", com ? "commutator" : "associator", t);
		/* "a, b and c" */
		{
			char *p = strstr(Q.raw[0], " and ");
			if (p) {
				char buf[600];
				snprintf(buf, sizeof buf, "%.*s, %s", (int)(p - Q.raw[0]),
				    Q.raw[0], p + 5);
				Q.raw[0] = S("%s", buf);
			}
		}
		goto done;
	}
	/* domain colouring */
	wi = k;
	if (mat("domain colouring|coloring|colourings|colorings ?plot ?of") ||
	    mat("colour|color wheel plot of")) {
		ex = form_expr(wi, b);
		if (!*ex)
			goto no;
		Q.raw[0] = S("plot %s", ex);
		goto done;
	}
no:
	wi = s;
	return 0;
done:
	Q.cmd = C_RAW;
	Q.nraw = 1;
	wi = nw;
	return 1;
}
