/*
 * English for the calc2 commands (calc2_syntax.inc), included by nlp.c
 * before sentence().  c2_sentence() reads the request into C2 and
 * c2_finish() writes the ucas lines:
 *
 *   differentiate F [twice | n times], the [second | nth | 3rd]
 *   derivative of F, d/dx F, d²/dx² F, d^2/dx^2 F, ∂/∂y F, ∂f/∂x,
 *   ∂²f/∂x∂y, the partial (derivative) of F with respect to y, fx, fxy,
 *   the gradient | hessian | laplacian of F [at (1, 2)], ∇F, ∇²F,
 *   the jacobian of F, G [wrt x, y], the directional derivative of F
 *   along | in the direction (3, 4) [at (1, 2)], the wronskian of f, g,
 *   are f and g linearly independent, the limit of F as x approaches |
 *   goes to | tends to | -> a [from the left | right], lim x->0+ F,
 *   what does F tend to as x goes to infinity, the taylor | maclaurin
 *   series | expansion | polynomial of F [at | about a] [to order n],
 *   dy/dx for | if L = R, differentiate L = R implicitly, the
 *   antiderivative | primitive | indefinite integral of F.
 */
static const char *target_fn(struct tgt *t, int mask, const char *defname,
    int inline_ok, int mask_first, int mask_all);
static int collect(int mask_first, int mask_all, struct var **ps);

/*
 * After an operand, a variable or a function name written after a space
 * multiplies: x^2 y, x sin x, x e^x (not dx, not a clause word, not a
 * name being given a value: "x y = 3").
 */
static int
c2_juxt(int i)
{
	static const char *const fns[] = {"sin", "cos", "tan", "exp", "log",
		"ln", "sqrt", "abs", "atan", "sec", "csc", "cot", NULL};

	if (W[i].k != W_WORD || clause_word(i))
		return 0;
	/* after a number, a symbol or a short name only: "blorp qq" is not */
	if (i > 0 && W[i - 1].k == W_WORD && strlen(W[i - 1].s) > 2 &&
	    !user_fn(W[i - 1].o))
		return 0;
	if (W[i].s[0] == 'd' && strlen(W[i].s) == 2)
		return 0;   /* dx */
	{
		/* plot's raw options (plot.c) are words, not factors */
		static const char *const popts[] = {"nofile", "field", "polar",
			"contour", "heatmap", "surface", "wireframe", "rotate",
			"slices", "parallel", "deriv", "tangent", "logx", "logy",
			"loglog", "frames", "fps", "points", "seconds", "animate",
			"to", "3d", "2d", NULL};
		if (in_list(popts, W[i].s))
			return 0;
	}
	if (W[i + 1].k == W_SYM && (!strcmp(W[i + 1].s, "=") ||
	    !strcmp(W[i + 1].s, ":")))
		return 0;
	if (in_list(fns, W[i].s) || W[i].prime)
		return 1;
	return varlike(i) && !isw(i + 1, "is") && !isw(i + 1, "are") &&
	    !isw(i + 1, "be") && !isw(i + 1, "and");
}

/* raw ucas command words of calc2 (nlp_is_english) */
static const char *const c2_cmds[] = {"partial", "grad", "gradient",
	"hessian", "laplacian", "jacobian", "ddir", "directional", "wronskian",
	"limit", "lim", "taylor", "series", "maclaurin", "implicit",
	"antiderivative", "primitive", NULL};

enum { K2_NONE, K2_DIFF, K2_GRAD, K2_HESS, K2_LAPL, K2_JAC, K2_DDIR,
	K2_WRON, K2_LIMIT, K2_TAYLOR, K2_IMPL, K2_ANTI, K2_INDEP };

static struct {
	int kind, order, nvars, side, nf, raw, symn, ordset;
	char vars[C2MAXO][32];
	char pt[96];           /* limit point, series centre */
	char at[160];          /* "1, 2" */
	char atname[MAXP][32]; /* at x = 1, y = 2: the names, in order */
	int natname;
	char dir[160];
	char lvar[32];         /* the limit's variable */
	struct tgt f[C2MAXF];
	int hasrhs;
	char dep[32], ind[32];
} C2;

/* a constant as ucas text: infinity -> inf; no variables are recorded */
static const char *
c2_const(void)
{
	int sm = cur_mask, neg = 0;
	struct ex e;

	if (mat("positive|plus ?infinity|inf|oo") && isw(wi - 1, "infinity"))
		return "inf";
	if (mat("negative|minus infinity|inf|oo") || (issym("-") &&
	    (isw(wi + 1, "infinity") || isw(wi + 1, "inf") ||
	    isw(wi + 1, "oo")) && (wi += 2)))
		return "-inf";
	if (issym("+") && (isw(wi + 1, "infinity") || isw(wi + 1, "inf")))
		wi++;
	if (mat("infinity|inf|oo|\xe2\x88\x9e"))
		return "inf";
	(void)neg;
	cur_mask = 0;
	e = parse_expr(P_ADD);
	cur_mask = sm;
	return e.s;
}

/* ordinals: first .. twelfth, 2nd, nth; 0 if none (n: C2.symn) */
static int
c2_ordinal(void)
{
	int k;

	if (W[wi].k == W_WORD && (k = lookup(ordw, W[wi].s)) >= 1) {
		wi++;
		return k;
	}
	if (W[wi].k == W_NUM && !W[wi + 1].sp && (isw(wi + 1, "th") ||
	    isw(wi + 1, "st") || isw(wi + 1, "nd") || isw(wi + 1, "rd"))) {
		k = atoi(W[wi].s);
		wi += 2;
		return k;
	}
	if (isw(wi, "nth") || isw(wi, "n-th") || (isw(wi, "n") &&
	    issym("-") && isw(wi + 2, "th"))) {
		wi += isw(wi, "n") ? 3 : 1;
		C2.symn = 1;
		return -1;
	}
	return 0;
}

/* "twice", "three times", "n times" after differentiate */
static void
c2_times(void)
{
	int n;

	if (mat("twice")) {
		C2.order = 2;
		return;
	}
	if (mat("thrice")) {
		C2.order = 3;
		return;
	}
	if (small_number(&n) && mat("times")) {
		C2.order = n;
		return;
	}
	if (isw(wi, "n") && isw(wi + 1, "times")) {
		wi += 2;
		C2.symn = 1;
	}
}

static void
c2_addvar(const char *v)
{
	if (C2.nvars < C2MAXO)
		snprintf(C2.vars[C2.nvars++], 32, "%s", v);
}

/* "at (1, 2)", "at x = 1, y = 2", "at x=1 and y=2", "at the origin" */
static int c2_at_value(void);

static int
c2_at(void)
{
	if (!mat("at|in"))
		return 0;
	return c2_at_value();
}

/* the point after "at" (or "f prime of") */
static int
c2_at_value(void)
{
	struct sbuf sb = {0};
	int n = 0, br;

	mat("?the ?point");
	if (mat("origin")) {
		snprintf(C2.at, sizeof C2.at, "origin");
		return 1;
	}
	br = acc_sym("(") || acc_sym("[");
	for (;;) {
		if (varlike(wi) && W[wi + 1].k == W_SYM && !strcmp(W[wi + 1].s,
		    "=")) {
			if (C2.natname < MAXP)
				snprintf(C2.atname[C2.natname++], 32, "%s", W[wi].o);
			wi += 2;
		}
		sb_printf(&sb, "%s%s", n++ ? ", " : "", c2_const());
		if (acc_sym(",") || (!br && mat("and") && varlike(wi)))
			continue;
		break;
	}
	if (br && !acc_sym(")"))
		acc_sym("]");
	snprintf(C2.at, sizeof C2.at, "%s", sb.p ? sb.p : "");
	free(sb.p);
	return 1;
}

/*
 * The function of a derivative: a defined function named before "at" is
 * itself ("the derivative of f at 3" is f'(3), not of f(3) = 27)
 */
static void
c2_fn_target(struct tgt *t)
{
	if (W[wi].k == W_WORD && user_fn(W[wi].o) && !var_find(W[wi].o) &&
	    isw(wi + 1, "at")) {
		named_target(t, W[wi].o);
		wi++;
		return;
	}
	target(t, P_TERN);
}

/* a vector: (3, 4), <3, 4>, 3i + 4j is not read */
static int
c2_vector(char *buf, size_t n)
{
	struct sbuf sb = {0};
	int k = 0, br;

	br = acc_sym("(") || acc_sym("[") || acc_sym("<");
	for (;;) {
		sb_printf(&sb, "%s%s", k++ ? ", " : "", c2_const());
		if (!acc_sym(","))
			break;
	}
	if (br && !acc_sym(")") && !acc_sym("]"))
		acc_sym(">");
	snprintf(buf, n, "%s", sb.p ? sb.p : "");
	free(sb.p);
	return k;
}

/* f, g and h: a list of targets */
static void
c2_list_targets(void)
{
	int br = acc_sym("(") || acc_sym("[") || acc_sym("{");

	C2.nf = 0;
	for (;;) {
		if (C2.nf == C2MAXF)
			nfail("at most %d functions", C2MAXF);
		cur_mask = 1;
		target(&C2.f[C2.nf++], P_BOR);
		if (acc_sym(","))
			mat("?and");
		else if (!mat("and"))
			break;
	}
	if (br && !acc_sym(")") && !acc_sym("]"))
		acc_sym("}");
}

/* "with respect to x and y", "wrt x, y", "in x" */
static void
c2_wrt(void)
{
	if (!(mat("with respect to") || mat("wrt") || mat("w . r . t ?.") ||
	    mat("by")))
		return;
	while (varlike(wi)) {
		c2_addvar(W[wi].o);
		wi++;
		if (!acc_sym(",") && !mat("and") && !mat("then"))
			break;
	}
}

/* d/dx, d²/dx², d^2/dx^2, d2/dx2: the order and the variable */
static int
c2_dnotation(void)
{
	int s = wi, ord = 1;

	if (!isw(wi, "d") && !(W[wi].k == W_SYM && !strcmp(W[wi].s, "\xe2\x88\x82")))
		return 0;
	wi++;
	if (acc_sym("^") && W[wi].k == W_NUM)
		ord = atoi(W[wi++].s);
	else if (acc_sym("\xc2\xb2"))
		ord = 2;
	else if (acc_sym("\xc2\xb3"))
		ord = 3;
	else if (W[wi].k == W_NUM && !W[wi].sp)
		ord = atoi(W[wi++].s);
	if (!acc_sym("/")) {
		wi = s;
		return 0;
	}
	/* dx, d x, dx^2, dx², ∂x */
	if (W[wi].k == W_WORD && W[wi].s[0] == 'd' && strlen(W[wi].s) >= 2 &&
	    strlen(W[wi].s) <= 3 && W[wi].s[1] != 'e') {
		char v[8];
		snprintf(v, sizeof v, "%s", W[wi].o + 1);
		/* dx2 */
		if (isdigit((unsigned char)v[strlen(v) - 1]))
			v[strlen(v) - 1] = 0;
		wi++;
		c2_addvar(v);
	} else if ((isw(wi, "d") || (W[wi].k == W_SYM &&
	    !strcmp(W[wi].s, "\xe2\x88\x82"))) && varlike(wi + 1)) {
		c2_addvar(W[wi + 1].o);
		wi += 2;
	} else {
		wi = s;
		return 0;
	}
	if (acc_sym("^") && W[wi].k == W_NUM)
		wi++;
	else
		acc_sym("\xc2\xb2") || acc_sym("\xc2\xb3");
	C2.order = ord;
	return 1;
}

/* ∂f/∂x, ∂²f/∂x∂y, ∂^2 f/∂x^2, df/dx (f a function name) */
static int
c2_dfrac(void)
{
	int s = wi, ord = 1, k;
	const char *fn;

	if (!(W[wi].k == W_SYM && !strcmp(W[wi].s, "\xe2\x88\x82")) &&
	    !isw(wi, "d"))
		return 0;
	wi++;
	if (acc_sym("^") && W[wi].k == W_NUM)
		ord = atoi(W[wi++].s);
	else if (acc_sym("\xc2\xb2"))
		ord = 2;
	else if (acc_sym("\xc2\xb3"))
		ord = 3;
	if (W[wi].k != W_WORD || !user_fn(W[wi].o)) {
		/* "dy/dx" for an implicit curve: handled by the caller */
		wi = s;
		return 0;
	}
	fn = W[wi].o;
	wi++;
	if (!acc_sym("/")) {
		wi = s;
		return 0;
	}
	for (k = 0; k < ord;) {
		int e = 1;
		if (!(W[wi].k == W_SYM && !strcmp(W[wi].s, "\xe2\x88\x82")) &&
		    !isw(wi, "d"))
			break;
		wi++;
		if (!varlike(wi)) {
			wi = s;
			return 0;
		}
		{
			const char *v = W[wi].o;
			wi++;
			if (acc_sym("^") && W[wi].k == W_NUM)
				e = atoi(W[wi++].s);
			else if (acc_sym("\xc2\xb2"))
				e = 2;
			while (e-- > 0 && k < ord) {
				c2_addvar(v);
				k++;
			}
		}
	}
	if (k == 0) {
		wi = s;
		return 0;
	}
	/* ∂²f/∂x∂y differentiates by y, then x (written right to left) */
	for (k = 0; k < C2.nvars / 2; k++) {
		char t[32];
		memcpy(t, C2.vars[k], 32);
		memcpy(C2.vars[k], C2.vars[C2.nvars - 1 - k], 32);
		memcpy(C2.vars[C2.nvars - 1 - k], t, 32);
	}
	named_target(&C2.f[0], fn);
	C2.order = 0;
	return 1;
}

/* fx, fxy, f_x, f_xy for a defined f whose parameters these are */
static int
c2_subscript(void)
{
	struct func *f;
	int i, k, n;
	char w[40];
	const char *suf;

	if (W[wi].k != W_WORD)
		return 0;
	snprintf(w, sizeof w, "%s", W[wi].o);
	for (i = (int)strlen(w) - 1; i > 0; i--) {
		char base[40];
		memcpy(base, w, (size_t)i);
		base[i] = 0;
		if (base[i - 1] == '_')
			base[i - 1] = 0;
		if ((f = user_fn(base)) == NULL || f->np < 1)
			continue;
		suf = w + i;
		n = (int)strlen(suf);
		/* every letter a one-letter parameter of f */
		for (k = 0; k < n; k++) {
			int j;
			for (j = 0; j < f->np; j++)
				if (strlen(sym_str(f->p[j].name)) == 1 &&
				    sym_str(f->p[j].name)[0] == suf[k])
					break;
			if (j == f->np)
				break;
		}
		if (k < n || n > C2MAXO)
			continue;
		for (k = 0; k < n; k++) {
			char v[2] = {suf[k], 0};
			c2_addvar(v);
		}
		wi++;
		named_target(&C2.f[0], base);
		return 1;
	}
	return 0;
}

/* x approaches|goes to|tends to|-> a [from the left|right|above|below] */
static int
c2_approach(void)
{
	int s = wi;

	if (!varlike(wi))
		return 0;
	snprintf(C2.lvar, sizeof C2.lvar, "%s", W[wi].o);
	wi++;
	if (!(mat("approaches|approaching|approach|nears") ||
	    mat("goes|going|go|tends|tending|tend|heads|heading|grows|growing|"
	    "increases|decreases ?to|towards|toward|without") ||
	    (issym("-") && W[wi + 1].k == W_SYM && !strcmp(W[wi + 1].s, ">") &&
	    (wi += 2)) || acc_sym("\xe2\x86\x92") || mat("to"))) {
		wi = s;
		return 0;
	}
	if (isw(wi - 1, "without"))
		mat("bound");
	if (isw(wi - 1, "grows") || isw(wi - 1, "growing") ||
	    isw(wi - 1, "increases") || isw(wi - 1, "bound"))
		snprintf(C2.pt, sizeof C2.pt, "inf");
	else if (isw(wi - 1, "decreases") && !peek("to|towards"))
		snprintf(C2.pt, sizeof C2.pt, "-inf");
	else
		snprintf(C2.pt, sizeof C2.pt, "%s", c2_const());
	/* 0+, 0^+, 0 from the right */
	if (W[wi].k == W_SYM && (!strcmp(W[wi].s, "+") || !strcmp(W[wi].s, "-"))
	    && !W[wi].sp && (W[wi + 1].k == W_END || clause_word(wi + 1) ||
	    W[wi + 1].sp)) {
		C2.side = !strcmp(W[wi].s, "+") ? 1 : -1;
		wi++;
	}
	if (mat("from|on ?the left|below|right|above ?side|hand") ||
	    mat("from the negative|positive side")) {
		int i;
		for (i = s; i < wi; i++) {
			if (isw(i, "left") || isw(i, "below") || isw(i, "negative"))
				C2.side = -1;
			if (isw(i, "right") || isw(i, "above") || isw(i, "positive"))
				C2.side = 1;
		}
	}
	return 1;
}

/* the limit's opening: lim, lim x->0, lim_{x->0}, lim(x->0) */
static int
c2_lim_head(void)
{
	int br;

	br = acc_sym("{") || acc_sym("(");
	if (!c2_approach())
		return 0;
	if (br)
		acc_sym("}") || acc_sym(")");
	return 1;
}

/* the sentence; 0 when it is not one of these */
static int
c2_sentence(void)
{
	int s = wi, k;

	memset(&C2, 0, sizeof C2);
	mat("?what|whats|which ?is|are");
	mat("?compute|calculate|find|get|determine|evaluate|take|give|show");
	mat("?me|us");
	mat("?the|a|an");
	k = wi;

	/* "f prime of 3", "f double prime", "f prime at x = 2": f'(3) */
	if (W[wi].k == W_WORD && user_fn(W[wi].o) && !var_find(W[wi].o) &&
	    (isw(wi + 1, "prime") || (isw(wi + 1, "double") &&
	    isw(wi + 2, "prime")) || (isw(wi + 1, "triple") &&
	    isw(wi + 2, "prime")))) {
		named_target(&C2.f[0], W[wi].o);
		C2.order = isw(wi + 1, "double") ? 2 : isw(wi + 1, "triple") ? 3 : 1;
		wi += C2.order > 1 ? 3 : 2;
		C2.kind = K2_DIFF;
		if (mat("of") && !c2_at_value())
			nfail("%s prime of what?", C2.f[0].fn);
		goto done;
	}
	wi = k;

	/* the range: "what values does F take"), "the set of values of F" */
	wi = s;
	if (mat("what|which values|outputs does|do|can|will") ||
	    mat("what values|outputs can") ) {
		mat("?the ?function");
		cur_mask = 1;
		target(&Q.a, P_TERN);
		if (mat("take|takes|attain|attains|have|has|output|outputs|produce|"
		    "produces|reach|reaches|give|gives|return|returns") ||
		    mat("take on")) {
			mat("?on");
			Q.cmd = C_RANGE;
			return 1;
		}
		memset(&Q.a, 0, sizeof Q.a);
	}
	wi = k;
	if (mat("set|range of ?all ?possible values|outputs ?of|for") ||
	    mat("values taken|attained by")) {
		cur_mask = 1;
		target(&Q.a, P_TERN);
		Q.cmd = C_RANGE;
		return 1;
	}
	wi = k;
	/* antiderivatives (with bounds, integrate takes over) */
	if (mat("antiderivative|antiderivatives|anti-derivative|"
	    "anti-derivatives|primitive|primitives|antidifferential") ||
	    mat("anti derivative|derivatives") ||
	    mat("indefinite integral|integrals")) {
		mat("?of|for");
		mat("?the ?function");
		cur_mask = 1;
		target(&Q.a, P_TERN);
		Q.cmd = C_INTEG;
		C2.kind = K2_ANTI;
		return 1;
	}
	wi = k;
	/* the wronskian; linear independence */
	if (mat("wronskian|wronskians ?matrix|determinant ?of|for") ||
	    mat("wronski matrix|determinant ?of|for")) {
		c2_list_targets();
		C2.kind = K2_WRON;
		goto done;
	}
	wi = s;
	if (mat("are|is ?the ?functions") || mat("check ?whether|if ?the "
	    "?functions") || mat("determine whether|if ?the ?functions")) {
		int s2 = wi;
		quiet++;
		{
			jmp_buf sv;
			int ok = 0;
			memcpy(sv, nl_jmp, sizeof sv);
			if (setjmp(nl_jmp) == 0) {
				c2_list_targets();
				ok = C2.nf >= 2 && (mat("?linearly independent|dependent") &&
				    (isw(wi - 1, "independent") || isw(wi - 1,
				    "dependent")));
			}
			memcpy(nl_jmp, sv, sizeof sv);
			quiet--;
			if (ok) {
				wi = s2;
				memset(&C2, 0, sizeof C2);
				c2_list_targets();
				mat("linearly independent|dependent");
				C2.kind = K2_WRON;
				goto done;
			}
		}
		wi = s;
	}
	wi = k;
	if (mat("linear independence|dependence ?of") ||
	    mat("linearly independent|dependent ?of")) {
		c2_list_targets();
		C2.kind = K2_WRON;
		goto done;
	}
	wi = k;
	/* gradient, hessian, laplacian, jacobian */
	if (mat("gradient|grad ?vector ?of|for") ||
	    mat("\xe2\x88\x87 ?of")) {
		if (issym("\xc2\xb2")) {
			wi++;
			C2.kind = K2_LAPL;
		} else
			C2.kind = K2_GRAD;
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		goto done;
	}
	if (mat("hessian ?matrix ?of|for") || mat("matrix of second "
	    "?partial derivatives ?of")) {
		C2.kind = K2_HESS;
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		goto done;
	}
	if (mat("laplacian|laplace ?operator ?of|for") ||
	    mat("del squared ?of") || mat("divergence of the gradient of")) {
		C2.kind = K2_LAPL;
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		goto done;
	}
	if (mat("jacobian ?matrix|determinant ?of|for")) {
		C2.kind = K2_JAC;
		c2_list_targets();
		goto done;
	}
	/* directional derivative */
	if (mat("directional derivative|derivatives ?of|for")) {
		C2.kind = K2_DDIR;
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		if (mat("along|toward|towards") || mat("in the direction ?of") ||
		    mat("in direction ?of") || mat("with direction"))
			c2_vector(C2.dir, sizeof C2.dir);
		else
			nfail("in which direction? (along (3, 4))");
		goto done;
	}
	/* limits */
	if (mat("limit|limits|lim ?value ?of") ||
	    mat("left|right|left-hand|right-hand|one-sided|two-sided ?hand "
	    "limit ?of")) {
		int i;
		for (i = k; i < wi; i++) {
			if (isw(i, "left") || isw(i, "left-hand"))
				C2.side = -1;
			if (isw(i, "right") || isw(i, "right-hand"))
				C2.side = 1;
		}
		C2.kind = K2_LIMIT;
		/* lim x->0 F, the limit as x goes to 0 of F */
		{
			int s3 = wi;
			mat("?as|when");
			if (c2_lim_head()) {
				int sd = C2.side;
				mat("?of");
				mat("?the ?function");
				cur_mask = 1;
				target(&C2.f[0], P_TERN);
				C2.side = C2.side ? C2.side : sd;
				goto done;
			}
			wi = s3;
		}
		mat("?of");
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		mat("?as|when|for|while");
		if (!c2_approach()) {
			if (mat("at")) {
				snprintf(C2.pt, sizeof C2.pt, "%s", c2_const());
				C2.lvar[0] = 0;
			} else
				nfail("as what approaches what? (as x approaches 0)");
		}
		goto done;
	}
	wi = s;
	/* what does F tend to / approach as x goes to inf */
	if (mat("what|where does|do|will")) {
		int s3 = wi;
		quiet++;
		{
			jmp_buf sv;
			int ok = 0;
			memcpy(sv, nl_jmp, sizeof sv);
			if (setjmp(nl_jmp) == 0) {
				struct tgt t;
				cur_mask = 1;
				target(&t, P_TERN);
				ok = mat("tend|approach|converge|go|head ?to|towards|toward");
			}
			memcpy(nl_jmp, sv, sizeof sv);
			quiet--;
			if (ok) {
				wi = s3;
				cur_mask = 1;
				target(&C2.f[0], P_TERN);
				mat("tend|approach|converge|go|head ?to|towards|toward");
				mat("?as|when|for");
				if (!c2_approach())
					nfail("as what approaches what?");
				C2.kind = K2_LIMIT;
				goto done;
			}
		}
		wi = s;
	}
	wi = k;
	/* Taylor and Maclaurin series */
	if (mat("taylor series|expansion|expansions|polynomial|polynomials "
	    "?of|for") || mat("maclaurin ?series|expansion|expansions|polynomial|"
	    "polynomials ?of|for") || mat("power series ?of|for") ||
	    mat("series expansion|expansions ?of|for")) {
		int mac = 0, i;
		for (i = k; i < wi; i++)
			if (isw(i, "maclaurin"))
				mac = 1;
		C2.kind = K2_TAYLOR;
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		for (;;) {
			int n;
			if (mat("at|about|around|centered|centred ?at|on")) {
				if (varlike(wi) && W[wi + 1].k == W_SYM &&
				    !strcmp(W[wi + 1].s, "=")) {
					c2_addvar(W[wi].o);
					wi += 2;
				}
				snprintf(C2.pt, sizeof C2.pt, "%s", c2_const());
			} else if (mat("?up to|through|of ?order|degree") &&
			    small_number(&n)) {
				C2.order = n;
				C2.ordset = 1;
				mat("?terms|term");
			} else if (mat("to|up to|through|with") && small_number(&n) &&
			    mat("terms|term")) {
				C2.order = n - 1;
				C2.ordset = 1;
			} else if ((n = c2_ordinal()) > 0 && mat("order|degree")) {
				C2.order = n;
				C2.ordset = 1;
			} else
				break;
		}
		if (mac && C2.pt[0] && strcmp(C2.pt, "0"))
			nfail("a Maclaurin series is at 0; say \"taylor series at %s\"",
			    C2.pt);
		goto done;
	}
	wi = s;
	if (mat("expand") ) {
		cur_mask = 1;
		target(&C2.f[0], P_TERN);
		if (mat("in|as ?a|its taylor|maclaurin|power series") ||
		    mat("into ?a taylor|maclaurin|power series")) {
			C2.kind = K2_TAYLOR;
			for (;;) {
				int n;
				if (mat("at|about|around"))
					snprintf(C2.pt, sizeof C2.pt, "%s", c2_const());
				else if (mat("?up to|through ?order|degree") &&
				    small_number(&n)) {
					C2.order = n;
					C2.ordset = 1;
				} else
					break;
			}
			goto done;
		}
		wi = s;
		memset(&C2, 0, sizeof C2);
	}
	wi = s;
	/* implicit differentiation: dy/dx for x^2 + y^2 = 1 */
	{
		int s3;
		mat("?what|whats ?is");
		mat("?find|compute|calculate|get");
		s3 = wi;
		if (W[wi].k == W_WORD) {
			/* dy/dx as one word "dy" "/" "dx" */
			if (W[wi].k == W_WORD && W[wi].s[0] == 'd' &&
			    strlen(W[wi].s) == 2 && W[wi + 1].k == W_SYM &&
			    !strcmp(W[wi + 1].s, "/") && W[wi + 2].k == W_WORD &&
			    W[wi + 2].s[0] == 'd' && strlen(W[wi + 2].s) == 2 &&
			    !user_fn(W[wi].o + 1)) {
				snprintf(C2.dep, sizeof C2.dep, "%s", W[wi].o + 1);
				snprintf(C2.ind, sizeof C2.ind, "%s", W[wi + 2].o + 1);
				wi += 3;
				if (mat("for|of|if|when|given|from|on ?the ?curve|"
				    "equation") || mat("where") || acc_sym(":") ||
				    acc_sym(",")) {
					C2.kind = K2_IMPL;
					C2.order = 1;
					goto implicit_eq;
				}
			}
			wi = s3;
		}
		if (mat("differentiate|derive") || mat("implicit "
		    "differentiation|derivative ?of|for") || mat("implicitly "
		    "differentiate")) {
			int s4 = wi, ok = 0;
			if (isw(s4 - 1, "differentiation") || isw(s4 - 1, "derivative") ||
			    isw(s4 - 1, "of") || isw(s4 - 1, "for") ||
			    isw(s4 - 2, "implicitly"))
				ok = 1;
			/* differentiate L = R [implicitly]: an equation */
			{
				int i, eq = 0;
				for (i = wi; W[i].k != W_END; i++)
					if (W[i].k == W_SYM && !strcmp(W[i].s, "="))
						eq = 1;
				if (eq && (ok || 1)) {
					C2.kind = K2_IMPL;
					C2.order = 1;
					goto implicit_eq;
				}
			}
			wi = s;
		}
	}
	wi = s;
	if (0) {
	implicit_eq:
		cur_mask = 1;
		target(&C2.f[0], P_BOR);
		if (!acc_sym("=") && !mat("equals|is equal to"))
			nfail("implicit differentiation needs an equation, L = R");
		cur_mask = 1;
		target(&C2.f[1], P_BOR);
		C2.hasrhs = 1;
		mat("?implicitly");
		for (;;) {
			if (mat("for") && varlike(wi)) {
				snprintf(C2.dep, sizeof C2.dep, "%s", W[wi].o);
				wi++;
			} else if ((mat("with respect to") || mat("wrt") ||
			    mat("in terms of")) &&
			    varlike(wi)) {
				snprintf(C2.ind, sizeof C2.ind, "%s", W[wi].o);
				wi++;
			} else if (mat("?and ?the second derivative|order"))
				C2.order = 2;
			else
				break;
		}
		goto done;
	}
	wi = k;
	/* derivatives: d/dx F, ∂f/∂x, fx, the second partial derivative of */
	if (c2_dfrac() || c2_subscript()) {
		C2.kind = K2_DIFF;
		goto done;
	}
	if (c2_dnotation()) {
		mat("?of");
		cur_mask = 1;
		c2_fn_target(&C2.f[0]);
		C2.kind = K2_DIFF;
		goto done;
	}
	{
		int ord = 0, part = 0, mixed = 0;
		if ((ord = c2_ordinal()) != 0)
			mat("?order");
		if (mat("mixed"))
			mixed = 1;
		if (mat("partial|partials"))
			part = 1;
		if (mat("derivative|derivatives|derivate|derivates|derivitive|"
		    "deriv") || part || mat("?instantaneous rate of change") ||
		    (isw(wi, "slope") && isw(wi + 1, "of") && ++wi)) {
			C2.kind = K2_DIFF;
			if (ord > 0)
				C2.order = ord;
			mat("?of|for");
			mat("?the ?function");
			cur_mask = 1;
			c2_fn_target(&C2.f[0]);
			c2_wrt();
			if (mixed && C2.nvars < 2)
				nfail("mixed partials need two variables: with respect "
				    "to x and y");
			(void)part;
			goto done;
		}
		wi = k;
		C2.symn = 0;
		C2.order = 0;
	}
	wi = s;
	if (mat("differentiate|derive")) {
		mat("?the ?function");
		cur_mask = 1;
		c2_fn_target(&C2.f[0]);
		c2_times();
		c2_wrt();
		c2_times();
		C2.kind = K2_DIFF;
		goto done;
	}
	wi = s;
	memset(&C2, 0, sizeof C2);
	return 0;
done:
	/* options after the function */
	for (;;) {
		if (C2.kind != K2_LIMIT && C2.kind != K2_TAYLOR && c2_at())
			continue;
		if (C2.kind == K2_DIFF || C2.kind == K2_JAC || C2.kind == K2_WRON ||
		    C2.kind == K2_TAYLOR) {
			int n0 = C2.nvars;
			c2_wrt();
			if (C2.nvars > n0)
				continue;
		}
		if (C2.kind == K2_DIFF) {
			int n0 = C2.order, s5 = wi;
			c2_times();
			if (C2.order != n0 || C2.symn)
				if (wi != s5)
					continue;
		}
		if (C2.kind == K2_DDIR && !C2.dir[0] && (mat("along|toward|towards")
		    || mat("in the direction ?of"))) {
			c2_vector(C2.dir, sizeof C2.dir);
			continue;
		}
		if (C2.kind == K2_DDIR && mat("unnormalized|unnormalised|raw") ) {
			C2.raw = 1;
			continue;
		}
		break;
	}
	Q.cmd = C_RAW;
	return 1;
}

/* "1, 2" ordered by the function's parameters when they were named */
static const char *
c2_at_text(struct func *f)
{
	struct sbuf sb = {0};
	char vals[MAXP][64];
	int n = 0, k, j;
	const char *p = C2.at, *r;

	if (!strcmp(C2.at, "origin")) {
		int np = f ? f->np : 1;
		for (k = 0; k < np; k++)
			sb_printf(&sb, "%s0", k ? ", " : "");
		r = S("%s", sb.p);
		free(sb.p);
		return r;
	}
	if (!C2.natname || !f)
		return C2.at;
	while (*p && n < MAXP) {
		size_t L = strcspn(p, ",");
		snprintf(vals[n], 64, "%.*s", (int)L, p);
		n++;
		p += L;
		while (*p == ',' || *p == ' ')
			p++;
	}
	for (k = 0; k < f->np; k++) {
		for (j = 0; j < C2.natname && j < n; j++)
			if (!strcmp(C2.atname[j], sym_str(f->p[k].name)))
				break;
		if (j == C2.natname || j >= n)
			nfail("the point has no value for %s", sym_str(f->p[k].name));
		sb_printf(&sb, "%s%s", k ? ", " : "", vals[j]);
	}
	r = S("%s", sb.p);
	free(sb.p);
	return r;
}

/* the parameters of the function the target becomes, by name */
static struct func *
c2_target_params(struct tgt *t, struct func *tmp)
{
	struct var *ps[MAXP];
	struct func *f;
	int np, k;

	if (t->named && (f = user_fn(t->fn)) != NULL)
		return f;
	memset(tmp, 0, sizeof *tmp);
	np = collect(1, 1, ps);
	for (k = 0; k < np; k++)
		tmp->p[k].name = sym_cstr(ps[k]->name);
	tmp->np = np;
	return tmp;
}

static int
c2_finish(void)
{
	struct sbuf sb = {0};
	struct func tmp, *pf;
	const char *F;
	int k;

	if (C2.kind == K2_NONE)
		return 0;
	if (C2.kind == K2_ANTI) {
		/* bounds: the definite integral (integrate) */
		int bounds = Q.ilo != NULL;
		for (k = 0; k < nv; k++)
			if (V[k].order && (V[k].uses & 1) && V[k].lo[0] && V[k].hi[0] &&
			    !V[k].ty[0])
				bounds = 1;
		if (bounds) {
			if (!dry)
				printf("  note: an antiderivative is indefinite; with "
				    "bounds this is the definite integral\n");
			return 0;
		}
		F = target_fn(&Q.a, 1, "it", 0, 1, 1);
		sb_printf(&sb, "antiderivative %s", F);
		if (Q.wrt[0])
			sb_printf(&sb, " wrt %s", Q.wrt);
		emit(sb.p);
		free(sb.p);
		return 1;
	}
	if (Q.wrt[0] && C2.nvars == 0)
		c2_addvar(Q.wrt);
	if (C2.kind == K2_WRON || C2.kind == K2_JAC) {
		if (C2.nf < (C2.kind == K2_WRON ? 2 : 1))
			nfail("%s needs at least two functions", C2.kind == K2_WRON ?
			    "the Wronskian" : "the Jacobian");
		sb_printf(&sb, "%s ", C2.kind == K2_WRON ? "wronskian" : "jacobian");
		for (k = 0; k < C2.nf; k++) {
			const char *e;
			if (C2.f[k].named) {
				struct func *g = user_fn(C2.f[k].fn);
				int j;
				struct sbuf c = {0};
				sb_printf(&c, "%s(", C2.f[k].fn);
				for (j = 0; g && j < g->np; j++)
					sb_printf(&c, "%s%s", j ? ", " : "",
					    sym_str(g->p[j].name));
				sb_putc(&c, ')');
				e = S("%s", c.p);
				free(c.p);
			} else
				e = C2.f[k].e.s;
			sb_printf(&sb, "%s%s", k ? ", " : "", e);
		}
		if (C2.nvars) {
			sb_puts(&sb, " wrt ");
			for (k = 0; k < C2.nvars; k++)
				sb_printf(&sb, "%s%s", k ? ", " : "", C2.vars[k]);
		}
		if (C2.at[0] && C2.kind == K2_JAC)
			sb_printf(&sb, " at %s", C2.at);
		emit(sb.p);
		free(sb.p);
		return 1;
	}
	if (C2.kind == K2_IMPL) {
		const char *l = C2.f[0].named ? C2.f[0].fn : C2.f[0].e.s;
		const char *r = C2.f[1].named ? C2.f[1].fn : C2.f[1].e.s;
		sb_printf(&sb, "implicit %s = %s", l, r);
		if (C2.dep[0])
			sb_printf(&sb, " for %s", C2.dep);
		if (C2.ind[0])
			sb_printf(&sb, " wrt %s", C2.ind);
		if (C2.order == 2)
			sb_printf(&sb, " order 2");
		emit(sb.p);
		free(sb.p);
		return 1;
	}
	pf = NULL;
	F = target_fn(&C2.f[0], 1, "it", 0, 1, 1);
	pf = c2_target_params(&C2.f[0], &tmp);
	switch (C2.kind) {
	case K2_DIFF:
		sb_printf(&sb, "diff %s", F);
		for (k = 0; k < C2.nvars; k++)
			sb_printf(&sb, " %s", C2.vars[k]);
		if (C2.symn) {
			if (pf->np > 1 && C2.nvars == 0)
				nfail("with respect to which variable?");
			sb_puts(&sb, " n");
		} else if (C2.order > 1 && C2.nvars <= 1) {
			if (C2.nvars == 0 && pf->np > 1) {
				const char *o[MAXP];
				int j, c;
				for (j = 0; j < pf->np; j++)
					o[j] = sym_str(pf->p[j].name);
				c = clarify(NULL, S("the %d%s derivative of %s with respect "
				    "to which variable", C2.order, C2.order == 2 ? "nd" :
				    C2.order == 3 ? "rd" : "th", F), o, pf->np, 0,
				    "say \"with respect to y\"");
				sb_printf(&sb, " %s", o[c]);
			}
			sb_printf(&sb, " %d", C2.order);
		}
		if (C2.at[0] && pf->np == 1 && !C2.symn && C2.nvars <= 1) {
			/* the value at the point: it'(3) */
			int o = C2.order > 1 ? C2.order : 1;
			sb_clear(&sb);
			sb_printf(&sb, "%s%.*s(%s)", F, o, "''''''''''''",
			    c2_at_text(pf));
		} else if (C2.at[0] && !dry)
			printf("  note: the derivative is printed as a formula (the "
			    "value at a point needs one variable: F'(a))\n");
		break;
	case K2_GRAD:
	case K2_HESS:
	case K2_LAPL:
		sb_printf(&sb, "%s %s", C2.kind == K2_GRAD ? "grad" :
		    C2.kind == K2_HESS ? "hessian" : "laplacian", F);
		if (C2.at[0])
			sb_printf(&sb, " at %s", c2_at_text(pf));
		break;
	case K2_DDIR:
		sb_printf(&sb, "ddir %s along %s", F, C2.dir);
		if (C2.at[0])
			sb_printf(&sb, " at %s", c2_at_text(pf));
		if (C2.raw)
			sb_puts(&sb, " raw");
		break;
	case K2_LIMIT:
		if (!C2.lvar[0]) {
			if (pf->np != 1)
				nfail("as which variable approaches %s?", C2.pt);
			snprintf(C2.lvar, sizeof C2.lvar, "%s",
			    sym_str(pf->p[0].name));
		}
		sb_printf(&sb, "limit %s %s -> %s", F, C2.lvar, C2.pt);
		if (C2.side)
			sb_printf(&sb, " from %s", C2.side > 0 ? "right" : "left");
		break;
	case K2_TAYLOR:
		sb_printf(&sb, "taylor %s", F);
		if (C2.nvars)
			sb_printf(&sb, " %s", C2.vars[0]);
		else if (pf->np > 1)
			nfail("a series in which variable? (with respect to x)");
		if (C2.pt[0] && strcmp(C2.pt, "0"))
			sb_printf(&sb, " at %s", C2.pt);
		if (C2.ordset)
			sb_printf(&sb, " order %d", C2.order);
		else if (!dry)
			printf("  note: no order given: through order 5 (say \"to "
			    "order 8\")\n");
		break;
	}
	emit(sb.p);
	free(sb.p);
	return 1;
}
