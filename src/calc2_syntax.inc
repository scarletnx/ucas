/*
 * calc2 commands (calc2.c), included at the end of parse.c:
 *
 *   diff F [x y ...] [n]        derivatives: d/dx, d²/dx², mixed ∂²/∂y∂x;
 *   diff F wrt x [, y]          no variable for several: the gradient;
 *   partial F [x ...]           a symbolic order n gives the n-th
 *   diff F x n                  derivative of exp/sin/cos/log/powers
 *   F''  F'(2)                  a defined function's derivatives
 *   grad F [at a, b]            gradient (also gradient), hessian,
 *   hessian F [at a, b]         laplacian; at a point: values, and for
 *   laplacian F [at a, b]       the Hessian the second-derivative test
 *   jacobian F1, F2 [wrt x, y] [at ...]
 *   ddir F along v1, v2 [at ...] [raw]    directional derivative
 *   wronskian F1, F2, ... [wrt x]
 *   limit F [as] x -> a[+|-] [from left|right]   (also lim; a may be inf)
 *   lim x->a[+|-] F,  lim(x -> a) F,  limit F at a
 *   taylor F [x] [at a] [order n]   (also series; maclaurin F [order n])
 *   implicit L [= R] [for y] [wrt x] [order 2]
 *   range of F                  "of" is allowed after range
 *
 * A function and the variables after it are told apart by spacing: the
 * trailing names written with a space before them, that are not
 * function names, and that the function's text does not end in an
 * operator before.  `diff x^2*y x` is d/dx of x^2*y.
 */
#include "calc2.h"

static int
c2_is_word(struct P *p, uint32_t i, const char *w)
{
	return i < p->t->n && tok_is(p->t, i, w);
}

/* the first T_NL/T_EOF from i */
static uint32_t
c2_eos(struct P *p, uint32_t i)
{
	while (i < p->t->n && p->t->kind[i] != T_NL && p->t->kind[i] != T_EOF)
		i++;
	return i;
}

/* the first word of list[] at paren depth 0 in [i, end), else end */
static uint32_t
c2_find_kw(struct P *p, uint32_t i, uint32_t end, const char *const *list)
{
	int depth = 0, k;

	for (; i < end; i++) {
		int t = p->t->kind[i];
		if (t == T_LP || t == T_LB || t == T_LC)
			depth++;
		else if (t == T_RP || t == T_RB || t == T_RC)
			depth--;
		else if (depth == 0 && t == T_ID)
			for (k = 0; list[k]; k++)
				if (tok_is(p->t, i, list[k]))
					return i;
	}
	return end;
}

/* parse a command function from p->i that must end exactly at end */
static struct func *
c2_func_upto(struct P *p, uint32_t end)
{
	uint8_t saved;
	struct func *f;

	if (p->i >= end)
		fail("expected a function");
	saved = p->t->kind[end];
	p->t->kind[end] = T_NL;
	f = cmd_func(p);
	p->t->kind[end] = saved;
	if (p->i != end) {
		diag_tok = p->i;
		fail("unexpected '%.*s' in the function", (int)p->t->len[p->i],
		    p->t->src + p->t->off[p->i]);
	}
	return f;
}

/* a constant in [i, end): inf, -inf, oo, infinity, pi/2 ... */
static double
c2_const_upto(struct P *p, uint32_t end)
{
	uint8_t saved;
	nid v;
	double d = 0;
	int neg = 0;

	if (p->i >= end)
		fail("expected a number");
	if (pk(p) == T_PLUS && p->i + 1 < end)
		adv(p);
	if (pk(p) == T_MINUS && p->i + 2 == end && (c2_is_word(p, p->i + 1,
	    "oo") || c2_is_word(p, p->i + 1, "infinity") ||
	    c2_is_word(p, p->i + 1, "inf"))) {
		adv(p);
		neg = 1;
	}
	if (p->i + 1 == end && (is_kw(p, "oo") || is_kw(p, "infinity") ||
	    is_kw(p, "inf") || tok_is(p->t, p->i, "\xe2\x88\x9e"))) {
		adv(p);
		return neg ? -INFINITY : INFINITY;
	}
	saved = p->t->kind[end];
	p->t->kind[end] = T_NL;
	v = cs_const_expr(p);
	p->t->kind[end] = saved;
	if (p->i != end)
		fail("expected a number");
	num_of(v, &d);
	return neg ? -d : d;
}

/* the slot of a parameter name of f, else -1 */
static int
c2_slot(struct func *f, uint32_t name)
{
	int k;

	for (k = 0; k < f->np; k++)
		if (f->p[k].name == name)
			return k;
	return -1;
}

static int
c2_slot_or_fail(struct func *f, uint32_t name)
{
	int k = c2_slot(f, name);

	if (k < 0)
		fail("%s has no variable %s", f->anon ? node_str(f->body) :
		    sym_str(f->name), sym_str(name));
	return k;
}

/* could the token end an expression (so what follows can be a variable)? */
static int
c2_ends_expr(struct P *p, uint32_t i)
{
	int t = p->t->kind[i];

	if (t == T_ID)
		return builtin(p, i) == NULL || builtin(p, i)->arity != 1 ||
		    (i + 1 < p->t->n && p->t->kind[i + 1] == T_LP);
	return t == T_NUM || t == T_RP || t == T_RB || t == T_SUP ||
	    t == T_BAR;
}

/*
 * The trailing variable list in [s, end): names with a space before,
 * not functions, after a token that can end an expression.  Returns
 * the start of the list (end if none).
 */
static uint32_t
c2_tail(struct P *p, uint32_t s, uint32_t end, int allow_num)
{
	uint32_t j = end;

	if (allow_num && j > s + 1 && p->t->kind[j - 1] == T_NUM &&
	    (p->t->fl[j - 1] & TF_SPACE) && c2_ends_expr(p, j - 2))
		j--;
	while (j > s + 1 && p->t->kind[j - 1] == T_ID &&
	    (p->t->fl[j - 1] & TF_SPACE) && builtin(p, j - 1) == NULL &&
	    func_find(tsym(p, j - 1)) == NULL && c2_ends_expr(p, j - 2) &&
	    !word_in(p, j - 1, keywords))
		j--;
	/* x, y: a comma-separated list after the function is not handled */
	return j;
}

/* a point: (a, b), a, b, [a, b] up to end */
static int
c2_point(struct P *p, uint32_t end, double *v, int max)
{
	int n = 0, close = 0;
	uint32_t e;

	if (pk(p) == T_LP || pk(p) == T_LB) {
		close = pk(p) == T_LP ? T_RP : T_RB;
		adv(p);
	}
	while (p->i < end) {
		int depth = 0;
		for (e = p->i; e < end; e++) {
			int t = p->t->kind[e];
			if (t == T_LP || t == T_LB)
				depth++;
			if ((t == T_RP || t == T_RB) && depth-- == 0)
				break;
			if (t == T_COMMA && depth == 0)
				break;
		}
		if (n == max)
			fail("too many coordinates");
		v[n++] = c2_const_upto(p, e);
		if (!accept(p, T_COMMA))
			break;
	}
	if (close && !accept(p, close))
		fail("expected a closing bracket");
	return n;
}

/*
 * A list of functions F1, F2, ... in [i, end), merged into one function
 * whose parameters are the union of theirs (by name, in order).
 */
static void
c2_list(struct P *p, uint32_t end, struct func *out, nid *bodies, int *nb)
{
	struct func parts[C2MAXF];
	int n = 0, k, j;
	uint32_t e;
	int strip = 0;

	if ((pk(p) == T_LP || pk(p) == T_LB) && cs_match(p, p->i) + 1 == end) {
		strip = 1;
		adv(p);
		end--;
	}
	while (p->i < end) {
		int depth = 0;
		for (e = p->i; e < end; e++) {
			int t = p->t->kind[e];
			if (t == T_LP || t == T_LB || t == T_LC)
				depth++;
			if (t == T_RP || t == T_RB || t == T_RC)
				depth--;
			if (t == T_COMMA && depth == 0)
				break;
			/* "and" between the last two: f, g and h */
			if (depth == 0 && t == T_ID && tok_is(p->t, e, "and"))
				break;
		}
		if (n == C2MAXF)
			fail("at most %d functions", C2MAXF);
		parts[n++] = *c2_func_upto(p, e);
		if (!accept(p, T_COMMA))
			accept_kw(p, "and");
	}
	if (strip)
		accept(p, pk(p) == T_RP ? T_RP : T_RB);
	memset(out, 0, sizeof *out);
	out->name = sym_cstr("_");
	out->anon = 1;
	for (k = 0; k < n; k++) {
		nid map[NSLOT];
		memset(map, 0, sizeof map);
		for (j = 0; j < parts[k].np; j++) {
			int s = c2_slot(out, parts[k].p[j].name);
			if (s < 0) {
				if (out->np == MAXP)
					fail("too many variables");
				s = out->np++;
				out->p[s] = parts[k].p[j];
			}
			map[j] = mk_var((uint32_t)s, parts[k].p[j].name,
			    parts[k].p[j].ty);
		}
		bodies[k] = subst(parts[k].body, map, NSLOT);
	}
	*nb = n;
	out->body = bodies[0];
}

/* "wrt x, y" or "wrt x y": slots of out */
static int
c2_wrt_list(struct P *p, uint32_t end, struct func *f, int *slots, int max)
{
	int n = 0;

	while (p->i < end && pk(p) == T_ID) {
		if (n == max)
			fail("too many variables");
		slots[n++] = c2_slot_or_fail(f, tsym(p, p->i));
		adv(p);
		accept(p, T_COMMA);
		accept_kw(p, "and");
	}
	return n;
}

static void
c2_need_eos(struct P *p)
{
	if (!at_eos(p)) {
		diag_tok = p->i;
		fail("unexpected '%.*s' after the command", (int)p->t->len[p->i],
		    p->t->src + p->t->off[p->i]);
	}
}

static const char *const c2_at_kw[] = {"at", NULL};

/* diff / partial */
static void
c2_cmd_diff(struct P *p)
{
	uint32_t E = c2_eos(p, p->i), w = c2_find_kw(p, p->i,
	    E, (const char *const[]){"wrt", NULL}), tail, start = p->i;
	struct func *f;
	int slots[C2MAXO], n = 0, order = 0, k;
	uint32_t symn = 0;

	if (w < E) {
		f = c2_func_upto(p, w);
		adv(p);
		n = c2_wrt_list(p, E, f, slots, C2MAXO);
		if (pk(p) == T_NUM)
			order = (int)cs_num(p);
	} else {
		tail = c2_tail(p, p->i, E, 1);
		f = c2_func_upto(p, tail);
		/* trailing names that are not variables of an anonymous F */
		while (p->i < E && pk(p) == T_ID) {
			uint32_t nm = tsym(p, p->i);
			k = c2_slot(f, nm);
			if (k < 0 && p->i + 1 == E && n <= 1 &&
			    (tok_is(p->t, p->i, "n") || tok_is(p->t, p->i, "k"))) {
				symn = nm;
				adv(p);
				break;
			}
			if (k < 0 && f->anon) {
				/* not a variable: it was part of the product */
				p->i = start;
				note("%.*s is not a variable of %s: read as a product (for "
				    "a derivative wrt %s write `diff F wrt %s`)",
				    (int)p->t->len[tail], p->t->src + p->t->off[tail],
				    node_str(f->body), sym_str(nm), sym_str(nm));
				f = c2_func_upto(p, E);
				n = 0;
				break;
			}
			if (k < 0)
				fail("%s has no variable %s", sym_str(f->name),
				    sym_str(nm));
			if (n == C2MAXO)
				fail("derivatives up to order %d", C2MAXO);
			slots[n++] = k;
			adv(p);
		}
		if (p->i < E && pk(p) == T_NUM)
			order = (int)cs_num(p);
	}
	c2_need_eos(p);
	notes_flush();
	if (symn) {
		k = n ? slots[0] : 0;
		if (f->np != 1 && !n)
			fail("which variable? (diff F x n)");
		if (!c2_nth(f, k, symn))
			printf("no closed form for the %s-th derivative of %s (known: "
			    "exp, sin, cos, log and powers of a linear argument); give "
			    "a number: diff F %s 3\n", sym_str(symn), node_str(f->body),
			    sym_str(f->p[k].name));
		return;
	}
	if (order) {
		if (order < 1 || order > C2MAXO)
			fail("the order is 1 to %d", C2MAXO);
		if (n == 0) {
			if (f->np != 1)
				fail("%s has several variables: diff F x %d", node_str(
				    f->body), order);
			slots[0] = 0;
			n = 1;
		}
		if (n == 1)
			for (k = 1; k < order; k++)
				slots[k] = slots[0];
		else
			fail("give either variables (diff F x y) or one variable and "
			    "an order (diff F x 2)");
		n = order;
	}
	c2_deriv(f, slots, n);
}

static int
calc2_command(struct P *p)
{
	uint32_t E, kwi;
	struct func *f;
	double at[MAXP];
	int nat = 0, k;

	if (pk(p) != T_ID)
		return 0;
	/* F'' and F'(a): a defined function's derivatives */
	{
		uint32_t base;
		int ord = cs_primes(p, p->i, &base);
		struct func *g;
		if (ord > 0 && (g = func_find(base)) != NULL && !g->anon &&
		    (pk2(p) == T_NL || pk2(p) == T_EOF || pk2(p) == T_LP)) {
			int slots[C2MAXO];
			if (g->np != 1)
				fail("%s has %d variables: use diff %s x", sym_str(base),
				    g->np, sym_str(base));
			if (ord > C2MAXO)
				fail("derivatives up to order %d", C2MAXO);
			adv(p);
			for (k = 0; k < ord; k++)
				slots[k] = 0;
			if (accept(p, T_LP)) {
				/* the value at a point */
				uint32_t rp = cs_match(p, p->i - 1);
				nid d = mk_cast(TY_REAL, g->body);
				double x;
				int ok;
				x = c2_const_upto(p, rp);
				expect(p, T_RP);
				c2_need_eos(p);
				notes_flush();
				for (k = 0; k < ord; k++)
					d = tidy(diff_node(d, 0));
				{
					const char *why = c2_point_check(g, ord, x);
					if (why) {
						printf("%s%.*s(%s) does not exist: %s\n",
						    sym_str(base), ord, "''''''''''''", c2_fnum(x),
						    why);
						return 1;
					}
				}
				{
					double v = eval_const_at(d, 0, x, &ok);
					printf("%s%.*s(%s) = %s%s\n", sym_str(base), ord,
					    "''''''''''''", c2_fnum(x), ok ? c2_fnum(v) :
					    "undefined", ok ? "" : " (or the derivative does "
					    "not exist there)");
					printf("  from %s%.*s = %s\n", sym_str(base), ord,
					    "''''''''''''", node_str(d));
				}
				return 1;
			}
			c2_need_eos(p);
			notes_flush();
			c2_deriv(g, slots, ord);
			return 1;
		}
	}
	if (func_find(tsym(p, p->i)) != NULL)
		return 0;
	/* fx, fxy, f_x: partial derivatives of a defined f */
	if (pk2(p) == T_NL || pk2(p) == T_EOF) {
		size_t len, b;
		const char *w = ttext(p, p->i, &len);
		for (b = len - 1; b > 0; b--) {
			struct func *g;
			size_t bl = w[b - 1] == '_' ? b - 1 : b, j;
			int slots[C2MAXO], n = 0;
			if (bl == 0 || (g = func_find(sym_intern(w, bl))) == NULL ||
			    g->anon || len - b > C2MAXO)
				continue;
			for (j = b; j < len; j++) {
				char v[2] = {w[j], 0};
				int sl = c2_slot(g, sym_cstr(v));
				if (sl < 0)
					break;
				slots[n++] = sl;
			}
			if (j < len)
				continue;
			adv(p);
			notes_flush();
			c2_deriv(g, slots, n);
			return 1;
		}
	}
	if (is_kw(p, "range") && c2_is_word(p, p->i + 1, "of") &&
	    func_find(sym_cstr("of")) == NULL) {
		int nocrit;
		adv(p);
		adv(p);
		f = cmd_func(p);
		nocrit = accept_kw(p, "nocrit");
		expect_eos(p);
		notes_flush();
		range_nocrit = nocrit;
		cmd_range(f);
		range_nocrit = 0;
		return 1;
	}
	if (is_kw(p, "diff") || is_kw(p, "partial")) {
		adv(p);
		c2_cmd_diff(p);
		return 1;
	}
	if (is_kw(p, "grad") || is_kw(p, "gradient") || is_kw(p, "hessian") ||
	    is_kw(p, "laplacian")) {
		int what = is_kw(p, "hessian") ? 1 : is_kw(p, "laplacian") ? 2 : 0;
		adv(p);
		E = c2_eos(p, p->i);
		kwi = c2_find_kw(p, p->i, E, c2_at_kw);
		f = c2_func_upto(p, kwi);
		if (accept_kw(p, "at"))
			nat = c2_point(p, E, at, MAXP);
		c2_need_eos(p);
		if (nat && nat != f->np)
			fail("the point has %d coordinates; %s has %d variables", nat,
			    f->anon ? node_str(f->body) : sym_str(f->name), f->np);
		notes_flush();
		c2_grad(f, what, at, nat > 0);
		return 1;
	}
	if (is_kw(p, "jacobian") || is_kw(p, "wronskian")) {
		int jac = is_kw(p, "jacobian"), vars[MAXP], nv = 0, nb;
		struct func m;
		nid bodies[C2MAXF];
		uint32_t w;
		adv(p);
		accept_kw(p, "of");
		E = c2_eos(p, p->i);
		w = c2_find_kw(p, p->i, E, (const char *const[]){"wrt", "at", NULL});
		c2_list(p, w, &m, bodies, &nb);
		if (accept_kw(p, "wrt")) {
			uint32_t w2 = c2_find_kw(p, p->i, E, c2_at_kw);
			nv = c2_wrt_list(p, w2, &m, vars, MAXP);
		}
		if (jac && accept_kw(p, "at"))
			nat = c2_point(p, E, at, MAXP);
		c2_need_eos(p);
		notes_flush();
		f = func_new(sym_cstr("_"));
		*f = m;
		if (jac) {
			if (nv == 0)
				for (nv = 0; nv < m.np; nv++)
					vars[nv] = nv;
			if (nat && nat != m.np)
				fail("the point has %d coordinates; there are %d "
				    "variables", nat, m.np);
			c2_jacobian(f, bodies, nb, vars, nv, at, nat > 0);
		} else {
			int s = 0;
			if (nv > 1)
				fail("the Wronskian is wrt one variable");
			if (nv == 1)
				s = vars[0];
			else if (m.np > 1) {
				s = c2_slot(&m, sym_cstr("x"));
				if (s < 0)
					s = c2_slot(&m, sym_cstr("t"));
				if (s < 0)
					s = 0;
				printf("  note: the functions have %d variables: the "
				    "Wronskian wrt %s, the others held fixed (say `wrt v`)\n",
				    m.np, sym_str(m.p[s].name));
			}
			c2_wronskian(f, bodies, nb, s);
		}
		return 1;
	}
	if (is_kw(p, "ddir") || is_kw(p, "directional")) {
		double v[MAXP];
		int nv, raw = 0;
		uint32_t a;
		adv(p);
		accept_kw(p, "derivative");
		accept_kw(p, "of");
		E = c2_eos(p, p->i);
		if (c2_is_word(p, E - 1, "raw")) {
			raw = 1;
			E--;
		}
		a = c2_find_kw(p, p->i, E, (const char *const[]){"along", "dir",
		    "direction", "toward", "towards", NULL});
		if (a == E)
			fail("ddir F along v1, v2 [at a, b]");
		f = c2_func_upto(p, a);
		adv(p);
		kwi = c2_find_kw(p, p->i, E, c2_at_kw);
		nv = c2_point(p, kwi, v, MAXP);
		if (accept_kw(p, "at"))
			nat = c2_point(p, E, at, MAXP);
		if (raw)
			adv(p);
		c2_need_eos(p);
		if (nat && nat != f->np)
			fail("the point has %d coordinates; %s has %d variables", nat,
			    f->anon ? node_str(f->body) : sym_str(f->name), f->np);
		notes_flush();
		c2_ddir(f, v, nv, at, nat > 0, raw);
		return 1;
	}
	if (is_kw(p, "limit") || is_kw(p, "lim")) {
		uint32_t arrow, varpos, ptend, fstart, fend;
		int side = 0, slot = 0;
		double a;
		uint32_t vname = 0;
		adv(p);
		accept_kw(p, "of");
		E = c2_eos(p, p->i);
		/* "from the left/right" at the end */
		{
			uint32_t fr = c2_find_kw(p, p->i, E, (const char *const[]){
			    "from", NULL});
			if (fr < E) {
				uint32_t j = fr + 1;
				if (c2_is_word(p, j, "the"))
					j++;
				if (c2_is_word(p, j, "left") || c2_is_word(p, j, "below"))
					side = -1;
				else if (c2_is_word(p, j, "right") || c2_is_word(p, j,
				    "above"))
					side = 1;
				else
					fail("from left or from right");
				if (j + 1 != E)
					fail("unexpected words after 'from %.*s'",
					    (int)p->t->len[j], p->t->src + p->t->off[j]);
				E = fr;
			}
		}
		for (arrow = p->i; arrow < E && p->t->kind[arrow] != T_ARROW;
		    arrow++)
			;
		if (arrow == E) {
			/* limit F at a */
			kwi = c2_find_kw(p, p->i, E, c2_at_kw);
			if (kwi == E)
				fail("limit F x -> a, lim x->a F, or limit F at a");
			f = c2_func_upto(p, kwi);
			adv(p);
			ptend = E;
			if (ptend > p->i + 1 && (p->t->kind[ptend - 1] == T_PLUS ||
			    p->t->kind[ptend - 1] == T_MINUS)) {
				side = p->t->kind[ptend - 1] == T_PLUS ? 1 : -1;
				ptend--;
			}
			a = c2_const_upto(p, ptend);
			if (ptend < E)
				adv(p);
			if (f->np != 1)
				fail("which variable? (limit F x -> a)");
			slot = 0;
		} else {
			if (arrow == p->i || p->t->kind[arrow - 1] != T_ID)
				fail("expected a variable before '->'");
			varpos = arrow - 1;
			vname = tsym(p, varpos);
			if (varpos == p->i || (varpos == p->i + 1 && pk(p) == T_LP)) {
				/* lim x->a F, lim(x->a) F */
				int paren = varpos == p->i + 1;
				uint32_t s = arrow + 1, t;
				if (paren) {
					t = cs_match(p, p->i);
					ptend = t;
				} else {
					/* the point: tokens up to the first space */
					for (t = s + 1; t < E && !(p->t->fl[t] & TF_SPACE) &&
					    p->t->kind[t] != T_COMMA; t++)
						;
					ptend = t;
				}
				if (ptend > s + 1 && (p->t->kind[ptend - 1] == T_PLUS ||
				    p->t->kind[ptend - 1] == T_MINUS) && !paren) {
					side = p->t->kind[ptend - 1] == T_PLUS ? 1 : -1;
					ptend--;
				} else if (paren && ptend > s + 1 &&
				    (p->t->kind[ptend - 1] == T_PLUS ||
				    p->t->kind[ptend - 1] == T_MINUS)) {
					side = p->t->kind[ptend - 1] == T_PLUS ? 1 : -1;
					ptend--;
				}
				if (ptend > s + 1 && p->t->kind[ptend - 1] == T_CARET)
					ptend--;   /* 0^+ */
				p->i = s;
				a = c2_const_upto(p, ptend);
				p->i = paren ? t + 1 : t;
				accept(p, T_COMMA);
				accept(p, T_COLON);
				accept_kw(p, "of");
				fstart = p->i;
				fend = E;
				f = c2_func_upto(p, fend);
				(void)fstart;
			} else {
				/* limit F [as] x -> a */
				fend = varpos;
				if (fend > p->i && c2_is_word(p, fend - 1, "as"))
					fend--;
				f = c2_func_upto(p, fend);
				p->i = arrow + 1;
				ptend = E;
				if (ptend > p->i + 1 && (p->t->kind[ptend - 1] == T_PLUS ||
				    p->t->kind[ptend - 1] == T_MINUS)) {
					side = p->t->kind[ptend - 1] == T_PLUS ? 1 : -1;
					ptend--;
					if (ptend > p->i + 1 && p->t->kind[ptend - 1] == T_CARET)
						ptend--;
				}
				a = c2_const_upto(p, ptend);
				p->i = E;
			}
			slot = c2_slot(f, vname);
			if (slot < 0) {
				if (f->np == 0 && f->anon) {
					printf("%s does not depend on %s: the limit is itself\n",
					    node_str(f->body), sym_str(vname));
					return 1;
				}
				fail("%s has no variable %s", f->anon ? node_str(f->body) :
				    sym_str(f->name), sym_str(vname));
			}
		}
		p->i = E;
		if (pk(p) == T_ID)
			p->i = c2_eos(p, p->i);
		c2_need_eos(p);
		notes_flush();
		c2_limit(f, slot, a, side);
		return 1;
	}
	if (is_kw(p, "taylor") || is_kw(p, "series") || is_kw(p, "maclaurin")) {
		int mac = is_kw(p, "maclaurin"), order = 5, slot = 0;
		double a = 0;
		uint32_t t;
		adv(p);
		accept_kw(p, "series");
		accept_kw(p, "of");
		E = c2_eos(p, p->i);
		kwi = c2_find_kw(p, p->i, E, (const char *const[]){"at", "order",
		    "degree", "to", "about", "around", NULL});
		t = c2_tail(p, p->i, kwi, 0);
		if (t + 1 < kwi)
			t = kwi;   /* one variable at most */
		f = c2_func_upto(p, t);
		if (p->i < kwi) {
			slot = c2_slot_or_fail(f, tsym(p, p->i));
			adv(p);
		} else if (f->np > 1)
			fail("which variable? (taylor F x at a)");
		while (p->i < E) {
			if (accept_kw(p, "at") || accept_kw(p, "about") ||
			    accept_kw(p, "around")) {
				uint32_t e2 = c2_find_kw(p, p->i, E, (const char *const[]){
				    "order", "degree", "to", NULL});
				a = c2_const_upto(p, e2);
				if (mac && a != 0)
					fail("a Maclaurin series is at 0 (taylor F at a)");
			} else if (accept_kw(p, "order") || accept_kw(p, "degree") ||
			    accept_kw(p, "to")) {
				accept_kw(p, "order");
				accept_kw(p, "degree");
				order = (int)cs_num(p);
			} else
				break;
		}
		c2_need_eos(p);
		notes_flush();
		c2_taylor(f, slot, a, order);
		return 1;
	}
	if (is_kw(p, "implicit")) {
		uint32_t eq, ke;
		struct func m;
		nid bodies[2], F;
		int nb, xs = -1, ys = -1, order = 1;
		uint32_t xn = 0, yn = 0;
		adv(p);
		E = c2_eos(p, p->i);
		ke = c2_find_kw(p, p->i, E, (const char *const[]){"for", "wrt",
		    "order", NULL});
		for (eq = p->i; eq < ke && p->t->kind[eq] != T_ASSIGN &&
		    p->t->kind[eq] != T_EQ; eq++)
			;
		{
			/* L = R as the list "L, R" */
			uint8_t sv = p->t->kind[eq < ke ? eq : ke];
			if (eq < ke)
				p->t->kind[eq] = T_COMMA;
			c2_list(p, ke, &m, bodies, &nb);
			if (eq < ke)
				p->t->kind[eq] = sv;
		}
		F = nb == 2 ? SUB(mk_cast(TY_REAL, bodies[0]), mk_cast(TY_REAL,
		    bodies[1])) : bodies[0];
		if (nb > 2)
			fail("implicit L = R");
		while (p->i < E) {
			if (accept_kw(p, "for")) {
				if (pk(p) != T_ID)
					fail("for which variable?");
				yn = tsym(p, p->i);
				adv(p);
			} else if (accept_kw(p, "wrt")) {
				if (pk(p) != T_ID)
					fail("wrt which variable?");
				xn = tsym(p, p->i);
				adv(p);
			} else if (accept_kw(p, "order")) {
				order = (int)cs_num(p);
			} else
				break;
		}
		c2_need_eos(p);
		if (m.np < 2)
			fail("implicit differentiation needs two variables (F(x, y) = "
			    "0)");
		if (yn)
			ys = c2_slot_or_fail(&m, yn);
		if (xn)
			xs = c2_slot_or_fail(&m, xn);
		if (ys < 0 && (k = c2_slot(&m, sym_cstr("y"))) >= 0 && k != xs)
			ys = k;
		if (xs < 0 && (k = c2_slot(&m, sym_cstr("x"))) >= 0 && k != ys)
			xs = k;
		if (ys < 0)
			ys = xs == m.np - 1 ? 0 : m.np - 1;
		if (xs < 0)
			for (xs = 0; xs == ys; xs++)
				;
		if (!yn && !(c2_slot(&m, sym_cstr("y")) == ys &&
		    c2_slot(&m, sym_cstr("x")) == xs))
			note("taking %s as a function of %s (say `for %s wrt %s` for "
			    "the other way)", sym_str(m.p[ys].name),
			    sym_str(m.p[xs].name), sym_str(m.p[xs].name),
			    sym_str(m.p[ys].name));
		notes_flush();
		f = func_new(sym_cstr("_"));
		*f = m;
		c2_implicit(f, F, xs, ys, order);
		return 1;
	}
	return 0;
}
