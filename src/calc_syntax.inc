/*
 * calc syntax: loops, sums, recurrences, ODE and PDE statements, and the
 * calc commands.  This file is #included at the end of parse.c (it uses
 * the parser's static helpers) and is not compiled on its own.
 *
 *   sum(i, lo, hi, body)      also sum(i in lo..hi, body), sum(i in lo..hi) body,
 *   prod(...), Σ(...), ∏(...) the index runs lo..hi inclusive
 *   x[0] = 1                  a recurrence: initial values and one step rule
 *   x[k+1] = a*x[k] + b       (x[n] = x[n-1] + x[n-2] works too)
 *   f(n) = { s = 0            a code block: assignments (= += -= *= /= ++),
 *     for i in 1..n {         for i in lo..hi { }, while c { }, if c { } else { },
 *       if i % 2 == 0 { s += i } and a final expression or 'return e'
 *     }
 *     s }
 *   y' = -2y + 1; y(0) = 3    an ODE and its initial values
 *   pde name { u_t = 0.1 u_xx; u(x, 0) = sin(pi x); x in 0..1; t in 0..0.5 }
 */
#include "calc.h"

static int cs_base = MAXP;        /* next free bound-variable slot */
static uint32_t cs_rec;           /* recurrence whose step is being parsed */
static nid cs_ridx;               /* its index variable */
static ty_t cs_rty;               /* type of x[...] in the step */
static int cs_dmin, cs_dmax, cs_dseen;
#define PHSLOT 32                 /* x[k+d] placeholders: PHSLOT + d + 16 */
static ty_t cs_dty[MAXBIND];      /* declared types of block variables */

static int calc_block_stmts(struct P *p, nid *result);

/* the source text of tokens [a, b) */
static char *
cs_text(struct P *p, uint32_t a, uint32_t b)
{
	uint32_t s, e;
	char *r;

	if (b <= a)
		{ r = xmalloc(1); *r = 0; return r; }
	s = p->t->off[a];
	e = p->t->off[b - 1] + p->t->len[b - 1];
	r = xmalloc(e - s + 1);
	memcpy(r, p->t->src + s, e - s);
	r[e - s] = 0;
	return r;
}

/* end of the statement starting at i (NL/EOF at brace depth 0) */
static uint32_t
cs_stmt_end(struct P *p, uint32_t i)
{
	int depth = 0;

	for (; i < p->t->n; i++) {
		int k = p->t->kind[i];
		if (k == T_LC || k == T_LP || k == T_LB)
			depth++;
		if (k == T_RC || k == T_RP || k == T_RB)
			depth--;
		if (k == T_EOF || (k == T_NL && depth <= 0))
			break;
	}
	return i;
}

/* the matching close of the bracket at i */
static uint32_t
cs_match(struct P *p, uint32_t i)
{
	int depth = 0;

	for (; i < p->t->n; i++) {
		int k = p->t->kind[i];
		if (k == T_LC || k == T_LP || k == T_LB)
			depth++;
		if (k == T_RC || k == T_RP || k == T_RB)
			if (--depth == 0)
				return i;
		if (k == T_EOF)
			break;
	}
	fail("unbalanced brackets");
	return 0;
}

static int
cs_is_name(struct P *p, uint32_t i, const char *a, const char *b)
{
	return tok_is(p->t, i, a) || (b && tok_is(p->t, i, b));
}

/* integer division whose divisor reads slot s, in an integer type */
static int
cs_intdiv(nid n, uint64_t slots, uint8_t *seen)
{
	if (n == 0 || seen[n])
		return 0;
	seen[n] = 1;
	if (N.op[n] == O_DIV && TY_ISINT(N.ty[n]) && (node_slots(N.b[n]) & slots))
		return 1;
	if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_TABLE ||
	    N.op[n] == O_LOOP)
		return 0;
	return cs_intdiv(N.a[n], slots, seen) || cs_intdiv(N.b[n], slots, seen) ||
	    cs_intdiv(N.c[n], slots, seen);
}

static int
has_intdiv(nid n, uint64_t slots)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int r = cs_intdiv(n, slots, seen);

	free(seen);
	return r;
}

static ty_t
cs_state_ty(ty_t t)
{
	if (TYK(t) == K_LIT || t == TY_BOOL)
		return TY_INT;
	return t;
}

/* ---- sum and prod ---- */

static nid
cs_sumprod(struct P *p, int isprod)
{
	uint32_t iname, bodyat, bndat;
	nid lo, hi, body = 0, iv, init;
	struct loopdef d;
	int np0, form, saved_base = cs_base, j, tries;
	ty_t ity = TY_INT, t;
	const char *what = isprod ? "prod" : "sum";

	adv(p);
	expect(p, T_LP);
	if (pk(p) != T_ID)
		fail("%s needs an index variable: %s(i, lo, hi, body)", what, what);
	iname = tsym(p, p->i);
	adv(p);
	form = 0;
	if (accept(p, T_COMMA))
		form = 1;
	else if (accept_kw(p, "in") || accept(p, T_ASSIGN))
		form = 2;
	else
		fail("expected ',' or 'in' after the index of %s", what);
	np0 = p->fn ? p->fn->np : 0;
	bndat = p->i;
	for (tries = 0;; tries++) {
		p->i = bndat;
		lo = parse_expr(p, form == 1 ? 0 : BP_CMP + 1);
		if (form == 1)
			expect(p, T_COMMA);
		else
			expect(p, T_DOTDOT);
		hi = parse_expr(p, form == 1 ? 0 : BP_CMP + 1);
		/* a free variable used as a bound is an integer */
		if (tries == 0 && p->fn && p->autop && p->fn->np > np0) {
			int redo = 0;
			for (j = np0; j < p->fn->np; j++)
				if (p->fn->p[j].ty == TY_REAL && !p->fn->p[j].ranged) {
					p->fn->p[j].ty = TY_INT;
					p->fn->p[j].ranged = 1;
					p->fn->p[j].lo = 0;
					p->fn->p[j].hi = 100;
					p->fn->p[j].rlo = 0;
					p->fn->p[j].rhi = 100;
					note("%s is a bound of %s: taken as int in 0..100",
					    sym_str(p->fn->p[j].name), what);
					redo = 1;
				}
			if (redo)
				continue;
		}
		break;
	}
	if (TY_FP(N.ty[lo]) || TY_FP(N.ty[hi]))
		note("bounds of %s are rounded toward zero to integers", what);
	if (form == 1)
		expect(p, T_COMMA);
	else if (!accept(p, T_COMMA)) {
		expect(p, T_RP);
		form = 3;   /* sum(i in 1..n) body */
	}
	bodyat = p->i;
	for (tries = 0; tries < 2; tries++) {
		int saved_nb = p->nb, saved_abs = p->abs_depth;
		p->i = bodyat;
		if (p->nb == MAXBIND)
			fail("too many bindings");
		cs_base = saved_base + 2;
		iv = var_of(saved_base, iname, ity);
		p->bname[p->nb] = iname;
		p->bval[p->nb++] = iv;
		if (form == 3) {
			/* sum(i in 1..n) body: the body is a product term */
			body = parse_expr(p, BP_MUL);
		} else {
			p->abs_depth = 0;
			body = parse_expr(p, 0);
			p->abs_depth = saved_abs;
		}
		p->nb = saved_nb;
		cs_base = saved_base;
		if (tries == 0 && has_intdiv(body, 1ULL << saved_base)) {
			note("%s would divide integers by the index %s: %s taken as "
			    "real", node_str(body), sym_str(iname), sym_str(iname));
			ity = TY_REAL;
			continue;
		}
		break;
	}
	if (form != 3)
		expect(p, T_RP);
	t = cs_state_ty(N.ty[body]);
	if (TYK(t) == K_LIT)
		t = TY_INT;
	memset(&d, 0, sizeof d);
	d.kind = LK_FOR;
	d.orig = isprod ? LO_PROD : LO_SUM;
	d.name = iname;
	d.islot = saved_base;
	d.ity = ity;
	d.nst = 1;
	d.sslot[0] = saved_base + 1;
	d.sname[0] = sym_cstr(isprod ? "prod" : "sum");
	d.sty[0] = t;
	init = mk_const_i(t, isprod ? 1 : 0);
	d.init[0] = init;
	{
		nid s = var_of(d.sslot[0], d.sname[0], t);
		d.next[0] = mk_cast(t, isprod ? MUL(s, body) : ADD(s, body));
	}
	d.aux = body;
	p->top_jux = 0;
	return mk_loop(loop_intern(&d), 0, lo, hi, 0);
}

/* ---- recurrences ---- */

#define MAXREC 32
struct rec {
	uint32_t name, iname;
	int off;                  /* the step defines x[iname + off] */
	char *step;
	char *init[MAXST];
	int nparam;
	uint32_t pname[MAXP];
	int done;
};
static struct rec recs[MAXREC];
static int nrecs;

static struct rec *
rec_find(uint32_t name)
{
	int i;

	for (i = 0; i < nrecs; i++)
		if (recs[i].name == name)
			return &recs[i];
	return NULL;
}

/* x[k + d] while parsing x's step */
static nid
cs_rec_ref(struct P *p)
{
	nid e, k = cs_ridx, x;
	int64_t d = 0;
	char nm[64];

	adv(p);
	expect(p, T_LB);
	e = parse_expr(p, 0);
	expect(p, T_RB);
	if (e == k)
		d = 0;
	else if (N.op[e] == O_ADD && N.a[e] == k && int_of(N.b[e], &d))
		;
	else if (N.op[e] == O_SUB && N.a[e] == k && int_of(N.b[e], &d))
		d = -d;
	else
		fail("in a recurrence, %s[...] takes %s plus or minus a constant",
		    sym_str(cs_rec), node_str(k));
	if (d < -16 || d > 7)
		fail("recurrence offsets are limited to -16..7");
	if (!cs_dseen || d < cs_dmin)
		cs_dmin = (int)d;
	if (!cs_dseen || d > cs_dmax)
		cs_dmax = (int)d;
	cs_dseen = 1;
	if (d == 0)
		snprintf(nm, sizeof nm, "%s[%s]", sym_str(cs_rec), node_str(k));
	else
		snprintf(nm, sizeof nm, "%s[%s%+d]", sym_str(cs_rec), node_str(k),
		    (int)d);
	x = var_of(PHSLOT + (int)d + 16, sym_cstr(nm), cs_rty);
	p->top_jux = 0;
	return x;
}

/* parse a stored fragment with fn's parameters (free names are added) */
static nid
cs_parse_frag(const char *src, struct func *fn, ty_t autoty, uint32_t bname,
    nid bval)
{
	struct toks *t = xcalloc(1, sizeof *t);
	struct P *q = xcalloc(1, sizeof *q);
	struct toks *saved_toks = diag_toks;
	uint32_t saved_tok = diag_tok;
	jmp_buf saved;
	nid r;

	memcpy(saved, fail_jmp, sizeof saved);
	lex(t, src);
	q->t = t;
	q->fn = fn;
	q->autop = 1;
	q->autoty = autoty;
	if (bname) {
		q->bname[0] = bname;
		q->bval[0] = bval;
		q->nb = 1;
	}
	diag_toks = t;
	if (setjmp(fail_jmp)) {
		memcpy(fail_jmp, saved, sizeof saved);
		diag_toks = saved_toks;
		diag_tok = saved_tok;
		longjmp(fail_jmp, 1);
	}
	r = parse_expr(q, 0);
	skip_nl(q);
	if (pk(q) != T_EOF)
		fail("unexpected '%.*s'", (int)t->len[q->i], t->src + t->off[q->i]);
	memcpy(fail_jmp, saved, sizeof saved);
	diag_toks = saved_toks;
	diag_tok = saved_tok;
	toks_free(t);
	free(t);
	free(q);
	return r;
}

nid
calc_parse_text(const char *src, struct func *fn, ty_t autoty)
{
	return cs_parse_frag(src, fn, autoty, 0, 0);
}

static void
rec_build(struct rec *r)
{
	struct func fn;
	struct loopdef d;
	nid init[MAXST], step = 0, map[NSLOT], k;
	ty_t t = 0;
	int m, j, tries, saved_base = cs_base;
	struct func *f;

	if (r->step == NULL)
		return;   /* the step rule comes later */
	memset(&fn, 0, sizeof fn);
	fn.name = r->name;
	fn.np = 1;
	fn.p[0].name = sym_cstr("n");
	fn.p[0].ty = TY_INT;
	fn.p[0].ranged = 1;
	fn.p[0].lo = 0;
	fn.p[0].hi = 60;
	fn.p[0].rhi = 60;
	/* the step first, to learn the order */
	cs_base = MAXP + 1 + MAXST;
	k = var_of(MAXP, r->iname, TY_INT);
	t = TY_INT;
	for (j = 0; j < MAXST && r->init[j]; j++) {
		init[j] = cs_parse_frag(r->init[j], &fn, TY_REAL, 0, 0);
		if (TY_FP(N.ty[init[j]]))
			t = N.ty[init[j]];
		else if (!TY_FP(t) && TY_SIZED(N.ty[init[j]]) &&
		    N.ty[init[j]] != TY_BOOL)
			t = N.ty[init[j]];
	}
	for (tries = 0; tries < 3; tries++) {
		cs_rec = r->name;
		cs_ridx = k;
		cs_rty = t;
		cs_dseen = 0;
		step = cs_parse_frag(r->step, &fn, TY_REAL, r->iname, k);
		cs_rec = 0;
		if (TY_FP(N.ty[step]) && !TY_FP(t)) {
			t = TY_REAL;
			continue;
		}
		if (!TY_FP(t) && has_intdiv(step, ((1ULL << 24) - 1) << PHSLOT)) {
			note("%s[...] would be divided as an integer: %s taken as "
			    "real", sym_str(r->name), sym_str(r->name));
			t = TY_REAL;
			continue;
		}
		break;
	}
	cs_base = saved_base;
	if (!cs_dseen)
		fail("the step of %s does not read %s[...]", sym_str(r->name),
		    sym_str(r->name));
	if (cs_dmax >= r->off)
		fail("%s[%s%+d] is defined in terms of itself or later terms",
		    sym_str(r->name), sym_str(r->iname), r->off);
	m = r->off - cs_dmin;
	if (m > MAXST)
		fail("recurrence of order %d (at most %d)", m, MAXST);
	for (j = 0; j < m; j++)
		if (r->init[j] == NULL) {
			note("%s needs the initial value %s[%d] before it can be "
			    "used", sym_str(r->name), sym_str(r->name), j);
			return;   /* not complete yet */
		}
	memset(&d, 0, sizeof d);
	memset(map, 0, sizeof map);
	d.kind = LK_FOR;
	d.orig = LO_REC;
	d.flags = LF_NEGUNDEF;
	d.name = r->name;
	d.islot = MAXP;
	d.ity = TY_INT;
	d.nst = m;
	for (j = 0; j < m; j++) {
		char nm[64];
		snprintf(nm, sizeof nm, "%s%d", sym_str(r->name), j);
		d.sslot[j] = MAXP + 1 + j;
		d.sname[j] = sym_cstr(nm);
		d.sty[j] = t;
		map[PHSLOT + cs_dmin + j + 16] = var_of(d.sslot[j], d.sname[j], t);
	}
	step = subst(step, map, NSLOT);
	for (j = 0; j < m; j++) {
		d.init[j] = mk_cast(t, init[j]);
		d.next[j] = j + 1 < m ? var_of(d.sslot[j + 1], d.sname[j + 1], t) :
		    mk_cast(t, step);
	}
	f = func_new(r->name);
	*f = fn;
	{
		nid n = var_of(0, fn.p[0].name, TY_INT);
		/* states hold x[k+dmin..]; the index starts where x[m] is made */
		f->body = mk_loop(loop_intern(&d), 0, KI(-cs_dmin),
		    ADD(n, KI(-cs_dmin - 1)), n);
	}
	r->nparam = fn.np - 1;
	for (j = 1; j < fn.np; j++)
		r->pname[j - 1] = fn.p[j].name;
	r->done = 1;
}

/* x[e] for a defined recurrence x */
static nid
cs_rec_use(struct P *p, struct rec *r)
{
	struct func *g = func_find(r->name);
	nid args[MAXP];
	int j;

	if (!r->done || g == NULL)
		fail("%s is not complete: it needs a step rule and initial values",
		    sym_str(r->name));
	adv(p);
	expect(p, T_LB);
	args[0] = parse_expr(p, 0);
	expect(p, T_RB);
	for (j = 0; j < r->nparam; j++) {
		const char *s = sym_str(r->pname[j]);
		nid v = value_name(p, s, strlen(s));
		if (!v) {
			if (p->autop && p->fn)
				v = new_auto_param(p, r->pname[j]);
			else
				fail("%s[...] reads %s, which is not defined here",
				    sym_str(r->name), s);
		}
		args[j + 1] = v;
	}
	p->top_jux = 0;
	return call_user(p, g, args, r->nparam + 1);
}

/* x[...] = ...  at the start of a statement */
static int
cs_rec_stmt(struct P *p)
{
	uint32_t name = tsym(p, p->i), rb, end, a;
	struct rec *r;
	int64_t v;

	if (pk2(p) != T_LB)
		return 0;
	rb = cs_match(p, p->i + 1);
	if (rb + 1 >= p->t->n || p->t->kind[rb + 1] != T_ASSIGN)
		return 0;
	if ((r = rec_find(name)) == NULL) {
		if (nrecs == MAXREC)
			fail("too many recurrences");
		r = &recs[nrecs++];
		memset(r, 0, sizeof *r);
		r->name = name;
	}
	a = p->i + 2;
	end = cs_stmt_end(p, rb + 2);
	if (rb == a + 1 && p->t->kind[a] == T_NUM) {
		p->i = a;
		v = N.ki[number(p)];
		if (v < 0 || v >= MAXST)
			fail("initial values x[0]..x[%d]", MAXST - 1);
		free(r->init[v]);
		r->init[v] = cs_text(p, rb + 2, end);
	} else {
		int off = 0;
		if (p->t->kind[a] != T_ID)
			fail("the step is written x[k+1] = ... or x[n] = ...");
		if (rb == a + 3 && (p->t->kind[a + 1] == T_PLUS ||
		    p->t->kind[a + 1] == T_MINUS) && p->t->kind[a + 2] == T_NUM) {
			p->i = a + 2;
			off = (int)N.ki[number(p)];
			if (p->t->kind[a + 1] == T_MINUS)
				off = -off;
		} else if (rb != a + 1)
			fail("the step is written x[k+1] = ... or x[n] = ...");
		r->iname = tsym(p, a);
		r->off = off;
		free(r->step);
		r->step = cs_text(p, rb + 2, end);
	}
	p->i = end;
	r->done = 0;
	rec_build(r);
	return 1;
}

/* ---- code blocks ---- */

static int
cs_is_block(struct P *p)
{
	uint32_t i = p->i + 1;
	int k;

	while (i < p->t->n && p->t->kind[i] == T_NL)
		i++;
	if (cs_is_name(p, i, "for", "while") || cs_is_name(p, i, "if", "return"))
		return 1;
	if (p->t->kind[i] != T_ID)
		return 0;
	k = p->t->kind[i + 1];
	if (k == T_ASSIGN || k == T_COLON)
		return 1;
	if ((k == T_PLUS || k == T_MINUS || k == T_STAR || k == T_SLASH ||
	    k == T_PCT) && p->t->kind[i + 2] == T_ASSIGN)
		return 1;
	if ((k == T_PLUS || k == T_MINUS) && p->t->kind[i + 2] == k)
		return 1;
	if (k == T_COMMA) {
		while (p->t->kind[i] == T_ID && p->t->kind[i + 1] == T_COMMA)
			i += 2;
		return p->t->kind[i] == T_ID && p->t->kind[i + 1] == T_ASSIGN;
	}
	return 0;
}

/* a, b = ... */
static int
cs_tuple(struct P *p)
{
	uint32_t i = p->i;

	while (p->t->kind[i] == T_ID && p->t->kind[i + 1] == T_COMMA)
		i += 2;
	return p->t->kind[i] == T_ID && p->t->kind[i + 1] == T_ASSIGN;
}

static void
cs_set(struct P *p, uint32_t name, nid v)
{
	int j = binding(p, name);

	name_hint(v, name);
	if (j >= 0) {
		if (cs_dty[j])
			v = mk_cast(cs_dty[j], v);
		p->bval[j] = v;
		name_hint(v, name);
		return;
	}
	if (p->nb == MAXBIND)
		fail("too many variables in a block (%d)", MAXBIND);
	cs_dty[p->nb] = 0;
	p->bname[p->nb] = name;
	p->bval[p->nb++] = v;
}

static nid
cs_get(struct P *p, uint32_t name)
{
	const char *s = sym_str(name);
	nid v = value_name(p, s, strlen(s));

	if (!v)
		fail("%s is not defined before this point", s);
	return v;
}

/* names assigned in tokens [a, b) */
static int
cs_assigned(struct P *p, uint32_t a, uint32_t b, uint32_t *names, int max)
{
	uint32_t i;
	int n = 0, j, k;

	for (i = a; i < b; i++) {
		int hit = 0;
		uint32_t first = i;
		if (p->t->kind[i] != T_ID)
			continue;
		k = p->t->kind[i + 1];
		if (k == T_ASSIGN)
			hit = 1;
		else if ((k == T_PLUS || k == T_MINUS || k == T_STAR ||
		    k == T_SLASH || k == T_PCT) && p->t->kind[i + 2] == T_ASSIGN)
			hit = 1;
		else if ((k == T_PLUS || k == T_MINUS) && p->t->kind[i + 2] == k)
			hit = 1;
		else if (k == T_COMMA && (p->t->kind[i - 1] == T_NL ||
		    p->t->kind[i - 1] == T_LC)) {
			uint32_t e = i;
			while (p->t->kind[e] == T_ID && p->t->kind[e + 1] == T_COMMA)
				e += 2;
			if (p->t->kind[e] == T_ID && p->t->kind[e + 1] == T_ASSIGN) {
				for (; first <= e; first += 2) {
					uint32_t nm = tsym(p, first);
					for (j = 0; j < n; j++)
						if (names[j] == nm)
							break;
					if (j == n && n < max)
						names[n++] = nm;
				}
				i = e;
			}
			continue;
		}
		if (!hit)
			continue;
		{
			uint32_t nm = tsym(p, i);
			for (j = 0; j < n; j++)
				if (names[j] == nm)
					break;
			if (j == n && n < max)
				names[n++] = nm;
		}
	}
	return n;
}

/*
 * What makes n a value per lane: a parameter that is not uniform (its
 * name), a loop state of lanes (what that one follows), a scan's lane
 * ("lane"); not what goes in a vote, a ballot or a sum, whose value the
 * lanes share, nor a shuffle's value at a shared index.  0: none.
 */
static uint32_t
cs_per_lane(struct P *p, nid n, uint64_t lanes, uint8_t *seen)
{
	uint32_t r, j;
	int k;

	if (n == 0 || n >= N.n || seen[n])
		return 0;
	seen[n] = 1;
	switch (N.op[n]) {
	case O_CONST:
	case O_UNDEF:
		return 0;
	case O_VAR:
		k = (int)N.ki[n];
		if (k >= 0 && k < p->fn->np && !p->fn->p[k].uniform)
			return p->fn->p[k].name;
		if (k >= MAXP && k < 64 && (lanes >> k & 1))
			return p->lane_from[k];
		return 0;
	case O_TABLE:
		for (j = 0; j < N.b[n]; j++)
			if ((r = cs_per_lane(p, N.lst[N.a[n] + j], lanes, seen)) != 0)
				return r;
		return 0;
	case O_SGSCAN:
		return sym_cstr("lane");
	case O_SGSHUF:
		return cs_per_lane(p, N.b[n], lanes, seen);
	case O_LOOP: {
		uint64_t f = loops.free[(uint32_t)N.ki[n]];
		for (k = 0; k < 64; k++)
			if ((f >> k & 1) && ((k < p->fn->np && !p->fn->p[k].uniform) ||
			    (k >= MAXP && (lanes >> k & 1))))
				return k < p->fn->np ? p->fn->p[k].name : p->lane_from[k];
		break;
	}
	default:
		if (IS_SG(N.op[n]))
			return 0;
	}
	if ((r = cs_per_lane(p, N.a[n], lanes, seen)) != 0 ||
	    (r = cs_per_lane(p, N.b[n], lanes, seen)) != 0)
		return r;
	return cs_per_lane(p, N.c[n], lanes, seen);
}

/*
 * The states of a loop (slots base + 1 ..) that follow a value per lane:
 * an initial or a next value reads one, or reads such a state; *from,
 * *name and *ty of each are set.  The loops around it count with theirs.
 */
static uint64_t
cs_lane_states(struct P *p, int base, int m, const nid *init, const nid *next,
    const uint32_t *names, const ty_t *sty)
{
	uint64_t lanes = p->lanest, mine = 0;
	int j, more = 1;

	if (p->fn == NULL || p->autop)
		return 0;
	while (more) {
		more = 0;
		for (j = 0; j < m; j++) {
			int s = base + 1 + j;
			uint8_t *seen;
			uint32_t o;
			if (s >= 64 || (lanes >> s & 1))
				continue;
			seen = xcalloc(N.n, 1);
			o = cs_per_lane(p, init[j], lanes, seen);
			if (!o) {
				memset(seen, 0, N.n);
				o = cs_per_lane(p, next[j], lanes, seen);
			}
			free(seen);
			if (o) {
				lanes |= 1ULL << s;
				mine |= 1ULL << s;
				p->lane_from[s] = o;
				p->lane_name[s] = names[j];
				p->lane_ty[s] = sty[j];
				more = 1;
			}
		}
	}
	return mine;
}

static void
cs_loop(struct P *p, int kind)
{
	uint32_t iname = 0, names[32], bodyat, condat = 0, endb;
	nid lo = 0, hi = 0, cond = 0, snap[MAXBIND], next[MAXST];
	int nn, m = 0, bidx[MAXST], j, tries, saved_base = cs_base;
	ty_t sty[MAXST];
	struct loopdef d;
	uint32_t e;

	adv(p);
	if (kind == LK_FOR) {
		if (pk(p) != T_ID)
			fail("for needs an index: for i in lo..hi { ... }");
		iname = tsym(p, p->i);
		adv(p);
		if (!accept_kw(p, "in"))
			expect(p, T_ASSIGN);
		lo = parse_expr(p, BP_CMP + 1);
		expect(p, T_DOTDOT);
		hi = parse_expr(p, BP_CMP + 1);
	} else {
		int depth = 0;
		condat = p->i;
		for (; p->i < p->t->n; p->i++) {
			int k = pk(p);
			if (k == T_LP || k == T_LB)
				depth++;
			if (k == T_RP || k == T_RB)
				depth--;
			if ((k == T_LC && depth == 0) || k == T_EOF)
				break;
		}
	}
	skip_nl(p);
	if (pk(p) != T_LC)
		fail("expected '{' to start the loop body");
	bodyat = p->i;
	endb = cs_match(p, bodyat);
	nn = cs_assigned(p, bodyat, endb, names, 32);
	for (j = 0; j < nn; j++) {
		const char *s = sym_str(names[j]);
		int bj = binding(p, names[j]);
		if (kind == LK_FOR && names[j] == iname)
			fail("the loop body assigns the index %s", s);
		if (bj < 0) {
			int ps = param_slot(p, names[j]);
			if (ps < 0)
				continue;   /* a temporary of the body */
			cs_set(p, names[j], param_var(p, ps));
			bj = binding(p, names[j]);
		}
		if (m == MAXST)
			fail("a loop carries at most %d variables", MAXST);
		bidx[m] = bj;
		sty[m] = cs_state_ty(N.ty[p->bval[bj]]);
		if (cs_dty[bj])
			sty[m] = cs_dty[bj];
		m++;
	}
	if (m == 0)
		fail("the loop body changes no variable defined before the loop");
	if (kind == LK_FOR) {
		lo = mk_cast(TY_INT, lo);
		hi = mk_cast(TY_INT, hi);
	}
	for (tries = 0;; tries++) {
		int saved_nb = p->nb, retry = 0, lretry = 0;
		nid res = 0;
		uint64_t sgread0 = p->sgread;
		p->sgread = 0;
		memcpy(snap, p->bval, sizeof snap);
		cs_base = saved_base + 1 + m;
		if (cs_base + 1 >= PHSLOT)
			fail("loops nested too deeply");
		for (j = 0; j < m; j++)
			p->bval[bidx[j]] = var_of(saved_base + 1 + j,
			    p->bname[bidx[j]], sty[j]);
		if (kind == LK_FOR) {
			if (p->nb == MAXBIND)
				fail("too many variables");
			cs_dty[p->nb] = 0;
			p->bname[p->nb] = iname;
			p->bval[p->nb++] = var_of(saved_base, iname, TY_INT);
		} else {
			p->i = condat;
			cond = mk_bool(parse_expr(p, 0));
		}
		p->i = bodyat;
		adv(p);
		if (calc_block_stmts(p, &res) || res)
			fail("a loop body has no result; assign to a variable "
			    "defined before the loop");
		for (j = 0; j < m; j++) {
			next[j] = p->bval[bidx[j]];
			if (TY_FP(N.ty[next[j]]) && !TY_FP(sty[j]) &&
			    !cs_dty[bidx[j]]) {
				sty[j] = N.ty[next[j]];
				retry = 1;
			}
		}
		{
			/* a state that follows a value per lane, read across the
			 * lanes by a subgroup operation of the loop: once more,
			 * with its copies in the other lanes */
			nid in0[MAXST];
			uint32_t nm[MAXST];
			uint64_t lm = 0, mine = 0;
			for (j = 0; j < m; j++) {
				in0[j] = snap[bidx[j]];
				nm[j] = p->bname[bidx[j]];
				if (saved_base + 1 + j < 64)
					mine |= 1ULL << (saved_base + 1 + j);
			}
			if (p->sgread & mine & ~p->lanest)
				lm = cs_lane_states(p, saved_base, m, in0, next, nm,
				    sty);
			if (lm & p->sgread & ~p->lanest) {
				p->lanest |= lm;
				lretry = 1;
			}
		}
		p->sgread |= sgread0;
		p->nb = saved_nb;
		memcpy(p->bval, snap, sizeof snap);
		cs_base = saved_base;
		if ((!retry || tries >= 3) && !lretry)
			break;
	}
	/* the slots are the next loop's */
	for (j = 0; j < m && saved_base + 1 + j < 64; j++)
		p->lanest &= ~(1ULL << (saved_base + 1 + j));
	memset(&d, 0, sizeof d);
	d.kind = kind;
	d.orig = LO_BLOCK;
	d.name = kind == LK_FOR ? iname : sym_cstr("while");
	d.islot = saved_base;
	d.ity = TY_INT;
	d.nst = m;
	d.cond = cond;
	for (j = 0; j < m; j++) {
		d.sslot[j] = saved_base + 1 + j;
		d.sname[j] = p->bname[bidx[j]];
		d.sty[j] = sty[j];
		d.init[j] = mk_cast(sty[j], p->bval[bidx[j]]);
		d.next[j] = mk_cast(sty[j], next[j]);
	}
	e = loop_intern(&d);
	for (j = 0; j < m; j++)
		p->bval[bidx[j]] = mk_loop(e, j, lo, hi, 0);
	p->i = endb + 1;
}

static void
cs_if(struct P *p)
{
	nid c, s0[MAXBIND], s1[MAXBIND], r = 0;
	int nb0 = p->nb, j;
	uint32_t save;

	adv(p);
	c = mk_bool(parse_expr(p, 0));
	skip_nl(p);
	if (pk(p) != T_LC)
		fail("expected '{' after the condition of if");
	memcpy(s0, p->bval, sizeof s0);
	adv(p);
	if (calc_block_stmts(p, &r) || r)
		fail("'if' branches assign variables; they have no value "
		    "(use c ? a : b for a value)");
	p->nb = nb0;
	memcpy(s1, p->bval, sizeof s1);
	memcpy(p->bval, s0, sizeof s0);
	save = p->i;
	skip_nl(p);
	if (accept_kw(p, "else")) {
		skip_nl(p);
		if (is_kw(p, "if"))
			cs_if(p);
		else {
			if (pk(p) != T_LC)
				fail("expected '{' after else");
			adv(p);
			if (calc_block_stmts(p, &r) || r)
				fail("'else' branch with a value");
			p->nb = nb0;
		}
	} else
		p->i = save;
	for (j = 0; j < nb0; j++)
		if (s1[j] != p->bval[j])
			p->bval[j] = mk_sel(c, s1[j], p->bval[j]);
}

/*
 * The statements of a block after its '{', through the '}'.  Returns 1
 * when the block ended with 'return e' (r = e), and sets *r to a final
 * bare expression too.
 */
static int
calc_block_stmts(struct P *p, nid *result)
{
	int k;

	*result = 0;
	for (;;) {
		skip_nl(p);
		if (accept(p, T_RC))
			return 0;
		if (*result)
			fail("the value of a block must be its last statement");
		diag_tok = p->i;
		if (is_kw(p, "for")) {
			cs_loop(p, LK_FOR);
			continue;
		}
		if (is_kw(p, "while")) {
			cs_loop(p, LK_WHILE);
			continue;
		}
		if (is_kw(p, "if")) {
			cs_if(p);
			continue;
		}
		if (accept_kw(p, "return")) {
			*result = parse_expr(p, 0);
			continue;
		}
		if (pk(p) == T_ID) {
			uint32_t name = tsym(p, p->i);
			k = pk2(p);
			if (k == T_ASSIGN) {
				adv(p);
				adv(p);
				cs_set(p, name, parse_expr(p, 0));
				continue;
			}
			if (k == T_COLON) {
				ty_t t;
				size_t n;
				const char *s = ttext(p, p->i + 2, &n);
				if (!ty_parse(s, n, &t))
					fail("unknown type '%.*s'", (int)n, s);
				adv(p);
				adv(p);
				adv(p);
				expect(p, T_ASSIGN);
				{
					nid v = mk_cast(t, parse_expr(p, 0));
					int j = binding(p, name);
					if (j < 0) {
						cs_set(p, name, v);
						j = p->nb - 1;
					} else
						p->bval[j] = v;
					cs_dty[j] = t;
				}
				continue;
			}
			if ((k == T_PLUS || k == T_MINUS || k == T_STAR ||
			    k == T_SLASH || k == T_PCT) &&
			    p->t->kind[p->i + 2] == T_ASSIGN) {
				static const int ops[] = {
					[T_PLUS] = O_ADD, [T_MINUS] = O_SUB,
					[T_STAR] = O_MUL, [T_SLASH] = O_DIV,
					[T_PCT] = O_MOD,
				};
				nid old = cs_get(p, name);
				adv(p);
				adv(p);
				adv(p);
				cs_set(p, name, mk_bin(ops[k], old, parse_expr(p, 0)));
				continue;
			}
			if ((k == T_PLUS || k == T_MINUS) &&
			    p->t->kind[p->i + 2] == k) {
				nid old = cs_get(p, name);
				adv(p);
				adv(p);
				adv(p);
				cs_set(p, name, mk_bin(k == T_PLUS ? O_ADD : O_SUB, old,
				    KI(1)));
				continue;
			}
			if (k == T_COMMA && cs_tuple(p)) {
				uint32_t nm[MAXST];
				nid v[MAXST];
				int n = 0, j;
				do {
					if (n == MAXST)
						fail("too many names");
					nm[n++] = tsym(p, p->i);
					adv(p);
				} while (accept(p, T_COMMA));
				expect(p, T_ASSIGN);
				for (j = 0; j < n; j++) {
					if (j)
						expect(p, T_COMMA);
					v[j] = parse_expr(p, 0);
				}
				for (j = 0; j < n; j++)
					cs_set(p, nm[j], v[j]);
				continue;
			}
		}
		*result = parse_expr(p, 0);
	}
}

static nid
calc_block(struct P *p)
{
	int saved_nb = p->nb, saved_abs = p->abs_depth, saved_fnarg = p->fnarg;
	nid r = 0;
	uint32_t end = cs_match(p, p->i);
	jmp_buf saved;

	expect(p, T_LC);
	p->abs_depth = 0;
	p->fnarg = 0;
	/* an error inside resumes after the whole block, not in its middle */
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp)) {
		memcpy(fail_jmp, saved, sizeof saved);
		p->i = end + 1;
		longjmp(fail_jmp, 1);
	}
	calc_block_stmts(p, &r);
	memcpy(fail_jmp, saved, sizeof saved);
	if (!r)
		fail("a block needs a value: end it with an expression or "
		    "'return e'");
	p->nb = saved_nb;
	p->abs_depth = saved_abs;
	p->fnarg = saved_fnarg;
	p->top_jux = 0;
	p->top_paren = 1;
	return r;
}

/* ---- names in operand position ---- */

static nid
calc_ident(struct P *p)
{
	uint32_t name = tsym(p, p->i);
	struct rec *r;

	if (cs_rec && name == cs_rec && pk2(p) == T_LB)
		return cs_rec_ref(p);
	if (pk2(p) == T_LP && func_find(name) == NULL) {
		if (cs_is_name(p, p->i, "sum", "\xce\xa3"))
			return cs_sumprod(p, 0);
		if (cs_is_name(p, p->i, "prod", "\xe2\x88\x8f"))
			return cs_sumprod(p, 1);
	}
	if (pk2(p) == T_LB && (r = rec_find(name)) != NULL)
		return cs_rec_use(p, r);
	return 0;
}

/* ---- ODE and PDE statements ---- */

static int
cs_primes(struct P *p, uint32_t i, uint32_t *base)
{
	size_t n;
	const char *s = ttext(p, i, &n);
	int k = 0;

	while (n > 0 && s[n - 1] == '\'') {
		n--;
		k++;
	}
	*base = sym_intern(s, n);
	return k;
}

static int
cs_ode_stmt(struct P *p)
{
	uint32_t base, end, rp;
	int ord;

	if (pk(p) != T_ID)
		return 0;
	ord = cs_primes(p, p->i, &base);
	if (ord > 0 && pk2(p) == T_ASSIGN) {
		end = cs_stmt_end(p, p->i + 2);
		{
			char *s = cs_text(p, p->i + 2, end);
			ode_equation(base, ord, s, p->i);
			free(s);
		}
		p->i = end;
		return 1;
	}
	if (pk2(p) != T_LP)
		return 0;
	rp = cs_match(p, p->i + 1);
	if (rp + 1 >= p->t->n || p->t->kind[rp + 1] != T_ASSIGN)
		return 0;
	{
		uint32_t i;
		for (i = p->i + 2; i < rp; i++)
			if (p->t->kind[i] == T_ID && !tok_is(p->t, i, "pi") &&
			    !tok_is(p->t, i, "e"))
				return 0;
	}
	end = cs_stmt_end(p, rp + 2);
	{
		char *t0 = cs_text(p, p->i + 2, rp), *v = cs_text(p, rp + 2, end);
		ode_initial(base, ord, t0, v);
		free(t0);
		free(v);
	}
	p->i = end;
	return 1;
}

static nid
cs_const_expr(struct P *p)
{
	struct func *saved = p->fn;
	int sc = p->cmdargs, sa = p->autop;
	nid v;

	p->fn = NULL;
	p->autop = 0;
	p->cmdargs = 1;
	v = parse_expr(p, BP_CMP + 1);
	p->fn = saved;
	p->cmdargs = sc;
	p->autop = sa;
	if (N.op[v] == O_NEG && is_const(N.a[v]) && TY_FP(N.ty[v]))
		v = mk_const_r(-N.kr[N.a[v]]);   /* -inf does not fold */
	if (!is_const(v))
		fail("expected a constant, found %s", node_str(v));
	return v;
}

static double
cs_num(struct P *p)
{
	double v;

	num_of(cs_const_expr(p), &v);
	return v;
}

/* pde name { u_t = ...; u(x, 0) = ...; u(0, t) = ...; x in a..b; t in a..b } */
static void
cs_pde(struct P *p)
{
	struct pde pd;
	uint32_t name;

	adv(p);
	if (pk(p) != T_ID)
		fail("pde needs a name");
	name = tsym(p, p->i);
	adv(p);
	skip_nl(p);
	expect(p, T_LC);
	memset(&pd, 0, sizeof pd);
	pd.xb = 1;
	pd.tb = 1;
	pd.nx = 200;
	for (;;) {
		while (accept(p, T_NL) || accept(p, T_COMMA))
			;
		if (accept(p, T_RC))
			break;
		diag_tok = p->i;
		if (pk(p) != T_ID)
			fail("expected u_t = ..., u(x, 0) = ..., x in a..b, ...");
		if (accept_kw(p, "periodic")) {
			pd.periodic = 1;
			continue;
		}
		if (accept_kw(p, "grid")) {
			pd.nx = (int)cs_num(p);
			continue;
		}
		if (pk2(p) == T_ID && tok_is(p->t, p->i + 1, "in")) {
			uint32_t v = tsym(p, p->i);
			double a, b;
			adv(p);
			adv(p);
			a = cs_num(p);
			expect(p, T_DOTDOT);
			b = cs_num(p);
			if (v == pd.t || (pd.t == 0 && v == sym_cstr("t"))) {
				pd.t = v;
				pd.ta = a;
				pd.tb = b;
			} else {
				pd.x = v;
				pd.xa = a;
				pd.xb = b;
			}
			continue;
		}
		if (pk2(p) == T_ASSIGN) {
			size_t n;
			const char *s = ttext(p, p->i, &n);
			struct func fn;
			const char *us = memchr(s, '_', n);
			if (us == NULL)
				fail("the equation is written u_t = ... or u_tt = ...");
			pd.u = sym_intern(s, (size_t)(us - s));
			pd.order_t = (int)(n - (size_t)(us - s) - 1);
			if (pd.order_t < 1 || pd.order_t > 2)
				fail("u_t or u_tt");
			adv(p);
			adv(p);
			memset(&fn, 0, sizeof fn);
			fn.name = sym_cstr("_");
			p->fn = &fn;
			p->autop = 1;
			p->autoty = TY_REAL;
			pd.rhs = parse_expr(p, 0);
			p->fn = NULL;
			p->autop = 0;
			pd.rf = fn;
			continue;
		}
		if (pk2(p) == T_LP) {
			/* u(x, 0) = ..., u_t(x, 0) = ..., u(0, t) = ... */
			size_t n;
			const char *s = ttext(p, p->i, &n);
			int vel = memchr(s, '_', n) != NULL;
			struct func fn;
			nid *dst;
			adv(p);
			adv(p);
			memset(&fn, 0, sizeof fn);
			fn.name = sym_cstr("_");
			fn.np = 1;
			fn.p[0].ty = TY_REAL;
			if (pk(p) == T_ID) {
				fn.p[0].name = tsym(p, p->i);
				if (!pd.x)
					pd.x = fn.p[0].name;
				adv(p);
				expect(p, T_COMMA);
				(void)cs_num(p);
				dst = vel ? &pd.v0 : &pd.u0;
			} else {
				double xv = cs_num(p);
				expect(p, T_COMMA);
				if (pk(p) != T_ID)
					fail("boundary values are written u(0, t) = ...");
				fn.p[0].name = tsym(p, p->i);
				if (!pd.t)
					pd.t = fn.p[0].name;
				adv(p);
				if (vel)
					fail("only value (Dirichlet) boundaries");
				dst = (pd.nbset++ == 0) ? &pd.bl : &pd.br;
				if (dst == &pd.bl)
					pd.blx = xv;
				else
					pd.brx = xv;
			}
			expect(p, T_RP);
			expect(p, T_ASSIGN);
			p->fn = &fn;
			p->autop = 0;
			*dst = mk_cast(TY_REAL, parse_expr(p, 0));
			p->fn = NULL;
			continue;
		}
		fail("unexpected '%.*s' in a pde block", (int)p->t->len[p->i],
		    p->t->src + p->t->off[p->i]);
	}
	expect_eos(p);
	if (!pd.rhs)
		fail("the pde needs an equation u_t = ...");
	if (!pd.u0)
		fail("the pde needs an initial value u(x, 0) = ...");
	if (!pd.x)
		pd.x = sym_cstr("x");
	if (!pd.t)
		pd.t = sym_cstr("t");
	pde_define(name, &pd);
}

/* statements that are not definitions of the usual kind */
static int
calc_statement(struct P *p)
{
	if (pk(p) != T_ID)
		return 0;
	if (cs_rec_stmt(p))
		return 1;
	if (cs_ode_stmt(p))
		return 1;
	return 0;
}

/* ---- commands ---- */

static int
calc_command(struct P *p)
{
	struct func *f;
	nid a = 0, b = 0;
	int slot = 0, steps = 0;
	uint32_t nm;

	if (pk(p) != T_ID || func_find(tsym(p, p->i)) != NULL)
		return 0;
	if (is_kw(p, "pde") && pk2(p) == T_ID) {
		cs_pde(p);
		return 1;
	}
	if (is_kw(p, "closed") || is_kw(p, "scheme") || is_kw(p, "continuous")) {
		int which = is_kw(p, "closed") ? 0 : is_kw(p, "scheme") ? 1 : 2;
		struct model *m = NULL;
		int more = 0;
		adv(p);
		f = cmd_func(p);
		/* scheme F more: the forms beyond Horner and Estrin too */
		if (which == 1)
			more = accept_kw(p, "more");
		if (which == 1 && accept_kw(p, "model")) {
			if (pk(p) != T_ID || (m = model_find(tsym(p, p->i))) == NULL)
				fail("unknown model");
			adv(p);
		}
		expect_eos(p);
		notes_flush();
		if (which == 0)
			cmd_closed(f);
		else if (which == 1)
			cmd_scheme(f, m, more);
		else
			cmd_continuous(f);
		return 1;
	}
	if (is_kw(p, "integrate") || tok_is(p->t, p->i, "\xe2\x88\xab") ||
	    is_kw(p, "antiderivative") || is_kw(p, "primitive")) {
		/* antiderivative F [wrt x]: integrate without bounds */
		struct qmcopt qo;
		int numq = 0;
		int anti = !is_kw(p, "integrate") && !tok_is(p->t, p->i,
		    "\xe2\x88\xab");
		adv(p);
		if (anti)
			accept_kw(p, "of");
		f = cmd_func(p);
		for (;;) {
			if (accept_kw(p, "wrt")) {
				if (pk(p) != T_ID)
					fail("wrt needs a parameter name");
				nm = tsym(p, p->i);
				for (slot = 0; slot < f->np; slot++)
					if (f->p[slot].name == nm)
						break;
				if (slot == f->np && f->np == 0 && f->anon) {
					/* a constant: integrate it over the variable named */
					f->p[0].name = nm;
					f->p[0].ty = TY_REAL;
					f->np = 1;
					slot = 0;
				} else if (slot == f->np)
					fail("no parameter %s", sym_str(nm));
				adv(p);
			} else if (accept_kw(p, "from")) {
				a = cs_const_expr(p);
				if (!accept_kw(p, "to"))
					expect(p, T_DOTDOT);
				b = cs_const_expr(p);
			} else if (accept_kw(p, "numerically")) {
				/* by name: quasi-Monte Carlo (qmc.c) */
				if (!accept_kw(p, "qmc"))
					fail("integrate ... numerically qmc [points N] "
					    "[replicates R] [seed K] [noscramble] (without "
					    "it a definite integral is checked by adaptive "
					    "Gauss-Kronrod)");
				qmc_defaults(&qo);
				while (qmc_options(p, &qo))
					;
				numq = 1;
			} else
				break;
		}
		expect_eos(p);
		if (numq) {
			double av = 0, bv = 0;
			if (anti)
				fail("an antiderivative is not a number: integrate F "
				    "from a to b numerically qmc");
			if (a && (!num_of(a, &av) || !num_of(b, &bv)))
				fail("qmc: the bounds must be numbers");
			notes_flush();
			printf("\xe2\x88\xab %s numerically", node_str(f->body));
			if (a)
				printf(", %s from %g to %g\n", f->np > slot ?
				    sym_str(f->p[slot].name) : "x", av, bv);
			else
				printf(", over the box of its parameters' ranges\n");
			rep_begin("integrate");
			rep_expr("integrand", f->body);
			qmc_integrate(f, f->body, a ? slot : -1, av, bv, &qo, "  ");
			return 1;
		}
		if (anti && a)
			fail("an antiderivative is indefinite: for the definite "
			    "integral say integrate F from a to b");
		if (f->np == 0 && f->anon) {
			/* "integrate 1": the integrand names no variable */
			static const char *const vars[] = {"x (dx)", "t (dt)",
			    "y (dy)", "another: say `integrate ... wrt v`"};
			static const char *const names[] = {"x", "t", "y", "x"};
			char what[160];
			int k = 0;
			snprintf(what, sizeof what, "integrate %s with respect to which "
			    "variable", node_str(f->body));
			if (nlp_can_ask()) {
				k = nlp_ask_choice(what, vars, 4, 0);
				if (nlp_asked)
					return 1;
			} else
				note("%s names no variable: integrating with respect to x "
				    "(say `wrt t` for another)", node_str(f->body));
			f->p[0].name = sym_cstr(names[k]);
			f->p[0].ty = TY_REAL;
			f->np = 1;
			slot = 0;
		}
		notes_flush();
		cmd_integrate(f, slot, a, b);
		return 1;
	}
	if (is_kw(p, "dsolve") && pk2(p) == T_ID) {
		adv(p);
		nm = tsym(p, p->i);
		{
			uint32_t base;
			cs_primes(p, p->i, &base);
			nm = base;
		}
		adv(p);
		for (;;) {
			if (accept_kw(p, "from")) {
				a = cs_const_expr(p);
				if (!accept_kw(p, "to"))
					expect(p, T_DOTDOT);
				b = cs_const_expr(p);
			} else if (accept_kw(p, "to")) {
				b = cs_const_expr(p);
			} else if (accept_kw(p, "steps")) {
				steps = (int)cs_num(p);
			} else
				break;
		}
		expect_eos(p);
		notes_flush();
		cmd_dsolve(nm, a, b, steps);
		return 1;
	}
	return 0;
}
