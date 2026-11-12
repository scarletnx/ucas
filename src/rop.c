/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * rop: graphics fixed function as compute.  `rop F` takes the update a
 * scatter applies to a cell, F(s, d): s the value a fragment brings, d
 * the cell's value (the parameter named d or dst, else the last one), an
 * optional `uniform` parameter a pipeline constant, an optional second
 * source.  It looks through the
 * catalogue of fixed-function states for the ones that compute F, each
 * state's result defined exactly:
 *
 *   logic ops      VkLogicOp on an unsigned integer target: the 16
 *                  bitwise functions of (s, d)
 *   stencil ops    VkStencilOp on the 8-bit stencil value, with a write
 *                  mask and the reference value: counters, an invert
 *   blending       VkBlendOp ADD, SUBTRACT, REVERSE_SUBTRACT with every
 *                  pair of factors (0, 1, the colours, the constant, the
 *                  second source and their 1 - x), MIN, MAX
 *   advanced       VK_EXT_blend_operation_advanced on opaque colours:
 *                  MULTIPLY, SCREEN, DARKEN, LIGHTEN, DIFFERENCE,
 *                  EXCLUSION
 *   depth test     LESS or GREATER_OR_EQUAL on D32_SFLOAT: a min or max
 *                  whose high bits are the key and whose low bits, the
 *                  payload, ride in a colour target (an argmin)
 *
 * Nothing is recognised by name: every state is compared with F on F's
 * domain.  Integers are decided on every point or by SAT; a blend
 * equation by expanding both sides to the same polynomial, and it is
 * exact on integers when the ranges keep every product and the sum
 * inside (-2^24, 2^24), where f32 does not round.  Whether the result
 * depends on the order the fragments arrive in is decided the same way
 * (F(s2, F(s1, d)) against F(s1, F(s2, d))).
 *
 * A state is a proposal: it changes the pipeline (a render target, a
 * blend or stencil state, one point per contribution), which is spelled
 * out, and it is priced by a class of its own (rop_blend, rop_adv,
 * rop_logic, rop_stencil, rop_depth) that a model lists only once it is
 * measured.
 */
#include "calc.h"
#include "alg.h"
#include "fuse.h"
#include "memk.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define P24 16777216.0

struct rctx {
	struct func *f;
	struct model *m;
	int id, is, it, ic;	/* slots: the cell, the sources, the constant */
	nid vd, vs, vt, vc;
	ty_t ty;
	int flt, w;
	int nsrc;
	int ordered;		/* 1: the result depends on the order */
	int nforms, navail;
	float against;
	struct sbuf againstwhy;
	/*
	 * rop_offer (simplify): nothing printed; the cheapest form the model
	 * prices below the atomic is kept, and why a form was refused
	 */
	int quiet;
	struct rop_offer *best;
};

static void
say(const struct rctx *c, const char *fmt, ...)
{
	va_list ap;

	if (c->quiet)
		return;
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
}

static const char *
pname(const struct rctx *c, int slot)
{
	return sym_str(c->f->p[slot].name);
}

/* ---- the report ---- */

static void
price(struct rctx *c, int cls)
{
	const char *mn = sym_str(c->m->name);

	if (!cclass_available(c->m, cls)) {
		printf("      price: %s is not priced by %s (not measured): "
		    "unavailable\n", cclass_name[cls], mn);
		rep_str("class", cclass_name[cls]);
		rep_bool("available", 0);
		return;
	}
	c->navail++;
	printf("      price: %s %g a fragment under %s, against %g (%s): %s\n",
	    cclass_name[cls], (double)c->m->c[cls], mn, (double)c->against,
	    sb_str(&c->againstwhy), c->m->c[cls] < c->against ? "cheaper" :
	    "not cheaper");
	rep_str("class", cclass_name[cls]);
	rep_bool("available", 1);
	rep_num("cost", c->m->c[cls]);
	rep_num("against", c->against);
	rep_bool("cheaper", c->m->c[cls] < c->against);
}

static void
form(struct rctx *c, const char *family, const char *state,
    const char *holds, const char *proof, int cls, const char *pipeline)
{
	if (c->quiet) {
		struct rop_offer *o = c->best;
		c->nforms++;
		if (!cclass_available(c->m, cls) || !(c->m->c[cls] < c->against)) {
			if (!o->found && !o->near[0] && !cclass_available(c->m, cls))
				snprintf(o->near, sizeof o->near, "%s (a %s) computes "
				    "it, but %s does not price %s (not measured)", state,
				    family, sym_str(c->m->name), cclass_name[cls]);
			else if (!o->found && !o->near[0])
				snprintf(o->near, sizeof o->near, "%s (a %s) computes "
				    "it, at %s %g a fragment: not below %g (%s)", state,
				    family, cclass_name[cls], (double)c->m->c[cls],
				    (double)c->against, sb_str(&c->againstwhy));
			return;
		}
		if (o->found && !(c->m->c[cls] < o->cost))
			return;
		o->found = 1;
		o->cls = cls;
		o->cost = c->m->c[cls];
		o->against = c->against;
		snprintf(o->family, sizeof o->family, "%s", family);
		snprintf(o->state, sizeof o->state, "%s", state);
		snprintf(o->holds, sizeof o->holds, "%s", holds ? holds : "");
		snprintf(o->proof, sizeof o->proof, "%s", proof);
		snprintf(o->pipeline, sizeof o->pipeline, "%s", pipeline);
		snprintf(o->againstwhy, sizeof o->againstwhy, "%s",
		    sb_str(&c->againstwhy));
		return;
	}
	if (!c->nforms++) {
		printf("  fixed-function states that compute it (each a proposal: "
		    "the pipeline changes):\n");
		rep_list("forms");
	}
	printf("  %s: %s\n", family, state);
	if (holds && holds[0])
		printf("      holds where %s\n", holds);
	printf("      %s\n", proof);
	rep_obj(NULL);
	rep_str("family", family);
	rep_str("state", state);
	if (holds && holds[0])
		rep_str("holds_where", holds);
	rep_str("proof", proof);
	price(c, cls);
	printf("      pipeline: %s\n", pipeline);
	rep_str("pipeline", pipeline);
	rep_end();
}

/*
 * a equals b on f's domain: 1 proved (how says the way), 0 refuted by a
 * point, -1 not decided (a sample agrees and SAT gave no answer)
 */
static int
same(const struct func *f, nid a, nid b, const char **how)
{
	struct eqproof pr;
	jmp_buf saved;
	volatile int bad = -1;

	memset(&pr, 0, sizeof pr);
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		bad = equiv_decide(f, a, b, NULL, &pr) != 0;
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	if (bad == 1)
		return 0;
	if (bad < 0 || pr.how == EQ_SAMPLED)
		return -1;
	*how = pr.how == EQ_EXHAUSTIVE ? "checked on every point of the domain" :
	    "proved by SAT";
	return 1;
}

/* ---- the order of the fragments ---- */

static nid
subst4(const struct rctx *c, nid e, nid s, nid t, nid d, nid k)
{
	nid map[MAXP] = {0};

	if (c->is >= 0)
		map[c->is] = s;
	if (c->it >= 0)
		map[c->it] = t;
	map[c->id] = d;
	if (c->ic >= 0)
		map[c->ic] = k;
	return subst(e, map, c->f->np);
}

/*
 * e over real variables and real constants: +, -, *, negation, and for
 * floats /, min, max, |.| (an integer / truncates: not the real one).
 * 0: it is something else.
 */
static nid
to_real(nid e, int flt, int depth)
{
	int op = N.op[e];
	nid a, b;

	if (depth > 60)
		return 0;
	if (op == O_VAR)
		return mk_var((uint32_t)N.ki[e], N.a[e], TY_REAL);
	if (op == O_CONST)
		return mk_const_r(TY_FP(N.ty[e]) ? N.kr[e] : (double)N.ki[e]);
	if (op == O_NEG || (flt && op == O_ABS)) {
		a = to_real(N.a[e], flt, depth + 1);
		return a ? mk_un(op, a) : 0;
	}
	if (op == O_ADD || op == O_SUB || op == O_MUL || (flt && (op == O_DIV ||
	    op == O_MIN || op == O_MAX))) {
		a = to_real(N.a[e], flt, depth + 1);
		b = to_real(N.b[e], flt, depth + 1);
		return a && b ? mk_bin(op, a, b) : 0;
	}
	return 0;
}

/* the same, guarded: building it must not end the command */
static nid
as_real(const struct rctx *c, nid e)
{
	jmp_buf saved;
	volatile nid r = 0;

	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = to_real(e, c->flt, 0);
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return r;
}

static void
order(struct rctx *c)
{
	struct func g;
	nid s1, s2, t1 = 0, t2 = 0, d, k = 0, a, b;
	int n = 0;
	const char *how = "";
	char nm[16];
	const char *fn = c->f->anon ? "F" : sym_str(c->f->name);

	if (!c->nsrc) {
		printf("  order: every fragment applies the same update: their "
		    "order is immaterial\n");
		rep_str("order", "immaterial");
		return;
	}
	memset(&g, 0, sizeof g);
	g.anon = 1;
#define PAR(v, slot, name_) do { \
		g.p[n] = c->f->p[slot]; \
		snprintf(nm, sizeof nm, "%s", name_); \
		g.p[n].name = sym_cstr(nm); \
		v = mk_var((uint32_t)n, g.p[n].name, c->ty); \
		n++; \
	} while (0)
	PAR(s1, c->is, "s1");
	PAR(s2, c->is, "s2");
	if (c->it >= 0) {
		PAR(t1, c->it, "t1");
		PAR(t2, c->it, "t2");
	}
	PAR(d, c->id, pname(c, c->id));
	if (c->ic >= 0)
		PAR(k, c->ic, pname(c, c->ic));
#undef PAR
	g.np = n;
	a = subst4(c, c->f->body, s2, t2, subst4(c, c->f->body, s1, t1, d, k), k);
	b = subst4(c, c->f->body, s1, t1, subst4(c, c->f->body, s2, t2, d, k), k);
	g.body = a;
	if (!c->flt) {
		int r = same(&g, a, b, &how);
		if (r < 0) {
			c->ordered = 1;
			printf("  order: not decided (a sample of both orders agrees, "
			    "SAT gave no answer): taken as depending on the order\n");
			rep_str("order", "undecided");
		} else if (r) {
			printf("  order: the result does not depend on the order of "
			    "the fragments: %s(s2, %s(s1, d)) = %s(s1, %s(s2, d)) (%s)"
			    "\n", fn, fn, fn, fn, how);
			rep_str("order", "independent");
			rep_str("order_proof", how);
		} else {
			c->ordered = 1;
			printf("  order: the result depends on the order of the "
			    "fragments (%s(s2, %s(s1, d)) is not %s(s1, %s(s2, d))): "
			    "a draw applies them in primitive order, atomics in "
			    "any\n", fn, fn, fn, fn);
			rep_str("order", "dependent");
		}
		return;
	}
	/* floats: min and max exactly; the rest as real functions */
	if ((N.op[c->f->body] == O_MIN || N.op[c->f->body] == O_MAX) &&
	    ((N.a[c->f->body] == c->vs && N.b[c->f->body] == c->vd) ||
	    (N.a[c->f->body] == c->vd && N.b[c->f->body] == c->vs))) {
		printf("  order: the result does not depend on the order of the "
		    "fragments: %s is associative and commutative on floats other "
		    "than NaN (up to the sign of a zero)\n",
		    opinfo[N.op[c->f->body]].name);
		rep_str("order", "independent");
		return;
	}
	{
		struct rf ra, rb;
		size_t mk = ar_mark();
		nid ar, br;
		int eq = 0;
		ar = as_real(c, a);
		br = as_real(c, b);
		atoms_reset();
		if (ar && br && rf_of(ar, &ra, NULL) && rf_of(br, &rb, NULL)) {
			ra = rf_norm(ra);
			rb = rf_norm(rb);
			eq = mp_eq(ra.n, rb.n) && mp_eq(ra.d, rb.d) && !alg_ovf;
		}
		ar_release(mk);
		if (eq) {
			c->ordered = 1;
			printf("  order: as real functions the result does not depend "
			    "on the order of the fragments (both orders expand to "
			    "the same rational function); in f32 each operation "
			    "rounds, so it does unless every step is exact\n");
			rep_str("order", "independent as real functions");
		} else {
			c->ordered = 1;
			printf("  order: not shown independent of the order of the "
			    "fragments: a draw applies them in primitive order, "
			    "atomics in any\n");
			rep_str("order", "dependent");
		}
	}
}

/* ---- integers: logic ops, stencil ops, UNORM targets ---- */

static const char *
uint_format(int w)
{
	return w <= 8 ? "R8_UINT" : w <= 16 ? "R16_UINT" : w <= 32 ? "R32_UINT" :
	    "R32G32_UINT";
}

/* the largest value of the parameter in slot, from its type and range */
static uint64_t
umax_of(const struct rctx *c, int slot)
{
	const struct param *p = &c->f->p[slot];
	int64_t lo, hi;

	if (TYW(p->ty) == 64 && !p->ranged)
		return UINT64_MAX;
	param_bounds(p, &lo, &hi);
	return (uint64_t)hi;
}

static void
logic_ops(struct rctx *c)
{
	static const char *const name[16] = {"CLEAR", "AND", "AND_REVERSE",
	    "COPY", "AND_INVERTED", "NO_OP", "XOR", "OR", "NOR", "EQUIVALENT",
	    "INVERT", "OR_REVERSE", "COPY_INVERTED", "OR_INVERTED", "NAND",
	    "SET"};
	nid s = c->vs, d = c->vd, e;
	int k;
	const char *how;
	char st[320], pl[400];

	if (c->flt || c->nsrc != 1 || c->ic >= 0)
		return;
	for (k = 0; k < 16; k++) {
		nid ns = mk_un(O_NOT, s), nd = mk_un(O_NOT, d);
		switch (k) {
		case 0: e = mk_const_i(c->ty, 0); break;
		case 1: e = mk_bin(O_AND, s, d); break;
		case 2: e = mk_bin(O_AND, s, nd); break;
		case 3: e = s; break;
		case 4: e = mk_bin(O_AND, ns, d); break;
		case 5: e = d; break;
		case 6: e = mk_bin(O_XOR, s, d); break;
		case 7: e = mk_bin(O_OR, s, d); break;
		case 8: e = mk_un(O_NOT, mk_bin(O_OR, s, d)); break;
		case 9: e = mk_un(O_NOT, mk_bin(O_XOR, s, d)); break;
		case 10: e = nd; break;
		case 11: e = mk_bin(O_OR, s, nd); break;
		case 12: e = ns; break;
		case 13: e = mk_bin(O_OR, ns, d); break;
		case 14: e = mk_un(O_NOT, mk_bin(O_AND, s, d)); break;
		default: e = mk_un(O_NOT, mk_const_i(c->ty, 0));
		}
		if (same(c->f, c->f->body, e, &how) != 1)
			continue;
		snprintf(st, sizeof st, "VK_LOGIC_OP_%s on an unsigned integer "
		    "target (%s%s): %s", name[k], uint_format(c->w), c->w == 64 ?
		    ", the low word in R and the high one in G: a logic op works "
		    "bit by bit, so the two channels compute it" : "",
		    shared_str(e, 80));
		snprintf(pl, sizeof pl, "VkPipelineColorBlendStateCreateInfo: "
		    "logicOpEnable, logicOp = VK_LOGIC_OP_%s (the logicOp "
		    "feature); an %s colour attachment; one point per "
		    "contribution, its position the cell, its colour %s", name[k],
		    uint_format(c->w), pname(c, c->is));
		form(c, "logic op", st, "", how, C_ROP_LOGIC, pl);
		return;		/* the 16 functions are distinct */
	}
}

static void
stencil_ops(struct rctx *c)
{
	static const char *const name[8] = {"KEEP", "ZERO", "REPLACE",
	    "INCREMENT_AND_CLAMP", "DECREMENT_AND_CLAMP", "INVERT",
	    "INCREMENT_AND_WRAP", "DECREMENT_AND_WRAP"};
	ty_t t = c->ty, W = TY(K_UINT, 32);
	nid d = c->vd, e, wd = mk_cast(W, d), one = mk_const_i(W, 1);
	int k, m, masks[15], nm = 0;
	const char *how;
	char st[240], pl[400];

	if (c->flt || c->w != 8 || c->nsrc != 0)
		return;
	masks[nm++] = 0xff;
	for (m = 1; m < 8; m++) {
		masks[nm++] = (1 << m) - 1;		/* the low m bits */
		masks[nm++] = 0xff & ~((1 << m) - 1);	/* the rest */
	}
	for (m = 0; m < nm; m++)
		for (k = 0; k < 8; k++) {
			nid r;
			switch (k) {
			case 0: r = d; break;
			case 1: r = mk_const_i(t, 0); break;
			case 2:
				if (c->ic < 0)
					continue;
				r = c->vc;
				break;
			case 3: r = mk_cast(t, mk_bin(O_MIN, mk_bin(O_ADD, wd, one),
			    mk_const_i(W, 255))); break;
			case 4: r = mk_cast(t, mk_bin(O_SUB, mk_bin(O_MAX, wd, one),
			    one)); break;
			case 5: r = mk_un(O_NOT, d); break;
			case 6: r = mk_bin(O_ADD, d, mk_const_i(t, 1)); break;
			default: r = mk_bin(O_SUB, d, mk_const_i(t, 1));
			}
			if (k != 2 && c->ic >= 0)
				continue;	/* the constant would go unused */
			e = masks[m] == 0xff ? r : mk_bin(O_OR, mk_bin(O_AND, d,
			    mk_const_i(t, 0xff & ~masks[m])), mk_bin(O_AND, r,
			    mk_const_i(t, masks[m])));
			if (same(c->f, c->f->body, e, &how) != 1)
				continue;
			snprintf(st, sizeof st, "VK_STENCIL_OP_%s%s on the 8-bit "
			    "stencil value: %s", name[k], masks[m] == 0xff ? "" :
			    " under a write mask", shared_str(e, 100));
			snprintf(pl, sizeof pl, "VkStencilOpState: passOp = "
			    "VK_STENCIL_OP_%s, writeMask = 0x%02x%s%s; a stencil "
			    "attachment (S8_UINT); one fragment per contribution, "
			    "its position the cell", name[k], masks[m], k == 2 ?
			    ", reference = " : "", k == 2 ? pname(c, c->ic) : "");
			form(c, "stencil op", st, "", how, C_ROP_STENCIL, pl);
			return;
		}
}

/* additive and subtractive blending on a UNORM target: integers n/(2^w-1) */
static void
unorm_blend(struct rctx *c)
{
	ty_t t = c->ty, W = TY(K_SINT, 32);
	nid s = c->vs, d = c->vd, ws, wd, e;
	int k;
	const char *how;
	char st[240], pl[400], pr[400];
	static const char *const op[5] = {"ADD", "SUBTRACT", "REVERSE_SUBTRACT",
	    "MIN", "MAX"};

	if (c->flt || c->w > 16 || c->nsrc != 1 || c->ic >= 0)
		return;
	ws = mk_cast(W, s);
	wd = mk_cast(W, d);
	for (k = 0; k < 5; k++) {
		int64_t top = (1LL << c->w) - 1;
		switch (k) {
		case 0: e = mk_bin(O_MIN, mk_bin(O_ADD, ws, wd), mk_const_i(W,
		    top)); break;
		case 1: e = mk_bin(O_MAX, mk_bin(O_SUB, ws, wd), mk_const_i(W,
		    0)); break;
		case 2: e = mk_bin(O_MAX, mk_bin(O_SUB, wd, ws), mk_const_i(W,
		    0)); break;
		case 3: e = mk_bin(O_MIN, s, d); break;
		default: e = mk_bin(O_MAX, s, d);
		}
		e = mk_cast(t, e);
		if (same(c->f, c->f->body, e, &how) != 1)
			continue;
		snprintf(st, sizeof st, "VK_BLEND_OP_%s%s on a normalized target "
		    "(R%d_UNORM), the integer n standing for n/%lld: %s", op[k],
		    k < 3 ? ", factors ONE and ONE," : "", c->w, (long long)top,
		    shared_str(e, 90));
		snprintf(pr, sizeof pr, "%s; %s; exact if the conversion to UNORM "
		    "rounds to nearest (the float's error is far below half a "
		    "step; Vulkan prefers nearest and allows either neighbour: to "
		    "measure)", how, k < 3 ? "the blender clamps the result to [0, "
		    "1], which is the saturation" : "the result is one of the two "
		    "values");
		snprintf(pl, sizeof pl, "VkPipelineColorBlendAttachmentState: "
		    "blendEnable, colorBlendOp = VK_BLEND_OP_%s, "
		    "srcColorBlendFactor = dstColorBlendFactor = "
		    "VK_BLEND_FACTOR_ONE; an R%d_UNORM colour attachment; one "
		    "point per contribution, its position the cell, its colour "
		    "%s/%lld", op[k], c->w, pname(c, c->is), (long long)top);
		form(c, "blending", st, "", pr, C_ROP_BLEND, pl);
		return;
	}
}

/* ---- the blend equation, by polynomial identity ---- */

enum { BF_ZERO, BF_ONE, BF_SRC, BF_OMSRC, BF_DST, BF_OMDST, BF_CONST,
	BF_OMCONST, BF_SRC1, BF_OMSRC1, BF_N };
static const char *const bf_name[BF_N] = {"ZERO", "ONE", "SRC_COLOR",
	"ONE_MINUS_SRC_COLOR", "DST_COLOR", "ONE_MINUS_DST_COLOR",
	"CONSTANT_COLOR", "ONE_MINUS_CONSTANT_COLOR", "SRC1_COLOR",
	"ONE_MINUS_SRC1_COLOR"};

struct beq {
	int as, ad, ac, at;	/* atoms */
	const struct rctx *c;
};

static int
bf_atom(const struct beq *q, int f)
{
	switch (f) {
	case BF_SRC: case BF_OMSRC: return q->as;
	case BF_DST: case BF_OMDST: return q->ad;
	case BF_CONST: case BF_OMCONST: return q->ac;
	case BF_SRC1: case BF_OMSRC1: return q->at;
	}
	return -1;
}

static int
bf_ok(const struct beq *q, int f)
{
	return f <= BF_ONE || bf_atom(q, f) >= 0;
}

static struct mp
bf_mp(const struct beq *q, int f)
{
	if (f == BF_ZERO)
		return mp_zero();
	if (f == BF_ONE)
		return mp_const(qi(1));
	if (f & 1)	/* 1 - x */
		return mp_sub(mp_const(qi(1)), mp_atom(bf_atom(q, f)));
	return mp_atom(bf_atom(q, f));
}

static void
bf_text(const struct beq *q, int f, const char *x, struct sbuf *b)
{
	const struct rctx *c = q->c;
	const char *v = f == BF_SRC || f == BF_OMSRC ? pname(c, c->is) :
	    f == BF_DST || f == BF_OMDST ? pname(c, c->id) : f == BF_CONST ||
	    f == BF_OMCONST ? pname(c, c->ic) : f >= BF_SRC1 ? pname(c, c->it) :
	    "";

	if (f == BF_ZERO)
		sb_puts(b, "0");
	else if (f == BF_ONE)
		sb_puts(b, x);
	else if (f & 1)
		sb_printf(b, "%s*(1 - %s)", x, v);
	else
		sb_printf(b, "%s*%s", x, v);
}

/* how many operations the blender does for these factors and this op */
static int
bf_ops(int sf, int df)
{
	int n = 0;

	n += sf > BF_ONE;
	n += sf > BF_ONE && (sf & 1);
	n += df > BF_ONE;
	n += df > BF_ONE && (df & 1);
	n += sf != BF_ZERO && df != BF_ZERO;
	return n;
}

/* [lo, hi] of a factor's value from the parameters' integer ranges */
static void
bf_range(const struct beq *q, int f, const double *lo, const double *hi,
    double *flo, double *fhi)
{
	int a = bf_atom(q, f);

	if (f == BF_ZERO)
		*flo = *fhi = 0;
	else if (f == BF_ONE)
		*flo = *fhi = 1;
	else if (f & 1) {
		*flo = 1 - hi[a];
		*fhi = 1 - lo[a];
	} else {
		*flo = lo[a];
		*fhi = hi[a];
	}
}

static void
mulr(double alo, double ahi, double blo, double bhi, double *lo, double *hi)
{
	double x[4] = {alo * blo, alo * bhi, ahi * blo, ahi * bhi};
	int j;

	*lo = *hi = x[0];
	for (j = 1; j < 4; j++) {
		*lo = x[j] < *lo ? x[j] : *lo;
		*hi = x[j] > *hi ? x[j] : *hi;
	}
}

/* an integer-valued double as text: exact below 2^53, else rounded */
static void
num53(double x, char *b, size_t n)
{
	if (fabs(x) < 9007199254740992.0)
		snprintf(b, n, "%.0f", x);
	else
		snprintf(b, n, "about %.6g", x);
}

/* is e only +, -, * of the parameters and constants (integers) */
static int
int_poly(nid e, int depth)
{
	int op = N.op[e];

	if (depth > 60)
		return 0;
	if (op == O_VAR || (op == O_CONST && !TY_FP(N.ty[e])))
		return 1;
	if (op == O_NEG)
		return int_poly(N.a[e], depth + 1);
	if (op == O_ADD || op == O_SUB || op == O_MUL)
		return int_poly(N.a[e], depth + 1) && int_poly(N.b[e], depth + 1);
	return 0;
}

static void
blend_eq(struct rctx *c)
{
	static const char *const opn[3] = {"ADD", "SUBTRACT", "REVERSE_SUBTRACT"};
	struct beq q;
	struct mp F;
	size_t mk;
	nid body;
	int op, sf, df, best = -1, bop = 0, bsf = 0, bdf = 0, more = 0, j;
	double lo[AV] = {0}, hi[AV] = {0};
	struct sbuf eq = {0}, st = {0}, holds = {0}, pr = {0}, pl = {0};
	char x[32];

	if (c->nsrc < 1)
		return;
	if (!c->flt && !int_poly(c->f->body, 0))
		return;
	body = as_real(c, c->f->body);
	if (!body)
		return;
	mk = ar_mark();
	atoms_reset();
	q.c = c;
	q.as = atom_of(mk_var((uint32_t)c->is, c->f->p[c->is].name, TY_REAL));
	q.ad = atom_of(mk_var((uint32_t)c->id, c->f->p[c->id].name, TY_REAL));
	q.ac = c->ic < 0 ? -1 : atom_of(mk_var((uint32_t)c->ic,
	    c->f->p[c->ic].name, TY_REAL));
	q.at = c->it < 0 ? -1 : atom_of(mk_var((uint32_t)c->it,
	    c->f->p[c->it].name, TY_REAL));
	j = nat;
	if (!mp_of(body, &F) || nat != j) {
		/* not a polynomial in the parameters alone */
		ar_release(mk);
		return;
	}
	for (op = 0; op < 3; op++)
		for (sf = 0; sf < BF_N; sf++)
			for (df = 0; df < BF_N; df++) {
				struct mp a, b, r;
				int n;
				if (!bf_ok(&q, sf) || !bf_ok(&q, df))
					continue;
				a = mp_mul(mp_atom(q.as), bf_mp(&q, sf));
				b = mp_mul(mp_atom(q.ad), bf_mp(&q, df));
				r = op == 0 ? mp_add(a, b) : op == 1 ? mp_sub(a, b) :
				    mp_sub(b, a);
				if (alg_ovf || !mp_eq(r, F))
					continue;
				n = bf_ops(sf, df);
				if (best < 0 || n < best) {
					more += best >= 0;
					best = n;
					bop = op;
					bsf = sf;
					bdf = df;
				} else
					more++;
			}
	if (best < 0) {
		ar_release(mk);
		return;
	}
	/* the equation as the blender computes it */
	{
		struct sbuf ta = {0}, tb = {0};
		bf_text(&q, bsf, pname(c, c->is), &ta);
		bf_text(&q, bdf, pname(c, c->id), &tb);
		if (bop == 0)
			sb_printf(&eq, "%s + %s", sb_str(&ta), sb_str(&tb));
		else if (bop == 1)
			sb_printf(&eq, "%s - %s", sb_str(&ta), sb_str(&tb));
		else
			sb_printf(&eq, "%s - %s", sb_str(&tb), sb_str(&ta));
		free(ta.p);
		free(tb.p);
	}
	sb_printf(&st, "VK_BLEND_OP_%s, source factor %s, destination factor "
	    "%s on a float target (R32_SFLOAT): %s", opn[bop], bf_name[bsf],
	    bf_name[bdf], sb_str(&eq));
	if (more)
		sb_printf(&st, " (%d more pair%s of factors give the same)", more,
		    more == 1 ? "" : "s");
	sb_printf(&pr, "%s and %s expand to the same polynomial", c->f->anon ?
	    "the update" : sym_str(c->f->name), sb_str(&eq));
	if (!c->flt) {
		/* integers: exact in f32 while everything stays below 2^24 */
		double flo, fhi, alo, ahi, blo, bhi, rlo, rhi, mx = 0;
		int ok = 1, atoms[4] = {q.as, q.ad, q.ac, q.at};
		int slots[4] = {c->is, c->id, c->ic, c->it};
		for (j = 0; j < 4; j++) {
			int64_t l, h;
			if (atoms[j] < 0)
				continue;
			param_bounds(&c->f->p[slots[j]], &l, &h);
			lo[atoms[j]] = (double)l;
			hi[atoms[j]] = (double)umax_of(c, slots[j]);
			mx = fmax(mx, fmax(fabs((double)l), hi[atoms[j]]));
		}
		bf_range(&q, bsf, lo, hi, &flo, &fhi);
		mx = fmax(mx, fmax(fabs(flo), fabs(fhi)));
		mulr(lo[q.as], hi[q.as], flo, fhi, &alo, &ahi);
		bf_range(&q, bdf, lo, hi, &flo, &fhi);
		mx = fmax(mx, fmax(fabs(flo), fabs(fhi)));
		mulr(lo[q.ad], hi[q.ad], flo, fhi, &blo, &bhi);
		if (bop == 0) {
			rlo = alo + blo;
			rhi = ahi + bhi;
		} else if (bop == 1) {
			rlo = alo - bhi;
			rhi = ahi - blo;
		} else {
			rlo = blo - ahi;
			rhi = bhi - alo;
		}
		mx = fmax(mx, fmax(fmax(fabs(alo), fabs(ahi)), fmax(fabs(blo),
		    fabs(bhi))));
		mx = fmax(mx, fmax(fabs(rlo), fabs(rhi)));
		ok = mx < P24 && rlo >= 0 && rhi < ldexp(1.0, c->w);
		if (!ok) {
			char nx[40], nl[40], nh[40];
			num53(mx, nx, sizeof nx);
			num53(rlo, nl, sizeof nl);
			num53(rhi, nh, sizeof nh);
			say(c, "  blending: VK_BLEND_OP_%s with %s and %s would "
			    "compute it (%s), but is not exact on this domain: its "
			    "values reach %s in magnitude and its result [%s, %s]; "
			    "f32 holds integers below 2^24 = 16777216, and the result "
			    "must stay inside u%d\n", opn[bop], bf_name[bsf],
			    bf_name[bdf], sb_str(&eq), nx, nl, nh, c->w);
			if (!c->quiet)
				rep_strf("not_exact", "blending %s: values up to %s, "
				    "result in [%s, %s]", sb_str(&eq), nx, nl, nh);
			else if (!c->best->near[0]) {
				c->best->bound = 1;
				snprintf(c->best->near, sizeof c->best->near,
				    "VK_BLEND_OP_%s with %s and %s computes it (%s), "
				    "exactly while its values stay below 2^24 = 16777216 "
				    "and its result inside u%d: here they reach %s, the "
				    "result [%s, %s]", opn[bop], bf_name[bsf],
				    bf_name[bdf], sb_str(&eq), c->w, nx, nl, nh);
			}
			goto out;
		}
		sb_printf(&holds, "the operands, the factors, both products and "
		    "the result stay below 2^24 in magnitude (at most %.0f, the "
		    "result in [%.0f, %.0f], from the parameters' ranges)", mx,
		    rlo, rhi);
		sb_printf(&pr, "; every value is an integer below 2^24, which "
		    "f32 holds, so no step of the blender rounds whatever its "
		    "arithmetic, and the integer result (inside u%d: no wrap) is "
		    "the polynomial's", c->w);
	} else if (best <= 1)
		sb_printf(&pr, "; one f32 operation: exact if the blender rounds "
		    "it to nearest (Vulkan asks for the target's precision, not "
		    "the rounding: to measure)");
	else
		sb_printf(&pr, "; the blender does %d operations whose rounding "
		    "Vulkan does not fix (it may fuse the multiply and the add): "
		    "equal as real functions, bit for bit only where every step "
		    "is exact", best);
	sb_printf(&pl, "VkPipelineColorBlendAttachmentState: blendEnable, "
	    "colorBlendOp = VK_BLEND_OP_%s, srcColorBlendFactor = "
	    "VK_BLEND_FACTOR_%s, dstColorBlendFactor = VK_BLEND_FACTOR_%s",
	    opn[bop], bf_name[bsf], bf_name[bdf]);
	if (bsf == BF_CONST || bsf == BF_OMCONST || bdf == BF_CONST ||
	    bdf == BF_OMCONST)
		sb_printf(&pl, ", blendConstants = %s", pname(c, c->ic));
	if (bsf >= BF_SRC1 || bdf >= BF_SRC1)
		sb_printf(&pl, " (the dualSrcBlend feature: %s is the fragment's "
		    "second output, index 1)", pname(c, c->it));
	snprintf(x, sizeof x, "%s", pname(c, c->is));
	sb_printf(&pl, "; an R32_SFLOAT colour attachment; one point per "
	    "contribution, its position the cell, its colour %s", x);
	form(c, "blending", sb_str(&st), sb_str(&holds), sb_str(&pr),
	    C_ROP_BLEND, sb_str(&pl));
out:
	ar_release(mk);
	free(eq.p);
	free(st.p);
	free(holds.p);
	free(pr.p);
	free(pl.p);
}

/* ---- floats: min, max, the advanced blend operations ---- */

static int
unit_range(const struct rctx *c, int slot)
{
	const struct param *p = &c->f->p[slot];

	return p->ranged && p->rlo >= 0 && p->rhi <= 1;
}

static void
minmax_blend(struct rctx *c)
{
	nid b = c->f->body;
	int mn = N.op[b] == O_MIN;
	char st[200], pl[400];

	if (!c->flt || c->nsrc != 1 || c->ic >= 0 || (N.op[b] != O_MIN &&
	    N.op[b] != O_MAX))
		return;
	if (!((N.a[b] == c->vs && N.b[b] == c->vd) || (N.a[b] == c->vd &&
	    N.b[b] == c->vs)))
		return;
	snprintf(st, sizeof st, "VK_BLEND_OP_%s on a float target "
	    "(R32_SFLOAT): %s (the factors are not used)", mn ? "MIN" : "MAX",
	    shared_str(b, 60));
	snprintf(pl, sizeof pl, "VkPipelineColorBlendAttachmentState: "
	    "blendEnable, colorBlendOp = VK_BLEND_OP_%s; an R32_SFLOAT colour "
	    "attachment; one point per contribution, its position the cell, "
	    "its colour %s", mn ? "MIN" : "MAX", pname(c, c->is));
	form(c, "blending", st, "neither value is NaN; the sign of a zero "
	    "result may differ", "the same operation: the result is one of the "
	    "two values, nothing rounds", C_ROP_BLEND, pl);
}

static void
advanced(struct rctx *c)
{
	static const char *const name[6] = {"MULTIPLY", "SCREEN", "DARKEN",
	    "LIGHTEN", "DIFFERENCE", "EXCLUSION"};
	nid s, d, body, e[2];
	struct mp F, G;
	size_t mk;
	int k, v, hit = -1;
	char st[240], pl[400], pr[300];
	const char *holds;

	if (!c->flt || c->nsrc != 1 || c->ic >= 0)
		return;
	body = as_real(c, c->f->body);
	if (!body)
		return;
	s = mk_var((uint32_t)c->is, c->f->p[c->is].name, TY_REAL);
	d = mk_var((uint32_t)c->id, c->f->p[c->id].name, TY_REAL);
	mk = ar_mark();
	for (k = 0; k < 6 && hit < 0; k++) {
		e[1] = 0;
		switch (k) {
		case 0: e[0] = mk_bin(O_MUL, s, d); break;
		case 1: e[0] = mk_bin(O_SUB, mk_bin(O_ADD, s, d), mk_bin(O_MUL, s,
		    d)); break;
		case 2: e[0] = mk_bin(O_MIN, s, d); e[1] = mk_bin(O_MIN, d, s);
			break;
		case 3: e[0] = mk_bin(O_MAX, s, d); e[1] = mk_bin(O_MAX, d, s);
			break;
		case 4: e[0] = mk_un(O_ABS, mk_bin(O_SUB, s, d));
			e[1] = mk_un(O_ABS, mk_bin(O_SUB, d, s)); break;
		default: e[0] = mk_bin(O_SUB, mk_bin(O_ADD, s, d), mk_bin(O_MUL,
		    mk_const_r(2), mk_bin(O_MUL, s, d)));
		}
		for (v = 0; v < 2 && hit < 0; v++) {
			if (!e[v])
				continue;
			atoms_reset();
			if (mp_of(body, &F) && mp_of(e[v], &G) && !alg_ovf &&
			    mp_eq(F, G))
				hit = k;
		}
	}
	ar_release(mk);
	if (hit < 0)
		return;
	snprintf(st, sizeof st, "VK_BLEND_OP_%s_EXT on opaque colours "
	    "(VK_EXT_blend_operation_advanced): %s", name[hit], hit == 0 ?
	    "s*d" : hit == 1 ? "s + d - s*d" : hit == 2 ? "min(s, d)" :
	    hit == 3 ? "max(s, d)" : hit == 4 ? "|s - d|" : "s + d - 2*s*d");
	holds = unit_range(c, c->is) && unit_range(c, c->id) ? "the colours "
	    "are in [0, 1] (the parameters' ranges) and opaque" : NULL;
	if (!holds && c->nforms)
		return;		/* a state above computes it already */
	if (!holds) {
		say(c, "  advanced blending: VK_BLEND_OP_%s_EXT would compute it, "
		    "but is defined on colours in [0, 1]: give %s and %s that "
		    "range (%s: f32 in 0..1)\n", name[hit], pname(c, c->is),
		    pname(c, c->id), pname(c, c->is));
		if (!c->quiet)
			rep_strf("needs_range", "VK_BLEND_OP_%s_EXT needs colours in "
			    "[0, 1]", name[hit]);
		return;
	}
	snprintf(pr, sizeof pr, "the update and the operation's f(s, d) expand "
	    "to the same polynomial%s; the blender's rounding is not fixed by "
	    "the extension: equal as real functions, bit for bit to measure",
	    hit >= 2 && hit <= 4 ? " (min, max and |.| taken whole)" : "");
	snprintf(pl, sizeof pl, "VkPipelineColorBlendAttachmentState: "
	    "blendEnable, colorBlendOp = alphaBlendOp = VK_BLEND_OP_%s_EXT "
	    "(the VK_EXT_blend_operation_advanced extension, alpha 1 on both "
	    "sides); a UNORM or float colour attachment; one point per "
	    "contribution, its position the cell, its colour %s", name[hit],
	    pname(c, c->is));
	form(c, "advanced blending", st, holds, pr, C_ROP_ADV, pl);
}

/* ---- the depth test: a min or a max that keeps a payload ---- */

/*
 * A key k is written as the depth whose f32 bits are k + 2^23: a normal
 * float (no flush of denormals can touch it) in [2^-126, 1), so no clamp
 * to the viewport's depth range either, and increasing with k while
 * k < DEPTH_KEYS.  D32_SFLOAT stores it as written, and the depth compare
 * of two such floats is the compare of their keys: nothing rounds.
 */
#define DEPTH_KEYS 0x3F000000ull

static void
depth_test(struct rctx *c)
{
	nid s = c->vs, d = c->vd, e, g, keep, ks, kd, cond;
	ty_t t = c->ty;
	int mx, b, w = c->w;
	uint64_t top, kmax, mask;
	const char *how, *cmp, *fmt;
	char st[400], hold[400], pr[700], pl[700], key[80], pay[80], gt[120];

	if (c->flt || c->nsrc != 1 || c->ic >= 0)
		return;
	/* F is the min or the max of s and d on F's domain */
	for (mx = 0; mx < 2; mx++)
		if (same(c->f, c->f->body, mk_bin(mx ? O_MAX : O_MIN, s, d), &how)
		    == 1)
			break;
	if (mx == 2)
		return;
	/* the key: the fewest low bits left out so that it fits */
	top = umax_of(c, c->is) > umax_of(c, c->id) ? umax_of(c, c->is) :
	    umax_of(c, c->id);
	for (b = 0; b < w && (top >> b) >= DEPTH_KEYS; b++)
		;
	kmax = top >> b;
	mask = b == 64 ? UINT64_MAX : ((uint64_t)1 << b) - 1;
	/* LESS keeps the cell on a tie, GREATER_OR_EQUAL takes the new one */
	cmp = mx ? (b ? "GREATER_OR_EQUAL" : "GREATER") : "LESS";
	if (b == 0) {
		snprintf(key, sizeof key, "%s", pname(c, c->is));
		snprintf(st, sizeof st, "VK_COMPARE_OP_%s on a D32_SFLOAT depth "
		    "attachment, the value its own key: %s(%s, %s)", cmp, mx ?
		    "max" : "min", pname(c, c->is), pname(c, c->id));
		snprintf(hold, sizeof hold, "the values stay below 0x3F000000 "
		    "(at most %llu, from the parameters' ranges)", (unsigned long
		    long)top);
		snprintf(pr, sizeof pr, "the update is %s(%s, %s) (%s); a key's "
		    "depth is the f32 whose bits are key + 2^23, normal, in "
		    "[2^-126, 1) and increasing with the key, so the depth "
		    "compare is the keys' integer compare and nothing rounds; on "
		    "a tie both values are the same", mx ? "max" : "min",
		    pname(c, c->is), pname(c, c->id), how);
		snprintf(pl, sizeof pl, "VkPipelineDepthStencilStateCreateInfo: "
		    "depthTestEnable, depthWriteEnable, depthCompareOp = "
		    "VK_COMPARE_OP_%s; a D32_SFLOAT depth attachment cleared to "
		    "(or copied from) the cells' keys; one point per "
		    "contribution, its position the cell, gl_FragDepth = "
		    "uintBitsToFloat(%s + 0x00800000); the cell's value is the "
		    "depth's bits - 0x00800000", cmp, pname(c, c->is));
		form(c, "depth test", st, hold, pr, C_ROP_DEPTH, pl);
		return;
	}
	/*
	 * with a payload: the test keeps (ks cmp kd) ? s : d, which is the
	 * min or the max wherever the keys differ or the payloads come in
	 * increasing order; proved there, and refuted where they do not
	 */
	ks = mk_bin(O_SHR, s, mk_const_i(t, b));
	kd = mk_bin(O_SHR, d, mk_const_i(t, b));
	keep = mx ? mk_bin(O_GE, ks, kd) : mk_bin(O_LT, ks, kd);
	g = mk_sel(keep, s, d);
	cond = mk_bin(O_LOR, mk_bin(O_NE, ks, kd), mk_bin(O_GT, mk_bin(O_AND,
	    s, mk_const_i(t, (int64_t)mask)), mk_bin(O_AND, d, mk_const_i(t,
	    (int64_t)mask))));
	e = mk_sel(cond, g, c->f->body);
	if (same(c->f, c->f->body, e, &how) != 1)
		return;
	fmt = uint_format(b);
	snprintf(gt, sizeof gt, "%s >> %d %s %s >> %d ? %s : %s", pname(c,
	    c->is), b, mx ? ">=" : "<", pname(c, c->id), b, pname(c, c->is),
	    pname(c, c->id));
	snprintf(key, sizeof key, "%s >> %d", pname(c, c->is), b);
	snprintf(pay, sizeof pay, "%s & 0x%llx", pname(c, c->is),
	    (unsigned long long)mask);
	snprintf(st, sizeof st, "VK_COMPARE_OP_%s on a D32_SFLOAT depth "
	    "attachment holding the key %s, the payload %s in an %s colour "
	    "attachment written where the test passes: %s(%s, %s)", cmp, key,
	    pay, fmt, mx ? "max" : "min", pname(c, c->is), pname(c, c->id));
	snprintf(hold, sizeof hold, "two values that meet in a cell differ in "
	    "their keys, or come in increasing order of payload (the payload "
	    "the contribution's index in the draw: a cell's fragments pass in "
	    "primitive order); the keys below 0x3F000000 (at most %llu, from "
	    "the parameters' ranges)", (unsigned long long)kmax);
	if (!mx)
		snprintf(pr, sizeof pr, "the test keeps %s, which is min(%s, %s) "
		    "wherever the keys differ or the new payload is the larger "
		    "(%s); on a tie the test keeps the cell, and min keeps the "
		    "smaller payload, hence the order; a key's depth is the f32 "
		    "whose bits are key + 2^23, normal, in [2^-126, 1) and "
		    "increasing with the key, so the depth compare is the keys' "
		    "integer compare and nothing rounds", gt,
		    pname(c, c->is), pname(c, c->id), how);
	else
		snprintf(pr, sizeof pr, "the test keeps %s, which is max(%s, %s) "
		    "wherever the keys differ or the new payload is the larger "
		    "(%s); on a tie GREATER_OR_EQUAL takes the new value, and max "
		    "the larger payload, hence the order; a key's depth is the "
		    "f32 whose bits are key + 2^23, normal, in [2^-126, 1) and "
		    "increasing with the key, so the depth compare is the keys' "
		    "integer compare and nothing rounds", gt,
		    pname(c, c->is), pname(c, c->id), how);
	snprintf(pl, sizeof pl, "VkPipelineDepthStencilStateCreateInfo: "
	    "depthTestEnable, depthWriteEnable, depthCompareOp = "
	    "VK_COMPARE_OP_%s; a D32_SFLOAT depth attachment and an %s colour "
	    "attachment (blending off), cleared to (or copied from) the cells' "
	    "keys and payloads; one point per contribution, its position the "
	    "cell, gl_FragDepth = uintBitsToFloat(uint(%s) + 0x00800000), its "
	    "colour %s; the cell's value is (key << %d) | payload, the key the "
	    "depth's bits - 0x00800000", cmp, fmt, key, pay, b);
	form(c, "depth test", st, hold, pr, C_ROP_DEPTH, pl);
}

/* ---- the command ---- */

static int
native_atomic(const struct rctx *c)
{
	nid b = c->f->body;
	int op = N.op[b];

	if (c->nsrc != 1 || c->ic >= 0)
		return 0;
	if (!(op == O_ADD || op == O_MIN || op == O_MAX || (!c->flt &&
	    (op == O_AND || op == O_OR || op == O_XOR))))
		return 0;
	return (N.a[b] == c->vs && N.b[b] == c->vd) || (N.a[b] == c->vd &&
	    N.b[b] == c->vs);
}

/*
 * The roles of f's parameters and what the update is priced against.
 * NULL, or why rop cannot take f (the text of the command's error).
 */
static const char *
rop_setup(struct rctx *c, struct func *f, struct model *m)
{
	static char why[200];
	int j;

	c->f = f;
	c->m = m;
	c->is = c->it = c->ic = -1;
	if (f->np < 1)
		return "rop needs an update F(s, d): d (the parameter of that "
		    "name, or dst, else the last one) is the cell's value, s what "
		    "a fragment brings";
	if (f->where)
		return "rop does not take a where clause: give the parameters "
		    "ranges (s: u32 in 0..255)";
	/* the cell: the parameter named d or dst, else the last one */
	c->id = f->np - 1;
	for (j = 0; j < f->np; j++)
		if (!strcmp(sym_str(f->p[j].name), "d") ||
		    !strcmp(sym_str(f->p[j].name), "dst"))
			c->id = j;
	c->ty = f->p[c->id].ty;
	c->flt = c->ty == TY_F32;
	c->w = TYW(c->ty);
	if (!c->flt && !(TYK(c->ty) == K_UINT && (c->w == 8 || c->w == 16 ||
	    c->w == 32 || c->w == 64))) {
		snprintf(why, sizeof why, "rop works on u8, u16, u32, u64 (integer "
		    "targets, stencil, depth) and f32 (float targets), not %s",
		    ty_str(c->ty));
		return why;
	}
	if (N.ty[f->body] != c->ty) {
		snprintf(why, sizeof why, "the update's result (%s) must have the "
		    "cell's type (%s)", ty_str(N.ty[f->body]), ty_str(c->ty));
		return why;
	}
	for (j = 0; j < f->np; j++) {
		if (j == c->id)
			continue;
		if (f->p[j].ty != c->ty) {
			snprintf(why, sizeof why, "%s is %s and the cell %s: one type "
			    "for the sources, the constant and the cell",
			    sym_str(f->p[j].name), ty_str(f->p[j].ty), ty_str(c->ty));
			return why;
		}
		if (f->p[j].uniform) {
			if (c->ic >= 0)
				return "rop takes one uniform parameter (the blend "
				    "constant or the stencil reference)";
			c->ic = j;
		} else if (c->is < 0)
			c->is = j;
		else if (c->it < 0)
			c->it = j;
		else
			return "rop takes at most two sources (dual-source blending)";
	}
	if (f->p[c->id].uniform) {
		snprintf(why, sizeof why, "%s is the cell's value: it cannot be "
		    "uniform", sym_str(f->p[c->id].name));
		return why;
	}
	c->nsrc = (c->is >= 0) + (c->it >= 0);
	c->vd = mk_var((uint32_t)c->id, f->p[c->id].name, c->ty);
	if (c->is >= 0)
		c->vs = mk_var((uint32_t)c->is, f->p[c->is].name, c->ty);
	if (c->it >= 0)
		c->vt = mk_var((uint32_t)c->it, f->p[c->it].name, c->ty);
	if (c->ic >= 0)
		c->vc = mk_var((uint32_t)c->ic, f->p[c->ic].name, c->ty);
	/*
	 * what it is priced against: the same update as a global atomic,
	 * by the model's memory kinds (memk.c: a float add and a
	 * compare-and-swap have classes of their own, priced as
	 * atomic_global by a model that does not list them)
	 */
	if (c->w == 64) {
		float b = native_atomic(c) ? 0 : fuse_expr_cost(m, f->body);
		c->against = model_cost(m, C_ATOM_GL_64) + b;
		if (native_atomic(c))
			sb_printf(&c->againstwhy, "one global 64-bit atomic, "
			    "atomic_global_64");
		else
			sb_printf(&c->againstwhy, "a global 64-bit compare-and-swap "
			    "around the update: atomic_global_64 %g + %g",
			    (double)model_cost(m, C_ATOM_GL_64), (double)b);
	} else if (native_atomic(c) && !(c->flt && N.op[f->body] == O_ADD)) {
		c->against = m->c[C_ATOM_GL];
		sb_printf(&c->againstwhy, "one global atomic, atomic_global");
	} else if (native_atomic(c)) {
		c->against = model_cost(m, C_ATOM_GL_F);
		sb_printf(&c->againstwhy, "one global float atomic add, "
		    "atomic_global_f");
	} else {
		float b = fuse_expr_cost(m, f->body);
		c->against = model_cost(m, C_ATOM_GL_CAS) + b;
		sb_printf(&c->againstwhy, "a global compare-and-swap around the "
		    "update: atomic_global_cas %g + %g", (double)model_cost(m,
		    C_ATOM_GL_CAS), (double)b);
	}
	return NULL;
}

static void
catalogue(struct rctx *c)
{
	logic_ops(c);
	stencil_ops(c);
	unorm_blend(c);
	minmax_blend(c);
	blend_eq(c);
	advanced(c);
	depth_test(c);
}

static void
cmd_rop(struct func *f, struct model *m)
{
	struct rctx c;
	const char *fn = f->anon ? "the update" : sym_str(f->name), *why;

	memset(&c, 0, sizeof c);
	why = rop_setup(&c, f, m);
	if (why) {
		free(c.againstwhy.p);
		fail("%s", why);
	}
	{
		struct sbuf who = {0};
		if (c.is >= 0)
			sb_printf(&who, "%s what a fragment brings, ", pname(&c, c.is));
		if (c.it >= 0)
			sb_printf(&who, "%s its second value, ", pname(&c, c.it));
		sb_printf(&who, "%s the cell's value", pname(&c, c.id));
		if (c.ic >= 0)
			sb_printf(&who, ", %s a pipeline constant", pname(&c, c.ic));
		printf("rop %s: %s = %s, applied once per fragment (%s)\n",
		    f->anon ? "F" : sym_str(f->name), fn, shared_str(f->body, 200),
		    sb_str(&who));
		free(who.p);
	}
	rep_begin("rop");
	rep_expr("update", f->body);
	rep_str("model", sym_str(m->name));
	order(&c);
	catalogue(&c);
	if (c.nforms)
		rep_end();
	else
		printf("  no fixed-function state in the catalogue computes it "
		    "(logic ops, stencil ops, the blend equation with its "
		    "factors, min, max, the advanced blend operations, the depth "
		    "test)\n");
	if (c.nforms && c.ordered && !c.flt)
		printf("  as the result depends on the order, a state above "
		    "stands for a sequential loop over the fragments in "
		    "primitive order, not for unordered atomics\n");
	if (c.nforms)
		printf("  not in any price: the draw's setup (a render pass, the "
		    "pipeline, one vertex per contribution) and reading the "
		    "target back\n");
	rep_int("count", c.nforms);
	free(c.againstwhy.p);
}

/*
 * simplify's question (spv_mem.inc): the cheapest state of the catalogue
 * that computes f and that m prices below f as a global atomic.  1 found
 * (in *o), 0 none (o->near may say what would), -1 f is not an update rop
 * takes (o->near says why).  Nothing is printed or reported.
 */
int
rop_offer(struct func *f, struct model *m, struct rop_offer *o)
{
	struct rctx c;
	jmp_buf saved;
	volatile int r = -1;
	int ron = rep_on;
	const char *why;

	memset(o, 0, sizeof *o);
	memset(&c, 0, sizeof c);
	c.quiet = 1;
	c.best = o;
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	rep_on = 0;
	if (setjmp(fail_jmp) == 0) {
		why = rop_setup(&c, f, m);
		if (why)
			snprintf(o->near, sizeof o->near, "%s", why);
		else {
			catalogue(&c);
			r = o->found;
		}
	} else
		snprintf(o->near, sizeof o->near, "rop could not read the update");
	rep_on = ron;
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	free(c.againstwhy.p);
	return r;
}

int
rop_command(struct P *p)
{
	static const char *const stop[] = {"model", "under", NULL};
	uint32_t at = cmdp_pos(p);
	struct func *f;
	struct model *m = model_active();
	char w[64];

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (strcmp(w, "rop") != 0 || func_find(sym_cstr(w)) || cmdp_eos(p)) {
		cmdp_seek(p, at);
		return 0;
	}
	f = cmdp_func(p, stop);
	if (cmdp_kw(p, "model") || cmdp_kw(p, "under")) {
		if (!cmdp_word(p, w, sizeof w) || (m = model_find(sym_cstr(w))) ==
		    NULL)
			fail("unknown model (see 'models')");
	}
	if (!cmdp_eos(p))
		fail("rop F [model M]");
	cmd_rop(f, m);
	return 1;
}
