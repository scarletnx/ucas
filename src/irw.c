/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Integer rewrites that seed find: identities that hold on the domain
 * the user gave (the parameters' ranges, the where clause), each with
 * its derivation and the conditions it holds under, proved from the
 * domain by interval arithmetic or by bound.c's monotonicity proofs.
 * Nothing here decides whether a rewrite is worth it: find keeps one
 * only when the active cost model prices it lower, and then checks it
 * again on the domain (exhaustively or by SAT) as a second opinion.
 *
 *   - a signed division or remainder whose operands are never negative
 *     (the divisor at least 1) is the unsigned one;
 *   - x < y ? x : y is min(x, y), and the other orders and arms (any
 *     integers: a total order);
 *   - a clamp written with selects, y > x ? y : min(x, h), is max(y,
 *     min(x, h)) when y <= h (and the mirror image);
 *   - a division (or remainder) by a variable through f32: q0 =
 *     trunc(f32(a) / f32(b)), r0 = a - q0 b, then one step each way,
 *     exact when a, b < 2^24 and 5 * 2^-24 * max(a/b) < 1 (Vulkan's
 *     2.5 ulp division).
 */
#include "calc.h"
#include "bound.h"
#include "fuse.h"
#include "irw.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- ranges of subterms on f's domain ---- */

struct ivc {
	const struct func *f;
	struct func g;
	struct plan p;
	struct ival *iv;
	nid root;
};

static void
ivc_open(struct ivc *c, const struct func *f, nid root)
{
	int64_t lo[MAXP], hi[MAXP];
	int j;

	memset(c, 0, sizeof *c);
	c->f = f;
	c->g = *f;
	c->root = root;
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty))
			param_bounds(&f->p[j], &lo[j], &hi[j]);
	where_bounds(f, lo, hi, NULL);
	for (j = 0; j < f->np; j++)
		if (TY_ISINT(f->p[j].ty) && !(TYK(f->p[j].ty) == K_UINT &&
		    TYW(f->p[j].ty) == 64)) {
			c->g.p[j].ranged = 1;
			c->g.p[j].lo = lo[j];
			c->g.p[j].hi = hi[j];
		}
	plan_build(&c->p, &root, 1, 0);
	c->iv = xcalloc(c->p.n ? c->p.n : 1, sizeof *c->iv);
	plan_ival(&c->p, &c->g, c->iv);
}

static void
ivc_close(struct ivc *c)
{
	free(c->iv);
	plan_free(&c->p);
}

/*
 * s's values on the domain lie in [*lo, *hi]: from the intervals, else
 * (a lower end that is not enough) from bound.c's proof.  why says how.
 */
static int
range_of(struct ivc *c, nid s, int64_t need_lo, int64_t *lo, int64_t *hi,
    char *why, size_t nwhy)
{
	uint32_t k;
	const struct ival *v;

	if (is_const(s)) {
		*lo = *hi = N.ki[s];
		snprintf(why, nwhy, "a constant");
		return 1;
	}
	k = plan_loc(&c->p, s);
	if (k == NOTOK)
		return 0;
	v = &c->iv[k];
	if (v->top)
		return 0;
	*lo = v->lo;
	*hi = v->hi;
	if (N.op[s] == O_VAR)
		snprintf(why, nwhy, "its range");
	else
		snprintf(why, nwhy, "interval arithmetic over the parameters' "
		    "ranges");
	if (*lo < need_lo && c->f->np > 0 && TY_ISINT(N.ty[s])) {
		struct func h = c->g;
		struct pbound pb;
		h.body = s;
		h.anon = 1;
		if (prove_bounds(&h, &pb) && pb.have_lo && pb.lo >= (double)need_lo
		    && pb.lo > (double)*lo) {
			const char *w = sb_str(&pb.why_lo);
			const char *nl = strchr(w, '\n');
			*lo = (int64_t)ceil(pb.lo);
			snprintf(why, nwhy, "proved: %.*s", nl ? (int)(nl - w) :
			    (int)strlen(w), w);
		}
		pbound_free(&pb);
	}
	return 1;
}

/* ---- candidates ---- */

static void
add(struct irw **out, int *n, int *cap, nid root, nid from, nid to,
    const char *what, const char *cond, const char *proof, int keep)
{
	struct irw *r;
	nid e;

	if (!to || to == from)
		return;
	e = xreplace(root, from, to);
	if (!e || e == root)
		return;
	if (*n == *cap) {
		*cap = *cap ? *cap * 2 : 16;
		*out = xrealloc(*out, (size_t)*cap * sizeof **out);
	}
	r = &(*out)[(*n)++];
	memset(r, 0, sizeof *r);
	r->e = e;
	r->at = from;
	r->to = to;
	snprintf(r->what, sizeof r->what, "%s", what);
	snprintf(r->cond, sizeof r->cond, "%s", cond);
	r->proof = strdup(proof);
	r->nearmiss = !keep;
	r->loose = 0;
}

static const char *
sstr(nid e)
{
	return shared_str(e, 60);
}

static ty_t
unsigned_of(ty_t t, int64_t ahi, int64_t bhi)
{
	if (TYK(t) == K_SINT)
		return TY(K_UINT, TYW(t));
	/* int: the narrowest machine word that holds both */
	return ahi < (1LL << 32) && bhi < (1LL << 32) ? TY(K_UINT, 32) :
	    TY(K_UINT, 64);
}

/* a signed division or remainder of non-negative operands, unsigned */
static void
rw_unsigned(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	ty_t t = N.ty[n], u;
	int64_t alo, ahi, blo, bhi;
	char wa[200], wb[200], cond[300], what[200];
	struct sbuf pf = {0};
	nid a = N.a[n], b = N.b[n], to;
	int div = N.op[n] == O_DIV, w;

	if (TYK(t) != K_SINT && TYK(t) != K_INT)
		return;
	if (!range_of(c, a, 0, &alo, &ahi, wa, sizeof wa) ||
	    !range_of(c, b, 1, &blo, &bhi, wb, sizeof wb))
		return;
	if (alo < 0 || blo < 1)
		return;
	u = unsigned_of(t, ahi, bhi);
	w = TYW(u);
	to = mk_cast(t, mk_bin(N.op[n], mk_cast(u, a), mk_cast(u, b)));
	snprintf(what, sizeof what, "%s %s as %s (its operands are never "
	    "negative)", div ? "the signed division" : "the signed remainder",
	    sstr(n), ty_str(u));
	snprintf(cond, sizeof cond, "%s >= 0 (%s in [%lld, %lld]: %s) and "
	    "%s >= 1 (%s in [%lld, %lld]: %s)", sstr(a), sstr(a),
	    (long long)alo, (long long)ahi, wa, sstr(b), sstr(b),
	    (long long)blo, (long long)bhi, wb);
	if (div)
		sb_printf(&pf, "%s / %s on %s truncates toward zero; with the "
		    "dividend >= 0 and the divisor >= 1 the quotient is >= 0, so "
		    "trunc(a/b) = floor(a/b)", sstr(a), sstr(b), ty_str(t));
	else
		sb_printf(&pf, "%s %% %s on %s is a - b*trunc(a/b), with the "
		    "sign of the dividend; with a >= 0 and b >= 1 it is a - "
		    "b*floor(a/b), in [0, b)", sstr(a), sstr(b), ty_str(t));
	sb_printf(&pf, "\n0 <= a, b < 2^%d: as %s they are the same numbers, "
	    "and the unsigned %s (at most a < 2^%d) converts back to %s "
	    "unchanged", TYK(t) == K_SINT ? TYW(t) - 1 : w, ty_str(u),
	    div ? "quotient floor(a/b)" : "remainder", TYK(t) == K_SINT ?
	    TYW(t) - 1 : w, ty_str(t));
	add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	free(pf.p);
}

static int
cmp_lt(int op)
{
	return op == O_LT || op == O_LE;
}

static int
cmp_gt(int op)
{
	return op == O_GT || op == O_GE;
}

/* x op y with x and y swapped so that y is `want` (0: not there) */
static int
orient(nid c, nid want, nid *x, int *op)
{
	int o = N.op[c];

	if (!cmp_lt(o) && !cmp_gt(o))
		return 0;
	if (N.b[c] == want) {
		*x = N.a[c];
		*op = o;
		return 1;
	}
	if (N.a[c] == want) {
		*x = N.b[c];
		*op = o == O_LT ? O_GT : o == O_LE ? O_GE : o == O_GT ? O_LT :
		    O_LE;
		return 1;
	}
	return 0;
}

/* compare and select: min, max, and clamps made of them */
static void
rw_minmax(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	nid cnd = N.a[n], u = N.b[n], v = N.c[n], x, y, e, to;
	int op, k, below;
	char what[200], cond[300];
	struct sbuf pf = {0};

	if (!TY_ISINT(N.ty[n]) || N.ty[n] == TY_BOOL)
		return;
	/* y is the arm that is a compare operand; e the other arm */
	for (k = 0; k < 2; k++) {
		y = k ? v : u;
		e = k ? u : v;
		if (!orient(cnd, y, &x, &op) || N.ty[x] != N.ty[n])
			continue;
		/* below: y is taken when x is below y */
		below = (k == 0) == (cmp_lt(op) != 0);
		if (e == x) {
			if (k)
				continue;	/* the same select seen from its other arm */
			to = mk_bin(below ? O_MAX : O_MIN, N.a[cnd], N.b[cnd]);
			snprintf(what, sizeof what, "the select %s as %s", sstr(n),
			    sstr(to));
			sb_clear(&pf);
			sb_printf(&pf, "the select takes %s exactly where %s is %s "
			    "it (%s), and %s otherwise; where they are equal both "
			    "arms are the same value; integers are totally ordered, "
			    "so that is %s at every point", sstr(y), sstr(x),
			    below ? "below" : "above", sstr(cnd), sstr(x), sstr(to));
			add(out, cnt, cap, root, n, to, what, "",
			    sb_str(&pf), 1);
			continue;
		}
		/* a clamp's other side: e = min(x, h) under "below" (y <= h
		 * needed), e = max(x, l) under "above" (l <= y) */
		if (N.op[e] == (below ? O_MIN : O_MAX) && (N.a[e] == x ||
		    N.b[e] == x)) {
			nid h = N.a[e] == x ? N.b[e] : N.a[e];
			int64_t ylo, yhi, hlo, hhi;
			char wy[200], wh[200];
			if (!range_of(c, y, INT64_MIN, &ylo, &yhi, wy, sizeof wy) ||
			    !range_of(c, h, INT64_MIN, &hlo, &hhi, wh, sizeof wh))
				continue;
			if (below ? yhi > hlo : hhi > ylo)
				continue;
			to = mk_bin(below ? O_MAX : O_MIN, y, e);
			snprintf(what, sizeof what, "the clamp %s as %s", sstr(n),
			    sstr(to));
			snprintf(cond, sizeof cond, "%s %s %s (%s in [%lld, %lld]: "
			    "%s; %s in [%lld, %lld]: %s)", sstr(y), below ? "<=" :
			    ">=", sstr(h), sstr(y), (long long)ylo, (long long)yhi,
			    wy, sstr(h), (long long)hlo, (long long)hhi, wh);
			sb_clear(&pf);
			sb_printf(&pf, "where %s is %s %s: %s %s %s %s %s, so %s(%s, "
			    "%s) = %s, which the select takes", sstr(x), below ?
			    "below" : "above", sstr(y), sstr(e), below ? "<=" : ">=",
			    sstr(x), below ? "<" : ">", sstr(y), below ? "max" :
			    "min", sstr(y), sstr(e), sstr(y));
			sb_printf(&pf, "\nelsewhere %s is %s %s and %s %s %s, so %s "
			    "%s %s and %s(%s, %s) = %s, which the select takes",
			    sstr(x), below ? "at least" : "at most", sstr(y),
			    sstr(h), below ? ">=" : "<=", sstr(y), sstr(e), below ?
			    ">=" : "<=", sstr(y), below ? "max" : "min", sstr(y),
			    sstr(e), sstr(e));
			add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
		}
	}
	free(pf.p);
}

/*
 * a / b (or a % b) by a variable through f32.  Error of the quotient:
 * f32(a), f32(b) exact below 2^24; p = f32(a) / f32(b) within 2.5 ulp of
 * x = a/b (Vulkan OpFDiv), ulp(x) <= 2^-23 x, so |p - x| <= 5 * 2^-24 x.
 * When that is below 1, q0 = trunc(p) = floor(p) is within 1 of
 * floor(x); r0 = a - q0 b is exact and in [-b, 2b), so one compare each
 * way corrects it.
 */
static void
rw_fdiv(struct ivc *c, nid root, nid n, struct irw **out, int *cnt,
    int *cap)
{
	ty_t t = N.ty[n], S = TY(K_SINT, 32);
	int64_t alo, ahi, blo, bhi;
	char wa[200], wb[200], cond[400], what[200];
	struct sbuf pf = {0};
	nid a = N.a[n], b = N.b[n], a2, b2, q0, r0, q, to;
	double E;
	int div = N.op[n] == O_DIV, ok;

	if (!TY_ISINT(t) || t == TY_BOOL || is_const(b))
		return;
	if (!range_of(c, a, 0, &alo, &ahi, wa, sizeof wa) ||
	    !range_of(c, b, 1, &blo, &bhi, wb, sizeof wb))
		return;
	if (alo < 0 || blo < 1 || ahi >= (1 << 24) || bhi >= (1 << 24))
		return;
	E = 5.0 * ldexp(1.0, -24) * (double)ahi / (double)blo;
	ok = E < 1;
	a2 = mk_cast(S, a);
	b2 = mk_cast(S, b);
	q0 = mk_cast(S, mk_bin(O_DIV, mk_cast(TY_F32, a2), mk_cast(TY_F32,
	    b2)));
	r0 = mk_bin(O_SUB, a2, mk_bin(O_MUL, q0, b2));
	q = mk_bin(O_SUB, mk_bin(O_ADD, q0, mk_cast(S, mk_bin(O_GE, r0, b2))),
	    mk_cast(S, mk_bin(O_LT, r0, mk_const_i(S, 0))));
	to = mk_cast(t, div ? q : mk_bin(O_SUB, a2, mk_bin(O_MUL, q, b2)));
	snprintf(what, sizeof what, "the %s %s through f32 with one "
	    "correction step", div ? "division" : "remainder", sstr(n));
	snprintf(cond, sizeof cond, "%s in [%lld, %lld] (%s), %s in [%lld, "
	    "%lld] (%s): both below 2^24, the divisor >= 1, %s 5*2^-24 * "
	    "%lld/%lld = %.4g %s", sstr(a), (long long)alo, (long long)ahi,
	    wa, sstr(b), (long long)blo, (long long)bhi, wb, ok ? "and" :
	    "but", (long long)ahi, (long long)blo, E, ok ? "< 1" :
	    "is not below 1");
	sb_printf(&pf, "a, b < 2^24: f32(a) and f32(b) are exact");
	sb_printf(&pf, "\np = f32(a) / f32(b) is within 2.5 ulp of x = a/b "
	    "(Vulkan's OpFDiv; correctly rounded is 0.5 ulp), and ulp(x) <= "
	    "2^-23 x, so |p - x| <= 5*2^-24 x <= 5*2^-24 * %lld/%lld = %.4g",
	    (long long)ahi, (long long)blo, E);
	if (!ok) {
		sb_printf(&pf, "\nthat is not below 1: trunc(p) can be off by "
		    "more than one, which one correction step does not fix "
		    "(it would need a/b < %.0f)", ldexp(1.0, 24) / 5);
		add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 0);
		if (*cnt)
			(*out)[*cnt - 1].loose = 1;
		free(pf.p);
		return;
	}
	sb_printf(&pf, "\np >= 0, so q0 = i32(p) = floor(p), and |p - x| < 1 "
	    "gives floor(x) - 1 <= q0 <= floor(x) + 1");
	sb_printf(&pf, "\nr0 = a - q0*b is exact in i32 (q0*b <= a + b < 2^25) "
	    "and is a mod b - b, a mod b or a mod b + b: floor(x) is q0 - 1 "
	    "where r0 < 0, q0 + 1 where r0 >= b, else q0 (the two compares)");
	if (!div)
		sb_printf(&pf, "\nand a %% b = a - floor(x)*b");
	sb_printf(&pf, "\n(with a reciprocal, f32(a) * (1/f32(b)), the bound is "
	    "(6*2^-24 + 5*2^-48) x)");
	add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
	(*out)[*cnt - 1].loose = 1;
	free(pf.p);
}

int
irw_cands(const struct func *f, nid root, struct irw **out)
{
	uint8_t *mark;
	nid *st, n;
	uint32_t sp = 0, lim = N.n;
	struct ivc c;
	int cnt = 0, cap = 0;
	jmp_buf saved;

	*out = NULL;
	if (!TY_ISINT(N.ty[root]))
		return 0;
	mark = xcalloc(lim, 1);
	st = xmalloc(lim * sizeof *st);
	ivc_open(&c, f, root);
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0) {
		st[sp++] = root;
		while (sp) {
			int op;
			n = st[--sp];
			if (n == 0 || n >= lim || mark[n])
				continue;
			mark[n] = 1;
			op = N.op[n];
			if (op == O_CONST || op == O_VAR || op == O_UNDEF ||
			    op == O_TABLE || op == O_LOOP)
				continue;
			if (N.a[n])
				st[sp++] = N.a[n];
			if (N.b[n])
				st[sp++] = N.b[n];
			if (N.c[n])
				st[sp++] = N.c[n];
			if (op == O_DIV || op == O_MOD) {
				if (!TY_ISINT(N.ty[n]))
					continue;
				rw_unsigned(&c, root, n, out, &cnt, &cap);
				rw_fdiv(&c, root, n, out, &cnt, &cap);
			}
			if (op == O_SEL)
				rw_minmax(&c, root, n, out, &cnt, &cap);
		}
	}
	memcpy(fail_jmp, saved, sizeof saved);
	note_mute--;
	notes_drop();
	ivc_close(&c);
	free(mark);
	free(st);
	return cnt;
}

void
irw_free(struct irw *r, int n)
{
	int i;

	for (i = 0; i < n; i++)
		free(r[i].proof);
	free(r);
}

/* the derivation, one line each, indented */
void
irw_print(const struct irw *r, const char *ind)
{
	const char *s = r->proof, *nl;

	if (!r->cond[0])
		printf("%sholds everywhere: integers are totally ordered\n", ind);
	else
		printf("%s%s %s\n", ind, r->nearmiss ? "needs" : "holds where",
		    r->cond);
	while (*s) {
		nl = strchr(s, '\n');
		printf("%s%.*s\n", ind, nl ? (int)(nl - s) : (int)strlen(s), s);
		if (!nl)
			break;
		s = nl + 1;
	}
}
