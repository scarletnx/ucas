/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Proved bounds of a function over its domain, by monotonicity and exact
 * algebra rather than by plain intervals.
 *
 * If f does not decrease in x (on the box the parameters' ranges allow),
 * its maximum over x has x at its largest: the range's upper end, or,
 * when the where clause has x <= e (x < e) with e inside x's range, x = e
 * (e - 1).  Substituting gives a function of fewer variables; repeat.
 * Monotonicity comes from the structure, which is the sign of the
 * derivative by the chain rule: sums, products of terms of known sign,
 * quotients by a positive divisor, increasing functions (sqrt, exp, log,
 * floor, shifts by constants, conversions that keep the value).
 *
 * After a substitution, integer quotients are simplified exactly:
 * floor((k*d + r)/d) = k + floor(r/d) = k when 0 <= r < d (r < d proved
 * from d/c, d >> s, d - c, x % d, or the intervals).  So
 * (t*64 + dd/2)/dd with t <= dd has its maximum at t = dd:
 * (64dd + dd/2)/dd = 64, since 0 <= dd/2 < dd for dd >= 1.
 *
 * Integer types must not wrap anywhere on the box (checked by intervals),
 * so machine and ideal arithmetic agree; otherwise nothing is claimed.
 */
#include "calc.h"
#include "bound.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct biv {
	double lo, hi;
	int ok;
};

struct bctx {
	const struct func *f;
	double lo[MAXP], hi[MAXP];
	struct biv *memo;
	uint8_t *done;
	uint32_t lim;
	int wraps;          /* an integer subterm can leave its type */
};

static void
bctx_reset(struct bctx *c)
{
	free(c->memo);
	free(c->done);
	c->lim = N.n;
	c->memo = xcalloc(c->lim, sizeof *c->memo);
	c->done = xcalloc(c->lim, 1);
	c->wraps = 0;
}

static int
isint(nid n)
{
	return TY_ISINT(N.ty[n]);
}

static double
cval(nid n)
{
	if (TY_FP(N.ty[n]))
		return N.kr[n];
	if (TYK(N.ty[n]) == K_UINT && TYW(N.ty[n]) == 64)
		return (double)(uint64_t)N.ki[n];
	return (double)N.ki[n];
}

static struct biv
mk(double lo, double hi)
{
	struct biv r;

	r.lo = lo;
	r.hi = hi;
	r.ok = !isnan(lo) && !isnan(hi);
	return r;
}

static struct biv bad = {0, 0, 0};

static struct biv iv(struct bctx *c, nid n);

static struct biv
iv1(struct bctx *c, nid n)
{
	struct biv a, b, r = bad;
	double x[4];
	int k, in = isint(n);

	switch (N.op[n]) {
	case O_CONST:
		return mk(cval(n), cval(n));
	case O_VAR:
		if (N.ki[n] >= c->f->np)
			return bad;
		return mk(c->lo[N.ki[n]], c->hi[N.ki[n]]);
	}
	if (opinfo[N.op[n]].arity == 0 || N.op[n] == O_INDEX ||
	    N.op[n] == O_LOOP || N.op[n] == O_TABLE || IS_TEX(N.op[n]))
		return bad;
	a = iv(c, N.a[n]);
	if (!a.ok)
		return bad;
	b = opinfo[N.op[n]].arity >= 2 && N.b[n] ? iv(c, N.b[n]) : bad;
	switch (N.op[n]) {
	case O_ADD:
		if (b.ok)
			r = mk(a.lo + b.lo, a.hi + b.hi);
		break;
	case O_SUB:
		if (b.ok)
			r = mk(a.lo - b.hi, a.hi - b.lo);
		break;
	case O_NEG:
		r = mk(-a.hi, -a.lo);
		break;
	case O_MUL:
		if (!b.ok)
			break;
		x[0] = a.lo * b.lo, x[1] = a.lo * b.hi;
		x[2] = a.hi * b.lo, x[3] = a.hi * b.hi;
		r = mk(x[0], x[0]);
		for (k = 1; k < 4; k++) {
			if (x[k] < r.lo)
				r.lo = x[k];
			if (x[k] > r.hi)
				r.hi = x[k];
		}
		break;
	case O_DIV:
		if (!b.ok || b.lo <= 0)
			break;
		if (in) {
			if (a.lo >= 0)
				r = mk(floor(a.lo / b.hi), floor(a.hi / b.lo));
		} else {
			x[0] = a.lo / b.lo, x[1] = a.lo / b.hi;
			x[2] = a.hi / b.lo, x[3] = a.hi / b.hi;
			r = mk(fmin(fmin(x[0], x[1]), fmin(x[2], x[3])),
			    fmax(fmax(x[0], x[1]), fmax(x[2], x[3])));
		}
		break;
	case O_MOD:
		if (in && b.ok && b.lo > 0 && a.lo >= 0)
			r = mk(0, fmin(a.hi, b.hi - 1));
		break;
	case O_SHL:
		if (b.ok && b.lo >= 0 && b.hi < 63 && a.lo >= 0)
			r = mk(a.lo * ldexp(1, (int)b.lo), a.hi * ldexp(1, (int)b.hi));
		break;
	case O_SHR:
		if (b.ok && b.lo >= 0 && b.hi < 64 && a.lo >= 0)
			r = mk(floor(a.lo / ldexp(1, (int)b.hi)),
			    floor(a.hi / ldexp(1, (int)b.lo)));
		break;
	case O_AND:
		if (b.ok && a.lo >= 0 && b.lo >= 0)
			r = mk(0, fmin(a.hi, b.hi));
		break;
	case O_OR:
	case O_XOR:
		if (b.ok && a.lo >= 0 && b.lo >= 0) {
			double m = fmax(a.hi, b.hi), p = 1;
			while (p <= m)
				p *= 2;
			r = mk(N.op[n] == O_OR ? fmax(a.lo, b.lo) : 0, p - 1);
		}
		break;
	case O_MIN:
		if (b.ok)
			r = mk(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
		break;
	case O_MAX:
		if (b.ok)
			r = mk(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
		break;
	case O_SEL: {
		struct biv e = iv(c, N.c[n]);
		b = iv(c, N.b[n]);
		if (b.ok && e.ok)
			r = mk(fmin(b.lo, e.lo), fmax(b.hi, e.hi));
		break;
	}
	case O_CAST:
		if (TY_FP(N.ty[N.a[n]]) && in)
			r = mk(a.lo < 0 ? ceil(a.lo) : floor(a.lo),
			    a.hi < 0 ? ceil(a.hi) : floor(a.hi));
		else
			r = a;
		break;
	case O_ABS:
		r = mk(a.lo >= 0 ? a.lo : a.hi <= 0 ? -a.hi : 0,
		    fmax(fabs(a.lo), fabs(a.hi)));
		break;
	case O_SQRT:
		if (a.lo >= 0)
			r = mk(sqrt(a.lo), sqrt(a.hi));
		break;
	case O_ISQRT:
		if (a.lo >= 0)
			r = mk(floor(sqrt(a.lo)), floor(sqrt(a.hi)));
		break;
	case O_EXP:
		r = mk(exp(a.lo), exp(a.hi));
		break;
	case O_LOG:
		if (a.lo > 0)
			r = mk(log(a.lo), log(a.hi));
		break;
	case O_ATAN:
		r = mk(atan(a.lo), atan(a.hi));
		break;
	case O_FLOOR:
		r = mk(floor(a.lo), floor(a.hi));
		break;
	case O_CEIL:
		r = mk(ceil(a.lo), ceil(a.hi));
		break;
	case O_POW:
		if (b.ok && b.lo == b.hi && b.lo >= 0 && b.lo == floor(b.lo) &&
		    a.lo >= 0)
			r = mk(pow(a.lo, b.lo), pow(a.hi, b.lo));
		break;
	}
	if (!r.ok)
		return bad;
	/* machine integers: the ideal value must fit, or nothing holds */
	if (TY_SIZED(N.ty[n]) && N.ty[n] != TY_BOOL) {
		struct ity it;
		double tlo, thi;
		ity_of(&it, N.ty[n], 0);
		if (it.k == K_UINT)
			tlo = 0, thi = ldexp(1, it.w) - 1;
		else
			tlo = (double)it.smin, thi = (double)it.smax;
		if (r.lo < tlo || r.hi > thi)
			c->wraps = 1;
	} else if (TYK(N.ty[n]) == K_INT &&
	    (r.lo < -9.2e18 || r.hi > 9.2e18))
		c->wraps = 1;
	return r;
}

static struct biv
iv(struct bctx *c, nid n)
{
	if (n >= c->lim)
		return iv1(c, n);
	if (!c->done[n]) {
		c->memo[n] = iv1(c, n);
		c->done[n] = 1;
	}
	return c->memo[n];
}

/* monotonicity: 1 nondecreasing, -1 nonincreasing, 0 constant, 2 unknown */
static int
comb(int x, int y)
{
	if (x == 0)
		return y;
	if (y == 0 || x == y)
		return x;
	return 2;
}

static int
neg(int x)
{
	return x == 1 ? -1 : x == -1 ? 1 : x;
}

static int
mono(struct bctx *c, nid n, int j)
{
	struct biv a, b;
	int op = N.op[n];

	if (!has_slot(n, j))
		return 0;
	switch (op) {
	case O_VAR:
		return 1;
	case O_ADD:
		return comb(mono(c, N.a[n], j), mono(c, N.b[n], j));
	case O_SUB:
		return comb(mono(c, N.a[n], j), neg(mono(c, N.b[n], j)));
	case O_NEG:
		return neg(mono(c, N.a[n], j));
	case O_MUL:
		a = iv(c, N.a[n]);
		b = iv(c, N.b[n]);
		if (!a.ok || !b.ok)
			return 2;
		{
			int ma = mono(c, N.a[n], j), mb = mono(c, N.b[n], j);
			int sa = a.lo >= 0 ? 1 : a.hi <= 0 ? -1 : 0;
			int sb = b.lo >= 0 ? 1 : b.hi <= 0 ? -1 : 0;
			/* d(ab) = a'b + ab': each part's sign */
			int p = ma == 0 ? 0 : sb == 0 ? 2 : sb > 0 ? ma : neg(ma);
			int q = mb == 0 ? 0 : sa == 0 ? 2 : sa > 0 ? mb : neg(mb);
			return comb(p, q);
		}
	case O_DIV:
		a = iv(c, N.a[n]);
		b = iv(c, N.b[n]);
		if (!a.ok || !b.ok || b.lo <= 0)
			return 2;
		{
			int ma = mono(c, N.a[n], j), mb = mono(c, N.b[n], j);
			if (a.lo >= 0)
				return comb(ma, neg(mb));
			if (a.hi <= 0)
				return comb(ma, mb);
			return mb == 0 ? ma : 2;
		}
	case O_SHL:
	case O_SHR:
		a = iv(c, N.a[n]);
		b = iv(c, N.b[n]);
		if (!a.ok || !b.ok || a.lo < 0 || b.lo < 0)
			return 2;
		return comb(mono(c, N.a[n], j), op == O_SHL ?
		    mono(c, N.b[n], j) : neg(mono(c, N.b[n], j)));
	case O_MIN:
	case O_MAX:
		return comb(mono(c, N.a[n], j), mono(c, N.b[n], j));
	case O_CAST:
	case O_SQRT: case O_ISQRT: case O_EXP: case O_LOG: case O_ATAN:
	case O_FLOOR: case O_CEIL:
		return mono(c, N.a[n], j);
	case O_POW:
		a = iv(c, N.a[n]);
		if (a.ok && a.lo >= 0 && is_const(N.b[n]) && cval(N.b[n]) > 0 &&
		    !has_slot(N.b[n], j))
			return mono(c, N.a[n], j);
		return 2;
	}
	return 2;
}

/* r < d for every point of the box, proved */
static int
lt_proved(struct bctx *c, nid r, nid d, struct sbuf *why)
{
	struct biv ri = iv(c, r), di = iv(c, d);

	if (ri.ok && di.ok && ri.hi < di.lo) {
		sb_printf(why, "%s <= %.17g < %.17g <= %s", node_str(r), ri.hi,
		    di.lo, node_str(d));
		return 1;
	}
	if (!di.ok || di.lo < 1)
		return 0;
	while (N.op[r] == O_CAST && isint(N.a[r]))
		r = N.a[r];
	if (N.op[r] == O_DIV && N.a[r] == d && is_const(N.b[r]) &&
	    cval(N.b[r]) >= 2) {
		sb_printf(why, "%s < %s for %s >= 1", node_str(r), node_str(d),
		    node_str(d));
		return 1;
	}
	if (N.op[r] == O_SHR && N.a[r] == d && is_const(N.b[r]) &&
	    cval(N.b[r]) >= 1) {
		sb_printf(why, "%s <= %s/2 < %s for %s >= 1", node_str(r),
		    node_str(d), node_str(d), node_str(d));
		return 1;
	}
	if (N.op[r] == O_SUB && N.a[r] == d && is_const(N.b[r]) &&
	    cval(N.b[r]) >= 1) {
		sb_printf(why, "%s < %s", node_str(r), node_str(d));
		return 1;
	}
	if (N.op[r] == O_MOD && N.b[r] == d) {
		struct biv x = iv(c, N.a[r]);
		if (x.ok && x.lo >= 0) {
			sb_printf(why, "%s < %s", node_str(r), node_str(d));
			return 1;
		}
	}
	return 0;
}

/* split a sum into multiples of d (k) and the rest */
static int
split_terms(nid n, nid d, int64_t *k, nid *rest, int depth)
{
	if (depth > 32)
		return 0;
	if (N.op[n] == O_ADD)
		return split_terms(N.a[n], d, k, rest, depth + 1) &&
		    split_terms(N.b[n], d, k, rest, depth + 1);
	if (n == d) {
		*k += 1;
		return 1;
	}
	if (N.op[n] == O_MUL && ((N.b[n] == d && is_const(N.a[n])) ||
	    (N.a[n] == d && is_const(N.b[n])))) {
		*k += (int64_t)cval(N.b[n] == d ? N.a[n] : N.b[n]);
		return 1;
	}
	*rest = *rest ? mk_bin(O_ADD, *rest, n) : n;
	return 1;
}

/* floor((k*d + r)/d) = k when 0 <= r < d, bottom-up */
static nid
fdiv_simp(struct bctx *c, nid n, struct sbuf *steps)
{
	nid a, b, r;

	if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_TABLE ||
	    N.op[n] == O_LOOP || N.op[n] == O_INDEX || N.op[n] == O_UNDEF ||
	    IS_TEX(N.op[n]))
		return n;
	a = N.a[n] ? fdiv_simp(c, N.a[n], steps) : 0;
	b = opinfo[N.op[n]].arity >= 2 && N.b[n] ? fdiv_simp(c, N.b[n], steps) :
	    0;
	if (a != N.a[n] || (b && b != N.b[n])) {
		if (N.op[n] == O_SEL)
			n = mk_sel(a, b, fdiv_simp(c, N.c[n], steps));
		else if (N.op[n] == O_CAST)
			n = mk_cast(N.ty[n], a);
		else if (opinfo[N.op[n]].arity == 1)
			n = mk_un(N.op[n], a);
		else if (IS_DOT(N.op[n]) || IS_SG(N.op[n]))
			n = mk_op(N.op[n], a, b, N.c[n]);
		else
			n = mk_bin(N.op[n], a, b);
		bctx_reset(c);
	}
	if (N.op[n] == O_DIV && isint(n)) {
		int64_t k = 0;
		struct biv di;
		nid rest = 0;
		struct sbuf why = {0};
		a = N.a[n];
		b = N.b[n];
		di = iv(c, b);
		if (!di.ok || di.lo < 1 || !split_terms(a, b, &k, &rest, 0) ||
		    k < 0)
			return n;
		if (rest) {
			struct biv ri = iv(c, rest);
			if (!ri.ok || ri.lo < 0 || !lt_proved(c, rest, b, &why)) {
				free(why.p);
				return n;
			}
		}
		r = mk_const_i(N.ty[n], k);
		if (k && rest)
			sb_printf(steps, "\n      %s = %lld + (%s)/%s = %lld, since "
			    "0 <= %s", node_str(n), (long long)k, node_str(rest),
			    node_str(b), (long long)k, sb_str(&why));
		else if (rest)
			sb_printf(steps, "\n      %s = 0, since 0 <= %s",
			    node_str(n), sb_str(&why));
		else
			sb_printf(steps, "\n      %s = %lld", node_str(n),
			    (long long)k);
		free(why.p);
		bctx_reset(c);
		return r;
	}
	return n;
}

static int
var_under_casts(nid e)
{
	while (N.op[e] == O_CAST && isint(N.a[e]) && isint(e))
		e = N.a[e];
	return N.op[e] == O_VAR && N.ki[e] < MAXP ? (int)N.ki[e] : -1;
}

/*
 * A where conjunct x <= e / x < e (want_up) or x >= e / x > e with e
 * inside x's range: e (e - 1, e + 1 for strict integer relations).
 */
static nid
relation(struct bctx *c, nid w, int j, int up, char *txt, size_t ntxt)
{
	nid cj[64], st[64];
	int ns = 0, nc = 0, i, k;

	if (!w)
		return 0;
	st[ns++] = w;
	while (ns > 0 && nc < 64) {
		nid x = st[--ns];
		if (N.op[x] == O_LAND && ns < 62) {
			st[ns++] = N.a[x];
			st[ns++] = N.b[x];
		} else if (opinfo[N.op[x]].fl & OF_CMP)
			cj[nc++] = x;
	}
	for (i = 0; i < nc; i++)
		for (k = 0; k < 2; k++) {
			nid x = k ? N.b[cj[i]] : N.a[cj[i]];
			nid e = k ? N.a[cj[i]] : N.b[cj[i]];
			int op = N.op[cj[i]], strict;
			struct biv ei;
			if (var_under_casts(x) != j || has_slot(e, j))
				continue;
			if (k)
				op = op == O_LT ? O_GT : op == O_GT ? O_LT :
				    op == O_LE ? O_GE : op == O_GE ? O_LE : op;
			if (up ? !(op == O_LT || op == O_LE || op == O_EQ) :
			    !(op == O_GT || op == O_GE || op == O_EQ))
				continue;
			strict = op == O_LT || op == O_GT;
			if (strict && !TY_ISINT(c->f->p[j].ty))
				continue;   /* x < e on reals: a supremum only */
			if (strict)
				e = mk_bin(up ? O_SUB : O_ADD, e,
				    mk_const_i(N.ty[e], 1));
			bctx_reset(c);
			ei = iv(c, e);
			if (!ei.ok || ei.lo < c->lo[j] || ei.hi > c->hi[j])
				continue;
			snprintf(txt, ntxt, "%s", node_str(cj[i]));
			return e;
		}
	return 0;
}

/*
 * The maximum (up) or minimum of f->body, proved; 1 with *val and the
 * steps, 0 when it cannot be shown.
 */
static int
prove_side(const struct func *f, int up, double *val, struct sbuf *steps)
{
	struct bctx c;
	nid e = f->body;
	int j, round, ok = 0;
	uint8_t fixed[MAXP] = {0};
	const char *fn = f->anon ? "f" : sym_str(f->name);

	memset(&c, 0, sizeof c);
	c.f = f;
	{
		int64_t lo[MAXP], hi[MAXP];
		for (j = 0; j < f->np; j++) {
			if (TY_ISINT(f->p[j].ty))
				param_bounds(&f->p[j], &lo[j], &hi[j]);
			else if (f->p[j].ranged) {
				c.lo[j] = f->p[j].rlo;
				c.hi[j] = f->p[j].rhi;
				continue;
			} else
				return 0;
			if (TYK(f->p[j].ty) == K_UINT && TYW(f->p[j].ty) == 64)
				return 0;
		}
		where_bounds(f, lo, hi, NULL);
		for (j = 0; j < f->np; j++)
			if (TY_ISINT(f->p[j].ty)) {
				c.lo[j] = (double)lo[j];
				c.hi[j] = (double)hi[j];
			}
	}
	bctx_reset(&c);
	(void)iv(&c, e);
	if (c.wraps || !iv(&c, e).ok)
		goto out;
	for (round = 0; round < 2 * MAXP; round++) {
		int any = 0;
		for (j = 0; j < f->np; j++) {
			int m;
			nid to;
			char rel[160] = "";
			nid map[MAXP] = {0};
			if (fixed[j] || !has_slot(e, j))
				continue;
			m = mono(&c, e, j);
			if (m != 1 && m != -1)
				continue;
			/* the side of x that makes e largest (smallest) */
			to = relation(&c, f->where, j, (m == 1) == up, rel,
			    sizeof rel);
			bctx_reset(&c);
			if (!to)
				to = mk_const_i(f->p[j].ty, (int64_t)((m == 1) == up ?
				    c.hi[j] : c.lo[j]));
			if (!TY_ISINT(f->p[j].ty) && is_const(to))
				to = mk_const_r((m == 1) == up ? c.hi[j] : c.lo[j]);
			sb_printf(steps, "%s%s is %s in %s%s%s, so take %s = %s:",
			    steps->n ? "\n    then " : "", steps->n ? "it" : fn,
			    m == 1 ? "nondecreasing" : "nonincreasing",
			    sym_str(f->p[j].name), rel[0] ? "; where " : "", rel,
			    sym_str(f->p[j].name), node_str(to));
			map[j] = mk_cast(f->p[j].ty, to);
			e = subst(e, map, f->np);
			bctx_reset(&c);
			{
				size_t before = steps->n;
				nid e0 = e;
				e = fdiv_simp(&c, e, steps);
				if (steps->n == before || !is_const(e))
					sb_printf(steps, "\n      %s%s", e0 != e ?
					    "= " : "", shared_str(e, 300));
			}
			bctx_reset(&c);
			fixed[j] = 1;
			any = 1;
			if (!iv(&c, e).ok || c.wraps)
				goto out;
			break;
		}
		if (!any)
			break;
	}
	{
		struct biv r;
		bctx_reset(&c);
		r = iv(&c, e);
		if (!r.ok || c.wraps)
			goto out;
		if (is_const(e)) {
			*val = cval(e);
			ok = 1;
		} else if (steps->n == 0) {
			goto out;   /* nothing beyond plain intervals */
		} else {
			/* what is left: its interval, still a sound bound */
			*val = up ? r.hi : r.lo;
			sb_printf(steps, "\n      in %.17g..%.17g by intervals",
			    r.lo, r.hi);
			ok = 2;
		}
	}
out:
	free(c.memo);
	free(c.done);
	return ok;
}

int
prove_bounds(const struct func *f, struct pbound *pb)
{
	memset(pb, 0, sizeof *pb);
	if (f->np == 0)
		return 0;
	jmp_buf saved;

	note_mute++;
	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0) {
		pb->have_hi = prove_side(f, 1, &pb->hi, &pb->why_hi);
		pb->have_lo = prove_side(f, 0, &pb->lo, &pb->why_lo);
	}
	memcpy(fail_jmp, saved, sizeof saved);
	note_mute--;
	return pb->have_lo || pb->have_hi;
}

void
pbound_free(struct pbound *pb)
{
	free(pb->why_lo.p);
	free(pb->why_hi.p);
}
