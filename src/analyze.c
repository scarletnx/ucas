/*
 * Analyses over a function's domain: show, table, range, domain, wrap,
 * crit, diff, equiv, error, fit.
 *
 * Everything runs through the same loop: a domain iterator fills a chunk
 * of points, the plan evaluates every node for the chunk, and the analysis
 * reads the columns it needs.  Finite domains up to EXH points are
 * enumerated; larger ones are sampled and the report says so.
 */
#include "ucas.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EXH (1ULL << 22)
#define NSAMPLE (1ULL << 16)

struct run {
	const struct func *f;
	struct plan p;
	struct diter d;
	int64_t *in_i;
	double *in_r;
	uint8_t ok[CH];
	uint32_t wloc;
};

static void
run_open(struct run *r, const struct func *f, const nid *roots, int nroots,
    int ideal, uint64_t maxexh, uint32_t rsteps, uint64_t nsample)
{
	nid rs[16];
	int i;

	memset(r, 0, sizeof *r);
	r->f = f;
	for (i = 0; i < nroots; i++)
		rs[i] = roots[i];
	rs[nroots] = f->where;
	plan_build(&r->p, rs, (uint32_t)nroots + 1, ideal);
	r->wloc = f->where ? plan_loc(&r->p, f->where) : NOTOK;
	diter_init(&r->d, f, maxexh, rsteps, nsample);
	r->in_i = xcalloc((size_t)MAXP * CH, sizeof *r->in_i);
	r->in_r = xcalloc((size_t)MAXP * CH, sizeof *r->in_r);
}

static uint32_t
run_next(struct run *r)
{
	uint32_t np = diter_next(&r->d, r->in_i, r->in_r);

	if (np == 0)
		return 0;
	plan_run(&r->p, np, r->in_i, r->in_r);
	valid_mask(&r->p, r->wloc, np, r->ok);
	return np;
}

static void
run_close(struct run *r)
{
	plan_free(&r->p);
	free(r->in_i);
	free(r->in_r);
}

static const char *
fname(const struct func *f)
{
	return f->anon ? node_str(f->body) : sym_str(f->name);
}

static void
domain_note(const struct run *r, uint64_t seen)
{
	if (r->d.exhaustive)
		printf("  over all %llu points of the domain\n",
		    (unsigned long long)seen);
	else if (r->d.total == UINT64_MAX)
		printf("  sampled %llu points (domain has 2^64 or more points); "
		    "not exhaustive\n", (unsigned long long)seen);
	else
		printf("  sampled %llu points (domain has %llu points); "
		    "not exhaustive\n", (unsigned long long)seen,
		    (unsigned long long)r->d.total);
}

/* integer value in its type's order, as a comparable key */
static int
int_unsigned(ty_t t)
{
	return TYK(t) == K_UINT || TYK(t) == K_BOOL;
}

static int
ilt(ty_t t, int64_t a, int64_t b)
{
	return int_unsigned(t) ? (uint64_t)a < (uint64_t)b : a < b;
}

static void
print_header(const struct func *f)
{
	struct sbuf b = {0};
	int j;

	sb_printf(&b, "%s(", f->anon ? "_" : sym_str(f->name));
	for (j = 0; j < f->np; j++) {
		const struct param *p = &f->p[j];
		sb_printf(&b, "%s%s: %s", j ? ", " : "", sym_str(p->name),
		    ty_str(p->ty));
		if (p->ranged) {
			if (TY_FP(p->ty))
				sb_printf(&b, " in %g..%g", p->rlo, p->rhi);
			else
				sb_printf(&b, " in %lld..%lld", (long long)p->lo,
				    (long long)p->hi);
		}
		if (p->uniform)
			sb_puts(&b, " uniform");
	}
	sb_printf(&b, ") = ");
	print_node(&b, f->body, NULL);
	if (f->where) {
		sb_puts(&b, "  where ");
		print_node(&b, f->where, NULL);
	}
	printf("%s\n  -> %s\n", sb_str(&b), ty_str(N.ty[f->body]));
	free(b.p);
}

void
cmd_show(struct func *f)
{
	print_header(f);
}

void
cmd_table(struct func *f, int max)
{
	struct run r;
	struct sbuf b = {0};
	uint32_t np, i, loc;
	int shown = 0;
	nid root = f->body;

	run_open(&r, f, &root, 1, 0, EXH, 21, 64);
	loc = r.p.root[0];
	while ((np = run_next(&r)) != 0 && shown < max) {
		for (i = 0; i < np && shown < max; i++) {
			if (!r.ok[i])
				continue;
			sb_clear(&b);
			fmt_point(&b, f, r.in_i, r.in_r, i);
			sb_puts(&b, "  ->  ");
			if (COLF(&r.p, loc)[i] & F_UNDEF)
				sb_puts(&b, "undefined");
			else
				fmt_val(&b, N.ty[root], COLI(&r.p, loc)[i],
				    COLR(&r.p, loc)[i]);
			printf("  %s\n", sb_str(&b));
			shown++;
		}
	}
	if (shown == max && (np || r.d.pos < r.d.limit))
		printf("  ... (table max %d)\n", max);
	free(b.p);
	run_close(&r);
}

/* print a sorted set of integers as compact ranges */
static void
print_runs(const char *lead, const int64_t *v, uint64_t n, int maxruns)
{
	uint64_t i = 0;
	int runs = 0;

	printf("%s", lead);
	while (i < n) {
		uint64_t j = i;
		while (j + 1 < n && v[j + 1] == v[j] + 1)
			j++;
		if (runs == maxruns) {
			printf(", ...");
			break;
		}
		printf("%s", runs ? ", " : "");
		if (j == i)
			printf("%lld", (long long)v[i]);
		else
			printf("%lld..%lld", (long long)v[i], (long long)v[j]);
		runs++;
		i = j + 1;
	}
	printf("\n");
}

/* ---- derivative ---- */

static nid *dmemo;
static uint32_t dmemon;

static nid
d1(nid e, uint32_t slot)
{
	nid a, b, da, db, r = 0;
	ty_t t = N.ty[e];
	int op = N.op[e];

	if (e < dmemon && dmemo[e])
		return dmemo[e];
	a = N.a[e];
	b = N.b[e];
#define C(v) mk_const_i(TY_FP(t) ? t : TY_REAL, v)
	switch (op) {
	case O_CONST:
	case O_UNDEF:
		r = C(0);
		break;
	case O_VAR:
		r = C((uint32_t)N.ki[e] == slot ? 1 : 0);
		break;
	case O_ADD:
	case O_SUB:
		r = mk_bin(op, d1(a, slot), d1(b, slot));
		break;
	case O_NEG:
		r = mk_un(O_NEG, d1(a, slot));
		break;
	case O_MUL:
		da = d1(a, slot);
		db = d1(b, slot);
		r = mk_bin(O_ADD, mk_bin(O_MUL, da, b), mk_bin(O_MUL, a, db));
		break;
	case O_DIV:
		da = d1(a, slot);
		db = d1(b, slot);
		if (const_is(db, 0))
			r = mk_bin(O_DIV, da, b);
		else
			r = mk_bin(O_DIV, mk_bin(O_SUB, mk_bin(O_MUL, da, b),
			    mk_bin(O_MUL, a, db)), mk_bin(O_POW, b, C(2)));
		break;
	case O_POW:
		da = d1(a, slot);
		db = d1(b, slot);
		if (const_is(db, 0))
			r = mk_bin(O_MUL, mk_bin(O_MUL, b, mk_bin(O_POW, a,
			    mk_bin(O_SUB, b, C(1)))), da);
		else
			r = mk_bin(O_MUL, e, mk_bin(O_ADD,
			    mk_bin(O_MUL, db, mk_un(O_LOG, a)),
			    mk_bin(O_DIV, mk_bin(O_MUL, b, da), a)));
		break;
	case O_EXP:
		r = mk_bin(O_MUL, e, d1(a, slot));
		break;
	case O_LOG:
		r = mk_bin(O_DIV, d1(a, slot), a);
		break;
	case O_SIN:
		r = mk_bin(O_MUL, mk_un(O_COS, a), d1(a, slot));
		break;
	case O_COS:
		r = mk_un(O_NEG, mk_bin(O_MUL, mk_un(O_SIN, a), d1(a, slot)));
		break;
	case O_TAN:
		r = mk_bin(O_DIV, d1(a, slot), mk_bin(O_POW, mk_un(O_COS, a),
		    C(2)));
		break;
	case O_SQRT:
		r = mk_bin(O_DIV, d1(a, slot), mk_bin(O_MUL, C(2), e));
		break;
	case O_ABS:
		r = mk_bin(O_MUL, mk_un(O_SIGN, a), d1(a, slot));
		break;
	case O_FLOOR:
	case O_CEIL:
	case O_SIGN:
		r = C(0);
		break;
	case O_MIN:
		r = mk_sel(mk_bin(O_LE, a, b), d1(a, slot), d1(b, slot));
		break;
	case O_MAX:
		r = mk_sel(mk_bin(O_GE, a, b), d1(a, slot), d1(b, slot));
		break;
	case O_SEL:
		r = mk_sel(a, d1(b, slot), d1(N.c[e], slot));
		break;
	case O_CAST:
		r = TY_FP(N.ty[a]) ? mk_cast(t, d1(a, slot)) : C(0);
		break;
	case O_BARRIER:
		r = d1(a, slot);
		break;
	case O_LOOP:   /* calc: the loop carries its derivative (loop.c) */
		r = loop_diff(e, slot);
		break;
	default:
		if (TY_ISINT(t) && !(N.fl[e] & NF_VAR))
			r = C(0);
		else
			fail("%s is not differentiable (integer or bit-level "
			    "operation); use 'crit' for the discrete view",
			    opinfo[op].name);
	}
#undef C
	if (e < dmemon)
		dmemo[e] = r;
	return r;
}

nid
diff_node(nid e, uint32_t slot)
{
	nid r, *saved = dmemo;   /* reentrant: loop_diff calls it */
	uint32_t savedn = dmemon;

	dmemon = N.n;
	dmemo = xcalloc(dmemon, sizeof *dmemo);
	r = d1(e, slot);
	free(dmemo);
	dmemo = saved;
	dmemon = savedn;
	return r;
}

void
cmd_diff(struct func *f, int slot)
{
	nid d;

	if (slot >= f->np)
		fail("%s has no parameters", fname(f));
	if (TY_FP(f->p[slot].ty)) {
		d = diff_node(f->body, (uint32_t)slot);
		printf("d/d%s %s = %s\n", sym_str(f->p[slot].name), fname(f),
		    node_str(d));
	} else {
		nid map[MAXP] = {0}, x = mk_var((uint32_t)slot,
		    f->p[slot].name, f->p[slot].ty);
		map[slot] = mk_bin(O_ADD, x, mk_const_i(TY_LIT, 1));
		d = mk_bin(O_SUB, subst(f->body, map, f->np), f->body);
		printf("forward difference over %s: %s(%s+1) - %s(%s) = %s\n",
		    sym_str(f->p[slot].name), fname(f),
		    sym_str(f->p[slot].name), fname(f),
		    sym_str(f->p[slot].name), node_str(d));
	}
}

/* ---- single-point evaluation for refinement ---- */

static int
eval_at(struct plan *p, uint32_t loc, double x, double *out)
{
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];

	memset(in_i, 0, sizeof in_i);
	in_r[0] = x;
	plan_run(p, 1, in_i, in_r);
	*out = COLR(p, loc)[0];
	return !(COLF(p, loc)[0] & F_UNDEF) && !isnan(*out);
}

/* x where the column changes sign / definedness between a and b */
static double
bisect(struct plan *p, uint32_t loc, double a, double b, int by_def)
{
	double fa, fm, m;
	int da, dm, k;

	da = eval_at(p, loc, a, &fa);
	for (k = 0; k < 100; k++) {
		m = 0.5 * (a + b);
		if (m == a || m == b)
			break;
		dm = eval_at(p, loc, m, &fm);
		if (by_def ? (dm == da) : (dm && da && ((fm > 0) == (fa > 0)) &&
		    fm != 0)) {
			a = m;
			fa = fm;
			da = dm;
		} else
			b = m;
		if (!by_def && dm && fm == 0)
			return m;
	}
	return 0.5 * (a + b);
}

/* ---- range ---- */

#define GRID 20001

static void
range_real1(struct func *f)
{
	struct run r;
	nid roots[2];
	double lo = INFINITY, hi = -INFINITY, xlo = 0, xhi = 0, prevd = 0,
	    prevx = 0, v, x;
	uint32_t np, i, loc, dl = NOTOK;
	uint64_t nund = 0, nnan = 0, ninf = 0, seen = 0;
	int prevok = 0, have_d = 0;
	struct plan dp;
	nid d = 0;

	roots[0] = f->body;
	{
		/* not every body is differentiable; then only the grid is used */
		jmp_buf saved;
		memcpy(saved, fail_jmp, sizeof saved);
		if (setjmp(fail_jmp) == 0) {
			d = diff_node(f->body, 0);
			have_d = 1;
		}
		memcpy(fail_jmp, saved, sizeof saved);
		notes_drop();
	}
	roots[1] = have_d ? d : f->body;
	run_open(&r, f, roots, 2, 0, EXH, GRID, NSAMPLE);
	loc = r.p.root[0];
	if (have_d)
		dl = r.p.root[1];
	plan_build(&dp, roots, 2, 0);
	while ((np = run_next(&r)) != 0) {
		for (i = 0; i < np; i++) {
			if (!r.ok[i])
				continue;
			seen++;
			x = r.in_r[i];
			if (COLF(&r.p, loc)[i] & F_UNDEF) {
				nund++;
				prevok = 0;
				continue;
			}
			v = COLR(&r.p, loc)[i];
			if (isnan(v)) {
				nnan++;
				continue;
			}
			if (isinf(v))
				ninf++;
			if (v < lo)
				lo = v, xlo = x;
			if (v > hi)
				hi = v, xhi = x;
			if (have_d && !(COLF(&r.p, dl)[i] & F_UNDEF)) {
				double dv = COLR(&r.p, dl)[i];
				if (prevok && ((dv > 0) != (prevd > 0))) {
					double c = bisect(&dp, dp.root[1], prevx, x, 0), fc;
					if (eval_at(&dp, dp.root[0], c, &fc)) {
						if (fc < lo)
							lo = fc, xlo = c;
						if (fc > hi)
							hi = fc, xhi = c;
					}
				}
				prevd = dv;
				prevx = x;
				prevok = 1;
			} else
				prevok = 0;
		}
	}
	printf("range of %s:\n", fname(f));
	if (lo > hi)
		printf("  empty (undefined everywhere)\n");
	else
		printf("  [%.10g, %.10g]   min at %s ≈ %.10g, max at %s ≈ %.10g\n",
		    lo, hi, sym_str(f->p[0].name), xlo, sym_str(f->p[0].name),
		    xhi);
	if (nund)
		printf("  undefined at %llu of %llu grid points\n",
		    (unsigned long long)nund, (unsigned long long)seen);
	if (nnan || ninf)
		printf("  IEEE: NaN at %llu points, ±inf at %llu points\n",
		    (unsigned long long)nnan, (unsigned long long)ninf);
	if (TYK(f->p[0].ty) == K_REAL)
		printf("  grid of %d points over [%g, %g]%s, extrema refined "
		    "at critical points\n", GRID, r.d.rlo[0], r.d.rhi[0],
		    f->p[0].ranged ? "" : " (default; give x a range)");
	else
		domain_note(&r, seen);
	plan_free(&dp);
	run_close(&r);
}

void
cmd_range(struct func *f)
{
	struct run r;
	nid root = f->body;
	ty_t t = N.ty[root];
	uint32_t np, i, loc;
	uint64_t seen = 0, nund = 0, nnan = 0, ninf = 0, ndist = 0;
	int64_t mn = 0, mx = 0;
	double rmn = INFINITY, rmx = -INFINITY;
	int any = 0;
	uint8_t *bm = NULL;
	int64_t base = 0;
	uint64_t span = 0;

	if (f->np == 1 && TY_FP(f->p[0].ty) && TY_FP(t) &&
	    TYK(f->p[0].ty) == K_REAL) {
		range_real1(f);
		return;
	}
	run_open(&r, f, &root, 1, 0, EXH, 101, NSAMPLE);
	loc = r.p.root[0];
	if (TY_ISINT(t)) {
		/* value bitmap for small result types */
		struct ity it;
		ity_of(&it, t, 0);
		if (TY_SIZED(t) && it.w <= 24) {
			span = it.mask + 1;
			base = TYK(t) == K_SINT ? it.smin : 0;
			bm = xcalloc((size_t)(span + 7) / 8, 1);
		}
	}
	while ((np = run_next(&r)) != 0) {
		const int64_t *vi = COLI(&r.p, loc);
		const double *vr = COLR(&r.p, loc);
		const uint8_t *vf = COLF(&r.p, loc);
		for (i = 0; i < np; i++) {
			if (!r.ok[i])
				continue;
			seen++;
			if (vf[i] & F_UNDEF) {
				nund++;
				continue;
			}
			if (TY_FP(t)) {
				if (isnan(vr[i])) {
					nnan++;
					continue;
				}
				if (isinf(vr[i]))
					ninf++;
				if (vr[i] < rmn)
					rmn = vr[i];
				if (vr[i] > rmx)
					rmx = vr[i];
				any = 1;
				continue;
			}
			if (!any || ilt(t, vi[i], mn))
				mn = vi[i];
			if (!any || ilt(t, mx, vi[i]))
				mx = vi[i];
			any = 1;
			if (bm) {
				uint64_t k = (uint64_t)(vi[i] - base);
				if (!(bm[k >> 3] & (1 << (k & 7)))) {
					bm[k >> 3] |= (uint8_t)(1 << (k & 7));
					ndist++;
				}
			}
		}
	}
	printf("range of %s -> %s:\n", fname(f), ty_str(t));
	if (!any)
		printf("  empty (undefined everywhere)\n");
	else if (TY_FP(t))
		printf("  [%.9g, %.9g]\n", rmn, rmx);
	else {
		struct sbuf b = {0};
		fmt_val(&b, t, mn, 0);
		sb_puts(&b, "..");
		fmt_val(&b, t, mx, 0);
		printf("  %s", sb_str(&b));
		if (bm)
			printf("   (%llu distinct values)",
			    (unsigned long long)ndist);
		printf("\n");
		if (bm && ndist <= 64) {
			int64_t *vals = xmalloc(ndist * sizeof *vals);
			uint64_t k, n = 0;
			for (k = 0; k < span; k++)
				if (bm[k >> 3] & (1 << (k & 7)))
					vals[n++] = (int64_t)k + base;
			print_runs("  values: ", vals, n, 24);
			free(vals);
		}
		if (int_unsigned(t) || mn >= 0)
			printf("  fits in u%d\n", bits_for((uint64_t)mx));
		else {
			int bb = bits_for((uint64_t)(mn < 0 ? ~mn : mn));
			int bb2 = bits_for((uint64_t)(mx < 0 ? ~mx : mx));
			printf("  fits in i%d\n", (bb > bb2 ? bb : bb2) + 1);
		}
		free(b.p);
	}
	if (nund)
		printf("  undefined at %llu points\n", (unsigned long long)nund);
	if (nnan || ninf)
		printf("  IEEE: NaN at %llu points, ±inf at %llu points\n",
		    (unsigned long long)nnan, (unsigned long long)ninf);
	domain_note(&r, seen);
	if (!r.d.exhaustive && TY_ISINT(t)) {
		struct ival *iv = xcalloc(r.p.n, sizeof *iv);
		plan_ival(&r.p, f, iv);
		if (!iv[loc].top)
			printf("  sound bound (intervals): %lld..%lld\n",
			    (long long)iv[loc].lo, (long long)iv[loc].hi);
		free(iv);
	}
	free(bm);
	run_close(&r);
}

/* ---- domain ---- */

/* round a bisected boundary to the shortest nearby decimal */
static double
snap(double x, double span)
{
	char buf[64];
	int d;
	double y;

	if (fabs(x) < span * 1e-12)
		return 0;
	for (d = 1; d <= 15; d++) {
		snprintf(buf, sizeof buf, "%.*g", d, x);
		y = strtod(buf, NULL);
		if (fabs(y - x) <= span * 1e-12)
			return y;
	}
	return x;
}

static void
list_conditions(const struct func *f, int *nc)
{
	uint8_t *mark = xcalloc(N.n, 1);
	nid *st = xmalloc(N.n * sizeof *st), n;
	uint32_t sp = 0, j;
	char w[16];

	*nc = 0;
	st[sp++] = f->body;
	mark[f->body] = 1;
	while (sp) {
		n = st[--sp];
		switch (N.op[n]) {
		case O_CONST:
		case O_VAR:
			continue;
		case O_TABLE:
			for (j = 0; j < N.b[n]; j++)
				if (!mark[N.lst[N.a[n] + j]]) {
					mark[N.lst[N.a[n] + j]] = 1;
					st[sp++] = N.lst[N.a[n] + j];
				}
			continue;
		}
		if (N.a[n] && !mark[N.a[n]])
			mark[N.a[n]] = 1, st[sp++] = N.a[n];
		if (N.b[n] && !mark[N.b[n]])
			mark[N.b[n]] = 1, st[sp++] = N.b[n];
		if (N.c[n] && !mark[N.c[n]])
			mark[N.c[n]] = 1, st[sp++] = N.c[n];
		if (TYK(N.ty[n]) == K_FLOAT)
			continue;
#define COND(...) do { printf("  requires "); printf(__VA_ARGS__); \
		printf("\n"); (*nc)++; } while (0)
#define UB(...) do { printf("  C assumes "); printf(__VA_ARGS__); \
		printf("\n"); (*nc)++; } while (0)
		switch (N.op[n]) {
		case O_DIV:
		case O_MOD:
			if (!(is_const(N.b[n]) && !const_is(N.b[n], 0)))
				COND("%s != 0  (in %s)", node_str(N.b[n]),
				    node_str(n));
			if (TYK(N.ty[n]) == K_SINT && !is_const(N.b[n]))
				UB("not (%s == min and %s == -1)  (in %s)",
				    node_str(N.a[n]), node_str(N.b[n]),
				    node_str(n));
			break;
		case O_SQRT:
			COND("%s >= 0", node_str(N.a[n]));
			break;
		case O_LOG:
			COND("%s > 0", node_str(N.a[n]));
			break;
		case O_TAN:
			COND("cos(%s) != 0", node_str(N.a[n]));
			break;
		case O_POW:
			if (TYK(N.ty[n]) == K_REAL && !(is_const(N.b[n]) &&
			    N.kr[N.b[n]] == floor(N.kr[N.b[n]])))
				COND("%s >= 0  (real power)", node_str(N.a[n]));
			else if (TY_ISINT(N.ty[n]) && !is_const(N.b[n]))
				COND("%s >= 0  (integer power)", node_str(N.b[n]));
			break;
		case O_ISQRT:
			if (!int_unsigned(N.ty[N.a[n]]))
				COND("%s >= 0", node_str(N.a[n]));
			break;
		case O_SHL:
		case O_SHR:
			if (!is_const(N.b[n]))
				COND("0 <= %s < %d  (shift amount; C UB "
				    "otherwise)", node_str(N.b[n]),
				    TYW(N.ty[n]) == 64 && !TY_SIZED(N.ty[n]) ?
				    64 : TYW(N.ty[n]));
			if (N.op[n] == O_SHL && TYK(N.ty[n]) == K_SINT)
				UB("%s does not overflow  (signed shift)",
				    node_str(n));
			break;
		case O_INDEX:
			COND("0 <= %s < %u  (table index)", node_str(N.b[n]),
			    N.b[N.a[n]]);
			break;
		case O_UNDEF:
			COND("an arm of match/piecewise covers the input");
			break;
		case O_CAST:
			if (TY_FP(N.ty[N.a[n]]) && TY_ISINT(N.ty[n]))
				COND("%s fits %s  (float to integer)",
				    node_str(N.a[n]), ty_str(N.ty[n]));
			break;
		case O_ADD:
		case O_SUB:
		case O_MUL:
		case O_NEG:
		case O_ABS:
			if (TYK(N.ty[n]) == K_SINT)
				UB("%s does not overflow %s  (signed overflow)",
				    node_str(n), ty_str(N.ty[n]));
			break;
		case O_CLZ:
		case O_CTZ:
			UB("%s != 0  (__builtin_%s)", node_str(N.a[n]),
			    opinfo[N.op[n]].name);
			break;
		}
#undef COND
#undef UB
	}
	(void)w;
	free(mark);
	free(st);
}

static void
domain_real1(struct func *f)
{
	struct run r;
	nid root = f->body;
	uint32_t np, i, loc;
	double prevx = 0, x, starts[64], ends[64];
	int prevok = -1, nint = 0, ok;
	struct plan dp;

	run_open(&r, f, &root, 1, 0, EXH, GRID, NSAMPLE);
	plan_build(&dp, &root, 1, 0);
	loc = r.p.root[0];
	while ((np = run_next(&r)) != 0) {
		for (i = 0; i < np; i++) {
			if (!r.ok[i])
				continue;
			x = r.in_r[i];
			ok = !(COLF(&r.p, loc)[i] & F_UNDEF) &&
			    !isnan(COLR(&r.p, loc)[i]);
			if (prevok < 0) {
				if (ok && nint < 64)
					starts[nint] = x;
			} else if (ok != prevok) {
				double b = bisect(&dp, dp.root[0], prevx, x, 1);
				if (ok && nint < 64)
					starts[nint] = b;
				else if (!ok && nint < 64)
					ends[nint++] = b;
			}
			prevok = ok;
			prevx = x;
		}
	}
	if (prevok == 1 && nint < 64)
		ends[nint++] = prevx;
	printf("  defined on (numerically, grid of %d over [%g, %g]):\n", GRID,
	    r.d.rlo[0], r.d.rhi[0]);
	if (nint == 0)
		printf("    nowhere\n");
	for (i = 0; i < (uint32_t)nint; i++) {
		/* snap to a short decimal, then test whether it is included */
		double a = snap(starts[i], r.d.rhi[0] - r.d.rlo[0]);
		double b = snap(ends[i], r.d.rhi[0] - r.d.rlo[0]), v;
		int ina = eval_at(&dp, dp.root[0], a, &v);
		int inb = eval_at(&dp, dp.root[0], b, &v);
		printf("    %s in %c%.10g, %.10g%c\n", sym_str(f->p[0].name),
		    ina ? '[' : '(', a, b, inb ? ']' : ')');
	}
	/* isolated singularities: a denominator crossing zero */
	{
		uint8_t *mark = xcalloc(N.n, 1);
		uint32_t g;
		for (g = 1; g < N.n; g++) {
			nid den;
			struct plan pp;
			double a, fa, b, fb, step;
			int k;
			if (N.op[g] != O_DIV || TYK(N.ty[g]) != K_REAL)
				continue;
			if (plan_loc(&dp, g) == NOTOK)
				continue;
			den = N.b[g];
			if (is_const(den) || mark[den])
				continue;
			mark[den] = 1;
			plan_build(&pp, &den, 1, 0);
			step = (r.d.rhi[0] - r.d.rlo[0]) / (GRID - 1);
			a = r.d.rlo[0];
			for (k = 0; k + 1 < GRID; k++, a = b) {
				int da2, db2;
				b = r.d.rlo[0] + step * (k + 1);
				da2 = eval_at(&pp, pp.root[0], a, &fa);
				db2 = eval_at(&pp, pp.root[0], b, &fb);
				if (da2 && db2 && ((fa < 0 && fb > 0) ||
				    (fa > 0 && fb < 0) || fa == 0)) {
					double z = fa == 0 ? a :
					    bisect(&pp, pp.root[0], a, b, 0);
					printf("    except %s ≈ %.10g  (%s = 0)\n",
					    sym_str(f->p[0].name), z,
					    node_str(den));
				}
			}
			plan_free(&pp);
		}
		free(mark);
	}
	plan_free(&dp);
	run_close(&r);
}

void
cmd_domain(struct func *f)
{
	struct run r;
	nid root = f->body;
	uint32_t np, i, loc, k;
	uint64_t seen = 0, ndef = 0, nub = 0, nalloc = 0, nv = 0;
	int64_t *defv = NULL;
	struct sbuf w = {0}, wub = {0};
	int nc, shown = 0;
	uint8_t *lv;

	printf("domain of %s:\n", fname(f));
	for (k = 0; k < (uint32_t)f->np; k++) {
		const struct param *p = &f->p[k];
		int64_t lo, hi;
		if (TY_FP(p->ty))
			printf("  %s: %s%s\n", sym_str(p->name), ty_str(p->ty),
			    p->ranged ? "" : TYK(p->ty) == K_FLOAT ?
			    " (every finite value)" : " (default [-10, 10])");
		else {
			param_bounds(p, &lo, &hi);
			printf("  %s: %s, %lld..%lld\n", sym_str(p->name),
			    ty_str(p->ty), (long long)lo, (long long)hi);
		}
	}
	if (f->where)
		printf("  where %s\n", node_str(f->where));
	list_conditions(f, &nc);
	if (nc == 0)
		printf("  no conditions: total on its parameter types\n");
	if (f->np == 1 && TYK(f->p[0].ty) == K_REAL) {
		domain_real1(f);
		return;
	}
	run_open(&r, f, &root, 1, 0, EXH, 101, NSAMPLE);
	loc = r.p.root[0];
	lv = xmalloc((size_t)r.p.n * CH);
	while ((np = run_next(&r)) != 0) {
		plan_live(&r.p, np, lv);
		for (i = 0; i < np; i++) {
			int ub = 0;
			if (!r.ok[i])
				continue;
			seen++;
			for (k = 0; k < r.p.n; k++)
				if (lv[(size_t)k * CH + i] &&
				    (COLF(&r.p, k)[i] & F_UB))
					ub = 1;
			if (ub) {
				nub++;
				if (!wub.n)
					fmt_point(&wub, f, r.in_i, r.in_r, i);
			}
			if (COLF(&r.p, loc)[i] & F_UNDEF) {
				if (shown < 4) {
					if (w.n)
						sb_puts(&w, "; ");
					fmt_point(&w, f, r.in_i, r.in_r, i);
					shown++;
				}
				continue;
			}
			ndef++;
			if (f->np == 1 && r.d.exhaustive && !TY_FP(f->p[0].ty)) {
				if (nv == nalloc) {
					nalloc = nalloc ? nalloc * 2 : 1024;
					defv = xrealloc(defv, nalloc * sizeof *defv);
				}
				defv[nv++] = r.in_i[i];
			}
		}
	}
	printf("  defined at %llu of %llu points\n", (unsigned long long)ndef,
	    (unsigned long long)seen);
	if (defv && ndef < seen)
		print_runs("  defined for: ", defv, nv, 16);
	if (w.n)
		printf("  undefined e.g. at %s\n", sb_str(&w));
	if (nub)
		printf("  C undefined behaviour at %llu points, e.g. %s\n",
		    (unsigned long long)nub, sb_str(&wub));
	domain_note(&r, seen);
	free(defv);
	free(lv);
	free(w.p);
	free(wub.p);
	run_close(&r);
}

/* ---- wrap ---- */

struct wstat {
	uint64_t uw, sw, tr, ub, nan, inf;
	struct sbuf wu, ws, wt, wn, wi;
};

void
cmd_wrap(struct func *f)
{
	struct run r, ir;
	nid root = f->body;
	uint32_t np, np2, i, k, loc, iloc;
	struct wstat *ws;
	uint8_t *lv;
	uint64_t seen = 0, mism = 0, nmis_def = 0, novf = 0;
	int64_t *exact = NULL, *inex = NULL;
	uint64_t nex = 0, nin = 0, cap1 = 0, cap2 = 0;
	struct sbuf wm = {0};
	ty_t t = N.ty[root];
	double maxabs = 0, maxrel = 0, maxulp = 0;
	struct sbuf wr = {0};
	int printed = 0;
	struct ival *iv;

	run_open(&r, f, &root, 1, 0, EXH, 101, NSAMPLE);
	run_open(&ir, f, &root, 1, 1, EXH, 101, NSAMPLE);
	loc = r.p.root[0];
	iloc = ir.p.root[0];
	ws = xcalloc(r.p.n, sizeof *ws);
	lv = xmalloc((size_t)r.p.n * CH);
	while ((np = run_next(&r)) != 0) {
		np2 = run_next(&ir);
		(void)np2;
		plan_live(&r.p, np, lv);
		for (i = 0; i < np; i++) {
			if (!r.ok[i])
				continue;
			seen++;
			for (k = 0; k < r.p.n; k++) {
				uint8_t fl = COLF(&r.p, k)[i];
				struct wstat *s = &ws[k];
				if (!lv[(size_t)k * CH + i] || !fl)
					continue;
#define HIT(cnt, buf, bit) if (fl & (bit)) { if (!s->cnt++) \
		fmt_point(&s->buf, f, r.in_i, r.in_r, i); }
				HIT(uw, wu, F_UW);
				HIT(sw, ws, F_SW);
				HIT(tr, wt, F_TRUNC);
				HIT(nan, wn, F_NAN);
				HIT(inf, wi, F_INF);
#undef HIT
				if (fl & F_UB)
					s->ub++;
			}
			/* end to end: machine result vs ideal result */
			if ((COLF(&r.p, loc)[i] | COLF(&ir.p, iloc)[i]) & F_UNDEF) {
				if ((COLF(&r.p, loc)[i] ^ COLF(&ir.p, iloc)[i]) & F_UNDEF)
					nmis_def++;
				continue;
			}
			if (TY_FP(t)) {
				double m = COLR(&r.p, loc)[i], id = COLR(&ir.p, iloc)[i];
				double ae = fabs(m - id), re;
				if (isnan(m) || isnan(id))
					continue;
				if (isinf(m) != isinf(id)) {
					novf++;
					continue;
				}
				if (isinf(m))
					continue;
				re = id != 0 ? ae / fabs(id) : ae;
				if (ae > maxabs)
					maxabs = ae;
				if (re > maxrel) {
					maxrel = re;
					sb_clear(&wr);
					fmt_point(&wr, f, r.in_i, r.in_r, i);
				}
				if (TYK(t) == K_FLOAT) {
					double u = fabs((double)(fkey(TYW(t), m) -
					    fkey(TYW(t), id)));
					if (u > maxulp)
						maxulp = u;
				}
			} else {
				struct ity it;
				int64_t m = COLI(&r.p, loc)[i], id = COLI(&ir.p, iloc)[i];
				int same;
				ity_of(&it, t, 0);
				same = int_unsigned(t) ? (id >= 0 &&
				    (uint64_t)id == (uint64_t)m) : id == m;
				if (!same) {
					if (!mism++)
						fmt_point(&wm, f, r.in_i, r.in_r, i);
				}
				if (f->np == 1 && r.d.exhaustive) {
					int64_t **arr = same ? &exact : &inex;
					uint64_t *cnt = same ? &nex : &nin;
					uint64_t *cap = same ? &cap1 : &cap2;
					if (*cnt == *cap) {
						*cap = *cap ? *cap * 2 : 1024;
						*arr = xrealloc(*arr, *cap * sizeof **arr);
					}
					(*arr)[(*cnt)++] = r.in_i[i];
				}
			}
		}
	}
	printf("wrap analysis of %s:\n", fname(f));
	iv = xcalloc(r.p.n, sizeof *iv);
	plan_ival(&r.p, f, iv);
	for (k = 0; k < r.p.n; k++) {
		nid g = r.p.gid[k];
		int op = r.p.op[k];
		ty_t tk = r.p.ty[k];
		struct wstat *s = &ws[k];
		if (TYK(tk) == K_FLOAT && (op == O_ADD || op == O_SUB ||
		    op == O_MUL || op == O_DIV || op == O_SQRT || op == O_LOG ||
		    op == O_POW || op == O_EXP || op == O_ASFLOAT ||
		    op == O_CAST || op == O_TAN)) {
			printf("  %s   (%s)\n", node_str(g), ty_str(tk));
			if (s->nan)
				printf("      nnan fails: NaN at %llu points, e.g. %s\n",
				    (unsigned long long)s->nan, sb_str(&s->wn));
			else
				printf("      nnan holds (no NaN from non-NaN inputs)\n");
			if (s->inf)
				printf("      ninf fails: inf at %llu points, e.g. %s\n",
				    (unsigned long long)s->inf, sb_str(&s->wi));
			else
				printf("      ninf holds\n");
			printed++;
			continue;
		}
		if (!TY_SIZED(tk) || TYK(tk) == K_BOOL)
			continue;
		if (op == O_ADD || op == O_SUB || op == O_MUL || op == O_NEG ||
		    op == O_SHL || op == O_ABS) {
			printf("  %s   (%s)\n", node_str(g), ty_str(tk));
			if (s->uw)
				printf("      nuw fails: unsigned wrap at %llu "
				    "points, e.g. %s\n", (unsigned long long)s->uw,
				    sb_str(&s->wu));
			else
				printf("      nuw holds%s\n", r.d.exhaustive ?
				    "" : (iv[k].wrap ? " on the sample (intervals "
				    "cannot prove it)" : " (proved by intervals)"));
			if (s->sw)
				printf("      nsw fails: signed wrap at %llu points, "
				    "e.g. %s%s\n", (unsigned long long)s->sw,
				    sb_str(&s->ws), TYK(tk) == K_SINT ?
				    "  <- UB in C; a compiler assumes it cannot "
				    "happen" : "");
			else
				printf("      nsw holds%s\n", TYK(tk) == K_SINT ?
				    " (C requires it: signed overflow is UB)" : "");
			printed++;
		} else if (op == O_CAST && s->tr) {
			printf("  %s   (%s)\n", node_str(g), ty_str(tk));
			printf("      truncates at %llu points, e.g. %s\n",
			    (unsigned long long)s->tr, sb_str(&s->wt));
			printed++;
		}
	}
	if (!printed)
		printf("  no wrapping or rounding operations\n");
	if (TY_FP(t)) {
		printf("  end to end vs the real-number result: max abs error "
		    "%.3g, max rel error %.3g", maxabs, maxrel);
		if (TYK(t) == K_FLOAT)
			printf(", max %g ulp%s", maxulp, maxulp == 0 ?
			    " (correctly rounded)" : "");
		printf("\n");
		if (wr.n)
			printf("      worst relative error at %s\n", sb_str(&wr));
		if (novf)
			printf("      the float result is infinite where the real one "
			    "is finite at %llu points (overflow)\n",
			    (unsigned long long)novf);
	} else if (mism == 0 && !nmis_def) {
		printf("  end to end: equals the mathematical (unbounded) result "
		    "everywhere\n");
	} else {
		printf("  end to end: differs from the mathematical result at "
		    "%llu points, e.g. %s\n", (unsigned long long)mism,
		    sb_str(&wm));
		if (exact && nex)
			print_runs("      exact for: ", exact, nex, 12);
	}
	domain_note(&r, seen);
	for (k = 0; k < r.p.n; k++) {
		free(ws[k].wu.p);
		free(ws[k].ws.p);
		free(ws[k].wt.p);
		free(ws[k].wn.p);
		free(ws[k].wi.p);
	}
	free(ws);
	free(iv);
	free(lv);
	free(exact);
	free(inex);
	free(wm.p);
	free(wr.p);
	run_close(&r);
	run_close(&ir);
}

/* ---- critical points ---- */

static void
crit_int1(struct func *f)
{
	struct run r;
	nid root = f->body;
	uint32_t np, i, loc;
	int64_t px = 0, py = 0, segx0 = 0, segy0 = 0, segd = 0, prevd = 0;
	int have = 0, inseg = 0, nseg = 0, nlist = 0;
	ty_t t = N.ty[root];
	struct sbuf ext = {0};
	int nexts = 0;

	run_open(&r, f, &root, 1, 0, 1u << 20, 2, 0);
	if (!r.d.exhaustive) {
		printf("  domain too large for the discrete view (%llu points); "
		    "give %s a range\n", (unsigned long long)r.d.total,
		    sym_str(f->p[0].name));
		run_close(&r);
		return;
	}
	loc = r.p.root[0];
	printf("  piecewise-affine segments (consecutive x with a constant "
	    "step):\n");
#define FLUSH() do { if (inseg) { nseg++; if (nlist < 16) { nlist++; \
	if (segd == 0) printf("    %s in %lld..%lld: %s = %lld\n", \
	    sym_str(f->p[0].name), (long long)segx0, (long long)px, \
	    fname(f), (long long)segy0); \
	else { long long c0_ = (long long)(segy0 - segd * segx0); \
	printf("    %s in %lld..%lld: %s = %lld%s %c %lld\n", \
	    sym_str(f->p[0].name), (long long)segx0, (long long)px, \
	    fname(f), (long long)segd, sym_str(f->p[0].name), \
	    c0_ < 0 ? '-' : '+', c0_ < 0 ? -c0_ : c0_); } } } inseg = 0; \
	} while (0)
	while ((np = run_next(&r)) != 0) {
		for (i = 0; i < np; i++) {
			int64_t x = r.in_i[i], y;
			if (!r.ok[i] || (COLF(&r.p, loc)[i] & F_UNDEF)) {
				FLUSH();
				have = 0;
				continue;
			}
			y = COLI(&r.p, loc)[i];
			if (int_unsigned(t) && TYW(t) == 64 && y < 0)
				y = INT64_MAX;
			if (!have) {
				segx0 = x;
				segy0 = y;
				inseg = 1;
				segd = 0;
				have = 1;
				px = x;
				py = y;
				prevd = 0;
				continue;
			}
			if (x != px + 1) {
				FLUSH();
				segx0 = x;
				segy0 = y;
				inseg = 1;
				segd = 0;
				px = x;
				py = y;
				prevd = 0;
				continue;
			}
			{
				int64_t d = y - py;
				if (px == segx0) {
					segd = d;
				} else if (d != segd) {
					/* the segment ends at px; the new one starts there */
					int64_t sx = px, sy = py;
					FLUSH();
					segx0 = sx;
					segy0 = sy;
					segd = d;
					inseg = 1;
				}
				if (prevd && d && ((prevd > 0) != (d > 0))) {
					if (nexts < 12)
						sb_printf(&ext, "    %s = %lld: local %s, "
						    "%s = %lld\n", sym_str(f->p[0].name),
						    (long long)px, prevd > 0 ? "maximum" :
						    "minimum", fname(f), (long long)py);
					nexts++;
				}
				if (d)
					prevd = d;
				px = x;
				py = y;
			}
		}
	}
	FLUSH();
#undef FLUSH
	if (nseg > nlist)
		printf("    ... %d segments in all\n", nseg);
	if (nseg == 1)
		printf("  -> affine on its whole domain\n");
	if (nexts) {
		printf("  local extrema (where the step changes sign):\n%s",
		    sb_str(&ext));
		if (nexts > 12)
			printf("    ... %d in all\n", nexts);
	}
	free(ext.p);
	run_close(&r);
}

void
cmd_crit(struct func *f)
{
	nid d, roots[2];
	struct run r;
	struct plan dp;
	uint32_t np, i, fl, dl;
	double prevx = 0, prevd = 0, x;
	int prevok = 0, prevfdef = -1, nfound = 0, j, waszero = 0;

	printf("critical points of %s:\n", fname(f));
	if (f->np == 0) {
		printf("  constant\n");
		return;
	}
	if (f->np > 1) {
		for (j = 0; j < f->np; j++)
			if (!TY_FP(f->p[j].ty))
				break;
		if (j < f->np) {
			printf("  only one integer parameter is supported for the "
			    "discrete view\n");
			return;
		}
		printf("  gradient (solve all = 0):\n");
		for (j = 0; j < f->np; j++)
			printf("    d/d%s = %s\n", sym_str(f->p[j].name),
			    node_str(diff_node(f->body, (uint32_t)j)));
		return;
	}
	if (!TY_FP(f->p[0].ty)) {
		crit_int1(f);
		return;
	}
	d = diff_node(f->body, 0);
	printf("  %s' = %s\n", fname(f), node_str(d));
	roots[0] = f->body;
	roots[1] = d;
	run_open(&r, f, roots, 2, 0, EXH, 4001, 4001);
	plan_build(&dp, roots, 2, 0);
	fl = r.p.root[0];
	dl = r.p.root[1];
	if (TYK(f->p[0].ty) == K_FLOAT && !f->p[0].ranged) {
		printf("  give %s a range (x: f32 in a..b)\n",
		    sym_str(f->p[0].name));
		plan_free(&dp);
		run_close(&r);
		return;
	}
	while ((np = run_next(&r)) != 0) {
		for (i = 0; i < np; i++) {
			int fdef, ddef;
			double dv;
			if (!r.ok[i])
				continue;
			x = r.in_r[i];
			fdef = !(COLF(&r.p, fl)[i] & F_UNDEF) &&
			    !isnan(COLR(&r.p, fl)[i]);
			ddef = !(COLF(&r.p, dl)[i] & F_UNDEF) &&
			    !isnan(COLR(&r.p, dl)[i]) && !isinf(COLR(&r.p, dl)[i]);
			dv = COLR(&r.p, dl)[i];
			if (fdef && prevfdef == 1 && ddef && prevok && !waszero &&
			    (dv == 0 || (dv > 0) != (prevd > 0))) {
				double c = dv == 0 ? x : bisect(&dp, dp.root[1],
				    prevx, x, 0), fc = 0, dl2, dr2;
				double h = (r.d.rhi[0] - r.d.rlo[0]) * 1e-7;
				int okl, okr;
				const char *kind;
				eval_at(&dp, dp.root[0], c, &fc);
				okl = eval_at(&dp, dp.root[1], c - h, &dl2);
				okr = eval_at(&dp, dp.root[1], c + h, &dr2);
				if (okl && okr && dl2 < 0 && dr2 > 0)
					kind = "local minimum";
				else if (okl && okr && dl2 > 0 && dr2 < 0)
					kind = "local maximum";
				else if (okl && okr && (dl2 > 0) == (dr2 > 0))
					kind = "stationary inflection";
				else
					kind = "kink (derivative jumps)";
				if (nfound < 32)
					printf("  %s ≈ %.10g: %s, %s = %.10g\n",
					    sym_str(f->p[0].name), c, kind, fname(f),
					    fc);
				nfound++;
			} else if (fdef && prevfdef == 1 && ddef != prevok) {
				double c = bisect(&dp, dp.root[1], prevx, x, 1), fc = 0;
				eval_at(&dp, dp.root[0], c, &fc);
				if (nfound < 32)
					printf("  %s ≈ %.10g: not differentiable, "
					    "%s = %.10g\n", sym_str(f->p[0].name), c,
					    fname(f), fc);
				nfound++;
			}
			/* an exact zero on the grid was reported; skip its far side */
			waszero = ddef && dv == 0;
			prevok = ddef;
			if (dv != 0 || !ddef)
				prevd = dv;
			prevx = x;
			prevfdef = fdef;
		}
	}
	if (nfound == 0)
		printf("  none in [%g, %g]\n", r.d.rlo[0], r.d.rhi[0]);
	else if (nfound > 32)
		printf("  ... %d in all\n", nfound);
	if (!f->p[0].ranged)
		printf("  searched [%g, %g] (default; give %s a range)\n",
		    r.d.rlo[0], r.d.rhi[0], sym_str(f->p[0].name));
	plan_free(&dp);
	run_close(&r);
}

/* ---- equivalence, error, fit ---- */

static nid
rebind(const struct func *f, const struct func *g)
{
	nid map[MAXP];
	int j;

	if (g->np > f->np)
		fail("%s has %d parameters, %s has %d", fname(g), g->np,
		    fname(f), f->np);
	for (j = 0; j < g->np; j++)
		map[j] = mk_cast(g->p[j].ty, mk_var((uint32_t)j, f->p[j].name,
		    f->p[j].ty));
	return subst(g->body, map, g->np);
}

static int
same_val(ty_t t, int64_t a, double ar, int64_t b, double br)
{
	if (TY_FP(t))
		return (isnan(ar) && isnan(br)) || ar == br;
	return a == b;
}

int
equiv_check(const struct func *f, nid a, nid b, struct sbuf *wit, int *sampled)
{
	struct run r;
	nid roots[2];
	uint32_t np, i, la, lb;
	int64_t mism = 0;
	ty_t t;

	roots[0] = a;
	roots[1] = mk_cast(N.ty[a], b);
	t = N.ty[a];
	run_open(&r, f, roots, 2, 0, EXH, 1001, NSAMPLE);
	la = r.p.root[0];
	lb = r.p.root[1];
	while ((np = run_next(&r)) != 0) {
		for (i = 0; i < np; i++) {
			int ua, ub;
			if (!r.ok[i])
				continue;
			ua = COLF(&r.p, la)[i] & F_UNDEF;
			ub = COLF(&r.p, lb)[i] & F_UNDEF;
			if (ua && ub)
				continue;
			if (ua != ub || !same_val(t, COLI(&r.p, la)[i],
			    COLR(&r.p, la)[i], COLI(&r.p, lb)[i],
			    COLR(&r.p, lb)[i])) {
				if (!mism++ && wit) {
					fmt_point(wit, f, r.in_i, r.in_r, i);
					sb_puts(wit, ": ");
					if (ua)
						sb_puts(wit, "undefined");
					else
						fmt_val(wit, t, COLI(&r.p, la)[i],
						    COLR(&r.p, la)[i]);
					sb_puts(wit, " vs ");
					if (ub)
						sb_puts(wit, "undefined");
					else
						fmt_val(wit, t, COLI(&r.p, lb)[i],
						    COLR(&r.p, lb)[i]);
				}
			}
		}
	}
	if (sampled)
		*sampled = !r.d.exhaustive;
	run_close(&r);
	return (int)(mism > 0x7fffffff ? 0x7fffffff : mism);
}

void
cmd_equiv(struct func *f, struct func *g)
{
	struct sbuf w = {0};
	int sampled, m;
	nid gb = rebind(f, g);

	if (g->np != f->np)
		fail("%s and %s take different numbers of parameters", fname(f),
		    fname(g));
	m = equiv_check(f, f->body, gb, &w, &sampled);
	if (m == 0)
		printf("%s and %s are equal on the domain of %s%s\n", fname(f),
		    fname(g), fname(f), sampled ? " (sampled, not proved)" :
		    " (exhaustive)");
	else
		printf("%s and %s differ at %d points, e.g. %s\n", fname(f),
		    fname(g), m, sb_str(&w));
	free(w.p);
}

struct errst {
	double maxabs, maxrel, sumrel, maxulp;
	uint64_t n, bad;
	struct sbuf wabs, wrel, wbad;
};

static void
measure(const struct func *f, nid approx, struct errst *e, int ideal_target)
{
	struct run r;
	nid roots[2];
	uint32_t np, i, lt, la;
	ty_t ta = N.ty[approx];

	memset(e, 0, sizeof *e);
	roots[0] = f->body;
	roots[1] = approx;
	run_open(&r, f, roots, 2, 0, EXH, 1001, NSAMPLE);
	lt = r.p.root[0];
	la = r.p.root[1];
	(void)ideal_target;
	while ((np = run_next(&r)) != 0) {
		for (i = 0; i < np; i++) {
			double t, a, ae, re;
			if (!r.ok[i] || (COLF(&r.p, lt)[i] & F_UNDEF))
				continue;
			t = TY_FP(N.ty[f->body]) ? COLR(&r.p, lt)[i] :
			    (double)COLI(&r.p, lt)[i];
			if (isnan(t) || isinf(t))
				continue;
			a = TY_FP(ta) ? COLR(&r.p, la)[i] : (double)COLI(&r.p, la)[i];
			if ((COLF(&r.p, la)[i] & F_UNDEF) || isnan(a) || isinf(a)) {
				if (!e->bad++)
					fmt_point(&e->wbad, f, r.in_i, r.in_r, i);
				continue;
			}
			e->n++;
			ae = fabs(a - t);
			re = t != 0 ? ae / fabs(t) : ae;
			if (ae > e->maxabs) {
				e->maxabs = ae;
				sb_clear(&e->wabs);
				fmt_point(&e->wabs, f, r.in_i, r.in_r, i);
			}
			if (re > e->maxrel) {
				e->maxrel = re;
				sb_clear(&e->wrel);
				fmt_point(&e->wrel, f, r.in_i, r.in_r, i);
			}
			e->sumrel += re;
			if (TYK(ta) == K_FLOAT) {
				double u = fabs((double)(fkey(TYW(ta), a) -
				    fkey(TYW(ta), t)));
				if (u > e->maxulp)
					e->maxulp = u;
			}
		}
	}
	run_close(&r);
}

static void
print_err(const struct errst *e, ty_t ta)
{
	printf("  max abs error %.6g at %s\n", e->maxabs,
	    e->wabs.n ? e->wabs.p : "-");
	printf("  max rel error %.6g (%.3f%%) at %s\n", e->maxrel,
	    e->maxrel * 100, e->wrel.n ? e->wrel.p : "-");
	printf("  mean rel error %.6g over %llu points\n",
	    e->n ? e->sumrel / (double)e->n : 0, (unsigned long long)e->n);
	if (TYK(ta) == K_FLOAT)
		printf("  max %.0f ulp\n", e->maxulp);
	if (e->bad)
		printf("  approximation is NaN/inf/undefined at %llu points "
		    "where the target is finite, e.g. %s\n",
		    (unsigned long long)e->bad, e->wbad.p);
}

void
cmd_error(struct func *g, struct func *f)
{
	struct errst e;
	nid a;

	if (g->np != f->np)
		fail("%s has %d parameters, %s has %d: 'fit' the extra ones "
		    "first", fname(g), g->np, fname(f), f->np);
	a = rebind(f, g);
	measure(f, a, &e, 0);
	printf("error of %s against %s:\n", fname(g), fname(f));
	print_err(&e, N.ty[a]);
	free(e.wabs.p);
	free(e.wrel.p);
	free(e.wbad.p);
}

/*
 * fit: g's first parameters are f's inputs; the rest are constants chosen
 * to minimise the error metric over f's domain (sampled).  Each constant
 * is searched by a coarse scan over its range followed by a pattern search
 * with halving steps; several constants are fitted by coordinate descent.
 */
#define FITPTS 4096

int fit_quiet;
double fit_last_k[MAXP], fit_last_obj;

struct fitctx {
	struct plan p;
	uint32_t la;
	uint32_t npts;
	double *tgt;
	int64_t *pin_i;     /* f's inputs per point */
	double *pin_r;
	int nin, nk;
	ty_t kty[MAXP];
	int metric;
};

static double
fit_obj(struct fitctx *c, const double *k, uint32_t stride)
{
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH], worst = 0, sum = 0;
	uint32_t base, i, np, cnt = 0;
	int j;
	ty_t ta = c->p.ty[c->la];

	for (base = 0; base < c->npts; base += CH * stride) {
		np = 0;
		for (i = base; i < c->npts && np < CH; i += stride, np++) {
			for (j = 0; j < c->nin; j++) {
				in_i[(size_t)j * CH + np] = c->pin_i[(size_t)i * MAXP + j];
				in_r[(size_t)j * CH + np] = c->pin_r[(size_t)i * MAXP + j];
			}
			for (j = 0; j < c->nk; j++) {
				in_i[(size_t)(c->nin + j) * CH + np] = (int64_t)k[j];
				in_r[(size_t)(c->nin + j) * CH + np] = k[j];
			}
		}
		plan_run(&c->p, np, in_i, in_r);
		for (i = 0; i < np; i++) {
			double t = c->tgt[base + i * stride], a, e;
			if (isnan(t))
				continue;
			if (COLF(&c->p, c->la)[i] & F_UNDEF)
				return INFINITY;
			a = TY_FP(ta) ? COLR(&c->p, c->la)[i] :
			    (double)COLI(&c->p, c->la)[i];
			if (isnan(a) || isinf(a))
				return INFINITY;
			e = fabs(a - t);
			if (c->metric == 0 && t != 0)
				e /= fabs(t);
			if (c->metric == 2)
				sum += e * e;
			else if (e > worst)
				worst = e;
			cnt++;
		}
	}
	if (c->metric == 2)
		return cnt ? sqrt(sum / cnt) : 0;
	return worst;
}

static void
krange(ty_t t, const struct param *p, double *lo, double *hi, int *isint)
{
	int64_t a, b;

	*isint = TY_ISINT(t);
	if (*isint) {
		param_bounds(p, &a, &b);
		*lo = (double)a;
		*hi = (double)b;
	} else if (p->ranged) {
		*lo = p->rlo;
		*hi = p->rhi;
	} else {
		*lo = -10;
		*hi = 10;
	}
}

void
cmd_fit(struct func *g, struct func *f, int metric)
{
	struct fitctx c;
	struct diter d;
	nid map[MAXP], tmpl, consts[MAXP], fitted;
	double k[MAXP], best, lo[MAXP], hi[MAXP];
	int isint[MAXP], j, round;
	uint32_t i, np, tl;
	struct plan tp;
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];
	struct errst e;
	static const char *const mname[] = {"max relative", "max absolute",
	    "rms"};

	memset(&c, 0, sizeof c);
	c.nin = f->np;
	c.nk = g->np - f->np;
	c.metric = metric;
	if (c.nk <= 0)
		fail("%s has no parameters left to fit (it takes %d, %s takes "
		    "%d)", fname(g), g->np, fname(f), f->np);
	for (j = 0; j < g->np; j++) {
		if (j < f->np)
			map[j] = mk_cast(g->p[j].ty, mk_var((uint32_t)j,
			    f->p[j].name, f->p[j].ty));
		else {
			map[j] = mk_var((uint32_t)j, g->p[j].name, g->p[j].ty);
			c.kty[j - f->np] = g->p[j].ty;
			krange(g->p[j].ty, &g->p[j], &lo[j - f->np],
			    &hi[j - f->np], &isint[j - f->np]);
		}
	}
	tmpl = subst(g->body, map, g->np);
	plan_build(&c.p, &tmpl, 1, 0);
	c.la = c.p.root[0];

	/* sample f's domain and its target values */
	plan_build(&tp, &f->body, 1, 0);
	tl = tp.root[0];
	diter_init(&d, f, FITPTS, 257, FITPTS);
	c.tgt = xmalloc(FITPTS * sizeof *c.tgt);
	c.pin_i = xcalloc((size_t)FITPTS * MAXP, sizeof *c.pin_i);
	c.pin_r = xcalloc((size_t)FITPTS * MAXP, sizeof *c.pin_r);
	while ((np = diter_next(&d, in_i, in_r)) != 0) {
		plan_run(&tp, np, in_i, in_r);
		for (i = 0; i < np && c.npts < FITPTS; i++) {
			double t = TY_FP(N.ty[f->body]) ? COLR(&tp, tl)[i] :
			    (double)COLI(&tp, tl)[i];
			if (COLF(&tp, tl)[i] & F_UNDEF || isinf(t))
				t = NAN;
			c.tgt[c.npts] = t;
			for (j = 0; j < f->np; j++) {
				c.pin_i[(size_t)c.npts * MAXP + j] = in_i[(size_t)j * CH + i];
				c.pin_r[(size_t)c.npts * MAXP + j] = in_r[(size_t)j * CH + i];
			}
			c.npts++;
		}
	}
	plan_free(&tp);

	for (j = 0; j < c.nk; j++)
		k[j] = isint[j] ? floor((lo[j] + hi[j]) / 2) : (lo[j] + hi[j]) / 2;
	best = fit_obj(&c, k, 1);
	for (round = 0; round < (c.nk > 1 ? 6 : 1); round++) {
		for (j = 0; j < c.nk; j++) {
			double span = hi[j] - lo[j], step, cand[8], cv[8], save;
			int s, nb = 0, q;
			save = k[j];
			/* coarse scan on a subsample: keep the best 8 */
			if (round == 0) {
				for (q = 0; q < 8; q++)
					cv[q] = INFINITY, cand[q] = save;
				for (s = 0; s <= 4096; s++) {
					double v, o;
					int w;
					v = lo[j] + span * s / 4096.0;
					if (isint[j])
						v = floor(v);
					k[j] = v;
					o = fit_obj(&c, k, 16);
					for (w = 0; w < 8 && cv[w] <= o; w++)
						;
					if (w < 8) {
						memmove(&cv[w + 1], &cv[w],
						    (size_t)(7 - w) * sizeof *cv);
						memmove(&cand[w + 1], &cand[w],
						    (size_t)(7 - w) * sizeof *cand);
						cv[w] = o;
						cand[w] = v;
					}
				}
				nb = 8;
			} else {
				cand[0] = save;
				nb = 1;
			}
			k[j] = save;
			best = fit_obj(&c, k, 1);
			for (q = 0; q < nb; q++) {
				double cur, co;
				if (isinf(cv[q]) && round == 0)
					continue;
				cur = cand[q];
				k[j] = cur;
				co = fit_obj(&c, k, 1);
				step = round == 0 ? span / 4096.0 :
				    fabs(cur) * 1e-3 + (isint[j] ? 1 : 1e-6);
				if (isint[j] && step < 1)
					step = 1;
				while (isint[j] ? step >= 1 : step > fabs(cur) * 1e-15 + 1e-300) {
					int moved = 0, dir;
					for (dir = -1; dir <= 1; dir += 2) {
						double v = cur + dir * step, o;
						if (isint[j])
							v = floor(v);
						if (v < lo[j] || v > hi[j])
							continue;
						k[j] = v;
						o = fit_obj(&c, k, 1);
						if (o < co) {
							co = o;
							cur = v;
							moved = 1;
							break;
						}
					}
					if (!moved)
						step = isint[j] ? floor(step / 2) : step / 2;
				}
				if (co < best) {
					best = co;
					save = cur;
				}
			}
			k[j] = save;
		}
	}
	for (j = 0; j < c.nk; j++)
		fit_last_k[j] = k[j];
	fit_last_obj = best;
	if (fit_quiet) {
		plan_free(&c.p);
		free(c.tgt);
		free(c.pin_i);
		free(c.pin_r);
		return;
	}
	printf("fit %s to %s (%s error, %u sample points):\n", fname(g),
	    fname(f), mname[metric], c.npts);
	for (j = 0; j < c.nk; j++) {
		const struct param *p = &g->p[f->np + j];
		if (isint[j]) {
			uint64_t u = (uint64_t)(int64_t)k[j];
			printf("  %s = %lld  (0x%llx)\n", sym_str(p->name),
			    (long long)(int64_t)k[j], (unsigned long long)u);
			consts[j] = mk_const_i(p->ty, (int64_t)k[j]);
		} else {
			printf("  %s = %.10g\n", sym_str(p->name), k[j]);
			consts[j] = mk_cast(p->ty, mk_const_r(k[j]));
		}
	}
	printf("  objective %.6g\n", best);
	for (j = 0; j < c.nk; j++)
		map[f->np + j] = consts[j];
	fitted = subst(g->body, map, g->np);
	{
		struct sbuf b = {0};
		struct pctx pc = {PM_UCAS, NULL};
		print_node(&b, fitted, &pc);
		printf("  %s\n", sb_str(&b));
		free(b.p);
	}
	measure(f, fitted, &e, 0);
	print_err(&e, N.ty[fitted]);
	free(e.wabs.p);
	free(e.wrel.p);
	free(e.wbad.p);
	plan_free(&c.p);
	free(c.tgt);
	free(c.pin_i);
	free(c.pin_r);
}
