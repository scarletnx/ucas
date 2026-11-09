/*
 * alg: the algebra commands (syntax in alg_syntax.inc) and the
 * simplifier.  Every result is compared with its input at sample
 * points, and the comparison is printed.
 *
 *   expand F            products and integer powers multiplied out
 *   factor F            over Q: content, square-free, rational roots,
 *                       Kronecker; homogeneous in two variables
 *   collect F [x]       a polynomial in x with coefficients in the rest
 *   simplify F          the shortest of the normal forms, with the
 *                       conditions where a factor was cancelled
 *   together F          one fraction (numerator expanded, denominator
 *                       factored); cancel F: both expanded
 *   apart F [x]         partial fractions over Q
 *   gcd P, Q            lcm P, Q; resultant P, Q [x]; discriminant P [x]
 *   quo P, Q [x]        rem P, Q [x]; divide P by Q (both)
 *   substitute x = a, ... in F
 */
#include "alg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- the canonical variables of a command ---- */

static uint32_t vname[MAXP];
static int nvname;

void alg_vars_reset(void);
nid alg_canon(struct func *f);
nid alg_var_named(uint32_t name);

void
alg_vars_reset(void)
{
	nvname = 0;
}

static int
vslot(uint32_t name)
{
	int k;

	for (k = 0; k < nvname; k++)
		if (vname[k] == name)
			return k;
	if (nvname == MAXP)
		fail("too many variables (at most %d)", MAXP);
	vname[nvname] = name;
	return nvname++;
}

/* f's body in the command's variables: the same name is the same variable */
nid
alg_canon(struct func *f)
{
	nid map[NSLOT];
	int k;

	memset(map, 0, sizeof map);
	for (k = 0; k < f->np; k++) {
		ty_t t = f->p[k].ty;
		if (!TY_FP(t))
			t = TY_REAL;
		map[k] = mk_var((uint32_t)vslot(f->p[k].name), f->p[k].name, t);
	}
	for (k = 0; k < f->np; k++)
		if (!TY_FP(f->p[k].ty))
			map[k] = map[k];
	return mk_cast(TY_REAL, subst(f->body, map, NSLOT));
}

nid
alg_var_named(uint32_t name)
{
	return mk_var((uint32_t)vslot(name), name, TY_REAL);
}

/* ---- numeric checks ---- */

static void
slots_of(nid e, uint64_t *m, int depth)
{
	uint32_t j;

	if (e == 0 || depth > 400)
		return;
	if (N.op[e] == O_VAR) {
		if (N.ki[e] < 64)
			*m |= (uint64_t)1 << N.ki[e];
		return;
	}
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			slots_of(N.lst[N.a[e] + j], m, depth + 1);
		return;
	}
	if (N.op[e] == O_CONST || N.op[e] == O_UNDEF)
		return;
	slots_of(N.a[e], m, depth + 1);
	slots_of(N.b[e], m, depth + 1);
	slots_of(N.c[e], m, depth + 1);
}

/*
 * a and b at 64 sample points (seeded) where both are defined; returns
 * the number compared and the worst relative difference.  Points where
 * only one is defined are counted in *onlyone.
 */
int alg_check(nid a, nid b, double *worst, int *onlyone);

int
alg_check(nid a, nid b, double *worst, int *onlyone)
{
	struct plan p;
	nid roots[2];
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	uint64_t rng = 0x9e3779b97f4a7c15ULL;
	uint32_t la, lb;
	int i, k, n = 0, one = 0;

	*worst = 0;
	roots[0] = mk_cast(TY_REAL, a);
	roots[1] = mk_cast(TY_REAL, b);
	plan_build(&p, roots, 2, 1);
	for (i = 0; i < 64; i++)
		for (k = 0; k < MAXP; k++) {
			double u;
			rng ^= rng << 13;
			rng ^= rng >> 7;
			rng ^= rng << 17;
			u = (double)(rng >> 11) / 9007199254740992.0;
			/* mostly in [-3, 3], some larger */
			in_r[(size_t)k * CH + i] = (i < 48 ? 6 * u - 3 : 20 * u - 10) +
			    0.0123 * k;
			in_i[(size_t)k * CH + i] = (int64_t)floor(in_r[(size_t)k * CH +
			    i]);
		}
	plan_run(&p, 64, in_i, in_r);
	la = p.root[0];
	lb = p.root[1];
	for (i = 0; i < 64; i++) {
		int da = !(COLF(&p, la)[i] & F_UNDEF) && isfinite(COLR(&p, la)[i]);
		int db = !(COLF(&p, lb)[i] & F_UNDEF) && isfinite(COLR(&p, lb)[i]);
		double x, y, d;
		if (da != db)
			one++;
		if (!da || !db)
			continue;
		x = COLR(&p, la)[i];
		y = COLR(&p, lb)[i];
		d = fabs(x - y) / fmax(1, fmax(fabs(x), fabs(y)));
		if (d > *worst)
			*worst = d;
		n++;
	}
	plan_free(&p);
	free(in_i);
	free(in_r);
	if (onlyone)
		*onlyone = one;
	return n;
}

static void
print_check(nid a, nid b, const char *cond)
{
	double w;
	int one, n = alg_check(a, b, &w, &one);
	uint64_t m = 0;

	slots_of(a, &m, 0);
	slots_of(b, &m, 0);
	if (m == 0) {
		printf("  check: %s numerically (difference %.2g)\n", w < 1e-9 ?
		    "equal" : "DIFFERENT", w);
		return;
	}
	if (n == 0)
		printf("  check: no sample point where both are defined\n");
	else if (w <= 1e-9)
		printf("  check: equal at %d sample points (max relative difference "
		    "%.2g)%s%s\n", n, w, one && cond ? "; they differ in where "
		    "they are defined: " : "", one && cond ? cond : "");
	else
		printf("  check FAILED: they differ by %.3g (relative) at %d "
		    "sample points\n", w, n);
}

/* ---- the simplifier ---- */

static nid simp_rec(nid e, struct sbuf *conds, int depth);

static void
add_cond(struct sbuf *conds, const char *fmt, nid u)
{
	char buf[512];

	if (!conds)
		return;
	snprintf(buf, sizeof buf, fmt, node_str(u));
	if (conds->p && strstr(conds->p, buf))
		return;
	if (conds->n)
		sb_puts(conds, ", ");
	sb_puts(conds, buf);
}

/* sqrt(u) with u a square: |w| sqrt(c) */
static nid
sqrt_rule(nid u)
{
	struct mp a, f[32];
	int m[32], n, i, partial;
	aq c, k;
	int64_t sm;
	nid w = 0;
	size_t mk = ar_mark();

	atoms_reset();
	if (!mp_of(u, &a) || mp_isconst(a, NULL)) {
		ar_release(mk);
		return 0;
	}
	n = mp_factor(a, &c, f, m, 32, &partial);
	if (alg_ovf || qsgn(c) <= 0) {
		ar_release(mk);
		return 0;
	}
	for (i = 0; i < n; i++)
		if (m[i] & 1) {
			ar_release(mk);
			return 0;
		}
	sqrt_split(c, &k, &sm);
	for (i = 0; i < n; i++) {
		nid t = mp_node(f[i]);
		if (m[i] > 2)
			t = POW(t, KI(m[i] / 2));
		w = w ? MUL(w, t) : t;
	}
	w = mk_un(O_ABS, w);
	if (sm != 1)
		w = MUL(sqrt_cnode(sm), w);
	w = qmul_node(k, w);
	ar_release(mk);
	return w;
}

/* the operands simplified, then the rules for this function */
static nid
simp_atom(nid e, struct sbuf *conds, int depth)
{
	int op = N.op[e];
	nid a, r;

	if (opinfo[op].arity != 1 || op == O_CAST || N.a[e] == 0)
		return e;
	a = simp_rec(N.a[e], conds, depth + 1);
	switch (op) {
	case O_SQRT:
		if (N.op[a] == O_POW && const_is(N.b[a], 2))
			return mk_un(O_ABS, N.a[a]);
		if ((r = sqrt_rule(a)) != 0)
			return r;
		break;
	case O_EXP:
		if (N.op[a] == O_LOG) {
			add_cond(conds, "%s > 0", N.a[a]);
			return N.a[a];
		}
		break;
	case O_LOG:
		if (N.op[a] == O_EXP)
			return N.a[a];
		break;
	case O_ABS:
		if (N.op[a] == O_ABS || N.op[a] == O_EXP)
			return a;
		if (N.op[a] == O_NEG)
			a = N.a[a];
		break;
	}
	return mk_un(op, a);
}

/* every atom of the polynomials replaced by its simplified form */
static nid
simp_rec(nid e, struct sbuf *conds, int depth)
{
	int op = N.op[e];

	if (depth > 100 || !(N.fl[e] & NF_VAR))
		return e;
	switch (op) {
	case O_CONST:
	case O_VAR:
	case O_UNDEF:
		return e;
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_DIV:
	case O_POW:
		if (!TY_FP(N.ty[e]))
			return e;
		{
			nid a = simp_rec(N.a[e], conds, depth + 1), b = simp_rec(N.b[e],
			    conds, depth + 1);
			double k;
			/* exp(u)^k = exp(k u) */
			if (op == O_POW && N.op[a] == O_EXP && num_of(b, &k) &&
			    k == floor(k))
				return mk_un(O_EXP, alg_simplify(MUL(b, N.a[a]), NULL));
			return mk_bin(op, a, b);
		}
	case O_NEG:
		return mk_un(O_NEG, simp_rec(N.a[e], conds, depth + 1));
	case O_CAST:
		if (TY_FP(N.ty[e]))
			return mk_cast(N.ty[e], simp_rec(N.a[e], conds, depth + 1));
		return e;
	}
	if (opinfo[op].arity == 1)
		return simp_atom(e, conds, depth);
	return e;
}

/* sin^2 -> 1 - cos^2 (or cosh^2 -> 1 + sinh^2) in a */
struct mp
mp_pyth(struct mp a, int s, int c, int hyper)
{
	struct mp r = mp_zero(), one = mp_const(qi(1)), q = mp_sub(one,
	    mp_mul(mp_atom(c), mp_atom(c)));
	int i;

	if (hyper)
		q = mp_add(one, mp_mul(mp_atom(c), mp_atom(c)));
	for (i = 0; i < a.n; i++) {
		struct mp t;
		int e = a.t[i].e[s];
		t.n = 1;
		t.t = &a.t[i];
		t = mp_copy(t);
		t.t[0].e[s] = (uint8_t)(e & 1);
		r = mp_add(r, mp_mul(t, mp_pow(q, e / 2)));
	}
	return r;
}

/* exp(a)^i exp(b)^j ... in each monomial of the numerator -> one exp */
static struct mp
mp_expjoin(struct mp a, int *changed)
{
	struct mp r = mp_zero();
	int i, k;

	for (i = 0; i < a.n; i++) {
		int nexp = 0;
		nid arg = 0;
		struct mp t;
		t.n = 1;
		t.t = &a.t[i];
		t = mp_copy(t);
		for (k = 0; k < nat; k++)
			if (t.t[0].e[k] && N.op[at_node[k]] == O_EXP)
				nexp += t.t[0].e[k] > 1 ? 2 : 1;
		if (nexp >= 2) {
			for (k = 0; k < nat; k++)
				if (t.t[0].e[k] && N.op[at_node[k]] == O_EXP) {
					nid u = N.a[at_node[k]];
					if (t.t[0].e[k] > 1)
						u = MUL(KR(t.t[0].e[k]), u);
					arg = arg ? ADD(arg, u) : u;
					t.t[0].e[k] = 0;
				}
			if (arg) {
				nid ex = mk_un(O_EXP, alg_simplify(arg, NULL));
				int v = atom_of(ex);
				if (v >= 0) {
					t.t[0].e[v] = 1;
					*changed = 1;
				} else
					t = mp_zero();
			}
		}
		r = mp_add(r, t);
	}
	return r;
}

static size_t
nlen(nid e)
{
	const char *s = node_str(e);
	size_t n = strlen(s);
	const char *p;

	/* a function call counts as 6 characters, like trig's simplest form */
	for (p = s; (p = strchr(p, '(')) != NULL; p++)
		if (p > s && p[-1] >= 'a' && p[-1] <= 'z')
			n += 4;
	return n;
}

nid
alg_simplify(nid e, struct sbuf *conds)
{
	struct rf r, best;
	struct mp dall;
	nid cand[6], res;
	int nc = 0, i, k, j, ch = 0;
	size_t mk = ar_mark();
	int saved_nat = nat;
	nid saved_at[AV];

	memcpy(saved_at, at_node, sizeof saved_at);
	if (!TY_FP(N.ty[e])) {
		return e;
	}
	e = simp_rec(e, conds, 0);
	atoms_reset();
	if (!rf_of(e, &r, &dall)) {
		ar_release(mk);
		nat = saved_nat;
		memcpy(at_node, saved_at, sizeof saved_at);
		return e;
	}
	{
		struct mp nn = mp_expjoin(r.n, &ch);
		if (ch) {
			r.n = nn;
			r = rf_norm(r);
		}
	}
	if (conds)
		rf_conds(dall, r.d, conds);
	best = r;
	cand[nc++] = rf_node(r, 0);
	cand[nc++] = rf_node(r, 1);
	/* sin^2 + cos^2 = 1, cosh^2 - sinh^2 = 1 */
	for (i = 0; i < nat; i++)
		for (k = 0; k < nat; k++) {
			int s = N.op[at_node[i]], c = N.op[at_node[k]];
			if (i == k || N.a[at_node[i]] != N.a[at_node[k]])
				continue;
			if ((s == O_SIN && c == O_COS) || (s == O_COSH &&
			    c == O_SINH)) {
				struct rf t;
				int hy = s == O_COSH;
				t.n = mp_pyth(best.n, i, k, hy);
				t.d = mp_pyth(best.d, i, k, hy);
				t = rf_norm(t);
				if (!alg_ovf && nc < 6) {
					cand[nc++] = rf_node(t, 0);
					if (nc < 6)
						cand[nc++] = rf_node(t, 1);
				}
			}
		}
	res = cand[0];
	for (j = 1; j < nc; j++)
		if (nlen(cand[j]) < nlen(res))
			res = cand[j];
	ar_release(mk);
	nat = saved_nat;
	memcpy(at_node, saved_at, sizeof saved_at);
	return res;
}

/* the function rules only (log|e^u| = u ...), no normal form */
nid
alg_light(nid e)
{
	jmp_buf saved;
	nid r = e;

	memcpy(saved, fail_jmp, sizeof saved);
	if (setjmp(fail_jmp) == 0)
		r = simp_rec(e, NULL, 0);
	memcpy(fail_jmp, saved, sizeof saved);
	return r;
}

nid
alg_tidy(nid e)
{
	return alg_simplify(e, NULL);
}

/* ---- commands ---- */

static void
head(const char *what, nid e)
{
	printf("%s %s\n", what, node_str(e));
}

static int
conv_or_say(nid e, struct rf *r, struct mp *dall)
{
	atoms_reset();
	if (!rf_of(e, r, dall)) {
		if (alg_ovf)
			printf("  the coefficients grow past 64 bits: not done\n");
		else
			printf("  not a rational expression in at most %d atoms\n",
			    AV);
		return 0;
	}
	return 1;
}

static void
show_result(nid in, nid out, struct sbuf *conds)
{
	printf("  = %s\n", node_str(out));
	if (conds && conds->n)
		printf("  holds where %s (cancelled there)\n", conds->p);
	print_check(in, out, conds ? conds->p : NULL);
}

void alg_cmd1(const char *cmd, nid e, nid var);
void alg_cmd2(const char *cmd, nid a, nid b, nid var);
void alg_subst(nid e, const nid *vars, const nid *vals, int n);

static int
var_atom(nid var)
{
	int v;

	if (var == 0)
		return -1;
	v = atom_find(var);
	return v;
}

/* the variable to work in: given, or the only/first variable */
static int
pick_var(nid var, uint32_t used, const char *cmd)
{
	int v, k;

	if (var) {
		v = var_atom(var);
		if (v < 0)
			printf("  note: %s does not occur\n", node_str(var));
		return v;
	}
	for (k = 0; k < nat; k++)
		if ((used & (1u << k)) && N.op[at_node[k]] == O_VAR)
			return k;
	for (k = 0; k < nat; k++)
		if (used & (1u << k))
			return k;
	(void)cmd;
	return -1;
}

static void
cmd_apart(nid e, nid var)
{
	struct rf r;
	struct mp dall;
	struct up n, d, pol;
	int v, nt, i, partial = 0;
	struct pfterm {
		struct up a, f;
		int k;
	};
	extern int up_apart(const struct up *n, const struct up *d,
	    struct up *poly, void *terms, int max, int *partial);
	struct pfterm *t;
	nid acc = 0, x;

	if (!conv_or_say(e, &r, &dall))
		return;
	v = pick_var(var, mp_vars(r.n) | mp_vars(r.d), "apart");
	if (v < 0) {
		printf("  = %s (no variable)\n", node_str(rf_node(r, 0)));
		return;
	}
	x = at_node[v];
	if (!up_of_mp(r.n, v, &n) || !up_of_mp(r.d, v, &d)) {
		printf("  apart needs numeric coefficients in %s (the others are "
		    "symbols)\n", node_str(x));
		return;
	}
	t = ar_alloc(sizeof *t * 64);
	nt = up_apart(&n, &d, &pol, t, 64, &partial);
	if (nt < 0) {
		printf("  the coefficients grow past 64 bits: not done\n");
		return;
	}
	if (!up_iszero(&pol))
		acc = up_node(&pol, x);
	for (i = 0; i < nt; i++) {
		nid den = up_pnode(&t[i].f, x), num;
		aq s;
		struct up fm, am;
		/* a/f^k with f written primitive: scale a */
		up_monic(&fm, &t[i].f);
		{
			int64_t lcf;
			struct up pf;
			/* f = lc(prim) * fm / ... : prim = fm * L */
			aq L;
			up_set(&pf, &fm);
			/* the primitive form of fm is fm * (lcm of denominators)/g */
			L = qi(1);
			{
				int j;
				int64_t l = 1, g = 0;
				for (j = 0; j <= fm.d; j++)
					l = lcm64(l, fm.c[j].d);
				for (j = 0; j <= fm.d; j++)
					g = gcd64(g, qmul(fm.c[j], qi(l)).n);
				L = qmk(l, g ? g : 1);
			}
			lcf = 0;
			(void)lcf;
			(void)pf;
			/* a/fm^k = a L^k / prim^k */
			s = qi(1);
			{
				int j;
				for (j = 0; j < t[i].k; j++)
					s = qmul(s, L);
			}
		}
		up_scale(&am, &t[i].a, s);
		{
			aq c = qi(1);
			int neg;
			if (am.d == 0) {
				c = am.c[0];
				num = 0;
			} else {
				/* the content of the numerator in front */
				int j;
				int64_t l = 1, g = 0;
				for (j = 0; j <= am.d; j++)
					l = lcm64(l, am.c[j].d);
				for (j = 0; j <= am.d; j++)
					g = gcd64(g, qmul(am.c[j], qi(l)).n);
				if (qsgn(am.c[am.d]) < 0)
					g = -g;
				c = qmk(g, l);
				up_scale(&am, &am, qdiv(qi(1), c));
				num = up_node(&am, x);
			}
			if (t[i].k > 1)
				den = POW(den, KI(t[i].k));
			neg = c.n < 0;
			if (neg)
				c.n = -c.n;
			if (num == 0)
				num = KR((double)c.n);
			else if (c.n != 1)
				num = MUL(KR((double)c.n), num);
			if (c.d != 1)
				den = MUL(KR((double)c.d), den);
			num = DIV(num, den);
			if (acc == 0)
				acc = neg ? mk_un(O_NEG, num) : num;
			else
				acc = neg ? SUB(acc, num) : ADD(acc, num);
		}
	}
	if (acc == 0)
		acc = KR(0);
	{
		struct sbuf conds = {0};
		rf_conds(dall, r.d, &conds);
		printf("  = %s\n", node_str(acc));
		if (partial)
			printf("  note: the denominator may factor further (the search "
			    "for factors was cut short)\n");
		if (conds.n)
			printf("  holds where %s (cancelled there)\n", conds.p);
		print_check(e, acc, conds.p);
		free(conds.p);
	}
}

void
alg_cmd1(const char *cmd, nid e, nid var)
{
	struct rf r;
	struct mp dall;
	struct sbuf conds = {0};
	size_t mk = ar_mark();
	nid out;

	head(cmd, e);
	if (!strcmp(cmd, "simplify")) {
		out = alg_simplify(e, &conds);
		show_result(e, out, &conds);
		goto done;
	}
	if (!conv_or_say(e, &r, &dall))
		goto done;
	if (strcmp(cmd, "expand"))
		rf_conds(dall, r.d, &conds);
	if (!strcmp(cmd, "expand") || !strcmp(cmd, "cancel")) {
		if (!strcmp(cmd, "expand"))
			rf_conds(dall, r.d, &conds);
		out = rf_node(r, 0);
		show_result(e, out, &conds);
	} else if (!strcmp(cmd, "factor")) {
		struct mp f[32];
		int m[32], n, partial = 0;
		aq c;
		if (!mp_isconst(r.n, NULL)) {
			n = mp_factor(r.n, &c, f, m, 32, &partial);
			(void)n;
		}
		if (!mp_isconst(r.d, NULL) && !partial) {
			n = mp_factor(r.d, &c, f, m, 32, &partial);
			(void)n;
		}
		out = rf_node(r, 1);
		show_result(e, out, &conds);
		if (partial)
			printf("  note: some factors may factor further (several "
			    "variables, or a degree beyond the search)\n");
		else if (mp_tdeg(r.n) > 1 || mp_tdeg(r.d) > 1)
			printf("  (factors are irreducible over the rationals)\n");
	} else if (!strcmp(cmd, "collect")) {
		int v = pick_var(var, mp_vars(r.n) | mp_vars(r.d), cmd);
		if (v < 0)
			out = rf_node(r, 0);
		else if (mp_isconst(r.d, NULL))
			out = mp_node_in(mp_scale(r.n, qdiv(qi(1), mp_qcontent(r.d))), v);
		else
			out = DIV(mp_node_in(r.n, v), mp_node_in(r.d, v));
		show_result(e, out, &conds);
	} else if (!strcmp(cmd, "together")) {
		if (mp_isconst(r.d, NULL))
			out = rf_node(r, 0);
		else {
			struct rf t = r;
			aq cn = mp_qcontent(t.n);
			nid num = mp_node(mp_scale(t.n, qdiv(qi(1), cn)));
			nid den = mp_fnode(t.d);
			int neg = cn.n < 0;
			if (neg)
				cn.n = -cn.n;
			if (cn.n != 1)
				num = mp_isconst(t.n, NULL) ? KR((double)cn.n) :
				    MUL(KR((double)cn.n), num);
			if (cn.d != 1)
				den = MUL(KR((double)cn.d), den);
			out = DIV(num, den);
			if (neg)
				out = mk_un(O_NEG, out);
		}
		show_result(e, out, &conds);
	} else if (!strcmp(cmd, "apart")) {
		cmd_apart(e, var);
	} else if (!strcmp(cmd, "discriminant")) {
		int v = pick_var(var, mp_vars(r.n), cmd);
		struct mp d;
		if (!mp_isconst(r.d, NULL) || v < 0) {
			printf("  the discriminant needs a polynomial in a variable\n");
			goto done;
		}
		d = mp_discriminant(r.n, v);
		if (alg_ovf) {
			printf("  the coefficients grow past 64 bits: not done\n");
			goto done;
		}
		printf("  in %s (degree %d): %s\n", node_str(at_node[v]),
		    mp_deg(r.n, v), node_str(mp_node(d)));
		if (mp_vars(d) && !mp_isconst(d, NULL)) {
			nid fd = mp_fnode(d);
			if (strcmp(node_str(fd), node_str(mp_node(d))))
				printf("    = %s\n", node_str(fd));
		} else {
			aq c;
			mp_isconst(d, &c);
			printf("  %s\n", qsgn(c) > 0 ? (mp_deg(r.n, v) == 2 ?
			    "> 0: two distinct real roots" : "> 0") : qsgn(c) < 0 ?
			    (mp_deg(r.n, v) == 2 ? "< 0: two complex conjugate roots" :
			    "< 0") : "= 0: a repeated root");
		}
		/* check: the discriminant of a quadratic is b^2 - 4ac etc.; check
		 * by the product of squared root differences when numeric */
		if (mp_isconst(d, NULL)) {
			struct up u;
			if (up_of_mp(r.n, v, &u) && u.d <= 12) {
				double c[16], prod = 1, lc = qd(u.c[u.d]);
				struct root rr[16];
				int nr, i, j, cnt = 0;
				double complex_re[16], complex_im[16];
				for (i = 0; i <= u.d; i++)
					c[i] = qd(u.c[i]);
				nr = poly_roots(c, u.d, rr);
				for (i = 0; i < nr; i++) {
					int mlt;
					for (mlt = 0; mlt < rr[i].mult && cnt < 16; mlt++) {
						complex_re[cnt] = rr[i].re;
						complex_im[cnt++] = rr[i].im;
						if (rr[i].im != 0 && cnt < 16) {
							complex_re[cnt] = rr[i].re;
							complex_im[cnt++] = -rr[i].im;
						}
					}
				}
				if (cnt == u.d) {
					/* lc^(2n-2) prod_{i<j} (ri - rj)^2, real part */
					double pr = 1, pim = 0;
					for (i = 0; i < cnt; i++)
						for (j = i + 1; j < cnt; j++) {
							double dr = complex_re[i] - complex_re[j];
							double di = complex_im[i] - complex_im[j];
							double sr = dr * dr - di * di, si = 2 * dr * di;
							double nr2 = pr * sr - pim * si;
							pim = pr * si + pim * sr;
							pr = nr2;
						}
					prod = pr * pow(lc, 2 * u.d - 2);
					{
						aq dc;
						mp_isconst(d, &dc);
						printf("  check: lc^(2n-2) * product of (ri - rj)^2 "
						    "over the roots = %.10g (%s)\n", prod,
						    fabs(prod - qd(dc)) <= 1e-6 * fmax(1,
						    fabs(qd(dc))) ? "agrees" : "DISAGREES");
					}
				}
			}
		} else {
			/* symbolic: compare with the numeric discriminant at a point */
			double xs[AV], ok = 1;
			int i, k2;
			for (k2 = 0; k2 < nat; k2++)
				xs[k2] = 0.37 + 0.61 * k2;
			{
				struct up u;
				struct mp sp = r.n;
				for (i = 0; i < nat; i++)
					if (i != v) {
						aq qv = qmk(37 + 61 * i, 100);
						xs[i] = qd(qv);
						sp = mp_subst(sp, i, mp_const(qv));
					}
				if (up_of_mp(sp, v, &u)) {
					struct mp dd = mp_discriminant(mp_of_up(&u, v), v);
					aq c1;
					double dv = mp_eval(d, xs);
					if (mp_isconst(dd, &c1))
						printf("  check: at %s the numeric polynomial's "
						    "discriminant is %.10g, the formula gives %.10g "
						    "(%s)\n", "sample values of the coefficients",
						    qd(c1), dv, fabs(qd(c1) - dv) <= 1e-9 *
						    fmax(1, fabs(dv)) ? "agrees" : "DISAGREES");
				}
			}
			(void)ok;
		}
	} else
		fail("%s: unknown", cmd);
done:
	free(conds.p);
	ar_release(mk);
}

void
alg_cmd2(const char *cmd, nid a, nid b, nid var)
{
	struct rf ra, rb;
	struct mp da, db;
	size_t mk = ar_mark();
	aq ca, cb;

	printf("%s %s, %s\n", cmd, node_str(a), node_str(b));
	atoms_reset();
	if (!rf_of(a, &ra, &da) || !rf_of(b, &rb, &db)) {
		printf("  not rational expressions in at most %d atoms\n", AV);
		goto done;
	}
	if (!mp_isconst(ra.d, &ca) || !mp_isconst(rb.d, &cb)) {
		printf("  %s needs polynomials\n", cmd);
		goto done;
	}
	ra.n = mp_scale(ra.n, qdiv(qi(1), ca));
	rb.n = mp_scale(rb.n, qdiv(qi(1), cb));
	if (!strcmp(cmd, "gcd") || !strcmp(cmd, "lcm")) {
		struct mp g, l;
		aq x, y;
		nid out;
		if (mp_isconst(ra.n, &x) && mp_isconst(rb.n, &y) && x.d == 1 &&
		    y.d == 1) {
			int64_t gg = gcd64(x.n, y.n);
			if (!strcmp(cmd, "gcd"))
				printf("  = %lld\n", (long long)gg);
			else
				printf("  = %lld\n", (long long)(gg ? llabs(x.n / gg *
				    y.n) : 0));
			goto done;
		}
		g = mp_gcd(ra.n, rb.n);
		if (alg_ovf) {
			printf("  the coefficients grow past 64 bits: not done\n");
			goto done;
		}
		if (!strcmp(cmd, "gcd")) {
			struct mp qa, qb;
			out = mp_fnode(g);
			printf("  = %s\n", node_str(out));
			if (strcmp(node_str(out), mp_str(g)))
				printf("    = %s\n", mp_str(g));
			/* check: g divides both, the cofactors are coprime */
			if (mp_divexact(ra.n, g, &qa) && mp_divexact(rb.n, g, &qb)) {
				struct mp h = mp_gcd(qa, qb);
				printf("  check: it divides both exactly, and the "
				    "cofactors %s have gcd %s\n", mp_isconst(h, NULL) ?
				    "are coprime:" : "are NOT coprime:", mp_str(h));
			} else
				printf("  check FAILED: it does not divide both\n");
			printf("  (normalized: integer coefficients without a common "
			    "factor, leading coefficient positive)\n");
		} else {
			struct mp q;
			if (!mp_divexact(mp_mul(ra.n, rb.n), g, &q)) {
				printf("  failed\n");
				goto done;
			}
			l = mp_prim(q);
			out = mp_fnode(l);
			printf("  = %s\n", node_str(out));
			if (strcmp(node_str(out), mp_str(l)))
				printf("    = %s\n", mp_str(l));
			{
				struct mp t1, t2;
				printf("  check: %s\n", mp_divexact(l, mp_prim(ra.n), &t1) &&
				    mp_divexact(l, mp_prim(rb.n), &t2) ?
				    "both divide it exactly" : "FAILED");
			}
		}
	} else if (!strcmp(cmd, "resultant")) {
		uint32_t both = mp_vars(ra.n) & mp_vars(rb.n);
		int v = pick_var(var, both ? both : mp_vars(ra.n) | mp_vars(rb.n),
		    cmd);
		struct mp res;
		if (v < 0) {
			printf("  the resultant needs a variable\n");
			goto done;
		}
		res = mp_resultant(ra.n, rb.n, v);
		if (alg_ovf) {
			printf("  the coefficients grow past 64 bits: not done\n");
			goto done;
		}
		printf("  in %s: %s\n", node_str(at_node[v]), mp_str(res));
		if (!mp_isconst(res, NULL)) {
			nid fr = mp_fnode(res);
			if (strcmp(node_str(fr), mp_str(res)))
				printf("    = %s\n", node_str(fr));
		} else {
			aq c;
			mp_isconst(res, &c);
			printf("  %s\n", c.n ? "nonzero: no common root" :
			    "zero: a common root (or factor)");
		}
		/* check numerically: res = lc(a)^m prod b(ri) over a's roots */
		{
			struct up ua, ub;
			aq c;
			if (mp_isconst(res, &c) && up_of_mp(ra.n, v, &ua) &&
			    up_of_mp(rb.n, v, &ub) && ua.d >= 1 && ua.d <= 12) {
				double cc[16], pr = 1, pi = 0;
				struct root rr[16];
				int nr, i, mm;
				for (i = 0; i <= ua.d; i++)
					cc[i] = qd(ua.c[i]);
				nr = poly_roots(cc, ua.d, rr);
				for (i = 0; i < nr; i++)
					for (mm = 0; mm < rr[i].mult; mm++) {
						int s;
						for (s = 0; s < (rr[i].im != 0 ? 2 : 1); s++) {
							double xr = rr[i].re, xi = s ? -rr[i].im :
							    rr[i].im, vr = 0, vi = 0, t;
							int j;
							for (j = ub.d; j >= 0; j--) {
								t = vr * xr - vi * xi + qd(ub.c[j]);
								vi = vr * xi + vi * xr;
								vr = t;
							}
							t = pr * vr - pi * vi;
							pi = pr * vi + pi * vr;
							pr = t;
						}
					}
				pr *= pow(qd(ua.c[ua.d]), ub.d);
				printf("  check: lc(P)^deg(Q) * product of Q over the roots "
				    "of P = %.10g (%s)\n", pr, fabs(pr - qd(c)) <= 1e-6 *
				    fmax(1, fabs(qd(c))) ? "agrees" : "DISAGREES");
			}
		}
	} else if (!strcmp(cmd, "quo") || !strcmp(cmd, "rem") ||
	    !strcmp(cmd, "divide")) {
		int v = pick_var(var, mp_vars(ra.n) | mp_vars(rb.n), cmd);
		struct mp q = mp_zero(), r = ra.n, lb;
		int db2, it = 0;
		nid qn, rn;
		if (rb.n.n == 0) {
			printf("  division by zero\n");
			goto done;
		}
		if (v < 0) {
			aq x, y;
			mp_isconst(ra.n, &x);
			mp_isconst(rb.n, &y);
			printf("  = %s\n", node_str(q_cnode(qdiv(x, y))));
			goto done;
		}
		db2 = mp_deg(rb.n, v);
		lb = mp_lcv(rb.n, v);
		while (r.n && mp_deg(r, v) >= db2 && it++ < 200) {
			struct mp lr = mp_lcv(r, v), c, mo;
			int dr = mp_deg(r, v);
			if (!mp_divexact(lr, lb, &c)) {
				printf("  the leading coefficient %s does not divide; "
				    "divide by a polynomial with a numeric leading "
				    "coefficient in %s\n", mp_str(lb), node_str(at_node[v]));
				goto done;
			}
			mo = mp_const(qi(1));
			mo.t[0].e[v] = (uint8_t)(dr - db2);
			c = mp_mul(c, mo);
			q = mp_add(q, c);
			r = mp_sub(r, mp_mul(c, rb.n));
		}
		qn = mp_node_in(q, v);
		rn = mp_node_in(r, v);
		if (!strcmp(cmd, "quo"))
			printf("  = %s\n", node_str(qn));
		else if (!strcmp(cmd, "rem"))
			printf("  = %s\n", node_str(rn));
		else
			printf("  quotient %s, remainder %s\n", node_str(qn),
			    node_str(rn));
		printf("  (in %s: P = Q*quotient + remainder, deg remainder < %d)\n",
		    node_str(at_node[v]), db2);
		/* check */
		{
			nid back = ADD(MUL(b, qn), rn);
			print_check(a, back, NULL);
		}
	} else
		fail("%s: unknown", cmd);
done:
	ar_release(mk);
}

void
alg_subst(nid e, const nid *vars, const nid *vals, int n)
{
	nid map[NSLOT], r, s;
	struct sbuf conds = {0};
	int i;

	memset(map, 0, sizeof map);
	printf("%s with", node_str(e));
	for (i = 0; i < n; i++) {
		printf("%s %s = %s", i ? "," : "", node_str(vars[i]),
		    node_str(vals[i]));
		map[N.ki[vars[i]]] = vals[i];
	}
	printf("\n");
	r = subst(e, map, NSLOT);
	s = alg_simplify(r, &conds);
	printf("  = %s\n", node_str(s));
	if (conds.n)
		printf("  holds where %s\n", conds.p);
	print_check(r, s, NULL);
	free(conds.p);
}
