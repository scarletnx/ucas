/*
 * kauto.c - decision procedures and proof search that return kernel
 * theorems (prove.h).  Nothing here is trusted: every result is built
 * with kernel.c's rules.
 *
 *   linear arithmetic over Z: Fourier-Motzkin with a certificate (the
 *     multipliers become one kr_lin), with products of two hypotheses
 *     (axiom mul_nonneg) and case splits n <= 0 or 1 <= n (axiom
 *     int_discrete) when the rational relaxation is not enough;
 *   abelian-group equalities: elimination on unit pivots over the
 *     equations in context (a kr_lin certificate);
 *   subgroup membership: a linear combination of members (lemmas
 *     sub_zero, sub_add, sub_smul);
 *   diagram chasing: goal-directed chaining through the facts in
 *     context and the lemmas offered, with unification modulo the
 *     normal forms (metavariables solved through linear terms),
 *     opening existentials in context, congruence of arguments.
 */
#include "ucas.h"
#include "prove.h"

#include <stdlib.h>
#include <string.h>

char ka_why[512];
uint64_t ka_budget;

#define MAXM 512
static kt mval[MAXM];
static uint32_t nmeta;
static uint32_t trail[MAXM];
static int ntrail;

static kt
newmeta(int sort)
{
	if (nmeta >= MAXM)
		return KNONE;
	mval[nmeta] = 0;
	return kt_meta(nmeta++, sort);
}

static void
bindm(uint32_t i, kt v)
{
	mval[i] = v;
	trail[ntrail++] = i;
}

static void
undo(int mark)
{
	while (ntrail > mark)
		mval[trail[--ntrail]] = 0;
}

static kt
resolve(kt t)
{
	if (!kt_hasmeta(t))
		return t;
	return kt_metasub(t, mval, nmeta);
}

/* ---- unification modulo normal forms ---- */

static int uni(kt p, kt t);

static int
isrelsort(kt t)
{
	return KSORT(t) == KS_INT || KSORT(t) == KS_EL;
}

/* p, t terms of sort int or el; t ground */
static int
luni(kt p, kt t)
{
	kt pa[64], pc[64], ta[64], tc[64], ground = KNONE;
	int np, nt, i, j, nm = 0, mi = -1, used[64], pm[64];
	int mark = ntrail;

	np = kn_split(p, pa, pc, 64);
	nt = kn_split(t, ta, tc, 64);
	if (np < 0 || nt < 0)
		return 0;
	for (i = 0; i < np; i++) {
		pm[i] = kt_hasmeta(pa[i]) || kt_hasmeta(pc[i]);
		if (pm[i]) {
			nm++;
			mi = i;
		} else {
			kt s = KSORT(t) == KS_INT ? kt_mul(pc[i], pa[i]) :
			    kt_mul(pc[i], pa[i]);
			if (KSORT(t) == KS_INT && pa[i] == kt_num(1))
				s = pc[i];
			ground = ground == KNONE ? s : kt_add(ground, s);
		}
	}
	if (nm == 0)
		return kn_norm(kf_eq(p, t)) == kf_true();
	/* a bare metavariable with coefficient +-1: match the other
	 * metavariable summands, then solve for it */
	{
		int bi = -1;
		for (i = 0; i < np; i++)
			if (KOP(pa[i]) == KO_META && KOP(pc[i]) == KO_NUM &&
			    (KT.num[pc[i]] == 1 || KT.num[pc[i]] == -1) &&
			    KSORT(pa[i]) == KSORT(t) && !mval[KT.aux[pa[i]]]) {
				bi = i;
				break;
			}
		if (bi >= 0) {
			kt rest = KNONE, v;
			int ok = 1;
			memset(used, 0, sizeof used);
			for (i = 0; i < np && ok; i++) {
				if (i == bi || !pm[i])
					continue;
				ok = 0;
				for (j = 0; j < nt && !ok; j++) {
					int m2 = ntrail;
					if (used[j])
						continue;
					if (uni(pa[i], ta[j]) && uni(pc[i], tc[j])) {
						used[j] = 1;
						ok = 1;
					} else
						undo(m2);
				}
			}
			if (ok) {
				for (i = 0; i < np; i++) {
					kt sterm;
					if (i == bi)
						continue;
					sterm = KSORT(t) == KS_INT && pa[i] == kt_num(1) ?
					    pc[i] : kt_mul(pc[i], pa[i]);
					sterm = resolve(sterm);
					rest = rest == KNONE ? sterm : kt_add(rest, sterm);
				}
				v = rest == KNONE ? t : kt_sub(t, rest);
				if (KT.num[pc[bi]] == -1)
					v = kt_neg(v);
				v = kn_norm(v);
				if (!KT.lb[v] && !kt_hasmeta(v)) {
					bindm(KT.aux[pa[bi]], v);
					return 1;
				}
			}
			undo(mark);
		}
	}
	(void)mi;
	(void)ground;
	/* match the ground summands exactly, the rest pairwise */
	memset(used, 0, sizeof used);
	for (i = 0; i < np; i++) {
		if (pm[i])
			continue;
		for (j = 0; j < nt; j++)
			if (!used[j] && ta[j] == pa[i] && tc[j] == pc[i])
				break;
		if (j == nt)
			return 0;
		used[j] = 1;
	}
	if (nt - (np - nm) != nm)
		return 0;
	for (i = 0; i < np; i++) {
		int ok = 0;
		if (!pm[i])
			continue;
		for (j = 0; j < nt && !ok; j++) {
			int m2 = ntrail;
			if (used[j])
				continue;
			if (uni(pa[i], ta[j]) && uni(pc[i], tc[j])) {
				used[j] = 1;
				ok = 1;
			} else
				undo(m2);
		}
		if (!ok) {
			undo(mark);
			return 0;
		}
	}
	return 1;
}

static int
uni(kt p, kt t)
{
	int i, mark = ntrail;

	p = resolve(p);
	if (!kt_hasmeta(p))
		return p == t || kn_norm(p) == kn_norm(t);
	if (KOP(p) == KO_META) {
		if (KSORT(p) != KSORT(t) || KT.lb[t])
			return 0;
		bindm(KT.aux[p], t);
		return 1;
	}
	if (KSORT(p) != KS_PROP && isrelsort(p)) {
		kt pn = kn_norm(p);
		if (KSORT(p) != KSORT(t))
			return 0;
		t = kn_norm(t);
		if (KOP(pn) == KO_ADD || KOP(pn) == KO_NEG || (KOP(pn) == KO_MUL &&
		    (KSORT(pn) == KS_EL || KOP(KA(pn, 0)) == KO_NUM)) ||
		    KOP(t) == KO_ADD || KOP(t) == KO_NEG || KOP(t) == KO_ZERO ||
		    KOP(t) == KO_NUM || (KOP(t) == KO_MUL && (KSORT(t) ==
		    KS_EL || KOP(KA(t, 0)) == KO_NUM)))
			return luni(pn, t);
		p = pn;
		if (KOP(p) == KO_META) {
			bindm(KT.aux[p], t);
			return 1;
		}
	}
	if (KSORT(p) == KS_PROP)
		p = kn_norm(p);
	if (KOP(p) != KOP(t) || KT.aux[p] != KT.aux[t] || KN(p) != KN(t))
		return 0;
	if (KOP(p) == KO_EQ && isrelsort(KA(p, 0))) {
		if (uni(KA(p, 0), KA(t, 0)))
			return 1;
		undo(mark);
		return uni(KA(p, 0), kn_norm(kt_neg(KA(t, 0))));
	}
	for (i = 0; i < KN(p); i++)
		if (!uni(KA(p, i), KA(t, i))) {
			undo(mark);
			return 0;
		}
	return 1;
}

/* ---- context ---- */

static kthm prove(struct kctx *c, kt g, int depth);

void
ka_add(struct kctx *c, kthm th, uint32_t name)
{
	kt f;

	if (th == KNONE)
		return;
	if (c->n >= 512)
		fail("too many facts in a proof context");
	c->fact[c->n] = th;
	c->name[c->n++] = name;
	f = kthm_concl(th);
	if (KOP(f) == KO_AND) {
		ka_add(c, kr_ande1(th), 0);
		ka_add(c, kr_ande2(th), 0);
	}
}

struct opened {
	kthm ex[64];
	kt var[64];
	kt asm_[64];
	int n;
};

/* open the existential facts from index i on */
static void
open_ex(struct kctx *c, int i, struct opened *o)
{
	for (; i < c->n; i++) {
		kt f = kthm_concl(c->fact[i]);
		if (KOP(f) == KO_EX && o->n < 64) {
			kt v = kt_fresh(KT.hint[f] ? sym_str(KT.hint[f]) : "w",
			    (int)KT.aux[f]);
			kt a = kt_inst(KA(f, 0), v);
			kthm th = kr_assume(a);
			o->ex[o->n] = c->fact[i];
			o->var[o->n] = v;
			o->asm_[o->n++] = a;
			ka_add(c, th, c->name[i]);
		}
	}
}

static kthm
discharge(kthm th, struct opened *o)
{
	int k;
	uint32_t h;

	for (k = o->n - 1; k >= 0 && th; k--) {
		int has = 0;
		for (h = 0; h < kthm_nhyps(th); h++)
			if (kthm_hyp(th, h) == o->asm_[k])
				has = 1;
		if (has)
			th = kr_exe(o->ex[k], o->var[k], th);
	}
	return th;
}

static kthm
truth(void)
{
	return kr_conv(kr_refl(kt_num(0)), kf_true());
}

kthm
ka_close(struct kctx *c, kt g)
{
	int i;
	kt gn = kn_norm(g);

	if (gn == kf_true())
		return kr_conv(truth(), g);
	for (i = c->n - 1; i >= 0; i--)
		if (kn_norm(kthm_concl(c->fact[i])) == gn)
			return kr_conv(c->fact[i], g);
	for (i = 0; i < c->n; i++)
		if (KOP(kthm_concl(c->fact[i])) == KO_FALSE)
			return kr_falsee(c->fact[i], g);
	return KNONE;
}

/* ---- linear combinations of equations ---- */

static kt
relterm(kt c, int *ge)
{
	switch (KOP(c)) {
	case KO_EQ:
		if (!isrelsort(KA(c, 0)))
			return KNONE;
		*ge = 0;
		return kt_sub(KA(c, 0), KA(c, 1));
	case KO_LE:
		*ge = 1;
		return kt_sub(KA(c, 1), KA(c, 0));
	case KO_LT:
		*ge = 1;
		return kt_sub(kt_sub(KA(c, 1), KA(c, 0)), kt_num(1));
	}
	return KNONE;
}

/* f(.., s, ..) = f(.., t, ..) from s = t (congruence by subst) */
static kthm
lift_eq(kthm eq, kt atom, int k)
{
	kt c = kthm_concl(eq), x[16], pat, l;
	int q;

	for (q = 0; q < KN(atom); q++)
		x[q] = KA(atom, q);
	x[k] = KA(c, 0);
	l = kt_mk(KOP(atom), KSORT(atom), KT.aux[atom], 0, x, KN(atom));
	x[k] = kt_bvar(0, KSORT(KA(c, 0)));
	pat = kf_eq(l, kt_mk(KOP(atom), KSORT(atom), KT.aux[atom], 0, x,
	    KN(atom)));
	return kr_subst(eq, pat, kr_refl(l));
}

static int
has_atom(kt t, kt a)
{
	kt sa[64], sc[64];
	int n = kn_split(t, sa, sc, 64), i;

	for (i = 0; i < n; i++)
		if (sa[i] == a)
			return 1;
	return 0;
}

/* collect the application atoms of t (one level) */
static int
app_atoms(kt t, kt *out, int n, int max)
{
	kt sa[64], sc[64];
	int m = kn_split(t, sa, sc, 64), i, j, dup;

	for (i = 0; i < m; i++) {
		kt a = sa[i];
		if (KSORT(t) == KS_INT)
			continue;
		if (KOP(a) != KO_APP)
			continue;
		for (j = 0, dup = 0; j < n; j++)
			dup |= out[j] == a;
		if (!dup && n < max)
			out[n++] = a;
	}
	return n;
}

static kthm eqsolve_int(struct kctx *c, kt g);

/* m = d * q for int monomials (products of atoms): q, or KNONE */
static kt
mono_div(kt m, kt d)
{
	kt a[64], b[64];
	int na = 0, nb = 0, i, j, used[64];
	kt q = KNONE;

	if (d == kt_num(1))
		return m;
	while (KOP(m) == KO_MUL && na < 63) {
		a[na++] = KA(m, 1);
		m = KA(m, 0);
	}
	a[na++] = m;
	while (KOP(d) == KO_MUL && nb < 63) {
		b[nb++] = KA(d, 1);
		d = KA(d, 0);
	}
	b[nb++] = d;
	memset(used, 0, sizeof used);
	for (j = 0; j < nb; j++) {
		for (i = 0; i < na; i++)
			if (!used[i] && a[i] == b[j])
				break;
		if (i == na)
			return KNONE;
		used[i] = 1;
	}
	for (i = 0; i < na; i++)
		if (!used[i])
			q = q == KNONE ? a[i] : kt_mul(q, a[i]);
	return q == KNONE ? kt_num(1) : q;
}

/* a = b from the equations in context: Gaussian elimination on unit
 * pivots, equations lifted through applications (congruence), and
 * coefficients that vanish modulo the integer equations */
static kthm
eqsolve2(struct kctx *c, kt g, int lift)
{
	kthm fth[96];
	kt pv[96], pu[96];
	kt T[96], prov[96][96], R, M[96], fa[64], fc[64], ra[64], rc[64];
	int nf = 0, i, j, k, q, sort, ge, done[96];

	if (KOP(g) != KO_EQ || !isrelsort(KA(g, 0)))
		return KNONE;
	sort = KSORT(KA(g, 0));
	for (i = c->n - 1; i >= 0 && nf < 64; i--) {
		kt f = kthm_concl(c->fact[i]);
		if (KOP(f) == KO_EQ && KSORT(KA(f, 0)) == sort) {
			for (j = 0; j < nf; j++)
				if (kn_norm(kthm_concl(fth[j])) == kn_norm(f))
					break;
			if (j == nf)
				fth[nf++] = c->fact[i];
		}
	}
	R = kn_norm(kt_sub(KA(g, 0), KA(g, 1)));
	if (lift && sort == KS_EL) {
		kt ats[32];
		int na = app_atoms(R, ats, 0, 32), n0 = nf;
		for (i = 0; i < n0; i++)
			na = app_atoms(relterm(kthm_concl(fth[i]), &ge), ats, na,
			    32);
		for (i = 0; i < na; i++)
			for (k = 0; k < KN(ats[i]); k++) {
				kt arg = KA(ats[i], k);
				if (KSORT(arg) != KS_EL)
					continue;
				for (j = 0; j < n0 && nf < 96; j++) {
					kt f = kthm_concl(fth[j]), d;
					int hom = (ksyms[KT.aux[ats[i]]].flags & KSF_HOM) &&
					    k == KN(ats[i]) - 1;
					if (KN(f) != 2)
						continue;
					d = kn_norm(kt_sub(KA(f, 0), KA(f, 1)));
					if (kn_norm(KA(f, 0)) == kn_norm(arg) ||
					    (hom && has_atom(d, arg))) {
						kthm l = lift_eq(fth[j], ats[i], k);
						if (l)
							fth[nf++] = l;
					} else if (kn_norm(KA(f, 1)) == kn_norm(arg)) {
						kthm sym = kr_subst(fth[j], kf_eq(kt_bvar(0,
						    sort), KA(f, 0)), kr_refl(KA(f, 0)));
						kthm l = sym ? lift_eq(sym, ats[i], k) : KNONE;
						if (l)
							fth[nf++] = l;
					}
				}
			}
	}
	for (i = 0; i < nf; i++) {
		T[i] = kn_norm(relterm(kthm_concl(fth[i]), &ge));
		for (j = 0; j < nf; j++)
			prov[i][j] = kt_num(i == j);
		M[i] = kt_num(0);
		done[i] = 0;
		pv[i] = KNONE;
	}
	for (;;) {
		int progress = 0;
		for (i = 0; i < nf; i++) {
			int nt, piv = -1, best = -1, nr, score;
			kt u, cR;
			if (done[i])
				continue;
			nt = kn_split(T[i], fa, fc, 64);
			if (nt <= 0) {
				done[i] = 1;
				continue;
			}
			nr = kn_split(R, ra, rc, 64);
			for (j = 0; j < nt; j++) {
				int inR = 0, inO = 0;
				if (KOP(fc[j]) != KO_NUM || (KT.num[fc[j]] != 1 &&
				    KT.num[fc[j]] != -1) || (sort == KS_INT &&
				    fa[j] == kt_num(1)))
					continue;
				for (q = 0; q < nr; q++)
					inR |= ra[q] == fa[j] || (sort == KS_INT &&
					    ra[q] != kt_num(1) &&
					    mono_div(ra[q], fa[j]) != KNONE);
				for (k = 0; k < nf && !inO; k++)
					if (k != i && !done[k])
						inO = has_atom(T[k], fa[j]);
				/* eliminate defined atoms first, then the goal's */
				score = inO && !inR ? 3 : inR ? 2 : 1;
				if (score > best) {
					best = score;
					piv = j;
				}
			}
			if (piv < 0)
				continue;
			done[i] = 1;
			progress = 1;
			u = fc[piv];
			pv[i] = fa[piv];
			pu[i] = u;
			if (getenv("KDEBUG2"))
				fprintf(stderr, "eq: R=%s pivot %s in %s\n", kt_str(R),
				    kt_str(fa[piv]), kt_str(T[i]));
			cR = KNONE;
			for (q = 0; q < nr; q++)
				if (ra[q] == fa[piv])
					cR = rc[q];
			if (cR) {
				kt m = kn_norm(kt_mul(cR, u));
				R = kn_norm(kt_sub(R, kt_mul(m, T[i])));
				for (k = 0; k < nf; k++)
					M[k] = kn_norm(kt_add(M[k], kt_mul(m, prov[i][k])));
			}
			/* int: the pivot also divides monomials of R */
			if (sort == KS_INT) {
				int guard;
				for (guard = 0; guard < 16; guard++) {
					kt qm = KNONE, cq = KNONE;
					nr = kn_split(R, ra, rc, 64);
					for (q = 0; q < nr && !qm; q++)
						if (ra[q] != kt_num(1) && ra[q] != fa[piv] &&
						    (qm = mono_div(ra[q], fa[piv])) != KNONE)
							cq = rc[q];
					if (!qm)
						break;
					{
						kt m = kn_norm(kt_mul(kt_mul(cq, u), qm));
						R = kn_norm(kt_sub(R, kt_mul(m, T[i])));
						for (k = 0; k < nf; k++)
							M[k] = kn_norm(kt_add(M[k], kt_mul(m,
							    prov[i][k])));
					}
				}
			}
			for (j = 0; j < nf; j++) {
				int n2;
				if (j == i || done[j])
					continue;
				n2 = kn_split(T[j], ra, rc, 64);
				for (q = 0; q < n2; q++)
					if (ra[q] == fa[piv]) {
						kt m = kn_norm(kt_mul(rc[q], u));
						T[j] = kn_norm(kt_sub(T[j], kt_mul(m, T[i])));
						for (k = 0; k < nf; k++)
							prov[j][k] = kn_norm(kt_sub(prov[j][k],
							    kt_mul(m, prov[i][k])));
						break;
					}
			}
		}
		if (!progress)
			break;
	}
	/* reduce R again by every pivot (one may divide a later monomial) */
	{
		int pass, ch = 1;
		for (pass = 0; pass < 8 && ch; pass++) {
			ch = 0;
			for (i = 0; i < nf; i++) {
				int nr, guard;
				if (!done[i] || !pv[i])
					continue;
				for (guard = 0; guard < 16; guard++) {
					kt qm = KNONE, cq = KNONE, m;
					nr = kn_split(R, ra, rc, 64);
					for (q = 0; q < nr && !qm; q++) {
						if (ra[q] == pv[i])
							qm = kt_num(1);
						else if (sort == KS_INT && ra[q] != kt_num(1))
							qm = mono_div(ra[q], pv[i]);
						if (qm)
							cq = rc[q];
					}
					if (!qm)
						break;
					m = kn_norm(kt_mul(kt_mul(cq, pu[i]), qm));
					R = kn_norm(kt_sub(R, kt_mul(m, T[i])));
					for (k = 0; k < nf; k++)
						M[k] = kn_norm(kt_add(M[k], kt_mul(m,
						    prov[i][k])));
					ch = 1;
				}
			}
		}
	}
	/* what is left may be +-1 times a remaining equation */
	if (kn_split(R, ra, rc, 64) != 0)
		for (j = 0; j < nf; j++) {
			int sg;
			if (done[j])
				continue;
			for (sg = -1; sg <= 1; sg += 2)
				if (kn_split(kn_norm(kt_sub(R, kt_mul(kt_num(sg),
				    T[j]))), ra, rc, 64) == 0) {
					for (k = 0; k < nf; k++)
						M[k] = kn_norm(kt_add(M[k],
						    kt_mul(kt_num(sg), prov[j][k])));
					R = kn_norm(kt_sub(R, kt_mul(kt_num(sg), T[j])));
					j = nf;
					break;
				}
		}
	(void)done;
	{
		kthm used[64], extra = KNONE;
		kt co[64];
		int nu = 0, nr = kn_split(R, ra, rc, 64);
		if (nr < 0)
			return KNONE;
		if (nr > 0) {
			/* the coefficients left must vanish (integer equations) */
			kthm acc = KNONE;
			if (sort != KS_EL)
				return KNONE;
			for (q = 0; q < nr; q++) {
				kthm z = eqsolve_int(c, kf_eq(rc[q], kt_num(0))), t;
				kt l;
				if (!z)
					return KNONE;
				l = kt_mul(rc[q], ra[q]);
				t = kr_subst(z, kf_eq(l, kt_mul(kt_bvar(0, KS_INT),
				    ra[q])), kr_refl(l));
				if (!t)
					return KNONE;
				t = kr_conv(t, kf_eq(l, kt_zero()));
				acc = acc ? kr_conv(kr_lin((kthm[]){acc, t},
				    (kt[]){kt_num(1), kt_num(1)}, 2),
				    kf_eq(kt_add(KA(kthm_concl(acc), 0), l),
				    kt_zero())) : t;
				if (!acc)
					return KNONE;
			}
			extra = acc;
		}
		for (k = 0; k < nf && nu < 63; k++)
			if (kn_norm(kf_eq(M[k], kt_num(0))) != kf_true()) {
				used[nu] = fth[k];
				co[nu++] = M[k];
			}
		if (extra) {
			used[nu] = extra;
			co[nu++] = kt_num(1);
		}
		if (nu == 0)
			return KNONE;
		{
			kthm th = kr_lin(used, co, nu);
			return th ? kr_conv(th, g) : KNONE;
		}
	}
}

static kthm
eqsolve_int(struct kctx *c, kt g)
{
	if (kn_norm(g) == kf_true())
		return kr_conv(truth(), g);
	return eqsolve2(c, g, 0);
}

static kthm
eqsolve(struct kctx *c, kt g)
{
	kthm th;

	if (kn_norm(g) == kf_true())
		return kr_conv(truth(), g);
	th = eqsolve2(c, g, 0);
	if (!th && KSORT(KA(g, 0)) == KS_EL)
		th = eqsolve2(c, g, 1);
	return th;
}

/* ---- Fourier-Motzkin with certificates ---- */

#define FV 48	/* atoms */
#define FF 64	/* facts */
#define FR 600	/* rows */
struct row {
	int64_t a[FV], c, pv[FF];
	int eq;
};
static struct row *rows;

static int
fm_ok(__int128 v)
{
	return v < ((__int128)1 << 62) && v > -((__int128)1 << 62);
}

static int64_t
g64(int64_t a, int64_t b)
{
	if (a < 0)
		a = -a;
	if (b < 0)
		b = -b;
	while (b) {
		int64_t t = a % b;
		a = b;
		b = t;
	}
	return a;
}

/* r = x*A + y*B; 0 on overflow */
static int
combine(struct row *r, const struct row *A, int64_t x, const struct row *B,
    int64_t y, int nv, int nfct)
{
	int i;
	__int128 v;
	int64_t g = 0;

	for (i = 0; i < nv; i++) {
		v = (__int128)x * A->a[i] + (__int128)y * B->a[i];
		if (!fm_ok(v))
			return 0;
		r->a[i] = (int64_t)v;
	}
	v = (__int128)x * A->c + (__int128)y * B->c;
	if (!fm_ok(v))
		return 0;
	r->c = (int64_t)v;
	for (i = 0; i < nfct; i++) {
		v = (__int128)x * A->pv[i] + (__int128)y * B->pv[i];
		if (!fm_ok(v))
			return 0;
		r->pv[i] = (int64_t)v;
	}
	r->eq = A->eq && B->eq;
	for (i = 0; i < nv; i++)
		g = g64(g, r->a[i]);
	g = g64(g, r->c);
	for (i = 0; i < nfct; i++)
		g = g64(g, r->pv[i]);
	if (g > 1) {
		for (i = 0; i < nv; i++)
			r->a[i] /= g;
		r->c /= g;
		for (i = 0; i < nfct; i++)
			r->pv[i] /= g;
	}
	return 1;
}

static int
contradicts(const struct row *r, int nv)
{
	int i;

	for (i = 0; i < nv; i++)
		if (r->a[i])
			return 0;
	return r->eq ? r->c != 0 : r->c < 0;
}

/* facts are relations (EQ, LE, LT over int); a certificate, or -1 */
static int
fm(const kthm *f, int nf, int64_t *lam)
{
	kt atoms[FV], sa[64], sc[64], t;
	int nv = 0, nr = 0, i, j, k, ge, n;
	int found = -1;

	if (nf > FF)
		nf = FF;
	if (!rows)
		rows = xmalloc(2 * FR * sizeof *rows);
	for (i = 0; i < nf; i++) {
		struct row *r = &rows[nr];
		t = relterm(kthm_concl(f[i]), &ge);
		if (t == KNONE)
			continue;
		n = kn_split(t, sa, sc, 64);
		if (n < 0)
			continue;
		memset(r, 0, sizeof *r);
		r->eq = !ge;
		r->pv[i] = 1;
		for (j = 0; j < n; j++) {
			if (sa[j] == kt_num(1)) {
				r->c = KT.num[sc[j]];
				continue;
			}
			for (k = 0; k < nv && atoms[k] != sa[j]; k++)
				;
			if (k == nv) {
				if (nv == FV)
					return -1;
				atoms[nv++] = sa[j];
			}
			r->a[k] = KT.num[sc[j]];
		}
		nr++;
	}
	for (;;) {
		int var = -1, ei = -1;
		long best = -1;
		for (i = 0; i < nr; i++)
			if (contradicts(&rows[i], nv)) {
				found = i;
				goto done;
			}
		/* an equation with a variable: substitute it away */
		for (i = 0; i < nr && ei < 0; i++)
			if (rows[i].eq)
				for (k = 0; k < nv; k++)
					if (rows[i].a[k]) {
						ei = i;
						var = k;
						break;
					}
		if (ei >= 0) {
			struct row e = rows[ei];
			int64_t a = e.a[var], s = a > 0 ? 1 : -1;
			rows[ei] = rows[--nr];
			for (i = 0; i < nr; i++) {
				int64_t b = rows[i].a[var];
				struct row r;
				if (!b)
					continue;
				if (rows[i].eq) {
					if (!combine(&r, &rows[i], a, &e, -b, nv, nf))
						return -1;
				} else if (!combine(&r, &rows[i], a * s, &e,
				    -b * s, nv, nf))
					return -1;
				rows[i] = r;
			}
			continue;
		}
		/* Fourier-Motzkin on the cheapest variable */
		for (k = 0; k < nv; k++) {
			long p = 0, m = 0;
			for (i = 0; i < nr; i++) {
				p += rows[i].a[k] > 0;
				m += rows[i].a[k] < 0;
			}
			if ((p || m) && (best < 0 || p * m < best)) {
				best = p * m;
				var = k;
			}
		}
		if (var < 0)
			break;
		{
			int nn = 0;
			struct row *out = rows + FR;
			for (i = 0; i < nr; i++)
				if (rows[i].a[var] == 0) {
					if (nn >= FR)
						return -1;
					out[nn++] = rows[i];
				}
			for (i = 0; i < nr; i++) {
				if (rows[i].a[var] <= 0)
					continue;
				for (j = 0; j < nr; j++) {
					if (rows[j].a[var] >= 0)
						continue;
					if (nn >= FR)
						return -1;
					if (!combine(&out[nn], &rows[i],
					    -rows[j].a[var], &rows[j],
					    rows[i].a[var], nv, nf))
						return -1;
					nn++;
				}
			}
			memcpy(rows, out, (size_t)nn * sizeof *rows);
			nr = nn;
		}
	}
	return -1;
done:
	for (i = 0; i < nf; i++)
		lam[i] = rows[found].pv[i];
	return found;
}

static kthm
fm_proof(const kthm *f, int nf)
{
	int64_t lam[FF];
	kthm used[FF];
	kt co[FF];
	int i, n = 0, ge;
	kthm th;

	if (nf > FF)
		nf = FF;
	if (fm(f, nf, lam) < 0)
		return KNONE;
	for (i = 0; i < nf; i++)
		if (lam[i]) {
			if (relterm(kthm_concl(f[i]), &ge) == KNONE ||
			    (ge && lam[i] < 0))
				return KNONE;
			used[n] = f[i];
			co[n++] = kt_num(lam[i]);
		}
	if (!n)
		return KNONE;
	th = kr_lin(used, co, n);
	return th ? kr_conv(th, kf_false()) : KNONE;
}

static kthm
axiom_inst(const char *name, const kt *a, int n)
{
	kthm th = kr_axiom(name);
	int i;

	for (i = 0; i < n && th; i++)
		th = kr_alle(th, a[i]);
	return th;
}

/* from not (s <= t) (or not s < t), the integer complement */
static kthm
neg_le(kthm th)
{
	kt f = KA(kthm_concl(th), 0), d, a;
	int ge;
	kthm disj, c1, c2, h1, h2;

	d = relterm(f, &ge);	/* f says 0 <= d */
	if (d == KNONE || !ge)
		return KNONE;
	/* int_discrete at -d: -d <= 0 or 1 <= -d */
	a = kt_neg(d);
	disj = axiom_inst("int_discrete", &a, 1);
	if (!disj)
		return KNONE;
	{
		kt dj = kthm_concl(disj);
		kt want = kf_le(kt_num(0), kt_sub(kt_neg(d), kt_num(1)));
		h1 = kr_assume(KA(dj, 0));
		h2 = kr_assume(KA(dj, 1));
		c1 = kr_falsee(kr_note(th, kr_conv(h1, f)), want);
		c2 = kr_conv(h2, want);
		if (!c1 || !c2)
			return KNONE;
		return kr_ore(disj, c1, c2);
	}
}

static int
isintrel(kt f)
{
	return (KOP(f) == KO_LE || KOP(f) == KO_LT || KOP(f) == KO_EQ) &&
	    KSORT(KA(f, 0)) == KS_INT;
}

static kthm refute(struct kctx *c, int depth);

static int
int_facts(struct kctx *c, kthm *out, int max)
{
	int i, n = 0;

	for (i = 0; i < c->n && n < max; i++) {
		kt f = kthm_concl(c->fact[i]);
		if (isintrel(f))
			out[n++] = c->fact[i];
		else if (KOP(f) == KO_NOT && isintrel(KA(f, 0)) &&
		    KOP(KA(f, 0)) != KO_EQ) {
			kthm t = neg_le(c->fact[i]);
			if (t)
				out[n++] = t;
		}
	}
	return n;
}

/* an int variable worth a case split */
static int
split_vars(kthm *f, int nf, kt *out, int max)
{
	kt sa[64], sc[64], at[16];
	int i, j, k, n = 0, ge, na, m;

	for (i = 0; i < nf; i++) {
		kt t = relterm(kthm_concl(f[i]), &ge);
		if (t == KNONE)
			continue;
		na = kn_split(t, sa, sc, 64);
		for (j = 0; j < na; j++) {
			kt mo = sa[j];
			m = 0;
			while (KOP(mo) == KO_MUL && m < 15) {
				at[m++] = KA(mo, 1);
				mo = KA(mo, 0);
			}
			at[m++] = mo;
			if (m < 2)
				continue;
			for (k = 0; k < m; k++) {
				int q, dup = 0;
				if (KOP(at[k]) != KO_VAR)
					continue;
				for (q = 0; q < n; q++)
					dup |= out[q] == at[k];
				if (!dup && n < max)
					out[n++] = at[k];
			}
		}
	}
	return n;
}

static kthm
refute(struct kctx *c, int depth)
{
	kthm f[FF], th;
	int nf, i, j, n0;

	if (ka_budget == 0)
		return KNONE;
	ka_budget--;
	for (i = 0; i < c->n; i++)
		if (KOP(kthm_concl(c->fact[i])) == KO_FALSE)
			return c->fact[i];
	nf = int_facts(c, f, FF / 2);
	if ((th = fm_proof(f, nf)) != KNONE)
		return th;
	if (depth <= 0)
		return KNONE;
	/* products of two inequalities */
	n0 = nf;
	for (i = 0; i < n0 && nf < FF; i++)
		for (j = i + 1; j < n0 && nf < FF; j++) {
			int g1, g2;
			kt a = relterm(kthm_concl(f[i]), &g1);
			kt b = relterm(kthm_concl(f[j]), &g2);
			kt x[2];
			kthm p;
			if (!g1 || !g2)
				continue;
			x[0] = a;
			x[1] = b;
			p = axiom_inst("mul_nonneg", x, 2);
			if (!p)
				continue;
			p = kr_impe(p, kr_andi(kr_conv(f[i], kf_le(kt_num(0), a)),
			    kr_conv(f[j], kf_le(kt_num(0), b))));
			if (p)
				f[nf++] = p;
		}
	if ((th = fm_proof(f, nf)) != KNONE)
		return th;
	/* split an integer variable: v <= 0 or 1 <= v */
	{
		kt vs[4];
		int nvs = split_vars(f, n0, vs, 4), k;
		for (k = 0; k < nvs; k++) {
			kthm disj = axiom_inst("int_discrete", &vs[k], 1), t1, t2;
			struct kctx *d;
			kt dj;
			if (!disj)
				continue;
			dj = kthm_concl(disj);
			d = xmalloc(sizeof *d);
			*d = *c;
			ka_add(d, kr_assume(KA(dj, 0)), 0);
			t1 = refute(d, depth - 1);
			*d = *c;
			ka_add(d, kr_assume(KA(dj, 1)), 0);
			t2 = t1 ? refute(d, depth - 1) : KNONE;
			free(d);
			if (t1 && t2)
				return kr_ore(disj, t1, t2);
		}
	}
	return KNONE;
}

static kthm
arith(struct kctx *c, kt g, int depth)
{
	kt gn = kn_norm(g);
	kthm th;

	if (gn == kf_true())
		return kr_conv(truth(), g);
	if (KOP(gn) == KO_EQ) {
		kt t = KA(gn, 0), x[1];
		kthm a, b, l;
		if ((th = eqsolve(c, g)) != KNONE)
			return th;
		a = arith(c, kf_le(kt_num(0), t), depth);
		if (!a)
			return KNONE;
		b = arith(c, kf_le(kt_num(0), kt_neg(t)), depth);
		if (!b)
			return KNONE;
		x[0] = t;
		l = axiom_inst("le_antisym", x, 1);
		if (!l)
			return KNONE;
		th = kr_impe(l, kr_andi(a, b));
		return th ? kr_conv(th, g) : KNONE;
	}
	if (KOP(gn) == KO_LE) {
		kt t = KA(gn, 1), x[1];
		kthm disj, c1, c2, r;
		struct kctx *d;
		kt dj;
		x[0] = kt_add(t, kt_num(1));
		disj = axiom_inst("int_discrete", x, 1);
		if (!disj)
			return KNONE;
		dj = kthm_concl(disj);
		d = xmalloc(sizeof *d);
		*d = *c;
		ka_add(d, kr_assume(KA(dj, 0)), 0);
		r = refute(d, depth);
		free(d);
		if (!r)
			return KNONE;
		c1 = kr_falsee(r, g);
		c2 = kr_conv(kr_assume(KA(dj, 1)), g);
		return c1 && c2 ? kr_ore(disj, c1, c2) : KNONE;
	}
	if (KOP(gn) == KO_FALSE)
		return refute(c, depth);
	return KNONE;
}

/* ---- subgroup membership ---- */

static kthm
lemma_th(const char *name)
{
	int i = kl_find(name);

	return i >= 0 ? klemmas[i].th : KNONE;
}

static kthm
membership(struct kctx *c, kt g, int depth)
{
	kt x = KA(g, 0), S = KA(g, 1), sa[32], sc[32], a[3];
	int n, i, sg = ksym_lookup("subgroup");
	kthm sub, acc = KNONE, s, lz, la, lm;

	lz = lemma_th("sub_zero");
	la = lemma_th("sub_add");
	lm = lemma_th("sub_smul");
	if (sg < 0 || !lz || !la || !lm || depth <= 0)
		return KNONE;
	n = kn_split(x, sa, sc, 32);
	if (n < 0 || (n == 1 && sc[0] == kt_num(1)))
		return KNONE;
	sub = prove(c, kf_pred(sg, &S, 1), depth - 1);
	if (!sub)
		return KNONE;
	if (n == 0) {
		s = kr_impe(kr_alle(lz, S), sub);
		return s ? kr_conv(s, g) : KNONE;
	}
	for (i = 0; i < n; i++) {
		kthm m = prove(c, kf_in(sa[i], S), depth - 1);
		if (!m)
			return KNONE;
		if (sc[i] != kt_num(1)) {
			a[0] = S;
			a[1] = sc[i];
			a[2] = sa[i];
			m = kr_impe(kr_alle(kr_alle(kr_alle(lm, a[0]), a[1]), a[2]),
			    kr_andi(sub, m));
			if (!m)
				return KNONE;
		}
		if (!acc)
			acc = m;
		else {
			kt l = KA(kthm_concl(acc), 0), r = KA(kthm_concl(m), 0);
			acc = kr_impe(kr_alle(kr_alle(kr_alle(la, S), l), r),
			    kr_andi(kr_andi(sub, acc), m));
			if (!acc)
				return KNONE;
		}
	}
	return kr_conv(acc, g);
}

/* ---- congruence: P(s...) and s = t give P(t...) ---- */

static kthm
congruence(struct kctx *c, kt g)
{
	int i, k;

	for (i = c->n - 1; i >= 0; i--) {
		kt f = kthm_concl(c->fact[i]);
		kthm th = c->fact[i];
		int ok = 1;
		if (KOP(f) != KOP(g) || KT.aux[f] != KT.aux[g] || KN(f) != KN(g))
			continue;
		for (k = 0; k < KN(g) && ok; k++) {
			kt s = KA(f, k), t = KA(g, k), pat, x[16];
			kthm eq;
			int q;
			if (s == t || kn_norm(kf_eq(s, t)) == kf_true())
				continue;
			if (!isrelsort(s)) {
				ok = 0;
				break;
			}
			eq = eqsolve(c, kf_eq(s, t));
			if (!eq) {
				ok = 0;
				break;
			}
			f = kthm_concl(th);
			for (q = 0; q < KN(f); q++)
				x[q] = q == k ? kt_bvar(0, KSORT(s)) : KA(f, q);
			pat = kt_mk(KOP(f), KS_PROP, KT.aux[f], 0, x, KN(f));
			th = kr_subst(eq, pat, th);
			if (!th)
				ok = 0;
		}
		if (ok && th)
			return kr_conv(th, g);
	}
	return KNONE;
}

/* ---- chaining through rules ---- */

enum { P_ALLE, P_IMPE, P_IFF1, P_IFF2, P_AND1, P_AND2 };
struct path {
	uint8_t op[64];
	kt arg[64];	/* P_ALLE: the meta; P_IMPE: the premise */
	int n;
};

struct chase {
	struct kctx *c;
	kt goal;
	int depth;
	kthm rule;
	kthm result;
};

static kthm
build(kthm th, const struct path *p, const kthm *prem)
{
	int i, k = 0;

	for (i = 0; i < p->n && th; i++)
		switch (p->op[i]) {
		case P_ALLE: {
			kt v = resolve(p->arg[i]);
			if (kt_hasmeta(v))
				return KNONE;
			th = kr_alle(th, v);
			break;
		}
		case P_IMPE: {
			kt a = KA(kthm_concl(th), 0);
			kthm q = kr_conv(prem[k++], a);
			th = q ? kr_impe(th, q) : KNONE;
			break;
		}
		case P_IFF1:
			th = kr_iffe1(th);
			break;
		case P_IFF2:
			th = kr_iffe2(th);
			break;
		case P_AND1:
			th = kr_ande1(th);
			break;
		case P_AND2:
			th = kr_ande2(th);
			break;
		}
	return th;
}

/* the conjunct atoms of a premise */
static int
conjuncts(kt f, kt *out, int n, int max)
{
	if (KOP(f) == KO_AND) {
		n = conjuncts(KA(f, 0), out, n, max);
		return conjuncts(KA(f, 1), out, n, max);
	}
	if (n < max)
		out[n++] = f;
	return n;
}

/* bind the metas left in the premises by matching facts, then prove */
static int
premises(struct chase *ch, const struct path *p, kt *atoms, int na, int ai)
{
	struct kctx *c = ch->c;
	int i;

	if (ka_budget == 0)
		return 0;
	while (ai < na && !kt_hasmeta(resolve(atoms[ai])))
		ai++;
	if (ai < na) {
		kt a = kn_norm(resolve(atoms[ai]));
		int tries = 0;
		for (i = c->n - 1; i >= 0 && tries < 24; i--) {
			int mark = ntrail;
			kt f = kn_norm(kthm_concl(c->fact[i]));
			if (KOP(f) != KOP(a))
				continue;
			ka_budget = ka_budget ? ka_budget - 1 : 0;
			if (uni(a, f)) {
				tries++;
				if (premises(ch, p, atoms, na, ai + 1))
					return 1;
			}
			undo(mark);
		}
		return 0;
	}
	/* all bound: prove the premises */
	{
		kthm prem[64];
		int k = 0;
		for (i = 0; i < p->n; i++)
			if (p->op[i] == P_IMPE) {
				kt a = resolve(p->arg[i]);
				if (kt_hasmeta(a))
					return 0;
				prem[k] = prove(c, a, ch->depth - 1);
				if (!prem[k])
					return 0;
				k++;
			} else if (p->op[i] == P_ALLE &&
			    kt_hasmeta(resolve(p->arg[i])))
				return 0;
		ch->result = build(ch->rule, p, prem);
		if (ch->result)
			ch->result = kr_conv(ch->result, ch->goal);
		return ch->result != KNONE;
	}
}

static int
walk(struct chase *ch, kt f, struct path *p)
{
	int mark = ntrail, n0 = p->n;

	if (p->n >= 60 || ka_budget == 0)
		return 0;
	/* this level as the conclusion */
	if (KOP(f) == KOP(kn_norm(ch->goal)) || KOP(f) == KOP(ch->goal) ||
	    KOP(f) == KO_EQ || KOP(f) == KO_LE || KOP(f) == KO_LT) {
		ka_budget = ka_budget ? ka_budget - 1 : 0;
		if (uni(f, kn_norm(ch->goal))) {
			kt atoms[64];
			int na = 0, i;
			for (i = 0; i < p->n; i++)
				if (p->op[i] == P_IMPE)
					na = conjuncts(p->arg[i], atoms, na, 64);
			if (premises(ch, p, atoms, na, 0))
				return 1;
		}
		undo(mark);
	}
	switch (KOP(f)) {
	case KO_ALL: {
		kt m = newmeta((int)KT.aux[f]);
		if (!m)
			return 0;
		p->op[p->n] = P_ALLE;
		p->arg[p->n++] = m;
		if (walk(ch, kt_inst(KA(f, 0), m), p))
			return 1;
		break;
	}
	case KO_IMP:
		p->op[p->n] = P_IMPE;
		p->arg[p->n++] = KA(f, 0);
		if (walk(ch, KA(f, 1), p))
			return 1;
		break;
	case KO_IFF:
		p->op[p->n] = P_IFF1;
		p->n++;
		if (walk(ch, kf_imp(KA(f, 0), KA(f, 1)), p))
			return 1;
		p->n = n0;
		undo(mark);
		p->op[p->n] = P_IFF2;
		p->n++;
		if (walk(ch, kf_imp(KA(f, 1), KA(f, 0)), p))
			return 1;
		break;
	case KO_AND:
		p->op[p->n] = P_AND1;
		p->n++;
		if (walk(ch, KA(f, 0), p))
			return 1;
		p->n = n0;
		undo(mark);
		p->op[p->n] = P_AND2;
		p->n++;
		if (walk(ch, KA(f, 1), p))
			return 1;
		break;
	}
	p->n = n0;
	undo(mark);
	return 0;
}

static kthm
chain(struct kctx *c, kt g, int depth, int rulesonly)
{
	struct chase ch;
	struct path p;
	int i, k;
	uint32_t m0 = nmeta;

	if (depth <= 0)
		return KNONE;
	ch.c = c;
	ch.goal = g;
	ch.depth = depth;
	for (k = 0; k < c->nref + c->n; k++) {
		kthm r = k < c->nref ? c->ref[k] : c->fact[c->n - 1 -
		    (k - c->nref)];
		kt f = kthm_concl(r);
		if (rulesonly && KOP(f) != KO_ALL && KOP(f) != KO_IMP &&
		    KOP(f) != KO_IFF && KOP(f) != KO_AND)
			continue;
		if (KOP(f) != KO_ALL && KOP(f) != KO_IMP && KOP(f) != KO_IFF)
			continue;
		ch.rule = r;
		ch.result = KNONE;
		p.n = 0;
		i = ntrail;
		if (walk(&ch, kn_norm(f), &p) && ch.result) {
			undo(i);
			nmeta = m0;
			return ch.result;
		}
		undo(i);
		nmeta = m0;
		if (ka_budget == 0)
			break;
	}
	return KNONE;
}

/* ---- existentials: witnesses ---- */

static kthm
exists_goal(struct kctx *c, kt g, int depth)
{
	kt body = KA(g, 0), cand[16], m, inst, atoms[32];
	int nc = 0, i, j, k, na, s = (int)KT.aux[g];
	uint32_t m0 = nmeta;
	kthm th;

	if (c->nwith > 0) {
		kt w = c->with[0];
		struct kctx *d;
		if (s == KS_EL && KOP(w) == KO_NUM && KT.num[w] == 0)
			w = kt_zero();
		d = xmalloc(sizeof *d);
		*d = *c;
		memmove(d->with, d->with + 1, (size_t)(d->nwith - 1) *
		    sizeof *d->with);
		d->nwith--;
		th = KSORT(w) == s ? prove(d, kt_inst(body, w), depth) : KNONE;
		free(d);
		return th ? kr_exi(g, w, th) : KNONE;
	}
	m = newmeta(s);
	if (!m)
		return KNONE;
	inst = kn_norm(kt_inst(body, m));
	na = conjuncts(inst, atoms, 0, 32);
	for (i = 0; i < na && nc < 16; i++) {
		if (!kt_hasmeta(atoms[i]))
			continue;
		for (j = c->n - 1; j >= 0 && nc < 16; j--) {
			int mark = ntrail;
			kt f = kn_norm(kthm_concl(c->fact[j]));
			if (KOP(f) != KOP(atoms[i]))
				continue;
			if (uni(atoms[i], f)) {
				kt v = resolve(m);
				if (v != m && !kt_hasmeta(v)) {
					for (k = 0; k < nc && cand[k] != v; k++)
						;
					if (k == nc)
						cand[nc++] = v;
				}
			}
			undo(mark);
		}
	}
	nmeta = m0;
	if (nc < 16)
		cand[nc++] = s == KS_EL ? kt_zero() : s == KS_INT ?
		    kt_num(0) : KNONE;
	for (i = 0; i < nc; i++) {
		if (!cand[i])
			continue;
		th = prove(c, kt_inst(body, cand[i]), depth);
		if (th)
			return kr_exi(g, cand[i], th);
	}
	return KNONE;
}

/* ---- the prover ---- */

static kthm
atom_goal(struct kctx *c, kt g, int depth)
{
	kt gn = kn_norm(g);
	kthm th;

	if (gn == kf_true())
		return kr_conv(truth(), g);
	if (KOP(gn) == KO_FALSE)
		return refute(c, depth > 1 ? 2 : 1);
	if ((KOP(gn) == KO_EQ || KOP(gn) == KO_LE) &&
	    KSORT(KA(gn, KOP(gn) == KO_LE)) == KS_INT)
		if ((th = arith(c, g, depth > 1 ? 2 : 1)) != KNONE)
			return th;
	if (KOP(gn) == KO_EQ && KSORT(KA(gn, 0)) == KS_EL)
		if ((th = eqsolve(c, g)) != KNONE)
			return th;
	if (KOP(gn) == KO_IN || KOP(gn) == KO_PRED)
		if ((th = congruence(c, gn)) != KNONE)
			return kr_conv(th, g);
	if (KOP(gn) == KO_IN)
		if ((th = membership(c, gn, depth)) != KNONE)
			return kr_conv(th, g);
	if ((th = chain(c, g, depth, 1)) != KNONE)
		return th;
	return KNONE;
}

static kthm
prove(struct kctx *c, kt g, int depth)
{
	kthm th = KNONE, a, b;
	int n0 = c->n;
	struct opened o;

	if (ka_budget == 0 || g == KNONE)
		return KNONE;
	ka_budget--;
	if ((th = ka_close(c, g)) != KNONE)
		return th;
	o.n = 0;
	if (KOP(g) == KO_ALL || KOP(g) == KO_IMP || KOP(g) == KO_IFF ||
	    KOP(g) == KO_AND || KOP(g) == KO_NOT)
		if ((th = chain(c, g, depth, 1)) != KNONE)
			return th;
	switch (KOP(g)) {
	case KO_AND:
		a = prove(c, KA(g, 0), depth);
		b = a ? prove(c, KA(g, 1), depth) : KNONE;
		return a && b ? kr_andi(a, b) : KNONE;
	case KO_IMP:
		ka_add(c, kr_assume(KA(g, 0)), 0);
		open_ex(c, n0, &o);
		th = prove(c, KA(g, 1), depth);
		th = th ? discharge(th, &o) : KNONE;
		c->n = n0;
		return th ? kr_impi(th, KA(g, 0)) : KNONE;
	case KO_IFF:
		a = prove(c, kf_imp(KA(g, 0), KA(g, 1)), depth);
		b = a ? prove(c, kf_imp(KA(g, 1), KA(g, 0)), depth) : KNONE;
		return a && b ? kr_iffi(a, b) : KNONE;
	case KO_NOT:
		ka_add(c, kr_assume(KA(g, 0)), 0);
		open_ex(c, n0, &o);
		th = prove(c, kf_false(), depth);
		th = th ? discharge(th, &o) : KNONE;
		c->n = n0;
		return th ? kr_noti(th, KA(g, 0)) : KNONE;
	case KO_ALL: {
		kt v = kt_fresh(KT.hint[g] ? sym_str(KT.hint[g]) : "x",
		    (int)KT.aux[g]);
		th = prove(c, kt_inst(KA(g, 0), v), depth);
		return th ? kr_alli(th, v) : KNONE;
	}
	case KO_EX:
		if ((th = chain(c, g, depth, 1)) != KNONE)
			return th;
		return exists_goal(c, g, depth);
	case KO_OR:
		if ((th = prove(c, KA(g, 0), depth)) != KNONE)
			return kr_ori1(th, KA(g, 1));
		if ((th = prove(c, KA(g, 1), depth)) != KNONE)
			return kr_ori2(KA(g, 0), th);
		return chain(c, g, depth, 1);
	case KO_FALSE:
		if ((th = refute(c, depth > 1 ? 2 : 1)) != KNONE)
			return th;
		if ((th = chain(c, g, depth, 1)) != KNONE)
			return th;
		/* a negated fact whose positive form we can prove */
		{
			int i;
			for (i = c->n - 1; i >= 0; i--) {
				kt f = kthm_concl(c->fact[i]);
				if (KOP(f) == KO_NOT && depth > 0) {
					kthm p = prove(c, KA(f, 0), depth - 1);
					if (p)
						return kr_note(c->fact[i], p);
				}
			}
		}
		return KNONE;
	default:
		return atom_goal(c, g, depth);
	}
}

/* modus ponens on ground implications whose premise is at hand */
static void
saturate(struct kctx *c)
{
	int i, round, n;

	for (round = 0; round < 2; round++) {
		n = c->n;
		for (i = 0; i < n; i++) {
			kt f = kthm_concl(c->fact[i]), a;
			kthm p;
			if (KOP(f) != KO_IMP || KT.lb[f])
				continue;
			a = KA(f, 0);
			p = ka_close(c, a);
			if (!p && (KOP(a) == KO_LE || KOP(a) == KO_LT ||
			    (KOP(a) == KO_EQ && KSORT(KA(a, 0)) == KS_INT)))
				p = arith(c, a, 1);
			if (p && (p = kr_conv(p, a)) != KNONE) {
				kthm q = kr_impe(c->fact[i], p);
				int k, dup = 0;
				for (k = 0; k < c->n; k++)
					dup |= kthm_concl(c->fact[k]) == kthm_concl(q);
				if (q && !dup)
					ka_add(c, q, 0);
			}
		}
	}
}

kthm
ka_prove(struct kctx *c, kt goal)
{
	kthm th;
	struct opened o;
	int n0 = c->n;
	uint64_t b0 = ka_budget;

	if (!ka_budget)
		ka_budget = b0 = 40000;
	nmeta = 0;
	ntrail = 0;
	o.n = 0;
	saturate(c);
	open_ex(c, 0, &o);
	if (!c->depth)
		c->depth = 3;
	th = prove(c, goal, c->depth);
	if (th)
		th = discharge(th, &o);
	c->n = n0;
	if (!th)
		snprintf(ka_why, sizeof ka_why, "%s%s", ka_budget == 0 ?
		    "gave up (search budget) on " : "could not prove ",
		    kt_str(goal));
	ka_budget = 0;
	(void)b0;
	return th;
}
