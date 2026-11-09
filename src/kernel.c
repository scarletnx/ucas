/*
 * kernel.c - the LCF kernel.  A theorem is a sequent (hypotheses |-
 * conclusion) in this file's SoA table, and only the rules below append
 * to it.  Each records its rule, premises and term arguments: the proof
 * DAG that kcheck.c replays independently and prove_cmd.c prints.
 *
 * The rules are natural deduction for many-sorted classical first-order
 * logic with equality, plus: conversion modulo the normal forms of
 * knorm.c (ring/module/homomorphism identities, numerals), linear
 * combination of equalities and inequalities (Farkas certificates),
 * induction over the integers from a base, and declaring a symbol a
 * homomorphism once its additivity and Z-linearity are theorems.
 */
#include "ucas.h"
#include "prove.h"

#include <stdlib.h>
#include <string.h>

const char *const kr_name[KR_NRULES] = {
	"assume", "axiom", "imp_intro", "imp_elim", "and_intro", "and_elim1",
	"and_elim2", "or_intro1", "or_intro2", "or_elim", "false_elim",
	"not_intro", "not_elim", "raa", "iff_intro", "iff_elim1", "iff_elim2",
	"all_intro", "all_elim", "ex_intro", "ex_elim", "refl", "subst",
	"conv", "lin", "induct", "hom_decl"
};

static struct {
	kt *concl;
	uint8_t *rule;
	uint32_t *hoff, *hn, *poff, *pn, *aoff, *an, *aux;
	uint32_t len, cap;
	kt *hyps;
	uint32_t nh, caph;
	kthm *prems;
	uint32_t np, capp;
	kt *args;
	uint32_t na, capa;
} T;

struct kaxiom *kaxioms;
uint32_t nkaxioms;
static uint32_t capax;
static char errbuf[512];

const char *
kerr(void)
{
	return errbuf;
}

static kthm
bad(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(errbuf, sizeof errbuf, fmt, ap);
	va_end(ap);
	return KNONE;
}

kt kthm_concl(kthm th) { return T.concl[th]; }
uint32_t kthm_nhyps(kthm th) { return T.hn[th]; }
kt kthm_hyp(kthm th, uint32_t i) { return T.hyps[T.hoff[th] + i]; }
int kthm_rule(kthm th) { return T.rule[th]; }
uint32_t kthm_nprem(kthm th) { return T.pn[th]; }
kthm kthm_prem(kthm th, uint32_t i) { return T.prems[T.poff[th] + i]; }
uint32_t kthm_nargs(kthm th) { return T.an[th]; }
kt kthm_arg(kthm th, uint32_t i) { return T.args[T.aoff[th] + i]; }
uint32_t kthm_count(void) { return T.len; }
uint32_t kthm_aux(kthm th) { return T.aux[th]; }

static void
reserve(void)
{
	if (T.len == 0)
		T.len = 1;
	if (T.len + 1 >= T.cap) {
		T.cap = T.cap ? T.cap * 2 : 4096;
		T.concl = xrealloc(T.concl, T.cap * sizeof *T.concl);
		T.rule = xrealloc(T.rule, T.cap);
		T.hoff = xrealloc(T.hoff, T.cap * sizeof *T.hoff);
		T.hn = xrealloc(T.hn, T.cap * sizeof *T.hn);
		T.poff = xrealloc(T.poff, T.cap * sizeof *T.poff);
		T.pn = xrealloc(T.pn, T.cap * sizeof *T.pn);
		T.aoff = xrealloc(T.aoff, T.cap * sizeof *T.aoff);
		T.an = xrealloc(T.an, T.cap * sizeof *T.an);
		T.aux = xrealloc(T.aux, T.cap * sizeof *T.aux);
	}
}

static void
hpush(kt h)
{
	if (T.nh == T.caph) {
		T.caph = T.caph ? T.caph * 2 : 65536;
		T.hyps = xrealloc(T.hyps, T.caph * sizeof *T.hyps);
	}
	T.hyps[T.nh++] = h;
}

/* scratch hypothesis set, sorted */
static kt hs[4096];
static uint32_t nhs;

static void
hs_add(kt h)
{
	uint32_t i, j;

	for (i = 0; i < nhs && hs[i] < h; i++)
		;
	if (i < nhs && hs[i] == h)
		return;
	if (nhs == 4096)
		fail("too many hypotheses in a proof");
	for (j = nhs; j > i; j--)
		hs[j] = hs[j - 1];
	hs[i] = h;
	nhs++;
}

static void
hs_from(kthm th, kt except)
{
	uint32_t i;

	for (i = 0; i < T.hn[th]; i++)
		if (T.hyps[T.hoff[th] + i] != except)
			hs_add(T.hyps[T.hoff[th] + i]);
}

static kthm
emit(int rule, kt concl, const kthm *p, int np, const kt *a, int na,
    uint32_t aux)
{
	kthm th;
	int i;

	reserve();
	th = T.len++;
	T.concl[th] = concl;
	T.rule[th] = (uint8_t)rule;
	T.aux[th] = aux;
	T.hoff[th] = T.nh;
	T.hn[th] = nhs;
	for (i = 0; i < (int)nhs; i++)
		hpush(hs[i]);
	nhs = 0;
	if (T.np + (uint32_t)np >= T.capp) {
		T.capp = newcap(T.capp, T.np + (uint32_t)np + 65536);
		T.prems = xrealloc(T.prems, T.capp * sizeof *T.prems);
	}
	T.poff[th] = T.np;
	T.pn[th] = (uint32_t)np;
	for (i = 0; i < np; i++)
		T.prems[T.np++] = p[i];
	if (T.na + (uint32_t)na >= T.capa) {
		T.capa = newcap(T.capa, T.na + (uint32_t)na + 65536);
		T.args = xrealloc(T.args, T.capa * sizeof *T.args);
	}
	T.aoff[th] = T.na;
	T.an[th] = (uint32_t)na;
	for (i = 0; i < na; i++)
		T.args[T.na++] = a[i];
	return th;
}

static int
okthm(kthm th)
{
	return th > 0 && th < T.len;
}

/* a closed formula without metavariables */
static int
wff(kt f)
{
	return f != KNONE && KSORT(f) == KS_PROP && KT.lb[f] == 0 &&
	    !kt_hasmeta(f);
}

static int
hyps_have(kthm th, kt v)
{
	uint32_t i;

	for (i = 0; i < T.hn[th]; i++)
		if (kt_occurs(v, T.hyps[T.hoff[th] + i]))
			return 1;
	return 0;
}

kthm
kr_assume(kt phi)
{
	if (!wff(phi))
		return bad("assume: not a closed formula");
	nhs = 0;
	hs_add(phi);
	return emit(KR_ASSUME, phi, NULL, 0, &phi, 1, 0);
}

int
kax_find(const char *name)
{
	uint32_t id = sym_cstr(name), i;

	for (i = 0; i < nkaxioms; i++)
		if (kaxioms[i].name == id)
			return (int)i;
	return -1;
}

int
kax_add(const char *name, kt phi, const char *kind, const char *text)
{
	if (!wff(phi) || kax_find(name) >= 0)
		return -1;
	if (nkaxioms == capax) {
		capax = capax ? capax * 2 : 64;
		kaxioms = xrealloc(kaxioms, capax * sizeof *kaxioms);
	}
	kaxioms[nkaxioms].name = sym_cstr(name);
	kaxioms[nkaxioms].phi = phi;
	kaxioms[nkaxioms].kind = kind;
	kaxioms[nkaxioms].text = text;
	return (int)nkaxioms++;
}

kthm
kr_axiom(const char *name)
{
	int i = kax_find(name);

	if (i < 0)
		return bad("no axiom %s", name);
	nhs = 0;
	return emit(KR_AXIOM, kaxioms[i].phi, NULL, 0, NULL, 0, (uint32_t)i);
}

kthm
kr_impi(kthm th, kt phi)
{
	kt c;

	if (!okthm(th) || !wff(phi))
		return bad("imp_intro: bad arguments");
	c = kf_imp(phi, T.concl[th]);
	nhs = 0;
	hs_from(th, phi);
	return emit(KR_IMPI, c, &th, 1, &phi, 1, 0);
}

kthm
kr_impe(kthm imp, kthm a)
{
	kt c, p[2];

	if (!okthm(imp) || !okthm(a))
		return bad("imp_elim: bad arguments");
	c = T.concl[imp];
	if (KOP(c) != KO_IMP || KA(c, 0) != T.concl[a])
		return bad("imp_elim: %s does not match the premise of %s",
		    kt_str(T.concl[a]), kt_str(c));
	nhs = 0;
	hs_from(imp, KNONE);
	hs_from(a, KNONE);
	p[0] = imp;
	p[1] = a;
	return emit(KR_IMPE, KA(c, 1), p, 2, NULL, 0, 0);
}

kthm
kr_andi(kthm a, kthm b)
{
	kthm p[2];

	if (!okthm(a) || !okthm(b))
		return bad("and_intro: bad arguments");
	nhs = 0;
	hs_from(a, KNONE);
	hs_from(b, KNONE);
	p[0] = a;
	p[1] = b;
	return emit(KR_ANDI, kf_and(T.concl[a], T.concl[b]), p, 2, NULL, 0, 0);
}

static kthm
ande(kthm th, int k)
{
	kt c;

	if (!okthm(th) || KOP(c = T.concl[th]) != KO_AND)
		return bad("and_elim: not a conjunction");
	nhs = 0;
	hs_from(th, KNONE);
	return emit(k ? KR_ANDE2 : KR_ANDE1, KA(c, k), &th, 1, NULL, 0, 0);
}

kthm kr_ande1(kthm th) { return ande(th, 0); }
kthm kr_ande2(kthm th) { return ande(th, 1); }

kthm
kr_ori1(kthm th, kt psi)
{
	if (!okthm(th) || !wff(psi))
		return bad("or_intro: bad arguments");
	nhs = 0;
	hs_from(th, KNONE);
	return emit(KR_ORI1, kf_or(T.concl[th], psi), &th, 1, &psi, 1, 0);
}

kthm
kr_ori2(kt phi, kthm th)
{
	if (!okthm(th) || !wff(phi))
		return bad("or_intro: bad arguments");
	nhs = 0;
	hs_from(th, KNONE);
	return emit(KR_ORI2, kf_or(phi, T.concl[th]), &th, 1, &phi, 1, 0);
}

kthm
kr_ore(kthm or, kthm c1, kthm c2)
{
	kt c;
	kthm p[3];

	if (!okthm(or) || !okthm(c1) || !okthm(c2) ||
	    KOP(c = T.concl[or]) != KO_OR)
		return bad("or_elim: not a disjunction");
	if (T.concl[c1] != T.concl[c2])
		return bad("or_elim: the cases prove different things");
	nhs = 0;
	hs_from(or, KNONE);
	hs_from(c1, KA(c, 0));
	hs_from(c2, KA(c, 1));
	p[0] = or;
	p[1] = c1;
	p[2] = c2;
	return emit(KR_ORE, T.concl[c1], p, 3, NULL, 0, 0);
}

kthm
kr_falsee(kthm th, kt phi)
{
	if (!okthm(th) || KOP(T.concl[th]) != KO_FALSE || !wff(phi))
		return bad("false_elim: not false");
	nhs = 0;
	hs_from(th, KNONE);
	return emit(KR_FALSEE, phi, &th, 1, &phi, 1, 0);
}

kthm
kr_noti(kthm th, kt phi)
{
	if (!okthm(th) || KOP(T.concl[th]) != KO_FALSE || !wff(phi))
		return bad("not_intro: not false");
	nhs = 0;
	hs_from(th, phi);
	return emit(KR_NOTI, kf_not(phi), &th, 1, &phi, 1, 0);
}

kthm
kr_note(kthm notphi, kthm phi)
{
	kthm p[2];

	if (!okthm(notphi) || !okthm(phi) || KOP(T.concl[notphi]) != KO_NOT ||
	    KA(T.concl[notphi], 0) != T.concl[phi])
		return bad("not_elim: %s and %s do not contradict",
		    kt_str(T.concl[notphi]), kt_str(T.concl[phi]));
	nhs = 0;
	hs_from(notphi, KNONE);
	hs_from(phi, KNONE);
	p[0] = notphi;
	p[1] = phi;
	return emit(KR_NOTE, kf_false(), p, 2, NULL, 0, 0);
}

kthm
kr_raa(kthm th, kt phi)
{
	if (!okthm(th) || KOP(T.concl[th]) != KO_FALSE || !wff(phi))
		return bad("raa: not false");
	nhs = 0;
	hs_from(th, kf_not(phi));
	return emit(KR_RAA, phi, &th, 1, &phi, 1, 0);
}

kthm
kr_iffi(kthm ab, kthm ba)
{
	kt x, y;
	kthm p[2];

	if (!okthm(ab) || !okthm(ba))
		return bad("iff_intro: bad arguments");
	x = T.concl[ab];
	y = T.concl[ba];
	if (KOP(x) != KO_IMP || KOP(y) != KO_IMP || KA(x, 0) != KA(y, 1) ||
	    KA(x, 1) != KA(y, 0))
		return bad("iff_intro: not converse implications");
	nhs = 0;
	hs_from(ab, KNONE);
	hs_from(ba, KNONE);
	p[0] = ab;
	p[1] = ba;
	return emit(KR_IFFI, kf_iff(KA(x, 0), KA(x, 1)), p, 2, NULL, 0, 0);
}

static kthm
iffe(kthm th, int k)
{
	kt c;

	if (!okthm(th) || KOP(c = T.concl[th]) != KO_IFF)
		return bad("iff_elim: not an equivalence");
	nhs = 0;
	hs_from(th, KNONE);
	return emit(k ? KR_IFFE2 : KR_IFFE1, k ? kf_imp(KA(c, 1), KA(c, 0)) :
	    kf_imp(KA(c, 0), KA(c, 1)), &th, 1, NULL, 0, 0);
}

kthm kr_iffe1(kthm th) { return iffe(th, 0); }
kthm kr_iffe2(kthm th) { return iffe(th, 1); }

kthm
kr_alli(kthm th, kt v)
{
	kt c;

	if (!okthm(th) || KOP(v) != KO_VAR)
		return bad("all_intro: not a variable");
	if (hyps_have(th, v))
		return bad("all_intro: %s occurs in a hypothesis", kt_str(v));
	c = kf_bind(KO_ALL, KSORT(v), KT.aux[v], kt_abstract(T.concl[th], v));
	nhs = 0;
	hs_from(th, KNONE);
	return emit(KR_ALLI, c, &th, 1, &v, 1, 0);
}

kthm
kr_alle(kthm th, kt t)
{
	kt c;

	if (!okthm(th) || KOP(c = T.concl[th]) != KO_ALL)
		return bad("all_elim: not a universal statement");
	if (KSORT(t) != (int)KT.aux[c] || KT.lb[t] || kt_hasmeta(t) ||
	    KSORT(t) == KS_PROP)
		return bad("all_elim: %s is not a closed %s", kt_str(t),
		    ks_name((int)KT.aux[c]));
	nhs = 0;
	hs_from(th, KNONE);
	return emit(KR_ALLE, kt_inst(KA(c, 0), t), &th, 1, &t, 1, 0);
}

kthm
kr_exi(kt ex, kt t, kthm th)
{
	kt a[2];

	if (!okthm(th) || !wff(ex) || KOP(ex) != KO_EX)
		return bad("ex_intro: not an existential");
	if (KSORT(t) != (int)KT.aux[ex] || KT.lb[t] || kt_hasmeta(t))
		return bad("ex_intro: bad witness");
	if (kt_inst(KA(ex, 0), t) != T.concl[th])
		return bad("ex_intro: %s is not an instance of %s",
		    kt_str(T.concl[th]), kt_str(ex));
	nhs = 0;
	hs_from(th, KNONE);
	a[0] = ex;
	a[1] = t;
	return emit(KR_EXI, ex, &th, 1, a, 2, 0);
}

kthm
kr_exe(kthm ex, kt v, kthm th)
{
	kt c, inst;
	kthm p[2];
	uint32_t i;

	if (!okthm(ex) || !okthm(th) || KOP(c = T.concl[ex]) != KO_EX ||
	    KOP(v) != KO_VAR || KSORT(v) != (int)KT.aux[c])
		return bad("ex_elim: bad arguments");
	inst = kt_inst(KA(c, 0), v);
	if (kt_occurs(v, c) || kt_occurs(v, T.concl[th]) || hyps_have(ex, v))
		return bad("ex_elim: %s is not fresh", kt_str(v));
	for (i = 0; i < T.hn[th]; i++)
		if (T.hyps[T.hoff[th] + i] != inst &&
		    kt_occurs(v, T.hyps[T.hoff[th] + i]))
			return bad("ex_elim: %s occurs in another hypothesis",
			    kt_str(v));
	nhs = 0;
	hs_from(ex, KNONE);
	hs_from(th, inst);
	p[0] = ex;
	p[1] = th;
	return emit(KR_EXE, T.concl[th], p, 2, &v, 1, 0);
}

kthm
kr_refl(kt t)
{
	if (t == KNONE || KSORT(t) == KS_PROP || KT.lb[t] || kt_hasmeta(t))
		return bad("refl: not a closed term");
	nhs = 0;
	return emit(KR_REFL, kf_eq(t, t), NULL, 0, &t, 1, 0);
}

kthm
kr_subst(kthm eq, kt pat, kthm th)
{
	kt c, a[1];
	kthm p[2];

	if (!okthm(eq) || !okthm(th) || KOP(c = T.concl[eq]) != KO_EQ)
		return bad("subst: not an equation");
	if (KT.lb[pat] > 1 || kt_hasmeta(pat) || KSORT(pat) != KS_PROP)
		return bad("subst: bad pattern");
	if (kt_inst(pat, KA(c, 0)) != T.concl[th])
		return bad("subst: %s is not the pattern at %s",
		    kt_str(T.concl[th]), kt_str(KA(c, 0)));
	nhs = 0;
	hs_from(eq, KNONE);
	hs_from(th, KNONE);
	p[0] = eq;
	p[1] = th;
	a[0] = pat;
	return emit(KR_SUBST, kt_inst(pat, KA(c, 1)), p, 2, a, 1, 0);
}

kthm
kr_conv(kthm th, kt phi)
{
	if (!okthm(th) || !wff(phi))
		return bad("conv: bad arguments");
	if (T.concl[th] == phi)
		return th;
	if (!kn_equiv(T.concl[th], phi))
		return bad("conv: %s and %s differ", kt_str(T.concl[th]),
		    kt_str(phi));
	nhs = 0;
	hs_from(th, KNONE);
	return emit(KR_CONV, phi, &th, 1, &phi, 1, 0);
}

/* the term that a relation says is zero (EQ) or nonnegative (LE, LT) */
static kt
reltm(kt c, int *ge)
{
	switch (KOP(c)) {
	case KO_EQ:
		if (KSORT(KA(c, 0)) != KS_INT && KSORT(KA(c, 0)) != KS_EL)
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

kthm
kr_lin(const kthm *ths, const kt *coef, int n)
{
	kt sum = KNONE, t, c;
	int i, ge, anyge = 0, sort = -1;
	int64_t v;

	if (n < 1 || n > 64)
		return bad("lin: 1 to 64 facts");
	for (i = 0; i < n; i++) {
		if (!okthm(ths[i]))
			return bad("lin: bad theorem");
		c = T.concl[ths[i]];
		if ((t = reltm(c, &ge)) == KNONE)
			return bad("lin: %s is not a relation", kt_str(c));
		if (sort < 0)
			sort = KSORT(t);
		if (KSORT(t) != sort || KSORT(coef[i]) != KS_INT ||
		    KT.lb[coef[i]] || kt_hasmeta(coef[i]))
			return bad("lin: sorts differ");
		if (ge) {
			if (!kn_poly_const(coef[i], &v) || v < 0)
				return bad("lin: an inequality needs a "
				    "nonnegative numeral factor");
			anyge = 1;
		}
		t = kt_mul(coef[i], t);
		sum = sum == KNONE ? t : kt_add(sum, t);
	}
	nhs = 0;
	for (i = 0; i < n; i++)
		hs_from(ths[i], KNONE);
	c = anyge ? kf_le(kt_num(0), sum) : kf_eq(sum, sort == KS_EL ?
	    kt_zero() : kt_num(0));
	{
		kt a[128];
		for (i = 0; i < n; i++)
			a[i] = coef[i];
		return emit(KR_LIN, c, ths, n, a, n, 0);
	}
}

kthm
kr_induct(kt base, kt pat, kthm b, kthm step)
{
	kt m = kt_bvar(0, KS_INT), want, a[2];
	kthm p[2];

	if (!okthm(b) || !okthm(step) || KSORT(base) != KS_INT ||
	    KT.lb[base] || kt_hasmeta(base) || KT.lb[pat] > 1 ||
	    kt_hasmeta(pat) || KSORT(pat) != KS_PROP)
		return bad("induct: bad arguments");
	if (T.concl[b] != kt_inst(pat, base))
		return bad("induct: the base case is %s, not %s",
		    kt_str(T.concl[b]), kt_str(kt_inst(pat, base)));
	want = kf_bind(KO_ALL, KS_INT, sym_cstr("m"), kf_imp(kf_and(kf_le(base,
	    m), pat), kt_inst(pat, kt_add(m, kt_num(1)))));
	if (T.concl[step] != want)
		return bad("induct: the step is %s, not %s",
		    kt_str(T.concl[step]), kt_str(want));
	nhs = 0;
	hs_from(b, KNONE);
	hs_from(step, KNONE);
	p[0] = b;
	p[1] = step;
	a[0] = base;
	a[1] = pat;
	return emit(KR_INDUCT, kf_bind(KO_ALL, KS_INT, sym_cstr("m"),
	    kf_imp(kf_le(base, m), pat)), p, 2, a, 2, 0);
}

/* forall (args) x y: f(args, x + y) = f(args, x) + f(args, y), and
 * forall (args) n x: f(args, n*x) = n*f(args, x) */
static kt
homlaw(int sym, int which)
{
	struct ksym *s = &ksyms[sym];
	int n = s->arity, i, k = n - 1;
	kt x[8], body, lhs, rhs;
	int nb = k + 2;	/* binders: args 0..k-1, then two */

	for (i = 0; i < k; i++)
		x[i] = kt_bvar((uint32_t)(nb - 1 - i), s->arg[i]);
	if (which == 0) {
		kt u = kt_bvar(1, KS_EL), v = kt_bvar(0, KS_EL);
		x[k] = kt_add(u, v);
		lhs = kt_app(sym, x, n);
		x[k] = u;
		rhs = kt_app(sym, x, n);
		x[k] = v;
		rhs = kt_add(rhs, kt_app(sym, x, n));
		body = kf_eq(lhs, rhs);
		body = kf_bind(KO_ALL, KS_EL, sym_cstr("y"), body);
		body = kf_bind(KO_ALL, KS_EL, sym_cstr("x"), body);
	} else {
		kt c = kt_bvar(1, KS_INT), v = kt_bvar(0, KS_EL);
		x[k] = kt_mul(c, v);
		lhs = kt_app(sym, x, n);
		x[k] = v;
		rhs = kt_mul(c, kt_app(sym, x, n));
		body = kf_eq(lhs, rhs);
		body = kf_bind(KO_ALL, KS_EL, sym_cstr("x"), body);
		body = kf_bind(KO_ALL, KS_INT, sym_cstr("n"), body);
	}
	for (i = k - 1; i >= 0; i--)
		body = kf_bind(KO_ALL, s->arg[i], sym_cstr("a"), body);
	return body;
}

kthm
kr_homdecl(int sym, kthm add, kthm smul)
{
	kthm p[2];
	struct ksym *s;

	if (sym <= 0 || (uint32_t)sym >= nksyms || !okthm(add) || !okthm(smul))
		return bad("hom_decl: bad arguments");
	s = &ksyms[sym];
	if (s->kind != KSY_FUNC || s->res != KS_EL || s->arity < 1 ||
	    s->arg[s->arity - 1] != KS_EL || (s->flags & KSF_HOM))
		return bad("hom_decl: %s cannot be declared a homomorphism",
		    sym_str(s->name));
	if (T.hn[add] || T.hn[smul])
		return bad("hom_decl: the laws must have no hypotheses");
	if (T.concl[add] != homlaw(sym, 0) || T.concl[smul] != homlaw(sym, 1))
		return bad("hom_decl: the laws are not %s and %s",
		    kt_str(homlaw(sym, 0)), kt_str(homlaw(sym, 1)));
	p[0] = add;
	p[1] = smul;
	nhs = 0;
	{
		kthm th = emit(KR_HOMDECL, kf_true(), p, 2, NULL, 0,
		    (uint32_t)sym);
		s->flags |= KSF_HOM;
		s->homthm = th;
		kn_reset();
		return th;
	}
}
