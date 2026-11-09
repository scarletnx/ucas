/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * prove_cmd.c - the statements of the proof framework (parse.c calls
 * prove_command first on every statement):
 *
 *   theorem NAME: FORMULA     state a theorem (ucas formula syntax)
 *   prove NAME [using L ...]  prove it: by the library's proofs of the
 *                             same statement (each route), else by
 *                             kauto with the lemmas named
 *   proof NAME [kernel]       print the proof: the outline of each route,
 *                             the lemmas and axioms it rests on (a tree),
 *                             kernel statistics; "kernel" lists the
 *                             theorem's own inferences
 *   check NAME                re-check with the independent checker
 *   axioms                    every axiom, with its kind
 *   lemmas                    the library's lemmas
 */
#include "ucas.h"
#include "prove.h"
#include "report.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

extern char ks_load_msg[];

static int
user_find(const char *name)
{
	int i = kl_find(name);

	return i;
}

/* the theorem a lemma's proof rests on: lemmas and axioms it cites */
static uint32_t *lemma_of;	/* thm -> lemma index + 1 */
static uint32_t lemma_of_n;

static void
index_lemmas(void)
{
	uint32_t i, n = kthm_count();

	free(lemma_of);
	lemma_of = xcalloc(n + 1, sizeof *lemma_of);
	lemma_of_n = n + 1;
	for (i = 0; i < nklemmas; i++)
		if (klemmas[i].th && klemmas[i].th < lemma_of_n)
			lemma_of[klemmas[i].th] = i + 1;
}

static int
uses_sym(kt t, uint32_t s, int depth)
{
	int i;

	if (KOP(t) == KO_APP && KT.aux[t] == s)
		return 1;
	if (depth > 64)
		return 1;
	for (i = 0; i < KN(t); i++)
		if (uses_sym(KA(t, i), s, depth + 1))
			return 1;
	return 0;
}

struct deps {
	int lem[256], nl;
	int ax[256], na;
	uint32_t inferences;
	uint32_t byrule[KR_NRULES];
};

static void
deps_of(int li, struct deps *d)
{
	uint32_t n = kthm_count(), ns = 0, i;
	uint8_t *mark = xcalloc(n, 1);
	kthm *st = xmalloc(n * sizeof *st), top = klemmas[li].th;

	memset(d, 0, sizeof *d);
	st[ns++] = top;
	mark[top] = 1;
	while (ns) {
		kthm th = st[--ns];
		int r = kthm_rule(th);
		if (th != top && th < lemma_of_n && lemma_of[th]) {
			int l = (int)lemma_of[th] - 1, k;
			for (k = 0; k < d->nl && d->lem[k] != l; k++)
				;
			if (k == d->nl && d->nl < 256)
				d->lem[d->nl++] = l;
			continue;
		}
		d->inferences++;
		d->byrule[r]++;
		if (r == KR_AXIOM) {
			int a = (int)kthm_aux(th), k;
			for (k = 0; k < d->na && d->ax[k] != a; k++)
				;
			if (k == d->na && d->na < 256)
				d->ax[d->na++] = a;
		}
		if (r == KR_CONV) {
			/* a conversion may use a declared homomorphism */
			uint32_t s;
			for (s = 1; s < nksyms; s++) {
				kthm h = ksyms[s].homthm;
				if (h && h < th && !mark[h] && (uses_sym(kthm_concl(th),
				    s, 0) || uses_sym(kthm_concl(kthm_prem(th, 0)), s,
				    0))) {
					mark[h] = 1;
					st[ns++] = h;
				}
			}
		}
		for (i = 0; i < kthm_nprem(th); i++) {
			kthm q = kthm_prem(th, i);
			if (!mark[q]) {
				mark[q] = 1;
				st[ns++] = q;
			}
		}
	}
	free(mark);
	free(st);
}

/* the kernel DAG size of a whole proof, sub-lemmas included */
static uint32_t
total_inferences(kthm top)
{
	uint32_t n = kthm_count(), ns = 0, i, cnt = 0;
	uint8_t *mark = xcalloc(n, 1);
	kthm *st = xmalloc(n * sizeof *st);

	st[ns++] = top;
	mark[top] = 1;
	while (ns) {
		kthm th = st[--ns];
		cnt++;
		for (i = 0; i < kthm_nprem(th); i++) {
			kthm q = kthm_prem(th, i);
			if (!mark[q]) {
				mark[q] = 1;
				st[ns++] = q;
			}
		}
	}
	free(mark);
	free(st);
	return cnt;
}

static void
sort_ints(int *a, int n)
{
	int i, j, t;

	for (i = 1; i < n; i++)
		for (j = i; j > 0 && a[j - 1] > a[j]; j--) {
			t = a[j];
			a[j] = a[j - 1];
			a[j - 1] = t;
		}
}

static void
tree(int li, int depth, uint8_t *seen, int *axused, int *nax)
{
	struct deps d;
	int k, m;

	deps_of(li, &d);
	sort_ints(d.lem, d.nl);
	for (k = 0; k < d.na; k++) {
		for (m = 0; m < *nax && axused[m] != d.ax[k]; m++)
			;
		if (m == *nax && *nax < 512)
			axused[(*nax)++] = d.ax[k];
	}
	printf("%*s%s  (%u inferences", depth * 2, "",
	    sym_str(klemmas[li].name), d.inferences);
	if (d.na) {
		printf("; axioms:");
		sort_ints(d.ax, d.na);
		for (k = 0; k < d.na; k++)
			printf(" %s", sym_str(kaxioms[d.ax[k]].name));
	}
	printf(")%s\n", seen[li] ? " ^" : "");
	if (seen[li])
		return;
	seen[li] = 1;
	for (k = 0; k < d.nl; k++)
		tree(d.lem[k], depth + 1, seen, axused, nax);
}

/* the routes: library lemmas with the same statement */
static int
routes(int ui, int *out, int max)
{
	uint32_t i;
	int n = 0;
	kt f = kn_norm(klemmas[ui].phi);

	for (i = 0; i < nklemmas && n < max; i++)
		if (klemmas[i].lib && klemmas[i].th && (int)i != ui &&
		    kn_norm(klemmas[i].phi) == f)
			out[n++] = (int)i;
	return n;
}

/* the predicates of a formula's conclusion */
static void
preds(kt f, uint32_t *out, int *n)
{
	int i, k;

	if (KOP(f) == KO_PRED) {
		for (k = 0; k < *n && out[k] != KT.aux[f]; k++)
			;
		if (k == *n && *n < 64)
			out[(*n)++] = KT.aux[f];
		return;
	}
	if (KSORT(f) != KS_PROP)
		return;
	for (i = 0; i < KN(f); i++)
		preds(KA(f, i), out, n);
}

/* do two statements talk about the same things (a replay may apply) */
static int
same_shape(kt a, kt b)
{
	uint32_t pa[64], pb[64];
	int na = 0, nb = 0, i, k, common = 0;

	while (KOP(a) == KO_ALL && KOP(b) == KO_ALL) {
		if (KT.aux[a] != KT.aux[b])
			return 0;
		a = KA(a, 0);
		b = KA(b, 0);
	}
	if (KOP(a) == KO_ALL || KOP(b) == KO_ALL)
		return 0;
	if (KOP(a) == KO_IMP)
		a = KA(a, 1);
	if (KOP(b) == KO_IMP)
		b = KA(b, 1);
	preds(a, pa, &na);
	preds(b, pb, &nb);
	for (i = 0; i < na; i++)
		for (k = 0; k < nb; k++)
			common += pa[i] == pb[k];
	return common > 0;
}

static void
cmd_theorem(const char *text)
{
	char name[64], err[256];
	const char *s = text, *colon;
	size_t k = 0;
	kt phi;
	int li;

	while (*s == ' ')
		s++;
	while ((s[k] >= 'a' && s[k] <= 'z') || (s[k] >= 'A' && s[k] <= 'Z') ||
	    (s[k] >= '0' && s[k] <= '9') || s[k] == '_')
		k++;
	if (!k || k >= sizeof name)
		fail("theorem NAME: FORMULA");
	memcpy(name, s, k);
	name[k] = 0;
	colon = strchr(s + k, ':');
	if (!colon)
		fail("theorem NAME: FORMULA");
	phi = kt_parse(colon + 1, NULL, err, sizeof err);
	if (!phi)
		fail("%s", err);
	if (KSORT(phi) != KS_PROP || KT.lb[phi])
		fail("not a closed formula");
	li = kl_find(name);
	if (li >= 0 && klemmas[li].lib)
		fail("%s is a library lemma; choose another name", name);
	{
		size_t n = strlen(colon + 1) + 1;
		char *src = xmalloc(n);
		memcpy(src, colon + 1, n);
		/* re-stating a theorem forgets its proof */
		li = kl_find(name);
		if (li < 0) {
			extern int ks_add_user(const char *, kt, const char *);
			li = ks_add_user(name, phi, src);
		} else {
			klemmas[li].phi = phi;
			klemmas[li].th = KNONE;
			klemmas[li].src = src;
		}
	}
	rep_begin("theorem");
	rep_str("name", name);
	rep_str("statement", kt_str(phi));
	printf("theorem %s: %s\n", name, kt_str(phi));
}

static int
need(const char *name)
{
	int li = user_find(name);

	if (li < 0)
		fail("no theorem %s (state it with: theorem %s: ...)", name, name);
	return li;
}

static void
cmd_prove(const char *rest)
{
	char name[64], w[64], msg[512];
	const char *s = rest;
	int li, r[8], nr, k;
	kthm th = KNONE;
	uint32_t steps;

	if (sscanf(s, "%63s", name) != 1)
		fail("prove NAME");
	li = need(name);
	s = strstr(s, name) + strlen(name);
	rep_begin("prove");
	rep_str("name", name);
	rep_str("statement", kt_str(klemmas[li].phi));
	nr = klemmas[li].lib ? 0 : routes(li, r, 8);
	if (klemmas[li].lib) {
		r[0] = li;
		nr = 1;
	}
	if (nr) {
		printf("%s: %s\n", name, kt_str(klemmas[li].phi));
		rep_list("routes");
		for (k = 0; k < nr; k++) {
			int ok;
			kthm t = kr_conv(klemmas[r[k]].th, klemmas[li].phi);
			index_lemmas();
			ok = kcheck(t, msg, sizeof msg, &steps);
			printf("  proved by %s (%u kernel inferences with its "
			    "lemmas); independent re-check: %s\n",
			    sym_str(klemmas[r[k]].name), total_inferences(t),
			    ok ? "ok" : msg);
			rep_obj(NULL);
			rep_str("lemma", sym_str(klemmas[r[k]].name));
			rep_int("inferences", total_inferences(t));
			rep_bool("rechecked", ok);
			rep_end();
			if (ok && !th)
				th = t;
		}
		rep_end();
		if (th) {
			klemmas[li].th = th;
			rep_bool("proved", 1);
			rep_proof("kernel", "LCF kernel proof, re-checked by "
			    "kcheck.c");
			return;
		}
	}
	/* automation with the lemmas named after "using" */
	{
		struct kctx *c = xmalloc(sizeof *c);
		memset(c, 0, sizeof *c);
		c->depth = 4;
		s = strstr(s, "using") ? strstr(s, "using") + 5 : "";
		while (sscanf(s, "%63s", w) == 1) {
			int l = kl_find(w), a = kax_find(w);
			kthm t = l >= 0 && klemmas[l].th ? klemmas[l].th :
			    a >= 0 ? kr_axiom(w) : KNONE;
			if (!t)
				fail("no lemma or axiom %s", w);
			ka_add(c, t, sym_cstr(w));
			s = strstr(s, w) + strlen(w);
		}
		ka_budget = 60000;
		th = ka_prove(c, klemmas[li].phi);
		free(c);
	}
	if (th) {
		index_lemmas();
		k = kcheck(th, msg, sizeof msg, &steps);
		klemmas[li].th = th;
		printf("%s: proved automatically (%u kernel inferences); "
		    "independent re-check: %s\n", name, total_inferences(th),
		    k ? "ok" : msg);
		rep_bool("proved", 1);
		rep_bool("rechecked", k);
		return;
	}
	/* the library's theorem proofs, replayed on this statement */
	{
		char why[512];
		snprintf(why, sizeof why, "%s", ka_why);
		uint32_t i;
		char m[512];
		for (i = 0; i < nklemmas; i++) {
			if (!klemmas[i].lib || !klemmas[i].route ||
			    strcmp(klemmas[i].route, "theorem") ||
			    !same_shape(klemmas[i].phi, klemmas[li].phi))
				continue;
			th = ks_replay((int)i, klemmas[li].phi, m, sizeof m);
			if (th) {
				index_lemmas();
				k = kcheck(th, msg, sizeof msg, &steps);
				klemmas[li].th = th;
				printf("%s: proved by the proof of %s (%u kernel "
				    "inferences); independent re-check: %s\n", name,
				    sym_str(klemmas[i].name), total_inferences(th),
				    k ? "ok" : msg);
				rep_bool("proved", 1);
				rep_bool("rechecked", k);
				return;
			}
			printf("  the proof of %s does not apply: %s\n",
			    sym_str(klemmas[i].name), m);
			rep_list("replays");
			rep_obj(NULL);
			rep_str("lemma", sym_str(klemmas[i].name));
			rep_str("fails", m);
			rep_end();
			rep_end();
		}
		snprintf(ka_why, sizeof ka_why, "%s", why);
	}
	printf("%s: NOT proved\n  automation: %s\n", name, ka_why);
	rep_bool("proved", 0);
	rep_str("reason", ka_why);
	{
		struct sbuf b = {0};
		if (pm_refute(klemmas[li].phi, &b)) {
			printf("  counterexample: %s\n  (evaluated in a concrete "
			    "model: complexes are standard spaces' cochains, "
			    "bock(C,p) homalg.h's exact Bockstein spectral "
			    "sequence; a computation, not a kernel proof)\n",
			    sb_str(&b));
			rep_str("counterexample", sb_str(&b));
		} else
			printf("  no counterexample among the standard spaces "
			    "(S^1 S^2 RP^2 T^2 RP^3 L(4;1) L(9;1) RP^4), "
			    "p, n, r <= 4\n");
		free(b.p);
	}
}

static void
cmd_proof(const char *rest)
{
	char name[64];
	int li, r[8], nr, k, nax = 0, axused[512];
	uint8_t *seen;

	if (sscanf(rest, "%63s", name) != 1)
		fail("proof NAME");
	li = need(name);
	if (!klemmas[li].th)
		fail("%s is not proved (prove %s)", name, name);
	nr = klemmas[li].lib ? 0 : routes(li, r, 8);
	if (klemmas[li].lib || !nr) {
		r[0] = li;
		nr = 1;
	}
	index_lemmas();
	rep_begin("proof");
	rep_str("name", name);
	printf("%s: %s\n", name, kt_str(klemmas[li].phi));
	seen = xcalloc(nklemmas, 1);
	for (k = 0; k < nr; k++) {
		printf("route %c: %s\n", 'a' + k, sym_str(klemmas[r[k]].name));
		if (klemmas[r[k]].lib) {
			extern void ks_print_script(int);
			ks_print_script(r[k]);
		}
		printf("  rests on:\n");
		memset(seen, 0, nklemmas);
		tree(r[k], 2, seen, axused, &nax);
	}
	free(seen);
	sort_ints(axused, nax);
	printf("axioms used:\n");
	rep_list("axioms");
	for (k = 0; k < nax; k++) {
		printf("  %-14s [%s] %s\n", sym_str(kaxioms[axused[k]].name),
		    kaxioms[axused[k]].kind, kaxioms[axused[k]].text);
		rep_str(NULL, sym_str(kaxioms[axused[k]].name));
	}
	rep_end();
}

static void
cmd_check(const char *rest)
{
	char name[64], msg[512];
	int li, ok;
	uint32_t steps = 0;

	if (sscanf(rest, "%63s", name) != 1)
		fail("check NAME");
	li = need(name);
	if (!klemmas[li].th)
		fail("%s is not proved", name);
	ok = kcheck(klemmas[li].th, msg, sizeof msg, &steps);
	rep_begin("check");
	rep_str("name", name);
	rep_bool("ok", ok);
	rep_int("inferences", steps);
	printf("%s: %s\n", name, ok ? msg : "FAILED");
	if (!ok)
		printf("  %s\n", msg);
}

/* the axioms a proof rests on (every axiom in its DAG) */
static void
used_axioms(kthm top, uint8_t *used)
{
	uint32_t n = kthm_count(), ns = 0, i;
	uint8_t *mark = xcalloc(n, 1);
	kthm *st = xmalloc(n * sizeof *st);

	st[ns++] = top;
	mark[top] = 1;
	while (ns) {
		kthm th = st[--ns];
		if (kthm_rule(th) == KR_AXIOM)
			used[kthm_aux(th)] = 1;
		if (kthm_rule(th) == KR_CONV) {
			uint32_t s;
			for (s = 1; s < nksyms; s++) {
				kthm h = ksyms[s].homthm;
				if (h && h < th && !mark[h]) {
					mark[h] = 1;
					st[ns++] = h;
				}
			}
		}
		for (i = 0; i < kthm_nprem(th); i++) {
			kthm q = kthm_prem(th, i);
			if (!mark[q]) {
				mark[q] = 1;
				st[ns++] = q;
			}
		}
	}
	free(mark);
	free(st);
}

static void
cmd_axioms(const char *rest)
{
	uint32_t i;
	char name[64];
	uint8_t *used = NULL;

	rep_begin("axioms");
	if (sscanf(rest, "%63s", name) == 1) {
		int li = need(name), r[8], nr, k;
		if (!klemmas[li].th)
			fail("%s is not proved (prove %s)", name, name);
		used = xcalloc(nkaxioms + 1, 1);
		nr = klemmas[li].lib ? 0 : routes(li, r, 8);
		if (!nr) {
			r[0] = li;
			nr = 1;
		}
		for (k = 0; k < nr; k++)
			used_axioms(klemmas[r[k]].th, used);
		rep_str("name", name);
		printf("the axioms %s rests on (every route):\n", name);
	}
	rep_list("axioms");
	for (i = 0; i < nkaxioms; i++) {
		if (used && !used[i])
			continue;
		printf("%-14s [%s] %s\n", sym_str(kaxioms[i].name),
		    kaxioms[i].kind, kaxioms[i].text);
		rep_obj(NULL);
		rep_str("name", sym_str(kaxioms[i].name));
		rep_str("kind", kaxioms[i].kind);
		rep_str("statement", kaxioms[i].text);
		rep_end();
	}
	rep_end();
	free(used);
}

static void
cmd_lemmas(void)
{
	uint32_t i;

	rep_begin("lemmas");
	rep_list("lemmas");
	for (i = 0; i < nklemmas; i++) {
		if (!klemmas[i].lib)
			continue;
		printf("%-18s %s\n", sym_str(klemmas[i].name), klemmas[i].src);
		rep_obj(NULL);
		rep_str("name", sym_str(klemmas[i].name));
		rep_str("statement", klemmas[i].src);
		rep_end();
	}
	rep_end();
}

int
prove_command(struct P *p)
{
	static char text[16384];
	char w[32];
	uint32_t pos = cmdp_pos(p);
	int c;

	if (!cmdp_word(p, w, sizeof w))
		return 0;
	if (!strcmp(w, "theorem"))
		c = 1;
	else if (!strcmp(w, "prove"))
		c = 2;
	else if (!strcmp(w, "proof"))
		c = 3;
	else if (!strcmp(w, "check"))
		c = 4;
	else if (!strcmp(w, "axioms"))
		c = 5;
	else if (!strcmp(w, "lemmas") && cmdp_eos(p))
		c = 6;
	else {
		cmdp_seek(p, pos);
		return 0;
	}
	if (c <= 4 && (cmdp_eos(p) || func_find(sym_cstr(w)))) {
		cmdp_seek(p, pos);
		return 0;
	}
	text[0] = 0;
	if (!cmdp_eos(p))
		cmdp_rest(p, text, sizeof text, NULL);
	notes_flush();
	ks_load_library();
	switch (c) {
	case 1:
		cmd_theorem(text);
		break;
	case 2:
		cmd_prove(text);
		break;
	case 3:
		cmd_proof(text);
		break;
	case 4:
		cmd_check(text);
		break;
	case 5:
		cmd_axioms(text);
		break;
	case 6:
		cmd_lemmas();
		break;
	}
	return 1;
}

/* nlp.c: "theorem NAME: ...", "prove NAME [using ...]", "proof NAME
 * [kernel]", "check NAME", "axioms", "lemmas" are statements */
int
prove_is_cmd(const char *s, unsigned long n)
{
	static const char *const w[] = {"theorem", "prove", "proof", "check",
		"axioms", "lemmas", NULL};
	char a[16], b[64];
	unsigned long i = 0, k;
	int c;

	for (c = 0; w[c]; c++) {
		k = strlen(w[c]);
		if (n >= k && !strncmp(s, w[c], k) && (n == k || s[k] == ' '))
			break;
	}
	if (!w[c])
		return 0;
	if (c == 5)
		return n == strlen(w[c]);
	if (c == 4 && n == strlen(w[c]))
		return 1;
	i = strlen(w[c]);
	while (i < n && s[i] == ' ')
		i++;
	k = 0;
	while (i < n && k < sizeof b - 1 && (isalnum((unsigned char)s[i]) ||
	    s[i] == '_'))
		b[k++] = s[i++];
	b[k] = 0;
	if (!k)
		return 0;
	while (i < n && s[i] == ' ')
		i++;
	snprintf(a, sizeof a, "%s", w[c]);
	if (c == 0)
		return i < n && s[i] == ':';
	if (i == n)
		return 1;
	return (c == 1 && !strncmp(s + i, "using ", 6)) ||
	    (c == 2 && !strncmp(s + i, "kernel", 6));
}
