/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cf_bridge.c: ucas's engines as cfront's candidate generators.
 *
 * cfront lowers each C value to a ucas node over typed leaves (with the
 * call sites' and assumed ranges, and C's UB as the domain).  The bridge
 * offers that value to the rest of ucas, engine by engine, each within
 * its share of a CPU budget (option ebudget; engines all|find,gates,...):
 *
 *   find    the enumerative search (find.c): cheaper terms, the peephole
 *           drops, per-case (x*K + C) >> S fits; pext/pdep matches become
 *           the BMI2 instruction
 *   gates   bitwise functions of up to 6 word inputs: the per-bit truth
 *           table, its minimal network over the target's basis (gates.c:
 *           x86bmi, avx512 with vpternlog as one gate)
 *   alg     the value's polynomial structure over Z (alg.c): factored,
 *           collected, and P/Q cancelled where Q divides P exactly
 *   ring    modular inverses, linear congruences, CRT, magic numbers
 *   dio     congruence obstructions (residues mod q), sums of two
 *           squares, rank-0 elliptic curves (ecq.c's 2-isogeny descent)
 *   gf2     GF(2)-affine bit maps: gf2p8affine, pext/pdep, CRCs by
 *           pclmul (a Barrett reduction from the ring's GF(2)[t])
 *   loops   closed forms beyond degree 2 (cf_lower.c calls back here)
 *           and float sums as integrals (an approximation, opt-in)
 *   float   the IEEE rewrites of fsimp that are bit-exact
 *   bound   comparisons decided by bound.c's proved bounds
 *
 * Every candidate carries its engine's certificate (a witness, an
 * inverse multiplied back, a residue table, a factorisation, a GF(2)
 * linearity proof ...) and is then proved by cfront like any other
 * (exhaustively, by SAT, or, when SAT gives up, by the certificate's
 * derivation with a sample checking its premises), priced by the
 * target's costs, and applied only when cheaper.
 *
 * Machine arithmetic is not the integers: an identity over Z holds mod
 * 2^w only through ring operations (+ - *), never through a division;
 * a congruence mod q survives wrapping only when q divides 2^w.  So
 * every engine works either in Z/2^w (ring operations), or on values
 * shown not to wrap (sound intervals, or signed arithmetic whose
 * overflow is outside the domain as C UB).
 */
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfront.h"
#include "alg.h"
#include "bound.h"
#include "ecq.h"
#include "ring.h"
#include "x86.h"

/* ---- budget ---- */

static double eng_used[CF_ENG_N];
static const struct cf_opts *eng_run;
static clock_t eng_t0;
static int eng_cur = -1;

static double
eng_total(void)
{
	const struct cf_opts *o = cf_run_opts();

	if (o && o->ebudget > 0)
		return o->ebudget;
	return o && o->budget > 0 ? o->budget : 10;
}

/* the searches get more of the total than the analyses */
static const int eng_weight[CF_ENG_N] = {3, 1, 1, 1, 1, 1, 1, 1, 1, 4};

static double
eng_share(int k)
{
	int j, sum = 0;

	for (j = 0; j < CF_ENG_N; j++)
		if (cf_engine_on(j))
			sum += eng_weight[j];
	return eng_total() * eng_weight[k] / (sum ? sum : 1);
}

/* start engine k on a value: 0 when it is off or out of its share */
static int
eng_begin(int k)
{
	const struct cf_opts *o = cf_run_opts();

	if (o != eng_run) {
		memset(eng_used, 0, sizeof eng_used);
		eng_run = o;
	}
	if (!cf_engine_on(k))
		return 0;
	/* the synthesis is budgeted in sketches (deterministic), not time */
	if (k != CF_ENG_SYNTH && eng_used[k] > eng_share(k))
		return 0;
	eng_cur = k;
	eng_t0 = clock();
	return 1;
}

static void
eng_end(void)
{
	if (eng_cur >= 0)
		eng_used[eng_cur] += (double)(clock() - eng_t0) / CLOCKS_PER_SEC;
	eng_cur = -1;
}

/* seconds left of the current engine's share */
static double
eng_left(void)
{
	return eng_share(eng_cur) - eng_used[eng_cur] -
	    (double)(clock() - eng_t0) / CLOCKS_PER_SEC;
}

/* ---- the value's context ---- */

struct bx {
	const struct cf_target *t;
	const struct cf_value *v;
	nid e;
	struct func f;
	struct cf_cands *cs;
	struct plan p;
	struct ival *iv;
	nid *rn;		/* the nodes of e, in id order */
	uint32_t nr;
	int wrapv;
};

static int
cmp_nid(const void *a, const void *b)
{
	nid x = *(const nid *)a, y = *(const nid *)b;

	return x < y ? -1 : x > y;
}

static void
bx_init(struct bx *B, const struct cf_target *t, const struct cf_value *v,
    nid e, struct cf_cands *cs)
{
	uint8_t *seen;
	nid *st;
	uint32_t sp = 0;

	memset(B, 0, sizeof *B);
	B->t = t;
	B->v = v;
	B->e = e;
	B->cs = cs;
	B->wrapv = cf_run_opts() ? cf_run_opts()->wrapv : 0;
	cf_value_func(v, &B->f);
	B->f.body = e;
	plan_build(&B->p, &e, 1, 0);
	B->iv = xcalloc(B->p.n + 1, sizeof *B->iv);
	plan_ival(&B->p, &B->f, B->iv);
	seen = xcalloc(N.n, 1);
	st = xmalloc((N.n + 1) * sizeof *st);
	B->rn = xmalloc((N.n + 1) * sizeof *B->rn);
	st[sp++] = e;
	while (sp) {
		nid x = st[--sp];
		if (x == 0 || seen[x])
			continue;
		seen[x] = 1;
		B->rn[B->nr++] = x;
		if (N.op[x] == O_CONST || N.op[x] == O_VAR || N.op[x] == O_TABLE)
			continue;
		if (N.op[x] == O_INDEX) {
			st[sp++] = N.b[x];
			continue;
		}
		if (N.op[x] == O_LOOP)
			continue;
		if (N.a[x])
			st[sp++] = N.a[x];
		if (N.b[x])
			st[sp++] = N.b[x];
		if (N.c[x] && (N.op[x] == O_SEL || N.op[x] == O_FMA))
			st[sp++] = N.c[x];
	}
	qsort(B->rn, B->nr, sizeof *B->rn, cmp_nid);
	free(seen);
	free(st);
}

static void
bx_free(struct bx *B)
{
	plan_free(&B->p);
	free(B->iv);
	free(B->rn);
}

/* the sound interval of n's value (ideal integers): 0 if none */
static int
ival(const struct bx *B, nid n, int64_t *lo, int64_t *hi)
{
	uint32_t k;

	if (is_const(n) && TY_ISINT(N.ty[n])) {
		*lo = *hi = N.ki[n];
		return 1;
	}
	k = plan_loc(&B->p, n);
	if (k == NOTOK || B->iv[k].top)
		return 0;
	*lo = B->iv[k].lo;
	*hi = B->iv[k].hi;
	return 1;
}

static int
ty_range(ty_t t, int64_t *lo, int64_t *hi)
{
	int w = TYW(t);

	if (t == TY_BOOL) {
		*lo = 0;
		*hi = 1;
		return 1;
	}
	if (TYK(t) == K_UINT) {
		*lo = 0;
		*hi = w >= 63 ? INT64_MAX : (int64_t)((1ULL << w) - 1);
		return w < 64;
	}
	if (TYK(t) == K_SINT) {
		*lo = w >= 64 ? INT64_MIN : -((int64_t)1 << (w - 1));
		*hi = w >= 64 ? INT64_MAX : ((int64_t)1 << (w - 1)) - 1;
		return 1;
	}
	return 0;
}

/*
 * n's machine value is the integer its operations compute (no wrap
 * anywhere in it): each operation's ideal interval fits its type, or
 * it is signed arithmetic whose overflow C makes UB (outside the
 * domain, unless wrapv).  Leaves and atoms are taken as they are.
 */
static int
exact(const struct bx *B, nid n, int depth)
{
	int op = N.op[n];
	ty_t t = N.ty[n];
	int64_t lo, hi, tl, th;

	if (depth > 40)
		return 0;
	switch (op) {
	case O_CONST:
	case O_VAR:
		return 1;
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_NEG:
	case O_SHL:
	case O_CAST:
		break;
	default:
		return 1;	/* an atom */
	}
	if (!TY_SIZED(t))
		return 0;
	if (!(TYK(t) == K_SINT && !B->wrapv && op != O_CAST && op != O_SHL)) {
		uint32_t kk = plan_loc(&B->p, n);
		if (kk == NOTOK || B->iv[kk].wrap)
			return 0;
		if (!ival(B, n, &lo, &hi) || !ty_range(t, &tl, &th) || lo < tl ||
		    hi > th)
			return 0;
		if (op == O_CAST && (!ival(B, N.a[n], &lo, &hi) || lo < tl ||
		    hi > th))
			return 0;
	}
	if (!exact(B, N.a[n], depth + 1))
		return 0;
	if (N.b[n] && op != O_CAST && !exact(B, N.b[n], depth + 1))
		return 0;
	return 1;
}

static nid
brepl(nid root, nid from, nid to)
{
	jmp_buf saved;
	nid r = 0;

	if (from == root)
		return to;
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = xreplace(root, from, to);
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return r;
}

/* a candidate: e with n replaced by r (cast to n's type) */
static int
cand(struct bx *B, nid n, nid r, int derived, const char *what,
    const char *kind, const char *fmt, ...)
{
	va_list ap;
	nid sem;
	struct cf_cand *c;

	if (r == 0)
		return 0;
	if (N.ty[r] != N.ty[n]) {
		jmp_buf saved;
		nid rr = 0;
		memcpy(saved, fail_jmp, sizeof saved);
		note_mute++;
		if (setjmp(fail_jmp) == 0)
			rr = mk_cast(N.ty[n], r);
		note_mute--;
		memcpy(fail_jmp, saved, sizeof saved);
		notes_drop();
		if (!(r = rr))
			return 0;
	}
	if ((sem = brepl(B->e, n, r)) == 0 || sem == B->e)
		return 0;
	cf_cand_add(B->cs, sem, NULL, NULL, 0, 0, what);
	c = &B->cs->c[B->cs->n - 1];
	snprintf(c->engine, sizeof c->engine, "%s", cf_engine_name(eng_cur));
	snprintf(c->cert_kind, sizeof c->cert_kind, "%s", kind);
	va_start(ap, fmt);
	vsnprintf(c->cert, sizeof c->cert, fmt, ap);
	va_end(ap);
	if (derived) {
		/* the certificate is a derivation: when SAT gives up it is
		 * the proof (the sample then checks its premises) */
		snprintf(c->lemma, sizeof c->lemma, "%.119s", c->cert);
		c->derived = 1;
	}
	return 1;
}

static nid
kint(ty_t t, int64_t v)
{
	return mk_const_i(t, int_norm(t, v));
}

/* a condition seen through C's int conversions: i32(c) != 0, i32(c) */
static nid
cond_of(nid n)
{
	for (;;) {
		if (N.op[n] == O_CAST && TY_ISINT(N.ty[N.a[n]]) &&
		    (N.ty[N.a[n]] == TY_BOOL || (opinfo[N.op[N.a[n]]].fl & OF_CMP) ||
		    N.op[N.a[n]] == O_LAND || N.op[N.a[n]] == O_LOR))
			n = N.a[n];
		else if (N.op[n] == O_NE && is_const(N.b[n]) && N.ki[N.b[n]] == 0 &&
		    N.op[N.a[n]] == O_CAST && (N.ty[N.a[N.a[n]]] == TY_BOOL ||
		    (opinfo[N.op[N.a[N.a[n]]]].fl & OF_CMP)))
			n = N.a[N.a[n]];
		else
			return n;
	}
}

/* through casts that keep every value of their operand (by intervals) */
static nid
strip_keep(const struct bx *B, nid n)
{
	while (N.op[n] == O_CAST && TY_SIZED(N.ty[n]) && TY_SIZED(N.ty[N.a[n]])) {
		int64_t lo, hi, tl, th;
		if (!ival(B, N.a[n], &lo, &hi) || !ty_range(N.ty[n], &tl, &th) ||
		    lo < tl || hi > th)
			break;
		n = N.a[n];
	}
	return n;
}

/* ---- polynomials over Z in a few atoms (exact int64) ---- */

#define PA 4
#define PT 48

struct bpoly {
	int n;
	int64_t c[PT];
	uint8_t e[PT][PA];
};

struct patoms {
	int n;
	nid a[PA];
};

static int ovf;

static int64_t
sadd(int64_t a, int64_t b)
{
	int64_t r;

	if (__builtin_add_overflow(a, b, &r))
		ovf = 1;
	return r;
}

static int64_t
smul(int64_t a, int64_t b)
{
	int64_t r;

	if (__builtin_mul_overflow(a, b, &r))
		ovf = 1;
	return r;
}

static void
bp_addterm(struct bpoly *p, int64_t c, const uint8_t *e)
{
	int k;

	if (c == 0)
		return;
	for (k = 0; k < p->n; k++)
		if (!memcmp(p->e[k], e, PA)) {
			p->c[k] = sadd(p->c[k], c);
			if (p->c[k] == 0) {
				p->n--;
				memmove(&p->c[k], &p->c[k + 1], (size_t)(p->n - k) *
				    sizeof p->c[0]);
				memmove(&p->e[k], &p->e[k + 1], (size_t)(p->n - k) * PA);
			}
			return;
		}
	if (p->n == PT) {
		ovf = 1;
		return;
	}
	p->c[p->n] = c;
	memcpy(p->e[p->n], e, PA);
	p->n++;
}

static void
bp_mul(struct bpoly *r, const struct bpoly *a, const struct bpoly *b)
{
	int i, j, k;
	struct bpoly t;

	t.n = 0;
	for (i = 0; i < a->n; i++)
		for (j = 0; j < b->n; j++) {
			uint8_t e[PA];
			for (k = 0; k < PA; k++) {
				e[k] = (uint8_t)(a->e[i][k] + b->e[j][k]);
				if (e[k] > 12)
					ovf = 1;
			}
			bp_addterm(&t, smul(a->c[i], b->c[j]), e);
		}
	*r = t;
}

/* p = n as a polynomial in atoms: + - * neg, shifts by constants and
 * integer casts are taken as ring operations (callers decide whether
 * the machine value is that integer, or that integer mod 2^w) */
static int
bp_of(nid n, struct bpoly *p, struct patoms *A, int depth)
{
	struct bpoly a, b;
	uint8_t z[PA] = {0};
	int k;

	p->n = 0;
	if (depth > 30 || ovf)
		return 0;
	switch (N.op[n]) {
	case O_CONST:
		if (!TY_ISINT(N.ty[n]))
			return 0;
		bp_addterm(p, N.ki[n], z);
		return 1;
	case O_ADD:
	case O_SUB:
		if (!bp_of(N.a[n], &a, A, depth + 1) ||
		    !bp_of(N.b[n], &b, A, depth + 1))
			return 0;
		*p = a;
		for (k = 0; k < b.n; k++)
			bp_addterm(p, N.op[n] == O_ADD ? b.c[k] : smul(-1, b.c[k]),
			    b.e[k]);
		return !ovf;
	case O_MUL:
		if (!bp_of(N.a[n], &a, A, depth + 1) ||
		    !bp_of(N.b[n], &b, A, depth + 1))
			return 0;
		bp_mul(p, &a, &b);
		return !ovf;
	case O_NEG:
		if (!bp_of(N.a[n], &a, A, depth + 1))
			return 0;
		for (k = 0; k < a.n; k++)
			bp_addterm(p, smul(-1, a.c[k]), a.e[k]);
		return !ovf;
	case O_SHL:
		if (is_const(N.b[n]) && N.ki[N.b[n]] >= 0 && N.ki[N.b[n]] < 62) {
			if (!bp_of(N.a[n], &a, A, depth + 1))
				return 0;
			for (k = 0; k < a.n; k++)
				bp_addterm(p, smul(a.c[k], (int64_t)1 << N.ki[N.b[n]]),
				    a.e[k]);
			return !ovf;
		}
		break;
	case O_CAST:
		if (TY_SIZED(N.ty[n]) && TY_SIZED(N.ty[N.a[n]]) &&
		    N.ty[n] != TY_BOOL && N.ty[N.a[n]] != TY_BOOL)
			return bp_of(N.a[n], p, A, depth + 1);
		break;
	}
	/* an atom */
	if (!TY_SIZED(N.ty[n]))
		return 0;
	for (k = 0; k < A->n && A->a[k] != n; k++)
		;
	if (k == A->n) {
		if (A->n == PA)
			return 0;
		A->a[A->n++] = n;
	}
	z[k] = 1;
	bp_addterm(p, 1, z);
	return 1;
}

/* p mod q at the atoms' residues r[] */
static int64_t
bp_evalmod(const struct bpoly *p, const int64_t *r, int64_t q)
{
	int k, j, i;
	int64_t s = 0;

	for (k = 0; k < p->n; k++) {
		int64_t t = ((p->c[k] % q) + q) % q;
		for (j = 0; j < PA; j++)
			for (i = 0; i < p->e[k][j]; i++)
				t = (int64_t)((__int128)t * r[j] % q);
		s = (s + t) % q;
	}
	return s;
}

static void
bp_str(struct sbuf *b, const struct bpoly *p, const struct patoms *A,
    const char *const *nm)
{
	int k, j, any = 0;

	for (k = 0; k < p->n; k++) {
		int64_t c = p->c[k];
		int deg = 0;
		for (j = 0; j < PA; j++)
			deg += p->e[k][j];
		if (any)
			sb_puts(b, c < 0 ? " - " : " + ");
		else if (c < 0)
			sb_puts(b, "-");
		if ((c != 1 && c != -1) || deg == 0)
			sb_printf(b, "%lld", (long long)(c < 0 ? -c : c));
		for (j = 0; j < A->n; j++)
			if (p->e[k][j])
				sb_printf(b, "%s%s%s", (c != 1 && c != -1) || deg == 0 ?
				    "" : "", nm[j], p->e[k][j] > 1 ? (p->e[k][j] == 2 ?
				    "^2" : p->e[k][j] == 3 ? "^3" : "^k") : "");
		any = 1;
	}
	if (!any)
		sb_puts(b, "0");
}

/* a short name for an atom: the leaf's C text, or a placeholder */
static const char *
atom_name(const struct bx *B, nid a, int k)
{
	static char buf[PA][48];

	while (N.op[a] == O_CAST)
		a = N.a[a];
	if (N.op[a] == O_VAR && N.ki[a] < B->v->nleaf)
		snprintf(buf[k], sizeof buf[k], "%.40s", B->v->leaf[N.ki[a]].text);
	else
		snprintf(buf[k], sizeof buf[k], "u%d", k);
	return buf[k];
}

/* the residues of e mod q: each atom over all q residues (q^k <= 2^16) */
static int
residues(const struct bpoly *p, int na, int64_t q, uint8_t *hit)
{
	int64_t r[PA] = {0}, tot = 1;
	int k;

	for (k = 0; k < na; k++) {
		tot *= q;
		if (tot > 65536)
			return 0;
	}
	memset(hit, 0, (size_t)q);
	for (;;) {
		hit[bp_evalmod(p, r, q)] = 1;
		for (k = 0; k < na; k++) {
			if (++r[k] < q)
				break;
			r[k] = 0;
		}
		if (k == na)
			break;
	}
	return 1;
}

static void
residue_list(struct sbuf *b, const uint8_t *hit, int64_t q)
{
	int64_t r;
	int any = 0;

	sb_puts(b, "{");
	for (r = 0; r < q; r++)
		if (hit[r])
			sb_printf(b, "%s%lld", any++ ? ", " : "", (long long)r);
	sb_puts(b, "}");
}

/* ================================================================
 * ring: modular inverses, linear congruences, CRT, magic numbers
 * ================================================================ */

/* the inverse of odd k mod 2^64 (Newton: each step doubles the bits) */
static uint64_t
inv2k(uint64_t k)
{
	uint64_t x = k;
	int i;

	for (i = 0; i < 6; i++)
		x *= 2 - k * x;
	return x;
}

static uint64_t
wmask(int w)
{
	return w >= 64 ? ~0ULL : (1ULL << w) - 1;
}

/* x * K == C (or !=) in uW: x == C K^-1 (mod 2^W) */
static void
ring_inverse(struct bx *B, nid n)
{
	nid a = N.a[n], b = N.b[n], x, k;
	ty_t t = N.ty[a];
	int w = TYW(t), s;
	uint64_t K, C, Ko, inv, m;
	nid r;

	if (!TY_SIZED(t) || t == TY_BOOL || !is_const(b) || N.op[a] != O_MUL)
		return;
	if (is_const(N.b[a])) {
		x = N.a[a];
		k = N.b[a];
	} else if (is_const(N.a[a])) {
		x = N.b[a];
		k = N.a[a];
	} else
		return;
	K = (uint64_t)N.ki[k] & wmask(w);
	C = (uint64_t)N.ki[b] & wmask(w);
	if (K == 0 || K == 1)
		return;
	if (TYK(t) == K_SINT && !B->wrapv) {
		/* no overflow where defined: x K = C over Z */
		int64_t ks = N.ki[k], cs = N.ki[b];
		if (ks == 0 || ks == -1)
			return;
		if (cs % ks != 0) {
			r = mk_const_i(TY_BOOL, N.op[n] == O_NE);
			cand(B, n, r, 1, N.op[n] == O_EQ ? "x * K == C is never true" :
			    "x * K != C is always true", "derivation",
			    "%lld does not divide %lld, and where the product is defined "
			    "(signed overflow is UB) it is the integer product",
			    (long long)ks, (long long)cs);
		} else {
			r = mk_bin(N.op[n], x, kint(t, cs / ks));
			cand(B, n, r, 1, "x * K == C as x == C / K", "derivation",
			    "%lld * %lld = %lld exactly; the integer product (no "
			    "overflow where defined: C UB) is injective in x",
			    (long long)(cs / ks), (long long)ks, (long long)cs);
		}
		return;
	}
	if (TYK(t) != K_UINT && TYK(t) != K_SINT)
		return;
	/* K = 2^s Ko, Ko odd */
	for (s = 0, Ko = K; !(Ko & 1); s++)
		Ko >>= 1;
	inv = inv2k(Ko);
	if (((Ko * inv) & wmask(w)) != 1)
		return;
	if (s && (C & ((1ULL << s) - 1))) {
		r = mk_const_i(TY_BOOL, N.op[n] == O_NE);
		cand(B, n, r, 1, N.op[n] == O_EQ ? "x * K == C is never true" :
		    "x * K != C is always true", "inverse checked",
		    "x*%llu is a multiple of 2^%d mod 2^%d, and %llu is not",
		    (unsigned long long)K, s, w, (unsigned long long)C);
		return;
	}
	m = wmask(w - s);
	{
		uint64_t c2 = ((C >> s) * inv) & m;
		char what[96];
		nid lhs = s ? mk_bin(O_AND, x, kint(t, (int64_t)m)) : x;
		r = mk_bin(N.op[n], lhs, kint(t, (int64_t)c2));
		snprintf(what, sizeof what, "x * %llu %s %llu as %sx %s %llu",
		    (unsigned long long)K, N.op[n] == O_EQ ? "==" : "!=",
		    (unsigned long long)C, s ? "the low bits of " : "",
		    N.op[n] == O_EQ ? "==" : "!=", (unsigned long long)c2);
		cand(B, n, r, 1, what, "inverse checked",
		    "%llu^-1 = 0x%llx mod 2^%d (%llu * 0x%llx = 1 mod 2^%d, "
		    "checked); multiplying by a unit is a bijection of Z/2^%d%s",
		    (unsigned long long)Ko, (unsigned long long)(inv & m), w - s,
		    (unsigned long long)Ko, (unsigned long long)(inv & m), w - s,
		    w - s, s ? " (after the factor 2^s)" : "");
	}
}

/* a x + b over one atom (int64, exact): the atom, or 0 for a constant */
static int
lin_of(nid n, nid *atom, int64_t *a, int64_t *b)
{
	struct bpoly p;
	struct patoms A;
	int k;

	A.n = 0;
	ovf = 0;
	if (!bp_of(n, &p, &A, 0) || ovf || A.n > 1)
		return 0;
	*a = 0;
	*b = 0;
	for (k = 0; k < p.n; k++) {
		if (p.e[k][0] > 1)
			return 0;
		if (p.e[k][0])
			*a = p.c[k];
		else
			*b = p.c[k];
	}
	*atom = A.n ? A.a[0] : 0;
	return 1;
}

static int64_t
pmod(int64_t a, int64_t m)
{
	a %= m;
	return a < 0 ? a + m : a;
}

/* (a X + b) % m == c  ->  X % m' == x0 (the linear congruence theorem) */
static void
ring_congruence(struct bx *B, nid n)
{
	nid md = N.a[n], cn = N.b[n], D, X;
	int64_t m, c, a, b, lo, hi, g, rhs, mp, x0, inv;
	ty_t t;
	uint64_t gg;
	int mod2 = 0;

	if (N.op[md] != O_MOD || !is_const(cn) || !is_const(N.b[md]))
		return;
	D = N.a[md];
	t = N.ty[D];
	m = N.ki[N.b[md]];
	c = N.ki[cn];
	if (m < 2 || !TY_SIZED(t) || t == TY_BOOL)
		return;
	if (!lin_of(D, &X, &a, &b) || X == 0 || a == 0)
		return;
	/* D's machine value is a X + b, or it is unsigned and m | 2^w (the
	 * residue mod m survives wrapping) */
	if (TYK(t) == K_UINT && (m & (m - 1)) == 0 && TYW(t) < 64 &&
	    m <= ((int64_t)1 << TYW(t)))
		mod2 = 1;
	if (!mod2 && !exact(B, D, 0))
		return;
	if (!ival(B, D, &lo, &hi) || lo < 0)
		return;		/* C's % of a negative value is negative */
	if (!ival(B, X, &lo, &hi) || lo < 0)
		return;
	if (c < 0 || c >= m) {
		cand(B, n, mk_const_i(TY_BOOL, N.op[n] == O_NE), 1,
		    "a remainder compared with a value it never takes",
		    "derivation", "0 <= D %% %lld < %lld for D >= 0 (D's interval), "
		    "and %lld is outside", (long long)m, (long long)m, (long long)c);
		return;
	}
	gg = nt_gcd((uint64_t)pmod(a, m), (uint64_t)m);
	g = (int64_t)gg;
	rhs = pmod(c - b, m);
	if (rhs % g) {
		cand(B, n, mk_const_i(TY_BOOL, N.op[n] == O_NE), 1,
		    "a linear congruence without solutions", "derivation",
		    "%lld x + %lld = %lld (mod %lld) needs gcd(%lld, %lld) = %lld to "
		    "divide %lld", (long long)a, (long long)b, (long long)c,
		    (long long)m, (long long)pmod(a, m), (long long)m, (long long)g,
		    (long long)rhs);
		return;
	}
	mp = m / g;
	if (mp == 1) {
		cand(B, n, mk_const_i(TY_BOOL, N.op[n] == O_EQ), 1,
		    "a congruence that always holds", "derivation",
		    "%lld divides %lld x + %lld - %lld for every x", (long long)m,
		    (long long)a, (long long)b, (long long)c);
		return;
	}
	{
		uint64_t gi;
		inv = (int64_t)nt_inv((uint64_t)pmod(a / g, mp), (uint64_t)mp, &gi);
		if (inv == 0 && mp > 1)
			return;
	}
	x0 = (int64_t)((__int128)(rhs / g) * inv % mp);
	/* the certificate: a x0 + b = c (mod m), and a/g is a unit mod m' */
	if (pmod((int64_t)((__int128)a * x0 % m) + b - c, m) != 0)
		return;
	{
		ty_t xt = N.ty[X];
		nid r = mk_bin(N.op[n], mk_bin(O_MOD, X, kint(xt, mp)), kint(xt, x0));
		char what[120];
		snprintf(what, sizeof what, "(%lld x + %lld) %% %lld %s %lld as "
		    "x %% %lld %s %lld", (long long)a, (long long)b, (long long)m,
		    N.op[n] == O_EQ ? "==" : "!=", (long long)c, (long long)mp,
		    N.op[n] == O_EQ ? "==" : "!=", (long long)x0);
		cand(B, n, r, 1, what, "derivation",
		    "%lld*%lld + %lld = %lld = %lld (mod %lld); gcd(%lld, %lld) = %lld "
		    "and %lld^-1 = %lld (mod %lld) (%lld*%lld = 1, checked), so the "
		    "solutions are x = %lld (mod %lld)%s", (long long)a, (long long)x0,
		    (long long)b, (long long)(a * x0 + b), (long long)c, (long long)m,
		    (long long)pmod(a, m), (long long)m, (long long)g,
		    (long long)pmod(a / g, mp), (long long)inv, (long long)mp,
		    (long long)pmod(a / g, mp), (long long)inv, (long long)x0,
		    (long long)mp, mod2 ? "; the residue mod 2^k survives wrapping" :
		    "; no wrap (intervals)");
	}
}

/* X % m1 == r1 && X % m2 == r2  ->  X % lcm == r (CRT) */
static void
ring_crt(struct bx *B, nid n)
{
	nid c1 = cond_of(N.a[n]), c2 = cond_of(N.b[n]), X;
	int64_t m1, r1, m2, r2, lo, hi, tl, th;
	uint64_t r, m;
	int ok;

	if (N.op[c1] != O_EQ || N.op[c2] != O_EQ)
		return;
	if (N.op[N.a[c1]] != O_MOD || N.op[N.a[c2]] != O_MOD ||
	    !is_const(N.b[c1]) || !is_const(N.b[c2]) ||
	    !is_const(N.b[N.a[c1]]) || !is_const(N.b[N.a[c2]]))
		return;
	X = N.a[N.a[c1]];
	if (N.a[N.a[c2]] != X || !ival(B, X, &lo, &hi) || lo < 0)
		return;
	m1 = N.ki[N.b[N.a[c1]]];
	m2 = N.ki[N.b[N.a[c2]]];
	r1 = N.ki[N.b[c1]];
	r2 = N.ki[N.b[c2]];
	if (m1 < 2 || m2 < 2 || m1 > (1 << 30) || m2 > (1 << 30))
		return;
	if (r1 < 0 || r1 >= m1 || r2 < 0 || r2 >= m2)
		return;
	ok = nt_crt((uint64_t)r1, (uint64_t)m1, (uint64_t)r2, (uint64_t)m2, &r,
	    &m);
	if (ok < 0)
		return;
	if (ok == 0) {
		cand(B, n, mk_const_i(TY_BOOL, 0), 1, "two incompatible congruences",
		    "CRT", "x = %lld (mod %lld) and x = %lld (mod %lld) disagree "
		    "mod gcd(%lld, %lld) = %llu", (long long)r1, (long long)m1,
		    (long long)r2, (long long)m2, (long long)m1, (long long)m2,
		    (unsigned long long)nt_gcd((uint64_t)m1, (uint64_t)m2));
		return;
	}
	if (!ty_range(N.ty[X], &tl, &th) || (int64_t)m > th)
		return;
	if ((int64_t)(r % (uint64_t)m1) != r1 || (int64_t)(r % (uint64_t)m2) != r2)
		return;
	{
		nid e = mk_bin(O_EQ, mk_bin(O_MOD, X, kint(N.ty[X], (int64_t)m)),
		    kint(N.ty[X], (int64_t)r));
		char what[120];
		snprintf(what, sizeof what, "x %% %lld == %lld && x %% %lld == %lld "
		    "as x %% %llu == %llu", (long long)m1, (long long)r1,
		    (long long)m2, (long long)r2, (unsigned long long)m,
		    (unsigned long long)r);
		cand(B, n, e, 1, what, "CRT",
		    "%llu %% %lld = %lld, %llu %% %lld = %lld, lcm(%lld, %lld) = %llu: "
		    "the solutions of both are x = %llu (mod %llu) (Chinese remainder "
		    "theorem), x >= 0 by its interval", (unsigned long long)r,
		    (long long)m1, (long long)r1, (unsigned long long)r, (long long)m2,
		    (long long)r2, (long long)m1, (long long)m2, (unsigned long long)m,
		    (unsigned long long)r, (unsigned long long)m);
	}
}

/* the high 64 bits of x m (u64), as nodes: four 32x32 products */
static nid
mulhi64(nid x, uint64_t m)
{
	ty_t u = TY(K_UINT, 64);
	nid lo32 = mk_const_i(u, 0xffffffffLL), s32 = mk_const_i(u, 32);
	nid xl = mk_bin(O_AND, x, lo32), xh = mk_bin(O_SHR, x, s32);
	nid ml = mk_const_i(u, (int64_t)(m & 0xffffffffULL));
	nid mh = mk_const_i(u, (int64_t)(m >> 32));
	nid ll = mk_bin(O_MUL, xl, ml), lh = mk_bin(O_MUL, xl, mh);
	nid hl = mk_bin(O_MUL, xh, ml), hh = mk_bin(O_MUL, xh, mh);
	nid mid = mk_bin(O_ADD, mk_bin(O_ADD, mk_bin(O_SHR, ll, s32),
	    mk_bin(O_AND, lh, lo32)), mk_bin(O_AND, hl, lo32));

	return mk_bin(O_ADD, mk_bin(O_ADD, mk_bin(O_ADD, hh, mk_bin(O_SHR, lh,
	    s32)), mk_bin(O_SHR, hl, s32)), mk_bin(O_SHR, mid, s32));
}

/* x / d for a constant d by ring.c's magic numbers (Granlund-Montgomery) */
static void
ring_magic(struct bx *B, nid n)
{
	nid x = N.a[n], dn = N.b[n], q;
	ty_t t = N.ty[n];
	int w = TYW(t), s, add, sgn = TYK(t) == K_SINT;
	uint64_t d, m, e;
	char what[120];

	if (N.op[n] != O_DIV || !is_const(dn) || !TY_SIZED(t) || t == TY_BOOL)
		return;
	if (N.ki[dn] < 3 || (N.ki[dn] & (N.ki[dn] - 1)) == 0)
		return;
	d = (uint64_t)N.ki[dn];
	if (!((w == 64 && !sgn) || (w == 32 && sgn)))
		return;		/* u32: cfront's own magic; others: not here */
	if (!ring_magic_number(d, w, sgn, &m, &s, &add, &e))
		return;
	if (!sgn) {
		ty_t u = TY(K_UINT, 64);
		nid hi = mulhi64(x, m);
		char text[200];
		if (!add)
			q = mk_bin(O_SHR, hi, mk_const_i(u, s));
		else
			q = mk_bin(O_SHR, mk_bin(O_ADD, hi, mk_bin(O_SHR, mk_bin(O_SUB, x,
			    hi), mk_const_i(u, 1))), mk_const_i(u, s - 1));
		snprintf(what, sizeof what, "intrinsic: x / %llu as the high half of "
		    "x * 0x%llx (mulx)", (unsigned long long)d, (unsigned long long)m);
		if (!cand(B, n, q, 1, what, "derivation", "m*%llu = 2^%d + %llu with "
		    "%llu <= 2^%d (Granlund-Montgomery; ring.c's magic): floor(x m / "
		    "2^%d) = floor(x / %llu) for x < 2^64%s", (unsigned long long)d,
		    64 + s, (unsigned long long)e, (unsigned long long)e, s, 64 + s,
		    (unsigned long long)d, add ? " (m has 65 bits: the add sequence)" :
		    ""))
			return;
		if ((B->t->feat & CF_F(CF_BMI2)) && n == B->e) {
			struct cf_cand *c = &B->cs->c[B->cs->n - 1];
			if (!add)
				snprintf(text, sizeof text, "{ unsigned long long h; "
				    "(void)_mulx_u64($0, 0x%llxull, &h); return h >> %d; }",
				    (unsigned long long)m, s);
			else
				snprintf(text, sizeof text, "{ unsigned long long h; "
				    "(void)_mulx_u64($0, 0x%llxull, &h); return (h + (($0 - h)"
				    " >> 1)) >> %d; }", (unsigned long long)m, s - 1);
			c->text = strdup(text);
			c->feat = CF_F(CF_BMI2);
			c->fixcost = 1;
			c->cost = B->t->c_mul + 1 + 2 * B->t->c_alu * add;
			if (N.op[x] != O_VAR) {
				free((char *)c->text);
				c->text = NULL;
				c->fixcost = 0;
			}
		}
	} else {
		/* i32: the product in 64 bits (|x| m < 2^31 2^32), then + (x < 0) */
		ty_t s64 = TY(K_SINT, 64);
		nid p = mk_bin(O_SHR, mk_bin(O_MUL, mk_cast(s64, x), mk_const_i(s64,
		    (int64_t)m)), mk_const_i(s64, 31 + s));
		q = mk_cast(t, mk_bin(O_ADD, p, mk_cast(s64, mk_bin(O_LT, x,
		    mk_const_i(t, 0)))));
		snprintf(what, sizeof what, "x / %llu (signed) by a multiply and a "
		    "shift", (unsigned long long)d);
		cand(B, n, q, 1, what, "derivation", "m*%llu = 2^%d + %llu with "
		    "%llu <= 2^%d (Granlund-Montgomery; ring.c's magic): floor(x m / "
		    "2^%d) is floor(x / %llu), one less than C's quotient for x < 0 "
		    "(e > 0)", (unsigned long long)d, 31 + s, (unsigned long long)e,
		    (unsigned long long)e, s, 31 + s, (unsigned long long)d);
	}
}

static void
eng_ring(struct bx *B)
{
	uint32_t i;

	for (i = 0; i < B->nr && eng_left() > 0; i++) {
		nid n = B->rn[i];
		switch (N.op[n]) {
		case O_EQ:
		case O_NE:
			ring_inverse(B, n);
			ring_congruence(B, n);
			break;
		case O_LAND:
			ring_crt(B, n);
			break;
		case O_DIV:
			ring_magic(B, n);
			break;
		}
	}
}

/* ================================================================
 * dio: congruence obstructions, two squares, elliptic curves
 * ================================================================ */

/* the moduli tried: powers of two (they survive wrapping), then small
 * primes and prime squares (only on exact values) */
static const int64_t qs_two[] = {4, 8, 16, 32, 64, 3, 9, 5, 25, 7, 49, 11,
	13, 27, 0};

static void
dio_two_squares(struct bx *B, nid n, const struct bpoly *p,
    const struct patoms *A)
{
	/* p = u^2 + v^2 - C (the comparison moved to one side) */
	int k, sq = 0;
	int64_t C = 0;
	struct nt_fac f;

	for (k = 0; k < p->n; k++) {
		int j, deg = 0, one = -1;
		for (j = 0; j < PA; j++) {
			deg += p->e[k][j];
			if (p->e[k][j])
				one = j;
		}
		if (deg == 0)
			C = -p->c[k];
		else if (deg == 2 && p->e[k][one] == 2 && p->c[k] == 1)
			sq++;
		else
			return;
	}
	if (sq != 2 || A->n != 2)
		return;
	if (C < 0) {
		cand(B, n, mk_const_i(TY_BOOL, N.op[n] == O_NE), 1,
		    "a sum of two squares compared with a negative number",
		    "derivation", "u^2 + v^2 >= 0 > %lld (no wrap where defined)",
		    (long long)C);
		return;
	}
	if (C == 0)
		return;
	nt_factor((uint64_t)C, &f);
	for (k = 0; k < f.k; k++)
		if (f.p[k] % 4 == 3 && (f.e[k] & 1)) {
			struct sbuf fs = {0};
			int j;
			uint64_t prod = 1;
			for (j = 0; j < f.k; j++) {
				int e2;
				sb_printf(&fs, "%s%llu", j ? "*" : "",
				    (unsigned long long)f.p[j]);
				if (f.e[j] > 1)
					sb_printf(&fs, "^%d", f.e[j]);
				for (e2 = 0; e2 < f.e[j]; e2++)
					prod *= f.p[j];
			}
			if (prod != (uint64_t)C) {
				free(fs.p);
				return;
			}
			cand(B, n, mk_const_i(TY_BOOL, N.op[n] == O_NE), 1,
			    "a sum of two squares that cannot be", "Diophantine certificate",
			    "%lld = %s (multiplied back); %llu = 3 (mod 4) divides it to "
			    "the odd power %d, so %lld is not a sum of two squares "
			    "(Fermat-Euler: descent in Z[i]); the values are integers "
			    "where defined (no wrap: %s)", (long long)C, fs.p,
			    (unsigned long long)f.p[k], f.e[k], (long long)C,
			    B->wrapv ? "intervals" : "signed overflow is C UB");
			free(fs.p);
			return;
		}
}

/* y^2 = x^3 + a x + b: integer points from the torsion of a rank-0 curve */
static void
dio_elliptic(struct bx *B, nid n, nid side, const struct bpoly *p,
    const struct patoms *A)
{
	int k, ix = -1, iy = -1;
	int64_t a = 0, b = 0;
	struct ec E;
	struct ecq_mw M;
	static struct ecf QF;
	static int qinit;
	fe ai[5];
	struct sbuf pts = {0};
	int npts = 0, nok = 0, j;
	nid sx = 0, sy = 0;
	int64_t yne = INT64_MIN;	/* a side condition y != c */
	zmark zm;

	if (A->n != 2)
		return;
	/* find the cubic atom and the square atom: p = y^2 - x^3 - a x - b */
	for (k = 0; k < p->n; k++) {
		int j2;
		for (j2 = 0; j2 < 2; j2++) {
			if (p->e[k][j2] == 3 && p->e[k][1 - j2] == 0 && p->c[k] == -1)
				ix = j2;
			if (p->e[k][j2] == 2 && p->e[k][1 - j2] == 0 && p->c[k] == 1)
				iy = j2;
		}
	}
	if (ix < 0 || iy < 0 || ix == iy) {
		/* the comparison may be the other way round: x^3 + ... - y^2 */
		return;
	}
	for (k = 0; k < p->n; k++) {
		int ex = p->e[k][ix], ey = p->e[k][iy];
		if ((ex == 3 && ey == 0) || (ey == 2 && ex == 0))
			continue;
		if (ex == 1 && ey == 0)
			a = -p->c[k];
		else if (ex == 0 && ey == 0)
			b = -p->c[k];
		else
			return;
	}
	if (a > 1000000 || a < -1000000 || b > 1000000 || b < -1000000)
		return;
	if (4 * a * a * a + 27 * b * b == 0)
		return;		/* singular */
	/* a side condition: the comparison is in an && with y != c */
	if (side) {
		nid c = cond_of(side);
		if (N.op[c] == O_NE && is_const(N.b[c]) &&
		    strip_keep(B, N.a[c]) == strip_keep(B, A->a[iy]))
			yne = N.ki[N.b[c]];
		else
			return;
	}
	if (!qinit++)
		ecf_q(&QF);
	zm = z_mark();
	ai[0] = fe_from_z(z_from_i64(0));
	ai[1] = fe_from_z(z_from_i64(0));
	ai[2] = fe_from_z(z_from_i64(0));
	ai[3] = fe_from_z(z_from_i64(a));
	ai[4] = fe_from_z(z_from_i64(b));
	ec_init(&E, &QF, ai);
	memset(&M, 0, sizeof M);
	ecq_mw_compute(&E, &M, 200);
	if (M.rank_hi != 0 || M.rank_how != 1) {
		ecq_mw_free(&M);
		z_release(zm);
		return;
	}
	/* E(Q) = torsion: its affine points on the given model */
	{
		struct ecq_tors T;
		ecq_torsion(&E, &T);
		for (j = 0; j < T.n; j++) {
			const char *xs, *ys;
			if (T.pts[j].inf)
				continue;
			xs = fe_str(&QF, &T.pts[j].x);
			ys = fe_str(&QF, &T.pts[j].y);
			sb_printf(&pts, "%s(%s, %s)", npts++ ? ", " : "", xs, ys);
			if (strchr(xs, '/') || strchr(ys, '/'))
				continue;	/* not integral */
			if (yne != INT64_MIN && atoll(ys) == yne)
				continue;
			nok++;
			sx = mk_const_i(N.ty[A->a[ix]], atoll(xs));
			sy = mk_const_i(N.ty[A->a[iy]], atoll(ys));
		}
		(void)sx;
		(void)sy;
	}
	if (nok == 0) {
		{
			/* the whole conjunction (or the equation) never holds */
			nid r = mk_const_i(TY_BOOL, 0);
			nid at = n;
			if (side) {
				uint32_t i;
				for (i = 0; i < B->nr; i++)
					if (N.op[B->rn[i]] == O_LAND &&
					    ((cond_of(N.a[B->rn[i]]) == n && N.b[B->rn[i]] == side) ||
					    (cond_of(N.b[B->rn[i]]) == n &&
					    N.a[B->rn[i]] == side)))
						at = B->rn[i];
				if (at == n)
					goto done;
			} else if (N.op[n] == O_NE)
				r = mk_const_i(TY_BOOL, 1);
			char eq[96];
			snprintf(eq, sizeof eq, "x^3");
			if (a)
				snprintf(eq + strlen(eq), sizeof eq - strlen(eq), " %c %lldx",
				    a < 0 ? '-' : '+', (long long)(a < 0 ? -a : a));
			if (b)
				snprintf(eq + strlen(eq), sizeof eq - strlen(eq), " %c %lld",
				    b < 0 ? '-' : '+', (long long)(b < 0 ? -b : b));
			cand(B, at, r, 1, "an elliptic curve without such integer "
			    "points", "Diophantine certificate", "E: y^2 = %s has rank 0 (2-isogeny descent, ecq.c) and torsion of "
			    "order %d: E(Q) = {O%s%s}%s; the values are integers where "
			    "defined", eq, M.T.n, npts ? ", " : "",
			    pts.p ? pts.p : "", yne != INT64_MIN ? ", each with y = 0" :
			    ", none integral");
		}
	}
done:
	free(pts.p);
	ecq_mw_free(&M);
	z_release(zm);
}

static void
eng_dio(struct bx *B)
{
	uint32_t i;

	for (i = 0; i < B->nr && eng_left() > 0; i++) {
		nid n = B->rn[i], L, R, side = 0;
		struct bpoly p, pr;
		struct patoms A;
		int64_t q = 0, target = 0, m = 0;
		int qi, k, ex, wrapping;
		ty_t t;

		if (N.op[n] != O_EQ && N.op[n] != O_NE)
			continue;
		L = N.a[n];
		R = N.b[n];
		t = N.ty[L];
		if (!TY_SIZED(t) || t == TY_BOOL)
			continue;
		A.n = 0;
		ovf = 0;
		if (N.op[L] == O_MOD && is_const(N.b[L]) && is_const(R)) {
			/* P % m == c: P (mod m) */
			m = N.ki[N.b[L]];
			target = N.ki[R];
			if (m < 2 || m > 4096 || !bp_of(N.a[L], &p, &A, 0) || ovf)
				continue;
			if (A.n == 0)
				continue;
		} else {
			if (!bp_of(L, &p, &A, 0) || !bp_of(R, &pr, &A, 0) || ovf)
				continue;
			for (k = 0; k < pr.n; k++)
				bp_addterm(&p, -pr.c[k], pr.e[k]);
			if (ovf || A.n == 0)
				continue;
			{
				/* only nonlinear equations are of interest here */
				int nl = 0, j;
				for (k = 0; k < p.n; k++) {
					int deg = 0;
					for (j = 0; j < PA; j++)
						deg += p.e[k][j];
					nl |= deg > 1;
				}
				if (!nl)
					continue;
			}
		}
		ex = m ? exact(B, N.a[L], 0) : exact(B, L, 0) && exact(B, R, 0);
		wrapping = TYK(t) == K_UINT || B->wrapv;
		/* congruence obstructions */
		for (qi = 0; qs_two[qi]; qi++) {
			uint8_t hit[4096];
			int64_t want;
			q = qs_two[qi];
			if (m) {
				/* P % m == c: only q = m, and P >= 0 */
				int64_t lo, hi;
				if (q != 4)
					break;
				q = m;
				if (!ival(B, N.a[L], &lo, &hi) || lo < 0)
					break;
				if (!ex && !(TYK(t) == K_UINT && (m & (m - 1)) == 0))
					break;
				if (target < 0 || target >= m) {
					break;	/* ring_congruence says so */
				}
				want = target;
			} else {
				/* P == 0 over Z (exact), or mod 2^w (q | 2^w) */
				if (!ex && !((q & (q - 1)) == 0 && wrapping &&
				    q <= ((int64_t)1 << (TYW(t) < 62 ? TYW(t) : 62))))
					continue;
				want = 0;
			}
			if (!residues(&p, A.n, q, hit))
				continue;
			if (!hit[want]) {
				struct sbuf b = {0}, ps = {0};
				const char *nm[PA];
				for (k = 0; k < A.n; k++)
					nm[k] = atom_name(B, A.a[k], k);
				bp_str(&ps, &p, &A, nm);
				residue_list(&b, hit, q);
				cand(B, n, mk_const_i(TY_BOOL, N.op[n] == O_NE), 1, m ?
				    "a remainder the polynomial never leaves" :
				    "an equation without solutions mod q",
				    "Diophantine certificate", "%s takes only the residues %s "
				    "mod %lld (every residue of its %d input%s tried), never "
				    "%lld%s", ps.p, b.p, (long long)q, A.n, A.n > 1 ? "s" : "",
				    (long long)want, ex ? "; the values are exact integers "
				    "where defined" : "; q divides 2^w, so wrapping keeps the "
				    "residue");
				free(b.p);
				free(ps.p);
				goto next;
			}
			if (m)
				break;
		}
		if (m || !ex)
			continue;
		dio_two_squares(B, n, &p, &A);
		/* an elliptic curve, maybe with a side condition y != c */
		{
			uint32_t j;
			for (j = 0; j < B->nr; j++)
				if (N.op[B->rn[j]] == O_LAND) {
					if (cond_of(N.b[B->rn[j]]) == n)
						side = N.a[B->rn[j]];
					else if (cond_of(N.a[B->rn[j]]) == n)
						side = N.b[B->rn[j]];
				}
			if (N.op[n] == O_EQ)
				dio_elliptic(B, n, side, &p, &A);
		}
next:
		;
	}
}

/* ================================================================
 * alg: the polynomial structure, factored, collected, cancelled
 * ================================================================ */

/* every ring operation of n is at least w bits wide: n mod 2^w is its
 * polynomial mod 2^w (reduction commutes with + - *) */
static int
ring_closed(nid n, int w, int depth)
{
	int op = N.op[n];

	if (depth > 40)
		return 0;
	switch (op) {
	case O_ADD:
	case O_SUB:
	case O_MUL:
	case O_NEG:
	case O_SHL:
	case O_CAST:
		if (!TY_SIZED(N.ty[n]) || TYW(N.ty[n]) < w || N.ty[n] == TY_BOOL)
			return 0;
		if (op == O_CAST)
			return !TY_SIZED(N.ty[N.a[n]]) || TYW(N.ty[N.a[n]]) >= w ||
			    N.op[N.a[n]] == O_VAR || N.op[N.a[n]] == O_CONST ?
			    (TY_SIZED(N.ty[N.a[n]]) ? ring_closed(N.a[n], w, depth + 1) :
			    0) : 1;
		if (!ring_closed(N.a[n], w, depth + 1))
			return 0;
		return op == O_NEG || op == O_SHL || ring_closed(N.b[n], w,
		    depth + 1);
	}
	return 1;		/* an atom */
}

/* the polynomial as a machine node in the unsigned type U */
static nid
bp_node(const struct bpoly *p, const struct patoms *A, ty_t U)
{
	nid r = 0, at[PA];
	int k, j, i;

	for (j = 0; j < A->n; j++)
		at[j] = mk_cast(U, A->a[j]);
	for (k = 0; k < p->n; k++) {
		nid m = 0;
		int64_t c = p->c[k];
		int neg = 0;
		for (j = 0; j < A->n; j++)
			for (i = 0; i < p->e[k][j]; i++)
				m = m ? mk_bin(O_MUL, m, at[j]) : at[j];
		if (m && c < 0 && c != INT64_MIN && r) {
			neg = 1;
			c = -c;
		}
		if (!m)
			m = mk_const_i(U, int_norm(U, c));
		else if (c != 1)
			m = mk_bin(O_MUL, mk_const_i(U, int_norm(U, c)), m);
		r = !r ? m : neg ? mk_bin(O_SUB, r, m) : mk_bin(O_ADD, r, m);
	}
	return r ? r : mk_const_i(U, 0);
}

/* Horner in the one atom: ((c_d x + c_{d-1}) x + ...) + c_0 */
static nid
bp_horner(const struct bpoly *p, const struct patoms *A, ty_t U)
{
	int64_t c[13] = {0};
	int k, d = 0;
	nid x, r;

	if (A->n != 1)
		return 0;
	for (k = 0; k < p->n; k++) {
		c[p->e[k][0]] = p->c[k];
		if (p->e[k][0] > d)
			d = p->e[k][0];
	}
	if (d < 3)
		return 0;
	x = mk_cast(U, A->a[0]);
	r = mk_const_i(U, int_norm(U, c[d]));
	for (k = d - 1; k >= 0; k--) {
		r = mk_bin(O_MUL, r, x);
		if (c[k])
			r = mk_bin(O_ADD, r, mk_const_i(U, int_norm(U, c[k])));
	}
	return r;
}

/* bpoly -> alg's mp (atoms k = alg atoms k) */
static struct mp
mp_from_bp(const struct bpoly *p, int na)
{
	struct mp r = mp_zero();
	int k, j, i;

	for (k = 0; k < p->n; k++) {
		struct mp m = mp_const(qi(p->c[k]));
		for (j = 0; j < na; j++)
			for (i = 0; i < p->e[k][j]; i++)
				m = mp_mul(m, mp_atom(j));
		r = mp_add(r, m);
	}
	return r;
}

/* alg's mp with integer coefficients -> bpoly; 0 if not integral */
static int
bp_from_mp(struct mp a, struct bpoly *p)
{
	int k, j;

	p->n = 0;
	for (k = 0; k < a.n; k++) {
		uint8_t e[PA] = {0};
		if (a.t[k].c.d != 1)
			return 0;
		for (j = 0; j < AV; j++) {
			if (j >= PA && a.t[k].e[j])
				return 0;
			if (j < PA)
				e[j] = a.t[k].e[j];
		}
		bp_addterm(p, a.t[k].c.n, e);
	}
	return !ovf;
}

static void
alg_poly(struct bx *B, nid n)
{
	struct bpoly p, f;
	struct patoms A;
	ty_t t = N.ty[n], U;
	int w = TYW(t), k, nf, mult[8], partial = 0, deg = 0, j;
	struct mp a, fac[8];
	aq c;
	size_t mark;
	nid r;
	jmp_buf saved;
	const char *nm[PA];
	struct sbuf fs = {0}, ps = {0};

	if (!TY_SIZED(t) || t == TY_BOOL || !ring_closed(n, w, 0))
		return;
	A.n = 0;
	ovf = 0;
	if (!bp_of(n, &p, &A, 0) || ovf || A.n == 0 || p.n < 2)
		return;
	for (k = 0; k < p.n; k++) {
		int d = 0;
		for (j = 0; j < PA; j++)
			d += p.e[k][j];
		if (d > deg)
			deg = d;
	}
	if (deg < 2)
		return;
	U = TY(K_UINT, w);
	for (k = 0; k < A.n; k++)
		nm[k] = atom_name(B, A.a[k], k);
	bp_str(&ps, &p, &A, nm);
	/* the expanded form and Horner: identities of Z[x], so of Z/2^w */
	cand(B, n, bp_node(&p, &A, U), 1, "the polynomial expanded",
	    "derivation", "%s over Z; only + - * and constants: the same mod "
	    "2^%d", ps.p, w);
	if ((r = bp_horner(&p, &A, U)) != 0)
		cand(B, n, r, 1, "the polynomial by Horner's rule", "derivation",
		    "%s collected in %s (Horner) over Z; the same mod 2^%d", ps.p,
		    nm[0], w);
	/* factored by alg.c */
	mark = ar_mark();
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0) {
		atoms_reset();
		for (k = 0; k < A.n; k++)
			atom_of(A.a[k]);
		a = mp_from_bp(&p, A.n);
		nf = mp_factor(a, &c, fac, mult, 8, &partial);
		if (nf >= 2 || (nf == 1 && mult[0] > 1)) {
			nid prod = 0;
			int ok = c.d == 1;
			for (k = 0; k < nf && ok; k++) {
				nid fn;
				int m2;
				ovf = 0;
				if (!bp_from_mp(fac[k], &f)) {
					ok = 0;
					break;
				}
				fn = bp_node(&f, &A, U);
				sb_printf(&fs, "%s(%s)", k ? "*" : "", mp_str(fac[k]));
				if (mult[k] > 1)
					sb_printf(&fs, "^%d", mult[k]);
				for (m2 = 0; m2 < mult[k]; m2++)
					prod = prod ? mk_bin(O_MUL, prod, fn) : fn;
			}
			if (ok && prod) {
				if (c.n != 1)
					prod = mk_bin(O_MUL, mk_const_i(U, int_norm(U, c.n)),
					    prod);
				cand(B, n, prod, 1, "the polynomial factored", "derivation",
				    "%s = %s%s over Z (alg.c's factor, multiplied back); "
				    "only + - * and constants: the same mod 2^%d", ps.p,
				    c.n != 1 ? "c*" : "", fs.p, w);
			}
		}
	}
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	ar_release(mark);
	free(fs.p);
	free(ps.p);
}

/* P / Q with Q | P over Z and no wrap: the quotient polynomial */
static void
alg_cancel(struct bx *B, nid n)
{
	struct bpoly p, q, r;
	struct patoms A;
	ty_t t = N.ty[n];
	struct mp a, b, quo;
	size_t mark;
	jmp_buf saved;
	int ok = 0, k;
	const char *nm[PA];
	struct sbuf ps = {0}, qs = {0}, rs = {0};

	if (!TY_SIZED(t) || t == TY_BOOL || !exact(B, N.a[n], 0) ||
	    !exact(B, N.b[n], 0))
		return;
	A.n = 0;
	ovf = 0;
	if (!bp_of(N.a[n], &p, &A, 0) || !bp_of(N.b[n], &q, &A, 0) || ovf ||
	    A.n == 0 || q.n == 0)
		return;
	{
		int nonconst = 0, j;
		for (k = 0; k < q.n; k++)
			for (j = 0; j < PA; j++)
				nonconst |= q.e[k][j] != 0;
		if (!nonconst)
			return;
	}
	mark = ar_mark();
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0) {
		atoms_reset();
		for (k = 0; k < A.n; k++)
			atom_of(A.a[k]);
		a = mp_from_bp(&p, A.n);
		b = mp_from_bp(&q, A.n);
		ovf = 0;
		if (mp_divexact(a, b, &quo) && bp_from_mp(quo, &r) &&
		    mp_eq(mp_mul(quo, b), a))
			ok = 1;
	}
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	ar_release(mark);
	if (!ok)
		return;
	for (k = 0; k < A.n; k++)
		nm[k] = atom_name(B, A.a[k], k);
	bp_str(&ps, &p, &A, nm);
	bp_str(&qs, &q, &A, nm);
	bp_str(&rs, &r, &A, nm);
	cand(B, n, bp_node(&r, &A, TY(K_UINT, TYW(t))), 1, N.op[n] == O_DIV ?
	    "a quotient of polynomials cancelled" : "a remainder of a divisible "
	    "polynomial", "derivation", "%s = (%s)(%s) over Z (exact division, "
	    "multiplied back); neither side wraps (%s), so C's truncating %s "
	    "is %s where %s != 0 (0 is UB in the original)", ps.p, qs.p, rs.p,
	    B->wrapv ? "intervals" : "intervals or C UB", N.op[n] == O_DIV ? "/" :
	    "%", N.op[n] == O_DIV ? rs.p : "0", qs.p);
	free(ps.p);
	free(qs.p);
	free(rs.p);
}

static void
eng_alg(struct bx *B)
{
	uint32_t i;
	int tried = 0;

	for (i = B->nr; i-- > 0 && tried < 12 && eng_left() > 0;) {
		nid n = B->rn[i];
		switch (N.op[n]) {
		case O_ADD:
		case O_SUB:
		case O_MUL:
			alg_poly(B, n);
			tried++;
			break;
		case O_DIV:
		case O_MOD:
			if (N.op[n] == O_DIV)
				alg_cancel(B, n);
			tried++;
			break;
		}
	}
}

/* ================================================================
 * bound: comparisons and clamps decided by bound.c's proved bounds
 * ================================================================ */

static void
eng_bound(struct bx *B)
{
	uint32_t i;
	int tried = 0;

	for (i = 0; i < B->nr && tried < 8 && eng_left() > 0; i++) {
		nid n = B->rn[i], X, K;
		int op = N.op[n], flip = 0, res = -1;
		struct func g;
		struct pbound pb;
		int64_t k, lo, hi;

		if (op != O_LT && op != O_LE && op != O_GT && op != O_GE &&
		    op != O_MIN && op != O_MAX)
			continue;
		X = N.a[n];
		K = N.b[n];
		if (is_const(X)) {
			X = N.b[n];
			K = N.a[n];
			flip = 1;
		}
		if (!is_const(K) || is_const(X) || !TY_SIZED(N.ty[X]) ||
		    N.ty[X] == TY_BOOL)
			continue;
		/* intervals already decide it: cfront's own drops do that */
		if (ival(B, X, &lo, &hi) && (hi < N.ki[K] || lo > N.ki[K]))
			continue;
		k = N.ki[K];
		g = B->f;
		g.body = X;
		memset(&pb, 0, sizeof pb);
		tried++;
		note_mute++;
		{
			jmp_buf saved;
			int ok = 0;
			memcpy(saved, fail_jmp, sizeof saved);
			if (setjmp(fail_jmp) == 0)
				ok = prove_bounds(&g, &pb);
			memcpy(fail_jmp, saved, sizeof saved);
			note_mute--;
			notes_drop();
			if (!ok) {
				pbound_free(&pb);
				continue;
			}
		}
		if (flip)
			op = op == O_LT ? O_GT : op == O_LE ? O_GE : op == O_GT ? O_LT :
			    op == O_GE ? O_LE : op;
		/* res: 1 always true / X is the min, 0 always false / K is */
		if (pb.have_hi && ((op == O_LT && pb.hi < k) || (op == O_LE &&
		    pb.hi <= k) || (op == O_MIN && pb.hi <= k)))
			res = 1;
		else if (pb.have_lo && ((op == O_GT && pb.lo > k) || (op == O_GE &&
		    pb.lo >= k) || (op == O_MAX && pb.lo >= k)))
			res = 1;
		else if (pb.have_lo && ((op == O_LT && pb.lo >= k) || (op == O_LE &&
		    pb.lo > k)))
			res = 0;
		else if (pb.have_hi && ((op == O_GT && pb.hi <= k) || (op == O_GE &&
		    pb.hi < k)))
			res = 0;
		if (res >= 0) {
			nid r = op == O_MIN || op == O_MAX ? X :
			    mk_const_i(N.ty[n], res);
			char why[200];
			const char *w = res == 1 && (op == O_LT || op == O_LE ||
			    op == O_MIN) ? sb_str(&pb.why_hi) : sb_str(&pb.why_lo);
			snprintf(why, sizeof why, "%s", w && *w ? w : "bound.c");
			for (k = 0; why[k]; k++)
				if (why[k] == '\n')
					why[k] = ';';
			cand(B, n, r, 1, op == O_MIN || op == O_MAX ? "a clamp that never "
			    "bites (proved bound)" : "a comparison decided by a proved "
			    "bound", "lemma", "the value is in [%g, %g] (bound.c: %.150s)",
			    pb.have_lo ? pb.lo : -INFINITY, pb.have_hi ? pb.hi : INFINITY,
			    why);
		}
		pbound_free(&pb);
	}
}

/* ================================================================
 * gf2: GF(2)-affine maps: gf2p8affine, pext/pdep, CRC by pclmul
 * ================================================================ */

/* memos for the DAG walks below (shared subterms: a CRC's rounds) */
static uint64_t *m1memo;
static uint8_t *m1have, *g2memo;
static uint32_t g2n;

static void
g2_memo_reset(void)
{
	free(m1memo);
	free(m1have);
	free(g2memo);
	g2n = N.n + 1;
	m1memo = xcalloc(g2n, sizeof *m1memo);
	m1have = xcalloc(g2n, 1);
	g2memo = xcalloc(g2n, 1);
}

static uint64_t may1_(nid n, int depth);

/* bits that may be 1 (sound) */
static uint64_t
may1(nid n, int depth)
{
	uint64_t r;

	if (n < g2n && m1have && m1have[n])
		return m1memo[n];
	r = may1_(n, depth);
	if (n < g2n && m1have) {
		m1have[n] = 1;
		m1memo[n] = r;
	}
	return r;
}

static uint64_t
may1_(nid n, int depth)
{
	ty_t t = N.ty[n];
	uint64_t m = TY_SIZED(t) ? wmask(TYW(t)) : ~0ULL, a;
	nid x = N.a[n], y = N.b[n];

	if (depth > 2000)
		return m;
	switch (N.op[n]) {
	case O_CONST:
		return (uint64_t)N.ki[n] & m;
	case O_AND:
		return may1(x, depth + 1) & may1(y, depth + 1);
	case O_OR:
	case O_XOR:
		return may1(x, depth + 1) | may1(y, depth + 1);
	case O_SHL:
		if (is_const(y) && N.ki[y] >= 0 && N.ki[y] < 64)
			return (may1(x, depth + 1) << N.ki[y]) & m;
		return m;
	case O_SHR:
		if (is_const(y) && N.ki[y] >= 0 && N.ki[y] < 64) {
			a = may1(x, depth + 1);
			if (TYK(N.ty[x]) == K_SINT && (a >> (TYW(N.ty[x]) - 1) & 1))
				return m;
			return (a >> N.ki[y]) & m;
		}
		return m;
	case O_CAST:
		if (!TY_SIZED(N.ty[x]))
			return m;
		a = may1(x, depth + 1);
		if (TYK(N.ty[x]) == K_SINT && TYW(t) > TYW(N.ty[x]) &&
		    (a >> (TYW(N.ty[x]) - 1) & 1))
			return m;
		return a & m;
	}
	return m;
}

static int g2_struct_(nid n, int depth);

/* affine over GF(2) in the leaves, by the structure of its operations */
static int
g2_struct(nid n, int depth)
{
	int r;

	if (n < g2n && g2memo && g2memo[n])
		return g2memo[n] == 1;
	r = g2_struct_(n, depth);
	if (n < g2n && g2memo)
		g2memo[n] = r ? 1 : 2;
	return r;
}

static int
g2_struct_(nid n, int depth)
{
	nid x = N.a[n], y = N.b[n];

	if (depth > 2000 || (!TY_SIZED(N.ty[n])))
		return 0;
	switch (N.op[n]) {
	case O_CONST:
	case O_VAR:
		return 1;
	case O_XOR:
		return g2_struct(x, depth + 1) && g2_struct(y, depth + 1);
	case O_OR:
	case O_ADD:
		return g2_struct(x, depth + 1) && g2_struct(y, depth + 1) &&
		    !(may1(x, 0) & may1(y, 0));
	case O_AND:
		return (is_const(y) && g2_struct(x, depth + 1)) || (is_const(x) &&
		    g2_struct(y, depth + 1));
	case O_NOT:
		return g2_struct(x, depth + 1);
	case O_SHL:
	case O_SHR:
		return is_const(y) && N.ki[y] >= 0 && N.ki[y] < TYW(N.ty[n]) &&
		    g2_struct(x, depth + 1);
	case O_CAST:
		return TY_SIZED(N.ty[x]) && g2_struct(x, depth + 1);
	case O_NEG:
		/* -(b) for b in {0, 1}: b copied to every bit */
		return may1(x, 0) <= 1 && g2_struct(x, depth + 1);
	case O_MUL:
		if (is_const(y) && may1(x, 0) <= 1)
			return g2_struct(x, depth + 1);
		if (is_const(x) && may1(y, 0) <= 1)
			return g2_struct(y, depth + 1);
		if (is_const(y) && N.ki[y] > 0 && (N.ki[y] & (N.ki[y] - 1)) == 0)
			return g2_struct(x, depth + 1);
		return 0;
	}
	return 0;
}

struct g2 {
	int nin, nout;		/* bits */
	int slot[MAXP], off[MAXP], wid[MAXP], ns;
	uint64_t col[64];	/* f(e_j) ^ f(0) */
	uint64_t c;		/* f(0) */
	char how[160];		/* the linearity proof */
};

static int
g2_eval(struct bx *B, const struct g2 *G, uint64_t bits, uint64_t *out)
{
	int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
	double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
	int k, ok;

	for (k = 0; k < G->ns; k++) {
		int s = G->slot[k];
		in_i[(size_t)s * CH] = int_norm(ct_to_ucas(B->v->leaf[s].cty),
		    (int64_t)((bits >> G->off[k]) & wmask(G->wid[k])));
	}
	plan_run(&B->p, 1, in_i, in_r);
	ok = !(COLF(&B->p, B->p.root[0])[0] & F_UNDEF);
	*out = (uint64_t)COLI(&B->p, B->p.root[0])[0] & wmask(G->nout);
	free(in_i);
	free(in_r);
	return ok;
}

/* the matrix of e and a proof that e is affine: 0 if not */
static int
g2_matrix(struct bx *B, struct g2 *G)
{
	uint64_t s = node_slots(B->e), y;
	int k, j;
	ty_t rt = N.ty[B->e];

	memset(G, 0, sizeof *G);
	if (!TY_SIZED(rt) || rt == TY_BOOL)
		return 0;
	G->nout = TYW(rt);
	for (k = 0; k < B->v->nleaf && k < MAXP; k++) {
		ty_t lt = ct_to_ucas(B->v->leaf[k].cty);
		if (!(s >> k & 1))
			continue;
		if (!TY_SIZED(lt) || lt == TY_BOOL || B->v->leaf[k].ranged)
			return 0;
		G->slot[G->ns] = k;
		G->off[G->ns] = G->nin;
		G->wid[G->ns] = TYW(lt);
		G->nin += TYW(lt);
		G->ns++;
	}
	/* a domain condition: only the structural proof (it holds for the
	 * expression everywhere; cfront proves the form on the domain) */
	if (G->nin == 0 || G->nin > 64 || (B->v->where && !g2_struct(B->e, 0)))
		return 0;
	if (!g2_eval(B, G, 0, &G->c))
		return 0;
	for (j = 0; j < G->nin; j++) {
		if (!g2_eval(B, G, 1ULL << j, &y))
			return 0;
		G->col[j] = y ^ G->c;
	}
	if (g2_struct(B->e, 0)) {
		snprintf(G->how, sizeof G->how, "affine by structure (xor, "
		    "shifts and masks by constants, or/add of bit-disjoint "
		    "operands, conversions)");
		return 1;
	}
	if (G->nin <= 8) {
		/* f(a ^ b) = f(a) ^ f(b) ^ f(0) for every pair */
		uint64_t a, b, fa[256], fb;
		for (a = 0; a < (1u << G->nin); a++)
			if (!g2_eval(B, G, a, &fa[a]))
				return 0;
		for (a = 0; a < (1u << G->nin); a++)
			for (b = 0; b < (1u << G->nin); b++) {
				fb = fa[a ^ b];
				if (fb != (fa[a] ^ fa[b] ^ G->c))
					return 0;
			}
		snprintf(G->how, sizeof G->how, "f(a ^ b) = f(a) ^ f(b) ^ f(0) for "
		    "all %u pairs (exhaustive)", (1u << G->nin) * (1u << G->nin));
		return 1;
	}
	if (G->nin <= 16) {
		uint64_t x;
		for (x = 0; x < (1u << G->nin); x++) {
			uint64_t v = G->c;
			for (j = 0; j < G->nin; j++)
				if (x >> j & 1)
					v ^= G->col[j];
			if (!g2_eval(B, G, x, &y) || y != v)
				return 0;
		}
		snprintf(G->how, sizeof G->how, "f(x) = M x ^ f(0) at all %u "
		    "inputs (exhaustive)", 1u << G->nin);
		return 1;
	}
	return 0;
}

/* ---- the catalogue's lanes as nodes ---- */

struct lanectx {
	nid opnd[4][64];	/* operand k's lane j */
};

static nid
lane_cb(void *c, int k, int lane)
{
	struct lanectx *L = c;

	return L->opnd[k][lane & 63];
}

/* lane `lane` of intrinsic name over the operands' lanes */
static nid
intr_lane(const char *name, struct lanectx *L, int lane, int64_t imm)
{
	int id = xi_find(name);
	jmp_buf saved;
	nid r = 0;

	if (id < 0)
		return 0;
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = xi_lane(id, lane, imm, lane_cb, L);
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	return r;
}

/* a bit gather (pext) or scatter (pdep) of one word */
static void
g2_pext(struct bx *B, const struct g2 *G)
{
	int i, j, src[64], k = 0, w = G->nout, isext = 1, isdep = 1;
	uint64_t mask = 0, dmask = 0;
	nid x, sem;
	struct lanectx L;
	const char *nm;
	char text[200], what[120];
	struct cf_cand *c;

	if (G->ns != 1 || G->nin != w || (w != 32 && w != 64) || G->c ||
	    !(B->t->feat & CF_F(CF_BMI2)))
		return;
	for (i = 0; i < w; i++)
		src[i] = -1;
	for (j = 0; j < G->nin; j++) {
		uint64_t col = G->col[j];
		if (col & (col - 1))
			return;		/* an input bit in two places */
		if (col)
			src[__builtin_ctzll(col)] = j;
	}
	/* pext: outputs 0..k-1 from increasing inputs, the rest 0 */
	for (i = 0; i < w && src[i] >= 0; i++) {
		if (i && src[i] <= src[i - 1])
			isext = 0;
		mask |= 1ULL << src[i];
	}
	k = i;
	for (; i < w; i++)
		if (src[i] >= 0)
			isext = 0;
	/* pdep: input bits 0..k-1 to increasing outputs */
	{
		int next = 0;
		for (i = 0; i < w; i++)
			if (src[i] >= 0) {
				if (src[i] != next++)
					isdep = 0;
				dmask |= 1ULL << i;
			}
	}
	if (k < 2 && isext)
		isext = 0;
	if (!isext && !isdep)
		return;
	if (isext && (mask == wmask(k) << __builtin_ctzll(mask)))
		return;		/* a shift and a mask */
	if (!isext && (dmask == wmask(__builtin_popcountll(dmask)) <<
	    __builtin_ctzll(dmask)))
		return;
	nm = isext ? (w == 32 ? "_pext_u32" : "_pext_u64") : (w == 32 ?
	    "_pdep_u32" : "_pdep_u64");
	memset(&L, 0, sizeof L);
	x = mk_var((uint32_t)G->slot[0], sym_intern(B->v->leaf[G->slot[0]].text,
	    strlen(B->v->leaf[G->slot[0]].text)), ct_to_ucas(B->v->leaf[
	    G->slot[0]].cty));
	L.opnd[0][0] = mk_cast(TY(K_UINT, w), x);
	L.opnd[1][0] = mk_const_i(TY(K_UINT, w), (int64_t)(isext ? mask : dmask));
	if ((sem = intr_lane(nm, &L, 0, 0)) == 0)
		return;
	sem = mk_cast(N.ty[B->e], sem);
	snprintf(what, sizeof what, "intrinsic: %s(x, 0x%llx)", nm,
	    (unsigned long long)(isext ? mask : dmask));
	cf_cand_add(B->cs, sem, NULL, NULL, 0, 0, what);
	c = &B->cs->c[B->cs->n - 1];
	snprintf(c->engine, sizeof c->engine, "gf2");
	snprintf(c->cert_kind, sizeof c->cert_kind, "GF(2) linearity proof");
	snprintf(c->cert, sizeof c->cert, "%s; its matrix moves single bits in "
	    "order: a %s with mask 0x%llx", G->how, isext ? "gather" : "scatter",
	    (unsigned long long)(isext ? mask : dmask));
	snprintf(text, sizeof text, "%s($%d, 0x%llx%s)", nm, G->slot[0],
	    (unsigned long long)(isext ? mask : dmask), w == 64 ? "ull" : "u");
	c->text = strdup(text);
	c->feat = CF_F(CF_BMI2);
	c->fixcost = 1;
	c->cost = B->t->c_mul;	/* pext/pdep: latency 3, like imul */
}

/* GF(2)[t]: degree, and a mod b (b != 0) by the ring (Z/2)[t] */
static void
gf2_divmod(uint64_t ahi, uint64_t alo, uint64_t b, int db, uint64_t *qo)
{
	/* a has up to 128 coefficients (ahi:alo), b degree db <= 63 */
	struct ring R2;
	struct rpoly a, bb, q, r;
	int k;

	ring_zn(&R2, 2);
	memset(&a, 0, sizeof a);
	memset(&bb, 0, sizeof bb);
	rp_zero(&a);
	rp_zero(&bb);
	for (k = 0; k < 64; k++) {
		if (alo >> k & 1)
			a.c[k] = 1, a.d = k > a.d ? k : a.d;
		if (ahi >> k & 1)
			a.c[64 + k] = 1, a.d = 64 + k > a.d ? 64 + k : a.d;
		if (k <= db && (b >> k & 1))
			bb.c[k] = 1, bb.d = k > bb.d ? k : bb.d;
	}
	rp_norm(&a);
	rp_norm(&bb);
	rp_divmod(&R2, &a, &bb, &q, &r);
	*qo = 0;
	for (k = 0; k <= q.d && k < 64; k++)
		if (q.c[k])
			*qo |= 1ULL << k;
	ring_free(&R2);
}

/* carry-less product, low 64 bits */
static uint64_t
clmul_lo(uint64_t a, uint64_t b)
{
	uint64_t r = 0;
	int j;

	for (j = 0; j < 64; j++)
		if (b >> j & 1)
			r ^= a << j;
	return r;
}

/* reflect the low n bits */
static uint64_t
refl(uint64_t v, int n)
{
	uint64_t r = 0;
	int k;

	for (k = 0; k < n; k++)
		if (v >> k & 1)
			r |= 1ULL << (n - 1 - k);
	return r;
}

/* one step of a reflected CRC-32: (crc >> k) ^ T((crc ^ data) mod 2^k),
 * T = multiplication by t^32 mod P: two pclmul (Barrett) */
static void
g2_crc(struct bx *B, const struct g2 *G)
{
	int ic, idt, k, j, kk;
	uint64_t Plow, P, mu = 0, MU, PQ, maskk;
	nid crc, dat, x, u, v, sem;
	ty_t u64 = TY(K_UINT, 64), u32 = TY(K_UINT, 32);
	struct lanectx L;
	char text[700], what[120];
	struct cf_cand *c;

	if (G->ns != 2 || G->nout != 32 || G->c ||
	    !(B->t->feat & CF_F(CF_PCLMUL)))
		return;
	if (G->wid[0] == 32 && G->wid[1] <= 32) {
		ic = 0;
		idt = 1;
	} else if (G->wid[1] == 32 && G->wid[0] <= 32) {
		ic = 1;
		idt = 0;
	} else
		return;
	k = G->wid[idt];
	if (k != 8 && k != 16 && k != 32)
		return;
	maskk = wmask(k);
	/* the crc's bits j < k act as the data's; bits j >= k shift down */
	for (j = 0; j < 32; j++) {
		uint64_t cc = G->col[G->off[ic] + j];
		if (j < k) {
			if (cc != G->col[G->off[idt] + j])
				return;
		} else if (cc != (1ULL << (j - k)) && !(j >= k && cc == (1ULL <<
		    (j - k))))
			return;
	}
	/* x = data bit k-1 is R = t^0: T(x) = t^32 mod P = P's low part */
	Plow = refl(G->col[G->off[idt] + k - 1], 32);
	P = (1ULL << 32) | Plow;
	if (!(P & 1))
		return;
	/* mu = floor(t^(32+k) / P), degree k */
	if (32 + k < 64)
		gf2_divmod(0, 1ULL << (32 + k), P, 32, &mu);
	else
		gf2_divmod(1, 0, P, 32, &mu);	/* t^64 */
	MU = refl(mu, k + 1);
	PQ = refl(P, 33);
	/* check T on the basis: linear, so that is all of it */
	for (j = 0; j < k; j++) {
		uint64_t xx = 1ULL << j, q = clmul_lo(xx, MU) & maskk;
		uint64_t T = (clmul_lo(q, PQ) >> k) & 0xffffffffULL;
		if (T != G->col[G->off[idt] + j])
			return;
	}
	/* the semantics through the catalogue's _mm_clmulepi64_si128 */
	crc = mk_var((uint32_t)G->slot[ic], sym_intern(B->v->leaf[G->slot[ic]].text,
	    strlen(B->v->leaf[G->slot[ic]].text)), ct_to_ucas(B->v->leaf[
	    G->slot[ic]].cty));
	dat = mk_var((uint32_t)G->slot[idt], sym_intern(B->v->leaf[
	    G->slot[idt]].text, strlen(B->v->leaf[G->slot[idt]].text)),
	    ct_to_ucas(B->v->leaf[G->slot[idt]].cty));
	x = mk_bin(O_AND, mk_bin(O_XOR, mk_cast(u64, crc), mk_cast(u64,
	    mk_cast(TY(K_UINT, G->wid[idt]), dat))), mk_const_i(u64,
	    (int64_t)maskk));
	memset(&L, 0, sizeof L);
	L.opnd[0][0] = x;
	L.opnd[0][1] = mk_const_i(u64, 0);
	L.opnd[1][0] = mk_const_i(u64, (int64_t)MU);
	L.opnd[1][1] = mk_const_i(u64, 0);
	if ((u = intr_lane("_mm_clmulepi64_si128", &L, 0, 0)) == 0)
		return;
	L.opnd[0][0] = mk_bin(O_AND, u, mk_const_i(u64, (int64_t)maskk));
	L.opnd[1][0] = mk_const_i(u64, (int64_t)PQ);
	if ((v = intr_lane("_mm_clmulepi64_si128", &L, 0, 0)) == 0)
		return;
	sem = mk_cast(u32, mk_bin(O_SHR, v, mk_const_i(u64, k)));
	if (k < 32)
		sem = mk_bin(O_XOR, mk_bin(O_SHR, mk_cast(u32, crc), mk_const_i(u32,
		    k)), sem);
	sem = mk_cast(N.ty[B->e], sem);
	kk = k;
	snprintf(what, sizeof what, "intrinsic: a CRC-32 step of %d bits by two "
	    "pclmul (Barrett)", k);
	cf_cand_add(B->cs, sem, NULL, NULL, 0, 0, what);
	c = &B->cs->c[B->cs->n - 1];
	snprintf(c->engine, sizeof c->engine, "gf2");
	snprintf(c->cert_kind, sizeof c->cert_kind, "GF(2) linearity proof");
	snprintf(c->cert, sizeof c->cert, "%s; f = (crc >> %d) ^ T((crc ^ data) "
	    "mod 2^%d), T = x t^32 mod P, P = t^32 + 0x%08llx (reflected "
	    "0x%08llx), mu = t^%d div P (the ring (Z/2)[t]); T checked on all %d "
	    "basis vectors (it is linear)", G->how, k, k,
	    (unsigned long long)Plow, (unsigned long long)refl(Plow, 32), 32 + k,
	    k);
	snprintf(text, sizeof text, "{ const uint64_t x = ((uint64_t)$%d ^ "
	    "(uint64_t)$%d) & 0x%llxull; const __m128i u = "
	    "_mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)x), "
	    "_mm_cvtsi64_si128((long long)0x%llxull), 0); const uint64_t q = "
	    "(uint64_t)_mm_cvtsi128_si64(u) & 0x%llxull; const __m128i v = "
	    "_mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)q), "
	    "_mm_cvtsi64_si128((long long)0x%llxull), 0); return %s(uint32_t)"
	    "((uint64_t)_mm_cvtsi128_si64(v) >> %d); }", G->slot[ic],
	    G->slot[idt], (unsigned long long)maskk, (unsigned long long)MU,
	    (unsigned long long)maskk, (unsigned long long)PQ, kk < 32 ?
	    (G->slot[ic] == 0 ? "(uint32_t)($0 >> " : "(uint32_t)($1 >> ") : "",
	    kk);
	if (kk < 32) {
		/* finish "($c >> k) ^ ..." */
		char t2[700];
		snprintf(t2, sizeof t2, "%s", text);
		{
			char *p = strstr(t2, "return ");
			if (p) {
				char tail[400];
				snprintf(tail, sizeof tail, "return (uint32_t)((uint32_t)"
				    "$%d >> %d) ^ (uint32_t)((uint64_t)_mm_cvtsi128_si64(v) "
				    ">> %d); }", G->slot[ic], kk, kk);
				snprintf(p, sizeof t2 - (size_t)(p - t2), "%s", tail);
			}
		}
		snprintf(text, sizeof text, "%s", t2);
	}
	/* a proof that does not need SAT: both sides are GF(2)-affine by
	 * structure, so they are equal when equal at 0 and on the basis */
	if (g2_struct(sem, 0) && g2_struct(B->e, 0)) {
		struct plan sp;
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		int ok = 1, jj, q;
		plan_build(&sp, &sem, 1, 0);
		for (jj = -1; jj < G->nin && ok; jj++) {
			uint64_t bits = jj < 0 ? 0 : 1ULL << jj, want;
			for (q = 0; q < G->ns; q++)
				in_i[(size_t)G->slot[q] * CH] = int_norm(ct_to_ucas(
				    B->v->leaf[G->slot[q]].cty), (int64_t)((bits >>
				    G->off[q]) & wmask(G->wid[q])));
			plan_run(&sp, 1, in_i, in_r);
			want = jj < 0 ? G->c : G->c ^ G->col[jj];
			if ((COLF(&sp, sp.root[0])[0] & F_UNDEF) ||
			    ((uint64_t)COLI(&sp, sp.root[0])[0] & 0xffffffffULL) != want)
				ok = 0;
		}
		plan_free(&sp);
		free(in_i);
		free(in_r);
		if (ok) {
			snprintf(c->lemma, sizeof c->lemma, "both sides GF(2)-affine by "
			    "structure, equal at 0 and on all %d basis vectors", G->nin);
			c->derived = 1;
		}
	}
	c->text = strdup(text);
	c->feat = CF_F(CF_PCLMUL) | CF_F(CF_SSE2);
	c->fixcost = 1;
	/* two pclmul (latency 6-7, throughput 1) and moves: estimated */
	c->cost = 2 * B->t->c_mul + 4 * B->t->c_alu;
	/* the 8-bit step also as Sarwate's table, 256 x 4 bytes */
	if (k == 8) {
		nid ent[256], tab, r;
		uint64_t t8;
		int m2;
		for (m2 = 0; m2 < 256; m2++) {
			uint64_t vv = 0;
			for (j = 0; j < 8; j++)
				if (m2 >> j & 1)
					vv ^= G->col[G->off[idt] + j];
			ent[m2] = mk_const_i(u32, (int64_t)vv);
		}
		tab = mk_table(ent, 256);
		t8 = 0xff;
		r = mk_bin(O_XOR, mk_bin(O_SHR, mk_cast(u32, crc), mk_const_i(u32, 8)),
		    mk_index(tab, mk_bin(O_AND, mk_bin(O_XOR, mk_cast(u32, crc),
		    mk_cast(u32, dat)), mk_const_i(u32, (int64_t)t8))));
		cand(B, B->e, r, 1, "a CRC step by a table of 256 values (Sarwate)",
		    "GF(2) linearity proof", "%s; f = (crc >> 8) ^ T[(crc ^ data) & "
		    "0xff], T linear (its 8 columns are the matrix's)", G->how);
	}
}

static void
eng_gf2(struct bx *B)
{
	struct g2 G;

	g2_memo_reset();
	if (!g2_matrix(B, &G))
		return;
	g2_pext(B, &G);
	g2_crc(B, &G);
}

/* ================================================================
 * gates: bitwise functions of up to 6 word inputs
 * ================================================================ */

static int
bw_collect(nid n, ty_t t, nid *at, int *na, int *nops, int depth)
{
	int k;

	if (depth > 40)
		return 0;
	if (N.ty[n] == t && (N.op[n] == O_AND || N.op[n] == O_OR ||
	    N.op[n] == O_XOR)) {
		(*nops)++;
		return bw_collect(N.a[n], t, at, na, nops, depth + 1) &&
		    bw_collect(N.b[n], t, at, na, nops, depth + 1);
	}
	if (N.ty[n] == t && N.op[n] == O_NOT) {
		(*nops)++;
		return bw_collect(N.a[n], t, at, na, nops, depth + 1);
	}
	if (is_const(n))
		return N.ki[n] == 0 || (uint64_t)N.ki[n] == wmask(TYW(t)) ||
		    N.ki[n] == -1;
	for (k = 0; k < *na; k++)
		if (at[k] == n)
			return 1;
	if (*na == 6)
		return 0;
	at[(*na)++] = n;
	return 1;
}

/* the tree's value at minterm m: atom j all ones when bit j of m */
static int
bw_eval(nid n, const nid *at, int na, int m)
{
	int k;

	switch (N.op[n]) {
	case O_AND:
		return bw_eval(N.a[n], at, na, m) & bw_eval(N.b[n], at, na, m);
	case O_OR:
		return bw_eval(N.a[n], at, na, m) | bw_eval(N.b[n], at, na, m);
	case O_XOR:
		return bw_eval(N.a[n], at, na, m) ^ bw_eval(N.b[n], at, na, m);
	case O_NOT:
		return !bw_eval(N.a[n], at, na, m);
	}
	for (k = 0; k < na; k++)
		if (at[k] == n)
			return m >> k & 1;
	return N.ki[n] != 0;
}

/* a gate applied to word operands: its body with the words substituted */
static nid
gate_node(const struct gate *g, const nid *args, int lut, ty_t t)
{
	if (g->lut) {
		/* any function of its inputs: the sum of its minterms */
		nid r = 0;
		int m, k;
		for (m = 0; m < (1 << g->arity); m++) {
			nid p = 0;
			if (!(lut >> m & 1))
				continue;
			for (k = 0; k < g->arity; k++) {
				nid a = m >> k & 1 ? args[k] : mk_un(O_NOT, args[k]);
				p = p ? mk_bin(O_AND, p, a) : a;
			}
			r = r ? mk_bin(O_OR, r, p) : p;
		}
		return r ? r : mk_const_i(t, 0);
	}
	{
		nid map[3 * 2];
		int k;
		for (k = 0; k < g->arity; k++) {
			map[2 * k] = mk_var((uint32_t)k, 0, TY_BOOL);
			map[2 * k + 1] = args[k];
		}
		(void)map;
	}
	/* bodies over 1-bit inputs x0..x2 (slots 0..2): and, or, xor, not */
	return 0;
}

static nid
gate_word(nid body, const nid *args, ty_t t)
{
	switch (N.op[body]) {
	case O_VAR:
		return args[N.ki[body]];
	case O_AND:
	case O_OR:
	case O_XOR:
		return mk_bin(N.op[body], gate_word(N.a[body], args, t),
		    gate_word(N.b[body], args, t));
	case O_NOT:
		return mk_un(O_NOT, gate_word(N.a[body], args, t));
	case O_CAST:
		return gate_word(N.a[body], args, t);
	case O_CONST:
		return mk_const_i(t, N.ki[body] & 1 ? -1 : 0);
	}
	return 0;
}

static void
eng_gates(struct bx *B)
{
	uint32_t i;
	int tried = 0;

	for (i = B->nr; i-- > 0 && tried < 3 && eng_left() > 0;) {
		nid n = B->rn[i], at[6], vals[6 + 64], r;
		int na = 0, nops = 0, m, k;
		ty_t t = N.ty[n];
		uint64_t tt = 0, full;
		struct basis *bs;
		struct gprog gp;
		int cost;

		if (!TY_SIZED(t) || t == TY_BOOL || (N.op[n] != O_AND &&
		    N.op[n] != O_OR && N.op[n] != O_XOR))
			continue;
		if (!bw_collect(n, t, at, &na, &nops, 0) || na < 2 || nops < 3)
			continue;
		/* only maximal trees */
		{
			uint32_t j;
			int inner = 0;
			for (j = 0; j < B->nr && !inner; j++) {
				nid p = B->rn[j];
				if (N.ty[p] == t && (N.op[p] == O_AND || N.op[p] == O_OR ||
				    N.op[p] == O_XOR || N.op[p] == O_NOT) &&
				    (N.a[p] == n || N.b[p] == n))
					inner = 1;
			}
			if (inner)
				continue;
		}
		tried++;
		full = na == 6 ? ~0ULL : (1ULL << (1 << na)) - 1;
		for (m = 0; m < (1 << na); m++)
			if (bw_eval(n, at, na, m))
				tt |= 1ULL << m;
		bs = basis_find(sym_cstr(B->t->feat & CF_F(CF_BMI1) ? "x86bmi" :
		    "c"));
		if (!bs)
			continue;
		cost = gates_synth(bs, na, tt, full, nops - 1, 1u << 22, &gp);
		if (cost >= 0 && gp.out >= na) {
			for (k = 0; k < na; k++)
				vals[k] = at[k];
			for (k = 0; k < gp.nstep; k++) {
				const struct gate *g = &bs->g[gp.g[k]];
				nid args[3];
				int q;
				for (q = 0; q < g->arity; q++)
					args[q] = vals[gp.a[k][q]];
				vals[na + k] = g->lut ? gate_node(g, args, gp.a[k][3], t) :
				    gate_word(g->body, args, t);
				if (!vals[na + k])
					break;
			}
			if (k == gp.nstep && (r = vals[gp.out]) != 0)
				cand(B, n, r, 0, "a bitwise function by a minimal gate "
				    "network", "exhaustive search", "per-bit truth table "
				    "0x%llx of %d inputs (%d operations as written); gates "
				    "over basis %s: cost %d, minimal (no program of cost %d "
				    "exists, %llu candidates); bits are independent lanes",
				    (unsigned long long)tt, na, nops, sym_str(bs->name), cost,
				    cost - 1, (unsigned long long)gp.nodes);
		}
		/* vpternlog: any function of 3 words in one instruction */
		if (na == 3 && (B->t->feat & CF_F(CF_AVX512F)) &&
		    (B->t->feat & CF_F(CF_AVX512VL)) && (TYW(t) == 32 ||
		    TYW(t) == 64)) {
			int imm = 0;
			char text[400];
			struct cf_cand *c;
			for (m = 0; m < 8; m++)
				if (tt >> m & 1)
					imm |= 1 << ((m & 1) << 2 | (m & 2) | (m >> 2 & 1));
			{
				uint64_t sl = node_slots(n);
				int ok = 1;
				for (k = 0; k < 3; k++)
					ok &= N.op[at[k]] == O_VAR || (N.op[at[k]] == O_CAST &&
					    N.op[N.a[at[k]]] == O_VAR);
				if (!ok || __builtin_popcountll(sl) != 3 || n != B->e)
					continue;
			}
			{
				int s0 = (int)N.ki[N.op[at[0]] == O_VAR ? at[0] : N.a[at[0]]];
				int s1 = (int)N.ki[N.op[at[1]] == O_VAR ? at[1] : N.a[at[1]]];
				int s2 = (int)N.ki[N.op[at[2]] == O_VAR ? at[2] : N.a[at[2]]];
				if (TYW(t) == 32)
					snprintf(text, sizeof text, "(uint32_t)_mm_cvtsi128_si32("
					    "_mm_ternarylogic_epi32(_mm_cvtsi32_si128((int)$%d), "
					    "_mm_cvtsi32_si128((int)$%d), _mm_cvtsi32_si128((int)$%d), "
					    "0x%02x))", s0, s1, s2, imm);
				else
					snprintf(text, sizeof text, "(uint64_t)_mm_cvtsi128_si64("
					    "_mm_ternarylogic_epi64(_mm_cvtsi64_si128((long long)$%d), "
					    "_mm_cvtsi64_si128((long long)$%d), _mm_cvtsi64_si128("
					    "(long long)$%d), 0x%02x))", s0, s1, s2, imm);
			}
			cf_cand_add(B->cs, n, NULL, NULL, 0, 0, "intrinsic: vpternlog");
			c = &B->cs->c[B->cs->n - 1];
			snprintf(c->what, sizeof c->what, "intrinsic: vpternlog with "
			    "table 0x%02x", imm);
			snprintf(c->engine, sizeof c->engine, "gates");
			snprintf(c->cert_kind, sizeof c->cert_kind, "truth table");
			snprintf(c->cert, sizeof c->cert, "per-bit truth table of 3 "
			    "inputs = vpternlog's immediate 0x%02x (bit a<<2 | b<<1 | c); "
			    "basis avx512's ternlog gate", imm);
			c->text = strdup(text);
			c->feat = CF_F(CF_AVX512F) | CF_F(CF_AVX512VL);
			c->fixcost = 1;
			/* three moves in, one out, the instruction */
			c->cost = 5 * B->t->c_alu;
		}
	}
}

/* ================================================================
 * find: the enumerative search, the peephole drops, per-case fits
 * ================================================================ */

static const char *
x86_model_for(const struct cf_target *t)
{
	const char *n = t->name;

	if (strstr(n, "v4"))
		return "x86v4";
	if (strstr(n, "v3"))
		return "x86v3";
	if (!strcmp(n, "x86-64") || strstr(n, "v2"))
		return "x86v2";
	if (!strcmp(n, "skylake"))
		return "skl";
	if (!strcmp(n, "skylake-avx512"))
		return "skx";
	if (!strcmp(n, "alderlake") || !strcmp(n, "raptorlake"))
		return "rpl";
	if (!strcmp(n, "icelake") || !strcmp(n, "sapphirerapids"))
		return "glc";
	if (!strncmp(n, "zen", 3))
		return "zen4";
	return "x86v3";
}

static void
eng_find(struct bx *B)
{
	nid out[24];
	char why[24][96];
	int n, k, j;
	struct model *old = model_active(), *m;
	struct func f = B->f;
	double secs = eng_left();
	uint64_t sl = node_slots(B->e);

	if (secs <= 0 || !TY_SIZED(N.ty[B->e]) || N.ty[B->e] == TY_BOOL)
		return;
	for (k = 0; k < f.np; k++)
		if (!TY_ISINT(f.p[k].ty))
			return;
	if (__builtin_popcountll(sl) > 3)
		return;
	if ((m = model_find(sym_cstr(x86_model_for(B->t)))) != NULL)
		model_set_active(m);
	else if (!old)
		return;
	/* the depth is fixed (deterministic); the time only a safety cap */
	n = find_cands(&f, 5, secs, out, why, 24);
	model_set_active(old);
	for (k = 0; k < n; k++) {
		char what[120];
		if (out[k] == 0 || out[k] == B->e)
			continue;
		snprintf(what, sizeof what, "%s", why[k][0] ? why[k] :
		    "a cheaper term");
		if (!cand(B, B->e, out[k], 0, what, "search", "find: %s, verified "
		    "below like every candidate", why[k][0] ? why[k] : "enumerated"))
			continue;
		/* pext/pdep: the instruction itself */
		if ((!strncmp(why[k], "pext(", 5) || !strncmp(why[k], "pdep(", 5)) &&
		    (B->t->feat & CF_F(CF_BMI2))) {
			struct cf_cand *c = &B->cs->c[B->cs->n - 1];
			const char *mk = strstr(why[k], ", 0x");
			int s = -1, w = TYW(N.ty[B->e]);
			for (j = 0; j < B->v->nleaf; j++)
				if (sl >> j & 1)
					s = j;
			if (mk && s >= 0 && __builtin_popcountll(sl) == 1 &&
			    (w == 32 || w == 64)) {
				char text[160];
				unsigned long long K = strtoull(mk + 2, NULL, 16);
				snprintf(text, sizeof text, "_%.4s_u%d($%d, 0x%llx%s)",
				    why[k], w, s, K, w == 64 ? "ull" : "u");
				c->text = strdup(text);
				c->feat = CF_F(CF_BMI2);
				c->fixcost = 1;
				c->cost = B->t->c_mul;
				snprintf(c->what, sizeof c->what, "intrinsic: %s", why[k]);
			}
		}
	}
}

/* ================================================================
 * synth: CEGIS over the intrinsics catalogue (cf_synth.c)
 * ================================================================ */

static double
mov_cost(const struct cf_target *t)
{
	/* movd/movq between a general register and xmm, each way: the
	 * µarch's latency and throughput blended like cfront's numbers */
	int ua = cf_synth_ua(t), id = xi_find("_mm_cvtsi32_si128");
	const struct xcost *c = id >= 0 ? xcost_get(ua, xi_cls(id), xi_vw(id)) :
	    NULL;

	return c ? (c->lat + c->rtp) / 2 : 2;
}

static void
eng_synth(struct bx *B)
{
	uint64_t sl = node_slots(B->e);
	int s, k, w;
	ty_t it, rt = N.ty[B->e];
	double cur, mv, bound;
	struct cf_synth_res R;
	const struct cf_opts *o = cf_run_opts();
	long steps = o && o->synth_steps > 0 ? o->synth_steps : 3000;
	nid in16[16], out16[16], x, sem = 0;
	ty_t u64 = TY(K_UINT, 64);
	struct sbuf code = {0}, txt = {0}, beat = {0};
	char *text;
	struct cf_cand *c;

	if (__builtin_popcountll(sl) != 1 || B->v->where || !TY_SIZED(rt) ||
	    rt == TY_BOOL || TYW(rt) % 8)
		return;
	s = __builtin_ctzll(sl);
	if (s >= B->v->nleaf || B->v->leaf[s].ranged)
		return;
	it = ct_to_ucas(B->v->leaf[s].cty);
	if (!TY_SIZED(it) || it == TY_BOOL || TYW(it) % 8)
		return;
	if (B->nr < 5)
		return;		/* nothing to win */
	for (k = 0; k < (int)B->nr; k++) {
		int op = N.op[B->rn[k]];
		if (op != O_AND && op != O_OR && op != O_XOR && op != O_NOT &&
		    op != O_SHL && op != O_SHR && op != O_ADD && op != O_SUB &&
		    op != O_MUL && op != O_NEG && op != O_CAST && op != O_CONST &&
		    op != O_VAR)
			return;	/* a bit-manipulating value only */
	}
	cur = cf_cost(B->t, B->e);
	mv = 2 * mov_cost(B->t);
	bound = cur - mv;
	if (bound <= 0.5)
		return;
	{
		int ok = cf_synth_scalar(B->t, B->e, s, it, bound, steps, &R);
		if (o && o->trace)
			printf("  trace:   synth: %s under cost %.1f: %ld of %ld sketches, "
			    "%ld SAT calls, %ld counterexamples, %ld linear, %ld unknown\n",
			    ok ? "found" : "nothing", bound, R.st.sketches, R.st.space,
			    R.st.sat_calls, R.st.cex, R.st.linear, R.st.unknown);
		if (!ok)
			return;
	}
	/* the semantics: the catalogue's lanes with the solved constants */
	w = TYW(it);
	x = mk_cast(u64, mk_var((uint32_t)s, sym_intern(B->v->leaf[s].text,
	    strlen(B->v->leaf[s].text)), it));
	for (k = 0; k < 16; k++)
		in16[k] = k < w / 8 ? mk_cast(TY(K_UINT, 8), k ? mk_bin(O_SHR, x,
		    mk_const_i(u64, 8 * k)) : x) : mk_const_i(TY(K_UINT, 8), 0);
	if (!cf_synth_sem(&R.best, in16, out16))
		return;
	for (k = 0; k < TYW(rt) / 8; k++) {
		nid t = mk_cast(u64, out16[k]);
		if (k)
			t = mk_bin(O_SHL, t, mk_const_i(u64, 8 * k));
		sem = sem ? mk_bin(O_OR, sem, t) : t;
	}
	sem = mk_cast(rt, sem);
	cf_synth_text(&R.best, &txt);
	sb_printf(&code, "{ const __m128i cf_x = %s; ", w <= 32 ?
	    "_mm_cvtsi32_si128((int)$S)" : "_mm_cvtsi64_si128((long long)$S)");
	cf_synth_code(&R.best, 128, "cf_x", "cf_s", "cf_r", &code);
	sb_printf(&code, "return (%s)%s; }", TYW(rt) == 8 ? "uint8_t" :
	    TYW(rt) == 16 ? "uint16_t" : TYW(rt) == 32 ? "uint32_t" : "uint64_t",
	    TYW(rt) <= 32 ? "_mm_cvtsi128_si32(cf_r)" :
	    "_mm_cvtsi128_si64(cf_r)");
	/* $S: the slot; newlines out (one helper body) */
	{
		struct sbuf t2 = {0};
		const char *p;
		for (p = code.p; *p; p++)
			if (*p == '$' && p[1] == 'S') {
				sb_printf(&t2, "$%d", s);
				p++;
			} else if (*p == '\n')
				sb_putc(&t2, ' ');
			else
				sb_putc(&t2, *p);
		text = t2.p;
	}
	sb_printf(&beat, "the value as written (cost %.1f)", cur);
	for (k = 0; k < R.nbeat; k++) {
		struct sbuf b2 = {0};
		cf_synth_text(&R.beat[k], &b2);
		sb_printf(&beat, ", %s (cost %.1f)", b2.p, R.beat[k].cost + mv);
		free(b2.p);
	}
	{
		char what[96];
		snprintf(what, sizeof what, "intrinsic: synthesised: %.70s", txt.p);
		cf_cand_add(B->cs, sem, NULL, NULL, 0, 0, what);
	}
	c = &B->cs->c[B->cs->n - 1];
	snprintf(c->engine, sizeof c->engine, "synth");
	snprintf(c->cert_kind, sizeof c->cert_kind, "synthesis");
	snprintf(c->cert, sizeof c->cert, "CEGIS: %ld of %ld sketches under cost "
	    "%.1f tried, %ld SAT calls, %ld counterexamples, %ld linear solves; "
	    "winner %.60s (cost %.1f with the moves); beat %.60s",
	    R.st.sketches, R.st.space, bound, R.st.sat_calls, R.st.cex,
	    R.st.linear, txt.p, R.best.cost + mv, beat.p);
	c->text = text;
	c->feat = cf_synth_feat(&R.best) | CF_F(CF_SSE2);
	c->fixcost = 1;
	c->cost = R.best.cost + mv;
	free(code.p);
	free(txt.p);
	free(beat.p);
}

/* ================================================================
 * loops: closed forms (cf_lower.c calls these)
 * ================================================================ */

/*
 * C(T, m) mod 2^64, exactly, for 1 <= m <= 8: the product of the m
 * consecutive integers T, T - 1, ..., T - m + 1 holds at least v2(m!)
 * factors of two (it is m! C(T, m)); they are taken out term by term,
 * at most ctz(term) from each, a greedy split that always reaches
 * v2(m!) (and a zero term gives 0 whatever is shifted), and the odd
 * part of m! is a unit mod 2^64.  For T < m one term is 0: C(T, m) = 0.
 */
nid
cf_binom_node(nid T, int m)
{
	ty_t u = TY(K_UINT, 64);
	uint64_t f = 1, odd;
	int a = 0, k;
	nid prod = 0, left;

	if (m < 1 || m > 8)
		return 0;
	if (m == 1)
		return T;
	for (k = 2; k <= m; k++)
		f *= (uint64_t)k;
	while (!(f & 1)) {
		f >>= 1;
		a++;
	}
	odd = f;
	left = mk_const_i(u, a);
	for (k = 0; k < m; k++) {
		nid t = k ? mk_bin(O_SUB, T, mk_const_i(u, k)) : T, sh;
		sh = mk_bin(O_MIN, mk_un(O_CTZ, t), left);
		left = mk_bin(O_SUB, left, sh);
		t = mk_bin(O_SHR, t, sh);
		prod = prod ? mk_bin(O_MUL, prod, t) : t;
	}
	if (odd != 1)
		prod = mk_bin(O_MUL, prod, mk_const_i(u, (int64_t)inv2k(odd)));
	return prod;
}

/*
 * sum_{j<T} r^j mod 2^w for a constant r: (r^T - 1) = (r - 1) G in Z.
 * With r - 1 = 2^a o (o odd), mod 2^64: (r^T - 1) >> a = o G mod
 * 2^(64-a), so G = ((r^T - 1) >> a) o^-1 mod 2^(64 - a), which holds
 * the w bits wanted when a + w <= 64.  r^T by squaring (cf_powu64).
 */
nid
cf_geom_node(nid T, uint64_t r, int w, char *why, size_t nwhy)
{
	ty_t u = TY(K_UINT, 64);
	uint64_t d = r - 1, o, inv;
	int a = 0;
	nid P;

	if (r == 0 || r == 1)
		return 0;
	while (!(d & 1)) {
		d >>= 1;
		a++;
	}
	o = d;
	if (a + w > 64)
		return 0;
	inv = inv2k(o);
	if (o * inv != 1)
		return 0;
	P = mk_bin(O_POW, mk_const_i(u, (int64_t)r), T);
	P = mk_bin(O_SUB, P, mk_const_i(u, 1));
	if (a)
		P = mk_bin(O_SHR, P, mk_const_i(u, a));
	if (why)
		snprintf(why, nwhy, "sum r^j = (r^T - 1)/(r - 1) for r = %llu: r - 1 "
		    "= 2^%d*%llu, %llu^-1 = 0x%llx mod 2^64 (checked), exact in the "
		    "low %d >= %d bits", (unsigned long long)r, a,
		    (unsigned long long)o, (unsigned long long)o,
		    (unsigned long long)inv, 64 - a, w);
	return o == 1 ? P : mk_bin(O_MUL, P, mk_const_i(u, (int64_t)inv));
}


/* ================================================================
 * loops into integrals: float sums proposed in closed form (opt-in)
 * ================================================================ */

/* a rational as a C double literal */
static void
qlit(struct sbuf *b, aq c)
{
	if (c.d == 1)
		sb_printf(b, "%lld.0", (long long)c.n);
	else
		sb_printf(b, "(%lld.0 / %lld.0)", (long long)c.n, (long long)c.d);
}

/* an alg polynomial in the atoms (but atom ia) as C, atoms by their text */
static int
mp_c(struct sbuf *b, struct mp a, int ia, const char *const *atxt)
{
	int k, j, i;

	if (a.n == 0) {
		sb_puts(b, "0.0");
		return 1;
	}
	sb_puts(b, "(");
	for (k = 0; k < a.n; k++) {
		sb_puts(b, k ? " + " : "");
		qlit(b, a.t[k].c);
		for (j = 0; j < nat; j++) {
			if (j == ia && a.t[k].e[j])
				return 0;
			for (i = 0; i < a.t[k].e[j]; i++)
				sb_printf(b, " * (double)(%s)", atxt[j]);
		}
	}
	sb_puts(b, ")");
	return 1;
}

static int
loop_hook(struct cf_unit *u, uint32_t fn, uint32_t loop, struct cf_emit *em)
{
	struct cf_loop l;
	struct cf_iter it;
	const char *why = NULL;
	nid g, acc, rn;
	struct mp P;
	int ia = -1, d, k, j, ok = 0, top;
	size_t mark;
	jmp_buf saved;
	const char *atxt[AV];
	char abuf[AV][100];
	struct sbuf code = {0}, cl = {0}, b2 = {0};
	uint32_t from, to;
	const struct cf_opts *o = cf_run_opts();
	struct cf_rewrite *w;
	int termops = 0;

	(void)em;
	if (u->A.kind[loop] != CA_FOR || !cf_engine_on(CF_ENG_LOOPS) ||
	    !cf_span(u, loop, &from, &to))
		return 0;
	memset(&l, 0, sizeof l);
	if (!cf_extract_loop(u, fn, loop, &l, &it, &why) || l.nred != 1 ||
	    l.nload || l.nstore || l.step != 1 || !l.idecl)
		goto out;
	rn = l.rnext[0];
	if (TYK(N.ty[rn]) != K_FLOAT || N.op[rn] != O_ADD || l.rslot[0] < 0)
		goto out;
	acc = N.a[rn];
	g = N.b[rn];
	if (!(N.op[acc] == O_VAR && N.ki[acc] == l.rslot[0])) {
		acc = N.b[rn];
		g = N.a[rn];
		if (!(N.op[acc] == O_VAR && N.ki[acc] == l.rslot[0]))
			goto out;
	}
	if (has_slot(g, l.rslot[0]) || !has_slot(g, l.islot))
		goto out;
	{
		/* the term's float operations, for the rounding bound */
		struct bx Bt;
		memset(&Bt, 0, sizeof Bt);
		(void)Bt;
		termops = 0;
		{
			nid st[256];
			int sp = 0;
			uint8_t *seen = xcalloc(N.n, 1);
			st[sp++] = g;
			while (sp) {
				nid x = st[--sp];
				if (seen[x] || N.op[x] == O_CONST || N.op[x] == O_VAR)
					continue;
				seen[x] = 1;
				if (TYK(N.ty[x]) == K_FLOAT && N.op[x] != O_CAST)
					termops++;
				if (N.a[x] && sp < 250)
					st[sp++] = N.a[x];
				if (N.b[x] && sp < 250)
					st[sp++] = N.b[x];
			}
			free(seen);
		}
	}
	mark = ar_mark();
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0) {
		atoms_reset();
		if (mp_of(g, &P)) {
			for (k = 0; k < nat; k++) {
				nid a = at_node[k];
				if (has_slot(a, l.islot)) {
					if (ia >= 0 || !(N.op[a] == O_CAST && N.op[N.a[a]] == O_VAR &&
					    N.ki[N.a[a]] == l.islot))
						ia = -2;
					else
						ia = k;
				}
				if (N.op[a] == O_VAR && N.ki[a] < l.nleaf)
					snprintf(abuf[k], sizeof abuf[k], "%s", l.leaf[N.ki[a]]);
				else if (N.op[a] == O_CAST && N.op[N.a[a]] == O_VAR &&
				    N.ki[N.a[a]] < l.nleaf)
					snprintf(abuf[k], sizeof abuf[k], "%s", l.leaf[N.ki[N.a[a]]]);
				else
					ia = -2;
				atxt[k] = abuf[k];
			}
			d = ia >= 0 ? mp_deg(P, ia) : -1;
			if (ia >= 0 && d >= 0 && d <= 6) {
				/* sum_{i=lo}^{hi-1} i^k = F_k(hi - 1) - F_k(lo - 1) */
				ok = 1;
				sb_printf(&code, "if ((%s) > (%s)) {\n\t\tconst double cf_m1 = "
				    "(double)(%s) - 1.0, cf_m0 = (double)(%s) - 1.0;\n\t\t%s += "
				    "0.0", l.hi_text, l.lo_text, l.hi_text, l.lo_text,
				    l.rname[0]);
				for (k = 0; k <= d && ok; k++) {
					struct q c[16];
					struct mp ck = mp_coef(P, ia, k);
					int nc;
					if (mp_iszero(ck))
						continue;
					nc = faulhaber(k, c) ? k + 2 : 0;
					if (!nc) {
						ok = 0;
						break;
					}
					sb_puts(&code, " + ");
					ok = mp_c(&code, ck, ia, atxt);
					sb_puts(&code, " * ((0.0");
					for (j = 0; j < nc; j++) {
						aq qq;
						qq.n = c[j].n;
						qq.d = c[j].d;
						if (!c[j].n)
							continue;
						sb_puts(&code, " + ");
						qlit(&code, qq);
						for (top = 0; top < j; top++)
							sb_puts(&code, " * cf_m1");
					}
					sb_puts(&code, ") - (0.0");
					for (j = 0; j < nc; j++) {
						aq qq;
						qq.n = c[j].n;
						qq.d = c[j].d;
						if (!c[j].n)
							continue;
						sb_puts(&code, " + ");
						qlit(&code, qq);
						for (top = 0; top < j; top++)
							sb_puts(&code, " * cf_m0");
					}
					sb_puts(&code, "))");
				}
				sb_puts(&code, ";\n\t}");
				sb_printf(&cl, "%s", mp_str(P));
			}
		}
	}
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	ar_release(mark);
	if (!ok)
		goto out;
	/* the rounding of the loop itself: each term from exact inputs by
	 * termops roundings, then N additions (Higham, recursive summation) */
	w = cf_run_rewrite(o && o->approx && !(o->error > 0) ?
	    "approximation applied" : "proposed, not applied",
	    (int)u->tk.line[u->A.t0[loop]]);
	w->before = strdup("");
	w->after = strdup("");
	sb_printf(&b2, "a float sum as its exact real value: the term %s is a "
	    "polynomial of degree %d in %s, so the sum is a left Riemann sum of "
	    "a polynomial F and Euler-Maclaurin terminates (the integral of F "
	    "plus the h/2, h^2/12 ... corrections: Faulhaber's sums); not "
	    "bit-exact: the loop rounds, |loop - exact| <= g(N + %d) (|%s0| + "
	    "sum |terms|), g(k) = k u/(1 - k u), u = 2^-53, N the trip count, and "
	    "the closed form's own rounding is a few ulps of its terms; %s",
	    cl.p ? cl.p : "?", mp_deg(P, ia), l.iname, termops, l.rname[0],
	    o && o->approx ? "applied (approx)" : "'approx' applies it");
	snprintf(w->proof_text, sizeof w->proof_text, "%.399s", b2.p);
	w->proof = CF_PROOF_APPROX;
	snprintf(w->engine, sizeof w->engine, "loops");
	snprintf(w->cert_kind, sizeof w->cert_kind, "error bound");
	w->cert = strdup("Euler-Maclaurin remainder 0 (F is a polynomial); "
	    "the loop's rounding by the standard model");
	if (o && o->approx && !(o->error > 0)) {
		struct sbuf full = {0};
		sb_printf(&full, "{ %s }", code.p);
		w->info = 0;
		w->applied = 1;
		w->after = strdup(full.p);
		cf_run_replace(from, to, full.p);
		free(full.p);
	} else {
		w->info = 1;
		w->code = strdup(code.p);
	}
	free(code.p);
	free(cl.p);
	free(b2.p);
	free((char *)l.lo_text);
	free((char *)l.hi_text);
	free((char *)l.init_text);
	return w->applied;
out:
	free((char *)l.lo_text);
	free((char *)l.hi_text);
	free((char *)l.init_text);
	return 0;
}

/* ================================================================
 * registration
 * ================================================================ */

static void
value_hook(const struct cf_target *t, const struct cf_value *v, nid e,
    struct cf_cands *cs)
{
	struct bx B;
	jmp_buf saved;

	if (!TY_ISINT(N.ty[e]))
		return;
	bx_init(&B, t, v, e, cs);
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0) {
		if (eng_begin(CF_ENG_RING)) {
			eng_ring(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_DIO)) {
			eng_dio(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_ALG)) {
			eng_alg(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_GF2)) {
			eng_gf2(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_GATES)) {
			eng_gates(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_BOUND)) {
			eng_bound(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_FIND)) {
			eng_find(&B);
			eng_end();
		}
		if (eng_begin(CF_ENG_SYNTH)) {
			eng_synth(&B);
			eng_end();
		}
	}
	eng_end();
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	bx_free(&B);
}

void
cf_bridge_init(void)
{
	cf_add_value_hook(value_hook);
	cf_set_loop_hook(loop_hook);
}
