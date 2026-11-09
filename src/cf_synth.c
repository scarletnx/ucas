/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cf_synth.c: counterexample-guided inductive synthesis (CEGIS) over the
 * intrinsics catalogue, for cfront's bridge and the vectoriser.
 *
 * A specification is a byte-lane function (every byte of a vector maps
 * by f) or a scalar function of one word.  A program is a sequence of up
 * to SY_MAXLEN instructions on 128-bit registers; each instruction is a
 * template from the catalogue (x86intr_tab.inc) whose constant operands
 * are holes: gf2p8affine's matrix and xor byte, pshufb tables (alone, or
 * the nibble pair: two 16-entry lookups of the low and high nibbles,
 * combined), and/or/xor/add constants, gf2p8mul's multiplier,
 * vpternlog's imm8, shift and rotate amounts (enumerated).
 *
 * Sketches are enumerated cheapest first under the target's µarch costs
 * (x86data.c), up to the bound the caller gives (the value's current
 * cost) and a budget counted in sketches and SAT calls, so the results
 * are deterministic.  For each sketch the holes are solved against a
 * growing set of counterexample inputs: by SAT (the holes are the
 * parameters of a conjunction "program = spec at every counterexample",
 * bit-blasted by nodesat.c), or, for gf2p8affine of the input, by linear
 * algebra over GF(2) (the matrix from f on the basis vectors).  A
 * solution is then verified for all inputs (exhaustively up to 16 input
 * bits, else by SAT over the whole input register), and a failure adds
 * its counterexample.  Lane semantics come from the catalogue, so a
 * verified program is proved against the very semantics the vectoriser
 * proves its lanes with (x86check validates the catalogue against the
 * host's cc; GFNI and AVX-512 cannot run on this host: there it is the
 * catalogue's definition, checked against Intel's pseudo-code).
 *
 * Symmetry and equivalence pruning: no step whose result is unused, no
 * two constant masks in a row (they merge), binary operations with
 * ordered operands, a shift pair only when the amounts differ, and a
 * per-run cache of solved specifications.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfront.h"
#include "nodesat.h"
#include "x86.h"

enum {
	TP_GF2AFF, TP_GF2MUL, TP_PSHUFB, TP_NIB, TP_ANDK, TP_ORK, TP_XORK,
	TP_ADDK, TP_SRLI16, TP_SLLI16, TP_SRLI32, TP_SLLI32, TP_ROLD,
	TP_AND2, TP_OR2, TP_XOR2, TP_ANDN2, TP_ADD2, TP_TERN, TP_N
};

static const struct tpl {
	const char *name;	/* the catalogue's 128-bit form */
	int arity;		/* value operands */
	int nh;			/* u64 holes (a 128-bit table: 2) */
	int hb;			/* a byte hole */
	int nk;			/* enumerated amounts: 0 none */
	uint64_t feat;
	int ops;		/* instructions it stands for */
} T[TP_N] = {
	{"_mm_gf2p8affine_epi64_epi8", 1, 1, 1, 0, CF_F(CF_GFNI), 1},
	{"_mm_gf2p8mul_epi8", 1, 0, 1, 0, CF_F(CF_GFNI), 1},
	{"_mm_shuffle_epi8", 1, 2, 0, 0, CF_F(CF_SSSE3), 1},
	{"_mm_shuffle_epi8", 1, 4, 0, 3, CF_F(CF_SSSE3), 6},
	{"_mm_and_si128", 1, 1, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_or_si128", 1, 1, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_xor_si128", 1, 1, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_add_epi8", 1, 1, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_srli_epi16", 1, 0, 0, 7, CF_F(CF_SSE2), 1},
	{"_mm_slli_epi16", 1, 0, 0, 7, CF_F(CF_SSE2), 1},
	{"_mm_srli_epi32", 1, 0, 0, 7, CF_F(CF_SSE2), 1},
	{"_mm_slli_epi32", 1, 0, 0, 7, CF_F(CF_SSE2), 1},
	{"_mm_rol_epi32", 1, 0, 0, 31, CF_F(CF_AVX512F) | CF_F(CF_AVX512VL), 1},
	{"_mm_and_si128", 2, 0, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_or_si128", 2, 0, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_xor_si128", 2, 0, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_andnot_si128", 2, 0, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_add_epi8", 2, 0, 0, 0, CF_F(CF_SSE2), 1},
	{"_mm_ternarylogic_epi32", 3, 1, 1, 0, CF_F(CF_AVX512F) |
	    CF_F(CF_AVX512VL), 1},
};

/* the nibble pair: comb(pshufb(L, v & 0x0f), pshufb(H, (v >> 4) & 0x0f)) */
static const char *const nibcomb[3] = {"_mm_xor_si128", "_mm_or_si128",
	"_mm_add_epi8"};

/* ---- 128-bit values as 16 byte lanes ---- */

struct v16 {
	nid b[16];
};

static ty_t U8, U64;

static nid
ku8(int v)
{
	return mk_const_i(U8, v & 0xff);
}

/* bytes of a 64-bit node (a hole or a constant), repeated per qword */
static void
q_bytes(nid q, struct v16 *v)
{
	int i;

	for (i = 0; i < 16; i++)
		v->b[i] = mk_cast(U8, (i % 8) ? mk_bin(O_SHR, q, mk_const_i(U64,
		    8 * (i % 8))) : q);
}

/* the 16 bytes of a 128-bit table held in two 64-bit nodes */
static void
t_bytes(nid lo, nid hi, struct v16 *v)
{
	int i;

	for (i = 0; i < 16; i++) {
		nid q = i < 8 ? lo : hi;
		v->b[i] = mk_cast(U8, (i % 8) ? mk_bin(O_SHR, q, mk_const_i(U64,
		    8 * (i % 8))) : q);
	}
}

static void
b_bytes(nid b, struct v16 *v)
{
	int i;

	for (i = 0; i < 16; i++)
		v->b[i] = mk_cast(U8, b);
}

struct opv {
	const struct v16 *v[4];
	int w[4];		/* operand lane bits */
};

/* lane j of width w from bytes */
static nid
lane_of(const struct v16 *v, int j, int w)
{
	ty_t t = TY(K_UINT, w);
	nid r = 0;
	int q, n = w / 8;

	if (w == 8)
		return v->b[j];
	for (q = 0; q < n; q++) {
		nid p = mk_cast(t, v->b[j * n + q]);
		if (q)
			p = mk_bin(O_SHL, p, mk_const_i(t, 8 * q));
		r = r ? mk_bin(O_OR, r, p) : p;
	}
	return r;
}

static nid
opv_cb(void *c, int k, int lane)
{
	struct opv *o = c;

	return o->v[k] ? lane_of(o->v[k], lane, o->w[k]) : 0;
}

static int
apply(const char *name, const struct v16 *const *args, int64_t imm,
    struct v16 *out)
{
	int id = xi_find(name), k, nl, rw, j, q;
	struct opv o;

	if (id < 0)
		return 0;
	memset(&o, 0, sizeof o);
	for (k = 0; k < xi_nops(id) && k < 4; k++)
		if (xi_okind(id, k) == 'v') {
			o.v[k] = args[k];
			o.w[k] = TYW(xi_otype(id, k));
		}
	nl = xi_lanes(id);
	rw = TYW(xi_rtype(id));
	for (j = 0; j < nl; j++) {
		nid l = xi_lane(id, j, imm, opv_cb, &o);
		l = mk_cast(TY(K_UINT, rw), l);
		for (q = 0; q < rw / 8; q++)
			out->b[j * (rw / 8) + q] = mk_cast(U8, q ? mk_bin(O_SHR, l,
			    mk_const_i(TY(K_UINT, rw), 8 * q)) : l);
	}
	return 1;
}

/* vpternlogd with its immediate a node (the solver's form); the
 * catalogue's form is used once imm is known */
static void
tern_node(const struct v16 *a, const struct v16 *b, const struct v16 *c,
    nid imm, struct v16 *out)
{
	int i, m;

	for (i = 0; i < 16; i++) {
		nid r = 0;
		for (m = 0; m < 8; m++) {
			nid x = m & 4 ? a->b[i] : mk_un(O_NOT, a->b[i]);
			nid y = m & 2 ? b->b[i] : mk_un(O_NOT, b->b[i]);
			nid z = m & 1 ? c->b[i] : mk_un(O_NOT, c->b[i]);
			nid bit = mk_bin(O_AND, mk_bin(O_SHR, imm, ku8(m)), ku8(1));
			nid t = mk_bin(O_AND, mk_bin(O_AND, mk_bin(O_AND, x, y), z),
			    mk_un(O_NEG, bit));
			r = r ? mk_bin(O_OR, r, t) : t;
		}
		out->b[i] = r;
	}
}

/* ---- programs ---- */

static double
step_cost(int ua, int tp)
{
	int id = xi_find(T[tp].name);
	const struct xcost *c = id >= 0 ? xcost_get(ua, xi_cls(id),
	    xi_vw(id)) : NULL;
	double one = c ? (c->lat + c->rtp) / 2 : 1;

	if (one < 0.5)
		one = 0.5;
	return one * T[tp].ops;
}

/*
 * The program's bytes from the input bytes: holes are nodes (solving)
 * or constants (verifying, emitting).  hv[k]: the k-th u64 hole, hb[k]
 * the k-th byte hole, in step order.
 */
static int
run(const struct cf_synth_prog *p, const struct v16 *in, const nid *hv,
    const nid *hb, int tern_solving, struct v16 *out)
{
	struct v16 val[SY_MAXLEN], tmp, tmp2, tmp3, kv, kv2;
	int s, h = 0, b = 0;

	for (s = 0; s < p->n; s++) {
		const struct cf_synth_step *st = &p->s[s];
		const struct v16 *a = st->a < 0 ? in : &val[st->a];
		const struct v16 *args[4];
		struct v16 *r = &val[s];
		int i;

		switch (st->tpl) {
		case TP_GF2AFF:
			q_bytes(hv[h++], &kv);
			args[0] = a;
			args[1] = &kv;
			if (!apply(T[st->tpl].name, args, 0, r))
				return 0;
			for (i = 0; i < 16; i++)
				r->b[i] = mk_bin(O_XOR, r->b[i], hb[b]);
			b++;
			break;
		case TP_GF2MUL:
			b_bytes(hb[b++], &kv);
			args[0] = a;
			args[1] = &kv;
			if (!apply(T[st->tpl].name, args, 0, r))
				return 0;
			break;
		case TP_PSHUFB:
			t_bytes(hv[h], hv[h + 1], &kv);
			h += 2;
			args[0] = &kv;
			args[1] = a;
			if (!apply(T[st->tpl].name, args, 0, r))
				return 0;
			break;
		case TP_NIB: {
			struct v16 lo, hi, m;
			b_bytes(ku8(0x0f), &m);
			args[0] = a;
			args[1] = &m;
			apply("_mm_and_si128", args, 0, &lo);
			args[0] = a;
			apply("_mm_srli_epi16", args, 4, &tmp);
			args[0] = &tmp;
			args[1] = &m;
			apply("_mm_and_si128", args, 0, &hi);
			t_bytes(hv[h], hv[h + 1], &kv);
			t_bytes(hv[h + 2], hv[h + 3], &kv2);
			h += 4;
			args[0] = &kv;
			args[1] = &lo;
			apply("_mm_shuffle_epi8", args, 0, &tmp2);
			args[0] = &kv2;
			args[1] = &hi;
			apply("_mm_shuffle_epi8", args, 0, &tmp3);
			args[0] = &tmp2;
			args[1] = &tmp3;
			if (!apply(nibcomb[st->k], args, 0, r))
				return 0;
			break;
		}
		case TP_ANDK:
		case TP_ORK:
		case TP_XORK:
		case TP_ADDK:
			q_bytes(hv[h++], &kv);
			args[0] = a;
			args[1] = &kv;
			if (!apply(T[st->tpl].name, args, 0, r))
				return 0;
			break;
		case TP_SRLI16:
		case TP_SLLI16:
		case TP_SRLI32:
		case TP_SLLI32:
		case TP_ROLD:
			args[0] = a;
			if (!apply(T[st->tpl].name, args, st->k, r))
				return 0;
			break;
		case TP_AND2:
		case TP_OR2:
		case TP_XOR2:
		case TP_ANDN2:
		case TP_ADD2:
			args[0] = a;
			args[1] = st->b < 0 ? in : &val[st->b];
			if (!apply(T[st->tpl].name, args, 0, r))
				return 0;
			break;
		case TP_TERN: {
			const struct v16 *bb = st->b < 0 ? in : &val[st->b];
			q_bytes(hv[h++], &kv);
			if (tern_solving)
				tern_node(a, bb, &kv, hb[b], r);
			else {
				args[0] = a;
				args[1] = bb;
				args[2] = &kv;
				if (!apply(T[st->tpl].name, args, N.ki[hb[b]] & 0xff, r))
					return 0;
			}
			b++;
			break;
		}
		default:
			return 0;
		}
	}
	*out = val[p->n - 1];
	return 1;
}

static int
holes_of(const struct cf_synth_prog *p, int *nh, int *nb)
{
	int s;

	*nh = *nb = 0;
	for (s = 0; s < p->n; s++) {
		*nh += T[p->s[s].tpl].nh;
		*nb += T[p->s[s].tpl].hb;
	}
	return *nh * 64 + *nb * 8;
}

/* ---- the problem ---- */

struct prob {
	int lanes;		/* 1: every byte maps by f; 0: a scalar word */
	nid f;			/* f over slot 0 (u8, lanes) or the word slot */
	int slot;
	ty_t ity, oty;		/* scalar: the input's and the output's types */
	int win;		/* input bits (scalar) */
	struct plan pl;
	int ncex;
	uint64_t cex[64][2];	/* inputs: the word, or 16 bytes */
};

static uint64_t
ev(struct prob *P, uint64_t x)
{
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];

	memset(in_i, 0, sizeof in_i);
	memset(in_r, 0, sizeof in_r);
	in_i[(size_t)P->slot * CH] = int_norm(P->ity, (int64_t)x);
	plan_run(&P->pl, 1, in_i, in_r);
	return (uint64_t)COLI(&P->pl, P->pl.root[0])[0];
}

static void
in_bytes(const struct prob *P, const uint64_t *x, struct v16 *v)
{
	int i;

	for (i = 0; i < 16; i++)
		v->b[i] = ku8((int)((x[i / 8] >> (8 * (i % 8))) & 0xff));
	if (!P->lanes)
		for (i = P->win / 8; i < 16; i++)
			v->b[i] = ku8(0);
}

/* "the program's output is the spec's at input x": a bool node */
static nid
agree(struct prob *P, const struct v16 *out, const uint64_t *x)
{
	nid r = 0, e;
	int i, ob = P->lanes ? 16 : TYW(P->oty) / 8;

	if (P->lanes) {
		for (i = 0; i < 16; i++) {
			uint64_t y = ev(P, (x[i / 8] >> (8 * (i % 8))) & 0xff) & 0xff;
			e = mk_bin(O_EQ, out->b[i], ku8((int)y));
			r = r ? mk_bin(O_LAND, r, e) : e;
		}
		return r;
	}
	{
		uint64_t y = ev(P, x[0]);
		for (i = 0; i < ob; i++) {
			e = mk_bin(O_EQ, out->b[i], ku8((int)(y >> (8 * i))));
			r = r ? mk_bin(O_LAND, r, e) : e;
		}
	}
	return r;
}

static struct cf_synth_stats *ST;

/* holes satisfying every counterexample: 1, 0 none, -1 gave up */
static int
solve_sat(struct prob *P, const struct cf_synth_prog *p, uint64_t *hval,
    int *bval)
{
	int nh, nb, k, c;
	nid hv[8], hb[8], conj = 0;
	struct func f;
	struct nsat_res res;

	holes_of(p, &nh, &nb);
	if (nh + nb > MAXP)
		return -1;
	memset(&f, 0, sizeof f);
	f.name = sym_cstr("synth");
	f.anon = 1;
	f.np = nh + nb;
	for (k = 0; k < nh; k++) {
		char nm[8];
		snprintf(nm, sizeof nm, "h%d", k);
		f.p[k].name = sym_cstr(nm);
		f.p[k].ty = U64;
		hv[k] = mk_var((uint32_t)k, f.p[k].name, U64);
	}
	for (k = 0; k < nb; k++) {
		char nm[8];
		snprintf(nm, sizeof nm, "b%d", k);
		f.p[nh + k].name = sym_cstr(nm);
		f.p[nh + k].ty = U8;
		hb[k] = mk_var((uint32_t)(nh + k), f.p[nh + k].name, U8);
	}
	for (c = 0; c < P->ncex; c++) {
		struct v16 in, out;
		in_bytes(P, P->cex[c], &in);
		if (!run(p, &in, hv, hb, 1, &out))
			return -1;
		conj = conj ? mk_bin(O_LAND, conj, agree(P, &out, P->cex[c])) :
		    agree(P, &out, P->cex[c]);
	}
	f.body = conj;
	ST->sat_calls++;
	nsat_equiv(&f, conj, mk_const_i(TY_BOOL, 0), 20000, &res);
	if (res.status == 1) {
		for (k = 0; k < nh; k++)
			hval[k] = (uint64_t)res.in[k];
		for (k = 0; k < nb; k++)
			bval[k] = (int)(res.in[nh + k] & 0xff);
		return 1;
	}
	if (res.status == 0)
		return 0;
	return -1;
}

/* gf2p8affine of the input: the matrix by linear algebra over GF(2) */
static int
solve_affine(struct prob *P, uint64_t *hval, int *bval)
{
	uint64_t c0, col[8], A = 0, x;
	int i, j, nb = P->lanes ? 1 : P->win / 8;

	if (!P->lanes && (P->win != TYW(P->oty) || P->win % 8))
		return 0;
	ST->linear++;
	c0 = ev(P, 0);
	for (j = 0; j < 8; j++)
		col[j] = (ev(P, 1ULL << j) ^ c0) & 0xff;
	for (i = 0; i < 8; i++) {
		uint64_t row = 0;
		for (j = 0; j < 8; j++)
			if (col[j] >> i & 1)
				row |= 1ULL << j;
		A |= row << (8 * (7 - i));
	}
	/* the other bytes must use the same map (checked by verify) */
	(void)nb;
	(void)x;
	hval[0] = A;
	bval[0] = (int)(c0 & 0xff);
	return 1;
}

/* f with slot replaced by x */
static nid
spec_at(const struct prob *P, nid x)
{
	nid map[MAXP + 1];

	memset(map, 0, sizeof map);
	map[P->slot] = x;
	return subst(P->f, map, P->slot + 1);
}

static void
add_cex(struct prob *P, uint64_t x0, uint64_t x1)
{
	if (P->ncex < 64) {
		P->cex[P->ncex][0] = x0;
		P->cex[P->ncex][1] = x1;
		P->ncex++;
	}
	ST->cex++;
}

/* every input: 1, or 0 with a new counterexample, -1 unknown */
static int
verify(struct prob *P, const struct cf_synth_prog *p, const uint64_t *hval,
    const int *bval)
{
	int nh, nb, k, i, ob = TYW(P->oty) / 8;
	nid hv[8], hb[8], q0, q1, conj = 0, e, word = 0, spec;
	struct v16 in, out;
	struct func f;
	struct nsat_res res;

	holes_of(p, &nh, &nb);
	for (k = 0; k < nh; k++)
		hv[k] = mk_const_i(U64, (int64_t)hval[k]);
	for (k = 0; k < nb; k++)
		hb[k] = ku8(bval[k]);
	ST->verified++;
	memset(&f, 0, sizeof f);
	f.name = sym_cstr("synthv");
	f.anon = 1;
	f.np = P->lanes ? 2 : 1;
	f.p[0].name = sym_cstr("x0");
	f.p[0].ty = P->lanes ? U64 : P->ity;
	f.p[1].name = sym_cstr("x1");
	f.p[1].ty = U64;
	q0 = mk_var(0, f.p[0].name, f.p[0].ty);
	q1 = mk_var(1, f.p[1].name, U64);
	if (P->lanes) {
		t_bytes(q0, q1, &in);
		if (!run(p, &in, hv, hb, 0, &out))
			return -1;
		for (i = 0; i < 16; i++) {
			e = mk_bin(O_EQ, out.b[i], mk_cast(U8, spec_at(P, in.b[i])));
			conj = conj ? mk_bin(O_LAND, conj, e) : e;
		}
		f.body = conj;
		ST->sat_calls++;
		nsat_equiv(&f, conj, mk_const_i(TY_BOOL, 1), 100000, &res);
		if (res.status == 0)
			return 1;
		if (res.status == 1) {
			add_cex(P, (uint64_t)res.in[0], (uint64_t)res.in[1]);
			return 0;
		}
		return -1;
	}
	{
		nid x = mk_cast(U64, q0);
		for (i = 0; i < 16; i++)
			in.b[i] = i < P->win / 8 ? mk_cast(U8, i ? mk_bin(O_SHR, x,
			    mk_const_i(U64, 8 * i)) : x) : ku8(0);
	}
	if (!run(p, &in, hv, hb, 0, &out))
		return -1;
	for (i = 0; i < ob; i++) {
		nid t = mk_cast(U64, out.b[i]);
		if (i)
			t = mk_bin(O_SHL, t, mk_const_i(U64, 8 * i));
		word = word ? mk_bin(O_OR, word, t) : t;
	}
	spec = mk_cast(TY(K_UINT, TYW(P->oty)), spec_at(P, q0));
	word = mk_cast(TY(K_UINT, TYW(P->oty)), word);
	if (P->win <= 16) {
		/* exhaustive over the input */
		struct plan pl;
		nid roots[2];
		int64_t *in_i = xcalloc((size_t)MAXP * CH, sizeof *in_i);
		double *in_r = xcalloc((size_t)MAXP * CH, sizeof *in_r);
		uint64_t x = 0, top = 1ULL << P->win;
		int bad = 0;
		roots[0] = spec;
		roots[1] = word;
		plan_build(&pl, roots, 2, 0);
		while (x < top && !bad) {
			uint32_t np = 0, j;
			for (; np < CH && x < top; np++, x++)
				in_i[np] = int_norm(P->ity, (int64_t)x);
			plan_run(&pl, np, in_i, in_r);
			for (j = 0; j < np; j++)
				if (!(COLF(&pl, pl.root[0])[j] & F_UNDEF) &&
				    COLI(&pl, pl.root[0])[j] != COLI(&pl, pl.root[1])[j]) {
					add_cex(P, (uint64_t)in_i[j] & ((1ULL << P->win) - 1), 0);
					bad = 1;
					break;
				}
		}
		plan_free(&pl);
		free(in_i);
		free(in_r);
		return !bad;
	}
	f.body = spec;
	ST->sat_calls++;
	nsat_equiv(&f, spec, word, 100000, &res);
	if (res.status == 0)
		return 1;
	if (res.status == 1) {
		add_cex(P, (uint64_t)res.in[0], 0);
		return 0;
	}
	return -1;
}

/* CEGIS on one sketch: 1 solved (holes filled), 0 impossible, -1 unknown */
static int
cegis(struct prob *P, struct cf_synth_prog *p)
{
	uint64_t hval[8];
	int bval[8], it, r, nh, nb;

	holes_of(p, &nh, &nb);
	for (it = 0; it < 16; it++) {
		if (nh + nb == 0) {
			memset(hval, 0, sizeof hval);
			memset(bval, 0, sizeof bval);
		} else if (p->n == 1 && p->s[0].tpl == TP_GF2AFF && p->s[0].a < 0) {
			solve_affine(P, hval, bval);
		} else {
			r = solve_sat(P, p, hval, bval);
			if (r <= 0)
				return r;
		}
		r = verify(P, p, hval, bval);
		if (r == 1) {
			memcpy(p->h, hval, sizeof p->h);
			memcpy(p->hb, bval, sizeof p->hb);
			return 1;
		}
		if (r < 0)
			return -1;
		if (nh + nb == 0 || (p->n == 1 && p->s[0].tpl == TP_GF2AFF &&
		    p->s[0].a < 0))
			return 0;
	}
	return -1;
}

/* ---- sketches ---- */

static int ua_cur;

static double
prog_cost(const struct cf_synth_prog *p)
{
	double c = 0;
	int s;

	for (s = 0; s < p->n; s++)
		c += step_cost(ua_cur, p->s[s].tpl);
	return c;
}

/* every step's result is used, and no obviously redundant pair */
static int
sane(const struct cf_synth_prog *p)
{
	int s, used[SY_MAXLEN] = {0};

	for (s = 0; s < p->n; s++) {
		const struct cf_synth_step *st = &p->s[s];
		if (st->a >= 0)
			used[st->a] = 1;
		if (T[st->tpl].arity >= 2 && st->b >= 0)
			used[st->b] = 1;
		if (T[st->tpl].arity == 2 && st->a > st->b && st->tpl != TP_ANDN2)
			return 0;	/* commutative: ordered */
		if (T[st->tpl].arity >= 2 && st->a == st->b)
			return 0;
		if (st->a >= 0 && T[st->tpl].arity == 1 && T[st->tpl].nh == 1 &&
		    p->s[st->a].tpl == st->tpl && st->tpl != TP_GF2AFF)
			return 0;	/* two masks merge */
		if (st->a >= 0 && st->tpl == TP_GF2AFF && p->s[st->a].tpl == TP_GF2AFF)
			return 0;	/* affine maps compose */
	}
	for (s = 0; s + 1 < p->n; s++)
		if (!used[s])
			return 0;
	return 1;
}

struct sk {
	struct cf_synth_prog p;
	double cost;
};

static int
cmp_sk(const void *a, const void *b)
{
	const struct sk *x = a, *y = b;

	/* shortest first, then cheapest: a solution bounds the rest */
	if (x->p.n != y->p.n)
		return x->p.n - y->p.n;
	if (x->cost != y->cost)
		return x->cost < y->cost ? -1 : 1;
	return memcmp(&x->p.s, &y->p.s, sizeof x->p.s);
}

static struct sk *sks;
static int nsk, capsk;

static void
add_sk(const struct cf_synth_prog *p, double bound)
{
	double c;

	if (!sane(p))
		return;
	c = prog_cost(p);
	if (c >= bound)
		return;
	if (nsk == capsk) {
		capsk = capsk ? capsk * 2 : 1024;
		sks = xrealloc(sks, (size_t)capsk * sizeof *sks);
	}
	sks[nsk].p = *p;
	sks[nsk].cost = c;
	nsk++;
}

static const int kset[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 24};
#define NKSHIFT 7	/* byte-sized shifts: 1..7 (8 and up move whole bytes) */

static void
enum_steps(struct cf_synth_prog *p, int len, uint64_t feat, double bound)
{
	int s = p->n, tp, a, b, kk;

	if (s == len) {
		add_sk(p, bound);
		return;
	}
	for (tp = 0; tp < TP_N; tp++) {
		const struct tpl *t = &T[tp];
		int nk = t->nk ? (int)(sizeof kset / sizeof kset[0]) : 1;
		if ((t->feat & feat) != t->feat)
			continue;
		if (tp == TP_NIB)
			nk = 3;
		for (a = -1; a < s; a++)
			for (b = -1; b < (t->arity >= 2 ? s : 0); b++)
				for (kk = 0; kk < nk; kk++) {
					struct cf_synth_step *st = &p->s[s];
					memset(st, 0, sizeof *st);
					st->tpl = tp;
					st->a = a;
					st->b = t->arity >= 2 ? b : -1;
					st->k = tp == TP_NIB ? kk : t->nk ? kset[kk] : 0;
					if (t->nk && st->k > t->nk)
						continue;
					if (t->nk && tp != TP_ROLD && tp != TP_NIB &&
					    kk >= NKSHIFT)
						continue;
					p->n = s + 1;
					enum_steps(p, len, feat, bound);
					p->n = s;
				}
	}
}

/* ---- the search ---- */

static struct {
	nid f;
	int lanes;
	const char *target;
	int ok;
	struct cf_synth_res r;
} cache[64];
static int ncache;

int
cf_synth_ua(const struct cf_target *t)
{
	const char *n = t->name;
	const char *m = strstr(n, "v4") ? "x86v4" : strstr(n, "v3") ? "x86v3" :
	    (!strcmp(n, "x86-64") || strstr(n, "v2")) ? "x86v2" :
	    !strcmp(n, "skylake") ? "skl" : !strcmp(n, "skylake-avx512") ? "skx" :
	    !strcmp(n, "alderlake") || !strcmp(n, "raptorlake") ? "rpl" :
	    !strcmp(n, "icelake") || !strcmp(n, "sapphirerapids") ? "glc" :
	    !strncmp(n, "zen", 3) ? "zen4" : "x86v3";

	return uarch_find(m);
}

static int
search(struct prob *P, const struct cf_target *t, double bound, long steps,
    struct cf_synth_res *R)
{
	struct cf_synth_prog p;
	int len, i, best = -1;
	clock_t t0 = clock();

	U8 = TY(K_UINT, 8);
	U64 = TY(K_UINT, 64);
	ST = &R->st;
	ua_cur = cf_synth_ua(t);
	nsk = 0;
	for (len = 1; len <= SY_MAXLEN; len++) {
		memset(&p, 0, sizeof p);
		enum_steps(&p, len, t->feat, bound);
	}
	qsort(sks, (size_t)nsk, sizeof *sks, cmp_sk);
	/* fixed starting counterexamples (a fixed seed: deterministic) */
	{
		uint64_t s = 0x9e3779b97f4a7c15ULL;
		P->ncex = 0;
		for (i = 0; i < 4; i++) {
			s ^= s << 13;
			s ^= s >> 7;
			s ^= s << 17;
			P->cex[P->ncex][0] = i == 0 ? 0 : i == 1 ? ~0ULL : s;
			P->cex[P->ncex][1] = i == 0 ? 0 : i == 1 ? ~0ULL : s * 31;
			if (!P->lanes && P->win < 64) {
				P->cex[P->ncex][0] &= (1ULL << P->win) - 1;
				P->cex[P->ncex][1] = 0;
			}
			P->ncex++;
		}
	}
	R->nbeat = 0;
	for (i = 0; i < nsk && R->st.sketches < steps; i++) {
		int r;
		if (best >= 0 && sks[i].cost >= R->best.cost)
			continue;	/* a solution bounds the rest */
		R->st.sketches++;
		p = sks[i].p;
		r = cegis(P, &p);
		if (r == 1) {
			p.cost = sks[i].cost;
			if (best >= 0 && R->nbeat < SY_MAXBEAT)
				R->beat[R->nbeat++] = R->best;
			R->best = p;
			best = i;
		}
		if (r < 0)
			R->st.unknown++;
	}
	R->st.exhausted = i == nsk;
	R->st.space = nsk;
	R->st.seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
	return best >= 0;
}

static int
cached(nid f, int lanes, const struct cf_target *t, struct cf_synth_res *r,
    int *ok)
{
	int k;

	for (k = 0; k < ncache; k++)
		if (cache[k].f == f && cache[k].lanes == lanes &&
		    cache[k].target == t->name) {
			*r = cache[k].r;
			r->st.cached = 1;
			*ok = cache[k].ok;
			return 1;
		}
	return 0;
}

static void
remember(nid f, int lanes, const struct cf_target *t,
    const struct cf_synth_res *r, int ok)
{
	if (ncache == 64)
		return;
	cache[ncache].f = f;
	cache[ncache].lanes = lanes;
	cache[ncache].target = t->name;
	cache[ncache].r = *r;
	cache[ncache].ok = ok;
	ncache++;
}

int
cf_synth_bytes(const struct cf_target *t, nid f, int slot, double bound,
    long steps, struct cf_synth_res *r)
{
	struct prob P;
	int ok;

	memset(r, 0, sizeof *r);
	if (cached(f, 1, t, r, &ok))
		return ok;
	memset(&P, 0, sizeof P);
	P.lanes = 1;
	P.f = f;
	P.slot = slot;
	P.ity = TY(K_UINT, 8);
	P.oty = TY(K_UINT, 8);
	plan_build(&P.pl, &f, 1, 0);
	ok = search(&P, t, bound, steps, r);
	plan_free(&P.pl);
	remember(f, 1, t, r, ok);
	return ok;
}

int
cf_synth_scalar(const struct cf_target *t, nid f, int slot, ty_t ity,
    double bound, long steps, struct cf_synth_res *r)
{
	struct prob P;
	int ok;
	ty_t oty = N.ty[f];

	memset(r, 0, sizeof *r);
	if (!TY_SIZED(ity) || !TY_SIZED(oty) || TYW(ity) % 8 || TYW(oty) % 8 ||
	    TYW(ity) > 64 || TYW(oty) > 64)
		return 0;
	if (cached(f, 0, t, r, &ok))
		return ok;
	memset(&P, 0, sizeof P);
	P.f = f;
	P.slot = slot;
	P.ity = ity;
	P.oty = oty;
	P.win = TYW(ity);
	plan_build(&P.pl, &f, 1, 0);
	ok = search(&P, t, bound, steps, r);
	plan_free(&P.pl);
	remember(f, 0, t, r, ok);
	return ok;
}

/* ---- results as nodes and as C ---- */

/* the program's output bytes from input bytes, with the catalogue's
 * semantics and the solved constants */
int
cf_synth_sem(const struct cf_synth_prog *p, const nid *in16, nid *out16)
{
	struct v16 in, out;
	nid hv[8], hb[8];
	int k, nh, nb;

	U8 = TY(K_UINT, 8);
	U64 = TY(K_UINT, 64);
	holes_of(p, &nh, &nb);
	for (k = 0; k < nh; k++)
		hv[k] = mk_const_i(U64, (int64_t)p->h[k]);
	for (k = 0; k < nb; k++)
		hb[k] = ku8(p->hb[k]);
	memcpy(in.b, in16, sizeof in.b);
	if (!run(p, &in, hv, hb, 0, &out))
		return 0;
	memcpy(out16, out.b, sizeof out.b);
	return 1;
}

static const char *
wname(const char *n128, int vb)
{
	static char buf[4][64];
	static int r;

	r = (r + 1) & 3;
	if (vb == 128)
		snprintf(buf[r], sizeof buf[r], "%s", n128);
	else
		snprintf(buf[r], sizeof buf[r], "_mm%d%s", vb, n128 + 3);
	if (vb > 128) {
		char *q = strstr(buf[r], "_si128");
		if (q)
			snprintf(q, 8, "_si%d", vb);
	}
	return buf[r];
}

/* a 128-bit constant repeated across the vector */
static void
kconst(struct sbuf *b, int vb, uint64_t lo, uint64_t hi)
{
	int k;

	if (vb == 128)
		sb_printf(b, "_mm_set_epi64x((long long)0x%llxull, (long long)"
		    "0x%llxull)", (unsigned long long)hi, (unsigned long long)lo);
	else if (vb == 256)
		sb_printf(b, "_mm256_set_epi64x((long long)0x%llxull, (long long)"
		    "0x%llxull, (long long)0x%llxull, (long long)0x%llxull)",
		    (unsigned long long)hi, (unsigned long long)lo,
		    (unsigned long long)hi, (unsigned long long)lo);
	else {
		sb_puts(b, "_mm512_set_epi64(");
		for (k = 0; k < 4; k++)
			sb_printf(b, "%s(long long)0x%llxull, (long long)0x%llxull",
			    k ? ", " : "", (unsigned long long)hi,
			    (unsigned long long)lo);
		sb_puts(b, ")");
	}
}

/*
 * The program as C statements at vector width vb, the input register in
 * `in`, the result in a register named out; names get the prefix pfx.
 */
void
cf_synth_code(const struct cf_synth_prog *p, int vb, const char *in,
    const char *pfx, const char *out, struct sbuf *b)
{
	const char *vt = vb == 128 ? "__m128i" : vb == 256 ? "__m256i" :
	    "__m512i";
	int s, h = 0, hb = 0;
	char nm[SY_MAXLEN][32], an[32], bn[32];

	for (s = 0; s < p->n; s++) {
		const struct cf_synth_step *st = &p->s[s];
		struct sbuf k = {0}, k2 = {0};
		snprintf(nm[s], sizeof nm[s], "%s%d", pfx, s);
		if (s == p->n - 1)
			snprintf(nm[s], sizeof nm[s], "%s", out);
		snprintf(an, sizeof an, "%s", st->a < 0 ? in : nm[st->a]);
		snprintf(bn, sizeof bn, "%s", st->b < 0 ? in : nm[st->b]);
		switch (st->tpl) {
		case TP_GF2AFF:
			kconst(&k, vb, p->h[h], p->h[h]);
			sb_printf(b, "%s %s = %s(%s, %s, %d);\n", vt, nm[s],
			    wname("_mm_gf2p8affine_epi64_epi8", vb), an, k.p,
			    p->hb[hb]);
			h++;
			hb++;
			break;
		case TP_GF2MUL: {
			uint64_t q = 0x0101010101010101ULL * (uint64_t)p->hb[hb];
			kconst(&k, vb, q, q);
			sb_printf(b, "%s %s = %s(%s, %s);\n", vt, nm[s],
			    wname("_mm_gf2p8mul_epi8", vb), an, k.p);
			hb++;
			break;
		}
		case TP_PSHUFB:
			kconst(&k, vb, p->h[h], p->h[h + 1]);
			sb_printf(b, "%s %s = %s(%s, %s);\n", vt, nm[s],
			    wname("_mm_shuffle_epi8", vb), k.p, an);
			h += 2;
			break;
		case TP_NIB:
			kconst(&k, vb, 0x0f0f0f0f0f0f0f0fULL, 0x0f0f0f0f0f0f0f0fULL);
			sb_printf(b, "%s %sl = %s(%s, %s);\n", vt, nm[s],
			    wname("_mm_and_si128", vb), an, k.p);
			sb_printf(b, "%s %sh = %s(%s(%s, 4), %s);\n", vt, nm[s],
			    wname("_mm_and_si128", vb), wname("_mm_srli_epi16", vb), an,
			    k.p);
			sb_clear(&k);
			kconst(&k, vb, p->h[h], p->h[h + 1]);
			kconst(&k2, vb, p->h[h + 2], p->h[h + 3]);
			sb_printf(b, "%s %s = %s(%s(%s, %sl), %s(%s, %sh));\n", vt, nm[s],
			    wname(nibcomb[st->k], vb), wname("_mm_shuffle_epi8", vb), k.p,
			    nm[s], wname("_mm_shuffle_epi8", vb), k2.p, nm[s]);
			h += 4;
			break;
		case TP_ANDK:
		case TP_ORK:
		case TP_XORK:
		case TP_ADDK:
			kconst(&k, vb, p->h[h], p->h[h]);
			sb_printf(b, "%s %s = %s(%s, %s);\n", vt, nm[s],
			    wname(T[st->tpl].name, vb), an, k.p);
			h++;
			break;
		case TP_SRLI16:
		case TP_SLLI16:
		case TP_SRLI32:
		case TP_SLLI32:
		case TP_ROLD:
			sb_printf(b, "%s %s = %s(%s, %d);\n", vt, nm[s],
			    wname(T[st->tpl].name, vb), an, st->k);
			break;
		case TP_TERN:
			kconst(&k, vb, p->h[h], p->h[h]);
			sb_printf(b, "%s %s = %s(%s, %s, %s, 0x%02x);\n", vt, nm[s],
			    wname(T[st->tpl].name, vb), an, bn, k.p, p->hb[hb]);
			h++;
			hb++;
			break;
		default:
			sb_printf(b, "%s %s = %s(%s, %s);\n", vt, nm[s],
			    wname(T[st->tpl].name, vb), an, bn);
		}
		free(k.p);
		free(k2.p);
	}
}

/* a short description: "pshufb nibble pair (add)" ... */
void
cf_synth_text(const struct cf_synth_prog *p, struct sbuf *b)
{
	int s;

	for (s = 0; s < p->n; s++) {
		const struct cf_synth_step *st = &p->s[s];
		const char *n = T[st->tpl].name + 4;
		if (s)
			sb_puts(b, "; ");
		if (st->tpl == TP_NIB)
			sb_printf(b, "nibble LUT pair (two pshufb, %s)", nibcomb[st->k] + 4);
		else if (T[st->tpl].nk)
			sb_printf(b, "%s %d", n, st->k);
		else
			sb_puts(b, n);
		if (st->a >= 0 || (T[st->tpl].arity >= 2))
			sb_printf(b, " (of %s%s%s)", st->a < 0 ? "x" : "t",
			    T[st->tpl].arity >= 2 ? (st->b < 0 ? ", x" : ", t") : "",
			    "");
	}
}

uint64_t
cf_synth_feat(const struct cf_synth_prog *p)
{
	uint64_t f = 0;
	int s;

	for (s = 0; s < p->n; s++)
		f |= T[p->s[s].tpl].feat;
	return f;
}

void
cf_synth_reset(void)
{
	ncache = 0;
}

double
cf_synth_names(const struct cf_synth_prog *p, int vb, int ua, uint64_t *xneed)
{
	double c = 0;
	int s, id, k;

	*xneed = 0;
	for (s = 0; s < p->n; s++) {
		const char *nm[4] = {T[p->s[s].tpl].name, NULL, NULL, NULL};
		int cnt = T[p->s[s].tpl].ops;
		if (p->s[s].tpl == TP_NIB) {
			nm[1] = "_mm_and_si128";
			nm[2] = "_mm_srli_epi16";
			nm[3] = nibcomb[p->s[s].k];
		}
		for (k = 0; k < 4; k++) {
			const struct xcost *xc;
			if (!nm[k])
				continue;
			id = xi_find(wname(nm[k], vb));
			if (id < 0)
				continue;
			*xneed |= xi_feat(id);
			xc = xcost_get(ua, xi_cls(id), xi_vw(id));
			c += (xc ? xc->rtp : 1) * (k == 0 ? (p->s[s].tpl == TP_NIB ? 2 :
			    1) : k == 1 ? 2 : 1);
		}
		(void)cnt;
	}
	return c;
}
