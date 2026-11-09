/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * cfx86.c: the x86 side of cfront (cfront.md, F2): the loop vectoriser.
 *
 * cf_optimize offers every counted for loop of a function with memory as
 * a struct cf_loop (cfront.h).  vec_loop turns it into
 *
 *   { init;
 *   #if <the features the vector code needs>
 *     if (<the arrays do not overlap>)       only when the IR cannot say
 *       for (; i < hi && hi - i >= VL; i += VL) { vector body }
 *   #endif
 *     for (; i < hi; i += 1) { the scalar body, rebuilt from the IR } }
 *
 * The vector body computes each store's value (and each reduction's
 * operand) with one intrinsic per node of the scalar value, in lanes of
 * W bits (W from the narrowest that works: u8 data computed in u16 lanes
 * when the scalar code's int arithmetic needs more than 8 bits), loads
 * widened and stores narrowed as needed.
 *
 * Proof (CF_PROOF_LANES).  Every vector value carries its lanes as ucas
 * nodes, built from the intrinsics catalogue's lane semantics (x86intr.c,
 * validated against cc by x86check) composed operand by operand; loads
 * give lane j the element p[i + c + j] (a fresh input per lane: data
 * moving between lanes would show), stores write lane j to p[i + c + j].
 * For each lane j the stored lane is proved equal to the scalar store
 * value of iteration i + j (the scalar node with i := i + j and each load
 * replaced by that lane's element): equal DAGs, or exhaustively, or by
 * SAT (equiv_decide).  Bitwise si-intrinsics act on every lane view
 * (and, or, xor, andnot commute with cutting a word into fields).  The
 * remainder loop is the scalar code; the vector loop runs only while
 * hi - i >= VL, so together they run every iteration once, in order of
 * blocks.  Order: within an iteration block all loads are issued before
 * the stores that follow them in the scalar body; that keeps the
 * scalar's values when no store of one iteration feeds a load of a later
 * one: a written array is read only at the element it writes (or later
 * ones), and different arrays are disjoint (restrict / distinct arrays
 * from the IR, or the run-time overlap test).
 *
 * Integer reductions acc = acc op g (op + - ^ | & min max) keep VL
 * partial results and combine them after the loop: exact, because op is
 * associative and commutative mod 2^n (signed sums are computed
 * unsigned; where C's sum is defined it is the sum mod 2^n).  Float
 * reductions are refused: reassociation changes the rounding.
 */
#include "cfront.h"
#include "x86.h"

#include <stdlib.h>
#include <string.h>

#define MAXVV 512
#define MAXL 64
#define LSLOT 2000		/* per-lane load inputs: LSLOT + k*64 + j */
#define ISLOT 1900		/* the index at the block start */

struct vv {
	char kind;		/* 'i' integer lanes, 'm' mask lanes, 'f' float */
	int w;			/* bits per lane */
	int n;			/* lanes */
	int zb, sb;		/* the lanes are zero- (sign-) extended from zb
				 * (sb) bits; 0 unknown (an optimisation only:
				 * the proof checks the result) */
	nid lane[MAXL];
	char nm[12];
};

static struct {
	const struct cf_loop *l;
	const struct cf_target *t;
	int ua;			/* the µarch whose costs price the code */
	int VB, W, VL, flt;	/* vector bits, lane bits, lanes, float */
	struct vv v[MAXVV];
	int nv;
	nid memo_n[MAXVV];
	int memo_v[MAXVV];
	int nmemo;
	int ldv[CF_MAXMEM];	/* each load's vector, -1 */
	xfeat need;		/* features the vector code uses */
	struct sbuf body;	/* the vector loop's body */
	struct sbuf pre;	/* before the vector loop */
	double cost;		/* reciprocal throughput sum */
	char why[160];
	int ntmp;
	nid ivar;		/* the index at the block start */
} X;

static void
why(const char *fmt, const char *a)
{
	if (!X.why[0])
		snprintf(X.why, sizeof X.why, fmt, a ? a : "");
}

static const char *
pfx(void)
{
	return X.VB == 512 ? "_mm512" : X.VB == 256 ? "_mm256" : "_mm";
}

static const char *
vtype(char kind, int w)
{
	static const char *const ti[] = {"__m128i", "__m256i", "__m512i"};
	static const char *const tf[] = {"__m128", "__m256", "__m512"};
	static const char *const td[] = {"__m128d", "__m256d", "__m512d"};
	int k = X.VB == 512 ? 2 : X.VB == 256 ? 1 : 0;

	if (kind == 'f')
		return w == 32 ? tf[k] : td[k];
	return ti[k];
}

static const char *
si(void)
{
	return X.VB == 512 ? "si512" : X.VB == 256 ? "si256" : "si128";
}

/* the C type of an element */
static const char *
ctype_of(ty_t t)
{
	int w = TYW(t);

	if (TYK(t) == K_FLOAT)
		return w == 32 ? "float" : "double";
	if (TYK(t) == K_UINT || t == TY_BOOL)
		return w <= 8 ? "uint8_t" : w <= 16 ? "uint16_t" : w <= 32 ?
		    "uint32_t" : "uint64_t";
	return w <= 8 ? "int8_t" : w <= 16 ? "int16_t" : w <= 32 ? "int32_t" :
	    "int64_t";
}

static int
new_vv(char kind, int w, int n)
{
	struct vv *v;

	if (X.nv == MAXVV) {
		why("too many vector values", NULL);
		return -1;
	}
	v = &X.v[X.nv];
	memset(v, 0, sizeof *v);
	v->kind = kind;
	v->w = w;
	v->n = n;
	snprintf(v->nm, sizeof v->nm, "cf_v%d", X.nv);
	return X.nv++;
}

static int same_val(nid a, nid b);
static int memo_get(nid n);
static int memo_put(nid n, int v);
static int t_has(int xf);

/* ---- applying an intrinsic: its code and its lanes ---- */

struct opctx {
	int id;
	const int *arg;		/* vector operands (vv index) */
	const nid *scal;	/* scalar operands */
};

/*
 * Lane j of v seen as w-bit lanes: a register is the same bits under any
 * lane width (little endian: lane j of width w is bits j*w .. j*w+w-1),
 * so a wider lane concatenates narrower ones and a narrower one is a
 * field of a wider one.
 */
static nid
vlane(int v, int j, int w)
{
	const struct vv *x = &X.v[v];
	ty_t xu = TY(K_UINT, x->w);
	nid r = 0, p;
	int m, q;

	if (x->w == w) {
		if (j >= x->n)
			fail("cfx86: lane %d of a %d-lane value", j, x->n);
		return x->lane[j];
	}
	if (w > x->w) {
		ty_t ut = TY(K_UINT, w);
		m = w / x->w;
		if ((j + 1) * m > x->n)
			fail("cfx86: lane %d (%d bits) of a %d-lane value", j, w, x->n);
		for (q = 0; q < m; q++) {
			nid e = x->lane[j * m + q];
			e = TYK(N.ty[e]) == K_FLOAT ? mk_un(O_ASUINT, e) : mk_cast(xu, e);
			p = mk_cast(ut, e);
			if (q)
				p = mk_bin(O_SHL, p, mk_const_i(ut, q * x->w));
			r = r ? mk_bin(O_OR, r, p) : p;
		}
		return r;
	}
	m = x->w / w;
	if (j / m >= x->n)
		fail("cfx86: lane %d (%d bits) of a %d-lane value", j, w, x->n);
	p = x->lane[j / m];
	p = TYK(N.ty[p]) == K_FLOAT ? mk_un(O_ASUINT, p) : mk_cast(xu, p);
	if ((j % m) * w)
		p = mk_bin(O_SHR, p, mk_const_i(xu, (j % m) * w));
	return mk_cast(TY(K_UINT, w), p);
}

static nid
op_lane(void *c, int k, int lane)
{
	struct opctx *o = c;
	ty_t t = xi_otype(o->id, k);

	if (xi_okind(o->id, k) == 'v') {
		const struct vv *v = &X.v[o->arg[k]];
		nid e;
		if (TYK(t) == K_FLOAT && v->kind == 'f' && v->w == TYW(t))
			return v->lane[lane];
		e = vlane(o->arg[k], lane, TYW(t));
		if (TYK(t) == K_FLOAT)
			return TYK(N.ty[e]) == K_FLOAT ? e : mk_un(O_ASFLOAT,
			    mk_cast(TY(K_UINT, TYW(t)), e));
		return mk_cast(t, e);
	}
	return mk_cast(t, o->scal[k]);
}

/* the cost of an intrinsic on the µarch */
static void
price(int id)
{
	const struct xcost *c = xcost_get(X.ua, xi_cls(id), xi_vw(id));

	X.cost += c ? c->rtp : 1;
}

/*
 * name(args): a new value; args[k] is a vv index for a vector operand,
 * text[k] the C text of a scalar or immediate operand, scal[k] its node
 */
static int
vop(const char *name, char kind, const int *args, const char *const *text,
    const nid *scal, int64_t imm)
{
	int id = xi_find(name), r, k, n;
	struct opctx o;
	ty_t rt;

	if (id < 0) {
		why("%s is not in the intrinsics catalogue", name);
		return -1;
	}
	n = xi_lanes(id);
	rt = xi_rtype(id);
	if ((r = new_vv(kind, TYW(rt), n)) < 0)
		return -1;
	o.id = id;
	o.arg = args;
	o.scal = scal;
	for (k = 0; k < n && k < MAXL; k++)
		X.v[r].lane[k] = xi_lane(id, k, imm, op_lane, &o);
	X.need |= xi_feat(id);
	price(id);
	{
		int bits = n * TYW(rt), vb = X.VB;
		X.VB = bits <= 128 ? 128 : bits <= 256 ? 256 : 512;
		sb_printf(&X.body, "\t\t%s %s = %s(", vtype(kind, TYW(rt)),
		    X.v[r].nm, name);
		X.VB = vb;
	}
	for (k = 0; k < xi_nops(id); k++) {
		const char *s = "";
		switch (xi_okind(id, k)) {
		case 'v':
			s = X.v[args[k]].nm;
			break;
		default:
			s = text[k];
		}
		sb_printf(&X.body, "%s%s", k ? ", " : "", s);
	}
	sb_puts(&X.body, ");\n");
	return r;
}

static int
vop1(const char *op, const char *sfx, char kind, int a)
{
	char nm[64];
	int args[1];

	args[0] = a;
	snprintf(nm, sizeof nm, "%s_%s_%s", pfx(), op, sfx);
	return vop(nm, kind, args, NULL, NULL, 0);
}

static int
vop2(const char *op, const char *sfx, char kind, int a, int b)
{
	char nm[64];
	int args[2];

	args[0] = a;
	args[1] = b;
	snprintf(nm, sizeof nm, "%s_%s_%s", pfx(), op, sfx);
	return vop(nm, kind, args, NULL, NULL, 0);
}

/* a shift by an immediate */
static int
vshift(const char *op, int a, int amount)
{
	char nm[64], im[16];
	int args[2];
	const char *text[2];
	nid scal[2];

	args[0] = a;
	args[1] = -1;
	snprintf(im, sizeof im, "%d", amount);
	text[0] = NULL;
	text[1] = im;
	scal[0] = scal[1] = 0;
	snprintf(nm, sizeof nm, "%s_%s_epi%d", pfx(), op, X.W);
	return vop(nm, 'i', args, text, scal, amount);
}

/*
 * Bitwise and, or, xor, andnot on whole registers act on every lane view
 * (cutting a word into fields commutes with them), so their lanes are
 * built at this view directly.  The catalogue's si forms carry the same
 * semantics at 64-bit lanes (x86check).
 */
static int
vbit(int op, int a, int b)
{
	static const char *const nm[] = {"and", "or", "xor", "andnot"};
	char name[64];
	int r, j, id;
	struct vv *va = &X.v[a], *vb = &X.v[b];
	ty_t t = TY(K_SINT, X.W);

	snprintf(name, sizeof name, "%s_%s_%s", pfx(), nm[op], si());
	if ((id = xi_find(name)) < 0) {
		why("%s is not in the intrinsics catalogue", name);
		return -1;
	}
	if ((r = new_vv(va->kind == 'm' && vb->kind == 'm' ? 'm' : 'i', X.W,
	    X.VL)) < 0)
		return -1;
	for (j = 0; j < X.VL; j++) {
		nid x = mk_cast(t, vlane(a, j, X.W)), y = mk_cast(t, vlane(b, j,
		    X.W));
		X.v[r].lane[j] = op == 0 ? mk_bin(O_AND, x, y) : op == 1 ?
		    mk_bin(O_OR, x, y) : op == 2 ? mk_bin(O_XOR, x, y) :
		    mk_bin(O_AND, mk_un(O_NOT, x), y);
	}
	X.need |= xi_feat(id);
	price(id);
	sb_printf(&X.body, "\t\t%s %s = %s(%s, %s);\n", vtype('i', X.W),
	    X.v[r].nm, name, va->nm, vb->nm);
	if (op == 0 && (va->zb || vb->zb))
		X.v[r].zb = !va->zb ? vb->zb : !vb->zb ? va->zb : va->zb < vb->zb ?
		    va->zb : vb->zb;
	if (op == 3 && vb->zb)
		X.v[r].zb = vb->zb;
	if (op == 1 && va->zb && vb->zb)
		X.v[r].zb = va->zb > vb->zb ? va->zb : vb->zb;
	return r;
}

/* set1: every lane the scalar node e, its C text c */
static int
vset1(nid e, const char *c)
{
	char nm[64], cast[160];
	int args[1] = {-1};
	const char *text[1];
	nid scal[1];
	ty_t t;

	if (X.flt) {
		snprintf(nm, sizeof nm, "%s_set1_%s", pfx(), X.flt == 32 ? "ps" :
		    "pd");
		t = X.flt == 32 ? TY_F32 : TY_F64;
		snprintf(cast, sizeof cast, "(%s)(%s)", X.flt == 32 ? "float" :
		    "double", c);
	} else {
		snprintf(nm, sizeof nm, "%s_set1_epi%d%s", pfx(), X.W,
		    X.W == 64 && X.VB < 512 ? "x" : "");
		t = TY(K_SINT, X.W);
		snprintf(cast, sizeof cast, "(%s)(%s)", ctype_of(t), c);
	}
	text[0] = cast;
	scal[0] = mk_cast(t, e);
	return vop(nm, X.flt ? 'f' : 'i', args, text, scal, 0);
}

static int
vconst(int64_t k)
{
	char c[48];
	ty_t t = TY(K_SINT, X.W);
	nid kn;
	int r;

	k = int_norm(t, k);
	kn = mk_const_i(t, k);
	if ((r = memo_get(kn)) >= 0)
		return r;
	snprintf(c, sizeof c, "%lldLL", (long long)k);
	if (k == INT64_MIN)
		snprintf(c, sizeof c, "(-9223372036854775807LL - 1)");
	r = vset1(kn, c);
	if (r >= 0 && k >= 0) {
		int b = 0;
		while (b < 64 && (k >> b))
			b++;
		X.v[r].zb = b;
	}
	return memo_put(kn, r);
}

/* ---- memory ---- */

/* index = i + c, through conversions that keep the value: c, or 0 */
static int
affine(nid e, int64_t *c)
{
	int islot = X.l->islot;

	while (N.op[e] == O_CAST && TY_ISINT(N.ty[e]) && TYW(N.ty[e]) >=
	    TYW(N.ty[N.a[e]]) && TY_ISINT(N.ty[N.a[e]]))
		e = N.a[e];
	if (N.op[e] == O_VAR && N.ki[e] == islot) {
		*c = 0;
		return 1;
	}
	if ((N.op[e] == O_ADD || N.op[e] == O_SUB) && N.op[N.a[e]] == O_VAR &&
	    N.ki[N.a[e]] == islot && N.op[N.b[e]] == O_CONST) {
		*c = N.op[e] == O_ADD ? N.ki[N.b[e]] : -N.ki[N.b[e]];
		return *c > -65 && *c < 65;
	}
	return 0;
}

static const char *
base_text(int b, int64_t c)
{
	static char buf[4][200];
	static int r;

	r = (r + 1) & 3;
	if (c)
		snprintf(buf[r], sizeof buf[r], "(%s) + (cf_i + (%lld))",
		    X.l->base[b], (long long)c);
	else
		snprintf(buf[r], sizeof buf[r], "(%s) + cf_i", X.l->base[b]);
	return buf[r];
}

/* load k: the raw elements (lane j: a fresh input per lane), widened to W */
static int
vload(int k)
{
	const struct cf_mem *m = &X.l->load[k];
	int E = m->esize * 8, r, j, f = X.W / E, nraw, nbytes;
	int64_t c;
	char nm[64], src[256];

	if (X.ldv[k] >= 0)
		return X.ldv[k];
	affine(m->index, &c);
	if (X.flt && (TYK(m->ety) != K_FLOAT || E != X.flt)) {
		why("a load of another type in float code", NULL);
		return -1;
	}
	if (!X.flt && TYK(m->ety) == K_FLOAT) {
		why("a float load in integer code", NULL);
		return -1;
	}
	nbytes = X.VB / 8 / f;	/* bytes this block reads */
	if (nbytes < 8) {
		why("a widening too wide for this vector size", NULL);
		return -1;
	}
	/* the raw register: nbytes of data, the rest zero (loadl) */
	{
		int rb = nbytes <= 8 ? 128 : nbytes * 8;
		const char *ld;
		nraw = rb / E;
		if ((r = new_vv(X.flt ? 'f' : 'i', E, nraw)) < 0)
			return -1;
		for (j = 0; j < nraw; j++) {
			char ln[96];
			if (j * E / 8 >= nbytes) {
				X.v[r].lane[j] = mk_const_i(m->ety, 0);
				continue;
			}
			snprintf(ln, sizeof ln, "%s[%s+%lld]", X.l->base[m->base],
			    X.l->iname, (long long)(c + j));
			X.v[r].lane[j] = mk_var((uint32_t)(LSLOT + k * 64 + j),
			    sym_cstr(ln), m->ety);
		}
		snprintf(src, sizeof src, "%s", base_text(m->base, c));
		if (X.flt)
			ld = X.flt == 32 ? (X.VB == 512 ? "_mm512_loadu_ps" : X.VB == 256 ?
			    "_mm256_loadu_ps" : "_mm_loadu_ps") : (X.VB == 512 ?
			    "_mm512_loadu_pd" : X.VB == 256 ? "_mm256_loadu_pd" :
			    "_mm_loadu_pd");
		else
			ld = nbytes == 8 ? "_mm_loadl_epi64" : rb == 128 ?
			    "_mm_loadu_si128" : rb == 256 ? "_mm256_loadu_si256" :
			    "_mm512_loadu_si512";
		{
			int id = xi_find(ld);
			if (id < 0) {
				why("%s is not in the intrinsics catalogue", ld);
				return -1;
			}
			X.need |= xi_feat(id);
			price(id);
		}
		sb_printf(&X.body, "\t\t%s %s = %s((const void *)(%s));\n",
		    X.flt ? vtype('f', E) : rb == 128 ? "__m128i" : rb == 256 ?
		    "__m256i" : "__m512i", X.v[r].nm, ld, src);
	}
	if (f > 1) {
		/* pmovzx / pmovsx */
		snprintf(nm, sizeof nm, "%s_cvtep%c%d_epi%d", pfx(),
		    TYK(m->ety) == K_UINT ? 'u' : 'i', E, X.W);
		{
			int args[1];
			args[0] = r;
			r = vop(nm, 'i', args, NULL, NULL, 0);
		}
		if (r < 0)
			return -1;
		if (TYK(m->ety) == K_UINT)
			X.v[r].zb = E;
		else
			X.v[r].sb = E;
	}
	if (f == 1 && !X.flt) {
		if (TYK(m->ety) == K_UINT)
			X.v[r].zb = E;
		else
			X.v[r].sb = E;
	}
	if (X.v[r].n != X.VL) {
		why("a load that does not fill the lanes", NULL);
		return -1;
	}
	X.ldv[k] = r;
	return r;
}

/* the lanes of v narrowed to E bits (truncation), in a register whose
 * first VL lanes are the elements; *bytes: what to store */
static int
vnarrow(int v, int E)
{
	int r = v, m;

	if (X.W == E)
		return v;
	if (X.VB == 512) {
		char nm[64];
		int args[1];
		snprintf(nm, sizeof nm, "_mm512_cvtepi%d_epi%d", X.W, E);
		args[0] = v;
		return vop(nm, 'i', args, NULL, NULL, 0);
	}
	/* SSE / AVX2: mask, then pack pairs of 128-bit halves */
	if (X.v[v].zb && X.v[v].zb <= E)
		r = v;
	else {
		if ((m = vconst(E == 8 ? 0xff : E == 16 ? 0xffff : 0xffffffffLL)) < 0)
			return -1;
		if ((r = vbit(0, v, m)) < 0)
			return -1;
	}
	{
		int lo = r, hi = r, args[2];
		const char *text[2];
		nid scal[2];
		char nm[64];
		int w = X.W;
		if (X.VB == 256) {
			text[0] = NULL;
			text[1] = "0";
			scal[0] = scal[1] = 0;
			args[0] = r;
			args[1] = -1;
			lo = vop("_mm256_extracti128_si256", 'i', args, text, scal, 0);
			text[1] = "1";
			hi = vop("_mm256_extracti128_si256", 'i', args, text, scal, 1);
			if (lo < 0 || hi < 0)
				return -1;
		}
		/* now 128-bit pieces; pack until E */
		while (w > E) {
			int vb = X.VB;
			X.VB = 128;
			snprintf(nm, sizeof nm, "_mm_packus_epi%d", w);
			args[0] = lo;
			args[1] = hi;
			r = vop(nm, 'i', args, NULL, NULL, 0);
			X.VB = vb;
			if (r < 0)
				return -1;
			lo = hi = r;
			w /= 2;
		}
		return r;
	}
}

/* ---- the scalar node DAG to vector values ---- */

static int tr(nid n);

static int
memo_get(nid n)
{
	int k;

	for (k = 0; k < X.nmemo; k++)
		if (X.memo_n[k] == n)
			return X.memo_v[k];
	return -1;
}

static int
memo_put(nid n, int v)
{
	if (v >= 0 && X.nmemo < MAXVV) {
		X.memo_n[X.nmemo] = n;
		X.memo_v[X.nmemo++] = v;
	}
	return v;
}

static int
is_signed(ty_t t)
{
	return TYK(t) == K_SINT;
}

/* an integer value as W-bit lanes holding it mod 2^W; casts to narrower
 * types truncate and re-extend */
static int
tr_cast(nid n)
{
	ty_t to = N.ty[n], from = N.ty[N.a[n]];
	int a, wt, m;

	if (from == TY_BOOL || TYK(from) == K_BOOL) {
		/* a mask to 0 / 1 */
		if ((a = tr(N.a[n])) < 0 || (m = vconst(1)) < 0)
			return -1;
		return vbit(0, a, m);
	}
	if (!TY_ISINT(to) || !TY_ISINT(from) || TY_FP(to) || TY_FP(from)) {
		why("a conversion between integer and float", NULL);
		return -1;
	}
	if ((a = tr(N.a[n])) < 0)
		return -1;
	wt = TYW(to) < TYW(from) ? TYW(to) : TYW(from);
	if (wt >= X.W)
		return a;
	/* the value is the low wt bits, extended by the signedness of the
	 * narrower type */
	if (!is_signed(TYW(to) < TYW(from) ? to : from)) {
		if (X.v[a].zb && X.v[a].zb <= wt)
			return a;	/* already zero above bit wt */
		if ((m = vconst((int64_t)((1ULL << wt) - 1))) < 0)
			return -1;
		return vbit(0, a, m);
	}
	if (X.v[a].sb && X.v[a].sb <= wt)
		return a;		/* already sign-extended from wt bits */
	if (X.W == 8 || (X.W == 64 && X.VB != 512)) {
		why("a sign extension at this lane width", NULL);
		return -1;
	}
	if ((a = vshift("slli", a, X.W - wt)) < 0)
		return -1;
	return vshift("srai", a, X.W - wt);
}

/* a comparison as a mask (all ones / zero lanes) */
static int
tr_cmp(nid n)
{
	int op = N.op[n], a, b, s, r, sb;
	ty_t t = N.ty[N.a[n]];
	char sfx[16];

	if (X.VB == 512) {
		why("a comparison at 512 bits (mask registers)", NULL);
		return -1;
	}
	if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
		return -1;
	snprintf(sfx, sizeof sfx, "epi%d", X.W);
	if (op == O_EQ || op == O_NE) {
		r = vop2("cmpeq", sfx, 'm', a, b);
		if (r >= 0 && op == O_NE) {
			int ones = vconst(-1);
			r = ones < 0 ? -1 : vbit(2, r, ones);
		}
		return r;
	}
	if (!is_signed(t) || TYW(t) > X.W) {
		/* unsigned: flip the sign bits, then signed */
		if ((sb = vconst((int64_t)(1ULL << (X.W - 1)))) < 0)
			return -1;
		if ((a = vbit(2, a, sb)) < 0 || (b = vbit(2, b, sb)) < 0)
			return -1;
	}
	/* a < b is b > a; a <= b is !(a > b) */
	switch (op) {
	case O_GT: r = vop2("cmpgt", sfx, 'm', a, b); break;
	case O_LT: r = vop2("cmpgt", sfx, 'm', b, a); break;
	case O_LE: r = vop2("cmpgt", sfx, 'm', a, b); s = 1; goto neg;
	case O_GE: r = vop2("cmpgt", sfx, 'm', b, a); s = 1; goto neg;
	default:
		why("a comparison", NULL);
		return -1;
	}
	return r;
neg:
	(void)s;
	if (r < 0)
		return -1;
	{
		int ones = vconst(-1);
		return ones < 0 ? -1 : vbit(2, r, ones);
	}
}

/* the same value up to integer conversions (a guess; the proof decides) */
static int
same_val(nid a, nid b)
{
	while (N.op[a] == O_CAST && TY_ISINT(N.ty[a]) && TY_ISINT(N.ty[N.a[a]]))
		a = N.a[a];
	while (N.op[b] == O_CAST && TY_ISINT(N.ty[b]) && TY_ISINT(N.ty[N.a[b]]))
		b = N.a[b];
	if (a == b)
		return 1;
	return N.op[a] == O_CONST && N.op[b] == O_CONST && TY_ISINT(N.ty[a]) &&
	    TY_ISINT(N.ty[b]) && N.ki[a] == N.ki[b];
}

/* x op K == K ? K : x, K ? ... : the clamps as min / max */
static int
tr_sel(nid n)
{
	nid c = N.a[n], x = N.b[n], y = N.c[n];
	int m, a, b, t, u;

	/* (int)(cond) != 0 is cond */
	while (N.op[c] == O_NE && N.op[N.b[c]] == O_CONST && N.ki[N.b[c]] == 0 &&
	    N.op[N.a[c]] == O_CAST && (N.ty[N.a[N.a[c]]] == TY_BOOL ||
	    TYK(N.ty[N.a[N.a[c]]]) == K_BOOL))
		c = N.a[N.a[c]];

	/* min / max when the arms are the comparison's operands */
	if (N.op[c] >= O_LT && N.op[c] <= O_GE && N.op[c] != O_EQ &&
	    N.op[c] != O_NE) {
		nid p = N.a[c], q = N.b[c];
		int op = N.op[c], sgn = is_signed(N.ty[p]);
		const char *f = NULL;
		if (same_val(x, p) && same_val(y, q))
			f = op == O_GT || op == O_GE ? "max" : "min";
		else if (same_val(x, q) && same_val(y, p))
			f = op == O_GT || op == O_GE ? "min" : "max";
		if (f && TYW(N.ty[p]) <= 64) {
			char sfx[16];
			snprintf(sfx, sizeof sfx, "ep%c%d", sgn ? 'i' : 'u', X.W);
			if ((a = tr(p)) >= 0 && (b = tr(q)) >= 0) {
				int r = vop2(f, sfx, 'i', a, b);
				if (r >= 0 && !sgn && f[1] == 'i' && (X.v[a].zb ||
				    X.v[b].zb))
					X.v[r].zb = !X.v[a].zb ? X.v[b].zb : !X.v[b].zb ?
					    X.v[a].zb : X.v[a].zb < X.v[b].zb ? X.v[a].zb :
					    X.v[b].zb;
				if (r >= 0)
					return r;
			}
			X.why[0] = 0;	/* try the general form */
		}
	}
	if ((m = tr(c)) < 0 || (a = tr(x)) < 0 || (b = tr(y)) < 0)
		return -1;
	if (X.VB == 128 ? t_has(XF_SSE41) : X.VB == 256 ? t_has(XF_AVX2) : 0) {
		/* blendv picks by each byte's top bit: a mask's bytes are all
		 * ones or all zeros (the proof checks the lanes) */
		int args[3], r;
		char nm[64];
		args[0] = b;
		args[1] = a;
		args[2] = m;
		snprintf(nm, sizeof nm, "%s_blendv_epi8", pfx());
		r = vop(nm, 'i', args, NULL, NULL, 0);
		if (r >= 0) {
			if (X.v[a].zb && X.v[b].zb)
				X.v[r].zb = X.v[a].zb > X.v[b].zb ? X.v[a].zb : X.v[b].zb;
			return r;
		}
		X.why[0] = 0;
	}
	if ((t = vbit(0, m, a)) < 0 || (u = vbit(3, m, b)) < 0)
		return -1;
	return vbit(1, t, u);
}

/* e = x + y + 1 (any association), x and y W-bit unsigned values:
 * their vectors (a guess for pavg; the proof checks it) */
static int
avg_operands(nid e, int *a, int *b)
{
	nid t[3], x;
	int nt = 0, k, one = -1;

	while (N.op[e] == O_CAST && TY_ISINT(N.ty[e]))
		e = N.a[e];
	if (N.op[e] != O_ADD)
		return 0;
	t[nt++] = N.b[e];
	x = N.a[e];
	while (N.op[x] == O_CAST && TY_ISINT(N.ty[x]))
		x = N.a[x];
	if (N.op[x] != O_ADD)
		return 0;
	t[nt++] = N.a[x];
	t[nt++] = N.b[x];
	for (k = 0; k < 3; k++)
		if (N.op[t[k]] == O_CONST && N.ki[t[k]] == 1)
			one = k;
	if (one < 0)
		return 0;
	{
		nid p = t[(one + 1) % 3], q = t[(one + 2) % 3];
		int va, vb;
		if ((va = tr(p)) < 0 || (vb = tr(q)) < 0)
			return 0;
		if (!X.v[va].zb || X.v[va].zb > X.W || !X.v[vb].zb ||
		    X.v[vb].zb > X.W)
			return 0;
		*a = va;
		*b = vb;
	}
	return 1;
}

static int
tr_int(nid n)
{
	int op = N.op[n], a, b, r;
	ty_t t = N.ty[n];
	char sfx[16];
	int64_t k;

	snprintf(sfx, sizeof sfx, "epi%d", X.W);
	switch (op) {
	case O_CONST:
		return vconst(N.ki[n]);
	case O_CAST:
		return tr_cast(n);
	case O_ADD: case O_SUB:
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
			return -1;
		return vop2(op == O_ADD ? "add" : "sub", sfx, 'i', a, b);
	case O_MUL:
		/* by a constant 2^s, 2^s + 1, 2^s - 1: shifts and adds */
		if (X.W > 8 && (N.op[N.b[n]] == O_CONST || N.op[N.a[n]] == O_CONST)) {
			nid kn = N.op[N.b[n]] == O_CONST ? N.b[n] : N.a[n];
			nid xn = kn == N.b[n] ? N.a[n] : N.b[n];
			uint64_t kk = (uint64_t)int_norm(TY(K_UINT, X.W), N.ki[kn]);
			int sh;
			for (sh = 1; sh < X.W - 1; sh++) {
				uint64_t p2 = 1ULL << sh;
				if (kk != p2 && kk != p2 + 1 && kk != p2 - 1)
					continue;
				if ((a = tr(xn)) < 0 || (b = vshift("slli", a, sh)) < 0)
					return -1;
				if (kk == p2)
					return b;
				return vop2(kk == p2 + 1 ? "add" : "sub", sfx, 'i', b, a);
			}
		}
		if (X.W == 8 || (X.W == 64 && X.VB != 512)) {
			why("a multiply at this lane width", NULL);
			return -1;
		}
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
			return -1;
		return vop2("mullo", sfx, 'i', a, b);
	case O_AND: case O_OR: case O_XOR:
		/* x & all-ones (mod 2^W) is x; so is x & (2^z - 1) when x is
		 * zero above bit z */
		if (op == O_AND && N.op[N.b[n]] == O_CONST) {
			uint64_t kk = (uint64_t)int_norm(TY(K_UINT, X.W), N.ki[N.b[n]]);
			if (kk == (uint64_t)int_norm(TY(K_UINT, X.W), -1))
				return tr(N.a[n]);
			if ((a = tr(N.a[n])) < 0)
				return -1;
			if (X.v[a].zb && X.v[a].zb < 64 &&
			    (kk & ((1ULL << X.v[a].zb) - 1)) ==
			    (1ULL << X.v[a].zb) - 1)
				return a;
		}
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
			return -1;
		return vbit(op == O_AND ? 0 : op == O_OR ? 1 : 2, a, b);
	case O_NOT:
		if ((a = tr(N.a[n])) < 0 || (b = vconst(-1)) < 0)
			return -1;
		return vbit(2, a, b);
	case O_NEG:
		if ((b = tr(N.a[n])) < 0 || (a = vconst(0)) < 0)
			return -1;
		return vop2("sub", sfx, 'i', a, b);
	case O_SHL: case O_SHR:
		/* (x + y + 1) >> 1 of two W-bit unsigned values: pavg */
		if (op == O_SHR && N.op[N.b[n]] == O_CONST && N.ki[N.b[n]] == 1 &&
		    X.W <= 16 && avg_operands(N.a[n], &a, &b)) {
			snprintf(sfx, sizeof sfx, "epu%d", X.W);
			r = vop2("avg", sfx, 'i', a, b);
			if (r >= 0)
				return r;
			X.why[0] = 0;
		}
		if (N.op[N.b[n]] != O_CONST || (k = N.ki[N.b[n]]) < 0 ||
		    k >= TYW(t)) {
			why("a shift by a variable or out-of-range amount", NULL);
			return -1;
		}
		if (X.W == 8) {
			why("an 8-bit shift", NULL);
			return -1;
		}
		if ((a = tr(N.a[n])) < 0)
			return -1;
		if (k >= X.W)
			return op == O_SHL ? vconst(0) : -1;
		if (op == O_SHR && is_signed(t) && X.W == 64 && X.VB != 512) {
			why("a 64-bit arithmetic shift below AVX-512", NULL);
			return -1;
		}
		return vshift(op == O_SHL ? "slli" : is_signed(t) ? "srai" : "srli",
		    a, (int)k);
	case O_DIV: case O_MOD:
		/* by a power of two, unsigned */
		if (N.op[N.b[n]] != O_CONST || is_signed(t) ||
		    (k = N.ki[N.b[n]]) <= 0 || (k & (k - 1))) {
			why("a division other than an unsigned one by 2^k", NULL);
			return -1;
		}
		if ((a = tr(N.a[n])) < 0)
			return -1;
		if (op == O_MOD) {
			if ((b = vconst(k - 1)) < 0)
				return -1;
			return vbit(0, a, b);
		}
		if (X.W == 8) {
			why("an 8-bit shift", NULL);
			return -1;
		}
		for (r = 0; (1LL << r) < k; r++)
			;
		return r ? vshift("srli", a, r) : a;
	case O_MIN: case O_MAX:
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
			return -1;
		snprintf(sfx, sizeof sfx, "ep%c%d", is_signed(t) ? 'i' : 'u', X.W);
		return vop2(op == O_MIN ? "min" : "max", sfx, 'i', a, b);
	case O_ABS:
		if ((a = tr(N.a[n])) < 0)
			return -1;
		return vop1("abs", sfx, 'i', a);
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE:
		return tr_cmp(n);
	case O_SEL:
		return tr_sel(n);
	case O_LNOT:
		if ((a = tr(N.a[n])) < 0 || (b = vconst(-1)) < 0)
			return -1;
		return vbit(2, a, b);
	case O_LAND: case O_LOR:
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
			return -1;
		return vbit(op == O_LAND ? 0 : 1, a, b);
	}
	why("an operation without a vector form here (%s)", opinfo[op].name);
	return -1;
}

static int
tr_flt(nid n)
{
	int op = N.op[n], a, b, c;
	const char *sfx = X.flt == 32 ? "ps" : "pd";

	if (TYK(N.ty[n]) != K_FLOAT || TYW(N.ty[n]) != X.flt) {
		why("a value of another type in float code", NULL);
		return -1;
	}
	switch (op) {
	case O_CONST:
		{
			char txt[64];
			snprintf(txt, sizeof txt, "%.17g", N.kr[n]);
			if (N.kr[n] != N.kr[n] || N.kr[n] - N.kr[n] != 0) {
				why("a NaN or infinite constant", NULL);
				return -1;
			}
			return vset1(n, txt);
		}
	case O_ADD: case O_SUB: case O_MUL: case O_DIV:
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0)
			return -1;
		return vop2(op == O_ADD ? "add" : op == O_SUB ? "sub" : op == O_MUL ?
		    "mul" : "div", sfx, 'f', a, b);
	case O_SQRT:
		if ((a = tr(N.a[n])) < 0)
			return -1;
		return vop1("sqrt", sfx, 'f', a);
	case O_FMA:
		if (!(X.t->feat & CF_F(CF_FMA))) {
			why("fma without the FMA feature", NULL);
			return -1;
		}
		if ((a = tr(N.a[n])) < 0 || (b = tr(N.b[n])) < 0 ||
		    (c = tr(N.c[n])) < 0)
			return -1;
		{
			int args[3];
			char nm[64];
			args[0] = a;
			args[1] = b;
			args[2] = c;
			snprintf(nm, sizeof nm, "%s_fmadd_%s", pfx(), sfx);
			return vop(nm, 'f', args, NULL, NULL, 0);
		}
	}
	why("a float operation without a vector form here (%s)",
	    opinfo[op].name);
	return -1;
}

static int
tr(nid n)
{
	int r, k, slot;

	if ((r = memo_get(n)) >= 0)
		return r;
	if (N.op[n] == O_VAR) {
		slot = (int)N.ki[n];
		if (slot == X.l->islot) {
			/* (i + j) in each lane: set1(i) + (0, 1, 2 ...) */
			int s, io, j, args[2];
			char nm[64];
			if (X.flt) {
				why("the index in float code", NULL);
				return -1;
			}
			if ((s = vset1(X.ivar, "cf_i")) < 0)
				return -1;
			if ((io = new_vv('i', X.W, X.VL)) < 0)
				return -1;
			for (j = 0; j < X.VL; j++)
				X.v[io].lane[j] = mk_const_i(TY(K_SINT, X.W), j);
			sb_printf(&X.pre, "\tstatic const %s cf_iota%d[%d] = {",
			    ctype_of(TY(K_SINT, X.W)), io, X.VL);
			for (j = 0; j < X.VL; j++)
				sb_printf(&X.pre, "%s%d", j ? ", " : "", j);
			sb_printf(&X.pre, "};\n\t(void)cf_iota%d;\n", io);
			sb_printf(&X.body, "\t\t%s %s = %s_loadu_%s((const void *)"
			    "cf_iota%d);\n", vtype('i', X.W), X.v[io].nm, pfx(), si(),
			    io);
			args[0] = s;
			args[1] = io;
			snprintf(nm, sizeof nm, "%s_add_epi%d", pfx(), X.W);
			return memo_put(n, vop(nm, 'i', args, NULL, NULL, 0));
		}
		for (k = 0; k < X.l->nload; k++)
			if (X.l->load[k].slot == slot)
				return memo_put(n, vload(k));
		for (k = 0; k < X.l->nred; k++)
			if (X.l->rslot[k] == slot) {
				why("a reduction variable read in the body", NULL);
				return -1;
			}
		/* a loop invariant */
		return memo_put(n, vset1(mk_var((uint32_t)slot,
		    sym_cstr(X.l->leaf[slot]), X.l->leafty[slot]),
		    X.l->leaf[slot]));
	}
	r = X.flt ? tr_flt(n) : tr_int(n);
	return memo_put(n, r);
}

/* ---- the lane proof ---- */

static void
collect_vars(nid n, uint8_t *seen, uint32_t *slots, int *ns, nid *vars)
{
	uint32_t i;

	if (n == 0 || seen[n])
		return;
	seen[n] = 1;
	if (N.op[n] == O_VAR) {
		if (*ns < 64) {
			slots[*ns] = (uint32_t)N.ki[n];
			vars[(*ns)++] = n;
		}
		return;
	}
	if (N.op[n] == O_TABLE) {
		for (i = 0; i < N.b[n]; i++)
			collect_vars(N.lst[N.a[n] + i], seen, slots, ns, vars);
		return;
	}
	collect_vars(N.a[n], seen, slots, ns, vars);
	collect_vars(N.b[n], seen, slots, ns, vars);
	collect_vars(N.c[n], seen, slots, ns, vars);
}

struct lproof {
	int lanes, dag, exh, sat;
	nid done_a[64], done_b[64];
	int ndone;
};

/* a == b for every value of their inputs: 1 proved, 0 not */
static int
prove_eq(nid a, nid b, struct lproof *lp)
{
	uint8_t *seen;
	uint32_t slots[64], maxs = 0;
	nid vars[64], *map, a2, b2;
	int ns = 0, k;
	struct func f;
	struct eqproof pr;

	if (a == b) {
		lp->dag++;
		return 1;
	}
	seen = xcalloc(N.n, 1);
	collect_vars(a, seen, slots, &ns, vars);
	collect_vars(b, seen, slots, &ns, vars);
	free(seen);
	if (ns > MAXP) {
		why("a lane with more than 8 inputs", NULL);
		return 0;
	}
	for (k = 0; k < ns; k++)
		if (slots[k] + 1 > maxs)
			maxs = slots[k] + 1;
	map = xcalloc(maxs + 1, sizeof *map);
	memset(&f, 0, sizeof f);
	f.name = sym_cstr("lane");
	f.anon = 1;
	f.np = ns;
	for (k = 0; k < ns; k++) {
		f.p[k].name = N.op[vars[k]] == O_VAR ? (uint32_t)sym_cstr(node_str(
		    vars[k])) : sym_cstr("x");
		f.p[k].ty = N.ty[vars[k]];
		map[slots[k]] = mk_var((uint32_t)k, f.p[k].name, f.p[k].ty);
	}
	a2 = subst(a, map, (int)maxs);
	b2 = subst(b, map, (int)maxs);
	free(map);
	/* the same obligation as an earlier lane's */
	for (k = 0; k < lp->ndone; k++)
		if (lp->done_a[k] == a2 && lp->done_b[k] == b2)
			return 1;
	if (a2 == b2) {
		lp->dag++;
		return 1;
	}
	f.body = a2;
	if (equiv_decide(&f, a2, b2, NULL, &pr) != 0 || pr.how == EQ_SAMPLED) {
		why("a lane not proved equal to the scalar value", NULL);
		return 0;
	}
	if (pr.how == EQ_SAT)
		lp->sat++;
	else
		lp->exh++;
	if (lp->ndone < 64) {
		lp->done_a[lp->ndone] = a2;
		lp->done_b[lp->ndone++] = b2;
	}
	return 1;
}

/* the scalar value e at iteration i + j, its loads lane j's inputs */
static nid
scalar_at(nid e, int j)
{
	const struct cf_loop *l = X.l;
	nid *map = xcalloc(NSLOT, sizeof *map), r;
	int k;

	map[l->islot] = mk_bin(O_ADD, X.ivar, mk_const_i(l->ity, j));
	for (k = 0; k < l->nload; k++) {
		const struct cf_mem *m = &l->load[k];
		int64_t c = 0;
		char ln[96];
		affine(m->index, &c);
		snprintf(ln, sizeof ln, "%s[%s+%lld]", l->base[m->base], l->iname,
		    (long long)(c + j));
		map[m->slot] = mk_var((uint32_t)(LSLOT + k * 64 + j), sym_cstr(ln),
		    m->ety);
	}
	for (k = 0; k < l->nleaf; k++)
		if (!map[k] && k != l->islot)
			map[k] = mk_var((uint32_t)k, sym_cstr(l->leaf[k]),
			    l->leafty[k]);
	r = subst(e, map, NSLOT);
	free(map);
	return r;
}

/* lane j of v (W bits) against the scalar s (E bits) */
static int
prove_lanes(int v, nid s, ty_t ety, struct lproof *lp)
{
	int j;
	ty_t ut = TYK(ety) == K_FLOAT ? ety : TY(K_UINT, TYW(ety));

	for (j = 0; j < X.VL; j++) {
		nid a = mk_cast(ut, scalar_at(s, j));
		nid b = mk_cast(ut, vlane(v, j, TYW(ety)));
		if (!prove_eq(a, b, lp))
			return 0;
		lp->lanes++;
	}
	return 1;
}

/* ---- the loop ---- */

/* acc op g: the operation and g, or 0 */
static int
red_split(nid e, int slot, nid *g)
{
	int op = N.op[e];
	nid a, b;

	while (op == O_CAST && TY_ISINT(N.ty[e]) && TYW(N.ty[e]) <=
	    TYW(N.ty[N.a[e]])) {
		e = N.a[e];
		op = N.op[e];
	}
	if (op == O_SEL) {
		/* acc = x > acc ? x : acc and the like: a max or a min, if
		 * the proof agrees (vec_loop proves rnext = op(acc, g)) */
		nid c = N.a[e], x = N.b[e], y = N.c[e], p, q;
		int co;
		while (N.op[c] == O_NE && N.op[N.b[c]] == O_CONST &&
		    N.ki[N.b[c]] == 0 && N.op[N.a[c]] == O_CAST)
			c = N.a[N.a[c]];
		co = N.op[c];
		if (co < O_LT || co > O_GE || co == O_EQ || co == O_NE)
			return 0;
		p = N.a[c];
		q = N.b[c];
		if (!(same_val(x, p) && same_val(y, q)) && !(same_val(x, q) &&
		    same_val(y, p)))
			return 0;
		if (N.op[x] == O_VAR && N.ki[x] == slot)
			*g = y;
		else if (N.op[y] == O_VAR && N.ki[y] == slot)
			*g = x;
		else {
			nid xs = x, ys = y;
			while (N.op[xs] == O_CAST)
				xs = N.a[xs];
			while (N.op[ys] == O_CAST)
				ys = N.a[ys];
			if (N.op[xs] == O_VAR && N.ki[xs] == slot)
				*g = y;
			else if (N.op[ys] == O_VAR && N.ki[ys] == slot)
				*g = x;
			else
				return 0;
		}
		/* x > y ? x : y is max(x, y); x < y ? x : y is min */
		if (same_val(x, p))
			return co == O_GT || co == O_GE ? O_MAX : O_MIN;
		return co == O_GT || co == O_GE ? O_MIN : O_MAX;
	}
	if (op != O_ADD && op != O_SUB && op != O_XOR && op != O_OR &&
	    op != O_AND && op != O_MIN && op != O_MAX)
		return 0;
	a = N.a[e];
	b = N.b[e];
	while (N.op[a] == O_CAST && TY_ISINT(N.ty[a]))
		a = N.a[a];
	(void)b;
	if (N.op[a] == O_VAR && N.ki[a] == slot) {
		*g = N.b[e];
		return op;
	}
	if (op != O_SUB && N.op[N.b[e]] == O_VAR && N.ki[N.b[e]] == slot) {
		*g = N.a[e];
		return op;
	}
	return 0;
}

static int
uses_slot(nid n, int slot, uint8_t *seen)
{
	uint32_t i;

	if (n == 0 || seen[n])
		return 0;
	seen[n] = 1;
	if (N.op[n] == O_VAR)
		return N.ki[n] == slot;
	if (N.op[n] == O_TABLE) {
		for (i = 0; i < N.b[n]; i++)
			if (uses_slot(N.lst[N.a[n] + i], slot, seen))
				return 1;
		return 0;
	}
	return uses_slot(N.a[n], slot, seen) || uses_slot(N.b[n], slot, seen) ||
	    uses_slot(N.c[n], slot, seen);
}

static int
node_uses(nid n, int slot)
{
	uint8_t *seen = xcalloc(N.n, 1);
	int r = uses_slot(n, slot, seen);

	free(seen);
	return r;
}

static const char *
cmacro(xfeat f, struct sbuf *b)
{
	int k, any = 0;

	for (k = 0; k < XF_N; k++)
		if (f & XB(k)) {
			sb_printf(b, "%sdefined(%s)", any++ ? " && " : "",
			    xfeat_macro[k]);
		}
	if (!any)
		sb_puts(b, "1");
	return b->p;
}

/* the target has the x86 feature */
static int
t_has(int xf)
{
	int f;

	for (f = 0; f < CF_NFEAT; f++)
		if (!strcmp(cf_feat_macro(f), xfeat_macro[xf]))
			return (X.t->feat & CF_F(f)) != 0;
	return 0;
}

static int
t_has_all(xfeat need)
{
	int k;

	for (k = 0; k < XF_N; k++)
		if ((need & XB(k)) && !t_has(k))
			return 0;
	return 1;
}

static int
pick_uarch(const struct cf_target *t)
{
	int ua = uarch_find(t->name);

	if (ua < 0 && t->march)
		ua = uarch_find(t->march);
	if (ua >= 0)
		return ua;
	return (t->feat & CF_F(CF_AVX512F)) ? UA_V4 : (t->feat & CF_F(CF_AVX2)) ?
	    UA_V3 : UA_V2;
}

struct vplan {
	int st[CF_MAXMEM];	/* each store's final vector */
	int rv[MAXST];		/* each reduction's g vector */
	int rop[MAXST];
	struct lproof lp;
};

/* one attempt at vector bits VB and lane bits W */
static int
attempt(int VB, int W, struct vplan *pl)
{
	const struct cf_loop *l = X.l;
	int k;

	X.VB = VB;
	X.W = W;
	X.VL = VB / W;
	X.nv = 0;
	X.nmemo = 0;
	X.need = 0;
	X.cost = 0;
	X.why[0] = 0;
	sb_clear(&X.body);
	sb_clear(&X.pre);
	for (k = 0; k < CF_MAXMEM; k++)
		X.ldv[k] = -1;
	memset(&pl->lp, 0, sizeof pl->lp);
	if (X.VL > MAXL)
		return 0;
	/* loads in their order, then the values */
	for (k = 0; k < l->nload; k++)
		if (vload(k) < 0)
			return 0;
	for (k = 0; k < l->nstore; k++) {
		int v = tr(l->store[k].value), E = l->store[k].esize * 8;
		if (v < 0)
			return 0;
		if (!X.flt && (v = vnarrow(v, E)) < 0)
			return 0;
		pl->st[k] = v;
	}
	for (k = 0; k < l->nred; k++) {
		nid g = 0;
		pl->rop[k] = red_split(l->rnext[k], l->rslot[k], &g);
		if ((pl->rv[k] = tr(g)) < 0)
			return 0;
	}
	if (!t_has_all(X.need)) {
		why("an intrinsic the target lacks", NULL);
		return 0;
	}
	/* the proof, lane by lane */
	for (k = 0; k < l->nstore; k++)
		if (!prove_lanes(pl->st[k], l->store[k].value, l->store[k].ety,
		    &pl->lp))
			return 0;
	for (k = 0; k < l->nred; k++) {
		nid g = 0, acc, step;
		ty_t rt = N.ty[l->rnext[k]];
		int op = red_split(l->rnext[k], l->rslot[k], &g);
		if (!prove_lanes(pl->rv[k], g, rt, &pl->lp))
			return 0;
		/* the step is acc op g (for the sel forms of min and max
		 * this is a proof obligation of its own) */
		acc = mk_var((uint32_t)l->rslot[k], sym_cstr(l->leaf[l->rslot[k]]),
		    l->leafty[l->rslot[k]]);
		step = mk_bin(op, mk_cast(rt, acc), mk_cast(rt, g));
		if (op == O_MIN || op == O_MAX) {
			struct lproof tmp;
			memset(&tmp, 0, sizeof tmp);
			if (!prove_eq(mk_cast(rt, scalar_at(l->rnext[k], 0)),
			    scalar_at(step, 0), &tmp))
				return 0;
		}
	}
	return 1;
}

static const char *
red_ident(int op, ty_t t)
{
	switch (op) {
	case O_AND: return "-1";
	case O_MIN: return TYK(t) == K_UINT ? "-1" : TYW(t) == 64 ?
	    "0x7fffffffffffffffLL" : TYW(t) == 32 ? "0x7fffffff" : TYW(t) == 16 ?
	    "0x7fff" : "0x7f";
	case O_MAX: return TYK(t) == K_UINT ? "0" : TYW(t) == 64 ?
	    "(-0x7fffffffffffffffLL - 1)" : TYW(t) == 32 ? "(-0x7fffffff - 1)" :
	    TYW(t) == 16 ? "(-0x7fff - 1)" : "(-0x7f - 1)";
	}
	return "0";
}

static int
vec_loop(const struct cf_loop *l, const struct cf_target *t,
    struct cf_loopout *out)
{
	static const int vbs[] = {512, 256, 128};
	struct vplan pl;
	struct cf_loop rem;
	struct cf_loopout ro;
	int k, j, vi, W, Wmin = 8, ok = 0, needcheck = 0;
	uint8_t written[CF_MAXMEM] = {0};
	int64_t off;
	struct sbuf chk = {0};
	int U, u;
	char rop_name[MAXST][64], comb_name[MAXST][64];
	const char *UT;
	double cb = 0;

	memset(&X, 0, sizeof X);
	X.l = l;
	X.t = t;
	X.ua = pick_uarch(t);
	snprintf(out->what, sizeof out->what, "not vectorised");
	if (l->step != 1 || !TY_ISINT(l->ity) || TYW(l->ity) < 32 ||
	    N.ty[l->hi] != l->ity || node_uses(l->hi, l->islot))
		return 0;
	for (k = 0; k < l->nload; k++)
		if (node_uses(l->hi, l->load[k].slot))
			return 0;
	for (k = 0; k < l->nred; k++)
		if (l->rslot[k] >= 0 && node_uses(l->hi, l->rslot[k]))
			return 0;
	/* memory: contiguous, and no store feeds a later iteration's load */
	for (k = 0; k < l->nstore; k++) {
		if (!affine(l->store[k].index, &off))
			return 0;
		if (written[l->store[k].base]++)
			return 0;	/* two stores to one array */
	}
	for (k = 0; k < l->nload; k++) {
		int s;
		if (!affine(l->load[k].index, &off))
			return 0;	/* strided, or a gather: not yet */
		for (s = 0; s < l->nstore; s++) {
			int64_t so;
			if (l->store[s].base != l->load[k].base)
				continue;
			affine(l->store[s].index, &so);
			/* a later element read before it is written is fine;
			 * an earlier one carries a dependence */
			if (off < so || l->load[k].epoch > l->store[s].epoch)
				return 0;	/* (a load after the store: refused,
						 * all loads are issued first) */
			if (l->load[k].esize != l->store[s].esize)
				return 0;
		}
	}
	for (k = 0; k < l->nbase; k++)
		if (written[k]) {
			for (j = 0; j < l->nbase; j++)
				if (j != k && !l->noalias[k] && !l->noalias[j])
					needcheck = 1;
		}
	/* reductions: integer acc = acc op g */
	X.flt = 0;
	for (k = 0; k < l->nload; k++)
		if (TYK(l->load[k].ety) == K_FLOAT)
			X.flt = l->load[k].esize * 8;
	for (k = 0; k < l->nstore; k++)
		if (TYK(l->store[k].ety) == K_FLOAT)
			X.flt = l->store[k].esize * 8;
	for (k = 0; k < l->nred; k++) {
		nid g;
		ty_t rt = N.ty[l->rnext[k]];
		if (l->rslot[k] < 0 || !TY_ISINT(rt) || TYK(rt) == K_BOOL)
			return 0;	/* float sums: reassociation rounds */
		if (!red_split(l->rnext[k], l->rslot[k], &g) ||
		    node_uses(g, l->rslot[k]))
			return 0;
		if (TYW(rt) > Wmin)
			Wmin = TYW(rt);
	}
	if (X.flt && l->nred)
		return 0;
	for (k = 0; k < l->nload; k++)
		if (l->load[k].esize * 8 > Wmin)
			Wmin = l->load[k].esize * 8;
	for (k = 0; k < l->nstore; k++)
		if (l->store[k].esize * 8 > Wmin)
			Wmin = l->store[k].esize * 8;
	if (!l->nstore && !l->nred)
		return 0;
	X.ivar = mk_var(ISLOT, sym_cstr(l->iname), l->ity);
	/* the narrowest lanes first, the widest vectors first */
	for (W = X.flt ? X.flt : Wmin; W <= 64 && !ok; W *= 2)
		for (vi = 0; vi < 3 && !ok; vi++) {
			int VB = vbs[vi];
			if (VB == 512 && !(t->feat & CF_F(CF_AVX512F)))
				continue;
			if (VB == 512 && W <= 16 && !(t->feat & CF_F(CF_AVX512BW)))
				continue;
			if (VB == 256 && !(t->feat & CF_F(X.flt ? CF_AVX : CF_AVX2)))
				continue;
			{
				jmp_buf saved;
				memcpy(saved, fail_jmp, sizeof saved);
				if (setjmp(fail_jmp)) {
					memcpy(fail_jmp, saved, sizeof saved);
					why("the lane semantics did not build", NULL);
					continue;
				}
				ok = attempt(VB, W, &pl);
				memcpy(fail_jmp, saved, sizeof saved);
			}
		}
	if (!ok)
		return 0;
	/* the remainder: the scalar loop from the IR, from the current i */
	rem = *l;
	rem.init_text = "";
	memset(&ro, 0, sizeof ro);
	if (!cf_vec_stub(&rem, t, &ro)) {
		free(ro.text.p);
		return 0;
	}
	cb = ro.cost_before;
	UT = TYW(l->ity) == 32 ? "unsigned" : "unsigned long";
	if (needcheck) {
		/* every written array against every other: byte ranges */
		int first = 1;
		sb_puts(&chk, "\tif (");
		for (k = 0; k < l->nbase; k++) {
			if (!written[k])
				continue;
			for (j = 0; j < l->nbase; j++) {
				int64_t lo[2] = {64, 64}, hi[2] = {-64, -64};
				int b[2], q, a;
				if (j == k || l->noalias[k] || l->noalias[j] ||
				    (written[j] && j < k))
					continue;
				b[0] = k;
				b[1] = j;
				for (q = 0; q < 2; q++) {
					for (a = 0; a < l->nload; a++)
						if (l->load[a].base == b[q]) {
							affine(l->load[a].index, &off);
							if (off < lo[q]) lo[q] = off;
							if (off > hi[q]) hi[q] = off;
						}
					for (a = 0; a < l->nstore; a++)
						if (l->store[a].base == b[q]) {
							affine(l->store[a].index, &off);
							if (off < lo[q]) lo[q] = off;
							if (off > hi[q]) hi[q] = off;
						}
				}
				/* [base + (lo_text + lo) * size, base + (hi_text + hi + 1)
				 * * size) as integers: disjoint when one ends first */
				sb_printf(&chk, "%s((unsigned long)(const void *)(%s) + "
				    "(unsigned long)(%s + %lld) * sizeof *(%s) <= "
				    "(unsigned long)(const void *)(%s) + (unsigned long)(%s + "
				    "%lld) * sizeof *(%s) || (unsigned long)(const void *)(%s) "
				    "+ (unsigned long)(%s + %lld) * sizeof *(%s) <= "
				    "(unsigned long)(const void *)(%s) + (unsigned long)(%s + "
				    "%lld) * sizeof *(%s))", first ? "" : "\n\t    && ",
				    l->base[k], l->hi_text, (long long)hi[0], l->base[k],
				    l->base[j], l->iname, (long long)lo[1], l->base[j],
				    l->base[j], l->hi_text, (long long)hi[1], l->base[j],
				    l->base[k], l->iname, (long long)lo[0], l->base[k]);
				first = 0;
			}
		}
		sb_puts(&chk, ") {\n");
	}
	/* the block: body, stores, accumulator updates (cf_accK_@U) */
	for (k = 0; k < l->nstore; k++) {
		const struct cf_mem *m = &l->store[k];
		int bytes = X.VL * m->esize, v = pl.st[k];
		const char *st;
		affine(m->index, &off);
		if (X.flt)
			st = X.flt == 32 ? (X.VB == 512 ? "_mm512_storeu_ps" :
			    X.VB == 256 ? "_mm256_storeu_ps" : "_mm_storeu_ps") :
			    (X.VB == 512 ? "_mm512_storeu_pd" : X.VB == 256 ?
			    "_mm256_storeu_pd" : "_mm_storeu_pd");
		else
			st = bytes == 8 ? "_mm_storel_epi64" : bytes == 16 ?
			    "_mm_storeu_si128" : bytes == 32 ? "_mm256_storeu_si256" :
			    "_mm512_storeu_si512";
		{
			int id = xi_find(st);
			if (id >= 0) {
				X.need |= xi_feat(id);
				price(id);
			}
		}
		sb_printf(&X.body, "\t\t%s((void *)(%s), %s);\n", st,
		    base_text(m->base, off), X.v[v].nm);
	}
	for (k = 0; k < l->nred; k++) {
		const char *opn;
		switch (pl.rop[k]) {
		case O_ADD: opn = "add"; break;
		case O_SUB: opn = "sub"; break;
		case O_XOR: opn = "xor"; break;
		case O_OR: opn = "or"; break;
		case O_AND: opn = "and"; break;
		default: opn = pl.rop[k] == O_MIN ? "min" : "max";
		}
		if (pl.rop[k] == O_MIN || pl.rop[k] == O_MAX)
			snprintf(rop_name[k], sizeof rop_name[k], "%s_%s_ep%c%d", pfx(),
			    opn, TYK(N.ty[l->rnext[k]]) == K_UINT ? 'u' : 'i', X.W);
		else if (pl.rop[k] == O_ADD || pl.rop[k] == O_SUB)
			snprintf(rop_name[k], sizeof rop_name[k], "%s_%s_epi%d", pfx(),
			    opn, X.W);
		else
			snprintf(rop_name[k], sizeof rop_name[k], "%s_%s_%s", pfx(), opn,
			    si());
		/* combining partial sums of a difference adds them */
		snprintf(comb_name[k], sizeof comb_name[k], "%s", rop_name[k]);
		if (pl.rop[k] == O_SUB)
			snprintf(comb_name[k], sizeof comb_name[k], "%s_add_epi%d",
			    pfx(), X.W);
		{
			int id = xi_find(rop_name[k]);
			if (id < 0 || !t_has_all(xi_feat(id))) {
				free(ro.text.p);
				return 0;
			}
			X.need |= xi_feat(id);
			price(id);
		}
		sb_printf(&X.body, "\t\tcf_acc%d_@U = %s(cf_acc%d_@U, %s);\n", k,
		    rop_name[k], k, X.v[pl.rv[k]].nm);
	}
	/* assemble */
	sb_puts(&out->text, "{\n");
	if (l->init_text && l->init_text[0]) {
		size_t n = strlen(l->init_text);
		while (n && l->init_text[n - 1] == ';')
			n--;
		sb_printf(&out->text, "\t%.*s;\n", (int)n, l->init_text);
	}
	sb_puts(&out->text, X.pre.p ? X.pre.p : "");
	{
		struct sbuf m = {0};
		sb_printf(&out->text, "#if %s\n", cmacro(X.need, &m));
		free(m.p);
	}
	sb_puts(&out->text, chk.p ? chk.p : "\t{\n");
	U = 4;	/* unrolled: independent accumulators and fewer loop branches */
	for (k = 0; k < l->nred; k++)
		for (u = 0; u < U; u++)
			sb_printf(&out->text, "\t%s cf_acc%d_%d = %s_set1_epi%d%s((%s)"
			    "%s);\n", vtype('i', X.W), k, u, pfx(), X.W, X.W == 64 &&
			    X.VB < 512 ? "x" : "", ctype_of(TY(K_SINT, X.W)), red_ident(pl.rop[k],
			    N.ty[l->rnext[k]]));
	for (; U >= 1; U = U > 1 ? 1 : 0) {
		sb_printf(&out->text, "\tfor (; %s < (%s) && (%s)((%s)(%s) - (%s)%s) "
		    ">= %d; %s += %d) {\n", l->iname, l->hi_text, UT, UT, l->hi_text,
		    UT, l->iname, U * X.VL, l->iname, U * X.VL);
		for (u = 0; u < U; u++) {
			const char *q = X.body.p ? X.body.p : "";
			sb_printf(&out->text, "\t    {\n\t\tconst %s cf_i = %s + %d;\n",
			    ctype_of(l->ity), l->iname, u * X.VL);
			for (; *q; q++)
				if (q[0] == '@' && q[1] == 'U') {
					sb_printf(&out->text, "%d", u);
					q++;
				} else
					sb_putc(&out->text, *q);
			sb_puts(&out->text, "\t    }\n");
		}
		sb_puts(&out->text, "\t}\n");
	}
	for (k = 0; k < l->nred; k++)
		for (u = 1; u < 4; u++)
			sb_printf(&out->text, "\tcf_acc%d_0 = %s(cf_acc%d_0, "
			    "cf_acc%d_%d);\n", k, comb_name[k], k, k, u);
	/* combine the partial results into the variable */
	for (k = 0; k < l->nred; k++) {
		ty_t rt = N.ty[l->rnext[k]], lt = TY(TYK(rt) == K_UINT ? K_UINT :
		    K_SINT, X.W);
		const char *ct = ctype_of(rt), *ut = ctype_of(TY(K_UINT, TYW(rt)));
		sb_printf(&out->text, "\t{\n\t\t%s cf_part[%d];\n\t\t%s_storeu_%s("
		    "(void *)cf_part, cf_acc%d_0);\n\t\tfor (int cf_j = 0; cf_j < %d; "
		    "cf_j++)\n", ctype_of(lt), X.VL, pfx(), si(), k, X.VL);
		switch (pl.rop[k]) {
		case O_MIN: case O_MAX:
			sb_printf(&out->text, "\t\t\t%s = (%s)cf_part[cf_j] %c %s ? "
			    "(%s)cf_part[cf_j] : %s;\n", l->rname[k], ct,
			    pl.rop[k] == O_MIN ? '<' : '>', l->rname[k], ct,
			    l->rname[k]);
			break;
		default:
			sb_printf(&out->text, "\t\t\t%s = (%s)((%s)%s %c (%s)"
			    "cf_part[cf_j]);\n", l->rname[k], ct, ut, l->rname[k],
			    pl.rop[k] == O_XOR ? '^' : pl.rop[k] == O_OR ? '|' :
			    pl.rop[k] == O_AND ? '&' : '+', ut);
		}
		sb_puts(&out->text, "\t}\n");
	}
	sb_puts(&out->text, "\t}\n#endif\n\t");
	sb_puts(&out->text, ro.text.p ? ro.text.p : "");
	sb_puts(&out->text, "\n}");
	free(ro.text.p);
	{
		struct sbuf f = {0};
		xfeat_text(&f, X.need);
		snprintf(out->what, sizeof out->what, "%d-bit vectors, %d lanes of "
		    "%d bits (%s)%s%s; %d lanes proved (%d equal DAGs, %d "
		    "exhaustive, %d SAT)", X.VB, X.VL, X.W, f.p ? f.p : "",
		    needcheck ? ", run-time overlap test" : "",
		    l->nred ? ", reductions reassociated (exact mod 2^n)" : "",
		    pl.lp.lanes, pl.lp.dag, pl.lp.exh, pl.lp.sat);
		free(f.p);
	}
	out->proof = CF_PROOF_LANES;
	out->cost_before = cb;
	out->cost_after = X.cost / X.VL;
	free(X.body.p);
	free(X.pre.p);
	free(chk.p);
	X.body.p = X.pre.p = NULL;
	return 1;
}

void
cfx86_init(void)
{
	x86_intr_init();
	cf_set_vectoriser(vec_loop);
}
