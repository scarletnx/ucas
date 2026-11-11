/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * simplify's fp64 part, included by spirv.c (it reads the parsed module).
 *
 * dword_pass runs on the module as the integer and float parts left it.
 * Every scalar f64 computation that ends in a conversion to f32 becomes a
 * function of its inputs (f32 values and integers converted to f64, f64
 * constants; any other f64 value is an input split in place), and
 * dword.c decides: the cheapest f32 form whose derived bound meets the
 * requirement replaces the f64 instructions when the active model prices
 * it lower.  The form is written back as f32 instructions, the operations
 * of its error-free transformations decorated NoContraction; the Float64
 * capability goes when nothing uses the type any more.
 *
 * The same pass reports the float round trips the model says its driver
 * folds (fpfold.c), with an input where the folded program differs, and
 * with `foldproof` writes their rounding as integer instructions.
 */
#include "dword.h"

#include <unistd.h>

static int dw_foldproof;	/* apply the fold-proof forms */
static int dw_nfas;		/* float assumptions float_pass left in pend_fas */

void
spirv_dword_opts(int foldproof)
{
	dw_foldproof = foldproof;
}

static int
scalar_float_w(int t)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_FLOAT)
		return 0;
	return T(t)->w;
}

static int
scalar_int_w(int t, int *sign)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_INT)
		return 0;
	*sign = T(t)->sign;
	return T(t)->w;
}

/* ---- a SPIR-V f64 value as a ucas expression ---- */

struct dwg {
	struct func f;
	int psid[MAXP];		/* the SPIR-V value of each parameter */
	int id[256];
	nid node[256];
	int nmemo;
	int ins[256];		/* the f64 instructions under the root */
	int nins;
	int toomany;
};

static int
dwg_param(struct dwg *G, int sid, ty_t ty)
{
	char nb[48], nm[56];
	const char *n;
	struct param *p;
	int j;

	for (j = 0; j < G->f.np; j++)
		if (G->psid[j] == sid)
			return j;
	if (G->f.np == MAXP) {
		G->toomany = 1;
		return 0;
	}
	j = G->f.np++;
	G->psid[j] = sid;
	n = load_name(sid, nb, sizeof nb);
	if (n == NULL) {
		n = V.id[sid].name;
		if (*n == '%')
			n++;
	}
	if (!isalpha((unsigned char)*n) && *n != '_')
		snprintf(nm, sizeof nm, "v%s", n);
	else
		snprintf(nm, sizeof nm, "%s", n);
	p = &G->f.p[j];
	memset(p, 0, sizeof *p);
	p->name = sym_cstr(nm);
	p->ty = ty;
	if (ty == TY_F32) {
		int q;
		for (q = 0; q < dw_nfas; q++)
			if (name_is(n, pend_fas[q].name) || !strcmp(n,
			    pend_fas[q].name)) {
				p->ranged = 1;
				p->rlo = (double)pend_fas[q].lo;
				p->rhi = (double)pend_fas[q].hi;
				if (p->rlo == 0)
					p->rlo = 0;	/* -0: the same interval */
			}
	} else if (TY_ISINT(ty)) {
		int64_t lo, hi;
		if (assumed(sid, &lo, &hi)) {
			p->ranged = 1;
			p->lo = lo;
			p->hi = hi;
			p->rlo = (double)lo;
			p->rhi = (double)hi;
		}
	}
	return j;
}

static nid
dwg_var(struct dwg *G, int sid, ty_t ty)
{
	int j = dwg_param(G, sid, ty);

	return mk_var((uint32_t)j, G->f.p[j].name, ty);
}

static nid
dwg_build(struct dwg *G, int id, int depth)
{
	int k = SP.def[id], i, t = V.id[id].type, sg;
	const struct sins *I;
	nid r = 0, a, b, c;

	for (i = 0; i < G->nmemo; i++)
		if (G->id[i] == id)
			return G->node[i];
	if (V.id[id].kind == K_CONST && !V.id[id].spec &&
	    scalar_float_w(t) == 64) {
		r = mk_cast(TY_F64, mk_const_r(getf(t, V.id[id].c[0])));
		goto done;
	}
	if (k < 0 || depth > 48)
		return dwg_var(G, id, TY_F64);
	I = &V.in[k];
	if (I->nop < 2 || scalar_float_w(idx(I->opnd[0])) != 64)
		return dwg_var(G, id, TY_F64);
	if (!strcmp(I->op, "OpFConvert")) {
		int s = idx(I->opnd[1]);
		if (scalar_float_w(V.id[s].type) == 32)
			r = mk_cast(TY_F64, dwg_var(G, s, TY_F32));
	} else if (!strcmp(I->op, "OpConvertSToF") || !strcmp(I->op,
	    "OpConvertUToF")) {
		int s = idx(I->opnd[1]), w = scalar_int_w(V.id[s].type, &sg);
		if (w > 0 && w <= 32)
			r = mk_cast(TY_F64, dwg_var(G, s, TY(I->op[9] == 'U' ?
			    K_UINT : K_SINT, w)));
	} else if (!strcmp(I->op, "OpFNegate")) {
		a = dwg_build(G, idx(I->opnd[1]), depth + 1);
		r = mk_un(O_NEG, a);
	} else if (!strcmp(I->op, "OpFAdd") || !strcmp(I->op, "OpFSub") ||
	    !strcmp(I->op, "OpFMul") || !strcmp(I->op, "OpFDiv")) {
		if (I->nop >= 3) {
			a = dwg_build(G, idx(I->opnd[1]), depth + 1);
			b = dwg_build(G, idx(I->opnd[2]), depth + 1);
			r = mk_bin(I->op[3] == 'A' ? O_ADD : I->op[3] == 'S' ? O_SUB :
			    I->op[3] == 'M' ? O_MUL : O_DIV, a, b);
		}
	} else if (!strcmp(I->op, "OpExtInst") && I->nop >= 4 &&
	    idx(I->opnd[1]) == V.glsl) {
		if (!strcmp(I->opnd[2], "Sqrt") || !strcmp(I->opnd[2], "FAbs")) {
			a = dwg_build(G, idx(I->opnd[3]), depth + 1);
			r = mk_un(I->opnd[2][0] == 'S' ? O_SQRT : O_ABS, a);
		} else if (!strcmp(I->opnd[2], "Fma") && I->nop >= 6) {
			a = dwg_build(G, idx(I->opnd[3]), depth + 1);
			b = dwg_build(G, idx(I->opnd[4]), depth + 1);
			c = dwg_build(G, idx(I->opnd[5]), depth + 1);
			r = mk_fma(a, b, c);
		}
	}
	if (r == 0)
		return dwg_var(G, id, TY_F64);
	if (G->nins < 256)
		G->ins[G->nins++] = k;
done:
	if (G->nmemo < 256) {
		G->id[G->nmemo] = id;
		G->node[G->nmemo++] = r;
	}
	return r;
}

/* ---- a ucas form as SPIR-V instructions ---- */

struct dwemit {
	struct sbuf *code, *decl, *deco;
	const struct dwg *G;
	nid memo_n[1024];
	char memo_s[1024][40];
	int nmemo;
	const uint8_t *eft;
	uint32_t neft;
	int isl, serial;
	nid root;
	const char *final;	/* the id the root defines */
	int fail, need_glsl, ndeco, nins;
};

static const char *
dwe_type(struct dwemit *E, ty_t t)
{
	static char nm[8][48];
	static int at;
	char *b = nm[at++ & 7];
	int i;

	if (t == TY_BOOL)
		return bool_type(E->decl, 1);
	for (i = 0; i < V.nid; i++) {
		const struct sty *y;
		if (V.id[i].kind != K_TYPE)
			continue;
		y = &V.id[i].t;
		if (TYK(t) == K_FLOAT && y->k == S_FLOAT && y->w == TYW(t))
			return V.id[i].tok;
		if ((TYK(t) == K_UINT || TYK(t) == K_SINT) && y->k == S_INT &&
		    y->w == TYW(t) && y->sign == (TYK(t) == K_SINT))
			return V.id[i].tok;
	}
	if (TYK(t) == K_FLOAT) {
		snprintf(b, 48, "%%ucas_f%d", TYW(t));
		if (E->decl->p == NULL || !strstr(E->decl->p, b))
			sb_printf(E->decl, "%s = OpTypeFloat %d\n", b, TYW(t));
		return b;
	}
	snprintf(b, 48, "%%ucas_%c%d", TYK(t) == K_SINT ? 'i' : 'u', TYW(t));
	if (E->decl->p == NULL || !strstr(E->decl->p, b))
		sb_printf(E->decl, "%s = OpTypeInt %d %d\n", b, TYW(t),
		    TYK(t) == K_SINT);
	return b;
}

static const char *
dwe_const(struct dwemit *E, nid n)
{
	static char nm[8][64];
	static int at;
	char *b = nm[at++ & 7], line[160];
	ty_t t = N.ty[n];
	const char *tt = dwe_type(E, t);

	if (TYK(t) == K_FLOAT) {
		uint64_t bits = fbits(TYW(t), N.kr[n]);
		snprintf(b, 64, "%%ucas_k%s_%llx", ty_str(t),
		    (unsigned long long)bits);
		snprintf(line, sizeof line, "%s = OpConstant %s %.*g\n", b, tt,
		    TYW(t) == 64 ? 17 : 9, N.kr[n]);
	} else {
		uint64_t v = (uint64_t)N.ki[n];
		if (TYW(t) < 64)
			v &= ((uint64_t)1 << TYW(t)) - 1;
		snprintf(b, 64, "%%ucas_k%s_%llu", ty_str(t),
		    (unsigned long long)v);
		if (TYK(t) == K_SINT)
			snprintf(line, sizeof line, "%s = OpConstant %s %lld\n", b,
			    tt, (long long)N.ki[n]);
		else
			snprintf(line, sizeof line, "%s = OpConstant %s %llu\n", b,
			    tt, (unsigned long long)v);
	}
	{
		/* once: the name with its " = " */
		char key[80];
		snprintf(key, sizeof key, "%s = ", b);
		if (E->decl->p == NULL || !strstr(E->decl->p, key))
			sb_puts(E->decl, line);
	}
	return b;
}

static const char *
dwe(struct dwemit *E, nid n)
{
	char a[40], b[40], c[40], nm[40];
	const char *op = NULL, *ext = NULL, *rt;
	ty_t t = N.ty[n], ta;
	int i, fl, arith = 0;

	for (i = 0; i < E->nmemo; i++)
		if (E->memo_n[i] == n)
			return E->memo_s[i];
	if (E->nmemo >= 1024) {
		E->fail = 1;
		return "%ucas_toolong";
	}
	if (N.op[n] == O_VAR)
		return V.id[E->G->psid[N.ki[n]]].tok;
	if (N.op[n] == O_CONST) {
		const char *k;
		if (t == TY_BOOL) {
			E->fail = 1;
			return "%ucas_bool_const";
		}
		k = dwe_const(E, n);
		i = E->nmemo++;
		E->memo_n[i] = n;
		snprintf(E->memo_s[i], 40, "%s", k);
		return E->memo_s[i];
	}
	a[0] = b[0] = c[0] = 0;
	if (N.a[n])
		snprintf(a, sizeof a, "%s", dwe(E, N.a[n]));
	if (N.b[n] && N.op[n] != O_CAST)
		snprintf(b, sizeof b, "%s", dwe(E, N.b[n]));
	if (N.c[n])
		snprintf(c, sizeof c, "%s", dwe(E, N.c[n]));
	ta = N.a[n] ? N.ty[N.a[n]] : 0;
	fl = TYK(t) == K_FLOAT;
	rt = dwe_type(E, t);
	if (n == E->root && E->final)
		snprintf(nm, sizeof nm, "%s", E->final);
	else
		snprintf(nm, sizeof nm, "%%ucas_dw%d_%d", E->isl, ++E->serial);
	switch (N.op[n]) {
	case O_ADD: op = fl ? "OpFAdd" : "OpIAdd"; arith = fl; break;
	case O_SUB: op = fl ? "OpFSub" : "OpISub"; arith = fl; break;
	case O_MUL: op = fl ? "OpFMul" : "OpIMul"; arith = fl; break;
	case O_DIV: op = fl ? "OpFDiv" : NULL; arith = fl; break;
	case O_NEG: op = fl ? "OpFNegate" : "OpSNegate"; arith = fl; break;
	case O_SQRT: ext = "Sqrt"; break;
	case O_ABS: ext = fl ? "FAbs" : "SAbs"; break;
	case O_FMA: ext = "Fma"; break;
	case O_MIN: ext = fl ? "FMin" : TYK(t) == K_SINT ? "SMin" : "UMin"; break;
	case O_MAX: ext = fl ? "FMax" : TYK(t) == K_SINT ? "SMax" : "UMax"; break;
	case O_AND: op = "OpBitwiseAnd"; break;
	case O_OR: op = "OpBitwiseOr"; break;
	case O_XOR: op = "OpBitwiseXor"; break;
	case O_SHL: op = "OpShiftLeftLogical"; break;
	case O_SHR:
		op = TYK(t) == K_SINT ? "OpShiftRightArithmetic" :
		    "OpShiftRightLogical";
		break;
	case O_LAND: op = "OpLogicalAnd"; break;
	case O_LOR: op = "OpLogicalOr"; break;
	case O_ASUINT: case O_ASINT: case O_ASFLOAT: op = "OpBitcast"; break;
	case O_SEL: op = "OpSelect"; break;
	case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE: {
		static const char *const fo[] = {"OpFOrdEqual", "OpFUnordNotEqual",
			"OpFOrdLessThan", "OpFOrdLessThanEqual", "OpFOrdGreaterThan",
			"OpFOrdGreaterThanEqual"};
		static const char *const uo[] = {"OpIEqual", "OpINotEqual",
			"OpULessThan", "OpULessThanEqual", "OpUGreaterThan",
			"OpUGreaterThanEqual"};
		static const char *const so[] = {"OpIEqual", "OpINotEqual",
			"OpSLessThan", "OpSLessThanEqual", "OpSGreaterThan",
			"OpSGreaterThanEqual"};
		int q = N.op[n] - O_EQ;
		op = TYK(ta) == K_FLOAT ? fo[q] : TYK(ta) == K_SINT ? so[q] :
		    TYK(ta) == K_UINT ? uo[q] : NULL;
		break;
	}
	case O_CAST:
		if (fl && TYK(ta) == K_FLOAT)
			op = "OpFConvert";
		else if (fl && TYK(ta) == K_UINT)
			op = "OpConvertUToF";
		else if (fl && TYK(ta) == K_SINT)
			op = "OpConvertSToF";
		else if (!fl && ta == TY_BOOL) {
			/* a bool as 0 or 1 */
			const char *one = dwe_const(E, mk_const_i(t, 1));
			char o1[64];
			snprintf(o1, sizeof o1, "%s", one);
			sb_printf(E->code, "%s = OpSelect %s %s %s %s\n", nm, rt, a,
			    o1, dwe_const(E, mk_const_i(t, 0)));
			E->nins++;
			goto named;
		} else if (!fl && TY_SIZED(ta) && TYW(ta) == TYW(t))
			op = "OpBitcast";
		break;
	}
	if (op == NULL && ext == NULL) {
		E->fail = 1;
		return "%ucas_noform";
	}
	if (ext) {
		E->need_glsl = 1;
		sb_printf(E->code, "%s = OpExtInst %s %s %s %s", nm, rt,
		    V.glsl >= 0 ? V.id[V.glsl].tok : "%ucas_glsl", ext, a);
		if (b[0])
			sb_printf(E->code, " %s", b);
		if (c[0])
			sb_printf(E->code, " %s", c);
		sb_putc(E->code, '\n');
	} else {
		sb_printf(E->code, "%s = %s %s %s", nm, op, rt, a);
		if (b[0])
			sb_printf(E->code, " %s", b);
		if (c[0])
			sb_printf(E->code, " %s", c);
		sb_putc(E->code, '\n');
	}
	E->nins++;
	if (arith && n < E->neft && E->eft && E->eft[n]) {
		sb_printf(E->deco, "OpDecorate %s NoContraction\n", nm);
		E->ndeco++;
	}
named:
	i = E->nmemo++;
	E->memo_n[i] = n;
	snprintf(E->memo_s[i], 40, "%s", nm);
	return E->memo_s[i];
}

static void
dw_indent(const char *text, const char *ind)
{
	const char *p = text, *e;

	while (*p) {
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		printf("%s%.*s\n", ind, (int)(e - p), p);
		p = *e ? e + 1 : e;
	}
}

/* spirv-as and spirv-val on the written module: 1 accepted, 0 rejected
 * (why: the tool's first line), -1 the tools cannot run here */
static int
dw_validate(const char *text, char *why, size_t n)
{
	static const char *const envs[] = {"vulkan1.1", NULL};
	char src[] = "/tmp/ucas-dw-XXXXXX.spvasm", bin[] = "/tmp/ucas-dw-XXXXXX.spv";
	struct sbuf log = {0};
	int rc = -1, e, res = -1;

	why[0] = 0;
	if (tool_tmp(src, text, strlen(text)) < 0)
		return -1;
	if (tool_tmp(bin, "", 0) < 0) {
		unlink(src);
		return -1;
	}
	for (e = 0; e < 2 && res != 1; e++) {
		char *a1[] = {"spirv-as", "--target-env", (char *)envs[0], src, "-o",
		    bin, NULL};
		char *a2[] = {"spirv-as", src, "-o", bin, NULL};
		char *v1[] = {"spirv-val", "--target-env", (char *)envs[0], bin, NULL};
		char *v2[] = {"spirv-val", bin, NULL};
		sb_clear(&log);
		rc = tool_run(e == 0 ? a1 : a2, &log);
		if (rc == 127 || rc < 0)
			break;
		if (rc == 0) {
			sb_clear(&log);
			rc = tool_run(e == 0 ? v1 : v2, &log);
			if (rc == 127 || rc < 0)
				break;
		}
		res = rc == 0;
		if (!res && why[0] == 0 && log.p)
			snprintf(why, n, "%.*s", (int)strcspn(log.p, "\n"), log.p);
	}
	unlink(src);
	unlink(bin);
	free(log.p);
	return res;
}

static char *
dword_pass(const char *text, int *changed)
{
	struct sbuf decl = {0}, deco = {0}, out = {0};
	char **repl, *dead;
	int k, i, nsink = 0, napplied = 0, nfold = 0, nfoldfix = 0, need_glsl = 0;
	int f64t = -1, isl = 0, has64 = 0, hdr = 0;
	const struct model *m = model_active();
	struct dwreq rq = {DW_DEFAULT, 0};
	int *uses;

	*changed = 0;
	memset(&V, 0, sizeof V);
	symmap_reset();
	V.glsl = -1;
	V.lanes = 1;
	V.rng = 1;
	cols = 1;
	parse(text);
	declare();
	SP.def = xmalloc((size_t)(V.nid + 64) * sizeof *SP.def);
	uses = xcalloc((size_t)(V.nid + 64), sizeof *uses);
	for (i = 0; i < V.nid + 64; i++)
		SP.def[i] = -1;
	{
		int inbody = 0;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (!strcmp(I->op, "OpFunction"))
				inbody = 1;
			if (inbody && I->res >= 0) {
				SP.def[I->res] = k;
				if (I->nop >= 1 && I->opnd[0][0] == '%' &&
				    V.id[idx(I->opnd[0])].kind == K_TYPE)
					V.id[I->res].type = idx(I->opnd[0]);
			}
			if (!strncmp(I->op, "OpName", 6) || !strncmp(I->op,
			    "OpMemberName", 12) || !strncmp(I->op, "OpDecorate", 10) ||
			    !strncmp(I->op, "OpMemberDecorate", 16))
				continue;
			for (i = 0; i < I->nop; i++)
				if (I->opnd[i][0] == '%')
					uses[idx(I->opnd[i])]++;
		}
	}
	for (i = 0; i < V.nid; i++)
		if (scalar_float_w(i) == 64)
			f64t = i;
	repl = xcalloc((size_t)V.nin + 1, sizeof *repl);
	dead = xcalloc((size_t)V.nin + 1, 1);
	if (fopt.mode == 2) {
		rq.kind = DW_ULP;
		rq.v = fopt.ulp > 0 ? fopt.ulp : 0x1p-10;
	}
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		struct dwg *G;
		struct dwres res;
		struct sbuf rep = {0};
		char sl[200];
		nid root;
		int src, q, shared = 0;

		if (strcmp(I->op, "OpFConvert") != 0 || I->nop < 2 || I->res < 0 ||
		    SP.def[I->res] != k || scalar_float_w(idx(I->opnd[0])) != 32)
			continue;
		src = idx(I->opnd[1]);
		/* a folded round trip: f32 <- f16 <- f32 */
		if (scalar_float_w(V.id[src].type) == 16 && SP.def[src] >= 0 &&
		    !strcmp(V.in[SP.def[src]].op, "OpFConvert") &&
		    scalar_float_w(V.id[idx(V.in[SP.def[src]].opnd[1])].type) == 32
		    && model_folds(m, TY_F32, TY_F16)) {
			struct dwg *F2 = xcalloc(1, sizeof *F2);
			struct dwemit *E;
			struct sbuf code = {0};
			nid x, trip, form;
			int ident = 0;

			F2->f.name = sym_cstr("_trip");
			F2->f.anon = 1;
			x = dwg_var(F2, idx(V.in[SP.def[src]].opnd[1]), TY_F32);
			trip = mk_cast(TY_F32, mk_cast(TY_F16, x));
			F2->f.body = trip;
			if (!hdr++)
				printf("fp64 and conversions under %s:\n",
				    sym_str(m->name));
			printf("  %s\n", I->text);
			src_line(ins_line(k), sl, sizeof sl);
			fputs(sl, stdout);
			nfold++;
			form = fold_one(&F2->f, trip, m, &ident);
			if (form != 0 && (dw_foldproof || ident)) {
				E = xcalloc(1, sizeof *E);
				E->code = &code;
				E->decl = &decl;
				E->deco = &deco;
				E->G = F2;
				E->isl = ++isl;
				E->root = form;
				E->final = V.id[I->res].tok;
				if (N.op[form] == O_VAR)
					sb_printf(&code, "%s = OpCopyObject %s %s\n",
					    V.id[I->res].tok, V.id[idx(I->opnd[0])].tok,
					    V.id[F2->psid[0]].tok);
				else
					dwe(E, form);
				if (!E->fail) {
					repl[k] = code.p;
					code.p = NULL;
					need_glsl |= E->need_glsl;
					nfoldfix++;
					uses[src]--;
					printf("    written as %d instruction%s in place "
					    "of the round trip\n", E->nins ? E->nins : 1,
					    E->nins == 1 ? "" : "s");
				} else
					printf("    not written: an operation of the "
					    "form has no SPIR-V instruction here\n");
				free(E);
			} else if (form != 0)
				printf("    not applied: it costs more than the "
				    "folded round trip; `foldproof` (simplify ... "
				    "foldproof) writes it into the module\n");
			free(code.p);
			free(F2);
			continue;
		}
		if (scalar_float_w(V.id[src].type) != 64)
			continue;
		has64 = 1;
		G = xcalloc(1, sizeof *G);
		G->f.name = sym_cstr("_fp64");
		G->f.anon = 1;
		root = dwg_build(G, src, 0);
		nsink++;
		if (!hdr++)
			printf("fp64 and conversions under %s:\n", sym_str(m->name));
		printf("  %s\n", I->text);
		src_line(ins_line(k), sl, sizeof sl);
		fputs(sl, stdout);
		if (G->toomany || G->nins == 0) {
			printf("    not rewritten: %s\n", G->toomany ? "more than "
			    "8 inputs" : "its operand is an f64 value computed "
			    "elsewhere (a load, a call, a phi)");
			free(G);
			continue;
		}
		/* a value another result also reads would be computed twice */
		for (q = 0; q < G->nins && !shared; q++) {
			int rid = V.in[G->ins[q]].res, inside = rid == src, j, o;
			for (j = 0; j < G->nins; j++)
				for (o = 1; o < V.in[G->ins[j]].nop; o++)
					if (V.in[G->ins[j]].opnd[o][0] == '%' &&
					    idx(V.in[G->ins[j]].opnd[o]) == rid)
						inside++;
			if (inside != uses[rid])
				shared = 1;
		}
		if (shared) {
			printf("    not rewritten: an f64 value under it is read by "
			    "another result too (one result at a time)\n");
			free(G);
			continue;
		}
		G->f.body = mk_cast(TY_F32, root);
		if (dword_lower(&G->f, m, &rq, &res, &rep)) {
			struct dwemit *E = xcalloc(1, sizeof *E);
			struct sbuf code = {0};
			E->code = &code;
			E->decl = &decl;
			E->deco = &deco;
			E->G = G;
			E->eft = res.eft;
			E->neft = res.neft;
			E->isl = ++isl;
			E->root = res.body;
			E->final = V.id[I->res].tok;
			if (N.op[res.body] == O_VAR || N.op[res.body] == O_CONST)
				sb_printf(&code, "%s = OpCopyObject %s %s\n",
				    V.id[I->res].tok, V.id[idx(I->opnd[0])].tok,
				    dwe(E, res.body));
			else
				dwe(E, res.body);
			dw_indent(rep.p ? rep.p : "", "    ");
			if (E->fail)
				printf("    not written: an operation of the form has "
				    "no SPIR-V instruction here\n");
			else {
				repl[k] = code.p;
				code.p = NULL;
				need_glsl |= E->need_glsl;
				napplied++;
				uses[src]--;
				printf("    written as %d f32 instruction%s, %d of them "
				    "NoContraction, in place of %d f64 one%s\n",
				    E->nins, E->nins == 1 ? "" : "s", E->ndeco,
				    G->nins, G->nins == 1 ? "" : "s");
			}
			free(code.p);
			free(E);
		} else {
			if (rep.p)
				dw_indent(rep.p, "    ");
			else
				printf("    %s\n", res.why);
		}
		dword_res_free(&res);
		free(rep.p);
		free(G);
	}
	if (napplied + nfoldfix > 0) {
		/* what the rewrites left unread */
		int ch = 1;
		while (ch) {
			ch = 0;
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				if (dead[k] || repl[k] || I->res < 0 ||
				    SP.def[I->res] != k || uses[I->res] > 0 ||
				    !(fpure(I) || !strcmp(I->op, "OpFConvert") ||
				    !strcmp(I->op, "OpConvertSToF") ||
				    !strcmp(I->op, "OpConvertUToF")))
					continue;
				dead[k] = 1;
				ch = 1;
				for (i = 0; i < I->nop; i++)
					if (I->opnd[i][0] == '%')
						uses[idx(I->opnd[i])]--;
			}
		}
	}
	/* the f64 type, when nothing is left of it: its constants and the
	 * capability go with it */
	if (napplied > 0 && f64t >= 0) {
		char *d64 = xcalloc((size_t)V.nid + 1, 1);
		int left = 0, inbody = 0, ch = 1;
		d64[f64t] = 1;
		while (ch) {
			ch = 0;
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				if (!strcmp(I->op, "OpFunction"))
					break;
				if (I->res < 0 || d64[I->res])
					continue;
				for (i = 0; i < I->nop; i++)
					if (I->opnd[i][0] == '%' && d64[idx(I->opnd[i])]) {
						d64[I->res] = 1;
						ch = 1;
						break;
					}
			}
		}
		/* a use that stays: in a body, or by a global not of the type */
		for (k = 0; k < V.nin && !left; k++) {
			const struct sins *I = &V.in[k];
			if (!strcmp(I->op, "OpFunction"))
				inbody = 1;
			if (dead[k] || !strncmp(I->op, "OpName", 6) || !strncmp(I->op,
			    "OpMemberName", 12) || !strncmp(I->op, "OpDecorate", 10) ||
			    !strncmp(I->op, "OpMemberDecorate", 16))
				continue;
			if (repl[k]) {
				for (i = 0; i < V.nid && !left; i++) {
					const char *h = d64[i] ? strstr(repl[k], V.id[i].tok)
					    : NULL;
					size_t tl = strlen(V.id[i].tok);
					if (h && (h[tl] == ' ' || h[tl] == '\n' || h[tl] == 0))
						left = 1;
				}
				if (strstr(repl[k], "%ucas_kf64") != NULL ||
				    strstr(repl[k], "%ucas_f64") != NULL)
					left = 1;
				continue;
			}
			if (!inbody && I->res >= 0 && d64[I->res])
				continue;
			for (i = 0; i < I->nop; i++)
				if (I->opnd[i][0] == '%' && d64[idx(I->opnd[i])])
					left = 1;
		}
		if (decl.p && strstr(decl.p, "ucas_kf64"))
			left = 1;
		if (!left) {
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				if (!strcmp(I->op, "OpFunction"))
					break;
				if ((I->res >= 0 && d64[I->res]) || (!strcmp(I->op,
				    "OpCapability") && I->nop >= 1 && !strcmp(I->opnd[0],
				    "Float64")))
					dead[k] = 1;
				if ((!strcmp(I->op, "OpName") || !strncmp(I->op,
				    "OpDecorate", 10)) && I->nop >= 1 &&
				    I->opnd[0][0] == '%' && d64[idx(I->opnd[0])])
					dead[k] = 1;
			}
			printf("  no f64 value is left: the Float64 capability and "
			    "the type are removed\n");
		} else
			printf("  other f64 values remain: the module still needs "
			    "the Float64 capability\n");
		free(d64);
	}
	if (napplied + nfoldfix > 0) {
		int first_fn = -1, first_ty = -1, mm = -1;
		for (k = 0; k < V.nin; k++) {
			if (first_fn < 0 && !strcmp(V.in[k].op, "OpFunction"))
				first_fn = k;
			if (first_ty < 0 && (!strncmp(V.in[k].op, "OpType", 6) ||
			    !strncmp(V.in[k].op, "OpConstant", 10) ||
			    !strcmp(V.in[k].op, "OpVariable")))
				first_ty = k;
			if (mm < 0 && !strcmp(V.in[k].op, "OpMemoryModel"))
				mm = k;
		}
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (k == mm && need_glsl && V.glsl < 0)
				sb_puts(&out, "%ucas_glsl = OpExtInstImport "
				    "\"GLSL.std.450\"\n");
			if (k == first_ty && deco.p)
				sb_puts(&out, deco.p);
			if (k == first_fn && decl.p)
				sb_puts(&out, decl.p);
			if (dead[k])
				continue;
			if ((!strcmp(I->op, "OpName") || !strncmp(I->op, "OpDecorate",
			    10)) && I->nop >= 1 && I->opnd[0][0] == '%') {
				int t = idx(I->opnd[0]);
				if (SP.def[t] >= 0 && dead[SP.def[t]])
					continue;
			}
			if (repl[k])
				sb_puts(&out, repl[k]);
			else
				sb_printf(&out, "%s\n", I->text);
		}
		*changed = 1;
		{
			char why[200];
			int v = dw_validate(out.p, why, sizeof why);
			if (v == 1)
				printf("  spirv-val accepts the written module\n");
			else if (v == 0)
				printf("  spirv-val REJECTS the written module: %s\n", why);
			else
				printf("  spirv-as and spirv-val cannot run here: the "
				    "written module is not validated\n");
		}
	}
	(void)nsink;
	(void)has64;
	(void)nfold;
	for (k = 0; k < V.nin; k++)
		free(repl[k]);
	free(repl);
	free(dead);
	free(decl.p);
	free(deco.p);
	free(uses);
	free(SP.def);
	SP.def = NULL;
	for (k = 0; k < V.nin; k++) {
		for (i = 0; i < V.in[k].nop; i++)
			free(V.in[k].opnd[i]);
		free(V.in[k].text);
	}
	free(V.in);
	memset(&V, 0, sizeof V);
	dw_foldproof = 0;
	if (!*changed) {
		free(out.p);
		return NULL;
	}
	return out.p;
}
