/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * simplify's memory part (memk.c), included by spirv.c.
 *
 * mem_pass reads the module's memory accesses by kind, each priced by
 * the active model's class for it, and says what the prices make worth
 * doing:
 *
 *   a block (UBO) read at an address that differs per lane, when the
 *   model prices that above a storage-buffer load: moving the block to a
 *   storage buffer is proposed, with the host's side of it (descriptor
 *   type, and the std140 to std430 offsets that change), never applied;
 *
 *   a texel fetch, when the model prices a sampled read lower: a nearest
 *   sampler at the texel's centre is offered as a candidate to measure,
 *   with the proof that the float coordinates select the same texel and
 *   the sampler state it needs;
 *
 *   a float atomic add on workgroup memory: intatomic's analysis of its
 *   addend;
 *
 *   the accesses of each workgroup array: their bank-conflict degrees
 *   and, when a proved swizzle is cheaper, every access of the array
 *   rewritten with it.
 *
 * An address is uniform when it depends only on constants, uniform
 * blocks read at uniform addresses and the workgroup's own builtins;
 * the invocation builtins, buffer contents and function parameters vary
 * per lane.  A phi is uniform when its inputs are and no branch of its
 * function has a varying condition.
 */
#include "memk.h"

/* 1 uniform across a warp, 0 varying */
static char *mem_uni;

static int
mem_base_var(int ptr, int *chain)
{
	int k, guard = 0;

	*chain = -1;
	while (guard++ < 16) {
		if (V.id[ptr].kind == K_VAR)
			return ptr;
		k = SP.def[ptr];
		if (k < 0) {
			int q;
			for (q = 0; q < V.nin; q++)
				if (V.in[q].res == ptr) {
					k = q;
					break;
				}
		}
		if (k < 0)
			return -1;
		if (!strcmp(V.in[k].op, "OpAccessChain") || !strcmp(V.in[k].op,
		    "OpInBoundsAccessChain")) {
			if (*chain < 0)
				*chain = k;
			ptr = idx(V.in[k].opnd[1]);
		} else if (!strcmp(V.in[k].op, "OpCopyObject"))
			ptr = idx(V.in[k].opnd[1]);
		else
			return -1;
	}
	return -1;
}

/* the storage class as written: "Uniform", "StorageBuffer", ... */
static const char *
mem_sc(int var)
{
	int q;

	for (q = 0; q < V.nin; q++)
		if (V.in[q].res == var && !strcmp(V.in[q].op, "OpVariable") &&
		    V.in[q].nop >= 2)
			return V.in[q].opnd[1];
	return "";
}

/* the variable's struct type is decorated BufferBlock (an old SSBO) */
static int
mem_bufferblock(int var)
{
	int t = V.id[var].type, q;

	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_PTR)
		return 0;
	t = T(t)->e;
	while (V.id[t].kind == K_TYPE && T(t)->k == S_ARRAY)
		t = T(t)->e;
	for (q = 0; q < V.nin; q++)
		if (!strcmp(V.in[q].op, "OpDecorate") && V.in[q].nop >= 2 &&
		    idx(V.in[q].opnd[0]) == t && !strcmp(V.in[q].opnd[1],
		    "BufferBlock"))
			return 1;
	return 0;
}

static void
mem_uniformity(void)
{
	int k, i, ch = 1, pass, varying_branch = 0;

	/* everything starts uniform and falls: the greatest fixed point */
	mem_uni = xmalloc((size_t)V.nid + 1);
	memset(mem_uni, 1, (size_t)V.nid + 1);
	for (pass = 0; pass < 2; pass++) {
		if (pass == 1)
			for (k = 0; k < V.nin; k++)
				if (!strcmp(V.in[k].op, "OpPhi") && V.in[k].res >= 0)
					mem_uni[V.in[k].res] = 0;
		ch = 1;
		while (ch) {
			ch = 0;
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				int u = 1, r = I->res;
				if (r < 0 || SP.def[r] != k || !mem_uni[r])
					continue;
				if (!strcmp(I->op, "OpLoad")) {
					int chain, var = mem_base_var(idx(I->opnd[1]), &chain);
					const char *sc = var >= 0 ? mem_sc(var) : "";
					if (var < 0)
						u = 0;
					else if (!strcmp(sc, "Input"))
						u = V.id[var].builtin == B_WGID ||
						    V.id[var].builtin == B_NUMWG;
					else if ((!strcmp(sc, "Uniform") &&
					    !mem_bufferblock(var)) || !strcmp(sc,
					    "PushConstant") || !strcmp(sc, "UniformConstant")) {
						if (chain >= 0)
							for (i = 2; i < V.in[chain].nop; i++)
								if (!mem_uni[idx(V.in[chain].opnd[i])])
									u = 0;
					} else
						u = 0;
				} else if (!strcmp(I->op, "OpPhi")) {
					for (i = 1; i < I->nop && u; i += 2)
						if (!mem_uni[idx(I->opnd[i])])
							u = 0;
				} else if (!strcmp(I->op, "OpFunctionParameter") ||
				    !strcmp(I->op, "OpFunctionCall") || !strncmp(I->op,
				    "OpAtomic", 8) || !strncmp(I->op, "OpImage", 7) ||
				    !strcmp(I->op, "OpVariable") || !strcmp(I->op,
				    "OpAccessChain") || !strcmp(I->op,
				    "OpInBoundsAccessChain"))
					u = !strcmp(I->op, "OpAccessChain") || !strcmp(I->op,
					    "OpInBoundsAccessChain");	/* a pointer: its load says */
				else
					for (i = 1; i < I->nop; i++)
						if (I->opnd[i][0] == '%' &&
						    !mem_uni[idx(I->opnd[i])])
							u = 0;
				if (!u) {
					mem_uni[r] = 0;
					ch = 1;
				}
			}
		}
		for (k = 0; k < V.nin; k++)
			if ((!strcmp(V.in[k].op, "OpBranchConditional") ||
			    !strcmp(V.in[k].op, "OpSwitch")) && V.in[k].nop >= 1 &&
			    !mem_uni[idx(V.in[k].opnd[0])])
				varying_branch = 1;
		if (!varying_branch)
			break;
	}
}

/* ---- std140 and std430 ---- */

static int
mem_roundup(int v, int a)
{
	return (v + a - 1) / a * a;
}

/* a type's base alignment and size under std140 (0) or std430 (1) */
static void
mem_layout(int t, int std430, int *align, int *size)
{
	const struct sty *y = T(t);
	int a, s, i, off = 0, maxa = 1;

	switch (y->k) {
	case S_VEC:
		mem_layout(y->e, std430, &a, &s);
		*align = y->n == 2 ? 2 * a : 4 * a;
		*size = y->n * s;
		return;
	case S_MAT:
		/* an array of its columns */
		mem_layout(y->e, std430, &a, &s);
		if (!std430)
			a = mem_roundup(a, 16);
		*align = a;
		*size = y->n * mem_roundup(s, a);
		return;
	case S_ARRAY:
		mem_layout(y->e, std430, &a, &s);
		if (!std430)
			a = mem_roundup(a, 16);
		*align = a;
		*size = y->n * mem_roundup(s, a);
		return;
	case S_STRUCT:
		for (i = 0; i < y->nm; i++) {
			mem_layout(y->m[i], std430, &a, &s);
			off = mem_roundup(off, a) + s;
			if (a > maxa)
				maxa = a;
		}
		if (!std430)
			maxa = mem_roundup(maxa, 16);
		*align = maxa;
		*size = mem_roundup(off, maxa);
		return;
	}
	*align = *size = y->w > 0 ? (y->w + 7) / 8 : 4;
	if (y->k == S_BOOL)
		*align = *size = 4;
}

static const char *
mem_member(int st, int i)
{
	static char b[64];
	int q;

	snprintf(b, sizeof b, "member %d", i);
	for (q = 0; q < V.nin; q++)
		if (!strcmp(V.in[q].op, "OpMemberName") && V.in[q].nop >= 3 &&
		    idx(V.in[q].opnd[0]) == st && atoi(V.in[q].opnd[1]) == i)
			snprintf(b, sizeof b, "%.*s", (int)strlen(V.in[q].opnd[2]) - 2,
			    V.in[q].opnd[2] + 1);
	return b;
}

/* the block's members whose offset or stride changes from std140 to
 * std430, a line each; the count */
static int
mem_layout_diff(int st, struct sbuf *o)
{
	const struct sty *y = T(st);
	int i, off140 = 0, off430 = 0, a1, s1, a4, s4, n = 0;

	for (i = 0; i < y->nm; i++) {
		const struct sty *m = T(y->m[i]);
		mem_layout(y->m[i], 0, &a1, &s1);
		mem_layout(y->m[i], 1, &a4, &s4);
		off140 = mem_roundup(off140, a1);
		off430 = mem_roundup(off430, a4);
		if (off140 != off430 || s1 != s4) {
			sb_printf(o, "        %s: offset %d -> %d", mem_member(st, i),
			    off140, off430);
			if (m->k == S_ARRAY || m->k == S_MAT) {
				int ea, es, fa, fs;
				mem_layout(m->e, 0, &ea, &es);
				mem_layout(m->e, 1, &fa, &fs);
				sb_printf(o, ", %s stride %d -> %d", m->k == S_ARRAY ?
				    "array" : "column", mem_roundup(es, mem_roundup(ea,
				    16)), mem_roundup(fs, fa));
			}
			sb_printf(o, ", size %d -> %d bytes\n", s1, s4);
			n++;
		}
		off140 += s1;
		off430 += s4;
	}
	return n;
}

/* ---- an integer or float value as a ucas expression ---- */

struct memg {
	struct func f;
	int psid[MAXP];
	int id[256];
	nid node[256];
	int nmemo;
	int lsize;		/* the lane parameter's range: LocalSize.x */
	int fail;
};

static ty_t
mem_ty(int t)
{
	if (t < 0 || V.id[t].kind != K_TYPE)
		return 0;
	if (T(t)->k == S_INT && T(t)->w <= 64)
		return TY(T(t)->sign ? K_SINT : K_UINT, T(t)->w);
	if (T(t)->k == S_FLOAT && (T(t)->w == 32 || T(t)->w == 64))
		return TY(K_FLOAT, T(t)->w);
	if (T(t)->k == S_BOOL)
		return TY_BOOL;
	return 0;
}

/*
 * A loop counter: a phi of a constant and of itself plus a positive
 * constant, compared below a constant where the loop is left; its range.
 */
static int
mem_counter(int id, int64_t *lo, int64_t *hi)
{
	int k = SP.def[id], q, init = -1, step = -1, t;
	const struct sins *I;

	if (k < 0 || strcmp(V.in[k].op, "OpPhi") != 0 || V.in[k].nop != 5)
		return 0;
	I = &V.in[k];
	for (q = 1; q <= 3; q += 2) {
		int v = idx(I->opnd[q]), kv = SP.def[v];
		if (V.id[v].kind == K_CONST && !V.id[v].spec)
			init = v;
		else if (kv >= 0 && !strcmp(V.in[kv].op, "OpIAdd") &&
		    V.in[kv].nop >= 3) {
			int a = idx(V.in[kv].opnd[1]), b = idx(V.in[kv].opnd[2]);
			if (a == id && V.id[b].kind == K_CONST)
				step = b;
			else if (b == id && V.id[a].kind == K_CONST)
				step = a;
		}
	}
	if (init < 0 || step < 0)
		return 0;
	t = V.id[init].type;
	if (geti(t, V.id[step].c[0]) <= 0)
		return 0;
	*lo = geti(t, V.id[init].c[0]);
	for (q = 0; q < V.nin; q++) {
		const struct sins *C = &V.in[q];
		int eq;
		if (C->nop < 3 || idx(C->opnd[1]) != id ||
		    V.id[idx(C->opnd[2])].kind != K_CONST)
			continue;
		if (!strcmp(C->op, "OpULessThan") || !strcmp(C->op, "OpSLessThan"))
			eq = 0;
		else if (!strcmp(C->op, "OpULessThanEqual") || !strcmp(C->op,
		    "OpSLessThanEqual"))
			eq = 1;
		else
			continue;
		*hi = geti(t, V.id[idx(C->opnd[2])].c[0]) - 1 + eq;
		return *hi >= *lo;
	}
	return 0;
}

static nid
memg_param(struct memg *G, int sid, ty_t ty, int uniform, int lane)
{
	char nb[48], nm[56];
	const char *n;
	struct param *p;
	int j;

	for (j = 0; j < G->f.np; j++)
		if (G->psid[j] == sid)
			return mk_var((uint32_t)j, G->f.p[j].name, G->f.p[j].ty);
	if (G->f.np == MAXP || ty == 0) {
		G->fail = 1;
		return mk_const_i(TY(K_UINT, 32), 0);
	}
	j = G->f.np++;
	G->psid[j] = sid;
	n = lane ? "lane" : load_name(sid, nb, sizeof nb);
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
	p->uniform = uniform;
	if (lane) {
		p->ranged = 1;
		p->lo = 0;
		p->hi = G->lsize > 0 ? G->lsize - 1 : 31;
	} else if (TY_ISINT(ty)) {
		int64_t lo, hi;
		if (assumed(sid, &lo, &hi) || mem_counter(sid, &lo, &hi)) {
			p->ranged = 1;
			p->lo = lo;
			p->hi = hi;
		}
	}
	return mk_var((uint32_t)j, p->name, ty);
}

static nid
memg_build(struct memg *G, int id, int depth)
{
	int k = SP.def[id], i, t = V.id[id].type;
	const struct sins *I;
	ty_t ty = mem_ty(t);
	nid r = 0, a, b;

	for (i = 0; i < G->nmemo; i++)
		if (G->id[i] == id)
			return G->node[i];
	if (V.id[id].kind == K_CONST && !V.id[id].spec && ty != 0) {
		if (TY_FP(ty))
			r = mk_cast(ty, mk_const_r(getf(t, V.id[id].c[0])));
		else
			r = mk_const_i(ty, geti(t, V.id[id].c[0]));
		goto done;
	}
	if (k < 0 || depth > 48 || ty == 0)
		return memg_param(G, id, ty, mem_uni[id], 0);
	I = &V.in[k];
	/* the lane: gl_LocalInvocationIndex, or gl_LocalInvocationID.x */
	if (!strcmp(I->op, "OpLoad")) {
		int chain, var = mem_base_var(idx(I->opnd[1]), &chain);
		if (var >= 0 && (V.id[var].builtin == B_LOCALINDEX ||
		    (V.id[var].builtin == B_LOCALID && chain >= 0 &&
		    V.in[chain].nop == 3 && V.id[idx(V.in[chain].opnd[2])].kind ==
		    K_CONST && V.id[idx(V.in[chain].opnd[2])].c[0] == 0)))
			return memg_param(G, var, ty, 0, 1);
		return memg_param(G, id, ty, mem_uni[id], 0);
	}
	if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 3 &&
	    atoi(I->opnd[2]) == 0) {
		int s = idx(I->opnd[1]), ks = SP.def[s];
		if (ks >= 0 && !strcmp(V.in[ks].op, "OpLoad")) {
			int chain, var = mem_base_var(idx(V.in[ks].opnd[1]), &chain);
			if (var >= 0 && V.id[var].builtin == B_LOCALID && chain < 0)
				return memg_param(G, var, ty, 0, 1);
		}
		return memg_param(G, id, ty, mem_uni[id], 0);
	}
	if (I->nop >= 3) {
		static const struct { const char *op; int o; } bin[] = {
			{"OpIAdd", O_ADD}, {"OpISub", O_SUB}, {"OpIMul", O_MUL},
			{"OpUDiv", O_DIV}, {"OpUMod", O_MOD},
			{"OpShiftLeftLogical", O_SHL}, {"OpShiftRightLogical", O_SHR},
			{"OpBitwiseAnd", O_AND}, {"OpBitwiseOr", O_OR},
			{"OpBitwiseXor", O_XOR}, {"OpFAdd", O_ADD}, {"OpFSub", O_SUB},
			{"OpFMul", O_MUL}, {"OpFDiv", O_DIV}, {NULL, 0}};
		for (i = 0; bin[i].op; i++)
			if (!strcmp(I->op, bin[i].op)) {
				a = memg_build(G, idx(I->opnd[1]), depth + 1);
				b = memg_build(G, idx(I->opnd[2]), depth + 1);
				/* the result type, as SPIR-V computes it */
				r = mk_cast(ty, mk_bin(bin[i].o, mk_cast(ty, a),
				    bin[i].o == O_SHL || bin[i].o == O_SHR ? b :
				    mk_cast(ty, b)));
				goto done;
			}
	}
	if (I->nop >= 2 && (!strcmp(I->op, "OpConvertUToF") || !strcmp(I->op,
	    "OpConvertSToF") || !strcmp(I->op, "OpUConvert") || !strcmp(I->op,
	    "OpSConvert") || !strcmp(I->op, "OpBitcast") || !strcmp(I->op,
	    "OpCopyObject"))) {
		ty_t ta = mem_ty(V.id[idx(I->opnd[1])].type);
		if (ta != 0 && (strcmp(I->op, "OpBitcast") != 0 || (TY_ISINT(ta)
		    && TY_ISINT(ty)))) {
			r = mk_cast(ty, memg_build(G, idx(I->opnd[1]), depth + 1));
			goto done;
		}
	}
	if (I->nop >= 2 && (!strcmp(I->op, "OpSNegate") || !strcmp(I->op,
	    "OpFNegate"))) {
		r = mk_un(O_NEG, memg_build(G, idx(I->opnd[1]), depth + 1));
		goto done;
	}
	return memg_param(G, id, ty, mem_uni[id], 0);
done:
	if (G->nmemo < 256) {
		G->id[G->nmemo] = id;
		G->node[G->nmemo++] = r;
	}
	return r;
}

/* ---- the pass ---- */

#define MEM_MAXARR 8
#define MEM_MAXACC 8

static char *
mem_pass(const char *text, int *changed)
{
	const struct model *m = model_active();
	struct sbuf out = {0}, decl = {0};
	char **pre, sl[200];
	int k, i, hdr = 0, lsize[3] = {0, 0, 0}, serial = 0;
	int nubo = 0, nubodiv = 0, nfetch = 0, nfatom = 0;
	struct {
		int var, chain[MEM_MAXACC], nch, total;
	} arr[MEM_MAXARR];
	int narr = 0;
	int *newidx;		/* per access chain: the swizzled index id name */
	char (*newname)[40];

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
	for (i = 0; i < V.nid + 64; i++)
		SP.def[i] = -1;
	{
		int inbody = 0;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (!strcmp(I->op, "OpFunction"))
				inbody = 1;
			if (!strcmp(I->op, "OpExecutionMode") && I->nop >= 5 &&
			    !strcmp(I->opnd[1], "LocalSize"))
				for (i = 0; i < 3; i++)
					lsize[i] = atoi(I->opnd[2 + i]);
			if (inbody && I->res >= 0) {
				SP.def[I->res] = k;
				if (I->nop >= 1 && I->opnd[0][0] == '%' &&
				    V.id[idx(I->opnd[0])].kind == K_TYPE)
					V.id[I->res].type = idx(I->opnd[0]);
			}
		}
	}
	mem_uniformity();
	pre = xcalloc((size_t)V.nin + 1, sizeof *pre);
	newidx = xcalloc((size_t)V.nin + 1, sizeof *newidx);
	newname = xcalloc((size_t)V.nin + 1, sizeof *newname);
#define MEM_HDR() do { \
	if (!hdr++) \
		printf("memory under %s:\n", sym_str(m->name)); \
} while (0)
	/* block reads at a divergent address */
	{
		int blocks[16], nb = 0, b;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			int chain, var, div = 0;
			if (strcmp(I->op, "OpLoad") != 0 || I->res < 0 ||
			    SP.def[I->res] != k)
				continue;
			var = mem_base_var(idx(I->opnd[1]), &chain);
			if (var < 0 || strcmp(mem_sc(var), "Uniform") != 0 ||
			    mem_bufferblock(var))
				continue;
			nubo++;
			if (chain >= 0)
				for (i = 2; i < V.in[chain].nop; i++)
					if (!mem_uni[idx(V.in[chain].opnd[i])])
						div = 1;
			if (!div)
				continue;
			nubodiv++;
			for (b = 0; b < nb; b++)
				if (blocks[b] == var)
					break;
			if (b == nb && nb < 16)
				blocks[nb++] = var;
		}
		for (b = 0; b < nb; b++) {
			double cdiv = (double)model_cost(m, C_LOAD_UBO_DIV);
			double cuni = (double)model_cost(m, C_LOAD_UBO);
			double cst = (double)m->c[C_LOAD];
			int var = blocks[b], st, n = 0, shown = 0;
			struct sbuf ld = {0};
			const char *bn = V.id[var].name;

			if (!(cdiv > cst))
				continue;	/* the model sees nothing to gain */
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				int chain, v2, div = 0;
				if (strcmp(I->op, "OpLoad") != 0 || I->res < 0)
					continue;
				v2 = mem_base_var(idx(I->opnd[1]), &chain);
				if (v2 != var || chain < 0)
					continue;
				for (i = 2; i < V.in[chain].nop; i++)
					if (!mem_uni[idx(V.in[chain].opnd[i])])
						div = 1;
				if (!div)
					continue;
				n++;
				if (!shown++) {
					MEM_HDR();
					printf("  block %s (a uniform buffer) is read at an "
					    "address that differs per lane:\n", *bn == '%' ?
					    bn + 1 : bn);
				}
				if (shown <= 4) {
					printf("    %s\n", I->text);
					src_line(ins_line(k), sl, sizeof sl);
					fputs(sl, stdout);
				}
				shown++;
			}
			if (n == 0)
				continue;
			if (n > 3)
				printf("    ... %d such reads\n", n);
			printf("    each is priced load_ubo_div %g (a uniform address: "
			    "load_ubo %g); a storage-buffer load is %g\n", cdiv, cuni,
			    cst);
			st = T(V.id[var].type)->e;
			while (T(st)->k == S_ARRAY)
				st = T(st)->e;
			printf("    proposal (an interface change, not applied): "
			    "declare the block `layout(std430) readonly buffer` "
			    "instead of\n    `layout(std140) uniform`: %d read%s at "
			    "%g instead of %g (%+g)\n", n, n == 1 ? "" : "s", cst, cdiv,
			    n * (cst - cdiv));
			printf("      the host then binds it as a storage buffer: "
			    "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER in the set layout and "
			    "the\n      write, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT on "
			    "the buffer (its size within maxStorageBufferRange)\n");
			if (T(st)->k == S_STRUCT && mem_layout_diff(st, &ld) > 0)
				printf("      and fills it by std430, which packs arrays "
				    "and structs tighter than std140:\n%s", ld.p);
			else
				printf("      its bytes stay as they are: no member's "
				    "offset or stride differs between std140 and "
				    "std430\n");
			printf("      reads at a uniform address would then cost %g "
			    "instead of %g: worth it when the divergent ones "
			    "dominate\n", cst, cuni);
			free(ld.p);
		}
	}
	/* texel fetches */
	{
		double cf = (double)model_cost(m, C_TEX_FETCH);
		double cs = (double)model_cost(m, C_TEX_SAMPLE);
		int shown = 0;
		for (k = 0; k < V.nin && cs < cf; k++) {
			const struct sins *I = &V.in[k];
			if (strcmp(I->op, "OpImageFetch") != 0 || I->res < 0)
				continue;
			nfetch++;
			if (!shown++) {
				MEM_HDR();
				printf("  texel fetches (tex_fetch %g; a sampled read, "
				    "tex_sample, %g):\n", cf, cs);
			}
			if (shown <= 4) {
				printf("    %s\n", I->text);
				src_line(ins_line(k), sl, sizeof sl);
				fputs(sl, stdout);
			}
		}
		if (nfetch) {
			printf("    a candidate worth testing (not applied: it needs "
			    "a sampler, and the model's price is for bilinear "
			    "rgba8):\n      textureLod(sampler2D(tex, nearest), "
			    "(vec2(p) + 0.5) / vec2(textureSize(tex, 0)), 0.0) for "
			    "texelFetch(tex, p, 0):\n      %d fetch%s at %g instead "
			    "of %g (%+g)\n", nfetch, nfetch == 1 ? "" : "es", cs, cf,
			    nfetch * (cs - cf));
			printf("      the same texel, for every size up to 2^20 and "
			    "0 <= p < size: p + 0.5 is exact in f32 (p < 2^23);\n"
			    "      the quotient is within 2.5 ulp (Vulkan's OpFDiv), "
			    "so u * size is within (p + 0.5) * 2.5 * 2^-23 < 0.32\n"
			    "      of p + 0.5; the sampler's fixed point (at least "
			    "4 sub-texel bits) moves it by at most 1/32 more;\n"
			    "      nearest takes floor(u * size) = p.  Out of "
			    "range a fetch is undefined, a sample clamps: a "
			    "refinement.\n");
			printf("      the sampler: magFilter = minFilter = NEAREST, "
			    "mipmapMode = NEAREST, any address mode, no "
			    "anisotropy,\n      no compare; with "
			    "unnormalizedCoordinates = VK_TRUE the coordinate is "
			    "vec2(p) + 0.5 and nothing is divided\n");
		}
	}
	/* float atomic adds on workgroup memory; the workgroup arrays */
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int chain, var, pi = -1, a;

		if (!strcmp(I->op, "OpLoad") && I->nop >= 2)
			pi = 1;
		else if (!strcmp(I->op, "OpStore") && I->nop >= 2)
			pi = 0;
		else if (!strncmp(I->op, "OpAtomic", 8) && I->nop >= 2)
			pi = 1;
		if (pi < 0 || I->opnd[pi][0] != '%')
			continue;
		var = mem_base_var(idx(I->opnd[pi]), &chain);
		if (var < 0 || strcmp(mem_sc(var), "Workgroup") != 0)
			continue;
		if (!strcmp(I->op, "OpAtomicFAddEXT") && I->nop >= 5) {
			struct memg *G = xcalloc(1, sizeof *G);
			nid v;
			G->lsize = lsize[0];
			G->f.name = sym_cstr("_fadd");
			G->f.anon = 1;
			v = memg_build(G, idx(I->opnd[4]), 0);
			nfatom++;
			MEM_HDR();
			printf("  %s\n", I->text);
			src_line(ins_line(k), sl, sizeof sl);
			fputs(sl, stdout);
			if (G->fail || !TY_FP(N.ty[v]))
				printf("    its addend is not read as an expression "
				    "(more than 8 inputs, or a type outside f32 and "
				    "the integers)\n");
			else {
				long cnt = (long)(lsize[0] > 0 ? lsize[0] : 1) *
				    (lsize[1] > 0 ? lsize[1] : 1) * (lsize[2] > 0 ?
				    lsize[2] : 1);
				G->f.body = v;
				printf("    (count: the workgroup's %ld invocations, "
				    "once each; the array and its other accesses are "
				    "not rewritten here:\n    the change below is for "
				    "every access of it)\n", cnt);
				memk_intatomic(&G->f, cnt, 0, 0, m);
			}
			free(G);
		}
		if (chain < 0 || V.in[chain].nop != 3)
			continue;
		for (a = 0; a < narr; a++)
			if (arr[a].var == var)
				break;
		if (a == narr) {
			if (narr == MEM_MAXARR)
				continue;
			arr[narr].var = var;
			arr[narr].nch = 0;
			arr[narr++].total = 0;
		}
		arr[a].total++;
		for (i = 0; i < arr[a].nch; i++)
			if (arr[a].chain[i] == chain)
				break;
		if (i == arr[a].nch && arr[a].nch < MEM_MAXACC)
			arr[a].chain[arr[a].nch++] = chain;
		else if (i == arr[a].nch)
			arr[a].nch = MEM_MAXACC + 1;	/* too many to search */
	}
	if ((int)m->par[MP_BANKS] > 0 && model_cost(m, C_BANK_CONFLICT) > 0)
		for (i = 0; i < narr; i++) {
			struct memg *G[MEM_MAXACC];
			uint32_t names[MEM_MAXACC];
			struct bankswz z;
			const char *an = V.id[arr[i].var].name;
			int at = T(V.id[arr[i].var].type)->e, elem = 4, al, es, q;
			int ok = 1, worst = 1;

			if (arr[i].nch > MEM_MAXACC || T(at)->k != S_ARRAY ||
			    (lsize[1] > 1 || lsize[2] > 1))
				continue;
			mem_layout(T(at)->e, 1, &al, &es);
			elem = es;
			for (q = 0; q < arr[i].nch; q++) {
				struct func *h;
				char nm[40];
				struct bankres br;
				G[q] = xcalloc(1, sizeof **G);
				G[q]->lsize = lsize[0];
				G[q]->f.body = mk_cast(TY(K_UINT, 32), memg_build(G[q],
				    idx(V.in[arr[i].chain[q]].opnd[2]), 0));
				if (G[q]->fail)
					ok = 0;
				snprintf(nm, sizeof nm, "_%.30s_%d", *an == '%' ? an + 1 :
				    an, q + 1);
				G[q]->f.name = sym_cstr(nm);
				h = func_new(G[q]->f.name);
				*h = G[q]->f;
				names[q] = h->name;
				if (ok && bank_degree(h, h->body, m, elem, &br) > 0 &&
				    br.degree > worst)
					worst = br.degree;
			}
			/* nothing to say about an array without conflicts */
			if (ok && worst > 1) {
				MEM_HDR();
				printf("  workgroup array %s (%d elements of %d bytes, "
				    "%d access%s):\n", *an == '%' ? an + 1 : an,
				    T(at)->n, elem, arr[i].nch, arr[i].nch == 1 ? "" :
				    "es");
				if (memk_banks(names, arr[i].nch, T(at)->n, elem, m,
				    &z)) {
					/* every access chain takes the swizzle */
					for (q = 0; q < arr[i].nch; q++) {
						int ck = arr[i].chain[q];
						const char *ix = V.in[ck].opnd[2];
						int it = V.id[idx(ix)].type;
						const char *tt = V.id[it].tok;
						struct sbuf c = {0};
						char r[40], t1[40], kc[2][48];
						int banks = (int)m->par[MP_BANKS], cv[2], j;
						cv[0] = z.s;
						cv[1] = banks - 1;
						for (j = 0; j < (z.kind == SW_PAD ? 1 : 2); j++) {
							snprintf(kc[j], sizeof kc[j], "%%ucas_m%s_%d",
							    tt + 1, cv[j]);
							if (decl.p == NULL || !strstr(decl.p, kc[j]))
								sb_printf(&decl, "%s = OpConstant %s "
								    "%d\n", kc[j], tt, cv[j]);
						}
						snprintf(r, sizeof r, "%%ucas_sw%d", ++serial);
						snprintf(t1, sizeof t1, "%%ucas_sw%d", ++serial);
						snprintf(newname[ck], 40, "%%ucas_sw%d", ++serial);
						sb_printf(&c, "%s = OpShiftRightLogical %s %s "
						    "%s\n", r, tt, ix, kc[0]);
						if (z.kind == SW_XOR) {
							sb_printf(&c, "%s = OpBitwiseAnd %s %s %s\n",
							    t1, tt, r, kc[1]);
							sb_printf(&c, "%s = OpBitwiseXor %s %s %s\n",
							    newname[ck], tt, ix, t1);
						} else if (z.kind == SW_PAD) {
							sb_printf(&c, "%s = OpIAdd %s %s %s\n",
							    newname[ck], tt, ix, r);
						} else {
							char t2[40], t3[40], km[48];
							snprintf(t2, sizeof t2, "%%ucas_sw%d",
							    ++serial);
							snprintf(t3, sizeof t3, "%%ucas_sw%d",
							    ++serial);
							snprintf(km, sizeof km, "%%ucas_mn%s_%d",
							    tt + 1, banks - 1);
							if (decl.p == NULL || !strstr(decl.p, km))
								sb_printf(&decl, "%s = OpConstant %s "
								    "%lld\n", km, tt, T(it)->sign ?
								    (long long)~(int64_t)(banks - 1) :
								    (long long)(uint32_t)~(uint32_t)(banks
								    - 1));
							sb_printf(&c, "%s = OpIAdd %s %s %s\n", t1, tt,
							    ix, r);
							sb_printf(&c, "%s = OpBitwiseAnd %s %s %s\n",
							    t2, tt, t1, kc[1]);
							sb_printf(&c, "%s = OpBitwiseAnd %s %s %s\n",
							    t3, tt, ix, km);
							sb_printf(&c, "%s = OpBitwiseOr %s %s %s\n",
							    newname[ck], tt, t3, t2);
						}
						pre[ck] = c.p;
						newidx[ck] = 1;
					}
					if (z.kind == SW_PAD) {
						/* the array grows: a new length for its type */
						for (q = 0; q < V.nin; q++)
							if (V.in[q].res == at && !strcmp(V.in[q].op,
							    "OpTypeArray") && V.in[q].nop >= 2) {
								struct sbuf c = {0};
								int lt = V.id[idx(V.in[q].opnd[1])].type;
								sb_printf(&c, "%%ucas_len%d = OpConstant %s "
								    "%lld\n", i, V.id[lt].tok,
								    (long long)z.size);
								pre[q] = c.p;
								newidx[q] = 2;
								snprintf(newname[q], 40, "%%ucas_len%d", i);
							}
					}
					printf("    written into the module: the %d access "
					    "chain%s of the array index it through the "
					    "swizzle\n", arr[i].nch, arr[i].nch == 1 ? "" :
					    "s");
					*changed = 1;
				}
			}
			for (q = 0; q < arr[i].nch; q++)
				free(G[q]);
		}
	if (*changed) {
		int first_fn = -1;
		for (k = 0; k < V.nin; k++)
			if (!strcmp(V.in[k].op, "OpFunction")) {
				first_fn = k;
				break;
			}
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (k == first_fn && decl.p)
				sb_puts(&out, decl.p);
			if (pre[k])
				sb_puts(&out, pre[k]);
			if (newidx[k] && I->res >= 0) {
				/* the access chain's index, or the array's length */
				sb_printf(&out, "%s = %s", V.id[I->res].tok, I->op);
				for (i = 0; i < I->nop; i++)
					sb_printf(&out, " %s", (newidx[k] == 1 && i == 2) ||
					    (newidx[k] == 2 && i == 1) ? newname[k] :
					    I->opnd[i]);
				sb_putc(&out, '\n');
			} else
				sb_printf(&out, "%s\n", I->text);
		}
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
	(void)nubo;
	(void)nubodiv;
	(void)nfatom;
#undef MEM_HDR
	for (k = 0; k < V.nin; k++)
		free(pre[k]);
	free(pre);
	free(newidx);
	free(newname);
	free(decl.p);
	free(mem_uni);
	mem_uni = NULL;
	free(SP.def);
	SP.def = NULL;
	for (k = 0; k < V.nin; k++) {
		for (i = 0; i < V.in[k].nop; i++)
			free(V.in[k].opnd[i]);
		free(V.in[k].text);
	}
	free(V.in);
	memset(&V, 0, sizeof V);
	if (!*changed) {
		free(out.p);
		return NULL;
	}
	return out.p;
}
