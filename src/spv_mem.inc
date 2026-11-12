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
 * per lane.  A branch on a varying condition splits the warp, and what
 * depends on which side a lane took varies with it: the phis of the
 * blocks where two paths from the branch that share no other block meet,
 * and, when the branch lets lanes leave a loop at different iterations,
 * every value read after that loop that it changes as it goes round.
 * Nothing else does: a phi
 * after an unrelated varying branch stays uniform when its inputs are.
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

/*
 * The control flow of the bodies: blocks, their edges, the loops (a
 * header's OpLoopMerge names the merge block; the loop is what the header
 * reaches without passing it, since structured control flow leaves a loop
 * only there or by ending the function).
 */
static struct {
	int nblk;
	int *ins_blk;		/* per instruction: its block, -1 outside one */
	int *lab_blk;		/* per id: the block the label opens, -1 */
	int *term;		/* per block: its last instruction */
	int *poff, *pred;	/* per block: its predecessors */
	int nloop;
	int *lhdr, *lmerge;	/* per loop: the header and merge blocks */
	char *lin;		/* nloop x nblk: the block is in the loop */
	char *ldiv;		/* per loop: lanes leave at different iterations */
	char *itv;		/* per id: may differ from one iteration to the next */
	char *join;		/* per block: paths from a varying branch meet */
	char *done;		/* per block: its varying branch is accounted */
} MC;

static int
mem_is_label(const char *o)
{
	return o[0] == '%' && MC.lab_blk[idx(o)] >= 0;
}

static void
mem_cfg(void)
{
	int k, b = -1, i, e, l, *cnt, *stack, sp;

	memset(&MC, 0, sizeof MC);
	MC.ins_blk = xmalloc((size_t)(V.nin + 1) * sizeof *MC.ins_blk);
	MC.lab_blk = xmalloc((size_t)(V.nid + 1) * sizeof *MC.lab_blk);
	for (i = 0; i <= V.nid; i++)
		MC.lab_blk[i] = -1;
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strcmp(I->op, "OpLabel") && I->res >= 0) {
			b = MC.nblk++;
			MC.lab_blk[I->res] = b;
		} else if (!strcmp(I->op, "OpFunctionEnd") || !strcmp(I->op,
		    "OpFunction"))
			b = -1;
		MC.ins_blk[k] = b;
	}
	MC.term = xmalloc((size_t)(MC.nblk + 1) * sizeof *MC.term);
	for (k = 0; k < V.nin; k++)
		if (MC.ins_blk[k] >= 0)
			MC.term[MC.ins_blk[k]] = k;
	/* the predecessors, and the loops */
	cnt = xcalloc((size_t)MC.nblk + 2, sizeof *cnt);
	for (b = 0; b < MC.nblk; b++) {
		const struct sins *I = &V.in[MC.term[b]];
		for (i = 0; i < I->nop; i++)
			if (mem_is_label(I->opnd[i]))
				cnt[MC.lab_blk[idx(I->opnd[i])] + 1]++;
		if (MC.term[b] > 0 && !strcmp(V.in[MC.term[b] - 1].op,
		    "OpLoopMerge") && MC.ins_blk[MC.term[b] - 1] == b)
			MC.nloop++;
	}
	MC.poff = xmalloc((size_t)(MC.nblk + 2) * sizeof *MC.poff);
	MC.poff[0] = 0;
	for (b = 0; b < MC.nblk; b++)
		MC.poff[b + 1] = MC.poff[b] + cnt[b + 1];
	MC.pred = xmalloc((size_t)(MC.poff[MC.nblk] + 1) * sizeof *MC.pred);
	memset(cnt, 0, ((size_t)MC.nblk + 2) * sizeof *cnt);
	for (b = 0; b < MC.nblk; b++) {
		const struct sins *I = &V.in[MC.term[b]];
		for (i = 0; i < I->nop; i++)
			if (mem_is_label(I->opnd[i])) {
				e = MC.lab_blk[idx(I->opnd[i])];
				MC.pred[MC.poff[e] + cnt[e]++] = b;
			}
	}
	free(cnt);
	MC.lhdr = xmalloc((size_t)(MC.nloop + 1) * sizeof *MC.lhdr);
	MC.lmerge = xmalloc((size_t)(MC.nloop + 1) * sizeof *MC.lmerge);
	MC.lin = xcalloc((size_t)(MC.nloop + 1) * (size_t)(MC.nblk + 1), 1);
	MC.ldiv = xcalloc((size_t)MC.nloop + 1, 1);
	MC.join = xcalloc((size_t)MC.nblk + 1, 1);
	MC.done = xcalloc((size_t)MC.nblk + 1, 1);
	stack = xmalloc((size_t)(MC.nblk + 1) * sizeof *stack);
	for (b = 0, l = 0; b < MC.nblk; b++) {
		const struct sins *M;
		char *in;
		if (MC.term[b] == 0 || MC.ins_blk[MC.term[b] - 1] != b)
			continue;
		M = &V.in[MC.term[b] - 1];
		if (strcmp(M->op, "OpLoopMerge") != 0 || M->nop < 1 ||
		    !mem_is_label(M->opnd[0]))
			continue;
		MC.lhdr[l] = b;
		MC.lmerge[l] = MC.lab_blk[idx(M->opnd[0])];
		in = MC.lin + (size_t)l * (size_t)(MC.nblk + 1);
		sp = 0;
		stack[sp++] = b;
		in[b] = 1;
		while (sp > 0) {
			const struct sins *I = &V.in[MC.term[stack[--sp]]];
			for (i = 0; i < I->nop; i++) {
				if (!mem_is_label(I->opnd[i]))
					continue;
				e = MC.lab_blk[idx(I->opnd[i])];
				if (e == MC.lmerge[l] || in[e])
					continue;
				in[e] = 1;
				stack[sp++] = e;
			}
		}
		l++;
	}
	MC.nloop = l;
	free(stack);
	/* what a loop can change as it goes round: its phis and what is
	 * computed from them (memory a loop writes is varying already) */
	MC.itv = xcalloc((size_t)V.nid + 1, 1);
	for (e = 1; e;) {
		e = 0;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			int r = I->res, v = 0;
			if (r < 0 || SP.def[r] != k || MC.itv[r] || (b = MC.ins_blk[k]) < 0)
				continue;
			for (l = 0; l < MC.nloop && !v; l++)
				v = MC.lin[(size_t)l * (size_t)(MC.nblk + 1) + (size_t)b];
			if (!v)
				continue;
			v = !strcmp(I->op, "OpPhi");
			for (i = 1; i < I->nop && !v; i++)
				v = I->opnd[i][0] == '%' && MC.itv[idx(I->opnd[i])];
			if (v)
				MC.itv[r] = e = 1;
		}
	}
}

static void
mem_cfg_free(void)
{
	free(MC.ins_blk);
	free(MC.lab_blk);
	free(MC.term);
	free(MC.poff);
	free(MC.pred);
	free(MC.lhdr);
	free(MC.lmerge);
	free(MC.lin);
	free(MC.ldiv);
	free(MC.itv);
	free(MC.join);
	free(MC.done);
	memset(&MC, 0, sizeof MC);
}

/* value id, read by instruction k, is the same in every lane there */
static int
mem_uni_at(int id, int k)
{
	int d, l, bd, bu;

	if (!mem_uni[id])
		return 0;
	d = SP.def[id];
	if (d < 0 || (bd = MC.ins_blk[d]) < 0 || !MC.itv[id])
		return 1;
	bu = k >= 0 ? MC.ins_blk[k] : -1;
	for (l = 0; l < MC.nloop; l++) {
		const char *in = MC.lin + (size_t)l * (size_t)(MC.nblk + 1);
		if (MC.ldiv[l] && in[bd] && (bu < 0 || !in[bu]))
			return 0;
	}
	return 1;
}

/*
 * The branch that ends block h has a varying condition.  Each block the
 * branch reaches is labelled with the last block every path to it from
 * the branch goes through: the target of an edge of the branch for what
 * only that edge reaches, the block itself where paths with different
 * labels meet.  Such a block is a join: two paths from the branch that
 * share nothing else end in it, so which predecessor a lane came from
 * depends on the condition.  A loop around the branch whose merge block
 * and whose back edge carry different labels is left by some lanes while
 * others go round.  1 when a join or a loop was newly marked.
 */
static int
mem_split(int h)
{
	int *lab = xmalloc((size_t)(MC.nblk + 1) * sizeof *lab);
	const char **hin = xcalloc((size_t)MC.nblk + 1, sizeof *hin);
	int b, q, l, pass, ch = 1, changed = 0;

	for (b = 0; b < MC.nblk; b++)
		lab[b] = -1;
	/* lanes that left a loop around the branch meet again at its merge
	 * block before any of them is back at its header: the header of
	 * such a loop hears of the branch by its back edge alone */
	for (l = 0; l < MC.nloop; l++) {
		const char *in = MC.lin + (size_t)l * (size_t)(MC.nblk + 1);
		if (in[h])
			hin[MC.lhdr[l]] = in;
	}
	for (pass = 0; ch && pass < 2 * MC.nblk + 8; pass++) {
		ch = 0;
		for (b = 0; b < MC.nblk; b++) {
			int s = -1;
			for (q = MC.poff[b]; q < MC.poff[b + 1]; q++) {
				int c = MC.pred[q] == h ? b : lab[MC.pred[q]];
				if (c < 0 || (hin[b] && !hin[b][MC.pred[q]]))
					continue;
				if (s < 0)
					s = c;
				else if (s != c)
					s = b;
			}
			if (s != lab[b]) {
				lab[b] = s;
				ch = 1;
			}
		}
	}
	for (b = 0; b < MC.nblk; b++) {
		int nlab = 0, first = -1;
		if (lab[b] < 0)
			continue;
		for (q = MC.poff[b]; q < MC.poff[b + 1]; q++) {
			int c = MC.pred[q] == h ? b : lab[MC.pred[q]];
			if (c < 0 || (hin[b] && !hin[b][MC.pred[q]]))
				continue;
			if (nlab == 0) {
				first = c;
				nlab = 1;
			} else if (c != first)
				nlab = 2;
		}
		/* labels still moving: every block the branch reaches */
		if ((nlab == 2 || ch) && !MC.join[b]) {
			MC.join[b] = 1;
			changed = 1;
		}
	}
	for (l = 0; l < MC.nloop; l++) {
		const char *in = MC.lin + (size_t)l * (size_t)(MC.nblk + 1);
		int hd = MC.lhdr[l], lh = -1, lm = lab[MC.lmerge[l]];
		if (!in[h] || MC.ldiv[l])
			continue;
		/* what comes round by the back edge */
		for (q = MC.poff[hd]; q < MC.poff[hd + 1]; q++) {
			int c = MC.pred[q] == h ? hd : lab[MC.pred[q]];
			if (in[MC.pred[q]] && c >= 0)
				lh = c;
		}
		if (ch ? (lh >= 0 || lm >= 0) : (lh >= 0 && lm >= 0 && lh != lm)) {
			MC.ldiv[l] = 1;
			changed = 1;
		}
	}
	free(lab);
	free(hin);
	return changed;
}

static void
mem_uniformity(void)
{
	int k, i, b, ch = 1, more = 1;

	/* everything starts uniform and falls: the greatest fixed point */
	mem_uni = xmalloc((size_t)V.nid + 1);
	memset(mem_uni, 1, (size_t)V.nid + 1);
	mem_cfg();
	while (more) {
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
								if (!mem_uni_at(idx(V.in[chain].opnd[i]),
								    k))
									u = 0;
					} else
						u = 0;
				} else if (!strcmp(I->op, "OpPhi")) {
					if (MC.ins_blk[k] >= 0 && MC.join[MC.ins_blk[k]])
						u = 0;
					for (i = 1; i < I->nop && u; i += 2)
						if (!mem_uni_at(idx(I->opnd[i]), k))
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
						    !mem_uni_at(idx(I->opnd[i]), k))
							u = 0;
				if (!u) {
					mem_uni[r] = 0;
					ch = 1;
				}
			}
		}
		more = 0;
		for (b = 0; b < MC.nblk; b++) {
			const struct sins *I = &V.in[MC.term[b]];
			if (MC.done[b] || (strcmp(I->op, "OpBranchConditional") != 0 &&
			    strcmp(I->op, "OpSwitch") != 0) || I->nop < 1 ||
			    I->opnd[0][0] != '%' || mem_uni_at(idx(I->opnd[0]),
			    MC.term[b]))
				continue;
			MC.done[b] = 1;
			more |= mem_split(b);
		}
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
	int at;			/* the instruction that reads the value */
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
		return memg_param(G, id, ty, mem_uni_at(id, G->at), 0);
	I = &V.in[k];
	/* the lane: gl_LocalInvocationIndex, or gl_LocalInvocationID.x */
	if (!strcmp(I->op, "OpLoad")) {
		int chain, var = mem_base_var(idx(I->opnd[1]), &chain);
		if (var >= 0 && (V.id[var].builtin == B_LOCALINDEX ||
		    (V.id[var].builtin == B_LOCALID && chain >= 0 &&
		    V.in[chain].nop == 3 && V.id[idx(V.in[chain].opnd[2])].kind ==
		    K_CONST && V.id[idx(V.in[chain].opnd[2])].c[0] == 0))) {
			r = memg_param(G, var, ty, 0, 1);
			goto done;
		}
		return memg_param(G, id, ty, mem_uni_at(id, G->at), 0);
	}
	if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 3 &&
	    atoi(I->opnd[2]) == 0) {
		int s = idx(I->opnd[1]), ks = SP.def[s];
		if (ks >= 0 && !strcmp(V.in[ks].op, "OpLoad")) {
			int chain, var = mem_base_var(idx(V.in[ks].opnd[1]), &chain);
			if (var >= 0 && V.id[var].builtin == B_LOCALID && chain < 0) {
				r = memg_param(G, var, ty, 0, 1);
				goto done;
			}
		}
		return memg_param(G, id, ty, mem_uni_at(id, G->at), 0);
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
	return memg_param(G, id, ty, mem_uni_at(id, G->at), 0);
done:
	if (G->nmemo < 256) {
		G->id[G->nmemo] = id;
		G->node[G->nmemo++] = r;
	}
	return r;
}

#define MEM_MAXARR 8
#define MEM_MAXACC 8

/* ---- a float atomic add as an integer one, written in ---- */

#define MEM_MAXLEV 8

/* where mem_pass's rewrites go */
struct mem_out {
	char **pre;		/* per instruction: what goes before it */
	char **repl;		/* per instruction: what replaces it ("" drops it) */
	char (*newty)[48];	/* per instruction: its result type, changed */
	struct sbuf *decl;	/* declarations before the first function */
	int *hdr;		/* the "memory under" line is out */
};

/* a workgroup variable that becomes i32: its types are declared after
 * the bank pass, which may grow the outermost array */
struct mem_retype {
	int var, kvar, nlev, lev[MEM_MAXLEV + 1], stem;
	char ity[48];
};

static void
mem_hdr(const struct mem_out *o, const struct model *m)
{
	if (!(*o->hdr)++)
		printf("memory under %s:\n", sym_str(m->name));
}

/* a number n such that no id of the module is named %<stem><n>_...: the
 * names written stay apart from the module's, a simplified one's too */
static int
mem_free_stem(const char *stem, int from)
{
	char p[48];
	size_t l;
	int i;

	for (;; from++) {
		snprintf(p, sizeof p, "%%%s%d_", stem, from);
		l = strlen(p);
		for (i = 0; i < V.nid; i++)
			if (!strncmp(V.id[i].tok, p, l))
				break;
		if (i == V.nid)
			return from;
	}
}

static int
mem_debug_op(const char *op)
{
	return !strcmp(op, "OpName") || !strcmp(op, "OpMemberName") ||
	    !strncmp(op, "OpDecorate", 10) || !strncmp(op, "OpMemberDecorate",
	    16) || !strcmp(op, "OpEntryPoint") || !strncmp(op,
	    "OpExecutionMode", 15) || !strcmp(op, "OpLine") || !strcmp(op,
	    "OpNoLine");
}

/* an instruction other than a name or a decoration reads id */
static int
mem_used(int id)
{
	int k, i;

	for (k = 0; k < V.nin; k++) {
		if (mem_debug_op(V.in[k].op))
			continue;
		for (i = 0; i < V.in[k].nop; i++)
			if (!strcmp(V.in[k].opnd[i], V.id[id].tok))
				return 1;
	}
	return 0;
}

/* text names tok (a whole id, not the start of a longer one) */
static int
mem_text_uses(const char *text, const char *tok)
{
	size_t l = strlen(tok);
	const char *p = text;

	while ((p = strstr(p, tok)) != NULL) {
		if (!isalnum((unsigned char)p[l]) && p[l] != '_')
			return 1;
		p += l;
	}
	return 0;
}

/* an instruction the module keeps, or text written into it, reads id */
static int
mem_live(int id, char **pre, char **repl)
{
	const char *tok = V.id[id].tok;
	int k, i;

	for (k = 0; k < V.nin; k++) {
		if (pre[k] && mem_text_uses(pre[k], tok))
			return 1;
		if (repl[k]) {
			if (mem_text_uses(repl[k], tok))
				return 1;
			continue;
		}
		if (mem_debug_op(V.in[k].op))
			continue;
		for (i = 0; i < V.in[k].nop; i++)
			if (!strcmp(V.in[k].opnd[i], tok))
				return 1;
	}
	return 0;
}

/* id's instruction is arithmetic or a conversion (what memg_build reads):
 * nothing but its value comes of it */
static int
mem_pure(int k)
{
	static const char *const ops[] = {"OpIAdd", "OpISub", "OpIMul",
		"OpUDiv", "OpUMod", "OpShiftLeftLogical", "OpShiftRightLogical",
		"OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor", "OpFAdd", "OpFSub",
		"OpFMul", "OpFDiv", "OpConvertUToF", "OpConvertSToF",
		"OpUConvert", "OpSConvert", "OpBitcast", "OpCopyObject",
		"OpSNegate", "OpFNegate", NULL};
	int i;

	for (i = 0; ops[i]; i++)
		if (!strcmp(V.in[k].op, ops[i]))
			return 1;
	return 0;
}

/* id is no longer read: its instruction goes, with its names and
 * decorations, and so do its operands' when nothing else reads them */
static void
mem_dce(int id, char **pre, char **repl, int depth)
{
	int k = id >= 0 ? SP.def[id] : -1, q, i;

	if (k < 0 || depth > 48 || repl[k] || !mem_pure(k) || mem_live(id, pre,
	    repl))
		return;
	repl[k] = xcalloc(1, 1);
	for (q = 0; q < V.nin; q++)
		if (mem_debug_op(V.in[q].op) && repl[q] == NULL &&
		    V.in[q].nop >= 1 && !strcmp(V.in[q].opnd[0], V.id[id].tok))
			repl[q] = xcalloc(1, 1);
	for (i = 1; i < V.in[k].nop; i++)
		if (V.in[k].opnd[i][0] == '%')
			mem_dce(idx(V.in[k].opnd[i]), pre, repl, depth + 1);
}

/* instruction k runs at most once an invocation: it is in an entry
 * point's function (which nothing calls) and in no loop */
static int
mem_once(int k)
{
	int q, fn = -1, l, b = MC.ins_blk[k];

	for (q = k; q >= 0; q--)
		if (!strcmp(V.in[q].op, "OpFunction")) {
			fn = V.in[q].res;
			break;
		}
	if (fn < 0 || b < 0)
		return 0;
	for (l = 0; l < MC.nloop; l++)
		if (MC.lin[(size_t)l * (size_t)(MC.nblk + 1) + (size_t)b])
			return 0;
	for (q = 0; q < V.nin; q++)
		if (!strcmp(V.in[q].op, "OpEntryPoint") && V.in[q].nop >= 2 &&
		    idx(V.in[q].opnd[1]) == fn)
			return 1;
	return 0;
}

/* "on line N" from the source, or the instruction's result id */
static const char *
mem_where(int k, char *b, size_t n)
{
	if (ins_line(k) > 0)
		snprintf(b, n, "on line %d", ins_line(k));
	else if (V.in[k].res >= 0)
		snprintf(b, n, "%s", V.id[V.in[k].res].tok);
	else
		snprintf(b, n, "at instruction %d", k);
	return b;
}

/* an emitter for values computed before instruction G->at: what G read
 * stands for itself (an operand there, so defined before it), the rest
 * is written anew under the names %ucas_dw<isl>_... */
static struct dwemit *
mem_emitter(const struct memg *G, struct dwg *D, struct sbuf *code,
    struct sbuf *deco, struct sbuf *decl, int isl)
{
	struct dwemit *E = xcalloc(1, sizeof *E);
	int i;

	memset(D, 0, sizeof *D);
	for (i = 0; i < MAXP; i++)
		D->pcomp[i] = -1;	/* every parameter a scalar value */
	if (G) {
		memcpy(D->psid, G->psid, sizeof D->psid);
		for (i = 0; i < G->nmemo && E->nmemo < 1024; i++) {
			if (strlen(V.id[G->id[i]].tok) >= 40) {
				E->fail = 1;
				continue;
			}
			E->memo_n[E->nmemo] = G->node[i];
			snprintf(E->memo_s[E->nmemo++], 40, "%s",
			    V.id[G->id[i]].tok);
			/* the lane parameter: the value it was read from */
			if (N.op[G->node[i]] == O_VAR)
				D->psid[N.ki[G->node[i]]] = G->id[i];
		}
	}
	E->code = code;
	E->deco = deco;
	E->decl = decl;
	E->G = D;
	E->isl = isl;
	return E;
}

/* the i32 value of form into out (an id); 0 when it cannot be written */
static int
mem_emit_int(struct dwemit *E, nid form, const char *ity, char *out,
    size_t n)
{
	if (N.op[form] == O_CAST && TY_FP(N.ty[N.a[form]])) {
		char a[40];
		snprintf(a, sizeof a, "%s", dwe(E, N.a[form]));
		if (E->fail || E->need_glsl)
			return 0;
		snprintf(out, n, "%%ucas_dw%d_%d", E->isl, ++E->serial);
		sb_printf(E->code, "%s = OpConvertFToS %s %s\n", out, ity, a);
		E->nins++;
		return 1;
	}
	snprintf(out, n, "%s", dwe(E, form));
	return !E->fail && !E->need_glsl;
}

/* "res = f32(ival) * 2^g" */
static void
mem_from_int(struct sbuf *c, const char *res, const char *fty,
    const char *ival, int g, const char *k2g, const char *tmp)
{
	if (g == 0) {
		sb_printf(c, "%s = OpConvertSToF %s %s\n", res, fty, ival);
		return;
	}
	sb_printf(c, "%s = OpConvertSToF %s %s\n", tmp, fty, ival);
	sb_printf(c, "%s = OpFMul %s %s %s\n", res, fty, tmp, k2g);
}

/* the f32 constant v as an id */
static void
mem_fconst(struct dwemit *E, double v, char *out, size_t n)
{
	snprintf(out, n, "%s", dwe_const(E, mk_cast(TY_F32, mk_const_r(v))));
}

/*
 * The float atomic adds into workgroup variable var, read with every
 * other access of it.  Each add gets intatomic's report, with the
 * workgroup's size as the count when it runs once an invocation.  When
 * every add is cheaper as an integer one on a common grid 2^g, every
 * store puts a value on that grid, the stored value plus every add stays
 * within 32 bits, and the module's whole price falls with the loads and
 * stores converted, the variable becomes i32 holding its value times
 * 2^-g: each add an OpAtomicIAdd of the integer addend, each load
 * converted back, each store converted.  Workgroup memory is the
 * module's own: no interface changes.  1 when written (rt filled).
 */
static int
mem_fatomic(int var, const int *lsize, const struct model *m,
    struct mem_out *o, struct mem_retype *rt)
{
	struct {
		int k;
		struct memg *G;
		nid v, form;
		struct intatom ia;
		int once;
		char *pre, *repl;	/* what the module gets */
	} ad[MEM_MAXACC];
	struct {
		int k, konst;
		struct memg *G;
		nid v;
		int g;
		double kmax;
	} st[MEM_MAXACC];
	int nad = 0, nst = 0, nload = 0, *chain_lev, *loads;
	int lev[MEM_MAXLEV + 1], nlev = 0, t, k, i, q, g = 0, written = 0;
	int bad_k = -1, ft, kvar = -1;
	long cnt = (long)(lsize[0] > 0 ? lsize[0] : 1) * (lsize[1] > 0 ?
	    lsize[1] : 1) * (lsize[2] > 0 ? lsize[2] : 1);
	const char *an = V.id[var].name, *bad = NULL;
	double cf = 0, ci = 0, cextra = 0, smax = 0, total = 0;
	double cconv = (double)m->c[C_CONV], cfmul = (double)m->c[C_FMUL];
	char sl[200];

	memset(ad, 0, sizeof ad);
	memset(st, 0, sizeof st);
	if (*an == '%')
		an++;
	chain_lev = xcalloc((size_t)V.nin + 1, sizeof *chain_lev);
	loads = xcalloc((size_t)V.nin + 1, sizeof *loads);
	/* the element: f32, or nested arrays of it */
	t = T(V.id[var].type)->e;
	while (nlev < MEM_MAXLEV && V.id[t].kind == K_TYPE && T(t)->k ==
	    S_ARRAY) {
		lev[nlev++] = t;
		t = T(t)->e;
	}
	lev[nlev] = ft = t;
	if (V.id[t].kind != K_TYPE || T(t)->k != S_FLOAT || T(t)->w != 32)
		bad = "its element is not an f32 (nor an array of them)";
	/* every access (all of them: each add is reported) */
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int c, d = -1;
		if (I->res == var && !strcmp(I->op, "OpVariable")) {
			kvar = k;
			if (I->nop > 2 && !bad) {
				bad = "it has an initializer";
				bad_k = k;
			}
			continue;
		}
		if (mem_debug_op(I->op))
			continue;
		for (i = 0; i < I->nop && d < 0; i++)
			if (I->opnd[i][0] == '%' && mem_base_var(idx(I->opnd[i]),
			    &c) == var)
				d = i;
		if (d < 0)
			continue;
		if ((!strcmp(I->op, "OpAccessChain") || !strcmp(I->op,
		    "OpInBoundsAccessChain") || !strcmp(I->op, "OpCopyObject")) &&
		    d == 1 && I->res >= 0) {
			int pt = idx(I->opnd[0]), j;
			for (j = 0; j <= nlev; j++)
				if (V.id[pt].kind == K_TYPE && T(pt)->k == S_PTR &&
				    T(pt)->e == lev[j])
					break;
			if (j > nlev && !bad) {
				bad = "an access chain reaches a type outside its "
				    "own";
				bad_k = k;
			} else if (j <= nlev)
				chain_lev[k] = j + 1;
		} else if (!strcmp(I->op, "OpLoad") && d == 1 && I->res >= 0) {
			if (idx(I->opnd[0]) != ft && !bad) {
				bad = "a load reads more than one element";
				bad_k = k;
			} else if (idx(I->opnd[0]) == ft) {
				loads[k] = 1;
				nload++;
			}
		} else if (!strcmp(I->op, "OpStore") && d == 0 && I->nop >= 2 &&
		    V.id[idx(I->opnd[1])].type == ft && nst < MEM_MAXACC)
			st[nst++].k = k;
		else if (!strcmp(I->op, "OpAtomicFAddEXT") && d == 1 &&
		    I->nop >= 5 && idx(I->opnd[0]) == ft && nad < MEM_MAXACC)
			ad[nad++].k = k;
		else if (!bad) {
			bad = !strcmp(I->op, "OpStore") ? "a store writes more "
			    "than one element" : "an access of a kind this "
			    "rewrite does not convert";
			if (nst == MEM_MAXACC || nad == MEM_MAXACC)
				bad = "more than 8 stores or adds";
			bad_k = k;
		}
	}
	/* each add: intatomic's report */
	for (q = 0; q < nad; q++) {
		const struct sins *I = &V.in[ad[q].k];
		ad[q].G = xcalloc(1, sizeof *ad[q].G);
		ad[q].G->lsize = lsize[0];
		ad[q].G->at = ad[q].k;
		ad[q].G->f.name = sym_cstr("_fadd");
		ad[q].G->f.anon = 1;
		ad[q].v = memg_build(ad[q].G, idx(I->opnd[4]), 0);
		ad[q].once = mem_once(ad[q].k);
		mem_hdr(o, m);
		printf("  %s\n", I->text);
		src_line(ins_line(ad[q].k), sl, sizeof sl);
		fputs(sl, stdout);
		if (ad[q].G->fail || !TY_FP(N.ty[ad[q].v])) {
			printf("    its addend is not read as an expression (more "
			    "than 8 inputs, or a type outside f32 and the "
			    "integers)\n");
			continue;
		}
		ad[q].G->f.body = ad[q].v;
		if (ad[q].once)
			printf("    (count: the workgroup's %ld invocations, once "
			    "each)\n", cnt);
		else
			printf("    (count: not known: the add is in a loop, or "
			    "outside the entry point's function)\n");
		memk_intatomic(&ad[q].G->f, ad[q].once ? cnt : 0, 0, 0, m,
		    &ad[q].ia);
	}
	if (nad == 0)
		goto out;
	/* the variable as a whole */
	if (kvar < 0 && !bad)
		bad = "its declaration is not in the module";
	if (!bad)
		for (q = 0; q < nad; q++) {
			if (!ad[q].ia.ok || !ad[q].ia.exact) {
				bad = "an add is not rewritten (above)";
				bad_k = ad[q].k;
				break;
			}
			if (!ad[q].once) {
				bad = "how many adds reach one cell is not known";
				bad_k = ad[q].k;
				break;
			}
			if (q == 0 || ad[q].ia.g < g)
				g = ad[q].ia.g;
		}
	/* each store: a value on a grid */
	for (q = 0; q < nst && !bad; q++) {
		const struct sins *I = &V.in[st[q].k];
		struct sbuf why = {0};
		int sg;
		double sk;
		st[q].G = xcalloc(1, sizeof *st[q].G);
		st[q].G->lsize = lsize[0];
		st[q].G->at = st[q].k;
		st[q].v = memg_build(st[q].G, idx(I->opnd[1]), 0);
		st[q].konst = N.op[st[q].v] == O_CONST;
		if (st[q].G->fail || !fix_grid(&st[q].G->f, st[q].v, &sg, &sk,
		    &why)) {
			bad = "a store puts a value there that is not on a binary "
			    "grid";
			bad_k = st[q].k;
		} else if (st[q].konst && N.kr[st[q].v] == 0 &&
		    signbit(N.kr[st[q].v])) {
			bad = "a store puts -0.0 there, which an integer cannot "
			    "hold";
			bad_k = st[q].k;
		} else {
			st[q].g = sg;
			st[q].kmax = sk;
			if (sk > 0 && sg < g)
				g = sg;
		}
		free(why.p);
	}
	if (!bad) {
		/* every add's form on the common grid, and the bound */
		for (q = 0; q < nad; q++) {
			ad[q].form = g == ad[q].ia.g ? ad[q].ia.form :
			    memk_fixform(ad[q].v, g);
			cf += ad[q].ia.cost_f;
			ci += (double)model_cost(m, C_ATOM_SH) +
			    (double)fuse_expr_cost(m, ad[q].form);
			total += (double)cnt * ldexp(ad[q].ia.kmax, ad[q].ia.g - g);
			if (mem_used(V.in[ad[q].k].res))
				cextra += cconv + (g ? cfmul : 0);
		}
		for (q = 0; q < nst; q++) {
			double s = ldexp(st[q].kmax, st[q].g - g);
			if (s > smax)
				smax = s;
			if (!st[q].konst)
				cextra += cconv + (g ? cfmul : 0);
		}
		cextra += nload * (cconv + (g ? cfmul : 0));
		total += smax;
		if (total > 2147483647.0)
			bad = "its value can pass 2^31 - 1 on the grid";
		else if (ci + cextra >= cf)
			bad = "not cheaper with its loads and stores converted";
	}
	if (bad) {
		printf("    %s is left as it is: %s", an, bad);
		if (!strcmp(bad, "its value can pass 2^31 - 1 on the grid"))
			printf(", %.0f 2^%d", total, g);
		else if (!strncmp(bad, "not cheaper", 11))
			printf(": %g against %g", ci + cextra, cf);
		else if (!strncmp(bad, "an access of", 12))
			printf(", %s", V.in[bad_k].op);
		if (bad_k >= 0) {
			char where[48];
			printf(" (%s)", mem_where(bad_k, where, sizeof where));
		}
		printf("\n");
		goto out;
	}
	/* written: the adds first, the only part that can fail; the
	 * declarations wait for the bank pass (rt) */
	{
		struct sbuf deco = {0}, scratch = {0};
		struct dwg *D = xcalloc(1, sizeof *D);
		struct dwemit *E0 = mem_emitter(NULL, D, &scratch, &deco, o->decl,
		    0);
		char k2g[64], k2mg[64], nm[48], tmp[48], ival[48], where[48];
		const char *fty = V.id[ft].tok;
		int isl = 1000, serial = 0, ok = 1;

		rt->var = var;
		rt->kvar = kvar;
		rt->nlev = nlev;
		memcpy(rt->lev, lev, sizeof rt->lev);
		rt->stem = mem_free_stem("ucas_ia", 1);
		snprintf(rt->ity, sizeof rt->ity, "%s", dwe_type(E0,
		    TY(K_SINT, 32)));
		k2g[0] = k2mg[0] = 0;
		if (g != 0 && nload > 0)
			mem_fconst(E0, ldexp(1.0, g), k2g, sizeof k2g);
		for (q = 0; q < nad && g != 0 && !k2g[0]; q++)
			if (mem_used(V.in[ad[q].k].res))
				mem_fconst(E0, ldexp(1.0, g), k2g, sizeof k2g);
		for (q = 0; q < nst && g != 0 && !k2mg[0]; q++)
			if (!st[q].konst)
				mem_fconst(E0, ldexp(1.0, -g), k2mg, sizeof k2mg);
#define IA_NAME(b) snprintf(b, sizeof b, "%%ucas_ia%d_%d", rt->stem, \
	++serial)
		for (q = 0; q < nad && ok; q++) {
			const struct sins *I = &V.in[ad[q].k];
			struct dwg *Dq = xcalloc(1, sizeof *Dq);
			struct sbuf code = {0}, c = {0};
			struct dwemit *E;
			isl = mem_free_stem("ucas_dw", isl);
			E = mem_emitter(ad[q].G, Dq, &code, &deco, o->decl, isl++);
			ok = mem_emit_int(E, ad[q].form, rt->ity, ival, sizeof ival);
			if (ok) {
				int used = mem_used(I->res);
				if (used)
					IA_NAME(nm);
				else
					snprintf(nm, sizeof nm, "%s", V.id[I->res].tok);
				sb_printf(&c, "%s = OpAtomicIAdd %s %s %s %s %s\n", nm,
				    rt->ity, I->opnd[1], I->opnd[2], I->opnd[3], ival);
				if (used) {
					IA_NAME(tmp);
					mem_from_int(&c, V.id[I->res].tok, fty, nm, g, k2g,
					    tmp);
				}
			}
			ad[q].pre = code.p;
			ad[q].repl = c.p;
			free(E);
			free(Dq);
		}
		if (!ok) {
			for (q = 0; q < nad; q++) {
				free(ad[q].pre);
				free(ad[q].repl);
			}
			printf("    %s is left as it is: an integer addend could not "
			    "be written as SPIR-V\n", an);
		}
		for (k = 0; k < V.nin && ok; k++) {
			const struct sins *I = &V.in[k];
			struct sbuf c = {0};
			if (chain_lev[k]) {
				snprintf(o->newty[k], 48, "%%ucas_ia%d_p%d", rt->stem,
				    chain_lev[k] - 1);
				continue;
			}
			if (!loads[k])
				continue;
			IA_NAME(nm);
			IA_NAME(tmp);
			sb_printf(&c, "%s = OpLoad %s", nm, rt->ity);
			for (i = 1; i < I->nop; i++)
				sb_printf(&c, " %s", I->opnd[i]);
			sb_putc(&c, '\n');
			mem_from_int(&c, V.id[I->res].tok, fty, nm, g, k2g, tmp);
			o->repl[k] = c.p;
		}
		for (q = 0; q < nad && ok; q++) {
			o->pre[ad[q].k] = ad[q].pre;
			o->repl[ad[q].k] = ad[q].repl;
		}
		for (q = 0; q < nst && ok; q++) {
			const struct sins *I = &V.in[st[q].k];
			struct sbuf c = {0};
			char val[64];
			if (st[q].konst)
				snprintf(val, sizeof val, "%s", dwe_const(E0,
				    mk_const_i(TY(K_SINT, 32), (int64_t)ldexp(
				    N.kr[st[q].v], -g))));
			else {
				const char *x = I->opnd[1];
				if (g != 0) {
					IA_NAME(tmp);
					sb_printf(&c, "%s = OpFMul %s %s %s\n", tmp, fty, x,
					    k2mg);
					x = tmp;
				}
				IA_NAME(nm);
				sb_printf(&c, "%s = OpConvertFToS %s %s\n", nm, rt->ity, x);
				snprintf(val, sizeof val, "%s", nm);
			}
			sb_printf(&c, "OpStore %s %s", I->opnd[0], val);
			for (i = 2; i < I->nop; i++)
				sb_printf(&c, " %s", I->opnd[i]);
			sb_putc(&c, '\n');
			o->repl[st[q].k] = c.p;
		}
#undef IA_NAME
		free(deco.p);
		free(scratch.p);
		free(E0);
		free(D);
		if (!ok)
			goto out;
		if (g)
			printf("    %s becomes i32 holding its value times %g, "
			    "written into the module (workgroup memory is the "
			    "module's own):\n", an, ldexp(1.0, -g));
		else
			printf("    %s becomes i32 holding its value, written into "
			    "the module (workgroup memory is the module's own):\n",
			    an);
		for (q = 0; q < nad; q++)
			printf("      the add %s: OpAtomicIAdd of %s\n", mem_where(
			    ad[q].k, where, sizeof where), shared_str(ad[q].form,
			    120));
		if (nload && g)
			printf("      %d load%s converted back: f32(cell) * %g (%g "
			    "each)\n", nload, nload == 1 ? "" : "s", ldexp(1.0, g),
			    cconv + cfmul);
		else if (nload)
			printf("      %d load%s converted back: f32(cell) (%g each)\n",
			    nload, nload == 1 ? "" : "s", cconv);
		for (q = 0; q < nst; q++)
			if (st[q].konst && g)
				printf("      the store %s: the i32 constant %.0f (%g "
				    "times %g)\n", mem_where(st[q].k, where, sizeof where),
				    ldexp(N.kr[st[q].v], -g), N.kr[st[q].v], ldexp(1.0,
				    -g));
			else if (st[q].konst)
				printf("      the store %s: the i32 constant %.0f\n",
				    mem_where(st[q].k, where, sizeof where), N.kr[st[q].v]);
			else if (g)
				printf("      the store %s: i32(v * %g) (%g), v = %s on "
				    "the grid 2^%d\n", mem_where(st[q].k, where, sizeof
				    where), ldexp(1.0, -g), cconv + cfmul,
				    shared_str(st[q].v, 80), st[q].g);
			else
				printf("      the store %s: i32(v) (%g), v = %s on the "
				    "grid 2^%d\n", mem_where(st[q].k, where, sizeof
				    where), cconv, shared_str(st[q].v, 80), st[q].g);
		if (g)
			printf("      its value stays within %.0f 2^%d", total, g);
		else
			printf("      its value stays within %.0f", total);
		printf(" (the stores' largest, %.0f, and %ld invocations' adds): "
		    "%s\n", smax, cnt, total <= 16777216.0 ? "within 24 bits, so the f32 "
		    "version held the same values, bit for bit" : "past 24 bits "
		    "the f32 sums rounded in the order the invocations arrived; "
		    "the i32 sum is exact and a read rounds it once");
		for (q = 0; q < nst; q++)
			if (!st[q].konst) {
				printf("      a stored -0.0 reads back as 0.0 (equal "
				    "under every comparison; the zero's sign is not "
				    "kept)\n");
				break;
			}
		printf("      priced: the adds and their addends %g, the "
		    "conversions %g: %g against %g (%+.1f%%)\n", ci, cextra, ci +
		    cextra, cf, 100 * (ci + cextra - cf) / cf);
		o->repl[kvar] = xcalloc(1, 1);
		/* the float addends nothing reads any more */
		for (q = 0; q < nad; q++)
			mem_dce(idx(V.in[ad[q].k].opnd[4]), o->pre, o->repl, 0);
		written = 1;
	}
out:
	for (q = 0; q < nad; q++)
		free(ad[q].G);
	for (q = 0; q < nst; q++)
		free(st[q].G);
	free(chain_lev);
	free(loads);
	return written;
}

/* the access chain whose base is var, under pointer ptr; -1 none */
static int
mem_root_chain(int ptr, int var)
{
	int guard = 0, k;

	while (guard++ < 16 && (k = SP.def[ptr]) >= 0) {
		if (!strcmp(V.in[k].op, "OpAccessChain") || !strcmp(V.in[k].op,
		    "OpInBoundsAccessChain")) {
			if (idx(V.in[k].opnd[1]) == var)
				return V.in[k].nop >= 3 ? k : -1;
		} else if (strcmp(V.in[k].op, "OpCopyObject") != 0)
			return -1;
		ptr = idx(V.in[k].opnd[1]);
	}
	return -1;
}

/*
 * A float atomic add into a storage buffer: intatomic's report with the
 * adds into one cell not known (the dispatch is the host's) and, when the
 * integer add is cheaper and exact, a proposal: the buffer is read outside
 * the module (other shaders, the host), so its type is an interface.
 */
static void
mem_fatomic_buf(int k, int var, const int *lsize, const struct model *m,
    struct mem_out *o)
{
	const struct sins *I = &V.in[k];
	struct memg *G = xcalloc(1, sizeof *G);
	struct intatom ia;
	char sl[200], what[96];
	const char *bn = V.id[var].name;
	int nload = 0, nstore = 0, nadd = 0, ck, q, g;
	nid v;

	if (*bn == '%')
		bn++;
	G->lsize = lsize[0];
	G->at = k;
	G->f.name = sym_cstr("_fadd");
	G->f.anon = 1;
	v = memg_build(G, idx(I->opnd[4]), 0);
	mem_hdr(o, m);
	printf("  %s\n", I->text);
	src_line(ins_line(k), sl, sizeof sl);
	fputs(sl, stdout);
	if (G->fail || !TY_FP(N.ty[v])) {
		printf("    its addend is not read as an expression (more than 8 "
		    "inputs, or a type outside f32 and the integers)\n");
		free(G);
		return;
	}
	G->f.body = v;
	printf("    (count: not known here: every invocation of the dispatch "
	    "may add into one cell of the buffer)\n");
	memk_intatomic(&G->f, 0, 0, 1, m, &ia);
	free(G);
	if (!ia.ok || !ia.exact)
		return;
	g = ia.g;
	/* the member, and this module's other accesses of it */
	snprintf(what, sizeof what, "the buffer %s", bn);
	ck = mem_root_chain(idx(I->opnd[1]), var);
	if (ck >= 0) {
		int stt = T(V.id[var].type)->e, mi = idx(V.in[ck].opnd[2]);
		if (V.id[stt].kind == K_TYPE && T(stt)->k == S_STRUCT &&
		    V.id[mi].kind == K_CONST)
			snprintf(what, sizeof what, "%s in the buffer %s",
			    mem_member(stt, (int)V.id[mi].c[0]), bn);
		for (q = 0; q < V.nin; q++) {
			const struct sins *J = &V.in[q];
			int pi = !strcmp(J->op, "OpLoad") ? 1 : !strcmp(J->op,
			    "OpStore") ? 0 : !strcmp(J->op, "OpAtomicFAddEXT") ? 1 :
			    -1, c2;
			if (pi < 0 || J->nop <= pi || J->opnd[pi][0] != '%' ||
			    (c2 = mem_root_chain(idx(J->opnd[pi]), var)) < 0 ||
			    strcmp(V.in[c2].opnd[2], V.in[ck].opnd[2]) != 0)
				continue;
			if (J->op[2] == 'L')
				nload++;
			else if (J->op[2] == 'S')
				nstore++;
			else
				nadd++;
		}
	}
	printf("    proposal (an interface change, not applied): %s becomes "
	    "i32, holding its value times %g:\n", what, ldexp(1.0, -g));
	printf("      here the add is OpAtomicIAdd of %s", shared_str(ia.form,
	    120));
	if (nadd > 1)
		printf(" (this module adds into it %d times)", nadd);
	if (nload)
		printf(", %d load%s f32(cell) * %g", nload, nload == 1 ? "" : "s",
		    ldexp(1.0, g));
	if (nstore)
		printf(", %d store%s i32(v * %g) of a v on the grid 2^%d", nstore,
		    nstore == 1 ? "" : "s", ldexp(1.0, -g), g);
	printf("\n");
	printf("      every other shader that reads or writes it, and the "
	    "host's upload and readback, convert the same way;\n      the "
	    "dispatch must send at most %.0f adds into one cell\n", ia.fits);
}

/* the types of a variable that became i32, and the variable itself,
 * into decl: the arrays from the inside out (the outermost with the
 * length the bank pass gave it), a pointer to each level */
static void
mem_retype_decl(const struct mem_retype *rt, struct sbuf *decl,
    const int *newidx, char (*newname)[40])
{
	int j, q;

	for (j = rt->nlev - 1; j >= 0; j--) {
		const char *len = NULL;
		for (q = 0; q < V.nin && len == NULL; q++)
			if (V.in[q].res == rt->lev[j] && !strcmp(V.in[q].op,
			    "OpTypeArray") && V.in[q].nop >= 2)
				len = newidx[q] == 2 ? newname[q] : V.in[q].opnd[1];
		if (j + 1 == rt->nlev)
			sb_printf(decl, "%%ucas_ia%d_t%d = OpTypeArray %s %s\n",
			    rt->stem, j, rt->ity, len ? len : "%ucas_nolength");
		else
			sb_printf(decl, "%%ucas_ia%d_t%d = OpTypeArray %%ucas_ia%d_t%d "
			    "%s\n", rt->stem, j, rt->stem, j + 1, len ? len :
			    "%ucas_nolength");
	}
	for (j = 0; j <= rt->nlev; j++)
		if (j == rt->nlev)
			sb_printf(decl, "%%ucas_ia%d_p%d = OpTypePointer Workgroup "
			    "%s\n", rt->stem, j, rt->ity);
		else
			sb_printf(decl, "%%ucas_ia%d_p%d = OpTypePointer Workgroup "
			    "%%ucas_ia%d_t%d\n", rt->stem, j, rt->stem, j);
	sb_printf(decl, "%s = OpVariable %%ucas_ia%d_p0 Workgroup\n",
	    V.id[rt->var].tok, rt->stem);
}

/* no f32 atomic add left: the module stops asking for the capability
 * (and for its extension, when f64 adds do not need it either) */
static void
mem_drop_fadd_caps(char **repl)
{
	int k, f32 = 0, f64 = 0;

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strcmp(I->op, "OpAtomicFAddEXT") && repl[k] == NULL &&
		    I->nop >= 1 && T(idx(I->opnd[0]))->w == 32)
			f32 = 1;
		if (!strcmp(I->op, "OpCapability") && I->nop >= 1 &&
		    !strcmp(I->opnd[0], "AtomicFloat64AddEXT"))
			f64 = 1;
	}
	if (f32)
		return;
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strcmp(I->op, "OpCapability") && I->nop >= 1 &&
		    !strcmp(I->opnd[0], "AtomicFloat32AddEXT")) {
			repl[k] = xcalloc(1, 1);
			printf("  the module no longer declares AtomicFloat32AddEXT: "
			    "it runs where the device has no float atomic add\n");
		}
		if (!f64 && !strcmp(I->op, "OpExtension") && I->nop >= 1 &&
		    strstr(I->opnd[0], "SPV_EXT_shader_atomic_float_add"))
			repl[k] = xcalloc(1, 1);
	}
}

/* ---- the pass ---- */

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
	char **repl, (*newty)[48];
	struct mem_retype rt[MEM_MAXARR];
	int nrt = 0, done_var[64], ndone = 0;
	struct mem_out mo;

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
	repl = xcalloc((size_t)V.nin + 1, sizeof *repl);
	newty = xcalloc((size_t)V.nin + 1, sizeof *newty);
	mo.pre = pre;
	mo.repl = repl;
	mo.newty = newty;
	mo.decl = &decl;
	mo.hdr = &hdr;
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
					if (!mem_uni_at(idx(V.in[chain].opnd[i]), k))
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
					if (!mem_uni_at(idx(V.in[chain].opnd[i]), k))
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
		if (var >= 0 && !strcmp(I->op, "OpAtomicFAddEXT") && I->nop >= 5 &&
		    (!strcmp(mem_sc(var), "StorageBuffer") || (!strcmp(mem_sc(var),
		    "Uniform") && mem_bufferblock(var)))) {
			nfatom++;
			mem_fatomic_buf(k, var, lsize, m, &mo);
			continue;
		}
		if (var < 0 || strcmp(mem_sc(var), "Workgroup") != 0)
			continue;
		if (!strcmp(I->op, "OpAtomicFAddEXT") && I->nop >= 5) {
			for (a = 0; a < ndone; a++)
				if (done_var[a] == var)
					break;
			if (a == ndone && ndone < 64) {
				done_var[ndone++] = var;
				nfatom++;
				if (nrt < MEM_MAXARR && mem_fatomic(var, lsize, m, &mo,
				    &rt[nrt])) {
					nrt++;
					*changed = 1;
				} else if (nrt == MEM_MAXARR)
					printf("    %s is left as it is: 8 workgroup "
					    "variables were rewritten already\n", V.id[var].name);
			} else if (a == ndone) {
				MEM_HDR();
				printf("  %s\n    not analysed: float atomic adds into "
				    "more than 64 workgroup variables\n", I->text);
			}
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
				G[q]->at = arr[i].chain[q];
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
	/* the variables that became i32: their types, after the bank pass
	 * (a padded row grows the outermost array) */
	for (i = 0; i < nrt; i++)
		mem_retype_decl(&rt[i], &decl, newidx, newname);
	if (nrt > 0)
		mem_drop_fadd_caps(repl);
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
			if (repl[k])
				sb_puts(&out, repl[k]);
			else if ((newidx[k] || newty[k][0]) && I->res >= 0) {
				/* the access chain's index or result type, or the
				 * array's length */
				sb_printf(&out, "%s = %s", V.id[I->res].tok, I->op);
				for (i = 0; i < I->nop; i++)
					sb_printf(&out, " %s", i == 0 && newty[k][0] ?
					    newty[k] : (newidx[k] == 1 && i == 2) ||
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
	for (k = 0; k < V.nin; k++) {
		free(pre[k]);
		free(repl[k]);
	}
	free(pre);
	free(repl);
	free(newty);
	free(newidx);
	free(newname);
	free(decl.p);
	free(mem_uni);
	mem_uni = NULL;
	mem_cfg_free();
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
