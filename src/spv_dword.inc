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
 * A loop's f64 sum (a phi of its header that only adds) is carried as a
 * pair of f32 phis when the trip count is a constant, the bound composed
 * over the passes (dword_accum).
 *
 * The same pass reports the float round trips the model says its driver
 * folds (fpfold.c), with an input where the folded program differs, and
 * with `foldproof` writes their rounding as integer instructions.
 */
#include "dword.h"

#include <unistd.h>

static int dw_foldproof;	/* apply the fold-proof forms */
static struct dwreq dw_req;	/* the request's tol or atol (DW_DEFAULT: none) */
static int dw_nfas;		/* float assumptions float_pass left in pend_fas */

void
spirv_dword_opts(int foldproof)
{
	dw_foldproof = foldproof;
}

void
spirv_dword_req(int kind, double v)
{
	dw_req.kind = kind;
	dw_req.v = v;
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

/*
 * The f64 values of a group of results as expressions over one set of
 * inputs.  An input is a SPIR-V value or one component of it (pcomp: -1
 * the value, 0..3 a vector's component, 16 + 4 * column + row a
 * matrix's).  dw_leaf marks the f64 values that stay as they are (another
 * result that keeps its f64 reads them): they are inputs too.
 */
struct dwg {
	struct func f;
	int psid[MAXP];		/* the SPIR-V value of each parameter */
	int pcomp[MAXP];
	int id[512], comp[512];
	nid node[512];
	int nmemo;
	int ins[256];		/* the f64 instructions under the roots */
	int nins;
	int toomany;
	const char *pname[MAXP];	/* a value the rewrite makes: its name */
};

static char *dw_leaf;
static char *dw_keep;	/* per instruction: under a result, and stays for another */
static int dw_recompute;	/* a value that stays is computed again in f32 */
static int dw_rdepth;

static int dwg_leaf_range(int sid, int comp, double *lo, double *hi);

static int
vec_float_w(int t, int *n)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_VEC)
		return 0;
	*n = T(t)->n;
	return scalar_float_w(T(t)->e);
}

static int
vec_int_w(int t, int *n, int *sign)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_VEC)
		return 0;
	*n = T(t)->n;
	return scalar_int_w(T(t)->e, sign);
}

/* a matrix of f64 columns: its columns and rows */
static int
mat_f64(int t, int *cols, int *rows)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_MAT)
		return 0;
	*cols = T(t)->n;
	return vec_float_w(T(t)->e, rows) == 64;
}

/* GLSL.std.450's float minimum and maximum (the N forms differ only on
 * a NaN, which the forms do not carry anyway): O_MIN, O_MAX, or 0 */
static int
dw_minmax(const char *ext)
{
	if (!strcmp(ext, "FMin") || !strcmp(ext, "NMin"))
		return O_MIN;
	if (!strcmp(ext, "FMax") || !strcmp(ext, "NMax"))
		return O_MAX;
	return 0;
}

static void
dwg_ins(struct dwg *G, int k)
{
	int i;

	for (i = 0; i < G->nins; i++)
		if (G->ins[i] == k)
			return;
	if (G->nins < 256)
		G->ins[G->nins++] = k;
	else
		G->toomany = 1;
}

static int
dwg_param(struct dwg *G, int sid, int comp, ty_t ty)
{
	char nb[48], nm[64];
	const char *n;
	struct param *p;
	int j;

	for (j = 0; j < G->f.np; j++)
		if (G->psid[j] == sid && G->pcomp[j] == comp)
			return j;
	if (G->f.np == MAXP) {
		G->toomany = 1;
		return 0;
	}
	j = G->f.np++;
	G->psid[j] = sid;
	G->pcomp[j] = comp;
	n = load_name(sid, nb, sizeof nb);
	if (n == NULL) {
		n = V.id[sid].name;
		if (*n == '%')
			n++;
	}
	if (!isalpha((unsigned char)*n) && *n != '_')
		snprintf(nm, sizeof nm, "v%.40s", n);
	else
		snprintf(nm, sizeof nm, "%.40s", n);
	if (comp >= 16)
		snprintf(nm + strlen(nm), sizeof nm - strlen(nm), "_%d%d",
		    (comp - 16) / 4, (comp - 16) % 4);
	else if (comp >= 0)
		snprintf(nm + strlen(nm), sizeof nm - strlen(nm), "_%c",
		    "xyzw"[comp & 3]);
	p = &G->f.p[j];
	memset(p, 0, sizeof *p);
	p->name = sym_cstr(nm);
	p->ty = ty;
	if (ty == TY_F32) {
		int q;
		/* a range given for a vector holds for each component */
		for (q = 0; q < dw_nfas; q++)
			if (name_is(n, pend_fas[q].name) || !strcmp(n,
			    pend_fas[q].name)) {
				p->ranged = 1;
				p->rlo = (double)pend_fas[q].lo;
				p->rhi = (double)pend_fas[q].hi;
				if (p->rlo == 0)
					p->rlo = 0;	/* -0: the same interval */
			}
	} else if (TY_ISINT(ty) && comp < 0) {
		int64_t lo, hi;
		if (assumed(sid, &lo, &hi)) {
			p->ranged = 1;
			p->lo = lo;
			p->hi = hi;
			p->rlo = (double)lo;
			p->rhi = (double)hi;
		}
	} else if (ty == TY_F64 && dw_leaf[sid]) {
		double lo, hi;
		if (dwg_leaf_range(sid, comp, &lo, &hi)) {
			p = &G->f.p[j];
			p->ranged = 1;
			p->rlo = lo;
			p->rhi = hi;
		}
	}
	return j;
}

static nid
dwg_var(struct dwg *G, int sid, int comp, ty_t ty)
{
	int j = dwg_param(G, sid, comp, ty);

	return mk_var((uint32_t)j, G->f.p[j].name, ty);
}

static nid dwg_build(struct dwg *G, int id, int depth);
static nid dwg_comp(struct dwg *G, int id, int ci, int depth);

static nid
dwg_memo(struct dwg *G, int id, int comp, nid r)
{
	if (G->nmemo < 512) {
		G->id[G->nmemo] = id;
		G->comp[G->nmemo] = comp;
		G->node[G->nmemo++] = r;
	}
	return r;
}

static nid dwg_mcomp(struct dwg *G, int id, int col, int row, int depth);

/*
 * The range of an f64 value that stays, from the ranges of what it is
 * computed from: interval arithmetic in f64 itself, whose rounding is
 * monotone, so the computed value lies between the computed ends.
 */
static int
dwg_leaf_range(int sid, int comp, double *lo, double *hi)
{
	struct dwg *T;
	nid e;
	int ok;

	if (dw_rdepth > 6)
		return 0;
	T = xcalloc(1, sizeof *T);
	T->f.name = sym_cstr("_leaf");
	T->f.anon = 1;
	dw_leaf[sid] = 0;
	dw_rdepth++;
	e = comp < 0 ? dwg_build(T, sid, 0) : comp >= 16 ? dwg_mcomp(T, sid,
	    (comp - 16) / 4, (comp - 16) % 4, 0) : dwg_comp(T, sid, comp, 0);
	dw_rdepth--;
	dw_leaf[sid] = 1;
	ok = !T->toomany && N.op[e] != O_VAR && dword_interval(&T->f, e, lo, hi);
	free(T);
	return ok;
}

/* the sum of the products of two vectors' components */
static nid
dwg_dot(struct dwg *G, int a, int b, int n, int depth)
{
	nid r = 0, p;
	int j;

	for (j = 0; j < n; j++) {
		p = mk_bin(O_MUL, dwg_comp(G, a, j, depth), dwg_comp(G, b, j,
		    depth));
		r = j ? mk_bin(O_ADD, r, p) : p;
	}
	return r;
}

/* one entry of an f64 matrix */
static nid
dwg_mcomp(struct dwg *G, int id, int col, int row, int depth)
{
	int k = SP.def[id], i, comp = 16 + 4 * col + row, cols, rows;
	const struct sins *I;
	nid r = 0;

	for (i = 0; i < G->nmemo; i++)
		if (G->id[i] == id && G->comp[i] == comp)
			return G->node[i];
	if (k < 0 || depth > 48 || dw_leaf[id] || !mat_f64(V.id[id].type,
	    &cols, &rows) || col >= cols || row >= rows)
		return dwg_var(G, id, comp, TY_F64);
	I = &V.in[k];
	if (!strcmp(I->op, "OpCompositeConstruct") && I->nop == 1 + cols)
		r = dwg_comp(G, idx(I->opnd[1 + col]), row, depth + 1);
	else if (!strcmp(I->op, "OpTranspose") && I->nop >= 2)
		r = dwg_mcomp(G, idx(I->opnd[1]), row, col, depth + 1);
	else if (!strcmp(I->op, "OpMatrixTimesScalar") && I->nop >= 3)
		r = mk_bin(O_MUL, dwg_mcomp(G, idx(I->opnd[1]), col, row,
		    depth + 1), dwg_build(G, idx(I->opnd[2]), depth + 1));
	else if (!strcmp(I->op, "OpMatrixTimesMatrix") && I->nop >= 3) {
		int a = idx(I->opnd[1]), b = idx(I->opnd[2]), ac, ar, j;
		if (mat_f64(V.id[a].type, &ac, &ar)) {
			nid q;
			for (j = 0; j < ac; j++) {
				q = mk_bin(O_MUL, dwg_mcomp(G, a, j, row, depth + 1),
				    dwg_mcomp(G, b, col, j, depth + 1));
				r = j ? mk_bin(O_ADD, r, q) : q;
			}
		}
	}
	if (r == 0)
		return dwg_var(G, id, comp, TY_F64);
	dwg_ins(G, k);
	return dwg_memo(G, id, comp, r);
}

/* one component of an f64 vector */
static nid
dwg_comp(struct dwg *G, int id, int ci, int depth)
{
	int k = SP.def[id], i, t = V.id[id].type, n = 0, sg, m;
	const struct sins *I;
	nid r = 0, a, b, c;

	for (i = 0; i < G->nmemo; i++)
		if (G->id[i] == id && G->comp[i] == ci)
			return G->node[i];
	if (vec_float_w(t, &n) != 64 || ci >= n)
		return dwg_var(G, id, ci, TY_F64);
	if (V.id[id].kind == K_CONST && !V.id[id].spec && ci < V.id[id].nc)
		return dwg_memo(G, id, ci, mk_cast(TY_F64, mk_const_r(getf(T(t)->e,
		    V.id[id].c[ci]))));
	if (k < 0 || depth > 48 || dw_leaf[id])
		return dwg_var(G, id, ci, TY_F64);
	I = &V.in[k];
	if (!strcmp(I->op, "OpFConvert") && I->nop >= 2) {
		int s = idx(I->opnd[1]);
		if (vec_float_w(V.id[s].type, &m) == 32 && m == n)
			r = mk_cast(TY_F64, dwg_var(G, s, ci, TY_F32));
	} else if ((!strcmp(I->op, "OpConvertSToF") || !strcmp(I->op,
	    "OpConvertUToF")) && I->nop >= 2) {
		int s = idx(I->opnd[1]), w = vec_int_w(V.id[s].type, &m, &sg);
		if (w > 0 && w <= 32 && m == n)
			r = mk_cast(TY_F64, dwg_var(G, s, ci, TY(I->op[9] == 'U' ?
			    K_UINT : K_SINT, w)));
	} else if (!strcmp(I->op, "OpFNegate") && I->nop >= 2) {
		r = mk_un(O_NEG, dwg_comp(G, idx(I->opnd[1]), ci, depth + 1));
	} else if ((!strcmp(I->op, "OpFAdd") || !strcmp(I->op, "OpFSub") ||
	    !strcmp(I->op, "OpFMul") || !strcmp(I->op, "OpFDiv")) &&
	    I->nop >= 3) {
		a = dwg_comp(G, idx(I->opnd[1]), ci, depth + 1);
		b = dwg_comp(G, idx(I->opnd[2]), ci, depth + 1);
		r = mk_bin(I->op[3] == 'A' ? O_ADD : I->op[3] == 'S' ? O_SUB :
		    I->op[3] == 'M' ? O_MUL : O_DIV, a, b);
	} else if (!strcmp(I->op, "OpVectorTimesScalar") && I->nop >= 3) {
		a = dwg_comp(G, idx(I->opnd[1]), ci, depth + 1);
		b = dwg_build(G, idx(I->opnd[2]), depth + 1);
		r = mk_bin(O_MUL, a, b);
	} else if (!strcmp(I->op, "OpCompositeConstruct")) {
		int at = 0, o;
		for (o = 1; o < I->nop && r == 0; o++) {
			int s = idx(I->opnd[o]);
			if (vec_float_w(V.id[s].type, &m) == 64) {
				if (ci < at + m)
					r = dwg_comp(G, s, ci - at, depth + 1);
				at += m;
			} else if (scalar_float_w(V.id[s].type) == 64) {
				if (ci == at)
					r = dwg_build(G, s, depth + 1);
				at++;
			} else
				break;
		}
	} else if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 3) {
		/* a matrix's column */
		int mt = idx(I->opnd[1]), col = atoi(I->opnd[2]), mc, mr;
		if (mat_f64(V.id[mt].type, &mc, &mr) && col >= 0 && col < mc &&
		    ci < mr)
			r = dwg_mcomp(G, mt, col, ci, depth + 1);
	} else if (!strcmp(I->op, "OpVectorShuffle") && I->nop >= 4 + ci) {
		int v1 = idx(I->opnd[1]), v2 = idx(I->opnd[2]);
		long lit = strtol(I->opnd[3 + ci], NULL, 10);
		if (vec_float_w(V.id[v1].type, &m) == 64 && lit >= 0 && lit < 8)
			r = lit < m ? dwg_comp(G, v1, (int)lit, depth + 1) :
			    dwg_comp(G, v2, (int)lit - m, depth + 1);
	} else if (!strcmp(I->op, "OpMatrixTimesVector") && I->nop >= 3) {
		int mt = idx(I->opnd[1]), v = idx(I->opnd[2]), mc, mr, j;
		if (mat_f64(V.id[mt].type, &mc, &mr) && ci < mr)
			for (j = 0; j < mc; j++) {
				a = mk_bin(O_MUL, dwg_mcomp(G, mt, j, ci, depth + 1),
				    dwg_comp(G, v, j, depth + 1));
				r = j ? mk_bin(O_ADD, r, a) : a;
			}
	} else if (!strcmp(I->op, "OpVectorTimesMatrix") && I->nop >= 3) {
		int v = idx(I->opnd[1]), mt = idx(I->opnd[2]), mc, mr, j;
		if (mat_f64(V.id[mt].type, &mc, &mr) && ci < mc)
			for (j = 0; j < mr; j++) {
				a = mk_bin(O_MUL, dwg_comp(G, v, j, depth + 1),
				    dwg_mcomp(G, mt, ci, j, depth + 1));
				r = j ? mk_bin(O_ADD, r, a) : a;
			}
	} else if (!strcmp(I->op, "OpExtInst") && I->nop >= 4 &&
	    idx(I->opnd[1]) == V.glsl) {
		if (!strcmp(I->opnd[2], "Sqrt") || !strcmp(I->opnd[2], "FAbs")) {
			a = dwg_comp(G, idx(I->opnd[3]), ci, depth + 1);
			r = mk_un(I->opnd[2][0] == 'S' ? O_SQRT : O_ABS, a);
		} else if (!strcmp(I->opnd[2], "Fma") && I->nop >= 6) {
			a = dwg_comp(G, idx(I->opnd[3]), ci, depth + 1);
			b = dwg_comp(G, idx(I->opnd[4]), ci, depth + 1);
			c = dwg_comp(G, idx(I->opnd[5]), ci, depth + 1);
			r = mk_fma(a, b, c);
		} else if (!strcmp(I->opnd[2], "Normalize")) {
			int v = idx(I->opnd[3]);
			r = mk_bin(O_DIV, dwg_comp(G, v, ci, depth + 1),
			    mk_un(O_SQRT, dwg_dot(G, v, v, n, depth + 1)));
		} else if (dw_minmax(I->opnd[2]) && I->nop >= 5) {
			a = dwg_comp(G, idx(I->opnd[3]), ci, depth + 1);
			b = dwg_comp(G, idx(I->opnd[4]), ci, depth + 1);
			r = mk_bin(dw_minmax(I->opnd[2]), a, b);
		} else if (!strcmp(I->opnd[2], "FClamp") && I->nop >= 6) {
			a = dwg_comp(G, idx(I->opnd[3]), ci, depth + 1);
			b = dwg_comp(G, idx(I->opnd[4]), ci, depth + 1);
			c = dwg_comp(G, idx(I->opnd[5]), ci, depth + 1);
			r = mk_bin(O_MIN, mk_bin(O_MAX, a, b), c);
		}
	}
	if (r == 0)
		return dwg_var(G, id, ci, TY_F64);
	dwg_ins(G, k);
	return dwg_memo(G, id, ci, r);
}

static nid
dwg_build(struct dwg *G, int id, int depth)
{
	int k = SP.def[id], i, t = V.id[id].type, sg, n;
	const struct sins *I;
	nid r = 0, a, b, c;

	for (i = 0; i < G->nmemo; i++)
		if (G->id[i] == id && G->comp[i] == -1)
			return G->node[i];
	if (V.id[id].kind == K_CONST && !V.id[id].spec &&
	    scalar_float_w(t) == 64)
		return dwg_memo(G, id, -1, mk_cast(TY_F64, mk_const_r(getf(t,
		    V.id[id].c[0]))));
	if (k < 0 || depth > 48 || dw_leaf[id])
		return dwg_var(G, id, -1, TY_F64);
	I = &V.in[k];
	if (I->nop < 2 || scalar_float_w(idx(I->opnd[0])) != 64)
		return dwg_var(G, id, -1, TY_F64);
	if (!strcmp(I->op, "OpFConvert")) {
		int s = idx(I->opnd[1]);
		if (scalar_float_w(V.id[s].type) == 32)
			r = mk_cast(TY_F64, dwg_var(G, s, -1, TY_F32));
	} else if (!strcmp(I->op, "OpConvertSToF") || !strcmp(I->op,
	    "OpConvertUToF")) {
		int s = idx(I->opnd[1]), w = scalar_int_w(V.id[s].type, &sg);
		if (w > 0 && w <= 32)
			r = mk_cast(TY_F64, dwg_var(G, s, -1, TY(I->op[9] == 'U' ?
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
	} else if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 3) {
		int s = idx(I->opnd[1]), ci = atoi(I->opnd[2]);
		if (vec_float_w(V.id[s].type, &n) == 64 && ci >= 0 && ci < n)
			r = dwg_comp(G, s, ci, depth + 1);
	} else if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 4) {
		int s = idx(I->opnd[1]), col = atoi(I->opnd[2]);
		int row = atoi(I->opnd[3]), cols, rows;
		if (mat_f64(V.id[s].type, &cols, &rows) && col >= 0 &&
		    col < cols && row >= 0 && row < rows)
			r = dwg_mcomp(G, s, col, row, depth + 1);
	} else if (!strcmp(I->op, "OpDot") && I->nop >= 3) {
		int va = idx(I->opnd[1]), vb = idx(I->opnd[2]);
		if (vec_float_w(V.id[va].type, &n) == 64)
			r = dwg_dot(G, va, vb, n, depth + 1);
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
		} else if (dw_minmax(I->opnd[2]) && I->nop >= 5) {
			a = dwg_build(G, idx(I->opnd[3]), depth + 1);
			b = dwg_build(G, idx(I->opnd[4]), depth + 1);
			r = mk_bin(dw_minmax(I->opnd[2]), a, b);
		} else if (!strcmp(I->opnd[2], "FClamp") && I->nop >= 6) {
			a = dwg_build(G, idx(I->opnd[3]), depth + 1);
			b = dwg_build(G, idx(I->opnd[4]), depth + 1);
			c = dwg_build(G, idx(I->opnd[5]), depth + 1);
			r = mk_bin(O_MIN, mk_bin(O_MAX, a, b), c);
		} else if (!strcmp(I->opnd[2], "Length")) {
			int v = idx(I->opnd[3]);
			if (vec_float_w(V.id[v].type, &n) == 64)
				r = mk_un(O_SQRT, dwg_dot(G, v, v, n, depth + 1));
		} else if (!strcmp(I->opnd[2], "Distance") && I->nop >= 5) {
			int va = idx(I->opnd[3]), vb = idx(I->opnd[4]), j;
			if (vec_float_w(V.id[va].type, &n) == 64) {
				for (j = 0; j < n; j++) {
					a = mk_bin(O_SUB, dwg_comp(G, va, j, depth + 1),
					    dwg_comp(G, vb, j, depth + 1));
					a = mk_bin(O_MUL, a, a);
					r = j ? mk_bin(O_ADD, r, a) : a;
				}
				r = mk_un(O_SQRT, r);
			}
		}
	}
	if (r == 0)
		return dwg_var(G, id, -1, TY_F64);
	dwg_ins(G, k);
	return dwg_memo(G, id, -1, r);
}

/* ---- a ucas form as SPIR-V instructions ---- */

struct dwemit {
	struct sbuf *code, *decl, *deco;
	const struct dwg *G;
	nid memo_n[1024];
	char memo_s[1024][40];
	uint8_t memo_d[1024];	/* decorated NoContraction */
	int nmemo;
	uint8_t puse[MAXP];	/* the parameters the written code reads */
	int nex;		/* components taken out of a vector */
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
		/* once: the name with its " = ", and not when the module has it
		 * (a pass before this one wrote it) */
		char key[80];
		int i, have = 0;
		snprintf(key, sizeof key, "%s = ", b);
		for (i = 0; i < V.nid && !have; i++)
			have = V.id[i].kind == K_CONST && !strcmp(V.id[i].tok, b);
		if (!have && (E->decl->p == NULL || !strstr(E->decl->p, key)))
			sb_puts(E->decl, line);
	}
	return b;
}

/* a float operation a contraction could fuse */
static int
dwe_rounds(nid n)
{
	return TYK(N.ty[n]) == K_FLOAT && (N.op[n] == O_ADD || N.op[n] == O_SUB ||
	    N.op[n] == O_MUL || N.op[n] == O_DIV || N.op[n] == O_NEG);
}

static const char *
dwe(struct dwemit *E, nid n)
{
	char a[40], b[40], c[40], nm[40];
	const char *op = NULL, *ext = NULL, *rt;
	ty_t t = N.ty[n], ta;
	int i, fl, arith = 0, deco = 0;

	for (i = 0; i < E->nmemo; i++)
		if (E->memo_n[i] == n) {
			/* written for a result that did not lean on its rounding */
			if (!E->memo_d[i] && dwe_rounds(n) && n < E->neft && E->eft &&
			    E->eft[n]) {
				sb_printf(E->deco, "OpDecorate %s NoContraction\n",
				    E->memo_s[i]);
				E->memo_d[i] = 1;
				E->ndeco++;
			}
			return E->memo_s[i];
		}
	if (E->nmemo >= 1024) {
		E->fail = 1;
		return "%ucas_toolong";
	}
	if (N.op[n] == O_VAR) {
		int j = (int)N.ki[n], pc = E->G->pcomp[j];
		E->puse[j] = 1;
		if (E->G->pname[j])
			return E->G->pname[j];
		if (pc < 0)
			return V.id[E->G->psid[j]].tok;
		rt = dwe_type(E, t);
		snprintf(nm, sizeof nm, "%%ucas_dw%d_%d", E->isl, ++E->serial);
		if (pc >= 16)
			sb_printf(E->code, "%s = OpCompositeExtract %s %s %d %d\n", nm,
			    rt, V.id[E->G->psid[j]].tok, (pc - 16) / 4, (pc - 16) % 4);
		else
			sb_printf(E->code, "%s = OpCompositeExtract %s %s %d\n", nm,
			    rt, V.id[E->G->psid[j]].tok, pc);
		E->nex++;
		goto named;
	}
	if (N.op[n] == O_CONST) {
		const char *k;
		if (t == TY_BOOL) {
			char line[160], bt[48];
			snprintf(bt, sizeof bt, "%s", bool_type(E->decl, 1));
			snprintf(nm, sizeof nm, "%%ucas_%s", N.ki[n] ? "true" :
			    "false");
			snprintf(line, sizeof line, "%s = OpConstant%s %s\n", nm,
			    N.ki[n] ? "True" : "False", bt);
			if (E->decl->p == NULL || !strstr(E->decl->p, line))
				sb_puts(E->decl, line);
			i = E->nmemo++;
			E->memo_n[i] = n;
			snprintf(E->memo_s[i], 40, "%s", nm);
			return E->memo_s[i];
		}
		k = dwe_const(E, n);
		i = E->nmemo++;
		E->memo_n[i] = n;
		E->memo_d[i] = 1;
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
		deco = 1;
	}
named:
	i = E->nmemo++;
	E->memo_n[i] = n;
	E->memo_d[i] = (uint8_t)deco;
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

/* a component's report: the lemmas of the derivation are said once, when
 * they are the first component's word for word */
static void
dw_report(const char *text, const char *first)
{
	const char *d = strstr(text, "\n  derivation ("), *e, *fd, *fe;

	if (first == NULL || d == NULL || (e = strstr(d, "\n  conditions:")) ==
	    NULL || (fd = strstr(first, "\n  derivation (")) == NULL || (fe =
	    strstr(fd, "\n  conditions:")) == NULL || e - d != fe - fd ||
	    strncmp(d, fd, (size_t)(e - d)) != 0) {
		dw_indent(text, "    ");
		return;
	}
	{
		char *head = xmalloc((size_t)(d - text) + 2);
		memcpy(head, text, (size_t)(d - text) + 1);
		head[d - text + 1] = 0;
		dw_indent(head, "    ");
		free(head);
	}
	printf("      derivation: the lemmas of the first component\n");
	dw_indent(e + 1, "    ");
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

/* ---- the f64 results of a module, grouped by what they share ---- */

struct dwsink {
	int k, src, n;		/* the conversion, its operand, its components */
	int cmp, src2;		/* a comparison (O_LT ...) and its second operand */
	int st;			/* a store of the f64 value: it stays f64 */
	int quiet;		/* a store of an f64 value no arithmetic made */
	int stay;		/* not rewritten */
	int grp;		/* the first result of its group */
	int cone[256], ncone;	/* the f64 instructions under it */
	nid root[4], body[4];
	struct dwres res[4];
	struct sbuf rep[4];
	int nlow;		/* how many of its components were lowered */
	char why[240];
	struct dwg *G;		/* its group's inputs */
	struct sbuf code;
	int failed, nins, ndeco, nex, nsplit;
	int priced;		/* stays for its group's price: its forms met */
	int last, members, gnins, gstay;	/* its group: the last one, how many */
	double gold, gnew;	/* its group's price: as written, the forms */
};

/* a conversion of an f32 or an integer to f64: exact, and free to read
 * again from its operand */
static int
dw_conv32(int j)
{
	const struct sins *I = &V.in[j];
	int n, t;

	if ((strcmp(I->op, "OpFConvert") != 0 && strcmp(I->op, "OpConvertSToF")
	    != 0 && strcmp(I->op, "OpConvertUToF") != 0) || I->nop < 2)
		return 0;
	t = V.id[idx(I->opnd[1])].type;
	return scalar_float_w(t) != 64 && vec_float_w(t, &n) != 64;
}

/* a comparison of f64 values as dword reads it (O_LT ...); 0 another op.
 * Unordered < and ordered != differ from ucas's own on a NaN only, which
 * the forms' conditions (finite inputs) exclude, but they are left alone */
static int
dw_cmp_op(const char *op)
{
	static const struct { const char *s; int op; } t[] = {
		{"OpFOrdLessThan", O_LT}, {"OpFOrdLessThanEqual", O_LE},
		{"OpFOrdGreaterThan", O_GT}, {"OpFOrdGreaterThanEqual", O_GE},
		{"OpFOrdEqual", O_EQ}, {"OpFUnordNotEqual", O_NE}, {NULL, 0}};
	int i;

	for (i = 0; t[i].s; i++)
		if (!strcmp(op, t[i].s))
			return t[i].op;
	return 0;
}

static void
dw_roots(struct dwg *G, struct dwsink *S)
{
	int i;

	if (S->cmp)
		S->root[0] = mk_bin(S->cmp, dwg_build(G, S->src, 0), dwg_build(G,
		    S->src2, 0));
	else if (S->n == 1 && scalar_float_w(V.id[S->src].type) == 64)
		S->root[0] = dwg_build(G, S->src, 0);
	else
		for (i = 0; i < S->n; i++)
			S->root[i] = dwg_comp(G, S->src, i, 0);
}

static void
dw_release(struct dwsink *S)
{
	int i;

	for (i = 0; i < S->nlow; i++) {
		dword_res_free(&S->res[i]);
		free(S->rep[i].p);
		memset(&S->rep[i], 0, sizeof S->rep[i]);
	}
	S->nlow = 0;
}

/*
 * Which results are rewritten, and with which inputs.  A result's f64
 * instructions go only when nothing else reads them, so:
 *   an f64 value read by something that keeps its f64 (a store, a phi, a
 *   result that is not rewritten) stays, and is an input of the results
 *   above it (dw_leaf);
 *   results that share f64 instructions form a group with one set of
 *   inputs, so the forms share what the f64 code shared;
 *   a group is rewritten when each form meets its requirement and the
 *   forms together cost less than the f64 instructions they replace,
 *   what is shared counted once on both sides (a form is written again in
 *   each block that needs it, and priced so).
 * A result that falls out leaves its f64 values to the others as inputs,
 * and the rest is planned again.
 */
static double
dw_plan(struct dwsink *sk, int nsk, const int *uses, const int *blk,
    const struct model *m, const struct dwreq *rq)
{
	double saved = 0;
	int *own = xmalloc((size_t)(V.nin + 1) * sizeof *own);
	int again = 1, round, s, t, q, i, o, j;

	for (round = 0; again && round < 4096; round++) {
		again = 0;
		for (s = 0; s < nsk; s++) {
			if (sk[s].G && sk[s].grp == s)
				free(sk[s].G);
		}
		for (s = 0; s < nsk; s++) {
			sk[s].G = NULL;
			if (!sk[s].stay)
				dw_release(&sk[s]);
		}
		/* the instructions under each, with the inputs so far */
		for (s = 0; s < nsk; s++) {
			struct dwg *G1;
			if (sk[s].stay)
				continue;
			G1 = xcalloc(1, sizeof *G1);
			dw_roots(G1, &sk[s]);
			sk[s].ncone = G1->nins;
			memcpy(sk[s].cone, G1->ins, (size_t)G1->nins * sizeof(int));
			if (G1->toomany || G1->nins == 0) {
				sk[s].stay = 1;
				/* a stored copy of an f64 value: nothing to say */
				sk[s].quiet = sk[s].st && !G1->toomany;
				snprintf(sk[s].why, sizeof sk[s].why, "not rewritten: %s",
				    G1->toomany ? "more than 8 inputs" :
				    dw_leaf[sk[s].src] ? "its operand is an f64 value "
				    "that stays (something else reads it too)" :
				    "its operand is an f64 value computed elsewhere (a "
				    "load, a call, a phi)");
			}
			free(G1);
		}
		/* a value something else reads stays, and is an input */
		for (j = 0; j < V.nin; j++)
			own[j] = dw_keep[j] = 0;
		for (s = 0; s < nsk; s++)
			for (q = 0; q < sk[s].ncone && !sk[s].stay; q++)
				own[sk[s].cone[q]] = 1;
		for (j = 0; j < V.nin; j++) {
			int rid = V.in[j].res, inside = 0;
			if (!own[j] || rid < 0)
				continue;
			for (t = 0; t < V.nin; t++)
				for (o = 1; own[t] && o < V.in[t].nop; o++)
					if (V.in[t].opnd[o][0] == '%' &&
					    idx(V.in[t].opnd[o]) == rid)
						inside++;
			for (t = 0; t < nsk; t++)
				if (!sk[t].stay && (sk[t].src == rid || (sk[t].cmp &&
				    sk[t].src2 == rid)))
					inside++;
			if (inside == uses[rid])
				continue;
			/* a conversion of an f32 or an integer stays for whoever
			 * else reads it and is read here from its operand */
			if (dw_conv32(j) || dw_recompute)
				dw_keep[j] = 1;
			else {
				dw_leaf[rid] = 1;
				again = 1;
			}
		}
		if (again)
			continue;
		/* results that share an f64 instruction: one group */
		for (s = 0; s < nsk; s++)
			sk[s].grp = s;
		for (j = 0; j < V.nin; j++)
			own[j] = -1;
		for (s = 0; s < nsk; s++)
			for (q = 0; q < sk[s].ncone && !sk[s].stay; q++) {
				int c = sk[s].cone[q], x = s, y;
				if (own[c] < 0) {
					own[c] = s;
					continue;
				}
				for (y = own[c]; sk[y].grp != y; y = sk[y].grp)
					;
				for (; sk[x].grp != x; x = sk[x].grp)
					;
				if (x < y)
					sk[y].grp = x;
				else
					sk[x].grp = y;
			}
		for (s = 0; s < nsk; s++) {
			int x = s;
			for (; sk[x].grp != x; x = sk[x].grp)
				;
			sk[s].grp = x;
		}
		for (s = 0; s < nsk; s++) {
			struct dwg *G;
			if (sk[s].stay || sk[s].grp != s)
				continue;
			G = xcalloc(1, sizeof *G);
			G->f.name = sym_cstr("_fp64");
			G->f.anon = 1;
			for (t = s; t < nsk; t++)
				if (!sk[t].stay && sk[t].grp == s) {
					sk[t].G = G;
					dw_roots(G, &sk[t]);
				}
			if (!G->toomany)
				continue;
			/* too many inputs together: what two of them share stays
			 * f64, and each is on its own */
			for (j = 0; j < V.nin; j++)
				own[j] = 0;
			for (t = s; t < nsk; t++)
				for (q = 0; q < sk[t].ncone && !sk[t].stay &&
				    sk[t].grp == s; q++)
					own[sk[t].cone[q]]++;
			for (j = 0; j < V.nin; j++)
				if (own[j] > 1 && V.in[j].res >= 0 && !dw_conv32(j) &&
				    !dw_leaf[V.in[j].res]) {
					dw_leaf[V.in[j].res] = 1;
					again = 1;
				}
			if (!again)
				for (t = s; t < nsk; t++)
					if (!sk[t].stay && sk[t].grp == s) {
						sk[t].stay = 1;
						snprintf(sk[t].why, sizeof sk[t].why,
						    "not rewritten: more than 8 inputs");
						again = 1;
					}
			for (j = 0; j < V.nin; j++)
				own[j] = -1;
		}
		if (again)
			continue;
		/* each result's form: priced against its own f64 code when
		 * that is all that goes, with its group otherwise */
		for (s = 0; s < nsk; s++) {
			struct dwg *G = sk[s].G;
			struct dwreq r1 = *rq;
			if (sk[s].stay)
				continue;
			/* ulp is an f32 result's: one that stays f64 needs tol or
			 * atol */
			if (sk[s].st && r1.kind == DW_ULP)
				r1.kind = DW_DEFAULT;
			for (t = 0; t < nsk; t++)
				if (t != s && !sk[t].stay && sk[t].grp == sk[s].grp)
					r1.anyprice = 1;
			for (q = 0; q < G->nins; q++)
				if (dw_keep[G->ins[q]] || sk[s].n > 1)
					r1.anyprice = 1;
			for (i = 0; i < sk[s].n; i++) {
				sk[s].body[i] = sk[s].cmp || sk[s].st ? sk[s].root[i] :
				    mk_cast(TY_F32, sk[s].root[i]);
				G->f.body = sk[s].body[i];
				sk[s].nlow = i + 1;
				if (!dword_lower(&G->f, m, &r1, &sk[s].res[i],
				    &sk[s].rep[i])) {
					sk[s].stay = 1;
					again = 1;
					break;
				}
			}
		}
		if (again)
			continue;
		/* a group's price */
		for (s = 0; s < nsk; s++) {
			nid old[320], new[256], kept[64];
			int nold = 0, nnew = 0, nkept = 0, cur = -1, members = 0;
			int last = s, nroots, gone;
			double cold, cnew = 0;
			if (sk[s].stay || sk[s].grp != s)
				continue;
			for (t = s; t < nsk; t++) {
				if (sk[t].stay || sk[t].grp != s)
					continue;
				if (blk[sk[t].k] != cur && nnew) {
					cnew += (double)fuse_roots_cost(m, new, nnew);
					nnew = 0;
				}
				cur = blk[sk[t].k];
				for (i = 0; i < sk[t].n && nold < 256; i++) {
					old[nold++] = sk[t].body[i];
					new[nnew++] = sk[t].res[i].body;
				}
				members++;
				last = t;
			}
			if (nnew)
				cnew += (double)fuse_roots_cost(m, new, nnew);
			/* what goes: the f64 instructions less the ones that stay */
			nroots = nold;
			gone = sk[s].G->nins;
			for (q = 0; q < sk[s].G->nmemo; q++) {
				int d = SP.def[sk[s].G->id[q]];
				if (d >= 0 && dw_keep[d] && nkept < 64) {
					kept[nkept++] = sk[s].G->node[q];
					old[nold++] = sk[s].G->node[q];
				}
			}
			/* what a staying instruction reads stays with it */
			for (q = 0; q < sk[s].G->nins; q++)
				own[sk[s].G->ins[q]] = dw_keep[sk[s].G->ins[q]] ? -2 : -1;
			for (i = 1; i;) {
				i = 0;
				for (q = 0; q < sk[s].G->nins; q++) {
					const struct sins *I = &V.in[sk[s].G->ins[q]];
					if (own[sk[s].G->ins[q]] != -2)
						continue;
					for (o = 1; o < I->nop; o++) {
						int d = I->opnd[o][0] == '%' ?
						    SP.def[idx(I->opnd[o])] : -1;
						for (j = 0; d >= 0 && j < sk[s].G->nins; j++)
							if (sk[s].G->ins[j] == d && own[d] == -1) {
								own[d] = -2;
								i = 1;
							}
					}
				}
			}
			for (q = 0; q < sk[s].G->nins; q++) {
				gone -= own[sk[s].G->ins[q]] == -2;
				own[sk[s].G->ins[q]] = -1;
			}
			cold = (double)fuse_roots_cost(m, old, nold);
			if (nkept)
				cold -= (double)fuse_roots_cost(m, kept, nkept);
			for (t = s; t < nsk; t++) {
				if (sk[t].stay || sk[t].grp != s)
					continue;
				sk[t].members = members;
				sk[t].last = t == last;
				sk[t].gold = cold;
				sk[t].gnew = cnew;
				sk[t].gnins = gone;
				sk[t].gstay = sk[s].G->nins - gone;
				if ((nroots > 1 || nkept) && !(cnew < cold)) {
					sk[t].stay = 1;
					sk[t].priced = 1;
					snprintf(sk[t].why, sizeof sk[t].why, "not "
					    "rewritten: the f32 form%s cost%s %g, the f64 "
					    "instructions that would go %g under %s",
					    nroots == 1 ? "" : "s", nroots == 1 ? "s" : "",
					    cnew, cold, sym_str(m->name));
				}
			}
		}
	}
	if (again)
		for (s = 0; s < nsk; s++)
			if (!sk[s].stay) {
				sk[s].stay = 1;
				snprintf(sk[s].why, sizeof sk[s].why, "not rewritten: "
				    "the results that share f64 values did not settle "
				    "into groups");
			}
	for (s = 0; s < nsk; s++)
		if (!sk[s].stay && sk[s].grp == s)
			saved += sk[s].gold - sk[s].gnew;
	free(own);
	return saved;
}

/* the forms of the results that are rewritten, as instructions in place
 * of each conversion; a value two results share is written once where
 * the first is in the same block as the second */
static void
dw_write(struct dwsink *sk, int nsk, int *uses, const int *blk,
    struct sbuf *decl, struct sbuf *deco, int *isl, int *need_glsl)
{
	int s, t, i, j;

	for (s = 0; s < nsk; s++) {
		struct dwemit *E;
		int cur = -1;
		if (sk[s].stay || sk[s].grp != s)
			continue;
		E = xcalloc(1, sizeof *E);
		E->decl = decl;
		E->deco = deco;
		E->G = sk[s].G;
		E->isl = ++*isl;
		for (t = s; t < nsk; t++) {
			struct dwsink *S = &sk[t];
			const struct sins *I = &V.in[S->k];
			char nm[4][40];
			int n0 = E->nins, d0 = E->ndeco, x0 = E->nex;
			if (S->stay || S->grp != s)
				continue;
			if (blk[S->k] != cur)
				E->nmemo = 0;
			cur = blk[S->k];
			E->code = &S->code;
			E->fail = 0;
			memset(E->puse, 0, sizeof E->puse);
			for (i = 0; i < S->n; i++) {
				E->eft = S->res[i].eft;
				E->neft = S->res[i].neft;
				E->root = S->res[i].body;
				E->final = S->n == 1 && !S->st ? V.id[I->res].tok : NULL;
				snprintf(nm[i], sizeof nm[i], "%s", dwe(E, S->res[i].body));
			}
			if (E->fail) {
				S->failed = 1;
				free(S->code.p);
				memset(&S->code, 0, sizeof S->code);
				E->nmemo = 0;
				cur = -1;
				continue;
			}
			if (S->st) {
				/* the store, of the value the form recombined */
				char vn[40];
				snprintf(vn, sizeof vn, "%s", nm[0]);
				if (S->n > 1) {
					snprintf(vn, sizeof vn, "%%ucas_dw%d_%d", E->isl,
					    ++E->serial);
					sb_printf(&S->code, "%s = OpCompositeConstruct %s",
					    vn, V.id[V.id[S->src].type].tok);
					for (i = 0; i < S->n; i++)
						sb_printf(&S->code, " %s", nm[i]);
					sb_putc(&S->code, '\n');
					E->nins++;
				}
				sb_printf(&S->code, "OpStore %s %s", I->opnd[0], vn);
				for (i = 2; i < I->nop; i++)
					sb_printf(&S->code, " %s", I->opnd[i]);
				sb_putc(&S->code, '\n');
			} else if (S->n > 1) {
				sb_printf(&S->code, "%s = OpCompositeConstruct %s",
				    V.id[I->res].tok, V.id[idx(I->opnd[0])].tok);
				for (i = 0; i < S->n; i++)
					sb_printf(&S->code, " %s", nm[i]);
				sb_putc(&S->code, '\n');
			} else if (strcmp(nm[0], V.id[I->res].tok) != 0)
				sb_printf(&S->code, "%s = OpCopyObject %s %s\n",
				    V.id[I->res].tok, V.id[idx(I->opnd[0])].tok, nm[0]);
			S->nins = E->nins - n0;
			S->ndeco = E->ndeco - d0;
			S->nex = E->nex - x0;
			*need_glsl |= E->need_glsl;
			uses[S->src]--;
			if (S->cmp)
				uses[S->src2]--;
			/* what the written code reads stays */
			for (j = 0; j < S->G->f.np; j++)
				if (E->puse[j]) {
					uses[S->G->psid[j]]++;
					if (dw_leaf[S->G->psid[j]])
						S->nsplit++;
				}
		}
		free(E);
	}
}

/* ---- a loop's f64 sum, carried as a pair ---- */

static int64_t loop_trips_iv(int h, int m, int c, int *iv, int64_t *first,
    int64_t *inc);

/*
 * An f64 phi of a loop's header that only sums: s0 before the loop,
 * s + T (or s - T) on the back edge, T not reading s, and the sum read
 * after the loop only by conversions to f32 and f64 stores.  With the
 * trip count a constant, dword_accum decides whether f32 phis carry it
 * instead: the pass is written in place of s + T, the phis in place of
 * s's, each conversion reads the high word and each store the pair
 * recombined.
 */
struct dwloop {
	int kphi, kvn, t, v0;	/* the phi, its pass, the term, s0 */
	int ea, eb;		/* the phi's operands: s0's, the back edge's */
	int ksink[8], nsink;	/* its readers after the loop */
	char kst[8];		/* the reader is an f64 store */
	int64_t n;
	int iv;			/* the loop's counter, -1 none */
	int64_t iv0, ivd;	/* its first value and step */
	int fired, tried;
	struct dwacc a;		/* the sum's pass and start, for dword_accum */
	char hname[32], lname[32];	/* the phis the rewrite writes */
	struct dwg *G;
	struct dwres res;
	struct sbuf rep;
	char why[200];
	int nins, ndeco;
};

static int
dw_loop_opnd(const struct sins *I, int id)
{
	int o, n = 0;

	for (o = 0; o < I->nop; o++)
		n += I->opnd[o][0] == '%' && idx(I->opnd[o]) == id;
	return n;
}

/* does n's expression read the variable j */
static int
dw_reads(nid n, int j, int depth)
{
	if (depth > 64)
		return 1;
	if (N.op[n] == O_VAR)
		return (int)N.ki[n] == j;
	if (N.op[n] == O_CONST)
		return 0;
	return (N.a[n] && dw_reads(N.a[n], j, depth + 1)) || (N.b[n] &&
	    dw_reads(N.b[n], j, depth + 1)) || (N.c[n] && dw_reads(N.c[n], j,
	    depth + 1));
}

static void
dw_loop_try(struct dwloop *Lp, const uint8_t *in, const int *blk,
    const int *uses, const struct model *m, const struct dwreq *rq)
{
	const struct sins *P = &V.in[Lp->kphi], *I;
	struct dwacc a;
	int i, kv, s = P->res, ps = -1, j, q, o;
	nid body, t;
	char nb[48], nm[64];
	const char *sn;

	memset(&a, 0, sizeof a);
	Lp->ea = Lp->eb = -1;
	if (P->nop != 5) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's f64 "
		    "phi has %d incoming edges, not one before the loop and one "
		    "back", (P->nop - 1) / 2);
		return;
	}
	for (i = 1; i + 1 < P->nop; i += 2) {
		int l = idx(P->opnd[i + 1]);
		if (l >= 0 && V.id[l].kind == K_LABEL && in[blk[V.id[l].label_at]])
			Lp->eb = i;
		else
			Lp->ea = i;
	}
	if (Lp->ea < 0 || Lp->eb < 0) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's f64 "
		    "phi has no edge from before the loop and one back");
		return;
	}
	Lp->v0 = idx(P->opnd[Lp->ea]);
	kv = SP.def[idx(P->opnd[Lp->eb])];
	I = kv >= 0 ? &V.in[kv] : NULL;
	if (I == NULL || I->nop < 3 || (strcmp(I->op, "OpFAdd") && strcmp(I->op,
	    "OpFSub"))) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's f64 "
		    "phi is not a sum (s + T or s - T on its back edge)");
		return;
	}
	Lp->kvn = kv;
	if (idx(I->opnd[1]) == s)
		Lp->t = idx(I->opnd[2]);
	else if (!strcmp(I->op, "OpFAdd") && idx(I->opnd[2]) == s)
		Lp->t = idx(I->opnd[1]);
	else {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's f64 "
		    "phi is not a sum (s + T or s - T on its back edge)");
		return;
	}
	if (Lp->t == s || uses[I->res] != 1) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: %s", Lp->t == s ?
		    "the pass adds the sum to itself" : "the pass's f64 value is "
		    "read by more than the phi");
		return;
	}
	/* who reads the sum: the pass, and conversions to f32 after the loop */
	for (q = 0; q < V.nin; q++) {
		const struct sins *R = &V.in[q];
		if (q == kv || q == Lp->kphi || !dw_loop_opnd(R, s) ||
		    !strncmp(R->op, "OpName", 6) || !strncmp(R->op, "OpDecorate",
		    10))
			continue;
		if (((!strcmp(R->op, "OpFConvert") && R->nop == 2 &&
		    scalar_float_w(idx(R->opnd[0])) == 32) || (!strcmp(R->op,
		    "OpStore") && R->nop >= 2 && idx(R->opnd[1]) == s &&
		    dw_loop_opnd(R, s) == 1)) && !in[blk[q]] && Lp->nsink < 8) {
			Lp->kst[Lp->nsink] = R->op[2] == 'S';
			Lp->ksink[Lp->nsink++] = q;
			continue;
		}
		if (Lp->nsink == 8)
			snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's "
			    "f64 sum has more than 8 readers");
		else
			snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's "
			    "f64 sum is read %s by %s", in[blk[q]] ? "inside the loop" :
			    "after it", R->op);
		return;
	}
	if (dw_loop_opnd(I, s) != 1) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the pass reads "
		    "the sum twice");
		return;
	}
	if (Lp->n < 0) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the loop's trip "
		    "count is not a constant (i from a constant, i < K or i <= K, "
		    "i + a constant), so there is no bound across its passes");
		return;
	}
	Lp->G = xcalloc(1, sizeof *Lp->G);
	Lp->G->f.name = sym_cstr("_sum");
	Lp->G->f.anon = 1;
	body = dwg_build(Lp->G, I->res, 0);
	for (j = 0; j < Lp->G->f.np; j++)
		if (Lp->G->psid[j] == s && Lp->G->pcomp[j] == -1)
			ps = j;
	if (ps < 0 || Lp->G->toomany || (N.op[body] != O_ADD && N.op[body] !=
	    O_SUB)) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: %s",
		    Lp->G->toomany ? "more than 8 inputs" : "the pass is not read "
		    "as s + T");
		return;
	}
	if (N.op[N.a[body]] != O_VAR || (int)N.ki[N.a[body]] != ps)
		body = mk_bin(O_ADD, N.b[body], N.a[body]);
	t = N.b[body];
	if (N.op[N.a[body]] != O_VAR || (int)N.ki[N.a[body]] != ps ||
	    dw_reads(t, ps, 0)) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the term reads "
		    "the sum");
		return;
	}
	/* the term's f64 values are its own; those before the loop are paid
	 * once as written */
	for (q = 0; q < Lp->G->nins; q++) {
		int k = Lp->G->ins[q], r = V.in[k].res, inside = 0, z;
		if (k == kv)
			continue;
		for (z = 0; z < Lp->G->nins; z++)
			inside += dw_loop_opnd(&V.in[Lp->G->ins[z]], r);
		if (inside != uses[r]) {
			snprintf(Lp->why, sizeof Lp->why, "not rewritten: an f64 "
			    "value under the term (%s) is read elsewhere too",
			    V.id[r].tok);
			return;
		}
		if (!in[blk[k]] && a.nonce < 16)
			for (z = 0; z < Lp->G->nmemo; z++)
				if (Lp->G->id[z] == r && Lp->G->comp[z] == -1)
					a.once[a.nonce++] = Lp->G->node[z];
	}
	/* s0: an f64 constant or an f32 converted */
	if (V.id[Lp->v0].kind == K_CONST && !V.id[Lp->v0].spec &&
	    scalar_float_w(V.id[Lp->v0].type) == 64) {
		double c = getf(V.id[Lp->v0].type, V.id[Lp->v0].c[0]);
		float hi = (float)c, lo = (float)(c - (double)hi);
		if (isinf(hi) || isnan(c)) {
			snprintf(Lp->why, sizeof Lp->why, "not rewritten: the sum "
			    "starts at %g, past f32", c);
			return;
		}
		a.s0h = mk_cast(TY_F32, mk_const_r((double)hi));
		a.s0l = lo != 0 ? mk_cast(TY_F32, mk_const_r((double)lo)) : 0;
		a.s0lo = a.s0hi = c;
		a.e0 = fabsl((long double)c - hi - lo);
	} else if (SP.def[Lp->v0] >= 0 && !strcmp(V.in[SP.def[Lp->v0]].op,
	    "OpFConvert") && scalar_float_w(V.id[idx(V.in[SP.def[Lp->v0]].
	    opnd[1])].type) == 32) {
		int x = idx(V.in[SP.def[Lp->v0]].opnd[1]);
		int jx = dwg_param(Lp->G, x, -1, TY_F32);
		if (!Lp->G->f.p[jx].ranged) {
			snprintf(Lp->why, sizeof Lp->why, "not rewritten: the sum "
			    "starts at %s, which has no range", V.id[x].tok);
			return;
		}
		a.s0h = mk_var((uint32_t)jx, Lp->G->f.p[jx].name, TY_F32);
		a.s0lo = Lp->G->f.p[jx].rlo;
		a.s0hi = Lp->G->f.p[jx].rhi;
		if (uses[Lp->v0] == 1 && a.nonce < 16)
			a.once[a.nonce++] = mk_cast(TY_F64, a.s0h);
	} else {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: the sum starts "
		    "at an f64 value computed before the loop (a constant or an "
		    "f32 converted is taken)");
		return;
	}
	/* the pair the passes carry: the phis the rewrite writes, named after
	 * the sum (s, or sum when a parameter is s) */
	sn = load_name(s, nb, sizeof nb);
	if (sn == NULL) {
		sn = "s";
		for (j = 0; j < Lp->G->f.np; j++)
			if (j != ps && !strcmp(sym_str(Lp->G->f.p[j].name), "s"))
				sn = "sum";
		Lp->G->f.p[ps].name = sym_cstr(sn);
		body = mk_bin(N.op[body], mk_var((uint32_t)ps,
		    Lp->G->f.p[ps].name, TY_F64), t);
	}
	snprintf(Lp->hname, sizeof Lp->hname, "%%ucas_sh%d", Lp->kphi);
	snprintf(Lp->lname, sizeof Lp->lname, "%%ucas_sl%d", Lp->kphi);
	for (o = 0; o < 2; o++) {
		j = dwg_param(Lp->G, s, -3 - o, TY_F32);
		snprintf(nm, sizeof nm, "%.40s_%s", sn, o ? "lo" : "hi");
		Lp->G->f.p[j].name = sym_cstr(nm);
		Lp->G->pname[j] = o ? Lp->lname : Lp->hname;
		Lp->G->f.p[j].ranged = 0;
		if (o)
			a.pl = j;
		else
			a.ph = j;
	}
	if (Lp->G->toomany) {
		snprintf(Lp->why, sizeof Lp->why, "not rewritten: more than 8 "
		    "inputs");
		return;
	}
	/* the loop's counter, read by the term (converted), runs from its
	 * first value to the one that leaves */
	for (j = 0; Lp->iv >= 0 && j < Lp->G->f.np; j++) {
		struct param *pp = &Lp->G->f.p[j];
		int d = SP.def[Lp->G->psid[j]], ok = Lp->G->psid[j] == Lp->iv &&
		    TY_ISINT(pp->ty);
		if (!ok && d >= 0 && pp->ty == TY_F32 && V.in[d].nop >= 2 &&
		    (!strcmp(V.in[d].op, "OpConvertSToF") || !strcmp(V.in[d].op,
		    "OpConvertUToF")))
			ok = idx(V.in[d].opnd[1]) == Lp->iv;
		if (!ok || pp->ranged || Lp->G->pcomp[j] != -1)
			continue;
		pp->ranged = 1;
		pp->rlo = (double)Lp->iv0;
		pp->rhi = (double)(Lp->iv0 + Lp->n * Lp->ivd);
		if (TY_ISINT(pp->ty)) {
			pp->lo = Lp->iv0;
			pp->hi = Lp->iv0 + Lp->n * Lp->ivd;
		}
	}
	for (q = 0; q < Lp->nsink; q++) {
		a.nconv += !Lp->kst[q];
		a.nstore += Lp->kst[q];
	}
	a.ps = ps;
	a.n = Lp->n;
	Lp->G->f.body = body;
	Lp->tried = 1;
	Lp->fired = dword_accum(&Lp->G->f, &a, m, rq, &Lp->res, &Lp->rep);
	if (!Lp->fired)
		snprintf(Lp->why, sizeof Lp->why, "%s", Lp->res.why);
	Lp->a = a;
}

/* the loops' f64 phis, each tried as a sum; lsink: the conversions that
 * read one (1 + its index) */
static int
dw_loops(struct dwloop **out, const int *uses, const struct model *m,
    const struct dwreq *rq, int *lsink)
{
	int nb = 0, k, pc, q, n = 0, cap = 0;
	int *blk = xmalloc(((size_t)V.nin + 1) * sizeof *blk);
	int *bstart = xmalloc(((size_t)V.nin + 1) * sizeof *bstart), *st;
	uint8_t *in;
	struct dwloop *lp;

	for (k = 0; k < V.nin; k++) {
		if (!strcmp(V.in[k].op, "OpLabel"))
			bstart[nb++] = k;
		blk[k] = nb - 1;
		cap += !strcmp(V.in[k].op, "OpPhi");
	}
	lp = xcalloc((size_t)cap + 1, sizeof *lp);
	in = xmalloc((size_t)nb + 1);
	st = xmalloc(((size_t)nb + 1) * sizeof *st);
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int h, mg, ct, sp = 0, b0, iv;
		int64_t trips, iv0 = 0, ivd = 0;
		if (strcmp(I->op, "OpLoopMerge") || I->nop < 2 || blk[k] < 0)
			continue;
		b0 = blk[k];
		h = V.in[bstart[b0]].res;
		mg = idx(I->opnd[0]);
		ct = idx(I->opnd[1]);
		/* the loop's blocks: reached from its header, not through its
		 * merge */
		memset(in, 0, (size_t)nb + 1);
		st[sp++] = b0;
		in[b0] = 1;
		while (sp) {
			int b = st[--sp], e = b + 1 < nb ? bstart[b + 1] : V.nin;
			for (pc = bstart[b] + 1; pc < e; pc++) {
				const struct sins *T2 = &V.in[pc];
				if (strcmp(T2->op, "OpBranch") &&
				    strcmp(T2->op, "OpBranchConditional") &&
				    strcmp(T2->op, "OpSwitch"))
					continue;
				for (q = !strcmp(T2->op, "OpBranch") ? 0 : 1; q < T2->nop;
				    q++) {
					int l = T2->opnd[q][0] == '%' ? idx(T2->opnd[q]) : -1;
					if (l < 0 || l == mg || V.id[l].kind != K_LABEL ||
					    in[blk[V.id[l].label_at]])
						continue;
					in[blk[V.id[l].label_at]] = 1;
					st[sp++] = blk[V.id[l].label_at];
				}
			}
		}
		iv = -1;
		trips = h >= 0 && mg >= 0 && ct >= 0 ? loop_trips_iv(h, mg, ct, &iv,
		    &iv0, &ivd) : -1;
		for (pc = bstart[b0] + 1; pc < k && n < cap; pc++) {
			const struct sins *P = &V.in[pc];
			struct dwloop *Lp = &lp[n];
			if (strcmp(P->op, "OpPhi") || P->res < 0 || P->nop < 1 ||
			    scalar_float_w(idx(P->opnd[0])) != 64)
				continue;
			Lp->kphi = pc;
			Lp->n = trips;
			Lp->iv = trips >= 0 ? iv : -1;
			Lp->iv0 = iv0;
			Lp->ivd = ivd;
			dw_loop_try(Lp, in, blk, uses, m, rq);
			if (!Lp->tried)
				for (q = 0; q < V.nin && Lp->nsink < 8; q++)
					if (!strcmp(V.in[q].op, "OpFConvert") &&
					    V.in[q].nop == 2 && scalar_float_w(idx(
					    V.in[q].opnd[0])) == 32 && idx(V.in[q].opnd[1]) ==
					    P->res)
						Lp->ksink[Lp->nsink++] = q;
			for (q = 0; q < Lp->nsink; q++)
				lsink[Lp->ksink[q]] = n + 1;
			n++;
		}
	}
	free(in);
	free(st);
	free(blk);
	free(bstart);
	*out = lp;
	return n;
}

/* the pass in place of s + T, the phis in place of s's, and each
 * conversion reading the high word */
static void
dw_loop_write(struct dwloop *Lp, int *uses, char **repl, struct sbuf *decl,
    struct sbuf *deco, int *isl, int *need_glsl)
{
	struct dwemit *E = xcalloc(1, sizeof *E);
	const struct sins *P = &V.in[Lp->kphi];
	struct sbuf code = {0}, phis = {0};
	char hn[40], ln[40], h0[64], l0[64];
	const char *ft;
	int i, j;

	E->decl = decl;
	E->deco = deco;
	E->G = Lp->G;
	E->isl = ++*isl;
	E->code = &code;
	E->eft = Lp->res.eft;
	E->neft = Lp->res.neft;
	snprintf(hn, sizeof hn, "%s", dwe(E, Lp->res.hi));
	snprintf(ln, sizeof ln, "%s", Lp->res.lo ? dwe(E, Lp->res.lo) : "");
	snprintf(h0, sizeof h0, "%s", dwe(E, Lp->a.s0h));
	snprintf(l0, sizeof l0, "%s", dwe(E, Lp->a.s0l ? Lp->a.s0l :
	    mk_cast(TY_F32, mk_const_r(0))));
	if (E->fail) {
		Lp->fired = 0;
		snprintf(Lp->why, sizeof Lp->why, "not written: an operation of "
		    "the form has no SPIR-V instruction here");
		free(code.p);
		free(E);
		return;
	}
	ft = dwe_type(E, TY_F32);
	sb_printf(&phis, "%s = OpPhi %s %s %s %s %s\n", Lp->hname, ft, h0,
	    P->opnd[Lp->ea + 1], hn, P->opnd[Lp->eb + 1]);
	if (Lp->res.lo)
		sb_printf(&phis, "%s = OpPhi %s %s %s %s %s\n", Lp->lname, ft, l0,
		    P->opnd[Lp->ea + 1], ln, P->opnd[Lp->eb + 1]);
	repl[Lp->kphi] = phis.p;
	repl[Lp->kvn] = code.p;
	for (i = 0; i < Lp->nsink; i++) {
		const struct sins *R = &V.in[Lp->ksink[i]];
		struct sbuf c = {0};
		if (Lp->kst[i]) {
			/* the store, of the pair recombined */
			const char *dt = dwe_type(E, TY_F64);
			char d[48];
			snprintf(d, sizeof d, "%%ucas_ss%d_%d", Lp->kphi, i);
			sb_printf(&c, "%sh = OpFConvert %s %s\n", d, dt, Lp->hname);
			if (Lp->res.lo) {
				sb_printf(&c, "%sl = OpFConvert %s %s\n", d, dt,
				    Lp->lname);
				sb_printf(&c, "%s = OpFAdd %s %sh %sl\n", d, dt, d, d);
			} else
				sb_printf(&c, "%s = OpCopyObject %s %sh\n", d, dt, d);
			sb_printf(&c, "OpStore %s %s", R->opnd[0], d);
			for (j = 2; j < R->nop; j++)
				sb_printf(&c, " %s", R->opnd[j]);
			sb_putc(&c, '\n');
		} else
			sb_printf(&c, "%s = OpCopyObject %s %s\n", V.id[R->res].tok,
			    ft, Lp->hname);
		repl[Lp->ksink[i]] = c.p;
	}
	/* what the replaced instructions read goes, what the code reads
	 * stays */
	uses[P->res] = 0;
	uses[V.in[Lp->kvn].res] = 0;
	uses[Lp->t]--;
	uses[Lp->v0]--;
	for (j = 0; j < Lp->G->f.np; j++)
		if (E->puse[j] && !Lp->G->pname[j])
			uses[Lp->G->psid[j]]++;
	Lp->nins = E->nins;
	Lp->ndeco = E->ndeco;
	*need_glsl |= E->need_glsl;
	free(E);
}

static void
dw_loop_report(const struct dwloop *Lp, int k)
{
	if (Lp->ksink[0] != k) {
		printf("    the same loop's sum as above\n");
		return;
	}
	printf("    a loop's f64 sum (%s, the phi of its header):\n",
	    V.id[V.in[Lp->kphi].res].tok);
	if (Lp->rep.p)
		dw_report(Lp->rep.p, NULL);
	else
		printf("    %s\n", Lp->why);
	if (Lp->rep.p && !Lp->fired && Lp->why[0] && !strstr(Lp->rep.p,
	    Lp->why))
		printf("    %s\n", Lp->why);
	if (!Lp->fired)
		return;
	printf("    written: %s carr%s the sum across the passes, each pass %d "
	    "f32 instruction%s,\n      %d of them NoContraction, in place of s "
	    "%c T; %s\n", Lp->res.lo ? "two f32 phis (hi, lo)" : "an f32 phi",
	    Lp->res.lo ? "y" : "ies", Lp->nins, Lp->nins == 1 ? "" : "s",
	    Lp->ndeco, strcmp(V.in[Lp->kvn].op, "OpFSub") ? '+' : '-',
	    Lp->a.nstore == 0 ? "each conversion after the loop reads the high "
	    "word" : Lp->a.nconv == 0 ? "a store takes the pair recombined "
	    "(an f64 add stays)" : "a conversion after the loop reads the high "
	    "word, a store the pair\n      recombined (an f64 add stays)");
}

static char *
dword_pass(const char *text, int *changed)
{
	struct sbuf decl = {0}, deco = {0}, out = {0};
	char **repl, *dead;
	int k, i, i2 = 0, nsink = 0, napplied = 0, nfold = 0, nfoldfix = 0, need_glsl = 0;
	int f64t = -1, isl = 0, has64 = 0, hdr = 0;
	const struct model *m = model_active();
	struct dwreq rq = {DW_DEFAULT, 0};
	struct dwsink *sk;
	int *uses, *blk, nsk = 0, cursk = 0, *lsink, nlp, q;
	struct dwloop *lp;

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
	if (dw_req.kind != DW_DEFAULT)
		rq = dw_req;
	else if (fopt.mode == 2) {
		rq.kind = DW_ULP;
		rq.v = fopt.ulp > 0 ? fopt.ulp : 0x1p-10;
	}
	/* the f64 results: which are rewritten, and their code */
	for (k = 0, i = 1; k < V.nin; k++)
		i += !strcmp(V.in[k].op, "OpFConvert") || dw_cmp_op(V.in[k].op) ||
		    !strcmp(V.in[k].op, "OpStore");
	sk = xcalloc((size_t)i, sizeof *sk);
	blk = xcalloc((size_t)V.nin + 1, sizeof *blk);
	dw_leaf = xcalloc((size_t)V.nid + 64, 1);
	dw_keep = xcalloc((size_t)V.nin + 1, 1);
	/* a loop's f64 sum: the conversions that read it are its own */
	lsink = xcalloc((size_t)V.nin + 1, sizeof *lsink);
	nlp = dw_loops(&lp, uses, m, &rq, lsink);
	for (k = 0, i = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int n = 1, src, t;
		if (!strcmp(I->op, "OpLabel"))
			i++;
		blk[k] = i;
		if (lsink[k])
			continue;
		if (dw_cmp_op(I->op) && I->nop >= 3 && I->res >= 0 &&
		    SP.def[I->res] == k && scalar_float_w(V.id[idx(I->opnd[1])].type)
		    == 64 && scalar_float_w(V.id[idx(I->opnd[2])].type) == 64) {
			sk[nsk].k = k;
			sk[nsk].cmp = dw_cmp_op(I->op);
			sk[nsk].src = idx(I->opnd[1]);
			sk[nsk].src2 = idx(I->opnd[2]);
			sk[nsk].n = 1;
			sk[nsk].grp = nsk;
			nsk++;
			continue;
		}
		/* an f64 value a function computes and stores: it stays f64 */
		if (!strcmp(I->op, "OpStore") && I->nop >= 2 && I->opnd[1][0] ==
		    '%') {
			src = idx(I->opnd[1]);
			t = V.id[src].type;
			if (SP.def[src] < 0 || (scalar_float_w(t) != 64 &&
			    vec_float_w(t, &n) != 64))
				continue;
			sk[nsk].k = k;
			sk[nsk].src = src;
			sk[nsk].n = n;
			sk[nsk].st = 1;
			sk[nsk].grp = nsk;
			nsk++;
			continue;
		}
		if (strcmp(I->op, "OpFConvert") != 0 || I->nop < 2 || I->res < 0 ||
		    SP.def[I->res] != k)
			continue;
		t = idx(I->opnd[0]);
		src = idx(I->opnd[1]);
		if (scalar_float_w(t) == 32) {
			if (scalar_float_w(V.id[src].type) != 64)
				continue;
		} else if (vec_float_w(t, &n) != 32 || vec_float_w(V.id[src].type,
		    &i2) != 64 || i2 != n)
			continue;
		sk[nsk].k = k;
		sk[nsk].src = src;
		sk[nsk].n = n;
		sk[nsk].grp = nsk;
		nsk++;
	}
	/* an f64 value that stays for another reader: split where the
	 * results above it need it, or computed again in f32; the plan that
	 * saves more under the model */
	{
		struct dwsink *sk2 = xmalloc(((size_t)nsk + 1) * sizeof *sk2);
		char *leaf2 = xcalloc((size_t)V.nid + 64, 1), *leaf1 = dw_leaf;
		char *keep2 = xcalloc((size_t)V.nin + 1, 1), *keep1 = dw_keep;
		double s1, s2;

		memcpy(sk2, sk, ((size_t)nsk + 1) * sizeof *sk2);
		dw_recompute = 0;
		s1 = dw_plan(sk, nsk, uses, blk, m, &rq);
		dw_leaf = leaf2;
		dw_keep = keep2;
		dw_recompute = 1;
		s2 = dw_plan(sk2, nsk, uses, blk, m, &rq);
		dw_recompute = 0;
		if (s2 >= s1) {
			struct dwsink *x = sk;
			sk = sk2;
			sk2 = x;
			leaf2 = leaf1;
			keep2 = keep1;
		} else {
			dw_leaf = leaf1;
			dw_keep = keep1;
		}
		for (k = 0; k < nsk; k++) {
			if (sk2[k].G && sk2[k].grp == k)
				free(sk2[k].G);
			dw_release(&sk2[k]);
		}
		free(sk2);
		free(leaf2);
		free(keep2);
	}
	dw_write(sk, nsk, uses, blk, &decl, &deco, &isl, &need_glsl);
	for (q = 0; q < nlp; q++)
		if (lp[q].fired) {
			dw_loop_write(&lp[q], uses, repl, &decl, &deco, &isl,
			    &need_glsl);
			napplied += lp[q].fired;
		}
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		struct dwsink *S;
		char sl[200];
		int src, issk = cursk < nsk && sk[cursk].k == k;

		if (lsink[k]) {
			if (!hdr++)
				printf("fp64 and conversions under %s:\n",
				    sym_str(m->name));
			printf("  %s\n", I->text);
			src_line(ins_line(k), sl, sizeof sl);
			fputs(sl, stdout);
			dw_loop_report(&lp[lsink[k] - 1], k);
			continue;
		}
		if (!issk && (strcmp(I->op, "OpFConvert") != 0 || I->nop < 2 ||
		    I->res < 0 || SP.def[I->res] != k))
			continue;
		if (issk) {
			S = &sk[cursk++];
			if (S->quiet)
				continue;
			has64 = 1;
			nsink++;
			if (!hdr++)
				printf("fp64 and conversions under %s:\n",
				    sym_str(m->name));
			printf("  %s\n", I->text);
			src_line(ins_line(k), sl, sizeof sl);
			fputs(sl, stdout);
			for (i = 0; i < S->nlow; i++) {
				char *cut = S->priced && S->rep[i].p ? strstr(S->rep[i].p,
				    "\n  -> ") : NULL;
				/* what its form alone would save is not what goes */
				if (cut)
					cut[1] = 0;
				if (S->rep[i].p)
					dw_report(S->rep[i].p, i ? S->rep[0].p : NULL);
				else
					printf("    %s\n", S->res[i].why);
			}
			if (S->stay) {
				if (S->why[0])
					printf("    %s\n", S->why);
				continue;
			}
			if (S->failed) {
				printf("    not written: an operation of the form has "
				    "no SPIR-V instruction here\n");
				continue;
			}
			repl[k] = S->code.p;
			S->code.p = NULL;
			napplied++;
			if (S->nsplit)
				printf("    %d f64 value%s under it stay%s f64 (read by "
				    "what keeps its f64) and %s split here\n", S->nsplit,
				    S->nsplit == 1 ? "" : "s", S->nsplit == 1 ? "s" : "",
				    S->nsplit == 1 ? "is" : "are");
			if (S->gstay && (S->last || (S->members == 1 && S->n == 1)))
				printf("    %d of the %d f64 instructions under it stay "
				    "for what else reads them: their values are "
				    "computed again here, in f32;\n    the %d that go"
				    "%s cost%s %g, the form%s %g (%+.1f%%) under %s\n",
				    S->gstay, S->gstay + S->gnins, S->gnins,
				    S->gnins == 1 ? "es" : "", S->gnins == 1 ? "s" : "",
				    S->gold,
				    S->members == 1 && S->n == 1 ? "" : "s", S->gnew,
				    100 * (S->gnew - S->gold) / S->gold, sym_str(m->name));
			if (S->st)
				printf("    stored as the pair recombined, f64(hi) + "
				    "f64(lo): an f64 add stays\n");
			if (S->members == 1 && S->n == 1) {
				printf("    written as %d %sinstruction%s, %d of them "
				    "NoContraction, in place of %d f64 one%s\n",
				    S->nins, S->st ? "" : "f32 ", S->nins == 1 ? "" :
				    "s", S->ndeco, S->gnins, S->gnins == 1 ? "" : "s");
				continue;
			}
			printf("    written as %d %sinstruction%s, %d of them "
			    "NoContraction", S->nins, S->st ? "" : "f32 ",
			    S->nins == 1 ? "" : "s", S->ndeco);
			if (S->nex)
				printf(", and %d component%s taken out of a vector",
				    S->nex, S->nex == 1 ? "" : "s");
			putchar('\n');
			if (S->last) {
				if (S->members > 1) {
					int t, first = 1;
					printf("    with");
					for (t = 0; t < nsk; t++)
						if (&sk[t] != S && !sk[t].stay &&
						    sk[t].grp == S->grp) {
							printf("%s %s", first ? "" : ",",
							    V.id[V.in[sk[t].k].res].tok);
							first = 0;
						}
					printf(", which share%s its f64 values:",
					    S->members == 2 ? "s" : "");
				} else
					printf("    its %d components together:", S->n);
				printf(" cost %g against %g (%+.1f%%) under %s, what "
				    "they share counted once; in place of %d f64 "
				    "instruction%s\n", S->gnew, S->gold, 100 *
				    (S->gnew - S->gold) / S->gold, sym_str(m->name),
				    S->gnins, S->gnins == 1 ? "" : "s");
			}
			continue;
		}
		if (scalar_float_w(idx(I->opnd[0])) != 32)
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
			x = dwg_var(F2, idx(V.in[SP.def[src]].opnd[1]), -1, TY_F32);
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
				    !strcmp(I->op, "OpConvertUToF") ||
				    !strcmp(I->op, "OpDot") || !strncmp(I->op,
				    "OpMatrixTimes", 13) || !strcmp(I->op,
				    "OpVectorTimesMatrix") || !strcmp(I->op,
				    "OpTranspose")))
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
	for (k = 0; k < nsk; k++) {
		if (sk[k].G && sk[k].grp == k)
			free(sk[k].G);
		dw_release(&sk[k]);
		free(sk[k].code.p);
	}
	free(sk);
	free(blk);
	for (q = 0; q < nlp; q++) {
		free(lp[q].G);
		free(lp[q].rep.p);
		dword_res_free(&lp[q].res);
	}
	free(lp);
	free(lsink);
	free(dw_leaf);
	dw_leaf = NULL;
	free(dw_keep);
	dw_keep = NULL;
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
	memset(&dw_req, 0, sizeof dw_req);
	if (!*changed) {
		free(out.p);
		return NULL;
	}
	return out.p;
}
