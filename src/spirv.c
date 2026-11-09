/*
 * SPIR-V assembly (spirv-dis text) executed on a group of SIMT lanes, for
 * "animate exec" followed by a ```spv block.
 *
 * The module is read line by line: types, constants, names, decorations
 * and variables first, then the entry point's function.  Execution is in
 * lockstep with a lane mask:
 *   - a conditional branch under OpSelectionMerge runs the true side with
 *     the other lanes masked, then the false side, then reconverges at the
 *     merge block (every lane pays for both sides when they disagree);
 *   - a loop (OpLoopMerge) iterates while any lane still runs; lanes that
 *     left wait masked; break and continue are followed per lane;
 *   - OpSwitch groups the lanes by target;
 *   - OpReturn, OpKill and friends retire lanes; OpFunctionCall runs the
 *     callee with the caller's mask;
 *   - OpControlBarrier stalls every lane, atomics serialize the lanes.
 * Memory: Uniform, StorageBuffer, Workgroup, PushConstant and
 * UniformConstant variables are shared by the lanes; Function, Private,
 * Input and Output are per lane.  Values are raw bit patterns per
 * component, read through their type.  Each executed instruction goes to
 * the animation (anim.c) with its cost under the active model.
 */
#include "ucas.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define NL 32
#define MAXOP 80
#define MAXCOMP 4

enum { K_NONE, K_TYPE, K_CONST, K_VAR, K_FUNC, K_LABEL, K_VALUE, K_EXT,
	K_PARAM };
enum { S_VOID, S_BOOL, S_INT, S_FLOAT, S_VEC, S_STRUCT, S_ARRAY, S_PTR,
	S_FN, S_OTHER };
enum { SC_FUNCTION, SC_PRIVATE, SC_INPUT, SC_OUTPUT, SC_SHARED, SC_WG };

struct sins {
	char op[40];
	int res;		/* result id index, -1 */
	int nop;
	char *opnd[MAXOP];
	char *text;
};

struct sty {
	int k, w, sign, n;	/* kind, width, signed, components / length */
	int e;			/* element / pointee / component type */
	int m[48];		/* struct members */
	int nm;
	int sc;			/* pointer: storage class */
	int slots;
};

struct sid {
	int kind;
	char name[40];
	char tok[40];		/* the id as written ("%23"), for writing back */
	int type;		/* the value's type id */
	struct sty t;		/* when kind == K_TYPE */
	uint64_t c[MAXCOMP];	/* constant components (scalar or vector) */
	int nc;
	int sc, builtin, nslots;
	uint64_t *mem;		/* slots, shared or lanes * nslots */
	int *slotty;		/* scalar type per slot */
	int label_at;		/* K_LABEL: index of OpLabel */
	int spec;		/* a specialization constant: may change */
	int func_at;		/* K_FUNC: index of OpFunction */
};

struct lval {
	uint64_t b[MAXCOMP];
	int var;		/* pointer: variable id */
	int off;		/* pointer: first slot */
};

static struct {
	struct sins *in;
	int nin, cap;
	struct sid *id;
	int nid, idcap;
	struct lval *v[NL];	/* per lane, per id: lazily lv[id * NL + lane] */
	struct lval *lv;
	int lvcap;		/* ids the value table has room for */
	int lanes;
	uint32_t all, retired;
	int curblk[NL], prevblk[NL];
	int glsl;		/* the GLSL.std.450 import id */
	int depth;
	int steps;
	struct lval ret[NL];
	char err[256];
	uint64_t rng;
	int seeded;
	long oob;		/* accesses outside a variable */
	char oobvar[40];	/* the first one's variable */
} V;

static void
die(const char *fmt, const char *a)
{
	snprintf(V.err, sizeof V.err, fmt, a);
	fail("SPIR-V: %s", V.err);
}

/* ---- ids ---- */

static int *symmap;
static uint32_t nsymmap;

static int
idx(const char *name)
{
	uint32_t s = sym_cstr(name);
	int i;

	if (s >= nsymmap) {
		uint32_t n = nsymmap ? nsymmap : 256;
		while (n <= s)
			n *= 2;
		symmap = xrealloc(symmap, n * sizeof *symmap);
		for (; nsymmap < n; nsymmap++)
			symmap[nsymmap] = -1;
	}
	if (symmap[s] >= 0)
		return symmap[s];
	if (V.nid == V.idcap) {
		V.idcap = V.idcap ? V.idcap * 2 : 256;
		V.id = xrealloc(V.id, (size_t)V.idcap * sizeof *V.id);
	}
	i = V.nid++;
	memset(&V.id[i], 0, sizeof V.id[i]);
	snprintf(V.id[i].name, sizeof V.id[i].name, "%s", name);
	snprintf(V.id[i].tok, sizeof V.id[i].tok, "%s", name);
	V.id[i].type = -1;
	V.id[i].builtin = -1;
	symmap[s] = i;
	return i;
}

/* a new module: no id is known yet (V is reset by the caller) */
static void
symmap_reset(void)
{
	uint32_t i;

	for (i = 0; i < nsymmap; i++)
		symmap[i] = -1;
}

static int
opid(const struct sins *I, int k)
{
	if (k >= I->nop || I->opnd[k][0] != '%')
		die("expected an id in \"%s\"", I->text);
	return idx(I->opnd[k]);
}

static struct sty *
T(int t)
{
	if (t < 0 || V.id[t].kind != K_TYPE)
		die("not a type: %s", t >= 0 ? V.id[t].name : "?");
	return &V.id[t].t;
}

/* ---- parsing ---- */

static void
parse(const char *src)
{
	const char *p = src, *e;
	char line[1024], *q, *tok;
	int n;

	while (*p) {
		struct sins I;
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		n = (int)(e - p) < (int)sizeof line - 1 ? (int)(e - p) :
		    (int)sizeof line - 1;
		memcpy(line, p, (size_t)n);
		line[n] = 0;
		p = *e ? e + 1 : e;
		if ((q = strchr(line, ';')) != NULL) {
			/* a comment, unless inside a string */
			int inq = 0;
			for (q = line; *q; q++) {
				if (*q == '"')
					inq = !inq;
				if (*q == ';' && !inq) {
					*q = 0;
					break;
				}
			}
		}
		for (q = line; isspace((unsigned char)*q); q++)
			;
		if (*q == 0)
			continue;
		memset(&I, 0, sizeof I);
		I.res = -1;
		I.text = strdup(q);
		{
			char *k = q + strlen(q);
			while (k > q && isspace((unsigned char)k[-1]))
				*--k = 0;
			I.text[strlen(q)] = 0;
		}
		/* "%id = OpX ..." or "OpX ..." */
		tok = q;
		if (*q == '%') {
			char *eq = strchr(q, '=');
			char name[64];
			int ln;
			if (eq == NULL)
				die("no '=' in \"%s\"", I.text);
			for (ln = 0; q[ln] && !isspace((unsigned char)q[ln]) &&
			    q[ln] != '=' && ln < 63; ln++)
				name[ln] = q[ln];
			name[ln] = 0;
			I.res = idx(name);
			tok = eq + 1;
		}
		while (isspace((unsigned char)*tok))
			tok++;
		n = 0;
		while (tok[n] && !isspace((unsigned char)tok[n]) && n < 39) {
			I.op[n] = tok[n];
			n++;
		}
		I.op[n] = 0;
		if (strncmp(I.op, "Op", 2) != 0)
			die("not an instruction: \"%s\"", I.text);
		tok += n;
		/* operands: ids, numbers, words, "strings" */
		for (;;) {
			char *st;
			while (isspace((unsigned char)*tok))
				tok++;
			if (*tok == 0 || I.nop == MAXOP)
				break;
			st = tok;
			if (*tok == '"') {
				tok++;
				while (*tok && *tok != '"')
					tok++;
				if (*tok == '"')
					tok++;
			} else
				while (*tok && !isspace((unsigned char)*tok))
					tok++;
			I.opnd[I.nop] = xmalloc((size_t)(tok - st) + 1);
			memcpy(I.opnd[I.nop], st, (size_t)(tok - st));
			I.opnd[I.nop][tok - st] = 0;
			I.nop++;
		}
		if (V.nin == V.cap) {
			V.cap = V.cap ? V.cap * 2 : 128;
			V.in = xrealloc(V.in, (size_t)V.cap * sizeof *V.in);
		}
		V.in[V.nin++] = I;
	}
}

/* ---- completing a snippet ---- */

/*
 * A snippet may lack the module around it: types, constants, the
 * extended-instruction import, an entry point, the function, its return.
 * Missing types and constants are made from glslang's names (%int,
 * %v4float, %_ptr_Uniform_float, %uint_1, %float_0_5, %int_n1, %true);
 * a missing function wraps the instructions; any other undefined operand
 * is an input, typed from its use, with a value per lane that is asked
 * for when it can be (random per lane, the lane index, 0 or 1).
 */
#define MAXIN 32
struct spvin {
	char name[40];
	char ty[40];		/* type id name, "%float" */
	int mode;		/* 0 random, 1 lane index, 2 zero, 3 one, 4 value */
	double v;
	int var;		/* used as a pointer: a per-lane variable */
};
static struct spvin inps[MAXIN];
static int values_unneeded;	/* simplify: inputs need types, not values */
static int ninps;
struct spvset spv_set[16];
int nspv_set;

static int
defined_name(const char *nm)
{
	int k;

	for (k = 0; k < V.nin; k++)
		if (V.in[k].res >= 0 && !strcmp(V.id[V.in[k].res].name, nm))
			return 1;
	return 0;
}

static int
is_scalar_name(const char *b, const char **decl)
{
	static const struct { const char *n, *d; } t[] = {
		{"void", "OpTypeVoid"}, {"bool", "OpTypeBool"},
		{"int", "OpTypeInt 32 1"}, {"uint", "OpTypeInt 32 0"},
		{"float", "OpTypeFloat 32"}, {"double", "OpTypeFloat 64"},
		{"half", "OpTypeFloat 16"}, {"int8", "OpTypeInt 8 1"},
		{"uint8", "OpTypeInt 8 0"}, {"int16", "OpTypeInt 16 1"},
		{"uint16", "OpTypeInt 16 0"}, {"int64", "OpTypeInt 64 1"},
		{"uint64", "OpTypeInt 64 0"}, {"long", "OpTypeInt 64 1"},
		{"ulong", "OpTypeInt 64 0"},
	};
	size_t i;

	for (i = 0; i < sizeof t / sizeof t[0]; i++)
		if (!strcmp(b, t[i].n)) {
			*decl = t[i].d;
			return 1;
		}
	return 0;
}

/* declare a type by its glslang name; 0 if the name says nothing */
static int
ensure_type(const char *nm, struct sbuf *out)
{
	const char *d;
	char el[48];

	if (defined_name(nm) || (out->p && strstr(out->p, nm) &&
	    strstr(out->p, nm)[strlen(nm)] == ' '))
		return 1;
	if (nm[0] != '%')
		return 0;
	if (is_scalar_name(nm + 1, &d)) {
		sb_printf(out, "%s = %s\n", nm, d);
		return 1;
	}
	if (nm[1] == 'v' && nm[2] >= '2' && nm[2] <= '4' &&
	    is_scalar_name(nm + 3, &d)) {
		snprintf(el, sizeof el, "%%%s", nm + 3);
		ensure_type(el, out);
		sb_printf(out, "%s = OpTypeVector %s %c\n", nm, el, nm[2]);
		return 1;
	}
	if (!strncmp(nm, "%_ptr_", 6)) {
		const char *u = strchr(nm + 6, '_');
		char sc[24];
		if (u == NULL || u - (nm + 6) >= (int)sizeof sc)
			return 0;
		snprintf(sc, sizeof sc, "%.*s", (int)(u - (nm + 6)), nm + 6);
		snprintf(el, sizeof el, "%%%s", u + 1);
		if (!ensure_type(el, out))
			return 0;
		sb_printf(out, "%s = OpTypePointer %s %s\n", nm, sc, el);
		return 1;
	}
	return 0;
}

/* a constant by its glslang name: %int_5 %uint_0 %float_0_5 %int_n1 */
static int
ensure_const(const char *nm, struct sbuf *out)
{
	const char *u, *d;
	char ty[40], num[48];
	size_t i, n = 0;

	if (!strcmp(nm, "%true") || !strcmp(nm, "%false")) {
		ensure_type("%bool", out);
		sb_printf(out, "%s = OpConstant%s %%bool\n", nm, nm[1] == 't' ?
		    "True" : "False");
		return 1;
	}
	if ((u = strchr(nm + 1, '_')) == NULL || u[1] == 0)
		return 0;
	snprintf(ty, sizeof ty, "%.*s", (int)(u - nm), nm);
	if (!is_scalar_name(ty + 1, &d) || !strcmp(ty, "%void") ||
	    !strcmp(ty, "%bool"))
		return 0;
	u++;
	if (*u == 'n')
		num[n++] = '-', u++;
	for (i = 0; u[i] && n + 1 < sizeof num; i++) {
		if (isdigit((unsigned char)u[i]))
			num[n++] = u[i];
		else if (u[i] == '_' && !strchr(num, '.'))
			num[n++] = '.';
		else
			return 0;
	}
	num[n] = 0;
	ensure_type(ty, out);
	sb_printf(out, "%s = OpConstant %s %s\n", nm, ty, num);
	return 1;
}

/* the type an undefined operand must have, from one use; "" if unknown */
static void
infer_use(const struct sins *I, int k, char *ty, size_t n, int *ptr)
{
	const char *op = I->op;
	const char *rt = I->res >= 0 && I->nop > 0 ? I->opnd[0] : "";
	int fl = 0;

	*ptr = 0;
	if (!strcmp(op, "OpLoad") && k == 1) {
		*ptr = 1;
		snprintf(ty, n, "%s", rt);
		return;
	}
	if (!strcmp(op, "OpStore") && k == 0) {
		*ptr = 2;   /* the type comes from the stored value */
		ty[0] = 0;
		return;
	}
	if (!strcmp(op, "OpConvertFToS") || !strcmp(op, "OpConvertFToU")) {
		snprintf(ty, n, "%s", strstr(rt, "%v") == rt ? "%v4float" : "%float");
		if (rt[0] == '%' && rt[1] == 'v')
			snprintf(ty, n, "%%v%cfloat", rt[2]);
		return;
	}
	if (!strcmp(op, "OpConvertSToF")) {
		snprintf(ty, n, "%%int");
		return;
	}
	if (!strcmp(op, "OpConvertUToF")) {
		snprintf(ty, n, "%%uint");
		return;
	}
	if (!strcmp(op, "OpBranchConditional") || (!strcmp(op, "OpSelect") &&
	    k == 1) || !strncmp(op, "OpLogical", 9)) {
		snprintf(ty, n, "%%bool");
		return;
	}
	if (!strcmp(op, "OpSwitch")) {
		snprintf(ty, n, "%%int");
		return;
	}
	if (strstr(op, "Equal") || strstr(op, "Less") || strstr(op, "Greater")) {
		/* a comparison: the other operand's type, else by the opcode */
		int other = k == 2 ? 3 : 2, j;
		if (other < I->nop)
			for (j = 0; j < V.nin; j++)
				if (V.in[j].res >= 0 && V.in[j].nop > 0 &&
				    !strcmp(V.id[V.in[j].res].name, I->opnd[other])) {
					snprintf(ty, n, "%s", V.in[j].opnd[0]);
					return;
				}
		fl = op[2] == 'F';
		snprintf(ty, n, fl ? "%%float" : op[2] == 'U' ? "%%uint" : "%%int");
		return;
	}
	if (!strcmp(op, "OpBitcast") || !strcmp(op, "OpCompositeExtract") ||
	    !strcmp(op, "OpAccessChain")) {
		ty[0] = 0;
		return;
	}
	/* arithmetic, OpExtInst, OpSelect values: the result's type */
	snprintf(ty, n, "%s", rt);
}

static char *
complete(const char *src, struct sbuf *said)
{
	struct sbuf decl = {0}, body = {0}, out = {0};
	int k, j, hasfn = 0, hasentry = 0, hasglsl = 0, needglsl = 0;
	int haslabel = 0, last_term = 0, hasmem = 0, hascap = 0;

	ninps = 0;
	for (k = 0; k < V.nin; k++) {
		const char *op = V.in[k].op;
		hasfn |= !strcmp(op, "OpFunction");
		hasentry |= !strcmp(op, "OpEntryPoint");
		hasglsl |= !strcmp(op, "OpExtInstImport");
		needglsl |= !strcmp(op, "OpExtInst");
		hasmem |= !strcmp(op, "OpMemoryModel");
		hascap |= !strcmp(op, "OpCapability");
	}
	/* undefined operands */
	for (k = 0; k < V.nin; k++) {
		struct sins *I = &V.in[k];
		for (j = 0; j < I->nop; j++) {
			const char *nm = I->opnd[j];
			char ty[40];
			int ptr, q;
			if (nm[0] != '%' || defined_name(nm))
				continue;
			if (!strcmp(I->op, "OpExtInst") && j == 1)
				continue;   /* the import: added below */
			if (!strcmp(I->op, "OpName") || !strcmp(I->op, "OpDecorate") ||
			    !strcmp(I->op, "OpEntryPoint"))
				continue;
			if (ensure_type(nm, &decl) || ensure_const(nm, &decl))
				continue;
			if (!strcmp(I->op, "OpBranch") || !strcmp(I->op,
			    "OpBranchConditional") || !strcmp(I->op, "OpSelectionMerge") ||
			    !strcmp(I->op, "OpLoopMerge") || !strcmp(I->op, "OpPhi"))
				if (!(!strcmp(I->op, "OpBranchConditional") && j == 0) &&
				    !(!strcmp(I->op, "OpPhi") && j % 2 == 1))
					fail("SPIR-V: the snippet branches to %s, which it "
					    "does not contain", nm);
			for (q = 0; q < ninps; q++)
				if (!strcmp(inps[q].name, nm))
					break;
			if (q < ninps) {
				if (!inps[q].ty[0]) {
					infer_use(I, j, ty, sizeof ty, &ptr);
					snprintf(inps[q].ty, sizeof inps[q].ty, "%s", ty);
				}
				continue;
			}
			if (ninps == MAXIN)
				fail("SPIR-V: more than %d undefined operands", MAXIN);
			memset(&inps[ninps], 0, sizeof inps[ninps]);
			snprintf(inps[ninps].name, sizeof inps[ninps].name, "%s", nm);
			infer_use(I, j, ty, sizeof ty, &ptr);
			if (ptr == 2 && I->nop > 1) {
				/* a store's target: typed by the value stored */
				int v;
				for (v = 0; v < V.nin; v++)
					if (V.in[v].res >= 0 && V.in[v].nop > 0 &&
					    !strcmp(V.id[V.in[v].res].name, I->opnd[1]))
						snprintf(ty, sizeof ty, "%s", V.in[v].opnd[0]);
			}
			snprintf(inps[ninps].ty, sizeof inps[ninps].ty, "%s", ty);
			inps[ninps].var = ptr;
			ninps++;
		}
	}
	/* the types the inputs need, asking for the ones nothing tells */
	for (k = 0; k < ninps; k++) {
		if (!inps[k].ty[0] || !ensure_type(inps[k].ty, &decl)) {
			static const char *const tys[] = {"float (f32)", "int (i32)",
			    "uint (u32)", "bool"};
			static const char *const tn[] = {"%float", "%int", "%uint",
			    "%bool"};
			char what[128];
			int c = 0;
			snprintf(what, sizeof what, "SPIR-V: %s is not defined and its "
			    "use does not say its type; which", inps[k].name);
			if (nlp_can_ask()) {
				c = nlp_ask_choice(what, tys, 4, 0);
				if (nlp_asked)
					return NULL;
			} else
				sb_printf(said, "  note: %s: type unknown, float assumed\n",
				    inps[k].name);
			snprintf(inps[k].ty, sizeof inps[k].ty, "%s", tn[c]);
			ensure_type(inps[k].ty, &decl);
		}
	}
	/* values: given in the command, else asked, else random per lane */
	for (k = 0; k < ninps; k++) {
		static const char *const md[] = {"random per lane", "the lane index",
		    "0 in every lane", "1 in every lane"};
		char what[160];
		int c = 0, q;
		if (inps[k].var == 2) {   /* only stored to: an output */
			sb_printf(said, "  output %s: %s (stored to; its values are "
			    "listed after the run)\n", inps[k].name, inps[k].ty + 1);
			continue;
		}
		for (q = 0; q < nspv_set; q++)
			if (!strcmp(spv_set[q].name, inps[k].name + 1))
				break;
		if (q < nspv_set) {
			sb_printf(said, "  input %s: %s, given\n", inps[k].name,
			    inps[k].ty + 1);
			inps[k].mode = spv_set[q].mode;
			inps[k].v = spv_set[q].v;
			continue;
		}
		snprintf(what, sizeof what, "SPIR-V: %s (%s%s) is not defined in the "
		    "snippet; its value", inps[k].name, inps[k].ty + 1,
		    inps[k].var ? ", through a pointer" : "");
		if (values_unneeded) {
			sb_printf(said, "  input %s: %s\n", inps[k].name,
			    inps[k].ty + 1);
			continue;
		}
		if (nlp_can_ask()) {
			c = nlp_ask_choice(what, md, 4, 0);
			if (nlp_asked)
				return NULL;
		}
		inps[k].mode = c;
		sb_printf(said, "  input %s: %s, %s (say \"%s = 2.5\" in the request "
		    "to set it)\n", inps[k].name, inps[k].ty + 1, md[c], inps[k].name);
	}
	/* the module around the snippet */
	if (!hascap)
		sb_puts(&out, "OpCapability Shader\n");
	if (needglsl && !hasglsl) {
		/* the import, under the id the snippet uses for it */
		for (k = 0; k < V.nin; k++)
			if (!strcmp(V.in[k].op, "OpExtInst") && V.in[k].nop > 1) {
				sb_printf(&out, "%s = OpExtInstImport \"GLSL.std.450\"\n",
				    V.in[k].opnd[1]);
				break;
			}
	}
	if (!hasmem)
		sb_puts(&out, "OpMemoryModel Logical GLSL450\n");
	if (!hasfn)
		sb_puts(&out, "OpEntryPoint GLCompute %main \"main\"\n");
	else if (!hasentry)
		for (k = 0; k < V.nin; k++)
			if (!strcmp(V.in[k].op, "OpFunction") && V.in[k].res >= 0) {
				sb_printf(&out, "OpEntryPoint GLCompute %s \"main\"\n",
				    V.id[V.in[k].res].name);
				break;
			}
	sb_puts(&out, decl.p ? decl.p : "");
	for (k = 0; k < ninps; k++)
		if (inps[k].var) {
			char pt[64];
			snprintf(pt, sizeof pt, "%%_ptr_Private_%s", inps[k].ty + 1);
			ensure_type(pt, &out);
			sb_printf(&out, "%s = OpVariable %s Private\n", inps[k].name, pt);
		}
	/* module-level declarations first, then the code */
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int declop = !strncmp(I->op, "OpType", 6) ||
		    !strncmp(I->op, "OpConstant", 10) ||
		    !strncmp(I->op, "OpSpecConstant", 14) ||
		    !strcmp(I->op, "OpCapability") || !strcmp(I->op, "OpExtension") ||
		    !strcmp(I->op, "OpExtInstImport") ||
		    !strcmp(I->op, "OpMemoryModel") || !strcmp(I->op, "OpEntryPoint") ||
		    !strcmp(I->op, "OpExecutionMode") || !strcmp(I->op, "OpName") ||
		    !strcmp(I->op, "OpMemberName") || !strcmp(I->op, "OpDecorate") ||
		    !strcmp(I->op, "OpMemberDecorate") || !strcmp(I->op, "OpSource") ||
		    (!strcmp(I->op, "OpVariable") && !hasfn && I->nop > 1 &&
		    strcmp(I->opnd[1], "Function"));
		char *t = I->text;
		if (declop)
			sb_printf(&out, "%s\n", t);
		else {
			if (!strcmp(I->op, "OpLabel"))
				haslabel = 1;
			last_term = !strcmp(I->op, "OpReturn") ||
			    !strcmp(I->op, "OpReturnValue") || !strcmp(I->op, "OpKill") ||
			    !strcmp(I->op, "OpBranch") || !strcmp(I->op, "OpFunctionEnd") ||
			    !strcmp(I->op, "OpUnreachable");
			sb_printf(&body, "%s\n", t);
		}
	}
	if (!hasfn) {
		if (!defined_name("%void"))
			ensure_type("%void", &out);
		sb_puts(&out, "%main_fn = OpTypeFunction %void\n"
		    "%main = OpFunction %void None %main_fn\n");
		if (!haslabel)
			sb_puts(&out, "%entry = OpLabel\n");
		sb_puts(&out, body.p ? body.p : "");
		if (!last_term)
			sb_puts(&out, "OpReturn\n");
		sb_puts(&out, "OpFunctionEnd\n");
	} else {
		sb_puts(&out, body.p ? body.p : "");
		if (!last_term)
			sb_puts(&out, "OpReturn\nOpFunctionEnd\n");
	}
	free(decl.p);
	free(body.p);
	(void)src;
	return out.p;
}

/* ---- types, constants, variables ---- */

static int
slots_of(int t)
{
	struct sty *y = T(t);
	int i, s = 0;

	switch (y->k) {
	case S_BOOL: case S_INT: case S_FLOAT:
		return 1;
	case S_VEC:
		return y->n;
	case S_ARRAY:
		return y->n * slots_of(y->e);
	case S_STRUCT:
		for (i = 0; i < y->nm; i++)
			s += slots_of(y->m[i]);
		return s;
	}
	return 1;
}

/* the scalar type of each slot of a type, from slot `at` */
static void
fill_slotty(int t, int *out, int at)
{
	struct sty *y = T(t);
	int i;

	switch (y->k) {
	case S_VEC:
		for (i = 0; i < y->n; i++)
			out[at + i] = y->e;
		return;
	case S_ARRAY:
		for (i = 0; i < y->n; i++)
			fill_slotty(y->e, out, at + i * slots_of(y->e));
		return;
	case S_STRUCT:
		for (i = 0; i < y->nm; i++) {
			fill_slotty(y->m[i], out, at);
			at += slots_of(y->m[i]);
		}
		return;
	}
	out[at] = t;
}

static int
ncomp(int t)
{
	struct sty *y = T(t);

	return y->k == S_VEC ? y->n : 1;
}

static int
scalar_of(int t)
{
	struct sty *y = T(t);

	return y->k == S_VEC ? y->e : t;
}

static uint64_t
wmask(int w)
{
	return w >= 64 ? ~0ULL : (1ULL << w) - 1;
}

static double
getf(int st, uint64_t b)
{
	return frombits(T(st)->w, b);
}

static uint64_t
putf(int st, double v)
{
	return fbits(T(st)->w, v) & wmask(T(st)->w);
}

static int64_t
geti(int st, uint64_t b)
{
	struct sty *y = T(st);
	uint64_t m = wmask(y->w);

	if (y->k == S_BOOL)
		return b != 0;
	b &= m;
	if (y->sign && y->w < 64 && (b >> (y->w - 1) & 1))
		return (int64_t)(b | ~m);
	return (int64_t)b;
}

static uint64_t
puti(int st, int64_t v)
{
	struct sty *y = T(st);

	if (y->k == S_BOOL)
		return v != 0;
	return (uint64_t)v & wmask(y->w);
}

static uint64_t
parse_literal(int st, const char *s)
{
	if (T(st)->k == S_FLOAT)
		return putf(st, strtod(s, NULL));
	if (s[0] == '-')
		return puti(st, strtoll(s, NULL, 0));
	return puti(st, (int64_t)strtoull(s, NULL, 0));
}

static uint64_t
rnd(void)
{
	V.rng ^= V.rng << 13;
	V.rng ^= V.rng >> 7;
	V.rng ^= V.rng << 17;
	return V.rng;
}

static uint64_t
random_scalar(int st)
{
	if (T(st)->k == S_FLOAT)
		return putf(st, (double)(int)(rnd() % 401 - 200) / 100.0);
	if (T(st)->k == S_BOOL)
		return rnd() & 1;
	return puti(st, (int64_t)(rnd() % 10));
}

static int
storage(const char *s)
{
	if (!strcmp(s, "Function"))
		return SC_FUNCTION;
	if (!strcmp(s, "Private"))
		return SC_PRIVATE;
	if (!strcmp(s, "Input"))
		return SC_INPUT;
	if (!strcmp(s, "Output"))
		return SC_OUTPUT;
	if (!strcmp(s, "Workgroup"))
		return SC_WG;
	return SC_SHARED;   /* Uniform, StorageBuffer, PushConstant, ... */
}

static int
shared_sc(int sc)
{
	return sc == SC_SHARED || sc == SC_WG;
}

enum { B_FRAGCOORD = 1, B_LOCALID, B_GLOBALID, B_LOCALINDEX, B_SUBLANE,
	B_WGID, B_NUMWG, B_FRONT, B_HELPER, B_VERTEX };

static int
builtin_code(const char *s)
{
	static const struct { const char *n; int c; } t[] = {
		{"FragCoord", B_FRAGCOORD}, {"LocalInvocationId", B_LOCALID},
		{"GlobalInvocationId", B_GLOBALID},
		{"LocalInvocationIndex", B_LOCALINDEX},
		{"SubgroupLocalInvocationId", B_SUBLANE},
		{"WorkgroupId", B_WGID}, {"NumWorkgroups", B_NUMWG},
		{"FrontFacing", B_FRONT}, {"HelperInvocation", B_HELPER},
		{"VertexIndex", B_VERTEX}, {"InstanceIndex", B_VERTEX},
		{"VertexId", B_VERTEX}, {"InstanceId", B_VERTEX},
	};
	size_t i;

	for (i = 0; i < sizeof t / sizeof t[0]; i++)
		if (!strcmp(s, t[i].n))
			return t[i].c;
	return 0;
}

static int cols;

static void
init_var(int v, int lane, uint64_t *mem)
{
	struct sid *d = &V.id[v];
	int s;

	for (s = 0; s < d->nslots; s++) {
		int st = d->slotty[s];
		uint64_t b = 0;
		switch (d->builtin) {
		case B_FRAGCOORD:
			b = putf(st, s == 0 ? lane % cols + 0.5 : s == 1 ?
			    lane / cols + 0.5 : s == 2 ? 0.5 : 1.0);
			break;
		case B_LOCALID: case B_GLOBALID:
			b = puti(st, s == 0 ? lane : 0);
			break;
		case B_LOCALINDEX: case B_SUBLANE: case B_VERTEX:
			b = puti(st, lane);
			break;
		case B_NUMWG:
			b = puti(st, 1);
			break;
		case B_FRONT:
			b = 1;
			break;
		case B_WGID: case B_HELPER:
			b = 0;
			break;
		default:
			if (d->sc == SC_SHARED || d->sc == SC_INPUT)
				b = random_scalar(st);
		}
		mem[s] = b;
	}
}

static uint64_t alu(const char *op, int rt, int at, uint64_t a, uint64_t b,
    uint64_t c, int *ok);

/*
 * OpSpecConstantOp: an instruction on constants, folded once (the spec
 * constants keep their default values)
 */
static void
spec_op(const struct sins *I, struct sid *d)
{
	char op[48];
	int rt = d->type, n, i, ok = 1;

	if (I->nop < 3)
		die("short OpSpecConstantOp: \"%s\"", I->text);
	snprintf(op, sizeof op, "Op%s", I->opnd[1]);
	n = T(rt)->k == S_VEC ? T(rt)->n : 1;
	d->nc = n;
	if (!strcmp(op, "OpCompositeExtract")) {
		struct sid *c = &V.id[opid(I, 2)];
		int k = I->nop > 3 ? atoi(I->opnd[3]) : 0;
		d->c[0] = k >= 0 && k < MAXCOMP ? c->c[k] : 0;
		return;
	}
	if (!strcmp(op, "OpSelect")) {
		struct sid *c = &V.id[opid(I, 2)], *a = &V.id[opid(I, 3)],
		    *b = &V.id[opid(I, 4)];
		for (i = 0; i < n; i++)
			d->c[i] = (c->nc > 1 ? c->c[i] : c->c[0]) ? a->c[i] : b->c[i];
		return;
	}
	{
		struct sid *a = &V.id[opid(I, 2)];
		struct sid *b = I->nop > 3 ? &V.id[opid(I, 3)] : a;
		struct sid *c = I->nop > 4 ? &V.id[opid(I, 4)] : a;
		int at = a->type >= 0 ? (T(a->type)->k == S_VEC ? T(a->type)->e :
		    a->type) : rt;
		int st = T(rt)->k == S_VEC ? T(rt)->e : rt;
		for (i = 0; i < n; i++) {
			d->c[i] = alu(op, st, at, a->c[a->nc > 1 ? i : 0],
			    b->c[b->nc > 1 ? i : 0], c->c[c->nc > 1 ? i : 0], &ok);
			if (!ok)
				die("unsupported OpSpecConstantOp: \"%s\"", I->text);
		}
	}
}

static void
declare(void)
{
	int k, i;

	for (k = 0; k < V.nin; k++) {
		struct sins *I = &V.in[k];
		struct sid *d = I->res >= 0 ? &V.id[I->res] : NULL;
		if (!strcmp(I->op, "OpExtInstImport")) {
			d->kind = K_EXT;
			if (I->nop >= 1 && strstr(I->opnd[0], "GLSL.std.450"))
				V.glsl = I->res;
		} else if (!strcmp(I->op, "OpName") && I->nop >= 2) {
			struct sid *n = &V.id[opid(I, 0)];
			if (strlen(I->opnd[1]) > 2 && I->opnd[1][1] != '"')
				snprintf(n->name, sizeof n->name, "%.*s",
				    (int)strlen(I->opnd[1]) - 2, I->opnd[1] + 1);
		} else if (!strcmp(I->op, "OpDecorate") && I->nop >= 3 &&
		    !strcmp(I->opnd[1], "BuiltIn")) {
			V.id[opid(I, 0)].builtin = builtin_code(I->opnd[2]);
		} else if (!strncmp(I->op, "OpType", 6)) {
			struct sty *y = &d->t;
			d->kind = K_TYPE;
			memset(y, 0, sizeof *y);
			if (!strcmp(I->op, "OpTypeVoid"))
				y->k = S_VOID;
			else if (!strcmp(I->op, "OpTypeBool"))
				y->k = S_BOOL, y->w = 1;
			else if (!strcmp(I->op, "OpTypeInt"))
				y->k = S_INT, y->w = atoi(I->opnd[0]),
				    y->sign = atoi(I->opnd[1]);
			else if (!strcmp(I->op, "OpTypeFloat"))
				y->k = S_FLOAT, y->w = atoi(I->opnd[0]);
			else if (!strcmp(I->op, "OpTypeVector"))
				y->k = S_VEC, y->e = opid(I, 0), y->n = atoi(I->opnd[1]);
			else if (!strcmp(I->op, "OpTypeArray")) {
				int len = opid(I, 1);
				y->k = S_ARRAY;
				y->e = opid(I, 0);
				y->n = V.id[len].kind == K_CONST ?
				    (int)geti(V.id[len].type, V.id[len].c[0]) : 1;
			} else if (!strcmp(I->op, "OpTypeRuntimeArray"))
				y->k = S_ARRAY, y->e = opid(I, 0), y->n = 16;
			else if (!strcmp(I->op, "OpTypeStruct")) {
				y->k = S_STRUCT;
				for (i = 0; i < I->nop && i < 48; i++)
					y->m[y->nm++] = opid(I, i);
			} else if (!strcmp(I->op, "OpTypePointer"))
				y->k = S_PTR, y->sc = storage(I->opnd[0]),
				    y->e = opid(I, 1);
			else if (!strcmp(I->op, "OpTypeFunction"))
				y->k = S_FN;
			else
				y->k = S_OTHER;
			if (y->n > 4096)
				y->n = 4096;
		} else if (!strncmp(I->op, "OpConstant", 10) ||
		    !strncmp(I->op, "OpSpecConstant", 14)) {
			const char *o = I->op + (I->op[2] == 'S' ? 14 : 10);
			d->kind = K_CONST;
			d->spec = I->op[2] == 'S';
			d->type = opid(I, 0);
			if (!strcmp(o, "True"))
				d->c[0] = 1, d->nc = 1;
			else if (!strcmp(o, "False") || !strcmp(o, "Null")) {
				d->nc = ncomp(d->type);
				for (i = 0; i < d->nc; i++)
					d->c[i] = 0;
			} else if (!strcmp(o, "Composite")) {
				/* every component of each constituent, in order */
				for (i = 1; i < I->nop; i++) {
					struct sid *e = &V.id[opid(I, i)];
					int j;
					for (j = 0; j < (e->nc ? e->nc : 1) && d->nc < MAXCOMP;
					    j++)
						d->c[d->nc++] = e->c[j];
				}
			} else if (!strcmp(o, "Op"))
				spec_op(I, d);
			else if (o[0] == 0)
				d->c[0] = parse_literal(d->type, I->opnd[1]), d->nc = 1;
		} else if (!strcmp(I->op, "OpUndef")) {
			d->kind = K_CONST;	/* any value will do: zero */
			d->type = opid(I, 0);
			d->nc = T(d->type)->k == S_VEC ? T(d->type)->n : 1;
		} else if (!strcmp(I->op, "OpVariable")) {
			int pt = opid(I, 0);
			d->kind = K_VAR;
			d->type = pt;
			d->sc = storage(I->opnd[1]);
			d->nslots = slots_of(T(pt)->e);
			d->slotty = xcalloc((size_t)d->nslots, sizeof *d->slotty);
			fill_slotty(T(pt)->e, d->slotty, 0);
		} else if (!strcmp(I->op, "OpFunction")) {
			d->kind = K_FUNC;
			d->func_at = k;
		} else if (!strcmp(I->op, "OpLabel")) {
			d->kind = K_LABEL;
			d->label_at = k;
		}
	}
	/* memory, after every decoration is known */
	for (i = 0; i < V.nid; i++) {
		struct sid *d = &V.id[i];
		int l;
		if (d->kind != K_VAR)
			continue;
		if (shared_sc(d->sc)) {
			d->mem = xcalloc((size_t)d->nslots, sizeof *d->mem);
			if (d->sc == SC_SHARED)
				init_var(i, 0, d->mem);
		} else {
			d->mem = xcalloc((size_t)d->nslots * V.lanes, sizeof *d->mem);
			for (l = 0; l < V.lanes; l++)
				init_var(i, l, d->mem + (size_t)l * d->nslots);
		}
	}
}

/* ---- values ---- */

static struct lval *
LV(int id, int lane)
{
	if (id >= V.lvcap) {
		int n = V.idcap > id ? V.idcap : id + 1;
		V.lv = xrealloc(V.lv, (size_t)n * NL * sizeof *V.lv);
		memset(V.lv + (size_t)V.lvcap * NL, 0,
		    (size_t)(n - V.lvcap) * NL * sizeof *V.lv);
		V.lvcap = n;
	}
	return &V.lv[(size_t)id * NL + lane];
}

/* an operand's value in a lane: constant, variable address, or computed */
static struct lval
val(int id, int lane)
{
	struct lval r;
	struct sid *d = &V.id[id];
	int i;

	memset(&r, 0, sizeof r);
	if (d->kind == K_CONST) {
		for (i = 0; i < d->nc; i++)
			r.b[i] = d->c[i];
		return r;
	}
	if (d->kind == K_VAR) {
		r.var = id;
		r.off = 0;
		return r;
	}
	return *LV(id, lane);
}

static int
type_of(int id)
{
	return V.id[id].type;
}

static uint64_t *
slot(int var, int lane, int off)
{
	struct sid *d = &V.id[var];

	if (off < 0 || off >= d->nslots) {
		/*
		 * outside the variable: like robustBufferAccess, a read gives
		 * 0 and a write goes nowhere; counted for a note
		 */
		static uint64_t scratch;
		if (V.oob++ == 0)
			snprintf(V.oobvar, sizeof V.oobvar, "%s", d->name);
		scratch = 0;
		return &scratch;
	}
	if (shared_sc(d->sc))
		return &d->mem[off];
	return &d->mem[(size_t)lane * d->nslots + off];
}

static void
fmt(char *buf, size_t n, int t, const struct lval *v)
{
	struct sty *y = T(t);
	int i, at = 0, st = scalar_of(t), nc = ncomp(t);

	if (y->k == S_PTR) {
		snprintf(buf, n, "&%s[%d]", V.id[v->var].name, v->off);
		return;
	}
	if (y->k == S_STRUCT || y->k == S_ARRAY) {
		snprintf(buf, n, "{...}");
		return;
	}
	if (nc > 1)
		at += snprintf(buf + at, n - (size_t)at, "(");
	for (i = 0; i < nc && (size_t)at < n; i++) {
		if (i)
			at += snprintf(buf + at, n - (size_t)at, ",");
		if (T(st)->k == S_FLOAT)
			at += snprintf(buf + at, n - (size_t)at, "%.5g",
			    getf(st, v->b[i]));
		else if (T(st)->k == S_BOOL)
			at += snprintf(buf + at, n - (size_t)at, "%s",
			    v->b[i] ? "true" : "false");
		else if (T(st)->sign)
			at += snprintf(buf + at, n - (size_t)at, "%lld",
			    (long long)geti(st, v->b[i]));
		else
			at += snprintf(buf + at, n - (size_t)at, "%llu",
			    (unsigned long long)(v->b[i] & wmask(T(st)->w)));
	}
	if (nc > 1 && (size_t)at < n)
		snprintf(buf + at, n - (size_t)at, ")");
}

/* components of an operand (a constant or a computed value) */
static int
vcomp(int id)
{
	int t = V.id[id].type;

	return t >= 0 && V.id[t].kind == K_TYPE && T(t)->k == S_VEC ?
	    T(t)->n : 1;
}

/* ---- costs ---- */

static int
cost_class(const char *op, const char *ext, int st, double *mul)
{
	int fl = st >= 0 && T(st)->k == S_FLOAT;
	*mul = 1;
	if (ext) {
		if (!strcmp(ext, "Sqrt"))
			return C_FSQRT;
		if (!strcmp(ext, "Pow"))
			return C_FPOW;
		if (strstr("InverseSqrt Exp Log Exp2 Log2 Sin Cos Tan Asin Acos "
		    "Atan Atan2 Sinh Cosh Tanh", ext))
			return C_FTRANS;
		if (strstr("Fma FMix", ext))
			return C_FMUL;
		if (strstr("Floor Ceil Round RoundEven Trunc Fract", ext))
			return C_CONV;
		if (!strcmp(ext, "Length") || !strcmp(ext, "Normalize") ||
		    !strcmp(ext, "Distance")) {
			*mul = 3;
			return C_FMUL;
		}
		if (strstr(ext, "Clamp"))
			*mul = 2;	/* a min and a max */
		return C_MINMAX;   /* min, max, clamp, abs, sign, step */
	}
	if (!strcmp(op, "OpIAdd") || !strcmp(op, "OpISub") ||
	    !strcmp(op, "OpSNegate"))
		return C_IADD;
	if (!strcmp(op, "OpIMul"))
		return C_IMUL;
	if (!strcmp(op, "OpSDiv") || !strcmp(op, "OpUDiv") ||
	    !strcmp(op, "OpSRem") || !strcmp(op, "OpSMod") ||
	    !strcmp(op, "OpUMod"))
		return C_IDIV;
	if (!strcmp(op, "OpFAdd") || !strcmp(op, "OpFSub") ||
	    !strcmp(op, "OpFNegate"))
		return C_FADD;
	if (!strcmp(op, "OpFMul") || !strcmp(op, "OpVectorTimesScalar"))
		return C_FMUL;
	if (!strcmp(op, "OpDot")) {
		*mul = 1;
		return C_FMUL;
	}
	if (!strcmp(op, "OpFDiv") || !strcmp(op, "OpFRem") ||
	    !strcmp(op, "OpFMod"))
		return C_FDIV;
	if (!strncmp(op, "OpShift", 7))
		return C_SHIFT;
	if (!strncmp(op, "OpBitwise", 9) || !strcmp(op, "OpNot") ||
	    !strncmp(op, "OpLogical", 9))
		return C_LOGIC;
	if (!strcmp(op, "OpBitCount") || !strcmp(op, "OpBitReverse"))
		return C_BITCNT;
	if (!strncmp(op, "OpBitField", 10))
		return C_SHIFT;
	if (strstr(op, "Equal") || strstr(op, "Less") || strstr(op, "Greater") ||
	    !strcmp(op, "OpIsNan") || !strcmp(op, "OpIsInf"))
		return fl ? C_CMP : C_CMP;
	if (!strcmp(op, "OpSelect"))
		return C_SELECT;
	if (!strncmp(op, "OpConvert", 9) || !strcmp(op, "OpFConvert") ||
	    !strcmp(op, "OpSConvert") || !strcmp(op, "OpUConvert"))
		return C_CONV;
	return -1;
}

/* emit one executed instruction to the animation */
static int
emit(const struct sins *I, int kind, int cls, double cost, uint32_t mask)
{
	char dst[16] = "";

	if (I->res >= 0)
		snprintf(dst, sizeof dst, "%.15s", V.id[I->res].name);
	return xr_add(kind, I->text, cls, cost, mask, dst);
}

static void
emit_vals(int k, const struct sins *I, uint32_t mask)
{
	int l;
	char buf[64];

	if (I->res < 0 || V.id[I->res].type < 0)
		return;
	for (l = 0; l < V.lanes; l++)
		if (mask >> l & 1) {
			fmt(buf, sizeof buf, V.id[I->res].type, LV(I->res, l));
			xr_val(k, l, buf);
		}
}

/* ---- arithmetic ---- */

static uint64_t
alu(const char *op, int rt, int at, uint64_t a, uint64_t b, uint64_t c,
    int *ok)
{
	int fl = T(at)->k == S_FLOAT;
	double fa = fl ? getf(at, a) : 0, fb = fl ? getf(at, b) : 0;
	int64_t ia = fl ? 0 : geti(at, a), ib = fl ? 0 : geti(at, b);
	uint64_t ua = a & wmask(T(at)->w), ub = b & wmask(T(at)->w);

#define OP(n) (!strcmp(op, n))
	if (OP("OpIAdd")) return puti(rt, ia + ib);
	if (OP("OpISub")) return puti(rt, ia - ib);
	if (OP("OpIMul")) return puti(rt, (int64_t)((uint64_t)ia * (uint64_t)ib));
	if (OP("OpSNegate")) return puti(rt, -ia);
	if (OP("OpUDiv")) return ub ? puti(rt, (int64_t)(ua / ub)) : 0;
	if (OP("OpSDiv")) return ib ? puti(rt, ia / ib) : 0;
	if (OP("OpUMod")) return ub ? puti(rt, (int64_t)(ua % ub)) : 0;
	if (OP("OpSRem")) return ib ? puti(rt, ia % ib) : 0;
	if (OP("OpSMod")) {
		int64_t r;
		if (!ib)
			return 0;
		r = ia % ib;
		if (r && ((r < 0) != (ib < 0)))
			r += ib;
		return puti(rt, r);
	}
	if (OP("OpFAdd")) return putf(rt, fa + fb);
	if (OP("OpFSub")) return putf(rt, fa - fb);
	if (OP("OpFMul")) return putf(rt, fa * fb);
	if (OP("OpFDiv")) return putf(rt, fa / fb);
	if (OP("OpFRem")) return putf(rt, fmod(fa, fb));
	if (OP("OpFMod")) return putf(rt, fa - fb * floor(fa / fb));
	if (OP("OpFNegate")) return putf(rt, -fa);
	if (OP("OpShiftLeftLogical")) return puti(rt, (int64_t)(ua << (ub & 63)));
	if (OP("OpShiftRightLogical")) return puti(rt, (int64_t)(ua >> (ub & 63)));
	if (OP("OpShiftRightArithmetic")) return puti(rt, ia >> (ub & 63));
	if (OP("OpBitwiseAnd")) return puti(rt, (int64_t)(ua & ub));
	if (OP("OpBitwiseOr")) return puti(rt, (int64_t)(ua | ub));
	if (OP("OpBitwiseXor")) return puti(rt, (int64_t)(ua ^ ub));
	if (OP("OpNot")) return puti(rt, (int64_t)~ua);
	if (OP("OpBitFieldUExtract") || OP("OpBitFieldSExtract")) {
		int off = (int)(b & 63), cnt = (int)(c & 127);
		uint64_t v;
		if (cnt == 0)
			return 0;
		if (cnt > 64 - off)
			cnt = 64 - off;
		v = (ua >> off) & wmask(cnt);
		if (OP("OpBitFieldSExtract") && (v >> (cnt - 1) & 1))
			v |= ~wmask(cnt);
		return puti(rt, (int64_t)v);
	}
	if (OP("OpBitReverse")) {
		uint64_t r = 0;
		int k, w = T(at)->w;
		for (k = 0; k < w; k++)
			if (ua >> k & 1)
				r |= 1ULL << (w - 1 - k);
		return puti(rt, (int64_t)r);
	}
	if (OP("OpBitCount")) {
		int n = 0;
		for (; ua; ua &= ua - 1)
			n++;
		return puti(rt, n);
	}
	if (OP("OpLogicalAnd")) return (a && b);
	if (OP("OpLogicalOr")) return (a || b);
	if (OP("OpLogicalNot")) return !a;
	if (OP("OpLogicalEqual")) return !a == !b;
	if (OP("OpLogicalNotEqual")) return !a != !b;
	if (OP("OpIEqual")) return ua == ub;
	if (OP("OpINotEqual")) return ua != ub;
	if (OP("OpSLessThan")) return ia < ib;
	if (OP("OpSLessThanEqual")) return ia <= ib;
	if (OP("OpSGreaterThan")) return ia > ib;
	if (OP("OpSGreaterThanEqual")) return ia >= ib;
	if (OP("OpULessThan")) return ua < ub;
	if (OP("OpULessThanEqual")) return ua <= ub;
	if (OP("OpUGreaterThan")) return ua > ub;
	if (OP("OpUGreaterThanEqual")) return ua >= ub;
	if (OP("OpFOrdEqual")) return fa == fb;
	if (OP("OpFOrdNotEqual")) return fa != fb && !isnan(fa) && !isnan(fb);
	if (OP("OpFOrdLessThan")) return fa < fb;
	if (OP("OpFOrdLessThanEqual")) return fa <= fb;
	if (OP("OpFOrdGreaterThan")) return fa > fb;
	if (OP("OpFOrdGreaterThanEqual")) return fa >= fb;
	if (OP("OpFUnordEqual")) return fa == fb || isnan(fa) || isnan(fb);
	if (OP("OpFUnordNotEqual")) return fa != fb;
	if (OP("OpFUnordLessThan")) return !(fa >= fb);
	if (OP("OpFUnordLessThanEqual")) return !(fa > fb);
	if (OP("OpFUnordGreaterThan")) return !(fa <= fb);
	if (OP("OpFUnordGreaterThanEqual")) return !(fa < fb);
	if (OP("OpIsNan")) return isnan(fa);
	if (OP("OpIsInf")) return isinf(fa);
	if (OP("OpConvertFToS") || OP("OpConvertFToU"))
		return puti(rt, isfinite(fa) ? (int64_t)fa : 0);
	if (OP("OpConvertSToF")) return putf(rt, (double)ia);
	if (OP("OpConvertUToF")) return putf(rt, (double)ua);
	if (OP("OpFConvert")) return putf(rt, fa);
	if (OP("OpSConvert")) return puti(rt, ia);
	if (OP("OpUConvert")) return puti(rt, (int64_t)ua);
	if (OP("OpBitcast")) return a & wmask(T(rt)->w);
	if (OP("OpCopyObject")) return a;
#undef OP
	*ok = 0;
	return 0;
}

static uint64_t
glsl(const char *f, int rt, int at, uint64_t a, uint64_t b, uint64_t c,
    int *ok)
{
	int fl = T(at)->k == S_FLOAT;
	double x = fl ? getf(at, a) : 0, y = fl ? getf(at, b) : 0,
	    z = fl ? getf(at, c) : 0;
	int64_t ia = fl ? 0 : geti(at, a), ib = fl ? 0 : geti(at, b),
	    ic = fl ? 0 : geti(at, c);
	uint64_t ua = a & wmask(T(at)->w), ub = b & wmask(T(at)->w),
	    uc = c & wmask(T(at)->w);

#define F(n) (!strcmp(f, n))
	if (F("FAbs")) return putf(rt, fabs(x));
	if (F("SAbs")) return puti(rt, ia < 0 ? -ia : ia);
	if (F("FSign")) return putf(rt, (x > 0) - (x < 0));
	if (F("SSign")) return puti(rt, (ia > 0) - (ia < 0));
	if (F("Floor")) return putf(rt, floor(x));
	if (F("Ceil")) return putf(rt, ceil(x));
	if (F("Trunc")) return putf(rt, trunc(x));
	if (F("Round")) return putf(rt, round(x));
	if (F("RoundEven")) return putf(rt, nearbyint(x));
	if (F("Fract")) return putf(rt, x - floor(x));
	if (F("Sqrt")) return putf(rt, sqrt(x));
	if (F("InverseSqrt")) return putf(rt, 1 / sqrt(x));
	if (F("Exp")) return putf(rt, exp(x));
	if (F("Exp2")) return putf(rt, exp2(x));
	if (F("Log")) return putf(rt, log(x));
	if (F("Log2")) return putf(rt, log2(x));
	if (F("Sin")) return putf(rt, sin(x));
	if (F("Cos")) return putf(rt, cos(x));
	if (F("Tan")) return putf(rt, tan(x));
	if (F("Asin")) return putf(rt, asin(x));
	if (F("Acos")) return putf(rt, acos(x));
	if (F("Atan")) return putf(rt, atan(x));
	if (F("Atan2")) return putf(rt, atan2(x, y));
	if (F("Pow")) return putf(rt, pow(x, y));
	if (F("FMin") || F("NMin")) return putf(rt, fmin(x, y));
	if (F("FMax") || F("NMax")) return putf(rt, fmax(x, y));
	if (F("SMin")) return puti(rt, ia < ib ? ia : ib);
	if (F("SMax")) return puti(rt, ia > ib ? ia : ib);
	if (F("UMin")) return puti(rt, (int64_t)(ua < ub ? ua : ub));
	if (F("UMax")) return puti(rt, (int64_t)(ua > ub ? ua : ub));
	if (F("FClamp") || F("NClamp")) return putf(rt, fmin(fmax(x, y), z));
	if (F("SClamp")) return puti(rt, ia < ib ? ib : ia > ic ? ic : ia);
	if (F("UClamp")) return puti(rt, (int64_t)(ua < ub ? ub : ua > uc ? uc :
	    ua));
	if (F("FMix")) return putf(rt, x * (1 - z) + y * z);
	if (F("Step")) return putf(rt, y < x ? 0 : 1);
	if (F("Fma")) return putf(rt, fma(x, y, z));
	if (F("FindILsb")) {
		int k;
		for (k = 0; k < T(at)->w; k++)
			if (ua >> k & 1)
				return puti(rt, k);
		return puti(rt, -1);
	}
	if (F("FindUMsb") || F("FindSMsb")) {
		int k;
		uint64_t v = F("FindSMsb") && ia < 0 ? ~ua & wmask(T(at)->w) : ua;
		for (k = T(at)->w - 1; k >= 0; k--)
			if (v >> k & 1)
				return puti(rt, k);
		return puti(rt, -1);
	}
	if (F("Radians")) return putf(rt, x * 0.017453292519943295);
	if (F("Degrees")) return putf(rt, x * 57.29577951308232);
	if (F("Sinh")) return putf(rt, sinh(x));
	if (F("Cosh")) return putf(rt, cosh(x));
	if (F("Tanh")) return putf(rt, tanh(x));
	if (F("Asinh")) return putf(rt, asinh(x));
	if (F("Acosh")) return putf(rt, acosh(x));
	if (F("Atanh")) return putf(rt, atanh(x));
	if (F("SmoothStep")) {
		double t = (z - x) / (y - x);
		t = t < 0 ? 0 : t > 1 ? 1 : t;
		return putf(rt, t * t * (3 - 2 * t));
	}
	if (F("Ldexp")) return putf(rt, ldexp(x, (int)(int32_t)b));
	if (F("IMix")) return puti(rt, c ? ib : ia);
#undef F
	*ok = 0;
	return 0;
}

/* ---- execution ---- */

static int region(int lbl, uint32_t mask, int stop, int brk, uint32_t *arrived);

static int
label_target(const struct sins *I, int k)
{
	return opid(I, k);
}

static void
enter_block(int lbl, uint32_t mask)
{
	int l;

	for (l = 0; l < V.lanes; l++)
		if (mask >> l & 1) {
			V.prevblk[l] = V.curblk[l];
			V.curblk[l] = lbl;
		}
}

/* execute one non-terminator instruction for the lanes in mask */
static void
exec_one(const struct sins *I, uint32_t mask)
{
	int l, i, k, cls = -1, rt = -1, st = -1, nc = 1, ok = 1;
	double mul = 1, cost = 0;
	const char *ext = NULL;
	int first = 1;

	if (++V.steps > 4000000)
		fail("SPIR-V: more than 4000000 steps (an endless loop?)");
	if (!strcmp(I->op, "OpLine") || !strcmp(I->op, "OpNoLine") ||
	    !strcmp(I->op, "OpSelectionMerge") || !strcmp(I->op, "OpLoopMerge") ||
	    !strcmp(I->op, "OpVariable") || !strcmp(I->op, "OpFunctionParameter")
	    || !strcmp(I->op, "OpLabel"))
		return;
	if (I->res >= 0 && I->nop >= 1 && I->opnd[0][0] == '%' &&
	    V.id[opid(I, 0)].kind == K_TYPE) {
		rt = opid(I, 0);
		V.id[I->res].type = rt;
		V.id[I->res].kind = K_VALUE;
		if (T(rt)->k != S_PTR && T(rt)->k != S_STRUCT &&
		    T(rt)->k != S_ARRAY && T(rt)->k != S_VOID) {
			st = scalar_of(rt);
			nc = ncomp(rt);
		}
	}

	if (!strcmp(I->op, "OpUndef")) {
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1)
				memset(LV(I->res, l), 0, sizeof(struct lval));
		return;
	}
	if (!strcmp(I->op, "OpStore")) {
		int p = opid(I, 0), v = opid(I, 1), sc;
		struct lval pv = val(p, 0);
		sc = V.id[pv.var].sc;
		for (l = 0; l < V.lanes; l++) {
			struct lval a, x;
			int pt;
			if (!(mask >> l & 1))
				continue;
			a = val(p, l);
			x = val(v, l);
			pt = type_of(v) >= 0 ? type_of(v) : -1;
			for (i = 0; i < (pt >= 0 && T(pt)->k == S_VEC ? T(pt)->n : 1); i++)
				*slot(a.var, l, a.off + i) = x.b[i];
		}
		k = emit(I, I_OP, sc == SC_SHARED || sc == SC_WG ? C_STORE : -1,
		    sc == SC_SHARED || sc == SC_WG ? model_active()->c[C_STORE] : 0,
		    mask);
		(void)k;
		return;
	}
	if (!strcmp(I->op, "OpLoad")) {
		int p = opid(I, 1), sc;
		struct lval pv = val(p, 0);
		sc = V.id[pv.var].sc;
		for (l = 0; l < V.lanes; l++) {
			struct lval a, *r;
			if (!(mask >> l & 1))
				continue;
			a = val(p, l);
			r = LV(I->res, l);
			memset(r, 0, sizeof *r);
			for (i = 0; i < nc; i++)
				r->b[i] = *slot(a.var, l, a.off + i);
		}
		k = emit(I, I_OP, shared_sc(sc) ? C_LOAD : -1, shared_sc(sc) ?
		    model_active()->c[C_LOAD] : 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpAccessChain") || !strcmp(I->op,
	    "OpInBoundsAccessChain")) {
		int base = opid(I, 1), runtime = 0;
		for (l = 0; l < V.lanes; l++) {
			struct lval a, *r;
			int t;
			if (!(mask >> l & 1))
				continue;
			a = val(base, l);
			t = V.id[base].kind == K_VAR ? T(V.id[base].type)->e :
			    T(type_of(base))->e;
			for (i = 2; i < I->nop; i++) {
				int ix = opid(I, i), n, j;
				int64_t iv = geti(V.id[ix].kind == K_CONST ? V.id[ix].type :
				    type_of(ix), val(ix, l).b[0]);
				struct sty *y = T(t);
				if (V.id[ix].kind != K_CONST)
					runtime = 1;
				if (y->k == S_STRUCT) {
					for (j = 0, n = 0; j < iv && j < y->nm; j++)
						n += slots_of(y->m[j]);
					a.off += n;
					t = y->m[iv < y->nm ? iv : 0];
				} else if (y->k == S_ARRAY) {
					a.off += (int)iv * slots_of(y->e);
					t = y->e;
				} else if (y->k == S_VEC) {
					a.off += (int)iv;
					t = y->e;
				}
			}
			r = LV(I->res, l);
			*r = a;
		}
		k = emit(I, I_OP, runtime ? C_IADD : -1, runtime ?
		    model_active()->c[C_IADD] : 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpPhi")) {
		for (l = 0; l < V.lanes; l++) {
			if (!(mask >> l & 1))
				continue;
			for (i = 1; i + 1 < I->nop; i += 2)
				if (opid(I, i + 1) == V.prevblk[l]) {
					*LV(I->res, l) = val(opid(I, i), l);
					break;
				}
		}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpCompositeExtract")) {
		int c = opid(I, 1);
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(c, l), *r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				r->b[0] = x.b[atoi(I->opnd[2]) & 3];
			}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpCompositeConstruct")) {
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval *r = LV(I->res, l);
				int at = 0;
				memset(r, 0, sizeof *r);
				for (i = 1; i < I->nop && at < MAXCOMP; i++) {
					int c = opid(I, i), cn = ncomp(type_of(c) >= 0 ?
					    type_of(c) : V.id[c].type), j;
					struct lval x = val(c, l);
					for (j = 0; j < cn && at < MAXCOMP; j++)
						r->b[at++] = x.b[j];
				}
			}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpCompositeInsert")) {
		int obj = opid(I, 1), c = opid(I, 2);
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(c, l), *r = LV(I->res, l);
				*r = x;
				r->b[atoi(I->opnd[3]) & 3] = val(obj, l).b[0];
			}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpVectorShuffle")) {
		int a = opid(I, 1), b = opid(I, 2), na = ncomp(type_of(a) >= 0 ?
		    type_of(a) : V.id[a].type);
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(a, l), y = val(b, l), *r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				for (i = 3; i < I->nop && i - 3 < MAXCOMP; i++) {
					int s = atoi(I->opnd[i]);
					r->b[i - 3] = s < na ? x.b[s & 3] : y.b[(s - na) & 3];
				}
			}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpDot") || !strcmp(I->op, "OpVectorTimesScalar")) {
		int a = opid(I, 1), b = opid(I, 2), at = type_of(a) >= 0 ?
		    type_of(a) : V.id[a].type, n = ncomp(at), sa = scalar_of(at);
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(a, l), y = val(b, l), *r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				if (I->op[2] == 'D') {
					double s = 0;
					for (i = 0; i < n; i++)
						s += getf(sa, x.b[i]) * getf(sa, y.b[i]);
					r->b[0] = putf(rt, s);
				} else
					for (i = 0; i < n; i++)
						r->b[i] = putf(sa, getf(sa, x.b[i]) *
						    getf(sa, y.b[0]));
			}
		cls = C_FMUL;
		cost = model_active()->c[C_FMUL] * n;
		k = emit(I, I_OP, cls, cost, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpSelect")) {
		int c = opid(I, 1), a = opid(I, 2), b = opid(I, 3);
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval cv = val(c, l), *r = LV(I->res, l);
				*r = cv.b[0] ? val(a, l) : val(b, l);
			}
		k = emit(I, I_OP, C_SELECT, model_active()->c[C_SELECT] * nc, mask);
		emit_vals(k, I, mask);
		return;
	}
	if (!strcmp(I->op, "OpControlBarrier") ||
	    !strcmp(I->op, "OpMemoryBarrier")) {
		emit(I, I_BARRIER, C_BARRIER, I->op[2] == 'C' ?
		    model_active()->c[C_BARRIER] : model_active()->c[C_BARRIER] / 2,
		    mask);
		return;
	}
	if (!strncmp(I->op, "OpAtomic", 8)) {
		int p = opid(I, 1), sc;
		struct lval pv = val(p, 0);
		const char *a = I->op + 8;
		sc = V.id[pv.var].sc;
		for (l = 0; l < V.lanes; l++) {
			struct lval ptr, *r;
			uint64_t *m, old, x = 0;
			if (!(mask >> l & 1))
				continue;
			ptr = val(p, l);
			m = slot(ptr.var, l, ptr.off);
			old = *m;
			if (!strcmp(a, "Store"))
				x = val(opid(I, 3), l).b[0];
			else if (I->nop > 4)
				x = val(opid(I, 4), l).b[0];
			st = rt >= 0 ? rt : V.id[ptr.var].slotty[ptr.off];
			if (!strcmp(a, "IAdd")) *m = puti(st, geti(st, old) + geti(st, x));
			else if (!strcmp(a, "ISub")) *m = puti(st, geti(st, old) - geti(st, x));
			else if (!strcmp(a, "IIncrement")) *m = puti(st, geti(st, old) + 1);
			else if (!strcmp(a, "IDecrement")) *m = puti(st, geti(st, old) - 1);
			else if (!strcmp(a, "SMin")) *m = geti(st, x) < geti(st, old) ? x : old;
			else if (!strcmp(a, "SMax")) *m = geti(st, x) > geti(st, old) ? x : old;
			else if (!strcmp(a, "UMin")) *m = x < old ? x : old;
			else if (!strcmp(a, "UMax")) *m = x > old ? x : old;
			else if (!strcmp(a, "And")) *m = old & x;
			else if (!strcmp(a, "Or")) *m = old | x;
			else if (!strcmp(a, "Xor")) *m = old ^ x;
			else if (!strcmp(a, "Exchange") || !strcmp(a, "Store")) *m = x;
			else if (!strcmp(a, "CompareExchange")) {
				uint64_t cmp = val(opid(I, I->nop - 1), l).b[0];
				x = val(opid(I, I->nop - 2), l).b[0];
				if (old == cmp)
					*m = x;
			}
			if (I->res >= 0) {
				r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				r->b[0] = old;
			}
			/* one lane at a time */
			k = emit(I, I_ATOMIC, sc == SC_WG ? C_ATOM_SH : C_ATOM_GL,
			    model_active()->c[sc == SC_WG ? C_ATOM_SH : C_ATOM_GL],
			    1u << l);
			if (I->res >= 0) {
				char buf[64];
				fmt(buf, sizeof buf, V.id[I->res].type, LV(I->res, l));
				xr_val(k, l, buf);
			}
		}
		return;
	}
	if (!strcmp(I->op, "OpFunctionCall")) {
		int fn = opid(I, 1), at = V.id[fn].func_at, p = 0, j, lb = -1;
		uint32_t saved = V.retired;
		if (++V.depth > 16)
			fail("SPIR-V: calls nested too deeply");
		/* bind the parameters */
		for (j = at + 1; j < V.nin && !strcmp(V.in[j].op,
		    "OpFunctionParameter"); j++, p++) {
			int prm = V.in[j].res;
			V.id[prm].type = opid(&V.in[j], 0);
			V.id[prm].kind = K_VALUE;
			for (l = 0; l < V.lanes; l++)
				if (mask >> l & 1)
					*LV(prm, l) = val(opid(I, 2 + p), l);
		}
		for (; j < V.nin; j++)
			if (!strcmp(V.in[j].op, "OpLabel")) {
				lb = V.in[j].res;
				break;
			}
		emit(I, I_OP, -1, 0, mask);
		V.retired = 0;
		if (lb >= 0)
			region(lb, mask, -1, -1, NULL);
		V.retired = saved;
		V.depth--;
		if (I->res >= 0 && rt >= 0) {
			for (l = 0; l < V.lanes; l++)
				if (mask >> l & 1)
					*LV(I->res, l) = V.ret[l];
			k = emit(I, I_MERGE, -1, 0, mask);
			snprintf(V.err, sizeof V.err, "%s = result", V.id[I->res].name);
			emit_vals(k, I, mask);
		}
		return;
	}
	/* arithmetic, comparisons, conversions, extended instructions */
	if (rt < 0)
		die("unsupported instruction \"%s\"", I->text);
	{
		int base = 1, ar;
		if (!strcmp(I->op, "OpExtInst")) {   /* type, set, name, args */
			ext = I->opnd[2];
			base = 3;
		}
		ar = I->nop - base;
		for (l = 0; l < V.lanes; l++) {
			struct lval x, y, z, w4, *r;
			int at;
			if (!(mask >> l & 1))
				continue;
			x = val(opid(I, base), l);
			y = ar > 1 ? val(opid(I, base + 1), l) : x;
			z = ar > 2 ? val(opid(I, base + 2), l) : x;
			w4 = ar > 3 ? val(opid(I, base + 3), l) : x;
			/* a scalar operand next to vectors applies to each component */
			if (nc > 1) {
				int j;
				if (ar > 1 && vcomp(opid(I, base + 1)) == 1)
					for (j = 1; j < nc; j++)
						y.b[j] = y.b[0];
				if (ar > 2 && vcomp(opid(I, base + 2)) == 1)
					for (j = 1; j < nc; j++)
						z.b[j] = z.b[0];
				if (ar > 3 && vcomp(opid(I, base + 3)) == 1)
					for (j = 1; j < nc; j++)
						w4.b[j] = w4.b[0];
			}
			at = type_of(opid(I, base)) >= 0 ? type_of(opid(I, base)) :
			    V.id[opid(I, base)].type;
			at = scalar_of(at);
			r = LV(I->res, l);
			memset(r, 0, sizeof *r);
			for (i = 0; i < nc; i++) {
				if (!ext && !strcmp(I->op, "OpBitFieldInsert")) {
					/* base, insert, offset, count */
					int off = (int)(z.b[i] & 63), cnt = (int)(w4.b[i] & 127);
					uint64_t m = cnt >= 64 ? ~0ULL : wmask(cnt) << off;
					r->b[i] = puti(st, (int64_t)((x.b[i] & ~m) |
					    (y.b[i] << off & m)));
					continue;
				}
				r->b[i] = ext ? glsl(ext, st, at, x.b[i], y.b[i], z.b[i], &ok) :
				    alu(I->op, st, at, x.b[i], y.b[i], z.b[i], &ok);
				if (!ok && first)
					die("unsupported instruction \"%s\"", I->text);
			}
			first = 0;
		}
		cls = cost_class(I->op, ext, st, &mul);
		cost = cls >= 0 ? model_active()->c[cls] * mul * nc : 0;
		if (cls >= 0 && st >= 0 && T(st)->k == S_FLOAT && T(st)->w == 64 &&
		    cls >= C_FADD && cls <= C_FPOW)
			cost *= model_active()->c[C_F64];
		k = emit(I, I_OP, cls, cost, mask);
		emit_vals(k, I, mask);
	}
}

static int
block_start(int lbl)
{
	if (V.id[lbl].kind != K_LABEL)
		die("no block %s", V.id[lbl].name);
	return V.id[lbl].label_at + 1;
}

static int
popcount32(uint32_t m)
{
	int n = 0;

	for (; m; m &= m - 1)
		n++;
	return n;
}

/*
 * Run from block lbl with the lanes in mask until they reach stop (or
 * return).  Lanes that branch to brk (a loop's merge: a break) are added
 * to *arrived and stop there.
 */
static int
region(int lbl, uint32_t mask, int stop, int brk, uint32_t *arrived)
{
	int guard = 0;

	while (mask && lbl != stop) {
		int pc, merge = -1, loopmerge = -1, cont = -1;
		const struct sins *I;
		if (lbl == brk) {
			if (arrived)
				*arrived |= mask;
			return 0;
		}
		if (++guard > 4096)
			fail("SPIR-V: control flow does not settle");
		enter_block(lbl, mask);
		pc = block_start(lbl);
		for (;; pc++) {
			if (pc >= V.nin)
				die("block %s has no terminator", V.id[lbl].name);
			I = &V.in[pc];
			if (!strcmp(I->op, "OpSelectionMerge"))
				merge = opid(I, 0);
			else if (!strcmp(I->op, "OpLoopMerge")) {
				loopmerge = opid(I, 0);
				cont = opid(I, 1);
			} else if (!strcmp(I->op, "OpBranch") ||
			    !strcmp(I->op, "OpBranchConditional") ||
			    !strcmp(I->op, "OpSwitch") || !strcmp(I->op, "OpReturn") ||
			    !strcmp(I->op, "OpReturnValue") || !strcmp(I->op, "OpKill") ||
			    !strcmp(I->op, "OpTerminateInvocation") ||
			    !strcmp(I->op, "OpUnreachable"))
				break;
			else
				exec_one(I, mask);
		}
		if (!strcmp(I->op, "OpReturn") || !strcmp(I->op, "OpKill") ||
		    !strcmp(I->op, "OpTerminateInvocation") ||
		    !strcmp(I->op, "OpUnreachable")) {
			V.retired |= mask;
			return 0;
		}
		if (!strcmp(I->op, "OpReturnValue")) {
			int v = opid(I, 0), l;
			for (l = 0; l < V.lanes; l++)
				if (mask >> l & 1)
					V.ret[l] = val(v, l);
			V.retired |= mask;
			return 0;
		}
		if (loopmerge >= 0) {
			/* a loop: iterate while any lane runs */
			uint32_t active = mask, out = 0;
			int iter = 0, header = lbl;
			for (;;) {
				uint32_t go = 0, brk2 = 0;
				if (++iter > 1000)
					fail("SPIR-V: more than 1000 loop iterations");
				if (!strcmp(I->op, "OpBranchConditional")) {
					int c = opid(I, 0), t = label_target(I, 1),
					    f = label_target(I, 2), l, k;
					uint32_t mt = 0;
					for (l = 0; l < V.lanes; l++)
						if ((active >> l & 1) && val(c, l).b[0])
							mt |= 1u << l;
					k = emit(I, I_BRANCH, C_BRANCH,
					    model_active()->c[C_BRANCH], active);
					(void)k;
					go = t == loopmerge ? active & ~mt : mt;
					out |= active & ~go;
					if (go)
						region(t == loopmerge ? f : t, go, cont, loopmerge,
						    &brk2);
				} else {   /* OpBranch into the body */
					go = active;
					region(label_target(I, 0), go, cont, loopmerge, &brk2);
				}
				out |= brk2;
				go &= ~brk2 & ~V.retired;
				if (go)   /* the continue block back to the header */
					region(cont, go, header, loopmerge, &brk2);
				go &= ~brk2 & ~V.retired;
				out |= brk2;
				active = go;
				if (!active)
					break;
				/* the header again, for the lanes still looping */
				enter_block(header, active);
				for (pc = block_start(header);; pc++) {
					I = &V.in[pc];
					if (!strcmp(I->op, "OpBranch") ||
					    !strcmp(I->op, "OpBranchConditional"))
						break;
					if (strcmp(I->op, "OpLoopMerge"))
						exec_one(I, active);
				}
			}
			if (popcount32(out & ~V.retired) > 0 &&
			    out != (mask & ~V.retired)) {
				/* nothing: lanes may have retired */
			}
			{
				static const struct sins rc = {"OpReconverge", -1, 0, {0},
				    (char *)"reconverge after the loop (keep-up)"};
				if (iter > 1)
					emit(&rc, I_RECONV, C_DIVERGE,
					    model_active()->c[C_DIVERGE], mask & ~V.retired);
			}
			mask = mask & ~V.retired;
			lbl = loopmerge;
			continue;
		}
		if (!strcmp(I->op, "OpBranch")) {
			lbl = label_target(I, 0);
			continue;
		}
		if (!strcmp(I->op, "OpBranchConditional")) {
			int c = opid(I, 0), t = label_target(I, 1), f = label_target(I, 2);
			uint32_t mt = 0, mf, b1 = 0;
			int l;
			static const struct sins rc = {"OpReconverge", -1, 0, {0},
			    (char *)"reconverge (keep-up)"};
			for (l = 0; l < V.lanes; l++)
				if ((mask >> l & 1) && val(c, l).b[0])
					mt |= 1u << l;
			mf = mask & ~mt;
			emit(I, I_BRANCH, C_BRANCH, model_active()->c[C_BRANCH], mask);
			if (merge < 0) {
				/* no merge: break / continue style; both sides to stop */
				if (mt)
					region(t, mt, stop, brk, arrived);
				if (mf)
					region(f, mf, stop, brk, arrived);
				return 0;
			}
			if (mt)
				region(t, mt, merge, brk, &b1);
			if (mf)
				region(f, mf, merge, brk, &b1);
			if (arrived)
				*arrived |= b1;
			if (mt && mf)
				emit(&rc, I_RECONV, C_DIVERGE, model_active()->c[C_DIVERGE],
				    mask);
			mask &= ~b1 & ~V.retired;
			lbl = merge;
			continue;
		}
		if (!strcmp(I->op, "OpSwitch")) {
			int sel = opid(I, 0), dflt = label_target(I, 1), l, j, groups = 0;
			uint32_t left = mask, b1 = 0;
			int st = type_of(sel) >= 0 ? type_of(sel) : V.id[sel].type;
			emit(I, I_BRANCH, C_BRANCH, model_active()->c[C_BRANCH], mask);
			for (j = 2; j + 1 < I->nop; j += 2) {
				int64_t lit = strtoll(I->opnd[j], NULL, 0);
				int tg = label_target(I, j + 1);
				uint32_t g = 0;
				for (l = 0; l < V.lanes; l++)
					if ((left >> l & 1) && geti(st, val(sel, l).b[0]) == lit)
						g |= 1u << l;
				if (g) {
					region(tg, g, merge >= 0 ? merge : stop, brk, &b1);
					groups++;
				}
				left &= ~g;
			}
			if (left) {
				region(dflt, left, merge >= 0 ? merge : stop, brk, &b1);
				groups++;
			}
			if (arrived)
				*arrived |= b1;
			if (groups > 1) {
				static const struct sins rc = {"OpReconverge", -1, 0, {0},
				    (char *)"reconverge after the switch (keep-up)"};
				emit(&rc, I_RECONV, C_DIVERGE, model_active()->c[C_DIVERGE],
				    mask);
			}
			if (merge < 0)
				return 0;
			mask &= ~b1 & ~V.retired;
			lbl = merge;
			continue;
		}
	}
	return 0;
}

static void
dump_buffers(const char *when)
{
	int i, s;
	char buf[64];

	for (i = 0; i < V.nid; i++) {
		struct sid *d = &V.id[i];
		if (d->kind != K_VAR || d->sc != SC_SHARED)
			continue;
		printf("  %s %s:", when, d->name);
		for (s = 0; s < d->nslots && s < 16; s++) {
			struct lval v;
			memset(&v, 0, sizeof v);
			v.b[0] = d->mem[s];
			fmt(buf, sizeof buf, d->slotty[s], &v);
			printf(" %s", buf);
		}
		printf("%s\n", d->nslots > 16 ? " ..." : "");
	}
}

void
spirv_exec(const char *src, int lanes, int fps, double seconds, int random,
    uint64_t seed, const char *name, int nofile)
{
	int k, entry = -1, lb = -1, l, model = 0;
	char title[200];

	memset(&V, 0, sizeof V);
	symmap_reset();
	code_block_used = 1;
	V.glsl = -1;
	V.lanes = lanes;
	V.rng = seed * 0x9e3779b97f4a7c15ULL + 1;
	cols = lanes == 4 ? 2 : lanes == 8 ? 4 : lanes == 16 ? 4 : 8;
	(void)random;
	parse(src);
	{
		/* a snippet: complete it, then read the whole module afresh */
		struct sbuf said = {0};
		char *full = complete(src, &said);
		int i;
		if (full == NULL) {   /* a question is out */
			free(said.p);
			return;
		}
		for (k = 0; k < V.nin; k++) {
			int j;
			for (j = 0; j < V.in[k].nop; j++)
				free(V.in[k].opnd[j]);
			free(V.in[k].text);
		}
		V.nin = 0;
		V.nid = 0;
		for (i = 0; i < (int)nsymmap; i++)
			symmap[i] = -1;
		parse(full);
		free(full);
		if (said.n)
			fputs(said.p, stdout);
		free(said.p);
	}
	declare();
	/* the inputs' values in every lane */
	for (k = 0; k < ninps; k++) {
		int id = idx(inps[k].name), st;
		if (inps[k].var) {
			struct sid *d = &V.id[id];
			for (l = 0; l < lanes && d->mem; l++) {
				uint64_t *m = d->mem + (size_t)l * d->nslots;
				st = d->slotty[0];
				m[0] = inps[k].mode == 1 ? (T(st)->k == S_FLOAT ? putf(st, l) :
				    puti(st, l)) : inps[k].mode == 2 ? 0 : inps[k].mode == 3 ?
				    (T(st)->k == S_FLOAT ? putf(st, 1) : puti(st, 1)) :
				    inps[k].mode == 4 ? (T(st)->k == S_FLOAT ?
				    putf(st, inps[k].v) : puti(st, (int64_t)inps[k].v)) :
				    m[0];
			}
			continue;
		}
		V.id[id].kind = K_VALUE;
		V.id[id].type = idx(inps[k].ty);
		st = scalar_of(V.id[id].type);
		for (l = 0; l < lanes; l++) {
			struct lval *r = LV(id, l);
			int c, nc = ncomp(V.id[id].type);
			memset(r, 0, sizeof *r);
			for (c = 0; c < nc; c++) {
				double x = inps[k].mode == 1 ? l : inps[k].mode == 2 ? 0 :
				    inps[k].mode == 3 ? 1 : inps[k].mode == 4 ? inps[k].v :
				    T(st)->k == S_FLOAT ? (double)(int)(rnd() % 401 - 200) /
				    100.0 : (double)(rnd() % 10);
				r->b[c] = T(st)->k == S_FLOAT ? putf(st, x) :
				    puti(st, (int64_t)x);
			}
		}
	}
	for (k = 0; k < V.nin; k++)
		if (!strcmp(V.in[k].op, "OpEntryPoint") && V.in[k].nop >= 2) {
			entry = opid(&V.in[k], 1);
			model = !strcmp(V.in[k].opnd[0], "Fragment") ? 1 :
			    !strcmp(V.in[k].opnd[0], "GLCompute") ? 2 : 0;
			break;
		}
	if (entry < 0)
		fail("SPIR-V: no OpEntryPoint");
	for (k = V.id[entry].func_at; k < V.nin; k++)
		if (!strcmp(V.in[k].op, "OpLabel")) {
			lb = V.in[k].res;
			break;
		}
	if (lb < 0)
		fail("SPIR-V: the entry point has no body");
	V.all = lanes == 32 ? 0xffffffffu : (1u << lanes) - 1;
	snprintf(title, sizeof title, "exec of SPIR-V %s on %d lanes (%s), cost "
	    "model %s", V.id[entry].name, lanes, lanes == 4 ? "a 2x2 quad" :
	    lanes == 32 ? "a warp" : "lockstep", sym_str(model_active()->name));
	xr_begin(lanes, title);
	for (l = 0; l < lanes; l++) {
		char b[48], v[32];
		if (model == 1)
			snprintf(b, sizeof b, "frag (%d.5,%d.5)", l % cols, l / cols);
		else if (ninps > 0 && !inps[0].var) {
			int id = idx(inps[0].name);
			fmt(v, sizeof v, V.id[id].type, LV(id, l));
			snprintf(b, sizeof b, "%s=%s%s", inps[0].name, v, ninps > 1 ?
			    " ..." : "");
		} else
			snprintf(b, sizeof b, "id %d", l);
		xr_lanein(l, b);
	}
	printf("exec of SPIR-V (%d instructions), entry %s, %d lanes in "
	    "lockstep, cost model %s\n", V.nin, V.id[entry].name, lanes,
	    sym_str(model_active()->name));
	printf("  buffers start with random values (seed %llu)\n",
	    (unsigned long long)seed);
	for (k = 0; k < V.nin; k++)
		if (!strcmp(V.in[k].op, "OpExecutionMode") && V.in[k].nop >= 5 &&
		    !strcmp(V.in[k].opnd[1], "LocalSize")) {
			long wg = atol(V.in[k].opnd[2]) * atol(V.in[k].opnd[3]) *
			    atol(V.in[k].opnd[4]);
			if (wg > lanes)
				printf("  note: the workgroup has %ld invocations (LocalSize "
				    "%s %s %s); these are the first %d%s, so shared memory "
				    "the others would write stays as it started\n", wg,
				    V.in[k].opnd[2], V.in[k].opnd[3], V.in[k].opnd[4], lanes,
				    lanes < 32 ? " (add `lanes 32` for a warp)" : "");
		}
	dump_buffers("before:");
	region(lb, V.all, -1, -1, NULL);
	dump_buffers("after: ");
	if (V.oob)
		printf("  note: %ld access%s outside a variable (first: %s): read "
		    "as 0 and writes dropped, as with robustBufferAccess; random "
		    "inputs may not be valid data for this shader\n", V.oob,
		    V.oob == 1 ? "" : "es", V.oobvar);
	for (k = 0; k < ninps; k++) {
		struct sid *d;
		char v[48];
		if (inps[k].var != 2)
			continue;
		d = &V.id[idx(inps[k].name)];
		printf("  output %s per lane:", inps[k].name);
		for (l = 0; l < lanes && d->mem; l++) {
			struct lval x;
			memset(&x, 0, sizeof x);
			x.b[0] = d->mem[(size_t)l * d->nslots];
			fmt(v, sizeof v, d->slotty[0], &x);
			printf(" %s", v);
		}
		printf("\n");
	}
	code_block_used = 1;
	xr_done(fps, seconds, name, nofile);
	for (k = 0; k < V.nin; k++) {
		int j;
		for (j = 0; j < V.in[k].nop; j++)
			free(V.in[k].opnd[j]);
		free(V.in[k].text);
	}
	for (k = 0; k < V.nid; k++) {
		free(V.id[k].mem);
		free(V.id[k].slotty);
	}
	free(V.in);
	free(V.id);
	free(V.lv);
	free(symmap);
	symmap = NULL;
	nsymmap = 0;
}

/* ---- simplify: rewrites proved by SAT ---- */

/*
 * Every pure integer instruction of the module (arithmetic, bit
 * operations, comparisons, selects, integer min/max/clamp) is offered
 * cheaper replacements: one of its operands, a constant, or a single
 * cheaper instruction (a shift for a multiply by a power of two, an AND
 * for a bitfield extract at 0, a min for a clamp whose other side always
 * holds, ...).  A candidate is first run on random inputs with the
 * interpreter's own arithmetic, then the two sides are bit-blasted and
 * the solver must show that no input makes them differ.  The operands'
 * definitions are expanded a few levels deep (2, 4, 8, 16); anything
 * deeper, and every load, phi, parameter or specialization constant, is
 * a free input, so a proof holds for every value it could have.
 * Proved rewrites are applied in order (later ones see earlier ones),
 * dead pure instructions are removed, and the module is written back.
 */
#include <time.h>
#include "sat.h"

enum { RW_ID, RW_CONST, RW_INS };

struct cand {
	int kind;
	int id;			/* RW_ID: the operand */
	uint64_t v;		/* RW_CONST */
	char op[40], ext[24];	/* RW_INS: the instruction */
	int a[3], na;		/* its operands (ids) */
	double cost;
	const char *why;
};

static struct {
	int *def;		/* id -> defining instruction, -1 */
	int *alias;		/* id -> replacement id, -1 */
	char *dead, *rw;	/* per instruction */
	struct aig *g;
	struct word *memo;
	char *have;
	int *touched, ntouched;
	uint64_t *ev;		/* per id: value in the current sample */
	uint32_t *evstamp, stamp;
	int sample;
	uint64_t rng;
	int newk[256], nnewk;	/* constants made here */
	double deadline;
	long undecided, timeout, proved;
} SP;

static int
res_of(int id)
{
	while (SP.alias[id] >= 0)
		id = SP.alias[id];
	return id;
}

/* a scalar integer or boolean type id: its width, else 0 */
static int
int_width(int t)
{
	struct sty *y;

	if (t < 0 || V.id[t].kind != K_TYPE)
		return 0;
	y = &V.id[t].t;
	if (y->k == S_BOOL)
		return 1;
	if (y->k == S_INT && y->w >= 1 && y->w <= 64)
		return y->w;
	return 0;
}

static const char *const pure_core[] = {"OpIAdd", "OpISub", "OpIMul",
	"OpSNegate", "OpUDiv", "OpUMod", "OpSDiv", "OpSRem", "OpSMod",
	"OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic",
	"OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor", "OpNot", "OpBitCount",
	"OpBitReverse", "OpBitFieldUExtract", "OpBitFieldSExtract",
	"OpBitFieldInsert", "OpLogicalAnd", "OpLogicalOr", "OpLogicalNot",
	"OpLogicalEqual", "OpLogicalNotEqual", "OpIEqual", "OpINotEqual",
	"OpULessThan", "OpULessThanEqual", "OpUGreaterThan",
	"OpUGreaterThanEqual", "OpSLessThan", "OpSLessThanEqual",
	"OpSGreaterThan", "OpSGreaterThanEqual", "OpSelect", "OpBitcast",
	"OpUConvert", "OpSConvert", "OpCopyObject", NULL};
static const char *const pure_ext[] = {"SMin", "SMax", "UMin", "UMax",
	"SClamp", "UClamp", "SAbs", NULL};

static int
in_list(const char *w, const char *const *l)
{
	for (; *l; l++)
		if (!strcmp(w, *l))
			return 1;
	return 0;
}

/* the instruction computes a scalar integer from scalar integers only */
static int
pure_ins(int k)
{
	const struct sins *I = &V.in[k];
	int base = 1, i;

	if (SP.dead[k] || I->res < 0 || I->nop < 2 || I->opnd[0][0] != '%' ||
	    !int_width(idx(I->opnd[0])))
		return 0;
	if (!strcmp(I->op, "OpExtInst")) {
		if (I->nop < 4 || V.glsl < 0 || idx(I->opnd[1]) != V.glsl ||
		    !in_list(I->opnd[2], pure_ext))
			return 0;
		base = 3;
	} else if (!in_list(I->op, pure_core))
		return 0;
	for (i = base; i < I->nop; i++) {
		int a;
		if (I->opnd[i][0] != '%')
			return 0;
		a = res_of(idx(I->opnd[i]));
		if (!int_width(V.id[a].type))
			return 0;
	}
	if (!strcmp(I->op, "OpBitcast") &&
	    int_width(V.id[res_of(idx(I->opnd[1]))].type) !=
	    int_width(idx(I->opnd[0])))
		return 0;
	return 1;
}

/*
 * A phi at a selection merge: one of its incoming values, all defined
 * before it.  A loop header's phi (a value from the back edge, defined
 * later) stays a free input: its incoming value is from the previous
 * iteration, not the one the rest of the expression sees.
 */
static int
fwd_phi(int k)
{
	const struct sins *I = &V.in[k];
	int i;

	if (SP.dead[k] || strcmp(I->op, "OpPhi") != 0 || I->nop < 3 ||
	    !int_width(idx(I->opnd[0])))
		return 0;
	for (i = 1; i + 1 < I->nop; i += 2) {
		int v = res_of(idx(I->opnd[i])), d = SP.def[v];
		if (V.id[v].kind == K_CONST && !V.id[v].spec)
			continue;
		if (d < 0 || d >= k || !int_width(V.id[v].type))
			return 0;
	}
	return 1;
}

static int
first_arg(const struct sins *I)
{
	return !strcmp(I->op, "OpExtInst") ? 3 : 1;
}

/* ---- concrete values (the interpreter's arithmetic) ---- */

static uint64_t
leafval(int id, int w)
{
	uint64_t h = (uint64_t)id * 0x9e3779b97f4a7c15ULL ^
	    (uint64_t)SP.sample * 0xbf58476d1ce4e5b9ULL ^ SP.rng;
	int s = SP.sample % 8;

	h ^= h >> 31;
	h *= 0x94d049bb133111ebULL;
	h ^= h >> 29;
	if (w == 1)
		return h & 1;
	switch (s) {
	case 0: return 0;
	case 1: return 1;
	case 2: return wmask(w);
	case 3: return h % 17;
	case 4: return (1ULL << (h % (unsigned)w)) & wmask(w);
	case 5: return (wmask(w) >> 1) + (h & 1);	/* around the sign */
	}
	return h & wmask(w);
}

static uint64_t
evalid(int id)
{
	int k, rt, at, i, base, w;
	const struct sins *I;
	uint64_t x[4] = {0}, r;
	int ok = 1;

	id = res_of(id);
	if (SP.evstamp[id] == SP.stamp)
		return SP.ev[id];
	w = int_width(V.id[id].type);
	k = SP.def[id];
	if (V.id[id].kind == K_CONST && !V.id[id].spec)
		r = V.id[id].c[0] & wmask(w ? w : 64);
	else if (k >= 0 && fwd_phi(k)) {
		/* the sample takes one of the ways in */
		int np = (V.in[k].nop - 1) / 2;
		uint64_t h = (uint64_t)id * 0x2545f4914f6cdd1dULL ^
		    (uint64_t)SP.sample * 0x9e3779b97f4a7c15ULL;
		h ^= h >> 29;
		r = evalid(idx(V.in[k].opnd[1 + 2 * (int)(h % (uint64_t)np)]));
	} else if (k < 0 || !pure_ins(k))
		r = leafval(id, w ? w : 32);
	else {
		I = &V.in[k];
		rt = idx(I->opnd[0]);
		base = first_arg(I);
		for (i = base; i < I->nop && i - base < 4; i++)
			x[i - base] = evalid(idx(I->opnd[i]));
		at = V.id[res_of(idx(I->opnd[base]))].type;
		if (!strcmp(I->op, "OpSelect"))
			r = x[0] ? x[1] : x[2];
		else if (!strcmp(I->op, "OpBitFieldInsert")) {
			int off = (int)(x[2] & 63), cnt = (int)(x[3] & 127);
			uint64_t m = cnt >= 64 ? ~0ULL : wmask(cnt) << off;
			r = puti(rt, (int64_t)((x[0] & ~m) | (x[1] << off & m)));
		} else if (!strcmp(I->op, "OpUConvert"))
			r = x[0] & wmask(int_width(at)) & wmask(w);
		else if (!strcmp(I->op, "OpSConvert"))
			r = puti(rt, geti(at, x[0]));
		else if (!strcmp(I->op, "OpExtInst"))
			r = glsl(I->opnd[2], rt, at, x[0], x[1], x[2], &ok);
		else
			r = alu(I->op, rt, at, x[0], x[1], x[2], &ok);
		r &= wmask(w);
	}
	SP.evstamp[id] = SP.stamp;
	SP.ev[id] = r;
	return r;
}

/* the candidate's value in the current sample */
static uint64_t
evalcand(const struct cand *c, int rt)
{
	uint64_t x[3] = {0};
	int i, ok = 1, at;

	if (c->kind == RW_ID)
		return evalid(c->id);
	if (c->kind == RW_CONST)
		return c->v;
	for (i = 0; i < c->na; i++)
		x[i] = evalid(c->a[i]);
	at = V.id[res_of(c->a[0])].type;
	if (c->ext[0])
		return glsl(c->ext, rt, at, x[0], x[1], x[2], &ok) &
		    wmask(int_width(rt));
	return alu(c->op, rt, at, x[0], x[1], x[2], &ok) & wmask(int_width(rt));
}

/* ---- circuits ---- */

static struct word *blast(int id, int depth);

static void
sclamp_like(struct word *r, const struct word *x, const struct word *lo,
    const struct word *hi, int sign, int mode)
{
	struct aig *g = SP.g;
	lit lt, gt;

	/* mode 0: clamp; 1: min(x, lo); 2: max(x, lo) */
	if (mode == 1) {
		lt = sign ? w_slt(g, x, lo) : w_ult(g, x, lo);
		w_mux(g, r, lt, x, lo);
		return;
	}
	if (mode == 2) {
		gt = sign ? w_slt(g, lo, x) : w_ult(g, lo, x);
		w_mux(g, r, gt, x, lo);
		return;
	}
	{
		struct word t;
		lt = sign ? w_slt(g, x, lo) : w_ult(g, x, lo);
		gt = sign ? w_slt(g, hi, x) : w_ult(g, hi, x);
		w_mux(g, &t, gt, hi, x);
		w_mux(g, r, lt, lo, &t);
	}
}

/* r = op(a...) for the core and GLSL operations simplify handles */
static void
blast_op(struct word *r, const char *op, const char *ext, int rt,
    struct word **a, int na, int at)
{
	struct aig *g = SP.g;
	int w = int_width(rt), aw = int_width(at);
	struct word t, u, z;

	(void)na;
#define IS(n) (!strcmp(op, n))
	if (ext && ext[0]) {
		if (!strcmp(ext, "SMin")) sclamp_like(r, a[0], a[1], NULL, 1, 1);
		else if (!strcmp(ext, "UMin")) sclamp_like(r, a[0], a[1], NULL, 0, 1);
		else if (!strcmp(ext, "SMax")) sclamp_like(r, a[0], a[1], NULL, 1, 2);
		else if (!strcmp(ext, "UMax")) sclamp_like(r, a[0], a[1], NULL, 0, 2);
		else if (!strcmp(ext, "SClamp")) sclamp_like(r, a[0], a[1], a[2], 1, 0);
		else if (!strcmp(ext, "UClamp")) sclamp_like(r, a[0], a[1], a[2], 0, 0);
		else if (!strcmp(ext, "SAbs")) {
			w_neg(g, &t, a[0]);
			w_mux(g, r, a[0]->b[a[0]->w - 1], &t, a[0]);
		}
		return;
	}
	if (IS("OpIAdd")) w_add(g, r, a[0], a[1]);
	else if (IS("OpISub")) w_sub(g, r, a[0], a[1]);
	else if (IS("OpIMul")) w_mul(g, r, a[0], a[1]);
	else if (IS("OpSNegate")) w_neg(g, r, a[0]);
	else if (IS("OpNot")) w_not(g, r, a[0]);
	else if (IS("OpUDiv")) w_udivrem(g, r, &t, a[0], a[1]);
	else if (IS("OpUMod")) w_udivrem(g, &t, r, a[0], a[1]);
	else if (IS("OpSDiv") || IS("OpSRem") || IS("OpSMod")) {
		lit sa = a[0]->b[aw - 1], sb = a[1]->b[aw - 1];
		struct word na_, nb_, ua, ub, q, rm, nq;
		w_neg(g, &na_, a[0]);
		w_neg(g, &nb_, a[1]);
		w_mux(g, &ua, sa, &na_, a[0]);
		w_mux(g, &ub, sb, &nb_, a[1]);
		w_udivrem(g, &q, &rm, &ua, &ub);
		if (IS("OpSDiv")) {
			w_neg(g, &nq, &q);
			w_mux(g, r, aig_xor(g, sa, sb), &nq, &q);
		} else {
			struct word srem, adj;
			w_neg(g, &nq, &rm);
			w_mux(g, &srem, sa, &nq, &rm);
			if (IS("OpSRem"))
				*r = srem;
			else {
				lit nz, sr;
				w_const(&z, aw, 0);
				nz = w_ne(g, &srem, &z);
				sr = srem.b[aw - 1];
				w_add(g, &adj, &srem, a[1]);
				w_mux(g, r, aig_and(g, nz, aig_xor(g, sr, sb)), &adj, &srem);
			}
		}
	} else if (IS("OpShiftLeftLogical")) w_shl(g, r, a[0], a[1]);
	else if (IS("OpShiftRightLogical")) w_lshr(g, r, a[0], a[1]);
	else if (IS("OpShiftRightArithmetic")) w_ashr(g, r, a[0], a[1]);
	else if (IS("OpBitwiseAnd") || IS("OpLogicalAnd")) w_and(g, r, a[0], a[1]);
	else if (IS("OpBitwiseOr") || IS("OpLogicalOr")) w_or(g, r, a[0], a[1]);
	else if (IS("OpBitwiseXor") || IS("OpLogicalNotEqual"))
		w_xor(g, r, a[0], a[1]);
	else if (IS("OpLogicalNot")) w_not(g, r, a[0]);
	else if (IS("OpLogicalEqual")) {
		w_xor(g, &t, a[0], a[1]);
		w_not(g, r, &t);
	} else if (IS("OpBitCount")) {
		w_popcount(g, &t, a[0]);
		w_resize(r, &t, w, 0);
	} else if (IS("OpBitReverse")) w_reverse(r, a[0]);
	else if (IS("OpBitFieldUExtract") || IS("OpBitFieldSExtract")) {
		struct word ones, m, v, c7, sh, cm1, one, sb_, ext_;
		lit big;
		w_lshr(g, &t, a[0], a[1]);
		w_const(&ones, aw, wmask(aw));
		w_resize(&c7, a[2], a[2]->w, 0);
		{	/* count & 127 */
			int i;
			for (i = 7; i < c7.w; i++)
				c7.b[i] = L_FALSE;
		}
		w_const(&u, c7.w, (uint64_t)aw);
		big = L_NOT(w_ult(g, &c7, &u));
		w_shl(g, &sh, &ones, &c7);
		w_not(g, &m, &sh);
		w_mux(g, &m, big, &ones, &m);
		w_and(g, &v, &t, &m);
		if (IS("OpBitFieldUExtract"))
			*r = v;
		else {
			w_const(&one, c7.w, 1);
			w_sub(g, &cm1, &c7, &one);
			w_lshr(g, &sb_, &v, &cm1);
			w_not(g, &ext_, &m);
			w_or(g, &ext_, &v, &ext_);
			w_mux(g, &u, sb_.b[0], &ext_, &v);
			w_mux(g, r, big, &t, &u);
		}
	} else if (IS("OpBitFieldInsert")) {
		struct word ones, mk, m, c7, sh, ins, keep, nm;
		lit big;
		w_const(&ones, aw, wmask(aw));
		w_resize(&c7, a[3], a[3]->w, 0);
		{
			int i;
			for (i = 7; i < c7.w; i++)
				c7.b[i] = L_FALSE;
		}
		w_const(&u, c7.w, (uint64_t)aw);
		big = L_NOT(w_ult(g, &c7, &u));
		w_shl(g, &sh, &ones, &c7);
		w_not(g, &mk, &sh);
		w_mux(g, &mk, big, &ones, &mk);
		w_shl(g, &m, &mk, a[2]);
		w_shl(g, &ins, a[1], a[2]);
		w_and(g, &ins, &ins, &m);
		w_not(g, &nm, &m);
		w_and(g, &keep, a[0], &nm);
		w_or(g, r, &keep, &ins);
	} else if (IS("OpIEqual")) {
		w_const(r, 1, 0);
		r->b[0] = w_eq(g, a[0], a[1]);
	} else if (IS("OpINotEqual")) {
		w_const(r, 1, 0);
		r->b[0] = w_ne(g, a[0], a[1]);
	} else if (IS("OpULessThan") || IS("OpUGreaterThan") ||
	    IS("OpULessThanEqual") || IS("OpUGreaterThanEqual") ||
	    IS("OpSLessThan") || IS("OpSGreaterThan") ||
	    IS("OpSLessThanEqual") || IS("OpSGreaterThanEqual")) {
		int sgn = op[2] == 'S', swap = strstr(op, "Greater") != NULL,
		    eq = strstr(op, "Equal") != NULL;
		const struct word *x = swap ? a[1] : a[0], *y = swap ? a[0] : a[1];
		lit l;
		/* x < y, or x <= y as !(y < x) */
		if (eq)
			l = L_NOT(sgn ? w_slt(g, y, x) : w_ult(g, y, x));
		else
			l = sgn ? w_slt(g, x, y) : w_ult(g, x, y);
		w_const(r, 1, 0);
		r->b[0] = l;
	} else if (IS("OpSelect"))
		w_mux(g, r, a[0]->b[0], a[1], a[2]);
	else if (IS("OpBitcast") || IS("OpCopyObject"))
		*r = *a[0];
	else if (IS("OpUConvert"))
		w_resize(r, a[0], w, 0);
	else if (IS("OpSConvert"))
		w_resize(r, a[0], w, 1);
	else
		w_input(g, r, w);	/* not reached: pure_ins */
#undef IS
}

static struct word *
blast(int id, int depth)
{
	int k, w;
	struct word *r;

	id = res_of(id);
	r = &SP.memo[id];
	if (SP.have[id])
		return r;
	SP.have[id] = 1;
	SP.touched[SP.ntouched++] = id;
	w = int_width(V.id[id].type);
	if (w == 0)
		w = 32;
	k = SP.def[id];
	if (V.id[id].kind == K_CONST && !V.id[id].spec)
		w_const(r, w, V.id[id].c[0] & wmask(w));
	else if (k >= 0 && depth > 0 && pure_ins(k)) {
		const struct sins *I = &V.in[k];
		struct word *a[4], tmp;
		int base = first_arg(I), i, n = 0;
		for (i = base; i < I->nop && n < 4; i++)
			a[n++] = blast(idx(I->opnd[i]), depth - 1);
		blast_op(&tmp, I->op, !strcmp(I->op, "OpExtInst") ? I->opnd[2] :
		    NULL, idx(I->opnd[0]), a, n,
		    V.id[res_of(idx(I->opnd[base]))].type);
		*r = tmp;
	} else if (k >= 0 && depth > 0 && fwd_phi(k)) {
		/* one of the incoming values; which one is free */
		const struct sins *I = &V.in[k];
		struct word acc, *v;
		int i;
		acc = *blast(idx(I->opnd[1]), depth - 1);
		for (i = 3; i + 1 < I->nop; i += 2) {
			v = blast(idx(I->opnd[i]), depth - 1);
			w_mux(SP.g, &acc, aig_input(SP.g), v, &acc);
		}
		SP.memo[id] = acc;
		r = &SP.memo[id];
	} else
		w_input(SP.g, r, w);
	return r;
}

static void
blast_reset(void)
{
	while (SP.ntouched > 0)
		SP.have[SP.touched[--SP.ntouched]] = 0;
	aig_free(SP.g);
	SP.g = aig_new();
}

/* the cost of an instruction under the model */
static double
ins_cost(const char *op, const char *ext, int rt)
{
	double mul = 1;
	int cls = cost_class(op, ext && ext[0] ? ext : NULL, rt, &mul);

	return cls >= 0 ? model_active()->c[cls] * mul : 0;
}

/*
 * 1: proved equal for every input, 0: a difference (or none found to be
 * impossible within the depths), -1: the solver ran out of budget
 */
static int
prove(int x, const struct cand *c, int rt)
{
	static const int depths[] = {2, 4, 8, 16};
	int di, out = 0;

	for (di = 0; di < 4; di++) {
		struct word *wx, wc, *a[3];
		int i, r;
		if ((double)clock() / CLOCKS_PER_SEC > SP.deadline)
			return -2;
		blast_reset();
		wx = blast(x, depths[di]);
		if (c->kind == RW_ID)
			wc = *blast(c->id, depths[di] - 1);
		else if (c->kind == RW_CONST)
			w_const(&wc, wx->w, c->v);
		else {
			for (i = 0; i < c->na; i++)
				a[i] = blast(c->a[i], depths[di] - 1);
			blast_op(&wc, c->op, c->ext, rt, a, c->na,
			    V.id[res_of(c->a[0])].type);
		}
		if (wc.w != wx->w)
			return 0;
		r = aig_sat(SP.g, w_ne(SP.g, wx, &wc), 20000);
		if (r == 0)
			return 1;
		if (r < 0)
			out = -1;
		/* a difference: real, or only with the cut-off inputs free */
		if (r == 1 && aig_nodes(SP.g) > 400000)
			break;
	}
	return out;
}

/* a constant id of the type with the value: an existing one, or new */
static int
const_id(int rt, uint64_t v)
{
	int i;
	char nm[48];

	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_CONST && !V.id[i].spec && V.id[i].type == rt &&
		    V.id[i].nc <= 1 && (V.id[i].c[0] & wmask(int_width(rt))) == v)
			return i;
	if (SP.nnewk >= 256)
		return -1;
	if (int_width(rt) == 1)
		snprintf(nm, sizeof nm, "%%ucas_%s", v ? "true" : "false");
	else
		snprintf(nm, sizeof nm, "%%ucas_%s_%llu", V.id[rt].tok + 1,
		    (unsigned long long)v);
	i = idx(nm);
	if (V.id[i].kind != K_CONST) {
		V.id[i].kind = K_CONST;
		V.id[i].type = rt;
		V.id[i].c[0] = v;
		V.id[i].nc = 1;
		SP.newk[SP.nnewk++] = i;
	}
	return i;
}

static int
is_pow2(uint64_t v)
{
	return v && !(v & (v - 1));
}

static int
log2u(uint64_t v)
{
	int k = 0;

	while (v > 1) {
		v >>= 1;
		k++;
	}
	return k;
}

/* a constant operand's value, or -1 */
static int
const_of(int id, uint64_t *v)
{
	id = res_of(id);
	if (V.id[id].kind != K_CONST || V.id[id].spec)
		return 0;
	*v = V.id[id].c[0] & wmask(int_width(V.id[id].type) ?
	    int_width(V.id[id].type) : 64);
	return 1;
}

static int
ins_cand(struct cand *c, const char *op, const char *ext, int rt, int a0,
    int a1, int a2, const char *why)
{
	(void)rt;
	memset(c, 0, sizeof *c);
	c->kind = RW_INS;
	snprintf(c->op, sizeof c->op, "%s", op);
	snprintf(c->ext, sizeof c->ext, "%s", ext ? ext : "");
	c->a[0] = a0;
	c->a[1] = a1;
	c->a[2] = a2;
	c->na = a2 >= 0 ? 3 : a1 >= 0 ? 2 : 1;
	c->cost = ins_cost(op, ext, rt);
	c->why = why;
	return 1;
}

/* candidates for instruction k, cheapest first */
static int
candidates(int k, struct cand *cs, int max)
{
	const struct sins *I = &V.in[k];
	int rt = idx(I->opnd[0]), w = int_width(rt), base = first_arg(I),
	    n = 0, i, s, nargs = I->nop - base;
	int arg[4];
	uint64_t v0 = 0, cv;
	const char *ext = !strcmp(I->op, "OpExtInst") ? I->opnd[2] : NULL;
	int isconst = 1;

	for (i = 0; i < nargs && i < 4; i++)
		arg[i] = res_of(idx(I->opnd[base + i]));
	/* the value on random inputs: a constant? an operand? */
	for (s = 0; s < 24; s++) {
		uint64_t v;
		SP.sample = s;
		SP.stamp++;
		v = evalid(I->res);
		if (s == 0)
			v0 = v;
		else if (v != v0)
			isconst = 0;
	}
	if (isconst && n < max) {
		memset(&cs[n], 0, sizeof cs[n]);
		cs[n].kind = RW_CONST;
		cs[n].v = v0;
		cs[n].why = "always this value";
		n++;
	}
	for (i = 0; i < nargs && i < 4 && n < max; i++) {
		int same = 1;
		if (V.id[arg[i]].type != rt)
			continue;
		for (s = 0; s < 24 && same; s++) {
			SP.sample = s;
			SP.stamp++;
			same = evalid(I->res) == evalid(arg[i]);
		}
		if (same) {
			memset(&cs[n], 0, sizeof cs[n]);
			cs[n].kind = RW_ID;
			cs[n].id = arg[i];
			cs[n].why = "equals its operand";
			n++;
		}
	}
	/* strength reductions (each may need a new constant: room for it) */
	if (SP.nnewk >= 240)
		return n;
	if (!strcmp(I->op, "OpIMul") && n < max) {
		for (i = 0; i < 2; i++)
			if (const_of(arg[i], &cv) && is_pow2(cv) && cv > 1) {
				n += ins_cand(&cs[n], "OpShiftLeftLogical", NULL, rt,
				    arg[1 - i], const_id(rt, (uint64_t)log2u(cv)), -1,
				    "a multiply by a power of two is a shift");
				break;
			}
	}
	if ((!strcmp(I->op, "OpUDiv") || !strcmp(I->op, "OpSDiv")) &&
	    const_of(arg[1], &cv) && is_pow2(cv) && cv > 1 && n < max)
		n += ins_cand(&cs[n], "OpShiftRightLogical", NULL, rt, arg[0],
		    const_id(rt, (uint64_t)log2u(cv)), -1, I->op[2] == 'U' ?
		    "a divide by a power of two is a shift" :
		    "the dividend is never negative: a shift");
	if ((!strcmp(I->op, "OpUMod") || !strcmp(I->op, "OpSRem") ||
	    !strcmp(I->op, "OpSMod")) && const_of(arg[1], &cv) && is_pow2(cv) &&
	    n < max)
		n += ins_cand(&cs[n], "OpBitwiseAnd", NULL, rt, arg[0],
		    const_id(rt, cv - 1), -1, I->op[2] == 'U' ?
		    "a remainder by a power of two is a mask" :
		    "the dividend is never negative: a mask");
	if (!strcmp(I->op, "OpBitFieldUExtract") && const_of(arg[1], &cv) &&
	    cv == 0 && n < max) {
		uint64_t cnt;
		if (const_of(arg[2], &cnt) && cnt < (uint64_t)w)
			n += ins_cand(&cs[n], "OpBitwiseAnd", NULL, rt, arg[0],
			    const_id(rt, wmask((int)cnt)), -1,
			    "an extract at bit 0 is a mask");
	}
	if (ext && (!strcmp(ext, "SClamp") || !strcmp(ext, "UClamp")) &&
	    n + 2 <= max) {
		n += ins_cand(&cs[n], "OpExtInst", ext[0] == 'S' ? "SMin" : "UMin",
		    rt, arg[0], arg[2], -1, "never below the low bound: a min");
		n += ins_cand(&cs[n], "OpExtInst", ext[0] == 'S' ? "SMax" : "UMax",
		    rt, arg[0], arg[1], -1, "never above the high bound: a max");
	}
	/* (x >> k) & (2^n - 1): one bitfield extract */
	if (!strcmp(I->op, "OpBitwiseAnd") && n < max)
		for (i = 0; i < 2; i++) {
			int d = SP.def[arg[1 - i]];
			uint64_t m, sh;
			if (const_of(arg[i], &m) && m && is_pow2(m + 1) && d >= 0 &&
			    !SP.dead[d] && !strcmp(V.in[d].op, "OpShiftRightLogical") &&
			    const_of(res_of(idx(V.in[d].opnd[2])), &sh)) {
				n += ins_cand(&cs[n], "OpBitFieldUExtract", NULL, rt,
				    res_of(idx(V.in[d].opnd[1])),
				    const_id(rt, sh), const_id(rt, (uint64_t)log2u(m + 1)),
				    "a shift then a mask is one bitfield extract");
				break;
			}
		}
	(void)w;
	return n;
}

/* a constant made here is referenced by what is left */
static int
const_used(int c)
{
	int k, i;

	for (k = 0; k < V.nin; k++) {
		if (SP.dead[k])
			continue;
		for (i = 0; i < V.in[k].nop; i++)
			if (V.in[k].opnd[i][0] == '%' &&
			    res_of(idx(V.in[k].opnd[i])) == c)
				return 1;
	}
	return 0;
}

static void
cand_text(char *b, size_t n, const struct cand *c, const struct sins *I)
{
	int i, at = 0;

	if (c->kind == RW_ID) {
		snprintf(b, n, "%s", V.id[c->id].tok);
		return;
	}
	if (c->kind == RW_CONST) {
		if (int_width(idx(I->opnd[0])) == 1)
			snprintf(b, n, "%s", c->v ? "true" : "false");
		else
			snprintf(b, n, "%llu", (unsigned long long)c->v);
		return;
	}
	if (c->ext[0])
		at = snprintf(b, n, "OpExtInst %s %s %s", I->opnd[0],
		    V.glsl >= 0 ? V.id[V.glsl].tok : "%glsl", c->ext);
	else
		at = snprintf(b, n, "%s %s", c->op, I->opnd[0]);
	for (i = 0; i < c->na && (size_t)at < n; i++)
		at += snprintf(b + at, n - (size_t)at, " %s", V.id[c->a[i]].tok);
}

/* apply: alias, constant alias, or the instruction rewritten in place */
static void
apply(int k, const struct cand *c)
{
	struct sins *I = &V.in[k];
	int rt = idx(I->opnd[0]), i;
	char buf[64];

	if (c->kind == RW_ID) {
		SP.alias[I->res] = c->id;
		SP.dead[k] = 1;
		return;
	}
	if (c->kind == RW_CONST) {
		int ci = const_id(rt, c->v);
		if (ci < 0)
			return;
		SP.alias[I->res] = ci;
		SP.dead[k] = 1;
		return;
	}
	for (i = 1; i < I->nop; i++)
		free(I->opnd[i]);
	I->nop = 1;
	if (c->ext[0]) {
		snprintf(I->op, sizeof I->op, "OpExtInst");
		I->opnd[I->nop++] = strdup(V.id[V.glsl].tok);
		I->opnd[I->nop++] = strdup(c->ext);
	} else
		snprintf(I->op, sizeof I->op, "%s", c->op);
	for (i = 0; i < c->na; i++) {
		snprintf(buf, sizeof buf, "%s", V.id[c->a[i]].tok);
		I->opnd[I->nop++] = strdup(buf);
	}
	SP.rw[k] = 1;
}

void
spirv_simplify(const char *src, double budget, int nofile)
{
	int k, i, nrw = 0, ndead = 0, inbody = 0, nlog = 0;
#define MAXLOG 4096
	struct rwlog {
		char from[160], to[160];
		const char *why;
		double gain;
	} *rl = xcalloc(MAXLOG, sizeof *rl);
	double before = 0, after = 0;
	struct sbuf out = {0}, list = {0};
	clock_t t0 = clock();

	memset(&V, 0, sizeof V);
	symmap_reset();
	memset(&SP, 0, sizeof SP);
	code_block_used = 1;
	V.glsl = -1;
	V.lanes = 1;
	V.rng = 1;
	cols = 1;
	parse(src);
	{
		/* a snippet: complete it (types only), then read it afresh */
		struct sbuf said = {0};
		char *full;
		values_unneeded = 1;
		full = complete(src, &said);
		values_unneeded = 0;
		if (full == NULL) {
			free(said.p);
			return;
		}
		for (k = 0; k < V.nin; k++) {
			int j;
			for (j = 0; j < V.in[k].nop; j++)
				free(V.in[k].opnd[j]);
			free(V.in[k].text);
		}
		V.nin = 0;
		V.nid = 0;
		for (i = 0; i < (int)nsymmap; i++)
			symmap[i] = -1;
		parse(full);
		free(full);
		if (said.n)
			fputs(said.p, stdout);
		free(said.p);
	}
	declare();
	/* a snippet's inputs: their inferred types (they are free inputs) */
	for (k = 0; k < ninps; k++)
		if (!inps[k].var && inps[k].ty[0])
			V.id[idx(inps[k].name)].type = idx(inps[k].ty);
	/* result types, definitions */
	SP.def = xmalloc((size_t)(V.nid + 512) * sizeof *SP.def);
	SP.alias = xmalloc((size_t)(V.nid + 512) * sizeof *SP.alias);
	for (i = 0; i < V.nid + 512; i++)
		SP.def[i] = SP.alias[i] = -1;
	SP.dead = xcalloc((size_t)V.nin, 1);
	SP.rw = xcalloc((size_t)V.nin, 1);
	for (k = 0; k < V.nin; k++) {
		struct sins *I = &V.in[k];
		if (!strcmp(I->op, "OpFunction"))
			inbody = 1;
		if (I->res < 0 || !inbody)
			continue;
		SP.def[I->res] = k;
		if (I->nop >= 1 && I->opnd[0][0] == '%' &&
		    V.id[idx(I->opnd[0])].kind == K_TYPE)
			V.id[I->res].type = idx(I->opnd[0]);
	}
	SP.memo = xcalloc((size_t)(V.nid + 512), sizeof *SP.memo);
	SP.have = xcalloc((size_t)(V.nid + 512), 1);
	SP.touched = xmalloc((size_t)(V.nid + 512) * sizeof *SP.touched);
	SP.ev = xcalloc((size_t)(V.nid + 512), sizeof *SP.ev);
	SP.evstamp = xcalloc((size_t)(V.nid + 512), sizeof *SP.evstamp);
	SP.rng = 0x51c3a5e1ULL;
	SP.deadline = (double)t0 / CLOCKS_PER_SEC + budget;
	SP.g = aig_new();
	for (k = 0; k < V.nin; k++)
		if (pure_ins(k)) {
			const char *ext = !strcmp(V.in[k].op, "OpExtInst") ?
			    V.in[k].opnd[2] : NULL;
			before += ins_cost(V.in[k].op, ext, idx(V.in[k].opnd[0]));
		}
	/* in order: a rewrite is seen by the instructions after it */
	for (k = 0; k < V.nin; k++) {
		struct cand cs[12];
		int nc, best = -1, rt;
		double cost, gain = 0;
		const char *ext;
		if (!pure_ins(k))
			continue;
		rt = idx(V.in[k].opnd[0]);
		ext = !strcmp(V.in[k].op, "OpExtInst") ? V.in[k].opnd[2] : NULL;
		cost = ins_cost(V.in[k].op, ext, rt);
		nc = candidates(k, cs, 12);
		/* the cheapest that is cheaper and proves */
		for (;;) {
			int j, r;
			best = -1;
			for (j = 0; j < nc; j++)
				if (cs[j].why && cost - cs[j].cost > 1e-9 &&
				    (best < 0 || cs[j].cost < cs[best].cost))
					best = j;
			if (best < 0)
				break;
			/* random inputs first: the interpreter's arithmetic */
			{
				int s, same = 1;
				for (s = 0; s < 24 && same; s++) {
					SP.sample = s;
					SP.stamp++;
					same = evalid(V.in[k].res) == evalcand(&cs[best], rt);
				}
				if (!same) {
					cs[best].why = NULL;
					continue;
				}
			}
			r = prove(V.in[k].res, &cs[best], rt);
			if (r == 1) {
				gain = cost - cs[best].cost;
				break;
			}
			if (r == -1)
				SP.undecided++;
			if (r == -2) {
				SP.timeout++;
				best = -1;
				break;
			}
			cs[best].why = NULL;
		}
		if (best < 0)
			continue;
		if (nlog < MAXLOG) {
			struct rwlog *L = &rl[nlog++];
			cand_text(L->to, sizeof L->to, &cs[best], &V.in[k]);
			snprintf(L->from, sizeof L->from, "%s", V.in[k].text);
			L->why = cs[best].why;
			L->gain = gain;
		}
		apply(k, &cs[best]);
		nrw++;
		SP.proved++;
	}
	/* dead pure instructions: no uses left */
	{
		int *uses = xcalloc((size_t)(V.nid + 512), sizeof *uses), changed = 1;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (SP.dead[k] || !strncmp(I->op, "OpName", 6) ||
			    !strncmp(I->op, "OpMemberName", 12) ||
			    !strncmp(I->op, "OpDecorate", 10) ||
			    !strncmp(I->op, "OpMemberDecorate", 16))
				continue;
			for (i = 0; i < I->nop; i++)
				if (I->opnd[i][0] == '%')
					uses[res_of(idx(I->opnd[i]))]++;
		}
		while (changed) {
			changed = 0;
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				if (SP.dead[k] || I->res < 0 || uses[I->res] ||
				    SP.def[I->res] != k || !pure_ins(k))
					continue;
				SP.dead[k] = 1;
				ndead++;
				changed = 1;
				for (i = 0; i < I->nop; i++)
					if (I->opnd[i][0] == '%')
						uses[res_of(idx(I->opnd[i]))]--;
			}
		}
		free(uses);
	}
	for (k = 0; k < V.nin; k++)
		if (!SP.dead[k] && pure_ins(k)) {
			const char *ext = !strcmp(V.in[k].op, "OpExtInst") ?
			    V.in[k].opnd[2] : NULL;
			after += ins_cost(V.in[k].op, ext, idx(V.in[k].opnd[0]));
		}
	/* the module, rewritten */
	{
		int first_fn = -1;
		for (k = 0; k < V.nin; k++)
			if (!strcmp(V.in[k].op, "OpFunction")) {
				first_fn = k;
				break;
			}
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			int subst = 0;
			if (k == first_fn)
				for (i = 0; i < SP.nnewk; i++) {
					struct sid *d = &V.id[SP.newk[i]];
					if (!const_used(SP.newk[i]))
						continue;
					if (int_width(d->type) == 1)
						sb_printf(&out, "%s = OpConstant%s %s\n", d->tok,
						    d->c[0] ? "True" : "False", V.id[d->type].tok);
					else
						sb_printf(&out, "%s = OpConstant %s %llu\n", d->tok,
						    V.id[d->type].tok, (unsigned long long)d->c[0]);
				}
			if (SP.dead[k])
				continue;
			/* names and decorations of removed ids go too */
			if ((!strcmp(I->op, "OpName") || !strncmp(I->op, "OpDecorate", 10))
			    && I->nop >= 1 && I->opnd[0][0] == '%') {
				int t = idx(I->opnd[0]);
				if (SP.def[t] >= 0 && SP.dead[SP.def[t]])
					continue;
			}
			for (i = 0; i < I->nop; i++)
				if (I->opnd[i][0] == '%' && SP.alias[idx(I->opnd[i])] >= 0)
					subst = 1;
			if (!subst && !SP.rw[k]) {
				sb_printf(&out, "%s\n", I->text);
				continue;
			}
			if (I->res >= 0)
				sb_printf(&out, "%s = ", V.id[I->res].tok);
			sb_puts(&out, I->op);
			for (i = 0; i < I->nop; i++)
				sb_printf(&out, " %s", I->opnd[i][0] == '%' ?
				    V.id[res_of(idx(I->opnd[i]))].tok : I->opnd[i]);
			sb_putc(&out, '\n');
		}
	}
	printf("simplify: %d SPIR-V instructions, %d rewrite%s proved by SAT "
	    "(bit-precise: equal for every input), %d dead instruction%s "
	    "removed\n", V.nin, nrw, nrw == 1 ? "" : "s", ndead,
	    ndead == 1 ? "" : "s");
	printf("  integer arithmetic cost under %s (each instruction once): "
	    "%g -> %g", sym_str(model_active()->name), before, after);
	if (before > 0)
		printf(" (-%.1f%%)", 100 * (before - after) / before);
	printf("\n");
	if (nlog > 0) {
		/* by kind, then the largest savings first */
		const char *kinds[32];
		int cnt[32], nk = 0, j, shown;
		double sum[32];
		for (i = 0; i < nlog; i++) {
			for (j = 0; j < nk; j++)
				if (kinds[j] == rl[i].why)
					break;
			if (j == nk && nk < 32) {
				kinds[nk] = rl[i].why;
				cnt[nk] = 0;
				sum[nk++] = 0;
			}
			if (j < 32) {
				cnt[j]++;
				sum[j] += rl[i].gain;
			}
		}
		printf("  by kind:\n");
		for (j = 0; j < nk; j++)
			printf("    %4d  %-44s  -%g\n", cnt[j], kinds[j], sum[j]);
		for (i = 1; i < nlog; i++) {	/* stable: by gain, then order */
			struct rwlog x = rl[i];
			for (j = i; j > 0 && rl[j - 1].gain < x.gain; j--)
				rl[j] = rl[j - 1];
			rl[j] = x;
		}
		shown = nlog < 25 ? nlog : 25;
		printf("  %s:\n", shown < nlog ? "the largest savings" : "rewrites");
		for (i = 0; i < shown; i++)
			sb_printf(&list, "    %s\n      ->  %s   (%s; -%g)\n",
			    rl[i].from, rl[i].to, rl[i].why, rl[i].gain);
		if (shown < nlog)
			sb_printf(&list, "    ... %d more (all in the attached "
			    "module)\n", nlog - shown);
		fputs(list.p, stdout);
	} else
		printf("  nothing to rewrite: no cheaper form of an integer "
		    "instruction could be proved\n");
	if (SP.undecided)
		printf("  %ld candidate%s too hard for the solver's budget (kept as "
		    "they are)\n", SP.undecided, SP.undecided == 1 ? "" : "s");
	if (SP.timeout)
		printf("  note: stopped proving after %g s (the time budget; "
		    "`budget N` for more)\n", budget);
	printf("  proved in %.2f s; the operands' definitions were expanded up "
	    "to 16 levels, the rest (loads, loop phis, spec constants) are free "
	    "inputs\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
	if (nrw + ndead > 0)
		deliver_file("simplified.spvasm", "text/plain", out.p, out.n,
		    nofile);
	free(out.p);
	free(list.p);
	free(rl);
	aig_free(SP.g);
	free(SP.def);
	free(SP.alias);
	free(SP.dead);
	free(SP.rw);
	free(SP.memo);
	free(SP.have);
	free(SP.touched);
	free(SP.ev);
	free(SP.evstamp);
	for (k = 0; k < V.nin; k++) {
		for (i = 0; i < V.in[k].nop; i++)
			free(V.in[k].opnd[i]);
		free(V.in[k].text);
	}
	free(V.in);
	memset(&V, 0, sizeof V);
}
