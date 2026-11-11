/* SPDX-License-Identifier: GPL-3.0-or-later */
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
#define MAXCOMP 16		/* a value's components: a mat4 at most */

enum { K_NONE, K_TYPE, K_CONST, K_VAR, K_FUNC, K_LABEL, K_VALUE, K_EXT,
	K_PARAM };
enum { S_VOID, S_BOOL, S_INT, S_FLOAT, S_VEC, S_STRUCT, S_ARRAY, S_PTR,
	S_FN, S_OTHER, S_MAT, S_IMAGE, S_SAMPLER, S_SIMAGE };
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
	/* an image: Dim (TD_), Depth, Arrayed, MS, Sampled (1 sampled, 2
	 * storage), the format (tex.c's, -1 Unknown) */
	signed char dim, depth, arrayed, ms, sampled, ifmt;
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
	int set, binding, location;	/* decorations, -1 none */
	int nonread, nonwrite;		/* NonReadable, NonWritable */
	int setix;		/* "%name = value" for an input variable, -1 */
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
	int quiet;		/* the whole-image run: nothing recorded */
	uint32_t killed;	/* lanes that discarded */
	long divderiv;		/* derivatives with a partner lane masked */
} V;

#include "img.h"

/*
 * Images: every image, sampled image and sampler binding (and each
 * element of an array of them) gets a texture or a sampler here; its
 * variable's memory slot holds a handle, (texture + 1) | (sampler + 1)
 * << 16, so loads, access chains, phis, selects and calls carry images
 * like any other value.
 */
#define MAXTEX 32
static struct {
	struct tex t[MAXTEX];
	struct tex init[MAXTEX];	/* as it started, for the image run */
	char label[MAXTEX][96];
	int slot[MAXTEX];		/* the input slot (0..), -1 none */
	int storage[MAXTEX], written[MAXTEX];
	int nt;
	struct samp s[MAXTEX];
	int ns;
	long oob;			/* texel accesses outside the image */
	long nsample;
	int on;				/* the shader has images, or asked */
	int W, H;			/* the image run's size */
	int x0, y0;			/* the lanes' first pixel */
	int uv;				/* the input taken as uv, -1 */
	int lsize[3];			/* compute: LocalSize */
	int run;			/* in the image run */
} G;

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
	V.id[i].set = V.id[i].binding = V.id[i].location = V.id[i].setix = -1;
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

void glslfn_pass(const char *src, double budget, int only);
/*
 * Whether an elapsed time is worth printing: shorter than half a second is
 * noise, not news, and UCAS_NO_TIMES (set by `make test`) leaves times out
 * so that outputs do not depend on the machine's load.
 */
static int
show_time(double el)
{
	return el >= 0.5 && getenv("UCAS_NO_TIMES") == NULL;
}

static int fnl_mode;	/* the function level: 0 too, 1 alone, 2 not */
static char *src_text;	/* OpSource's text, when there is one */

static void
parse(const char *src)
{
	const char *p = src, *e;
	static char *line;	/* one instruction: a table's constant can be long */
	static size_t linecap;
	char *q, *tok;
	int n;

	while (*p) {
		struct sins I;
		int quotes = 0;
		const char *c;
		e = strchr(p, '\n');
		if (e == NULL)
			e = p + strlen(p);
		/* a string literal may run over lines (OpSource's text): join
		 * them, the text kept with its newlines */
		for (;;) {
			for (c = p, quotes = 0; c < e; c++)
				if (*c == '"' && (c == p || c[-1] != '\\'))
					quotes++;
				else if (*c == ';' && !(quotes & 1))
					break;
			if (!(quotes & 1) || *e == 0)
				break;
			{
				const char *e2 = strchr(e + 1, '\n');
				e = e2 ? e2 : e + 1 + strlen(e + 1);
			}
		}
		/* OpSource's text goes to src_text; the instruction keeps the rest */
		{
			const char *q = memchr(p, '"', (size_t)(e - p));
			const char *os = strstr(p, "OpSource");
			/* OpSource itself: not OpSourceExtension's name */
			int is_src = q != NULL && (e - p) > 9 && os != NULL &&
			    os < q && (os[8] == ' ' || os[8] == '\t');
			const char *end = is_src ? q : e;
			if (is_src) {
				const char *z = e;
				while (z > q && *z != '"')
					z--;
				free(src_text);
				src_text = xmalloc((size_t)(z - q) + 1);
				memcpy(src_text, q + 1, (size_t)(z > q ? z - q - 1 : 0));
				src_text[z > q ? z - q - 1 : 0] = 0;
			}
			n = (int)(end - p);
			if ((size_t)n + 1 > linecap) {
				linecap = (size_t)n + 1024;
				line = xrealloc(line, linecap);
			}
			memcpy(line, p, (size_t)n);
			line[n] = 0;
		}
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
	/* a whole module with nothing undefined: as it is */
	if (ninps == 0 && hasfn && hasentry && hasmem && hascap &&
	    (!needglsl || hasglsl))
		return strdup(src);
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
		    !strcmp(I->op, "OpString") || !strcmp(I->op, "OpModuleProcessed") ||
		    !strncmp(I->op, "OpSourceExtension", 17) ||
		    !strcmp(I->op, "OpSourceContinued") ||
		    !strcmp(I->op, "OpDecorateString") ||
		    !strcmp(I->op, "OpDecorateId") ||
		    !strcmp(I->op, "OpMemberDecorateString") ||
		    !strcmp(I->op, "OpExecutionModeId") ||
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
	case S_ARRAY: case S_MAT:
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
	case S_ARRAY: case S_MAT:
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

	return y->k == S_VEC ? y->n : y->k == S_MAT ? y->n * ncomp(y->e) : 1;
}

static int
scalar_of(int t)
{
	struct sty *y = T(t);

	return y->k == S_VEC ? y->e : y->k == S_MAT ? scalar_of(y->e) : t;
}

/* a value's words: components of a vector or matrix, a small struct's
 * or array's slots */
static int
nwords(int t)
{
	int n;

	if (t < 0 || V.id[t].kind != K_TYPE)
		return 1;
	n = T(t)->k == S_PTR ? 1 : slots_of(t);
	return n > MAXCOMP ? MAXCOMP : n;
}

/*
 * The first word and type of a composite's part, by literal indices
 * opnd[from..]: a vector's component, a matrix's column or element, a
 * struct's member, an array's element
 */
static int
part_of(int t, const struct sins *I, int from, int *rt)
{
	int off = 0, i, j;

	for (i = from; i < I->nop; i++) {
		struct sty *y = T(t);
		int ix = atoi(I->opnd[i]);
		if (y->k == S_STRUCT) {
			if (ix < 0 || ix >= y->nm)
				ix = 0;
			for (j = 0; j < ix; j++)
				off += slots_of(y->m[j]);
			t = y->m[ix];
		} else if (y->k == S_VEC || y->k == S_MAT || y->k == S_ARRAY) {
			if (ix < 0 || ix >= y->n)
				ix = 0;
			off += ix * slots_of(y->e);
			t = y->e;
		} else
			break;
	}
	*rt = t;
	return off;
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
		int px = G.x0 + lane % cols, py = G.y0 + lane / cols;
		uint64_t b = 0;
		switch (d->builtin) {
		case B_FRAGCOORD:
			b = putf(st, s == 0 ? px + 0.5 : s == 1 ? py + 0.5 : s == 2 ?
			    0.5 : 1.0);
			break;
		case B_LOCALID: case B_GLOBALID:
			if (G.on) {	/* an invocation per pixel */
				int g = s == 0 ? px : s == 1 ? py : 0;
				b = puti(st, d->builtin == B_GLOBALID ? g :
				    g % (G.lsize[s < 3 ? s : 0] > 0 ? G.lsize[s] : 1));
			} else
				b = puti(st, s == 0 ? lane : 0);
			break;
		case B_WGID:
			if (G.on && s < 2)
				b = puti(st, (s == 0 ? px : py) / (G.lsize[s] > 0 ?
				    G.lsize[s] : 1));
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
		case B_HELPER:
			b = 0;
			break;
		default:
			if (G.on && v == G.uv)	/* the pixel's normalized coordinate */
				b = putf(st, s == 0 ? (px + 0.5) / G.W : s == 1 ?
				    (py + 0.5) / G.H : 0);
			else if (d->sc == SC_INPUT && d->setix >= 0)
				b = spv_set[d->setix].mode == 1 ? (T(st)->k == S_FLOAT ?
				    putf(st, lane) : puti(st, lane)) : T(st)->k == S_FLOAT ?
				    putf(st, spv_set[d->setix].v) :
				    puti(st, (int64_t)spv_set[d->setix].v);
			else if ((d->sc == SC_SHARED || d->sc == SC_INPUT) &&
			    T(st)->k != S_IMAGE && T(st)->k != S_SIMAGE &&
			    T(st)->k != S_SAMPLER)
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
		} else if (!strcmp(I->op, "OpDecorate") && I->nop >= 2) {
			struct sid *x = &V.id[opid(I, 0)];
			int v = I->nop >= 3 ? atoi(I->opnd[2]) : 0;
			if (!strcmp(I->opnd[1], "Binding"))
				x->binding = v;
			else if (!strcmp(I->opnd[1], "DescriptorSet"))
				x->set = v;
			else if (!strcmp(I->opnd[1], "Location"))
				x->location = v;
			else if (!strcmp(I->opnd[1], "NonReadable"))
				x->nonread = 1;
			else if (!strcmp(I->opnd[1], "NonWritable"))
				x->nonwrite = 1;
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
			else if (!strcmp(I->op, "OpTypeMatrix"))
				y->k = S_MAT, y->e = opid(I, 0), y->n = atoi(I->opnd[1]);
			else if (!strcmp(I->op, "OpTypeSampler"))
				y->k = S_SAMPLER;
			else if (!strcmp(I->op, "OpTypeSampledImage"))
				y->k = S_SIMAGE, y->e = opid(I, 0);
			else if (!strcmp(I->op, "OpTypeImage") && I->nop >= 7) {
				static const char *const dims[] = {"1D", "2D", "3D", "Cube",
				    "Rect", "Buffer", "SubpassData"};
				int j;
				y->k = S_IMAGE;
				y->e = opid(I, 0);
				y->dim = TD_2D;
				for (j = 0; j < 7; j++)
					if (!strcmp(I->opnd[1], dims[j]))
						y->dim = (signed char)(j == 6 ? TD_2D : j);
				y->depth = (signed char)atoi(I->opnd[2]);
				y->arrayed = (signed char)atoi(I->opnd[3]);
				y->ms = (signed char)atoi(I->opnd[4]);
				y->sampled = (signed char)atoi(I->opnd[5]);
				y->ifmt = (signed char)(I->nop > 6 &&
				    strcmp(I->opnd[6], "Unknown") ?
				    tex_fmt_find(I->opnd[6]) : -1);
			} else
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
			/* room for a warp: the image run uses 32 lanes */
			d->mem = xcalloc((size_t)d->nslots * NL, sizeof *d->mem);
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
	if (y->k == S_IMAGE || y->k == S_SIMAGE || y->k == S_SAMPLER) {
		int ti = (int)(v->b[0] & 0xffff) - 1, si = (int)(v->b[0] >> 16 &
		    0xffff) - 1;
		if (ti >= 0 && si >= 0)
			snprintf(buf, n, "tex%d+smp%d", ti + 1, si + 1);
		else if (ti >= 0)
			snprintf(buf, n, "tex%d", ti + 1);
		else
			snprintf(buf, n, si >= 0 ? "smp%d" : "none", si + 1);
		return;
	}
	if (y->k == S_MAT) {
		struct lval c;
		int j, cn = ncomp(y->e);
		at = snprintf(buf, n, "[");
		for (j = 0; j < y->n && (size_t)at + 2 < n; j++) {
			memset(&c, 0, sizeof c);
			memcpy(c.b, v->b + j * cn, (size_t)cn * sizeof c.b[0]);
			fmt(buf + at, n - (size_t)at, y->e, &c);
			at += (int)strlen(buf + at);
			if (j + 1 < y->n && (size_t)at + 1 < n)
				buf[at++] = ',', buf[at] = 0;
		}
		if ((size_t)at + 1 < n)
			buf[at++] = ']', buf[at] = 0;
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
	if (!strcmp(op, "OpUDiv"))
		return C_IDIV;
	if (!strcmp(op, "OpSDiv"))
		return C_SDIV;
	if (!strcmp(op, "OpUMod"))
		return C_IREM;
	if (!strcmp(op, "OpSRem") || !strcmp(op, "OpSMod"))
		return C_SREM;
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

	if (V.quiet)
		return -1;
	if (I->res >= 0)
		snprintf(dst, sizeof dst, "%.15s", V.id[I->res].name);
	return xr_add(kind, I->text, cls, cost, mask, dst);
}

static void
emit_vals(int k, const struct sins *I, uint32_t mask)
{
	int l;
	char buf[64];

	if (k < 0 || I->res < 0 || V.id[I->res].type < 0)
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

/* ---- images and derivatives ---- */

static int
handle_tex(uint64_t h)
{
	int t = (int)(h & 0xffff) - 1;

	return t >= 0 && t < G.nt ? t : -1;
}

static const struct samp *
handle_samp(uint64_t h)
{
	static struct samp dflt;
	int s = (int)(h >> 16 & 0xffff) - 1;

	if (s >= 0 && s < G.ns)
		return &G.s[s];
	samp_default(&dflt);
	return &dflt;
}

/* the image type behind an operand (an image or a sampled image) */
static const struct sty *
image_type(int id)
{
	int t = V.id[id].type;

	if (t >= 0 && T(t)->k == S_SIMAGE)
		t = T(t)->e;
	if (t < 0 || T(t)->k != S_IMAGE)
		die("not an image: %s", V.id[id].name);
	return T(t);
}

/*
 * A derivative across the 2x2 quad, as the hardware computes it: lanes
 * are laid out `cols` wide, so the x neighbour is lane ^ 1 and the y
 * neighbour lane ^ cols.  Fine: from the lane's own row (column);
 * coarse: from the quad's top-left pixel's row (column), the same for
 * all four.  A neighbour that is masked off (divergent control flow)
 * counts as equal, and is counted for a note.
 */
static double
deriv(const double *v, int l, int axis, int coarse, uint32_t mask)
{
	int base = coarse ? l & ~1 & ~cols : l, a, b;

	if (axis == 1)
		a = base & ~1, b = a | 1;
	else
		a = base & ~cols, b = a | cols;
	if (b >= V.lanes || !(mask >> a & 1) || !(mask >> b & 1)) {
		V.divderiv++;
		return 0;
	}
	return v[b] - v[a];
}

static double
comp_f(int id, int l, int c)
{
	int t = V.id[id].type;
	struct lval x = val(id, l);

	if (t < 0)
		return 0;
	t = scalar_of(t);
	return T(t)->k == S_FLOAT ? getf(t, x.b[c]) : (double)geti(t, x.b[c]);
}

/* image operands: the mask's names, then their ids in bit order */
enum { IO_BIAS, IO_LOD, IO_GRAD, IO_COFF, IO_OFF, IO_COFFS, IO_SAMPLE,
	IO_MINLOD, IO_N };
static void
image_operands(const struct sins *I, int at, int *id)
{
	static const char *const nm[] = {"Bias", "Lod", "Grad", "ConstOffset",
	    "Offset", "ConstOffsets", "Sample", "MinLod", "MakeTexelAvailable",
	    "MakeTexelVisible"};
	unsigned m = 0;
	int k;
	char w[160], *s, *e;

	for (k = 0; k < IO_N + 1; k++)
		id[k] = -1;
	if (at >= I->nop || I->opnd[at][0] == '%')
		return;
	snprintf(w, sizeof w, "%s", I->opnd[at]);
	for (s = w; s && *s; s = e) {
		if ((e = strchr(s, '|')) != NULL)
			*e++ = 0;
		for (k = 0; k < 10; k++)
			if (!strcmp(s, nm[k]))
				m |= 1u << k;
	}
	at++;
	for (k = 0; k < 10; k++) {
		if (!(m >> k & 1))
			continue;
		if (k < IO_N && at < I->nop)
			id[k] = opid(I, at);
		if (k == IO_GRAD && at + 1 < I->nop)
			id[IO_N] = opid(I, at + 1);	/* the y gradient */
		at += k == IO_GRAD ? 2 : 1;
	}
}

static void
put_result(struct lval *r, int rt, const double *v, int n)
{
	int st = scalar_of(rt), c;

	memset(r, 0, sizeof *r);
	for (c = 0; c < n && c < MAXCOMP; c++)
		r->b[c] = T(st)->k == S_FLOAT ? putf(st, v[c]) :
		    puti(st, (int64_t)v[c]);
}

/* texel coordinates (x, y, z, layer) from an integer coordinate */
static int
texel_coords(const struct tex *t, const struct sty *y, int lv, const int64_t *c,
    int n, int *xyz)
{
	int axes = y->dim == TD_1D || y->dim == TD_BUFFER ? 1 : y->dim == TD_3D
	    ? 3 : 2, a, lw, lh, ld;

	if (!tex_level_size(t, lv, &lw, &lh, &ld))
		return 0;
	xyz[0] = xyz[1] = xyz[2] = xyz[3] = 0;
	for (a = 0; a < axes && a < n; a++)
		xyz[a] = (int)(c[a] > 1 << 30 ? 1 << 30 : c[a] < -(1 << 30) ?
		    -(1 << 30) : c[a]);
	if (y->dim == TD_CUBE)	/* (x, y, face + 6 * layer) */
		xyz[3] = (int)(n > 2 ? c[2] : 0);
	else if (y->arrayed && n > axes)
		xyz[3] = (int)c[axes];
	if (xyz[0] < 0 || xyz[0] >= lw || xyz[1] < 0 || xyz[1] >= lh ||
	    xyz[2] < 0 || xyz[2] >= ld || xyz[3] < 0 || xyz[3] >= t->layers)
		return 0;
	return 1;
}

static int
exec_image(const struct sins *I, uint32_t mask, int rt, int st, int nc)
{
	const char *op = I->op;
	int l, c, k;

	if (!strncmp(op, "OpDPd", 5) || !strncmp(op, "OpFwidth", 8)) {
		int a = opid(I, 1), fw = op[2] == 'F';
		const char *suf = op + (fw ? 8 : 6);
		int coarse = !strcmp(suf, "Coarse");
		double v[NL];
		struct lval *r;
		for (c = 0; c < nc; c++) {
			double out[NL];
			for (l = 0; l < V.lanes; l++)
				v[l] = mask >> l & 1 ? comp_f(a, l, c) : 0;
			for (l = 0; l < V.lanes; l++)
				if (mask >> l & 1) {
					double dx = deriv(v, l, 1, coarse, mask),
					    dy = deriv(v, l, 2, coarse, mask);
					out[l] = fw ? fabs(dx) + fabs(dy) : op[5] == 'x' ? dx :
					    dy;
				}
			for (l = 0; l < V.lanes; l++)
				if (mask >> l & 1) {
					r = LV(I->res, l);
					if (c == 0)
						memset(r, 0, sizeof *r);
					r->b[c] = putf(st, out[l]);
				}
		}
		k = emit(I, I_OP, C_FADD, model_active()->c[C_FADD] * nc *
		    (fw ? 3 : 1), mask);
		emit_vals(k, I, mask);
		return 1;
	}
	if (!strcmp(op, "OpSampledImage") || !strcmp(op, "OpImage")) {
		int a = opid(I, 1), b = I->nop > 2 ? opid(I, 2) : -1;
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval *r = LV(I->res, l);
				uint64_t h = val(a, l).b[0];
				memset(r, 0, sizeof *r);
				r->b[0] = op[7] == 0 ? (h & 0xffff) : (h & 0xffff) |
				    (val(b, l).b[0] & 0xffff0000u);
			}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return 1;
	}
	if (!strncmp(op, "OpImageQuery", 12)) {
		const char *q = op + 12;
		int img = opid(I, 1);
		const struct sty *y = image_type(img);
		for (l = 0; l < V.lanes; l++) {
			double v[4] = {0, 0, 0, 0};
			int ti, lv = 0, w, h, d, n = 0;
			if (!(mask >> l & 1))
				continue;
			ti = handle_tex(val(img, l).b[0]);
			if (ti < 0) {
				put_result(LV(I->res, l), rt, v, nc);
				continue;
			}
			if (!strcmp(q, "SizeLod"))
				lv = (int)comp_f(opid(I, 2), l, 0);
			if (!strcmp(q, "Levels"))
				v[n++] = G.t[ti].levels;
			else if (!strcmp(q, "Samples"))
				v[n++] = 1;
			else if (!strcmp(q, "Lod")) {
				/* (the level accessed, lambda) from the coordinate's
				 * derivatives across the quad */
				int co = opid(I, 2), nco = vcomp(co), a;
				double dx[3] = {0, 0, 0}, dy[3] = {0, 0, 0}, cv[NL], lam;
				const struct samp *s = handle_samp(val(img, l).b[0]);
				for (a = 0; a < nco && a < 3; a++) {
					int j;
					for (j = 0; j < V.lanes; j++)
						cv[j] = mask >> j & 1 ? comp_f(co, j, a) : 0;
					dx[a] = deriv(cv, l, 1, 0, mask);
					dy[a] = deriv(cv, l, 2, 0, mask);
				}
				lam = tex_lambda(&G.t[ti], nco, dx, dy) + s->bias;
				v[1] = lam;
				lam = lam < s->minlod ? s->minlod : lam > s->maxlod ?
				    s->maxlod : lam;
				v[0] = lam < 0 ? 0 : lam > G.t[ti].levels - 1 ?
				    G.t[ti].levels - 1 : lam;
				n = 2;
			} else if (tex_level_size(&G.t[ti], lv, &w, &h, &d)) {
				v[n++] = w;
				if (y->dim != TD_1D && y->dim != TD_BUFFER)
					v[n++] = h;
				if (y->dim == TD_3D)
					v[n++] = d;
				if (y->arrayed)
					v[n++] = y->dim == TD_CUBE ? G.t[ti].layers / 6 :
					    G.t[ti].layers;
			}
			put_result(LV(I->res, l), rt, v, nc);
		}
		k = emit(I, I_OP, -1, 0, mask);
		emit_vals(k, I, mask);
		return 1;
	}
	if (!strcmp(op, "OpImageTexelPointer")) {
		/* a pointer to one texel's first word, for an atomic */
		int p = opid(I, 1), co = opid(I, 2);
		for (l = 0; l < V.lanes; l++) {
			struct lval pv, *r;
			int ti, xyz[4], n = vcomp(co), a;
			int64_t ic[4] = {0, 0, 0, 0};
			if (!(mask >> l & 1))
				continue;
			pv = val(p, l);
			ti = pv.var >= 0 ? handle_tex(*slot(pv.var, l, pv.off)) : -1;
			r = LV(I->res, l);
			memset(r, 0, sizeof *r);
			r->var = ti >= 0 ? -2 - ti : -1;
			r->off = -1;
			for (a = 0; a < n && a < 4; a++)
				ic[a] = (int64_t)comp_f(co, l, a);
			if (ti >= 0) {
				int it = T(V.id[pv.var].type)->e;
				const struct sty *y;
				while (T(it)->k == S_ARRAY)
					it = T(it)->e;
				y = T(it);
				if (texel_coords(&G.t[ti], y, 0, ic, n, xyz))
					r->off = (int)(tex_word(&G.t[ti], 0, xyz[0], xyz[1],
					    xyz[2], xyz[3]) - G.t[ti].t);
				else
					G.oob++;
			}
		}
		k = emit(I, I_OP, C_IADD, model_active()->c[C_IADD], mask);
		return 1;
	}
	if (!strcmp(op, "OpImageFetch") || !strcmp(op, "OpImageRead") ||
	    !strcmp(op, "OpImageWrite")) {
		int wr = op[7] == 'W', base = wr ? 0 : 1;
		int img = opid(I, base), co = opid(I, base + 1), ids[IO_N + 1];
		const struct sty *y = image_type(img);
		int n = vcomp(co);
		image_operands(I, base + (wr ? 3 : 2), ids);
		for (l = 0; l < V.lanes; l++) {
			int ti, xyz[4], lv = 0, a;
			int64_t ic[4] = {0, 0, 0, 0};
			double v[4] = {0, 0, 0, 0};
			if (!(mask >> l & 1))
				continue;
			ti = handle_tex(val(img, l).b[0]);
			for (a = 0; a < n && a < 4; a++)
				ic[a] = (int64_t)comp_f(co, l, a);
			if (ids[IO_LOD] >= 0)
				lv = (int)comp_f(ids[IO_LOD], l, 0);
			for (a = 0; a < 3; a++) {
				int oid = ids[IO_COFF] >= 0 ? ids[IO_COFF] : ids[IO_OFF];
				if (oid >= 0 && a < vcomp(oid))
					ic[a] += (int64_t)comp_f(oid, l, a);
			}
			if (ti < 0 || !texel_coords(&G.t[ti], y, lv, ic, n, xyz)) {
				G.oob++;
				if (!wr) {	/* robust: zeros, alpha 1 if the format has none */
					v[3] = ti >= 0 && tex_fmts[G.t[ti].fmt].nc < 4 ? 1 : 0;
					put_result(LV(I->res, l), rt, v, nc);
				}
				continue;
			}
			if (wr) {
				int tx = opid(I, 2), m = vcomp(tx);
				for (a = 0; a < 4; a++)
					v[a] = a < m ? comp_f(tx, l, a) : a == 3 ? 1 : 0;
				tex_store(&G.t[ti], lv, xyz[0], xyz[1], xyz[2], xyz[3], v);
				G.written[ti] = 1;
			} else {
				tex_load(&G.t[ti], lv, xyz[0], xyz[1], xyz[2], xyz[3], v);
				put_result(LV(I->res, l), rt, v, nc);
			}
		}
		k = emit(I, I_OP, wr ? C_STORE : C_LOAD, model_active()->c[wr ?
		    C_STORE : C_LOAD], mask);
		emit_vals(k, I, mask);
		return 1;
	}
	if (!strncmp(op, "OpImageSample", 13) || !strcmp(op, "OpImageGather") ||
	    !strcmp(op, "OpImageDrefGather")) {
		int proj = strstr(op, "Proj") != NULL, dref = strstr(op, "Dref") !=
		    NULL, gather = strstr(op, "Gather") != NULL;
		int explicit_lod = strstr(op, "ExplicitLod") != NULL;
		int img = opid(I, 1), co = opid(I, 2), at = 3, ids[IO_N + 1];
		int dref_id = -1, comp_id = -1;
		const struct sty *y = image_type(img);
		int axes = y->dim == TD_1D || y->dim == TD_BUFFER ? 1 : y->dim ==
		    TD_3D || y->dim == TD_CUBE ? 3 : 2;
		int nco = vcomp(co);
		double cc[4][NL];
		if (dref || (gather && !dref)) {
			if (dref)
				dref_id = opid(I, at++);
			else
				comp_id = opid(I, at++);
		}
		image_operands(I, at, ids);
		/* every lane's coordinate first: the derivatives need the quad */
		for (c = 0; c < 4; c++)
			for (l = 0; l < V.lanes; l++)
				cc[c][l] = (mask >> l & 1) && c < nco ? comp_f(co, l, c) : 0;
		if (proj)
			for (l = 0; l < V.lanes; l++) {
				double q = cc[nco - 1][l];
				for (c = 0; c < axes; c++)
					cc[c][l] = q != 0 ? cc[c][l] / q : cc[c][l];
			}
		for (l = 0; l < V.lanes; l++) {
			struct texreq rq;
			const struct samp *s;
			struct samp s2;
			double out[4] = {0, 0, 0, 0}, dx[3] = {0, 0, 0},
			    dy[3] = {0, 0, 0};
			int ti, fc;
			if (!(mask >> l & 1))
				continue;
			ti = handle_tex(val(img, l).b[0]);
			if (ti < 0) {
				put_result(LV(I->res, l), rt, out, nc);
				continue;
			}
			s = handle_samp(val(img, l).b[0]);
			fc = tex_fmts[G.t[ti].fmt].cls;
			if (fc == TC_UINT || fc == TC_SINT) {
				/* integer formats cannot be filtered: nearest */
				s2 = *s;
				s2.mag = s2.min = s2.mip = SF_NEAREST;
				s = &s2;
			}
			memset(&rq, 0, sizeof rq);
			rq.nc = axes;
			rq.arrayed = y->arrayed;
			for (c = 0; c < 4; c++)
				rq.c[c] = cc[c][l];
			rq.gather = gather ? (comp_id >= 0 ? (int)comp_f(comp_id, l, 0) &
			    3 : 0) : -1;
			if (dref_id >= 0) {
				rq.dref = 1;
				rq.ref = comp_f(dref_id, l, 0);
			}
			{
				int oid = ids[IO_COFF] >= 0 ? ids[IO_COFF] : ids[IO_OFF];
				for (c = 0; c < 3 && oid >= 0; c++)
					rq.off[c] = c < vcomp(oid) ? (int)comp_f(oid, l, c) : 0;
			}
			if (ids[IO_LOD] >= 0)
				rq.lod = comp_f(ids[IO_LOD], l, 0);
			else {
				if (ids[IO_GRAD] >= 0) {
					for (c = 0; c < axes; c++) {
						dx[c] = c < vcomp(ids[IO_GRAD]) ?
						    comp_f(ids[IO_GRAD], l, c) : 0;
						dy[c] = c < vcomp(ids[IO_N]) ?
						    comp_f(ids[IO_N], l, c) : 0;
					}
				} else if (!explicit_lod)
					for (c = 0; c < axes; c++) {
						dx[c] = deriv(cc[c], l, 1, 0, mask);
						dy[c] = deriv(cc[c], l, 2, 0, mask);
					}
				if (s->unnorm)	/* coordinates in texels already */
					rq.lod = 0;
				else
					rq.lod = tex_lambda(&G.t[ti], y->dim == TD_CUBE ? 2 :
					    axes, dx, dy);
			}
			rq.lod += s->bias + (ids[IO_BIAS] >= 0 ?
			    comp_f(ids[IO_BIAS], l, 0) : 0);
			if (ids[IO_MINLOD] >= 0) {
				double ml = comp_f(ids[IO_MINLOD], l, 0);
				if (rq.lod < ml)
					rq.lod = ml;
			}
			tex_sample(&G.t[ti], s, &rq, out);
			G.nsample++;
			put_result(LV(I->res, l), rt, out, nc);
		}
		k = emit(I, I_OP, C_LOAD, model_active()->c[C_LOAD], mask);
		emit_vals(k, I, mask);
		return 1;
	}
	return 0;
}

/* ---- GLSL.std.450 on whole vectors and matrices ---- */

static double
half_round(double v)
{
	/* to binary16 and back, round to nearest even */
	int e;
	double a = fabs(v), q;

	if (isnan(v) || a == 0)
		return v;
	if (a >= 65520)
		return v < 0 ? -INFINITY : INFINITY;
	frexp(a, &e);
	e -= 1;
	if (e < -14)
		e = -14;
	q = ldexp(nearbyint(ldexp(a, 10 - e)), e - 10);
	return v < 0 ? -q : q;
}

static uint32_t
half_bits(double v)
{
	float f = (float)half_round(v);
	uint32_t u, s, e, m;

	memcpy(&u, &f, 4);
	s = u >> 16 & 0x8000;
	e = u >> 23 & 0xff;
	m = u & 0x7fffff;
	if (e == 0xff)
		return s | 0x7c00 | (m ? 0x200 : 0);
	if (e == 0)
		return s;
	if ((int)e - 127 < -14) {	/* subnormal */
		int sh = -14 - ((int)e - 127);
		return s | ((m | 0x800000) >> (13 + sh));
	}
	return s | ((e - 127 + 15) << 10) | (m >> 13);
}

static double
half_value(uint32_t h)
{
	int e = h >> 10 & 0x1f, m = h & 0x3ff;
	double v = e == 0 ? ldexp(m, -24) : e == 31 ? (m ? NAN : INFINITY) :
	    ldexp(m | 0x400, e - 25);

	return h & 0x8000 ? -v : v;
}

static double
det(const double *m, int n)
{
	if (n == 2)
		return m[0] * m[3] - m[2] * m[1];
	if (n == 3)
		return m[0] * (m[4] * m[8] - m[7] * m[5]) -
		    m[3] * (m[1] * m[8] - m[7] * m[2]) +
		    m[6] * (m[1] * m[5] - m[4] * m[2]);
	{
		double s = 0, sub[9];
		int c, i, j, k;
		for (c = 0; c < 4; c++) {	/* along column c's row 0 */
			for (i = 0, k = 0; i < 4; i++)
				if (i != c)
					for (j = 1; j < 4; j++)
						sub[k++] = m[i * 4 + j];
			s += (c & 1 ? -1 : 1) * m[c * 4] * det(sub, 3);
		}
		return s;
	}
}

static int
exec_glslvec(const struct sins *I, uint32_t mask, int rt, int st, int nc)
{
	static const char *const fs[] = {"Length", "Distance", "Normalize",
	    "Cross", "Reflect", "Refract", "FaceForward", "Determinant",
	    "MatrixInverse", "PackUnorm4x8", "PackSnorm4x8", "PackUnorm2x16",
	    "PackSnorm2x16", "PackHalf2x16", "UnpackUnorm4x8", "UnpackSnorm4x8",
	    "UnpackUnorm2x16", "UnpackSnorm2x16", "UnpackHalf2x16", NULL};
	const char *f;
	int fi, l, i, j, k, a, b = -1, c = -1, n;

	if (strcmp(I->op, "OpExtInst") || I->nop < 4)
		return 0;
	f = I->opnd[2];
	for (fi = 0; fs[fi] && strcmp(fs[fi], f); fi++)
		;
	if (!fs[fi])
		return 0;
	a = opid(I, 3);
	if (I->nop > 4)
		b = opid(I, 4);
	if (I->nop > 5)
		c = opid(I, 5);
	n = vcomp(a);
	for (l = 0; l < V.lanes; l++) {
		double x[MAXCOMP], y[MAXCOMP], z[MAXCOMP], r[MAXCOMP];
		int nr = nc, at = V.id[a].type, sa = at >= 0 ? scalar_of(at) : st;
		struct lval xv, *o;
		if (!(mask >> l & 1))
			continue;
		xv = val(a, l);
		memset(r, 0, sizeof r);
		for (i = 0; i < MAXCOMP; i++) {
			x[i] = i < nwords(at) && T(sa)->k == S_FLOAT ? getf(sa, xv.b[i]) :
			    0;
			y[i] = b >= 0 && i < nwords(V.id[b].type) ? comp_f(b, l, i) : 0;
			z[i] = c >= 0 && i < nwords(V.id[c].type) ? comp_f(c, l, i) : 0;
		}
		switch (fi) {
		case 0: case 1: {	/* Length, Distance */
			double s = 0;
			for (i = 0; i < n; i++)
				s += (x[i] - (fi ? y[i] : 0)) * (x[i] - (fi ? y[i] : 0));
			r[0] = sqrt(s);
			break;
		}
		case 2: {	/* Normalize */
			double s = 0;
			for (i = 0; i < n; i++)
				s += x[i] * x[i];
			s = sqrt(s);
			for (i = 0; i < n; i++)
				r[i] = x[i] / s;
			break;
		}
		case 3:		/* Cross */
			r[0] = x[1] * y[2] - y[1] * x[2];
			r[1] = x[2] * y[0] - y[2] * x[0];
			r[2] = x[0] * y[1] - y[0] * x[1];
			break;
		case 4: {	/* Reflect: I - 2 dot(N, I) N */
			double d = 0;
			for (i = 0; i < n; i++)
				d += x[i] * y[i];
			for (i = 0; i < n; i++)
				r[i] = x[i] - 2 * d * y[i];
			break;
		}
		case 5: {	/* Refract(I, N, eta) */
			double d = 0, e = comp_f(c, l, 0), kk;
			for (i = 0; i < n; i++)
				d += x[i] * y[i];
			kk = 1 - e * e * (1 - d * d);
			for (i = 0; i < n; i++)
				r[i] = kk < 0 ? 0 : e * x[i] - (e * d + sqrt(kk)) * y[i];
			break;
		}
		case 6: {	/* FaceForward(N, I, Nref) */
			double d = 0;
			for (i = 0; i < n; i++)
				d += z[i] * y[i];
			for (i = 0; i < n; i++)
				r[i] = d < 0 ? x[i] : -x[i];
			break;
		}
		case 7: case 8: {	/* Determinant, MatrixInverse */
			int m = at >= 0 && T(at)->k == S_MAT ? T(at)->n : 2;
			double dd = det(x, m), sub[9];
			if (fi == 7) {
				r[0] = dd;
				break;
			}
			/* inverse: the adjugate over the determinant */
			for (i = 0; i < m; i++)
				for (j = 0; j < m; j++) {
					int ii, jj, q = 0;
					for (ii = 0; ii < m; ii++)
						for (jj = 0; jj < m; jj++)
							if (ii != j && jj != i)
								sub[q++] = x[ii * m + jj];
					r[i * m + j] = ((i + j) & 1 ? -1 : 1) * (m == 2 ? sub[0] :
					    det(sub, m - 1)) / dd;
				}
			nr = m * m;
			break;
		}
		case 9: case 10: case 11: case 12: case 13: {	/* Pack* */
			uint64_t u = 0;
			int cnt = fi <= 10 ? 4 : 2, bits = fi <= 10 ? 8 : 16;
			for (i = 0; i < cnt; i++) {
				double v = x[i];
				uint64_t q;
				if (fi == 13)
					q = half_bits(v);
				else if (fi == 9 || fi == 11) {
					v = v < 0 ? 0 : v > 1 ? 1 : v;
					q = (uint64_t)nearbyint(v * (double)((1 << bits) - 1));
				} else {
					v = v < -1 ? -1 : v > 1 ? 1 : v;
					q = (uint64_t)(int64_t)nearbyint(v * (double)((1 <<
					    (bits - 1)) - 1)) & ((1u << bits) - 1);
				}
				u |= q << (i * bits);
			}
			r[0] = (double)(u & 0xffffffffu);
			nr = 1;
			break;
		}
		default: {	/* Unpack* */
			uint32_t u = (uint32_t)(xv.b[0] & 0xffffffffu);
			int cnt = fi <= 15 ? 4 : 2, bits = fi <= 15 ? 8 : 16;
			for (i = 0; i < cnt; i++) {
				uint32_t q = u >> (i * bits) & ((1u << bits) - 1);
				if (fi == 18)
					r[i] = half_value(q);
				else if (fi == 14 || fi == 16)
					r[i] = q / (double)((1u << bits) - 1);
				else {
					int sv = (int)q - (q >> (bits - 1) ? 1 << bits : 0);
					double v = sv / (double)((1 << (bits - 1)) - 1);
					r[i] = v < -1 ? -1 : v;
				}
			}
			nr = cnt;
		}
		}
		o = LV(I->res, l);
		memset(o, 0, sizeof *o);
		for (k = 0; k < nr && k < MAXCOMP; k++)
			o->b[k] = T(st)->k == S_FLOAT ? putf(st, r[k]) :
			    puti(st, (int64_t)r[k]);
	}
	k = emit(I, I_OP, fi <= 2 ? C_FMUL : fi <= 8 ? C_FMUL : C_CONV,
	    model_active()->c[fi <= 8 ? C_FMUL : C_CONV] * (fi <= 2 ? 3 : 2),
	    mask);
	emit_vals(k, I, mask);
	return 1;
}

/* ---- matrices ---- */

static int
exec_matrix(const struct sins *I, uint32_t mask, int rt, int st)
{
	const char *op = I->op;
	int l, i, j, q, k, n = 0;

	if (strcmp(op, "OpTranspose") && strcmp(op, "OpMatrixTimesScalar") &&
	    strcmp(op, "OpVectorTimesMatrix") && strcmp(op,
	    "OpMatrixTimesVector") && strcmp(op, "OpMatrixTimesMatrix") &&
	    strcmp(op, "OpOuterProduct"))
		return 0;
	for (l = 0; l < V.lanes; l++) {
		int a = opid(I, 1), b = I->nop > 2 ? opid(I, 2) : a;
		int at = V.id[a].type, bt = V.id[b].type;
		struct lval x, y, *r;
		double v[MAXCOMP];
		if (!(mask >> l & 1))
			continue;
		x = val(a, l);
		y = val(b, l);
		memset(v, 0, sizeof v);
#define AF(m, i) getf(st, (m).b[i])
		if (!strcmp(op, "OpTranspose")) {
			/* a: C columns of R; the result: R columns of C */
			int C = T(at)->n, R = ncomp(T(at)->e);
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					v[j * C + i] = AF(x, i * R + j);
			n = C * R;
		} else if (!strcmp(op, "OpMatrixTimesScalar")) {
			n = ncomp(at);
			for (i = 0; i < n; i++)
				v[i] = AF(x, i) * AF(y, 0);
		} else if (!strcmp(op, "OpVectorTimesMatrix")) {
			/* v (R) * M (C columns of R): C results, dot with columns */
			int C = T(bt)->n, R = ncomp(T(bt)->e);
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					v[i] += AF(x, j) * AF(y, i * R + j);
			n = C;
		} else if (!strcmp(op, "OpMatrixTimesVector")) {
			int C = T(at)->n, R = ncomp(T(at)->e);
			for (j = 0; j < R; j++)
				for (i = 0; i < C; i++)
					v[j] += AF(x, i * R + j) * AF(y, i);
			n = R;
		} else if (!strcmp(op, "OpMatrixTimesMatrix")) {
			/* A (K columns of R) * B (C columns of K) */
			int K = T(at)->n, R = ncomp(T(at)->e), C = T(bt)->n;
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					for (q = 0; q < K; q++)
						v[i * R + j] += AF(x, q * R + j) * AF(y, i * K + q);
			n = C * R;
		} else {	/* OuterProduct: column i is a * b[i] */
			int R = vcomp(a), C = vcomp(b);
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					v[i * R + j] = AF(x, j) * AF(y, i);
			n = C * R;
		}
#undef AF
		r = LV(I->res, l);
		memset(r, 0, sizeof *r);
		for (i = 0; i < n && i < MAXCOMP; i++)
			r->b[i] = putf(st, v[i]);
	}
	k = emit(I, I_OP, C_FMUL, model_active()->c[C_FMUL] * (n > 0 ? n : 1),
	    mask);
	emit_vals(k, I, mask);
	return 1;
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
static int exec_idot(const struct sins *I, uint32_t mask, int rt);

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

	if (!strcmp(I->op, "OpDemoteToHelperInvocation") ||
	    !strcmp(I->op, "OpDemoteToHelperInvocationEXT")) {
		V.killed |= mask;
		emit(I, I_OP, -1, 0, mask);
		return;
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
			for (i = 0; i < nwords(pt); i++)
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
			for (i = 0; i < nwords(rt); i++)
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
				} else if (y->k == S_ARRAY || y->k == S_MAT) {
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
		int c = opid(I, 1), pt, off, n;
		off = part_of(V.id[c].type, I, 2, &pt);
		n = nwords(pt);
		if (off + n > MAXCOMP)
			off = 0, n = 1;
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(c, l), *r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				memcpy(r->b, x.b + off, (size_t)n * sizeof r->b[0]);
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
					int c = opid(I, i), cn = nwords(type_of(c) >= 0 ?
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
		int obj = opid(I, 1), c = opid(I, 2), pt, off, n;
		off = part_of(V.id[c].type, I, 3, &pt);
		n = nwords(pt);
		if (off + n > MAXCOMP)
			off = 0, n = 1;
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(c, l), o = val(obj, l), *r = LV(I->res, l);
				*r = x;
				memcpy(r->b + off, o.b, (size_t)n * sizeof r->b[0]);
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
					r->b[i - 3] = s < na ? x.b[s & 15] : y.b[(s - na) & 15];
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
		sc = pv.var >= 0 ? V.id[pv.var].sc : SC_SHARED;
		for (l = 0; l < V.lanes; l++) {
			struct lval ptr, *r;
			uint64_t *m, old, x = 0, tw = 0;
			uint32_t *texw = NULL;
			if (!(mask >> l & 1))
				continue;
			ptr = val(p, l);
			if (ptr.var < 0) {	/* a texel (OpImageTexelPointer) */
				int ti = -ptr.var - 2;
				if (ti >= 0 && ptr.off >= 0) {
					texw = G.t[ti].t + ptr.off;
					tw = *texw;
					G.written[ti] = 1;
				}
				m = &tw;
			} else
				m = slot(ptr.var, l, ptr.off);
			old = *m;
			if (!strcmp(a, "Store"))
				x = val(opid(I, 3), l).b[0];
			else if (I->nop > 4)
				x = val(opid(I, 4), l).b[0];
			st = rt >= 0 ? rt : ptr.var < 0 ? type_of(opid(I, 3)) :
			    V.id[ptr.var].slotty[ptr.off >= 0 && ptr.off <
			    V.id[ptr.var].nslots ? ptr.off : 0];
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
			if (texw != NULL)
				*texw = (uint32_t)*m;
			if (I->res >= 0) {
				r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				r->b[0] = old;
			}
			/* one lane at a time */
			k = emit(I, I_ATOMIC, sc == SC_WG ? C_ATOM_SH : C_ATOM_GL,
			    model_active()->c[sc == SC_WG ? C_ATOM_SH : C_ATOM_GL],
			    1u << l);
			if (I->res >= 0 && k >= 0) {
				char buf[64];
				fmt(buf, sizeof buf, V.id[I->res].type, LV(I->res, l));
				xr_val(k, l, buf);
			}
		}
		return;
	}
	if (exec_image(I, mask, rt, st, nc) || exec_matrix(I, mask, rt, st) ||
	    exec_glslvec(I, mask, rt, st, nc) || exec_idot(I, mask, rt))
		return;
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

/*
 * SPV_KHR_integer_dot_product: OpUDot, OpSDot, OpSUDot and their AccSat
 * forms, on 32-bit words of 4 x 8-bit lanes (PackedVectorFormat4x8Bit)
 * or on integer vectors.  The plain forms give the exact sum's low bits
 * in the result's width, the AccSat forms the exact sum plus the
 * accumulator, saturated (unsigned for OpUDotAccSat, signed otherwise).
 * Priced by the model's class for the form (dot.c), or, where the model
 * has none, as the multiplies and adds it stands for.
 */
static void
idot_shape(const struct sins *I, int *n, int *w, int *packed)
{
	int a = opid(I, 1), at = type_of(a) >= 0 ? type_of(a) : V.id[a].type;

	*packed = !strcmp(I->opnd[I->nop - 1], "PackedVectorFormat4x8Bit") ||
	    !strcmp(I->opnd[I->nop - 1], "PackedVectorFormat4x8BitKHR");
	if (*packed || T(at)->k != S_VEC) {
		*n = 4;
		*w = 8;
		return;
	}
	*n = T(at)->n;
	*w = T(scalar_of(at))->w;
}

static int
idot_parts(const char *op, int *su, int *sa, int *sb, int *sat)
{
	const char *r;

	if (!strncmp(op, "OpUDot", 6))
		*sa = *sb = 0, r = op + 6, *su = 0;
	else if (!strncmp(op, "OpSUDot", 7))
		*sa = 1, *sb = 0, r = op + 7, *su = 1;
	else if (!strncmp(op, "OpSDot", 6))
		*sa = *sb = 1, r = op + 6, *su = 0;
	else
		return 0;
	if (!strncmp(r, "AccSat", 6))
		*sat = 1, r += 6;
	else
		*sat = 0;
	return *r == 0 || !strcmp(r, "KHR");
}

static double
idot_cost(const struct sins *I, int *cls)
{
	int n, w, packed, su, sa, sb, sat, op;
	const struct model *m = model_active();

	idot_parts(I->op, &su, &sa, &sb, &sat);
	idot_shape(I, &n, &w, &packed);
	op = dot_op(n, w, sa, sb, sat);
	if (op && cclass_available(m, DOT_CLASS(op))) {
		*cls = DOT_CLASS(op);
		return m->c[*cls];
	}
	/* the lanes taken apart, multiplied and added */
	*cls = C_IMUL;
	return n * (m->c[C_IMUL] + m->c[C_IADD]) + (packed ? 2 * n *
	    m->c[C_SHIFT] : 0) + (sat ? 2 * m->c[C_MINMAX] : 0);
}

static int
exec_idot(const struct sins *I, uint32_t mask, int rt)
{
	int su, sa, sb, sat, n, w, packed, l, i, k, cls, a, b, acc = -1;
	int rw, rs;
	double cost;

	if (!idot_parts(I->op, &su, &sa, &sb, &sat) || rt < 0)
		return 0;
	idot_shape(I, &n, &w, &packed);
	a = opid(I, 1);
	b = opid(I, 2);
	if (sat)
		acc = opid(I, 3);
	rw = T(rt)->w;
	rs = sa || sb;
	(void)su;
	for (l = 0; l < V.lanes; l++) {
		struct lval x, y, *r;
		int64_t sum = 0;
		if (!(mask >> l & 1))
			continue;
		x = val(a, l);
		y = val(b, l);
		for (i = 0; i < n; i++) {
			uint64_t ux = packed ? x.b[0] >> (8 * i) : x.b[i];
			uint64_t uy = packed ? y.b[0] >> (8 * i) : y.b[i];
			int64_t vx = (int64_t)(ux & wmask(w)), vy = (int64_t)(uy & wmask(w));
			if (sa && w < 64 && (vx >> (w - 1) & 1))
				vx -= (int64_t)1 << w;
			if (sb && w < 64 && (vy >> (w - 1) & 1))
				vy -= (int64_t)1 << w;
			sum += vx * vy;
		}
		r = LV(I->res, l);
		memset(r, 0, sizeof *r);
		if (sat) {
			int64_t c = geti(rt, val(acc, l).b[0]), lo, hi;
			if (!rs)
				c = (int64_t)(val(acc, l).b[0] & wmask(rw));
			lo = rs ? (rw >= 64 ? INT64_MIN : -((int64_t)1 << (rw - 1))) : 0;
			hi = rs ? (rw >= 64 ? INT64_MAX : ((int64_t)1 << (rw - 1)) - 1) :
			    (rw >= 63 ? INT64_MAX : ((int64_t)1 << rw) - 1);
			sum += c;
			sum = sum < lo ? lo : sum > hi ? hi : sum;
		}
		r->b[0] = (uint64_t)sum & wmask(rw);
	}
	cost = idot_cost(I, &cls);
	k = emit(I, I_OP, cls, cost, mask);
	emit_vals(k, I, mask);
	return 1;
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
			if (strcmp(I->op, "OpReturn"))
				V.killed |= mask;
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

/* ---- image inputs and the image run ---- */

/* a texture's level 0 checksum (FNV-1a over its channels' words) */
static uint32_t
tex_sum(const struct tex *t)
{
	const struct texfmt *f = &tex_fmts[t->fmt];
	size_t n = t->levels > 1 ? t->lo[1] : t->ntexel, i;
	uint32_t h = 2166136261u, w;
	int c, b;

	for (i = 0; i < n; i++)
		for (c = 0; c < f->nc; c++)
			for (w = t->t[4 * i + c], b = 0; b < 4; b++, w >>= 8)
				h = (h ^ (w & 0xff)) * 16777619u;
	return h;
}

static const char *
samp_text(const struct samp *s)
{
	static char b[80];
	static const char *const am[] = {"repeat", "mirrored repeat",
	    "clamp to edge", "clamp to border", "mirror clamp to edge"};

	snprintf(b, sizeof b, "%s filtering, %s mips, %s", s->mag == SF_NEAREST ?
	    "nearest" : "linear", s->mip == SF_NEAREST ? "nearest" : "linear",
	    am[s->addr[0] % 5]);
	return b;
}

/* the element image type of a binding variable, -1 if not an image */
static int
binding_elem(int var, int *count)
{
	int t;

	*count = 1;
	if (V.id[var].kind != K_VAR || V.id[var].type < 0)
		return -1;
	t = T(V.id[var].type)->e;
	while (T(t)->k == S_ARRAY) {
		*count *= T(t)->n;
		t = T(t)->e;
	}
	if (*count > 16)
		*count = 16;
	return T(t)->k == S_IMAGE || T(t)->k == S_SIMAGE ||
	    T(t)->k == S_SAMPLER ? t : -1;
}

static void
samp_from_opts(struct samp *s, const struct imgin *in)
{
	samp_default(s);
	if (img_opt.filt >= 0)
		s->mag = s->min = (unsigned char)img_opt.filt;
	if (img_opt.mipf >= 0)
		s->mip = (unsigned char)img_opt.mipf;
	else if (img_opt.filt >= 0)
		s->mip = (unsigned char)img_opt.filt;
	if (img_opt.addr >= 0)
		s->addr[0] = s->addr[1] = s->addr[2] = (unsigned char)img_opt.addr;
	if (img_opt.border >= 0)
		s->border = (unsigned char)img_opt.border;
	if (in != NULL && in->filt >= 0)
		s->mag = s->min = s->mip = (unsigned char)in->filt;
	if (in != NULL && in->addr >= 0)
		s->addr[0] = s->addr[1] = s->addr[2] = (unsigned char)in->addr;
}

static const char *
kind_name(const struct imgin *in)
{
	static char b[24];
	const char *n;

	switch (in->kind) {
	case IK_COLOR:
		if ((n = img_color_name(in->rgba)) != NULL)
			return n;
		snprintf(b, sizeof b, "#%02x%02x%02x%02x", in->rgba[0], in->rgba[1],
		    in->rgba[2], in->rgba[3]);
		return b;
	case IK_GRADIENT:
		return "gradient";
	case IK_CHECKER:
		return "checkerboard";
	}
	return "noise";
}

/* read a file for `input N file PATH` (not in the bot: no files there) */
static unsigned char *
read_file(const char *path, size_t *n)
{
	FILE *fp;
	unsigned char *d = NULL;
	size_t cap = 0, r;

	*n = 0;
	if (nlp_defer)
		fail("animate exec: input files are read on the command line; in "
		    "the chat, attach the images");
	if ((fp = fopen(path, "rb")) == NULL)
		fail("animate exec: cannot open %s", path);
	for (;;) {
		if (*n == cap) {
			if (cap >= IMG_MAXFILE) {
				fclose(fp);
				free(d);
				fail("animate exec: %s is over %u MiB", path,
				    IMG_MAXFILE >> 20);
			}
			cap = cap ? cap * 2 : 65536;
			d = xrealloc(d, cap);
		}
		if ((r = fread(d + *n, 1, cap - *n, fp)) == 0)
			break;
		*n += r;
	}
	fclose(fp);
	return d;
}

/*
 * One input slot's pixels: a file, a gallery image, or a pattern, as
 * 8-bit RGBA (w, h); the label says which
 */
static unsigned gal_used;	/* the gallery's images taken */

static unsigned char *
input_pixels(int slot, const struct imgin *in, int *gal, uint64_t seed,
    int *w, int *h, char *what, size_t wn, struct sbuf *said)
{
	unsigned char *px = NULL;
	struct imgdec im;
	char err[160];
	int kind = in->kind;

	if (kind == IK_GALLERY && in->gal > 0 && in->gal > img_opt.ngal)
		fail("animate exec: input %d: there is no image %d", slot + 1,
		    in->gal);
	if (kind == IK_DEFAULT && *gal < img_opt.ngal)
		kind = IK_GALLERY;
	if (kind == IK_FILE || kind == IK_GALLERY) {
		const unsigned char *d;
		size_t n;
		unsigned char *own = NULL;
		const char *nm;
		if (kind == IK_FILE) {
			own = read_file(in->file, &n);
			d = own;
			nm = in->file;
		} else if (in->gal > 0) {	/* a given one of the gallery */
			gal_used |= 1u << (in->gal - 1);
			d = img_opt.gal[in->gal - 1].data;
			n = img_opt.gal[in->gal - 1].n;
			nm = img_opt.gal[in->gal - 1].name;
		} else {
			d = img_opt.gal[*gal].data;
			n = img_opt.gal[*gal].n;
			nm = img_opt.gal[*gal].name;
			gal_used |= 1u << *gal;
			(*gal)++;
		}
		if (!img_decode(d, n, &im, err, sizeof err)) {
			free(own);
			if (kind == IK_FILE)
				fail("animate exec: %s: %s", in->file, err);
			sb_printf(said, "  note: input %d: %s: %s; noise instead\n",
			    slot + 1, nm, err);
			kind = IK_NOISE;
		} else {
			free(own);
			{
				const char *b = strrchr(nm, '/');
				snprintf(what, wn, "%s", b ? b + 1 : nm);
			}
			*w = im.w;
			*h = im.h;
			if (im.w > TEX_MAXSIDE || im.h > TEX_MAXSIDE) {
				/* box-filtered down, the aspect kept */
				double s = (double)TEX_MAXSIDE / (im.w > im.h ? im.w :
				    im.h);
				*w = (int)(im.w * s + 0.5) > 0 ? (int)(im.w * s + 0.5) : 1;
				*h = (int)(im.h * s + 0.5) > 0 ? (int)(im.h * s + 0.5) : 1;
				px = xmalloc((size_t)*w * *h * 4);
				img_downscale(im.rgba, im.w, im.h, px, *w, *h);
				sb_printf(said, "  note: input %d: %s is %dx%d, scaled down "
				    "to %dx%d\n", slot + 1, what, im.w, im.h, *w, *h);
			} else {
				px = xmalloc((size_t)im.w * im.h * 4);
				memcpy(px, im.rgba, (size_t)im.w * im.h * 4);
			}
			img_free(&im);
			return px;
		}
	}
	*w = in->w > 0 ? in->w : img_opt.w > 0 ? img_opt.w : 64;
	*h = in->h > 0 ? in->h : img_opt.h > 0 ? img_opt.h : 64;
	if (*w > TEX_MAXSIDE)
		*w = TEX_MAXSIDE;
	if (*h > TEX_MAXSIDE)
		*h = TEX_MAXSIDE;
	px = xmalloc((size_t)*w * *h * 4);
	{
		struct imgin k2 = *in;
		k2.kind = kind;
		img_pattern(px, *w, *h, kind, in->rgba, seed + (uint64_t)slot);
		snprintf(what, wn, "%s", kind_name(&k2));
	}
	return px;
}

/*
 * Textures for every image binding, in (set, binding) order, then
 * declaration order; sampled images and readable storage images are the
 * input slots.  Returns 1 when the shader has images.
 */
static int
img_setup(uint64_t seed, int frag, struct sbuf *said)
{
	int vars[64], nv = 0, i, j, slot = 0, gal = 0;
	int firstw = 0, firsth = 0;

	memset(&G, 0, sizeof G);
	G.uv = -1;
	gal_used = 0;
	for (i = 0; i < V.nid && nv < 64; i++) {
		int cnt;
		if (V.id[i].kind == K_VAR && binding_elem(i, &cnt) >= 0)
			vars[nv++] = i;
	}
	/* by (set, binding); unbound ones after, as declared */
	for (i = 1; i < nv; i++)
		for (j = i; j > 0; j--) {
			struct sid *a = &V.id[vars[j - 1]], *b = &V.id[vars[j]];
			long ka = a->binding < 0 ? 1L << 40 : (long)(a->set < 0 ? 0 :
			    a->set) << 20 | a->binding;
			long kb = b->binding < 0 ? 1L << 40 : (long)(b->set < 0 ? 0 :
			    b->set) << 20 | b->binding;
			if (kb < ka) {
				int t = vars[j];
				vars[j] = vars[j - 1];
				vars[j - 1] = t;
			} else
				break;
		}
	for (i = 0; i < nv; i++) {
		struct sid *d = &V.id[vars[i]];
		int cnt, et = binding_elem(vars[i], &cnt), e;
		for (e = 0; e < cnt && e < d->nslots; e++) {
			const struct sty *y;
			const struct imgin *in;
			int it, storage, w, h, fmt, depth, layers, ti;
			unsigned char *px;
			char what[96], bind[48];
			uint64_t hnd = 0;
			if (T(et)->k == S_SAMPLER) {
				if (G.ns < MAXTEX) {
					samp_from_opts(&G.s[G.ns], NULL);
					hnd = (uint64_t)(G.ns + 1) << 16;
					G.ns++;
				}
				d->mem[e] = hnd;
				continue;
			}
			if (G.nt == MAXTEX)
				fail("animate exec: more than %d images", MAXTEX);
			it = T(et)->k == S_SIMAGE ? T(et)->e : et;
			y = T(it);
			storage = y->sampled == 2;
			ti = G.nt++;
			if (d->binding >= 0)
				snprintf(bind, sizeof bind, " (binding %d%s)", d->binding,
				    cnt > 1 ? ", element" : "");
			else
				bind[0] = 0;
			if (cnt > 1)
				snprintf(bind + strlen(bind), sizeof bind - strlen(bind),
				    "[%d]", e);
			/* write-only storage: an output, not an input */
			if (storage && d->nonread) {
				w = img_opt.w > 0 ? img_opt.w : firstw > 0 ? firstw : 64;
				h = img_opt.h > 0 ? img_opt.h : firsth > 0 ? firsth : 64;
				fmt = y->ifmt >= 0 ? y->ifmt : img_opt.outfmt >= 0 ?
				    img_opt.outfmt : 0;
				tex_alloc(&G.t[ti], y->dim, w, h, y->dim == TD_3D ? 8 : 1,
				    y->dim == TD_CUBE ? 6 : y->arrayed ? 4 : 1, fmt, 0);
				G.slot[ti] = -1;
				G.storage[ti] = 1;
				snprintf(G.label[ti], sizeof G.label[ti], "%s%s %s %dx%d",
				    d->name, bind, tex_fmts[fmt].name, w, h);
				sb_printf(said, "  image %s%s: write-only storage, %s %dx%d, "
				    "starts clear\n", d->name, bind, tex_fmts[fmt].name, w,
				    h);
				d->mem[e] = (uint64_t)(ti + 1);
				continue;
			}
			in = &img_opt.in[img_opt.all ? 0 : slot < IMG_MAXIN ? slot :
			    IMG_MAXIN - 1];
			px = input_pixels(slot, in, &gal, seed, &w, &h, what,
			    sizeof what, said);
			fmt = in->fmt;
			if (storage && y->ifmt >= 0) {
				if (fmt >= 0 && fmt != y->ifmt)
					sb_printf(said, "  note: input %d is a storage image "
					    "declared %s: %s ignored\n", slot + 1,
					    tex_fmts[(int)y->ifmt].name, tex_fmts[fmt].name);
				fmt = y->ifmt;
			}
			if (fmt < 0) {
				int sk = T(y->e)->k == S_INT ? (T(y->e)->sign ? 2 : 1) : 0;
				fmt = sk == 1 ? tex_fmt_find("RGBA8UI") : sk == 2 ?
				    tex_fmt_find("RGBA8I") : 0;
			}
			depth = y->dim == TD_3D ? 8 : 1;
			layers = y->dim == TD_CUBE ? 6 * (y->arrayed ? 2 : 1) :
			    y->arrayed ? 4 : 1;
			if (y->dim == TD_CUBE && w != h)
				w = h = w < h ? w : h;
			if (y->dim == TD_1D || y->dim == TD_BUFFER)
				h = 1;
			tex_alloc(&G.t[ti], y->dim, w, h, depth, layers, fmt, !storage &&
			    y->dim != TD_RECT && y->dim != TD_BUFFER);
			tex_from_rgba8(&G.t[ti], px, w, y->dim == TD_1D ||
			    y->dim == TD_BUFFER ? 1 : h);
			free(px);
			tex_mips(&G.t[ti]);
			G.slot[ti] = slot;
			G.storage[ti] = storage;
			if (slot == 0) {
				firstw = w;
				firsth = h;
			}
			snprintf(G.label[ti], sizeof G.label[ti], "input %d: %s %s %dx%d",
			    slot + 1, what, tex_fmts[fmt].name, w, h);
			sb_printf(said, "  input %d = %s%s: %s %s %dx%d%s%s, checksum "
			    "%08x\n", slot + 1, d->name, bind, what, tex_fmts[fmt].name, w,
			    h, G.t[ti].levels > 1 ? ", mipmapped" : "", storage ?
			    " (storage)" : "", tex_sum(&G.t[ti]));
			hnd = (uint64_t)(ti + 1);
			if (T(et)->k == S_SIMAGE && G.ns < MAXTEX) {
				samp_from_opts(&G.s[G.ns], in);
				hnd |= (uint64_t)(G.ns + 1) << 16;
				G.ns++;
			}
			d->mem[e] = hnd;
			slot++;
		}
	}
	if (img_opt.use_gallery && img_opt.ngal == 0)
		sb_printf(said, "  note: no images came with the request: the inputs "
		    "are noise\n");
	{
		int unused = 0;
		for (i = 0; i < img_opt.ngal; i++)
			unused += !(gal_used >> i & 1);
		if (unused)
			sb_printf(said, "  note: %d image%s not used (the shader has "
			    "%d image input%s)\n", unused, unused == 1 ? "" : "s", slot,
			    slot == 1 ? "" : "s");
	}
	for (i = 0; img_opt.any && i < IMG_MAXIN; i++)
		if (i >= slot && !img_opt.all && (img_opt.in[i].kind != IK_DEFAULT ||
		    img_opt.in[i].fmt >= 0)) {
			sb_printf(said, "  note: input %d was given, but the shader has "
			    "%d image input%s\n", i + 1, slot, slot == 1 ? "" : "s");
			break;
		}
	G.on = G.nt > 0 || (frag && img_opt.any);
	if (!G.on)
		return 0;
	for (i = 0; i < G.nt; i++)
		tex_copy(&G.init[i], &G.t[i]);
	/* the image run's size */
	G.W = img_opt.w > 0 ? img_opt.w : firstw > 0 ? firstw : 64;
	G.H = img_opt.h > 0 ? img_opt.h : firsth > 0 ? firsth : 64;
	if (img_opt.w <= 0 && (G.W > 256 || G.H > 256)) {
		double s = 256.0 / (G.W > G.H ? G.W : G.H);
		G.W = (int)(G.W * s) > 0 ? (int)(G.W * s) : 1;
		G.H = (int)(G.H * s) > 0 ? (int)(G.H * s) : 1;
	}
	if (G.W > 1024)
		G.W = 1024;
	if (G.H > 1024)
		G.H = 1024;
	/* the input taken as uv: location 0, else the first vec2/vec3 */
	if (frag) {
		int best = -1;
		for (i = 0; i < V.nid; i++) {
			struct sid *d = &V.id[i];
			int pt;
			if (d->kind != K_VAR || d->sc != SC_INPUT || d->builtin > 0)
				continue;
			pt = T(d->type)->e;
			if (T(pt)->k != S_VEC || T(pt)->n > 3 ||
			    T(T(pt)->e)->k != S_FLOAT)
				continue;
			if (d->location == 0) {
				best = i;
				break;
			}
			if (best < 0)
				best = i;
		}
		G.uv = best;
		if (best >= 0)
			sb_printf(said, "  note: %s (location %d) is taken as the "
			    "pixel's uv: its centre over the image size, 0..1\n",
			    V.id[best].name, V.id[best].location);
	}
	return 1;
}

/* every per-lane variable's starting values, for lanes 0..n-1 */
static void
lanes_init(int n)
{
	int i, l;

	for (i = 0; i < V.nid; i++) {
		struct sid *d = &V.id[i];
		if (d->kind != K_VAR || shared_sc(d->sc) || d->mem == NULL)
			continue;
		for (l = 0; l < n; l++)
			init_var(i, l, d->mem + (size_t)l * d->nslots);
	}
}

/* the outputs of the image run */
#define MAXOUT 8
static struct {
	int var[MAXOUT], n;
	struct tex t[MAXOUT];
	unsigned char *cover[MAXOUT];	/* written (not discarded) */
	long discarded;
	int rows;			/* rows done */
} O;

/*
 * The whole image: the shader on every pixel (or invocation), 8x4 lanes
 * at a time so derivatives see 2x2 quads; edge lanes outside the image
 * run as helpers (fragment) or are masked off (compute)
 */
static void
image_run(int lb, int frag, int *stopped)
{
	int saved_lanes = V.lanes, saved_cols = cols, i, tx, ty, l;
	long total = 0;

	memset(&O, 0, sizeof O);
	for (i = 0; i < G.nt; i++) {
		tex_free(&G.t[i]);
		tex_copy(&G.t[i], &G.init[i]);
		G.written[i] = 0;
	}
	if (frag)
		for (i = 0; i < V.nid && O.n < MAXOUT; i++) {
			struct sid *d = &V.id[i];
			int pt, st, fmt;
			if (d->kind != K_VAR || d->sc != SC_OUTPUT || d->builtin > 0)
				continue;
			pt = T(d->type)->e;
			st = scalar_of(pt);
			if (T(pt)->k != S_VEC && T(pt)->k != S_FLOAT && T(pt)->k != S_INT)
				continue;
			fmt = T(st)->k == S_INT ? tex_fmt_find(T(st)->sign ? "RGBA32I" :
			    "RGBA32UI") : d->location <= 0 && img_opt.outfmt >= 0 ?
			    img_opt.outfmt : 0;
			O.var[O.n] = i;
			tex_alloc(&O.t[O.n], TD_2D, G.W, G.H, 1, 1, fmt, 0);
			O.cover[O.n] = xcalloc((size_t)G.W * G.H, 1);
			O.n++;
		}
	V.lanes = 32;
	cols = 8;
	V.quiet = 1;
	G.run = 1;
	*stopped = 0;
	for (ty = 0; ty < G.H; ty += 4) {
		for (tx = 0; tx < G.W; tx += 8) {
			uint32_t m = 0xffffffffu;
			G.x0 = tx;
			G.y0 = ty;
			if (!frag)
				for (l = 0, m = 0; l < 32; l++)
					if (tx + l % 8 < G.W && ty + l / 8 < G.H)
						m |= 1u << l;
			lanes_init(32);
			V.retired = V.killed = 0;
			V.steps = 0;
			memset(V.curblk, 0, sizeof V.curblk);
			memset(V.prevblk, 0, sizeof V.prevblk);
			region(lb, m, -1, -1, NULL);
			total += V.steps;
			for (l = 0; l < 32 && frag; l++) {
				int x = tx + l % 8, y = ty + l / 8, k;
				if (x >= G.W || y >= G.H)
					continue;
				if (V.killed >> l & 1) {
					O.discarded++;
					continue;
				}
				for (k = 0; k < O.n; k++) {
					struct sid *d = &V.id[O.var[k]];
					double v[4] = {0, 0, 0, 1};
					int c;
					for (c = 0; c < d->nslots && c < 4; c++) {
						int st = d->slotty[c];
						uint64_t b = d->mem[(size_t)l * d->nslots + c];
						v[c] = T(st)->k == S_FLOAT ? getf(st, b) :
						    (double)geti(st, b);
					}
					tex_store(&O.t[k], 0, x, y, 0, 0, v);
					O.cover[k][(size_t)y * G.W + x] = 1;
				}
			}
		}
		O.rows = ty + 4 < G.H ? ty + 4 : G.H;
		/* a bound on the work, so a heavy shader still answers */
		if (total > 20000000L && ty + 4 < G.H) {
			*stopped = 1;
			break;
		}
	}
	V.quiet = 0;
	G.run = 0;
	V.lanes = saved_lanes;
	cols = saved_cols;
}

/* the run's images, for the report and the animation */
static void
image_report(int frag, int stopped)
{
	int i, k;
	unsigned char *px;

	if (stopped)
		printf("  note: the image run stopped after %d of %d rows (over "
		    "20000000 instructions); the rest is left empty\n", O.rows,
		    G.H);
	for (i = 0; i < G.nt; i++) {
		if (G.slot[i] < 0)
			continue;
		px = xmalloc((size_t)G.init[i].w * G.init[i].h * 4);
		tex_to_rgba8(&G.init[i], px);
		xr_panel(G.label[i], G.init[i].w, G.init[i].h, px, NULL, 0);
		free(px);
	}
	for (k = 0; k < O.n; k++) {
		struct sid *d = &V.id[O.var[k]];
		char lab[96];
		px = xmalloc((size_t)G.W * G.H * 4);
		tex_to_rgba8(&O.t[k], px);
		snprintf(lab, sizeof lab, "output %s%s %s %dx%d", d->name,
		    d->location >= 0 ? (d->location == 0 ? " (location 0)" :
		    " (location >0)") : "", tex_fmts[O.t[k].fmt].name, G.W, G.H);
		if (d->location > 0)
			snprintf(lab, sizeof lab, "output %s (location %d) %s %dx%d",
			    d->name, d->location, tex_fmts[O.t[k].fmt].name, G.W, G.H);
		printf("  output %s: %s %dx%d, checksum %08x", d->name,
		    tex_fmts[O.t[k].fmt].name, G.W, G.H, tex_sum(&O.t[k]));
		if (O.discarded)
			printf(", %ld pixel%s discarded", O.discarded, O.discarded == 1 ?
			    "" : "s");
		printf("\n");
		xr_panel(lab, G.W, G.H, px, O.cover[k], 1);
		free(px);
	}
	for (i = 0; i < G.nt; i++) {
		char lab[120];
		if (!G.written[i])
			continue;
		printf("  image %s after the run: checksum %08x\n", G.label[i],
		    tex_sum(&G.t[i]));
		px = xmalloc((size_t)G.t[i].w * G.t[i].h * 4);
		tex_to_rgba8(&G.t[i], px);
		snprintf(lab, sizeof lab, "%s%s, after", G.slot[i] >= 0 ? "" :
		    "storage ", G.label[i]);
		xr_panel(lab, G.t[i].w, G.t[i].h, px, NULL, 1);
		free(px);
	}
	if (G.oob)
		printf("  note: %ld texel access%s outside an image: read as 0 "
		    "(alpha 1 when the format has none) and writes dropped, as with "
		    "robustImageAccess2\n", G.oob, G.oob == 1 ? "" : "es");
	if (V.divderiv)
		printf("  note: %ld derivative%s had a quad neighbour that was not "
		    "running (divergent control flow, or a discarded pixel): its "
		    "difference counted as 0\n", V.divderiv, V.divderiv == 1 ?
		    "" : "s");
	if (G.nsample) {
		struct samp d;
		int given = img_opt.filt >= 0 || img_opt.addr >= 0 ||
		    img_opt.mipf >= 0, k;
		for (k = 0; k < IMG_MAXIN; k++)
			given |= img_opt.in[k].filt >= 0 || img_opt.in[k].addr >= 0;
		samp_from_opts(&d, NULL);
		printf("  sampling: %s%s; implicit LOD from the quad's fine "
		    "derivatives, mips by a 2x2 box filter, filter weights in "
		    "double precision\n", samp_text(G.ns > 0 ? &G.s[0] : &d),
		    given ? " (as given)" : " (the sampler's state is not in "
		    "SPIR-V: assumed; say `sampler nearest clamp`, or `input 1 "
		    "nearest mirror`)");
	}
	(void)frag;
}

static void
image_free(void)
{
	int i;

	for (i = 0; i < G.nt; i++) {
		tex_free(&G.t[i]);
		tex_free(&G.init[i]);
	}
	for (i = 0; i < O.n; i++) {
		tex_free(&O.t[i]);
		free(O.cover[i]);
	}
	memset(&O, 0, sizeof O);
	G.nt = 0;
}

static void
dump_buffers(const char *when)
{
	int i, s;
	char buf[64];

	for (i = 0; i < V.nid; i++) {
		struct sid *d = &V.id[i];
		int cnt;
		if (d->kind != K_VAR || d->sc != SC_SHARED ||
		    binding_elem(i, &cnt) >= 0)
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
	struct sbuf isaid = {0};

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
	/* images: textures for the bindings, the image run's size and uv */
	{
		int frag = 0, nset = 0;
		for (k = 0; k < V.nin; k++) {
			if (!strcmp(V.in[k].op, "OpEntryPoint") && V.in[k].nop >= 1)
				frag = !strcmp(V.in[k].opnd[0], "Fragment");
			if (!strcmp(V.in[k].op, "OpExecutionMode") && V.in[k].nop >= 5 &&
			    !strcmp(V.in[k].opnd[1], "LocalSize")) {
				G.lsize[0] = atoi(V.in[k].opnd[2]);
				G.lsize[1] = atoi(V.in[k].opnd[3]);
				G.lsize[2] = atoi(V.in[k].opnd[4]);
			}
		}
		{
			int ls[3];
			memcpy(ls, G.lsize, sizeof ls);
			img_setup(seed, frag, &isaid);
			memcpy(G.lsize, ls, sizeof ls);
		}
		/* "%uv = 0.5" on an input variable of a whole module */
		for (k = 0; k < V.nid; k++) {
			struct sid *d = &V.id[k];
			int q;
			if (d->kind != K_VAR || d->sc != SC_INPUT)
				continue;
			for (q = 0; q < nspv_set; q++)
				if (!strcmp(spv_set[q].name, d->name) ||
				    !strcmp(spv_set[q].name, d->tok + 1)) {
					d->setix = q;
					nset++;
				}
		}
		if (G.on) {
			/* the traced lanes: the tile at the image's centre */
			int rows = lanes / cols;
			G.x0 = G.W / 2 / cols * cols;
			G.y0 = G.H / 2 / rows * rows;
		}
		if (G.on || nset)
			lanes_init(lanes);
	}
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
			snprintf(b, sizeof b, "frag (%d.5,%d.5)", G.x0 + l % cols,
			    G.y0 + l / cols);
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
	if (isaid.n)
		fputs(isaid.p, stdout);
	free(isaid.p);
	if (G.on)
		printf("  the trace: lanes at pixels (%d,%d)..(%d,%d) of the %dx%d "
		    "image\n", G.x0, G.y0, G.x0 + cols - 1, G.y0 + lanes / cols - 1,
		    G.W, G.H);
	for (k = 0; k < V.nin; k++)
		if (!strcmp(V.in[k].op, "OpExecutionMode") && V.in[k].nop >= 5 &&
		    !strcmp(V.in[k].opnd[1], "LocalSize")) {
			long wg = atol(V.in[k].opnd[2]) * atol(V.in[k].opnd[3]) *
			    atol(V.in[k].opnd[4]);
			if (wg > lanes && G.on)
				printf("  note: the workgroup has %ld invocations (LocalSize "
				    "%s %s %s); the trace shows %d of them, the image run "
				    "every one, 32 at a time (shared memory and barriers "
				    "across more than 32 are not modelled there)\n", wg,
				    V.in[k].opnd[2], V.in[k].opnd[3], V.in[k].opnd[4],
				    lanes);
			else if (wg > lanes)
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
	if (G.on) {
		int stopped, qx = G.x0, qy = G.y0;
		image_run(lb, model == 1, &stopped);
		image_report(model == 1, stopped);
		xr_quad(qx, qy, cols, lanes / cols, G.W, G.H);
	}
	code_block_used = 1;
	xr_done(fps, seconds, name, nofile);
	image_free();
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
	char pre_op[40];	/* RW_INS: a new instruction for operand 1 */
	int pre_a[2];
	double extra;		/* instructions left dead by it: their cost */
	int algebra;		/* exact by ring identities mod 2^w (affine) */
	int kind;
	int id;			/* RW_ID: the operand */
	uint64_t v;		/* RW_CONST */
	char op[40], ext[24];	/* RW_INS: the instruction */
	int a[3], na;		/* its operands (ids) */
	double cost;
	const char *why;
	int via_u, ut;		/* RW_INS on the bits as unsigned type ut:
				 * bitcasts in and out (OpUDiv and OpUMod
				 * want an unsigned result type) */
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
	lit *dl;		/* per id: "its value is defined" */
	char cex[240];		/* the last counterexample, as text */
	struct {
		int id;
		struct word w;
	} leaf[96];		/* the free inputs of the current query */
	int nleaf;
	struct {
		char name[40];
		int64_t lo, hi;
	} as[8];		/* "n in [0, 32)": assumptions */
	int nas;
	int orig_nin;
	int *before;		/* per new instruction: emitted before this */
	char notes[6][700];	/* near misses worth telling */
	int nnotes;
	int force_id;		/* range search: this input has force_val */
	uint64_t force_val;
	int orig_nid;
	int *uses;		/* per id: operand references, before rewriting */
} SP;

/*
 * A struct member's name for an access chain "%base %int_k" into a
 * struct (OpMemberName), or NULL: after inlining, a parameter is a load
 * from the harness's buffer member "Fn_n"
 */
static const char *
member_name(int ptr)
{
	int k = SP.def[ptr], base, bt, st, q;
	uint64_t mi;

	if (k < 0 && ptr < V.nid) {
		/* module-level access chains are not in SP.def */
		for (q = 0; q < V.nin; q++)
			if (V.in[q].res == ptr) {
				k = q;
				break;
			}
	}
	if (k < 0 || (strcmp(V.in[k].op, "OpAccessChain") != 0 &&
	    strcmp(V.in[k].op, "OpInBoundsAccessChain") != 0) ||
	    V.in[k].nop != 3)
		return NULL;
	base = idx(V.in[k].opnd[1]);
	bt = V.id[base].type;
	if (bt < 0 || V.id[bt].kind != K_TYPE || T(bt)->k != S_PTR)
		return NULL;
	st = T(bt)->e;
	if (V.id[st].kind != K_TYPE || T(st)->k != S_STRUCT ||
	    V.id[idx(V.in[k].opnd[2])].kind != K_CONST)
		return NULL;
	mi = V.id[idx(V.in[k].opnd[2])].c[0];
	for (q = 0; q < V.nin; q++)
		if (!strcmp(V.in[q].op, "OpMemberName") && V.in[q].nop >= 3 &&
		    idx(V.in[q].opnd[0]) == st &&
		    (uint64_t)atoll(V.in[q].opnd[1]) == mi) {
			static char buf[64];
			const char *m = V.in[q].opnd[2];
			snprintf(buf, sizeof buf, "%.*s", (int)strlen(m) - 2, m + 1);
			return buf;
		}
	return NULL;
}

/* NAME, or "Fn_NAME" (a harness member) */
static int
name_is(const char *have, const char *want)
{
	size_t h = strlen(have), w = strlen(want);

	if (!strcmp(have, want))
		return 1;
	return h > w + 1 && have[h - w - 1] == '_' && !strcmp(have + h - w, want);
}

/* the assumption on an input: a value named NAME, or a load from it */
static int
assumed(int id, int64_t *lo, int64_t *hi)
{
	const char *nm = V.id[id].name, *mn = NULL;
	int k = SP.def[id], j;

	if (SP.nas == 0)
		return 0;
	if (k >= 0 && !strcmp(V.in[k].op, "OpLoad") && V.in[k].nop >= 2) {
		int ptr = idx(V.in[k].opnd[1]), q;
		/* a pointer nothing stores to: every load sees the same value */
		for (q = 0; q < V.nin; q++)
			if (!strcmp(V.in[q].op, "OpStore") && V.in[q].nop >= 1 &&
			    idx(V.in[q].opnd[0]) == ptr)
				return 0;
		nm = V.id[ptr].name;
		mn = member_name(ptr);
	}
	if (*nm == '%')
		nm++;
	for (j = 0; j < SP.nas; j++)
		if (!strcmp(SP.as[j].name, nm) || (mn != NULL &&
		    name_is(mn, SP.as[j].name))) {
			*lo = SP.as[j].lo;
			*hi = SP.as[j].hi;
			return 1;
		}
	return 0;
}

static void
leaf_made(int id, const struct word *w)
{
	if (SP.nleaf < 96) {
		SP.leaf[SP.nleaf].id = id;
		SP.leaf[SP.nleaf++].w = *w;
	}
}

/* the assumptions on the current query's inputs, as one literal */
static lit
assumptions(void)
{
	lit a = L_TRUE;
	int i;

	for (i = 0; i < SP.nleaf; i++) {
		int64_t lo, hi;
		const struct word *x = &SP.leaf[i].w;
		struct word l, h;
		if (!assumed(SP.leaf[i].id, &lo, &hi))
			continue;
		w_const(&l, x->w, (uint64_t)lo & wmask(x->w));
		w_const(&h, x->w, (uint64_t)hi & wmask(x->w));
		if (lo < 0)	/* signed bounds */
			a = aig_and(SP.g, a, aig_and(SP.g, L_NOT(w_slt(SP.g, x, &l)),
			    L_NOT(w_slt(SP.g, &h, x))));
		else
			a = aig_and(SP.g, a, aig_and(SP.g, L_NOT(w_ult(SP.g, x, &l)),
			    L_NOT(w_ult(SP.g, &h, x))));
	}
	return a;
}

static const char *member_name(int ptr);

/* the counterexample's named inputs: "%n = 0" */
static void
cex_text(char *b, size_t n, int undefined)
{
	int i, at = 0, shown = 0;

	b[0] = 0;
	for (i = 0; i < SP.nleaf && shown < 3; i++) {
		int id = SP.leaf[i].id, k = SP.def[id], w = SP.leaf[i].w.w;
		const char *nm = V.id[id].name;
		uint64_t v = aig_word_value(SP.g, &SP.leaf[i].w);
		int64_t lo = 0, hi = 0, sv;
		int as = assumed(id, &lo, &hi);
		if (k >= 0 && !strcmp(V.in[k].op, "OpLoad")) {
			const char *mn = member_name(idx(V.in[k].opnd[1]));
			nm = V.id[idx(V.in[k].opnd[1])].name;
			if (mn != NULL) {	/* "Skip_n": the parameter n */
				const char *u = strrchr(mn, '_');
				static char pn[64];
				snprintf(pn, sizeof pn, "%s", u && u[1] ? u + 1 : mn);
				nm = pn;
			}
		}
		/* named inputs and assumed ones tell the most */
		if (!as && nm[0] == '%' && isdigit((unsigned char)nm[1]))
			continue;
		sv = (int64_t)v;
		if (as && lo < 0 && w < 64 && (v >> (w - 1) & 1))
			sv = (int64_t)(v | ~wmask(w));
		at += snprintf(b + at, n - (size_t)at, "%s%s%s = %lld",
		    shown ? ", " : "", nm[0] == '%' ? "" : "%", nm, (long long)sv);
		shown++;
		if ((size_t)at >= n)
			return;
	}
	if (undefined && (size_t)at < n)
		snprintf(b + at, n - (size_t)at, "%s(where the replacement is "
		    "undefined: a shift by the width or more, a bitfield past it, "
		    "or a division by zero)", at ? " " : "");
}


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
	} else if (id == SP.force_id) {
		r = SP.force_val & wmask(w ? w : 32);
	} else if (k < 0 || !pure_ins(k)) {
		int64_t lo, hi;
		r = leafval(id, w ? w : 32);
		if (assumed(id, &lo, &hi)) {	/* inside the range, ends too */
			uint64_t span = (uint64_t)(hi - lo) + 1;
			int s = SP.sample % 8;
			r = (uint64_t)(s == 0 ? lo : s == 1 ? hi : lo +
			    (int64_t)(span ? r % span : r)) & wmask(w ? w : 32);
		}
	}
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
	if (c->pre_op[0]) {
		int pt = V.id[res_of(c->pre_a[0])].type;
		x[1] = alu(c->pre_op, pt, pt, evalid(c->pre_a[0]),
		    evalid(c->pre_a[1]), 0, &ok) & wmask(int_width(pt));
	}
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
/* x >= v for a constant v (x unsigned) */
static lit
w_ge_const(struct aig *g, const struct word *x, uint64_t v)
{
	struct word c;

	if (x->w < 64 && v > wmask(x->w))
		return L_FALSE;
	w_const(&c, x->w, v);
	return L_NOT(w_ult(g, x, &c));
}

/* offset + count > width, both unsigned (in 64 bits: no overflow) */
static lit
range_past(struct aig *g, const struct word *off, const struct word *cnt,
    int w)
{
	struct word o, c, sum, lim;

	w_resize(&o, off, 64, 0);
	w_resize(&c, cnt, 64, 0);
	w_add(g, &sum, &o, &c);
	w_const(&lim, 64, (uint64_t)w);
	return w_ult(g, &lim, &sum);
}

/*
 * The undefined cases SPIR-V leaves open, for the operation on these
 * operands: a shift by the width or more, a bitfield past the width, a
 * division by zero, the most negative number divided by -1.
 */
static lit
op_undefined(const char *op, struct word **a, int aw)
{
	struct aig *g = SP.g;
	struct word z, m1, mn;

	if (!strncmp(op, "OpShift", 7))
		return w_ge_const(g, a[1], (uint64_t)aw);
	if (!strcmp(op, "OpBitFieldUExtract") || !strcmp(op, "OpBitFieldSExtract"))
		return range_past(g, a[1], a[2], aw);
	if (!strcmp(op, "OpBitFieldInsert"))
		return range_past(g, a[2], a[3], aw);
	if (!strcmp(op, "OpUDiv") || !strcmp(op, "OpUMod")) {
		w_const(&z, aw, 0);
		return w_eq(g, a[1], &z);
	}
	if (!strcmp(op, "OpSDiv") || !strcmp(op, "OpSRem") ||
	    !strcmp(op, "OpSMod")) {
		w_const(&z, aw, 0);
		w_const(&m1, aw, wmask(aw));
		w_const(&mn, aw, 1ULL << (aw - 1));
		return aig_or(g, w_eq(g, a[1], &z),
		    aig_and(g, w_eq(g, a[0], &mn), w_eq(g, a[1], &m1)));
	}
	return L_FALSE;
}

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
	SP.dl[id] = L_TRUE;
	if (V.id[id].kind == K_CONST && !V.id[id].spec)
		w_const(r, w, V.id[id].c[0] & wmask(w));
	else if (k >= 0 && depth > 0 && pure_ins(k)) {
		const struct sins *I = &V.in[k];
		struct word *a[4], tmp;
		int base = first_arg(I), i, n = 0, at, ids[4];
		lit d = L_TRUE;
		for (i = base; i < I->nop && n < 4; i++, n++) {
			ids[n] = res_of(idx(I->opnd[i]));
			a[n] = blast(ids[n], depth - 1);
		}
		at = V.id[res_of(idx(I->opnd[base]))].type;
		for (i = 0; i < n; i++)
			d = aig_and(SP.g, d, SP.dl[ids[i]]);
		if (strcmp(I->op, "OpExtInst") != 0)
			d = aig_and(SP.g, d, L_NOT(op_undefined(I->op, a,
			    int_width(at))));
		blast_op(&tmp, I->op, !strcmp(I->op, "OpExtInst") ? I->opnd[2] :
		    NULL, idx(I->opnd[0]), a, n, at);
		r = &SP.memo[id];	/* (blast may have moved nothing: arrays) */
		*r = tmp;
		SP.dl[id] = d;
	} else if (k >= 0 && depth > 0 && fwd_phi(k)) {
		/* one of the incoming values; which one is free */
		const struct sins *I = &V.in[k];
		struct word acc, *v;
		lit d, sel;
		int i, v0 = res_of(idx(I->opnd[1]));
		acc = *blast(v0, depth - 1);
		d = SP.dl[v0];
		for (i = 3; i + 1 < I->nop; i += 2) {
			int vi = res_of(idx(I->opnd[i]));
			v = blast(vi, depth - 1);
			sel = aig_input(SP.g);
			w_mux(SP.g, &acc, sel, v, &acc);
			d = aig_mux(SP.g, sel, SP.dl[vi], d);
		}
		SP.memo[id] = acc;
		r = &SP.memo[id];
		SP.dl[id] = d;
	} else {
		w_input(SP.g, r, w);
		leaf_made(id, r);
	}
	return r;
}

static void
blast_reset(void)
{
	SP.nleaf = 0;
	while (SP.ntouched > 0)
		SP.have[SP.touched[--SP.ntouched]] = 0;
	aig_free(SP.g);
	SP.g = aig_new();
}

static int const_of(int id, uint64_t *v);

/*
 * The cost of an instruction under the model; a division's divisor b
 * (an id, or -1) picks the class by a constant, and a power of two
 * makes it a shift (unsigned) or the shifts and add of a signed one.
 */
static double
ins_cost_b(const char *op, const char *ext, int rt, int b)
{
	double mul = 1;
	int cls = cost_class(op, ext && ext[0] ? ext : NULL, rt, &mul);
	const struct model *m = model_active();
	uint64_t v;

	if (cls == C_IDIV || cls == C_SDIV || cls == C_IREM || cls == C_SREM) {
		if (b >= 0 && const_of(b, &v)) {
			int w = int_width(rt), sg = cls == C_SDIV || cls == C_SREM;
			if (v > 1 && !(v & (v - 1)) && (!sg || (w > 0 && w <= 64 &&
			    !(v >> (w - 1)))))
				return !sg ? m->c[cls == C_IDIV ? C_SHIFT : C_LOGIC] :
				    cls == C_SDIV ? 3 * m->c[C_SHIFT] + m->c[C_IADD] :
				    2 * m->c[C_SHIFT] + 2 * m->c[C_IADD] +
				    m->c[C_LOGIC];
			cls = C_DIVK(cls);
		}
	}
	return cls >= 0 ? m->c[cls] * mul : 0;
}

static double
ins_cost(const char *op, const char *ext, int rt)
{
	return ins_cost_b(op, ext, rt, -1);
}

/* instruction k's cost (its divisor read from its operands) */
static double
ins_cost_k(int k)
{
	const struct sins *I = &V.in[k];
	int ext = !strcmp(I->op, "OpExtInst");

	return ins_cost_b(I->op, ext ? I->opnd[2] : NULL, idx(I->opnd[0]),
	    !ext && I->nop >= 3 && I->opnd[2][0] == '%' ? idx(I->opnd[2]) : -1);
}

/*
 * 1: proved equal for every input, 0: a difference (or none found to be
 * impossible within the depths), -1: the solver ran out of budget
 */
/*
 * 1: proved, for every input the assumptions allow: where the original
 * is defined, the candidate is defined and equal (a refinement: what
 * SPIR-V leaves undefined may become anything); 0: a counterexample (or
 * none found impossible within the depths: SP.cex says why); -1: the
 * solver ran out of budget; -2: out of time
 */
static int
prove(int x, const struct cand *c, int rt)
{
	static const int depths[] = {2, 4, 8, 16};
	int di, out = 0;

	SP.cex[0] = 0;
	if (c->algebra) {
		/*
		 * a*x + b came from adds, subtracts, negation, constant shifts
		 * (below the width: defined) and constant multiplies, all exact
		 * in Z/2^w: equal by the ring laws.  A multiplier against a
		 * shift-and-add network is hard for SAT, so the algebra is the
		 * proof, checked again on 1024 random inputs.
		 */
		int smp;
		for (smp = 0; smp < 1024; smp++) {
			SP.sample = 100 + smp;
			SP.stamp++;
			if (evalid(x) != evalcand(c, rt))
				return 0;
		}
		return 1;
	}
	for (di = 0; di < 4; di++) {
		struct word *wx, wc, *a[4], pre;
		lit dx, dc = L_TRUE, goal;
		int i, r, j;
		if ((double)clock() / CLOCKS_PER_SEC > SP.deadline)
			return -2;
		blast_reset();
		wx = blast(x, depths[di]);
		dx = SP.dl[res_of(x)];
		if (c->kind == RW_ID) {
			wc = *blast(c->id, depths[di] - 1);
			dc = SP.dl[res_of(c->id)];
		} else if (c->kind == RW_CONST)
			w_const(&wc, wx->w, c->v);
		else {
			for (i = 0; i < c->na; i++) {
				a[i] = blast(c->a[i], depths[di] - 1);
				dc = aig_and(SP.g, dc, SP.dl[res_of(c->a[i])]);
			}
			if (c->pre_op[0]) {	/* a new instruction feeds operand 1 */
				struct word *pa[2];
				int pt = V.id[res_of(c->pre_a[0])].type;
				for (j = 0; j < 2; j++) {
					pa[j] = blast(c->pre_a[j], depths[di] - 1);
					dc = aig_and(SP.g, dc, SP.dl[res_of(c->pre_a[j])]);
				}
				blast_op(&pre, c->pre_op, NULL, pt, pa, 2, pt);
				dc = aig_and(SP.g, dc, L_NOT(op_undefined(c->pre_op, pa,
				    int_width(pt))));
				a[1] = &pre;
			}
			if (!c->ext[0])
				dc = aig_and(SP.g, dc, L_NOT(op_undefined(c->op, a,
				    int_width(V.id[res_of(c->a[0])].type))));
			blast_op(&wc, c->op, c->ext, rt, a, c->na,
			    V.id[res_of(c->a[0])].type);
		}
		if (wc.w != wx->w)
			return 0;
		/* defined original, and the candidate undefined or different */
		goal = aig_and(SP.g, dx, aig_or(SP.g, L_NOT(dc),
		    w_ne(SP.g, wx, &wc)));
		goal = aig_and(SP.g, goal, assumptions());
		r = aig_sat(SP.g, goal, 20000);
		if (r == 0)
			return 1;
		if (r < 0)
			out = -1;
		if (r == 1)
			cex_text(SP.cex, sizeof SP.cex, aig_value(SP.g, dc) == 0);
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
	c->cost = ins_cost_b(op, ext, rt, a1);
	c->why = why;
	return 1;
}

/*
 * Affine forms over Z/2^w: an integer value as a*x + b for one base x,
 * through adds, subtracts, negation, shifts left and multiplies by
 * constants.  "p -= p << 17; p += p << 7" is one multiply of p.
 */
struct aff {
	int base;		/* -1: a constant (b) */
	uint64_t a, b;
};

static struct aff
aff_of(int id, int w, int depth)
{
	struct aff r, x, y;
	const struct sins *I;
	int k;
	uint64_t c, m = wmask(w);

	id = res_of(id);
	r.base = id;
	r.a = 1;
	r.b = 0;
	if (V.id[id].kind == K_CONST && !V.id[id].spec) {
		r.base = -1;
		r.a = 0;
		r.b = V.id[id].c[0] & m;
		return r;
	}
	k = SP.def[id];
	if (depth <= 0 || k < 0 || !pure_ins(k) ||
	    int_width(V.id[id].type) != w)
		return r;
	I = &V.in[k];
	if (!strcmp(I->op, "OpIAdd") || !strcmp(I->op, "OpISub")) {
		x = aff_of(idx(I->opnd[1]), w, depth - 1);
		y = aff_of(idx(I->opnd[2]), w, depth - 1);
		if (x.base >= 0 && y.base >= 0 && x.base != y.base)
			return r;
		if (I->op[3] == 'S') {
			y.a = (0 - y.a) & m;
			y.b = (0 - y.b) & m;
		}
		r.base = x.base >= 0 ? x.base : y.base;
		r.a = (x.a + y.a) & m;
		r.b = (x.b + y.b) & m;
		return r;
	}
	if (!strcmp(I->op, "OpSNegate")) {
		x = aff_of(idx(I->opnd[1]), w, depth - 1);
		x.a = (0 - x.a) & m;
		x.b = (0 - x.b) & m;
		return x;
	}
	if (!strcmp(I->op, "OpIMul")) {
		x = aff_of(idx(I->opnd[1]), w, depth - 1);
		y = aff_of(idx(I->opnd[2]), w, depth - 1);
		if (y.base < 0) {
			struct aff t = x;
			x = y;
			y = t;
		}
		if (x.base >= 0)	/* x * y, neither constant */
			return r;
		y.a = (y.a * x.b) & m;
		y.b = (y.b * x.b) & m;
		return y;
	}
	if (!strcmp(I->op, "OpShiftLeftLogical") &&
	    const_of(idx(I->opnd[2]), &c) && c < (uint64_t)w) {
		x = aff_of(idx(I->opnd[1]), w, depth - 1);
		x.a = (x.a << c) & m;
		x.b = (x.b << c) & m;
		return x;
	}
	if (!strcmp(I->op, "OpCopyObject") || (!strcmp(I->op, "OpBitcast") &&
	    int_width(V.id[res_of(idx(I->opnd[1]))].type) == w))
		return aff_of(idx(I->opnd[1]), w, depth - 1);
	return r;
}

/*
 * The cost of the instructions an affine rewrite of id leaves dead: the
 * cone down to base whose every use is inside the cone
 */
static double
aff_dead_cost(int id, int base, int rt)
{
	int cone[64], n = 0, i, j, q, top = 0, stack[64];
	double c = 0;

	stack[top++] = res_of(id);
	while (top > 0 && n < 64) {
		int v = stack[--top], k = SP.def[v], seen = 0;
		if (v == base || k < 0 || !pure_ins(k) ||
		    (V.id[v].kind == K_CONST && !V.id[v].spec))
			continue;
		for (j = 0; j < n; j++)
			if (cone[j] == v)
				seen = 1;
		if (seen)
			continue;
		cone[n++] = v;
		for (i = first_arg(&V.in[k]); i < V.in[k].nop && top < 64; i++)
			stack[top++] = res_of(idx(V.in[k].opnd[i]));
	}
	/* dead: the top (replaced), then any node whose every use is by a
	 * dead node, until nothing changes */
	{
		char dead[64] = {0};
		int changed = 1;
		dead[0] = 1;
		while (changed) {
			changed = 0;
			for (j = 1; j < n; j++) {
				int from_dead = 0;
				if (dead[j])
					continue;
				for (q = 0; q < n; q++) {
					const struct sins *I;
					if (!dead[q])
						continue;
					I = &V.in[SP.def[cone[q]]];
					for (i = first_arg(I); i < I->nop; i++)
						if (res_of(idx(I->opnd[i])) == cone[j])
							from_dead++;
				}
				if (from_dead >= SP.uses[cone[j]]) {
					dead[j] = 1;
					changed = 1;
					c += ins_cost_k(SP.def[cone[j]]);
				}
			}
		}
	}
	return c;
}

static int
is_param(int id)
{
	return SP.def[id] >= 0 && !strcmp(V.in[SP.def[id]].op,
	    "OpFunctionParameter");
}

/* the variable (or pointer parameter) a pointer points into */
static int
ptr_root(int id)
{
	int k;

	for (;;) {
		k = SP.def[id];
		if (k < 0 || (strcmp(V.in[k].op, "OpAccessChain") != 0 &&
		    strcmp(V.in[k].op, "OpInBoundsAccessChain") != 0))
			return id;
		id = idx(V.in[k].opnd[1]);
	}
}

/*
 * Store-to-load forwarding in a block: a load from a per-invocation
 * pointer (Function, Private) after a store to the same pointer, with no
 * call and no store that could alias in between, is the stored value.
 * Returns how many loads it forwarded.
 */
static int
forward_loads(void)
{
	int k, n = 0;

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int ptr, root, j, pt;
		if (strcmp(I->op, "OpLoad") != 0 || I->res < 0 || I->nop < 2)
			continue;
		ptr = idx(I->opnd[1]);
		pt = V.id[ptr].type;
		if (pt < 0 || V.id[pt].kind != K_TYPE || T(pt)->k != S_PTR ||
		    (T(pt)->sc != SC_FUNCTION && T(pt)->sc != SC_PRIVATE))
			continue;
		root = ptr_root(ptr);
		for (j = k - 1; j >= 0; j--) {
			const struct sins *J = &V.in[j];
			if (!strcmp(J->op, "OpLabel") || !strcmp(J->op, "OpFunction") ||
			    !strcmp(J->op, "OpFunctionCall") ||
			    !strncmp(J->op, "OpAtomic", 8) ||
			    !strcmp(J->op, "OpCopyMemory"))
				break;
			if (!strcmp(J->op, "OpStore") && J->nop >= 2) {
				int sp = idx(J->opnd[0]), sr = ptr_root(sp);
				if (sp == ptr) {
					int v = idx(J->opnd[1]);
					if (V.id[v].type == V.id[I->res].type ||
					    (V.id[v].kind == K_CONST &&
					    V.id[v].type == idx(I->opnd[0]))) {
						SP.alias[I->res] = res_of(v);
						SP.dead[k] = 1;	/* a load has no effect */
						n++;
					}
					break;
				}
				/* another pointer into the same thing, or one that
				 * could point anywhere: stop */
				if (sr == root || is_param(sr) || is_param(root))
					break;
			}
		}
	}
	return n;
}

/*
 * A store to a per-invocation pointer that the same block stores to
 * again before anything could read it (a load from the same variable
 * that is still there, a call, the block's end) is dead.
 */
static int
dead_stores(void)
{
	int k, n = 0;

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int ptr, root, j, pt;
		if (SP.dead[k] || strcmp(I->op, "OpStore") != 0 || I->nop < 2)
			continue;
		ptr = idx(I->opnd[0]);
		pt = V.id[ptr].type;
		if (pt < 0 || V.id[pt].kind != K_TYPE || T(pt)->k != S_PTR ||
		    (T(pt)->sc != SC_FUNCTION && T(pt)->sc != SC_PRIVATE))
			continue;
		root = ptr_root(ptr);
		for (j = k + 1; j < V.nin; j++) {
			const struct sins *J = &V.in[j];
			if (SP.dead[j])
				continue;
			if (!strcmp(J->op, "OpStore") && J->nop >= 1 &&
			    idx(J->opnd[0]) == ptr) {
				SP.dead[k] = 1;
				n++;
				break;
			}
			if (!strcmp(J->op, "OpLoad") && J->nop >= 2) {
				int lr = ptr_root(idx(J->opnd[1]));
				if (lr == root || is_param(lr) || is_param(root))
					break;
				continue;
			}
			if (!strcmp(J->op, "OpStore")) {
				int sr = ptr_root(idx(J->opnd[0]));
				if (sr == root || is_param(sr) || is_param(root))
					break;	/* (a partial overwrite: keep it simple) */
				continue;
			}
			/* anything that ends the block or could read it */
			if (!strcmp(J->op, "OpLabel") || !strncmp(J->op, "OpBranch",
			    8) || !strcmp(J->op, "OpReturn") ||
			    !strcmp(J->op, "OpReturnValue") || !strcmp(J->op, "OpSwitch")
			    || !strcmp(J->op, "OpKill") ||
			    !strcmp(J->op, "OpFunctionCall") ||
			    !strncmp(J->op, "OpAtomic", 8) ||
			    !strcmp(J->op, "OpCopyMemory") ||
			    !strcmp(J->op, "OpFunctionEnd") ||
			    !strcmp(J->op, "OpUnreachable") ||
			    !strncmp(J->op, "OpTerminate", 11))
				break;
		}
	}
	return n;
}

/* the module's unsigned scalar integer type of width w, or -1 */
static int
uint_type(int w)
{
	int i;

	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_TYPE && V.id[i].t.k == S_INT &&
		    V.id[i].t.w == w && !V.id[i].t.sign)
			return i;
	return -1;
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
	/*
	 * a signed division or remainder of operands that are never
	 * negative (the divisor never 0) is the unsigned one: trunc(a/b) =
	 * floor(a/b) for a >= 0, b > 0, and the bits read as unsigned are
	 * the same numbers.  SAT decides it under the assumptions; the
	 * model's sdiv/udiv (srem/irem) prices decide whether it is wanted.
	 */
	if ((!strcmp(I->op, "OpSDiv") || !strcmp(I->op, "OpSRem") ||
	    !strcmp(I->op, "OpSMod")) && nargs == 2 && n < max &&
	    !(const_of(arg[1], &cv) && is_pow2(cv))) {
		int ut = uint_type(w), nb = 1 + !const_of(arg[1], &cv);
		if (ut >= 0) {
			const char *uo = I->op[3] == 'D' ? "OpUDiv" : "OpUMod";
			n += ins_cand(&cs[n], uo, NULL, rt, arg[0], arg[1], -1,
			    I->op[3] == 'D' ?
			    "operands never negative: the unsigned divide" :
			    "operands never negative: the unsigned remainder");
			cs[n - 1].via_u = 1;
			cs[n - 1].ut = ut;
			cs[n - 1].cost = ins_cost_b(uo, NULL, ut, arg[1]) +
			    (nb + 1) * model_active()->c[C_BITCAST];
		}
	}
	/*
	 * a compare of the two arms and a select of them is a min or a max
	 * (x < y ? x : y = min(x, y): integers are totally ordered); the
	 * compare goes too when the select is its only use
	 */
	if (!strcmp(I->op, "OpSelect") && nargs == 3 && V.glsl >= 0 &&
	    n < max && w > 1) {
		int cd = SP.def[arg[0]];
		static const char *const cmps[] = {"OpSLessThan",
		    "OpSLessThanEqual", "OpSGreaterThan", "OpSGreaterThanEqual",
		    "OpULessThan", "OpULessThanEqual", "OpUGreaterThan",
		    "OpUGreaterThanEqual", NULL};
		if (cd >= 0 && !SP.dead[cd] && in_list(V.in[cd].op, cmps) &&
		    V.in[cd].nop >= 3) {
			const char *co = V.in[cd].op;
			int x = res_of(idx(V.in[cd].opnd[1]));
			int y = res_of(idx(V.in[cd].opnd[2]));
			int lt = strstr(co, "Less") != NULL, sg = co[2] == 'S';
			if ((arg[1] == x && arg[2] == y) ||
			    (arg[1] == y && arg[2] == x)) {
				int mn = lt == (arg[1] == x);
				n += ins_cand(&cs[n], "OpExtInst", sg ? (mn ? "SMin" :
				    "SMax") : (mn ? "UMin" : "UMax"), rt, arg[1], arg[2],
				    -1, mn ? "a compare and select of its operands is a "
				    "min" : "a compare and select of its operands is a "
				    "max");
				cs[n - 1].extra = SP.uses[arg[0]] == 1 ?
				    ins_cost_k(cd) : 0;
			}
		}
	}
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
	/*
	 * shift chains: (y << c1) << c2 is y << (c1 + c2); (y << c1) <<
	 * (c2 - n) is y << ((c1 + c2) - n), one subtract in place of a
	 * shift when the inner ones are used only here
	 */
	if ((!strcmp(I->op, "OpShiftLeftLogical") ||
	    !strcmp(I->op, "OpShiftRightLogical")) && n < max) {
		int d = SP.def[arg[0]], ds = SP.def[arg[1]];
		uint64_t c1, c2;
		if (d >= 0 && !SP.dead[d] && !strcmp(V.in[d].op, I->op) &&
		    const_of(res_of(idx(V.in[d].opnd[2])), &c1)) {
			int y = res_of(idx(V.in[d].opnd[1]));
			double inner = SP.uses[arg[0]] == 1 ?
			    ins_cost(V.in[d].op, NULL, rt) : 0;
			if (const_of(arg[1], &c2) && c1 + c2 < (uint64_t)w) {
				n += ins_cand(&cs[n], I->op, NULL, rt, y,
				    const_id(V.id[arg[1]].type, c1 + c2), -1,
				    "two shifts by constants are one");
				cs[n - 1].extra = inner;
			} else if (ds >= 0 && !SP.dead[ds] &&
			    !strcmp(V.in[ds].op, "OpISub") &&
			    const_of(res_of(idx(V.in[ds].opnd[1])), &c2) && n < max) {
				int nn = res_of(idx(V.in[ds].opnd[2]));
				int st = V.id[arg[1]].type;
				n += ins_cand(&cs[n], I->op, NULL, rt, y, nn, -1,
				    "a shift of a shift is one shift by the sum");
				snprintf(cs[n - 1].pre_op, sizeof cs[n - 1].pre_op,
				    "OpISub");
				cs[n - 1].pre_a[0] = const_id(st, (c1 + c2) &
				    wmask(int_width(st)));
				cs[n - 1].pre_a[1] = nn;
				cs[n - 1].cost += ins_cost("OpISub", NULL, st);
				cs[n - 1].extra = inner + (SP.uses[arg[1]] == 1 ?
				    ins_cost("OpISub", NULL, st) : 0);
			}
		}
	}
	/* an affine chain: one multiply (and an add) of its base */
	if (w > 1 && n + 1 < max && (!strcmp(I->op, "OpIAdd") ||
	    !strcmp(I->op, "OpISub") || !strcmp(I->op, "OpIMul") ||
	    !strcmp(I->op, "OpSNegate"))) {
		struct aff f = aff_of(I->res, w, 24);
		if (f.base >= 0 && f.base != res_of(I->res) && f.a != 1 &&
		    (f.a & (f.a - 1)) != 0) {
			double dead = aff_dead_cost(I->res, f.base, rt);
			int ca = const_id(rt, f.a);
			if (ca >= 0 && f.b == 0) {
				n += ins_cand(&cs[n], "OpIMul", NULL, rt, f.base, ca, -1,
				    "a chain of shifts and adds is one multiply");
				cs[n - 1].extra = dead;
				cs[n - 1].algebra = 1;
			} else if (ca >= 0) {
				int cb = const_id(rt, f.b);
				if (cb >= 0) {
					n += ins_cand(&cs[n], "OpIAdd", NULL, rt, cb, f.base,
					    -1, "a chain of shifts and adds is a multiply-add");
					snprintf(cs[n - 1].pre_op, sizeof cs[n - 1].pre_op,
					    "OpIMul");
					cs[n - 1].pre_a[0] = f.base;
					cs[n - 1].pre_a[1] = ca;
					cs[n - 1].cost += ins_cost("OpIMul", NULL, rt);
					cs[n - 1].extra = dead;
					cs[n - 1].algebra = 1;
				}
			}
		}
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
	if (c->via_u) {
		uint64_t v;
		snprintf(b, n, "OpBitcast %s (%s %s (OpBitcast %s %s) %s)",
		    I->opnd[0], c->op, V.id[c->ut].tok, V.id[c->ut].tok,
		    V.id[c->a[0]].tok, const_of(c->a[1], &v) ?
		    V.id[const_id(c->ut, v)].tok : V.id[c->a[1]].tok);
		return;
	}
	if (c->ext[0])
		at = snprintf(b, n, "OpExtInst %s %s %s", I->opnd[0],
		    V.glsl >= 0 ? V.id[V.glsl].tok : "%glsl", c->ext);
	else
		at = snprintf(b, n, "%s %s", c->op, I->opnd[0]);
	for (i = 0; i < c->na && (size_t)at < n; i++)
		if (i == 1 && c->pre_op[0])
			at += snprintf(b + at, n - (size_t)at, " (%s %s %s)",
			    c->pre_op, V.id[c->pre_a[0]].tok, V.id[c->pre_a[1]].tok);
		else
			at += snprintf(b + at, n - (size_t)at, " %s",
			    V.id[c->a[i]].tok);
}

/*
 * A rewrite any driver's back end makes on its own: a strength reduction
 * by a constant (a multiply, unsigned divide or remainder by a power of
 * two, an extract at bit 0, a shift then a mask, two constant shifts), an
 * operation with an identity operand (x + 0, x * 1, x & ~0, x | 0, x ^ 0,
 * x << 0, x >> 0, x / 1), one whose operands are all constants, or x - x
 * and x ^ x.  These cost the GPU nothing to leave in the source, so they
 * are listed apart and kept out of the headline saving.
 */
static int
driver_trivial(int k, const struct cand *c)
{
	static const char *const red[] = {
		"a multiply by a power of two is a shift",
		"a divide by a power of two is a shift",
		"a remainder by a power of two is a mask",
		"an extract at bit 0 is a mask",
		"a shift then a mask is one bitfield extract",
		"two shifts by constants are one", NULL
	};
	const struct sins *I = &V.in[k];
	int base = first_arg(I), na = I->nop - base, i, nk = 0, w;
	uint64_t cv[2];
	int isk[2] = {0, 0};

	for (i = 0; red[i]; i++)
		if (c->why == red[i] || !strcmp(c->why, red[i]))
			return 1;
	if (c->kind == RW_INS || na < 1 || na > 2)
		return 0;
	w = int_width(idx(I->opnd[0]));
	for (i = 0; i < na; i++)
		if (const_of(res_of(idx(I->opnd[base + i])), &cv[i])) {
			isk[i] = 1;
			nk++;
		}
	if (nk == na)
		return 1;	/* constant folding */
	if (na == 2 && res_of(idx(I->opnd[base])) ==
	    res_of(idx(I->opnd[base + 1])) && (!strcmp(I->op, "OpISub") ||
	    !strcmp(I->op, "OpBitwiseXor")))
		return 1;
	if (na != 2 || nk != 1)
		return 0;
	if (c->kind == RW_ID) {
		uint64_t v = isk[0] ? cv[0] : cv[1];
		int right = isk[1];	/* the constant is the second operand */
		if ((!strcmp(I->op, "OpIAdd") || !strcmp(I->op, "OpBitwiseOr") ||
		    !strcmp(I->op, "OpBitwiseXor")) && v == 0)
			return 1;
		if ((!strcmp(I->op, "OpISub") || !strncmp(I->op, "OpShift", 7)) &&
		    right && v == 0)
			return 1;
		if (!strcmp(I->op, "OpIMul") && v == 1)
			return 1;
		if ((!strcmp(I->op, "OpUDiv") || !strcmp(I->op, "OpSDiv")) &&
		    right && v == 1)
			return 1;
		if (!strcmp(I->op, "OpBitwiseAnd") && w > 0 && v == wmask(w))
			return 1;
	}
	if (c->kind == RW_CONST) {
		uint64_t v = isk[0] ? cv[0] : cv[1];
		if ((!strcmp(I->op, "OpIMul") || !strcmp(I->op, "OpBitwiseAnd")) &&
		    v == 0)
			return 1;
		if (!strcmp(I->op, "OpBitwiseOr") && w > 0 && v == wmask(w))
			return 1;
	}
	return 0;
}

/* a new instruction of type ty, emitted just before k: its id */
static int
new_ins(int k, const char *op, int ty, int a, int b)
{
	char nm[40];
	int nid_, nk;
	struct sins N;

	snprintf(nm, sizeof nm, "%%ucas_t%d", SP.nnewk + V.nin);
	nid_ = idx(nm);
	memset(&N, 0, sizeof N);
	snprintf(N.op, sizeof N.op, "%s", op);
	N.res = nid_;
	N.opnd[0] = strdup(V.id[ty].tok);
	N.opnd[1] = strdup(V.id[a].tok);
	N.nop = 2;
	if (b >= 0)
		N.opnd[N.nop++] = strdup(V.id[b].tok);
	N.text = xmalloc(200);
	snprintf(N.text, 200, "%s = %s %s %s%s%s", nm, N.op, N.opnd[0],
	    N.opnd[1], b >= 0 ? " " : "", b >= 0 ? N.opnd[2] : "");
	if (V.nin == V.cap) {
		V.cap *= 2;
		V.in = xrealloc(V.in, (size_t)V.cap * sizeof *V.in);
	}
	nk = V.nin++;
	V.in[nk] = N;
	SP.def[nid_] = nk;
	V.id[nid_].type = ty;
	V.id[nid_].kind = K_VALUE;
	SP.before[nk] = k;
	return nid_;
}

/* apply: alias, constant alias, or the instruction rewritten in place */
static void
apply(int k, const struct cand *c)
{
	struct sins *I = &V.in[k];
	int rt = idx(I->opnd[0]), i;
	char buf[64];
	struct cand c2 = *c;

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
	if (c->pre_op[0]) {
		/* the new subtract, emitted just before k */
		c2.a[1] = new_ins(k, c->pre_op, V.id[res_of(c->pre_a[0])].type,
		    c->pre_a[0], c->pre_a[1]);
		I = &V.in[k];	/* (V.in may have moved) */
		c = &c2;
	}
	if (c->via_u) {
		/* on the bits as unsigned: bitcasts in, the operation, a
		 * bitcast out in place of k */
		uint64_t v;
		int a0 = new_ins(k, "OpBitcast", c->ut, c->a[0], -1), a1, r;
		a1 = const_of(c->a[1], &v) ? const_id(c->ut, v) :
		    new_ins(k, "OpBitcast", c->ut, c->a[1], -1);
		r = new_ins(k, c->op, c->ut, a0, a1);
		I = &V.in[k];
		memset(&c2, 0, sizeof c2);
		c2.kind = RW_INS;
		snprintf(c2.op, sizeof c2.op, "OpBitcast");
		c2.a[0] = r;
		c2.na = 1;
		c = &c2;
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

/* the source line an instruction came from (OpLine before it), or 0 */
static int
ins_line(int k)
{
	for (; k >= 0; k--) {
		if (!strcmp(V.in[k].op, "OpLine") && V.in[k].nop >= 2)
			return atoi(V.in[k].opnd[1]);
		if (!strcmp(V.in[k].op, "OpFunction"))
			break;
	}
	return 0;
}

/* " (GLSL line 3: s.x = ...)" from OpSource's text, or "" */
static void
src_line(int ln, char *b, size_t n)
{
	const char *t = src_text, *m;
	int at = 1;

	b[0] = 0;
	if (t == NULL || ln <= 0 || ln >= 10000)
		return;
	/* glslc.c's "#line N" before the source: the line after it is N */
	if ((m = strstr(t, "#line ")) != NULL && atoi(m + 6) > 0 &&
	    atoi(m + 6) < 10000) {
		at = atoi(m + 6);
		t = strchr(m, '\n') ? strchr(m, '\n') + 1 : m + strlen(m);
	}
	while (*t && at < ln) {
		if (*t == '\n')
			at++;
		t++;
	}
	if (*t == 0)
		return;
	while (*t == ' ' || *t == '\t')
		t++;
	{
		size_t len = strcspn(t, "\n");
		if (len > 90)
			len = 90;
		snprintf(b, n, "      (line %d: %.*s)\n", ln, (int)len, t);
	}
}


/* ---- range search: one bounded input, every value checked ---- */

/* the pure cone under id: its free inputs and its cost */
static void
cone_of(int id, int *leaves, int *nl, double *cost, char *seen, int rt)
{
	int k, i;

	id = res_of(id);
	if (seen[id])
		return;
	seen[id] = 1;
	if (V.id[id].kind == K_CONST && !V.id[id].spec)
		return;
	k = SP.def[id];
	if (k < 0 || SP.dead[k] || !pure_ins(k)) {
		if (*nl < 8)
			leaves[*nl] = id;
		(*nl)++;
		return;
	}
	*cost += ins_cost(V.in[k].op, !strcmp(V.in[k].op, "OpExtInst") ?
	    V.in[k].opnd[2] : NULL, rt);
	for (i = first_arg(&V.in[k]); i < V.in[k].nop; i++)
		cone_of(idx(V.in[k].opnd[i]), leaves, nl, cost, seen, rt);
}

/* f over the domain: the root's value with the input forced */
static uint64_t *
range_values(int root, int leaf, int64_t lo, int64_t hi)
{
	uint64_t *f = xmalloc((size_t)(hi - lo + 1) * sizeof *f);
	int64_t x;

	SP.force_id = leaf;
	for (x = lo; x <= hi; x++) {
		SP.force_val = (uint64_t)x;
		SP.stamp++;
		f[x - lo] = evalid(root);
	}
	SP.force_id = -1;
	return f;
}

static int
has_mul(int id, char *seen)
{
	int k, i;

	id = res_of(id);
	if (seen[id])
		return 0;
	seen[id] = 1;
	k = SP.def[id];
	if (k < 0 || !pure_ins(k))
		return 0;
	if (!strcmp(V.in[k].op, "OpIMul") || !strcmp(V.in[k].op, "OpUDiv") ||
	    !strcmp(V.in[k].op, "OpSDiv"))
		return 1;
	for (i = first_arg(&V.in[k]); i < V.in[k].nop; i++)
		if (has_mul(idx(V.in[k].opnd[i]), seen))
			return 1;
	return 0;
}

/*
 * For a stored or returned value with one bounded input (at most 2^20
 * values): what it is (floor/round/ceil of a ratio), and cheaper forms
 * from a few template families, each checked on every input.
 */
static void
range_search(int root, int line, struct sbuf *out)
{
	int leaves[8], nl = 0, w, mul;
	double cost = 0, cadd, cshr;
	char *seen = xcalloc((size_t)(V.nid + 1024), 1);
	int64_t lo, hi, x, n;
	uint64_t *f, m;
	const char *nm;
	char sl[160], nmb[64];
	int rt = V.id[res_of(root)].type, found = 0;
	long tried = 0;

	w = int_width(rt);
	cone_of(root, leaves, &nl, &cost, seen, rt);
	if (nl != 1 || w < 2 || !assumed(leaves[0], &lo, &hi) || lo < 0 ||
	    hi - lo > (1 << 20) || cost <= 0) {
		free(seen);
		return;
	}
	memset(seen, 0, (size_t)(V.nid + 1024));
	mul = has_mul(root, seen);
	free(seen);
	m = wmask(w);
	n = hi - lo + 1;
	f = range_values(root, leaves[0], lo, hi);
	/* the input's name, as the user wrote it */
	nm = NULL;
	{
		int j;
		for (j = 0; j < SP.nas && nm == NULL; j++) {
			int64_t a, b;
			if (assumed(leaves[0], &a, &b))
				nm = SP.as[j].name;
		}
	}
	snprintf(nmb, sizeof nmb, "%s", nm ? nm : "x");
	src_line(line, sl, sizeof sl);
	sb_printf(out, "  range search on a result (one input, %s in [%lld, "
	    "%lld]; every one of the %lld inputs checked):\n", nmb, (long long)lo,
	    (long long)hi, (long long)n);
	if (sl[0])
		sb_printf(out, "%s", sl);
	/* what it is: f(lo) + floor/round/ceil((x - lo) * p / q) */
	{
		uint64_t p = (f[n - 1] - f[0]) & m, q = (uint64_t)(hi - lo);
		static const char *const kind[] = {"floor", "round", "ceil"};
		int kk;
		for (kk = 0; kk < 3 && q > 0 && p > 0; kk++) {
			int ok = 1;
			for (x = 0; x < n && ok; x++) {
				uint64_t num = (uint64_t)x * p, v;
				v = kk == 0 ? num / q : kk == 1 ? (2 * num + q) / (2 * q) :
				    (num + q - 1) / q;
				ok = ((f[0] + v) & m) == f[x];
			}
			if (ok) {
				if (lo == 0 && f[0] == 0)
					sb_printf(out, "    it is %s(%s * %llu / %llu)\n",
					    kind[kk], nmb, (unsigned long long)p,
					    (unsigned long long)q);
				else
					sb_printf(out, "    it is %llu + %s((%s - %lld) * %llu "
					    "/ %llu)\n", (unsigned long long)f[0], kind[kk], nmb,
					    (long long)lo, (unsigned long long)p,
					    (unsigned long long)q);
				break;
			}
		}
	}
	/* template families, K solved exactly where it is an offset */
	cadd = ins_cost("OpIAdd", NULL, rt);
	cshr = ins_cost("OpShiftRightLogical", NULL, rt);
	{
		int S, S1, sg;
		int64_t K0;
		/* (h + K) >> S, h = x, x + (x >> S1), x - (x >> S1) */
		for (S1 = 0; S1 < w && found < 4; S1++)
			for (sg = (S1 ? -1 : 1); sg <= 1 && found < 4; sg += 2)
				for (S = 0; S < w && found < 4; S++) {
					int64_t klo = INT64_MIN, khi = INT64_MAX;
					double c = (S1 ? cshr + cadd : 0) + cadd + (S ? cshr : 0);
					tried++;
					if (c > cost || (c == cost && !mul))
						continue;
					for (x = 0; x < n && klo < khi; x++) {
						int64_t xv = lo + x, h = xv + sg * (S1 ? (xv >> S1) : 0);
						int64_t a = (int64_t)(f[x] << S) - h;
						int64_t b = (int64_t)((f[x] + 1) << S) - h;
						if (a > klo)
							klo = a;
						if (b < khi)
							khi = b;
					}
					if (klo < khi && klo + (hi + (hi >> (S1 ? S1 : 63))) <
					    (int64_t)m) {
						char e[160];
						int64_t K = klo < 0 && khi > 0 ? 0 : klo;
						if (S1)
							snprintf(e, sizeof e, "(%s %c (%s >> %d) %c %lld)"
							    " >> %d", nmb, sg > 0 ? '+' : '-', nmb, S1,
							    K < 0 ? '-' : '+', (long long)(K < 0 ? -K : K),
							    S);
						else
							snprintf(e, sizeof e, "(%s %c %lld) >> %d", nmb,
							    K < 0 ? '-' : '+', (long long)(K < 0 ? -K : K),
							    S);
						sb_printf(out, "    %s: %s   (cost %g)\n", c < cost ?
						    "cheaper" : "same cost, no multiply", e, c);
						found++;
					}
				}
		/* t = x + K0; (t +- (t >> S1)) >> S */
		for (K0 = 0; K0 < 1024 && found < 4 &&
		    (double)clock() / CLOCKS_PER_SEC < SP.deadline; K0++)
			for (S1 = 1; S1 < w && found < 4; S1++)
				for (sg = -1; sg <= 1 && found < 4; sg += 2)
					for (S = 1; S < w && found < 4; S++) {
						double c = 2 * cadd + 2 * cshr;
						int ok = 1;
						tried++;
						if (c > cost || (c == cost && !mul))
							continue;
						for (x = 0; x < n && ok; x++) {
							int64_t t = lo + x + K0;
							ok = (uint64_t)((t + sg * (t >> S1)) >> S) == f[x];
						}
						if (ok) {
							sb_printf(out, "    %s: t = %s + %lld; (t %c (t >> "
							    "%d)) >> %d   (cost %g)\n", c < cost ?
							    "cheaper" : "same cost, no multiply", nmb,
							    (long long)K0, sg > 0 ? '+' : '-', S1, S, c);
							found++;
						}
					}
	}
	if (!found)
		sb_printf(out, "    nothing cheaper than its cost %g among %ld forms "
		    "of (x + K) >> S, (x +- (x >> S1) + K) >> S and t = x + K; (t "
		    "+- (t >> S1)) >> S\n", cost, tried);
	else
		sb_printf(out, "    (its cost now: %g; each form above equals it on "
		    "every input in the range)\n", cost);
	free(f);
}

/* ---- dot products: sums of products as one instruction (dot.c) ---- */

static int
is_idot(int k)
{
	int su, sa, sb, sat;

	return V.in[k].res >= 0 && idot_parts(V.in[k].op, &su, &sa, &sb, &sat);
}

/* the capabilities and the extension, those the module lacks */
static void
dot_caps_out(struct sbuf *out)
{
	static const char *const cap[] = {"DotProduct",
		"DotProductInput4x8BitPacked", NULL};
	int i, k, have;

	for (i = 0; cap[i]; i++) {
		have = 0;
		for (k = 0; k < SP.orig_nin; k++)
			if (!strcmp(V.in[k].op, "OpCapability") && V.in[k].nop >= 1 &&
			    (!strcmp(V.in[k].opnd[0], cap[i]) ||
			    (!strncmp(V.in[k].opnd[0], cap[i], strlen(cap[i])) &&
			    !strcmp(V.in[k].opnd[0] + strlen(cap[i]), "KHR"))))
				have = 1;
		if (!have)
			sb_printf(out, "OpCapability %s\n", cap[i]);
	}
	have = 0;
	for (k = 0; k < SP.orig_nin; k++)
		if (!strcmp(V.in[k].op, "OpExtension") && V.in[k].nop >= 1 &&
		    strstr(V.in[k].opnd[0], "SPV_KHR_integer_dot_product"))
			have = 1;
	if (!have)
		sb_puts(out, "OpExtension \"SPV_KHR_integer_dot_product\"\n");
}

/*
 * A maximal sum of pure integer instructions (an add, subtract, negate
 * or multiply whose value is used by something other than an add or a
 * subtract) is read into a ucas function: its free inputs (loads, phis,
 * anything else) are the parameters, with their types and the assumed
 * ranges.  dot.c searches it under the active model and proves what it
 * finds; the result is written back as SPIR-V before the sum's last
 * instruction: OpUDot, OpSDot, OpSUDot (or their AccSat forms) on
 * PackedVectorFormat4x8Bit words, with the shifts, ors and constants
 * that pack them, and the module gets the capabilities and the
 * extension.  The translation is checked against the interpreter's
 * arithmetic on 64 samples before anything is written.
 */
#define DOTMAXP MAXP

struct spvdot {
	int leaf[DOTMAXP], nleaf;
	struct func f;
	nid *memo;
	int fail;
	int k;			/* new instructions go before this one */
	struct { nid n; int id; } em[256];
	int nem;
	int ty32[2];		/* the module's u32 and i32 type ids */
	int ndot;		/* dot instructions written */
};

static int dotcaps;		/* the module needs the dot capabilities */

static ty_t
spv_ty(int t)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_INT || T(t)->w > 64)
		return 0;
	return TY(T(t)->sign ? K_SINT : K_UINT, T(t)->w);
}

static nid
spv_leaf(struct spvdot *c, int id)
{
	ty_t t = spv_ty(V.id[id].type);
	int j;
	int64_t lo, hi;

	for (j = 0; j < c->nleaf; j++)
		if (c->leaf[j] == id)
			return mk_var((uint32_t)j, c->f.p[j].name, c->f.p[j].ty);
	if (!t || c->nleaf == DOTMAXP) {
		c->fail = 1;
		return mk_const_i(TY(K_UINT, 32), 0);
	}
	j = c->nleaf++;
	c->leaf[j] = id;
	memset(&c->f.p[j], 0, sizeof c->f.p[j]);
	c->f.p[j].name = sym_cstr(V.id[id].name);
	c->f.p[j].ty = t;
	if (assumed(id, &lo, &hi)) {
		c->f.p[j].ranged = 1;
		c->f.p[j].lo = lo;
		c->f.p[j].hi = hi;
	}
	c->f.np = c->nleaf;
	return mk_var((uint32_t)j, c->f.p[j].name, t);
}

static nid
spv_node(struct spvdot *c, int id, int depth)
{
	int k, base;
	const struct sins *I;
	ty_t t, ut, st;
	nid x[3] = {0, 0, 0}, r = 0;
	uint64_t kv;

	id = res_of(id);
	if (c->memo[id])
		return c->memo[id];
	t = spv_ty(V.id[id].type);
	if (V.id[id].kind == K_CONST && !V.id[id].spec && t)
		return c->memo[id] = mk_const_i(t, int_norm(t,
		    (int64_t)V.id[id].c[0]));
	k = SP.def[id];
	if (!t || k < 0 || SP.dead[k] || !pure_ins(k) || depth > 16)
		return c->memo[id] = spv_leaf(c, id);
	I = &V.in[k];
	base = first_arg(I);
	ut = TY(K_UINT, TYW(t));
	st = TY(K_SINT, TYW(t));
#define OPN(q) (I->opnd[base + (q)])
#define IS(o) (!strcmp(I->op, o))
	if (IS("OpIAdd") || IS("OpISub") || IS("OpIMul") || IS("OpBitwiseAnd") ||
	    IS("OpBitwiseOr") || IS("OpBitwiseXor")) {
		int op = IS("OpIAdd") ? O_ADD : IS("OpISub") ? O_SUB : IS("OpIMul") ?
		    O_MUL : IS("OpBitwiseAnd") ? O_AND : IS("OpBitwiseOr") ? O_OR :
		    O_XOR;
		x[0] = mk_cast(t, spv_node(c, idx(OPN(0)), depth + 1));
		x[1] = mk_cast(t, spv_node(c, idx(OPN(1)), depth + 1));
		r = mk_bin(op, x[0], x[1]);
	} else if (IS("OpSNegate") || IS("OpNot")) {
		x[0] = mk_cast(t, spv_node(c, idx(OPN(0)), depth + 1));
		r = mk_un(IS("OpNot") ? O_NOT : O_NEG, x[0]);
	} else if (IS("OpShiftLeftLogical") || IS("OpShiftRightLogical") ||
	    IS("OpShiftRightArithmetic")) {
		ty_t vt = IS("OpShiftRightLogical") ? ut :
		    IS("OpShiftRightArithmetic") ? st : t;
		x[0] = mk_cast(vt, spv_node(c, idx(OPN(0)), depth + 1));
		x[1] = spv_node(c, idx(OPN(1)), depth + 1);
		r = mk_bin(IS("OpShiftLeftLogical") ? O_SHL : O_SHR, x[0], x[1]);
	} else if (IS("OpBitFieldUExtract") && const_of(idx(OPN(1)), &kv) &&
	    kv < 64 && const_of(idx(OPN(2)), &kv)) {
		uint64_t off, cnt;
		const_of(idx(OPN(1)), &off);
		const_of(idx(OPN(2)), &cnt);
		if (cnt == 0 || off + cnt > (uint64_t)TYW(t))
			return c->memo[id] = spv_leaf(c, id);
		x[0] = mk_cast(ut, spv_node(c, idx(OPN(0)), depth + 1));
		if (off)
			x[0] = mk_bin(O_SHR, x[0], mk_const_i(ut, (int64_t)off));
		r = cnt < (uint64_t)TYW(t) ? mk_bin(O_AND, x[0], mk_const_i(ut,
		    (int64_t)wmask((int)cnt))) : x[0];
	} else if (IS("OpUConvert") || IS("OpSConvert") || IS("OpBitcast")) {
		int a = res_of(idx(OPN(0)));
		ty_t at = spv_ty(V.id[a].type);
		if (!at)
			return c->memo[id] = spv_leaf(c, id);
		x[0] = spv_node(c, a, depth + 1);
		if (IS("OpUConvert"))
			x[0] = mk_cast(TY(K_UINT, TYW(at)), x[0]);
		else if (IS("OpSConvert"))
			x[0] = mk_cast(TY(K_SINT, TYW(at)), x[0]);
		r = x[0];
	} else
		return c->memo[id] = spv_leaf(c, id);
#undef OPN
#undef IS
	return c->memo[id] = mk_cast(t, r);
}

/* a new instruction "res = op type a b ..." before c->k; its id */
static int
dot_ins(struct spvdot *c, const char *op, int rt, const char *const *opnd,
    int n)
{
	char nm[40];
	struct sins N;
	int nid_, nk, i, at;

	if (V.nin >= SP.orig_nin + 1000 || V.nid + 8 >= SP.orig_nid + 1024) {
		c->fail = 1;
		return -1;
	}
	snprintf(nm, sizeof nm, "%%ucas_d%d", V.nin);
	nid_ = idx(nm);
	memset(&N, 0, sizeof N);
	snprintf(N.op, sizeof N.op, "%s", op);
	N.res = nid_;
	N.opnd[0] = strdup(V.id[rt].tok);
	for (i = 0; i < n; i++)
		N.opnd[1 + i] = strdup(opnd[i]);
	N.nop = 1 + n;
	N.text = xmalloc(400);
	at = snprintf(N.text, 400, "%s = %s %s", nm, op, V.id[rt].tok);
	for (i = 0; i < n && at < 380; i++)
		at += snprintf(N.text + at, 400 - (size_t)at, " %s", opnd[i]);
	if (V.nin == V.cap) {
		V.cap *= 2;
		V.in = xrealloc(V.in, (size_t)V.cap * sizeof *V.in);
	}
	nk = V.nin++;
	V.in[nk] = N;
	SP.def[nid_] = nk;
	V.id[nid_].type = rt;
	V.id[nid_].kind = K_VALUE;
	SP.before[nk] = c->k;
	return nid_;
}

static int
ty_id(struct spvdot *c, ty_t t)
{
	if (TYW(t) != 32 || (TYK(t) != K_UINT && TYK(t) != K_SINT))
		return -1;
	return c->ty32[TYK(t) == K_SINT];
}

/* the SPIR-V id computing node n (the ucas form of the rewrite) */
static int
dot_emit(struct spvdot *c, nid n)
{
	int i, op = N.op[n], rt, a, b, r = -1;
	const char *o[4];

	for (i = 0; i < c->nem; i++)
		if (c->em[i].n == n)
			return c->em[i].id;
	if (c->fail)
		return -1;
	if (op == O_VAR) {
		int leaf = c->leaf[N.ki[n]];
		rt = ty_id(c, N.ty[n]);
		if (rt < 0 || spv_ty(V.id[leaf].type) != N.ty[n]) {
			c->fail = 1;
			return -1;
		}
		r = leaf;
		goto done;
	}
	rt = ty_id(c, N.ty[n]);
	if (rt < 0) {
		c->fail = 1;
		return -1;
	}
	if (op == O_CONST) {
		r = const_id(rt, (uint64_t)N.ki[n] & 0xffffffffULL);
		if (r < 0)
			c->fail = 1;
		goto done;
	}
	if (op == O_CAST) {
		ty_t at = N.ty[N.a[n]];
		a = dot_emit(c, N.a[n]);
		if (a < 0 || TYW(at) != 32 || !TY_ISINT(at)) {
			c->fail = 1;
			return -1;
		}
		if (at == N.ty[n])
			r = a;
		else {
			o[0] = V.id[a].tok;
			r = dot_ins(c, "OpBitcast", rt, o, 1);
		}
		goto done;
	}
	if (IS_DOT(op)) {
		const struct dotform *df = dot_form(op);
		char nm[40];
		int acc = -1;
		if (df->lanes != 4 || df->bits != 8) {
			c->fail = 1;	/* only the packed 4 x 8 forms are written */
			return -1;
		}
		a = dot_emit(c, N.a[n]);
		b = dot_emit(c, N.b[n]);
		if (df->sat)
			acc = dot_emit(c, N.c[n]);
		if (a < 0 || b < 0 || (df->sat && acc < 0))
			return -1;
		snprintf(nm, sizeof nm, "Op%sDot%s", df->sa && df->sb ? "S" :
		    df->sa ? "SU" : "U", df->sat ? "AccSat" : "");
		o[0] = V.id[a].tok;
		o[1] = V.id[b].tok;
		i = 2;
		if (df->sat)
			o[i++] = V.id[acc].tok;
		o[i++] = "PackedVectorFormat4x8Bit";
		r = dot_ins(c, nm, rt, o, i);
		dotcaps = 1;
		c->ndot++;
		goto done;
	}
	{
		const char *sop = NULL;
		switch (op) {
		case O_ADD: sop = "OpIAdd"; break;
		case O_SUB: sop = "OpISub"; break;
		case O_MUL: sop = "OpIMul"; break;
		case O_AND: sop = "OpBitwiseAnd"; break;
		case O_OR: sop = "OpBitwiseOr"; break;
		case O_XOR: sop = "OpBitwiseXor"; break;
		case O_SHL: sop = "OpShiftLeftLogical"; break;
		case O_SHR: sop = TYK(N.ty[n]) == K_SINT ?
		    "OpShiftRightArithmetic" : "OpShiftRightLogical"; break;
		case O_NEG: sop = "OpSNegate"; break;
		case O_NOT: sop = "OpNot"; break;
		}
		if (!sop) {
			c->fail = 1;
			return -1;
		}
		a = dot_emit(c, N.a[n]);
		if (a < 0)
			return -1;
		o[0] = V.id[a].tok;
		if (opinfo[op].arity == 2) {
			b = dot_emit(c, N.b[n]);
			if (b < 0)
				return -1;
			o[1] = V.id[b].tok;
		}
		r = dot_ins(c, sop, rt, o, opinfo[op].arity);
	}
done:
	if (r >= 0 && c->nem < 256) {
		c->em[c->nem].n = n;
		c->em[c->nem++].id = r;
	}
	return r;
}

/* the translation agrees with the interpreter on 64 samples */
static int
dot_translation_ok(struct spvdot *c, int root, nid body)
{
	struct plan p;
	int64_t in_i[MAXP * CH];
	double in_r[MAXP * CH];
	int s, j, ok = 1;
	nid roots[1];

	roots[0] = body;
	plan_build(&p, roots, 1, 0);
	memset(in_r, 0, sizeof in_r);
	for (s = 0; s < 64 && ok; s++) {
		SP.sample = 300 + s;
		SP.stamp++;
		for (j = 0; j < c->nleaf; j++)
			in_i[(size_t)j * CH] = int_norm(c->f.p[j].ty,
			    (int64_t)evalid(c->leaf[j]));
		plan_run(&p, 1, in_i, in_r);
		if (COLF(&p, p.root[0])[0] & F_UNDEF)
			continue;
		ok = ((uint64_t)COLI(&p, p.root[0])[0] &
		    wmask(int_width(V.id[res_of(root)].type))) == evalid(root);
	}
	plan_free(&p);
	return ok;
}

static int
dot_type_ids(int *ty32)
{
	int i;

	ty32[0] = ty32[1] = -1;
	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_TYPE && V.id[i].t.k == S_INT &&
		    V.id[i].t.w == 32) {
			if (V.id[i].t.sign && ty32[1] < 0)
				ty32[1] = i;
			if (!V.id[i].t.sign && ty32[0] < 0)
				ty32[0] = i;
		}
	return ty32[0] >= 0 || ty32[1] >= 0;
}

/*
 * The pass.  Each rewrite goes to the log (from, to, why, gain) and its
 * lanes and proof to rep; the dot instructions' cost to *dcost.
 */
struct dotlog {
	char from[160], to[160];
	double gain, before, after;
	int line;
	char *desc, *proof;	/* malloc'd */
};

static int
dot_pass(struct sbuf *rep, struct dotlog *log, int maxlog)
{
	int k, n = 0, i, ty32[2];
	int *sumuse = xcalloc((size_t)(V.nid + 1024), sizeof *sumuse);
	int nmodel = 0;

	if (!dot_type_ids(ty32)) {
		free(sumuse);
		return 0;
	}
	for (i = O_DOT_FIRST; i <= O_DOT_LAST; i++)
		if (dot_form(i)->lanes == 4 && dot_form(i)->bits == 8 &&
		    cclass_available(model_active(), DOT_CLASS(i)))
			nmodel = 1;
	for (k = 0; k < SP.orig_nin; k++) {
		const struct sins *I = &V.in[k];
		if (SP.dead[k] || !pure_ins(k))
			continue;
		if (!strcmp(I->op, "OpIAdd") || !strcmp(I->op, "OpISub") ||
		    !strcmp(I->op, "OpSNegate"))
			for (i = 1; i < I->nop; i++)
				if (I->opnd[i][0] == '%')
					sumuse[res_of(idx(I->opnd[i]))]++;
	}
	for (k = 0; k < SP.orig_nin && (double)clock() / CLOCKS_PER_SEC <
	    SP.deadline; k++) {
		struct spvdot c;
		struct dotres r;
		const struct sins *I = &V.in[k];
		int root, fin, w;
		if (SP.dead[k] || !pure_ins(k))
			continue;
		if (strcmp(I->op, "OpIAdd") && strcmp(I->op, "OpISub") &&
		    strcmp(I->op, "OpIMul") && strcmp(I->op, "OpSNegate"))
			continue;
		root = I->res;
		w = int_width(V.id[root].type);
		if (w != 32 || SP.uses[root] <= sumuse[root])
			continue;
		memset(&c, 0, sizeof c);
		c.memo = xcalloc((size_t)(V.nid + 1024), sizeof *c.memo);
		c.k = k;
		c.ty32[0] = ty32[0];
		c.ty32[1] = ty32[1];
		c.f.anon = 1;
		c.f.body = spv_node(&c, root, 0);
		free(c.memo);
		if (c.fail || c.nleaf == 0)
			continue;
		if (!nmodel || n >= maxlog)
			continue;
		dot_search(&c.f, 1, &r);
		if (r.status != DOT_PROVED || !dot_translation_ok(&c, root,
		    r.body)) {
			dotres_free(&r);
			continue;
		}
		{
			int nin0 = V.nin, nid0 = V.nid, nk0 = SP.nnewk;
			fin = dot_emit(&c, r.body);
			if (c.fail || fin < 0) {
				/* take back what was written */
				for (i = nin0; i < V.nin; i++) {
					int j;
					for (j = 0; j < V.in[i].nop; j++)
						free(V.in[i].opnd[j]);
					free(V.in[i].text);
				}
				V.nin = nin0;
				(void)nid0;
				SP.nnewk = nk0;
				dotres_free(&r);
				continue;
			}
		}
		SP.alias[root] = fin;
		SP.dead[k] = 1;
		{
			char to[200];
			char sl[160];
			snprintf(to, sizeof to, "%s", shared_str(r.body, 150));
			snprintf(log[n].from, sizeof log[n].from, "%s", I->text);
			snprintf(log[n].to, sizeof log[n].to, "%s", to);
			log[n].gain = r.before - r.after;
			log[n].before = r.before;
			log[n].after = r.after;
			log[n].line = ins_line(k);
			log[n].desc = strdup(sb_str(&r.desc));
			log[n].proof = strdup(sb_str(&r.proof));
			sb_printf(rep, "    %s\n      ->  %s   (cost %g -> %g)\n",
			    I->text, to, (double)r.before, (double)r.after);
			src_line(ins_line(k), sl, sizeof sl);
			if (sl[0])
				sb_printf(rep, "%s", sl);
			{
				const char *s = sb_str(&r.desc), *e;
				while (*s) {
					e = strchr(s, '\n');
					if (!e)
						e = s + strlen(s);
					sb_printf(rep, "      %.*s\n", (int)(e - s), s);
					s = *e ? e + 1 : e;
				}
			}
			sb_printf(rep, "      %s\n", sb_str(&r.proof));
		}
		n++;
		dotres_free(&r);
	}
	free(sumuse);
	return n;
}

/* "n in [0, 32)" for the next simplify */
static struct {
	char name[40];
	int64_t lo, hi;
} pend_as[8];
static int npend_as;

void
spirv_assume(const char *name, int64_t lo, int64_t hi)
{
	if (npend_as < 8) {
		snprintf(pend_as[npend_as].name, sizeof pend_as[0].name, "%s",
		    name[0] == '%' ? name + 1 : name);
		pend_as[npend_as].lo = lo;
		pend_as[npend_as++].hi = hi;
	}
}

/* ---- float simplify (fsimp.c does the mathematics) ---- */

#include "fsimp.h"
#include "fuse.h"

static struct {
	char name[40];
	float lo, hi;
} pend_fas[8];
static int npend_fas;
static struct fxopt fopt;
static int fopt_percomp, fopt_only;

void
spirv_assume_float(const char *name, double lo, double hi, int lo_open,
    int hi_open)
{
	float l = (float)lo, h = (float)hi;

	if (npend_fas >= 8)
		return;
	/* "c in [0, 1]": 0 includes -0 (they compare equal) */
	if (l == 0 && !lo_open)
		l = -0.0f;
	if (lo_open)
		l = nextafterf(l == 0 ? 0.0f : l, INFINITY);
	if (hi_open)
		h = nextafterf(h, -INFINITY);
	snprintf(pend_fas[npend_fas].name, sizeof pend_fas[0].name, "%s",
	    name[0] == '%' ? name + 1 : name);
	pend_fas[npend_fas].lo = l;
	pend_fas[npend_fas++].hi = h;
}

void
spirv_fn_opts(int mode)
{
	fnl_mode = mode;
}

void
spirv_float_opts(int mode, double ulp, int approx, int percomp, int only)
{
	fopt.mode = mode;
	fopt.ulp = ulp;
	fopt.approx = approx;
	fopt_percomp = percomp;
	fopt_only = only;
}

/* float simplify's type: binary32, scalar or vector (its reference and
 * its precision table are binary32's; f64 is spv_dword.inc's) */
static int
is_float_t(int t)
{
	return t >= 0 && V.id[t].kind == K_TYPE && ((T(t)->k == S_FLOAT &&
	    T(t)->w == 32) || (T(t)->k == S_VEC && T(T(t)->e)->k == S_FLOAT &&
	    T(T(t)->e)->w == 32));
}

/* the input name of a load (its variable or buffer member), else NULL */
static const char *
load_name(int id, char *buf, size_t n)
{
	int k = SP.def[id], ptr;
	const char *mn, *nm;

	if (k < 0 || strcmp(V.in[k].op, "OpLoad") != 0 || V.in[k].nop < 2)
		return NULL;
	ptr = idx(V.in[k].opnd[1]);
	mn = member_name(ptr);
	if (mn != NULL) {	/* "decode_c": the parameter c */
		const char *u = strrchr(mn, '_');
		snprintf(buf, n, "%s", u && u[1] ? u + 1 : mn);
		return buf;
	}
	nm = V.id[ptr].name;
	snprintf(buf, n, "%s", nm[0] == '%' ? nm + 1 : nm);
	return buf;
}

/* an assumption's name only loads floats (the float part reports it) */
static int
as_float_only(const char *name)
{
	int k, f = 0, n = 0;
	char nb[48];

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		const char *nm;
		if (strcmp(I->op, "OpLoad") != 0 || I->res < 0)
			continue;
		nm = load_name(I->res, nb, sizeof nb);
		if (nm == NULL || !(name_is(nm, name) || !strcmp(nm, name)))
			continue;
		if (is_float_t(idx(I->opnd[0])))
			f = 1;
		else
			n = 1;
	}
	return f && !n;
}

static int
fx_input(struct fxg *g, int sid, int lane, int op, const char *name,
    float lo, float hi, int nan)
{
	int i;

	for (i = 0; i < g->nin; i++)
		if (g->in[i].sid == sid && g->in[i].lane == lane)
			return i;
	if (g->nin >= FX_MAXIN)
		return -1;
	memset(&g->in[i], 0, sizeof g->in[i]);
	snprintf(g->in[i].name, sizeof g->in[i].name, "%s", name);
	g->in[i].sid = sid;
	g->in[i].lane = lane;
	g->in[i].lo = lo;
	g->in[i].hi = hi;
	g->in[i].nan = nan;
	g->in[i].cvt = op == FX_CVT;
	if (op == FX_IN) {
		int j;
		for (j = 0; j < npend_fas; j++)
			if (name_is(name, pend_fas[j].name) ||
			    !strcmp(name, pend_fas[j].name)) {
				g->in[i].lo = pend_fas[j].lo;
				g->in[i].hi = pend_fas[j].hi;
				g->in[i].nan = 0;
				g->in[i].assumed = 1;
			}
	}
	g->nin++;
	return i;
}

static int fxmemo_id[FX_MAXN], fxmemo_lane[FX_MAXN], fxmemo_node[FX_MAXN];
static int nfxmemo;

static int
fx_new(struct fxg *g, int op, int a, int b, int c, int ins, int sid)
{
	struct fxn *n;

	if (g->nn >= FX_MAXN)
		return -1;
	n = &g->n[g->nn];
	memset(n, 0, sizeof *n);
	n->op = op;
	n->a = a;
	n->b = b;
	n->c = c;
	n->d = -1;
	n->ins = ins;
	n->sid = sid;
	n->uses = sid >= 0 && SP.uses ? SP.uses[sid] : 0;
	return g->nn++;
}

static int fx_build(struct fxg *g, int id, int lane, int depth);

static int
fx_leaf(struct fxg *g, int id, int lane)
{
	char nb[48];
	const char *nm = load_name(id, nb, sizeof nb);
	int k = SP.def[id], in, n;

	if (k >= 0 && (!strcmp(V.in[k].op, "OpConvertUToF") ||
	    !strcmp(V.in[k].op, "OpConvertSToF"))) {
		int u = V.in[k].op[9] == 'U';
		char cn[48];
		snprintf(cn, sizeof cn, "%s", V.id[idx(V.in[k].opnd[1])].name);
		in = fx_input(g, id, lane, FX_CVT, cn,
		    u ? 0.0f : -2147483648.0f, u ? 4294967296.0f : 2147483648.0f,
		    0);
		if (in < 0)
			return -1;
		n = fx_new(g, FX_CVT, -1, -1, -1, k, id);
	} else {
		if (nm == NULL) {
			snprintf(nb, sizeof nb, "%s", V.id[id].name[0] == '%' ?
			    V.id[id].name + 1 : V.id[id].name);
			nm = nb;
		}
		in = fx_input(g, id, lane, nm ? FX_IN : FX_OPQ, nm, -INFINITY,
		    INFINITY, 1);
		if (in < 0)
			return -1;
		n = fx_new(g, FX_IN, -1, -1, -1, -1, id);
	}
	if (n >= 0)
		g->n[n].in = in;
	return n;
}

static const struct {
	const char *ext;
	int op, na;
} fx_ext[] = {
	{"Pow", FX_POW, 2}, {"Exp2", FX_EXP2, 1}, {"Log2", FX_LOG2, 1},
	{"Sqrt", FX_SQRT, 1}, {"InverseSqrt", FX_ISQRT, 1},
	{"Step", FX_STEP, 2}, {"FMix", FX_MIX, 3}, {"Fma", FX_FMA, 3},
	{"FMin", FX_MIN, 2}, {"FMax", FX_MAX, 2}, {"FAbs", FX_ABS, 1},
	{NULL, 0, 0}
};

/* an operand's lane: a scalar operand of a vector op is the same in
 * every lane */
static int
fx_opnd(struct fxg *g, const char *o, int lane, int depth)
{
	int id = idx(o), t = V.id[id].type;

	if (t >= 0 && V.id[t].kind == K_TYPE && T(t)->k != S_VEC)
		lane = 0;
	return fx_build(g, id, lane, depth + 1);
}

static int
fx_build(struct fxg *g, int id, int lane, int depth)
{
	int k = SP.def[id], i, r = -1, op = -1, base = 1, na = 0;
	const struct sins *I;

	for (i = 0; i < nfxmemo; i++)
		if (fxmemo_id[i] == id && fxmemo_lane[i] == lane)
			return fxmemo_node[i];
	if (depth > 24)
		return -1;
	if (V.id[id].kind == K_CONST && !V.id[id].spec) {
		int t = V.id[id].type, st;
		if (!is_float_t(t))
			return -1;
		st = T(t)->k == S_VEC ? T(t)->e : t;
		r = fx_new(g, FX_K, -1, -1, -1, -1, id);
		if (r >= 0)
			g->n[r].k = (float)getf(st, V.id[id].c[T(t)->k == S_VEC &&
			    lane < V.id[id].nc ? lane : 0]);
		goto done;
	}
	if (k < 0)
		return fx_leaf(g, id, lane);
	I = &V.in[k];
	if (!strcmp(I->op, "OpCopyObject") && I->nop >= 2) {
		r = fx_opnd(g, I->opnd[1], lane, depth);
		goto done;
	}
	if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 3) {
		r = fx_build(g, idx(I->opnd[1]), atoi(I->opnd[2]), depth + 1);
		goto done;
	}
	if (!strcmp(I->op, "OpCompositeConstruct") && is_float_t(idx(I->opnd[0]))) {
		int at = 0, j;
		for (j = 1; j < I->nop; j++) {
			int cid = idx(I->opnd[j]), n = vcomp(cid);
			if (lane < at + n) {
				r = n > 1 ? fx_build(g, cid, lane - at, depth + 1) :
				    fx_build(g, cid, 0, depth + 1);
				break;
			}
			at += n;
		}
		goto done;
	}
	if (!strcmp(I->op, "OpVectorShuffle") && I->nop >= 4) {
		int a = idx(I->opnd[1]), b = idx(I->opnd[2]), na2 = vcomp(a), s;
		if (3 + lane >= I->nop)
			return -1;
		s = atoi(I->opnd[3 + lane]);
		r = s < na2 ? fx_build(g, a, s, depth + 1) : fx_build(g, b, s - na2,
		    depth + 1);
		goto done;
	}
	if (!is_float_t(idx(I->opnd[0])))
		return fx_leaf(g, id, lane);
	if (!strcmp(I->op, "OpFAdd"))
		op = FX_ADD, na = 2;
	else if (!strcmp(I->op, "OpFSub"))
		op = FX_SUB, na = 2;
	else if (!strcmp(I->op, "OpFMul") || !strcmp(I->op, "OpVectorTimesScalar"))
		op = FX_MUL, na = 2;
	else if (!strcmp(I->op, "OpFDiv"))
		op = FX_DIV, na = 2;
	else if (!strcmp(I->op, "OpFNegate"))
		op = FX_NEG, na = 1;
	else if (!strcmp(I->op, "OpExtInst") && I->nop >= 4 &&
	    idx(I->opnd[1]) == V.glsl) {
		for (i = 0; fx_ext[i].ext; i++)
			if (!strcmp(I->opnd[2], fx_ext[i].ext))
				op = fx_ext[i].op, na = fx_ext[i].na;
		base = 3;
	}
	if (op < 0 || I->nop < base + na)
		return fx_leaf(g, id, lane);
	{
		int o[3] = {-1, -1, -1};
		for (i = 0; i < na; i++)
			if ((o[i] = fx_opnd(g, I->opnd[base + i], lane, depth)) < 0)
				return -1;
		r = fx_new(g, op, o[0], o[1], o[2], k, id);
	}
done:
	if (r >= 0 && nfxmemo < FX_MAXN) {
		fxmemo_id[nfxmemo] = id;
		fxmemo_lane[nfxmemo] = lane;
		fxmemo_node[nfxmemo++] = r;
	}
	return r;
}

static int
fx_same(const struct fxg *a, const struct fxg *b)
{
	int i;

	if (a->nn != b->nn || a->nin != b->nin)
		return 0;
	for (i = 0; i < a->nn; i++) {
		const struct fxn *x = &a->n[i], *y = &b->n[i];
		uint32_t kx, ky;
		memcpy(&kx, &x->k, 4);
		memcpy(&ky, &y->k, 4);
		if (x->op != y->op || x->a != y->a || x->b != y->b || x->c != y->c ||
		    x->in != y->in || x->ins != y->ins || (x->op == FX_K && kx != ky))
			return 0;
	}
	for (i = 0; i < a->nin; i++)
		if (a->in[i].sid != b->in[i].sid)
			return 0;
	return 1;
}

/* per component cost of a SPIR-V instruction, as lowered (0: not float) */
static double
fprice_ins(const char *op, const char *ext, int rt)
{
	const float *c = model_active()->c;
	int i, n = rt >= 0 && V.id[rt].kind == K_TYPE ? ncomp(rt) : 1;
	double p = -1;

	/* a bool vector declared here: "%ucas_v3bool" */
	if (rt >= 0 && V.id[rt].kind != K_TYPE &&
	    !strncmp(V.id[rt].name, "%ucas_v", 7))
		n = atoi(V.id[rt].name + 7);
	/* a float of another width is not this pass's to price */
	if (rt >= 0 && V.id[rt].kind == K_TYPE && (T(rt)->k == S_FLOAT ||
	    (T(rt)->k == S_VEC && T(T(rt)->e)->k == S_FLOAT)) && !is_float_t(rt))
		return 0;

	if (!strcmp(op, "OpFAdd") || !strcmp(op, "OpFSub") ||
	    !strcmp(op, "OpFNegate"))
		p = c[C_FADD];
	else if (!strcmp(op, "OpFMul") || !strcmp(op, "OpVectorTimesScalar"))
		p = c[C_FMUL];
	else if (!strcmp(op, "OpFDiv"))
		p = c[C_FDIV];
	else if (!strncmp(op, "OpFOrd", 6) || !strncmp(op, "OpFUnord", 8))
		p = c[C_CMP];
	else if (!strcmp(op, "OpSelect") && is_float_t(rt))
		p = c[C_SELECT];
	else if (!strcmp(op, "OpExtInst") && ext) {
		for (i = 0; fx_ext[i].ext; i++)
			if (!strcmp(ext, fx_ext[i].ext))
				p = fx_price(fx_ext[i].op);
		if (p < 0 && is_float_t(rt))
			p = !strcmp(ext, "Exp") || !strcmp(ext, "Log") ?
			    c[C_FTRANS] : c[C_MINMAX];
	}
	return p < 0 ? 0 : p * n;
}

/* the float pure instructions dead code removal may take */
static int
fpure(const struct sins *I)
{
	static const char *const ops[] = {"OpFAdd", "OpFSub", "OpFMul", "OpFDiv",
		"OpFNegate", "OpVectorTimesScalar", "OpCompositeExtract",
		"OpCompositeConstruct", "OpVectorShuffle", "OpCopyObject", NULL};

	if (I->res < 0)
		return 0;
	if (!strncmp(I->op, "OpFOrd", 6) || !strncmp(I->op, "OpFUnord", 8))
		return 1;
	if (!strcmp(I->op, "OpExtInst"))
		return I->nop >= 3 && idx(I->opnd[1]) == V.glsl;
	if (!strcmp(I->op, "OpSelect"))
		return 1;
	return in_list(I->op, ops);
}

/* a constant of the float type rt (scalar or vector) with these lane
 * values: its declaration added to decl, its name returned */
static const char *
fconst(struct sbuf *decl, int rt, const float *v, int n)
{
	static char names[64][96];
	static int at;
	int st = T(rt)->k == S_VEC ? T(rt)->e : rt, i;
	char *nm = names[at++ & 63], sn[4][48];

	for (i = 0; i < n; i++) {
		uint32_t b;
		char d[40];
		memcpy(&b, &v[i], 4);
		snprintf(sn[i], sizeof sn[i], "%%ucas_%.10s_%08x", V.id[st].tok + 1, b);
		if (decl->p == NULL || !strstr(decl->p, sn[i])) {
			snprintf(d, sizeof d, "%.9g", (double)v[i]);
			sb_printf(decl, "%s = OpConstant %s %s\n", sn[i], V.id[st].tok,
			    d);
		}
	}
	if (n == 1) {
		snprintf(nm, 96, "%s", sn[0]);
		return nm;
	}
	snprintf(nm, 96, "%%ucas_k%d", n);
	for (i = 0; i < n; i++)
		snprintf(nm + strlen(nm), 96 - strlen(nm), "_%s", sn[i] + strlen(sn[i])
		    - 8);
	if (!strstr(decl->p, nm) || strstr(decl->p, nm)[strlen(nm)] != ' ') {
		sb_printf(decl, "%s = OpConstantComposite %s", nm, V.id[rt].tok);
		for (i = 0; i < n; i++)
			sb_printf(decl, " %s", sn[i]);
		sb_putc(decl, '\n');
	}
	return nm;
}

/* a bool type with n components: an existing one, or declared */
static const char *
bool_type(struct sbuf *decl, int n)
{
	static char nm[48];
	int i, b = -1;

	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_TYPE && V.id[i].t.k == S_BOOL) {
			b = i;
			break;
		}
	if (b < 0) {
		if (decl->p == NULL || !strstr(decl->p, "%ucas_bool = "))
			sb_puts(decl, "%ucas_bool = OpTypeBool\n");
	}
	if (n == 1) {
		snprintf(nm, sizeof nm, "%s", b >= 0 ? V.id[b].tok : "%ucas_bool");
		return nm;
	}
	if (b >= 0)
		for (i = 0; i < V.nid; i++)
			if (V.id[i].kind == K_TYPE && V.id[i].t.k == S_VEC &&
			    V.id[i].t.e == b && V.id[i].t.n == n) {
				snprintf(nm, sizeof nm, "%s", V.id[i].tok);
				return nm;
			}
	snprintf(nm, sizeof nm, "%%ucas_v%dbool", n);
	if (decl->p == NULL || !strstr(decl->p, nm))
		sb_printf(decl, "%s = OpTypeVector %s %d\n", nm, b >= 0 ?
		    V.id[b].tok : "%ucas_bool", n);
	return nm;
}

struct fplan {
	int k;			/* the instruction */
	int kind, deg;
	struct fxrw *rw[4];	/* one per lane group */
	int ng;
	double gain;		/* per instruction (lanes counted) */
	int applied;
	int line;
};

static void
pr_lines(struct sbuf *o, const char *text, const char *ind)
{
	const char *p = text ? text : "";

	while (*p) {
		const char *e = strchr(p, '\n');
		size_t n = e ? (size_t)(e - p) : strlen(p);
		sb_printf(o, "%s%.*s\n", ind, (int)n, p);
		p += n + (e != NULL);
	}
}

/* the lanes of a lane group: "xy" */
static void
glanes(char *b, size_t n, int gj, const int *grp, const int *lane_of, int ngr)
{
	int j, at = 0;

	for (j = 0; j < ngr && (size_t)at + 1 < n; j++)
		if (grp[j] == gj)
			b[at++] = "xyzw"[lane_of[j] & 3];
	b[at] = 0;
}

static const char *
kind_text(int kind)
{
	switch (kind) {
	case FXR_SELECT: return "mix by a step as a select";
	case FXR_RECIP: return "divide by a constant as a multiply";
	case FXR_FMA: return "add and multiply as one fma";
	case FXR_FMA2: return "multiply-add as one fma";
	case FXR_POWEXP: return "pow as exp2(y * log2 x)";
	case FXR_SQRT: return "sqrt as x * inversesqrt(x)";
	case FXR_POLY: return "pow as a polynomial";
	}
	return "?";
}

/* report.h's short name of proof_kind */
static const char *
proof_key(const struct fxrw *r)
{
	if (r->exact && r->kind == FXR_SELECT)
		return "bit_exact_where_defined";
	if (r->exact)
		return "bit_exact";
	if (r->vulkan && r->approx == 0)
		return "vulkan_precision";
	return "approximation";
}

/* one lane group of a float rewrite as data */
static void
rep_lane(const struct fxrw *rr, const char *lanes)
{
	if (!rep_on)
		return;
	rep_obj(NULL);
	if (lanes)
		rep_str("lanes", lanes);
	rep_bool("exact", rr->exact);
	rep_str("proof", proof_key(rr));
	if (rr->ulp > 0)
		rep_num("ulp_bound", rr->ulp);
	if (rr->kind == FXR_POLY)
		rep_num("rel_error", rr->rel);
	if (!rr->exact && rr->exc[0])
		rep_str("not_exact_at", rr->exc);
	if (rr->wit[0])
		rep_str("witness", rr->wit);
	if (rr->fail[0])
		rep_str("fail", rr->fail);
	rep_num("cost", rr->cost);
	rep_num("gain", rr->gain);
	rep_lines("derivation", rr->proof.p);
	rep_end();
}

static const char *
proof_kind(const struct fxrw *r)
{
	if (r->exact && r->kind == FXR_SELECT)
		return "bit-exact where the original is defined (derived)";
	if (r->exact)
		return "bit-exact (derived)";
	if (r->vulkan && r->approx == 0)
		return "within the original's Vulkan precision (derived), not "
		    "bit-exact";
	return "an approximation (derived bound)";
}

/*
 * Float simplify of a complete module's text: the report on stdout, the
 * rewritten module returned (NULL when nothing changed).
 */
static char *
float_pass(const char *text, int *changed)
{
	struct fxg *G = xcalloc(64, sizeof *G);	/* lane graphs */
	struct fxrw *RW = xcalloc(64 * 48, sizeof *RW);
	int nrw_g[64], root_g[64], grp[64], lane_of[64], ngr = 0, ng = 0, k, i,
	    j;
	struct fplan *pl = xcalloc(256, sizeof *pl);
	int npl = 0, preserve = 0, nroots = 0, napplied = 0;
	int tmp = 0;	/* names of new ids: %ucas_fN */
	char **repl, *dead;
	struct sbuf decl = {0}, out = {0}, rep = {0}, lanes_txt = {0};
	double before = 0, after = 0;
	int nlanes_total = 0;
	clock_t t0 = clock();

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
	SP.uses = xcalloc((size_t)(V.nid + 64), sizeof *SP.uses);
	for (i = 0; i < V.nid + 64; i++)
		SP.def[i] = -1;
	{
		int inbody = 0;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (!strcmp(I->op, "OpFunction"))
				inbody = 1;
			if (!strcmp(I->op, "OpExecutionMode") && I->nop >= 2 &&
			    !strcmp(I->opnd[1], "SignedZeroInfNanPreserve"))
				preserve = 1;
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
					SP.uses[idx(I->opnd[i])]++;
		}
	}
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		const char *ext = !strcmp(I->op, "OpExtInst") && I->nop >= 3 ?
		    I->opnd[2] : NULL;
		if (I->res >= 0 && SP.def[I->res] == k && I->nop >= 1 &&
		    I->opnd[0][0] == '%')
			before += fprice_ins(I->op, ext, idx(I->opnd[0]));
	}
	/* the roots: stored and returned float values */
	for (k = 0; k < V.nin && ngr < 60; k++) {
		const struct sins *I = &V.in[k];
		int v = -1, n, l, r0;
		if (!strcmp(I->op, "OpStore") && I->nop >= 2)
			v = idx(I->opnd[1]);
		else if (!strcmp(I->op, "OpReturnValue") && I->nop >= 1)
			v = idx(I->opnd[0]);
		if (v < 0 || SP.def[v] < 0 || !is_float_t(V.id[v].type) ||
		    fprice_ins(V.in[SP.def[v]].op, !strcmp(V.in[SP.def[v]].op,
		    "OpExtInst") ? V.in[SP.def[v]].opnd[2] : NULL,
		    V.id[v].type) <= 0)
			continue;
		n = ncomp(V.id[v].type);
		nroots++;
		r0 = ngr;
		for (l = 0; l < n && ngr < 64; l++) {
			struct fxg *g = &G[ngr];
			int root;
			memset(g, 0, sizeof *g);
			nfxmemo = 0;
			g->preserve = preserve;
			root = fx_build(g, v, l, 0);
			if (root < 0)
				continue;
			{
				static char sl[200];
				src_line(ins_line(SP.def[v]), sl, sizeof sl);
				{
					char *c = strchr(sl, ':');
					g->srcline = c ? strdup(c + 1) : NULL;
				}
			}
			root_g[ngr] = root;
			lane_of[ngr] = l;
			/* the same scalar function as an earlier lane? */
			grp[ngr] = ngr;
			if (!fopt_percomp)
				for (j = r0; j < ngr; j++)
					if (grp[j] == j && fx_same(&G[j], g)) {
						grp[ngr] = j;
						break;
					}
			ngr++;
		}
		{
			int ngroups = 0, gl[4] = {0, 0, 0, 0};
			for (j = r0; j < ngr; j++)
				if (grp[j] == j)
					ngroups++;
			for (j = r0; j < ngr; j++)
				gl[j - r0 < 4 ? j - r0 : 3] = grp[j] - r0;
			nlanes_total += ngr - r0;
			if (ngr - r0 > 1) {
				char ln[200];
				src_line(ins_line(SP.def[v]), ln, sizeof ln);
				if (ngroups == 1)
					sb_printf(&lanes_txt, "  the vec%d %s: its lanes %s compute one "
					    "scalar function (splat constants): analysed once, "
					    "applied to every lane%s\n", ngr - r0, V.id[v].name,
					    ngr - r0 == 2 ? "x, y" : ngr - r0 == 3 ?
					    "x, y, z" : "x, y, z, w", ngr - r0 < 4 ? "; any "
					    "width is lanes of a vec4, so the same search covers "
					    "it" : "");
				else
					sb_printf(&lanes_txt, "  the vec%d %s: %d different scalar functions "
					    "over its lanes (%s): analysed per component; a "
					    "rewrite applies when every lane's analysis allows "
					    "it\n", ngr - r0, V.id[v].name, ngroups,
					    fopt_percomp ? "per component, as asked" :
					    "the constants differ per component");
			}
			(void)gl;
		}
	}
	if (ngr == 0) {
		free(G);
		free(RW);
		free(pl);
		free(SP.def);
		free(SP.uses);
		SP.def = NULL;
		SP.uses = NULL;
		for (k = 0; k < V.nin; k++) {
			for (i = 0; i < V.in[k].nop; i++)
				free(V.in[k].opnd[i]);
			free(V.in[k].text);
		}
		free(V.in);
		memset(&V, 0, sizeof V);
		npend_fas = 0;
		if (fopt_only)
			printf("float simplify: no float arithmetic reaches a store "
			    "or a return\n");
		return NULL;
	}
	/* the analyses, per lane group */
	for (j = 0; j < ngr; j++) {
		nrw_g[j] = 0;
		if (grp[j] != j)
			continue;
		nrw_g[j] = fx_rewrites(&G[j], root_g[j], &fopt, RW + j * 48, 48);
		ng++;
	}
	/* plans per instruction: every group of the root has the rewrite */
	for (j = 0; j < ngr; j++) {
		int q;
		if (grp[j] != j)
			continue;
		for (q = 0; q < nrw_g[j]; q++) {
			struct fxrw *r = &RW[j * 48 + q];
			int ins = G[j].n[r->node].ins, p;
			for (p = 0; p < npl; p++)
				if (pl[p].k == ins && pl[p].kind == r->kind &&
				    pl[p].deg == r->deg)
					break;
			if (p == npl) {
				if (npl >= 256)
					continue;
				memset(&pl[p], 0, sizeof pl[p]);
				pl[p].k = ins;
				pl[p].kind = r->kind;
				pl[p].deg = r->deg;
				pl[p].line = ins_line(ins);
				npl++;
			}
			if (pl[p].ng < 4)
				pl[p].rw[pl[p].ng++] = r;
		}
	}
	/* accepted: every lane group of the instruction allows it, and what
	 * it depends on is applied; one rewrite per instruction (the most
	 * gain) */
	{
		int pass;
		for (pass = 0; pass < 3; pass++)
			for (i = 0; i < npl; i++) {
				struct fplan *P = &pl[i];
				int ok = 1, q, groups = 0, rt;
				if (P->applied || P->k < 0)
					continue;
				rt = idx(V.in[P->k].opnd[0]);
				for (j = 0; j < ngr; j++)
					if (grp[j] == j)
						for (q = 0; q < G[j].nn; q++)
							if (G[j].n[q].ins == P->k) {
								groups++;
								break;
							}
				if (P->ng < groups)
					ok = 0;
				for (q = 0; q < P->ng && ok; q++) {
					struct fxrw *r = P->rw[q];
					if (!fx_accept(r, &fopt))
						ok = 0;
					if (r->dep >= 0) {
						/* the select it relies on, in the same group */
						struct fxrw *d = NULL;
						int gj = (int)((r - RW) / 48), p2, found = 0;
						d = &RW[gj * 48 + r->dep];
						for (p2 = 0; p2 < npl; p2++)
							if (pl[p2].applied && pl[p2].kind == d->kind &&
							    pl[p2].k == G[gj].n[d->node].ins)
								found = 1;
						if (!found)
							ok = 0;
					}
				}
				for (q = 0; q < npl && ok; q++)
					if (q != i && pl[q].applied && pl[q].k == P->k)
						ok = 0;
				if (!ok)
					continue;
				/* the best of the alternatives on this instruction */
				for (q = 0; q < npl; q++)
					if (q != i && pl[q].k == P->k && !pl[q].applied &&
					    pl[q].rw[0] && fx_accept(pl[q].rw[0], &fopt) &&
					    pl[q].rw[0]->gain > P->rw[0]->gain &&
					    pl[q].ng >= groups)
						ok = 0;
				if (!ok)
					continue;
				P->applied = 1;
				P->gain = P->rw[0]->gain * ncomp(rt);
				napplied++;
			}
	}
	/* the rewritten module */
	repl = xcalloc((size_t)V.nin, sizeof *repl);
	dead = xcalloc((size_t)V.nin, 1);
	for (i = 0; i < npl; i++) {
		struct fplan *P = &pl[i];
		const struct sins *I;
		const char *R, *RT, *GL;
		struct sbuf s = {0};
		int rt, n, q;
		float lv[FX_MAXK][4];
		if (!P->applied)
			continue;
		I = &V.in[P->k];
		R = V.id[I->res].tok;
		RT = I->opnd[0];
		rt = idx(RT);
		n = ncomp(rt);
		GL = V.glsl >= 0 ? V.id[V.glsl].tok : "%1";
		/* lane values of the new constants */
		for (q = 0; q < FX_MAXK; q++) {
			int l;
			for (l = 0; l < n && l < 4; l++) {
				/* the graph built for lane l that has the instruction:
				 * its group's rewrite */
				int gi = -1, jj, rr, z;
				for (jj = 0; jj < ngr && gi < 0; jj++) {
					if (lane_of[jj] != l)
						continue;
					for (z = 0; z < G[jj].nn; z++)
						if (G[jj].n[z].ins == P->k)
							gi = grp[jj];
				}
				for (rr = 0; rr < P->ng; rr++)
					if ((int)((P->rw[rr] - RW) / 48) == gi)
						break;
				if (rr == P->ng)
					rr = 0;
				lv[q][l] = P->rw[rr]->k[q];
			}
		}
		switch (P->kind) {
		case FXR_SELECT: {
			int t = idx(I->opnd[5]), sk = SP.def[t];
			const char *bt = bool_type(&decl, n);
			if (sk < 0)
				break;
			sb_printf(&s, "%%ucas_f%d = OpFUnordGreaterThanEqual %s %s %s\n",
			    ++tmp, bt, V.in[sk].opnd[4], V.in[sk].opnd[3]);
			sb_printf(&s, "%s = OpSelect %s %%ucas_f%d %s %s\n", R, RT, tmp,
			    I->opnd[4], I->opnd[3]);
			break;
		}
		case FXR_RECIP:
			sb_printf(&s, "%s = OpFMul %s %s %s\n", R, RT, I->opnd[1],
			    fconst(&decl, rt, lv[0], n));
			break;
		case FXR_FMA: {
			int a = idx(I->opnd[1]), b = idx(I->opnd[2]), ak, x;
			int s1 = SP.def[a] >= 0 && !strcmp(V.in[SP.def[a]].op, "OpFAdd") ?
			    a : b;
			const char *k1 = fconst(&decl, rt, lv[0], n);
			const char *k2 = fconst(&decl, rt, lv[1], n);
			ak = SP.def[s1];
			x = V.id[idx(V.in[ak].opnd[1])].kind == K_CONST ? 2 : 1;
			sb_printf(&s, "%s = OpExtInst %s %s Fma %s %s %s\n", R, RT, GL,
			    V.in[ak].opnd[x], k1, k2);
			break;
		}
		case FXR_FMA2: {
			int a = idx(I->opnd[1]), m, x, kk;
			int mi = SP.def[a] >= 0 && !strcmp(V.in[SP.def[a]].op, "OpFMul") ?
			    1 : 2;
			m = SP.def[idx(I->opnd[mi])];
			x = V.id[idx(V.in[m].opnd[1])].kind == K_CONST ? 2 : 1;
			kk = 3 - x;
			sb_printf(&s, "%s = OpExtInst %s %s Fma %s %s %s\n", R, RT, GL,
			    V.in[m].opnd[x], V.in[m].opnd[kk], I->opnd[3 - mi]);
			break;
		}
		case FXR_POWEXP:
			tmp++;
			sb_printf(&s, "%%ucas_f%dl = OpExtInst %s %s Log2 %s\n", tmp, RT,
			    GL, I->opnd[3]);
			sb_printf(&s, "%%ucas_f%dm = OpFMul %s %%ucas_f%dl %s\n", tmp, RT,
			    tmp, I->opnd[4]);
			sb_printf(&s, "%s = OpExtInst %s %s Exp2 %%ucas_f%dm\n", R, RT,
			    GL, tmp);
			break;
		case FXR_SQRT:
			tmp++;
			sb_printf(&s, "%%ucas_f%dz = OpExtInst %s %s InverseSqrt %s\n",
			    tmp, RT, GL, I->opnd[3]);
			sb_printf(&s, "%s = OpFMul %s %s %%ucas_f%dz\n", R, RT,
			    I->opnd[3], tmp);
			break;
		case FXR_POLY: {
			int d = P->deg, e;
			char prev[48];
			tmp++;
			sb_printf(&s, "%%ucas_f%dt = OpFSub %s %s %s\n", tmp, RT,
			    I->opnd[3], fconst(&decl, rt, lv[0], n));
			snprintf(prev, sizeof prev, "%s", fconst(&decl, rt, lv[1], n));
			for (e = 1; e <= d; e++) {
				const char *ce = fconst(&decl, rt, lv[1 + e], n);
				if (e == d)
					sb_printf(&s, "%s = OpExtInst %s %s Fma %s %%ucas_f%dt "
					    "%s\n", R, RT, GL, prev, tmp, ce);
				else {
					sb_printf(&s, "%%ucas_f%dh%d = OpExtInst %s %s Fma %s "
					    "%%ucas_f%dt %s\n", tmp, e, RT, GL, prev, tmp, ce);
					snprintf(prev, sizeof prev, "%%ucas_f%dh%d", tmp, e);
				}
			}
			break;
		}
		}
		repl[P->k] = s.p;
	}
	/* dead float instructions: no uses left */
	{
		int *uses = xcalloc((size_t)V.nid + 64, sizeof *uses), ch = 1;
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			if (!strncmp(I->op, "OpName", 6) || !strncmp(I->op,
			    "OpMemberName", 12) || !strncmp(I->op, "OpDecorate", 10) ||
			    !strncmp(I->op, "OpMemberDecorate", 16))
				continue;
			if (repl[k]) {	/* the replacement's operands */
				const char *p = repl[k];
				while ((p = strchr(p, '%')) != NULL) {
					char nm[64];
					int ln = 0;
					while (p[ln] && !isspace((unsigned char)p[ln]) && ln < 63)
						ln++;
					memcpy(nm, p, (size_t)ln);
					nm[ln] = 0;
					/* ids after the '=' only */
					{
						const char *ls = p;
						while (ls > repl[k] && ls[-1] != '\n')
							ls--;
						if (strchr(ls, '=') && strchr(ls, '=') < p &&
						    nm[1] != 'u')
							uses[idx(nm)]++;
					}
					p += ln;
				}
				continue;
			}
			for (i = 0; i < I->nop; i++)
				if (I->opnd[i][0] == '%')
					uses[idx(I->opnd[i])]++;
		}
		while (ch) {
			ch = 0;
			for (k = 0; k < V.nin; k++) {
				const struct sins *I = &V.in[k];
				if (dead[k] || repl[k] || I->res < 0 || SP.def[I->res] != k ||
				    uses[I->res] || !fpure(I))
					continue;
				dead[k] = 1;
				ch = 1;
				for (i = 0; i < I->nop; i++)
					if (I->opnd[i][0] == '%')
						uses[idx(I->opnd[i])]--;
			}
		}
		free(uses);
	}
	/* the text, and the cost after */
	{
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
			if (dead[k])
				continue;
			if ((!strcmp(I->op, "OpName") || !strncmp(I->op, "OpDecorate",
			    10)) && I->nop >= 1 && I->opnd[0][0] == '%') {
				int t = idx(I->opnd[0]);
				if (SP.def[t] >= 0 && dead[SP.def[t]])
					continue;
			}
			if (repl[k]) {
				const char *p = repl[k];
				sb_puts(&out, p);
				while (*p) {	/* each new line's price */
					char op[40] = "", ext[24] = "", rtn[64] = "";
					const char *e = strchr(p, '\n'), *q = strchr(p, '=');
					if (q && (!e || q < e))
						sscanf(q + 1, " %39s %63s", op, rtn);
					if (!strcmp(op, "OpExtInst"))
						sscanf(q + 1, " %*s %*s %*s %23s", ext);
					if (op[0])
						after += fprice_ins(op, ext[0] ? ext : NULL,
						    idx(rtn));
					p = e ? e + 1 : p + strlen(p);
				}
				continue;
			}
			sb_printf(&out, "%s\n", I->text);
			if (I->res >= 0 && SP.def[I->res] == k && I->nop >= 1 &&
			    I->opnd[0][0] == '%')
				after += fprice_ins(I->op, !strcmp(I->op, "OpExtInst") &&
				    I->nop >= 3 ? I->opnd[2] : NULL, idx(I->opnd[0]));
		}
	}
	/* the report */
	{
		int nd = 0, nap = 0, nrej = 0, napx = 0, q;
		int ord[256], no = 0;
		for (i = 0; i < npl; i++) {
			if (pl[i].kind == FXR_DIVNOTE)
				nd++;
			else if (pl[i].applied)
				ord[no++] = i;
			else if (pl[i].kind == FXR_POLY)
				napx++;
			else
				nrej++;
		}
		for (i = 1; i < no; i++) {	/* largest savings first */
			int x = ord[i];
			for (j = i; j > 0 && pl[ord[j - 1]].gain < pl[x].gain; j--)
				ord[j] = ord[j - 1];
			ord[j] = x;
		}
		nap = no;
		sb_printf(&rep, "float simplify: %d rewrite%s of float instructions "
		    "applied, each with a derivation (no sampling)", nap,
		    nap == 1 ? "" : "s");
		if (nd)
			sb_printf(&rep, "; %d done already by spirv-opt -O, checked "
			    "the same way", nd);
		sb_puts(&rep, "\n");
		if (lanes_txt.p)
			sb_puts(&rep, lanes_txt.p);
		for (j = 0; j < ngr; j++) {
			int z;
			if (grp[j] != j)
				continue;
			for (z = 0; z < G[j].nin; z++)
				if (G[j].in[z].assumed && G[j].in[z].lane == 0) {
					char a[40], b[40];
					sb_printf(&rep, "  assuming %s in [%s, %s] (each "
					    "component; no NaN)\n", G[j].in[z].name,
					    fx_str(G[j].in[z].lo, a, sizeof a),
					    fx_str(G[j].in[z].hi, b, sizeof b));
				}
			break;
		}
		sb_printf(&rep, "  reference: IEEE binary32, round to nearest even;"
		    " pow, exp2, log2 correctly rounded (long double, then f32); NaN "
		    "payloads ignored, -0 and +0 distinct; GLSL's undefined results "
		    "(pow of x < 0, log2 of x <= 0 ...) may become anything\n");
		sb_printf(&rep, "  float cost under %s (each component counted; step = "
		    "compare + select, mix = subtract + fma): %g -> %g",
		    sym_str(model_active()->name), before, after);
		if (before > 0)
			sb_printf(&rep, " (%+.1f%%)", 100 * (after - before) / before);
		sb_puts(&rep, "\n");
		rep_obj("float");
		rep_num("cost_before", before);
		rep_num("cost_after", after);
		rep_str("mode", fopt.mode == 1 ? "exact" : fopt.mode == 2 ? "ulp" :
		    "vulkan_precision");
		if (fopt.mode == 2)
			rep_num("ulp", fopt.ulp);
		rep_bool("approx", fopt.approx);
		rep_bool("percomp", fopt_percomp);
		rep_list("rewrites");
		for (q = 0; q < no; q++) {
			struct fplan *P = &pl[ord[q]];
			struct fxrw *r = P->rw[0];
			int g2;
			for (g2 = 0; g2 < P->ng; g2++)
				if (!P->rw[g2]->exact)
					r = P->rw[g2];
			rep_obj(NULL);
			rep_str("instruction", V.in[P->k].text);
			rep_lines("replacement", repl[P->k]);
			rep_str("kind", kind_text(P->kind));
			rep_num("cost_delta", -P->gain);
			rep_str("proof", proof_key(r));
			rep_str("proof_text", proof_kind(r));
			rep_int("line", P->line);
			rep_list("lanes");
			for (g2 = 0; g2 < P->ng; g2++) {
				char lb[16] = "";
				if (P->ng > 1)
					glanes(lb, sizeof lb, (int)((P->rw[g2] - RW) / 48), grp,
					    lane_of, ngr);
				rep_lane(P->rw[g2], P->ng > 1 ? lb : NULL);
			}
			rep_end();
			rep_end();
		}
		rep_end();
		rep_list("approximations");
		for (i = 0; i < npl; i++) {
			int c;
			if (pl[i].kind != FXR_POLY || pl[i].applied)
				continue;
			rep_obj(NULL);
			rep_str("instruction", V.in[pl[i].k].text);
			rep_str("why", pl[i].rw[0]->why);
			rep_int("degree", pl[i].rw[0]->deg);
			rep_list("lanes");
			for (c = 0; c < pl[i].ng; c++)
				rep_lane(pl[i].rw[c], NULL);
			rep_end();
			rep_end();
		}
		rep_end();
		rep_list("not_applied");
		for (i = 0; i < npl; i++) {
			int c;
			if (pl[i].applied || pl[i].kind == FXR_DIVNOTE ||
			    pl[i].kind == FXR_POLY)
				continue;
			rep_obj(NULL);
			rep_str("instruction", V.in[pl[i].k].text);
			rep_str("kind", kind_text(pl[i].kind));
			rep_list("lanes");
			for (c = 0; c < pl[i].ng; c++)
				rep_lane(pl[i].rw[c], NULL);
			rep_end();
			rep_end();
		}
		rep_end();
		rep_end();
		if (fopt.mode == 1)
			sb_puts(&rep, "  mode: bit-exact only\n");
		else if (fopt.mode == 2)
			sb_printf(&rep, "  mode: within %g ulp of the reference%s\n",
			    fopt.ulp, fopt.approx ? " (approximations too)" : "");
		for (q = 0; q < no; q++) {
			struct fplan *P = &pl[ord[q]];
			char sl[200];
			struct fxrw *r = P->rw[0];
			int g2;
			src_line(P->line, sl, sizeof sl);
			if (q == 0)
				sb_printf(&rep, "  rewrites, the largest savings first:\n");
			sb_printf(&rep, "    %s\n", V.in[P->k].text);
			{
				const char *p = repl[P->k];
				int first = 1;
				while (p && *p) {
					const char *e = strchr(p, '\n');
					size_t ln = e ? (size_t)(e - p) : strlen(p);
					sb_printf(&rep, "      %s %.*s\n", first ? "->" : "  ",
					    (int)ln, p);
					first = 0;
					p += ln + (e != NULL);
				}
			}
			sb_printf(&rep, "      (%s; -%g)\n", kind_text(P->kind), P->gain);
			for (g2 = 0; g2 < P->ng; g2++)
				if (!P->rw[g2]->exact)
					r = P->rw[g2];
			sb_printf(&rep, "      proof: %s\n", proof_kind(r));
			for (g2 = 0; g2 < P->ng; g2++) {
				struct fxrw *rr = P->rw[g2];
				if (P->ng > 1) {
					char lb[16];
					glanes(lb, sizeof lb, (int)((rr - RW) / 48), grp, lane_of,
					    ngr);
					sb_printf(&rep, "      lane%s %s:%s\n", strlen(lb) > 1 ?
					    "s" : "", lb, rr->exact ? " bit-exact" : "");
				}
				pr_lines(&rep, rr->proof.p, P->ng > 1 ? "        " :
				    "        ");
				if (!rr->exact && rr->exc[0])
					sb_printf(&rep, "        not bit-exact at %s\n",
					    rr->exc);
				else if (!rr->exact && rr->wit[0])
					sb_printf(&rep, "        not bit-exact: %s\n", rr->wit);
				if (rr->ulp > 0)
					sb_printf(&rep, "        bound against the reference: "
					    "%g ulp\n", rr->ulp);
			}
			if (sl[0])
				sb_printf(&rep, "  %s", sl);
		}
		if (no == 0)
			sb_printf(&rep, "  nothing applied: no cheaper float form could "
			    "be proved%s\n", fopt.mode == 1 ? " bit-exact" : "");
		if (nd) {
			sb_printf(&rep, "  done already by spirv-opt -O (the GLSL "
			    "divides, the module multiplies):\n");
			for (i = 0; i < npl; i++) {
				struct fxrw *r = pl[i].rw[0];
				char sl[200];
				int c;
				if (pl[i].kind != FXR_DIVNOTE)
					continue;
				sb_printf(&rep, "    %s   (%s)\n", V.in[pl[i].k].text,
				    r->why);
				for (c = 0; c < pl[i].ng; c++) {
					char lb[16];
					r = pl[i].rw[c];
					glanes(lb, sizeof lb, (int)((r - RW) / 48), grp, lane_of,
					    ngr);
					if (pl[i].ng > 1)
						sb_printf(&rep, "      lane%s %s: %s\n", strlen(lb) > 1 ?
						    "s" : "", lb, r->why);
					pr_lines(&rep, r->proof.p, "        ");
					if (r->wit[0])
						sb_printf(&rep, "        not bit-exact: %s\n", r->wit);
				}
				src_line(pl[i].line, sl, sizeof sl);
				if (sl[0])
					sb_printf(&rep, "  %s", sl);
			}
		}
		if (napx) {
			sb_printf(&rep, "  approximations (cheaper, %s):\n",
			    fopt.approx ? "applied only within `ulp N`" : "not applied: "
			    "say `approx ulp N` to allow one");
			for (i = 0; i < npl; i++) {
				struct fxrw *r = pl[i].rw[0];
				int c, any = 0;
				if (pl[i].kind != FXR_POLY || pl[i].applied)
					continue;
				for (c = 0; c < pl[i].ng; c++)
					any |= pl[i].rw[c]->approx;
				sb_printf(&rep, "    %s, cost %g (-%g): relative error < %.3g (%g ulp), %s\n",
				    r->why, r->cost, r->gain, r->rel, r->ulp, any ? "beyond the "
				    "original's precision" : "within the original's precision");
				if (i == npl - 1 || pl[i + 1].kind != FXR_POLY ||
				    pl[i + 1].k != pl[i].k) {
					pr_lines(&rep, r->proof.p, "        ");
					{
						int e;
						sb_printf(&rep, "        P(t) = ");
						for (e = 0; e < r->deg; e++)
							sb_printf(&rep, "fma(");
						sb_printf(&rep, "%.9g", (double)r->k[1]);
						for (e = 1; e <= r->deg; e++)
							sb_printf(&rep, ", t, %.9g)", (double)r->k[1 + e]);
						sb_printf(&rep, ", t = q - %.9g\n", (double)r->k[0]);
					}
				}
			}
		}
		if (nrej) {
			sb_printf(&rep, "  not applied:\n");
			for (i = 0; i < npl; i++) {
				struct fxrw *r = NULL;
				int c;
				if (pl[i].applied || pl[i].kind == FXR_DIVNOTE ||
				    pl[i].kind == FXR_POLY)
					continue;
				for (c = 0; c < pl[i].ng; c++)
					if (!fx_accept(pl[i].rw[c], &fopt))
						r = pl[i].rw[c];
				if (r == NULL)
					r = pl[i].rw[0];
				sb_printf(&rep, "    %s   (%s)\n", V.in[pl[i].k].text,
				    kind_text(pl[i].kind));
				if (pl[i].ng > 1) {
					char lb[16];
					glanes(lb, sizeof lb, (int)((r - RW) / 48), grp, lane_of,
					    ngr);
					sb_printf(&rep, "      lane%s %s:\n", strlen(lb) > 1 ? "s" :
					    "", lb);
				}
				if (r->gain <= 0)
					sb_printf(&rep, "        not cheaper under %s\n",
					    sym_str(model_active()->name));
				else if (!r->refine)
					sb_printf(&rep, "        not a refinement: %s\n",
					    r->fail);
				else if (r->fail[0])
					sb_printf(&rep, "        %s\n", r->fail);
				else if (fopt.mode == 1 && !r->exact)
					sb_printf(&rep, "        not bit-exact%s%s\n",
					    r->exc[0] || r->wit[0] ? ": " : "", r->exc[0] ?
					    r->exc : r->wit);
				else if (fopt.mode == 2)
					sb_printf(&rep, "        its bound (%g ulp) exceeds %g "
					    "ulp\n", r->ulp, fopt.ulp);
				else
					sb_printf(&rep, "        a lane or what it relies on "
					    "is not allowed\n");
				if (r->kind == FXR_FMA && r->wit[0] && r->fail[0] == 0 &&
				    fopt.mode != 1)
					sb_printf(&rep, "        %s\n", r->wit);
			}
		}
		/* the select still runs both sides; a branch would not */
		for (i = 0; i < npl; i++)
			if (pl[i].applied && pl[i].kind == FXR_SELECT) {
				const float *mc = model_active()->c;
				sb_printf(&rep, "  the select still computes both sides (no "
				    "divergence); a branch on the compare would skip the "
				    "unused side: under %s it costs a branch (%g) when the "
				    "lanes agree, and both sides plus %g when they diverge "
				    "(keep-up)\n", sym_str(model_active()->name),
				    mc[C_BRANCH], mc[C_DIVERGE]);
				break;
			}
		{
			double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
			if (show_time(el))
				sb_printf(&rep, "  derived in %.1f s\n", el);
		}
		fputs(rep.p, stdout);
	}
	*changed = napplied > 0;
	for (i = 0; i < ngr; i++)
		free((void *)G[i].srcline);
	for (j = 0; j < ngr; j++)
		if (grp[j] == j)
			fx_free(RW + j * 48, nrw_g[j]);
	for (k = 0; k < V.nin; k++)
		free(repl[k]);
	free(repl);
	free(dead);
	free(decl.p);
	free(rep.p);
	free(lanes_txt.p);
	free(G);
	free(RW);
	free(pl);
	free(SP.def);
	free(SP.uses);
	SP.def = NULL;
	SP.uses = NULL;
	for (k = 0; k < V.nin; k++) {
		for (i = 0; i < V.in[k].nop; i++)
			free(V.in[k].opnd[i]);
		free(V.in[k].text);
	}
	free(V.in);
	memset(&V, 0, sizeof V);
	npend_fas = 0;
	(void)nroots;
	(void)nlanes_total;
	(void)ng;
	if (!*changed) {
		free(out.p);
		return NULL;
	}
	return out.p;
}

#include "spv_dword.inc"	/* fp64 as f32 pairs, the driver's folds */

static char *orig_mod;	/* the module before the rewrites */

/*
 * The live integer instructions as a DAG for the model's fusions
 * (fuse.c): in order, with their operands that are not such
 * instructions (loads, constants, phis) as leaves; a result something
 * else uses (a store, a phi, a branch) is computed as it is.
 */
static double
spv_cover(struct fuse_tally *t)
{
	static const struct {
		const char *op;
		int o;
	} map[] = {{"OpIAdd", O_ADD}, {"OpISub", O_SUB}, {"OpIMul", O_MUL},
	    {"OpSNegate", O_NEG}, {"OpBitwiseAnd", O_AND},
	    {"OpBitwiseOr", O_OR}, {"OpBitwiseXor", O_XOR}, {"OpNot", O_NOT},
	    {"OpShiftLeftLogical", O_SHL}, {"OpShiftRightLogical", O_SHR},
	    {"OpShiftRightArithmetic", O_SHR}, {"OpUDiv", O_DIV},
	    {"OpSDiv", O_DIV}, {NULL, 0}};
	int nid_ = V.nid + 1024, *gi = xmalloc((size_t)nid_ * sizeof *gi);
	int nins = 0, nleaf = 0, k, i, j, q;
	struct fgraph G;
	double total;

	for (i = 0; i < nid_; i++)
		gi[i] = -1;
	for (k = 0; k < V.nin; k++)
		if (pure_ins(k))
			gi[V.in[k].res] = nins++;
	for (k = 0; k < V.nin; k++)
		if (pure_ins(k))
			for (i = first_arg(&V.in[k]); i < V.in[k].nop; i++) {
				int a = res_of(idx(V.in[k].opnd[i]));
				if (gi[a] < 0)
					gi[a] = nins + nleaf++;
			}
	fg_init(&G, nins + nleaf);
	for (i = 0; i < nid_; i++)
		if (gi[i] >= nins) {
			j = gi[i];
			G.op[j] = O_NONE;
			if (V.id[i].kind == K_CONST && !V.id[i].spec) {
				G.op[j] = O_CONST;
				G.isk[j] = 1;
				G.kv[j] = (int64_t)V.id[i].c[0];
			}
		}
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		double mul;
		int base;
		if (!pure_ins(k))
			continue;
		j = gi[I->res];
		G.id[j] = (uint32_t)k;
		G.op[j] = O_NONE;
		for (q = 0; map[q].op; q++)
			if (!strcmp(I->op, map[q].op))
				G.op[j] = (uint8_t)map[q].o;
		G.uns[j] = !strcmp(I->op, "OpUDiv");
		G.isbool[j] = int_width(idx(I->opnd[0])) == 1;
		G.own[j] = (float)ins_cost_k(k);
		G.cls[j] = cost_class(I->op, !strcmp(I->op, "OpExtInst") ?
		    I->opnd[2] : NULL, idx(I->opnd[0]), &mul);
		base = first_arg(I);
		for (i = base; i < I->nop && i - base < 3; i++) {
			int a = gi[res_of(idx(I->opnd[i]))];
			G.uses[a]++;
			if (G.op[j] != O_NONE)
				G.arg[j][i - base] = a;
			else
				G.pinned[a] = 1;
		}
	}
	/* used by anything else: a result */
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (SP.dead[k] || pure_ins(k) || !strncmp(I->op, "OpName", 6) ||
		    !strncmp(I->op, "OpMemberName", 12) ||
		    !strncmp(I->op, "OpDecorate", 10) ||
		    !strncmp(I->op, "OpMemberDecorate", 16))
			continue;
		for (i = 0; i < I->nop; i++)
			if (I->opnd[i][0] == '%') {
				int a = gi[res_of(idx(I->opnd[i]))];
				if (a >= 0 && a < nins) {
					G.pinned[a] = 1;
					G.uses[a]++;
				}
			}
	}
	total = fg_cover(model_active(), &G);
	if (t)
		fg_tally(&G, NULL, 1, t);
	fg_free(&G);
	free(gi);
	return total;
}

void
spirv_simplify(const char *src, double budget, int nofile)
{
	int k, i, nrw = 0, ndead = 0, inbody = 0, nlog = 0, nfwd = 0, ndst = 0;
	int ndotrw = 0;
	int ntriv = 0;
	double tgain = 0;
	char *range_report = NULL;
#define MAXLOG 4096
	struct rwlog {
		char from[160], to[160];
		const char *why;
		double gain;
		int line, algebra, trivial, dot;
	} *rl = xcalloc(MAXLOG, sizeof *rl);
	struct dotlog dotl[32];
	int ndotlog = 0;
	double before = 0, after = 0;
	struct fuse_tally ftally;
	struct sbuf out = {0}, list = {0}, dotrep = {0};
	clock_t t0 = clock();

	memset(&V, 0, sizeof V);
	symmap_reset();
	memset(&SP, 0, sizeof SP);
	dotcaps = 0;
	SP.force_id = -1;
	code_block_used = 1;
	V.glsl = -1;
	V.lanes = 1;
	V.rng = 1;
	cols = 1;
	/* this module's own OpSource text, not a previous one's */
	free(src_text);
	src_text = NULL;
	parse(src);
	if (fnl_mode == 1) {
		/* the function level alone (glslfn.c) */
		rep_begin("spirv_simplify");
		if (src_text != NULL)
			glslfn_pass(src_text, budget, 1);
		else
			printf("function level: the module carries no GLSL source "
			    "(OpSource): give the GLSL, or compile it with -g\n");
		for (k = 0; k < V.nin; k++) {
			for (i = 0; i < V.in[k].nop; i++)
				free(V.in[k].opnd[i]);
			free(V.in[k].text);
		}
		free(V.in);
		memset(&V, 0, sizeof V);
		spirv_float_opts(0, 0, 0, 0, 0);
		fnl_mode = 0;
		return;
	}
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
		free(orig_mod);
		orig_mod = full;	/* for the GLSL view (spvcross.c) */
		if (said.n)
			fputs(said.p, stdout);
		free(said.p);
	}
	declare();
	SP.orig_nid = V.nid;
	/* a snippet's inputs: their inferred types (they are free inputs) */
	for (k = 0; k < ninps; k++)
		if (!inps[k].var && inps[k].ty[0])
			V.id[idx(inps[k].name)].type = idx(inps[k].ty);
	/* result types, definitions */
	SP.def = xmalloc((size_t)(V.nid + 1024) * sizeof *SP.def);
	SP.alias = xmalloc((size_t)(V.nid + 1024) * sizeof *SP.alias);
	for (i = 0; i < V.nid + 1024; i++)
		SP.def[i] = SP.alias[i] = -1;
	SP.orig_nin = V.nin;
	SP.dead = xcalloc((size_t)V.nin + 1024, 1);
	SP.rw = xcalloc((size_t)V.nin + 1024, 1);
	SP.before = xmalloc(((size_t)V.nin + 1024) * sizeof *SP.before);
	for (i = 0; i < V.nin + 1024; i++)
		SP.before[i] = -1;
	for (i = 0; i < npend_as; i++) {
		snprintf(SP.as[SP.nas].name, sizeof SP.as[0].name, "%s",
		    pend_as[i].name);
		SP.as[SP.nas].lo = pend_as[i].lo;
		SP.as[SP.nas++].hi = pend_as[i].hi;
	}
	npend_as = 0;
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
	SP.dl = xcalloc((size_t)(V.nid + 1024), sizeof *SP.dl);
	nfwd = forward_loads();
	ndst = dead_stores();
	SP.uses = xcalloc((size_t)(V.nid + 1024), sizeof *SP.uses);
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strncmp(I->op, "OpName", 6) || !strncmp(I->op, "OpMemberName",
		    12) || !strncmp(I->op, "OpDecorate", 10) ||
		    !strncmp(I->op, "OpMemberDecorate", 16))
			continue;
		if (SP.dead[k])
			continue;	/* a forwarded load, a dead store: gone */
		for (i = 0; i < I->nop; i++)
			if (I->opnd[i][0] == '%')
				SP.uses[res_of(idx(I->opnd[i]))]++;
	}
	SP.memo = xcalloc((size_t)(V.nid + 1024), sizeof *SP.memo);
	SP.have = xcalloc((size_t)(V.nid + 1024), 1);
	SP.touched = xmalloc((size_t)(V.nid + 1024) * sizeof *SP.touched);
	SP.ev = xcalloc((size_t)(V.nid + 1024), sizeof *SP.ev);
	SP.evstamp = xcalloc((size_t)(V.nid + 1024), sizeof *SP.evstamp);
	SP.rng = 0x51c3a5e1ULL;
	SP.deadline = (double)t0 / CLOCKS_PER_SEC + budget;
	SP.g = aig_new();
	before = spv_cover(NULL);
	for (k = 0; k < V.nin; k++)
		if (is_idot(k)) {
			int cls;
			before += idot_cost(&V.in[k], &cls);
		}
	/* in order: a rewrite is seen by the instructions after it */
	for (k = 0; k < SP.orig_nin; k++) {
		struct cand cs[12];
		int nc, best = -1, rt;
		double cost, gain = 0;
		const char *ext;
		if (!pure_ins(k) || V.nin >= SP.orig_nin + 1000 ||
		    V.nid + 64 >= SP.orig_nid + 1024)
			continue;
		rt = idx(V.in[k].opnd[0]);
		ext = !strcmp(V.in[k].op, "OpExtInst") ? V.in[k].opnd[2] : NULL;
		(void)ext;
		cost = ins_cost_k(k);
		nc = candidates(k, cs, 12);
		/* the cheapest that is cheaper and proves */
		for (;;) {
			int j, r;
			best = -1;
			for (j = 0; j < nc; j++)
				if (cs[j].why && cost + cs[j].extra - cs[j].cost > 1e-9 &&
				    (best < 0 || cs[j].cost - cs[j].extra <
				    cs[best].cost - cs[best].extra))
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
				gain = cost + cs[best].extra - cs[best].cost;
				break;
			}
			/* a near miss worth telling: a shift that merges, but not
			 * everywhere */
			if (r == 0 && cs[best].pre_op[0] && SP.cex[0] &&
			    SP.nnotes < 6) {
				char ct[200];
				cand_text(ct, sizeof ct, &cs[best], &V.in[k]);
				char sl[160];
				src_line(ins_line(k), sl, sizeof sl);
				snprintf(SP.notes[SP.nnotes++], sizeof SP.notes[0],
				    "%s\n      ->  %s\n      fails at %s\n%s",
				    V.in[k].text, ct, SP.cex, sl);
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
			L->algebra = cs[best].algebra;
			L->trivial = driver_trivial(k, &cs[best]);
			L->line = ins_line(k);
		}
		apply(k, &cs[best]);
		nrw++;
		SP.proved++;
	}
	/* sums of products as dot products, under the active model */
	{
		struct dotlog dl[32];
		int nd = dot_pass(&dotrep, dl, 32), q;
		for (q = 0; q < nd && nlog < MAXLOG; q++) {
			struct rwlog *L = &rl[nlog++];
			snprintf(L->from, sizeof L->from, "%s", dl[q].from);
			snprintf(L->to, sizeof L->to, "%s", dl[q].to);
			L->why = "a sum of products as one dot product";
			L->gain = dl[q].gain;
			L->line = dl[q].line;
			L->dot = 1;
		}
		ndotlog = nd;
		memcpy(dotl, dl, (size_t)nd * sizeof *dl);
		nrw += nd;
		ndotrw = nd;
		SP.proved += nd;
	}
	/* dead pure instructions: no uses left */
	{
		int *uses = xcalloc((size_t)(V.nid + 1024), sizeof *uses), changed = 1;
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
	memset(&ftally, 0, sizeof ftally);
	after = spv_cover(&ftally);
	for (k = 0; k < V.nin; k++)
		if (!SP.dead[k] && is_idot(k)) {
			int cls;
			after += idot_cost(&V.in[k], &cls);
		}
	/* range search on stored and returned values with a bounded input */
	if (SP.nas > 0) {
		struct sbuf rs = {0};
		int nr = 0;
		SP.force_id = -1;
		for (k = 0; k < SP.orig_nin && nr < 4; k++) {
			const struct sins *I = &V.in[k];
			int v = -1;
			if (SP.dead[k])
				continue;
			if (!strcmp(I->op, "OpStore") && I->nop >= 2)
				v = res_of(idx(I->opnd[1]));
			else if (!strcmp(I->op, "OpReturnValue") && I->nop >= 1)
				v = res_of(idx(I->opnd[0]));
			if (v < 0 || SP.def[v] < 0 || !pure_ins(SP.def[v]))
				continue;
			{
				size_t before_n = rs.n;
				range_search(v, ins_line(SP.def[v]), &rs);
				if (rs.n > before_n)
					nr++;
			}
		}
		range_report = rs.p;
	}
	/* the module, rewritten */
	{
		int first_fn = -1, last_cap = -1;
		for (k = 0; k < SP.orig_nin; k++)
			if (!strcmp(V.in[k].op, "OpCapability"))
				last_cap = k;
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
			if (k >= SP.orig_nin)
				break;
			if (dotcaps && k == last_cap + 1)
				dot_caps_out(&out);
			{
				int q;
				for (q = SP.orig_nin; q < V.nin; q++)
					if (SP.before[q] == k && !SP.dead[q]) {
						const struct sins *N = &V.in[q];
						int j;
						sb_printf(&out, "%s = %s %s", V.id[N->res].tok,
						    N->op, N->opnd[0]);
						for (j = 1; j < N->nop; j++)
							sb_printf(&out, " %s", N->opnd[j][0] == '%' ?
							    V.id[res_of(idx(N->opnd[j]))].tok :
							    N->opnd[j]);
						sb_putc(&out, '\n');
					}
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
	{
		int nal = 0;
		for (i = 0; i < nlog; i++)
			nal += rl[i].algebra;
		printf("simplify: %d SPIR-V instructions, %d rewrite%s proved by SAT "
		    "(bit-precise: equal for every input)", SP.orig_nin, nrw - nal -
		    ndotrw, nrw - nal - ndotrw == 1 ? "" : "s");
		if (nal)
			printf(" and %d by algebra mod 2^w", nal);
		if (ndotrw)
			printf(", %d sum%s of products made dot products (proved by "
			    "dot.c, below)", ndotrw, ndotrw == 1 ? "" : "s");
		printf(", %d dead instruction%s removed\n", ndead,
		    ndead == 1 ? "" : "s");
	}
	if (nfwd || ndst)
		printf("  %d load%s of a variable just stored to became the stored "
		    "value; %d store%s overwritten before any read removed\n", nfwd,
		    nfwd == 1 ? "" : "s", ndst, ndst == 1 ? "" : "s");
	for (i = 0; i < SP.nas; i++)
		if (!as_float_only(SP.as[i].name))
			printf("  assuming %s in [%lld, %lld] (every load of it)\n",
		    SP.as[i].name, (long long)SP.as[i].lo, (long long)SP.as[i].hi);
	{
		ntriv = 0;
		tgain = 0;
		for (i = 0; i < nlog; i++)
			if (rl[i].trivial) {
				ntriv++;
				tgain += rl[i].gain;
			}
	}
	rep_begin("spirv_simplify");
	rep_int("instructions", SP.orig_nin);
	rep_int("dead_removed", ndead);
	rep_str("model", sym_str(model_active()->name));
	rep_num("cost_before", before);
	rep_num("cost_after", after);
	if (ntriv) {
		/* what a driver's back end would make of the input anyway */
		rep_num("cost_driver_baseline", before - tgain);
		rep_int("driver_trivial_count", ntriv);
		rep_num("driver_trivial_gain", tgain);
	}
	rep_list("assumptions");
	for (i = 0; i < SP.nas; i++) {
		rep_obj(NULL);
		rep_str("name", SP.as[i].name);
		rep_int("lo", SP.as[i].lo);
		rep_int("hi", SP.as[i].hi);
		rep_end();
	}
	rep_end();
	rep_list("rewrites");
	for (i = 0; i < nlog; i++) {
		rep_obj(NULL);
		rep_str("from", rl[i].from);
		rep_str("to", rl[i].to);
		rep_str("why", rl[i].why);
		rep_num("gain", rl[i].gain);
		rep_int("line", rl[i].line);
		rep_str("proof", rl[i].algebra ? "algebra mod 2^w" : rl[i].dot ?
		    "dot" : "sat");
		if (ntriv)
			rep_bool("driver_trivial", rl[i].trivial);
		rep_end();
	}
	rep_end();
	if (ndotlog) {
		rep_list("dot_products");
		for (i = 0; i < ndotlog; i++) {
			rep_obj(NULL);
			rep_str("from", dotl[i].from);
			rep_str("to", dotl[i].to);
			rep_int("line", dotl[i].line);
			rep_num("cost_before", dotl[i].before);
			rep_num("cost_after", dotl[i].after);
			rep_str("relies_on", dotl[i].desc);
			rep_str("proof", dotl[i].proof);
			rep_end();
			free(dotl[i].desc);
			free(dotl[i].proof);
		}
		rep_end();
	}
	if (nlog)
		rep_proof("sat", "each rewrite bit-blasted and proved equal for "
		    "every input (refinement)");
	if (ntriv) {
		/* the headline: beyond spirv-opt -O and a driver's back end */
		double b2 = before - tgain;
		printf("  integer arithmetic cost under %s (each instruction once), "
		    "beyond what spirv-opt -O and a driver's back end do: %g -> %g",
		    sym_str(model_active()->name), b2, after);
		if (b2 > 0)
			printf(" (%+.1f%%)", 100 * (after - b2) / b2);
		printf("\n  counting the strength reductions any driver also does: "
		    "%g -> %g", before, after);
		if (before > 0)
			printf(" (%+.1f%%)", 100 * (after - before) / before);
		printf("\n");
	} else {
		printf("  integer arithmetic cost under %s (each instruction once): "
		    "%g -> %g", sym_str(model_active()->name), before, after);
		if (before > 0)
			printf(" (%+.1f%%)", 100 * (after - before) / before);
		printf("\n");
	}
	if (ntriv) {
		/* one line: the kinds and how many, kept in the module */
		const char *kinds[16];
		int cnt[16], nk = 0, j;
		for (i = 0; i < nlog; i++) {
			if (!rl[i].trivial)
				continue;
			for (j = 0; j < nk; j++)
				if (!strcmp(kinds[j], rl[i].why))
					break;
			if (j == nk && nk < 16) {
				kinds[nk] = rl[i].why;
				cnt[nk++] = 0;
			}
			if (j < 16)
				cnt[j]++;
		}
		printf("  also done by any driver compiler (%d rewrite%s, -%g; kept in "
		    "the written module, not counted above):", ntriv, ntriv == 1 ?
		    "" : "s", tgain);
		for (j = 0; j < nk; j++)
			printf("%s %d %s", j ? ";" : "", cnt[j], kinds[j]);
		printf("\n");
	}
	if (ftally.n) {
		/* the model's fusions in the rewritten module (both totals are
		 * priced with them) */
		printf("  fused by the model in the result (one instruction for "
		    "several):\n");
		rep_list("fusions");
		fuse_tally_print(&ftally, "    ");
		rep_end();
	}
	if (nlog - ntriv > 0) {
		/* by kind, then the largest savings first */
		const char *kinds[32];
		int cnt[32], nk = 0, j, shown;
		double sum[32];
		for (i = 0; i < nlog; i++) {
			if (rl[i].trivial)
				continue;
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
		/* stable: the driver's own last, then by gain, then order */
		for (i = 1; i < nlog; i++) {
			struct rwlog x = rl[i];
			for (j = i; j > 0 && (rl[j - 1].trivial > x.trivial ||
			    (rl[j - 1].trivial == x.trivial && rl[j - 1].gain <
			    x.gain)); j--)
				rl[j] = rl[j - 1];
			rl[j] = x;
		}
		shown = nlog - ntriv < 25 ? nlog - ntriv : 25;
		printf("  %s:\n", shown < nlog - ntriv ? "the largest savings" :
		    "rewrites");
		for (i = 0; i < shown; i++) {
			char sl[160];
			src_line(rl[i].line, sl, sizeof sl);
			sb_printf(&list, "    %s\n      ->  %s   (%s; -%g%s)\n%s",
			    rl[i].from, rl[i].to, rl[i].why, rl[i].gain, rl[i].algebra ?
			    "; by algebra mod 2^w, not SAT" : "", sl);
		}
		if (shown < nlog - ntriv)
			sb_printf(&list, "    ... %d more (all in the attached "
			    "module)\n", nlog - ntriv - shown);
		fputs(list.p, stdout);
	} else if (ntriv)
		printf("  beyond those, no cheaper form of an integer instruction "
		    "could be proved\n");
	else
		printf("  nothing to rewrite: no cheaper form of an integer "
		    "instruction could be proved\n");
	if (dotrep.n) {
		printf("  dot products (a sum of products as one instruction, "
		    "laid out and proved by dot.c under %s):\n",
		    sym_str(model_active()->name));
		fputs(dotrep.p, stdout);
	}
	free(dotrep.p);
	if (range_report) {
		fputs(range_report, stdout);
		free(range_report);
	}
	if (SP.nnotes) {
		int q;
		printf("  not applied (the solver found an input where it "
		    "differs):\n");
		for (q = 0; q < SP.nnotes; q++)
			printf("    %s", SP.notes[q]);
		printf("  if those inputs cannot happen, say so (a narrower range, "
		    "e.g. `n in [1, 32)`) and ask again\n");
	}
	if (SP.undecided)
		printf("  %ld candidate%s too hard for the solver's budget (kept as "
		    "they are)\n", SP.undecided, SP.undecided == 1 ? "" : "s");
	if (SP.timeout)
		printf("  note: stopped proving after %g s (the time budget; "
		    "`budget N` for more)\n", budget);
	{
		double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
		char tb[32] = "";
		if (show_time(el))
			snprintf(tb, sizeof tb, " in %.1f s", el);
		printf("  proved%s; the operands' definitions were expanded up to "
		    "16 levels, the rest (loads, loop phis, spec constants) are "
		    "free inputs\n", tb);
	}
	/* the function level: the GLSL source's functions (glslfn.c) */
	if (src_text != NULL && fnl_mode != 2)
		glslfn_pass(src_text, budget, 0);
	free(list.p);
	free(rl);
	free(SP.dl);
	free(SP.uses);
	free(SP.before);
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
	/* the float instructions, on the module as the integer part left it */
	{
		int fch = 0, dch = 0;
		char *ft, *dt;
		dw_nfas = npend_fas;	/* float_pass ends the assumptions */
		ft = float_pass(out.p ? out.p : "", &fch);
		/* f64 over f32 and folded round trips, on what that left */
		dt = dword_pass(fch ? ft : out.p ? out.p : "", &dch);
		if (dch) {
			free(ft);
			ft = dt;
			fch = 1;
		}
		if (fch)
			deliver_file("simplified.spvasm", "text/plain", ft, strlen(ft),
			    nofile);
		else if (nrw + ndead > 0)
			deliver_file("simplified.spvasm", "text/plain", out.p, out.n,
			    nofile);
		/* spirv-cross's GLSL of it, beside spirv-opt -O's */
		if (fch || nrw + ndead > 0)
			spvcross_after_simplify(orig_mod, fch ? ft : out.p, nofile);
		free(ft);
		free(orig_mod);
		orig_mod = NULL;
	}
	free(out.p);
	spirv_float_opts(0, 0, 0, 0, 0);
	fnl_mode = 0;
}
