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
 *   - OpControlBarrier stalls every lane; an atomic is a warp instruction
 *     for the lanes on distinct addresses, replayed once more for each
 *     further lane on the busiest address.
 * Memory: Uniform, StorageBuffer, Workgroup, PushConstant and
 * UniformConstant variables are shared by the lanes; Function, Private,
 * Input and Output are per lane.  Values are raw bit patterns per
 * component, read through their type.  Each executed instruction goes to
 * the animation (anim.c) with its cost under the active model.
 */
#include "ucas.h"
#include "calc.h"
#include "fuse.h"

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
#include "memk.h"

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

void glslfn_pass(const char *src, double budget, int only, const char *model,
    int dotok);

/* the execution model of the module's first entry point ("GLCompute",
 * "Fragment" ...), or NULL */
static const char *
entry_model(void)
{
	int k;

	for (k = 0; k < V.nin; k++)
		if (!strcmp(V.in[k].op, "OpEntryPoint") && V.in[k].nop >= 1)
			return V.in[k].opnd[0];
	return NULL;
}
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
				{
					/* spirv-dis writes a quote as \" and a backslash as \\ */
					char *r = src_text, *w = src_text;
					while (*r) {
						if (r[0] == '\\' && (r[1] == '"' || r[1] == '\\'))
							r++;
						*w++ = *r++;
					}
					*w = 0;
				}
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

/* a w-bit value read signed */
static int64_t
sx64(uint64_t v, int w)
{
	return w >= 64 ? (int64_t)v : (int64_t)(v << (64 - w)) >> (64 - w);
}

/* the high w bits (w <= 32) of the 2w-bit product of x and y, both read
 * unsigned or both signed (sg): Op[US]MulExtended's second member */
static uint64_t
mulx_hi(uint64_t x, uint64_t y, int w, int sg)
{
	uint64_t p = sg ? (uint64_t)(sx64(x & wmask(w), w) * sx64(y & wmask(w),
	    w)) : (x & wmask(w)) * (y & wmask(w));

	return p >> w & wmask(w);
}

static uint64_t
puti(int st, int64_t v)
{
	struct sty *y = T(st);

	if (y->k == S_BOOL)
		return v != 0;
	return (uint64_t)v & wmask(y->w);
}

/*
 * spirv-dis writes an infinity or a NaN of w bits as a hex float whose exponent is one
 * past the largest (0x1p+128, 0x1.8p+128 for a float), the fraction's bits left-aligned
 * after the point; strtod reads that as a finite double, which rounds to an infinity
 */
static int
parse_fspecial(int w, const char *s, uint64_t *bits)
{
	int fw = w == 16 ? 10 : w == 32 ? 23 : 52;
	long emax = w == 16 ? 16 : w == 32 ? 128 : 1024;
	uint64_t sign = 0, frac = 0;
	int n = 0;

	if (*s == '-' || *s == '+')
		sign = *s++ == '-';
	if (strncmp(s, "0x1", 3) && strncmp(s, "0X1", 3))
		return 0;
	s += 3;
	if (*s == '.')
		for (s++; isxdigit((unsigned char)*s) && n < 16; s++, n++)
			frac = frac << 4 |
			    (uint64_t)(isdigit((unsigned char)*s) ? *s - '0' :
			    tolower((unsigned char)*s) - 'a' + 10);
	if ((*s != 'p' && *s != 'P') || strtol(s + 1, NULL, 10) != emax)
		return 0;
	frac = 4 * n > fw ? frac >> (4 * n - fw) : frac << (fw - 4 * n);
	*bits = sign << (w - 1) | wmask(w - 1 - fw) << fw | (frac & wmask(fw));
	return 1;
}

static uint64_t
parse_literal(int st, const char *s)
{
	uint64_t b;

	if (T(st)->k == S_FLOAT && parse_fspecial(T(st)->w, s, &b))
		return b;
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
	B_WGID, B_NUMWG, B_FRONT, B_HELPER, B_VERTEX, B_SGSIZE, B_SGEQ, B_SGGE,
	B_SGGT, B_SGLE, B_SGLT };

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
		{"SubgroupSize", B_SGSIZE}, {"SubgroupEqMask", B_SGEQ},
		{"SubgroupGeMask", B_SGGE}, {"SubgroupGtMask", B_SGGT},
		{"SubgroupLeMask", B_SGLE}, {"SubgroupLtMask", B_SGLT},
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
		case B_SGSIZE:	/* the lanes run are one subgroup */
			b = puti(st, V.lanes);
			break;
		case B_SGEQ: case B_SGGE: case B_SGGT: case B_SGLE: case B_SGLT:
			/* word s of the mask; lanes past the subgroup have no bit */
			if (s == 0) {
				uint64_t all = wmask(V.lanes), at = 1ULL << lane;
				b = d->builtin == B_SGEQ ? at : d->builtin == B_SGLE ?
				    (at << 1) - 1 : d->builtin == B_SGLT ? at - 1 :
				    d->builtin == B_SGGE ? all & ~(at - 1) :
				    all & ~((at << 1) - 1);
				b = puti(st, (int64_t)b);
			}
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

/* the catalogue's class k (inst.c) where the model lists it, else fb */
static int
cat_class(int k, int fb)
{
	return cclass_available(model_active(), k) ? k : fb;
}

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
	if (!strcmp(op, "OpBitCount"))
		return C_BITCNT;
	/* the catalogue's classes where the model has them, else what the
	 * instruction was priced as before there were any */
	if (!strcmp(op, "OpBitReverse"))
		return cat_class(C_BREV, C_BITCNT);
	if (!strcmp(op, "OpBitFieldInsert"))
		return cat_class(C_BFI, C_SHIFT);
	if (!strncmp(op, "OpBitField", 10))
		return cat_class(C_BFE, C_SHIFT);
	if (!strcmp(op, "OpUMulExtended") || !strcmp(op, "OpSMulExtended"))
		return cat_class(C_MULHI, C_IMUL);
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

/*
 * A load or a store of workgroup memory by the lanes of mask through
 * pointer p: its kind's price with the bank conflicts of the addresses
 * the lanes use (the largest number of different addresses on one bank,
 * each after the first at bank_conflict).  A model that lists no memory
 * kinds prices it as any load or store, as before.  *cls: the class it is
 * shown under.
 */
static double
wg_price(int p, uint32_t mask, int store, int *cls)
{
	const struct model *m = model_active();
	int banks = (int)m->par[MP_BANKS], bytes = (int)m->par[MP_BANK_BYTES];
	long addr[NL];
	int n = 0, l, i, j, deg = 1, kind = store ? C_STORE_SHARED :
	    C_LOAD_SHARED;

	*cls = model_lists(m, kind) ? kind : store ? C_STORE : C_LOAD;
	if (banks <= 0 || bytes <= 0)
		return bank_price(m, store, 1);
	for (l = 0; l < V.lanes; l++) {
		struct lval a;
		const struct sid *d;
		long at = 0;
		if (!(mask >> l & 1))
			continue;
		a = val(p, l);
		d = &V.id[a.var];
		/* the byte the access starts at: the slots before it */
		for (i = 0; i < a.off && i < d->nslots; i++)
			at += T(d->slotty[i])->w > 32 ? 8 : 4;
		for (i = 0; i < n; i++)
			if (addr[i] == at)
				break;
		if (i == n)
			addr[n++] = at;
	}
	for (i = 0; i < n; i++) {
		int same = 0;
		for (j = 0; j < n; j++)
			same += addr[j] / bytes % banks == addr[i] / bytes % banks;
		if (same > deg)
			deg = same;
	}
	return bank_price(m, store, deg);
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
	if (OP("OpShiftRightArithmetic")) {
		/* the sign bit of the bits, whatever the type's signedness */
		int sw = 64 - T(at)->w;
		return puti(rt, ((int64_t)(ua << sw) >> sw) >> (ub & 63));
	}
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
	int l, c, k, cls, f32 = 0;

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
		/* a fetch, a storage image's read: their own kinds (load where
		 * the model lists neither) */
		cls = wr ? C_STORE : op[7] == 'F' ? C_TEX_FETCH : C_IMAGE_LOAD;
		k = emit(I, I_OP, model_class(model_active(), cls),
		    (double)model_cost(model_active(), cls), mask);
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
			if (tex_fmts[G.t[ti].fmt].cls == TC_FLOAT &&
			    tex_fmts[G.t[ti].fmt].b[0] == 32)
				f32 = 1;
		}
		/* a filtered read: its kind, an f32 texture's apart (measured on
		 * rgba8 and rgba32f) */
		cls = f32 ? C_TEX_SAMPLE_F32 : C_TEX_SAMPLE;
		k = emit(I, I_OP, model_class(model_active(), cls),
		    (double)model_cost(model_active(), cls), mask);
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

/* nops multiplies or multiply-adds of scalar type st, as the cover prices
 * the same arithmetic written out with its multiply-adds fused: fmul
 * each, times the model's f64 factor for f64 */
static double
fma_price(int st, int nops)
{
	double c = model_active()->c[C_FMUL] * nops;

	if (st >= 0 && T(st)->k == S_FLOAT && T(st)->w == 64)
		c *= model_active()->c[C_F64];
	return c;
}

static int
exec_matrix(const struct sins *I, uint32_t mask, int rt, int st)
{
	const char *op = I->op;
	int l, i, j, q, k, n = 0, nops = 0;

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
			nops = 0;	/* moves */
		} else if (!strcmp(op, "OpMatrixTimesScalar")) {
			n = nops = ncomp(at);
			for (i = 0; i < n; i++)
				v[i] = AF(x, i) * AF(y, 0);
		} else if (!strcmp(op, "OpVectorTimesMatrix")) {
			/* v (R) * M (C columns of R): C results, dot with columns */
			int C = T(bt)->n, R = ncomp(T(bt)->e);
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					v[i] += AF(x, j) * AF(y, i * R + j);
			n = C;
			nops = C * R;
		} else if (!strcmp(op, "OpMatrixTimesVector")) {
			int C = T(at)->n, R = ncomp(T(at)->e);
			for (j = 0; j < R; j++)
				for (i = 0; i < C; i++)
					v[j] += AF(x, i * R + j) * AF(y, i);
			n = R;
			nops = C * R;
		} else if (!strcmp(op, "OpMatrixTimesMatrix")) {
			/* A (K columns of R) * B (C columns of K) */
			int K = T(at)->n, R = ncomp(T(at)->e), C = T(bt)->n;
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					for (q = 0; q < K; q++)
						v[i * R + j] += AF(x, q * R + j) * AF(y, i * K + q);
			n = C * R;
			nops = C * R * K;
		} else {	/* OuterProduct: column i is a * b[i] */
			int R = vcomp(a), C = vcomp(b);
			for (i = 0; i < C; i++)
				for (j = 0; j < R; j++)
					v[i * R + j] = AF(x, j) * AF(y, i);
			n = nops = C * R;
		}
#undef AF
		r = LV(I->res, l);
		memset(r, 0, sizeof *r);
		for (i = 0; i < n && i < MAXCOMP; i++)
			r->b[i] = putf(st, v[i]);
	}
	/* a product of R x K by K x C is C R dot products of K terms: one
	 * multiply and K - 1 multiply-adds each, every one priced as fmul */
	k = nops ? emit(I, I_OP, C_FMUL, fma_price(st, nops), mask) :
	    emit(I, I_OP, -1, 0, mask);
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
static int exec_subgroup(const struct sins *I, uint32_t mask, int rt, int nc);
static void sgx_reset(void);
static void sgx_notes(void);

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
		if (sc == SC_WG) {
			int cls;
			double c = wg_price(p, mask, 1, &cls);
			k = emit(I, I_OP, cls, c, mask);
		} else
			k = emit(I, I_OP, sc == SC_SHARED ? C_STORE : -1,
			    sc == SC_SHARED ? model_active()->c[C_STORE] : 0, mask);
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
		if (sc == SC_WG) {
			int cls;
			double c = wg_price(p, mask, 0, &cls);
			k = emit(I, I_OP, cls, c, mask);
		} else
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
	if ((!strcmp(I->op, "OpUMulExtended") || !strcmp(I->op,
	    "OpSMulExtended")) && T(rt)->k == S_STRUCT && T(rt)->nm == 2) {
		/* { the low words, the high words } of each component */
		int mt = T(rt)->m[0], ms = scalar_of(mt), mn = ncomp(mt), sg =
		    I->op[2] == 'S', wd = T(ms)->w, j;
		if (wd > 32 || 2 * mn > MAXCOMP)
			die("unsupported instruction \"%s\"", I->text);
		for (l = 0; l < V.lanes; l++)
			if (mask >> l & 1) {
				struct lval x = val(opid(I, 1), l), y = val(opid(I, 2), l),
				    *r = LV(I->res, l);
				memset(r, 0, sizeof *r);
				for (j = 0; j < mn; j++) {
					r->b[j] = puti(ms, (int64_t)(x.b[j] * y.b[j]));
					r->b[mn + j] = mulx_hi(x.b[j], y.b[j], wd, sg);
				}
			}
		cls = cost_class(I->op, NULL, ms, &mul);
		k = emit(I, I_OP, cls, cls >= 0 ? model_active()->c[cls] * mul * mn :
		    0, mask);
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
		cost = fma_price(sa, n);
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
		static const char *const known[] = {"IAdd", "ISub", "IIncrement",
			"IDecrement", "SMin", "SMax", "UMin", "UMax", "And", "Or",
			"Xor", "Exchange", "Store", "Load", "CompareExchange",
			"FAddEXT", "FMinEXT", "FMaxEXT", NULL};
		const struct model *mo = model_active();
		int p = opid(I, 1), sc, cls, j, same = 0, nact = 0, wt, first = -1;
		int round[64] = {0}, nround = 1;
		struct lval pv = val(p, 0);
		const char *a = I->op + 8;
		const void *addr[64] = {0};
		double cost;
		for (j = 0; known[j] && strcmp(a, known[j]) != 0; j++)
			;
		if (!known[j])
			fail("SPIR-V: %s is not run by the executor", I->op);
		sc = pv.var >= 0 ? V.id[pv.var].sc : SC_SHARED;
		/* a float add, min or max has its own class; a model without
		 * it prices it as an integer atomic */
		cls = sc == SC_WG ? C_ATOM_SH : C_ATOM_GL;
		wt = I->res >= 0 ? opid(I, 0) : type_of(opid(I, 3));
		if ((!strcmp(a, "FAddEXT") || !strcmp(a, "FMinEXT") || !strcmp(a,
		    "FMaxEXT")) && model_lists(mo, sc == SC_WG ?
		    C_ATOM_SH_F : C_ATOM_GL_F))
			cls = sc == SC_WG ? C_ATOM_SH_F : C_ATOM_GL_F;
		else if (sc != SC_WG && !strcmp(a, "CompareExchange") &&
		    model_lists(mo, C_ATOM_GL_CAS))
			cls = C_ATOM_GL_CAS;
		else if (sc != SC_WG && wt >= 0 && T(wt)->w == 64 &&
		    model_lists(mo, C_ATOM_GL_64))
			cls = C_ATOM_GL_64;
		else if (sc == SC_WG && model_lists(mo, C_ATOM_SH_SAME)) {
			/* every active lane on one address: one warp instruction
			 * that serialises its lanes, priced for its share of a warp
			 * (atomic_shared_same is 32 lanes on one address) */
			same = 1;
			for (l = 0; l < V.lanes; l++)
				if (mask >> l & 1) {
					struct lval q = val(p, l);
					nact++;
					if (first < 0)
						first = l;
					else if (q.var != val(p, first).var || q.off !=
					    val(p, first).off)
						same = 0;
				}
			same = same && nact > 1;
		}
		if (same) {
			double w = mo->par[MP_WARP] >= 2 ? mo->par[MP_WARP] : 32;
			cls = C_ATOM_SH_SAME;
			k = emit(I, I_ATOMIC, cls, (double)model_cost(mo, cls) *
			    (nact < w ? nact : w) / w, mask);
		}
		cost = (double)model_cost(mo, cls);
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
			else if (!strcmp(a, "FAddEXT")) *m = putf(st, getf(st, old) + getf(st, x));
			/* FMin and FMax: a NaN operand gives the other */
			else if (!strcmp(a, "FMinEXT")) *m = putf(st, fmin(getf(st, old), getf(st, x)));
			else if (!strcmp(a, "FMaxEXT")) *m = putf(st, fmax(getf(st, old), getf(st, x)));
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
			if (same) {
				/* the warp's one instruction: the values on it */
				if (I->res >= 0 && k >= 0) {
					char buf[64];
					fmt(buf, sizeof buf, V.id[I->res].type,
					    LV(I->res, l));
					xr_val(k, l, buf);
				}
				continue;
			}
			/* the lanes before it on its address: its replay */
			addr[l] = texw != NULL ? (const void *)texw : (const void *)m;
			for (j = 0; j < l; j++)
				if ((mask >> j & 1) && addr[j] == addr[l])
					round[l]++;
			if (round[l] + 1 > nround)
				nround = round[l] + 1;
		}
		/*
		 * One warp instruction for lanes on distinct addresses, the
		 * class's price per warp; the lanes on one address go one at a
		 * time, so it is replayed once more for each further lane on the
		 * busiest address.  The lanes ran above in their order, which
		 * is the order the replays give.
		 */
		for (j = 0; !same && j < nround; j++) {
			uint32_t rm = 0;
			for (l = 0; l < V.lanes; l++)
				if ((mask >> l & 1) && round[l] == j)
					rm |= 1u << l;
			k = emit(I, I_ATOMIC, cls, cost, rm);
			if (I->res < 0 || k < 0)
				continue;
			for (l = 0; l < V.lanes; l++)
				if (rm >> l & 1) {
					char buf[64];
					fmt(buf, sizeof buf, V.id[I->res].type,
					    LV(I->res, l));
					xr_val(k, l, buf);
				}
		}
		return;
	}
	if (exec_image(I, mask, rt, st, nc) || exec_matrix(I, mask, rt, st) ||
	    exec_glslvec(I, mask, rt, st, nc) || exec_idot(I, mask, rt) ||
	    exec_subgroup(I, mask, rt, nc))
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
	sgx_reset();
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
	sgx_notes();
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
	int a[4], na;		/* its operands (ids) */
	double cost;
	const char *why;
	int via_u, ut;		/* RW_INS on the bits as unsigned type ut:
				 * bitcasts in and out (OpUDiv and OpUMod
				 * want an unsigned result type) */
	int mulx;		/* op is Op[US]MulExtended of a[0], a[1]: its
				 * high word, shifted right by msh (by an
				 * arithmetic shift when mash) */
	int msh, mash;
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
	int mxty[4], nmxty;	/* struct { T, T } types made here */
	int nval;		/* instructions of kinds the module may not have
				 * had: spirv-val is asked */
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
		lit lo_ok;
		if (!assumed(SP.leaf[i].id, &lo, &hi))
			continue;
		w_const(&l, x->w, (uint64_t)lo & wmask(x->w));
		w_const(&h, x->w, (uint64_t)hi & wmask(x->w));
		if (lo < 0)	/* signed bounds */
			lo_ok = L_NOT(w_slt(SP.g, x, &l));
		else
			lo_ok = L_NOT(w_ult(SP.g, x, &l));
		a = aig_and(SP.g, a, aig_and(SP.g, lo_ok, lo < 0 ?
		    L_NOT(w_slt(SP.g, &h, x)) : L_NOT(w_ult(SP.g, &h, x))));
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
	uint64_t x[4] = {0};
	int i, ok = 1, at;

	if (c->kind == RW_ID)
		return evalid(c->id);
	if (c->kind == RW_CONST)
		return c->v;
	for (i = 0; i < c->na; i++)
		x[i] = evalid(c->a[i]);
	if (c->mulx) {
		int w = int_width(rt);
		uint64_t h = mulx_hi(x[0], x[1], w, c->op[2] == 'S');
		h = c->mash ? (uint64_t)(sx64(h, w) >> c->msh) : h >> c->msh;
		return h & wmask(w);
	}
	if (c->pre_op[0]) {
		int pt = V.id[res_of(c->pre_a[0])].type;
		x[1] = alu(c->pre_op, pt, pt, evalid(c->pre_a[0]),
		    evalid(c->pre_a[1]), 0, &ok) & wmask(int_width(pt));
	}
	at = V.id[res_of(c->a[0])].type;
	if (!strcmp(c->op, "OpBitFieldInsert")) {
		int off = (int)(x[2] & 63), cnt = (int)(x[3] & 127);
		uint64_t m = cnt >= 64 ? ~0ULL : wmask(cnt) << off;
		return (uint64_t)puti(rt, (int64_t)((x[0] & ~m) | (x[1] << off &
		    m))) & wmask(int_width(rt));
	}
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
		lit dz = w_eq(g, a[1], &z), mn0 = w_eq(g, a[0], &mn);
		return aig_or(g, dz, aig_and(g, mn0, w_eq(g, a[1], &m1)));
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
		else if (c->mulx) {
			/* the high half of the product of the operands made
			 * 2w bits wide, then the shift */
			struct word ea, eb, pr, sh;
			int w = int_width(rt), sg = c->op[2] == 'S';
			for (i = 0; i < 2; i++) {
				a[i] = blast(c->a[i], depths[di] - 1);
				dc = aig_and(SP.g, dc, SP.dl[res_of(c->a[i])]);
			}
			w_resize(&ea, a[0], 2 * w, sg);
			w_resize(&eb, a[1], 2 * w, sg);
			w_mul(SP.g, &pr, &ea, &eb);
			wc.w = w;
			for (i = 0; i < w; i++)
				wc.b[i] = pr.b[w + i];
			if (c->msh) {
				pr = wc;
				w_const(&sh, w, (uint64_t)c->msh);
				if (c->mash)
					w_ashr(SP.g, &wc, &pr, &sh);
				else
					w_lshr(SP.g, &wc, &pr, &sh);
			}
		} else {
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
	c->a[3] = -1;
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
/*
 * The one word id is made of through bitwise ops and shifts by constants
 * (constants aside), within depth levels: *leaf, 1; 0 when there are two
 * or more, or none
 */
static int
one_leaf(int id, int depth, int *leaf)
{
	static const char *const ops[] = {"OpBitwiseAnd", "OpBitwiseOr",
	    "OpBitwiseXor", "OpShiftLeftLogical", "OpShiftRightLogical", NULL};
	int k = SP.def[id], i;
	uint64_t v;

	if (const_of(id, &v))
		return 1;
	if (k < 0 || depth == 0 || SP.dead[k] || !in_list(V.in[k].op, ops)) {
		if (*leaf >= 0 && *leaf != id)
			return 0;
		*leaf = id;
		return 1;
	}
	if (!strncmp(V.in[k].op, "OpShift", 7) && !const_of(res_of(idx(
	    V.in[k].opnd[2])), &v))
		return 0;	/* a shift by a variable */
	for (i = first_arg(&V.in[k]); i < V.in[k].nop; i++)
		if (!one_leaf(res_of(idx(V.in[k].opnd[i])), depth - 1, leaf))
			return 0;
	return 1;
}

/* id as x & k (either order), k a constant: x and k */
static int
and_k(int id, int *x, uint64_t *k)
{
	int d = SP.def[id];

	if (d < 0 || SP.dead[d] || strcmp(V.in[d].op, "OpBitwiseAnd"))
		return 0;
	if (const_of(res_of(idx(V.in[d].opnd[2])), k)) {
		*x = res_of(idx(V.in[d].opnd[1]));
		return 1;
	}
	if (const_of(res_of(idx(V.in[d].opnd[1])), k)) {
		*x = res_of(idx(V.in[d].opnd[2]));
		return 1;
	}
	return 0;
}

/* id as x << k, k a constant */
static int
shl_k(int id, int *x, uint64_t *k)
{
	int d = SP.def[id];

	if (d < 0 || SP.dead[d] || strcmp(V.in[d].op, "OpShiftLeftLogical") ||
	    !const_of(res_of(idx(V.in[d].opnd[2])), k))
		return 0;
	*x = res_of(idx(V.in[d].opnd[1]));
	return 1;
}

/* id's instruction's cost when nothing but its one user reads it */
static double
only_use_cost(int id)
{
	return SP.def[id] >= 0 && SP.uses[id] == 1 ? ins_cost_k(SP.def[id]) : 0;
}

/*
 * l = base & ~m and r the field of a bitfield insert at bits p..p+n-1 (m
 * those bits, 0 < n < w): r = (ins << p) & m, (ins & (2^n - 1)) << p, or
 * ins & m with p = 0.  *extra: what goes dead with them.
 */
static int
bfi_parts(int l, int r, int w, int *base, int *ins, int *p, int *nb,
    double *extra)
{
	uint64_t km, k2, sh, m, all = wmask(w);
	int t, t2;

	if (!and_k(l, base, &km))
		return 0;
	m = ~km & all;
	if (m == 0 || m == all)
		return 0;
	for (*p = 0; !(m >> *p & 1); (*p)++)
		;
	for (*nb = 0; *p + *nb < w && (m >> (*p + *nb) & 1); (*nb)++)
		;
	if ((wmask(*nb) << *p) != m)
		return 0;	/* not one run of bits */
	*extra = only_use_cost(l);
	if (and_k(r, &t, &k2) && (k2 & all) == m) {
		*extra += only_use_cost(r);
		if (*p == 0) {
			*ins = t;
			return 1;
		}
		if (shl_k(t, ins, &sh) && sh == (uint64_t)*p) {
			if (SP.uses[r] == 1)
				*extra += only_use_cost(t);
			return 1;
		}
		return 0;
	}
	if (*p > 0 && shl_k(r, &t2, &sh) && sh == (uint64_t)*p &&
	    and_k(t2, ins, &k2) && (k2 & all) == wmask(*nb)) {
		*extra += only_use_cost(r);
		if (SP.uses[r] == 1)
			*extra += only_use_cost(t2);
		return 1;
	}
	return 0;
}

/*
 * id = (e(x) * e(y)) >> s narrowed to t (w bits): s a constant, w <= s <
 * 2w, the product 2w bits wide and e both zero extensions (*sg 0) or both
 * sign extensions (*sg 1) of values of type t; a constant operand that is
 * such an extension of a w-bit value is that value.  *sh = s - w, *ar an
 * arithmetic shift, *extra what goes dead with them.
 */
static int
mulx_parts(int id, int t, int w, int *x, int *y, int *sg, int *sh, int *ar,
    double *extra)
{
	int d = SP.def[id], pid, pk, i, o[2], e[2] = {-1, -1}, sgs[2];
	uint64_t s, v[2];

	if (d < 0 || SP.dead[d] || int_width(V.id[id].type) != 2 * w ||
	    (strcmp(V.in[d].op, "OpShiftRightLogical") &&
	    strcmp(V.in[d].op, "OpShiftRightArithmetic")) ||
	    !const_of(res_of(idx(V.in[d].opnd[2])), &s) ||
	    s < (uint64_t)w || s >= (uint64_t)(2 * w))
		return 0;
	pid = res_of(idx(V.in[d].opnd[1]));
	pk = SP.def[pid];
	if (pk < 0 || SP.dead[pk] || strcmp(V.in[pk].op, "OpIMul"))
		return 0;
	for (i = 0; i < 2; i++) {
		int ek;
		o[i] = res_of(idx(V.in[pk].opnd[1 + i]));
		sgs[i] = -1;	/* a constant: either, if it fits */
		if (const_of(o[i], &v[i]))
			continue;
		ek = SP.def[o[i]];
		if (ek < 0 || SP.dead[ek] || (strcmp(V.in[ek].op, "OpUConvert") &&
		    strcmp(V.in[ek].op, "OpSConvert")))
			return 0;
		e[i] = o[i];
		o[i] = res_of(idx(V.in[ek].opnd[1]));
		if (V.id[o[i]].type != t)
			return 0;
		sgs[i] = V.in[ek].op[2] == 'S';
	}
	if (sgs[0] < 0 && sgs[1] < 0)
		return 0;	/* constants: folded before */
	*sg = sgs[0] >= 0 ? sgs[0] : sgs[1];
	if (sgs[0] >= 0 && sgs[1] >= 0 && sgs[0] != sgs[1])
		return 0;	/* one signed, one unsigned: no such instruction */
	if (!*sg && T(t)->sign)
		return 0;	/* OpUMulExtended's members are unsigned */
	for (i = 0; i < 2; i++)
		if (sgs[i] < 0) {
			uint64_t lo = v[i] & wmask(w);
			if (*sg ? ((uint64_t)sx64(lo, w) & wmask(2 * w)) != v[i] :
			    v[i] != lo)
				return 0;	/* not a w-bit value made wide */
			if ((o[i] = const_id(t, lo)) < 0)
				return 0;
		}
	*x = o[0];
	*y = o[1];
	*sh = (int)s - w;
	*ar = !strcmp(V.in[d].op, "OpShiftRightArithmetic");
	*extra = only_use_cost(id);
	if (SP.uses[id] == 1) {
		*extra += only_use_cost(pid);
		if (SP.uses[pid] == 1)
			for (i = 0; i < 2; i++)
				if (e[i] >= 0 && (i == 0 || e[1] != e[0]))
					*extra += only_use_cost(e[i]);
	}
	return 1;
}

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
	/*
	 * a select of a value and a constant is a min or a max of the two
	 * when the compare decides where they cross (scale != 0 ? scale : 1
	 * is max(scale, 1) on unsigned values), whatever it compares: the
	 * four are offered and SAT keeps one that holds; and when the value
	 * is a min or a max of x and a constant itself, the two are one
	 * clamp of x (x < 0 ? 0 : min(x, 255) is clamp(x, 0, 255))
	 */
	if (!strcmp(I->op, "OpSelect") && nargs == 3 && V.glsl >= 0 && w > 1) {
		static const char *const mm[] = {"UMax", "UMin", "SMax", "SMin"};
		int cd = SP.def[arg[0]], q, j, both = 0;
		double cx = cd >= 0 && !SP.dead[cd] && SP.uses[arg[0]] == 1 ?
		    ins_cost_k(cd) : 0;
		uint64_t kv, kw;
		if (cd >= 0 && !SP.dead[cd] && V.in[cd].nop >= 3) {
			int x = res_of(idx(V.in[cd].opnd[1]));
			int y = res_of(idx(V.in[cd].opnd[2]));
			both = (arg[1] == x && arg[2] == y) ||
			    (arg[1] == y && arg[2] == x);
		}
		for (q = 1; q <= 2 && !both; q++) {
			int v = arg[3 - q], dv = SP.def[v];
			if (!const_of(arg[q], &kv) || const_of(v, &kw))
				continue;
			for (j = 0; j < 4 && n < max; j++) {
				n += ins_cand(&cs[n], "OpExtInst", mm[j], rt, v, arg[q],
				    -1, j & 1 ? "a select of a value and a constant "
				    "is their min" : "a select of a value and a constant "
				    "is their max");
				cs[n - 1].extra = cx;
			}
			/* the value: min(x, hi) or max(x, lo), a constant on one
			 * side; the select's constant the other bound */
			if (dv >= 0 && !SP.dead[dv] && !strcmp(V.in[dv].op,
			    "OpExtInst") && V.in[dv].nop == 5 && n < max) {
				const char *e2 = V.in[dv].opnd[2];
				int a0 = res_of(idx(V.in[dv].opnd[3]));
				int a1 = res_of(idx(V.in[dv].opnd[4]));
				int sg = e2[0] == 'S', mn = !strcmp(e2 + 1, "Min");
				int kid = const_of(a1, &kw) ? a1 : const_of(a0, &kw) ?
				    a0 : -1, xv = kid == a1 ? a0 : a1;
				int lo = mn ? arg[q] : kid, hi = mn ? kid : arg[q];
				uint64_t lv, hv;
				if (kid < 0 || const_of(xv, &lv) || (e2[0] != 'S' &&
				    e2[0] != 'U') || (strcmp(e2 + 1, "Min") &&
				    strcmp(e2 + 1, "Max")))
					continue;
				const_of(lo, &lv);
				const_of(hi, &hv);
				if (sg ? sx64(lv, w) > sx64(hv, w) : lv > hv)
					continue;
				n += ins_cand(&cs[n], "OpExtInst", sg ? "SClamp" :
				    "UClamp", rt, xv, lo, hi, "a select of a constant "
				    "and a min or max of x is one clamp of x");
				cs[n - 1].extra = cx + (SP.uses[v] == 1 ?
				    ins_cost_k(dv) : 0);
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
	/* (x << j) >> k, k >= j: the field k - j .. w - j - 1 of x, sign- or
	 * zero-extended by the right shift */
	if ((!strcmp(I->op, "OpShiftRightArithmetic") ||
	    !strcmp(I->op, "OpShiftRightLogical")) && w > 1 && n < max) {
		int d = SP.def[arg[0]];
		uint64_t j, kk;
		if (d >= 0 && !SP.dead[d] && !strcmp(V.in[d].op,
		    "OpShiftLeftLogical") && const_of(arg[1], &kk) &&
		    const_of(res_of(idx(V.in[d].opnd[2])), &j) && j <= kk &&
		    kk < (uint64_t)w && j > 0) {
			n += ins_cand(&cs[n], !strcmp(I->op, "OpShiftRightArithmetic") ?
			    "OpBitFieldSExtract" : "OpBitFieldUExtract", NULL, rt,
			    res_of(idx(V.in[d].opnd[1])), const_id(rt, kk - j),
			    const_id(rt, (uint64_t)w - kk),
			    "two shifts by constants are one bitfield extract");
			cs[n - 1].extra = SP.uses[arg[0]] == 1 ? ins_cost_k(d) : 0;
		}
	}
	/* the bits of one word in reverse order, however many swaps make it:
	 * one bit reverse (the random runs and the proof decide) */
	if ((!strcmp(I->op, "OpBitwiseOr") || !strcmp(I->op, "OpBitwiseXor") ||
	    !strcmp(I->op, "OpIAdd")) && w > 1 && n < max) {
		int leaf = -1, i, ok = 1;
		for (i = 0; i < 2 && ok; i++)
			ok = one_leaf(arg[i], 15, &leaf);	/* the proof's 16 levels */
		if (ok && leaf >= 0 && V.id[leaf].type == rt) {
			n += ins_cand(&cs[n], "OpBitReverse", NULL, rt, leaf, -1, -1,
			    "the bits of one word reversed are one bit reverse");
			cs[n - 1].extra = aff_dead_cost(I->res, leaf, rt);
		}
	}
	/* (base & ~m) | field, m the bits p..p+n-1 and the field (ins << p) &
	 * m or (ins & (2^n - 1)) << p (ins itself for p = 0): one bitfield
	 * insert of ins's low n bits */
	if ((!strcmp(I->op, "OpBitwiseOr") || !strcmp(I->op, "OpBitwiseXor") ||
	    !strcmp(I->op, "OpIAdd")) && w > 1 && w <= 64 && n < max) {
		int i;
		for (i = 0; i < 2 && n < max; i++) {
			int base, ins, p, nb;
			double extra;
			if (bfi_parts(arg[i], arg[1 - i], w, &base, &ins, &p, &nb,
			    &extra)) {
				n += ins_cand(&cs[n], "OpBitFieldInsert", NULL, rt, base,
				    ins, const_id(rt, (uint64_t)p),
				    "a field put into a word is one bitfield insert");
				cs[n - 1].a[3] = const_id(rt, (uint64_t)nb);
				cs[n - 1].na = cs[n - 1].a[2] >= 0 && cs[n - 1].a[3] >= 0 ?
				    4 : 0;
				cs[n - 1].extra = extra;
				if (cs[n - 1].na == 0)
					n--;	/* no constant id for p or n */
				break;
			}
		}
	}
	/* the high word of a product made wide: Op[US]MulExtended's second
	 * member, shifted by what the shift takes past the low word */
	if ((!strcmp(I->op, "OpUConvert") || !strcmp(I->op, "OpSConvert")) &&
	    T(rt)->k == S_INT && w >= 8 && w <= 32 && n < max) {
		int x, y, sg, sh, ar;
		double extra;
		if (mulx_parts(arg[0], rt, w, &x, &y, &sg, &sh, &ar, &extra) &&
		    (sh == 0 || const_id(rt, (uint64_t)sh) >= 0)) {
			const char *mo = sg ? "OpSMulExtended" : "OpUMulExtended";
			n += ins_cand(&cs[n], mo, NULL, rt, x, y, -1,
			    "a wide product's high word is one multiply extended");
			cs[n - 1].mulx = 1;
			cs[n - 1].msh = sh;
			cs[n - 1].mash = ar;
			cs[n - 1].cost = ins_cost(mo, NULL, rt) + (sh ? ins_cost(ar ?
			    "OpShiftRightArithmetic" : "OpShiftRightLogical", NULL,
			    rt) : 0);
			cs[n - 1].extra = extra;
		}
	}
	(void)w;
	return n;
}

/* the type struct { t, t } Op[US]MulExtended returns, declared before
 * the first function */
static int
mulx_type(int t)
{
	char nm[48];
	int i;

	for (i = 0; i < SP.nmxty; i++)
		if (T(SP.mxty[i])->m[0] == t)
			return SP.mxty[i];
	if (SP.nmxty == 4)
		return -1;
	snprintf(nm, sizeof nm, "%%ucas_mulx_%s", V.id[t].tok + 1);
	i = idx(nm);
	if (V.id[i].kind == K_TYPE)
		return i;	/* the module has it: simplified before */
	V.id[i].kind = K_TYPE;
	memset(&V.id[i].t, 0, sizeof V.id[i].t);
	V.id[i].t.k = S_STRUCT;
	V.id[i].t.m[0] = V.id[i].t.m[1] = t;
	V.id[i].t.nm = 2;
	SP.mxty[SP.nmxty++] = i;
	return i;
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
	if (c->mulx) {
		char hw[200];
		snprintf(hw, sizeof hw, "OpCompositeExtract %s (%s %s %s) 1",
		    I->opnd[0], c->op, V.id[c->a[0]].tok, V.id[c->a[1]].tok);
		if (c->msh)
			snprintf(b, n, "%s %s (%s) %d", c->mash ?
			    "OpShiftRightArithmetic" : "OpShiftRightLogical",
			    I->opnd[0], hw, c->msh);
		else
			snprintf(b, n, "%s", hw);
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
		"two shifts by constants are one",
		"two shifts by constants are one bitfield extract", NULL
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
	if (c->mulx) {
		/* the multiply extended and its high word before k; k is the
		 * extract, or the shift of it */
		int st = mulx_type(rt), m, e = -1;
		if (st < 0)
			return;
		m = new_ins(k, c->op, st, c->a[0], c->a[1]);
		SP.nval++;
		if (c->msh) {
			struct sins *N;
			e = new_ins(k, "OpCompositeExtract", rt, m, -1);
			N = &V.in[SP.def[e]];
			N->opnd[N->nop++] = strdup("1");
			snprintf(N->text, 200, "%s = OpCompositeExtract %s %s 1",
			    V.id[e].tok, N->opnd[0], N->opnd[1]);
		}
		I = &V.in[k];
		for (i = 1; i < I->nop; i++)
			free(I->opnd[i]);
		I->nop = 1;
		if (c->msh) {
			snprintf(I->op, sizeof I->op, "%s", c->mash ?
			    "OpShiftRightArithmetic" : "OpShiftRightLogical");
			I->opnd[I->nop++] = strdup(V.id[e].tok);
			I->opnd[I->nop++] = strdup(V.id[const_id(rt,
			    (uint64_t)c->msh)].tok);
		} else {
			snprintf(I->op, sizeof I->op, "OpCompositeExtract");
			I->opnd[I->nop++] = strdup(V.id[m].tok);
			I->opnd[I->nop++] = strdup("1");
		}
		SP.rw[k] = 1;
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

/*
 * The module declares SPV_KHR_integer_dot_product: its capability, the
 * extension, or GLSL's GL_EXT_integer_dot_product (OpSourceExtension,
 * which a source that enables it and calls none of it still has).  A
 * dot form needs it, so a module without it is not given one.
 */
static int
dot_declared(void)
{
	int k;

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (I->nop < 1)
			continue;
		if (!strcmp(I->op, "OpCapability") && (!strcmp(I->opnd[0],
		    "DotProduct") || !strcmp(I->opnd[0], "DotProductKHR")))
			return 1;
		if (!strcmp(I->op, "OpExtension") && strstr(I->opnd[0],
		    "SPV_KHR_integer_dot_product"))
			return 1;
		if (!strcmp(I->op, "OpSourceExtension") && strstr(I->opnd[0],
		    "GL_EXT_integer_dot_product"))
			return 1;
	}
	return 0;
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
	int rdsel;		/* also read compares, selects, bools */
	int tybool;		/* the module's bool type id, -1 */
};

static int dotcaps;		/* the module needs the dot capabilities */
static struct {
	int n;			/* proved where the module lacks the extension */
	double gain;
} dotund;

static ty_t
spv_ty(int t)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_INT || T(t)->w > 64)
		return 0;
	return TY(T(t)->sign ? K_SINT : K_UINT, T(t)->w);
}

/* t as a ucas type: an integer, or a bool when the reader takes them */
static ty_t
spv_ty_c(const struct spvdot *c, int t)
{
	if (c->rdsel && t >= 0 && V.id[t].kind == K_TYPE && T(t)->k == S_BOOL)
		return TY_BOOL;
	return spv_ty(t);
}

static nid
spv_leaf(struct spvdot *c, int id)
{
	ty_t t = spv_ty_c(c, V.id[id].type);
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
	t = spv_ty_c(c, V.id[id].type);
	if (V.id[id].kind == K_CONST && !V.id[id].spec && t == TY_BOOL)
		return c->memo[id] = mk_const_i(TY_BOOL, V.id[id].c[0] != 0);
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
	} else if (IS("OpBitFieldSExtract") && const_of(idx(OPN(1)), &kv) &&
	    kv < 64 && const_of(idx(OPN(2)), &kv)) {
		/* the field moved to the top and shifted back, arithmetic */
		uint64_t off, cnt;
		const_of(idx(OPN(1)), &off);
		const_of(idx(OPN(2)), &cnt);
		if (cnt == 0 || off + cnt > (uint64_t)TYW(t))
			return c->memo[id] = spv_leaf(c, id);
		x[0] = mk_cast(st, spv_node(c, idx(OPN(0)), depth + 1));
		if (off + cnt < (uint64_t)TYW(t))
			x[0] = mk_bin(O_SHL, x[0], mk_const_i(st, (int64_t)(TYW(t) -
			    off - cnt)));
		r = cnt < (uint64_t)TYW(t) ? mk_bin(O_SHR, x[0], mk_const_i(st,
		    (int64_t)(TYW(t) - cnt))) : x[0];
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
	} else if (c->rdsel && t == TY_BOOL && (IS("OpIEqual") ||
	    IS("OpINotEqual") || IS("OpULessThan") || IS("OpULessThanEqual") ||
	    IS("OpUGreaterThan") || IS("OpUGreaterThanEqual") ||
	    IS("OpSLessThan") || IS("OpSLessThanEqual") ||
	    IS("OpSGreaterThan") || IS("OpSGreaterThanEqual"))) {
		/* a compare reads its operands signed or unsigned, as it says */
		int a0 = res_of(idx(OPN(0)));
		ty_t at = spv_ty(V.id[a0].type), ct;
		const char *o = I->op + 2;
		int op = IS("OpIEqual") ? O_EQ : IS("OpINotEqual") ? O_NE :
		    strstr(o, "LessThanEqual") ? O_LE : strstr(o, "LessThan") ?
		    O_LT : strstr(o, "GreaterThanEqual") ? O_GE : O_GT;
		if (!at)
			return c->memo[id] = spv_leaf(c, id);
		ct = op == O_EQ || op == O_NE ? at : TY(o[0] == 'S' ? K_SINT :
		    K_UINT, TYW(at));
		x[0] = mk_cast(ct, spv_node(c, a0, depth + 1));
		x[1] = mk_cast(ct, spv_node(c, idx(OPN(1)), depth + 1));
		r = mk_bin(op, x[0], x[1]);
	} else if (c->rdsel && IS("OpSelect") && spv_ty_c(c,
	    V.id[res_of(idx(OPN(0)))].type) == TY_BOOL) {
		x[0] = spv_node(c, idx(OPN(0)), depth + 1);
		x[1] = mk_cast(t, spv_node(c, idx(OPN(1)), depth + 1));
		x[2] = mk_cast(t, spv_node(c, idx(OPN(2)), depth + 1));
		r = mk_sel(x[0], x[1], x[2]);
	} else if (c->rdsel && t == TY_BOOL && (IS("OpLogicalAnd") ||
	    IS("OpLogicalOr") || IS("OpLogicalNot") || IS("OpLogicalEqual") ||
	    IS("OpLogicalNotEqual"))) {
		x[0] = spv_node(c, idx(OPN(0)), depth + 1);
		if (IS("OpLogicalNot"))
			r = mk_un(O_LNOT, x[0]);
		else {
			x[1] = spv_node(c, idx(OPN(1)), depth + 1);
			r = mk_bin(IS("OpLogicalAnd") ? O_LAND : IS("OpLogicalOr") ?
			    O_LOR : IS("OpLogicalEqual") ? O_EQ : O_NE, x[0], x[1]);
		}
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
	/* the operands are copied first: they may point into V.id, which
	 * idx() moves when it grows */
	memset(&N, 0, sizeof N);
	for (i = 0; i < n; i++)
		N.opnd[1 + i] = strdup(opnd[i]);
	snprintf(nm, sizeof nm, "%%ucas_d%d", V.nin);
	nid_ = idx(nm);
	snprintf(N.op, sizeof N.op, "%s", op);
	N.res = nid_;
	N.opnd[0] = strdup(V.id[rt].tok);
	N.nop = 1 + n;
	N.text = xmalloc(400);
	at = snprintf(N.text, 400, "%s = %s %s", nm, op, V.id[rt].tok);
	for (i = 0; i < n && at < 380; i++)
		at += snprintf(N.text + at, 400 - (size_t)at, " %s", N.opnd[1 + i]);
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
	if (t == TY_BOOL && c->rdsel)
		return c->tybool;
	if (TYW(t) != 32 || (TYK(t) != K_UINT && TYK(t) != K_SINT))
		return -1;
	return c->ty32[TYK(t) == K_SINT];
}

/* the SPIR-V id computing node n (the ucas form of the rewrite) */
static int
dot_emit(struct spvdot *c, nid n)
{
	int i, op = N.op[n], rt, a, b, cc, r = -1;
	const char *o[4];

	for (i = 0; i < c->nem; i++)
		if (c->em[i].n == n)
			return c->em[i].id;
	if (c->fail)
		return -1;
	if (op == O_VAR) {
		int leaf = c->leaf[N.ki[n]];
		rt = ty_id(c, N.ty[n]);
		if (rt < 0 || spv_ty_c(c, V.id[leaf].type) != N.ty[n]) {
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
		r = const_id(rt, (uint64_t)N.ki[n] & (N.ty[n] == TY_BOOL ? 1 :
		    0xffffffffULL));
		if (r < 0)
			c->fail = 1;
		goto done;
	}
	if (op == O_CAST && (N.ty[n] == TY_BOOL) != (N.ty[N.a[n]] == TY_BOOL)) {
		/* int to bool: its low bit; bool to int: 1 or 0 */
		int one, zero, ut = ty_id(c, N.ty[n] == TY_BOOL ? N.ty[N.a[n]] :
		    N.ty[n]);
		a = dot_emit(c, N.a[n]);
		if (a < 0 || ut < 0) {
			c->fail = 1;
			return -1;
		}
		one = const_id(ut, 1);
		zero = const_id(ut, 0);
		if (one < 0 || zero < 0) {
			c->fail = 1;
			return -1;
		}
		if (N.ty[n] == TY_BOOL) {
			o[0] = V.id[a].tok;
			o[1] = V.id[one].tok;
			b = dot_ins(c, "OpBitwiseAnd", ut, o, 2);
			if (b < 0)
				return -1;
			o[0] = V.id[b].tok;
			o[1] = V.id[zero].tok;
			r = dot_ins(c, "OpINotEqual", rt, o, 2);
		} else {
			o[0] = V.id[a].tok;
			o[1] = V.id[one].tok;
			o[2] = V.id[zero].tok;
			r = dot_ins(c, "OpSelect", rt, o, 3);
		}
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
		case O_LAND: sop = "OpLogicalAnd"; break;
		case O_LOR: sop = "OpLogicalOr"; break;
		case O_LNOT: sop = "OpLogicalNot"; break;
		case O_EQ: case O_NE: case O_LT: case O_LE: case O_GT: case O_GE: {
			/* by the operands' signedness; bools compare logically */
			static const char *const u[] = {"OpIEqual", "OpINotEqual",
			    "OpULessThan", "OpULessThanEqual", "OpUGreaterThan",
			    "OpUGreaterThanEqual"};
			static const char *const sg[] = {"OpIEqual", "OpINotEqual",
			    "OpSLessThan", "OpSLessThanEqual", "OpSGreaterThan",
			    "OpSGreaterThanEqual"};
			ty_t at = N.ty[N.a[n]];
			if (at == TY_BOOL)
				sop = op == O_EQ ? "OpLogicalEqual" : op == O_NE ?
				    "OpLogicalNotEqual" : NULL;
			else
				sop = (TYK(at) == K_SINT ? sg : u)[op - O_EQ];
			break;
		}
		case O_SEL: sop = "OpSelect"; break;
		case O_DIV: sop = TYK(N.ty[n]) == K_SINT ? "OpSDiv" : "OpUDiv";
			break;
		case O_MOD: sop = TYK(N.ty[n]) == K_SINT ? "OpSRem" : "OpUMod";
			break;
		}
		if (!sop) {
			c->fail = 1;
			return -1;
		}
		a = dot_emit(c, N.a[n]);
		if (a < 0)
			return -1;
		b = cc = a;
		if (opinfo[op].arity >= 2 && (b = dot_emit(c, N.b[n])) < 0)
			return -1;
		if (opinfo[op].arity == 3 && (cc = dot_emit(c, N.c[n])) < 0)
			return -1;
		/* the names are read once every operand is emitted: an
		 * operand's dot_ins grows V.id, which moves them */
		o[0] = V.id[a].tok;
		o[1] = V.id[b].tok;
		o[2] = V.id[cc].tok;
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
	int kind;		/* idrw: 0 a division, 1 a divisibility test,
				 * 2 a map of one value */
};

static int
dot_pass(struct sbuf *rep, struct dotlog *log, int maxlog)
{
	int k, n = 0, i, ty32[2];
	int *sumuse = xcalloc((size_t)(V.nid + 1024), sizeof *sumuse);
	int nmodel = 0, decl = dot_declared();

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
		fuse_domain(&c.f);	/* the cover's ranges are the sum's */
		dot_search(&c.f, 1, &r);
		fuse_domain(NULL);
		if (r.status != DOT_PROVED || !dot_translation_ok(&c, root,
		    r.body)) {
			dotres_free(&r);
			continue;
		}
		if (!decl) {
			dotund.n++;
			dotund.gain += r.before - r.after;
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
		I = &V.in[k];	/* dot_emit() may have moved V.in */
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
#include "irw.h"
#include "numsearch.h"

/*
 * Divisions and remainders by a constant whose dividend the assumptions
 * bound (idrw.c): the instruction and what its dividend is computed from
 * are read into a ucas function as for the dot products, irw_cands
 * derives the multiply and the shift for the proved bound, and it is
 * written back before the instruction when the active model prices it
 * lower.  The derivation is the proof; the translation is checked
 * against the interpreter on 64 samples.
 */
/* id was read into c as an instruction, not as a leaf or a constant */
static int
read_as_ins(const struct spvdot *c, int id)
{
	int j;

	if (!c->memo[id] || SP.def[id] < 0 || V.id[id].kind == K_CONST)
		return 0;
	for (j = 0; j < c->nleaf; j++)
		if (c->leaf[j] == id)
			return 0;
	return 1;
}

static void
mark_dag(nid e, uint8_t *in)
{
	uint32_t j;

	if (e == 0 || in[e])
		return;
	in[e] = 1;
	if (N.op[e] == O_CONST || N.op[e] == O_VAR || N.op[e] == O_UNDEF)
		return;
	if (N.op[e] == O_TABLE) {
		for (j = 0; j < N.b[e]; j++)
			mark_dag(N.lst[N.a[e] + j], in);
		return;
	}
	mark_dag(N.a[e], in);
	mark_dag(N.b[e], in);
	mark_dag(N.c[e], in);
}

/*
 * What the rewrite to e drops (the instructions read into c, but the
 * root, whose value e does not use) goes with the root when nothing else
 * reads it; one that something else reads stays, and its price is
 * returned, to be added to e's.  What e keeps (a map's key) is priced on
 * both sides already.
 */
static double
drops_kept(const struct spvdot *c, int root, nid e)
{
	int *in = xcalloc((size_t)(V.nid + 1024), sizeof *in), id, i;
	uint8_t *keep = xcalloc(N.n + 1, 1);
	double stays = 0;

	mark_dag(e, keep);
	for (id = 0; id < V.nid; id++)
		if (read_as_ins(c, id)) {
			const struct sins *I = &V.in[SP.def[id]];
			for (i = first_arg(I); i < I->nop; i++)
				if (I->opnd[i][0] == '%')
					in[res_of(idx(I->opnd[i]))]++;
		}
	for (id = 0; id < V.nid; id++)
		if (id != root && read_as_ins(c, id) && !keep[c->memo[id]] &&
		    SP.uses[id] != in[id])
			stays += ins_cost_k(SP.def[id]);
	free(in);
	free(keep);
	return stays;
}

/* the module's bool type id, -1 */
static int
bool_type_id(void)
{
	int i;

	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_TYPE && V.id[i].t.k == S_BOOL)
			return i;
	return -1;
}

/* id is an OpUMod, OpSRem or OpSMod */
static int
is_rem(int id)
{
	int d = SP.def[id];

	return d >= 0 && (!strcmp(V.in[d].op, "OpUMod") ||
	    !strcmp(V.in[d].op, "OpSRem") || !strcmp(V.in[d].op, "OpSMod"));
}

/*
 * Instruction k as a root for the identities beyond the divisions: a
 * divisibility test (1), an OpIEqual or OpINotEqual of a remainder by a
 * constant (used nowhere else) and a constant; or the top of a map of
 * one value (2), an OpSelect with an integer result, an OpLogicalOr or
 * OpLogicalAnd, that no other such instruction reads.  0 neither.
 */
/*
 * The values a divisor takes when it is a few constants: a constant, a
 * select of such values (either arm), a phi at a selection merge of them
 * (any incoming), or a value the request assumes in a range of at most 8
 * (`width in [2, 4]`).  Up to 8 values; 0 when it is none of those.
 */
/* a w-bit value read signed */
static int64_t
few_sext(uint64_t v, int w)
{
	return w >= 64 ? (int64_t)v : (int64_t)(v << (64 - w)) >> (64 - w);
}

static int
few_vals(int id, int w, int64_t *v, int *n, int depth)
{
	int d = SP.def[res_of(id)], i, j;
	int64_t lo, hi;
	uint64_t c;

	id = res_of(id);
	if (depth > 4)
		return 0;
	if (const_of(id, &c)) {
		for (j = 0; j < *n && v[j] != few_sext(c, w); j++)
			;
		if (j == *n) {
			if (*n == 8)
				return 0;
			v[(*n)++] = few_sext(c, w);
		}
		return 1;
	}
	if (d >= 0 && !SP.dead[d] && !strcmp(V.in[d].op, "OpSelect") &&
	    V.in[d].nop == 4)
		return few_vals(idx(V.in[d].opnd[2]), w, v, n, depth + 1) &&
		    few_vals(idx(V.in[d].opnd[3]), w, v, n, depth + 1);
	if (d >= 0 && fwd_phi(d)) {
		for (i = 1; i + 1 < V.in[d].nop; i += 2)
			if (!few_vals(idx(V.in[d].opnd[i]), w, v, n, depth + 1))
				return 0;
		return 1;
	}
	if (assumed(id, &lo, &hi) && hi - lo < 8) {
		for (; lo <= hi; lo++) {
			for (j = 0; j < *n && v[j] != lo; j++)
				;
			if (j == *n) {
				if (*n == 8)
					return 0;
				v[(*n)++] = lo;
			}
		}
		return 1;
	}
	return 0;
}

/* where few_vals found the values, for the report */
static const char *
few_why(int id)
{
	int d = SP.def[res_of(id)];
	int64_t lo, hi;

	if (d >= 0 && !strcmp(V.in[d].op, "OpSelect"))
		return "a select of constants";
	if (d >= 0 && !strcmp(V.in[d].op, "OpPhi"))
		return "the constants its phi merges";
	if (assumed(res_of(id), &lo, &hi))
		return "the request's assumption";
	return "constants";
}

/*
 * x op d as a division by a constant per value of d: a select's arms by
 * its own condition, otherwise a chain of compares of d with its values
 * (the last value takes what the others do not, 0 among them: a division
 * by 0 is undefined anyway).
 */
static nid
div_cases(struct spvdot *c, int id, nid x, int op, ty_t vt, int w, int depth)
{
	int d = SP.def[res_of(id)], n = 0, i, z;
	int64_t v[8];
	uint64_t cv;
	nid r = 0, dl;

	id = res_of(id);
	if (const_of(id, &cv))
		return few_sext(cv, w) == 0 ? 0 : mk_bin(op, x, mk_const_i(vt,
		    few_sext(cv, w)));
	if (d >= 0 && !SP.dead[d] && !strcmp(V.in[d].op, "OpSelect") &&
	    V.in[d].nop == 4 && depth < 4 && c->tybool >= 0) {
		nid a = div_cases(c, idx(V.in[d].opnd[2]), x, op, vt, w, depth + 1);
		nid b = div_cases(c, idx(V.in[d].opnd[3]), x, op, vt, w, depth + 1);
		if (!a || !b)
			return 0;
		return mk_sel(spv_node(c, idx(V.in[d].opnd[1]), 0), a, b);
	}
	if (!few_vals(id, w, v, &n, 0) || n < 2)
		return 0;
	dl = mk_cast(vt, spv_node(c, id, 0));
	/* an unsigned remainder by any power of two among them is one
	 * case, x & (d - 1): the last, without a compare */
	if (op == O_MOD && TYK(vt) == K_UINT)
		for (i = 0; i < n; i++)
			if (v[i] > 1 && !(v[i] & (v[i] - 1)))
				r = mk_bin(O_AND, x, mk_bin(O_SUB, dl, mk_const_i(vt,
				    1)));
	/* else the last case is one that is not 0 */
	for (z = n - 1; z > 0 && v[z] == 0; z--)
		;
	if (!r)
		r = mk_bin(op, x, mk_const_i(vt, v[z]));
	else
		z = -1;
	for (i = n - 1; i >= 0; i--)
		if (i != z && v[i] != 0 && !(r && z < 0 && v[i] > 1 &&
		    !(v[i] & (v[i] - 1))))
			r = mk_sel(mk_bin(O_EQ, dl, mk_const_i(vt, v[i])),
			    mk_bin(op, x, mk_const_i(vt, v[i])), r);
	return r;
}

/* the factors of a product id (through multiplies and shifts left by a
 * constant), its constants left out */
static void
spv_factors(int id, int *f, int *nf, int depth)
{
	int d = SP.def[id];
	uint64_t v;

	if (*nf >= 8 || depth > 8)
		return;
	if (d >= 0 && !SP.dead[d] && pure_ins(d) && V.in[d].nop == 3 &&
	    (!strcmp(V.in[d].op, "OpIMul") ||
	    (!strcmp(V.in[d].op, "OpShiftLeftLogical") &&
	    const_of(res_of(idx(V.in[d].opnd[2])), &v)))) {
		spv_factors(res_of(idx(V.in[d].opnd[1])), f, nf, depth + 1);
		if (!strcmp(V.in[d].op, "OpIMul"))
			spv_factors(res_of(idx(V.in[d].opnd[2])), f, nf, depth + 1);
		return;
	}
	if (!const_of(id, &v))
		f[(*nf)++] = id;
}

/* the terms of the sum id (through adds and subtracts) */
static void
spv_terms(int id, int *t, int *nt, int depth)
{
	int d = SP.def[id];

	if (*nt >= 16)
		return;
	if (depth < 8 && d >= 0 && !SP.dead[d] && pure_ins(d) &&
	    (depth == 0 || SP.uses[id] == 1) &&
	    V.in[d].nop == 3 && (!strcmp(V.in[d].op, "OpIAdd") ||
	    !strcmp(V.in[d].op, "OpISub"))) {
		spv_terms(res_of(idx(V.in[d].opnd[1])), t, nt, depth + 1);
		spv_terms(res_of(idx(V.in[d].opnd[2])), t, nt, depth + 1);
		return;
	}
	t[(*nt)++] = id;
}

/* two terms of the sum id share a factor that is not a constant */
static int
sum_shares_factor(int id)
{
	int t[16], nt = 0, f[16][8], nf[16], i, j, a, b;

	spv_terms(id, t, &nt, 0);
	for (i = 0; i < nt; i++) {
		nf[i] = 0;
		spv_factors(t[i], f[i], &nf[i], 0);
	}
	for (i = 0; i < nt; i++)
		for (j = i + 1; j < nt; j++)
			for (a = 0; a < nf[i]; a++)
				for (b = 0; b < nf[j]; b++)
					if (f[i][a] == f[j][b] && (nf[i] > 1 ||
					    nf[j] > 1))
						return 1;
	return 0;
}

static int
idrw_root_kind(int k, const char *partof)
{
	const struct sins *I = &V.in[k];
	int i;

	if ((!strcmp(I->op, "OpIEqual") || !strcmp(I->op, "OpINotEqual")) &&
	    I->nop == 3)
		for (i = 1; i < 3; i++) {
			int a = res_of(idx(I->opnd[i])), d = SP.def[a];
			int o = res_of(idx(I->opnd[3 - i]));
			uint64_t dv, kv;
			if (!is_rem(a) || SP.dead[d] || SP.uses[a] != 1 ||
			    V.in[d].nop != 3 || !const_of(idx(V.in[d].opnd[2]),
			    &dv) || dv < 3 || !const_of(o, &kv))
				continue;
			return 1;
		}
	if ((!strcmp(I->op, "OpSelect") && int_width(idx(I->opnd[0])) == 32) ||
	    !strcmp(I->op, "OpLogicalOr") || !strcmp(I->op, "OpLogicalAnd"))
		return partof[I->res] == 1 ? 0 : 2;
	/* a sum no other add or subtract reads, two of whose terms have a
	 * factor in common: it may be taken out (ifx.c) */
	if ((!strcmp(I->op, "OpIAdd") || !strcmp(I->op, "OpISub")) &&
	    int_width(idx(I->opnd[0])) == 32 && partof[I->res] != 3 &&
	    sum_shares_factor(I->res))
		return 3;
	return 0;
}

static int
idrw_pass(struct sbuf *rep, struct sbuf *rep2, struct sbuf *rep3,
    struct sbuf *rep4, struct dotlog *log, int maxlog)
{
	const struct model *m = model_active();
	int k, n = 0, i, ty32[2], tybool = bool_type_id();
	char *partof;		/* a value another select or logic op reads */

	if (!dot_type_ids(ty32))
		return 0;
	partof = xcalloc((size_t)(V.nid + 1024), 1);
	for (k = 0; k < SP.orig_nin; k++) {
		const struct sins *I = &V.in[k];
		int same = !strcmp(I->op, "OpSelect") ? 2 :
		    !strcmp(I->op, "OpLogicalOr") || !strcmp(I->op, "OpLogicalAnd") ?
		    1 : 0;
		if (SP.dead[k])
			continue;
		if (!same) {
			/* a sum's terms that are sums read nowhere else: read
			 * with it (one read by several is a sum of its own) */
			if (pure_ins(k) && (!strcmp(I->op, "OpIAdd") ||
			    !strcmp(I->op, "OpISub")))
				for (i = 1; i < I->nop; i++) {
					int a;
					if (I->opnd[i][0] != '%')
						continue;
					a = res_of(idx(I->opnd[i]));
					if (!partof[a] && SP.uses[a] == 1)
						partof[a] = 3;
				}
			continue;
		}
		/* a select's arms; a logic op's operands */
		for (i = same == 2 ? 2 : 1; i < I->nop; i++)
			if (I->opnd[i][0] == '%')
				partof[res_of(idx(I->opnd[i]))] = 1;
	}
	for (k = 0; k < SP.orig_nin && n < maxlog && (double)clock() /
	    CLOCKS_PER_SEC < SP.deadline; k++) {
		struct spvdot c;
		const struct sins *I = &V.in[k];
		struct irw *r;
		int root, fin, nr, best = -1, base, uns, div, nin0, nk0, kind;
		int smod = 0, remid = -1;
		double kept = 0;
		uint64_t dv;
		ty_t t, vt;
		nid body;
		float before, after = 0;
		char sl[160];
		const char *s, *e;

		if (SP.dead[k] || !pure_ins(k))
			continue;
		root = I->res;
		uns = !strcmp(I->op, "OpUDiv") || !strcmp(I->op, "OpUMod");
		div = !strcmp(I->op, "OpUDiv") || !strcmp(I->op, "OpSDiv");
		memset(&c, 0, sizeof c);
		c.memo = xcalloc((size_t)(V.nid + 1024), sizeof *c.memo);
		c.k = k;
		c.ty32[0] = ty32[0];
		c.ty32[1] = ty32[1];
		c.tybool = tybool;
		c.f.anon = 1;
		if (uns || !strcmp(I->op, "OpSDiv") || !strcmp(I->op, "OpSRem") ||
		    !strcmp(I->op, "OpSMod")) {
			kind = 0;
			base = first_arg(I);
			if (I->nop >= base + 2 && int_width(V.id[root].type) == 32 &&
			    !const_of(idx(I->opnd[base + 1]), &dv) &&
			    strcmp(I->op, "OpSMod") && tybool >= 0) {
				/* a divisor of a few constants: one case each */
				nid x0, cs;
				t = spv_ty(V.id[root].type);
				vt = TY(uns ? K_UINT : K_SINT, 32);
				c.rdsel = 1;
				x0 = mk_cast(vt, spv_node(&c, idx(I->opnd[base]), 0));
				body = mk_cast(t, mk_bin(div ? O_DIV : O_MOD, x0,
				    mk_cast(vt, spv_node(&c, idx(I->opnd[base + 1]),
				    0))));
				cs = c.fail ? 0 : div_cases(&c, idx(I->opnd[base + 1]),
				    x0, div ? O_DIV : O_MOD, vt, 32, 0);
				if (!cs || c.fail || c.nleaf == 0) {
					free(c.memo);
					continue;
				}
				kind = 4;
				r = xcalloc(1, sizeof *r);
				nr = 1;
				r->e = mk_cast(t, cs);
				r->at = body;
				r->to = r->e;
				r->derived = 1;
				snprintf(r->what, sizeof r->what, "%s, its divisor %s "
				    "a few constants: a %s by each", shared_str(body, 80),
				    V.id[res_of(idx(I->opnd[base + 1]))].tok, div ?
				    "division" : "remainder");
				snprintf(r->cond, sizeof r->cond, "%s takes no other "
				    "value (%s)", V.id[res_of(idx(I->opnd[base + 1]))].tok,
				    few_why(idx(I->opnd[base + 1])));
				r->proof = strdup("in each case the divisor is that "
				    "constant, so the quotient (the remainder) is the "
				    "division by it; a select of the divisor's own "
				    "condition picks its arm's case, a compare of the "
				    "divisor with each value picks the others' (the last "
				    "case takes what no compare does)");
				goto cands;
			}
			if (I->nop < base + 2 || int_width(V.id[root].type) != 32 ||
			    !const_of(idx(I->opnd[base + 1]), &dv) || dv < 3 ||
			    dv >= (1ULL << 31)) {
				free(c.memo);
				continue;
			}
			t = spv_ty(V.id[root].type);
			vt = TY(uns ? K_UINT : K_SINT, 32);
			body = mk_cast(t, mk_bin(div ? O_DIV : O_MOD, mk_cast(vt,
			    spv_node(&c, idx(I->opnd[base]), 0)), mk_const_i(vt,
			    (int64_t)dv)));
		} else if ((kind = idrw_root_kind(k, partof)) == 1 && tybool >= 0) {
			/* n % d == r: the remainder (used only here) read as
			 * the division roots read theirs; an OpSMod is the
			 * remainder where it is 0 (both say d divides n) */
			int a1 = res_of(idx(I->opnd[1])), a2 = res_of(idx(I->opnd[2]));
			int ra = is_rem(a1) ? a1 : a2, q = ra == a1 ? a2 : a1, sg;
			int64_t rv;
			uint64_t kv;
			const struct sins *M = &V.in[SP.def[ra]];
			sg = strcmp(M->op, "OpUMod") != 0;
			smod = !strcmp(M->op, "OpSMod");
			remid = ra;
			const_of(idx(M->opnd[2]), &dv);
			const_of(q, &kv);
			rv = int_norm(TY(sg ? K_SINT : K_UINT, 32), (int64_t)kv);
			c.rdsel = 1;
			t = TY_BOOL;
			vt = TY(sg ? K_SINT : K_UINT, 32);
			if (int_width(V.id[ra].type) != 32 || dv >= (1ULL << 31) ||
			    (!strcmp(M->op, "OpSMod") && rv != 0)) {
				free(c.memo);
				continue;
			}
			body = mk_bin(!strcmp(I->op, "OpIEqual") ? O_EQ : O_NE,
			    mk_bin(O_MOD, mk_cast(vt, spv_node(&c, idx(M->opnd[1]),
			    0)), mk_const_i(vt, (int64_t)dv)), mk_const_i(vt, rv));
		} else if (kind == 2 && tybool >= 0) {
			/* the compares, selects and logic under it are read as
			 * well */
			c.rdsel = 1;
			t = spv_ty_c(&c, V.id[root].type);
			body = t ? mk_cast(t, spv_node(&c, root, 0)) : 0;
		} else if (kind == 3) {
			t = spv_ty(V.id[root].type);
			body = t ? mk_cast(t, spv_node(&c, root, 0)) : 0;
		} else {
			free(c.memo);
			continue;
		}
		c.f.body = body;
		if (!t || c.fail || c.nleaf == 0) {
			free(c.memo);
			continue;
		}
		nr = irw_cands(&c.f, body, &r);
cands:
		c.f.body = body;
		before = fuse_expr_cost(m, body);
		/* the solved identities, whose derivation is the proof (the
		 * signed-as-unsigned ones are the SAT pass's): for a division
		 * those SAT does not finish on, otherwise every derived one */
		for (i = 0; i < nr; i++) {
			float cost;
			/* a sum: its own top only, not one inside it that
			 * something else may read too */
			if (r[i].nearmiss || !(kind == 3 ? r[i].factor &&
			    (r[i].at == body || r[i].at == N.a[body]) : kind ?
			    r[i].derived : r[i].nosat))
				continue;
			cost = fuse_expr_cost(m, r[i].e);
			if (cost < before && (best < 0 || cost < after)) {
				best = i;
				after = cost;
			}
		}
		if (best >= 0 && kind && kind != 4)
			after += (float)(kept = drops_kept(&c, root, r[best].e));
		if (best < 0 || !(after < before) ||
		    !dot_translation_ok(&c, root, r[best].e)) {
			irw_free(r, nr);
			free(c.memo);
			continue;
		}
		nin0 = V.nin;
		nk0 = SP.nnewk;
		fin = dot_emit(&c, r[best].e);
		if (c.fail || fin < 0) {
			/* take back what was written */
			for (i = nin0; i < V.nin; i++) {
				int j;
				for (j = 0; j < V.in[i].nop; j++)
					free(V.in[i].opnd[j]);
				free(V.in[i].text);
			}
			V.nin = nin0;
			SP.nnewk = nk0;
			irw_free(r, nr);
			free(c.memo);
			continue;
		}
		I = &V.in[k];	/* dot_emit() may have moved V.in */
		SP.alias[root] = fin;
		SP.dead[k] = 1;
		snprintf(log[n].from, sizeof log[n].from, "%s", I->text);
		snprintf(log[n].to, sizeof log[n].to, "%s", shared_str(r[best].e,
		    150));
		log[n].gain = before - after;
		log[n].before = before;
		log[n].after = after;
		log[n].line = ins_line(k);
		log[n].desc = strdup(r[best].cond);
		{
			struct sbuf pf = {0};
			if (smod)
				sb_printf(&pf, "%s is OpSMod, whose remainder takes the "
				    "divisor's sign where %% (OpSRem) takes the "
				    "dividend's; either is 0 exactly where %llu divides "
				    "the dividend\n", V.id[remid].tok,
				    (unsigned long long)dv);
			sb_puts(&pf, r[best].proof);
			log[n].proof = pf.p;
		}
		log[n].kind = kind;
		{
			struct sbuf *o = kind == 4 ? rep4 : kind == 3 ? rep3 : kind ?
			    rep2 : rep;
			sb_printf(o, "    %s\n      ->  %s   (cost %g -> %g)\n",
			    I->text, log[n].to, (double)before, (double)after);
			src_line(ins_line(k), sl, sizeof sl);
			if (sl[0])
				sb_printf(o, "%s", sl);
			if (kind)
				sb_printf(o, "      %s\n", r[best].what);
			if (kept > 0)
				sb_printf(o, "      the price counts %g for what it drops "
				    "that something else reads too: that stays\n", kept);
			sb_printf(o, "      holds where %s\n", r[best].cond);
			for (s = log[n].proof; *s; s = *e ? e + 1 : e) {
				e = strchr(s, '\n');
				if (!e)
					e = s + strlen(s);
				sb_printf(o, "      %.*s\n", (int)(e - s), s);
			}
		}
		n++;
		irw_free(r, nr);
		free(c.memo);
	}
	free(partof);
	return n;
}


static struct {
	char name[40];
	float lo, hi;
} pend_fas[8];
static int npend_fas;
static struct fxopt fopt;
static int fopt_percomp, fopt_only;
static int fopt_nofile;	/* the table reads' texels: a checksum only */

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

/*
 * The sink of the module's stored and returned floats, by request
 * (`sink unorm8`): a format with fewer bits than f32, whose step is the
 * error budget float simplify may spend (fsimp.c).  An image store with
 * a format in its type has its own.  One step: 1/(2^n - 1) for unormN,
 * 1/(2^(n-1) - 1) for snormN (absolute); 2^-11 of the value for f16,
 * whose 11 significant bits make every normal step at least that.
 */
static struct {
	double abs, rel, min;
	char name[48];
	int integer;	/* `sink int`: one unit at a floor or an int() */
} fsink;

/*
 * A format's name to its step: unormN, snormN, f16, or the channels and
 * the kind as SPIR-V and Vulkan write them (Rgba8, Rgba8Snorm, Rgb10A2,
 * Rgba16f, R8G8B8A8_UNORM, B8G8R8A8_SRGB, R16G16B16A16_SFLOAT).  With
 * channels of different widths the widest sets the step for all (the
 * smallest step).  An sRGB format stores 8 bits of the encoded value:
 * the encoding's slope is at most 12.92, so one step is at least
 * 1/(12.92 (2^n - 1)) of the linear value the shader writes.  Integer
 * formats and floats of 32 bits are not sinks.
 */
static int
sink_lookup(const char *fmt, double *abs, double *rel, double *min)
{
	char b[48];
	const char *q;
	size_t i, n = 0;
	int bits = 0, kind = -1;	/* 0 unorm, 1 snorm, 2 float, 3 srgb */

	for (i = 0; fmt[i] && n + 1 < sizeof b; i++)
		if (fmt[i] != '_')
			b[n++] = (char)tolower((unsigned char)fmt[i]);
	b[n] = 0;
	q = b;
	if (!strncmp(q, "vkformat", 8))
		q += 8;
	if ((!strncmp(q, "unorm", 5) || !strncmp(q, "snorm", 5)) &&
	    isdigit((unsigned char)q[5])) {
		bits = atoi(q + 5);
		kind = q[0] == 's';
		for (q += 5; isdigit((unsigned char)*q); q++)
			;
		if (*q)
			return 0;
	} else if (!strcmp(q, "f16") || !strcmp(q, "half")) {
		bits = 16;
		kind = 2;
	} else {
		/* channel letters and their bits, then the kind */
		while (*q && strchr("rgbax", *q)) {
			int v;
			while (*q && strchr("rgbax", *q))
				q++;
			if (!isdigit((unsigned char)*q))
				return 0;
			v = atoi(q);
			if (v > bits)
				bits = v;
			while (isdigit((unsigned char)*q))
				q++;
		}
		if (!bits)
			return 0;
		if (!*q || !strncmp(q, "unorm", 5))
			kind = 0;
		else if (!strncmp(q, "snorm", 5))
			kind = 1;
		else if (!strncmp(q, "srgb", 4))
			kind = 3;
		else if (!strcmp(q, "f") || !strncmp(q, "sfloat", 6) ||
		    !strcmp(q, "float"))
			kind = 2;
		else
			return 0;	/* an integer format, or not a format */
	}
	if (kind == 2 ? bits != 16 : bits < 2 || bits > 23)
		return 0;
	*abs = *rel = *min = 0;
	if (kind == 2) {
		*rel = 0x1p-11;
		*min = 0x1p-14;
	} else {
		*abs = 1.0 / (double)((1 << (kind == 1 ? bits - 1 : bits)) - 1);
		if (kind == 3)
			*abs /= 12.92;
	}
	return 1;
}

int
spirv_sink_known(const char *fmt)
{
	double a, r, m;

	return sink_lookup(fmt, &a, &r, &m);
}

void
spirv_sink_int(void)
{
	fsink.integer = 1;
}

/* 0: not a sink ucas knows */
int
spirv_sink_opt(const char *fmt)
{
	if (fmt == NULL) {
		memset(&fsink, 0, sizeof fsink);
		return 1;
	}
	if (!sink_lookup(fmt, &fsink.abs, &fsink.rel, &fsink.min)) {
		fsink.abs = fsink.rel = fsink.min = 0;
		return 0;
	}
	snprintf(fsink.name, sizeof fsink.name, "%s", fmt);
	return 1;
}

/* the format in the type of the image an OpImageWrite stores to */
static const char *
image_format(int img)
{
	int k, t;

	if (img < 0 || img >= V.nid)
		return NULL;
	t = V.id[img].type;
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strcmp(I->op, "OpFunction"))
			break;
		if (I->res == t && !strcmp(I->op, "OpTypeImage") && I->nop >= 7)
			return I->opnd[6];
	}
	return NULL;
}

/*
 * GLSL's pack functions round their argument to 8 or 16 bits themselves
 * (packUnorm4x8: round(clamp(c, 0, 1) * 255)): the format their
 * argument ends in, or NULL.
 */
static const char *
pack_format(const struct sins *I)
{
	static const char *const pk[][2] = {{"PackUnorm4x8", "unorm8"},
	    {"PackSnorm4x8", "snorm8"}, {"PackUnorm2x16", "unorm16"},
	    {"PackSnorm2x16", "snorm16"}};
	size_t i;

	if (strcmp(I->op, "OpExtInst") || I->nop < 4 || V.glsl < 0 ||
	    idx(I->opnd[1]) != V.glsl)
		return NULL;
	for (i = 0; i < sizeof pk / sizeof *pk; i++)
		if (!strcmp(I->opnd[2], pk[i][0]))
			return pk[i][1];
	return NULL;
}

static double fprice_ins(const char *op, const char *ext, int rt);

/* float arithmetic under a value put together from lanes (a construct,
 * a shuffle, an extract or a copy of one) */
static int
float_under(int v, int depth)
{
	const struct sins *I;
	int i;

	if (v < 0 || SP.def[v] < 0 || depth > 8)
		return 0;
	I = &V.in[SP.def[v]];
	/* (a conversion is priced, but it is not arithmetic to simplify) */
	if (strncmp(I->op, "OpConvert", 9) && fprice_ins(I->op,
	    !strcmp(I->op, "OpExtInst") && I->nop >= 3 ? I->opnd[2] : NULL,
	    V.id[v].type) > 0)
		return 1;
	if (strcmp(I->op, "OpCompositeConstruct") && strcmp(I->op,
	    "OpVectorShuffle") && strcmp(I->op, "OpCompositeExtract") &&
	    strcmp(I->op, "OpCopyObject"))
		return 0;
	for (i = 1; i < I->nop; i++)
		if (I->opnd[i][0] == '%' && float_under(idx(I->opnd[i]), depth + 1))
			return 1;
	return 0;
}

/*
 * A float used only through a discrete decision: the operand of a
 * conversion to an integer (*disc 1), or the side of a compare whose
 * other side is a constant (*disc 2).  A root with no budget beyond the
 * original's precision: moving it by less than a unit moves the integer
 * by at most one, and a compare has no continuous budget at all.
 */
static int
disc_root(const struct sins *I, int *disc)
{
	int a, b;

	*disc = 0;
	if ((!strcmp(I->op, "OpConvertFToU") || !strcmp(I->op,
	    "OpConvertFToS")) && I->nop >= 2 && I->opnd[1][0] == '%') {
		*disc = 1;
		return idx(I->opnd[1]);
	}
	if ((strncmp(I->op, "OpFOrd", 6) && strncmp(I->op, "OpFUnord", 8)) ||
	    I->nop < 3 || I->opnd[1][0] != '%' || I->opnd[2][0] != '%')
		return -1;
	a = idx(I->opnd[1]);
	b = idx(I->opnd[2]);
	if ((V.id[a].kind == K_CONST) == (V.id[b].kind == K_CONST))
		return -1;	/* two values, or two constants */
	*disc = 2;
	return V.id[a].kind == K_CONST ? b : a;
}

/* x of floor(x) (GLSL.std.450 Floor), or -1 */
static int
floor_arg(int v)
{
	const struct sins *I;

	if (v < 0 || SP.def[v] < 0)
		return -1;
	I = &V.in[SP.def[v]];
	if (strcmp(I->op, "OpExtInst") || I->nop < 4 || V.glsl < 0 ||
	    idx(I->opnd[1]) != V.glsl || strcmp(I->opnd[2], "Floor") ||
	    I->opnd[3][0] != '%')
		return -1;
	return idx(I->opnd[3]);
}

/* a root's sink: the image's or the pack function's format (named by
 * what set it), else the request's */
static void
root_sink(struct fxg *g, const char *imgfmt, const char *by)
{
	if (imgfmt && sink_lookup(imgfmt, &g->sink, &g->sinkrel, &g->sinkmin)) {
		snprintf(g->sinkname, sizeof g->sinkname, "%s", by ? by : imgfmt);
		return;
	}
	if (imgfmt)
		return;		/* an image whose format has f32's bits or more */
	g->sink = fsink.abs;
	g->sinkrel = fsink.rel;
	g->sinkmin = fsink.min;
	snprintf(g->sinkname, sizeof g->sinkname, "%s", fsink.name);
}

void
spirv_float_opts(int mode, double ulp, int approx, int percomp, int only,
    int search, uint64_t seed)
{
	fopt.mode = mode;
	fopt.ulp = ulp;
	fopt.approx = approx;
	fopt.search = search;
	fopt.seed = seed;
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
	/* clamp(x, lo, hi) is min(max(x, lo), hi) (GLSL.std.450 FClamp): two
	 * nodes for the one instruction, so that what follows knows x's
	 * range */
	if (op < 0 && !strcmp(I->op, "OpExtInst") && I->nop >= 6 &&
	    idx(I->opnd[1]) == V.glsl && !strcmp(I->opnd[2], "FClamp")) {
		int x = fx_opnd(g, I->opnd[3], lane, depth), lo, hi, mx;
		if (x < 0 || (lo = fx_opnd(g, I->opnd[4], lane, depth)) < 0 ||
		    (hi = fx_opnd(g, I->opnd[5], lane, depth)) < 0 ||
		    (mx = fx_new(g, FX_MAX, x, lo, -1, k, id)) < 0)
			return -1;
		r = fx_new(g, FX_MIN, mx, hi, -1, k, id);
		goto done;
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

	if (a->nn != b->nn || a->nin != b->nin || a->sink != b->sink ||
	    a->sinkrel != b->sinkrel)
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

/*
 * GLSL.std.450's other float instructions, per component of an n-wide
 * result, as model.c prices the operations each computes (floor, ceil
 * and roundeven by node_cost, the rest as the sum of their operations'
 * classes); -1: not one of them
 */
static double
fprice_std(const char *ext, int n)
{
	static const struct { const char *ext; int op; } one[] = {
		{"Floor", O_FLOOR}, {"Ceil", O_CEIL}, {"Trunc", O_FLOOR},
		{"Round", O_ROUNDEVEN}, {"RoundEven", O_ROUNDEVEN}, {NULL, 0}};
	const struct model *m = model_active();
	const float *c = m->c;
	nid x = mk_var(0, sym_cstr("x"), TY_F32);
	int i, cls;

	for (i = 0; one[i].ext; i++)
		if (!strcmp(ext, one[i].ext))
			return node_cost(m, mk_un(one[i].op, x), &cls);
	/*
	 * A model without ldexp's or frexp's instruction gets them lowered
	 * as GLSL lets a driver (Mesa's nir_opt_algebraic and
	 * nir_lower_frexp): ldexp clamps k to +-254 and multiplies by 2^(k
	 * >> 1) and 2^(k - (k >> 1)), each built as an exponent field;
	 * frexp's mantissa is the bits' sign and mantissa under the exponent
	 * of [0.5, 1) where x is finite and not 0, its exponent |x|'s field
	 * less 126 where x is not 0
	 */
	if (!strcmp(ext, "Ldexp"))
		return cclass_available(m, C_LDEXP) ? c[C_LDEXP] :
		    2 * c[C_MINMAX] + 3 * c[C_SHIFT] + 3 * c[C_IADD] +
		    2 * c[C_FMUL];
	if (!strcmp(ext, "Frexp"))
		return cclass_available(m, C_FREXP) ? 2 * c[C_FREXP] :
		    c[C_MINMAX] + 3 * c[C_CMP] + 3 * c[C_LOGIC] +
		    2 * c[C_SELECT] + c[C_SHIFT] + c[C_IADD];
	/* x - floor(x); Modf's x - trunc(x) */
	if (!strcmp(ext, "Fract") || !strcmp(ext, "Modf"))
		return node_cost(m, mk_un(O_FLOOR, x), &cls) + c[C_FADD];
	if (!strcmp(ext, "FClamp") || !strcmp(ext, "NClamp"))
		return 2 * c[C_MINMAX];
	if (!strcmp(ext, "Radians") || !strcmp(ext, "Degrees"))
		return c[C_FMUL];
	if (!strcmp(ext, "SmoothStep"))	/* two subtracts, a divide, the
					 * clamp, t*t*(3 - 2t) */
		return 2 * c[C_FADD] + c[C_FDIV] + 2 * c[C_MINMAX] +
		    3 * c[C_FMUL];
	if (n < 1)
		n = 1;
	/* the vector ones: their whole work, spread over the components */
	if (!strcmp(ext, "Normalize"))	/* the dot, its inverse root, a
					 * multiply each */
		return (2 * n * c[C_FMUL] + fx_price(FX_ISQRT)) / n;
	if (!strcmp(ext, "Cross"))	/* a multiply and an fma each */
		return 2 * c[C_FMUL];
	if (!strcmp(ext, "Reflect"))	/* the dot, 2 dot, an fma each */
		return ((2 * n + 1) * c[C_FMUL]) / n;
	if (!strcmp(ext, "Refract"))	/* the dot, k, its root and the
					 * select, two products each */
		return ((3 * n + 4) * c[C_FMUL] + 2 * c[C_FADD] + c[C_CMP] +
		    c[C_FSQRT] + n * c[C_SELECT]) / n;
	if (!strcmp(ext, "FaceForward"))	/* the dot, its sign, a negate
						 * and a select each */
		return (n * (c[C_FMUL] + c[C_FADD] + c[C_SELECT]) + c[C_CMP]) /
		    n;
	return -1;
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
		/* the library calls, as model.c prices the same operation */
		static const struct { const char *ext; int op; } lib[] = {
			{"Exp", O_EXP}, {"Log", O_LOG}, {"Sin", O_SIN},
			{"Cos", O_COS}, {"Tan", O_TAN}, {"Asin", O_ASIN},
			{"Acos", O_ACOS}, {"Atan", O_ATAN}, {"Atan2", O_ATAN2},
			{"Sinh", O_SINH}, {"Cosh", O_COSH}, {"Tanh", O_TANH},
			{"Asinh", O_ASINH}, {"Acosh", O_ACOSH}, {"Atanh", O_ATANH},
			{NULL, 0}};
		for (i = 0; fx_ext[i].ext; i++)
			if (!strcmp(ext, fx_ext[i].ext))
				p = fx_price(fx_ext[i].op);
		for (i = 0; p < 0 && is_float_t(rt) && lib[i].ext; i++)
			if (!strcmp(ext, lib[i].ext)) {
				nid x = mk_var(0, sym_cstr("x"), TY_F32);
				int cls;
				p = node_cost(model_active(), lib[i].op == O_ATAN2 ?
				    mk_bin(O_ATAN2, x, x) : mk_un(lib[i].op, x), &cls);
			}
		if (p < 0 && is_float_t(rt))
			p = fprice_std(ext, n);
		if (p < 0 && is_float_t(rt))
			p = c[C_MINMAX];
	} else if (!strcmp(op, "OpConvertUToF") ||
	    !strcmp(op, "OpConvertSToF") || !strcmp(op, "OpConvertFToU") ||
	    !strcmp(op, "OpConvertFToS"))
		p = c[C_CONV];	/* as model.c prices a cast to or from a float */
	return p < 0 ? 0 : p * n;
}

/*
 * The price of one instruction, where more than its result type says:
 * a length or a distance of a vector is a scalar of n components' work,
 * and frexp and modf return a struct
 */
static double
fprice_i(const struct sins *I)
{
	const char *ext = !strcmp(I->op, "OpExtInst") && I->nop >= 3 ?
	    I->opnd[2] : NULL;
	int rt = idx(I->opnd[0]);
	const float *c = model_active()->c;

	if (ext && (!strcmp(ext, "Length") || !strcmp(ext, "Distance")) &&
	    I->nop >= 4 && I->opnd[3][0] == '%' && is_float_t(rt)) {
		int t = V.id[idx(I->opnd[3])].type, n = t >= 0 ? ncomp(t) : 1;
		/* the squares and their sum (fmas), the root; a distance's
		 * differences first */
		return (ext[0] == 'D' ? n * c[C_FADD] : 0) + n * c[C_FMUL] +
		    c[C_FSQRT];
	}
	/* frexp and modf as glslang writes them: a struct whose first
	 * member is the float result */
	if (ext && (!strcmp(ext, "FrexpStruct") || !strcmp(ext,
	    "ModfStruct")) && rt >= 0 && V.id[rt].kind == K_TYPE &&
	    T(rt)->k == S_STRUCT && T(rt)->nm >= 1)
		return fprice_ins(I->op, ext[0] == 'F' ? "Frexp" : "Modf",
		    T(rt)->m[0]);
	return fprice_ins(I->op, ext, rt);
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

/* a polynomial's: the function it stands for, from its why ("exp2 ~
 * degree-2 ..."); the rest by their kind */
/* an approximation's kind: listed with its error */
static int
approx_kind(int k)
{
	return k == FXR_POLY || k == FXR_LOGDOM;
}

/* a polynomial the shader had, refitted (fsimp.c an_fewer) */
static int
is_refit(const struct fxrw *r)
{
	return r && r->kind == FXR_POLY && !strncmp(r->why, "polynomial of "
	    "degree", 20);
}

/* node x of graph g as an operand of a result of type rt: its value
 * when it has that type, else -2 when it is a component of an input
 * vector (refit_comp), else -1 */
static int refit_comp(const struct fxg *g, int x, int rt);

static int
operand_ok(const struct fxg *g, int x, int rt)
{
	int xs = x >= 0 ? g->n[x].sid : -1;

	if (xs < 0)
		return -1;
	if (type_of(xs) == rt)
		return xs;
	if (ncomp(rt) > 1 && type_of(xs) == scalar_of(rt))
		return -3;	/* a scalar the vector result splats */
	return refit_comp(g, x, rt) >= 0 ? -2 : -1;
}

/* the token of node x as an operand (a component read out first when
 * it is one of an input vector, a scalar splatted when the result is a
 * vector of it; its name made from tmp and c) */
static const char *
operand_tok(struct sbuf *s, const struct fxg *g, int x, int rt,
    const char *RT, int tmp, int c, char *buf, size_t nb)
{
	int xs = g->n[x].sid, i;

	if (type_of(xs) == rt)
		return V.id[xs].tok;
	snprintf(buf, nb, "%%ucas_f%dc%d", tmp, c);
	if (ncomp(rt) > 1 && type_of(xs) == scalar_of(rt)) {
		sb_printf(s, "%s = OpCompositeConstruct %s", buf, RT);
		for (i = 0; i < ncomp(rt); i++)
			sb_printf(s, " %s", V.id[xs].tok);
		sb_puts(s, "\n");
		return buf;
	}
	sb_printf(s, "%s = OpCompositeExtract %s %s %d\n", buf, RT,
	    V.id[xs].tok, refit_comp(g, x, rt));
	return buf;
}

/* a whole-expression rewrite (a refit, a product in the log domain) can
 * be written: each operand is a value of the result's type, or one it
 * reads (a component, a splat); else the reason, in why */
static int
whole_writable(const struct fxg *g, const struct fxrw *r, int rt, char *why,
    size_t nw)
{
	int f2, bad = -1;

	if (r->kind == FXR_LOGDOM)
		for (f2 = 0; f2 < r->nf && bad < 0; f2++) {
			if (operand_ok(g, r->fa[f2], rt) == -1)
				bad = r->fa[f2];
			else if (r->fy[f2] >= 0 && operand_ok(g, r->fy[f2], rt) == -1)
				bad = r->fy[f2];
		}
	else if (operand_ok(g, r->x, rt) == -1 || operand_ok(g, r->x, rt) == -3)
		bad = r->x;
	if (bad < 0)
		return 1;
	snprintf(why, nw, "not written: its operand %s is not a value of the "
	    "result's type", g->n[bad].sid >= 0 ? V.id[g->n[bad].sid].tok :
	    "(a value inside the expression)");
	return 0;
}

/* the component of its vector an input lane is, for a scalar result of
 * that component's type; -1 when x is not that */
static int
refit_comp(const struct fxg *g, int x, int rt)
{
	int xs = g->n[x].sid, t;

	if (xs < 0 || g->n[x].op != FX_IN || ncomp(rt) != 1)
		return -1;
	t = type_of(xs);
	if (t < 0 || T(t)->k != S_VEC || scalar_of(t) != rt)
		return -1;
	return g->in[g->n[x].in].lane;
}

static const char *
rw_kind_text(const struct fxrw *r)
{
	static char b[48];
	size_t n;

	if (r == NULL || r->kind != FXR_POLY)
		return NULL;
	if (is_refit(r))
		return "a polynomial with fewer terms";
	n = strcspn(r->why, " ");
	snprintf(b, sizeof b, "%.*s as a polynomial", (int)(n < 20 ? n : 20),
	    r->why);
	return b;
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
	case FXR_LOGDOM: return "a product of powers as exp2 of a sum of logs";
	case FXR_TABLE: return "a function of one value as a table read";
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
	if (approx_kind(rr->kind))
		rep_num("rel_error", rr->rel);
	if (approx_kind(rr->kind) && rr->abserr > 0)
		rep_num("abs_error", rr->abserr);
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
 * The table reads float simplify proposes (fsimp.c an_table): never
 * written, since the module cannot add the texture's binding; each with
 * the read in GLSL, what the pipeline needs, its derivation, and its
 * texels as a file of their own (tableN.bin: R8 bytes, or little-endian
 * f32), numbered in the order they are listed.
 */
static const char *
tab_fmt(const struct fxrw *r)
{
	return r->tfmt == FXT_R8 ? "R8_UNORM" : "R32_SFLOAT";
}

/* a float literal of GLSL: "%.9g" with a point */
static const char *
tab_lit(float v)
{
	static char ring[4][40];
	static int at;
	char *b = ring[at++ & 3];

	snprintf(b, 40, "%.9g", (double)v);
	if (!strpbrk(b, ".en"))
		strcat(b, ".0");
	return b;
}

/* the read in GLSL: x the input (float(x) for an integer's) */
static void
tab_glsl(struct sbuf *b, const struct fxg *g, const struct fxrw *r)
{
	const struct fxin *in = &g->in[r->tin];
	char x[64];

	snprintf(x, sizeof x, in->cvt ? "float(%s)" : "%s", in->name);
	if (r->tfmt == FXT_R8 && r->tdeg == 1) {
		/* tab_lit keeps four literals at a time */
		sb_printf(b, "fma(textureLod(T, fma(%s, %s, %s), 0.0).r, %s, ", x,
		    tab_lit(r->k[0]), tab_lit(r->k[1]), tab_lit(r->k[2]));
		sb_printf(b, "fma(%s, %s, %s))", tab_lit(r->k[7]), x,
		    tab_lit(r->k[6]));
	} else if (r->tfmt == FXT_R8 && r->tdeg == 2) {
		sb_printf(b, "fma(textureLod(T, fma(%s, %s, %s), 0.0).r, %s, ", x,
		    tab_lit(r->k[0]), tab_lit(r->k[1]), tab_lit(r->k[2]));
		sb_printf(b, "fma(fma(%s, %s, %s), %s, %s))", tab_lit(r->k[8]), x,
		    tab_lit(r->k[7]), x, tab_lit(r->k[6]));
	} else if (r->tfmt == FXT_R8)
		sb_printf(b, "fma(textureLod(T, fma(%s, %s, %s), 0.0).r, %s, %s)",
		    x, tab_lit(r->k[0]), tab_lit(r->k[1]), tab_lit(r->k[2]),
		    tab_lit(r->k[3]));
	else
		sb_printf(b, "textureLod(T, fma(%s, %s, %s), 0.0).r", x,
		    tab_lit(r->k[0]), tab_lit(r->k[1]));
}

static void
tab_text(struct sbuf *rep, const struct fplan *pl, int npl,
    const struct fxg *G, struct fxrw *RW, const int *grp, const int *lane_of,
    int ngr)
{
	int i, c, nf = 0;

	sb_printf(rep, "  table reads (a proposal, never applied: a module "
	    "cannot add the texture's binding):\n");
	for (i = 0; i < npl; i++) {
		int vec;
		if (pl[i].kind != FXR_TABLE)
			continue;
		vec = ncomp(idx(V.in[pl[i].k].opnd[0])) > 1;
		sb_printf(rep, "    %s\n", V.in[pl[i].k].text);
		for (c = 0; c < pl[i].ng; c++) {
			struct fxrw *r = pl[i].rw[c];
			const struct fxg *g = &G[(int)((r - RW) / 48)];
			const char *ind = pl[i].ng > 1 ? "        " : "      ";
			int j;
			if (pl[i].ng > 1) {
				char lb[16];
				glanes(lb, sizeof lb, (int)((r - RW) / 48), grp, lane_of,
				    ngr);
				sb_printf(rep, "      lane%s %s:\n", strlen(lb) > 1 ? "s" :
				    "", lb);
			}
			sb_printf(rep, "%s%s, cost %g (-%g)\n", ind, r->why, r->cost,
			    r->gain);
			if (r->fail[0] || !r->sink) {
				sb_printf(rep, "%s  %s\n", ind, r->fail[0] ? r->fail :
				    r->sinkfail);
				continue;
			}
			sb_printf(rep, "%s  as ", ind);
			tab_glsl(rep, g, r);
			sb_printf(rep, "%s\n", vec ? ", component by component" : "");
			sb_printf(rep, "%s  T: a texture %s, %d x 1, one level, the "
			    "texels in table%d.bin%s; a sampler: LINEAR mag and min "
			    "filters, NEAREST mipmap mode, CLAMP_TO_EDGE, "
			    "unnormalizedCoordinates, lod 0; one combined image sampler "
			    "more in the layout\n", ind, tab_fmt(r), r->ntab, ++nf,
			    r->tfmt == FXT_R8 ? " as bytes" : " as f32");
			pr_lines(rep, r->proof.p, pl[i].ng > 1 ? "          " :
			    "        ");
			if (r->ntab <= 17) {
				sb_printf(rep, "%s  texels:", ind);
				for (j = 0; j < r->ntab; j++)
					sb_printf(rep, " %.9g", (double)r->tab[j]);
				sb_puts(rep, "\n");
			}
		}
	}
}

static void
tab_json(const struct fplan *P, struct fxrw *RW, const int *grp,
    const int *lane_of, int ngr)
{
	int c;

	if (!rep_on)
		return;
	rep_obj(NULL);
	rep_str("instruction", V.in[P->k].text);
	rep_list("lanes");
	for (c = 0; c < P->ng; c++) {
		const struct fxrw *r = P->rw[c];
		rep_obj(NULL);
		if (P->ng > 1) {
			char lb[16];
			glanes(lb, sizeof lb, (int)((r - RW) / 48), grp, lane_of, ngr);
			rep_str("lanes", lb);
		}
		rep_str("why", r->why);
		rep_num("cost", r->cost);
		rep_num("gain", r->gain);
		if (r->fail[0] || !r->sink) {
			rep_str("refused", r->fail[0] ? r->fail : r->sinkfail);
			rep_end();
			continue;
		}
		rep_str("format", tab_fmt(r));
		rep_list("texels");	/* R8: the bytes; R32F: the f32 values */
		{
			int j;
			for (j = 0; j < r->ntab; j++)
				rep_num(NULL, r->tab[j]);
		}
		rep_end();
		rep_num("coord_scale", r->k[0]);
		rep_num("coord_offset", r->k[1]);
		if (r->tfmt == FXT_R8) {
			rep_num("value_scale", r->k[2]);
			if (r->tdeg) {
				/* R's coefficients, from the constant up */
				int j;
				rep_list("trend");
				for (j = 0; j <= r->tdeg; j++)
					rep_num(NULL, r->k[6 + j]);
				rep_end();
			} else
				rep_num("value_offset", r->k[3]);
		}
		rep_num("abs_error", r->abserr);
		rep_num("sink_used", r->sinkdev);
		rep_lines("derivation", r->proof.p);
		rep_end();
	}
	rep_end();
	rep_end();
}

/* each table's texels, in the order tab_text numbered them */
static void
tab_files(const struct fplan *pl, int npl, struct fxrw *RW)
{
	int i, c, j, nf = 0;

	(void)RW;
	for (i = 0; i < npl; i++) {
		if (pl[i].kind != FXR_TABLE)
			continue;
		for (c = 0; c < pl[i].ng; c++) {
			const struct fxrw *r = pl[i].rw[c];
			unsigned char *b;
			size_t n;
			char name[32];
			if (r->fail[0] || !r->sink)
				continue;
			n = (size_t)r->ntab * (r->tfmt == FXT_R8 ? 1 : 4);
			b = xmalloc(n);
			for (j = 0; j < r->ntab; j++)
				if (r->tfmt == FXT_R8)
					b[j] = (unsigned char)r->tab[j];
				else {
					uint32_t w;
					memcpy(&w, &r->tab[j], 4);
					b[4 * j] = (unsigned char)w;
					b[4 * j + 1] = (unsigned char)(w >> 8);
					b[4 * j + 2] = (unsigned char)(w >> 16);
					b[4 * j + 3] = (unsigned char)(w >> 24);
				}
			snprintf(name, sizeof name, "table%d.bin", ++nf);
			deliver_file(name, "application/octet-stream", (const char *)b,
			    n, fopt_nofile);
			free(b);
		}
	}
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
	int discv[64], ndiscv = 0;
	int tmp = 0;	/* names of new ids: %ucas_fN */
	char **repl, *dead, *inner;
	struct sbuf decl = {0}, out = {0}, rep = {0}, lanes_txt = {0};
	struct sbuf disc_txt = {0};
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
		if (I->res >= 0 && SP.def[I->res] == k && I->nop >= 1 &&
		    I->opnd[0][0] == '%')
			before += fprice_i(I);
	}
	/* the roots: stored and returned float values */
	for (k = 0; k < V.nin && ngr < 60; k++) {
		const struct sins *I = &V.in[k];
		int v = -1, n, l, r0, disc = 0, unit = 0;
		const char *imgfmt = NULL, *sinkby = NULL;
		if (!strcmp(I->op, "OpStore") && I->nop >= 2)
			v = idx(I->opnd[1]);
		else if (!strcmp(I->op, "OpReturnValue") && I->nop >= 1)
			v = idx(I->opnd[0]);
		else if (!strcmp(I->op, "OpImageWrite") && I->nop >= 3 &&
		    (imgfmt = image_format(idx(I->opnd[0]))) != NULL &&
		    sink_lookup(imgfmt, &(double){0}, &(double){0}, &(double){0}))
			v = idx(I->opnd[2]);	/* a format with a step: a sink */
		else if ((imgfmt = pack_format(I)) != NULL) {
			v = idx(I->opnd[3]);	/* rounded by the function itself */
			sinkby = I->opnd[2];
		} else {
			imgfmt = NULL;
			v = disc_root(I, &disc);
		}
		/*
		 * `sink int`: what a floor or a conversion to an integer reads
		 * may move by one unit (|v' - v| <= 1 moves floor and
		 * truncation by at most one); a floor is read through, to its
		 * argument, whether it is converted or stored
		 */
		if (fsink.integer && v >= 0 && (disc == 1 || (!disc && !imgfmt &&
		    floor_arg(v) >= 0))) {
			int a = floor_arg(v);
			if (a >= 0 && SP.def[a] >= 0 && is_float_t(V.id[a].type) &&
			    float_under(a, 0))
				v = a;
			if (disc == 1 || a == v)
				unit = disc == 1 && !strcmp(I->op, "OpConvertFToU") ?
				    2 : 1;
			if (!disc && a == v)
				disc = 3;	/* a stored floor, read through */
		}
		/* a stored value is usually put together from its lanes: the
		 * arithmetic is under the construct */
		if (v < 0 || SP.def[v] < 0 || !is_float_t(V.id[v].type) ||
		    !float_under(v, 0))
			continue;
		/* a value compared or converted twice is one such root; one
		 * also stored is a root both ways, and a rewrite of what they
		 * share must hold for the decision's root, which has no sink */
		for (j = 0; disc && j < ndiscv; j++)
			if (discv[j] == v)
				break;
		if (disc && j < ndiscv)
			continue;
		if (disc && ndiscv < 64)
			discv[ndiscv++] = v;
		if (disc && !unit)
			sb_printf(&disc_txt, "%s %s (%s)", disc_txt.n ? "," : "",
			    V.id[v].name, disc == 1 ? "converted to an integer" :
			    "compared with a constant");
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
			if (unit) {
				/* below one for an unsigned conversion: a value
				 * that was at least 0 stays above -1, which still
				 * truncates to 0 */
				g->sink = unit == 2 ? 1 - 0x1p-24 : 1;
				g->sinkrel = g->sinkmin = 0;
				snprintf(g->sinkname, sizeof g->sinkname, "int");
			} else if (disc)
				g->sink = g->sinkrel = g->sinkmin = 0;
			else
				root_sink(g, imgfmt, sinkby);
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
		free(lanes_txt.p);
		free(disc_txt.p);
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
			printf("float simplify: no float arithmetic reaches a store, "
			    "a return, a conversion to an integer or a compare with a "
			    "constant\n");
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
	/* the instructions inside a polynomial that is refitted as a whole:
	 * no rewrite of their own */
	inner = xcalloc((size_t)V.nin, 1);
	for (i = 0; i < npl; i++) {
		struct fxrw *r = pl[i].rw[0];
		unsigned char in[FX_MAXN];
		int gj, q, all = pl[i].ng > 0;
		if (!is_refit(r) && pl[i].kind != FXR_LOGDOM)
			continue;
		for (q = 0; q < pl[i].ng; q++)
			all = all && fx_accept(pl[i].rw[q], &fopt) && whole_writable(
			    &G[(int)((pl[i].rw[q] - RW) / 48)], pl[i].rw[q],
			    idx(V.in[pl[i].k].opnd[0]), pl[i].rw[q]->sinkfail,
			    sizeof pl[i].rw[q]->sinkfail);
		if (!all)
			continue;
		gj = (int)((r - RW) / 48);
		fx_inner(&G[gj], r, in);
		for (q = 0; q < G[gj].nn && q < FX_MAXN; q++)
			if (in[q] && q != r->node && G[gj].n[q].ins >= 0)
				inner[G[gj].n[q].ins] = pl[i].kind == FXR_LOGDOM ? 2 : 1;
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
				if (P->ng < groups || inner[P->k])
					ok = 0;
				for (q = 0; q < P->ng && ok; q++) {
					struct fxrw *r = P->rw[q];
					if (!fx_accept(r, &fopt))
						ok = 0;
					if (ok && (is_refit(r) || r->kind == FXR_LOGDOM) &&
					    !whole_writable(&G[(int)((r - RW) / 48)], r, rt,
					    r->sinkfail, sizeof r->sinkfail))
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
			/* the multiply is OpFMul, or OpVectorTimesScalar (vector
			 * first; Fma wants its scalar as a vector constant) */
			int a = idx(I->opnd[1]), m, x, kk, vts;
			int mi = SP.def[a] >= 0 && (!strcmp(V.in[SP.def[a]].op,
			    "OpFMul") || !strcmp(V.in[SP.def[a]].op,
			    "OpVectorTimesScalar")) ? 1 : 2;
			m = SP.def[idx(I->opnd[mi])];
			vts = !strcmp(V.in[m].op, "OpVectorTimesScalar");
			x = !vts && V.id[idx(V.in[m].opnd[1])].kind == K_CONST ? 2 : 1;
			kk = 3 - x;
			sb_printf(&s, "%s = OpExtInst %s %s Fma %s %s %s\n", R, RT, GL,
			    V.in[m].opnd[x], vts ? fconst(&decl, rt, lv[0], n) :
			    V.in[m].opnd[kk], I->opnd[3 - mi]);
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
		case FXR_LOGDOM: {
			const struct fxrw *r0 = P->rw[0];
			const struct fxg *gg = &G[(int)((r0 - RW) / 48)];
			char acc[48], vb[48], yb[48], tb[48];
			int f2, l2, c = 0;
			tmp++;
			for (f2 = 0; f2 < r0->nf; f2++) {
				const char *v = operand_tok(&s, gg, r0->fa[f2], rt, RT, tmp,
				    c++, vb, sizeof vb), *y = NULL;
				int one = r0->fy[f2] < 0, mone = one;
				if (!r0->fe[f2]) {
					snprintf(tb, sizeof tb, "%%ucas_f%dl%d", tmp, f2);
					sb_printf(&s, "%s = OpExtInst %s %s Log2 %s\n", tb, RT, GL,
					    v);
					snprintf(vb, sizeof vb, "%s", tb);
					v = vb;
				}
				for (l2 = 0; l2 < n && l2 < 4; l2++) {
					one = one && lv[f2][l2] == 1;
					mone = mone && lv[f2][l2] == -1;
				}
				if (r0->fy[f2] >= 0)
					y = operand_tok(&s, gg, r0->fy[f2], rt, RT, tmp, c++, yb,
					    sizeof yb);
				else if (!one && !mone)
					y = fconst(&decl, rt, lv[f2], n);
				snprintf(tb, sizeof tb, "%%ucas_f%ds%d", tmp, f2);
				if (f2 == 0) {
					if (one)
						snprintf(tb, sizeof tb, "%s", v);
					else if (mone)
						sb_printf(&s, "%s = OpFNegate %s %s\n", tb, RT, v);
					else
						sb_printf(&s, "%s = OpFMul %s %s %s\n", tb, RT, v, y);
				} else if (one)
					sb_printf(&s, "%s = OpFAdd %s %s %s\n", tb, RT, acc, v);
				else if (mone)
					sb_printf(&s, "%s = OpFSub %s %s %s\n", tb, RT, acc, v);
				else
					sb_printf(&s, "%s = OpExtInst %s %s Fma %s %s %s\n", tb, RT,
					    GL, v, y, acc);
				snprintf(acc, sizeof acc, "%s", tb);
			}
			for (l2 = 0; l2 < n && l2 < 4 && lv[FX_MAXF][l2] == 0; l2++)
				;
			if (l2 < n && l2 < 4) {
				sb_printf(&s, "%%ucas_f%dk = OpFAdd %s %s %s\n", tmp, RT, acc,
				    fconst(&decl, rt, lv[FX_MAXF], n));
				snprintf(acc, sizeof acc, "%%ucas_f%dk", tmp);
			}
			sb_printf(&s, "%s = OpExtInst %s %s Exp2 %s\n", R, RT, GL, acc);
			break;
		}
		case FXR_POLY: {
			int d = P->deg, e;
			char prev[48], tv[48], xb[48];
			const char *xv = I->opnd[3];
			tmp++;
			if (is_refit(P->rw[0])) {
				const struct fxg *gg = &G[(int)((P->rw[0] - RW) / 48)];
				int xs = gg->n[P->rw[0]->x].sid;
				xv = V.id[xs].tok;
				if (type_of(xs) != rt) {
					/* a component of a vector input */
					sb_printf(&s, "%%ucas_f%dx = OpCompositeExtract %s %s "
					    "%d\n", tmp, RT, xv, refit_comp(gg,
					    P->rw[0]->x, rt));
					snprintf(xb, sizeof xb, "%%ucas_f%dx", tmp);
					xv = xb;
				}
			}
			snprintf(tv, sizeof tv, "%%ucas_f%dt", tmp);
			if (is_refit(P->rw[0]) && lv[0][0] == 0 && (n < 2 ||
			    lv[0][1] == 0) && (n < 3 || lv[0][2] == 0) && (n < 4 ||
			    lv[0][3] == 0))
				/* centred at 0: t is x itself */
				snprintf(tv, sizeof tv, "%s", xv);
			else
				sb_printf(&s, "%s = OpFSub %s %s %s\n", tv, RT, xv,
				    fconst(&decl, rt, lv[0], n));
			snprintf(prev, sizeof prev, "%s", fconst(&decl, rt, lv[1], n));
			for (e = 1; e <= d; e++) {
				const char *ce = fconst(&decl, rt, lv[1 + e], n);
				if (e == d)
					sb_printf(&s, "%s = OpExtInst %s %s Fma %s %s %s\n", R, RT,
					    GL, prev, tv, ce);
				else {
					sb_printf(&s, "%%ucas_f%dh%d = OpExtInst %s %s Fma %s %s "
					    "%s\n", tmp, e, RT, GL, prev, tv, ce);
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
				after += fprice_i(I);
		}
	}
	/* the report */
	{
		int nd = 0, nap = 0, nrej = 0, napx = 0, ntb = 0, q;
		int ord[256], no = 0;
		for (i = 0; i < npl; i++) {
			if (pl[i].kind == FXR_DIVNOTE)
				nd++;
			else if (pl[i].kind == FXR_TABLE)
				ntb++;
			else if (pl[i].applied)
				ord[no++] = i;
			else if (approx_kind(pl[i].kind) && !inner[pl[i].k])
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
		if (disc_txt.p)
			sb_printf(&rep, "  values that decide (a compare with a "
			    "constant, a conversion to an integer) take no budget "
			    "beyond the original's precision, nor does what is under "
			    "them:%s\n", disc_txt.p);
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
		if (fopt.search) {
			rep_str("search", fopt.search == NS_ANNEAL ? "anneal" : "cmaes");
			rep_int("seed", (long long)fopt.seed);
		}
		rep_bool("percomp", fopt_percomp);
		{
			int any = 0;
			for (j = 0; j < ngr; j++)
				any |= grp[j] == j && (G[j].sink > 0 || G[j].sinkrel > 0);
			if (any) {
				rep_list("sinks");
				for (j = 0; j < ngr; j++) {
					const struct fxg *g = &G[j];
					if (grp[j] != j || (g->sink <= 0 && g->sinkrel <= 0))
						continue;
					rep_obj(NULL);
					rep_str("format", g->sinkname);
					if (g->sink > 0)
						rep_num("step", g->sink);
					else
						rep_num("relative_step", g->sinkrel);
					if (g->sinkbudget > 0) {
						rep_num("budget", g->sinkbudget);
						rep_num("used", g->sinkdev);
					}
					if (g->sinknote[0])
						rep_str("note", g->sinknote);
					if (g->sinkoffer[0])
						rep_str("offer", g->sinkoffer);
					rep_end();
				}
				rep_end();
			}
		}
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
			rep_str("kind", P->kind == FXR_POLY ? rw_kind_text(P->rw[0]) :
			    kind_text(P->kind));
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
			if (!approx_kind(pl[i].kind) || pl[i].applied || inner[pl[i].k])
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
		if (ntb) {
			rep_list("table_reads");
			for (i = 0; i < npl; i++)
				if (pl[i].kind == FXR_TABLE)
					tab_json(&pl[i], RW, grp, lane_of, ngr);
			rep_end();
		}
		rep_list("not_applied");
		for (i = 0; i < npl; i++) {
			int c;
			if (pl[i].applied || pl[i].kind == FXR_DIVNOTE ||
			    pl[i].kind == FXR_TABLE || approx_kind(pl[i].kind))
				continue;
			rep_obj(NULL);
			rep_str("instruction", V.in[pl[i].k].text);
			rep_str("kind", pl[i].kind == FXR_POLY ?
			    rw_kind_text(pl[i].rw[0]) : kind_text(pl[i].kind));
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
		for (j = 0; j < ngr; j++) {
			const struct fxg *g = &G[j];
			int z, first = 1;
			if (grp[j] != j || (g->sink <= 0 && g->sinkrel <= 0))
				continue;
			for (z = 0; z < j; z++)
				if (grp[z] == z && !strcmp(G[z].sinkname, g->sinkname))
					first = 0;
			if (first && !strcmp(g->sinkname, "int"))
				sb_printf(&rep, "  sink int: one unit.  A floor or a "
				    "conversion to an integer of a value within one of the "
				    "reference's is the reference's integer or a "
				    "neighbour (|v' - v| <= 1 moves floor and truncation by "
				    "at most one; below one for an unsigned conversion, "
				    "which a value under -1 would leave undefined): the "
				    "rewrites may move what each reads by that much in "
				    "all, wherever the reference's is a number\n");
			else if (first && g->sink > 0 && !strncmp(g->sinkname, "Pack",
			    4))
				sb_printf(&rep, "  sink %s: one step is %.6g.  The "
				    "function clamps, scales and rounds to nearest (GLSL's "
				    "round; the direction of an exact tie is the "
				    "implementation's): a value within one step of the "
				    "reference's is packed as the same integer or the "
				    "next.  The rewrites may move its argument by that "
				    "much in all, wherever the reference's is a number\n",
				    g->sinkname, g->sink);
			else if (first && g->sink > 0)
				sb_printf(&rep, "  sink %s: one step is %.6g.  The "
				    "conversion clamps, scales and rounds; rounding to "
				    "nearest (what Vulkan asks of it; it allows either "
				    "neighbour), a value within one step of the "
				    "reference's is stored as the same integer or the "
				    "next.  The rewrites may move a stored value by that "
				    "much in all, wherever the reference's is a number\n",
				    g->sinkname, g->sink);
			else if (first)
				sb_printf(&rep, "  sink %s: one step is at least %.6g of "
				    "the value (for magnitudes from %.3g): the rewrites "
				    "may move a stored value by that fraction of its "
				    "least magnitude in all, which is within one step "
				    "of the format\n", g->sinkname, g->sinkrel,
				    g->sinkmin);
			if (g->sinknote[0])
				sb_printf(&rep, "    %s\n", g->sinknote);
			else if (g->sinkdev > 0)
				sb_printf(&rep, "    budget %.3g, the rewrites accepted "
				    "use at most %.3g of it\n", g->sinkbudget, g->sinkdev);
			if (g->sinkoffer[0])
				sb_puts(&rep, g->sinkoffer);
		}
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
			sb_printf(&rep, "      (%s; -%g)\n", P->kind == FXR_POLY ?
			    rw_kind_text(P->rw[0]) : kind_text(P->kind), P->gain);
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
				if (!approx_kind(pl[i].kind) || pl[i].applied ||
				    inner[pl[i].k])
					continue;
				for (c = 0; c < pl[i].ng; c++)
					any |= pl[i].rw[c]->approx;
				if (r->rel > 0 || r->abserr <= 0)
					sb_printf(&rep, "    %s, cost %g (-%g): relative error "
					    "< %.3g (%g ulp), %s\n", r->why, r->cost, r->gain,
					    r->rel, r->ulp, any ? "beyond the original's "
					    "precision" : "within the original's precision");
				else
					sb_printf(&rep, "    %s, cost %g (-%g): absolute error "
					    "< %.3g (the function is 0 inside the range: no "
					    "relative bound), %s\n", r->why, r->cost, r->gain,
					    r->abserr, any ? "beyond the original's precision" :
					    "within the original's precision");
				for (c = 0; c < pl[i].ng; c++)
					if (pl[i].rw[c]->sinkfail[0]) {
						sb_printf(&rep, "      %s%s\n", strncmp(pl[i].rw[c]->
						    sinkfail, "not written", 11) ? "the sink refuses "
						    "it: " : "",
						    pl[i].rw[c]->sinkfail);
						break;
					}
				if (i == npl - 1 || pl[i + 1].kind != pl[i].kind ||
				    pl[i + 1].k != pl[i].k) {
					pr_lines(&rep, r->proof.p, "        ");
					if (pl[i].kind == FXR_POLY) {
						int e;
						sb_printf(&rep, "        P(t) = ");
						for (e = 0; e < r->deg; e++)
							sb_printf(&rep, "fma(");
						sb_printf(&rep, "%.9g", (double)r->k[1]);
						for (e = 1; e <= r->deg; e++)
							sb_printf(&rep, ", t, %.9g)", (double)r->k[1 + e]);
						sb_printf(&rep, ", t = %s - %.9g\n", strncmp(r->why,
						    "pow ", 4) ? "x" : "q", (double)r->k[0]);
					}
				}
			}
		}
		if (ntb)
			tab_text(&rep, pl, npl, G, RW, grp, lane_of, ngr);
		if (nrej) {
			sb_printf(&rep, "  not applied:\n");
			for (i = 0; i < npl; i++) {
				struct fxrw *r = NULL;
				int c;
				if (pl[i].applied || pl[i].kind == FXR_DIVNOTE ||
				    pl[i].kind == FXR_TABLE || approx_kind(pl[i].kind))
					continue;
				for (c = 0; c < pl[i].ng; c++)
					if (!fx_accept(pl[i].rw[c], &fopt))
						r = pl[i].rw[c];
				if (r == NULL)
					r = pl[i].rw[0];
				sb_printf(&rep, "    %s   (%s)\n", V.in[pl[i].k].text,
				    kind_text(pl[i].kind));
				if (inner[pl[i].k]) {
					sb_printf(&rep, "        %s\n", inner[pl[i].k] == 2 ?
					    "inside a product that is rewritten in the log domain" :
					    "inside a polynomial that is refitted as a whole");
					continue;
				}
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
		if (ntb)
			tab_files(pl, npl, RW);
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
	free(inner);
	free(decl.p);
	free(rep.p);
	free(lanes_txt.p);
	free(disc_txt.p);
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
#include "spv_mem.inc"	/* memory kinds: what their prices make worth doing */
#include "spv_sg.inc"	/* subgroups: run, priced, and written by simplify */
#include "spv_tex.inc"	/* constant table reads through the texture unit */

/* ---- advice: values that change less often than the shader runs ---- */

/*
 * A shader runs once a pixel (or an invocation), but part of what it
 * computes depends only on constants, uniforms and push constants: that
 * part is the same for a whole draw, and could be computed once and
 * passed in.  In a fragment shader, what is affine in the varyings
 * could be computed at the vertices and interpolated.  Neither can be
 * done inside the module (the interface changes), so `simplify` only
 * reports them, whenever it finds some (`advice` also says when it finds
 * none), priced under the active model: each instruction gets the
 * lowest frequency its operands allow,
 *
 *   constant < draw (uniform blocks, push constants) < primitive (flat
 *   inputs of a fragment shader: the provoking vertex's) < vertex
 *   (affine in non-flat float inputs of a fragment shader) < screen
 *   (affine in gl_FragCoord) < every run
 *
 * and the values of a lower frequency that something of a higher one
 * uses are what would be passed in.  Affine: sums, differences, and
 * products, quotients, dots, matrix products and mixes with every other
 * operand of draw or primitive frequency (the same at every pixel of a
 * primitive); anything else of a varying is per run.  gl_FragCoord is
 * the window position, the depth and 1/w, each affine across a
 * primitive in window coordinates: what is affine in it, with
 * coefficients of the draw, is a noperspective varying.  A value that
 * mixes it with the (perspective-correct) varyings is per run.
 */
enum { FQ_CONST, FQ_DRAW, FQ_PRIM, FQ_VERTEX, FQ_SCREEN, FQ_RUN };
static int advice_on;

void
spirv_advice_opt(int on)
{
	advice_on = on;
}

static int
adv_decorated(int id, const char *d)
{
	int k;

	for (k = 0; k < V.nin && strcmp(V.in[k].op, "OpFunction"); k++)
		if (!strcmp(V.in[k].op, "OpDecorate") && V.in[k].nop >= 2 &&
		    idx(V.in[k].opnd[0]) == id && !strcmp(V.in[k].opnd[1], d))
			return 1;
	return 0;
}

/* an instruction's price: float instructions per component, the rest by
 * their class */
static double
adv_price(const struct sins *I)
{
	const char *ext = !strcmp(I->op, "OpExtInst") && I->nop >= 3 ?
	    I->opnd[2] : NULL;
	int rt = idx(I->opnd[0]), cls;
	double mul = 1, c = fprice_i(I);

	if (c > 0)
		return c;
	cls = cost_class(I->op, ext, rt, &mul);
	return cls >= 0 && cls < C_NCLASS && cclass_available(model_active(),
	    cls) ? model_active()->c[cls] * mul : 0;
}

/* computes from its operands alone, the same on the host as here */
static int
adv_pure(const struct sins *I)
{
	static const char *const core[] = {"OpFAdd", "OpFSub", "OpFMul",
	    "OpFDiv", "OpFNegate", "OpFRem", "OpFMod", "OpIAdd", "OpISub",
	    "OpIMul", "OpUDiv", "OpSDiv", "OpUMod", "OpSMod", "OpSRem",
	    "OpSNegate", "OpShiftLeftLogical", "OpShiftRightLogical",
	    "OpShiftRightArithmetic", "OpBitwiseAnd", "OpBitwiseOr",
	    "OpBitwiseXor", "OpNot", "OpSelect", "OpIEqual", "OpINotEqual",
	    "OpConvertFToU", "OpConvertFToS", "OpConvertSToF", "OpConvertUToF",
	    "OpBitcast", "OpUConvert", "OpSConvert", "OpFConvert",
	    "OpCompositeConstruct", "OpCompositeExtract", "OpCompositeInsert",
	    "OpVectorShuffle", "OpCopyObject", "OpVectorTimesScalar",
	    "OpMatrixTimesScalar", "OpVectorTimesMatrix", "OpMatrixTimesVector",
	    "OpMatrixTimesMatrix", "OpDot", "OpTranspose", "OpLogicalAnd",
	    "OpLogicalOr", "OpLogicalNot", "OpLogicalEqual",
	    "OpLogicalNotEqual", NULL};

	if (!strcmp(I->op, "OpExtInst"))
		return I->nop >= 4 && V.glsl >= 0 && idx(I->opnd[1]) == V.glsl;
	return in_list(I->op, core) || !strncmp(I->op, "OpFOrd", 6) ||
	    !strncmp(I->op, "OpFUnord", 8) || !strncmp(I->op, "OpULess", 7) ||
	    !strncmp(I->op, "OpUGreater", 10) || !strncmp(I->op, "OpSLess", 7) ||
	    !strncmp(I->op, "OpSGreater", 10);
}

/* correctly rounded (or exact) in the shader as on a host: the rest is
 * within the precision Vulkan allows and may differ in the last places */
static int
adv_exact(const struct sins *I)
{
	static const char *const ext[] = {"FAbs", "FSign", "Floor", "Ceil",
	    "Trunc", "FMin", "FMax", "FClamp", "SAbs", "SSign", "SMin", "SMax",
	    "SClamp", "UMin", "UMax", "UClamp", "Step", NULL};

	if (!strcmp(I->op, "OpExtInst"))
		return in_list(I->opnd[2], ext);
	return strcmp(I->op, "OpFDiv") && strcmp(I->op, "OpFRem") &&
	    strcmp(I->op, "OpFMod");
}

/* an affine operation: at most one operand of frequency f (vertex or
 * screen), in a place where the result is linear in it */
static int
adv_affine(const struct sins *I, const uint8_t *fq, int f)
{
	int base = first_arg(I), i, nv = 0, at = -1;
	const char *e = !strcmp(I->op, "OpExtInst") ? I->opnd[2] : "";

	for (i = base; i < I->nop; i++)
		if (I->opnd[i][0] == '%' && fq[idx(I->opnd[i])] == f) {
			nv++;
			at = i - base;
		}
	if (!strcmp(I->op, "OpFAdd") || !strcmp(I->op, "OpFSub") ||
	    !strcmp(I->op, "OpFNegate") || !strcmp(I->op, "OpCompositeConstruct")
	    || !strcmp(I->op, "OpCompositeExtract") ||
	    !strcmp(I->op, "OpVectorShuffle") || !strcmp(I->op, "OpCopyObject"))
		return 1;
	if (!strcmp(I->op, "OpFMul") || !strcmp(I->op, "OpVectorTimesScalar") ||
	    !strcmp(I->op, "OpMatrixTimesVector") ||
	    !strcmp(I->op, "OpVectorTimesMatrix") || !strcmp(I->op, "OpDot") ||
	    !strcmp(I->op, "OpMatrixTimesScalar"))
		return nv == 1;
	if (!strcmp(I->op, "OpFDiv"))
		return nv == 1 && at == 0;
	if (!strcmp(e, "FMix"))		/* x (1 - a) + y a, a of the draw */
		return I->nop >= 6 && fq[idx(I->opnd[5])] <= FQ_DRAW;
	if (!strcmp(e, "Fma"))		/* a b + c: one of a, b of the draw */
		return I->nop >= 6 && (fq[idx(I->opnd[3])] <= FQ_DRAW ||
		    fq[idx(I->opnd[4])] <= FQ_DRAW);
	return 0;
}

struct advroot {
	int k;			/* the instruction */
	double cost;		/* of what it is computed from, at its frequency */
	int nins, nload, ninexact;
};

/* the instructions of frequency f under k, each once (seen) */
static void
adv_cone(int k, int f, const uint8_t *fq, char *seen, struct advroot *r,
    const double *price, int depth)
{
	const struct sins *I = &V.in[k];
	int i;

	if (seen[k] || depth > 400)
		return;
	seen[k] = 1;
	if (!strcmp(I->op, "OpLoad"))
		r->nload++;
	else if (price[k] > 0) {
		r->nins++;
		r->ninexact += !adv_exact(I);
	}
	r->cost += price[k];
	for (i = 1; i < I->nop; i++) {
		int a, d;
		if (I->opnd[i][0] != '%')
			continue;
		a = idx(I->opnd[i]);
		d = SP.def[a];
		if (d >= 0 && fq[a] == f)
			adv_cone(d, f, fq, seen, r, price, depth + 1);
	}
}

/* ---- advice: what the input assembler computes for a vertex shader ---- */

/*
 * The vertex shader's integer values as ucas expressions of its inputs:
 * a word an Input variable holds (or one component of it), and
 * gl_InstanceIndex, each a parameter; constants, shifts, masks,
 * arithmetic, bitcasts and bitfield extracts between them.
 */
struct vib {
	struct func f;
	int sid[MAXP], comp[MAXP], inst[MAXP];
	int fail;
};

static int
adv_builtin_is(int var, const char *name)
{
	int k;

	for (k = 0; k < V.nin && strcmp(V.in[k].op, "OpFunction"); k++)
		if (!strcmp(V.in[k].op, "OpDecorate") && V.in[k].nop >= 3 &&
		    idx(V.in[k].opnd[0]) == var && !strcmp(V.in[k].opnd[1],
		    "BuiltIn") && !strcmp(V.in[k].opnd[2], name))
			return 1;
	return 0;
}

static ty_t
vib_ty(int t)
{
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_INT ||
	    T(t)->w != 32)
		return 0;
	return TY(T(t)->sign ? K_SINT : K_UINT, 32);
}

static nid
vib_param(struct vib *B, int sid, int comp, ty_t ty, int inst)
{
	char nm[64];
	const char *n = V.id[sid].name;
	int j;

	for (j = 0; j < B->f.np; j++)
		if (B->sid[j] == sid && B->comp[j] == comp)
			return mk_cast(ty, mk_var((uint32_t)j, B->f.p[j].name,
			    B->f.p[j].ty));
	if (B->f.np == MAXP) {
		B->fail = 1;
		return mk_const_i(ty, 0);
	}
	j = B->f.np++;
	if (*n == '%')
		n++;
	if (inst)
		snprintf(nm, sizeof nm, "instance");
	else if (comp >= 0)
		snprintf(nm, sizeof nm, "%s_%c", isalpha((unsigned char)*n) ? n :
		    "w", "xyzw"[comp & 3]);
	else
		snprintf(nm, sizeof nm, "%s", isalpha((unsigned char)*n) ? n :
		    "w");
	memset(&B->f.p[j], 0, sizeof B->f.p[j]);
	B->f.p[j].name = sym_cstr(nm);
	B->f.p[j].ty = inst ? TY(K_SINT, 32) : TY(K_UINT, 32);
	if (inst) {	/* an instance's index is not negative */
		B->f.p[j].ranged = 1;
		B->f.p[j].lo = 0;
		B->f.p[j].hi = INT32_MAX;
	}
	B->sid[j] = sid;
	B->comp[j] = comp;
	B->inst[j] = inst;
	return mk_cast(ty, mk_var((uint32_t)j, B->f.p[j].name, B->f.p[j].ty));
}

/* a equals b on f's domain, proved (by enumeration or SAT) */
static int
adv_same(const struct func *f, nid a, nid b, const char **how)
{
	struct eqproof pr;
	jmp_buf saved;
	volatile int r = 0;

	memset(&pr, 0, sizeof pr);
	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		r = equiv_decide(f, a, b, NULL, &pr) == 0 && pr.how != EQ_SAMPLED;
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	notes_drop();
	if (r)
		*how = pr.how == EQ_EXHAUSTIVE ? "checked on every value" :
		    "proved by SAT";
	return r;
}

/* the Input variable a load reads, or -1 */
static int
vib_input(int ptr)
{
	int k;

	if (V.id[ptr].kind != K_VAR)
		return -1;
	for (k = 0; k < V.nin; k++)
		if (V.in[k].res == ptr && !strcmp(V.in[k].op, "OpVariable") &&
		    V.in[k].nop >= 2 && !strcmp(V.in[k].opnd[1], "Input"))
			return ptr;
	return -1;
}

static nid
vib_build(struct vib *B, int id, int depth)
{
	int k = SP.def[id], t = V.id[id].type, i;
	const struct sins *I;
	ty_t ty = vib_ty(t);

	if (ty == 0 || depth > 32) {
		B->fail = 1;
		return mk_const_i(TY(K_UINT, 32), 0);
	}
	if (V.id[id].kind == K_CONST && !V.id[id].spec)
		return mk_const_i(ty, geti(t, V.id[id].c[0]));
	if (k < 0) {
		B->fail = 1;
		return mk_const_i(ty, 0);
	}
	I = &V.in[k];
	if (!strcmp(I->op, "OpLoad") && I->nop >= 2) {
		int var = vib_input(idx(I->opnd[1]));
		if (var >= 0 && T(V.id[var].type)->e == t)
			return vib_param(B, var, -1, ty, adv_builtin_is(var,
			    "InstanceIndex"));
	}
	if (!strcmp(I->op, "OpCompositeExtract") && I->nop == 3) {
		int src = idx(I->opnd[1]), ks = SP.def[src];
		if (ks >= 0 && !strcmp(V.in[ks].op, "OpLoad")) {
			int var = vib_input(idx(V.in[ks].opnd[1]));
			if (var >= 0)
				return vib_param(B, var, atoi(I->opnd[2]), ty, 0);
		}
	}
	if (I->nop >= 3) {
		static const struct { const char *op; int o; } bin[] = {
			{"OpIAdd", O_ADD}, {"OpISub", O_SUB}, {"OpIMul", O_MUL},
			{"OpUDiv", O_DIV}, {"OpSDiv", O_DIV}, {"OpUMod", O_MOD},
			{"OpShiftLeftLogical", O_SHL}, {"OpShiftRightLogical", O_SHR},
			{"OpShiftRightArithmetic", O_SHR}, {"OpBitwiseAnd", O_AND},
			{"OpBitwiseOr", O_OR}, {"OpBitwiseXor", O_XOR}, {NULL, 0}};
		for (i = 0; bin[i].op; i++)
			if (!strcmp(I->op, bin[i].op)) {
				/* signed or unsigned by the instruction, as SPIR-V says */
				int sg = !strcmp(I->op, "OpSDiv") ||
				    !strcmp(I->op, "OpShiftRightArithmetic");
				ty_t ot = TY(sg ? K_SINT : K_UINT, 32);
				nid a = mk_cast(ot, vib_build(B, idx(I->opnd[1]), depth +
				    1));
				nid b = vib_build(B, idx(I->opnd[2]), depth + 1);
				if (bin[i].o != O_SHL && bin[i].o != O_SHR)
					b = mk_cast(ot, b);
				return mk_cast(ty, mk_bin(bin[i].o, a, b));
			}
	}
	if ((!strcmp(I->op, "OpBitcast") || !strcmp(I->op, "OpCopyObject")) &&
	    I->nop >= 2)
		return mk_cast(ty, vib_build(B, idx(I->opnd[1]), depth + 1));
	if ((!strcmp(I->op, "OpBitFieldUExtract") ||
	    !strcmp(I->op, "OpBitFieldSExtract")) && I->nop >= 4) {
		int o = idx(I->opnd[2]), c = idx(I->opnd[3]);
		if (V.id[o].kind == K_CONST && V.id[c].kind == K_CONST) {
			int64_t off = geti(V.id[o].type, V.id[o].c[0]);
			int64_t cnt = geti(V.id[c].type, V.id[c].c[0]);
			int sg = !strcmp(I->op, "OpBitFieldSExtract");
			ty_t ot = TY(sg ? K_SINT : K_UINT, 32);
			nid a = mk_cast(ot, vib_build(B, idx(I->opnd[1]), depth + 1));
			if (off >= 0 && cnt > 0 && off + cnt <= 32)
				/* (a << (32 - off - cnt)) >> (32 - cnt), arithmetic
				 * when signed */
				return mk_cast(ty, mk_bin(O_SHR, mk_bin(O_SHL, a,
				    mk_const_i(TY(K_UINT, 32), 32 - off - cnt)),
				    mk_const_i(TY(K_UINT, 32), 32 - cnt)));
		}
	}
	B->fail = 1;
	return mk_const_i(ty, 0);
}

/* what instruction k and the instructions under it cost, each once */
static double
adv_cone_price(int k, char *seen, int depth)
{
	const struct sins *I = &V.in[k];
	double c;
	int i;

	if (seen[k] || depth > 64 || !strcmp(I->op, "OpLoad"))
		return 0;
	seen[k] = 1;
	c = adv_price(I);
	for (i = first_arg(I); i < I->nop; i++)
		if (I->opnd[i][0] == '%' && SP.def[idx(I->opnd[i])] >= 0)
			c += adv_cone_price(SP.def[idx(I->opnd[i])], seen, depth + 1);
	return c;
}

/*
 * A vertex shader's values the input assembler could give it: a field of
 * an input word that a vertex format extracts (a byte, a half, signed or
 * not, and its conversion to float), and gl_InstanceIndex divided by a
 * constant, which an instance-rate attribute with that divisor fetches
 * from a buffer of 0, 1, 2, ...  Each is proved on its own (equal for
 * every word, by enumeration or SAT) and offered as a proposal: the
 * vertex input state changes.  Not priced: no model prices a vertex
 * attribute (the GT 1030 run measured that the format barely matters
 * and the attribute count does).
 */
static void
adv_vertex_inputs(struct sbuf *rep, const struct model *m)
{
	int k, n = 0, i;
	char *seen = xcalloc((size_t)V.nin + 1, 1);
	char sl[160];

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		struct vib *B;
		nid e, x;
		int w, matched = 0;
		char what[400] = "";
		const char *how = "";
		int B_inst;
		double fieldmax, cone;

		if (I->res < 0 || SP.def[I->res] != k || !vib_ty(V.id[I->res].type)
		    || !adv_pure(I) || !strncmp(I->op, "OpComposite", 11) ||
		    !strcmp(I->op, "OpCopyObject") || !strcmp(I->op, "OpBitcast"))
			continue;
		B = xcalloc(1, sizeof *B);
		B->f.name = sym_cstr("_vin");
		B->f.anon = 1;
		e = vib_build(B, I->res, 0);
		B->f.body = e;
		B_inst = B->inst[0];
		fieldmax = 0;
		if (B->fail || B->f.np != 1 || N.op[e] == O_VAR) {
			free(B);
			continue;
		}
		x = mk_var(0, B->f.p[0].name, B->f.p[0].ty);
		if (B->inst[0]) {
			/* instance / k: k read off the expression, then proved */
			nid q = e;
			while (N.op[q] == O_CAST)
				q = N.a[q];
			if (N.op[q] == O_DIV && N.op[N.b[q]] == O_CONST &&
			    N.ki[N.b[q]] >= 2 && N.ki[N.b[q]] <= INT32_MAX) {
				int64_t d = N.ki[N.b[q]];
				q = mk_cast(N.ty[e], mk_bin(O_DIV, x, mk_const_i(TY(K_SINT,
				    32), d)));
				if (adv_same(&B->f, e, q, &how)) {
					snprintf(what, sizeof what, "gl_InstanceIndex / %lld "
					    "(%s) is the element an instance-rate attribute "
					    "with divisor %lld fetches from a buffer holding 0, "
					    "1, 2, ... (VK_EXT_vertex_attribute_divisor, core "
					    "in Vulkan 1.4): it fetches element firstInstance + "
					    "(InstanceIndex - firstInstance) / divisor, which "
					    "is this when firstInstance is 0", (long long)d,
					    how, (long long)d);
					matched = 1;
				}
			}
		} else
			for (w = 8; w <= 16 && !matched; w += 8) {
				int c, sg;
				for (sg = 0; sg < 2 && !matched; sg++)
					for (c = 0; c < 32 / w && !matched; c++) {
						/* component c of the word read as w-bit fields */
						ty_t st = TY(sg ? K_SINT : K_UINT, 32);
						nid q = mk_bin(O_SHR, mk_bin(O_SHL, mk_cast(st, x),
						    mk_const_i(TY(K_UINT, 32), 32 - w * (c + 1))),
						    mk_const_i(TY(K_UINT, 32), 32 - w));
						q = mk_cast(N.ty[e], q);
						if (!adv_same(&B->f, e, q, &how))
							continue;
						fieldmax = sg ? (double)((1 << (w - 1)) - 1) :
						    (double)((1 << w) - 1);
						snprintf(what, sizeof what, "component %d of the "
						    "same bytes read as %s (%s): an attribute on %s's "
						    "binding and offset with that format, or %s at "
						    "offset + %d", c, w == 8 ? (sg ? "R8G8B8A8_SINT" :
						    "R8G8B8A8_UINT") : (sg ? "R16G16_SINT" :
						    "R16G16_UINT"), how, sym_str(B->f.p[0].name),
						    w == 8 ? (sg ? "R8_SINT" : "R8_UINT") : (sg ?
						    "R16_SINT" : "R16_UINT"), c * w / 8);
						matched = 1;
					}
			}
		free(B);
		if (!matched)
			continue;
		if (!rep->n)
			sb_printf(rep, "advice (changes outside the module; nothing "
			    "here is rewritten), priced under %s:\n",
			    sym_str(m->name));
		if (!n++) {
			sb_printf(rep, "  vertex inputs: what the input assembler "
			    "computes before the shader runs (each a proposal: the "
			    "vertex input state changes; not priced: %s has no price "
			    "for an attribute, and the GT 1030 run measured that the "
			    "format barely matters while the count does):\n",
			    sym_str(m->name));
			rep_list("advice_vertex_inputs");
		}
		memset(seen, 0, (size_t)V.nin + 1);
		cone = adv_cone_price(k, seen, 0);
		sb_printf(rep, "    %s   (%g every vertex)\n", I->text, cone);
		rep_obj(NULL);
		rep_str("instruction", I->text);
		rep_num("cost", cone);
		rep_int("line", ins_line(k));
		rep_str("input", what);
		rep_end();
		src_line(ins_line(k), sl, sizeof sl);
		if (sl[0])
			sb_printf(rep, "  %s", sl);
		sb_printf(rep, "      %s\n", what);
		if (B_inst)
			continue;
		/* the field converted to float: a SCALED or a NORM format */
		for (i = 0; i < V.nin; i++) {
			const struct sins *J = &V.in[i];
			int sg, q;
			if ((strcmp(J->op, "OpConvertUToF") && strcmp(J->op,
			    "OpConvertSToF")) || J->nop < 2 || J->res < 0 ||
			    idx(J->opnd[1]) != I->res)
				continue;
			sg = !strcmp(J->op, "OpConvertSToF");
			sb_printf(rep, "      and %s: the %s format of that component "
			    "gives it as a float, exactly (a field of at most 16 bits "
			    "is an f32 as it is)\n", J->text, sg ? "SSCALED" :
			    "USCALED");
			for (q = 0; q < V.nin; q++) {
				const struct sins *D = &V.in[q];
				double kd;
				int c, other;
				if (D->res < 0 || D->nop < 3 || (strcmp(D->op, "OpFDiv") &&
				    strcmp(D->op, "OpFMul")))
					continue;
				if (idx(D->opnd[1]) == J->res)
					other = idx(D->opnd[2]);
				else if (idx(D->opnd[2]) == J->res && !strcmp(D->op,
				    "OpFMul"))
					other = idx(D->opnd[1]);
				else
					continue;
				c = other;
				if (V.id[c].kind != K_CONST || V.id[c].spec)
					continue;
				kd = getf(V.id[c].type, V.id[c].c[0]);
				/* over the field's largest value, or times the f32
				 * nearest its inverse (what a compiler makes of it) */
				if (!strcmp(D->op, "OpFDiv") ? kd != fieldmax : kd !=
				    (double)(float)(1.0 / fieldmax))
					continue;
				sb_printf(rep, "      and %s: the %s format gives the field "
				    "over %g as a real number; the shader's %s and the "
				    "format's conversion may differ in the last bit\n",
				    D->text, sg ? "SNORM" : "UNORM", fieldmax,
				    !strcmp(D->op, "OpFDiv") ? "division" : "product");
			}
		}
	}
	if (n)
		rep_end();
	free(seen);
}

/* ---- advice: a discard the clipper could do ---- */

/* the cone under id reads an Input decorated d (a NoPerspective one) */
static int
adv_reads_decorated(int id, const char *d, int depth)
{
	int k = SP.def[id], i;
	const struct sins *I;

	if (k < 0 || depth > 64)
		return 0;
	I = &V.in[k];
	if (!strcmp(I->op, "OpLoad") || !strcmp(I->op, "OpAccessChain") ||
	    !strcmp(I->op, "OpInBoundsAccessChain")) {
		int chain, var = mem_base_var(idx(I->opnd[1]), &chain);
		if (var >= 0 && adv_decorated(var, d))
			return 1;
	}
	for (i = first_arg(I); i < I->nop; i++)
		if (I->opnd[i][0] == '%' && adv_reads_decorated(idx(I->opnd[i]),
		    d, depth + 1))
			return 1;
	return 0;
}

/* the instruction of the module that does something besides writing
 * its outputs (a store to memory others see, an atomic, an image
 * write): its index, -1 none */
static int
adv_effect(void)
{
	int k;

	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strncmp(I->op, "OpAtomic", 8) || !strcmp(I->op,
		    "OpImageWrite"))
			return k;
		if (!strcmp(I->op, "OpStore") && I->nop >= 1) {
			int chain, var = mem_base_var(idx(I->opnd[0]), &chain);
			const char *sc = var >= 0 ? mem_sc(var) : "";
			if (var < 0 || (strcmp(sc, "Output") && strcmp(sc, "Function")
			    && strcmp(sc, "Private")))
				return k;
		}
	}
	return -1;
}

/*
 * A fragment shader's discard on the sign of a value affine in the
 * perspective-correct varyings, compared with a value of the draw: the
 * vertex shader could write the difference as gl_ClipDistance, which is
 * interpolated in clip space as those varyings are, and the clipper
 * removes the part where it is below 0, so no fragment runs there.
 * Only for a shader whose sole effect is its outputs: a clipped fragment
 * never runs, a discarded one ran up to the discard.  Not priced: no
 * model prices a fragment's whole run or the clipper.
 */
static void
adv_clip(struct sbuf *rep, const struct model *m, const uint8_t *fq)
{
	int k, j, n = 0, eff = -2;
	char sl[160];

	for (k = 1; k < V.nin; k++) {
		const struct sins *I = &V.in[k], *C;
		const char *lab, *why = NULL;
		int br = -1, kill_true = 0, c, a, b, va, sign, l = k - 1;
		if (strcmp(I->op, "OpKill") && strcmp(I->op, "OpTerminateInvocation"))
			continue;
		while (l > 0 && mem_debug_op(V.in[l].op))
			l--;	/* OpLine and the like, which -g keeps */
		if (strcmp(V.in[l].op, "OpLabel") || V.in[l].res < 0)
			continue;	/* a block that only discards */
		lab = V.id[V.in[l].res].tok;
		for (j = 0; j < V.nin && br < 0; j++)
			if (!strcmp(V.in[j].op, "OpBranchConditional") &&
			    V.in[j].nop >= 3 && (!strcmp(V.in[j].opnd[1], lab) ||
			    !strcmp(V.in[j].opnd[2], lab)))
				br = j;
		if (br < 0)
			continue;
		kill_true = !strcmp(V.in[br].opnd[1], lab);
		c = SP.def[idx(V.in[br].opnd[0])];
		if (c < 0)
			continue;
		C = &V.in[c];
		if (C->nop < 3 || (strncmp(C->op, "OpFOrd", 6) && strncmp(C->op,
		    "OpFUnord", 8)) || (!strstr(C->op, "LessThan") &&
		    !strstr(C->op, "GreaterThan")))
			continue;
		a = idx(C->opnd[1]);
		b = idx(C->opnd[2]);
		if (fq[a] == FQ_VERTEX && fq[b] <= FQ_DRAW)
			va = a;
		else if (fq[b] == FQ_VERTEX && fq[a] <= FQ_DRAW)
			va = b;
		else
			continue;
		if (eff == -2)
			eff = adv_effect();
		if (adv_reads_decorated(va, "NoPerspective", 0))
			why = "the value reads a noperspective varying, and a clip "
			    "distance is interpolated in clip space";
		else if (eff >= 0)
			why = "the shader does more than write its outputs (a store "
			    "others see, an atomic or an image write), which a clipped "
			    "fragment would not do";
		/* kept where a >= b (or a > b): the distance a - b; else b - a */
		sign = strstr(C->op, "LessThan") != NULL ? 1 : -1;
		if (!kill_true)
			sign = -sign;
		if (!rep->n)
			sb_printf(rep, "advice (changes outside the module; nothing "
			    "here is rewritten), priced under %s:\n",
			    sym_str(m->name));
		if (!n++) {
			sb_printf(rep, "  discards the clipper could do (each a "
			    "proposal: the vertex shader writes gl_ClipDistance and no "
			    "fragment runs where it is below 0; not priced: %s has no "
			    "price for a fragment's whole run):\n", sym_str(m->name));
			rep_list("advice_clip");
		}
		sb_printf(rep, "    %s\n", C->text);
		src_line(ins_line(c), sl, sizeof sl);
		if (sl[0])
			sb_printf(rep, "  %s", sl);
		rep_obj(NULL);
		rep_str("instruction", C->text);
		rep_int("line", ins_line(c));
		if (why) {
			sb_printf(rep, "      not offered: %s\n", why);
			rep_str("not_offered", why);
		} else {
			sb_printf(rep, "      the fragment is discarded where it %s, %s "
			    "affine in the perspective-correct varyings and %s a value "
			    "of the draw: gl_ClipDistance[i] = %s - %s at each vertex keeps "
			    "the rest (the plane where they are equal aside), since a "
			    "clip distance is interpolated in clip space as those "
			    "varyings are; equal as real functions, the clipper's "
			    "arithmetic and the interpolation's rounding apart\n",
			    kill_true ? "holds" : "fails", V.id[va].tok, V.id[va == a ?
			    b : a].tok, sign > 0 ? V.id[a].tok : V.id[b].tok, sign > 0 ?
			    V.id[b].tok : V.id[a].tok);
			rep_str("clip_distance", "offered");
		}
		rep_end();
	}
	if (n)
		rep_end();
}

/*
 * A point sprite's coordinate (#203, Liz: as extra advice): (gl_FragCoord
 * - c) / s + 1/2, of x, y or the two, with c and s values of one
 * primitive (flat inputs, the draw's, constants; s may be a multiply by
 * its reciprocal), is gl_PointCoord when the draw is points of size s
 * centred at c: the rasteriser makes it for each pixel, and the
 * subtract, the divide and the add go.  Vulkan's gl_FragCoord and
 * gl_PointCoord both have their origin at the upper left, so y needs no
 * flip.
 */
enum { PC_NONE, PC_PTR, PC_FC, PC_DIFF, PC_SCALED };

/* a value of one primitive, or a vector of copies of one */
static int
pc_prim(const uint8_t *fq, int id)
{
	const struct sins *I;
	int j, a;

	if (fq[id] <= FQ_PRIM)
		return 1;
	if (SP.def[id] < 0)
		return 0;
	I = &V.in[SP.def[id]];
	if (strcmp(I->op, "OpCompositeConstruct") || I->nop < 2 ||
	    I->opnd[1][0] != '%')
		return 0;
	a = idx(I->opnd[1]);
	for (j = 2; j < I->nop; j++)
		if (I->opnd[j][0] != '%' || idx(I->opnd[j]) != a)
			return 0;
	return fq[a] <= FQ_PRIM;
}

/* the f32 constant 1/2, or a vector of them */
static int
pc_half(int id)
{
	int j;

	if (V.id[id].kind != K_CONST || V.id[id].spec || V.id[id].nc < 1)
		return 0;
	for (j = 0; j < V.id[id].nc; j++)
		if (V.id[id].c[j] != 0x3f000000u)
			return 0;
	return 1;
}

static int rk_const(const char *opnd, int *v);

/* a value as the variable it loads ("%center", "%pc[0]"), else its id */
static const char *
pc_name(int id, char *b, size_t n)
{
	const struct sins *L, *C;
	int p, j;

	/* a vector of copies of one value: that value */
	if (SP.def[id] >= 0 && !strcmp(V.in[SP.def[id]].op,
	    "OpCompositeConstruct") && V.in[SP.def[id]].nop >= 2 &&
	    V.in[SP.def[id]].opnd[1][0] == '%')
		id = idx(V.in[SP.def[id]].opnd[1]);
	snprintf(b, n, "%s", V.id[id].tok);
	if (SP.def[id] < 0 || strcmp((L = &V.in[SP.def[id]])->op, "OpLoad") ||
	    L->nop < 2 || L->opnd[1][0] != '%')
		return b;
	p = idx(L->opnd[1]);
	if (SP.def[p] < 0)
		snprintf(b, n, "%s", V.id[p].tok);
	else if (!strcmp((C = &V.in[SP.def[p]])->op, "OpAccessChain") &&
	    C->nop == 3 && C->opnd[1][0] == '%' && rk_const(C->opnd[2], &j))
		snprintf(b, n, "%s[%d]", V.id[idx(C->opnd[1])].tok, j);
	return b;
}

static void
adv_pointcoord(struct sbuf *rep, const struct model *m, const uint8_t *fq)
{
	static const char *const comp[4] = {"", ".x", ".y", ".xy"};
	int nid = V.nid + 64, *kind, *mask, *cen, *size, *mul, *ksub, *kscl, k,
	    n = 0;
	char sl[160], cn[80], zn[80];

	kind = xcalloc((size_t)nid, sizeof *kind);
	mask = xcalloc((size_t)nid, sizeof *mask);
	cen = xcalloc((size_t)nid, sizeof *cen);
	size = xcalloc((size_t)nid, sizeof *size);
	mul = xcalloc((size_t)nid, sizeof *mul);
	ksub = xcalloc((size_t)nid, sizeof *ksub);
	kscl = xcalloc((size_t)nid, sizeof *kscl);
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int r = I->res, a, b, j, d, o, sc, h;
		double save;
		if (r < 0 || r >= nid || I->nop < 2 || I->opnd[1][0] != '%')
			continue;
		a = idx(I->opnd[1]);
		b = I->nop >= 3 && I->opnd[2][0] == '%' ? idx(I->opnd[2]) : -1;
		if (!strcmp(I->op, "OpLoad") && (adv_builtin_is(a,
		    "FragCoord") || kind[a] == PC_PTR)) {
			kind[r] = PC_FC;
			mask[r] = kind[a] == PC_PTR ? mask[a] : 15;
		} else if ((!strcmp(I->op, "OpAccessChain") ||
		    !strcmp(I->op, "OpInBoundsAccessChain")) && I->nop == 3 &&
		    adv_builtin_is(a, "FragCoord") && rk_const(I->opnd[2],
		    &j) && (j == 0 || j == 1)) {
			/* a pointer to gl_FragCoord.x or .y */
			kind[r] = PC_PTR;
			mask[r] = 1 << j;
		} else if (!strcmp(I->op, "OpCompositeExtract") && kind[a] ==
		    PC_FC && I->nop == 3 && sscanf(I->opnd[2], "%d", &j) == 1 &&
		    (j == 0 || j == 1) && (mask[a] >> j & 1)) {
			kind[r] = PC_FC;
			mask[r] = 1 << j;
		} else if (!strcmp(I->op, "OpVectorShuffle") && kind[a] ==
		    PC_FC && I->nop == 5 && b == a && !strcmp(I->opnd[3], "0") &&
		    !strcmp(I->opnd[4], "1")) {
			kind[r] = PC_FC;
			mask[r] = 3;
		} else if (!strcmp(I->op, "OpFSub") && b >= 0 && kind[a] ==
		    PC_FC && mask[a] <= 3 && pc_prim(fq, b)) {
			kind[r] = PC_DIFF;
			mask[r] = mask[a];
			cen[r] = b;
			ksub[r] = k;
		} else if ((!strcmp(I->op, "OpFDiv") || !strcmp(I->op,
		    "OpFMul") || !strcmp(I->op, "OpVectorTimesScalar")) &&
		    b >= 0) {
			/* (fc - c) / s, or times 1/s in either order */
			d = kind[a] == PC_DIFF ? a : b;
			o = d == a ? b : a;
			if (kind[d] != PC_DIFF || !pc_prim(fq, o) ||
			    (!strcmp(I->op, "OpFDiv") && d != a))
				continue;
			kind[r] = PC_SCALED;
			mask[r] = mask[d];
			cen[r] = cen[d];
			size[r] = o;
			mul[r] = strcmp(I->op, "OpFDiv") != 0;
			ksub[r] = ksub[d];
			kscl[r] = k;
		} else if (!strcmp(I->op, "OpFAdd") && b >= 0) {
			sc = kind[a] == PC_SCALED ? a : b;
			h = sc == a ? b : a;
			if (kind[sc] != PC_SCALED || !pc_half(h))
				continue;
			save = adv_price(I) + adv_price(&V.in[kscl[sc]]) +
			    adv_price(&V.in[ksub[sc]]);
			if (!rep->n)
				sb_printf(rep, "advice (changes outside the "
				    "module; nothing here is rewritten), priced under "
				    "%s:\n", sym_str(m->name));
			if (!n++) {
				sb_printf(rep, "  point sprite coordinates the rasteriser "
				    "makes (each a proposal: the draw as points, "
				    "gl_PointSize the size in the vertex shader and the "
				    "point at the centre, within the device's "
				    "pointSizeRange; the shader reads gl_PointCoord):\n");
				rep_list("advice_pointcoord");
			}
			sb_printf(rep, "    %s\n", I->text);
			src_line(ins_line(k), sl, sizeof sl);
			if (sl[0])
				sb_printf(rep, "  %s", sl);
			pc_name(cen[sc], cn, sizeof cn);
			pc_name(size[sc], zn, sizeof zn);
			sb_printf(rep, "      (gl_FragCoord%s - %s) %s %s + 0.5 is "
			    "gl_PointCoord%s for points of size %s%s centred at %s: "
			    "%g a pixel less (the subtract, the %s and the add)\n",
			    comp[mask[sc]], cn, mul[sc] ? "*" : "/", zn,
			    comp[mask[sc]], mul[sc] ? "1 / " : "", zn, cn, save,
			    mul[sc] ? "multiply" : "divide");
			rep_obj(NULL);
			rep_str("instruction", I->text);
			rep_int("line", ins_line(k));
			rep_str("components", comp[mask[sc]] + 1);
			rep_str("centre", cn);
			rep_str(mul[sc] ? "reciprocal_size" : "size", zn);
			rep_num("saves", save);
			rep_end();
		}
	}
	if (n)
		rep_end();
	free(kind);
	free(mask);
	free(cen);
	free(size);
	free(mul);
	free(ksub);
	free(kscl);
}

/*
 * A fragment shader's inputs, recognised and reported, never proposed
 * (#202, Liz: report what is recognised, nothing more): flat inputs, the
 * provoking vertex's value, one a primitive; per-vertex inputs
 * (PerVertexKHR), the three vertices' values; the barycentric builtins;
 * and a sum over the three vertices of a per-vertex input's value times
 * the barycentric of that vertex, which is the interpolation the
 * rasteriser does for a smooth varying (gl_BaryCoordEXT) or a
 * noperspective one (gl_BaryCoordNoPerspEXT).
 */
enum { RK_NONE, RK_BARYV, RK_BARYC, RK_PVARR, RK_PV, RK_TERM };

static int
rk_const(const char *opnd, int *v)
{
	int id;

	if (opnd[0] != '%')
		return 0;
	id = idx(opnd);
	if (V.id[id].kind != K_CONST || V.id[id].spec)
		return 0;
	*v = (int)V.id[id].c[0];
	return 1;
}

static void
rk_name(struct sbuf *b, int *n, int id)
{
	sb_printf(b, "%s%s", (*n)++ ? ", " : "", V.id[id].tok);
}

static void
adv_recognise(int body)
{
	int nid = V.nid + 64, *kind, *var, *vk, *np, *mask, *part, k, nfl = 0,
	    npv = 0, nba = 0, nit = 0;
	struct sbuf fl = {0}, pv = {0}, ba = {0}, it = {0};

	kind = xcalloc((size_t)nid, sizeof *kind);
	var = xcalloc((size_t)nid, sizeof *var);
	vk = xcalloc((size_t)nid, sizeof *vk);
	np = xcalloc((size_t)nid, sizeof *np);
	mask = xcalloc((size_t)nid, sizeof *mask);
	/* the indices below the vertex's (a component, a member), coded:
	 * the terms of one interpolation read the same part of each vertex */
	part = xcalloc((size_t)nid, sizeof *part);
	for (k = 0; k < V.nin && k != body; k++) {
		const struct sins *I = &V.in[k];
		if (strcmp(I->op, "OpVariable") || I->res < 0 || I->nop < 2 ||
		    strcmp(I->opnd[1], "Input"))
			continue;
		if (adv_decorated(I->res, "Flat"))
			rk_name(&fl, &nfl, I->res);
		if (adv_decorated(I->res, "PerVertexKHR") ||
		    adv_decorated(I->res, "PerVertexNV")) {
			rk_name(&pv, &npv, I->res);
			kind[I->res] = RK_PVARR;
			var[I->res] = I->res;
			vk[I->res] = -1;
		}
		if (adv_builtin_is(I->res, "BaryCoordKHR") ||
		    adv_builtin_is(I->res, "BaryCoordNV") ||
		    adv_builtin_is(I->res, "BaryCoordNoPerspKHR") ||
		    adv_builtin_is(I->res, "BaryCoordNoPerspNV")) {
			rk_name(&ba, &nba, I->res);
			sb_printf(&ba, " (%s)", adv_builtin_is(I->res,
			    "BaryCoordNoPerspKHR") || adv_builtin_is(I->res,
			    "BaryCoordNoPerspNV") ? "noperspective" :
			    "perspective-correct");
			kind[I->res] = RK_BARYV;
			np[I->res] = adv_builtin_is(I->res, "BaryCoordNoPerspKHR") ||
			    adv_builtin_is(I->res, "BaryCoordNoPerspNV");
		}
	}
	for (k = body < 0 ? V.nin : body; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int r = I->res, a, b, c;
		if (r < 0 || r >= nid || I->nop < 2 || I->opnd[1][0] != '%')
			continue;
		a = idx(I->opnd[1]);
		if (!strcmp(I->op, "OpLoad") || !strcmp(I->op, "OpCopyObject")) {
			kind[r] = kind[a];
			var[r] = var[a];
			vk[r] = vk[a];
			np[r] = np[a];
			mask[r] = mask[a];
			part[r] = part[a];
		} else if (!strcmp(I->op, "OpAccessChain") ||
		    !strcmp(I->op, "OpInBoundsAccessChain") ||
		    !strcmp(I->op, "OpCompositeExtract")) {
			int lit = !strcmp(I->op, "OpCompositeExtract"), j, ok = 1, pt;
			if (I->nop < 3 || !(lit ? sscanf(I->opnd[2], "%d", &c) == 1 :
			    rk_const(I->opnd[2], &c)))
				continue;
			/* a barycentric's component c; vertex c of a per-vertex
			 * input, then the part of it the indices after name */
			if (kind[a] == RK_BARYV) {
				if (I->nop == 3) {
					kind[r] = RK_BARYC;
					vk[r] = c;
					np[r] = np[a];
				}
				continue;
			}
			if (kind[a] != RK_PVARR && kind[a] != RK_PV)
				continue;
			j = vk[a] < 0 ? 3 : 2;
			pt = part[a];
			for (; j < I->nop && ok; j++) {
				int d;
				ok = (lit ? sscanf(I->opnd[j], "%d", &d) == 1 :
				    rk_const(I->opnd[j], &d)) && d >= 0 && d < 63 &&
				    pt < (1 << 24);
				if (ok)
					pt = pt * 64 + d + 1;
			}
			if (!ok)
				continue;
			kind[r] = kind[a];
			var[r] = var[a];
			vk[r] = vk[a] < 0 ? c : vk[a];
			part[r] = pt;
			/* a load of vertex c's pointer is its value */
		} else if ((!strcmp(I->op, "OpFMul") ||
		    !strcmp(I->op, "OpVectorTimesScalar")) && I->nop >= 3 &&
		    I->opnd[2][0] == '%') {
			int p, q;
			b = idx(I->opnd[2]);
			p = kind[a] == RK_BARYC ? a : b;
			q = p == a ? b : a;
			if (kind[p] == RK_BARYC && (kind[q] == RK_PV ||
			    kind[q] == RK_PVARR) && vk[q] == vk[p]) {
				kind[r] = RK_TERM;
				var[r] = var[q];
				mask[r] = 1 << vk[p];
				np[r] = np[p];
				part[r] = part[q];
			}
		} else if (!strcmp(I->op, "OpFAdd") && I->nop >= 3 &&
		    I->opnd[2][0] == '%') {
			b = idx(I->opnd[2]);
			if (kind[a] == RK_TERM && kind[b] == RK_TERM && var[a] ==
			    var[b] && part[a] == part[b] && np[a] == np[b] &&
			    !(mask[a] & mask[b])) {
				kind[r] = RK_TERM;
				var[r] = var[a];
				np[r] = np[a];
				mask[r] = mask[a] | mask[b];
				part[r] = part[a];
				if (mask[r] == 7) {
					sb_printf(&it, "%sthe interpolation of %s%s by them",
					    nit++ ? ", " : "", part[r] ? "a part of " : "",
					    V.id[var[r]].tok);
					if (ins_line(k) > 0)
						sb_printf(&it, " at line %d", ins_line(k));
					else
						sb_printf(&it, " in %s", V.id[r].tok);
					sb_printf(&it, " (what a %s varying is)", np[r] ?
					    "noperspective" : "smooth");
				}
			}
		}
		/* a load through a per-vertex element's pointer is that
		 * vertex's value */
		if (!strcmp(I->op, "OpLoad") && kind[r] == RK_PVARR && vk[r] >= 0)
			kind[r] = RK_PV;
	}
	if (nfl || npv || nba || nit) {
		printf("recognised in the inputs (nothing is proposed for them):");
		rep_obj("recognised");
		if (nfl) {
			printf(" flat %s (the provoking vertex's value, one a "
			    "primitive)%s", sb_str(&fl), npv || nba || nit ? ";" : "");
			rep_str("flat", sb_str(&fl));
		}
		if (npv) {
			printf(" per-vertex %s (the three vertices' values)%s",
			    sb_str(&pv), nba || nit ? ";" : "");
			rep_str("per_vertex", sb_str(&pv));
		}
		if (nba) {
			printf(" the barycentrics %s%s", sb_str(&ba), nit ? ";" : "");
			rep_str("barycentrics", sb_str(&ba));
		}
		if (nit) {
			printf(" %s", sb_str(&it));
			rep_str("interpolations", sb_str(&it));
		}
		printf("\n");
		rep_end();
	}
	free(fl.p);
	free(pv.p);
	free(ba.p);
	free(it.p);
	free(kind);
	free(var);
	free(vk);
	free(np);
	free(mask);
	free(part);
}

static void
advice_pass(const char *text, int asked)
{
	const struct model *m = model_active();
	uint8_t *fq, *used;
	double *price;
	char *seen, stage[24] = "";
	struct advroot root[4][64];
	int nroot[4] = {0, 0, 0, 0}, k, i, f, body = -1, mix;
	struct sbuf rep = {0};

	memset(&V, 0, sizeof V);
	symmap_reset();
	V.glsl = -1;
	V.lanes = 1;
	V.rng = 1;
	cols = 1;
	parse(text);
	declare();
	fq = xcalloc((size_t)(V.nid + 64), 1);
	used = xcalloc((size_t)(V.nid + 64), 1);
	price = xcalloc((size_t)V.nin + 1, sizeof *price);
	seen = xcalloc((size_t)V.nin + 1, 1);
	SP.def = xmalloc((size_t)(V.nid + 64) * sizeof *SP.def);
	for (i = 0; i < V.nid + 64; i++) {
		SP.def[i] = -1;
		fq[i] = FQ_RUN;
	}
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (!strcmp(I->op, "OpEntryPoint") && I->nop >= 1 && !stage[0])
			snprintf(stage, sizeof stage, "%s", I->opnd[0]);
		if (!strcmp(I->op, "OpFunction") && body < 0)
			body = k;
	}
	if (!strcmp(stage, "Fragment"))
		adv_recognise(body);
	/* constants, and the variables by where their value comes from */
	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_CONST && !V.id[i].spec)
			fq[i] = FQ_CONST;
		else if (V.id[i].kind == K_CONST || V.id[i].kind == K_TYPE ||
		    V.id[i].kind == K_EXT)
			fq[i] = V.id[i].kind == K_CONST ? FQ_DRAW : FQ_CONST;
	for (k = 0; k < V.nin && k != body; k++) {
		const struct sins *I = &V.in[k];
		int t, e;
		if (strcmp(I->op, "OpVariable") || I->res < 0 || I->nop < 2)
			continue;
		t = idx(I->opnd[0]);
		e = V.id[t].kind == K_TYPE ? T(t)->e : -1;
		if (!strcmp(I->opnd[1], "PushConstant") ||
		    (!strcmp(I->opnd[1], "Uniform") && e >= 0 &&
		    !adv_decorated(e, "BufferBlock")))
			fq[I->res] = FQ_DRAW;
		else if (!strcmp(I->opnd[1], "Input") && !strcmp(stage, "Fragment")
		    && adv_decorated(I->res, "Flat"))
			fq[I->res] = FQ_PRIM;
		else if (!strcmp(I->opnd[1], "Input") && !strcmp(stage, "Fragment")
		    && V.id[I->res].builtin == B_FRAGCOORD)
			fq[I->res] = FQ_SCREEN;
		else if (!strcmp(I->opnd[1], "Input") && !strcmp(stage, "Fragment")
		    && e >= 0 && is_float_t(e) && !adv_decorated(I->res, "Flat") &&
		    !adv_decorated(I->res, "BuiltIn") &&
		    !adv_decorated(I->res, "Centroid") &&
		    !adv_decorated(I->res, "Sample"))
			fq[I->res] = FQ_VERTEX;
	}
	for (k = body < 0 ? V.nin : body; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int base, c = FQ_CONST;
		if (I->res < 0 || I->nop < 1 || I->opnd[0][0] != '%' ||
		    V.id[idx(I->opnd[0])].kind != K_TYPE)
			continue;
		SP.def[I->res] = k;
		V.id[I->res].type = idx(I->opnd[0]);
		if (!strcmp(I->op, "OpAccessChain") ||
		    !strcmp(I->op, "OpInBoundsAccessChain") ||
		    !strcmp(I->op, "OpLoad")) {
			/* a pointer into a variable, or its value: the variable's
			 * frequency (a function's own variable: unknown stores) */
			for (i = 1; i < I->nop; i++)
				if (I->opnd[i][0] == '%' && fq[idx(I->opnd[i])] > c)
					c = fq[idx(I->opnd[i])];
			fq[I->res] = (uint8_t)c;
			if (!strcmp(I->op, "OpLoad") && c == FQ_DRAW)
				price[k] = m->c[C_LOAD];
			continue;
		}
		if (!adv_pure(I))
			continue;	/* per run */
		base = first_arg(I);
		mix = 0;
		for (i = base; i < I->nop; i++)
			if (I->opnd[i][0] == '%') {
				int q = fq[idx(I->opnd[i])];
				mix |= 1 << q;
				if (q > c)
					c = q;
			}
		/* a flat value beside a varying is the provoking vertex's at
		 * every pixel, where each vertex would use its own */
		if (c == FQ_VERTEX && ((mix & 1 << FQ_PRIM) ||
		    !adv_affine(I, fq, FQ_VERTEX)))
			c = FQ_RUN;
		/* in window coordinates the coefficients are the draw's: a flat
		 * value or a varying beside it is not affine in it */
		if (c == FQ_SCREEN && ((mix & (1 << FQ_PRIM | 1 << FQ_VERTEX)) ||
		    !adv_affine(I, fq, FQ_SCREEN)))
			c = FQ_RUN;
		fq[I->res] = (uint8_t)c;
		if (c == FQ_DRAW || c == FQ_PRIM || c == FQ_VERTEX || c == FQ_SCREEN)
			price[k] = adv_price(I);
	}
	/* the highest frequency that uses each value */
	for (k = body < 0 ? V.nin : body; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		int mine = I->res >= 0 && SP.def[I->res] == k ? fq[I->res] : FQ_RUN;
		for (i = 0; i < I->nop; i++)
			if (I->opnd[i][0] == '%' && idx(I->opnd[i]) != I->res &&
			    mine > used[idx(I->opnd[i])])
				used[idx(I->opnd[i])] = (uint8_t)mine;
	}
	/* the roots: a draw or vertex value that something more frequent
	 * uses, and that costs something to compute */
	for (k = body < 0 ? V.nin : body; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		struct advroot r;
		if (I->res < 0 || SP.def[I->res] != k)
			continue;
		f = fq[I->res];
		if ((f != FQ_DRAW && f != FQ_PRIM && f != FQ_VERTEX && f !=
		    FQ_SCREEN) || used[I->res] <= f)
			continue;
		memset(&r, 0, sizeof r);
		memset(seen, 0, (size_t)V.nin + 1);
		r.k = k;
		adv_cone(k, f, fq, seen, &r, price, 0);
		if (r.nins == 0 || nroot[f - FQ_DRAW] >= 64)
			continue;
		root[f - FQ_DRAW][nroot[f - FQ_DRAW]++] = r;
	}
	for (f = FQ_DRAW; f <= FQ_SCREEN; f++) {
		int z = f - FQ_DRAW, n = nroot[z], nins = 0, nload = 0, ninx = 0;
		double tot = 0, arith, pass, save;
		struct advroot all;
		if (n == 0)
			continue;
		/* everything under the roots, each instruction once */
		memset(&all, 0, sizeof all);
		memset(seen, 0, (size_t)V.nin + 1);
		for (i = 0; i < n; i++)
			adv_cone(root[z][i].k, f, fq, seen, &all, price, 0);
		tot = all.cost;
		nins = all.nins;
		nload = all.nload;
		ninx = all.ninexact;
		arith = tot - nload * m->c[C_LOAD];
		pass = n * m->c[C_LOAD];
		save = tot - pass;
		if (!rep.n)
			sb_printf(&rep, "advice (changes outside the module; nothing "
			    "here is rewritten), priced under %s:\n", sym_str(m->name));
		rep_obj(f == FQ_DRAW ? "advice_draw" : f == FQ_PRIM ?
		    "advice_primitive" : f == FQ_VERTEX ? "advice_vertex" :
		    "advice_screen");
		rep_int("instructions", nins);
		rep_num("cost", tot);
		rep_int("values", n);
		if (f == FQ_DRAW) {
			sb_printf(&rep, "  once a draw: %d instruction%s that read%s "
			    "only constants, uniform blocks and push constants cost%s "
			    "%g every run (%g of arithmetic, %d load%s at %g)\n", nins,
			    nins == 1 ? "" : "s", nins == 1 ? "s" : "", nins == 1 ?
			    "s" : "", tot, arith, nload, nload == 1 ? "" : "s",
			    (double)m->c[C_LOAD]);
			if (save > 0)
				sb_printf(&rep, "    computed once a draw and passed in "
				    "as %d value%s (push constants or a uniform block: "
				    "%d load%s, %g), that saves %g every run\n", n, n == 1 ?
				    "" : "s", n, n == 1 ? "" : "s", pass, save);
			else
				sb_printf(&rep, "    derived, but not cheaper under %s: "
				    "passing %d value%s costs %d load%s (%g) against "
				    "%g\n", sym_str(m->name), n, n == 1 ? "" : "s", n,
				    n == 1 ? "" : "s", pass, tot);
			rep_num("saves", save);
			sb_printf(&rep, "    the same instructions on the same values: "
			    "add, subtract, multiply, conversions and selections are "
			    "exact or correctly rounded on a host as in the shader");
			if (ninx)
				sb_printf(&rep, "; %d of them (divisions, square roots, "
				    "transcendentals) are only within Vulkan's precision "
				    "in the shader and may differ from a host's in the "
				    "last places", ninx);
			sb_puts(&rep, "\n");
			rep_int("within_vulkan_precision", ninx);
		} else if (f == FQ_PRIM) {
			sb_printf(&rep, "  once a primitive: %d instruction%s costing "
			    "%g every pixel read%s only flat inputs and values of the "
			    "draw\n", nins, nins == 1 ? "" : "s", tot, nins == 1 ? "s" :
			    "");
			sb_printf(&rep, "    a flat input is the provoking vertex's "
			    "output, the same at every pixel of the primitive: "
			    "computed in the vertex shader and passed as %d flat "
			    "output%s, each pixel reads the value it computes now "
			    "(no interpolation)\n", n, n == 1 ? "" : "s");
			sb_printf(&rep, "    not priced: %s has no price for a vertex "
			    "or for a varying, so nothing says when %g a pixel is "
			    "worth %g a vertex and %d more flat varying%s\n",
			    sym_str(m->name), tot, tot, n, n == 1 ? "" : "s");
			rep_bool("priced", 0);
			rep_int("within_vulkan_precision", ninx);
		} else if (f == FQ_SCREEN) {
			sb_printf(&rep, "  once a vertex, in window coordinates: %d "
			    "instruction%s costing %g every pixel %s affine in "
			    "gl_FragCoord (sums, and products with values of the "
			    "draw)\n", nins, nins == 1 ? "" : "s", tot, nins == 1 ?
			    "is" : "are");
			sb_printf(&rep, "    gl_FragCoord is the window position, the "
			    "depth and 1/w, each affine across a primitive in window "
			    "coordinates, as a noperspective varying v = sum b_i v_i "
			    "is (the barycentrics b_i of the window position): an "
			    "affine f has f(v) = sum b_i f(v_i), so computed in the "
			    "vertex shader from each vertex's window position (the "
			    "viewport transform of gl_Position) and passed as %d "
			    "noperspective varying%s, the interpolated value is the "
			    "same real number at every pixel; in floats the two "
			    "differ by the interpolation's rounding (not "
			    "bit-exact)\n", n, n == 1 ? "" : "s");
			sb_printf(&rep, "    not priced: %s has no price for a vertex "
			    "or for a varying, so nothing says when %g a pixel is "
			    "worth %g a vertex and %d more varying%s\n",
			    sym_str(m->name), tot, tot, n, n == 1 ? "" : "s");
			rep_bool("priced", 0);
		} else {
			sb_printf(&rep, "  once a vertex: %d instruction%s costing %g "
			    "every pixel are affine in the varyings (sums, and "
			    "products with values of the draw)\n",
			    nins, nins == 1 ?
			    "" : "s", tot);
			sb_printf(&rep, "    a varying is v = sum b_i v_i over the "
			    "vertices with sum b_i = 1 (perspective-correct "
			    "barycentrics), and an affine f has f(v) = sum b_i "
			    "f(v_i): computed in the vertex shader and passed as %d "
			    "varying%s, the interpolated value is the same real "
			    "number at every pixel; in floats the two differ by the "
			    "interpolation's rounding (not bit-exact)\n", n, n == 1 ?
			    "" : "s");
			sb_printf(&rep, "    not priced: %s has no price for a vertex "
			    "or for a varying, so nothing says when %g a pixel is "
			    "worth %g a vertex and %d more varying%s\n",
			    sym_str(m->name), tot, tot, n, n == 1 ? "" : "s");
			rep_bool("priced", 0);
		}
		rep_list("values");
		for (i = 0; i < n; i++) {
			char sl[160];
			if (i < 8) {
				sb_printf(&rep, "    %s   (%g)\n", V.in[root[z][i].k].text,
				    root[z][i].cost);
				src_line(ins_line(root[z][i].k), sl, sizeof sl);
				if (sl[0])
					sb_printf(&rep, "  %s", sl);
			}
			rep_obj(NULL);
			rep_str("instruction", V.in[root[z][i].k].text);
			rep_num("cost", root[z][i].cost);
			rep_int("line", ins_line(root[z][i].k));
			rep_end();
		}
		if (n > 8)
			sb_printf(&rep, "    ... and %d more\n", n - 8);
		rep_end();
		rep_end();
	}
	if (!strcmp(stage, "Vertex"))
		adv_vertex_inputs(&rep, m);
	if (!strcmp(stage, "Fragment")) {
		adv_clip(&rep, m, fq);
		adv_pointcoord(&rep, m, fq);
	}
	if (!rep.n && asked)
		sb_printf(&rep, "advice: nothing in this module depends only on "
		    "its uniforms or its flat inputs, is affine in its varyings "
		    "or in gl_FragCoord, is what a vertex format or an instance "
		    "divisor computes, is a discard the clipper could do, or is a "
		    "point sprite's coordinate\n");
	if (rep.n)
		fputs(rep.p, stdout);
	free(rep.p);
	free(fq);
	free(used);
	free(price);
	free(seen);
	free(SP.def);
	SP.def = NULL;
	for (k = 0; k < V.nin; k++) {
		for (i = 0; i < V.in[k].nop; i++)
			free(V.in[k].opnd[i]);
		free(V.in[k].text);
	}
	free(V.in);
	memset(&V, 0, sizeof V);
}

/* ---- a rounding over a loop's period (#46, a proposal) ---- */

/*
 * A loop of T = 2^k passes (an index from c by 1 while below c + T, and
 * no store, atomic or call in it) that adds to an accumulator the same
 * value rounded to nearest on every pass: floor(z + 0.5), round,
 * roundEven, a conversion of z + 0.5, or the integer (v + 2^(S-1)) >>
 * S.  Each pass makes the same rounding error, so the sum is off by T
 * times it, up to T/2 steps.  Rounded instead against offsets that run
 * over one period, (g - c' + 1/2)/T with g the index or a value of it
 * the loop already computes that takes each of T consecutive integers
 * c', ..., c' + T - 1 once, the T roundings add up to T z rounded to
 * nearest, within half a step: Hermite's identity, which closed proves
 * for the offsets the loop has.  Proposed, never applied: the sum
 * changes.  An offset the loop does not compute (a frame counter, a
 * pixel's place) is not invented.
 */
enum { RD_NONE, RD_FLOOR, RD_ROUND, RD_ROUNDEVEN, RD_CVT, RD_INT };

/* id is the same on every pass of the loop, instructions [lo, hi):
 * defined outside it, or from such values; a load only when the loop
 * stores nothing */
static int
rd_invariant(int id, int lo, int hi, int stores, int depth)
{
	const struct sins *I;
	int k, i;

	id = res_of(id);
	if (V.id[id].kind != K_VALUE && V.id[id].kind != K_VAR &&
	    V.id[id].kind != K_NONE)
		return 1;
	k = SP.def[id];
	if (k < lo || k >= hi)
		return 1;
	I = &V.in[k];
	if (depth > 32 || !strcmp(I->op, "OpPhi") || !strcmp(I->op,
	    "OpFunctionCall") || !strncmp(I->op, "OpAtomic", 8) ||
	    !strncmp(I->op, "OpImage", 7) || (!strcmp(I->op, "OpLoad") &&
	    stores))
		return 0;
	for (i = 1; i < I->nop; i++)
		if (I->opnd[i][0] == '%' && !rd_invariant(idx(I->opnd[i]), lo, hi,
		    stores, depth + 1))
			return 0;
	return 1;
}

/* id's value reads target's (within the loop [lo, hi)) */
static int
rd_reads(int id, int target, int lo, int hi, int depth)
{
	const struct sins *I;
	int k, i;

	id = res_of(id);
	if (id == target)
		return 1;
	k = SP.def[id];
	if (k < lo || k >= hi || depth > 32)
		return 0;
	I = &V.in[k];
	if (!strcmp(I->op, "OpPhi"))
		return 0;
	for (i = 1; i < I->nop; i++)
		if (I->opnd[i][0] == '%' && rd_reads(idx(I->opnd[i]), target, lo,
		    hi, depth + 1))
			return 1;
	return 0;
}

/* the f32 constant 0.5 */
static int
rd_half(int id)
{
	id = res_of(id);
	return V.id[id].kind == K_CONST && !V.id[id].spec &&
	    is_float_t(V.id[id].type) && T(V.id[id].type)->k == S_FLOAT &&
	    (uint32_t)V.id[id].c[0] == 0x3f000000u;
}

/* q = z + 0.5: z, else -1 */
static int
rd_plus_half(int q)
{
	int k = SP.def[res_of(q)];
	const struct sins *I;

	if (k < 0)
		return -1;
	I = &V.in[k];
	if (strcmp(I->op, "OpFAdd") || I->nop < 3)
		return -1;
	if (rd_half(idx(I->opnd[2])))
		return idx(I->opnd[1]);
	if (rd_half(idx(I->opnd[1])))
		return idx(I->opnd[2]);
	return -1;
}

/* r rounds a value to nearest: its kind, the value (*z) and for the
 * integer form the shift (*S) */
static int
rd_rounding(int r, int *z, int *S)
{
	int k = SP.def[res_of(r)], q;
	const struct sins *I;
	uint64_t s, h;

	if (k < 0)
		return RD_NONE;
	I = &V.in[k];
	if (!strcmp(I->op, "OpExtInst") && I->nop == 4 && V.glsl >= 0 &&
	    idx(I->opnd[1]) == V.glsl && is_float_t(idx(I->opnd[0])) &&
	    T(idx(I->opnd[0]))->k == S_FLOAT) {
		if (!strcmp(I->opnd[2], "Floor") &&
		    (*z = rd_plus_half(idx(I->opnd[3]))) >= 0)
			return RD_FLOOR;
		*z = idx(I->opnd[3]);
		if (!strcmp(I->opnd[2], "Round"))
			return RD_ROUND;
		if (!strcmp(I->opnd[2], "RoundEven"))
			return RD_ROUNDEVEN;
		return RD_NONE;
	}
	if ((!strcmp(I->op, "OpConvertFToS") || !strcmp(I->op,
	    "OpConvertFToU")) && I->nop == 2 &&
	    (*z = rd_plus_half(idx(I->opnd[1]))) >= 0) {
		*S = !strcmp(I->op, "OpConvertFToU");	/* uint, else int */
		return RD_CVT;
	}
	if ((!strcmp(I->op, "OpShiftRightLogical") || !strcmp(I->op,
	    "OpShiftRightArithmetic")) && I->nop == 3 &&
	    int_width(idx(I->opnd[0])) == 32 && const_of(idx(I->opnd[2]), &s) &&
	    s >= 1 && s < 32 && (q = SP.def[res_of(idx(I->opnd[1]))]) >= 0 &&
	    !strcmp(V.in[q].op, "OpIAdd") && V.in[q].nop == 3) {
		const struct sins *A = &V.in[q];
		int j;
		for (j = 1; j <= 2; j++)
			if (const_of(idx(A->opnd[j]), &h) && h == (uint64_t)1 << (s - 1)) {
				*z = idx(A->opnd[3 - j]);
				*S = (int)s;
				return RD_INT;
			}
	}
	return RD_NONE;
}

/* the loop's index: a header phi from the constant c by 1, and the
 * passes T its condition gives; 0 when there is none */
static int
rd_index(int hdr, int kmerge, int lo, int hi, int merge, int *phi,
    int64_t *c, int64_t *T)
{
	int k, cond, t, f, i, p, kp, ok = 0, uns;
	uint64_t e, v0 = 0, one;
	const struct sins *I;

	/* the condition: the header's own branch, or the block it goes to */
	for (k = kmerge + 1; k < hi; k++)
		if (!strcmp(V.in[k].op, "OpBranchConditional") ||
		    !strcmp(V.in[k].op, "OpBranch"))
			break;
	if (k < hi && !strcmp(V.in[k].op, "OpBranch")) {
		int l = idx(V.in[k].opnd[0]);
		if (V.id[l].kind != K_LABEL)
			return 0;
		for (k = V.id[l].label_at + 1; k < hi && strcmp(V.in[k].op,
		    "OpBranchConditional") && strcmp(V.in[k].op, "OpBranch"); k++)
			;
	}
	if (k >= hi || strcmp(V.in[k].op, "OpBranchConditional") ||
	    V.in[k].nop < 3)
		return 0;
	cond = SP.def[res_of(idx(V.in[k].opnd[0]))];
	t = idx(V.in[k].opnd[1]);
	f = idx(V.in[k].opnd[2]);
	if (cond < 0 || f != merge || t == merge || V.in[cond].nop != 3)
		return 0;
	I = &V.in[cond];
	/* the phi is the condition's first operand, its bound a constant */
	p = res_of(idx(I->opnd[1]));
	kp = SP.def[p];
	if (kp < hdr || kp >= kmerge || strcmp(V.in[kp].op, "OpPhi") ||
	    V.in[kp].nop != 5 || !const_of(idx(I->opnd[2]), &e) ||
	    int_width(V.id[p].type) != 32)
		return 0;
	for (i = 1; i + 1 < 5; i += 2) {
		int v = idx(V.in[kp].opnd[i]), pl = idx(V.in[kp].opnd[i + 1]), kv;
		if (!(V.id[pl].kind == K_LABEL && V.id[pl].label_at >= lo &&
		    V.id[pl].label_at < hi)) {
			if (!const_of(v, &v0))
				return 0;
			ok |= 1;
			continue;
		}
		/* from inside: the phi plus 1 */
		kv = SP.def[res_of(v)];
		if (kv < 0 || strcmp(V.in[kv].op, "OpIAdd") || V.in[kv].nop != 3)
			return 0;
		if (!(res_of(idx(V.in[kv].opnd[1])) == p && const_of(idx(
		    V.in[kv].opnd[2]), &one) && one == 1) &&
		    !(res_of(idx(V.in[kv].opnd[2])) == p && const_of(idx(
		    V.in[kv].opnd[1]), &one) && one == 1))
			return 0;
		ok |= 2;
	}
	if (ok != 3)
		return 0;
	uns = !strcmp(I->op, "OpULessThan") || !strcmp(I->op,
	    "OpULessThanEqual");
	*c = uns ? (int64_t)(uint32_t)v0 : (int64_t)(int32_t)v0;
	e = uns ? (uint64_t)(uint32_t)e : (uint64_t)(int64_t)(int32_t)e;
	if (!strcmp(I->op, "OpSLessThan") || !strcmp(I->op, "OpULessThan") ||
	    !strcmp(I->op, "OpINotEqual"))
		*T = (int64_t)e - *c;
	else if (!strcmp(I->op, "OpSLessThanEqual") ||
	    !strcmp(I->op, "OpULessThanEqual"))
		*T = (int64_t)e - *c + 1;
	else
		return 0;
	*phi = p;
	return 1;
}

/*
 * A value of the index the loop computes (a pure integer instruction of
 * the index alone, not the index plus a constant) that takes each of T
 * consecutive integers once as the index runs c..c + T - 1: the node,
 * with its least value in *least; 0 when there is none
 */
static nid
rd_hash(int lo, int hi, int phi, int skip, int64_t c, int64_t T,
    int64_t *least, int *line)
{
	int kk;

	for (kk = lo; kk < hi; kk++) {
		struct spvdot cx;
		nid e;
		int64_t m, mn = 0, *vals;
		uint8_t *seen;
		int good = 1;
		if (V.in[kk].res < 0 || V.in[kk].res == phi || kk == skip ||
		    int_width(V.id[V.in[kk].res].type) != 32 || !pure_ins(kk))
			continue;
		memset(&cx, 0, sizeof cx);
		cx.memo = xcalloc((size_t)(V.nid + 1024), sizeof *cx.memo);
		cx.leaf[0] = phi;
		cx.nleaf = 1;
		cx.f.p[0].name = sym_cstr("i");
		cx.f.p[0].ty = spv_ty(V.id[phi].type);
		cx.f.np = 1;
		e = spv_node(&cx, V.in[kk].res, 0);
		free(cx.memo);
		if (cx.fail || cx.nleaf != 1 || !(node_slots(e) & 1) ||
		    N.op[e] == O_VAR || ((N.op[e] == O_ADD || N.op[e] == O_SUB) &&
		    (N.op[N.a[e]] == O_VAR || N.op[N.b[e]] == O_VAR)))
			continue;
		vals = xmalloc((size_t)T * sizeof *vals);
		seen = xcalloc((size_t)T, 1);
		for (m = 0; m < T && good; m++) {
			nid map[1];
			map[0] = mk_const_i(cx.f.p[0].ty, c + m);
			vals[m] = 0;
			good = int_of(subst(e, map, 1), &vals[m]);
			if (good && (m == 0 || vals[m] < mn))
				mn = vals[m];
		}
		for (m = 0; m < T && good; m++) {
			int64_t v = vals[m] - mn;
			if (v < 0 || v >= T || seen[v])
				good = 0;
			else
				seen[v] = 1;
		}
		free(vals);
		free(seen);
		if (good) {
			*least = mn;
			*line = ins_line(kk);
			return e;
		}
	}
	return 0;
}

static void
round_pass(const char *text)
{
	struct sbuf rep = {0};
	int k, i, n = 0;

	memset(&V, 0, sizeof V);
	symmap_reset();
	V.glsl = -1;
	V.lanes = 1;
	V.rng = 1;
	cols = 1;
	parse(text);
	declare();
	SP.def = xmalloc((size_t)(V.nid + 1024) * sizeof *SP.def);
	SP.alias = xmalloc((size_t)(V.nid + 1024) * sizeof *SP.alias);
	for (i = 0; i < V.nid + 1024; i++)
		SP.def[i] = SP.alias[i] = -1;
	SP.dead = xcalloc((size_t)V.nin + 1024, 1);
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (I->res < 0)
			continue;
		SP.def[I->res] = k;
		if (I->nop >= 1 && I->opnd[0][0] == '%' &&
		    V.id[idx(I->opnd[0])].kind == K_TYPE)
			V.id[I->res].type = idx(I->opnd[0]);
	}
	for (k = 0; k < V.nin; k++) {
		int hdr, merge, lo, hi, phi = -1, stores = 0, q, lg;
		int64_t c0 = 0, T = 0;
		ty_t ity;
		if (strcmp(V.in[k].op, "OpLoopMerge") || V.in[k].nop < 2)
			continue;
		for (hdr = k; hdr >= 0 && strcmp(V.in[hdr].op, "OpLabel"); hdr--)
			;
		merge = idx(V.in[k].opnd[0]);
		if (hdr < 0 || V.id[merge].kind != K_LABEL ||
		    V.id[merge].label_at <= k)
			continue;
		lo = hdr;
		hi = V.id[merge].label_at;
		for (q = lo; q < hi; q++)
			if (!strcmp(V.in[q].op, "OpStore") || !strncmp(V.in[q].op,
			    "OpAtomic", 8) || !strcmp(V.in[q].op, "OpFunctionCall") ||
			    !strcmp(V.in[q].op, "OpImageWrite"))
				stores = 1;
		if (!rd_index(hdr, k, lo, hi, merge, &phi, &c0, &T) || T < 2 ||
		    T > 4096 || (T & (T - 1)))
			continue;
		for (lg = 0; ((int64_t)1 << lg) < T; lg++)
			;
		ity = spv_ty(V.id[phi].type);
		/* the accumulators: header phis whose next value adds a rounding */
		for (q = hdr + 1; q < k; q++) {
			const struct sins *P = &V.in[q];
			int nx = -1, kn, r = -1, z = -1, S = 0, kind, j, hl = 0, rl;
			int64_t gc = c0;
			nid g, body, sum, zs, want;
			const char *how = NULL;
			char so[200], off[200], offp[204], what[80], zt[16], gcs[40];
			int vary;
			double pb = 0.5 + 2 * sqrt((double)T);
			if (strcmp(P->op, "OpPhi") || P->nop != 5 || P->res == phi)
				continue;
			for (j = 1; j + 1 < 5; j += 2) {
				int pl = idx(P->opnd[j + 1]);
				if (V.id[pl].kind == K_LABEL && V.id[pl].label_at >= lo &&
				    V.id[pl].label_at < hi)
					nx = idx(P->opnd[j]);
			}
			if (nx < 0 || (kn = SP.def[res_of(nx)]) < 0 || V.in[kn].nop != 3 ||
			    (strcmp(V.in[kn].op, "OpFAdd") && strcmp(V.in[kn].op,
			    "OpIAdd")))
				continue;
			for (j = 1; j <= 2; j++)
				if (res_of(idx(V.in[kn].opnd[j])) == P->res)
					r = idx(V.in[kn].opnd[3 - j]);
			if (r < 0 || (kind = rd_rounding(r, &z, &S)) == RD_NONE)
				continue;
			/* a value that changes with the pass (not with the sum
			 * itself): the offsets make the rounding unbiased, and the
			 * bound is a probable one; it gains from 32 passes on */
			vary = !rd_invariant(z, lo, hi, stores, 0);
			if (vary && (rd_reads(z, P->res, lo, hi, 0) || T < 32))
				continue;
			/* in whole numbers the offsets need T <= 2^(S-1) */
			if (kind == RD_INT && lg > S - 1)
				continue;
			if ((g = rd_hash(lo, hi, phi, kn, c0, T, &gc, &hl)) == 0) {
				g = mk_var(0, sym_cstr("i"), ity);
				gc = c0;
			}
			/* the proof: closed's sum over the period, with z' = z +
			 * (1/2 - c')/T a value free of the index */
			zs = mk_var(1, sym_cstr("z'"), TY_REAL);
			body = mk_un(O_FLOOR, mk_bin(O_ADD, zs, mk_bin(O_DIV,
			    mk_cast(TY_REAL, g), mk_const_r((double)T))));
			sum = sum_closed(body, 0, mk_const_i(ity, c0), mk_const_i(ity,
			    c0 + T - 1), &how);
			want = mk_bin(O_MUL, mk_const_r((double)T), zs);
			if (gc)
				want = mk_bin(O_ADD, want, mk_const_r((double)gc));
			/* the offsets cover the period: closed's sum is that floor
			 * (for a value that changes, what it proves is that each
			 * offset is met once, which makes the rounding unbiased) */
			if (sum == 0 || sum != mk_un(O_FLOOR, want))
				continue;
			if (!n++) {
				sb_printf(&rep, "  rounding over a loop's period (a "
				    "proposal, never applied: the sum changes):\n");
				rep_list("period_roundings");
			}
			rl = ins_line(SP.def[res_of(r)]);
			if (kind == RD_INT)
				snprintf(what, sizeof what, "(v + %u) >> %d",
				    1u << (S - 1), S);
			else
				snprintf(what, sizeof what, "%s", kind == RD_FLOOR ?
				    "floor(z + 0.5)" : kind == RD_ROUND ? "round(z)" :
				    kind == RD_ROUNDEVEN ? "roundEven(z)" : S ?
				    "uint(z + 0.5)" : "int(z + 0.5)");
			/* the offset as the proposal writes it: the index (less its
			 * first value), or the loop's own value of it, h */
			if (N.op[g] != O_VAR)
				snprintf(off, sizeof off, gc ? "h - %lld" : "h",
				    (long long)gc);
			else
				snprintf(off, sizeof off, c0 ? "i - %lld" : "i",
				    (long long)c0);
			snprintf(zt, sizeof zt, kind == RD_INT ? "v / 2^%d" : "z", S);
			snprintf(offp, sizeof offp, strchr(off, ' ') ? "(%s)" : "%s",
			    off);
			snprintf(gcs, sizeof gcs, "(0.5 - %lld)", (long long)gc);
			if (vary)
				sb_printf(&rep, "    line %d: the loop at line %d adds %s on "
				    "each of its %lld passes, %s changing with the pass: to "
				    "nearest, the sum is off by up to %lld steps where the "
				    "fractional parts lean one way (each pass's error then "
				    "has the same sign)\n", rl, ins_line(k), what,
				    (long long)T, kind == RD_INT ? "v" : "z",
				    (long long)(T / 2));
			else
				sb_printf(&rep, "    line %d: the loop at line %d adds %s on "
				    "each of its %lld passes, the same rounding of the same "
				    "value: the sum is off by %lld times its error, up to "
				    "%lld steps from %lld %s\n", rl, ins_line(k), what,
				    (long long)T, (long long)T, (long long)(T / 2),
				    (long long)T, zt);
			src_line(rl, so, sizeof so);
			if (so[0])
				sb_printf(&rep, "  %s", so);
			if (vary) {
				/* each pass meets an offset (j + 1/2)/T, j once each */
				if (kind == RD_INT)
					sb_printf(&rep, "      against offsets over the period, "
					    "(v + (%s << %d) + %d) >> %d (while v + %lld stays in "
					    "32 bits)", offp, S - lg, 1 << (S - lg - 1), S,
					    (long long)(((T - 1) << (S - lg)) + (1 << (S - lg -
					    1))));
				else
					sb_printf(&rep, "      against offsets over the period, "
					    "%s(z + (float(%s) + 0.5) / %lld.0)%s", kind !=
					    RD_CVT ? "floor" : S ? "uint" : "int", off,
					    (long long)T, kind == RD_CVT ? " (where z + 0.5 >= "
					    "0)" : "");
				sb_printf(&rep, ": a pass's rounding is unbiased over the "
				    "offsets (its mean error within 1/%lld of a step, "
				    "whatever the fractional part), so the sum is within "
				    "1/2 of the exact one on average, and probably within "
				    "1/2 + 2 sqrt(%lld) = %.3g steps: with probability at "
				    "least 0.99933 if the passes' fractional parts are "
				    "independent of one another and of the offsets "
				    "(Hoeffding's inequality, lambda = 4, each pass's error "
				    "in an interval of length 1: a model, not a theorem; "
				    "probabilistic)\n", (long long)(2 * T),
				    (long long)T, pb);
			} else if (kind == RD_INT)
				sb_printf(&rep, "      against offsets over the period, (v "
				    "+ (%s << %d) + %d) >> %d: the %lld passes add up to "
				    "(%lld v + %u) >> %d, %lld v / 2^%d to nearest, within "
				    "half a step (while v + %lld stays in 32 bits)\n", offp,
				    S - lg, 1 << (S - lg - 1), S, (long long)T, (long long)T,
				    1u << (S - 1), S, (long long)T, S, (long long)(((T - 1)
				    << (S - lg)) + (1 << (S - lg - 1))));
			else
				sb_printf(&rep, "      against offsets over the period, "
				    "%s(z + (float(%s) + 0.5) / %lld.0): the %lld passes "
				    "add up to floor(%lld z + 0.5), %lld z to nearest, within "
				    "half a step%s\n", kind != RD_CVT ? "floor" : S ? "uint" :
				    "int", off, (long long)T, (long long)T,
				    (long long)T, (long long)T, kind == RD_CVT ? " (the "
				    "conversion truncates: a floor where z + 0.5 >= 0)" : "");
			if (N.op[g] != O_VAR)
				sb_printf(&rep, "      h = %s, which the loop computes (line "
				    "%d), takes each of %lld..%lld once over the period\n",
				    node_str(g), hl, (long long)gc, (long long)(gc + T - 1));
			else
				sb_printf(&rep, "      i, the loop's index, runs %lld..%lld\n",
				    (long long)c0, (long long)(c0 + T - 1));
			if (vary)
				sb_printf(&rep, "      each offset met once over the period: "
				    "closed's sum(i, %lld, %lld, floor(z' + real(%s)/%lld)) "
				    "= %s for a z' that does not change\n", (long long)c0,
				    (long long)(c0 + T - 1), node_str(g), (long long)T,
				    node_str(sum));
			else
				sb_printf(&rep, "      proved by closed (Hermite's identity): "
				    "sum(i, %lld, %lld, floor(z' + real(%s)/%lld)) = %s, with "
				    "z' = %s + %s/%lld%s\n", (long long)c0, (long long)(c0 +
				    T - 1), node_str(g), (long long)T, node_str(sum), zt, gc ?
				    gcs : "0.5", (long long)T, kind == RD_INT ? "" : "; in "
				    "f32 the add can move a pass across a whole number only "
				    "within half an ulp of it");
			rep_obj(NULL);
			rep_int("line", rl);
			rep_int("loop_line", ins_line(k));
			rep_int("passes", T);
			rep_str("rounding", what);
			rep_str("offset", node_str(g));
			rep_num("error_nearest", (double)(T / 2));
			rep_num("error_period", vary ? pb : 0.5);
			if (vary) {
				rep_bool("probabilistic", 1);
				rep_num("probability", 1 - 2 * exp(-8.0));
			}
			rep_end();
		}
	}
	if (n)
		rep_end();
	if (rep.n)
		fputs(rep.p, stdout);
	free(rep.p);
	free(SP.def);
	free(SP.alias);
	free(SP.dead);
	SP.def = SP.alias = NULL;
	SP.dead = NULL;
	for (k = 0; k < V.nin; k++) {
		for (i = 0; i < V.in[k].nop; i++)
			free(V.in[k].opnd[i]);
		free(V.in[k].text);
	}
	free(V.in);
	memset(&V, 0, sizeof V);
}

static char *orig_mod;	/* the module before the rewrites */

/*
 * f32(x) of a 32-bit integer x by its bits (ifx.c's identity, #165): for
 * 0 <= x < 2^23 it is asfloat(x | 0x4B000000) - 2^23, and for -2^22 <= x
 * < 2^22 (a signed conversion) asfloat(x + 0x4B400000) - 1.5*2^23.  The
 * range is proved by SAT on x's definitions under the assumptions; the
 * rest is exact: the bits are those of the f32 2^23 + x (1.5*2^23 + x),
 * where one unit in the last place is 1, and the subtraction leaves an
 * integer below 2^24, which an f32 holds.  Taken where the model prices
 * the or (or add), the bitcast and the subtract below the conversion.
 * And f32(1 << k) is asfloat((k + 127) << 23), the exponent field of 2^k,
 * for every k a shift by it defines (0..31; k < 31 for a signed 1 << k,
 * proved by SAT, as 1 << 31 is negative there), where the shift goes
 * with the conversion and an add and a shift cost less than both.
 *
 * i2f_pass decides on the integer pass's module and keeps each one here;
 * i2f_patch writes them into the module the later passes leave, last:
 * the float, memory and advice passes read a conversion (its integer
 * range, an integer atomic's addend, a vertex format), not the bits.
 */
/*
 * A float id's range for every input the assumptions allow (a NaN aside:
 * it leaves the conversions it reaches undefined): a constant, a load
 * with a float assumption, a clamp, min or max by constants, an add,
 * subtract, negate or multiply (each widened by one f32 rounding),
 * floor, ceil, round and trunc, and an integer's conversion whose range
 * SAT bounds by a power of two.  integral: every value is a whole number.
 * A bound may be infinite (a load with no assumption: -inf..inf), so that
 * max(c, 0) then min(..., 1) is [0, 1]; ok 0: nothing known.
 */
struct frng {
	double lo, hi;
	int integral, ok;
};

/* one f32 rounding below v (above, f_up); never across 0, as a rounding
 * keeps the sign of what it rounds */
static double
f_down(double v)
{
	double d = (double)nextafterf((float)v, -INFINITY);

	return v >= 0 && d < 0 ? 0 : d;
}

static double
f_up(double v)
{
	double u = (double)nextafterf((float)v, INFINITY);

	return v <= 0 && u > 0 ? 0 : u;
}

/* the smallest 2^p (p in 8, 16, 22, 24) that bounds integer id: |id| <
 * 2^p (signed) or id < 2^p, proved by SAT; 0: none */
static int
i_bound(int id)
{
	static const int ps[] = {8, 16, 22, 24};
	int t = V.id[id].type, i, sg;
	struct word *wx, c, s;
	lit goal;

	if (t < 0 || T(t)->k != S_INT || T(t)->w != 32)
		return 0;
	sg = T(t)->sign;
	for (i = 0; i < 4; i++) {
		blast_reset();
		wx = blast(id, 16);
		if (sg) {
			struct word h;
			w_const(&h, 32, 1ULL << ps[i]);
			w_add(SP.g, &s, wx, &h);
			w_const(&c, 32, 2ULL << ps[i]);
		} else {
			s = *wx;
			w_const(&c, 32, 1ULL << ps[i]);
		}
		goal = aig_and(SP.g, SP.dl[id], assumptions());
		goal = aig_and(SP.g, goal, L_NOT(w_ult(SP.g, &s, &c)));
		if (aig_sat(SP.g, goal, 20000) == 0)
			return ps[i];
	}
	return 0;
}

static struct frng
f_range(int id, int depth)
{
	struct frng r = {0, 0, 0, 0}, a, b;
	const struct sins *I;
	const char *ext;
	int t, k, base;

	id = res_of(id);
	t = V.id[id].type;
	if (depth > 12 || t < 0 || T(t)->k != S_FLOAT || T(t)->w != 32)
		return r;
	if (V.id[id].kind == K_CONST && !V.id[id].spec && V.id[id].nc <= 1) {
		double v = getf(t, V.id[id].c[0]);
		if (isfinite(v)) {
			r.lo = r.hi = v;
			r.integral = v == floor(v);
			r.ok = 1;
		}
		return r;
	}
	k = SP.def[id];
	if (k < 0)
		return r;
	I = &V.in[k];
	if (!strcmp(I->op, "OpLoad")) {
		char nb[48];
		const char *nm = load_name(id, nb, sizeof nb);
		int j;
		r.lo = -INFINITY;
		r.hi = INFINITY;
		r.ok = 1;
		for (j = 0; nm && j < npend_fas; j++)
			if (name_is(nm, pend_fas[j].name) || !strcmp(nm,
			    pend_fas[j].name)) {
				r.lo = pend_fas[j].lo;
				r.hi = pend_fas[j].hi;
				r.ok = 1;
			}
		return r;
	}
	if (!strcmp(I->op, "OpConvertUToF") || !strcmp(I->op,
	    "OpConvertSToF")) {
		int x = res_of(idx(I->opnd[1])), p = i_bound(x);
		if (p) {
			r.lo = T(V.id[x].type)->sign ? -ldexp(1, p) : 0;
			r.hi = ldexp(1, p);
			r.integral = r.ok = 1;
		}
		return r;
	}
	if (!strcmp(I->op, "OpFNegate")) {
		a = f_range(idx(I->opnd[1]), depth + 1);
		r = a;
		r.lo = -a.hi;
		r.hi = -a.lo;
		return r;
	}
	if (!strcmp(I->op, "OpFAdd") || !strcmp(I->op, "OpFSub") ||
	    !strcmp(I->op, "OpFMul")) {
		a = f_range(idx(I->opnd[1]), depth + 1);
		b = f_range(idx(I->opnd[2]), depth + 1);
		if (!a.ok || !b.ok)
			return r;
		if (I->op[3] == 'A') {
			r.lo = a.lo + b.lo;
			r.hi = a.hi + b.hi;
		} else if (I->op[3] == 'S') {
			r.lo = a.lo - b.hi;
			r.hi = a.hi - b.lo;
		} else {
			double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo,
			    a.hi * b.hi};
			int j;
			r.lo = r.hi = p[0];
			for (j = 1; j < 4; j++) {
				r.lo = fmin(r.lo, p[j]);
				r.hi = fmax(r.hi, p[j]);
			}
		}
		/* whole numbers below 2^24 add and multiply exactly */
		r.integral = a.integral && b.integral && fmax(fabs(r.lo),
		    fabs(r.hi)) <= 16777216.0;
		if (!r.integral) {
			r.lo = f_down(r.lo);
			r.hi = f_up(r.hi);
		}
		r.ok = !isnan(r.lo) && !isnan(r.hi);	/* inf * 0: unknown */
		return r;
	}
	if (!strcmp(I->op, "OpSelect") && I->nop >= 4) {
		a = f_range(idx(I->opnd[2]), depth + 1);
		b = f_range(idx(I->opnd[3]), depth + 1);
		if (!a.ok || !b.ok)
			return r;
		r.lo = fmin(a.lo, b.lo);
		r.hi = fmax(a.hi, b.hi);
		r.integral = a.integral && b.integral;
		r.ok = 1;
		return r;
	}
	if (strcmp(I->op, "OpExtInst") || I->nop < 4)
		return r;
	ext = I->opnd[2];
	base = 3;
	a = f_range(idx(I->opnd[base]), depth + 1);
	if (!strcmp(ext, "Floor") || !strcmp(ext, "Ceil") || !strcmp(ext,
	    "Round") || !strcmp(ext, "RoundEven") || !strcmp(ext, "Trunc")) {
		if (!a.ok)
			return r;
		r.lo = floor(a.lo);
		r.hi = ceil(a.hi);
		r.integral = isfinite(r.lo) && isfinite(r.hi);
		r.ok = 1;
		return r;
	}
	if ((!strcmp(ext, "FClamp") || !strcmp(ext, "NClamp")) && I->nop >= 6) {
		b = f_range(idx(I->opnd[4]), depth + 1);
		{
			struct frng c = f_range(idx(I->opnd[5]), depth + 1);
			if (!b.ok || !c.ok || b.lo != b.hi || c.lo != c.hi)
				return r;
			r.lo = a.ok ? fmax(a.lo, b.lo) : b.lo;
			r.hi = a.ok ? fmin(a.hi, c.hi) : c.hi;
			/* (a NaN x aside, which the conversion leaves undefined) */
			r.integral = a.ok && a.integral && b.integral && c.integral;
			r.ok = r.lo <= r.hi;
		}
		return r;
	}
	if ((!strcmp(ext, "FMin") || !strcmp(ext, "FMax") || !strcmp(ext,
	    "NMin") || !strcmp(ext, "NMax")) && I->nop >= 5) {
		int mx = ext[2] == 'a';	/* FMax, NMax */
		b = f_range(idx(I->opnd[4]), depth + 1);
		if (!a.ok || !b.ok)
			return r;
		r.lo = mx ? fmax(a.lo, b.lo) : fmin(a.lo, b.lo);
		r.hi = mx ? fmax(a.hi, b.hi) : fmin(a.hi, b.hi);
		r.integral = a.integral && b.integral;
		r.ok = 1;
		return r;
	}
	return r;
}

static struct cvx {
	char res[40], rt[40], x[40], xt[40], op[40], k[40], kt[40];
	char from[160], src[160];
	char lx[40], it[40];	/* forms 7, 8: log2's operand; 7: an int type;
				 * 9: k and its type */
	char drop[3][40];	/* forms 7, 8, 10, 11: the calls they leave
				 * unused (10, 11: and the conversion) */
	int zero;		/* form 9: x may be 0 (an integer converted,
				 * else 1 or more in magnitude): a select keeps
				 * it; kt2 the bool type */
	int mul;		/* form 9: x times 2^k made by its exponent
				 * field, k in [-126, 127] (x anything) */
	char kt2[40];
	char fx[40];		/* forms 10, 11: the f32 x log2 reads */
	int form;		/* 0: the or with 0x4B000000; 1: the add;
				 * 2: f32(1 << k) by the exponent field;
				 * 3: iN(x) of a whole x by the add of
				 * 1.5*2^23, read as bits; 4, 5, 6: x
				 * rounded to even, floor, ceil by that
				 * add and its subtraction (k: the call,
				 * kt: the bool type); 7, 8: floor(log2
				 * x) by x's exponent field, as a float
				 * (k: the call) or converted; 9: ldexp(x,
				 * k) as an add of k to x's exponent field;
				 * 10, 11: floor(log2(x)) of an integer i
				 * converted as FindUMsb(i) (lx: i), converted
				 * or as a float (it: an int type, kt2: the
				 * GLSL import) */
	int line;
	double before, after;
	double flo, fhi;	/* form 3: x's range */
} cvx[32];
static int ncvx;

/* a form that rewrites a call (its line has 7 words), not a conversion */
#define CALLFORM(f) (((f) >= 4 && (f) <= 7) || (f) == 11)	/* (9: 8 words) */

/* the instruction that uses id, when one alone does; else -1 */
static int
only_user(int id)
{
	int k, i, u = -1;
	for (k = 0; k < V.nin; k++) {
		const struct sins *I = &V.in[k];
		if (SP.dead[k] || !strncmp(I->op, "OpName", 6) || !strncmp(I->op,
		    "OpDecorate", 10) || !strncmp(I->op, "OpMemberName", 12))
			continue;
		for (i = 0; i < I->nop; i++)
			if (I->opnd[i][0] == '%' && idx(I->opnd[i]) == id) {
				if (u >= 0 && u != k)
					return -1;
				u = k;
			}
	}
	return u;
}

/*
 * id is log2(x) of an f32 x that is positive and normal (2^-126 at least
 * and finite; *ge1: 1 at least), used by instruction user alone: x; else
 * -1.  Not with `exact`: a GPU's log2 is within 3 ulp, the field exact
 */
static int
log2_of(int id, int user, int *ge1)
{
	int d = id >= 0 ? SP.def[id] : -1, x, t;
	const struct sins *I;
	struct frng fr;
	if (d < 0 || fopt.mode == 1)
		return -1;
	I = &V.in[d];
	t = idx(I->opnd[0]);
	if (strcmp(I->op, "OpExtInst") || I->nop != 4 || V.glsl < 0 ||
	    idx(I->opnd[1]) != V.glsl || strcmp(I->opnd[2], "Log2") ||
	    I->opnd[3][0] != '%' || t < 0 || V.id[t].kind != K_TYPE ||
	    T(t)->k != S_FLOAT || T(t)->w != 32 || only_user(id) != user)
		return -1;
	x = res_of(idx(I->opnd[3]));
	fr = f_range(x, 0);
	if (!fr.ok || !isfinite(fr.hi) || fr.lo < 0x1p-126)
		return -1;
	*ge1 = fr.lo >= 1;
	return x;
}

/*
 * id is log2(x) of an f32 x that is a 32-bit integer i converted, used by
 * instruction user alone, with 1 <= i < 2^24 proved by SAT: the
 * conversion is exact, so floor(log2(x)) is i's most significant bit,
 * FindUMsb(i).  i, else -1; *xf x, *alone: the conversion has no other
 * reader than the log2.  Not with `exact` (a GPU's log2 is within 3 ulp)
 */
static int
log2_int(int id, int user, int *xf, int *alone)
{
	int d = id >= 0 ? SP.def[id] : -1, x, cd, i, t, sg;
	const struct sins *I;
	struct word *wi, one, top, s;
	lit goal;

	if (d < 0 || fopt.mode == 1 || V.glsl < 0)
		return -1;
	I = &V.in[d];
	t = idx(I->opnd[0]);
	if (strcmp(I->op, "OpExtInst") || I->nop != 4 || idx(I->opnd[1]) !=
	    V.glsl || strcmp(I->opnd[2], "Log2") || I->opnd[3][0] != '%' ||
	    t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_FLOAT ||
	    T(t)->w != 32 || only_user(id) != user)
		return -1;
	x = res_of(idx(I->opnd[3]));
	cd = SP.def[x];
	if (cd < 0 || (strcmp(V.in[cd].op, "OpConvertUToF") &&
	    strcmp(V.in[cd].op, "OpConvertSToF")) || V.in[cd].nop != 2 ||
	    V.in[cd].opnd[1][0] != '%')
		return -1;
	i = res_of(idx(V.in[cd].opnd[1]));
	t = V.id[i].type;
	if (t < 0 || V.id[t].kind != K_TYPE || T(t)->k != S_INT || T(t)->w != 32 ||
	    (double)clock() / CLOCKS_PER_SEC > SP.deadline)
		return -1;
	/* i - 1 < 2^24 - 1 unsigned: 1 <= i < 2^24 */
	sg = T(t)->sign;
	(void)sg;
	blast_reset();
	wi = blast(i, 16);
	w_const(&one, 32, (uint64_t)0xFFFFFFFFu);
	w_add(SP.g, &s, wi, &one);
	w_const(&top, 32, 0xFFFFFFu);
	goal = aig_and(SP.g, aig_and(SP.g, SP.dl[i], assumptions()),
	    L_NOT(w_ult(SP.g, &s, &top)));
	if (aig_sat(SP.g, goal, 20000) != 0)
		return -1;
	*xf = x;
	*alone = only_user(x) == d;
	return i;
}

/*
 * Every value the 32-bit integer k takes (its definitions and the
 * assumptions) is in [lo, hi], proved by SAT
 */
static int
k_within(int k, int64_t lo, int64_t hi)
{
	int t = V.id[k].type, sg;
	int64_t tlo, thi;
	struct word *wk, cl, ch;
	lit goal, out = 0;

	if (lo > hi || t < 0 || T(t)->k != S_INT || T(t)->w != 32)
		return 0;
	sg = T(t)->sign;
	tlo = sg ? INT32_MIN : 0;
	thi = sg ? INT32_MAX : (int64_t)UINT32_MAX;
	if (hi < tlo || lo > thi)
		return 0;
	if (lo <= tlo && hi >= thi)
		return 1;
	blast_reset();
	wk = blast(k, 16);
	/* below lo, or above hi, where the type reaches past them */
	if (lo > tlo) {
		w_const(&cl, 32, (uint64_t)(uint32_t)lo);
		out = sg ? w_slt(SP.g, wk, &cl) : w_ult(SP.g, wk, &cl);
	}
	if (hi < thi) {
		lit up;
		w_const(&ch, 32, (uint64_t)(uint32_t)hi);
		up = sg ? w_slt(SP.g, &ch, wk) : w_ult(SP.g, &ch, wk);
		out = lo > tlo ? aig_or(SP.g, out, up) : up;
	}
	goal = aig_and(SP.g, aig_and(SP.g, SP.dl[k], assumptions()), out);
	return aig_sat(SP.g, goal, 20000) == 0;
}

static void
i2f_pass(void)
{
	const struct model *m = model_active();
	int k, preserve = 0, bt = -1, it = -1;

	ncvx = 0;
	for (k = 0; k < V.nid && it < 0; k++)
		if (V.id[k].kind == K_TYPE && T(k)->k == S_INT && T(k)->w == 32 &&
		    T(k)->sign)
			it = k;
	for (k = 0; k < V.nin; k++)
		if (!strcmp(V.in[k].op, "OpExecutionMode") && V.in[k].nop >= 2 &&
		    !strcmp(V.in[k].opnd[1], "SignedZeroInfNanPreserve"))
			preserve = 1;
	for (k = 0; k < V.nid && bt < 0; k++)
		if (V.id[k].kind == K_TYPE && T(k)->k == S_BOOL)
			bt = k;
	for (k = 0; k < SP.orig_nin && ncvx < 32; k++) {
		const struct sins *I = &V.in[k];
		int rt, x, xt, sg, form = -1;
		uint64_t one;
		double before, after;
		lit goal;
		struct word *wx, c, sh;
		struct cvx *e;
		if (SP.dead[k] || SP.rw[k])
			continue;
		/*
		 * ldexp(x, k), two forms, the cheaper the model prices first:
		 * of an f32 x whose |x| is normal (2^-126 at least, finite), k
		 * proved by SAT to keep x * 2^k normal: x's exponent field plus
		 * k, asfloat(asint(x) + (k << 23)), exactly (the sign and the
		 * mantissa are x's); and of any x, k proved in [-126, 127]: x
		 * times 2^k made by its exponent field, asfloat((k << 23) +
		 * 0x3F800000), which rounds once as ldexp does
		 */
		if (!strcmp(I->op, "OpExtInst") && I->nop == 5 && I->res >= 0 &&
		    V.glsl >= 0 && idx(I->opnd[1]) == V.glsl && !strcmp(I->opnd[2],
		    "Ldexp") && I->opnd[3][0] == '%' && I->opnd[4][0] == '%') {
			int kid = res_of(idx(I->opnd[4])), kt, elo = 0, ehi = 0, zero = 0;
			int fld, c, mul = -1;
			struct frng fr;
			double alo = 0, ahi = 0, afld, amul;
			rt = idx(I->opnd[0]);
			x = res_of(idx(I->opnd[3]));
			kt = V.id[kid].type;
			if (rt < 0 || V.id[rt].kind != K_TYPE || T(rt)->k != S_FLOAT ||
			    T(rt)->w != 32 || kt < 0 || T(kt)->k != S_INT || T(kt)->w != 32)
				continue;
			fr = f_range(x, 0);
			fld = fr.ok && isfinite(fr.lo) && isfinite(fr.hi);
			/* |x| below 2^-126 somewhere: 0 or subnormal; an integer
			 * converted is 0 or 1 at least in magnitude, and its 0 is
			 * kept by a select */
			if (fld && fr.lo < 0x1p-126 && fr.hi > -0x1p-126) {
				int xd = SP.def[x];
				if (xd < 0 || (strcmp(V.in[xd].op, "OpConvertUToF") &&
				    strcmp(V.in[xd].op, "OpConvertSToF")))
					fld = 0;
				zero = 1;
			}
			if (fld) {
				/* |x|'s least (nonzero) and most */
				alo = zero ? 1 : fr.lo > 0 ? fr.lo : -fr.hi;
				ahi = zero ? fmax(-fr.lo, fr.hi) : fr.lo > 0 ? fr.hi :
				    -fr.lo;
				fld = ahi >= alo;
				if (fld) {
					elo = ilogb(alo);
					ehi = ilogb(ahi);
				}
			}
			before = fprice_std("Ldexp", 1);
			afld = m->c[C_SHIFT] + m->c[C_IADD] + 2 * m->c[C_BITCAST] +
			    (zero ? m->c[C_CMP] + m->c[C_SELECT] : 0);
			amul = m->c[C_SHIFT] + m->c[C_IADD] + m->c[C_BITCAST] +
			    m->c[C_FMUL];
			fld = fld && afld < before - 1e-9;
			for (c = 0; c < 2 && mul < 0; c++) {
				/* c 0: the cheaper of the two, c 1: the other */
				int try_mul = (c == 0) == (!fld || amul < afld);
				if (try_mul ? !(amul < before - 1e-9) : !fld)
					continue;
				if ((double)clock() / CLOCKS_PER_SEC > SP.deadline)
					break;
				if (try_mul ? k_within(kid, -126, 127) : k_within(kid,
				    -126 - elo, 127 - ehi))
					mul = try_mul;
			}
			if (mul < 0)
				continue;
			after = mul ? amul : afld;
			xt = rt;
			e = &cvx[ncvx++];
			memset(e, 0, sizeof *e);
			e->flo = fr.ok ? fr.lo : -INFINITY;
			e->fhi = fr.ok ? fr.hi : INFINITY;
			snprintf(e->k, sizeof e->k, "%s", V.id[kid].tok);
			snprintf(e->kt, sizeof e->kt, "%s", V.id[kt].tok);
			snprintf(e->lx, sizeof e->lx, "%d", mul ? -126 : -126 - elo);
			snprintf(e->it, sizeof e->it, "%d", mul ? 127 : 127 - ehi);
			e->zero = !mul && zero;
			e->mul = mul;
			snprintf(e->kt2, sizeof e->kt2, "%s", bt >= 0 ? V.id[bt].tok :
			    "%ucas_bool");
			form = 9;
			goto keep;
		}
		/*
		 * floor, ceil, trunc, round and roundeven of an f32 x in
		 * [-2^22, 2^22] (one unit of margin for what the float pass may
		 * move): x + 1.5*2^23 lies in [2^23, 2^24], where a unit in the
		 * last place is 1, so the add rounds x to an integer, to even,
		 * and the subtraction is exact; floor and ceil step back by one
		 * where that went past x.  A zero result may be +0 for -0,
		 * which Vulkan's default float controls allow (not with
		 * SignedZeroInfNanPreserve), and round's halves go to even,
		 * one of the two GLSL lets it take
		 */
		if (!strcmp(I->op, "OpExtInst") && I->nop == 4 && I->res >= 0 &&
		    !preserve && V.glsl >= 0 && idx(I->opnd[1]) == V.glsl &&
		    I->opnd[3][0] == '%') {
			static const char *const rnd[] = {"Floor", "Ceil", "Trunc",
			    "Round", "RoundEven", NULL};
			const char *f = I->opnd[2];
			struct frng fr;
			int ge1;
			rt = idx(I->opnd[0]);
			if (!in_list(f, rnd) || rt < 0 || V.id[rt].kind != K_TYPE ||
			    T(rt)->k != S_FLOAT || T(rt)->w != 32)
				continue;
			x = res_of(idx(I->opnd[3]));
			if (!strcmp(f, "Floor") && it >= 0) {
				/* floor(log2(f32(i))) of an integer i in [1, 2^24): its
				 * FindUMsb, converted (form 11), unless a conversion alone
				 * reads it (form 10, at the conversion) */
				int u = only_user(I->res), fx, alone = 0, iv;
				iv = log2_int(x, k, &fx, &alone);
				before = m->c[C_FTRANS] + m->c[C_CONV] + (alone ?
				    m->c[C_CONV] : 0);
				after = m->c[C_BITCNT] + m->c[C_CONV];
				if (iv >= 0 && !(u >= 0 && (!strcmp(V.in[u].op,
				    "OpConvertFToS") || !strcmp(V.in[u].op,
				    "OpConvertFToU"))) && after < before - 1e-9) {
					xt = rt;
					e = &cvx[ncvx++];
					memset(e, 0, sizeof *e);
					snprintf(e->k, sizeof e->k, "%s", f);
					snprintf(e->lx, sizeof e->lx, "%s", V.id[iv].tok);
					snprintf(e->it, sizeof e->it, "%s", V.id[it].tok);
					snprintf(e->kt2, sizeof e->kt2, "%s", V.id[V.glsl].tok);
					snprintf(e->fx, sizeof e->fx, "%s", V.id[fx].tok);
					snprintf(e->drop[0], sizeof e->drop[0], "%s",
					    V.id[x].tok);
					if (alone)
						snprintf(e->drop[2], sizeof e->drop[2], "%s",
						    V.id[fx].tok);
					form = 11;
					goto keep;
				}
			}
			if (!strcmp(f, "Floor") && log2_of(x, k, &ge1) >= 0 && it >= 0) {
				/* floor(log2(x)): the exponent field, as a float
				 * (form 7), unless a conversion alone reads it
				 * (form 8, at the conversion) */
				int u = only_user(I->res), lx = log2_of(x, k, &ge1);
				if (u >= 0 && (!strcmp(V.in[u].op, "OpConvertFToS") ||
				    !strcmp(V.in[u].op, "OpConvertFToU")))
					continue;
				before = m->c[C_FTRANS] + m->c[C_CONV];
				after = m->c[C_SHIFT] + m->c[C_IADD] + m->c[C_CONV] +
				    m->c[C_BITCAST];
				if (!(after < before - 1e-9))
					continue;
				xt = rt;
				e = &cvx[ncvx++];
				memset(e, 0, sizeof *e);
				snprintf(e->k, sizeof e->k, "%s", f);
				snprintf(e->lx, sizeof e->lx, "%s", V.id[lx].tok);
				snprintf(e->it, sizeof e->it, "%s", V.id[it].tok);
				snprintf(e->drop[0], sizeof e->drop[0], "%s", V.id[x].tok);
				form = 7;
				goto keep;
			}
			fr = f_range(x, 0);
			if (!fr.ok || !isfinite(fr.lo) || !isfinite(fr.hi) ||
			    fr.lo - 1 < -4194304.0 || fr.hi + 1 > 4194304.0)
				continue;
			/* trunc is floor where x >= 0, ceil where x <= 0 */
			form = !strncmp(f, "Round", 5) ? 4 : !strcmp(f, "Floor") ||
			    (f[0] == 'T' && fr.lo >= 0) ? 5 : !strcmp(f, "Ceil") ||
			    (f[0] == 'T' && fr.hi <= 0) ? 6 : -1;
			if (form < 0)
				continue;
			before = m->c[C_CONV];
			after = 2 * m->c[C_FADD] + (form > 4 ? m->c[C_CMP] +
			    m->c[C_FADD] + m->c[C_SELECT] : 0);
			if (!(after < before - 1e-9))
				continue;
			xt = rt;
			e = &cvx[ncvx++];
			memset(e, 0, sizeof *e);
			e->flo = fr.lo;
			e->fhi = fr.hi;
			snprintf(e->k, sizeof e->k, "%s", f);
			/* a module with no compare has no bool type: one declared */
			snprintf(e->kt, sizeof e->kt, "%s", bt >= 0 ? V.id[bt].tok :
			    "%ucas_bool");
			goto keep;
		}
		if (I->nop != 2 || I->opnd[1][0] != '%')
			continue;
		if (!strcmp(I->op, "OpConvertFToS") || !strcmp(I->op,
		    "OpConvertFToU")) {
			/* iN(floor(log2(x))), or iN(log2(x)) of x >= 1 (trunc is
			 * floor there): the exponent field less 127, the floor
			 * and the log2 unused after it */
			int y = res_of(idx(I->opnd[1])), yd = SP.def[y], ge1 = 0, lx = -1,
			    fl = -1, ct = idx(I->opnd[0]);
			int iv = -1, fx = -1, alone = 0;
			if (yd >= 0 && !strcmp(V.in[yd].op, "OpExtInst") &&
			    V.in[yd].nop == 4 && !strcmp(V.in[yd].opnd[2], "Floor") &&
			    V.in[yd].opnd[3][0] == '%' && only_user(y) == k) {
				fl = y;
				iv = log2_int(res_of(idx(V.in[yd].opnd[3])), yd, &fx,
				    &alone);
				lx = iv >= 0 ? fx : log2_of(res_of(idx(V.in[yd].opnd[3])),
				    yd, &ge1);
			} else {
				iv = log2_int(y, k, &fx, &alone);
				lx = iv >= 0 ? fx : log2_of(y, k, &ge1);
				if (iv < 0 && !ge1)
					lx = -1;
			}
			if (lx >= 0 && ct >= 0 && T(ct)->k == S_INT && T(ct)->w == 32) {
				int lg = fl >= 0 ? res_of(idx(V.in[yd].opnd[3])) : y;
				/* of an integer converted: its FindUMsb */
				before = m->c[C_FTRANS] + (fl >= 0 ? m->c[C_CONV] : 0) +
				    m->c[C_CONV] + (iv >= 0 && alone ? m->c[C_CONV] : 0);
				after = m->c[C_BITCNT];
				if (iv >= 0 && V.glsl >= 0 && after < before - 1e-9) {
					rt = ct;
					x = y;
					xt = V.id[y].type;
					e = &cvx[ncvx++];
					memset(e, 0, sizeof *e);
					snprintf(e->lx, sizeof e->lx, "%s", V.id[iv].tok);
					snprintf(e->kt2, sizeof e->kt2, "%s", V.id[V.glsl].tok);
					snprintf(e->fx, sizeof e->fx, "%s", V.id[lx].tok);
					snprintf(e->drop[0], sizeof e->drop[0], "%s",
					    V.id[lg].tok);
					if (fl >= 0)
						snprintf(e->drop[1], sizeof e->drop[1], "%s",
						    V.id[fl].tok);
					if (alone)
						snprintf(e->drop[2], sizeof e->drop[2], "%s",
						    V.id[lx].tok);
					form = 10;
					goto keep;
				}
				before = m->c[C_FTRANS] + (fl >= 0 ? m->c[C_CONV] : 0) +
				    m->c[C_CONV];
				after = m->c[C_SHIFT] + m->c[C_IADD] + m->c[C_BITCAST];
				if (after < before - 1e-9) {
					rt = ct;
					x = y;
					xt = V.id[y].type;
					e = &cvx[ncvx++];
					memset(e, 0, sizeof *e);
					snprintf(e->lx, sizeof e->lx, "%s", V.id[lx].tok);
					snprintf(e->drop[0], sizeof e->drop[0], "%s",
					    V.id[lg].tok);
					if (fl >= 0)
						snprintf(e->drop[1], sizeof e->drop[1], "%s",
						    V.id[fl].tok);
					form = 8;
					goto keep;
				}
			}
		}
		if (!strcmp(I->op, "OpConvertFToS") || !strcmp(I->op,
		    "OpConvertFToU")) {
			/*
			 * a whole x made by floor, ceil, round or trunc (leaves
			 * of the float pass, which may move what is under them
			 * within its precision: one unit of margin each side),
			 * |x| below 2^22: x + 1.5*2^23 is exact, and its bits
			 * less 0x4B400000 are x
			 */
			int xd, it = idx(I->opnd[0]);
			struct frng fr;
			x = res_of(idx(I->opnd[1]));
			xd = SP.def[x];
			static const char *const whole[] = {"Floor", "Ceil", "Round",
			    "RoundEven", "Trunc", NULL};
			if (it < 0 || T(it)->k != S_INT || T(it)->w != 32 || xd < 0 ||
			    strcmp(V.in[xd].op, "OpExtInst") || V.in[xd].nop < 4 ||
			    !in_list(V.in[xd].opnd[2], whole))
				continue;
			before = m->c[C_CONV];
			after = m->c[C_FADD] + m->c[C_BITCAST] + m->c[C_IADD];
			if (!(after < before - 1e-9))
				continue;
			fr = f_range(x, 0);
			if (!fr.ok || !fr.integral || !isfinite(fr.lo) ||
			    !isfinite(fr.hi) || fr.hi + 1 > 4194303.0 ||
			    fr.lo - 1 < -4194303.0)
				continue;	/* (a negative x leaves FToU undefined:
						 * any word will do there) */
			rt = it;
			xt = V.id[x].type;
			e = &cvx[ncvx++];
			memset(e, 0, sizeof *e);
			e->flo = fr.lo;
			e->fhi = fr.hi;
			form = 3;
			goto keep;
		}
		if ((strcmp(I->op, "OpConvertUToF") && strcmp(I->op,
		    "OpConvertSToF")))
			continue;
		rt = idx(I->opnd[0]);
		x = res_of(idx(I->opnd[1]));
		xt = V.id[x].type;
		if (V.id[rt].kind != K_TYPE || T(rt)->k != S_FLOAT ||
		    T(rt)->w != 32 || xt < 0 || T(xt)->k != S_INT || T(xt)->w != 32)
			continue;
		sg = I->op[9] == 'S';
		/* f32(1 << k), the shift used by the conversion alone */
		if (SP.def[x] >= 0 && !strcmp(V.in[SP.def[x]].op,
		    "OpShiftLeftLogical") && SP.uses[x] == 1 &&
		    const_of(res_of(idx(V.in[SP.def[x]].opnd[1])), &one) && one == 1) {
			int kid = res_of(idx(V.in[SP.def[x]].opnd[2])),
			    kt = V.id[kid].type;
			before = m->c[C_CONV] + ins_cost_k(SP.def[x]);
			after = m->c[C_IADD] + m->c[C_SHIFT] + m->c[C_BITCAST];
			if (kt >= 0 && T(kt)->k == S_INT && T(kt)->w == 32 &&
			    after < before - 1e-9) {
				int ok = 1;
				if (sg) {	/* 1 << 31 is negative: k < 31 */
					blast_reset();
					wx = blast(kid, 16);
					w_const(&c, 32, 31);
					goal = aig_and(SP.g, SP.dl[kid],
					    assumptions());
					goal = aig_and(SP.g, goal, L_NOT(w_ult(SP.g, wx,
					    &c)));
					ok = aig_sat(SP.g, goal, 20000) == 0;
				}
				if (ok) {
					e = &cvx[ncvx++];
					memset(e, 0, sizeof *e);
					snprintf(e->k, sizeof e->k, "%s", V.id[kid].tok);
					snprintf(e->kt, sizeof e->kt, "%s", V.id[kt].tok);
					form = 2;
					goto keep;
				}
			}
		}
		before = m->c[C_CONV];
		if (!(m->c[C_LOGIC] + m->c[C_BITCAST] + m->c[C_FADD] < before -
		    1e-9) && !(sg && m->c[C_IADD] + m->c[C_BITCAST] + m->c[C_FADD] <
		    before - 1e-9))
			continue;
		if ((double)clock() / CLOCKS_PER_SEC > SP.deadline)
			break;
		/* 0 <= x < 2^23: no input reaches 2^23 (unsigned) */
		blast_reset();
		wx = blast(x, 16);
		w_const(&c, 32, 0x800000);
		goal = aig_and(SP.g, SP.dl[x], assumptions());
		goal = aig_and(SP.g, goal, L_NOT(w_ult(SP.g, wx, &c)));
		if (aig_sat(SP.g, goal, 20000) == 0)
			form = 0;
		else if (sg) {
			/* -2^22 <= x < 2^22: x + 2^22 below 2^23, unsigned */
			struct word s2;
			w_const(&sh, 32, 0x400000);
			w_add(SP.g, &s2, wx, &sh);
			goal = aig_and(SP.g, SP.dl[x], assumptions());
			goal = aig_and(SP.g, goal, L_NOT(w_ult(SP.g, &s2, &c)));
			if (aig_sat(SP.g, goal, 20000) == 0)
				form = 1;
		}
		if (form < 0)
			continue;
		after = (form ? m->c[C_IADD] : m->c[C_LOGIC]) + m->c[C_BITCAST] +
		    m->c[C_FADD];
		if (!(after < before - 1e-9))
			continue;
		e = &cvx[ncvx++];
		memset(e, 0, sizeof *e);
keep:
		snprintf(e->res, sizeof e->res, "%s", V.id[I->res].tok);
		snprintf(e->rt, sizeof e->rt, "%s", V.id[rt].tok);
		snprintf(e->x, sizeof e->x, "%s", V.id[x].tok);
		snprintf(e->xt, sizeof e->xt, "%s", V.id[xt].tok);
		snprintf(e->op, sizeof e->op, "%s", I->op);
		snprintf(e->from, sizeof e->from, "%s", I->text);
		e->line = ins_line(k);
		src_line(e->line, e->src, sizeof e->src);
		e->form = form;
		e->before = before;
		e->after = after;
	}
}

/* the module text with the conversions i2f_pass kept written by their
 * bits, where each is still in it as it was (*out, malloc'd); the count,
 * and the report's section */
static int
i2f_patch(const char *text, char **out)
{
	struct sbuf o = {0}, decl = {0}, rep = {0}, dec = {0};
	const char *p = text, *q;
	int n = 0, i, declared = 0, decorated = 0, used[32] = {0}, base = 0;
	int nrnd = 0, nlog = 0;

	*out = NULL;
	if (ncvx == 0)
		return 0;
	/* which conversions are still there, as they were */
	while (*p) {
		const char *le = strchr(p, '\n');
		size_t ln = le ? (size_t)(le - p) : strlen(p);
		char buf[256], t[8][48];
		int nt = 0;
		if (ln < sizeof buf) {
			const char *q;
			memcpy(buf, p, ln);
			buf[ln] = 0;
			for (q = strtok(buf, " \t"); q && nt < 8; q = strtok(NULL, " \t"))
				snprintf(t[nt++], sizeof t[0], "%s", q);
		}
		for (i = 0; nt == 5 && i < ncvx; i++)
			if (!used[i] && !strcmp(t[0], cvx[i].res) && !strcmp(t[1],
			    "=") && !strcmp(t[2], cvx[i].op) && !strcmp(t[3],
			    cvx[i].rt) && !strcmp(t[4], cvx[i].x))
				used[i] = 1;
		for (i = 0; nt == 8 && i < ncvx; i++)	/* an ldexp */
			if (!used[i] && cvx[i].form == 9 && !strcmp(t[0], cvx[i].res) &&
			    !strcmp(t[1], "=") && !strcmp(t[2], "OpExtInst") &&
			    !strcmp(t[3], cvx[i].rt) && !strcmp(t[5], "Ldexp") &&
			    !strcmp(t[6], cvx[i].x) && !strcmp(t[7], cvx[i].k))
				used[i] = 1;
		for (i = 0; nt == 7 && i < ncvx; i++)	/* a rounding call */
			if (!used[i] && CALLFORM(cvx[i].form) && !strcmp(t[0],
			    cvx[i].res) && !strcmp(t[1], "=") && !strcmp(t[2],
			    "OpExtInst") && !strcmp(t[3], cvx[i].rt) &&
			    !strcmp(t[5], cvx[i].k) && !strcmp(t[6], cvx[i].x))
				used[i] = 1;
		p += ln + (le != NULL);
	}
	for (i = 0; i < ncvx; i++)
		n += used[i];
	if (n == 0)
		return 0;
	/* names past those of a module simplified before */
	for (q = text; (q = strstr(q, "%ucas_cv")) != NULL; q += 8)
		if (isdigit((unsigned char)q[8]) && atoi(q + 8) >= base)
			base = atoi(q + 8) + 1;
	/* the constants, once each */
	for (i = 0; i < ncvx; i++) {
		const struct cvx *e = &cvx[i];
		char kn[96], fn[96];
		unsigned bits = e->form ? 0x4B400000u : 0x4B000000u;
		if (!used[i])
			continue;
		if (e->form == 9) {	/* 23 in k's type; 0 in x's */
			snprintf(kn, sizeof kn, "%%ucas_cvk_%s_%08x = ", e->kt + 1, 23u);
			if (!strstr(decl.p ? decl.p : "", kn) && !strstr(text, kn))
				sb_printf(&decl, "%sOpConstant %s 23\n", kn, e->kt);
			snprintf(fn, sizeof fn, "%%ucas_cvf_%s_00000000 = ", e->rt + 1);
			if (e->zero && !strstr(decl.p ? decl.p : "", fn) &&
			    !strstr(text, fn))
				sb_printf(&decl, "%sOpConstant %s 0\n", fn, e->rt);
			if (e->zero && !strcmp(e->kt2, "%ucas_bool") &&
			    !strstr(decl.p ? decl.p : "", "%ucas_bool = ") &&
			    !strstr(text, "%ucas_bool = "))
				sb_puts(&decl, "%ucas_bool = OpTypeBool\n");
			/* the multiply: 127 << 23 in k's type, and it stays one
			 * rounding */
			snprintf(kn, sizeof kn, "%%ucas_cvk_%s_%08x = ", e->kt + 1,
			    0x3F800000u);
			if (e->mul && !strstr(decl.p ? decl.p : "", kn) &&
			    !strstr(text, kn))
				sb_printf(&decl, "%sOpConstant %s %u\n", kn, e->kt,
				    0x3F800000u);
			if (e->mul)
				sb_printf(&dec, "OpDecorate %s NoContraction\n", e->res);
			nlog++;
			continue;
		}
		if (e->form >= 10) {	/* FindUMsb: no constant */
			nlog++;
			continue;
		}
		if (e->form >= 7) {	/* 23 and 127 in the integer type */
			const char *it = e->form == 7 ? e->it : e->rt;
			unsigned kv[2] = {23, 127};
			int j;
			for (j = 0; j < 2; j++) {
				snprintf(kn, sizeof kn, "%%ucas_cvk_%s_%08x = ", it + 1, kv[j]);
				if (!strstr(decl.p ? decl.p : "", kn) && !strstr(text, kn))
					sb_printf(&decl, "%sOpConstant %s %u\n", kn, it, kv[j]);
			}
			nlog++;
			continue;
		}
		if (e->form >= 4) {	/* 1.5*2^23, and 1 for floor and ceil */
			snprintf(fn, sizeof fn, "%%ucas_cvf_%s_4b400000 = ", e->rt + 1);
			if (!strstr(decl.p ? decl.p : "", fn) && !strstr(text, fn))
				sb_printf(&decl, "%sOpConstant %s 12582912\n", fn, e->rt);
			snprintf(fn, sizeof fn, "%%ucas_cvf_%s_3f800000 = ", e->rt + 1);
			if (e->form > 4 && !strstr(decl.p ? decl.p : "", fn) &&
			    !strstr(text, fn))
				sb_printf(&decl, "%sOpConstant %s 1\n", fn, e->rt);
			if (e->form > 4 && !strcmp(e->kt, "%ucas_bool") &&
			    !strstr(decl.p ? decl.p : "", "%ucas_bool = ") &&
			    !strstr(text, "%ucas_bool = "))
				sb_puts(&decl, "%ucas_bool = OpTypeBool\n");
			/* the add and the subtraction stay as written */
			sb_printf(&dec, "OpDecorate %%ucas_cv%do NoContraction\n",
			    base + i);
			if (e->form == 4)
				sb_printf(&dec, "OpDecorate %s NoContraction\n", e->res);
			else
				sb_printf(&dec, "OpDecorate %%ucas_cv%dr NoContraction\n",
				    base + i);
			nrnd++;
			continue;
		}
		if (e->form == 3) {	/* 0x4B400000 in the result's type, 1.5*2^23 */
			snprintf(kn, sizeof kn, "%%ucas_cvk_%s_4b400000 = ", e->rt + 1);
			snprintf(fn, sizeof fn, "%%ucas_cvf_%s_4b400000 = ", e->xt + 1);
			if (!strstr(decl.p ? decl.p : "", kn) && !strstr(text, kn))
				sb_printf(&decl, "%sOpConstant %s 1262485504\n", kn, e->rt);
			if (!strstr(decl.p ? decl.p : "", fn) && !strstr(text, fn))
				sb_printf(&decl, "%sOpConstant %s 12582912\n", fn, e->xt);
			continue;
		}
		if (e->form == 2) {	/* 127 and 23, in k's type */
			unsigned kv[2] = {127, 23};
			int j;
			for (j = 0; j < 2; j++) {
				snprintf(kn, sizeof kn, "%%ucas_cvk_%s_%08x = ", e->kt + 1,
				    kv[j]);
				if (!strstr(decl.p ? decl.p : "", kn) && !strstr(text, kn))
					sb_printf(&decl, "%sOpConstant %s %u\n", kn, e->kt,
					    kv[j]);
			}
			continue;
		}
		snprintf(kn, sizeof kn, "%%ucas_cvk_%s_%08x = ", e->xt + 1, bits);
		snprintf(fn, sizeof fn, "%%ucas_cvf_%s_%08x = ", e->rt + 1, bits);
		if (!strstr(decl.p ? decl.p : "", kn) && !strstr(text, kn))
			sb_printf(&decl, "%sOpConstant %s %u\n", kn, e->xt, bits);
		if (!strstr(decl.p ? decl.p : "", fn) && !strstr(text, fn))
			sb_printf(&decl, "%sOpConstant %s %s\n", fn, e->rt, e->form ?
			    "12582912" : "8388608");
	}
	for (p = text; *p; ) {
		const char *le = strchr(p, '\n');
		size_t ln = le ? (size_t)(le - p) : strlen(p);
		char buf[256], t[8][48];
		int nt = 0, hit = -1;
		if (ln < sizeof buf) {
			const char *q;
			memcpy(buf, p, ln);
			buf[ln] = 0;
			for (q = strtok(buf, " \t"); q && nt < 8; q = strtok(NULL, " \t"))
				snprintf(t[nt++], sizeof t[0], "%s", q);
		}
		/* the decorations close the annotations, ahead of the types */
		if (!decorated && nt >= 3 && !strcmp(t[1], "=") && !strncmp(t[2],
		    "OpType", 6)) {
			if (dec.p)
				sb_puts(&o, dec.p);
			decorated = 1;
		}
		if (!declared && nt >= 3 && !strcmp(t[1], "=") && !strcmp(t[2],
		    "OpFunction")) {
			if (decl.p)	/* none when the module has them all */
				sb_puts(&o, decl.p);
			declared = 1;
		}
		for (i = 0; (nt == 5 || nt == 7 || nt == 8) && i < ncvx && hit < 0;
		    i++)
			if (used[i] && !strcmp(t[0], cvx[i].res) && (nt == 7) ==
			    CALLFORM(cvx[i].form) && (nt == 8) == (cvx[i].form == 9))
				hit = i;
		/* the log2 and floor a form 7 or 8 leaves unused, and their
		 * names */
		{
			int drop = 0, j;
			for (i = 0; i < ncvx && !drop && nt >= 2; i++)
				for (j = 0; used[i] && j < 3 && !drop; j++)
					drop = cvx[i].drop[j][0] && ((!strcmp(t[1], "=") &&
					    !strcmp(t[0], cvx[i].drop[j])) || ((!strcmp(t[0],
					    "OpName") || !strncmp(t[0], "OpDecorate", 10)) &&
					    !strcmp(t[1], cvx[i].drop[j])));
			if (drop) {
				p += ln + (le != NULL);
				continue;
			}
		}
		if (hit >= 0 && cvx[hit].form == 9 && cvx[hit].mul) {
			const struct cvx *e = &cvx[hit];
			int b = base + hit;
			char rng[96];
			if (isfinite(e->flo) && isfinite(e->fhi))
				snprintf(rng, sizeof rng, "in [%.9g, %.9g]", e->flo, e->fhi);
			else
				snprintf(rng, sizeof rng, "with no finite range from its "
				    "definitions");
			sb_printf(&o, "%%ucas_cv%ds = OpShiftLeftLogical %s %s "
			    "%%ucas_cvk_%s_%08x\n", b, e->kt, e->k, e->kt + 1, 23u);
			sb_printf(&o, "%%ucas_cv%do = OpIAdd %s %%ucas_cv%ds "
			    "%%ucas_cvk_%s_%08x\n", b, e->kt, b, e->kt + 1, 0x3F800000u);
			sb_printf(&o, "%%ucas_cv%df = OpBitcast %s %%ucas_cv%do\n", b,
			    e->rt, b);
			sb_printf(&o, "%s = OpFMul %s %s %%ucas_cv%df\n", e->res, e->rt,
			    e->x, b);
			sb_printf(&rep, "    %s\n      ->  x * asfloat((k << 23) + "
			    "0x3F800000)   (cost %g -> %g)\n      k = %s is in [-126, 127], "
			    "proved by SAT from its definitions, so (k << 23) + 0x3F800000 "
			    "are the bits of 2^k, a normal f32; x * 2^k is the correctly "
			    "rounded x 2^k, as ldexp's result is, so the two agree for "
			    "every x, 0, subnormal and infinite included (x = %s, %s); "
			    "the multiply is decorated NoContraction\n%s", e->from,
			    e->before, e->after, e->k, e->x, rng, e->src);
		} else if (hit >= 0 && cvx[hit].form == 9) {
			const struct cvx *e = &cvx[hit];
			int b = base + hit;
			sb_printf(&o, "%%ucas_cv%ds = OpShiftLeftLogical %s %s "
			    "%%ucas_cvk_%s_%08x\n", b, e->kt, e->k, e->kt + 1, 23u);
			sb_printf(&o, "%%ucas_cv%db = OpBitcast %s %s\n", b, e->kt, e->x);
			sb_printf(&o, "%%ucas_cv%do = OpIAdd %s %%ucas_cv%db %%ucas_cv%ds\n",
			    b, e->kt, b, b);
			if (e->zero) {
				sb_printf(&o, "%%ucas_cv%df = OpBitcast %s %%ucas_cv%do\n", b,
				    e->rt, b);
				sb_printf(&o, "%%ucas_cv%dz = OpFOrdEqual %s %s "
				    "%%ucas_cvf_%s_00000000\n", b, e->kt2, e->x, e->rt + 1);
				sb_printf(&o, "%s = OpSelect %s %%ucas_cv%dz %s %%ucas_cv%df\n",
				    e->res, e->rt, b, e->x, b);
			} else
				sb_printf(&o, "%s = OpBitcast %s %%ucas_cv%do\n", e->res,
				    e->rt, b);
			sb_printf(&rep, "    %s\n      ->  %sasfloat(asint(x) + (k << 23))   "
			    "(cost %g -> %g)\n      x = %s is in [%.9g, %.9g] (a range "
			    "from its definitions and the assumptions)%s: %s|x| is normal, "
			    "its exponent field E in a range k = %s keeps in 1..254, proved "
			    "by SAT (%s <= k <= %s), so x * 2^k is normal and its bits are "
			    "x's with k added to E, exactly%s\n%s", e->from, e->zero ?
			    "x == 0 ? x : " : "", e->before, e->after, e->x, e->flo, e->fhi,
			    e->zero ? ", an integer converted, so 0 or 1 at least in "
			    "magnitude" : "", e->zero ? "where it is not 0, " : "", e->k,
			    e->lx, e->it, e->zero ? "; at 0 ldexp is 0, which the select "
			    "keeps" : "", e->src);
		} else if (hit >= 0 && cvx[hit].form >= 10) {
			const struct cvx *e = &cvx[hit];
			int b = base + hit;
			if (e->form == 11) {
				sb_printf(&o, "%%ucas_cv%dm = OpExtInst %s %s FindUMsb %s\n", b,
				    e->it, e->kt2, e->lx);
				sb_printf(&o, "%s = OpConvertSToF %s %%ucas_cv%dm\n", e->res,
				    e->rt, b);
			} else
				sb_printf(&o, "%s = OpExtInst %s %s FindUMsb %s\n", e->res,
				    e->rt, e->kt2, e->lx);
			sb_printf(&rep, "    %s\n      ->  %sFindUMsb(i)%s   (cost %g -> "
			    "%g, the log2%s%s included)\n      x = %s is an integer i = %s "
			    "converted, i in [1, 2^24) (proved by SAT from its definitions "
			    "and the assumptions): the conversion is exact, so "
			    "floor(log2(x)) is i's most significant bit, FindUMsb(i), "
			    "exactly; a GPU's log2 is within 3 ulp, and this is the exact "
			    "value, one Vulkan lets the call give (not with `exact`)\n%s",
			    e->from, e->form == 11 ? "f32(" : "", e->form == 11 ? ")" : "",
			    e->before, e->after, e->drop[1][0] ? ", the floor" : "",
			    e->drop[2][0] ? " and i's conversion" : "", e->fx, e->lx,
			    e->src);
		} else if (hit >= 0 && cvx[hit].form >= 7) {
			const struct cvx *e = &cvx[hit];
			int b = base + hit;
			const char *it = e->form == 7 ? e->it : e->rt;
			sb_printf(&o, "%%ucas_cv%db = OpBitcast %s %s\n", b, it, e->lx);
			sb_printf(&o, "%%ucas_cv%ds = OpShiftRightLogical %s %%ucas_cv%db "
			    "%%ucas_cvk_%s_%08x\n", b, it, b, it + 1, 23u);
			if (e->form == 7) {
				sb_printf(&o, "%%ucas_cv%de = OpISub %s %%ucas_cv%ds "
				    "%%ucas_cvk_%s_%08x\n", b, it, b, it + 1, 127u);
				sb_printf(&o, "%s = OpConvertSToF %s %%ucas_cv%de\n", e->res,
				    e->rt, b);
			} else
				sb_printf(&o, "%s = OpISub %s %%ucas_cv%ds %%ucas_cvk_%s_%08x\n",
				    e->res, it, b, it + 1, 127u);
			sb_printf(&rep, "    %s\n      ->  %s(asuint(x) >> 23) - 127%s   "
			    "(cost %g -> %g, the log2%s included)\n      x = %s is "
			    "positive and normal (its range from its definitions and the "
			    "assumptions starts at 2^-126 at least, ends finite)%s: its "
			    "exponent field less 127 is floor(log2(x)) exactly; a GPU's "
			    "log2 is within 3 ulp, so the call's own floor may be one more "
			    "just below a power of two, and the field is the exact value, "
			    "one Vulkan lets the call give (not with `exact`); %s\n%s",
			    e->from, e->form == 7 ? "f32(" : "", e->form == 7 ? ")" : "",
			    e->before, e->after, e->drop[1][0] ? " and the floor" : "",
			    e->lx, e->drop[1][0] || e->form == 7 ? "" : ", 1 at least, "
			    "so its log2 is not negative and the conversion's truncation "
			    "is the floor", e->form == 7 ? "the log2 goes" : e->drop[1][0] ?
			    "the floor and the log2 go" : "the log2 goes", e->src);
		} else if (hit >= 0 && cvx[hit].form >= 4) {
			const struct cvx *e = &cvx[hit];
			int b = base + hit;
			const char *ty = e->rt, *tn = e->rt + 1;
			sb_printf(&o, "%%ucas_cv%do = OpFAdd %s %s "
			    "%%ucas_cvf_%s_4b400000\n", b, ty, e->x, tn);
			if (e->form == 4)
				sb_printf(&o, "%s = OpFSub %s %%ucas_cv%do "
				    "%%ucas_cvf_%s_4b400000\n", e->res, ty, b, tn);
			else {
				sb_printf(&o, "%%ucas_cv%dr = OpFSub %s %%ucas_cv%do "
				    "%%ucas_cvf_%s_4b400000\n", b, ty, b, tn);
				sb_printf(&o, "%%ucas_cv%dc = %s %s %%ucas_cv%dr %s\n", b,
				    e->form == 5 ? "OpFOrdGreaterThan" : "OpFOrdLessThan",
				    e->kt, b, e->x);
				sb_printf(&o, "%%ucas_cv%ds = %s %s %%ucas_cv%dr "
				    "%%ucas_cvf_%s_3f800000\n", b, e->form == 5 ?
				    "OpFSub" : "OpFAdd", ty, b, tn);
				sb_printf(&o, "%s = OpSelect %s %%ucas_cv%dc %%ucas_cv%ds "
				    "%%ucas_cv%dr\n", e->res, ty, b, b, b);
			}
			sb_printf(&rep, "    %s\n      ->  %s   (cost %g -> %g)\n      "
			    "%s is in [%.9g, %.9g] (a range from its definitions and the "
			    "assumptions; one unit of margin for what the float pass may "
			    "move), within 2^22: x + 1.5*2^23 lies in [2^23, 2^24], where "
			    "a unit in the last place is 1, so the add rounds x to an "
			    "integer, to even, and subtracting 1.5*2^23 is exact%s; the "
			    "add and the subtraction are decorated NoContraction, so a "
			    "driver keeps them; a zero result may be +0 for -0 (Vulkan's "
			    "default float controls, without "
			    "SignedZeroInfNanPreserve)\n%s",
			    e->from, e->form == 4 ? "(x + 1.5*2^23) - 1.5*2^23" :
			    e->form == 5 ? "r = (x + 1.5*2^23) - 1.5*2^23; "
			    "r > x ? r - 1 : r" : "r = (x + 1.5*2^23) - 1.5*2^23; "
			    "r < x ? r + 1 : r",
			    e->before, e->after, e->x, e->flo, e->fhi, e->form == 4 ?
			    (!strcmp(e->k, "Round") ? " (round's halves to even, one of "
			    "the two directions GLSL lets it take)" : "") : e->form == 5 ?
			    (e->k[0] == 'T' ? "; one less where that is above x: the "
			    "floor, which is trunc for x >= 0" : "; one less where that "
			    "is above x: the floor") : (e->k[0] == 'T' ? "; one more "
			    "where that is below x: the ceiling, which is trunc for x <= "
			    "0" : "; one more where that is below x: the ceiling"),
			    e->src);
		} else if (hit >= 0 && cvx[hit].form == 3) {
			const struct cvx *e = &cvx[hit];
			sb_printf(&o, "%%ucas_cv%do = OpFAdd %s %s %%ucas_cvf_%s_4b400000\n",
			    base + hit, e->xt, e->x, e->xt + 1);
			sb_printf(&o, "%%ucas_cv%db = OpBitcast %s %%ucas_cv%do\n",
			    base + hit, e->rt, base + hit);
			sb_printf(&o, "%s = OpISub %s %%ucas_cv%db %%ucas_cvk_%s_4b400000\n",
			    e->res, e->rt, base + hit, e->rt + 1);
			sb_printf(&rep, "    %s\n      ->  OpISub %s (OpBitcast %s (OpFAdd "
			    "%s %s 1.5*2^23)) 0x4B400000   (cost %g -> %g)\n      %s is a "
			    "whole number in [%.9g, %.9g] (floor, ceil, round or trunc of a "
			    "range from its definitions and the assumptions; one unit of "
			    "margin for what the float pass may move under it), within "
			    "2^22: x + 1.5*2^23 is an integer in [2^23, 2^24), exact, and "
			    "its bits are 0x4B400000 + x\n%s", e->from, e->rt, e->rt,
			    e->xt, e->x, e->before, e->after, e->x, e->flo, e->fhi, e->src);
		} else if (hit >= 0 && cvx[hit].form == 2) {
			const struct cvx *e = &cvx[hit];
			sb_printf(&o, "%%ucas_cv%do = OpIAdd %s %s %%ucas_cvk_%s_%08x\n",
			    base + hit, e->kt, e->k, e->kt + 1, 127u);
			sb_printf(&o, "%%ucas_cv%db = OpShiftLeftLogical %s %%ucas_cv%do "
			    "%%ucas_cvk_%s_%08x\n", base + hit, e->kt, base + hit,
			    e->kt + 1, 23u);
			sb_printf(&o, "%s = OpBitcast %s %%ucas_cv%db\n", e->res, e->rt,
			    base + hit);
			sb_printf(&rep, "    %s\n      ->  OpBitcast %s (OpShiftLeftLogical "
			    "%s (OpIAdd %s %s 127) 23)   (cost %g -> %g, the shift 1 << %s "
			    "included)\n      f32(1 << k) is asfloat((k + 127) << 23): the "
			    "exponent field of 2^k, mantissa 0, for every k a shift by it "
			    "defines (0..31)%s\n%s", e->from, e->rt, e->kt, e->kt, e->k,
			    e->before, e->after, e->k, e->op[9] == 'S' ? "; signed, k < 31 "
			    "proved by SAT (1 << 31 is negative)" : "", e->src);
		} else if (hit >= 0) {
			const struct cvx *e = &cvx[hit];
			unsigned bits = e->form ? 0x4B400000u : 0x4B000000u;
			sb_printf(&o, "%%ucas_cv%do = %s %s %s %%ucas_cvk_%s_%08x\n",
			    base + hit, e->form ? "OpIAdd" : "OpBitwiseOr", e->xt, e->x,
			    e->xt + 1, bits);
			sb_printf(&o, "%%ucas_cv%db = OpBitcast %s %%ucas_cv%do\n",
			    base + hit, e->rt, base + hit);
			sb_printf(&o, "%s = OpFSub %s %%ucas_cv%db %%ucas_cvf_%s_%08x\n",
			    e->res, e->rt, base + hit, e->rt + 1, bits);
			sb_printf(&rep, "    %s\n      ->  OpFSub %s (OpBitcast %s (%s "
			    "%s %s %s)) %s   (cost %g -> %g)\n      %s\n%s", e->from,
			    e->rt, e->rt, e->form ? "OpIAdd" : "OpBitwiseOr", e->xt,
			    e->x, e->form ? "0x4B400000" : "0x4B000000", e->form ?
			    "1.5*2^23" : "2^23", e->before, e->after, e->form ?
			    "-2^22 <= x < 2^22, proved by SAT: 0x4B400000 are the bits "
			    "of 1.5*2^23, and x moves the mantissa inside [0, 2^23) "
			    "without reaching the exponent: the bits of 1.5*2^23 + x "
			    "exactly; subtracting 1.5*2^23 leaves x, below 2^24: exact" :
			    "0 <= x < 2^23, proved by SAT: x | 0x4B000000 are the bits "
			    "of the f32 2^23 + x (one unit in the last place there is "
			    "1); subtracting 2^23 leaves x, below 2^24: exact", e->src);
		} else {
			sb_printf(&o, "%.*s\n", (int)ln, p);
		}
		p += ln + (le != NULL);
	}
	if (nlog)
		printf("conversions between integers and f32 by the bits, roundings "
		    "by the add of 1.5*2^23, and floor(log2 x) and ldexp by the "
		    "exponent field, "
		    "under %s (written last: the passes above read the conversions; "
		    "a conversion is not in the integer cost):\n",
		    sym_str(model_active()->name));
	else if (nrnd < n)
		printf("conversions between integers and f32 by the bits%s, under "
		    "%s (written last: the passes above read the conversions; a "
		    "conversion is not in the integer cost):\n", nrnd ? ", and "
		    "roundings by the add of 1.5*2^23" : "",
		    sym_str(model_active()->name));
	else
		printf("roundings of f32 by the add of 1.5*2^23, under %s (written "
		    "last, with the conversions by the bits):\n",
		    sym_str(model_active()->name));
	fputs(rep.p, stdout);
	{
		char why[200];
		int v = dw_validate(o.p, why, sizeof why);
		if (v == 1)
			printf("  spirv-val accepts the written module\n");
		else if (v == 0)
			printf("  spirv-val REJECTS the written module: %s\n", why);
		else
			printf("  spirv-as and spirv-val cannot run here: the written "
			    "module is not validated\n");
	}
	rep_list("conversions_by_bits");
	for (i = 0; i < ncvx; i++) {
		if (!used[i])
			continue;
		rep_obj(NULL);
		rep_str("from", cvx[i].from);
		rep_str("form", cvx[i].form == 10 ? "FindUMsb(i)" : cvx[i].form ==
		    11 ? "f32(FindUMsb(i))" : cvx[i].form == 9 && cvx[i].mul ?
		    "x * asfloat((k << 23) + 0x3F800000)" :
		    cvx[i].form == 9 && cvx[i].zero ?
		    "x == 0 ? x : "
		    "asfloat(asint(x) + (k << 23))" : cvx[i].form == 9 ?
		    "asfloat(asint(x) + (k << 23))" :
		    cvx[i].form == 7 ? "f32((asuint(x) >> 23) - 127)" :
		    cvx[i].form == 8 ? "(asuint(x) >> 23) - 127" :
		    cvx[i].form == 4 ? "(x + 1.5*2^23) - 1.5*2^23" :
		    cvx[i].form == 5 ? "r = (x + 1.5*2^23) - 1.5*2^23; r > x ? "
		    "r - 1 : r" : cvx[i].form == 6 ? "r = (x + 1.5*2^23) - "
		    "1.5*2^23; r < x ? r + 1 : r" : cvx[i].form == 3 ?
		    "asint(x + 1.5*2^23) - "
		    "0x4B400000" : cvx[i].form == 2 ? "asfloat((k + 127) << 23)" :
		    cvx[i].form ? "asfloat(x + 0x4B400000) - 1.5*2^23" :
		    "asfloat(x | 0x4B000000) - 2^23");
		rep_int("line", cvx[i].line);
		rep_num("cost_before", cvx[i].before);
		rep_num("cost_after", cvx[i].after);
		rep_end();
	}
	rep_end();
	free(decl.p);
	free(dec.p);
	free(rep.p);
	*out = o.p;
	return n;
}

/* an instruction the cost cover counts: a pure one, or a multiply
 * extended (its struct result read through extracts) */
static int
cover_ins(int k)
{
	const struct sins *I = &V.in[k];

	if (!SP.dead[k] && I->res >= 0 && (!strcmp(I->op, "OpUMulExtended") ||
	    !strcmp(I->op, "OpSMulExtended")))
		return 1;
	return pure_ins(k);
}

/*
 * The live integer instructions as a DAG for the model's fusions
 * (fuse.c): in order, with their operands that are not such
 * instructions (loads, constants, phis) as leaves; a result something
 * else uses (a store, a phi, a branch) is computed as it is.
 */
/* ---- loops of constant bounds: each instruction's passes ---- */

/* id as a small non-negative constant, -1 else */
static int64_t
small_const(int id)
{
	if (id < 0 || V.id[id].kind != K_CONST || V.id[id].spec ||
	    V.id[id].c[0] >= (1ULL << 31))
		return -1;
	return (int64_t)V.id[id].c[0];
}

/*
 * A structured loop's passes from its header h (merge m, continue c):
 * the first conditional branch on the way from h (through unconditional
 * branches) that leaves to m, on a compare of a phi of h with a constant,
 * the phi from a constant before the loop and phi + a constant from c.
 * -1 when the bounds are not constants of that shape.  iv, when given,
 * gets that phi, its first value and its step.
 */
static int64_t
loop_trips_iv(int h, int m, int c, int *iv, int64_t *first, int64_t *inc)
{
	int pc, hops, lbl = h, cond = -1, stay = 0, k, phi = -1, i;
	int64_t init = -1, step = -1, bound = -1;
	const struct sins *I = NULL;
	for (hops = 0; hops < 3; hops++) {
		for (pc = block_start(lbl); pc < V.nin; pc++) {
			I = &V.in[pc];
			if (!strcmp(I->op, "OpBranch") ||
			    !strcmp(I->op, "OpBranchConditional") ||
			    !strncmp(I->op, "OpReturn", 8) || !strcmp(I->op, "OpSwitch") ||
			    !strcmp(I->op, "OpKill") || !strcmp(I->op, "OpUnreachable"))
				break;
		}
		if (pc >= V.nin)
			return -1;
		if (!strcmp(I->op, "OpBranch") && I->nop >= 1 && I->opnd[0][0] == '%') {
			lbl = idx(I->opnd[0]);
			if (lbl == m || V.id[lbl].kind != K_LABEL)
				return -1;
			continue;
		}
		if (strcmp(I->op, "OpBranchConditional") || I->nop < 3)
			return -1;
		cond = idx(I->opnd[0]);
		if (idx(I->opnd[2]) == m)
			stay = 1;	/* true: stay in the loop */
		else if (idx(I->opnd[1]) == m)
			stay = 0;
		else
			return -1;
		break;
	}
	if (cond < 0 || SP.def == NULL || SP.def[cond] < 0)
		return -1;
	I = &V.in[SP.def[cond]];
	if (I->nop < 3)
		return -1;
	{
		int a = idx(I->opnd[1]), b = idx(I->opnd[2]), d;
		const char *op = I->op;
		int lt = !strcmp(op, "OpULessThan") || !strcmp(op, "OpSLessThan");
		int le = !strcmp(op, "OpULessThanEqual") ||
		    !strcmp(op, "OpSLessThanEqual");
		int ge = !strcmp(op, "OpUGreaterThanEqual") ||
		    !strcmp(op, "OpSGreaterThanEqual");
		int gt = !strcmp(op, "OpUGreaterThan") || !strcmp(op, "OpSGreaterThan");
		/* the phi on the left: i < K (stay) or i >= K (leave) */
		if (!(stay ? lt || le : ge || gt))
			return -1;
		phi = a;
		bound = small_const(b);
		if (bound < 0 || phi < 0 || SP.def[phi] < 0)
			return -1;
		I = &V.in[SP.def[phi]];
		if (strcmp(I->op, "OpPhi") || I->nop != 5)
			return -1;
		/* the header is the phi's block: (value, parent) pairs */
		for (k = 1; k + 1 < I->nop; k += 2) {
			int v = idx(I->opnd[k]), from = idx(I->opnd[k + 1]);
			if (from == c) {
				const struct sins *N;
				if (v < 0 || SP.def[v] < 0)
					return -1;
				N = &V.in[SP.def[v]];
				if (strcmp(N->op, "OpIAdd") || N->nop < 3)
					return -1;
				d = idx(N->opnd[1]) == phi ? idx(N->opnd[2]) :
				    idx(N->opnd[2]) == phi ? idx(N->opnd[1]) : -1;
				step = small_const(d);
			} else
				init = small_const(v);
		}
		if (init < 0 || step <= 0)
			return -1;
		if (iv) {
			*iv = phi;
			*first = init;
			*inc = step;
		}
		if (lt || ge) {
			/* passes while i < bound (or until i >= bound) */
			return bound > init ? (bound - init + step - 1) / step : 0;
		}
		return bound >= init ? (bound - init) / step + 1 : 0;
	}
	(void)i;
}

static int64_t
loop_trips(int h, int m, int c)
{
	return loop_trips_iv(h, m, c, NULL, NULL, NULL);
}

/*
 * Each instruction's passes (wt, by index): the product of the trip
 * counts of the structured loops of constant bounds around its block, 1
 * outside them and in loops of other bounds.  The loops counted, or 0.
 */
static int
loop_weights(double *wt)
{
	int k, n = 0, *blk, nb = 0, *bstart, i;
	uint8_t *in;
	for (k = 0; k < V.nin; k++)
		wt[k] = 1;
	blk = xmalloc(((size_t)V.nin + 1) * sizeof *blk);
	bstart = xmalloc(((size_t)V.nin + 1) * sizeof *bstart);
	for (k = 0; k < V.nin; k++) {
		if (!strcmp(V.in[k].op, "OpLabel"))
			bstart[nb++] = k;
		blk[k] = nb - 1;
	}
	in = xmalloc((size_t)nb + 1);
	for (i = 0; i < nb; i++) {
		int h = V.in[bstart[i]].res, m = -1, c = -1, pc, sp = 0, *st;
		int64_t trips;
		for (pc = bstart[i] + 1; pc < V.nin && strcmp(V.in[pc].op,
		    "OpLabel"); pc++)
			if (!strcmp(V.in[pc].op, "OpLoopMerge") && V.in[pc].nop >= 2) {
				m = idx(V.in[pc].opnd[0]);
				c = idx(V.in[pc].opnd[1]);
			}
		if (m < 0 || h < 0 || (trips = loop_trips(h, m, c)) < 0)
			continue;
		/* the loop's blocks: reached from its header, not through its
		 * merge */
		memset(in, 0, (size_t)nb + 1);
		st = xmalloc(((size_t)nb + 1) * sizeof *st);
		st[sp++] = blk[V.id[h].label_at];
		in[st[0]] = 1;
		while (sp) {
			int b = st[--sp];
			const struct sins *T2;
			int e = b + 1 < nb ? bstart[b + 1] : V.nin, q;
			for (pc = bstart[b] + 1; pc < e; pc++) {
				T2 = &V.in[pc];
				if (strcmp(T2->op, "OpBranch") &&
				    strcmp(T2->op, "OpBranchConditional") &&
				    strcmp(T2->op, "OpSwitch"))
					continue;
				for (q = !strcmp(T2->op, "OpBranch") ? 0 : 1; q < T2->nop;
				    q++) {
					int l, nbk;
					if (T2->opnd[q][0] != '%')
						continue;
					l = idx(T2->opnd[q]);
					if (l < 0 || l == m || V.id[l].kind != K_LABEL)
						continue;
					nbk = blk[V.id[l].label_at];
					if (!in[nbk]) {
						in[nbk] = 1;
						st[sp++] = nbk;
					}
				}
			}
		}
		free(st);
		for (k = 0; k < V.nin; k++)
			if (blk[k] >= 0 && in[blk[k]])
				wt[k] *= (double)trips;
		n++;
	}
	free(in);
	free(blk);
	free(bstart);
	return n;
}

/* spv_cover's weights, when set: each instruction's passes (by index),
 * and the total they give */
static double *cover_wt, cover_wsum;

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
	int nins = 0, nleaf = 0, k, i, j, q, o, *ord, no = 0;
	struct fgraph G;
	double total;

	for (i = 0; i < nid_; i++)
		gi[i] = -1;
	/* in the module's order: what a rewrite added goes before the
	 * instruction it was written for, so an operand comes before its
	 * user */
	ord = xmalloc((size_t)(V.nin ? V.nin : 1) * sizeof *ord);
	for (k = 0; k < SP.orig_nin; k++) {
		for (q = SP.orig_nin; q < V.nin; q++)
			if (SP.before[q] == k)
				ord[no++] = q;
		ord[no++] = k;
	}
	for (q = SP.orig_nin; q < V.nin; q++)
		if (SP.before[q] < 0 || SP.before[q] >= SP.orig_nin)
			ord[no++] = q;
	for (o = 0; o < no; o++)
		if (cover_ins(ord[o]))
			gi[V.in[ord[o]].res] = nins++;
	for (o = 0; o < no; o++)
		if (cover_ins(k = ord[o]))
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
	for (o = 0; o < no; o++) {
		const struct sins *I = &V.in[k = ord[o]];
		double mul;
		int base;
		if (!cover_ins(k))
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
		if (SP.dead[k] || cover_ins(k) || !strncmp(I->op, "OpName", 6) ||
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
	/* what no result needs (a rewrite took its last use; dead code
	 * removal drops it later) costs nothing */
	{
		uint8_t *live = xcalloc((size_t)(nins ? nins : 1), 1);
		int *st = xmalloc((size_t)(nins ? nins : 1) * sizeof *st), sp = 0;
		for (j = 0; j < nins; j++)
			if (G.pinned[j]) {
				live[j] = 1;
				st[sp++] = j;
			}
		while (sp) {
			int x = st[--sp];
			for (q = 0; q < 3; q++) {
				int a = G.arg[x][q];
				if (a >= 0 && a < nins && !live[a]) {
					live[a] = 1;
					st[sp++] = a;
				}
			}
		}
		for (j = 0; j < nins; j++)
			if (!live[j])
				G.own[j] = 0;
		free(live);
		free(st);
	}
	total = fg_cover(model_active(), &G);
	if (cover_wt) {
		/* a fusion is priced at its root: the root's passes */
		cover_wsum = 0;
		for (j = 0; j < nins; j++) {
			int at = (int)G.id[j];
			double w = at < SP.orig_nin ? cover_wt[at] : SP.before[at] >= 0 &&
			    SP.before[at] < SP.orig_nin ? cover_wt[SP.before[at]] : 1;
			cover_wsum += G.eff[j] * w;
		}
	}
	if (t)
		fg_tally(&G, NULL, 1, t);
	fg_free(&G);
	free(gi);
	free(ord);
	return total;
}

void
spirv_simplify(const char *src, double budget, int nofile)
{
	int k, i, nrw = 0, ndead = 0, inbody = 0, nlog = 0, nfwd = 0, ndst = 0;
	int ndotrw = 0;
	int ntriv = 0, nloops = 0;
	double tgain = 0, *lwt = NULL, wbefore = 0, wafter = 0;
	char *range_report = NULL;
#define MAXLOG 4096
	struct rwlog {
		char from[160], to[160];
		const char *why;
		double gain;
		int line, algebra, trivial, dot;
	} *rl = xcalloc(MAXLOG, sizeof *rl);
	struct dotlog dotl[32], idl[32], sgl[32];
	int ndotlog = 0, nidlog = 0, nsglog = 0, ntex = 0, ntexw = 0;
	struct texrd trd[16];
	struct sbuf texrep = {0};
	double before = 0, after = 0;
	struct fuse_tally ftally;
	struct sbuf out = {0}, list = {0}, dotrep = {0}, idrep = {0};
	struct sbuf sgrep = {0}, sgprop = {0}, sgadv = {0};
	struct sbuf idrep2 = {0}, idrep3 = {0}, idrep4 = {0};
	int nidk[5] = {0, 0, 0, 0, 0};	/* idrw's rewrites by kind */
	clock_t t0 = clock();

	memset(&V, 0, sizeof V);
	symmap_reset();
	memset(&SP, 0, sizeof SP);
	ncvx = 0;
	dotcaps = 0;
	memset(&dotund, 0, sizeof dotund);
	sg_simplify_reset();
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
			glslfn_pass(src_text, budget, 1, entry_model(),
			    dot_declared());
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
		spirv_float_opts(0, 0, 0, 0, 0, 0, 1);
		spirv_sink_opt(NULL);
		spirv_tex_binding(-1, -1);
		advice_on = 0;
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
	/* loops of constant bounds: their instructions counted a pass each,
	 * in a line of their own beside the headline (Liz, #369: "b") */
	lwt = xmalloc(((size_t)V.nin + 1) * sizeof *lwt);
	nloops = loop_weights(lwt);
	cover_wt = nloops ? lwt : NULL;
	before = spv_cover(NULL);
	wbefore = cover_wsum;
	for (k = 0; k < V.nin; k++)
		if (is_idot(k)) {
			int cls;
			before += idot_cost(&V.in[k], &cls);
		}
	before += sg_total(0);
	/* constant table reads through the texture unit, first: the index's
	 * remainder as written is what they replace (the division pass below
	 * would make it a multiply that SAT then has to see through); written
	 * with the request's binding, proposed without it */
	ntex = tex_pass(&texrep, trd, 16);
	for (i = 0; i < ntex; i++)
		ntexw += trd[i].written;
	ndead += tex_nkilled;
	/* divisions by a constant of a bounded dividend, under the model:
	 * first, so that a signed one is not left as an unsigned division */
	{
		int nd = idrw_pass(&idrep, &idrep2, &idrep3, &idrep4, idl, 32), q;
		for (q = 0; q < nd && nlog < MAXLOG; q++) {
			struct rwlog *L = &rl[nlog++];
			snprintf(L->from, sizeof L->from, "%s", idl[q].from);
			snprintf(L->to, sizeof L->to, "%s", idl[q].to);
			nidk[idl[q].kind]++;
			L->why = idl[q].kind == 1 ? "a divisibility test as a "
			    "multiply and a compare" : idl[q].kind == 2 ? "a map of "
			    "one small-range value as a table or a multiply and a "
			    "shift" : idl[q].kind == 3 ? "a factor several terms "
			    "share taken out once" : idl[q].kind == 4 ? "a division "
			    "by a few constants as one by each" : "a division by a "
			    "constant of a bounded dividend as a multiply and a shift";
			L->gain = idl[q].gain;
			L->line = idl[q].line;
			L->dot = 2;
		}
		nidlog = nd;
		nrw += nd;
		SP.proved += nd;
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
			if (L->trivial) {
				/* what it saves where the model already prices
				 * what it makes (x * 8 + y is a shift-add as
				 * written): the cover's, not the instruction's */
				double c0 = spv_cover(NULL);
				apply(k, &cs[best]);
				L->gain = c0 - spv_cover(NULL);
				nrw++;
				SP.proved++;
				continue;
			}
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
	/* sums and scans of bits over the lanes as a ballot's bit count */
	{
		int nd = sg_pass(&sgrep, sgl, 32), q;
		for (q = 0; q < nd && nlog < MAXLOG; q++) {
			struct rwlog *L = &rl[nlog++];
			snprintf(L->from, sizeof L->from, "%s", sgl[q].from);
			snprintf(L->to, sizeof L->to, "%s", sgl[q].to);
			L->why = "a sum of bits over the lanes as a ballot's bit count";
			L->gain = sgl[q].gain;
			L->line = sgl[q].line;
			L->dot = 3;
		}
		nsglog = nd;
		nrw += nd;
		SP.proved += nd;
	}
	/* conversions of integers to f32 by the bits: decided here, written
	 * after the passes that read conversions */
	i2f_pass();
	/* subgroup forms that need the pipeline's subgroup size: said */
	sg_proposals(&sgprop);
	/* a branch in a loop the lanes could be packed for: advice */
	sg_compaction(&sgadv);
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
	wafter = cover_wsum;
	cover_wt = NULL;
	for (k = 0; k < V.nin; k++)
		if (!SP.dead[k] && is_idot(k)) {
			int cls;
			after += idot_cost(&V.in[k], &cls);
		}
	after += sg_total(1);
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
		int first_fn = -1, last_cap = -1, first_type = -1;
		for (k = 0; k < SP.orig_nin; k++) {
			if (!strcmp(V.in[k].op, "OpCapability"))
				last_cap = k;
			if (first_type < 0 && !strncmp(V.in[k].op, "OpType", 6))
				first_type = k;
		}
		for (k = 0; k < V.nin; k++)
			if (!strcmp(V.in[k].op, "OpFunction")) {
				first_fn = k;
				break;
			}
		for (k = 0; k < V.nin; k++) {
			const struct sins *I = &V.in[k];
			int subst = 0;
			if (k == first_fn)
				sg_types_out(&out);
			if (k == first_fn && tex_decl.n)
				sb_puts(&out, tex_decl.p);
			if (k == first_type && tex_annot.n)
				sb_puts(&out, tex_annot.p);
			if (k == first_fn)
				for (i = 0; i < SP.nmxty; i++)
					sb_printf(&out, "%s = OpTypeStruct %s %s\n",
					    V.id[SP.mxty[i]].tok,
					    V.id[T(SP.mxty[i])->m[0]].tok,
					    V.id[T(SP.mxty[i])->m[0]].tok);
			if (k == first_fn)
				for (i = 0; i < SP.nnewk; i++) {
					struct sid *d = &V.id[SP.newk[i]];
					if (!const_used(SP.newk[i]))
						continue;
					if (int_width(d->type) == 1)
						sb_printf(&out, "%s = OpConstant%s %s\n", d->tok,
						    d->c[0] ? "True" : "False", V.id[d->type].tok);
					else if (T(d->type)->sign)	/* spirv-as reads it signed */
						sb_printf(&out, "%s = OpConstant %s %lld\n", d->tok,
						    V.id[d->type].tok, (long long)sx64(d->c[0],
						    int_width(d->type)));
					else
						sb_printf(&out, "%s = OpConstant %s %llu\n", d->tok,
						    V.id[d->type].tok, (unsigned long long)d->c[0]);
				}
			if (k >= SP.orig_nin)
				break;
			if (dotcaps && k == last_cap + 1)
				dot_caps_out(&out);
			if (sgcaps && k == last_cap + 1)
				sg_caps_out(&out);
			if (tex_cap && k == last_cap + 1)
				sb_puts(&out, "OpCapability Sampled1D\n");
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
		    ndotrw - nidlog - nsglog, nrw - nal - ndotrw - nidlog - nsglog ==
		    1 ? "" : "s");
		if (nal)
			printf(" and %d by algebra mod 2^w", nal);
		if (ndotrw)
			printf(", %d sum%s of products made dot products (proved by "
			    "dot.c, below)", ndotrw, ndotrw == 1 ? "" : "s");
		if (nidk[0])
			printf(", %d division%s by a constant made a multiply and a "
			    "shift (derived, below)", nidk[0], nidk[0] == 1 ? "" : "s");
		if (nidk[1])
			printf(", %d divisibility test%s made a multiply and a compare "
			    "(derived, below)", nidk[1], nidk[1] == 1 ? "" : "s");
		if (nidk[2])
			printf(", %d map%s of one value made a table or a multiply "
			    "(derived, below)", nidk[2], nidk[2] == 1 ? "" : "s");
		if (nidk[3])
			printf(", %d sum%s with a factor in several terms made one "
			    "product of it (derived, below)", nidk[3], nidk[3] == 1 ?
			    "" : "s");
		if (nidk[4])
			printf(", %d division%s by a value of a few constants made "
			    "one by each (derived, below)", nidk[4], nidk[4] == 1 ?
			    "" : "s");
		if (nsglog)
			printf(", %d sum%s of bits over the lanes made a ballot's bit "
			    "count (derived, below)", nsglog, nsglog == 1 ? "" : "s");
		if (ntexw)
			printf(", %d table read%s made a read through the texture "
			    "unit (below)", ntexw, ntexw == 1 ? "" : "s");
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
	if (nloops) {
		rep_int("loops_weighted", nloops);
		rep_num("cost_weighted_before", wbefore);
		rep_num("cost_weighted_after", wafter);
	}
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
		rep_str("proof", rl[i].algebra ? "algebra mod 2^w" : rl[i].dot >= 2 ?
		    "derivation" : rl[i].dot ? "dot" : "sat");
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
	if (dotund.n) {
		rep_obj("dot_products_not_declared");
		rep_int("sums", dotund.n);
		rep_num("gain", dotund.gain);
		rep_str("needs", "SPV_KHR_integer_dot_product");
		rep_end();
	}
	if (nsglog) {
		rep_list("subgroup_counts");
		for (i = 0; i < nsglog; i++) {
			rep_obj(NULL);
			rep_str("from", sgl[i].from);
			rep_str("to", sgl[i].to);
			rep_int("line", sgl[i].line);
			rep_num("cost_before", sgl[i].before);
			rep_num("cost_after", sgl[i].after);
			rep_str("holds_where", sgl[i].desc);
			rep_str("derivation", sgl[i].proof);
			rep_end();
			free(sgl[i].desc);
			free(sgl[i].proof);
		}
		rep_end();
	}
	if (nidlog) {
		rep_list("identities");
		for (i = 0; i < nidlog; i++) {
			rep_obj(NULL);
			rep_str("from", idl[i].from);
			rep_str("to", idl[i].to);
			rep_int("line", idl[i].line);
			rep_num("cost_before", idl[i].before);
			rep_num("cost_after", idl[i].after);
			rep_str("holds_where", idl[i].desc);
			rep_str("derivation", idl[i].proof);
			if (idl[i].kind)
				rep_str("kind", idl[i].kind == 4 ? "divisor_cases" :
				    idl[i].kind == 3 ? "common_factor" :
				    idl[i].kind == 1 ? "divisibility" :
				    "map");
			rep_end();
			free(idl[i].desc);
			free(idl[i].proof);
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
	if (nloops) {
		printf("  weighted by the trip counts of %d loop%s of constant bounds "
		    "(each instruction once a pass, an early exit making it less): "
		    "%g -> %g", nloops, nloops > 1 ? "s" : "", wbefore, wafter);
		if (wbefore > 0)
			printf(" (%+.1f%%)", 100 * (wafter - wbefore) / wbefore);
		printf("\n");
	}
	free(lwt);
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
		/* every one: a list cut short sent readers to the JSON */
		shown = nlog - ntriv;
		printf("  rewrites:\n");
		for (i = 0; i < shown; i++) {
			char sl[160];
			src_line(rl[i].line, sl, sizeof sl);
			sb_printf(&list, "    %s\n      ->  %s   (%s; -%g%s)\n%s",
			    rl[i].from, rl[i].to, rl[i].why, rl[i].gain, rl[i].algebra ?
			    "; by algebra mod 2^w, not SAT" : "", sl);
		}
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
	if (dotund.n)
		printf("  dot products: %d sum%s of products would be %s under %s "
		    "(cost -%g), not written: the module does not declare "
		    "SPV_KHR_integer_dot_product (OpCapability DotProduct; in GLSL, "
		    "#extension GL_EXT_integer_dot_product)\n", dotund.n, dotund.n ==
		    1 ? "" : "s", dotund.n == 1 ? "a dot product" : "dot products",
		    sym_str(model_active()->name), dotund.gain);
	free(dotrep.p);
	if (idrep.n) {
		printf("  divisions by a constant (the dividend's bound makes one "
		    "multiply and one shift exact; derived by idrw.c, kept under "
		    "%s):\n", sym_str(model_active()->name));
		fputs(idrep.p, stdout);
	}
	free(idrep.p);
	if (idrep2.n) {
		printf("  divisibility tests and maps of one value (derived by "
		    "idrw.c: a multiply and a compare, a table in one constant, a "
		    "multiply and a shift; kept under %s):\n",
		    sym_str(model_active()->name));
		fputs(idrep2.p, stdout);
	}
	free(idrep2.p);
	if (idrep3.n) {
		printf("  common factors (a factor several terms of a sum share, "
		    "taken out: one multiply for them; derived by ifx.c, kept "
		    "under %s):\n", sym_str(model_active()->name));
		fputs(idrep3.p, stdout);
	}
	free(idrep3.p);
	if (idrep4.n) {
		printf("  divisions by a value of a few constants (a division by "
		    "each, the case picked by the divisor; the driver makes each "
		    "a multiply and a shift; kept under %s):\n",
		    sym_str(model_active()->name));
		fputs(idrep4.p, stdout);
	}
	free(idrep4.p);
	if (sgprop.n) {
		printf("  subgroup forms that need the subgroup's size fixed, "
		    "proposed (pipeline state: the module is not changed), under "
		    "%s:\n", sym_str(model_active()->name));
		fputs(sgprop.p, stdout);
	}
	free(sgprop.p);
	if (sgadv.n) {
		printf("  in-warp compaction, advice (a restructuring of the loop: "
		    "the module is not changed), under %s:\n",
		    sym_str(model_active()->name));
		fputs(sgadv.p, stdout);
	}
	free(sgadv.p);
	if (sgrep.n) {
		printf("  subgroup sums of bits (a value that is 0 or 1 in every "
		    "lane, summed over the lanes, is a ballot's bit count), under "
		    "%s:\n", sym_str(model_active()->name));
		fputs(sgrep.p, stdout);
		if (nsglog) {
			char why[200];
			int v = dw_validate(out.p ? out.p : "", why, sizeof why);
			if (v == 1)
				printf("  spirv-val accepts the written module\n");
			else if (v == 0)
				printf("  spirv-val REJECTS the written module: %s\n",
				    why);
			else
				printf("  spirv-as and spirv-val cannot run here: the "
				    "written module is not validated\n");
		}
	}
	free(sgrep.p);
	if (texrep.n)
		fputs(texrep.p, stdout);
	free(texrep.p);
	if (ntex) {
		rep_list("texture_reads");
		for (i = 0; i < ntex; i++) {
			rep_obj(NULL);
			rep_str("from", trd[i].from);
			rep_str("mode", tm_name[trd[i].mode]);
			rep_str("format", tex_format(trd[i].elt, trd[i].lo, trd[i].hi));
			rep_int("texels", trd[i].n);
			rep_num("cost_before", trd[i].before);
			rep_num("cost_after", trd[i].after);
			rep_bool("written", trd[i].written);
			rep_end();
		}
		rep_end();
	}
	if (SP.nval) {
		/* a struct type and its extracts, or a texture, the module may
		 * not have had: what spirv-val says of them */
		char why[200];
		int v = dw_validate(out.p ? out.p : "", why, sizeof why);
		if (v == 1)
			printf("  spirv-val accepts the written module\n");
		else if (v == 0)
			printf("  spirv-val REJECTS the written module: %s\n", why);
		else
			printf("  spirv-as and spirv-val cannot run here: the written "
			    "module is not validated\n");
	}
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
		glslfn_pass(src_text, budget, 0, entry_model(),
		    dot_declared());
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
		int fch = 0, dch = 0, mch = 0, ncv;
		char *ft, *dt, *mt, *cvt;
		dw_nfas = npend_fas;	/* float_pass ends the assumptions */
		fopt_nofile = nofile;
		ft = float_pass(out.p ? out.p : "", &fch);
		/* f64 over f32 and folded round trips, on what that left */
		dt = dword_pass(fch ? ft : out.p ? out.p : "", &dch);
		if (dch) {
			free(ft);
			ft = dt;
			fch = 1;
		}
		/* the memory accesses, on what that left */
		mt = mem_pass(fch ? ft : out.p ? out.p : "", &mch);
		if (mch) {
			free(ft);
			ft = mt;
			fch = 1;
		}
		/* the conversions by the bits, last; the advice and the
		 * rounding below read the module before them */
		ncv = i2f_patch(fch ? ft : out.p ? out.p : "", &cvt);
		if (ncv)
			deliver_file("simplified.spvasm", "text/plain", cvt,
			    strlen(cvt), nofile);
		else if (fch)
			deliver_file("simplified.spvasm", "text/plain", ft, strlen(ft),
			    nofile);
		else if (nrw + ndead + ntexw > 0)
			deliver_file("simplified.spvasm", "text/plain", out.p, out.n,
			    nofile);
		/* spirv-cross's GLSL of it, beside spirv-opt -O's */
		if (fch || ncv || nrw + ndead + ntexw > 0)
			spvcross_after_simplify(orig_mod, ncv ? cvt : fch ? ft : out.p,
			    nofile);
		free(cvt);
		advice_pass(fch ? ft : out.p ? out.p : "", advice_on);
		/* the module before the float pass: its rounding patterns
		 * (floor(z + 0.5), an int() of it) are the source's, and an
		 * fma the float pass contracts there would hide them */
		round_pass(out.p ? out.p : "");
		free(ft);
		free(orig_mod);
		orig_mod = NULL;
	}
	free(out.p);
	spirv_float_opts(0, 0, 0, 0, 0, 0, 1);
	spirv_sink_opt(NULL);
	spirv_tex_binding(-1, -1);
	advice_on = 0;
	fnl_mode = 0;
}
