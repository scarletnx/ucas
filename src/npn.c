/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * NPN classes of the boolean functions of 4 inputs, and the minimum
 * network of 3-input lookup tables (LOP3 on NVIDIA, vpternlog on x86)
 * for each.
 *
 * Two functions are NPN-equivalent when one is the other with its inputs
 * permuted, some of them negated and maybe its output negated.  A network
 * of lookup tables absorbs all three (a negated input or output is only
 * another table), so the members of a class share one minimum size.  The
 * 65536 functions of 4 inputs fall into 222 classes.  The first time a
 * class is asked for, gates' exact search (gates_synth) finds the minimum
 * network of its representative, the class's least truth table, and it
 * is kept; a function of the class takes it with its inputs renamed and
 * the negations folded into the tables.
 *
 * find reads the largest bitwise cones of a function (and, or, xor, not
 * of one integer type) with 4 distinct leaves: bit for bit their value is
 * a function of the leaves' bits, so the network, each table written as
 * an expression of its 3 arguments, is a rewrite that the cost model
 * prices like any other (a lut fusion covers each table once).  A cone of
 * 3 leaves or fewer is one table already, which the cover finds.
 */
#include "calc.h"
#include "fuse.h"
#include "irw.h"

#include <stdlib.h>
#include <string.h>

#define NPN_MAXL 4		/* tables in a class's network (3 suffice) */
#define NPN_CLASSES 222

struct ncls {
	uint16_t rep;		/* the class's least truth table */
	int8_t n;		/* tables */
	int8_t out;		/* the output: 0..3 an input, 4 + i table i,
				 * -1 the constant 0 */
	uint8_t a[NPN_MAXL][3];	/* each table's arguments, as out */
	uint8_t tt[NPN_MAXL];	/* bit (x0 | x1 << 1 | x2 << 2) */
	uint8_t inv;		/* an input's negation, with no table to take
				 * it: the class of the literals */
	int exact;		/* minimum proved by the search */
};

static struct {
	int ready;
	uint8_t perm[24][4];
	uint8_t map[384][16];	/* transform t, minterm of g -> minterm of f */
	int16_t ix[65536];	/* class + 1, by representative */
	struct ncls c[NPN_CLASSES];
	int nc;
	struct basis lut3;
	/* the shortest expression of each 3-input table over x, y, z */
	uint8_t xop[256], xl[256], xr[256], xsz[256];
} g_npn;

/*
 * Transform t = 16 p + n (permutation p, input negations n) takes f to
 * g(y) = f(z) with z_perm[i] = y_i xor n_i: bit m of g's table is bit
 * map[t][m] of f's.
 */
static void
npn_init(void)
{
	int p = 0, a, b, c, d, t, m, i;

	if (g_npn.ready)
		return;
	for (a = 0; a < 4; a++)
		for (b = 0; b < 4; b++)
			for (c = 0; c < 4; c++)
				for (d = 0; d < 4; d++) {
					if (a == b || a == c || a == d || b == c ||
					    b == d || c == d)
						continue;
					g_npn.perm[p][0] = (uint8_t)a;
					g_npn.perm[p][1] = (uint8_t)b;
					g_npn.perm[p][2] = (uint8_t)c;
					g_npn.perm[p][3] = (uint8_t)d;
					p++;
				}
	for (t = 0; t < 384; t++)
		for (m = 0; m < 16; m++) {
			int z = 0;
			for (i = 0; i < 4; i++)
				if (((m >> i) & 1) ^ ((t % 16 >> i) & 1))
					z |= 1 << g_npn.perm[t / 16][i];
			g_npn.map[t][m] = (uint8_t)z;
		}
	g_npn.lut3.name = sym_cstr("lut3");
	g_npn.lut3.ng = 1;
	g_npn.lut3.g[0].name = sym_cstr("lut");
	g_npn.lut3.g[0].arity = 3;
	g_npn.lut3.g[0].lut = 1;
	g_npn.lut3.g[0].cost = 1;
	g_npn.lut3.g[0].body = mk_const_i(TY(K_UINT, 1), 0);
	basis_finish_gate(&g_npn.lut3, &g_npn.lut3.g[0]);
	/* expressions of 3 inputs: leaves x, y, z and the constants, then
	 * ~, &, |, xor to a fixpoint of the tree size */
	{
		static const uint8_t leaf[] = {0xaa, 0xcc, 0xf0, 0x00, 0xff};
		int changed = 1, k, l, r;
		memset(g_npn.xsz, 0xff, sizeof g_npn.xsz);
		for (k = 0; k < 5; k++) {
			g_npn.xsz[leaf[k]] = 0;
			g_npn.xop[leaf[k]] = (uint8_t)k;	/* which leaf */
		}
		while (changed) {
			changed = 0;
			for (l = 0; l < 256; l++) {
				if (g_npn.xsz[l] == 0xff)
					continue;
				k = (uint8_t)~l;
				if (g_npn.xsz[l] + 1 < g_npn.xsz[k]) {
					g_npn.xsz[k] = (uint8_t)(g_npn.xsz[l] + 1);
					g_npn.xop[k] = O_NOT;
					g_npn.xl[k] = (uint8_t)l;
					changed = 1;
				}
				for (r = 0; r < 256; r++) {
					int s, o, v[3] = {l & r, l | r, l ^ r};
					static const int ops[3] = {O_AND, O_OR, O_XOR};
					if (g_npn.xsz[r] == 0xff)
						continue;
					s = g_npn.xsz[l] + g_npn.xsz[r] + 1;
					for (o = 0; o < 3; o++)
						if (s < g_npn.xsz[v[o]]) {
							g_npn.xsz[v[o]] = (uint8_t)s;
							g_npn.xop[v[o]] = (uint8_t)ops[o];
							g_npn.xl[v[o]] = (uint8_t)l;
							g_npn.xr[v[o]] = (uint8_t)r;
							changed = 1;
						}
				}
			}
		}
	}
	g_npn.ready = 1;
}

static uint16_t
npn_apply(int t, uint16_t f)
{
	uint16_t g = 0;
	int m;

	for (m = 0; m < 16; m++)
		if ((f >> g_npn.map[t][m]) & 1)
			g |= (uint16_t)(1u << m);
	return g;
}

/* the class's least table, and the transform (t, o) that gives it */
static uint16_t
npn_canon(uint16_t f, int *bt, int *bo)
{
	uint32_t best = 0x10000;
	int t, o;

	for (t = 0; t < 384; t++) {
		uint16_t g = npn_apply(t, f);
		for (o = 0; o < 2; o++) {
			uint16_t v = o ? (uint16_t)~g : g;
			if (v < best) {
				best = v;
				*bt = t;
				*bo = o;
			}
		}
	}
	return (uint16_t)best;
}

/* an 8-bit table read with argument k negated */
static uint8_t
flip3(uint8_t tt, int k)
{
	uint8_t r = 0;
	int m;

	for (m = 0; m < 8; m++)
		if ((tt >> (m ^ (1 << k))) & 1)
			r |= (uint8_t)(1u << m);
	return r;
}

/* the values of a network on the 16 minterms */
static uint16_t
net_eval(const struct ncls *c, int inv)
{
	uint16_t v[4 + NPN_MAXL];
	int i, k, m;

	for (i = 0; i < 4; i++) {
		v[i] = 0;
		for (m = 0; m < 16; m++)
			if ((m >> i) & 1)
				v[i] |= (uint16_t)(1u << m);
	}
	for (i = 0; i < c->n; i++) {
		v[4 + i] = 0;
		for (m = 0; m < 16; m++) {
			int x = 0;
			for (k = 0; k < 3; k++)
				x |= ((v[c->a[i][k]] >> m) & 1) << k;
			if ((c->tt[i] >> x) & 1)
				v[4 + i] |= (uint16_t)(1u << m);
		}
	}
	return (uint16_t)((c->out < 0 ? 0 : v[c->out]) ^ (inv ? 0xffff : 0));
}

/* a class, its network found the first time */
static const struct ncls *
npn_class(uint16_t rep)
{
	struct ncls *c;
	struct gprog gp;
	int i, k, cost;

	if (g_npn.ix[rep])
		return &g_npn.c[g_npn.ix[rep] - 1];
	if (g_npn.nc == NPN_CLASSES)
		fail("npn: more than %d classes of 4-input functions (the "
		    "canonical form is wrong)", NPN_CLASSES);
	c = &g_npn.c[g_npn.nc];
	memset(c, 0, sizeof *c);
	c->rep = rep;
	c->out = -1;
	c->exact = 1;
	for (i = 0; i < 4 && rep; i++) {
		uint16_t pi = (uint16_t)(i == 0 ? 0xaaaa : i == 1 ? 0xcccc :
		    i == 2 ? 0xf0f0 : 0xff00);
		if (rep == pi || rep == (uint16_t)~pi) {
			c->out = (int8_t)i;
			c->inv = rep != pi;
			break;
		}
	}
	if (rep != 0 && c->out < 0) {
		cost = gates_synth(&g_npn.lut3, 4, rep, 0xffff, 3, 1ULL << 24,
		    &gp);
		if (cost >= 0 && gp.nstep <= NPN_MAXL) {
			c->n = (int8_t)gp.nstep;
			for (i = 0; i < gp.nstep; i++) {
				for (k = 0; k < 3; k++)
					c->a[i][k] = gp.a[i][k];
				c->tt[i] = gp.a[i][3];
			}
			c->out = (int8_t)gp.out;
		} else {
			/* x3 ? f1 : f0, each a table of x0..x2 */
			uint8_t t0 = 0, t1 = 0;
			int m;
			for (m = 0; m < 8; m++) {
				t0 |= (uint8_t)(((rep >> m) & 1) << m);
				t1 |= (uint8_t)(((rep >> (m + 8)) & 1) << m);
			}
			c->n = 3;
			for (i = 0; i < 2; i++) {
				c->a[i][0] = 0;
				c->a[i][1] = 1;
				c->a[i][2] = 2;
			}
			c->tt[0] = t0;
			c->tt[1] = t1;
			c->a[2][0] = 3;
			c->a[2][1] = 5;
			c->a[2][2] = 4;
			c->tt[2] = 0xe4;	/* x0 ? x1 : x2 */
			c->out = 6;
			c->exact = 0;
		}
	}
	if (net_eval(c, c->inv) != rep)
		fail("npn: the network for 0x%04x gives 0x%04x", rep,
		    net_eval(c, c->inv));
	g_npn.ix[rep] = (int16_t)(++g_npn.nc);
	return c;
}

/*
 * f's minimum network of 3-input tables over its inputs 0..3: the
 * class's, renamed and with the negations folded in.  *inv: the output
 * is the negated input (or the constant 1) when no table is left to take
 * the negation.  Returns the number of tables.
 */
static int
npn4_net(uint16_t f, struct ncls *out, int *inv, int *exact)
{
	const struct ncls *c;
	int t, o, i, k;
	const uint8_t *p;

	npn_init();
	c = npn_class(npn_canon(f, &t, &o));
	p = g_npn.perm[t / 16];
	*out = *c;
	*exact = c->exact;
	*inv = 0;
	/* g(y) = o xor f(z), z_p[i] = y_i xor n_i: so f(z) = o xor g(y)
	 * with y_i = z_p[i] xor n_i */
	for (i = 0; i < c->n; i++)
		for (k = 0; k < 3; k++) {
			int a = c->a[i][k];
			if (a >= 4)
				continue;
			out->a[i][k] = p[a];
			if ((t % 16 >> a) & 1)
				out->tt[i] = flip3(out->tt[i], k);
		}
	if (c->out < 0)
		*inv = o;
	else if (c->out < 4) {
		out->out = (int8_t)p[c->out];
		*inv = o ^ c->inv ^ ((t % 16 >> c->out) & 1);
	} else if (o) {
		int j = c->out - 4;
		out->tt[j] = (uint8_t)~out->tt[j];
		for (i = j + 1; i < c->n; i++)
			for (k = 0; k < 3; k++)
				if (out->a[i][k] == c->out)
					out->tt[i] = flip3(out->tt[i], k);
	}
	if (net_eval(out, *inv) != f)
		fail("npn: the network renamed for 0x%04x gives 0x%04x", f,
		    net_eval(out, *inv));
	return c->n;
}

/* ---- find's rewrites: bitwise cones of 4 leaves ---- */

static int
bitwise(nid n)
{
	int op = N.op[n];

	return TY_ISINT(N.ty[n]) && N.ty[n] != TY_BOOL && TY_SIZED(N.ty[n]) &&
	    (op == O_AND || op == O_OR || op == O_XOR || op == O_NOT);
}

/* the leaves of the cone under n (operands of n's type that are not
 * bitwise themselves); -1 past 4 */
static int
cone_leaves(nid n, nid *leaf, int nl, int *ops)
{
	nid kid[2];
	int k, i;

	(*ops)++;
	kid[0] = N.a[n];
	kid[1] = N.op[n] == O_NOT ? 0 : N.b[n];
	for (k = 0; k < 2; k++) {
		nid x = kid[k];
		if (!x)
			continue;
		if (bitwise(x) && N.ty[x] == N.ty[n]) {
			nl = cone_leaves(x, leaf, nl, ops);
			if (nl < 0)
				return -1;
			continue;
		}
		for (i = 0; i < nl && leaf[i] != x; i++)
			;
		if (i == nl) {
			if (nl == 4)
				return -1;
			leaf[nl++] = x;
		}
	}
	return nl;
}

static uint16_t
cone_tt(nid n, const nid *leaf)
{
	static const uint16_t proj[4] = {0xaaaa, 0xcccc, 0xf0f0, 0xff00};
	uint16_t a, b = 0;
	int i;

	for (i = 0; i < 4; i++)
		if (leaf[i] == n)
			return proj[i];
	a = cone_tt(N.a[n], leaf);
	if (N.op[n] != O_NOT)
		b = cone_tt(N.b[n], leaf);
	switch (N.op[n]) {
	case O_AND: return a & b;
	case O_OR: return a | b;
	case O_XOR: return a ^ b;
	}
	return (uint16_t)~a;
}

static nid
x3_expr(int tt, const nid *arg, ty_t t)
{
	int op = g_npn.xop[tt];

	if (g_npn.xsz[tt] == 0)
		return op < 3 ? arg[op] : mk_const_i(t, op == 3 ? 0 : -1);
	if (op == O_NOT)
		return mk_un(O_NOT, x3_expr(g_npn.xl[tt], arg, t));
	return mk_bin(op, x3_expr(g_npn.xl[tt], arg, t), x3_expr(g_npn.xr[tt],
	    arg, t));
}

/*
 * A table as written: 0 its shortest expression; 1..3 a split on argument
 * v - 1, ((g1 xor g0) & x) xor g0; 4 the complement's, negated.  Two
 * tables built alike can share a subterm (a | c in both), which the
 * cover then computes once and makes a leaf of each, so a table stops
 * being one cone of 3 leaves; another way of writing one avoids that.
 */
#define X3_WAYS 5

static nid
x3_way(int tt, const nid *arg, ty_t t, int v)
{
	int k = v - 1, m;
	uint8_t g0 = 0, g1 = 0;

	if (v == 4)
		return mk_un(O_NOT, x3_expr((uint8_t)~tt, arg, t));
	if (v == 0)
		return x3_expr(tt, arg, t);
	for (m = 0; m < 8; m++) {
		if ((tt >> (m & ~(1 << k))) & 1)
			g0 |= (uint8_t)(1u << m);
		if ((tt >> (m | (1 << k))) & 1)
			g1 |= (uint8_t)(1u << m);
	}
	if (g0 == g1)
		return x3_expr(tt, arg, t);
	return mk_bin(O_XOR, mk_bin(O_AND, x3_expr(g1 ^ g0, arg, t), arg[k]),
	    x3_expr(g0, arg, t));
}

static nid
net_expr(const struct ncls *net, int inv, const nid *leaf, ty_t t,
    const int *way)
{
	nid arg[4 + NPN_MAXL], to;
	int i, k;

	for (i = 0; i < 4; i++)
		arg[i] = leaf[i];
	for (i = 0; i < net->n; i++) {
		nid a3[3];
		for (k = 0; k < 3; k++)
			a3[k] = arg[net->a[i][k]];
		arg[4 + i] = x3_way(net->tt[i], a3, t, way[i]);
	}
	to = net->out < 0 ? mk_const_i(t, 0) : arg[net->out];
	return inv ? mk_un(O_NOT, to) : to;
}

void
npn_cands(struct ivc *c, nid root, struct irw **out, int *cnt, int *cap)
{
	uint32_t lim = (uint32_t)root + 1, sp = 0;
	uint8_t *mark, *inner;
	nid *st, n;

	(void)c;
	if (fuse_lut_k(model_active()) < 3)
		return;		/* no table instruction: nothing to fit */
	npn_init();
	mark = xcalloc(lim, 1);
	inner = xcalloc(lim, 1);
	st = xmalloc(((size_t)lim * 3 + N.nlst + 4) * sizeof *st);
	/* a cone's root is a bitwise node no bitwise node of its type reads */
	for (n = 1; n < lim; n++)
		if (bitwise(n)) {
			if (bitwise(N.a[n]) && N.ty[N.a[n]] == N.ty[n])
				inner[N.a[n]] = 1;
			if (N.op[n] != O_NOT && bitwise(N.b[n]) &&
			    N.ty[N.b[n]] == N.ty[n])
				inner[N.b[n]] = 1;
		}
	st[sp++] = root;
	while (sp) {
		nid leaf[4], to;
		int nl, ops = 0, nt, inv, exact, i;
		struct ncls net;
		uint16_t tt;
		char what[200], cond[200];
		struct sbuf pf = {0};
		struct irw *rw;

		n = st[--sp];
		if (n == 0 || n >= lim || mark[n])
			continue;
		mark[n] = 1;
		if (N.op[n] == O_CONST || N.op[n] == O_VAR || N.op[n] == O_UNDEF ||
		    N.op[n] == O_TABLE || N.op[n] == O_LOOP)
			continue;
		if (N.a[n])
			st[sp++] = N.a[n];
		if (N.b[n])
			st[sp++] = N.b[n];
		if (N.c[n])
			st[sp++] = N.c[n];
		if (!bitwise(n) || inner[n])
			continue;
		nl = cone_leaves(n, leaf, 0, &ops);
		if (nl != 4 || ops < 3)
			continue;
		tt = cone_tt(n, leaf);
		nt = npn4_net(tt, &net, &inv, &exact);
		if (nt >= ops)
			continue;	/* no fewer tables than operations */
		{
			/* the ways of writing the tables, until the cover prices
			 * each as one table (the price of a majority of 3) */
			int way[NPN_MAXL] = {0}, k, combos = 1;
			const struct model *md = model_active();
			float one = fuse_expr_cost(md, x3_expr(0xe8, leaf, N.ty[n])),
			    bc = 0, cc;
			for (i = 0; i < net.n; i++)
				combos *= X3_WAYS;
			to = 0;
			for (k = 0; k < combos; k++) {
				int r = k;
				nid e;
				for (i = 0; i < net.n; i++) {
					way[i] = r % X3_WAYS;
					r /= X3_WAYS;
				}
				e = net_expr(&net, inv, leaf, N.ty[n], way);
				cc = fuse_expr_cost(md, e);
				if (!to || cc < bc - 1e-6f) {
					to = e;
					bc = cc;
				}
				if (bc <= one * (float)(nt ? nt : 1) + 1e-6f)
					break;
			}
		}
		snprintf(what, sizeof what, "the bitwise function of %d values "
		    "as %d table%s of 3 inputs (its NPN class's %s network)", nl,
		    nt, nt == 1 ? "" : "s", exact ? "minimum" : "smallest found");
		snprintf(cond, sizeof cond, "every bit: and, or, xor and not act "
		    "on each bit alone");
		sb_printf(&pf, "bit for bit the value is a function of the bits "
		    "of a, b, c, d = ");
		for (i = 0; i < 4; i++)
			sb_printf(&pf, "%s%s", i ? ", " : "", shared_str(leaf[i], 40));
		sb_printf(&pf, ", truth table 0x%04x (bit a | b << 1 | c << 2 | "
		    "d << 3)", tt);
		sb_printf(&pf, "\n%s, by gates' exact search over tables of 3 "
		    "inputs on its class's least table; the tables here are that "
		    "network with the inputs renamed and the negations folded in, "
		    "and read back on all 16 rows", exact ? (nt == 1 ?
		    "one table is the minimum" : nt == 2 ? "no single table gives "
		    "it, two do" : "no two tables give it, three do") :
		    "the search stopped: a split on d, an upper bound");
		rw = irw_add(out, cnt, cap, root, n, to, what, cond, sb_str(&pf), 1);
		if (rw) {
			rw->derived = 1;
			rw->quiet = 1;
		}
		free(pf.p);
	}
	free(mark);
	free(inner);
	free(st);
}

/* `gates npn`: every class, its size, its minimum and how many functions */
void
cmd_npn(void)
{
	int f, t, o, i, hist[NPN_MAXL + 1] = {0}, *members;

	npn_init();
	members = xcalloc(65536, sizeof *members);
	for (f = 0; f < 65536; f++)
		members[npn_canon((uint16_t)f, &t, &o)]++;
	for (f = 0; f < 65536; f++)
		if (members[f])
			hist[npn_class((uint16_t)f)->n]++;
	printf("npn: the 65536 functions of 4 inputs fall in %d classes; "
	    "minimum networks of 3-input tables (lut3, gates' exact search):\n",
	    g_npn.nc);
	for (i = 0; i <= NPN_MAXL; i++)
		if (hist[i])
			printf("  %d table%s: %d classes\n", i, i == 1 ? "" : "s",
			    hist[i]);
	rep_begin("npn");
	rep_int("classes", g_npn.nc);
	for (f = 0, i = 0; f < 65536; f++)
		if (members[f]) {
			const struct ncls *c = npn_class((uint16_t)f);
			i += c->exact;
		}
	rep_int("exact", i);
	printf("  each proved minimal: %d of %d\n", i, g_npn.nc);
	free(members);
}
