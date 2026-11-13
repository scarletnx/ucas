/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Constant table reads in SPIR-V as nearest reads through the texture
 * unit (#103), included by spirv.c.  texalu.c finds them in ucas functions
 * and at the GLSL function level and proposes them; here simplify finds
 * them in the module: an OpLoad through an OpAccessChain into a function
 * or private array whose one store (or a private one's initializer) is a
 * constant composite, its index
 * an address mode of one of its subterms i (REPEAT i mod n, MIRROR, CLAMP
 * min(max(i, 0), n - 1), MIRROR_CLAMP), proved by SAT wherever the read
 * is defined, and |i| proved within the reach of an f32 coordinate.  With
 * `texture set S binding B` (the descriptor the host gives the table's
 * texture) the read is written: a sampled image variable at that binding,
 * its types, and OpImageSampleExplicitLod at (f32(i) + 0.5) * f32(1/n),
 * lod 0; without it, or for a second table, it is proposed.  Priced by
 * the model's tex_nearest; a model that does not list it, or does not
 * state subtexel, takes nothing.
 */

static int texb_set = -1, texb_binding = -1;

void
spirv_tex_binding(int set, int binding)
{
	texb_set = set;
	texb_binding = binding;
}

enum { TM_REPEAT, TM_MIRROR, TM_CLAMP, TM_MCLAMP };

static const char *const tm_name[] = {"REPEAT", "MIRRORED_REPEAT",
    "CLAMP_TO_EDGE", "MIRROR_CLAMP_TO_EDGE"};

struct texrd {
	int k, ac, comp, idx, at, mode, n, elt, written;
	double before, after;
	int64_t lo, hi;		/* the table's values */
	int64_t lim;
	char from[160], src[160];
};

static struct sbuf tex_decl, tex_annot;	/* declarations, decorations */
static int tex_cap;			/* OpCapability Sampled1D to add */
static int tex_nkilled;			/* instructions a written read left dead */

/*
 * A load of one entry of a constant table: k an OpLoad through an access
 * chain with one index into a Function or Private array variable whose
 * one store, or a Private one's initializer, is a constant composite
 * (*comp) of scalar constants
 */
static int
tex_table(int k, int *ac, int *comp, int *ix, int *n)
{
	const struct sins *I = &V.in[k], *A;
	int ptr, var, q, st = -1, ct, j;

	if (strcmp(I->op, "OpLoad") || I->nop < 2 || I->opnd[1][0] != '%')
		return 0;
	ptr = idx(I->opnd[1]);
	if (SP.def[ptr] < 0)
		return 0;
	A = &V.in[SP.def[ptr]];
	if ((strcmp(A->op, "OpAccessChain") && strcmp(A->op,
	    "OpInBoundsAccessChain")) || A->nop != 3)
		return 0;
	var = idx(A->opnd[1]);
	for (q = 0; q < V.nin; q++) {
		const struct sins *S = &V.in[q];
		if (!strcmp(S->op, "OpStore") && S->nop >= 2 && idx(S->opnd[0]) ==
		    var) {
			if (st >= 0)
				return 0;	/* stored twice */
			st = q;
		}
		/* a pointer into it that something stores through */
		if (!strcmp(S->op, "OpStore") && S->nop >= 2 && SP.def[idx(
		    S->opnd[0])] >= 0 && !strncmp(V.in[SP.def[idx(
		    S->opnd[0])]].op, "OpAccessChain", 13) && idx(V.in[SP.def[
		    idx(S->opnd[0])]].opnd[1]) == var)
			return 0;
	}
	if (st >= 0)
		ct = idx(V.in[st].opnd[1]);
	else {
		/* never stored: a Private variable's initializer */
		for (q = 0; q < V.nin; q++)
			if (V.in[q].res == var && !strcmp(V.in[q].op, "OpVariable"))
				break;
		if (q == V.nin || V.in[q].nop < 3 || strcmp(V.in[q].opnd[1],
		    "Private"))
			return 0;
		ct = idx(V.in[q].opnd[2]);
	}
	for (q = 0; q < V.nin; q++)
		if (V.in[q].res == ct && !strcmp(V.in[q].op,
		    "OpConstantComposite"))
			break;
	if (q == V.nin || V.in[q].nop < 3 || V.in[q].nop - 1 > 4096)
		return 0;
	for (j = 1; j < V.in[q].nop; j++) {
		int e = idx(V.in[q].opnd[j]);
		if (V.id[e].kind != K_CONST || V.id[e].spec || V.id[e].nc > 1)
			return 0;	/* an entry that is not a plain scalar */
	}
	*n = V.in[q].nop - 1;
	*comp = q;
	*ac = SP.def[ptr];
	*ix = res_of(idx(A->opnd[2]));
	return 1;
}

/* entry j of the table (its constant's bits) */
static uint64_t
tex_entry(int comp, int j)
{
	return V.id[idx(V.in[comp].opnd[1 + j])].c[0];
}

/* mode(x) on the circuit of x (width w), n entries */
static void
tex_mode_word(struct word *r, int mode, const struct word *x, int t, int64_t n)
{
	struct aig *g = SP.g;
	int sg = T(t)->sign, w = x->w;
	struct word kn, *a[2], m, s, k1, k0, nx;

	switch (mode) {
	case TM_REPEAT:
	case TM_MIRROR:
		w_const(&kn, w, (uint64_t)(mode == TM_MIRROR ? 2 * n : n));
		a[0] = (struct word *)x;
		a[1] = &kn;
		/* OpSMod takes the divisor's sign: floored for n > 0 */
		blast_op(&m, sg ? "OpSMod" : "OpUMod", NULL, t, a, 2, t);
		if (mode == TM_REPEAT) {
			*r = m;
			return;
		}
		w_const(&k1, w, (uint64_t)(2 * n - 1));
		w_sub(g, &s, &k1, &m);
		w_mux(g, r, w_ult(g, &m, &s), &m, &s);
		return;
	case TM_CLAMP:
		w_const(&k0, w, 0);
		w_const(&k1, w, (uint64_t)(n - 1));
		if (sg) {
			w_mux(g, &m, w_slt(g, x, &k0), &k0, x);
			w_mux(g, r, w_slt(g, &k1, &m), &k1, &m);
		} else
			w_mux(g, r, w_ult(g, &k1, x), &k1, x);
		return;
	default:	/* min(max(x, -1 - x), n - 1), signed */
		w_const(&k1, w, (uint64_t)-1 & wmask(w));
		w_sub(g, &nx, &k1, x);
		w_mux(g, &m, w_slt(g, x, &nx), &nx, x);
		w_const(&k1, w, (uint64_t)(n - 1));
		w_mux(g, r, w_slt(g, &k1, &m), &k1, &m);
		return;
	}
}

/* 1: idx = mode(x) wherever 0 <= idx < n (and idx is defined) */
static int
tex_mode_proved(int ix, int x, int mode, int64_t n)
{
	struct word *wi, *wx, mw, kn;
	lit goal;
	int t = V.id[x].type;

	blast_reset();
	wi = blast(ix, 16);
	wx = blast(x, 16);
	tex_mode_word(&mw, mode, wx, t, n);
	if (mw.w != wi->w)
		return 0;
	w_const(&kn, wi->w, (uint64_t)n);
	goal = aig_and(SP.g, aig_and(SP.g, SP.dl[ix], assumptions()),
	    aig_and(SP.g, w_ult(SP.g, wi, &kn), w_ne(SP.g, &mw, wi)));
	return aig_sat(SP.g, goal, 20000) == 0;
}

/* 1: |x| <= lim for every input the assumptions allow */
static int
tex_range_proved(int x, int64_t lim)
{
	struct word *wx, kl, s, kw;
	lit goal;
	int t = V.id[x].type;

	blast_reset();
	wx = blast(x, 16);
	if (T(t)->sign) {
		w_const(&kl, wx->w, (uint64_t)lim);
		w_add(SP.g, &s, wx, &kl);
		w_const(&kw, wx->w, (uint64_t)(2 * lim + 1));
	} else {
		s = *wx;
		w_const(&kw, wx->w, (uint64_t)(lim + 1));
	}
	goal = aig_and(SP.g, aig_and(SP.g, SP.dl[x], assumptions()),
	    L_NOT(w_ult(SP.g, &s, &kw)));
	return aig_sat(SP.g, goal, 20000) == 0;
}

/* the pure instructions under id, down to x (not x), into out[] */
static int
tex_cone(int id, int x, int *out, int n, int max, int depth)
{
	int d = id >= 0 ? SP.def[id] : -1, i;
	uint64_t v;

	if (id == x || depth > 16 || d < 0 || const_of(id, &v) || !pure_ins(d) ||
	    n >= max)
		return n;
	for (i = 0; i < n; i++)
		if (out[i] == id)
			return n;
	out[n++] = id;
	for (i = first_arg(&V.in[d]); i < V.in[d].nop; i++)
		n = tex_cone(res_of(idx(V.in[d].opnd[i])), x, out, n, max,
		    depth + 1);
	return n;
}

/*
 * What goes dead with the index (its access chain's only use is the read):
 * the instructions of its cone down to x whose every use is by another
 * that goes dead, into dead[] (a cone is a DAG: a value used twice inside
 * it still goes); their cost
 */
static double
tex_dead(int ix, int x, int *dead, int *nd)
{
	int cone[64], nc = tex_cone(ix, x, cone, 0, 64, 0), in[64] = {0}, i, j,
	    q, ch = 1;
	double c = 0;

	*nd = 0;
	if (nc == 0 || SP.uses[ix] != 1)
		return 0;
	in[0] = 1;	/* the index: the access chain is its one use */
	while (ch) {
		ch = 0;
		for (i = 1; i < nc; i++) {
			int refs = 0;
			if (in[i])
				continue;
			for (j = 0; j < nc; j++) {
				const struct sins *D = &V.in[SP.def[cone[j]]];
				if (!in[j])
					continue;
				for (q = first_arg(D); q < D->nop; q++)
					refs += res_of(idx(D->opnd[q])) == cone[i];
			}
			if (refs == SP.uses[cone[i]]) {
				in[i] = 1;
				ch = 1;
			}
		}
	}
	for (i = 0; i < nc; i++)
		if (in[i]) {
			c += ins_cost_k(SP.def[cone[i]]);
			if (dead)
				dead[(*nd)++] = SP.def[cone[i]];
		}
	return c;
}

/* the index's subterms, through pure integer instructions */
static int
tex_subterms(int id, int *out, int n, int max, int depth)
{
	int d, i;
	uint64_t v;

	if (id < 0 || const_of(id, &v) || depth > 4 || n >= max)
		return n;
	for (i = 0; i < n; i++)
		if (out[i] == id)
			return n;
	if (V.id[id].type < 0 || T(V.id[id].type)->k != S_INT ||
	    T(V.id[id].type)->w > 32)
		return n;
	out[n++] = id;
	d = SP.def[id];
	if (d < 0 || !pure_ins(d))
		return n;
	for (i = first_arg(&V.in[d]); i < V.in[d].nop; i++)
		n = tex_subterms(res_of(idx(V.in[d].opnd[i])), out, n, max,
		    depth + 1);
	return n;
}

/* the largest |i| whose coordinate (i + 1/2)/n made in f32 lands within
 * 1/2 - 2^-bits of the texel's centre (texalu.c's index_limit) */
static int64_t
tex_limit(int bits)
{
	double eps = ldexp(1.0, -23) + ldexp(1.0, -47);

	return (int64_t)floor((0.5 - ldexp(1.0, -bits)) / eps - 0.5) - 1;
}

static const char *
tex_format(int elt, int64_t lo, int64_t hi)
{
	if (T(elt)->k == S_FLOAT)
		return "R32_SFLOAT";
	if (!T(elt)->sign)
		return hi <= 255 ? "R8_UINT" : hi <= 65535 ? "R16_UINT" : "R32_UINT";
	return lo >= -128 && hi <= 127 ? "R8_SINT" : lo >= -32768 &&
	    hi <= 32767 ? "R16_SINT" : "R32_SINT";
}

/* an instruction before k with any operands (ids as tokens, literals) */
static int
tex_ins(int k, const char *name, const char *op, int ty, const char *const *o,
    int no)
{
	struct sins N;
	int id = idx(name), nk, j;

	memset(&N, 0, sizeof N);
	snprintf(N.op, sizeof N.op, "%s", op);
	N.res = id;
	N.opnd[0] = strdup(V.id[ty].tok);
	N.nop = 1;
	for (j = 0; j < no; j++)
		N.opnd[N.nop++] = strdup(o[j]);
	N.text = xmalloc(200);
	snprintf(N.text, 200, "%s = %s", name, op);
	if (V.nin == V.cap) {
		V.cap *= 2;
		V.in = xrealloc(V.in, (size_t)V.cap * sizeof *V.in);
	}
	nk = V.nin++;
	V.in[nk] = N;
	SP.def[id] = nk;
	V.id[id].type = ty;
	V.id[id].kind = K_VALUE;
	SP.before[nk] = k;
	return id;
}

/* a type of the module: OpTypeFloat 32 (want 0), or a vector of 4 of e */
static int
tex_type(int want, int e)
{
	int i;

	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_TYPE && (want == 0 ? V.id[i].t.k == S_FLOAT &&
		    V.id[i].t.w == 32 : V.id[i].t.k == S_VEC && V.id[i].t.n == 4 &&
		    V.id[i].t.e == e))
			return i;
	return -1;
}

/*
 * The declarations a written read needs, once: the image, the sampled
 * image and its pointer, the variable at the binding, f32 and its
 * vector if the module has none, the constants; the decorations; the
 * capability; and the variable in the entry point's interface where
 * SPIR-V 1.4 lists every global it uses (a global that is not Input or
 * Output already listed says the module is one)
 */
static void
tex_declare(int elt, int *ft, int *v4)
{
	int k, ep = -1, v14 = 0, i;

	if (tex_decl.n)
		return;
	*ft = tex_type(0, -1);
	if (*ft < 0) {
		*ft = idx("%ucas_tf32");
		V.id[*ft].kind = K_TYPE;
		memset(&V.id[*ft].t, 0, sizeof V.id[*ft].t);
		V.id[*ft].t.k = S_FLOAT;
		V.id[*ft].t.w = 32;
		sb_puts(&tex_decl, "%ucas_tf32 = OpTypeFloat 32\n");
	}
	*v4 = tex_type(1, elt);
	if (*v4 < 0) {
		*v4 = idx("%ucas_tv4");
		V.id[*v4].kind = K_TYPE;
		memset(&V.id[*v4].t, 0, sizeof V.id[*v4].t);
		V.id[*v4].t.k = S_VEC;
		V.id[*v4].t.n = 4;
		V.id[*v4].t.e = elt;
		sb_printf(&tex_decl, "%%ucas_tv4 = OpTypeVector %s 4\n",
		    V.id[elt].tok);
	}
	sb_printf(&tex_decl, "%%ucas_timg = OpTypeImage %s 1D 0 0 0 1 Unknown\n"
	    "%%ucas_tsi = OpTypeSampledImage %%ucas_timg\n"
	    "%%ucas_tptr = OpTypePointer UniformConstant %%ucas_tsi\n"
	    "%%ucas_tex = OpVariable %%ucas_tptr UniformConstant\n",
	    V.id[elt].tok);
	idx("%ucas_tsi");
	idx("%ucas_tex");
	sb_printf(&tex_annot, "OpDecorate %%ucas_tex DescriptorSet %d\n"
	    "OpDecorate %%ucas_tex Binding %d\n", texb_set, texb_binding);
	tex_cap = 1;
	for (k = 0; k < SP.orig_nin; k++) {
		if (!strcmp(V.in[k].op, "OpCapability") && V.in[k].nop >= 1 &&
		    !strcmp(V.in[k].opnd[0], "Sampled1D"))
			tex_cap = 0;
		if (!strcmp(V.in[k].op, "OpEntryPoint") && ep < 0)
			ep = k;
	}
	for (i = 2; ep >= 0 && i < V.in[ep].nop; i++) {
		int v = idx(V.in[ep].opnd[i]);
		if (V.id[v].kind == K_VAR && V.id[v].sc != SC_INPUT &&
		    V.id[v].sc != SC_OUTPUT)
			v14 = 1;
	}
	if (v14 && ep >= 0 && V.in[ep].nop < MAXOP) {
		V.in[ep].opnd[V.in[ep].nop++] = strdup("%ucas_tex");
		SP.rw[ep] = 1;
	}
}

static void
tex_write(struct texrd *r, int ft, int v4)
{
	char nm[6][48], cf[128], half[48], zero[48], inv[48];
	const char *o[4];
	int k = r->k, sg = T(V.id[r->at].type)->sign;
	float invn = 1.0f / (float)r->n;
	struct sins *I;
	int j;

	if (V.nin + 8 >= SP.orig_nin + 1024 || V.nid + 32 >= SP.orig_nid + 1024)
		return;		/* no room left for the instructions */
	for (j = 0; j < 5; j++)
		snprintf(nm[j], sizeof nm[j], "%%ucas_tx%d%c", k, "sface"[j]);
	snprintf(half, sizeof half, "%%ucas_tfh_%s", V.id[ft].tok + 1);
	snprintf(zero, sizeof zero, "%%ucas_tfz_%s", V.id[ft].tok + 1);
	snprintf(inv, sizeof inv, "%%ucas_tfi%d_%s", r->n, V.id[ft].tok + 1);
	snprintf(cf, sizeof cf, "%s = OpConstant %s ", half, V.id[ft].tok);
	if (!strstr(tex_decl.p, cf))
		sb_printf(&tex_decl, "%s0.5\n%s = OpConstant %s 0\n", cf, zero,
		    V.id[ft].tok);
	snprintf(cf, sizeof cf, "%s = OpConstant ", inv);
	if (!strstr(tex_decl.p, cf))
		sb_printf(&tex_decl, "%s%s %.9g\n", cf, V.id[ft].tok, invn);
	idx(half);
	idx(zero);
	idx(inv);
	o[0] = "%ucas_tex";
	tex_ins(k, nm[0], "OpLoad", idx("%ucas_tsi"), o, 1);
	o[0] = V.id[r->at].tok;
	tex_ins(k, nm[1], sg ? "OpConvertSToF" : "OpConvertUToF", ft, o, 1);
	o[0] = nm[1];
	o[1] = half;
	tex_ins(k, nm[2], "OpFAdd", ft, o, 2);
	o[0] = nm[2];
	o[1] = inv;
	tex_ins(k, nm[3], "OpFMul", ft, o, 2);
	o[0] = nm[0];
	o[1] = nm[3];
	o[2] = "Lod";
	o[3] = zero;
	tex_ins(k, nm[4], "OpImageSampleExplicitLod", v4, o, 4);
	I = &V.in[k];
	for (j = 1; j < I->nop; j++)
		free(I->opnd[j]);
	snprintf(I->op, sizeof I->op, "OpCompositeExtract");
	I->opnd[1] = strdup(nm[4]);
	I->opnd[2] = strdup("0");
	I->nop = 3;
	SP.rw[k] = 1;
	SP.nval++;	/* spirv-val is asked once, after the report */
	if (SP.uses[V.in[r->ac].res] == 1) {
		/* the access chain and what goes dead under it, now: the
		 * passes after this one leave them alone */
		int dk[64], nd, q;
		SP.dead[r->ac] = 1;
		tex_dead(r->idx, r->at, dk, &nd);
		for (q = 0; q < nd; q++)
			SP.dead[dk[q]] = 1;
		tex_nkilled += 1 + nd;
	}
	r->written = 1;
}

/*
 * The reads, the cheapest form of each; written where the request gave a
 * binding (the reads of the table with the largest saving: one binding is
 * one texture), proposed otherwise.  The report in *rep, the count.
 */
static int
tex_pass(struct sbuf *rep, struct texrd *rd, int max)
{
	const struct model *m = model_active();
	int bits = (int)m->par[MP_SUBTEXEL], k, n = 0, avail, i, best = -1,
	    wcomp = -1, ft = -1, v4 = -1, prior = 0;
	int64_t lim;

	sb_clear(&tex_decl);
	sb_clear(&tex_annot);
	tex_cap = 0;
	tex_nkilled = 0;
	avail = model_lists(m, C_TEX_NEAREST) && bits > 0;
	if (!avail) {
		if (texb_set >= 0)
			sb_printf(rep, "  texture set %d binding %d: nothing written: "
			    "%s %s\n", texb_set, texb_binding,
			    sym_str(m->name), !model_lists(m, C_TEX_NEAREST) ?
			    "lists no tex_nearest (a nearest read's price)" :
			    "does not state subtexel (the bits a texel's "
			    "coordinate is resolved to)");
		return 0;
	}
	lim = tex_limit(bits);
	for (k = 0; k < SP.orig_nin && n < max; k++) {
		int ac, comp, ix, nt, sub[12], ns, s, mode, elt, nk;
		struct texrd c;
		if (SP.dead[k] || !tex_table(k, &ac, &comp, &ix, &nt))
			continue;
		elt = idx(V.in[k].opnd[0]);
		if (V.id[elt].kind != K_TYPE || (T(elt)->k != S_INT &&
		    T(elt)->k != S_FLOAT) || T(elt)->w != 32)
			continue;
		if ((double)clock() / CLOCKS_PER_SEC > SP.deadline)
			break;
		memset(&c, 0, sizeof c);
		c.at = -1;
		c.before = INFINITY;
		ns = tex_subterms(ix, sub, 0, 12, 0);
		for (s = 0; s < ns; s++) {
			double before, after;
			if (sub[s] == ix || !tex_range_proved(sub[s], lim))
				continue;	/* a plain read stays a load */
			for (mode = TM_REPEAT; mode <= TM_MCLAMP; mode++) {
				if (mode == TM_MCLAMP && !T(V.id[sub[s]].type)->sign)
					continue;
				before = m->c[C_LOAD] + tex_dead(ix, sub[s], NULL,
				    &nk);
				after = m->c[C_CONV] + m->c[C_FADD] + m->c[C_FMUL] +
				    m->c[C_TEX_NEAREST];
				if (!(after < before - 1e-9) || (c.at >= 0 &&
				    before - after <= c.before - c.after))
					continue;
				if (!tex_mode_proved(ix, sub[s], mode, nt))
					continue;
				c.at = sub[s];
				c.mode = mode;
				c.before = before;
				c.after = after;
			}
		}
		if (c.at < 0)
			continue;
		c.k = k;
		c.ac = ac;
		c.comp = comp;
		c.idx = ix;
		c.n = nt;
		c.elt = elt;
		c.lim = lim;
		c.lo = INT64_MAX;
		c.hi = INT64_MIN;
		for (i = 0; i < nt && T(elt)->k == S_INT; i++) {
			int64_t v = geti(elt, tex_entry(comp, i));
			if (v < c.lo)
				c.lo = v;
			if (v > c.hi)
				c.hi = v;
		}
		snprintf(c.from, sizeof c.from, "%s", V.in[k].text);
		src_line(ins_line(k), c.src, sizeof c.src);
		rd[n] = c;
		if (best < 0 || rd[n].before - rd[n].after > rd[best].before -
		    rd[best].after)
			best = n;
		n++;
	}
	if (n == 0) {
		if (texb_set >= 0)
			sb_printf(rep, "  texture set %d binding %d: nothing written: "
			    "no constant table read whose index is an address mode "
			    "of a value within f32's reach was cheaper through the "
			    "texture unit under %s\n", texb_set, texb_binding,
			    sym_str(m->name));
		return 0;
	}
	/* a module simplify wrote a texture into before: not a second one */
	for (i = 0; i < V.nid; i++)
		if (V.id[i].kind == K_VAR && !strcmp(V.id[i].tok, "%ucas_tex"))
			prior = 1;
	if (texb_set >= 0 && !prior) {
		wcomp = V.in[rd[best].comp].res;
		tex_declare(rd[best].elt, &ft, &v4);
		for (i = 0; i < n; i++)
			if (V.in[rd[i].comp].res == wcomp && rd[i].elt == rd[best].elt)
				tex_write(&rd[i], ft, v4);
	}
	sb_printf(rep, "  table reads through the texture unit, under %s "
	    "(nearest reads: a conversion, an add, a multiply and tex_nearest "
	    "in place of the load and the index's address mode):\n",
	    sym_str(m->name));
	for (i = 0; i < n; i++) {
		const struct texrd *r = &rd[i];
		int j, sg = T(V.id[r->at].type)->sign;
		sb_printf(rep, "    %s\n      ->  a read of texel %s through the "
		    "unit, %s   (cost %g -> %g)%s\n", r->from, V.id[r->at].tok,
		    tm_name[r->mode], r->before, r->after, r->written ? "" :
		    "   (proposed)");
		sb_printf(rep, "      holds where the read is defined (0 <= %s < "
		    "%d): there %s = %s, proved by SAT\n", V.id[r->idx].tok, r->n,
		    V.id[r->idx].tok, r->mode == TM_REPEAT ? "i mod n" :
		    r->mode == TM_MIRROR ? "min(i mod 2n, 2n - 1 - i mod 2n)" :
		    r->mode == TM_CLAMP ? "min(max(i, 0), n - 1)" :
		    "min(max(i, -1 - i), n - 1)");
		sb_printf(rep, "      with i = %s, |i| <= %lld (proved by SAT): the "
		    "coordinate (f32(i) + 0.5) * f32(1/n) is within (|i| + 1/2) "
		    "2^-23 of the texel's centre, and the unit snaps by less than "
		    "2^-%d, below 1/2 in all: the texel is i's\n",
		    V.id[r->at].tok, (long long)r->lim, bits);
		sb_printf(rep, "%s", r->src);
		sb_printf(rep, "      the pipeline: a texture %s, %d x 1, one level, "
		    "its texels the table's entries (", tex_format(r->elt, r->lo,
		    r->hi), r->n);
		for (j = 0; j < r->n && j < 16; j++)
			if (T(r->elt)->k == S_FLOAT)
				sb_printf(rep, "%s%.9g", j ? ", " : "", (double)getf(r->elt,
				    tex_entry(r->comp, j)));
			else
				sb_printf(rep, "%s%lld", j ? ", " : "", (long long)geti(
				    r->elt, tex_entry(r->comp, j)));
		sb_printf(rep, "%s); a sampler: NEAREST, %s, normalised "
		    "coordinates, lod 0", r->n > 16 ? ", ..." : "", tm_name[r->mode]);
		if (r->written)
			sb_printf(rep, "; written: the combined image sampler at set "
			    "%d binding %d, %s\n", texb_set, texb_binding, sg ?
			    "OpConvertSToF" : "OpConvertUToF");
		else if (texb_set >= 0 && prior)
			sb_printf(rep, "; not written: the module has the texture "
			    "simplify wrote before (%%ucas_tex), and one more would "
			    "need its own binding\n");
		else if (texb_set >= 0)
			sb_printf(rep, "; not written: one binding is one texture, and "
			    "set %d binding %d went to another table\n", texb_set,
			    texb_binding);
		else
			sb_printf(rep, "; not written: say `texture set S binding B` "
			    "with the descriptor the host binds it at\n");
	}
	return n;
}
