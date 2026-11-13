/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * spv_sg.inc (included by spirv.c): the OpGroupNonUniform instructions.
 *
 * `animate exec` runs them.  The lanes it runs are one subgroup
 * (SubgroupSize is their number, SubgroupLocalInvocationId the lane), and
 * an instruction reads its operand in the lanes the mask has active: a
 * lane a branch left out takes no part, as the specification says.  A
 * read of a lane that is not active or not in the subgroup (a shuffle's,
 * a broadcast's) is undefined there; it gives 0 here and is counted for
 * a note.
 *
 * Each is priced as the ucas operation it stands for (sg.c), the way find
 * prices that operation under the model: the ballot, the votes, the sum,
 * the inclusive scan and the shuffle are their classes; the exclusive
 * scan is the inclusive one less the lane's own value, a shuffle by xor,
 * up or down, a quad swap or a quad broadcast the shuffle and its index
 * (the quad's lanes in fours, bit 0 its column), a bit count of a ballot the
 * popcount and the mask of the lanes it keeps.  An instruction no
 * operation stands for (a product, a min, a clustered sum ...), or one
 * whose class the model does not list, runs at cost 0, and a note names
 * it.
 *
 * `simplify` writes them: a sum or a scan over the lanes of a value that
 * is 0 or 1 in every lane is the number of lanes where it is not 0, the
 * bit count of a ballot.  That holds for any subgroup size and any set
 * of active lanes (both sides count the active lanes only), so the
 * module needs no more than the ballot capability.
 */

static struct {
	double *cost;		/* per instruction: NAN until priced, -1 none */
	int *cls;		/* its class for the timeline, -1 */
	int ncost;
	long undef;		/* reads of a lane that is not there */
	char undef_ins[96];	/* the first one's instruction */
	char unpriced[6][160];	/* run at cost 0, and why */
	int nunpriced;
	int fsum;		/* a float sum or product ran in lane order */
	int used;		/* a subgroup instruction ran */
} SGX;

static void
sgx_reset(void)
{
	free(SGX.cost);
	free(SGX.cls);
	memset(&SGX, 0, sizeof SGX);
}

static int
is_sg_ins(const char *op)
{
	return !strncmp(op, "OpGroupNonUniform", 17);
}

/* the group operation of an arithmetic or bit-count instruction */
static const char *
sg_gop(const struct sins *I)
{
	const char *o = I->op + 17;

	if (!strcmp(o, "BallotBitCount") || !strcmp(o, "IAdd") ||
	    !strcmp(o, "FAdd") || !strcmp(o, "IMul") || !strcmp(o, "FMul") ||
	    !strcmp(o, "SMin") || !strcmp(o, "UMin") || !strcmp(o, "FMin") ||
	    !strcmp(o, "SMax") || !strcmp(o, "UMax") || !strcmp(o, "FMax") ||
	    !strncmp(o, "Bitwise", 7) || !strncmp(o, "Logical", 7))
		return I->nop > 2 ? I->opnd[2] : "";
	return NULL;
}

/*
 * The ucas operation instruction I stands for, its operands free leaves
 * (the value v, the predicate p, an index or a ballot word, the lane):
 * what the model prices.  0 when none does; *cls its main class.
 */
static nid
sg_expr_of(const struct sins *I, int W, int *cls)
{
	const char *o = I->op + 17, *gop = sg_gop(I);
	ty_t u = TY(K_UINT, 32), bt = TY(K_UINT, W > 32 ? 64 : 32);
	nid v = mk_var(0, sym_cstr("v"), u), p = mk_var(1, sym_cstr("p"),
	    TY_BOOL), i = mk_var(2, sym_cstr("i"), u), b = mk_var(3,
	    sym_cstr("b"), bt), l = mk_var(4, sym_cstr("lane"), u);
	nid w = mk_const_i(TY_LIT, W);

	*cls = -1;
	if (W < 2 || W > SG_MAXW)
		return 0;
#define O(n) (!strcmp(o, n))
	if (O("Ballot")) {
		*cls = C_SG_BALLOT;
		return mk_sg(O_SGBALLOT, p, w, 0);
	}
	if (O("Any") || O("All")) {
		*cls = C_SG_VOTE;
		return mk_sg(O("Any") ? O_SGANY : O_SGALL, p, w, 0);
	}
	if (O("AllEqual")) {	/* each lane against the first one's value */
		*cls = C_SG_VOTE;
		return mk_sg(O_SGALL, mk_bin(O_EQ, v, mk_sg(O_SGSHUF, v,
		    mk_const_i(u, 0), w)), w, 0);
	}
	if (O("Broadcast") || O("BroadcastFirst") || O("Shuffle") ||
	    O("ShuffleXor") || O("ShuffleUp") || O("ShuffleDown")) {
		nid at = O("BroadcastFirst") ? mk_const_i(u, 0) : O("ShuffleXor") ?
		    mk_bin(O_XOR, l, i) : O("ShuffleUp") ? mk_bin(O_SUB, l, i) :
		    O("ShuffleDown") ? mk_bin(O_ADD, l, i) : i;
		*cls = C_SG_SHUF;
		return mk_sg(O_SGSHUF, v, at, w);
	}
	if (O("QuadSwap") || O("QuadBroadcast")) {
		nid at;
		if (O("QuadSwap")) {
			/* Direction 0, 1, 2: the lane bits 1, 2, 3 flipped */
			int d = I->nop > 3 ? opid(I, 3) : -1;
			if (d < 0 || V.id[d].kind != K_CONST || V.id[d].c[0] > 2)
				return 0;
			at = mk_bin(O_XOR, l, mk_const_i(u, (int64_t)V.id[d].c[0] + 1));
		} else
			at = mk_bin(O_OR, mk_bin(O_AND, l, mk_const_i(u, ~3u)), i);
		*cls = C_SG_SHUF;
		return mk_sg(O_SGSHUF, v, at, w);
	}
	if (O("Elect")) {	/* the lowest lane the ballot has */
		*cls = C_SG_BALLOT;
		return mk_bin(O_EQ, l, mk_cast(u, mk_un(O_CTZ, mk_sg(O_SGBALLOT,
		    p, w, 0))));
	}
	if (O("BallotBitCount") && gop) {
		nid one = mk_const_i(bt, 1), lb = mk_cast(bt, l);
		*cls = C_BITCNT;
		if (!strcmp(gop, "InclusiveScan"))
			b = mk_bin(O_AND, b, mk_bin(O_SUB, mk_bin(O_SHL,
			    mk_const_i(bt, 2), lb), one));
		else if (!strcmp(gop, "ExclusiveScan"))
			b = mk_bin(O_AND, b, mk_bin(O_SUB, mk_bin(O_SHL, one, lb),
			    one));
		else if (strcmp(gop, "Reduce"))
			return 0;
		return mk_un(O_POPCNT, b);
	}
	if (O("BallotBitExtract") || O("InverseBallot")) {
		*cls = C_SHIFT;
		return mk_bin(O_NE, mk_bin(O_AND, mk_bin(O_SHR, b, mk_cast(bt,
		    O("InverseBallot") ? l : i)), mk_const_i(bt, 1)),
		    mk_const_i(bt, 0));
	}
	if (O("BallotFindLSB") || O("BallotFindMSB")) {
		*cls = C_BITCNT;
		return O("BallotFindLSB") ? mk_un(O_CTZ, b) : mk_bin(O_SUB,
		    mk_const_i(bt, W > 32 ? 63 : 31), mk_un(O_CLZ, b));
	}
	if (O("IAdd") && gop) {
		if (!strcmp(gop, "Reduce")) {
			*cls = C_SG_ADD;
			return mk_sg(O_SGADD, v, w, 0);
		}
		*cls = C_SG_SCAN;
		if (!strcmp(gop, "InclusiveScan"))
			return mk_sg(O_SGSCAN, v, l, w);
		if (!strcmp(gop, "ExclusiveScan"))
			return mk_bin(O_SUB, mk_sg(O_SGSCAN, v, l, w), v);
	}
#undef O
	return 0;
}

/*
 * Its price under the active model, one lane's view, for W lanes: NAN
 * when no operation stands for it, INFINITY when the model lacks a class
 * it needs.  *cls the main class.
 */
static double
sg_price(const struct sins *I, int W, int *cls)
{
	nid e;
	double c;
	jmp_buf saved;
	int nc = 1;

	memcpy(saved, fail_jmp, sizeof saved);
	note_mute++;
	if (setjmp(fail_jmp) == 0)
		e = sg_expr_of(I, W, cls);
	else
		e = 0;
	note_mute--;
	memcpy(fail_jmp, saved, sizeof saved);
	if (!e)
		return NAN;
	c = fuse_expr_cost(model_active(), e);
	if (I->nop >= 1 && I->opnd[0][0] == '%') {
		int t = idx(I->opnd[0]);
		if (V.id[t].kind == K_TYPE && T(t)->k == S_VEC &&
		    strcmp(I->op + 17, "Ballot") != 0)
			nc = T(t)->n;	/* one instruction a component */
	}
	return c * nc;
}

/* "op[ gop]|why": instruction, and why it ran at cost 0 */
static void
sg_unpriced(const struct sins *I, const char *why)
{
	char what[80];
	int i;

	snprintf(what, sizeof what, "%s%s%s", I->op, sg_gop(I) ? " " : "",
	    sg_gop(I) ? sg_gop(I) : "");
	for (i = 0; i < SGX.nunpriced; i++)
		if (!strncmp(SGX.unpriced[i], what, strlen(what)) &&
		    SGX.unpriced[i][strlen(what)] == '|')
			return;
	if (SGX.nunpriced < 6)
		snprintf(SGX.unpriced[SGX.nunpriced++], sizeof SGX.unpriced[0],
		    "%s|%s", what, why);
}

static uint64_t
sg_identity(const char *o, int st)
{
	int w = T(st)->w;

	if (!strcmp(o, "IMul"))
		return 1;
	if (!strcmp(o, "FMul"))
		return putf(st, 1);
	if (!strcmp(o, "SMin"))
		return wmask(w) >> 1;
	if (!strcmp(o, "UMin") || !strcmp(o, "BitwiseAnd"))
		return wmask(w);
	if (!strcmp(o, "SMax"))
		return 1ULL << (w - 1);
	if (!strcmp(o, "FMin"))
		return putf(st, INFINITY);
	if (!strcmp(o, "FMax"))
		return putf(st, -INFINITY);
	if (!strcmp(o, "LogicalAnd"))
		return 1;
	return 0;	/* sums, or, xor, umax */
}

static uint64_t
sg_combine(const char *o, int st, uint64_t a, uint64_t b)
{
	static const struct {
		const char *sg, *core, *ext;
	} t[] = {{"IAdd", "OpIAdd", NULL}, {"FAdd", "OpFAdd", NULL},
	    {"IMul", "OpIMul", NULL}, {"FMul", "OpFMul", NULL},
	    {"BitwiseAnd", "OpBitwiseAnd", NULL},
	    {"BitwiseOr", "OpBitwiseOr", NULL},
	    {"BitwiseXor", "OpBitwiseXor", NULL},
	    {"LogicalAnd", "OpLogicalAnd", NULL},
	    {"LogicalOr", "OpLogicalOr", NULL},
	    {"LogicalXor", "OpLogicalNotEqual", NULL},
	    {"SMin", NULL, "SMin"}, {"UMin", NULL, "UMin"},
	    {"FMin", NULL, "FMin"}, {"SMax", NULL, "SMax"},
	    {"UMax", NULL, "UMax"}, {"FMax", NULL, "FMax"}};
	size_t i;
	int ok = 1;

	for (i = 0; i < sizeof t / sizeof t[0]; i++)
		if (!strcmp(o, t[i].sg))
			return t[i].core ? alu(t[i].core, st, st, a, b, 0, &ok) :
			    glsl(t[i].ext, st, st, a, b, 0, &ok);
	return 0;
}

/* lane s's value of id, s a lane the instruction may read; 0 counted */
static int
sg_lane_ok(int s, uint32_t mask, const struct sins *I)
{
	if (s >= 0 && s < V.lanes && (mask >> s & 1))
		return 1;
	if (SGX.undef++ == 0)
		snprintf(SGX.undef_ins, sizeof SGX.undef_ins, "%s", I->text);
	return 0;
}

static int
exec_subgroup(const struct sins *I, uint32_t mask, int rt, int nc)
{
	const char *o = I->op + 17, *gop;
	int l, s, c, k, cls, kq = (int)(I - V.in), first, st, vid = -1;
	struct lval r[NL];
	double cost;

	if (!is_sg_ins(I->op))
		return 0;
	if (rt < 0 || I->nop < (strcmp(I->op + 17, "Elect") ? 3 : 2))
		die("unsupported instruction \"%s\"", I->text);
	{
		int sc = opid(I, 1);
		if (V.id[sc].kind != K_CONST || V.id[sc].c[0] != 3)
			die("unsupported instruction \"%s\": only Subgroup scope (3) "
			    "runs here, the lanes run being one subgroup", I->text);
	}
	SGX.used = 1;
	gop = sg_gop(I);
	st = scalar_of(rt);
	first = mask ? __builtin_ctz(mask) : 0;
	memset(r, 0, sizeof r);
#define O(n) (!strcmp(o, n))
	if (O("Elect")) {
		for (l = 0; l < V.lanes; l++)
			r[l].b[0] = l == first;
	} else if (O("Any") || O("All") || O("Ballot")) {
		uint64_t bits = 0;
		vid = opid(I, 2);
		for (l = 0; l < V.lanes; l++)
			if ((mask >> l & 1) && val(vid, l).b[0])
				bits |= 1ULL << l;
		for (l = 0; l < V.lanes; l++)
			r[l].b[0] = O("Ballot") ? bits & 0xffffffffu : O("Any") ?
			    bits != 0 : bits == (mask & wmask(V.lanes));
	} else if (O("AllEqual")) {
		int same = 1, at, fl;
		vid = opid(I, 2);
		at = type_of(vid);
		fl = T(scalar_of(at))->k == S_FLOAT;
		for (l = 0; l < V.lanes && same; l++)
			if (mask >> l & 1)
				for (c = 0; c < ncomp(at); c++)
					if (fl ? getf(scalar_of(at), val(vid, l).b[c]) !=
					    getf(scalar_of(at), val(vid, first).b[c]) :
					    val(vid, l).b[c] != val(vid, first).b[c])
						same = 0;
		for (l = 0; l < V.lanes; l++)
			r[l].b[0] = same;
	} else if (O("Broadcast") || O("BroadcastFirst") || O("Shuffle") ||
	    O("ShuffleXor") || O("ShuffleUp") || O("ShuffleDown")) {
		vid = opid(I, 2);
		for (l = 0; l < V.lanes; l++) {
			int64_t x = O("BroadcastFirst") ? first :
			    geti(type_of(opid(I, 3)), val(opid(I, 3), l).b[0]);
			if (!(mask >> l & 1))
				continue;
			s = O("ShuffleXor") ? (int)(l ^ x) : O("ShuffleUp") ?
			    (int)(l - x) : O("ShuffleDown") ? (int)(l + x) :
			    x < 0 || x >= NL ? -1 : (int)x;
			if (sg_lane_ok(s, mask, I))
				r[l] = val(vid, s);
		}
	} else if (O("QuadSwap") || O("QuadBroadcast")) {
		/* a quad: lanes in fours, bit 0 the column, bit 1 the row; a
		 * swap's Direction 0, 1, 2 flips bit 0, bit 1, both; a
		 * broadcast's Index past 3 is undefined */
		vid = opid(I, 2);
		for (l = 0; l < V.lanes; l++) {
			int64_t x = geti(type_of(opid(I, 3)), val(opid(I, 3), l).b[0]);
			if (!(mask >> l & 1))
				continue;
			if (O("QuadSwap") && (x < 0 || x > 2))
				die("unsupported instruction \"%s\": a quad swap's "
				    "direction is 0, 1 or 2", I->text);
			s = O("QuadSwap") ? (int)(l ^ (x + 1)) : x < 0 || x > 3 ? -1 :
			    (int)((l & ~3) | x);
			if (sg_lane_ok(s, mask, I))
				r[l] = val(vid, s);
		}
	} else if (O("InverseBallot") || O("BallotBitExtract") ||
	    O("BallotFindLSB") || O("BallotFindMSB") || O("BallotBitCount")) {
		vid = opid(I, O("BallotBitCount") ? 3 : 2);
		for (l = 0; l < V.lanes; l++) {
			uint64_t bw = val(vid, l).b[0] & wmask(V.lanes), keep;
			if (!(mask >> l & 1))
				continue;
			if (O("InverseBallot"))
				r[l].b[0] = bw >> l & 1;
			else if (O("BallotBitExtract")) {
				int64_t ix = geti(type_of(opid(I, 3)),
				    val(opid(I, 3), l).b[0]);
				if (ix >= 0 && ix < V.lanes)
					r[l].b[0] = bw >> ix & 1;
				else
					sg_lane_ok(-1, mask, I);
			} else if (O("BallotBitCount")) {
				keep = !strcmp(gop, "InclusiveScan") ? wmask(l + 1) :
				    !strcmp(gop, "ExclusiveScan") ? wmask(l) :
				    !strcmp(gop, "Reduce") ? wmask(V.lanes) : 0;
				if (!keep && strcmp(gop, "ExclusiveScan"))
					die("unsupported instruction \"%s\": the group "
					    "operations are Reduce, InclusiveScan and "
					    "ExclusiveScan", I->text);
				r[l].b[0] = (uint64_t)__builtin_popcountll(bw & keep);
			} else if (bw == 0)
				sg_lane_ok(-1, mask, I);	/* no bit set */
			else
				r[l].b[0] = O("BallotFindLSB") ?
				    (uint64_t)__builtin_ctzll(bw) :
				    (uint64_t)(63 - __builtin_clzll(bw));
		}
	} else if (gop) {
		/* sums, products, min, max, and, or, xor over the lanes */
		int at, ncm, cl = 0;
		vid = opid(I, 3);
		at = type_of(vid);
		ncm = ncomp(at);
		if (!strcmp(gop, "ClusteredReduce")) {
			if (I->nop < 5)
				die("unsupported instruction \"%s\": a clustered "
				    "reduce takes a cluster size", I->text);
			cl = (int)val(opid(I, 4), 0).b[0];
			if (cl < 1 || (cl & (cl - 1)))
				die("unsupported instruction \"%s\": the cluster size "
				    "is a power of two", I->text);
		} else if (strcmp(gop, "Reduce") && strcmp(gop, "InclusiveScan") &&
		    strcmp(gop, "ExclusiveScan"))
			die("unsupported instruction \"%s\": the group operations "
			    "are Reduce, InclusiveScan, ExclusiveScan and "
			    "ClusteredReduce", I->text);
		if (O("FAdd") || O("FMul"))
			SGX.fsum = 1;
		for (l = 0; l < V.lanes; l++) {
			int lo = 0, hi = V.lanes - 1, j;
			if (!(mask >> l & 1))
				continue;
			if (cl)
				lo = l / cl * cl, hi = lo + cl - 1;
			else if (!strcmp(gop, "InclusiveScan"))
				hi = l;
			else if (!strcmp(gop, "ExclusiveScan"))
				hi = l - 1;
			for (c = 0; c < ncm; c++) {
				uint64_t acc = sg_identity(o, scalar_of(at));
				for (j = lo; j <= hi && j < V.lanes; j++)
					if (mask >> j & 1)
						acc = sg_combine(o, scalar_of(at), acc,
						    val(vid, j).b[c]);
				r[l].b[c] = acc;
			}
		}
	} else
		die("unsupported instruction \"%s\"", I->text);
#undef O
	for (l = 0; l < V.lanes; l++)
		if (mask >> l & 1) {
			*LV(I->res, l) = r[l];
			if (T(st)->k == S_BOOL || T(st)->k == S_INT)
				for (c = 0; c < nc && c < MAXCOMP; c++)
					LV(I->res, l)->b[c] &= wmask(T(st)->w);
		}
	/* the price, once an instruction */
	if (SGX.ncost < V.nin) {
		SGX.cost = xrealloc(SGX.cost, (size_t)V.nin * sizeof *SGX.cost);
		SGX.cls = xrealloc(SGX.cls, (size_t)V.nin * sizeof *SGX.cls);
		for (c = SGX.ncost; c < V.nin; c++)
			SGX.cost[c] = NAN;
		SGX.ncost = V.nin;
	}
	if (isnan(SGX.cost[kq])) {
		cost = sg_price(I, V.lanes, &cls);
		SGX.cls[kq] = cls;
		if (isnan(cost))
			sg_unpriced(I, "(no operation of the model stands for it)");
		else if (!isfinite(cost)) {
			char why[96];
			snprintf(why, sizeof why, "(%s lists no %s)",
			    sym_str(model_active()->name), cls >= 0 ?
			    cclass_name[cls] : "class for it");
			sg_unpriced(I, why);
		}
		SGX.cost[kq] = isfinite(cost) ? cost : -1;
	}
	cost = SGX.cost[kq];
	cls = SGX.cls[kq];
	k = emit(I, I_OP, cost >= 0 ? cls : -1, cost >= 0 ? cost : 0, mask);
	emit_vals(k, I, mask);
	return 1;
}

/* exec's notes on the subgroup instructions it ran */
static void
sgx_notes(void)
{
	int i;

	if (!SGX.used)
		return;
	printf("  subgroup: the %d lanes run are one subgroup (SubgroupSize %d)"
	    "\n", V.lanes, V.lanes);
	for (i = 0; i < SGX.nunpriced; i++) {
		const char *s = SGX.unpriced[i], *sp = strchr(s, '|');
		printf("  note: %.*s ran at cost 0 %s\n", (int)(sp - s), s, sp + 1);
	}
	if (SGX.undef)
		printf("  note: %ld subgroup read%s of a lane that is not active "
		    "or not in the subgroup (first: %s): undefined, read as 0 "
		    "here\n", SGX.undef, SGX.undef == 1 ? "" : "s",
		    SGX.undef_ins);
	if (SGX.fsum)
		printf("  note: a float sum or product over the lanes ran in lane "
		    "order; an implementation may group it otherwise, and the "
		    "low bits then differ\n");
}

/* ---- simplify: a sum or a scan of bits over the lanes ---- */

static int sgcaps;		/* the module needs GroupNonUniformBallot */
static int sg_newty[3], sg_nnewty;	/* types the pass declared, in order */

static void
sg_simplify_reset(void)
{
	sgcaps = 0;
	sg_nnewty = 0;
}

/* the module's 32-bit uint (0), bool (1) or uvec4 (2) type, declared
 * before the first function when it has none */
static int
sg_type_id(int which)
{
	static const char *const nm[] = {"%ucas_uint", "%ucas_bool",
	    "%ucas_v4uint"};
	int i, u = which == 2 ? sg_type_id(0) : -1;

	for (i = 0; i < V.nid; i++) {
		const struct sid *d = &V.id[i];
		if (d->kind != K_TYPE)
			continue;
		if ((which == 0 && d->t.k == S_INT && d->t.w == 32 && !d->t.sign) ||
		    (which == 1 && d->t.k == S_BOOL) ||
		    (which == 2 && d->t.k == S_VEC && d->t.n == 4 && d->t.e == u))
			return i;
	}
	i = idx(nm[which]);
	V.id[i].kind = K_TYPE;
	memset(&V.id[i].t, 0, sizeof V.id[i].t);
	V.id[i].t.k = which == 0 ? S_INT : which == 1 ? S_BOOL : S_VEC;
	V.id[i].t.w = which == 0 ? 32 : which == 1 ? 1 : 0;
	V.id[i].t.n = which == 2 ? 4 : 0;
	V.id[i].t.e = u;
	sg_newty[sg_nnewty++] = i;
	return i;
}

static void
sg_types_out(struct sbuf *out)
{
	int i, t;

	for (i = 0; i < sg_nnewty; i++) {
		t = sg_newty[i];
		if (V.id[t].t.k == S_INT)
			sb_printf(out, "%s = OpTypeInt 32 0\n", V.id[t].tok);
		else if (V.id[t].t.k == S_BOOL)
			sb_printf(out, "%s = OpTypeBool\n", V.id[t].tok);
		else
			sb_printf(out, "%s = OpTypeVector %s 4\n", V.id[t].tok,
			    V.id[V.id[t].t.e].tok);
	}
}

static void
sg_caps_out(struct sbuf *out)
{
	int k;

	for (k = 0; k < SP.orig_nin; k++)
		if (!strcmp(V.in[k].op, "OpCapability") && V.in[k].nop >= 1 &&
		    !strcmp(V.in[k].opnd[0], "GroupNonUniformBallot"))
			return;
	sb_puts(out, "OpCapability GroupNonUniformBallot\n");
}

/* the price of instruction "op" (with group operation gop) for the pass */
static double
sg_price_op(const char *op, const char *gop, int W)
{
	struct sins I;
	int cls;

	memset(&I, 0, sizeof I);
	snprintf(I.op, sizeof I.op, "%s", op);
	I.opnd[0] = (char *)"";
	I.opnd[1] = (char *)"";
	I.opnd[2] = (char *)(gop ? gop : "");
	I.nop = 3;
	return sg_price(&I, W, &cls);
}

/* the live subgroup instructions' prices (those the model has) */
static double
sg_total(int all)
{
	const struct model *m = model_active();
	int W = (int)m->par[MP_WARP], k, cls;
	double t = 0, c;

	for (k = 0; k < (all ? V.nin : SP.orig_nin); k++)
		if (!SP.dead[k] && is_sg_ins(V.in[k].op) && V.in[k].res >= 0) {
			c = sg_price(&V.in[k], W, &cls);
			if (isfinite(c))
				t += c;
		}
	return t;
}

/* e (a ucas form of the value) is 0 or 1 in every lane: by intervals */
static int
sg_bits01(const struct func *f, nid e)
{
	struct plan p;
	struct ival *iv;
	uint32_t l;
	int ok;
	nid roots[1];

	if (is_const(e))
		return N.ki[e] == 0 || N.ki[e] == 1;
	roots[0] = e;
	plan_build(&p, roots, 1, 0);
	iv = xcalloc(p.n ? p.n : 1, sizeof *iv);
	plan_ival(&p, f, iv);
	l = plan_loc(&p, e);
	ok = l != NOTOK && !iv[l].top && !iv[l].wrap && iv[l].lo >= 0 &&
	    iv[l].hi <= 1;
	free(iv);
	plan_free(&p);
	return ok;
}

/*
 * Each OpGroupNonUniformIAdd of Subgroup scope (Reduce, InclusiveScan,
 * ExclusiveScan) on 32 bits whose value v is 0 or 1 in every lane: a
 * select of 1 and 0 by a condition c, or anything spv_node reads whose
 * range by intervals is [0, 1].  The sum of v over the lanes it covers
 * (the subgroup, those up to the lane, those before it) is the number of
 * them where v is not 0, which is OpGroupNonUniformBallotBitCount with
 * the same group operation of the ballot of v != 0 (of c, or !c): the
 * bit count reads the same lanes, and an inactive lane has no bit in a
 * ballot and no term in the sum.  Written before the sum when the model
 * prices the compare, the ballot and the count below the sum (and the
 * select that would go dead); otherwise said and left.
 */
static int
sg_pass(struct sbuf *rep, struct dotlog *log, int maxlog)
{
	const struct model *m = model_active();
	int W = (int)m->par[MP_WARP], k, n = 0;

	for (k = 0; k < SP.orig_nin && n < maxlog; k++) {
		const struct sins *I = &V.in[k];
		struct spvdot c;
		int root, rt, v, vk, sel = 0, neg = 0, cond = -1, nin0, i;
		int pred, ball, cnt, fin, bt, ut, vt;
		uint64_t sv, ka, kb;
		const char *gop, *o[3];
		double before, after, cmpc = 0, selc = 0;
		char to[160], sl[160], form[200];

		if (SP.dead[k] || strcmp(I->op, "OpGroupNonUniformIAdd") ||
		    I->res < 0 || I->nop != 4)
			continue;
		gop = I->opnd[2];
		if (strcmp(gop, "Reduce") && strcmp(gop, "InclusiveScan") &&
		    strcmp(gop, "ExclusiveScan"))
			continue;
		if (!const_of(res_of(idx(I->opnd[1])), &sv) || sv != 3)
			continue;
		root = I->res;
		rt = idx(I->opnd[0]);
		if (V.id[rt].kind != K_TYPE || T(rt)->k != S_INT || T(rt)->w != 32)
			continue;
		v = res_of(idx(I->opnd[3]));
		vk = SP.def[v];
		form[0] = 0;
		/* v = c ? 1 : 0 (or c ? 0 : 1): the ballot of c (of !c) */
		if (vk >= 0 && !SP.dead[vk] && !strcmp(V.in[vk].op, "OpSelect") &&
		    V.in[vk].nop == 4 && const_of(res_of(idx(V.in[vk].opnd[2])),
		    &ka) && const_of(res_of(idx(V.in[vk].opnd[3])), &kb) &&
		    ((ka == 1 && kb == 0) || (ka == 0 && kb == 1)) &&
		    int_width(V.id[res_of(idx(V.in[vk].opnd[1]))].type) == 1) {
			sel = 1;
			neg = ka == 0;
			cond = res_of(idx(V.in[vk].opnd[1]));
			snprintf(form, sizeof form, "%s is %s ? %s : %s",
			    V.id[v].tok, V.id[cond].tok, neg ? "0" : "1",
			    neg ? "1" : "0");
		} else {
			memset(&c, 0, sizeof c);
			c.memo = xcalloc((size_t)(V.nid + 1024), sizeof *c.memo);
			c.f.anon = 1;
			c.f.body = spv_node(&c, v, 0);
			free(c.memo);
			if (c.fail || !sg_bits01(&c.f, c.f.body))
				continue;
			snprintf(form, sizeof form, "%s is %s, in [0, 1] by intervals",
			    V.id[v].tok, shared_str(c.f.body, 120));
		}
		/* the prices: the sum (and a select only it reads) against the
		 * compare, the ballot and the count */
		src_line(ins_line(k), sl, sizeof sl);
		if (W < 2 || W > SG_MAXW) {
			sb_printf(rep, "    %s\n%s      %s; %s gives no warp size, so "
			    "a ballot's width is not known: left as it is\n", I->text,
			    sl, form, sym_str(m->name));
			continue;
		}
		before = sg_price(I, W, &i);
		if (sel && SP.uses[v] == 1)
			selc = ins_cost_k(vk);
		if (!sel)
			cmpc = ins_cost("OpINotEqual", NULL, rt);
		else if (neg)
			cmpc = ins_cost("OpLogicalNot", NULL, rt);
		after = sg_price_op("OpGroupNonUniformBallot", NULL, W) +
		    sg_price_op("OpGroupNonUniformBallotBitCount", gop, W) + cmpc;
		if (!isfinite(before) || !isfinite(after)) {
			int need = !isfinite(before) ? (!strcmp(gop, "Reduce") ?
			    C_SG_ADD : C_SG_SCAN) : C_SG_BALLOT;
			sb_printf(rep, "    %s\n%s      %s; %s lists no %s: left as it "
			    "is\n", I->text, sl, form, sym_str(m->name),
			    cclass_name[need]);
			continue;
		}
		if (!(after < before + selc - 1e-9)) {
			sb_printf(rep, "    %s\n%s      %s; as a ballot's bit count it "
			    "costs %g against %g under %s: left as it is\n", I->text, sl,
			    form, after, before + selc, sym_str(m->name));
			continue;
		}
		/* written before the sum */
		memset(&c, 0, sizeof c);
		c.k = k;
		nin0 = V.nin;
		ut = sg_type_id(0);
		bt = sg_type_id(1);
		vt = sg_type_id(2);
		if (sel && !neg)
			pred = cond;
		else if (sel) {
			o[0] = V.id[cond].tok;
			pred = dot_ins(&c, "OpLogicalNot", bt, o, 1);
		} else {
			int z = const_id(V.id[v].type, 0);
			o[0] = V.id[v].tok;
			o[1] = z >= 0 ? V.id[z].tok : "";
			pred = z >= 0 ? dot_ins(&c, "OpINotEqual", bt, o, 2) : -1;
		}
		o[0] = I->opnd[1];
		o[1] = pred >= 0 ? V.id[pred].tok : "";
		ball = pred >= 0 ? dot_ins(&c, "OpGroupNonUniformBallot", vt, o, 2) :
		    -1;
		o[1] = gop;
		o[2] = ball >= 0 ? V.id[ball].tok : "";
		cnt = ball >= 0 ? dot_ins(&c, "OpGroupNonUniformBallotBitCount", ut,
		    o, 3) : -1;
		fin = cnt;
		if (cnt >= 0 && rt != ut) {
			o[0] = V.id[cnt].tok;
			fin = dot_ins(&c, "OpBitcast", rt, o, 1);
		}
		if (c.fail || fin < 0) {
			int q, j;
			for (q = nin0; q < V.nin; q++) {
				for (j = 0; j < V.in[q].nop; j++)
					free(V.in[q].opnd[j]);
				free(V.in[q].text);
			}
			V.nin = nin0;
			continue;
		}
		SP.alias[root] = fin;
		SP.dead[k] = 1;
		sgcaps = 1;
		snprintf(to, sizeof to, "OpGroupNonUniformBallotBitCount %s of the "
		    "ballot of %s%s", gop, sel ? (neg ? "!" : "") : "",
		    sel ? V.id[cond].tok : V.id[v].tok);
		if (!sel)
			snprintf(to + strlen(to), sizeof to - strlen(to), " != 0");
		snprintf(log[n].from, sizeof log[n].from, "%s", I->text);
		snprintf(log[n].to, sizeof log[n].to, "%s", to);
		log[n].gain = before + selc - after;
		log[n].before = before + selc;
		log[n].after = after;
		log[n].line = ins_line(k);
		{
			struct sbuf d = {0};
			sb_printf(&d, "%s, so its sum over the %s is the number of "
			    "those lanes where it is not 0: the set bits of their "
			    "ballot, which OpGroupNonUniformBallotBitCount %s counts",
			    form, !strcmp(gop, "Reduce") ? "lanes" :
			    !strcmp(gop, "InclusiveScan") ? "lanes up to this one" :
			    "lanes before this one", gop);
			log[n].proof = d.p;
			log[n].desc = strdup("any subgroup size and any set of active "
			    "lanes: both sides count the active lanes only");
		}
		sb_printf(rep, "    %s\n      ->  %s   (cost %g -> %g)\n%s"
		    "      %s\n      holds for %s\n", I->text, to, log[n].before,
		    after, sl, log[n].proof, log[n].desc);
		n++;
	}
	return n;
}

/* a name or a decoration: says nothing of how an id is used */
static int
sg_debug_ins(const char *op)
{
	return !strncmp(op, "OpName", 6) || !strncmp(op, "OpMemberName", 12) ||
	    !strncmp(op, "OpDecorate", 10) || !strncmp(op, "OpMemberDecorate",
	    16);
}

/*
 * Subgroup forms that need the subgroup's size W fixed and every lane of
 * it active, which a module cannot say (in Vulkan both are the pipeline's
 * state), so they are proposed with the host-side change, the module left
 * as it is:
 *
 *   a scan (OpGroupNonUniformIAdd InclusiveScan or ExclusiveScan) as a
 *   ladder of ceil(log2 W) steps, each a shuffle up by 1, 2, 4 ... and an
 *   add where the lane is at least that far up (a compare and a select),
 *   the exclusive one less the lane's own value; the steps are per W, and
 *   a shuffle from a lane that is not active is undefined;
 *
 *   a constant table of at most W entries read at a dynamic index (glslang
 *   stores it into a Function variable before each read) held one entry a
 *   lane, lane l reading entry l once, each read a shuffle from the lane
 *   its index names.
 *
 * The scan is priced as the module's instructions are, under the model's
 * warp; a table's read as a `load` (simplify's totals leave Function
 * memory out); only a cheaper form is said.  The number of proposals.
 */
static int
sg_proposals(struct sbuf *rep)
{
	const struct model *m = model_active();
	int W = (int)m->par[MP_WARP], k, j, n = 0, steps = 0;
	char need[400], sl[160];
	double shufc;

	if (W < 2 || W > SG_MAXW)
		return 0;
	while ((1 << steps) < W)
		steps++;
	snprintf(need, sizeof need, "it needs the subgroup's size fixed at %d "
	    "and every lane of it active: requiredSubgroupSize %d "
	    "(VkPipelineShaderStageRequiredSubgroupSizeCreateInfo), "
	    "VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT on a "
	    "compute stage, and no branch the lanes disagree on around it", W,
	    W);
	/* scans as ladders */
	shufc = sg_price_op("OpGroupNonUniformShuffleUp", NULL, W);
	for (k = 0; k < SP.orig_nin; k++) {
		const struct sins *I = &V.in[k];
		const char *gop;
		uint64_t sv;
		int rt, cls, ex;
		double before, after, stepc;

		if (SP.dead[k] || strcmp(I->op, "OpGroupNonUniformIAdd") ||
		    I->res < 0 || I->nop != 4)
			continue;
		gop = I->opnd[2];
		ex = !strcmp(gop, "ExclusiveScan");
		if ((!ex && strcmp(gop, "InclusiveScan")) ||
		    !const_of(res_of(idx(I->opnd[1])), &sv) || sv != 3)
			continue;
		rt = idx(I->opnd[0]);
		if (V.id[rt].kind != K_TYPE || T(rt)->k != S_INT)
			continue;
		before = sg_price(I, W, &cls);
		stepc = shufc + ins_cost("OpUGreaterThanEqual", NULL, rt) +
		    ins_cost("OpSelect", NULL, rt) + ins_cost("OpIAdd", NULL, rt);
		after = steps * stepc + (ex ? ins_cost("OpISub", NULL, rt) : 0);
		if (!isfinite(before) || !isfinite(after) ||
		    !(after < before - 1e-9))
			continue;
		src_line(ins_line(k), sl, sizeof sl);
		sb_printf(rep, "    %s\n%s      as a ladder of %d steps, each a "
		    "shuffle up by 1, 2, 4 ... %d added where the lane is at least "
		    "that far up (a compare and a select)%s: cost %g -> %g\n      "
		    "%s\n", I->text, sl, steps, 1 << (steps - 1), ex ? ", less the "
		    "lane's own value" : "", before, after, need);
		n++;
	}
	/* constant tables as registers of the subgroup */
	for (j = 0; j < V.nid; j++) {
		int t = V.id[j].type, nread = 0, first = 1, line = -1;
		double before = 0, one = 0, after;
		char lines[80] = "";

		if (V.id[j].kind != K_CONST || V.id[j].spec || t < 0 ||
		    V.id[t].kind != K_TYPE || T(t)->k != S_ARRAY || T(t)->n < 1 ||
		    T(t)->n > W || T(T(t)->e)->k != S_INT)
			continue;
		for (k = 0; k < SP.orig_nin; k++) {
			const struct sins *S = &V.in[k];
			int var, q, ok = 1, reads = 0;
			double c;
			if (SP.dead[k] || strcmp(S->op, "OpStore") || S->nop < 2 ||
			    res_of(idx(S->opnd[1])) != j)
				continue;
			var = res_of(idx(S->opnd[0]));
			if (V.id[var].kind != K_VAR || (V.id[var].sc != SC_FUNCTION &&
			    V.id[var].sc != SC_PRIVATE))
				continue;
			/* every other use of var: a dynamic index read by loads */
			c = 0;
			for (q = 0; q < SP.orig_nin && ok; q++) {
				const struct sins *A = &V.in[q];
				int i, r;
				uint64_t kv;
				if (SP.dead[q] || q == k || sg_debug_ins(A->op))
					continue;
				for (i = 0; i < A->nop; i++)
					if (A->opnd[i][0] == '%' && res_of(idx(A->opnd[i])) ==
					    var)
						break;
				if (i == A->nop)
					continue;
				if (!strcmp(A->op, "OpStore") || strcmp(A->op,
				    "OpAccessChain") || i != 1 || A->nop != 3 ||
				    const_of(res_of(idx(A->opnd[2])), &kv)) {
					ok = 0;
					break;
				}
				for (r = 0; r < SP.orig_nin; r++) {
					const struct sins *L = &V.in[r];
					int u;
					if (SP.dead[r] || r == q || sg_debug_ins(L->op))
						continue;
					for (u = 0; u < L->nop; u++)
						if (L->opnd[u][0] == '%' && res_of(idx(L->opnd[u]))
						    == A->res)
							break;
					if (u == L->nop)
						continue;
					if (strcmp(L->op, "OpLoad") || u != 1) {
						ok = 0;
						break;
					}
					c += m->c[C_LOAD];
					reads++;
					if (line < 0 || ins_line(r) != line) {
						line = ins_line(r);
						if (line > 0 && strlen(lines) < sizeof lines - 12)
							snprintf(lines + strlen(lines), sizeof lines -
							    strlen(lines), "%s%d", lines[0] ? ", " : "",
							    line);
					}
				}
			}
			if (!ok || reads == 0)
				continue;
			before += c;
			nread += reads;
			if (first) {
				one = m->c[C_LOAD];	/* lane l's entry */
				first = 0;
			}
		}
		if (nread < 1)
			continue;
		after = one + nread * sg_price_op("OpGroupNonUniformShuffle", NULL,
		    W);
		if (!isfinite(after) || !(after < before - 1e-9))
			continue;
		sb_printf(rep, "    %s, a table of %d entries read %d time%s at a "
		    "dynamic index%s%s%s\n      as a register of the subgroup, lane "
		    "l holding entry l (one read of the table), each read a shuffle "
		    "from the lane its index names: cost %g -> %g\n      %s\n",
		    V.id[j].tok, T(t)->n, nread, nread == 1 ? "" : "s", lines[0] ?
		    " (line " : "", lines, lines[0] ? ")" : "", before, after, need);
		n++;
	}
	return n;
}

/*
 * A branch in a loop's body on a condition that differs from lane to
 * lane, one arm dearer than the other (#135): the lanes that take the
 * dear arm packed into the first lanes across the loop's iterations (a
 * persistent loop) pay each a ballot, the and with the lanes below and
 * its bit count (where it goes), and a shuffle there and back, and the
 * dear arm runs once for W packed items: p of it a lane, p the share of
 * the lanes that take it.  As written a warp runs the dear arm whenever
 * one of its lanes takes it, 1 - (1 - p)^W of the warps for independent
 * lanes, and pays the divergence where they split (or both arms and a
 * select, the cheaper).  Said where the compaction is cheaper for some p, with the shares of the lanes where
 * it is (an interval: a warp rarely split pays little as written, and
 * nearly every lane on the dear arm leaves nothing to pack): a
 * restructuring of the loop, the author's, so advice.
 */
/* a branch as written, a lane's share: the warp runs the dear arm when
 * one of its W independent lanes takes it (share p), the cheap one when
 * one does not, and pays the divergence where they split; or, if that
 * is cheaper, both arms and a select */
static double
sg_written(const struct model *m, int W, double p, double dear, double cheap)
{
	double all = pow(p, W), none = pow(1 - p, W);
	double br = m->c[C_BRANCH] + (1 - none) * dear + (1 - all) * cheap +
	    (1 - all - none) * m->c[C_DIVERGE];

	/* or both arms and a select, the way a small branch is flattened */
	return fmin(br, dear + cheap + m->c[C_SELECT]);
}

static int
sg_compaction(struct sbuf *rep)
{
	const struct model *m = model_active();
	int W = (int)m->par[MP_WARP], b, n = 0, i;
	double ballot, shuf, over;

	if (W < 2 || W > SG_MAXW)
		return 0;
	ballot = sg_price_op("OpGroupNonUniformBallot", NULL, W);
	shuf = sg_price_op("OpGroupNonUniformShuffle", NULL, W);
	if (!isfinite(ballot) || !isfinite(shuf))
		return 0;
	over = ballot + m->c[C_LOGIC] + m->c[C_BITCNT] + 2 * shuf;
	mem_uniformity();
	for (b = 0; b < MC.nblk; b++) {
		const struct sins *T2 = &V.in[MC.term[b]], *S;
		int l, inl = -1, mb, cb, side, arm[2], q;
		double cost[2], dear, cheap, p, plo, phi, pbest, gbest;
		char sl[160];
		if (MC.term[b] < 1 || strcmp(T2->op, "OpBranchConditional") ||
		    T2->nop < 3 || MC.ins_blk[MC.term[b] - 1] != b)
			continue;
		S = &V.in[MC.term[b] - 1];
		if (strcmp(S->op, "OpSelectionMerge") || S->nop < 1 ||
		    !mem_is_label(S->opnd[0]) || mem_uni_at(idx(T2->opnd[0]),
		    MC.term[b]))
			continue;
		for (l = 0; l < MC.nloop && inl < 0; l++)
			if (MC.lin[(size_t)l * (size_t)(MC.nblk + 1) + (size_t)b] &&
			    MC.lhdr[l] != b)
				inl = l;
		if (inl < 0)
			continue;
		mb = MC.lab_blk[idx(S->opnd[0])];
		/* the loop's continue block: a continue is no arm of work */
		{
			const struct sins *LM = &V.in[MC.term[MC.lhdr[inl]] - 1];
			cb = LM->nop >= 2 && mem_is_label(LM->opnd[1]) ?
			    MC.lab_blk[idx(LM->opnd[1])] : -1;
		}
		/* each arm: the blocks its edge reaches before the merge, the
		 * loop's header and its continue block */
		for (side = 0; side < 2; side++) {
			char *seen = xcalloc((size_t)MC.nblk + 1, 1);
			int *st = xmalloc(((size_t)MC.nblk + 1) * sizeof *st), sp = 0;
			cost[side] = 0;
			arm[side] = mem_is_label(T2->opnd[1 + side]) ?
			    MC.lab_blk[idx(T2->opnd[1 + side])] : -1;
			if (arm[side] >= 0 && arm[side] != mb && arm[side] != cb &&
			    arm[side] != MC.lhdr[inl]) {
				st[sp++] = arm[side];
				seen[arm[side]] = 1;
			}
			while (sp > 0) {
				int a = st[--sp], k;
				const struct sins *E = &V.in[MC.term[a]];
				for (k = MC.term[a]; k >= 0 && MC.ins_blk[k] == a; k--)
					if (V.in[k].res >= 0 && V.in[k].nop >= 1 &&
					    V.in[k].opnd[0][0] == '%' &&
					    V.id[idx(V.in[k].opnd[0])].kind == K_TYPE)
						cost[side] += ins_cost_k(k);
				for (q = 0; q < E->nop; q++) {
					int e;
					if (!mem_is_label(E->opnd[q]))
						continue;
					e = MC.lab_blk[idx(E->opnd[q])];
					if (e == mb || e == cb || e == MC.lhdr[inl] || seen[e] ||
					    !MC.lin[(size_t)inl *
					    (size_t)(MC.nblk + 1) + (size_t)e])
						continue;
					seen[e] = 1;
					st[sp++] = e;
				}
			}
			free(seen);
			free(st);
		}
		i = cost[0] >= cost[1] ? 0 : 1;
		dear = cost[i];
		cheap = cost[1 - i];
		/* where it wins, on a grid of 1/1024: an interval of p (a warp
		 * rarely split pays little as written; nearly every lane on the
		 * dear arm leaves nothing to pack) */
		plo = phi = -1;
		pbest = 0;
		gbest = 0;
		for (q = 1; q < 1024; q++) {
			double g;
			p = q / 1024.0;
			g = sg_written(m, W, p, dear, cheap) - (cheap + over + p * dear);
			if (g > 0) {
				if (plo < 0)
					plo = p;
				phi = p;
				if (g > gbest) {
					gbest = g;
					pbest = p;
				}
			}
		}
		if (plo < 0)
			continue;
		src_line(ins_line(MC.term[b]), sl, sizeof sl);
		sb_printf(rep, "    %s\n%s      in a loop's body, on a condition that "
		    "differs from lane to lane; the %s arm costs %g, the other %g\n"
		    "      compacted across the loop's iterations (a persistent "
		    "loop): each lane a ballot, an and with the lanes below and "
		    "its bit count, a shuffle there and back (%g), the dear arm "
		    "once for %d packed items;\n      cheaper than the branch (or "
		    "both arms and a select, the cheaper) while %.4g to %.4g of the "
		    "lanes take that arm (independent lanes): at %.4g, %g against "
		    "%g a lane\n", T2->text, sl, i == 0 ? "true" :
		    "false", dear, cheap, over, W, plo, phi, pbest, cheap + over +
		    pbest * dear, sg_written(m, W, pbest, dear, cheap));
		n++;
	}
	mem_cfg_free();
	free(mem_uni);
	mem_uni = NULL;
	return n;
}
